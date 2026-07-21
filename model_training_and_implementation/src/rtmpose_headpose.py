from types import SimpleNamespace
from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .kwan_headpose import module_init


@dataclass
class HeadPoseResult:
    yaw: float
    pitch: float
    roll: float
    box_xyxy: Optional[list]
    crop_box_xyxy: Optional[list]
    person_index: int
    source: str
    valid_crop: bool


class RTMPoseHeadPoseEstimator:
    """
    Kwan-style head-pose estimator using RTMPose keypoints instead of MTCNN.

    Expected RTMPose detection format:

        [
            {
                "box_xyxy": [x1, y1, x2, y2],
                "score": float,
                "keypoints": [[x, y, conf], ...],
                "keypoints_xy": [[x, y], ...],
            },
            ...
        ]

    For RTMPose wholebody / COCO-WholeBody layout:
        body keypoints: 0:17
        foot keypoints: 17:23
        face keypoints: 23:91
        hand keypoints: 91:133

    This class uses face keypoints 23:91 when available.
    If those are unavailable, it falls back to body head/shoulder keypoints.
    """

    def __init__(
        self,
        weights_path: str = "kwan_pretrained_weights/head-pose-pretrained.pkl",
        gpu_id: int = 0,
        face_start: int = 23,
        face_end: int = 91,
        face_conf_thresh: float = 0.20,
        body_conf_thresh: float = 0.20,
        face_crop_scale: float = 2.0,
        body_crop_scale: float = 2.2,
        min_face_points: int = 5,
        min_body_points: int = 3,
        min_crop_size: int = 35,
        rtmpose_detector: Optional[Any] = None,
    ):
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu"
        )

        cfg = SimpleNamespace(
            HEAD_POSE=SimpleNamespace(
                PRETRAINED=weights_path,
                GPU_ID=gpu_id,
            )
        )

        self.model = module_init(cfg)
        self.model.to(self.device)
        self.model.eval()

        self.rtmpose_detector = rtmpose_detector

        self.face_start = face_start
        self.face_end = face_end
        self.face_conf_thresh = face_conf_thresh
        self.body_conf_thresh = body_conf_thresh
        self.face_crop_scale = face_crop_scale
        self.body_crop_scale = body_crop_scale
        self.min_face_points = min_face_points
        self.min_body_points = min_body_points
        self.min_crop_size = min_crop_size

        self.transformations = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.idx_tensor = torch.arange(66, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image_bgr: np.ndarray,
        rtmpose_results: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[HeadPoseResult]:
        """
        Predict head pose for each RTMPose-detected person.

        If rtmpose_results is None, this will call self.rtmpose_detector.predict(image_bgr).
        """

        if rtmpose_results is None:
            if self.rtmpose_detector is None:
                raise ValueError(
                    "rtmpose_results was not provided and no rtmpose_detector was supplied."
                )
            rtmpose_results = self.rtmpose_detector.predict(image_bgr)

        results: List[HeadPoseResult] = []

        for person_index, det in enumerate(rtmpose_results):
            keypoints = self._extract_keypoints(det)
            if keypoints is None:
                results.append(self._sentinel_result(person_index, source="no_keypoints"))
                continue

            crop_box, source = self._crop_box_from_rtmpose_keypoints(
                keypoints=keypoints,
                image_shape=image_bgr.shape,
                fallback_box_xyxy=det.get("box_xyxy"),
            )

            if crop_box is None:
                results.append(self._sentinel_result(person_index, source=source))
                continue

            hp = self._predict_from_crop_box(image_bgr, crop_box)

            results.append(
                HeadPoseResult(
                    yaw=float(hp[0]),
                    pitch=float(hp[1]),
                    roll=float(hp[2]),
                    box_xyxy=det.get("box_xyxy"),
                    crop_box_xyxy=[float(v) for v in crop_box],
                    person_index=person_index,
                    source=source,
                    valid_crop=True,
                )
            )

        return results

    def predict_first_or_sentinel(
        self,
        image_bgr: np.ndarray,
        rtmpose_results: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Drop-in-style helper returning [yaw, pitch, roll].

        Chooses the best valid RTMPose-derived crop.
        If none are valid, returns [-999, -999, -999].
        """

        results = self.predict(image_bgr, rtmpose_results)

        valid = [r for r in results if r.valid_crop and r.crop_box_xyxy is not None]
        if not valid:
            return self._sentinel_array()

        # Prefer largest crop, which is usually the closest / main person.
        best = max(valid, key=lambda r: self._box_area(r.crop_box_xyxy))

        return np.asarray([best.yaw, best.pitch, best.roll], dtype=np.float32)

    def draw_debug(
        self,
        image_bgr: np.ndarray,
        results: Sequence[HeadPoseResult],
    ) -> np.ndarray:
        """
        Draw RTMPose-derived head crop boxes and predicted yaw/pitch/roll.
        Useful for verifying that the crop is actually reasonable.
        """

        out = image_bgr.copy()

        for r in results:
            if r.crop_box_xyxy is None:
                continue

            x1, y1, x2, y2 = map(int, r.crop_box_xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{r.source}: yaw={r.yaw:.1f}, pitch={r.pitch:.1f}, roll={r.roll:.1f}"
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_keypoints(self, det: Dict[str, Any]) -> Optional[np.ndarray]:
        if "keypoints" not in det or det["keypoints"] is None:
            return None

        kpts = np.asarray(det["keypoints"], dtype=np.float32)

        if kpts.ndim != 2:
            return None

        # If keypoints are [N, 2], synthesize confidence=1.
        if kpts.shape[1] == 2:
            conf = np.ones((kpts.shape[0], 1), dtype=np.float32)
            kpts = np.concatenate([kpts, conf], axis=1)

        if kpts.shape[1] < 3:
            return None

        return kpts[:, :3]

    def _crop_box_from_rtmpose_keypoints(
        self,
        keypoints: np.ndarray,
        image_shape,
        fallback_box_xyxy: Optional[Sequence[float]] = None,
    ):
        """
        Returns:
            crop_box_xyxy, source_string
        """

        # Preferred: COCO-WholeBody face landmarks.
        if keypoints.shape[0] >= self.face_end:
            face_kpts = keypoints[self.face_start:self.face_end]

            crop = self._box_from_keypoints(
                face_kpts,
                image_shape=image_shape,
                conf_thresh=self.face_conf_thresh,
                min_points=self.min_face_points,
                scale=self.face_crop_scale,
                y_bias=-0.10,
                min_crop_size=self.min_crop_size,
                force_square=True,
            )

            if crop is not None:
                return crop, "rtmpose_face"

        # Fallback: body head points.
        #
        # COCO body indices:
        #   0 nose
        #   1 left eye
        #   2 right eye
        #   3 left ear
        #   4 right ear
        #   5 left shoulder
        #   6 right shoulder
        body_head_indices = [0, 1, 2, 3, 4, 5, 6]
        valid_indices = [i for i in body_head_indices if i < keypoints.shape[0]]

        if valid_indices:
            body_kpts = keypoints[valid_indices]

            crop = self._box_from_keypoints(
                body_kpts,
                image_shape=image_shape,
                conf_thresh=self.body_conf_thresh,
                min_points=self.min_body_points,
                scale=self.body_crop_scale,
                y_bias=-0.20,
                min_crop_size=self.min_crop_size,
                force_square=True,
            )

            if crop is not None:
                return crop, "rtmpose_body_head"

        # Last-resort fallback: use upper portion of person bounding box.
        if fallback_box_xyxy is not None:
            crop = self._head_crop_from_person_box(
                fallback_box_xyxy,
                image_shape=image_shape,
            )
            if crop is not None:
                return crop, "person_box_upper_fallback"

        return None, "no_valid_rtmpose_crop"

    def _box_from_keypoints(
        self,
        keypoints: np.ndarray,
        image_shape,
        conf_thresh: float,
        min_points: int,
        scale: float,
        y_bias: float,
        min_crop_size: int,
        force_square: bool = True,
    ) -> Optional[list]:
        H, W = image_shape[:2]

        kpts = np.asarray(keypoints, dtype=np.float32)
        valid = kpts[:, 2] >= conf_thresh

        if int(valid.sum()) < min_points:
            return None

        pts = kpts[valid, :2]

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)

        bw = float(x2 - x1)
        bh = float(y2 - y1)

        if bw <= 1 or bh <= 1:
            return None

        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        if force_square:
            size = max(bw, bh) * scale
            size = max(size, float(min_crop_size))

            # Negative y_bias shifts crop upward.
            cy = cy + y_bias * size

            nx1 = cx - size / 2.0
            ny1 = cy - size / 2.0
            nx2 = cx + size / 2.0
            ny2 = cy + size / 2.0
        else:
            new_w = max(bw * scale, float(min_crop_size))
            new_h = max(bh * scale, float(min_crop_size))
            cy = cy + y_bias * new_h

            nx1 = cx - new_w / 2.0
            ny1 = cy - new_h / 2.0
            nx2 = cx + new_w / 2.0
            ny2 = cy + new_h / 2.0

        nx1, ny1, nx2, ny2 = self._clip_box([nx1, ny1, nx2, ny2], W, H)

        if nx2 - nx1 < min_crop_size or ny2 - ny1 < min_crop_size:
            return None

        return [int(nx1), int(ny1), int(nx2), int(ny2)]

    def _head_crop_from_person_box(
        self,
        box_xyxy: Sequence[float],
        image_shape,
    ) -> Optional[list]:
        """
        Last-resort fallback when RTMPose gives a person box but no reliable face/head points.
        Uses the upper portion of the person box.
        """

        H, W = image_shape[:2]

        x1, y1, x2, y2 = map(float, box_xyxy)
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 1 or bh <= 1:
            return None

        # Approximate head region as upper 30% of person box.
        head_y1 = y1
        head_y2 = y1 + 0.35 * bh

        cx = (x1 + x2) / 2.0
        cy = (head_y1 + head_y2) / 2.0

        size = max(bw * 0.75, (head_y2 - head_y1) * 1.4, self.min_crop_size)

        nx1 = cx - size / 2.0
        ny1 = cy - size / 2.0
        nx2 = cx + size / 2.0
        ny2 = cy + size / 2.0

        nx1, ny1, nx2, ny2 = self._clip_box([nx1, ny1, nx2, ny2], W, H)

        if nx2 - nx1 < self.min_crop_size or ny2 - ny1 < self.min_crop_size:
            return None

        return [int(nx1), int(ny1), int(nx2), int(ny2)]

    def _predict_from_crop_box(
        self,
        image_bgr: np.ndarray,
        crop_box_xyxy: Sequence[int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = map(int, crop_box_xyxy)

        crop_bgr = image_bgr[y1:y2, x1:x2]

        if crop_bgr.size == 0:
            return self._sentinel_array()

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        img = self.transformations(crop_pil)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            yaw_logits, pitch_logits, roll_logits = self.model(img)

            yaw = self._logits_to_angle(yaw_logits)
            pitch = self._logits_to_angle(pitch_logits)
            roll = self._logits_to_angle(roll_logits)

        return np.asarray([yaw, pitch, roll], dtype=np.float32)

    def _logits_to_angle(self, logits: torch.Tensor) -> float:
        probs = self.softmax(logits)
        angle = torch.sum(probs * self.idx_tensor, dim=1) * 3.0 - 99.0
        return float(angle.item())

    def _sentinel_result(self, person_index: int, source: str) -> HeadPoseResult:
        return HeadPoseResult(
            yaw=-999.0,
            pitch=-999.0,
            roll=-999.0,
            box_xyxy=None,
            crop_box_xyxy=None,
            person_index=person_index,
            source=source,
            valid_crop=False,
        )

    def _sentinel_array(self) -> np.ndarray:
        return np.asarray([-999.0, -999.0, -999.0], dtype=np.float32)

    @staticmethod
    def _clip_box(box_xyxy, W: int, H: int):
        x1, y1, x2, y2 = box_xyxy

        x1 = max(0.0, min(float(W - 1), float(x1)))
        y1 = max(0.0, min(float(H - 1), float(y1)))
        x2 = max(0.0, min(float(W), float(x2)))
        y2 = max(0.0, min(float(H), float(y2)))

        if x2 <= x1 or y2 <= y1:
            return 0, 0, 0, 0

        return x1, y1, x2, y2

    @staticmethod
    def _box_area(box_xyxy: Sequence[float]) -> float:
        x1, y1, x2, y2 = map(float, box_xyxy)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Sequence, Tuple

# import cv2
# import numpy as np


# @dataclass
# class HeadPoseResult:
#     yaw: float
#     pitch: float
#     roll: float
#     box_xyxy: Optional[list]
#     crop_box_xyxy: Optional[list]
#     person_index: int
#     source: str
#     valid_crop: bool


# class RTMPoseHeadPoseEstimator:
#     """
#     Keypoint-only RTMPose head-pose estimator.

#     This does NOT use:
#       - MTCNN
#       - Kwan's head-pose CNN
#       - head-pose-pretrained.pkl
#       - image crops for prediction

#     It estimates head pose from RTMPose keypoints.

#     Preferred path:
#       RTMPose wholebody keypoints with dense face landmarks:
#           keypoints[23:91]
#       -> 2D facial landmarks
#       -> generic 3D face model
#       -> cv2.solvePnP
#       -> yaw, pitch, roll

#     Fallback path:
#       COCO 17 body keypoints:
#           nose, eyes, ears, shoulders
#       -> rough heuristic yaw/pitch/roll

#     Notes:
#       - The solvePnP estimate is much better than the 17-keypoint fallback.
#       - If your RTMPose model only outputs 17 body keypoints, head pose will be approximate.
#       - If your RTMPose model outputs 133 wholebody keypoints, this should use the face landmarks.
#     """

#     def __init__(
#         self,
#         # Kept for drop-in compatibility with your previous constructor.
#         # This keypoint-only version ignores weights_path and gpu_id.
#         weights_path: Optional[str] = None,
#         gpu_id: Optional[int] = None,

#         # COCO-WholeBody face landmark range.
#         face_start: int = 23,
#         face_end: int = 91,

#         face_conf_thresh: float = 0.15,
#         body_conf_thresh: float = 0.15,

#         # Drawing/debug box settings.
#         face_box_scale: float = 1.6,
#         body_box_scale: float = 1.8,
#         min_box_size: int = 35,

#         # If None, uses focal_length = image_width * focal_scale.
#         focal_length: Optional[float] = None,
#         focal_scale: float = 1.0,

#         # Keep output in the rough range Kwan's classifier likely saw.
#         # Set to None if you want unclipped geometric angles.
#         clip_degrees: Optional[float] = 99.0,

#         # Useful if you find the visualized cube is mirrored.
#         yaw_sign: float = 1.0,
#         pitch_sign: float = 1.0,
#         roll_sign: float = 1.0,

#         allow_body_fallback: bool = True,
#         rtmpose_detector: Optional[Any] = None,
#     ):
#         self.face_start = face_start
#         self.face_end = face_end
#         self.face_conf_thresh = face_conf_thresh
#         self.body_conf_thresh = body_conf_thresh

#         self.face_box_scale = face_box_scale
#         self.body_box_scale = body_box_scale
#         self.min_box_size = min_box_size

#         self.focal_length = focal_length
#         self.focal_scale = focal_scale
#         self.clip_degrees = clip_degrees

#         self.yaw_sign = yaw_sign
#         self.pitch_sign = pitch_sign
#         self.roll_sign = roll_sign

#         self.allow_body_fallback = allow_body_fallback
#         self.rtmpose_detector = rtmpose_detector

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------

#     def predict(
#         self,
#         image_bgr: np.ndarray,
#         rtmpose_results: Optional[Sequence[Any]] = None,
#     ) -> List[HeadPoseResult]:
#         """
#         Returns one HeadPoseResult per detected person.

#         image_bgr is used only for image dimensions / camera matrix.
#         The pixel contents are not used for pose prediction.
#         """

#         if rtmpose_results is None:
#             if self.rtmpose_detector is None:
#                 raise ValueError(
#                     "rtmpose_results was not provided and no rtmpose_detector was supplied."
#                 )
#             rtmpose_results = self.rtmpose_detector.predict(image_bgr)

#         results: List[HeadPoseResult] = []

#         for person_index, det in enumerate(rtmpose_results):
#             keypoints = self._extract_keypoints(det)
#             person_box = self._extract_box_xyxy(det)

#             if keypoints is None:
#                 results.append(
#                     self._sentinel_result(
#                         person_index=person_index,
#                         box_xyxy=person_box,
#                         source="no_keypoints",
#                     )
#                 )
#                 continue

#             pose = self._estimate_from_keypoints(
#                 keypoints=keypoints,
#                 image_shape=image_bgr.shape,
#             )

#             if pose is None:
#                 results.append(
#                     self._sentinel_result(
#                         person_index=person_index,
#                         box_xyxy=person_box,
#                         source="no_valid_head_pose",
#                     )
#                 )
#                 continue

#             yaw, pitch, roll, head_box, source = pose

#             results.append(
#                 HeadPoseResult(
#                     yaw=float(yaw),
#                     pitch=float(pitch),
#                     roll=float(roll),
#                     box_xyxy=self._list_or_none(person_box),
#                     crop_box_xyxy=self._list_or_none(head_box),
#                     person_index=person_index,
#                     source=source,
#                     valid_crop=True,
#                 )
#             )

#         return results

#     def predict_first_or_sentinel(
#         self,
#         image_bgr: np.ndarray,
#         rtmpose_results: Optional[Sequence[Any]] = None,
#     ) -> np.ndarray:
#         """
#         Returns [yaw, pitch, roll] for the best valid detected person.
#         If no valid head pose exists, returns [-999, -999, -999].
#         """

#         results = self.predict(image_bgr, rtmpose_results=rtmpose_results)
#         valid = [r for r in results if r.valid_crop]

#         if not valid:
#             return self._sentinel_array()

#         best = max(valid, key=self._result_score)
#         return np.asarray([best.yaw, best.pitch, best.roll], dtype=np.float32)

#     def draw_debug(
#         self,
#         image_bgr: np.ndarray,
#         results: Sequence[HeadPoseResult],
#     ) -> np.ndarray:
#         """
#         Draws the estimated head box and yaw/pitch/roll text.
#         """

#         out = image_bgr.copy()

#         for r in results:
#             if r.crop_box_xyxy is None:
#                 continue

#             x1, y1, x2, y2 = map(int, r.crop_box_xyxy)

#             cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)

#             label = (
#                 f"{r.source}: "
#                 f"yaw={r.yaw:.1f}, pitch={r.pitch:.1f}, roll={r.roll:.1f}"
#             )

#             cv2.putText(
#                 out,
#                 label,
#                 (x1, max(0, y1 - 8)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.45,
#                 (255, 255, 0),
#                 1,
#                 cv2.LINE_AA,
#             )

#         return out

#     # ------------------------------------------------------------------
#     # Main estimation logic
#     # ------------------------------------------------------------------

#     def _estimate_from_keypoints(
#         self,
#         keypoints: np.ndarray,
#         image_shape: Tuple[int, int, int],
#     ) -> Optional[Tuple[float, float, float, Optional[list], str]]:
#         """
#         Returns:
#             yaw, pitch, roll, head_box_xyxy, source
#         """

#         # Preferred: dense face landmarks from COCO-WholeBody.
#         if keypoints.shape[0] >= self.face_end:
#             pnp_result = self._estimate_from_face_pnp(
#                 keypoints=keypoints,
#                 image_shape=image_shape,
#             )

#             if pnp_result is not None:
#                 return pnp_result

#         # Fallback: rough estimate from 17 body keypoints.
#         if self.allow_body_fallback:
#             body_result = self._estimate_from_body_heuristic(
#                 keypoints=keypoints,
#                 image_shape=image_shape,
#             )

#             if body_result is not None:
#                 return body_result

#         return None

#     def _estimate_from_face_pnp(
#         self,
#         keypoints: np.ndarray,
#         image_shape: Tuple[int, int, int],
#     ) -> Optional[Tuple[float, float, float, Optional[list], str]]:
#         """
#         Uses 68 dense face landmarks with a generic 3D face model.

#         Assumes COCO-WholeBody face landmarks are keypoints[23:91] and follow
#         the common 68-point facial landmark ordering.
#         """

#         H, W = image_shape[:2]

#         face = keypoints[self.face_start:self.face_end]
#         if face.shape[0] < 68:
#             return None

#         # Standard 68-landmark indices, local to the face array.
#         #
#         # 30: nose tip
#         # 8:  chin
#         # 36: left eye outer corner in common head-pose examples
#         # 45: right eye outer corner
#         # 48: left mouth corner
#         # 54: right mouth corner
#         landmark_indices = [30, 8, 36, 45, 48, 54]

#         image_points = []
#         for idx in landmark_indices:
#             x, y, conf = face[idx, :3]
#             if conf < self.face_conf_thresh:
#                 return None
#             image_points.append([float(x), float(y)])

#         image_points = np.asarray(image_points, dtype=np.float64)

#         # Generic 3D face model points in arbitrary face-model units.
#         # This is the common 6-point model used for monocular head pose.
#         model_points = np.asarray(
#             [
#                 [0.0, 0.0, 0.0],          # Nose tip
#                 [0.0, -330.0, -65.0],     # Chin
#                 [-225.0, 170.0, -135.0],  # Left eye corner
#                 [225.0, 170.0, -135.0],   # Right eye corner
#                 [-150.0, -150.0, -125.0], # Left mouth corner
#                 [150.0, -150.0, -125.0],  # Right mouth corner
#             ],
#             dtype=np.float64,
#         )

#         focal_length = (
#             float(self.focal_length)
#             if self.focal_length is not None
#             else float(W) * float(self.focal_scale)
#         )

#         camera_matrix = np.asarray(
#             [
#                 [focal_length, 0.0, W / 2.0],
#                 [0.0, focal_length, H / 2.0],
#                 [0.0, 0.0, 1.0],
#             ],
#             dtype=np.float64,
#         )

#         dist_coeffs = np.zeros((4, 1), dtype=np.float64)

#         success, rotation_vec, translation_vec = cv2.solvePnP(
#             model_points,
#             image_points,
#             camera_matrix,
#             dist_coeffs,
#             flags=cv2.SOLVEPNP_ITERATIVE,
#         )

#         if not success:
#             return None

#         yaw, pitch, roll = self._rotation_vector_to_yaw_pitch_roll(
#             rotation_vec,
#             translation_vec,
#         )

#         yaw, pitch, roll = self._postprocess_angles(yaw, pitch, roll)

#         head_box = self._box_from_keypoints(
#             face,
#             image_shape=image_shape,
#             conf_thresh=self.face_conf_thresh,
#             scale=self.face_box_scale,
#             y_bias=-0.05,
#         )

#         return yaw, pitch, roll, head_box, "rtmpose_face_pnp"

#     def _estimate_from_body_heuristic(
#         self,
#         keypoints: np.ndarray,
#         image_shape: Tuple[int, int, int],
#     ) -> Optional[Tuple[float, float, float, Optional[list], str]]:
#         """
#         Rough fallback using COCO body keypoints.

#         COCO body indices:
#           0 nose
#           1 left eye
#           2 right eye
#           3 left ear
#           4 right ear
#           5 left shoulder
#           6 right shoulder

#         This is much less accurate than solvePnP with dense face landmarks.
#         """

#         if keypoints.shape[0] < 7:
#             return None

#         nose = keypoints[0]
#         left_eye = keypoints[1]
#         right_eye = keypoints[2]
#         left_ear = keypoints[3]
#         right_ear = keypoints[4]
#         left_shoulder = keypoints[5]
#         right_shoulder = keypoints[6]

#         if nose[2] < self.body_conf_thresh:
#             return None

#         have_eyes = (
#             left_eye[2] >= self.body_conf_thresh
#             and right_eye[2] >= self.body_conf_thresh
#         )

#         have_ears = (
#             left_ear[2] >= self.body_conf_thresh
#             and right_ear[2] >= self.body_conf_thresh
#         )

#         have_shoulders = (
#             left_shoulder[2] >= self.body_conf_thresh
#             and right_shoulder[2] >= self.body_conf_thresh
#         )

#         if not have_eyes and not have_ears:
#             return None

#         if have_eyes:
#             left_ref = left_eye[:2]
#             right_ref = right_eye[:2]
#             ref_mid = (left_ref + right_ref) / 2.0
#             ref_dist = float(np.linalg.norm(left_ref - right_ref))
#         else:
#             left_ref = left_ear[:2]
#             right_ref = right_ear[:2]
#             ref_mid = (left_ref + right_ref) / 2.0
#             ref_dist = float(np.linalg.norm(left_ref - right_ref))

#         if ref_dist < 1.0:
#             return None

#         # Roll: in-plane tilt of the eye/ear line.
#         roll = np.degrees(
#             np.arctan2(
#                 float(right_ref[1] - left_ref[1]),
#                 float(right_ref[0] - left_ref[0]),
#             )
#         )

#         # Yaw: nose horizontal offset from eye/ear midpoint.
#         # This is only a rough proxy.
#         yaw = 70.0 * float((nose[0] - ref_mid[0]) / ref_dist)

#         # Pitch: nose vertical position relative to eye/ear line.
#         # The 0.55 constant is a rough frontal-face prior.
#         vertical_ratio = float((nose[1] - ref_mid[1]) / ref_dist)
#         pitch = 60.0 * (vertical_ratio - 0.55)

#         # Shoulder relation can stabilize yaw slightly when eyes/ears are noisy.
#         if have_shoulders:
#             shoulder_mid = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
#             shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
#             if shoulder_width > 1.0:
#                 torso_yaw_proxy = 50.0 * float((nose[0] - shoulder_mid[0]) / shoulder_width)
#                 yaw = 0.75 * yaw + 0.25 * torso_yaw_proxy

#         yaw, pitch, roll = self._postprocess_angles(yaw, pitch, roll)

#         head_indices = [0, 1, 2, 3, 4]
#         if have_shoulders:
#             head_indices += [5, 6]

#         head_kpts = keypoints[[i for i in head_indices if i < keypoints.shape[0]]]

#         head_box = self._box_from_keypoints(
#             head_kpts,
#             image_shape=image_shape,
#             conf_thresh=self.body_conf_thresh,
#             scale=self.body_box_scale,
#             y_bias=-0.20,
#         )

#         return yaw, pitch, roll, head_box, "rtmpose_body_heuristic"

#     # ------------------------------------------------------------------
#     # Geometry helpers
#     # ------------------------------------------------------------------

#     def _rotation_vector_to_yaw_pitch_roll(
#         self,
#         rotation_vec: np.ndarray,
#         translation_vec: np.ndarray,
#     ) -> Tuple[float, float, float]:
#         """
#         Converts OpenCV solvePnP rotation vector to yaw/pitch/roll in degrees.

#         OpenCV decomposeProjectionMatrix returns Euler angles roughly as:
#           x rotation -> pitch
#           y rotation -> yaw
#           z rotation -> roll
#         """

#         rotation_mat, _ = cv2.Rodrigues(rotation_vec)

#         projection_mat = np.hstack((rotation_mat, translation_vec.reshape(3, 1)))

#         _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
#             projection_mat
#         )

#         pitch = float(euler_angles[0])
#         yaw = float(euler_angles[1])
#         roll = float(euler_angles[2])

#         return yaw, pitch, roll

#     def _postprocess_angles(
#         self,
#         yaw: float,
#         pitch: float,
#         roll: float,
#     ) -> Tuple[float, float, float]:
#         yaw = self._normalize_angle(yaw) * self.yaw_sign
#         pitch = self._normalize_angle(pitch) * self.pitch_sign
#         roll = self._normalize_angle(roll) * self.roll_sign

#         if self.clip_degrees is not None:
#             c = float(self.clip_degrees)
#             yaw = float(np.clip(yaw, -c, c))
#             pitch = float(np.clip(pitch, -c, c))
#             roll = float(np.clip(roll, -c, c))

#         return yaw, pitch, roll

#     @staticmethod
#     def _normalize_angle(angle: float) -> float:
#         """
#         Normalize angle to [-180, 180].
#         """

#         angle = float(angle)
#         while angle > 180.0:
#             angle -= 360.0
#         while angle < -180.0:
#             angle += 360.0
#         return angle

#     def _box_from_keypoints(
#         self,
#         keypoints: np.ndarray,
#         image_shape: Tuple[int, int, int],
#         conf_thresh: float,
#         scale: float,
#         y_bias: float,
#     ) -> Optional[list]:
#         H, W = image_shape[:2]

#         kpts = np.asarray(keypoints, dtype=np.float32)
#         if kpts.ndim != 2 or kpts.shape[1] < 3:
#             return None

#         valid = kpts[:, 2] >= conf_thresh
#         if int(valid.sum()) < 2:
#             return None

#         pts = kpts[valid, :2]

#         x1, y1 = pts.min(axis=0)
#         x2, y2 = pts.max(axis=0)

#         bw = float(x2 - x1)
#         bh = float(y2 - y1)

#         if bw <= 1.0 or bh <= 1.0:
#             return None

#         cx = float((x1 + x2) / 2.0)
#         cy = float((y1 + y2) / 2.0)

#         size = max(bw, bh) * float(scale)
#         size = max(size, float(self.min_box_size))

#         cy = cy + float(y_bias) * size

#         nx1 = cx - size / 2.0
#         ny1 = cy - size / 2.0
#         nx2 = cx + size / 2.0
#         ny2 = cy + size / 2.0

#         nx1 = max(0.0, min(float(W - 1), nx1))
#         ny1 = max(0.0, min(float(H - 1), ny1))
#         nx2 = max(0.0, min(float(W), nx2))
#         ny2 = max(0.0, min(float(H), ny2))

#         if nx2 <= nx1 or ny2 <= ny1:
#             return None

#         return [int(nx1), int(ny1), int(nx2), int(ny2)]

#     # ------------------------------------------------------------------
#     # Input/output helpers
#     # ------------------------------------------------------------------

#     def _extract_keypoints(self, det: Any) -> Optional[np.ndarray]:
#         keypoints = self._get_attr_or_key(det, "keypoints", None)

#         if keypoints is None:
#             keypoints = self._get_attr_or_key(det, "keypoints_xy", None)

#         if keypoints is None:
#             keypoints = self._get_attr_or_key(det, "pred_keypoints", None)

#         if keypoints is None:
#             return None

#         keypoints = np.asarray(keypoints, dtype=np.float32)

#         if keypoints.ndim != 2:
#             return None

#         if keypoints.shape[1] == 2:
#             conf = np.ones((keypoints.shape[0], 1), dtype=np.float32)
#             keypoints = np.concatenate([keypoints, conf], axis=1)

#         if keypoints.shape[1] < 3:
#             return None

#         return keypoints[:, :3]

#     def _extract_box_xyxy(self, det: Any) -> Optional[np.ndarray]:
#         box = self._get_attr_or_key(det, "box_xyxy", None)

#         if box is None:
#             box = self._get_attr_or_key(det, "bbox", None)

#         if box is None:
#             box = self._get_attr_or_key(det, "box", None)

#         if box is None:
#             return None

#         box = np.asarray(box, dtype=np.float32).flatten()

#         if box.shape[0] != 4:
#             return None

#         return box

#     @staticmethod
#     def _get_attr_or_key(x: Any, name: str, default=None):
#         if isinstance(x, dict):
#             return x.get(name, default)
#         return getattr(x, name, default)

#     @staticmethod
#     def _list_or_none(x):
#         if x is None:
#             return None
#         return [float(v) for v in np.asarray(x).flatten().tolist()]

#     def _sentinel_result(
#         self,
#         person_index: int,
#         box_xyxy: Optional[np.ndarray],
#         source: str,
#     ) -> HeadPoseResult:
#         return HeadPoseResult(
#             yaw=-999.0,
#             pitch=-999.0,
#             roll=-999.0,
#             box_xyxy=self._list_or_none(box_xyxy),
#             crop_box_xyxy=None,
#             person_index=person_index,
#             source=source,
#             valid_crop=False,
#         )

#     @staticmethod
#     def _sentinel_array() -> np.ndarray:
#         return np.asarray([-999.0, -999.0, -999.0], dtype=np.float32)

#     @staticmethod
#     def _box_area(box_xyxy: Optional[Sequence[float]]) -> float:
#         if box_xyxy is None:
#             return 0.0
#         x1, y1, x2, y2 = map(float, box_xyxy)
#         return max(0.0, x2 - x1) * max(0.0, y2 - y1)

#     def _result_score(self, r: HeadPoseResult) -> float:
#         # Prefer solvePnP face estimates over rough body heuristic.
#         if r.source == "rtmpose_face_pnp":
#             source_score = 10_000_000.0
#         elif r.source == "rtmpose_body_heuristic":
#             source_score = 1_000_000.0
#         else:
#             source_score = 0.0

#         return source_score + self._box_area(r.crop_box_xyxy)