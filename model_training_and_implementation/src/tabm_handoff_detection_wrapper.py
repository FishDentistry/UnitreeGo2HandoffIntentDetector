from pathlib import Path

import cv2
import numpy as np
import torch

from .rtmpose_keypoints import RTMPoseKeypointDetector
from .rtmpose_headpose import RTMPoseHeadPoseEstimator
from .hand_intent_tabm import HandIntentTabM
from .resnet_encoder import ResNet18ImageEncoder


# This file lives in:
# model_training_and_implementation/src/
MODEL_IMPL_ROOT = Path(__file__).resolve().parents[1]

MODEL_WEIGHTS_ROOT = (
    MODEL_IMPL_ROOT
    / "outputs"
    / "hand_intent_tabm"
)

HEAD_POSE_WEIGHTS = (
    MODEL_IMPL_ROOT
    / "kwan_pretrained_weights"
    / "head-pose-pretrained.pkl"
)


def get_depth_at_rgb_point(
    depth_image,
    rgb_image_shape,
    point_xy,
    patch_radius=1,
):
    """
    Return the median nonzero depth near an RGB-image point.
    """
    rgb_height, rgb_width = rgb_image_shape[:2]
    depth_height, depth_width = depth_image.shape[:2]

    x_rgb, y_rgb = map(float, point_xy)

    x_depth = int(round(x_rgb * depth_width / rgb_width))
    y_depth = int(round(y_rgb * depth_height / rgb_height))

    x_depth = int(np.clip(x_depth, 0, depth_width - 1))
    y_depth = int(np.clip(y_depth, 0, depth_height - 1))

    x_start = max(0, x_depth - patch_radius)
    x_end = min(depth_width, x_depth + patch_radius + 1)

    y_start = max(0, y_depth - patch_radius)
    y_end = min(depth_height, y_depth + patch_radius + 1)

    patch = np.asarray(
        depth_image[y_start:y_end, x_start:x_end],
        dtype=np.float32,
    )

    valid_depths = patch[
        np.isfinite(patch) & (patch > 0)
    ]

    if valid_depths.size == 0:
        raise ValueError(
            f"No valid depth near RGB point "
            f"({x_rgb:.1f}, {y_rgb:.1f})."
        )

    return float(np.median(valid_depths))


class HandoffDetector:
    def __init__(
        self,
        features_type="keypoints",
        model_path=None,
        threshold=0.5,
        confidence=0.15,
        normalize_keypoints=True,
        depth_patch_radius=1,
        crop_around_object=False,
        crop_object="person.",
        dino_classes="cup",
        head_pose_weights=None,
        device="auto",
        reject_back_facing=True,
        back_facing_min_torso_ratio=0.15,
        reject_side_without_forward_wrist=True,
        side_facing_max_torso_ratio=0.45,
        side_shoulder_depth_fraction=0.06,
        side_wrist_forward_depth_fraction=0.10,
        side_wrist_beyond_elbow_depth_fraction=0.04,
        side_max_probability_penalty=0.45,
        side_correction_min_raw_probability=0.50,
        side_probability_cap_enabled=True,
        side_probability_cap_min_score_2d=0.70,
        side_probability_cap_base=0.45,
        side_probability_cap_presentation_gain=0.35,
    ):
        self.features_type = features_type
        self.threshold = float(threshold)
        self.normalize_keypoints = bool(normalize_keypoints)
        self.depth_patch_radius = int(depth_patch_radius)
        self.crop_around_object = bool(crop_around_object)
        self.crop_object = crop_object
        self.dino_classes = dino_classes
        self.keypoint_confidence = float(confidence)

        # Optional deployment-only gate for obvious back-facing views.
        # This does not alter the feature vector or the trained TabM model.
        self.reject_back_facing = bool(reject_back_facing)
        self.back_facing_min_torso_ratio = float(
            back_facing_min_torso_ratio
        )

        if self.back_facing_min_torso_ratio < 0.0:
            raise ValueError(
                "back_facing_min_torso_ratio must be >= 0.0."
            )

        # Optional deployment-only side-view correction. This used to be a
        # hard gate that rejected a frame whenever a side-facing person did
        # not exceed fixed wrist-depth thresholds. The legacy enable flag is
        # retained for compatibility, but now enables a SOFT probability
        # correction applied only after TabM produces its raw probability.
        self.reject_side_without_forward_wrist = bool(
            reject_side_without_forward_wrist
        )
        self.side_facing_max_torso_ratio = float(
            side_facing_max_torso_ratio
        )
        self.side_shoulder_depth_fraction = float(
            side_shoulder_depth_fraction
        )
        self.side_wrist_forward_depth_fraction = float(
            side_wrist_forward_depth_fraction
        )
        self.side_wrist_beyond_elbow_depth_fraction = float(
            side_wrist_beyond_elbow_depth_fraction
        )
        self.side_max_probability_penalty = float(
            side_max_probability_penalty
        )
        self.side_correction_min_raw_probability = float(
            side_correction_min_raw_probability
        )

        self.side_probability_cap_enabled = bool(
            side_probability_cap_enabled
        )
        self.side_probability_cap_min_score_2d = float(
            side_probability_cap_min_score_2d
        )
        self.side_probability_cap_base = float(
            side_probability_cap_base
        )
        self.side_probability_cap_presentation_gain = float(
            side_probability_cap_presentation_gain
        )

        if self.side_facing_max_torso_ratio <= 0.0:
            raise ValueError(
                "side_facing_max_torso_ratio must be > 0.0."
            )
        if self.side_shoulder_depth_fraction <= 0.0:
            raise ValueError(
                "side_shoulder_depth_fraction must be > 0.0."
            )
        if self.side_wrist_forward_depth_fraction <= 0.0:
            raise ValueError(
                "side_wrist_forward_depth_fraction must be > 0.0."
            )
        if self.side_wrist_beyond_elbow_depth_fraction <= 0.0:
            raise ValueError(
                "side_wrist_beyond_elbow_depth_fraction must be > 0.0."
            )
        if not 0.0 <= self.side_max_probability_penalty <= 1.0:
            raise ValueError(
                "side_max_probability_penalty must be in [0.0, 1.0]."
            )
        if not 0.0 <= self.side_correction_min_raw_probability <= 1.0:
            raise ValueError(
                "side_correction_min_raw_probability must be in [0.0, 1.0]."
            )

        if not 0.0 <= self.side_probability_cap_min_score_2d <= 1.0:
            raise ValueError(
                "side_probability_cap_min_score_2d must be in [0.0, 1.0]."
            )
        if not 0.0 <= self.side_probability_cap_base <= 1.0:
            raise ValueError(
                "side_probability_cap_base must be in [0.0, 1.0]."
            )
        if self.side_probability_cap_presentation_gain < 0.0:
            raise ValueError(
                "side_probability_cap_presentation_gain must be >= 0.0."
            )
        if (
            self.side_probability_cap_base
            + self.side_probability_cap_presentation_gain
            > 1.0
        ):
            raise ValueError(
                "side_probability_cap_base + "
                "side_probability_cap_presentation_gain must be <= 1.0."
            )

        # Diagnostics from the latest successful model prediction.
        self.last_raw_handoff_probability = None
        self.last_adjusted_handoff_probability = None
        self.last_side_score = 0.0
        self.last_presentation_score = 1.0
        self.last_side_probability_penalty = 0.0
        self._side_metadata_for_last_features = {
            "side_score_2d": 0.0,
            "side_score": 0.0,
            "presentation_score": 1.0,
            "valid": False,
        }

        # Keep RTMPose crop generation identical to the updated training
        # pipeline. The full-frame RTMPose pass uses the normal detector
        # confidence to decide which keypoints define the person crop.
        self.crop_keypoint_confidence = self.keypoint_confidence
        self.crop_padding_fraction = 0.20

        # RTMPose may occasionally return a weak false-positive pose even
        # when no person is actually present. Use a slightly stricter
        # confidence threshold only for deciding whether a person exists.
        # This does not change the feature values supplied to TabM.
        self.person_presence_confidence = max(
            self.keypoint_confidence,
            0.30,
        )

        if self.depth_patch_radius < 0:
            raise ValueError(
                "depth_patch_radius must be >= 0."
            )

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------
        if device == "auto":
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        # ---------------------------------------------------------
        # Resolve model path
        # ---------------------------------------------------------
        if model_path is None:
            variation_name = (
                f"features-{self.features_type}"
                f"__crop-{self.crop_around_object}"
            )

            model_path = (
                MODEL_WEIGHTS_ROOT
                / variation_name
                / "TabM.pth"
            )
        else:
            model_path = Path(model_path)

        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Handoff model not found: "
                f"{self.model_path}"
            )

        # ---------------------------------------------------------
        # Load TabM
        # ---------------------------------------------------------
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.input_dim = int(checkpoint["input_dim"])
        self.tabm_k = int(checkpoint.get("tabm_k", 32))

        self.model = HandIntentTabM(
            input_size=self.input_dim,
            output_size=1,
            k=self.tabm_k,
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # ---------------------------------------------------------
        # Keypoint detector
        # ---------------------------------------------------------
        self.keypoint_detector = (
            RTMPoseKeypointDetector(
                confidence=confidence,
                device=str(self.device),
                mode = "lightweight"
            )
        )

        # ---------------------------------------------------------
        # DINO compatibility
        # ---------------------------------------------------------
        # DINO is not loaded at deployment. Person cropping is performed
        # with RTMPose using the same algorithm as the updated training
        # pipeline. Keep the legacy constructor arguments only so existing
        # callers do not need to change.
        self.object_detector = None

        if self.features_type == "keypoints_headpose_dino":
            raise ValueError(
                'features_type="keypoints_headpose_dino" requires DINO '
                "object-centroid features and is not supported by the "
                "RTMPose-crop deployment wrapper."
            )

        # ---------------------------------------------------------
        # Head-pose estimator
        # ---------------------------------------------------------
        self.head_pose_estimator = None

        if "headpose" in self.features_type:
            if head_pose_weights is None:
                head_pose_weights = (
                    HEAD_POSE_WEIGHTS
                )

            self.head_pose_estimator = (
                RTMPoseHeadPoseEstimator(
                    weights_path=Path(
                        head_pose_weights
                    ),
                    gpu_id=0,
                )
            )

        # ---------------------------------------------------------
        # ResNet encoder
        # ---------------------------------------------------------
        self.img_encoder = None

        if "resnet" in self.features_type:
            self.img_encoder = (
                ResNet18ImageEncoder(
                    pretrained=True,
                    device=str(self.device),
                    l2_normalize=True,
                )
            )

    def _is_valid_person(
        self,
        person,
    ):
        """
        Return True only when RTMPose produced a sufficiently
        confident upper-body pose to treat the frame as containing
        a real person.

        The handoff model depends primarily on shoulders, elbows,
        and wrists, so use those joints for the presence check.
        """
        keypoints = np.asarray(
            person.get("keypoints", []),
            dtype=np.float32,
        )

        if (
            keypoints.ndim != 2
            or keypoints.shape[0] < 11
            or keypoints.shape[1] < 2
        ):
            return False

        upper_body = keypoints[5:11]

        if not np.isfinite(upper_body[:, :2]).all():
            return False

        # RTMPose results normally contain [x, y, confidence].
        # If confidence values are unavailable, preserve backwards
        # compatibility and accept the geometrically valid pose.
        if keypoints.shape[1] < 3:
            return True

        scores = upper_body[:, 2]

        if not np.isfinite(scores).all():
            return False

        threshold = self.person_presence_confidence

        # Both shoulders should be confidently localized.
        if scores[0] < threshold or scores[1] < threshold:
            return False

        # Require a coherent upper-body detection rather than one or
        # two isolated high-confidence hallucinated joints.
        confident_joint_count = int(
            np.count_nonzero(scores >= threshold)
        )

        if confident_joint_count < 4:
            return False

        if float(np.mean(scores)) < threshold:
            return False

        return True

    def _is_facing_away(
        self,
        person,
    ):
        """
        Return True only for a clear back-facing pose.

        RTMPose uses COCO anatomical left/right labels:
          5, 6   = left/right shoulder
          11, 12 = left/right hip

        In an ordinary, unmirrored image:
          front-facing: anatomical left is to image-right
          back-facing:  anatomical left is to image-left

        A frame is rejected only when BOTH the shoulder pair and hip pair
        confidently have the back-facing ordering. The horizontal
        separations are normalized by torso length so that the test is
        approximately scale-invariant. Near-side views have small lateral
        separation and therefore remain ambiguous rather than being
        rejected.
        """
        if not self.reject_back_facing:
            return False

        keypoints = np.asarray(
            person.get("keypoints", []),
            dtype=np.float32,
        )

        # Standard COCO pose output has 17 keypoints. We specifically need
        # shoulders (5, 6) and hips (11, 12) for a conservative back-view
        # decision.
        if (
            keypoints.ndim != 2
            or keypoints.shape[0] < 13
            or keypoints.shape[1] < 2
        ):
            return False

        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        torso_points = np.stack(
            [
                left_shoulder[:2],
                right_shoulder[:2],
                left_hip[:2],
                right_hip[:2],
            ]
        )

        if not np.isfinite(torso_points).all():
            return False

        # Do not make a front/back decision from weak torso keypoints.
        if keypoints.shape[1] >= 3:
            confidence_threshold = (
                self.person_presence_confidence
            )

            torso_scores = np.asarray(
                [
                    left_shoulder[2],
                    right_shoulder[2],
                    left_hip[2],
                    right_hip[2],
                ],
                dtype=np.float32,
            )

            if not np.isfinite(torso_scores).all():
                return False

            if np.any(
                torso_scores < confidence_threshold
            ):
                return False

        shoulder_midpoint = (
            left_shoulder[:2] + right_shoulder[:2]
        ) / 2.0
        hip_midpoint = (
            left_hip[:2] + right_hip[:2]
        ) / 2.0

        torso_length = float(
            np.linalg.norm(
                shoulder_midpoint - hip_midpoint
            )
        )

        if torso_length <= 1e-6:
            return False

        # Positive values correspond to the anatomical left/right ordering
        # expected when the person is facing away from the camera.
        shoulder_back_ratio = float(
            right_shoulder[0] - left_shoulder[0]
        ) / torso_length

        hip_back_ratio = float(
            right_hip[0] - left_hip[0]
        ) / torso_length

        min_ratio = self.back_facing_min_torso_ratio

        return (
            shoulder_back_ratio >= min_ratio
            and hip_back_ratio >= min_ratio
        )

    def _side_facing_score_2d(
        self,
        person,
    ):
        """Return a continuous 0..1 side-view score from 2D shoulders."""
        if not self.reject_side_without_forward_wrist:
            return 0.0

        keypoints = np.asarray(
            person.get("keypoints", []),
            dtype=np.float32,
        )
        if (
            keypoints.ndim != 2
            or keypoints.shape[0] < 7
            or keypoints.shape[1] < 2
        ):
            return 0.0

        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        shoulder_xy = np.stack(
            [left_shoulder[:2], right_shoulder[:2]]
        )
        if not np.isfinite(shoulder_xy).all():
            return 0.0

        if keypoints.shape[1] >= 3:
            confidence_threshold = self.person_presence_confidence
            shoulder_scores = np.asarray(
                [left_shoulder[2], right_shoulder[2]],
                dtype=np.float32,
            )
            if not np.isfinite(shoulder_scores).all():
                return 0.0
            if np.any(shoulder_scores < confidence_threshold):
                return 0.0

        upper_body_xy = keypoints[5:11, :2]
        valid = np.isfinite(upper_body_xy).all(axis=1)
        valid &= ~np.all(upper_body_xy == 0, axis=1)
        if keypoints.shape[1] >= 3:
            upper_scores = keypoints[5:11, 2]
            valid &= np.isfinite(upper_scores)
            valid &= upper_scores >= self.keypoint_confidence
        if int(np.count_nonzero(valid)) < 4:
            return 0.0

        valid_xy = upper_body_xy[valid]
        upper_body_height = float(
            np.max(valid_xy[:, 1]) - np.min(valid_xy[:, 1])
        )
        if upper_body_height <= 1e-6:
            return 0.0

        shoulder_width_ratio = (
            abs(float(left_shoulder[0] - right_shoulder[0]))
            / upper_body_height
        )
        # The former hard threshold is now the half-score point.
        score = 1.0 - (
            shoulder_width_ratio
            / (2.0 * self.side_facing_max_torso_ratio)
        )
        return float(np.clip(score, 0.0, 1.0))

    def _crop_images(
        self,
        image,
        depth_image,
    ):
        """
        Crop RGB and depth images using the same RTMPose-based person
        localization used by the updated training pipeline.

        This is the first RTMPose pass. It is used only to determine the
        crop. A second RTMPose pass is run later on the cropped image to
        generate the actual model features.
        """
        crop_object_name = (
            str(self.crop_object)
            .strip()
            .strip(".")
            .lower()
        )

        if crop_object_name != "person":
            raise ValueError(
                "RTMPose-based cropping only supports "
                "crop_object='person'. "
                f"Received: {self.crop_object!r}"
            )

        crop_people = self.keypoint_detector.predict(image)

        if len(crop_people) == 0:
            return None, None

        # Build one candidate crop box per detected person and choose the
        # largest, matching the updated training function exactly.
        crop_candidates = []

        for crop_person in crop_people:
            crop_keypoints = np.asarray(
                crop_person.get("keypoints", []),
                dtype=np.float32,
            )

            if (
                crop_keypoints.ndim != 2
                or crop_keypoints.shape[0] == 0
                or crop_keypoints.shape[1] < 2
            ):
                continue

            crop_xy = crop_keypoints[:, :2]

            valid = np.isfinite(crop_xy).all(axis=1)
            valid &= ~np.all(crop_xy == 0, axis=1)

            if crop_keypoints.shape[1] >= 3:
                crop_scores = crop_keypoints[:, 2]
                valid &= np.isfinite(crop_scores)
                valid &= (
                    crop_scores >= self.crop_keypoint_confidence
                )

            # Match training: require at least four usable joints to form
            # a crop candidate.
            if int(np.count_nonzero(valid)) < 4:
                continue

            valid_xy = crop_xy[valid]

            raw_x1 = float(np.min(valid_xy[:, 0]))
            raw_y1 = float(np.min(valid_xy[:, 1]))
            raw_x2 = float(np.max(valid_xy[:, 0]))
            raw_y2 = float(np.max(valid_xy[:, 1]))

            raw_width = raw_x2 - raw_x1
            raw_height = raw_y2 - raw_y1

            if raw_width <= 0.0 or raw_height <= 0.0:
                continue

            raw_area = raw_width * raw_height

            crop_candidates.append(
                (
                    raw_area,
                    raw_x1,
                    raw_y1,
                    raw_x2,
                    raw_y2,
                )
            )

        if not crop_candidates:
            return None, None

        (
            _,
            x1,
            y1,
            x2,
            y2,
        ) = max(
            crop_candidates,
            key=lambda candidate: candidate[0],
        )

        box_width = x2 - x1
        box_height = y2 - y1

        padding_x = (
            self.crop_padding_fraction * box_width
        )
        padding_y = (
            self.crop_padding_fraction * box_height
        )

        x1 = int(np.floor(x1 - padding_x))
        y1 = int(np.floor(y1 - padding_y))
        x2 = int(np.ceil(x2 + padding_x))
        y2 = int(np.ceil(y2 + padding_y))

        rgb_height, rgb_width = image.shape[:2]
        depth_height, depth_width = depth_image.shape[:2]

        x1 = int(np.clip(x1, 0, rgb_width))
        x2 = int(np.clip(x2, 0, rgb_width))
        y1 = int(np.clip(y1, 0, rgb_height))
        y2 = int(np.clip(y2, 0, rgb_height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "RTMPose produced an invalid RGB crop."
            )

        # Map RGB crop coordinates into the depth image exactly as in
        # training.
        depth_x1 = int(
            round(x1 * depth_width / rgb_width)
        )
        depth_x2 = int(
            round(x2 * depth_width / rgb_width)
        )
        depth_y1 = int(
            round(y1 * depth_height / rgb_height)
        )
        depth_y2 = int(
            round(y2 * depth_height / rgb_height)
        )

        depth_x1 = int(
            np.clip(depth_x1, 0, depth_width)
        )
        depth_x2 = int(
            np.clip(depth_x2, 0, depth_width)
        )
        depth_y1 = int(
            np.clip(depth_y1, 0, depth_height)
        )
        depth_y2 = int(
            np.clip(depth_y2, 0, depth_height)
        )

        cropped_image = image[y1:y2, x1:x2]
        cropped_depth = depth_image[
            depth_y1:depth_y2,
            depth_x1:depth_x2,
        ]

        if cropped_image.size == 0:
            raise ValueError(
                "RTMPose crop produced an empty RGB image."
            )

        if cropped_depth.size == 0:
            raise ValueError(
                "RTMPose crop produced an empty depth image."
            )

        return cropped_image, cropped_depth

    def _extract_features(
        self,
        rgb_image,
        depth_image,
    ):
        """
        Construct the feature vector using the same
        preprocessing used during training.
        """
        if rgb_image is None:
            raise ValueError(
                "rgb_image cannot be None."
            )

        if depth_image is None:
            raise ValueError(
                "depth_image cannot be None."
            )

        # predict() accepts RGB, while training images were
        # loaded through cv2.imread() and were therefore BGR.
        image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

        depth_image = np.asarray(depth_image)


        # ---------------------------------------------------------
        # Optional RTMPose crop
        # ---------------------------------------------------------
        # First RTMPose pass: localize the person on the full frame and
        # crop RGB/depth. The second RTMPose pass below runs on the crop
        # and provides the actual model features, matching training.
        if self.crop_around_object:
            image, depth_image = self._crop_images(
                image,
                depth_image,
            )

            if image is None:
                return None

        # ---------------------------------------------------------
        # Person keypoints
        # ---------------------------------------------------------
        # Second RTMPose pass: this runs on the cropped image and is the
        # detection used for model features, head pose, and depth lookup.
        people = self.keypoint_detector.predict(
            image
        )

        if len(people) != 1:
            return None

        if not self._is_valid_person(people[0]):
            return None

        # Reject clear back-facing views before constructing any
        # handoff features.
        if self._is_facing_away(people[0]):
            return None

        # Continuous 2D side-view evidence. This never rejects the frame.
        side_score_2d = self._side_facing_score_2d(
            people[0]
        )

        keypoints = np.asarray(
            people[0]["keypoints"][:11],
            dtype=np.float32,
        )

        left_shoulder_xy = (
            keypoints[5, :2].copy()
        )
        right_shoulder_xy = (
            keypoints[6, :2].copy()
        )

        shoulder_midpoint_xy = (
            left_shoulder_xy
            + right_shoulder_xy
        ) / 2.0

        # ---------------------------------------------------------
        # Depth features
        # ---------------------------------------------------------
        shoulder_midpoint_depth = (
            get_depth_at_rgb_point(
                depth_image=depth_image,
                rgb_image_shape=image.shape,
                point_xy=shoulder_midpoint_xy,
                patch_radius=(
                    self.depth_patch_radius
                ),
            )
        )

        relative_joint_depths = []

        # COCO:
        # 5,6 = shoulders
        # 7,8 = elbows
        # 9,10 = wrists
        for keypoint_index in range(5, 11):
            absolute_depth = (
                get_depth_at_rgb_point(
                    depth_image=depth_image,
                    rgb_image_shape=image.shape,
                    point_xy=keypoints[
                        keypoint_index,
                        :2,
                    ],
                    patch_radius=(
                        self.depth_patch_radius
                    ),
                )
            )

            relative_joint_depths.append(
                absolute_depth
                - shoulder_midpoint_depth
            )

        # ---------------------------------------------------------
        # Continuous side-view / arm-presentation evidence
        # ---------------------------------------------------------
        # This section no longer rejects frames. It only computes evidence
        # for a soft post-TabM correction.
        side_score = 0.0
        presentation_score = 1.0
        side_correction_valid = False

        if self.reject_side_without_forward_wrist:
            left_shoulder_relative_depth = float(relative_joint_depths[0])
            right_shoulder_relative_depth = float(relative_joint_depths[1])
            left_elbow_relative_depth = float(relative_joint_depths[2])
            right_elbow_relative_depth = float(relative_joint_depths[3])
            left_wrist_relative_depth = float(relative_joint_depths[4])
            right_wrist_relative_depth = float(relative_joint_depths[5])

            shoulder_depth_difference = abs(
                left_shoulder_relative_depth
                - right_shoulder_relative_depth
            )
            shoulder_depth_fraction = (
                shoulder_depth_difference
                / max(float(shoulder_midpoint_depth), 1e-6)
            )
            side_score_depth = float(
                np.clip(
                    shoulder_depth_fraction
                    / (2.0 * self.side_shoulder_depth_fraction),
                    0.0,
                    1.0,
                )
            )

            # Unlike the previous OR, both independent side-view cues must
            # agree before the correction becomes strong.
            side_score = float(
                np.clip(
                    side_score_2d * side_score_depth,
                    0.0,
                    1.0,
                )
            )

            if keypoints.shape[1] >= 3:
                confidence_threshold = self.person_presence_confidence
                left_elbow_score = float(keypoints[7, 2])
                right_elbow_score = float(keypoints[8, 2])
                left_wrist_score = float(keypoints[9, 2])
                right_wrist_score = float(keypoints[10, 2])
                left_arm_reliable = (
                    np.isfinite(left_elbow_score)
                    and np.isfinite(left_wrist_score)
                    and left_elbow_score >= confidence_threshold
                    and left_wrist_score >= confidence_threshold
                )
                right_arm_reliable = (
                    np.isfinite(right_elbow_score)
                    and np.isfinite(right_wrist_score)
                    and right_elbow_score >= confidence_threshold
                    and right_wrist_score >= confidence_threshold
                )
            else:
                left_arm_reliable = True
                right_arm_reliable = True

            def _arm_presentation_score(
                shoulder_relative_depth,
                elbow_relative_depth,
                wrist_relative_depth,
            ):
                # Compare each wrist to its OWN anatomical shoulder instead of
                # using whichever shoulder happens to be nearest to the camera.
                shoulder_depth = (
                    shoulder_midpoint_depth
                    + shoulder_relative_depth
                )
                depth_scale = max(float(shoulder_depth), 1e-6)

                shoulder_forward_fraction = max(
                    0.0,
                    (shoulder_relative_depth - wrist_relative_depth)
                    / depth_scale,
                )
                elbow_forward_fraction = max(
                    0.0,
                    (elbow_relative_depth - wrist_relative_depth)
                    / depth_scale,
                )
                shoulder_component = float(
                    np.clip(
                        shoulder_forward_fraction
                        / self.side_wrist_forward_depth_fraction,
                        0.0,
                        1.0,
                    )
                )
                elbow_component = float(
                    np.clip(
                        elbow_forward_fraction
                        / self.side_wrist_beyond_elbow_depth_fraction,
                        0.0,
                        1.0,
                    )
                )

                # A side-view arm should only count as genuinely presented
                # when the wrist is meaningfully forward of BOTH its shoulder
                # and its elbow. Using min() makes this conjunctive while
                # retaining a continuous 0..1 score.
                return min(shoulder_component, elbow_component)

            arm_scores = []
            if left_arm_reliable:
                arm_scores.append(
                    _arm_presentation_score(
                        left_shoulder_relative_depth,
                        left_elbow_relative_depth,
                        left_wrist_relative_depth,
                    )
                )
            if right_arm_reliable:
                arm_scores.append(
                    _arm_presentation_score(
                        right_shoulder_relative_depth,
                        right_elbow_relative_depth,
                        right_wrist_relative_depth,
                    )
                )

            if arm_scores:
                presentation_score = float(max(arm_scores))
                side_correction_valid = True
            else:
                # Missing arm keypoints are uncertainty, not negative evidence.
                side_score = 0.0
                presentation_score = 1.0
                side_correction_valid = False

        # ---------------------------------------------------------
        # Normalize 2D keypoints
        # ---------------------------------------------------------
        if self.normalize_keypoints:
            keypoints[:, :2] = (
                keypoints[:, :2]
                - shoulder_midpoint_xy
            )

        feature_vector = (
            keypoints
            .astype(np.float32)
            .reshape(-1)
            .tolist()
        )

        # ---------------------------------------------------------
        # Head pose
        # ---------------------------------------------------------
        if "headpose" in self.features_type:
            head_pose = (
                self.head_pose_estimator
                .predict_first_or_sentinel(
                    image,
                    rtmpose_results=people,
                )
            )

            feature_vector += (
                np.asarray(
                    head_pose,
                    dtype=np.float32,
                )
                .reshape(-1)
                .tolist()
            )

        # ---------------------------------------------------------
        # ResNet embedding
        # ---------------------------------------------------------
        if "resnet" in self.features_type:
            embedding = (
                self.img_encoder.predict(
                    image
                )["embedding"]
            )

            feature_vector += (
                np.asarray(
                    embedding,
                    dtype=np.float32,
                )
                .reshape(-1)
                .tolist()
            )

        # Training appends relative depth LAST.
        feature_vector += relative_joint_depths

        features = np.asarray(
            feature_vector,
            dtype=np.float32,
        )

        if features.size != self.input_dim:
            raise ValueError(
                "Feature dimension mismatch: "
                f"model expects {self.input_dim}, "
                f"but preprocessing produced "
                f"{features.size}."
            )

        self._side_metadata_for_last_features = {
            "side_score_2d": float(side_score_2d),
            "side_score": float(side_score),
            "presentation_score": float(presentation_score),
            "valid": bool(side_correction_valid),
        }

        return features

    def predict(
        self,
        rgb_image,
        depth_image,
    ):
        """
        Predict whether the person is attempting
        a handoff.

        Args:
            rgb_image:
                HxWx3 RGB numpy array.

            depth_image:
                HxW depth numpy array. Depth units
                must match those used during training.

        Returns:
            classification:
                "handoff" or "not_handoff"

            confidence:
                Confidence in the returned class,
                from 0.0 to 1.0.
        """
        # Reset transient correction metadata so an early return from feature
        # extraction can never reuse geometry from the previous frame.
        self._side_metadata_for_last_features = {
            "side_score_2d": 0.0,
            "side_score": 0.0,
            "presentation_score": 1.0,
            "valid": False,
        }

        features = self._extract_features(
            rgb_image,
            depth_image,
        )

        # Missing/invalid person and clear back-facing cases remain normal
        # negatives. Side views are no longer force-rejected.
        if features is None:
            self.last_raw_handoff_probability = None
            self.last_adjusted_handoff_probability = None
            self.last_side_score = 0.0
            self.last_presentation_score = 1.0
            self.last_side_probability_penalty = 0.0
            return "not_handoff", 1.0

        side_metadata = self._side_metadata_for_last_features

        input_tensor = (
            torch.from_numpy(features)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(input_tensor)

            # HandIntentTabM returns one raw logit per ensemble member.
            # Expected shape is [batch, k, 1]. Some wrappers may squeeze
            # the final singleton dimension and return [batch, k].
            if logits.ndim == 2:
                logits = logits.unsqueeze(-1)

            if logits.ndim != 3:
                raise ValueError(
                    "Expected TabM output with shape [batch, k, 1] "
                    f"or [batch, k], got {tuple(logits.shape)}."
                )

            if logits.shape[-1] != 1:
                raise ValueError(
                    "Expected binary TabM output dimension 1, "
                    f"got {logits.shape[-1]}."
                )

            # For classification, convert each member logit to a
            # probability first and then average the probabilities.
            member_probabilities = torch.sigmoid(logits)

            handoff_probability = float(
                member_probabilities
                .mean(dim=1)
                .squeeze(-1)
                .item()
            )

        raw_handoff_probability = handoff_probability

        side_score_2d = float(
            side_metadata.get("side_score_2d", 0.0)
        )
        side_score = float(side_metadata.get("side_score", 0.0))
        presentation_score = float(
            side_metadata.get("presentation_score", 1.0)
        )
        side_correction_valid = bool(side_metadata.get("valid", False))
        side_probability_penalty = 0.0

        # Correct only predictions for which TabM itself is leaning positive.
        if (
            self.reject_side_without_forward_wrist
            and side_correction_valid
            and raw_handoff_probability
            >= self.side_correction_min_raw_probability
        ):
            suspicious_side_evidence = (
                side_score * (1.0 - presentation_score)
            )
            side_probability_penalty = float(
                np.clip(
                    self.side_max_probability_penalty
                    * suspicious_side_evidence,
                    0.0,
                    self.side_max_probability_penalty,
                )
            )
            handoff_probability = (
                raw_handoff_probability
                * (1.0 - side_probability_penalty)
            )

        # For clearly side-facing views, high handoff probability must be
        # justified by genuine arm presentation. This does NOT reject side
        # views outright: stronger presentation raises the allowed ceiling.
        if (
            self.side_probability_cap_enabled
            and side_correction_valid
            and side_score_2d >= self.side_probability_cap_min_score_2d
        ):
            side_view_probability_cap = (
                self.side_probability_cap_base
                + self.side_probability_cap_presentation_gain
                * presentation_score
            )
            side_view_probability_cap = float(
                np.clip(side_view_probability_cap, 0.0, 1.0)
            )
            handoff_probability = min(
                handoff_probability,
                side_view_probability_cap,
            )

        handoff_probability = float(
            np.clip(handoff_probability, 0.0, 1.0)
        )

        self.last_raw_handoff_probability = float(raw_handoff_probability)
        self.last_adjusted_handoff_probability = float(handoff_probability)
        self.last_side_score = float(side_score)
        self.last_presentation_score = float(presentation_score)
        self.last_side_probability_penalty = float(side_probability_penalty)

        if handoff_probability >= self.threshold:
            classification = "handoff"
            confidence = handoff_probability
        else:
            classification = "not_handoff"
            confidence = 1.0 - handoff_probability

        return classification, confidence