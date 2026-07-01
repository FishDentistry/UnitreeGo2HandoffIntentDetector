import numpy as np

from rtmlib import Body


UPPER_BODY_IDXS = list(range(11))  # 0-10: nose, eyes, ears, shoulders, elbows, wrists

# More important for handover classification than eyes/ears.
REQUIRED_UPPER_BODY_IDXS = [
    0,   # nose
    5,   # left_shoulder
    6,   # right_shoulder
    7,   # left_elbow
    8,   # right_elbow
    9,   # left_wrist
    10,  # right_wrist
]


class RTMPoseKeypointDetector:
    """
    Replacement for the old Detectron2 keypoint wrapper as in Kwan.

    Keeps the return format:
        [
            {
                "box_xyxy": [x1, y1, x2, y2],
                "score": float,
                "keypoints": [[x, y, conf], ...],      # still 17 keypoints
                "keypoints_xy": [[x, y], ...],         # still 17 keypoints
            },
            ...
        ]
    """

    def __init__(
        self,
        confidence: float = 0.35,
        config_path: str = None,
        weights_path: str = None,
        head_pose_pretrained_path: str = None,
        mlp_localized_path: str = None,
        mlp_nonlocalized_path: str = None,
        device: str = "cpu",
        backend: str = "onnxruntime",
        mode: str = "balanced",
        upper_body_confidence: float = None,
        required_keypoint_confidence: float = None,
    ):
        self.confidence = float(confidence)

        # Use separate thresholds so person filtering is not tied too tightly
        # to object/DINO confidence.
        self.upper_body_confidence = (
            float(upper_body_confidence)
            if upper_body_confidence is not None
            else float(confidence)
        )

        self.required_keypoint_confidence = (
            float(required_keypoint_confidence)
            if required_keypoint_confidence is not None
            else float(confidence)
        )

        self.body = Body(
            mode=mode,
            backend=backend,
            device=device,
            to_openpose=False,  # keep COCO-style ordering
        )

    def predict(self, image_bgr):
        keypoints, scores = self.body(image_bgr)

        if keypoints is None or len(keypoints) == 0:
            return []

        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        # In case scores comes back as [N, 17, 1].
        if scores.ndim == 3 and scores.shape[-1] == 1:
            scores = scores[..., 0]

        people = []

        for i in range(len(keypoints)):
            kpts_xy = keypoints[i]      # expected [17, 2]
            kpt_scores = scores[i]      # expected [17]

            if kpts_xy.size == 0:
                continue

            if kpts_xy.shape[0] < 11 or kpts_xy.shape[1] < 2:
                continue

            if not np.all(np.isfinite(kpts_xy[:11, :2])):
                continue

            if not np.all(np.isfinite(kpt_scores[:11])):
                continue

            upper_scores = kpt_scores[UPPER_BODY_IDXS]
            required_scores = kpt_scores[REQUIRED_UPPER_BODY_IDXS]

            # Score this person based on what Kwan's classifier actually uses.
            person_score = float(np.nanmean(upper_scores))

            # Reject people whose upper-body estimate is weak.
            if person_score < self.upper_body_confidence:
                continue

            # Reject if key handover-relevant joints are unreliable.
            if np.any(required_scores < self.required_keypoint_confidence):
                continue

            # Use visible upper-body keypoints for the person box.
            visible_upper = np.asarray(UPPER_BODY_IDXS)[
                upper_scores >= self.upper_body_confidence
            ]

            if len(visible_upper) > 0:
                xy_for_box = kpts_xy[visible_upper, :2]
            else:
                xy_for_box = kpts_xy[:11, :2]

            x1, y1 = np.min(xy_for_box[:, 0]), np.min(xy_for_box[:, 1])
            x2, y2 = np.max(xy_for_box[:, 0]), np.max(xy_for_box[:, 1])

            keypoint_set = [
                [float(x), float(y), float(s)]
                for (x, y), s in zip(kpts_xy, kpt_scores)
            ]

            person = {
                "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "score": person_score,
                "keypoints": keypoint_set,
                "keypoints_xy": [[float(x), float(y)] for x, y in kpts_xy],
                "upper_body_score": person_score,
                "required_upper_body_scores": {
                    "nose": float(kpt_scores[0]),
                    "left_shoulder": float(kpt_scores[5]),
                    "right_shoulder": float(kpt_scores[6]),
                    "left_elbow": float(kpt_scores[7]),
                    "right_elbow": float(kpt_scores[8]),
                    "left_wrist": float(kpt_scores[9]),
                    "right_wrist": float(kpt_scores[10]),
                },
            }

            people.append(person)

        return people