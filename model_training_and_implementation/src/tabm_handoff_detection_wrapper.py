from pathlib import Path

import cv2
import numpy as np
import torch

from .dino_detector import DINOObjectDetector
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
    ):
        self.features_type = features_type
        self.threshold = float(threshold)
        self.normalize_keypoints = bool(normalize_keypoints)
        self.depth_patch_radius = int(depth_patch_radius)
        self.crop_around_object = bool(crop_around_object)
        self.crop_object = crop_object
        self.dino_classes = dino_classes
        self.keypoint_confidence = float(confidence)

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
            )
        )

        # ---------------------------------------------------------
        # DINO object detector
        # ---------------------------------------------------------
        self.object_detector = None

        if (
            self.crop_around_object
            or self.features_type
            == "keypoints_headpose_dino"
        ):
            self.object_detector = DINOObjectDetector(
                model_id=(
                    "IDEA-Research/"
                    "grounding-dino-base"
                ),
                confidence=confidence,
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

    def _crop_images(
        self,
        image,
        depth_image,
    ):
        """
        Crop RGB and depth images around the same detected
        region, matching the training preprocessing.
        """
        detections = self.object_detector.predict(
            image,
            class_names=self.crop_object,
        )

        matching = [
            detection
            for detection in detections
            if detection["label"].strip(".")
            == self.crop_object.strip(".")
        ]

        if not matching:
            return None, None

        # Match training behavior: select largest
        # matching detection.
        detection = max(
            matching,
            key=lambda d: (
                d["box_xyxy"][2]
                - d["box_xyxy"][0]
            )
            * (
                d["box_xyxy"][3]
                - d["box_xyxy"][1]
            ),
        )

        x1, y1, x2, y2 = map(
            int,
            detection["box_xyxy"],
        )

        rgb_height, rgb_width = image.shape[:2]
        depth_height, depth_width = (
            depth_image.shape[:2]
        )

        x1 = int(np.clip(x1, 0, rgb_width))
        x2 = int(np.clip(x2, 0, rgb_width))
        y1 = int(np.clip(y1, 0, rgb_height))
        y2 = int(np.clip(y2, 0, rgb_height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "Object detection produced an "
                "invalid RGB crop."
            )

        # Map RGB crop coordinates into depth image.
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

        image = image[y1:y2, x1:x2]

        depth_image = depth_image[
            depth_y1:depth_y2,
            depth_x1:depth_x2,
        ]

        if image.size == 0:
            raise ValueError(
                "Crop produced an empty RGB image."
            )

        if depth_image.size == 0:
            raise ValueError(
                "Crop produced an empty depth image."
            )

        return image, depth_image

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
        # Optional DINO crop
        # ---------------------------------------------------------
        # This intentionally happens BEFORE RTMPose, head pose, and
        # ResNet so runtime preprocessing matches training.
        if self.crop_around_object:
            image, depth_image = self._crop_images(
                image,
                depth_image,
            )

            if image is None:
                return None

        # ---------------------------------------------------------
        # Optional DINO object feature
        # ---------------------------------------------------------
        objects = []

        if (
            self.features_type
            == "keypoints_headpose_dino"
        ):
            objects = self.object_detector.predict(
                image,
                class_names=self.dino_classes,
            )

            if len(objects) != 1:
                return None

        # ---------------------------------------------------------
        # Person keypoints
        # ---------------------------------------------------------
        # RTMPose runs on the same (possibly DINO-cropped) image
        # used during training.
        people = self.keypoint_detector.predict(
            image
        )

        if len(people) == 0:
            return None

        if len(people) != 1:
            raise RuntimeError(
                "Expected exactly one person, "
                f"found {len(people)}."
            )

        if not self._is_valid_person(people[0]):
            return None

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
        # DINO object centroid
        # ---------------------------------------------------------
        if (
            self.features_type
            == "keypoints_headpose_dino"
        ):
            x1, y1, x2, y2 = map(
                float,
                objects[0]["box_xyxy"],
            )

            feature_vector += [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            ]

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
        features = self._extract_features(
            rgb_image,
            depth_image,
        )

        # No sufficiently confident person means a handoff is not
        # possible. Return a normal negative prediction so ROS
        # publishes a fresh state instead of retaining the previous
        # classification after an exception.
        if features is None:
            return "not_handoff", 1.0

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

        if handoff_probability >= self.threshold:
            classification = "handoff"
            confidence = handoff_probability
        else:
            classification = "not_handoff"
            confidence = (
                1.0 - handoff_probability
            )

        return classification, confidence