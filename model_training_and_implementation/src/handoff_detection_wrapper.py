from pathlib import Path

import cv2
import numpy as np
import torch

from .rtmpose_keypoints import RTMPoseKeypointDetector
from .rtmpose_headpose import RTMPoseHeadPoseEstimator
from .hand_intent_mlp import HandIntentMLP
from .resnet_encoder import ResNet18ImageEncoder


# This file lives in:
# model_training_and_implementation/src/
MODEL_IMPL_ROOT = Path(__file__).resolve().parents[1]

MODEL_WEIGHTS_ROOT = (
    MODEL_IMPL_ROOT
    / "outputs"
    / "hand_intent_mlp_weights"
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
                / "MLP.pth"
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
        # Load MLP
        # ---------------------------------------------------------
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.input_dim = int(checkpoint["input_dim"])

        self.model = HandIntentMLP(
            input_size=self.input_dim,
            output_size=1,
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
        # DINO compatibility
        # ---------------------------------------------------------
        # Kept only so existing call signatures do not need to
        # change. DINO is no longer loaded or used.
        self.object_detector = None

        if self.features_type == "keypoints_headpose_dino":
            raise ValueError(
                'features_type="keypoints_headpose_dino" '
                "requires DINO object-centroid features and is "
                "not supported when DINO is disabled."
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

    def _crop_images(
        self,
        image,
        depth_image,
    ):
        """
        Crop RGB and depth images around the same detected
        region, matching the training preprocessing.
        """
        people = self.keypoint_detector.predict(
            image
        )

        if len(people) != 1:
            raise RuntimeError(
                "Expected exactly one person for cropping, "
                f"found {len(people)}."
            )

        keypoints = np.asarray(
            people[0]["keypoints"],
            dtype=np.float32,
        )

        xy = keypoints[:, :2]
        valid = np.isfinite(xy).all(axis=1)
        valid &= ~np.all(xy == 0, axis=1)

        if not np.any(valid):
            raise RuntimeError(
                "Could not determine person crop from "
                "RTMPose keypoints."
            )

        valid_xy = xy[valid]

        x1 = float(np.min(valid_xy[:, 0]))
        y1 = float(np.min(valid_xy[:, 1]))
        x2 = float(np.max(valid_xy[:, 0]))
        y2 = float(np.max(valid_xy[:, 1]))

        # Expand the keypoint extent to approximate the
        # person detector box previously supplied by DINO.
        box_width = x2 - x1
        box_height = y2 - y1
        padding_x = 0.15 * box_width
        padding_y = 0.15 * box_height

        x1 = int(x1 - padding_x)
        y1 = int(y1 - padding_y)
        x2 = int(x2 + padding_x)
        y2 = int(y2 + padding_y)

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
        # Optional crop
        # ---------------------------------------------------------
        if self.crop_around_object:
            image, depth_image = (
                self._crop_images(
                    image,
                    depth_image,
                )
            )

        # ---------------------------------------------------------
        # Person keypoints
        # ---------------------------------------------------------
        people = self.keypoint_detector.predict(
            image
        )

        if len(people) != 1:
            raise RuntimeError(
                "Expected exactly one person, "
                f"found {len(people)}."
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

        input_tensor = (
            torch.from_numpy(features)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            handoff_probability = float(
                self.model(input_tensor).item()
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