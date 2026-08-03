from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YOLOGo2Detector:
    """
    Ultralytics YOLO inference wrapper compatible with RobotDetectorTracker.

    The input image may have any width and height. Ultralytics automatically
    resizes and letterboxes it for inference, then maps detections back to the
    original image coordinates.

    Expected output:
        [
            {
                "label": "unitree go2",
                "score": 0.95,
                "box_xyxy": [x1, y1, x2, y2],
            }
        ]
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence: float = 0.40,
        iou_threshold: float = 0.50,
        image_size: int = 640,
        device: Optional[Union[int, str]] = None,
        max_detections: int = 10,
    ) -> None:
        if weights_path is None:
            weights_path = self._default_weights_path()

        self.weights_path = Path(weights_path).expanduser().resolve()

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                "Could not find YOLO weights:\n"
                f"{self.weights_path}\n\n"
                "Either place best.pt at the default location or pass "
                "weights_path explicitly."
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")

        if image_size < 32:
            raise ValueError("image_size must be at least 32.")

        if max_detections < 1:
            raise ValueError("max_detections must be at least 1.")

        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"

        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.image_size = int(image_size)
        self.device = device
        self.max_detections = int(max_detections)

        self.model = YOLO(str(self.weights_path))

        print("Loaded Go2 YOLO detector")
        print(f"  Weights: {self.weights_path}")
        print(f"  Device:  {self.device}")
        print(f"  Classes: {self.model.names}")

    @staticmethod
    def _default_weights_path() -> Path:
        """
        Find the robot_fov_estimation directory relative to this file and
        return the standard trained-weights location.
        """
        current_directory = Path(__file__).resolve().parent

        for candidate in (current_directory, *current_directory.parents):
            if candidate.name == "robot_fov_estimation":
                return (
                    candidate
                    / "outputs"
                    / "go2_yolo_det_weights"
                    / "best.pt"
                )

        raise RuntimeError(
            "Could not locate the robot_fov_estimation directory relative "
            f"to the YOLO wrapper file:\n{Path(__file__).resolve()}"
        )

    @staticmethod
    def _prepare_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
        """
        Validate and normalize an input image without changing its resolution.

        Supports:
        - RGB images with arbitrary width and height
        - Grayscale images
        - Single-channel images
        - RGBA images
        - uint8, integer, or floating-point arrays
        """
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError("image_rgb must be a NumPy array.")

        if image_rgb.size == 0:
            raise ValueError("image_rgb cannot be empty.")

        # Convert grayscale HxW to RGB.
        if image_rgb.ndim == 2:
            image_rgb = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_GRAY2RGB,
            )

        elif image_rgb.ndim == 3:
            channel_count = image_rgb.shape[2]

            if channel_count == 1:
                image_rgb = np.repeat(image_rgb, 3, axis=2)

            elif channel_count == 3:
                pass

            elif channel_count == 4:
                image_rgb = cv2.cvtColor(
                    image_rgb,
                    cv2.COLOR_RGBA2RGB,
                )

            else:
                raise ValueError(
                    "image_rgb must have 1, 3, or 4 channels. "
                    f"Received shape: {image_rgb.shape}"
                )

        else:
            raise ValueError(
                "image_rgb must have shape HxW, HxWx1, HxWx3, or HxWx4. "
                f"Received shape: {image_rgb.shape}"
            )

        height, width = image_rgb.shape[:2]

        if height < 1 or width < 1:
            raise ValueError(
                f"Invalid image dimensions: width={width}, height={height}"
            )

        # Convert floating-point or other numeric images to uint8.
        if image_rgb.dtype != np.uint8:
            if not (
                np.issubdtype(image_rgb.dtype, np.integer)
                or np.issubdtype(image_rgb.dtype, np.floating)
            ):
                raise ValueError(
                    "image_rgb must contain integer or floating-point values."
                )

            image_rgb = np.nan_to_num(
                image_rgb,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )

            # Treat floating-point images in [0, 1] as normalized images.
            if (
                np.issubdtype(image_rgb.dtype, np.floating)
                and image_rgb.min() >= 0.0
                and image_rgb.max() <= 1.0
            ):
                image_rgb = image_rgb * 255.0

            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(image_rgb)

    def _class_name(self, class_id: int) -> str:
        names = self.model.names

        if isinstance(names, dict):
            return str(names.get(class_id, class_id))

        if 0 <= class_id < len(names):
            return str(names[class_id])

        return str(class_id)

    def predict(
        self,
        image_rgb: np.ndarray,
        class_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """
        Run inference on one image.

        The image may have any resolution. For example:
            640x640
            1280x720
            1920x1080
            1024x768

        class_names is accepted for compatibility with the existing detector
        interface. The YOLO model's trained class names are used instead.
        """
        del class_names

        image_rgb = self._prepare_rgb_image(image_rgb)

        # RobotDetectorTracker converts its OpenCV BGR frame to RGB before
        # calling this method. Ultralytics expects NumPy/OpenCV inputs in BGR
        # order, so convert the image back here.
        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )

        results = self.model.predict(
            source=image_bgr,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            max_det=self.max_detections,
            rect=True,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return []

        xyxy_values = boxes.xyxy.detach().cpu().numpy()
        confidence_values = boxes.conf.detach().cpu().numpy()
        class_values = boxes.cls.detach().cpu().numpy()

        detections: List[Dict[str, Any]] = []

        for box, score, class_value in zip(
            xyxy_values,
            confidence_values,
            class_values,
        ):
            class_id = int(class_value)

            detections.append(
                {
                    "label": self._class_name(class_id),
                    "score": float(score),
                    "box_xyxy": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                }
            )

        return detections