"""
robot_detector_tracker_ostrack.py

Hybrid detector + OSTrack robot tracking for:

1. Real-time Quest image streams
   - Use LatestFrameRobotTracker.
   - Only the newest waiting frame is retained, preventing latency buildup.

2. Offline video testing
   - Use process_video(...), or edit the settings at the bottom and run this
     file directly.

The injected detector must provide:

    detector.predict(image_rgb, class_names)

and return a list of dictionaries shaped like:

    {
        "label": str,
        "score": float,
        "box_xyxy": [x1, y1, x2, y2],
    }

OSTrack setup expected by this file:

- Clone the official OSTrack repository.
- Download the matching OSTrack checkpoint.
- Set OSTRACK_ROOT and OSTRACK_CHECKPOINT_PATH near the bottom of this file.
- The official OSTrack inference code uses CUDA directly, so a CUDA-capable
  PyTorch installation is required unless you separately modify OSTrack.

Tracking flow:

1. While SEARCHING, run the detector on the full frame.
2. Require repeated high-confidence detections before acquisition.
3. Initialize OSTrack from the confirmed detector box.
4. Use OSTrack on every subsequent frame.
5. Periodically verify OSTrack with the semantic detector.
6. If full-frame verification misses, retry the detector on an expanded crop
   around the OSTrack box.
7. Correct OSTrack's search state whenever a detector verification succeeds.
8. After repeated detector-verification misses, discard the OSTrack track and
   resume global detector search.
9. While searching, periodically use overlapping tiled detector inference to
   improve recall for small, distant robots.

The detector remains responsible for semantic acquisition and reacquisition.
OSTrack replaces the old per-frame detector/Kalman tracking path.
"""

from __future__ import annotations

import importlib
import math
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

ROBOT_FOV_ROOT = Path(__file__).resolve().parents[1]

# Example layout:
# <project>/third_party/OSTrack/
# <project>/third_party/OSTrack/output/checkpoints/train/ostrack/
#     vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar
OSTRACK_ROOT = ROBOT_FOV_ROOT / "ostrack"
OSTRACK_CONFIG_NAME = "vitb_256_mae_ce_32x4_ep300"
OSTRACK_CHECKPOINT_PATH = (
    OSTRACK_ROOT
    / "output"
    / "checkpoints"
    / "train"
    / "ostrack"
    / OSTRACK_CONFIG_NAME
    / "OSTrack_ep0300.pth.tar"
)


BBoxXYXY = Tuple[int, int, int, int]
PathLike = Union[str, Path]


class TrackingState(str, Enum):
    SEARCHING = "searching"
    DETECTED = "detected"
    TRACKING = "tracking"


@dataclass(frozen=True)
class RobotTrackResult:
    state: TrackingState
    bbox_xyxy: Optional[BBoxXYXY]
    frame_id: int
    frame_timestamp: float
    completed_timestamp: float
    processing_time_ms: float
    source: str
    detector_score: Optional[float] = None
    detector_label: Optional[str] = None

    # True means a semantic detector supported the returned box on this frame.
    # False can still mean that OSTrack produced a valid short-term result.
    tracker_verified: bool = False

    acquisition_candidate_count: int = 0
    replacement_candidate_count: int = 0
    verification_misses: int = 0
    prediction_age_frames: int = 0
    prediction_age_seconds: float = 0.0
    association_score: Optional[float] = None
    used_crop_redetection: bool = False
    used_tiled_search: bool = False

    @property
    def found(self) -> bool:
        return self.bbox_xyxy is not None

    @property
    def end_to_end_latency_ms(self) -> float:
        return max(
            0.0,
            (self.completed_timestamp - self.frame_timestamp) * 1000.0,
        )


@dataclass(frozen=True)
class _DetectionCandidate:
    bbox_xyxy: BBoxXYXY
    score: float
    label: str
    source: str

class OSTrackAdapter:
    """
    Small adapter around the official OSTrack repository.

    The official tracker accepts and returns [x, y, width, height]. This
    adapter exposes [x1, y1, x2, y2] to the rest of this file.

    Imports are intentionally delayed until this class is instantiated, so
    simply importing this module does not require OSTrack to be installed.
    """

    def __init__(
        self,
        ostrack_root: Optional[PathLike] = None,
        checkpoint_path: Optional[PathLike] = None,
        config_name: str = "vitb_256_mae_ce_32x4_ep300",
        dataset_name: str = "robot",
    ) -> None:
        self.config_name = str(config_name)
        self.dataset_name = str(dataset_name)

        if ostrack_root is None:
            resolved_ostrack_root = ROBOT_FOV_ROOT / "ostrack"
        else:
            resolved_ostrack_root = Path(ostrack_root).expanduser()

        self.ostrack_root = resolved_ostrack_root.resolve()

        if checkpoint_path is None:
            resolved_checkpoint_path = (
                self.ostrack_root
                / "output"
                / "checkpoints"
                / "train"
                / "ostrack"
                / self.config_name
                / "OSTrack_ep0300.pth.tar"
            )
        else:
            resolved_checkpoint_path = Path(checkpoint_path).expanduser()

        self.checkpoint_path = resolved_checkpoint_path.resolve()

        if not self.ostrack_root.is_dir():
            raise FileNotFoundError(
                "OSTrack repository does not exist:\n"
                f"  {self.ostrack_root}\n\n"
                "Pass ostrack_root explicitly or place the repository at:\n"
                f"  {ROBOT_FOV_ROOT / 'ostrack'}"
            )

        if not (self.ostrack_root / "lib").is_dir():
            raise FileNotFoundError(
                "OSTrack root must contain the official repository's lib/ "
                "folder:\n"
                f"  {self.ostrack_root}"
            )

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "OSTrack checkpoint does not exist:\n"
                f"  {self.checkpoint_path}\n\n"
                "Pass checkpoint_path explicitly or place the checkpoint at "
                "the default location shown above."
            )

        config_path = (
            self.ostrack_root
            / "experiments"
            / "ostrack"
            / f"{self.config_name}.yaml"
        )

        if not config_path.is_file():
            raise FileNotFoundError(
                f"OSTrack config does not exist: {config_path}"
            )

        root_string = str(self.ostrack_root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)

        importlib.invalidate_caches()

        try:
            import torch
            from lib.config.ostrack.config import (
                cfg,
                update_config_from_file,
            )
            from lib.test.tracker.ostrack import OSTrack
            from lib.test.utils import TrackerParams
        except Exception as exc:
            raise RuntimeError(
                "Could not import the official OSTrack implementation. "
                "Confirm that the OSTrack repository and its dependencies "
                "are installed correctly."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The official OSTrack implementation used here hard-codes "
                "CUDA during inference, but torch.cuda.is_available() is "
                "False. Use a CUDA PyTorch environment or modify OSTrack's "
                "tracker/preprocessor code for CPU execution."
            )

        update_config_from_file(str(config_path))

        params = TrackerParams()
        params.cfg = cfg
        params.template_factor = cfg.TEST.TEMPLATE_FACTOR
        params.template_size = cfg.TEST.TEMPLATE_SIZE
        params.search_factor = cfg.TEST.SEARCH_FACTOR
        params.search_size = cfg.TEST.SEARCH_SIZE
        params.checkpoint = str(self.checkpoint_path)
        params.save_all_boxes = False
        params.debug = 0

        try:
            self._tracker = OSTrack(params, self.dataset_name)
        except Exception as exc:
            raise RuntimeError(
                "OSTrack model construction failed. Confirm that the "
                "checkpoint matches the selected config_name and that the "
                "installed PyTorch, torchvision, and timm versions are "
                "compatible."
            ) from exc

        self._initialized = False

    @staticmethod
    def _validate_frame(frame_bgr: np.ndarray) -> None:
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a NumPy array.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (height, width, 3).")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must use dtype uint8.")

    @staticmethod
    def _xyxy_to_xywh(box: BBoxXYXY) -> List[float]:
        x1, y1, x2, y2 = box
        return [
            float(x1),
            float(y1),
            float(max(1, x2 - x1)),
            float(max(1, y2 - y1)),
        ]

    @staticmethod
    def _xywh_to_clipped_xyxy(
        box_xywh,
        frame_width: int,
        frame_height: int,
    ) -> Optional[BBoxXYXY]:
        if box_xywh is None or len(box_xywh) != 4:
            return None

        x, y, width, height = (float(value) for value in box_xywh)
        if not np.all(np.isfinite([x, y, width, height])):
            return None
        if width <= 1.0 or height <= 1.0:
            return None

        x1 = int(round(max(0.0, x)))
        y1 = int(round(max(0.0, y)))
        x2 = int(round(min(float(frame_width), x + width)))
        y2 = int(round(min(float(frame_height), y + height)))

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def initialize(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: BBoxXYXY,
    ) -> None:
        self._validate_frame(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        init_bbox_xywh = self._xyxy_to_xywh(bbox_xyxy)

        self._tracker.initialize(
            frame_rgb,
            {"init_bbox": init_bbox_xywh},
        )
        self._initialized = True

    def correct(self, bbox_xyxy: BBoxXYXY) -> None:
        """
        Correct the search state using a verified detector box.

        The original appearance template is retained. This avoids replacing
        the template every verification cycle while still preventing position
        drift from accumulating.
        """
        if not self._initialized:
            raise RuntimeError("OSTrack has not been initialized.")
        self._tracker.state = self._xyxy_to_xywh(bbox_xyxy)

    def track(self, frame_bgr: np.ndarray) -> Optional[BBoxXYXY]:
        if not self._initialized:
            raise RuntimeError("OSTrack has not been initialized.")

        self._validate_frame(frame_bgr)
        frame_height, frame_width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        output = self._tracker.track(frame_rgb)
        return self._xywh_to_clipped_xyxy(
            output.get("target_bbox"),
            frame_width,
            frame_height,
        )

    def reset(self) -> None:
        self._initialized = False
        if hasattr(self._tracker, "state"):
            self._tracker.state = None
        if hasattr(self._tracker, "frame_id"):
            self._tracker.frame_id = 0


class RobotDetectorTracker:
    """Synchronous detector acquisition + OSTrack short-term tracking."""

    def __init__(
        self,
        detector,
        ostrack: OSTrackAdapter,
        class_names: Sequence[str] = ("robot",),
        verification_interval_seconds: float = 0.25,
        verification_iou_threshold: float = 0.10,
        acquisition_confirmations: int = 2,
        acquisition_iou_threshold: float = 0.30,
        # This now counts consecutive detector-verification attempts that miss,
        # not consecutive video frames.
        max_verification_misses: int = 3,
        replacement_confirmations: int = 2,
        replacement_iou_threshold: float = 0.30,
        min_detection_score: float = 0.05,
        acquisition_min_score: float = 0.40,
        # A detector result must meet this score before it can independently
        # verify an OSTrack result. Low-confidence detections may still be
        # returned by the detector for searching, but cannot reset the lost
        # counter or be exposed as a verified deployment output.
        verification_min_score: float = 0.40,
        minimum_association_score: float = 0.35,
        max_normalized_center_distance: float = 0.75,
        center_distance_sigma: float = 0.75,
        max_area_ratio: float = 3.0,
        association_iou_weight: float = 0.55,
        association_center_weight: float = 0.30,
        association_confidence_weight: float = 0.15,
        crop_redetection_scale: float = 2.5,
        crop_redetection_min_size: int = 192,
        max_unverified_seconds: float = 1.0,
        tiled_search_interval_frames: int = 8,
        tile_rows: int = 2,
        tile_columns: int = 2,
        tile_overlap_fraction: float = 0.20,
        detection_nms_iou_threshold: float = 0.60,
        bbox_padding_fraction: float = 0.05,
        min_box_size_pixels: int = 12,
    ) -> None:
        if detector is None:
            raise ValueError("detector cannot be None.")
        if ostrack is None:
            raise ValueError("ostrack cannot be None.")
        if not class_names:
            raise ValueError("class_names must contain at least one label.")
        if verification_interval_seconds < 0.0:
            raise ValueError(
                "verification_interval_seconds cannot be negative."
            )

        for name, value in (
            ("verification_iou_threshold", verification_iou_threshold),
            ("acquisition_iou_threshold", acquisition_iou_threshold),
            ("replacement_iou_threshold", replacement_iou_threshold),
            ("min_detection_score", min_detection_score),
            ("acquisition_min_score", acquisition_min_score),
            ("verification_min_score", verification_min_score),
            ("minimum_association_score", minimum_association_score),
            ("tile_overlap_fraction", tile_overlap_fraction),
            ("detection_nms_iou_threshold", detection_nms_iou_threshold),
            ("bbox_padding_fraction", bbox_padding_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        for name, value in (
            ("acquisition_confirmations", acquisition_confirmations),
            ("max_verification_misses", max_verification_misses),
            ("replacement_confirmations", replacement_confirmations),
            ("crop_redetection_min_size", crop_redetection_min_size),
            ("tile_rows", tile_rows),
            ("tile_columns", tile_columns),
            ("min_box_size_pixels", min_box_size_pixels),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1.")

        if tiled_search_interval_frames < 0:
            raise ValueError(
                "tiled_search_interval_frames cannot be negative."
            )
        if max_unverified_seconds <= 0.0:
            raise ValueError("max_unverified_seconds must be positive.")
        if crop_redetection_scale <= 1.0:
            raise ValueError("crop_redetection_scale must exceed 1.0.")
        if max_normalized_center_distance <= 0.0:
            raise ValueError(
                "max_normalized_center_distance must be positive."
            )
        if center_distance_sigma <= 0.0:
            raise ValueError("center_distance_sigma must be positive.")
        if max_area_ratio < 1.0:
            raise ValueError("max_area_ratio must be at least 1.0.")

        association_weight_sum = (
            association_iou_weight
            + association_center_weight
            + association_confidence_weight
        )
        if association_weight_sum <= 0.0:
            raise ValueError(
                "Association weights must sum to a positive value."
            )

        self.detector = detector
        self.ostrack = ostrack
        self.class_names = tuple(class_names)
        self.verification_interval_seconds = float(
            verification_interval_seconds
        )
        self.verification_iou_threshold = float(
            verification_iou_threshold
        )
        self.acquisition_confirmations = int(acquisition_confirmations)
        self.acquisition_iou_threshold = float(
            acquisition_iou_threshold
        )
        self.max_verification_misses = int(max_verification_misses)
        self.replacement_confirmations = int(replacement_confirmations)
        self.replacement_iou_threshold = float(
            replacement_iou_threshold
        )
        self.min_detection_score = float(min_detection_score)
        self.acquisition_min_score = float(acquisition_min_score)
        self.verification_min_score = float(verification_min_score)
        self.minimum_association_score = float(
            minimum_association_score
        )
        self.max_normalized_center_distance = float(
            max_normalized_center_distance
        )
        self.center_distance_sigma = float(center_distance_sigma)
        self.max_area_ratio = float(max_area_ratio)

        self.association_iou_weight = (
            float(association_iou_weight) / association_weight_sum
        )
        self.association_center_weight = (
            float(association_center_weight) / association_weight_sum
        )
        self.association_confidence_weight = (
            float(association_confidence_weight) / association_weight_sum
        )

        self.crop_redetection_scale = float(crop_redetection_scale)
        self.crop_redetection_min_size = int(crop_redetection_min_size)
        self.max_unverified_seconds = float(max_unverified_seconds)
        self.tiled_search_interval_frames = int(
            tiled_search_interval_frames
        )
        self.tile_rows = int(tile_rows)
        self.tile_columns = int(tile_columns)
        self.tile_overlap_fraction = float(tile_overlap_fraction)
        self.detection_nms_iou_threshold = float(
            detection_nms_iou_threshold
        )
        self.bbox_padding_fraction = float(bbox_padding_fraction)
        self.min_box_size_pixels = int(min_box_size_pixels)

        self._track_active = False
        self._last_bbox_xyxy: Optional[BBoxXYXY] = None
        self._last_verified_timestamp: Optional[float] = None
        self._last_verification_attempt_timestamp: Optional[float] = None

        self._next_frame_id = 0
        self._search_frame_count = 0

        self._acquisition_candidate: Optional[_DetectionCandidate] = None
        self._acquisition_candidate_count = 0

        self._replacement_candidate: Optional[_DetectionCandidate] = None
        self._replacement_candidate_count = 0

        self._verification_misses = 0
        self._frames_since_verification = 0

    @staticmethod
    def _validate_frame(frame_bgr: np.ndarray) -> None:
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a NumPy array.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                "frame_bgr must have shape (height, width, 3)."
            )
        if frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must use dtype uint8.")

    @staticmethod
    def _iou(
        box_a: Optional[BBoxXYXY],
        box_b: Optional[BBoxXYXY],
    ) -> float:
        if box_a is None or box_b is None:
            return 0.0

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_width = max(0, ix2 - ix1)
        intersection_height = max(0, iy2 - iy1)
        intersection_area = intersection_width * intersection_height

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = area_a + area_b - intersection_area

        if union_area <= 0:
            return 0.0
        return intersection_area / union_area

    @staticmethod
    def _box_area(box: BBoxXYXY) -> float:
        x1, y1, x2, y2 = box
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    @staticmethod
    def _box_center(box: BBoxXYXY) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @staticmethod
    def _box_diagonal(box: BBoxXYXY) -> float:
        x1, y1, x2, y2 = box
        return math.hypot(max(1, x2 - x1), max(1, y2 - y1))

    def _clip_and_pad_box(
        self,
        box,
        frame_width: int,
        frame_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> Optional[BBoxXYXY]:
        if box is None or len(box) != 4:
            return None

        x1, y1, x2, y2 = (float(value) for value in box)
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y

        if not np.all(np.isfinite([x1, y1, x2, y2])):
            return None

        width = x2 - x1
        height = y2 - y1
        if (
            width < self.min_box_size_pixels
            or height < self.min_box_size_pixels
        ):
            return None

        pad_x = width * self.bbox_padding_fraction
        pad_y = height * self.bbox_padding_fraction

        x1 = int(round(max(0.0, x1 - pad_x)))
        y1 = int(round(max(0.0, y1 - pad_y)))
        x2 = int(round(min(float(frame_width), x2 + pad_x)))
        y2 = int(round(min(float(frame_height), y2 + pad_y)))

        if (
            x2 - x1 < self.min_box_size_pixels
            or y2 - y1 < self.min_box_size_pixels
        ):
            return None
        return x1, y1, x2, y2

    def _run_detector(
        self,
        image_bgr: np.ndarray,
        full_frame_width: int,
        full_frame_height: int,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
        source: str,
    ) -> List[_DetectionCandidate]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw_detections = self.detector.predict(
            image_rgb,
            list(self.class_names),
        )

        candidates: List[_DetectionCandidate] = []
        for detection in raw_detections or []:
            try:
                score = float(detection["score"])
                raw_box = detection["box_xyxy"]
                label = str(detection.get("label", "robot"))
            except (KeyError, TypeError, ValueError):
                continue

            if score < self.min_detection_score:
                continue

            box = self._clip_and_pad_box(
                raw_box,
                full_frame_width,
                full_frame_height,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            if box is None:
                continue

            candidates.append(
                _DetectionCandidate(
                    bbox_xyxy=box,
                    score=score,
                    label=label,
                    source=source,
                )
            )
        return candidates

    def _detect_full_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> List[_DetectionCandidate]:
        height, width = frame_bgr.shape[:2]
        return self._run_detector(
            frame_bgr,
            width,
            height,
            source="full",
        )

    def _expand_box(
        self,
        box: BBoxXYXY,
        frame_width: int,
        frame_height: int,
        scale: float,
        minimum_size: int,
    ) -> Optional[BBoxXYXY]:
        x1, y1, x2, y2 = box
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        width = max(float(minimum_size), (x2 - x1) * scale)
        height = max(float(minimum_size), (y2 - y1) * scale)

        crop_x1 = int(round(max(0.0, center_x - width * 0.5)))
        crop_y1 = int(round(max(0.0, center_y - height * 0.5)))
        crop_x2 = int(round(min(float(frame_width), center_x + width * 0.5)))
        crop_y2 = int(round(min(float(frame_height), center_y + height * 0.5)))

        current_width = crop_x2 - crop_x1
        current_height = crop_y2 - crop_y1
        desired_width = min(frame_width, int(round(width)))
        desired_height = min(frame_height, int(round(height)))

        if current_width < desired_width:
            missing = desired_width - current_width
            shift_left = min(crop_x1, missing)
            crop_x1 -= shift_left
            crop_x2 = min(
                frame_width,
                crop_x2 + (missing - shift_left),
            )

        if current_height < desired_height:
            missing = desired_height - current_height
            shift_up = min(crop_y1, missing)
            crop_y1 -= shift_up
            crop_y2 = min(
                frame_height,
                crop_y2 + (missing - shift_up),
            )

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        return crop_x1, crop_y1, crop_x2, crop_y2

    def _detect_tracked_crop(
        self,
        frame_bgr: np.ndarray,
        tracked_box: BBoxXYXY,
    ) -> List[_DetectionCandidate]:
        frame_height, frame_width = frame_bgr.shape[:2]
        crop_box = self._expand_box(
            tracked_box,
            frame_width,
            frame_height,
            scale=self.crop_redetection_scale,
            minimum_size=self.crop_redetection_min_size,
        )
        if crop_box is None:
            return []

        x1, y1, x2, y2 = crop_box
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return []

        return self._run_detector(
            crop_bgr,
            frame_width,
            frame_height,
            offset_x=x1,
            offset_y=y1,
            source="crop",
        )

    @staticmethod
    def _axis_tiles(
        length: int,
        count: int,
        overlap_fraction: float,
    ) -> List[Tuple[int, int]]:
        if count <= 1:
            return [(0, length)]

        nominal = float(length) / float(count)
        overlap = nominal * overlap_fraction
        ranges: List[Tuple[int, int]] = []

        for index in range(count):
            start = index * nominal
            end = (index + 1) * nominal
            if index > 0:
                start -= overlap * 0.5
            if index < count - 1:
                end += overlap * 0.5

            start_int = int(round(max(0.0, start)))
            end_int = int(round(min(float(length), end)))
            if end_int > start_int:
                ranges.append((start_int, end_int))
        return ranges

    def _detect_tiled(
        self,
        frame_bgr: np.ndarray,
    ) -> List[_DetectionCandidate]:
        frame_height, frame_width = frame_bgr.shape[:2]
        x_ranges = self._axis_tiles(
            frame_width,
            self.tile_columns,
            self.tile_overlap_fraction,
        )
        y_ranges = self._axis_tiles(
            frame_height,
            self.tile_rows,
            self.tile_overlap_fraction,
        )

        candidates: List[_DetectionCandidate] = []
        for y1, y2 in y_ranges:
            for x1, x2 in x_ranges:
                tile_bgr = frame_bgr[y1:y2, x1:x2]
                if tile_bgr.size == 0:
                    continue
                candidates.extend(
                    self._run_detector(
                        tile_bgr,
                        frame_width,
                        frame_height,
                        offset_x=x1,
                        offset_y=y1,
                        source="tile",
                    )
                )
        return candidates

    def _nms_candidates(
        self,
        candidates: Sequence[_DetectionCandidate],
    ) -> List[_DetectionCandidate]:
        ordered = sorted(
            candidates,
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        kept: List[_DetectionCandidate] = []

        for candidate in ordered:
            if all(
                self._iou(candidate.bbox_xyxy, existing.bbox_xyxy)
                < self.detection_nms_iou_threshold
                for existing in kept
            ):
                kept.append(candidate)
        return kept

    def _clear_acquisition_candidate(self) -> None:
        self._acquisition_candidate = None
        self._acquisition_candidate_count = 0

    def _clear_replacement_candidate(self) -> None:
        self._replacement_candidate = None
        self._replacement_candidate_count = 0

    def _clear_tracking_only(self) -> None:
        self._track_active = False
        self.ostrack.reset()
        self._last_bbox_xyxy = None
        self._last_verified_timestamp = None
        self._last_verification_attempt_timestamp = None
        self._verification_misses = 0
        self._frames_since_verification = 0
        self._clear_replacement_candidate()

    def reset(self) -> None:
        """Forget the robot and all pending detection candidates."""
        self._clear_tracking_only()
        self._clear_acquisition_candidate()
        self._search_frame_count = 0

    def _choose_acquisition_candidate(
        self,
        detections: Sequence[_DetectionCandidate],
    ) -> Optional[_DetectionCandidate]:
        eligible = [
            detection
            for detection in detections
            if detection.score >= self.acquisition_min_score
        ]
        if not eligible:
            return None

        if self._acquisition_candidate is not None:
            consistent = [
                detection
                for detection in eligible
                if self._iou(
                    self._acquisition_candidate.bbox_xyxy,
                    detection.bbox_xyxy,
                )
                >= self.acquisition_iou_threshold
            ]
            if consistent:
                return max(
                    consistent,
                    key=lambda detection: detection.score,
                )

        return max(eligible, key=lambda detection: detection.score)

    def _update_acquisition_candidate(
        self,
        candidate: Optional[_DetectionCandidate],
    ) -> bool:
        if candidate is None:
            self._clear_acquisition_candidate()
            return False

        if (
            self._acquisition_candidate is not None
            and self._iou(
                self._acquisition_candidate.bbox_xyxy,
                candidate.bbox_xyxy,
            )
            >= self.acquisition_iou_threshold
        ):
            self._acquisition_candidate_count += 1
        else:
            self._acquisition_candidate_count = 1

        self._acquisition_candidate = candidate
        return (
            self._acquisition_candidate_count
            >= self.acquisition_confirmations
        )

    def _choose_replacement_candidate(
        self,
        detections: Sequence[_DetectionCandidate],
    ) -> Optional[_DetectionCandidate]:
        eligible = [
            detection
            for detection in detections
            if detection.score >= self.acquisition_min_score
        ]
        if not eligible:
            return None

        if self._replacement_candidate is not None:
            consistent = [
                detection
                for detection in eligible
                if self._iou(
                    self._replacement_candidate.bbox_xyxy,
                    detection.bbox_xyxy,
                )
                >= self.replacement_iou_threshold
            ]
            if consistent:
                return max(
                    consistent,
                    key=lambda detection: detection.score,
                )

        return max(eligible, key=lambda detection: detection.score)

    def _update_replacement_candidate(
        self,
        candidate: Optional[_DetectionCandidate],
    ) -> bool:
        if candidate is None:
            self._clear_replacement_candidate()
            return False

        if (
            self._replacement_candidate is not None
            and self._iou(
                self._replacement_candidate.bbox_xyxy,
                candidate.bbox_xyxy,
            )
            >= self.replacement_iou_threshold
        ):
            self._replacement_candidate_count += 1
        else:
            self._replacement_candidate_count = 1

        self._replacement_candidate = candidate
        return (
            self._replacement_candidate_count
            >= self.replacement_confirmations
        )

    def _start_track(
        self,
        frame_bgr: np.ndarray,
        candidate: _DetectionCandidate,
        frame_timestamp: float,
    ) -> None:
        self.ostrack.initialize(frame_bgr, candidate.bbox_xyxy)
        self._track_active = True
        self._last_bbox_xyxy = candidate.bbox_xyxy
        self._last_verified_timestamp = frame_timestamp
        self._last_verification_attempt_timestamp = frame_timestamp
        self._verification_misses = 0
        self._frames_since_verification = 0
        self._search_frame_count = 0
        self._clear_replacement_candidate()
        self._clear_acquisition_candidate()

    def _association_metrics(
        self,
        tracked_box: BBoxXYXY,
        candidate: _DetectionCandidate,
    ) -> Tuple[bool, float]:
        # Verification must be based on strong semantic evidence. The detector
        # is intentionally configured to emit very low-confidence candidates
        # for search/recovery, but those candidates must never certify a track.
        if candidate.score < self.verification_min_score:
            return False, 0.0

        candidate_box = candidate.bbox_xyxy
        overlap = self._iou(tracked_box, candidate_box)

        tracked_center = self._box_center(tracked_box)
        candidate_center = self._box_center(candidate_box)
        center_distance = math.hypot(
            tracked_center[0] - candidate_center[0],
            tracked_center[1] - candidate_center[1],
        )
        normalized_center_distance = (
            center_distance / max(1.0, self._box_diagonal(tracked_box))
        )

        tracked_area = max(1.0, self._box_area(tracked_box))
        candidate_area = max(1.0, self._box_area(candidate_box))
        area_ratio = max(
            tracked_area / candidate_area,
            candidate_area / tracked_area,
        )

        # A verification box must actually overlap OSTrack's box. Center
        # proximity alone is not sufficient: an empty-scene false positive
        # near a drifted tracker box must not count as independent evidence.
        passes_spatial_gate = (
            overlap >= self.verification_iou_threshold
            and normalized_center_distance
            <= self.max_normalized_center_distance
        )
        if not passes_spatial_gate or area_ratio > self.max_area_ratio:
            return False, 0.0

        center_similarity = math.exp(
            -0.5
            * (
                normalized_center_distance
                / self.center_distance_sigma
            )
            ** 2
        )
        confidence = float(np.clip(candidate.score, 0.0, 1.0))
        association_score = (
            self.association_iou_weight * overlap
            + self.association_center_weight * center_similarity
            + self.association_confidence_weight * confidence
        )

        return (
            association_score >= self.minimum_association_score,
            association_score,
        )

    def _associate_detection(
        self,
        tracked_box: BBoxXYXY,
        detections: Sequence[_DetectionCandidate],
    ) -> Tuple[Optional[_DetectionCandidate], Optional[float]]:
        best_candidate: Optional[_DetectionCandidate] = None
        best_score: Optional[float] = None

        for candidate in detections:
            valid, score = self._association_metrics(
                tracked_box,
                candidate,
            )
            if not valid:
                continue
            if best_score is None or score > best_score:
                best_candidate = candidate
                best_score = score

        return best_candidate, best_score

    def _verification_due(self, frame_timestamp: float) -> bool:
        if self.verification_interval_seconds == 0.0:
            return True
        if self._last_verification_attempt_timestamp is None:
            return True

        elapsed = (
            frame_timestamp - self._last_verification_attempt_timestamp
        )
        if not np.isfinite(elapsed) or elapsed < 0.0:
            return True
        return elapsed >= self.verification_interval_seconds

    def _prediction_age_seconds(self, frame_timestamp: float) -> float:
        if self._last_verified_timestamp is None:
            return float("inf")
        return max(0.0, frame_timestamp - self._last_verified_timestamp)

    def _should_run_tiled_search(self) -> bool:
        if self.tiled_search_interval_frames == 0:
            return False

        if self._acquisition_candidate is not None:
            return True

        return (
            self._search_frame_count
            % self.tiled_search_interval_frames
            == 0
        )

    def _make_result(
        self,
        *,
        state: TrackingState,
        bbox: Optional[BBoxXYXY],
        frame_id: int,
        frame_timestamp: float,
        start_monotonic: float,
        source: str,
        detector_score: Optional[float] = None,
        detector_label: Optional[str] = None,
        tracker_verified: bool = False,
        prediction_age_frames: Optional[int] = None,
        prediction_age_seconds: Optional[float] = None,
        association_score: Optional[float] = None,
        used_crop_redetection: bool = False,
        used_tiled_search: bool = False,
    ) -> RobotTrackResult:
        if prediction_age_frames is None:
            prediction_age_frames = self._frames_since_verification
        if prediction_age_seconds is None:
            prediction_age_seconds = (
                self._prediction_age_seconds(frame_timestamp)
                if self._track_active
                else 0.0
            )

        return RobotTrackResult(
            state=state,
            bbox_xyxy=bbox,
            frame_id=frame_id,
            frame_timestamp=frame_timestamp,
            completed_timestamp=time.time(),
            processing_time_ms=(
                time.monotonic() - start_monotonic
            )
            * 1000.0,
            source=source,
            detector_score=detector_score,
            detector_label=detector_label,
            tracker_verified=tracker_verified,
            acquisition_candidate_count=(
                self._acquisition_candidate_count
            ),
            replacement_candidate_count=(
                self._replacement_candidate_count
            ),
            verification_misses=self._verification_misses,
            prediction_age_frames=prediction_age_frames,
            prediction_age_seconds=prediction_age_seconds,
            association_score=association_score,
            used_crop_redetection=used_crop_redetection,
            used_tiled_search=used_tiled_search,
        )

    def _process_searching_frame(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        frame_timestamp: float,
        start_monotonic: float,
    ) -> RobotTrackResult:
        self._search_frame_count += 1

        detections = self._detect_full_frame(frame_bgr)
        used_tiled_search = False

        if self._should_run_tiled_search():
            detections.extend(self._detect_tiled(frame_bgr))
            detections = self._nms_candidates(detections)
            used_tiled_search = True

        acquisition_candidate = self._choose_acquisition_candidate(
            detections
        )
        acquisition_confirmed = self._update_acquisition_candidate(
            acquisition_candidate
        )

        if acquisition_confirmed:
            assert acquisition_candidate is not None
            self._start_track(
                frame_bgr,
                acquisition_candidate,
                frame_timestamp,
            )
            return self._make_result(
                state=TrackingState.DETECTED,
                bbox=acquisition_candidate.bbox_xyxy,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source=(
                    "detector_acquisition_tiled_confirmed"
                    if acquisition_candidate.source == "tile"
                    else "detector_acquisition_confirmed"
                ),
                detector_score=acquisition_candidate.score,
                detector_label=acquisition_candidate.label,
                tracker_verified=True,
                prediction_age_frames=0,
                prediction_age_seconds=0.0,
                used_tiled_search=used_tiled_search,
            )

        diagnostic_candidate = acquisition_candidate
        if diagnostic_candidate is None and detections:
            diagnostic_candidate = max(
                detections,
                key=lambda detection: detection.score,
            )

        return self._make_result(
            state=TrackingState.SEARCHING,
            bbox=None,
            frame_id=frame_id,
            frame_timestamp=frame_timestamp,
            start_monotonic=start_monotonic,
            source=(
                "detector_acquisition_candidate"
                if acquisition_candidate is not None
                else "detector_search_no_acquisition"
            ),
            detector_score=(
                diagnostic_candidate.score
                if diagnostic_candidate is not None
                else None
            ),
            detector_label=(
                diagnostic_candidate.label
                if diagnostic_candidate is not None
                else None
            ),
            tracker_verified=False,
            used_tiled_search=used_tiled_search,
        )

    def process_frame_detector_only(
        self,
        frame_bgr: np.ndarray,
        frame_id: Optional[int] = None,
        frame_timestamp: Optional[float] = None,
    ) -> RobotTrackResult:
        """Run full-frame object detection without temporal tracking."""
        self._validate_frame(frame_bgr)

        if frame_id is None:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
        if frame_timestamp is None:
            frame_timestamp = time.time()

        start_monotonic = time.monotonic()
        detections = self._detect_full_frame(frame_bgr)

        if not detections:
            return self._make_result(
                state=TrackingState.SEARCHING,
                bbox=None,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source="detector_only_no_detection",
            )

        best = max(detections, key=lambda detection: detection.score)
        return self._make_result(
            state=TrackingState.DETECTED,
            bbox=best.bbox_xyxy,
            frame_id=frame_id,
            frame_timestamp=frame_timestamp,
            start_monotonic=start_monotonic,
            source="detector_only",
            detector_score=best.score,
            detector_label=best.label,
            tracker_verified=True,
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_id: Optional[int] = None,
        frame_timestamp: Optional[float] = None,
    ) -> RobotTrackResult:
        """Process one OpenCV BGR frame."""
        self._validate_frame(frame_bgr)

        if frame_id is None:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
        if frame_timestamp is None:
            frame_timestamp = time.time()

        start_monotonic = time.monotonic()

        if not self._track_active:
            return self._process_searching_frame(
                frame_bgr,
                frame_id,
                frame_timestamp,
                start_monotonic,
            )

        tracked_box = self.ostrack.track(frame_bgr)
        if tracked_box is None:
            self._clear_tracking_only()
            return self._process_searching_frame(
                frame_bgr,
                frame_id,
                frame_timestamp,
                start_monotonic,
            )

        self._last_bbox_xyxy = tracked_box
        self._frames_since_verification += 1

        if not self._verification_due(frame_timestamp):
            return self._make_result(
                state=TrackingState.TRACKING,
                bbox=tracked_box,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source="ostrack",
                tracker_verified=False,
            )

        self._last_verification_attempt_timestamp = frame_timestamp

        full_detections = self._nms_candidates(
            self._detect_full_frame(frame_bgr)
        )
        matched_detection, association_score = self._associate_detection(
            tracked_box,
            full_detections,
        )

        if matched_detection is not None:
            self.ostrack.correct(matched_detection.bbox_xyxy)
            self._last_bbox_xyxy = matched_detection.bbox_xyxy
            self._last_verified_timestamp = frame_timestamp
            self._verification_misses = 0
            self._frames_since_verification = 0
            self._clear_replacement_candidate()

            confidence_kind = (
                "high"
                if matched_detection.score >= self.acquisition_min_score
                else "low"
            )
            return self._make_result(
                state=TrackingState.TRACKING,
                bbox=matched_detection.bbox_xyxy,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source=(
                    f"ostrack_detector_full_{confidence_kind}_confidence"
                ),
                detector_score=matched_detection.score,
                detector_label=matched_detection.label,
                tracker_verified=True,
                prediction_age_frames=0,
                prediction_age_seconds=0.0,
                association_score=association_score,
            )

        # Full-frame detection can miss a small target because the detector
        # resized the entire image. Retry on a crop around OSTrack's result.
        crop_detections = self._nms_candidates(
            self._detect_tracked_crop(frame_bgr, tracked_box)
        )
        matched_detection, association_score = self._associate_detection(
            tracked_box,
            crop_detections,
        )

        if matched_detection is not None:
            self.ostrack.correct(matched_detection.bbox_xyxy)
            self._last_bbox_xyxy = matched_detection.bbox_xyxy
            self._last_verified_timestamp = frame_timestamp
            self._verification_misses = 0
            self._frames_since_verification = 0
            self._clear_replacement_candidate()

            confidence_kind = (
                "high"
                if matched_detection.score >= self.acquisition_min_score
                else "low"
            )
            return self._make_result(
                state=TrackingState.TRACKING,
                bbox=matched_detection.bbox_xyxy,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source=(
                    f"ostrack_detector_crop_{confidence_kind}_confidence"
                ),
                detector_score=matched_detection.score,
                detector_label=matched_detection.label,
                tracker_verified=True,
                prediction_age_frames=0,
                prediction_age_seconds=0.0,
                association_score=association_score,
                used_crop_redetection=True,
            )

        # A strong detection elsewhere must persist before replacing the
        # current target. This avoids immediately jumping to a false positive.
        replacement_candidate = self._choose_replacement_candidate(
            full_detections
        )
        replacement_confirmed = self._update_replacement_candidate(
            replacement_candidate
        )

        if replacement_confirmed:
            assert replacement_candidate is not None
            self._start_track(
                frame_bgr,
                replacement_candidate,
                frame_timestamp,
            )
            return self._make_result(
                state=TrackingState.DETECTED,
                bbox=replacement_candidate.bbox_xyxy,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source="detector_replacement_confirmed_ostrack_reinitialized",
                detector_score=replacement_candidate.score,
                detector_label=replacement_candidate.label,
                tracker_verified=True,
                prediction_age_frames=0,
                prediction_age_seconds=0.0,
                used_crop_redetection=True,
            )

        self._verification_misses += 1
        prediction_age_seconds = self._prediction_age_seconds(
            frame_timestamp
        )

        may_continue = (
            self._verification_misses <= self.max_verification_misses
            and prediction_age_seconds <= self.max_unverified_seconds
        )
        if may_continue:
            return self._make_result(
                state=TrackingState.TRACKING,
                bbox=tracked_box,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source="ostrack_unverified_after_detector_miss",
                detector_score=(
                    replacement_candidate.score
                    if replacement_candidate is not None
                    else None
                ),
                detector_label=(
                    replacement_candidate.label
                    if replacement_candidate is not None
                    else None
                ),
                tracker_verified=False,
                prediction_age_frames=self._frames_since_verification,
                prediction_age_seconds=prediction_age_seconds,
                used_crop_redetection=True,
            )

        # OSTrack has lacked semantic detector support for too long. Stop
        # returning its box and resume global detector search.
        expired_candidate = replacement_candidate
        self._clear_tracking_only()
        if expired_candidate is not None:
            self._update_acquisition_candidate(expired_candidate)

        return self._make_result(
            state=TrackingState.SEARCHING,
            bbox=None,
            frame_id=frame_id,
            frame_timestamp=frame_timestamp,
            start_monotonic=start_monotonic,
            source=(
                "ostrack_expired_new_candidate"
                if expired_candidate is not None
                else "ostrack_expired_detector_absent"
            ),
            detector_score=(
                expired_candidate.score
                if expired_candidate is not None
                else None
            ),
            detector_label=(
                expired_candidate.label
                if expired_candidate is not None
                else None
            ),
            tracker_verified=False,
            prediction_age_frames=0,
            prediction_age_seconds=0.0,
            used_crop_redetection=True,
        )


class LatestFrameRobotTracker:
    """Latest-frame-only background wrapper for live Quest frames."""

    def __init__(
        self,
        tracker: RobotDetectorTracker,
        result_callback: Optional[
            Callable[[RobotTrackResult, np.ndarray], None]
        ] = None,
        copy_submitted_frames: bool = True,
    ) -> None:
        self.tracker = tracker
        self.result_callback = result_callback
        self.copy_submitted_frames = bool(copy_submitted_frames)

        self._condition = threading.Condition()
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_frame_id: Optional[int] = None
        self._pending_timestamp: Optional[float] = None

        self._latest_result: Optional[RobotTrackResult] = None
        self._latest_processed_frame: Optional[np.ndarray] = None

        self._next_frame_id = 0
        self._submitted_frames = 0
        self._processed_frames = 0
        self._dropped_frames = 0

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    @property
    def submitted_frames(self) -> int:
        return self._submitted_frames

    @property
    def processed_frames(self) -> int:
        return self._processed_frames

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def start(self) -> None:
        with self._condition:
            if self._running:
                return

            self._running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="robot-tracker-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self, wait: bool = True) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()

        if (
            wait
            and self._worker_thread is not None
            and self._worker_thread.is_alive()
        ):
            self._worker_thread.join()

    def submit_frame(
        self,
        frame_bgr: np.ndarray,
        frame_id: Optional[int] = None,
        frame_timestamp: Optional[float] = None,
    ) -> int:
        RobotDetectorTracker._validate_frame(frame_bgr)

        if frame_id is None:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
        if frame_timestamp is None:
            frame_timestamp = time.time()

        stored_frame = (
            frame_bgr.copy()
            if self.copy_submitted_frames
            else frame_bgr
        )

        with self._condition:
            if not self._running:
                raise RuntimeError(
                    "LatestFrameRobotTracker is not running. "
                    "Call start() before submit_frame()."
                )

            self._submitted_frames += 1
            if self._pending_frame is not None:
                self._dropped_frames += 1

            self._pending_frame = stored_frame
            self._pending_frame_id = int(frame_id)
            self._pending_timestamp = float(frame_timestamp)
            self._condition.notify()

        return int(frame_id)

    def get_latest_result(self) -> Optional[RobotTrackResult]:
        with self._condition:
            return self._latest_result

    def get_latest_processed_frame(
        self,
        copy_frame: bool = True,
    ) -> Optional[np.ndarray]:
        with self._condition:
            if self._latest_processed_frame is None:
                return None
            return (
                self._latest_processed_frame.copy()
                if copy_frame
                else self._latest_processed_frame
            )

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._running and self._pending_frame is None:
                    self._condition.wait()

                if not self._running and self._pending_frame is None:
                    return

                frame = self._pending_frame
                frame_id = self._pending_frame_id
                frame_timestamp = self._pending_timestamp

                self._pending_frame = None
                self._pending_frame_id = None
                self._pending_timestamp = None

            if frame is None or frame_id is None:
                continue

            result = self.tracker.process_frame(
                frame,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
            )

            with self._condition:
                self._latest_result = result
                self._latest_processed_frame = frame
                self._processed_frames += 1

            if self.result_callback is not None:
                try:
                    self.result_callback(result, frame)
                except Exception as exc:
                    print("Robot tracker callback failed:", exc)


def draw_result(
    frame_bgr: np.ndarray,
    result: RobotTrackResult,
) -> np.ndarray:
    output = frame_bgr.copy()

    if result.bbox_xyxy is not None:
        x1, y1, x2, y2 = result.bbox_xyxy
        # Green: detector-verified on this frame.
        # Cyan: OSTrack result awaiting the next semantic verification.
        color = (
            (0, 255, 0)
            if result.tracker_verified
            else (255, 255, 0)
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    status = (
        f"{result.state.value} | {result.source} | "
        f"{result.processing_time_ms:.1f} ms"
    )
    cv2.putText(
        output,
        status,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    counters = (
        f"acq={result.acquisition_candidate_count} "
        f"verify_miss={result.verification_misses} "
        f"repl={result.replacement_candidate_count} "
        f"age={result.prediction_age_seconds:.2f}s"
    )
    cv2.putText(
        output,
        counters,
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if result.detector_score is not None:
        detector_text = f"Detector: {result.detector_score:.3f}"
        if result.detector_label:
            detector_text += f" ({result.detector_label})"
        if result.association_score is not None:
            detector_text += f" assoc={result.association_score:.3f}"

        cv2.putText(
            output,
            detector_text,
            (12, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    mode_text = (
        f"crop={int(result.used_crop_redetection)} "
        f"tiles={int(result.used_tiled_search)} "
        f"detector_verified={int(result.tracker_verified)}"
    )
    cv2.putText(
        output,
        mode_text,
        (12, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def process_video(
    input_video_path: str,
    output_video_path: str,
    detector,
    ostrack: OSTrackAdapter,
    class_names: Sequence[str] = ("robot",),
    max_frames_to_process: Optional[int] = None,
    show_preview: bool = False,
) -> None:
    """Run the same detector + OSTrack engine on an offline video."""
    tracker = RobotDetectorTracker(
        detector=detector,
        ostrack=ostrack,
        class_names=class_names,
        verification_interval_seconds=0.25,
        acquisition_confirmations=2,
        acquisition_iou_threshold=0.30,
        max_verification_misses=3,
        replacement_confirmations=2,
        replacement_iou_threshold=0.30,
        min_detection_score=0.05,
        acquisition_min_score=0.40,
        verification_min_score=0.40,
        verification_iou_threshold=0.20,
        max_unverified_seconds=1.0,
        crop_redetection_scale=2.5,
        tiled_search_interval_frames=8,
        tile_rows=2,
        tile_columns=2,
        tile_overlap_fraction=0.20,
    )

    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video: {output_video_path}"
        )

    frame_id = 0
    video_start_timestamp = time.time()

    try:
        while True:
            if (
                max_frames_to_process is not None
                and frame_id >= max_frames_to_process
            ):
                break

            success, frame = capture.read()
            if not success:
                break

            # Video time keeps verification cadence independent of how slowly
            # the offline test itself runs.
            frame_timestamp = video_start_timestamp + frame_id / fps
            result = tracker.process_frame(
                frame,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
            )
            annotated = draw_result(frame, result)

            cv2.putText(
                annotated,
                f"Frame {frame_id + 1}/{total_frames}",
                (12, 142),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)

            if show_preview:
                cv2.imshow("Robot Detector + OSTrack", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            if frame_id % 100 == 0:
                print(f"Processed {frame_id}/{total_frames} frames")

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Finished. Saved annotated video to: {output_video_path}")


def process_video_detector_only(
    input_video_path: str,
    output_video_path: str,
    detector,
    class_names: Sequence[str] = ("unitree go2",),
    max_frames_to_process: Optional[int] = None,
) -> None:
    """Retained utility for inspecting raw detector output."""
    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open input video: {input_video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video: {output_video_path}"
        )

    frame_id = 0
    try:
        while True:
            if (
                max_frames_to_process is not None
                and frame_id >= max_frames_to_process
            ):
                break

            success, frame_bgr = capture.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = detector.predict(
                frame_rgb,
                list(class_names),
            )
            annotated = frame_bgr.copy()

            for detection in detections or []:
                score = float(detection["score"])
                x1, y1, x2, y2 = [
                    int(round(value))
                    for value in detection["box_xyxy"]
                ]
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated,
                    f"{score:.3f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(annotated)
            frame_id += 1

    finally:
        capture.release()
        writer.release()

    print(f"Saved detector-only video to {output_video_path}")


# ---------------------------------------------------------------------
# Video-test settings
# ---------------------------------------------------------------------

DETECTOR_TYPE = "yolo"

ROBOT_LABELS = (
    "robot dog",
    "quadruped robot",
    "four-legged robot",
)

SHOW_PREVIEW = False

# Set to None to process the entire video.
MAX_FRAMES_TO_PROCESS = 500

# This file is expected to be:
# robot_fov_estimation/src/robot_detector_tracker_ostrack.py


INPUT_VIDEO_PATH = (
    ROBOT_FOV_ROOT
    / "test_videos"
    / "dogtest.MP4"
)

OUTPUT_VIDEO_PATH = (
    ROBOT_FOV_ROOT
    / "outputs"
    / f"ostrack_tracking_{DETECTOR_TYPE}.mp4"
)

# Example layout:
# <project>/third_party/OSTrack/
# <project>/third_party/OSTrack/output/checkpoints/train/ostrack/
#     vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar


if __name__ == "__main__":
    from model_training_and_implementation.src.dino_detector import (
        DINOObjectDetector,
    )
    from robot_fov_estimation.src.go2_yolo_det_wrapper import (
        YOLOGo2Detector,
    )

    if not INPUT_VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Input video does not exist: {INPUT_VIDEO_PATH}"
        )

    OUTPUT_VIDEO_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector_type = DETECTOR_TYPE.strip().lower()

    if detector_type == "yolo":
        detector = YOLOGo2Detector(
            # Keep this low so detector verification can still use weak
            # detections. New tracks still require acquisition_min_score=0.40.
            confidence=0.05,
            iou_threshold=0.50,
            image_size=1280,
            device=0,
        )

    elif detector_type == "dino":
        detector = DINOObjectDetector(
            confidence=0.05,
        )

    else:
        raise ValueError(
            f"Unsupported DETECTOR_TYPE: {DETECTOR_TYPE!r}. "
            'Use either "yolo" or "dino".'
        )

    ostrack = OSTrackAdapter(
        ostrack_root=OSTRACK_ROOT,
        checkpoint_path=OSTRACK_CHECKPOINT_PATH,
        config_name=OSTRACK_CONFIG_NAME,
    )

    print(f"Detector: {detector_type}")
    print(f"OSTrack root: {OSTRACK_ROOT}")
    print(f"OSTrack checkpoint: {OSTRACK_CHECKPOINT_PATH}")
    print(f"Input: {INPUT_VIDEO_PATH}")
    print(f"Output: {OUTPUT_VIDEO_PATH}")

    process_video(
        input_video_path=str(INPUT_VIDEO_PATH),
        output_video_path=str(OUTPUT_VIDEO_PATH),
        detector=detector,
        ostrack=ostrack,
        class_names=ROBOT_LABELS,
        max_frames_to_process=MAX_FRAMES_TO_PROCESS,
        show_preview=SHOW_PREVIEW,
    )