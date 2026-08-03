# """
# robot_detector_tracker_realtime_robust.py

# Robot detection and tracking for both:

# 1. Real-time Quest image streams
#    - Use LatestFrameRobotTracker.
#    - Only the newest waiting frame is retained, preventing latency buildup.

# 2. Offline video testing
#    - Use process_video(...), or edit the settings at the bottom and run this
#      file directly.

# The object detector is injected and must provide:

#     detector.predict(image_rgb, class_names)

# returning a list of dictionaries shaped like:

#     {
#         "label": str,
#         "score": float,
#         "box_xyxy": [x1, y1, x2, y2],
#     }

# This version is deliberately conservative about robot presence:

# - A new robot track requires multiple consistent object detections.
# - A tracked robot is removed after multiple consecutive object verification
#   misses, even if CSRT continues returning a box.
# - A non-overlapping object box must appear consistently before replacing the
#   current CSRT track.
# """

# from __future__ import annotations

# import threading
# import time
# from dataclasses import dataclass
# from enum import Enum
# from typing import Callable, Optional, Sequence, Tuple
# from pathlib import Path

# import cv2
# import numpy as np


# BBoxXYXY = Tuple[int, int, int, int]


# class TrackingState(str, Enum):
#     SEARCHING = "searching"
#     DETECTED = "detected"
#     TRACKING = "tracking"


# @dataclass(frozen=True)
# class RobotTrackResult:
#     state: TrackingState
#     bbox_xyxy: Optional[BBoxXYXY]
#     frame_id: int
#     frame_timestamp: float
#     completed_timestamp: float
#     processing_time_ms: float
#     source: str
#     detector_score: Optional[float] = None
#     detector_label: Optional[str] = None
#     tracker_verified: bool = False
#     acquisition_candidate_count: int = 0
#     replacement_candidate_count: int = 0
#     verification_misses: int = 0

#     @property
#     def found(self) -> bool:
#         return self.bbox_xyxy is not None

#     @property
#     def end_to_end_latency_ms(self) -> float:
#         return max(
#             0.0,
#             (self.completed_timestamp - self.frame_timestamp) * 1000.0,
#         )


# class RobotDetectorTracker:
#     """Synchronous object detection and tracking."""

#     def __init__(
#         self,
#         detector,
#         class_names: Sequence[str] = ("robot",),
#         verification_interval_seconds: float = 0.4,
#         verification_iou_threshold: float = 0.40,
#         acquisition_confirmations: int = 2,
#         acquisition_iou_threshold: float = 0.40,
#         max_verification_misses: int = 2,
#         replacement_confirmations: int = 2,
#         replacement_iou_threshold: float = 0.40,
#         min_detection_score: float = 0.0,
#         bbox_padding_fraction: float = 0.05,
#         min_box_size_pixels: int = 20,
#     ) -> None:
#         if detector is None:
#             raise ValueError("detector cannot be None.")
#         if not class_names:
#             raise ValueError("class_names must contain at least one label.")
#         if verification_interval_seconds < 0.0:
#             raise ValueError(
#                 "verification_interval_seconds cannot be negative."
#             )

#         for name, value in (
#             ("verification_iou_threshold", verification_iou_threshold),
#             ("acquisition_iou_threshold", acquisition_iou_threshold),
#             ("replacement_iou_threshold", replacement_iou_threshold),
#             ("bbox_padding_fraction", bbox_padding_fraction),
#         ):
#             if not 0.0 <= value <= 1.0:
#                 raise ValueError(f"{name} must be between 0 and 1.")

#         for name, value in (
#             ("acquisition_confirmations", acquisition_confirmations),
#             ("max_verification_misses", max_verification_misses),
#             ("replacement_confirmations", replacement_confirmations),
#             ("min_box_size_pixels", min_box_size_pixels),
#         ):
#             if value < 1:
#                 raise ValueError(f"{name} must be at least 1.")

#         self.detector = detector
#         self.class_names = tuple(class_names)
#         self.verification_interval_seconds = float(
#             verification_interval_seconds
#         )
#         self.verification_iou_threshold = float(
#             verification_iou_threshold
#         )
#         self.acquisition_confirmations = int(acquisition_confirmations)
#         self.acquisition_iou_threshold = float(
#             acquisition_iou_threshold
#         )
#         self.max_verification_misses = int(max_verification_misses)
#         self.replacement_confirmations = int(replacement_confirmations)
#         self.replacement_iou_threshold = float(
#             replacement_iou_threshold
#         )
#         self.min_detection_score = float(min_detection_score)
#         self.bbox_padding_fraction = float(bbox_padding_fraction)
#         self.min_box_size_pixels = int(min_box_size_pixels)

#         self._tracker = None
#         self._last_bbox_xyxy: Optional[BBoxXYXY] = None
#         self._next_frame_id = 0
#         self._last_detector_attempt_monotonic = float("-inf")

#         self._acquisition_candidate_box: Optional[BBoxXYXY] = None
#         self._acquisition_candidate_count = 0

#         self._replacement_candidate_box: Optional[BBoxXYXY] = None
#         self._replacement_candidate_count = 0

#         self._verification_misses = 0

#     @staticmethod
#     def _validate_frame(frame_bgr: np.ndarray) -> None:
#         if not isinstance(frame_bgr, np.ndarray):
#             raise TypeError("frame_bgr must be a NumPy array.")
#         if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
#             raise ValueError(
#                 "frame_bgr must have shape (height, width, 3)."
#             )
#         if frame_bgr.dtype != np.uint8:
#             raise ValueError("frame_bgr must use dtype uint8.")

#     @staticmethod
#     def _create_csrt_tracker():
#         if hasattr(cv2, "TrackerCSRT_create"):
#             return cv2.TrackerCSRT_create()

#         if (
#             hasattr(cv2, "legacy")
#             and hasattr(cv2.legacy, "TrackerCSRT_create")
#         ):
#             return cv2.legacy.TrackerCSRT_create()

#         raise RuntimeError(
#             "OpenCV CSRT is unavailable. Install opencv-contrib-python "
#             "and remove conflicting opencv-python installations."
#         )

#     @staticmethod
#     def _xyxy_to_xywh(
#         box_xyxy: BBoxXYXY,
#     ) -> Tuple[int, int, int, int]:
#         x1, y1, x2, y2 = box_xyxy
#         return x1, y1, x2 - x1, y2 - y1

#     @staticmethod
#     def _xywh_to_xyxy(
#         box_xywh,
#         frame_width: int,
#         frame_height: int,
#     ) -> Optional[BBoxXYXY]:
#         x, y, width, height = (float(value) for value in box_xywh)

#         if not np.all(np.isfinite([x, y, width, height])):
#             return None

#         x1 = int(round(max(0.0, x)))
#         y1 = int(round(max(0.0, y)))
#         x2 = int(round(min(float(frame_width), x + width)))
#         y2 = int(round(min(float(frame_height), y + height)))

#         if x2 <= x1 or y2 <= y1:
#             return None

#         return x1, y1, x2, y2

#     def _clip_and_pad_box(
#         self,
#         box,
#         frame_width: int,
#         frame_height: int,
#     ) -> Optional[BBoxXYXY]:
#         if box is None or len(box) != 4:
#             return None

#         x1, y1, x2, y2 = (float(value) for value in box)

#         if not np.all(np.isfinite([x1, y1, x2, y2])):
#             return None

#         width = x2 - x1
#         height = y2 - y1

#         if (
#             width < self.min_box_size_pixels
#             or height < self.min_box_size_pixels
#         ):
#             return None

#         pad_x = width * self.bbox_padding_fraction
#         pad_y = height * self.bbox_padding_fraction

#         x1 = int(round(max(0.0, x1 - pad_x)))
#         y1 = int(round(max(0.0, y1 - pad_y)))
#         x2 = int(round(min(float(frame_width), x2 + pad_x)))
#         y2 = int(round(min(float(frame_height), y2 + pad_y)))

#         if (
#             x2 - x1 < self.min_box_size_pixels
#             or y2 - y1 < self.min_box_size_pixels
#         ):
#             return None

#         return x1, y1, x2, y2

#     @staticmethod
#     def _iou(
#         box_a: Optional[BBoxXYXY],
#         box_b: Optional[BBoxXYXY],
#     ) -> float:
#         if box_a is None or box_b is None:
#             return 0.0

#         ax1, ay1, ax2, ay2 = box_a
#         bx1, by1, bx2, by2 = box_b

#         ix1 = max(ax1, bx1)
#         iy1 = max(ay1, by1)
#         ix2 = min(ax2, bx2)
#         iy2 = min(ay2, by2)

#         intersection_width = max(0, ix2 - ix1)
#         intersection_height = max(0, iy2 - iy1)
#         intersection_area = intersection_width * intersection_height

#         area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
#         area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
#         union_area = area_a + area_b - intersection_area

#         if union_area <= 0:
#             return 0.0

#         return intersection_area / union_area

#     def _detect_robot(
#         self,
#         frame_bgr: np.ndarray,
#     ) -> Tuple[
#         Optional[BBoxXYXY],
#         Optional[float],
#         Optional[str],
#     ]:
#         self._last_detector_attempt_monotonic = time.monotonic()

#         frame_height, frame_width = frame_bgr.shape[:2]
#         frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

#         detections = self.detector.predict(
#             frame_rgb,
#             list(self.class_names),
#         )
        
#         if not detections:
#             return None, None, None

#         valid_detections = []

#         for detection in detections:
#             try:
#                 score = float(detection["score"])
#                 raw_box = detection["box_xyxy"]
#                 label = str(detection.get("label", "robot"))
#             except (KeyError, TypeError, ValueError):
#                 continue

#             if score < self.min_detection_score:
#                 continue

#             box = self._clip_and_pad_box(
#                 raw_box,
#                 frame_width,
#                 frame_height,
#             )

#             if box is not None:
#                 valid_detections.append((score, box, label))

#         if not valid_detections:
#             return None, None, None
        
#         score, box, label = max(
#             valid_detections,
#             key=lambda item: item[0],
#         )
#         return box, score, label

#     def _start_tracker(
#         self,
#         frame_bgr: np.ndarray,
#         bbox_xyxy: BBoxXYXY,
#     ) -> None:
#         tracker = self._create_csrt_tracker()
#         initialization_result = tracker.init(
#             frame_bgr,
#             self._xyxy_to_xywh(bbox_xyxy),
#         )

#         # Some OpenCV bindings return None on successful initialization.
#         if initialization_result is False:
#             raise RuntimeError(
#                 "OpenCV failed to initialize the CSRT tracker."
#             )

#         self._tracker = tracker
#         self._last_bbox_xyxy = bbox_xyxy
#         self._verification_misses = 0
#         self._clear_replacement_candidate()
#         self._clear_acquisition_candidate()

#     def _verification_due(self) -> bool:
#         if self.verification_interval_seconds == 0.0:
#             return False

#         return (
#             time.monotonic() - self._last_detector_attempt_monotonic
#             >= self.verification_interval_seconds
#         )

#     def _clear_acquisition_candidate(self) -> None:
#         self._acquisition_candidate_box = None
#         self._acquisition_candidate_count = 0

#     def _clear_replacement_candidate(self) -> None:
#         self._replacement_candidate_box = None
#         self._replacement_candidate_count = 0

#     def _clear_tracking_only(self) -> None:
#         self._tracker = None
#         self._last_bbox_xyxy = None
#         self._verification_misses = 0
#         self._clear_replacement_candidate()

#     def reset(self) -> None:
#         """Forget the robot and all pending detection candidates."""
#         self._clear_tracking_only()
#         self._clear_acquisition_candidate()

#     def _update_acquisition_candidate(
#         self,
#         detected_box: Optional[BBoxXYXY],
#     ) -> bool:
#         if detected_box is None:
#             self._clear_acquisition_candidate()
#             return False

#         if (
#             self._acquisition_candidate_box is not None
#             and self._iou(
#                 self._acquisition_candidate_box,
#                 detected_box,
#             )
#             >= self.acquisition_iou_threshold
#         ):
#             self._acquisition_candidate_count += 1
#         else:
#             self._acquisition_candidate_count = 1

#         self._acquisition_candidate_box = detected_box
#         return (
#             self._acquisition_candidate_count
#             >= self.acquisition_confirmations
#         )

#     def _update_replacement_candidate(
#         self,
#         detected_box: BBoxXYXY,
#     ) -> bool:
#         if (
#             self._replacement_candidate_box is not None
#             and self._iou(
#                 self._replacement_candidate_box,
#                 detected_box,
#             )
#             >= self.replacement_iou_threshold
#         ):
#             self._replacement_candidate_count += 1
#         else:
#             self._replacement_candidate_count = 1

#         self._replacement_candidate_box = detected_box
#         return (
#             self._replacement_candidate_count
#             >= self.replacement_confirmations
#         )

#     def _make_result(
#         self,
#         *,
#         state: TrackingState,
#         bbox: Optional[BBoxXYXY],
#         frame_id: int,
#         frame_timestamp: float,
#         start_monotonic: float,
#         source: str,
#         detector_score: Optional[float] = None,
#         detector_label: Optional[str] = None,
#         tracker_verified: bool = False,
#     ) -> RobotTrackResult:
#         return RobotTrackResult(
#             state=state,
#             bbox_xyxy=bbox,
#             frame_id=frame_id,
#             frame_timestamp=frame_timestamp,
#             completed_timestamp=time.time(),
#             processing_time_ms=(
#                 time.monotonic() - start_monotonic
#             )
#             * 1000.0,
#             source=source,
#             detector_score=detector_score,
#             detector_label=detector_label,
#             tracker_verified=tracker_verified,
#             acquisition_candidate_count=(
#                 self._acquisition_candidate_count
#             ),
#             replacement_candidate_count=(
#                 self._replacement_candidate_count
#             ),
#             verification_misses=self._verification_misses,
#         )

#     def _search_with_detector(
#         self,
#         frame_bgr: np.ndarray,
#         frame_id: int,
#         frame_timestamp: float,
#         start_monotonic: float,
#         source_prefix: str,
#     ) -> RobotTrackResult:
#         detected_box, score, label = self._detect_robot(frame_bgr)

#         if not self._update_acquisition_candidate(detected_box):
#             source = (
#                 f"{source_prefix}_candidate"
#                 if detected_box is not None
#                 else f"{source_prefix}_no_detection"
#             )
#             return self._make_result(
#                 state=TrackingState.SEARCHING,
#                 bbox=None,
#                 frame_id=frame_id,
#                 frame_timestamp=frame_timestamp,
#                 start_monotonic=start_monotonic,
#                 source=source,
#                 detector_score=score,
#                 detector_label=label,
#             )

#         # Confirmation was reached on this exact frame, so it is safe to
#         # initialize CSRT using this frame and this detection box.
#         assert detected_box is not None
#         self._start_tracker(frame_bgr, detected_box)

#         return self._make_result(
#             state=TrackingState.DETECTED,
#             bbox=detected_box,
#             frame_id=frame_id,
#             frame_timestamp=frame_timestamp,
#             start_monotonic=start_monotonic,
#             source=f"{source_prefix}_confirmed",
#             detector_score=score,
#             detector_label=label,
#             tracker_verified=True,
#         )

#     def process_frame_detector_only(
#     self,
#     frame_bgr: np.ndarray,
#     frame_id: Optional[int] = None,
#     frame_timestamp: Optional[float] = None,
# ) -> RobotTrackResult:
#         """
#         Run object detection on one frame without temporal tracking.

#         This method does not:
#         - initialize or update CSRT;
#         - require acquisition confirmations;
#         - use replacement candidates;
#         - use verification misses;
#         - depend on previous frames.
#         """
#         self._validate_frame(frame_bgr)

#         if frame_id is None:
#             frame_id = self._next_frame_id
#             self._next_frame_id += 1

#         if frame_timestamp is None:
#             frame_timestamp = time.time()

#         start_monotonic = time.monotonic()

#         detected_box, score, label = self._detect_robot(frame_bgr)

#         if detected_box is None:
#             return self._make_result(
#                 state=TrackingState.SEARCHING,
#                 bbox=None,
#                 frame_id=frame_id,
#                 frame_timestamp=frame_timestamp,
#                 start_monotonic=start_monotonic,
#                 source="detector_only_no_detection",
#                 detector_score=score,
#                 detector_label=label,
#                 tracker_verified=False,
#             )

#         return self._make_result(
#             state=TrackingState.DETECTED,
#             bbox=detected_box,
#             frame_id=frame_id,
#             frame_timestamp=frame_timestamp,
#             start_monotonic=start_monotonic,
#             source="detector_only",
#             detector_score=score,
#             detector_label=label,
#             tracker_verified=False,
#         )

#     def process_frame(
#         self,
#         frame_bgr: np.ndarray,
#         frame_id: Optional[int] = None,
#         frame_timestamp: Optional[float] = None,
#     ) -> RobotTrackResult:
#         """Process one OpenCV BGR frame."""
#         self._validate_frame(frame_bgr)

#         if frame_id is None:
#             frame_id = self._next_frame_id
#             self._next_frame_id += 1

#         if frame_timestamp is None:
#             frame_timestamp = time.time()

#         start_monotonic = time.monotonic()
#         frame_height, frame_width = frame_bgr.shape[:2]

#         if self._tracker is None:
#             return self._search_with_detector(
#                 frame_bgr=frame_bgr,
#                 frame_id=frame_id,
#                 frame_timestamp=frame_timestamp,
#                 start_monotonic=start_monotonic,
#                 source_prefix="detector_acquisition",
#             )

#         tracker_success, tracked_xywh = self._tracker.update(frame_bgr)

#         tracked_box = None
#         if tracker_success:
#             tracked_box = self._xywh_to_xyxy(
#                 tracked_xywh,
#                 frame_width,
#                 frame_height,
#             )

#         if tracked_box is None:
#             self._clear_tracking_only()

#             detected_box, score, label = self._detect_robot(frame_bgr)

#             if detected_box is None:
#                 return self._make_result(
#                     state=TrackingState.SEARCHING,
#                     bbox=None,
#                     frame_id=frame_id,
#                     frame_timestamp=frame_timestamp,
#                     start_monotonic=start_monotonic,
#                     source="detector_reacquisition_failed",
#                 )

#             self._start_tracker(frame_bgr, detected_box)

#             return self._make_result(
#                 state=TrackingState.DETECTED,
#                 bbox=detected_box,
#                 frame_id=frame_id,
#                 frame_timestamp=frame_timestamp,
#                 start_monotonic=start_monotonic,
#                 source="detector_reacquired",
#                 detector_score=score,
#                 detector_label=label,
#                 tracker_verified=True,
#             )

#         self._last_bbox_xyxy = tracked_box

#         if not self._verification_due():
#             return self._make_result(
#                 state=TrackingState.TRACKING,
#                 bbox=tracked_box,
#                 frame_id=frame_id,
#                 frame_timestamp=frame_timestamp,
#                 start_monotonic=start_monotonic,
#                 source="csrt",
#             )

#         detected_box, score, label = self._detect_robot(frame_bgr)

#         if detected_box is None:
#             self._verification_misses += 1
#             self._clear_replacement_candidate()

#             if self._verification_misses >= self.max_verification_misses:
#                 self._clear_tracking_only()
#                 self._clear_acquisition_candidate()
#                 return self._make_result(
#                     state=TrackingState.SEARCHING,
#                     bbox=None,
#                     frame_id=frame_id,
#                     frame_timestamp=frame_timestamp,
#                     start_monotonic=start_monotonic,
#                     source="detector_confirmed_absent",
#                     detector_score=score,
#                     detector_label=label,
#                 )

#             return self._make_result(
#                 state=TrackingState.TRACKING,
#                 bbox=tracked_box,
#                 frame_id=frame_id,
#                 frame_timestamp=frame_timestamp,
#                 start_monotonic=start_monotonic,
#                 source="csrt_detector_miss",
#                 detector_score=score,
#                 detector_label=label,
#             )

#         overlap = self._iou(tracked_box, detected_box)

#         if overlap >= self.verification_iou_threshold:
#             # Reset CSRT using the fresh detector box to correct accumulated drift.
#             self._start_tracker(frame_bgr, detected_box)

#             return self._make_result(
#                 state=TrackingState.TRACKING,
#                 bbox=detected_box,
#                 frame_id=frame_id,
#                 frame_timestamp=frame_timestamp,
#                 start_monotonic=start_monotonic,
#                 source="csrt_corrected_by_detector",
#                 detector_score=score,
#                 detector_label=label,
#                 tracker_verified=True,
#             )

#         # Detector sees a robot elsewhere. Do not jump after one disagreement.
#         self._verification_misses = 0
#         replacement_confirmed = self._update_replacement_candidate(
#             detected_box
#         )

#         if replacement_confirmed:
#             self._start_tracker(frame_bgr, detected_box)
#             return self._make_result(
#                 state=TrackingState.DETECTED,
#                 bbox=detected_box,
#                 frame_id=frame_id,
#                 frame_timestamp=frame_timestamp,
#                 start_monotonic=start_monotonic,
#                 source="detector_replacement_confirmed",
#                 detector_score=score,
#                 detector_label=label,
#                 tracker_verified=True,
#             )

#         return self._make_result(
#             state=TrackingState.TRACKING,
#             bbox=tracked_box,
#             frame_id=frame_id,
#             frame_timestamp=frame_timestamp,
#             start_monotonic=start_monotonic,
#             source="csrt_replacement_candidate",
#             detector_score=score,
#             detector_label=label,
#         )


# class LatestFrameRobotTracker:
#     """Latest-frame-only background wrapper for live Quest frames."""

#     def __init__(
#         self,
#         tracker: RobotDetectorTracker,
#         result_callback: Optional[
#             Callable[[RobotTrackResult, np.ndarray], None]
#         ] = None,
#         copy_submitted_frames: bool = True,
#     ) -> None:
#         self.tracker = tracker
#         self.result_callback = result_callback
#         self.copy_submitted_frames = bool(copy_submitted_frames)

#         self._condition = threading.Condition()
#         self._pending_frame: Optional[np.ndarray] = None
#         self._pending_frame_id: Optional[int] = None
#         self._pending_timestamp: Optional[float] = None

#         self._latest_result: Optional[RobotTrackResult] = None
#         self._latest_processed_frame: Optional[np.ndarray] = None

#         self._next_frame_id = 0
#         self._submitted_frames = 0
#         self._processed_frames = 0
#         self._dropped_frames = 0

#         self._running = False
#         self._worker_thread: Optional[threading.Thread] = None

#     @property
#     def submitted_frames(self) -> int:
#         return self._submitted_frames

#     @property
#     def processed_frames(self) -> int:
#         return self._processed_frames

#     @property
#     def dropped_frames(self) -> int:
#         return self._dropped_frames

#     def start(self) -> None:
#         with self._condition:
#             if self._running:
#                 return

#             self._running = True
#             self._worker_thread = threading.Thread(
#                 target=self._worker_loop,
#                 name="robot-tracker-worker",
#                 daemon=True,
#             )
#             self._worker_thread.start()

#     def stop(self, wait: bool = True) -> None:
#         with self._condition:
#             self._running = False
#             self._condition.notify_all()

#         if (
#             wait
#             and self._worker_thread is not None
#             and self._worker_thread.is_alive()
#         ):
#             self._worker_thread.join()

#     def submit_frame(
#         self,
#         frame_bgr: np.ndarray,
#         frame_id: Optional[int] = None,
#         frame_timestamp: Optional[float] = None,
#     ) -> int:
#         RobotDetectorTracker._validate_frame(frame_bgr)

#         if frame_id is None:
#             frame_id = self._next_frame_id
#             self._next_frame_id += 1

#         if frame_timestamp is None:
#             frame_timestamp = time.time()

#         stored_frame = (
#             frame_bgr.copy()
#             if self.copy_submitted_frames
#             else frame_bgr
#         )

#         with self._condition:
#             if not self._running:
#                 raise RuntimeError(
#                     "LatestFrameRobotTracker is not running. "
#                     "Call start() before submit_frame()."
#                 )

#             self._submitted_frames += 1
#             if self._pending_frame is not None:
#                 self._dropped_frames += 1

#             self._pending_frame = stored_frame
#             self._pending_frame_id = int(frame_id)
#             self._pending_timestamp = float(frame_timestamp)
#             self._condition.notify()

#         return int(frame_id)

#     def get_latest_result(self) -> Optional[RobotTrackResult]:
#         with self._condition:
#             return self._latest_result

#     def get_latest_processed_frame(
#         self,
#         copy_frame: bool = True,
#     ) -> Optional[np.ndarray]:
#         with self._condition:
#             if self._latest_processed_frame is None:
#                 return None
#             return (
#                 self._latest_processed_frame.copy()
#                 if copy_frame
#                 else self._latest_processed_frame
#             )

#     def _worker_loop(self) -> None:
#         while True:
#             with self._condition:
#                 while self._running and self._pending_frame is None:
#                     self._condition.wait()

#                 if not self._running and self._pending_frame is None:
#                     return

#                 frame = self._pending_frame
#                 frame_id = self._pending_frame_id
#                 frame_timestamp = self._pending_timestamp

#                 self._pending_frame = None
#                 self._pending_frame_id = None
#                 self._pending_timestamp = None

#             if frame is None or frame_id is None:
#                 continue

#             result = self.tracker.process_frame(
#                 frame,
#                 frame_id=frame_id,
#                 frame_timestamp=frame_timestamp,
#             )


#             with self._condition:
#                 self._latest_result = result
#                 self._latest_processed_frame = frame
#                 self._processed_frames += 1

#             if self.result_callback is not None:
#                 try:
#                     self.result_callback(result, frame)
#                 except Exception as exc:
#                     print("Robot tracker callback failed:", exc)


# def draw_result(
#     frame_bgr: np.ndarray,
#     result: RobotTrackResult,
# ) -> np.ndarray:
#     output = frame_bgr.copy()

#     if result.bbox_xyxy is not None:
#         x1, y1, x2, y2 = result.bbox_xyxy
#         cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     status = (
#         f"{result.state.value} | {result.source} | "
#         f"{result.processing_time_ms:.1f} ms"
#     )
#     cv2.putText(
#         output,
#         status,
#         (12, 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.65,
#         (0, 255, 0),
#         2,
#         cv2.LINE_AA,
#     )

#     counters = (
#         f"acq={result.acquisition_candidate_count} "
#         f"miss={result.verification_misses} "
#         f"repl={result.replacement_candidate_count}"
#     )
#     cv2.putText(
#         output,
#         counters,
#         (12, 58),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.55,
#         (255, 255, 255),
#         2,
#         cv2.LINE_AA,
#     )

#     if result.detector_score is not None:
#         detector_text = f"Detector: {result.detector_score:.3f}"
#         if result.detector_label:
#             detector_text += f" ({result.detector_label})"

#         cv2.putText(
#             output,
#             detector_text,
#             (12, 86),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.55,
#             (0, 255, 0),
#             2,
#             cv2.LINE_AA,
#         )

#     return output


# def process_video(
#     input_video_path: str,
#     output_video_path: str,
#     detector,
#     class_names: Sequence[str] = ("robot",),
#     verification_interval_seconds: float = 0.4,
#     max_frames_to_process: int = 100,
#     show_preview: bool = False,
# ) -> None:
#     """Run the same engine on every frame of an offline video."""
#     tracker = RobotDetectorTracker(
#         detector=detector,
#         class_names=class_names,
#         verification_interval_seconds=verification_interval_seconds,
#         verification_iou_threshold=0.40,
#         acquisition_confirmations=2,
#         acquisition_iou_threshold=0.40,
#         max_verification_misses=2,
#         replacement_confirmations=2,
#         replacement_iou_threshold=0.40,
#     )

#     capture = cv2.VideoCapture(input_video_path)
#     if not capture.isOpened():
#         raise RuntimeError(f"Could not open input video: {input_video_path}")

#     fps = capture.get(cv2.CAP_PROP_FPS)
#     if fps <= 0:
#         fps = 30.0

#     width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

#     writer = cv2.VideoWriter(
#         output_video_path,
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (width, height),
#     )

#     if not writer.isOpened():
#         capture.release()
#         raise RuntimeError(
#             f"Could not create output video: {output_video_path}"
#         )

#     frame_id = 0

#     try:
#         while True:
#             if max_frames_to_process is not None and frame_id >= max_frames_to_process:
#                 break
#             success, frame = capture.read()
#             if not success:
#                 break

#             result = tracker.process_frame(
#                 frame,
#                 frame_id=frame_id,
#                 frame_timestamp=time.time(),
#             )
#             annotated = draw_result(frame, result)

#             cv2.putText(
#                 annotated,
#                 f"Frame {frame_id + 1}/{total_frames}",
#                 (12, 114),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.55,
#                 (255, 255, 255),
#                 2,
#                 cv2.LINE_AA,
#             )

#             writer.write(annotated)

#             if show_preview:
#                 cv2.imshow("Robot Detection and Tracking", annotated)
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break

#             frame_id += 1
#             if frame_id % 100 == 0:
#                 print(f"Processed {frame_id}/{total_frames} frames")

#     finally:
#         capture.release()
#         writer.release()
#         cv2.destroyAllWindows()

#     print(f"Finished. Saved annotated video to: {output_video_path}")

# def process_video_detector_only(
#     input_video_path: str,
#     output_video_path: str,
#     detector,
#     class_names: Sequence[str] = ("unitree go2",),
#     max_frames_to_process: Optional[int] = None,
# ) -> None:
#     capture = cv2.VideoCapture(input_video_path)

#     if not capture.isOpened():
#         raise RuntimeError(
#             f"Could not open input video: {input_video_path}"
#         )

#     fps = capture.get(cv2.CAP_PROP_FPS)
#     if fps <= 0:
#         fps = 30.0

#     width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     writer = cv2.VideoWriter(
#         output_video_path,
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (width, height),
#     )

#     frame_id = 0

#     try:
#         while True:
#             if (
#                 max_frames_to_process is not None
#                 and frame_id >= max_frames_to_process
#             ):
#                 break

#             success, frame_bgr = capture.read()
#             if not success:
#                 break

#             frame_rgb = cv2.cvtColor(
#                 frame_bgr,
#                 cv2.COLOR_BGR2RGB,
#             )

#             detections = detector.predict(
#                 frame_rgb,
#                 list(class_names),
#             )

#             annotated = frame_bgr.copy()

#             for detection in detections:
#                 score = float(detection["score"])
#                 x1, y1, x2, y2 = [
#                     int(round(value))
#                     for value in detection["box_xyxy"]
#                 ]

#                 cv2.rectangle(
#                     annotated,
#                     (x1, y1),
#                     (x2, y2),
#                     (0, 255, 0),
#                     2,
#                 )

#                 cv2.putText(
#                     annotated,
#                     f"{score:.3f}",
#                     (x1, max(20, y1 - 8)),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 255, 0),
#                     2,
#                     cv2.LINE_AA,
#                 )

#             writer.write(annotated)
#             frame_id += 1

#     finally:
#         capture.release()
#         writer.release()

#     print(f"Saved detector-only video to {output_video_path}")


# # ---------------------------------------------------------------------
# # Video-test settings
# # ---------------------------------------------------------------------

# from pathlib import Path


# # Select either "yolo" or "dino".
# DETECTOR_TYPE = "yolo"

# ROBOT_LABELS = (
#     "robot dog",
#     "quadruped robot",
#     "four-legged robot",
# )

# SHOW_PREVIEW = False

# # Set to None to process the entire video.
# MAX_FRAMES_TO_PROCESS = 500


# # This file is expected to be:
# # robot_fov_estimation/src/robot_detector_tracker.py
# ROBOT_FOV_ROOT = Path(__file__).resolve().parents[1]

# INPUT_VIDEO_PATH = (
#     ROBOT_FOV_ROOT
#     / "test_videos"
#     / "dogtest.MP4"
# )

# OUTPUT_VIDEO_PATH = (
#     ROBOT_FOV_ROOT
#     / "outputs"
#     / f"detector_only_{DETECTOR_TYPE}.mp4"
# )


# if __name__ == "__main__":
#     from model_training_and_implementation.src.dino_detector import (
#         DINOObjectDetector,
#     )
#     from robot_fov_estimation.src.go2_yolo_det_wrapper import (
#         YOLOGo2Detector,
#     )

#     if not INPUT_VIDEO_PATH.is_file():
#         raise FileNotFoundError(
#             f"Input video does not exist: {INPUT_VIDEO_PATH}"
#         )

#     OUTPUT_VIDEO_PATH.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     detector_type = DETECTOR_TYPE.strip().lower()

#     if detector_type == "yolo":
#         detector = YOLOGo2Detector(
#             confidence=0.4,
#             iou_threshold=0.50,
#             image_size=1280,
#             device=0,
#         )

#     elif detector_type == "dino":
#         detector = DINOObjectDetector(
#             confidence=0.30,
#         )

#     else:
#         raise ValueError(
#             f"Unsupported DETECTOR_TYPE: {DETECTOR_TYPE!r}. "
#             'Use either "yolo" or "dino".'
#         )

#     print(f"Detector: {detector_type}")
#     print(f"Input: {INPUT_VIDEO_PATH}")
#     print(f"Output: {OUTPUT_VIDEO_PATH}")

#     process_video_detector_only(
#         input_video_path=str(INPUT_VIDEO_PATH),
#         output_video_path=str(OUTPUT_VIDEO_PATH),
#         detector=detector,
#         class_names=ROBOT_LABELS,
#         max_frames_to_process=MAX_FRAMES_TO_PROCESS,
        
#     )

"""
robot_detector_tracker_detector_driven.py

Detector-driven robot tracking for both:

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

Recommended detector configuration:

- Configure the detector itself with a LOW confidence threshold, such as 0.05.
- This class applies a separate, higher threshold before a detection can
  create a new track.

Tracking flow:

1. Run the detector on every full frame.
2. Require repeated high-confidence detections to acquire a new track.
3. Predict the next box with a constant-velocity Kalman filter.
4. Associate all full-frame detections, including low-confidence detections,
   with the prediction using IoU, center distance, confidence, and size.
5. If no full-frame detection matches, run the detector again on an expanded
   crop around the predicted box.
6. If both detector passes miss, return a Kalman-predicted box for only a
   small number of frames / a short amount of time.
7. If misses continue, delete the track and resume searching.
8. While searching, periodically run overlapping tiled inference to improve
   recall for small, distant robots.

Unlike the original CSRT version, every verified track update comes from the
semantic detector. Kalman prediction is used only for short gaps.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


BBoxXYXY = Tuple[int, int, int, int]


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
    tracker_verified: bool = False
    acquisition_candidate_count: int = 0
    replacement_candidate_count: int = 0
    verification_misses: int = 0

    # Additional diagnostics. Defaults preserve compatibility with callers
    # that instantiate or consume the original result fields.
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


class _BoxKalmanFilter:
    """
    Constant-velocity Kalman filter for [center_x, center_y, width, height].

    State:
        [cx, cy, w, h, vx, vy, vw, vh]

    Measurement:
        [cx, cy, w, h]
    """

    def __init__(
        self,
        process_noise_position: float = 8.0,
        process_noise_velocity: float = 60.0,
        measurement_noise_position: float = 16.0,
        measurement_noise_size: float = 25.0,
    ) -> None:
        self.process_noise_position = float(process_noise_position)
        self.process_noise_velocity = float(process_noise_velocity)
        self.measurement_noise_position = float(
            measurement_noise_position
        )
        self.measurement_noise_size = float(measurement_noise_size)

        self._x = np.zeros((8, 1), dtype=np.float64)
        self._p = np.eye(8, dtype=np.float64)
        self._h = np.zeros((4, 8), dtype=np.float64)
        self._h[:4, :4] = np.eye(4, dtype=np.float64)
        self._identity = np.eye(8, dtype=np.float64)
        self.initialized = False

    @staticmethod
    def _box_to_measurement(box: BBoxXYXY) -> np.ndarray:
        x1, y1, x2, y2 = box
        width = max(1.0, float(x2 - x1))
        height = max(1.0, float(y2 - y1))
        center_x = float(x1) + width * 0.5
        center_y = float(y1) + height * 0.5
        return np.array(
            [[center_x], [center_y], [width], [height]],
            dtype=np.float64,
        )

    def initialize(self, box: BBoxXYXY) -> None:
        measurement = self._box_to_measurement(box)
        self._x.fill(0.0)
        self._x[:4] = measurement

        # Location and size are known reasonably well. Velocity begins highly
        # uncertain so the filter can adapt quickly to early motion.
        self._p = np.diag(
            [
                25.0,
                25.0,
                64.0,
                64.0,
                400.0,
                400.0,
                400.0,
                400.0,
            ]
        ).astype(np.float64)
        self.initialized = True

    def predict(self, delta_seconds: float) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Kalman filter has not been initialized.")

        dt = float(np.clip(delta_seconds, 1.0 / 240.0, 0.5))

        transition = np.eye(8, dtype=np.float64)
        transition[0, 4] = dt
        transition[1, 5] = dt
        transition[2, 6] = dt
        transition[3, 7] = dt

        # A simple diagonal process-noise model is sufficient here because
        # detector measurements arrive on nearly every frame.
        position_noise = self.process_noise_position * max(dt, 1.0 / 30.0)
        velocity_noise = self.process_noise_velocity * max(dt, 1.0 / 30.0)
        process_noise = np.diag(
            [
                position_noise,
                position_noise,
                position_noise,
                position_noise,
                velocity_noise,
                velocity_noise,
                velocity_noise,
                velocity_noise,
            ]
        ).astype(np.float64)

        self._x = transition @ self._x
        self._p = transition @ self._p @ transition.T + process_noise
        self._enforce_valid_size()
        return self._x.copy()

    def correct(self, box: BBoxXYXY) -> np.ndarray:
        if not self.initialized:
            self.initialize(box)
            return self._x.copy()

        measurement = self._box_to_measurement(box)
        measurement_noise = np.diag(
            [
                self.measurement_noise_position,
                self.measurement_noise_position,
                self.measurement_noise_size,
                self.measurement_noise_size,
            ]
        ).astype(np.float64)

        innovation = measurement - self._h @ self._x
        innovation_covariance = (
            self._h @ self._p @ self._h.T + measurement_noise
        )
        kalman_gain = (
            self._p
            @ self._h.T
            @ np.linalg.pinv(innovation_covariance)
        )

        self._x = self._x + kalman_gain @ innovation
        self._p = (
            self._identity - kalman_gain @ self._h
        ) @ self._p
        self._enforce_valid_size()
        return self._x.copy()

    def _enforce_valid_size(self) -> None:
        self._x[2, 0] = max(2.0, self._x[2, 0])
        self._x[3, 0] = max(2.0, self._x[3, 0])

    def bbox_xyxy(
        self,
        frame_width: int,
        frame_height: int,
    ) -> Optional[BBoxXYXY]:
        if not self.initialized:
            return None

        center_x, center_y, width, height = self._x[:4, 0]
        if not np.all(np.isfinite([center_x, center_y, width, height])):
            return None

        half_width = max(1.0, width * 0.5)
        half_height = max(1.0, height * 0.5)

        x1 = int(round(max(0.0, center_x - half_width)))
        y1 = int(round(max(0.0, center_y - half_height)))
        x2 = int(round(min(float(frame_width), center_x + half_width)))
        y2 = int(round(min(float(frame_height), center_y + half_height)))

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def reset(self) -> None:
        self.initialized = False
        self._x.fill(0.0)
        self._p = np.eye(8, dtype=np.float64)


class RobotDetectorTracker:
    """Synchronous detector-driven robot tracker."""

    def __init__(
        self,
        detector,
        class_names: Sequence[str] = ("robot",),
        # Retained for compatibility with the former CSRT implementation.
        # The detector now runs on every frame, so this value is not used.
        verification_interval_seconds: float = 0.0,
        # Used as the minimum IoU association gate in this implementation.
        verification_iou_threshold: float = 0.10,
        acquisition_confirmations: int = 2,
        acquisition_iou_threshold: float = 0.30,
        # Number of consecutive detector misses for which Kalman coasting is
        # allowed. This replaces the old detector-verification miss count.
        max_verification_misses: int = 3,
        replacement_confirmations: int = 2,
        replacement_iou_threshold: float = 0.30,
        # The injected detector itself must also return detections down to this
        # score. For YOLO, configure its wrapper confidence to 0.05 or lower.
        min_detection_score: float = 0.05,
        # A detection below this score may maintain an existing track, but it
        # cannot create a new one or replace an existing one.
        acquisition_min_score: float = 0.40,
        minimum_association_score: float = 0.28,
        max_normalized_center_distance: float = 1.50,
        center_distance_sigma: float = 0.75,
        max_area_ratio: float = 6.0,
        association_iou_weight: float = 0.55,
        association_center_weight: float = 0.30,
        association_confidence_weight: float = 0.15,
        crop_redetection_scale: float = 2.5,
        crop_redetection_min_size: int = 192,
        max_unverified_seconds: float = 0.25,
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
            raise ValueError("Association weights must sum to a positive value.")

        self.detector = detector
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

        self._kalman = _BoxKalmanFilter()
        self._track_active = False
        self._last_bbox_xyxy: Optional[BBoxXYXY] = None
        self._last_filter_timestamp: Optional[float] = None
        self._last_verified_timestamp: Optional[float] = None

        self._next_frame_id = 0
        self._search_frame_count = 0

        self._acquisition_candidate: Optional[_DetectionCandidate] = None
        self._acquisition_candidate_count = 0

        self._replacement_candidate: Optional[_DetectionCandidate] = None
        self._replacement_candidate_count = 0

        self._verification_misses = 0

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
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

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

        # If clipping at an image edge made the crop too small, expand it in
        # the opposite direction where possible.
        current_width = crop_x2 - crop_x1
        current_height = crop_y2 - crop_y1
        desired_width = min(frame_width, int(round(width)))
        desired_height = min(frame_height, int(round(height)))

        if current_width < desired_width:
            missing = desired_width - current_width
            shift_left = min(crop_x1, missing)
            crop_x1 -= shift_left
            crop_x2 = min(frame_width, crop_x2 + (missing - shift_left))

        if current_height < desired_height:
            missing = desired_height - current_height
            shift_up = min(crop_y1, missing)
            crop_y1 -= shift_up
            crop_y2 = min(frame_height, crop_y2 + (missing - shift_up))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        return crop_x1, crop_y1, crop_x2, crop_y2

    def _detect_predicted_crop(
        self,
        frame_bgr: np.ndarray,
        predicted_box: BBoxXYXY,
    ) -> List[_DetectionCandidate]:
        frame_height, frame_width = frame_bgr.shape[:2]
        crop_box = self._expand_box(
            predicted_box,
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
        self._kalman.reset()
        self._last_bbox_xyxy = None
        self._last_filter_timestamp = None
        self._last_verified_timestamp = None
        self._verification_misses = 0
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
                return max(consistent, key=lambda detection: detection.score)

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
                return max(consistent, key=lambda detection: detection.score)

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
        candidate: _DetectionCandidate,
        frame_timestamp: float,
    ) -> None:
        self._kalman.initialize(candidate.bbox_xyxy)
        self._track_active = True
        self._last_bbox_xyxy = candidate.bbox_xyxy
        self._last_filter_timestamp = frame_timestamp
        self._last_verified_timestamp = frame_timestamp
        self._verification_misses = 0
        self._search_frame_count = 0
        self._clear_replacement_candidate()
        self._clear_acquisition_candidate()

    def _predict_track(
        self,
        frame_timestamp: float,
        frame_width: int,
        frame_height: int,
    ) -> Optional[BBoxXYXY]:
        if not self._track_active or not self._kalman.initialized:
            return None

        if self._last_filter_timestamp is None:
            delta_seconds = 1.0 / 30.0
        else:
            delta_seconds = frame_timestamp - self._last_filter_timestamp
            if not np.isfinite(delta_seconds) or delta_seconds <= 0.0:
                delta_seconds = 1.0 / 30.0

        self._kalman.predict(delta_seconds)
        self._last_filter_timestamp = frame_timestamp
        return self._kalman.bbox_xyxy(frame_width, frame_height)

    def _association_metrics(
        self,
        predicted_box: BBoxXYXY,
        candidate: _DetectionCandidate,
    ) -> Tuple[bool, float]:
        candidate_box = candidate.bbox_xyxy
        overlap = self._iou(predicted_box, candidate_box)

        predicted_center = self._box_center(predicted_box)
        candidate_center = self._box_center(candidate_box)
        center_distance = math.hypot(
            predicted_center[0] - candidate_center[0],
            predicted_center[1] - candidate_center[1],
        )
        normalized_center_distance = (
            center_distance / max(1.0, self._box_diagonal(predicted_box))
        )

        predicted_area = max(1.0, self._box_area(predicted_box))
        candidate_area = max(1.0, self._box_area(candidate_box))
        area_ratio = max(
            predicted_area / candidate_area,
            candidate_area / predicted_area,
        )

        passes_spatial_gate = (
            overlap >= self.verification_iou_threshold
            or normalized_center_distance
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
        predicted_box: BBoxXYXY,
        detections: Sequence[_DetectionCandidate],
    ) -> Tuple[Optional[_DetectionCandidate], Optional[float]]:
        best_candidate: Optional[_DetectionCandidate] = None
        best_score: Optional[float] = None

        for candidate in detections:
            valid, score = self._association_metrics(
                predicted_box,
                candidate,
            )
            if not valid:
                continue
            if best_score is None or score > best_score:
                best_candidate = candidate
                best_score = score

        return best_candidate, best_score

    def _correct_track(
        self,
        candidate: _DetectionCandidate,
        frame_timestamp: float,
        frame_width: int,
        frame_height: int,
    ) -> BBoxXYXY:
        self._kalman.correct(candidate.bbox_xyxy)
        corrected_box = self._kalman.bbox_xyxy(
            frame_width,
            frame_height,
        )
        if corrected_box is None:
            corrected_box = candidate.bbox_xyxy
            self._kalman.initialize(corrected_box)

        self._last_bbox_xyxy = corrected_box
        self._last_filter_timestamp = frame_timestamp
        self._last_verified_timestamp = frame_timestamp
        self._verification_misses = 0
        self._clear_replacement_candidate()
        return corrected_box

    def _prediction_age_seconds(self, frame_timestamp: float) -> float:
        if self._last_verified_timestamp is None:
            return float("inf")
        return max(0.0, frame_timestamp - self._last_verified_timestamp)

    def _should_run_tiled_search(self) -> bool:
        if self.tiled_search_interval_frames == 0:
            return False

        # Once a tile has produced a tentative acquisition, use tiles again on
        # the next frame so a distant robot can actually receive consecutive
        # confirmations.
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
            prediction_age_frames = self._verification_misses
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
        frame_height, frame_width = frame_bgr.shape[:2]

        if not self._track_active:
            return self._process_searching_frame(
                frame_bgr,
                frame_id,
                frame_timestamp,
                start_monotonic,
            )

        predicted_box = self._predict_track(
            frame_timestamp,
            frame_width,
            frame_height,
        )
        if predicted_box is None:
            self._clear_tracking_only()
            return self._process_searching_frame(
                frame_bgr,
                frame_id,
                frame_timestamp,
                start_monotonic,
            )

        full_detections = self._nms_candidates(
            self._detect_full_frame(frame_bgr)
        )
        matched_detection, association_score = self._associate_detection(
            predicted_box,
            full_detections,
        )

        if matched_detection is not None:
            corrected_box = self._correct_track(
                matched_detection,
                frame_timestamp,
                frame_width,
                frame_height,
            )
            confidence_kind = (
                "high"
                if matched_detection.score >= self.acquisition_min_score
                else "low"
            )
            return self._make_result(
                state=TrackingState.TRACKING,
                bbox=corrected_box,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source=f"detector_full_{confidence_kind}_confidence",
                detector_score=matched_detection.score,
                detector_label=matched_detection.label,
                tracker_verified=True,
                prediction_age_frames=0,
                prediction_age_seconds=0.0,
                association_score=association_score,
            )

        # A full-frame miss may simply mean that a distant robot became too
        # small after full-frame resizing. Re-run the same detector on a crop
        # centered on the Kalman prediction.
        crop_detections = self._nms_candidates(
            self._detect_predicted_crop(frame_bgr, predicted_box)
        )
        matched_detection, association_score = self._associate_detection(
            predicted_box,
            crop_detections,
        )

        if matched_detection is not None:
            corrected_box = self._correct_track(
                matched_detection,
                frame_timestamp,
                frame_width,
                frame_height,
            )
            confidence_kind = (
                "high"
                if matched_detection.score >= self.acquisition_min_score
                else "low"
            )
            return self._make_result(
                state=TrackingState.TRACKING,
                bbox=corrected_box,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source=f"detector_crop_{confidence_kind}_confidence",
                detector_score=matched_detection.score,
                detector_label=matched_detection.label,
                tracker_verified=True,
                prediction_age_frames=0,
                prediction_age_seconds=0.0,
                association_score=association_score,
                used_crop_redetection=True,
            )

        # A high-confidence detection elsewhere must persist before replacing
        # the current track. This retains the conservative replacement behavior
        # from the original implementation.
        replacement_candidate = self._choose_replacement_candidate(
            full_detections
        )
        replacement_confirmed = self._update_replacement_candidate(
            replacement_candidate
        )

        if replacement_confirmed:
            assert replacement_candidate is not None
            self._start_track(replacement_candidate, frame_timestamp)
            return self._make_result(
                state=TrackingState.DETECTED,
                bbox=replacement_candidate.bbox_xyxy,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source="detector_replacement_confirmed",
                detector_score=replacement_candidate.score,
                detector_label=replacement_candidate.label,
                tracker_verified=True,
                used_crop_redetection=True,
            )

        self._verification_misses += 1
        prediction_age_seconds = self._prediction_age_seconds(
            frame_timestamp
        )

        may_coast = (
            self._verification_misses <= self.max_verification_misses
            and prediction_age_seconds <= self.max_unverified_seconds
        )
        if may_coast:
            self._last_bbox_xyxy = predicted_box
            return self._make_result(
                state=TrackingState.TRACKING,
                bbox=predicted_box,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                start_monotonic=start_monotonic,
                source="kalman_prediction_unverified",
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
                prediction_age_frames=self._verification_misses,
                prediction_age_seconds=prediction_age_seconds,
                used_crop_redetection=True,
            )

        # The detector has failed for too long. Do not allow indefinite motion
        # prediction. Preserve the current high-confidence unmatched detection
        # as the first tentative acquisition for a possible new track.
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
                "track_expired_new_candidate"
                if expired_candidate is not None
                else "track_expired_detector_absent"
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
        # Yellow means an unverified Kalman prediction; green means the box
        # was supported by a detector result on this frame.
        color = (
            (0, 255, 0)
            if result.tracker_verified
            else (0, 255, 255)
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
        f"miss={result.verification_misses} "
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
        f"verified={int(result.tracker_verified)}"
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
    class_names: Sequence[str] = ("robot",),
    max_frames_to_process: Optional[int] = None,
    show_preview: bool = False,
) -> None:
    """Run the same detector-driven engine on an offline video."""
    tracker = RobotDetectorTracker(
        detector=detector,
        class_names=class_names,
        acquisition_confirmations=2,
        acquisition_iou_threshold=0.30,
        max_verification_misses=3,
        replacement_confirmations=2,
        replacement_iou_threshold=0.30,
        min_detection_score=0.05,
        acquisition_min_score=0.40,
        verification_iou_threshold=0.10,
        max_unverified_seconds=0.25,
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

            # Use video time rather than wall-clock processing time so Kalman
            # behavior matches the source frame rate even during slow testing.
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
                cv2.imshow("Robot Detection and Tracking", annotated)
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

# Your custom YOLO detector is the intended detector for this pipeline.
# DINO support is retained only for comparison.
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
# robot_fov_estimation/src/robot_detector_tracker.py
ROBOT_FOV_ROOT = Path(__file__).resolve().parents[1]

INPUT_VIDEO_PATH = (
    ROBOT_FOV_ROOT
    / "test_videos"
    / "dogtest.MP4"
)

OUTPUT_VIDEO_PATH = (
    ROBOT_FOV_ROOT
    / "outputs"
    / f"detector_driven_tracking_{DETECTOR_TYPE}.mp4"
)


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
            # Important: the wrapper must return low-confidence detections so
            # temporal association can use them. The tracker itself still
            # requires 0.40 by default to create a new track.
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

    print(f"Detector: {detector_type}")
    print(f"Input: {INPUT_VIDEO_PATH}")
    print(f"Output: {OUTPUT_VIDEO_PATH}")

    process_video(
        input_video_path=str(INPUT_VIDEO_PATH),
        output_video_path=str(OUTPUT_VIDEO_PATH),
        detector=detector,
        class_names=ROBOT_LABELS,
        max_frames_to_process=MAX_FRAMES_TO_PROCESS,
        show_preview=SHOW_PREVIEW,
    )