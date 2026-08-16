"""
Robot ground-truth + Quest image alignment server.

Place this file at:

    UnitreeGo2HandoffIntentDetector/
        robot_pose_prediction/
            data_collection_server.py

Then run it from the repository root with:

    python -m robot_pose_prediction.data_collection_server

Optional:

    python -m robot_pose_prediction.data_collection_server \
        --host 0.0.0.0 \
        --port 8000

What this server does
---------------------
1. Receives small robot ground-truth batches at /robot_ground_truth.
2. Keeps the robot data exactly in the coordinate convention sent by the robot.
3. Accumulates and saves robot-only recordings.
4. Receives Quest motion-history windows, optionally including JPEG images.
5. Normalizes Quest timestamps into the robot timestamp clock domain.
6. Buffers Quest windows until robot GT covers the complete required time range.
7. Interpolates robot position + 3-D heading to:
       - every Quest observation timestamp
       - every Quest image capture timestamp
8. Saves aligned .npz metadata and JPEG files.

What it deliberately does NOT do
--------------------------------
- ROS -> Unity coordinate conversion
- Quest <-> robot spatial-frame conversion
- ArUco/shared-marker calibration
- FoV logic
- model inference

Temporal alignment still requires the robot and Quest timestamps to be
meaningfully related. If the Quest uses monotonic/realtime timestamps, the
server estimates a Quest->Unix offset using sent_at_unix_seconds.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import os
import re
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Robot Ground Truth + Quest Image Alignment Server",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Robot samples are kept in memory for temporal interpolation. Raw recordings
# are also saved to disk, so this is only the live alignment buffer.
ROBOT_BUFFER_SECONDS = 600.0

# Backward-compatible sequence boundary for robot senders that do not provide
# recording_id. A gap larger than this starts a new robot-only recording.
LEGACY_RECORDING_BREAK_SECONDS = 2.0

# Quest clock normalization.
UNIX_TIMESTAMP_THRESHOLD = 1.0e9
QUEST_OFFSET_HISTORY_SIZE = 100
QUEST_OFFSET_PERCENTILE = 10.0
QUEST_CLOCK_RESET_THRESHOLD_SECONDS = 5.0

# Loose consistency check between raw sample timestamps and relative window time.
QUEST_WINDOW_SPAN_ABSOLUTE_TOLERANCE_SECONDS = 0.50
QUEST_WINDOW_SPAN_RELATIVE_TOLERANCE = 0.10

# Image payload safety.
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGES_PER_WINDOW = 500

# Diagnostic only. Alignment is still saved if the robot stream bridges a
# larger interval.
ALIGNMENT_GAP_WARNING_SECONDS = 0.25


# ---------------------------------------------------------------------------
# Repository/data paths
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """
    Find the repository root.

    When installed at robot_pose_prediction/data_collection_server.py this
    resolves directly from the module location, so launching from the repo root
    with `python -m ...` does not depend on the current working directory.
    """
    module_path = Path(__file__).resolve()
    module_dir = module_path.parent

    if module_dir.name == "robot_pose_prediction":
        return module_dir.parent

    for parent in [module_dir, *module_dir.parents]:
        if parent.name == "UnitreeGo2HandoffIntentDetector":
            return parent

        candidate = parent / "UnitreeGo2HandoffIntentDetector"
        if candidate.is_dir():
            return candidate

    return Path.cwd()


PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "robot_pose_prediction" / "data"

ROBOT_RECORDING_DIR = DATA_DIR / "robot_ground_truth"
ALIGNED_WINDOW_DIR = DATA_DIR / "quest_aligned"
IMAGE_DATA_DIR = DATA_DIR / "images"

for directory in (
    DATA_DIR,
    ROBOT_RECORDING_DIR,
    ALIGNED_WINDOW_DIR,
    IMAGE_DATA_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class Vector3(BaseModel):
    x: float
    y: float
    z: float


class QuaternionValue(BaseModel):
    """Quaternion serialized in Unity x, y, z, w component order."""
    x: float
    y: float
    z: float
    w: float


class RobotGroundTruthSample(BaseModel):
    timestamp: float
    position: Vector3
    heading: Vector3


class RobotGroundTruthBatch(BaseModel):
    """
    Small batch from the robot sender.

    recording_id is optional so the existing sender remains accepted unchanged.
    """
    schema_version: str = "1.0"
    source_id: str = "unitree_go2"
    recording_id: Optional[str] = None
    sample_count: int = Field(ge=1)
    samples: List[RobotGroundTruthSample]


class QuestRobotObservation(BaseModel):
    """
    Quest-side observation of the robot.

    timestamp may be Unix time or Quest/Unity realtime/monotonic seconds.
    """
    timestamp: float
    time_from_window_start: float
    position: Vector3
    heading: Vector3
    has_image: bool = False
    image_frame_id: int = -1


class QuestImage(BaseModel):
    """
    One JPEG captured by the Quest.

    capture_timestamp_unix is preferred for image-to-robot alignment.
    capture_realtime_seconds is retained as a fallback if the Unix field is not
    valid.
    """
    frame_id: int = Field(ge=0)
    mime_type: str = "image/jpeg"
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    capture_timestamp_unix: float
    capture_realtime_seconds: float
    camera_position: Optional[Vector3] = None
    camera_rotation: Optional[QuaternionValue] = None
    jpeg_base64: str


class QuestMotionHistoryWindow(BaseModel):
    schema_version: str = "1.0"
    window_id: str
    source_id: str

    # Which robot GT source to align against.
    robot_source_id: str = "unitree_go2"

    # Prefer a true Unix timestamp generated on the Quest at send time.
    sent_at_unix_seconds: float

    window_duration_seconds: float
    sample_count: int = Field(ge=1)

    includes_images: bool = False
    image_count: int = Field(default=0, ge=0)

    samples: List[QuestRobotObservation]
    images: List[QuestImage] = Field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class PreparedQuestWindow:
    window: QuestMotionHistoryWindow

    original_timestamps: np.ndarray
    aligned_timestamps: np.ndarray

    timestamp_mode: str
    clock_offset_seconds: float
    clock_offset_candidate_seconds: float
    clock_offset_reference: str

    # One alignment timestamp per image, in the same clock domain as robot GT.
    image_alignment_timestamps: np.ndarray


@dataclass
class LegacyRobotRecordingState:
    recording_id: str
    last_timestamp: float


# Live robot samples, by source.
robot_ground_truth_buffers: Dict[
    str,
    List[RobotGroundTruthSample],
] = defaultdict(list)

# Quest windows waiting for robot coverage.
pending_quest_windows: Dict[
    Tuple[str, str],
    PreparedQuestWindow,
] = {}

# Quest monotonic -> Unix clock-offset estimator.
quest_clock_offset_candidates: Dict[
    str,
    Deque[float],
] = defaultdict(
    lambda: deque(
        maxlen=QUEST_OFFSET_HISTORY_SIZE
    )
)

quest_clock_offset_estimates: Dict[
    str,
    float,
] = {}

quest_clock_last_raw_timestamp: Dict[
    str,
    float,
] = {}

legacy_robot_recordings: Dict[
    str,
    LegacyRobotRecordingState,
] = {}

state_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def sanitize_filename_component(
    value: str,
) -> str:
    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    return value.strip("._") or "unknown"


def vector3_list_to_array(
    vectors: List[Vector3],
) -> np.ndarray:
    return np.asarray(
        [
            [v.x, v.y, v.z]
            for v in vectors
        ],
        dtype=np.float64,
    )


def looks_like_unix_timestamp(
    timestamp: float,
) -> bool:
    return (
        math.isfinite(timestamp)
        and timestamp >= UNIX_TIMESTAMP_THRESHOLD
    )


def normalize_direction_rows(
    directions: np.ndarray,
) -> np.ndarray:
    """
    Normalize arbitrary 3-D direction vectors.

    This is intentionally coordinate-system agnostic. It does not assume that
    X/Z or X/Y is the horizontal plane.
    """
    directions = np.asarray(
        directions,
        dtype=np.float64,
    )

    norms = np.linalg.norm(
        directions,
        axis=1,
    )

    if np.any(norms < 1.0e-8):
        raise ValueError(
            "At least one heading vector has near-zero magnitude."
        )

    return directions / norms[:, None]


def get_aligned_window_path(
    source_id: str,
    window_id: str,
) -> Path:
    return (
        ALIGNED_WINDOW_DIR
        / (
            f"{sanitize_filename_component(source_id)}__"
            f"{sanitize_filename_component(window_id)}.npz"
        )
    )


def get_window_image_directory(
    source_id: str,
    window_id: str,
) -> Path:
    return (
        IMAGE_DATA_DIR
        / sanitize_filename_component(source_id)
        / sanitize_filename_component(window_id)
    )


# ---------------------------------------------------------------------------
# Robot GT validation, buffering, and persistence
# ---------------------------------------------------------------------------

def validate_robot_batch(
    batch: RobotGroundTruthBatch,
) -> None:
    if batch.sample_count != len(batch.samples):
        raise ValueError(
            "sample_count does not match the number of robot "
            "ground-truth samples in the payload."
        )

    if not batch.samples:
        raise ValueError(
            "Robot ground-truth batch has no samples."
        )

    timestamps = np.asarray(
        [
            sample.timestamp
            for sample in batch.samples
        ],
        dtype=np.float64,
    )

    positions = vector3_list_to_array(
        [
            sample.position
            for sample in batch.samples
        ]
    )

    headings = vector3_list_to_array(
        [
            sample.heading
            for sample in batch.samples
        ]
    )

    if not np.all(np.isfinite(timestamps)):
        raise ValueError(
            "Robot ground-truth timestamps contain non-finite values."
        )

    if not np.all(np.isfinite(positions)):
        raise ValueError(
            "Robot ground-truth positions contain non-finite values."
        )

    if not np.all(np.isfinite(headings)):
        raise ValueError(
            "Robot ground-truth headings contain non-finite values."
        )

    # Validate only magnitude; do not change the sender's coordinate convention.
    normalize_direction_rows(
        headings
    )


def sort_and_deduplicate_robot_buffer(
    source_id: str,
) -> None:
    samples = robot_ground_truth_buffers[
        source_id
    ]

    if not samples:
        return

    # Newer received duplicate wins.
    by_timestamp: Dict[
        float,
        RobotGroundTruthSample,
    ] = {}

    for sample in samples:
        by_timestamp[
            float(sample.timestamp)
        ] = sample

    robot_ground_truth_buffers[
        source_id
    ] = [
        by_timestamp[timestamp]
        for timestamp in sorted(
            by_timestamp.keys()
        )
    ]


def prune_robot_buffer(
    source_id: str,
) -> None:
    samples = robot_ground_truth_buffers[
        source_id
    ]

    if not samples:
        return

    newest = float(
        samples[-1].timestamp
    )

    cutoff = (
        newest
        - ROBOT_BUFFER_SECONDS
    )

    # Keep anything still needed by a pending Quest window for this robot.
    needed_starts = []

    for prepared in pending_quest_windows.values():
        if prepared.window.robot_source_id != source_id:
            continue

        required_times = get_required_alignment_times(
            prepared
        )

        if len(required_times):
            needed_starts.append(
                float(
                    required_times.min()
                )
            )

    if needed_starts:
        cutoff = min(
            cutoff,
            min(needed_starts) - 1.0,
        )

    first_keep = 0

    while (
        first_keep < len(samples)
        and float(samples[first_keep].timestamp) < cutoff
    ):
        first_keep += 1

    if first_keep > 0:
        robot_ground_truth_buffers[
            source_id
        ] = samples[first_keep:]


def resolve_robot_recording_id(
    batch: RobotGroundTruthBatch,
) -> str:
    if (
        batch.recording_id is not None
        and batch.recording_id.strip()
    ):
        return batch.recording_id.strip()

    first_timestamp = float(
        min(
            sample.timestamp
            for sample in batch.samples
        )
    )

    last_timestamp = float(
        max(
            sample.timestamp
            for sample in batch.samples
        )
    )

    current = legacy_robot_recordings.get(
        batch.source_id
    )

    new_recording = (
        current is None
        or (
            first_timestamp
            - current.last_timestamp
            > LEGACY_RECORDING_BREAK_SECONDS
        )
        or (
            first_timestamp
            < current.last_timestamp
            - LEGACY_RECORDING_BREAK_SECONDS
        )
    )

    if new_recording:
        recording_id = (
            "robot_gt_"
            + str(
                int(
                    round(
                        first_timestamp
                        * 1_000_000.0
                    )
                )
            )
        )

        legacy_robot_recordings[
            batch.source_id
        ] = LegacyRobotRecordingState(
            recording_id=recording_id,
            last_timestamp=last_timestamp,
        )

        return recording_id

    current.last_timestamp = max(
        current.last_timestamp,
        last_timestamp,
    )

    return current.recording_id


def get_robot_recording_path(
    source_id: str,
    recording_id: str,
) -> Path:
    return (
        ROBOT_RECORDING_DIR
        / (
            f"{sanitize_filename_component(source_id)}__"
            f"{sanitize_filename_component(recording_id)}.npz"
        )
    )


def merge_robot_recording_arrays(
    old_timestamps: np.ndarray,
    old_positions: np.ndarray,
    old_headings: np.ndarray,
    new_timestamps: np.ndarray,
    new_positions: np.ndarray,
    new_headings: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    timestamps = np.concatenate(
        [old_timestamps, new_timestamps],
        axis=0,
    )

    positions = np.concatenate(
        [old_positions, new_positions],
        axis=0,
    )

    headings = np.concatenate(
        [old_headings, new_headings],
        axis=0,
    )

    order = np.argsort(
        timestamps,
        kind="stable",
    )

    timestamps = timestamps[order]
    positions = positions[order]
    headings = headings[order]

    # Keep final occurrence of duplicate timestamp.
    rev_times = timestamps[::-1]

    _, rev_unique = np.unique(
        rev_times,
        return_index=True,
    )

    keep = (
        len(timestamps)
        - 1
        - rev_unique
    )

    keep.sort()

    return (
        timestamps[keep],
        positions[keep],
        headings[keep],
    )


def save_robot_recording_batch(
    batch: RobotGroundTruthBatch,
    recording_id: str,
) -> Tuple[Path, int]:
    """
    Append this HTTP batch to a robot-only .npz recording.

    The values are saved exactly as sent. No coordinate conversion occurs.
    """
    path = get_robot_recording_path(
        batch.source_id,
        recording_id,
    )

    new_timestamps = np.asarray(
        [
            sample.timestamp
            for sample in batch.samples
        ],
        dtype=np.float64,
    )

    new_positions = vector3_list_to_array(
        [
            sample.position
            for sample in batch.samples
        ]
    )

    # Persist the robot heading exactly as received. Normalization is applied
    # only when headings are used for temporal interpolation.
    new_headings = vector3_list_to_array(
        [
            sample.heading
            for sample in batch.samples
        ]
    )

    if path.exists():
        with np.load(
            path,
            allow_pickle=False,
        ) as existing:
            old_timestamps = np.asarray(
                existing["robot_timestamps"],
                dtype=np.float64,
            )

            old_positions = np.asarray(
                existing["robot_positions"],
                dtype=np.float64,
            )

            old_headings = np.asarray(
                existing["robot_headings"],
                dtype=np.float64,
            )
    else:
        old_timestamps = np.empty(
            (0,),
            dtype=np.float64,
        )

        old_positions = np.empty(
            (0, 3),
            dtype=np.float64,
        )

        old_headings = np.empty(
            (0, 3),
            dtype=np.float64,
        )

    (
        timestamps,
        positions,
        headings,
    ) = merge_robot_recording_arrays(
        old_timestamps,
        old_positions,
        old_headings,
        new_timestamps,
        new_positions,
        new_headings,
    )

    relative_time = (
        timestamps
        - timestamps[0]
    )

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    try:
        with temp_path.open(
            "wb"
        ) as file:
            np.savez_compressed(
                file,
                schema_version=np.asarray(
                    batch.schema_version
                ),
                source_id=np.asarray(
                    batch.source_id
                ),
                recording_id=np.asarray(
                    recording_id
                ),
                sample_count=np.asarray(
                    len(timestamps),
                    dtype=np.int64,
                ),
                duration_seconds=np.asarray(
                    float(relative_time[-1])
                    if len(relative_time)
                    else 0.0,
                    dtype=np.float64,
                ),
                robot_timestamps=(
                    timestamps.astype(
                        np.float64
                    )
                ),
                robot_time_from_window_start=(
                    relative_time.astype(
                        np.float64
                    )
                ),
                robot_positions=(
                    positions.astype(
                        np.float32
                    )
                ),
                robot_headings=(
                    headings.astype(
                        np.float32
                    )
                ),
            )

        os.replace(
            temp_path,
            path,
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise

    return (
        path,
        len(timestamps),
    )


# ---------------------------------------------------------------------------
# Quest clock normalization
# ---------------------------------------------------------------------------

def validate_quest_window_timing(
    window: QuestMotionHistoryWindow,
    raw_timestamps: np.ndarray,
) -> None:
    if window.sample_count != len(window.samples):
        raise ValueError(
            "sample_count does not match the number of Quest samples."
        )

    if len(raw_timestamps) < 2:
        raise ValueError(
            "Quest motion window must contain at least two samples."
        )

    if not np.all(
        np.isfinite(raw_timestamps)
    ):
        raise ValueError(
            "Quest timestamps contain non-finite values."
        )

    if np.any(
        np.diff(raw_timestamps) <= 0.0
    ):
        raise ValueError(
            "Quest sample timestamps must be strictly increasing."
        )

    relative_times = np.asarray(
        [
            sample.time_from_window_start
            for sample in window.samples
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(relative_times)
    ):
        raise ValueError(
            "Quest time_from_window_start contains non-finite values."
        )

    if np.any(
        np.diff(relative_times) <= 0.0
    ):
        raise ValueError(
            "Quest time_from_window_start values must be strictly increasing."
        )

    raw_span = float(
        raw_timestamps[-1]
        - raw_timestamps[0]
    )

    relative_span = float(
        relative_times[-1]
        - relative_times[0]
    )

    expected_duration = float(
        window.window_duration_seconds
    )

    tolerance = max(
        QUEST_WINDOW_SPAN_ABSOLUTE_TOLERANCE_SECONDS,
        QUEST_WINDOW_SPAN_RELATIVE_TOLERANCE
        * max(
            relative_span,
            expected_duration,
            1.0,
        ),
    )

    if abs(
        raw_span
        - relative_span
    ) > tolerance:
        raise ValueError(
            "Quest raw timestamp span does not agree with "
            "time_from_window_start. "
            f"raw_span={raw_span:.3f}s, "
            f"relative_span={relative_span:.3f}s, "
            f"tolerance={tolerance:.3f}s."
        )

    if (
        expected_duration > 0.0
        and abs(
            expected_duration
            - relative_span
        ) > tolerance
    ):
        raise ValueError(
            "Quest window_duration_seconds does not agree with "
            "time_from_window_start. "
            f"reported={expected_duration:.3f}s, "
            f"relative_span={relative_span:.3f}s, "
            f"tolerance={tolerance:.3f}s."
        )


def reset_quest_clock_estimator(
    source_id: str,
) -> None:
    quest_clock_offset_candidates[
        source_id
    ].clear()

    quest_clock_offset_estimates.pop(
        source_id,
        None,
    )


def current_quest_offset_estimate(
    source_id: str,
) -> Optional[float]:
    candidates = (
        quest_clock_offset_candidates.get(
            source_id
        )
    )

    if not candidates:
        return None

    values = np.asarray(
        list(candidates),
        dtype=np.float64,
    )

    estimate = float(
        np.percentile(
            values,
            QUEST_OFFSET_PERCENTILE,
        )
    )

    quest_clock_offset_estimates[
        source_id
    ] = estimate

    return estimate


def resolve_image_alignment_timestamps(
    window: QuestMotionHistoryWindow,
    aligned_sample_timestamps: np.ndarray,
    clock_offset_seconds: float,
) -> np.ndarray:
    """
    Resolve one robot-clock-domain timestamp for every image.

    Priority:
      1. image.capture_timestamp_unix, if it looks like Unix time.
      2. image.capture_realtime_seconds + Quest clock offset.
      3. aligned timestamp of the first Quest sample referencing that frame.
    """
    first_reference_by_frame: Dict[
        int,
        float,
    ] = {}

    for sample_index, sample in enumerate(
        window.samples
    ):
        if (
            sample.has_image
            and sample.image_frame_id >= 0
            and sample.image_frame_id
            not in first_reference_by_frame
        ):
            first_reference_by_frame[
                int(sample.image_frame_id)
            ] = float(
                aligned_sample_timestamps[
                    sample_index
                ]
            )

    resolved = []

    for image in window.images:
        capture_unix = float(
            image.capture_timestamp_unix
        )

        capture_realtime = float(
            image.capture_realtime_seconds
        )

        if looks_like_unix_timestamp(
            capture_unix
        ):
            timestamp = capture_unix

        elif math.isfinite(
            capture_realtime
        ):
            timestamp = (
                capture_realtime
                + clock_offset_seconds
            )

        elif image.frame_id in first_reference_by_frame:
            timestamp = (
                first_reference_by_frame[
                    image.frame_id
                ]
            )

        else:
            raise ValueError(
                "Could not determine an alignment timestamp for "
                f"Quest image frame {image.frame_id}."
            )

        resolved.append(
            timestamp
        )

    return np.asarray(
        resolved,
        dtype=np.float64,
    )


def prepare_quest_window(
    window: QuestMotionHistoryWindow,
    server_receive_unix_seconds: float,
) -> PreparedQuestWindow:
    raw_timestamps = np.asarray(
        [
            sample.timestamp
            for sample in window.samples
        ],
        dtype=np.float64,
    )

    validate_quest_window_timing(
        window,
        raw_timestamps,
    )

    median_timestamp = float(
        np.median(
            raw_timestamps
        )
    )

    if looks_like_unix_timestamp(
        median_timestamp
    ):
        aligned_timestamps = (
            raw_timestamps.copy()
        )

        timestamp_mode = "unix"
        offset = 0.0
        candidate = 0.0
        reference = "quest_sample_timestamp"

    else:
        previous_raw_end = (
            quest_clock_last_raw_timestamp.get(
                window.source_id
            )
        )

        if (
            previous_raw_end is not None
            and raw_timestamps[0]
            < previous_raw_end - 1.0
        ):
            reset_quest_clock_estimator(
                window.source_id
            )

        if looks_like_unix_timestamp(
            float(
                window.sent_at_unix_seconds
            )
        ):
            unix_reference = float(
                window.sent_at_unix_seconds
            )

            reference = (
                "window.sent_at_unix_seconds"
            )
        else:
            unix_reference = float(
                server_receive_unix_seconds
            )

            reference = (
                "server_receive_unix_seconds"
            )

        candidate = (
            unix_reference
            - float(
                raw_timestamps[-1]
            )
        )

        previous_estimate = (
            quest_clock_offset_estimates.get(
                window.source_id
            )
        )

        if (
            previous_estimate is not None
            and abs(
                candidate
                - previous_estimate
            )
            > QUEST_CLOCK_RESET_THRESHOLD_SECONDS
        ):
            reset_quest_clock_estimator(
                window.source_id
            )

        quest_clock_offset_candidates[
            window.source_id
        ].append(
            candidate
        )

        offset = (
            current_quest_offset_estimate(
                window.source_id
            )
        )

        if offset is None:
            raise ValueError(
                "Could not estimate Quest->Unix clock offset."
            )

        aligned_timestamps = (
            raw_timestamps
            + offset
        )

        timestamp_mode = (
            "rebased_monotonic"
        )

    quest_clock_last_raw_timestamp[
        window.source_id
    ] = float(
        raw_timestamps[-1]
    )

    image_alignment_timestamps = (
        resolve_image_alignment_timestamps(
            window=window,
            aligned_sample_timestamps=(
                aligned_timestamps
            ),
            clock_offset_seconds=offset,
        )
    )

    return PreparedQuestWindow(
        window=window,
        original_timestamps=(
            raw_timestamps.copy()
        ),
        aligned_timestamps=(
            aligned_timestamps
        ),
        timestamp_mode=timestamp_mode,
        clock_offset_seconds=float(
            offset
        ),
        clock_offset_candidate_seconds=float(
            candidate
        ),
        clock_offset_reference=reference,
        image_alignment_timestamps=(
            image_alignment_timestamps
        ),
    )


# ---------------------------------------------------------------------------
# Image validation/storage
# ---------------------------------------------------------------------------

def validate_and_decode_images(
    window: QuestMotionHistoryWindow,
) -> Dict[int, bytes]:
    if window.image_count != len(
        window.images
    ):
        raise ValueError(
            "image_count does not match the number of images."
        )

    if len(window.images) > MAX_IMAGES_PER_WINDOW:
        raise ValueError(
            "Motion window contains too many images: "
            f"{len(window.images)} > {MAX_IMAGES_PER_WINDOW}."
        )

    if (
        window.includes_images
        and not window.images
    ):
        raise ValueError(
            "includes_images is true but no images were supplied."
        )

    decoded: Dict[
        int,
        bytes,
    ] = {}

    for image in window.images:
        frame_id = int(
            image.frame_id
        )

        if frame_id in decoded:
            raise ValueError(
                f"Duplicate image frame_id {frame_id}."
            )

        if image.mime_type.lower() not in {
            "image/jpeg",
            "image/jpg",
        }:
            raise ValueError(
                f"Unsupported image MIME type {image.mime_type!r}; "
                "only JPEG is accepted."
            )

        if (
            image.camera_position is None
        ) != (
            image.camera_rotation is None
        ):
            raise ValueError(
                f"Image frame {frame_id} must provide both camera_position "
                "and camera_rotation, or neither."
            )

        if image.camera_rotation is not None:
            quaternion = np.asarray(
                [
                    image.camera_rotation.x,
                    image.camera_rotation.y,
                    image.camera_rotation.z,
                    image.camera_rotation.w,
                ],
                dtype=np.float64,
            )

            if (
                not np.all(
                    np.isfinite(
                        quaternion
                    )
                )
                or np.linalg.norm(
                    quaternion
                ) < 1.0e-8
            ):
                raise ValueError(
                    f"Image frame {frame_id} has an invalid camera quaternion."
                )

        try:
            jpeg_bytes = (
                base64.b64decode(
                    image.jpeg_base64,
                    validate=True,
                )
            )
        except (
            binascii.Error,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Invalid Base64 JPEG for image frame {frame_id}."
            ) from exc

        if not jpeg_bytes:
            raise ValueError(
                f"Decoded JPEG for image frame {frame_id} is empty."
            )

        if len(
            jpeg_bytes
        ) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image frame {frame_id} exceeds the image-size limit."
            )

        if not jpeg_bytes.startswith(
            b"\xff\xd8"
        ):
            raise ValueError(
                f"Image frame {frame_id} does not look like JPEG data."
            )

        decoded[
            frame_id
        ] = jpeg_bytes

    known_frame_ids = set(
        decoded.keys()
    )

    referenced_frame_ids = set()

    for index, sample in enumerate(
        window.samples
    ):
        if not sample.has_image:
            continue

        if sample.image_frame_id < 0:
            raise ValueError(
                f"Quest sample {index} has_image=true but image_frame_id < 0."
            )

        if (
            sample.image_frame_id
            not in known_frame_ids
        ):
            raise ValueError(
                f"Quest sample {index} references image frame "
                f"{sample.image_frame_id}, which is absent."
            )

        referenced_frame_ids.add(
            int(
                sample.image_frame_id
            )
        )

    unreferenced = (
        known_frame_ids
        - referenced_frame_ids
    )

    if unreferenced:
        raise ValueError(
            "Quest payload contains unreferenced image frames: "
            f"{sorted(unreferenced)}."
        )

    return decoded


def save_window_images(
    window: QuestMotionHistoryWindow,
    decoded_images: Dict[int, bytes],
) -> Optional[Path]:
    if not decoded_images:
        return None

    final_directory = (
        get_window_image_directory(
            window.source_id,
            window.window_id,
        )
    )

    if final_directory.exists():
        return final_directory

    final_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_directory = Path(
        tempfile.mkdtemp(
            prefix=".quest_images_",
            dir=final_directory.parent,
        )
    )

    try:
        manifest_images = []

        for image in window.images:
            frame_id = int(
                image.frame_id
            )

            filename = (
                f"frame_{frame_id:020d}.jpg"
            )

            (
                temp_directory
                / filename
            ).write_bytes(
                decoded_images[
                    frame_id
                ]
            )

            manifest_images.append(
                {
                    "frame_id": frame_id,
                    "file_name": filename,
                    "mime_type": image.mime_type,
                    "image_width": int(
                        image.image_width
                    ),
                    "image_height": int(
                        image.image_height
                    ),
                    "capture_timestamp_unix": float(
                        image.capture_timestamp_unix
                    ),
                    "capture_realtime_seconds": float(
                        image.capture_realtime_seconds
                    ),
                }
            )

        manifest = {
            "schema_version": (
                window.schema_version
            ),
            "window_id": window.window_id,
            "source_id": window.source_id,
            "image_count": len(
                window.images
            ),
            "images": manifest_images,
        }

        (
            temp_directory
            / "manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        os.replace(
            temp_directory,
            final_directory,
        )

    except Exception:
        # Best-effort cleanup for an incomplete temp directory.
        for child in (
            temp_directory.iterdir()
            if temp_directory.exists()
            else []
        ):
            try:
                child.unlink()
            except Exception:
                pass

        try:
            temp_directory.rmdir()
        except Exception:
            pass

        raise

    return final_directory


# ---------------------------------------------------------------------------
# Temporal interpolation
# ---------------------------------------------------------------------------

def get_required_alignment_times(
    prepared: PreparedQuestWindow,
) -> np.ndarray:
    if len(
        prepared.image_alignment_timestamps
    ):
        return np.concatenate(
            [
                prepared.aligned_timestamps,
                prepared.image_alignment_timestamps,
            ]
        )

    return (
        prepared.aligned_timestamps
    )


def has_robot_coverage(
    prepared: PreparedQuestWindow,
) -> bool:
    samples = (
        robot_ground_truth_buffers.get(
            prepared.window.robot_source_id,
            [],
        )
    )

    if len(samples) < 2:
        return False

    required_times = (
        get_required_alignment_times(
            prepared
        )
    )

    if not len(required_times):
        return False

    return (
        float(samples[0].timestamp)
        <= float(required_times.min())
        and float(samples[-1].timestamp)
        >= float(required_times.max())
    )


def nearest_timestamp_errors(
    query_times: np.ndarray,
    source_times: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(
        source_times,
        query_times,
        side="left",
    )

    errors = np.empty(
        len(query_times),
        dtype=np.float64,
    )

    for i, index in enumerate(indices):
        candidates = []

        if index < len(source_times):
            candidates.append(
                abs(
                    float(
                        source_times[index]
                        - query_times[i]
                    )
                )
            )

        if index > 0:
            candidates.append(
                abs(
                    float(
                        source_times[index - 1]
                        - query_times[i]
                    )
                )
            )

        errors[i] = min(
            candidates
        )

    return errors


def interpolation_bracket_gaps(
    query_times: np.ndarray,
    source_times: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(
        source_times,
        query_times,
        side="left",
    )

    gaps = np.zeros(
        len(query_times),
        dtype=np.float64,
    )

    for i, index in enumerate(indices):
        if (
            index < len(source_times)
            and math.isclose(
                float(
                    source_times[index]
                ),
                float(
                    query_times[i]
                ),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            gaps[i] = 0.0
            continue

        if (
            index == 0
            or index >= len(source_times)
        ):
            gaps[i] = np.inf
            continue

        gaps[i] = (
            source_times[index]
            - source_times[index - 1]
        )

    return gaps


def interpolate_robot_at_times(
    robot_source_id: str,
    query_times: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Interpolate raw robot GT at arbitrary timestamps.

    Position:
        component-wise linear interpolation.

    Heading:
        component-wise linear interpolation followed by 3-D normalization.

    The heading operation is intentionally not expressed as yaw, so this server
    does not assume ROS XY, Unity XZ, or any other particular ground plane.
    """
    samples = (
        robot_ground_truth_buffers.get(
            robot_source_id,
            [],
        )
    )

    if len(samples) < 2:
        raise ValueError(
            "Fewer than two robot GT samples are available."
        )

    robot_times = np.asarray(
        [
            sample.timestamp
            for sample in samples
        ],
        dtype=np.float64,
    )

    query_times = np.asarray(
        query_times,
        dtype=np.float64,
    )

    if (
        float(query_times.min())
        < float(robot_times[0])
        or float(query_times.max())
        > float(robot_times[-1])
    ):
        raise ValueError(
            "Robot GT buffer does not cover the complete requested time range."
        )

    robot_positions = vector3_list_to_array(
        [
            sample.position
            for sample in samples
        ]
    )

    robot_headings = normalize_direction_rows(
        vector3_list_to_array(
            [
                sample.heading
                for sample in samples
            ]
        )
    )

    positions = np.column_stack(
        [
            np.interp(
                query_times,
                robot_times,
                robot_positions[:, axis],
            )
            for axis in range(3)
        ]
    )

    raw_headings = np.column_stack(
        [
            np.interp(
                query_times,
                robot_times,
                robot_headings[:, axis],
            )
            for axis in range(3)
        ]
    )

    headings = normalize_direction_rows(
        raw_headings
    )

    nearest_errors = (
        nearest_timestamp_errors(
            query_times,
            robot_times,
        )
    )

    bracket_gaps = (
        interpolation_bracket_gaps(
            query_times,
            robot_times,
        )
    )

    left_index = max(
        0,
        int(
            np.searchsorted(
                robot_times,
                query_times.min(),
                side="left",
            )
        ) - 1,
    )

    right_index = min(
        len(robot_times),
        int(
            np.searchsorted(
                robot_times,
                query_times.max(),
                side="right",
            )
        ) + 1,
    )

    return {
        "timestamps": (
            query_times.copy()
        ),
        "positions": positions,
        "headings": headings,
        "nearest_timestamp_errors": (
            nearest_errors
        ),
        "bracket_gaps": bracket_gaps,
        "source_times": (
            robot_times[
                left_index:right_index
            ]
        ),
        "source_positions": (
            robot_positions[
                left_index:right_index
            ]
        ),
        "source_headings": (
            robot_headings[
                left_index:right_index
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Saving aligned Quest windows
# ---------------------------------------------------------------------------

def build_image_metadata(
    prepared: PreparedQuestWindow,
    image_alignment: Dict[
        str,
        np.ndarray,
    ],
) -> Dict[str, np.ndarray]:
    window = prepared.window

    frame_ids = np.asarray(
        [
            int(image.frame_id)
            for image in window.images
        ],
        dtype=np.int64,
    )

    widths = np.asarray(
        [
            int(image.image_width)
            for image in window.images
        ],
        dtype=np.int32,
    )

    heights = np.asarray(
        [
            int(image.image_height)
            for image in window.images
        ],
        dtype=np.int32,
    )

    capture_unix = np.asarray(
        [
            float(
                image.capture_timestamp_unix
            )
            for image in window.images
        ],
        dtype=np.float64,
    )

    capture_realtime = np.asarray(
        [
            float(
                image.capture_realtime_seconds
            )
            for image in window.images
        ],
        dtype=np.float64,
    )

    camera_positions = np.asarray(
        [
            (
                [
                    image.camera_position.x,
                    image.camera_position.y,
                    image.camera_position.z,
                ]
                if image.camera_position
                is not None
                else [
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            )
            for image in window.images
        ],
        dtype=np.float32,
    ).reshape((-1, 3))

    camera_rotations = np.asarray(
        [
            (
                [
                    image.camera_rotation.x,
                    image.camera_rotation.y,
                    image.camera_rotation.z,
                    image.camera_rotation.w,
                ]
                if image.camera_rotation
                is not None
                else [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            )
            for image in window.images
        ],
        dtype=np.float32,
    ).reshape((-1, 4))

    has_camera_pose = np.asarray(
        [
            (
                image.camera_position
                is not None
                and image.camera_rotation
                is not None
            )
            for image in window.images
        ],
        dtype=np.bool_,
    )

    relative_paths = np.asarray(
        [
            (
                Path("images")
                / sanitize_filename_component(
                    window.source_id
                )
                / sanitize_filename_component(
                    window.window_id
                )
                / (
                    f"frame_"
                    f"{int(image.frame_id):020d}"
                    f".jpg"
                )
            ).as_posix()
            for image in window.images
        ],
        dtype=str,
    )

    return {
        "frame_ids": frame_ids,
        "widths": widths,
        "heights": heights,
        "capture_unix": capture_unix,
        "capture_realtime": capture_realtime,
        "alignment_timestamps": (
            prepared.image_alignment_timestamps.astype(
                np.float64
            )
        ),
        "camera_positions": (
            camera_positions
        ),
        "camera_rotations_xyzw": (
            camera_rotations
        ),
        "has_camera_pose": (
            has_camera_pose
        ),
        "relative_paths": (
            relative_paths
        ),
        "robot_positions": (
            image_alignment[
                "positions"
            ].astype(
                np.float32
            )
        ),
        "robot_headings": (
            image_alignment[
                "headings"
            ].astype(
                np.float32
            )
        ),
        "nearest_robot_timestamp_error_seconds": (
            image_alignment[
                "nearest_timestamp_errors"
            ].astype(
                np.float64
            )
        ),
        "robot_interpolation_bracket_gap_seconds": (
            image_alignment[
                "bracket_gaps"
            ].astype(
                np.float64
            )
        ),
    }


def save_aligned_window(
    prepared: PreparedQuestWindow,
    decoded_images: Dict[
        int,
        bytes,
    ],
) -> Path:
    window = prepared.window

    quest_alignment = (
        interpolate_robot_at_times(
            robot_source_id=(
                window.robot_source_id
            ),
            query_times=(
                prepared.aligned_timestamps
            ),
        )
    )

    if len(
        prepared.image_alignment_timestamps
    ):
        image_alignment = (
            interpolate_robot_at_times(
                robot_source_id=(
                    window.robot_source_id
                ),
                query_times=(
                    prepared.image_alignment_timestamps
                ),
            )
        )
    else:
        image_alignment = {
            "positions": np.empty(
                (0, 3),
                dtype=np.float64,
            ),
            "headings": np.empty(
                (0, 3),
                dtype=np.float64,
            ),
            "nearest_timestamp_errors": (
                np.empty(
                    (0,),
                    dtype=np.float64,
                )
            ),
            "bracket_gaps": np.empty(
                (0,),
                dtype=np.float64,
            ),
        }

    image_directory = (
        save_window_images(
            window,
            decoded_images,
        )
    )

    image_metadata = (
        build_image_metadata(
            prepared,
            image_alignment,
        )
    )

    quest_positions = (
        vector3_list_to_array(
            [
                sample.position
                for sample in window.samples
            ]
        )
    )

    quest_headings = (
        vector3_list_to_array(
            [
                sample.heading
                for sample in window.samples
            ]
        )
    )

    quest_time_from_start = np.asarray(
        [
            sample.time_from_window_start
            for sample in window.samples
        ],
        dtype=np.float64,
    )

    sample_has_image = np.asarray(
        [
            bool(
                sample.has_image
            )
            for sample in window.samples
        ],
        dtype=np.bool_,
    )

    sample_image_frame_ids = np.asarray(
        [
            (
                int(
                    sample.image_frame_id
                )
                if sample.has_image
                else -1
            )
            for sample in window.samples
        ],
        dtype=np.int64,
    )

    output_path = (
        get_aligned_window_path(
            window.source_id,
            window.window_id,
        )
    )

    temp_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    max_sample_gap = (
        float(
            np.max(
                quest_alignment[
                    "bracket_gaps"
                ]
            )
        )
        if len(
            quest_alignment[
                "bracket_gaps"
            ]
        )
        else 0.0
    )

    try:
        with temp_path.open(
            "wb"
        ) as file:
            np.savez_compressed(
                file,
                schema_version=np.asarray(
                    window.schema_version
                ),
                window_id=np.asarray(
                    window.window_id
                ),
                quest_source_id=np.asarray(
                    window.source_id
                ),
                robot_source_id=np.asarray(
                    window.robot_source_id
                ),

                # Quest observations.
                quest_original_timestamps=(
                    prepared.original_timestamps.astype(
                        np.float64
                    )
                ),
                quest_timestamps=(
                    prepared.aligned_timestamps.astype(
                        np.float64
                    )
                ),
                quest_time_from_window_start=(
                    quest_time_from_start
                ),
                quest_positions=(
                    quest_positions.astype(
                        np.float32
                    )
                ),
                quest_headings=(
                    quest_headings.astype(
                        np.float32
                    )
                ),
                quest_sample_has_image=(
                    sample_has_image
                ),
                quest_sample_image_frame_ids=(
                    sample_image_frame_ids
                ),

                # Robot GT aligned to every Quest observation.
                robot_ground_truth_timestamps=(
                    prepared.aligned_timestamps.astype(
                        np.float64
                    )
                ),
                robot_ground_truth_positions=(
                    quest_alignment[
                        "positions"
                    ].astype(
                        np.float32
                    )
                ),
                robot_ground_truth_headings=(
                    quest_alignment[
                        "headings"
                    ].astype(
                        np.float32
                    )
                ),
                robot_nearest_timestamp_error_seconds=(
                    quest_alignment[
                        "nearest_timestamp_errors"
                    ].astype(
                        np.float64
                    )
                ),
                robot_interpolation_bracket_gap_seconds=(
                    quest_alignment[
                        "bracket_gaps"
                    ].astype(
                        np.float64
                    )
                ),

                # Raw robot samples surrounding this aligned window.
                robot_source_timestamps=(
                    quest_alignment[
                        "source_times"
                    ].astype(
                        np.float64
                    )
                ),
                robot_source_positions=(
                    quest_alignment[
                        "source_positions"
                    ].astype(
                        np.float32
                    )
                ),
                robot_source_headings=(
                    quest_alignment[
                        "source_headings"
                    ].astype(
                        np.float32
                    )
                ),

                # Per-image metadata + robot GT aligned at each image capture.
                quest_image_frame_ids=(
                    image_metadata[
                        "frame_ids"
                    ]
                ),
                quest_image_widths=(
                    image_metadata[
                        "widths"
                    ]
                ),
                quest_image_heights=(
                    image_metadata[
                        "heights"
                    ]
                ),
                quest_image_capture_timestamp_unix=(
                    image_metadata[
                        "capture_unix"
                    ]
                ),
                quest_image_capture_realtime_seconds=(
                    image_metadata[
                        "capture_realtime"
                    ]
                ),
                quest_image_alignment_timestamps=(
                    image_metadata[
                        "alignment_timestamps"
                    ]
                ),
                quest_image_has_camera_pose=(
                    image_metadata[
                        "has_camera_pose"
                    ]
                ),
                quest_image_camera_positions=(
                    image_metadata[
                        "camera_positions"
                    ]
                ),
                quest_image_camera_rotations_xyzw=(
                    image_metadata[
                        "camera_rotations_xyzw"
                    ]
                ),
                quest_image_relative_paths=(
                    image_metadata[
                        "relative_paths"
                    ]
                ),
                quest_image_robot_ground_truth_positions=(
                    image_metadata[
                        "robot_positions"
                    ]
                ),
                quest_image_robot_ground_truth_headings=(
                    image_metadata[
                        "robot_headings"
                    ]
                ),
                quest_image_nearest_robot_timestamp_error_seconds=(
                    image_metadata[
                        "nearest_robot_timestamp_error_seconds"
                    ]
                ),
                quest_image_robot_interpolation_bracket_gap_seconds=(
                    image_metadata[
                        "robot_interpolation_bracket_gap_seconds"
                    ]
                ),

                # Clock/alignment diagnostics.
                quest_timestamp_mode=np.asarray(
                    prepared.timestamp_mode
                ),
                quest_clock_offset_seconds=np.asarray(
                    prepared.clock_offset_seconds,
                    dtype=np.float64,
                ),
                quest_clock_offset_candidate_seconds=np.asarray(
                    prepared.clock_offset_candidate_seconds,
                    dtype=np.float64,
                ),
                quest_clock_offset_reference=np.asarray(
                    prepared.clock_offset_reference
                ),
                max_robot_interpolation_gap_seconds=np.asarray(
                    max_sample_gap,
                    dtype=np.float64,
                ),
                robot_ground_truth_coordinate_frame=np.asarray(
                    "as_sent_by_robot"
                ),
                spatial_transform_applied=np.asarray(
                    False,
                    dtype=np.bool_,
                ),
            )

        os.replace(
            temp_path,
            output_path,
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise

    if (
        max_sample_gap
        > ALIGNMENT_GAP_WARNING_SECONDS
    ):
        print(
            "WARNING: saved aligned Quest window "
            f"{window.window_id!r} with max robot interpolation "
            f"gap {max_sample_gap:.3f}s.",
            flush=True,
        )

    print(
        f"Saved aligned Quest window: {output_path}",
        flush=True,
    )

    if image_directory is not None:
        print(
            f"Saved Quest images: {image_directory}",
            flush=True,
        )

    return output_path


def try_save_prepared_window(
    prepared: PreparedQuestWindow,
    decoded_images: Optional[
        Dict[int, bytes]
    ] = None,
) -> Optional[Path]:
    if not has_robot_coverage(
        prepared
    ):
        return None

    if decoded_images is None:
        decoded_images = (
            validate_and_decode_images(
                prepared.window
            )
        )

    return save_aligned_window(
        prepared,
        decoded_images,
    )


def try_save_pending_windows_for_robot(
    robot_source_id: str,
) -> List[Path]:
    saved_paths: List[
        Path
    ] = []

    matching_keys = [
        key
        for key, prepared
        in pending_quest_windows.items()
        if (
            prepared.window.robot_source_id
            == robot_source_id
        )
    ]

    for key in matching_keys:
        prepared = (
            pending_quest_windows[
                key
            ]
        )

        output_path = (
            get_aligned_window_path(
                prepared.window.source_id,
                prepared.window.window_id,
            )
        )

        if output_path.exists():
            del pending_quest_windows[
                key
            ]
            continue

        saved = (
            try_save_prepared_window(
                prepared
            )
        )

        if saved is not None:
            saved_paths.append(
                saved
            )

            del pending_quest_windows[
                key
            ]

    return saved_paths


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    with state_lock:
        robot_sources = {
            source_id: len(samples)
            for source_id, samples
            in robot_ground_truth_buffers.items()
        }

        return {
            "status": "ok",
            "project_root": str(
                PROJECT_ROOT
            ),
            "data_directory": str(
                DATA_DIR
            ),
            "robot_recording_directory": str(
                ROBOT_RECORDING_DIR
            ),
            "aligned_window_directory": str(
                ALIGNED_WINDOW_DIR
            ),
            "image_directory": str(
                IMAGE_DATA_DIR
            ),
            "robot_buffer_sizes": (
                robot_sources
            ),
            "pending_quest_windows": len(
                pending_quest_windows
            ),
        }


@app.get("/collection_status")
def collection_status():
    with state_lock:
        robot_sources = {}

        for source_id, samples in (
            robot_ground_truth_buffers.items()
        ):
            if samples:
                robot_sources[
                    source_id
                ] = {
                    "sample_count": len(
                        samples
                    ),
                    "start_timestamp": float(
                        samples[0].timestamp
                    ),
                    "end_timestamp": float(
                        samples[-1].timestamp
                    ),
                    "duration_seconds": float(
                        samples[-1].timestamp
                        - samples[0].timestamp
                    ),
                }
            else:
                robot_sources[
                    source_id
                ] = {
                    "sample_count": 0
                }

        pending = [
            {
                "window_id": (
                    prepared.window.window_id
                ),
                "quest_source_id": (
                    prepared.window.source_id
                ),
                "robot_source_id": (
                    prepared.window.robot_source_id
                ),
                "aligned_start_timestamp": float(
                    get_required_alignment_times(
                        prepared
                    ).min()
                ),
                "aligned_end_timestamp": float(
                    get_required_alignment_times(
                        prepared
                    ).max()
                ),
                "images": len(
                    prepared.window.images
                ),
            }
            for prepared
            in pending_quest_windows.values()
        ]

        return {
            "robot_sources": robot_sources,
            "pending_quest_windows": pending,
            "quest_clock_offset_estimates": dict(
                quest_clock_offset_estimates
            ),
        }


@app.post("/robot_ground_truth")
def receive_robot_ground_truth(
    batch: RobotGroundTruthBatch,
):
    try:
        validate_robot_batch(
            batch
        )

        with state_lock:
            recording_id = (
                resolve_robot_recording_id(
                    batch
                )
            )

            recording_path, recording_count = (
                save_robot_recording_batch(
                    batch,
                    recording_id,
                )
            )

            robot_ground_truth_buffers[
                batch.source_id
            ].extend(
                batch.samples
            )

            sort_and_deduplicate_robot_buffer(
                batch.source_id
            )

            saved_pending = (
                try_save_pending_windows_for_robot(
                    batch.source_id
                )
            )

            prune_robot_buffer(
                batch.source_id
            )

            buffer = (
                robot_ground_truth_buffers[
                    batch.source_id
                ]
            )

            buffer_start = (
                float(
                    buffer[0].timestamp
                )
                if buffer
                else None
            )

            buffer_end = (
                float(
                    buffer[-1].timestamp
                )
                if buffer
                else None
            )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to store robot ground-truth batch."
            ),
        ) from exc

    return {
        "accepted": True,
        "source_id": batch.source_id,
        "recording_id": recording_id,
        "samples_received": len(
            batch.samples
        ),
        "samples_in_recording": (
            recording_count
        ),
        "recording_path": str(
            recording_path
        ),
        "buffer_start_timestamp": (
            buffer_start
        ),
        "buffer_end_timestamp": (
            buffer_end
        ),

        # Kept for compatibility with the existing sender's debug message.
        "pending_windows_saved": len(
            saved_pending
        ),
        "saved_paths": [
            str(path)
            for path in saved_pending
        ],
    }


@app.post("/robot-motion-history")
@app.post("/robot_motion_history")
def receive_quest_motion_history(
    window: QuestMotionHistoryWindow,
):
    receive_time = time.time()

    output_path = (
        get_aligned_window_path(
            window.source_id,
            window.window_id,
        )
    )

    pending_key = (
        window.source_id,
        window.window_id,
    )

    try:
        # Validate image payload immediately so malformed payloads do not sit in
        # the pending queue.
        decoded_images = (
            validate_and_decode_images(
                window
            )
        )

        with state_lock:
            if output_path.exists():
                return {
                    "accepted": True,
                    "duplicate": True,
                    "aligned": True,
                    "pending": False,
                    "window_id": (
                        window.window_id
                    ),
                    "samples_received": len(
                        window.samples
                    ),
                    "images_received": len(
                        window.images
                    ),
                    "saved_path": str(
                        output_path
                    ),
                }

            if (
                pending_key
                in pending_quest_windows
            ):
                existing = (
                    pending_quest_windows[
                        pending_key
                    ]
                )

                return {
                    "accepted": True,
                    "duplicate": True,
                    "aligned": False,
                    "pending": True,
                    "window_id": (
                        window.window_id
                    ),
                    "aligned_start_timestamp": float(
                        get_required_alignment_times(
                            existing
                        ).min()
                    ),
                    "aligned_end_timestamp": float(
                        get_required_alignment_times(
                            existing
                        ).max()
                    ),
                }

            prepared = (
                prepare_quest_window(
                    window=window,
                    server_receive_unix_seconds=(
                        receive_time
                    ),
                )
            )

            saved_path = (
                try_save_prepared_window(
                    prepared,
                    decoded_images=(
                        decoded_images
                    ),
                )
            )

            if saved_path is not None:
                return {
                    "accepted": True,
                    "duplicate": False,
                    "aligned": True,
                    "pending": False,
                    "window_id": (
                        window.window_id
                    ),
                    "samples_received": len(
                        window.samples
                    ),
                    "images_received": len(
                        window.images
                    ),
                    "timestamp_mode": (
                        prepared.timestamp_mode
                    ),
                    "clock_offset_seconds": (
                        prepared.clock_offset_seconds
                    ),
                    "aligned_start_timestamp": float(
                        get_required_alignment_times(
                            prepared
                        ).min()
                    ),
                    "aligned_end_timestamp": float(
                        get_required_alignment_times(
                            prepared
                        ).max()
                    ),
                    "saved_path": str(
                        saved_path
                    ),
                    "image_directory": (
                        str(
                            get_window_image_directory(
                                window.source_id,
                                window.window_id,
                            )
                        )
                        if window.images
                        else None
                    ),
                }

            pending_quest_windows[
                pending_key
            ] = prepared

            return {
                "accepted": True,
                "duplicate": False,
                "aligned": False,
                "pending": True,
                "window_id": (
                    window.window_id
                ),
                "samples_received": len(
                    window.samples
                ),
                "images_received": len(
                    window.images
                ),
                "timestamp_mode": (
                    prepared.timestamp_mode
                ),
                "clock_offset_seconds": (
                    prepared.clock_offset_seconds
                ),
                "clock_offset_reference": (
                    prepared.clock_offset_reference
                ),
                "aligned_start_timestamp": float(
                    get_required_alignment_times(
                        prepared
                    ).min()
                ),
                "aligned_end_timestamp": float(
                    get_required_alignment_times(
                        prepared
                    ).max()
                ),
                "detail": (
                    "Quest window accepted and buffered until robot "
                    "ground truth covers the complete observation/image "
                    "timestamp range."
                ),
            }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process Quest motion/image window."
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw robot ground truth and temporally align "
            "Quest observations/JPEG images."
        )
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=[
            "critical",
            "error",
            "warning",
            "info",
            "debug",
            "trace",
        ],
    )

    args = parser.parse_args()

    import uvicorn

    # Passing the app object works cleanly with `python -m` and avoids requiring
    # the caller to know the import string.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
