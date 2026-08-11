import asyncio
import math
import os
import re
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Robot Motion Alignment Server",
    version="3.0.0",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Keep enough robot ground-truth history to cover delayed Quest windows.
ROBOT_BUFFER_SECONDS = 300.0

# This does not prevent saving. It is recorded as a warning/diagnostic if
# interpolation has to bridge a larger-than-expected robot sampling gap.
ALIGNMENT_GAP_WARNING_SECONDS = 0.25

# Timestamps above this threshold are treated as Unix/wall-clock seconds.
# Current Unix timestamps are comfortably above 1e9.
UNIX_TIMESTAMP_THRESHOLD = 1.0e9

# Number of recent Quest->Unix clock-offset observations retained per source.
QUEST_OFFSET_HISTORY_SIZE = 100

# Use a low percentile rather than the mean because receive/send latency can
# only make an arrival-based offset estimate later, not earlier.
QUEST_OFFSET_PERCENTILE = 10.0

# If a new offset candidate suddenly differs from the current estimate by this
# much, assume Unity restarted or the clock domain changed and reset the
# offset estimator for that Quest source.
QUEST_CLOCK_RESET_THRESHOLD_SECONDS = 5.0

# Sanity check for the raw Quest timestamp span versus
# time_from_window_start/window_duration_seconds.
QUEST_WINDOW_SPAN_ABSOLUTE_TOLERANCE_SECONDS = 0.50
QUEST_WINDOW_SPAN_RELATIVE_TOLERANCE = 0.10


# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """
    Find the UnitreeGo2HandoffIntentDetector project directory.
    """
    script_dir = Path(__file__).resolve().parent

    for parent in [script_dir, *script_dir.parents]:
        if parent.name == "UnitreeGo2HandoffIntentDetector":
            return parent

        candidate = parent / "UnitreeGo2HandoffIntentDetector"
        if candidate.is_dir():
            return candidate

    return Path.cwd() / "UnitreeGo2HandoffIntentDetector"


PROJECT_ROOT = find_project_root()

DATA_DIR = (
    PROJECT_ROOT
    / "robot_pose_prediction"
    / "data"
)

DATA_DIR.mkdir(
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


class RobotMotionSample(BaseModel):
    """
    Quest-estimated robot motion sample.

    `timestamp` may be either:
      1. Unix seconds, or
      2. Unity/Quest monotonic elapsed seconds.

    If it is monotonic elapsed time, this server automatically rebases it
    into Unix time before aligning against robot ground truth.

    `time_from_window_start` remains the relative time within the motion
    window and is never rebased.
    """
    timestamp: float
    time_from_window_start: float

    position: Vector3
    heading: Vector3


class RobotMotionHistoryWindow(BaseModel):
    schema_version: str

    window_id: str
    source_id: str

    # Optional identifier for the robot ground-truth stream to align against.
    # Existing Unity payloads do not need to send this field if the robot
    # ground-truth sender uses the default "unitree_go2".
    robot_source_id: str = "unitree_go2"

    # If Unity already sends a real Unix timestamp here, it is preferred as
    # the clock-offset reference because it avoids network receive latency.
    # If it does not look like Unix time, the server's receive time is used.
    sent_at_unix_seconds: float

    window_duration_seconds: float
    sample_count: int = Field(ge=1)

    samples: List[RobotMotionSample]


class RobotGroundTruthSample(BaseModel):
    """
    Ground-truth robot pose sample.

    timestamp must be Unix/wall-clock seconds in the same clock domain as
    the server-rebased Quest timestamps.

    heading should be the robot/camera forward vector in the ground-truth
    world frame.
    """
    timestamp: float
    position: Vector3
    heading: Vector3


class RobotGroundTruthBatch(BaseModel):
    schema_version: str = "1.0"
    source_id: str = "unitree_go2"
    sample_count: int = Field(ge=1)
    samples: List[RobotGroundTruthSample]


# ---------------------------------------------------------------------------
# Internal Quest-window representation
# ---------------------------------------------------------------------------

@dataclass
class PreparedQuestWindow:
    """
    Quest window after timestamp normalization.

    The Pydantic `window` retains the original payload unchanged.
    """
    window: RobotMotionHistoryWindow

    original_timestamps: np.ndarray
    aligned_timestamps: np.ndarray

    timestamp_mode: str
    clock_offset_seconds: float
    clock_offset_candidate_seconds: float
    clock_offset_reference: str


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

# Ground-truth samples are buffered independently for each robot source.
robot_ground_truth_buffers: Dict[
    str,
    List[RobotGroundTruthSample],
] = defaultdict(list)

# Quest windows that arrived before sufficient robot ground-truth coverage.
pending_quest_windows: Dict[
    Tuple[str, str],
    PreparedQuestWindow,
] = {}

# Recent Quest monotonic->Unix offset candidates, per Quest source.
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

quest_clock_last_candidate: Dict[
    str,
    float,
] = {}

quest_clock_last_reference: Dict[
    str,
    str,
] = {}

state_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# General helpers
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


def get_window_file_path(
    source_id: str,
    window_id: str,
) -> Path:
    safe_source_id = sanitize_filename_component(
        source_id
    )

    safe_window_id = sanitize_filename_component(
        window_id
    )

    filename = (
        f"{safe_source_id}__"
        f"{safe_window_id}.npz"
    )

    return DATA_DIR / filename


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


def normalize_horizontal_headings(
    headings: np.ndarray,
) -> np.ndarray:
    """
    Normalize X/Z heading components.

    Returns:
        (N, 3), with Y set to zero because the current trajectory/FoV model
        uses ground-plane heading.
    """
    hx = headings[:, 0]
    hz = headings[:, 2]

    norm = np.sqrt(
        hx * hx + hz * hz
    )

    if np.any(norm < 1e-8):
        raise ValueError(
            "At least one heading has near-zero "
            "horizontal magnitude."
        )

    result = np.zeros_like(
        headings,
        dtype=np.float64,
    )

    result[:, 0] = hx / norm
    result[:, 2] = hz / norm

    return result


def heading_vectors_to_unwrapped_yaw(
    headings: np.ndarray,
) -> np.ndarray:
    """
    Unity-style convention:

        +X = right
        +Y = up
        +Z = forward

    yaw = atan2(heading_x, heading_z)
    """
    normalized = normalize_horizontal_headings(
        headings
    )

    yaw = np.arctan2(
        normalized[:, 0],
        normalized[:, 2],
    )

    return np.unwrap(yaw)


def yaw_to_heading_vectors(
    yaw: np.ndarray,
) -> np.ndarray:
    headings = np.zeros(
        (len(yaw), 3),
        dtype=np.float64,
    )

    headings[:, 0] = np.sin(yaw)
    headings[:, 2] = np.cos(yaw)

    return headings


def nearest_timestamp_errors(
    query_times: np.ndarray,
    source_times: np.ndarray,
) -> np.ndarray:
    """
    For each query timestamp, compute the absolute distance to the nearest
    raw robot ground-truth timestamp.
    """
    insertion_indices = np.searchsorted(
        source_times,
        query_times,
        side="left",
    )

    errors = np.empty(
        len(query_times),
        dtype=np.float64,
    )

    for i, insertion_index in enumerate(
        insertion_indices
    ):
        candidates = []

        if insertion_index < len(source_times):
            candidates.append(
                abs(
                    source_times[insertion_index]
                    - query_times[i]
                )
            )

        if insertion_index > 0:
            candidates.append(
                abs(
                    source_times[insertion_index - 1]
                    - query_times[i]
                )
            )

        errors[i] = min(candidates)

    return errors


def interpolation_bracket_gaps(
    query_times: np.ndarray,
    source_times: np.ndarray,
) -> np.ndarray:
    """
    For each query timestamp, return the time gap between the two robot
    samples that bracket it.

    Exact timestamp matches receive a gap of 0.
    """
    insertion_indices = np.searchsorted(
        source_times,
        query_times,
        side="left",
    )

    gaps = np.zeros(
        len(query_times),
        dtype=np.float64,
    )

    for i, insertion_index in enumerate(
        insertion_indices
    ):
        if (
            insertion_index < len(source_times)
            and math.isclose(
                source_times[insertion_index],
                query_times[i],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            gaps[i] = 0.0
            continue

        if (
            insertion_index == 0
            or insertion_index >= len(source_times)
        ):
            gaps[i] = np.inf
            continue

        gaps[i] = (
            source_times[insertion_index]
            - source_times[insertion_index - 1]
        )

    return gaps


# ---------------------------------------------------------------------------
# Quest clock normalization
# ---------------------------------------------------------------------------

def looks_like_unix_timestamp(
    timestamp: float,
) -> bool:
    return (
        math.isfinite(timestamp)
        and timestamp >= UNIX_TIMESTAMP_THRESHOLD
    )


def validate_quest_window_timing(
    window: RobotMotionHistoryWindow,
    raw_timestamps: np.ndarray,
) -> None:
    if len(raw_timestamps) < 2:
        raise ValueError(
            "Quest motion window must contain at "
            "least two samples."
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
            "Quest motion-window timestamps must be "
            "strictly increasing."
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
            "Quest time_from_window_start contains "
            "non-finite values."
        )

    if np.any(
        np.diff(relative_times) <= 0.0
    ):
        raise ValueError(
            "Quest time_from_window_start values must "
            "be strictly increasing."
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

    if (
        abs(
            raw_span
            - relative_span
        )
        > tolerance
    ):
        raise ValueError(
            "Quest raw timestamp span does not agree "
            "with time_from_window_start. "
            f"raw_span={raw_span:.3f}s, "
            f"relative_span={relative_span:.3f}s, "
            f"tolerance={tolerance:.3f}s."
        )

    if (
        expected_duration > 0.0
        and abs(
            expected_duration
            - relative_span
        )
        > tolerance
    ):
        raise ValueError(
            "Quest window_duration_seconds does not "
            "agree with time_from_window_start. "
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

    quest_clock_last_candidate.pop(
        source_id,
        None,
    )

    quest_clock_last_reference.pop(
        source_id,
        None,
    )


def current_quest_offset_estimate(
    source_id: str,
) -> Optional[float]:
    candidates = quest_clock_offset_candidates.get(
        source_id
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


def rebase_pending_windows_for_quest_source(
    source_id: str,
) -> None:
    """
    If the learned clock offset changes slightly as more Quest windows arrive,
    update pending windows from the same Quest source to the newest estimate.
    """
    estimate = quest_clock_offset_estimates.get(
        source_id
    )

    if estimate is None:
        return

    for prepared in (
        pending_quest_windows.values()
    ):
        if (
            prepared.window.source_id
            != source_id
        ):
            continue

        if (
            prepared.timestamp_mode
            != "rebased_monotonic"
        ):
            continue

        prepared.clock_offset_seconds = (
            estimate
        )

        prepared.aligned_timestamps = (
            prepared.original_timestamps
            + estimate
        )


def prepare_quest_window(
    window: RobotMotionHistoryWindow,
    server_receive_unix_seconds: float,
) -> PreparedQuestWindow:
    """
    Convert a Quest window to the Unix clock domain used by robot ground truth.

    If the Quest sample timestamps already look like Unix seconds, they are
    used directly.

    Otherwise, a persistent offset is learned:

        unix_time ~= unity_monotonic_time + offset

    Preferred offset reference:
        window.sent_at_unix_seconds

    Fallback:
        FastAPI server receive time

    The server retains a rolling history of offset candidates and uses a low
    percentile to reduce positive send/network-latency bias.
    """
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

    # Already in Unix time: no rebasing necessary.
    if looks_like_unix_timestamp(
        float(
            np.median(
                raw_timestamps
            )
        )
    ):
        quest_clock_last_raw_timestamp[
            window.source_id
        ] = float(
            raw_timestamps[-1]
        )

        return PreparedQuestWindow(
            window=window,
            original_timestamps=(
                raw_timestamps.copy()
            ),
            aligned_timestamps=(
                raw_timestamps.copy()
            ),
            timestamp_mode="unix",
            clock_offset_seconds=0.0,
            clock_offset_candidate_seconds=0.0,
            clock_offset_reference=(
                "quest_sample_timestamp"
            ),
        )

    # Unity/Quest elapsed clock may reset when the application restarts.
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
        print(
            "Quest monotonic clock appears to have "
            f"reset for source '{window.source_id}'. "
            "Resetting Quest->Unix offset estimator."
        )

        reset_quest_clock_estimator(
            window.source_id
        )

    # Prefer the Unix timestamp generated by the Quest sender itself if it
    # is valid. That avoids network receive latency.
    if looks_like_unix_timestamp(
        float(
            window.sent_at_unix_seconds
        )
    ):
        unix_reference = float(
            window.sent_at_unix_seconds
        )

        reference_name = (
            "window.sent_at_unix_seconds"
        )
    else:
        unix_reference = float(
            server_receive_unix_seconds
        )

        reference_name = (
            "server_receive_unix_seconds"
        )

    # The final sample is normally immediately before the window POST.
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
        print(
            "Large Quest clock-offset change detected "
            f"for source '{window.source_id}': "
            f"old={previous_estimate:.6f}s, "
            f"candidate={candidate:.6f}s. "
            "Resetting offset estimator."
        )

        reset_quest_clock_estimator(
            window.source_id
        )

    quest_clock_offset_candidates[
        window.source_id
    ].append(
        candidate
    )

    estimate = current_quest_offset_estimate(
        window.source_id
    )

    if estimate is None:
        raise ValueError(
            "Could not estimate Quest->Unix clock offset."
        )

    quest_clock_last_raw_timestamp[
        window.source_id
    ] = float(
        raw_timestamps[-1]
    )

    quest_clock_last_candidate[
        window.source_id
    ] = candidate

    quest_clock_last_reference[
        window.source_id
    ] = reference_name

    # Keep pending windows from this same Unity clock on the newest
    # persistent estimate.
    rebase_pending_windows_for_quest_source(
        window.source_id
    )

    aligned_timestamps = (
        raw_timestamps
        + estimate
    )

    return PreparedQuestWindow(
        window=window,
        original_timestamps=(
            raw_timestamps.copy()
        ),
        aligned_timestamps=(
            aligned_timestamps
        ),
        timestamp_mode=(
            "rebased_monotonic"
        ),
        clock_offset_seconds=(
            estimate
        ),
        clock_offset_candidate_seconds=(
            candidate
        ),
        clock_offset_reference=(
            reference_name
        ),
    )


# ---------------------------------------------------------------------------
# Ground-truth buffer helpers
# ---------------------------------------------------------------------------

def sort_and_deduplicate_ground_truth(
    source_id: str,
) -> None:
    """
    Sort by timestamp and keep the most recently received sample if duplicate
    timestamps occur.
    """
    samples = robot_ground_truth_buffers[
        source_id
    ]

    if not samples:
        return

    by_timestamp = {}

    for sample in samples:
        by_timestamp[
            float(sample.timestamp)
        ] = sample

    sorted_times = sorted(
        by_timestamp.keys()
    )

    robot_ground_truth_buffers[
        source_id
    ] = [
        by_timestamp[timestamp]
        for timestamp in sorted_times
    ]


def prune_ground_truth_buffer(
    source_id: str,
) -> None:
    samples = robot_ground_truth_buffers[
        source_id
    ]

    if not samples:
        return

    newest_timestamp = samples[-1].timestamp

    cutoff = (
        newest_timestamp
        - ROBOT_BUFFER_SECONDS
    )

    # Do not prune robot data that a VALID rebased pending Quest window may
    # still need.
    pending_start_times = []

    for prepared in (
        pending_quest_windows.values()
    ):
        if (
            prepared.window.robot_source_id
            == source_id
            and len(
                prepared.aligned_timestamps
            ) > 0
        ):
            pending_start_times.append(
                float(
                    prepared.aligned_timestamps[0]
                )
            )

    if pending_start_times:
        earliest_needed = (
            min(pending_start_times)
            - 1.0
        )

        cutoff = min(
            cutoff,
            earliest_needed,
        )

    first_index_to_keep = 0

    while (
        first_index_to_keep < len(samples)
        and samples[
            first_index_to_keep
        ].timestamp < cutoff
    ):
        first_index_to_keep += 1

    if first_index_to_keep > 0:
        robot_ground_truth_buffers[
            source_id
        ] = samples[
            first_index_to_keep:
        ]


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def has_ground_truth_coverage(
    prepared: PreparedQuestWindow,
) -> bool:
    window = prepared.window

    samples = robot_ground_truth_buffers.get(
        window.robot_source_id,
        [],
    )

    if len(samples) < 2:
        return False

    quest_start = float(
        prepared.aligned_timestamps[0]
    )

    quest_end = float(
        prepared.aligned_timestamps[-1]
    )

    robot_start = float(
        samples[0].timestamp
    )

    robot_end = float(
        samples[-1].timestamp
    )

    return (
        robot_start <= quest_start
        and robot_end >= quest_end
    )


def align_robot_ground_truth_to_quest(
    prepared: PreparedQuestWindow,
) -> dict:
    """
    Interpolate robot ground-truth pose to every REBASED Quest timestamp.

    Position is linearly interpolated.

    Heading is converted to yaw, unwrapped, linearly interpolated in yaw,
    then converted back to a normalized X/Z heading vector.
    """
    window = prepared.window

    robot_samples = (
        robot_ground_truth_buffers.get(
            window.robot_source_id,
            [],
        )
    )

    if len(robot_samples) < 2:
        raise ValueError(
            "Fewer than two robot ground-truth samples "
            "are available."
        )

    quest_times = np.asarray(
        prepared.aligned_timestamps,
        dtype=np.float64,
    )

    robot_times = np.asarray(
        [
            sample.timestamp
            for sample in robot_samples
        ],
        dtype=np.float64,
    )

    if (
        quest_times.min() < robot_times[0]
        or quest_times.max() > robot_times[-1]
    ):
        raise ValueError(
            "Robot ground-truth buffer does not cover "
            "the full rebased Quest window."
        )

    robot_positions = vector3_list_to_array(
        [
            sample.position
            for sample in robot_samples
        ]
    )

    robot_headings = vector3_list_to_array(
        [
            sample.heading
            for sample in robot_samples
        ]
    )

    robot_yaw = (
        heading_vectors_to_unwrapped_yaw(
            robot_headings
        )
    )

    aligned_positions = np.column_stack(
        [
            np.interp(
                quest_times,
                robot_times,
                robot_positions[:, axis],
            )
            for axis in range(3)
        ]
    )

    aligned_yaw = np.interp(
        quest_times,
        robot_times,
        robot_yaw,
    )

    aligned_headings = (
        yaw_to_heading_vectors(
            aligned_yaw
        )
    )

    nearest_errors = (
        nearest_timestamp_errors(
            quest_times,
            robot_times,
        )
    )

    bracket_gaps = (
        interpolation_bracket_gaps(
            quest_times,
            robot_times,
        )
    )

    max_bracket_gap = float(
        np.max(
            bracket_gaps
        )
    )

    # Save the subset of raw robot samples surrounding this Quest window.
    left_index = max(
        0,
        int(
            np.searchsorted(
                robot_times,
                quest_times.min(),
                side="left",
            )
        ) - 1,
    )

    right_index = min(
        len(robot_times),
        int(
            np.searchsorted(
                robot_times,
                quest_times.max(),
                side="right",
            )
        ) + 1,
    )

    source_times = robot_times[
        left_index:right_index
    ]

    source_positions = robot_positions[
        left_index:right_index
    ]

    source_headings = (
        normalize_horizontal_headings(
            robot_headings[
                left_index:right_index
            ]
        )
    )

    return {
        "aligned_positions": (
            aligned_positions
        ),
        "aligned_headings": (
            aligned_headings
        ),
        "aligned_yaw": aligned_yaw,
        "nearest_timestamp_errors": (
            nearest_errors
        ),
        "bracket_gaps": bracket_gaps,
        "max_bracket_gap": (
            max_bracket_gap
        ),
        "source_times": source_times,
        "source_positions": (
            source_positions
        ),
        "source_headings": (
            source_headings
        ),
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_aligned_motion_window(
    prepared: PreparedQuestWindow,
    alignment: dict,
) -> Path:
    window = prepared.window

    output_path = get_window_file_path(
        source_id=window.source_id,
        window_id=window.window_id,
    )

    quest_original_timestamps = np.asarray(
        prepared.original_timestamps,
        dtype=np.float64,
    )

    quest_timestamps = np.asarray(
        prepared.aligned_timestamps,
        dtype=np.float64,
    )

    quest_time_from_start = np.asarray(
        [
            sample.time_from_window_start
            for sample in window.samples
        ],
        dtype=np.float32,
    )

    quest_positions = vector3_list_to_array(
        [
            sample.position
            for sample in window.samples
        ]
    ).astype(np.float32)

    quest_headings = vector3_list_to_array(
        [
            sample.heading
            for sample in window.samples
        ]
    ).astype(np.float32)

    robot_positions = alignment[
        "aligned_positions"
    ].astype(np.float32)

    robot_headings = alignment[
        "aligned_headings"
    ].astype(np.float32)

    robot_yaw = alignment[
        "aligned_yaw"
    ].astype(np.float32)

    nearest_errors = alignment[
        "nearest_timestamp_errors"
    ].astype(np.float32)

    bracket_gaps = alignment[
        "bracket_gaps"
    ].astype(np.float32)

    max_bracket_gap = float(
        alignment[
            "max_bracket_gap"
        ]
    )

    alignment_warning = (
        max_bracket_gap
        > ALIGNMENT_GAP_WARNING_SECONDS
    )

    with tempfile.NamedTemporaryFile(
        dir=DATA_DIR,
        suffix=".npz",
        delete=False,
    ) as temp_file:
        temp_path = Path(
            temp_file.name
        )

    try:
        np.savez_compressed(
            temp_path,

            # -----------------------------------------------------------
            # Window metadata
            # -----------------------------------------------------------
            schema_version=np.asarray(
                window.schema_version
            ),

            window_id=np.asarray(
                window.window_id
            ),

            source_id=np.asarray(
                window.source_id
            ),

            robot_source_id=np.asarray(
                window.robot_source_id
            ),

            sent_at_unix_seconds=np.asarray(
                window.sent_at_unix_seconds,
                dtype=np.float64,
            ),

            window_duration_seconds=np.asarray(
                window.window_duration_seconds,
                dtype=np.float32,
            ),

            sample_count=np.asarray(
                window.sample_count,
                dtype=np.int32,
            ),

            # -----------------------------------------------------------
            # Quest clock diagnostics
            # -----------------------------------------------------------
            quest_original_timestamps=(
                quest_original_timestamps
            ),

            quest_timestamp_mode=np.asarray(
                prepared.timestamp_mode
            ),

            quest_timestamp_was_rebased=np.asarray(
                prepared.timestamp_mode
                == "rebased_monotonic",
                dtype=np.bool_,
            ),

            quest_clock_offset_seconds=np.asarray(
                prepared.clock_offset_seconds,
                dtype=np.float64,
            ),

            quest_clock_offset_candidate_seconds=(
                np.asarray(
                    prepared.clock_offset_candidate_seconds,
                    dtype=np.float64,
                )
            ),

            quest_clock_offset_reference=np.asarray(
                prepared.clock_offset_reference
            ),

            # -----------------------------------------------------------
            # Backward-compatible Quest keys.
            #
            # `timestamps` now means the normalized/rebased Unix timeline,
            # while positions/headings remain the Quest-estimated values.
            # -----------------------------------------------------------
            timestamps=quest_timestamps,
            time_from_window_start=(
                quest_time_from_start
            ),
            positions=quest_positions,
            headings=quest_headings,

            # -----------------------------------------------------------
            # Explicit Quest-estimate keys
            # -----------------------------------------------------------
            quest_timestamps=quest_timestamps,
            quest_time_from_window_start=(
                quest_time_from_start
            ),
            quest_positions=quest_positions,
            quest_headings=quest_headings,

            # -----------------------------------------------------------
            # Robot ground truth aligned exactly to Quest timestamps
            # -----------------------------------------------------------
            robot_ground_truth_timestamps=(
                quest_timestamps
            ),
            robot_ground_truth_positions=(
                robot_positions
            ),
            robot_ground_truth_headings=(
                robot_headings
            ),
            robot_ground_truth_yaw=(
                robot_yaw
            ),

            # -----------------------------------------------------------
            # Alignment diagnostics
            # -----------------------------------------------------------
            alignment_nearest_timestamp_error_seconds=(
                nearest_errors
            ),

            alignment_interpolation_gap_seconds=(
                bracket_gaps
            ),

            alignment_mean_nearest_timestamp_error_seconds=(
                np.asarray(
                    float(
                        np.mean(
                            nearest_errors
                        )
                    ),
                    dtype=np.float64,
                )
            ),

            alignment_max_nearest_timestamp_error_seconds=(
                np.asarray(
                    float(
                        np.max(
                            nearest_errors
                        )
                    ),
                    dtype=np.float64,
                )
            ),

            alignment_max_interpolation_gap_seconds=(
                np.asarray(
                    max_bracket_gap,
                    dtype=np.float64,
                )
            ),

            alignment_warning=np.asarray(
                alignment_warning,
                dtype=np.bool_,
            ),

            # -----------------------------------------------------------
            # Raw robot ground-truth samples surrounding this window.
            # These are retained so alignment can be audited/recomputed.
            # -----------------------------------------------------------
            robot_source_timestamps=(
                alignment[
                    "source_times"
                ].astype(np.float64)
            ),

            robot_source_positions=(
                alignment[
                    "source_positions"
                ].astype(np.float32)
            ),

            robot_source_headings=(
                alignment[
                    "source_headings"
                ].astype(np.float32)
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

    return output_path


def print_alignment_summary(
    prepared: PreparedQuestWindow,
    alignment: dict,
    saved_path: Path,
) -> None:
    window = prepared.window

    nearest_errors = alignment[
        "nearest_timestamp_errors"
    ]

    max_bracket_gap = alignment[
        "max_bracket_gap"
    ]

    print()
    print(
        "Saved aligned robot-motion window"
    )
    print(
        "---------------------------------"
    )
    print(
        f"Window ID: {window.window_id}"
    )
    print(
        f"Quest source: {window.source_id}"
    )
    print(
        "Robot GT source: "
        f"{window.robot_source_id}"
    )
    print(
        f"Samples: {window.sample_count}"
    )
    print(
        "Quest timestamp mode: "
        f"{prepared.timestamp_mode}"
    )

    if (
        prepared.timestamp_mode
        == "rebased_monotonic"
    ):
        print(
            "Quest->Unix offset: "
            f"{prepared.clock_offset_seconds:.6f} s"
        )
        print(
            "Offset reference: "
            f"{prepared.clock_offset_reference}"
        )

    print(
        "Aligned Quest range: "
        f"{prepared.aligned_timestamps[0]:.6f} "
        "-> "
        f"{prepared.aligned_timestamps[-1]:.6f}"
    )
    print(
        "Mean nearest timestamp error: "
        f"{np.mean(nearest_errors) * 1000.0:.2f} ms"
    )
    print(
        "Max nearest timestamp error: "
        f"{np.max(nearest_errors) * 1000.0:.2f} ms"
    )
    print(
        "Max robot interpolation gap: "
        f"{max_bracket_gap * 1000.0:.2f} ms"
    )

    if (
        max_bracket_gap
        > ALIGNMENT_GAP_WARNING_SECONDS
    ):
        print(
            "WARNING: robot ground-truth stream "
            "contains a relatively large sampling gap."
        )

    print(
        f"Saved: {saved_path}"
    )
    print(
        "---------------------------------"
    )
    print()


def try_align_and_save_window(
    prepared: PreparedQuestWindow,
) -> Optional[Path]:
    if not has_ground_truth_coverage(
        prepared
    ):
        return None

    alignment = (
        align_robot_ground_truth_to_quest(
            prepared
        )
    )

    saved_path = (
        save_aligned_motion_window(
            prepared,
            alignment,
        )
    )

    print_alignment_summary(
        prepared,
        alignment,
        saved_path,
    )

    return saved_path


def try_save_pending_windows_for_robot(
    robot_source_id: str,
) -> List[Path]:
    saved_paths = []

    pending_keys = [
        key
        for key, prepared
        in pending_quest_windows.items()
        if (
            prepared.window.robot_source_id
            == robot_source_id
        )
    ]

    for key in pending_keys:
        prepared = pending_quest_windows[
            key
        ]

        window = prepared.window

        output_path = get_window_file_path(
            source_id=window.source_id,
            window_id=window.window_id,
        )

        if output_path.exists():
            del pending_quest_windows[
                key
            ]
            continue

        saved_path = (
            try_align_and_save_window(
                prepared
            )
        )

        if saved_path is not None:
            saved_paths.append(
                saved_path
            )

            del pending_quest_windows[
                key
            ]

    return saved_paths


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    async with state_lock:
        robot_buffer_sizes = {
            source_id: len(samples)
            for source_id, samples
            in robot_ground_truth_buffers.items()
        }

        return {
            "status": "ok",
            "data_directory": str(
                DATA_DIR
            ),
            "pending_quest_windows": len(
                pending_quest_windows
            ),
            "robot_ground_truth_buffer_sizes": (
                robot_buffer_sizes
            ),
            "quest_clock_offset_estimates": {
                source_id: estimate
                for source_id, estimate
                in quest_clock_offset_estimates.items()
            },
        }


@app.get("/alignment_status")
async def alignment_status():
    async with state_lock:
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
                    "start_timestamp": (
                        samples[0].timestamp
                    ),
                    "end_timestamp": (
                        samples[-1].timestamp
                    ),
                    "duration_seconds": (
                        samples[-1].timestamp
                        - samples[0].timestamp
                    ),
                }
            else:
                robot_sources[
                    source_id
                ] = {
                    "sample_count": 0,
                }

        quest_clock_sources = {}

        known_quest_sources = set(
            quest_clock_offset_candidates.keys()
        ) | set(
            quest_clock_offset_estimates.keys()
        )

        for source_id in sorted(
            known_quest_sources
        ):
            candidates = list(
                quest_clock_offset_candidates.get(
                    source_id,
                    [],
                )
            )

            quest_clock_sources[
                source_id
            ] = {
                "offset_estimate_seconds": (
                    quest_clock_offset_estimates.get(
                        source_id
                    )
                ),
                "candidate_count": len(
                    candidates
                ),
                "last_candidate_seconds": (
                    quest_clock_last_candidate.get(
                        source_id
                    )
                ),
                "last_reference": (
                    quest_clock_last_reference.get(
                        source_id
                    )
                ),
                "last_raw_timestamp": (
                    quest_clock_last_raw_timestamp.get(
                        source_id
                    )
                ),
            }

        pending = []

        for prepared in (
            pending_quest_windows.values()
        ):
            window = prepared.window

            pending.append(
                {
                    "window_id": (
                        window.window_id
                    ),
                    "quest_source_id": (
                        window.source_id
                    ),
                    "robot_source_id": (
                        window.robot_source_id
                    ),
                    "timestamp_mode": (
                        prepared.timestamp_mode
                    ),
                    "clock_offset_seconds": (
                        prepared.clock_offset_seconds
                    ),
                    "raw_start_timestamp": (
                        float(
                            prepared.original_timestamps[0]
                        )
                    ),
                    "raw_end_timestamp": (
                        float(
                            prepared.original_timestamps[-1]
                        )
                    ),
                    "aligned_start_timestamp": (
                        float(
                            prepared.aligned_timestamps[0]
                        )
                    ),
                    "aligned_end_timestamp": (
                        float(
                            prepared.aligned_timestamps[-1]
                        )
                    ),
                }
            )

        return {
            "robot_sources": (
                robot_sources
            ),
            "quest_clock_sources": (
                quest_clock_sources
            ),
            "pending_windows": pending,
        }


@app.post("/robot_ground_truth")
async def receive_robot_ground_truth(
    batch: RobotGroundTruthBatch,
):
    if (
        batch.sample_count
        != len(batch.samples)
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "sample_count does not match the "
                "number of robot ground-truth "
                "samples in the payload."
            ),
        )

    if not batch.samples:
        raise HTTPException(
            status_code=400,
            detail=(
                "Robot ground-truth batch has "
                "no samples."
            ),
        )

    robot_timestamps = np.asarray(
        [
            sample.timestamp
            for sample in batch.samples
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            robot_timestamps
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Robot ground-truth timestamps "
                "contain non-finite values."
            ),
        )

    async with state_lock:
        buffer = (
            robot_ground_truth_buffers[
                batch.source_id
            ]
        )

        buffer.extend(
            batch.samples
        )

        sort_and_deduplicate_ground_truth(
            batch.source_id
        )

        saved_paths = (
            try_save_pending_windows_for_robot(
                batch.source_id
            )
        )

        prune_ground_truth_buffer(
            batch.source_id
        )

        current_buffer = (
            robot_ground_truth_buffers[
                batch.source_id
            ]
        )

        if current_buffer:
            buffer_start = float(
                current_buffer[
                    0
                ].timestamp
            )

            buffer_end = float(
                current_buffer[
                    -1
                ].timestamp
            )
        else:
            buffer_start = None
            buffer_end = None

    return {
        "accepted": True,
        "source_id": batch.source_id,
        "samples_received": len(
            batch.samples
        ),
        "buffer_start_timestamp": (
            buffer_start
        ),
        "buffer_end_timestamp": (
            buffer_end
        ),
        "pending_windows_saved": len(
            saved_paths
        ),
        "saved_paths": [
            str(path)
            for path in saved_paths
        ],
    }


@app.post("/robot_motion_history")
async def receive_robot_motion_history(
    window: RobotMotionHistoryWindow,
):
    server_receive_unix_seconds = (
        time.time()
    )

    if (
        window.sample_count
        != len(window.samples)
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "sample_count does not match "
                "the number of samples in the "
                "payload."
            ),
        )

    if not window.samples:
        raise HTTPException(
            status_code=400,
            detail=(
                "Motion window has no samples."
            ),
        )

    output_path = get_window_file_path(
        source_id=window.source_id,
        window_id=window.window_id,
    )

    pending_key = (
        window.source_id,
        window.window_id,
    )

    async with state_lock:
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
                "samples_received": len(
                    window.samples
                ),
                "timestamp_mode": (
                    existing.timestamp_mode
                ),
                "aligned_start_timestamp": (
                    float(
                        existing.aligned_timestamps[0]
                    )
                ),
                "aligned_end_timestamp": (
                    float(
                        existing.aligned_timestamps[-1]
                    )
                ),
            }

        try:
            prepared = prepare_quest_window(
                window,
                server_receive_unix_seconds,
            )

            # A new offset estimate can slightly improve older pending windows
            # from this Quest source. Try them again immediately against the
            # currently buffered robot ground truth.
            newly_saved_paths = (
                try_save_pending_windows_for_robot(
                    window.robot_source_id
                )
            )

            saved_path = (
                try_align_and_save_window(
                    prepared
                )
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
                    "Failed to normalize/align/save "
                    "robot motion-history window."
                ),
            ) from exc

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
                "timestamp_mode": (
                    prepared.timestamp_mode
                ),
                "clock_offset_seconds": (
                    prepared.clock_offset_seconds
                ),
                "clock_offset_reference": (
                    prepared.clock_offset_reference
                ),
                "aligned_start_timestamp": (
                    float(
                        prepared.aligned_timestamps[0]
                    )
                ),
                "aligned_end_timestamp": (
                    float(
                        prepared.aligned_timestamps[-1]
                    )
                ),
                "saved_path": str(
                    saved_path
                ),
                "other_pending_windows_saved": (
                    len(
                        newly_saved_paths
                    )
                ),
            }

        # Ground truth has not yet covered the entire normalized Quest window.
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
            "timestamp_mode": (
                prepared.timestamp_mode
            ),
            "clock_offset_seconds": (
                prepared.clock_offset_seconds
            ),
            "clock_offset_reference": (
                prepared.clock_offset_reference
            ),
            "raw_start_timestamp": (
                float(
                    prepared.original_timestamps[0]
                )
            ),
            "raw_end_timestamp": (
                float(
                    prepared.original_timestamps[-1]
                )
            ),
            "aligned_start_timestamp": (
                float(
                    prepared.aligned_timestamps[0]
                )
            ),
            "aligned_end_timestamp": (
                float(
                    prepared.aligned_timestamps[-1]
                )
            ),
            "other_pending_windows_saved": (
                len(
                    newly_saved_paths
                )
            ),
            "detail": (
                "Quest window timestamps were normalized. "
                "The window is buffered until robot ground "
                "truth covers its full aligned timestamp range."
            ),
        }