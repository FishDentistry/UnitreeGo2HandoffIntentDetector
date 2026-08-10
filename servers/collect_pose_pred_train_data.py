import asyncio
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Robot Motion Alignment Server",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Keep enough robot ground-truth history to cover delayed Quest windows.
ROBOT_BUFFER_SECONDS = 300.0

# This does not prevent saving. It is recorded as a warning/diagnostic if
# interpolation has to bridge a larger-than-expected robot sampling gap.
ALIGNMENT_GAP_WARNING_SECONDS = 0.25


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

    timestamp must use the same absolute time base as the robot
    ground-truth timestamps, preferably Unix seconds.
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

    sent_at_unix_seconds: float

    window_duration_seconds: float
    sample_count: int = Field(ge=1)

    samples: List[RobotMotionSample]


class RobotGroundTruthSample(BaseModel):
    """
    Ground-truth robot pose sample.

    timestamp must use the same absolute time base as the Quest timestamps,
    preferably Unix seconds.

    heading should be the robot/camera forward vector in the robot
    ground-truth world frame.
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
# In-memory alignment state
# ---------------------------------------------------------------------------

# Ground-truth samples are buffered independently for each robot source.
robot_ground_truth_buffers: Dict[
    str,
    List[RobotGroundTruthSample],
] = defaultdict(list)

# Quest windows that arrived before sufficient robot ground-truth coverage.
pending_quest_windows: Dict[
    Tuple[str, str],
    RobotMotionHistoryWindow,
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
        (N, 3), with Y preserved as zero because the current trajectory/FoV
        model uses ground-plane heading.
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

    # Do not prune data that may still be required by an existing pending
    # Quest window for this robot source.
    pending_start_times = []

    for window in pending_quest_windows.values():
        if (
            window.robot_source_id == source_id
            and window.samples
        ):
            pending_start_times.append(
                window.samples[0].timestamp
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
    window: RobotMotionHistoryWindow,
) -> bool:
    samples = robot_ground_truth_buffers.get(
        window.robot_source_id,
        [],
    )

    if len(samples) < 2:
        return False

    quest_start = min(
        sample.timestamp
        for sample in window.samples
    )

    quest_end = max(
        sample.timestamp
        for sample in window.samples
    )

    robot_start = samples[0].timestamp
    robot_end = samples[-1].timestamp

    return (
        robot_start <= quest_start
        and robot_end >= quest_end
    )


def align_robot_ground_truth_to_quest(
    window: RobotMotionHistoryWindow,
) -> dict:
    """
    Interpolate robot ground-truth pose to every Quest sample timestamp.

    Position is linearly interpolated.

    Heading is converted to yaw, unwrapped, linearly interpolated in yaw,
    then converted back to a normalized X/Z heading vector.
    """
    robot_samples = robot_ground_truth_buffers.get(
        window.robot_source_id,
        [],
    )

    if len(robot_samples) < 2:
        raise ValueError(
            "Fewer than two robot ground-truth samples "
            "are available."
        )

    quest_times = np.asarray(
        [
            sample.timestamp
            for sample in window.samples
        ],
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
            "the full Quest window."
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

    nearest_errors = nearest_timestamp_errors(
        quest_times,
        robot_times,
    )

    bracket_gaps = interpolation_bracket_gaps(
        quest_times,
        robot_times,
    )

    max_bracket_gap = float(
        np.max(bracket_gaps)
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
        "aligned_positions": aligned_positions,
        "aligned_headings": aligned_headings,
        "aligned_yaw": aligned_yaw,
        "nearest_timestamp_errors": nearest_errors,
        "bracket_gaps": bracket_gaps,
        "max_bracket_gap": max_bracket_gap,
        "source_times": source_times,
        "source_positions": source_positions,
        "source_headings": source_headings,
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_aligned_motion_window(
    window: RobotMotionHistoryWindow,
    alignment: dict,
) -> Path:
    output_path = get_window_file_path(
        source_id=window.source_id,
        window_id=window.window_id,
    )

    quest_timestamps = np.asarray(
        [
            sample.timestamp
            for sample in window.samples
        ],
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
        alignment["max_bracket_gap"]
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
        temp_path = Path(temp_file.name)

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
            # Backward-compatible Quest keys.
            #
            # Existing scripts that read:
            #   timestamps
            #   positions
            #   headings
            #
            # will continue to receive the Quest-estimated values.
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
    window: RobotMotionHistoryWindow,
    alignment: dict,
    saved_path: Path,
) -> None:
    nearest_errors = alignment[
        "nearest_timestamp_errors"
    ]

    max_bracket_gap = alignment[
        "max_bracket_gap"
    ]

    print()
    print("Saved aligned robot-motion window")
    print("---------------------------------")
    print(f"Window ID: {window.window_id}")
    print(f"Quest source: {window.source_id}")
    print(
        "Robot GT source: "
        f"{window.robot_source_id}"
    )
    print(
        f"Samples: {window.sample_count}"
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

    print(f"Saved: {saved_path}")
    print("---------------------------------")
    print()


def try_align_and_save_window(
    window: RobotMotionHistoryWindow,
) -> Optional[Path]:
    if not has_ground_truth_coverage(
        window
    ):
        return None

    alignment = (
        align_robot_ground_truth_to_quest(
            window
        )
    )

    saved_path = save_aligned_motion_window(
        window,
        alignment,
    )

    print_alignment_summary(
        window,
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
        for key, window
        in pending_quest_windows.items()
        if (
            window.robot_source_id
            == robot_source_id
        )
    ]

    for key in pending_keys:
        window = pending_quest_windows[
            key
        ]

        output_path = get_window_file_path(
            source_id=window.source_id,
            window_id=window.window_id,
        )

        if output_path.exists():
            del pending_quest_windows[key]
            continue

        saved_path = try_align_and_save_window(
            window
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
            "data_directory": str(DATA_DIR),
            "pending_quest_windows": len(
                pending_quest_windows
            ),
            "robot_ground_truth_buffer_sizes": (
                robot_buffer_sizes
            ),
        }


@app.get("/alignment_status")
async def alignment_status():
    async with state_lock:
        robot_sources = {}

        for source_id, samples in (
            robot_ground_truth_buffers.items()
        ):
            if samples:
                robot_sources[source_id] = {
                    "sample_count": len(samples),
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
                robot_sources[source_id] = {
                    "sample_count": 0,
                }

        pending = [
            {
                "window_id": window.window_id,
                "quest_source_id": (
                    window.source_id
                ),
                "robot_source_id": (
                    window.robot_source_id
                ),
                "start_timestamp": (
                    window.samples[0].timestamp
                    if window.samples
                    else None
                ),
                "end_timestamp": (
                    window.samples[-1].timestamp
                    if window.samples
                    else None
                ),
            }
            for window in (
                pending_quest_windows.values()
            )
        ]

        return {
            "robot_sources": robot_sources,
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

    async with state_lock:
        buffer = robot_ground_truth_buffers[
            batch.source_id
        ]

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
            buffer_start = (
                current_buffer[0].timestamp
            )

            buffer_end = (
                current_buffer[-1].timestamp
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
            detail="Motion window has no samples.",
        )

    # Ensure the Quest samples themselves are ordered.
    quest_timestamps = [
        sample.timestamp
        for sample in window.samples
    ]

    if any(
        later <= earlier
        for earlier, later in zip(
            quest_timestamps,
            quest_timestamps[1:],
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Quest motion-window timestamps "
                "must be strictly increasing."
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

        if pending_key in pending_quest_windows:
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
            }

        try:
            saved_path = (
                try_align_and_save_window(
                    window
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
                    "Failed to align/save robot "
                    "motion-history window."
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
                "saved_path": str(
                    saved_path
                ),
            }

        # Ground truth has not yet covered the entire Quest window.
        pending_quest_windows[
            pending_key
        ] = window

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
            "detail": (
                "Quest window is buffered until "
                "robot ground truth covers its "
                "full timestamp range."
            ),
        }