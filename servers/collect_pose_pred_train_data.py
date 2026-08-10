import os
import re
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Robot Motion History Server",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """
    Find the UnitreeGo2HandoffIntentDetector project directory.

    This allows the server script to be run from different working
    directories without relying on the current working directory.
    """
    script_dir = Path(__file__).resolve().parent

    for parent in [script_dir, *script_dir.parents]:
        if parent.name == "UnitreeGo2HandoffIntentDetector":
            return parent

        candidate = parent / "UnitreeGo2HandoffIntentDetector"
        if candidate.is_dir():
            return candidate

    # Fallback if the project directory cannot be found above the script.
    return Path.cwd() / "UnitreeGo2HandoffIntentDetector"


PROJECT_ROOT = find_project_root()

DATA_DIR = (
    PROJECT_ROOT
    / "robot_pose_prediction"
    / "data"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class Vector3(BaseModel):
    x: float
    y: float
    z: float


class RobotMotionSample(BaseModel):
    timestamp: float
    time_from_window_start: float

    position: Vector3
    heading: Vector3


class RobotMotionHistoryWindow(BaseModel):
    schema_version: str

    window_id: str
    source_id: str

    sent_at_unix_seconds: float

    window_duration_seconds: float
    sample_count: int = Field(ge=1)

    samples: List[RobotMotionSample]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_filename_component(value: str) -> str:
    """
    Convert a string into something safe to use inside a filename.
    """
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "unknown"


def get_window_file_path(
    source_id: str,
    window_id: str,
) -> Path:
    safe_source_id = sanitize_filename_component(source_id)
    safe_window_id = sanitize_filename_component(window_id)

    filename = f"{safe_source_id}__{safe_window_id}.npz"

    return DATA_DIR / filename


def save_motion_window(
    window: RobotMotionHistoryWindow,
) -> Path:
    """
    Save one robot motion-history window as a compressed NumPy file.

    Stored arrays:

        timestamps:
            shape (N,)

        time_from_window_start:
            shape (N,)

        positions:
            shape (N, 3)
            columns = x, y, z

        headings:
            shape (N, 3)
            columns = x, y, z

    Window-level metadata is stored in the same .npz file.
    """

    output_path = get_window_file_path(
        source_id=window.source_id,
        window_id=window.window_id,
    )

    timestamps = np.asarray(
        [sample.timestamp for sample in window.samples],
        dtype=np.float64,
    )

    time_from_window_start = np.asarray(
        [
            sample.time_from_window_start
            for sample in window.samples
        ],
        dtype=np.float32,
    )

    positions = np.asarray(
        [
            [
                sample.position.x,
                sample.position.y,
                sample.position.z,
            ]
            for sample in window.samples
        ],
        dtype=np.float32,
    )

    headings = np.asarray(
        [
            [
                sample.heading.x,
                sample.heading.y,
                sample.heading.z,
            ]
            for sample in window.samples
        ],
        dtype=np.float32,
    )

    # Write to a temporary file first, then atomically move it into place.
    # This prevents partially-written training files if the process exits
    # during a save.
    with tempfile.NamedTemporaryFile(
        dir=DATA_DIR,
        suffix=".npz",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        np.savez_compressed(
            temp_path,

            # Window metadata
            schema_version=np.asarray(window.schema_version),
            window_id=np.asarray(window.window_id),
            source_id=np.asarray(window.source_id),

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

            # Sequence data
            timestamps=timestamps,
            time_from_window_start=time_from_window_start,
            positions=positions,
            headings=headings,
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "data_directory": str(DATA_DIR),
    }


@app.post("/robot_motion_history")
async def receive_robot_motion_history(
    window: RobotMotionHistoryWindow,
):
    if window.sample_count != len(window.samples):
        raise HTTPException(
            status_code=400,
            detail=(
                "sample_count does not match the number "
                "of samples in the payload."
            ),
        )

    print()
    print("Received robot motion-history window")
    print("------------------------------------")
    print(f"Window ID: {window.window_id}")
    print(f"Source: {window.source_id}")
    print(f"Schema: {window.schema_version}")
    print(
        f"Duration: "
        f"{window.window_duration_seconds:.3f} seconds"
    )
    print(f"Samples: {window.sample_count}")

    if window.samples:
        first = window.samples[0]
        last = window.samples[-1]

        print(
            "Start position: "
            f"({first.position.x:.3f}, "
            f"{first.position.y:.3f}, "
            f"{first.position.z:.3f})"
        )

        print(
            "End position: "
            f"({last.position.x:.3f}, "
            f"{last.position.y:.3f}, "
            f"{last.position.z:.3f})"
        )

        print(
            "Latest heading: "
            f"({last.heading.x:.3f}, "
            f"{last.heading.y:.3f}, "
            f"{last.heading.z:.3f})"
        )

    print("------------------------------------")

    output_path = get_window_file_path(
        source_id=window.source_id,
        window_id=window.window_id,
    )

    # Treat a repeated source_id + window_id as a retransmission rather
    # than adding the same training sequence multiple times.
    if output_path.exists():
        print(
            "Window already exists; skipping duplicate: "
            f"{output_path}"
        )
        print()

        return {
            "accepted": True,
            "duplicate": True,
            "window_id": window.window_id,
            "samples_received": len(window.samples),
            "saved_path": str(output_path),
        }

    try:
        saved_path = save_motion_window(window)

    except Exception as exc:
        print(f"Failed to save motion window: {exc}")
        print()

        raise HTTPException(
            status_code=500,
            detail="Failed to save robot motion-history window.",
        ) from exc

    print(f"Saved motion window: {saved_path}")
    print()

    return {
        "accepted": True,
        "duplicate": False,
        "window_id": window.window_id,
        "samples_received": len(window.samples),
        "saved_path": str(saved_path),
    }