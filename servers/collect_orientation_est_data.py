#!/usr/bin/env python3
"""
Quest-side Go2 orientation-estimation data collection server.

Expected install location:
    <repo_root>/servers/collect_orientation_est_data.py

Default data location:
    <repo_root>/robot_fov_estimation/data/forward_estimation_data

The server receives JSON payloads produced by SetRobotForwardQuest.cs at:

    POST /quest_robot_forward_sample

Each session is stored as:

    forward_estimation_data/
      <session_id>/
        session_metadata.json
        samples.csv
        samples.jsonl
        images/
          frame_000000000000.jpg
          frame_000000000001.jpg
          ...

All paths are resolved relative to the repository root inferred from this
script's location. No absolute project path is hard-coded.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =============================================================================
# Repository-relative paths
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_DATA_DIR_RELATIVE = Path(
    "robot_fov_estimation/data/forward_estimation_data"
)


def resolve_repo_relative_path(relative_path: Path) -> Path:
    """
    Resolve a path relative to the repository root and ensure it stays inside
    the repository tree.
    """
    relative_path = Path(relative_path)

    if relative_path.is_absolute():
        raise ValueError(
            f"Expected a repo-relative path, got absolute path: {relative_path}"
        )

    resolved = (REPO_ROOT / relative_path).resolve()
    repo_resolved = REPO_ROOT.resolve()

    try:
        resolved.relative_to(repo_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Path escapes repository root: {relative_path}"
        ) from exc

    return resolved


# =============================================================================
# FastAPI / Pydantic payload definitions
# =============================================================================

class Vector3Payload(BaseModel):
    x: float
    y: float
    z: float


class QuaternionPayload(BaseModel):
    x: float
    y: float
    z: float
    w: float


class RobotForwardQuestPayload(BaseModel):
    schema_version: int = 1

    session_id: str
    session_label: str = ""

    frame_id: int
    unity_frame_count: int

    capture_timestamp_unix: float
    capture_realtime_seconds: float

    measurement_timestamp_unix: float
    measurement_realtime_seconds: float

    calibration_timestamp_unix: float
    calibration_realtime_seconds: float

    measurement_minus_capture_seconds: float

    image_width: int
    image_height: int
    camera_eye: str

    camera_position_world: Vector3Payload
    camera_rotation_world: QuaternionPayload

    robot_center_world: Vector3Payload
    robot_forward_point_world: Vector3Payload

    robot_forward_world: Vector3Payload
    robot_forward_planar_world: Vector3Payload

    robot_forward_quest_yaw_deg: float

    camera_to_robot_world: Vector3Payload
    camera_to_robot_planar_world: Vector3Payload

    camera_to_robot_quest_yaw_deg: float

    camera_robot_distance_m: float
    camera_robot_horizontal_distance_m: float

    signed_relative_yaw_deg: float
    magnitude_yaw_deg: float
    body_axis_yaw_deg: float

    image_encoding: str = "jpeg_base64"
    image_jpeg_base64: str


# =============================================================================
# Dataset writer
# =============================================================================

CSV_FIELDNAMES = [
    "schema_version",
    "session_id",
    "session_label",
    "frame_id",
    "unity_frame_count",

    "capture_timestamp_unix",
    "capture_realtime_seconds",
    "measurement_timestamp_unix",
    "measurement_realtime_seconds",
    "calibration_timestamp_unix",
    "calibration_realtime_seconds",
    "measurement_minus_capture_seconds",

    "server_received_timestamp_unix",
    "server_received_timestamp_iso_utc",

    "image_width",
    "image_height",
    "camera_eye",
    "image_relative_path",
    "image_num_bytes",

    "camera_position_world_x",
    "camera_position_world_y",
    "camera_position_world_z",

    "camera_rotation_world_x",
    "camera_rotation_world_y",
    "camera_rotation_world_z",
    "camera_rotation_world_w",

    "robot_center_world_x",
    "robot_center_world_y",
    "robot_center_world_z",

    "robot_forward_point_world_x",
    "robot_forward_point_world_y",
    "robot_forward_point_world_z",

    "robot_forward_world_x",
    "robot_forward_world_y",
    "robot_forward_world_z",

    "robot_forward_planar_world_x",
    "robot_forward_planar_world_y",
    "robot_forward_planar_world_z",

    "robot_forward_quest_yaw_deg",

    "camera_to_robot_world_x",
    "camera_to_robot_world_y",
    "camera_to_robot_world_z",

    "camera_to_robot_planar_world_x",
    "camera_to_robot_planar_world_y",
    "camera_to_robot_planar_world_z",

    "camera_to_robot_quest_yaw_deg",

    "camera_robot_distance_m",
    "camera_robot_horizontal_distance_m",

    "signed_relative_yaw_deg",
    "magnitude_yaw_deg",
    "body_axis_yaw_deg",
]


def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """
    Pydantic v1/v2 compatibility.
    """
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _safe_component(value: str, fallback: str = "unnamed") -> str:
    value = str(value or "").strip()
    if not value:
        value = fallback

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._")

    if not value:
        value = fallback

    return value[:160]


def _utc_iso(timestamp_unix: float) -> str:
    return datetime.fromtimestamp(
        timestamp_unix,
        tz=timezone.utc,
    ).isoformat()


def _validate_finite_number(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise HTTPException(
            status_code=422,
            detail=f"{name} must be finite, got {value!r}",
        )


def _validate_payload_numbers(payload: RobotForwardQuestPayload) -> None:
    scalar_fields = {
        "capture_timestamp_unix": payload.capture_timestamp_unix,
        "capture_realtime_seconds": payload.capture_realtime_seconds,
        "measurement_timestamp_unix": payload.measurement_timestamp_unix,
        "measurement_realtime_seconds": payload.measurement_realtime_seconds,
        "calibration_timestamp_unix": payload.calibration_timestamp_unix,
        "calibration_realtime_seconds": payload.calibration_realtime_seconds,
        "measurement_minus_capture_seconds":
            payload.measurement_minus_capture_seconds,
        "robot_forward_quest_yaw_deg":
            payload.robot_forward_quest_yaw_deg,
        "camera_to_robot_quest_yaw_deg":
            payload.camera_to_robot_quest_yaw_deg,
        "camera_robot_distance_m":
            payload.camera_robot_distance_m,
        "camera_robot_horizontal_distance_m":
            payload.camera_robot_horizontal_distance_m,
        "signed_relative_yaw_deg":
            payload.signed_relative_yaw_deg,
        "magnitude_yaw_deg":
            payload.magnitude_yaw_deg,
        "body_axis_yaw_deg":
            payload.body_axis_yaw_deg,
    }

    for name, value in scalar_fields.items():
        _validate_finite_number(name, value)

    vectors = {
        "camera_position_world": payload.camera_position_world,
        "robot_center_world": payload.robot_center_world,
        "robot_forward_point_world": payload.robot_forward_point_world,
        "robot_forward_world": payload.robot_forward_world,
        "robot_forward_planar_world": payload.robot_forward_planar_world,
        "camera_to_robot_world": payload.camera_to_robot_world,
        "camera_to_robot_planar_world": payload.camera_to_robot_planar_world,
    }

    for name, vec in vectors.items():
        _validate_finite_number(f"{name}.x", vec.x)
        _validate_finite_number(f"{name}.y", vec.y)
        _validate_finite_number(f"{name}.z", vec.z)

    quat = payload.camera_rotation_world
    _validate_finite_number("camera_rotation_world.x", quat.x)
    _validate_finite_number("camera_rotation_world.y", quat.y)
    _validate_finite_number("camera_rotation_world.z", quat.z)
    _validate_finite_number("camera_rotation_world.w", quat.w)


class OrientationDatasetWriter:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()

    def save_sample(
        self,
        payload: RobotForwardQuestPayload,
    ) -> Dict[str, Any]:
        _validate_payload_numbers(payload)

        if payload.schema_version != 1:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Unsupported schema_version "
                    f"{payload.schema_version}; expected 1."
                ),
            )

        if payload.image_encoding != "jpeg_base64":
            raise HTTPException(
                status_code=422,
                detail=(
                    "Unsupported image_encoding "
                    f"{payload.image_encoding!r}; expected 'jpeg_base64'."
                ),
            )

        if payload.image_width <= 0 or payload.image_height <= 0:
            raise HTTPException(
                status_code=422,
                detail="image_width and image_height must be positive.",
            )

        safe_session_id = _safe_component(
            payload.session_id,
            fallback="unnamed_session",
        )

        session_dir = self.data_root / safe_session_id
        images_dir = session_dir / "images"

        session_metadata_path = (
            session_dir / "session_metadata.json"
        )
        samples_csv_path = session_dir / "samples.csv"
        samples_jsonl_path = session_dir / "samples.jsonl"

        image_filename = (
            f"frame_{int(payload.frame_id):012d}.jpg"
        )
        image_path = images_dir / image_filename
        image_relative_path = Path("images") / image_filename

        try:
            image_bytes = base64.b64decode(
                payload.image_jpeg_base64,
                validate=True,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid base64 JPEG data: {exc}",
            ) from exc

        if not image_bytes:
            raise HTTPException(
                status_code=422,
                detail="Decoded JPEG data is empty.",
            )

        # Minimal JPEG sanity check: SOI marker.
        if len(image_bytes) < 2 or image_bytes[:2] != b"\xff\xd8":
            raise HTTPException(
                status_code=422,
                detail="Decoded image does not appear to be a JPEG.",
            )

        server_received_unix = time.time()
        server_received_iso = _utc_iso(
            server_received_unix
        )

        with self._lock:
            session_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            images_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if image_path.exists():
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Frame {payload.frame_id} already exists "
                        f"for session {safe_session_id!r}."
                    ),
                )

            # Write image atomically.
            temp_image_path = (
                image_path.with_suffix(
                    image_path.suffix + ".tmp"
                )
            )

            try:
                with temp_image_path.open("wb") as f:
                    f.write(image_bytes)
                    f.flush()
                    os.fsync(f.fileno())

                temp_image_path.replace(image_path)
            finally:
                if temp_image_path.exists():
                    temp_image_path.unlink(
                        missing_ok=True
                    )

            row = self._make_csv_row(
                payload=payload,
                image_relative_path=image_relative_path,
                image_num_bytes=len(image_bytes),
                server_received_unix=server_received_unix,
                server_received_iso=server_received_iso,
            )

            self._append_csv(
                samples_csv_path,
                row,
            )

            self._append_jsonl(
                samples_jsonl_path,
                payload=payload,
                row=row,
            )

            self._update_session_metadata(
                session_metadata_path,
                payload=payload,
                safe_session_id=safe_session_id,
                latest_frame_id=int(payload.frame_id),
                latest_capture_timestamp_unix=float(
                    payload.capture_timestamp_unix
                ),
                server_received_unix=server_received_unix,
            )

        return {
            "ok": True,
            "session_id": safe_session_id,
            "frame_id": int(payload.frame_id),
            "image_relative_path": str(
                image_relative_path.as_posix()
            ),
            "session_relative_path": str(
                session_dir.relative_to(
                    REPO_ROOT
                ).as_posix()
            ),
        }

    def _make_csv_row(
        self,
        payload: RobotForwardQuestPayload,
        image_relative_path: Path,
        image_num_bytes: int,
        server_received_unix: float,
        server_received_iso: str,
    ) -> Dict[str, Any]:
        cp = payload.camera_position_world
        cr = payload.camera_rotation_world
        rc = payload.robot_center_world
        rfp = payload.robot_forward_point_world
        rfw = payload.robot_forward_world
        rfpw = payload.robot_forward_planar_world
        ctr = payload.camera_to_robot_world
        ctrp = payload.camera_to_robot_planar_world

        return {
            "schema_version":
                payload.schema_version,

            "session_id":
                payload.session_id,

            "session_label":
                payload.session_label,

            "frame_id":
                int(payload.frame_id),

            "unity_frame_count":
                int(payload.unity_frame_count),

            "capture_timestamp_unix":
                payload.capture_timestamp_unix,

            "capture_realtime_seconds":
                payload.capture_realtime_seconds,

            "measurement_timestamp_unix":
                payload.measurement_timestamp_unix,

            "measurement_realtime_seconds":
                payload.measurement_realtime_seconds,

            "calibration_timestamp_unix":
                payload.calibration_timestamp_unix,

            "calibration_realtime_seconds":
                payload.calibration_realtime_seconds,

            "measurement_minus_capture_seconds":
                payload.measurement_minus_capture_seconds,

            "server_received_timestamp_unix":
                server_received_unix,

            "server_received_timestamp_iso_utc":
                server_received_iso,

            "image_width":
                int(payload.image_width),

            "image_height":
                int(payload.image_height),

            "camera_eye":
                payload.camera_eye,

            "image_relative_path":
                image_relative_path.as_posix(),

            "image_num_bytes":
                int(image_num_bytes),

            "camera_position_world_x": cp.x,
            "camera_position_world_y": cp.y,
            "camera_position_world_z": cp.z,

            "camera_rotation_world_x": cr.x,
            "camera_rotation_world_y": cr.y,
            "camera_rotation_world_z": cr.z,
            "camera_rotation_world_w": cr.w,

            "robot_center_world_x": rc.x,
            "robot_center_world_y": rc.y,
            "robot_center_world_z": rc.z,

            "robot_forward_point_world_x": rfp.x,
            "robot_forward_point_world_y": rfp.y,
            "robot_forward_point_world_z": rfp.z,

            "robot_forward_world_x": rfw.x,
            "robot_forward_world_y": rfw.y,
            "robot_forward_world_z": rfw.z,

            "robot_forward_planar_world_x": rfpw.x,
            "robot_forward_planar_world_y": rfpw.y,
            "robot_forward_planar_world_z": rfpw.z,

            "robot_forward_quest_yaw_deg":
                payload.robot_forward_quest_yaw_deg,

            "camera_to_robot_world_x": ctr.x,
            "camera_to_robot_world_y": ctr.y,
            "camera_to_robot_world_z": ctr.z,

            "camera_to_robot_planar_world_x": ctrp.x,
            "camera_to_robot_planar_world_y": ctrp.y,
            "camera_to_robot_planar_world_z": ctrp.z,

            "camera_to_robot_quest_yaw_deg":
                payload.camera_to_robot_quest_yaw_deg,

            "camera_robot_distance_m":
                payload.camera_robot_distance_m,

            "camera_robot_horizontal_distance_m":
                payload.camera_robot_horizontal_distance_m,

            "signed_relative_yaw_deg":
                payload.signed_relative_yaw_deg,

            "magnitude_yaw_deg":
                payload.magnitude_yaw_deg,

            "body_axis_yaw_deg":
                payload.body_axis_yaw_deg,
        }

    def _append_csv(
        self,
        path: Path,
        row: Dict[str, Any],
    ) -> None:
        file_exists = path.exists()

        with path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=CSV_FIELDNAMES,
                extrasaction="ignore",
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def _append_jsonl(
        self,
        path: Path,
        payload: RobotForwardQuestPayload,
        row: Dict[str, Any],
    ) -> None:
        payload_dict = _model_to_dict(
            payload
        )

        # Do not duplicate the very large base64 image in JSONL.
        payload_dict.pop(
            "image_jpeg_base64",
            None,
        )

        record = {
            "payload": payload_dict,
            "server": {
                "server_received_timestamp_unix":
                    row[
                        "server_received_timestamp_unix"
                    ],
                "server_received_timestamp_iso_utc":
                    row[
                        "server_received_timestamp_iso_utc"
                    ],
                "image_relative_path":
                    row["image_relative_path"],
                "image_num_bytes":
                    row["image_num_bytes"],
            },
        }

        line = json.dumps(
            record,
            separators=(",", ":"),
            allow_nan=False,
        )

        with path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(line)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    def _update_session_metadata(
        self,
        path: Path,
        payload: RobotForwardQuestPayload,
        safe_session_id: str,
        latest_frame_id: int,
        latest_capture_timestamp_unix: float,
        server_received_unix: float,
    ) -> None:
        if path.exists():
            try:
                metadata = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                metadata = {}
        else:
            metadata = {}

        if not metadata:
            metadata = {
                "schema_version":
                    int(payload.schema_version),

                "session_id":
                    payload.session_id,

                "safe_session_directory_name":
                    safe_session_id,

                "session_label":
                    payload.session_label,

                "created_server_timestamp_unix":
                    server_received_unix,

                "created_server_timestamp_iso_utc":
                    _utc_iso(
                        server_received_unix
                    ),

                "first_frame_id":
                    latest_frame_id,

                "first_capture_timestamp_unix":
                    latest_capture_timestamp_unix,

                "camera_eye":
                    payload.camera_eye,

                "image_width":
                    int(payload.image_width),

                "image_height":
                    int(payload.image_height),

                "calibration_timestamp_unix":
                    payload.calibration_timestamp_unix,

                "calibration_realtime_seconds":
                    payload.calibration_realtime_seconds,

                "initial_robot_center_world":
                    _model_to_dict(
                        payload.robot_center_world
                    ),

                "initial_robot_forward_point_world":
                    _model_to_dict(
                        payload.robot_forward_point_world
                    ),

                "initial_robot_forward_world":
                    _model_to_dict(
                        payload.robot_forward_world
                    ),

                "initial_robot_forward_planar_world":
                    _model_to_dict(
                        payload.robot_forward_planar_world
                    ),

                "initial_robot_forward_quest_yaw_deg":
                    payload.robot_forward_quest_yaw_deg,

                "num_samples":
                    0,
            }

        metadata["session_label"] = (
            payload.session_label
        )

        metadata["latest_frame_id"] = (
            latest_frame_id
        )

        metadata[
            "latest_capture_timestamp_unix"
        ] = latest_capture_timestamp_unix

        metadata[
            "latest_server_timestamp_unix"
        ] = server_received_unix

        metadata[
            "latest_server_timestamp_iso_utc"
        ] = _utc_iso(
            server_received_unix
        )

        metadata["num_samples"] = (
            int(metadata.get("num_samples", 0))
            + 1
        )

        temp_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        try:
            temp_path.write_text(
                json.dumps(
                    metadata,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink(
                    missing_ok=True
                )


# =============================================================================
# App factory / routes
# =============================================================================

DATA_ROOT = resolve_repo_relative_path(
    DEFAULT_DATA_DIR_RELATIVE
)

writer = OrientationDatasetWriter(
    DATA_ROOT
)

app = FastAPI(
    title="Quest Go2 Orientation Data Collector",
    version="1.0.0",
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "repo_root": str(REPO_ROOT),
        "data_root": str(DATA_ROOT),
        "endpoint": "/quest_robot_forward_sample",
        "server_timestamp_unix": time.time(),
    }


@app.post("/quest_robot_forward_sample")
def collect_orientation_sample(
    payload: RobotForwardQuestPayload,
) -> Dict[str, Any]:
    return writer.save_sample(
        payload
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Receive SetRobotForwardQuest.cs samples and "
            "store them as a repo-relative structured dataset."
        )
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help=(
            "Interface to bind. Default: 0.0.0.0"
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port. Default: 8000",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(
            DEFAULT_DATA_DIR_RELATIVE
        ),
        help=(
            "Data directory RELATIVE to repo root. "
            "Default: "
            "robot_fov_estimation/data/"
            "forward_estimation_data"
        ),
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help=(
            "Enable uvicorn auto-reload. Intended only "
            "for development."
        ),
    )

    return parser.parse_args()


def main() -> None:
    global DATA_ROOT
    global writer

    args = parse_args()

    data_dir_relative = Path(
        args.data_dir
    )

    try:
        DATA_ROOT = resolve_repo_relative_path(
            data_dir_relative
        )
    except ValueError as exc:
        raise SystemExit(
            f"Invalid --data-dir: {exc}"
        ) from exc

    writer = OrientationDatasetWriter(
        DATA_ROOT
    )

    print(
        "Quest Go2 Orientation Data Collector"
    )
    print(
        f"Repo root: {REPO_ROOT}"
    )
    print(
        f"Data root: {DATA_ROOT}"
    )
    print(
        "POST endpoint: "
        "/quest_robot_forward_sample"
    )
    print(
        f"Health endpoint: http://"
        f"{args.host}:{args.port}/health"
    )

    uvicorn.run(
        app,
        host=args.host,
        port=int(args.port),
        reload=bool(args.reload),
    )


if __name__ == "__main__":
    main()
