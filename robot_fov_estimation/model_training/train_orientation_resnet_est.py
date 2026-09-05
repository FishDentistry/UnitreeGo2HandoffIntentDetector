#!/usr/bin/env python3
"""
Train the earlier end-to-end ResNet-18 + orientation-head model on the NEW
Quest-native Unitree Go2 forward-orientation dataset.

This is intended as a direct comparison against:
    finetune_orient_anything_forward_estimation_YOLO.py

Model retained from the earlier ResNet experiment
-------------------------------------------------
    YOLO-cropped Go2 RGB
        -> ImageNet pretrained ResNet-18, fully fine-tuned
        -> 512-D embedding
        -> Linear(512, hidden_dim=128)
        -> ReLU
        -> Dropout(0.10)
        -> Linear(128, 2)
        -> L2 normalization
        -> [sin(alpha), cos(alpha)]

Training target:
    [sin(alpha), cos(alpha)]

Loss:
    mean(1 - dot(prediction_unit_vector, target_unit_vector))

Prediction:
    alpha_hat = atan2(pred_sin, pred_cos)

The model architecture and circular regression objective are intentionally kept
equivalent to the user's earlier ResNet-18 orientation script. What has changed
is the DATA/EVALUATION pipeline so that it matches the new Orient Anything
experiment.

New-data pipeline
-----------------
Expected data:
    <repo>/
      robot_fov_estimation/
        data/
          forward_estimation_data/
            <session_id>/
              samples.csv
              images/
                frame_*.jpg

Ground truth is NOT taken blindly from the saved yaw column. For every sample,
the script recomputes:

    camera_to_robot =
        robot_center_world - camera_position_world

    robot_forward =
        robot_forward_point_world - robot_center_world

After projecting both onto Unity's XZ plane:

    alpha = SignedAngle(
        camera_to_robot_planar,
        robot_forward_planar,
        +Y
    )

The recomputed target is checked against `signed_relative_yaw_deg`; the script
fails if they disagree beyond the configured tolerance.

Deployment-matched YOLO crop
----------------------------
By default this script first attempts to REUSE the exact YOLO crops and
`yolo_crop_manifest.csv` produced by the current Orient Anything trainer:

    robot_fov_estimation/
      weights/
        orient_anything_forward_estimation/
          yolo_crop_manifest.csv
          yolo_crops/

If that manifest fully covers the current raw dataset, only rows marked as
successful detections are used and their exact existing crop files are loaded.
This gives the strongest possible OA-vs-ResNet comparison: both models see the
same image bytes and the same detected-frame subset.

If the OA crop manifest is absent or does not fully cover the current raw
dataset, this script automatically generates its own crops using the same older
Go2 detector configuration:

    detector wrapper:
        robot_fov_estimation.src.go2_yolo_det_wrapper.YOLOGo2Detector

    class:
        "robot dog"

    confidence:
        0.05

    IoU threshold:
        0.50

    YOLO image size:
        640

    crop padding:
        0.15

    detector weights:
        if --robot-obj-det-weights-path is omitted, YOLOGo2Detector uses its
        normal project default weights, matching the earlier orientation scripts.

A failed YOLO detection is skipped by default. It NEVER silently becomes a
full-image training example.

Evaluation
----------
Default:
    --split-mode leave_one_session_out

For each fold:
    one whole session = TEST
    one DIFFERENT whole session = VALIDATION
    all remaining whole sessions = TRAIN

This exactly matches the session selection rule in the new OA trainer:
validation is the next session in chronological session order after the held-out
test session.

Training records are temporally thinned by default:
    --temporal-thin-seconds 0.5

and session-balanced sampling is enabled by default.

Unlike the older ResNet script, checkpoint selection defaults to VALIDATION
MEAN angular error rather than validation median. This is intentional so the
checkpoint-selection metric matches the current Orient Anything experiment.
Use:
    --selection-metric median
to restore the old median-based model-selection behavior.

Outputs
-------
Default output:
    robot_fov_estimation/
      weights/
        resnet18_forward_estimation/

Per LOSO fold:
    leave_one_session_out_folds/<test_session_id>/
        best_go2_resnet18_forward_estimation.pt
        split_manifest.csv
        training_history.csv
        train_predictions.csv
        validation_predictions.csv
        test_predictions.csv
        error_by_distance.csv
        summary.json

Aggregate:
    leave_one_session_out_summary.csv
    leave_one_session_out_summary.json
    leave_one_session_out_predictions.csv

Also written at the top-level:
    session_data_quality.csv
    dataset_configuration.json

After LOSO evaluation completes, the script retrains once more on all available
data and writes the final checkpoint under the default weights output directory.

Typical run
-----------
From the UnitreeGo2HandoffIntentDetector repository root:

    python -m robot_fov_estimation.model_training.train_go2_orientation_resnet18_new_data

No arguments are required for the normal comparison experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision.models import ResNet18_Weights


# =============================================================================
# Paths
# =============================================================================

def find_repo_root() -> Path:
    here = Path(__file__).resolve()

    for parent in [
        here.parent,
        *here.parents,
        Path.cwd(),
        *Path.cwd().parents,
    ]:
        if parent.name == "UnitreeGo2HandoffIntentDetector":
            return parent.resolve()

        candidate = parent / "UnitreeGo2HandoffIntentDetector"

        if candidate.is_dir():
            return candidate.resolve()

    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()

DEFAULT_DATA_ROOT = (
    REPO_ROOT
    / "robot_fov_estimation"
    / "data"
    / "forward_estimation_data"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "robot_fov_estimation"
    / "outputs"
    / "resnet18_forward_estimation_eval"
)

DEFAULT_WEIGHTS_OUTPUT_DIR = (
    REPO_ROOT
    / "robot_fov_estimation"
    / "outputs"
    / "resnet18_forward_estimation_weights"
)

DEFAULT_OA_OUTPUT_DIR = (
    REPO_ROOT
    / "robot_fov_estimation"
    / "outputs"
    / "orient_anything_forward_estimation_eval"
)

DEFAULT_OA_YOLO_MANIFEST = (
    DEFAULT_OA_OUTPUT_DIR
    / "yolo_crop_manifest.csv"
)


# =============================================================================
# Generic helpers
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(value: str) -> str:
    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value).strip(),
    ).strip("._")

    return value or "unnamed"


def write_rows(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    keys: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def finite(
    row: Dict[str, str],
    key: str,
) -> float:
    try:
        value = float(
            row[key]
        )
    except Exception as exc:
        raise ValueError(
            f"Invalid {key}={row.get(key)!r}"
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite {key}={value}"
        )

    return value


# =============================================================================
# Circular-angle helpers / Quest geometry
# =============================================================================

def wrap_deg_scalar(
    angle_deg: float,
) -> float:
    return (
        (
            float(angle_deg)
            + 180.0
        )
        % 360.0
        - 180.0
    )


def wrap_deg_np(
    angle_deg: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        angle_deg,
        dtype=np.float64,
    )

    return (
        (
            values
            + 180.0
        )
        % 360.0
        - 180.0
    )


def circular_error_deg(
    prediction_deg: np.ndarray,
    target_deg: np.ndarray,
) -> np.ndarray:
    return np.abs(
        wrap_deg_np(
            np.asarray(
                prediction_deg,
                dtype=np.float64,
            )
            - np.asarray(
                target_deg,
                dtype=np.float64,
            )
        )
    )


def recompute_signed_yaw(
    camera: Tuple[float, float, float],
    center: Tuple[float, float, float],
    forward_point: Tuple[float, float, float],
) -> Tuple[
    float,
    float,
    float,
]:
    """
    Match Unity:

        Vector3.SignedAngle(
            cameraToRobotPlanar,
            robotForwardPlanar,
            Vector3.up
        )

    For:
        from=(ax,0,az)
        to=(bx,0,bz)

    Unity's +Y signed angle is:

        sin = az*bx - ax*bz
        cos = ax*bx + az*bz
        angle = atan2(sin, cos)
    """
    cx, cy, cz = camera
    rx, ry, rz = center
    px, py, pz = forward_point

    ray_x = rx - cx
    ray_z = rz - cz

    fwd_x = px - rx
    fwd_z = pz - rz

    ray_norm = math.hypot(
        ray_x,
        ray_z,
    )

    fwd_norm = math.hypot(
        fwd_x,
        fwd_z,
    )

    if ray_norm < 1.0e-8:
        raise ValueError(
            "Camera and robot center coincide in XZ."
        )

    if fwd_norm < 1.0e-8:
        raise ValueError(
            "Robot center/forward point do not define a horizontal forward direction."
        )

    ax = ray_x / ray_norm
    az = ray_z / ray_norm

    bx = fwd_x / fwd_norm
    bz = fwd_z / fwd_norm

    sin_term = (
        az * bx
        - ax * bz
    )

    cos_term = (
        ax * bx
        + az * bz
    )

    alpha_deg = wrap_deg_scalar(
        math.degrees(
            math.atan2(
                sin_term,
                cos_term,
            )
        )
    )

    distance_3d = math.sqrt(
        (rx - cx) ** 2
        + (ry - cy) ** 2
        + (rz - cz) ** 2
    )

    horizontal_distance = ray_norm

    return (
        alpha_deg,
        distance_3d,
        horizontal_distance,
    )


def angle_to_sincos(
    angle_deg: float,
) -> Tuple[
    float,
    float,
]:
    radians = math.radians(
        float(angle_deg)
    )

    return (
        math.sin(
            radians
        ),
        math.cos(
            radians
        ),
    )


def sincos_to_degrees_np(
    sincos: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        sincos,
        dtype=np.float64,
    )

    if (
        values.ndim != 2
        or values.shape[1] != 2
    ):
        raise ValueError(
            f"Expected Nx2 sin/cos predictions, got {values.shape}"
        )

    return np.degrees(
        np.arctan2(
            values[:, 0],
            values[:, 1],
        )
    )


# =============================================================================
# Records
# =============================================================================

@dataclass
class Record:
    session_id: str
    session_label: str

    frame_id: int

    capture_timestamp_unix: float
    capture_realtime_seconds: float
    measurement_minus_capture_seconds: float

    image_path: Path

    target_signed_deg: float
    saved_signed_deg: float
    label_error_deg: float

    distance_m: float
    horizontal_distance_m: float

    camera_x: float
    camera_y: float
    camera_z: float

    center_x: float
    center_y: float
    center_z: float

    forward_x: float
    forward_y: float
    forward_z: float


REQUIRED_COLUMNS = {
    "session_id",
    "frame_id",
    "capture_timestamp_unix",
    "capture_realtime_seconds",
    "measurement_minus_capture_seconds",
    "image_relative_path",

    "camera_position_world_x",
    "camera_position_world_y",
    "camera_position_world_z",

    "robot_center_world_x",
    "robot_center_world_y",
    "robot_center_world_z",

    "robot_forward_point_world_x",
    "robot_forward_point_world_y",
    "robot_forward_point_world_z",

    "signed_relative_yaw_deg",
}


def discover_sessions(
    data_root: Path,
) -> List[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Data root does not exist: {data_root}"
        )

    sessions = sorted(
        path
        for path
        in data_root.iterdir()
        if (
            path.is_dir()
            and (
                path
                / "samples.csv"
            ).is_file()
        )
    )

    if not sessions:
        raise RuntimeError(
            f"No session directories containing samples.csv under {data_root}"
        )

    return sessions


def load_records(
    data_root: Path,
    allowed_sessions: Optional[
        Sequence[str]
    ],
    label_tolerance_deg: float,
) -> List[Record]:
    allowed = (
        set(
            allowed_sessions
        )
        if allowed_sessions
        else None
    )

    records: List[Record] = []
    seen = set()

    for session_dir in discover_sessions(
        data_root
    ):
        csv_path = (
            session_dir
            / "samples.csv"
        )

        with csv_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.DictReader(
                f
            )

            missing = (
                REQUIRED_COLUMNS
                - set(
                    reader.fieldnames
                    or []
                )
            )

            if missing:
                raise KeyError(
                    f"{csv_path} missing required columns: {sorted(missing)}"
                )

            for line_number, row in enumerate(
                reader,
                start=2,
            ):
                session_id = str(
                    row[
                        "session_id"
                    ]
                ).strip()

                if (
                    allowed is not None
                    and session_id not in allowed
                ):
                    continue

                frame_id = int(
                    row[
                        "frame_id"
                    ]
                )

                key = (
                    session_id,
                    frame_id,
                )

                if key in seen:
                    raise RuntimeError(
                        f"Duplicate sample: {key}"
                    )

                seen.add(
                    key
                )

                image_path = (
                    session_dir
                    / str(
                        row[
                            "image_relative_path"
                        ]
                    )
                ).resolve()

                if not image_path.is_file():
                    raise FileNotFoundError(
                        image_path
                    )

                camera = tuple(
                    finite(
                        row,
                        key_name,
                    )
                    for key_name
                    in (
                        "camera_position_world_x",
                        "camera_position_world_y",
                        "camera_position_world_z",
                    )
                )

                center = tuple(
                    finite(
                        row,
                        key_name,
                    )
                    for key_name
                    in (
                        "robot_center_world_x",
                        "robot_center_world_y",
                        "robot_center_world_z",
                    )
                )

                forward_point = tuple(
                    finite(
                        row,
                        key_name,
                    )
                    for key_name
                    in (
                        "robot_forward_point_world_x",
                        "robot_forward_point_world_y",
                        "robot_forward_point_world_z",
                    )
                )

                (
                    recomputed_yaw,
                    distance_m,
                    horizontal_distance_m,
                ) = recompute_signed_yaw(
                    camera,
                    center,
                    forward_point,
                )

                saved_yaw = wrap_deg_scalar(
                    finite(
                        row,
                        "signed_relative_yaw_deg",
                    )
                )

                label_error = float(
                    circular_error_deg(
                        np.asarray(
                            [
                                recomputed_yaw
                            ],
                            dtype=np.float64,
                        ),
                        np.asarray(
                            [
                                saved_yaw
                            ],
                            dtype=np.float64,
                        ),
                    )[0]
                )

                if (
                    label_error
                    > label_tolerance_deg
                ):
                    raise RuntimeError(
                        "Quest geometry label consistency check failed.\n"
                        f"  file: {csv_path}\n"
                        f"  line: {line_number}\n"
                        f"  session: {session_id}\n"
                        f"  frame: {frame_id}\n"
                        f"  recomputed: {recomputed_yaw:.6f} deg\n"
                        f"  saved:      {saved_yaw:.6f} deg\n"
                        f"  error:      {label_error:.6f} deg\n"
                        f"  tolerance:  {label_tolerance_deg:.6f} deg"
                    )

                records.append(
                    Record(
                        session_id=
                            session_id,

                        session_label=
                            str(
                                row.get(
                                    "session_label",
                                    "",
                                )
                                or ""
                            ),

                        frame_id=
                            frame_id,

                        capture_timestamp_unix=
                            finite(
                                row,
                                "capture_timestamp_unix",
                            ),

                        capture_realtime_seconds=
                            finite(
                                row,
                                "capture_realtime_seconds",
                            ),

                        measurement_minus_capture_seconds=
                            finite(
                                row,
                                "measurement_minus_capture_seconds",
                            ),

                        image_path=
                            image_path,

                        target_signed_deg=
                            recomputed_yaw,

                        saved_signed_deg=
                            saved_yaw,

                        label_error_deg=
                            label_error,

                        distance_m=
                            distance_m,

                        horizontal_distance_m=
                            horizontal_distance_m,

                        camera_x=
                            camera[0],

                        camera_y=
                            camera[1],

                        camera_z=
                            camera[2],

                        center_x=
                            center[0],

                        center_y=
                            center[1],

                        center_z=
                            center[2],

                        forward_x=
                            forward_point[0],

                        forward_y=
                            forward_point[1],

                        forward_z=
                            forward_point[2],
                    )
                )

    if not records:
        raise RuntimeError(
            "No samples were loaded."
        )

    records.sort(
        key=lambda record: (
            record.capture_timestamp_unix,
            record.session_id,
            record.frame_id,
        )
    )

    return records


def copy_record_with_image(
    record: Record,
    image_path: Path,
) -> Record:
    values = asdict(
        record
    )

    values[
        "image_path"
    ] = image_path.resolve()

    return Record(
        **values
    )


def ordered_session_ids(
    records: Sequence[Record],
) -> List[str]:
    first_time: Dict[
        str,
        float,
    ] = {}

    for record in records:
        first_time[
            record.session_id
        ] = min(
            first_time.get(
                record.session_id,
                math.inf,
            ),
            record.capture_timestamp_unix,
        )

    return sorted(
        first_time.keys(),
        key=lambda session_id: (
            first_time[
                session_id
            ],
            session_id,
        ),
    )


def group_sessions(
    records: Sequence[Record],
) -> Dict[
    str,
    List[Record],
]:
    grouped: Dict[
        str,
        List[Record],
    ] = defaultdict(
        list
    )

    for record in records:
        grouped[
            record.session_id
        ].append(
            record
        )

    for session_id in grouped:
        grouped[
            session_id
        ].sort(
            key=lambda record: (
                record.capture_timestamp_unix,
                record.frame_id,
            )
        )

    return dict(
        grouped
    )


# =============================================================================
# Splitting
# =============================================================================

@dataclass
class Split:
    name: str
    train: List[Record]
    validation: List[Record]
    test: List[Record]
    purged: List[Record]
    info: Dict[str, Any]


def loso_splits(
    records: Sequence[Record],
    only_test_session: Optional[str],
) -> List[Split]:
    sessions = ordered_session_ids(
        records
    )

    if len(
        sessions
    ) < 3:
        raise RuntimeError(
            "leave_one_session_out requires at least 3 sessions."
        )

    if len(
        sessions
    ) == 3:
        print(
            "WARNING: only three sessions are available; "
            "each LOSO fold has only one training session."
        )

    grouped = group_sessions(
        records
    )

    splits: List[Split] = []

    for test_index, test_session in enumerate(
        sessions
    ):
        if (
            only_test_session is not None
            and test_session
            != only_test_session
        ):
            continue

        validation_session = sessions[
            (
                test_index
                + 1
            )
            % len(
                sessions
            )
        ]

        train_sessions = [
            session_id
            for session_id
            in sessions
            if session_id
            not in {
                test_session,
                validation_session,
            }
        ]

        train_records: List[
            Record
        ] = []

        for session_id in train_sessions:
            train_records.extend(
                grouped[
                    session_id
                ]
            )

        splits.append(
            Split(
                name=
                    f"test_{safe_name(test_session)}",

                train=
                    train_records,

                validation=
                    list(
                        grouped[
                            validation_session
                        ]
                    ),

                test=
                    list(
                        grouped[
                            test_session
                        ]
                    ),

                purged=
                    [],

                info={
                    "strategy":
                        "whole-session leave-one-session-out",

                    "train_session_ids":
                        train_sessions,

                    "validation_session_id":
                        validation_session,

                    "test_session_id":
                        test_session,
                },
            )
        )

    if not splits:
        raise ValueError(
            f"No test session matched {only_test_session!r}"
        )

    return splits


def session_holdout(
    records: Sequence[Record],
    validation_session: str,
    test_session: str,
) -> Split:
    sessions = ordered_session_ids(
        records
    )

    if (
        validation_session
        not in sessions
        or test_session
        not in sessions
        or validation_session
        == test_session
    ):
        raise ValueError(
            f"Invalid validation/test session. Available: {sessions}"
        )

    grouped = group_sessions(
        records
    )

    train_sessions = [
        session_id
        for session_id
        in sessions
        if session_id
        not in {
            validation_session,
            test_session,
        }
    ]

    if not train_sessions:
        raise RuntimeError(
            "No training sessions remain."
        )

    train_records: List[
        Record
    ] = []

    for session_id in train_sessions:
        train_records.extend(
            grouped[
                session_id
            ]
        )

    return Split(
        name=
            "session_holdout",

        train=
            train_records,

        validation=
            list(
                grouped[
                    validation_session
                ]
            ),

        test=
            list(
                grouped[
                    test_session
                ]
            ),

        purged=
            [],

        info={
            "strategy":
                "whole-session holdout",

            "train_session_ids":
                train_sessions,

            "validation_session_id":
                validation_session,

            "test_session_id":
                test_session,
        },
    )


def temporal_split(
    records: Sequence[Record],
    train_fraction: float,
    validation_fraction: float,
    gap_seconds: float,
) -> Split:
    grouped = group_sessions(
        records
    )

    train: List[Record] = []
    validation: List[Record] = []
    test: List[Record] = []
    purged: List[Record] = []

    for session_id in ordered_session_ids(
        records
    ):
        session_records = grouped[
            session_id
        ]

        n = len(
            session_records
        )

        if n < 3:
            raise RuntimeError(
                f"Session {session_id} has only {n} samples."
            )

        train_end = min(
            max(
                1,
                int(
                    n
                    * train_fraction
                ),
            ),
            n - 2,
        )

        validation_end = min(
            max(
                train_end + 1,
                int(
                    n
                    * (
                        train_fraction
                        + validation_fraction
                    )
                ),
            ),
            n - 1,
        )

        first_boundary = 0.5 * (
            session_records[
                train_end - 1
            ].capture_timestamp_unix
            + session_records[
                train_end
            ].capture_timestamp_unix
        )

        second_boundary = 0.5 * (
            session_records[
                validation_end - 1
            ].capture_timestamp_unix
            + session_records[
                validation_end
            ].capture_timestamp_unix
        )

        first_left = (
            first_boundary
            - 0.5
            * gap_seconds
        )

        first_right = (
            first_boundary
            + 0.5
            * gap_seconds
        )

        second_left = (
            second_boundary
            - 0.5
            * gap_seconds
        )

        second_right = (
            second_boundary
            + 0.5
            * gap_seconds
        )

        for index, record in enumerate(
            session_records
        ):
            timestamp = (
                record.capture_timestamp_unix
            )

            if index < train_end:
                if (
                    gap_seconds > 0.0
                    and timestamp
                    > first_left
                ):
                    purged.append(
                        record
                    )
                else:
                    train.append(
                        record
                    )

            elif index < validation_end:
                if (
                    gap_seconds > 0.0
                    and (
                        timestamp
                        < first_right
                        or timestamp
                        > second_left
                    )
                ):
                    purged.append(
                        record
                    )
                else:
                    validation.append(
                        record
                    )

            else:
                if (
                    gap_seconds > 0.0
                    and timestamp
                    < second_right
                ):
                    purged.append(
                        record
                    )
                else:
                    test.append(
                        record
                    )

    if (
        not train
        or not validation
        or not test
    ):
        raise RuntimeError(
            "Temporal split produced an empty split."
        )

    return Split(
        name=
            "temporal",

        train=
            train,

        validation=
            validation,

        test=
            test,

        purged=
            purged,

        info={
            "strategy":
                "within-session chronological",

            "train_fraction":
                train_fraction,

            "validation_fraction":
                validation_fraction,

            "gap_seconds":
                gap_seconds,
        },
    )


def random_split(
    records: Sequence[Record],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> Split:
    items = list(
        records
    )

    random.Random(
        seed
    ).shuffle(
        items
    )

    n = len(
        items
    )

    train_count = max(
        1,
        int(
            n
            * train_fraction
        ),
    )

    validation_count = max(
        1,
        int(
            n
            * validation_fraction
        ),
    )

    if (
        train_count
        + validation_count
        >= n
    ):
        validation_count = max(
            1,
            n
            - train_count
            - 1,
        )

    test_start = (
        train_count
        + validation_count
    )

    if test_start >= n:
        raise RuntimeError(
            "Random split produced an empty test split."
        )

    return Split(
        name=
            "random",

        train=
            items[
                :train_count
            ],

        validation=
            items[
                train_count:
                test_start
            ],

        test=
            items[
                test_start:
            ],

        purged=
            [],

        info={
            "strategy":
                "random-frame diagnostic; temporal leakage possible",

            "train_fraction":
                train_fraction,

            "validation_fraction":
                validation_fraction,

            "seed":
                seed,
        },
    )


def thin_training(
    records: Sequence[Record],
    spacing_seconds: float,
) -> Tuple[
    List[Record],
    List[Record],
]:
    if spacing_seconds <= 0.0:
        return (
            list(
                records
            ),
            [],
        )

    grouped = group_sessions(
        records
    )

    kept: List[Record] = []
    removed: List[Record] = []

    for session_id in ordered_session_ids(
        records
    ):
        last_kept_time = (
            -math.inf
        )

        for record in grouped[
            session_id
        ]:
            if (
                record.capture_timestamp_unix
                - last_kept_time
                >= spacing_seconds
            ):
                kept.append(
                    record
                )

                last_kept_time = (
                    record.capture_timestamp_unix
                )

            else:
                removed.append(
                    record
                )

    return (
        kept,
        removed,
    )


# =============================================================================
# Deployment-matched YOLO crops
# =============================================================================

def parse_bool_cell(
    value: Any,
) -> bool:
    return str(
        value
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def try_reuse_oa_yolo_crops(
    raw_records: Sequence[Record],
    manifest_path: Path,
) -> Optional[
    Tuple[
        List[Record],
        List[Dict[str, Any]],
    ]
]:
    """
    Reuse OA's exact crop files only if its manifest fully accounts for every
    current raw record. If not, return None so we regenerate crops ourselves.
    """
    if not manifest_path.is_file():
        return None

    rows: List[
        Dict[str, Any]
    ] = []

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(
            f
        )

        for row in reader:
            rows.append(
                dict(
                    row
                )
            )

    by_key = {
        (
            str(
                row.get(
                    "session_id",
                    "",
                )
            ).strip(),
            int(
                row[
                    "frame_id"
                ]
            ),
        ):
            row
        for row
        in rows
        if str(
            row.get(
                "frame_id",
                "",
            )
        ).strip()
        != ""
    }

    raw_keys = {
        (
            record.session_id,
            record.frame_id,
        )
        for record
        in raw_records
    }

    manifest_keys = set(
        by_key.keys()
    )

    if not raw_keys.issubset(
        manifest_keys
    ):
        missing = (
            raw_keys
            - manifest_keys
        )

        print(
            "OA YOLO manifest exists but does not cover the current "
            f"raw dataset ({len(missing)} sample(s) missing). "
            "ResNet will regenerate crops instead."
        )

        return None

    usable: List[
        Record
    ] = []

    reuse_manifest: List[
        Dict[str, Any]
    ] = []

    for record in raw_records:
        key = (
            record.session_id,
            record.frame_id,
        )

        row = by_key[
            key
        ]

        detected = (
            parse_bool_cell(
                row.get(
                    "robot_detected",
                    False,
                )
            )
            or str(
                row.get(
                    "status",
                    "",
                )
            ).strip().lower()
            == "ok"
        )

        crop_text = str(
            row.get(
                "crop_image_path",
                "",
            )
            or ""
        ).strip()

        if not detected:
            reuse_manifest.append(
                {
                    "session_id":
                        record.session_id,

                    "frame_id":
                        record.frame_id,

                    "source":
                        "orient_anything_yolo_manifest",

                    "included":
                        False,

                    "crop_image_path":
                        "",

                    "status":
                        str(
                            row.get(
                                "status",
                                "no_detection",
                            )
                        ),
                }
            )

            continue

        crop_path = Path(
            crop_text
        ).expanduser().resolve()

        if not crop_path.is_file():
            print(
                "OA YOLO manifest refers to a crop file that no longer exists:\n"
                f"  {crop_path}\n"
                "ResNet will regenerate all crops instead."
            )

            return None

        usable.append(
            copy_record_with_image(
                record,
                crop_path,
            )
        )

        reuse_manifest.append(
            {
                "session_id":
                    record.session_id,

                "frame_id":
                    record.frame_id,

                "source":
                    "orient_anything_yolo_manifest",

                "included":
                    True,

                "crop_image_path":
                    str(
                        crop_path
                    ),

                "yolo_score":
                    row.get(
                        "yolo_score",
                        "",
                    ),

                "raw_x1":
                    row.get(
                        "raw_x1",
                        "",
                    ),

                "raw_y1":
                    row.get(
                        "raw_y1",
                        "",
                    ),

                "raw_x2":
                    row.get(
                        "raw_x2",
                        "",
                    ),

                "raw_y2":
                    row.get(
                        "raw_y2",
                        "",
                    ),

                "crop_x1":
                    row.get(
                        "crop_x1",
                        "",
                    ),

                "crop_y1":
                    row.get(
                        "crop_y1",
                        "",
                    ),

                "crop_x2":
                    row.get(
                        "crop_x2",
                        "",
                    ),

                "crop_y2":
                    row.get(
                        "crop_y2",
                        "",
                    ),

                "status":
                    "reused_exact_oa_crop",
            }
        )

    if not usable:
        return None

    raw_sessions = set(
        ordered_session_ids(
            raw_records
        )
    )

    usable_sessions = set(
        ordered_session_ids(
            usable
        )
    )

    if not raw_sessions.issubset(
        usable_sessions
    ):
        print(
            "Reusing the OA manifest would remove an entire current session. "
            "ResNet will regenerate crops instead."
        )

        return None

    return (
        usable,
        reuse_manifest,
    )


def load_yolo_detector(
    args: argparse.Namespace,
):
    if str(
        REPO_ROOT
    ) not in sys.path:
        sys.path.insert(
            0,
            str(
                REPO_ROOT
            ),
        )

    from robot_fov_estimation.src.go2_yolo_det_wrapper import (
        YOLOGo2Detector,
    )

    kwargs = {
        "confidence":
            args.yolo_confidence,

        "iou_threshold":
            args.yolo_iou,

        "image_size":
            args.yolo_image_size,

        "device":
            args.yolo_device,
    }

    if (
        args.robot_obj_det_weights_path
        is not None
    ):
        weights_path = Path(
            args.robot_obj_det_weights_path
        ).expanduser().resolve()

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights do not exist: {weights_path}"
            )

        kwargs[
            "weights_path"
        ] = str(
            weights_path
        )

    print(
        "Loading YOLO Go2 detector..."
    )

    return YOLOGo2Detector(
        **kwargs
    )


def clamp_box(
    box_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
) -> Tuple[
    int,
    int,
    int,
    int,
]:
    (
        x1,
        y1,
        x2,
        y2,
    ) = [
        float(
            value
        )
        for value
        in box_xyxy
    ]

    x1 = max(
        0,
        min(
            image_width - 1,
            int(
                round(
                    x1
                )
            ),
        ),
    )

    y1 = max(
        0,
        min(
            image_height - 1,
            int(
                round(
                    y1
                )
            ),
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            image_width,
            int(
                round(
                    x2
                )
            ),
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            image_height,
            int(
                round(
                    y2
                )
            ),
        ),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def expand_box(
    box_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> Tuple[
    int,
    int,
    int,
    int,
]:
    (
        x1,
        y1,
        x2,
        y2,
    ) = [
        float(
            value
        )
        for value
        in box_xyxy
    ]

    width = (
        x2
        - x1
    )

    height = (
        y2
        - y1
    )

    return clamp_box(
        (
            x1
            - width
            * padding_fraction,

            y1
            - height
            * padding_fraction,

            x2
            + width
            * padding_fraction,

            y2
            + height
            * padding_fraction,
        ),
        image_width,
        image_height,
    )


def detect_best_robot(
    detector,
    image_rgb: np.ndarray,
    class_name: str,
) -> Optional[
    Dict[
        str,
        Any,
    ]
]:
    detections = detector.predict(
        image_rgb,
        [
            class_name
        ],
    )

    if not detections:
        return None

    valid = [
        detection
        for detection
        in detections
        if (
            detection.get(
                "box_xyxy"
            )
            is not None
            and detection.get(
                "score"
            )
            is not None
        )
    ]

    if not valid:
        return None

    return max(
        valid,
        key=lambda detection: float(
            detection[
                "score"
            ]
        ),
    )


def make_yolo_crop_path(
    crop_root: Path,
    record: Record,
) -> Path:
    return (
        crop_root
        / safe_name(
            record.session_id
        )
        / (
            f"frame_"
            f"{record.frame_id:020d}.jpg"
        )
    )


def generate_yolo_crops(
    raw_records: Sequence[Record],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[
    List[Record],
    List[Dict[str, Any]],
]:
    detector = load_yolo_detector(
        args
    )

    crop_root = (
        output_dir
        / "yolo_crops"
    )

    crop_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    usable: List[Record] = []
    manifest: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    found_by_session: Counter = Counter()

    print()
    print(
        "=" * 88
    )
    print(
        "PRECOMPUTING DEPLOYMENT-MATCHED YOLO CROPS"
    )
    print(
        "=" * 88
    )
    print(
        f"Class name:      {args.yolo_class_name}"
    )
    print(
        f"Confidence:      {args.yolo_confidence:g}"
    )
    print(
        f"IoU threshold:   {args.yolo_iou:g}"
    )
    print(
        f"YOLO image size: {args.yolo_image_size}"
    )
    print(
        f"Crop padding:    {args.crop_padding:g}"
    )
    print(
        f"Frames:          {len(raw_records)}"
    )

    for index, record in enumerate(
        raw_records,
        start=1,
    ):
        source_path = (
            record.image_path
        )

        crop_path = make_yolo_crop_path(
            crop_root,
            record,
        )

        try:
            with Image.open(
                source_path
            ) as opened:
                image = (
                    ImageOps.exif_transpose(
                        opened
                    )
                    .convert(
                        "RGB"
                    )
                )

            image_rgb = np.asarray(
                image,
                dtype=np.uint8,
            )

            detection = detect_best_robot(
                detector,
                image_rgb,
                args.yolo_class_name,
            )

            if detection is None:
                manifest.append(
                    {
                        "session_id":
                            record.session_id,

                        "frame_id":
                            record.frame_id,

                        "source_image_path":
                            str(
                                source_path
                            ),

                        "crop_image_path":
                            "",

                        "robot_detected":
                            False,

                        "status":
                            "no_detection",
                    }
                )

                if (
                    args.yolo_missing_policy
                    == "error"
                ):
                    raise RuntimeError(
                        "YOLO did not detect Go2 in "
                        f"{record.session_id} frame {record.frame_id}: {source_path}"
                    )

                continue

            raw_box = clamp_box(
                detection[
                    "box_xyxy"
                ],
                image.width,
                image.height,
            )

            crop_box = expand_box(
                raw_box,
                image.width,
                image.height,
                args.crop_padding,
            )

            crop_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            crop = image.crop(
                crop_box
            )

            if (
                crop.width < 2
                or crop.height < 2
            ):
                raise RuntimeError(
                    f"Invalid crop {crop_box} for {source_path}"
                )

            crop.save(
                crop_path,
                format="JPEG",
                quality=
                    args.jpeg_quality,
            )

            found_by_session[
                record.session_id
            ] += 1

            usable.append(
                copy_record_with_image(
                    record,
                    crop_path,
                )
            )

            manifest.append(
                {
                    "session_id":
                        record.session_id,

                    "frame_id":
                        record.frame_id,

                    "source_image_path":
                        str(
                            source_path
                        ),

                    "crop_image_path":
                        str(
                            crop_path.resolve()
                        ),

                    "robot_detected":
                        True,

                    "yolo_score":
                        float(
                            detection[
                                "score"
                            ]
                        ),

                    "raw_x1":
                        raw_box[0],

                    "raw_y1":
                        raw_box[1],

                    "raw_x2":
                        raw_box[2],

                    "raw_y2":
                        raw_box[3],

                    "crop_x1":
                        crop_box[0],

                    "crop_y1":
                        crop_box[1],

                    "crop_x2":
                        crop_box[2],

                    "crop_y2":
                        crop_box[3],

                    "status":
                        "ok",
                }
            )

        except Exception:
            if (
                args.yolo_missing_policy
                == "skip"
            ):
                continue

            raise

        if (
            index % 100
            == 0
            or index
            == len(
                raw_records
            )
        ):
            print(
                f"YOLO: {index}/{len(raw_records)} processed | "
                f"{len(usable)} detected"
            )

    if not usable:
        raise RuntimeError(
            "YOLO produced no usable crops."
        )

    write_rows(
        output_dir
        / "yolo_crop_manifest.csv",

        manifest,
    )

    summary_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for session_id in ordered_session_ids(
        raw_records
    ):
        total = sum(
            1
            for record
            in raw_records
            if record.session_id
            == session_id
        )

        found = int(
            found_by_session[
                session_id
            ]
        )

        summary_rows.append(
            {
                "session_id":
                    session_id,

                "total_frames":
                    total,

                "detected_frames":
                    found,

                "missed_frames":
                    total
                    - found,

                "detection_fraction":
                    (
                        found
                        / total
                        if total
                        else 0.0
                    ),
            }
        )

    write_rows(
        output_dir
        / "yolo_crop_summary_by_session.csv",

        summary_rows,
    )

    return (
        usable,
        manifest,
    )


def prepare_image_records(
    raw_records: Sequence[Record],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[
    List[Record],
    str,
]:
    if args.no_yolo_crop:
        print(
            "WARNING: --no-yolo-crop supplied. "
            "This is a full-image ablation and does NOT match deployment."
        )

        return (
            list(
                raw_records
            ),
            "full_images_no_yolo",
        )

    if not args.no_reuse_oa_yolo_crops:
        manifest_path = Path(
            args.oa_yolo_manifest
        ).expanduser().resolve()

        reused = try_reuse_oa_yolo_crops(
            raw_records,
            manifest_path,
        )

        if reused is not None:
            (
                usable,
                reuse_manifest,
            ) = reused

            print()
            print(
                "=" * 88
            )
            print(
                "REUSING EXACT ORIENT ANYTHING YOLO CROPS"
            )
            print(
                "=" * 88
            )
            print(
                f"Manifest: {manifest_path}"
            )
            print(
                f"Raw samples: {len(raw_records)}"
            )
            print(
                f"Detected/cropped samples reused: {len(usable)}"
            )

            write_rows(
                output_dir
                / "reused_oa_yolo_crop_manifest.csv",

                reuse_manifest,
            )

            return (
                usable,
                "exact_existing_orient_anything_yolo_crops",
            )

    usable, _ = generate_yolo_crops(
        raw_records,
        args,
        output_dir,
    )

    return (
        usable,
        "resnet_generated_yolo_crops",
    )


# =============================================================================
# Session / dataset quality diagnostics
# =============================================================================

def session_quality_rows(
    records: Sequence[Record],
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    grouped = group_sessions(
        records
    )

    rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for session_id in ordered_session_ids(
        records
    ):
        rr = grouped[
            session_id
        ]

        labels = np.asarray(
            [
                record.target_signed_deg
                for record
                in rr
            ],
            dtype=np.float64,
        )

        distances = np.asarray(
            [
                record.distance_m
                for record
                in rr
            ],
            dtype=np.float64,
        )

        sync_delta = np.asarray(
            [
                record.measurement_minus_capture_seconds
                for record
                in rr
            ],
            dtype=np.float64,
        )

        consistency = np.asarray(
            [
                record.label_error_deg
                for record
                in rr
            ],
            dtype=np.float64,
        )

        rows.append(
            {
                "session_id":
                    session_id,

                "num_samples":
                    len(
                        rr
                    ),

                "duration_s":
                    float(
                        rr[
                            -1
                        ].capture_timestamp_unix
                        - rr[
                            0
                        ].capture_timestamp_unix
                    ),

                "target_min_deg":
                    float(
                        labels.min()
                    ),

                "target_max_deg":
                    float(
                        labels.max()
                    ),

                "distance_min_m":
                    float(
                        distances.min()
                    ),

                "distance_median_m":
                    float(
                        np.median(
                            distances
                        )
                    ),

                "distance_max_m":
                    float(
                        distances.max()
                    ),

                "measurement_capture_dt_abs_p95_ms":
                    float(
                        1000.0
                        * np.percentile(
                            np.abs(
                                sync_delta
                            ),
                            95,
                        )
                    ),

                "geometry_vs_saved_yaw_max_error_deg":
                    float(
                        consistency.max()
                    ),
            }
        )

    return rows


# =============================================================================
# The original ResNet-18 + attached orientation head
# =============================================================================

class Go2ResNet18OrientationModel(
    nn.Module
):
    """
    Same basic model as the earlier ResNet comparison:

        pretrained ResNet18
            -> 512-D feature
            -> Linear(512, hidden_dim)
            -> ReLU
            -> Dropout
            -> Linear(hidden_dim, 2)
            -> normalized [sin(alpha), cos(alpha)]

    ResNet-18 is end-to-end fine-tuned by default.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.10,
        l2_normalize_features: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = models.resnet18(
            weights=
                weights
        )

        embedding_dim = int(
            self.backbone.fc.in_features
        )

        self.backbone.fc = (
            nn.Identity()
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.dropout = float(
            dropout
        )

        self.l2_normalize_features = bool(
            l2_normalize_features
        )

        self.head = nn.Sequential(
            nn.Linear(
                embedding_dim,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                self.dropout
            ),
            nn.Linear(
                self.hidden_dim,
                2,
            ),
        )


    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(
            images
        )

        if self.l2_normalize_features:
            features = F.normalize(
                features,
                p=2,
                dim=-1,
                eps=1.0e-8,
            )

        output = self.head(
            features
        )

        return F.normalize(
            output,
            p=2,
            dim=-1,
            eps=1.0e-8,
        )


def orientation_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = F.normalize(
        prediction,
        p=2,
        dim=-1,
        eps=1.0e-8,
    )

    target = F.normalize(
        target,
        p=2,
        dim=-1,
        eps=1.0e-8,
    )

    return (
        1.0
        - (
            prediction
            * target
        ).sum(
            dim=-1
        )
    ).mean()


# =============================================================================
# Dataset / loaders
# =============================================================================

class OrientationImageDataset(
    Dataset
):
    def __init__(
        self,
        records: Sequence[Record],
        transform,
    ) -> None:
        self.records = list(
            records
        )

        self.transform = (
            transform
        )


    def __len__(
        self,
    ) -> int:
        return len(
            self.records
        )


    def __getitem__(
        self,
        index: int,
    ):
        record = self.records[
            index
        ]

        with Image.open(
            record.image_path
        ) as image:
            image = (
                ImageOps.exif_transpose(
                    image
                )
                .convert(
                    "RGB"
                )
            )

            tensor = self.transform(
                image
            )

        target_sin, target_cos = (
            angle_to_sincos(
                record.target_signed_deg
            )
        )

        target = torch.tensor(
            [
                target_sin,
                target_cos,
            ],
            dtype=torch.float32,
        )

        return (
            tensor,
            target,
            torch.tensor(
                index,
                dtype=torch.long,
            ),
        )


def make_session_balanced_sampler(
    records: Sequence[Record],
) -> WeightedRandomSampler:
    counts = Counter(
        record.session_id
        for record
        in records
    )

    weights = [
        1.0
        / float(
            counts[
                record.session_id
            ]
        )
        for record
        in records
    ]

    return WeightedRandomSampler(
        weights=
            torch.as_tensor(
                weights,
                dtype=torch.double,
            ),

        num_samples=
            len(
                records
            ),

        replacement=
            True,
    )


def make_loader(
    records: Sequence[Record],
    transform,
    batch_size: int,
    num_workers: int,
    training: bool,
    session_balanced_sampling: bool,
) -> DataLoader:
    dataset = OrientationImageDataset(
        records,
        transform,
    )

    sampler = None

    if (
        training
        and session_balanced_sampling
    ):
        sampler = (
            make_session_balanced_sampler(
                records
            )
        )

    return DataLoader(
        dataset,
        batch_size=
            batch_size,

        shuffle=(
            training
            and sampler
            is None
        ),

        sampler=
            sampler,

        drop_last=
            False,

        num_workers=
            num_workers,

        pin_memory=
            torch.cuda.is_available(),

        persistent_workers=(
            num_workers
            > 0
        ),
    )


# =============================================================================
# Metrics / prediction output
# =============================================================================

def metrics_from_errors(
    errors_deg: np.ndarray,
) -> Dict[
    str,
    float,
]:
    errors = np.asarray(
        errors_deg,
        dtype=np.float64,
    )

    if len(
        errors
    ) == 0:
        raise ValueError(
            "Cannot score empty errors."
        )

    return {
        "n":
            int(
                len(
                    errors
                )
            ),

        "mean_error_deg":
            float(
                np.mean(
                    errors
                )
            ),

        "median_error_deg":
            float(
                np.median(
                    errors
                )
            ),

        "p90_error_deg":
            float(
                np.percentile(
                    errors,
                    90,
                )
            ),

        "p95_error_deg":
            float(
                np.percentile(
                    errors,
                    95,
                )
            ),

        "max_error_deg":
            float(
                np.max(
                    errors
                )
            ),

        "within_5_deg":
            float(
                np.mean(
                    errors
                    <= 5.0
                )
            ),

        "within_10_deg":
            float(
                np.mean(
                    errors
                    <= 10.0
                )
            ),

        "within_15_deg":
            float(
                np.mean(
                    errors
                    <= 15.0
                )
            ),

        "within_30_deg":
            float(
                np.mean(
                    errors
                    <= 30.0
                )
            ),
    }


def print_metrics(
    title: str,
    metrics: Dict[
        str,
        float,
    ],
) -> None:
    print()
    print(
        title
    )
    print(
        "-" * len(
            title
        )
    )

    print(
        f"N={metrics['n']} "
        f"mean={metrics['mean_error_deg']:.2f} "
        f"med={metrics['median_error_deg']:.2f} "
        f"P90={metrics['p90_error_deg']:.2f} "
        f"P95={metrics['p95_error_deg']:.2f} deg"
    )

    print(
        f"<=5 {100.0 * metrics['within_5_deg']:.1f}% | "
        f"<=10 {100.0 * metrics['within_10_deg']:.1f}% | "
        f"<=15 {100.0 * metrics['within_15_deg']:.1f}% | "
        f"<=30 {100.0 * metrics['within_30_deg']:.1f}%"
    )


@torch.inference_mode()
def evaluate_model(
    model: Go2ResNet18OrientationModel,
    records: Sequence[Record],
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[
    str,
    Any,
]:
    model.eval()

    loader = make_loader(
        records=
            records,

        transform=
            transform,

        batch_size=
            batch_size,

        num_workers=
            num_workers,

        training=
            False,

        session_balanced_sampling=
            False,
    )

    predictions: List[
        np.ndarray
    ] = []

    targets: List[
        np.ndarray
    ] = []

    indices: List[
        np.ndarray
    ] = []

    losses: List[
        float
    ] = []

    sample_counts: List[
        int
    ] = []

    for (
        images,
        target_sincos,
        batch_indices,
    ) in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        target_sincos = (
            target_sincos.to(
                device,
                non_blocking=True,
            )
        )

        prediction = model(
            images
        )

        loss = orientation_cosine_loss(
            prediction,
            target_sincos,
        )

        predictions.append(
            prediction
            .detach()
            .cpu()
            .numpy()
        )

        targets.append(
            target_sincos
            .detach()
            .cpu()
            .numpy()
        )

        indices.append(
            batch_indices
            .cpu()
            .numpy()
        )

        losses.append(
            float(
                loss.item()
            )
        )

        sample_counts.append(
            int(
                images.shape[
                    0
                ]
            )
        )

    pred_sincos = np.concatenate(
        predictions,
        axis=0,
    )

    pred_deg = sincos_to_degrees_np(
        pred_sincos
    )

    index_values = np.concatenate(
        indices,
        axis=0,
    ).astype(
        np.int64
    )

    true_deg = np.asarray(
        [
            records[
                int(
                    index
                )
            ].target_signed_deg
            for index
            in index_values
        ],
        dtype=np.float64,
    )

    errors = circular_error_deg(
        pred_deg,
        true_deg,
    )

    rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for output_index, dataset_index in enumerate(
        index_values.tolist()
    ):
        record = records[
            int(
                dataset_index
            )
        ]

        rows.append(
            {
                "session_id":
                    record.session_id,

                "session_label":
                    record.session_label,

                "frame_id":
                    record.frame_id,

                "capture_timestamp_unix":
                    record.capture_timestamp_unix,

                "image_path":
                    str(
                        record.image_path
                    ),

                "target_signed_yaw_deg":
                    record.target_signed_deg,

                "prediction_signed_yaw_deg":
                    float(
                        pred_deg[
                            output_index
                        ]
                    ),

                "error_deg":
                    float(
                        errors[
                            output_index
                        ]
                    ),

                "predicted_sin":
                    float(
                        pred_sincos[
                            output_index,
                            0,
                        ]
                    ),

                "predicted_cos":
                    float(
                        pred_sincos[
                            output_index,
                            1,
                        ]
                    ),

                "distance_m":
                    record.distance_m,

                "horizontal_distance_m":
                    record.horizontal_distance_m,

                "measurement_minus_capture_seconds":
                    record.measurement_minus_capture_seconds,
            }
        )

    weighted_loss = (
        sum(
            loss
            * count
            for loss, count
            in zip(
                losses,
                sample_counts,
            )
        )
        / max(
            1,
            sum(
                sample_counts
            ),
        )
    )

    return {
        "loss":
            float(
                weighted_loss
            ),

        "metrics":
            metrics_from_errors(
                errors
            ),

        "rows":
            rows,

        "predictions_deg":
            pred_deg,

        "errors_deg":
            errors,
    }


# =============================================================================
# Circular constant baseline
# =============================================================================

def best_circular_constant(
    train_records: Sequence[Record],
) -> float:
    target = np.asarray(
        [
            record.target_signed_deg
            for record
            in train_records
        ],
        dtype=np.float64,
    )

    best_value = 0.0
    best_score = math.inf

    for candidate in range(
        -180,
        180,
    ):
        errors = circular_error_deg(
            np.full(
                len(
                    target
                ),
                float(
                    candidate
                ),
                dtype=np.float64,
            ),
            target,
        )

        score = float(
            np.mean(
                errors
            )
        )

        if score < best_score:
            best_score = score
            best_value = float(
                candidate
            )

    return best_value


def constant_metrics(
    constant_deg: float,
    records: Sequence[Record],
) -> Dict[
    str,
    float,
]:
    target = np.asarray(
        [
            record.target_signed_deg
            for record
            in records
        ],
        dtype=np.float64,
    )

    prediction = np.full(
        len(
            target
        ),
        constant_deg,
        dtype=np.float64,
    )

    return metrics_from_errors(
        circular_error_deg(
            prediction,
            target,
        )
    )


# =============================================================================
# Training
# =============================================================================

def freeze_batchnorm_running_stats(
    module: nn.Module,
) -> None:
    for child in module.modules():
        if isinstance(
            child,
            torch.nn.modules.batchnorm._BatchNorm,
        ):
            child.eval()


def make_grad_scaler(
    enabled: bool,
):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=
                enabled,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=
                enabled,
        )


def train_one_epoch(
    model: Go2ResNet18OrientationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip_norm: float,
    freeze_batchnorm_stats: bool,
) -> float:
    model.train()

    if freeze_batchnorm_stats:
        freeze_batchnorm_running_stats(
            model
        )

    total_loss = 0.0
    total_count = 0

    for (
        images,
        target_sincos,
        _,
    ) in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        target_sincos = (
            target_sincos.to(
                device,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=
                device.type,

            dtype=
                torch.float16,

            enabled=(
                use_amp
                and device.type
                == "cuda"
            ),
        ):
            prediction = model(
                images
            )

            loss = orientation_cosine_loss(
                prediction,
                target_sincos,
            )

        scaler.scale(
            loss
        ).backward()

        if (
            gradient_clip_norm
            > 0.0
        ):
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=
                    gradient_clip_norm,
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = int(
            images.shape[
                0
            ]
        )

        total_loss += (
            float(
                loss.item()
            )
            * batch_size
        )

        total_count += (
            batch_size
        )

    return (
        total_loss
        / max(
            1,
            total_count,
        )
    )


def train_fold_model(
    train_records: Sequence[Record],
    validation_records: Sequence[Record],
    transform,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Tuple[
    Go2ResNet18OrientationModel,
    List[Dict[str, Any]],
    int,
    float,
]:
    model = Go2ResNet18OrientationModel(
        hidden_dim=
            args.hidden_dim,

        dropout=
            args.dropout,

        l2_normalize_features=
            args.l2_normalize_embeddings,

        pretrained=
            True,
    ).to(
        device
    )

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(
            not args.freeze_backbone
        )

    train_loader = make_loader(
        records=
            train_records,

        transform=
            transform,

        batch_size=
            args.batch_size,

        num_workers=
            args.num_workers,

        training=
            True,

        session_balanced_sampling=
            args.session_balanced_sampling,
    )

    parameter_groups: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    trainable_backbone = [
        parameter
        for parameter
        in model.backbone.parameters()
        if parameter.requires_grad
    ]

    if trainable_backbone:
        parameter_groups.append(
            {
                "params":
                    trainable_backbone,

                "lr":
                    args.backbone_learning_rate,

                "name":
                    "backbone",
            }
        )

    parameter_groups.append(
        {
            "params":
                model.head.parameters(),

            "lr":
                args.learning_rate,

            "name":
                "head",
        }
    )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=
            args.weight_decay,
    )

    scaler = make_grad_scaler(
        enabled=(
            args.use_amp
            and device.type
            == "cuda"
        )
    )

    history: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    best_state = None
    best_epoch = 0
    best_selection_value = (
        math.inf
    )

    epochs_without_improvement = (
        0
    )

    print()
    print(
        "Training end-to-end ResNet-18 + orientation head..."
    )
    print(
        f"Backbone LR={args.backbone_learning_rate:g} "
        f"head LR={args.learning_rate:g} "
        f"selection={args.selection_metric} "
        f"session-balanced={args.session_balanced_sampling}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = train_one_epoch(
            model=
                model,

            loader=
                train_loader,

            optimizer=
                optimizer,

            scaler=
                scaler,

            device=
                device,

            use_amp=
                args.use_amp,

            gradient_clip_norm=
                args.gradient_clip_norm,

            freeze_batchnorm_stats=
                args.freeze_batchnorm_stats,
        )

        train_eval = evaluate_model(
            model,
            train_records,
            transform,
            device,
            args.batch_size,
            args.num_workers,
        )

        validation_eval = evaluate_model(
            model,
            validation_records,
            transform,
            device,
            args.batch_size,
            args.num_workers,
        )

        train_metrics = (
            train_eval[
                "metrics"
            ]
        )

        validation_metrics = (
            validation_eval[
                "metrics"
            ]
        )

        selection_value = (
            validation_metrics[
                "mean_error_deg"
            ]
            if args.selection_metric
            == "mean"
            else validation_metrics[
                "median_error_deg"
            ]
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_mean_error_deg":
                    train_metrics[
                        "mean_error_deg"
                    ],

                "train_median_error_deg":
                    train_metrics[
                        "median_error_deg"
                    ],

                "validation_loss":
                    validation_eval[
                        "loss"
                    ],

                "validation_mean_error_deg":
                    validation_metrics[
                        "mean_error_deg"
                    ],

                "validation_median_error_deg":
                    validation_metrics[
                        "median_error_deg"
                    ],

                "validation_p90_error_deg":
                    validation_metrics[
                        "p90_error_deg"
                    ],

                "selection_metric":
                    args.selection_metric,

                "selection_value_deg":
                    selection_value,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss {train_loss:.5f} | "
            f"train {train_metrics['mean_error_deg']:.2f}/"
            f"{train_metrics['median_error_deg']:.2f} | "
            f"val {validation_metrics['mean_error_deg']:.2f}/"
            f"{validation_metrics['median_error_deg']:.2f} | "
            f"P90 {validation_metrics['p90_error_deg']:.2f}"
        )

        if (
            selection_value
            < best_selection_value
            - args.minimum_improvement_deg
        ):
            best_selection_value = (
                selection_value
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            epochs_without_improvement = (
                0
            )

        else:
            epochs_without_improvement += (
                1
            )

        if (
            epoch
            >= args.min_epochs
            and epochs_without_improvement
            >= args.patience
        ):
            print(
                f"Early stopping at {epoch}; "
                f"best epoch={best_epoch}"
            )

            break

    if best_state is None:
        raise RuntimeError(
            "No best checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    checkpoint_path = (
        output_dir
        / "best_go2_resnet18_forward_estimation.pt"
    )

    torch.save(
        {
            "schema_version":
                1,

            "model_state_dict":
                best_state,

            "architecture":
                (
                    "ImageNet pretrained ResNet18 backbone + "
                    f"Linear(512,{args.hidden_dim}) + ReLU + "
                    f"Dropout({args.dropout}) + Linear({args.hidden_dim},2)"
                ),

            "output_encoding":
                "[sin(alpha), cos(alpha)] normalized to unit length",

            "prediction_decode":
                "alpha_deg = atan2(predicted_sin, predicted_cos)",

            "target_definition":
                (
                    "alpha = Unity SignedAngle("
                    "camera_to_robot_planar, "
                    "robot_forward_planar, +Y)"
                ),

            "embedding_dim":
                512,

            "hidden_dim":
                args.hidden_dim,

            "dropout":
                args.dropout,

            "l2_normalize_features":
                args.l2_normalize_embeddings,

            "backbone_fine_tuned":
                not args.freeze_backbone,

            "pretrained_backbone":
                "torchvision ResNet18_Weights.DEFAULT",

            "preprocessing":
                "ResNet18_Weights.DEFAULT.transforms()",

            "loss":
                "mean(1 - dot(pred_unit_sincos, target_unit_sincos))",

            "backbone_learning_rate":
                args.backbone_learning_rate,

            "head_learning_rate":
                args.learning_rate,

            "weight_decay":
                args.weight_decay,

            "best_epoch":
                best_epoch,

            "selection_metric":
                args.selection_metric,

            "best_selection_value_deg":
                best_selection_value,
        },
        checkpoint_path,
    )

    write_rows(
        output_dir
        / "training_history.csv",

        history,
    )

    return (
        model,
        history,
        best_epoch,
        best_selection_value,
    )


# =============================================================================
# Diagnostics / manifests
# =============================================================================

def split_manifest_rows(
    split: Split,
    train_used: Sequence[Record],
    train_thinned_out: Sequence[Record],
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for split_name, records in (
        (
            "train_used",
            train_used,
        ),
        (
            "train_thinned_out",
            train_thinned_out,
        ),
        (
            "validation",
            split.validation,
        ),
        (
            "test",
            split.test,
        ),
        (
            "purged",
            split.purged,
        ),
    ):
        for record in records:
            rows.append(
                {
                    "split":
                        split_name,

                    "session_id":
                        record.session_id,

                    "frame_id":
                        record.frame_id,

                    "capture_timestamp_unix":
                        record.capture_timestamp_unix,

                    "image_path":
                        str(
                            record.image_path
                        ),

                    "target_signed_yaw_deg":
                        record.target_signed_deg,

                    "distance_m":
                        record.distance_m,
                }
            )

    return rows


def parse_distance_edges(
    raw: str,
) -> List[
    float
]:
    values: List[
        float
    ] = []

    for token in str(
        raw
    ).split(
        ","
    ):
        token = token.strip()

        if not token:
            continue

        if token.lower() in {
            "inf",
            "+inf",
            "infinity",
        }:
            values.append(
                math.inf
            )
        else:
            values.append(
                float(
                    token
                )
            )

    if len(
        values
    ) < 2:
        raise ValueError(
            "At least two distance edges are required."
        )

    if any(
        right <= left
        for left, right
        in zip(
            values[:-1],
            values[1:],
        )
    ):
        raise ValueError(
            "Distance edges must be strictly increasing."
        )

    return values


def error_by_distance_rows(
    prediction_rows: Sequence[
        Dict[
            str,
            Any,
        ]
    ],
    split_name: str,
    edges: Sequence[float],
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for low, high in zip(
        edges[:-1],
        edges[1:],
    ):
        if math.isinf(
            high
        ):
            selected = [
                row
                for row
                in prediction_rows
                if float(
                    row[
                        "distance_m"
                    ]
                )
                >= low
            ]

            label = (
                f"[{low:g},inf)"
            )

        else:
            selected = [
                row
                for row
                in prediction_rows
                if (
                    low
                    <= float(
                        row[
                            "distance_m"
                        ]
                    )
                    < high
                )
            ]

            label = (
                f"[{low:g},{high:g})"
            )

        if selected:
            metrics = metrics_from_errors(
                np.asarray(
                    [
                        float(
                            row[
                                "error_deg"
                            ]
                        )
                        for row
                        in selected
                    ],
                    dtype=np.float64,
                )
            )

        else:
            metrics = {
                "n":
                    0,

                "mean_error_deg":
                    math.nan,

                "median_error_deg":
                    math.nan,

                "p90_error_deg":
                    math.nan,

                "p95_error_deg":
                    math.nan,

                "max_error_deg":
                    math.nan,

                "within_5_deg":
                    math.nan,

                "within_10_deg":
                    math.nan,

                "within_15_deg":
                    math.nan,

                "within_30_deg":
                    math.nan,
            }

        rows.append(
            {
                "split":
                    split_name,

                "distance_bin":
                    label,

                **metrics,
            }
        )

    return rows


# =============================================================================
# Run one fold
# =============================================================================

def run_fold(
    split: Split,
    args: argparse.Namespace,
    device: torch.device,
    transform,
    output_dir: Path,
    crop_source: str,
) -> Dict[
    str,
    Any,
]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 90
    )
    print(
        f"FOLD {split.name}"
    )
    print(
        "=" * 90
    )

    print(
        f"Raw train={len(split.train)} "
        f"val={len(split.validation)} "
        f"test={len(split.test)}"
    )

    print(
        f"Train sessions: "
        f"{split.info.get('train_session_ids', [])}"
    )

    print(
        f"Validation: "
        f"{split.info.get('validation_session_id', 'n/a')}"
    )

    print(
        f"Test: "
        f"{split.info.get('test_session_id', 'n/a')}"
    )

    (
        train_records,
        thinned_out,
    ) = thin_training(
        split.train,
        args.temporal_thin_seconds,
    )

    if not train_records:
        raise RuntimeError(
            "Temporal thinning removed every training sample."
        )

    print(
        f"Temporal thinning {args.temporal_thin_seconds:g}s: "
        f"kept {len(train_records)}, removed {len(thinned_out)}"
    )

    write_rows(
        output_dir
        / "split_manifest.csv",

        split_manifest_rows(
            split,
            train_records,
            thinned_out,
        ),
    )

    constant_prediction = (
        best_circular_constant(
            train_records
        )
    )

    constant_train = constant_metrics(
        constant_prediction,
        train_records,
    )

    constant_validation = constant_metrics(
        constant_prediction,
        split.validation,
    )

    constant_test = constant_metrics(
        constant_prediction,
        split.test,
    )

    print(
        f"Train-derived circular constant={constant_prediction:.2f} deg | "
        f"val MAE={constant_validation['mean_error_deg']:.2f} | "
        f"test MAE={constant_test['mean_error_deg']:.2f}"
    )

    (
        model,
        history,
        best_epoch,
        best_selection_value,
    ) = train_fold_model(
        train_records=
            train_records,

        validation_records=
            split.validation,

        transform=
            transform,

        args=
            args,

        device=
            device,

        output_dir=
            output_dir,
    )

    final_train = evaluate_model(
        model,
        train_records,
        transform,
        device,
        args.batch_size,
        args.num_workers,
    )

    final_validation = evaluate_model(
        model,
        split.validation,
        transform,
        device,
        args.batch_size,
        args.num_workers,
    )

    final_test = evaluate_model(
        model,
        split.test,
        transform,
        device,
        args.batch_size,
        args.num_workers,
    )

    print()
    print(
        f"BEST EPOCH {best_epoch}"
    )

    print_metrics(
        "Final train",
        final_train[
            "metrics"
        ],
    )

    print_metrics(
        "Final validation",
        final_validation[
            "metrics"
        ],
    )

    print_metrics(
        "Final test",
        final_test[
            "metrics"
        ],
    )

    write_rows(
        output_dir
        / "train_predictions.csv",

        final_train[
            "rows"
        ],
    )

    write_rows(
        output_dir
        / "validation_predictions.csv",

        final_validation[
            "rows"
        ],
    )

    write_rows(
        output_dir
        / "test_predictions.csv",

        final_test[
            "rows"
        ],
    )

    distance_edges = parse_distance_edges(
        args.distance_bin_edges
    )

    distance_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for split_name, result in (
        (
            "train",
            final_train,
        ),
        (
            "validation",
            final_validation,
        ),
        (
            "test",
            final_test,
        ),
    ):
        distance_rows.extend(
            error_by_distance_rows(
                result[
                    "rows"
                ],
                split_name,
                distance_edges,
            )
        )

    write_rows(
        output_dir
        / "error_by_distance.csv",

        distance_rows,
    )

    summary = {
        "split":
            split.info,

        "crop_source":
            crop_source,

        "dataset":
            {
                "raw_train_samples":
                    len(
                        split.train
                    ),

                "train_samples_after_temporal_thinning":
                    len(
                        train_records
                    ),

                "training_samples_removed_by_temporal_thinning":
                    len(
                        thinned_out
                    ),

                "validation_samples":
                    len(
                        split.validation
                    ),

                "test_samples":
                    len(
                        split.test
                    ),
            },

        "model":
            {
                "architecture":
                    (
                        "pretrained ResNet18 + "
                        f"MLP(512->{args.hidden_dim}->2)"
                    ),

                "target":
                    "[sin(alpha), cos(alpha)]",

                "loss":
                    "1 - cosine similarity",

                "backbone_fine_tuned":
                    not args.freeze_backbone,

                "backbone_learning_rate":
                    args.backbone_learning_rate,

                "head_learning_rate":
                    args.learning_rate,

                "hidden_dim":
                    args.hidden_dim,

                "dropout":
                    args.dropout,

                "selection_metric":
                    args.selection_metric,

                "best_epoch":
                    best_epoch,

                "best_selection_value_deg":
                    best_selection_value,
            },

        "constant_baseline":
            {
                "prediction_deg":
                    constant_prediction,

                "train":
                    constant_train,

                "validation":
                    constant_validation,

                "test":
                    constant_test,
            },

        "final":
            {
                "train":
                    final_train[
                        "metrics"
                    ],

                "validation":
                    final_validation[
                        "metrics"
                    ],

                "test":
                    final_test[
                        "metrics"
                    ],
            },
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            allow_nan=False,
        )

    return {
        "summary":
            summary,

        "test_rows":
            final_test[
                "rows"
            ],
    }


# =============================================================================
# LOSO aggregation
# =============================================================================

def aggregate_loso(
    output_dir: Path,
    fold_results: Sequence[
        Dict[
            str,
            Any,
        ]
    ],
) -> None:
    fold_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    pooled_predictions: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for result in fold_results:
        summary = result[
            "summary"
        ]

        split = summary[
            "split"
        ]

        test_metrics = summary[
            "final"
        ][
            "test"
        ]

        constant_metrics_test = (
            summary[
                "constant_baseline"
            ][
                "test"
            ]
        )

        fold_rows.append(
            {
                "test_session_id":
                    split[
                        "test_session_id"
                    ],

                "validation_session_id":
                    split[
                        "validation_session_id"
                    ],

                "train_session_ids":
                    "|".join(
                        split[
                            "train_session_ids"
                        ]
                    ),

                "best_epoch":
                    summary[
                        "model"
                    ][
                        "best_epoch"
                    ],

                "test_mean_error_deg":
                    test_metrics[
                        "mean_error_deg"
                    ],

                "test_median_error_deg":
                    test_metrics[
                        "median_error_deg"
                    ],

                "test_p90_error_deg":
                    test_metrics[
                        "p90_error_deg"
                    ],

                "test_p95_error_deg":
                    test_metrics[
                        "p95_error_deg"
                    ],

                "test_within_5_deg":
                    test_metrics[
                        "within_5_deg"
                    ],

                "test_within_10_deg":
                    test_metrics[
                        "within_10_deg"
                    ],

                "test_within_15_deg":
                    test_metrics[
                        "within_15_deg"
                    ],

                "test_within_30_deg":
                    test_metrics[
                        "within_30_deg"
                    ],

                "constant_test_mean_error_deg":
                    constant_metrics_test[
                        "mean_error_deg"
                    ],

                "improvement_vs_constant_deg":
                    (
                        constant_metrics_test[
                            "mean_error_deg"
                        ]
                        - test_metrics[
                            "mean_error_deg"
                        ]
                    ),
            }
        )

        pooled_predictions.extend(
            result[
                "test_rows"
            ]
        )

    write_rows(
        output_dir
        / "leave_one_session_out_summary.csv",

        fold_rows,
    )

    write_rows(
        output_dir
        / "leave_one_session_out_predictions.csv",

        pooled_predictions,
    )

    pooled_errors = np.asarray(
        [
            float(
                row[
                    "error_deg"
                ]
            )
            for row
            in pooled_predictions
        ],
        dtype=np.float64,
    )

    aggregate = {
        "num_folds":
            len(
                fold_rows
            ),

        "macro_mean_test_mae_deg":
            float(
                np.mean(
                    [
                        row[
                            "test_mean_error_deg"
                        ]
                        for row
                        in fold_rows
                    ]
                )
            ),

        "macro_mean_test_median_error_deg":
            float(
                np.mean(
                    [
                        row[
                            "test_median_error_deg"
                        ]
                        for row
                        in fold_rows
                    ]
                )
            ),

        "pooled_test_metrics":
            metrics_from_errors(
                pooled_errors
            ),

        "folds":
            fold_rows,
    }

    with (
        output_dir
        / "leave_one_session_out_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            aggregate,
            f,
            indent=2,
            allow_nan=False,
        )
    print(
        "LEAVE-ONE-SESSION-OUT RESNET-18 SUMMARY"
    )
    print(
        "=" * 90
    )

    for row in fold_rows:
        print(
            f"{row['test_session_id']}: "
            f"mean={row['test_mean_error_deg']:.2f} "
            f"med={row['test_median_error_deg']:.2f} "
            f"P90={row['test_p90_error_deg']:.2f}"
        )

    print(
        f"Macro mean test MAE: "
        f"{aggregate['macro_mean_test_mae_deg']:.2f} deg"
    )

    pooled = aggregate[
        "pooled_test_metrics"
    ]

    print(
        f"Pooled test: "
        f"mean={pooled['mean_error_deg']:.2f} "
        f"med={pooled['median_error_deg']:.2f} "
        f"P90={pooled['p90_error_deg']:.2f}"
    )


def train_final_model_on_all_data(
    records: Sequence[Record],
    args: argparse.Namespace,
    device: torch.device,
    transform,
    output_dir: Path,
) -> Dict[
    str,
    Any,
]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(
        args.seed,
    )

    sessions = ordered_session_ids(
        records,
    )

    print()
    print(
        "=" * 90
    )
    print(
        "FINAL TRAINING ON ALL AVAILABLE DATA"
    )
    print(
        "=" * 90
    )
    print(
        f"Train samples={len(records)} sessions={len(sessions)}"
    )

    model = Go2ResNet18OrientationModel(
        hidden_dim=
            args.hidden_dim,

        dropout=
            args.dropout,

        l2_normalize_features=
            args.l2_normalize_embeddings,

        pretrained=
            True,
    ).to(
        device
    )

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(
            not args.freeze_backbone
        )

    train_loader = make_loader(
        records=
            records,

        transform=
            transform,

        batch_size=
            args.batch_size,

        num_workers=
            args.num_workers,

        training=
            True,

        session_balanced_sampling=
            args.session_balanced_sampling,
    )

    eval_loader = make_loader(
        records=
            records,

        transform=
            transform,

        batch_size=
            args.batch_size,

        num_workers=
            args.num_workers,

        training=
            False,

        session_balanced_sampling=
            False,
    )

    parameter_groups: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    trainable_backbone = [
        parameter
        for parameter
        in model.backbone.parameters()
        if parameter.requires_grad
    ]

    if trainable_backbone:
        parameter_groups.append(
            {
                "params":
                    trainable_backbone,

                "lr":
                    args.backbone_learning_rate,

                "name":
                    "backbone",
            }
        )

    parameter_groups.append(
        {
            "params":
                model.head.parameters(),

            "lr":
                args.learning_rate,

            "name":
                "head",
        }
    )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=
            args.weight_decay,
    )

    scaler = make_grad_scaler(
        enabled=(
            args.use_amp
            and device.type
            == "cuda"
        )
    )

    print(
        "Training final all-data ResNet-18 + orientation head..."
    )
    print(
        f"Backbone LR={args.backbone_learning_rate:g} "
        f"head LR={args.learning_rate:g} "
        f"session-balanced={args.session_balanced_sampling}"
    )

    history: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    best_state = None
    best_epoch = 0
    best_train_mean_error_deg = math.inf

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = train_one_epoch(
            model=
                model,

            loader=
                train_loader,

            optimizer=
                optimizer,

            scaler=
                scaler,

            device=
                device,

            use_amp=
                args.use_amp,

            gradient_clip_norm=
                args.gradient_clip_norm,

            freeze_batchnorm_stats=
                args.freeze_batchnorm_stats,
        )

        train_eval = evaluate_model(
            model,
            records,
            transform,
            device,
            args.batch_size,
            args.num_workers,
        )

        train_metrics = train_eval[
            "metrics"
        ]

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_mean_error_deg":
                    train_metrics[
                        "mean_error_deg"
                    ],

                "train_median_error_deg":
                    train_metrics[
                        "median_error_deg"
                    ],

                "train_p90_error_deg":
                    train_metrics[
                        "p90_error_deg"
                    ],
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss {train_loss:.5f} | "
            f"train {train_metrics['mean_error_deg']:.2f}/"
            f"{train_metrics['median_error_deg']:.2f} | "
            f"P90 {train_metrics['p90_error_deg']:.2f}"
        )

        if train_metrics["mean_error_deg"] < best_train_mean_error_deg:
            best_train_mean_error_deg = train_metrics["mean_error_deg"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(
            "No best checkpoint was produced."
        )

    model.load_state_dict(best_state)

    final_eval = evaluate_model(
        model,
        records,
        transform,
        device,
        args.batch_size,
        args.num_workers,
    )

    checkpoint_path = (
        output_dir
        / "best_go2_resnet18_forward_estimation.pt"
    )

    torch.save(
        {
            "schema_version":
                1,

            "model_state_dict":
                model.state_dict(),

            "architecture":
                (
                    "ImageNet pretrained ResNet18 backbone + "
                    f"Linear(512,{args.hidden_dim}) + ReLU + "
                    f"Dropout({args.dropout}) + Linear({args.hidden_dim},2)"
                ),

            "output_encoding":
                "[sin(alpha), cos(alpha)] normalized to unit length",

            "prediction_decode":
                "alpha_deg = atan2(predicted_sin, predicted_cos)",

            "target_definition":
                (
                    "alpha = Unity SignedAngle("
                    "camera_to_robot_planar, "
                    "robot_forward_planar, +Y)"
                ),

            "embedding_dim":
                512,

            "hidden_dim":
                args.hidden_dim,

            "dropout":
                args.dropout,

            "l2_normalize_features":
                args.l2_normalize_embeddings,

            "backbone_fine_tuned":
                not args.freeze_backbone,

            "pretrained_backbone":
                "torchvision ResNet18_Weights.DEFAULT",

            "preprocessing":
                "ResNet18_Weights.DEFAULT.transforms()",

            "loss":
                "mean(1 - dot(pred_unit_sincos, target_unit_sincos))",

            "backbone_learning_rate":
                args.backbone_learning_rate,

            "head_learning_rate":
                args.learning_rate,

            "weight_decay":
                args.weight_decay,

            "target_mode":
                "all_available_data",

            "best_epoch":
                best_epoch,

            "best_train_mean_error_deg":
                best_train_mean_error_deg,

            "train_session_ids":
                sessions,

            "training_args":
                vars(args),
        },
        checkpoint_path,
    )

    write_rows(
        output_dir
        / "training_history.csv",

        history,
    )

    write_rows(
        output_dir
        / "train_predictions.csv",

        final_eval[
            "rows"
        ],
    )

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "split":
                    {
                        "strategy":
                            "all available data",

                        "train_session_ids":
                            sessions,
                    },

                "counts":
                    {
                        "train":
                            len(
                                records
                            ),
                    },

                "final_train":
                    final_eval[
                        "metrics"
                    ],

                "best_epoch":
                    best_epoch,

                "best_train_mean_error_deg":
                    best_train_mean_error_deg,

                "checkpoint_path":
                    str(
                        checkpoint_path
                    ),
            },
            f,
            indent=2,
            allow_nan=False,
        )

    print_metrics(
        "Final train",
        final_eval[
            "metrics"
        ],
    )

    return {
        "final_metrics":
            final_eval[
                "metrics"
            ],

        "checkpoint_path":
            checkpoint_path,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_sessions(
    raw: Optional[str],
) -> Optional[
    List[str]
]:
    if raw is None:
        return None

    values = [
        token.strip()
        for token
        in raw.split(
            ","
        )
        if token.strip()
    ]

    return (
        values
        if values
        else None
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the earlier ResNet18+MLP Go2 orientation model on "
            "the new Quest-native forward-estimation dataset."
        )
    )

    # Data.
    parser.add_argument(
        "--data-root",
        default=
            str(
                DEFAULT_DATA_ROOT
            ),
    )

    parser.add_argument(
        "--output-dir",
        default=
            str(
                DEFAULT_OUTPUT_DIR
            ),
    )

    parser.add_argument(
        "--sessions",
        default=None,
        help=(
            "Optional comma-separated exact session IDs."
        ),
    )

    parser.add_argument(
        "--label-consistency-tolerance-deg",
        type=float,
        default=0.25,
    )

    # Splits: same defaults as the new OA trainer.
    parser.add_argument(
        "--split-mode",
        choices=(
            "leave_one_session_out",
            "session_holdout",
            "temporal",
            "random",
        ),
        default=
            "leave_one_session_out",
    )

    parser.add_argument(
        "--only-test-session",
        default=None,
    )

    parser.add_argument(
        "--validation-session",
        default=None,
    )

    parser.add_argument(
        "--test-session",
        default=None,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--split-gap-seconds",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--temporal-thin-seconds",
        type=float,
        default=0.5,
        help=(
            "Minimum spacing between TRAINING frames within each session. "
            "Matches the current OA comparison default."
        ),
    )

    parser.add_argument(
        "--no-session-balanced-sampling",
        action="store_true",
    )

    # YOLO. Default is exact OA crop reuse if possible.
    parser.add_argument(
        "--no-yolo-crop",
        action="store_true",
        help=(
            "Full-image ablation only. Default behavior uses YOLO crops."
        ),
    )

    parser.add_argument(
        "--no-reuse-oa-yolo-crops",
        action="store_true",
        help=(
            "Do not reuse the existing Orient Anything YOLO manifest/crops. "
            "Instead run YOLO again and cache crops under this ResNet output directory."
        ),
    )

    parser.add_argument(
        "--oa-yolo-manifest",
        default=
            str(
                DEFAULT_OA_YOLO_MANIFEST
            ),
        help=(
            "Existing OA yolo_crop_manifest.csv used to guarantee identical "
            "input crops for the comparison when it fully covers the current dataset."
        ),
    )

    parser.add_argument(
        "--robot-obj-det-weights-path",
        default=None,
        help=(
            "Optional explicit Go2 YOLO checkpoint. If omitted, "
            "YOLOGo2Detector uses its normal project default."
        ),
    )

    parser.add_argument(
        "--yolo-class-name",
        default="robot dog",
    )

    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--yolo-image-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--yolo-device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
    )

    parser.add_argument(
        "--yolo-missing-policy",
        choices=(
            "skip",
            "error",
        ),
        default="skip",
    )

    # Original ResNet model configuration.
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--l2-normalize-embeddings",
        action="store_true",
        help=(
            "L2-normalize the 512-D ResNet feature before the MLP head. "
            "Off by default, matching the earlier script."
        ),
    )

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Diagnostic only. Default is end-to-end ResNet-18 fine-tuning."
        ),
    )

    # Preserve architecture-appropriate optimization defaults from the old script.
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
        help="Learning rate for the newly initialized orientation head.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--min-epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--minimum-improvement-deg",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--selection-metric",
        choices=(
            "mean",
            "median",
        ),
        default="mean",
        help=(
            "Validation metric used for best-checkpoint selection. "
            "Default mean matches the new OA trainer. "
            "Use median to reproduce the old ResNet script's selection rule."
        ),
    )

    parser.add_argument(
        "--freeze-batchnorm-stats",
        action="store_true",
        help=(
            "Keep BatchNorm running statistics fixed during fine-tuning. "
            "Off by default to retain the earlier ResNet training behavior."
        ),
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--distance-bin-edges",
        default="0,2,4,6,8,inf",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    args = parser.parse_args()

    args.session_balanced_sampling = (
        not args.no_session_balanced_sampling
    )

    args.use_amp = (
        not args.no_amp
    )

    return args


def validate_args(
    args: argparse.Namespace,
) -> None:
    if (
        args.label_consistency_tolerance_deg
        < 0.0
    ):
        raise ValueError(
            "--label-consistency-tolerance-deg must be >=0."
        )

    if (
        args.temporal_thin_seconds
        < 0.0
    ):
        raise ValueError(
            "--temporal-thin-seconds must be >=0."
        )

    if (
        args.crop_padding
        < 0.0
    ):
        raise ValueError(
            "--crop-padding must be >=0."
        )

    if not (
        0.0
        <= args.yolo_confidence
        <= 1.0
    ):
        raise ValueError(
            "--yolo-confidence must be in [0,1]."
        )

    if not (
        0.0
        <= args.yolo_iou
        <= 1.0
    ):
        raise ValueError(
            "--yolo-iou must be in [0,1]."
        )

    if (
        args.yolo_image_size
        < 1
    ):
        raise ValueError(
            "--yolo-image-size must be >=1."
        )

    if not (
        1
        <= args.jpeg_quality
        <= 100
    ):
        raise ValueError(
            "--jpeg-quality must be in [1,100]."
        )

    if (
        args.hidden_dim
        < 1
    ):
        raise ValueError(
            "--hidden-dim must be >=1."
        )

    if not (
        0.0
        <= args.dropout
        < 1.0
    ):
        raise ValueError(
            "--dropout must be in [0,1)."
        )

    if (
        args.backbone_learning_rate
        <= 0.0
        or args.learning_rate
        <= 0.0
    ):
        raise ValueError(
            "Learning rates must be >0."
        )

    if (
        args.batch_size
        < 1
        or args.epochs
        < 1
        or args.min_epochs
        < 1
        or args.min_epochs
        > args.epochs
        or args.patience
        < 1
    ):
        raise ValueError(
            "Invalid batch/epoch/patience configuration."
        )

    if (
        args.num_workers
        < 0
    ):
        raise ValueError(
            "--num-workers must be >=0."
        )

    if (
        args.split_mode
        in {
            "temporal",
            "random",
        }
    ):
        if not (
            0.0
            < args.train_fraction
            < 1.0
            and 0.0
            < args.validation_fraction
            < 1.0
            and args.train_fraction
            + args.validation_fraction
            < 1.0
        ):
            raise ValueError(
                "Invalid train/validation fractions."
            )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    seed_everything(
        args.seed
    )

    data_root = Path(
        args.data_root
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            args.device
        )

    print(
        f"Repo root: {REPO_ROOT}"
    )
    print(
        f"Data root: {data_root}"
    )
    print(
        f"Output:    {output_dir}"
    )
    print(
        f"Device:    {device}"
    )
    print(
        f"Split:     {args.split_mode}"
    )

    raw_records = load_records(
        data_root=
            data_root,

        allowed_sessions=
            parse_sessions(
                args.sessions
            ),

        label_tolerance_deg=
            args.label_consistency_tolerance_deg,
    )

    print(
        f"Raw samples before crop selection: "
        f"{len(raw_records)}"
    )

    (
        records,
        crop_source,
    ) = prepare_image_records(
        raw_records=
            raw_records,

        args=
            args,

        output_dir=
            output_dir,
    )

    sessions = ordered_session_ids(
        records
    )

    grouped = group_sessions(
        records
    )

    print(
        f"Samples used by ResNet: {len(records)}"
    )
    print(
        f"Crop source: {crop_source}"
    )
    print(
        f"Sessions: {len(sessions)}"
    )

    for session_id in sessions:
        rr = grouped[
            session_id
        ]

        print(
            f"  {session_id}: "
            f"n={len(rr)} "
            f"duration="
            f"{rr[-1].capture_timestamp_unix - rr[0].capture_timestamp_unix:.1f}s "
            f"distance="
            f"{min(r.distance_m for r in rr):.2f}.."
            f"{max(r.distance_m for r in rr):.2f}m"
        )

    write_rows(
        output_dir
        / "session_data_quality.csv",

        session_quality_rows(
            records
        ),
    )

    with (
        output_dir
        / "dataset_configuration.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "repo_root":
                    str(
                        REPO_ROOT
                    ),

                "data_root":
                    str(
                        data_root
                    ),

                "num_raw_samples":
                    len(
                        raw_records
                    ),

                "num_samples_after_crop_selection":
                    len(
                        records
                    ),

                "crop_source":
                    crop_source,

                "session_ids":
                    sessions,

                "target_source":
                    "recomputed from raw Quest-world geometry",

                "target_definition":
                    (
                        "alpha = SignedAngle("
                        "camera_to_robot_planar, "
                        "robot_forward_planar, +Y)"
                    ),

                "model":
                    (
                        "ImageNet-pretrained ResNet18 fully fine-tuned + "
                        "512->128->2 orientation head"
                    ),

                "output_encoding":
                    "[sin(alpha), cos(alpha)]",

                "args":
                    vars(
                        args
                    ),
            },
            f,
            indent=2,
            allow_nan=False,
        )

    # Exact deterministic preprocessing associated with pretrained ResNet18.
    transform = (
        ResNet18_Weights
        .DEFAULT
        .transforms()
    )

    if (
        args.split_mode
        == "leave_one_session_out"
    ):
        splits = loso_splits(
            records,
            args.only_test_session,
        )

        fold_results: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        for fold_index, split in enumerate(
            splits,
            start=1,
        ):
            print()
            print(
                "#" * 90
            )
            print(
                f"LOSO {fold_index}/{len(splits)} "
                f"TEST={split.info['test_session_id']} "
                f"VAL={split.info['validation_session_id']}"
            )
            print(
                "#" * 90
            )

            seed_everything(
                args.seed
            )

            fold_dir = (
                output_dir
                / "leave_one_session_out_folds"
                / safe_name(
                    split.info[
                        "test_session_id"
                    ]
                )
            )

            fold_results.append(
                run_fold(
                    split=
                        split,

                    args=
                        args,

                    device=
                        device,

                    transform=
                        transform,

                    output_dir=
                        fold_dir,

                    crop_source=
                        crop_source,
                )
            )

            if (
                device.type
                == "cuda"
            ):
                torch.cuda.empty_cache()

        aggregate_loso(
            output_dir,
            fold_results,
        )

        train_final_model_on_all_data(
            records,
            args,
            device,
            transform,
            DEFAULT_WEIGHTS_OUTPUT_DIR,
        )

        return

    if (
        args.split_mode
        == "session_holdout"
    ):
        if (
            not args.validation_session
            or not args.test_session
        ):
            raise ValueError(
                "session_holdout requires --validation-session and --test-session."
            )

        split = session_holdout(
            records,
            args.validation_session,
            args.test_session,
        )

    elif (
        args.split_mode
        == "temporal"
    ):
        split = temporal_split(
            records,
            args.train_fraction,
            args.validation_fraction,
            args.split_gap_seconds,
        )

    else:
        print(
            "WARNING: RANDOM FRAME SPLIT CAN LEAK TEMPORALLY ADJACENT FRAMES."
        )

        split = random_split(
            records,
            args.train_fraction,
            args.validation_fraction,
            args.seed,
        )

    run_fold(
        split=
            split,

        args=
            args,

        device=
            device,

        transform=
            transform,

        output_dir=
            output_dir,

        crop_source=
            crop_source,
    )


if __name__ == "__main__":
    main()
