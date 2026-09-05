#!/usr/bin/env python3
"""
Fine-tune Orient Anything V1 on Quest-native Unitree Go2 orientation data
collected by SetRobotForwardQuest.cs and collect_orientation_est_data.py.

Default input:
  <repo>/robot_fov_estimation/data/forward_estimation_data/<session>/
      samples.csv
      images/frame_*.jpg

Before Orient Anything sees a frame, the script runs the same Go2 YOLO detector
used by the older orientation pipeline, takes the highest-confidence valid
"robot dog" detection, expands the box by 15% on each side, saves the crop once,
and uses that crop for every train/validation/test pass. YOLO cropping is ON by
default because deployment also supplies an object crop.

Primary target:
  alpha = SignedAngle(camera_to_robot_planar, robot_forward_planar, +Y)

The target is RECOMPUTED from the raw Quest-world camera position, robot center,
and robot-forward point stored in samples.csv. The recomputed value is checked
against signed_relative_yaw_deg and training aborts if they disagree beyond a
configurable tolerance.

Default evaluation is whole-session leave-one-session-out. For each test fold,
one *different complete session* is used for validation and all remaining
sessions are used for training. Training samples are temporally thinned and
session-balanced by default to avoid the near-frame leakage seen previously.

The actual deployment target is signed yaw in [-180,180). Magnitude and
undirected body-axis modes are retained only as optional diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoImageProcessor


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if p.name == "UnitreeGo2HandoffIntentDetector":
            return p.resolve()
        q = p / "UnitreeGo2HandoffIntentDetector"
        if q.is_dir():
            return q.resolve()
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()
DEFAULT_DATA_ROOT = REPO_ROOT / "robot_fov_estimation/data/forward_estimation_data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "robot_fov_estimation/outputs/orient_anything_forward_estimation_eval"
DEFAULT_WEIGHTS_OUTPUT_DIR = REPO_ROOT / "robot_fov_estimation/outputs/orient_anything_forward_estimation_weights"
DEFAULT_OA_DIR = REPO_ROOT / "Orient-Anything"

OA_CONFIG = {
    "small": {
        "dino_model": "facebook/dinov2-small",
        "in_dim": 384,
        "checkpoint_filename": "cropsmallEx03/dino_weight.pt",
    },
    "base": {
        "dino_model": "facebook/dinov2-base",
        "in_dim": 768,
        "checkpoint_filename": "cropbaseEx03/dino_weight.pt",
    },
    "large": {
        "dino_model": "facebook/dinov2-large",
        "in_dim": 1024,
        "checkpoint_filename": "croplargeEX2/dino_weight.pt",
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_deg_scalar(x: float) -> float:
    return (float(x) + 180.0) % 360.0 - 180.0


def wrap_deg_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x + 180.0) % 360.0 - 180.0


def circular_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(wrap_deg_np(np.asarray(pred) - np.asarray(target)))


def target_values_np(signed: np.ndarray, mode: str) -> np.ndarray:
    signed = wrap_deg_np(np.asarray(signed, dtype=np.float64))
    if mode == "signed":
        return signed
    mag = np.abs(signed)
    if mode == "magnitude":
        return mag
    if mode == "body_axis":
        return np.minimum(mag, 180.0 - mag)
    raise ValueError(mode)


def errors_np(pred: np.ndarray, signed_target: np.ndarray, mode: str) -> np.ndarray:
    if mode == "signed":
        return circular_error(pred, signed_target)
    return np.abs(np.asarray(pred) - target_values_np(signed_target, mode))


def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip()).strip("._")
    return s or "unnamed"


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def finite(row: Dict[str, str], key: str) -> float:
    try:
        x = float(row[key])
    except Exception as exc:
        raise ValueError(f"Invalid {key}={row.get(key)!r}") from exc
    if not math.isfinite(x):
        raise ValueError(f"Non-finite {key}={x}")
    return x


def recompute_signed_yaw(
    camera: Tuple[float, float, float],
    center: Tuple[float, float, float],
    forward_point: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    cx, cy, cz = camera
    rx, ry, rz = center
    px, py, pz = forward_point
    ray_x, ray_z = rx - cx, rz - cz
    fwd_x, fwd_z = px - rx, pz - rz
    ray_n = math.hypot(ray_x, ray_z)
    fwd_n = math.hypot(fwd_x, fwd_z)
    if ray_n < 1e-8:
        raise ValueError("Camera and robot center coincide in XZ")
    if fwd_n < 1e-8:
        raise ValueError("Robot forward marker does not define horizontal forward")
    ax, az = ray_x / ray_n, ray_z / ray_n
    bx, bz = fwd_x / fwd_n, fwd_z / fwd_n
    sin_term = az * bx - ax * bz
    cos_term = ax * bx + az * bz
    alpha = wrap_deg_scalar(math.degrees(math.atan2(sin_term, cos_term)))
    d3 = math.sqrt((rx-cx)**2 + (ry-cy)**2 + (rz-cz)**2)
    return alpha, d3, ray_n


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


REQUIRED = {
    "session_id", "frame_id", "capture_timestamp_unix", "capture_realtime_seconds",
    "measurement_minus_capture_seconds", "image_relative_path",
    "camera_position_world_x", "camera_position_world_y", "camera_position_world_z",
    "robot_center_world_x", "robot_center_world_y", "robot_center_world_z",
    "robot_forward_point_world_x", "robot_forward_point_world_y", "robot_forward_point_world_z",
    "signed_relative_yaw_deg",
}


def discover_sessions(data_root: Path) -> List[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    out = sorted(p for p in data_root.iterdir() if p.is_dir() and (p / "samples.csv").is_file())
    if not out:
        raise RuntimeError(f"No session directories containing samples.csv under {data_root}")
    return out


def load_records(data_root: Path, allowed_sessions: Optional[Sequence[str]], tolerance: float) -> List[Record]:
    allowed = set(allowed_sessions) if allowed_sessions else None
    records: List[Record] = []
    seen = set()
    for session_dir in discover_sessions(data_root):
        csv_path = session_dir / "samples.csv"
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = REQUIRED - set(reader.fieldnames or [])
            if missing:
                raise KeyError(f"{csv_path} missing columns: {sorted(missing)}")
            for line_no, row in enumerate(reader, start=2):
                sid = str(row["session_id"]).strip()
                if allowed is not None and sid not in allowed:
                    continue
                frame = int(row["frame_id"])
                key = (sid, frame)
                if key in seen:
                    raise RuntimeError(f"Duplicate sample {key}")
                seen.add(key)
                image = (session_dir / row["image_relative_path"]).resolve()
                if not image.is_file():
                    raise FileNotFoundError(image)
                camera = tuple(finite(row, k) for k in (
                    "camera_position_world_x", "camera_position_world_y", "camera_position_world_z"))
                center = tuple(finite(row, k) for k in (
                    "robot_center_world_x", "robot_center_world_y", "robot_center_world_z"))
                point = tuple(finite(row, k) for k in (
                    "robot_forward_point_world_x", "robot_forward_point_world_y", "robot_forward_point_world_z"))
                alpha, distance, horizontal = recompute_signed_yaw(camera, center, point)
                saved = wrap_deg_scalar(finite(row, "signed_relative_yaw_deg"))
                label_err = float(circular_error(np.array([alpha]), np.array([saved]))[0])
                if label_err > tolerance:
                    raise RuntimeError(
                        f"Quest geometry label check failed at {csv_path}:{line_no}\n"
                        f"session={sid} frame={frame} recomputed={alpha:.6f} saved={saved:.6f} "
                        f"error={label_err:.6f} tolerance={tolerance:.6f}"
                    )
                records.append(Record(
                    session_id=sid,
                    session_label=str(row.get("session_label", "") or ""),
                    frame_id=frame,
                    capture_timestamp_unix=finite(row, "capture_timestamp_unix"),
                    capture_realtime_seconds=finite(row, "capture_realtime_seconds"),
                    measurement_minus_capture_seconds=finite(row, "measurement_minus_capture_seconds"),
                    image_path=image,
                    target_signed_deg=alpha,
                    saved_signed_deg=saved,
                    label_error_deg=label_err,
                    distance_m=distance,
                    horizontal_distance_m=horizontal,
                    camera_x=camera[0], camera_y=camera[1], camera_z=camera[2],
                    center_x=center[0], center_y=center[1], center_z=center[2],
                    forward_x=point[0], forward_y=point[1], forward_z=point[2],
                ))
    if not records:
        raise RuntimeError("No samples loaded")
    records.sort(key=lambda r: (r.capture_timestamp_unix, r.session_id, r.frame_id))
    return records


def ordered_session_ids(records: Sequence[Record]) -> List[str]:
    first: Dict[str, float] = {}
    for r in records:
        first[r.session_id] = min(first.get(r.session_id, math.inf), r.capture_timestamp_unix)
    return sorted(first, key=lambda s: (first[s], s))


def group_sessions(records: Sequence[Record]) -> Dict[str, List[Record]]:
    g: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        g[r.session_id].append(r)
    for sid in g:
        g[sid].sort(key=lambda r: (r.capture_timestamp_unix, r.frame_id))
    return dict(g)


@dataclass
class Split:
    name: str
    train: List[Record]
    validation: List[Record]
    test: List[Record]
    purged: List[Record]
    info: Dict[str, Any]


def loso_splits(records: Sequence[Record], only_test: Optional[str]) -> List[Split]:
    sessions = ordered_session_ids(records)
    if len(sessions) < 3:
        raise RuntimeError("leave_one_session_out needs at least 3 sessions")
    if len(sessions) == 3:
        print("WARNING: only 3 sessions: each fold has just one training session. Four+ preferred.")
    g = group_sessions(records)
    out: List[Split] = []
    for i, test_sid in enumerate(sessions):
        if only_test is not None and test_sid != only_test:
            continue
        val_sid = sessions[(i + 1) % len(sessions)]
        train_sids = [s for s in sessions if s not in {test_sid, val_sid}]
        train: List[Record] = []
        for s in train_sids:
            train.extend(g[s])
        out.append(Split(
            name=f"test_{safe_name(test_sid)}",
            train=train,
            validation=list(g[val_sid]),
            test=list(g[test_sid]),
            purged=[],
            info={"strategy": "whole-session LOSO", "train_session_ids": train_sids,
                  "validation_session_id": val_sid, "test_session_id": test_sid},
        ))
    if not out:
        raise ValueError(f"No test session matched {only_test!r}")
    return out


def session_holdout(records: Sequence[Record], val_sid: str, test_sid: str) -> Split:
    sessions = ordered_session_ids(records)
    if val_sid not in sessions or test_sid not in sessions or val_sid == test_sid:
        raise ValueError(f"Invalid validation/test sessions. Available: {sessions}")
    g = group_sessions(records)
    train_sids = [s for s in sessions if s not in {val_sid, test_sid}]
    if not train_sids:
        raise RuntimeError("No training sessions remain")
    train: List[Record] = []
    for s in train_sids:
        train.extend(g[s])
    return Split("session_holdout", train, list(g[val_sid]), list(g[test_sid]), [], {
        "strategy": "whole-session holdout", "train_session_ids": train_sids,
        "validation_session_id": val_sid, "test_session_id": test_sid})


def random_split(records: Sequence[Record], train_frac: float, val_frac: float, seed: int) -> Split:
    items = list(records)
    random.Random(seed).shuffle(items)
    n = len(items)
    nt = max(1, int(n * train_frac))
    nv = max(1, int(n * val_frac))
    if nt + nv >= n:
        nv = max(1, n - nt - 1)
    return Split("random", items[:nt], items[nt:nt+nv], items[nt+nv:], [], {
        "strategy": "random-frame diagnostic; temporal leakage possible"})


def temporal_split(records: Sequence[Record], train_frac: float, val_frac: float, gap_s: float) -> Split:
    g = group_sessions(records)
    train: List[Record] = []
    val: List[Record] = []
    test: List[Record] = []
    purged: List[Record] = []
    for sid in ordered_session_ids(records):
        rr = g[sid]
        n = len(rr)
        a = min(max(1, int(n * train_frac)), n - 2)
        b = min(max(a + 1, int(n * (train_frac + val_frac))), n - 1)
        t1 = 0.5 * (rr[a-1].capture_timestamp_unix + rr[a].capture_timestamp_unix)
        t2 = 0.5 * (rr[b-1].capture_timestamp_unix + rr[b].capture_timestamp_unix)
        for i, r in enumerate(rr):
            t = r.capture_timestamp_unix
            if i < a:
                (purged if gap_s > 0 and t > t1 - gap_s/2 else train).append(r)
            elif i < b:
                (purged if gap_s > 0 and (t < t1 + gap_s/2 or t > t2 - gap_s/2) else val).append(r)
            else:
                (purged if gap_s > 0 and t < t2 + gap_s/2 else test).append(r)
    if not train or not val or not test:
        raise RuntimeError("Temporal split produced empty split")
    return Split("temporal", train, val, test, purged, {"strategy": "within-session chronological"})


def thin_training(records: Sequence[Record], spacing_s: float) -> Tuple[List[Record], List[Record]]:
    if spacing_s <= 0:
        return list(records), []
    g = group_sessions(records)
    kept: List[Record] = []
    removed: List[Record] = []
    for sid in ordered_session_ids(records):
        last = -math.inf
        for r in g[sid]:
            if r.capture_timestamp_unix - last >= spacing_s:
                kept.append(r)
                last = r.capture_timestamp_unix
            else:
                removed.append(r)
    return kept, removed


# =============================================================================
# Deployment-matched YOLO crop preprocessing
# =============================================================================

def load_yolo_detector(args: argparse.Namespace):
    """
    Load the same project detector wrapper used by the previous Go2 orientation
    scripts. If --robot-obj-det-weights-path is omitted, YOLOGo2Detector uses
    its normal project-default weights, exactly as the old scripts did.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from robot_fov_estimation.src.go2_yolo_det_wrapper import YOLOGo2Detector

    kwargs = {
        "confidence": args.yolo_confidence,
        "iou_threshold": args.yolo_iou,
        "image_size": args.yolo_image_size,
        "device": args.yolo_device,
    }

    if args.robot_obj_det_weights_path is not None:
        weights_path = Path(args.robot_obj_det_weights_path).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights do not exist: {weights_path}"
            )
        kwargs["weights_path"] = str(weights_path)

    print("Loading YOLO Go2 detector...")
    return YOLOGo2Detector(**kwargs)


def clamp_box(
    box_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    x1 = max(0, min(image_width - 1, int(round(x1))))
    y1 = max(0, min(image_height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(image_width, int(round(x2))))
    y2 = max(y1 + 1, min(image_height, int(round(y2))))

    return x1, y1, x2, y2


def expand_box(
    box_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    width = x2 - x1
    height = y2 - y1

    return clamp_box(
        (
            x1 - width * padding_fraction,
            y1 - height * padding_fraction,
            x2 + width * padding_fraction,
            y2 + height * padding_fraction,
        ),
        image_width,
        image_height,
    )


def detect_best_robot(
    detector,
    image_rgb: np.ndarray,
    class_name: str,
) -> Optional[Dict[str, Any]]:
    detections = detector.predict(
        image_rgb,
        [class_name],
    )

    if not detections:
        return None

    valid = [
        detection
        for detection in detections
        if (
            detection.get("box_xyxy") is not None
            and detection.get("score") is not None
        )
    ]

    if not valid:
        return None

    return max(
        valid,
        key=lambda detection: float(detection["score"]),
    )


def make_yolo_crop_path(
    crop_root: Path,
    record: Record,
) -> Path:
    return (
        crop_root
        / safe_name(record.session_id)
        / f"frame_{record.frame_id:020d}.jpg"
    )


def preprocess_records_with_yolo(
    records: Sequence[Record],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[List[Record], List[Dict[str, Any]]]:
    """
    Run YOLO exactly once before any OA fold is created.

    This ensures:
      * every fold sees the identical crop for a given source frame;
      * YOLO is not re-run every epoch;
      * train/validation/test preprocessing matches deployment;
      * a YOLO miss never silently turns into a full-image OA input.

    Returns records whose image_path points to the generated YOLO crop.
    """
    if args.no_yolo_crop:
        print(
            "WARNING: YOLO cropping is DISABLED by --no-yolo-crop. "
            "OA will consume the collected full images."
        )
        return list(records), []

    detector = load_yolo_detector(args)
    crop_root = output_dir / "yolo_crops"
    crop_root.mkdir(parents=True, exist_ok=True)

    usable: List[Record] = []
    manifest: List[Dict[str, Any]] = []
    misses_by_session: Counter = Counter()
    found_by_session: Counter = Counter()

    print()
    print("=" * 88)
    print("PRECOMPUTING DEPLOYMENT-MATCHED YOLO CROPS")
    print("=" * 88)
    print(f"Class name:      {args.yolo_class_name}")
    print(f"Confidence:      {args.yolo_confidence:g}")
    print(f"IoU threshold:   {args.yolo_iou:g}")
    print(f"YOLO image size: {args.yolo_image_size}")
    print(f"Crop padding:    {args.crop_padding:g}")
    print(f"Frames:          {len(records)}")
    print(f"Crop root:       {crop_root}")

    for index, record in enumerate(records, start=1):
        source_path = record.image_path
        crop_path = make_yolo_crop_path(crop_root, record)

        try:
            with Image.open(source_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")

            image_rgb = np.asarray(image, dtype=np.uint8)
            detection = detect_best_robot(
                detector,
                image_rgb,
                args.yolo_class_name,
            )

            if detection is None:
                misses_by_session[record.session_id] += 1
                manifest.append({
                    "session_id": record.session_id,
                    "frame_id": record.frame_id,
                    "source_image_path": str(source_path),
                    "crop_image_path": "",
                    "robot_detected": False,
                    "yolo_score": "",
                    "raw_x1": "",
                    "raw_y1": "",
                    "raw_x2": "",
                    "raw_y2": "",
                    "crop_x1": "",
                    "crop_y1": "",
                    "crop_x2": "",
                    "crop_y2": "",
                    "status": "no_detection",
                })

                if args.yolo_missing_policy == "error":
                    raise RuntimeError(
                        "YOLO did not detect the Go2 in "
                        f"{record.session_id} frame {record.frame_id}: {source_path}"
                    )

                continue

            raw_box = clamp_box(
                detection["box_xyxy"],
                image.width,
                image.height,
            )

            crop_box = expand_box(
                raw_box,
                image.width,
                image.height,
                args.crop_padding,
            )

            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop = image.crop(crop_box)

            if crop.width < 2 or crop.height < 2:
                raise RuntimeError(
                    f"YOLO produced an invalid crop for {source_path}: {crop_box}"
                )

            crop.save(
                crop_path,
                format="JPEG",
                quality=args.jpeg_quality,
            )

            found_by_session[record.session_id] += 1

            # Record is intentionally reconstructed rather than mutated.
            # All geometry/labels are unchanged; only OA's image input changes.
            cropped_record = Record(
                session_id=record.session_id,
                session_label=record.session_label,
                frame_id=record.frame_id,
                capture_timestamp_unix=record.capture_timestamp_unix,
                capture_realtime_seconds=record.capture_realtime_seconds,
                measurement_minus_capture_seconds=record.measurement_minus_capture_seconds,
                image_path=crop_path.resolve(),
                target_signed_deg=record.target_signed_deg,
                saved_signed_deg=record.saved_signed_deg,
                label_error_deg=record.label_error_deg,
                distance_m=record.distance_m,
                horizontal_distance_m=record.horizontal_distance_m,
                camera_x=record.camera_x,
                camera_y=record.camera_y,
                camera_z=record.camera_z,
                center_x=record.center_x,
                center_y=record.center_y,
                center_z=record.center_z,
                forward_x=record.forward_x,
                forward_y=record.forward_y,
                forward_z=record.forward_z,
            )
            usable.append(cropped_record)

            score = float(detection["score"])
            manifest.append({
                "session_id": record.session_id,
                "frame_id": record.frame_id,
                "source_image_path": str(source_path),
                "crop_image_path": str(crop_path.resolve()),
                "robot_detected": True,
                "yolo_score": score,
                "raw_x1": raw_box[0],
                "raw_y1": raw_box[1],
                "raw_x2": raw_box[2],
                "raw_y2": raw_box[3],
                "crop_x1": crop_box[0],
                "crop_y1": crop_box[1],
                "crop_x2": crop_box[2],
                "crop_y2": crop_box[3],
                "status": "ok",
            })

        except Exception as exc:
            if (
                args.yolo_missing_policy == "skip"
                and not isinstance(exc, RuntimeError)
            ):
                misses_by_session[record.session_id] += 1
                manifest.append({
                    "session_id": record.session_id,
                    "frame_id": record.frame_id,
                    "source_image_path": str(source_path),
                    "crop_image_path": "",
                    "robot_detected": False,
                    "yolo_score": "",
                    "status": f"error_skipped: {type(exc).__name__}: {exc}",
                })
                continue
            raise

        if index % 100 == 0 or index == len(records):
            print(
                f"YOLO crops: {index}/{len(records)} processed | "
                f"{len(usable)} detected"
            )

    write_rows(
        output_dir / "yolo_crop_manifest.csv",
        manifest,
    )

    summary_rows: List[Dict[str, Any]] = []
    for sid in ordered_session_ids(records):
        total = sum(1 for r in records if r.session_id == sid)
        found = int(found_by_session[sid])
        missed = total - found
        summary_rows.append({
            "session_id": sid,
            "total_frames": total,
            "detected_frames": found,
            "missed_frames": missed,
            "detection_fraction": found / total if total else 0.0,
        })

    write_rows(
        output_dir / "yolo_crop_summary_by_session.csv",
        summary_rows,
    )

    if not usable:
        raise RuntimeError(
            "YOLO did not produce any usable Go2 crops."
        )

    remaining_sessions = ordered_session_ids(usable)
    print()
    print(
        f"YOLO preprocessing complete: {len(usable)}/{len(records)} "
        f"frames retained ({100.0 * len(usable) / len(records):.1f}%)."
    )
    for row in summary_rows:
        print(
            f"  {row['session_id']}: "
            f"{row['detected_frames']}/{row['total_frames']} "
            f"({100.0 * row['detection_fraction']:.1f}%)"
        )

    if len(remaining_sessions) < 3 and args.split_mode == "leave_one_session_out":
        raise RuntimeError(
            "After YOLO preprocessing fewer than 3 sessions contain usable "
            "frames, so leave-one-session-out cannot be run."
        )

    return usable, manifest


# =============================================================================
# Image dataset / augmentation
# =============================================================================

def center_crop_fraction(image: Image.Image, fraction: float) -> Image.Image:
    if fraction >= 0.999999:
        return image
    w, h = image.size
    cw = max(2, int(round(w * fraction)))
    ch = max(2, int(round(h * fraction)))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return image.crop((x0, y0, x0 + cw, y0 + ch))


def random_same_aspect_crop(image: Image.Image, prob: float, min_scale: float) -> Image.Image:
    if prob <= 0 or random.random() >= prob:
        return image
    w, h = image.size
    scale = math.sqrt(random.uniform(min_scale, 1.0))
    cw = max(2, int(round(w * scale)))
    ch = max(2, int(round(h * scale)))
    x0 = random.randint(0, max(0, w - cw)) if w > cw else 0
    y0 = random.randint(0, max(0, h - ch)) if h > ch else 0
    return image.crop((x0, y0, x0 + cw, y0 + ch))


def color_jitter(image: Image.Image, brightness: float, contrast: float, color: float) -> Image.Image:
    if brightness > 0:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(max(0, 1-brightness), 1+brightness))
    if contrast > 0:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(max(0, 1-contrast), 1+contrast))
    if color > 0:
        image = ImageEnhance.Color(image).enhance(random.uniform(max(0, 1-color), 1+color))
    return image


class OrientationDataset(Dataset):
    def __init__(self, records: Sequence[Record], processor: AutoImageProcessor, args: argparse.Namespace, training: bool):
        self.records = list(records)
        self.processor = processor
        self.args = args
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        r = self.records[index]
        with Image.open(r.image_path) as im:
            image = im.convert("RGB")
        image = center_crop_fraction(image, self.args.center_crop_fraction)
        target = float(r.target_signed_deg)
        if self.training:
            image = random_same_aspect_crop(image, self.args.random_crop_prob, self.args.random_crop_min_scale)
            image = color_jitter(image, self.args.brightness_jitter, self.args.contrast_jitter, self.args.color_jitter)
            if self.args.horizontal_flip_prob > 0 and random.random() < self.args.horizontal_flip_prob:
                T = getattr(Image, "Transpose", Image)
                image = image.transpose(T.FLIP_LEFT_RIGHT)
                target = wrap_deg_scalar(-target)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel_values,
            "target_signed_deg": torch.tensor(target, dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }


def make_sampler(records: Sequence[Record], enabled: bool) -> Optional[WeightedRandomSampler]:
    if not enabled:
        return None
    counts = Counter(r.session_id for r in records)
    weights = torch.tensor([1.0 / counts[r.session_id] for r in records], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


def make_loader(records: Sequence[Record], processor: AutoImageProcessor, args: argparse.Namespace,
                device: torch.device, training: bool) -> DataLoader:
    ds = OrientationDataset(records, processor, args, training)
    sampler = make_sampler(records, args.session_balanced_sampling) if training else None
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


# =============================================================================
# Orient Anything loading and bin mapping
# =============================================================================

def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = obj.get(key)
            if isinstance(candidate, dict) and candidate:
                return strip_module_prefix(candidate)
        if any(torch.is_tensor(v) for v in obj.values()):
            return strip_module_prefix(obj)
    raise TypeError("Could not extract OA state_dict")


def infer_out_dim(state: Dict[str, torch.Tensor]) -> int:
    for key in ("down_sampler.net2.0.weight", "down_sampler.net2.1.weight"):
        x = state.get(key)
        if torch.is_tensor(x) and x.ndim >= 1:
            return int(x.shape[0])
    candidates = [(k, v.shape) for k, v in state.items()
                  if k.startswith("down_sampler") and torch.is_tensor(v) and v.ndim == 2]
    if not candidates:
        raise KeyError("Cannot infer OA output dimension")
    candidates.sort(key=lambda x: x[0])
    return int(candidates[-1][1][0])


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.oa_checkpoint:
        p = Path(args.oa_checkpoint).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id="Viglong/Orient-Anything",
        filename=OA_CONFIG[args.oa_scale]["checkpoint_filename"],
        repo_type="model",
        cache_dir=args.hf_cache_dir or None,
    )).resolve()


def load_oa(args: argparse.Namespace, device: torch.device):
    oa_dir = Path(args.oa_dir).expanduser().resolve()
    if not (oa_dir / "vision_tower.py").is_file():
        raise FileNotFoundError(f"Expected Orient Anything vision_tower.py under {oa_dir}")
    if str(oa_dir) not in sys.path:
        sys.path.insert(0, str(oa_dir))
    from vision_tower import DINOv2_MLP
    ckpt = resolve_checkpoint(args)
    state = extract_state_dict(torch.load(ckpt, map_location="cpu"))
    out_dim = infer_out_dim(state)
    if out_dim < 360:
        raise RuntimeError(f"OA output dim {out_dim} < 360")
    model = DINOv2_MLP(
        dino_mode=args.oa_scale,
        in_dim=OA_CONFIG[args.oa_scale]["in_dim"],
        out_dim=out_dim,
        evaluate=True,
        mask_dino=False,
        frozen_back=False,
    )
    inc = model.load_state_dict(state, strict=False)
    if inc.missing_keys or inc.unexpected_keys:
        raise RuntimeError(f"OA checkpoint mismatch. Missing={inc.missing_keys[:10]} Unexpected={inc.unexpected_keys[:10]}")
    if args.freeze_backbone:
        for p in model.dinov2.parameters():
            p.requires_grad_(False)
    return model.to(device), ckpt, out_dim


def model_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    if isinstance(output, dict):
        for k in ("logits", "output"):
            if torch.is_tensor(output.get(k)):
                return output[k]
    raise TypeError("OA forward output did not contain tensor logits")


def remap_logits(raw: torch.Tensor, sign: int, offset: int) -> torch.Tensor:
    bins = torch.arange(360, device=raw.device, dtype=torch.long)
    source = torch.remainder(sign * (bins - int(offset)), 360)
    return raw.index_select(1, source)


@torch.no_grad()
def collect_raw_argmax(model, loader: DataLoader, device: torch.device, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred: List[float] = []
    tgt: List[float] = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        out = model_logits(model({"pixel_values": pixels}))[:, :360]
        pred.extend(torch.argmax(out, dim=1).cpu().numpy().astype(float).tolist())
        tgt.extend(batch["target_signed_deg"].numpy().astype(float).tolist())
        if max_samples > 0 and len(tgt) >= max_samples:
            break
    if max_samples > 0:
        pred, tgt = pred[:max_samples], tgt[:max_samples]
    return np.asarray(pred), np.asarray(tgt)


def fit_mapping(raw_bins: np.ndarray, target: np.ndarray) -> Tuple[int, int, Dict[str, float]]:
    best = None
    for sign in (1, -1):
        for offset in range(360):
            mapped = wrap_deg_np(np.mod(sign * raw_bins + offset, 360.0))
            err = circular_error(mapped, target)
            score = (float(np.median(err)), float(np.mean(err)))
            if best is None or score < best[0]:
                best = (score, sign, offset, err)
    _, sign, offset, err = best
    return int(sign), int(offset), {
        "mapping_samples": int(len(err)),
        "mapping_mean_error_deg": float(np.mean(err)),
        "mapping_median_error_deg": float(np.median(err)),
    }

# =============================================================================
# Targets, decoding, metrics
# =============================================================================

def circular_soft_targets(target_signed: torch.Tensor, sigma: float) -> torch.Tensor:
    centers = torch.arange(360, device=target_signed.device, dtype=target_signed.dtype)
    t = torch.remainder(target_signed, 360.0).unsqueeze(1)
    diff = torch.remainder(centers.unsqueeze(0) - t + 180.0, 360.0) - 180.0
    return F.softmax(-0.5 * (diff / max(float(sigma), 1e-6)) ** 2, dim=1)


def fold_to_magnitude(logits: torch.Tensor) -> torch.Tensor:
    zero = logits[:, 0:1]
    pos = logits[:, 1:180]
    neg = torch.flip(logits[:, 181:360], dims=[1])
    paired = torch.logsumexp(torch.stack([pos, neg], dim=0), dim=0)
    return torch.cat([zero, paired, logits[:, 180:181]], dim=1)


def fold_to_axis(mag_logits: torch.Tensor) -> torch.Tensor:
    low = mag_logits[:, 0:90]
    high = torch.flip(mag_logits[:, 91:181], dims=[1])
    paired = torch.logsumexp(torch.stack([low, high], dim=0), dim=0)
    return torch.cat([paired, mag_logits[:, 90:91]], dim=1)


def linear_soft_targets(target: torch.Tensor, bins: int, sigma: float) -> torch.Tensor:
    centers = torch.arange(bins, device=target.device, dtype=target.dtype)
    return F.softmax(-0.5 * ((centers.unsqueeze(0) - target.unsqueeze(1)) / max(float(sigma), 1e-6)) ** 2, dim=1)


def training_logits_and_targets(mapped: torch.Tensor, signed: torch.Tensor, mode: str, sigma: float):
    if mode == "signed":
        return mapped, circular_soft_targets(signed, sigma)
    if mode == "magnitude":
        logits = fold_to_magnitude(mapped)
        target = torch.abs(torch.remainder(signed + 180.0, 360.0) - 180.0)
        return logits, linear_soft_targets(target, 181, sigma)
    if mode == "body_axis":
        mag_logits = fold_to_magnitude(mapped)
        logits = fold_to_axis(mag_logits)
        mag = torch.abs(torch.remainder(signed + 180.0, 360.0) - 180.0)
        target = torch.minimum(mag, 180.0 - mag)
        return logits, linear_soft_targets(target, 91, sigma)
    raise ValueError(mode)


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def decode(logits: torch.Tensor, mode: str):
    probs = F.softmax(logits, dim=1)
    arg = torch.argmax(probs, dim=1).float()
    max_prob = probs.max(dim=1).values
    if mode == "signed":
        arg = torch.remainder(arg + 180.0, 360.0) - 180.0
        rad = torch.deg2rad(torch.arange(360, device=logits.device, dtype=logits.dtype))
        c = (probs * torch.cos(rad).unsqueeze(0)).sum(dim=1)
        s = (probs * torch.sin(rad).unsqueeze(0)).sum(dim=1)
        mean = torch.rad2deg(torch.atan2(s, c))
        concentration = torch.sqrt(c*c + s*s)
        return arg, mean, max_prob, concentration
    centers = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
    mean = (probs * centers.unsqueeze(0)).sum(dim=1)
    return arg, mean, max_prob, max_prob


def metrics(errors: np.ndarray) -> Dict[str, float]:
    e = np.asarray(errors, dtype=np.float64)
    return {
        "n": int(len(e)),
        "mean_error_deg": float(np.mean(e)),
        "median_error_deg": float(np.median(e)),
        "p90_error_deg": float(np.percentile(e, 90)),
        "p95_error_deg": float(np.percentile(e, 95)),
        "within_5_deg": float(np.mean(e <= 5)),
        "within_10_deg": float(np.mean(e <= 10)),
        "within_15_deg": float(np.mean(e <= 15)),
        "within_30_deg": float(np.mean(e <= 30)),
    }


def print_metrics(title: str, m: Dict[str, float]) -> None:
    print(f"\n{title}\n{'-'*len(title)}")
    print(f"N={m['n']} mean={m['mean_error_deg']:.2f} med={m['median_error_deg']:.2f} "
          f"P90={m['p90_error_deg']:.2f} P95={m['p95_error_deg']:.2f} deg")
    print(f"<=5 {100*m['within_5_deg']:.1f}% | <=10 {100*m['within_10_deg']:.1f}% | "
          f"<=15 {100*m['within_15_deg']:.1f}% | <=30 {100*m['within_30_deg']:.1f}%")


def best_constant(train_signed: np.ndarray, mode: str) -> float:
    if mode != "signed":
        return float(np.median(target_values_np(train_signed, mode)))
    best = None
    for c in range(-180, 180):
        e = circular_error(np.full(len(train_signed), c), train_signed)
        score = (float(np.mean(e)), float(np.median(e)))
        if best is None or score < best[0]:
            best = (score, float(c))
    return best[1]


def constant_metrics(value: float, records: Sequence[Record], mode: str) -> Dict[str, float]:
    target = np.asarray([r.target_signed_deg for r in records])
    return metrics(errors_np(np.full(len(target), value), target, mode))


# =============================================================================
# Optimization / training
# =============================================================================

def freeze_bn_stats(model) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def make_optimizer(model, args: argparse.Namespace):
    groups = []
    backbone = [p for p in model.dinov2.parameters() if p.requires_grad]
    head = [p for p in model.down_sampler.parameters() if p.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr})
    if head:
        groups.append({"params": head, "lr": args.head_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def f(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        x = min(max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * x))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, args, map_sign, map_offset) -> float:
    model.train()
    if args.freeze_batchnorm_stats:
        freeze_bn_stats(model)
    optimizer.zero_grad(set_to_none=True)
    total, n = 0.0, 0
    for step, batch in enumerate(loader):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["target_signed_deg"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=args.use_amp and device.type == "cuda"):
            raw = model_logits(model({"pixel_values": pixels}))[:, :360]
            mapped = remap_logits(raw, map_sign, map_offset)
            logits, target_probs = training_logits_and_targets(mapped, target, args.target_mode, args.target_sigma_deg)
            raw_loss = soft_cross_entropy(logits, target_probs)
            loss = raw_loss / args.grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        bs = pixels.shape[0]
        total += float(raw_loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, records: Sequence[Record], device, args, map_sign, map_offset) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    total_loss, n = 0.0, 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["target_signed_deg"].to(device, non_blocking=True)
        indices = batch["index"].cpu().numpy().tolist()
        raw = model_logits(model({"pixel_values": pixels}))[:, :360]
        mapped = remap_logits(raw, map_sign, map_offset)
        logits, target_probs = training_logits_and_targets(mapped, target, args.target_mode, args.target_sigma_deg)
        loss = soft_cross_entropy(logits, target_probs)
        arg, mean, max_prob, concentration = decode(logits, args.target_mode)
        target_np = target.cpu().numpy().astype(float)
        arg_np = arg.cpu().numpy().astype(float)
        mean_np = mean.cpu().numpy().astype(float)
        ae = errors_np(arg_np, target_np, args.target_mode)
        me = errors_np(mean_np, target_np, args.target_mode)
        tv = target_values_np(target_np, args.target_mode)
        mp = max_prob.cpu().numpy().astype(float)
        cc = concentration.cpu().numpy().astype(float)
        for j, idx in enumerate(indices):
            r = records[int(idx)]
            rows.append({
                "session_id": r.session_id,
                "session_label": r.session_label,
                "frame_id": r.frame_id,
                "capture_timestamp_unix": r.capture_timestamp_unix,
                "image_path": str(r.image_path),
                "target_mode": args.target_mode,
                "target_signed_yaw_deg": r.target_signed_deg,
                "target_value_deg": float(tv[j]),
                "argmax_prediction_deg": float(arg_np[j]),
                "mean_prediction_deg": float(mean_np[j]),
                "argmax_error_deg": float(ae[j]),
                "mean_error_deg": float(me[j]),
                "max_probability": float(mp[j]),
                "concentration": float(cc[j]),
                "distance_m": r.distance_m,
                "horizontal_distance_m": r.horizontal_distance_m,
                "measurement_minus_capture_seconds": r.measurement_minus_capture_seconds,
            })
        bs = pixels.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs
    arg_err = np.asarray([r["argmax_error_deg"] for r in rows])
    mean_err = np.asarray([r["mean_error_deg"] for r in rows])
    return {"loss": total_loss / max(n, 1), "argmax_metrics": metrics(arg_err),
            "mean_metrics": metrics(mean_err), "rows": rows}

# =============================================================================
# Diagnostics / fold execution
# =============================================================================

def session_quality(records: Sequence[Record]) -> List[Dict[str, Any]]:
    rows = []
    g = group_sessions(records)
    for sid in ordered_session_ids(records):
        rr = g[sid]
        centers = np.asarray([[r.center_x, r.center_y, r.center_z] for r in rr])
        center_drift = np.linalg.norm(centers - centers[0], axis=1)
        fwd_yaw = []
        for r in rr:
            dx, dz = r.forward_x - r.center_x, r.forward_z - r.center_z
            fwd_yaw.append(math.degrees(math.atan2(dx, dz)))
        fwd_yaw = np.asarray(fwd_yaw)
        fwd_drift = circular_error(fwd_yaw, np.full(len(fwd_yaw), fwd_yaw[0]))
        dt = np.asarray([r.measurement_minus_capture_seconds for r in rr])
        rows.append({
            "session_id": sid,
            "session_label": rr[0].session_label,
            "num_samples": len(rr),
            "duration_s": rr[-1].capture_timestamp_unix - rr[0].capture_timestamp_unix,
            "distance_min_m": min(r.distance_m for r in rr),
            "distance_median_m": float(np.median([r.distance_m for r in rr])),
            "distance_max_m": max(r.distance_m for r in rr),
            "robot_center_max_drift_m": float(np.max(center_drift)),
            "robot_forward_max_drift_deg": float(np.max(fwd_drift)),
            "measurement_capture_abs_p95_ms": float(1000*np.percentile(np.abs(dt), 95)),
            "max_geometry_vs_saved_yaw_error_deg": max(r.label_error_deg for r in rr),
        })
    return rows


def coverage_rows(records: Sequence[Record], split_name: str) -> List[Dict[str, Any]]:
    out = []
    for low in range(-180, 180, 30):
        high = low + 30
        selected = [r for r in records if low <= r.target_signed_deg < high]
        out.append({"split": split_name, "low_deg": low, "high_deg": high,
                    "count": len(selected), "fraction": len(selected)/max(1, len(records))})
    return out


def manifest_rows(split: Split, train_used: Sequence[Record], thinned: Sequence[Record]) -> List[Dict[str, Any]]:
    rows = []
    for name, rr in (("train_used", train_used), ("train_thinned_out", thinned),
                     ("validation", split.validation), ("test", split.test), ("purged", split.purged)):
        for r in rr:
            rows.append({
                "split": name, "session_id": r.session_id, "frame_id": r.frame_id,
                "capture_timestamp_unix": r.capture_timestamp_unix,
                "image_path": str(r.image_path), "target_signed_yaw_deg": r.target_signed_deg,
                "target_magnitude_deg": abs(r.target_signed_deg),
                "target_body_axis_deg": min(abs(r.target_signed_deg), 180-abs(r.target_signed_deg)),
                "distance_m": r.distance_m,
            })
    return rows


def parse_distance_edges(raw: str) -> List[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip().lower()
        vals.append(math.inf if token in {"inf", "+inf", "infinity"} else float(token))
    if len(vals) < 2 or any(b <= a for a, b in zip(vals[:-1], vals[1:])):
        raise ValueError("distance edges must be strictly increasing")
    return vals


def distance_diagnostics(rows: Sequence[Dict[str, Any]], split_name: str, edges: Sequence[float]) -> List[Dict[str, Any]]:
    out = []
    for decoder in ("argmax", "mean"):
        ek = decoder + "_error_deg"
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for r in rows if float(r["distance_m"]) >= lo and (math.isinf(hi) or float(r["distance_m"]) < hi)]
            if sel:
                m = metrics(np.asarray([r[ek] for r in sel]))
            else:
                m = {"n": 0, "mean_error_deg": "", "median_error_deg": "", "p90_error_deg": "", "p95_error_deg": "",
                     "within_5_deg": "", "within_10_deg": "", "within_15_deg": "", "within_30_deg": ""}
            out.append({"split": split_name, "decoder": decoder,
                        "distance_bin": f"[{lo:g},{'inf' if math.isinf(hi) else f'{hi:g}'})", **m})
    return out


def run_fold(split: Split, args: argparse.Namespace, device: torch.device, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*90}\nFOLD {split.name}\n{'='*90}")
    print(f"Raw train={len(split.train)} val={len(split.validation)} test={len(split.test)}")
    print(f"Train sessions: {split.info.get('train_session_ids', [])}")
    print(f"Validation: {split.info.get('validation_session_id', 'n/a')}")
    print(f"Test: {split.info.get('test_session_id', 'n/a')}")

    train_records, thinned = thin_training(split.train, args.temporal_thin_seconds)
    if not train_records:
        raise RuntimeError("No training records after thinning")
    print(f"Temporal thinning {args.temporal_thin_seconds:g}s: kept {len(train_records)}, removed {len(thinned)}")
    write_rows(output_dir / "split_manifest.csv", manifest_rows(split, train_records, thinned))
    cov = []
    for name, rr in (("train", train_records), ("validation", split.validation), ("test", split.test)):
        cov.extend(coverage_rows(rr, name))
    write_rows(output_dir / "signed_target_coverage.csv", cov)

    processor = AutoImageProcessor.from_pretrained(
        OA_CONFIG[args.oa_scale]["dino_model"], cache_dir=args.hf_cache_dir or None)
    model, oa_ckpt, out_dim = load_oa(args, device)
    print(f"OA checkpoint: {oa_ckpt}")
    print(f"Trainable DINO={sum(p.numel() for p in model.dinov2.parameters() if p.requires_grad):,} "
          f"head={sum(p.numel() for p in model.down_sampler.parameters() if p.requires_grad):,}")

    mapping_loader = make_loader(train_records, processor, args, device, False)
    if args.yaw_map_mode == "identity":
        map_sign, map_offset, map_info = 1, 0, {"mapping_mode": "identity"}
    elif args.yaw_map_mode == "manual":
        map_sign, map_offset = args.yaw_map_sign, args.yaw_map_offset_deg % 360
        map_info = {"mapping_mode": "manual"}
    else:
        raw, tgt = collect_raw_argmax(model, mapping_loader, device, args.yaw_map_max_samples)
        map_sign, map_offset, fit = fit_mapping(raw, tgt)
        map_info = {"mapping_mode": "auto", **fit}
    print(f"Yaw map: project_bin = {map_sign:+d} * OA_bin + {map_offset} (mod 360)")

    train_loader = make_loader(train_records, processor, args, device, True)
    train_eval_loader = make_loader(train_records, processor, args, device, False)
    val_loader = make_loader(split.validation, processor, args, device, False)
    test_loader = make_loader(split.test, processor, args, device, False)

    train_signed = np.asarray([r.target_signed_deg for r in train_records])
    const = best_constant(train_signed, args.target_mode)
    const_train = constant_metrics(const, train_records, args.target_mode)
    const_val = constant_metrics(const, split.validation, args.target_mode)
    const_test = constant_metrics(const, split.test, args.target_mode)
    print(f"Train-derived constant={const:.2f} deg | val MAE={const_val['mean_error_deg']:.2f} | test MAE={const_test['mean_error_deg']:.2f}")

    pretrained_val = evaluate(model, val_loader, split.validation, device, args, map_sign, map_offset)
    pretrained_test = evaluate(model, test_loader, split.test, device, args, map_sign, map_offset)
    print_metrics("Pretrained OA validation (mean decoder)", pretrained_val["mean_metrics"])
    print_metrics("Pretrained OA test (mean decoder)", pretrained_test["mean_metrics"])

    optimizer = make_optimizer(model, args)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = max(1, args.epochs * updates_per_epoch)
    scheduler = make_scheduler(optimizer, total_steps, int(round(total_steps * args.warmup_fraction)))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    history = []
    best_mae = math.inf
    best_epoch = 0
    stale = 0
    best_path = output_dir / "best_orient_anything_forward_estimation.pt"
    print(f"Backbone LR={args.backbone_lr:g} head LR={args.head_lr:g} sigma={args.target_sigma_deg:g} deg")
    print(f"Target={args.target_mode} session-balanced={args.session_balanced_sampling} flip={args.horizontal_flip_prob:g}")

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, args, map_sign, map_offset)
        tr = evaluate(model, train_eval_loader, train_records, device, args, map_sign, map_offset)
        va = evaluate(model, val_loader, split.validation, device, args, map_sign, map_offset)
        trm, vam = tr["mean_metrics"], va["mean_metrics"]
        improvement = const_val["mean_error_deg"] - vam["mean_error_deg"]
        print(f"Epoch {epoch:03d} | loss {loss:.4f} | train {trm['mean_error_deg']:.2f}/{trm['median_error_deg']:.2f} | "
              f"val {vam['mean_error_deg']:.2f}/{vam['median_error_deg']:.2f} | const {const_val['mean_error_deg']:.2f} | beat {improvement:+.2f}")
        history.append({
            "epoch": epoch, "train_loss": loss,
            "train_mean_error_deg": trm["mean_error_deg"], "train_median_error_deg": trm["median_error_deg"],
            "validation_loss": va["loss"], "validation_mean_error_deg": vam["mean_error_deg"],
            "validation_median_error_deg": vam["median_error_deg"],
            "validation_constant_mean_error_deg": const_val["mean_error_deg"],
            "validation_improvement_vs_constant_deg": improvement,
        })
        if vam["mean_error_deg"] < best_mae - args.minimum_improvement_deg:
            best_mae = vam["mean_error_deg"]
            best_epoch = epoch
            stale = 0
            torch.save({
                "schema_version": 1,
                "model_state_dict": model.state_dict(),
                "oa_scale": args.oa_scale,
                "oa_output_dim": out_dim,
                "dino_model": OA_CONFIG[args.oa_scale]["dino_model"],
                "oa_checkpoint": str(oa_ckpt),
                "target_mode": args.target_mode,
                "target_definition": "SignedAngle(camera_to_robot_planar, robot_forward_planar, +Y)",
            "image_preprocessing": (
                "YOLOGo2Detector highest-confidence robot dog detection + padded bbox crop"
                if not args.no_yolo_crop else "full collected image (--no-yolo-crop)"
            ),
                "yaw_map_sign": map_sign,
                "yaw_map_offset_deg": map_offset,
                "center_crop_fraction": args.center_crop_fraction,
                "yolo_crop_enabled": not args.no_yolo_crop,
                "yolo_class_name": args.yolo_class_name,
                "yolo_crop_padding": args.crop_padding,
                "yolo_confidence": args.yolo_confidence,
                "yolo_iou": args.yolo_iou,
                "yolo_image_size": args.yolo_image_size,
                "robot_obj_det_weights_path": args.robot_obj_det_weights_path,
                "best_epoch": best_epoch,
                "best_validation_mean_error_deg": best_mae,
                "split_info": split.info,
                "training_args": vars(args),
            }, best_path)
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f"Early stopping at {epoch}; best epoch={best_epoch}")
            break

    write_rows(output_dir / "training_history.csv", history)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_train = evaluate(model, train_eval_loader, train_records, device, args, map_sign, map_offset)
    final_val = evaluate(model, val_loader, split.validation, device, args, map_sign, map_offset)
    final_test = evaluate(model, test_loader, split.test, device, args, map_sign, map_offset)
    print(f"\nBEST EPOCH {best_epoch}")
    print_metrics("Final test argmax", final_test["argmax_metrics"])
    print_metrics("Final test mean/circular-mean", final_test["mean_metrics"])

    write_rows(output_dir / "train_predictions.csv", final_train["rows"])
    write_rows(output_dir / "validation_predictions.csv", final_val["rows"])
    write_rows(output_dir / "test_predictions.csv", final_test["rows"])
    edges = parse_distance_edges(args.distance_bin_edges)
    dr = []
    for name, result in (("train", final_train), ("validation", final_val), ("test", final_test)):
        dr.extend(distance_diagnostics(result["rows"], name, edges))
    write_rows(output_dir / "error_by_distance.csv", dr)

    summary = {
        "split": split.info,
        "counts": {"raw_train": len(split.train), "train_used": len(train_records), "train_thinned_out": len(thinned),
                   "validation": len(split.validation), "test": len(split.test)},
        "oa": {"checkpoint": str(oa_ckpt), "scale": args.oa_scale, "output_dim": out_dim,
               "yaw_map_sign": map_sign, "yaw_map_offset_deg": map_offset, **map_info},
        "training": {"target_mode": args.target_mode, "target_sigma_deg": args.target_sigma_deg,
                     "temporal_thin_seconds": args.temporal_thin_seconds,
                     "session_balanced_sampling": args.session_balanced_sampling,
                     "best_epoch": best_epoch, "best_validation_mean_error_deg": best_mae},
        "constant_baseline": {"prediction_deg": const, "train": const_train, "validation": const_val, "test": const_test},
        "pretrained": {"validation_argmax": pretrained_val["argmax_metrics"], "validation_mean": pretrained_val["mean_metrics"],
                       "test_argmax": pretrained_test["argmax_metrics"], "test_mean": pretrained_test["mean_metrics"]},
        "final": {"train_argmax": final_train["argmax_metrics"], "train_mean": final_train["mean_metrics"],
                  "validation_argmax": final_val["argmax_metrics"], "validation_mean": final_val["mean_metrics"],
                  "test_argmax": final_test["argmax_metrics"], "test_mean": final_test["mean_metrics"]},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    return {"summary": summary, "test_rows": final_test["rows"]}


def aggregate_loso(
    output_dir: Path,
    results: Sequence[Dict[str, Any]],
    target_mode: str,
) -> None:
    """
    Aggregate whole-session LOSO results.

    In addition to the original aggregate outputs, this reports:

      * macro mean test MAE across held-out sessions;
      * sample SD of held-out-session MAEs;
      * 95% t confidence interval across held-out sessions;
      * P90/P95 and threshold accuracies for each fold;
      * catastrophic-error rates >=90 deg and >=150 deg;
      * exact-180-degree oracle-corrected performance;
      * pooled performance by important target-yaw region.

    IMPORTANT:
        Confidence intervals are computed across LOSO folds/sessions, NOT
        across individual frames. The held-out session is the independent
        generalization unit.
    """

    # ------------------------------------------------------------------
    # Small statistical helpers
    # ------------------------------------------------------------------

    def t_critical_95(df: int) -> float:
        """
        Two-sided 95% Student-t critical value.

        A lookup table avoids adding scipy as a dependency. For df > 30,
        1.96 is a sufficiently close asymptotic approximation for this
        reporting purpose.
        """
        table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            11: 2.201,
            12: 2.179,
            13: 2.160,
            14: 2.145,
            15: 2.131,
            16: 2.120,
            17: 2.110,
            18: 2.101,
            19: 2.093,
            20: 2.086,
            21: 2.080,
            22: 2.074,
            23: 2.069,
            24: 2.064,
            25: 2.060,
            26: 2.056,
            27: 2.052,
            28: 2.048,
            29: 2.045,
            30: 2.042,
        }

        if df <= 0:
            return float("nan")

        return table.get(df, 1.96)


    def summarize_across_folds(
        values: Sequence[float],
    ) -> Dict[str, float]:
        """
        Mean, sample SD, SEM, and two-sided 95% t CI across LOSO folds.
        """
        x = np.asarray(values, dtype=np.float64)

        if len(x) == 0:
            return {
                "n_folds": 0,
                "mean": float("nan"),
                "sd": float("nan"),
                "sem": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
            }

        mean = float(np.mean(x))

        if len(x) == 1:
            return {
                "n_folds": 1,
                "mean": mean,
                "sd": float("nan"),
                "sem": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
            }

        sd = float(np.std(x, ddof=1))
        sem = sd / math.sqrt(len(x))
        critical = t_critical_95(len(x) - 1)
        half_width = critical * sem

        return {
            "n_folds": int(len(x)),
            "mean": mean,
            "sd": sd,
            "sem": sem,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
        }


    def catastrophic_stats(
        errors: np.ndarray,
    ) -> Dict[str, float]:
        e = np.asarray(errors, dtype=np.float64)

        if len(e) == 0:
            return {
                "fraction_ge_90_deg": float("nan"),
                "fraction_ge_150_deg": float("nan"),
                "count_ge_90_deg": 0,
                "count_ge_150_deg": 0,
            }

        ge90 = e >= 90.0
        ge150 = e >= 150.0

        return {
            "fraction_ge_90_deg": float(np.mean(ge90)),
            "fraction_ge_150_deg": float(np.mean(ge150)),
            "count_ge_90_deg": int(np.sum(ge90)),
            "count_ge_150_deg": int(np.sum(ge150)),
        }


    def oracle_180_stats(
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Diagnostic only.

        For each prediction, compare its signed-yaw error with the error
        obtained after adding exactly 180 degrees. This quantifies how much
        of the remaining error can be explained specifically by front/rear
        branch confusion.

        This is NOT a deployable metric and should be labeled as an oracle
        diagnostic if reported.
        """
        prediction = np.asarray(
            prediction,
            dtype=np.float64,
        )

        target = np.asarray(
            target,
            dtype=np.float64,
        )

        original_error = circular_error(
            prediction,
            target,
        )

        flipped_prediction = wrap_deg_np(
            prediction + 180.0
        )

        flipped_error = circular_error(
            flipped_prediction,
            target,
        )

        corrected_error = np.minimum(
            original_error,
            flipped_error,
        )

        improved = (
            flipped_error <
            original_error
        )

        return {
            "fraction_better_after_exact_180_flip":
                float(np.mean(improved)),
            "count_better_after_exact_180_flip":
                int(np.sum(improved)),
            "corrected_metrics":
                metrics(corrected_error),
        }


    # ------------------------------------------------------------------
    # Fold-level results
    # ------------------------------------------------------------------

    fold_rows: List[Dict[str, Any]] = []
    pooled: List[Dict[str, Any]] = []

    for result in results:
        summary = result["summary"]
        info = summary["split"]

        test_mean_metrics = (
            summary["final"]["test_mean"]
        )

        test_argmax_metrics = (
            summary["final"]["test_argmax"]
        )

        constant_metrics_test = (
            summary["constant_baseline"]["test"]
        )

        test_rows = result["test_rows"]

        mean_errors = np.asarray(
            [
                row["mean_error_deg"]
                for row in test_rows
            ],
            dtype=np.float64,
        )

        argmax_errors = np.asarray(
            [
                row["argmax_error_deg"]
                for row in test_rows
            ],
            dtype=np.float64,
        )

        target = np.asarray(
            [
                row["target_signed_yaw_deg"]
                for row in test_rows
            ],
            dtype=np.float64,
        )

        mean_prediction = np.asarray(
            [
                row["mean_prediction_deg"]
                for row in test_rows
            ],
            dtype=np.float64,
        )

        argmax_prediction = np.asarray(
            [
                row["argmax_prediction_deg"]
                for row in test_rows
            ],
            dtype=np.float64,
        )

        mean_cat = catastrophic_stats(
            mean_errors
        )

        argmax_cat = catastrophic_stats(
            argmax_errors
        )

        mean_oracle = oracle_180_stats(
            mean_prediction,
            target,
        )

        argmax_oracle = oracle_180_stats(
            argmax_prediction,
            target,
        )

        fold_rows.append({
            "test_session_id":
                info["test_session_id"],

            "validation_session_id":
                info["validation_session_id"],

            "train_session_ids":
                "|".join(
                    info["train_session_ids"]
                ),

            "best_epoch":
                summary["training"]["best_epoch"],

            # -----------------------------
            # Mean / circular-mean decoder
            # -----------------------------
            "test_mean_decoder_mean_error_deg":
                test_mean_metrics[
                    "mean_error_deg"
                ],

            "test_mean_decoder_median_error_deg":
                test_mean_metrics[
                    "median_error_deg"
                ],

            "test_mean_decoder_p90_error_deg":
                test_mean_metrics[
                    "p90_error_deg"
                ],

            "test_mean_decoder_p95_error_deg":
                test_mean_metrics[
                    "p95_error_deg"
                ],

            "test_mean_decoder_within_5":
                test_mean_metrics[
                    "within_5_deg"
                ],

            "test_mean_decoder_within_10":
                test_mean_metrics[
                    "within_10_deg"
                ],

            "test_mean_decoder_within_15":
                test_mean_metrics[
                    "within_15_deg"
                ],

            "test_mean_decoder_within_30":
                test_mean_metrics[
                    "within_30_deg"
                ],

            "test_mean_decoder_fraction_ge_90_deg":
                mean_cat[
                    "fraction_ge_90_deg"
                ],

            "test_mean_decoder_fraction_ge_150_deg":
                mean_cat[
                    "fraction_ge_150_deg"
                ],

            "test_mean_decoder_oracle_180_mae_deg":
                mean_oracle[
                    "corrected_metrics"
                ]["mean_error_deg"],

            "test_mean_decoder_fraction_better_after_180_flip":
                mean_oracle[
                    "fraction_better_after_exact_180_flip"
                ],

            # -----------------------------
            # Argmax decoder
            # -----------------------------
            "test_argmax_mean_error_deg":
                test_argmax_metrics[
                    "mean_error_deg"
                ],

            "test_argmax_median_error_deg":
                test_argmax_metrics[
                    "median_error_deg"
                ],

            "test_argmax_p90_error_deg":
                test_argmax_metrics[
                    "p90_error_deg"
                ],

            "test_argmax_p95_error_deg":
                test_argmax_metrics[
                    "p95_error_deg"
                ],

            "test_argmax_fraction_ge_90_deg":
                argmax_cat[
                    "fraction_ge_90_deg"
                ],

            "test_argmax_fraction_ge_150_deg":
                argmax_cat[
                    "fraction_ge_150_deg"
                ],

            "test_argmax_oracle_180_mae_deg":
                argmax_oracle[
                    "corrected_metrics"
                ]["mean_error_deg"],

            # -----------------------------
            # Constant baseline
            # -----------------------------
            "constant_mean_error_deg":
                constant_metrics_test[
                    "mean_error_deg"
                ],

            "improvement_vs_constant_deg":
                constant_metrics_test[
                    "mean_error_deg"
                ]
                -
                test_mean_metrics[
                    "mean_error_deg"
                ],
        })

        pooled.extend(test_rows)

    write_rows(
        output_dir /
        "leave_one_session_out_summary.csv",
        fold_rows,
    )

    write_rows(
        output_dir /
        "leave_one_session_out_predictions.csv",
        pooled,
    )

    # ------------------------------------------------------------------
    # Pooled frame-level arrays
    # ------------------------------------------------------------------

    target = np.asarray(
        [
            row["target_signed_yaw_deg"]
            for row in pooled
        ],
        dtype=np.float64,
    )

    argmax_prediction = np.asarray(
        [
            row["argmax_prediction_deg"]
            for row in pooled
        ],
        dtype=np.float64,
    )

    mean_prediction = np.asarray(
        [
            row["mean_prediction_deg"]
            for row in pooled
        ],
        dtype=np.float64,
    )

    argmax_error = np.asarray(
        [
            row["argmax_error_deg"]
            for row in pooled
        ],
        dtype=np.float64,
    )

    mean_error = np.asarray(
        [
            row["mean_error_deg"]
            for row in pooled
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Across-session / fold uncertainty
    # ------------------------------------------------------------------

    mean_fold_mae = [
        row[
            "test_mean_decoder_mean_error_deg"
        ]
        for row in fold_rows
    ]

    argmax_fold_mae = [
        row[
            "test_argmax_mean_error_deg"
        ]
        for row in fold_rows
    ]

    constant_fold_mae = [
        row[
            "constant_mean_error_deg"
        ]
        for row in fold_rows
    ]

    improvement_fold = [
        row[
            "improvement_vs_constant_deg"
        ]
        for row in fold_rows
    ]

    mean_across_folds = summarize_across_folds(
        mean_fold_mae
    )

    argmax_across_folds = summarize_across_folds(
        argmax_fold_mae
    )

    constant_across_folds = summarize_across_folds(
        constant_fold_mae
    )

    improvement_across_folds = summarize_across_folds(
        improvement_fold
    )

    # ------------------------------------------------------------------
    # Catastrophic and exact-180 diagnostics
    # ------------------------------------------------------------------

    pooled_mean_cat = catastrophic_stats(
        mean_error
    )

    pooled_argmax_cat = catastrophic_stats(
        argmax_error
    )

    pooled_mean_oracle = oracle_180_stats(
        mean_prediction,
        target,
    )

    pooled_argmax_oracle = oracle_180_stats(
        argmax_prediction,
        target,
    )

    # ------------------------------------------------------------------
    # Pooled error by target-yaw region
    #
    # For the current target definition:
    #   yaw ~= 0 deg    -> camera is behind robot / rear visible
    #   yaw ~= 180 deg  -> camera is in front / front visible
    # ------------------------------------------------------------------

    wrapped_target = wrap_deg_np(target)
    abs_target = np.abs(wrapped_target)

    yaw_regions = [
        (
            "rear_abs_yaw_le_20",
            abs_target <= 20.0,
        ),
        (
            "side_mid_20_to_160",
            (
                (abs_target > 20.0)
                &
                (abs_target < 160.0)
            ),
        ),
        (
            "front_abs_yaw_ge_160",
            abs_target >= 160.0,
        ),
    ]

    yaw_region_rows: List[
        Dict[str, Any]
    ] = []

    for region_name, mask in yaw_regions:
        count = int(np.sum(mask))

        if count == 0:
            continue

        for (
            decoder_name,
            prediction,
            errors,
        ) in (
            (
                "mean",
                mean_prediction,
                mean_error,
            ),
            (
                "argmax",
                argmax_prediction,
                argmax_error,
            ),
        ):
            region_errors = errors[mask]

            region_metrics = metrics(
                region_errors
            )

            region_cat = catastrophic_stats(
                region_errors
            )

            region_oracle = oracle_180_stats(
                prediction[mask],
                target[mask],
            )

            yaw_region_rows.append({
                "region":
                    region_name,

                "decoder":
                    decoder_name,

                "n":
                    count,

                **region_metrics,

                **region_cat,

                "oracle_180_mean_error_deg":
                    region_oracle[
                        "corrected_metrics"
                    ]["mean_error_deg"],

                "fraction_better_after_exact_180_flip":
                    region_oracle[
                        "fraction_better_after_exact_180_flip"
                    ],
            })

    write_rows(
        output_dir /
        "leave_one_session_out_error_by_yaw_region.csv",
        yaw_region_rows,
    )

    # ------------------------------------------------------------------
    # Final JSON aggregate
    # ------------------------------------------------------------------

    aggregate = {
        "target_mode":
            target_mode,

        "num_folds":
            len(fold_rows),

        "num_pooled_test_frames":
            len(pooled),

        "folds":
            fold_rows,

        # Preserve the original keys for compatibility.
        "macro_mean_test_argmax_mae_deg":
            argmax_across_folds["mean"],

        "macro_mean_test_mean_decoder_mae_deg":
            mean_across_folds["mean"],

        "pooled_argmax":
            metrics(argmax_error),

        "pooled_mean_decoder":
            metrics(mean_error),

        # New session-level uncertainty.
        "across_folds": {
            "mean_decoder_mae_deg":
                mean_across_folds,

            "argmax_mae_deg":
                argmax_across_folds,

            "constant_baseline_mae_deg":
                constant_across_folds,

            "improvement_vs_constant_deg":
                improvement_across_folds,
        },

        # New catastrophic diagnostics.
        "catastrophic_errors": {
            "mean_decoder":
                pooled_mean_cat,

            "argmax":
                pooled_argmax_cat,
        },

        # New exact-180 diagnostic.
        "oracle_180_diagnostic": {
            "mean_decoder":
                pooled_mean_oracle,

            "argmax":
                pooled_argmax_oracle,
        },

        "yaw_region_performance":
            yaw_region_rows,
    }

    with (
        output_dir /
        "leave_one_session_out_summary.json"
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

    # ------------------------------------------------------------------
    # Human-readable console summary
    # ------------------------------------------------------------------

    print(
        f"\n{'=' * 90}\n"
        "LEAVE-ONE-SESSION-OUT SUMMARY\n"
        f"{'=' * 90}"
    )

    for row in fold_rows:
        print(
            f"{row['test_session_id']}: "
            f"mean="
            f"{row['test_mean_decoder_mean_error_deg']:.2f} "
            f"argmax="
            f"{row['test_argmax_mean_error_deg']:.2f} "
            f"constant="
            f"{row['constant_mean_error_deg']:.2f} "
            f"improvement="
            f"{row['improvement_vs_constant_deg']:+.2f} "
            f"| >=90="
            f"{100.0 * row['test_mean_decoder_fraction_ge_90_deg']:.1f}% "
            f">=150="
            f"{100.0 * row['test_mean_decoder_fraction_ge_150_deg']:.1f}% "
            f"| oracle180="
            f"{row['test_mean_decoder_oracle_180_mae_deg']:.2f}"
        )

    print()
    print(
        "Mean/circular-mean decoder across held-out sessions:"
    )

    print(
        f"  MAE = "
        f"{mean_across_folds['mean']:.2f} "
        f"+/- "
        f"{mean_across_folds['sd']:.2f} deg "
        f"(SD across sessions)"
    )

    print(
        f"  95% CI = "
        f"[{mean_across_folds['ci95_low']:.2f}, "
        f"{mean_across_folds['ci95_high']:.2f}] deg"
    )

    print(
        f"  Pooled frame MAE = "
        f"{np.mean(mean_error):.2f} deg"
    )

    print(
        f"  Pooled >=90 deg errors = "
        f"{100.0 * pooled_mean_cat['fraction_ge_90_deg']:.2f}%"
    )

    print(
        f"  Pooled >=150 deg errors = "
        f"{100.0 * pooled_mean_cat['fraction_ge_150_deg']:.2f}%"
    )

    print(
        f"  Exact-180 oracle-corrected pooled MAE = "
        f"{pooled_mean_oracle['corrected_metrics']['mean_error_deg']:.2f} deg"
    )

    print(
        f"  Frames improved by exact 180 deg flip = "
        f"{100.0 * pooled_mean_oracle['fraction_better_after_exact_180_flip']:.2f}%"
    )

    print()
    print("Pooled target-yaw regions:")

    for row in yaw_region_rows:
        if row["decoder"] != "mean":
            continue

        print(
            f"  {row['region']}: "
            f"N={row['n']} "
            f"MAE={row['mean_error_deg']:.2f} "
            f"median={row['median_error_deg']:.2f} "
            f"P90={row['p90_error_deg']:.2f} "
            f">=150="
            f"{100.0 * row['fraction_ge_150_deg']:.1f}% "
            f"oracle180="
            f"{row['oracle_180_mean_error_deg']:.2f}"
        )


def full_data_split(records: Sequence[Record]) -> Split:
    sessions = ordered_session_ids(records)
    if not sessions:
        raise RuntimeError("No sessions available for full-data training")
    return Split(
        name="all_data",
        train=list(records),
        validation=[],
        test=[],
        purged=[],
        info={"strategy": "all available data", "train_session_ids": sessions},
    )


def train_final_model_on_all_data(
    records: Sequence[Record],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_args = argparse.Namespace(**vars(args))
    split = full_data_split(records)
    print(f"\n{'='*90}\nFINAL TRAINING ON ALL AVAILABLE DATA\n{'='*90}")
    print(f"Sessions:   {len(split.info['train_session_ids'])}")
    print(f"Train sessions: {split.info['train_session_ids']}")

    processor = AutoImageProcessor.from_pretrained(
        OA_CONFIG[final_args.oa_scale]["dino_model"], cache_dir=final_args.hf_cache_dir or None)
    model, oa_ckpt, out_dim = load_oa(final_args, device)
    print(f"OA checkpoint: {oa_ckpt}")
    print(f"Trainable DINO={sum(p.numel() for p in model.dinov2.parameters() if p.requires_grad):,} "
          f"head={sum(p.numel() for p in model.down_sampler.parameters() if p.requires_grad):,}")

    mapping_loader = make_loader(records, processor, final_args, device, False)
    if final_args.yaw_map_mode == "identity":
        map_sign, map_offset, map_info = 1, 0, {"mapping_mode": "identity"}
    elif final_args.yaw_map_mode == "manual":
        map_sign, map_offset = final_args.yaw_map_sign, final_args.yaw_map_offset_deg % 360
        map_info = {"mapping_mode": "manual"}
    else:
        raw, tgt = collect_raw_argmax(model, mapping_loader, device, final_args.yaw_map_max_samples)
        map_sign, map_offset, fit = fit_mapping(raw, tgt)
        map_info = {"mapping_mode": "auto", **fit}
    print(f"Yaw map: project_bin = {map_sign:+d} * OA_bin + {map_offset} (mod 360)")

    train_loader = make_loader(records, processor, final_args, device, True)
    train_eval_loader = make_loader(records, processor, final_args, device, False)

    train_signed = np.asarray([r.target_signed_deg for r in records])
    const = best_constant(train_signed, final_args.target_mode)
    const_train = constant_metrics(const, records, final_args.target_mode)
    print(f"Train-derived constant={const:.2f} deg | train MAE={const_train['mean_error_deg']:.2f}")

    pretrained_train = evaluate(model, train_eval_loader, records, device, final_args, map_sign, map_offset)
    print_metrics("Pretrained OA train (mean decoder)", pretrained_train["mean_metrics"])

    optimizer = make_optimizer(model, final_args)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / final_args.grad_accum))
    total_steps = max(1, final_args.epochs * updates_per_epoch)
    scheduler = make_scheduler(optimizer, total_steps, int(round(total_steps * final_args.warmup_fraction)))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=final_args.use_amp and device.type == "cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=final_args.use_amp and device.type == "cuda")

    history = []
    best_mae = math.inf
    best_epoch = 0
    best_path = output_dir / "best_orient_anything_forward_estimation.pt"
    print(f"Backbone LR={final_args.backbone_lr:g} head LR={final_args.head_lr:g} sigma={final_args.target_sigma_deg:g} deg")
    print(f"Target={final_args.target_mode} session-balanced={final_args.session_balanced_sampling} flip={final_args.horizontal_flip_prob:g}")

    for epoch in range(1, final_args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, final_args, map_sign, map_offset)
        tr = evaluate(model, train_eval_loader, records, device, final_args, map_sign, map_offset)
        trm = tr["mean_metrics"]
        print(f"Epoch {epoch:03d} | loss {loss:.4f} | train {trm['mean_error_deg']:.2f}/{trm['median_error_deg']:.2f} | "
              f"const {const_train['mean_error_deg']:.2f}")
        history.append({
            "epoch": epoch, "train_loss": loss,
            "train_mean_error_deg": trm["mean_error_deg"], "train_median_error_deg": trm["median_error_deg"],
            "train_constant_mean_error_deg": const_train["mean_error_deg"],
            "train_improvement_vs_constant_deg": const_train["mean_error_deg"] - trm["mean_error_deg"],
        })
        if trm["mean_error_deg"] < best_mae:
            best_mae = trm["mean_error_deg"]
            best_epoch = epoch
            torch.save({
                "schema_version": 1,
                "model_state_dict": model.state_dict(),
                "oa_scale": final_args.oa_scale,
                "oa_output_dim": out_dim,
                "dino_model": OA_CONFIG[final_args.oa_scale]["dino_model"],
                "oa_checkpoint": str(oa_ckpt),
                "target_mode": final_args.target_mode,
                "target_definition": "SignedAngle(camera_to_robot_planar, robot_forward_planar, +Y)",
                "image_preprocessing": (
                    "YOLOGo2Detector highest-confidence robot dog detection + padded bbox crop"
                    if not final_args.no_yolo_crop else "full collected image (--no-yolo-crop)"
                ),
                "yaw_map_sign": map_sign,
                "yaw_map_offset_deg": map_offset,
                "center_crop_fraction": final_args.center_crop_fraction,
                "yolo_crop_enabled": not final_args.no_yolo_crop,
                "yolo_class_name": final_args.yolo_class_name,
                "yolo_crop_padding": final_args.crop_padding,
                "yolo_confidence": final_args.yolo_confidence,
                "yolo_iou": final_args.yolo_iou,
                "yolo_image_size": final_args.yolo_image_size,
                "robot_obj_det_weights_path": final_args.robot_obj_det_weights_path,
                "training_args": vars(final_args),
                "train_session_ids": split.info["train_session_ids"],
                "train_constant_baseline": const_train,
                "pretrained_train": pretrained_train["mean_metrics"],
                "best_epoch": best_epoch,
                "best_train_mean_error_deg": best_mae,
            }, best_path)

    write_rows(output_dir / "training_history.csv", history)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(
    f"\nBest final-training checkpoint: "
    f"epoch {best_epoch}/{final_args.epochs} | "
    f"train mean MAE={best_mae:.2f} deg"
    )

    final_train = evaluate(model, train_eval_loader, records, device, final_args, map_sign, map_offset)
    print_metrics("Final train argmax", final_train["argmax_metrics"])
    print_metrics("Final train mean/circular-mean", final_train["mean_metrics"])
    write_rows(output_dir / "train_predictions.csv", final_train["rows"])

    summary = {
        "split": split.info,
        "counts": {"train": len(records)},
        "oa": {"checkpoint": str(oa_ckpt), "scale": final_args.oa_scale, "output_dim": out_dim,
               "yaw_map_sign": map_sign, "yaw_map_offset_deg": map_offset, **map_info},
        "training": {"target_mode": final_args.target_mode, "target_sigma_deg": final_args.target_sigma_deg,
                     "temporal_thin_seconds": final_args.temporal_thin_seconds,
                     "session_balanced_sampling": final_args.session_balanced_sampling,
                     "best_epoch": best_epoch, "best_train_mean_error_deg": best_mae},
        "constant_baseline": {"prediction_deg": const, "train": const_train},
        "pretrained": {"train_mean": pretrained_train["mean_metrics"], "train_argmax": pretrained_train["argmax_metrics"]},
        "final": {"train_argmax": final_train["argmax_metrics"], "train_mean": final_train["mean_metrics"]},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    return {"summary": summary, "train_rows": final_train["rows"]}

# =============================================================================
# CLI / main
# =============================================================================

def parse_session_filter(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return vals or None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Orient Anything on Quest-native Go2 orientation sessions")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--sessions", default=None, help="Comma-separated exact session IDs to include")
    p.add_argument("--label-consistency-tolerance-deg", type=float, default=0.25)

    p.add_argument("--target-mode", choices=("signed", "magnitude", "body_axis"), default="signed")
    p.add_argument("--target-sigma-deg", type=float, default=10.0)

    p.add_argument("--oa-dir", default=str(DEFAULT_OA_DIR))
    p.add_argument("--oa-scale", choices=("small", "base", "large"), default="small")
    p.add_argument("--oa-checkpoint", default=None)
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--yaw-map-mode", choices=("auto", "identity", "manual"), default="auto")
    p.add_argument("--yaw-map-sign", type=int, choices=(-1, 1), default=1)
    p.add_argument("--yaw-map-offset-deg", type=int, default=0)
    p.add_argument("--yaw-map-max-samples", type=int, default=1000)

    p.add_argument("--split-mode", choices=("leave_one_session_out", "session_holdout", "temporal", "random"),
                   default="leave_one_session_out")
    p.add_argument("--only-test-session", default=None)
    p.add_argument("--validation-session", default=None)
    p.add_argument("--test-session", default=None)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--split-gap-seconds", type=float, default=1.0)

    p.add_argument("--temporal-thin-seconds", type=float, default=0.5,
                   help="Minimum spacing between TRAINING frames in each session")
    p.add_argument("--no-session-balanced-sampling", action="store_true")

    # Deployment-matched Go2 YOLO crop. ON by default.
    p.add_argument(
        "--no-yolo-crop",
        action="store_true",
        help="Disable Go2 YOLO cropping (full-image ablation only).",
    )
    p.add_argument(
        "--robot-obj-det-weights-path",
        default=None,
        help=(
            "Optional explicit Go2 YOLO checkpoint. If omitted, "
            "YOLOGo2Detector uses the same project-default weights as the old scripts."
        ),
    )
    p.add_argument("--yolo-class-name", default="robot dog")
    p.add_argument("--crop-padding", type=float, default=0.15)
    p.add_argument("--yolo-confidence", type=float, default=0.05)
    p.add_argument("--yolo-iou", type=float, default=0.50)
    p.add_argument("--yolo-image-size", type=int, default=640)
    p.add_argument("--yolo-device", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument(
        "--yolo-missing-policy",
        choices=("skip", "error"),
        default="skip",
        help=(
            "What to do when YOLO does not detect the Go2. Default 'skip' "
            "never falls back to the full image."
        ),
    )

    p.add_argument("--center-crop-fraction", type=float, default=1.0,
                   help="Fixed centered crop AFTER the YOLO crop; 1.0 keeps the entire padded YOLO crop")
    p.add_argument("--horizontal-flip-prob", type=float, default=0.0,
                   help="Signed target is negated when image is flipped")
    p.add_argument("--random-crop-prob", type=float, default=0.0)
    p.add_argument("--random-crop-min-scale", type=float, default=0.80)
    p.add_argument("--brightness-jitter", type=float, default=0.0)
    p.add_argument("--contrast-jitter", type=float, default=0.0)
    p.add_argument("--color-jitter", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--minimum-improvement-deg", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--backbone-lr", type=float, default=1e-6)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--allow-batchnorm-stat-updates", action="store_true")
    p.add_argument("--no-amp", action="store_true")

    p.add_argument("--distance-bin-edges", default="0,2,4,6,8,inf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    args.use_amp = not args.no_amp
    args.freeze_batchnorm_stats = not args.allow_batchnorm_stat_updates
    args.session_balanced_sampling = not args.no_session_balanced_sampling
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.target_sigma_deg <= 0:
        raise ValueError("target sigma must be >0")
    if args.label_consistency_tolerance_deg < 0 or args.temporal_thin_seconds < 0:
        raise ValueError("tolerances/spacings must be >=0")
    if not (0.1 <= args.center_crop_fraction <= 1.0):
        raise ValueError("center crop fraction must be in [0.1,1]")
    if args.crop_padding < 0.0:
        raise ValueError("--crop-padding must be >= 0")
    if not (0.0 <= args.yolo_confidence <= 1.0):
        raise ValueError("--yolo-confidence must be in [0,1]")
    if not (0.0 <= args.yolo_iou <= 1.0):
        raise ValueError("--yolo-iou must be in [0,1]")
    if args.yolo_image_size < 1:
        raise ValueError("--yolo-image-size must be >= 1")
    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be in [1,100]")
    for x in (args.horizontal_flip_prob, args.random_crop_prob):
        if not (0 <= x <= 1):
            raise ValueError("augmentation probabilities must be in [0,1]")
    if not (0.1 <= args.random_crop_min_scale <= 1.0):
        raise ValueError("random crop min scale must be in [0.1,1]")
    if args.epochs < 1 or args.min_epochs < 1 or args.min_epochs > args.epochs:
        raise ValueError("require 1 <= min_epochs <= epochs")
    if args.patience < 1 or args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("patience/batch-size/grad-accum must be >=1")
    if args.split_mode in {"temporal", "random"}:
        if not (0 < args.train_fraction < 1 and 0 < args.validation_fraction < 1 and
                args.train_fraction + args.validation_fraction < 1):
            raise ValueError("invalid train/validation fractions")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)

    print(f"Repo root:  {REPO_ROOT}")
    print(f"Data root:  {data_root}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {device}")
    print(f"OA scale:   {args.oa_scale}")
    print(f"Target:     {args.target_mode}")
    print(f"Split:      {args.split_mode}")

    raw_records = load_records(
        data_root,
        parse_session_filter(args.sessions),
        args.label_consistency_tolerance_deg,
    )

    print(f"Raw samples before YOLO: {len(raw_records)}")

    records, _ = preprocess_records_with_yolo(
        raw_records,
        args,
        output_dir,
    )

    sessions = ordered_session_ids(records)
    g = group_sessions(records)
    print(f"Samples used by OA: {len(records)}")
    print(f"Sessions:   {len(sessions)}")
    for sid in sessions:
        rr = g[sid]
        print(f"  {sid}: n={len(rr)} duration={rr[-1].capture_timestamp_unix-rr[0].capture_timestamp_unix:.1f}s "
              f"distance={min(r.distance_m for r in rr):.2f}..{max(r.distance_m for r in rr):.2f}m")

    write_rows(output_dir / "session_data_quality.csv", session_quality(records))
    with (output_dir / "dataset_configuration.json").open("w", encoding="utf-8") as f:
        json.dump({
            "repo_root": str(REPO_ROOT),
            "data_root": str(data_root),
            "num_raw_samples": len(raw_records),
            "num_samples_after_yolo": len(records),
            "session_ids": sessions,
            "target_source": "recomputed from raw Quest-world geometry",
            "target_definition": "SignedAngle(camera_to_robot_planar, robot_forward_planar, +Y)",
            "args": vars(args),
        }, f, indent=2, allow_nan=False)

    if args.split_mode == "leave_one_session_out":
        splits = loso_splits(records, args.only_test_session)
        results = []
        for i, split in enumerate(splits, start=1):
            print(f"\n{'#'*90}\nLOSO {i}/{len(splits)} TEST={split.info['test_session_id']} "
                  f"VAL={split.info['validation_session_id']}\n{'#'*90}")
            seed_everything(args.seed)
            fold_dir = output_dir / "leave_one_session_out_folds" / safe_name(split.info["test_session_id"])
            results.append(run_fold(split, args, device, fold_dir))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        aggregate_loso(output_dir, results, args.target_mode)
        train_final_model_on_all_data(records, args, device, DEFAULT_WEIGHTS_OUTPUT_DIR)
        return

    if args.split_mode == "session_holdout":
        if not args.validation_session or not args.test_session:
            raise ValueError("session_holdout requires --validation-session and --test-session")
        split = session_holdout(records, args.validation_session, args.test_session)
    elif args.split_mode == "temporal":
        split = temporal_split(records, args.train_fraction, args.validation_fraction, args.split_gap_seconds)
    else:
        print("WARNING: RANDOM FRAME SPLIT CAN LEAK TEMPORALLY ADJACENT FRAMES.")
        split = random_split(records, args.train_fraction, args.validation_fraction, args.seed)

    run_fold(split, args, device, output_dir)


if __name__ == "__main__":
    main()
