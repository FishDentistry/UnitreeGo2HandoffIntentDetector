import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from robot_pose_prediction.src.lstm_model import RobotPoseMDNLSTM


# ---------------------------------------------------------------------------
# Configuration / data loading
# ---------------------------------------------------------------------------


@dataclass
class SequenceConfig:
    sample_rate_hz: float = 10.0
    history_seconds: float = 3.0
    prediction_seconds: float = 4.0
    anchor_stride_seconds: float = 0.5

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def history_steps(self) -> int:
        return int(round(self.history_seconds * self.sample_rate_hz))

    @property
    def prediction_steps(self) -> int:
        return int(round(self.prediction_seconds * self.sample_rate_hz))


def _first_existing_key(
    data: np.lib.npyio.NpzFile,
    candidates: Sequence[str],
) -> str:
    for key in candidates:
        if key in data:
            return key
    raise KeyError("None of these keys were present: " + ", ".join(candidates))


def load_robot_motion_window(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        time_key = _first_existing_key(
            data,
            [
                "robot_time_from_window_start",
                "robot_timestamps",
                "robot_ground_truth_timestamps",
                "time_from_window_start",
                "timestamps",
            ],
        )
        position_key = _first_existing_key(
            data,
            [
                "robot_positions",
                "robot_ground_truth_positions",
                "positions",
            ],
        )
        heading_key = _first_existing_key(
            data,
            [
                "robot_headings",
                "robot_ground_truth_headings",
                "headings",
            ],
        )

        times = np.asarray(data[time_key], dtype=np.float64)
        positions = np.asarray(data[position_key], dtype=np.float64)
        headings = np.asarray(data[heading_key], dtype=np.float64)

    if times.ndim != 1:
        raise ValueError(f"{path.name}: timestamps must have shape (N,).")

    n = len(times)
    if n < 2:
        raise ValueError(f"{path.name}: fewer than two samples.")
    if positions.shape != (n, 3):
        raise ValueError(
            f"{path.name}: positions shape {positions.shape}, expected {(n, 3)}."
        )
    if headings.shape != (n, 3):
        raise ValueError(
            f"{path.name}: headings shape {headings.shape}, expected {(n, 3)}."
        )

    if not np.all(np.isfinite(times)):
        raise ValueError(f"{path.name}: non-finite timestamps.")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{path.name}: non-finite positions.")
    if not np.all(np.isfinite(headings)):
        raise ValueError(f"{path.name}: non-finite headings.")

    order = np.argsort(times)
    times = times[order]
    positions = positions[order]
    headings = headings[order]

    times, unique_indices = np.unique(times, return_index=True)
    positions = positions[unique_indices]
    headings = headings[unique_indices]

    if len(times) < 2:
        raise ValueError(f"{path.name}: fewer than two unique timestamps.")

    times = times - times[0]
    return times, positions, headings


def heading_vectors_to_unwrapped_yaw(headings: np.ndarray) -> np.ndarray:
    """
    Same converted-coordinate convention as the current trainer:
        yaw = atan2(+X, +Z)
    """
    hx = headings[:, 0]
    hz = headings[:, 2]
    norm = np.sqrt(hx * hx + hz * hz)

    if np.any(norm < 1e-6):
        raise ValueError("At least one heading has near-zero horizontal norm.")

    return np.unwrap(np.arctan2(hx / norm, hz / norm))


def world_positions_to_robot_frame(
    world_x: np.ndarray,
    world_z: np.ndarray,
    anchor_x: float,
    anchor_z: float,
    anchor_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dx = world_x - anchor_x
    dz = world_z - anchor_z

    s = math.sin(anchor_yaw)
    c = math.cos(anchor_yaw)

    relative_x = dx * c - dz * s
    relative_z = dx * s + dz * c

    return relative_x, relative_z


def split_files(
    files: Sequence[Path],
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> Tuple[List[Path], List[Path], List[Path]]:
    files = list(files)

    if len(files) < 3:
        raise RuntimeError("At least 3 .npz files are required.")

    rng = random.Random(seed)
    rng.shuffle(files)

    count = len(files)
    train_count = max(1, int(round(count * train_fraction)))
    validation_count = max(1, int(round(count * validation_fraction)))

    if train_count + validation_count >= count:
        train_count = count - 2
        validation_count = 1

    return (
        files[:train_count],
        files[train_count:train_count + validation_count],
        files[train_count + validation_count:],
    )


# ---------------------------------------------------------------------------
# Example construction + diagnostic metadata
# ---------------------------------------------------------------------------


def build_example_and_metadata(
    times: np.ndarray,
    positions: np.ndarray,
    yaw_world: np.ndarray,
    anchor_time: float,
    config: SequenceConfig,
    recent_window_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    dt = config.dt

    history_relative_times = np.linspace(
        -config.history_seconds + dt,
        0.0,
        config.history_steps,
        dtype=np.float64,
    )

    future_relative_times = np.linspace(
        dt,
        config.prediction_seconds,
        config.prediction_steps,
        dtype=np.float64,
    )

    history_query = anchor_time + history_relative_times
    future_query = anchor_time + future_relative_times

    anchor_x = float(np.interp(anchor_time, times, positions[:, 0]))
    anchor_z = float(np.interp(anchor_time, times, positions[:, 2]))
    anchor_yaw = float(np.interp(anchor_time, times, yaw_world))

    history_x = np.interp(history_query, times, positions[:, 0])
    history_z = np.interp(history_query, times, positions[:, 2])

    future_x = np.interp(future_query, times, positions[:, 0])
    future_z = np.interp(future_query, times, positions[:, 2])

    history_yaw = np.interp(history_query, times, yaw_world)
    future_yaw = np.interp(future_query, times, yaw_world)

    history_relative_yaw = history_yaw - anchor_yaw

    history_x_rel, history_z_rel = world_positions_to_robot_frame(
        history_x,
        history_z,
        anchor_x,
        anchor_z,
        anchor_yaw,
    )

    future_x_rel, future_z_rel = world_positions_to_robot_frame(
        future_x,
        future_z,
        anchor_x,
        anchor_z,
        anchor_yaw,
    )

    velocity_x = np.gradient(history_x_rel, dt)
    velocity_z = np.gradient(history_z_rel, dt)

    relative_heading_sin = np.sin(history_relative_yaw)
    relative_heading_cos = np.cos(history_relative_yaw)
    yaw_rate = np.gradient(history_relative_yaw, dt)

    inputs = np.stack(
        [
            history_x_rel,
            history_z_rel,
            velocity_x,
            velocity_z,
            relative_heading_sin,
            relative_heading_cos,
            yaw_rate,
        ],
        axis=-1,
    ).astype(np.float32)

    targets = np.stack(
        [future_x_rel, future_z_rel],
        axis=-1,
    ).astype(np.float32)

    # ---- Diagnostic quantities ----
    recent_steps = max(
        1,
        int(round(recent_window_seconds * config.sample_rate_hz)),
    )

    recent_steps = min(recent_steps, len(velocity_x))

    history_speed = np.sqrt(
        velocity_x * velocity_x
        + velocity_z * velocity_z
    )

    recent_past_speed = float(
        history_speed[-recent_steps:].mean()
    )

    recent_abs_yaw_rate = float(
        np.abs(yaw_rate[-recent_steps:]).mean()
    )

    # Future speeds from the ground-truth future path.
    future_with_anchor = np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float64),
            targets.astype(np.float64),
        ],
        axis=0,
    )

    future_velocity = np.diff(
        future_with_anchor,
        axis=0,
    ) / dt

    future_speed = np.linalg.norm(
        future_velocity,
        axis=1,
    )

    future_early_speed = float(
        future_speed[:recent_steps].mean()
    )

    future_late_speed = float(
        future_speed[-recent_steps:].mean()
    )

    future_mean_speed = float(
        future_speed.mean()
    )

    # Use maximum absolute heading change over the prediction horizon rather
    # than only endpoint yaw, so a turn-and-return still counts as a turn.
    future_relative_yaw = (
        future_yaw
        - anchor_yaw
    )

    max_future_turn_radians = float(
        np.max(
            np.abs(
                future_relative_yaw
            )
        )
    )

    final_future_turn_radians = float(
        abs(
            future_relative_yaw[-1]
        )
    )

    path_length = float(
        np.linalg.norm(
            np.diff(
                future_with_anchor,
                axis=0,
            ),
            axis=1,
        ).sum()
    )

    endpoint_displacement = float(
        np.linalg.norm(
            targets[-1]
        )
    )

    metadata = {
        "recent_past_speed": recent_past_speed,
        "future_early_speed": future_early_speed,
        "future_late_speed": future_late_speed,
        "future_mean_speed": future_mean_speed,
        "recent_abs_yaw_rate": recent_abs_yaw_rate,
        "max_future_turn_radians": max_future_turn_radians,
        "final_future_turn_radians": final_future_turn_radians,
        "future_path_length": path_length,
        "future_endpoint_displacement": endpoint_displacement,
    }

    return (
        inputs,
        targets,
        metadata,
    )


def extract_examples_from_file(
    path: Path,
    config: SequenceConfig,
    recent_window_seconds: float,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[Dict[str, object]],
]:
    times, positions, headings = load_robot_motion_window(path)
    yaw_world = heading_vectors_to_unwrapped_yaw(headings)

    first_anchor = times[0] + config.history_seconds
    last_anchor = times[-1] - config.prediction_seconds

    if last_anchor < first_anchor:
        return [], [], []

    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []

    anchor_time = first_anchor

    while anchor_time <= last_anchor + 1e-9:
        x, y, m = build_example_and_metadata(
            times=times,
            positions=positions,
            yaw_world=yaw_world,
            anchor_time=anchor_time,
            config=config,
            recent_window_seconds=recent_window_seconds,
        )

        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            inputs.append(x)
            targets.append(y)

            m["file"] = path.name
            m["anchor_time"] = float(anchor_time)

            metadata.append(m)

        anchor_time += config.anchor_stride_seconds

    return inputs, targets, metadata


# ---------------------------------------------------------------------------
# Motion labels
# ---------------------------------------------------------------------------


def build_motion_masks(
    metadata: List[Dict[str, object]],
    stop_speed: float,
    moving_speed: float,
    straight_degrees: float,
    sharp_turn_degrees: float,
    active_turn_rate_degrees_per_second: float,
) -> Dict[str, np.ndarray]:
    recent_speed = np.asarray(
        [m["recent_past_speed"] for m in metadata],
        dtype=np.float64,
    )

    early_speed = np.asarray(
        [m["future_early_speed"] for m in metadata],
        dtype=np.float64,
    )

    late_speed = np.asarray(
        [m["future_late_speed"] for m in metadata],
        dtype=np.float64,
    )

    mean_future_speed = np.asarray(
        [m["future_mean_speed"] for m in metadata],
        dtype=np.float64,
    )

    turn_deg = np.degrees(
        np.asarray(
            [m["max_future_turn_radians"] for m in metadata],
            dtype=np.float64,
        )
    )

    yaw_rate_deg = np.degrees(
        np.asarray(
            [m["recent_abs_yaw_rate"] for m in metadata],
            dtype=np.float64,
        )
    )

    stopped_before = recent_speed <= stop_speed
    moving_before = recent_speed >= moving_speed

    stopped_late = late_speed <= stop_speed
    moving_future = mean_future_speed >= moving_speed

    masks = {
        # Transition/state slices.
        "stationary": (
            stopped_before
            & (mean_future_speed <= stop_speed)
        ),
        "stop_to_move": (
            stopped_before
            & moving_future
        ),
        "move_to_stop": (
            moving_before
            & stopped_late
        ),
        "moving_throughout": (
            moving_before
            & (early_speed >= moving_speed)
            & (late_speed >= moving_speed)
        ),

        # Future path geometry.
        f"straight_<_{straight_degrees:g}deg": (
            turn_deg < straight_degrees
        ),
        f"gradual_turn_{straight_degrees:g}_to_{sharp_turn_degrees:g}deg": (
            (turn_deg >= straight_degrees)
            & (turn_deg < sharp_turn_degrees)
        ),
        f"sharp_turn_>=_{sharp_turn_degrees:g}deg": (
            turn_deg >= sharp_turn_degrees
        ),

        # What the robot was already doing during the recent observed history.
        f"already_turning_>=_{active_turn_rate_degrees_per_second:g}deg_s": (
            yaw_rate_deg
            >= active_turn_rate_degrees_per_second
        ),
    }

    return masks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_slices(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> None:
    errors = np.linalg.norm(
        predictions - targets,
        axis=-1,
    )

    overall_ade = float(
        errors.mean()
    )

    overall_fde = float(
        errors[:, -1].mean()
    )

    print()
    print("=== OVERALL ===")
    print(
        f"N={len(targets)} | "
        f"ADE={overall_ade:.3f} m | "
        f"FDE={overall_fde:.3f} m"
    )

    print()
    print("=== MOTION-TYPE BREAKDOWN ===")

    rows = []

    for name, mask in masks.items():
        count = int(
            mask.sum()
        )

        if count == 0:
            rows.append(
                (
                    name,
                    0,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
            )
            continue

        slice_errors = errors[
            mask
        ]

        ade = float(
            slice_errors.mean()
        )

        fde = float(
            slice_errors[:, -1].mean()
        )

        relative_fde = (
            fde / overall_fde
            if overall_fde > 0.0
            else float("nan")
        )

        rows.append(
            (
                name,
                count,
                ade,
                fde,
                relative_fde,
            )
        )

    # Sort by FDE descending so the hardest slices are immediately obvious.
    rows.sort(
        key=lambda row: (
            -row[3]
            if math.isfinite(row[3])
            else float("inf")
        )
    )

    header = (
        f"{'Motion slice':42s} "
        f"{'N':>7s} "
        f"{'ADE (m)':>10s} "
        f"{'FDE (m)':>10s} "
        f"{'FDE/overall':>12s}"
    )

    print(header)
    print("-" * len(header))

    for name, count, ade, fde, relative_fde in rows:
        if count == 0:
            print(
                f"{name:42s} "
                f"{count:7d} "
                f"{'--':>10s} "
                f"{'--':>10s} "
                f"{'--':>12s}"
            )
        else:
            print(
                f"{name:42s} "
                f"{count:7d} "
                f"{ade:10.3f} "
                f"{fde:10.3f} "
                f"{relative_fde:12.2f}x"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Break a trained robot trajectory predictor's test error down "
            "by motion type."
        )
    )

    project_dir = (
        Path(__file__).resolve().parent.parent
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            project_dir
            / "data"
            / "robot_ground_truth"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            project_dir
            / "weights"
            / "robot_pose_per_horizon_mdn_heading_best.pt"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Must match the seed used for training so the same test files "
            "are selected."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--recent-window-seconds",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--stop-speed",
        type=float,
        default=0.08,
        help="m/s; below this is treated as stopped.",
    )

    parser.add_argument(
        "--moving-speed",
        type=float,
        default=0.15,
        help="m/s; above this is treated as definitely moving.",
    )

    parser.add_argument(
        "--straight-degrees",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--sharp-turn-degrees",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--active-turn-rate-degrees-per-second",
        type=float,
        default=15.0,
    )

    args = parser.parse_args()

    checkpoint_path = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    data_dir = (
        args.data_dir
        .expanduser()
        .resolve()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    sequence_config_dict = checkpoint[
        "sequence_config"
    ]

    config = SequenceConfig(
        **sequence_config_dict
    )

    model = RobotPoseMDNLSTM(
        **checkpoint["model_config"]
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    input_mean = np.asarray(
        checkpoint[
            "input_mean"
        ],
        dtype=np.float32,
    )

    input_std = np.asarray(
        checkpoint[
            "input_std"
        ],
        dtype=np.float32,
    )

    all_files = sorted(
        data_dir.glob("*.npz")
    )

    compatible_files = []

    for path in all_files:
        try:
            load_robot_motion_window(
                path
            )
            compatible_files.append(
                path
            )
        except Exception as exc:
            print(
                f"Ignoring {path.name}: {exc}"
            )

    _, _, test_files = split_files(
        compatible_files,
        args.seed,
    )

    print()
    print("=== Motion-type error analysis ===")
    print(
        f"Checkpoint: {checkpoint_path}"
    )
    print(
        f"Test files: {len(test_files)}"
    )
    print(
        "Test file names:"
    )

    for path in test_files:
        print(
            f"  {path.name}"
        )

    all_inputs: List[
        np.ndarray
    ] = []

    all_targets: List[
        np.ndarray
    ] = []

    all_metadata: List[
        Dict[str, object]
    ] = []

    for path in test_files:
        inputs, targets, metadata = (
            extract_examples_from_file(
                path=path,
                config=config,
                recent_window_seconds=(
                    args.recent_window_seconds
                ),
            )
        )

        all_inputs.extend(
            inputs
        )

        all_targets.extend(
            targets
        )

        all_metadata.extend(
            metadata
        )

    if not all_inputs:
        raise RuntimeError(
            "No valid test examples were generated."
        )

    inputs = np.stack(
        all_inputs
    ).astype(
        np.float32
    )

    targets = np.stack(
        all_targets
    ).astype(
        np.float32
    )

    normalized_inputs = (
        (inputs - input_mean)
        / input_std
    ).astype(
        np.float32
    )

    predictions = []

    with torch.no_grad():
        for start in range(
            0,
            len(normalized_inputs),
            args.batch_size,
        ):
            end = min(
                start + args.batch_size,
                len(normalized_inputs),
            )

            batch = torch.from_numpy(
                normalized_inputs[
                    start:end
                ]
            ).to(
                device
            )

            output = model(
                batch
            )

            expected = (
                model.expected_position(
                    output
                )
            )

            predictions.append(
                expected.cpu().numpy()
            )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    masks = build_motion_masks(
        metadata=all_metadata,
        stop_speed=args.stop_speed,
        moving_speed=args.moving_speed,
        straight_degrees=args.straight_degrees,
        sharp_turn_degrees=(
            args.sharp_turn_degrees
        ),
        active_turn_rate_degrees_per_second=(
            args.active_turn_rate_degrees_per_second
        ),
    )

    evaluate_slices(
        predictions=predictions,
        targets=targets,
        masks=masks,
    )


if __name__ == "__main__":
    main()
