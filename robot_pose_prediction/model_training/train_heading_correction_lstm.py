import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SequenceConfig:
    sample_rate_hz: float = 10.0

    # Short history for CURRENT-state filtering, not future prediction.
    # 0.8 s at 10 Hz gives 9 samples including both ends.
    history_seconds: float = 0.8

    # Predict the true heading change over the most recent 0.1 s.
    target_interval_seconds: float = 0.1

    # Generate a training/evaluation anchor every 0.1 s.
    anchor_stride_seconds: float = 0.1

    # Below this speed, an instantaneous trajectory direction is unreliable.
    # Low-speed timesteps inherit the nearest valid motion direction feature.
    min_speed_for_motion_direction_mps: float = 0.02

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def history_steps(self) -> int:
        # Include both history start and current anchor.
        return int(round(self.history_seconds * self.sample_rate_hz)) + 1


# ============================================================================
# Model
# ============================================================================

class IncrementalHeadingLSTM(nn.Module):
    """
    Short-history, frame-invariant heading-update model.

    Input:
        (B, T, 11) Quest-only features over the recent history.

        The Quest trajectory is expressed in a local frame:
            origin = Quest robot position at HISTORY START
            +Z     = Quest-estimated robot heading at HISTORY START

        Features per timestep:
            0  relative_x
            1  relative_z
            2  quest_heading_x_relative_to_history_start
            3  quest_heading_z_relative_to_history_start
            4  velocity_x
            5  velocity_z
            6  speed
            7  trajectory_heading_x
            8  trajectory_heading_z
            9  trajectory_turn_rate
            10 quest_heading_yaw_rate

    Output:
        [sin(delta_yaw), cos(delta_yaw)]

        delta_yaw is the TRUE ROBOT heading change over the latest
        target_interval_seconds.

    Since the target is a heading DIFFERENCE inside the robot GT frame,
    and all input geometry is relative inside the Quest frame, there is no
    robot<->Quest global-frame transformation or alignment assumption.

    Runtime:
        estimated_heading_t =
            estimated_heading_(t-1) + predicted_delta_yaw
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 2,
        decoder_hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, decoder_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        encoded = hidden[-1]
        output = self.decoder(encoded)
        return F.normalize(output, p=2, dim=-1, eps=1e-8)

    def get_config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "decoder_hidden_size": self.decoder_hidden_size,
            "dropout": self.dropout,
        }


# ============================================================================
# Dataset
# ============================================================================

class IncrementalHeadingDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        raw_quest_targets: np.ndarray,
        target_delta_degrees: np.ndarray,
        history_turn_degrees: np.ndarray,
        input_mean: np.ndarray,
        input_std: np.ndarray,
    ) -> None:
        self.inputs = ((inputs - input_mean) / input_std).astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.raw_quest_targets = raw_quest_targets.astype(np.float32)
        self.target_delta_degrees = target_delta_degrees.astype(np.float32)
        self.history_turn_degrees = history_turn_degrees.astype(np.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.inputs[index]),
            torch.from_numpy(self.targets[index]),
            torch.from_numpy(self.raw_quest_targets[index]),
            torch.tensor(
                self.target_delta_degrees[index],
                dtype=torch.float32,
            ),
            torch.tensor(
                self.history_turn_degrees[index],
                dtype=torch.float32,
            ),
        )


# ============================================================================
# Math helpers
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_angle_radians(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_vector(angle_radians: float) -> np.ndarray:
    return np.asarray(
        [
            math.sin(angle_radians),
            math.cos(angle_radians),
        ],
        dtype=np.float32,
    )


def vector_to_angle_radians(vectors: np.ndarray) -> np.ndarray:
    """
    vectors[..., 0] = sin(theta)
    vectors[..., 1] = cos(theta)
    """
    return np.arctan2(
        vectors[..., 0],
        vectors[..., 1],
    )


def heading_vectors_to_unwrapped_yaw(
    headings: np.ndarray,
) -> np.ndarray:
    """
    Saved data convention:
        +X = right
        +Y = up
        +Z = forward

    yaw = atan2(heading_x, heading_z)
    """
    hx = headings[:, 0]
    hz = headings[:, 2]

    norm = np.sqrt(
        hx * hx + hz * hz
    )

    if np.any(norm < 1e-8):
        raise ValueError(
            "Heading contains near-zero horizontal magnitude."
        )

    yaw = np.arctan2(
        hx / norm,
        hz / norm,
    )

    return np.unwrap(yaw)


def rotate_world_vector_into_local(
    x: np.ndarray,
    z: np.ndarray,
    reference_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    c = math.cos(reference_yaw)
    s = math.sin(reference_yaw)

    local_x = x * c - z * s
    local_z = x * s + z * c

    return local_x, local_z


def world_positions_to_local(
    world_x: np.ndarray,
    world_z: np.ndarray,
    origin_x: float,
    origin_z: float,
    reference_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dx = world_x - origin_x
    dz = world_z - origin_z

    return rotate_world_vector_into_local(
        dx,
        dz,
        reference_yaw,
    )


def fill_motion_direction_features(
    velocity_x: np.ndarray,
    velocity_z: np.ndarray,
    min_speed: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Low-speed samples inherit the nearest valid trajectory direction so
    position jitter near zero speed does not create arbitrary atan2 values.
    """
    speed = np.sqrt(
        velocity_x * velocity_x
        + velocity_z * velocity_z
    )

    valid = speed >= min_speed

    if not np.any(valid):
        zeros = np.zeros_like(speed)
        return zeros, zeros, zeros

    raw_yaw = np.zeros_like(speed)

    raw_yaw[valid] = np.arctan2(
        velocity_x[valid],
        velocity_z[valid],
    )

    valid_indices = np.flatnonzero(valid)
    filled_yaw = raw_yaw.copy()

    for index in np.flatnonzero(~valid):
        nearest_index = valid_indices[
            np.argmin(
                np.abs(valid_indices - index)
            )
        ]
        filled_yaw[index] = raw_yaw[nearest_index]

    filled_yaw = np.unwrap(filled_yaw)

    return (
        np.sin(filled_yaw),
        np.cos(filled_yaw),
        filled_yaw,
    )


# ============================================================================
# Data loading
# ============================================================================

REQUIRED_KEYS = {
    "quest_time_from_window_start",
    "quest_positions",
    "quest_headings",
    "robot_ground_truth_headings",
}


def has_required_aligned_data(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return REQUIRED_KEYS.issubset(set(data.files))
    except Exception as exc:
        print(
            f"WARNING: could not inspect {path.name}: {exc}"
        )
        return False


def load_motion_window(
    path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Returns:
        time_from_window_start
        Quest positions
        Quest headings
        aligned robot-GT headings

    Quest and robot GT need only be synchronized in TIME.
    Their global coordinate frames are never aligned or compared directly.
    """
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(
            data["quest_time_from_window_start"],
            dtype=np.float64,
        )

        quest_positions = np.asarray(
            data["quest_positions"],
            dtype=np.float64,
        )

        quest_headings = np.asarray(
            data["quest_headings"],
            dtype=np.float64,
        )

        gt_headings = np.asarray(
            data["robot_ground_truth_headings"],
            dtype=np.float64,
        )

        if "alignment_warning" in data:
            warning = bool(
                np.asarray(
                    data["alignment_warning"]
                ).item()
            )

            if warning:
                if (
                    "alignment_max_interpolation_gap_seconds"
                    in data
                ):
                    max_gap = float(
                        np.asarray(
                            data[
                                "alignment_max_interpolation_gap_seconds"
                            ]
                        ).item()
                    )

                    print(
                        f"WARNING: {path.name}: "
                        f"max alignment interpolation gap "
                        f"{max_gap:.3f}s."
                    )
                else:
                    print(
                        f"WARNING: {path.name}: alignment warning."
                    )

    sample_count = len(times)

    if sample_count < 2:
        raise ValueError(
            f"{path.name}: fewer than two samples."
        )

    expected_shape = (
        sample_count,
        3,
    )

    for name, array in {
        "quest_positions": quest_positions,
        "quest_headings": quest_headings,
        "robot_ground_truth_headings": gt_headings,
    }.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{path.name}: {name} shape={array.shape}; "
                f"expected={expected_shape}."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{path.name}: {name} contains non-finite values."
            )

    if not np.all(np.isfinite(times)):
        raise ValueError(
            f"{path.name}: times contain non-finite values."
        )

    order = np.argsort(times)

    times = times[order]
    quest_positions = quest_positions[order]
    quest_headings = quest_headings[order]
    gt_headings = gt_headings[order]

    unique_times, unique_indices = np.unique(
        times,
        return_index=True,
    )

    return (
        unique_times,
        quest_positions[unique_indices],
        quest_headings[unique_indices],
        gt_headings[unique_indices],
    )


# ============================================================================
# Example generation
# ============================================================================

def build_example(
    times: np.ndarray,
    quest_positions: np.ndarray,
    quest_yaw_world: np.ndarray,
    gt_yaw_world: np.ndarray,
    anchor_time: float,
    config: SequenceConfig,
):
    """
    Input:
        short Quest history ending at anchor_time.

    Target:
        GT heading change over:
            [anchor_time - target_interval_seconds, anchor_time]

    Because it is a GT heading DIFFERENCE, arbitrary robot-map yaw cancels.

    Raw Quest baseline:
        Quest heading change over the exact same interval.
    """
    history_start_time = (
        anchor_time
        - config.history_seconds
    )

    previous_target_time = (
        anchor_time
        - config.target_interval_seconds
    )

    query_times = np.linspace(
        history_start_time,
        anchor_time,
        config.history_steps,
        dtype=np.float64,
    )

    feature_dt = float(
        query_times[1] - query_times[0]
    )

    # ------------------------------------------------------------------
    # Frame-invariant Quest-local history.
    # ------------------------------------------------------------------
    origin_x = float(
        np.interp(
            history_start_time,
            times,
            quest_positions[:, 0],
        )
    )

    origin_z = float(
        np.interp(
            history_start_time,
            times,
            quest_positions[:, 2],
        )
    )

    quest_start_yaw = float(
        np.interp(
            history_start_time,
            times,
            quest_yaw_world,
        )
    )

    history_x_world = np.interp(
        query_times,
        times,
        quest_positions[:, 0],
    )

    history_z_world = np.interp(
        query_times,
        times,
        quest_positions[:, 2],
    )

    history_quest_yaw_world = np.interp(
        query_times,
        times,
        quest_yaw_world,
    )

    history_x_rel, history_z_rel = world_positions_to_local(
        world_x=history_x_world,
        world_z=history_z_world,
        origin_x=origin_x,
        origin_z=origin_z,
        reference_yaw=quest_start_yaw,
    )

    history_heading_yaw_rel = (
        history_quest_yaw_world
        - quest_start_yaw
    )

    history_heading_x = np.sin(
        history_heading_yaw_rel
    )

    history_heading_z = np.cos(
        history_heading_yaw_rel
    )

    velocity_x = np.gradient(
        history_x_rel,
        feature_dt,
    )

    velocity_z = np.gradient(
        history_z_rel,
        feature_dt,
    )

    speed = np.sqrt(
        velocity_x * velocity_x
        + velocity_z * velocity_z
    )

    (
        trajectory_heading_x,
        trajectory_heading_z,
        trajectory_yaw,
    ) = fill_motion_direction_features(
        velocity_x,
        velocity_z,
        config.min_speed_for_motion_direction_mps,
    )

    trajectory_turn_rate = np.gradient(
        trajectory_yaw,
        feature_dt,
    )

    quest_heading_yaw_rate = np.gradient(
        history_heading_yaw_rel,
        feature_dt,
    )

    inputs = np.stack(
        [
            history_x_rel,
            history_z_rel,
            history_heading_x,
            history_heading_z,
            velocity_x,
            velocity_z,
            speed,
            trajectory_heading_x,
            trajectory_heading_z,
            trajectory_turn_rate,
            quest_heading_yaw_rate,
        ],
        axis=-1,
    )

    # ------------------------------------------------------------------
    # Incremental, frame-invariant GT target.
    # ------------------------------------------------------------------
    gt_previous_yaw = float(
        np.interp(
            previous_target_time,
            times,
            gt_yaw_world,
        )
    )

    gt_current_yaw = float(
        np.interp(
            anchor_time,
            times,
            gt_yaw_world,
        )
    )

    gt_delta_yaw = wrap_angle_radians(
        gt_current_yaw
        - gt_previous_yaw
    )

    target = angle_vector(
        gt_delta_yaw
    )

    # ------------------------------------------------------------------
    # Raw Quest incremental-heading baseline.
    # ------------------------------------------------------------------
    quest_previous_yaw = float(
        np.interp(
            previous_target_time,
            times,
            quest_yaw_world,
        )
    )

    quest_current_yaw = float(
        np.interp(
            anchor_time,
            times,
            quest_yaw_world,
        )
    )

    quest_delta_yaw = wrap_angle_radians(
        quest_current_yaw
        - quest_previous_yaw
    )

    raw_quest_target = angle_vector(
        quest_delta_yaw
    )

    # ------------------------------------------------------------------
    # Diagnostics.
    # ------------------------------------------------------------------
    gt_history_start_yaw = float(
        np.interp(
            history_start_time,
            times,
            gt_yaw_world,
        )
    )

    history_turn_degrees = abs(
        wrap_angle_radians(
            gt_current_yaw
            - gt_history_start_yaw
        )
    ) * 180.0 / math.pi

    target_delta_degrees = abs(
        gt_delta_yaw
    ) * 180.0 / math.pi

    return (
        inputs.astype(np.float32),
        target,
        raw_quest_target,
        float(target_delta_degrees),
        float(history_turn_degrees),
    )


def extract_examples_from_file(
    path: Path,
    config: SequenceConfig,
):
    (
        times,
        quest_positions,
        quest_headings,
        gt_headings,
    ) = load_motion_window(path)

    quest_yaw_world = heading_vectors_to_unwrapped_yaw(
        quest_headings
    )

    gt_yaw_world = heading_vectors_to_unwrapped_yaw(
        gt_headings
    )

    first_anchor = (
        times[0]
        + max(
            config.history_seconds,
            config.target_interval_seconds,
        )
    )

    last_anchor = times[-1]

    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    raw_targets: List[np.ndarray] = []
    target_deltas: List[float] = []
    history_turns: List[float] = []

    if last_anchor < first_anchor:
        return (
            inputs,
            targets,
            raw_targets,
            target_deltas,
            history_turns,
        )

    anchor_time = first_anchor

    while anchor_time <= last_anchor + 1e-9:
        (
            x,
            y,
            raw_y,
            target_delta,
            history_turn,
        ) = build_example(
            times=times,
            quest_positions=quest_positions,
            quest_yaw_world=quest_yaw_world,
            gt_yaw_world=gt_yaw_world,
            anchor_time=anchor_time,
            config=config,
        )

        if (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(y))
            and np.all(np.isfinite(raw_y))
        ):
            inputs.append(x)
            targets.append(y)
            raw_targets.append(raw_y)
            target_deltas.append(target_delta)
            history_turns.append(history_turn)

        anchor_time += (
            config.anchor_stride_seconds
        )

    return (
        inputs,
        targets,
        raw_targets,
        target_deltas,
        history_turns,
    )


def build_dataset_arrays(
    files: Sequence[Path],
    config: SequenceConfig,
):
    all_inputs = []
    all_targets = []
    all_raw_targets = []
    all_target_deltas = []
    all_history_turns = []

    for path in files:
        try:
            (
                inputs,
                targets,
                raw_targets,
                target_deltas,
                history_turns,
            ) = extract_examples_from_file(
                path,
                config,
            )
        except ValueError as exc:
            print(
                f"Skipping {path.name}: {exc}"
            )
            continue

        all_inputs.extend(inputs)
        all_targets.extend(targets)
        all_raw_targets.extend(raw_targets)
        all_target_deltas.extend(target_deltas)
        all_history_turns.extend(history_turns)

    if not all_inputs:
        raise RuntimeError(
            "No valid incremental-heading examples could be generated."
        )

    return (
        np.stack(all_inputs),
        np.stack(all_targets),
        np.stack(all_raw_targets),
        np.asarray(
            all_target_deltas,
            dtype=np.float32,
        ),
        np.asarray(
            all_history_turns,
            dtype=np.float32,
        ),
    )


# ============================================================================
# File-level splitting
# ============================================================================

def split_files(
    files: Sequence[Path],
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
):
    files = list(files)

    if len(files) < 3:
        raise RuntimeError(
            "At least 3 aligned .npz files are required."
        )

    rng = random.Random(seed)
    rng.shuffle(files)

    count = len(files)

    train_count = max(
        1,
        int(round(count * train_fraction)),
    )

    validation_count = max(
        1,
        int(round(count * validation_fraction)),
    )

    if train_count + validation_count >= count:
        train_count = count - 2
        validation_count = 1

    train_files = files[:train_count]

    validation_files = files[
        train_count:
        train_count + validation_count
    ]

    test_files = files[
        train_count + validation_count:
    ]

    return (
        train_files,
        validation_files,
        test_files,
    )


# ============================================================================
# Loss / metrics
# ============================================================================

def weighted_cosine_heading_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_delta_degrees: torch.Tensor,
    turn_loss_gain: float,
    turn_reference_degrees: float,
) -> torch.Tensor:
    """
    Mildly upweight genuine turning updates so the network cannot achieve a
    deceptively good loss by predicting zero heading change everywhere.

    weight =
        1 + turn_loss_gain *
            clamp(|GT delta| / turn_reference_degrees, 0, 1)
    """
    prediction = F.normalize(
        prediction,
        dim=-1,
        eps=1e-8,
    )

    target = F.normalize(
        target,
        dim=-1,
        eps=1e-8,
    )

    cosine = (
        prediction * target
    ).sum(dim=-1)

    per_example_loss = (
        1.0 - cosine
    )

    if turn_loss_gain > 0.0:
        turn_fraction = torch.clamp(
            target_delta_degrees
            / max(
                turn_reference_degrees,
                1e-6,
            ),
            0.0,
            1.0,
        )

        weights = (
            1.0
            + turn_loss_gain
            * turn_fraction
        )

        per_example_loss = (
            per_example_loss
            * weights
        )

    return per_example_loss.mean()


def angular_error_degrees(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = F.normalize(
        prediction,
        dim=-1,
        eps=1e-8,
    )

    target = F.normalize(
        target,
        dim=-1,
        eps=1e-8,
    )

    cosine = (
        prediction * target
    ).sum(dim=-1)

    cosine = torch.clamp(
        cosine,
        -1.0,
        1.0,
    )

    return (
        torch.acos(cosine)
        * 180.0
        / math.pi
    )


def summarize_errors(
    errors: np.ndarray,
) -> dict:
    return {
        "mean_error_degrees": float(
            np.mean(errors)
        ),
        "median_error_degrees": float(
            np.median(errors)
        ),
        "p90_error_degrees": float(
            np.percentile(
                errors,
                90.0,
            )
        ),
        "max_error_degrees": float(
            np.max(errors)
        ),
    }


def run_epoch(
    model: IncrementalHeadingLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool,
    turn_loss_gain: float,
    turn_reference_degrees: float,
) -> float:
    model.train(training)

    loss_sum = 0.0
    example_count = 0

    for batch in loader:
        (
            inputs,
            targets,
            _,
            target_delta_degrees,
            _,
        ) = batch

        inputs = inputs.to(device)
        targets = targets.to(device)
        target_delta_degrees = (
            target_delta_degrees.to(device)
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):
            predictions = model(inputs)

            loss = weighted_cosine_heading_loss(
                prediction=predictions,
                target=targets,
                target_delta_degrees=(
                    target_delta_degrees
                ),
                turn_loss_gain=turn_loss_gain,
                turn_reference_degrees=(
                    turn_reference_degrees
                ),
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        batch_size = inputs.shape[0]

        loss_sum += (
            float(loss.item())
            * batch_size
        )

        example_count += batch_size

    return (
        loss_sum / example_count
    )


@torch.no_grad()
def evaluate_one_step(
    model: IncrementalHeadingLSTM,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    learned_errors = []
    raw_errors = []
    zero_errors = []
    target_deltas = []
    history_turns = []

    for batch in loader:
        (
            inputs,
            targets,
            raw_quest_targets,
            batch_target_deltas,
            batch_history_turns,
        ) = batch

        inputs = inputs.to(device)
        targets = targets.to(device)
        raw_quest_targets = (
            raw_quest_targets.to(device)
        )

        predictions = model(inputs)

        learned_errors.append(
            angular_error_degrees(
                predictions,
                targets,
            ).cpu()
        )

        raw_errors.append(
            angular_error_degrees(
                raw_quest_targets,
                targets,
            ).cpu()
        )

        # No-change baseline:
        # sin(0)=0, cos(0)=1
        zero_prediction = torch.zeros_like(
            targets
        )
        zero_prediction[:, 1] = 1.0

        zero_errors.append(
            angular_error_degrees(
                zero_prediction,
                targets,
            ).cpu()
        )

        target_deltas.append(
            batch_target_deltas.cpu()
        )

        history_turns.append(
            batch_history_turns.cpu()
        )

    learned_errors = torch.cat(
        learned_errors
    ).numpy()

    raw_errors = torch.cat(
        raw_errors
    ).numpy()

    zero_errors = torch.cat(
        zero_errors
    ).numpy()

    target_deltas = torch.cat(
        target_deltas
    ).numpy()

    history_turns = torch.cat(
        history_turns
    ).numpy()

    result = {
        "example_count": int(
            len(learned_errors)
        ),
        "lstm": summarize_errors(
            learned_errors
        ),
        "raw_quest_delta": summarize_errors(
            raw_errors
        ),
        "zero_delta_baseline": summarize_errors(
            zero_errors
        ),
    }

    result[
        "by_true_target_delta"
    ] = bucket_metrics(
        buckets=[
            (
                "near_zero_lt_1deg",
                0.0,
                1.0,
            ),
            (
                "small_turn_1_to_3deg",
                1.0,
                3.0,
            ),
            (
                "turn_3_to_6deg",
                3.0,
                6.0,
            ),
            (
                "large_step_ge_6deg",
                6.0,
                float("inf"),
            ),
        ],
        values=target_deltas,
        learned_errors=learned_errors,
        raw_errors=raw_errors,
        zero_errors=zero_errors,
        unit="deg",
    )

    result[
        "by_true_heading_change_over_history"
    ] = bucket_metrics(
        buckets=[
            (
                "stable_lt_5deg",
                0.0,
                5.0,
            ),
            (
                "changing_5_to_15deg",
                5.0,
                15.0,
            ),
            (
                "turning_15_to_30deg",
                15.0,
                30.0,
            ),
            (
                "strong_turn_ge_30deg",
                30.0,
                float("inf"),
            ),
        ],
        values=history_turns,
        learned_errors=learned_errors,
        raw_errors=raw_errors,
        zero_errors=zero_errors,
        unit="deg",
    )

    return result


def bucket_metrics(
    buckets,
    values: np.ndarray,
    learned_errors: np.ndarray,
    raw_errors: np.ndarray,
    zero_errors: np.ndarray,
    unit: str,
) -> dict:
    output = {}

    for (
        name,
        lower,
        upper,
    ) in buckets:
        if math.isinf(upper):
            mask = (
                values >= lower
            )
        else:
            mask = (
                (values >= lower)
                & (values < upper)
            )

        count = int(
            np.sum(mask)
        )

        if count == 0:
            output[name] = {
                "count": 0
            }
            continue

        output[name] = {
            "count": count,
            "mean_bucket_value": float(
                np.mean(
                    values[mask]
                )
            ),
            "bucket_unit": unit,
            "lstm_mean_error_degrees": float(
                np.mean(
                    learned_errors[mask]
                )
            ),
            "raw_quest_mean_error_degrees": float(
                np.mean(
                    raw_errors[mask]
                )
            ),
            "zero_delta_mean_error_degrees": float(
                np.mean(
                    zero_errors[mask]
                )
            ),
            "lstm_p90_error_degrees": float(
                np.percentile(
                    learned_errors[mask],
                    90.0,
                )
            ),
            "raw_quest_p90_error_degrees": float(
                np.percentile(
                    raw_errors[mask],
                    90.0,
                )
            ),
            "zero_delta_p90_error_degrees": float(
                np.percentile(
                    zero_errors[mask],
                    90.0,
                )
            ),
        }

    return output


# ============================================================================
# Sequential rollout evaluation
# ============================================================================

@torch.no_grad()
def evaluate_rollout_file(
    model: IncrementalHeadingLSTM,
    path: Path,
    config: SequenceConfig,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Roll the incremental estimator through one complete test recording.

    All headings are evaluated RELATIVE to the first evaluated GT heading.
    Therefore the unknown startup rotation between Quest and robot frames is
    irrelevant.

    The rollout starts with zero relative heading error. This isolates drift
    and responsiveness of the incremental update model from initialization.
    """
    (
        times,
        quest_positions,
        quest_headings,
        gt_headings,
    ) = load_motion_window(path)

    quest_yaw = heading_vectors_to_unwrapped_yaw(
        quest_headings
    )

    gt_yaw = heading_vectors_to_unwrapped_yaw(
        gt_headings
    )

    first_anchor = (
        times[0]
        + max(
            config.history_seconds,
            config.target_interval_seconds,
        )
    )

    last_anchor = times[-1]

    if last_anchor < first_anchor:
        return {
            "file": path.name,
            "example_count": 0,
        }

    anchor_times = []
    model_deltas = []
    raw_deltas = []
    gt_deltas = []

    anchor_time = first_anchor

    while anchor_time <= last_anchor + 1e-9:
        (
            x,
            y,
            raw_y,
            _,
            _,
        ) = build_example(
            times=times,
            quest_positions=quest_positions,
            quest_yaw_world=quest_yaw,
            gt_yaw_world=gt_yaw,
            anchor_time=anchor_time,
            config=config,
        )

        normalized = (
            (
                x
                - input_mean.squeeze(
                    axis=0
                )
            )
            / input_std.squeeze(
                axis=0
            )
        ).astype(np.float32)

        input_tensor = torch.from_numpy(
            normalized
        ).unsqueeze(0).to(device)

        prediction = model(
            input_tensor
        )[0].cpu().numpy()

        model_deltas.append(
            float(
                math.atan2(
                    prediction[0],
                    prediction[1],
                )
            )
        )

        raw_deltas.append(
            float(
                math.atan2(
                    raw_y[0],
                    raw_y[1],
                )
            )
        )

        gt_deltas.append(
            float(
                math.atan2(
                    y[0],
                    y[1],
                )
            )
        )

        anchor_times.append(
            anchor_time
        )

        anchor_time += (
            config.anchor_stride_seconds
        )

    if not anchor_times:
        return {
            "file": path.name,
            "example_count": 0,
        }

    model_deltas = np.asarray(
        model_deltas,
        dtype=np.float64,
    )

    raw_deltas = np.asarray(
        raw_deltas,
        dtype=np.float64,
    )

    gt_deltas = np.asarray(
        gt_deltas,
        dtype=np.float64,
    )

    # Because target interval and anchor stride default to the same value,
    # these deltas tile the trajectory without overlap.
    model_relative_yaw = np.cumsum(
        model_deltas
    )

    raw_relative_yaw = np.cumsum(
        raw_deltas
    )

    gt_relative_yaw = np.cumsum(
        gt_deltas
    )

    zero_relative_yaw = np.zeros_like(
        gt_relative_yaw
    )

    def relative_heading_errors(
        estimate: np.ndarray,
        truth: np.ndarray,
    ) -> np.ndarray:
        wrapped = np.asarray(
            [
                wrap_angle_radians(
                    float(a - b)
                )
                for a, b in zip(
                    estimate,
                    truth,
                )
            ],
            dtype=np.float64,
        )

        return (
            np.abs(wrapped)
            * 180.0
            / math.pi
        )

    model_errors = relative_heading_errors(
        model_relative_yaw,
        gt_relative_yaw,
    )

    raw_errors = relative_heading_errors(
        raw_relative_yaw,
        gt_relative_yaw,
    )

    zero_errors = relative_heading_errors(
        zero_relative_yaw,
        gt_relative_yaw,
    )

    return {
        "file": path.name,
        "example_count": int(
            len(anchor_times)
        ),
        "duration_seconds": float(
            anchor_times[-1]
            - anchor_times[0]
        ) if len(anchor_times) > 1 else 0.0,
        "lstm": summarize_errors(
            model_errors
        ),
        "raw_quest_delta": summarize_errors(
            raw_errors
        ),
        "zero_delta_baseline": summarize_errors(
            zero_errors
        ),
        "final_lstm_error_degrees": float(
            model_errors[-1]
        ),
        "final_raw_quest_error_degrees": float(
            raw_errors[-1]
        ),
        "final_zero_delta_error_degrees": float(
            zero_errors[-1]
        ),
    }


def aggregate_rollout_metrics(
    file_results: List[dict],
) -> dict:
    valid = [
        result
        for result in file_results
        if result.get(
            "example_count",
            0,
        ) > 0
    ]

    if not valid:
        return {
            "file_count": 0
        }

    def aggregate_method(
        key: str,
    ) -> dict:
        return {
            "mean_of_file_mean_errors_degrees": float(
                np.mean(
                    [
                        result[key][
                            "mean_error_degrees"
                        ]
                        for result in valid
                    ]
                )
            ),
            "median_of_file_mean_errors_degrees": float(
                np.median(
                    [
                        result[key][
                            "mean_error_degrees"
                        ]
                        for result in valid
                    ]
                )
            ),
            "mean_of_file_p90_errors_degrees": float(
                np.mean(
                    [
                        result[key][
                            "p90_error_degrees"
                        ]
                        for result in valid
                    ]
                )
            ),
        }

    return {
        "file_count": len(valid),
        "lstm": aggregate_method(
            "lstm"
        ),
        "raw_quest_delta": aggregate_method(
            "raw_quest_delta"
        ),
        "zero_delta_baseline": aggregate_method(
            "zero_delta_baseline"
        ),
    }


# ============================================================================
# Reporting
# ============================================================================

def print_one_step_metrics(
    metrics: dict,
) -> None:
    learned = metrics["lstm"]
    raw = metrics["raw_quest_delta"]
    zero = metrics["zero_delta_baseline"]

    print()
    print("One-step heading-update metrics")
    print("===============================")
    print(
        f"{'Metric':<24}"
        f"{'LSTM':>14}"
        f"{'Raw Quest':>14}"
        f"{'Zero change':>16}"
    )
    print("-" * 68)

    for label, key in [
        (
            "Mean error",
            "mean_error_degrees",
        ),
        (
            "Median error",
            "median_error_degrees",
        ),
        (
            "P90 error",
            "p90_error_degrees",
        ),
        (
            "Max error",
            "max_error_degrees",
        ),
    ]:
        print(
            f"{label:<24}"
            f"{learned[key]:>11.3f} deg"
            f"{raw[key]:>11.3f} deg"
            f"{zero[key]:>13.3f} deg"
        )


def print_bucket_section(
    title: str,
    buckets: dict,
) -> None:
    print()
    print(title)
    print("=" * len(title))

    for (
        name,
        metrics,
    ) in buckets.items():
        print(name)

        if metrics["count"] == 0:
            print("  No examples.")
            continue

        print(
            f"  examples: {metrics['count']}"
        )

        print(
            "  mean bucket value: "
            f"{metrics['mean_bucket_value']:.3f} "
            f"{metrics['bucket_unit']}"
        )

        print(
            "  mean error: "
            f"LSTM={metrics['lstm_mean_error_degrees']:.3f} deg | "
            f"Raw={metrics['raw_quest_mean_error_degrees']:.3f} deg | "
            f"Zero={metrics['zero_delta_mean_error_degrees']:.3f} deg"
        )

        print(
            "  P90 error:  "
            f"LSTM={metrics['lstm_p90_error_degrees']:.3f} deg | "
            f"Raw={metrics['raw_quest_p90_error_degrees']:.3f} deg | "
            f"Zero={metrics['zero_delta_p90_error_degrees']:.3f} deg"
        )


def print_rollout_metrics(
    aggregate: dict,
) -> None:
    print()
    print("Sequential rollout metrics")
    print("==========================")

    if aggregate.get(
        "file_count",
        0,
    ) == 0:
        print(
            "No valid test files for rollout evaluation."
        )
        return

    print(
        f"Test files rolled out: "
        f"{aggregate['file_count']}"
    )
    print()
    print(
        f"{'Metric':<38}"
        f"{'LSTM':>12}"
        f"{'Raw Quest':>12}"
        f"{'Zero':>12}"
    )
    print("-" * 74)

    rows = [
        (
            "Mean of per-file mean errors",
            "mean_of_file_mean_errors_degrees",
        ),
        (
            "Median of per-file mean errors",
            "median_of_file_mean_errors_degrees",
        ),
        (
            "Mean of per-file P90 errors",
            "mean_of_file_p90_errors_degrees",
        ),
    ]

    for label, key in rows:
        print(
            f"{label:<38}"
            f"{aggregate['lstm'][key]:>9.2f} deg"
            f"{aggregate['raw_quest_delta'][key]:>9.2f} deg"
            f"{aggregate['zero_delta_baseline'][key]:>9.2f} deg"
        )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    robot_pose_prediction_dir = (
        Path(__file__).resolve().parent.parent
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            robot_pose_prediction_dir
            / "data"
        ),
    )

    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=(
            robot_pose_prediction_dir
            / "weights"
        ),
    )

    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=(
            robot_pose_prediction_dir
            / "eval"
        ),
    )

    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--history-seconds",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--target-interval-seconds",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--anchor-stride-seconds",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--min-speed-for-motion-direction-mps",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--decoder-hidden-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=18,
    )

    # Turning examples are often rarer than straight-motion examples.
    # Default gain=1 means a sufficiently large target update gets 2x loss.
    parser.add_argument(
        "--turn-loss-gain",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--turn-reference-degrees",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.history_seconds <= 0.0:
        raise ValueError(
            "--history-seconds must be > 0."
        )

    if args.target_interval_seconds <= 0.0:
        raise ValueError(
            "--target-interval-seconds must be > 0."
        )

    if args.anchor_stride_seconds <= 0.0:
        raise ValueError(
            "--anchor-stride-seconds must be > 0."
        )

    # Sequential rollout assumes non-overlapping incremental targets.
    if not math.isclose(
        args.target_interval_seconds,
        args.anchor_stride_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        print(
            "WARNING: target interval and anchor stride differ. "
            "One-step training/evaluation is valid, but rollout cumulative "
            "deltas will not correspond to a simple non-overlapping integration."
        )

    set_seed(
        args.seed
    )

    config = SequenceConfig(
        sample_rate_hz=(
            args.sample_rate_hz
        ),
        history_seconds=(
            args.history_seconds
        ),
        target_interval_seconds=(
            args.target_interval_seconds
        ),
        anchor_stride_seconds=(
            args.anchor_stride_seconds
        ),
        min_speed_for_motion_direction_mps=(
            args.min_speed_for_motion_direction_mps
        ),
    )

    data_dir = (
        args.data_dir.resolve()
    )

    weights_dir = (
        args.weights_dir.resolve()
    )

    eval_dir = (
        args.eval_dir.resolve()
    )

    weights_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_files = sorted(
        data_dir.glob(
            "*.npz"
        )
    )

    if not all_files:
        raise RuntimeError(
            f"No .npz files found in {data_dir}"
        )

    files = [
        path
        for path in all_files
        if has_required_aligned_data(
            path
        )
    ]

    ignored_count = (
        len(all_files)
        - len(files)
    )

    if not files:
        raise RuntimeError(
            "No aligned Quest + robot-GT files found."
        )

    print()
    print("Incremental current-heading LSTM")
    print("================================")
    print(
        f"Aligned files: {len(files)}"
    )

    if ignored_count:
        print(
            f"Ignored incompatible files: {ignored_count}"
        )

    print(
        f"Input history: "
        f"{config.history_seconds:.3f}s "
        f"({config.history_steps} resampled points)"
    )

    print(
        f"Target: true heading change over latest "
        f"{config.target_interval_seconds:.3f}s"
    )

    print(
        f"Anchor stride: "
        f"{config.anchor_stride_seconds:.3f}s"
    )

    print(
        "No robot<->Quest global transformation is used."
    )
    print()

    (
        train_files,
        validation_files,
        test_files,
    ) = split_files(
        files=files,
        seed=args.seed,
    )

    print(
        f"Train files:      {len(train_files)}"
    )
    print(
        f"Validation files: {len(validation_files)}"
    )
    print(
        f"Test files:       {len(test_files)}"
    )
    print()

    (
        train_inputs,
        train_targets,
        train_raw_targets,
        train_target_deltas,
        train_history_turns,
    ) = build_dataset_arrays(
        train_files,
        config,
    )

    (
        validation_inputs,
        validation_targets,
        validation_raw_targets,
        validation_target_deltas,
        validation_history_turns,
    ) = build_dataset_arrays(
        validation_files,
        config,
    )

    (
        test_inputs,
        test_targets,
        test_raw_targets,
        test_target_deltas,
        test_history_turns,
    ) = build_dataset_arrays(
        test_files,
        config,
    )

    print(
        f"Train examples:      {len(train_inputs)}"
    )
    print(
        f"Validation examples: {len(validation_inputs)}"
    )
    print(
        f"Test examples:       {len(test_inputs)}"
    )
    print()

    print(
        "Training target magnitude distribution:"
    )
    print(
        f"  mean:   "
        f"{np.mean(train_target_deltas):.3f} deg"
    )
    print(
        f"  median: "
        f"{np.median(train_target_deltas):.3f} deg"
    )
    print(
        f"  P90:    "
        f"{np.percentile(train_target_deltas, 90.0):.3f} deg"
    )
    print(
        f"  max:    "
        f"{np.max(train_target_deltas):.3f} deg"
    )
    print()

    # ------------------------------------------------------------------
    # Train-data-only feature normalization.
    # ------------------------------------------------------------------
    input_mean = train_inputs.mean(
        axis=(0, 1),
        keepdims=True,
    ).astype(np.float32)

    input_std = train_inputs.std(
        axis=(0, 1),
        keepdims=True,
    ).astype(np.float32)

    input_std = np.maximum(
        input_std,
        1e-6,
    )

    def make_dataset(
        inputs,
        targets,
        raw_targets,
        target_deltas,
        history_turns,
    ):
        return IncrementalHeadingDataset(
            inputs=inputs,
            targets=targets,
            raw_quest_targets=raw_targets,
            target_delta_degrees=target_deltas,
            history_turn_degrees=history_turns,
            input_mean=input_mean,
            input_std=input_std,
        )

    train_dataset = make_dataset(
        train_inputs,
        train_targets,
        train_raw_targets,
        train_target_deltas,
        train_history_turns,
    )

    validation_dataset = make_dataset(
        validation_inputs,
        validation_targets,
        validation_raw_targets,
        validation_target_deltas,
        validation_history_turns,
    )

    test_dataset = make_dataset(
        test_inputs,
        test_targets,
        test_raw_targets,
        test_target_deltas,
        test_history_turns,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )
    print()

    model = IncrementalHeadingLSTM(
        input_size=11,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        decoder_hidden_size=(
            args.decoder_hidden_size
        ),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=6,
        )
    )

    checkpoint_path = (
        weights_dir
        / "incremental_heading_lstm_best.pt"
    )

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0
    training_history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            training=True,
            turn_loss_gain=args.turn_loss_gain,
            turn_reference_degrees=(
                args.turn_reference_degrees
            ),
        )

        validation_loss = run_epoch(
            model=model,
            loader=validation_loader,
            optimizer=optimizer,
            device=device,
            training=False,
            turn_loss_gain=args.turn_loss_gain,
            turn_reference_degrees=(
                args.turn_reference_degrees
            ),
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.6f} | "
            f"val={validation_loss:.6f} | "
            f"lr={current_lr:.2e}"
        )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": (
                    validation_loss
                ),
                "learning_rate": current_lr,
            }
        )

        if (
            validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_type": (
                        "incremental_current_heading_lstm"
                    ),
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "model_config": (
                        model.get_config()
                    ),
                    "sequence_config": (
                        asdict(config)
                    ),
                    "input_feature_names": [
                        "relative_x",
                        "relative_z",
                        "quest_heading_x_relative_to_history_start",
                        "quest_heading_z_relative_to_history_start",
                        "velocity_x",
                        "velocity_z",
                        "speed",
                        "trajectory_heading_x",
                        "trajectory_heading_z",
                        "trajectory_turn_rate",
                        "quest_heading_yaw_rate",
                    ],
                    "input_mean": (
                        input_mean.squeeze(
                            axis=(0, 1)
                        )
                    ),
                    "input_std": (
                        input_std.squeeze(
                            axis=(0, 1)
                        )
                    ),
                    "target_definition": (
                        "delta_yaw = GT_yaw(anchor) - "
                        "GT_yaw(anchor - target_interval_seconds)"
                    ),
                    "runtime_application": (
                        "estimated_heading_now = "
                        "estimated_heading_previous + predicted_delta_yaw"
                    ),
                    "turn_loss_gain": (
                        args.turn_loss_gain
                    ),
                    "turn_reference_degrees": (
                        args.turn_reference_degrees
                    ),
                    "best_validation_loss": (
                        best_validation_loss
                    ),
                    "epoch": epoch,
                    "train_files": [
                        p.name
                        for p in train_files
                    ],
                    "validation_files": [
                        p.name
                        for p in validation_files
                    ],
                    "test_files": [
                        p.name
                        for p in test_files
                    ],
                },
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print()
            print(
                "Early stopping: no validation improvement "
                f"for {args.patience} epochs."
            )
            break

    # ------------------------------------------------------------------
    # Reload best checkpoint.
    # ------------------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ------------------------------------------------------------------
    # One-step evaluation.
    # ------------------------------------------------------------------
    one_step_metrics = (
        evaluate_one_step(
            model=model,
            loader=test_loader,
            device=device,
        )
    )

    print_one_step_metrics(
        one_step_metrics
    )

    print_bucket_section(
        "Metrics by true heading change over target interval",
        one_step_metrics[
            "by_true_target_delta"
        ],
    )

    print_bucket_section(
        "Metrics by true heading change over input history",
        one_step_metrics[
            "by_true_heading_change_over_history"
        ],
    )

    # ------------------------------------------------------------------
    # Sequential rollout evaluation.
    # ------------------------------------------------------------------
    rollout_file_results = []

    for path in test_files:
        result = evaluate_rollout_file(
            model=model,
            path=path,
            config=config,
            input_mean=input_mean,
            input_std=input_std,
            device=device,
        )

        rollout_file_results.append(
            result
        )

    rollout_aggregate = (
        aggregate_rollout_metrics(
            rollout_file_results
        )
    )

    print_rollout_metrics(
        rollout_aggregate
    )

    # ------------------------------------------------------------------
    # Save full evaluation.
    # ------------------------------------------------------------------
    eval_path = (
        eval_dir
        / "incremental_heading_lstm_eval.json"
    )

    summary = {
        "model_type": (
            "incremental_current_heading_lstm"
        ),
        "sequence_config": (
            asdict(config)
        ),
        "model_config": (
            checkpoint[
                "model_config"
            ]
        ),
        "target_definition": (
            checkpoint[
                "target_definition"
            ]
        ),
        "runtime_application": (
            checkpoint[
                "runtime_application"
            ]
        ),
        "turn_loss_gain": (
            checkpoint[
                "turn_loss_gain"
            ]
        ),
        "turn_reference_degrees": (
            checkpoint[
                "turn_reference_degrees"
            ]
        ),
        "best_validation_loss": (
            checkpoint[
                "best_validation_loss"
            ]
        ),
        "best_epoch": (
            checkpoint[
                "epoch"
            ]
        ),
        "one_step_test_metrics": (
            one_step_metrics
        ),
        "rollout_aggregate_metrics": (
            rollout_aggregate
        ),
        "rollout_file_metrics": (
            rollout_file_results
        ),
        "training_history": (
            training_history
        ),
        "train_files": [
            p.name
            for p in train_files
        ],
        "validation_files": [
            p.name
            for p in validation_files
        ],
        "test_files": [
            p.name
            for p in test_files
        ],
    }

    with open(
        eval_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        f"Best checkpoint: {checkpoint_path}"
    )
    print(
        f"Evaluation:     {eval_path}"
    )


if __name__ == "__main__":
    main()