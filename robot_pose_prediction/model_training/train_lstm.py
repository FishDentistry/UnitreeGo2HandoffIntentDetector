import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from robot_pose_prediction.src.lstm_model import RobotPoseLSTM


# ---------------------------------------------------------------------------
# Configuration
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
        return int(
            round(
                self.history_seconds
                * self.sample_rate_hz
            )
        )

    @property
    def prediction_steps(self) -> int:
        return int(
            round(
                self.prediction_seconds
                * self.sample_rate_hz
            )
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RobotTrajectoryDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        input_mean: np.ndarray,
        input_std: np.ndarray,
    ) -> None:
        self.inputs = (
            (inputs - input_mean)
            / input_std
        ).astype(np.float32)

        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.inputs[index]),
            torch.from_numpy(self.targets[index]),
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_motion_window(
    path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load one temporally aligned motion window.

    The saved file must contain both:

        Quest-estimated robot motion:
            quest_time_from_window_start
            quest_positions
            quest_headings

        Robot ground truth aligned to the Quest timestamps:
            robot_ground_truth_positions
            robot_ground_truth_headings

    Returns:
        times:
            (N,)
            Quest-relative time in seconds.

        quest_positions:
            (N, 3)
            Noisy observer-side robot positions.

        quest_headings:
            (N, 3)
            Noisy/delayed observer-side robot forward vectors.

        ground_truth_positions:
            (N, 3)
            Robot ground-truth positions aligned to the same samples.

        ground_truth_headings:
            (N, 3)
            Robot ground-truth forward vectors aligned to the same samples.
    """

    required_keys = [
        "quest_time_from_window_start",
        "quest_positions",
        "quest_headings",
        "robot_ground_truth_positions",
        "robot_ground_truth_headings",
    ]

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        missing_keys = [
            key
            for key in required_keys
            if key not in data
        ]

        if missing_keys:
            raise ValueError(
                f"{path.name}: missing aligned-data keys: "
                f"{', '.join(missing_keys)}. "
                "This training script requires data collected "
                "with the Quest + robot-ground-truth alignment server."
            )

        times = np.asarray(
            data[
                "quest_time_from_window_start"
            ],
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

        ground_truth_positions = np.asarray(
            data[
                "robot_ground_truth_positions"
            ],
            dtype=np.float64,
        )

        ground_truth_headings = np.asarray(
            data[
                "robot_ground_truth_headings"
            ],
            dtype=np.float64,
        )

        # The alignment server stores both absolute timestamp arrays.
        # They should already match exactly because the robot ground truth
        # was interpolated onto the Quest timestamps. Verify this when
        # those keys are present.
        if (
            "quest_timestamps" in data
            and "robot_ground_truth_timestamps"
            in data
        ):
            quest_timestamps = np.asarray(
                data["quest_timestamps"],
                dtype=np.float64,
            )

            ground_truth_timestamps = np.asarray(
                data[
                    "robot_ground_truth_timestamps"
                ],
                dtype=np.float64,
            )

            if (
                quest_timestamps.shape
                != ground_truth_timestamps.shape
            ):
                raise ValueError(
                    f"{path.name}: Quest and ground-truth "
                    "timestamp arrays have different shapes."
                )

            if not np.allclose(
                quest_timestamps,
                ground_truth_timestamps,
                rtol=0.0,
                atol=1e-6,
            ):
                max_difference = float(
                    np.max(
                        np.abs(
                            quest_timestamps
                            - ground_truth_timestamps
                        )
                    )
                )

                raise ValueError(
                    f"{path.name}: aligned Quest and robot "
                    "ground-truth timestamps do not match. "
                    f"Maximum difference: "
                    f"{max_difference:.6f} seconds."
                )

        if (
            "alignment_warning" in data
            and bool(
                np.asarray(
                    data["alignment_warning"]
                ).item()
            )
        ):
            max_gap = None

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

            if max_gap is None:
                print(
                    f"WARNING: {path.name} has an "
                    "alignment warning."
                )
            else:
                print(
                    f"WARNING: {path.name} has an "
                    "alignment warning; maximum robot "
                    "interpolation gap was "
                    f"{max_gap:.3f} seconds."
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

    arrays_to_check = {
        "quest_positions": quest_positions,
        "quest_headings": quest_headings,
        "robot_ground_truth_positions": (
            ground_truth_positions
        ),
        "robot_ground_truth_headings": (
            ground_truth_headings
        ),
    }

    for name, array in arrays_to_check.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{path.name}: invalid {name} shape "
                f"{array.shape}; expected "
                f"{expected_shape}."
            )

        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                f"{path.name}: {name} contains "
                "non-finite values."
            )

    if not np.all(
        np.isfinite(times)
    ):
        raise ValueError(
            f"{path.name}: timestamps contain "
            "non-finite values."
        )

    # Sort once and apply the same ordering to BOTH streams so their
    # temporal correspondence is preserved.
    order = np.argsort(times)

    times = times[order]
    quest_positions = (
        quest_positions[order]
    )
    quest_headings = (
        quest_headings[order]
    )
    ground_truth_positions = (
        ground_truth_positions[order]
    )
    ground_truth_headings = (
        ground_truth_headings[order]
    )

    # Remove duplicate relative timestamps while preserving alignment.
    unique_times, unique_indices = np.unique(
        times,
        return_index=True,
    )

    times = unique_times

    quest_positions = (
        quest_positions[
            unique_indices
        ]
    )

    quest_headings = (
        quest_headings[
            unique_indices
        ]
    )

    ground_truth_positions = (
        ground_truth_positions[
            unique_indices
        ]
    )

    ground_truth_headings = (
        ground_truth_headings[
            unique_indices
        ]
    )

    return (
        times,
        quest_positions,
        quest_headings,
        ground_truth_positions,
        ground_truth_headings,
    )

def heading_vectors_to_unwrapped_yaw(
    headings: np.ndarray,
) -> np.ndarray:
    """
    Convert world-space heading vectors into continuous yaw.

    Assumes Unity-style coordinates:

        +X = right
        +Y = up
        +Z = forward

    yaw = atan2(heading_x, heading_z)
    """

    hx = headings[:, 0]
    hz = headings[:, 2]

    horizontal_norm = np.sqrt(
        hx * hx + hz * hz
    )

    if np.any(horizontal_norm < 1e-6):
        raise ValueError(
            "At least one heading has near-zero "
            "horizontal magnitude."
        )

    hx = hx / horizontal_norm
    hz = hz / horizontal_norm

    yaw = np.arctan2(
        hx,
        hz,
    )

    return np.unwrap(yaw)


def world_positions_to_robot_frame(
    world_x: np.ndarray,
    world_z: np.ndarray,
    anchor_x: float,
    anchor_z: float,
    anchor_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform world-space X/Z positions into the robot frame
    at the prediction anchor.

    At the anchor:
        position = (0, 0)
        heading = +Z
    """

    dx = world_x - anchor_x
    dz = world_z - anchor_z

    sin_yaw = math.sin(anchor_yaw)
    cos_yaw = math.cos(anchor_yaw)

    relative_x = (
        dx * cos_yaw
        - dz * sin_yaw
    )

    relative_z = (
        dx * sin_yaw
        + dz * cos_yaw
    )

    return (
        relative_x,
        relative_z,
    )


def build_example(
    times: np.ndarray,
    quest_positions: np.ndarray,
    quest_yaw_world: np.ndarray,
    ground_truth_positions: np.ndarray,
    ground_truth_yaw_world: np.ndarray,
    anchor_time: float,
    config: SequenceConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one supervised example centered around a prediction anchor.

    INPUT:
        The previous `history_seconds` of QUEST-ESTIMATED robot motion.

        The Quest stream is transformed into a coordinate system in which
        the Quest-estimated robot pose at anchor_time is:

            position = (0, 0)
            heading  = +Z

    TARGET:
        The next `prediction_seconds` of ROBOT GROUND-TRUTH motion.

        The ground-truth stream is independently transformed into a
        coordinate system in which the TRUE robot pose at the SAME
        anchor_time is:

            position = (0, 0)
            heading  = +Z

    Why transform the streams independently?

        The Quest world frame and robot map frame do not need to share the
        same global origin or absolute orientation. The model learns:

            noisy observer-side relative motion
                        ->
            true future robot-relative motion

        rather than trying to learn an arbitrary transform between two
        unrelated global coordinate frames.
    """

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

    history_query_times = (
        anchor_time
        + history_relative_times
    )

    future_query_times = (
        anchor_time
        + future_relative_times
    )

    # ------------------------------------------------------------------
    # Quest-estimated pose at the prediction anchor.
    # This defines the coordinate frame of the LSTM input.
    # ------------------------------------------------------------------

    quest_anchor_x = float(
        np.interp(
            anchor_time,
            times,
            quest_positions[:, 0],
        )
    )

    quest_anchor_z = float(
        np.interp(
            anchor_time,
            times,
            quest_positions[:, 2],
        )
    )

    quest_anchor_yaw = float(
        np.interp(
            anchor_time,
            times,
            quest_yaw_world,
        )
    )

    # ------------------------------------------------------------------
    # True robot pose at the SAME prediction anchor.
    # This independently defines the coordinate frame of the target.
    # ------------------------------------------------------------------

    ground_truth_anchor_x = float(
        np.interp(
            anchor_time,
            times,
            ground_truth_positions[:, 0],
        )
    )

    ground_truth_anchor_z = float(
        np.interp(
            anchor_time,
            times,
            ground_truth_positions[:, 2],
        )
    )

    ground_truth_anchor_yaw = float(
        np.interp(
            anchor_time,
            times,
            ground_truth_yaw_world,
        )
    )

    # ------------------------------------------------------------------
    # Resample Quest HISTORY.
    # ------------------------------------------------------------------

    history_quest_x_world = np.interp(
        history_query_times,
        times,
        quest_positions[:, 0],
    )

    history_quest_z_world = np.interp(
        history_query_times,
        times,
        quest_positions[:, 2],
    )

    history_quest_yaw_world = np.interp(
        history_query_times,
        times,
        quest_yaw_world,
    )

    # ------------------------------------------------------------------
    # Resample robot-ground-truth FUTURE.
    # ------------------------------------------------------------------

    future_ground_truth_x_world = np.interp(
        future_query_times,
        times,
        ground_truth_positions[:, 0],
    )

    future_ground_truth_z_world = np.interp(
        future_query_times,
        times,
        ground_truth_positions[:, 2],
    )

    future_ground_truth_yaw_world = np.interp(
        future_query_times,
        times,
        ground_truth_yaw_world,
    )

    # ------------------------------------------------------------------
    # Quest history -> Quest-anchor-relative coordinates.
    # ------------------------------------------------------------------

    (
        history_x_rel,
        history_z_rel,
    ) = world_positions_to_robot_frame(
        world_x=history_quest_x_world,
        world_z=history_quest_z_world,
        anchor_x=quest_anchor_x,
        anchor_z=quest_anchor_z,
        anchor_yaw=quest_anchor_yaw,
    )

    history_yaw_rel = (
        history_quest_yaw_world
        - quest_anchor_yaw
    )

    history_heading_x = np.sin(
        history_yaw_rel
    )

    history_heading_z = np.cos(
        history_yaw_rel
    )

    # ------------------------------------------------------------------
    # Ground-truth future -> TRUE-anchor-relative coordinates.
    # ------------------------------------------------------------------

    (
        future_x_rel,
        future_z_rel,
    ) = world_positions_to_robot_frame(
        world_x=(
            future_ground_truth_x_world
        ),
        world_z=(
            future_ground_truth_z_world
        ),
        anchor_x=ground_truth_anchor_x,
        anchor_z=ground_truth_anchor_z,
        anchor_yaw=ground_truth_anchor_yaw,
    )

    future_yaw_rel = (
        future_ground_truth_yaw_world
        - ground_truth_anchor_yaw
    )

    future_heading_x = np.sin(
        future_yaw_rel
    )

    future_heading_z = np.cos(
        future_yaw_rel
    )

    # ------------------------------------------------------------------
    # Derive observed motion features ONLY from the Quest history.
    # ------------------------------------------------------------------

    velocity_x = np.gradient(
        history_x_rel,
        dt,
    )

    velocity_z = np.gradient(
        history_z_rel,
        dt,
    )

    yaw_rate = np.gradient(
        history_yaw_rel,
        dt,
    )

    # ------------------------------------------------------------------
    # LSTM input:
    #
    # Quest-estimated:
    # [x, z, heading_x, heading_z, vx, vz, yaw_rate]
    # ------------------------------------------------------------------

    inputs = np.stack(
        [
            history_x_rel,
            history_z_rel,
            history_heading_x,
            history_heading_z,
            velocity_x,
            velocity_z,
            yaw_rate,
        ],
        axis=-1,
    )

    # ------------------------------------------------------------------
    # Supervision target:
    #
    # Robot ground truth:
    # [x, z, heading_x, heading_z]
    # ------------------------------------------------------------------

    targets = np.stack(
        [
            future_x_rel,
            future_z_rel,
            future_heading_x,
            future_heading_z,
        ],
        axis=-1,
    )

    return (
        inputs.astype(np.float32),
        targets.astype(np.float32),
    )

def extract_examples_from_file(
    path: Path,
    config: SequenceConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    (
        times,
        quest_positions,
        quest_headings,
        ground_truth_positions,
        ground_truth_headings,
    ) = load_motion_window(
        path
    )

    quest_yaw_world = (
        heading_vectors_to_unwrapped_yaw(
            quest_headings
        )
    )

    ground_truth_yaw_world = (
        heading_vectors_to_unwrapped_yaw(
            ground_truth_headings
        )
    )

    first_anchor = (
        times[0]
        + config.history_seconds
    )

    last_anchor = (
        times[-1]
        - config.prediction_seconds
    )

    if last_anchor < first_anchor:
        duration = float(
            times[-1] - times[0]
        )

        required_duration = (
            config.history_seconds
            + config.prediction_seconds
        )

        print(
            f"Skipping {path.name}: "
            f"duration={duration:.3f}s, "
            f"required>={required_duration:.3f}s."
        )

        return [], []

    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    anchor_time = first_anchor

    while (
        anchor_time
        <= last_anchor + 1e-9
    ):
        try:
            x, y = build_example(
                times=times,
                quest_positions=(
                    quest_positions
                ),
                quest_yaw_world=(
                    quest_yaw_world
                ),
                ground_truth_positions=(
                    ground_truth_positions
                ),
                ground_truth_yaw_world=(
                    ground_truth_yaw_world
                ),
                anchor_time=anchor_time,
                config=config,
            )

            if (
                np.all(np.isfinite(x))
                and np.all(np.isfinite(y))
            ):
                inputs.append(x)
                targets.append(y)

        except ValueError as exc:
            print(
                f"Skipping example from "
                f"{path.name}: {exc}"
            )

        anchor_time += (
            config.anchor_stride_seconds
        )

    return inputs, targets

def build_dataset_arrays(
    files: Sequence[Path],
    config: SequenceConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    all_inputs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for path in files:
        inputs, targets = (
            extract_examples_from_file(
                path,
                config,
            )
        )

        all_inputs.extend(inputs)
        all_targets.extend(targets)

    if not all_inputs:
        raise RuntimeError(
            "No valid trajectory examples could be "
            "generated from these files."
        )

    return (
        np.stack(all_inputs),
        np.stack(all_targets),
    )


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def split_files(
    files: Sequence[Path],
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> Tuple[
    List[Path],
    List[Path],
    List[Path],
]:
    files = list(files)

    if len(files) < 3:
        raise RuntimeError(
            "At least 3 .npz motion-window files are "
            "required for train/validation/test splitting."
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
        int(
            round(
                count * validation_fraction
            )
        ),
    )

    if (
        train_count
        + validation_count
        >= count
    ):
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


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    heading_weight: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    position_loss = F.smooth_l1_loss(
        prediction[..., :2],
        target[..., :2],
    )

    predicted_heading = F.normalize(
        prediction[..., 2:4],
        dim=-1,
        eps=1e-8,
    )

    target_heading = F.normalize(
        target[..., 2:4],
        dim=-1,
        eps=1e-8,
    )

    cosine_similarity = (
        predicted_heading
        * target_heading
    ).sum(dim=-1)

    heading_loss = (
        1.0 - cosine_similarity
    ).mean()

    total_loss = (
        position_loss
        + heading_weight * heading_loss
    )

    return (
        total_loss,
        position_loss,
        heading_loss,
    )


@torch.no_grad()
def compute_metrics(
    model: RobotPoseLSTM,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    all_position_errors = []
    all_heading_errors = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)

        position_error = torch.linalg.vector_norm(
            predictions[..., :2]
            - targets[..., :2],
            dim=-1,
        )

        predicted_heading = F.normalize(
            predictions[..., 2:4],
            dim=-1,
            eps=1e-8,
        )

        target_heading = F.normalize(
            targets[..., 2:4],
            dim=-1,
            eps=1e-8,
        )

        cosine = (
            predicted_heading
            * target_heading
        ).sum(dim=-1)

        cosine = torch.clamp(
            cosine,
            -1.0,
            1.0,
        )

        heading_error_radians = torch.acos(
            cosine
        )

        all_position_errors.append(
            position_error.cpu()
        )

        all_heading_errors.append(
            heading_error_radians.cpu()
        )

    position_errors = torch.cat(
        all_position_errors,
        dim=0,
    )

    heading_errors = torch.cat(
        all_heading_errors,
        dim=0,
    )

    heading_errors_degrees = (
        heading_errors
        * 180.0
        / math.pi
    )

    return {
        "ade_meters": float(
            position_errors.mean()
        ),
        "fde_meters": float(
            position_errors[:, -1].mean()
        ),
        "mean_heading_error_degrees": float(
            heading_errors_degrees.mean()
        ),
        "final_heading_error_degrees": float(
            heading_errors_degrees[:, -1].mean()
        ),
    }


@torch.no_grad()
def compute_horizon_metrics(
    model: RobotPoseLSTM,
    loader: DataLoader,
    device: torch.device,
    config: SequenceConfig,
) -> dict:
    model.eval()

    predictions_all = []
    targets_all = []

    for inputs, targets in loader:
        inputs = inputs.to(device)

        predictions = model(inputs)

        predictions_all.append(
            predictions.cpu()
        )

        targets_all.append(
            targets
        )

    predictions = torch.cat(
        predictions_all,
        dim=0,
    )

    targets = torch.cat(
        targets_all,
        dim=0,
    )

    results = {}

    whole_second_horizons = range(
        1,
        int(
            math.floor(
                config.prediction_seconds
            )
        ) + 1,
    )

    for seconds in whole_second_horizons:
        index = int(
            round(
                seconds
                * config.sample_rate_hz
            )
        ) - 1

        if index >= config.prediction_steps:
            continue

        position_error = torch.linalg.vector_norm(
            predictions[:, index, :2]
            - targets[:, index, :2],
            dim=-1,
        )

        predicted_heading = F.normalize(
            predictions[:, index, 2:4],
            dim=-1,
            eps=1e-8,
        )

        target_heading = F.normalize(
            targets[:, index, 2:4],
            dim=-1,
            eps=1e-8,
        )

        cosine = (
            predicted_heading
            * target_heading
        ).sum(dim=-1)

        cosine = torch.clamp(
            cosine,
            -1.0,
            1.0,
        )

        heading_error = (
            torch.acos(cosine)
            * 180.0
            / math.pi
        )

        results[f"{seconds}s"] = {
            "position_error_meters": float(
                position_error.mean()
            ),
            "heading_error_degrees": float(
                heading_error.mean()
            ),
        }

    return results


# ---------------------------------------------------------------------------
# Kinematic baseline
# ---------------------------------------------------------------------------

def predict_constant_turn_rate(
    inputs: np.ndarray,
    config: SequenceConfig,
    estimation_seconds: float = 0.5,
) -> np.ndarray:
    """
    Predict future robot poses with a simple kinematic baseline.

    The input examples must be the UNNORMALIZED QUEST-derived relative inputs
    produced by build_example():

        [x, z, heading_x, heading_z, velocity_x, velocity_z, yaw_rate]

    The baseline assumes that, over the prediction horizon:

        1. Translational velocity remains constant in the robot's local frame.
        2. Yaw rate remains constant.

    Velocity and yaw rate are estimated by averaging the final
    `estimation_seconds` of observed history. Averaging a short recent window
    makes the baseline less sensitive to frame-to-frame noise than using only
    the final sample.

    At the prediction anchor, the robot-relative frame is defined so that:

        position = (0, 0)
        heading = +Z

    Returns:
        predictions:
            shape (N, prediction_steps, 4)

            channels:
                0: future_relative_x
                1: future_relative_z
                2: future_relative_heading_x
                3: future_relative_heading_z
    """

    if inputs.ndim != 3 or inputs.shape[-1] != 7:
        raise ValueError(
            "Expected inputs with shape "
            "(N, history_steps, 7)."
        )

    if estimation_seconds <= 0.0:
        raise ValueError(
            "estimation_seconds must be greater than 0."
        )

    history_steps = inputs.shape[1]

    estimation_steps = max(
        1,
        int(
            round(
                estimation_seconds
                * config.sample_rate_hz
            )
        ),
    )

    estimation_steps = min(
        estimation_steps,
        history_steps,
    )

    recent = inputs[
        :,
        -estimation_steps:,
        :,
    ]

    # At t=0 the anchor frame and robot body frame coincide, so the recent
    # robot-relative velocity components are a useful estimate of the current
    # body-frame velocity.
    velocity_x = recent[..., 4].mean(
        axis=1
    )

    velocity_z = recent[..., 5].mean(
        axis=1
    )

    yaw_rate = recent[..., 6].mean(
        axis=1
    )

    future_times = (
        np.arange(
            1,
            config.prediction_steps + 1,
            dtype=np.float64,
        )
        * config.dt
    )

    batch_size = inputs.shape[0]
    prediction_steps = config.prediction_steps

    future_x = np.zeros(
        (batch_size, prediction_steps),
        dtype=np.float64,
    )

    future_z = np.zeros(
        (batch_size, prediction_steps),
        dtype=np.float64,
    )

    future_yaw = (
        yaw_rate[:, None]
        * future_times[None, :]
    )

    # Constant-turn-rate motion:
    #
    # The local body-frame velocity is rotated continuously as the robot
    # changes yaw. For very small yaw rates, use the straight-line limit to
    # avoid division by values near zero.
    turning_mask = (
        np.abs(yaw_rate)
        >= 1e-4
    )

    straight_mask = ~turning_mask

    if np.any(turning_mask):
        omega = yaw_rate[
            turning_mask
        ][:, None]

        vx = velocity_x[
            turning_mask
        ][:, None]

        vz = velocity_z[
            turning_mask
        ][:, None]

        angle = (
            omega
            * future_times[None, :]
        )

        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)

        future_x[turning_mask] = (
            vx * sin_angle / omega
            + vz * (1.0 - cos_angle) / omega
        )

        future_z[turning_mask] = (
            vx * (cos_angle - 1.0) / omega
            + vz * sin_angle / omega
        )

    if np.any(straight_mask):
        future_x[straight_mask] = (
            velocity_x[
                straight_mask
            ][:, None]
            * future_times[None, :]
        )

        future_z[straight_mask] = (
            velocity_z[
                straight_mask
            ][:, None]
            * future_times[None, :]
        )

    future_heading_x = np.sin(
        future_yaw
    )

    future_heading_z = np.cos(
        future_yaw
    )

    predictions = np.stack(
        [
            future_x,
            future_z,
            future_heading_x,
            future_heading_z,
        ],
        axis=-1,
    )

    return predictions.astype(
        np.float32
    )


def compute_array_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """
    Compute the same aggregate metrics used for the LSTM, but directly from
    NumPy arrays. This is used by the kinematic baseline, which sees only Quest observations.
    """

    position_errors = np.linalg.norm(
        predictions[..., :2]
        - targets[..., :2],
        axis=-1,
    )

    predicted_heading = (
        predictions[..., 2:4]
    )

    target_heading = (
        targets[..., 2:4]
    )

    predicted_heading = (
        predicted_heading
        / np.maximum(
            np.linalg.norm(
                predicted_heading,
                axis=-1,
                keepdims=True,
            ),
            1e-8,
        )
    )

    target_heading = (
        target_heading
        / np.maximum(
            np.linalg.norm(
                target_heading,
                axis=-1,
                keepdims=True,
            ),
            1e-8,
        )
    )

    cosine = np.sum(
        predicted_heading
        * target_heading,
        axis=-1,
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    heading_errors_degrees = np.degrees(
        np.arccos(cosine)
    )

    return {
        "ade_meters": float(
            position_errors.mean()
        ),
        "fde_meters": float(
            position_errors[:, -1].mean()
        ),
        "mean_heading_error_degrees": float(
            heading_errors_degrees.mean()
        ),
        "final_heading_error_degrees": float(
            heading_errors_degrees[:, -1].mean()
        ),
    }


def compute_array_horizon_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    config: SequenceConfig,
) -> dict:
    """
    Compute position and heading error at each whole-second horizon.
    """

    results = {}

    whole_second_horizons = range(
        1,
        int(
            math.floor(
                config.prediction_seconds
            )
        ) + 1,
    )

    for seconds in whole_second_horizons:
        index = int(
            round(
                seconds
                * config.sample_rate_hz
            )
        ) - 1

        if index >= config.prediction_steps:
            continue

        position_error = np.linalg.norm(
            predictions[:, index, :2]
            - targets[:, index, :2],
            axis=-1,
        )

        predicted_heading = (
            predictions[:, index, 2:4]
        )

        target_heading = (
            targets[:, index, 2:4]
        )

        predicted_heading = (
            predicted_heading
            / np.maximum(
                np.linalg.norm(
                    predicted_heading,
                    axis=-1,
                    keepdims=True,
                ),
                1e-8,
            )
        )

        target_heading = (
            target_heading
            / np.maximum(
                np.linalg.norm(
                    target_heading,
                    axis=-1,
                    keepdims=True,
                ),
                1e-8,
            )
        )

        cosine = np.sum(
            predicted_heading
            * target_heading,
            axis=-1,
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0,
        )

        heading_error = np.degrees(
            np.arccos(cosine)
        )

        results[f"{seconds}s"] = {
            "position_error_meters": float(
                position_error.mean()
            ),
            "heading_error_degrees": float(
                heading_error.mean()
            ),
        }

    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(
    model: RobotPoseLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    heading_weight: float,
    training: bool,
) -> dict:
    if training:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    position_loss_sum = 0.0
    heading_loss_sum = 0.0
    example_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):
            predictions = model(inputs)

            (
                loss,
                position_loss,
                heading_loss,
            ) = trajectory_loss(
                prediction=predictions,
                target=targets,
                heading_weight=heading_weight,
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        batch_size = inputs.shape[0]

        total_loss_sum += (
            float(loss.item())
            * batch_size
        )

        position_loss_sum += (
            float(position_loss.item())
            * batch_size
        )

        heading_loss_sum += (
            float(heading_loss.item())
            * batch_size
        )

        example_count += batch_size

    return {
        "loss": (
            total_loss_sum
            / example_count
        ),
        "position_loss": (
            position_loss_sum
            / example_count
        ),
        "heading_loss": (
            heading_loss_sum
            / example_count
        ),
    }



def has_aligned_ground_truth_data(
    path: Path,
) -> bool:
    """
    Return True only for .npz files produced by the Quest + robot-ground-truth
    alignment server.

    Older Quest-only motion-window files are ignored automatically.
    """
    required_keys = {
        "quest_time_from_window_start",
        "quest_positions",
        "quest_headings",
        "robot_ground_truth_positions",
        "robot_ground_truth_headings",
    }

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            return required_keys.issubset(
                set(data.files)
            )

    except Exception as exc:
        print(
            f"WARNING: could not inspect "
            f"{path.name}: {exc}"
        )

        return False


def main() -> None:
    parser = argparse.ArgumentParser()

    # train_lstm.py lives in:
    #
    # robot_pose_prediction/model_training/
    #
    # Therefore the parent of model_training is robot_pose_prediction.
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
        default=3.0,
    )

    parser.add_argument(
        "--prediction-seconds",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--anchor-stride-seconds",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--kinematic-estimation-seconds",
        type=float,
        default=0.5,
        help=(
            "Amount of recent observed history used to estimate "
            "velocity and yaw rate for the kinematic baseline."
        ),
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--decoder-hidden-size",
        type=int,
        default=128,
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
        default=100,
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
        "--heading-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
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

    set_seed(args.seed)

    config = SequenceConfig(
        sample_rate_hz=args.sample_rate_hz,
        history_seconds=args.history_seconds,
        prediction_seconds=args.prediction_seconds,
        anchor_stride_seconds=(
            args.anchor_stride_seconds
        ),
    )

    data_dir = args.data_dir.resolve()
    weights_dir = args.weights_dir.resolve()
    eval_dir = args.eval_dir.resolve()

    weights_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_npz_files = sorted(
        data_dir.glob("*.npz")
    )

    if not all_npz_files:
        raise RuntimeError(
            f"No .npz files found in {data_dir}"
        )

    files = [
        path
        for path in all_npz_files
        if has_aligned_ground_truth_data(
            path
        )
    ]

    ignored_file_count = (
        len(all_npz_files)
        - len(files)
    )

    if not files:
        raise RuntimeError(
            "No aligned Quest + robot-ground-truth "
            f".npz files found in {data_dir}. "
            "Collect data with the updated alignment "
            "server before training."
        )

    print()
    print(
        f"Aligned motion-window files: "
        f"{len(files)}"
    )

    if ignored_file_count > 0:
        print(
            "Ignored older/incompatible .npz files: "
            f"{ignored_file_count}"
        )
    print(f"Data directory: {data_dir}")
    print(f"Weights directory: {weights_dir}")
    print(f"Evaluation directory: {eval_dir}")
    print()
    print(
        "LSTM input:  Quest-estimated robot motion"
    )
    print(
        "LSTM target: aligned robot ground truth"
    )
    print(
        "Frames:      each stream independently "
        "normalized at the prediction anchor"
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
        f"Train files:      "
        f"{len(train_files)}"
    )
    print(
        f"Validation files: "
        f"{len(validation_files)}"
    )
    print(
        f"Test files:       "
        f"{len(test_files)}"
    )

    print()
    print("Generating robot-relative examples...")

    train_inputs, train_targets = (
        build_dataset_arrays(
            train_files,
            config,
        )
    )

    validation_inputs, validation_targets = (
        build_dataset_arrays(
            validation_files,
            config,
        )
    )

    test_inputs, test_targets = (
        build_dataset_arrays(
            test_files,
            config,
        )
    )

    print(
        f"Train examples:      "
        f"{len(train_inputs)}"
    )
    print(
        f"Validation examples: "
        f"{len(validation_inputs)}"
    )
    print(
        f"Test examples:       "
        f"{len(test_inputs)}"
    )

    # ---------------------------------------------------------------
    # Normalize using training data only.
    # ---------------------------------------------------------------

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

    train_dataset = RobotTrajectoryDataset(
        inputs=train_inputs,
        targets=train_targets,
        input_mean=input_mean,
        input_std=input_std,
    )

    validation_dataset = RobotTrajectoryDataset(
        inputs=validation_inputs,
        targets=validation_targets,
        input_mean=input_mean,
        input_std=input_std,
    )

    test_dataset = RobotTrajectoryDataset(
        inputs=test_inputs,
        targets=test_targets,
        input_mean=input_mean,
        input_std=input_std,
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

    print()
    print(f"Device: {device}")
    print()

    model = RobotPoseLSTM(
        input_size=7,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        prediction_steps=(
            config.prediction_steps
        ),
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
            patience=5,
        )
    )

    checkpoint_path = (
        weights_dir
        / "robot_pose_lstm_best.pt"
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    training_history = []

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            heading_weight=(
                args.heading_weight
            ),
            training=True,
        )

        validation_result = run_epoch(
            model=model,
            loader=validation_loader,
            optimizer=optimizer,
            device=device,
            heading_weight=(
                args.heading_weight
            ),
            training=False,
        )

        validation_loss = (
            validation_result["loss"]
        )

        scheduler.step(validation_loss)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_result['loss']:.5f} | "
            f"val={validation_loss:.5f} | "
            f"pos={validation_result['position_loss']:.5f} | "
            f"heading={validation_result['heading_loss']:.5f} | "
            f"lr={current_lr:.2e}"
        )

        training_history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train": train_result,
                "validation": validation_result,
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

            checkpoint = {
                "model_state_dict": (
                    model.state_dict()
                ),
                "model_config": (
                    model.get_config()
                ),
                "sequence_config": (
                    asdict(config)
                ),
                "data_sources": {
                    "input": (
                        "quest_estimated_pose_history"
                    ),
                    "target": (
                        "aligned_robot_ground_truth_future"
                    ),
                    "coordinate_normalization": (
                        "independent_anchor_relative_frames"
                    ),
                },
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
                "best_validation_loss": (
                    best_validation_loss
                ),
                "epoch": epoch,
                "train_files": [
                    path.name
                    for path in train_files
                ],
                "validation_files": [
                    path.name
                    for path in validation_files
                ],
                "test_files": [
                    path.name
                    for path in test_files
                ],
            }

            torch.save(
                checkpoint,
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
                "Early stopping: "
                f"validation loss did not improve "
                f"for {args.patience} epochs."
            )
            break

    # ---------------------------------------------------------------
    # Reload best checkpoint.
    # ---------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    test_metrics = compute_metrics(
        model=model,
        loader=test_loader,
        device=device,
    )

    horizon_metrics = compute_horizon_metrics(
        model=model,
        loader=test_loader,
        device=device,
        config=config,
    )

    # ---------------------------------------------------------------
    # Constant-turn-rate kinematic baseline.
    #
    # IMPORTANT:
    # Use the original UNNORMALIZED test_inputs here. The kinematic model
    # operates directly in meters, meters/second, and radians/second.
    # ---------------------------------------------------------------

    kinematic_predictions = (
        predict_constant_turn_rate(
            inputs=test_inputs,
            config=config,
            estimation_seconds=(
                args.kinematic_estimation_seconds
            ),
        )
    )

    kinematic_test_metrics = (
        compute_array_metrics(
            predictions=kinematic_predictions,
            targets=test_targets,
        )
    )

    kinematic_horizon_metrics = (
        compute_array_horizon_metrics(
            predictions=kinematic_predictions,
            targets=test_targets,
            config=config,
        )
    )

    print()
    print("Final test metrics")
    print("------------------")

    metric_names = [
        "ade_meters",
        "fde_meters",
        "mean_heading_error_degrees",
        "final_heading_error_degrees",
    ]

    print(
        f"{'Metric':<36}"
        f"{'LSTM':>12}"
        f"{'Kinematic':>14}"
    )

    print(
        "-" * 62
    )

    for name in metric_names:
        print(
            f"{name:<36}"
            f"{test_metrics[name]:>12.4f}"
            f"{kinematic_test_metrics[name]:>14.4f}"
        )

    print()
    print("Test metrics by prediction horizon")
    print("----------------------------------")

    print(
        f"{'Horizon':<9}"
        f"{'LSTM pos':>12}"
        f"{'Kin pos':>12}"
        f"{'LSTM head':>14}"
        f"{'Kin head':>14}"
    )

    print(
        "-" * 61
    )

    for horizon, metrics in horizon_metrics.items():
        kinematic_metrics = (
            kinematic_horizon_metrics[
                horizon
            ]
        )

        print(
            f"{horizon:<9}"
            f"{metrics['position_error_meters']:>10.3f} m"
            f"{kinematic_metrics['position_error_meters']:>10.3f} m"
            f"{metrics['heading_error_degrees']:>11.2f} deg"
            f"{kinematic_metrics['heading_error_degrees']:>11.2f} deg"
        )

    print(
        "-" * 61
    )
    print()

    # ---------------------------------------------------------------
    # Save evaluation results.
    # ---------------------------------------------------------------

    summary = {
        "sequence_config": asdict(config),
        "data_sources": {
            "input": (
                "quest_estimated_pose_history"
            ),
            "target": (
                "aligned_robot_ground_truth_future"
            ),
            "coordinate_normalization": (
                "independent_anchor_relative_frames"
            ),
        },
        "model_config": (
            checkpoint["model_config"]
        ),
        "best_validation_loss": (
            checkpoint[
                "best_validation_loss"
            ]
        ),
        "best_epoch": checkpoint["epoch"],
        "test_metrics": test_metrics,
        "horizon_metrics": horizon_metrics,
        "kinematic_baseline": {
            "type": (
                "constant_body_velocity_"
                "constant_yaw_rate"
            ),
            "estimation_seconds": (
                args.kinematic_estimation_seconds
            ),
            "test_metrics": (
                kinematic_test_metrics
            ),
            "horizon_metrics": (
                kinematic_horizon_metrics
            ),
        },
        "train_files": [
            path.name
            for path in train_files
        ],
        "validation_files": [
            path.name
            for path in validation_files
        ],
        "test_files": [
            path.name
            for path in test_files
        ],
        "training_history": training_history,
    }

    eval_summary_path = (
        eval_dir
        / "robot_pose_lstm_eval.json"
    )

    with open(
        eval_summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        f"Best checkpoint: {checkpoint_path}"
    )

    print(
        f"Evaluation results: {eval_summary_path}"
    )


if __name__ == "__main__":
    main()