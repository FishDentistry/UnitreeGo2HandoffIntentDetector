import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from robot_pose_prediction.src.lstm_model import RobotPoseMDNLSTM


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
        return int(round(self.history_seconds * self.sample_rate_hz))

    @property
    def prediction_steps(self) -> int:
        return int(round(self.prediction_seconds * self.sample_rate_hz))


class RobotTrajectoryDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        input_mean: np.ndarray,
        input_std: np.ndarray,
    ) -> None:
        self.inputs = ((inputs - input_mean) / input_std).astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.inputs[index]),
            torch.from_numpy(self.targets[index]),
        )


# ---------------------------------------------------------------------------
# Data loading and robot-relative example construction
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """
    Load robot-only motion from an .npz file.

    Preferred keys:
        robot_time_from_window_start  (N,)
        robot_positions               (N, 3)
        robot_headings                (N, 3)

    Existing aligned recordings are also accepted, but ONLY the robot stream is
    read:
        robot_ground_truth_timestamps
        robot_ground_truth_positions
        robot_ground_truth_headings

    Generic timestamps/positions/headings are also accepted.
    """
    with np.load(path, allow_pickle=False) as data:
        try:
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
        except KeyError as exc:
            raise ValueError(
                f"{path.name}: unrecognized robot-motion schema. {exc}"
            ) from exc

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

    # Internal time is always relative; absolute ROS timestamps are fine.
    times = times - times[0]
    return times, positions, headings


def heading_vectors_to_unwrapped_yaw(headings: np.ndarray) -> np.ndarray:
    """Unity-style yaw: atan2(+X component, +Z component)."""
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
    """Translate/rotate positions into the robot frame at the current anchor."""
    dx = world_x - anchor_x
    dz = world_z - anchor_z
    s = math.sin(anchor_yaw)
    c = math.cos(anchor_yaw)
    relative_x = dx * c - dz * s
    relative_z = dx * s + dz * c
    return relative_x, relative_z


def build_example(
    times: np.ndarray,
    positions: np.ndarray,
    yaw_world: np.ndarray,
    anchor_time: float,
    config: SequenceConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one paper-style robot-relative training example.

    Input, shape (history_steps, 7):
        [relative_x, relative_z,
         velocity_x, velocity_z,
         sin(relative_yaw), cos(relative_yaw), yaw_rate]

    relative_yaw is each historical robot heading expressed relative to the
    robot heading at the prediction anchor. Therefore the final historical
    heading is always 0 rad, represented as sin=0, cos=1. yaw_rate is computed
    from the unwrapped relative-yaw history.

    Target, shape (prediction_steps, 2):
        [future_relative_x, future_relative_z]

    The current robot pose defines the local frame. Thus the current position is
    exactly (0, 0), the current robot faces +Z, and neither robot-map coordinates
    nor Quest-world coordinates are learned by the network.
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

    history_query = anchor_time + history_relative_times
    future_query = anchor_time + future_relative_times

    anchor_x = float(np.interp(anchor_time, times, positions[:, 0]))
    anchor_z = float(np.interp(anchor_time, times, positions[:, 2]))
    anchor_yaw = float(np.interp(anchor_time, times, yaw_world))

    history_x = np.interp(history_query, times, positions[:, 0])
    history_z = np.interp(history_query, times, positions[:, 2])
    future_x = np.interp(future_query, times, positions[:, 0])
    future_z = np.interp(future_query, times, positions[:, 2])

    # Interpolate the unwrapped robot yaw over the observation history. Express
    # all historical headings relative to the current anchor heading so the
    # orientation features are robot-relative just like the position features.
    history_yaw = np.interp(history_query, times, yaw_world)
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

    # As in the published method, translational velocity is an explicit input
    # feature.
    velocity_x = np.gradient(history_x_rel, dt)
    velocity_z = np.gradient(history_z_rel, dt)

    # New heading-dynamics features. sin/cos avoid the discontinuity at +/-pi.
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
    )
    targets = np.stack([future_x_rel, future_z_rel], axis=-1)

    return inputs.astype(np.float32), targets.astype(np.float32)


def recent_history_mean_speed(
    inputs: np.ndarray,
    config: SequenceConfig,
    movement_check_seconds: float,
) -> float:
    """
    Estimate how fast the robot is moving immediately before the prediction
    anchor using ONLY the observed history.

    The input velocity channels are:
        inputs[..., 2] = relative_velocity_x
        inputs[..., 3] = relative_velocity_z

    We average speed magnitude over the final movement_check_seconds of the
    history. This is causal: no future target information is used to decide
    whether an anchor is retained.
    """
    if movement_check_seconds <= 0.0:
        raise ValueError("movement_check_seconds must be > 0.")

    check_steps = max(
        1,
        int(round(movement_check_seconds * config.sample_rate_hz)),
    )
    check_steps = min(check_steps, inputs.shape[0])

    recent_velocity = inputs[-check_steps:, 2:4]
    recent_speed = np.linalg.norm(recent_velocity, axis=-1)

    return float(recent_speed.mean())


def extract_examples_from_file(
    path: Path,
    config: SequenceConfig,
    min_recent_speed: float,
    movement_check_seconds: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    times, positions, headings = load_robot_motion_window(path)
    yaw_world = heading_vectors_to_unwrapped_yaw(headings)

    first_anchor = times[0] + config.history_seconds
    last_anchor = times[-1] - config.prediction_seconds

    if last_anchor < first_anchor:
        duration = float(times[-1] - times[0])
        required = config.history_seconds + config.prediction_seconds
        print(
            f"Skipping {path.name}: duration={duration:.3f}s, "
            f"required>={required:.3f}s."
        )
        return [], [], 0, 0

    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    candidate_count = 0
    filtered_stationary_count = 0

    anchor_time = first_anchor
    while anchor_time <= last_anchor + 1e-9:
        try:
            x, y = build_example(
                times=times,
                positions=positions,
                yaw_world=yaw_world,
                anchor_time=anchor_time,
                config=config,
            )

            candidate_count += 1

            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                mean_recent_speed = recent_history_mean_speed(
                    inputs=x,
                    config=config,
                    movement_check_seconds=movement_check_seconds,
                )

                if mean_recent_speed < min_recent_speed:
                    filtered_stationary_count += 1
                else:
                    inputs.append(x)
                    targets.append(y)
        except ValueError as exc:
            print(f"Skipping example from {path.name}: {exc}")

        anchor_time += config.anchor_stride_seconds

    return (
        inputs,
        targets,
        candidate_count,
        filtered_stationary_count,
    )


def build_dataset_arrays(
    files: Sequence[Path],
    config: SequenceConfig,
    min_recent_speed: float,
    movement_check_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    all_inputs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    candidate_count = 0
    filtered_stationary_count = 0

    for path in files:
        (
            inputs,
            targets,
            file_candidate_count,
            file_filtered_stationary_count,
        ) = extract_examples_from_file(
            path=path,
            config=config,
            min_recent_speed=min_recent_speed,
            movement_check_seconds=movement_check_seconds,
        )

        all_inputs.extend(inputs)
        all_targets.extend(targets)

        candidate_count += file_candidate_count
        filtered_stationary_count += file_filtered_stationary_count

    if not all_inputs:
        raise RuntimeError(
            "No valid moving robot trajectory examples were generated. "
            "Try lowering --min-recent-speed."
        )

    stats = {
        "candidate_examples": candidate_count,
        "filtered_stationary_examples": filtered_stationary_count,
        "retained_moving_examples": len(all_inputs),
    }

    return (
        np.stack(all_inputs),
        np.stack(all_targets),
        stats,
    )


def split_files(
    files: Sequence[Path],
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> Tuple[List[Path], List[Path], List[Path]]:
    files = list(files)
    if len(files) < 3:
        raise RuntimeError("At least 3 .npz files are required for file-level splitting.")

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


def has_robot_motion_data(path: Path) -> bool:
    try:
        load_robot_motion_window(path)
        return True
    except Exception as exc:
        print(f"Ignoring incompatible file {path.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Per-horizon bivariate GMM likelihood
# ---------------------------------------------------------------------------


def bivariate_component_log_prob(
    mean: torch.Tensor,
    std: torch.Tensor,
    rho: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Log density of each correlated bivariate Gaussian component.

    Args:
        mean:   (B, T, K, 2)
        std:    (B, T, K, 2)
        rho:    (B, T, K)
        target: (B, T, 2)

    Returns:
        (B, T, K)
    """
    target = target.unsqueeze(2)

    sigma_x = std[..., 0]
    sigma_z = std[..., 1]
    z_x = (target[..., 0] - mean[..., 0]) / sigma_x
    z_z = (target[..., 1] - mean[..., 1]) / sigma_z

    one_minus_rho2 = torch.clamp(1.0 - rho * rho, min=1e-6)
    quadratic = (
        z_x * z_x
        + z_z * z_z
        - 2.0 * rho * z_x * z_z
    ) / (2.0 * one_minus_rho2)

    log_normalizer = (
        math.log(2.0 * math.pi)
        + torch.log(sigma_x)
        + torch.log(sigma_z)
        + 0.5 * torch.log(one_minus_rho2)
    )

    return -quadratic - log_normalizer


def mdn_nll(
    model: RobotPoseMDNLSTM,
    output: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Published-style MDN NLL: one bivariate Gaussian mixture per future horizon,
    then average the negative log likelihood over batch and horizons.
    """
    params = model.parameters_from_output(output)
    component_log_prob = bivariate_component_log_prob(
        mean=params["mean"],
        std=params["std"],
        rho=params["rho"],
        target=target,
    )
    log_weights = torch.log_softmax(params["logits"], dim=-1)
    mixture_log_prob = torch.logsumexp(
        log_weights + component_log_prob,
        dim=-1,
    )
    return -mixture_log_prob.mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def displacement_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-example ADE and FDE for tensors shaped (B, T, 2)."""
    errors = torch.linalg.vector_norm(prediction - target, dim=-1)
    return errors.mean(dim=-1), errors[:, -1]


def sample_mdn_trajectories(
    model: RobotPoseMDNLSTM,
    output: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Sample K complete hypotheses from the paper's independent per-horizon GMMs.

    Returns:
        (num_samples, B, T, 2)

    Each future horizon samples its own mixture component, matching the official
    evaluation formulation rather than imposing a trajectory-level latent mode.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    params = model.parameters_from_output(output)
    probabilities = params["probabilities"]  # (B,T,K)
    mean = params["mean"]                    # (B,T,K,2)
    std = params["std"]                      # (B,T,K,2)
    rho = params["rho"]                      # (B,T,K)

    b, t, k = probabilities.shape

    flat_probs = probabilities.reshape(-1, k)
    component_indices = torch.multinomial(
        flat_probs,
        num_samples=num_samples,
        replacement=True,
    )
    # (B*T, S) -> (S, B, T)
    component_indices = component_indices.view(b, t, num_samples).permute(2, 0, 1)

    mean_expanded = mean.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
    std_expanded = std.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
    rho_expanded = rho.unsqueeze(0).expand(num_samples, -1, -1, -1)

    gather_index_2 = component_indices[..., None, None].expand(-1, -1, -1, 1, 2)
    gather_index_1 = component_indices[..., None].expand(-1, -1, -1, 1)

    selected_mean = torch.gather(mean_expanded, 3, gather_index_2).squeeze(3)
    selected_std = torch.gather(std_expanded, 3, gather_index_2).squeeze(3)
    selected_rho = torch.gather(rho_expanded, 3, gather_index_1).squeeze(3)

    eps_x = torch.randn_like(selected_rho)
    eps_z = torch.randn_like(selected_rho)
    correlated_z = (
        selected_rho * eps_x
        + torch.sqrt(torch.clamp(1.0 - selected_rho ** 2, min=1e-6)) * eps_z
    )

    sample_x = selected_mean[..., 0] + selected_std[..., 0] * eps_x
    sample_z = selected_mean[..., 1] + selected_std[..., 1] * correlated_z

    return torch.stack([sample_x, sample_z], dim=-1)


@torch.no_grad()
def evaluate_model(
    model: RobotPoseMDNLSTM,
    loader: DataLoader,
    device: torch.device,
    config: SequenceConfig,
    num_samples: int = 20,
) -> Dict[str, object]:
    model.eval()

    total_nll = 0.0
    total_count = 0

    expected_ade: List[torch.Tensor] = []
    expected_fde: List[torch.Tensor] = []
    modal_ade: List[torch.Tensor] = []
    modal_fde: List[torch.Tensor] = []
    min_ade: List[torch.Tensor] = []
    min_fde: List[torch.Tensor] = []
    paper_style_min_ade: List[torch.Tensor] = []
    paper_style_min_fde: List[torch.Tensor] = []

    # The official evaluator samples/evaluates at 0.8 s intervals (indices
    # 7,15,23,... at 10 Hz). For our 4.0 s horizon that becomes
    # 0.8, 1.6, 2.4, 3.2, and 4.0 seconds. We report both this paper-style
    # metric and the stricter all-40-timestep metric.
    paper_stride_steps = max(1, int(round(0.8 * config.sample_rate_hz)))
    paper_indices = list(
        range(paper_stride_steps - 1, config.prediction_steps, paper_stride_steps)
    )
    if not paper_indices or paper_indices[-1] != config.prediction_steps - 1:
        paper_indices.append(config.prediction_steps - 1)

    horizon_expected: Dict[int, List[torch.Tensor]] = {}
    horizon_modal: Dict[int, List[torch.Tensor]] = {}
    horizon_min_sample: Dict[int, List[torch.Tensor]] = {}

    whole_seconds = range(1, int(math.floor(config.prediction_seconds)) + 1)
    horizon_indices = {
        seconds: int(round(seconds * config.sample_rate_hz)) - 1
        for seconds in whole_seconds
        if int(round(seconds * config.sample_rate_hz)) - 1 < config.prediction_steps
    }

    for seconds in horizon_indices:
        horizon_expected[seconds] = []
        horizon_modal[seconds] = []
        horizon_min_sample[seconds] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        output = model(inputs)
        loss = mdn_nll(model, output, targets)

        batch_size = inputs.shape[0]
        total_nll += float(loss.item()) * batch_size
        total_count += batch_size

        expected = model.expected_position(output)
        modal = model.modal_component_position(output)
        samples = sample_mdn_trajectories(model, output, num_samples)

        ade, fde = displacement_metrics(expected, targets)
        expected_ade.append(ade.cpu())
        expected_fde.append(fde.cpu())

        ade, fde = displacement_metrics(modal, targets)
        modal_ade.append(ade.cpu())
        modal_fde.append(fde.cpu())

        sample_errors = torch.linalg.vector_norm(
            samples - targets.unsqueeze(0),
            dim=-1,
        )  # (S,B,T)
        sample_ade = sample_errors.mean(dim=-1)  # (S,B)
        sample_fde = sample_errors[..., -1]      # (S,B)
        min_ade.append(sample_ade.min(dim=0).values.cpu())
        min_fde.append(sample_fde.min(dim=0).values.cpu())

        paper_errors = sample_errors[:, :, paper_indices]
        paper_style_min_ade.append(
            paper_errors.mean(dim=-1).min(dim=0).values.cpu()
        )
        paper_style_min_fde.append(
            paper_errors[..., -1].min(dim=0).values.cpu()
        )

        for seconds, index in horizon_indices.items():
            horizon_expected[seconds].append(
                torch.linalg.vector_norm(
                    expected[:, index] - targets[:, index],
                    dim=-1,
                ).cpu()
            )
            horizon_modal[seconds].append(
                torch.linalg.vector_norm(
                    modal[:, index] - targets[:, index],
                    dim=-1,
                ).cpu()
            )
            horizon_min_sample[seconds].append(
                sample_errors[:, :, index].min(dim=0).values.cpu()
            )

    def cat_mean(values: List[torch.Tensor]) -> float:
        return float(torch.cat(values, dim=0).mean())

    horizon_metrics: Dict[str, Dict[str, float]] = {}
    for seconds in horizon_indices:
        horizon_metrics[f"{seconds}s"] = {
            "expected_position_error_meters": cat_mean(horizon_expected[seconds]),
            "modal_position_error_meters": cat_mean(horizon_modal[seconds]),
            f"min_position_error_at_{num_samples}_meters": cat_mean(
                horizon_min_sample[seconds]
            ),
        }

    return {
        "nll_per_horizon_point": total_nll / max(total_count, 1),
        "expected_trajectory_ade_meters": cat_mean(expected_ade),
        "expected_trajectory_fde_meters": cat_mean(expected_fde),
        "modal_trajectory_ade_meters": cat_mean(modal_ade),
        "modal_trajectory_fde_meters": cat_mean(modal_fde),
        f"minade_at_{num_samples}_meters": cat_mean(min_ade),
        f"minfde_at_{num_samples}_meters": cat_mean(min_fde),
        f"paper_style_0p8s_minade_at_{num_samples}_meters": cat_mean(
            paper_style_min_ade
        ),
        f"paper_style_0p8s_minfde_at_{num_samples}_meters": cat_mean(
            paper_style_min_fde
        ),
        "paper_style_horizon_indices": paper_indices,
        "horizon_metrics": horizon_metrics,
    }


# ---------------------------------------------------------------------------
# Constant-velocity baseline using the translational velocity input channels
# ---------------------------------------------------------------------------


def predict_constant_velocity(
    inputs: np.ndarray,
    config: SequenceConfig,
    estimation_seconds: float = 0.5,
) -> np.ndarray:
    if inputs.ndim != 3 or inputs.shape[-1] < 4:
        raise ValueError(
            "Expected inputs with shape (N, history_steps, >=4) and "
            "velocity channels at indices 2:4."
        )

    estimation_steps = max(
        1,
        int(round(estimation_seconds * config.sample_rate_hz)),
    )
    estimation_steps = min(estimation_steps, inputs.shape[1])

    recent_velocity = inputs[:, -estimation_steps:, 2:4].mean(axis=1)
    future_times = (
        np.arange(1, config.prediction_steps + 1, dtype=np.float64)
        * config.dt
    )

    return (
        recent_velocity[:, None, :] * future_times[None, :, None]
    ).astype(np.float32)


def compute_numpy_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    config: SequenceConfig,
) -> Dict[str, object]:
    errors = np.linalg.norm(prediction - target, axis=-1)
    horizon_metrics: Dict[str, Dict[str, float]] = {}

    for seconds in range(1, int(math.floor(config.prediction_seconds)) + 1):
        index = int(round(seconds * config.sample_rate_hz)) - 1
        if index < config.prediction_steps:
            horizon_metrics[f"{seconds}s"] = {
                "position_error_meters": float(errors[:, index].mean())
            }

    return {
        "ade_meters": float(errors.mean()),
        "fde_meters": float(errors[:, -1].mean()),
        "horizon_metrics": horizon_metrics,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_epoch(
    model: RobotPoseMDNLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool,
) -> float:
    if training:
        model.train()
    else:
        model.eval()

    total = 0.0
    count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(inputs)
            loss = mdn_nll(model, output, targets)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite MDN loss encountered. "
                    "Try a smaller learning rate or inspect trajectory scaling."
                )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        batch_size = inputs.shape[0]
        total += float(loss.item()) * batch_size
        count += batch_size

    return total / max(count, 1)


def linear_lr_factor(epoch: int, total_epochs: int, end_factor: float) -> float:
    if total_epochs <= 1:
        return 1.0
    progress = min(max(epoch, 0), total_epochs - 1) / float(total_epochs - 1)
    return 1.0 + progress * (end_factor - 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Robot-only, robot-relative per-horizon LSTM+MDN trajectory "
            "forecasting using anchors where the robot is already moving."
        )
    )

    project_dir = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data" /"robot_ground_truth",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=project_dir / "weights",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=project_dir / "eval",
    )

    parser.add_argument("--sample-rate-hz", type=float, default=10.0)
    parser.add_argument("--history-seconds", type=float, default=3.0)
    parser.add_argument("--prediction-seconds", type=float, default=4.0)
    parser.add_argument("--anchor-stride-seconds", type=float, default=0.5)

    # Keep only anchors where the robot has already begun moving according to
    # the OBSERVED history. This avoids asking the model to predict an
    # unobservable stop->move decision from a completely stationary history.
    parser.add_argument(
        "--min-recent-speed",
        type=float,
        default=0.15,
        help=(
            "Minimum mean planar speed (m/s) over the final observed speed "
            "window required to keep a training/evaluation anchor."
        ),
    )
    parser.add_argument(
        "--movement-check-seconds",
        type=float,
        default=0.2,
        help=(
            "Length of observed history immediately before the anchor used "
            "for the moving/stationary decision."
        ),
    )

    # Keep the released paper implementation's H=8, 1 layer, and 3 Gaussians,
    # but extend its 4-D pos+velocity input with 3 robot-heading-dynamics
    # features: sin(relative_yaw), cos(relative_yaw), and yaw_rate.
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-mixtures", type=int, default=3)

    # Paper reports batch 1024; released multi-dataset config uses 4096. Use 1024
    # here because robot datasets are typically much smaller. This remains
    # overridable without changing code.
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--final-learning-rate", type=float, default=1e-7)
    parser.add_argument("--num-eval-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=25)

    args = parser.parse_args()

    if args.sample_rate_hz <= 0:
        raise ValueError("sample-rate-hz must be > 0.")
    if args.history_seconds <= 0 or args.prediction_seconds <= 0:
        raise ValueError("history/prediction seconds must be > 0.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be > 0.")
    if args.final_learning_rate <= 0 or args.learning_rate <= 0:
        raise ValueError("learning rates must be > 0.")
    if args.final_learning_rate > args.learning_rate:
        raise ValueError("final-learning-rate should not exceed learning-rate.")
    if args.min_recent_speed < 0.0:
        raise ValueError("min-recent-speed must be >= 0.")
    if args.movement_check_seconds <= 0.0:
        raise ValueError("movement-check-seconds must be > 0.")

    set_seed(args.seed)

    config = SequenceConfig(
        sample_rate_hz=args.sample_rate_hz,
        history_seconds=args.history_seconds,
        prediction_seconds=args.prediction_seconds,
        anchor_stride_seconds=args.anchor_stride_seconds,
    )

    data_dir = args.data_dir.expanduser().resolve()
    weights_dir = args.weights_dir.expanduser().resolve()
    eval_dir = args.eval_dir.expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(data_dir.glob("*.npz"))
    if not all_files:
        raise RuntimeError(f"No .npz files found in {data_dir}")

    files = [path for path in all_files if has_robot_motion_data(path)]
    if len(files) < 3:
        raise RuntimeError(
            f"Only {len(files)} compatible robot-motion files found; need >=3."
        )

    train_files, validation_files, test_files = split_files(files, args.seed)

    print()
    print("=== Robot-only per-horizon LSTM+MDN (moving anchors only) ===")
    print(f"Data directory:      {data_dir}")
    print(f"Compatible files:    {len(files)}")
    print(f"Train/val/test:      {len(train_files)}/{len(validation_files)}/{len(test_files)} files")
    print(f"History/prediction:  {config.history_seconds:.1f}s -> {config.prediction_seconds:.1f}s")
    print(
        "Anchor filter:       mean speed over final "
        f"{args.movement_check_seconds:.2f}s >= "
        f"{args.min_recent_speed:.3f} m/s"
    )
    print("Input:               [x, z, vx, vz, sin(dyaw), cos(dyaw), yaw_rate]")
    print(f"Target:              future [x, z] in same robot frame")
    print(f"MDN:                 {args.num_mixtures} bivariate Gaussians PER horizon")
    print(f"LSTM hidden/layers:  {args.hidden_size}/{args.num_layers}")
    print(f"Epochs/batch:        {args.epochs}/{args.batch_size}")
    print()

    train_inputs, train_targets, train_filter_stats = build_dataset_arrays(
        train_files,
        config,
        min_recent_speed=args.min_recent_speed,
        movement_check_seconds=args.movement_check_seconds,
    )
    validation_inputs, validation_targets, validation_filter_stats = build_dataset_arrays(
        validation_files,
        config,
        min_recent_speed=args.min_recent_speed,
        movement_check_seconds=args.movement_check_seconds,
    )
    test_inputs, test_targets, test_filter_stats = build_dataset_arrays(
        test_files,
        config,
        min_recent_speed=args.min_recent_speed,
        movement_check_seconds=args.movement_check_seconds,
    )

    print(
        "Train examples:      "
        f"{len(train_inputs)} retained / "
        f"{train_filter_stats['candidate_examples']} candidates "
        f"({train_filter_stats['filtered_stationary_examples']} filtered)"
    )
    print(
        "Validation examples: "
        f"{len(validation_inputs)} retained / "
        f"{validation_filter_stats['candidate_examples']} candidates "
        f"({validation_filter_stats['filtered_stationary_examples']} filtered)"
    )
    print(
        "Test examples:       "
        f"{len(test_inputs)} retained / "
        f"{test_filter_stats['candidate_examples']} candidates "
        f"({test_filter_stats['filtered_stationary_examples']} filtered)"
    )

    input_mean = train_inputs.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    input_std = train_inputs.std(axis=(0, 1), keepdims=True).astype(np.float32)
    input_std = np.maximum(input_std, 1e-6)

    train_dataset = RobotTrajectoryDataset(
        train_inputs, train_targets, input_mean, input_std
    )
    validation_dataset = RobotTrajectoryDataset(
        validation_inputs, validation_targets, input_mean, input_std
    )
    test_dataset = RobotTrajectoryDataset(
        test_inputs, test_targets, input_mean, input_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RobotPoseMDNLSTM(
        input_size=7,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        prediction_steps=config.prediction_steps,
        num_mixtures=args.num_mixtures,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    end_factor = args.final_learning_rate / args.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: linear_lr_factor(
            epoch,
            args.epochs,
            end_factor,
        ),
    )

    best_validation_nll = float("inf")
    best_epoch = -1
    best_path = weights_dir / "robot_pose_per_horizon_mdn_heading_moving_only_best.pt"

    history = []

    for epoch in range(1, args.epochs + 1):
        train_nll = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training=True,
        )
        validation_nll = run_epoch(
            model,
            validation_loader,
            optimizer,
            device,
            training=False,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "validation_nll": validation_nll,
                "learning_rate": current_lr,
            }
        )

        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.get_config(),
                    "sequence_config": asdict(config),
                    "movement_filter": {
                        "min_recent_speed": args.min_recent_speed,
                        "movement_check_seconds": args.movement_check_seconds,
                        "uses_future_information": False,
                    },
                    "input_mean": input_mean,
                    "input_std": input_std,
                    "best_epoch": best_epoch,
                    "best_validation_nll": best_validation_nll,
                    "input_features": [
                        "relative_x",
                        "relative_z",
                        "relative_velocity_x",
                        "relative_velocity_z",
                        "sin_relative_yaw",
                        "cos_relative_yaw",
                        "yaw_rate",
                    ],
                    "target_features": [
                        "future_relative_x",
                        "future_relative_z",
                    ],
                },
                best_path,
            )

        scheduler.step()

        if (
            epoch == 1
            or epoch == args.epochs
            or epoch % args.print_every == 0
        ):
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train NLL {train_nll:.5f} | "
                f"val NLL {validation_nll:.5f} | "
                f"lr {current_lr:.3e} | "
                f"best {best_validation_nll:.5f} @ {best_epoch}"
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model = RobotPoseMDNLSTM(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Fixed seed makes sampled minADE@20 reproducible across repeated eval runs.
    torch.manual_seed(args.seed + 1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 1000)

    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        config,
        num_samples=args.num_eval_samples,
    )

    constant_velocity_predictions = predict_constant_velocity(
        test_inputs,
        config,
    )
    baseline_metrics = compute_numpy_metrics(
        constant_velocity_predictions,
        test_targets,
        config,
    )

    results = {
        "method": "paper_style_per_horizon_bivariate_mdn_with_heading_history_moving_only",
        "best_epoch": best_epoch,
        "best_validation_nll": best_validation_nll,
        "sequence_config": asdict(config),
        "model_config": model.get_config(),
        "movement_filter": {
            "min_recent_speed": args.min_recent_speed,
            "movement_check_seconds": args.movement_check_seconds,
            "uses_future_information": False,
        },
        "movement_filter_stats": {
            "train": train_filter_stats,
            "validation": validation_filter_stats,
            "test": test_filter_stats,
        },
        "train_file_count": len(train_files),
        "validation_file_count": len(validation_files),
        "test_file_count": len(test_files),
        "train_example_count": len(train_inputs),
        "validation_example_count": len(validation_inputs),
        "test_example_count": len(test_inputs),
        "test_metrics": test_metrics,
        "constant_velocity_baseline": baseline_metrics,
        "training_history": history,
    }

    metrics_path = eval_dir / "robot_pose_per_horizon_mdn_heading_moving_only_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print()
    print("=== TEST RESULTS ===")
    print(f"Best epoch:                 {best_epoch}")
    print(f"Test NLL / horizon point:   {test_metrics['nll_per_horizon_point']:.5f}")
    print(
        f"Expected-path ADE/FDE:      "
        f"{test_metrics['expected_trajectory_ade_meters']:.3f} / "
        f"{test_metrics['expected_trajectory_fde_meters']:.3f} m"
    )
    print(
        f"Modal-path ADE/FDE:         "
        f"{test_metrics['modal_trajectory_ade_meters']:.3f} / "
        f"{test_metrics['modal_trajectory_fde_meters']:.3f} m"
    )
    print(
        f"minADE/minFDE@{args.num_eval_samples} (all steps): "
        f"{test_metrics[f'minade_at_{args.num_eval_samples}_meters']:.3f} / "
        f"{test_metrics[f'minfde_at_{args.num_eval_samples}_meters']:.3f} m"
    )
    print(
        f"minADE/minFDE@{args.num_eval_samples} (0.8s):      "
        f"{test_metrics[f'paper_style_0p8s_minade_at_{args.num_eval_samples}_meters']:.3f} / "
        f"{test_metrics[f'paper_style_0p8s_minfde_at_{args.num_eval_samples}_meters']:.3f} m"
    )
    print(
        f"Constant-velocity ADE/FDE:  "
        f"{baseline_metrics['ade_meters']:.3f} / "
        f"{baseline_metrics['fde_meters']:.3f} m"
    )

    print("\nPer-horizon:")
    for horizon, values in test_metrics["horizon_metrics"].items():
        print(
            f"  {horizon}: expected={values['expected_position_error_meters']:.3f} m | "
            f"modal={values['modal_position_error_meters']:.3f} m | "
            f"min@{args.num_eval_samples}="
            f"{values[f'min_position_error_at_{args.num_eval_samples}_meters']:.3f} m"
        )

    print(f"\nCheckpoint: {best_path}")
    print(f"Metrics:    {metrics_path}")


if __name__ == "__main__":
    main()