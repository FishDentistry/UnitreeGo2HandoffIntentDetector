from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from quest_hand_intent_model_est.src.quest_hand_int_inf_wrapper import (
    QuestHandIntentEstInference,
)
from quest_hand_intent_model_est.src.quest_hand_int_tabm_inf_wrapper import (
    QuestHandIntentTabMEstInference,
)

QuestInferenceModel = Union[
    QuestHandIntentEstInference,
    QuestHandIntentTabMEstInference,
]
from quest_hand_intent_model_est.src.quest_joint_features import (
    SKELETON_EDGES,
    extract_quest_joint_features,
    feature_vector_to_joint_poses,
    construct_pose_img_from_joint_feature_vector
)
from shared.util.extract_all_samples import (
    Sample,
    discover_samples,
    get_sample_label_name,
    label_to_binary,
)
from shared.util.quest_joints_viz import visualize_joint_features
from model_training_and_implementation.src.resnet_encoder import ResNet18ImageEncoder



REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = REPO_ROOT / "quest_hand_intent_model_est" / "configs"

HANDOFF_CONFIG_PATH = CONFIG_DIR / "handoff_config.yaml"
COUNTERFACTUAL_CONFIG_PATH = CONFIG_DIR / "counterfactual_config.yaml"

WEIGHT_SEARCH_FEATURES_TYPE = "keypoints_projections"
RESNET_ENCODER: Optional[ResNet18ImageEncoder] = None


def get_resnet_encoder() -> ResNet18ImageEncoder:
    """Create the ResNet encoder only when the ResNet feature mode is used."""
    global RESNET_ENCODER
    if RESNET_ENCODER is None:
        RESNET_ENCODER = ResNet18ImageEncoder(
            pretrained=True,
            device="cuda",
            l2_normalize=True,
        )
    return RESNET_ENCODER

SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR / "visualizations"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# Classifier feature layout:
#
#   [63 joint XYZ values | 3 head roll/pitch/yaw values | derived features]
#
# Only the 63 joint XYZ values are modified by counterfactual optimization.
# The head-orientation values remain fixed at their observed values. The
# derived block (either chest-plane projections or a ResNet embedding of the
# rendered projection) is regenerated from each counterfactual joint pose.
JOINT_POSITION_DIMS = 3
JOINT_POSITION_FEATURE_COUNT = 63
HEAD_ORIENTATION_FEATURE_COUNT = 3
BASE_POSE_FEATURE_COUNT = (
    JOINT_POSITION_FEATURE_COUNT + HEAD_ORIENTATION_FEATURE_COUNT
)
PROJECTED_DIMS_PER_JOINT = 2
PROJECTED_FEATURE_COUNT = 21 * PROJECTED_DIMS_PER_JOINT
RESNET_SPSA_EPSILON = 1e-3
VERIFY_PROJECTION_FEATURES = False

MODIFIABLE_JOINT_NAMES = {
    "left_hand",
    "right_hand",
}

ARM_CHAINS = {
    "left": [
        "left_scapula",
        "left_upper_arm",
        "left_forearm",
        "left_hand",
    ],
    "right": [
        "right_scapula",
        "right_upper_arm",
        "right_forearm",
        "right_hand",
    ],
}

HAND_ATTACHED_JOINTS = {
    "left": [
        "left_wrist_twist",
        "left_palm",
    ],
    "right": [
        "right_wrist_twist",
        "right_palm",
    ],
}

COUNTERFACTUAL_SKELETON_EDGES = [
    ("left_scapula", "left_upper_arm"),
    ("left_upper_arm", "left_forearm"),
    ("left_forearm", "left_hand"),
    ("right_scapula", "right_upper_arm"),
    ("right_upper_arm", "right_forearm"),
    ("right_forearm", "right_hand"),
]

for replacement_edge in [
    ("left_forearm", "left_hand"),
    ("right_forearm", "right_hand"),
]:
    if replacement_edge not in COUNTERFACTUAL_SKELETON_EDGES:
        COUNTERFACTUAL_SKELETON_EDGES.append(
            replacement_edge
        )

@dataclass
class LatencyTracker:
    """Collect nested CPU, CUDA, and MPS latency measurements."""

    totals_ms: dict[str, float] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    _pending_cuda_events: list[
        tuple[
            str,
            torch.device,
            torch.cuda.Event,
            torch.cuda.Event,
        ]
    ] = field(default_factory=list)

    @staticmethod
    def _as_device(
        device: Optional[Union[torch.device, str]],
    ) -> Optional[torch.device]:
        if device is None:
            return None
        return torch.device(device)

    @staticmethod
    def _synchronize_mps() -> None:
        mps_module = getattr(torch, "mps", None)
        synchronize = getattr(mps_module, "synchronize", None)
        if callable(synchronize):
            synchronize()

    @contextmanager
    def measure(
        self,
        name: str,
        *,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Iterator[None]:
        resolved_device = self._as_device(device)

        if (
            resolved_device is not None
            and resolved_device.type == "cuda"
            and torch.cuda.is_available()
        ):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            with torch.cuda.device(resolved_device):
                start_event.record(
                    torch.cuda.current_stream(resolved_device)
                )

            try:
                yield
            finally:
                with torch.cuda.device(resolved_device):
                    end_event.record(
                        torch.cuda.current_stream(resolved_device)
                    )

                self._pending_cuda_events.append(
                    (
                        name,
                        resolved_device,
                        start_event,
                        end_event,
                    )
                )
            return

        if (
            resolved_device is not None
            and resolved_device.type == "mps"
        ):
            self._synchronize_mps()

        started_at = perf_counter()

        try:
            yield
        finally:
            if (
                resolved_device is not None
                and resolved_device.type == "mps"
            ):
                self._synchronize_mps()

            self.add_duration_ms(
                name,
                (perf_counter() - started_at) * 1000.0,
            )

    def add_duration_ms(
        self,
        name: str,
        duration_ms: float,
        *,
        call_count: int = 1,
    ) -> None:
        self.totals_ms[name] = (
            self.totals_ms.get(name, 0.0)
            + float(duration_ms)
        )
        self.call_counts[name] = (
            self.call_counts.get(name, 0)
            + int(call_count)
        )

    def _finalize_cuda_events(self) -> None:
        if not self._pending_cuda_events:
            return

        devices = {
            device
            for _, device, _, _ in self._pending_cuda_events
        }

        for device in devices:
            torch.cuda.synchronize(device)

        for (
            name,
            _,
            start_event,
            end_event,
        ) in self._pending_cuda_events:
            self.add_duration_ms(
                name,
                float(start_event.elapsed_time(end_event)),
            )

        self._pending_cuda_events.clear()

    def snapshot(self) -> dict[str, dict[str, Union[float, int]]]:
        self._finalize_cuda_events()

        snapshot: dict[
            str,
            dict[str, Union[float, int]],
        ] = {}

        for name in sorted(self.totals_ms):
            total_ms = float(self.totals_ms[name])
            call_count = int(self.call_counts.get(name, 0))

            snapshot[name] = {
                "total_ms": total_ms,
                "call_count": call_count,
                "mean_ms": (
                    total_ms / call_count
                    if call_count
                    else 0.0
                ),
            }

        return snapshot


@contextmanager
def measure_latency(
    tracker: Optional[LatencyTracker],
    name: str,
    *,
    device: Optional[Union[torch.device, str]] = None,
) -> Iterator[None]:
    if tracker is None:
        yield
        return

    with tracker.measure(name, device=device):
        yield


def format_latency_snapshot(
    latency: dict[str, dict[str, Union[float, int]]],
) -> str:
    def total(name: str) -> Optional[float]:
        component = latency.get(name)
        if component is None:
            return None
        return float(component["total_ms"])

    def format_value(name: str) -> str:
        value = total(name)
        return "n/a" if value is None else f"{value:.3f}"

    fabrik = latency.get("fabrik_solve")
    if fabrik is None:
        fabrik_text = "n/a"
    else:
        fabrik_text = (
            f"{float(fabrik['total_ms']):.3f}"
            f"/{int(fabrik['call_count'])} calls"
            f"/{float(fabrik['mean_ms']):.3f} avg"
        )

    return (
        "latency_ms("
        f"sample={format_value('sample_total')}, "
        f"minimal={format_value('minimal_perturbation')}, "
        f"optimization={format_value('optimization_loop')}, "
        f"fabrik={fabrik_text}, "
        f"classifier={format_value('classifier_forward')}, "
        f"binary_search={format_value('binary_search')}, "
        f"original_inference={format_value('original_inference')}, "
        f"final_inference={format_value('final_inference')}, "
        f"bone_validation={format_value('bone_validation')}"
        ")"
    )


DEFAULT_COUNTERFACTUAL_CONFIG: dict[str, Any] = {
    "counterfactual": {
        "classification_weight": 1000.0,
        "reachability_weight": 1000.0,
        "forward_extension_weight": 1000.0,
        "forward_extension_min": 0.20,
        "require_forward_extension_for_success": True,
        "max_iterations": 1000,
        "step_size": 0.01,
        "probability_margin": 1e-4,
        "bone_relative_tolerance": 0.01,
        "bone_absolute_tolerance": 1e-6,
        "target_mode": "opposite_prediction",
        "tuning": {
            "classification_weights": [
                10.0,
                100.0,
                1000.0,
                10000.0,
            ],
            "reachability_weights": [
                10.0,
                100.0,
                1000.0,
                10000.0,
            ],
            "minimum_success_rate": 0.95,
            "maximum_bone_violation_rate": 0.05,
        },
    }
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        return {}

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}, got {type(loaded).__name__}."
        )

    return loaded


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            default_flow_style=False,
        )


def load_or_create_counterfactual_config(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        write_yaml(path, DEFAULT_COUNTERFACTUAL_CONFIG)
        print(f"Created default counterfactual config: {path}")

    config = load_yaml(path)

    if "counterfactual" not in config:
        raise ValueError(
            f"{path} must contain a top-level 'counterfactual' section."
        )

    return config


def get_joint_samples(
    samples: list[Sample],
    include_rotation: bool,
) -> tuple[list[int], list[dict[str, Any]], int, int, int]:
    y_true: list[int] = []
    rows: list[dict[str, Any]] = []

    skipped_unreadable = 0
    skipped_no_object = 0
    skipped_no_people = 0
    errors = 0
    for local_idx, sample in enumerate(samples):
        try:
            label_name = get_sample_label_name(sample)
            y = label_to_binary(label_name)
            joint_head_feats = extract_quest_joint_features(
                    sample.joints_path,
                    include_rotations=include_rotation,
                )
            proj_joints, pose_img = construct_pose_img_from_joint_feature_vector(joint_head_feats)
            full_features =[]
            if(WEIGHT_SEARCH_FEATURES_TYPE == "keypoints_projections"):
                full_features = np.concatenate([joint_head_feats, proj_joints], axis=0)
            elif(WEIGHT_SEARCH_FEATURES_TYPE == "keypoints_resnet"):
                resnet_embedding = get_resnet_encoder().predict(
                    pose_img
                )["embedding"]
                full_features = np.concatenate([joint_head_feats, resnet_embedding.flatten().astype(np.float32)], axis=0)
            else:
                raise ValueError(f"Invalid WEIGHT_SEARCH_FEATURES_TYPE: {WEIGHT_SEARCH_FEATURES_TYPE}")
            row = {
                "participant_id": sample.participant_id,
                "index": local_idx,
                "label_name": label_name,
                "label_binary": int(y),
                "condition": sample.condition,
                "hand": sample.hand,
                "joint_features": full_features,
                "skipped": False,
            }

            rows.append(row)
            y_true.append(int(y))

        except Exception as error:
            errors += 1
            print(
                f"[{local_idx + 1}/{len(samples)}] ERROR: {error}"
            )
            rows.append(
                {
                    "index": local_idx,
                    "skipped": True,
                    "skip_reason": "error",
                    "error": str(error),
                }
            )

    print(
        "Joint sample loading summary: "
        f"loaded={len(y_true)}, "
        f"errors={errors}, "
        f"skipped_unreadable={skipped_unreadable}, "
        f"skipped_no_object={skipped_no_object}, "
        f"skipped_no_people={skipped_no_people}"
    )

    return (
        y_true,
        rows,
        skipped_unreadable,
        skipped_no_object,
        skipped_no_people,
    )


def probability_meets_target_class(
    probability: float,
    target_intent: bool,
    threshold: float,
) -> bool:
    return (
        probability >= threshold
        if target_intent
        else probability < threshold
    )


def determine_target_intent(
    *,
    row: dict[str, Any],
    original_probability: float,
    threshold: float,
    target_mode: str,
) -> bool:
    original_prediction = original_probability >= threshold

    if target_mode == "opposite_prediction":
        return not original_prediction

    if target_mode == "opposite_label":
        return not bool(row["label_binary"])

    if target_mode == "handoff":
        return True

    if target_mode == "not_handoff":
        return False

    raise ValueError(
        "Unsupported target_mode. Expected one of: "
        "'opposite_prediction', 'opposite_label', "
        "'handoff', or 'not_handoff'. "
        f"Received: {target_mode!r}."
    )


def skeleton_edge_length_statistics(
    original_joints: dict[str, dict[str, Any]],
    perturbed_joints: dict[str, dict[str, Any]],
    *,
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """
    Calculate post-optimization skeleton edge-length metrics.

    A normalized edge-length change is calculated as:

        abs(perturbed_length - original_length)
        / max(original_length, absolute_tolerance)
    """
    if relative_tolerance < 0.0:
        raise ValueError(
            "relative_tolerance must be nonnegative."
        )

    if absolute_tolerance <= 0.0:
        raise ValueError(
            "absolute_tolerance must be positive."
        )

    edge_changes: list[float] = []
    violating_edges: list[
        tuple[str, str, float, float, float]
    ] = []

    for joint_a, joint_b in COUNTERFACTUAL_SKELETON_EDGES:
        required = (
            joint_a in original_joints
            and joint_b in original_joints
            and joint_a in perturbed_joints
            and joint_b in perturbed_joints
        )

        if not required:
            continue

        original_pos_a = np.asarray(
            original_joints[joint_a]["position"],
            dtype=np.float64,
        )
        original_pos_b = np.asarray(
            original_joints[joint_b]["position"],
            dtype=np.float64,
        )
        perturbed_pos_a = np.asarray(
            perturbed_joints[joint_a]["position"],
            dtype=np.float64,
        )
        perturbed_pos_b = np.asarray(
            perturbed_joints[joint_b]["position"],
            dtype=np.float64,
        )

        positions = (
            original_pos_a,
            original_pos_b,
            perturbed_pos_a,
            perturbed_pos_b,
        )

        if any(position.shape != (3,) for position in positions):
            raise ValueError(
                f"Edge ({joint_a}, {joint_b}) contains a position "
                "that is not a three-element XYZ vector."
            )

        if not all(
            np.all(np.isfinite(position))
            for position in positions
        ):
            raise ValueError(
                f"Edge ({joint_a}, {joint_b}) contains NaN or "
                "infinite position values."
            )

        original_length = float(
            np.linalg.norm(original_pos_a - original_pos_b)
        )
        perturbed_length = float(
            np.linalg.norm(perturbed_pos_a - perturbed_pos_b)
        )

        absolute_change = abs(
            perturbed_length - original_length
        )
        normalized_change = (
            absolute_change
            / max(original_length, absolute_tolerance)
        )

        edge_changes.append(normalized_change)

        if normalized_change > relative_tolerance:
            violating_edges.append(
                (
                    joint_a,
                    joint_b,
                    original_length,
                    perturbed_length,
                    normalized_change,
                )
            )

    if edge_changes:
        mean_change = float(np.mean(edge_changes))
        max_change = float(np.max(edge_changes))
    else:
        mean_change = 0.0
        max_change = 0.0

    return {
        "has_violation": bool(violating_edges),
        "violating_edge_count": len(violating_edges),
        "evaluated_edge_count": len(edge_changes),
        "mean_relative_change": mean_change,
        "max_relative_change": max_change,
        "violating_edges": violating_edges,
    }


def unit_vector_or_fallback(
    vector: torch.Tensor,
    fallback: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize vector. If it is nearly zero, normalize fallback instead.
    """
    vector_length = torch.linalg.vector_norm(
        vector
    )

    fallback_length = (
        torch.linalg.vector_norm(fallback)
        .clamp_min(epsilon)
    )

    return torch.where(
        vector_length > epsilon,
        vector / vector_length.clamp_min(epsilon),
        fallback / fallback_length,
    )

def ensure_realistic_elbow(
    upper_arm_pos: torch.Tensor,
    elbow_pos: torch.Tensor,
    original_forearm_direction: torch.Tensor,
    proposed_direction: torch.Tensor,
    min_theta: float = 0.0,
    max_theta: float = np.pi,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    u = F.normalize(
        upper_arm_pos - elbow_pos,
        p=2,
        dim=0,
        eps=epsilon,
    )

    original_v = F.normalize(
        original_forearm_direction,
        p=2,
        dim=0,
        eps=epsilon,
    )

    proposed_v = F.normalize(
        proposed_direction,
        p=2,
        dim=0,
        eps=epsilon,
    )

    hinge_normal = F.normalize(
        torch.linalg.cross(u, original_v),
        p=2,
        dim=0,
        eps=epsilon,
    )

    signed_sine = torch.dot(
        hinge_normal,
        torch.linalg.cross(u, proposed_v),
    )

    cosine = torch.dot(
        u,
        proposed_v,
    ).clamp(-1.0, 1.0)

    proposed_angle = torch.atan2(
        signed_sine,
        cosine,
    )

    constrained_angle = torch.clamp(
        proposed_angle,
        min=min_theta,
        max=max_theta,
    )

    bend_direction = F.normalize(
        torch.linalg.cross(hinge_normal, u),
        p=2,
        dim=0,
        eps=epsilon,
    )

    constrained_v = (
        torch.cos(constrained_angle) * u
        + torch.sin(constrained_angle) * bend_direction
    )

    return F.normalize(
        constrained_v,
        p=2,
        dim=0,
        eps=epsilon,
    )

def solve_fabrik_chain(
    original_points: torch.Tensor,
    target_position: torch.Tensor,
    *,
    iterations: int = 10,
    epsilon: float = 1e-6,
    min_elbow_angle: float = 0.0,
    max_elbow_angle: float = np.pi,
) -> torch.Tensor:
    """
    Reposition a joint chain so its final point approaches the target.

    The first joint remains fixed, and every original segment length
    is preserved.

    original_points shape: (joint_count, 3)
    target_position shape: (3,)
    """
    if original_points.ndim != 2:
        raise ValueError(
            "original_points must have shape "
            "(joint_count, 3)."
        )

    if original_points.shape[1] != 3:
        raise ValueError(
            "Every joint position must contain XYZ."
        )

    root_position = original_points[0]

    original_segment_vectors = (
        original_points[1:]
        - original_points[:-1]
    )

    segment_lengths = (
        torch.linalg.vector_norm(
            original_segment_vectors,
            dim=1,
        )
        .clamp_min(epsilon)
    )

    maximum_reach = segment_lengths.sum()

    longest_segment = segment_lengths.max()

    minimum_reach = torch.relu(
        longest_segment
        - (
            maximum_reach
            - longest_segment
        )
    )

    root_to_target = (
        target_position
        - root_position
    )

    requested_distance = (
        torch.linalg.vector_norm(
            root_to_target
        )
    )

    original_direction = (
        original_points[-1]
        - root_position
    )

    target_direction = unit_vector_or_fallback(
        root_to_target,
        original_direction,
    )

    minimum_distance = (
        minimum_reach + epsilon
    )

    maximum_distance = (
        maximum_reach - epsilon
    )

    clamped_distance = torch.minimum(
        torch.maximum(
            requested_distance,
            minimum_distance,
        ),
        maximum_distance,
    )

    reachable_target = (
        root_position
        + target_direction
        * clamped_distance
    )

    points = list(
        original_points.unbind(dim=0)
    )

    ELBOW_INDEX = 2

    original_forearm_direction = (
        original_points[ELBOW_INDEX + 1]
        - original_points[ELBOW_INDEX]
    )

    for _ in range(iterations):
        # Backward pass:
        # start at the hand and work toward the root.
        backward_points = [None] * len(points)

        backward_points[-1] = reachable_target

        for index in range(
            len(points) - 2,
            -1,
            -1,
        ):
            direction = (
                points[index]
                - backward_points[index + 1]
            )

            fallback = (
                original_points[index]
                - original_points[index + 1]
            )

            direction = unit_vector_or_fallback(
                direction,
                fallback,
            )

            backward_points[index] = (
                backward_points[index + 1]
                + direction
                * segment_lengths[index]
            )

        # Forward pass:
        # restore the fixed root and work toward the hand.
        forward_points = [None] * len(points)

        forward_points[0] = root_position


        for index in range(
            len(points) - 1
        ):
            direction = (
                backward_points[index + 1]
                - forward_points[index]
            )

            fallback = (
                original_points[index + 1]
                - original_points[index]
            )

            direction = unit_vector_or_fallback(
                direction,
                fallback,
            )

            if index == ELBOW_INDEX:
                direction = ensure_realistic_elbow(
                    upper_arm_pos=forward_points[index - 1],
                    elbow_pos=forward_points[index],
                    original_forearm_direction=(
                        original_forearm_direction
                    ),
                    proposed_direction=direction,
                    min_theta=min_elbow_angle,
                    max_theta=max_elbow_angle,
                    epsilon=epsilon,
                )

            forward_points[index + 1] = (
                forward_points[index]
                + direction
                * segment_lengths[index]
            )

        points = forward_points

    return torch.stack(points)

def project_joint_positions_to_chest_plane_torch(
    joint_position_features: torch.Tensor,
    *,
    joint_indices: dict[str, int],
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiably project the 21 XYZ joint positions onto the same
    chest/shoulder plane used by construct_pose_img_from_joint_feature_vector.

    Parameters
    ----------
    joint_position_features:
        Tensor with shape (B, 63), containing 21 contiguous XYZ positions.
    joint_indices:
        Mapping from joint name to its index in the 21-joint ordering.

    Returns
    -------
    torch.Tensor
        Flattened projected coordinates with shape (B, 42), ordered as
        [joint0_u, joint0_v, joint1_u, joint1_v, ...].

    Notes
    -----
    Only the continuous projection math is reproduced here. Pixel rounding
    and OpenCV drawing from the pose-image helper are intentionally omitted,
    because keypoints_projections uses only the continuous projected joints.
    """
    if joint_position_features.ndim != 2:
        raise ValueError(
            "joint_position_features must have shape (B, 63), "
            f"received {tuple(joint_position_features.shape)}."
        )

    if int(joint_position_features.shape[1]) != JOINT_POSITION_FEATURE_COUNT:
        raise ValueError(
            f"Expected {JOINT_POSITION_FEATURE_COUNT} joint-position values, "
            f"received {int(joint_position_features.shape[1])}."
        )

    required = {"chest", "left_shoulder", "right_shoulder"}
    missing = sorted(required.difference(joint_indices))
    if missing:
        raise ValueError(
            "Chest-plane projection requires these joints: "
            + ", ".join(missing)
        )

    positions = joint_position_features.reshape(
        joint_position_features.shape[0],
        -1,
        JOINT_POSITION_DIMS,
    )

    chest = positions[:, joint_indices["chest"], :]
    right_shoulder = positions[:, joint_indices["right_shoulder"], :]
    left_shoulder = positions[:, joint_indices["left_shoulder"], :]

    normal_raw = torch.linalg.cross(
        right_shoulder - chest,
        left_shoulder - chest,
        dim=1,
    )
    normal_norm = torch.linalg.vector_norm(
        normal_raw,
        dim=1,
        keepdim=True,
    )

    plane_normal = normal_raw / normal_norm.clamp_min(epsilon)

    u_raw = left_shoulder - right_shoulder
    u_norm = torch.linalg.vector_norm(
        u_raw,
        dim=1,
        keepdim=True,
    )

    u = u_raw / u_norm.clamp_min(epsilon)

    v_raw = torch.linalg.cross(
        plane_normal,
        u,
        dim=1,
    )
    v_norm = torch.linalg.vector_norm(
        v_raw,
        dim=1,
        keepdim=True,
    )

    v = v_raw / v_norm.clamp_min(epsilon)

    # Match construct_pose_img_from_joint_feature_vector: keep positive v
    # approximately aligned with world +Y. The sign decision depends only on
    # the fixed chest/shoulder frame in the current counterfactual setup.
    world_up = torch.tensor(
        [0.0, 1.0, 0.0],
        dtype=positions.dtype,
        device=positions.device,
    ).view(1, 3)
    v_dot_up = (v * world_up).sum(dim=1, keepdim=True)
    v = torch.where(v_dot_up < 0.0, -v, v)

    relative_positions = positions - chest.unsqueeze(1)
    projected_u = (relative_positions * u.unsqueeze(1)).sum(dim=2)
    projected_v = (relative_positions * v.unsqueeze(1)).sum(dim=2)

    projected = torch.stack(
        (projected_u, projected_v),
        dim=2,
    )

    return projected.reshape(projected.shape[0], -1)


def regenerate_resnet_candidate_features(
    candidate_joint_position_features: torch.Tensor,
    fixed_head_orientation_features: torch.Tensor,
    *,
    model: QuestInferenceModel,
) -> torch.Tensor:
    """
    Regenerate the nondifferentiable pose image and its ResNet embedding for
    a candidate pose, then return the complete classifier feature tensor.

    This path intentionally detaches the candidate joints because the current
    pose-image renderer uses NumPy/OpenCV and is not differentiable. The
    ResNet counterfactual optimizer therefore uses SPSA for the six hand
    variables instead of autograd through this function.
    """
    candidate_joint_array = (
        candidate_joint_position_features[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    fixed_head_orientation_array = (
        fixed_head_orientation_features[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Match get_joint_samples(): the pose-image helper receives the joint XYZ
    # block followed by head roll/pitch/yaw. The helper ignores the final
    # three head-orientation values when constructing the projection image.
    candidate_joint_head_array = np.concatenate(
        [candidate_joint_array, fixed_head_orientation_array],
        axis=0,
    ).astype(np.float32)

    _, pose_img = construct_pose_img_from_joint_feature_vector(
        candidate_joint_head_array
    )
    resnet_embedding = get_resnet_encoder().predict(
        pose_img
    )["embedding"]

    full_features = np.concatenate(
        [
            candidate_joint_array,
            fixed_head_orientation_array,
            np.asarray(
                resnet_embedding,
                dtype=np.float32,
            ).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)

    return model.make_feature_tensor(
        full_features,
        require_single=True,
    )


def reconstruct_arms_from_hand_perturbation(
    *,
    original_joint_features: torch.Tensor,
    hand_perturbation: torch.Tensor,
    joint_indices: dict[str, int],
    object_arm: str = "both",
    fabrik_iterations: int = 4,
    latency_tracker: Optional[LatencyTracker] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply hand perturbations and reconstruct only the selected arm(s).

    object_arm="both":
        hand_perturbation =
            [left_dx, left_dy, left_dz,
             right_dx, right_dy, right_dz]

    object_arm="left" or "right":
        hand_perturbation = [dx, dy, dz]
        for only the selected arm. The opposite arm remains exactly as
        observed in original_joint_features.

    Returns:
        reconstructed 63-value joint-position tensor
        reachability loss for the selected arm(s)

    Derived projection/ResNet features are not stored here. They are rebuilt
    from the reconstructed joints by find_minimal_perturbation for every
    classifier candidate.
    """
    object_arm = str(object_arm).strip().lower()
    if object_arm not in ("both", "left", "right"):
        raise ValueError(
            "object_arm must be one of 'both', 'left', or 'right', "
            f"got {object_arm!r}."
        )

    expected_perturbation_width = 6 if object_arm == "both" else 3
    if hand_perturbation.shape != (1, expected_perturbation_width):
        raise ValueError(
            "hand_perturbation must have shape "
            f"(1, {expected_perturbation_width}) when "
            f"object_arm={object_arm!r}, received "
            f"{tuple(hand_perturbation.shape)}."
        )

    if original_joint_features.shape != (1, JOINT_POSITION_FEATURE_COUNT):
        raise ValueError(
            "original_joint_features must have shape "
            f"(1, {JOINT_POSITION_FEATURE_COUNT}), received "
            f"{tuple(original_joint_features.shape)}."
        )

    def get_position(
        features: torch.Tensor,
        joint_name: str,
    ) -> torch.Tensor:
        start = joint_indices[joint_name] * JOINT_POSITION_DIMS
        return features[
            0,
            start:start + JOINT_POSITION_DIMS,
        ]

    def set_position(
        features: torch.Tensor,
        joint_name: str,
        position: torch.Tensor,
    ) -> None:
        start = joint_indices[joint_name] * JOINT_POSITION_DIMS
        features[
            0,
            start:start + JOINT_POSITION_DIMS,
        ] = position

    candidate_joint_features = original_joint_features.clone()

    if object_arm == "both":
        active_sides = ("left", "right")
        hand_deltas = {
            "left": hand_perturbation[0, 0:3],
            "right": hand_perturbation[0, 3:6],
        }
    else:
        active_sides = (object_arm,)
        hand_deltas = {
            object_arm: hand_perturbation[0, 0:3],
        }

    reachability_loss = torch.zeros(
        (),
        dtype=original_joint_features.dtype,
        device=original_joint_features.device,
    )

    for side in active_sides:
        chain_names = ARM_CHAINS[side]

        original_chain = torch.stack(
            [
                get_position(
                    original_joint_features,
                    joint_name,
                )
                for joint_name in chain_names
            ]
        )

        original_hand_position = original_chain[-1]
        requested_hand_target = (
            original_hand_position + hand_deltas[side]
        )

        segment_lengths = torch.linalg.vector_norm(
            original_chain[1:] - original_chain[:-1],
            dim=1,
        )
        maximum_reach = segment_lengths.sum()
        longest_segment = segment_lengths.max()
        minimum_reach = torch.relu(
            longest_segment
            - (maximum_reach - longest_segment)
        )

        requested_distance = torch.linalg.vector_norm(
            requested_hand_target - original_chain[0]
        )
        too_far = torch.relu(
            requested_distance - maximum_reach
        )
        too_close = torch.relu(
            minimum_reach - requested_distance
        )
        reachability_loss = (
            reachability_loss
            + too_far.square()
            + too_close.square()
        )

        with measure_latency(
            latency_tracker,
            "fabrik_solve",
            device=original_joint_features.device,
        ):
            solved_chain = solve_fabrik_chain(
                original_points=original_chain,
                target_position=requested_hand_target,
                iterations=fabrik_iterations,
            )

        # The root stays fixed. Replace the remaining chain positions with
        # the FABRIK solution.
        for joint_name, solved_position in zip(
            chain_names[1:],
            solved_chain[1:],
        ):
            set_position(
                candidate_joint_features,
                joint_name,
                solved_position,
            )

        solved_hand_position = solved_chain[-1]
        actual_hand_displacement = (
            solved_hand_position - original_hand_position
        )

        # Palm and wrist twist retain their position relative to the hand.
        for attached_joint in HAND_ATTACHED_JOINTS[side]:
            if attached_joint not in joint_indices:
                continue

            original_attached_position = get_position(
                original_joint_features,
                attached_joint,
            )
            set_position(
                candidate_joint_features,
                attached_joint,
                original_attached_position
                + actual_hand_displacement,
            )

    return candidate_joint_features, reachability_loss


def find_minimal_perturbation(
    model: QuestInferenceModel,
    joint_feat_vec: np.ndarray,
    target_intent: bool,
    *,
    features_type: Optional[str] = None,
    original_probability: Optional[float] = None,
    classification_weight: float,
    reachability_weight: float,
    cross_body_weight: float = 1000.0,
    cross_body_margin: float = 0.05,
    forward_extension_weight: float = 1000.0,
    forward_extension_min: float = 0.20,
    require_forward_extension_for_success: bool = True,
    max_iterations: int = 1000,
    bin_search_iterations: int = 10,
    step_size: float = 0.01,
    probability_margin: float = 1e-4,
    object_arm: str = "both",
    robustness_enabled: bool = True,
    robustness_radius: float = 0.015,
    robustness_required_fraction: float = 5.0 / 6.0,
    robustness_neighbor_margin: float = 0.0,
    latency_tracker: Optional[LatencyTracker] = None,
) -> dict[str, Any]:
    """
    Find minimal hand-position changes that produce the requested
    classification while regenerating all joint-derived features.

    object_arm controls which hand(s) may be changed:

        "both"  -> optimize six values: left XYZ + right XYZ
        "left"  -> optimize three values: left XYZ only
        "right" -> optimize three values: right XYZ only

    The selected arm(s) are reconstructed with FABRIK for every candidate.
    Any unselected arm remains exactly as observed in the original pose. The
    classifier input is then regenerated from that reconstructed full-body pose:

        keypoints_projections:
            joints + fixed head RPY -> differentiable chest-plane projection
            -> classifier

        keypoints_resnet:
            joints + fixed head RPY -> pose image -> ResNet embedding
            -> classifier

    The projections path remains fully differentiable and therefore uses
    ordinary autograd + Adam. The ResNet path contains NumPy/OpenCV raster
    operations, so it uses a two-evaluation SPSA gradient estimate for the
    active hand variables rather than expensive per-dimension finite differences.

    Returns a dictionary with:

        features:
            The feature vector to display/use as guidance. If the requested
            target probability is reached, this is the binary-searched minimal
            successful perturbation. If optimization does not reach the target,
            this is the best target-directed candidate encountered.

        reached_target:
            True only when the returned features satisfy the central
            threshold +/- probability_margin target AND the local robustness
            criterion.

        used_fallback:
            True when optimization exhausted max_iterations without reaching the
            target and the returned features are the best-progress fallback.

        made_progress:
            True when the returned probability moved toward target_intent relative
            to the original probability.

        original_probability:
            Classifier probability for the original input pose.

        final_probability:
            Classifier probability for the returned feature vector.

        target_probability:
            Required central-pose probability boundary after applying
            probability_margin.

        robust_success:
            True only when the returned successful counterfactual also passes
            the local hand-target neighborhood check. A best-progress fallback
            keeps reached_target=False and robust_success=False.

        robust_neighbor_probabilities:
            Handoff probabilities for deterministic +/-XYZ hand-target
            neighbors around the returned successful perturbation.

    Cross-body regularization adds one soft term to the optimization objective.
    It penalizes only additional movement of an active hand beyond the torso
    midline compared with that hand's original pose. A small cross_body_margin
    permits natural near-midline positioning.

    Forward-extension regularization adds a second semantic term for handoff
    targets. A torso-forward axis is derived from the fixed chest/shoulder
    geometry, and active hands are softly encouraged to lie at least
    forward_extension_min meters in front of the shoulder midpoint. Once that
    minimum is reached, the forward term becomes zero, so it does not reward
    arbitrarily large reaches. When require_forward_extension_for_success is
    True, the same minimum is also included in the success test so the later
    binary search cannot shrink a good forward-reaching solution back behind
    the semantic minimum. This forward rule is disabled for not-handoff targets.

    Robustness does NOT add another term to the optimization objective. The
    distance/classification/reachability/cross-body/forward objective is otherwise
    unchanged. Robustness only changes the acceptance test for a successful
    counterfactual: the
    center pose must satisfy threshold +/- probability_margin, and most small
    hand-target neighbors must remain on the target side of the ordinary model
    threshold (optionally shifted by robustness_neighbor_margin). Binary search
    uses the same acceptance test, preserving minimum-movement behavior.
    """
    if max_iterations <= 0:
        raise ValueError(
            f"max_iterations must be positive, got {max_iterations}."
        )

    if step_size <= 0.0:
        raise ValueError(
            f"step_size must be positive, got {step_size}."
        )

    if classification_weight < 0.0:
        raise ValueError(
            "classification_weight must be nonnegative."
        )

    if reachability_weight < 0.0:
        raise ValueError(
            "reachability_weight must be nonnegative."
        )

    if cross_body_weight < 0.0:
        raise ValueError(
            "cross_body_weight must be nonnegative."
        )

    if cross_body_margin < 0.0:
        raise ValueError(
            "cross_body_margin must be nonnegative."
        )

    if forward_extension_weight < 0.0:
        raise ValueError(
            "forward_extension_weight must be nonnegative."
        )

    if forward_extension_min < 0.0:
        raise ValueError(
            "forward_extension_min must be nonnegative."
        )

    if probability_margin < 0.0:
        raise ValueError(
            "probability_margin must be nonnegative."
        )

    if robustness_radius < 0.0:
        raise ValueError(
            "robustness_radius must be nonnegative."
        )

    if not (0.0 < robustness_required_fraction <= 1.0):
        raise ValueError(
            "robustness_required_fraction must be in (0, 1]."
        )

    if robustness_neighbor_margin < 0.0:
        raise ValueError(
            "robustness_neighbor_margin must be nonnegative."
        )

    object_arm = str(object_arm).strip().lower()
    if object_arm not in ("both", "left", "right"):
        raise ValueError(
            "object_arm must be one of 'both', 'left', or 'right', "
            f"got {object_arm!r}."
        )

    active_sides = ("left", "right") if object_arm == "both" else (object_arm,)

    # MLP compatibility rule:
    # use the exact feature branch from the old working MLP implementation.
    #
    # TabM may use an explicitly supplied runtime feature type; if omitted,
    # infer it from the checkpoint input size.
    if isinstance(model, QuestHandIntentEstInference):
        resolved_features_type = WEIGHT_SEARCH_FEATURES_TYPE

        if features_type is not None:
            requested_features_type = str(features_type).strip().lower()

            if requested_features_type != resolved_features_type:
                raise ValueError(
                    "MLP feature-mode mismatch: the old working optimizer "
                    f"uses {resolved_features_type!r}, but the server passed "
                    f"{requested_features_type!r}."
                )
    else:
        if features_type is not None:
            resolved_features_type = str(features_type).strip().lower()
        elif int(model.input_dim) == (
            BASE_POSE_FEATURE_COUNT + PROJECTED_FEATURE_COUNT
        ):
            resolved_features_type = "keypoints_projections"
        elif int(model.input_dim) == (
            BASE_POSE_FEATURE_COUNT + 512
        ):
            resolved_features_type = "keypoints_resnet"
        else:
            raise ValueError(
                "Could not infer TabM feature type from input_dim="
                f"{model.input_dim}."
            )

    if resolved_features_type not in (
        "keypoints_projections",
        "keypoints_resnet",
    ):
        raise ValueError(
            "Unsupported Quest feature type: "
            f"{resolved_features_type!r}."
        )

    feature_array = np.asarray(
        joint_feat_vec,
        dtype=np.float32,
    )

    if feature_array.ndim != 1:
        raise ValueError(
            "joint_feat_vec must be a single 1D feature vector, "
            f"but received shape {feature_array.shape}."
        )

    if feature_array.shape[0] < BASE_POSE_FEATURE_COUNT:
        raise ValueError(
            "joint_feat_vec must contain at least 63 joint-position values "
            "followed by 3 head roll/pitch/yaw values, "
            f"but received only {feature_array.shape[0]} total values."
        )

    expected_joint_position_count = (
        len(model.joint_order) * JOINT_POSITION_DIMS
    )
    if expected_joint_position_count != JOINT_POSITION_FEATURE_COUNT:
        raise ValueError(
            f"Expected the first {JOINT_POSITION_FEATURE_COUNT} values "
            "to contain exactly one XYZ position for every joint in "
            f"model.joint_order, but model.joint_order contains "
            f"{len(model.joint_order)} joints "
            f"({expected_joint_position_count} XYZ values)."
        )

    if int(model.input_dim) != int(feature_array.shape[0]):
        raise ValueError(
            f"Model expects {model.input_dim} input features, but the supplied "
            f"feature vector contains {feature_array.shape[0]}."
        )

    original_features = model.make_feature_tensor(
        feature_array,
        require_single=True,
    ).detach()
    original_joint_features = original_features[
        :, :JOINT_POSITION_FEATURE_COUNT
    ]
    fixed_head_orientation_features = original_features[
        :,
        JOINT_POSITION_FEATURE_COUNT:BASE_POSE_FEATURE_COUNT,
    ]

    if fixed_head_orientation_features.shape != (
        1,
        HEAD_ORIENTATION_FEATURE_COUNT,
    ):
        raise ValueError(
            "Expected exactly three head-orientation features immediately "
            "after the 63 joint XYZ values."
        )

    threshold = float(model.threshold)
    target_intent = bool(target_intent)

    joint_indices = {
        joint_name: joint_index
        for joint_index, joint_name in enumerate(model.joint_order)
    }

    required_joint_names: set[str] = set()
    for side in active_sides:
        required_joint_names.update(ARM_CHAINS[side])
    required_joint_names.update(
        {"chest", "left_shoulder", "right_shoulder"}
    )

    missing_joints = sorted(
        required_joint_names.difference(joint_indices)
    )
    if missing_joints:
        raise ValueError(
            "The following required joints are absent from the model's "
            "joint order: " + ", ".join(missing_joints)
        )

    def get_original_joint_position(joint_name: str) -> torch.Tensor:
        start = joint_indices[joint_name] * JOINT_POSITION_DIMS
        return original_joint_features[
            0,
            start:start + JOINT_POSITION_DIMS,
        ]

    # Torso-local lateral axis. Positive points toward the person's left
    # shoulder, so a left hand has a positive lateral coordinate when it is on
    # its natural side of the torso, while a right hand has a negative one.
    left_shoulder_position = get_original_joint_position("left_shoulder")
    right_shoulder_position = get_original_joint_position("right_shoulder")
    shoulder_midpoint = 0.5 * (
        left_shoulder_position + right_shoulder_position
    )
    torso_lateral_axis = F.normalize(
        left_shoulder_position - right_shoulder_position,
        p=2,
        dim=0,
        eps=1e-8,
    )

    # Torso-local forward axis. With the Quest/Unity joint convention used by
    # this project, cross(right_shoulder - chest, left_shoulder - chest) points
    # out through the front of the torso. This removes global yaw from the
    # semantic rule: the axis turns with the person.
    chest_position = get_original_joint_position("chest")
    torso_forward_raw = torch.linalg.cross(
        right_shoulder_position - chest_position,
        left_shoulder_position - chest_position,
        dim=0,
    )

    # Degenerate chest/shoulder geometry is unlikely, but use the shoulder axis
    # and world-up direction as a stable fallback instead of producing NaNs.
    world_up = torch.tensor(
        [0.0, 1.0, 0.0],
        dtype=original_joint_features.dtype,
        device=original_joint_features.device,
    )
    torso_forward_fallback = torch.linalg.cross(
        world_up,
        torso_lateral_axis,
        dim=0,
    )
    torso_forward_axis = unit_vector_or_fallback(
        torso_forward_raw,
        torso_forward_fallback,
    ).detach()

    # The semantic forward rule is only meaningful when optimizing toward a
    # handoff. Setting forward_extension_weight=0 disables both the soft term
    # and the optional success guard without changing any other behavior.
    forward_extension_enabled = bool(
        target_intent
        and forward_extension_weight > 0.0
        and forward_extension_min > 0.0
    )

    original_hand_positions = {
        side: get_original_joint_position(f"{side}_hand")
        for side in active_sides
    }

    def cross_body_loss_from_perturbation(
        perturbation: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize only *additional* cross-body hand displacement.

        A hand may move freely toward the torso midline and up to
        cross_body_margin meters beyond it. If the original hand already lies
        farther across the body than that, the existing amount is treated as
        the baseline and is not penalized. Only making that crossing worse
        contributes loss.
        """
        loss = torch.zeros(
            (),
            dtype=perturbation.dtype,
            device=perturbation.device,
        )

        if object_arm == "both":
            hand_deltas = {
                "left": perturbation[0, 0:3],
                "right": perturbation[0, 3:6],
            }
        else:
            hand_deltas = {
                object_arm: perturbation[0, 0:3],
            }

        margin = torch.as_tensor(
            cross_body_margin,
            dtype=perturbation.dtype,
            device=perturbation.device,
        )

        for side in active_sides:
            original_hand = original_hand_positions[side]
            candidate_hand = original_hand + hand_deltas[side]

            original_lateral = torch.dot(
                original_hand - shoulder_midpoint,
                torso_lateral_axis,
            )
            candidate_lateral = torch.dot(
                candidate_hand - shoulder_midpoint,
                torso_lateral_axis,
            )

            if side == "left":
                original_crossing = torch.relu(
                    -original_lateral - margin
                )
                candidate_crossing = torch.relu(
                    -candidate_lateral - margin
                )
            else:
                original_crossing = torch.relu(
                    original_lateral - margin
                )
                candidate_crossing = torch.relu(
                    candidate_lateral - margin
                )

            additional_crossing = torch.relu(
                candidate_crossing - original_crossing
            )
            loss = loss + additional_crossing.square()

        return loss

    def forward_extension_values_from_perturbation(
        perturbation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Signed active-hand distances along torso forward from shoulders."""
        if object_arm == "both":
            hand_deltas = {
                "left": perturbation[0, 0:3],
                "right": perturbation[0, 3:6],
            }
        else:
            hand_deltas = {
                object_arm: perturbation[0, 0:3],
            }

        return {
            side: torch.dot(
                original_hand_positions[side]
                + hand_deltas[side]
                - shoulder_midpoint,
                torso_forward_axis,
            )
            for side in active_sides
        }

    def forward_extension_loss_from_perturbation(
        perturbation: torch.Tensor,
    ) -> torch.Tensor:
        """Softly penalize handoff hands that remain too close to the torso."""
        loss = torch.zeros(
            (),
            dtype=perturbation.dtype,
            device=perturbation.device,
        )

        if not forward_extension_enabled:
            return loss

        minimum = torch.as_tensor(
            forward_extension_min,
            dtype=perturbation.dtype,
            device=perturbation.device,
        )

        for extension in forward_extension_values_from_perturbation(
            perturbation
        ).values():
            deficit = torch.relu(minimum - extension)
            loss = loss + deficit.square()

        return loss

    def forward_extension_satisfied(
        perturbation: torch.Tensor,
    ) -> bool:
        """Whether the semantic handoff-forward minimum is satisfied."""
        if (
            not forward_extension_enabled
            or not require_forward_extension_for_success
        ):
            return True

        with torch.no_grad():
            values = forward_extension_values_from_perturbation(
                perturbation
            )
            return all(
                float(value.item()) + 1e-6
                >= float(forward_extension_min)
                for value in values.values()
            )

    def forward_extension_diagnostics(
        perturbation: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            return {
                side: float(value.item())
                for side, value in (
                    forward_extension_values_from_perturbation(perturbation)
                    .items()
                )
            }

    def build_projection_features(
        candidate_joint_features: torch.Tensor,
        *,
        track_latency: bool = True,
    ) -> torch.Tensor:
        tracker = latency_tracker if track_latency else None
        with measure_latency(
            tracker,
            "projection_regeneration",
            device=model.device,
        ):
            projected_features = (
                project_joint_positions_to_chest_plane_torch(
                    candidate_joint_features,
                    joint_indices=joint_indices,
                )
            )

        candidate_features = torch.cat(
            (
                candidate_joint_features,
                fixed_head_orientation_features,
                projected_features,
            ),
            dim=1,
        )

        if int(candidate_features.shape[1]) != model.input_dim:
            raise ValueError(
                "Regenerated keypoints_projections candidate contains "
                f"{int(candidate_features.shape[1])} features, but the model "
                f"expects {model.input_dim}. Expected a 63-value joint block, "
                f"3 head roll/pitch/yaw values, plus a "
                f"{PROJECTED_FEATURE_COUNT}-value projection block."
            )

        return candidate_features

    def build_resnet_features(
        candidate_joint_features: torch.Tensor,
        *,
        track_latency: bool = True,
    ) -> torch.Tensor:
        tracker = latency_tracker if track_latency else None
        with measure_latency(
            tracker,
            "resnet_feature_regeneration",
            device=model.device,
        ):
            candidate_features = regenerate_resnet_candidate_features(
                candidate_joint_features,
                fixed_head_orientation_features,
                model=model,
            )

        return candidate_features

    def reconstruct_joints(
        perturbation: torch.Tensor,
        *,
        track_latency: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tracker = latency_tracker if track_latency else None
        with measure_latency(
            tracker,
            "arm_reconstruction",
            device=model.device,
        ):
            return reconstruct_arms_from_hand_perturbation(
                original_joint_features=original_joint_features,
                hand_perturbation=perturbation,
                joint_indices=joint_indices,
                object_arm=object_arm,
                fabrik_iterations=10,
                latency_tracker=tracker,
            )

    def reconstruct_candidate(
        perturbation: torch.Tensor,
        *,
        track_latency: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_joint_features, reachability_loss = reconstruct_joints(
            perturbation,
            track_latency=track_latency,
        )

        if resolved_features_type == "keypoints_projections":
            candidate_features = build_projection_features(
                candidate_joint_features,
                track_latency=track_latency,
            )
        else:
            candidate_features = build_resnet_features(
                candidate_joint_features,
                track_latency=track_latency,
            )

        return (
            candidate_features,
            candidate_joint_features,
            reachability_loss,
        )

    def classify_candidate(
        candidate_features: torch.Tensor,
        *,
        track_latency: bool = True,
    ) -> torch.Tensor:
        tracker = latency_tracker if track_latency else None
        with measure_latency(
            tracker,
            "classifier_forward",
            device=model.device,
        ):
            # TabM returns k raw logits per sample. Convert every member logit
            # to a probability first, then average member probabilities.
            if isinstance(model, QuestHandIntentTabMEstInference):
                logits = model.model(
                    candidate_features
                )

                if logits.ndim == 2:
                    logits = logits.unsqueeze(-1)

                if (
                    logits.ndim != 3
                    or int(logits.shape[-1]) != 1
                ):
                    raise ValueError(
                        "Expected Quest TabM output with shape "
                        "[batch, k, 1] or [batch, k], got "
                        f"{tuple(logits.shape)}."
                    )

                return (
                    torch.sigmoid(logits)
                    .mean(dim=1)
                    .reshape(-1)[0]
                )

            # EXACT old working MLP expression.
            return model.model(
                candidate_features
            ).reshape(-1)[0]

    # Verify that the differentiable projection reproduces the exact feature
    # representation already present in the original sample. This catches
    # ordering/layout mismatches before optimization begins.
    if resolved_features_type == "keypoints_projections":
        expected_input_dim = (
            BASE_POSE_FEATURE_COUNT
            + PROJECTED_FEATURE_COUNT
        )
        if model.input_dim != expected_input_dim:
            raise ValueError(
                "keypoints_projections expects model input dimension "
                f"{expected_input_dim} (63 joint XYZ + 3 head RPY + "
                "42 projected values), "
                f"but this model expects {model.input_dim}."
            )
        
        if VERIFY_PROJECTION_FEATURES:
            with torch.inference_mode():
                regenerated_original_projection = (
                    project_joint_positions_to_chest_plane_torch(
                        original_joint_features,
                        joint_indices=joint_indices,
                    )
                )
                supplied_original_projection = original_features[
                    :, BASE_POSE_FEATURE_COUNT:
                ]
                if not torch.allclose(
                    regenerated_original_projection,
                    supplied_original_projection,
                    rtol=1e-4,
                    atol=1e-5,
                ):
                    max_abs_difference = float(
                        (
                            regenerated_original_projection
                            - supplied_original_projection
                        ).abs().max().item()
                    )
                    raise ValueError(
                        "Differentiable chest-plane projection does not match the "
                        "projection features supplied with the original sample. "
                        "This indicates a feature-ordering or projection-definition "
                        "mismatch. Maximum absolute difference="
                        f"{max_abs_difference:.6g}."
                    )
                
    if original_probability is None:
        with measure_latency(
            latency_tracker,
            "classifier_forward",
            device=model.device,
        ):
            with torch.inference_mode():
                original_probability = float(
                    model.forward_feature_probabilities(
                        original_features
                    )[0].item()
                )

    # Convert probability_margin into the actual probability boundary that a
    # robust counterfactual must satisfy. For a handoff target this is
    # threshold + margin; for a not-handoff target it is threshold - margin.
    if target_intent:
        target_probability = min(
            threshold + probability_margin,
            1.0,
        )
    else:
        target_probability = max(
            threshold - probability_margin,
            0.0,
        )

    # The central pose retains the existing probability-margin requirement.
    # Nearby hand-target variants only need to remain on the correct side of
    # the ordinary classifier threshold by default. This is intentionally a
    # weaker criterion than the central margin so robustness does not force
    # unnecessarily large human movements.
    if target_intent:
        robust_neighbor_target_probability = min(
            threshold + robustness_neighbor_margin,
            1.0,
        )
    else:
        robust_neighbor_target_probability = max(
            threshold - robustness_neighbor_margin,
            0.0,
        )

    # Optimize six values for both arms, or three values for one selected arm.
    # both:  [left_dx, left_dy, left_dz, right_dx, right_dy, right_dz]
    # single arm: [dx, dy, dz]
    perturbation_width = 6 if object_arm == "both" else 3

    def evaluate_local_robustness(
        perturbation: torch.Tensor,
        *,
        central_probability: float,
        track_latency: bool = True,
    ) -> dict[str, Any]:
        """Evaluate deterministic +/-XYZ hand-target neighbors.

        For one active hand this evaluates six neighbors. For both active hands
        this evaluates twelve neighbors, changing one hand coordinate at a time.
        Each neighbor is reconstructed through the same FABRIK + derived-feature
        pipeline as every other classifier candidate.
        """
        central_success = probability_meets_target_class(
            central_probability,
            target_intent,
            target_probability,
        )

        if not robustness_enabled or robustness_radius == 0.0:
            return {
                "success": bool(central_success),
                "central_success": bool(central_success),
                "neighbor_probabilities": [],
                "neighbor_success_count": 0,
                "neighbor_count": 0,
                "neighbor_success_fraction": 1.0,
                "required_neighbor_count": 0,
            }

        neighbor_probabilities: list[float] = []
        tracker = latency_tracker if track_latency else None

        with measure_latency(
            tracker,
            "robustness_check",
            device=model.device,
        ):
            with torch.inference_mode():
                base_perturbation = perturbation.detach()

                for dimension in range(perturbation_width):
                    for direction in (-1.0, 1.0):
                        neighbor_perturbation = (
                            base_perturbation.clone()
                        )
                        neighbor_perturbation[0, dimension] += (
                            direction * robustness_radius
                        )

                        neighbor_features, _, _ = (
                            reconstruct_candidate(
                                neighbor_perturbation,
                                track_latency=False,
                            )
                        )
                        neighbor_probability = float(
                            classify_candidate(
                                neighbor_features,
                                track_latency=False,
                            ).item()
                        )
                        neighbor_probabilities.append(
                            neighbor_probability
                        )

        neighbor_success_count = sum(
            1
            for probability_value in neighbor_probabilities
            if probability_meets_target_class(
                probability_value,
                target_intent,
                robust_neighbor_target_probability,
            )
        )
        neighbor_count = len(neighbor_probabilities)
        required_neighbor_count = int(
            np.ceil(
                robustness_required_fraction * neighbor_count
                - 1e-12
            )
        )
        neighbor_success_fraction = (
            neighbor_success_count / neighbor_count
            if neighbor_count
            else 1.0
        )

        return {
            "success": bool(
                central_success
                and neighbor_success_count >= required_neighbor_count
            ),
            "central_success": bool(central_success),
            "neighbor_probabilities": neighbor_probabilities,
            "neighbor_success_count": int(neighbor_success_count),
            "neighbor_count": int(neighbor_count),
            "neighbor_success_fraction": float(
                neighbor_success_fraction
            ),
            "required_neighbor_count": int(
                required_neighbor_count
            ),
        }

    def candidate_meets_success(
        perturbation: torch.Tensor,
        central_probability: float,
        *,
        track_latency: bool = True,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Avoid the extra neighbor evaluations until the center pose has already
        # reached the existing margin-shifted probability target.
        if not probability_meets_target_class(
            central_probability,
            target_intent,
            target_probability,
        ):
            return False, None

        # The forward term is a soft optimization bias, but the binary-search
        # minimizer below does not optimize the loss. When requested, guard the
        # same semantic minimum here so binary search cannot erase the forward
        # extension after a good solution has been found.
        if not forward_extension_satisfied(perturbation):
            return False, None

        robustness = evaluate_local_robustness(
            perturbation,
            central_probability=central_probability,
            track_latency=track_latency,
        )
        return bool(robustness["success"]), robustness

    zero_hand_perturbation = torch.zeros(
        (1, perturbation_width),
        dtype=original_features.dtype,
        device=model.device,
    )

    # No counterfactual is required only if the original pose satisfies BOTH
    # the existing central margin and the new local-stability criterion.
    original_success, original_robustness = candidate_meets_success(
        zero_hand_perturbation,
        original_probability,
    )
    if original_success:
        assert original_robustness is not None
        return {
            "features": feature_array.copy(),
            "reached_target": True,
            "used_fallback": False,
            "made_progress": False,
            "original_probability": original_probability,
            "final_probability": original_probability,
            "target_probability": target_probability,
            "robust_success": True,
            "robustness_enabled": bool(robustness_enabled),
            "robustness_radius": float(robustness_radius),
            "robustness_required_fraction": float(
                robustness_required_fraction
            ),
            "robust_neighbor_target_probability": float(
                robust_neighbor_target_probability
            ),
            "robust_neighbor_probabilities": list(
                original_robustness["neighbor_probabilities"]
            ),
            "robust_neighbor_success_count": int(
                original_robustness["neighbor_success_count"]
            ),
            "robust_neighbor_count": int(
                original_robustness["neighbor_count"]
            ),
            "robust_neighbor_success_fraction": float(
                original_robustness["neighbor_success_fraction"]
            ),
            "robust_required_neighbor_count": int(
                original_robustness["required_neighbor_count"]
            ),
            "forward_extension_enabled": bool(forward_extension_enabled),
            "forward_extension_weight": float(forward_extension_weight),
            "forward_extension_min": float(forward_extension_min),
            "require_forward_extension_for_success": bool(
                require_forward_extension_for_success
            ),
            "torso_forward_axis": [
                float(value)
                for value in torso_forward_axis.detach().cpu().tolist()
            ],
            "forward_extensions": forward_extension_diagnostics(
                zero_hand_perturbation
            ),
            "forward_extension_satisfied": bool(
                forward_extension_satisfied(zero_hand_perturbation)
            ),
        }

    hand_perturbation = torch.zeros(
        (1, perturbation_width),
        dtype=original_features.dtype,
        device=model.device,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam(
        [hand_perturbation],
        lr=step_size,
    )

    def objective_from_probability(
        probability: torch.Tensor,
        perturbation: torch.Tensor,
        reachability_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Counterfactual objective.

        Use binary cross-entropy rather than a squared hinge directly on the
        sigmoid probability.  For a sigmoid classifier, BCE supplies a much
        healthier gradient when the current prediction is very confident; a
        squared probability-space penalty contains an extra p*(1-p) factor and
        can effectively stall near probabilities of 0 or 1.

        Success is determined separately using target_probability, which is
        model.threshold shifted by probability_margin toward the target class.
        BCE is only the optimization surrogate used to reach that robust target.
        """
        distance_squared = perturbation.square().sum()
        cross_body_loss = cross_body_loss_from_perturbation(
            perturbation
        )
        forward_extension_loss = (
            forward_extension_loss_from_perturbation(
                perturbation
            )
        )

        classification_target = torch.ones_like(probability) if target_intent else torch.zeros_like(probability)
        classification_loss = F.binary_cross_entropy(
            probability,
            classification_target,
        )

        loss = (
            distance_squared
            + classification_weight * classification_loss
            + reachability_weight * reachability_loss
            + cross_body_weight * cross_body_loss
            + forward_extension_weight * forward_extension_loss
        )
        return loss, distance_squared

    def evaluate_resnet_objective_no_grad(
        perturbation: torch.Tensor,
    ) -> tuple[float, float, torch.Tensor]:
        with torch.inference_mode():
            (
                candidate_features,
                _,
                reachability_loss,
            ) = reconstruct_candidate(
                perturbation,
                track_latency=False,
            )
            probability = classify_candidate(
                candidate_features,
                track_latency=False,
            )
            loss, distance_squared = objective_from_probability(
                probability,
                perturbation,
                reachability_loss,
            )

        return (
            float(loss.item()),
            float(probability.item()),
            distance_squared.detach(),
        )

    best_hand_perturbation: Optional[torch.Tensor] = None
    best_distance_squared = float("inf")
    best_robustness: Optional[dict[str, Any]] = None
    last_probability = original_probability

    # Keep the best candidate seen even if the requested target boundary is never
    # reached. This lets the caller display useful intermediate guidance instead
    # of receiving an all-or-nothing optimization failure. Zero perturbation is
    # the initial fallback, so if no evaluated candidate improves on the original
    # pose, the original pose is returned with made_progress=False.
    best_probability_toward_target = original_probability
    best_progress_perturbation = torch.zeros_like(
        hand_perturbation
    ).detach().clone()

    def update_best_progress(
        probability_value: float,
        perturbation_value: torch.Tensor,
    ) -> None:
        nonlocal best_probability_toward_target
        nonlocal best_progress_perturbation

        if target_intent:
            improved = (
                probability_value > best_probability_toward_target
            )
        else:
            improved = (
                probability_value < best_probability_toward_target
            )

        if improved:
            best_probability_toward_target = probability_value
            best_progress_perturbation = (
                perturbation_value.detach().clone()
            )

    with measure_latency(
        latency_tracker,
        "optimization_loop",
        device=model.device,
    ):
        for _ in range(max_iterations):
            optimizer.zero_grad(set_to_none=True)

            if resolved_features_type == "keypoints_projections":
                (
                    candidate_features,
                    _,
                    reachability_loss,
                ) = reconstruct_candidate(
                    hand_perturbation
                )
                probability = classify_candidate(
                    candidate_features
                )
                loss, distance_squared = objective_from_probability(
                    probability,
                    hand_perturbation,
                    reachability_loss,
                )

                evaluated_probability = float(
                    probability.detach().item()
                )
                evaluated_distance_squared = float(
                    distance_squared.detach().item()
                )

                last_probability = evaluated_probability
                update_best_progress(
                    evaluated_probability,
                    hand_perturbation,
                )

                candidate_success, candidate_robustness = (
                    candidate_meets_success(
                        hand_perturbation.detach(),
                        evaluated_probability,
                    )
                )
                if candidate_success:
                    best_distance_squared = evaluated_distance_squared
                    best_hand_perturbation = (
                        hand_perturbation.detach().clone()
                    )
                    best_robustness = candidate_robustness
                    break

                with measure_latency(
                    latency_tracker,
                    "optimizer_step",
                    device=model.device,
                ):
                    loss.backward()
                    if hand_perturbation.grad is None:
                        raise RuntimeError(
                            "Counterfactual gradient is missing for the hand perturbation."
                        )
                    if not bool(torch.isfinite(hand_perturbation.grad).all()):
                        raise RuntimeError(
                            "Counterfactual gradient contains NaN or infinite values."
                        )
                    optimizer.step()

            else:
                # The pose-image rasterizer uses NumPy/OpenCV, so gradients
                # cannot flow from the ResNet branch to the active hand variables.
                # SPSA estimates the full objective gradient with only two
                # regenerated candidate evaluations, independent of dimension.
                (
                    current_loss,
                    evaluated_probability,
                    distance_squared,
                ) = evaluate_resnet_objective_no_grad(
                    hand_perturbation.detach()
                )
                del current_loss

                evaluated_distance_squared = float(
                    distance_squared.item()
                )
                last_probability = evaluated_probability
                update_best_progress(
                    evaluated_probability,
                    hand_perturbation,
                )

                candidate_success, candidate_robustness = (
                    candidate_meets_success(
                        hand_perturbation.detach(),
                        evaluated_probability,
                    )
                )
                if candidate_success:
                    best_distance_squared = evaluated_distance_squared
                    best_hand_perturbation = (
                        hand_perturbation.detach().clone()
                    )
                    best_robustness = candidate_robustness
                    break

                delta = torch.empty_like(
                    hand_perturbation
                ).bernoulli_(0.5).mul_(2.0).sub_(1.0)

                plus = (
                    hand_perturbation.detach()
                    + RESNET_SPSA_EPSILON * delta
                )
                minus = (
                    hand_perturbation.detach()
                    - RESNET_SPSA_EPSILON * delta
                )

                plus_loss, _, _ = evaluate_resnet_objective_no_grad(plus)
                minus_loss, _, _ = evaluate_resnet_objective_no_grad(minus)

                gradient_scale = (
                    plus_loss - minus_loss
                ) / (2.0 * RESNET_SPSA_EPSILON)

                hand_perturbation.grad = (
                    gradient_scale * delta
                ).detach()

                with measure_latency(
                    latency_tracker,
                    "optimizer_step",
                    device=model.device,
                ):
                    optimizer.step()

    if best_hand_perturbation is None:
        # Adam/SPSA performs optimizer.step() after the iteration's candidate was
        # evaluated. Evaluate the final optimizer state once so a useful last step
        # is not discarded merely because max_iterations was reached.
        with torch.inference_mode():
            final_attempt_features, _, _ = reconstruct_candidate(
                hand_perturbation.detach()
            )
            final_attempt_probability = float(
                classify_candidate(
                    final_attempt_features
                ).item()
            )

        last_probability = final_attempt_probability
        update_best_progress(
            final_attempt_probability,
            hand_perturbation,
        )

        # It is possible for the final post-step candidate to reach the requested
        # boundary. In that case treat it as a genuine success and continue into
        # the normal binary-search minimization path.
        final_attempt_success, final_attempt_robustness = (
            candidate_meets_success(
                hand_perturbation.detach(),
                final_attempt_probability,
            )
        )
        if final_attempt_success:
            best_hand_perturbation = (
                hand_perturbation.detach().clone()
            )
            best_robustness = final_attempt_robustness
        else:
            # No fully successful counterfactual was found. Return the candidate
            # that moved the model farthest toward the target class so it can be
            # displayed as intermediate guidance. On the next request, the user's
            # newly observed physical pose becomes the next optimization start.
            with torch.inference_mode():
                fallback_features, _, _ = reconstruct_candidate(
                    best_progress_perturbation
                )
                fallback_probability = float(
                    classify_candidate(
                        fallback_features
                    ).item()
                )

            if target_intent:
                made_progress = (
                    fallback_probability > original_probability
                )
            else:
                made_progress = (
                    fallback_probability < original_probability
                )

            return {
                "features": (
                    fallback_features[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ),
                "reached_target": False,
                "used_fallback": True,
                "made_progress": made_progress,
                "original_probability": original_probability,
                "final_probability": fallback_probability,
                "target_probability": target_probability,
                "robust_success": False,
                "robustness_enabled": bool(robustness_enabled),
                "robustness_radius": float(robustness_radius),
                "robustness_required_fraction": float(
                    robustness_required_fraction
                ),
                "robust_neighbor_target_probability": float(
                    robust_neighbor_target_probability
                ),
                "robust_neighbor_probabilities": [],
                "robust_neighbor_success_count": 0,
                "robust_neighbor_count": 0,
                "robust_neighbor_success_fraction": 0.0,
                "robust_required_neighbor_count": 0,
                "forward_extension_enabled": bool(forward_extension_enabled),
                "forward_extension_weight": float(forward_extension_weight),
                "forward_extension_min": float(forward_extension_min),
                "require_forward_extension_for_success": bool(
                    require_forward_extension_for_success
                ),
                "torso_forward_axis": [
                    float(value)
                    for value in torso_forward_axis.detach().cpu().tolist()
                ],
                "forward_extensions": forward_extension_diagnostics(
                    best_progress_perturbation
                ),
                "forward_extension_satisfied": bool(
                    forward_extension_satisfied(best_progress_perturbation)
                ),
            }

    # Reduce the successful movement along the line from zero to the first
    # successful perturbation. Every binary-search candidate regenerates the
    # derived projection/ResNet features before classification.
    low = 0.0
    high = 1.0

    with measure_latency(
        latency_tracker,
        "binary_search",
        device=model.device,
    ):
        with torch.inference_mode():
            for _ in range(bin_search_iterations):
                scale = (low + high) / 2.0
                scaled_hand_perturbation = (
                    scale * best_hand_perturbation
                )

                candidate_features, _, _ = reconstruct_candidate(
                    scaled_hand_perturbation
                )
                probability = float(
                    classify_candidate(
                        candidate_features
                    ).item()
                )

                scaled_success, _ = candidate_meets_success(
                    scaled_hand_perturbation,
                    probability,
                )
                if scaled_success:
                    high = scale
                else:
                    low = scale

            final_hand_perturbation = (
                high * best_hand_perturbation
            )
            final_features, _, _ = reconstruct_candidate(
                final_hand_perturbation
            )
            final_probability = float(
                classify_candidate(
                    final_features
                ).item()
            )

            final_success, final_robustness = candidate_meets_success(
                final_hand_perturbation,
                final_probability,
            )

            if not final_success:
                final_features, _, _ = reconstruct_candidate(
                    best_hand_perturbation
                )
                fallback_probability = float(
                    classify_candidate(
                        final_features
                    ).item()
                )
                fallback_success, fallback_robustness = (
                    candidate_meets_success(
                        best_hand_perturbation,
                        fallback_probability,
                    )
                )

                if not fallback_success:
                    # This should be extremely rare because the stored candidate
                    # already satisfied both the central and robustness criteria
                    # when it was recorded. Treat it as best-effort guidance.
                    if target_intent:
                        made_progress = (
                            fallback_probability > original_probability
                        )
                    else:
                        made_progress = (
                            fallback_probability < original_probability
                        )

                    return {
                        "features": (
                            final_features[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        ),
                        "reached_target": False,
                        "used_fallback": True,
                        "made_progress": made_progress,
                        "original_probability": original_probability,
                        "final_probability": fallback_probability,
                        "target_probability": target_probability,
                        "robust_success": False,
                        "robustness_enabled": bool(robustness_enabled),
                        "robustness_radius": float(robustness_radius),
                        "robustness_required_fraction": float(
                            robustness_required_fraction
                        ),
                        "robust_neighbor_target_probability": float(
                            robust_neighbor_target_probability
                        ),
                        "robust_neighbor_probabilities": [],
                        "robust_neighbor_success_count": 0,
                        "robust_neighbor_count": 0,
                        "robust_neighbor_success_fraction": 0.0,
                        "robust_required_neighbor_count": 0,
                        "forward_extension_enabled": bool(forward_extension_enabled),
                        "forward_extension_weight": float(forward_extension_weight),
                        "forward_extension_min": float(forward_extension_min),
                        "require_forward_extension_for_success": bool(
                            require_forward_extension_for_success
                        ),
                        "torso_forward_axis": [
                            float(value)
                            for value in torso_forward_axis.detach().cpu().tolist()
                        ],
                        "forward_extensions": forward_extension_diagnostics(
                            best_hand_perturbation
                        ),
                        "forward_extension_satisfied": bool(
                            forward_extension_satisfied(best_hand_perturbation)
                        ),
                    }

                final_probability = fallback_probability
                final_robustness = fallback_robustness
                final_hand_perturbation = best_hand_perturbation

            if final_robustness is None:
                # Defensive fallback; in normal operation a successful candidate
                # always has robustness diagnostics from candidate_meets_success.
                final_robustness = best_robustness

    if target_intent:
        made_progress = final_probability > original_probability
    else:
        made_progress = final_probability < original_probability

    if final_robustness is None:
        final_robustness = evaluate_local_robustness(
            final_hand_perturbation,
            central_probability=final_probability,
        )

    return {
        "features": (
            final_features[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),
        "reached_target": True,
        "used_fallback": False,
        "made_progress": made_progress,
        "original_probability": original_probability,
        "final_probability": final_probability,
        "target_probability": target_probability,
        "robust_success": bool(final_robustness["success"]),
        "robustness_enabled": bool(robustness_enabled),
        "robustness_radius": float(robustness_radius),
        "robustness_required_fraction": float(
            robustness_required_fraction
        ),
        "robust_neighbor_target_probability": float(
            robust_neighbor_target_probability
        ),
        "robust_neighbor_probabilities": list(
            final_robustness["neighbor_probabilities"]
        ),
        "robust_neighbor_success_count": int(
            final_robustness["neighbor_success_count"]
        ),
        "robust_neighbor_count": int(
            final_robustness["neighbor_count"]
        ),
        "robust_neighbor_success_fraction": float(
            final_robustness["neighbor_success_fraction"]
        ),
        "robust_required_neighbor_count": int(
            final_robustness["required_neighbor_count"]
        ),
        "forward_extension_enabled": bool(forward_extension_enabled),
        "forward_extension_weight": float(forward_extension_weight),
        "forward_extension_min": float(forward_extension_min),
        "require_forward_extension_for_success": bool(
            require_forward_extension_for_success
        ),
        "torso_forward_axis": [
            float(value)
            for value in torso_forward_axis.detach().cpu().tolist()
        ],
        "forward_extensions": forward_extension_diagnostics(
            final_hand_perturbation
        ),
        "forward_extension_satisfied": bool(
            forward_extension_satisfied(final_hand_perturbation)
        ),
    }


def _safe_mean(values: list[float]) -> Optional[float]:
    return (
        float(np.mean(values))
        if values
        else None
    )


def _safe_median(values: list[float]) -> Optional[float]:
    return (
        float(np.median(values))
        if values
        else None
    )


def _safe_max(values: list[float]) -> Optional[float]:
    return (
        float(np.max(values))
        if values
        else None
    )


def summarize_latency_snapshots(
    snapshots: list[
        dict[str, dict[str, Union[float, int]]]
    ],
    *,
    evaluation_total_ms: float,
) -> dict[str, Any]:
    component_sample_totals: dict[str, list[float]] = {}
    component_totals: dict[str, float] = {}
    component_call_counts: dict[str, int] = {}

    for snapshot in snapshots:
        for name, component in snapshot.items():
            total_ms = float(component["total_ms"])
            call_count = int(component["call_count"])

            component_sample_totals.setdefault(
                name,
                [],
            ).append(total_ms)
            component_totals[name] = (
                component_totals.get(name, 0.0)
                + total_ms
            )
            component_call_counts[name] = (
                component_call_counts.get(name, 0)
                + call_count
            )

    components: dict[str, dict[str, Any]] = {}

    for name in sorted(component_totals):
        sample_values = component_sample_totals[name]
        total_ms = component_totals[name]
        call_count = component_call_counts[name]

        components[name] = {
            "total_ms": float(total_ms),
            "call_count": int(call_count),
            "sample_count": len(sample_values),
            "mean_per_call_ms": (
                float(total_ms / call_count)
                if call_count
                else None
            ),
            "mean_per_sample_ms": _safe_mean(
                sample_values
            ),
            "median_per_sample_ms": _safe_median(
                sample_values
            ),
            "max_per_sample_ms": _safe_max(
                sample_values
            ),
        }

    return {
        "evaluation_total_ms": float(evaluation_total_ms),
        "attempted_sample_count": len(snapshots),
        "components": components,
        "timing_method": (
            "CUDA events for CUDA tensor regions; synchronized "
            "wall-clock timing for CPU/MPS and whole-sample timing."
        ),
        "instrumentation_note": (
            "Timing instrumentation adds a small amount of overhead, "
            "especially because every FABRIK invocation is measured."
        ),
    }


def evaluate_counterfactual_weights(
    *,
    model: QuestInferenceModel,
    rows: list[dict[str, Any]],
    classification_weight: float,
    reachability_weight: float,
    max_iterations: int,
    step_size: float,
    probability_margin: float,
    bone_relative_tolerance: float,
    bone_absolute_tolerance: float,
    target_mode: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Evaluate one weight pair over every supplied valid joint sample.
    """
    threshold = float(model.threshold)
    evaluation_started_at = perf_counter()

    sample_results: list[dict[str, Any]] = []
    sample_latency_snapshots: list[
        dict[str, dict[str, Union[float, int]]]
    ] = []
    perturbation_distances: list[float] = []
    probability_changes: list[float] = []
    maximum_bone_changes: list[float] = []
    mean_bone_changes: list[float] = []

    success_count = 0
    bone_valid_success_count = 0
    bone_violation_count = 0
    to_handoff_attempts = 0
    to_handoff_successes = 0
    to_not_handoff_attempts = 0
    to_not_handoff_successes = 0

    for sample_number, row in enumerate(rows, start=1):
        sample_started_at = perf_counter()
        latency_tracker = LatencyTracker()

        original_features = np.asarray(
            row["joint_features"],
            dtype=np.float32,
        )

        with measure_latency(
            latency_tracker,
            "original_inference",
            device=model.device,
        ):
            original_probability = float(
                model.predict_features(
                    original_features
                ).probability
            )

        target_intent = determine_target_intent(
            row=row,
            original_probability=original_probability,
            threshold=threshold,
            target_mode=target_mode,
        )

        if target_intent:
            to_handoff_attempts += 1
        else:
            to_not_handoff_attempts += 1

        result_record: dict[str, Any] = {
            "sample_index": int(row["index"]),
            "participant_id": row.get("participant_id"),
            "condition": row.get("condition"),
            "hand": row.get("hand"),
            "label_name": row.get("label_name"),
            "original_probability": original_probability,
            "target_intent": target_intent,
            "success": False,
            "bone_valid": False,
        }

        try:
            with measure_latency(
                latency_tracker,
                "minimal_perturbation",
                device=model.device,
            ):
                perturbation_result = (
                    find_minimal_perturbation(
                        model=model,
                        joint_feat_vec=original_features,
                        target_intent=target_intent,
                        classification_weight=(
                            classification_weight
                        ),
                        reachability_weight=(
                            reachability_weight
                        ),
                        forward_extension_weight=1000.0,
                        forward_extension_min=0.20,
                        require_forward_extension_for_success=True,
                        max_iterations=max_iterations,
                        step_size=step_size,
                        probability_margin=(
                            probability_margin
                        ),
                        latency_tracker=latency_tracker,
                    )
                )
                perturbed_features = np.asarray(
                    perturbation_result["features"],
                    dtype=np.float32,
                )

            with measure_latency(
                latency_tracker,
                "final_inference",
                device=model.device,
            ):
                final_probability = float(
                    model.predict_features(
                        perturbed_features
                    ).probability
                )

            # find_minimal_perturbation already evaluates success against the
            # margin-shifted target_probability. A fallback can cross the raw
            # classifier threshold without satisfying that requested margin, so
            # use the returned status rather than re-checking the raw threshold.
            success = bool(
                perturbation_result["reached_target"]
            )

            with measure_latency(
                latency_tracker,
                "feature_to_pose_conversion",
            ):
                original_joints = (
                    feature_vector_to_joint_poses(
                        original_features[
                            :JOINT_POSITION_FEATURE_COUNT
                        ]
                    )
                )
                perturbed_joints = (
                    feature_vector_to_joint_poses(
                        perturbed_features[
                            :JOINT_POSITION_FEATURE_COUNT
                        ]
                    )
                )

            with measure_latency(
                latency_tracker,
                "bone_validation",
            ):
                bone_statistics = (
                    skeleton_edge_length_statistics(
                        original_joints,
                        perturbed_joints,
                        relative_tolerance=(
                            bone_relative_tolerance
                        ),
                        absolute_tolerance=(
                            bone_absolute_tolerance
                        ),
                    )
                )

            perturbation_distance = float(
                np.linalg.norm(
                    perturbed_features[
                        :JOINT_POSITION_FEATURE_COUNT
                    ]
                    - original_features[
                        :JOINT_POSITION_FEATURE_COUNT
                    ]
                )
            )
            probability_change = abs(
                final_probability
                - original_probability
            )

            result_record.update(
                {
                    "final_probability": (
                        final_probability
                    ),
                    "success": success,
                    "used_fallback": bool(
                        perturbation_result["used_fallback"]
                    ),
                    "made_progress": bool(
                        perturbation_result["made_progress"]
                    ),
                    "target_probability": float(
                        perturbation_result["target_probability"]
                    ),
                    "bone_valid": (
                        not bone_statistics[
                            "has_violation"
                        ]
                    ),
                    "perturbation_l2": (
                        perturbation_distance
                    ),
                    "absolute_probability_change": (
                        probability_change
                    ),
                    "max_relative_bone_change": (
                        bone_statistics[
                            "max_relative_change"
                        ]
                    ),
                    "mean_relative_bone_change": (
                        bone_statistics[
                            "mean_relative_change"
                        ]
                    ),
                    "violating_edge_count": (
                        bone_statistics[
                            "violating_edge_count"
                        ]
                    ),
                    "violating_edges": (
                        bone_statistics[
                            "violating_edges"
                        ]
                    ),
                }
            )

            if success:
                success_count += 1
                perturbation_distances.append(
                    perturbation_distance
                )
                probability_changes.append(
                    probability_change
                )
                maximum_bone_changes.append(
                    float(
                        bone_statistics[
                            "max_relative_change"
                        ]
                    )
                )
                mean_bone_changes.append(
                    float(
                        bone_statistics[
                            "mean_relative_change"
                        ]
                    )
                )

                if target_intent:
                    to_handoff_successes += 1
                else:
                    to_not_handoff_successes += 1

                if bone_statistics["has_violation"]:
                    bone_violation_count += 1
                else:
                    bone_valid_success_count += 1

        except Exception as error:
            result_record["error"] = str(error)

        # Finalizing pending CUDA events synchronizes the sample, so
        # whole-sample wall-clock latency also includes GPU completion.
        latency_tracker.snapshot()
        latency_tracker.add_duration_ms(
            "sample_total",
            (perf_counter() - sample_started_at) * 1000.0,
        )
        latency_snapshot = latency_tracker.snapshot()

        result_record["latency_ms"] = latency_snapshot
        sample_latency_snapshots.append(latency_snapshot)
        sample_results.append(result_record)

        if verbose:
            status = (
                "valid"
                if (
                    result_record["success"]
                    and result_record["bone_valid"]
                )
                else (
                    "classification-only"
                    if result_record["success"]
                    else "failed"
                )
            )
            print(
                f"[{sample_number}/{len(rows)}] "
                f"sample={row['index']} "
                f"target={target_intent} "
                f"status={status} "
                f"{format_latency_snapshot(latency_snapshot)}"
            )

    attempted_count = len(rows)
    failed_count = (
        attempted_count - success_count
    )

    success_rate = (
        success_count / attempted_count
        if attempted_count
        else 0.0
    )
    bone_valid_success_rate = (
        bone_valid_success_count
        / attempted_count
        if attempted_count
        else 0.0
    )
    bone_violation_rate_among_successes = (
        bone_violation_count / success_count
        if success_count
        else 0.0
    )

    summary = {
        "classification_weight": float(
            classification_weight
        ),
        "reachability_weight": float(
            reachability_weight
        ),
        "attempted_count": attempted_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "success_rate": success_rate,
        "bone_valid_success_count": (
            bone_valid_success_count
        ),
        "bone_valid_success_rate": (
            bone_valid_success_rate
        ),
        "bone_violation_count_among_successes": (
            bone_violation_count
        ),
        "bone_violation_rate_among_successes": (
            bone_violation_rate_among_successes
        ),
        "mean_perturbation_l2": _safe_mean(
            perturbation_distances
        ),
        "median_perturbation_l2": _safe_median(
            perturbation_distances
        ),
        "max_perturbation_l2": _safe_max(
            perturbation_distances
        ),
        "mean_absolute_probability_change": (
            _safe_mean(probability_changes)
        ),
        "mean_max_relative_bone_change": (
            _safe_mean(maximum_bone_changes)
        ),
        "max_relative_bone_change": (
            _safe_max(maximum_bone_changes)
        ),
        "mean_relative_bone_change": (
            _safe_mean(mean_bone_changes)
        ),
        "to_handoff_attempts": (
            to_handoff_attempts
        ),
        "to_handoff_successes": (
            to_handoff_successes
        ),
        "to_handoff_success_rate": (
            to_handoff_successes
            / to_handoff_attempts
            if to_handoff_attempts
            else 0.0
        ),
        "to_not_handoff_attempts": (
            to_not_handoff_attempts
        ),
        "to_not_handoff_successes": (
            to_not_handoff_successes
        ),
        "to_not_handoff_success_rate": (
            to_not_handoff_successes
            / to_not_handoff_attempts
            if to_not_handoff_attempts
            else 0.0
        ),
    }

    summary["latency_ms"] = summarize_latency_snapshots(
        sample_latency_snapshots,
        evaluation_total_ms=(
            perf_counter() - evaluation_started_at
        ) * 1000.0,
    )

    return {
        "summary": summary,
        "samples": sample_results,
    }


def print_counterfactual_metrics(
    summary: dict[str, Any],
    *,
    heading: str,
) -> None:
    def percentage(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    def optional_float(
        value: Optional[float],
        decimals: int = 6,
    ) -> str:
        if value is None:
            return "n/a"
        return f"{value:.{decimals}f}"

    print()
    print("=" * 72)
    print(heading)
    print("=" * 72)
    print(
        "Weights: "
        f"classification={summary['classification_weight']}, "
        f"bone={summary['reachability_weight']}"
    )
    print(
        "Classification success: "
        f"{summary['success_count']}/"
        f"{summary['attempted_count']} "
        f"({percentage(summary['success_rate'])})"
    )
    print(
        "Classification + bone-valid success: "
        f"{summary['bone_valid_success_count']}/"
        f"{summary['attempted_count']} "
        f"({percentage(summary['bone_valid_success_rate'])})"
    )
    print(
        "Bone violations among classification successes: "
        f"{summary['bone_violation_count_among_successes']}/"
        f"{summary['success_count']} "
        f"({percentage(summary['bone_violation_rate_among_successes'])})"
    )
    print(
        "Perturbation L2 "
        f"(mean/median/max): "
        f"{optional_float(summary['mean_perturbation_l2'])} / "
        f"{optional_float(summary['median_perturbation_l2'])} / "
        f"{optional_float(summary['max_perturbation_l2'])}"
    )
    print(
        "Maximum relative edge change "
        f"(mean over successes / worst success): "
        f"{optional_float(summary['mean_max_relative_bone_change'])} / "
        f"{optional_float(summary['max_relative_bone_change'])}"
    )
    print(
        "Mean relative edge change: "
        f"{optional_float(summary['mean_relative_bone_change'])}"
    )
    print(
        "Mean absolute probability change: "
        f"{optional_float(summary['mean_absolute_probability_change'])}"
    )
    print(
        "Target handoff success: "
        f"{summary['to_handoff_successes']}/"
        f"{summary['to_handoff_attempts']} "
        f"({percentage(summary['to_handoff_success_rate'])})"
    )
    print(
        "Target not-handoff success: "
        f"{summary['to_not_handoff_successes']}/"
        f"{summary['to_not_handoff_attempts']} "
        f"({percentage(summary['to_not_handoff_success_rate'])})"
    )

    latency_metrics = summary.get("latency_ms", {})
    latency_components = latency_metrics.get("components", {})

    def print_latency_component(
        label: str,
        component_name: str,
    ) -> None:
        component = latency_components.get(component_name)
        if component is None:
            print(f"{label}: n/a")
            return

        print(
            f"{label} (mean/median/max ms per measured sample): "
            f"{optional_float(component['mean_per_sample_ms'], 3)} / "
            f"{optional_float(component['median_per_sample_ms'], 3)} / "
            f"{optional_float(component['max_per_sample_ms'], 3)}; "
            f"calls={component['call_count']}, "
            f"mean/call="
            f"{optional_float(component['mean_per_call_ms'], 3)} ms"
        )

    print(
        "Evaluation wall-clock latency: "
        f"{optional_float(latency_metrics.get('evaluation_total_ms'), 3)} ms"
    )
    print_latency_component(
        "Whole-sample latency",
        "sample_total",
    )
    print_latency_component(
        "Minimal-perturbation latency",
        "minimal_perturbation",
    )
    print_latency_component(
        "Optimization-loop latency",
        "optimization_loop",
    )
    print_latency_component(
        "FABRIK solve latency",
        "fabrik_solve",
    )
    print_latency_component(
        "Classifier-forward latency",
        "classifier_forward",
    )
    print_latency_component(
        "Binary-search latency",
        "binary_search",
    )
    print_latency_component(
        "Original-inference latency",
        "original_inference",
    )
    print_latency_component(
        "Final-inference latency",
        "final_inference",
    )
    print_latency_component(
        "Bone-validation latency",
        "bone_validation",
    )
    print("=" * 72)
    print()


def choose_best_weight_result(
    results: list[dict[str, Any]],
    *,
    minimum_success_rate: float,
    maximum_bone_violation_rate: float,
) -> tuple[dict[str, Any], bool]:
    """
    Prefer candidates meeting both configured requirements.

    Among eligible candidates, select the one with the smallest mean
    perturbation. If none are eligible, select the candidate with the
    highest bone-valid success rate, then classification success rate,
    then the lowest bone-violation rate and perturbation distance.
    """
    eligible = [
        result
        for result in results
        if (
            result["success_rate"]
            >= minimum_success_rate
            and result[
                "bone_violation_rate_among_successes"
            ]
            <= maximum_bone_violation_rate
        )
    ]

    def distance_or_infinity(
        result: dict[str, Any],
    ) -> float:
        distance = result[
            "mean_perturbation_l2"
        ]
        return (
            float(distance)
            if distance is not None
            else float("inf")
        )

    if eligible:
        best = min(
            eligible,
            key=lambda result: (
                distance_or_infinity(result),
                result[
                    "bone_violation_rate_among_successes"
                ],
                -result["success_rate"],
            ),
        )
        return best, True

    best = max(
        results,
        key=lambda result: (
            result[
                "bone_valid_success_rate"
            ],
            result["success_rate"],
            -result[
                "bone_violation_rate_among_successes"
            ],
            -distance_or_infinity(result),
        ),
    )

    return best, False


def tune_counterfactual_weights(
    *,
    model: QuestInferenceModel,
    rows: list[dict[str, Any]],
    counterfactual_config: dict[str, Any],
    verbose_samples: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    bool,
]:
    tuning_config = counterfactual_config["tuning"]

    classification_weights = [
        float(value)
        for value in tuning_config[
            "classification_weights"
        ]
    ]
    reachability_weights = [
        float(value)
        for value in tuning_config[
            "reachability_weights"
        ]
    ]

    all_summaries: list[dict[str, Any]] = []
    pair_count = (
        len(classification_weights)
        * len(reachability_weights)
    )
    pair_index = 0

    for classification_weight in classification_weights:
        for reachability_weight in reachability_weights:
            pair_index += 1

            print(
                f"\nWeight pair {pair_index}/{pair_count}: "
                f"classification={classification_weight}, "
                f"bone={reachability_weight}"
            )

            evaluation = evaluate_counterfactual_weights(
                model=model,
                rows=rows,
                classification_weight=(
                    classification_weight
                ),
                reachability_weight=(
                    reachability_weight
                ),
                max_iterations=int(
                    counterfactual_config[
                        "max_iterations"
                    ]
                ),
                step_size=float(
                    counterfactual_config[
                        "step_size"
                    ]
                ),
                probability_margin=float(
                    counterfactual_config[
                        "probability_margin"
                    ]
                ),
                bone_relative_tolerance=float(
                    counterfactual_config[
                        "bone_relative_tolerance"
                    ]
                ),
                bone_absolute_tolerance=float(
                    counterfactual_config[
                        "bone_absolute_tolerance"
                    ]
                ),
                target_mode=str(
                    counterfactual_config[
                        "target_mode"
                    ]
                ),
                verbose=verbose_samples,
            )

            summary = evaluation["summary"]
            all_summaries.append(summary)

            print_counterfactual_metrics(
                summary,
                heading=(
                    "Weight-pair evaluation"
                ),
            )

    best_summary, met_requirements = (
        choose_best_weight_result(
            all_summaries,
            minimum_success_rate=float(
                tuning_config[
                    "minimum_success_rate"
                ]
            ),
            maximum_bone_violation_rate=float(
                tuning_config[
                    "maximum_bone_violation_rate"
                ]
            ),
        )
    )

    return (
        best_summary,
        all_summaries,
        met_requirements,
    )


def save_selected_weights(
    *,
    config_path: Path,
    full_config: dict[str, Any],
    best_summary: dict[str, Any],
    met_requirements: bool,
    tuning_sample_count: int,
) -> None:
    counterfactual_config = full_config[
        "counterfactual"
    ]

    counterfactual_config[
        "classification_weight"
    ] = float(
        best_summary["classification_weight"]
    )
    counterfactual_config[
        "reachability_weight"
    ] = float(
        best_summary["reachability_weight"]
    )

    counterfactual_config["selection"] = {
        "selected_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "tuning_sample_count": (
            tuning_sample_count
        ),
        "met_configured_requirements": (
            met_requirements
        ),
        "metrics": {
            key: value
            for key, value in best_summary.items()
            if key not in {
                "classification_weight",
                "reachability_weight",
            }
        },
    }

    write_yaml(config_path, full_config)

    print(
        "Saved selected counterfactual weights to: "
        f"{config_path}"
    )


def save_metrics(
    *,
    path: Path,
    data: dict[str, Any],
) -> None:
    payload = {
        "generated_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        **data,
    }
    write_yaml(path, payload)
    print(f"Saved metrics to: {path}")



def get_visualization_output_path(
    *,
    base_path: Path,
    sample_index: int,
    multiple_outputs: bool,
) -> Path:
    """
    Use the requested path directly for one visualization. For multiple
    visualizations, append the sample index to the filename.
    """
    if not multiple_outputs:
        return base_path

    suffix = base_path.suffix or ".html"

    return base_path.with_name(
        f"{base_path.stem}_sample_{sample_index}{suffix}"
    )


def visualize_counterfactual_samples(
    *,
    model: QuestInferenceModel,
    rows: list[dict[str, Any]],
    sample_indices: list[int],
    counterfactual_config: dict[str, Any],
    visualization_output: Path,
) -> None:
    """
    Generate original-versus-counterfactual overlays for selected samples.
    """
    rows_by_index = {
        int(row["index"]): row
        for row in rows
    }

    requested_indices = list(
        dict.fromkeys(int(index) for index in sample_indices)
    )

    missing_indices = [
        index
        for index in requested_indices
        if index not in rows_by_index
    ]

    if missing_indices:
        available_indices = sorted(rows_by_index)

        raise ValueError(
            "The following visualization sample indices were not loaded: "
            f"{missing_indices}. Available loaded indices are: "
            f"{available_indices}."
        )

    multiple_outputs = len(requested_indices) > 1

    for sample_index in requested_indices:
        row = rows_by_index[sample_index]

        original_features = np.asarray(
            row["joint_features"],
            dtype=np.float32,
        )

        original_probability = float(
            model.predict_features(
                original_features
            ).probability
        )

        target_intent = determine_target_intent(
            row=row,
            original_probability=original_probability,
            threshold=float(model.threshold),
            target_mode=str(
                counterfactual_config[
                    "target_mode"
                ]
            ),
        )

        try:
            perturbation_result = find_minimal_perturbation(
                model=model,
                joint_feat_vec=original_features,
                target_intent=target_intent,
                classification_weight=float(
                    counterfactual_config[
                        "classification_weight"
                    ]
                ),
                reachability_weight=float(
                    counterfactual_config[
                        "reachability_weight"
                    ]
                ),
                forward_extension_weight=float(
                    counterfactual_config.get(
                        "forward_extension_weight",
                        1000.0,
                    )
                ),
                forward_extension_min=float(
                    counterfactual_config.get(
                        "forward_extension_min",
                        0.20,
                    )
                ),
                require_forward_extension_for_success=bool(
                    counterfactual_config.get(
                        "require_forward_extension_for_success",
                        True,
                    )
                ),
                max_iterations=int(
                    counterfactual_config[
                        "max_iterations"
                    ]
                ),
                step_size=float(
                    counterfactual_config[
                        "step_size"
                    ]
                ),
                probability_margin=float(
                    counterfactual_config[
                        "probability_margin"
                    ]
                ),
            )
            perturbed_features = np.asarray(
                perturbation_result["features"],
                dtype=np.float32,
            )
        except Exception as error:
            print(
                f"Could not generate perturbation for sample "
                f"{sample_index}: {error}"
            )
            continue

        final_probability = float(
            model.predict_features(
                perturbed_features
            ).probability
        )

        sample_output_path = get_visualization_output_path(
            base_path=visualization_output,
            sample_index=sample_index,
            multiple_outputs=multiple_outputs,
        )
        sample_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        condition = row["condition"]
        visualize_joint_features(
            feature_vector=(
                original_features[:JOINT_POSITION_FEATURE_COUNT]
            ),
            comparison_feature_vector=(
                perturbed_features[:JOINT_POSITION_FEATURE_COUNT]
            ),
            features_per_joint=JOINT_POSITION_DIMS,
            title=(
                f"Sample {sample_index}: original with sample condition {condition} and "
                f"counterfactual Quest poses "
                f"({original_probability:.4f} -> "
                f"{final_probability:.4f})"
            ),
            output_path=sample_output_path,
        )

        print(
            f"Saved sample {sample_index} visualization to: "
            f"{sample_output_path}"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quest-mlp-est-path",
        "--quest-estimator-path",
        dest="quest_estimator_path",
        type=str,
        required=True,
        help=(
            "Path to the trained Quest handoff-classification checkpoint. "
            "The legacy --quest-mlp-est-path name remains accepted."
        ),
    )
    parser.add_argument(
        "--quest-model-type",
        type=str,
        choices=["mlp", "tabm"],
        default="mlp",
        help=(
            "Quest estimator architecture used for evaluation/weight search. "
            "Defaults to mlp."
        ),
    )
    parser.add_argument(
        "--include-quest-joint-rotations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--participant",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hand",
        type=str,
        choices=["left", "right"],
        default=None,
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--tune-weights",
        action="store_true",
        help=(
            "Grid-search the configured classification and bone "
            "weights, save the selected pair, and then evaluate it."
        ),
    )
    parser.add_argument(
        "--max-tuning-samples",
        type=int,
        default=None,
        help=(
            "Optional cap on tuning samples. By default, every "
            "loaded sample is used."
        ),
    )
    parser.add_argument(
        "--verbose-samples",
        action="store_true",
        help=(
            "Print one status line for every sample and weight pair."
        ),
    )
    parser.add_argument(
        "--visualize-first-success",
        action="store_true",
        help=(
            "Generate an HTML overlay for the first successful "
            "counterfactual using the selected weights."
        ),
    )
    parser.add_argument(
        "--visualize-sample-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Generate original-versus-perturbed HTML visualizations "
            "for one or more loaded sample indices, for example: "
            "--visualize-sample-indices 0 4 12. These are the "
            "sample_index values shown in evaluation output."
        ),
    )

    parser.add_argument(
    "--visualization-output",
    type=Path,
    default=VISUALIZATION_DIR / "index.html",
    )

    parser.add_argument(
        "--handoff-config-path",
        type=Path,
        default=HANDOFF_CONFIG_PATH,
    )
    parser.add_argument(
        "--counterfactual-config-path",
        type=Path,
        default=COUNTERFACTUAL_CONFIG_PATH,
    )
    parser.add_argument(
        "--tuning-results-path",
        type=Path,
        default=(
            OUTPUT_DIR
            / "counterfactual_eval/counterfactual_weight_search_results.yaml"
        ),
    )
    parser.add_argument(
        "--evaluation-results-path",
        type=Path,
        default=(
            OUTPUT_DIR
            / "counterfactual_eval/counterfactual_evaluation_results.yaml"
        ),
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    handoff_config = load_yaml(
        args.handoff_config_path
    )
    counterfactual_full_config = (
        load_or_create_counterfactual_config(
            args.counterfactual_config_path
        )
    )
    counterfactual_config = (
        counterfactual_full_config[
            "counterfactual"
        ]
    )

    dataset_root = (
        REPO_ROOT
        / handoff_config["dataset"]["root"]
    )

    if args.quest_model_type == "tabm":
        model = QuestHandIntentTabMEstInference(
            checkpoint_path=args.quest_estimator_path,
        )
    else:
        model = QuestHandIntentEstInference(
            checkpoint_path=args.quest_estimator_path,
        )

    print(
        "Loaded Quest estimator for counterfactual evaluation: "
        f"type={args.quest_model_type}, "
        f"path={Path(args.quest_estimator_path).expanduser().resolve()}"
    )

    samples = discover_samples(
        dataset_root=dataset_root,
        participant_filter=args.participant,
        label_filter=args.label,
        condition_filter=args.condition,
        hand_filter=args.hand,
    )

    samples = samples[args.start_index:]

    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    (
        _,
        joint_rows,
        _,
        _,
        _,
    ) = get_joint_samples(
        samples,
        include_rotation=(
            args.include_quest_joint_rotations
        ),
    )

    valid_rows = [
        row
        for row in joint_rows
        if not row.get("skipped", False)
    ]

    if not valid_rows:
        raise RuntimeError(
            "No valid Quest joint samples were loaded."
        )

    if args.tune_weights:
        tuning_rows = valid_rows

        if args.max_tuning_samples is not None:
            tuning_rows = tuning_rows[
                : args.max_tuning_samples
            ]

        (
            best_summary,
            all_summaries,
            met_requirements,
        ) = tune_counterfactual_weights(
            model=model,
            rows=tuning_rows,
            counterfactual_config=(
                counterfactual_config
            ),
            verbose_samples=(
                args.verbose_samples
            ),
        )

        print_counterfactual_metrics(
            best_summary,
            heading=(
                "Selected counterfactual weights"
            ),
        )

        if not met_requirements:
            print(
                "WARNING: No weight pair met both configured "
                "selection requirements. The script selected "
                "the strongest fallback according to bone-valid "
                "success rate, classification success rate, "
                "bone violations, and perturbation distance."
            )

        save_selected_weights(
            config_path=(
                args.counterfactual_config_path
            ),
            full_config=(
                counterfactual_full_config
            ),
            best_summary=best_summary,
            met_requirements=met_requirements,
            tuning_sample_count=len(tuning_rows),
        )

        save_metrics(
            path=args.tuning_results_path,
            data={
                "selected": best_summary,
                "met_configured_requirements": (
                    met_requirements
                ),
                "weight_pair_results": (
                    all_summaries
                ),
            },
        )

        # Use the newly selected weights for the final evaluation.
        counterfactual_config[
            "classification_weight"
        ] = best_summary[
            "classification_weight"
        ]
        counterfactual_config[
            "reachability_weight"
        ] = best_summary[
            "reachability_weight"
        ]

    final_evaluation = (
        evaluate_counterfactual_weights(
            model=model,
            rows=valid_rows,
            classification_weight=float(
                counterfactual_config[
                    "classification_weight"
                ]
            ),
            reachability_weight=float(
                counterfactual_config[
                    "reachability_weight"
                ]
            ),
            max_iterations=int(
                counterfactual_config[
                    "max_iterations"
                ]
            ),
            step_size=float(
                counterfactual_config[
                    "step_size"
                ]
            ),
            probability_margin=float(
                counterfactual_config[
                    "probability_margin"
                ]
            ),
            bone_relative_tolerance=float(
                counterfactual_config[
                    "bone_relative_tolerance"
                ]
            ),
            bone_absolute_tolerance=float(
                counterfactual_config[
                    "bone_absolute_tolerance"
                ]
            ),
            target_mode=str(
                counterfactual_config[
                    "target_mode"
                ]
            ),
            verbose=args.verbose_samples,
        )
    )

    print_counterfactual_metrics(
        final_evaluation["summary"],
        heading=(
            "Overall counterfactual optimization metrics"
        ),
    )

    save_metrics(
        path=args.evaluation_results_path,
        data=final_evaluation,
    )

    visualization_sample_indices = list(
        args.visualize_sample_indices or []
    )

    if args.visualize_first_success:
        first_success = next(
            (
                sample_result
                for sample_result in (
                    final_evaluation["samples"]
                )
                if sample_result["success"]
            ),
            None,
        )

        if first_success is None:
            print(
                "No successful counterfactual was available "
                "for first-success visualization."
            )
        else:
            first_success_index = int(
                first_success["sample_index"]
            )

            if (
                first_success_index
                not in visualization_sample_indices
            ):
                visualization_sample_indices.append(
                    first_success_index
                )

    if visualization_sample_indices:
        visualize_counterfactual_samples(
            model=model,
            rows=valid_rows,
            sample_indices=(
                visualization_sample_indices
            ),
            counterfactual_config=(
                counterfactual_config
            ),
            visualization_output=(
                args.visualization_output
            ),
        )


if __name__ == "__main__":
    main()