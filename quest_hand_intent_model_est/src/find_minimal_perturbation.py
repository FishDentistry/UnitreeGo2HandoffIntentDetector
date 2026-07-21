from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from quest_hand_intent_model_est.src.quest_hand_int_inf_wrapper import (
    QuestHandIntentEstInference,
)
from quest_hand_intent_model_est.src.quest_joint_features import (
    SKELETON_EDGES,
    extract_quest_joint_features,
    feature_vector_to_joint_poses,
)
from shared.util.extract_all_samples import (
    Sample,
    discover_samples,
    get_sample_label_name,
    label_to_binary,
)
from shared.util.quest_joints_viz import visualize_joint_features


REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = REPO_ROOT / "quest_hand_intent_model_est" / "configs"

HANDOFF_CONFIG_PATH = CONFIG_DIR / "handoff_config.yaml"
COUNTERFACTUAL_CONFIG_PATH = CONFIG_DIR / "counterfactual_config.yaml"

SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR / "visualizations"

MODIFIABLE_JOINT_NAMES = {
    # "left_upper_arm",
    # "left_forearm",
    "left_hand",
    # "left_palm",
    # "right_upper_arm",
    # "right_forearm",
    "right_hand",
    # "right_palm",
}

IGNORED_SKELETON_JOINT_NAMES = {
    "left_wrist_twist",
    "right_wrist_twist",
}

COUNTERFACTUAL_SKELETON_EDGES = [
    (joint_a, joint_b)
    for joint_a, joint_b in SKELETON_EDGES
    if (
        joint_a not in IGNORED_SKELETON_JOINT_NAMES
        and joint_b not in IGNORED_SKELETON_JOINT_NAMES
    )
]

for replacement_edge in [
    ("left_forearm", "left_hand"),
    ("right_forearm", "right_hand"),
]:
    if replacement_edge not in COUNTERFACTUAL_SKELETON_EDGES:
        COUNTERFACTUAL_SKELETON_EDGES.append(
            replacement_edge
        )

DEFAULT_COUNTERFACTUAL_CONFIG: dict[str, Any] = {
    "counterfactual": {
        "classification_weight": 1000.0,
        "bone_length_weight": 1000.0,
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
            "bone_length_weights": [
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

            row = {
                "participant_id": sample.participant_id,
                "index": local_idx,
                "label_name": label_name,
                "label_binary": int(y),
                "condition": sample.condition,
                "hand": sample.hand,
                "joint_features": extract_quest_joint_features(
                    sample.joints_path,
                    include_rotations=include_rotation,
                ),
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


def check_skeleton_edge_lengths(
    original_joints: dict[str, dict[str, Any]],
    perturbed_joints: dict[str, dict[str, Any]],
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 1e-6,
) -> tuple[
    bool,
    list[tuple[str, str, float, float]],
]:
    """
    Preserve the original two-value interface while using the richer
    statistics function internally.
    """
    statistics = skeleton_edge_length_statistics(
        original_joints,
        perturbed_joints,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )

    violations = [
        (
            joint_a,
            joint_b,
            original_length,
            perturbed_length,
        )
        for (
            joint_a,
            joint_b,
            original_length,
            perturbed_length,
            _,
        ) in statistics["violating_edges"]
    ]

    return bool(statistics["has_violation"]), violations


def skeleton_edge_length_loss(
    original_features: torch.Tensor,
    perturbed_features: torch.Tensor,
    joint_order: list[str],
    skeleton_edges: list[tuple[str, str]],
    features_per_joint: int,
) -> torch.Tensor:
    """
    Penalize the mean squared relative change in each skeleton edge length.

    Both feature tensors must have shape (B, D).
    """
    loss = torch.zeros(
        (),
        dtype=perturbed_features.dtype,
        device=perturbed_features.device,
    )

    joint_indices = {
        joint_name: index
        for index, joint_name in enumerate(joint_order)
    }

    valid_edge_count = 0

    for joint_a, joint_b in skeleton_edges:
        if (
            joint_a not in joint_indices
            or joint_b not in joint_indices
        ):
            continue

        start_a = (
            joint_indices[joint_a]
            * features_per_joint
        )
        start_b = (
            joint_indices[joint_b]
            * features_per_joint
        )

        original_position_a = original_features[
            :, start_a : start_a + 3
        ]
        original_position_b = original_features[
            :, start_b : start_b + 3
        ]
        perturbed_position_a = perturbed_features[
            :, start_a : start_a + 3
        ]
        perturbed_position_b = perturbed_features[
            :, start_b : start_b + 3
        ]

        original_length = torch.linalg.vector_norm(
            original_position_a - original_position_b,
            dim=1,
        )
        perturbed_length = torch.linalg.vector_norm(
            perturbed_position_a - perturbed_position_b,
            dim=1,
        )

        relative_change = (
            perturbed_length - original_length
        ) / original_length.clamp_min(1e-6)

        loss = (
            loss
            + relative_change.square().mean()
        )
        valid_edge_count += 1

    if valid_edge_count == 0:
        return loss

    return loss / valid_edge_count


def find_minimal_perturbation(
    model: QuestHandIntentEstInference,
    joint_feat_vec: np.ndarray,
    target_intent: bool,
    *,
    classification_weight: float,
    bone_length_weight: float,
    max_iterations: int = 1000,
    step_size: float = 0.01,
    probability_margin: float = 1e-4,
) -> np.ndarray:
    """
    Find a small counterfactual perturbation of selected joint positions.

    Only the XYZ features belonging to MODIFIABLE_JOINT_NAMES may change.
    Rotations and all other joint features remain fixed.
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

    if bone_length_weight < 0.0:
        raise ValueError(
            "bone_length_weight must be nonnegative."
        )

    if probability_margin < 0.0:
        raise ValueError(
            "probability_margin must be nonnegative."
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

    if model.features_per_joint < 3:
        raise ValueError(
            "The model must contain at least three position features "
            "per joint."
        )

    original_features = model.make_feature_tensor(
        feature_array,
        require_single=True,
    ).detach()

    threshold = float(model.threshold)
    target_intent = bool(target_intent)

    with torch.inference_mode():
        original_probability = float(
            model.forward_feature_probabilities(
                original_features
            )[0].item()
        )

    if probability_meets_target_class(
        original_probability,
        target_intent,
        threshold,
    ):
        return feature_array.copy()

    missing_joints = sorted(
        MODIFIABLE_JOINT_NAMES.difference(
            model.joint_order
        )
    )

    if missing_joints:
        raise ValueError(
            "The following modifiable joints are absent from "
            "the model's joint order: "
            + ", ".join(missing_joints)
        )

    modifiable_feature_indices: list[int] = []

    for joint_index, joint_name in enumerate(
        model.joint_order
    ):
        if joint_name not in MODIFIABLE_JOINT_NAMES:
            continue

        feature_start = (
            joint_index
            * model.features_per_joint
        )

        modifiable_feature_indices.extend(
            [
                feature_start,
                feature_start + 1,
                feature_start + 2,
            ]
        )

    active_indices = torch.tensor(
        modifiable_feature_indices,
        dtype=torch.long,
        device=model.device,
    )

    perturbation = torch.zeros(
        (1, len(modifiable_feature_indices)),
        dtype=torch.float32,
        device=model.device,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam(
        [perturbation],
        lr=step_size,
    )

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

    best_perturbation: torch.Tensor | None = None
    best_distance_squared = float("inf")
    last_probability = original_probability

    for _ in range(max_iterations):
        optimizer.zero_grad()

        full_perturbation = torch.zeros_like(
            original_features
        ).scatter(
            dim=1,
            index=active_indices.unsqueeze(0),
            src=perturbation,
        )

        candidate_features = (
            original_features + full_perturbation
        )

        probability = model.forward_feature_probabilities(
            candidate_features
        )[0]

        distance_squared = perturbation.square().sum()

        if target_intent:
            classification_violation = torch.relu(
                target_probability - probability
            )
        else:
            classification_violation = torch.relu(
                probability - target_probability
            )

        classification_loss = (
            classification_violation.square()
        )

        bone_loss = skeleton_edge_length_loss(
            original_features=original_features,
            perturbed_features=candidate_features,
            joint_order=model.joint_order,
            skeleton_edges=COUNTERFACTUAL_SKELETON_EDGES,
            features_per_joint=model.features_per_joint,
        )

        loss = (
            distance_squared
            + classification_weight
            * classification_loss
            + bone_length_weight
            * bone_loss
        )

        # Save the exact perturbation whose probability and distance were
        # evaluated above. This avoids mixing pre-step metrics with a
        # post-step perturbation.
        with torch.no_grad():
            evaluated_probability = float(
                probability.item()
            )
            evaluated_distance_squared = float(
                distance_squared.item()
            )
            last_probability = evaluated_probability

            if (
                probability_meets_target_class(
                    evaluated_probability,
                    target_intent,
                    threshold,
                )
                and evaluated_distance_squared
                < best_distance_squared
            ):
                best_distance_squared = (
                    evaluated_distance_squared
                )
                best_perturbation = (
                    perturbation.detach().clone()
                )

        loss.backward()
        optimizer.step()

    if best_perturbation is None:
        raise RuntimeError(
            "Could not find a successful perturbation after "
            f"{max_iterations} iterations. "
            f"Original probability={original_probability:.6f}, "
            f"last probability={last_probability:.6f}, "
            f"threshold={threshold:.6f}, "
            f"target_intent={target_intent}, "
            f"classification_weight={classification_weight}, "
            f"bone_length_weight={bone_length_weight}."
        )

    # Reduce the successful perturbation along its direction. This is a
    # local heuristic and assumes the target class remains reachable along
    # this line segment.
    low = 0.0
    high = 1.0

    with torch.inference_mode():
        for _ in range(40):
            scale = (low + high) / 2.0
            scaled_perturbation = (
                scale * best_perturbation
            )

            full_perturbation = torch.zeros_like(
                original_features
            ).scatter(
                dim=1,
                index=active_indices.unsqueeze(0),
                src=scaled_perturbation,
            )

            candidate_features = (
                original_features
                + full_perturbation
            )

            probability = float(
                model.forward_feature_probabilities(
                    candidate_features
                )[0].item()
            )

            if probability_meets_target_class(
                probability,
                target_intent,
                threshold,
            ):
                high = scale
            else:
                low = scale

        final_active_perturbation = (
            high * best_perturbation
        )

        final_full_perturbation = torch.zeros_like(
            original_features
        ).scatter(
            dim=1,
            index=active_indices.unsqueeze(0),
            src=final_active_perturbation,
        )

        final_features = (
            original_features
            + final_full_perturbation
        )

        final_probability = float(
            model.forward_feature_probabilities(
                final_features
            )[0].item()
        )

        if not probability_meets_target_class(
            final_probability,
            target_intent,
            threshold,
        ):
            fallback_full_perturbation = (
                torch.zeros_like(
                    original_features
                ).scatter(
                    dim=1,
                    index=active_indices.unsqueeze(0),
                    src=best_perturbation,
                )
            )

            final_features = (
                original_features
                + fallback_full_perturbation
            )

    return (
        final_features[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def _safe_mean(values: list[float]) -> float | None:
    return (
        float(np.mean(values))
        if values
        else None
    )


def _safe_median(values: list[float]) -> float | None:
    return (
        float(np.median(values))
        if values
        else None
    )


def _safe_max(values: list[float]) -> float | None:
    return (
        float(np.max(values))
        if values
        else None
    )


def evaluate_counterfactual_weights(
    *,
    model: QuestHandIntentEstInference,
    rows: list[dict[str, Any]],
    classification_weight: float,
    bone_length_weight: float,
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

    sample_results: list[dict[str, Any]] = []
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
            perturbed_features = (
                find_minimal_perturbation(
                    model=model,
                    joint_feat_vec=original_features,
                    target_intent=target_intent,
                    classification_weight=classification_weight,
                    bone_length_weight=bone_length_weight,
                    max_iterations=max_iterations,
                    step_size=step_size,
                    probability_margin=probability_margin,
                )
            )

            final_probability = float(
                model.predict_features(
                    perturbed_features
                ).probability
            )

            success = probability_meets_target_class(
                final_probability,
                target_intent,
                threshold,
            )

            original_joints = (
                feature_vector_to_joint_poses(
                    original_features
                )
            )
            perturbed_joints = (
                feature_vector_to_joint_poses(
                    perturbed_features
                )
            )

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
                    perturbed_features
                    - original_features
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
                f"status={status}"
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
        "bone_length_weight": float(
            bone_length_weight
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
        value: float | None,
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
        f"bone={summary['bone_length_weight']}"
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
    model: QuestHandIntentEstInference,
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
    bone_length_weights = [
        float(value)
        for value in tuning_config[
            "bone_length_weights"
        ]
    ]

    all_summaries: list[dict[str, Any]] = []
    pair_count = (
        len(classification_weights)
        * len(bone_length_weights)
    )
    pair_index = 0

    for classification_weight in classification_weights:
        for bone_length_weight in bone_length_weights:
            pair_index += 1

            print(
                f"\nWeight pair {pair_index}/{pair_count}: "
                f"classification={classification_weight}, "
                f"bone={bone_length_weight}"
            )

            evaluation = evaluate_counterfactual_weights(
                model=model,
                rows=rows,
                classification_weight=(
                    classification_weight
                ),
                bone_length_weight=(
                    bone_length_weight
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
        "bone_length_weight"
    ] = float(
        best_summary["bone_length_weight"]
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
                "bone_length_weight",
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
    model: QuestHandIntentEstInference,
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
            perturbed_features = find_minimal_perturbation(
                model=model,
                joint_feat_vec=original_features,
                target_intent=target_intent,
                classification_weight=float(
                    counterfactual_config[
                        "classification_weight"
                    ]
                ),
                bone_length_weight=float(
                    counterfactual_config[
                        "bone_length_weight"
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

        visualize_joint_features(
            feature_vector=original_features,
            comparison_feature_vector=perturbed_features,
            features_per_joint=model.features_per_joint,
            title=(
                f"Sample {sample_index}: original and "
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
        type=str,
        required=True,
        help=(
            "Path to the trained handoff-classification model."
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
            CONFIG_DIR
            / "counterfactual_weight_search_results.yaml"
        ),
    )
    parser.add_argument(
        "--evaluation-results-path",
        type=Path,
        default=(
            CONFIG_DIR
            / "counterfactual_evaluation_results.yaml"
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

    model = QuestHandIntentEstInference(
        checkpoint_path=args.quest_mlp_est_path,
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
            "bone_length_weight"
        ] = best_summary[
            "bone_length_weight"
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
            bone_length_weight=float(
                counterfactual_config[
                    "bone_length_weight"
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