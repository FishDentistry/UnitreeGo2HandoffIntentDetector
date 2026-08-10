import argparse
import csv
import itertools
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from shared.util.extract_all_samples import (
    discover_samples,
    organize_samples,
    rows_to_xy,
)
from torch.utils.data import DataLoader, TensorDataset
from ..src.dino_detector import DINOObjectDetector
from ..src.rtmpose_keypoints import RTMPoseKeypointDetector
from ..src.rtmpose_headpose import RTMPoseHeadPoseEstimator
from ..src.hand_intent_tabm import HandIntentTabM
from ..src.resnet_encoder import ResNet18ImageEncoder
from ..src.dino_encoder import DINOv2ImageEncoder


FEATURES_TYPE = [
    "keypoints",
    "keypoints_headpose",
    "keypoints_headpose_dino",
    "keypoints_resnet",
    "keypoints_headpose_resnet",
]

MODEL_IMPL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    MODEL_IMPL_ROOT
    / "configs"
    / "handoff_config.yaml"
)


# ----------------------------------------------------------------------
# TabM tuning configuration
#
# These settings are used only when --tune-tabm is passed. Normal runs
# behave exactly as before.
#
# The search uses participant-level cross-validation, so a participant
# is never split between a tuning-training fold and its validation fold.
# No checkpoints are written during tuning.
# ----------------------------------------------------------------------
TABM_TUNE_NUM_TRIALS = 16
TABM_TUNE_NUM_FOLDS = 5

TABM_TUNE_SPACE = {
    "epochs": [20, 30, 50, 75],
    "batch_size": [16, 32, 64],
    "learning_rate": [
        5e-4,
        1e-3,
        2e-3,
        4e-3,
    ],
    "weight_decay": [
        0.0,
        1e-4,
        3e-4,
        1e-3,
    ],
    "k": [16, 24, 32, 48],
}


def make_json_serializable(value):
    """Convert common scientific-Python values into JSON-safe objects."""
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serializable(item)
            for item in value
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return make_json_serializable(
            value.tolist()
        )
    if isinstance(value, np.generic):
        return make_json_serializable(
            value.item()
        )
    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None
    return value


def sanitize_path_component(value):
    """Return a filesystem-safe, human-readable path component."""
    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        str(value).strip(),
    )
    cleaned = cleaned.strip("-._")
    return cleaned or "unspecified"


def build_model_variation_name(args):
    """Name a variation using only feature type and crop enablement."""
    parts = [
        f"features-{args.features_type}",
        f"crop-{args.crop_around_object}",
    ]
    return "__".join(
        sanitize_path_component(part)
        for part in parts
    )


def save_evaluation_results(
    *,
    lopo_evaluation,
    final_train_summary,
    args,
    run_id,
    variation_name,
    run_started_at,
    final_output_path,
    dataset_root,
    dataset_summary,
    tuning_summary=None,
):
    """Save evaluation files, overwriting the previous run of this variation."""
    eval_root = (
        Path(args.eval_output_dir)
        if args.eval_output_dir is not None
        else (
            MODEL_IMPL_ROOT
            / "outputs"
            / "eval_results"
            / "tabm"
        )
    )
    variation_dir = (
        eval_root
        / sanitize_path_component(
            variation_name
        )
    )
    variation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_at = datetime.now().astimezone()

    result_payload = {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "variation_name": variation_name,
            "started_at": run_started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        },
        "configuration": vars(args),
        "paths": {
            "dataset_root": str(
                dataset_root
            ),
            "model_weights": str(
                final_output_path
            ),
            "evaluation_directory": str(
                variation_dir
            ),
        },
        "dataset_summary": dataset_summary,
        "tabm_tuning": tuning_summary,
        "lopo_evaluation": lopo_evaluation,
        "final_training_summary": (
            final_train_summary
        ),
    }

    full_json_path = (
        variation_dir
        / "evaluation_results.json"
    )
    with full_json_path.open(
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(
            make_json_serializable(
                result_payload
            ),
            file_obj,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file_obj.write("\n")

    pooled = lopo_evaluation[
        "pooled_metrics"
    ]

    summary_row = {
        "run_id": run_id,
        "variation_name": variation_name,
        "features_type": args.features_type,
        "crop_around_object": bool(
            args.crop_around_object
        ),
        "decision_threshold": float(
            args.decision_threshold
        ),
        "tune_tabm": bool(
            args.tune_tabm
        ),
        "num_participants": (
            lopo_evaluation[
                "num_participants"
            ]
        ),
        "num_folds": (
            lopo_evaluation[
                "num_folds"
            ]
        ),
        "pooled_total": pooled.get(
            "total"
        ),
        "pooled_accuracy": pooled.get(
            "accuracy"
        ),
        "pooled_balanced_accuracy": (
            pooled.get(
                "balanced_accuracy"
            )
        ),
        "pooled_precision": pooled.get(
            "precision"
        ),
        "pooled_recall": pooled.get(
            "recall"
        ),
        "pooled_specificity": pooled.get(
            "specificity"
        ),
        "pooled_f1": pooled.get(
            "f1"
        ),
        "pooled_tn": pooled.get("tn"),
        "pooled_fp": pooled.get("fp"),
        "pooled_fn": pooled.get("fn"),
        "pooled_tp": pooled.get("tp"),
        "final_train_num_samples": (
            final_train_summary.get(
                "num_samples"
            )
        ),
        "final_train_input_dim": (
            final_train_summary.get(
                "input_dim"
            )
        ),
        "final_train_seconds": (
            final_train_summary.get(
                "train_seconds"
            )
        ),
        "final_train_loss": (
            final_train_summary.get(
                "final_loss"
            )
        ),
        "tabm_epochs": (
            final_train_summary.get(
                "epochs"
            )
        ),
        "tabm_batch_size": (
            final_train_summary.get(
                "batch_size"
            )
        ),
        "tabm_learning_rate": (
            final_train_summary.get(
                "learning_rate"
            )
        ),
        "tabm_k": (
            final_train_summary.get(
                "tabm_k"
            )
        ),
        "weight_decay": (
            final_train_summary.get(
                "weight_decay"
            )
        ),
        "model_weights": str(
            final_output_path
        ),
    }

    summary_csv_path = (
        variation_dir
        / "summary.csv"
    )
    with summary_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=list(
                summary_row.keys()
            ),
        )
        writer.writeheader()
        writer.writerow(summary_row)

    fold_rows = []
    for fold in lopo_evaluation[
        "folds"
    ]:
        metrics = fold["metrics"]
        split = fold["split"]
        fold_rows.append(
            {
                "run_id": run_id,
                "variation_name": (
                    variation_name
                ),
                "fold_index": (
                    fold["fold_index"]
                ),
                "held_out_participant": (
                    fold[
                        "held_out_participant"
                    ]
                ),
                "train_size": split.get(
                    "train_size"
                ),
                "test_size": split.get(
                    "test_size_count"
                ),
                "train_label_counts": (
                    json.dumps(
                        make_json_serializable(
                            split.get(
                                "train_label_counts"
                            )
                        ),
                        sort_keys=True,
                    )
                ),
                "test_label_counts": (
                    json.dumps(
                        make_json_serializable(
                            split.get(
                                "test_label_counts"
                            )
                        ),
                        sort_keys=True,
                    )
                ),
                "accuracy": metrics.get(
                    "accuracy"
                ),
                "balanced_accuracy": (
                    metrics.get(
                        "balanced_accuracy"
                    )
                ),
                "precision": metrics.get(
                    "precision"
                ),
                "recall": metrics.get(
                    "recall"
                ),
                "specificity": metrics.get(
                    "specificity"
                ),
                "f1": metrics.get("f1"),
                "tn": metrics.get("tn"),
                "fp": metrics.get("fp"),
                "fn": metrics.get("fn"),
                "tp": metrics.get("tp"),
            }
        )

    folds_csv_path = (
        variation_dir
        / "fold_metrics.csv"
    )
    with folds_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=list(
                fold_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            fold_rows
        )

    tuning_json_path = None
    if tuning_summary is not None:
        tuning_json_path = (
            variation_dir
            / "tuning_results.json"
        )
        with tuning_json_path.open(
            "w",
            encoding="utf-8",
        ) as file_obj:
            json.dump(
                make_json_serializable(
                    tuning_summary
                ),
                file_obj,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            file_obj.write("\n")

    return {
        "evaluation_dir": str(
            variation_dir
        ),
        "evaluation_json": str(
            full_json_path
        ),
        "summary_csv": str(
            summary_csv_path
        ),
        "fold_metrics_csv": str(
            folds_csv_path
        ),
        "tuning_json": (
            str(tuning_json_path)
            if tuning_json_path is not None
            else None
        ),
    }


def train_tabm(
    rows,
    learning_rate: float = 2e-3,
    weight_decay: float = 3e-4,
    batch_size: int = 32,
    epochs: int = 30,
    k: int = 32,
    random_state: int = 0,
    device: str = "auto",
    output_path=None,
):
    X, y, summary = rows_to_xy(
        rows
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected X to be 2D, "
            f"got shape {X.shape}"
        )

    torch.manual_seed(
        int(random_state)
    )
    np.random.seed(
        int(random_state)
    )

    if device == "auto":
        resolved_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        resolved_device = (
            torch.device(device)
        )

    model = HandIntentTabM(
        input_size=int(
            X.shape[1]
        ),
        output_size=1,
        k=int(k),
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            learning_rate
        ),
        weight_decay=float(
            weight_decay
        ),
    )

    # TabM returns k independent logits per sample. Compute the BCE loss
    # for every member prediction independently, then average the losses.
    criterion = (
        torch.nn.BCEWithLogitsLoss(
            reduction="mean"
        )
    )

    x_tensor = torch.from_numpy(
        X.astype(np.float32)
    )
    y_tensor = torch.from_numpy(
        y.astype(
            np.float32
        ).reshape(-1, 1)
    )

    dataset = TensorDataset(
        x_tensor,
        y_tensor,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=True,
    )

    loss_history = []
    num_batches_per_epoch = len(
        loader
    )
    train_start = (
        time.perf_counter()
    )

    model.train()

    for epoch_index in range(
        int(epochs)
    ):
        print(
            "Starting epoch",
            epoch_index + 1,
            "of",
            int(epochs),
        )

        running_loss = 0.0
        sample_count = 0

        for xb, yb in loader:
            xb = xb.to(
                resolved_device
            )
            yb = yb.to(
                resolved_device
            )

            optimizer.zero_grad()

            logits = model(xb)

            # Expected TabM shape: [batch, k, 1].
            # Accept [batch, k] as well to make the trainer tolerant of
            # a wrapper that squeezes the final output dimension.
            if logits.ndim == 2:
                logits = (
                    logits.unsqueeze(-1)
                )

            if logits.ndim != 3:
                raise ValueError(
                    "Expected TabM output "
                    "with shape "
                    "[batch, k, 1] or "
                    "[batch, k], got "
                    f"{tuple(logits.shape)}"
                )

            if logits.shape[-1] != 1:
                raise ValueError(
                    "Expected binary TabM "
                    "output dimension 1, "
                    "got "
                    f"{logits.shape[-1]}"
                )

            targets = (
                yb.unsqueeze(1).expand(
                    -1,
                    logits.shape[1],
                    -1,
                )
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()
            optimizer.step()

            current_batch = int(
                xb.shape[0]
            )
            running_loss += (
                float(loss.item())
                * current_batch
            )
            sample_count += (
                current_batch
            )

        epoch_loss = (
            running_loss
            / max(
                sample_count,
                1,
            )
        )
        loss_history.append(
            epoch_loss
        )

    train_seconds = float(
        time.perf_counter()
        - train_start
    )

    if output_path is not None:
        path_obj = Path(
            output_path
        )
        path_obj.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "model_type": "tabm",
                "input_dim": int(
                    X.shape[1]
                ),
                "output_dim": 1,
                "tabm_k": int(k),
                "model_state_dict": (
                    model.state_dict()
                ),
                "learning_rate": float(
                    learning_rate
                ),
                "weight_decay": float(
                    weight_decay
                ),
                "batch_size": int(
                    batch_size
                ),
                "epochs": int(
                    epochs
                ),
                "random_state": int(
                    random_state
                ),
                "device": str(
                    resolved_device
                ),
            },
            path_obj,
        )

    train_summary = {
        **summary,
        "model_type": "tabm",
        "num_samples": int(
            X.shape[0]
        ),
        "input_dim": int(
            X.shape[1]
        ),
        "epochs": int(epochs),
        "batch_size": int(
            batch_size
        ),
        "learning_rate": float(
            learning_rate
        ),
        "weight_decay": float(
            weight_decay
        ),
        "tabm_k": int(k),
        "random_state": int(
            random_state
        ),
        "device": str(
            resolved_device
        ),
        "num_batches_per_epoch": int(
            num_batches_per_epoch
        ),
        "total_optimizer_steps": int(
            num_batches_per_epoch
            * int(epochs)
        ),
        "train_seconds": (
            train_seconds
        ),
        "final_loss": (
            float(
                loss_history[-1]
            )
            if loss_history
            else None
        ),
        "loss_history": (
            loss_history
        ),
    }

    return model, train_summary


def compute_binary_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int32,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=np.int32,
    )

    total = int(
        len(y_true)
    )
    correct = int(
        np.sum(
            y_true == y_pred
        )
    )
    accuracy = (
        float(
            correct / total
        )
        if total > 0
        else 0.0
    )

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )
    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )
    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )
    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    precision = (
        float(
            tp / (tp + fp)
        )
        if (tp + fp) > 0
        else None
    )
    recall = (
        float(
            tp / (tp + fn)
        )
        if (tp + fn) > 0
        else None
    )
    specificity = (
        float(
            tn / (tn + fp)
        )
        if (tn + fp) > 0
        else None
    )
    f1 = (
        float(
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
        if (
            precision is not None
            and recall is not None
            and (
                precision
                + recall
            ) > 0
        )
        else 0.0
    )

    balanced_accuracy = None
    if (
        recall is not None
        and specificity is not None
    ):
        balanced_accuracy = float(
            0.5
            * (
                recall
                + specificity
            )
        )

    return {
        "total": total,
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [
            [tn, fp],
            [fn, tp],
        ],
    }


def evaluate_tabm(
    model,
    train_rows,
    test_rows,
    device: str,
    threshold: float = 0.5,
):
    def participant_ids(rows):
        ids = []
        for row in rows:
            if row.get(
                "skipped",
                False,
            ):
                continue
            participant_id = (
                row.get(
                    "participant_id"
                )
            )
            if participant_id is None:
                continue
            ids.append(
                str(
                    participant_id
                )
            )
        return sorted(
            set(ids)
        )

    X_train, y_train, train_summary = (
        rows_to_xy(
            train_rows
        )
    )
    X_test, y_test, test_summary = (
        rows_to_xy(
            test_rows
        )
    )

    if len(
        np.unique(y_train)
    ) < 2:
        raise ValueError(
            "Need both classes present "
            "in the training split."
        )

    model.eval()

    x_test_tensor = (
        torch.from_numpy(
            X_test.astype(
                np.float32
            )
        ).to(
            torch.device(
                device
            )
        )
    )

    with torch.no_grad():
        logits = model(
            x_test_tensor
        )

        if logits.ndim == 2:
            logits = (
                logits.unsqueeze(-1)
            )

        if logits.ndim != 3:
            raise ValueError(
                "Expected TabM output "
                "with shape [batch, k, 1] "
                "or [batch, k], got "
                f"{tuple(logits.shape)}"
            )

        # For classification, average member probabilities rather than
        # averaging logits first.
        member_probabilities = (
            torch.sigmoid(
                logits
            )
        )

        scores = (
            member_probabilities
            .mean(dim=1)
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    y_pred = (
        scores
        >= float(
            threshold
        )
    ).astype(
        np.int32
    )

    metrics = (
        compute_binary_metrics(
            y_test,
            y_pred,
        )
    )

    evaluation = {
        "model_name": "tabm",
        "train_summary": (
            train_summary
        ),
        "test_summary": (
            test_summary
        ),
        "split": {
            "train_size": int(
                len(train_rows)
            ),
            "test_size_count": int(
                len(test_rows)
            ),
            "train_label_counts": (
                train_summary[
                    "label_counts"
                ]
            ),
            "test_label_counts": (
                test_summary[
                    "label_counts"
                ]
            ),
            "train_participants": (
                participant_ids(
                    train_rows
                )
            ),
            "test_participants": (
                participant_ids(
                    test_rows
                )
            ),
        },
        "metrics": metrics,
        "y_true": (
            y_test.tolist()
        ),
        "y_pred": (
            np.asarray(
                y_pred
            ).tolist()
        ),
        "scores": (
            np.asarray(
                scores
            ).tolist()
        ),
    }

    return evaluation


def get_usable_participant_rows(rows):
    """Return rows usable for participant-level training/evaluation."""
    usable_rows = []

    for row in rows:
        if row.get(
            "skipped",
            False,
        ):
            continue

        if row.get(
            "feature_vector"
        ) is None:
            continue

        if row.get(
            "label_binary"
        ) is None:
            continue

        if row.get(
            "participant_id"
        ) is None:
            continue

        usable_rows.append(
            row
        )

    return usable_rows


def make_participant_cv_folds(
    rows,
    *,
    num_folds: int,
    random_state: int,
):
    """
    Build participant-level validation folds.

    Every participant appears in exactly one validation fold. No participant
    is split between the training and validation sides of a fold.
    """
    usable_rows = (
        get_usable_participant_rows(
            rows
        )
    )

    participants = sorted(
        {
            str(
                row[
                    "participant_id"
                ]
            )
            for row in usable_rows
        }
    )

    if len(participants) < 2:
        raise ValueError(
            "TabM tuning requires at least "
            "two participants."
        )

    fold_count = min(
        int(num_folds),
        len(participants),
    )

    if fold_count < 2:
        raise ValueError(
            "TabM tuning requires at least "
            "two participant folds."
        )

    rng = np.random.default_rng(
        int(random_state)
    )
    shuffled = np.asarray(
        participants,
        dtype=object,
    )
    rng.shuffle(
        shuffled
    )

    participant_groups = [
        [
            str(value)
            for value in group.tolist()
        ]
        for group in np.array_split(
            shuffled,
            fold_count,
        )
        if len(group) > 0
    ]

    folds = []

    for fold_index, validation_ids in enumerate(
        participant_groups
    ):
        validation_set = set(
            validation_ids
        )

        train_rows = [
            row
            for row in usable_rows
            if str(
                row[
                    "participant_id"
                ]
            )
            not in validation_set
        ]

        validation_rows = [
            row
            for row in usable_rows
            if str(
                row[
                    "participant_id"
                ]
            )
            in validation_set
        ]

        if not train_rows:
            raise ValueError(
                "A tuning fold contained no "
                "training rows."
            )

        if not validation_rows:
            raise ValueError(
                "A tuning fold contained no "
                "validation rows."
            )

        folds.append(
            {
                "fold_index": (
                    fold_index
                ),
                "validation_participants": (
                    sorted(
                        validation_set
                    )
                ),
                "train_rows": (
                    train_rows
                ),
                "validation_rows": (
                    validation_rows
                ),
            }
        )

    return folds


def generate_tabm_tuning_candidates(
    *,
    current_params,
    random_state: int,
    num_trials: int,
):
    """
    Generate a reproducible random subset of the discrete TabM search space.

    The script's current/default hyperparameters are always trial 1, so tuning
    can never omit the existing baseline configuration from the search.
    """
    all_candidates = []

    for (
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        k,
    ) in itertools.product(
        TABM_TUNE_SPACE[
            "epochs"
        ],
        TABM_TUNE_SPACE[
            "batch_size"
        ],
        TABM_TUNE_SPACE[
            "learning_rate"
        ],
        TABM_TUNE_SPACE[
            "weight_decay"
        ],
        TABM_TUNE_SPACE[
            "k"
        ],
    ):
        all_candidates.append(
            {
                "epochs": int(
                    epochs
                ),
                "batch_size": int(
                    batch_size
                ),
                "learning_rate": float(
                    learning_rate
                ),
                "weight_decay": float(
                    weight_decay
                ),
                "k": int(k),
            }
        )

    baseline = {
        "epochs": int(
            current_params[
                "epochs"
            ]
        ),
        "batch_size": int(
            current_params[
                "batch_size"
            ]
        ),
        "learning_rate": float(
            current_params[
                "learning_rate"
            ]
        ),
        "weight_decay": float(
            current_params[
                "weight_decay"
            ]
        ),
        "k": int(
            current_params["k"]
        ),
    }

    # Include the current configuration even if the user passed a value
    # outside the predefined tuning grid.
    remaining = [
        candidate
        for candidate in all_candidates
        if candidate != baseline
    ]

    rng = np.random.default_rng(
        int(random_state)
    )
    rng.shuffle(
        remaining
    )

    trial_count = max(
        1,
        int(num_trials),
    )

    selected = [baseline]

    if trial_count > 1:
        selected.extend(
            remaining[
                : trial_count - 1
            ]
        )

    return selected


def tune_tabm_hyperparameters(
    rows,
    *,
    current_params,
    random_state: int = 0,
    device: str = "auto",
    threshold: float = 0.5,
    num_trials: int = TABM_TUNE_NUM_TRIALS,
    num_folds: int = TABM_TUNE_NUM_FOLDS,
):
    """
    Search TabM hyperparameters using participant-level cross-validation.

    Objective
    ---------
    Maximize mean validation balanced accuracy across participant folds.

    Tie-breaking
    ------------
    1. Higher mean validation balanced accuracy.
    2. Higher pooled validation balanced accuracy.
    3. Higher mean validation accuracy.

    No checkpoints or evaluation files are written during this search.
    """
    tune_started_at = (
        datetime.now().astimezone()
    )
    tune_start = (
        time.perf_counter()
    )

    folds = make_participant_cv_folds(
        rows,
        num_folds=num_folds,
        random_state=random_state,
    )

    candidates = (
        generate_tabm_tuning_candidates(
            current_params=(
                current_params
            ),
            random_state=(
                random_state
            ),
            num_trials=(
                num_trials
            ),
        )
    )

    print(
        "\n"
        "========================================"
    )
    print(
        "Starting TabM hyperparameter tuning"
    )
    print(
        "========================================"
    )
    print(
        f"Trials: {len(candidates)}"
    )
    print(
        f"Participant CV folds: {len(folds)}"
    )
    print(
        "Objective: mean participant-fold "
        "balanced accuracy"
    )
    print(
        "No checkpoints will be saved "
        "during tuning."
    )

    trial_results = []

    for trial_index, candidate in enumerate(
        candidates
    ):
        print(
            "\n"
            "----------------------------------------"
        )
        print(
            f"Tuning trial "
            f"{trial_index + 1}/"
            f"{len(candidates)}"
        )
        print(
            "Parameters:",
            candidate,
        )

        fold_results = []
        all_y_true = []
        all_y_pred = []

        for tune_fold_index, fold in enumerate(
            folds
        ):
            print(
                f"\nTune fold "
                f"{tune_fold_index + 1}/"
                f"{len(folds)} "
                f"validation participants="
                f"{fold['validation_participants']}"
            )

            fold_seed = (
                int(random_state)
                + (
                    trial_index
                    * 1000
                )
                + tune_fold_index
            )

            model, train_summary = train_tabm(
                fold[
                    "train_rows"
                ],
                learning_rate=(
                    candidate[
                        "learning_rate"
                    ]
                ),
                weight_decay=(
                    candidate[
                        "weight_decay"
                    ]
                ),
                batch_size=(
                    candidate[
                        "batch_size"
                    ]
                ),
                epochs=(
                    candidate[
                        "epochs"
                    ]
                ),
                k=candidate["k"],
                random_state=(
                    fold_seed
                ),
                device=device,
                output_path=None,
            )

            evaluation = evaluate_tabm(
                model=model,
                train_rows=(
                    fold[
                        "train_rows"
                    ]
                ),
                test_rows=(
                    fold[
                        "validation_rows"
                    ]
                ),
                device=(
                    train_summary[
                        "device"
                    ]
                ),
                threshold=threshold,
            )

            metrics = (
                evaluation[
                    "metrics"
                ]
            )

            fold_results.append(
                {
                    "fold_index": (
                        tune_fold_index
                    ),
                    "validation_participants": (
                        fold[
                            "validation_participants"
                        ]
                    ),
                    "metrics": metrics,
                }
            )

            all_y_true.extend(
                evaluation[
                    "y_true"
                ]
            )
            all_y_pred.extend(
                evaluation[
                    "y_pred"
                ]
            )

            print(
                "Tune fold metrics: "
                f"accuracy="
                f"{metrics['accuracy']:.4f}, "
                "balanced_accuracy="
                f"{metrics['balanced_accuracy']}"
            )

            # Explicitly release the fold model before the next training run.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        balanced_values = [
            fold_result[
                "metrics"
            ][
                "balanced_accuracy"
            ]
            for fold_result in fold_results
            if (
                fold_result[
                    "metrics"
                ][
                    "balanced_accuracy"
                ]
                is not None
            )
        ]

        accuracy_values = [
            fold_result[
                "metrics"
            ][
                "accuracy"
            ]
            for fold_result in fold_results
        ]

        mean_balanced_accuracy = (
            float(
                np.mean(
                    balanced_values
                )
            )
            if balanced_values
            else None
        )

        std_balanced_accuracy = (
            float(
                np.std(
                    balanced_values
                )
            )
            if balanced_values
            else None
        )

        mean_accuracy = float(
            np.mean(
                accuracy_values
            )
        )

        pooled_metrics = (
            compute_binary_metrics(
                all_y_true,
                all_y_pred,
            )
        )

        result = {
            "trial_index": (
                trial_index
            ),
            "parameters": (
                candidate
            ),
            "num_folds": len(
                fold_results
            ),
            "mean_balanced_accuracy": (
                mean_balanced_accuracy
            ),
            "std_balanced_accuracy": (
                std_balanced_accuracy
            ),
            "mean_accuracy": (
                mean_accuracy
            ),
            "pooled_metrics": (
                pooled_metrics
            ),
            "folds": fold_results,
        }

        trial_results.append(
            result
        )

        print(
            "\nTrial result:",
            {
                "mean_balanced_accuracy": (
                    mean_balanced_accuracy
                ),
                "std_balanced_accuracy": (
                    std_balanced_accuracy
                ),
                "pooled_balanced_accuracy": (
                    pooled_metrics.get(
                        "balanced_accuracy"
                    )
                ),
                "mean_accuracy": (
                    mean_accuracy
                ),
            },
        )

    def rank_key(result):
        mean_balanced = (
            result[
                "mean_balanced_accuracy"
            ]
        )
        pooled_balanced = (
            result[
                "pooled_metrics"
            ].get(
                "balanced_accuracy"
            )
        )
        mean_accuracy = (
            result[
                "mean_accuracy"
            ]
        )

        return (
            (
                -math.inf
                if mean_balanced is None
                else float(
                    mean_balanced
                )
            ),
            (
                -math.inf
                if pooled_balanced is None
                else float(
                    pooled_balanced
                )
            ),
            float(
                mean_accuracy
            ),
        )

    best_result = max(
        trial_results,
        key=rank_key,
    )

    tune_seconds = float(
        time.perf_counter()
        - tune_start
    )

    tuning_summary = {
        "enabled": True,
        "started_at": (
            tune_started_at.isoformat()
        ),
        "completed_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "train_seconds": (
            tune_seconds
        ),
        "objective": (
            "mean_validation_balanced_accuracy"
        ),
        "selection_tie_breakers": [
            (
                "pooled_validation_"
                "balanced_accuracy"
            ),
            (
                "mean_validation_"
                "accuracy"
            ),
        ],
        "num_trials": len(
            trial_results
        ),
        "num_folds": len(
            folds
        ),
        "search_space": (
            TABM_TUNE_SPACE
        ),
        "best_trial_index": (
            best_result[
                "trial_index"
            ]
        ),
        "best_parameters": (
            best_result[
                "parameters"
            ]
        ),
        "best_mean_balanced_accuracy": (
            best_result[
                "mean_balanced_accuracy"
            ]
        ),
        "best_std_balanced_accuracy": (
            best_result[
                "std_balanced_accuracy"
            ]
        ),
        "best_pooled_metrics": (
            best_result[
                "pooled_metrics"
            ]
        ),
        "trials": trial_results,
    }

    print(
        "\n"
        "========================================"
    )
    print(
        "TabM hyperparameter tuning complete"
    )
    print(
        "========================================"
    )
    print(
        "Best parameters:",
        tuning_summary[
            "best_parameters"
        ],
    )
    print(
        "Best mean validation "
        "balanced accuracy:",
        tuning_summary[
            "best_mean_balanced_accuracy"
        ],
    )
    print(
        "Best pooled validation metrics:",
        tuning_summary[
            "best_pooled_metrics"
        ],
    )

    return (
        dict(
            tuning_summary[
                "best_parameters"
            ]
        ),
        tuning_summary,
    )


def leave_one_participant_out_evaluation(
    rows,
    learning_rate: float = 2e-3,
    weight_decay: float = 3e-4,
    batch_size: int = 32,
    epochs: int = 30,
    k: int = 32,
    random_state: int = 0,
    device: str = "auto",
    threshold: float = 0.5,
):
    """
    Perform leave-one-participant-out evaluation.

    For each fold:
        - Hold out one participant for testing.
        - Train a new TabM model from scratch on all other participants.
        - Evaluate on the held-out participant.

    Returns per-participant results and pooled metrics.
    """

    usable_rows = (
        get_usable_participant_rows(
            rows
        )
    )

    participants = sorted(
        {
            str(
                row[
                    "participant_id"
                ]
            )
            for row in usable_rows
        }
    )

    if len(participants) < 2:
        raise ValueError(
            "Leave-one-participant-out "
            "evaluation requires at least "
            "two participants."
        )

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_scores = []

    for fold_index, held_out_participant in enumerate(
        participants
    ):
        print(
            f"\nLOPO fold "
            f"{fold_index + 1}/"
            f"{len(participants)}: "
            "holding out participant "
            f"{held_out_participant}"
        )

        train_rows = [
            row
            for row in usable_rows
            if str(
                row[
                    "participant_id"
                ]
            )
            != held_out_participant
        ]

        test_rows = [
            row
            for row in usable_rows
            if str(
                row[
                    "participant_id"
                ]
            )
            == held_out_participant
        ]

        if len(
            train_rows
        ) == 0:
            raise ValueError(
                "No training rows for fold "
                f"{held_out_participant}."
            )

        if len(
            test_rows
        ) == 0:
            raise ValueError(
                "No test rows for fold "
                f"{held_out_participant}."
            )

        # Important: create and train a completely new model for every fold.
        fold_model, train_summary = train_tabm(
            train_rows,
            learning_rate=(
                learning_rate
            ),
            weight_decay=(
                weight_decay
            ),
            batch_size=(
                batch_size
            ),
            epochs=epochs,
            k=k,
            random_state=(
                random_state
                + fold_index
            ),
            device=device,
            output_path=None,
        )

        fold_evaluation = evaluate_tabm(
            model=fold_model,
            train_rows=train_rows,
            test_rows=test_rows,
            device=(
                train_summary[
                    "device"
                ]
            ),
            threshold=threshold,
        )

        fold_evaluation[
            "fold_index"
        ] = fold_index
        fold_evaluation[
            "held_out_participant"
        ] = held_out_participant

        fold_results.append(
            fold_evaluation
        )

        all_y_true.extend(
            fold_evaluation[
                "y_true"
            ]
        )
        all_y_pred.extend(
            fold_evaluation[
                "y_pred"
            ]
        )
        all_scores.extend(
            fold_evaluation[
                "scores"
            ]
        )

        metrics = (
            fold_evaluation[
                "metrics"
            ]
        )

        print(
            f"Participant "
            f"{held_out_participant}: "
            f"accuracy="
            f"{metrics['accuracy']:.4f}, "
            f"f1="
            f"{metrics['f1']:.4f}, "
            f"recall="
            f"{metrics['recall']}"
        )

    pooled_metrics = (
        compute_binary_metrics(
            y_true=all_y_true,
            y_pred=all_y_pred,
        )
    )

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]

    participant_metric_summary = {}

    for metric_name in metric_names:
        values = [
            fold["metrics"][
                metric_name
            ]
            for fold in fold_results
            if (
                fold["metrics"][
                    metric_name
                ]
                is not None
            )
        ]

        participant_metric_summary[
            metric_name
        ] = {
            "mean": (
                float(
                    np.mean(values)
                )
                if values
                else None
            ),
            "std": (
                float(
                    np.std(values)
                )
                if values
                else None
            ),
            "minimum": (
                float(
                    np.min(values)
                )
                if values
                else None
            ),
            "maximum": (
                float(
                    np.max(values)
                )
                if values
                else None
            ),
        }

    return {
        "evaluation_type": (
            "leave_one_participant_out"
        ),
        "num_participants": (
            len(participants)
        ),
        "participants": (
            participants
        ),
        "num_folds": (
            len(fold_results)
        ),
        "folds": fold_results,

        # Metrics after pooling predictions from every held-out person.
        "pooled_metrics": (
            pooled_metrics
        ),

        # Every participant receives equal weight in this summary.
        "participant_metric_summary": (
            participant_metric_summary
        ),

        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "scores": all_scores,
    }


def main():
    with open(
        CONFIG_PATH,
        "r",
    ) as f:
        config = yaml.safe_load(
            f
        )

    dataset_root = (
        REPO_ROOT
        / config[
            "dataset"
        ][
            "root"
        ]
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dino-classes",
        default="cup",
    )
    parser.add_argument(
        "--crop-around-object",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--crop-object",
        default="person.",
    )
    parser.add_argument(
        "--normalize-keypoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--features-type",
        type=str,
        choices=FEATURES_TYPE,
        default="keypoints",
    )
    parser.add_argument(
        "--img-encoder",
        type=str,
        choices=[
            "dino",
            "resnet",
        ],
        default="resnet",
    )
    parser.add_argument(
        "--participant",
        type=str,
        default=None,
        help=(
            "Filter to one participant ID, "
            "e.g. 1 or P01"
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help=(
            "Filter to label_name, e.g. "
            "handoff or not_handoff"
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Filter to one condition",
    )
    parser.add_argument(
        "--hand",
        type=str,
        choices=[
            "left",
            "right",
        ],
        default=None,
        help="Filter to one hand",
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

    # --------------------------------------------------------------
    # NEW:
    # When passed, tune TabM first. If omitted, the script follows
    # the original behavior and uses the values below directly.
    # --------------------------------------------------------------
    parser.add_argument(
        "--tune-tabm",
        action="store_true",
        help=(
            "Run participant-level TabM "
            "hyperparameter tuning first. "
            "No model/evaluation files are "
            "written during tuning. After "
            "the best settings are selected, "
            "the normal full LOPO evaluation, "
            "final training, checkpoint save, "
            "and evaluation exports run once "
            "using those settings."
        ),
    )

    parser.add_argument(
        "--tabm-epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--tabm-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--tabm-learning-rate",
        type=float,
        default=2e-3,
    )
    parser.add_argument(
        "--tabm-weight-decay",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--tabm-k",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--tabm-random-state",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--tabm-output-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=None,
        help=(
            "Root directory for evaluation "
            "files. Defaults to "
            "MODEL_IMPL_ROOT/outputs/"
            "eval_results/tabm."
        ),
    )

    args = parser.parse_args()

    run_started_at = (
        datetime.now().astimezone()
    )
    run_id = (
        run_started_at.strftime(
            "%Y%m%dT%H%M%S_%f%z"
        )
    )
    variation_name = (
        build_model_variation_name(
            args
        )
    )

    samples = discover_samples(
        dataset_root=dataset_root,
        participant_filter=(
            args.participant
        ),
        label_filter=args.label,
        condition_filter=(
            args.condition
        ),
        hand_filter=args.hand,
    )

    samples = samples[
        args.start_index:
    ]

    if (
        args.max_samples
        is not None
    ):
        samples = samples[
            : args.max_samples
        ]

    object_detector = (
        DINOObjectDetector(
            model_id=(
                "IDEA-Research/"
                "grounding-dino-base"
            ),
            confidence=args.confidence,
        )
    )

    keypoint_detector = (
        RTMPoseKeypointDetector(
            confidence=args.confidence,
            device="cuda",
        )
    )

    head_pose_estimator = (
        RTMPoseHeadPoseEstimator(
            weights_path=(
                REPO_ROOT
                / (
                    "model_training_"
                    "and_implementation"
                )
                / (
                    "kwan_pretrained_"
                    "weights"
                )
                / (
                    "head-pose-"
                    "pretrained.pkl"
                )
            ),
            gpu_id=0,
        )
    )

    if args.img_encoder == "dino":
        img_encoder = (
            DINOv2ImageEncoder()
        )
    elif args.img_encoder == "resnet":
        img_encoder = (
            ResNet18ImageEncoder(
                pretrained=True,
                device="cuda",
                l2_normalize=True,
            )
        )
    else:
        raise ValueError(
            "Unsupported encoder type: "
            f"{args.img_encoder}"
        )

    (
        y_true,
        rows,
        skipped_unreadable,
        skipped_no_object,
        skipped_no_people,
    ) = organize_samples(
        samples=samples,
        features_type=(
            args.features_type
        ),
        obj_detector=(
            object_detector
        ),
        keypoint_detector=(
            keypoint_detector
        ),
        head_pose_estimator=(
            head_pose_estimator
        ),
        img_encoder=(
            img_encoder
        ),
        args=args,
    )

    # --------------------------------------------------------------
    # Optional hyperparameter search.
    #
    # If --tune-tabm is absent, this entire block is skipped and the
    # original CLI/default TabM settings remain untouched.
    #
    # If --tune-tabm is present:
    #   1. Search candidate settings with participant-level CV.
    #   2. Pick the setting with the best mean validation balanced acc.
    #   3. Replace the TabM training args with that winning setting.
    #   4. Continue into the original LOPO/final-training/save workflow.
    #
    # Nothing is saved while candidates are being searched.
    # --------------------------------------------------------------
    tuning_summary = None

    if args.tune_tabm:
        current_params = {
            "epochs": (
                args.tabm_epochs
            ),
            "batch_size": (
                args.tabm_batch_size
            ),
            "learning_rate": (
                args.tabm_learning_rate
            ),
            "weight_decay": (
                args.tabm_weight_decay
            ),
            "k": args.tabm_k,
        }

        (
            best_params,
            tuning_summary,
        ) = tune_tabm_hyperparameters(
            rows=rows,
            current_params=(
                current_params
            ),
            random_state=(
                args.tabm_random_state
            ),
            device="auto",
            threshold=(
                args.decision_threshold
            ),
        )

        # Make the rest of the script operate exactly as a normal run,
        # but with the hyperparameters selected by the tuning stage.
        args.tabm_epochs = int(
            best_params[
                "epochs"
            ]
        )
        args.tabm_batch_size = int(
            best_params[
                "batch_size"
            ]
        )
        args.tabm_learning_rate = float(
            best_params[
                "learning_rate"
            ]
        )
        args.tabm_weight_decay = float(
            best_params[
                "weight_decay"
            ]
        )
        args.tabm_k = int(
            best_params["k"]
        )

        print(
            "\nUsing tuned TabM "
            "hyperparameters for normal "
            "LOPO/final training:"
        )
        print(
            {
                "tabm_epochs": (
                    args.tabm_epochs
                ),
                "tabm_batch_size": (
                    args.tabm_batch_size
                ),
                "tabm_learning_rate": (
                    args.tabm_learning_rate
                ),
                "tabm_weight_decay": (
                    args.tabm_weight_decay
                ),
                "tabm_k": (
                    args.tabm_k
                ),
            }
        )

    lopo_evaluation = (
        leave_one_participant_out_evaluation(
            rows=rows,
            learning_rate=(
                args.tabm_learning_rate
            ),
            weight_decay=(
                args.tabm_weight_decay
            ),
            batch_size=(
                args.tabm_batch_size
            ),
            epochs=(
                args.tabm_epochs
            ),
            k=args.tabm_k,
            random_state=(
                args.tabm_random_state
            ),
            device="auto",
            threshold=(
                args.decision_threshold
            ),
        )
    )

    print(
        "\nLeave-one-participant-out "
        "evaluation complete"
    )

    print(
        "Pooled metrics:",
        lopo_evaluation[
            "pooled_metrics"
        ],
    )

    print(
        "Participant-level metric summary:",
        lopo_evaluation[
            "participant_metric_summary"
        ],
    )

    for fold in lopo_evaluation[
        "folds"
    ]:
        participant = (
            fold[
                "held_out_participant"
            ]
        )
        metrics = (
            fold[
                "metrics"
            ]
        )

        print(
            f"Participant "
            f"{participant}: "
            f"accuracy="
            f"{metrics['accuracy']:.4f}, "
            f"balanced_accuracy="
            f"{metrics['balanced_accuracy']}, "
            f"precision="
            f"{metrics['precision']}, "
            f"recall="
            f"{metrics['recall']}, "
            f"f1="
            f"{metrics['f1']:.4f}"
        )

    # Train one final deployment model using all available participants.
    output_path = (
        args.tabm_output_path
    )

    if output_path is None:
        output_path = (
            MODEL_IMPL_ROOT
            / "outputs"
            / "hand_intent_tabm"
            / sanitize_path_component(
                variation_name
            )
            / "TabM.pth"
        )
    else:
        output_path = Path(
            output_path
        )

    (
        final_model,
        final_train_summary,
    ) = train_tabm(
        rows=rows,
        learning_rate=(
            args.tabm_learning_rate
        ),
        weight_decay=(
            args.tabm_weight_decay
        ),
        batch_size=(
            args.tabm_batch_size
        ),
        epochs=args.tabm_epochs,
        k=args.tabm_k,
        random_state=(
            args.tabm_random_state
        ),
        device="auto",
        output_path=(
            output_path
        ),
    )

    print(
        f"\nFinal TabM saved to: "
        f"{output_path}"
    )
    print(
        "Final training summary:",
        final_train_summary,
    )

    dataset_summary = {
        (
            "num_samples_after_"
            "discovery_filters_and_"
            "slicing"
        ): int(
            len(samples)
        ),
        (
            "num_rows_after_"
            "feature_organization"
        ): int(
            len(rows)
        ),
        "num_labels_returned": int(
            len(y_true)
        ),
        "skipped_unreadable": int(
            skipped_unreadable
        ),
        "skipped_no_object": int(
            skipped_no_object
        ),
        "skipped_no_people": int(
            skipped_no_people
        ),
        "participant_filter": (
            args.participant
        ),
        "label_filter": (
            args.label
        ),
        "condition_filter": (
            args.condition
        ),
        "hand_filter": args.hand,
        "start_index": int(
            args.start_index
        ),
        "max_samples": (
            args.max_samples
        ),
    }

    saved_eval_paths = (
        save_evaluation_results(
            lopo_evaluation=(
                lopo_evaluation
            ),
            final_train_summary=(
                final_train_summary
            ),
            args=args,
            run_id=run_id,
            variation_name=(
                variation_name
            ),
            run_started_at=(
                run_started_at
            ),
            final_output_path=(
                output_path
            ),
            dataset_root=(
                dataset_root
            ),
            dataset_summary=(
                dataset_summary
            ),
            tuning_summary=(
                tuning_summary
            ),
        )
    )

    print(
        "Evaluation results saved to: "
        f"{saved_eval_paths['evaluation_dir']}"
    )
    print(
        "  Full JSON: "
        f"{saved_eval_paths['evaluation_json']}"
    )
    print(
        "  Summary CSV: "
        f"{saved_eval_paths['summary_csv']}"
    )
    print(
        "  Fold metrics CSV: "
        f"{saved_eval_paths['fold_metrics_csv']}"
    )

    if (
        saved_eval_paths[
            "tuning_json"
        ]
        is not None
    ):
        print(
            "  Tuning results: "
            f"{saved_eval_paths['tuning_json']}"
        )

    print(
        "FINAL LOPO POOLED BALANCE ACCURACY:",
        lopo_evaluation[
            "pooled_metrics"
        ][
            "balanced_accuracy"
        ],
    )

    return {
        "lopo_evaluation": (
            lopo_evaluation
        ),
        "final_train_summary": (
            final_train_summary
        ),
        "final_output_path": str(
            output_path
        ),
        "saved_eval_paths": (
            saved_eval_paths
        ),
        "tuning_summary": (
            tuning_summary
        ),
        "run_id": run_id,
        "variation_name": (
            variation_name
        ),
    }


if __name__ == "__main__":
    main()


