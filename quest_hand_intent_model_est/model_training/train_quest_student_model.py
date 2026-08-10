#python -m quest_hand_intent_model_est.model_training.train_quest_hand_int_est_mlp --teacher-features-type keypoints --student-features-type keypoints_projections
import numpy as np
from shared.util.extract_all_samples import discover_samples, organize_samples
from pathlib import Path
import yaml
import argparse
import csv
import json
import math
import re
import time
import importlib
import joblib
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from model_training_and_implementation.src.dino_detector import DINOObjectDetector
from model_training_and_implementation.src.rtmpose_keypoints import RTMPoseKeypointDetector
from model_training_and_implementation.src.rtmpose_headpose import RTMPoseHeadPoseEstimator
from model_training_and_implementation.src.hand_intent_mlp import HandIntentMLP
from model_training_and_implementation.src.resnet_encoder import ResNet18ImageEncoder


from quest_hand_intent_model_est.src.quest_hand_intent_est_mlp import QuestHandIntentEstimatorMLP
from quest_hand_intent_model_est.src.quest_joint_features import extract_quest_joint_features, construct_pose_img_from_joint_feature_vector, QUEST_JOINT_ORDER


TEACHER_FEATURES_TYPE = ["keypoints","keypoints_headpose","keypoints_headpose_dino","keypoints_resnet","keypoints_headpose_resnet"]
STUDENT_FEATURES_TYPE = ["keypoints","keypoints_projections","keypoints_resnet"]
MODEL_TYPES = ["mlp", "xgboost", "tabm"]


MODEL_IMPL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = MODEL_IMPL_ROOT / "configs" / "handoff_config.yaml"


def make_json_serializable(value):
    """Convert common scientific-Python values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return make_json_serializable(value.tolist())
    if isinstance(value, np.generic):
        return make_json_serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sanitize_path_component(value):
    """Return a filesystem-safe, human-readable path component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "unspecified"


def build_model_variation_name(args):
    """Build a descriptive name for the teacher/student configuration."""
    parts = [
        f"model-{args.model_type}",
        f"teacher_features-{args.teacher_features_type}",
        f"student_features-{args.student_features_type}",
        f"logit_weight-{args.logit_loss_weight}",
        f"hidden_weight-{args.hidden_loss_weight}",
        f"crop-{args.crop_around_object}",
        f"rotations-{args.include_quest_joint_rotations}",
    ]
    return "__".join(sanitize_path_component(part) for part in parts)

def metric_value(metrics, key):
    if metrics is None:
        return None
    return metrics.get(key)


def save_evaluation_results(
    *,
    lopo_evaluation,
    final_train_summary,
    args,
    run_id,
    variation_name,
    run_started_at,
    final_output_path,
    teacher_weights_path,
    dataset_root,
    dataset_summary,
):
    """Save one non-overwriting evaluation bundle as JSON and CSV files."""
    eval_root = (
        Path(args.eval_output_dir)
        if args.eval_output_dir is not None
        else MODEL_IMPL_ROOT / "outputs" / "eval_results" / args.model_type
    )
    run_dir = eval_root / sanitize_path_component(variation_name) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

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
            "dataset_root": str(dataset_root),
            "teacher_model_weights": str(teacher_weights_path),
            "student_model_weights": str(final_output_path),
            "evaluation_directory": str(run_dir),
        },
        "dataset_summary": dataset_summary,
        "lopo_evaluation": lopo_evaluation,
        "final_training_summary": final_train_summary,
    }

    full_json_path = run_dir / "evaluation_results.json"
    with full_json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            make_json_serializable(result_payload),
            file_obj,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file_obj.write("\n")

    pooled = lopo_evaluation["pooled_agreement_metrics"]
    student_gt = lopo_evaluation.get("student_ground_truth_metrics")
    teacher_gt = lopo_evaluation.get("teacher_ground_truth_metrics")

    summary_row = {
        "run_id": run_id,
        "variation_name": variation_name,
        "model_type": args.model_type,
        "teacher_model_type": args.model_type,
        "student_model_type": args.model_type,
        "num_participants": lopo_evaluation["num_participants"],
        "num_folds": lopo_evaluation["num_folds"],
        "include_rotations": lopo_evaluation["include_rotations"],
        "pooled_agreement_accuracy": pooled.get("accuracy"),
        "pooled_agreement_balanced_accuracy": pooled.get("balanced_accuracy"),
        "pooled_agreement_precision": pooled.get("precision"),
        "pooled_agreement_recall": pooled.get("recall"),
        "pooled_agreement_specificity": pooled.get("specificity"),
        "pooled_agreement_f1": pooled.get("f1"),
        "pooled_probability_mae": lopo_evaluation["pooled_probability_mae"],
        "pooled_probability_mse": lopo_evaluation["pooled_probability_mse"],
        "student_label_accuracy": metric_value(student_gt, "accuracy"),
        "student_label_balanced_accuracy": metric_value(
            student_gt, "balanced_accuracy"
        ),
        "student_label_precision": metric_value(student_gt, "precision"),
        "student_label_recall": metric_value(student_gt, "recall"),
        "student_label_specificity": metric_value(student_gt, "specificity"),
        "student_label_f1": metric_value(student_gt, "f1"),
        "teacher_label_accuracy": metric_value(teacher_gt, "accuracy"),
        "teacher_label_balanced_accuracy": metric_value(
            teacher_gt, "balanced_accuracy"
        ),
        "final_train_num_samples": final_train_summary.get("num_samples"),
        "final_train_input_dim": final_train_summary.get("input_dim"),
        "final_train_loss": final_train_summary.get("final_loss"),
        "final_train_teacher_student_agreement": final_train_summary.get(
            "teacher_student_agreement"
        ),
        "final_train_probability_mae": final_train_summary.get(
            "probability_mae"
        ),
        "student_model_weights": str(final_output_path),
    }

    summary_csv_path = run_dir / "summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    fold_rows = []
    for fold in lopo_evaluation["folds"]:
        agreement = fold["agreement_metrics"]
        fold_student_gt = fold.get("student_ground_truth_metrics")
        fold_teacher_gt = fold.get("teacher_ground_truth_metrics")
        fold_rows.append(
            {
                "run_id": run_id,
                "variation_name": variation_name,
                "fold_index": fold["fold_index"],
                "held_out_participant": fold["held_out_participant"],
                "num_samples": fold["num_samples"],
                "agreement_accuracy": agreement.get("accuracy"),
                "agreement_balanced_accuracy": agreement.get(
                    "balanced_accuracy"
                ),
                "agreement_precision": agreement.get("precision"),
                "agreement_recall": agreement.get("recall"),
                "agreement_specificity": agreement.get("specificity"),
                "agreement_f1": agreement.get("f1"),
                "agreement_tn": agreement.get("tn"),
                "agreement_fp": agreement.get("fp"),
                "agreement_fn": agreement.get("fn"),
                "agreement_tp": agreement.get("tp"),
                "probability_mae": fold["probability_mae"],
                "probability_mse": fold["probability_mse"],
                "student_label_accuracy": metric_value(
                    fold_student_gt, "accuracy"
                ),
                "student_label_balanced_accuracy": metric_value(
                    fold_student_gt, "balanced_accuracy"
                ),
                "student_label_f1": metric_value(fold_student_gt, "f1"),
                "teacher_label_accuracy": metric_value(
                    fold_teacher_gt, "accuracy"
                ),
                "train_num_samples": fold["train_summary"].get("num_samples"),
                "train_seconds": fold["train_summary"].get("train_seconds"),
                "train_final_loss": fold["train_summary"].get("final_loss"),
            }
        )

    folds_csv_path = run_dir / "fold_metrics.csv"
    with folds_csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    return {
        "run_dir": str(run_dir),
        "evaluation_json": str(full_json_path),
        "summary_csv": str(summary_csv_path),
        "fold_metrics_csv": str(folds_csv_path),
    }


def get_model_input_dim(model, model_type="mlp"):
    """Return the expected input feature count for any supported student."""
    model_type = str(model_type).lower()

    if model_type == "xgboost":
        if hasattr(model, "n_features_in_"):
            return int(model.n_features_in_)
        raise ValueError("Could not determine XGBoost input dimension.")

    if hasattr(model, "input_size"):
        return int(model.input_size)
    if hasattr(model, "input_dim"):
        return int(model.input_dim)

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)

    raise ValueError(
        f"Could not determine input dimension for model_type={model_type}."
    )


def get_final_linear_layer(model):
    """Return the model's final Linear layer (MLP hidden distillation only)."""
    linear_layers = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if not linear_layers:
        raise ValueError("Could not find a Linear layer in the model.")
    return linear_layers[-1]


def forward_with_penultimate_hidden(model, features):
    """
    Run an MLP and capture the representation entering its final Linear layer.

    This is intentionally used only for the MLP -> MLP experiment. TabM has an
    ensemble dimension and XGBoost has no neural hidden representation.
    """
    captured = {}
    final_linear = get_final_linear_layer(model)

    def capture_hidden(module, inputs):
        if not inputs:
            raise RuntimeError("Final Linear layer received no positional input.")
        captured["hidden"] = inputs[0]

    handle = final_linear.register_forward_pre_hook(capture_hidden)
    try:
        output = model(features)
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError(
            "Failed to capture the penultimate hidden representation."
        )

    return output, captured["hidden"]


def _import_first_class(candidates):
    """Import the first available (module, class) pair."""
    errors = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:
            errors.append(f"{module_name}.{class_name}: {exc}")

    raise ImportError(
        "Could not import a compatible TabM class. Tried:\n  "
        + "\n  ".join(errors)
    )


def get_teacher_tabm_class():
    return _import_first_class(
        [
            (
                "model_training_and_implementation.src.hand_intent_tabm",
                "HandIntentTabM",
            ),
        ]
    )


def get_quest_tabm_class():
    # The first entry matches the filename/class name described for this
    # experiment. The fallbacks make the trainer tolerant of snake_case module
    # naming or an "Estimator" class spelling.
    return _import_first_class(
        [
            (
                "quest_hand_intent_model_est.src.QuestHandIntentEstTabM",
                "QuestHandIntentEstTabM",
            ),
            (
                "quest_hand_intent_model_est.src.quest_hand_intent_est_tabm",
                "QuestHandIntentEstTabM",
            ),
            (
                "quest_hand_intent_model_est.src.quest_hand_intent_est_tabm",
                "QuestHandIntentEstimatorTabM",
            ),
        ]
    )


def tabm_logits_to_probabilities(logits):
    """Convert TabM [B,k,1] logits to one probability per sample."""
    if logits.ndim == 2:
        logits = logits.unsqueeze(-1)

    if logits.ndim != 3 or logits.shape[-1] != 1:
        raise ValueError(
            "Expected TabM logits with shape [batch, k, 1] or [batch, k], "
            f"got {tuple(logits.shape)}"
        )

    return torch.sigmoid(logits).mean(dim=1).reshape(-1, 1)


def predict_model_scores(model, model_type, X, device="auto"):
    """Return one probability-like score in [0,1] for every row in X."""
    model_type = str(model_type).lower()
    X = np.asarray(X, dtype=np.float32)

    if model_type == "xgboost":
        # Teacher XGBoost is a classifier and exposes predict_proba. The Quest
        # student is trained as a regressor on the teacher's soft probability.
        if hasattr(model, "predict_proba"):
            scores = np.asarray(model.predict_proba(X), dtype=np.float32)[:, 1]
        else:
            scores = np.asarray(model.predict(X), dtype=np.float32).reshape(-1)
        return np.clip(scores, 0.0, 1.0).reshape(-1, 1).astype(np.float32)

    if device == "auto":
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    model = model.to(resolved_device)
    model.eval()
    x_tensor = torch.from_numpy(X).to(resolved_device)

    with torch.no_grad():
        output = model(x_tensor)
        if model_type == "mlp":
            probabilities = output.reshape(-1, 1)
        elif model_type == "tabm":
            probabilities = tabm_logits_to_probabilities(output)
        else:
            raise ValueError(f"Unsupported model_type={model_type}")

    return (
        probabilities.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
    )


def load_teacher_model(model_type, weights_path):
    """Load one of the three teacher model types."""
    model_type = str(model_type).lower()
    weights_path = Path(weights_path)

    if model_type == "mlp":
        checkpoint = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        model = HandIntentMLP(
            input_size=int(checkpoint["input_dim"]),
            output_size=1,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    if model_type == "tabm":
        checkpoint = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        TeacherTabM = get_teacher_tabm_class()
        model = TeacherTabM(
            input_size=int(checkpoint["input_dim"]),
            output_size=1,
            k=int(checkpoint.get("tabm_k", 32)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    if model_type == "xgboost":
        payload = joblib.load(weights_path)
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"]
        return payload

    raise ValueError(f"Unsupported model_type={model_type}")


def construct_quest_training_data(
    rows,
    teacher_model,
    teacher_model_type="mlp",
    device="auto",
    include_rotations=False,
    student_features_type="keypoints",
):
    """
    Construct aligned Quest inputs and teacher soft targets.

    Returns:
        X_quest: Quest joint features, shape (N, D_quest)
        y_teacher: teacher probabilities, shape (N, 1)
        h_teacher: MLP teacher hidden states, or None for TabM/XGBoost
        usable_rows: rows corresponding exactly to those N samples
    """
    if device == "auto":
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    resnet_encoder = ResNet18ImageEncoder(
        pretrained=True,
        device="cuda",
        l2_normalize=True,
    )

    quest_feature_vectors = []
    teacher_feature_vectors = []
    usable_rows = []

    for row in rows:
        json_path = row.get("quest_joints_pth")
        teacher_features = row.get("feature_vector")

        if json_path is None:
            print("Skipping row: missing quest_joints_pth")
            continue

        if teacher_features is None:
            print(f"Skipping {json_path}: missing teacher feature_vector")
            continue

        try:
            quest_features = np.asarray(
                extract_quest_joint_features(
                    json_path,
                    include_rotations=include_rotations,
                ),
                dtype=np.float32,
            )

            if student_features_type == "keypoints_projections":
                proj_joints, pose_img = construct_pose_img_from_joint_feature_vector(
                    quest_features
                )
                quest_features = np.concatenate(
                    [quest_features, proj_joints], axis=0
                )
            elif student_features_type == "keypoints_resnet":
                proj_joints, pose_img = construct_pose_img_from_joint_feature_vector(
                    quest_features
                )
                resnet_embedding = resnet_encoder.predict(pose_img)["embedding"]
                quest_features = np.concatenate(
                    [
                        quest_features,
                        resnet_embedding.flatten().astype(np.float32),
                    ],
                    axis=0,
                )

            teacher_features = np.asarray(
                teacher_features,
                dtype=np.float32,
            )

            if quest_features.ndim != 1:
                raise ValueError(
                    f"Quest feature vector must be 1D, got {quest_features.shape}"
                )

            if teacher_features.ndim != 1:
                raise ValueError(
                    "Teacher feature vector must be 1D, "
                    f"got {teacher_features.shape}"
                )

            quest_feature_vectors.append(quest_features)
            teacher_feature_vectors.append(teacher_features)
            usable_rows.append(row)

        except Exception as error:
            print(f"Skipping {json_path}: {error}")

    if not quest_feature_vectors:
        raise ValueError("No usable Quest samples were found.")

    X_quest = np.stack(quest_feature_vectors).astype(np.float32)
    X_teacher = np.stack(teacher_feature_vectors).astype(np.float32)

    y_teacher = predict_model_scores(
        teacher_model,
        teacher_model_type,
        X_teacher,
        device=str(resolved_device),
    )

    h_teacher = None
    if str(teacher_model_type).lower() == "mlp":
        teacher_model = teacher_model.to(resolved_device)
        teacher_model.eval()
        teacher_tensor = torch.from_numpy(X_teacher).to(resolved_device)
        with torch.no_grad():
            _, teacher_hidden = forward_with_penultimate_hidden(
                teacher_model,
                teacher_tensor,
            )
        h_teacher = (
            teacher_hidden.detach().cpu().numpy().astype(np.float32)
        )

    return X_quest, y_teacher, h_teacher, usable_rows


def train_quest_student_mlp(
    rows,
    teacher_model,
    teacher_model_type="mlp",
    student_features_type="keypoints",
    learning_rate: float = 1e-3,
    logit_loss_weight: float = 1.0,
    hidden_loss_weight: float = 0.1,
    batch_size: int = 32,
    epochs: int = 30,
    random_state: int = 0,
    device: str = "auto",
    threshold: float = 0.5,
    include_rotations: bool = False,
    output_path=None,
):
    if str(teacher_model_type).lower() != "mlp":
        raise ValueError("MLP student mode expects an MLP teacher.")

    if device == "auto":
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_state))

    X, y, teacher_hidden, usable_rows = construct_quest_training_data(
        rows=rows,
        teacher_model=teacher_model,
        teacher_model_type=teacher_model_type,
        device=str(resolved_device),
        include_rotations=include_rotations,
        student_features_type=student_features_type,
    )

    if teacher_hidden is None:
        raise ValueError("MLP hidden-state distillation requires MLP teacher hidden states.")

    x_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))
    teacher_hidden_tensor = torch.from_numpy(teacher_hidden.astype(np.float32))

    dataset = TensorDataset(x_tensor, y_tensor, teacher_hidden_tensor)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

    student_model = QuestHandIntentEstimatorMLP(
        input_size=int(X.shape[1]),
        output_size=1,
    ).to(resolved_device)

    teacher_hidden_dim = int(teacher_hidden.shape[1])
    student_hidden_dim = int(get_final_linear_layer(student_model).in_features)

    if student_hidden_dim == teacher_hidden_dim:
        hidden_adapter = torch.nn.Identity().to(resolved_device)
    else:
        hidden_adapter = torch.nn.Linear(
            student_hidden_dim,
            teacher_hidden_dim,
            bias=False,
        ).to(resolved_device)

    optimizer = torch.optim.Adam(
        list(student_model.parameters()) + list(hidden_adapter.parameters()),
        lr=float(learning_rate),
    )

    probability_criterion = torch.nn.BCELoss()
    logit_criterion = torch.nn.SmoothL1Loss()
    hidden_criterion = torch.nn.SmoothL1Loss()
    logit_epsilon = 1e-6

    loss_history = []
    probability_loss_history = []
    logit_loss_history = []
    hidden_loss_history = []
    train_start = time.perf_counter()

    student_model.train()
    hidden_adapter.train()

    for epoch in range(int(epochs)):
        running_loss = 0.0
        running_probability_loss = 0.0
        running_logit_loss = 0.0
        running_hidden_loss = 0.0
        sample_count = 0

        for xb, yb, hb in loader:
            xb = xb.to(resolved_device)
            yb = yb.to(resolved_device)
            hb = hb.to(resolved_device)

            optimizer.zero_grad()

            student_output, student_hidden = forward_with_penultimate_hidden(
                student_model,
                xb,
            )
            student_probabilities = student_output.reshape(-1, 1)

            probability_loss = probability_criterion(
                student_probabilities,
                yb,
            )

            teacher_logits = torch.logit(
                yb.clamp(logit_epsilon, 1.0 - logit_epsilon)
            )
            student_logits = torch.logit(
                student_probabilities.clamp(
                    logit_epsilon,
                    1.0 - logit_epsilon,
                )
            )
            logit_loss = logit_criterion(student_logits, teacher_logits)

            projected_student_hidden = hidden_adapter(student_hidden)
            hidden_loss = hidden_criterion(projected_student_hidden, hb)

            loss = (
                probability_loss
                + float(logit_loss_weight) * logit_loss
                + float(hidden_loss_weight) * hidden_loss
            )

            loss.backward()
            optimizer.step()

            current_batch_size = int(xb.shape[0])
            running_loss += float(loss.item()) * current_batch_size
            running_probability_loss += (
                float(probability_loss.item()) * current_batch_size
            )
            running_logit_loss += float(logit_loss.item()) * current_batch_size
            running_hidden_loss += float(hidden_loss.item()) * current_batch_size
            sample_count += current_batch_size

        epoch_loss = running_loss / max(sample_count, 1)
        epoch_probability_loss = running_probability_loss / max(sample_count, 1)
        epoch_logit_loss = running_logit_loss / max(sample_count, 1)
        epoch_hidden_loss = running_hidden_loss / max(sample_count, 1)

        loss_history.append(epoch_loss)
        probability_loss_history.append(epoch_probability_loss)
        logit_loss_history.append(epoch_logit_loss)
        hidden_loss_history.append(epoch_hidden_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"distillation_loss={epoch_loss:.6f}, "
            f"probability_bce={epoch_probability_loss:.6f}, "
            f"logit_loss={epoch_logit_loss:.6f}, "
            f"hidden_loss={epoch_hidden_loss:.6f}"
        )

    train_seconds = float(time.perf_counter() - train_start)

    student_scores = predict_model_scores(
        student_model,
        "mlp",
        X,
        device=str(resolved_device),
    ).reshape(-1)
    teacher_scores = y.reshape(-1)

    teacher_student_agreement = float(
        np.mean(
            (student_scores >= float(threshold))
            == (teacher_scores >= float(threshold))
        )
    )
    probability_mae = float(np.mean(np.abs(student_scores - teacher_scores)))

    train_summary = {
        "model_type": "mlp",
        "num_samples": int(len(X)),
        "input_dim": int(X.shape[1]),
        "target_shape": tuple(y.shape),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "logit_loss_weight": float(logit_loss_weight),
        "hidden_loss_weight": float(hidden_loss_weight),
        "teacher_hidden_dim": teacher_hidden_dim,
        "student_hidden_dim": student_hidden_dim,
        "random_state": int(random_state),
        "device": str(resolved_device),
        "num_batches_per_epoch": int(len(loader)),
        "train_seconds": train_seconds,
        "final_loss": float(loss_history[-1]) if loss_history else None,
        "loss_history": loss_history,
        "probability_loss_history": probability_loss_history,
        "logit_loss_history": logit_loss_history,
        "hidden_loss_history": hidden_loss_history,
        "teacher_student_agreement": teacher_student_agreement,
        "probability_mae": probability_mae,
        "threshold": float(threshold),
        "include_rotations": bool(include_rotations),
        "features_per_joint": 7 if include_rotations else 3,
        "participants": sorted(
            {
                str(row["participant_id"])
                for row in usable_rows
                if row.get("participant_id") is not None
            }
        ),
    }

    if output_path is not None:
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": "mlp",
                "input_dim": int(X.shape[1]),
                "model_state_dict": student_model.state_dict(),
                "joint_order": list(QUEST_JOINT_ORDER),
                "shoulder_centered": True,
                "include_rotations": bool(include_rotations),
                "features_per_joint": 7 if include_rotations else 3,
                "target_type": "teacher_probability_logit_and_hidden",
                "logit_loss_weight": float(logit_loss_weight),
                "hidden_loss_weight": float(hidden_loss_weight),
                "teacher_hidden_dim": teacher_hidden_dim,
                "student_hidden_dim": student_hidden_dim,
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "random_state": int(random_state),
                "threshold": float(threshold),
            },
            path_obj,
        )
        print(f"Saved Quest student MLP to: {path_obj}")

    return student_model, train_summary


def train_quest_student_tabm(
    rows,
    teacher_model,
    teacher_model_type="tabm",
    student_features_type="keypoints",
    learning_rate: float = 2e-3,
    weight_decay: float = 3e-4,
    logit_loss_weight: float = 1.0,
    batch_size: int = 32,
    epochs: int = 30,
    k: int = 32,
    random_state: int = 0,
    device: str = "auto",
    threshold: float = 0.5,
    include_rotations: bool = False,
    output_path=None,
):
    if str(teacher_model_type).lower() != "tabm":
        raise ValueError("TabM student mode expects a TabM teacher.")

    if device == "auto":
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_state))

    X, y, _, usable_rows = construct_quest_training_data(
        rows=rows,
        teacher_model=teacher_model,
        teacher_model_type=teacher_model_type,
        device=str(resolved_device),
        include_rotations=include_rotations,
        student_features_type=student_features_type,
    )

    QuestTabM = get_quest_tabm_class()
    student_model = QuestTabM(
        input_size=int(X.shape[1]),
        output_size=1,
        k=int(k),
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    probability_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    logit_criterion = torch.nn.SmoothL1Loss()
    logit_epsilon = 1e-6

    x_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=int(batch_size),
        shuffle=True,
    )

    loss_history = []
    probability_loss_history = []
    logit_loss_history = []
    train_start = time.perf_counter()
    student_model.train()

    for epoch in range(int(epochs)):
        running_loss = 0.0
        running_probability_loss = 0.0
        running_logit_loss = 0.0
        sample_count = 0

        for xb, yb in loader:
            xb = xb.to(resolved_device)
            yb = yb.to(resolved_device)
            optimizer.zero_grad()

            member_logits = student_model(xb)
            if member_logits.ndim == 2:
                member_logits = member_logits.unsqueeze(-1)
            if member_logits.ndim != 3 or member_logits.shape[-1] != 1:
                raise ValueError(
                    "Quest TabM must return [batch,k,1] logits; got "
                    f"{tuple(member_logits.shape)}"
                )

            expanded_targets = yb.unsqueeze(1).expand(
                -1,
                member_logits.shape[1],
                -1,
            )
            probability_loss = probability_criterion(
                member_logits,
                expanded_targets,
            )

            student_probabilities = tabm_logits_to_probabilities(member_logits)
            teacher_logits = torch.logit(
                yb.clamp(logit_epsilon, 1.0 - logit_epsilon)
            )
            student_logits = torch.logit(
                student_probabilities.clamp(
                    logit_epsilon,
                    1.0 - logit_epsilon,
                )
            )
            logit_loss = logit_criterion(student_logits, teacher_logits)
            loss = probability_loss + float(logit_loss_weight) * logit_loss

            loss.backward()
            optimizer.step()

            current_batch_size = int(xb.shape[0])
            running_loss += float(loss.item()) * current_batch_size
            running_probability_loss += (
                float(probability_loss.item()) * current_batch_size
            )
            running_logit_loss += float(logit_loss.item()) * current_batch_size
            sample_count += current_batch_size

        epoch_loss = running_loss / max(sample_count, 1)
        epoch_probability_loss = running_probability_loss / max(sample_count, 1)
        epoch_logit_loss = running_logit_loss / max(sample_count, 1)
        loss_history.append(epoch_loss)
        probability_loss_history.append(epoch_probability_loss)
        logit_loss_history.append(epoch_logit_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"distillation_loss={epoch_loss:.6f}, "
            f"member_soft_bce={epoch_probability_loss:.6f}, "
            f"ensemble_logit_loss={epoch_logit_loss:.6f}"
        )

    train_seconds = float(time.perf_counter() - train_start)
    student_scores = predict_model_scores(
        student_model,
        "tabm",
        X,
        device=str(resolved_device),
    ).reshape(-1)
    teacher_scores = y.reshape(-1)

    teacher_student_agreement = float(
        np.mean(
            (student_scores >= float(threshold))
            == (teacher_scores >= float(threshold))
        )
    )
    probability_mae = float(np.mean(np.abs(student_scores - teacher_scores)))

    train_summary = {
        "model_type": "tabm",
        "num_samples": int(len(X)),
        "input_dim": int(X.shape[1]),
        "target_shape": tuple(y.shape),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "tabm_k": int(k),
        "logit_loss_weight": float(logit_loss_weight),
        "hidden_loss_weight": 0.0,
        "hidden_distillation_used": False,
        "random_state": int(random_state),
        "device": str(resolved_device),
        "num_batches_per_epoch": int(len(loader)),
        "train_seconds": train_seconds,
        "final_loss": float(loss_history[-1]) if loss_history else None,
        "loss_history": loss_history,
        "probability_loss_history": probability_loss_history,
        "logit_loss_history": logit_loss_history,
        "teacher_student_agreement": teacher_student_agreement,
        "probability_mae": probability_mae,
        "threshold": float(threshold),
        "include_rotations": bool(include_rotations),
        "features_per_joint": 7 if include_rotations else 3,
        "participants": sorted(
            {
                str(row["participant_id"])
                for row in usable_rows
                if row.get("participant_id") is not None
            }
        ),
    }

    if output_path is not None:
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": "tabm",
                "input_dim": int(X.shape[1]),
                "output_dim": 1,
                "tabm_k": int(k),
                "model_state_dict": student_model.state_dict(),
                "joint_order": list(QUEST_JOINT_ORDER),
                "shoulder_centered": True,
                "include_rotations": bool(include_rotations),
                "features_per_joint": 7 if include_rotations else 3,
                "target_type": "teacher_soft_probability_and_logit",
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "logit_loss_weight": float(logit_loss_weight),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "random_state": int(random_state),
                "threshold": float(threshold),
            },
            path_obj,
        )
        print(f"Saved Quest student TabM to: {path_obj}")

    return student_model, train_summary


def train_quest_student_xgboost(
    rows,
    teacher_model,
    teacher_model_type="xgboost",
    student_features_type="keypoints",
    random_state: int = 0,
    device: str = "auto",
    threshold: float = 0.5,
    include_rotations: bool = False,
    output_path=None,
):
    if str(teacher_model_type).lower() != "xgboost":
        raise ValueError("XGBoost student mode expects an XGBoost teacher.")

    X, y, _, usable_rows = construct_quest_training_data(
        rows=rows,
        teacher_model=teacher_model,
        teacher_model_type=teacher_model_type,
        device=device,
        include_rotations=include_rotations,
        student_features_type=student_features_type,
    )

    # Distillation target is continuous teacher probability, so the Quest
    # XGBoost student is a regressor rather than a hard-label classifier.
    student_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=int(random_state),
        n_jobs=-1,
    )

    train_start = time.perf_counter()
    student_model.fit(X, y.reshape(-1))
    train_seconds = float(time.perf_counter() - train_start)

    student_scores = np.clip(
        np.asarray(student_model.predict(X), dtype=np.float32).reshape(-1),
        0.0,
        1.0,
    )
    teacher_scores = y.reshape(-1)
    probability_mse = float(
        np.mean((student_scores - teacher_scores) ** 2)
    )
    probability_mae = float(
        np.mean(np.abs(student_scores - teacher_scores))
    )
    teacher_student_agreement = float(
        np.mean(
            (student_scores >= float(threshold))
            == (teacher_scores >= float(threshold))
        )
    )

    train_summary = {
        "model_type": "xgboost",
        "student_objective": "teacher_probability_regression",
        "num_samples": int(len(X)),
        "input_dim": int(X.shape[1]),
        "target_shape": tuple(y.shape),
        "epochs": None,
        "batch_size": None,
        "learning_rate": None,
        "logit_loss_weight": 0.0,
        "hidden_loss_weight": 0.0,
        "hidden_distillation_used": False,
        "random_state": int(random_state),
        "device": "cpu",
        "train_seconds": train_seconds,
        "final_loss": probability_mse,
        "teacher_student_agreement": teacher_student_agreement,
        "probability_mae": probability_mae,
        "probability_mse": probability_mse,
        "threshold": float(threshold),
        "include_rotations": bool(include_rotations),
        "features_per_joint": 7 if include_rotations else 3,
        "participants": sorted(
            {
                str(row["participant_id"])
                for row in usable_rows
                if row.get("participant_id") is not None
            }
        ),
    }

    if output_path is not None:
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": "xgboost",
                "student_objective": "teacher_probability_regression",
                "input_dim": int(X.shape[1]),
                "random_state": int(random_state),
                "threshold": float(threshold),
                "include_rotations": bool(include_rotations),
                "features_per_joint": 7 if include_rotations else 3,
                "joint_order": list(QUEST_JOINT_ORDER),
                "model": student_model,
            },
            path_obj,
        )
        print(f"Saved Quest student XGBoost to: {path_obj}")

    return student_model, train_summary


def train_quest_student(
    rows,
    teacher_model,
    model_type,
    student_features_type="keypoints",
    learning_rate=None,
    weight_decay=3e-4,
    logit_loss_weight=1.0,
    hidden_loss_weight=0.1,
    batch_size=32,
    epochs=30,
    tabm_k=32,
    random_state=0,
    device="auto",
    threshold=0.5,
    include_rotations=False,
    output_path=None,
):
    model_type = str(model_type).lower()

    if model_type == "mlp":
        resolved_lr = 1e-3 if learning_rate is None else float(learning_rate)
        return train_quest_student_mlp(
            rows=rows,
            teacher_model=teacher_model,
            teacher_model_type="mlp",
            student_features_type=student_features_type,
            learning_rate=resolved_lr,
            logit_loss_weight=logit_loss_weight,
            hidden_loss_weight=hidden_loss_weight,
            batch_size=batch_size,
            epochs=epochs,
            random_state=random_state,
            device=device,
            threshold=threshold,
            include_rotations=include_rotations,
            output_path=output_path,
        )

    if model_type == "tabm":
        resolved_lr = 2e-3 if learning_rate is None else float(learning_rate)
        if float(hidden_loss_weight) != 0.0:
            print(
                "Note: --hidden-loss-weight is ignored for TabM because "
                "the TabM ensemble does not expose the same single "
                "penultimate hidden representation as the MLP."
            )
        return train_quest_student_tabm(
            rows=rows,
            teacher_model=teacher_model,
            teacher_model_type="tabm",
            student_features_type=student_features_type,
            learning_rate=resolved_lr,
            weight_decay=weight_decay,
            logit_loss_weight=logit_loss_weight,
            batch_size=batch_size,
            epochs=epochs,
            k=tabm_k,
            random_state=random_state,
            device=device,
            threshold=threshold,
            include_rotations=include_rotations,
            output_path=output_path,
        )

    if model_type == "xgboost":
        if float(hidden_loss_weight) != 0.0 or float(logit_loss_weight) != 0.0:
            print(
                "Note: logit/hidden neural distillation weights are ignored "
                "for XGBoost. The student directly regresses the teacher's "
                "soft probability."
            )
        return train_quest_student_xgboost(
            rows=rows,
            teacher_model=teacher_model,
            teacher_model_type="xgboost",
            student_features_type=student_features_type,
            random_state=random_state,
            device=device,
            threshold=threshold,
            include_rotations=include_rotations,
            output_path=output_path,
        )

    raise ValueError(f"Unsupported model_type={model_type}")


def compute_binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    total = int(len(y_true))
    correct = int(np.sum(y_true == y_pred))
    accuracy = float(correct / total) if total > 0 else 0.0

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and (precision + recall) > 0
        else 0.0
    )

    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = float(0.5 * (recall + specificity))

    return {
        "total": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def evaluate_quest_student(
    model,
    teacher_model,
    model_type,
    test_rows,
    device: str,
    student_features_type="keypoints",
    threshold: float = 0.5,
    include_rotations: bool = False,
):
    """Evaluate teacher-student agreement on held-out rows."""
    X_test, teacher_targets, _, usable_rows = construct_quest_training_data(
        rows=test_rows,
        teacher_model=teacher_model,
        teacher_model_type=model_type,
        device=device,
        include_rotations=include_rotations,
        student_features_type=student_features_type,
    )

    expected_input_dim = get_model_input_dim(model, model_type=model_type)
    actual_input_dim = int(X_test.shape[1])
    if actual_input_dim != expected_input_dim:
        raise ValueError(
            "Quest feature dimension mismatch: "
            f"model expects {expected_input_dim} features, "
            f"but evaluation produced {actual_input_dim}. "
            f"include_rotations={include_rotations}"
        )

    student_scores = predict_model_scores(
        model,
        model_type,
        X_test,
        device=device,
    ).reshape(-1).astype(np.float32)
    teacher_scores = teacher_targets.reshape(-1).astype(np.float32)

    teacher_predictions = (
        teacher_scores >= float(threshold)
    ).astype(np.int32)
    student_predictions = (
        student_scores >= float(threshold)
    ).astype(np.int32)

    agreement_metrics = compute_binary_metrics(
        y_true=teacher_predictions,
        y_pred=student_predictions,
    )
    probability_mae = float(
        np.mean(np.abs(student_scores - teacher_scores))
    )
    probability_mse = float(
        np.mean((student_scores - teacher_scores) ** 2)
    )

    ground_truth_indices = [
        index
        for index, row in enumerate(usable_rows)
        if row.get("label_binary") is not None
    ]

    ground_truth = []
    teacher_ground_truth_metrics = None
    student_ground_truth_metrics = None

    if ground_truth_indices:
        ground_truth = np.asarray(
            [
                int(usable_rows[index]["label_binary"])
                for index in ground_truth_indices
            ],
            dtype=np.int32,
        )
        teacher_ground_truth_metrics = compute_binary_metrics(
            y_true=ground_truth,
            y_pred=teacher_predictions[ground_truth_indices],
        )
        student_ground_truth_metrics = compute_binary_metrics(
            y_true=ground_truth,
            y_pred=student_predictions[ground_truth_indices],
        )

    participant_ids = sorted(
        {
            str(row["participant_id"])
            for row in usable_rows
            if row.get("participant_id") is not None
        }
    )

    return {
        "model_name": f"quest_student_{model_type}",
        "model_type": model_type,
        "num_samples": int(len(X_test)),
        "input_dim": int(X_test.shape[1]),
        "participants": participant_ids,
        "threshold": float(threshold),
        "include_rotations": bool(include_rotations),
        "agreement_metrics": agreement_metrics,
        "probability_mae": probability_mae,
        "probability_mse": probability_mse,
        "teacher_ground_truth_metrics": teacher_ground_truth_metrics,
        "student_ground_truth_metrics": student_ground_truth_metrics,
        "teacher_scores": teacher_scores.tolist(),
        "student_scores": student_scores.tolist(),
        "teacher_predictions": teacher_predictions.tolist(),
        "student_predictions": student_predictions.tolist(),
        "ground_truth": ground_truth.tolist() if len(ground_truth) else [],
        "teacher_predictions_with_ground_truth": (
            teacher_predictions[ground_truth_indices].tolist()
            if ground_truth_indices else []
        ),
        "student_predictions_with_ground_truth": (
            student_predictions[ground_truth_indices].tolist()
            if ground_truth_indices else []
        ),
    }


def leave_one_participant_out_evaluation(
    rows,
    teacher_model,
    model_type,
    learning_rate=None,
    weight_decay: float = 3e-4,
    logit_loss_weight: float = 1.0,
    hidden_loss_weight: float = 0.1,
    batch_size: int = 32,
    epochs: int = 30,
    tabm_k: int = 32,
    random_state: int = 0,
    device: str = "auto",
    threshold: float = 0.5,
    include_rotations: bool = False,
    student_features_type: str = "keypoints",
):
    """Perform participant-held-out teacher/student distillation evaluation."""
    usable_rows = []
    for row in rows:
        if row.get("skipped", False):
            continue
        if row.get("feature_vector") is None:
            continue
        if row.get("quest_joints_pth") is None:
            continue
        if row.get("participant_id") is None:
            continue
        usable_rows.append(row)

    participants = sorted(
        {str(row["participant_id"]) for row in usable_rows}
    )
    if len(participants) < 2:
        raise ValueError(
            "Leave-one-participant-out evaluation requires at least two participants."
        )

    fold_results = []
    all_teacher_predictions = []
    all_student_predictions = []
    all_teacher_scores = []
    all_student_scores = []
    all_ground_truth = []
    all_teacher_predictions_with_gt = []
    all_student_predictions_with_gt = []

    for fold_index, held_out_participant in enumerate(participants):
        print(
            f"\nLOPO fold {fold_index + 1}/{len(participants)}: "
            f"holding out participant {held_out_participant}"
        )

        train_rows = [
            row for row in usable_rows
            if str(row["participant_id"]) != held_out_participant
        ]
        test_rows = [
            row for row in usable_rows
            if str(row["participant_id"]) == held_out_participant
        ]

        if not train_rows:
            raise ValueError(f"No training rows for fold {held_out_participant}.")
        if not test_rows:
            raise ValueError(f"No test rows for fold {held_out_participant}.")

        fold_model, train_summary = train_quest_student(
            rows=train_rows,
            teacher_model=teacher_model,
            model_type=model_type,
            student_features_type=student_features_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logit_loss_weight=logit_loss_weight,
            hidden_loss_weight=hidden_loss_weight,
            batch_size=batch_size,
            epochs=epochs,
            tabm_k=tabm_k,
            random_state=random_state + fold_index,
            device=device,
            threshold=threshold,
            include_rotations=include_rotations,
            output_path=None,
        )

        fold_evaluation = evaluate_quest_student(
            model=fold_model,
            teacher_model=teacher_model,
            model_type=model_type,
            test_rows=test_rows,
            device=train_summary["device"],
            threshold=threshold,
            include_rotations=include_rotations,
            student_features_type=student_features_type,
        )

        fold_evaluation["fold_index"] = int(fold_index)
        fold_evaluation["held_out_participant"] = held_out_participant
        fold_evaluation["train_summary"] = train_summary
        fold_results.append(fold_evaluation)

        all_teacher_predictions.extend(fold_evaluation["teacher_predictions"])
        all_student_predictions.extend(fold_evaluation["student_predictions"])
        all_teacher_scores.extend(fold_evaluation["teacher_scores"])
        all_student_scores.extend(fold_evaluation["student_scores"])

        if fold_evaluation["ground_truth"]:
            all_ground_truth.extend(fold_evaluation["ground_truth"])
            all_teacher_predictions_with_gt.extend(
                fold_evaluation["teacher_predictions_with_ground_truth"]
            )
            all_student_predictions_with_gt.extend(
                fold_evaluation["student_predictions_with_ground_truth"]
            )

        agreement = fold_evaluation["agreement_metrics"]
        print(
            f"Participant {held_out_participant}: "
            f"agreement={agreement['accuracy']:.4f}, "
            f"probability_mae={fold_evaluation['probability_mae']:.4f}, "
            f"probability_mse={fold_evaluation['probability_mse']:.4f}"
        )

    pooled_agreement_metrics = compute_binary_metrics(
        y_true=all_teacher_predictions,
        y_pred=all_student_predictions,
    )

    teacher_scores_array = np.asarray(all_teacher_scores, dtype=np.float32)
    student_scores_array = np.asarray(all_student_scores, dtype=np.float32)
    pooled_probability_mae = float(
        np.mean(np.abs(student_scores_array - teacher_scores_array))
    )
    pooled_probability_mse = float(
        np.mean((student_scores_array - teacher_scores_array) ** 2)
    )

    teacher_ground_truth_metrics = None
    student_ground_truth_metrics = None
    if all_ground_truth:
        teacher_ground_truth_metrics = compute_binary_metrics(
            y_true=all_ground_truth,
            y_pred=all_teacher_predictions_with_gt,
        )
        student_ground_truth_metrics = compute_binary_metrics(
            y_true=all_ground_truth,
            y_pred=all_student_predictions_with_gt,
        )

    def summarize(values):
        values = [float(value) for value in values]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    participant_metric_summary = {
        "agreement_accuracy": summarize(
            [fold["agreement_metrics"]["accuracy"] for fold in fold_results]
        ),
        "probability_mae": summarize(
            [fold["probability_mae"] for fold in fold_results]
        ),
        "probability_mse": summarize(
            [fold["probability_mse"] for fold in fold_results]
        ),
    }

    return {
        "evaluation_type": "leave_one_participant_out_distillation",
        "model_type": model_type,
        "teacher_model_type": model_type,
        "student_model_type": model_type,
        "num_participants": len(participants),
        "participants": participants,
        "num_folds": len(fold_results),
        "include_rotations": bool(include_rotations),
        "folds": fold_results,
        "pooled_agreement_metrics": pooled_agreement_metrics,
        "pooled_probability_mae": pooled_probability_mae,
        "pooled_probability_mse": pooled_probability_mse,
        "teacher_ground_truth_metrics": teacher_ground_truth_metrics,
        "student_ground_truth_metrics": student_ground_truth_metrics,
        "participant_metric_summary": participant_metric_summary,
        "teacher_predictions": all_teacher_predictions,
        "student_predictions": all_student_predictions,
        "teacher_scores": all_teacher_scores,
        "student_scores": all_student_scores,
        "ground_truth": all_ground_truth,
    }


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    dataset_root = REPO_ROOT / config["dataset"]["root"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-classes", default="cup")
    parser.add_argument(
        "--crop-around-object",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--crop-object", default="person.")
    parser.add_argument(
        "--normalize-keypoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-quest-joint-rotations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--model-type", type=str, choices=MODEL_TYPES, default="mlp", help="Matched teacher/student model family: mlp, xgboost, or tabm.")
    parser.add_argument("--teacher-features-type", type=str, choices=TEACHER_FEATURES_TYPE, default="keypoints")
    parser.add_argument("--student-features-type", type=str, choices=STUDENT_FEATURES_TYPE, default="keypoints")
    parser.add_argument("--participant", type=str, default=None, help="Filter to one participant ID, e.g. 1 or P01")
    parser.add_argument("--label", type=str, default=None, help="Filter to label_name, e.g. handoff or not_handoff")
    parser.add_argument("--condition", type=str, default=None, help="Filter to one condition")
    parser.add_argument("--hand", type=str, choices=["left", "right"], default=None, help="Filter to one hand")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--student-epochs", "--mlp-epochs", dest="student_epochs", type=int, default=30)
    parser.add_argument("--student-batch-size", "--mlp-batch-size", dest="student_batch_size", type=int, default=32)
    parser.add_argument(
        "--student-learning-rate",
        "--mlp-learning-rate",
        dest="student_learning_rate",
        type=float,
        default=None,
        help="Student learning rate. Default: 1e-3 for MLP, 2e-3 for TabM; unused by XGBoost.",
    )
    parser.add_argument("--tabm-weight-decay", type=float, default=3e-4)
    parser.add_argument("--tabm-k", type=int, default=32)
    parser.add_argument(
        "--logit-loss-weight",
        "--mlp-logit-loss-weight",
        dest="logit_loss_weight",
        type=float,
        default=1.0,
        help="Logit-matching weight for neural students. Ignored by XGBoost.",
    )
    parser.add_argument(
        "--hidden-loss-weight",
        "--mlp-hidden-loss-weight",
        dest="hidden_loss_weight",
        type=float,
        default=0.1,
        help="Penultimate-hidden matching weight. Used only for MLP -> MLP.",
    )
    parser.add_argument("--student-random-state", "--mlp-random-state", dest="student_random_state", type=int, default=0)
    parser.add_argument("--student-output-path", "--mlp-output-path", dest="student_output_path", type=str, default=None)
    parser.add_argument(
        "--teacher-weights-path",
        type=str,
        default=None,
        help="Optional explicit teacher checkpoint/joblib path. Otherwise a model-type-specific default is used.",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=None,
        help=(
            "Root directory for evaluation bundles. Defaults to "
            "MODEL_IMPL_ROOT/outputs/eval_results/<model-type>."
        ),
    )
    parser.add_argument(
        "--eval-run-name",
        type=str,
        default=None,
        help=(
            "Optional human-readable variation name. When omitted, a name is "
            "constructed from the selected feature configuration."
        ),
    )

    args = parser.parse_args()

    run_started_at = datetime.now().astimezone()
    run_id = run_started_at.strftime("%Y%m%dT%H%M%S_%f%z")
    variation_name = args.eval_run_name or build_model_variation_name(args)

    print(
        "Quest feature configuration: "
        f"include_rotations={args.include_quest_joint_rotations}, "
        f"num_joints={len(QUEST_JOINT_ORDER)}, "
        f"expected_input_dim="
        f"{len(QUEST_JOINT_ORDER) * (7 if args.include_quest_joint_rotations else 3)}"
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


    object_detector = DINOObjectDetector(
        model_id="IDEA-Research/grounding-dino-base",
        confidence=args.confidence,
    )

    keypoint_detector = RTMPoseKeypointDetector(
        confidence=args.confidence,
        device="cuda",
    )

    head_pose_estimator = RTMPoseHeadPoseEstimator(
        weights_path=REPO_ROOT / "model_training_and_implementation" / "kwan_pretrained_weights" / "head-pose-pretrained.pkl",
        gpu_id=0,
    )

    resnet_encoder = ResNet18ImageEncoder(pretrained=True, device="cuda", l2_normalize=True)

    teacher_variation_name = (
        "features-"
        + str(args.teacher_features_type)
        + "__crop-"
        + str(args.crop_around_object)
    )

    if args.teacher_weights_path is not None:
        HAND_INTENT_WEIGHTS_PATH = Path(args.teacher_weights_path)
    elif args.model_type == "mlp":
        HAND_INTENT_WEIGHTS_PATH = (
            REPO_ROOT
            / "model_training_and_implementation"
            / "outputs"
            / "hand_intent_mlp_weights"
            / teacher_variation_name
            / "MLP.pth"
        )
    elif args.model_type == "xgboost":
        HAND_INTENT_WEIGHTS_PATH = (
            REPO_ROOT
            / "model_training_and_implementation"
            / "outputs"
            / "hand_intent_classical_models"
            / teacher_variation_name
            / "xgboost.joblib"
        )
    elif args.model_type == "tabm":
        HAND_INTENT_WEIGHTS_PATH = (
            REPO_ROOT
            / "model_training_and_implementation"
            / "outputs"
            / "hand_intent_tabm"
            / teacher_variation_name
            / "TabM.pth"
        )
    else:
        raise ValueError(f"Unsupported model_type={args.model_type}")

    if not HAND_INTENT_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Teacher weights do not exist: {HAND_INTENT_WEIGHTS_PATH}"
        )

    model = load_teacher_model(
        model_type=args.model_type,
        weights_path=HAND_INTENT_WEIGHTS_PATH,
    )

    print(
        f"Loaded {args.model_type} teacher from: "
        f"{HAND_INTENT_WEIGHTS_PATH}"
    )

    y_true, rows, skipped_unreadable, skipped_no_object, skipped_no_people = organize_samples(
        samples=samples,
        features_type=args.teacher_features_type,
        obj_detector=object_detector,
        keypoint_detector=keypoint_detector,
        head_pose_estimator=head_pose_estimator,
        img_encoder=resnet_encoder,
        args=args,
    )

    lopo_evaluation = leave_one_participant_out_evaluation(
        rows=rows,
        teacher_model=model,
        model_type=args.model_type,
        learning_rate=args.student_learning_rate,
        weight_decay=args.tabm_weight_decay,
        logit_loss_weight=args.logit_loss_weight,
        hidden_loss_weight=args.hidden_loss_weight,
        batch_size=args.student_batch_size,
        epochs=args.student_epochs,
        tabm_k=args.tabm_k,
        random_state=args.student_random_state,
        device="auto",
        threshold=args.decision_threshold,
        include_rotations=args.include_quest_joint_rotations,
        student_features_type=args.student_features_type,
    )

    print("\nQuest student LOPO evaluation complete")
    print(
        "Pooled teacher-student agreement:",
        lopo_evaluation["pooled_agreement_metrics"],
    )
    print(
        "Pooled probability MAE:",
        lopo_evaluation["pooled_probability_mae"],
    )
    print(
        "Pooled probability MSE:",
        lopo_evaluation["pooled_probability_mse"],
    )

    if lopo_evaluation["student_ground_truth_metrics"] is not None:
        print(
            "Student metrics against dataset labels:",
            lopo_evaluation["student_ground_truth_metrics"],
        )
        print(
            "Teacher metrics against dataset labels:",
            lopo_evaluation["teacher_ground_truth_metrics"],
        )
    

    print("\nPer-participant LOPO results")

    for fold in lopo_evaluation["folds"]:
        participant = fold["held_out_participant"]

        agreement = fold["agreement_metrics"]
        student_metrics = fold.get("student_ground_truth_metrics")
        teacher_metrics = fold.get("teacher_ground_truth_metrics")

        print(f"\nParticipant: {participant}")
        print(f"  Samples: {agreement['total']}")
        print(
            f"  Teacher-student agreement: "
            f"{agreement['accuracy']:.4f}"
        )
        print(
            f"  Balanced agreement: "
            f"{agreement['balanced_accuracy']:.4f}"
        )
        print(
            f"  Probability MAE: "
            f"{fold['probability_mae']:.4f}"
        )
        print(
            f"  Probability MSE: "
            f"{fold['probability_mse']:.4f}"
        )

        if student_metrics is not None:
            print(
                f"  Student accuracy vs labels: "
                f"{student_metrics['accuracy']:.4f}"
            )
            print(
                f"  Student balanced accuracy vs labels: "
                f"{student_metrics['balanced_accuracy']:.4f}"
            )

        if teacher_metrics is not None:
            print(
                f"  Teacher accuracy vs labels: "
                f"{teacher_metrics['accuracy']:.4f}"
            )

    final_output_path = args.student_output_path
    if final_output_path is None:
        if args.model_type == "mlp":
            final_output_path = (
                MODEL_IMPL_ROOT
                / "outputs"
                / "quest_hand_intent_mlp_weights"
                / sanitize_path_component(variation_name)
                / "MLP.pth"
            )
        elif args.model_type == "tabm":
            final_output_path = (
                MODEL_IMPL_ROOT
                / "outputs"
                / "quest_hand_intent_tabm"
                / sanitize_path_component(variation_name)
                / "TabM.pth"
            )
        elif args.model_type == "xgboost":
            final_output_path = (
                MODEL_IMPL_ROOT
                / "outputs"
                / "quest_hand_intent_xgboost"
                / sanitize_path_component(variation_name)
                / "xgboost.joblib"
            )
        else:
            raise ValueError(f"Unsupported model_type={args.model_type}")
    else:
        final_output_path = Path(final_output_path)

    final_model, final_train_summary = train_quest_student(
        rows=rows,
        teacher_model=model,
        model_type=args.model_type,
        student_features_type=args.student_features_type,
        learning_rate=args.student_learning_rate,
        weight_decay=args.tabm_weight_decay,
        logit_loss_weight=args.logit_loss_weight,
        hidden_loss_weight=args.hidden_loss_weight,
        batch_size=args.student_batch_size,
        epochs=args.student_epochs,
        tabm_k=args.tabm_k,
        random_state=args.student_random_state,
        device="auto",
        threshold=args.decision_threshold,
        include_rotations=args.include_quest_joint_rotations,
        output_path=final_output_path,
    )
    print(f"\nFinal Quest student saved to: {final_output_path}")
    print("Final training summary:", final_train_summary)

    dataset_summary = {
        "model_type": args.model_type,
        "num_samples_after_discovery_filters_and_slicing": int(len(samples)),
        "num_rows_after_feature_organization": int(len(rows)),
        "num_labels_returned": int(len(y_true)),
        "skipped_unreadable": int(skipped_unreadable),
        "skipped_no_object": int(skipped_no_object),
        "skipped_no_people": int(skipped_no_people),
        "participant_filter": args.participant,
        "label_filter": args.label,
        "condition_filter": args.condition,
        "hand_filter": args.hand,
        "start_index": int(args.start_index),
        "max_samples": args.max_samples,
    }

    saved_eval_paths = save_evaluation_results(
        lopo_evaluation=lopo_evaluation,
        final_train_summary=final_train_summary,
        args=args,
        run_id=run_id,
        variation_name=variation_name,
        run_started_at=run_started_at,
        final_output_path=final_output_path,
        teacher_weights_path=HAND_INTENT_WEIGHTS_PATH,
        dataset_root=dataset_root,
        dataset_summary=dataset_summary,
    )

    print(f"Evaluation results saved to: {saved_eval_paths['run_dir']}")
    print(f"  Full JSON: {saved_eval_paths['evaluation_json']}")
    print(f"  Summary CSV: {saved_eval_paths['summary_csv']}")
    print(f"  Fold metrics CSV: {saved_eval_paths['fold_metrics_csv']}")

    return {
        "lopo_evaluation": lopo_evaluation,
        "final_train_summary": final_train_summary,
        "final_output_path": str(final_output_path),
        "saved_eval_paths": saved_eval_paths,
        "run_id": run_id,
        "variation_name": variation_name,
    }


if __name__ == "__main__":
    main()