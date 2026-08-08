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
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_training_and_implementation.src.dino_detector import DINOObjectDetector
from model_training_and_implementation.src.rtmpose_keypoints import RTMPoseKeypointDetector
from model_training_and_implementation.src.rtmpose_headpose import RTMPoseHeadPoseEstimator
from model_training_and_implementation.src.hand_intent_mlp import HandIntentMLP
from model_training_and_implementation.src.resnet_encoder import ResNet18ImageEncoder


from quest_hand_intent_model_est.src.quest_hand_intent_est_mlp import QuestHandIntentEstimatorMLP
from quest_hand_intent_model_est.src.quest_joint_features import extract_quest_joint_features, construct_pose_img_from_joint_feature_vector, QUEST_JOINT_ORDER


TEACHER_FEATURES_TYPE = ["keypoints","keypoints_headpose","keypoints_headpose_dino","keypoints_resnet","keypoints_headpose_resnet"]
STUDENT_FEATURES_TYPE = ["keypoints","keypoints_projections","keypoints_resnet"]


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
    """Build a descriptive name for the feature/model configuration."""
    parts = [
        f"teacher_features-{args.teacher_features_type}",
        f"student_features-{args.student_features_type}",
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
        else MODEL_IMPL_ROOT / "outputs" / "eval_results"
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


def get_model_input_dim(model):
    """Return the input size expected by the model's first Linear layer."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)

    raise ValueError("Could not find a Linear layer in the model.")




def construct_quest_training_data(rows, old_model, device="auto", include_rotations=False, student_features_type="keypoints"):
    """
    Construct aligned Quest inputs and teacher-model targets.

    Returns:
        X_quest: Quest joint features, shape (N, D_quest)
        y_teacher: teacher probabilities, shape (N, 1)
        usable_rows: rows corresponding exactly to those N samples
    """
    if device == "auto":
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    resnet_encoder = ResNet18ImageEncoder(pretrained=True, device="cuda", l2_normalize=True)

    old_model = old_model.to(resolved_device)
    old_model.eval()

    quest_feature_vectors = []
    old_model_feature_vectors = []
    usable_rows = []

    for row in rows:
        json_path = row.get("quest_joints_pth")
        old_features = row.get("feature_vector")

        if json_path is None:
            print("Skipping row: missing quest_joints_pth")
            continue

        if old_features is None:
            print(f"Skipping {json_path}: missing old model feature_vector")
            continue

        try:
            quest_features = np.asarray(
                extract_quest_joint_features(json_path, include_rotations=include_rotations),
                dtype=np.float32,
            )

            if(student_features_type == "keypoints_projections"):
                proj_joints, pose_img = construct_pose_img_from_joint_feature_vector(quest_features)
                quest_features = np.concatenate([quest_features, proj_joints], axis=0)
            elif(student_features_type == "keypoints_resnet"):
                proj_joints, pose_img = construct_pose_img_from_joint_feature_vector(quest_features)
                resnet_embedding = resnet_encoder.predict(pose_img)["embedding"]
                quest_features = np.concatenate([quest_features, resnet_embedding.flatten().astype(np.float32)], axis=0)
                               

            teacher_features = np.asarray(
                old_features,
                dtype=np.float32,
            )

            if quest_features.ndim != 1:
                raise ValueError(
                    f"Quest feature vector must be 1D, got {quest_features.shape}"
                )

            if teacher_features.ndim != 1:
                raise ValueError(
                    f"Teacher feature vector must be 1D, got {teacher_features.shape}"
                )

            quest_feature_vectors.append(quest_features)
            old_model_feature_vectors.append(teacher_features)
            usable_rows.append(row)

        except Exception as error:
            print(f"Skipping {json_path}: {error}")

    if not quest_feature_vectors:
        raise ValueError("No usable Quest samples were found.")

    X_quest = np.stack(quest_feature_vectors).astype(np.float32)
    X_old = np.stack(old_model_feature_vectors).astype(np.float32)

    old_input_tensor = torch.from_numpy(X_old).to(resolved_device)

    with torch.no_grad():
        y_teacher = (
            old_model(old_input_tensor)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 1)
            .astype(np.float32)
        )

    return X_quest, y_teacher, usable_rows

    


def train_quest_student_mlp(
    rows,
    teacher_model,
    student_features_type:str = "keypoints",
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 30,
    random_state: int = 0,
    device: str = "auto",
    threshold: float = 0.5,
    include_rotations: bool = False,
    output_path=None,
):
    # Resolve the device used by the student model and training batches.
    if device == "auto":
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    torch.manual_seed(int(random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_state))

    # X contains shoulder-centered Quest joint features.
    # y contains the teacher model's soft probability for each aligned sample.
    X, y, usable_rows = construct_quest_training_data(
        rows=rows,
        old_model=teacher_model,
        device=str(resolved_device),
        include_rotations=include_rotations,
        student_features_type=student_features_type,
    )

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")

    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(
            f"Expected teacher targets with shape (N, 1), got {y.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"Quest inputs and teacher targets are misaligned: "
            f"{len(X)} != {len(y)}"
        )

    x_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
    )

    student_model = QuestHandIntentEstimatorMLP(
        input_size=int(X.shape[1]),
        output_size=1,
    ).to(resolved_device)

    optimizer = torch.optim.Adam(
        student_model.parameters(),
        lr=float(learning_rate),
    )

    # This assumes the student model returns a sigmoid probability.
    # Soft teacher probabilities are valid targets for BCELoss.
    criterion = torch.nn.BCELoss()

    loss_history = []
    train_start = time.perf_counter()

    student_model.train()

    for epoch in range(int(epochs)):
        running_loss = 0.0
        sample_count = 0

        for xb, yb in loader:
            xb = xb.to(resolved_device)
            yb = yb.to(resolved_device)

            optimizer.zero_grad()

            student_probabilities = student_model(xb).reshape(-1, 1)
            loss = criterion(student_probabilities, yb)

            loss.backward()
            optimizer.step()

            current_batch_size = int(xb.shape[0])
            running_loss += float(loss.item()) * current_batch_size
            sample_count += current_batch_size

        epoch_loss = running_loss / max(sample_count, 1)
        loss_history.append(epoch_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"distillation_loss={epoch_loss:.6f}"
        )

    train_seconds = float(time.perf_counter() - train_start)

    # Measure how closely the trained student reproduces the teacher
    # on the data supplied to this function.
    student_model.eval()
    with torch.no_grad():
        student_scores = (
            student_model(x_tensor.to(resolved_device))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    teacher_scores = y.reshape(-1)

    teacher_student_agreement = float(
        np.mean(
            (student_scores >= float(threshold))
            == (teacher_scores >= float(threshold))
        )
    )

    probability_mae = float(
        np.mean(np.abs(student_scores - teacher_scores))
    )

    train_summary = {
        "num_samples": int(len(X)),
        "input_dim": int(X.shape[1]),
        "target_shape": tuple(y.shape),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "random_state": int(random_state),
        "device": str(resolved_device),
        "num_batches_per_epoch": int(len(loader)),
        "train_seconds": train_seconds,
        "final_loss": (
            float(loss_history[-1])
            if loss_history
            else None
        ),
        "loss_history": loss_history,
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
                "input_dim": int(X.shape[1]),
                "model_state_dict": student_model.state_dict(),
                "joint_order": list(QUEST_JOINT_ORDER),
                "shoulder_centered": True,
                "include_rotations": bool(include_rotations),
                "features_per_joint": 7 if include_rotations else 3,
                "target_type": "teacher_probability",
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


def evaluate_quest_student_mlp(
    model,
    teacher_model,
    test_rows,
    device: str,
    student_features_type:str = "keypoints",
    threshold: float = 0.5,
    include_rotations: bool = False,
):
    """
    Evaluate the Quest student on held-out rows.

    The primary target is the fixed teacher model's output. Binary metrics
    therefore measure teacher-student agreement, while MAE/MSE measure how
    closely the student reproduces the teacher probability.
    """
    resolved_device = torch.device(device)

    X_test, teacher_targets, usable_rows = construct_quest_training_data(
        rows=test_rows,
        old_model=teacher_model,
        device=str(resolved_device),
        include_rotations=include_rotations,
        student_features_type=student_features_type
    )

    expected_input_dim = get_model_input_dim(model)
    actual_input_dim = int(X_test.shape[1])

    if actual_input_dim != expected_input_dim:
        raise ValueError(
            "Quest feature dimension mismatch: "
            f"model expects {expected_input_dim} features, "
            f"but evaluation produced {actual_input_dim}. "
            f"include_rotations={include_rotations}"
        )

    model = model.to(resolved_device)
    model.eval()

    x_test_tensor = torch.from_numpy(
        X_test.astype(np.float32)
    ).to(resolved_device)

    with torch.no_grad():
        student_scores = (
            model(x_test_tensor)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32)
        )

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
        "model_name": "quest_student_mlp",
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
            if ground_truth_indices
            else []
        ),
        "student_predictions_with_ground_truth": (
            student_predictions[ground_truth_indices].tolist()
            if ground_truth_indices
            else []
        ),
    }


def leave_one_participant_out_evaluation(
    rows,
    teacher_model,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 30,
    random_state: int = 0,
    device: str = "auto",
    threshold: float = 0.5,
    include_rotations: bool = False,
    student_features_type: str = "keypoints",
):
    """
    Perform leave-one-participant-out evaluation of the Quest student.

    For each fold, the student is trained from scratch on all participants
    except one and evaluated on the held-out participant. The fixed teacher
    model supplies soft targets for both training and evaluation.
    """
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
            "Leave-one-participant-out evaluation requires at least "
            "two participants."
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
            row
            for row in usable_rows
            if str(row["participant_id"]) != held_out_participant
        ]
        test_rows = [
            row
            for row in usable_rows
            if str(row["participant_id"]) == held_out_participant
        ]

        if not train_rows:
            raise ValueError(
                f"No training rows for fold {held_out_participant}."
            )
        if not test_rows:
            raise ValueError(
                f"No test rows for fold {held_out_participant}."
            )

        fold_model, train_summary = train_quest_student_mlp(
            rows=train_rows,
            teacher_model=teacher_model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            random_state=random_state + fold_index,
            device=device,
            threshold=threshold,
            include_rotations=include_rotations,
            student_features_type = student_features_type,
            output_path=None,
        )

        fold_evaluation = evaluate_quest_student_mlp(
            model=fold_model,
            teacher_model=teacher_model,
            test_rows=test_rows,
            device=train_summary["device"],
            threshold=threshold,
            include_rotations=include_rotations,
            student_features_type = student_features_type
        )

        fold_evaluation["fold_index"] = int(fold_index)
        fold_evaluation["held_out_participant"] = held_out_participant
        fold_evaluation["train_summary"] = train_summary
        fold_results.append(fold_evaluation)

        all_teacher_predictions.extend(
            fold_evaluation["teacher_predictions"]
        )
        all_student_predictions.extend(
            fold_evaluation["student_predictions"]
        )
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

    teacher_scores_array = np.asarray(
        all_teacher_scores,
        dtype=np.float32,
    )
    student_scores_array = np.asarray(
        all_student_scores,
        dtype=np.float32,
    )

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
    parser.add_argument("--teacher-features-type", type=str, choices=TEACHER_FEATURES_TYPE, default="keypoints")
    parser.add_argument("--student-features-type", type=str, choices=STUDENT_FEATURES_TYPE, default="keypoints")
    parser.add_argument("--participant", type=str, default=None, help="Filter to one participant ID, e.g. 1 or P01")
    parser.add_argument("--label", type=str, default=None, help="Filter to label_name, e.g. handoff or not_handoff")
    parser.add_argument("--condition", type=str, default=None, help="Filter to one condition")
    parser.add_argument("--hand", type=str, choices=["left", "right"], default=None, help="Filter to one hand")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-random-state", type=int, default=0)
    parser.add_argument("--mlp-output-path", type=str, default=None)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=None,
        help=(
            "Root directory for evaluation bundles. Defaults to "
            "MODEL_IMPL_ROOT/outputs/eval_results."
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

    HAND_INTENT_WEIGHTS_PATH = REPO_ROOT / "model_training_and_implementation" / "outputs" / "hand_intent_mlp_weights" / ("features-"+str(args.teacher_features_type)+"__crop-"+str(args.crop_around_object)) / "MLP.pth"

    checkpoint = torch.load(
    HAND_INTENT_WEIGHTS_PATH,
    map_location="cpu",
    weights_only=True,
    )

    model = HandIntentMLP(
        input_size=checkpoint["input_dim"],
        output_size=1,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
        learning_rate=args.mlp_learning_rate,
        batch_size=args.mlp_batch_size,
        epochs=args.mlp_epochs,
        random_state=args.mlp_random_state,
        device="auto",
        threshold=args.decision_threshold,
        include_rotations=args.include_quest_joint_rotations,
        student_features_type=args.student_features_type
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

    final_output_path = args.mlp_output_path
    if final_output_path is None:
        final_output_path = (
            MODEL_IMPL_ROOT
            / "outputs"
            / "quest_hand_intent_mlp_weights"
            / sanitize_path_component(variation_name)
            / f"MLP.pth"
        )
    else:
        final_output_path = Path(final_output_path)

    final_model, final_train_summary = train_quest_student_mlp(
        rows=rows,
        teacher_model=model,
        learning_rate=args.mlp_learning_rate,
        batch_size=args.mlp_batch_size,
        epochs=args.mlp_epochs,
        random_state=args.mlp_random_state,
        device="auto",
        threshold=args.decision_threshold,
        include_rotations=args.include_quest_joint_rotations,
        student_features_type=args.student_features_type,
        output_path=final_output_path,
    )

    print(f"\nFinal Quest student saved to: {final_output_path}")
    print("Final training summary:", final_train_summary)

    dataset_summary = {
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