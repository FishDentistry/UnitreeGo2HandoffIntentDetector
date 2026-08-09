import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import yaml
from shared.util.extract_all_samples import discover_samples, organize_samples, rows_to_xy
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from ..src.dino_detector import DINOObjectDetector
from ..src.rtmpose_keypoints import RTMPoseKeypointDetector
from ..src.rtmpose_headpose import RTMPoseHeadPoseEstimator
from ..src.resnet_encoder import ResNet18ImageEncoder
from ..src.dino_encoder import DINOv2ImageEncoder


FEATURES_TYPE = ["keypoints","keypoints_headpose","keypoints_headpose_dino","keypoints_resnet","keypoints_headpose_resnet"]
MODEL_TYPES = ["logistic_regression", "linear_svm", "rbf_svm", "xgboost"]

MODEL_IMPL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = MODEL_IMPL_ROOT / "configs" / "handoff_config.yaml"


def make_json_serializable(value):
    """Convert common scientific-Python values into JSON-safe objects."""
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }
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
    """Name a variation using only feature type and crop enablement."""
    parts = [
        f"features-{args.features_type}",
        f"crop-{args.crop_around_object}",
    ]
    return "__".join(sanitize_path_component(part) for part in parts)


def save_evaluation_results(
    *,
    model_type,
    lopo_evaluation,
    final_train_summary,
    args,
    run_id,
    variation_name,
    run_started_at,
    final_output_path,
    dataset_root,
    dataset_summary,
):
    """Save evaluation files for one model type and variation."""
    eval_root = (
        Path(args.eval_output_dir)
        if args.eval_output_dir is not None
        else MODEL_IMPL_ROOT / "outputs" / "eval_results" / "classical_models"
    )
    variation_dir = (
        eval_root
        / sanitize_path_component(variation_name)
        / sanitize_path_component(model_type)
    )
    variation_dir.mkdir(parents=True, exist_ok=True)

    completed_at = datetime.now().astimezone()
    result_payload = {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "variation_name": variation_name,
            "model_type": model_type,
            "started_at": run_started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        },
        "configuration": vars(args),
        "paths": {
            "dataset_root": str(dataset_root),
            "model_weights": str(final_output_path),
            "evaluation_directory": str(variation_dir),
        },
        "dataset_summary": dataset_summary,
        "lopo_evaluation": lopo_evaluation,
        "final_training_summary": final_train_summary,
    }

    full_json_path = variation_dir / "evaluation_results.json"
    with full_json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            make_json_serializable(result_payload),
            file_obj,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file_obj.write("\n")

    pooled = lopo_evaluation["pooled_metrics"]
    summary_row = {
        "run_id": run_id,
        "variation_name": variation_name,
        "model_type": model_type,
        "features_type": args.features_type,
        "crop_around_object": bool(args.crop_around_object),
        "decision_threshold": float(args.decision_threshold),
        "num_participants": lopo_evaluation["num_participants"],
        "num_folds": lopo_evaluation["num_folds"],
        "pooled_total": pooled.get("total"),
        "pooled_accuracy": pooled.get("accuracy"),
        "pooled_balanced_accuracy": pooled.get("balanced_accuracy"),
        "pooled_precision": pooled.get("precision"),
        "pooled_recall": pooled.get("recall"),
        "pooled_specificity": pooled.get("specificity"),
        "pooled_f1": pooled.get("f1"),
        "pooled_tn": pooled.get("tn"),
        "pooled_fp": pooled.get("fp"),
        "pooled_fn": pooled.get("fn"),
        "pooled_tp": pooled.get("tp"),
        "final_train_num_samples": final_train_summary.get("num_samples"),
        "final_train_input_dim": final_train_summary.get("input_dim"),
        "final_train_seconds": final_train_summary.get("train_seconds"),
        "model_weights": str(final_output_path),
    }

    summary_csv_path = variation_dir / "summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    fold_rows = []
    for fold in lopo_evaluation["folds"]:
        metrics = fold["metrics"]
        split = fold["split"]
        fold_rows.append(
            {
                "run_id": run_id,
                "variation_name": variation_name,
                "model_type": model_type,
                "fold_index": fold["fold_index"],
                "held_out_participant": fold["held_out_participant"],
                "train_size": split.get("train_size"),
                "test_size": split.get("test_size_count"),
                "train_label_counts": json.dumps(
                    make_json_serializable(split.get("train_label_counts")),
                    sort_keys=True,
                ),
                "test_label_counts": json.dumps(
                    make_json_serializable(split.get("test_label_counts")),
                    sort_keys=True,
                ),
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "specificity": metrics.get("specificity"),
                "f1": metrics.get("f1"),
                "tn": metrics.get("tn"),
                "fp": metrics.get("fp"),
                "fn": metrics.get("fn"),
                "tp": metrics.get("tp"),
            }
        )

    folds_csv_path = variation_dir / "fold_metrics.csv"
    with folds_csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    return {
        "evaluation_dir": str(variation_dir),
        "evaluation_json": str(full_json_path),
        "summary_csv": str(summary_csv_path),
        "fold_metrics_csv": str(folds_csv_path),
        "summary_row": summary_row,
    }


def build_classifier(model_type: str, random_state: int = 0):
    """Create one of the four classical hand-intent classifiers."""
    if model_type == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )

    if model_type == "linear_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="linear",
                        probability=True,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )

    if model_type == "rbf_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )

    if model_type == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(random_state),
            n_jobs=-1,
        )

    raise ValueError(
        f"Unsupported model_type={model_type}. "
        f"Expected one of: {MODEL_TYPES}"
    )


def train_classifier(
    rows,
    model_type: str,
    random_state: int = 0,
    output_path=None,
):
    X, y, summary = rows_to_xy(rows)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")

    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes present in the training split.")

    model = build_classifier(
        model_type=model_type,
        random_state=random_state,
    )

    train_start = time.perf_counter()
    model.fit(X, y)
    train_seconds = float(time.perf_counter() - train_start)

    if output_path is not None:
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": model_type,
                "input_dim": int(X.shape[1]),
                "random_state": int(random_state),
                "model": model,
            },
            path_obj,
        )

    train_summary = {
        **summary,
        "model_type": model_type,
        "num_samples": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "random_state": int(random_state),
        "train_seconds": train_seconds,
    }

    return model, train_summary


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


def get_positive_class_scores(model, X):
    """Return P(class=1) for classifiers that expose predict_proba."""
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    positive_matches = np.where(classes == 1)[0]

    if len(positive_matches) != 1:
        raise ValueError(
            f"Expected exactly one positive class labeled 1, got classes={classes.tolist()}"
        )

    return np.asarray(
        probabilities[:, int(positive_matches[0])],
        dtype=np.float64,
    )


def evaluate_classifier(
    model,
    model_type: str,
    train_rows,
    test_rows,
    threshold: float = 0.5,
):
    def participant_ids(rows):
        ids = []
        for row in rows:
            if row.get("skipped", False):
                continue
            participant_id = row.get("participant_id")
            if participant_id is None:
                continue
            ids.append(str(participant_id))
        return sorted(set(ids))

    X_train, y_train, train_summary = rows_to_xy(train_rows)
    X_test, y_test, test_summary = rows_to_xy(test_rows)

    if len(np.unique(y_train)) < 2:
        raise ValueError("Need both classes present in the training split.")

    scores = get_positive_class_scores(model, X_test)
    y_pred = (scores >= float(threshold)).astype(np.int32)
    metrics = compute_binary_metrics(y_test, y_pred)

    evaluation = {
        "model_name": model_type,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "split": {
            "train_size": int(len(train_rows)),
            "test_size_count": int(len(test_rows)),
            "train_label_counts": train_summary["label_counts"],
            "test_label_counts": test_summary["label_counts"],
            "train_participants": participant_ids(train_rows),
            "test_participants": participant_ids(test_rows),
        },
        "metrics": metrics,
        "y_true": y_test.tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "scores": np.asarray(scores).tolist(),
    }

    return evaluation


def leave_one_participant_out_evaluation(
    rows,
    model_type: str,
    random_state: int = 0,
    threshold: float = 0.5,
):
    """
    Perform leave-one-participant-out evaluation.

    For each fold:
        - Hold out one participant for testing.
        - Train a new classifier from scratch on all other participants.
        - Evaluate on the held-out participant.

    Returns per-participant results and pooled metrics.
    """

    usable_rows = []

    for row in rows:
        if row.get("skipped", False):
            continue

        if row.get("feature_vector") is None:
            continue

        if row.get("label_binary") is None:
            continue

        if row.get("participant_id") is None:
            continue

        usable_rows.append(row)

    participants = sorted(
        {
            str(row["participant_id"])
            for row in usable_rows
        }
    )

    if len(participants) < 2:
        raise ValueError(
            "Leave-one-participant-out evaluation requires at least "
            "two participants."
        )

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_scores = []

    for fold_index, held_out_participant in enumerate(participants):
        print(
            f"\n{model_type}: LOPO fold {fold_index + 1}/{len(participants)}: "
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

        if len(train_rows) == 0:
            raise ValueError(
                f"No training rows for fold {held_out_participant}."
            )

        if len(test_rows) == 0:
            raise ValueError(
                f"No test rows for fold {held_out_participant}."
            )

        # Important: create and train a completely new model for every fold.
        fold_model, train_summary = train_classifier(
            train_rows,
            model_type=model_type,
            random_state=random_state + fold_index,
            output_path=None,
        )

        fold_evaluation = evaluate_classifier(
            model=fold_model,
            model_type=model_type,
            train_rows=train_rows,
            test_rows=test_rows,
            threshold=threshold,
        )

        fold_evaluation["fold_index"] = fold_index
        fold_evaluation["held_out_participant"] = held_out_participant
        fold_evaluation["train_seconds"] = train_summary["train_seconds"]

        fold_results.append(fold_evaluation)

        all_y_true.extend(fold_evaluation["y_true"])
        all_y_pred.extend(fold_evaluation["y_pred"])
        all_scores.extend(fold_evaluation["scores"])

        metrics = fold_evaluation["metrics"]

        print(
            f"Participant {held_out_participant}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"recall={metrics['recall']}"
        )

    pooled_metrics = compute_binary_metrics(
        y_true=all_y_true,
        y_pred=all_y_pred,
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
            fold["metrics"][metric_name]
            for fold in fold_results
            if fold["metrics"][metric_name] is not None
        ]

        participant_metric_summary[metric_name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "minimum": float(np.min(values)) if values else None,
            "maximum": float(np.max(values)) if values else None,
        }

    return {
        "evaluation_type": "leave_one_participant_out",
        "model_type": model_type,
        "num_participants": len(participants),
        "participants": participants,
        "num_folds": len(fold_results),
        "folds": fold_results,

        # Metrics after pooling predictions from every held-out person.
        "pooled_metrics": pooled_metrics,

        # Every participant receives equal weight in this summary.
        "participant_metric_summary": participant_metric_summary,

        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "scores": all_scores,
    }


def save_comparison_summary(
    *,
    args,
    variation_name,
    summary_rows,
):
    """Save one compact CSV comparing all four model types."""
    eval_root = (
        Path(args.eval_output_dir)
        if args.eval_output_dir is not None
        else MODEL_IMPL_ROOT / "outputs" / "eval_results"
    )
    variation_dir = eval_root / sanitize_path_component(variation_name)
    variation_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = variation_dir / "model_comparison_summary.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return comparison_path


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    dataset_root = REPO_ROOT / config["dataset"]["root"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-classes",default="cup",)
    parser.add_argument("--crop-around-object",action=argparse.BooleanOptionalAction,default=False,)
    parser.add_argument("--crop-object",default="person.",)
    parser.add_argument("--normalize-keypoints",action=argparse.BooleanOptionalAction,default=True,)
    parser.add_argument("--confidence",type=float,default=0.15,)
    parser.add_argument("--features-type",type=str,choices=FEATURES_TYPE,default="keypoints",)
    parser.add_argument("--img-encoder",type=str,choices=["dino", "resnet"],default="resnet",)
    parser.add_argument("--participant",type=str,default=None,help="Filter to one participant ID, e.g. 1 or P01",)
    parser.add_argument("--label",type=str,default=None,help="Filter to label_name, e.g. handoff or not_handoff",)
    parser.add_argument("--condition",type=str,default=None,help="Filter to one condition",)
    parser.add_argument("--hand",type=str,choices=["left", "right"],default=None,help="Filter to one hand",)
    parser.add_argument("--start-index",type=int,default=0,)
    parser.add_argument("--max-samples",type=int,default=None,)
    parser.add_argument("--model-random-state",type=int,default=0,)
    parser.add_argument("--decision-threshold",type=float,default=0.5,)
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default=None,
        help=(
            "Root directory for final classical model files. Defaults to "
            "MODEL_IMPL_ROOT/outputs/hand_intent_classical_models."
        ),
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=None,
        help=(
            "Root directory for evaluation files. Defaults to "
            "MODEL_IMPL_ROOT/outputs/eval_results."
        ),
    )

    args = parser.parse_args()

    run_started_at = datetime.now().astimezone()
    run_id = run_started_at.strftime("%Y%m%dT%H%M%S_%f%z")
    variation_name = build_model_variation_name(args)
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
        weights_path= REPO_ROOT / "model_training_and_implementation" / "kwan_pretrained_weights" / "head-pose-pretrained.pkl",
        gpu_id=0,
    )

    if(args.img_encoder == "dino"):
        img_encoder = DINOv2ImageEncoder()
    elif(args.img_encoder == "resnet"):
        img_encoder = ResNet18ImageEncoder(pretrained=True, device="cuda", l2_normalize=True)
    else:
        raise ValueError(f"Unsupported encoder type: {args.img_encoder}")


    y_true, rows, skipped_unreadable, skipped_no_object, skipped_no_people = organize_samples(
        samples=samples,
        features_type=args.features_type,
        obj_detector=object_detector,
        keypoint_detector=keypoint_detector,
        head_pose_estimator=head_pose_estimator,
        img_encoder=img_encoder,
        args=args,
    )

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

    if args.model_output_dir is None:
        model_output_root = (
            MODEL_IMPL_ROOT
            / "outputs"
            / "hand_intent_classical_models"
        )
    else:
        model_output_root = Path(args.model_output_dir)

    model_output_dir = (
        model_output_root
        / sanitize_path_component(variation_name)
    )

    all_results = {}
    comparison_summary_rows = []

    for model_type in MODEL_TYPES:
        print("\n" + "=" * 80)
        print(f"Training/evaluating model type: {model_type}")
        print("=" * 80)

        lopo_evaluation = leave_one_participant_out_evaluation(
            rows=rows,
            model_type=model_type,
            random_state=args.model_random_state,
            threshold=args.decision_threshold,
        )

        print(f"\n{model_type}: leave-one-participant-out evaluation complete")

        print(
            "Pooled metrics:",
            lopo_evaluation["pooled_metrics"],
        )

        print(
            "Participant-level metric summary:",
            lopo_evaluation["participant_metric_summary"],
        )

        for fold in lopo_evaluation["folds"]:
            participant = fold["held_out_participant"]
            metrics = fold["metrics"]

            print(
                f"Participant {participant}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"balanced_accuracy={metrics['balanced_accuracy']}, "
                f"precision={metrics['precision']}, "
                f"recall={metrics['recall']}, "
                f"f1={metrics['f1']:.4f}"
            )

        # Train one final deployment model using all available participants.
        output_path = model_output_dir / f"{model_type}.joblib"

        final_model, final_train_summary = train_classifier(
            rows=rows,
            model_type=model_type,
            random_state=args.model_random_state,
            output_path=output_path,
        )

        print(f"\nFinal {model_type} model saved to: {output_path}")
        print("Final training summary:", final_train_summary)

        saved_eval_paths = save_evaluation_results(
            model_type=model_type,
            lopo_evaluation=lopo_evaluation,
            final_train_summary=final_train_summary,
            args=args,
            run_id=run_id,
            variation_name=variation_name,
            run_started_at=run_started_at,
            final_output_path=output_path,
            dataset_root=dataset_root,
            dataset_summary=dataset_summary,
        )

        print(f"Evaluation results saved to: {saved_eval_paths['evaluation_dir']}")
        print(f"  Full JSON: {saved_eval_paths['evaluation_json']}")
        print(f"  Summary CSV: {saved_eval_paths['summary_csv']}")
        print(f"  Fold metrics CSV: {saved_eval_paths['fold_metrics_csv']}")

        comparison_summary_rows.append(saved_eval_paths["summary_row"])

        all_results[model_type] = {
            "lopo_evaluation": lopo_evaluation,
            "final_train_summary": final_train_summary,
            "final_output_path": str(output_path),
            "saved_eval_paths": saved_eval_paths,
        }

    comparison_summary_path = save_comparison_summary(
        args=args,
        variation_name=variation_name,
        summary_rows=comparison_summary_rows,
    )

    print(f"\nCombined model comparison saved to: {comparison_summary_path}")

    return {
        "models": all_results,
        "comparison_summary_csv": str(comparison_summary_path),
        "run_id": run_id,
        "variation_name": variation_name,
    }


if __name__ == "__main__":
    main()

