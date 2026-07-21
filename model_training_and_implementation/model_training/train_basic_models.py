import numpy as np
from src.hand_intent_svm import LinearSVMClassifier, LinearSVMConfig
from shared.util.extract_all_samples import discover_samples, get_sample_label_name, label_to_binary
from pathlib import Path
import yaml
import argparse
import cv2
from src.dino_detector import DINOObjectDetector
from src.rtmpose_keypoints import RTMPoseKeypointDetector
from src.rtmpose_headpose import RTMPoseHeadPoseEstimator
from src.hand_intent_rf import RandomForestClassifier, RandomForestConfig


FEATURES_TYPE = ["keypoints","keypoints_headpose","keypoints_headpose_dino"]

from pathlib import Path

MODEL_IMPL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = MODEL_IMPL_ROOT / "configs" / "handoff_config.yaml"



def flatten_feature_vector(feature_vector):
    flattened = []

    def visit(value):
        if isinstance(value, np.ndarray):
            flattened.extend(value.astype(np.float32).reshape(-1).tolist())
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return

        flattened.append(float(value))

    visit(feature_vector)
    return flattened

def organize_samples(samples, features_type,obj_detector, keypoint_detector, head_pose_estimator, args):
    y_true = []
    rows = []
    skipped_unreadable = 0
    skipped_no_object = 0
    skipped_no_people = 0
    errors = 0

    for local_idx, sample in enumerate(samples):
        image_path = str(sample.rgb_path)
        image = cv2.imread(image_path)

        if image is None:
            skipped_unreadable += 1
            print(f"[{local_idx + 1}/{len(samples)}] SKIP unreadable image: {image_path}")
            rows.append(
                {
                    "index": local_idx,
                    "image": image_path,
                    "skipped": True,
                    "skip_reason": "unreadable_image",
                }
            )
            continue

        try:
            label_name = get_sample_label_name(sample)
            y = label_to_binary(label_name)
            if(features_type == "keypoints_headpose_dino"):
                objects = obj_detector.predict(image, class_names=args.dino_classes)

                # New filter: skip samples where no target object was detected.
                if len(objects) == 0 or len(objects) > 1:
                    skipped_no_object += 1
                    print(
                        f"[{local_idx + 1}/{len(samples)}] "
                        f"SKIP no object detected or too many det:{len(objects)}"
                        f"label={label_name:<12} "
                        f"{Path(image_path).name}"
                    )
                    rows.append(
                        {
                            "index": local_idx,
                            "image": image_path,
                            "label_name": label_name,
                            "label_binary": int(y),
                            "skipped": True,
                            "skip_reason": "no_object_detected",
                            "num_objects": 0,
                        }
                    )
                    continue
            if(features_type == "keypoints" or features_type == "keypoints_headpose" or features_type == "keypoints_headpose_dino"):
                people = keypoint_detector.predict(image)

                # Optional but usually useful: skip if no person/keypoints.
                # If you want to include these as automatic negatives, remove this block.
                if len(people) == 0 or len(people) > 1:
                    skipped_no_people += 1
                    print(
                        f"[{local_idx + 1}/{len(samples)}] "
                        f"SKIP no person detected "
                        f"label={label_name:<12} "
                        f"objs={len(objects)} "
                        f"{Path(image_path).name}"
                    )
                    rows.append(
                        {
                            "index": local_idx,
                            "image": image_path,
                            "label_name": label_name,
                            "label_binary": int(y),
                            "skipped": True,
                            "skip_reason": "no_person_detected",
                            "num_objects": len(objects),
                            "num_people": 0,
                        }
                    )
                    continue

            if features_type == "keypoints_headpose" or features_type == "keypoints_headpose_dino":
                head_pose = head_pose_estimator.predict_first_or_sentinel(
                    image,
                    rtmpose_results=people,
                )
            

            keypoints = np.asarray(people[0]["keypoints"][:11], dtype=np.float32)
            if(args.normalize_keypoints):
                left_shoulder = keypoints[5, :2]
                right_shoulder = keypoints[6, :2]
                shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
                keypoints[:, :2] = keypoints[:, :2] - shoulder_midpoint
                
            

            if(features_type == "keypoints"):
                row = {
                    "participant_id": sample.participant_id,
                    "index": local_idx,
                    "image": image_path,
                    "label_name": label_name,
                    "label_binary": int(y),
                    "num_people": len(people),
                    "feature_vector": flatten_feature_vector(keypoints),
                    "skipped": False,
                }
            elif(features_type == "keypoints_headpose"):
                row = {
                    "participant_id": sample.participant_id,
                    "index": local_idx,
                    "image": image_path,
                    "label_name": label_name,
                    "label_binary": int(y),
                    "num_people": len(people),
                    "feature_vector": flatten_feature_vector(keypoints)
                    + head_pose.flatten().astype(np.float32).tolist(),
                    "skipped": False,
                }
            elif features_type == "keypoints_headpose_dino":
                box_xyxy = objects[0]["box_xyxy"]
                x1, y1, x2, y2 = map(float, box_xyxy)

                object_centroid = [
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                ]

                row = {
                    "participant_id": sample.participant_id,
                    "index": local_idx,
                    "image": image_path,
                    "label_name": label_name,
                    "label_binary": int(y),
                    "num_objects": len(objects),
                    "num_people": len(people),
                    "head_pose": head_pose.tolist(),
                    "object_centroid": object_centroid,
                    "feature_vector": (
                        flatten_feature_vector(keypoints)
                        + head_pose.flatten().astype(np.float32).tolist()
                        + object_centroid
                    ),
                    "skipped": False,
                }

            rows.append(row)
            y_true.append(y)
            print("Appended new row for sample:", local_idx)
            print("Skipped unreadable:", skipped_unreadable, "skipped no object:", skipped_no_object, "skipped no people:", skipped_no_people, "errors:", errors)

        except Exception as e:
            errors += 1
            print(f"[{local_idx + 1}/{len(samples)}] ERROR {image_path}: {e}")
            rows.append(
                {
                    "index": local_idx,
                    "image": image_path,
                    "skipped": True,
                    "skip_reason": "error",
                    "error": str(e),
                }
            )
    return y_true, rows, skipped_unreadable, skipped_no_object, skipped_no_people


def train_linear_svm(rows, output_path=None, c: float = 1.0, random_state: int = 0):
    X, y, summary = rows_to_xy(rows)

    clf = LinearSVMClassifier(LinearSVMConfig(c=c, random_state=random_state))
    clf.fit(X, y)

    if output_path is not None:
        clf.save(output_path)

    return clf, summary


def train_random_forest(
    rows,
    output_path=None,
    n_estimators: int = 300,
    max_depth: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 0,
):
    X, y, summary = rows_to_xy(rows)

    clf = RandomForestClassifier(
        RandomForestConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
        )
    )
    clf.fit(X, y)

    if output_path is not None:
        clf.save(output_path)

    return clf, summary


def rows_to_xy(rows):
    X = []
    y = []

    skipped = 0
    missing_feature = 0
    missing_label = 0

    for row in rows:
        if row.get("skipped", False):
            skipped += 1
            continue

        feature_vector = row.get("feature_vector")
        label_binary = row.get("label_binary")

        if feature_vector is None:
            missing_feature += 1
            continue

        if label_binary is None:
            missing_label += 1
            continue

        X.append(np.asarray(flatten_feature_vector(feature_vector), dtype=np.float32))
        y.append(int(label_binary))

    if len(X) == 0:
        raise ValueError("No usable rows were found.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    return X, y, {
        "num_rows": int(len(rows)),
        "num_used": int(len(y)),
        "num_skipped": int(skipped),
        "missing_feature": int(missing_feature),
        "missing_label": int(missing_label),
        "X_shape": tuple(X.shape),
        "label_counts": np.bincount(y, minlength=2).tolist(),
    }


def rows_to_xy_and_groups(rows):
    X = []
    y = []
    groups = []

    skipped = 0
    missing_feature = 0
    missing_label = 0
    missing_group = 0

    for row in rows:
        if row.get("skipped", False):
            skipped += 1
            continue

        feature_vector = row.get("feature_vector")
        label_binary = row.get("label_binary")
        participant_id = row.get("participant_id")

        if feature_vector is None:
            missing_feature += 1
            continue

        if label_binary is None:
            missing_label += 1
            continue

        if participant_id is None:
            missing_group += 1
            continue

        X.append(np.asarray(flatten_feature_vector(feature_vector), dtype=np.float32))
        y.append(int(label_binary))
        groups.append(str(participant_id))

    if len(X) == 0:
        raise ValueError("No usable rows were found.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray(groups)

    return X, y, groups, {
        "num_rows": int(len(rows)),
        "num_used": int(len(y)),
        "num_skipped": int(skipped),
        "missing_feature": int(missing_feature),
        "missing_label": int(missing_label),
        "missing_group": int(missing_group),
        "X_shape": tuple(X.shape),
        "label_counts": np.bincount(y, minlength=2).tolist(),
        "groups": sorted(set(groups.tolist())),
        "group_counts": {
            g: int(np.sum(groups == g)) for g in sorted(set(groups.tolist()))
        },
    }


def split_rows_by_participant(rows, test_size: float = 0.25, random_state: int = 0):
    X, y, groups, summary = rows_to_xy_and_groups(rows)

    unique_groups = np.array(sorted(set(groups.tolist())))
    if len(unique_groups) < 2:
        raise ValueError("Need at least two participants to do participant-based evaluation.")

    rng = np.random.default_rng(random_state)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    if len(unique_groups) == 2:
        test_groups = np.asarray([shuffled_groups[0]])
    else:
        num_test_groups = max(1, int(round(len(unique_groups) * test_size)))
        num_test_groups = min(num_test_groups, len(unique_groups) - 1)
        test_groups = shuffled_groups[:num_test_groups]

    test_mask = np.isin(groups, test_groups)
    train_mask = ~test_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Participant split produced an empty train or test set.")

    split_summary = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "train_size": int(len(train_idx)),
        "test_size_count": int(len(test_idx)),
        "train_label_counts": np.bincount(y[train_idx], minlength=2).tolist(),
        "test_label_counts": np.bincount(y[test_idx], minlength=2).tolist(),
        "train_participants": sorted(set(groups[train_idx].tolist())),
        "test_participants": sorted(set(groups[test_idx].tolist())),
    }

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    return train_rows, test_rows, summary, split_summary


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


def evaluate_model_on_rows(
    model,
    train_rows,
    test_rows,
    fit_model: bool = True,
    model_name: str = "model",
):
    """Evaluate any binary classifier on participant-separated rows.

    The model only needs `fit` and `predict`. If it also exposes
    `decision_function` or `predict_proba`, those scores are included in the
    return value for downstream analysis.
    """

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

    if fit_model:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_binary_metrics(y_test, y_pred)

    scores = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] > 1:
            scores = proba[:, 1]

    evaluation = {
        "model_name": model_name,
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
        "scores": None if scores is None else np.asarray(scores).tolist(),
    }

    return evaluation


def evaluate_linear_svm(model, train_rows, test_rows, fit_model: bool = False):
    return evaluate_model_on_rows(
        model=model,
        train_rows=train_rows,
        test_rows=test_rows,
        fit_model=fit_model,
        model_name="linear_svm",
    )


def evaluate_random_forest(model, train_rows, test_rows, fit_model: bool = False):
    return evaluate_model_on_rows(
        model=model,
        train_rows=train_rows,
        test_rows=test_rows,
        fit_model=fit_model,
        model_name="random_forest",
    )


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    dataset_root = REPO_ROOT / config["dataset"]["root"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-classes", default="cup")
    parser.add_argument("--normalize-keypoints", type=bool, choices=[True, False], default=True)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--features-type", type=str, choices=FEATURES_TYPE, default="keypoints")
    parser.add_argument("--participant", type=str, default=None, help="Filter to one participant ID, e.g. 1 or P01")
    parser.add_argument("--label", type=str, default=None, help="Filter to label_name, e.g. handoff or not_handoff")
    parser.add_argument("--condition", type=str, default=None, help="Filter to one condition")
    parser.add_argument("--hand", type=str, choices=["left", "right"], default=None, help="Filter to one hand")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()
    samples = discover_samples(
        dataset_root=dataset_root,
        participant_filter=args.participant,
        label_filter=args.label,
        condition_filter=args.condition,
        hand_filter=args.hand,
    )
    samples = samples[args.start_index:]


    object_detector = DINOObjectDetector(
        model_id="IDEA-Research/grounding-dino-base",
        confidence=args.confidence,
    )

    keypoint_detector = RTMPoseKeypointDetector(
        confidence=args.confidence,
        device="cuda",
    )

    head_pose_estimator = RTMPoseHeadPoseEstimator(
        weights_path="kwan_pretrained_weights/head-pose-pretrained.pkl",
        gpu_id=0,
    )

    y_true, rows, skipped_unreadable, skipped_no_object, skipped_no_people = organize_samples(
        samples=samples,
        features_type=args.features_type,
        obj_detector=object_detector,
        keypoint_detector=keypoint_detector,
        head_pose_estimator=head_pose_estimator,
        args=args,
    )

    train_rows, test_rows, split_summary, participant_split_summary = split_rows_by_participant(
        rows,
        test_size=0.5,
        random_state=0,
    )

    svm_clf, svm_summary = train_linear_svm(train_rows)
    svm_evaluation = evaluate_linear_svm(
        svm_clf,
        train_rows,
        test_rows,
        fit_model=False,
    )

    rf_clf, rf_summary = train_random_forest(train_rows)
    rf_evaluation = evaluate_random_forest(
        rf_clf,
        train_rows,
        test_rows,
        fit_model=False,
    )

    print("Linear SVM training summary:", svm_summary)
    print("Participant split summary:", participant_split_summary)
    print("Linear SVM evaluation metrics:", svm_evaluation["metrics"])
    print("Random forest training summary:", rf_summary)
    print("Random forest evaluation metrics:", rf_evaluation["metrics"])
    return {
        "linear_svm": {
            "model": svm_clf,
            "summary": svm_summary,
            "evaluation": svm_evaluation,
        },
        "random_forest": {
            "model": rf_clf,
            "summary": rf_summary,
            "evaluation": rf_evaluation,
        },
        "participant_split_summary": participant_split_summary,
    }

    

if __name__ == "__main__":
    main()



