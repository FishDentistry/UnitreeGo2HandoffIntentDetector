# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional, Any 
# import json
# import numpy as np
# import cv2


# REQUIRED_FILES = ("rgb.png", "joints_raw.json", "meta.json")

# @dataclass(frozen=True)
# class Sample:
#     sample_dir: Path
#     rgb_path: Path
#     joints_path: Path
#     meta_path: Path
#     participant_id: str
#     label_name: str
#     condition: str
#     hand: str
#     sample_type: str
#     sample_id: str


# def get_sample_label_name(sample):
#     """
#     Tries to robustly extract the ground-truth label from a Sample.
#     Adjust this if your Sample dataclass uses a different field.
#     """

#     for attr in ["label_name", "label", "class_name", "target", "y"]:
#         if hasattr(sample, attr):
#             value = getattr(sample, attr)
#             if value is not None:
#                 return str(value)

#     # Fallback: infer from path.
#     # Check not_handoff first because it contains the substring "handoff".
#     path = str(sample.rgb_path).lower()

#     if "not_handoff" in path or "non_handoff" in path or "no_handoff" in path:
#         return "not_handoff"

#     if "handoff" in path:
#         return "handoff"

#     raise ValueError(f"Could not determine label for sample: {sample}")

# def label_to_binary(label_name):
#     """
#     Returns 1 for handoff, 0 for not handoff.
#     """

#     label = str(label_name).lower()

#     if label in ["handoff", "positive", "pos", "1", "true"]:
#         return 1

#     if label in ["not_handoff", "non_handoff", "no_handoff", "negative", "neg", "0", "false"]:
#         return 0

#     # Path-style labels sometimes contain these substrings.
#     if "not_handoff" in label or "non_handoff" in label or "no_handoff" in label:
#         return 0

#     if "handoff" in label:
#         return 1

#     raise ValueError(f"Unknown label name: {label_name}")


# def read_json(path: Path) -> Dict[str, Any]:
#     try:
#         with path.open("r", encoding="utf-8") as f:
#             data = json.load(f)
#         return data if isinstance(data, dict) else {}
#     except Exception as exc:
#         print(f"[warning] Could not read JSON {path}: {exc}")
#         return {}

# def sample_root_from_dataset_root(dataset_root: Path) -> Path:
#     dataset_root = dataset_root.expanduser().resolve()
#     if (dataset_root / "samples").is_dir():
#         return dataset_root / "samples"
#     return dataset_root


# def infer_metadata_from_path(sample_dir: Path, sample_root: Path) -> Dict[str, str]:
#     try:
#         rel = sample_dir.relative_to(sample_root).parts
#     except ValueError:
#         rel = sample_dir.parts

#     # Expected: participant / label_name / condition / hand / sample_id
#     return {
#         "participant_id": rel[0] if len(rel) > 0 else "unknown_participant",
#         "label_name": rel[1] if len(rel) > 1 else "unknown_label",
#         "condition": rel[2] if len(rel) > 2 else "unknown_condition",
#         "hand": rel[3] if len(rel) > 3 else "unknown_hand",
#         "sample_id": rel[4] if len(rel) > 4 else sample_dir.name,
#         "sample_type": "unknown_sample_type",
#     }


# def discover_samples(
#     dataset_root: Path,
#     participant_filter: Optional[str],
#     label_filter: Optional[str],
#     condition_filter: Optional[str],
#     hand_filter: Optional[str],
# ) -> List[Sample]:
#     sample_root = sample_root_from_dataset_root(dataset_root)
#     if not sample_root.is_dir():
#         raise FileNotFoundError(f"Sample root does not exist: {sample_root}")

#     samples: List[Sample] = []
#     for rgb_path in sample_root.rglob("rgb.png"):
#         sample_dir = rgb_path.parent
#         if not all((sample_dir / filename).is_file() for filename in REQUIRED_FILES):
#             continue

#         inferred = infer_metadata_from_path(sample_dir, sample_root)
#         meta = read_json(sample_dir / "meta.json")

#         participant_id = str(meta.get("participant_id") or inferred["participant_id"])
#         label_name = str(meta.get("label_name") or inferred["label_name"])
#         condition = str(meta.get("condition") or inferred["condition"])
#         hand = str(meta.get("hand") or inferred["hand"])
#         sample_type = str(meta.get("sample_type") or inferred["sample_type"])
#         sample_id = str(meta.get("sample_id") or inferred["sample_id"])

#         if participant_filter and participant_id != participant_filter:
#             continue
#         if label_filter and label_name != label_filter:
#             continue
#         if condition_filter and condition != condition_filter:
#             continue
#         if hand_filter and hand != hand_filter:
#             continue

#         samples.append(
#             Sample(
#                 sample_dir=sample_dir,
#                 rgb_path=rgb_path,
#                 joints_path=sample_dir / "joints_raw.json",
#                 meta_path=sample_dir / "meta.json",
#                 participant_id=participant_id,
#                 label_name=label_name,
#                 condition=condition,
#                 hand=hand,
#                 sample_type=sample_type,
#                 sample_id=sample_id,
#             )
#         )

#     samples.sort(key=lambda s: (s.participant_id, s.label_name, s.condition, s.hand, s.sample_id, str(s.sample_dir)))
#     return samples


# def flatten_feature_vector(feature_vector):
#     flattened = []

#     def visit(value):
#         if isinstance(value, np.ndarray):
#             flattened.extend(value.astype(np.float32).reshape(-1).tolist())
#             return

#         if isinstance(value, (list, tuple)):
#             for item in value:
#                 visit(item)
#             return

#         flattened.append(float(value))

#     visit(feature_vector)
#     return flattened

# def organize_samples(samples, features_type,obj_detector, keypoint_detector, head_pose_estimator, img_encoder, args):
#     y_true = []
#     rows = []
#     skipped_unreadable = 0
#     skipped_no_object = 0
#     skipped_no_people = 0
#     errors = 0



#     for local_idx, sample in enumerate(samples):
#         image_path = str(sample.rgb_path)
#         image = cv2.imread(image_path)
#         objects = []
#         people = []

        

#         if image is None:
#             skipped_unreadable += 1
#             print(f"[{local_idx + 1}/{len(samples)}] SKIP unreadable image: {image_path}")
#             rows.append(
#                 {
#                     "index": local_idx,
#                     "image": image_path,
#                     "skipped": True,
#                     "skip_reason": "unreadable_image",
#                 }
#             )
#             continue

#         try:
#             if(args.crop_around_object):
#                 crop_objs = obj_detector.predict(image, class_names=args.crop_object)
#                 if len(crop_objs) == 0:
#                     print("SKIP no person detected ")
#                     continue

#                 sorted_boxes = sorted(
#                                         [d for d in crop_objs if d["label"].strip(".") == args.crop_object.strip(".")],
#                                         key=lambda d: (
#                                             (d["box_xyxy"][2] - d["box_xyxy"][0]) *
#                                             (d["box_xyxy"][3] - d["box_xyxy"][1])
#                                         ),
#                                         reverse=True,  # largest first
#                                     )
#                 box_xyxy = sorted_boxes[0]["box_xyxy"]
#                 x1, y1, x2, y2 = map(int, box_xyxy)
#                 cropped_image = image[y1:y2, x1:x2]
#                 image = cropped_image

#             det_hands = obj_detector.predict(image, class_names="hand.")
#             if len(det_hands) == 0:
#                             print("SKIP no hands detected ")
#                             skipped_no_people += 1
#                             continue
            
#             sorted_hands = sorted(
#                                     [d for d in det_hands if d["label"].strip(".") == "hand"],
#                                     key=lambda d: (
#                                         (d["box_xyxy"][2] - d["box_xyxy"][0]) *
#                                         (d["box_xyxy"][3] - d["box_xyxy"][1])
#                                     ),
#                                     reverse=True,  # largest first
#                                 )
#             biggest_hand_bbox = sorted_hands[0]["box_xyxy"]
#             x1, y1, x2, y2 = map(int, biggest_hand_bbox)
#             outstretched_hand_mp = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            
                

#             label_name = get_sample_label_name(sample)
#             y = label_to_binary(label_name)
#             if(features_type == "keypoints_headpose_dino"):
#                 objects = obj_detector.predict(image, class_names=args.dino_classes)

#                 # New filter: skip samples where no target object was detected.
#                 if len(objects) == 0 or len(objects) > 1:
#                     skipped_no_object += 1
#                     print(
#                         f"[{local_idx + 1}/{len(samples)}] "
#                         f"SKIP no object detected or too many det:{len(objects)}"
#                         f"label={label_name:<12} "
#                         f"{Path(image_path).name}"
#                     )
#                     rows.append(
#                         {
#                             "index": local_idx,
#                             "image": image_path,
#                             "label_name": label_name,
#                             "label_binary": int(y),
#                             "skipped": True,
#                             "skip_reason": "no_object_detected",
#                             "num_objects": 0,
#                         }
#                     )
#                     continue
#             if (
#                 features_type == "keypoints"
#                 or features_type == "keypoints_headpose"
#                 or features_type == "keypoints_headpose_dino"
#                 or features_type == "keypoints_resnet"
#                 or features_type == "keypoints_headpose_resnet"
#             ):
#                 people = keypoint_detector.predict(image)

#                 # Optional but usually useful: skip if no person/keypoints.
#                 # If you want to include these as automatic negatives, remove this block.
#                 if len(people) == 0 or len(people) > 1:
#                     skipped_no_people += 1
#                     print(
#                         f"[{local_idx + 1}/{len(samples)}] "
#                         f"SKIP no person detected "
#                         f"label={label_name:<12} "
#                         f"objs={len(objects)} "
#                         f"{Path(image_path).name}"
#                     )
#                     rows.append(
#                         {
#                             "index": local_idx,
#                             "image": image_path,
#                             "label_name": label_name,
#                             "label_binary": int(y),
#                             "skipped": True,
#                             "skip_reason": "no_person_detected",
#                             "num_objects": len(objects),
#                             "num_people": 0,
#                         }
#                     )
#                     continue

#             if (
#                 features_type == "keypoints_headpose"
#                 or features_type == "keypoints_headpose_dino"
#                 or features_type == "keypoints_headpose_resnet"
#             ):
#                 head_pose = head_pose_estimator.predict_first_or_sentinel(
#                     image,
#                     rtmpose_results=people,
#                 )
            

#             keypoints = np.asarray(people[0]["keypoints"][:11], dtype=np.float32)
#             shoulder_midpoint = [0.0, 0.0]
#             if(args.normalize_keypoints):
#                 left_shoulder = keypoints[5, :2]
#                 right_shoulder = keypoints[6, :2]
#                 shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
#                 keypoints[:, :2] = keypoints[:, :2] - shoulder_midpoint


                
            

#             if(features_type == "keypoints"):
#                 row = {
#                     "participant_id": sample.participant_id,
#                     "index": local_idx,
#                     "image": image_path,
#                     "label_name": label_name,
#                     "label_binary": int(y),
#                     "num_people": len(people),
#                     "feature_vector": flatten_feature_vector(keypoints),
#                     "skipped": False,
#                 }
#             elif(features_type == "keypoints_headpose"):
#                 row = {
#                     "participant_id": sample.participant_id,
#                     "index": local_idx,
#                     "image": image_path,
#                     "label_name": label_name,
#                     "label_binary": int(y),
#                     "num_people": len(people),
#                     "feature_vector": flatten_feature_vector(keypoints)
#                     + head_pose.flatten().astype(np.float32).tolist(),
#                     "skipped": False,
#                 }
#             elif features_type == "keypoints_headpose_dino":
#                 box_xyxy = objects[0]["box_xyxy"]
#                 x1, y1, x2, y2 = map(float, box_xyxy)

#                 object_centroid = [
#                     (x1 + x2) / 2.0,
#                     (y1 + y2) / 2.0,
#                 ]

#                 row = {
#                     "participant_id": sample.participant_id,
#                     "index": local_idx,
#                     "image": image_path,
#                     "label_name": label_name,
#                     "label_binary": int(y),
#                     "num_objects": len(objects),
#                     "num_people": len(people),
#                     "head_pose": head_pose.tolist(),
#                     "object_centroid": object_centroid,
#                     "feature_vector": (
#                         flatten_feature_vector(keypoints)
#                         + head_pose.flatten().astype(np.float32).tolist()
#                         + object_centroid
#                     ),
#                     "skipped": False,
#                 }
#             elif features_type == "keypoints_resnet":
#                 resnet_embedding = img_encoder.predict(image)["embedding"]
#                 row = {
#                     "participant_id": sample.participant_id,
#                     "index": local_idx,
#                     "image": image_path,
#                     "label_name": label_name,
#                     "label_binary": int(y),
#                     "num_people": len(people),
#                     "feature_vector": (
#                         flatten_feature_vector(keypoints)
#                         + resnet_embedding.flatten().astype(np.float32).tolist()
#                     ),
#                     "skipped": False,
#                 }
#             elif features_type == "keypoints_headpose_resnet":
#                 resnet_embedding = img_encoder.predict(image)["embedding"]
#                 row = {
#                     "participant_id": sample.participant_id,
#                     "index": local_idx,
#                     "image": image_path,
#                     "label_name": label_name,
#                     "label_binary": int(y),
#                     "num_people": len(people),
#                     "feature_vector": (
#                         flatten_feature_vector(keypoints)
#                         + head_pose.flatten().astype(np.float32).tolist()
#                         + resnet_embedding.flatten().astype(np.float32).tolist()
#                     ),
#                     "skipped": False,
#                 }
#             if(sample.joints_path is not None):
#                 row["quest_joints_pth"] = sample.joints_path

            
#             hand_to_shoulder_vector = (np.asarray(outstretched_hand_mp) - np.asarray(shoulder_midpoint)).tolist()
#             row["feature_vector"] = row["feature_vector"] + hand_to_shoulder_vector
            
#             rows.append(row)
#             y_true.append(y)
#             print("Appended new row for sample:", local_idx)
#             print("Skipped unreadable:", skipped_unreadable, "skipped no object:", skipped_no_object, "skipped no people:", skipped_no_people, "errors:", errors)

#         except Exception as e:
#             errors += 1
#             print(f"[{local_idx + 1}/{len(samples)}] ERROR {image_path}: {e}")
#             rows.append(
#                 {
#                     "index": local_idx,
#                     "image": image_path,
#                     "skipped": True,
#                     "skip_reason": "error",
#                     "error": str(e),
#                 }
#             )
#     return y_true, rows, skipped_unreadable, skipped_no_object, skipped_no_people





# def rows_to_xy(rows,feature_key="feature_vector", label_key="label_binary"):
#     X = []
#     y = []

#     skipped = 0
#     missing_feature = 0
#     missing_label = 0

#     for row in rows:
#         if row.get("skipped", False):
#             skipped += 1
#             continue

#         feature_vector = row.get(feature_key)
#         label_binary = row.get(label_key)

#         if feature_vector is None:
#             missing_feature += 1
#             continue

#         if label_binary is None:
#             missing_label += 1
#             continue

#         X.append(np.asarray(flatten_feature_vector(feature_vector), dtype=np.float32))
#         y.append(int(label_binary))

#     if len(X) == 0:
#         raise ValueError("No usable rows were found.")

#     X = np.asarray(X, dtype=np.float32)
#     y = np.asarray(y, dtype=np.int32)

#     return X, y, {
#         "num_rows": int(len(rows)),
#         "num_used": int(len(y)),
#         "num_skipped": int(skipped),
#         "missing_feature": int(missing_feature),
#         "missing_label": int(missing_label),
#         "X_shape": tuple(X.shape),
#         "label_counts": np.bincount(y, minlength=2).tolist(),
#     }


# def rows_to_xy_and_groups(rows):
#     X = []
#     y = []
#     groups = []

#     skipped = 0
#     missing_feature = 0
#     missing_label = 0
#     missing_group = 0

#     for row in rows:
#         if row.get("skipped", False):
#             skipped += 1
#             continue

#         feature_vector = row.get("feature_vector")
#         label_binary = row.get("label_binary")
#         participant_id = row.get("participant_id")

#         if feature_vector is None:
#             missing_feature += 1
#             continue

#         if label_binary is None:
#             missing_label += 1
#             continue

#         if participant_id is None:
#             missing_group += 1
#             continue

#         X.append(np.asarray(flatten_feature_vector(feature_vector), dtype=np.float32))
#         y.append(int(label_binary))
#         groups.append(str(participant_id))

#     if len(X) == 0:
#         raise ValueError("No usable rows were found.")

#     X = np.asarray(X, dtype=np.float32)
#     y = np.asarray(y, dtype=np.int32)
#     groups = np.asarray(groups)

#     return X, y, groups, {
#         "num_rows": int(len(rows)),
#         "num_used": int(len(y)),
#         "num_skipped": int(skipped),
#         "missing_feature": int(missing_feature),
#         "missing_label": int(missing_label),
#         "missing_group": int(missing_group),
#         "X_shape": tuple(X.shape),
#         "label_counts": np.bincount(y, minlength=2).tolist(),
#         "groups": sorted(set(groups.tolist())),
#         "group_counts": {
#             g: int(np.sum(groups == g)) for g in sorted(set(groups.tolist()))
#         },
#     }


# def split_rows_by_participant(rows, test_size: float = 0.25, random_state: int = 0):
#     X, y, groups, summary = rows_to_xy_and_groups(rows)

#     # Keep an aligned list of usable rows so split indices map correctly.
#     usable_rows = []
#     for row in rows:
#         if row.get("skipped", False):
#             continue
#         if row.get("feature_vector") is None:
#             continue
#         if row.get("label_binary") is None:
#             continue
#         if row.get("participant_id") is None:
#             continue
#         usable_rows.append(row)

#     if len(usable_rows) != len(y):
#         raise ValueError(
#             "Internal split mismatch: usable rows and label array length differ "
#             f"({len(usable_rows)} != {len(y)})."
#         )

#     unique_groups = np.array(sorted(set(groups.tolist())))
#     if len(unique_groups) < 2:
#         raise ValueError("Need at least two participants to do participant-based evaluation.")

#     rng = np.random.default_rng(random_state)
#     shuffled_groups = unique_groups.copy()
#     rng.shuffle(shuffled_groups)

#     if len(unique_groups) == 2:
#         test_groups = np.asarray([shuffled_groups[0]])
#     else:
#         num_test_groups = max(1, int(round(len(unique_groups) * test_size)))
#         num_test_groups = min(num_test_groups, len(unique_groups) - 1)
#         test_groups = shuffled_groups[:num_test_groups]

#     test_mask = np.isin(groups, test_groups)
#     train_mask = ~test_mask

#     train_idx = np.where(train_mask)[0]
#     test_idx = np.where(test_mask)[0]

#     if len(train_idx) == 0 or len(test_idx) == 0:
#         raise ValueError("Participant split produced an empty train or test set.")

#     split_summary = {
#         "test_size": float(test_size),
#         "random_state": int(random_state),
#         "train_size": int(len(train_idx)),
#         "test_size_count": int(len(test_idx)),
#         "train_label_counts": np.bincount(y[train_idx], minlength=2).tolist(),
#         "test_label_counts": np.bincount(y[test_idx], minlength=2).tolist(),
#         "train_participants": sorted(set(groups[train_idx].tolist())),
#         "test_participants": sorted(set(groups[test_idx].tolist())),
#     }

#     train_rows = [usable_rows[i] for i in train_idx]
#     test_rows = [usable_rows[i] for i in test_idx]

#     return train_rows, test_rows, summary, split_summary

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any 
import json
import numpy as np
import cv2


REQUIRED_FILES = ("rgb.png", "depth_z16.png", "joints_raw.json", "meta.json")

@dataclass(frozen=True)
class Sample:
    sample_dir: Path
    rgb_path: Path
    depth_path: Path
    joints_path: Path
    meta_path: Path
    participant_id: str
    label_name: str
    condition: str
    hand: str
    sample_type: str
    sample_id: str


def get_sample_label_name(sample):
    """
    Tries to robustly extract the ground-truth label from a Sample.
    Adjust this if your Sample dataclass uses a different field.
    """

    for attr in ["label_name", "label", "class_name", "target", "y"]:
        if hasattr(sample, attr):
            value = getattr(sample, attr)
            if value is not None:
                return str(value)

    # Fallback: infer from path.
    # Check not_handoff first because it contains the substring "handoff".
    path = str(sample.rgb_path).lower()

    if "not_handoff" in path or "non_handoff" in path or "no_handoff" in path:
        return "not_handoff"

    if "handoff" in path:
        return "handoff"

    raise ValueError(f"Could not determine label for sample: {sample}")

def label_to_binary(label_name):
    """
    Returns 1 for handoff, 0 for not handoff.
    """

    label = str(label_name).lower()

    if label in ["handoff", "positive", "pos", "1", "true"]:
        return 1

    if label in ["not_handoff", "non_handoff", "no_handoff", "negative", "neg", "0", "false"]:
        return 0

    # Path-style labels sometimes contain these substrings.
    if "not_handoff" in label or "non_handoff" in label or "no_handoff" in label:
        return 0

    if "handoff" in label:
        return 1

    raise ValueError(f"Unknown label name: {label_name}")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[warning] Could not read JSON {path}: {exc}")
        return {}

def sample_root_from_dataset_root(dataset_root: Path) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    if (dataset_root / "samples").is_dir():
        return dataset_root / "samples"
    return dataset_root


def infer_metadata_from_path(sample_dir: Path, sample_root: Path) -> Dict[str, str]:
    try:
        rel = sample_dir.relative_to(sample_root).parts
    except ValueError:
        rel = sample_dir.parts

    # Expected: participant / label_name / condition / hand / sample_id
    return {
        "participant_id": rel[0] if len(rel) > 0 else "unknown_participant",
        "label_name": rel[1] if len(rel) > 1 else "unknown_label",
        "condition": rel[2] if len(rel) > 2 else "unknown_condition",
        "hand": rel[3] if len(rel) > 3 else "unknown_hand",
        "sample_id": rel[4] if len(rel) > 4 else sample_dir.name,
        "sample_type": "unknown_sample_type",
    }


def discover_samples(
    dataset_root: Path,
    participant_filter: Optional[str],
    label_filter: Optional[str],
    condition_filter: Optional[str],
    hand_filter: Optional[str],
) -> List[Sample]:
    sample_root = sample_root_from_dataset_root(dataset_root)
    if not sample_root.is_dir():
        raise FileNotFoundError(f"Sample root does not exist: {sample_root}")

    samples: List[Sample] = []
    for rgb_path in sample_root.rglob("rgb.png"):
        sample_dir = rgb_path.parent
        if not all((sample_dir / filename).is_file() for filename in REQUIRED_FILES):
            continue

        inferred = infer_metadata_from_path(sample_dir, sample_root)
        meta = read_json(sample_dir / "meta.json")

        participant_id = str(meta.get("participant_id") or inferred["participant_id"])
        label_name = str(meta.get("label_name") or inferred["label_name"])
        condition = str(meta.get("condition") or inferred["condition"])
        hand = str(meta.get("hand") or inferred["hand"])
        sample_type = str(meta.get("sample_type") or inferred["sample_type"])
        sample_id = str(meta.get("sample_id") or inferred["sample_id"])

        if participant_filter and participant_id != participant_filter:
            continue
        if label_filter and label_name != label_filter:
            continue
        if condition_filter and condition != condition_filter:
            continue
        if hand_filter and hand != hand_filter:
            continue

        samples.append(
            Sample(
                sample_dir=sample_dir,
                rgb_path=rgb_path,
                depth_path=sample_dir / "depth_z16.png",
                joints_path=sample_dir / "joints_raw.json",
                meta_path=sample_dir / "meta.json",
                participant_id=participant_id,
                label_name=label_name,
                condition=condition,
                hand=hand,
                sample_type=sample_type,
                sample_id=sample_id,
            )
        )

    samples.sort(key=lambda s: (s.participant_id, s.label_name, s.condition, s.hand, s.sample_id, str(s.sample_dir)))
    return samples


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


def get_depth_at_rgb_point(
    depth_image: np.ndarray,
    rgb_image_shape,
    point_xy,
    patch_radius: int = 2,
) -> float:
    """
    Returns the median nonzero depth near an RGB-image point.

    The RGB and depth images may have different resolutions, so the point is
    mapped into depth-image coordinates before sampling. A small patch is used
    because a single Z16 depth pixel may be invalid (zero).
    """

    rgb_height, rgb_width = rgb_image_shape[:2]
    depth_height, depth_width = depth_image.shape[:2]

    if rgb_width <= 0 or rgb_height <= 0:
        raise ValueError("RGB image has invalid dimensions.")

    x_rgb, y_rgb = map(float, point_xy)
    x_depth = int(round(x_rgb * depth_width / rgb_width))
    y_depth = int(round(y_rgb * depth_height / rgb_height))

    x_depth = int(np.clip(x_depth, 0, depth_width - 1))
    y_depth = int(np.clip(y_depth, 0, depth_height - 1))

    x_start = max(0, x_depth - patch_radius)
    x_end = min(depth_width, x_depth + patch_radius + 1)
    y_start = max(0, y_depth - patch_radius)
    y_end = min(depth_height, y_depth + patch_radius + 1)

    patch = np.asarray(
        depth_image[y_start:y_end, x_start:x_end],
        dtype=np.float32,
    )
    valid_depths = patch[np.isfinite(patch) & (patch > 0)]

    if valid_depths.size == 0:
        raise ValueError(
            f"No valid depth near RGB point ({x_rgb:.1f}, {y_rgb:.1f})."
        )

    return float(np.median(valid_depths))


def organize_samples(samples, features_type,obj_detector, keypoint_detector, head_pose_estimator, img_encoder, args):
    y_true = []
    rows = []
    skipped_unreadable = 0
    skipped_no_object = 0
    skipped_no_people = 0
    errors = 0



    for local_idx, sample in enumerate(samples):
        image_path = str(sample.rgb_path)
        depth_path = str(sample.depth_path)
        image = cv2.imread(image_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        objects = []
        people = []

        

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

        if depth_image is None:
            skipped_unreadable += 1
            print(f"[{local_idx + 1}/{len(samples)}] SKIP unreadable depth image: {depth_path}")
            rows.append(
                {
                    "index": local_idx,
                    "image": image_path,
                    "depth": depth_path,
                    "skipped": True,
                    "skip_reason": "unreadable_depth_image",
                }
            )
            continue

        try:
            if(args.crop_around_object):
                crop_objs = obj_detector.predict(image, class_names=args.crop_object)
                if len(crop_objs) == 0:
                    print("SKIP no person detected ")
                    continue

                sorted_boxes = sorted(
                                        [d for d in crop_objs if d["label"].strip(".") == args.crop_object.strip(".")],
                                        key=lambda d: (
                                            (d["box_xyxy"][2] - d["box_xyxy"][0]) *
                                            (d["box_xyxy"][3] - d["box_xyxy"][1])
                                        ),
                                        reverse=True,  # largest first
                                    )
                box_xyxy = sorted_boxes[0]["box_xyxy"]
                x1, y1, x2, y2 = map(int, box_xyxy)

                rgb_height, rgb_width = image.shape[:2]
                depth_height, depth_width = depth_image.shape[:2]

                depth_x1 = int(round(x1 * depth_width / rgb_width))
                depth_y1 = int(round(y1 * depth_height / rgb_height))
                depth_x2 = int(round(x2 * depth_width / rgb_width))
                depth_y2 = int(round(y2 * depth_height / rgb_height))

                cropped_image = image[y1:y2, x1:x2]
                cropped_depth = depth_image[
                    depth_y1:depth_y2,
                    depth_x1:depth_x2,
                ]

                if cropped_image.size == 0 or cropped_depth.size == 0:
                    raise ValueError("Object crop produced an empty RGB or depth image.")

                image = cropped_image
                depth_image = cropped_depth

            # det_hands = obj_detector.predict(image, class_names="hand.")
            # if len(det_hands) == 0:
            #                 print("SKIP no hands detected ")
            #                 skipped_no_people += 1
            #                 continue

            # sorted_hands = sorted(
            #                         [d for d in det_hands if d["label"].strip(".") == "hand"],
            #                         key=lambda d: (
            #                             (d["box_xyxy"][2] - d["box_xyxy"][0]) *
            #                             (d["box_xyxy"][3] - d["box_xyxy"][1])
            #                         ),
            #                         reverse=True,  # largest first
            #                     )
            # biggest_hand_bbox = sorted_hands[0]["box_xyxy"]
            # x1, y1, x2, y2 = map(int, biggest_hand_bbox)
            # outstretched_hand_mp = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            
                

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
            if (
                features_type == "keypoints"
                or features_type == "keypoints_headpose"
                or features_type == "keypoints_headpose_dino"
                or features_type == "keypoints_resnet"
                or features_type == "keypoints_headpose_resnet"
            ):
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

            if (
                features_type == "keypoints_headpose"
                or features_type == "keypoints_headpose_dino"
                or features_type == "keypoints_headpose_resnet"
            ):
                head_pose = head_pose_estimator.predict_first_or_sentinel(
                    image,
                    rtmpose_results=people,
                )
            

            keypoints = np.asarray(people[0]["keypoints"][:11], dtype=np.float32)
            left_shoulder = keypoints[5, :2]
            right_shoulder = keypoints[6, :2]
            shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0

            keypoints_depth = []
            for i in range(5,11):
                try:
                    kp_depth = get_depth_at_rgb_point(depth_image, image.shape, keypoints[i, :2])
                    keypoints_depth.append(kp_depth)
                except Exception as e:
                    print(f"Error occurred while calculating depth for keypoint {i}: {e}")
                    skipped_no_people += 1
                    continue

            # hand_depth = get_depth_at_rgb_point(
            #     depth_image,
            #     image.shape,
            #     outstretched_hand_mp,
            # )
            # shoulder_depth = get_depth_at_rgb_point(
            #     depth_image,
            #     image.shape,
            #     shoulder_midpoint,
            # )

            # Analogous to the old hand-minus-shoulder 2D vector:
            # positive means the hand is farther from the camera than the shoulders,
            # while negative means the hand is closer.
            #hand_to_shoulder_depth_difference = hand_depth - shoulder_depth

            if(args.normalize_keypoints):
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
            elif features_type == "keypoints_resnet":
                resnet_embedding = img_encoder.predict(image)["embedding"]
                row = {
                    "participant_id": sample.participant_id,
                    "index": local_idx,
                    "image": image_path,
                    "label_name": label_name,
                    "label_binary": int(y),
                    "num_people": len(people),
                    "feature_vector": (
                        flatten_feature_vector(keypoints)
                        + resnet_embedding.flatten().astype(np.float32).tolist()
                    ),
                    "skipped": False,
                }
            elif features_type == "keypoints_headpose_resnet":
                resnet_embedding = img_encoder.predict(image)["embedding"]
                row = {
                    "participant_id": sample.participant_id,
                    "index": local_idx,
                    "image": image_path,
                    "label_name": label_name,
                    "label_binary": int(y),
                    "num_people": len(people),
                    "feature_vector": (
                        flatten_feature_vector(keypoints)
                        + head_pose.flatten().astype(np.float32).tolist()
                        + resnet_embedding.flatten().astype(np.float32).tolist()
                    ),
                    "skipped": False,
                }
            if(sample.joints_path is not None):
                row["quest_joints_pth"] = sample.joints_path

            
            # row["hand_depth"] = hand_depth
            # row["shoulder_depth"] = shoulder_depth
            # row["hand_to_shoulder_depth_difference"] = hand_to_shoulder_depth_difference
            # row["feature_vector"] = row["feature_vector"] + [
            #     hand_to_shoulder_depth_difference
            # ]
            row["joint_depths"] = keypoints_depth
            row["feature_vector"] = row["feature_vector"] + keypoints_depth
            
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





def rows_to_xy(rows,feature_key="feature_vector", label_key="label_binary"):
    X = []
    y = []

    skipped = 0
    missing_feature = 0
    missing_label = 0

    for row in rows:
        if row.get("skipped", False):
            skipped += 1
            continue

        feature_vector = row.get(feature_key)
        label_binary = row.get(label_key)

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

    # Keep an aligned list of usable rows so split indices map correctly.
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

    if len(usable_rows) != len(y):
        raise ValueError(
            "Internal split mismatch: usable rows and label array length differ "
            f"({len(usable_rows)} != {len(y)})."
        )

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

    train_rows = [usable_rows[i] for i in train_idx]
    test_rows = [usable_rows[i] for i in test_idx]

    return train_rows, test_rows, summary, split_summary