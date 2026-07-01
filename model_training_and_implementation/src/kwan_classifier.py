import numpy as np
import torch

try:
    from .kwan_models.mlp import MLP
except ImportError:
    from src.kwan_models.mlp import MLP


LEFT_WRIST = 9
RIGHT_WRIST = 10
NUM_UPPER_BODY_KEYPOINTS = 11


def _as_box_array(obj):
    """
    Accept either:
      - dict with "box_xyxy"
      - object with .box_xyxy
      - raw [x1, y1, x2, y2]
    """
    if isinstance(obj, dict):
        return np.asarray(obj["box_xyxy"], dtype=np.float32)

    if hasattr(obj, "box_xyxy"):
        return np.asarray(obj.box_xyxy, dtype=np.float32)

    return np.asarray(obj, dtype=np.float32)


def _as_keypoints_array(person):
    """
    Accept either:
      - dict with "keypoints" or "keypoints_xy"
      - object with .keypoints or .keypoints_xy
      - raw array shaped [17, 2] or [17, 3]
    """
    if isinstance(person, dict):
        if "keypoints" in person:
            return np.asarray(person["keypoints"], dtype=np.float32)
        return np.asarray(person["keypoints_xy"], dtype=np.float32)

    if hasattr(person, "keypoints"):
        return np.asarray(person.keypoints, dtype=np.float32)

    if hasattr(person, "keypoints_xy"):
        return np.asarray(person.keypoints_xy, dtype=np.float32)

    return np.asarray(person, dtype=np.float32)


def box_center_xy(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def select_person_object_pair(objects, people):
    """
    Select the person-object pair with the smallest distance between
    object center and either wrist.

    This improves over Kwan's original assumption of object[0], person[0],
    while keeping the final 26-D feature vector compatible with their MLP.
    """

    if len(objects) == 0 or len(people) == 0:
        return None

    best = None

    for person_idx, person in enumerate(people):
        keypoints = _as_keypoints_array(person)

        if keypoints.shape[0] <= RIGHT_WRIST:
            continue

        left_wrist = keypoints[LEFT_WRIST, 0:2]
        right_wrist = keypoints[RIGHT_WRIST, 0:2]

        for object_idx, obj in enumerate(objects):
            box = _as_box_array(obj)
            center = box_center_xy(box)

            left_dist = float(np.linalg.norm(center - left_wrist))
            right_dist = float(np.linalg.norm(center - right_wrist))

            if left_dist <= right_dist:
                dist = left_dist
                wrist = "left_wrist"
            else:
                dist = right_dist
                wrist = "right_wrist"

            candidate = {
                "person_index": person_idx,
                "object_index": object_idx,
                "wrist": wrist,
                "wrist_object_distance_px": dist,
            }

            if best is None or dist < best["wrist_object_distance_px"]:
                best = candidate

    return best


def build_kwan_localized_features(objects, people, head_pose):
    """
    Builds the 26-D feature vector expected by Kwan's localized MLP:

      1 object-present flag
      22 localized upper-body keypoint coordinates
      3 head-pose values

    Output shape: [26]
    """

    association = select_person_object_pair(objects, people)

    if association is None:
        feature = np.concatenate(
            [
                np.asarray([0.0], dtype=np.float32),
                np.zeros((NUM_UPPER_BODY_KEYPOINTS * 2,), dtype=np.float32),
                np.asarray([-999.0, -999.0, -999.0], dtype=np.float32),
            ]
        )

        return feature, {
            "selected_person_index": None,
            "selected_object_index": None,
            "selected_wrist": None,
            "wrist_object_distance_px": None,
            "object_present": 0,
        }

    person_idx = association["person_index"]
    object_idx = association["object_index"]

    keypoints = _as_keypoints_array(people[person_idx])
    object_box = _as_box_array(objects[object_idx])

    # Kwan uses first 11 COCO keypoints:
    # nose, eyes, ears, shoulders, elbows, wrists.
    upper_body_xy = keypoints[:NUM_UPPER_BODY_KEYPOINTS, 0:2]

    # Kwan localizes keypoints relative to the object centroid.
    object_center = box_center_xy(object_box)
    localized_keypoints = upper_body_xy - object_center
    localized_keypoints = localized_keypoints.flatten().astype(np.float32)

    if head_pose is None:
        head_pose_arr = np.asarray([-999.0, -999.0, -999.0], dtype=np.float32)
    else:
        head_pose_arr = np.asarray(head_pose, dtype=np.float32).flatten()
        if head_pose_arr.shape[0] != 3:
            head_pose_arr = np.asarray([-999.0, -999.0, -999.0], dtype=np.float32)

    feature = np.concatenate(
        [
            np.asarray([1.0], dtype=np.float32),
            localized_keypoints,
            head_pose_arr,
        ]
    )

    assert feature.shape[0] == 26, f"Expected 26 features, got {feature.shape[0]}"

    debug = {
        "selected_person_index": int(person_idx),
        "selected_object_index": int(object_idx),
        "selected_wrist": association["wrist"],
        "wrist_object_distance_px": float(association["wrist_object_distance_px"]),
        "object_present": 1,
        "selected_object_box_xyxy": object_box.tolist(),
    }

    return feature.astype(np.float32), debug


class KwanLocalizedMLPClassifier:
    def __init__(
        self,
        weights_path="kwan_pretrained_weights/MLP_localized.pth",
        device=None,
        threshold=0.5,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.threshold = threshold

        self.model = MLP(input_size=26, output_size=1).to(self.device)

        checkpoint = torch.load(weights_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            raise TypeError(
                f"Unexpected checkpoint type: {type(checkpoint)}. "
                "Expected a state_dict-like dict."
            )

        self.model.eval()

    def predict_from_features(self, feature_vector):
        x = torch.from_numpy(feature_vector.astype(np.float32)).float().to(self.device)

        with torch.no_grad():
            y = self.model(x)

        score = float(y.detach().cpu().view(-1)[0])

        return {
            "handoff_score": score,
            "handoff_detected": bool(score >= self.threshold),
            "threshold": self.threshold,
        }

    def predict(self, objects, people, head_pose):
        feature_vector, debug = build_kwan_localized_features(
            objects=objects,
            people=people,
            head_pose=head_pose,
        )

        pred = self.predict_from_features(feature_vector)

        return {
            **pred,
            "feature_vector": feature_vector.tolist(),
            "association": debug,
        }