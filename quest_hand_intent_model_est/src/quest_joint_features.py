import json
import numpy as np
from typing import List,Tuple, Any 


QUEST_JOINT_ORDER = [
    # "hmd_center_eye",
    # "root",
    "hips",
    "spine_lower",
    "spine_middle",
    "spine_upper",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_scapula",
    "right_scapula",
    "left_upper_arm",
    "right_upper_arm",
    "left_forearm",
    "right_forearm",
    "left_hand",
    "right_hand",
    "left_wrist_twist",
    "right_wrist_twist",
    "left_palm",
    "right_palm",
]

SKELETON_EDGES: List[Tuple[str, str]] = [
    ("hips", "spine_lower"),
    ("spine_lower", "spine_middle"),
    ("spine_middle", "spine_upper"),
    ("spine_upper", "chest"),
    ("chest", "neck"),
    ("neck", "head"),
    #("head", "hmd_center_eye"),
    ("chest", "left_scapula"),
    ("left_scapula", "left_shoulder"),
    ("left_shoulder", "left_upper_arm"),
    ("left_upper_arm", "left_forearm"),
    ("left_forearm", "left_wrist_twist"),
    ("left_forearm", "left_hand"),
    ("left_hand", "left_palm"),
    ("chest", "right_scapula"),
    ("right_scapula", "right_shoulder"),
    ("right_shoulder", "right_upper_arm"),
    ("right_upper_arm", "right_forearm"),
    ("right_forearm", "right_wrist_twist"),
    ("right_forearm", "right_hand"),
    ("right_hand", "right_palm"),
]


def normalize_quaternion(quaternion):
    quaternion = np.asarray(
        quaternion,
        dtype=np.float32,
    )

    magnitude = np.linalg.norm(quaternion)

    if magnitude < 1e-8:
        raise ValueError(
            "Quaternion has near-zero magnitude."
        )

    quaternion = quaternion / magnitude

    # q and -q represent the same rotation.
    # Enforce one representation for consistent features.
    if quaternion[3] < 0:
        quaternion = -quaternion

    return quaternion


def quaternion_conjugate(quaternion):
    x, y, z, w = quaternion

    return np.asarray(
        [-x, -y, -z, w],
        dtype=np.float32,
    )


def quaternion_multiply(q1, q2):
    """
    Multiply quaternions stored in Unity order: [x, y, z, w].
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def normalize_quest_joints(
    joints,
    include_rotations=False,
):
    """
    Normalize Quest joint data.

    Positions are expressed relative to the midpoint of the shoulders.

    When rotations are included, each joint rotation is expressed
    relative to the chest rotation.

    Returns
    -------
    dict
        A dictionary with the same joint names. Each joint contains
        a normalized ``position`` NumPy array and, when requested,
        a normalized ``rotation`` NumPy array.
    """
    left_shoulder = np.asarray(
        joints["left_shoulder"]["position"],
        dtype=np.float32,
    )

    right_shoulder = np.asarray(
        joints["right_shoulder"]["position"],
        dtype=np.float32,
    )

    shoulder_midpoint = (
        left_shoulder + right_shoulder
    ) / 2.0

    inverse_reference_rotation = None

    if include_rotations:
        reference_rotation = normalize_quaternion(
            joints["chest"]["rotation"]
        )

        inverse_reference_rotation = quaternion_conjugate(
            reference_rotation
        )

    normalized_joints = {}

    for joint_name in QUEST_JOINT_ORDER:
        joint = joints[joint_name]

        position = np.asarray(
            joint["position"],
            dtype=np.float32,
        )

        normalized_joint = {
            "position": position - shoulder_midpoint,
        }

        if include_rotations:
            joint_rotation = normalize_quaternion(
                joint["rotation"]
            )

            relative_rotation = quaternion_multiply(
                inverse_reference_rotation,
                joint_rotation,
            )

            normalized_joint["rotation"] = (
                normalize_quaternion(relative_rotation)
            )

        normalized_joints[joint_name] = normalized_joint

    return normalized_joints


def construct_quest_joint_feature_vector(
    normalized_joints,
    include_rotations=False,
):
    """
    Flatten normalized joint data into the model feature vector.
    """
    feature_vector = []

    for joint_name in QUEST_JOINT_ORDER:
        joint = normalized_joints[joint_name]

        feature_vector.extend(
            joint["position"].tolist()
        )

        if include_rotations:
            feature_vector.extend(
                joint["rotation"].tolist()
            )

    return np.asarray(
        feature_vector,
        dtype=np.float32,
    )


def extract_quest_joint_features_from_joints(
    joints,
    include_rotations=False,
):
    """
    Normalize an in-memory joints dictionary and construct its features.
    """
    normalized_joints = normalize_quest_joints(
        joints,
        include_rotations=include_rotations,
    )

    return construct_quest_joint_feature_vector(
        normalized_joints,
        include_rotations=include_rotations,
    )


def extract_quest_joint_features(
    json_path,
    include_rotations=False,
):
    """
    Extract normalized joint features from a Quest joint JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return extract_quest_joint_features_from_joints(
        data["joints"],
        include_rotations=include_rotations,
    )


def extract_quest_joint_features_json(
    raw_json,
    include_rotations=False,
):
    """
    Extract normalized joint features from an in-memory Quest joint JSON object.
    """
    return extract_quest_joint_features_from_joints(
        raw_json["joints"],
        include_rotations=include_rotations,
    )

def feature_vector_to_joint_poses(
    feature_vector: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """
    Convert a flattened Quest joint feature vector into joint poses.

    Supports either:

        3 features per joint:
            [x, y, z]

        7 features per joint:
            [x, y, z, qx, qy, qz, qw]

    Returns
    -------
    dict
        Mapping from joint name to:

        {
            "position": np.ndarray with shape (3,),
            "rotation": np.ndarray with shape (4,) or None,
        }
    """
    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32,
    ).reshape(-1)

    joint_count = len(QUEST_JOINT_ORDER)

    position_only_size = joint_count * 3
    position_rotation_size = joint_count * 7

    if feature_vector.size == position_only_size:
        features_per_joint = 3
        rotations_included = False

    elif feature_vector.size == position_rotation_size:
        features_per_joint = 7
        rotations_included = True

    else:
        raise ValueError(
            "Unexpected feature-vector length. "
            f"Expected {position_only_size} values for XYZ-only features "
            f"or {position_rotation_size} values for XYZ plus quaternion "
            f"features, but received {feature_vector.size}."
        )

    joint_poses = {}

    for joint_index, joint_name in enumerate(
        QUEST_JOINT_ORDER
    ):
        start = joint_index * features_per_joint

        position = feature_vector[
            start : start + 3
        ].copy()

        rotation = None

        if rotations_included:
            rotation = feature_vector[
                start + 3 : start + 7
            ].copy()

        joint_poses[joint_name] = {
            "position": position,
            "rotation": rotation,
        }

    return joint_poses