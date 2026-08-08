# import json
# import numpy as np
# from typing import List,Tuple, Any 
# import cv2

# QUEST_JOINT_ORDER = [
#     # "hmd_center_eye",
#     # "root",
#     "hips",
#     "spine_lower",
#     "spine_middle",
#     "spine_upper",
#     "chest",
#     "neck",
#     "head",
#     "left_shoulder",
#     "right_shoulder",
#     "left_scapula",
#     "right_scapula",
#     "left_upper_arm",
#     "right_upper_arm",
#     "left_forearm",
#     "right_forearm",
#     "left_hand",
#     "right_hand",
#     "left_wrist_twist",
#     "right_wrist_twist",
#     "left_palm",
#     "right_palm",
# ]

# SKELETON_EDGES: List[Tuple[str, str]] = [
#     ("hips", "spine_lower"),
#     ("spine_lower", "spine_middle"),
#     ("spine_middle", "spine_upper"),
#     ("spine_upper", "chest"),
#     ("chest", "neck"),
#     ("neck", "head"),
#     #("head", "hmd_center_eye"),
#     ("chest", "left_scapula"),
#     ("left_scapula", "left_shoulder"),
#     ("left_shoulder", "left_upper_arm"),
#     ("left_upper_arm", "left_forearm"),
#     ("left_forearm", "left_wrist_twist"),
#     ("left_forearm", "left_hand"),
#     ("left_hand", "left_palm"),
#     ("chest", "right_scapula"),
#     ("right_scapula", "right_shoulder"),
#     ("right_shoulder", "right_upper_arm"),
#     ("right_upper_arm", "right_forearm"),
#     ("right_forearm", "right_wrist_twist"),
#     ("right_forearm", "right_hand"),
#     ("right_hand", "right_palm"),
# ]


# def normalize_quaternion(quaternion):
#     quaternion = np.asarray(
#         quaternion,
#         dtype=np.float32,
#     )

#     magnitude = np.linalg.norm(quaternion)

#     if magnitude < 1e-8:
#         raise ValueError(
#             "Quaternion has near-zero magnitude."
#         )

#     quaternion = quaternion / magnitude

#     # q and -q represent the same rotation.
#     # Enforce one representation for consistent features.
#     if quaternion[3] < 0:
#         quaternion = -quaternion

#     return quaternion


# def quaternion_conjugate(quaternion):
#     x, y, z, w = quaternion

#     return np.asarray(
#         [-x, -y, -z, w],
#         dtype=np.float32,
#     )


# def quaternion_multiply(q1, q2):
#     """
#     Multiply quaternions stored in Unity order: [x, y, z, w].
#     """
#     x1, y1, z1, w1 = q1
#     x2, y2, z2, w2 = q2

#     return np.asarray(
#         [
#             w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
#             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
#             w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
#             w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
#         ],
#         dtype=np.float32,
#     )


# def normalize_quest_joints(
#     joints,
#     include_rotations=False,
# ):
#     """
#     Normalize Quest joint data.

#     Positions are expressed relative to the midpoint of the shoulders.

#     When rotations are included, each joint rotation is expressed
#     relative to the chest rotation.

#     Returns
#     -------
#     dict
#         A dictionary with the same joint names. Each joint contains
#         a normalized ``position`` NumPy array and, when requested,
#         a normalized ``rotation`` NumPy array.
#     """
#     left_shoulder = np.asarray(
#         joints["left_shoulder"]["position"],
#         dtype=np.float32,
#     )

#     right_shoulder = np.asarray(
#         joints["right_shoulder"]["position"],
#         dtype=np.float32,
#     )

#     shoulder_midpoint = (
#         left_shoulder + right_shoulder
#     ) / 2.0

#     inverse_reference_rotation = None

#     if include_rotations:
#         reference_rotation = normalize_quaternion(
#             joints["chest"]["rotation"]
#         )

#         inverse_reference_rotation = quaternion_conjugate(
#             reference_rotation
#         )

#     normalized_joints = {}

#     for joint_name in QUEST_JOINT_ORDER:
#         joint = joints[joint_name]

#         position = np.asarray(
#             joint["position"],
#             dtype=np.float32,
#         )

#         normalized_joint = {
#             "position": position - shoulder_midpoint,
#         }

#         if include_rotations:
#             joint_rotation = normalize_quaternion(
#                 joint["rotation"]
#             )

#             relative_rotation = quaternion_multiply(
#                 inverse_reference_rotation,
#                 joint_rotation,
#             )

#             normalized_joint["rotation"] = (
#                 normalize_quaternion(relative_rotation)
#             )

#         normalized_joints[joint_name] = normalized_joint

#     return normalized_joints


# def construct_quest_joint_feature_vector(
#     normalized_joints,
#     include_rotations=False,
# ):
#     """
#     Flatten normalized joint data into the model feature vector.
#     """
#     feature_vector = []

#     for joint_name in QUEST_JOINT_ORDER:
#         joint = normalized_joints[joint_name]

#         feature_vector.extend(
#             joint["position"].tolist()
#         )


#         if include_rotations:
#             feature_vector.extend(
#                 joint["rotation"].tolist()
#             )

#     return np.asarray(
#         feature_vector,
#         dtype=np.float32,
#     )

# def draw_skeleton_connections(
#     pose_img,
#     pixel_x,
#     pixel_y,
#     valid,
#     thickness=2,
#     value=1.0,
# ):
#     """Draw valid skeleton edges onto a single-channel pose image."""
#     joint_indices = {
#         joint_name: index
#         for index, joint_name in enumerate(QUEST_JOINT_ORDER)
#     }

#     for start_name, end_name in SKELETON_EDGES:
#         start_index = joint_indices[start_name]
#         end_index = joint_indices[end_name]

#         if not (valid[start_index] and valid[end_index]):
#             continue

#         start_point = (
#             int(pixel_x[start_index]),
#             int(pixel_y[start_index]),
#         )
#         end_point = (
#             int(pixel_x[end_index]),
#             int(pixel_y[end_index]),
#         )

#         cv2.line(
#             pose_img,
#             start_point,
#             end_point,
#             color=float(value),
#             thickness=thickness,
#             lineType=cv2.LINE_AA,
#         )

# def construct_pose_img_from_joint_feature_vector(
#     normalized_joints,
#     im_w=640,
#     im_h=640,
#     margin_in_img=10,
#     plane_extent=0.5,
# ):
#     """
#     Construct a pose image by projecting 3D joints onto the plane defined
#     by the chest and shoulders. Returns the joints projected onto the chest-shoulder plane, as well as a 2D image of the projected joints and skeleton edges.

#     plane_extent defines the represented distance from the chest in each
#     plane direction: [-plane_extent, plane_extent].
#     """
#     eps = 1e-8

#     joints = np.asarray(normalized_joints, dtype=np.float32)

#     if joints.size % 3 != 0:
#         raise ValueError("Joint feature vector length must be divisible by 3.")

#     positions = joints.reshape(-1, 3)


#     if len(positions) != len(QUEST_JOINT_ORDER):
#         raise ValueError("Joint count does not match QUEST_JOINT_ORDER.")

#     if not np.all(np.isfinite(positions)):
#         raise ValueError("Joint positions contain non-finite values.")

#     if im_w <= 0 or im_h <= 0:
#         raise ValueError("Image dimensions must be positive.")

#     if margin_in_img < 0:
#         raise ValueError("Image margin cannot be negative.")

#     if 2 * margin_in_img >= min(im_w, im_h):
#         raise ValueError("Image margin is too large.")

#     if plane_extent <= 0:
#         raise ValueError("Plane extent must be positive.")

#     chest = positions[QUEST_JOINT_ORDER.index("chest")]
#     right_shoulder = positions[
#         QUEST_JOINT_ORDER.index("right_shoulder")
#     ]
#     left_shoulder = positions[
#         QUEST_JOINT_ORDER.index("left_shoulder")
#     ]

#     normal_raw = np.cross(
#         right_shoulder - chest,
#         left_shoulder - chest,
#     )
#     normal_norm = np.linalg.norm(normal_raw)

#     if normal_norm < eps:
#         raise ValueError(
#             "Chest and shoulders do not define a stable plane."
#         )

#     plane_normal = normal_raw / normal_norm

#     u_raw = left_shoulder - right_shoulder
#     u_norm = np.linalg.norm(u_raw)

#     if u_norm < eps:
#         raise ValueError("Shoulder positions are coincident.")

#     u = u_raw / u_norm

#     v = np.cross(plane_normal, u)
#     v_norm = np.linalg.norm(v)

#     if v_norm < eps:
#         raise ValueError(
#             "Could not construct the plane coordinate frame."
#         )

#     v /= v_norm

#     # Keep positive v approximately aligned with world up.
#     world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
#     if np.dot(v, world_up) < 0:
#         v = -v

#     relative_positions = positions - chest

#     proj_joints = np.column_stack(
#         (
#             relative_positions @ u,
#             relative_positions @ v,
#         )
#     )

#     c_x = (im_w - 1) / 2.0
#     c_y = (im_h - 1) / 2.0

#     x_scale = (c_x - margin_in_img) / plane_extent
#     y_scale = (c_y - margin_in_img) / plane_extent

#     # Use one scale to preserve geometric proportions.
#     scale = min(x_scale, y_scale)

#     pixel_x = np.rint(
#         c_x + proj_joints[:, 0] * scale
#     ).astype(np.int32)

#     pixel_y = np.rint(
#         c_y - proj_joints[:, 1] * scale
#     ).astype(np.int32)

#     valid = (
#         (pixel_x >= 0)
#         & (pixel_x < im_w)
#         & (pixel_y >= 0)
#         & (pixel_y < im_h)
#     )

#     pose_img = np.zeros((im_h, im_w), dtype=np.float32)

#     draw_skeleton_connections(
#         pose_img,
#         pixel_x,
#         pixel_y,
#         valid,
#         thickness=2,
#     )

#     for x, y in zip(pixel_x[valid], pixel_y[valid]):
#         cv2.circle(
#             pose_img,
#             (int(x), int(y)),
#             radius=3,
#             color=1.0,
#             thickness=-1,
#             lineType=cv2.LINE_AA,
#         )
#     pose_img_vis = np.clip(pose_img * 255.0, 0, 255).astype(np.uint8)

#     return proj_joints.reshape(-1), pose_img_vis 

    



    
        
    


# def extract_quest_joint_features_from_joints(
#     joints,
#     include_rotations=False,
# ):
#     """
#     Normalize an in-memory joints dictionary and construct its features.
#     """
#     normalized_joints = normalize_quest_joints(
#         joints,
#         include_rotations=include_rotations,
#     )

#     return construct_quest_joint_feature_vector(
#         normalized_joints,
#         include_rotations=include_rotations,
#     )


# def extract_quest_joint_features(
#     json_path,
#     include_rotations=False,
# ):
#     """
#     Extract normalized joint features from a Quest joint JSON file.
#     """
#     with open(json_path, "r", encoding="utf-8") as file:
#         data = json.load(file)

    
#     return extract_quest_joint_features_from_joints(
#         data["joints"],
#         include_rotations=include_rotations,
#     )


# def extract_quest_joint_features_json(
#     raw_json,
#     include_rotations=False,
# ):
#     """
#     Extract normalized joint features from an in-memory Quest joint JSON object.
#     """
#     return extract_quest_joint_features_from_joints(
#         raw_json["joints"],
#         include_rotations=include_rotations,
#     )

# def feature_vector_to_joint_poses(
#     feature_vector: np.ndarray,
# ) -> dict[str, dict[str, Any]]:
#     """
#     Convert a flattened Quest joint feature vector into joint poses.

#     Supports either:

#         3 features per joint:
#             [x, y, z]

#         7 features per joint:
#             [x, y, z, qx, qy, qz, qw]

#     Returns
#     -------
#     dict
#         Mapping from joint name to:

#         {
#             "position": np.ndarray with shape (3,),
#             "rotation": np.ndarray with shape (4,) or None,
#         }
#     """
#     feature_vector = np.asarray(
#         feature_vector,
#         dtype=np.float32,
#     ).reshape(-1)

#     joint_count = len(QUEST_JOINT_ORDER)

#     position_only_size = joint_count * 3
#     position_rotation_size = joint_count * 7

#     if feature_vector.size == position_only_size:
#         features_per_joint = 3
#         rotations_included = False

#     elif feature_vector.size == position_rotation_size:
#         features_per_joint = 7
#         rotations_included = True

#     else:
#         raise ValueError(
#             "Unexpected feature-vector length. "
#             f"Expected {position_only_size} values for XYZ-only features "
#             f"or {position_rotation_size} values for XYZ plus quaternion "
#             f"features, but received {feature_vector.size}."
#         )

#     joint_poses = {}

#     for joint_index, joint_name in enumerate(
#         QUEST_JOINT_ORDER
#     ):
#         start = joint_index * features_per_joint

#         position = feature_vector[
#             start : start + 3
#         ].copy()

#         rotation = None

#         if rotations_included:
#             rotation = feature_vector[
#                 start + 3 : start + 7
#             ].copy()

#         joint_poses[joint_name] = {
#             "position": position,
#             "rotation": rotation,
#         }

#     return joint_poses

import json
import numpy as np
from typing import List,Tuple, Any 
import cv2

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


def quaternion_to_yaw_pitch_roll_degrees(quaternion):
    """
    Convert a quaternion in Unity order [x, y, z, w] to signed
    [yaw, pitch, roll] angles in degrees.

    The decomposition uses:
        yaw   about +Y
        pitch about +X
        roll  about +Z

    and corresponds to R = Ry(yaw) @ Rx(pitch) @ Rz(roll), which is
    consistent with Unity's Z-X-Y Euler application order. The returned
    angles are signed and centered around zero rather than represented in
    Unity's usual [0, 360) form.
    """
    x, y, z, w = normalize_quaternion(quaternion)

    # Quaternion -> rotation matrix.
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    # Y-X-Z decomposition. Clip before asin to avoid small numerical
    # excursions outside [-1, 1].
    pitch = np.arcsin(np.clip(-r12, -1.0, 1.0))
    cos_pitch = np.cos(pitch)

    if abs(cos_pitch) > 1e-6:
        yaw = np.arctan2(r02, r22)
        roll = np.arctan2(r10, r11)
    else:
        # At gimbal lock yaw and roll are not independently identifiable.
        # Choose roll = 0 and retain the combined horizontal rotation as yaw.
        yaw = np.arctan2(-r20, r00)
        roll = 0.0

    return np.rad2deg(
        np.asarray([yaw, pitch, roll], dtype=np.float32)
    ).astype(np.float32)


def extract_head_relative_to_chest_ypr(joints):
    """
    Return head orientation relative to the chest as [yaw, pitch, roll].

    The Quest joint quaternions are used only to compute the relative
    orientation robustly. The model-facing result remains three Euler-angle
    features in degrees.
    """
    chest_rotation = normalize_quaternion(
        joints["chest"]["rotation"]
    )
    head_rotation = normalize_quaternion(
        joints["head"]["rotation"]
    )

    head_relative_to_chest = quaternion_multiply(
        quaternion_conjugate(chest_rotation),
        head_rotation,
    )
    head_relative_to_chest = normalize_quaternion(
        head_relative_to_chest
    )

    return quaternion_to_yaw_pitch_roll_degrees(
        head_relative_to_chest
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

def draw_skeleton_connections(
    pose_img,
    pixel_x,
    pixel_y,
    valid,
    thickness=2,
    value=1.0,
):
    """Draw valid skeleton edges onto a single-channel pose image."""
    joint_indices = {
        joint_name: index
        for index, joint_name in enumerate(QUEST_JOINT_ORDER)
    }

    for start_name, end_name in SKELETON_EDGES:
        start_index = joint_indices[start_name]
        end_index = joint_indices[end_name]

        if not (valid[start_index] and valid[end_index]):
            continue

        start_point = (
            int(pixel_x[start_index]),
            int(pixel_y[start_index]),
        )
        end_point = (
            int(pixel_x[end_index]),
            int(pixel_y[end_index]),
        )

        cv2.line(
            pose_img,
            start_point,
            end_point,
            color=float(value),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

def construct_pose_img_from_joint_feature_vector(
    normalized_joints,
    im_w=640,
    im_h=640,
    margin_in_img=10,
    plane_extent=0.5,
):
    """
    Construct a pose image by projecting 3D joints onto the plane defined
    by the chest and shoulders. Returns the joints projected onto the chest-shoulder plane, as well as a 2D image of the projected joints and skeleton edges.

    plane_extent defines the represented distance from the chest in each
    plane direction: [-plane_extent, plane_extent].
    """
    eps = 1e-8

    joints = np.asarray(
        normalized_joints,
        dtype=np.float32,
    ).reshape(-1)

    joint_count = len(QUEST_JOINT_ORDER)
    position_only_size = joint_count * 3
    position_rotation_size = joint_count * 7
    head_orientation_size = 3

    # The final three values, when present, are the body-relative
    # [yaw, pitch, roll] head-orientation features. They are not joint XYZ
    # coordinates and should not be projected into the pose image.
    if joints.size in (
        position_only_size,
        position_only_size + head_orientation_size,
    ):
        positions = joints[:position_only_size].reshape(joint_count, 3)

    elif joints.size in (
        position_rotation_size,
        position_rotation_size + head_orientation_size,
    ):
        joint_features = joints[:position_rotation_size].reshape(
            joint_count,
            7,
        )
        positions = joint_features[:, :3]

    else:
        raise ValueError(
            "Unexpected joint feature-vector length for pose-image construction. "
            f"Received {joints.size} values."
        )

    if not np.all(np.isfinite(positions)):
        raise ValueError("Joint positions contain non-finite values.")

    if im_w <= 0 or im_h <= 0:
        raise ValueError("Image dimensions must be positive.")

    if margin_in_img < 0:
        raise ValueError("Image margin cannot be negative.")

    if 2 * margin_in_img >= min(im_w, im_h):
        raise ValueError("Image margin is too large.")

    if plane_extent <= 0:
        raise ValueError("Plane extent must be positive.")

    chest = positions[QUEST_JOINT_ORDER.index("chest")]
    right_shoulder = positions[
        QUEST_JOINT_ORDER.index("right_shoulder")
    ]
    left_shoulder = positions[
        QUEST_JOINT_ORDER.index("left_shoulder")
    ]

    normal_raw = np.cross(
        right_shoulder - chest,
        left_shoulder - chest,
    )
    normal_norm = np.linalg.norm(normal_raw)

    if normal_norm < eps:
        raise ValueError(
            "Chest and shoulders do not define a stable plane."
        )

    plane_normal = normal_raw / normal_norm

    u_raw = left_shoulder - right_shoulder
    u_norm = np.linalg.norm(u_raw)

    if u_norm < eps:
        raise ValueError("Shoulder positions are coincident.")

    u = u_raw / u_norm

    v = np.cross(plane_normal, u)
    v_norm = np.linalg.norm(v)

    if v_norm < eps:
        raise ValueError(
            "Could not construct the plane coordinate frame."
        )

    v /= v_norm

    # Keep positive v approximately aligned with world up.
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if np.dot(v, world_up) < 0:
        v = -v

    relative_positions = positions - chest

    proj_joints = np.column_stack(
        (
            relative_positions @ u,
            relative_positions @ v,
        )
    )

    c_x = (im_w - 1) / 2.0
    c_y = (im_h - 1) / 2.0

    x_scale = (c_x - margin_in_img) / plane_extent
    y_scale = (c_y - margin_in_img) / plane_extent

    # Use one scale to preserve geometric proportions.
    scale = min(x_scale, y_scale)

    pixel_x = np.rint(
        c_x + proj_joints[:, 0] * scale
    ).astype(np.int32)

    pixel_y = np.rint(
        c_y - proj_joints[:, 1] * scale
    ).astype(np.int32)

    valid = (
        (pixel_x >= 0)
        & (pixel_x < im_w)
        & (pixel_y >= 0)
        & (pixel_y < im_h)
    )

    pose_img = np.zeros((im_h, im_w), dtype=np.float32)

    draw_skeleton_connections(
        pose_img,
        pixel_x,
        pixel_y,
        valid,
        thickness=2,
    )

    for x, y in zip(pixel_x[valid], pixel_y[valid]):
        cv2.circle(
            pose_img,
            (int(x), int(y)),
            radius=3,
            color=1.0,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    pose_img_vis = np.clip(pose_img * 255.0, 0, 255).astype(np.uint8)

    return proj_joints.reshape(-1), pose_img_vis 

    



    
        
    


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

    feat_vec = extract_quest_joint_features_from_joints(
        data["joints"],
        include_rotations=include_rotations,
    )

    # Append three teacher-like head-orientation features, but make them
    # relative to the participant's chest rather than the Quest/world frame.
    # Output order is [yaw, pitch, roll], in signed degrees.
    head_ypr = extract_head_relative_to_chest_ypr(
        data["joints"]
    )

    return np.concatenate([feat_vec, head_ypr]).astype(np.float32)


def extract_quest_joint_features_json(
    raw_json,
    include_rotations=False,
):
    """
    Extract normalized joint features from an in-memory Quest joint JSON object.
    """
    feat_vec = extract_quest_joint_features_from_joints(
        raw_json["joints"],
        include_rotations=include_rotations,
    )

    head_ypr = extract_head_relative_to_chest_ypr(
        raw_json["joints"]
    )

    return np.concatenate([feat_vec, head_ypr]).astype(np.float32)

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
    head_orientation_size = 3

    # Feature vectors produced by the public extraction functions append
    # [yaw, pitch, roll]. Strip those values before reconstructing joints so
    # existing callers of this function do not need a new argument.
    if feature_vector.size in (
        position_only_size + head_orientation_size,
        position_rotation_size + head_orientation_size,
    ):
        feature_vector = feature_vector[:-head_orientation_size]

    if feature_vector.size == position_only_size:
        features_per_joint = 3
        rotations_included = False

    elif feature_vector.size == position_rotation_size:
        features_per_joint = 7
        rotations_included = True

    else:
        raise ValueError(
            "Unexpected feature-vector length. "
            f"Expected {position_only_size} or "
            f"{position_only_size + head_orientation_size} values for "
            "XYZ-only features, or "
            f"{position_rotation_size} or "
            f"{position_rotation_size + head_orientation_size} values for "
            "XYZ plus quaternion features, but received "
            f"{feature_vector.size}."
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