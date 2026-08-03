from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from text2pose.posefix.correcting import (
    main as run_posefix_pipeline,

)
from text2pose.posefix.paircodes import (
    PAIRCODE_OPERATORS,
)
from text2pose.posescript.posecodes import (
    POSECODE_OPERATORS,
)
import text2pose.posefix.correcting as posefix_correcting



# PoseFix accepts either the complete 52-joint SMPL-H skeleton or
# the first 22 joints in this exact order.
POSEFIX_SMPL22_JOINT_ORDER = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


def _validate_position(
    value: Any,
    name: str,
) -> np.ndarray:
    position = np.asarray(
        value,
        dtype=np.float32,
    ).squeeze()

    if position.shape != (3,):
        raise ValueError(
            f"{name} must contain exactly three position values. "
            f"Received shape {position.shape}."
        )

    if not np.all(np.isfinite(position)):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return position


def _read_position(
    joints: Mapping[str, Any],
    joint_name: str,
) -> np.ndarray:
    if joint_name not in joints:
        raise KeyError(
            f"Missing joint {joint_name!r}."
        )

    joint_value = joints[joint_name]

    if isinstance(joint_value, Mapping):
        if "position" not in joint_value:
            raise KeyError(
                f"Joint {joint_name!r} has no 'position' field."
            )

        joint_value = joint_value["position"]

    return _validate_position(
        joint_value,
        joint_name,
    )


def _read_first_available_position(
    joints: Mapping[str, Any],
    candidate_names: tuple[str, ...],
    description: str,
) -> np.ndarray:
    for joint_name in candidate_names:
        if joint_name in joints:
            return _read_position(
                joints,
                joint_name,
            )

    raise KeyError(
        f"Could not locate {description}. "
        f"Tried: {', '.join(candidate_names)}."
    )


def _normalize(
    vector: np.ndarray,
    name: str,
    epsilon: float = 1e-8,
) -> np.ndarray:
    vector = _validate_position(
        vector,
        name,
    )

    magnitude = float(
        np.linalg.norm(vector)
    )

    if magnitude <= epsilon:
        raise ValueError(
            f"{name} is zero or nearly zero."
        )

    return vector / magnitude


def _calculate_posefix_basis(
    original_joints: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate the canonical coordinate frame expected by PoseFix.

    PoseFix convention:
        +X: person's left
        +Y: upward
        +Z: person's forward

    Returns:
        origin
        left_axis
        up_axis
        forward_axis
    """
    left_arm_root = _read_first_available_position(
        original_joints,
        (
            "left_upper_arm",
            "left_shoulder",
            "left_scapula",
        ),
        "the left shoulder/upper-arm root",
    )

    right_arm_root = _read_first_available_position(
        original_joints,
        (
            "right_upper_arm",
            "right_shoulder",
            "right_scapula",
        ),
        "the right shoulder/upper-arm root",
    )

    shoulder_midpoint = (
        left_arm_root + right_arm_root
    ) / 2.0

    # Anatomical right points from the person's left shoulder
    # toward their right shoulder.
    right_axis = _normalize(
        right_arm_root - left_arm_root,
        "body-right axis",
    )

    if "hips" in original_joints:
        hips = _read_position(
            original_joints,
            "hips",
        )

        raw_up_axis = (
            shoulder_midpoint - hips
        )
    else:
        lower_spine = _read_first_available_position(
            original_joints,
            (
                "spine_lower",
                "spine_middle",
            ),
            "a lower-spine joint",
        )

        neck = _read_first_available_position(
            original_joints,
            (
                "neck",
                "spine_upper",
                "chest",
            ),
            "an upper torso joint",
        )

        raw_up_axis = neck - lower_spine

    # Remove any sideways component to make up perpendicular
    # to the shoulder line.
    raw_up_axis = (
        raw_up_axis
        - np.dot(
            raw_up_axis,
            right_axis,
        )
        * right_axis
    )

    up_axis = _normalize(
        raw_up_axis,
        "body-up axis",
    )

    # Unity is right-handed here:
    #
    # body right × body up = body forward
    forward_axis = _normalize(
        np.cross(
            right_axis,
            up_axis,
        ),
        "body-forward axis",
    )

    # PoseFix's positive X direction is the person's left.
    left_axis = -right_axis

    return (
        shoulder_midpoint,
        left_axis,
        up_axis,
        forward_axis,
    )


def _transform_to_posefix_frame(
    world_position: np.ndarray,
    *,
    origin: np.ndarray,
    left_axis: np.ndarray,
    up_axis: np.ndarray,
    forward_axis: np.ndarray,
) -> np.ndarray:
    """
    Convert a Quest/Unity position into PoseFix's canonical frame.
    """
    relative_position = (
        world_position - origin
    )

    return np.array(
        [
            np.dot(
                relative_position,
                left_axis,
            ),
            np.dot(
                relative_position,
                up_axis,
            ),
            np.dot(
                relative_position,
                forward_axis,
            ),
        ],
        dtype=np.float32,
    )


def _canonicalize_joint_dictionary(
    joints: Mapping[str, Any],
    *,
    origin: np.ndarray,
    left_axis: np.ndarray,
    up_axis: np.ndarray,
    forward_axis: np.ndarray,
) -> dict[str, np.ndarray]:
    canonical_joints: dict[
        str,
        np.ndarray,
    ] = {}

    for joint_name, joint_value in joints.items():
        if isinstance(joint_value, Mapping):
            if "position" not in joint_value:
                continue

            position_value = joint_value[
                "position"
            ]
        else:
            position_value = joint_value

        try:
            position = _validate_position(
                position_value,
                joint_name,
            )
        except (TypeError, ValueError):
            continue

        canonical_joints[joint_name] = (
            _transform_to_posefix_frame(
                position,
                origin=origin,
                left_axis=left_axis,
                up_axis=up_axis,
                forward_axis=forward_axis,
            )
        )

    return canonical_joints


def _canonical_position(
    joints: Mapping[str, np.ndarray],
    candidate_names: tuple[str, ...],
    description: str,
) -> np.ndarray:
    for joint_name in candidate_names:
        if joint_name in joints:
            return joints[joint_name].copy()

    raise KeyError(
        f"Could not locate {description}. "
        f"Tried: {', '.join(candidate_names)}."
    )


def _create_fixed_lower_body(
    original_canonical_joints: Mapping[
        str,
        np.ndarray,
    ],
) -> dict[str, np.ndarray]:
    """
    Create a simple lower-body scaffold.

    Your Quest pipeline is primarily upper-body based. PoseFix nevertheless
    requires 22 SMPL-H body joints. The same fixed lower-body positions are
    supplied to both the original and target poses so they cannot generate
    false lower-body correction instructions.
    """
    pelvis = _canonical_position(
        original_canonical_joints,
        (
            "hips",
            "root",
            "spine_lower",
        ),
        "the pelvis",
    )

    return {
        "pelvis": pelvis,
        "left_hip": (
            pelvis
            + np.array(
                [0.10, -0.05, 0.0],
                dtype=np.float32,
            )
        ),
        "right_hip": (
            pelvis
            + np.array(
                [-0.10, -0.05, 0.0],
                dtype=np.float32,
            )
        ),
        "left_knee": (
            pelvis
            + np.array(
                [0.10, -0.45, 0.0],
                dtype=np.float32,
            )
        ),
        "right_knee": (
            pelvis
            + np.array(
                [-0.10, -0.45, 0.0],
                dtype=np.float32,
            )
        ),
        "left_ankle": (
            pelvis
            + np.array(
                [0.10, -0.90, 0.0],
                dtype=np.float32,
            )
        ),
        "right_ankle": (
            pelvis
            + np.array(
                [-0.10, -0.90, 0.0],
                dtype=np.float32,
            )
        ),
        "left_foot": (
            pelvis
            + np.array(
                [0.10, -0.95, 0.12],
                dtype=np.float32,
            )
        ),
        "right_foot": (
            pelvis
            + np.array(
                [-0.10, -0.95, 0.12],
                dtype=np.float32,
            )
        ),
    }


def _quest_pose_to_smpl22(
    canonical_joints: Mapping[
        str,
        np.ndarray,
    ],
    *,
    fixed_lower_body: Mapping[
        str,
        np.ndarray,
    ],
) -> np.ndarray:
    """
    Convert one canonical Quest upper-body pose into PoseFix's
    22-joint SMPL-H layout.
    """
    pelvis = fixed_lower_body[
        "pelvis"
    ]

    spine1 = _canonical_position(
        canonical_joints,
        (
            "spine_lower",
            "root",
            "hips",
        ),
        "spine1",
    )

    spine2 = _canonical_position(
        canonical_joints,
        (
            "spine_middle",
            "spine_lower",
        ),
        "spine2",
    )

    spine3 = _canonical_position(
        canonical_joints,
        (
            "spine_upper",
            "chest",
            "spine_middle",
        ),
        "spine3",
    )

    neck = _canonical_position(
        canonical_joints,
        (
            "neck",
            "chest",
            "spine_upper",
        ),
        "the neck",
    )

    head = _canonical_position(
        canonical_joints,
        (
            "head",
            "hmd_center_eye",
            "neck",
        ),
        "the head",
    )

    left_collar = _canonical_position(
        canonical_joints,
        (
            "left_scapula",
            "left_shoulder",
            "left_upper_arm",
        ),
        "the left collar",
    )

    right_collar = _canonical_position(
        canonical_joints,
        (
            "right_scapula",
            "right_shoulder",
            "right_upper_arm",
        ),
        "the right collar",
    )

    # These mappings follow the arm chain used by your Quest skeleton:
    #
    # upper_arm -> forearm -> hand
    #
    # Verify them once by plotting the resulting SMPL22 skeleton.
    left_shoulder = _canonical_position(
        canonical_joints,
        (
            "left_upper_arm",
            "left_shoulder",
        ),
        "the left shoulder joint",
    )

    right_shoulder = _canonical_position(
        canonical_joints,
        (
            "right_upper_arm",
            "right_shoulder",
        ),
        "the right shoulder joint",
    )

    left_elbow = _canonical_position(
        canonical_joints,
        (
            "left_forearm",
            "left_upper_arm",
        ),
        "the left elbow joint",
    )

    right_elbow = _canonical_position(
        canonical_joints,
        (
            "right_forearm",
            "right_upper_arm",
        ),
        "the right elbow joint",
    )

    left_wrist = _canonical_position(
        canonical_joints,
        (
            "left_hand",
            "left_wrist_twist",
            "left_palm",
        ),
        "the left wrist/hand endpoint",
    )

    right_wrist = _canonical_position(
        canonical_joints,
        (
            "right_hand",
            "right_wrist_twist",
            "right_palm",
        ),
        "the right wrist/hand endpoint",
    )

    smpl22 = np.stack(
        (
            pelvis,
            fixed_lower_body["left_hip"],
            fixed_lower_body["right_hip"],
            spine1,
            fixed_lower_body["left_knee"],
            fixed_lower_body["right_knee"],
            spine2,
            fixed_lower_body["left_ankle"],
            fixed_lower_body["right_ankle"],
            spine3,
            fixed_lower_body["left_foot"],
            fixed_lower_body["right_foot"],
            neck,
            left_collar,
            right_collar,
            head,
            left_shoulder,
            right_shoulder,
            left_elbow,
            right_elbow,
            left_wrist,
            right_wrist,
        ),
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    if smpl22.shape != (22, 3):
        raise RuntimeError(
            "Internal error: generated SMPL pose does not "
            f"have shape (22, 3): {smpl22.shape}."
        )

    return smpl22


def quest_pose_pair_to_posefix_coords(
    original_joints: Mapping[str, Any],
    perturbed_joints: Mapping[str, Any],
) -> torch.Tensor:
    """
    Convert the original and target Quest pose dictionaries into
    a PoseFix coordinate tensor with shape (2, 22, 3).
    """
    (
        origin,
        left_axis,
        up_axis,
        forward_axis,
    ) = _calculate_posefix_basis(
        original_joints
    )

    original_canonical = (
        _canonicalize_joint_dictionary(
            original_joints,
            origin=origin,
            left_axis=left_axis,
            up_axis=up_axis,
            forward_axis=forward_axis,
        )
    )

    # Use the original pose's basis for both poses. Otherwise, rotating
    # each pose independently could erase or distort the actual change.
    perturbed_canonical = (
        _canonicalize_joint_dictionary(
            perturbed_joints,
            origin=origin,
            left_axis=left_axis,
            up_axis=up_axis,
            forward_axis=forward_axis,
        )
    )

    fixed_lower_body = (
        _create_fixed_lower_body(
            original_canonical
        )
    )

    original_smpl22 = (
        _quest_pose_to_smpl22(
            original_canonical,
            fixed_lower_body=fixed_lower_body,
        )
    )

    perturbed_smpl22 = (
        _quest_pose_to_smpl22(
            perturbed_canonical,
            fixed_lower_body=fixed_lower_body,
        )
    )

    return torch.from_numpy(
        np.stack(
            (
                original_smpl22,
                perturbed_smpl22,
            ),
            axis=0,
        )
    )


def configure_posefix_for_small_movements(
    *,
    deadzone_m: float = 0.02,
    strong_movement_m: float = 0.10,
) -> None:
    """
    Adapt PoseFix's existing paircode bins to the small counterfactual
    movements produced by your optimizer.

    This does not replace PoseFix's paircodes or templates. It only changes
    the numeric category boundaries used by the official implementation.
    """
    if deadzone_m <= 0.0:
        raise ValueError(
            "deadzone_m must be greater than zero."
        )

    if strong_movement_m <= deadzone_m:
        raise ValueError(
            "strong_movement_m must exceed deadzone_m."
        )

    # Remove the built-in random threshold variation.
    for operator in PAIRCODE_OPERATORS.values():
        operator.random_max_offset = 0.0

    for operator in POSECODE_OPERATORS.values():
        operator.random_max_offset = 0.0

    movement_thresholds = [
        -strong_movement_m,
        -deadzone_m,
        deadzone_m,
        strong_movement_m,
    ]

    for operator_name in (
        "pair_relativePosX",
        "pair_relativePosY",
        "pair_relativePosZ",
    ):
        PAIRCODE_OPERATORS[
            operator_name
        ].category_thresholds = (
            movement_thresholds.copy()
        )

    # Use the same scale for changes in hand-to-torso or
    # hand-to-hand distance.
    PAIRCODE_OPERATORS[
        "pair_distance"
    ].category_thresholds = (
        movement_thresholds.copy()
    )

def configure_posefix_for_hand_guidance(
    *,
    deadzone_m: float = 0.02,
    strong_movement_m: float = 0.10,
) -> None:

    posefix_correcting.PAIR_PROP_AGGREGATION_HAPPENS = 1.0
    posefix_correcting.PROP_SKIP_PAIRCODES = 0.35

    posefix_correcting.DETERMINERS = ["your"]
    posefix_correcting.DETERMINERS_PROP = [1.0]

    configure_posefix_for_small_movements(
        deadzone_m=deadzone_m,
        strong_movement_m=strong_movement_m,
    )


def generate_posefix_guidance(
    original_joints: Mapping[str, Any],
    perturbed_joints: Mapping[str, Any],
    *,
    simplified_instructions: bool = True,
    seed: int = 0,
) -> str:
    """
    Generate correctional text with the official PoseFix comparative
    pipeline.
    """
    coordinates = (
        quest_pose_pair_to_posefix_coords(
            original_joints,
            perturbed_joints,
        )
    )

    pose_pairs = torch.tensor(
        [[0, 1]],
        dtype=torch.long,
    )

    # PoseFix intentionally randomizes some template and aggregation choices.
    # Resetting all seeds makes repeated server calls reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with torch.inference_mode():
        instructions = run_posefix_pipeline(
            pose_pairs=pose_pairs,
            coords=coordinates,
            global_rotation_change=None,
            joint_rotations_type="smplh",
            joint_rotations=None,
            use_contact_codes=False,
            add_description_text_pieces=False,
            save_dir=None,
            simplified_instructions=(
                simplified_instructions
            ),
            random_skip=False,
            verbose=False,
            ret_type="list",
        )

    if not instructions:
        return (
            "Keep your hands in their current positions."
        )

    instruction = str(
        instructions[0]
    ).strip()

    if instruction in (
        "",
        ".",
    ):
        return (
            "Keep your hands in their current positions."
        )

    return instruction