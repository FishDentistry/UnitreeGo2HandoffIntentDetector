from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Joint-name aliases
# ---------------------------------------------------------------------

HAND_JOINT_NAMES = {
    "left": (
        "left_hand",
    ),
    "right": (
        "right_hand",
    ),
}

SHOULDER_JOINT_NAMES = {
    "left": (
        "left_shoulder",
        "left_scapula",
    ),
    "right": (
        "right_shoulder",
        "right_scapula",
    ),
}

# Options for estimating the torso's upward direction, listed in
# preference order.
TORSO_UP_JOINT_PAIRS = (
    ("hips", "shoulder_midpoint"),
    ("spine_lower", "spine_upper"),
    ("spine_middle", "neck"),
    ("chest", "neck"),
    ("spine_lower", "chest"),
)


# ---------------------------------------------------------------------
# Position extraction and validation
# ---------------------------------------------------------------------

def validate_position(
    position: Any,
    name: str,
) -> np.ndarray:
    """
    Validate and return one XYZ joint position.

    Args:
        position:
            Array-like XYZ position.

        name:
            Name used in validation errors.

    Returns:
        A float32 NumPy array with shape (3,).

    Raises:
        ValueError:
            If the value is not a finite three-element position.
    """
    position_array = np.asarray(
        position,
        dtype=np.float32,
    ).squeeze()

    if position_array.shape != (3,):
        raise ValueError(
            f"{name} must be a three-element XYZ position, "
            f"but received shape {position_array.shape}."
        )

    if not np.all(np.isfinite(position_array)):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return position_array


def get_joint_position(
    joints: Mapping[str, Any],
    candidate_names: tuple[str, ...],
    *,
    required: bool = True,
    description: str | None = None,
) -> np.ndarray | None:
    """
    Retrieve a position from a named-joint dictionary.

    Supported joint representations:

        {
            "left_hand": {
                "position": [x, y, z]
            }
        }

    or:

        {
            "left_hand": [x, y, z]
        }

    Args:
        joints:
            Mapping from joint names to joint values.

        candidate_names:
            Joint names to try in order.

        required:
            Whether to raise an error if none are present.

        description:
            Human-readable description for error messages.

    Returns:
        The first matching joint position, or None when the joint is
        optional and unavailable.
    """
    for joint_name in candidate_names:
        if joint_name not in joints:
            continue

        joint_value = joints[joint_name]

        if isinstance(joint_value, Mapping):
            if "position" not in joint_value:
                raise ValueError(
                    f"Joint {joint_name!r} does not contain "
                    "a 'position' field."
                )

            joint_value = joint_value["position"]

        return validate_position(
            joint_value,
            joint_name,
        )

    if required:
        readable_description = (
            description
            if description is not None
            else "required joint"
        )

        raise KeyError(
            f"Could not find {readable_description}. "
            f"Expected one of: {', '.join(candidate_names)}."
        )

    return None


def get_named_joint_position(
    joints: Mapping[str, Any],
    joint_name: str,
    *,
    required: bool = True,
) -> np.ndarray | None:
    """
    Convenience wrapper for retrieving one exact joint name.
    """
    return get_joint_position(
        joints,
        (joint_name,),
        required=required,
        description=joint_name,
    )


# ---------------------------------------------------------------------
# Vector and body-frame utilities
# ---------------------------------------------------------------------

def normalize_vector(
    vector: np.ndarray,
    name: str,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Normalize a three-dimensional vector.
    """
    vector_array = validate_position(
        vector,
        name,
    )

    magnitude = float(
        np.linalg.norm(vector_array)
    )

    if magnitude <= epsilon:
        raise ValueError(
            f"{name} is zero or nearly zero."
        )

    return vector_array / magnitude


def remove_axis_component(
    vector: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """
    Remove the component of vector that is parallel to axis.
    """
    return (
        vector
        - float(np.dot(vector, axis))
        * axis
    )


def calculate_shoulder_midpoint(
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
) -> np.ndarray:
    """
    Calculate the midpoint between the shoulders.
    """
    return (
        left_shoulder + right_shoulder
    ) / 2.0


def estimate_torso_up_vector(
    original_joints: Mapping[str, Any],
    shoulder_midpoint: np.ndarray,
) -> np.ndarray:
    """
    Estimate the torso's upward direction from available joints.

    The preferred estimate is:

        shoulder midpoint - hips

    Fallbacks use lower-to-upper spine or chest-to-neck vectors.

    Returns:
        A non-normalized upward vector.

    Raises:
        KeyError:
            If none of the required torso joint combinations exist.
    """
    hips = get_named_joint_position(
        original_joints,
        "hips",
        required=False,
    )

    if hips is not None:
        return (
            shoulder_midpoint - hips
        )

    for lower_joint_name, upper_joint_name in (
        ("spine_lower", "spine_upper"),
        ("spine_middle", "neck"),
        ("chest", "neck"),
        ("spine_lower", "chest"),
    ):
        lower_joint = get_named_joint_position(
            original_joints,
            lower_joint_name,
            required=False,
        )
        upper_joint = get_named_joint_position(
            original_joints,
            upper_joint_name,
            required=False,
        )

        if (
            lower_joint is not None
            and upper_joint is not None
        ):
            return (
                upper_joint - lower_joint
            )

    raise KeyError(
        "Could not estimate the torso-up direction. "
        "Expected either 'hips', or one of these joint pairs: "
        "spine_lower/spine_upper, spine_middle/neck, "
        "chest/neck, or spine_lower/chest."
    )


def calculate_body_basis(
    original_joints: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Construct a person-relative orthonormal coordinate frame.

    The axes are:

        right:
            From the person's left shoulder toward their right shoulder.

        up:
            From the lower torso toward the upper torso.

        forward:
            Perpendicular to the shoulder and torso-up directions.

    Because the shoulder joints are anatomically labeled left and right,
    the cross product has a consistent forward sign:

        forward = cross(right, up)

    Args:
        original_joints:
            Original joint pose dictionary.

    Returns:
        right_axis:
            Positive projection means movement to the person's right.

        up_axis:
            Positive projection means upward movement.

        forward_axis:
            Positive projection means movement forward from the torso.

        shoulder_width:
            Distance between the original shoulder positions.
    """
    left_shoulder = get_joint_position(
        original_joints,
        SHOULDER_JOINT_NAMES["left"],
        description="the left shoulder",
    )
    right_shoulder = get_joint_position(
        original_joints,
        SHOULDER_JOINT_NAMES["right"],
        description="the right shoulder",
    )

    assert left_shoulder is not None
    assert right_shoulder is not None

    shoulder_vector = (
        right_shoulder - left_shoulder
    )

    shoulder_width = float(
        np.linalg.norm(shoulder_vector)
    )

    if not np.isfinite(shoulder_width):
        raise ValueError(
            "Calculated shoulder width is not finite."
        )

    if shoulder_width <= 1e-6:
        raise ValueError(
            "The shoulder positions do not define a valid "
            "left-to-right body axis."
        )

    right_axis = normalize_vector(
        shoulder_vector,
        "body right axis",
    )

    shoulder_midpoint = (
        calculate_shoulder_midpoint(
            left_shoulder,
            right_shoulder,
        )
    )

    raw_up_axis = estimate_torso_up_vector(
        original_joints,
        shoulder_midpoint,
    )

    # Remove any lateral component from the estimated up vector so
    # the right and up axes are perpendicular.
    orthogonal_up_axis = remove_axis_component(
        raw_up_axis,
        right_axis,
    )

    up_axis = normalize_vector(
        orthogonal_up_axis,
        "body up axis",
    )

    # With anatomical left/right shoulder labels:
    #
    #     right × up = forward
    #
    # For a person facing Unity +Z:
    #
    #     [1, 0, 0] × [0, 1, 0] = [0, 0, 1]
    forward_axis = normalize_vector(
        np.cross(
            right_axis,
            up_axis,
        ),
        "body forward axis",
    )

    # Recompute up from forward and right to make the basis exactly
    # orthonormal after floating-point operations.
    up_axis = normalize_vector(
        np.cross(
            forward_axis,
            right_axis,
        ),
        "orthogonal body up axis",
    )

    return (
        right_axis,
        up_axis,
        forward_axis,
        shoulder_width,
    )


# ---------------------------------------------------------------------
# Movement-description utilities
# ---------------------------------------------------------------------

def describe_component_amount(
    component_distance: float,
    shoulder_width: float | None = None,
    *,
    ignore_below_m: float = 0.015,
) -> str | None:
    """
    Convert one movement component into qualitative language.

    The values are assumed to be in meters.
    """
    distance = abs(
        float(component_distance)
    )

    if distance < ignore_below_m:
        return None

    if distance < 0.04:
        return "very slightly"

    if distance < 0.08:
        return "a little"

    if distance < 0.15:
        return "a moderate amount"

    return "a lot"


def join_phrases(
    phrases: list[str],
) -> str:
    """
    Join up to three directional movement phrases.
    """
    if not phrases:
        return ""

    if len(phrases) == 1:
        return phrases[0]

    if len(phrases) == 2:
        return (
            f"{phrases[0]} and {phrases[1]}"
        )

    return (
        f"{phrases[0]}, "
        f"{phrases[1]}, "
        f"and {phrases[2]}"
    )


def get_body_relative_movement_components(
    original_position: np.ndarray,
    perturbed_position: np.ndarray,
    *,
    right_axis: np.ndarray,
    up_axis: np.ndarray,
    forward_axis: np.ndarray,
) -> dict[str, float]:
    """
    Project the original-to-perturbed movement line onto the body axes.

    Returns:
        A dictionary containing signed right, up, and forward
        components.
    """
    original_position = validate_position(
        original_position,
        "original hand position",
    )
    perturbed_position = validate_position(
        perturbed_position,
        "perturbed hand position",
    )

    movement_vector = (
        perturbed_position
        - original_position
    )

    return {
        "right": float(
            np.dot(
                movement_vector,
                right_axis,
            )
        ),
        "up": float(
            np.dot(
                movement_vector,
                up_axis,
            )
        ),
        "forward": float(
            np.dot(
                movement_vector,
                forward_axis,
            )
        ),
    }


def describe_hand_movement(
    *,
    hand_name: str,
    original_position: np.ndarray,
    perturbed_position: np.ndarray,
    right_axis: np.ndarray,
    up_axis: np.ndarray,
    forward_axis: np.ndarray,
    shoulder_width: float,
    debug: bool = False,
) -> tuple[float, str] | None:
    """
    Describe the complete 3D line from the original hand position to
    the perturbed hand position.

    All meaningful dimensions are included:

        right or left
        upward or downward
        forward or backward

    Args:
        hand_name:
            Either "left" or "right".

        original_position:
            Original hand position.

        perturbed_position:
            Target hand position.

        right_axis:
            Person-relative right direction.

        up_axis:
            Person-relative upward direction.

        forward_axis:
            Person-relative forward direction.

        shoulder_width:
            Used to normalize qualitative movement amounts.

        debug:
            Print raw and body-relative movement values.

    Returns:
        Tuple of:
            total movement distance,
            generated instruction

        Returns None if the movement is too small to describe.
    """
    original_position = validate_position(
        original_position,
        f"original {hand_name} hand",
    )
    perturbed_position = validate_position(
        perturbed_position,
        f"perturbed {hand_name} hand",
    )

    movement_vector = (
        perturbed_position
        - original_position
    )

    movement_distance = float(
        np.linalg.norm(movement_vector)
    )

    if movement_distance <= 1e-8:
        return None

    components = (
        get_body_relative_movement_components(
            original_position,
            perturbed_position,
            right_axis=right_axis,
            up_axis=up_axis,
            forward_axis=forward_axis,
        )
    )

    if debug:
        print()
        print(
            f"{hand_name.upper()} HAND GUIDANCE DEBUG"
        )
        print(
            f"  original:  {original_position}"
        )
        print(
            f"  perturbed: {perturbed_position}"
        )
        print(
            f"  raw delta: {movement_vector}"
        )
        print(
            "  body-relative components:"
        )
        print(
            f"    right:   {components['right']:+.6f}"
        )
        print(
            f"    up:      {components['up']:+.6f}"
        )
        print(
            f"    forward: {components['forward']:+.6f}"
        )

    phrase_candidates: list[
        tuple[float, str]
    ] = []

    right_component = components["right"]

    right_amount = describe_component_amount(
        right_component,
        shoulder_width,
    )

    if right_amount is not None:
        right_direction = (
            "to your right"
            if right_component > 0.0
            else "to your left"
        )

        phrase_candidates.append(
            (
                abs(right_component),
                f"{right_direction} {right_amount}",
            )
        )

    up_component = components["up"]

    up_amount = describe_component_amount(
        up_component,
        shoulder_width,
    )

    if up_amount is not None:
        up_direction = (
            "upward"
            if up_component > 0.0
            else "downward"
        )

        phrase_candidates.append(
            (
                abs(up_component),
                f"{up_direction} {up_amount}",
            )
        )

    forward_component = components["forward"]

    forward_amount = describe_component_amount(
        forward_component,
        shoulder_width,
    )

    if forward_amount is not None:
        forward_direction_text = (
            "forward"
            if forward_component > 0.0
            else "backward"
        )

        phrase_candidates.append(
            (
                abs(forward_component),
                (
                    f"{forward_direction_text} "
                    f"{forward_amount}"
                ),
            )
        )

    if not phrase_candidates:
        return None

    # Mention the largest line component first, but retain every
    # meaningful component.
    phrase_candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    movement_phrases = [
        phrase
        for _, phrase in phrase_candidates
    ]

    if len(movement_phrases) == 1:
        instruction = (
            f"Move your {hand_name} hand "
            f"{movement_phrases[0]}."
        )
    else:
        instruction = (
            f"Move your {hand_name} hand: "
            f"{join_phrases(movement_phrases)}."
        )

    if debug:
        print(
            f"  instruction: {instruction}"
        )

    return (
        movement_distance,
        instruction,
    )


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def generate_text_guidance_from_perturbation(
    original_joints: Mapping[str, Any],
    perturbed_joints: Mapping[str, Any],
    *,
    debug: bool = False,
) -> str:
    """
    Generate hand guidance from original and counterfactual joint poses.

    The original and perturbed hand positions define a 3D movement line.
    That line is projected onto a body-relative coordinate frame derived
    from the original skeleton.

    The generated text includes every meaningful component of the line:

        left or right
        upward or downward
        forward or backward

    The positions may remain shoulder-centered. Adding the shoulder
    midpoint back is unnecessary because the same translation would
    cancel when calculating:

        perturbed_position - original_position

    Args:
        original_joints:
            Original named joint-pose dictionary.

        perturbed_joints:
            Perturbed named joint-pose dictionary.

        debug:
            Print body axes, hand positions, movement lines, and
            projected components.

    Returns:
        Text describing how one or both hands should move.
    """
    (
        right_axis,
        up_axis,
        forward_axis,
        shoulder_width,
    ) = calculate_body_basis(
        original_joints
    )

    if debug:
        print()
        print("BODY BASIS DEBUG")
        print(
            f"  right axis:   {right_axis}"
        )
        print(
            f"  up axis:      {up_axis}"
        )
        print(
            f"  forward axis: {forward_axis}"
        )
        print(
            f"  shoulder width: {shoulder_width:.6f}"
        )

    generated_guidance: list[
        tuple[float, str]
    ] = []

    for hand_name in ("right", "left"):
        original_hand = get_joint_position(
            original_joints,
            HAND_JOINT_NAMES[hand_name],
            description=(
                f"the original {hand_name} hand"
            ),
        )
        perturbed_hand = get_joint_position(
            perturbed_joints,
            HAND_JOINT_NAMES[hand_name],
            description=(
                f"the perturbed {hand_name} hand"
            ),
        )

        assert original_hand is not None
        assert perturbed_hand is not None

        hand_guidance = describe_hand_movement(
            hand_name=hand_name,
            original_position=original_hand,
            perturbed_position=perturbed_hand,
            right_axis=right_axis,
            up_axis=up_axis,
            forward_axis=forward_axis,
            shoulder_width=shoulder_width,
            debug=debug,
        )

        if hand_guidance is not None:
            generated_guidance.append(
                hand_guidance
            )

    if not generated_guidance:
        return (
            "Keep your hands in their current positions."
        )

    # Describe the hand with the longer overall movement first.
    generated_guidance.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return " ".join(
        instruction
        for _, instruction
        in generated_guidance
    )