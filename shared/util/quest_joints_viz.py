from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from quest_hand_intent_model_est.src.quest_joint_features import (
    QUEST_JOINT_ORDER,
    SKELETON_EDGES,
)


def feature_vector_to_joint_positions(
    feature_vector: np.ndarray,
    features_per_joint: int = 3,
) -> dict[str, np.ndarray]:
    """
    Extract XYZ joint positions from a flattened feature vector.

    Parameters
    ----------
    feature_vector:
        Flattened feature vector.

    features_per_joint:
        Use 3 for XYZ-only features.

        Use 7 when each joint contains:
            [x, y, z, quaternion_x, quaternion_y,
             quaternion_z, quaternion_w]

        Only the first three values for each joint are visualized.
    """
    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32,
    ).reshape(-1)

    if features_per_joint < 3:
        raise ValueError(
            "features_per_joint must be at least 3."
        )

    expected_size = (
        len(QUEST_JOINT_ORDER)
        * features_per_joint
    )

    if feature_vector.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} features "
            f"({len(QUEST_JOINT_ORDER)} joints × "
            f"{features_per_joint} features per joint), "
            f"but received {feature_vector.size}."
        )

    joints: dict[str, np.ndarray] = {}

    for joint_index, joint_name in enumerate(
        QUEST_JOINT_ORDER
    ):
        start = joint_index * features_per_joint

        joints[joint_name] = feature_vector[
            start : start + 3
        ].astype(np.float64)

    return joints


def unity_to_plotly_coordinates(
    joints: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Convert Unity coordinates to display coordinates.

    Unity:
        X = right
        Y = up
        Z = forward

    Plot:
        X = right
        Y = forward
        Z = up
    """
    return {
        joint_name: np.asarray(
            [
                position[0],
                position[2],
                position[1],
            ],
            dtype=np.float64,
        )
        for joint_name, position in joints.items()
    }


def add_skeleton_to_figure(
    figure: go.Figure,
    joints: dict[str, np.ndarray],
    *,
    name: str,
    line_color: str,
    marker_color: str,
    visible: bool = True,
    show_labels: bool = True,
) -> None:
    """
    Add one labeled skeleton to a Plotly figure.
    """
    joints = unity_to_plotly_coordinates(joints)

    line_x: list[Optional[float]] = []
    line_y: list[Optional[float]] = []
    line_z: list[Optional[float]] = []

    for joint_a, joint_b in SKELETON_EDGES:
        if joint_a not in joints or joint_b not in joints:
            continue

        point_a = joints[joint_a]
        point_b = joints[joint_b]

        line_x.extend(
            [point_a[0], point_b[0], None]
        )
        line_y.extend(
            [point_a[1], point_b[1], None]
        )
        line_z.extend(
            [point_a[2], point_b[2], None]
        )

    # Skeleton bones.
    figure.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            name=f"{name} bones",
            legendgroup=name,
            visible=visible,
            line={
                "color": line_color,
                "width": 6,
            },
            hoverinfo="skip",
        )
    )

    ordered_names = [
        joint_name
        for joint_name in QUEST_JOINT_ORDER
        if joint_name in joints
    ]

    positions = np.asarray(
        [
            joints[joint_name]
            for joint_name in ordered_names
        ],
        dtype=np.float64,
    )

    hover_text = [
        (
            f"{name}: {joint_name}"
            f"<br>X: {position[0]:.4f}"
            f"<br>Y: {position[1]:.4f}"
            f"<br>Z: {position[2]:.4f}"
        )
        for joint_name, position in zip(
            ordered_names,
            positions,
        )
    ]

    trace_mode = (
        "markers+text"
        if show_labels
        else "markers"
    )

    label_text = (
        ordered_names
        if show_labels
        else None
    )

    # Skeleton joints and labels.
    figure.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode=trace_mode,
            name=f"{name} joints",
            legendgroup=name,
            visible=visible,
            marker={
                "color": marker_color,
                "size": 5,
            },
            text=label_text,
            textposition="top center",
            textfont={
                "color": marker_color,
                "size": 10,
            },
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )


def visualize_joint_features(
    feature_vector: np.ndarray,
    *,
    comparison_feature_vector: np.ndarray | None = None,
    features_per_joint: int = 3,
    title: str = "Quest joint visualization",
    output_path: str | Path = "joint_visualization.html",
    show_labels: bool = True,
) -> Path:
    """
    Create an interactive HTML visualization of one or two poses.

    When comparison_feature_vector is supplied, both poses are overlaid.

    Original joints and labels are blue.
    Perturbed joints and labels are red.

    The legend can be used to hide or show either pose.
    """
    original_joints = feature_vector_to_joint_positions(
        feature_vector,
        features_per_joint=features_per_joint,
    )

    figure = go.Figure()

    add_skeleton_to_figure(
        figure,
        original_joints,
        name="Original",
        line_color="#2563eb",
        marker_color="#1d4ed8",
        show_labels=show_labels,
    )

    if comparison_feature_vector is not None:
        comparison_joints = (
            feature_vector_to_joint_positions(
                comparison_feature_vector,
                features_per_joint=features_per_joint,
            )
        )

        add_skeleton_to_figure(
            figure,
            comparison_joints,
            name="Perturbed",
            line_color="#dc2626",
            marker_color="#b91c1c",
            show_labels=show_labels,
        )

    figure.update_layout(
        title=title,
        template="plotly_white",
        margin={
            "l": 0,
            "r": 0,
            "t": 60,
            "b": 0,
        },
        legend={
            "groupclick": "togglegroup",
        },
        scene={
            "aspectmode": "data",
            "xaxis": {
                "title": "X — right",
            },
            "yaxis": {
                "title": "Y — forward",
            },
            "zaxis": {
                "title": "Z — up",
            },
            "camera": {
                "eye": {
                    "x": 1.5,
                    "y": -1.8,
                    "z": 1.0,
                },
            },
        },
    )

    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
        },
    )

    print(
        f"Visualization written to: {output_path}"
    )

    return output_path