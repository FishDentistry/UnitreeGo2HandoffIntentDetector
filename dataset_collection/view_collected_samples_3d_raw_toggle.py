#!/usr/bin/env python3
"""
VERSION: 3D_RAW_TOGGLE_2026_07_01

Scroll through collected handoff-intent samples with two separate windows:

  1. OpenCV window: RGB image with metadata overlaid.
  2. Matplotlib window: corresponding Quest joints drawn in true 3D.

The 3D joint window can toggle between:
  - head-normalized mode: joint_position - head_position
  - raw mode: original Quest/Unity joint positions from joints_raw.json

Expected dataset layout:
  handoff_dataset/samples/<participant_id>/<label_name>/<condition>/<hand>/<sample_id>/
      rgb.png
      depth_z16.png
      joints_raw.json
      meta.json

Run:
  python view_collected_samples_3d_raw_toggle.py

Examples:
  python view_collected_samples_3d_raw_toggle.py --dataset-root handoff_dataset
  python view_collected_samples_3d_raw_toggle.py --participant P01
  python view_collected_samples_3d_raw_toggle.py --joint-mode raw
  python view_collected_samples_3d_raw_toggle.py --show-joint-labels

Controls while the RGB OpenCV window is focused:
  d / n / right / space : next sample
  a / p / left          : previous sample
  j / m                 : toggle raw vs head-normalized joints
  l                     : toggle joint labels
  r                     : reset 3D view limits/view angle
  q / Esc               : quit

Use the Matplotlib toolbar/mouse in the 3D window to rotate, pan, and zoom.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - enables 3D projection on older installs

from util.extract_all_samples import discover_samples, sample_root_from_dataset_root, read_json, Sample


VERSION = "3D_RAW_TOGGLE_2026_07_01"
IMAGE_WINDOW = "RGB sample + metadata"

# Edges are based on the joint names produced by the Quest packet in your server.
SKELETON_EDGES: List[Tuple[str, str]] = [
    ("hips", "spine_lower"),
    ("spine_lower", "spine_middle"),
    ("spine_middle", "spine_upper"),
    ("spine_upper", "chest"),
    ("chest", "neck"),
    ("neck", "head"),
    ("head", "hmd_center_eye"),
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

PREFERRED_JOINT_ORDER: List[str] = [
    "hmd_center_eye",
    "head",
    "neck",
    "chest",
    "spine_upper",
    "spine_middle",
    "spine_lower",
    "hips",
    "left_scapula",
    "left_shoulder",
    "left_upper_arm",
    "left_forearm",
    "left_wrist_twist",
    "left_hand",
    "left_palm",
    "right_scapula",
    "right_shoulder",
    "right_upper_arm",
    "right_forearm",
    "right_wrist_twist",
    "right_hand",
    "right_palm",
]




class ViewerState:
    def __init__(self, total: int, start_index: int, joint_mode: str, show_labels: bool):
        self.total = total
        self.index = min(max(start_index, 0), max(total - 1, 0))
        self.joint_mode = joint_mode
        self.show_labels = show_labels
        self.needs_redraw = True
        self.reset_3d_view = True
        self.quit = False

    def move(self, delta: int) -> None:
        self.index = (self.index + delta) % self.total
        self.needs_redraw = True

    def toggle_joint_mode(self) -> None:
        self.joint_mode = "raw" if self.joint_mode == "head" else "head"
        self.needs_redraw = True
        self.reset_3d_view = True
        print(f"[viewer] joint mode -> {self.joint_mode}")

    def toggle_labels(self) -> None:
        self.show_labels = not self.show_labels
        self.needs_redraw = True
        print(f"[viewer] joint labels -> {'on' if self.show_labels else 'off'}")

    def request_reset(self) -> None:
        self.reset_3d_view = True
        self.needs_redraw = True
        print("[viewer] reset 3D view")


class QuestJoints3DWindow:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure("Quest joints - 3D raw/head-normalized")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.default_elev = 18.0
        self.default_azim = -65.0
        self.fig.show()
    

    def match_unity_coord_conv(self, joints_xyz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Alter coordinates so visualized skeleton looks correct in Matplotlib 3D, matching the Unity coordinate system used by the Quest.
        Unity uses a left-handed coordinate system with Y up, Z forward, and X right.
        Matplotlib uses a right-handed coordinate system with Z up, Y forward, and X right.
        This function converts from Unity to Matplotlib coordinates by swapping Y and Z axes and negating the Z coordinate.
        """
        converted = {}
        for name, pos in joints_xyz.items():
            converted[name] = np.array([pos[0], pos[2], pos[1]], dtype=np.float64)
        return converted

    def draw(
        self,
        sample: Sample,
        index: int,
        total: int,
        joints_xyz: Dict[str, np.ndarray],
        mode: str,
        show_labels: bool,
        fixed_range_m: Optional[float],
        reset_view: bool,
        warning: str = "",
    ) -> None:
        self.ax.cla()
        mode_text = "RAW QUEST/UNITY COORDINATES" if mode == "raw" else "HEAD-NORMALIZED COORDINATES"
        self.ax.set_title(
            f"{mode_text}\n"
            f"sample {index + 1}/{total} | participant={sample.participant_id} | "
            f"condition={sample.condition} | hand={sample.hand}",
            fontsize=10,
        )
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        joints_xyz = self.match_unity_coord_conv(joints_xyz)

        if warning:
            self.ax.text2D(0.02, 0.95, warning, transform=self.ax.transAxes, color="red", fontsize=9)

        if not joints_xyz:
            self.ax.text2D(0.02, 0.88, "No valid joint positions found.", transform=self.ax.transAxes, color="red")
            self._set_default_limits(mode, fixed_range_m)
            self._finish(reset_view)
            return

        # Draw bones.
        for a, b in SKELETON_EDGES:
            if a not in joints_xyz or b not in joints_xyz:
                continue
            pa = joints_xyz[a]
            pb = joints_xyz[b]
            self.ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], linewidth=2)

        names = self._ordered_joint_names(joints_xyz)
        xyz = np.array([joints_xyz[name] for name in names], dtype=np.float64)

        # Draw joints.
        self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=34, depthshade=True)

        # Emphasize the head or head-relative origin.
        if "head" in joints_xyz:
            head = joints_xyz["head"]
            self.ax.scatter([head[0]], [head[1]], [head[2]], s=95, marker="o", depthshade=True)
            self.ax.text(head[0], head[1], head[2], " head", fontsize=9)
        
        if "left_palm" in joints_xyz:
            left_palm = joints_xyz["left_palm"]
            self.ax.scatter([left_palm[0]], [left_palm[1]], [left_palm[2]], s=100, marker="o", depthshade=True,color='blue')
            self.ax.text(left_palm[0], left_palm[1], left_palm[2], " left_palm", fontsize=9,color='blue')

        if "right_palm" in joints_xyz:
            right_palm = joints_xyz["right_palm"]
            self.ax.scatter([right_palm[0]], [right_palm[1]], [right_palm[2]], s=100, marker="o", depthshade=True,color='red')
            self.ax.text(right_palm[0], right_palm[1], right_palm[2], " right_palm", fontsize=9,color='red')

        if mode == "head":
            self.ax.scatter([0.0], [0.0], [0.0], s=80, marker="x")
            self.ax.text(0.0, 0.0, 0.0, " origin/head", fontsize=9)

        if show_labels:
            for name in names:
                p = joints_xyz[name]
                self.ax.text(p[0], p[1], p[2], f" {name}", fontsize=8)

        self._set_equal_limits_from_points(xyz, mode, fixed_range_m)
        self.ax.grid(True)
        self.ax.set_zlim(joints_xyz["root"][2]-0.5, joints_xyz["head"][2]+0.5)
        self._finish(reset_view)

    @staticmethod
    def _ordered_joint_names(joints_xyz: Dict[str, np.ndarray]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for name in PREFERRED_JOINT_ORDER:
            if name in joints_xyz:
                ordered.append(name)
                seen.add(name)
        ordered.extend(sorted(name for name in joints_xyz if name not in seen))
        return ordered

    def _set_default_limits(self, mode: str, fixed_range_m: Optional[float]) -> None:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        span = fixed_range_m if fixed_range_m is not None else 1.5
        if mode == "raw":
            center = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self._apply_equal_limits(center, span)

    def _set_equal_limits_from_points(self, xyz: np.ndarray, mode: str, fixed_range_m: Optional[float]) -> None:
        if xyz.size == 0 or not np.all(np.isfinite(xyz)):
            self._set_default_limits(mode, fixed_range_m)
            return

        if fixed_range_m is not None:
            span = max(0.05, float(fixed_range_m))
            center = np.zeros(3, dtype=np.float64) if mode == "head" else np.mean(xyz, axis=0)
        elif mode == "head":
            center = np.zeros(3, dtype=np.float64)
            span = max(0.75, 2.3 * float(np.max(np.abs(xyz))))
        else:
            mins = np.min(xyz, axis=0)
            maxs = np.max(xyz, axis=0)
            center = (mins + maxs) / 2.0
            span = max(0.75, 1.35 * float(np.max(maxs - mins)))

        self._apply_equal_limits(center, span)

    def _apply_equal_limits(self, center: np.ndarray, span: float) -> None:
        half = max(0.025, float(span) / 2.0)
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)

    def _finish(self, reset_view: bool) -> None:
        if reset_view:
            self.ax.view_init(elev=self.default_elev, azim=self.default_azim)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)




def safe_float_array3(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        arr = np.array(value[:3], dtype=np.float64)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr




def extract_raw_joint_positions(joints_payload: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], str]:
    joints = joints_payload.get("joints")
    if not isinstance(joints, dict):
        return {}, "joints_raw.json has no top-level 'joints' dictionary"

    raw: Dict[str, np.ndarray] = {}
    skipped = 0
    for joint_name, joint_entry in joints.items():
        if not isinstance(joint_entry, dict):
            skipped += 1
            continue
        if joint_entry.get("valid") is False:
            skipped += 1
            continue
        pos = safe_float_array3(joint_entry.get("position"))
        if pos is None:
            skipped += 1
            continue
        raw[str(joint_name)] = pos

    warning = f"Skipped {skipped} invalid joints" if skipped else ""
    return raw, warning


def joint_positions_for_mode(joints_payload: Dict[str, Any], mode: str) -> Tuple[Dict[str, np.ndarray], str]:
    raw, warning = extract_raw_joint_positions(joints_payload)
    if mode == "raw":
        return raw, warning

    if "head" not in raw:
        fallback = "No valid head joint found; showing raw coordinates instead"
        return raw, f"{warning}; {fallback}" if warning else fallback

    head = raw["head"]
    normalized = {name: pos - head for name, pos in raw.items()}
    return normalized, warning


def overlay_text(image_bgr: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    out = image_bgr.copy()
    lines = [line for line in lines if line]
    if not lines:
        return out

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 1
    x, y = 14, 28
    line_h = 24
    pad = 10

    max_w = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, w)

    x1 = max(0, x - pad)
    y1 = max(0, y - 19 - pad)
    x2 = min(out.shape[1] - 1, x + max_w + pad)
    y2 = min(out.shape[0] - 1, y + len(lines) * line_h + pad)

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    for i, line in enumerate(lines):
        cv2.putText(out, line, (x, y + i * line_h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def short_path(path: Path, max_len: int = 105) -> str:
    text = str(path)
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3):]


def optional_float_text(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def make_rgb_window_image(sample: Sample, index: int, total: int, state: ViewerState) -> np.ndarray:
    image = cv2.imread(str(sample.rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(image, f"Could not read {sample.rgb_path}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    meta = read_json(sample.meta_path)
    time_alignment = meta.get("time_alignment", {}) if isinstance(meta.get("time_alignment"), dict) else {}
    quest_meta = meta.get("quest_joints", {}) if isinstance(meta.get("quest_joints"), dict) else {}

    location = meta.get("location") or meta.get("location_id") or meta.get("collection_location")
    obj = meta.get("object") or meta.get("object_id") or meta.get("object_name")

    lines = [
        f"VIEWER VERSION: {VERSION}",
        f"Sample {index + 1}/{total}",
        f"participant={sample.participant_id}  sample_id={sample.sample_id}",
        f"label={sample.label_name}  condition={sample.condition}  hand={sample.hand}",
        f"sample_type={sample.sample_type}",
        f"3D joints window mode={'RAW' if state.joint_mode == 'raw' else 'HEAD-NORMALIZED'}  labels={'on' if state.show_labels else 'off'}",
    ]

    if location is not None or obj is not None:
        lines.append(f"location={location if location is not None else 'N/A'}  object={obj if obj is not None else 'N/A'}")

    if "time_skew_ms" in time_alignment:
        lines.append(f"server-side time_skew_ms={optional_float_text(time_alignment.get('time_skew_ms'), 1)}")

    if quest_meta:
        lines.append(f"quest_frame_id={quest_meta.get('frame_id')}  quest_time_ns={quest_meta.get('quest_time_ns')}")

    lines.extend([
        short_path(sample.sample_dir),
        "Controls: next=d/right/space, prev=a/left, raw/head=j, labels=l, reset=r, quit=q",
    ])

    return overlay_text(image, lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View collected RGB samples and Quest joints in a separate 3D window.")
    parser.add_argument("--dataset-root", type=Path, default=Path("handoff_dataset"), help="Path to handoff_dataset or handoff_dataset/samples")
    parser.add_argument("--participant", type=str, default=None, help="Filter to one participant ID, e.g. P01")
    parser.add_argument("--label", type=str, default=None, help="Filter to label_name, e.g. handoff or not_handoff")
    parser.add_argument("--condition", type=str, default=None, help="Filter to one condition")
    parser.add_argument("--hand", type=str, choices=["left", "right"], default=None, help="Filter to one hand")
    parser.add_argument("--joint-mode", choices=["head", "raw"], default="head", help="Initial 3D joint display mode")
    parser.add_argument("--joint-range-m", type=float, default=None, help="Optional fixed axis span in meters for the 3D joint window")
    parser.add_argument("--show-joint-labels", action="store_true", help="Start with joint labels visible in the 3D window")
    parser.add_argument("--start-index", type=int, default=0, help="Start at this zero-based sample index after sorting/filtering")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"view_collected_samples_3d_raw_toggle.py VERSION={VERSION}")

    samples = discover_samples(
        dataset_root=args.dataset_root,
        participant_filter=args.participant,
        label_filter=args.label,
        condition_filter=args.condition,
        hand_filter=args.hand,
    )
    if not samples:
        print("No complete samples found.")
        print(f"Checked sample root: {sample_root_from_dataset_root(args.dataset_root)}")
        return 1

    print(f"Found {len(samples)} complete samples.")
    print("Controls in RGB window: next=d/right/space, prev=a/left, raw/head=j, labels=l, reset=r, quit=q")
    print("The Quest joints are drawn in a separate Matplotlib 3D window.")

    state = ViewerState(total=len(samples), start_index=args.start_index, joint_mode=args.joint_mode, show_labels=args.show_joint_labels)
    joints_window = QuestJoints3DWindow()

    cv2.namedWindow(IMAGE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(IMAGE_WINDOW, 1280, 720)

    while not state.quit:
        if state.needs_redraw:
            sample = samples[state.index]

            rgb_view = make_rgb_window_image(sample, state.index, len(samples), state)
            cv2.imshow(IMAGE_WINDOW, rgb_view)

            joints_payload = read_json(sample.joints_path)
            joints_xyz, warning = joint_positions_for_mode(joints_payload, state.joint_mode)
            joints_window.draw(
                sample=sample,
                index=state.index,
                total=len(samples),
                joints_xyz=joints_xyz,
                mode=state.joint_mode,
                show_labels=state.show_labels,
                fixed_range_m=args.joint_range_m,
                reset_view=state.reset_3d_view,
                warning=warning,
            )

            state.needs_redraw = False
            state.reset_3d_view = False

        # Pump both GUI event loops.
        plt.pause(0.001)
        key = cv2.waitKey(50) & 0xFF
        if key == 255:
            continue

        # Arrow key codes vary by OS/OpenCV backend. The letter keys are the reliable controls.
        if key in (ord("q"), 27):
            state.quit = True
        elif key in (ord("d"), ord("n"), ord(" "), 83):
            state.move(1)
        elif key in (ord("a"), ord("p"), 81):
            state.move(-1)
        elif key in (ord("j"), ord("m")):
            state.toggle_joint_mode()
        elif key == ord("l"):
            state.toggle_labels()
        elif key == ord("r"):
            state.request_reset()

    cv2.destroyAllWindows()
    plt.close(joints_window.fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
