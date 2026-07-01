"""
Simple image viewer for collected handoff intent samples.

Usage examples:
  python ViewCollectedImages.py
  python ViewCollectedImages.py --participant P01
  python ViewCollectedImages.py --dataset-root handoff_dataset

Controls:
  n / RightArrow : next image
  p / LeftArrow  : previous image
  q / Esc        : quit
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


@dataclass
class SampleRecord:
    participant_id: str
    label_name: str
    condition: str
    hand: str
    sample_id: str
    rgb_path: Path


def _safe_get(row: Dict[str, str], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return value.strip() if isinstance(value, str) else default


def load_samples_from_csv(dataset_root: Path) -> List[SampleRecord]:
    csv_path = dataset_root / "samples_granular.csv"
    if not csv_path.exists():
        return []

    samples: List[SampleRecord] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rgb_text = _safe_get(row, "rgb_path")
            if not rgb_text:
                continue

            rgb_path = Path(rgb_text)
            if not rgb_path.exists():
                continue

            participant_id = _safe_get(row, "participant_id", "participant_unset")
            label_name = _safe_get(row, "label_name", "unknown")
            condition = _safe_get(row, "condition", "unknown")
            hand = _safe_get(row, "hand", "unknown")
            sample_id = _safe_get(row, "sample_id", rgb_path.parent.name)

            samples.append(
                SampleRecord(
                    participant_id=participant_id,
                    label_name=label_name,
                    condition=condition,
                    hand=hand,
                    sample_id=sample_id,
                    rgb_path=rgb_path,
                )
            )

    return samples


def load_samples_from_tree(dataset_root: Path) -> List[SampleRecord]:
    """
    Fallback loader when CSV is missing. Expects:
    samples/<participant>/<label_name>/<condition>/<hand>/<sample_id>/rgb.png
    """
    samples_root = dataset_root / "samples"
    if not samples_root.exists():
        return []

    samples: List[SampleRecord] = []
    for rgb_path in samples_root.glob("*/**/rgb.png"):
        parts = rgb_path.parts
        try:
            # .../samples/<participant>/<label>/<condition>/<hand>/<sample_id>/rgb.png
            i = parts.index("samples")
            participant_id = parts[i + 1]
            label_name = parts[i + 2]
            condition = parts[i + 3]
            hand = parts[i + 4]
            sample_id = parts[i + 5]
        except (ValueError, IndexError):
            continue

        samples.append(
            SampleRecord(
                participant_id=participant_id,
                label_name=label_name,
                condition=condition,
                hand=hand,
                sample_id=sample_id,
                rgb_path=rgb_path,
            )
        )

    return samples


def draw_label_overlay(image_bgr: np.ndarray, text_lines: List[str]) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]

    panel_h = 30 + (30 * len(text_lines))
    cv2.rectangle(out, (0, 0), (w, min(panel_h, h)), (0, 0, 0), thickness=-1)

    y = 28
    for line in text_lines:
        cv2.putText(
            out,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    return out


def show_samples(samples: List[SampleRecord]) -> None:
    if not samples:
        print("No images found to display.")
        return

    samples.sort(
        key=lambda s: (
            s.participant_id,
            s.label_name,
            s.condition,
            s.hand,
            s.sample_id,
        )
    )

    idx = 0
    window_name = "Collected Handoff Images"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Controls: n/Right = next, p/Left = previous, q/Esc = quit")

    while True:
        s = samples[idx]
        frame = cv2.imread(str(s.rgb_path), cv2.IMREAD_COLOR)
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Could not load image: {s.rgb_path}",
                (20, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        info = [
            f"Participant: {s.participant_id}   Sample {idx + 1}/{len(samples)}",
            f"Label: {s.label_name}   Condition: {s.condition}   Hand: {s.hand}",
            f"Sample ID: {s.sample_id}",
        ]
        display = draw_label_overlay(frame, info)

        cv2.imshow(window_name, display)
        key = cv2.waitKeyEx(0)

        if key in (ord("q"), 27):
            break

        # Right arrow or n
        if key in (ord("n"), 2555904):
            idx = (idx + 1) % len(samples)
            continue

        # Left arrow or p
        if key in (ord("p"), 2424832):
            idx = (idx - 1) % len(samples)
            continue

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="View collected participant images with labels")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("handoff_dataset"),
        help="Path to the dataset root (default: handoff_dataset)",
    )
    parser.add_argument(
        "--participant",
        type=str,
        default=None,
        help="Optional participant ID filter, e.g. P01",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    samples = load_samples_from_csv(dataset_root)
    if not samples:
        samples = load_samples_from_tree(dataset_root)

    if args.participant:
        samples = [s for s in samples if s.participant_id == args.participant]

    if not samples and args.participant:
        print(f"No images found for participant '{args.participant}'.")
        return

    show_samples(samples)


if __name__ == "__main__":
    main()
