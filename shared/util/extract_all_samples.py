from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any 
import json

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - enables 3D projection on older installs

REQUIRED_FILES = ("rgb.png", "joints_raw.json", "meta.json")

@dataclass(frozen=True)
class Sample:
    sample_dir: Path
    rgb_path: Path
    joints_path: Path
    meta_path: Path
    participant_id: str
    label_name: str
    condition: str
    hand: str
    sample_type: str
    sample_id: str


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