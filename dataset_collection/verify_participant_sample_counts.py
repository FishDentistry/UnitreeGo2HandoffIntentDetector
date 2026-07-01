#!/usr/bin/env python3
"""
Verify that every participant folder has the expected number of saved samples.

Default expectation:
    9 locations * 8 conditions * 3 objects = 216 samples per participant

Expected dataset layout from the collection server:
    handoff_dataset/
      samples/
        <participant_id>/
          .../
            <sample_id>/
              rgb.png
              depth_z16.png
              joints_raw.json
              meta.json

Run:
    python verify_participant_sample_counts.py

Or specify a different dataset root:
    python verify_participant_sample_counts.py --dataset-root path/to/handoff_dataset

You can also pass the samples directory directly:
    python verify_participant_sample_counts.py --samples-dir path/to/handoff_dataset/samples
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Optional


REQUIRED_SAMPLE_FILES = ("rgb.png", "depth_z16.png", "joints_raw.json", "meta.json")

DEFAULT_NUM_LOCATIONS = 9
DEFAULT_NUM_CONDITIONS = 8
DEFAULT_NUM_OBJECTS = 3

LOCATION_KEYS = (
    "location",
    "location_id",
    "location_name",
    "collection_location",
    "collection_location_id",
    "capture_location",
    "capture_location_id",
    "target_location",
    "target_location_id",
    "station",
    "station_id",
)

OBJECT_KEYS = (
    "object",
    "object_id",
    "object_name",
    "held_object",
    "held_object_id",
    "held_object_name",
    "prop",
    "prop_id",
    "item",
    "item_id",
)

CONDITION_KEYS = (
    "condition_id",
    "condition_name",
    "sample_type",
)


def parse_csv_list(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return values or None


def resolve_samples_dir(dataset_root: Optional[str], samples_dir: Optional[str]) -> Path:
    if samples_dir:
        return Path(samples_dir)

    root = Path(dataset_root or "handoff_dataset")
    if root.name == "samples":
        return root
    return root / "samples"


def find_sample_dirs(participant_dir: Path) -> list[Path]:
    """
    A sample directory is any directory containing at least one of the expected
    sample files. This catches incomplete samples too, instead of only dirs with
    meta.json.
    """
    sample_dirs: set[Path] = set()
    for filename in REQUIRED_SAMPLE_FILES:
        for file_path in participant_dir.rglob(filename):
            if file_path.is_file():
                sample_dirs.add(file_path.parent)
    return sorted(sample_dirs)


def is_complete_sample(sample_dir: Path) -> bool:
    return all((sample_dir / filename).is_file() for filename in REQUIRED_SAMPLE_FILES)


def read_meta(sample_dir: Path) -> dict[str, Any]:
    meta_path = sample_dir / "meta.json"
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def first_present(meta: dict[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = meta.get(key)
        if value is not None and value != "":
            return str(value)
    return None


def condition_from_meta(meta: dict[str, Any]) -> Optional[str]:
    # Prefer a specific 8-way label if available.
    value = first_present(meta, CONDITION_KEYS)
    if value is not None:
        return value

    # The provided collection script stores condition + hand separately.
    # Combining them reconstructs the 8 hotkey/sample_type-style conditions.
    condition = meta.get("condition")
    hand = meta.get("hand")
    if condition is not None and hand is not None:
        return f"{condition}_{hand}"
    if condition is not None:
        return str(condition)
    return None


def infer_dimensions(sample_dirs: list[Path]) -> tuple[Counter[tuple[str, str, str]], dict[str, set[str]], list[Path]]:
    """
    Returns counts keyed by (location, condition, object) when all three fields
    are present in meta.json. Samples missing any of those fields are returned
    separately.
    """
    combo_counts: Counter[tuple[str, str, str]] = Counter()
    values: dict[str, set[str]] = defaultdict(set)
    missing_dimension_samples: list[Path] = []

    for sample_dir in sample_dirs:
        meta = read_meta(sample_dir)
        location = first_present(meta, LOCATION_KEYS)
        condition = condition_from_meta(meta)
        obj = first_present(meta, OBJECT_KEYS)

        if location is None or condition is None or obj is None:
            missing_dimension_samples.append(sample_dir)
            continue

        values["location"].add(location)
        values["condition"].add(condition)
        values["object"].add(obj)
        combo_counts[(location, condition, obj)] += 1

    return combo_counts, values, missing_dimension_samples


def print_participant_problem(
    participant_id: str,
    participant_dir: Path,
    complete_count: int,
    expected_total: int,
    incomplete_dirs: list[Path],
) -> None:
    delta = complete_count - expected_total
    if delta < 0:
        status = f"{abs(delta)} short"
    else:
        status = f"{delta} extra"

    print(f"[COUNT MISMATCH] {participant_id}: {complete_count}/{expected_total} complete samples ({status})")
    print(f"  folder: {participant_dir}")

    if incomplete_dirs:
        print(f"  incomplete sample dirs: {len(incomplete_dirs)}")
        for sample_dir in incomplete_dirs[:10]:
            missing = [name for name in REQUIRED_SAMPLE_FILES if not (sample_dir / name).is_file()]
            print(f"    - {sample_dir} missing {missing}")
        if len(incomplete_dirs) > 10:
            print(f"    ... {len(incomplete_dirs) - 10} more incomplete dirs not shown")


def print_balance_problems(
    participant_id: str,
    complete_sample_dirs: list[Path],
    expected_locations: int,
    expected_conditions: int,
    expected_objects: int,
    explicit_locations: Optional[list[str]],
    explicit_conditions: Optional[list[str]],
    explicit_objects: Optional[list[str]],
) -> None:
    """
    Optional deeper check. This only works if location/object/condition fields
    are present in meta.json, or if explicit value lists are supplied.
    """
    combo_counts, observed_values, missing_dimension_samples = infer_dimensions(complete_sample_dirs)

    if not combo_counts:
        # The current collector script does not appear to write location/object.
        return

    locations = explicit_locations or sorted(observed_values["location"])
    conditions = explicit_conditions or sorted(observed_values["condition"])
    objects = explicit_objects or sorted(observed_values["object"])

    dimension_issue = False
    if len(locations) != expected_locations:
        print(f"[DIMENSION MISMATCH] {participant_id}: found {len(locations)} locations, expected {expected_locations}")
        print(f"  locations found: {locations}")
        dimension_issue = True
    if len(conditions) != expected_conditions:
        print(f"[DIMENSION MISMATCH] {participant_id}: found {len(conditions)} conditions, expected {expected_conditions}")
        print(f"  conditions found: {conditions}")
        dimension_issue = True
    if len(objects) != expected_objects:
        print(f"[DIMENSION MISMATCH] {participant_id}: found {len(objects)} objects, expected {expected_objects}")
        print(f"  objects found: {objects}")
        dimension_issue = True

    if missing_dimension_samples:
        print(
            f"[BALANCE WARNING] {participant_id}: "
            f"{len(missing_dimension_samples)} complete samples are missing location, condition, or object metadata"
        )
        for sample_dir in missing_dimension_samples[:10]:
            print(f"  - {sample_dir}")
        if len(missing_dimension_samples) > 10:
            print(f"  ... {len(missing_dimension_samples) - 10} more not shown")

    # If explicit lists are supplied, this can identify missing locations/objects
    # that do not appear in the data at all. Without explicit lists, it checks
    # the Cartesian product of observed values.
    expected_combos = list(product(locations, conditions, objects))
    missing_combos = [combo for combo in expected_combos if combo_counts.get(combo, 0) == 0]
    extra_combos = [(combo, count) for combo, count in combo_counts.items() if count > 1]

    if missing_combos or extra_combos or dimension_issue:
        if missing_combos:
            print(f"[MISSING COMBOS] {participant_id}: {len(missing_combos)} location/condition/object combos have 0 samples")
            for location, condition, obj in missing_combos[:20]:
                print(f"  - location={location}, condition={condition}, object={obj}")
            if len(missing_combos) > 20:
                print(f"  ... {len(missing_combos) - 20} more missing combos not shown")

        if extra_combos:
            print(f"[DUPLICATE COMBOS] {participant_id}: {len(extra_combos)} location/condition/object combos have more than 1 sample")
            for (location, condition, obj), count in extra_combos[:20]:
                print(f"  - location={location}, condition={condition}, object={obj}: {count} samples")
            if len(extra_combos) > 20:
                print(f"  ... {len(extra_combos) - 20} more duplicate combos not shown")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify expected sample count per participant folder."
    )
    parser.add_argument("--dataset-root", default="handoff_dataset", help="Dataset root. Default: handoff_dataset")
    parser.add_argument("--samples-dir", default=None, help="Samples directory. Overrides --dataset-root.")
    parser.add_argument("--locations", type=int, default=DEFAULT_NUM_LOCATIONS)
    parser.add_argument("--conditions", type=int, default=DEFAULT_NUM_CONDITIONS)
    parser.add_argument("--objects", type=int, default=DEFAULT_NUM_OBJECTS)
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Expected total samples per participant. Overrides locations*conditions*objects.",
    )
    parser.add_argument(
        "--location-values",
        default=None,
        help="Optional comma-separated list of valid location IDs/names for exact missing-combo checks.",
    )
    parser.add_argument(
        "--condition-values",
        default=None,
        help="Optional comma-separated list of valid condition IDs/names for exact missing-combo checks.",
    )
    parser.add_argument(
        "--object-values",
        default=None,
        help="Optional comma-separated list of valid object IDs/names for exact missing-combo checks.",
    )
    parser.add_argument(
        "--print-ok",
        action="store_true",
        help="Also print participant folders that have the expected count.",
    )
    args = parser.parse_args()

    samples_dir = resolve_samples_dir(args.dataset_root, args.samples_dir)
    expected_total = args.expected or (args.locations * args.conditions * args.objects)

    if not samples_dir.is_dir():
        print(f"[ERROR] Samples directory does not exist: {samples_dir}")
        return 2

    participant_dirs = sorted(p for p in samples_dir.iterdir() if p.is_dir())
    if not participant_dirs:
        print(f"[ERROR] No participant folders found in: {samples_dir}")
        return 2

    any_problem = False

    explicit_locations = parse_csv_list(args.location_values)
    explicit_conditions = parse_csv_list(args.condition_values)
    explicit_objects = parse_csv_list(args.object_values)

    for participant_dir in participant_dirs:
        participant_id = participant_dir.name
        sample_dirs = find_sample_dirs(participant_dir)
        complete_sample_dirs = [p for p in sample_dirs if is_complete_sample(p)]
        incomplete_dirs = [p for p in sample_dirs if not is_complete_sample(p)]
        complete_count = len(complete_sample_dirs)

        if complete_count != expected_total or incomplete_dirs:
            any_problem = True
            print_participant_problem(
                participant_id=participant_id,
                participant_dir=participant_dir,
                complete_count=complete_count,
                expected_total=expected_total,
                incomplete_dirs=incomplete_dirs,
            )

        elif args.print_ok:
            print(f"[OK] {participant_id}: {complete_count}/{expected_total} complete samples")

        print_balance_problems(
            participant_id=participant_id,
            complete_sample_dirs=complete_sample_dirs,
            expected_locations=args.locations,
            expected_conditions=args.conditions,
            expected_objects=args.objects,
            explicit_locations=explicit_locations,
            explicit_conditions=explicit_conditions,
            explicit_objects=explicit_objects,
        )

    if not any_problem and not args.print_ok:
        print(f"[OK] All {len(participant_dirs)} participant folders have exactly {expected_total} complete samples.")

    return 1 if any_problem else 0


if __name__ == "__main__":
    raise SystemExit(main())
