from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from shared.util.extract_all_samples import (
    discover_samples,
    get_sample_label_name,
    label_to_binary,
)
from src.hand_intent_cloud_vlm import CloudHandoffClassifier, HANDOFF_PROMPT


MODEL_IMPL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = MODEL_IMPL_ROOT / "configs" / "handoff_config.yaml"


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=np.int32)
    y_pred_array = np.asarray(y_pred, dtype=np.int32)

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have matching shapes")

    total = int(len(y_true_array))
    tp = int(np.sum((y_true_array == 1) & (y_pred_array == 1)))
    tn = int(np.sum((y_true_array == 0) & (y_pred_array == 0)))
    fp = int(np.sum((y_true_array == 0) & (y_pred_array == 1)))
    fn = int(np.sum((y_true_array == 1) & (y_pred_array == 0)))

    accuracy = float((tp + tn) / total) if total else None
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    specificity = float(tn / (tn + fp)) if tn + fp else None
    balanced_accuracy = (
        float((recall + specificity) / 2)
        if recall is not None and specificity is not None
        else None
    )
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else 0.0
    )

    return {
        "total": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def participant_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        participant_id = row.get("participant_id")
        if participant_id is not None:
            grouped[str(participant_id)].append(row)

    participant_results = []
    for participant_id in sorted(grouped):
        participant_rows = grouped[participant_id]
        metrics = compute_binary_metrics(
            [int(row["label_binary"]) for row in participant_rows],
            [int(row["prediction"]) for row in participant_rows],
        )
        participant_results.append(
            {"participant_id": participant_id, "metrics": metrics}
        )

    metric_summary = {}
    for metric_name in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]:
        values = [
            result["metrics"][metric_name]
            for result in participant_results
            if result["metrics"][metric_name] is not None
        ]
        metric_summary[metric_name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }

    return {
        "participants": participant_results,
        "participant_metric_summary": metric_summary,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cache(
    path: Path,
    provider: str,
    model: str,
    prompt_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Return the latest successful matching result for every image."""
    cache = {}
    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if (
                row.get("provider") == provider
                and row.get("model") == model
                and row.get("prompt_sha256") == prompt_sha256
                and row.get("prediction") in (0, 1)
            ):
                cache[str(Path(row["image"]).resolve())] = row
    return cache


def predict_with_retries(
    classifier: CloudHandoffClassifier,
    image_path: Path,
    max_retries: int,
    retry_backoff: float,
) -> tuple[int, str, int]:
    for attempt in range(max_retries + 1):
        try:
            prediction, raw_response = classifier.predict_with_response(image_path)
            return prediction, raw_response, attempt + 1
        except Exception as exc:
            if attempt == max_retries:
                raise
            delay = retry_backoff * (2**attempt)
            print(f"  {type(exc).__name__}: {exc}; retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("Unreachable retry state")


def evaluate_samples(
    samples: list[Any],
    classifier: CloudHandoffClassifier,
    output_jsonl: Path,
    start_index: int,
    resume: bool,
    max_retries: int,
    retry_backoff: float,
    request_delay: float,
) -> list[dict[str, Any]]:
    cache = (
        load_cache(
            output_jsonl,
            classifier.provider,
            classifier.model,
            classifier.prompt_sha256,
        )
        if resume
        else {}
    )
    rows = []

    for local_index, sample in enumerate(samples):
        image_path = Path(sample.rgb_path).resolve()
        cache_key = str(image_path)
        label_name = get_sample_label_name(sample)
        label_binary = int(label_to_binary(label_name))
        participant_id = getattr(sample, "participant_id", None)

        if (
            cache_key in cache
            and int(cache[cache_key].get("label_binary", -1)) == label_binary
        ):
            row = cache[cache_key]
            rows.append(row)
            print(
                f"[{local_index + 1}/{len(samples)}] cached "
                f"label={label_binary} pred={row['prediction']} {image_path.name}"
            )
            continue

        print(
            f"[{local_index + 1}/{len(samples)}] requesting "
            f"participant={participant_id} label={label_binary} {image_path.name}"
        )
        row = {
            "index": start_index + local_index,
            "participant_id": (
                str(participant_id) if participant_id is not None else None
            ),
            "image": str(image_path),
            "label_name": label_name,
            "label_binary": label_binary,
            "provider": classifier.provider,
            "model": classifier.model,
            "prompt_sha256": classifier.prompt_sha256,
        }

        request_start = time.perf_counter()
        try:
            prediction, raw_response, attempts = predict_with_retries(
                classifier,
                image_path,
                max_retries,
                retry_backoff,
            )
            row.update(
                {
                    "prediction": prediction,
                    "correct": prediction == label_binary,
                    "raw_response": raw_response,
                    "attempts": attempts,
                    "error": None,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "prediction": None,
                    "correct": None,
                    "raw_response": None,
                    "attempts": max_retries + 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        row["latency_seconds"] = float(time.perf_counter() - request_start)

        append_jsonl(output_jsonl, row)
        rows.append(row)
        print(
            f"  pred={row['prediction']} correct={row['correct']} "
            f"latency={row['latency_seconds']:.2f}s"
        )

        if request_delay > 0 and local_index + 1 < len(samples):
            time.sleep(request_delay)

    return rows


def build_summary(
    rows: list[dict[str, Any]],
    classifier: CloudHandoffClassifier,
    dataset_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("prediction") in (0, 1)]
    errors = [row for row in rows if row.get("prediction") not in (0, 1)]
    y_true = [int(row["label_binary"]) for row in successful]
    y_pred = [int(row["prediction"]) for row in successful]
    latencies = [float(row["latency_seconds"]) for row in successful]

    return {
        "evaluation_type": "zero_shot_cloud_vlm",
        "provider": classifier.provider,
        "model": classifier.model,
        "dataset_root": str(dataset_root),
        "prompt": HANDOFF_PROMPT,
        "prompt_sha256": classifier.prompt_sha256,
        "filters": {
            "participant": args.participant,
            "label": args.label,
            "condition": args.condition,
            "hand": args.hand,
            "start_index": args.start_index,
            "max_samples": args.max_samples,
        },
        "counts": {
            "selected": len(rows),
            "successful": len(successful),
            "errors": len(errors),
            "true_labels": np.bincount(y_true, minlength=2).tolist(),
            "predicted_labels": np.bincount(y_pred, minlength=2).tolist(),
        },
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
            "total": sum(latencies),
        },
        "pooled_metrics": compute_binary_metrics(y_true, y_pred),
        **participant_evaluation(successful),
        "errors": [
            {"image": row["image"], "error": row["error"]} for row in errors
        ],
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\nCloud VLM evaluation complete")
    print(f"Provider/model: {summary['provider']} / {summary['model']}")
    print(f"Successful: {summary['counts']['successful']}")
    print(f"Errors: {summary['counts']['errors']}")
    print("Pooled metrics:")
    for name, value in summary["pooled_metrics"].items():
        if name == "confusion_matrix":
            print(f"  {name}: {value}")
        elif isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    print("Per-participant metrics:")
    for result in summary["participants"]:
        metrics = result["metrics"]
        print(
            f"  {result['participant_id']}: n={metrics['total']}, "
            f"accuracy={metrics['accuracy']}, f1={metrics['f1']:.4f}"
        )


def resolve_dataset_root(override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        configured = Path(yaml.safe_load(config_file)["dataset"]["root"])
    return configured if configured.is_absolute() else (REPO_ROOT / configured).resolve()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a cloud VLM on the handoff-intention dataset."
    )
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--dataset-root", default=None)

    parser.add_argument("--participant", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--hand", choices=["left", "right"], default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--max-image-dimension", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        default=str(MODEL_IMPL_ROOT / "outputs" / "cloud_vlm"),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    samples = discover_samples(
        dataset_root=dataset_root,
        participant_filter=args.participant,
        label_filter=args.label,
        condition_filter=args.condition,
        hand_filter=args.hand,
    )
    samples = samples[args.start_index :]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No samples matched the selected filters")

    classifier = CloudHandoffClassifier(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        max_image_dimension=args.max_image_dimension,
        jpeg_quality=args.jpeg_quality,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    stem = f"handoff_{safe_name(classifier.provider)}_{safe_name(classifier.model)}"
    output_jsonl = output_dir / f"{stem}.jsonl"
    output_summary = output_dir / f"{stem}.summary.json"
    if not args.resume and output_jsonl.exists():
        output_jsonl.unlink()

    print(f"Dataset: {dataset_root}")
    print(f"Samples: {len(samples)}")
    print(f"Provider/model: {classifier.provider} / {classifier.model}")
    print(f"Results: {output_jsonl}")

    rows = evaluate_samples(
        samples=samples,
        classifier=classifier,
        output_jsonl=output_jsonl,
        start_index=args.start_index,
        resume=args.resume,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        request_delay=args.request_delay,
    )
    summary = build_summary(rows, classifier, dataset_root, args)

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"Summary: {output_summary}")
    return summary


if __name__ == "__main__":
    main()