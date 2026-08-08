#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Union 

import torch
import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    REPO_ROOT
    / "robot_fov_estimation"
    / "data"
    / "go2_yolo_det_data"
)

EVAL_RESULTS_DIR = (
    REPO_ROOT
    / "robot_fov_estimation"
    / "outputs"
    / "eval_results"
    / "go2_yolo"
)

WEIGHTS_DIR = (
    REPO_ROOT
    / "robot_fov_estimation"
    / "outputs"
    / "go2_yolo_det_weights"
)

TRAIN_RUN_DIR = WEIGHTS_DIR / "training_run"


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

MODEL_WEIGHTS = "yolo26s.pt"

EPOCHS = 100
IMAGE_SIZE = 1280
TRAIN_BATCH_SIZE = -1
EVAL_BATCH_SIZE = 4
PATIENCE = 20
WORKERS = 8
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--resume-training",action=argparse.BooleanOptionalAction,default=False,)
parser.add_argument("--eval-only",action=argparse.BooleanOptionalAction,default=False,)

RESUME_TRAINING = parser.parse_args().resume_training
EVAL_ONLY = parser.parse_args().eval_only 

if(RESUME_TRAINING and EVAL_ONLY):
    raise ValueError("Cannot specify both --resume-training and --eval-only.")

RESUME_CHECKPOINT = (
    TRAIN_RUN_DIR
    / "weights"
    / "last.pt"
)


def find_data_yaml(dataset_dir: Path) -> Path:
    """
    Locate the Roboflow-generated data.yaml file.

    This handles both:
        go2_yolo_det_data/data.yaml

    and:
        go2_yolo_det_data/some_extracted_folder/data.yaml
    """
    direct_path = dataset_dir / "data.yaml"

    if direct_path.is_file():
        return direct_path.resolve()

    candidates = list(dataset_dir.rglob("data.yaml"))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find data.yaml anywhere under:\n{dataset_dir}"
        )

    if len(candidates) > 1:
        formatted_candidates = "\n".join(f"  - {path}" for path in candidates)
        raise RuntimeError(
            "Found multiple data.yaml files. Remove the unwanted copies or "
            f"select one explicitly:\n{formatted_candidates}"
        )

    return candidates[0].resolve()


def load_dataset_config(data_yaml: Path) -> dict[str, Any]:
    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid dataset configuration: {data_yaml}")

    for required_key in ("train", "val", "names"):
        if required_key not in config:
            raise ValueError(
                f"Dataset configuration is missing '{required_key}': "
                f"{data_yaml}"
            )

    return config


def metrics_to_dictionary(metrics: Any) -> dict[str, Any]:
    """Convert the main Ultralytics detection metrics to JSON-safe values."""
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": [
            float(value) for value in metrics.box.maps
        ],
        "speed_ms_per_image": {
            name: float(value)
            for name, value in metrics.speed.items()
        },
        "ultralytics_results": {
            str(name): float(value)
            for name, value in metrics.results_dict.items()
        },
    }


def evaluate_split(
    model: YOLO,
    data_yaml: Path,
    split: str,
    output_dir: Path,
    device: Union[int, str] = 0,
) -> dict[str, Any]:
    print(f"\nEvaluating the '{split}' split...")
    print(f"Evaluation output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=IMAGE_SIZE,
        batch=EVAL_BATCH_SIZE,
        device=device,
        workers=WORKERS,
        plots=True,
        save_dir=str(output_dir),
        exist_ok=True,
        verbose=True,
        single_cls=True,
    )

    summary = metrics_to_dictionary(metrics)

    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Saved metric summary to: {summary_path}")

    return summary


def copy_weights(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Expected weights were not created: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    print(f"Copied weights:\n  {source}\n  -> {destination}")


def main() -> None:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist:\n{DATASET_DIR}"
        )

    data_yaml = find_data_yaml(DATASET_DIR)
    dataset_config = load_dataset_config(data_yaml)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_RUN_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device: Union[int, str] = 0 
        device_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        device_name = "CPU"

    print("Go2 YOLO detector training")
    print("--------------------------")
    print(f"Dataset YAML:     {data_yaml}")
    print(f"Training output:  {TRAIN_RUN_DIR}")
    print(f"Weights output:   {WEIGHTS_DIR}")
    print(f"Evaluation output:{EVAL_RESULTS_DIR}")
    print(f"Device:           {device_name}")
    print(f"Starting model:   {MODEL_WEIGHTS}")

    if EVAL_ONLY:
        saved_best = WEIGHTS_DIR / "best.pt"

        if not saved_best.is_file():
            raise FileNotFoundError(
                f"Best weights were not found:\n{saved_best}"
            )

        best_model = YOLO(str(saved_best))

        evaluation_summary = {
            "best_weights": str(saved_best),
            "dataset_yaml": str(data_yaml),
            "device": device_name,
            "validation": evaluate_split(
                model=best_model,
                data_yaml=data_yaml,
                split="val",
                output_dir=EVAL_RESULTS_DIR / "val",
                device=device,
            ),
        }

        if dataset_config.get("test"):
            evaluation_summary["test"] = evaluate_split(
                model=best_model,
                data_yaml=data_yaml,
                split="test",
                output_dir=EVAL_RESULTS_DIR / "test",
                device=device,
            )

        combined_summary_path = (
            EVAL_RESULTS_DIR / "evaluation_summary.json"
        )

        with combined_summary_path.open("w", encoding="utf-8") as file:
            json.dump(evaluation_summary, file, indent=4)

        print("\nEvaluation completed.")
        print(f"Evaluation summary: {combined_summary_path}")
        return

    # Loading a named pretrained checkpoint automatically downloads it on the
    # first run if it is not already present.
    if RESUME_TRAINING:
        if not RESUME_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint was not found:\n{RESUME_CHECKPOINT}"
            )

        print(f"Resuming training from:\n{RESUME_CHECKPOINT}")

        model = YOLO(str(RESUME_CHECKPOINT))
        model.train(resume=True)

    else:
        model = YOLO(MODEL_WEIGHTS)

        model.train(
            data=str(data_yaml),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=TRAIN_BATCH_SIZE,
            patience=PATIENCE,
            device=device,
            workers=WORKERS,
            seed=SEED,
            deterministic=True,
            pretrained=True,
            amp=torch.cuda.is_available(),
            cache=False,
            plots=True,
            val=True,
            save=True,
            save_period=-1,
            save_dir=str(TRAIN_RUN_DIR),
            exist_ok=True,
            verbose=True,
            single_cls=True,
        )

    if model.trainer is None:
        raise RuntimeError("Ultralytics did not retain the trainer instance.")

    trained_best = Path(model.trainer.best)
    trained_last = Path(model.trainer.last)

    saved_best = WEIGHTS_DIR / "best.pt"
    saved_last = WEIGHTS_DIR / "last.pt"

    copy_weights(trained_best, saved_best)
    copy_weights(trained_last, saved_last)

    # Evaluate the selected best checkpoint rather than the final epoch.
    best_model = YOLO(str(saved_best))

    evaluation_summary: dict[str, Any] = {
        "model": MODEL_WEIGHTS,
        "best_weights": str(saved_best),
        "last_weights": str(saved_last),
        "dataset_yaml": str(data_yaml),
        "device": device_name,
        "validation": evaluate_split(
            model=best_model,
            data_yaml=data_yaml,
            split="val",
            output_dir=EVAL_RESULTS_DIR / "val",
            device=device,
        ),
    }

    # Roboflow exports ordinarily include a test split. Skip it gracefully
    # when the YAML does not define one.
    if dataset_config.get("test"):
        evaluation_summary["test"] = evaluate_split(
            model=best_model,
            data_yaml=data_yaml,
            split="test",
            output_dir=EVAL_RESULTS_DIR / "test",
            device=device,
        )
    else:
        print("\nNo test split was defined in data.yaml; skipping test evaluation.")

    combined_summary_path = EVAL_RESULTS_DIR / "evaluation_summary.json"

    with combined_summary_path.open("w", encoding="utf-8") as file:
        json.dump(evaluation_summary, file, indent=4)

    print("\nTraining and evaluation completed.")
    print(f"Best weights:       {saved_best}")
    print(f"Last weights:       {saved_last}")
    print(f"Evaluation summary: {combined_summary_path}")


if __name__ == "__main__":
    main()