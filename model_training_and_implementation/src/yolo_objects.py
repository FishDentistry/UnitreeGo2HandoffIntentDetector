"""Run YOLO object detection on a single image.

This script loads a YOLO model, runs inference on one image, and returns the
detected bounding boxes with confidence scores and class names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from ultralytics import YOLO


class YOLOObjectDetector:
	def __init__(self, weights: str = "yolov8s.pt", class_names: str | list[str] | None = None, confidence: float = 0.25):
		self.weights = weights
		self.confidence = confidence
		if isinstance(class_names, str):
			self.class_names = [name.strip() for name in class_names.split(",") if name.strip()]
		else:
			self.class_names = class_names
		self.model = YOLO(weights)
		self.device = 0 if torch.cuda.is_available() else "cpu"

	def predict(self, image_bgr):
		results = self.model.predict(source=image_bgr, conf=self.confidence, device=self.device, verbose=False)
		result = results[0]

		detections: List[Dict[str, Any]] = []
		boxes = result.boxes
		if boxes is None:
			return detections

		for box in boxes:
			class_id = int(box.cls.item())
			class_name = result.names[class_id]
			if self.class_names and class_name not in self.class_names:
				continue
			detections.append(
				{
					"box_xyxy": [float(value) for value in box.xyxy[0].tolist()],
					"score": float(box.conf.item()),
					"class_id": class_id,
					"class_name": class_name,
				}
			)

		return detections


def detect_objects(image_path: str, model_path: str = "yolov8s.pt", conf: float = 0.25) -> Dict[str, Any]:
	"""Run YOLO on an image and return structured detections."""

	image_file = Path(image_path)
	if not image_file.is_file():
		raise FileNotFoundError(f"Image not found: {image_path}")

	model = YOLO(model_path)
	device = 0 if torch.cuda.is_available() else "cpu"
	results = model.predict(source=str(image_file), conf=conf, device=device, verbose=False)
	result = results[0]

	detections: List[Dict[str, Any]] = []
	boxes = result.boxes
	if boxes is not None:
		class_names = result.names
		for box in boxes:
			class_id = int(box.cls.item())
			detections.append(
				{
					"box_xyxy": [float(value) for value in box.xyxy[0].tolist()],
					"score": float(box.conf.item()),
					"class_id": class_id,
					"class_name": class_names[class_id],
				}
			)

	return {
		"image_path": str(image_file),
		"model_path": model_path,
		"confidence_threshold": conf,
		"detections": detections,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run YOLO object detection on an image.")
	parser.add_argument("image_path", help="Path to the input image")
	parser.add_argument(
		"--model",
		default="yolov8s.pt",
		help="YOLO model path or model name (default: yolov8s.pt)",
	)
	parser.add_argument(
		"--conf",
		type=float,
		default=0.25,
		help="Confidence threshold for detections (default: 0.25)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	detections = detect_objects(args.image_path, model_path=args.model, conf=args.conf)
	print(json.dumps(detections, indent=2))


if __name__ == "__main__":
	main()
