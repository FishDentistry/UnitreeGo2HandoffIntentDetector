"""Draw Detectron2, YOLO, and head-pose outputs on one image.

The script reads the single test image in ``data/test_images/test.jpg`` by
default, overlays detections from the three model paths already present in this
repo, and writes ``result.jpg`` next to this script.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "test_images" / "testhand.png"
RESULT_IMAGE_PATH = PROJECT_ROOT / "result.jpg"
DEFAULT_DETECTRON_CONFIG_PATH = PROJECT_ROOT / "configs" / "keypoint_rcnn_R_101_FPN_3x.yaml"
DEFAULT_HEAD_POSE_WEIGHTS_PATH = PROJECT_ROOT / "kwan_pretrained_weights" / "head-pose-pretrained.pkl"
DEFAULT_YOLO_MODEL_PATH = "yolov8n.pt"


detectron_keypoints = importlib.import_module("src.detectron_keypoints")
head_pose_module = importlib.import_module("src.head_pose")
yolo_objects = importlib.import_module("src.yolo_objects")

build_pose_detector = detectron_keypoints.build_pose_detector
KwanHeadPoseEstimator = head_pose_module.KwanHeadPoseEstimator
detect_objects = yolo_objects.detect_objects


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run all available models on one test image.")
	parser.add_argument(
		"--image",
		default=str(DEFAULT_IMAGE_PATH),
		help="Path to the input image (default: data/test_images/test.jpg)",
	)
	return parser.parse_args()


def load_image_bgr(image_path: Path) -> Any:
	image_bgr = cv2.imread(str(image_path))
	if image_bgr is None:
		raise FileNotFoundError(f"Unable to read image: {image_path}")
	return image_bgr


def clamp_point(x: float, y: float, image_shape: Tuple[int, int, int]) -> Tuple[int, int]:
	height, width = image_shape[:2]
	return max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))


def draw_label(image: Any, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.5
	thickness = 1
	(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
	x, y = origin
	box_top_left = (x, max(0, y - text_height - baseline - 4))
	box_bottom_right = (x + text_width + 6, y)
	cv2.rectangle(image, box_top_left, box_bottom_right, color, thickness=-1)
	cv2.putText(image, text, (x + 3, y - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_box(image: Any, box_xyxy: Iterable[float], color: Tuple[int, int, int], thickness: int = 2) -> None:
	x1, y1, x2, y2 = box_xyxy
	height, width = image.shape[:2]
	start = (max(0, min(int(x1), width - 1)), max(0, min(int(y1), height - 1)))
	end = (max(0, min(int(x2), width - 1)), max(0, min(int(y2), height - 1)))
	cv2.rectangle(image, start, end, color, thickness)


def annotate_detectron(image_bgr: Any) -> Dict[str, Any]:
	predictor = build_pose_detector(config_path=str(DEFAULT_DETECTRON_CONFIG_PATH))
	outputs = predictor(image_bgr)
	instances = outputs["instances"].to("cpu")

	result: Dict[str, Any] = {
		"num_instances": len(instances),
		"boxes_xyxy": [],
		"scores": [],
		"class_ids": [],
		"keypoints": [],
	}

	if not len(instances):
		return result

	boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
	scores = instances.scores.tolist() if instances.has("scores") else []
	class_ids = instances.pred_classes.tolist() if instances.has("pred_classes") else []
	keypoints = instances.pred_keypoints.tolist() if instances.has("pred_keypoints") else []

	result["boxes_xyxy"] = boxes
	result["scores"] = scores
	result["class_ids"] = class_ids
	result["keypoints"] = keypoints

	for index, box in enumerate(boxes):
		score = scores[index] if index < len(scores) else None
		class_id = class_ids[index] if index < len(class_ids) else None
		label = f"det {class_id}" if class_id is not None else "det"
		if score is not None:
			label = f"{label} {score:.2f}"
		draw_box(image_bgr, box, (0, 200, 0), 2)
		x1, y1, _, _ = box
		draw_label(image_bgr, label, clamp_point(x1, y1, image_bgr.shape), (0, 200, 0))

	for instance_keypoints in keypoints:
		for x, y, confidence in instance_keypoints:
			if confidence < 0.2:
				continue
			center = clamp_point(x, y, image_bgr.shape)
			cv2.circle(image_bgr, center, 3, (0, 255, 255), thickness=-1)

	return result


def annotate_yolo(image_bgr: Any, image_path: Path) -> Dict[str, Any]:
	result = detect_objects(str(image_path), model_path=DEFAULT_YOLO_MODEL_PATH)
	for detection in result.get("detections", []):
		draw_box(image_bgr, detection["bbox_xyxy"], (255, 0, 0), 2)
		x1, y1, _, _ = detection["bbox_xyxy"]
		label = f'{detection["class_name"]} {detection["score"]:.2f}'
		draw_label(image_bgr, label, clamp_point(x1, y1, image_bgr.shape), (255, 0, 0))
	return result


def annotate_head_pose(image_bgr: Any) -> Dict[str, Any]:
	if not torch.cuda.is_available():
		return {"error": "CUDA is not available, and head_pose.py currently requires CUDA."}

	estimator = KwanHeadPoseEstimator(weights_path=str(DEFAULT_HEAD_POSE_WEIGHTS_PATH))
	results = estimator.predict(image_bgr)
	serialized = [asdict(result) for result in results]

	for head_pose_result in results:
		box = head_pose_result.box_xyxy or head_pose_result.crop_box_xyxy
		if box is not None:
			draw_box(image_bgr, box, (0, 0, 255), 2)
			x1, y1, _, _ = box
			label = f'yaw {head_pose_result.yaw:.1f} pitch {head_pose_result.pitch:.1f} roll {head_pose_result.roll:.1f}'
			draw_label(image_bgr, label, clamp_point(x1, y1, image_bgr.shape), (0, 0, 255))

	return {"num_faces": len(results), "results": serialized}


def main() -> None:
	args = parse_args()
	image_path = Path(args.image)
	if not image_path.is_file():
		raise FileNotFoundError(f"Image not found: {image_path}")

	original_bgr = load_image_bgr(image_path)
	annotated_bgr = original_bgr.copy()

	results: Dict[str, Any] = {"image_path": str(image_path)}

	try:
		results["detectron"] = annotate_detectron(annotated_bgr)
	except Exception as error:
		results["detectron"] = {"error": str(error)}

	try:
		results["yolo"] = annotate_yolo(annotated_bgr, image_path)
	except Exception as error:
		results["yolo"] = {"error": str(error)}

	try:
		results["head_pose"] = annotate_head_pose(annotated_bgr)
	except Exception as error:
		results["head_pose"] = {"error": str(error)}

	if not cv2.imwrite(str(RESULT_IMAGE_PATH), annotated_bgr):
		raise RuntimeError(f"Failed to write result image: {RESULT_IMAGE_PATH}")

	results["result_image"] = str(RESULT_IMAGE_PATH)
	print(json.dumps(results, indent=2))


if __name__ == "__main__":
	main()
