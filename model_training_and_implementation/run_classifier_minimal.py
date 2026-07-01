import argparse
import json
import os

import cv2
import numpy as np

from src.yolo_objects import YOLOObjectDetector
from src.detectron_keypoints import DetectronKeypointDetector
from src.head_pose import KwanHeadPoseEstimator
from src.kwan_classifier import KwanLocalizedMLPClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-json", default="outputs/predictions.json")
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--yolo-classes", default="cup,bottle,cell phone,book,apple")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--mlp-weights", default="kwan_pretrained_weights/MLP_localized.pth")
    parser.add_argument("--head-pose-weights", default="kwan_pretrained_weights/head-pose-pretrained.pkl")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    object_detector = YOLOObjectDetector(
        weights=args.yolo_weights,
        class_names=args.yolo_classes,
        confidence=args.confidence,
    )

    keypoint_detector = DetectronKeypointDetector(
        confidence=args.confidence,
    )

    head_pose_estimator = KwanHeadPoseEstimator(
        weights_path=args.head_pose_weights,
    )

    classifier = KwanLocalizedMLPClassifier(
        weights_path=args.mlp_weights,
        threshold=0.5,
    )

    objects = object_detector.predict(image)
    people = keypoint_detector.predict(image)
    head_pose = head_pose_estimator.predict_first_or_sentinel(image)


    result = classifier.predict(
        objects=objects,
        people=people,
        head_pose=head_pose,
    )

    result["image"] = args.image
    result["num_objects"] = len(objects)
    result["num_people"] = len(people)
    result["head_pose"] = head_pose.tolist()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()