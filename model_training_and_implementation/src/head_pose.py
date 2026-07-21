from types import SimpleNamespace
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from mtcnn.mtcnn import MTCNN

from src.kwan_headpose import module_init, head_pose_estimation


@dataclass
class HeadPoseResult:
    yaw: float
    pitch: float
    roll: float
    box_xyxy: Optional[list]
    crop_box_xyxy: Optional[list]


class KwanHeadPoseEstimator:
    def __init__(
        self,
        weights_path: str = "kwan_pretrained_weights/head-pose-pretrained.pkl",
        gpu_id: int = 0,
    ):
        cfg = SimpleNamespace(
            HEAD_POSE=SimpleNamespace(
                PRETRAINED=weights_path,
                GPU_ID=gpu_id,
            )
        )

        self.model = module_init(cfg)
        self.mtcnn = MTCNN()

        self.transformations = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.softmax = nn.Softmax(dim=1).cuda()
        self.idx_tensor = torch.FloatTensor([idx for idx in range(66)]).cuda()

    def predict(self, image_bgr: np.ndarray) -> List[HeadPoseResult]:
        # predictions, bounding_boxes, face_keypoints, widths, face_areas = head_pose_estimation(
        #     image_bgr,
        #     self.mtcnn,
        #     self.model,
        #     self.transformations,
        #     self.softmax,
        #     self.idx_tensor,
        # )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        predictions, bounding_boxes, face_keypoints, widths, face_areas = head_pose_estimation(
            image_rgb,
            self.mtcnn,
            self.model,
            self.transformations,
            self.softmax,
            self.idx_tensor,
        )

        results = []
        for i, pred in enumerate(predictions):
            # bounding_boxes from MTCNN are [x, y, w, h]
            box_xywh = bounding_boxes[i] if i < len(bounding_boxes) else None
            box_xyxy = None
            if box_xywh is not None:
                x, y, w, h = box_xywh
                box_xyxy = [float(x), float(y), float(x + w), float(y + h)]

            crop_box = None
            if i < len(face_keypoints):
                crop_box = [float(v) for v in face_keypoints[i]]

            results.append(
                HeadPoseResult(
                    yaw=float(pred[0]),
                    pitch=float(pred[1]),
                    roll=float(pred[2]),
                    box_xyxy=box_xyxy,
                    crop_box_xyxy=crop_box,
                )
            )

        return results

    def predict_first_or_sentinel(self, image_bgr: np.ndarray) -> np.ndarray:
        results = self.predict(image_bgr)

        if not results:
            return np.asarray([-999.0, -999.0, -999.0], dtype=np.float32)

        H, W = image_bgr.shape[:2]

        valid = []
        for r in results:
            if r.box_xyxy is None:
                continue

            x1, y1, x2, y2 = r.box_xyxy
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            # reject tiny detections
            if bw < 25 or bh < 25:
                continue

            # reject boxes that are implausibly small relative to image
            if area < 0.0005 * W * H:
                continue

            valid.append((area, r))

        if not valid:
            return np.asarray([-999.0, -999.0, -999.0], dtype=np.float32)

        # Use largest detected face/head-ish region
        _, r = max(valid, key=lambda t: t[0])

        return np.asarray([r.yaw, r.pitch, r.roll], dtype=np.float32)