from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Boxes, Instances
from detectron2.config import CfgNode as CN

import numpy as np


KEYPOINT_CONFIG_PATH = "./configs/keypoint_rcnn_R_101_FPN_3x.yaml"
KEYPOINT_WEIGHTS_PATH = (
    "detectron2://COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl"
)
HEAD_POSE_PRETRAINED_PATH = "./pretrained-weights/head-pose-pretrained.pkl"
MLP_LOCALIZED_PATH = "./pretrained-weights/MLP_localized.pth"
MLP_NONLOCALIZED_PATH = "./pretrained-weights/MLP_nonlocalized.pth"


def add_custom_config(
    cfg,
    head_pose_pretrained_path: str = HEAD_POSE_PRETRAINED_PATH,
    mlp_localized_path: str = MLP_LOCALIZED_PATH,
    mlp_nonlocalized_path: str = MLP_NONLOCALIZED_PATH,
):
    """
    Add config for head pose estimation
    """
    _C = cfg

    _C.HEAD_POSE = CN()
    _C.HEAD_POSE.PRETRAINED = head_pose_pretrained_path
    _C.HEAD_POSE.GPU_ID = 0

    _C.MLP = CN()
    _C.MLP.PRETRAINED = mlp_localized_path
    _C.MLP.PRETRAINED_NONLOCALIZED = mlp_nonlocalized_path


def build_pose_detector(
    config_path: str = KEYPOINT_CONFIG_PATH,
    weights_path: str = KEYPOINT_WEIGHTS_PATH,
    score_thresh: float = 0.85,
    head_pose_pretrained_path: str = HEAD_POSE_PRETRAINED_PATH,
    mlp_localized_path: str = MLP_LOCALIZED_PATH,
    mlp_nonlocalized_path: str = MLP_NONLOCALIZED_PATH,
):
    """Build and freeze the Detectron2 pose detector."""

    cfg_keypoint = get_cfg()
    add_custom_config(
        cfg_keypoint,
        head_pose_pretrained_path=head_pose_pretrained_path,
        mlp_localized_path=mlp_localized_path,
        mlp_nonlocalized_path=mlp_nonlocalized_path,
    )
    cfg_keypoint.merge_from_file(config_path)
    cfg_keypoint.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg_keypoint.MODEL.WEIGHTS = weights_path
    cfg_keypoint.freeze()
    return DefaultPredictor(cfg_keypoint)


class DetectronKeypointDetector:
    def __init__(
        self,
        confidence: float = 0.85,
        config_path: str = KEYPOINT_CONFIG_PATH,
        weights_path: str = KEYPOINT_WEIGHTS_PATH,
        head_pose_pretrained_path: str = HEAD_POSE_PRETRAINED_PATH,
        mlp_localized_path: str = MLP_LOCALIZED_PATH,
        mlp_nonlocalized_path: str = MLP_NONLOCALIZED_PATH,
    ):
        self.predictor = build_pose_detector(
            config_path=config_path,
            weights_path=weights_path,
            score_thresh=confidence,
            head_pose_pretrained_path=head_pose_pretrained_path,
            mlp_localized_path=mlp_localized_path,
            mlp_nonlocalized_path=mlp_nonlocalized_path,
        )

    def predict(self, image_bgr):
        outputs = self.predictor(image_bgr)
        instances = outputs["instances"].to("cpu")

        people = []
        if not len(instances):
            return people

        boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
        scores = instances.scores.tolist() if instances.has("scores") else []
        keypoints = instances.pred_keypoints.tolist() if instances.has("pred_keypoints") else []

        for index, keypoint_set in enumerate(keypoints):
            person = {
                "box_xyxy": boxes[index] if index < len(boxes) else None,
                "score": scores[index] if index < len(scores) else None,
                "keypoints": keypoint_set,
                "keypoints_xy": [[float(x), float(y)] for x, y, *_ in keypoint_set],
            }
            people.append(person)

        return people


#pose_detector = build_pose_detector()
