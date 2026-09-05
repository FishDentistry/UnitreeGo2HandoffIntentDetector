#!/usr/bin/env python3
"""
Deployment wrapper for the fine-tuned Orient Anything Go2 forward estimator.

Default checkpoint:
  robot_fov_estimation/outputs/orient_anything_forward_estimation_weights/
      best_orient_anything_forward_estimation.pt

The learned model itself uses only the Quest RGB image. It predicts:

  alpha = SignedAngle(camera_to_robot_planar, robot_forward_planar, +Y)

So image-only inference returns signed relative yaw alpha in [-180, 180).

To convert alpha into a robot-forward unit vector in the Quest/Unity world
frame, also provide ONE Quest-side source of the camera-to-robot bearing:

  1) camera_position_world + robot_center_world   (preferred)
  2) camera_to_robot_world                       (already-computed bearing)
  3) camera_rotation_world_xyzw + camera_intrinsics
     (uses the detected YOLO box center as an image bearing; no depth required)

No robot-forward marker, ground-truth yaw, training metadata, or robot-side
telemetry is required at deployment.

The wrapper reproduces the training pipeline from the saved checkpoint:
Quest image -> Go2 YOLO -> padded crop -> optional JPEG round-trip -> DINO
processor -> fine-tuned OA -> saved yaw-map -> circular-mean/argmax decoding.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoImageProcessor


# -----------------------------------------------------------------------------
# Paths / configuration
# -----------------------------------------------------------------------------

def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if parent.name == "UnitreeGo2HandoffIntentDetector":
            return parent.resolve()
        candidate = parent / "UnitreeGo2HandoffIntentDetector"
        if candidate.is_dir():
            return candidate.resolve()
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "robot_fov_estimation/outputs/orient_anything_forward_estimation_weights"
    / "best_orient_anything_forward_estimation.pt"
)
DEFAULT_OA_DIR = REPO_ROOT / "Orient-Anything"

OA_IN_DIM = {
    "small": 384,
    "base": 768,
    "large": 1024,
}

ImageInput = Union[Image.Image, np.ndarray, bytes, bytearray, str, Path]


# -----------------------------------------------------------------------------
# Public data types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    # Calibration resolution. If provided, intrinsics are scaled to the actual
    # image resolution supplied to predict().
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class OrientationEstimate:
    robot_detected: bool
    status: str

    # Selected deployment output.
    relative_yaw_deg: Optional[float]
    decoder: str

    # Both decoders are exposed for diagnostics.
    argmax_yaw_deg: Optional[float]
    mean_yaw_deg: Optional[float]
    max_probability: Optional[float]
    concentration: Optional[float]

    # Detection/crop diagnostics.
    detector_score: Optional[float]
    raw_box_xyxy: Optional[Tuple[int, int, int, int]]
    crop_box_xyxy: Optional[Tuple[int, int, int, int]]

    # Filled only if a Quest-world camera-to-robot bearing can be determined.
    camera_to_robot_world_unit: Optional[Tuple[float, float, float]]
    robot_forward_world_unit: Optional[Tuple[float, float, float]]
    world_bearing_source: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def wrap_deg(x: float) -> float:
    return (float(x) + 180.0) % 360.0 - 180.0


def normalize_xz(v: Sequence[float], name: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be 3 finite values; got {a}")
    out = np.array([a[0], 0.0, a[2]], dtype=np.float64)
    n = math.hypot(float(out[0]), float(out[2]))
    if n < 1e-10:
        raise ValueError(f"{name} has near-zero XZ magnitude")
    return out / n


def rotate_xz_unity(v: Sequence[float], yaw_deg: float) -> np.ndarray:
    """
    Rotate around Unity +Y. Positive 90 deg maps +Z to +X, matching:
      Vector3.SignedAngle(from, to, Vector3.up)
    """
    v = normalize_xz(v, "bearing")
    a = math.radians(float(yaw_deg))
    c, s = math.cos(a), math.sin(a)
    x, z = float(v[0]), float(v[2])
    return normalize_xz(
        (c * x + s * z, 0.0, -s * x + c * z),
        "robot forward",
    )


def rotate_vector_quat_xyzw(
    v: Sequence[float],
    q_xyzw: Sequence[float],
) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)
    if v.shape != (3,) or q.shape != (4,):
        raise ValueError("Expected vector=(3,) and quaternion=(x,y,z,w)")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(q)):
        raise ValueError("Vector/quaternion contains non-finite values")
    qn = float(np.linalg.norm(q))
    if qn < 1e-10:
        raise ValueError("Quaternion has near-zero magnitude")
    q = q / qn
    xyz, w = q[:3], float(q[3])
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def coerce_intrinsics(
    value: Union[CameraIntrinsics, Dict[str, Any], Sequence[float]],
) -> CameraIntrinsics:
    if isinstance(value, CameraIntrinsics):
        return value
    if isinstance(value, dict):
        return CameraIntrinsics(
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            cx=float(value["cx"]),
            cy=float(value["cy"]),
            width=int(value["width"]) if value.get("width") is not None else None,
            height=int(value["height"]) if value.get("height") is not None else None,
        )
    vals = list(value)
    if len(vals) == 4:
        return CameraIntrinsics(*map(float, vals))
    if len(vals) == 6:
        return CameraIntrinsics(
            float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]),
            int(vals[4]), int(vals[5]),
        )
    raise ValueError("Intrinsics must contain fx,fy,cx,cy[,width,height]")


def world_ray_from_pixel(
    u: float,
    v: float,
    image_size: Tuple[int, int],
    camera_rotation_world_xyzw: Sequence[float],
    camera_intrinsics: Union[CameraIntrinsics, Dict[str, Any], Sequence[float]],
) -> np.ndarray:
    """
    Pixel +u is right and +v is down.
    Unity camera-local axes are +X right, +Y up, +Z forward.
    """
    width, height = image_size
    intr = coerce_intrinsics(camera_intrinsics)
    fx, fy, cx, cy = intr.fx, intr.fy, intr.cx, intr.cy
    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive")

    if intr.width is not None and intr.height is not None:
        if intr.width <= 0 or intr.height <= 0:
            raise ValueError("Intrinsics width/height must be positive")
        sx, sy = width / intr.width, height / intr.height
        fx, cx = fx * sx, cx * sx
        fy, cy = fy * sy, cy * sy

    local = np.array(
        [(u - cx) / fx, -(v - cy) / fy, 1.0],
        dtype=np.float64,
    )
    local /= np.linalg.norm(local)
    world = rotate_vector_quat_xyzw(local, camera_rotation_world_xyzw)
    return world / np.linalg.norm(world)


# -----------------------------------------------------------------------------
# Image / crop helpers
# -----------------------------------------------------------------------------

def load_rgb_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")

    if isinstance(image, np.ndarray):
        a = np.asarray(image)
        if a.ndim != 3 or a.shape[2] not in (3, 4):
            raise ValueError("NumPy image must be HxWx3 or HxWx4")
        if a.dtype != np.uint8:
            if np.issubdtype(a.dtype, np.floating) and np.nanmin(a) >= 0 and np.nanmax(a) <= 1:
                a = np.rint(a * 255.0)
            a = np.clip(a, 0, 255).astype(np.uint8)
        if a.shape[2] == 4:
            a = a[:, :, :3]
        return Image.fromarray(a).convert("RGB")

    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(bytes(image))) as im:
            return ImageOps.exif_transpose(im).convert("RGB")

    path = Path(image).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def clamp_box(
    box_xyxy: Sequence[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must contain x1,y1,x2,y2")
    x1, y1, x2, y2 = map(float, box_xyxy)
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return x1, y1, x2, y2


def expand_box(
    box_xyxy: Sequence[float],
    width: int,
    height: int,
    padding: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    bw, bh = x2 - x1, y2 - y1
    return clamp_box(
        (x1 - bw * padding, y1 - bh * padding, x2 + bw * padding, y2 + bh * padding),
        width,
        height,
    )


def center_crop_fraction(image: Image.Image, fraction: float) -> Image.Image:
    if fraction >= 0.999999:
        return image
    if not 0 < fraction <= 1:
        raise ValueError("center_crop_fraction must be in (0,1]")
    w, h = image.size
    cw, ch = max(2, round(w * fraction)), max(2, round(h * fraction))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return image.crop((x0, y0, x0 + cw, y0 + ch))


def jpeg_round_trip(image: Image.Image, quality: int) -> Image.Image:
    """
    Training saved each YOLO crop to JPEG and later re-opened it for OA.
    This reproduces that compression step in memory.
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB").copy()


# -----------------------------------------------------------------------------
# OA helpers
# -----------------------------------------------------------------------------

def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def model_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    if isinstance(output, dict):
        for key in ("logits", "output"):
            if torch.is_tensor(output.get(key)):
                return output[key]
    raise TypeError("OA forward output did not contain tensor logits")


def remap_logits(raw: torch.Tensor, sign: int, offset: int) -> torch.Tensor:
    bins = torch.arange(360, device=raw.device, dtype=torch.long)
    source = torch.remainder(int(sign) * (bins - int(offset)), 360)
    return raw.index_select(1, source)


def decode_signed(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=1)

    argmax = torch.argmax(probs, dim=1).float()
    argmax = torch.remainder(argmax + 180.0, 360.0) - 180.0

    radians = torch.deg2rad(
        torch.arange(360, device=logits.device, dtype=logits.dtype)
    )
    c = (probs * torch.cos(radians).unsqueeze(0)).sum(dim=1)
    s = (probs * torch.sin(radians).unsqueeze(0)).sum(dim=1)
    mean = torch.rad2deg(torch.atan2(s, c))
    max_prob = probs.max(dim=1).values
    concentration = torch.sqrt(c * c + s * s)

    return argmax, mean, max_prob, concentration


# -----------------------------------------------------------------------------
# Deployment wrapper
# -----------------------------------------------------------------------------

class OrientAnythingGo2ForwardEstimator:
    """
    Load once and reuse for the entire deployment session.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        oa_dir: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "auto",
        decoder: str = "mean",
        yolo_weights_path: Optional[Union[str, Path]] = None,
        yolo_device: Optional[Union[int, str]] = None,
        match_training_jpeg: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.checkpoint_path = Path(
            checkpoint_path or DEFAULT_CHECKPOINT
        ).expanduser().resolve()
        self.oa_dir = Path(oa_dir or DEFAULT_OA_DIR).expanduser().resolve()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Fine-tuned OA checkpoint not found: {self.checkpoint_path}"
            )
        if not (self.oa_dir / "vision_tower.py").is_file():
            raise FileNotFoundError(
                f"Expected Orient Anything vision_tower.py under: {self.oa_dir}"
            )

        self.device = (
            device
            if isinstance(device, torch.device)
            else torch.device(
                "cuda" if str(device) == "auto" and torch.cuda.is_available()
                else "cpu" if str(device) == "auto"
                else str(device)
            )
        )

        self.decoder = str(decoder).lower()
        if self.decoder not in {"mean", "argmax"}:
            raise ValueError("decoder must be 'mean' or 'argmax'")

        try:
            ckpt = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise TypeError("Checkpoint must be a dictionary")
        self.checkpoint = ckpt

        if ckpt.get("target_mode") != "signed":
            raise RuntimeError(
                f"Deployment requires target_mode='signed'; got {ckpt.get('target_mode')!r}"
            )

        self.oa_scale = str(ckpt["oa_scale"])
        self.oa_output_dim = int(ckpt["oa_output_dim"])
        self.dino_model = str(ckpt["dino_model"])
        self.yaw_map_sign = int(ckpt["yaw_map_sign"])
        self.yaw_map_offset_deg = int(ckpt["yaw_map_offset_deg"]) % 360

        self.center_crop_fraction = float(ckpt.get("center_crop_fraction", 1.0))
        self.yolo_crop_enabled = bool(ckpt.get("yolo_crop_enabled", True))
        self.yolo_class_name = str(ckpt.get("yolo_class_name", "robot dog"))
        self.yolo_crop_padding = float(ckpt.get("yolo_crop_padding", 0.15))
        self.yolo_confidence = float(ckpt.get("yolo_confidence", 0.05))
        self.yolo_iou = float(ckpt.get("yolo_iou", 0.50))
        self.yolo_image_size = int(ckpt.get("yolo_image_size", 640))

        training_args = ckpt.get("training_args", {})
        training_args = training_args if isinstance(training_args, dict) else {}
        self.jpeg_quality = int(training_args.get("jpeg_quality", 95))
        self.yolo_device = (
            yolo_device if yolo_device is not None else training_args.get("yolo_device", 0)
        )
        self.match_training_jpeg = bool(match_training_jpeg)

        saved_yolo_path = ckpt.get("robot_obj_det_weights_path")
        chosen_yolo_path = yolo_weights_path if yolo_weights_path is not None else saved_yolo_path
        self.yolo_weights_path = (
            Path(chosen_yolo_path).expanduser().resolve()
            if chosen_yolo_path
            else None
        )
        if self.yolo_weights_path is not None and not self.yolo_weights_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights recorded/requested but not found: {self.yolo_weights_path}"
            )

        # Current training used the saved slow DINO processor. Pin use_fast=False
        # so a future Transformers default change cannot alter deployment input.
        self.processor = AutoImageProcessor.from_pretrained(
            self.dino_model,
            local_files_only=bool(local_files_only),
            use_fast=False,
        )

        self.model = self._load_model()
        self.detector = self._load_detector() if self.yolo_crop_enabled else None

    def _load_model(self):
        if self.oa_scale not in OA_IN_DIM:
            raise RuntimeError(f"Unsupported OA scale: {self.oa_scale}")
        if self.oa_output_dim < 360:
            raise RuntimeError(f"OA output dimension must be >=360, got {self.oa_output_dim}")

        if str(self.oa_dir) not in sys.path:
            sys.path.insert(0, str(self.oa_dir))
        from vision_tower import DINOv2_MLP

        model = DINOv2_MLP(
            dino_mode=self.oa_scale,
            in_dim=OA_IN_DIM[self.oa_scale],
            out_dim=self.oa_output_dim,
            evaluate=True,
            mask_dino=False,
            frozen_back=False,
        )

        state = strip_module_prefix(self.checkpoint["model_state_dict"])
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint/model mismatch. "
                f"Missing={incompatible.missing_keys[:10]} "
                f"Unexpected={incompatible.unexpected_keys[:10]}"
            )

        return model.to(self.device).eval()

    def _load_detector(self):
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from robot_fov_estimation.src.go2_yolo_det_wrapper import YOLOGo2Detector

        kwargs = {
            "confidence": self.yolo_confidence,
            "iou_threshold": self.yolo_iou,
            "image_size": self.yolo_image_size,
            "device": self.yolo_device,
        }
        if self.yolo_weights_path is not None:
            kwargs["weights_path"] = str(self.yolo_weights_path)
        return YOLOGo2Detector(**kwargs)

    def _best_detection(self, image_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        detections = self.detector.predict(image_rgb, [self.yolo_class_name])
        valid = [
            d for d in (detections or [])
            if d.get("box_xyxy") is not None and d.get("score") is not None
        ]
        return max(valid, key=lambda d: float(d["score"])) if valid else None

    def _prepare_oa_image(
        self,
        image: Image.Image,
        robot_box_xyxy: Optional[Sequence[float]],
    ):
        width, height = image.size
        raw_box = crop_box = None
        detector_score = None

        if self.yolo_crop_enabled:
            if robot_box_xyxy is not None:
                raw_box = clamp_box(robot_box_xyxy, width, height)
            else:
                detection = self._best_detection(np.asarray(image, dtype=np.uint8))
                if detection is None:
                    return None, None, None, None
                raw_box = clamp_box(detection["box_xyxy"], width, height)
                detector_score = float(detection["score"])

            crop_box = expand_box(
                raw_box,
                width,
                height,
                self.yolo_crop_padding,
            )
            oa_image = image.crop(crop_box)
            if self.match_training_jpeg:
                oa_image = jpeg_round_trip(oa_image, self.jpeg_quality)
        else:
            oa_image = image
            if robot_box_xyxy is not None:
                raw_box = clamp_box(robot_box_xyxy, width, height)

        oa_image = center_crop_fraction(oa_image, self.center_crop_fraction)
        return oa_image, raw_box, crop_box, detector_score

    @torch.inference_mode()
    def _infer_crop(self, image: Image.Image) -> Tuple[float, float, float, float]:
        pixels = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].to(self.device)

        raw = model_logits(self.model({"pixel_values": pixels}))[:, :360]
        mapped = remap_logits(raw, self.yaw_map_sign, self.yaw_map_offset_deg)
        argmax, mean, max_prob, concentration = decode_signed(mapped)

        return (
            wrap_deg(float(argmax[0].item())),
            wrap_deg(float(mean[0].item())),
            float(max_prob[0].item()),
            float(concentration[0].item()),
        )

    def predict_relative_yaw(
        self,
        image: ImageInput,
        *,
        robot_box_xyxy: Optional[Sequence[float]] = None,
    ) -> OrientationEstimate:
        """Image-only deployment inference."""
        return self.predict(image, robot_box_xyxy=robot_box_xyxy)

    def predict(
        self,
        image: ImageInput,
        *,
        robot_box_xyxy: Optional[Sequence[float]] = None,
        camera_to_robot_world: Optional[Sequence[float]] = None,
        camera_position_world: Optional[Sequence[float]] = None,
        robot_center_world: Optional[Sequence[float]] = None,
        camera_rotation_world_xyzw: Optional[Sequence[float]] = None,
        camera_intrinsics: Optional[
            Union[CameraIntrinsics, Dict[str, Any], Sequence[float]]
        ] = None,
    ) -> OrientationEstimate:
        """
        Predict signed relative yaw and, when possible, Quest-world forward.

        World-bearing priority:
          1. camera_to_robot_world
          2. camera_position_world + robot_center_world
          3. camera_rotation_world_xyzw + camera_intrinsics + YOLO box center
        """
        full_image = load_rgb_image(image)
        width, height = full_image.size

        oa_image, raw_box, crop_box, detector_score = self._prepare_oa_image(
            full_image,
            robot_box_xyxy,
        )

        if oa_image is None:
            return OrientationEstimate(
                robot_detected=False,
                status="robot_not_detected",
                relative_yaw_deg=None,
                decoder=self.decoder,
                argmax_yaw_deg=None,
                mean_yaw_deg=None,
                max_probability=None,
                concentration=None,
                detector_score=None,
                raw_box_xyxy=None,
                crop_box_xyxy=None,
                camera_to_robot_world_unit=None,
                robot_forward_world_unit=None,
                world_bearing_source=None,
            )

        argmax, mean, max_prob, concentration = self._infer_crop(oa_image)
        selected = mean if self.decoder == "mean" else argmax

        bearing = None
        source = None

        if camera_to_robot_world is not None:
            bearing = normalize_xz(camera_to_robot_world, "camera_to_robot_world")
            source = "supplied_camera_to_robot_world"

        elif camera_position_world is not None or robot_center_world is not None:
            if camera_position_world is None or robot_center_world is None:
                raise ValueError(
                    "camera_position_world and robot_center_world must be supplied together"
                )
            camera = np.asarray(camera_position_world, dtype=np.float64).reshape(-1)
            robot = np.asarray(robot_center_world, dtype=np.float64).reshape(-1)
            if camera.shape != (3,) or robot.shape != (3,):
                raise ValueError("camera_position_world and robot_center_world must be 3-vectors")
            bearing = normalize_xz(
                robot - camera,
                "robot_center_world - camera_position_world",
            )
            source = "quest_world_camera_and_robot_positions"

        elif camera_rotation_world_xyzw is not None or camera_intrinsics is not None:
            if camera_rotation_world_xyzw is None or camera_intrinsics is None:
                raise ValueError(
                    "camera_rotation_world_xyzw and camera_intrinsics must be supplied together"
                )
            if raw_box is None:
                raise RuntimeError(
                    "Cannot infer an image bearing because no robot box is available"
                )

            x1, y1, x2, y2 = raw_box
            u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            ray_world = world_ray_from_pixel(
                u,
                v,
                (width, height),
                camera_rotation_world_xyzw,
                camera_intrinsics,
            )
            bearing = normalize_xz(ray_world, "back-projected robot image ray")
            source = "yolo_box_center_plus_quest_camera_pose_intrinsics"

        forward = rotate_xz_unity(bearing, selected) if bearing is not None else None

        return OrientationEstimate(
            robot_detected=True,
            status="ok",
            relative_yaw_deg=selected,
            decoder=self.decoder,
            argmax_yaw_deg=argmax,
            mean_yaw_deg=mean,
            max_probability=max_prob,
            concentration=concentration,
            detector_score=detector_score,
            raw_box_xyxy=raw_box,
            crop_box_xyxy=crop_box,
            camera_to_robot_world_unit=(
                tuple(map(float, bearing)) if bearing is not None else None
            ),
            robot_forward_world_unit=(
                tuple(map(float, forward)) if forward is not None else None
            ),
            world_bearing_source=source,
        )

    def checkpoint_summary(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "decoder": self.decoder,
            "oa_scale": self.oa_scale,
            "dino_model": self.dino_model,
            "yaw_map_sign": self.yaw_map_sign,
            "yaw_map_offset_deg": self.yaw_map_offset_deg,
            "yolo_crop_enabled": self.yolo_crop_enabled,
            "yolo_class_name": self.yolo_class_name,
            "yolo_crop_padding": self.yolo_crop_padding,
            "yolo_confidence": self.yolo_confidence,
            "yolo_iou": self.yolo_iou,
            "yolo_image_size": self.yolo_image_size,
            "yolo_weights_path": (
                str(self.yolo_weights_path) if self.yolo_weights_path else None
            ),
            "center_crop_fraction": self.center_crop_fraction,
            "match_training_jpeg": self.match_training_jpeg,
            "jpeg_quality": self.jpeg_quality,
            "train_session_ids": self.checkpoint.get("train_session_ids"),
        }


# -----------------------------------------------------------------------------
# Minimal command-line smoke test
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deployment OA wrapper on one Quest image"
    )
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--oa-dir", default=str(DEFAULT_OA_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decoder", choices=("mean", "argmax"), default="mean")
    parser.add_argument("--yolo-weights-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-training-jpeg-roundtrip", action="store_true")

    parser.add_argument("--camera-position-world", nargs=3, type=float)
    parser.add_argument("--robot-center-world", nargs=3, type=float)
    parser.add_argument("--camera-to-robot-world", nargs=3, type=float)
    parser.add_argument("--camera-rotation-world-xyzw", nargs=4, type=float)
    parser.add_argument(
        "--intrinsics",
        nargs=4,
        type=float,
        metavar=("FX", "FY", "CX", "CY"),
    )
    parser.add_argument(
        "--intrinsics-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
    )
    args = parser.parse_args()

    estimator = OrientAnythingGo2ForwardEstimator(
        args.checkpoint,
        oa_dir=args.oa_dir,
        device=args.device,
        decoder=args.decoder,
        yolo_weights_path=args.yolo_weights_path,
        match_training_jpeg=not args.no_training_jpeg_roundtrip,
        local_files_only=args.local_files_only,
    )

    intrinsics = None
    if args.intrinsics:
        fx, fy, cx, cy = args.intrinsics
        width, height = (args.intrinsics_size or (None, None))
        intrinsics = CameraIntrinsics(fx, fy, cx, cy, width, height)

    result = estimator.predict(
        args.image,
        camera_to_robot_world=args.camera_to_robot_world,
        camera_position_world=args.camera_position_world,
        robot_center_world=args.robot_center_world,
        camera_rotation_world_xyzw=args.camera_rotation_world_xyzw,
        camera_intrinsics=intrinsics,
    )

    print(json.dumps(
        {
            "model": estimator.checkpoint_summary(),
            "prediction": result.to_dict(),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
