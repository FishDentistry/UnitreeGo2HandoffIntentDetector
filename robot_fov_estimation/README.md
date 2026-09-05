# Robot FoV Estimation

This directory contains the robot-vision components used to:

1. detect a Unitree Go2 in Quest RGB images,
2. track the Go2 between detector updates,
3. fine-tune **Orient Anything V1** to estimate the Go2's horizontal forward direction, and
4. run the fine-tuned orientation model at deployment.

The orientation model predicts the signed horizontal angle

\[
\alpha = \operatorname{SignedAngle}
(\text{camera-to-robot}_{XZ},\ \text{robot-forward}_{XZ},\ +Y)
\]

with output in `[-180°, 180°)`:

- `0°`: the robot points approximately away from the camera,
- `±180°`: the robot points approximately toward the camera,
- `±90°`: side view.

The learned model does **not** require robot telemetry at deployment. Image-only inference produces the relative yaw above. Quest-side geometry is only needed if that relative yaw must be converted to a Quest-world forward vector.

---

## 1. Directory map

The pipeline-relevant layout is:

```text
UnitreeGo2HandoffIntentDetector/
├── robot_fov_estimation/
│   ├── README.md
│   ├── data/
│   │   ├── go2_yolo_det_data/
│   │   └── forward_estimation_data/
│   ├── model_training/
│   │   ├── train_go2_yolo_det.py
│   │   └── finetune_orient_anything_forward_estimation.py
│   ├── outputs/
│   │   ├── go2_yolo_det_weights/
│   │   ├── eval_results/
│   │   ├── orient_anything_forward_estimation_eval/
│   │   └── orient_anything_forward_estimation_weights/
│   ├── src/
│   │   ├── go2_yolo_det_wrapper.py
│   │   ├── robot_detector_tracker.py
│   │   └── orient_anything_forward_estimation_wrapper.py
│   └── ostrack/                       # OSTrack checkout, if kept locally here
│
├── Orient-Anything/                   # EXTERNAL checkout; not part of this repo
└── servers/
    ├── collect_orientation_est_data.py
    └── check_current_handoff_pose_server.py
```

### What each component does

| Component | Purpose |
|---|---|
| `train_go2_yolo_det.py` | Fine-tunes the Go2 object detector. |
| `go2_yolo_det_wrapper.py` | Loads the detector and returns Go2 bounding boxes. |
| `robot_detector_tracker.py` | Combines YOLO detection with OSTrack for live tracking/reacquisition. |
| `finetune_orient_anything_forward_estimation.py` | Fine-tunes Orient Anything on Quest-native Go2 orientation data and performs LOSO evaluation. |
| `orient_anything_forward_estimation_wrapper.py` | Deployment wrapper for the final fine-tuned OA checkpoint. |
| `data/go2_yolo_det_data/` | YOLO training data. |
| `data/forward_estimation_data/` | Quest-native orientation collection sessions. |
| `outputs/` | Generated weights, predictions, diagnostics, and evaluation results. |

`Orient-Anything/` is intentionally treated as a third-party dependency and should not be assumed to be committed with this repository.

---

# 2. Reproducing the environment

## 2.1 Python / CUDA

Use a CUDA-enabled PyTorch environment for training and live deployment.

A known working development environment for this project was:

```text
Python 3.9
PyTorch 2.4.x + CUDA 12.1
NVIDIA GPU
```

Exact PyTorch/CUDA installation is machine-specific. Install the appropriate PyTorch build for the target GPU before installing the remaining Python dependencies.

Core packages used by this pipeline include:

```text
torch
torchvision
numpy
Pillow
transformers
huggingface-hub
ultralytics
opencv-python
```

Live server use additionally requires packages such as:

```text
fastapi
uvicorn
```

OSTrack has its own dependency setup described below.

For strict reproducibility, the repository should also include a pinned `requirements.txt`, Conda environment file, or lockfile. The README documents the required components, but a dependency lock is still preferable for archival reproduction.

---

# 3. Install Orient Anything V1

This project uses **Orient Anything V1**, not Orient Anything V2.

Official source:

```text
https://github.com/SpatialVision/Orient-Anything
```

The training and deployment code expects the checkout at:

```text
<repo-root>/Orient-Anything/
```

Install it from the repository root:

```bash
git clone https://github.com/SpatialVision/Orient-Anything.git Orient-Anything
```

The important file used directly by this project is:

```text
Orient-Anything/vision_tower.py
```

The project imports `DINOv2_MLP` from that source tree. The final fine-tuned checkpoint does **not** replace the need for this source file.

## 3.1 OA Python dependencies

The official OA repository currently provides its own `requirements.txt` and recommends:

```bash
pip install -r Orient-Anything/requirements.txt
```

However, that file pins its own PyTorch version. If this project is being installed into an existing CUDA environment, check that requirement before allowing pip to replace the project's working PyTorch build.

The OA code path used here primarily depends on:

```text
torch
transformers
numpy
Pillow
huggingface-hub
```

The official OA demo additionally uses packages such as Gradio, ONNX Runtime, and `rembg`.

## 3.2 OA pretrained weights

The training script downloads the upstream OA V1 checkpoint from Hugging Face on first use:

```text
Viglong/Orient-Anything
```

Default model mapping:

| `--oa-scale` | DINO backbone | OA checkpoint |
|---|---|---|
| `small` | `facebook/dinov2-small` | `cropsmallEx03/dino_weight.pt` |
| `base` | `facebook/dinov2-base` | `cropbaseEx03/dino_weight.pt` |
| `large` | `facebook/dinov2-large` | `croplargeEX2/dino_weight.pt` |

The DINO image processor/model metadata are also obtained through Hugging Face/Transformers.

Therefore, the first training or deployment run normally requires internet access unless the necessary Hugging Face files have already been cached.

For offline deployment, pre-cache the Hugging Face assets and use the wrapper's `--local-files-only` option.


---

# 4. Go2 YOLO detector

The orientation pipeline expects a Go2 bounding box before OA inference. The same detector is used during training and deployment.

## 4.1 Dataset layout

Place the Ultralytics/Roboflow-style dataset under:

```text
robot_fov_estimation/data/go2_yolo_det_data/
```

The dataset must contain exactly one `data.yaml` discoverable by the training script. It should define the training/validation paths, optional test path, and class names.

The detector is trained as a single-class robot detector.

## 4.2 Train

From the repository root:

```bash
python -m robot_fov_estimation.model_training.train_go2_yolo_det
```

The current training script uses a pretrained Ultralytics detector as its starting point and writes trained weights under:

```text
robot_fov_estimation/outputs/go2_yolo_det_weights/
```

Evaluation artifacts are written under:

```text
robot_fov_estimation/outputs/eval_results/go2_yolo/
```

Useful modes include:

```bash
--resume-training
--eval-only
```

## 4.3 Detector wrapper

Deployment code should normally use:

```python
from robot_fov_estimation.src.go2_yolo_det_wrapper import YOLOGo2Detector

detector = YOLOGo2Detector()
detections = detector.predict(image_rgb, ["robot dog"])
```

A detection dictionary contains fields such as:

```text
label
score
box_xyxy
```

The OA trainer's current deployment-matched defaults are:

```text
class name:   robot dog
confidence:   0.05
IoU:          0.50
image size:   640
crop padding: 0.15
```

If detector weights are not stored at the wrapper's default path, pass an explicit path.

---

# 5. OSTrack

OSTrack is used only for temporal robot tracking. 

The project adapter is:

```text
robot_fov_estimation/src/robot_detector_tracker.py
```

This repo includes a modified OSTrack checkout that works with this.


The current adapter defaults to the configuration:

```text
vitb_256_mae_ce_32x4_ep300
```

and expects the model checkpoint at the corresponding OSTrack output path, typically:

```text
robot_fov_estimation/ostrack/
    output/checkpoints/train/ostrack/
        vitb_256_mae_ce_32x4_ep300/
            OSTrack_ep0300.pth.tar
```

If your checkout or checkpoint lives elsewhere, pass the explicit paths to `OSTrackAdapter`.


---

# 6. Orientation data collection

The OA fine-tuning data are Quest-native sessions stored under:

```text
robot_fov_estimation/data/forward_estimation_data/
```

Each session has:

```text
<session_id>/
├── samples.csv
└── images/
    └── frame_*.jpg
```

The current collection protocol keeps the Go2 stationary while the Quest user walks around it, providing full-around views. Multiple independent sessions are preferred so LOSO evaluation measures true cross-session generalization.

The current collection pipeline uses:

```text
Unity:  SetRobotForwardQuest.cs
Server: servers/collect_orientation_est_data.py
```

The collected Quest-world geometry is used to create the training target. It is **not** required at deployment.

## 6.1 Required CSV fields

The current trainer requires:

```text
session_id
frame_id
capture_timestamp_unix
capture_realtime_seconds
measurement_minus_capture_seconds
image_relative_path

camera_position_world_x
camera_position_world_y
camera_position_world_z

robot_center_world_x
robot_center_world_y
robot_center_world_z

robot_forward_point_world_x
robot_forward_point_world_y
robot_forward_point_world_z

signed_relative_yaw_deg
```

The trainer recomputes the signed yaw from the raw geometry and verifies it against `signed_relative_yaw_deg`.

This protects against silently training on inconsistent labels.

---

# 7. Fine-tuning Orient Anything

Training script:

```text
robot_fov_estimation/model_training/
    finetune_orient_anything_forward_estimation.py
```

Run from the repository root:

```bash
python -m robot_fov_estimation.model_training.finetune_orient_anything_forward_estimation
```

## 7.1 Training target

The default target is:

```text
signed
```

with:

\[
\alpha =
\operatorname{SignedAngle}
(\text{camera-to-robot}_{XZ},
 \text{robot-forward}_{XZ},
 +Y)
\]

Optional `magnitude` and `body_axis` modes exist only for diagnostic experiments. Deployment expects a checkpoint trained with:

```text
target_mode = signed
```

## 7.2 Deployment-matched image preprocessing

Before OA sees a training image:

1. Load the original Quest image.
2. Run `YOLOGo2Detector`.
3. Select the highest-confidence valid `"robot dog"` box.
4. Expand the box by the configured padding, default `0.15`.
5. Crop from the original-resolution image.
6. Save/use the crop consistently for all train/validation/test passes.
7. Apply the DINO image processor.
8. Feed the crop to OA.

If YOLO misses the robot, the default behavior is:

```text
skip
```

The trainer does **not** silently substitute the full image.

This preprocessing must remain consistent at deployment.

## 7.3 Default training configuration

Important defaults:

```text
target mode:                  signed
OA scale:                     small
target sigma:                 10°
split:                        leave-one-session-out
training temporal thinning:   0.5 s
session-balanced sampling:    enabled

epochs:                       100
minimum epochs:               20
early-stop patience:          15
batch size:                   16

DINO/backbone LR:             1e-6
head LR:                      1e-4
weight decay:                 0.01
warmup fraction:              0.05
max grad norm:                1.0

AMP:                          enabled on CUDA
BatchNorm running stats:      frozen
```

Useful command-line overrides include:

```bash
--data-root PATH
--sessions session1,session2,...
--robot-obj-det-weights-path PATH
--oa-dir PATH
--oa-scale small|base|large
--oa-checkpoint PATH
--device auto|cpu|cuda
--only-test-session SESSION_ID
```

## 7.4 LOSO evaluation

The default split is complete-session leave-one-session-out (LOSO).

For each fold:

```text
one entire session -> test
one different entire session -> validation
all remaining sessions -> training
```

Training frames are temporally thinned and session-balanced by default.

This is the primary evaluation because random frame splits can leak near-duplicate neighboring frames from the same collection run.

Evaluation output is written to:

```text
robot_fov_estimation/outputs/
    orient_anything_forward_estimation_eval/
```

Typical artifacts include:

```text
dataset_configuration.json
session_data_quality.csv
yolo_crop_manifest.csv
yolo_crop_summary_by_session.csv
leave_one_session_out_folds/
aggregate / summary files
per-frame prediction CSVs
distance diagnostics
```

## 7.5 Final deployment checkpoint

After a complete LOSO run, the script automatically trains a new model on **all available orientation sessions**.

That final checkpoint is written to:

```text
robot_fov_estimation/outputs/
    orient_anything_forward_estimation_weights/
        best_orient_anything_forward_estimation.pt
```

This is the checkpoint intended for deployment.

Do not deploy one of the LOSO fold checkpoints unless intentionally testing a fold-specific model.

The final checkpoint stores the important inference metadata, including:

```text
model_state_dict
OA scale/output dimension
DINO model identifier
target mode/definition
yaw-map sign
yaw-map offset
center-crop fraction
YOLO crop enabled/disabled
YOLO class
YOLO padding
YOLO confidence
YOLO IoU
YOLO image size
training arguments
training session IDs
```

---

# 8. Deployment wrapper

Deployment wrapper:

```text
robot_fov_estimation/src/
    orient_anything_forward_estimation_wrapper.py
```

The wrapper is intentionally independent of the training dataset. It never uses:

```text
robot_forward_point_world
signed_relative_yaw_deg
ground-truth robot heading
session IDs for inference
robot-side telemetry
```

## 8.1 Basic use

```python
from robot_fov_estimation.src.orient_anything_forward_estimation_wrapper import (
    OrientAnythingGo2ForwardEstimator,
)

estimator = OrientAnythingGo2ForwardEstimator()

result = estimator.predict(frame_rgb)

if result.robot_detected:
    print(result.relative_yaw_deg)
```

Load the estimator **once** and reuse it across frames.

The default checkpoint is:

```text
robot_fov_estimation/outputs/
    orient_anything_forward_estimation_weights/
        best_orient_anything_forward_estimation.pt
```

The default decoder is:

```text
mean
```

which is the circular mean of the predicted 360-bin distribution.

`argmax` is also supported.

## 8.2 Accepted image inputs

`predict()` accepts:

```text
PIL.Image
RGB numpy.ndarray
JPEG/PNG bytes
image path
```

NumPy arrays must be RGB. If an image comes from OpenCV, convert BGR to RGB first.

## 8.3 Reuse an existing tracking box

If YOLO/OSTrack has already produced a robot box, pass it directly:

```python
result = estimator.predict(
    frame_rgb,
    robot_box_xyxy=(x1, y1, x2, y2),
)
```

This avoids a second YOLO pass.

The wrapper still applies the checkpoint's saved padding/crop preprocessing before OA inference.

This is the preferred live-server integration.

## 8.4 Portable YOLO weights

The final OA checkpoint may contain the detector weight path used during training.

If that path was absolute and does not exist on another machine, override it explicitly:

```python
estimator = OrientAnythingGo2ForwardEstimator(
    yolo_weights_path="/new/path/to/best.pt",
)
```

Likewise, custom OA source/checkpoint paths can be passed explicitly.

## 8.5 Important outputs

`OrientationEstimate` includes:

```text
robot_detected
status

relative_yaw_deg
argmax_yaw_deg
mean_yaw_deg
max_probability
concentration

raw_box_xyxy
crop_box_xyxy
detector_score

camera_to_robot_world_unit
robot_forward_world_unit
world_bearing_source
```

The main learned output is:

```text
relative_yaw_deg
```

---

# 9. Converting the relative yaw to Quest-world forward

The OA model estimates orientation relative to the camera-to-robot direction.

To obtain a horizontal robot-forward vector in Quest world coordinates, the wrapper needs one source of Quest-side camera-to-robot bearing.

## Preferred: known camera and robot positions

```python
result = estimator.predict(
    frame_rgb,
    robot_box_xyxy=box,
    camera_position_world=(cx, cy, cz),
    robot_center_world=(rx, ry, rz),
)

forward_world = result.robot_forward_world_unit
```

## Already-computed camera-to-robot vector

```python
result = estimator.predict(
    frame_rgb,
    robot_box_xyxy=box,
    camera_to_robot_world=(dx, dy, dz),
)
```

## Camera rotation + intrinsics

The wrapper can also estimate the bearing from the robot box center:

```python
result = estimator.predict(
    frame_rgb,
    camera_rotation_world_xyzw=(qx, qy, qz, qw),
    camera_intrinsics={
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
    },
)
```

No depth is required for this last option because only the horizontal bearing is needed.

---

# 10. Command-line smoke test

Run the deployment wrapper directly on one image:

```bash
python -m robot_fov_estimation.src.orient_anything_forward_estimation_wrapper \
    /path/to/image.jpg
```

Useful options:

```bash
--checkpoint PATH
--oa-dir PATH
--device auto
--decoder mean
--yolo-weights-path PATH
--local-files-only
--no-training-jpeg-roundtrip
```

A successful run should load:

```text
final fine-tuned checkpoint
external Orient-Anything source
DINO processor/model assets
Go2 detector
```

and return an orientation estimate.

---

# 11. Live server integration

The current deployment server is:

```text
servers/check_current_handoff_pose_server.py
```

The live pipeline is:

```text
Quest RGB frame
    -> YOLO / OSTrack robot tracking
    -> current robot bounding box
    -> OA wrapper using that existing box
    -> signed relative forward yaw
    -> response to Quest
```

The server reuses the OA wrapper's YOLO detector rather than loading a duplicate detector.

The current Unity side pairs each server response with the exact `CameraFrameCapture` that produced the submitted image.

`EstimateRobotPosAndForward` keeps the existing robot-position tracking/reacquisition behavior, but the forward visual now uses the learned server orientation estimate rather than the old motion-derived heading estimator.

Quest-side camera pose and 3-D robot position can then be used to express the returned relative yaw as a Quest-world forward vector.

---

# 12. Robot tracking behavior

`robot_detector_tracker.py` combines semantic detection and OSTrack.

Conceptually:

```text
SEARCHING
    -> YOLO finds candidate
    -> confirmed acquisition
    -> initialize OSTrack

TRACKING
    -> OSTrack predicts every frame
    -> YOLO periodically verifies/corrects track
    -> repeated verification failure
    -> return to global search
```

The tracker returns fields such as:

```text
bbox_xyxy
tracker_verified
detector_score
source
frame_id
processing_time_ms
end_to_end_latency_ms
```

The OA wrapper should consume the returned `bbox_xyxy` directly rather than rerunning detection.

---

# 13. Timing / synchronization

Network and model inference latency do not require special camera-pose compensation as long as the response remains paired with the original capture that produced the submitted JPEG.

The current deployment code does this pairing.

The pipeline does not currently apply an explicit correction for Quest RGB exposure-to-frame-availability latency. If future testing shows that the world-space forward visual shifts systematically during rapid head rotation, that sensor/pose timing should be investigated separately.

---


# 14. Reproducibility checklist

To reproduce training from scratch, archive or publish all of the following:

- repository Git commit,
- Python dependency lock / environment file,
- orientation collection sessions,
- YOLO training dataset,
- trained YOLO checkpoint,
- external Orient Anything Git commit,
- external OSTrack Git commit,
- OSTrack checkpoint,
- training command/arguments,
- final all-data OA checkpoint.

For a deployment-only release, the minimum required pieces are:

```text
project source
Orient-Anything source checkout
fine-tuned OA deployment checkpoint
Go2 detector weights
DINO/Hugging Face assets or internet access for first download
OSTrack source + checkpoint if live tracking is used
```

Generated `outputs/` and large external model repositories should normally be stored with Git LFS, release assets, or an external artifact store rather than silently assumed to exist.

---

# 15. Third-party projects

## Orient Anything V1

Official code:

```text
https://github.com/SpatialVision/Orient-Anything
```

Official model repository:

```text
https://huggingface.co/Viglong/Orient-Anything
```

Paper:

```text
Orient Anything: Learning Robust Object Orientation Estimation from Rendering 3D Models
Wang et al., ICML 2025
```

## OSTrack

Official code:

```text
https://github.com/botaoye/OSTrack
```

Paper:

```text
Joint Feature Learning and Relation Modeling for Tracking:
A One-Stream Framework
Ye et al., ECCV 2022
```

## Ultralytics

The Go2 detector uses the Ultralytics Python package:

```bash
pip install -U ultralytics
```

See the upstream project/documentation for installation and licensing details.

---

# 16. Go2 detector dataset citation

Part of the Go2 detector dataset came from the following Roboflow dataset; additional images are custom:

```bibtex
@misc{unitree-go2-soavc_dataset,
    title        = {unitree go2 Dataset},
    type         = {Open Source Dataset},
    author       = {heejin},
    howpublished = {\url{https://universe.roboflow.com/heejin-icg8e/unitree-go2-soavc}},
    url          = {https://universe.roboflow.com/heejin-icg8e/unitree-go2-soavc},
    journal      = {Roboflow Universe},
    publisher    = {Roboflow},
    year         = {2026},
    month        = {jul},
    note         = {visited on 2026-08-03}
}
```
