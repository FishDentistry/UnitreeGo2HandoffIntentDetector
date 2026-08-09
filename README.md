# UnitreeGo2HandoffIntentDetector

A collection of training and deployment resources for **human-to-robot handoff intent recognition**, **communication-free robot-belief estimation and AR pose guidance**, and **robot detection/tracking for augmented-reality field-of-view visualization**.

Although the current robot deployment is built around the **Unitree Go2**, the handoff intent recognition pipeline is intended to be applicable to other robots equipped with an RGB-D/depth camera.

---

## Overview

This repository contains three primary components:

1. **Robot-side handoff intent recognition**

   * Trains and deploys a model that determines whether a human is currently attempting to hand an object to a robot.
   * Uses RGB and depth information.
   * Supports multiple feature configurations including human keypoints, head pose, image features, and object-related information.
   * Includes a ROS 2 deployment node for running the trained model from RGB-D camera topics.
   * The current robot platform is a Unitree Go2, but the recognition model is not inherently Go2-specific.

2. **Quest-side robot-belief estimation and pose guidance**

   * Trains a model that estimates how the robot-side handoff recognizer is likely to interpret the human.
   * At runtime, this model operates from **Meta Quest 3 body-joint data without communicating with the robot**.
   * If the estimated robot interpretation indicates that the user's current pose is unlikely to be understood as a handoff, the system can  perform a counterfactual search for a small pose change that increases the predicted handoff probability. This change can be visualized in AR as 3D guidance or textual guidance using PoseFix/PoseScript.

3. **Unitree Go2 detection, tracking, and FoV estimation**

   * Detects and tracks the Unitree Go2 from imagery available to the AR system.
   * Estimates the robot's location and forward direction from its motion.
   * Uses this to supports estimation of the robot camera's field of view.
   * The resulting FoV can be visualized in AR on the Meta Quest 3 to help the user understand what the robot is likely able to see.


---

# Repository Structure

The repository is organized approximately as follows:

```text
UnitreeGo2HandoffIntentDetector/
│
├── model_training_and_implementation/
│   ├── model_training/
│   ├── src/
│   ├── outputs/
│   └── kwan_pretrained_weights/
│
├── quest_hand_intent_model_est/
│   ├── model_training/
│   └── src/
│
├── robot_fov_estimation/
│   ├── model_training/
│   ├── src/
│   └── ostrack/
│
├── ros_handoff_detection/
│   ├── package.xml
│   ├── setup.py
│   ├── setup.cfg
│   ├── resource/
│   └── ros_handoff_detection/
│
├── servers/
│
├── shared/
│
└── pyproject.toml
```

---

# 1. Robot-Side Handoff Intent Recognition

The handoff intent recognition pipeline determines whether a person is currently attempting to hand an object to a robot.

The current implementation operates on RGB-D observations and can combine features derived from:

* Human body keypoints
* Head orientation
* Relative joint depth
* Image features
* Object-related features

The task is treated as binary classification:

```text
handoff
```

or:

```text
not_handoff
```

with an associated confidence score.

---

## Handoff Model Implementation

The handoff-model source code is primarily located under:

```text
model_training_and_implementation/src/
```

The trained-model inference wrapper is located at:

```text
model_training_and_implementation/src/handoff_detection_wrapper.py
```

and provides an interface for passing an RGB image and corresponding depth image through the trained model.

Model training resources are under:

```text
model_training_and_implementation/model_training/
```

Trained handoff-model checkpoints are stored under:

```text
model_training_and_implementation/outputs/hand_intent_mlp_weights/
```

with separate directories for different feature configurations.

---

## Head-Pose Estimation Attribution

Part of the head-pose estimation implementation used by this repository is **modified from code associated with**:

> Jun Kwan, Chinkye Tan, and Akansel Cosgun,
> **“Gesture Recognition for Initiating Human-to-Robot Handovers,”** 2020.

Paper:

https://arxiv.org/abs/2007.09945


The head-pose implementation in this repository has been modified for integration with the perception and inference pipeline used here and should not be considered an entirely original head-pose implementation.

Relevant code derived from or associated with this implementation is located under:

```text
model_training_and_implementation/src/kwan_headpose/
```

and associated pretrained resources are located under:

```text
model_training_and_implementation/kwan_pretrained_weights/
```

Users interested in the original method should refer to the paper and its associated repository.

---

# 2. Quest-Side Robot-Belief Estimation and Pose Guidance

The second major component estimates how the **robot's handoff recognition model is likely to interpret the human**.

This enables the AR system to reason about the robot's likely belief without requiring runtime communication between the Meta Quest 3 and the robot.

The high-level architecture is:

```text
Robot-side handoff recognizer
             │
             │ offline supervision / training
             ▼
        Quest 3 application
             │
             │ Quest 3 body joints at runtime
             ▼
    Robot belief estimator
             │
             ▼
Pose-change optimization
             │
             ▼
Human-readable AR guidance
```

During training, the estimator can learn to approximate the behavior of the robot-side handoff recognition model.

During deployment, however, the Quest-side model does **not** require the robot to transmit:

* its current classifier output;
* its camera observations;
* its internal model state; or
* its current handoff belief.

Instead, inference is performed from body-joint information available from the Quest 3.

---

## Pose Guidance

If the belief estimator model predicts that the robot is unlikely to interpret the user's current pose as a handoff, the guidance pipeline can search for a small perturbation to the user's current body pose that increases the predicted handoff probability.

The resulting target pose can then be communicated to the user through AR guidance.

Relevant code is located under:

```text
quest_hand_intent_model_est/src/
```

including components for:

* Quest joint feature processing
* Robot-belief inference
* Finding minimal pose perturbations
* Pose-based guidance
* PoseFix-based textual correction generation

Training resources are located under:

```text
quest_hand_intent_model_est/model_training/
```

---

# 3. Unitree Go2 Detection, Tracking, and FoV Estimation

The third component detects and tracks the Unitree Go2 so that the AR system can estimate the robot's pose and visualize its approximate camera field of view in AR.

Relevant code is primarily located under:

```text
robot_fov_estimation/
```

The pipeline includes resources for:

* Go2 detection
* Robot tracking
* Forward-direction estimation

The primary runtime implementations are located under:

```text
robot_fov_estimation/src/
```

The repository also contains modified OSTrack-related resources under:

```text
robot_fov_estimation/ostrack/
```

and Go2 detector training resources under:

```text
robot_fov_estimation/model_training/
```

The detection component is currently Go2-specific because its object detector is trained to recognize the Unitree Go2.

The overall tracking and FoV approach can be adapted to another robot by replacing or retraining the detector and providing the appropriate camera/FoV parameters.

---

# Requirements

## Python

The repository is developed using:

```text
Python 3.9
```

A Python 3.9 environment is therefore recommended.

Verify the installed version with:

```bash
python3.9 --version
```

---

## GPU

A CUDA-capable GPU is strongly recommended for the machine-learning and computer-vision components.

The development environment has used a CUDA-enabled PyTorch installation.

PyTorch and TorchVision should be installed using versions appropriate for the CUDA environment on the target machine.

Do not manually reproduce the individual `nvidia-*` packages shown by `pip list` on another machine. Install the appropriate CUDA-enabled PyTorch distribution instead.

---

# Main Python Installation

## 1. Clone the Repository

```bash
git clone <REPOSITORY_URL>
cd UnitreeGo2HandoffIntentDetector
```

---

## 2. Create a Python 3.9 Environment

For example:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
python --version
```

The result should indicate Python 3.9.

If using Conda or another environment manager, an equivalent Python 3.9 environment can be used instead.

---

## 3. Install PyTorch

Install a PyTorch/TorchVision build compatible with the CUDA version available on the system.

The exact installation command depends on the CUDA configuration of the machine.

Verify the installation with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 4. Install the Main Repository

From the root of this repository:

```bash
pip install -e .
```

The editable installation makes the repository's Python packages importable without depending on the current working directory.

Important local packages include:

```text
model_training_and_implementation
quest_hand_intent_model_est
robot_fov_estimation
servers
shared
```

Verify that the main handoff-model package is visible:

```bash
python -c "import model_training_and_implementation; print('Import successful')"
```

---

# Installing PoseScript / PoseFix

The pose-guidance pipeline uses the rule-based **PoseFix** comparative pipeline contained in the external [PoseScript repository](https://github.com/naver/posescript).

PoseScript should remain a separate repository rather than being copied into this repository.

The PoseFix rule-based pipeline does not require a pretrained PoseFix model checkpoint.

> **Important:** PoseScript is research code and its original environment uses older package versions. The following installation procedure deliberately avoids allowing the PoseScript dependency configuration to replace the PyTorch environment used by this project.

---

## 1. Clone PoseScript

Navigate to the location where external repositories are stored:

```bash
cd /path/to/external/repos
git clone https://github.com/naver/posescript.git
cd posescript
```

The relevant PoseFix implementation is under:

```text
src/text2pose/posefix/
```

The rule-based correction-generation entry point used by this project is:

```python
from text2pose.posefix.correcting import main
```

---

## 2. Use the Tested PoseScript Version

For reproducibility, a specific tested PoseScript commit should ideally be used:

```bash
git checkout <TESTED_POSESCRIPT_COMMIT>
```

If a tested commit has not yet been recorded, use:

```bash
git checkout main
```

Once the final version used by this project has been established, replace `<TESTED_POSESCRIPT_COMMIT>` in these instructions with the corresponding Git commit hash.

---

## 3. Install PoseScript Without Replacing PyTorch

Activate the same Python environment used by the main repository:

```bash
source /path/to/UnitreeGo2HandoffIntentDetector/.venv/bin/activate
```

From the PoseScript root:

```bash
pip install -e . --no-deps
```

The `--no-deps` option is intentional.

Do **not** initially install PoseScript using:

```bash
pip install -r requirements.txt
```

because its original environment may pin older versions of packages including PyTorch and could replace dependencies required by this project.

Install the lightweight dependencies needed by the rule-based PoseFix pipeline:

```bash
pip install roma networkx tabulate
```

Verify that the `text2pose` package is visible:

```bash
python -c "import text2pose; print(text2pose.__file__)"
```

The resulting path should point into the cloned PoseScript repository.

Then test the PoseFix entry point:

```bash
python -c "from text2pose.posefix.correcting import main; print('PoseFix imported successfully')"
```

A successful setup should print:

```text
PoseFix imported successfully
```

---

## 4. Make the Optional Contact Dependency Lazy

PoseFix supports contact-based descriptions, but this project's handoff-guidance integration does not use contact codes.

The pipeline is called with:

```python
use_contact_codes=False
```

Some versions of PoseScript nevertheless import the optional contact-processing module as soon as `correcting.py` is imported.

If the previous import test fails with an error involving:

```text
selfcontact
smplx
format_contact_info
```

make the following modification.

Open:

```text
posescript/src/text2pose/posefix/correcting.py
```

Find the top-level import resembling:

```python
from text2pose.posescript.format_contact_info import (
    from_joint_rotations_to_contact_list,
)
```

remove or comment out the top-level import:

```python
# from text2pose.posescript.format_contact_info import (
#     from_joint_rotations_to_contact_list,
# )
```

Then locate:

```python
if use_contact_codes and joint_rotations is not None:
```

and import the function inside the conditional instead:

```python
if use_contact_codes and joint_rotations is not None:
    from text2pose.posescript.format_contact_info import (
        from_joint_rotations_to_contact_list,
    )

    # Existing contact-processing code continues here.
```

This modification does not change the PoseFix correction-generation approach.

It only prevents an unused optional dependency from being loaded when:

```python
use_contact_codes=False
```

Run the import test again:

```bash
python -c "from text2pose.posefix.correcting import main; print('PoseFix imported successfully')"
```

---

# ROS 2 Handoff-Detection Setup

The robot-side handoff recognizer can be deployed as a ROS 2 node.

The ROS package is located at:

```text
ros_handoff_detection/
```

and uses:

```text
ament_python
```

The current deployment targets:

```text
ROS 2 Foxy
```

---

## Python 3.9 and ROS 2 Foxy

The Python components in this repository use Python 3.9.

A standard binary installation of ROS 2 Foxy on Ubuntu 20.04 is normally associated with the system Python version used by that Foxy distribution. As a result, an arbitrary Python 3.9 virtual environment may not automatically be able to import Foxy's Python packages.

Before attempting to run the ROS node, verify:

```bash
source /opt/ros/foxy/setup.bash
python -c "import rclpy; print('rclpy imported successfully')"
```

The Python interpreter used to launch the ROS node must have access to both:

* ROS 2 Python packages such as `rclpy`; and
* this repository's machine-learning packages.

How this is configured may depend on the ROS installation on the target robot computer.

---

## 1. Source ROS 2

```bash
source /opt/ros/foxy/setup.bash
```

---

## 2. Install ROS Dependencies

If `rosdep` is configured:

```bash
rosdep install \
    --from-paths ros_handoff_detection \
    --ignore-src \
    -r \
    -y
```

ROS-specific dependencies are declared in:

```text
ros_handoff_detection/package.xml
```

and include packages such as:

```text
rclpy
sensor_msgs
std_msgs
cv_bridge
message_filters
```

These are intentionally kept separate from the normal Python dependencies in the root `pyproject.toml`.

---

## 3. Build the ROS Package

From the root of this repository:

```bash
colcon build \
    --packages-select ros_handoff_detection \
    --symlink-install
```

Using:

```text
--symlink-install
```

is convenient during development because modifications to Python node source files can generally be used without copying the files again on each build.

For a fixed deployment installation, a normal build can also be used:

```bash
colcon build --packages-select ros_handoff_detection
```

After building:

```bash
source install/setup.bash
```

This makes the newly built package discoverable by ROS in the current shell.

A new terminal will need to source the environment again.

---

## 4. Verify the ROS Package

```bash
ros2 pkg list | grep ros_handoff_detection
```

View the executables exported by the package:

```bash
ros2 pkg executables ros_handoff_detection
```

---

# Running Robot-Side Handoff Recognition

The current ROS node is located under:

```text
ros_handoff_detection/ros_handoff_detection/
```

The current RGB-D inference implementation consumes:

```text
/realsense_rgb_image
/realsense_depth_image
```

The RGB and depth observations are approximately time-synchronized before inference because depth values are sampled relative to locations detected in the RGB image.

Launch the executable registered by the package, for example:

```bash
ros2 run ros_handoff_detection detect_handoff_node
```

If the executable name changes, check the currently registered executables with:

```bash
ros2 pkg executables ros_handoff_detection
```

The handoff classifier produces a binary prediction:

```text
handoff
```

or:

```text
not_handoff
```

together with a confidence score.

---

# Running the Server

The primary server implementation is:

```text
servers/check_current_handoff_pose_server.py
```

After activating the Python environment and installing the repository:

```bash
source .venv/bin/activate
python -m servers.check_current_handoff_pose_server
```

The server integrates the higher-level components needed for the AR-side system, including functionality related to:

* Quest body-joint processing
* Robot-belief estimation
* Handoff pose assessment
* Pose guidance
* Robot detection/tracking information

Depending on which functionality is enabled, the relevant trained models and external resources must be available.

---

# Training the Robot-Side Handoff Model

Training code is under:

```text
model_training_and_implementation/model_training/
```

The primary MLP training entry point is:

```bash
python -m model_training_and_implementation.model_training.train_mlp
```

Example:

```bash
python -m model_training_and_implementation.model_training.train_mlp \
    --features-type keypoints
```

Other supported configurations include:

```bash
python -m model_training_and_implementation.model_training.train_mlp \
    --features-type keypoints_headpose
```

and:

```bash
python -m model_training_and_implementation.model_training.train_mlp \
    --features-type keypoints_headpose_resnet
```

Person-centered cropping can also be enabled, for example:

```bash
python -m model_training_and_implementation.model_training.train_mlp \
    --features-type keypoints_headpose_resnet \
    --crop-around-object
```

Model checkpoints are written under:

```text
model_training_and_implementation/outputs/hand_intent_mlp_weights/
```

with directories corresponding to the selected model configuration.

---

# Training the Robot-Belief Estimator

Training resources for the robot belief estimator are located under:

```text
quest_hand_intent_model_est/model_training/
```

The primary training entry point is:

```bash
python -m quest_hand_intent_model_est.model_training.train_quest_hand_int_est_mlp
```

The purpose of this model is not simply to independently recognize a handoff.

Instead, it is trained to estimate the output of the **robot-side handoff recognition system** from the body-joint information available to the Quest.

The resulting estimator can therefore approximate the robot's interpretation at deployment time without receiving information from the robot.

---

# Training the Unitree Go2 Detector

Go2 detector training resources are located under:

```text
robot_fov_estimation/model_training/
```

The primary YOLO detector training entry point is:

```bash
python -m robot_fov_estimation.model_training.train_go2_yolo_det
```

The resulting detector is used by the robot detection/tracking pipeline.

The current detector is specifically trained for the Unitree Go2. To use the FoV-estimation pipeline with another robot platform, the detector should be retrained or replaced appropriately.

---

# OSTrack

OSTrack-related code is included under:

```text
robot_fov_estimation/ostrack/
```

The repository contains OSTrack runtime, training, evaluation, analysis, and visualization utilities modified to work with this repo.

Many dependencies associated with:

```text
ostrack/lib/train/
ostrack/lib/test/
ostrack/tracking/
```

are only needed for OSTrack development or evaluation and are not necessarily required to run the deployed tracking pipeline.

---

# Model and Weight Files

Several components require trained weights or pretrained resources.

## Handoff Intent Models

Stored under:

```text
model_training_and_implementation/outputs/hand_intent_mlp_weights/
```

Example configuration directories may resemble:

```text
features-keypoints_headpose_resnet__crop-False/
```

with the MLP checkpoint inside the configuration directory.

---

## Head-Pose Weights

Resources associated with the modified Kwan et al. head-pose implementation are located under:

```text
model_training_and_implementation/kwan_pretrained_weights/
```

---

## Go2 Detector / Tracker Weights

The robot detection/tracking pipeline also requires its corresponding detector and tracker weights.

Ensure the expected files are available at the paths configured by the relevant wrappers before deployment.

---

## Downloaded Pretrained Models

Some components use pretrained models from libraries such as Hugging Face or TorchVision.

These models may be downloaded automatically the first time the relevant component is initialized if they are not already cached.

A machine being deployed without Internet access should therefore have all required models downloaded in advance.

---

# Dependency Organization

This repository intentionally separates several categories of dependencies.

## Main Python Dependencies

Normal project dependencies are managed from:

```text
pyproject.toml
```

and the repository is installed with:

```bash
pip install -e .
```

---

## ROS Dependencies

ROS-specific dependencies are declared in:

```text
ros_handoff_detection/package.xml
```

rather than installed as ordinary PyPI packages.

Examples include:

```text
rclpy
sensor_msgs
std_msgs
cv_bridge
message_filters
```

---

## PoseScript / PoseFix

PoseScript remains an external repository and is installed separately with:

```bash
pip install -e . --no-deps
```

from its repository root.

This prevents PoseScript's original dependency configuration from modifying the machine-learning environment used by this project.

---

# Typical Deployment Installation

A machine intended to run the complete system can be configured approximately as follows.

## 1. Main Repository

```bash
git clone <REPOSITORY_URL>
cd UnitreeGo2HandoffIntentDetector

python3.9 -m venv .venv
source .venv/bin/activate
```

Install the appropriate CUDA-enabled PyTorch build, then:

```bash
pip install -e .
```

---

## 2. PoseScript

In the same Python environment:

```bash
cd /path/to/external/repos

git clone https://github.com/naver/posescript.git
cd posescript

git checkout <TESTED_POSESCRIPT_COMMIT>

pip install -e . --no-deps
pip install roma networkx tabulate
```

Apply the lazy contact-import modification described above if required.

Verify:

```bash
python -c "from text2pose.posefix.correcting import main; print('PoseFix imported successfully')"
```

---

## 3. ROS Package

Return to the main repository:

```bash
cd /path/to/UnitreeGo2HandoffIntentDetector
```

Source ROS:

```bash
source /opt/ros/foxy/setup.bash
```

Install ROS dependencies:

```bash
rosdep install \
    --from-paths ros_handoff_detection \
    --ignore-src \
    -r \
    -y
```

Build:

```bash
colcon build \
    --packages-select ros_handoff_detection \
    --symlink-install
```

Source the built package:

```bash
source install/setup.bash
```

---

# Typical Runtime

The main user-facing components can be run in separate terminals.

## Server

```bash
cd /path/to/UnitreeGo2HandoffIntentDetector
source .venv/bin/activate

python -m servers.check_current_handoff_pose_server
```

## Robot-Side ROS Handoff Detector

```bash
cd /path/to/UnitreeGo2HandoffIntentDetector

source /opt/ros/foxy/setup.bash
source install/setup.bash

ros2 run ros_handoff_detection detect_handoff_node
```

If necessary, inspect the registered executable name with:

```bash
ros2 pkg executables ros_handoff_detection
```

---

# Troubleshooting

## `ModuleNotFoundError: model_training_and_implementation`

Install the root repository into the active Python environment:

```bash
pip install -e .
```

Then verify:

```bash
python -c "import model_training_and_implementation"
```

---

## `ModuleNotFoundError: text2pose`

Make sure PoseScript was installed into the same environment:

```bash
cd /path/to/posescript
pip install -e . --no-deps
```

Verify:

```bash
python -c "import text2pose; print(text2pose.__file__)"
```

---

## PoseFix fails with a contact-related import error

If the error refers to:

```text
selfcontact
smplx
format_contact_info
```

apply the lazy contact-import change described in the PoseScript installation section.

The handoff-guidance application uses:

```python
use_contact_codes=False
```

and does not require these optional contact-processing dependencies.

---

## `Package 'ros_handoff_detection' not found`

Make sure the workspace has been built:

```bash
colcon build --packages-select ros_handoff_detection --symlink-install
```

and then sourced:

```bash
source install/setup.bash
```

Verify:

```bash
ros2 pkg list | grep ros_handoff_detection
```

---

## `ModuleNotFoundError: rclpy`

First source ROS:

```bash
source /opt/ros/foxy/setup.bash
```

Then test:

```bash
python -c "import rclpy"
```

If this still fails, the Python 3.9 interpreter being used does not currently have access to the ROS 2 Foxy Python installation.

The ROS Python environment and the machine-learning Python environment will need to be configured so that the ROS node can access both dependency sets.

---

## CUDA / PyTorch Problems

Check the active PyTorch installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA is unavailable unexpectedly, verify that:

* the NVIDIA driver is installed;
* the installed PyTorch build supports the available CUDA environment; and
* the correct Python environment is active.

---

## OpenCV Conflicts

Avoid installing both:

```text
opencv-python
```

and:

```text
opencv-contrib-python
```

unless the latter is specifically required.

Both provide the Python module:

```python
cv2
```

and having conflicting OpenCV wheels installed simultaneously can produce unexpected behavior.

Check the active version with:

```bash
python -c "import cv2; print(cv2.__version__)"
```

---

# External Code and Acknowledgments

This repository incorporates or builds on several external research/code resources.

## Kwan et al. Handoff Recognition / Head-Pose Code

The head-pose estimation implementation used here is modified from code associated with:

> Jun Kwan, Chinkye Tan, and Akansel Cosgun.
> **Gesture Recognition for Initiating Human-to-Robot Handovers.**
> 2020.

Paper:

https://arxiv.org/abs/2007.09945

Please refer to the original project for the corresponding license and attribution requirements.

## PoseScript / PoseFix

Pose-based correction generation uses components from the PoseScript project:

https://github.com/naver/posescript

PoseScript is maintained as a separate external dependency and is not redistributed as part of this repository.

## OSTrack

The robot-tracking subsystem includes OSTrack-related code and resources.

Users should consult the corresponding upstream OSTrack project and license when redistributing or modifying those components.

