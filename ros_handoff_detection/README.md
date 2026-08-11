# UnitreeGo2HandoffIntentDetector

This repository contains the robot-side handoff intent detector used with the Unitree Go2, including the ROS 2 inference node and the supporting perception/model code.

The deployment below is the tested configuration used on the robot.

## Tested Robot Environment

* NVIDIA Jetson, aarch64
* JetPack 5.1.1 / L4T R35.3.1
* Ubuntu 20.04
* ROS 2 Foxy
* Python 3.8
* CUDA 11.4
* cuDNN 8.6
* TensorRT 8.5
* NVIDIA Jetson PyTorch 2.1.0
* Transformers 4.45.1

A standard ROS 2 Foxy installation is used. No custom Python or custom ROS build is required.

---

# Initial Setup

## 1. Install System Prerequisites

ROS 2 Foxy should already be installed under:

```text
/opt/ros/foxy
```

Install the additional system packages if needed:

```bash
sudo apt update

sudo apt install -y \
    git \
    python3-pip \
    python3.8-venv \
    libopenblas-dev \
    libopenmpi-dev \
    libomp-dev \
    libjpeg-dev \
    zlib1g-dev \
    ros-foxy-rmw-cyclonedds-cpp
```

Source ROS:

```bash
source /opt/ros/foxy/setup.bash
```

## 2. Clone the Repository

```bash
git clone <REPOSITORY_URL>
cd UnitreeGo2HandoffIntentDetector
```

All commands below assume the repository root is the current directory unless stated otherwise.

## 3. Create the Python 3.8 Environment

Create a Python 3.8 virtual environment that can also see the ROS 2 Foxy Python packages:

```bash
python3.8 -m venv --system-site-packages ~/venvs/foxy-gpu
source ~/venvs/foxy-gpu/bin/activate
```

Verify:

```bash
python --version
python -c "import rclpy; print('rclpy OK')"
```

The Python version should be 3.8 and the `rclpy` import should succeed.

Upgrade pip and wheel:

```bash
python -m pip install --upgrade pip wheel
```

## 4. Install CUDA-Enabled PyTorch

Do **not** install the normal PyPI build of PyTorch on the Jetson. Use NVIDIA's Jetson-specific wheel.

For JetPack 5.1/5.1.1/5.1.2 with Python 3.8, the tested wheel is:

```text
torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

NVIDIA's Jetson PyTorch downloads are listed at:

```text
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```

Install the downloaded wheel:

```bash
python -m pip install numpy==1.24.4
python -m pip install /path/to/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

Verify CUDA:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

`torch.cuda.is_available()` must return:

```text
True
```

## 5. Install TorchVision

Use TorchVision 0.16.1 with the Jetson PyTorch 2.1 build.

Build it from source so pip does not replace the NVIDIA PyTorch package:

```bash
mkdir -p ~/src
cd ~/src

git clone https://github.com/pytorch/vision.git torchvision
cd torchvision
git checkout v0.16.1

python -m pip install . --no-deps
```

Return to the main repository:

```bash
cd /path/to/UnitreeGo2HandoffIntentDetector
```

Verify:

```bash
python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

## 6. Install the Python Runtime Dependencies

Install the Python-3.8-compatible dependency versions used by the robot deployment:

```bash
python -m pip install \
    numpy==1.24.4 \
    scipy==1.10.1 \
    packaging==23.2 \
    transformers==4.45.1 \
    rtdl_num_embeddings==0.0.10 \
    typing_extensions \
    pillow \
    pyyaml
```

Install RTMPose support:

```bash
python -m pip install rtmlib==0.0.12
```
or install with ```python -m pip install ros_python_reqs.txt```

If pip reports that a requested package would replace the NVIDIA `torch` installation, cancel the command and install that package with `--no-deps` instead.

The repository includes the Python-3.8-compatible TabM implementation used by the deployed model, so the upstream `tabm` PyPI package does not need to be installed.

## 7. Install This Repository

From the repository root:

```bash
source ~/venvs/foxy-gpu/bin/activate

python -m pip install -e . --no-deps
```

Verify that the project is importable:

```bash
cd /tmp

python - <<'PY'
import model_training_and_implementation

print(model_training_and_implementation.__file__)
PY
```

The printed path should point into the cloned repository.

## 8. Install ROS Dependencies

Return to the repository:

```bash
cd /path/to/UnitreeGo2HandoffIntentDetector
```

Then:

```bash
source /opt/ros/foxy/setup.bash
source ~/venvs/foxy-gpu/bin/activate

rosdep install \
    --from-paths ros_handoff_detection \
    --ignore-src \
    -r \
    -y
```

## 9. Build the ROS Package

The ROS Python package must be built using the same Python environment that contains the ML dependencies.

```bash
source /opt/ros/foxy/setup.bash
source ~/venvs/foxy-gpu/bin/activate

cd /path/to/UnitreeGo2HandoffIntentDetector/ros_handoff_detection

rm -rf build install log

python -c \
'from colcon_core.command import main; raise SystemExit(main())' \
build --symlink-install
```

After the build completes:

```bash
source install/setup.bash
```

Verify that the generated ROS executable uses the virtual environment:

```bash
head -n 1 \
install/ros_handoff_detection/lib/ros_handoff_detection/handoff_inference_node
```

It should point to the virtual environment, for example:

```text
#!/home/unitree/venvs/foxy-gpu/bin/python
```

## 10. Final Verification

With ROS and the virtual environment active:

```bash
python - <<'PY'
import torch
import numpy
import scipy
import transformers
import rclpy
import cv_bridge
import model_training_and_implementation

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("transformers:", transformers.__version__)
print("ROS imports OK")
print("Project imports OK")
PY
```

The most important check is:

```text
CUDA available: True
```

The first Grounding DINO run may also download the Hugging Face model files for:

```text
IDEA-Research/grounding-dino-base
```

The robot therefore needs internet access for the first download unless the model is already cached.

---

# Starting the System After Setup

The following steps are all that should be required after the one-time installation is complete.

## 1. Load the Environment

```bash
source /opt/ros/foxy/setup.bash
source ~/venvs/foxy-gpu/bin/activate

cd /path/to/UnitreeGo2HandoffIntentDetector/ros_handoff_detection
source install/setup.bash
```

## 2. Make Sure the RGB-D Camera Topics Are Available

The handoff node requires the robot's RGB and depth image streams.

Check the available topics:

```bash
ros2 topic list
```

For the current deployment, the expected image topics are:

```text
/realsense_rgb_image
/realsense_depth_image
```

Start the camera/image publisher before launching handoff inference if these topics are not already being published.

## 3. Start Handoff Inference

```bash
ros2 run ros_handoff_detection handoff_inference_node
```

The node should initialize the perception models and begin processing synchronized RGB/depth observations.

---

# Normal Startup Command Summary

After initial installation, the usual startup sequence is:

```bash
source /opt/ros/foxy/setup.bash
source ~/venvs/foxy-gpu/bin/activate

cd /path/to/UnitreeGo2HandoffIntentDetector/ros_handoff_detection
source install/setup.bash

ros2 run ros_handoff_detection handoff_inference_node
```
