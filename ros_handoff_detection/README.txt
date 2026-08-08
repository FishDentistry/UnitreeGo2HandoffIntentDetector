ros_handoff_detection
=====================

Expected wrapper:
model_training_and_implementation/src/handoff_detector.py

Build from the repository/workspace root:

    source /opt/ros/foxy/setup.bash
    colcon build --symlink-install
    source install/setup.bash

Run:

    ros2 run ros_handoff_detection handoff_inference_node

Default input topics:
    /realsense_rgb_image
    /realsense_depth_image

Published outputs:
    /handoff_classification   std_msgs/String
    /handoff_confidence       std_msgs/Float32

Example model variant:

    ros2 run ros_handoff_detection handoff_inference_node --ros-args \
        -p features_type:=keypoints_headpose_resnet \
        -p crop_around_object:=true

IMPORTANT:
The node imports:
    model_training_and_implementation.src.handoff_detector

Therefore the repository root must be importable by Python. Building with
--symlink-install from the repository/workspace root is recommended. If that
sibling package is not on PYTHONPATH in your environment, either install the
repository's Python package or add the repository root to PYTHONPATH.
