from setuptools import setup

package_name = "ros_handoff_detection"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=[
        "setuptools",
        "requests",
    ],
    zip_safe=True,
    maintainer="Maintainer",
    maintainer_email="maintainer@example.com",
    description="ROS 2 Foxy node for RGB-D handoff intent detection.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "handoff_inference_node = "
            "ros_handoff_detection.detect_handoff_node:main",

            "robot_ground_truth_node = "
            "ros_handoff_detection.robot_ground_truth_node:main",

            "robot_patrol = "
            "ros_handoff_detection.robot_patrol_pub_point_loop:main",
        ],
    },
)