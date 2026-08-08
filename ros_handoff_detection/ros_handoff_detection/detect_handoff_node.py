#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from message_filters import Subscriber, ApproximateTimeSynchronizer

from model_training_and_implementation.src.handoff_detection_wrapper import HandoffDetector


class HandoffInferenceNode(Node):
    def __init__(self):
        super().__init__("handoff_inference_node")

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------
        self.declare_parameter(
            "features_type",
            "keypoints_headpose_resnet",
        )
        self.declare_parameter(
            "crop_around_object",
            True,
        )
        self.declare_parameter(
            "threshold",
            0.5,
        )

        features_type = (
            self.get_parameter("features_type")
            .get_parameter_value()
            .string_value
        )

        crop_around_object = (
            self.get_parameter("crop_around_object")
            .get_parameter_value()
            .bool_value
        )

        threshold = (
            self.get_parameter("threshold")
            .get_parameter_value()
            .double_value
        )

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------
        self.detector = HandoffDetector(
            features_type=features_type,
            crop_around_object=crop_around_object,
            threshold=threshold,
        )

        self.get_logger().info(
            f"Loaded handoff detector: {self.detector.model_path}"
        )

        # ---------------------------------------------------------
        # ROS image conversion
        # ---------------------------------------------------------
        self.bridge = CvBridge()

        # ---------------------------------------------------------
        # Subscribers
        #
        # Use approximate synchronization so RGB and depth frames
        # correspond to approximately the same point in time.
        # ---------------------------------------------------------
        self.rgb_sub = Subscriber(
            self,
            Image,
            "/realsense_rgb_image",
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            "/realsense_depth_image",
        )

        self.synchronizer = ApproximateTimeSynchronizer(
            [
                self.rgb_sub,
                self.depth_sub,
            ],
            queue_size=10,
            slop=0.05,
        )

        self.synchronizer.registerCallback(
            self.image_callback
        )

        # ---------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------
        self.classification_pub = self.create_publisher(
            String,
            "/handoff_classification",
            10,
        )

        self.confidence_pub = self.create_publisher(
            Float32,
            "/handoff_confidence",
            10,
        )

        self.get_logger().info(
            "Handoff inference node started."
        )

    def image_callback(
        self,
        rgb_msg,
        depth_msg,
    ):
        try:
            # Wrapper expects an actual RGB numpy image.
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding="rgb8",
            )

            # Preserve the native depth representation,
            # e.g. uint16 Z16.
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )

            classification, confidence = (
                self.detector.predict(
                    rgb_image,
                    depth_image,
                )
            )

            # Publish result.
            classification_msg = String()
            classification_msg.data = classification

            confidence_msg = Float32()
            confidence_msg.data = float(confidence)

            self.classification_pub.publish(
                classification_msg
            )

            self.confidence_pub.publish(
                confidence_msg
            )

            self.get_logger().info(
                f"{classification} "
                f"(confidence={confidence:.3f})"
            )

        except Exception as exc:
            self.get_logger().warning(
                f"Handoff inference failed: {exc}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = HandoffInferenceNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()