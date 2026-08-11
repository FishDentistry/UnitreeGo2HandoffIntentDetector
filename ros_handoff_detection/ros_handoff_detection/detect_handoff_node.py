import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from message_filters import Subscriber, ApproximateTimeSynchronizer

from .control_box_servo import ServoController


MODEL_TYPES = ("mlp", "tabm")


class HandoffInferenceNode(Node):
    def __init__(self):
        super().__init__("handoff_inference_node")

        self.get_logger().info(
            "Starting HandoffInferenceNode initialization..."
        )

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------
        self.get_logger().info(
            "Declaring ROS parameters..."
        )

        self.declare_parameter(
            "model_type",
            "tabm",
        )

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

        self.declare_parameter(
            "servo_port",
            "COM3",
        )

        self.declare_parameter(
            "servo_baud_rate",
            9600,
        )

        self.declare_parameter(
            "servo_hold_seconds",
            5.0,
        )

        self.get_logger().info(
            "Reading ROS parameters..."
        )

        model_type = (
            self.get_parameter("model_type")
            .get_parameter_value()
            .string_value
            .strip()
            .lower()
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

        servo_port = (
            self.get_parameter("servo_port")
            .get_parameter_value()
            .string_value
        )

        servo_baud_rate = (
            self.get_parameter("servo_baud_rate")
            .get_parameter_value()
            .integer_value
        )

        servo_hold_seconds = (
            self.get_parameter("servo_hold_seconds")
            .get_parameter_value()
            .double_value
        )

        self.get_logger().info(
            "Parameters:"
        )

        self.get_logger().info(
            f"  model_type={model_type}"
        )

        self.get_logger().info(
            f"  features_type={features_type}"
        )

        self.get_logger().info(
            f"  crop_around_object={crop_around_object}"
        )

        self.get_logger().info(
            f"  threshold={threshold}"
        )

        self.get_logger().info(
            f"  servo_port={servo_port}"
        )

        self.get_logger().info(
            f"  servo_baud_rate={servo_baud_rate}"
        )

        self.get_logger().info(
            f"  servo_hold_seconds={servo_hold_seconds}"
        )

        if model_type not in MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Expected one of: {', '.join(MODEL_TYPES)}."
            )

        self.model_type = model_type

        # ---------------------------------------------------------
        # Servo controller
        # ---------------------------------------------------------
        self.servo_hold_seconds = float(servo_hold_seconds)
        self._servo_active = False
        self._servo_reset_timer = None
        self._last_classification = None

        self.get_logger().info(
            f"Connecting to servo controller on {servo_port}..."
        )

        try:
            self.servo = ServoController(
                port=servo_port,
                baud_rate=int(servo_baud_rate),
            )
            self.servo.connect()
        except Exception as exc:
            self.get_logger().error(
                f"Failed to connect to servo controller: {exc}"
            )
            raise

        self.get_logger().info(
            "Servo controller connected successfully."
        )
        self.servo.send_command(0)

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------
        #
        # Import the selected wrapper lazily so that we can tell
        # whether an issue occurs:
        #
        #   1. importing the wrapper, or
        #   2. constructing the detector/models.
        #
        # This also avoids importing the unused MLP/TabM wrapper.
        # ---------------------------------------------------------

        if self.model_type == "mlp":
            self.get_logger().info(
                "Importing MLP handoff detector wrapper..."
            )

            from model_training_and_implementation.src.handoff_detection_wrapper import (
                HandoffDetector as DetectorClass,
            )

            self.get_logger().info(
                "MLP handoff detector wrapper imported successfully."
            )

        elif self.model_type == "tabm":
            self.get_logger().info(
                "Importing TabM handoff detector wrapper..."
            )

            from model_training_and_implementation.src.tabm_handoff_detection_wrapper import (
                HandoffDetector as DetectorClass,
            )

            self.get_logger().info(
                "TabM handoff detector wrapper imported successfully."
            )

        else:
            # Defensive fallback. Validation above should make
            # this branch unreachable.
            raise ValueError(
                f"Unsupported model_type: {self.model_type}"
            )

        self.get_logger().info(
            "Constructing handoff detector..."
        )

        self.get_logger().info(
            "Detector construction may initialize the classifier, "
            "RTMPose, head-pose model, and/or ResNet depending "
            "on the selected parameters."
        )

        try:
            self.detector = DetectorClass(
                features_type=features_type,
                crop_around_object=crop_around_object,
                threshold=threshold,
            )

        except Exception as exc:
            self.get_logger().error(
                "Handoff detector construction FAILED."
            )

            self.get_logger().error(
                f"Exception type: {type(exc).__name__}"
            )

            self.get_logger().error(
                f"Exception: {exc}"
            )

            raise

        self.get_logger().info(
            "Handoff detector constructed successfully."
        )

        self.get_logger().info(
            f"Loaded {self.model_type.upper()} handoff detector: "
            f"{self.detector.model_path}"
        )

        # ---------------------------------------------------------
        # ROS image conversion
        # ---------------------------------------------------------
        self.get_logger().info(
            "Initializing CvBridge..."
        )

        self.bridge = CvBridge()

        self.get_logger().info(
            "CvBridge initialized successfully."
        )

        # ---------------------------------------------------------
        # Subscribers
        #
        # Use approximate synchronization so RGB and depth frames
        # correspond to approximately the same point in time.
        # ---------------------------------------------------------
        self.get_logger().info(
            "Creating RGB and depth subscribers..."
        )

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

        self.get_logger().info(
            "Creating approximate time synchronizer..."
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

        self.get_logger().info(
            "RGB-D synchronization configured."
        )

        # ---------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------
        self.get_logger().info(
            "Creating output publishers..."
        )

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

        # Used only so that we can explicitly report when the first
        # synchronized frame reaches the callback.
        self._received_first_frame = False

        self.get_logger().info(
            "Handoff inference node initialized successfully."
        )

        self.get_logger().info(
            f"Running with model_type={self.model_type}."
        )

        self.get_logger().info(
            "Waiting for synchronized RGB-D frames..."
        )

    def image_callback(
        self,
        rgb_msg,
        depth_msg,
    ):
        if not self._received_first_frame:
            self._received_first_frame = True

            self.get_logger().info(
                "Received first synchronized RGB-D frame."
            )

        try:
            # -----------------------------------------------------
            # Convert ROS images
            # -----------------------------------------------------
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding="rgb8",
            )

            # Preserve native depth representation,
            # e.g. uint16 Z16.
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )

            # -----------------------------------------------------
            # Inference
            # -----------------------------------------------------
            classification, confidence = (
                self.detector.predict(
                    rgb_image,
                    depth_image,
                )
            )

            # -----------------------------------------------------
            # Publish result
            # -----------------------------------------------------
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
                f"[{self.model_type.upper()}] "
                f"{classification} "
                f"(confidence={confidence:.3f})"
            )

            # -----------------------------------------------------
            # Servo trigger
            # -----------------------------------------------------
            # Trigger only on the transition into "handoff". This
            # prevents every handoff-classified camera frame from
            # repeatedly sending command 1.
            if (
                classification == "handoff"
                and self._last_classification != "handoff"
                and not self._servo_active
            ):
                self._activate_servo_for_handoff()

            self._last_classification = classification

        except Exception as exc:
            self.get_logger().warning(
                "Handoff inference failed "
                f"[{self.model_type.upper()}]."
            )

            self.get_logger().warning(
                f"Exception type: {type(exc).__name__}"
            )

            self.get_logger().warning(
                f"Exception: {exc}"
            )

    def _activate_servo_for_handoff(self):
        """Send command 1 now and schedule command 0 after the hold time."""
        try:
            self.servo.send_command(1)
            self._servo_active = True

            self.get_logger().info(
                "Handoff detected: sent servo command 1."
            )

            # create_timer() is periodic, so the reset callback cancels
            # and destroys it after its first invocation to make it a
            # one-shot timer. This avoids blocking inference with sleep().
            self._servo_reset_timer = self.create_timer(
                self.servo_hold_seconds,
                self._reset_servo_after_handoff,
            )

        except Exception as exc:
            self._servo_active = False
            self.get_logger().error(
                f"Failed to send servo command 1: {exc}"
            )

    def _reset_servo_after_handoff(self):
        """Return the servo to command 0 and stop the one-shot timer."""
        try:
            self.servo.send_command(0)
            self.get_logger().info(
                "Servo hold complete: sent servo command 0."
            )
        except Exception as exc:
            self.get_logger().error(
                f"Failed to send servo command 0: {exc}"
            )
        finally:
            self._servo_active = False

            if self._servo_reset_timer is not None:
                timer = self._servo_reset_timer
                self._servo_reset_timer = None
                timer.cancel()
                self.destroy_timer(timer)

    def destroy_node(self):
        """Safely return the servo to 0 and close the serial port."""
        if self._servo_reset_timer is not None:
            timer = self._servo_reset_timer
            self._servo_reset_timer = None
            timer.cancel()
            self.destroy_timer(timer)

        if hasattr(self, "servo"):
            try:
                self.servo.send_command(0)
            except Exception as exc:
                self.get_logger().warning(
                    f"Could not reset servo during shutdown: {exc}"
                )

            try:
                self.servo.close()
            except Exception as exc:
                self.get_logger().warning(
                    f"Could not close servo serial connection: {exc}"
                )

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = HandoffInferenceNode()

        node.get_logger().info(
            "Entering ROS spin loop..."
        )

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        print(
            "[FATAL] Handoff inference node terminated during "
            f"startup/runtime: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()