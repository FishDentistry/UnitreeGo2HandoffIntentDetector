import cv2
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
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

        # This is the handoff-confidence threshold used by the detector.
        # A frame at/above this threshold contributes toward the consecutive-
        # frame confirmation required before patrol is paused.
        self.declare_parameter(
            "threshold",
            0.6,
        )

        # Require this many consecutive frames at/above the committed
        # handoff threshold before starting the physical interaction.
        # A value of 2 suppresses isolated one-frame false positives while
        # adding only one frame of confirmation latency.
        self.declare_parameter(
            "handoff_confirmation_frames",
            2,
        )

        self.declare_parameter(
            "handoff_pause_topic",
            "/handoff_pause_patrol",
        )

        self.declare_parameter(
            "show_output_window",
            False,
        )

        self.declare_parameter(
            "debug",
            False,
        )

        # When enabled, annotated frames from the visualization window are
        # saved to confidence-binned folders. Saving is only active when
        # debug=True and show_output_window=True as well.
        self.declare_parameter(
            "save_viz_images",
            False,
        )

        self.declare_parameter(
            "servo_port",
            "/dev/ttyACM0",
        )

        self.declare_parameter(
            "servo_baud_rate",
            9600,
        )

        self.declare_parameter(
            "servo_hold_seconds",
            10.0,
        )

        # ---------------------------------------------------------
        # Aborted-handoff attempt logging
        # ---------------------------------------------------------
        # An "aborted attempt" is operationally defined as handoff
        # probability rising into an incipient-attempt region, but then
        # falling away again without ever satisfying the debounced committed
        # handoff decision. This is intentionally robot-side only and does
        # not require any synchronization with the AR headset.
        self.declare_parameter(
            "aborted_handoff_logging_enabled",
            True,
        )

        self.declare_parameter(
            "aborted_handoff_server_scheme",
            "http",
        )

        self.declare_parameter(
            "aborted_handoff_server_ip",
            "10.237.193.186",
        )

        self.declare_parameter(
            "aborted_handoff_server_port",
            8001,
        )

        self.declare_parameter(
            "aborted_handoff_server_endpoint",
            "/aborted_handoff_counts",
        )

        self.declare_parameter(
            "aborted_handoff_post_timeout_seconds",
            2.0,
        )

        # ---------------------------------------------------------
        # Robot reaction-time logging
        # ---------------------------------------------------------
        # Reaction time is measured entirely on the robot:
        # first observed P(handoff) >= the attempt-start threshold through
        # the debounced committed-handoff confirmation.
        self.declare_parameter(
            "robot_reaction_time_logging_enabled",
            True,
        )

        # Use the same server scheme/IP/port as aborted-handoff logging.
        # Only the endpoint is separate.
        self.declare_parameter(
            "robot_reaction_time_server_endpoint",
            "/indv_handoff_times",
        )

        # The attempt threshold must remain below the committed handoff
        # threshold (0.60 by default).
        self.declare_parameter(
            "aborted_handoff_attempt_start_probability",
            0.40,
        )

        # Hysteresis: once an incipient attempt begins, it is only treated
        # as having ended after probability falls below this lower value.
        self.declare_parameter(
            "aborted_handoff_attempt_end_probability",
            0.25,
        )

        # Require the signal to remain below the end threshold for this
        # long before declaring the attempt abandoned.
        self.declare_parameter(
            "aborted_handoff_attempt_end_grace_seconds",
            0.75,
        )

        # Very short probability spikes are discarded rather than counted
        # as human handoff attempts.
        self.declare_parameter(
            "aborted_handoff_attempt_min_duration_seconds",
            0.50,
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

        handoff_confirmation_frames = (
            self.get_parameter("handoff_confirmation_frames")
            .get_parameter_value()
            .integer_value
        )

        handoff_pause_topic = (
            self.get_parameter("handoff_pause_topic")
            .get_parameter_value()
            .string_value
        )

        show_output_window = (
            self.get_parameter("show_output_window")
            .get_parameter_value()
            .bool_value
        )

        debug = (
            self.get_parameter("debug")
            .get_parameter_value()
            .bool_value
        )

        save_viz_images = (
            self.get_parameter("save_viz_images")
            .get_parameter_value()
            .bool_value
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

        aborted_handoff_logging_enabled = (
            self.get_parameter("aborted_handoff_logging_enabled")
            .get_parameter_value()
            .bool_value
        )

        aborted_handoff_server_scheme = (
            self.get_parameter("aborted_handoff_server_scheme")
            .get_parameter_value()
            .string_value
            .strip()
        )

        aborted_handoff_server_ip = (
            self.get_parameter("aborted_handoff_server_ip")
            .get_parameter_value()
            .string_value
            .strip()
        )

        aborted_handoff_server_port = (
            self.get_parameter("aborted_handoff_server_port")
            .get_parameter_value()
            .integer_value
        )

        aborted_handoff_server_endpoint = (
            self.get_parameter("aborted_handoff_server_endpoint")
            .get_parameter_value()
            .string_value
            .strip()
        )

        aborted_handoff_post_timeout_seconds = (
            self.get_parameter("aborted_handoff_post_timeout_seconds")
            .get_parameter_value()
            .double_value
        )

        robot_reaction_time_logging_enabled = (
            self.get_parameter("robot_reaction_time_logging_enabled")
            .get_parameter_value()
            .bool_value
        )

        robot_reaction_time_server_endpoint = (
            self.get_parameter("robot_reaction_time_server_endpoint")
            .get_parameter_value()
            .string_value
            .strip()
        )

        aborted_handoff_attempt_start_probability = (
            self.get_parameter("aborted_handoff_attempt_start_probability")
            .get_parameter_value()
            .double_value
        )

        aborted_handoff_attempt_end_probability = (
            self.get_parameter("aborted_handoff_attempt_end_probability")
            .get_parameter_value()
            .double_value
        )

        aborted_handoff_attempt_end_grace_seconds = (
            self.get_parameter("aborted_handoff_attempt_end_grace_seconds")
            .get_parameter_value()
            .double_value
        )

        aborted_handoff_attempt_min_duration_seconds = (
            self.get_parameter("aborted_handoff_attempt_min_duration_seconds")
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
            "  handoff_confirmation_frames="
            f"{handoff_confirmation_frames}"
        )

        self.get_logger().info(
            f"  handoff_pause_topic={handoff_pause_topic}"
        )

        self.get_logger().info(
            f"  show_output_window={show_output_window}"
        )

        self.get_logger().info(
            f"  debug={debug}"
        )

        self.get_logger().info(
            f"  save_viz_images={save_viz_images}"
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

        self.get_logger().info(
            "  aborted_handoff_logging_enabled="
            f"{aborted_handoff_logging_enabled}"
        )

        self.get_logger().info(
            "  aborted_handoff_server="
            f"{aborted_handoff_server_scheme}://"
            f"{aborted_handoff_server_ip}:"
            f"{aborted_handoff_server_port}"
            f"{aborted_handoff_server_endpoint}"
        )

        self.get_logger().info(
            "  robot_reaction_time_logging_enabled="
            f"{robot_reaction_time_logging_enabled}"
        )

        self.get_logger().info(
            "  robot_reaction_time_server="
            f"{aborted_handoff_server_scheme}://"
            f"{aborted_handoff_server_ip}:"
            f"{aborted_handoff_server_port}"
            f"{robot_reaction_time_server_endpoint}"
        )

        self.get_logger().info(
            "  aborted_handoff_attempt_start_probability="
            f"{aborted_handoff_attempt_start_probability}"
        )

        self.get_logger().info(
            "  aborted_handoff_attempt_end_probability="
            f"{aborted_handoff_attempt_end_probability}"
        )

        self.get_logger().info(
            "  aborted_handoff_attempt_end_grace_seconds="
            f"{aborted_handoff_attempt_end_grace_seconds}"
        )

        self.get_logger().info(
            "  aborted_handoff_attempt_min_duration_seconds="
            f"{aborted_handoff_attempt_min_duration_seconds}"
        )

        if model_type not in MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Expected one of: {', '.join(MODEL_TYPES)}."
            )

        self.model_type = model_type
        self.features_type = str(features_type)
        self.crop_around_object = bool(crop_around_object)
        self.handoff_stop_threshold = float(threshold)
        self.handoff_confirmation_frames = int(
            handoff_confirmation_frames
        )
        self.handoff_pause_topic = str(handoff_pause_topic)
        self.show_output_window = bool(show_output_window)
        self.debug = bool(debug)
        self.output_window_name = "Handoff Classification"

        # Visualization-image saving is deliberately restricted to debug mode
        # with the visualization window enabled, so this cannot silently write
        # images during normal deployment.
        self.save_viz_images_requested = bool(save_viz_images)
        self.save_viz_images = (
            self.save_viz_images_requested
            and self.debug
            and self.show_output_window
        )
        self.viz_image_save_root = (
            Path.cwd() / "handoff_viz_images"
        ).resolve()
        self._viz_saved_frame_count = 0

        if self.save_viz_images:
            self.viz_image_save_root.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.get_logger().info(
                "Annotated visualization image saving enabled: "
                f"{self.viz_image_save_root}"
            )
        elif self.save_viz_images_requested:
            self.get_logger().warning(
                "save_viz_images=True was requested, but image saving is "
                "disabled unless debug=True and show_output_window=True."
            )

        # ---------------------------------------------------------
        # Aborted-handoff logging configuration/state
        # ---------------------------------------------------------
        self.aborted_handoff_logging_enabled = bool(
            aborted_handoff_logging_enabled
        )
        self.aborted_handoff_server_scheme = (
            aborted_handoff_server_scheme or "http"
        )
        self.aborted_handoff_server_ip = aborted_handoff_server_ip
        self.aborted_handoff_server_port = int(
            aborted_handoff_server_port
        )
        self.aborted_handoff_server_endpoint = (
            aborted_handoff_server_endpoint
        )
        self.aborted_handoff_post_timeout_seconds = float(
            aborted_handoff_post_timeout_seconds
        )
        self.robot_reaction_time_logging_enabled = bool(
            robot_reaction_time_logging_enabled
        )
        self.robot_reaction_time_server_endpoint = str(
            robot_reaction_time_server_endpoint
        )
        self.aborted_handoff_attempt_start_probability = float(
            aborted_handoff_attempt_start_probability
        )
        self.aborted_handoff_attempt_end_probability = float(
            aborted_handoff_attempt_end_probability
        )
        self.aborted_handoff_attempt_end_grace_seconds = float(
            aborted_handoff_attempt_end_grace_seconds
        )
        self.aborted_handoff_attempt_min_duration_seconds = float(
            aborted_handoff_attempt_min_duration_seconds
        )

        if self.handoff_confirmation_frames < 1:
            raise ValueError(
                "handoff_confirmation_frames must be >= 1."
            )

        if not (
            0.0
            <= self.aborted_handoff_attempt_end_probability
            < self.aborted_handoff_attempt_start_probability
            < self.handoff_stop_threshold
            <= 1.0
        ):
            raise ValueError(
                "Aborted-handoff thresholds must satisfy: "
                "0 <= end_probability < start_probability < "
                "handoff threshold <= 1."
            )

        if self.aborted_handoff_attempt_end_grace_seconds < 0.0:
            raise ValueError(
                "aborted_handoff_attempt_end_grace_seconds must be >= 0."
            )

        if self.aborted_handoff_attempt_min_duration_seconds < 0.0:
            raise ValueError(
                "aborted_handoff_attempt_min_duration_seconds must be >= 0."
            )

        if self.aborted_handoff_server_port <= 0:
            raise ValueError(
                "aborted_handoff_server_port must be > 0."
            )

        if self.aborted_handoff_post_timeout_seconds <= 0.0:
            raise ValueError(
                "aborted_handoff_post_timeout_seconds must be > 0."
            )

        if (
            self.aborted_handoff_server_endpoint
            and not self.aborted_handoff_server_endpoint.startswith("/")
        ):
            self.aborted_handoff_server_endpoint = (
                "/" + self.aborted_handoff_server_endpoint
            )

        if (
            self.robot_reaction_time_server_endpoint
            and not self.robot_reaction_time_server_endpoint.startswith("/")
        ):
            self.robot_reaction_time_server_endpoint = (
                "/" + self.robot_reaction_time_server_endpoint
            )

        self._aborted_attempt_active = False
        self._aborted_attempt_detection_armed = True
        self._aborted_attempt_start_monotonic = None
        self._aborted_attempt_start_utc = None
        self._aborted_attempt_peak_probability = 0.0
        self._aborted_attempt_below_end_since = None
        self._aborted_attempt_candidate_end_utc = None
        self._aborted_attempt_count_since_start = 0
        self._robot_reaction_time_count_since_start = 0

        # ---------------------------------------------------------
        # Servo controller
        # ---------------------------------------------------------
        self.servo_hold_seconds = float(servo_hold_seconds)
        self._servo_active = False
        self._handoff_active = False
        self._handoff_reset_timer = None
        self._last_handoff_stop_condition = False
        self._handoff_confirmation_count = 0

        self.get_logger().info(
            f"Connecting to servo controller on {servo_port}..."
        )

        if(debug == False):
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

        # Transient-local durability makes the most recent pause state
        # available if the patrol node starts after this node.
        pause_qos = QoSProfile(depth=1)
        pause_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.handoff_pause_pub = self.create_publisher(
            Bool,
            self.handoff_pause_topic,
            pause_qos,
        )

        # Start in the non-paused state.
        self._publish_handoff_pause(False)

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

            # The detector wrapper returns confidence in whichever class
            # it selected. Convert that back to P(handoff) so a negative
            # result with confidence 0.80 correctly means P(handoff)=0.20.
            handoff_probability = self._handoff_probability_from_result(
                classification,
                float(confidence),
            )

            # -----------------------------------------------------
            # Debounce the committed handoff decision
            # -----------------------------------------------------
            # The classifier output/confidence published above remain raw so
            # they can still be inspected diagnostically. Only the committed
            # handoff event is debounced.
            raw_handoff_stop_condition = (
                classification == "handoff"
                and handoff_probability >= self.handoff_stop_threshold
            )

            if raw_handoff_stop_condition:
                self._handoff_confirmation_count += 1
            else:
                self._handoff_confirmation_count = 0

            handoff_stop_condition = (
                self._handoff_confirmation_count
                >= self.handoff_confirmation_frames
            )

            if (
                self.aborted_handoff_logging_enabled
                or self.robot_reaction_time_logging_enabled
            ):
                self._update_aborted_handoff_attempt_tracking(
                    handoff_probability,
                    handoff_committed=handoff_stop_condition,
                )

            # -----------------------------------------------------
            # Optional output window
            # -----------------------------------------------------
            if self.show_output_window:
                display_image = cv2.cvtColor(
                    rgb_image,
                    cv2.COLOR_RGB2BGR,
                )

                label = (
                    f"{classification} "
                    f"({confidence:.3f})"
                )

                cv2.putText(
                    display_image,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if self.save_viz_images:
                    self._save_visualization_image(
                        display_image=display_image,
                        classification=classification,
                        confidence=float(confidence),
                    )

                cv2.imshow(
                    self.output_window_name,
                    display_image,
                )
                cv2.waitKey(1)

            # -----------------------------------------------------
            # Handoff interaction trigger
            # -----------------------------------------------------
            # The robot keeps patrolling normally until the detector
            # identifies a handoff for the configured number of consecutive
            # frames. At that point, request that the patrol node stop and
            # hold the stop for the same interval used by the handoff servo.
            #
            # Using the transition into the debounced condition prevents
            # repeated triggering on every camera frame while the person
            # remains in a handoff pose.
            if (
                handoff_stop_condition
                and not self._last_handoff_stop_condition
                and not self._handoff_active
            ):
                self._start_handoff_interaction()

            self._last_handoff_stop_condition = handoff_stop_condition

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

    def _save_visualization_image(
        self,
        display_image,
        classification: str,
        confidence: float,
    ):
        """Save one annotated visualization frame into its 0.05 confidence bin."""
        confidence = max(0.0, min(1.0, float(confidence)))

        # Round to the nearest 0.05 using conventional half-up behavior.
        # Examples: 0.623 -> 0.60, 0.628 -> 0.65.
        confidence_bin = int(confidence * 20.0 + 0.5) / 20.0
        confidence_folder = self.viz_image_save_root / f"{confidence_bin:.2f}"
        confidence_folder.mkdir(parents=True, exist_ok=True)

        self._viz_saved_frame_count += 1
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
        safe_classification = str(classification).replace("/", "_")
        filename = (
            f"{timestamp}_"
            f"frame_{self._viz_saved_frame_count:08d}_"
            f"{safe_classification}_"
            f"confidence_{confidence:.3f}.png"
        )
        output_path = confidence_folder / filename

        if not cv2.imwrite(str(output_path), display_image):
            self.get_logger().warning(
                f"Failed to save visualization image: {output_path}"
            )

    @staticmethod
    def _handoff_probability_from_result(
        classification: str,
        confidence: float,
    ) -> float:
        """
        Convert detector output to P(handoff).

        The detector returns confidence in the returned class:
          handoff     -> confidence == P(handoff)
          not_handoff -> confidence == P(not_handoff)
        """
        confidence = max(0.0, min(1.0, float(confidence)))

        if classification == "handoff":
            return confidence

        if classification == "not_handoff":
            return 1.0 - confidence

        # Unknown labels should not create false attempt events.
        return 0.0

    def _reset_aborted_attempt_candidate(self):
        """Clear the current incipient-attempt state."""
        self._aborted_attempt_active = False
        self._aborted_attempt_start_monotonic = None
        self._aborted_attempt_start_utc = None
        self._aborted_attempt_peak_probability = 0.0
        self._aborted_attempt_below_end_since = None
        self._aborted_attempt_candidate_end_utc = None

    def _update_aborted_handoff_attempt_tracking(
        self,
        handoff_probability: float,
        handoff_committed: bool = False,
    ):
        """
        Track incipient handoff behavior that disappears before commitment.

        State logic:
          1. P(handoff) >= start threshold begins a candidate attempt.
          2. A handoff becomes committed only after the main trigger's
             consecutive-frame debounce confirms it.
          3. Otherwise, if P(handoff) stays below the lower end threshold
             for the configured grace period, the candidate is counted as
             aborted (provided it lasted at least the minimum duration).
          4. After a committed handoff, detection remains disarmed until
             probability returns below the end threshold. This prevents one
             long handoff from creating a false aborted attempt when the
             10-second handoff interval finishes.
        """
        now_monotonic = time.monotonic()
        handoff_probability = max(
            0.0,
            min(1.0, float(handoff_probability)),
        )

        # A committed handoff is never an aborted attempt. Commitment is
        # supplied by the same consecutive-frame debounce used to start the
        # physical interaction, so an isolated high-probability frame cannot
        # be logged as a real handoff.
        if handoff_committed:
            # A committed interaction disarms detection until the signal
            # clears. This also ensures reaction time is recorded once,
            # rather than on every frame that remains above 0.60.
            if not self._aborted_attempt_detection_armed:
                return

            if self._aborted_attempt_active:
                reaction_time_seconds = max(
                    0.0,
                    now_monotonic
                    - self._aborted_attempt_start_monotonic,
                )
                reaction_start_utc = self._aborted_attempt_start_utc
            else:
                reaction_time_seconds = 0.0
                reaction_start_utc = datetime.now(timezone.utc)

            reaction_end_utc = datetime.now(timezone.utc)

            if self.robot_reaction_time_logging_enabled:
                self._record_robot_reaction_time(
                    reaction_time_seconds=reaction_time_seconds,
                    start_utc=(
                        reaction_start_utc or reaction_end_utc
                    ),
                    end_utc=reaction_end_utc,
                    committed_handoff_probability=handoff_probability,
                    direct_commit=(not self._aborted_attempt_active),
                )

            if self._aborted_attempt_active:
                self.get_logger().info(
                    "Incipient handoff reached committed threshold; "
                    "not counting it as aborted."
                )

            self._reset_aborted_attempt_candidate()
            self._aborted_attempt_detection_armed = False
            return

        # After a committed handoff, require the signal to clear before
        # allowing a new incipient attempt to begin.
        if not self._aborted_attempt_detection_armed:
            if (
                handoff_probability
                <= self.aborted_handoff_attempt_end_probability
            ):
                self._aborted_attempt_detection_armed = True

            return

        # Do not start/finish candidate attempts while the robot is already
        # servicing a committed handoff.
        if self._handoff_active:
            return

        if not self._aborted_attempt_active:
            if (
                handoff_probability
                >= self.aborted_handoff_attempt_start_probability
            ):
                self._aborted_attempt_active = True
                self._aborted_attempt_start_monotonic = now_monotonic
                self._aborted_attempt_start_utc = datetime.now(
                    timezone.utc
                )
                self._aborted_attempt_peak_probability = (
                    handoff_probability
                )
                self._aborted_attempt_below_end_since = None
                self._aborted_attempt_candidate_end_utc = None

                self.get_logger().info(
                    "Incipient handoff attempt detected: "
                    f"P(handoff)={handoff_probability:.3f}."
                )

            return

        # Candidate is active.
        self._aborted_attempt_peak_probability = max(
            self._aborted_attempt_peak_probability,
            handoff_probability,
        )

        if (
            handoff_probability
            <= self.aborted_handoff_attempt_end_probability
        ):
            if self._aborted_attempt_below_end_since is None:
                self._aborted_attempt_below_end_since = now_monotonic
                self._aborted_attempt_candidate_end_utc = datetime.now(
                    timezone.utc
                )

            below_duration = (
                now_monotonic
                - self._aborted_attempt_below_end_since
            )

            if (
                below_duration
                >= self.aborted_handoff_attempt_end_grace_seconds
            ):
                # Measure the attempt itself only until the signal first
                # dropped below the end threshold. Do not include the
                # debounce/grace period in the attempt duration.
                attempt_duration = (
                    self._aborted_attempt_below_end_since
                    - self._aborted_attempt_start_monotonic
                )

                if (
                    attempt_duration
                    >= self.aborted_handoff_attempt_min_duration_seconds
                ):
                    if self.aborted_handoff_logging_enabled:
                        self._record_aborted_handoff_attempt(
                            attempt_duration_seconds=attempt_duration,
                            end_utc=(
                                self._aborted_attempt_candidate_end_utc
                                or datetime.now(timezone.utc)
                            ),
                        )
                else:
                    self.get_logger().info(
                        "Discarding short incipient-handoff spike "
                        f"({attempt_duration:.3f} s)."
                    )

                self._reset_aborted_attempt_candidate()

        else:
            # Probability recovered before the grace period completed.
            self._aborted_attempt_below_end_since = None
            self._aborted_attempt_candidate_end_utc = None

    def _record_aborted_handoff_attempt(
        self,
        attempt_duration_seconds: float,
        end_utc: datetime,
    ):
        """Create an abort event and POST it without blocking inference."""
        self._aborted_attempt_count_since_start += 1

        event_id = str(uuid.uuid4())

        start_utc = self._aborted_attempt_start_utc
        if start_utc is None:
            start_utc = end_utc

        payload = {
            "eventType": "aborted_handoff_attempt",
            "eventId": event_id,
            "eventNumberSinceNodeStart": (
                self._aborted_attempt_count_since_start
            ),
            "attemptStartUtc": start_utc.isoformat(),
            "attemptEndUtc": end_utc.isoformat(),
            "durationSeconds": float(attempt_duration_seconds),
            "peakHandoffProbability": float(
                self._aborted_attempt_peak_probability
            ),
            "attemptStartProbabilityThreshold": float(
                self.aborted_handoff_attempt_start_probability
            ),
            "attemptEndProbabilityThreshold": float(
                self.aborted_handoff_attempt_end_probability
            ),
            "committedHandoffProbabilityThreshold": float(
                self.handoff_stop_threshold
            ),
            "handoffConfirmationFrames": int(
                self.handoff_confirmation_frames
            ),
            "modelType": self.model_type,
            "featuresType": self.features_type,
            "cropAroundObject": self.crop_around_object,
        }

        self.get_logger().warning(
            "Aborted handoff attempt detected: "
            f"duration={attempt_duration_seconds:.3f}s, "
            "peak P(handoff)="
            f"{self._aborted_attempt_peak_probability:.3f}, "
            f"event_id={event_id}."
        )

        thread = threading.Thread(
            target=self._post_aborted_handoff_event,
            args=(payload,),
            daemon=True,
        )
        thread.start()

    def _record_robot_reaction_time(
        self,
        reaction_time_seconds: float,
        start_utc: datetime,
        end_utc: datetime,
        committed_handoff_probability: float,
        direct_commit: bool,
    ):
        """POST one robot-side handoff reaction-time measurement."""
        self._robot_reaction_time_count_since_start += 1

        event_id = str(uuid.uuid4())

        payload = {
            "eventType": "robot_handoff_reaction_time",
            "eventId": event_id,
            "eventNumberSinceNodeStart": (
                self._robot_reaction_time_count_since_start
            ),
            "reactionStartUtc": start_utc.isoformat(),
            "reactionEndUtc": end_utc.isoformat(),
            "reactionTimeSeconds": float(reaction_time_seconds),
            "attemptStartProbabilityThreshold": float(
                self.aborted_handoff_attempt_start_probability
            ),
            "committedHandoffProbabilityThreshold": float(
                self.handoff_stop_threshold
            ),
            "committedHandoffProbability": float(
                committed_handoff_probability
            ),
            "handoffConfirmationFrames": int(
                self.handoff_confirmation_frames
            ),
            "directCommit": bool(direct_commit),
            "modelType": self.model_type,
            "featuresType": self.features_type,
            "cropAroundObject": self.crop_around_object,
        }

        self.get_logger().info(
            "Robot handoff reaction time: "
            f"{reaction_time_seconds:.3f}s, "
            f"event_id={event_id}."
        )

        thread = threading.Thread(
            target=self._post_robot_reaction_time_event,
            args=(payload,),
            daemon=True,
        )
        thread.start()

    def _shared_server_url(self, endpoint: str) -> str:
        """Construct a URL using the configured study server."""
        return (
            f"{self.aborted_handoff_server_scheme}://"
            f"{self.aborted_handoff_server_ip}:"
            f"{self.aborted_handoff_server_port}"
            f"{endpoint}"
        )

    def _robot_reaction_time_server_url(self) -> str:
        return self._shared_server_url(
            self.robot_reaction_time_server_endpoint
        )

    def _post_robot_reaction_time_event(self, payload: dict):
        """POST one reaction-time event without blocking inference."""
        url = self._robot_reaction_time_server_url()

        try:
            body = json.dumps(payload).encode("utf-8")

            request = urllib.request.Request(
                url=url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(
                request,
                timeout=self.aborted_handoff_post_timeout_seconds,
            ) as response:
                status = getattr(response, "status", None)

            self.get_logger().info(
                "Posted robot reaction time "
                f"to {url} (HTTP {status})."
            )

        except urllib.error.HTTPError as exc:
            self.get_logger().error(
                "Server rejected robot reaction-time event: "
                f"HTTP {exc.code} from {url}."
            )

        except urllib.error.URLError as exc:
            self.get_logger().error(
                "Could not POST robot reaction-time event "
                f"to {url}: {exc.reason}"
            )

        except Exception as exc:
            self.get_logger().error(
                "Unexpected error posting robot reaction-time event "
                f"to {url}: {type(exc).__name__}: {exc}"
            )

    def _aborted_handoff_server_url(self) -> str:
        """Construct the configured aborted-handoff logging URL."""
        return self._shared_server_url(
            self.aborted_handoff_server_endpoint
        )

    def _post_aborted_handoff_event(self, payload: dict):
        """
        POST one aborted-handoff event.

        Uses only the Python standard library so this does not introduce a
        requests dependency on the ROS Foxy robot environment.
        """
        url = self._aborted_handoff_server_url()

        try:
            body = json.dumps(payload).encode("utf-8")

            request = urllib.request.Request(
                url=url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(
                request,
                timeout=self.aborted_handoff_post_timeout_seconds,
            ) as response:
                status = getattr(response, "status", None)

            self.get_logger().info(
                "Posted aborted handoff attempt "
                f"to {url} (HTTP {status})."
            )

        except urllib.error.HTTPError as exc:
            self.get_logger().error(
                "Server rejected aborted handoff event: "
                f"HTTP {exc.code} from {url}."
            )

        except urllib.error.URLError as exc:
            self.get_logger().error(
                "Could not POST aborted handoff event "
                f"to {url}: {exc.reason}"
            )

        except Exception as exc:
            self.get_logger().error(
                "Unexpected error posting aborted handoff event "
                f"to {url}: {type(exc).__name__}: {exc}"
            )

    def _publish_handoff_pause(self, should_pause: bool):
        """Publish whether the patrol node should temporarily stop."""
        msg = Bool()
        msg.data = bool(should_pause)
        self.handoff_pause_pub.publish(msg)

    def _start_handoff_interaction(self):
        """Pause patrol, activate the servo, and schedule interaction end."""
        self._handoff_active = True
        self._publish_handoff_pause(True)

        self.get_logger().info(
            "Confident handoff detected: requesting patrol pause."
        )

        if not self.debug:
            try:
                self.servo.send_command(1)
                self._servo_active = True

                self.get_logger().info(
                    "Handoff detected: sent servo command 1."
                )
            except Exception as exc:
                self._servo_active = False
                self.get_logger().error(
                    f"Failed to send servo command 1: {exc}"
                )

        # create_timer() is periodic, so the callback below destroys it
        # after its first invocation to make it a one-shot timer.
        self._handoff_reset_timer = self.create_timer(
            self.servo_hold_seconds,
            self._finish_handoff_interaction,
        )

    def _finish_handoff_interaction(self):
        """Finish the handoff interval and allow patrol to resume."""
        if not self.debug and self._servo_active:
            try:
                self.servo.send_command(0)
                self.get_logger().info(
                    "Servo hold complete: sent servo command 0."
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to send servo command 0: {exc}"
                )

        self._servo_active = False
        self._handoff_active = False
        self._publish_handoff_pause(False)

        self.get_logger().info(
            "Handoff interval complete: allowing patrol to resume."
        )

        if self._handoff_reset_timer is not None:
            timer = self._handoff_reset_timer
            self._handoff_reset_timer = None
            timer.cancel()
            self.destroy_timer(timer)

    def destroy_node(self):
        """Safely return the servo to 0 and close the serial port."""
        if self.show_output_window:
            try:
                cv2.destroyWindow(self.output_window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass

        if self._handoff_reset_timer is not None:
            timer = self._handoff_reset_timer
            self._handoff_reset_timer = None
            timer.cancel()
            self.destroy_timer(timer)

        if hasattr(self, "handoff_pause_pub"):
            try:
                self._publish_handoff_pause(False)
            except Exception as exc:
                self.get_logger().warning(
                    f"Could not release patrol pause during shutdown: {exc}"
                )

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