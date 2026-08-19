import cv2
import csv
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

from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters

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

        # When enabled, pressing S while the visualization window is active
        # saves the current annotated frame. Saving requires the visualization
        # window, but it works regardless of whether debug mode is enabled.
        self.declare_parameter(
            "save_viz_images",
            False,
        )

        # ---------------------------------------------------------
        # Person-presence gating / Nav2 slowdown
        # ---------------------------------------------------------
        # RTMPose person presence is checked before the handoff model.
        # When no valid person is present, detector.predict() is not called.
        self.declare_parameter(
            "person_absence_grace_seconds",
            0.50,
        )

        # While a valid person is visible, reduce the Nav2 DWB controller's
        # configured translational speed limits to this fraction of their
        # normal values. 0.50 means 50% of normal patrol speed.
        self.declare_parameter(
            "person_slowdown_fraction",
            0.50,
        )

        # Standard Nav2 controller-server parameter service and DWB plugin
        # parameter names. These defaults match the common FollowPath DWB
        # configuration and remain configurable for a different setup.
        self.declare_parameter(
            "nav2_controller_node",
            "/controller_server",
        )
        self.declare_parameter(
            "nav2_max_vel_x_parameter",
            "FollowPath.max_vel_x",
        )
        self.declare_parameter(
            "nav2_max_speed_xy_parameter",
            "FollowPath.max_speed_xy",
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

        person_absence_grace_seconds = (
            self.get_parameter("person_absence_grace_seconds")
            .get_parameter_value()
            .double_value
        )

        person_slowdown_fraction = (
            self.get_parameter("person_slowdown_fraction")
            .get_parameter_value()
            .double_value
        )

        nav2_controller_node = (
            self.get_parameter("nav2_controller_node")
            .get_parameter_value()
            .string_value
            .strip()
        )

        nav2_max_vel_x_parameter = (
            self.get_parameter("nav2_max_vel_x_parameter")
            .get_parameter_value()
            .string_value
            .strip()
        )

        nav2_max_speed_xy_parameter = (
            self.get_parameter("nav2_max_speed_xy_parameter")
            .get_parameter_value()
            .string_value
            .strip()
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
            "  person_absence_grace_seconds="
            f"{person_absence_grace_seconds}"
        )

        self.get_logger().info(
            f"  person_slowdown_fraction={person_slowdown_fraction}"
        )

        self.get_logger().info(
            f"  nav2_controller_node={nav2_controller_node}"
        )

        self.get_logger().info(
            "  nav2_speed_parameters="
            f"[{nav2_max_vel_x_parameter}, "
            f"{nav2_max_speed_xy_parameter}]"
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

        # Person-presence gating and slowdown state. The current-frame
        # presence decision gates handoff inference. Slowdown release uses a
        # short grace interval so a one-frame RTMPose miss does not cause the
        # robot to jump immediately back to full speed.
        self.person_absence_grace_seconds = float(
            person_absence_grace_seconds
        )
        self.person_slowdown_fraction = float(
            person_slowdown_fraction
        )
        self.nav2_controller_node = str(
            nav2_controller_node or "/controller_server"
        )
        if not self.nav2_controller_node.startswith("/"):
            self.nav2_controller_node = "/" + self.nav2_controller_node

        self.nav2_speed_parameter_names = [
            str(nav2_max_vel_x_parameter),
            str(nav2_max_speed_xy_parameter),
        ]
        self._last_person_seen_monotonic = None
        self._person_slowdown_desired = False
        self._person_slowdown_applied = False
        self._controller_speed_initialized = False
        self._controller_speed_get_inflight = False
        self._controller_speed_set_inflight = False
        self._controller_normal_speeds = {}
        self._controller_speed_service_wait_logged = False

        # Visualization-image saving is available whenever explicitly enabled
        # and the visualization window is active. Debug mode is not required.
        self.save_viz_images_requested = bool(save_viz_images)
        self.save_viz_images = (
            self.save_viz_images_requested
            and self.show_output_window
        )
        self.viz_image_save_root = (
            Path.cwd() / "handoff_viz_images"
        ).resolve()
        self.viz_confidence_csv_path = (
            self.viz_image_save_root / "handoff_confidences.csv"
        )
        self._viz_saved_frame_count = 0

        if self.save_viz_images:
            self.viz_image_save_root.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.get_logger().info(
                "Annotated visualization image saving enabled. "
                "Press S in the visualization window to save the current frame: "
                f"{self.viz_image_save_root}"
            )
        elif self.save_viz_images_requested:
            self.get_logger().warning(
                "save_viz_images=True was requested, but image saving is "
                "disabled unless show_output_window=True."
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

        if self.person_absence_grace_seconds < 0.0:
            raise ValueError(
                "person_absence_grace_seconds must be >= 0."
            )

        if not (0.0 < self.person_slowdown_fraction <= 1.0):
            raise ValueError(
                "person_slowdown_fraction must satisfy 0 < value <= 1."
            )

        if not any(self.nav2_speed_parameter_names):
            raise ValueError(
                "At least one Nav2 speed parameter name must be non-empty."
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
        # Nav2 controller speed-parameter clients
        # ---------------------------------------------------------
        # This is compatible with the normal ROS 2 parameter services and
        # does not require changing the patrol node. The node first reads the
        # controller's configured normal speeds, then scales/restores those
        # exact values as person presence changes.
        controller_base = self.nav2_controller_node.rstrip("/")
        self._controller_get_parameters_client = self.create_client(
            GetParameters,
            controller_base + "/get_parameters",
        )
        self._controller_set_parameters_client = self.create_client(
            SetParameters,
            controller_base + "/set_parameters",
        )

        self._controller_speed_init_timer = self.create_timer(
            1.0,
            self._try_initialize_controller_speed_limits,
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

            # Preserve native depth representation, e.g. uint16 Z16.
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )

            # -----------------------------------------------------
            # Person-presence gate
            # -----------------------------------------------------
            # Do a lightweight RTMPose pass first. This intentionally uses
            # the detector wrapper's existing keypoint detector and existing
            # _is_valid_person() validator without changing the wrapper.
            person_present = self._person_present_in_frame(rgb_image)
            self._update_person_slowdown_state(person_present)

            # -----------------------------------------------------
            # Handoff inference ONLY when a valid person is present
            # -----------------------------------------------------
            if person_present:
                classification, confidence = self.detector.predict(
                    rgb_image,
                    depth_image,
                )

                handoff_probability = (
                    self._handoff_probability_from_result(
                        classification,
                        float(confidence),
                    )
                )
            else:
                # Publish a fresh explicit negative instead of allowing a
                # previous handoff result to remain visible downstream. TabM
                # (and the rest of detector.predict()) is not run here.
                classification = "not_handoff"
                confidence = 1.0
                handoff_probability = 0.0

            # -----------------------------------------------------
            # Publish result
            # -----------------------------------------------------
            classification_msg = String()
            classification_msg.data = classification

            confidence_msg = Float32()
            confidence_msg.data = float(confidence)

            self.classification_pub.publish(classification_msg)
            self.confidence_pub.publish(confidence_msg)

            if person_present:
                self.get_logger().info(
                    f"[{self.model_type.upper()}] "
                    f"{classification} "
                    f"(confidence={confidence:.3f}, "
                    f"P(handoff)={handoff_probability:.3f})"
                )
            else:
                self.get_logger().info(
                    "No valid person detected; handoff classifier skipped."
                )

            # -----------------------------------------------------
            # Debounce the committed handoff decision
            # -----------------------------------------------------
            raw_handoff_stop_condition = (
                person_present
                and classification == "handoff"
                and handoff_probability >= self.handoff_stop_threshold
            )

            if raw_handoff_stop_condition:
                self._handoff_confirmation_count += 1
            else:
                # This explicitly resets across any person-tracking loss, so
                # two positive frames separated by a missing-person frame do
                # not count as consecutive evidence.
                self._handoff_confirmation_count = 0

            handoff_stop_condition = (
                self._handoff_confirmation_count
                >= self.handoff_confirmation_frames
            )

            if (
                self.aborted_handoff_logging_enabled
                or self.robot_reaction_time_logging_enabled
            ):
                # P(handoff)=0 while no person is present lets any incipient
                # attempt naturally clear through the existing hysteresis.
                self._update_aborted_handoff_attempt_tracking(
                    handoff_probability,
                    handoff_committed=handoff_stop_condition,
                )

            # -----------------------------------------------------
            # Optional output window + existing S/Q key behavior
            # -----------------------------------------------------
            if self.show_output_window:
                display_image = cv2.cvtColor(
                    rgb_image,
                    cv2.COLOR_RGB2BGR,
                )

                if person_present:
                    label = (
                        f"{classification} | "
                        f"P(handoff)={handoff_probability:.3f}"
                    )
                else:
                    label = "NO PERSON | handoff inference skipped"

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

                speed_label = (
                    "SLOW"
                    if self._person_slowdown_desired
                    else "NORMAL SPEED"
                )
                cv2.putText(
                    display_image,
                    speed_label,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    self.output_window_name,
                    display_image,
                )

                key = cv2.waitKey(1) & 0xFF

                # Existing behavior: Q/q stops future study-server logging.
                if key in (ord("q"), ord("Q")):
                    self._stop_server_logging()

                # Existing behavior: S/s saves exactly the annotated frame
                # being displayed and appends its P(handoff) to the CSV.
                if (
                    self.save_viz_images
                    and key in (ord("s"), ord("S"))
                ):
                    self._save_visualization_image(
                        display_image=display_image,
                        classification=classification,
                        confidence=float(confidence),
                        handoff_probability=float(handoff_probability),
                    )

            # -----------------------------------------------------
            # Handoff interaction trigger
            # -----------------------------------------------------
            # Person present -> slowdown + inference. Two consecutive handoff
            # frames at/above threshold -> existing cancel/servo interval.
            if (
                handoff_stop_condition
                and not self._last_handoff_stop_condition
                and not self._handoff_active
            ):
                self._start_handoff_interaction()

            self._last_handoff_stop_condition = handoff_stop_condition

        except Exception as exc:
            # A failed callback must not carry positive evidence into the next
            # frame. The slowdown itself is released only through the person
            # absence grace logic on subsequent successful callbacks.
            self._handoff_confirmation_count = 0
            self._last_handoff_stop_condition = False

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

    def _person_present_in_frame(self, rgb_image) -> bool:
        """Return True when RTMPose finds at least one valid person."""
        try:
            # The wrapper converts incoming RGB to BGR before passing images
            # into the same RTMPose detector, so mirror that preprocessing.
            pose_image = cv2.cvtColor(
                rgb_image,
                cv2.COLOR_RGB2BGR,
            )
            people = self.detector.keypoint_detector.predict(pose_image)

            if not people:
                return False

            validator = getattr(self.detector, "_is_valid_person", None)
            if callable(validator):
                return any(bool(validator(person)) for person in people)

            # Defensive fallback for an older wrapper without
            # _is_valid_person(): require a geometrically valid upper body.
            for person in people:
                keypoints = person.get("keypoints", [])
                try:
                    import numpy as np
                    keypoints = np.asarray(keypoints, dtype=np.float32)
                except Exception:
                    continue

                if (
                    keypoints.ndim == 2
                    and keypoints.shape[0] >= 11
                    and keypoints.shape[1] >= 2
                    and np.isfinite(keypoints[5:11, :2]).all()
                ):
                    return True

            return False

        except Exception as exc:
            self.get_logger().warning(
                "Person-presence RTMPose check failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    def _update_person_slowdown_state(self, person_present: bool):
        """Apply slowdown on presence; restore speed after absence grace."""
        now = time.monotonic()

        if person_present:
            self._last_person_seen_monotonic = now
            self._set_person_slowdown_desired(True)
            return

        if not self._person_slowdown_desired:
            return

        if self._last_person_seen_monotonic is None:
            self._set_person_slowdown_desired(False)
            return

        absent_for = now - self._last_person_seen_monotonic
        if absent_for >= self.person_absence_grace_seconds:
            self._set_person_slowdown_desired(False)

    def _set_person_slowdown_desired(self, should_slow: bool):
        """Update desired Nav2 speed state and reconcile controller params."""
        should_slow = bool(should_slow)
        if should_slow == self._person_slowdown_desired:
            return

        self._person_slowdown_desired = should_slow

        if should_slow:
            self.get_logger().info(
                "Valid person detected: requesting reduced patrol speed and "
                "enabling handoff inference."
            )
        else:
            self.get_logger().info(
                "Person no longer present: requesting normal patrol speed."
            )

        self._request_controller_speed_reconcile()

    @staticmethod
    def _parameter_value_as_float(value):
        if value.type == ParameterType.PARAMETER_DOUBLE:
            return float(value.double_value)
        if value.type == ParameterType.PARAMETER_INTEGER:
            return float(value.integer_value)
        return None

    def _try_initialize_controller_speed_limits(self):
        """Read and cache normal Nav2 DWB speed parameters once available."""
        if self._controller_speed_initialized:
            if self._controller_speed_init_timer is not None:
                self._controller_speed_init_timer.cancel()
            return

        if self._controller_speed_get_inflight:
            return

        if not self._controller_get_parameters_client.service_is_ready():
            if not self._controller_speed_service_wait_logged:
                self._controller_speed_service_wait_logged = True
                self.get_logger().info(
                    "Waiting for Nav2 controller parameter service at "
                    f"{self.nav2_controller_node}/get_parameters ..."
                )
            return

        self._controller_speed_service_wait_logged = False
        self._controller_speed_get_inflight = True

        request = GetParameters.Request()
        request.names = [
            name for name in self.nav2_speed_parameter_names if name
        ]
        future = self._controller_get_parameters_client.call_async(request)
        future.add_done_callback(
            self._controller_speed_parameters_received
        )

    def _controller_speed_parameters_received(self, future):
        self._controller_speed_get_inflight = False

        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warning(
                "Could not read Nav2 controller speed parameters: "
                f"{exc}"
            )
            return

        names = [
            name for name in self.nav2_speed_parameter_names if name
        ]
        normal_speeds = {}

        for name, value in zip(names, response.values):
            numeric_value = self._parameter_value_as_float(value)
            if numeric_value is None or numeric_value <= 0.0:
                self.get_logger().warning(
                    f"Nav2 speed parameter {name!r} was not a positive "
                    "numeric value; it will not be modified."
                )
                continue
            normal_speeds[name] = numeric_value

        if not normal_speeds:
            self.get_logger().error(
                "Could not initialize person slowdown: none of the configured "
                "Nav2 speed parameters were readable. Handoff detection will "
                "still work, but patrol speed cannot be reduced."
            )
            return

        self._controller_normal_speeds = normal_speeds
        self._controller_speed_initialized = True

        if self._controller_speed_init_timer is not None:
            self._controller_speed_init_timer.cancel()

        values_text = ", ".join(
            f"{name}={value:.3f}"
            for name, value in normal_speeds.items()
        )
        self.get_logger().info(
            "Cached normal Nav2 controller speeds: " + values_text
        )

        self._request_controller_speed_reconcile()

    def _request_controller_speed_reconcile(self):
        """Set cached Nav2 speeds to normal or the configured slow fraction."""
        if not self._controller_speed_initialized:
            return
        if self._controller_speed_set_inflight:
            return
        if (
            self._person_slowdown_applied
            == self._person_slowdown_desired
        ):
            return
        if not self._controller_set_parameters_client.service_is_ready():
            self.get_logger().warning(
                "Nav2 controller set_parameters service is not ready; "
                "cannot change patrol speed yet."
            )
            return

        target_slow = self._person_slowdown_desired
        scale = self.person_slowdown_fraction if target_slow else 1.0

        request = SetParameters.Request()
        request.parameters = []
        target_values = {}

        for name, normal_value in self._controller_normal_speeds.items():
            target_value = normal_value * scale
            target_values[name] = target_value
            request.parameters.append(
                Parameter(
                    name=name,
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_DOUBLE,
                        double_value=float(target_value),
                    ),
                )
            )

        self._controller_speed_set_inflight = True
        future = self._controller_set_parameters_client.call_async(request)
        future.add_done_callback(
            lambda completed_future: self._controller_speed_set_done(
                completed_future,
                target_slow,
                target_values,
            )
        )

    def _controller_speed_set_done(
        self,
        future,
        target_slow: bool,
        target_values: dict,
    ):
        self._controller_speed_set_inflight = False

        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(
                "Failed to update Nav2 controller speed parameters: "
                f"{exc}"
            )
            return

        failures = [
            result.reason or "parameter update rejected"
            for result in response.results
            if not result.successful
        ]

        if failures:
            self.get_logger().error(
                "Nav2 rejected person-slowdown parameter update: "
                + "; ".join(failures)
            )
            return

        self._person_slowdown_applied = target_slow
        mode = "SLOW" if target_slow else "NORMAL"
        values_text = ", ".join(
            f"{name}={value:.3f}"
            for name, value in target_values.items()
        )
        self.get_logger().info(
            f"Nav2 patrol speed state -> {mode}: {values_text}"
        )

        # Presence can change while the asynchronous parameter request is in
        # flight. Reconcile again immediately if the desired state changed.
        if self._person_slowdown_applied != self._person_slowdown_desired:
            self._request_controller_speed_reconcile()

    def _save_visualization_image(
        self,
        display_image,
        classification: str,
        confidence: float,
        handoff_probability: float,
    ):
        """Save one annotated frame and append its P(handoff) to CSV."""
        confidence = max(0.0, min(1.0, float(confidence)))
        handoff_probability = max(
            0.0,
            min(1.0, float(handoff_probability)),
        )

        self._viz_saved_frame_count += 1
        now_utc = datetime.now(timezone.utc)
        timestamp = now_utc.strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
        safe_classification = str(classification).replace("/", "_")
        filename = (
            f"{timestamp}_"
            f"frame_{self._viz_saved_frame_count:08d}_"
            f"{safe_classification}_"
            f"confidence_{confidence:.3f}.png"
        )
        output_path = self.viz_image_save_root / filename

        if not cv2.imwrite(str(output_path), display_image):
            self.get_logger().warning(
                f"Failed to save visualization image: {output_path}"
            )
            return

        self.get_logger().info(
            f"Saved visualization image: {output_path}"
        )

        csv_exists_with_content = (
            self.viz_confidence_csv_path.exists()
            and self.viz_confidence_csv_path.stat().st_size > 0
        )

        try:
            with self.viz_confidence_csv_path.open(
                "a",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.writer(csv_file)

                if not csv_exists_with_content:
                    writer.writerow(
                        [
                            "timestamp_utc",
                            "image_filename",
                            "classification",
                            "classification_confidence",
                            "handoff_probability",
                        ]
                    )

                writer.writerow(
                    [
                        now_utc.isoformat(),
                        filename,
                        classification,
                        f"{confidence:.6f}",
                        f"{handoff_probability:.6f}",
                    ]
                )

            self.get_logger().info(
                "Recorded saved-frame P(handoff)="
                f"{handoff_probability:.6f} in "
                f"{self.viz_confidence_csv_path}"
            )

        except Exception as exc:
            self.get_logger().warning(
                "Saved visualization image, but failed to append its "
                f"confidence to CSV: {type(exc).__name__}: {exc}"
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

    def _stop_server_logging(self):
        """Disable all future study-server logging for this node run."""
        if (
            not self.aborted_handoff_logging_enabled
            and not self.robot_reaction_time_logging_enabled
        ):
            return

        self.aborted_handoff_logging_enabled = False
        self.robot_reaction_time_logging_enabled = False

        # Discard any in-progress incipient-attempt state so it cannot be
        # carried forward if logging settings are changed later.
        self._reset_aborted_attempt_candidate()
        self._aborted_attempt_detection_armed = True

        self.get_logger().warning(
            "Q pressed: study-server logging stopped. "
            "No further aborted-handoff or robot reaction-time events "
            "will be posted during this node run."
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

    def _restore_normal_controller_speed_for_shutdown(self):
        """Best-effort restoration of cached normal Nav2 speeds on shutdown."""
        if not self._controller_speed_initialized:
            return
        if not self._controller_set_parameters_client.service_is_ready():
            return

        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name=name,
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=float(value),
                ),
            )
            for name, value in self._controller_normal_speeds.items()
        ]

        try:
            self._controller_set_parameters_client.call_async(request)
        except Exception as exc:
            self.get_logger().warning(
                "Could not request normal Nav2 speed during shutdown: "
                f"{exc}"
            )

    def destroy_node(self):
        """Safely restore speed/servo state and close resources."""
        self._restore_normal_controller_speed_for_shutdown()
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