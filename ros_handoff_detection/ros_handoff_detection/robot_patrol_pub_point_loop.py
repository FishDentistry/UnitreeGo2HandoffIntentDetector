import math
import random
import select
import sys
import termios
import tty
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool


@dataclass
class Waypoint:
    x: float
    y: float
    frame_id: str


class RandomNav2Patrol(Node):
    """
    Two-waypoint Nav2 patrol with one pickup stop at the route midpoint.

    Patrol behavior:
      1. Exactly two clicked waypoints are collected: P1 and P2.
      2. The robot first moves to P1 and faces P2.
      3. It then shuttles continuously P1 <-> P2.
      4. On every traversal, the robot always stops at the exact midpoint of
         the P1-P2 segment.
      5. At the midpoint it rotates in place by a random configured angle,
         dwells for stop_duration_sec, then resumes toward the same endpoint.
      6. P1 -> P2 midpoint stops turn RIGHT relative to the route heading.
      7. P2 -> P1 midpoint stops turn LEFT relative to the route heading.
      8. The random turn magnitude defaults to 30..140 degrees.
      9. pickup_state_topic is True only after the midpoint orientation has
         been reached and remains True until patrol travel actually resumes.
     10. On reaching an endpoint, the robot explicitly turns in place to face
         the opposite endpoint before beginning the return traversal.

    ROS planar convention assumed here:
      +x = forward, +y = left, +z = up, positive yaw = counter-clockwise/left.
    Therefore a right turn subtracts yaw and a left turn adds yaw.
    """

    def __init__(self):
        super().__init__('random_nav2_patrol')

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        # This node intentionally accepts exactly two patrol points.
        self.declare_parameter('num_points', 2)
        self.declare_parameter('point_topic', '/clicked_point')
        self.declare_parameter('nav_action', '/navigate_to_pose')
        self.declare_parameter(
            'handoff_pause_topic',
            '/handoff_pause_patrol',
        )
        self.declare_parameter(
            'pickup_state_topic',
            '/pickup_state',
        )

        # If true, continue after a Nav2 failure. For a failed travel goal,
        # the failed endpoint is skipped. For an auxiliary turn goal, patrol
        # resumes toward the current endpoint.
        self.declare_parameter('continue_on_failure', True)

        # -1 = nondeterministic randomness. Any non-negative value makes the
        # midpoint pickup angles reproducible.
        self.declare_parameter('random_seed', -1)

        # Pickup behavior: stop at the exact route midpoint on every leg, turn
        # to a variable arrival orientation, dwell, then continue.
        self.declare_parameter('stop_duration_sec', 15.0)
        self.declare_parameter('pickup_turn_min_deg', 50.0)
        self.declare_parameter('pickup_turn_max_deg', 110.0)

        self.num_points = int(
            self.get_parameter('num_points').value
        )

        if self.num_points != 2:
            raise ValueError(
                'This patrol behavior requires num_points=2 exactly.'
            )

        self.point_topic = str(
            self.get_parameter('point_topic').value
        )

        self.nav_action = str(
            self.get_parameter('nav_action').value
        )

        self.handoff_pause_topic = str(
            self.get_parameter('handoff_pause_topic').value
        )

        self.pickup_state_topic = str(
            self.get_parameter('pickup_state_topic').value
        )

        self.continue_on_failure = bool(
            self.get_parameter('continue_on_failure').value
        )

        self.stop_duration_sec = float(
            self.get_parameter('stop_duration_sec').value
        )

        self.pickup_turn_min_deg = float(
            self.get_parameter('pickup_turn_min_deg').value
        )
        self.pickup_turn_max_deg = float(
            self.get_parameter('pickup_turn_max_deg').value
        )

        if self.stop_duration_sec < 0.0:
            raise ValueError('stop_duration_sec must be >= 0.')

        if not (
            0.0 <= self.pickup_turn_min_deg
            <= self.pickup_turn_max_deg
            <= 180.0
        ):
            raise ValueError(
                'pickup turn limits must satisfy '
                '0 <= min <= max <= 180 degrees.'
            )

        random_seed = int(
            self.get_parameter('random_seed').value
        )

        if random_seed >= 0:
            self.random_generator = random.Random(random_seed)
        else:
            self.random_generator = random.Random()

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------

        self.points: List[Waypoint] = []
        self.frame_id: Optional[str] = None

        self.collecting_points = True
        self.navigation_started = False

        # Before the shuttle begins, the robot is first positioned at P1.
        self.initial_positioning_complete = False

        # While shuttling, target_index is the endpoint currently being
        # approached: 0 = P1, 1 = P2.
        self.target_index = 0
        self.leg_number = 0

        # Nav2 action state.
        self.goal_in_progress = False
        self.active_goal_handle = None
        self.active_goal_kind: Optional[str] = None

        # Latest pose / distance reported by NavigateToPose feedback. These
        # are retained from the prior node's feedback handling.
        self.latest_current_pose: Optional[PoseStamped] = None
        self.latest_distance_remaining: Optional[float] = None

        # Pause state retained from the original node.
        self.handoff_pause_requested = False
        self.manual_pause_requested = False
        self.pause_cancel_requested = False

        # If an auxiliary orientation goal is canceled by a pause, retry it
        # when every external pause source is released.
        self.retry_goal_kind_after_pause: Optional[str] = None

        # Every leg contains exactly one pickup stop at the route midpoint.
        # The existing stop-list state is retained so pause/failure behavior
        # remains minimally changed.
        self.current_leg_stops = []
        self.current_leg_stop_index = 0

        # True only while the robot is stopped in the midpoint pickup pose.
        self.pickup_state_active = False

        # Random orientation selected for the midpoint pickup stop.
        self.pending_random_yaw: Optional[float] = None
        self.pending_random_angle_deg: Optional[float] = None
        self.pending_stop_name: Optional[str] = None
        self.pending_stop_waypoint: Optional[Waypoint] = None

        # One-shot dwell timer implemented with a normal ROS timer that is
        # canceled after its first callback.
        self.dwell_timer = None
        self.dwell_complete_waiting_for_resume = False

        # Terminal keyboard handling for the manual P-key pause.
        self._stdin_fd = None
        self._stdin_termios_original = None
        self.keyboard_timer = None

        # ------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------

        self.point_subscription = self.create_subscription(
            PointStamped,
            self.point_topic,
            self.point_callback,
            10,
        )

        # Receive temporary pause requests from the handoff detector.
        # Match the detector's transient-local QoS so a newly started patrol
        # node receives the most recent pause state.
        pause_qos = QoSProfile(depth=1)
        pause_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.handoff_pause_subscription = self.create_subscription(
            Bool,
            self.handoff_pause_topic,
            self.handoff_pause_callback,
            pause_qos,
        )

        # Publish the pickup state with transient-local durability so the
        # detector always receives the latest state, even if it starts later.
        pickup_qos = QoSProfile(depth=1)
        pickup_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pickup_state_pub = self.create_publisher(
            Bool,
            self.pickup_state_topic,
            pickup_qos,
        )
        self._publish_pickup_state(False)

        # ------------------------------------------------------------
        # Nav2 action client / timers
        # ------------------------------------------------------------

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            self.nav_action,
        )

        self.nav2_timer = self.create_timer(
            0.5,
            self.check_nav2_ready,
        )

        self._setup_keyboard_input()

        self.get_logger().info(
            f'Waiting for exactly 2 points on {self.point_topic}.'
        )

    # ==================================================================
    # Utility math
    # ==================================================================

    @staticmethod
    def normalize_angle(yaw: float) -> float:
        """Normalize an angle to [-pi, pi)."""
        return math.atan2(math.sin(yaw), math.cos(yaw))


    @staticmethod
    def yaw_from_pose(pose: PoseStamped) -> float:
        """Extract planar yaw from a PoseStamped quaternion."""
        q = pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def forward_route_yaw(self) -> float:
        """Yaw from P1 toward P2."""
        p1 = self.points[0]
        p2 = self.points[1]
        return math.atan2(p2.y - p1.y, p2.x - p1.x)

    def reverse_route_yaw(self) -> float:
        """Yaw from P2 toward P1."""
        return self.normalize_angle(self.forward_route_yaw() + math.pi)

    def endpoint_facing_yaw(self, endpoint_index: int) -> float:
        """At an endpoint, face directly toward the opposite endpoint."""
        if endpoint_index == 0:
            return self.forward_route_yaw()
        return self.reverse_route_yaw()

    def travel_arrival_yaw(self, target_index: int) -> float:
        """
        Arrive at the endpoint still facing in the direction of travel.

        The explicit 180-degree endpoint turnaround is issued only after the
        endpoint has actually been reached.
        """
        if target_index == 1:
            return self.forward_route_yaw()
        return self.reverse_route_yaw()


    def current_leg_yaw(self) -> float:
        """Return the nominal route heading for the active traversal."""
        if self.target_index == 1:
            return self.forward_route_yaw()
        return self.reverse_route_yaw()

    def _publish_pickup_state(self, active: bool) -> None:
        """Publish whether the robot is currently stopped in pickup state."""
        active = bool(active)
        self.pickup_state_active = active

        msg = Bool()
        msg.data = active
        self.pickup_state_pub.publish(msg)

        self.get_logger().info(
            f'Pickup state: {"ACTIVE" if active else "INACTIVE"}.'
        )

    # ==================================================================
    # Manual keyboard pause / resume
    # ==================================================================

    def _setup_keyboard_input(self) -> None:
        """Enable non-blocking single-key input when stdin is a terminal."""

        if not sys.stdin.isatty():
            self.get_logger().warning(
                'stdin is not an interactive terminal; '
                'P-key patrol pause is unavailable.'
            )
            return

        try:
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_termios_original = termios.tcgetattr(
                self._stdin_fd
            )

            tty.setcbreak(self._stdin_fd)

            self.keyboard_timer = self.create_timer(
                0.10,
                self._keyboard_poll_callback,
            )

            self.get_logger().info(
                'Keyboard control enabled: press P to pause/resume patrol.'
            )

        except Exception as exc:
            self.get_logger().warning(
                f'Could not enable P-key patrol pause: {exc}'
            )
            self._restore_keyboard_input()

    def _restore_keyboard_input(self) -> None:
        """Restore terminal settings changed for single-key input."""

        if (
            self._stdin_fd is not None
            and self._stdin_termios_original is not None
        ):
            try:
                termios.tcsetattr(
                    self._stdin_fd,
                    termios.TCSADRAIN,
                    self._stdin_termios_original,
                )
            except Exception as exc:
                self.get_logger().warning(
                    f'Could not restore terminal settings: {exc}'
                )

        self._stdin_fd = None
        self._stdin_termios_original = None

    def _keyboard_poll_callback(self) -> None:
        """Process pending terminal keypresses without blocking ROS."""

        if self._stdin_fd is None:
            return

        try:
            while True:
                readable, _, _ = select.select(
                    [sys.stdin],
                    [],
                    [],
                    0.0,
                )

                if not readable:
                    break

                key = sys.stdin.read(1)

                if key.lower() == 'p':
                    self._toggle_manual_pause()

        except Exception as exc:
            self.get_logger().warning(
                f'Keyboard input error: {exc}'
            )

    def _toggle_manual_pause(self) -> None:
        """Toggle the manual patrol pause without advancing the patrol."""

        self.manual_pause_requested = (
            not self.manual_pause_requested
        )

        if self.manual_pause_requested:
            self.get_logger().info(
                'Manual patrol pause enabled with P.'
            )
            self.request_pause_cancel()
            return

        self.get_logger().info(
            'Manual patrol pause released with P.'
        )

        self.resume_patrol_if_possible()

    def _patrol_paused(self) -> bool:
        """Return True while either manual or handoff pause is active."""

        return (
            self.handoff_pause_requested
            or self.manual_pause_requested
        )

    # ==================================================================
    # Handoff pause / resume
    # ==================================================================

    def handoff_pause_callback(self, msg: Bool) -> None:
        should_pause = bool(msg.data)

        if should_pause == self.handoff_pause_requested:
            return

        self.handoff_pause_requested = should_pause

        if should_pause:
            self.get_logger().info(
                'Handoff pause requested. Temporarily stopping patrol.'
            )
            self.request_pause_cancel()
            return

        self.get_logger().info(
            'Handoff pause released. Resuming patrol.'
        )

        self.resume_patrol_if_possible()

    def request_pause_cancel(self) -> None:
        """Cancel the active Nav2 goal without advancing the patrol."""

        if not self.goal_in_progress:
            return

        # Mark this cancellation as pause-related so the result callback
        # preserves the current patrol/stop state instead of advancing it.
        self.pause_cancel_requested = True

        # The goal may have been sent but not accepted yet. The goal-response
        # callback will cancel it as soon as a handle exists.
        if self.active_goal_handle is None:
            return

        self.get_logger().info(
            'Canceling current Nav2 goal for patrol pause.'
        )

        cancel_future = self.active_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(
            self.cancel_response_callback
        )

    def resume_patrol_if_possible(self) -> None:
        """Resume the correct operation once all external pauses are gone."""

        if self._patrol_paused():
            return

        if not self.navigation_started:
            return

        if self.collecting_points:
            return

        if self.goal_in_progress:
            return

        # A 5-second random-stop dwell continues to elapse while an external
        # pause is active, but movement never resumes until the pause releases.
        if self.dwell_timer is not None:
            return

        if self.dwell_complete_waiting_for_resume:
            self.dwell_complete_waiting_for_resume = False
            self._publish_pickup_state(False)
            self.send_current_goal()
            return

        if self.retry_goal_kind_after_pause == 'random_orientation':
            self.retry_goal_kind_after_pause = None
            self.send_random_orientation_goal()
            return

        if self.retry_goal_kind_after_pause == 'endpoint_turn':
            self.retry_goal_kind_after_pause = None
            self.send_endpoint_turn_goal()
            return

        if self.retry_goal_kind_after_pause == 'initial_position':
            self.retry_goal_kind_after_pause = None
            self.send_initial_position_goal()
            return

        self.retry_goal_kind_after_pause = None
        self.send_current_goal()

    # ==================================================================
    # Point collection
    # ==================================================================

    def point_callback(self, msg: PointStamped) -> None:
        if not self.collecting_points:
            return

        frame_id = msg.header.frame_id

        if not frame_id:
            self.get_logger().warning(
                'Received point with an empty frame_id. Ignoring it.'
            )
            return

        if self.frame_id is None:
            self.frame_id = frame_id
            self.get_logger().info(
                f'Using waypoint frame: {self.frame_id}'
            )

        elif frame_id != self.frame_id:
            self.get_logger().warning(
                f'Ignoring point in frame "{frame_id}". '
                f'Expected frame "{self.frame_id}".'
            )
            return

        waypoint = Waypoint(
            x=float(msg.point.x),
            y=float(msg.point.y),
            frame_id=frame_id,
        )

        self.points.append(waypoint)
        point_number = len(self.points)

        self.get_logger().info(
            f'Received point {point_number}/2: '
            f'({waypoint.x:.3f}, {waypoint.y:.3f})'
        )

        if len(self.points) >= 2:
            self.collecting_points = False

            if (
                abs(self.points[1].x - self.points[0].x) < 1.0e-6
                and abs(self.points[1].y - self.points[0].y) < 1.0e-6
            ):
                raise ValueError(
                    'P1 and P2 must be different positions.'
                )

            self.get_logger().info(
                'Both patrol endpoints received.'
            )
            self.print_collected_points()

            if not self.nav_client.server_is_ready():
                self.get_logger().info(
                    f'Waiting for Nav2 action server '
                    f'"{self.nav_action}"...'
                )

    def print_collected_points(self) -> None:
        p1 = self.points[0]
        p2 = self.points[1]

        self.get_logger().info('Collected patrol endpoints:')
        self.get_logger().info(
            f'  P1: ({p1.x:.3f}, {p1.y:.3f}) [FIRST]'
        )
        self.get_logger().info(
            f'  P2: ({p2.x:.3f}, {p2.y:.3f}) [LAST]'
        )

    # ==================================================================
    # Nav2 startup
    # ==================================================================

    def check_nav2_ready(self) -> None:
        if self.collecting_points:
            return

        if self.navigation_started:
            return

        if not self.nav_client.server_is_ready():
            return

        self.navigation_started = True
        self.nav2_timer.cancel()

        self.get_logger().info(
            f'Nav2 action server "{self.nav_action}" is available.'
        )

        # First establish the deterministic starting state: at P1, facing P2.
        self.target_index = 0
        self.send_initial_position_goal()

    # ==================================================================
    # Midpoint pickup behavior
    # ==================================================================

    def midpoint_pickup_waypoint(self) -> Waypoint:
        """Return the exact midpoint of the P1-P2 patrol segment."""
        p1 = self.points[0]
        p2 = self.points[1]
        return Waypoint(
            x=0.5 * (p1.x + p2.x),
            y=0.5 * (p1.y + p2.y),
            frame_id=p1.frame_id,
        )

    def prepare_stops_for_current_leg(self) -> None:
        """Configure the single mandatory midpoint pickup stop for this leg."""
        self.current_leg_stops = [
            ('PICKUP_CENTER', self.midpoint_pickup_waypoint())
        ]
        self.current_leg_stop_index = 0

        direction = 'P1 -> P2' if self.target_index == 1 else 'P2 -> P1'
        self.get_logger().info(
            f'Leg {self.leg_number} ({direction}): midpoint pickup stop enabled.'
        )

    def choose_random_stop_orientation(self) -> None:
        """Choose the configured direction-specific random pickup orientation."""
        angle_deg = self.random_generator.uniform(
            self.pickup_turn_min_deg,
            self.pickup_turn_max_deg,
        )
        angle_rad = math.radians(angle_deg)
        base_yaw = self.current_leg_yaw()

        if self.target_index == 1:
            # P1 -> P2: turn RIGHT. In standard ROS yaw, right is negative.
            random_yaw = self.normalize_angle(base_yaw - angle_rad)
            direction_text = 'RIGHT'
        else:
            # P2 -> P1: turn LEFT. Positive yaw is counter-clockwise/left.
            random_yaw = self.normalize_angle(base_yaw + angle_rad)
            direction_text = 'LEFT'

        self.pending_random_yaw = random_yaw
        self.pending_random_angle_deg = angle_deg

        self.get_logger().info(
            f'Midpoint pickup: selected {direction_text} turn of '
            f'{angle_deg:.1f} deg relative to route heading.'
        )

    def begin_random_stop_dwell(self) -> None:
        """Enter pickup state and hold the randomized midpoint orientation."""
        self._publish_pickup_state(True)

        self.get_logger().info(
            f'Holding pickup orientation for {self.stop_duration_sec:.2f} s.'
        )

        if self.stop_duration_sec <= 0.0:
            self.random_stop_dwell_complete()
            return

        self.dwell_timer = self.create_timer(
            self.stop_duration_sec,
            self.random_stop_dwell_complete,
        )

    def random_stop_dwell_complete(self) -> None:
        if self.dwell_timer is not None:
            try:
                self.dwell_timer.cancel()
            except Exception:
                pass
            self.dwell_timer = None

        completed_stop = self.pending_stop_name
        self.pending_random_yaw = None
        self.pending_random_angle_deg = None
        self.pending_stop_name = None
        self.pending_stop_waypoint = None

        # The midpoint stop is complete for this traversal.
        self.current_leg_stop_index += 1

        self.get_logger().info(
            f'{completed_stop or "Midpoint pickup"} dwell complete.'
        )

        # If a handoff/manual pause is active, remain in pickup state until the
        # robot is actually allowed to resume moving.
        if self._patrol_paused():
            self.dwell_complete_waiting_for_resume = True
            return

        self._publish_pickup_state(False)
        self.send_current_goal()

    # ==================================================================
    # Nav2 goal generation
    # ==================================================================

    def make_goal(
        self,
        x: float,
        y: float,
        frame_id: str,
        yaw: float,
    ) -> NavigateToPose.Goal:
        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0

        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        return goal

    def send_goal(
        self,
        goal: NavigateToPose.Goal,
        goal_kind: str,
        description: str,
    ) -> None:
        if self._patrol_paused():
            return

        if self.goal_in_progress:
            return

        self.get_logger().info(description)

        self.goal_in_progress = True
        self.active_goal_kind = goal_kind

        future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self.navigation_feedback_callback,
        )

        future.add_done_callback(
            self.goal_response_callback
        )

    def send_initial_position_goal(self) -> None:
        """Move to P1 and finish facing P2."""

        if self._patrol_paused() or self.goal_in_progress:
            return

        p1 = self.points[0]
        yaw = self.endpoint_facing_yaw(0)

        goal = self.make_goal(
            p1.x,
            p1.y,
            p1.frame_id,
            yaw,
        )

        self.send_goal(
            goal,
            'initial_position',
            f'Sending initial goal to P1 '
            f'({p1.x:.3f}, {p1.y:.3f}), facing P2.',
        )

    def send_current_goal(self) -> None:
        """Send/resend the midpoint pickup stop or current endpoint goal."""

        if self._patrol_paused():
            return

        if self.goal_in_progress:
            return

        if not self.initial_positioning_complete:
            self.send_initial_position_goal()
            return

        # Visit the mandatory midpoint pickup stop once per traversal.
        if self.current_leg_stop_index < len(self.current_leg_stops):
            stop_name, stop = self.current_leg_stops[
                self.current_leg_stop_index
            ]
            yaw = self.current_leg_yaw()

            goal = self.make_goal(
                stop.x,
                stop.y,
                stop.frame_id,
                yaw,
            )

            self.send_goal(
                goal,
                'candidate_travel',
                f'Heading to midpoint pickup {stop_name} '
                f'({stop.x:.3f}, {stop.y:.3f}) on leg '
                f'{self.leg_number}.',
            )
            return

        waypoint = self.points[self.target_index]
        yaw = self.travel_arrival_yaw(self.target_index)

        goal = self.make_goal(
            waypoint.x,
            waypoint.y,
            waypoint.frame_id,
            yaw,
        )

        direction_text = (
            'P1 -> P2' if self.target_index == 1 else 'P2 -> P1'
        )

        self.send_goal(
            goal,
            'travel',
            f'Continuing leg {self.leg_number}: {direction_text}. '
            f'Target P{self.target_index + 1} '
            f'({waypoint.x:.3f}, {waypoint.y:.3f}).',
        )

    def send_random_orientation_goal(self) -> None:
        """Rotate in place to the already-selected legal random heading."""

        if self._patrol_paused() or self.goal_in_progress:
            return

        if self.pending_random_yaw is None:
            self.get_logger().warning(
                'No pending random yaw; resuming travel instead.'
            )
            self.send_current_goal()
            return

        if self.pending_stop_waypoint is None:
            self.get_logger().warning(
                'No pending candidate stop position; resuming travel.'
            )
            self.pending_random_yaw = None
            self.pending_random_angle_deg = None
            self.pending_stop_name = None
            self.send_current_goal()
            return

        stop = self.pending_stop_waypoint

        goal = self.make_goal(
            stop.x,
            stop.y,
            stop.frame_id,
            self.pending_random_yaw,
        )

        angle_text = (
            f'{self.pending_random_angle_deg:.1f}'
            if self.pending_random_angle_deg is not None
            else '?'
        )

        self.send_goal(
            goal,
            'random_orientation',
            f'Rotating in place at {self.pending_stop_name or "midpoint pickup"} '
            f'({angle_text} deg from route heading).',
        )

    def send_endpoint_turn_goal(self) -> None:
        """At the reached endpoint, turn to face the opposite endpoint."""

        if self._patrol_paused() or self.goal_in_progress:
            return

        endpoint_index = self.target_index
        waypoint = self.points[endpoint_index]
        yaw = self.endpoint_facing_yaw(endpoint_index)
        opposite_index = 1 - endpoint_index

        goal = self.make_goal(
            waypoint.x,
            waypoint.y,
            waypoint.frame_id,
            yaw,
        )

        self.send_goal(
            goal,
            'endpoint_turn',
            f'At P{endpoint_index + 1}; turning in place to face '
            f'P{opposite_index + 1}.',
        )

    # ==================================================================
    # Nav2 feedback / action callbacks
    # ==================================================================

    def navigation_feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback

        # Copy the pose so the most recent observation remains available after
        # the travel goal is canceled for a random stop.
        pose = feedback.current_pose
        pose_copy = PoseStamped()
        pose_copy.header = pose.header
        pose_copy.pose = pose.pose
        self.latest_current_pose = pose_copy

        try:
            self.latest_distance_remaining = float(
                feedback.distance_remaining
            )
        except Exception:
            self.latest_distance_remaining = None

    def goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()

        except Exception as exc:
            goal_kind = self.active_goal_kind

            self.get_logger().error(
                f'Error sending Nav2 goal: {exc}'
            )

            self.goal_in_progress = False
            self.active_goal_kind = None
            self.active_goal_handle = None

            self.handle_goal_failure(goal_kind)
            return

        if not goal_handle.accepted:
            goal_kind = self.active_goal_kind

            self.get_logger().error(
                f'Nav2 rejected {goal_kind} goal.'
            )

            self.goal_in_progress = False
            self.active_goal_kind = None
            self.active_goal_handle = None

            self.handle_goal_failure(goal_kind)
            return

        self.get_logger().info(
            f'{self.active_goal_kind} goal accepted by Nav2.'
        )

        self.active_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            self.goal_result_callback
        )

        # A pause request can arrive between send_goal_async() and this
        # callback. Cancel immediately once the goal handle exists.
        if self._patrol_paused():
            self.request_pause_cancel()

    def cancel_response_callback(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(
                f'Error requesting Nav2 cancellation: {exc}'
            )
            return

        if not response.goals_canceling:
            self.get_logger().warning(
                'Nav2 did not accept the cancellation request.'
            )

    def goal_result_callback(self, future) -> None:
        goal_kind = self.active_goal_kind

        self.goal_in_progress = False
        self.active_goal_handle = None
        self.active_goal_kind = None

        was_pause_cancel = self.pause_cancel_requested
        self.pause_cancel_requested = False

        try:
            result = future.result()

        except Exception as exc:
            self.get_logger().error(
                f'Error getting Nav2 result for {goal_kind}: {exc}'
            )
            self.handle_goal_failure(goal_kind)
            return

        status = result.status

        # ------------------------------------------------------------
        # Successful goal
        # ------------------------------------------------------------
        if status == GoalStatus.STATUS_SUCCEEDED:
            if goal_kind == 'initial_position':
                self.get_logger().info(
                    'Reached P1 and facing P2. Starting shuttle patrol.'
                )
                self.initial_positioning_complete = True
                self.target_index = 1
                self.leg_number = 1
                self.prepare_stops_for_current_leg()
                self.send_current_goal()
                return

            if goal_kind == 'candidate_travel':
                stop_name, stop = self.current_leg_stops[
                    self.current_leg_stop_index
                ]
                self.pending_stop_name = stop_name
                self.pending_stop_waypoint = stop
                self.choose_random_stop_orientation()

                self.get_logger().info(
                    f'Reached midpoint pickup {stop_name}; '
                    'starting in-place randomized arrival orientation.'
                )
                self.send_random_orientation_goal()
                return

            if goal_kind == 'travel':
                self.pending_random_yaw = None
                self.pending_random_angle_deg = None
                self.pending_stop_name = None
                self.pending_stop_waypoint = None
                self.get_logger().info(
                    f'Reached P{self.target_index + 1}.'
                )

                # Requirement: only after actually reaching the endpoint,
                # explicitly turn to face the opposite endpoint.
                self.send_endpoint_turn_goal()
                return

            if goal_kind == 'endpoint_turn':
                reached_index = self.target_index
                opposite_index = 1 - reached_index

                self.get_logger().info(
                    f'P{reached_index + 1} turnaround complete; '
                    f'now facing P{opposite_index + 1}.'
                )

                self.target_index = opposite_index
                self.leg_number += 1
                self.prepare_stops_for_current_leg()
                self.send_current_goal()
                return

            if goal_kind == 'random_orientation':
                self.get_logger().info(
                    'Midpoint pickup orientation reached.'
                )
                self.begin_random_stop_dwell()
                return

            self.get_logger().warning(
                f'Unhandled successful goal kind: {goal_kind}'
            )
            self.send_current_goal()
            return

        # ------------------------------------------------------------
        # Cancellation caused by manual/handoff pause
        # ------------------------------------------------------------
        if (
            status == GoalStatus.STATUS_CANCELED
            and was_pause_cancel
        ):
            self.get_logger().info(
                f'{goal_kind} goal paused. Patrol state will be preserved.'
            )

            if goal_kind in (
                'random_orientation',
                'endpoint_turn',
                'initial_position',
            ):
                self.retry_goal_kind_after_pause = goal_kind
            else:
                self.retry_goal_kind_after_pause = None

            # If all pause sources were already released while cancellation
            # was in flight, resume immediately.
            self.resume_patrol_if_possible()
            return

        # ------------------------------------------------------------
        # Other cancellation / abort / failure
        # ------------------------------------------------------------
        if status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warning(
                f'{goal_kind} goal was canceled.'
            )

        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warning(
                f'{goal_kind} goal was aborted.'
            )

        else:
            self.get_logger().warning(
                f'{goal_kind} goal ended with status {status}.'
            )

        self.handle_goal_failure(goal_kind)

    # ==================================================================
    # Failure handling
    # ==================================================================

    def handle_goal_failure(self, goal_kind: Optional[str]) -> None:
        if not self.continue_on_failure:
            self.get_logger().error(
                'Stopping patrol because continue_on_failure=false.'
            )
            self.navigation_started = False
            return

        # Auxiliary turn failures should not cause an endpoint to be skipped.
        if goal_kind == 'random_orientation':
            self.get_logger().warning(
                'Pickup orientation goal failed; leaving pickup inactive and '
                'resuming travel toward the same endpoint.'
            )
            self._publish_pickup_state(False)
            self.pending_random_yaw = None
            self.pending_random_angle_deg = None
            self.pending_stop_name = None
            self.pending_stop_waypoint = None
            self.current_leg_stop_index += 1
            self.send_current_goal()
            return

        if goal_kind == 'endpoint_turn':
            reached_index = self.target_index
            opposite_index = 1 - reached_index

            self.get_logger().warning(
                'Endpoint turnaround failed; continuing patrol '
                'toward the opposite endpoint.'
            )

            self.target_index = opposite_index
            self.leg_number += 1
            self.prepare_stops_for_current_leg()
            self.send_current_goal()
            return

        if goal_kind == 'initial_position':
            self.get_logger().warning(
                'Initial P1 positioning failed. Skipping P1 and '
                'starting toward P2.'
            )
            self.initial_positioning_complete = True
            self.target_index = 1
            self.leg_number = 1
            self.prepare_stops_for_current_leg()
            self.send_current_goal()
            return

        if goal_kind == 'candidate_travel':
            stop_name = '?'
            if self.current_leg_stop_index < len(self.current_leg_stops):
                stop_name = self.current_leg_stops[
                    self.current_leg_stop_index
                ][0]

            self.get_logger().warning(
                f'Midpoint pickup {stop_name} travel failed; skipping this '
                'pickup opportunity and continuing toward the same endpoint.'
            )
            self.current_leg_stop_index += 1
            self.send_current_goal()
            return

        # Match the original continue_on_failure behavior for endpoint travel:
        # skip the failed endpoint and continue toward the opposite endpoint.
        if goal_kind == 'travel':
            failed_index = self.target_index
            self.target_index = 1 - failed_index
            self.leg_number += 1
            self.prepare_stops_for_current_leg()

            self.get_logger().warning(
                f'Skipping failed P{failed_index + 1} and continuing '
                f'toward P{self.target_index + 1}.'
            )

            self.send_current_goal()
            return

        self.get_logger().warning(
            'Unknown goal failure; attempting to resume patrol.'
        )
        self.send_current_goal()

    # ==================================================================
    # Shutdown
    # ==================================================================

    def destroy_node(self):
        """Restore terminal state and stop timers before shutdown."""

        if hasattr(self, 'pickup_state_pub'):
            try:
                self._publish_pickup_state(False)
            except Exception:
                pass

        if self.keyboard_timer is not None:
            try:
                self.keyboard_timer.cancel()
            except Exception:
                pass

        if self.dwell_timer is not None:
            try:
                self.dwell_timer.cancel()
            except Exception:
                pass

        self._restore_keyboard_input()

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = RandomNav2Patrol()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()