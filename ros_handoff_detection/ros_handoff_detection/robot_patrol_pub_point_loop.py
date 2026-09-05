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
    Two-waypoint Nav2 patrol with random in-place orientation stops.

    Patrol behavior:
      1. Exactly two clicked waypoints are collected: P1 and P2.
      2. The robot first moves to P1 and faces P2.
      3. It then shuttles continuously P1 <-> P2.
      4. Two candidate stopping positions are placed near the middle of the
         P1-P2 segment. On each traversal, each candidate independently has a
         configurable chance of being used. Unselected candidates are ignored.
      5. At a selected candidate, the robot stops, rotates in place, dwells for
         stop_duration_sec, then resumes toward the same endpoint.
      6. P1 -> P2 stops only turn RIGHT by 0..180 degrees relative to the
         P1->P2 route heading.
      7. P2 -> P1 stops only turn LEFT by 0..180 degrees relative to the
         P2->P1 route heading.
      8. On reaching an endpoint, the robot explicitly turns in place to face
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

        # If true, continue after a Nav2 failure. For a failed travel goal,
        # the failed endpoint is skipped. For an auxiliary turn goal, patrol
        # resumes toward the current endpoint.
        self.declare_parameter('continue_on_failure', True)

        # -1 = nondeterministic randomness. Any non-negative value makes the
        # candidate-stop choices and stop angles reproducible.
        self.declare_parameter('random_seed', -1)

        # Random stop behavior. The two stop fractions are measured from P1
        # toward P2, so 0.40 and 0.60 place them on either side of path center.
        # On every P1<->P2 traversal, each stop is independently selected with
        # candidate_stop_probability.
        self.declare_parameter('stop_duration_sec', 5.0)
        self.declare_parameter('candidate_stop_1_fraction', 0.40)
        self.declare_parameter('candidate_stop_2_fraction', 0.60)
        self.declare_parameter('candidate_stop_probability', 0.50)

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

        self.continue_on_failure = bool(
            self.get_parameter('continue_on_failure').value
        )

        self.stop_duration_sec = float(
            self.get_parameter('stop_duration_sec').value
        )

        self.candidate_stop_1_fraction = float(
            self.get_parameter('candidate_stop_1_fraction').value
        )
        self.candidate_stop_2_fraction = float(
            self.get_parameter('candidate_stop_2_fraction').value
        )
        self.candidate_stop_probability = float(
            self.get_parameter('candidate_stop_probability').value
        )

        if self.stop_duration_sec < 0.0:
            raise ValueError('stop_duration_sec must be >= 0.')

        if not (
            0.0 < self.candidate_stop_1_fraction
            < self.candidate_stop_2_fraction < 1.0
        ):
            raise ValueError(
                'candidate stop fractions must satisfy '
                '0 < stop_1 < stop_2 < 1.'
            )

        if not 0.0 <= self.candidate_stop_probability <= 1.0:
            raise ValueError(
                'candidate_stop_probability must be between 0 and 1.'
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

        # If initial positioning is canceled by a pause, retry it when the
        # pause is released. Interrupted in-place turn goals are deliberately
        # NOT retried; retrying a same-position orientation goal immediately
        # after Nav2 cancellation can leave Nav2 in its Spin recovery behavior.
        self.retry_goal_kind_after_pause: Optional[str] = None

        # Candidate stops selected for the current traversal. Each tuple is
        # (name, waypoint), ordered in the actual direction of travel.
        self.current_leg_stops = []
        self.current_leg_stop_index = 0

        # Random orientation selected for the current candidate stop.
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
            self.send_current_goal()
            return

        # Only initial positioning is retried after a pause. Random-stop and
        # endpoint in-place turns are converted into travel state when their
        # cancellation result is received (see goal_result_callback).
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
    # Candidate stop selection / behavior
    # ==================================================================

    def candidate_stop_waypoints(self):
        """Return the two fixed candidate stops measured from P1 toward P2."""
        p1 = self.points[0]
        p2 = self.points[1]

        def interpolate(fraction: float) -> Waypoint:
            return Waypoint(
                x=p1.x + fraction * (p2.x - p1.x),
                y=p1.y + fraction * (p2.y - p1.y),
                frame_id=p1.frame_id,
            )

        return [
            ('S1', interpolate(self.candidate_stop_1_fraction)),
            ('S2', interpolate(self.candidate_stop_2_fraction)),
        ]

    def prepare_stops_for_current_leg(self) -> None:
        """Randomly choose which of the two candidate stops this leg uses."""
        candidates = self.candidate_stop_waypoints()

        # When traveling P2 -> P1, visit the same physical stops in reverse
        # spatial order so the robot always progresses monotonically along the
        # segment.
        if self.target_index == 0:
            candidates = list(reversed(candidates))

        selected = []
        decisions = []
        for name, waypoint in candidates:
            use_stop = (
                self.random_generator.random()
                < self.candidate_stop_probability
            )
            decisions.append(f'{name}={"STOP" if use_stop else "SKIP"}')
            if use_stop:
                selected.append((name, waypoint))

        self.current_leg_stops = selected
        self.current_leg_stop_index = 0

        direction = 'P1 -> P2' if self.target_index == 1 else 'P2 -> P1'
        self.get_logger().info(
            f'Leg {self.leg_number} candidate stops ({direction}): '
            + ', '.join(decisions)
        )

    def choose_random_stop_orientation(self) -> None:
        """Choose a legal route-relative orientation for the current stop."""
        angle_deg = self.random_generator.uniform(0.0, 180.0)
        angle_rad = math.radians(angle_deg)
        base_yaw = self.current_leg_yaw()

        if self.target_index == 1:
            # P1 -> P2: RIGHT only. In standard ROS yaw, right is negative.
            random_yaw = self.normalize_angle(base_yaw - angle_rad)
            direction_text = 'RIGHT'
        else:
            # P2 -> P1: LEFT only. Positive yaw is counter-clockwise/left.
            random_yaw = self.normalize_angle(base_yaw + angle_rad)
            direction_text = 'LEFT'

        self.pending_random_yaw = random_yaw
        self.pending_random_angle_deg = angle_deg

        stop_name = self.pending_stop_name or '?'
        self.get_logger().info(
            f'{stop_name}: selected {direction_text} turn of '
            f'{angle_deg:.1f} deg relative to route heading.'
        )

    def begin_random_stop_dwell(self) -> None:
        """Hold the selected orientation for stop_duration_sec."""

        self.get_logger().info(
            f'Holding random orientation for '
            f'{self.stop_duration_sec:.2f} s.'
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

        # The selected candidate has now been fully serviced. Advance to the
        # next selected candidate (if any), otherwise the endpoint.
        self.current_leg_stop_index += 1

        self.get_logger().info(
            f'{completed_stop or "Candidate stop"} dwell complete.'
        )

        if self._patrol_paused():
            self.dwell_complete_waiting_for_resume = True
            return

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
        """Send/resend the next selected stop or current endpoint goal."""

        if self._patrol_paused():
            return

        if self.goal_in_progress:
            return

        if not self.initial_positioning_complete:
            self.send_initial_position_goal()
            return

        # Visit only the candidate stops selected for this traversal.
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
                f'Heading to selected candidate {stop_name} '
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
            f'Rotating in place at {self.pending_stop_name or "candidate stop"} '
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
                    f'Reached selected candidate {stop_name}; '
                    'starting in-place random orientation.'
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
                    'Random stop orientation reached.'
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

            if goal_kind == 'random_orientation':
                # Do not re-issue an interrupted same-position turn after a
                # pause. Treat this random stop as serviced and continue toward
                # the same endpoint once the pause releases. This avoids Nav2
                # immediately re-entering its Spin recovery behavior.
                completed_stop = self.pending_stop_name
                self.pending_random_yaw = None
                self.pending_random_angle_deg = None
                self.pending_stop_name = None
                self.pending_stop_waypoint = None
                self.current_leg_stop_index += 1
                self.retry_goal_kind_after_pause = None
                self.get_logger().info(
                    f'{completed_stop or "Random stop"} turn was interrupted by '
                    'pause; skipping the remainder of that turn.'
                )

            elif goal_kind == 'endpoint_turn':
                # The robot is already physically at the endpoint. Do not retry
                # the pure orientation goal after cancellation; advance the
                # patrol state to the return leg and let the next travel goal
                # perform whatever heading correction is necessary.
                reached_index = self.target_index
                opposite_index = 1 - reached_index
                self.target_index = opposite_index
                self.leg_number += 1
                self.prepare_stops_for_current_leg()
                self.retry_goal_kind_after_pause = None
                self.get_logger().info(
                    f'Endpoint P{reached_index + 1} turn was interrupted by '
                    f'pause; continuing toward P{opposite_index + 1} on resume.'
                )

            elif goal_kind == 'initial_position':
                # Initial positioning establishes the patrol's starting state,
                # so this one must be retried.
                self.retry_goal_kind_after_pause = 'initial_position'

            else:
                # Travel/candidate-travel goals are safely reconstructed by
                # send_current_goal() from the preserved patrol state.
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
                'Random orientation goal failed; skipping this stop and '
                'resuming travel toward the same endpoint.'
            )
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
                f'Candidate {stop_name} travel failed; skipping this '
                'candidate and continuing toward the same endpoint.'
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