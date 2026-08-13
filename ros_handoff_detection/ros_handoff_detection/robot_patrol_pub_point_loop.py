#!/usr/bin/env python3

import math
import random
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped
from nav2_msgs.action import NavigateToPose


@dataclass
class Waypoint:
    x: float
    y: float
    frame_id: str


class RandomNav2Patrol(Node):

    def __init__(self):
        super().__init__('random_nav2_patrol')

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.declare_parameter('num_points', 5)
        self.declare_parameter('point_topic', '/publish_point')
        self.declare_parameter('nav_action', '/navigate_to_pose')

        # If true, continue to the next waypoint if Nav2 aborts/rejects
        # a goal. If false, patrol stops on the first failed goal.
        self.declare_parameter('continue_on_failure', True)

        # -1 = nondeterministic random ordering.
        # Any non-negative value gives reproducible random routes.
        self.declare_parameter('random_seed', -1)

        self.num_points = int(
            self.get_parameter('num_points').value
        )

        self.point_topic = str(
            self.get_parameter('point_topic').value
        )

        self.nav_action = str(
            self.get_parameter('nav_action').value
        )

        self.continue_on_failure = bool(
            self.get_parameter(
                'continue_on_failure'
            ).value
        )

        random_seed = int(
            self.get_parameter('random_seed').value
        )

        if self.num_points < 2:
            raise ValueError(
                'num_points must be at least 2.'
            )

        if random_seed >= 0:
            self.random_generator = random.Random(
                random_seed
            )
        else:
            self.random_generator = random.Random()

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------

        self.points: List[Waypoint] = []

        self.route: List[Waypoint] = []

        self.route_index = 0

        self.loop_number = 0

        self.collecting_points = True

        self.navigation_started = False

        self.goal_in_progress = False

        self.frame_id: Optional[str] = None

        # ------------------------------------------------------------
        # /publish_point subscriber
        # ------------------------------------------------------------

        self.point_subscription = self.create_subscription(
            PointStamped,
            self.point_topic,
            self.point_callback,
            10,
        )

        # ------------------------------------------------------------
        # Nav2 action client
        # ------------------------------------------------------------

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            self.nav_action,
        )

        # Periodically check whether Nav2 is available once all points
        # have been collected.
        self.nav2_timer = self.create_timer(
            0.5,
            self.check_nav2_ready,
        )

        self.get_logger().info(
            f'Waiting for {self.num_points} points on '
            f'{self.point_topic}.'
        )

    # ==================================================================
    # Point collection
    # ==================================================================

    def point_callback(
        self,
        msg: PointStamped,
    ) -> None:

        if not self.collecting_points:
            return

        frame_id = msg.header.frame_id

        if not frame_id:
            self.get_logger().warning(
                'Received point with an empty frame_id. '
                'Ignoring it.'
            )
            return

        # Require every clicked point to use the same reference frame.
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
            f'Received point {point_number}/'
            f'{self.num_points}: '
            f'({waypoint.x:.3f}, {waypoint.y:.3f})'
        )

        if len(self.points) >= self.num_points:

            self.collecting_points = False

            self.get_logger().info(
                'All patrol points received.'
            )

            self.print_collected_points()

            if not self.nav_client.server_is_ready():
                self.get_logger().info(
                    f'Waiting for Nav2 action server '
                    f'"{self.nav_action}"...'
                )

    def print_collected_points(self) -> None:

        self.get_logger().info(
            'Collected patrol points:'
        )

        for index, point in enumerate(
            self.points,
            start=1,
        ):
            role = 'middle'

            if index == 1:
                role = 'START'
            elif index == len(self.points):
                role = 'END'

            self.get_logger().info(
                f'  P{index}: '
                f'({point.x:.3f}, {point.y:.3f}) '
                f'[{role}]'
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
            f'Nav2 action server "{self.nav_action}" '
            'is available.'
        )

        self.start_new_loop()

    # ==================================================================
    # Route generation
    # ==================================================================

    def start_new_loop(self) -> None:

        self.loop_number += 1

        first_point = self.points[0]
        last_point = self.points[-1]

        middle_points = list(
            self.points[1:-1]
        )

        self.random_generator.shuffle(
            middle_points
        )

        # First and last are fixed.
        self.route = (
            [first_point]
            + middle_points
            + [last_point]
        )

        self.route_index = 0

        self.get_logger().info('')
        self.get_logger().info(
            '======================================'
        )
        self.get_logger().info(
            f'Starting patrol loop {self.loop_number}'
        )

        route_text = ' -> '.join(
            self.point_label(point)
            for point in self.route
        )

        self.get_logger().info(
            f'Route: {route_text}'
        )

        self.get_logger().info(
            '======================================'
        )

        self.send_current_goal()

    def point_label(
        self,
        point: Waypoint,
    ) -> str:

        # Find the point's original collection index.
        for index, original in enumerate(
            self.points,
            start=1,
        ):
            if original is point:
                return f'P{index}'

        return '?'

    # ==================================================================
    # Nav2 goal generation
    # ==================================================================

    def send_current_goal(self) -> None:

        if self.goal_in_progress:
            return

        if self.route_index >= len(self.route):

            self.get_logger().info(
                f'Completed patrol loop '
                f'{self.loop_number}.'
            )

            # Immediately create the next randomized loop.
            self.start_new_loop()
            return

        waypoint = self.route[
            self.route_index
        ]

        yaw = self.calculate_goal_yaw(
            self.route_index
        )

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = (
            waypoint.frame_id
        )

        goal.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        goal.pose.pose.position.x = (
            waypoint.x
        )

        goal.pose.pose.position.y = (
            waypoint.y
        )

        # Nav2 is normally planar.
        goal.pose.pose.position.z = 0.0

        # Convert yaw to quaternion.
        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0

        goal.pose.pose.orientation.z = (
            math.sin(yaw / 2.0)
        )

        goal.pose.pose.orientation.w = (
            math.cos(yaw / 2.0)
        )

        label = self.point_label(
            waypoint
        )

        self.get_logger().info(
            f'Sending goal '
            f'{self.route_index + 1}/'
            f'{len(self.route)}: '
            f'{label} '
            f'({waypoint.x:.3f}, '
            f'{waypoint.y:.3f})'
        )

        self.goal_in_progress = True

        future = self.nav_client.send_goal_async(
            goal
        )

        future.add_done_callback(
            self.goal_response_callback
        )

    def calculate_goal_yaw(
        self,
        index: int,
    ) -> float:
        """
        Because /publish_point only provides a point and not an
        orientation, choose a useful orientation automatically.

        For every point except the final point:
            face toward the next waypoint.

        At the final point:
            face along the direction from the previous waypoint
            into the final waypoint.
        """

        if len(self.route) < 2:
            return 0.0

        current = self.route[index]

        if index < len(self.route) - 1:

            next_point = self.route[
                index + 1
            ]

            dx = (
                next_point.x
                - current.x
            )

            dy = (
                next_point.y
                - current.y
            )

        else:

            previous = self.route[
                index - 1
            ]

            dx = (
                current.x
                - previous.x
            )

            dy = (
                current.y
                - previous.y
            )

        # If two points are effectively identical,
        # just use zero yaw.
        if (
            abs(dx) < 1.0e-6
            and abs(dy) < 1.0e-6
        ):
            return 0.0

        return math.atan2(
            dy,
            dx,
        )

    # ==================================================================
    # Nav2 action callbacks
    # ==================================================================

    def goal_response_callback(
        self,
        future,
    ) -> None:

        try:
            goal_handle = future.result()

        except Exception as exc:
            self.get_logger().error(
                f'Error sending Nav2 goal: {exc}'
            )

            self.goal_in_progress = False

            self.handle_goal_failure()

            return

        if not goal_handle.accepted:

            self.get_logger().error(
                'Nav2 rejected the goal.'
            )

            self.goal_in_progress = False

            self.handle_goal_failure()

            return

        self.get_logger().info(
            'Goal accepted by Nav2.'
        )

        result_future = (
            goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.goal_result_callback
        )

    def goal_result_callback(
        self,
        future,
    ) -> None:

        self.goal_in_progress = False

        try:
            result = future.result()

        except Exception as exc:
            self.get_logger().error(
                f'Error getting Nav2 result: {exc}'
            )

            self.handle_goal_failure()

            return

        status = result.status

        current_point = self.route[
            self.route_index
        ]

        label = self.point_label(
            current_point
        )

        if (
            status
            == GoalStatus.STATUS_SUCCEEDED
        ):

            self.get_logger().info(
                f'Reached {label}.'
            )

            self.route_index += 1

            self.send_current_goal()

            return

        if (
            status
            == GoalStatus.STATUS_CANCELED
        ):
            self.get_logger().warning(
                f'Navigation to {label} '
                'was canceled.'
            )

        elif (
            status
            == GoalStatus.STATUS_ABORTED
        ):
            self.get_logger().warning(
                f'Navigation to {label} '
                'was aborted.'
            )

        else:
            self.get_logger().warning(
                f'Navigation to {label} '
                f'ended with status {status}.'
            )

        self.handle_goal_failure()

    def handle_goal_failure(self) -> None:

        if not self.continue_on_failure:

            self.get_logger().error(
                'Stopping patrol because '
                'continue_on_failure=false.'
            )

            self.navigation_started = False
            return

        self.get_logger().warning(
            'Skipping failed waypoint and '
            'continuing patrol.'
        )

        self.route_index += 1

        self.send_current_goal()


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