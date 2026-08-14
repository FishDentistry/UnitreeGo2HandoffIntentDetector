#!/usr/bin/env python3

import math
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
import requests
from std_msgs.msg import Float64MultiArray


class PostMarkerTransform(Node):
    """
    Subscribe to /marker_transform_matrix and forward the latest 4x4 matrix
    to the FastAPI server.

    Expected ROS message:
        std_msgs/msg/Float64MultiArray

    Expected payload:
        16 values, row-major, representing

            ROS map <- marker

        i.e. H_map_marker.reshape(-1).tolist().
    """

    def __init__(self) -> None:
        super().__init__("post_marker_transform")

        self.declare_parameter(
            "server_url",
            "http://10.237.193.186:8001",
        )
        self.declare_parameter(
            "topic_name",
            "/marker_transform_matrix",
        )
        self.declare_parameter(
            "source_id",
            "unitree_go2",
        )
        self.declare_parameter(
            "post_rate_hz",
            2.0,
        )
        self.declare_parameter(
            "request_timeout_seconds",
            2.0,
        )

        self.server_url = str(
            self.get_parameter(
                "server_url"
            ).value
        ).rstrip("/")

        self.topic_name = str(
            self.get_parameter(
                "topic_name"
            ).value
        )

        self.source_id = str(
            self.get_parameter(
                "source_id"
            ).value
        )

        self.post_rate_hz = float(
            self.get_parameter(
                "post_rate_hz"
            ).value
        )

        self.request_timeout_seconds = float(
            self.get_parameter(
                "request_timeout_seconds"
            ).value
        )

        if self.post_rate_hz <= 0.0:
            raise ValueError(
                "post_rate_hz must be > 0."
            )

        self.endpoint = (
            self.server_url
            + "/marker_transform_matrix"
        )

        self.session = requests.Session()

        self.latest_matrix: Optional[List[float]] = None
        self.latest_sequence = 0
        self.last_posted_sequence = 0

        self.subscription = self.create_subscription(
            Float64MultiArray,
            self.topic_name,
            self.marker_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.post_rate_hz,
            self.post_latest_if_needed,
        )

        self.get_logger().info(
            "post_marker_transform started."
        )
        self.get_logger().info(
            "Subscribing to: "
            + self.topic_name
        )
        self.get_logger().info(
            "Posting to: "
            + self.endpoint
        )
        self.get_logger().info(
            "source_id: "
            + self.source_id
        )

    def marker_callback(
        self,
        msg: Float64MultiArray,
    ) -> None:
        values = [
            float(value)
            for value in msg.data
        ]

        if len(values) != 16:
            self.get_logger().error(
                "Ignoring /marker_transform_matrix "
                "message: expected 16 values, got "
                + str(len(values))
                + "."
            )
            return

        if not all(
            math.isfinite(value)
            for value in values
        ):
            self.get_logger().error(
                "Ignoring /marker_transform_matrix "
                "message containing non-finite values."
            )
            return

        # Existing publisher flattens H_map_marker row-major.
        self.latest_matrix = values
        self.latest_sequence += 1

    def post_latest_if_needed(self) -> None:
        if self.latest_matrix is None:
            return

        if (
            self.latest_sequence
            == self.last_posted_sequence
        ):
            return

        sequence_to_post = (
            self.latest_sequence
        )

        payload = {
            "schema_version": "1.0",
            "source_id": self.source_id,
            "timestamp_unix_seconds": (
                time.time()
            ),
            "matrix": list(
                self.latest_matrix
            ),
        }

        try:
            response = self.session.post(
                self.endpoint,
                json=payload,
                timeout=(
                    self.request_timeout_seconds
                ),
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            # Do not advance last_posted_sequence. The newest matrix will be
            # retried on the next timer tick.
            self.get_logger().warning(
                "Failed to POST marker transform: "
                + str(exc)
            )
            return

        self.last_posted_sequence = (
            sequence_to_post
        )

        self.get_logger().debug(
            "Posted marker transform matrix."
        )

    def destroy_node(self) -> bool:
        try:
            self.session.close()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = PostMarkerTransform()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()