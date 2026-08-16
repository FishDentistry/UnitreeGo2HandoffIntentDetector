import math
import queue
import threading
import time
from typing import Dict, List, Optional

import requests

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


class RobotGroundTruthNode(Node):
    """
    Collect the Unitree's ground-truth pose from TF and send it to the
    MindReadAR robot-motion alignment server.

    Pose source:
        target_frame -> base_frame
        default: map -> base_link

    The node samples at 20 Hz by default and sends samples in batches of 10
    (one HTTP request every ~0.5 s).

    IMPORTANT:
        The timestamps sent here must share the same absolute clock domain as
        the Quest timestamps received by the Python alignment server.

        With timestamp_source="tf" (recommended), the TransformStamped header
        time is sent. This assumes the ROS clock is synchronized with the
        machine producing the Quest Unix timestamps.

        If that is not true, timestamp_source can be set to "system", which
        uses time.time() on the robot computer. That computer's clock must
        still be synchronized with the Quest/server clock.
    """

    def __init__(self) -> None:
        super().__init__(
            "robot_ground_truth_node"
        )

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------

        self.declare_parameter(
            "server_url",
            "http://127.0.0.1:8000",
        )

        self.declare_parameter(
            "endpoint",
            "/robot_ground_truth",
        )

        self.declare_parameter(
            "source_id",
            "unitree_go2",
        )

        self.declare_parameter(
            "target_frame",
            "map",
        )

        self.declare_parameter(
            "base_frame",
            "base_link",
        )

        self.declare_parameter(
            "sample_interval_seconds",
            0.05,
        )

        self.declare_parameter(
            "batch_size",
            10,
        )

        self.declare_parameter(
            "request_timeout_seconds",
            3.0,
        )

        self.declare_parameter(
            "retry_interval_seconds",
            1.0,
        )

        self.declare_parameter(
            "timestamp_source",
            "tf",
        )

        self.declare_parameter(
            "convert_ros_to_server_coordinates",
            False,
        )

        self.server_url = str(
            self.get_parameter(
                "server_url"
            ).value
        ).rstrip("/")

        endpoint = str(
            self.get_parameter(
                "endpoint"
            ).value
        )

        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        self.endpoint = endpoint

        self.post_url = (
            self.server_url
            + self.endpoint
        )

        self.source_id = str(
            self.get_parameter(
                "source_id"
            ).value
        )

        self.target_frame = str(
            self.get_parameter(
                "target_frame"
            ).value
        )

        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )

        self.sample_interval_seconds = float(
            self.get_parameter(
                "sample_interval_seconds"
            ).value
        )

        self.batch_size = int(
            self.get_parameter(
                "batch_size"
            ).value
        )

        self.request_timeout_seconds = float(
            self.get_parameter(
                "request_timeout_seconds"
            ).value
        )

        self.retry_interval_seconds = float(
            self.get_parameter(
                "retry_interval_seconds"
            ).value
        )

        self.timestamp_source = str(
            self.get_parameter(
                "timestamp_source"
            ).value
        ).lower()

        self.convert_ros_to_server_coordinates = bool(
            self.get_parameter(
                "convert_ros_to_server_coordinates"
            ).value
        )

        if self.sample_interval_seconds <= 0.0:
            raise ValueError(
                "sample_interval_seconds must be > 0."
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1."
            )

        if self.timestamp_source not in (
            "tf",
            "system",
        ):
            raise ValueError(
                "timestamp_source must be either "
                "'tf' or 'system'."
            )

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # ------------------------------------------------------------------
        # Sampling / networking state
        # ------------------------------------------------------------------

        self.sample_buffer: List[
            Dict
        ] = []

        self.last_tf_timestamp: Optional[
            float
        ] = None

        self.warned_non_unix_tf_time = False

        # HTTP work is performed outside the ROS timer callback so network
        # delays do not stop pose sampling.
        self.send_queue: queue.Queue = (
            queue.Queue(
                maxsize=10000
            )
        )

        self.stop_event = (
            threading.Event()
        )

        self.sender_thread = (
            threading.Thread(
                target=self._sender_loop,
                name=(
                    "robot_ground_truth_sender"
                ),
                daemon=True,
            )
        )

        self.sender_thread.start()

        # ------------------------------------------------------------------
        # Sampling timer
        # ------------------------------------------------------------------

        self.timer = self.create_timer(
            self.sample_interval_seconds,
            self._sample_robot_pose,
        )

        self.get_logger().info(
            "Robot ground-truth collection started."
        )

        self.get_logger().info(
            "TF pose: "
            f"{self.target_frame} -> "
            f"{self.base_frame}"
        )

        self.get_logger().info(
            "Sampling interval: "
            f"{self.sample_interval_seconds:.3f} s "
            f"({1.0 / self.sample_interval_seconds:.1f} Hz)"
        )

        self.get_logger().info(
            f"Batch size: {self.batch_size}"
        )

        self.get_logger().info(
            f"Posting to: {self.post_url}"
        )

        self.get_logger().info(
            "Timestamp source: "
            f"{self.timestamp_source}"
        )

        self.get_logger().info(
            "ROS -> server coordinate conversion: "
            f"{self.convert_ros_to_server_coordinates}"
        )

    # ----------------------------------------------------------------------
    # Timestamp helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _ros_stamp_to_seconds(
        stamp,
    ) -> float:
        return (
            float(stamp.sec)
            + float(stamp.nanosec)
            * 1e-9
        )

    def _choose_timestamp(
        self,
        transform,
    ) -> float:
        if self.timestamp_source == "system":
            return time.time()

        timestamp = (
            self._ros_stamp_to_seconds(
                transform.header.stamp
            )
        )

        # Unix timestamps in current deployments are comfortably above 1e9.
        # If this is not true, ROS may be using simulation time or another
        # clock domain, which will not align directly with Quest Unix time.
        if (
            timestamp < 1.0e9
            and not self.warned_non_unix_tf_time
        ):
            self.warned_non_unix_tf_time = True

            self.get_logger().warn(
                "TF timestamp does not look like Unix "
                "seconds. Quest/robot alignment requires "
                "both streams to use the same absolute "
                "clock domain. If ROS is not using "
                "Unix/system time, either synchronize the "
                "clock or run with "
                "-p timestamp_source:=system."
            )

        return timestamp

    # ----------------------------------------------------------------------
    # Coordinate conversion
    # ----------------------------------------------------------------------

    @staticmethod
    def _normalize_quaternion(
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ):
        norm = math.sqrt(
            qx * qx
            + qy * qy
            + qz * qz
            + qw * qw
        )

        if norm < 1e-12:
            raise ValueError(
                "Received a zero-length quaternion."
            )

        return (
            qx / norm,
            qy / norm,
            qz / norm,
            qw / norm,
        )

    @classmethod
    def _forward_vector_from_quaternion(
        cls,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ):
        """
        Rotate ROS base_link's +X forward axis into the target/map frame.

        ROS REP-103 base-frame convention:
            +X = forward
            +Y = left
            +Z = up

        This is the first column of the rotation matrix represented by the
        quaternion.
        """

        qx, qy, qz, qw = (
            cls._normalize_quaternion(
                qx,
                qy,
                qz,
                qw,
            )
        )

        forward_x = (
            1.0
            - 2.0 * (
                qy * qy
                + qz * qz
            )
        )

        forward_y = (
            2.0 * (
                qx * qy
                + qw * qz
            )
        )

        forward_z = (
            2.0 * (
                qx * qz
                - qw * qy
            )
        )

        norm = math.sqrt(
            forward_x * forward_x
            + forward_y * forward_y
            + forward_z * forward_z
        )

        if norm < 1e-12:
            raise ValueError(
                "Could not compute robot forward vector."
            )

        return (
            forward_x / norm,
            forward_y / norm,
            forward_z / norm,
        )

    @staticmethod
    def _ros_vector_to_server(
        x: float,
        y: float,
        z: float,
    ):
        """
        Convert a ROS REP-103 vector into the server's Unity-like convention.

        ROS:
            +X = forward
            +Y = left
            +Z = up

        Server / Quest-style:
            +X = right
            +Y = up
            +Z = forward

        Mapping:
            server_x = -ros_y
            server_y =  ros_z
            server_z =  ros_x

        The same mapping is applied to both position and heading.
        """

        return (
            -y,
            z,
            x,
        )

    # ----------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------

    def _sample_robot_pose(
        self,
    ) -> None:
        try:
            transform = (
                self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.base_frame,
                    Time(),
                    timeout=Duration(
                        seconds=0.1
                    ),
                )
            )

        except Exception as exc:
            self.get_logger().warn(
                "Transform unavailable "
                f"({self.target_frame} -> "
                f"{self.base_frame}): {exc}"
            )
            return

        try:
            timestamp = (
                self._choose_timestamp(
                    transform
                )
            )

            # When querying Time(), TF may return the same latest transform
            # more than once if our timer is faster than the TF publisher.
            # Do not duplicate that measurement.
            if (
                self.timestamp_source == "tf"
                and self.last_tf_timestamp is not None
                and timestamp
                <= self.last_tf_timestamp + 1e-9
            ):
                return

            translation = (
                transform.transform.translation
            )

            rotation = (
                transform.transform.rotation
            )

            forward = (
                self._forward_vector_from_quaternion(
                    rotation.x,
                    rotation.y,
                    rotation.z,
                    rotation.w,
                )
            )

            position = (
                translation.x,
                translation.y,
                translation.z,
            )

            if (
                self.convert_ros_to_server_coordinates
            ):
                position = (
                    self._ros_vector_to_server(
                        *position
                    )
                )

                forward = (
                    self._ros_vector_to_server(
                        *forward
                    )
                )

            sample = {
                "timestamp": timestamp,
                "position": {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                },
                "heading": {
                    "x": float(forward[0]),
                    "y": float(forward[1]),
                    "z": float(forward[2]),
                },
            }

            self.sample_buffer.append(
                sample
            )

            if self.timestamp_source == "tf":
                self.last_tf_timestamp = (
                    timestamp
                )

            if (
                len(self.sample_buffer)
                >= self.batch_size
            ):
                self._flush_sample_buffer()

        except Exception as exc:
            self.get_logger().warn(
                "Error processing robot pose: "
                f"{exc}"
            )

    def _flush_sample_buffer(
        self,
    ) -> None:
        if not self.sample_buffer:
            return

        samples = self.sample_buffer

        self.sample_buffer = []

        payload = {
            "schema_version": "1.0",
            "source_id": self.source_id,
            "sample_count": len(samples),
            "samples": samples,
        }

        try:
            self.send_queue.put_nowait(
                payload
            )

        except queue.Full:
            self.get_logger().error(
                "HTTP send queue is full. "
                "Dropping one robot ground-truth batch."
            )

    # ----------------------------------------------------------------------
    # HTTP sender
    # ----------------------------------------------------------------------

    def _sender_loop(
        self,
    ) -> None:
        session = requests.Session()

        while (
            not self.stop_event.is_set()
            or not self.send_queue.empty()
        ):
            try:
                payload = self.send_queue.get(
                    timeout=0.2
                )

            except queue.Empty:
                continue

            sent = False

            while (
                not sent
                and not self.stop_event.is_set()
            ):
                try:
                    response = session.post(
                        self.post_url,
                        json=payload,
                        timeout=(
                            self.request_timeout_seconds
                        ),
                    )

                    response.raise_for_status()

                    result = response.json()

                    sent = True

                    self.get_logger().debug(
                        "Posted "
                        f"{payload['sample_count']} "
                        "robot GT samples. "
                        "Pending Quest windows saved: "
                        f"{result.get('pending_windows_saved', 0)}"
                    )

                except Exception as exc:
                    self.get_logger().warn(
                        "Could not post robot ground "
                        "truth to server: "
                        f"{exc}. Retrying."
                    )

                    self.stop_event.wait(
                        self.retry_interval_seconds
                    )

            self.send_queue.task_done()

        session.close()

    # ----------------------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------------------

    def destroy_node(self):
        # Queue any final partial batch.
        self._flush_sample_buffer()

        self.stop_event.set()

        if self.sender_thread.is_alive():
            self.sender_thread.join(
                timeout=2.0
            )

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = RobotGroundTruthNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()