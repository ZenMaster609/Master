#!/usr/bin/env python3
"""Republish odometry with a fixed delay for planner/control consumers."""

from __future__ import annotations

from collections import deque
from typing import Deque

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class OdomDelayNode(Node):
    """Buffer odometry messages and republish them after a fixed delay."""

    def __init__(self) -> None:
        super().__init__('odom_delay_node')
        self.declare_parameter('input_topic', '/sim/odom')
        self.declare_parameter('output_topic', '/sim/odom_delayed')
        self.declare_parameter('delay_ms', 40.0)
        self.declare_parameter('flush_rate_hz', 200.0)

        self.input_topic = str(self.get_parameter('input_topic').value).strip() or '/sim/odom'
        self.output_topic = str(self.get_parameter('output_topic').value).strip() or '/sim/odom_delayed'
        self.delay_sec = max(0.0, float(self.get_parameter('delay_ms').value) / 1000.0)
        flush_rate_hz = max(1.0, float(self.get_parameter('flush_rate_hz').value))

        self._buffer: Deque[tuple[float, Odometry]] = deque()
        self._publisher = self.create_publisher(Odometry, self.output_topic, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.input_topic, self._odom_cb, qos_profile_sensor_data)
        self.create_timer(1.0 / flush_rate_hz, self._flush_ready_messages)

        self.get_logger().info(
            f'odom_delay_node ready input={self.input_topic} output={self.output_topic} '
            f'delay_ms={self.delay_sec * 1000.0:.1f}'
        )

    def _odom_cb(self, msg: Odometry) -> None:
        stamp = msg.header.stamp
        stamp_sec = float(stamp.sec) + (float(stamp.nanosec) * 1e-9)
        if stamp_sec <= 0.0:
            stamp_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        self._buffer.append((stamp_sec + self.delay_sec, msg))

    def _flush_ready_messages(self) -> None:
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        while self._buffer and self._buffer[0][0] <= now_sec:
            _release_time_sec, msg = self._buffer.popleft()
            self._publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomDelayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
