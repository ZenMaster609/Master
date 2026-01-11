#!/usr/bin/env python3
"""
CAN Decoder Node - Decodes wheel encoder CAN messages and publishes structured data.

Subscribes to /can/rx (from ros2_socketcan) and decodes CAN frames with IDs 185/186
(front/rear axle wheel encoders). Publishes structured topics for wheel velocities,
suspension displacements, and steering angle.
"""

import rclpy
from rclpy.node import Node
from can_msgs.msg import Frame
from std_msgs.msg import Float32, Float32MultiArray
from collections import defaultdict
from typing import Optional
import time

from .can_protocol import (
    decode_wheel_pair,
    combine_axle_data,
    WheelPairData,
    CAN_ID_FRONT_AXLE,
    CAN_ID_REAR_AXLE,
)


class CanDecoderNode(Node):
    """
    Decode CAN bus wheel encoder messages and publish structured data.

    Subscribes to /can/rx and filters for wheel encoder CAN IDs (185, 186).
    Synchronizes front and rear axle messages and publishes combined vehicle data.

    Parameters:
        can_topic (str): Input CAN topic (default: /can/rx)
        stale_timeout_ms (int): Timeout for stale data in ms (default: 100)
        publish_rate_hz (float): Publishing rate in Hz (default: 50.0)
        show_stats (bool): Show periodic statistics (default: True)
        stats_interval (float): Statistics interval in seconds (default: 5.0)
    """

    def __init__(self):
        super().__init__('can_decoder_node')

        # Declare parameters
        self.declare_parameter('can_topic', '/can/rx')
        self.declare_parameter('stale_timeout_ms', 100)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('show_stats', True)
        self.declare_parameter('stats_interval', 5.0)

        # Get parameters
        self.can_topic = self.get_parameter('can_topic').value
        self.stale_timeout = self.get_parameter('stale_timeout_ms').value / 1000.0  # Convert to seconds
        self.publish_rate = self.get_parameter('publish_rate_hz').value
        self.show_stats = self.get_parameter('show_stats').value
        self.stats_interval = self.get_parameter('stats_interval').value

        # Latest decoded data for each axle
        self._front_data: Optional[WheelPairData] = None
        self._rear_data: Optional[WheelPairData] = None
        self._front_timestamp: float = 0.0
        self._rear_timestamp: float = 0.0

        # Statistics tracking
        self._frame_counts = defaultdict(int)
        self._decode_errors = defaultdict(int)
        self._last_stats_time = time.time()
        self._stale_warnings = 0

        # Create subscriber
        self._can_sub = self.create_subscription(
            Frame,
            self.can_topic,
            self._can_callback,
            10,  # QoS history depth
        )

        # Create publishers
        self._wheel_vel_pub = self.create_publisher(
            Float32MultiArray,
            '/can/wheel_velocities',
            10,
        )

        self._suspension_pub = self.create_publisher(
            Float32MultiArray,
            '/can/suspension',
            10,
        )

        self._steering_pub = self.create_publisher(
            Float32,
            '/can/steering_angle',
            10,
        )

        # Create timer for periodic publishing
        self._publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self._publish_callback,
        )

        # Create timer for statistics
        if self.show_stats:
            self._stats_timer = self.create_timer(
                self.stats_interval,
                self._stats_callback,
            )

        self.get_logger().info(
            f'CAN Decoder Node started'
            f'\n  Input topic: {self.can_topic}'
            f'\n  Publish rate: {self.publish_rate} Hz'
            f'\n  Stale timeout: {self.stale_timeout*1000:.0f} ms'
            f'\n  Listening for CAN IDs: 0x{CAN_ID_FRONT_AXLE:02X} (front), 0x{CAN_ID_REAR_AXLE:02X} (rear)'
        )

    def _can_callback(self, msg: Frame) -> None:
        """
        CAN frame callback - decode wheel encoder messages.

        Args:
            msg: CAN frame from ros2_socketcan
        """
        can_id = msg.id

        # Filter for wheel encoder CAN IDs
        if can_id not in [CAN_ID_FRONT_AXLE, CAN_ID_REAR_AXLE]:
            return

        # Get current time
        current_time = time.time()

        # Decode the frame
        decoded = decode_wheel_pair(can_id, bytes(msg.data))

        if decoded is None:
            # Decode error
            self._decode_errors[can_id] += 1
            self.get_logger().warn(
                f'Failed to decode CAN ID 0x{can_id:02X}, DLC={msg.dlc}',
                throttle_duration_sec=1.0,
            )
            return

        # Update frame count
        self._frame_counts[can_id] += 1

        # Store decoded data
        if can_id == CAN_ID_FRONT_AXLE:
            self._front_data = decoded
            self._front_timestamp = current_time
        elif can_id == CAN_ID_REAR_AXLE:
            self._rear_data = decoded
            self._rear_timestamp = current_time

    def _publish_callback(self) -> None:
        """Periodic publishing callback - publishes combined vehicle data."""
        current_time = time.time()

        # Check if data is stale
        front_stale = (current_time - self._front_timestamp) > self.stale_timeout
        rear_stale = (current_time - self._rear_timestamp) > self.stale_timeout

        # Warn about stale data (throttled)
        if front_stale and self._front_data is not None:
            self.get_logger().warn(
                f'Front axle data stale ({(current_time - self._front_timestamp)*1000:.0f} ms)',
                throttle_duration_sec=2.0,
            )
            self._stale_warnings += 1

        if rear_stale and self._rear_data is not None:
            self.get_logger().warn(
                f'Rear axle data stale ({(current_time - self._rear_timestamp)*1000:.0f} ms)',
                throttle_duration_sec=2.0,
            )
            self._stale_warnings += 1

        # Use latest data, even if stale (will be zeros if never received)
        front_to_use = None if front_stale else self._front_data
        rear_to_use = None if rear_stale else self._rear_data

        # Combine axle data
        vehicle_data = combine_axle_data(front_to_use, rear_to_use)

        # Publish wheel velocities
        wheel_vel_msg = Float32MultiArray()
        wheel_vel_msg.data = vehicle_data.wheel_velocities
        self._wheel_vel_pub.publish(wheel_vel_msg)

        # Publish suspension displacements
        suspension_msg = Float32MultiArray()
        suspension_msg.data = vehicle_data.suspension_displacements
        self._suspension_pub.publish(suspension_msg)

        # Publish steering angle
        steering_msg = Float32()
        steering_msg.data = vehicle_data.steering_angle if vehicle_data.steering_angle is not None else 0.0
        self._steering_pub.publish(steering_msg)

    def _stats_callback(self) -> None:
        """Periodic statistics callback."""
        current_time = time.time()
        elapsed = current_time - self._last_stats_time

        if elapsed < 0.1:
            return  # Avoid division by zero

        # Compute rates
        front_rate = self._frame_counts[CAN_ID_FRONT_AXLE] / elapsed
        rear_rate = self._frame_counts[CAN_ID_REAR_AXLE] / elapsed

        # Data freshness
        front_age = current_time - self._front_timestamp
        rear_age = current_time - self._rear_timestamp

        self.get_logger().info(
            f'\n=== CAN Decoder Statistics ==='
            f'\n  Front axle (0x{CAN_ID_FRONT_AXLE:02X}): {front_rate:.1f} Hz, age: {front_age*1000:.0f} ms'
            f'\n  Rear axle  (0x{CAN_ID_REAR_AXLE:02X}): {rear_rate:.1f} Hz, age: {rear_age*1000:.0f} ms'
            f'\n  Decode errors: Front={self._decode_errors[CAN_ID_FRONT_AXLE]}, Rear={self._decode_errors[CAN_ID_REAR_AXLE]}'
            f'\n  Stale warnings: {self._stale_warnings}'
        )

        # Reset counters
        self._frame_counts.clear()
        self._decode_errors.clear()
        self._stale_warnings = 0
        self._last_stats_time = current_time


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = CanDecoderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
