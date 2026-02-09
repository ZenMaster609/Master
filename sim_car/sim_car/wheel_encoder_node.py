#!/usr/bin/env python3

"""
Wheel encoder node for the simulated car.
Subscribes to wheel joint states and publishes wheel RPM and speed.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from .topic_utils import apply_topic_prefix


class WheelEncoderNode(Node):
    def __init__(self):
        super().__init__('wheel_encoder_node')

        # Declare parameters
        self.declare_parameter('publish_rate', 50.0)  # Hz
        self.declare_parameter('publish_rpm', True)
        self.declare_parameter('publish_angle_accum', True)
        self.declare_parameter('publish_speed_mm_s', True)
        self.declare_parameter('wheel_radius', 0.23)  # meters
        self.declare_parameter('min_dt', 1e-4)  # seconds
        self.declare_parameter('min_window_sec', 0.04)  # seconds
        self.declare_parameter('min_delta', 1e-4)  # radians
        self.declare_parameter('topic_prefix', '/sim/raw')

        self.publish_rate = self.get_parameter('publish_rate').value
        self.publish_rpm = self.get_parameter('publish_rpm').value
        self.publish_angle_accum = self.get_parameter('publish_angle_accum').value
        self.publish_speed_mm_s = self.get_parameter('publish_speed_mm_s').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.min_dt = self.get_parameter('min_dt').value
        self.min_window_sec = self.get_parameter('min_window_sec').value
        self.min_delta = self.get_parameter('min_delta').value
        self.topic_prefix = str(self.get_parameter('topic_prefix').value)

        # Subscribe to joint states from Gazebo (via ros_gz_bridge)
        # Uses /sim/raw/ namespace for all raw simulation topics
        joint_states_topic = apply_topic_prefix('/sim/raw/joint_states', self.topic_prefix)
        self.joint_state_sub = self.create_subscription(
            JointState,
            joint_states_topic,
            self.joint_state_callback,
            10
        )

        # Publisher for wheel RPM under /sim/raw/ namespace
        self.encoder_rpm_pub = None
        if self.publish_rpm:
            self.encoder_rpm_pub = self.create_publisher(
                Float32MultiArray,
                apply_topic_prefix('/sim/raw/wheel_encoder/rpm', self.topic_prefix),
                10
            )
        self.encoder_angle_pub = None
        if self.publish_angle_accum:
            self.encoder_angle_pub = self.create_publisher(
                Float32MultiArray,
                apply_topic_prefix('/sim/raw/wheel_encoder/angle_accum', self.topic_prefix),
                10
            )
        self.encoder_speed_pub = None
        if self.publish_speed_mm_s:
            self.encoder_speed_pub = self.create_publisher(
                Float32MultiArray,
                apply_topic_prefix('/sim/raw/wheel_encoder/speed_mm_s', self.topic_prefix),
                10
            )

        # Wheel names (for reference)
        self.wheel_names = [
            'front_left_wheel_joint',
            'front_right_wheel_joint',
            'rear_left_wheel_joint',
            'rear_right_wheel_joint'
        ]

        # Store last RPM for status updates
        self._last_rpm = [0.0, 0.0, 0.0, 0.0]
        self._joint_state_received = False
        self._last_positions = {}
        self._last_stamp = {}
        self._angle_accum = {name: 0.0 for name in self.wheel_names}
        self._time_accum = {name: 0.0 for name in self.wheel_names}

        # Publish at configured rate
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_wheel_velocities,
        )

        # Timer for status updates
        self.status_timer = self.create_timer(2.0, self.status_callback)

        self.get_logger().info('Wheel encoder node started')
        self.get_logger().info(f'Publishing at {self.publish_rate} Hz')
        if self.publish_rpm:
            rpm_topic = apply_topic_prefix('/sim/raw/wheel_encoder/rpm', self.topic_prefix)
            self.get_logger().info(f'Publishing wheel RPM on {rpm_topic}')
        if self.publish_angle_accum:
            angle_topic = apply_topic_prefix('/sim/raw/wheel_encoder/angle_accum', self.topic_prefix)
            self.get_logger().info(f'Publishing wheel angle accum on {angle_topic}')
        if self.publish_speed_mm_s:
            speed_topic = apply_topic_prefix('/sim/raw/wheel_encoder/speed_mm_s', self.topic_prefix)
            self.get_logger().info(f'Publishing wheel speeds on {speed_topic}')

    def _rpm_to_mm_s(self, rpm: float) -> float:
        """Convert wheel RPM to linear speed in mm/s."""
        return rpm * self.wheel_radius * 2.0 * math.pi / 60.0 * 1000.0

    def _get_msg_time(self, msg: JointState) -> float:
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return self.get_clock().now().nanoseconds * 1e-9

    def joint_state_callback(self, msg):
        """Process joint state data and accumulate wheel rotation."""
        msg_time = self._get_msg_time(msg)

        name_to_index = {name: i for i, name in enumerate(msg.name)}

        for idx_name, wheel_name in enumerate(self.wheel_names):
            idx = name_to_index.get(wheel_name)
            if idx is None:
                continue

            if idx < len(msg.position):
                pos = msg.position[idx]
                last_pos = self._last_positions.get(wheel_name)
                last_time = self._last_stamp.get(wheel_name)
                if last_pos is not None and last_time is not None:
                    dt = msg_time - last_time
                    if dt >= self.min_dt:
                        delta = pos - last_pos
                        # Unwrap assuming per-sample delta stays within [-pi, pi]
                        while delta > math.pi:
                            delta -= 2.0 * math.pi
                        while delta < -math.pi:
                            delta += 2.0 * math.pi
                        if abs(delta) >= self.min_delta:
                            self._angle_accum[wheel_name] += abs(delta)
                            self._time_accum[wheel_name] += dt
                self._last_positions[wheel_name] = pos
                self._last_stamp[wheel_name] = msg_time

        self._joint_state_received = True

    def publish_wheel_velocities(self):
        """Publish wheel RPM at the configured rate."""
        if not self._joint_state_received or self.wheel_radius <= 0.0:
            return

        wheel_rpm = []
        wheel_angle_accum = []
        wheel_speed_mm_s = []
        for idx_name, wheel_name in enumerate(self.wheel_names):
            accum_time = self._time_accum[wheel_name]
            angle_accum = self._angle_accum[wheel_name]
            if accum_time >= self.min_window_sec and accum_time > 0.0:
                omega_avg = angle_accum / accum_time
                rpm = omega_avg * 60.0 / (2.0 * math.pi)
                self._angle_accum[wheel_name] = 0.0
                self._time_accum[wheel_name] = 0.0
            else:
                rpm = self._last_rpm[idx_name]
            wheel_rpm.append(rpm)
            wheel_angle_accum.append(angle_accum)
            wheel_speed_mm_s.append(self._rpm_to_mm_s(rpm))
        self._last_rpm = wheel_rpm

        if self.encoder_rpm_pub is not None:
            rpm_msg = Float32MultiArray()
            rpm_msg.data = wheel_rpm
            self.encoder_rpm_pub.publish(rpm_msg)
        if self.encoder_angle_pub is not None:
            angle_msg = Float32MultiArray()
            angle_msg.data = wheel_angle_accum
            self.encoder_angle_pub.publish(angle_msg)
        if self.encoder_speed_pub is not None:
            speed_msg = Float32MultiArray()
            speed_msg.data = wheel_speed_mm_s
            self.encoder_speed_pub.publish(speed_msg)

    def status_callback(self):
        """Periodic status update"""
        if self.publish_rpm:
            self.get_logger().info(
                f'Wheel RPM - FL: {self._last_rpm[0]:.2f}, '
                f'FR: {self._last_rpm[1]:.2f}, '
                f'RL: {self._last_rpm[2]:.2f}, '
                f'RR: {self._last_rpm[3]:.2f}'
            )


def main(args=None):
    rclpy.init(args=args)

    node = WheelEncoderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
