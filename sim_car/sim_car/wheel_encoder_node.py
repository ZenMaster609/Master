#!/usr/bin/env python3

"""
Wheel encoder node for the simulated car.
Subscribes to wheel joint states and publishes wheel velocities in m/s.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


class WheelEncoderNode(Node):
    def __init__(self):
        super().__init__('wheel_encoder_node')

        # Declare parameters
        self.declare_parameter('wheel_radius', 0.1)  # meters
        self.declare_parameter('publish_rate', 50.0)  # Hz

        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # Subscribe to joint states from Gazebo (via ros_gz_bridge)
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publisher for wheel velocities
        self.encoder_velocity_pub = self.create_publisher(
            Float32MultiArray,
            'wheel_encoder/velocities',
            10
        )

        # Wheel names (for reference)
        self.wheel_names = [
            'front_left_wheel_joint',
            'front_right_wheel_joint',
            'rear_left_wheel_joint',
            'rear_right_wheel_joint'
        ]

        # Store last velocities for status updates
        self._last_velocities = [0.0, 0.0, 0.0, 0.0]

        # Timer for status updates
        self.status_timer = self.create_timer(2.0, self.status_callback)

        self.get_logger().info('Wheel encoder node started')
        self.get_logger().info(f'Wheel radius: {self.wheel_radius} m')
        self.get_logger().info(f'Publishing at {self.publish_rate} Hz')

    def joint_state_callback(self, msg):
        """Process joint state data and publish wheel velocities."""

        # Create a mapping of joint names to velocities
        joint_data = {}
        for i, name in enumerate(msg.name):
            joint_data[name] = {
                'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0
            }

        # Extract velocities for our wheels in order [FL, FR, RL, RR]
        velocities = []
        for wheel_name in self.wheel_names:
            if wheel_name in joint_data:
                velocities.append(joint_data[wheel_name]['velocity'])
            else:
                velocities.append(0.0)

        # Convert rad/s to m/s and publish
        wheel_velocities = [v * self.wheel_radius for v in velocities]
        self._last_velocities = wheel_velocities

        velocity_msg = Float32MultiArray()
        velocity_msg.data = wheel_velocities
        self.encoder_velocity_pub.publish(velocity_msg)

    def status_callback(self):
        """Periodic status update"""
        self.get_logger().info(
            f'Wheel Velocities (m/s) - FL: {self._last_velocities[0]:.2f}, '
            f'FR: {self._last_velocities[1]:.2f}, '
            f'RL: {self._last_velocities[2]:.2f}, '
            f'RR: {self._last_velocities[3]:.2f}'
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
