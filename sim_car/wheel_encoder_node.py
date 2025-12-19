#!/usr/bin/env python3

"""
Wheel encoder node for the simulated car.
Subscribes to wheel joint states and publishes encoder ticks and velocities.
Simulates realistic wheel encoders with configurable ticks per revolution.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray, Float32MultiArray
import math


class WheelEncoderNode(Node):
    def __init__(self):
        super().__init__('wheel_encoder_node')

        # Declare parameters
        self.declare_parameter('ticks_per_revolution', 2048)  # Common encoder resolution
        self.declare_parameter('wheel_radius', 0.1)  # meters
        self.declare_parameter('publish_rate', 50.0)  # Hz

        self.ticks_per_rev = self.get_parameter('ticks_per_revolution').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # Subscribe to joint states from the wheel encoder plugin
        self.joint_state_sub = self.create_subscription(
            JointState,
            'wheel_encoder/joint_states',
            self.joint_state_callback,
            10
        )

        # Publishers for encoder data
        self.encoder_ticks_pub = self.create_publisher(
            Int32MultiArray,
            'wheel_encoder/ticks',
            10
        )

        self.encoder_velocity_pub = self.create_publisher(
            Float32MultiArray,
            'wheel_encoder/velocities',
            10
        )

        # Store previous positions for tick counting
        self.previous_positions = None
        self.cumulative_ticks = [0, 0, 0, 0]  # FL, FR, RL, RR

        # Wheel names (for reference)
        self.wheel_names = [
            'front_left_wheel_joint',
            'front_right_wheel_joint',
            'rear_left_wheel_joint',
            'rear_right_wheel_joint'
        ]

        # Timer for status updates
        self.status_timer = self.create_timer(2.0, self.status_callback)

        self.get_logger().info('Wheel encoder node started')
        self.get_logger().info(f'Ticks per revolution: {self.ticks_per_rev}')
        self.get_logger().info(f'Wheel radius: {self.wheel_radius} m')
        self.get_logger().info(f'Publishing at {self.publish_rate} Hz')

    def joint_state_callback(self, msg):
        """Process joint state data and convert to encoder ticks"""

        # Create a mapping of joint names to positions and velocities
        joint_data = {}
        for i, name in enumerate(msg.name):
            joint_data[name] = {
                'position': msg.position[i] if i < len(msg.position) else 0.0,
                'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0
            }

        # Extract data for our wheels in order
        positions = []
        velocities = []
        for wheel_name in self.wheel_names:
            if wheel_name in joint_data:
                positions.append(joint_data[wheel_name]['position'])
                velocities.append(joint_data[wheel_name]['velocity'])
            else:
                positions.append(0.0)
                velocities.append(0.0)

        # Calculate encoder ticks from positions (angular position in radians)
        if self.previous_positions is not None:
            # Calculate delta position for each wheel
            for i in range(4):
                delta_position = positions[i] - self.previous_positions[i]

                # Handle angle wrapping (continuous joints can exceed 2*pi)
                # Convert delta radians to ticks
                delta_ticks = int((delta_position / (2 * math.pi)) * self.ticks_per_rev)
                self.cumulative_ticks[i] += delta_ticks

        self.previous_positions = positions.copy()

        # Publish encoder ticks
        ticks_msg = Int32MultiArray()
        ticks_msg.data = self.cumulative_ticks.copy()
        self.encoder_ticks_pub.publish(ticks_msg)

        # Publish velocities (rad/s to m/s)
        velocity_msg = Float32MultiArray()
        velocity_msg.data = [v * self.wheel_radius for v in velocities]
        self.encoder_velocity_pub.publish(velocity_msg)

    def status_callback(self):
        """Periodic status update"""
        if self.previous_positions is not None:
            self.get_logger().info(
                f'Encoder Ticks - FL: {self.cumulative_ticks[0]}, '
                f'FR: {self.cumulative_ticks[1]}, '
                f'RL: {self.cumulative_ticks[2]}, '
                f'RR: {self.cumulative_ticks[3]}'
            )

            # Calculate distance traveled by each wheel
            distances = [
                (ticks / self.ticks_per_rev) * (2 * math.pi * self.wheel_radius)
                for ticks in self.cumulative_ticks
            ]
            avg_distance = sum(distances) / len(distances)
            self.get_logger().info(f'Average distance traveled: {avg_distance:.2f} m')


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
