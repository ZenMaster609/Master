#!/usr/bin/env python3

"""
Sensor processor node for the simulated car.
Subscribes to IMU, GPS, Odometry, and Wheel Encoder data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import math


class SensorProcessorNode(Node):
    def __init__(self):
        super().__init__('sensor_processor_node')

        # Declare parameters
        self.declare_parameter('publish_rate', 1.0)  # Hz
        publish_rate = self.get_parameter('publish_rate').value

        # Topic parameters (default to /sim namespace for EUFS Gazebo sim)
        self.declare_parameter('imu_topic', '/sim/imu')
        self.declare_parameter('gps_topic', '/sim/navsat')
        self.declare_parameter('odom_topic', '/sim/odom')
        self.declare_parameter('encoder_topic', '/sim/wheel_encoder/velocities')

        imu_topic = self.get_parameter('imu_topic').value
        gps_topic = self.get_parameter('gps_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        encoder_topic = self.get_parameter('encoder_topic').value

        # Create subscriptions for all sensors
        self.imu_sub = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10
        )

        self.gps_sub = self.create_subscription(
            NavSatFix,
            gps_topic,
            self.gps_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )

        self.encoder_velocity_sub = self.create_subscription(
            Float32MultiArray,
            encoder_topic,
            self.encoder_velocity_callback,
            10
        )

        # Data storage
        self.last_imu_data = None
        self.last_gps_data = None
        self.last_odom_data = None
        self.last_encoder_velocities = None

        # Timer for periodic status updates
        self.timer = self.create_timer(1.0 / publish_rate, self.status_callback)

        self.get_logger().info('Sensor processor node started')
        self.get_logger().info(
            f'Subscribing to: {imu_topic}, {gps_topic}, {odom_topic}, {encoder_topic}'
        )

    def imu_callback(self, msg):
        """Process IMU data"""
        self.last_imu_data = msg

        # Extract orientation (quaternion)
        q = msg.orientation
        # Convert to Euler angles (simplified)
        # This is a simplified conversion - for production use tf_transformations
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # self.get_logger().debug(f'IMU yaw: {math.degrees(yaw):.1f}°')

    def gps_callback(self, msg):
        """Process GPS data"""
        self.last_gps_data = msg
        # self.get_logger().debug(
        #     f'GPS: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}, alt={msg.altitude:.2f}m'
        # )

    def odom_callback(self, msg):
        """Process odometry data"""
        self.last_odom_data = msg

        # Extract position and velocity
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear

        # self.get_logger().debug(
        #     f'Odom: pos=({pos.x:.2f}, {pos.y:.2f}), vel=({vel.x:.2f}, {vel.y:.2f})'
        # )

    def encoder_velocity_callback(self, msg):
        """Process wheel encoder velocity data"""
        self.last_encoder_velocities = msg.data
        # self.get_logger().debug(
        #     f'Wheel velocities (m/s): FL={msg.data[0]:.2f}, FR={msg.data[1]:.2f}, '
        #     f'RL={msg.data[2]:.2f}, RR={msg.data[3]:.2f}'
        # )

    def status_callback(self):
        """Periodic status update"""
        status_parts = []

        if self.last_odom_data:
            pos = self.last_odom_data.pose.pose.position
            vel = self.last_odom_data.twist.twist.linear
            speed = math.sqrt(vel.x**2 + vel.y**2)
            status_parts.append(f'Pos: ({pos.x:.2f}, {pos.y:.2f}), Speed: {speed:.2f} m/s')

        if self.last_gps_data:
            status_parts.append(
                f'GPS: ({self.last_gps_data.latitude:.6f}, {self.last_gps_data.longitude:.6f})'
            )

        if self.last_imu_data:
            # Simplified yaw calculation
            q = self.last_imu_data.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            status_parts.append(f'Heading: {math.degrees(yaw):.1f}°')

        if self.last_encoder_velocities:
            # Show average wheel velocity
            avg_vel = sum(self.last_encoder_velocities) / len(self.last_encoder_velocities)
            status_parts.append(f'Avg Wheel Vel: {avg_vel:.2f} m/s')

        if status_parts:
            self.get_logger().info(' | '.join(status_parts))
        else:
            self.get_logger().warn('No sensor data received yet')


def main(args=None):
    rclpy.init(args=args)

    node = SensorProcessorNode()

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
