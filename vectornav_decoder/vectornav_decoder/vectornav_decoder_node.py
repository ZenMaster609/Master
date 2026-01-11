#!/usr/bin/env python3
"""
VectorNav Decoder Node - Decodes VN-200 binary data and publishes ROS2 topics.

Connects to VN-200 via serial, parses binary packets, and publishes:
- /vectornav/imu (sensor_msgs/Imu)
- /vectornav/gps (sensor_msgs/NavSatFix)
- /vectornav/ins (nav_msgs/Odometry)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header
import time
import os
import math

from .vn200_protocol import VN200Packet, GPSFix, INSMode
from .vn200_serial import VN200Serial
from .vn200_config import (
    VectorNavConfig,
    SerialConfig,
    load_config_from_yaml,
    create_default_config,
)


class VectorNavDecoderNode(Node):
    """
    VectorNav VN-200 decoder node.

    Connects to VN-200 via serial, configures binary output, parses packets,
    and publishes to ROS2 topics.

    Parameters:
        config_file (str): Path to YAML configuration file
        serial_port (str): Serial port override (default from config)
        baudrate (int): Baud rate override (default from config)
        publish_rate_hz (float): Max publish rate (default: 200.0)
        frame_id (str): TF frame ID (default: 'vectornav')
        imu_topic (str): IMU topic name (default: /vectornav/imu)
        gps_topic (str): GPS topic name (default: /vectornav/gps)
        ins_topic (str): INS topic name (default: /vectornav/ins)
        show_stats (bool): Show periodic statistics (default: True)
        stats_interval (float): Statistics interval in seconds (default: 5.0)
    """

    def __init__(self):
        super().__init__('vectornav_decoder_node')

        # Declare parameters
        self.declare_parameter('config_file', '')
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 921600)
        self.declare_parameter('publish_rate_hz', 200.0)
        self.declare_parameter('frame_id', 'vectornav')
        self.declare_parameter('imu_topic', '/vectornav/imu')
        self.declare_parameter('gps_topic', '/vectornav/gps')
        self.declare_parameter('ins_topic', '/vectornav/ins')
        self.declare_parameter('show_stats', True)
        self.declare_parameter('stats_interval', 5.0)

        # Get parameters
        config_file = self.get_parameter('config_file').value
        self.serial_port = self.get_parameter('serial_port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.publish_rate = self.get_parameter('publish_rate_hz').value
        self.frame_id = self.get_parameter('frame_id').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.gps_topic = self.get_parameter('gps_topic').value
        self.ins_topic = self.get_parameter('ins_topic').value
        self.show_stats = self.get_parameter('show_stats').value
        self.stats_interval = self.get_parameter('stats_interval').value

        # Load configuration
        if config_file and os.path.exists(config_file):
            self.config = load_config_from_yaml(config_file)
            self.get_logger().info(f"Loaded config from {config_file}")
        else:
            # Use default configuration
            self.config = VectorNavConfig()
            self.config.binary_output = create_default_config()
            self.get_logger().info("Using default configuration")

        # Override serial settings from parameters
        self.config.serial.port = self.serial_port
        self.config.serial.baudrate = self.baudrate

        # Statistics
        self._packet_count = 0
        self._imu_count = 0
        self._gps_count = 0
        self._ins_count = 0
        self._last_stats_time = time.time()

        # Create publishers
        self._imu_pub = self.create_publisher(Imu, self.imu_topic, 10)
        self._gps_pub = self.create_publisher(NavSatFix, self.gps_topic, 10)
        self._ins_pub = self.create_publisher(Odometry, self.ins_topic, 10)

        # Create serial handler
        serial_config = SerialConfig(
            port=self.config.serial.port,
            baudrate=self.config.serial.baudrate,
            timeout_sec=self.config.serial.timeout_sec,
            reconnect_delay_sec=self.config.serial.reconnect_delay_sec,
            max_reconnect_attempts=self.config.serial.max_reconnect_attempts,
        )

        self._serial = VN200Serial(
            config=serial_config,
            binary_config=self.config.binary_output,
            logger=self.get_logger(),
        )

        self._serial.set_packet_callback(self._on_packet)
        self._serial.set_error_callback(self._on_error)

        # Connect and start streaming
        if self._serial.connect():
            self._serial.start_streaming()
            self.get_logger().info(
                f"VectorNav decoder started"
                f"\n  Port: {self.config.serial.port}"
                f"\n  Baudrate: {self.config.serial.baudrate}"
                f"\n  Output rate: {self.config.binary_output.output_rate_hz:.1f} Hz"
            )
        else:
            self.get_logger().error("Failed to connect to VN-200")

        # Statistics timer
        if self.show_stats:
            self._stats_timer = self.create_timer(
                self.stats_interval,
                self._stats_callback,
            )

    def _on_packet(self, packet: VN200Packet) -> None:
        """Handle received VN-200 packet."""
        self._packet_count += 1

        now = self.get_clock().now()
        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = self.frame_id

        # Publish IMU data
        if packet.has_common or packet.has_imu:
            self._publish_imu(packet, header)

        # Publish GPS data
        if packet.has_gps:
            self._publish_gps(packet, header)

        # Publish INS data
        if packet.has_ins:
            self._publish_ins(packet, header)

    def _publish_imu(self, packet: VN200Packet, header: Header) -> None:
        """Publish IMU message."""
        msg = Imu()
        msg.header = header

        # Orientation from common group (yaw-pitch-roll to quaternion)
        if packet.common.yaw_deg is not None:
            q = self._ypr_to_quaternion(
                packet.common.yaw_rad,
                packet.common.pitch_rad,
                packet.common.roll_rad,
            )
            msg.orientation = q
            msg.orientation_covariance[0] = 0.01  # Small uncertainty
        elif packet.common.quat_w is not None:
            msg.orientation.w = packet.common.quat_w
            msg.orientation.x = packet.common.quat_x
            msg.orientation.y = packet.common.quat_y
            msg.orientation.z = packet.common.quat_z
            msg.orientation_covariance[0] = 0.01
        else:
            msg.orientation_covariance[0] = -1  # Unknown

        # Angular velocity (prefer common, fallback to IMU group)
        if packet.common.angular_rate_x is not None:
            msg.angular_velocity.x = packet.common.angular_rate_x
            msg.angular_velocity.y = packet.common.angular_rate_y
            msg.angular_velocity.z = packet.common.angular_rate_z
        elif packet.imu.angular_rate_x is not None:
            msg.angular_velocity.x = packet.imu.angular_rate_x
            msg.angular_velocity.y = packet.imu.angular_rate_y
            msg.angular_velocity.z = packet.imu.angular_rate_z
        elif packet.imu.uncomp_gyro_x is not None:
            msg.angular_velocity.x = packet.imu.uncomp_gyro_x
            msg.angular_velocity.y = packet.imu.uncomp_gyro_y
            msg.angular_velocity.z = packet.imu.uncomp_gyro_z

        # Linear acceleration (prefer common, fallback to IMU group)
        if packet.common.accel_x is not None:
            msg.linear_acceleration.x = packet.common.accel_x
            msg.linear_acceleration.y = packet.common.accel_y
            msg.linear_acceleration.z = packet.common.accel_z
        elif packet.imu.accel_x is not None:
            msg.linear_acceleration.x = packet.imu.accel_x
            msg.linear_acceleration.y = packet.imu.accel_y
            msg.linear_acceleration.z = packet.imu.accel_z
        elif packet.imu.uncomp_accel_x is not None:
            msg.linear_acceleration.x = packet.imu.uncomp_accel_x
            msg.linear_acceleration.y = packet.imu.uncomp_accel_y
            msg.linear_acceleration.z = packet.imu.uncomp_accel_z

        self._imu_pub.publish(msg)
        self._imu_count += 1

    def _publish_gps(self, packet: VN200Packet, header: Header) -> None:
        """Publish GPS message."""
        msg = NavSatFix()
        msg.header = header

        # Status
        msg.status = NavSatStatus()
        if packet.gps.fix is None or packet.gps.fix == GPSFix.NO_FIX:
            msg.status.status = NavSatStatus.STATUS_NO_FIX
        elif packet.gps.fix == GPSFix.TIME_ONLY:
            msg.status.status = NavSatStatus.STATUS_NO_FIX
        elif packet.gps.fix == GPSFix.FIX_2D:
            msg.status.status = NavSatStatus.STATUS_FIX
        elif packet.gps.fix == GPSFix.FIX_3D:
            msg.status.status = NavSatStatus.STATUS_FIX
        elif packet.gps.fix == GPSFix.SBAS:
            msg.status.status = NavSatStatus.STATUS_SBAS_FIX
        elif packet.gps.fix in [GPSFix.RTK_FLOAT, GPSFix.RTK_FIXED]:
            msg.status.status = NavSatStatus.STATUS_GBAS_FIX

        msg.status.service = NavSatStatus.SERVICE_GPS

        # Position
        if packet.gps.latitude_deg is not None:
            msg.latitude = packet.gps.latitude_deg
            msg.longitude = packet.gps.longitude_deg
            msg.altitude = packet.gps.altitude_m if packet.gps.altitude_m is not None else 0.0

        # Covariance
        if packet.gps.pos_uncertainty_m is not None:
            n, e, d = packet.gps.pos_uncertainty_m
            msg.position_covariance = [
                n * n, 0.0, 0.0,
                0.0, e * e, 0.0,
                0.0, 0.0, d * d,
            ]
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        else:
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        self._gps_pub.publish(msg)
        self._gps_count += 1

    def _publish_ins(self, packet: VN200Packet, header: Header) -> None:
        """Publish INS odometry message."""
        msg = Odometry()
        msg.header = header
        msg.child_frame_id = 'base_link'

        # Position from INS
        if packet.ins.latitude_deg is not None:
            # Store lat/lon in position (for now - would need local frame conversion)
            msg.pose.pose.position.x = packet.ins.longitude_deg
            msg.pose.pose.position.y = packet.ins.latitude_deg
            msg.pose.pose.position.z = packet.ins.altitude_m if packet.ins.altitude_m is not None else 0.0

        # Orientation from common group
        if packet.common.yaw_deg is not None:
            msg.pose.pose.orientation = self._ypr_to_quaternion(
                packet.common.yaw_rad,
                packet.common.pitch_rad,
                packet.common.roll_rad,
            )
        elif packet.common.quat_w is not None:
            msg.pose.pose.orientation.w = packet.common.quat_w
            msg.pose.pose.orientation.x = packet.common.quat_x
            msg.pose.pose.orientation.y = packet.common.quat_y
            msg.pose.pose.orientation.z = packet.common.quat_z

        # Body-frame velocity from INS
        if packet.ins.vel_body_x is not None:
            msg.twist.twist.linear.x = packet.ins.vel_body_x
            msg.twist.twist.linear.y = packet.ins.vel_body_y
            msg.twist.twist.linear.z = packet.ins.vel_body_z

        # Angular velocity from common/IMU
        if packet.common.angular_rate_x is not None:
            msg.twist.twist.angular.x = packet.common.angular_rate_x
            msg.twist.twist.angular.y = packet.common.angular_rate_y
            msg.twist.twist.angular.z = packet.common.angular_rate_z

        # Position covariance
        if packet.ins.pos_uncertainty_m is not None:
            pu = packet.ins.pos_uncertainty_m
            msg.pose.covariance[0] = pu * pu
            msg.pose.covariance[7] = pu * pu
            msg.pose.covariance[14] = pu * pu

        # Velocity covariance
        if packet.ins.vel_uncertainty_mps is not None:
            vu = packet.ins.vel_uncertainty_mps
            msg.twist.covariance[0] = vu * vu
            msg.twist.covariance[7] = vu * vu
            msg.twist.covariance[14] = vu * vu

        self._ins_pub.publish(msg)
        self._ins_count += 1

    def _ypr_to_quaternion(self, yaw: float, pitch: float, roll: float) -> Quaternion:
        """
        Convert yaw-pitch-roll (radians) to quaternion.

        Args:
            yaw: Yaw angle in radians
            pitch: Pitch angle in radians
            roll: Roll angle in radians

        Returns:
            ROS Quaternion message
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy

        return q

    def _on_error(self, error: Exception) -> None:
        """Handle serial errors."""
        self.get_logger().error(f"Serial error: {error}")

    def _stats_callback(self) -> None:
        """Print periodic statistics."""
        current_time = time.time()
        elapsed = current_time - self._last_stats_time

        if elapsed < 0.1:
            return

        packet_rate = self._packet_count / elapsed
        imu_rate = self._imu_count / elapsed

        stats = self._serial.stats
        parser_stats = stats.get('parser_stats', {})

        self.get_logger().info(
            f"\n=== VectorNav Statistics ==="
            f"\n  Packet rate: {packet_rate:.1f} Hz"
            f"\n  IMU published: {self._imu_count} ({imu_rate:.1f} Hz)"
            f"\n  GPS published: {self._gps_count}"
            f"\n  INS published: {self._ins_count}"
            f"\n  Bytes received: {stats.get('bytes_received', 0)}"
            f"\n  CRC errors: {parser_stats.get('crc_errors', 0)}"
        )

        # Reset counters
        self._packet_count = 0
        self._imu_count = 0
        self._gps_count = 0
        self._ins_count = 0
        self._last_stats_time = current_time

    def destroy_node(self):
        """Clean up on shutdown."""
        if hasattr(self, '_serial'):
            self._serial.stop_streaming()
            self._serial.disconnect()
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = VectorNavDecoderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
