#!/usr/bin/env python3
"""
VectorNav Monitor Node - Debug tool for raw packet inspection.

Connects to VN-200 via serial and displays raw packet information
for debugging and development purposes.
"""

import rclpy
from rclpy.node import Node
import time
import os

from .vn200_protocol import VN200Packet, GPSFix, INSMode
from .vn200_serial import VN200Serial
from .vn200_config import (
    VectorNavConfig,
    SerialConfig,
    load_config_from_yaml,
    create_default_config,
)


class VectorNavMonitorNode(Node):
    """
    VectorNav VN-200 monitor node for debugging.

    Displays raw packet information including:
    - Packet rate and statistics
    - Field values
    - CRC errors and sync losses

    Parameters:
        config_file (str): Path to YAML configuration file
        serial_port (str): Serial port (default: /dev/ttyUSB0)
        baudrate (int): Baud rate (default: 921600)
        verbose (bool): Show detailed field values (default: False)
        stats_interval (float): Statistics interval in seconds (default: 5.0)
    """

    def __init__(self):
        super().__init__('vectornav_monitor_node')

        # Declare parameters
        self.declare_parameter('config_file', '')
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 921600)
        self.declare_parameter('verbose', False)
        self.declare_parameter('stats_interval', 5.0)

        # Get parameters
        config_file = self.get_parameter('config_file').value
        self.serial_port = self.get_parameter('serial_port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.verbose = self.get_parameter('verbose').value
        self.stats_interval = self.get_parameter('stats_interval').value

        # Load configuration
        if config_file and os.path.exists(config_file):
            self.config = load_config_from_yaml(config_file)
            self.get_logger().info(f"Loaded config from {config_file}")
        else:
            self.config = VectorNavConfig()
            self.config.binary_output = create_default_config()
            self.get_logger().info("Using default configuration")

        # Override serial settings
        self.config.serial.port = self.serial_port
        self.config.serial.baudrate = self.baudrate

        # Statistics
        self._packet_count = 0
        self._last_stats_time = time.time()
        self._groups_seen = set()

        # Create serial handler
        serial_config = SerialConfig(
            port=self.config.serial.port,
            baudrate=self.config.serial.baudrate,
            timeout_sec=self.config.serial.timeout_sec,
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
                f"VectorNav monitor started"
                f"\n  Port: {self.config.serial.port}"
                f"\n  Baudrate: {self.config.serial.baudrate}"
                f"\n  Verbose: {self.verbose}"
            )
        else:
            self.get_logger().error("Failed to connect to VN-200")

        # Statistics timer
        self._stats_timer = self.create_timer(
            self.stats_interval,
            self._stats_callback,
        )

    def _on_packet(self, packet: VN200Packet) -> None:
        """Handle received VN-200 packet."""
        self._packet_count += 1

        # Track which groups we've seen
        if packet.has_common:
            self._groups_seen.add('Common')
        if packet.has_imu:
            self._groups_seen.add('IMU')
        if packet.has_gps:
            self._groups_seen.add('GPS')
        if packet.has_attitude:
            self._groups_seen.add('Attitude')
        if packet.has_ins:
            self._groups_seen.add('INS')

        # Verbose output
        if self.verbose:
            self._print_packet(packet)

    def _print_packet(self, packet: VN200Packet) -> None:
        """Print detailed packet information."""
        lines = [f"--- Packet #{self._packet_count} ---"]

        if packet.has_common:
            lines.append("  Common:")
            if packet.common.yaw_deg is not None:
                lines.append(f"    YPR: {packet.common.yaw_deg:.2f}, {packet.common.pitch_deg:.2f}, {packet.common.roll_deg:.2f} deg")
            if packet.common.angular_rate_x is not None:
                lines.append(f"    AngRate: {packet.common.angular_rate_x:.4f}, {packet.common.angular_rate_y:.4f}, {packet.common.angular_rate_z:.4f} rad/s")
            if packet.common.accel_x is not None:
                lines.append(f"    Accel: {packet.common.accel_x:.4f}, {packet.common.accel_y:.4f}, {packet.common.accel_z:.4f} m/s^2")

        if packet.has_imu:
            lines.append("  IMU:")
            if packet.imu.temperature_c is not None:
                lines.append(f"    Temp: {packet.imu.temperature_c:.2f} C")
            if packet.imu.uncomp_accel_x is not None:
                lines.append(f"    UncompAccel: {packet.imu.uncomp_accel_x:.4f}, {packet.imu.uncomp_accel_y:.4f}, {packet.imu.uncomp_accel_z:.4f} m/s^2")
            if packet.imu.uncomp_gyro_x is not None:
                lines.append(f"    UncompGyro: {packet.imu.uncomp_gyro_x:.4f}, {packet.imu.uncomp_gyro_y:.4f}, {packet.imu.uncomp_gyro_z:.4f} rad/s")

        if packet.has_gps:
            lines.append("  GPS:")
            if packet.gps.fix is not None:
                lines.append(f"    Fix: {packet.gps.fix.name}, Sats: {packet.gps.num_sats}")
            if packet.gps.latitude_deg is not None:
                lines.append(f"    Pos: {packet.gps.latitude_deg:.7f}, {packet.gps.longitude_deg:.7f}, {packet.gps.altitude_m:.2f}m")
            if packet.gps.vel_north_mps is not None:
                lines.append(f"    Vel: {packet.gps.vel_north_mps:.3f}N, {packet.gps.vel_east_mps:.3f}E, {packet.gps.vel_down_mps:.3f}D m/s")

        if packet.has_ins:
            lines.append("  INS:")
            if packet.ins.mode is not None:
                lines.append(f"    Mode: {packet.ins.mode.name}, GPS: {'Yes' if packet.ins.gps_fix else 'No'}")
            if packet.ins.vel_body_x is not None:
                lines.append(f"    VelBody: {packet.ins.vel_body_x:.3f}, {packet.ins.vel_body_y:.3f}, {packet.ins.vel_body_z:.3f} m/s")

        self.get_logger().info('\n'.join(lines))

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
        stats = self._serial.stats
        parser_stats = stats.get('parser_stats', {})

        self.get_logger().info(
            f"\n=== VectorNav Monitor Statistics ==="
            f"\n  Packet rate: {packet_rate:.1f} Hz"
            f"\n  Groups seen: {', '.join(sorted(self._groups_seen)) or 'None'}"
            f"\n  Bytes received: {stats.get('bytes_received', 0)}"
            f"\n  CRC errors: {parser_stats.get('crc_errors', 0)}"
            f"\n  Sync losses: {parser_stats.get('sync_losses', 0)}"
            f"\n  Connected: {stats.get('is_connected', False)}"
        )

        # Reset counters
        self._packet_count = 0
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
    node = VectorNavMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
