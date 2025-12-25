#!/usr/bin/env python3
"""
LoggerNode - Logs vehicle state to files.

Subscribes to /vehicle_plotter/state and writes data to
Parquet or CSV files in ~/.ros/vehicle_logs/.
"""

import rclpy
from rclpy.node import Node
from pathlib import Path

from vehicle_plotter_msgs.msg import VehicleState as VehicleStateMsg

from ..core.vehicle_state import VehicleState
from ..core.qos_profiles import PLOTTER_QOS
from ..logging.log_writer import LogWriter
from ..logging.log_config import LogConfig


class LoggerNode(Node):
    """
    Data logging node for vehicle state.

    Subscribes to /vehicle_plotter/state and writes synchronized
    state data to disk in configurable formats (Parquet, CSV).

    Parameters:
        format (str): Output format ('parquet' or 'csv')
        compression (str): Compression for parquet ('snappy', 'gzip', 'zstd', 'none')
        base_path (str): Base directory for logs (default: ~/.ros/vehicle_logs)
        session_name (str): Custom session name (auto-generated if empty)
        flush_interval_sec (float): Seconds between disk flushes
        buffer_size (int): Max records to buffer before flush
    """

    def __init__(self):
        super().__init__('logger_node')

        # Declare parameters
        self.declare_parameter('format', 'parquet')
        self.declare_parameter('compression', 'snappy')
        self.declare_parameter('base_path', str(Path.home() / '.ros' / 'vehicle_logs'))
        self.declare_parameter('session_name', '')
        self.declare_parameter('flush_interval_sec', 5.0)
        self.declare_parameter('buffer_size', 1000)
        self.declare_parameter('state_topic', 'vehicle_plotter/state')
        self.declare_parameter('enable_logging', True)

        # Get parameters
        log_format = self.get_parameter('format').value
        compression = self.get_parameter('compression').value
        base_path_str = self.get_parameter('base_path').value
        # Expand ~ to home directory
        base_path = Path(base_path_str).expanduser()
        session_name = self.get_parameter('session_name').value
        flush_interval = self.get_parameter('flush_interval_sec').value
        buffer_size = self.get_parameter('buffer_size').value
        state_topic = self.get_parameter('state_topic').value
        enable_logging = self.get_parameter('enable_logging').value

        self.get_logger().info(f'LoggerNode starting...')
        self.get_logger().info(f'  Format: {log_format}')
        self.get_logger().info(f'  Base path: {base_path}')
        self.get_logger().info(f'  Logging enabled: {enable_logging}')

        self._enable_logging = enable_logging

        if enable_logging:
            # Create log configuration
            config = LogConfig(
                base_path=base_path,
                format=log_format,
                compression=compression,
                session_name=session_name if session_name else None,
                flush_interval_sec=flush_interval,
                buffer_size=buffer_size,
            )

            # Initialize log writer
            self.log_writer = LogWriter(config)
            self.get_logger().info(f'  Session: {self.log_writer.session_path}')
        else:
            self.log_writer = None

        # Subscribe to vehicle state
        self.state_sub = self.create_subscription(
            VehicleStateMsg,
            state_topic,
            self.state_callback,
            PLOTTER_QOS,
        )

        # Flush timer
        if enable_logging:
            self.flush_timer = self.create_timer(flush_interval, self.flush_callback)

        # Status timer
        self.status_timer = self.create_timer(10.0, self.status_callback)

        self.get_logger().info(f'LoggerNode started, subscribed to {state_topic}')

    def state_callback(self, msg: VehicleStateMsg) -> None:
        """Handle incoming vehicle state message."""
        if not self._enable_logging or self.log_writer is None:
            return

        state = VehicleState.from_msg(msg)
        self.log_writer.write(state)

    def flush_callback(self) -> None:
        """Periodic flush to disk."""
        if self.log_writer is not None:
            self.log_writer.flush()

    def status_callback(self) -> None:
        """Log periodic status update."""
        if self.log_writer is None:
            return

        status = self.log_writer.get_status()
        self.get_logger().info(
            f"Logged {status['total_records']} records, "
            f"{status['bytes_written'] / 1024:.1f} KB written"
        )

    def shutdown(self) -> None:
        """Clean shutdown with final flush."""
        if self.log_writer is not None:
            self.log_writer.close()
            self.get_logger().info(
                f"Logger closed, {self.log_writer.total_records} records written to "
                f"{self.log_writer.session_path}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = LoggerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
