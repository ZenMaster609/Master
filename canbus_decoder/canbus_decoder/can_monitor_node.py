#!/usr/bin/env python3
"""
CAN Monitor Node - Displays raw CAN bus traffic for debugging.

Subscribes to /can/rx (from ros2_socketcan) and logs all CAN frames
with their ID, data bytes, and timestamp.
"""

import rclpy
from rclpy.node import Node
from can_msgs.msg import Frame
from collections import defaultdict
import time


class CanMonitorNode(Node):
    """
    Monitor and display raw CAN bus traffic.

    Subscribes to /can/rx and logs CAN frames. Useful for:
    - Verifying CAN bus connectivity
    - Identifying CAN IDs present on the bus
    - Debugging signal decoding

    Parameters:
        can_topic (str): Topic to subscribe to (default: /can/rx)
        filter_ids (list): List of CAN IDs to show (empty = show all)
        show_stats (bool): Show periodic statistics (default: True)
        stats_interval (float): Statistics interval in seconds (default: 5.0)
        verbose (bool): Show every frame (default: False, shows unique IDs only)
    """

    def __init__(self):
        super().__init__('can_monitor_node')

        # Declare parameters
        self.declare_parameter('can_topic', '/can/rx')
        self.declare_parameter('filter_ids', [])  # Empty = all IDs
        self.declare_parameter('show_stats', True)
        self.declare_parameter('stats_interval', 5.0)
        self.declare_parameter('verbose', False)

        # Get parameters
        self.can_topic = self.get_parameter('can_topic').value
        self.filter_ids = self.get_parameter('filter_ids').value
        self.show_stats = self.get_parameter('show_stats').value
        self.stats_interval = self.get_parameter('stats_interval').value
        self.verbose = self.get_parameter('verbose').value

        # Statistics tracking
        self._frame_counts = defaultdict(int)  # Count per CAN ID
        self._last_data = {}  # Last data seen per CAN ID
        self._total_frames = 0
        self._start_time = time.time()
        self._seen_ids = set()

        # Subscribe to CAN frames
        self.can_sub = self.create_subscription(
            Frame,
            self.can_topic,
            self.can_callback,
            10
        )

        # Statistics timer
        if self.show_stats:
            self.stats_timer = self.create_timer(
                self.stats_interval,
                self.print_stats
            )

        self.get_logger().info(f'CAN Monitor started')
        self.get_logger().info(f'  Subscribed to: {self.can_topic}')
        if self.filter_ids:
            filter_hex = [f'0x{id:03X}' for id in self.filter_ids]
            self.get_logger().info(f'  Filtering IDs: {filter_hex}')
        else:
            self.get_logger().info(f'  Showing all CAN IDs')
        self.get_logger().info(f'  Verbose mode: {self.verbose}')

    def can_callback(self, msg: Frame):
        """Process incoming CAN frame."""
        can_id = msg.id

        # Apply filter if configured
        if self.filter_ids and can_id not in self.filter_ids:
            return

        # Update statistics
        self._frame_counts[can_id] += 1
        self._total_frames += 1

        # Format data bytes as hex string
        data_hex = ' '.join(f'{b:02X}' for b in msg.data[:msg.dlc])

        # Check if this is a new ID we haven't seen
        is_new_id = can_id not in self._seen_ids
        if is_new_id:
            self._seen_ids.add(can_id)
            self.get_logger().info(
                f'NEW ID: 0x{can_id:03X} | DLC: {msg.dlc} | Data: [{data_hex}]'
            )

        # Check if data changed for this ID
        data_bytes = bytes(msg.data[:msg.dlc])
        data_changed = self._last_data.get(can_id) != data_bytes
        self._last_data[can_id] = data_bytes

        # Log frame if verbose or if data changed for existing ID
        if self.verbose or (data_changed and not is_new_id):
            ext_str = 'EXT' if msg.is_extended else 'STD'
            self.get_logger().debug(
                f'[{ext_str}] 0x{can_id:03X} | DLC: {msg.dlc} | Data: [{data_hex}]'
            )

    def print_stats(self):
        """Print periodic statistics about CAN traffic."""
        elapsed = time.time() - self._start_time

        if self._total_frames == 0:
            self.get_logger().warn('No CAN frames received yet!')
            return

        # Sort IDs for display
        sorted_ids = sorted(self._frame_counts.keys())

        self.get_logger().info('=' * 60)
        self.get_logger().info(f'CAN Statistics (elapsed: {elapsed:.1f}s)')
        self.get_logger().info(f'Total frames: {self._total_frames} | Unique IDs: {len(sorted_ids)}')
        self.get_logger().info('-' * 60)
        self.get_logger().info(f'{"CAN ID":<12} {"Count":<10} {"Rate (Hz)":<10} {"Last Data"}')
        self.get_logger().info('-' * 60)

        for can_id in sorted_ids:
            count = self._frame_counts[can_id]
            rate = count / elapsed if elapsed > 0 else 0
            last_data = self._last_data.get(can_id, b'')
            data_hex = ' '.join(f'{b:02X}' for b in last_data)

            self.get_logger().info(
                f'0x{can_id:03X}       {count:<10} {rate:<10.1f} [{data_hex}]'
            )

        self.get_logger().info('=' * 60)


def main(args=None):
    rclpy.init(args=args)

    node = CanMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Print final stats
        node.get_logger().info('Shutting down CAN Monitor')
        node.print_stats()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
