#!/usr/bin/env python3
"""
DataCollectorNode - Aggregates sensor data and computes unified vehicle state.

This node is the central data aggregation point for the vehicle_plotter package.
It subscribes to all sensor topics via a sensor adapter, synchronizes the data,
computes the unified VehicleState, and publishes it for other nodes to consume.
"""

import rclpy
from rclpy.node import Node

from vehicle_plotter_msgs.msg import VehicleState as VehicleStateMsg

from ..core.vehicle_state import VehicleState
from ..core.time_sync import TimeSynchronizer
from ..core.qos_profiles import PLOTTER_QOS
from ..adapters.gazebo_adapter import GazeboAdapter
from ..adapters.can_adapter import CANAdapter
from ..adapters.vectornav_adapter import VectorNavAdapter


class DataCollectorNode(Node):
    """
    Main data collection node for vehicle_plotter.

    Responsibilities:
    1. Subscribe to all sensor topics via sensor adapter
    2. Synchronize multi-rate sensor data
    3. Compute unified VehicleState
    4. Publish synchronized state for plotting/logging

    Parameters:
        adapter (str): Sensor adapter type ('gazebo', 'can', or 'vectornav')
        output_rate_hz (float): Output rate in Hz (default: 50.0)
        gps_origin_lat (float): GPS origin latitude (0.0 = auto)
        gps_origin_lon (float): GPS origin longitude (0.0 = auto)
        enable_virtual_sensors (bool): Enable virtual sensor subscriptions (gazebo only)
    """

    def __init__(self):
        super().__init__('data_collector_node')

        # Declare parameters
        self.declare_parameter('adapter', 'gazebo')
        self.declare_parameter('output_rate_hz', 50.0)
        self.declare_parameter('gps_origin_lat', 0.0)
        self.declare_parameter('gps_origin_lon', 0.0)
        self.declare_parameter('sync_buffer_sec', 0.2)
        self.declare_parameter('enable_virtual_sensors', True)

        # Get parameters
        adapter_type = self.get_parameter('adapter').value
        output_rate = self.get_parameter('output_rate_hz').value
        gps_origin_lat = self.get_parameter('gps_origin_lat').value
        gps_origin_lon = self.get_parameter('gps_origin_lon').value
        sync_buffer_sec = self.get_parameter('sync_buffer_sec').value
        enable_virtual_sensors = self.get_parameter('enable_virtual_sensors').value

        self.get_logger().info(f'DataCollectorNode starting...')
        self.get_logger().info(f'  Adapter: {adapter_type}')
        self.get_logger().info(f'  Output rate: {output_rate} Hz')
        if adapter_type == 'gazebo':
            self.get_logger().info(
                f'  Virtual sensors enabled: {str(enable_virtual_sensors).lower()}'
            )

        # Initialize time synchronizer
        self.synchronizer = TimeSynchronizer(
            output_rate_hz=output_rate,
            buffer_duration_sec=sync_buffer_sec,
        )

        # Initialize sensor adapter
        auto_set_gps_origin = (gps_origin_lat == 0.0 and gps_origin_lon == 0.0)

        if not auto_set_gps_origin:
            from ..utils.transforms import set_gps_origin
            set_gps_origin(gps_origin_lat, gps_origin_lon)
            self.get_logger().info(f'  GPS origin: ({gps_origin_lat}, {gps_origin_lon})')

        if adapter_type == 'gazebo':
            self.adapter = GazeboAdapter(
                node=self,
                synchronizer=self.synchronizer,
                auto_set_gps_origin=auto_set_gps_origin,
                enable_virtual_sensors=enable_virtual_sensors,
            )
        elif adapter_type == 'can':
            # CAN bus adapter for real hardware wheel encoders
            self.adapter = CANAdapter(
                node=self,
                synchronizer=self.synchronizer,
            )
        elif adapter_type == 'vectornav':
            # VectorNav VN-200 adapter
            self.adapter = VectorNavAdapter(
                node=self,
                synchronizer=self.synchronizer,
                auto_set_gps_origin=auto_set_gps_origin,
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        # State publisher
        self.state_pub = self.create_publisher(
            VehicleStateMsg,
            'vehicle_plotter/state',
            PLOTTER_QOS,
        )

        # State computation timer
        self.state_timer = self.create_timer(
            1.0 / output_rate,
            self.compute_and_publish_state,
        )

        # Internal state
        self._last_state: VehicleState = VehicleState()
        self._distance_accumulator: float = 0.0
        self._state_count: int = 0

        # Status timer (log every 5 seconds)
        self.status_timer = self.create_timer(5.0, self.log_status)

        self.get_logger().info('DataCollectorNode started')

    def compute_and_publish_state(self):
        """
        Called at output_rate_hz.
        Synchronizes sensor data and publishes unified state.
        """
        # Debug: log first callback
        if not hasattr(self, '_first_callback_logged'):
            self._first_callback_logged = True
            self.get_logger().info('compute_and_publish_state callback started')

        # Use the latest sensor timestamp instead of clock time
        # This works correctly with both real time and simulation time
        target_time = self.synchronizer.get_latest_time()
        if target_time is None:
            # No sensor data yet - log occasionally for debugging
            if self._state_count == 0 and not hasattr(self, '_no_data_logged'):
                self._no_data_logged = True
                self.get_logger().warn('No sensor data available yet (get_latest_time returned None)')
            return

        # Debug: log first successful time
        if not hasattr(self, '_first_time_logged'):
            self._first_time_logged = True
            self.get_logger().info(f'First target_time: {target_time:.2f}s')

        # Get synchronized sensor readings
        synced = self.synchronizer.get_synchronized(target_time)
        if synced is None:
            # Required sensors not available - log for debugging
            if not hasattr(self, '_sync_fail_logged'):
                self._sync_fail_logged = True
                self.get_logger().warn(f'Synchronization failed at time {target_time:.2f}s')
            return

        # Debug: log first sync success
        if not hasattr(self, '_first_sync_logged'):
            self._first_sync_logged = True
            self.get_logger().info(f'First successful sync, keys: {list(synced.keys())}')

        # Compute vehicle state from synchronized data
        state = self.adapter.compute_state(synced, self._last_state)

        # Update distance traveled
        if self._last_state.timestamp > 0:
            dt = state.timestamp - self._last_state.timestamp
            if dt > 0 and dt < 0.1:  # Sanity check
                ds = state.speed * dt
                self._distance_accumulator += ds

        state.distance_traveled = self._distance_accumulator

        # Publish state
        msg = state.to_msg()
        self.state_pub.publish(msg)

        # Store for next iteration
        self._last_state = state
        self._state_count += 1

        # Log first successful publish
        if self._state_count == 1:
            self.get_logger().info(f'First state published! x={state.x:.2f}, y={state.y:.2f}, speed={state.speed:.2f}')

    def log_status(self):
        """Log periodic status update."""
        buffer_status = self.synchronizer.get_buffer_status()

        status_parts = []
        for name, info in buffer_status.items():
            status_parts.append(f"{name}:{info['samples_received']}")

        # Get latest time for debugging
        latest_time = self.synchronizer.get_latest_time()

        latest_str = f"{latest_time:.2f}s" if latest_time else "None"
        self.get_logger().info(
            f"States published: {self._state_count}, "
            f"Distance: {self._distance_accumulator:.2f}m, "
            f"Samples: [{', '.join(status_parts)}], "
            f"Latest: {latest_str}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = DataCollectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('DataCollectorNode shutting down')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
