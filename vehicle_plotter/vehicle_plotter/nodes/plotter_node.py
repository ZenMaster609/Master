#!/usr/bin/env python3
"""
State aggregation and plotting node for the simulation sensor pipeline.

Subscribes to measured `/sim/*` topics through the Gazebo adapter, builds
`VehicleState` samples, publishes `/vehicle_plotter/state`, and maintains the
in-memory plot buffers used for the live dashboard and static dashboard export.
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from vehicle_plotter_msgs.msg import VehicleState as VehicleStateMsg
from vehicle_plotter_msgs.msg import RunSession as RunSessionMsg

from ..adapters.gazebo_adapter import GazeboAdapter
from ..core.qos_profiles import PLOTTER_QOS, RELIABLE_SENSOR_QOS
from ..core.run_session import RunSession
from ..core.time_sync import TimeSynchronizer
from ..core.vehicle_state import VehicleState
from ..plotting.plot_config import get_all_plots
from ..plotting.plot_manager import PlotManager


class PlotterNode(Node):
    """Aggregate measured sim topics into VehicleState and plot buffers."""

    def __init__(self) -> None:
        super().__init__('plotter_node')

        self.declare_parameter('adapter', 'gazebo')
        self.declare_parameter('state_topic', '/vehicle_plotter/state')
        self.declare_parameter('output_rate_hz', 50.0)
        self.declare_parameter('backend', 'pyqtgraph')
        self.declare_parameter('enable_gui', True)
        self.declare_parameter('auto_export_dashboard', True)
        self.declare_parameter('dashboard_filename', 'virtual_sensors.png')

        adapter_name = str(self.get_parameter('adapter').value).strip().lower() or 'gazebo'
        self._state_topic = str(self.get_parameter('state_topic').value).strip() or '/vehicle_plotter/state'
        self._output_rate_hz = max(1.0, float(self.get_parameter('output_rate_hz').value))
        backend = str(self.get_parameter('backend').value).strip() or 'pyqtgraph'
        enable_gui = bool(self.get_parameter('enable_gui').value)
        self._auto_export_dashboard = bool(self.get_parameter('auto_export_dashboard').value)
        self._dashboard_filename = str(
            self.get_parameter('dashboard_filename').value
        ).strip() or 'virtual_sensors.png'

        if adapter_name != 'gazebo':
            self.get_logger().warn(
                f"Unsupported adapter '{adapter_name}', falling back to 'gazebo'"
            )
            adapter_name = 'gazebo'
            self.set_parameters([
                Parameter('adapter', value='gazebo'),
            ])

        self._run_session: Optional[RunSession] = None
        self._prev_state: Optional[VehicleState] = None
        self._last_publish_time: Optional[float] = None
        self._published_count = 0
        self._shutdown_called = False

        self._state_pub = self.create_publisher(
            VehicleStateMsg,
            self._state_topic,
            PLOTTER_QOS,
        )
        self._session_sub = self.create_subscription(
            RunSessionMsg,
            '/run_session',
            self._session_callback,
            RELIABLE_SENSOR_QOS,
        )

        self._sync = TimeSynchronizer(
            output_rate_hz=self._output_rate_hz,
            buffer_duration_sec=max(0.5, 4.0 / self._output_rate_hz),
        )
        self._adapter = GazeboAdapter(self, self._sync)
        self._plot_manager = PlotManager(
            layout_config=get_all_plots(),
            backend=backend,
            enable_gui=enable_gui,
        )

        self._publish_timer = self.create_timer(
            1.0 / self._output_rate_hz,
            self._publish_state_callback,
        )
        refresh_rate_hz = max(1.0, float(self._plot_manager.layout.update_rate_hz))
        self._refresh_timer = self.create_timer(
            1.0 / refresh_rate_hz,
            self._refresh_plots_callback,
        )
        self._status_timer = self.create_timer(10.0, self._status_callback)

        self.get_logger().info(
            f"PlotterNode started: adapter={adapter_name} state_topic={self._state_topic} "
            f"output_rate={self._output_rate_hz:.1f}Hz gui={enable_gui}"
        )

        rclpy.get_default_context().on_shutdown(self.shutdown)

    def _session_callback(self, msg: RunSessionMsg) -> None:
        self._run_session = RunSession.from_msg(msg)

    def _publish_state_callback(self) -> None:
        target_time = self._sync.get_latest_time()
        if target_time is None:
            return
        if self._last_publish_time is not None and target_time <= self._last_publish_time:
            return

        synced = self._sync.get_synchronized(target_time)
        if synced is None:
            return

        state = self._adapter.compute_state(synced, self._prev_state)
        self._finalize_state(state)
        self._state_pub.publish(state.to_msg())
        self._plot_manager.push_state(state)

        self._prev_state = state
        self._last_publish_time = state.timestamp
        self._published_count += 1

    def _refresh_plots_callback(self) -> None:
        self._plot_manager.refresh()

    def _finalize_state(self, state: VehicleState) -> None:
        if self._prev_state is None:
            state.distance_traveled = 0.0
        else:
            dx = float(state.x) - float(self._prev_state.x)
            dy = float(state.y) - float(self._prev_state.y)
            if math.isfinite(dx) and math.isfinite(dy):
                state.distance_traveled = self._prev_state.distance_traveled + math.hypot(dx, dy)
            else:
                state.distance_traveled = self._prev_state.distance_traveled

        # In the sim pipeline, the fused position channel is the current measured pose.
        state.ins_x = float(state.x)
        state.ins_y = float(state.y)

    def _status_callback(self) -> None:
        self.get_logger().info(
            f"Published {self._published_count} vehicle states on {self._state_topic}"
        )

    def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True

        if self._auto_export_dashboard and self._run_session is not None:
            output_path = self._run_session.plots_path / self._dashboard_filename
            try:
                self._plot_manager.export_static_dashboard(str(output_path))
                self.get_logger().info(f"Saved static dashboard: {output_path}")
            except Exception as exc:
                self.get_logger().warn(f"Failed to export static dashboard: {exc}")

        try:
            self._plot_manager.close()
        except Exception:
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlotterNode()

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
