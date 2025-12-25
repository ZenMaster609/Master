#!/usr/bin/env python3
"""
PlotterNode - Real-time visualization of vehicle state.

Subscribes to /vehicle_plotter/state and displays real-time plots
using PyQtGraph. Highly configurable via parameters.
"""

import rclpy
from rclpy.node import Node

from vehicle_plotter_msgs.msg import VehicleState as VehicleStateMsg

from ..core.vehicle_state import VehicleState
from ..core.qos_profiles import PLOTTER_QOS
from ..plotting.plot_manager import PlotManager
from ..plotting.plot_config import get_default_plots, PlotLayoutConfig


class PlotterNode(Node):
    """
    Real-time plotting node for vehicle state visualization.

    Subscribes to /vehicle_plotter/state and displays configurable
    real-time plots using PyQtGraph.

    Parameters:
        backend (str): Plotting backend ('pyqtgraph' or 'dummy')
        update_rate_hz (float): Plot refresh rate (default: 30.0)
        window_title (str): Window title
        dark_mode (bool): Use dark theme (default: true)
        enable_gui (bool): Enable GUI (set false for headless)
    """

    def __init__(self):
        super().__init__('plotter_node')

        # Declare parameters
        self.declare_parameter('backend', 'pyqtgraph')
        self.declare_parameter('update_rate_hz', 30.0)
        self.declare_parameter('window_title', 'Vehicle Plotter')
        self.declare_parameter('dark_mode', True)
        self.declare_parameter('enable_gui', True)
        self.declare_parameter('state_topic', 'vehicle_plotter/state')

        # Get parameters
        backend = self.get_parameter('backend').value
        update_rate = self.get_parameter('update_rate_hz').value
        window_title = self.get_parameter('window_title').value
        dark_mode = self.get_parameter('dark_mode').value
        enable_gui = self.get_parameter('enable_gui').value
        state_topic = self.get_parameter('state_topic').value

        self.get_logger().info(f'PlotterNode starting...')
        self.get_logger().info(f'  Backend: {backend}')
        self.get_logger().info(f'  Update rate: {update_rate} Hz')
        self.get_logger().info(f'  GUI enabled: {enable_gui}')

        # Create plot configuration
        layout_config = get_default_plots()
        layout_config.window_title = window_title
        layout_config.dark_mode = dark_mode
        layout_config.update_rate_hz = update_rate

        # Initialize plot manager
        self.plot_manager = PlotManager(
            layout_config=layout_config,
            backend=backend,
            enable_gui=enable_gui,
        )

        # Check if GUI actually initialized
        self._gui_available = self.plot_manager._backend.is_available()
        if not self._gui_available:
            self.get_logger().warn('GUI not available (no display), running in headless mode')

        # Subscribe to vehicle state
        self.state_sub = self.create_subscription(
            VehicleStateMsg,
            state_topic,
            self.state_callback,
            PLOTTER_QOS,
        )

        # Plot refresh timer
        self.refresh_timer = self.create_timer(
            1.0 / update_rate,
            self.refresh_callback,
        )

        # Statistics
        self._state_count = 0
        self._window_open = True

        self.get_logger().info(f'PlotterNode started, subscribed to {state_topic}')

    def state_callback(self, msg: VehicleStateMsg) -> None:
        """Handle incoming vehicle state message."""
        state = VehicleState.from_msg(msg)
        self.plot_manager.push_state(state)
        self._state_count += 1

    def refresh_callback(self) -> None:
        """Refresh plots at the configured rate."""
        if not self._window_open:
            return

        self._window_open = self.plot_manager.refresh()

        if not self._window_open:
            self.get_logger().info('Plot window closed, shutting down')
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown."""
        self.plot_manager.close()
        rclpy.shutdown()

    def export_plots(self, path: str) -> None:
        """Export current plots to image file."""
        self.plot_manager.export(path)
        self.get_logger().info(f'Plots exported to {path}')


def main(args=None):
    rclpy.init(args=args)

    node = PlotterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'PlotterNode shutting down, received {node._state_count} states')
        node.plot_manager.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
