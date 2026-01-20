#!/usr/bin/env python3
"""
PlotterNode - Real-time visualization of vehicle state.

Subscribes to /vehicle_plotter/state and displays real-time plots
using PyQtGraph. Supports auto-save of plots on shutdown.
"""

import rclpy
from rclpy.node import Node
from pathlib import Path
from typing import Optional
from datetime import datetime

from vehicle_plotter_msgs.msg import VehicleState as VehicleStateMsg
from vehicle_plotter_msgs.msg import RunSession as RunSessionMsg

from ..core.vehicle_state import VehicleState
from ..core.run_session import RunSession
from ..core.qos_profiles import PLOTTER_QOS, RELIABLE_SENSOR_QOS
from ..plotting.plot_manager import PlotManager
from ..plotting.plot_config import get_default_plots, get_virtual_sensor_plots, PlotLayoutConfig


class PlotterNode(Node):
    """
    Real-time plotting node for vehicle state visualization.

    Subscribes to /vehicle_plotter/state and displays configurable
    real-time plots using PyQtGraph.

    Supports:
    - RunSession integration for coordinated data storage
    - Auto-save plots on shutdown (PNG images)
    - Auto-save plot data on shutdown (CSV files)

    Parameters:
        backend (str): Plotting backend ('pyqtgraph' or 'dummy')
        update_rate_hz (float): Plot refresh rate (default: 30.0)
        window_title (str): Window title
        dark_mode (bool): Use dark theme (default: true)
        enable_gui (bool): Enable GUI (set false for headless)
        save_plots_on_exit (bool): Save plot images on shutdown
        save_plot_data_on_exit (bool): Save plot data CSV on shutdown
        wait_for_session (bool): Wait for /run_session before starting
    """

    def __init__(self):
        super().__init__('plotter_node')

        # Declare parameters
        self.declare_parameter('backend', 'pyqtgraph')
        self.declare_parameter('update_rate_hz', 30.0)
        self.declare_parameter('window_title', 'Vehicle Plotter')
        self.declare_parameter('plot_layout', 'default')
        self.declare_parameter('dark_mode', True)
        self.declare_parameter('enable_gui', True)
        self.declare_parameter('state_topic', 'vehicle_plotter/state')
        self.declare_parameter('save_plots_on_exit', True)
        self.declare_parameter('save_plot_data_on_exit', True)
        self.declare_parameter('wait_for_session', True)
        self.declare_parameter('session_timeout_sec', 5.0)
        self.declare_parameter('base_path', '')

        # Get parameters
        backend = self.get_parameter('backend').value
        update_rate = self.get_parameter('update_rate_hz').value
        window_title = self.get_parameter('window_title').value
        plot_layout = self.get_parameter('plot_layout').value
        dark_mode = self.get_parameter('dark_mode').value
        enable_gui = self.get_parameter('enable_gui').value
        state_topic = self.get_parameter('state_topic').value
        self._save_plots = self.get_parameter('save_plots_on_exit').value
        self._save_plot_data = self.get_parameter('save_plot_data_on_exit').value
        self._wait_for_session = self.get_parameter('wait_for_session').value
        self._session_timeout = self.get_parameter('session_timeout_sec').value
        base_path_str = self.get_parameter('base_path').value

        # Parse base path
        if base_path_str:
            self._base_path = Path(base_path_str).expanduser()
        else:
            self._base_path = None

        self.get_logger().info(f'PlotterNode starting...')
        self.get_logger().info(f'  Backend: {backend}')
        self.get_logger().info(f'  Update rate: {update_rate} Hz')
        self.get_logger().info(f'  GUI enabled: {enable_gui}')
        self.get_logger().info(f'  Save plots on exit: {self._save_plots}')
        self.get_logger().info(f'  Save plot data on exit: {self._save_plot_data}')

        # Create plot configuration
        if plot_layout == 'virtual_sensors':
            layout_config = get_virtual_sensor_plots()
        else:
            if plot_layout != 'default':
                self.get_logger().warn(
                    f"Unknown plot_layout '{plot_layout}', using default"
                )
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

        # RunSession handling
        self._run_session: Optional[RunSession] = None
        self._session_initialized = False

        if self._save_plots or self._save_plot_data:
            # Subscribe to run session
            self.session_sub = self.create_subscription(
                RunSessionMsg,
                '/run_session',
                self.session_callback,
                RELIABLE_SENSOR_QOS,
            )

            if self._wait_for_session:
                self.session_timer = self.create_timer(
                    self._session_timeout,
                    self.session_timeout_callback
                )
                self.get_logger().info(
                    f'  Waiting up to {self._session_timeout}s for /run_session...'
                )
            else:
                self._initialize_session(None)
        else:
            self._session_initialized = True

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

    def session_callback(self, msg: RunSessionMsg) -> None:
        """Handle incoming run session message."""
        if self._session_initialized:
            return

        self.get_logger().info(
            f'Received run_session: {msg.run_id} from {msg.originator_hostname}'
        )

        if hasattr(self, 'session_timer') and self.session_timer:
            self.session_timer.cancel()
            self.session_timer = None

        self._initialize_session(msg)

    def session_timeout_callback(self) -> None:
        """Called when session timeout expires."""
        if self._session_initialized:
            return

        self.get_logger().info('No /run_session received, creating own session')

        if self.session_timer:
            self.session_timer.cancel()
            self.session_timer = None

        self._initialize_session(None)

    def _initialize_session(self, msg: Optional[RunSessionMsg]) -> None:
        """Initialize the session for plot saving."""
        if self._session_initialized:
            return

        if msg:
            self._run_session = RunSession.from_msg(msg, self._base_path)
        else:
            self._run_session = RunSession.create_new(self._base_path)

        self._run_session.ensure_directories()

        self.get_logger().info(f'  Run ID: {self._run_session.run_id}')
        self.get_logger().info(f'  Plots path: {self._run_session.plots_path}')
        self.get_logger().info(f'  Plot data path: {self._run_session.plot_data_path}')

        self._session_initialized = True

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
        """Clean shutdown with optional plot/data export."""
        self._save_on_exit()
        self.plot_manager.close()
        rclpy.shutdown()

    def _save_on_exit(self) -> None:
        """Save plots and data if configured."""
        if not self._session_initialized or not self._run_session:
            # Create a fallback session if none exists
            if self._save_plots or self._save_plot_data:
                self._run_session = RunSession.create_new(self._base_path)
                self._run_session.ensure_directories()

        if not self._run_session:
            return

        # Save plot image
        if self._save_plots and self._gui_available:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self._run_session.plots_path / f"plots_{timestamp}.png"
            try:
                self.plot_manager.export(str(plot_path))
                self.get_logger().info(f'Plots saved to: {plot_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to save plots: {e}')

        # Save plot data
        if self._save_plot_data:
            try:
                exported = self.plot_manager.export_data(self._run_session.plot_data_path)
                self.get_logger().info(f'Plot data saved: {len(exported)} files to {self._run_session.plot_data_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to save plot data: {e}')

    def export_plots(self, path: str) -> None:
        """Export current plots to image file."""
        self.plot_manager.export(path)
        self.get_logger().info(f'Plots exported to {path}')

    @property
    def run_session(self) -> Optional[RunSession]:
        """Get the current run session."""
        return self._run_session


def main(args=None):
    rclpy.init(args=args)

    node = PlotterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'PlotterNode shutting down, received {node._state_count} states')
        node._save_on_exit()
        node.plot_manager.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
