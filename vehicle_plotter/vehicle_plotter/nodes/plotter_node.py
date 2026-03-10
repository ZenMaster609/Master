#!/usr/bin/env python3
"""
PlotterNode - Real-time visualization of vehicle state.

Can either aggregate live simulation sensor topics directly or subscribe to
/vehicle_plotter/state for replay/offline plotting. Supports auto-save of
plots on shutdown.
"""

import rclpy
import yaml
import signal
from rclpy.node import Node
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import replace
import time

from vehicle_plotter_msgs.msg import VehicleState as VehicleStateMsg
from vehicle_plotter_msgs.msg import RunSession as RunSessionMsg
from ament_index_python.packages import get_package_share_directory

from ..core.vehicle_state import VehicleState
from ..core.run_session import RunSession
from ..core.qos_profiles import PLOTTER_QOS, RELIABLE_SENSOR_QOS
from ..core.time_sync import TimeSynchronizer
from ..adapters.gazebo_adapter import GazeboAdapter
from ..plotting.plot_manager import PlotManager
from ..plotting.plot_config import (
    get_default_plots,
    get_virtual_sensor_plots,
    get_all_plots,
    PlotLayoutConfig,
)

_SIGNAL_OVERRIDE_BY_VARIABLE = {
    'x': 'realistic_position',
    'y': 'realistic_position',
    'vx': 'realistic_velocity',
    'vy': 'realistic_velocity',
    'speed': 'realistic_velocity',
    'yaw': 'realistic_yaw',
    'raw_x': 'raw_position',
    'raw_y': 'raw_position',
    'raw_vx': 'raw_velocity',
    'raw_vy': 'raw_velocity',
    'raw_speed': 'raw_velocity',
    'raw_yaw': 'raw_yaw',
    'imu_vx': 'imu_velocity',
    'imu_vy': 'imu_velocity',
    'imu_yaw': 'imu_yaw',
    'gps_local_x': 'gnss_position',
    'gps_local_y': 'gnss_position',
}

_VARIABLE_TO_TOPICS = {
    'x': ['/sim/odom'],
    'y': ['/sim/odom'],
    'vx': ['/sim/odom'],
    'vy': ['/sim/odom'],
    'speed': ['/sim/odom'],
    'yaw': ['/sim/odom'],
    'raw_x': ['/sim/raw/odom'],
    'raw_y': ['/sim/raw/odom'],
    'raw_vx': ['/sim/raw/odom'],
    'raw_vy': ['/sim/raw/odom'],
    'raw_speed': ['/sim/raw/odom'],
    'raw_yaw': ['/sim/raw/odom'],
    'encoder_velocities': ['/sim/wheel_encoder/rpm'],
    'encoder_speeds_mm_s': ['/sim/wheel_encoder/speed_mm_s'],
    'encoder_angle_accum': ['/sim/wheel_encoder/angle_accum'],
    'suspension': ['/sim/suspension'],
    'steering_angle': ['/sim/steering_angle'],
    'water_pressure': ['/sim/cooling/water_pressure'],
    'water_flow': ['/sim/cooling/water_flow'],
    'water_temp_in': ['/sim/cooling/water_temp_in'],
    'water_temp_out': ['/sim/cooling/water_temp_out'],
    'water_temp_radiator': ['/sim/cooling/water_temp_radiator'],
    'brake_temp_fr': ['/sim/brakes/temp_fr'],
    'brake_temp_rl': ['/sim/brakes/temp_rl'],
    'pitot_dynamic_pressure': ['/sim/pitot/dynamic_pressure'],
    'imu_vx': ['/sim/imu'],
    'imu_vy': ['/sim/imu'],
    'imu_yaw': ['/sim/imu'],
    'gps_local_x': ['/sim/navsat'],
    'gps_local_y': ['/sim/navsat'],
}


class PlotterNode(Node):
    """
    Real-time plotting node for vehicle state visualization.

    Displays configurable real-time plots using PyQtGraph. In live simulation
    it can aggregate measured sensor topics directly and publish
    `/vehicle_plotter/state`; in replay/offline mode it subscribes to an
    existing state topic.

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
        self.declare_parameter('plot_layout', 'all')
        self.declare_parameter('dark_mode', True)
        self.declare_parameter('enable_gui', True)
        self.declare_parameter('direct_from_sensors', False)
        self.declare_parameter('state_topic', 'vehicle_plotter/state')
        self.declare_parameter('state_output_rate_hz', 50.0)
        self.declare_parameter('save_plots_on_exit', False)
        self.declare_parameter('save_plot_data_on_exit', True)
        self.declare_parameter('close_plots_on_shutdown', True)
        self.declare_parameter('wait_for_session', True)
        self.declare_parameter('session_timeout_sec', 5.0)
        self.declare_parameter('base_path', '')
        self.declare_parameter('sensor_config_path', '')

        # Get parameters
        backend = self.get_parameter('backend').value
        update_rate = self.get_parameter('update_rate_hz').value
        window_title = self.get_parameter('window_title').value
        plot_layout = self.get_parameter('plot_layout').value
        dark_mode = self.get_parameter('dark_mode').value
        enable_gui = self.get_parameter('enable_gui').value
        direct_from_sensors = bool(self.get_parameter('direct_from_sensors').value)
        state_topic = self.get_parameter('state_topic').value
        state_output_rate_hz = float(self.get_parameter('state_output_rate_hz').value)
        self._save_plots = self.get_parameter('save_plots_on_exit').value
        self._save_plot_data = self.get_parameter('save_plot_data_on_exit').value
        self._close_plots_on_shutdown = self.get_parameter('close_plots_on_shutdown').value
        self._wait_for_session = self.get_parameter('wait_for_session').value
        self._session_timeout = self.get_parameter('session_timeout_sec').value
        base_path_str = self.get_parameter('base_path').value
        sensor_config_path = self.get_parameter('sensor_config_path').value

        # Parse base path
        if base_path_str:
            self._base_path = Path(base_path_str).expanduser()
        else:
            self._base_path = None

        self.get_logger().info(f'PlotterNode starting...')
        self.get_logger().info(f'  Backend: {backend}')
        self.get_logger().info(f'  Update rate: {update_rate} Hz')
        self.get_logger().info(f'  GUI enabled: {enable_gui}')
        self.get_logger().info(f'  Direct from sensors: {direct_from_sensors}')
        self.get_logger().info(f'  Save plots on exit: {self._save_plots}')
        self.get_logger().info(f'  Save plot data on exit: {self._save_plot_data}')
        self.get_logger().info(f'  Close plots on shutdown: {self._close_plots_on_shutdown}')

        # Create plot configuration
        if plot_layout == 'virtual_sensors':
            layout_config = get_virtual_sensor_plots()
        elif plot_layout == 'default':
            layout_config = get_default_plots()
        elif plot_layout == 'all':
            layout_config = get_all_plots()
        else:
            self.get_logger().warn(
                f"Unknown plot_layout '{plot_layout}', using all"
            )
            layout_config = get_all_plots()
        layout_config.window_title = window_title
        layout_config.dark_mode = dark_mode
        layout_config.update_rate_hz = update_rate
        layout_config = self._filter_layout_by_sensor_config(layout_config, sensor_config_path)

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

        self._direct_from_sensors = direct_from_sensors
        self._state_pub = None
        self._state_sub = None
        self._last_state = VehicleState()
        self._distance_accumulator = 0.0

        if self._direct_from_sensors:
            self.synchronizer = TimeSynchronizer(
                output_rate_hz=state_output_rate_hz,
                buffer_duration_sec=0.2,
            )
            self.adapter = GazeboAdapter(
                node=self,
                synchronizer=self.synchronizer,
                auto_set_gps_origin=True,
            )
            self._state_pub = self.create_publisher(
                VehicleStateMsg,
                state_topic,
                PLOTTER_QOS,
            )
            self.state_timer = self.create_timer(
                1.0 / max(1.0, state_output_rate_hz),
                self.compute_and_publish_state,
            )
        else:
            self._state_sub = self.create_subscription(
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
        self._shutdown_from_window_close = False
        self._artifacts_saved = False

        if self._direct_from_sensors:
            self.get_logger().info(
                f'PlotterNode started, aggregating live sensors and publishing {state_topic}'
            )
        else:
            self.get_logger().info(f'PlotterNode started, subscribed to {state_topic}')

    def _filter_layout_by_sensor_config(
        self,
        layout: PlotLayoutConfig,
        sensor_config_path: str,
    ) -> PlotLayoutConfig:
        config_path = sensor_config_path or self._default_sensor_config_path()
        if not config_path:
            return layout
        config = self._load_sensor_config(config_path)
        if not config:
            return layout

        enabled_by_name, enabled_by_topic = self._build_signal_index(config)

        def is_enabled_for_variable(variable: str) -> bool:
            base = variable.split('[', 1)[0]
            override = _SIGNAL_OVERRIDE_BY_VARIABLE.get(base)
            if override and override in enabled_by_name:
                return bool(enabled_by_name[override])
            topics = _VARIABLE_TO_TOPICS.get(base)
            if not topics:
                return True
            return any(enabled_by_topic.get(t, True) for t in topics)

        filtered_plots = []
        for plot in layout.plots:
            if plot.x_axis and plot.x_axis.variable:
                if not is_enabled_for_variable(plot.x_axis.variable):
                    continue

            filtered_series = [
                series for series in plot.series
                if is_enabled_for_variable(series.variable)
            ]
            if not filtered_series:
                continue
            filtered_plots.append(replace(plot, series=filtered_series))

        if not filtered_plots:
            return layout

        return replace(layout, plots=filtered_plots)

    def _default_sensor_config_path(self) -> str:
        try:
            sim_car_share = get_package_share_directory('sim_car')
            return str(Path(sim_car_share) / 'config' / 'sensor_config.yaml')
        except Exception:
            return ''

    def _load_sensor_config(self, path: str) -> Dict:
        try:
            with open(path, 'r') as config_file:
                return yaml.safe_load(config_file) or {}
        except (OSError, yaml.YAMLError):
            return {}

    def _build_signal_index(self, config: Dict) -> tuple:
        signals = config.get('signals', {})
        if not isinstance(signals, dict):
            return {}, {}

        enabled_by_name: Dict[str, bool] = {}
        enabled_by_topic: Dict[str, bool] = {}

        for name, raw in signals.items():
            if not isinstance(raw, dict):
                continue
            enabled = bool(raw.get('enabled', True))
            enabled_by_name[name] = enabled

            if raw.get('plot_only', False):
                continue

            for key in ('input_topic', 'output_topic'):
                topic = raw.get(key)
                if not topic:
                    continue
                if topic in enabled_by_topic:
                    enabled_by_topic[topic] = enabled_by_topic[topic] or enabled
                else:
                    enabled_by_topic[topic] = enabled

        return enabled_by_name, enabled_by_topic

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
        self.get_logger().info(f'  Plot data path: {self._run_session.logs_path}')

        self._session_initialized = True

    def state_callback(self, msg: VehicleStateMsg) -> None:
        """Handle incoming vehicle state message."""
        state = VehicleState.from_msg(msg)
        self.plot_manager.push_state(state)
        self._state_count += 1

    def compute_and_publish_state(self) -> None:
        """Aggregate measured sim topics into VehicleState, then plot and publish it."""
        target_time = self.synchronizer.get_latest_time()
        if target_time is None:
            return

        synced = self.synchronizer.get_synchronized(target_time)
        if synced is None:
            return

        state = self.adapter.compute_state(synced, self._last_state)

        if self._last_state.timestamp > 0.0:
            dt = state.timestamp - self._last_state.timestamp
            if 0.0 < dt < 0.1:
                self._distance_accumulator += state.speed * dt

        state.distance_traveled = self._distance_accumulator
        self.plot_manager.push_state(state)

        if self._state_pub is not None:
            self._state_pub.publish(state.to_msg())

        self._last_state = state
        self._state_count += 1

    def refresh_callback(self) -> None:
        """Refresh plots at the configured rate."""
        if not self._window_open:
            return

        self._window_open = self.plot_manager.refresh()

        if not self._window_open:
            self._shutdown_from_window_close = True
            self.get_logger().info('Plot window closed, shutting down')
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown with optional plot/data export."""
        self._save_on_exit()
        if self._close_plots_on_shutdown or self._shutdown_from_window_close:
            self.plot_manager.close()
        rclpy.shutdown()

    def _linger_until_window_closed(self) -> None:
        """Keep GUI alive after ROS shutdown until user closes the window."""
        if not self._gui_available:
            return
        if self._shutdown_from_window_close:
            return
        self.get_logger().info('ROS shutdown complete. Close the plot window to exit.')
        while self.plot_manager._backend.process_events():
            time.sleep(0.05)

    def _save_on_exit(self) -> None:
        """Save plots and data if configured."""
        if self._artifacts_saved:
            return

        if not self._session_initialized or not self._run_session:
            # Create a fallback session if none exists
            if self._save_plots or self._save_plot_data:
                self._run_session = RunSession.create_new(self._base_path)
                self._run_session.ensure_directories()

        if not self._run_session:
            return

        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            artifacts_saved = False

            # Save plot image
            if self._save_plots:
                plot_path = self._run_session.plots_path / "virtual_sensors.png"
                try:
                    self.plot_manager.export_static_dashboard(str(plot_path))
                    if plot_path.exists():
                        self.get_logger().info(f'Plots saved to: {plot_path}')
                        artifacts_saved = True
                    else:
                        self.get_logger().error(f'Plot export did not create file: {plot_path}')
                except BaseException as e:
                    self.get_logger().error(f'Failed to save plots: {e}')

            # Save plot data
            if self._save_plot_data:
                try:
                    exported = self.plot_manager.export_data(self._run_session.logs_path)
                    self.get_logger().info(f'Plot data saved: {len(exported)} files to {self._run_session.logs_path}')
                    artifacts_saved = True
                except BaseException as e:
                    self.get_logger().error(f'Failed to save plot data: {e}')

            self._artifacts_saved = artifacts_saved
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

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
        if node._close_plots_on_shutdown or node._shutdown_from_window_close:
            node.plot_manager.close()
        else:
            node._linger_until_window_closed()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
