#!/usr/bin/env python3
"""
LoggerNode - Logs vehicle state and diagnostics to files.

Subscribes to /vehicle_plotter/state and writes data to
Parquet or CSV files. Supports multi-machine run synchronization
via /run_session topic.

Storage Layout:
    ./multidata/<prefix>_<timestamp>/logs/
        vehicle_state_0000.parquet
        metadata.json

Where <prefix> is 'sim' for simulation.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from pathlib import Path
from typing import Dict, List, Optional
import csv
import math
import os
import signal
import threading
import time
import numpy as np

from diagnostic_msgs.msg import DiagnosticArray
from eufs_msgs.msg import ConeArrayWithCovariance
from nav_msgs.msg import Odometry, Path as NavPath
from tf2_ros import Buffer, TransformListener
from vehicle_plotter_msgs.msg import VehicleState as VehicleStateMsg
from vehicle_plotter_msgs.msg import RunSession as RunSessionMsg
from std_msgs.msg import String

from ..core.vehicle_state import VehicleState
from ..core.run_session import RunSession
from ..core.qos_profiles import PLOTTER_QOS, RELIABLE_SENSOR_QOS
from ..logging.log_writer import LogWriter
from ..logging.log_config import LogConfig, declare_and_load_config
from ..logging.path_tracking_eval import GateLapCounter, GTMidline
from .path_eval_runner import PathEvalRunner
from .plot_runner import PlotRunner

_OFF_TRACK_NO_CONE_REASONS: frozenset[str] = frozenset({
    'no_safe_chain',
    'hold_expired_no_path',
    'holding_previous_valid',
    'hysteresis_holding',
    'stop_if_no_path',
})


class LoggerNode(Node, PathEvalRunner, PlotRunner):
    """
    Data logging node for vehicle state and diagnostics.

    Subscribes to /vehicle_plotter/state and writes synchronized
    state data to disk in configurable formats (Parquet, CSV).

    Supports multi-machine synchronization via /run_session topic:
    - If run_session received, uses that run_id
    - Otherwise generates own run_id after timeout

    Parameters:
        format (str): Output format ('parquet' or 'csv')
        compression (str): Compression for parquet ('snappy', 'gzip', 'zstd', 'none')
        base_path (str): Base directory for logs (default: ./multidata)
        session_name (str): Custom session name (auto-generated if empty)
        flush_interval_sec (float): Seconds between disk flushes
        buffer_size (int): Max records to buffer before flush
        wait_for_session (bool): Wait for /run_session before logging
        session_timeout_sec (float): Timeout to wait for /run_session
        adapter (str): Sensor adapter type ('gazebo') - determines directory prefix
        auto_plot_on_shutdown (bool): Generate offline plots when logger shuts down
    """

    def __init__(self):
        super().__init__('logger_node')

        self._cfg = declare_and_load_config(self)
        self._log_format = self._cfg.log_format
        self._compression = self._cfg.compression
        base_path_str = self._cfg.base_path_str
        self._session_name = self._cfg.session_name
        self._flush_interval = self._cfg.flush_interval_sec
        self._buffer_size = self._cfg.buffer_size
        state_topic = self._cfg.state_topic
        enable_logging = self._cfg.enable_logging
        enable_state_logging = self._cfg.enable_state_logging
        self._wait_for_session = self._cfg.wait_for_session
        self._session_timeout = self._cfg.session_timeout_sec
        self._adapter_type = self._cfg.adapter_type
        self._auto_plot = self._cfg.auto_plot_on_shutdown
        self._camera_cone_eval_topic = self._cfg.camera_cone_eval_topic
        self._lidar_cone_eval_topic = self._cfg.lidar_cone_eval_topic
        self._path_tracking_eval_enabled = self._cfg.path_tracking_eval_enabled
        self._path_tracking_eval_rate_hz = self._cfg.path_tracking_eval_rate_hz
        self._path_tracking_eval_gt_track_topic = self._cfg.path_tracking_eval_gt_track_topic
        self._path_tracking_eval_odom_topic = self._cfg.path_tracking_eval_odom_topic
        self._path_tracking_eval_planner_path_topic = self._cfg.path_tracking_eval_planner_path_topic
        self._path_tracking_eval_track_name = self._cfg.path_tracking_eval_track_name
        self._path_tracking_eval_tf_timeout_sec = self._cfg.path_tracking_eval_tf_timeout_sec
        self._path_tracking_eval_filename = self._cfg.path_tracking_eval_filename
        self._path_tracking_eval_summary_json = self._cfg.path_tracking_eval_summary_json
        self._path_tracking_eval_summary_txt = self._cfg.path_tracking_eval_summary_txt
        self._path_tracking_eval_autostop_laps = self._cfg.path_tracking_eval_autostop_laps
        self._off_track_autostop_enabled = self._cfg.off_track_autostop_enabled
        self._off_track_autostop_timeout_s = self._cfg.off_track_autostop_timeout_s
        self._off_track_autostop_planner_diag_topic = self._cfg.off_track_autostop_planner_diag_topic
        self._control_reference_wheelbase_m = self._cfg.control_reference_wheelbase_m

        # Parse base path
        if base_path_str:
            self._base_path = Path(base_path_str).expanduser()
        else:
            self._base_path = None  # Will be determined by RunSession

        self.get_logger().info(f'LoggerNode starting...')
        self.get_logger().info(f'  Format: {self._log_format}')
        self.get_logger().info(f'  Logging enabled: {enable_logging}')
        self.get_logger().info(f'  Wait for session: {self._wait_for_session}')

        self._enable_logging = enable_logging
        self._enable_state_logging = bool(enable_state_logging)
        self._run_session: Optional[RunSession] = None
        self.log_writer: Optional[LogWriter] = None
        self._session_initialized = False
        self._buffered_states = []  # Buffer states until session is ready
        self._shutdown_called = False
        self._cone_range_rmse_samples_by_source: Dict[str, List[Dict[str, object]]] = {
            'monocular': [],
            'stereo': [],
            'lidar': [],
        }
        self._path_eval_tf_buffer = None
        self._path_eval_tf_listener = None
        self._path_eval_latest_gt_msg: Optional[ConeArrayWithCovariance] = None
        self._path_eval_gt_midline_source: Optional[GTMidline] = None
        self._path_eval_last_gt_midline_xy = np.empty((0, 2), dtype=np.float64)
        self._path_eval_last_gt_left_xy = np.empty((0, 2), dtype=np.float64)
        self._path_eval_last_gt_right_xy = np.empty((0, 2), dtype=np.float64)
        self._path_eval_last_target_frame = ''
        self._path_eval_start_xy: Optional[np.ndarray] = None
        self._path_eval_start_heading_xy: Optional[np.ndarray] = None
        self._path_eval_vehicle_xy = np.asarray([float('nan'), float('nan')], dtype=np.float64)
        self._path_eval_vehicle_frame = ''
        self._path_eval_vehicle_child_frame = ''
        self._path_eval_vehicle_yaw_rad = float('nan')
        self._path_eval_vehicle_stamp = None
        self._path_eval_planner_xy = np.empty((0, 2), dtype=np.float64)
        self._path_eval_planner_frame = ''
        self._path_eval_planner_stamp = None
        self._path_eval_reference_trace_points: list[np.ndarray] = []
        self._path_eval_file_handle = None
        self._path_eval_csv_writer: Optional[csv.DictWriter] = None
        self._path_eval_flush_counter = 0
        self._path_eval_flush_stride = max(1, int(self._path_tracking_eval_rate_hz * 2.0))
        self._path_eval_timer = None
        self._path_eval_identity_warned_pairs: set[tuple[str, str]] = set()
        self._path_eval_smalltrack_gate_source = None
        self._path_eval_smalltrack_lap_counter: Optional[GateLapCounter] = None
        self._path_eval_smalltrack_completed_laps = 0
        self._path_eval_smalltrack_lap_times_sec: list[float] = []
        self._path_eval_smalltrack_autostop_triggered = False
        self._path_eval_run_start_sec: Optional[float] = None
        self._path_eval_run_last_sec: Optional[float] = None
        self._off_track_no_cone_since: Optional[float] = None
        self._off_track_autostop_triggered = False

        if self._path_tracking_eval_enabled:
            self._path_eval_tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
            self._path_eval_tf_listener = TransformListener(self._path_eval_tf_buffer, self)

        if enable_logging:
            # Subscribe to run session for multi-machine sync
            self.session_sub = self.create_subscription(
                RunSessionMsg,
                '/run_session',
                self.session_callback,
                RELIABLE_SENSOR_QOS,
            )

            # Set timeout to create own session if not received
            if self._wait_for_session:
                self.session_timer = self.create_timer(
                    self._session_timeout,
                    self.session_timeout_callback
                )
                self.get_logger().info(
                    f'  Waiting up to {self._session_timeout}s for /run_session...'
                )
            else:
                # Create session immediately
                self._initialize_session(None)

        self._setup_cone_subscriptions()
        self._setup_path_tracking_eval_subscriptions()
        self._setup_off_track_autostop_subscription()

        if self._enable_state_logging:
            self.state_sub = self.create_subscription(
                VehicleStateMsg,
                state_topic,
                self.state_callback,
                PLOTTER_QOS,
            )
        else:
            self.state_sub = None

        # Flush timer (will be created after session init)
        self._flush_timer = None

        # Status timer
        self.status_timer = self.create_timer(10.0, self.status_callback)

        if self._enable_state_logging:
            self.get_logger().info(f'LoggerNode started, subscribed to {state_topic}')
        else:
            self.get_logger().info('LoggerNode started without vehicle state subscription')

        # Ensure shutdown handler runs on ROS shutdown as well
        # NOTE: rclpy.on_shutdown / Node.add_on_shutdown are not available in Humble
        rclpy.get_default_context().on_shutdown(self.shutdown)

    def _setup_cone_subscriptions(self) -> None:
        self._cone_subscriptions = []
        if self._camera_cone_eval_topic:
            self._register_cone_stream(self._camera_cone_eval_topic)
        if self._lidar_cone_eval_topic:
            self._register_cone_stream(self._lidar_cone_eval_topic)
        if self._cone_subscriptions:
            self.get_logger().info(
                "Cone logging enabled: "
                f"camera_prefix={self._camera_cone_eval_topic or 'disabled'} "
                f"lidar_prefix={self._lidar_cone_eval_topic or 'disabled'}"
            )

    def _register_cone_stream(
        self,
        topic_prefix: str,
    ) -> None:
        prefix = self._derive_cone_metrics_prefix(topic_prefix)
        self._cone_subscriptions.append(
            self.create_subscription(
                String,
                f'{prefix}/cone_depth_samples',
                self._cone_depth_samples_callback,
                10,
            )
        )

    def _setup_path_tracking_eval_subscriptions(self) -> None:
        if not self._path_tracking_eval_enabled:
            return
        self._path_eval_gt_sub = self.create_subscription(
            ConeArrayWithCovariance,
            self._path_tracking_eval_gt_track_topic,
            self._path_tracking_eval_gt_callback,
            10,
        )
        self._path_eval_odom_sub = self.create_subscription(
            Odometry,
            self._path_tracking_eval_odom_topic,
            self._path_tracking_eval_odom_callback,
            10,
        )
        self._path_eval_path_sub = self.create_subscription(
            NavPath,
            self._path_tracking_eval_planner_path_topic,
            self._path_tracking_eval_path_callback,
            10,
        )
        self.get_logger().info(
            'Path tracking evaluation subscriptions enabled: '
            f'gt_track={self._path_tracking_eval_gt_track_topic} '
            f'odom={self._path_tracking_eval_odom_topic} '
            f'planner_path={self._path_tracking_eval_planner_path_topic}'
        )

    def _setup_off_track_autostop_subscription(self) -> None:
        if not self._off_track_autostop_enabled:
            return
        self._off_track_planner_diag_sub = self.create_subscription(
            DiagnosticArray,
            self._off_track_autostop_planner_diag_topic,
            self._off_track_planner_diag_callback,
            10,
        )
        self.get_logger().info(
            f'Off-track autostop enabled: timeout={self._off_track_autostop_timeout_s}s '
            f'topic={self._off_track_autostop_planner_diag_topic}'
        )

    @staticmethod
    def _extract_operator_reason(msg: DiagnosticArray) -> str:
        for status in getattr(msg, 'status', []):
            name = str(getattr(status, 'name', ''))
            if not name.endswith('/stability'):
                continue
            for item in getattr(status, 'values', []):
                if str(getattr(item, 'key', '')) == 'operator_reason':
                    return str(getattr(item, 'value', '')).strip()
        return ''

    def _off_track_planner_diag_callback(self, msg: DiagnosticArray) -> None:
        if self._off_track_autostop_triggered:
            return
        reason = self._extract_operator_reason(msg)
        if reason in _OFF_TRACK_NO_CONE_REASONS:
            now = float(self.get_clock().now().nanoseconds) * 1e-9
            if self._off_track_no_cone_since is None:
                self._off_track_no_cone_since = now
            elapsed = now - self._off_track_no_cone_since
            if elapsed >= self._off_track_autostop_timeout_s:
                self._off_track_autostop_triggered = True
                self.get_logger().warn(
                    f'No visible cones for {elapsed:.1f}s (reason={reason!r}). '
                    'Triggering clean exit to generate plots.'
                )
                self.shutdown()
                self._request_process_exit(parent_delay_s=0.1, force_delay_s=5.0)
                if rclpy.ok():
                    rclpy.shutdown()
        else:
            self._off_track_no_cone_since = None

    def _request_process_exit(self, *, parent_delay_s: float = 0.0, force_delay_s: float = 5.0) -> None:
        # Autostop is meant to terminate the full launched sim, not only this logger node.
        parent_pid = os.getppid()
        process_group_id = os.getpgrp()

        def _signal_parent_launch() -> None:
            try:
                if parent_pid > 1:
                    os.kill(parent_pid, signal.SIGINT)
            except Exception as exc:
                self._safe_log_warn(f'Failed to interrupt parent launch process for autostop: {exc}')

        def _force_exit_process_group() -> None:
            try:
                if parent_pid > 1:
                    os.kill(parent_pid, signal.SIGINT)
            except Exception:
                pass
            try:
                os.killpg(process_group_id, signal.SIGINT)
            except Exception:
                pass
            os._exit(0)

        try:
            parent_timer = threading.Timer(max(0.0, float(parent_delay_s)), _signal_parent_launch)
            parent_timer.daemon = True
            parent_timer.start()
            force_timer = threading.Timer(max(0.0, float(force_delay_s)), _force_exit_process_group)
            force_timer.daemon = True
            force_timer.start()
        except Exception as exc:
            self.get_logger().warn(f'Failed to schedule launch shutdown for autostop exit: {exc}')
            if parent_pid > 1:
                os.kill(parent_pid, signal.SIGINT)
            os.killpg(process_group_id, signal.SIGINT)

    @staticmethod
    def _derive_cone_metrics_prefix(cone_eval_topic: str) -> str:
        return cone_eval_topic.strip().rstrip('/')

    @staticmethod
    def _parse_float(value: str) -> float:
        text = value.strip().lower()
        if text in ('', 'n/a', 'nan', 'none'):
            return float('nan')
        try:
            number = float(text)
        except ValueError:
            return float('nan')
        if math.isfinite(number):
            return number
        return float('nan')

    @staticmethod
    def _normalize_cone_source_name(value: str) -> str:
        source = str(value).strip().lower()
        if source in {'mono', 'monocular'}:
            return 'monocular'
        if source in {'stereo', 'camera'}:
            return 'stereo'
        if source == 'lidar':
            return 'lidar'
        return ''

    def _cone_depth_samples_callback(self, msg: String) -> None:
        payload = str(msg.data)
        lines = [line.strip() for line in payload.splitlines() if line.strip()]
        if not lines:
            return

        start_idx = 0
        first = lines[0].lower().replace(' ', '')
        use_new_schema = first.startswith('source,gt_range_m,error_m')
        use_old_schema = first.startswith('gt_range_m,ex_m,ey_m')
        if use_new_schema or use_old_schema:
            start_idx = 1

        timestamp_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        if timestamp_sec <= 0.0:
            timestamp_sec = time.monotonic()

        for line in lines[start_idx:]:
            parts = [part.strip() for part in line.split(',')]
            source = ''
            gt_range_m = float('nan')
            error_m = float('nan')
            predicted_class_id = float('nan')
            ground_truth_class_id = float('nan')
            if len(parts) >= 3 and use_new_schema:
                source = self._normalize_cone_source_name(parts[0])
                gt_range_m = self._parse_float(parts[1])
                error_m = self._parse_float(parts[2])
                if len(parts) >= 5:
                    predicted_class_id = self._parse_float(parts[3])
                    ground_truth_class_id = self._parse_float(parts[4])
            elif len(parts) >= 3:
                gt_range_m = self._parse_float(parts[0])
                ex_m = self._parse_float(parts[1])
                ey_m = self._parse_float(parts[2])
                error_m = math.hypot(ex_m, ey_m) if math.isfinite(ex_m) and math.isfinite(ey_m) else float('nan')
                source = 'stereo'
            if not source or not (math.isfinite(gt_range_m) and math.isfinite(error_m)):
                continue
            self._cone_range_rmse_samples_by_source[source].append(
                {
                    'timestamp': timestamp_sec,
                    'source': source,
                    'gt_range_m': gt_range_m,
                    'error_m': error_m,
                    'predicted_class_id': predicted_class_id,
                    'ground_truth_class_id': ground_truth_class_id,
                }
            )

    def session_callback(self, msg: RunSessionMsg) -> None:
        """Handle incoming run session message."""
        if self._session_initialized:
            return  # Already initialized

        self.get_logger().info(
            f'Received run_session: {msg.run_id} from {msg.originator_hostname}'
        )

        # Cancel timeout timer
        if hasattr(self, 'session_timer') and self.session_timer:
            self.session_timer.cancel()
            self.session_timer = None

        # Initialize with received session
        self._initialize_session(msg)

    def session_timeout_callback(self) -> None:
        """Called when session timeout expires - create own session."""
        if self._session_initialized:
            return

        self.get_logger().info('No /run_session received, creating own session')

        # Cancel timer
        if self.session_timer:
            self.session_timer.cancel()
            self.session_timer = None

        # Create own session
        self._initialize_session(None)

    def _initialize_session(self, msg: Optional[RunSessionMsg]) -> None:
        """Initialize the logging session."""
        if self._session_initialized:
            return

        # Create or adopt run session
        if msg:
            self._run_session = RunSession.from_msg(msg, self._base_path)
        else:
            self._run_session = RunSession.create_new(self._base_path, self._adapter_type)

        # Ensure directories exist
        self._run_session.ensure_directories()

        # Save session info
        self._run_session.save_session_info()

        # Create log configuration
        config = LogConfig(
            base_path=self._run_session.logs_path,
            format=self._log_format,
            compression=self._compression,
            session_name='',  # Use base_path directly, no sub-session
            flush_interval_sec=self._flush_interval,
            buffer_size=self._buffer_size,
        )

        # Initialize log writer
        self.log_writer = LogWriter(config)

        self.get_logger().info(f'  Run ID: {self._run_session.run_id}')
        self.get_logger().info(f'  Session path: {self._run_session.session_path}')
        self.get_logger().info(f'  Logs path: {self._run_session.logs_path}')

        # Create flush timer
        self._flush_timer = self.create_timer(self._flush_interval, self.flush_callback)
        self._initialize_path_tracking_eval_output()

        # Flush any buffered states
        for state in self._buffered_states:
            self.log_writer.write(state)
        self._buffered_states.clear()

        self._session_initialized = True

    def state_callback(self, msg: VehicleStateMsg) -> None:
        """Handle incoming vehicle state message."""
        if not self._enable_logging:
            return

        state = VehicleState.from_msg(msg)

        if self._session_initialized and self.log_writer is not None:
            self.log_writer.write(state)
        elif self._wait_for_session:
            # Buffer states until session is ready (limit buffer size)
            if len(self._buffered_states) < 1000:
                self._buffered_states.append(state)

    def flush_callback(self) -> None:
        """Periodic flush to disk."""
        if self.log_writer is not None:
            self.log_writer.flush()
        if self._path_eval_file_handle is not None:
            self._path_eval_file_handle.flush()

    def status_callback(self) -> None:
        """Log periodic status update."""
        if self.log_writer is None:
            if not self._session_initialized:
                self.get_logger().info('Waiting for session initialization...')
            return

        status = self.log_writer.get_status()
        self.get_logger().info(
            f"Logged {status['total_records']} records, "
            f"{status['bytes_written'] / 1024:.1f} KB written"
        )

    def shutdown(self) -> None:
        """Clean shutdown with final flush and optional plot generation."""
        if self._shutdown_called:
            return
        self._shutdown_called = True

        # Block shutdown signals for the duration of finalization so that external
        # shutdown signals (e.g. ros2 launch killing the process group when
        # another node exits) cannot interrupt matplotlib plot generation before
        # the post-run plots are saved.
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except (OSError, ValueError):
            pass

        if self.log_writer is not None:
            self.log_writer.close()
            self.get_logger().info(
                f"Logger closed, {self.log_writer.total_records} records written to "
                f"{self.log_writer.session_path}"
            )

            if self._run_session is not None:
                self._save_cone_range_rmse_samples_csv()
                self._finalize_path_tracking_eval_outputs()

            # Auto-generate plots if enabled
            if self._auto_plot and self._run_session is not None:
                self._generate_offline_plots()

    def _save_cone_range_rmse_samples_csv(self) -> None:
        if self._run_session is None:
            return
        fieldnames = [
            'timestamp',
            'source',
            'gt_range_m',
            'error_m',
            'predicted_class_id',
            'ground_truth_class_id',
        ]
        filenames = {
            'monocular': 'cone_range_rmse_samples_mono.csv',
            'stereo': 'cone_range_rmse_samples_stereo.csv',
            'lidar': 'cone_range_rmse_samples_lidar.csv',
        }
        for source, filename in filenames.items():
            rows = self._cone_range_rmse_samples_by_source.get(source, [])
            out_path = self._run_session.logs_path / filename
            with open(out_path, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self._safe_log_info(f'Saved cone range RMSE sample log: {out_path}')

    def _safe_log_info(self, message: str) -> None:
        try:
            self.get_logger().info(message)
        except Exception:
            print(message)

    def _safe_log_warn(self, message: str) -> None:
        try:
            self.get_logger().warn(message)
        except Exception:
            print(message)

    @property
    def run_session(self) -> Optional[RunSession]:
        """Get the current run session."""
        return self._run_session


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
