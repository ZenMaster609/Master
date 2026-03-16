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
from pathlib import Path
from typing import Dict, List, Optional
import csv
import math
import time
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import JointState
from vehicle_plotter_msgs.msg import VehicleState as VehicleStateMsg
from vehicle_plotter_msgs.msg import RunSession as RunSessionMsg
from std_msgs.msg import Float32, String

from ..core.vehicle_state import VehicleState
from ..core.run_session import RunSession
from ..core.qos_profiles import PLOTTER_QOS, RELIABLE_SENSOR_QOS
from ..logging.log_writer import LogWriter
from ..logging.log_config import LogConfig
from ..logging.steering_diagnostics import (
    PLANNER_DIAG_DEFAULTS,
    analyze_csv,
    heading_error,
    nearest_point_on_polyline,
    parse_planner_diag,
    signed_cross_track_error,
    write_summary_files,
)
from ..logging.stanley_debug_plots import generate_stanley_debug_plot
from ..logging.thesis_controller_diagnostics import (
    THESIS_DIAG_FIELDNAMES,
    analyze_thesis_csv,
    build_thesis_sample_row,
    write_thesis_summary_files,
)
from ..logging.thesis_controller_plots import generate_thesis_controller_plot


class LoggerNode(Node):
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

        # Declare parameters
        self.declare_parameter('format', 'parquet')
        self.declare_parameter('compression', 'snappy')
        self.declare_parameter('base_path', '')  # Empty = auto-detect
        self.declare_parameter('session_name', '')
        self.declare_parameter('flush_interval_sec', 5.0)
        self.declare_parameter('buffer_size', 1000)
        self.declare_parameter('state_topic', 'vehicle_plotter/state')
        self.declare_parameter('enable_logging', True)
        self.declare_parameter('enable_state_logging', True)
        self.declare_parameter('wait_for_session', True)
        self.declare_parameter('session_timeout_sec', 5.0)
        self.declare_parameter('adapter', 'gazebo')  # Determines directory prefix
        self.declare_parameter('auto_plot_on_shutdown', True)
        self.declare_parameter('camera_cone_eval_topic', '/sim/stereo/eval')
        self.declare_parameter('lidar_cone_eval_topic', '/sim/lidar/eval')
        self.declare_parameter('controller_diagnostics_enabled', False)
        self.declare_parameter('thesis_controller_diagnostics_enabled', False)
        self.declare_parameter('controller_diagnostics_rate_hz', 50.0)
        self.declare_parameter('controller_diagnostics_cmd_topic', '/cmd')
        self.declare_parameter('controller_diagnostics_steering_topic', '/sim/steering_angle')
        self.declare_parameter('controller_diagnostics_joint_states_topic', '/sim/raw/joint_states')
        self.declare_parameter('controller_diagnostics_odom_topic', '/sim/odom')
        self.declare_parameter('controller_diagnostics_path_topic', '/planned_centerline')
        self.declare_parameter('controller_diagnostics_planner_diag_topic', '/delaunay_planner/diagnostics')
        self.declare_parameter('controller_diagnostics_filename', 'steering_tracking_diagnostics.csv')
        self.declare_parameter('controller_diagnostics_summary_json', 'steering_tracking_summary.json')
        self.declare_parameter('controller_diagnostics_summary_txt', 'steering_tracking_summary.txt')
        self.declare_parameter('thesis_controller_diagnostics_filename', 'thesis_controller_diagnostics.csv')
        self.declare_parameter('thesis_controller_diagnostics_summary_json', 'thesis_controller_diagnostics_summary.json')
        self.declare_parameter('thesis_controller_diagnostics_summary_txt', 'thesis_controller_diagnostics_summary.txt')

        # Get parameters
        self._log_format = self.get_parameter('format').value
        self._compression = self.get_parameter('compression').value
        base_path_str = self.get_parameter('base_path').value
        self._session_name = self.get_parameter('session_name').value
        self._flush_interval = self.get_parameter('flush_interval_sec').value
        self._buffer_size = self.get_parameter('buffer_size').value
        state_topic = self.get_parameter('state_topic').value
        enable_logging = self.get_parameter('enable_logging').value
        enable_state_logging = self.get_parameter('enable_state_logging').value
        self._wait_for_session = self.get_parameter('wait_for_session').value
        self._session_timeout = self.get_parameter('session_timeout_sec').value
        self._adapter_type = self.get_parameter('adapter').value
        self._auto_plot = self.get_parameter('auto_plot_on_shutdown').value
        self._camera_cone_eval_topic = str(
            self.get_parameter('camera_cone_eval_topic').value
        ).strip()
        self._lidar_cone_eval_topic = str(
            self.get_parameter('lidar_cone_eval_topic').value
        ).strip()
        self._steering_diag_enabled = bool(
            self.get_parameter('controller_diagnostics_enabled').value
        )
        self._thesis_diag_enabled = bool(
            self.get_parameter('thesis_controller_diagnostics_enabled').value
        )
        self._controller_diag_enabled = self._steering_diag_enabled or self._thesis_diag_enabled
        self._steering_diag_rate_hz = max(
            1.0,
            float(self.get_parameter('controller_diagnostics_rate_hz').value),
        )
        self._steering_diag_cmd_topic = str(
            self.get_parameter('controller_diagnostics_cmd_topic').value
        ).strip() or '/cmd'
        self._steering_diag_steering_topic = str(
            self.get_parameter('controller_diagnostics_steering_topic').value
        ).strip() or '/sim/steering_angle'
        self._steering_diag_joint_states_topic = str(
            self.get_parameter('controller_diagnostics_joint_states_topic').value
        ).strip() or '/sim/raw/joint_states'
        self._steering_diag_odom_topic = str(
            self.get_parameter('controller_diagnostics_odom_topic').value
        ).strip() or '/sim/odom'
        self._steering_diag_path_topic = str(
            self.get_parameter('controller_diagnostics_path_topic').value
        ).strip() or '/planned_centerline'
        self._steering_diag_planner_diag_topic = str(
            self.get_parameter('controller_diagnostics_planner_diag_topic').value
        ).strip() or '/delaunay_planner/diagnostics'
        self._steering_diag_filename = str(
            self.get_parameter('controller_diagnostics_filename').value
        ).strip() or 'steering_tracking_diagnostics.csv'
        self._steering_diag_summary_json = str(
            self.get_parameter('controller_diagnostics_summary_json').value
        ).strip() or 'steering_tracking_summary.json'
        self._steering_diag_summary_txt = str(
            self.get_parameter('controller_diagnostics_summary_txt').value
        ).strip() or 'steering_tracking_summary.txt'
        self._thesis_diag_filename = str(
            self.get_parameter('thesis_controller_diagnostics_filename').value
        ).strip() or 'thesis_controller_diagnostics.csv'
        self._thesis_diag_summary_json = str(
            self.get_parameter('thesis_controller_diagnostics_summary_json').value
        ).strip() or 'thesis_controller_diagnostics_summary.json'
        self._thesis_diag_summary_txt = str(
            self.get_parameter('thesis_controller_diagnostics_summary_txt').value
        ).strip() or 'thesis_controller_diagnostics_summary.txt'

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
        self._cone_range_rmse_samples_by_suffix: Dict[str, List[Dict[str, object]]] = {
            '': [],
            'lidar': [],
        }
        self._diag_cmd_stamp_sec = float('nan')
        self._diag_cmd_recv_sec = float('nan')
        self._diag_desired_steering_rad = float('nan')
        self._diag_desired_speed_mps = float('nan')
        self._diag_actual_steering_deg = float('nan')
        self._diag_vehicle_x_m = float('nan')
        self._diag_vehicle_y_m = float('nan')
        self._diag_vehicle_yaw_rad = float('nan')
        self._diag_vehicle_yaw_rate_rps = float('nan')
        self._diag_vehicle_speed_mps = float('nan')
        self._diag_centerline_xy = np.empty((0, 2), dtype=np.float64)
        self._diag_planner_metrics: Dict[str, float] = dict(PLANNER_DIAG_DEFAULTS)
        self._diag_file_handle = None
        self._diag_csv_writer: Optional[csv.DictWriter] = None
        self._thesis_diag_file_handle = None
        self._thesis_diag_csv_writer: Optional[csv.DictWriter] = None
        self._diag_flush_counter = 0
        self._diag_flush_stride = max(1, int(self._steering_diag_rate_hz * 2.0))
        self._diag_timer = None

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
        self._setup_steering_diag_subscriptions()

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
            self._register_cone_stream(
                self._camera_cone_eval_topic,
                suffix='',
            )
        if self._lidar_cone_eval_topic:
            self._register_cone_stream(
                self._lidar_cone_eval_topic,
                suffix='lidar',
            )
        if self._cone_subscriptions:
            self.get_logger().info(
                "Cone logging enabled: "
                f"camera_prefix={self._camera_cone_eval_topic or 'disabled'} "
                f"lidar_prefix={self._lidar_cone_eval_topic or 'disabled'}"
            )

    def _register_cone_stream(
        self,
        topic_prefix: str,
        *,
        suffix: str,
    ) -> None:
        prefix = self._derive_cone_metrics_prefix(topic_prefix)
        self._cone_subscriptions.append(
            self.create_subscription(
                String,
                f'{prefix}/cone_depth_samples',
                lambda msg, log_suffix=suffix: self._cone_depth_samples_callback(msg, log_suffix),
                10,
            )
        )

    def _setup_steering_diag_subscriptions(self) -> None:
        if not self._controller_diag_enabled:
            return
        self._diag_cmd_sub = self.create_subscription(
            AckermannDriveStamped,
            self._steering_diag_cmd_topic,
            self._steering_diag_cmd_callback,
            10,
        )
        self._diag_steering_sub = self.create_subscription(
            Float32,
            self._steering_diag_steering_topic,
            self._steering_diag_actual_callback,
            10,
        )
        self._diag_joint_states_sub = self.create_subscription(
            JointState,
            self._steering_diag_joint_states_topic,
            self._steering_diag_joint_states_callback,
            10,
        )
        self._diag_odom_sub = self.create_subscription(
            Odometry,
            self._steering_diag_odom_topic,
            self._steering_diag_odom_callback,
            10,
        )
        self._diag_path_sub = self.create_subscription(
            NavPath,
            self._steering_diag_path_topic,
            self._steering_diag_path_callback,
            10,
        )
        self._diag_planner_sub = self.create_subscription(
            DiagnosticArray,
            self._steering_diag_planner_diag_topic,
            self._steering_diag_planner_callback,
            10,
        )
        self.get_logger().info(
            'Controller diagnostics subscriptions enabled: '
            f'cmd={self._steering_diag_cmd_topic} '
            f'steering={self._steering_diag_steering_topic} '
            f'joint_states={self._steering_diag_joint_states_topic} '
            f'odom={self._steering_diag_odom_topic} '
            f'path={self._steering_diag_path_topic} '
            f'oscillation={self._steering_diag_enabled} thesis={self._thesis_diag_enabled}'
        )

    def _steering_diag_cmd_callback(self, msg: AckermannDriveStamped) -> None:
        self._diag_desired_steering_rad = float(msg.drive.steering_angle)
        self._diag_desired_speed_mps = float(msg.drive.speed)
        self._diag_cmd_recv_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        self._diag_cmd_stamp_sec = (
            float(msg.header.stamp.sec) + (float(msg.header.stamp.nanosec) * 1e-9)
        )

    def _steering_diag_actual_callback(self, msg: Float32) -> None:
        self._diag_actual_steering_deg = float(msg.data)

    def _steering_diag_joint_states_callback(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        candidates = (
            ('steering_fl_joint', 'steering_fr_joint'),
            ('left_steering_hinge_joint', 'right_steering_hinge_joint'),
            ('left_steering_joint', 'right_steering_joint'),
        )

        idx_left = -1
        idx_right = -1
        for left_name, right_name in candidates:
            if left_name in msg.name and right_name in msg.name:
                idx_left = msg.name.index(left_name)
                idx_right = msg.name.index(right_name)
                break

        # Last-resort heuristic: use first left/right steering-like joints.
        if idx_left < 0 or idx_right < 0:
            for idx, joint_name in enumerate(msg.name):
                token = str(joint_name).lower()
                if 'steering' not in token:
                    continue
                if idx_left < 0 and ('left' in token or token.endswith('_fl_joint')):
                    idx_left = idx
                elif idx_right < 0 and ('right' in token or token.endswith('_fr_joint')):
                    idx_right = idx
            if idx_left < 0 or idx_right < 0:
                return

        if idx_left >= len(msg.position) or idx_right >= len(msg.position):
            return
        left_rad = float(msg.position[idx_left])
        right_rad = float(msg.position[idx_right])
        avg_deg = math.degrees(0.5 * (left_rad + right_rad))
        self._diag_actual_steering_deg = avg_deg

    def _steering_diag_odom_callback(self, msg: Odometry) -> None:
        self._diag_vehicle_x_m = float(msg.pose.pose.position.x)
        self._diag_vehicle_y_m = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self._diag_vehicle_yaw_rad = self._yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        self._diag_vehicle_yaw_rate_rps = float(msg.twist.twist.angular.z)
        self._diag_vehicle_speed_mps = float(
            math.hypot(float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y))
        )

    def _steering_diag_path_callback(self, msg: NavPath) -> None:
        if not msg.poses:
            self._diag_centerline_xy = np.empty((0, 2), dtype=np.float64)
            return
        points = np.empty((len(msg.poses), 2), dtype=np.float64)
        for idx, pose_stamped in enumerate(msg.poses):
            points[idx, 0] = float(pose_stamped.pose.position.x)
            points[idx, 1] = float(pose_stamped.pose.position.y)
        self._diag_centerline_xy = points

    def _steering_diag_planner_callback(self, msg: DiagnosticArray) -> None:
        self._diag_planner_metrics = parse_planner_diag(msg)

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

    def _cone_depth_samples_callback(self, msg: String, log_suffix: str = '') -> None:
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
            source = 'unknown'
            gt_range_m = float('nan')
            error_m = float('nan')
            predicted_class_id = float('nan')
            ground_truth_class_id = float('nan')
            if len(parts) >= 3 and use_new_schema:
                source = parts[0].strip().lower() or 'unknown'
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
            if not (math.isfinite(gt_range_m) and math.isfinite(error_m)):
                continue
            if log_suffix not in self._cone_range_rmse_samples_by_suffix:
                self._cone_range_rmse_samples_by_suffix[log_suffix] = []
            self._cone_range_rmse_samples_by_suffix[log_suffix].append(
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
        self._initialize_steering_diag_output()

        # Flush any buffered states
        for state in self._buffered_states:
            self.log_writer.write(state)
        self._buffered_states.clear()

        self._session_initialized = True

    def _initialize_steering_diag_output(self) -> None:
        if not self._controller_diag_enabled or self._run_session is None:
            return
        if self._steering_diag_enabled:
            diag_path = self._run_session.logs_path / self._steering_diag_filename
            self._diag_file_handle = open(diag_path, 'w', newline='', encoding='utf-8')
            fieldnames = [
                'timestamp_sec',
                'cmd_stamp_sec',
                'cmd_age_sec',
                'desired_steering_rad',
                'desired_speed_mps',
                'actual_steering_deg',
                'actual_steering_rad',
                'steering_error_rad',
                'steering_error_abs_rad',
                'raw_steering_cmd_rad',
                'final_steering_cmd_rad',
                'steering_after_clamp_rad',
                'steering_after_filter_rad',
                'steering_after_rate_limit_rad',
                'steering_saturated_flag',
                'vehicle_x_m',
                'vehicle_y_m',
                'vehicle_yaw_rad',
                'vehicle_yaw_rate_rps',
                'vehicle_speed_mps',
                'speed_term_mps',
                'centerline_available',
                'centerline_point_count',
                'cte_m',
                'cte_abs_m',
                'heading_error_rad',
                'heading_error_abs_rad',
                'heading_contribution_rad',
                'cross_track_contribution_rad',
                'yaw_rate_damping_contribution_rad',
                'nearest_path_index',
                'heading_path_index',
                'target_point_x_base_m',
                'target_point_y_base_m',
                'target_point_x_frame_m',
                'target_point_y_frame_m',
                'nearest_path_point_x_m',
                'nearest_path_point_y_m',
                'planner_centerline_jump_max_m',
                'planner_selected_edge_churn_ratio',
                'planner_tracked_cones_frame_delta_p95_m',
            ]
            self._diag_csv_writer = csv.DictWriter(self._diag_file_handle, fieldnames=fieldnames)
            self._diag_csv_writer.writeheader()
            self.get_logger().info(f'Controller diagnostics CSV: {diag_path}')
        if self._thesis_diag_enabled:
            thesis_path = self._run_session.logs_path / self._thesis_diag_filename
            self._thesis_diag_file_handle = open(thesis_path, 'w', newline='', encoding='utf-8')
            self._thesis_diag_csv_writer = csv.DictWriter(
                self._thesis_diag_file_handle,
                fieldnames=THESIS_DIAG_FIELDNAMES,
            )
            self._thesis_diag_csv_writer.writeheader()
            self.get_logger().info(f'Thesis controller diagnostics CSV: {thesis_path}')
        self._diag_flush_counter = 0
        self._diag_timer = self.create_timer(1.0 / self._steering_diag_rate_hz, self._steering_diag_sample)

    def _steering_diag_sample(self) -> None:
        if (
            (self._diag_csv_writer is None or self._diag_file_handle is None)
            and (self._thesis_diag_csv_writer is None or self._thesis_diag_file_handle is None)
        ):
            return

        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        desired = float(self._diag_desired_steering_rad)
        actual_deg = float(self._diag_actual_steering_deg)
        actual_rad = math.radians(actual_deg) if math.isfinite(actual_deg) else float('nan')
        cmd_age_sec = now_sec - self._diag_cmd_recv_sec if math.isfinite(self._diag_cmd_recv_sec) else float('nan')

        centerline_count = int(self._diag_centerline_xy.shape[0])
        centerline_available = 1 if centerline_count >= 2 else 0
        cte_geom = float('nan')
        heading_geom = float('nan')
        nearest_idx_geom = float('nan')
        nearest_path_point = np.array([float('nan'), float('nan')], dtype=np.float64)
        if centerline_available and math.isfinite(self._diag_vehicle_x_m) and math.isfinite(self._diag_vehicle_y_m):
            nearest_idx_int, nearest_path_point = nearest_point_on_polyline(
                self._diag_vehicle_x_m,
                self._diag_vehicle_y_m,
                self._diag_centerline_xy,
            )
            nearest_idx_geom = float(nearest_idx_int)
            cte_geom, tangent_yaw = signed_cross_track_error(
                self._diag_vehicle_x_m,
                self._diag_vehicle_y_m,
                self._diag_centerline_xy,
            )
            if math.isfinite(tangent_yaw) and math.isfinite(self._diag_vehicle_yaw_rad):
                heading_geom = heading_error(self._diag_vehicle_yaw_rad, tangent_yaw)

        planner_jump = float(self._diag_planner_metrics.get('centerline_jump_max_m', float('nan')))
        planner_churn = float(self._diag_planner_metrics.get('selected_edge_churn_ratio', float('nan')))
        planner_tracked_delta = float(self._diag_planner_metrics.get('tracked_cones_frame_delta_p95_m', float('nan')))
        cte_ctrl = float(self._diag_planner_metrics.get('cross_track_error_m', float('nan')))
        heading_ctrl = float(self._diag_planner_metrics.get('heading_error_rad', float('nan')))
        cte_m = cte_ctrl if math.isfinite(cte_ctrl) else cte_geom
        heading_err = heading_ctrl if math.isfinite(heading_ctrl) else heading_geom

        raw_cmd = float(self._diag_planner_metrics.get('raw_steering_cmd_rad', float('nan')))
        final_cmd = float(self._diag_planner_metrics.get('final_steering_cmd_rad', float('nan')))
        if not math.isfinite(final_cmd):
            final_cmd = desired
        steering_ref = final_cmd if math.isfinite(final_cmd) else desired
        err = (actual_rad - steering_ref) if math.isfinite(actual_rad) and math.isfinite(steering_ref) else float('nan')

        vehicle_speed_dbg = float(self._diag_planner_metrics.get('vehicle_speed_mps', float('nan')))
        vehicle_speed = vehicle_speed_dbg if math.isfinite(vehicle_speed_dbg) else self._diag_vehicle_speed_mps
        nearest_path_index = float(self._diag_planner_metrics.get('nearest_path_index', float('nan')))
        if not math.isfinite(nearest_path_index):
            nearest_path_index = nearest_idx_geom

        row = {
            'timestamp_sec': now_sec,
            'cmd_stamp_sec': self._diag_cmd_stamp_sec,
            'cmd_age_sec': cmd_age_sec,
            'desired_steering_rad': desired,
            'desired_speed_mps': self._diag_desired_speed_mps,
            'actual_steering_deg': actual_deg,
            'actual_steering_rad': actual_rad,
            'steering_error_rad': err,
            'steering_error_abs_rad': abs(err) if math.isfinite(err) else float('nan'),
            'raw_steering_cmd_rad': raw_cmd,
            'final_steering_cmd_rad': final_cmd,
            'steering_after_clamp_rad': float(
                self._diag_planner_metrics.get('steering_after_clamp_rad', float('nan'))
            ),
            'steering_after_filter_rad': float(
                self._diag_planner_metrics.get('steering_after_filter_rad', float('nan'))
            ),
            'steering_after_rate_limit_rad': float(
                self._diag_planner_metrics.get('steering_after_rate_limit_rad', float('nan'))
            ),
            'steering_saturated_flag': float(
                self._diag_planner_metrics.get('steering_saturated_flag', float('nan'))
            ),
            'vehicle_x_m': self._diag_vehicle_x_m,
            'vehicle_y_m': self._diag_vehicle_y_m,
            'vehicle_yaw_rad': self._diag_vehicle_yaw_rad,
            'vehicle_yaw_rate_rps': self._diag_vehicle_yaw_rate_rps,
            'vehicle_speed_mps': vehicle_speed,
            'speed_term_mps': float(self._diag_planner_metrics.get('speed_term_mps', float('nan'))),
            'centerline_available': centerline_available,
            'centerline_point_count': centerline_count,
            'cte_m': cte_m,
            'cte_abs_m': abs(cte_m) if math.isfinite(cte_m) else float('nan'),
            'heading_error_rad': heading_err,
            'heading_error_abs_rad': abs(heading_err) if math.isfinite(heading_err) else float('nan'),
            'heading_contribution_rad': float(
                self._diag_planner_metrics.get('heading_contribution_rad', float('nan'))
            ),
            'cross_track_contribution_rad': float(
                self._diag_planner_metrics.get('cross_track_contribution_rad', float('nan'))
            ),
            'yaw_rate_damping_contribution_rad': float(
                self._diag_planner_metrics.get('yaw_rate_damping_contribution_rad', float('nan'))
            ),
            'nearest_path_index': nearest_path_index,
            'heading_path_index': float(self._diag_planner_metrics.get('heading_path_index', float('nan'))),
            'target_point_x_base_m': float(self._diag_planner_metrics.get('target_point_x_base_m', float('nan'))),
            'target_point_y_base_m': float(self._diag_planner_metrics.get('target_point_y_base_m', float('nan'))),
            'target_point_x_frame_m': float(self._diag_planner_metrics.get('target_point_x_frame_m', float('nan'))),
            'target_point_y_frame_m': float(self._diag_planner_metrics.get('target_point_y_frame_m', float('nan'))),
            'nearest_path_point_x_m': float(nearest_path_point[0]),
            'nearest_path_point_y_m': float(nearest_path_point[1]),
            'planner_centerline_jump_max_m': planner_jump,
            'planner_selected_edge_churn_ratio': planner_churn,
            'planner_tracked_cones_frame_delta_p95_m': planner_tracked_delta,
        }
        if self._diag_csv_writer is not None:
            self._diag_csv_writer.writerow(row)
        if self._thesis_diag_csv_writer is not None:
            thesis_row = build_thesis_sample_row(
                now_sec=now_sec,
                cmd_stamp_sec=self._diag_cmd_stamp_sec,
                cmd_recv_sec=self._diag_cmd_recv_sec,
                desired_steering_rad=desired,
                desired_speed_mps=self._diag_desired_speed_mps,
                actual_steering_deg=actual_deg,
                vehicle_x_m=self._diag_vehicle_x_m,
                vehicle_y_m=self._diag_vehicle_y_m,
                vehicle_yaw_rad=self._diag_vehicle_yaw_rad,
                vehicle_yaw_rate_rps=self._diag_vehicle_yaw_rate_rps,
                vehicle_speed_mps=vehicle_speed,
                centerline_xy=self._diag_centerline_xy,
                planner_metrics=self._diag_planner_metrics,
            )
            self._thesis_diag_csv_writer.writerow(thesis_row)
        self._diag_flush_counter += 1
        if self._diag_flush_counter >= self._diag_flush_stride:
            if self._diag_file_handle is not None:
                self._diag_file_handle.flush()
            if self._thesis_diag_file_handle is not None:
                self._thesis_diag_file_handle.flush()
            self._diag_flush_counter = 0

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
        if self._diag_file_handle is not None:
            self._diag_file_handle.flush()
        if self._thesis_diag_file_handle is not None:
            self._thesis_diag_file_handle.flush()

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

        if self.log_writer is not None:
            self.log_writer.close()
            self.get_logger().info(
                f"Logger closed, {self.log_writer.total_records} records written to "
                f"{self.log_writer.session_path}"
            )

            if self._run_session is not None:
                self._save_cone_range_rmse_samples_csv()
                self._finalize_steering_diag_outputs()

            # Auto-generate plots if enabled
            if self._auto_plot and self._run_session is not None:
                self._generate_offline_plots()

    def _finalize_steering_diag_outputs(self) -> None:
        if not self._controller_diag_enabled or self._run_session is None:
            return
        if self._diag_timer is not None:
            self._diag_timer.cancel()
            self._diag_timer = None
        if self._diag_file_handle is not None:
            self._diag_file_handle.flush()
            self._diag_file_handle.close()
            self._diag_file_handle = None
        self._diag_csv_writer = None
        if self._thesis_diag_file_handle is not None:
            self._thesis_diag_file_handle.flush()
            self._thesis_diag_file_handle.close()
            self._thesis_diag_file_handle = None
        self._thesis_diag_csv_writer = None

        if self._steering_diag_enabled:
            csv_path = self._run_session.logs_path / self._steering_diag_filename
            summary_json = self._run_session.logs_path / self._steering_diag_summary_json
            summary_txt = self._run_session.logs_path / self._steering_diag_summary_txt
            try:
                summary = analyze_csv(csv_path)
                write_summary_files(summary, summary_json, summary_txt)
                self._safe_log_info(
                    'Controller diagnostics summary: '
                    f"rms={summary.get('steering_error_rms_rad', float('nan')):.4f} rad "
                    f"cte_rms={summary.get('cte_rms_m', float('nan')):.4f} m "
                    f"lag={summary.get('lag_sec', float('nan')):.4f} s"
                )
            except Exception as exc:
                self._safe_log_warn(f'Failed controller diagnostics analysis: {exc}')

            try:
                debug_plot_path = self._run_session.plots_path / 'stanley_debug_plots.png'
                generated_path = generate_stanley_debug_plot(csv_path, debug_plot_path)
                if generated_path is not None:
                    self._safe_log_info(f'Generated controller diagnostics plot: {generated_path}')
                else:
                    self._safe_log_warn('Controller diagnostics plot skipped: no diagnostics rows')
            except Exception as exc:
                self._safe_log_warn(f'Failed controller diagnostics plot generation: {exc}')

        if self._thesis_diag_enabled:
            thesis_csv_path = self._run_session.logs_path / self._thesis_diag_filename
            thesis_summary_json = self._run_session.logs_path / self._thesis_diag_summary_json
            thesis_summary_txt = self._run_session.logs_path / self._thesis_diag_summary_txt
            try:
                summary = analyze_thesis_csv(thesis_csv_path)
                write_thesis_summary_files(summary, thesis_summary_json, thesis_summary_txt)
                self._safe_log_info(
                    'Thesis controller diagnostics summary: '
                    f"cte_rms={summary.get('cte_rms_m', float('nan')):.4f} m "
                    f"heading_rms={summary.get('heading_error_rms_rad', float('nan')):.4f} rad "
                    f"lag={summary.get('lag_sec', float('nan')):.4f} s"
                )
            except Exception as exc:
                self._safe_log_warn(f'Failed thesis controller diagnostics analysis: {exc}')

            try:
                thesis_plot_path = self._run_session.plots_path / 'thesis_controller_diagnostics.png'
                generated_path = generate_thesis_controller_plot(thesis_csv_path, thesis_plot_path)
                if generated_path is not None:
                    self._safe_log_info(f'Generated thesis controller diagnostics plot: {generated_path}')
                else:
                    self._safe_log_warn('Thesis controller diagnostics plot skipped: no diagnostics rows')
            except Exception as exc:
                self._safe_log_warn(f'Failed thesis controller diagnostics plot generation: {exc}')

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
        for suffix, rows in self._cone_range_rmse_samples_by_suffix.items():
            if not rows:
                continue
            out_path = self._run_session.logs_path / self._cone_output_filename(
                'cone_range_rmse_samples',
                'csv',
                suffix=suffix,
            )
            with open(out_path, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self._safe_log_info(f'Saved cone range RMSE sample log: {out_path}')

    def _generate_offline_plots(self) -> None:
        """Generate offline plots from logged data."""
        total = 0
        self._safe_log_info("Generating offline plots...")

        try:
            from ..plotting.offline_plotter import OfflinePlotter
            plotter = OfflinePlotter(self._run_session.session_path)
            generated = plotter.generate_plots()
            total += len(generated)
        except ImportError as e:
            self._safe_log_warn(f"Could not import offline plotter: {e}")
        except Exception as e:
            self._safe_log_warn(f"Failed to generate vehicle offline plots: {e}")

        try:
            from ..plotting.offline_cone_plotter import OfflineConePlotter
            cone_plotter = OfflineConePlotter(
                self._run_session.session_path,
                range_rmse_filename=self._cone_output_filename('cone_range_rmse_samples', 'csv'),
                output_suffix='',
            )
            cone_range_generated = cone_plotter.generate_range_rmse_plot()
            if cone_range_generated is not None:
                total += 1
                self._safe_log_info(f"Generated cone range RMSE offline plot: {cone_range_generated}")
            else:
                self._safe_log_warn("Cone range RMSE offline plot skipped: no cone range samples found")

            combined_generated = cone_plotter.generate_combined_range_rmse_plot(
                right_range_rmse_filename='cone_range_rmse_samples_lidar.csv',
            )
            if combined_generated is not None:
                total += 1
                self._safe_log_info(f"Generated combined camera/lidar RMSE offline plot: {combined_generated}")
        except ImportError as e:
            self._safe_log_warn(f"Could not import cone offline plotter: {e}")
            msg = str(e).lower()
            if 'numpy' in msg or 'multiarray' in msg or '_array_api' in msg:
                self._safe_log_warn(
                    "Detected NumPy/Matplotlib binary mismatch. "
                    "Reinstall compatible versions to enable cone range RMSE plot generation."
                )
        except Exception as e:
            self._safe_log_warn(f"Failed to generate cone offline plots: {e}")

        self._safe_log_info(f"Generated {total} plots in {self._run_session.plots_path}")

    def _cone_output_filename(self, stem: str, ext: str, suffix: str = '') -> str:
        clean_stem = stem.strip()
        clean_ext = ext.strip().lstrip('.')
        if suffix:
            return f'{clean_stem}_{suffix}.{clean_ext}'
        return f'{clean_stem}.{clean_ext}'

    @staticmethod
    def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

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
