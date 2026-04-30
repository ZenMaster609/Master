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
from rclpy.time import Time
from pathlib import Path
from typing import Dict, List, Optional
import csv
import math
import os
import signal
import threading
import time
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray
from eufs_msgs.msg import ConeArrayWithCovariance
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import VehicleState as VehicleStateMsg
from vehicle_plotter_msgs.msg import RunSession as RunSessionMsg
from std_msgs.msg import Float32, String
from sim_car.cones.tracking.pose import convert_odom_child_pose_to_base_frame
from sim_car.sensors.steering_convention import steering_joint_mean_to_deg

from ..core.vehicle_state import VehicleState
from ..core.run_session import RunSession
from ..core.qos_profiles import PLOTTER_QOS, RELIABLE_SENSOR_QOS
from ..utils.transforms import quaternion_to_yaw
from ..logging.log_writer import LogWriter
from ..logging.log_config import LogConfig
from ..logging.path_tracking_eval import (
    PATH_TRACKING_EVAL_FIELDNAMES,
    GateLapCounter,
    GTMidline,
    analyze_path_tracking_csv,
    build_smalltrack_lap_gate,
    build_gt_midline_from_cones,
    build_stitched_reference_trace,
    compare_planner_path_to_gt,
    nearest_point_on_polyline as eval_nearest_point_on_polyline,
    nearest_point_on_polyline_with_progress as eval_nearest_point_on_polyline_with_progress,
    path_cumulative_lengths,
    signed_cross_track_error as eval_signed_cross_track_error,
    should_assume_identity_transform,
    transform_xy_point,
    transform_xy_points,
    write_path_tracking_summary_files,
)
from ..logging.path_tracking_eval_plots import (
    _estimate_average_track_width_m,
    build_skidpad_gt_overlay_segments,
    compute_path_tracking_overlay_average_distances,
    compute_skidpad_circle_times_sec,
    generate_path_tracking_cte_plot,
    generate_path_tracking_overlay_plot,
)
from ..logging.steering_diagnostics import (
    CONE_AUDIT_DIAG_KEYS,
    PLANNER_DIAG_DEFAULTS,
    PLANNER_DIAG_TEXT_DEFAULTS,
    STEERING_DIAG_FIELDNAMES,
    analyze_csv,
    heading_error,
    nearest_point_on_polyline,
    parse_planner_diag,
    parse_planner_diag_text,
    signed_cross_track_error,
    write_summary_files,
)
from ..logging.stanley_debug_plots import generate_stanley_debug_plot
from ..logging.corridor_oscillation_analysis import (
    generate_corridor_oscillation_plot,
    analyze_corridor_oscillation,
    write_corridor_oscillation_summary_files,
)
from ..logging.thesis_controller_diagnostics import (
    THESIS_DIAG_FIELDNAMES,
    analyze_thesis_csv,
    build_thesis_sample_row,
    write_thesis_summary_files,
)
from ..logging.thesis_controller_plots import generate_thesis_controller_plot
from ..logging.track_metrics_report import write_track_metrics_report

_OFF_TRACK_NO_CONE_REASONS: frozenset[str] = frozenset({
    'no_safe_chain',
    'hold_expired_no_path',
    'holding_previous_valid',
    'hysteresis_holding',
    'stop_if_no_path',
})


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
        self.declare_parameter('controller_diagnostics_planner_diag_topic', '/midpoint_planner/diagnostics')
        self.declare_parameter('controller_diagnostics_physics_diag_topic', '/sim/physics_diagnostics')
        self.declare_parameter('controller_diagnostics_filename', 'steering_tracking_diagnostics.csv')
        self.declare_parameter('controller_diagnostics_summary_json', 'steering_tracking_summary.json')
        self.declare_parameter('controller_diagnostics_summary_txt', 'steering_tracking_summary.txt')
        self.declare_parameter('thesis_controller_diagnostics_filename', 'thesis_controller_diagnostics.csv')
        self.declare_parameter('thesis_controller_diagnostics_summary_json', 'thesis_controller_diagnostics_summary.json')
        self.declare_parameter('thesis_controller_diagnostics_summary_txt', 'thesis_controller_diagnostics_summary.txt')
        self.declare_parameter('path_tracking_eval_enabled', False)
        self.declare_parameter('path_tracking_eval_rate_hz', 20.0)
        self.declare_parameter('path_tracking_eval_gt_track_topic', '/ground_truth/track')
        self.declare_parameter('path_tracking_eval_odom_topic', '/sim/odom')
        self.declare_parameter('path_tracking_eval_planner_path_topic', '/planned_centerline')
        self.declare_parameter('path_tracking_eval_track_name', '')
        self.declare_parameter('path_tracking_eval_tf_timeout_sec', 0.05)
        self.declare_parameter('path_tracking_eval_filename', 'path_tracking_eval.csv')
        self.declare_parameter('path_tracking_eval_summary_json', 'path_tracking_eval_summary.json')
        self.declare_parameter('path_tracking_eval_summary_txt', 'path_tracking_eval_summary.txt')
        self.declare_parameter('path_tracking_eval_autostop_laps', 0)
        self.declare_parameter('control_reference_wheelbase_m', 1.65)
        self.declare_parameter('off_track_autostop_enabled', True)
        self.declare_parameter('off_track_autostop_timeout_s', 5.0)

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
        ).strip() or '/midpoint_planner/diagnostics'
        self._steering_diag_physics_diag_topic = str(
            self.get_parameter('controller_diagnostics_physics_diag_topic').value
        ).strip() or '/sim/physics_diagnostics'
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
        self._path_tracking_eval_enabled = bool(
            self.get_parameter('path_tracking_eval_enabled').value
        )
        self._path_tracking_eval_rate_hz = max(
            1.0,
            float(self.get_parameter('path_tracking_eval_rate_hz').value),
        )
        self._path_tracking_eval_gt_track_topic = str(
            self.get_parameter('path_tracking_eval_gt_track_topic').value
        ).strip() or '/ground_truth/track'
        self._path_tracking_eval_odom_topic = str(
            self.get_parameter('path_tracking_eval_odom_topic').value
        ).strip() or '/sim/odom'
        self._path_tracking_eval_planner_path_topic = str(
            self.get_parameter('path_tracking_eval_planner_path_topic').value
        ).strip() or '/planned_centerline'
        self._path_tracking_eval_track_name = str(
            self.get_parameter('path_tracking_eval_track_name').value
        ).strip().lower()
        self._path_tracking_eval_tf_timeout_sec = max(
            0.0,
            float(self.get_parameter('path_tracking_eval_tf_timeout_sec').value),
        )
        self._path_tracking_eval_filename = str(
            self.get_parameter('path_tracking_eval_filename').value
        ).strip() or 'path_tracking_eval.csv'
        self._path_tracking_eval_summary_json = str(
            self.get_parameter('path_tracking_eval_summary_json').value
        ).strip() or 'path_tracking_eval_summary.json'
        self._path_tracking_eval_summary_txt = str(
            self.get_parameter('path_tracking_eval_summary_txt').value
        ).strip() or 'path_tracking_eval_summary.txt'
        self._path_tracking_eval_autostop_laps = max(
            0,
            int(self.get_parameter('path_tracking_eval_autostop_laps').value),
        )
        self._off_track_autostop_enabled = bool(
            self.get_parameter('off_track_autostop_enabled').value
        )
        self._off_track_autostop_timeout_s = max(
            1.0,
            float(self.get_parameter('off_track_autostop_timeout_s').value),
        )
        self._control_reference_wheelbase_m = max(
            0.1,
            float(self.get_parameter('control_reference_wheelbase_m').value),
        )

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
        self._diag_planner_text_metrics: Dict[str, str] = dict(PLANNER_DIAG_TEXT_DEFAULTS)
        self._diag_file_handle = None
        self._diag_csv_writer: Optional[csv.DictWriter] = None
        self._thesis_diag_file_handle = None
        self._thesis_diag_csv_writer: Optional[csv.DictWriter] = None
        self._diag_flush_counter = 0
        self._diag_flush_stride = max(1, int(self._steering_diag_rate_hz * 2.0))
        self._diag_timer = None
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
        self._setup_steering_diag_subscriptions()
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
        self._diag_physics_sub = self.create_subscription(
            DiagnosticArray,
            self._steering_diag_physics_diag_topic,
            self._steering_diag_physics_callback,
            10,
        )
        self.get_logger().info(
            'Controller diagnostics subscriptions enabled: '
            f'cmd={self._steering_diag_cmd_topic} '
            f'steering={self._steering_diag_steering_topic} '
            f'joint_states={self._steering_diag_joint_states_topic} '
            f'odom={self._steering_diag_odom_topic} '
            f'path={self._steering_diag_path_topic} '
            f'planner_diag={self._steering_diag_planner_diag_topic} '
            f'physics_diag={self._steering_diag_physics_diag_topic} '
            f'oscillation={self._steering_diag_enabled} thesis={self._thesis_diag_enabled}'
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
            self._steering_diag_planner_diag_topic,
            self._off_track_planner_diag_callback,
            10,
        )
        self.get_logger().info(
            f'Off-track autostop enabled: timeout={self._off_track_autostop_timeout_s}s '
            f'topic={self._steering_diag_planner_diag_topic}'
        )

    def _off_track_planner_diag_callback(self, msg: DiagnosticArray) -> None:
        if self._off_track_autostop_triggered:
            return
        text_metrics = parse_planner_diag_text(msg)
        reason = text_metrics.get('operator_reason', '')
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
        avg_deg = steering_joint_mean_to_deg(left_rad, right_rad)
        self._diag_actual_steering_deg = avg_deg

    def _steering_diag_odom_callback(self, msg: Odometry) -> None:
        resolved_control_pose = self._resolve_control_pose_from_odom(
            msg,
            base_frame='front_axle',
        )
        if resolved_control_pose is None:
            self._diag_vehicle_x_m = float(msg.pose.pose.position.x)
            self._diag_vehicle_y_m = float(msg.pose.pose.position.y)
        else:
            self._diag_vehicle_x_m = float(resolved_control_pose[0])
            self._diag_vehicle_y_m = float(resolved_control_pose[1])
        q = msg.pose.pose.orientation
        self._diag_vehicle_yaw_rad = quaternion_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
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
        self._merge_diag_metrics(parse_planner_diag(msg))
        self._merge_diag_text_metrics(parse_planner_diag_text(msg))

    def _steering_diag_physics_callback(self, msg: DiagnosticArray) -> None:
        self._merge_diag_metrics(parse_planner_diag(msg))

    def _path_tracking_eval_gt_callback(self, msg: ConeArrayWithCovariance) -> None:
        self._path_eval_latest_gt_msg = msg
        self._path_tracking_eval_rebuild_gt_midline()

    def _path_tracking_eval_odom_callback(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        self._path_eval_vehicle_xy = np.asarray([x, y], dtype=np.float64)
        self._path_eval_vehicle_frame = str(msg.header.frame_id).strip() or 'odom'
        self._path_eval_vehicle_child_frame = str(msg.child_frame_id).strip()
        self._path_eval_vehicle_stamp = msg.header.stamp

        q = msg.pose.pose.orientation
        yaw = quaternion_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
        self._path_eval_vehicle_yaw_rad = yaw
        heading_xy = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        if self._path_eval_start_xy is None:
            self._path_eval_start_xy = np.asarray([x, y], dtype=np.float64)
            self._path_eval_start_heading_xy = heading_xy
            self._path_tracking_eval_rebuild_gt_midline()
        self._path_tracking_eval_update_smalltrack_laps(msg.header.stamp)

    def _path_tracking_eval_path_callback(self, msg: NavPath) -> None:
        if not msg.poses:
            self._path_eval_planner_xy = np.empty((0, 2), dtype=np.float64)
            self._path_eval_planner_frame = str(msg.header.frame_id).strip() or self._path_eval_planner_frame
            self._path_eval_planner_stamp = msg.header.stamp
            return
        points = np.empty((len(msg.poses), 2), dtype=np.float64)
        for idx, pose_stamped in enumerate(msg.poses):
            points[idx, 0] = float(pose_stamped.pose.position.x)
            points[idx, 1] = float(pose_stamped.pose.position.y)
        self._path_eval_planner_xy = points
        self._path_eval_planner_frame = str(msg.header.frame_id).strip() or 'odom'
        self._path_eval_planner_stamp = msg.header.stamp

    def _path_tracking_eval_rebuild_gt_midline(self) -> None:
        if self._path_eval_latest_gt_msg is None:
            return
        if self._path_eval_start_xy is None or self._path_eval_start_heading_xy is None:
            return

        blue_xy = []
        yellow_xy = []
        for cone in self._path_eval_latest_gt_msg.blue_cones:
            blue_xy.append((float(cone.point.x), float(cone.point.y)))
        for cone in self._path_eval_latest_gt_msg.yellow_cones:
            yellow_xy.append((float(cone.point.x), float(cone.point.y)))

        frame_id = str(self._path_eval_latest_gt_msg.header.frame_id).strip() or 'map'
        self._path_eval_gt_midline_source = build_gt_midline_from_cones(
            blue_xy=np.asarray(blue_xy, dtype=np.float64),
            yellow_xy=np.asarray(yellow_xy, dtype=np.float64),
            start_xy=self._path_eval_start_xy,
            heading_xy=self._path_eval_start_heading_xy,
            frame_id=frame_id,
            resolution_m=0.5,
        )
        big_orange_xy = [
            (float(cone.point.x), float(cone.point.y))
            for cone in self._path_eval_latest_gt_msg.big_orange_cones
        ]
        self._path_eval_smalltrack_gate_source = build_smalltrack_lap_gate(
            big_orange_xy=np.asarray(big_orange_xy, dtype=np.float64),
            frame_id=frame_id,
        )
        self._path_tracking_eval_maybe_init_smalltrack_lap_counter()

    def _path_tracking_eval_maybe_init_smalltrack_lap_counter(self) -> None:
        if self._path_tracking_eval_track_name != 'smalltrack':
            return
        if self._path_eval_smalltrack_lap_counter is not None:
            return
        if self._path_eval_smalltrack_gate_source is None:
            return
        if self._path_eval_gt_midline_source is None or self._path_eval_gt_midline_source.midline_xy.shape[0] < 2:
            return

        track_length_m = float(
            path_cumulative_lengths(self._path_eval_gt_midline_source.midline_xy)[-1]
        )
        min_lap_travel_m = max(15.0, 0.6 * track_length_m) if math.isfinite(track_length_m) else 25.0
        gate_length_m = float(
            np.hypot(
                *(
                    self._path_eval_smalltrack_gate_source.segment_xy[1]
                    - self._path_eval_smalltrack_gate_source.segment_xy[0]
                )
            )
        )
        self._path_eval_smalltrack_lap_counter = GateLapCounter(
            self._path_eval_smalltrack_gate_source.segment_xy,
            min_lap_travel_m=min_lap_travel_m,
            min_lap_time_sec=5.0,
            near_gate_distance_m=max(4.0, 2.0 * gate_length_m),
        )
        self.get_logger().info(
            'Smalltrack lap counting enabled: '
            f'min_lap_travel_m={min_lap_travel_m:.1f} '
            f'autostop_laps={self._path_tracking_eval_autostop_laps}'
        )

    def _path_tracking_eval_update_smalltrack_laps(self, stamp) -> None:
        if self._path_tracking_eval_track_name != 'smalltrack':
            return
        self._path_tracking_eval_maybe_init_smalltrack_lap_counter()
        if self._path_eval_smalltrack_lap_counter is None or self._path_eval_smalltrack_gate_source is None:
            return
        if not np.all(np.isfinite(self._path_eval_vehicle_xy)) or not self._path_eval_vehicle_frame:
            return

        vehicle_xy_gate = self._path_eval_transform_point_to_frame(
            self._path_eval_vehicle_xy,
            source_frame=self._path_eval_vehicle_frame,
            target_frame=self._path_eval_smalltrack_gate_source.frame_id,
            stamp=stamp,
        )
        if vehicle_xy_gate is None:
            return

        timestamp_sec = float(getattr(stamp, 'sec', 0)) + (float(getattr(stamp, 'nanosec', 0)) * 1e-9)
        if timestamp_sec <= 0.0:
            timestamp_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        snapshot = self._path_eval_smalltrack_lap_counter.update(vehicle_xy_gate, timestamp_sec)
        self._path_eval_smalltrack_completed_laps = int(snapshot.completed_laps)
        if snapshot.just_completed_lap:
            if snapshot.last_lap_time_sec is not None:
                self._path_eval_smalltrack_lap_times_sec.append(float(snapshot.last_lap_time_sec))
            self.get_logger().info(
                f'Smalltrack lap completed: laps={snapshot.completed_laps} crossings={snapshot.gate_crossings}'
            )

        if (
            not self._path_eval_smalltrack_autostop_triggered
            and self._path_tracking_eval_autostop_laps > 0
            and snapshot.completed_laps >= self._path_tracking_eval_autostop_laps
        ):
            self._path_eval_smalltrack_autostop_triggered = True
            self.get_logger().info(
                'Smalltrack lap target reached: '
                f'{snapshot.completed_laps}/{self._path_tracking_eval_autostop_laps}. '
                'Shutting down logger to finalize outputs.'
            )
            self.shutdown()
            self._request_process_exit(parent_delay_s=0.1, force_delay_s=5.0)
            if rclpy.ok():
                rclpy.shutdown()

    def _compute_average_lap_time_sec(self) -> Optional[float]:
        if self._path_tracking_eval_track_name == 'smalltrack':
            if self._path_eval_smalltrack_lap_times_sec:
                return float(np.mean(self._path_eval_smalltrack_lap_times_sec))
            return None
        if (
            self._path_eval_run_start_sec is not None
            and self._path_eval_run_last_sec is not None
            and self._path_eval_run_last_sec > self._path_eval_run_start_sec
        ):
            return self._path_eval_run_last_sec - self._path_eval_run_start_sec
        return None

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

    def _merge_diag_metrics(self, values: Dict[str, float]) -> None:
        for key, value in values.items():
            if math.isfinite(float(value)):
                self._diag_planner_metrics[key] = float(value)

    def _merge_diag_text_metrics(self, values: Dict[str, str]) -> None:
        for key, value in values.items():
            self._diag_planner_text_metrics[key] = str(value).strip()

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
        self._initialize_steering_diag_output()
        self._initialize_path_tracking_eval_output()

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
            self._diag_csv_writer = csv.DictWriter(
                self._diag_file_handle, fieldnames=list(STEERING_DIAG_FIELDNAMES)
            )
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

    def _initialize_path_tracking_eval_output(self) -> None:
        if not self._path_tracking_eval_enabled or self._run_session is None:
            return
        eval_path = self._run_session.logs_path / self._path_tracking_eval_filename
        self._path_eval_file_handle = open(eval_path, 'w', newline='', encoding='utf-8')
        self._path_eval_csv_writer = csv.DictWriter(
            self._path_eval_file_handle,
            fieldnames=PATH_TRACKING_EVAL_FIELDNAMES,
        )
        self._path_eval_csv_writer.writeheader()
        self._path_eval_flush_counter = 0
        self._path_eval_timer = self.create_timer(
            1.0 / self._path_tracking_eval_rate_hz,
            self._path_tracking_eval_sample,
        )
        self.get_logger().info(f'Path tracking evaluation CSV: {eval_path}')

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
        planner_churn = float(
            self._diag_planner_metrics.get(
                'selected_chain_churn_ratio',
                self._diag_planner_metrics.get('selected_edge_churn_ratio', float('nan')),
            )
        )
        planner_tracked_delta = float(self._diag_planner_metrics.get('tracked_cones_frame_delta_p95_m', float('nan')))
        planner_state_code = float(self._diag_planner_metrics.get('planner_state_code', float('nan')))
        planner_fresh_publish_flag = float(self._diag_planner_metrics.get('fresh_publish_flag', float('nan')))
        planner_held_publish_flag = float(self._diag_planner_metrics.get('held_publish_flag', float('nan')))
        planner_stopped_flag = float(self._diag_planner_metrics.get('stopped_flag', float('nan')))
        planner_waiting_flag = float(self._diag_planner_metrics.get('waiting_flag', float('nan')))
        planner_operator_reason_code = float(self._diag_planner_metrics.get('operator_reason_code', float('nan')))
        planner_hold_remaining_s = float(self._diag_planner_metrics.get('hold_remaining_s', float('nan')))
        planner_control_path_point_count = float(
            self._diag_planner_metrics.get('control_path_point_count', float('nan'))
        )
        planner_zero_cmd_sent_flag = float(self._diag_planner_metrics.get('zero_cmd_sent_flag', float('nan')))
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
            'physics_state_vx_mps': float(self._diag_planner_metrics.get('state_vx_mps', float('nan'))),
            'physics_state_vy_mps': float(self._diag_planner_metrics.get('state_vy_mps', float('nan'))),
            'physics_state_speed_mps': float(
                self._diag_planner_metrics.get('state_speed_mps', float('nan'))
            ),
            'physics_state_yaw_rad': float(self._diag_planner_metrics.get('state_yaw_rad', float('nan'))),
            'physics_state_yaw_rate_rps': float(
                self._diag_planner_metrics.get('state_yaw_rate_rps', float('nan'))
            ),
            'physics_state_ax_mps2': float(self._diag_planner_metrics.get('state_ax_mps2', float('nan'))),
            'physics_state_ay_mps2': float(self._diag_planner_metrics.get('state_ay_mps2', float('nan'))),
            'physics_desired_accel_mps2': float(
                self._diag_planner_metrics.get('desired_accel_mps2', float('nan'))
            ),
            'physics_actual_accel_mps2': float(
                self._diag_planner_metrics.get('actual_accel_mps2', float('nan'))
            ),
            'physics_desired_steering_rad': float(
                self._diag_planner_metrics.get('desired_steering_rad', float('nan'))
            ),
            'physics_actual_steering_rad': float(
                self._diag_planner_metrics.get('actual_steering_rad', float('nan'))
            ),
            'physics_steering_tracking_error_rad': float(
                self._diag_planner_metrics.get('steering_tracking_error_rad', float('nan'))
            ),
            'physics_sideslip_rad': float(self._diag_planner_metrics.get('sideslip_rad', float('nan'))),
            'physics_slip_angle_front_rad': float(
                self._diag_planner_metrics.get('slip_angle_front_rad', float('nan'))
            ),
            'physics_slip_angle_rear_rad': float(
                self._diag_planner_metrics.get('slip_angle_rear_rad', float('nan'))
            ),
            'physics_kinematic_blend': float(
                self._diag_planner_metrics.get('kinematic_blend', float('nan'))
            ),
            'physics_kinematic_vy_ref_mps': float(
                self._diag_planner_metrics.get('kinematic_vy_ref_mps', float('nan'))
            ),
            'physics_kinematic_yaw_rate_ref_rps': float(
                self._diag_planner_metrics.get('kinematic_yaw_rate_ref_rps', float('nan'))
            ),
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
            'planner_selected_chain_churn_ratio': planner_churn,
            'planner_tracked_cones_frame_delta_p95_m': planner_tracked_delta,
            'planner_state_code': planner_state_code,
            'planner_fresh_publish_flag': planner_fresh_publish_flag,
            'planner_held_publish_flag': planner_held_publish_flag,
            'planner_stopped_flag': planner_stopped_flag,
            'planner_waiting_flag': planner_waiting_flag,
            'planner_operator_reason_code': planner_operator_reason_code,
            'planner_hold_remaining_s': planner_hold_remaining_s,
            'planner_control_path_point_count': planner_control_path_point_count,
            'planner_zero_cmd_sent_flag': planner_zero_cmd_sent_flag,
        }
        for key in CONE_AUDIT_DIAG_KEYS:
            row[f'planner_{key}'] = float(self._diag_planner_metrics.get(key, float('nan')))
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
                planner_text_metrics=self._diag_planner_text_metrics,
            )
            self._thesis_diag_csv_writer.writerow(thesis_row)
        self._diag_flush_counter += 1
        if self._diag_flush_counter >= self._diag_flush_stride:
            if self._diag_file_handle is not None:
                self._diag_file_handle.flush()
            if self._thesis_diag_file_handle is not None:
                self._thesis_diag_file_handle.flush()
            self._diag_flush_counter = 0

    def _path_tracking_eval_sample(self) -> None:
        if self._path_eval_csv_writer is None or self._path_eval_file_handle is None:
            return

        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        row = {
            'timestamp_sec': now_sec,
            'sample_valid_flag': 0.0,
            'status': 'waiting_for_inputs',
            'frame_id': '',
            'gt_source_frame': '',
            'odom_child_frame_id': '',
            'resolved_control_frame': '',
            'vehicle_x_m': float('nan'),
            'vehicle_y_m': float('nan'),
            'body_center_x_m': float('nan'),
            'body_center_y_m': float('nan'),
            'front_axle_x_m': float('nan'),
            'front_axle_y_m': float('nan'),
            'planner_reference_x_m': float('nan'),
            'planner_reference_y_m': float('nan'),
            'planner_reference_s_m': float('nan'),
            'gt_reference_x_m': float('nan'),
            'gt_reference_y_m': float('nan'),
            'gt_reference_s_m': float('nan'),
            'planner_reference_vs_gt_cte_m': float('nan'),
            'body_center_vs_planner_cte_m': float('nan'),
            'front_axle_vs_planner_cte_m': float('nan'),
            'controller_vs_planner_cte_m': float('nan'),
            'body_center_vs_gt_cte_m': float('nan'),
            'front_axle_vs_gt_cte_m': float('nan'),
            'controller_vs_gt_cte_m': float('nan'),
            'planner_vs_gt_cte_rms_m': float('nan'),
            'planner_vs_gt_cte_p95_m': float('nan'),
            'planner_vs_gt_cte_max_m': float('nan'),
        }

        if not np.all(np.isfinite(self._path_eval_vehicle_xy)) or not self._path_eval_vehicle_frame:
            row['status'] = 'waiting_for_odom'
            self._path_eval_csv_writer.writerow(row)
            return

        vehicle_xy = np.asarray(self._path_eval_vehicle_xy, dtype=np.float64)
        vehicle_frame = str(self._path_eval_vehicle_frame).strip() or 'odom'
        odom_child_frame = str(self._path_eval_vehicle_child_frame).strip()
        row['odom_child_frame_id'] = odom_child_frame

        if self._path_eval_gt_midline_source is None or self._path_eval_gt_midline_source.midline_xy.shape[0] < 2:
            row['status'] = 'waiting_for_gt_track'
            self._path_eval_csv_writer.writerow(row)
            return

        gt_source = self._path_eval_gt_midline_source
        row['gt_source_frame'] = gt_source.frame_id

        planner_available = self._path_eval_planner_xy.shape[0] >= 2 and bool(self._path_eval_planner_frame)
        target_frame = self._path_eval_planner_frame if planner_available else vehicle_frame
        row['frame_id'] = target_frame

        body_center_xy_target = self._path_eval_transform_point_to_frame(
            vehicle_xy,
            source_frame=vehicle_frame,
            target_frame=target_frame,
            stamp=self._path_eval_vehicle_stamp,
        )
        if body_center_xy_target is None:
            row['status'] = 'frame_transform_unavailable'
            self._path_eval_csv_writer.writerow(row)
            return
        row['vehicle_x_m'] = float(body_center_xy_target[0])
        row['vehicle_y_m'] = float(body_center_xy_target[1])
        row['body_center_x_m'] = float(body_center_xy_target[0])
        row['body_center_y_m'] = float(body_center_xy_target[1])

        front_axle_xy_target = self._path_eval_resolve_control_point_to_frame(
            point_xy=vehicle_xy,
            source_frame=vehicle_frame,
            child_frame=odom_child_frame,
            yaw_rad=self._path_eval_vehicle_yaw_rad,
            target_frame=target_frame,
            stamp=self._path_eval_vehicle_stamp,
            base_frame='front_axle',
        )
        if front_axle_xy_target is not None:
            row['resolved_control_frame'] = 'front_axle'
            row['front_axle_x_m'] = float(front_axle_xy_target[0])
            row['front_axle_y_m'] = float(front_axle_xy_target[1])

        gt_midline_target = self._path_eval_transform_path_to_frame(
            gt_source.midline_xy,
            source_frame=gt_source.frame_id,
            target_frame=target_frame,
            stamp=self._path_eval_planner_stamp if planner_available else self._path_eval_vehicle_stamp,
        )
        if gt_midline_target is None or gt_midline_target.shape[0] < 2:
            row['status'] = 'frame_transform_unavailable'
            self._path_eval_csv_writer.writerow(row)
            return
        gt_left_target = self._path_eval_transform_path_to_frame(
            gt_source.left_xy,
            source_frame=gt_source.frame_id,
            target_frame=target_frame,
            stamp=self._path_eval_planner_stamp if planner_available else self._path_eval_vehicle_stamp,
        )
        gt_right_target = self._path_eval_transform_path_to_frame(
            gt_source.right_xy,
            source_frame=gt_source.frame_id,
            target_frame=target_frame,
            stamp=self._path_eval_planner_stamp if planner_available else self._path_eval_vehicle_stamp,
        )

        self._path_eval_last_gt_midline_xy = gt_midline_target
        self._path_eval_last_gt_left_xy = (
            np.asarray(gt_left_target, dtype=np.float64)
            if gt_left_target is not None
            else np.empty((0, 2), dtype=np.float64)
        )
        self._path_eval_last_gt_right_xy = (
            np.asarray(gt_right_target, dtype=np.float64)
            if gt_right_target is not None
            else np.empty((0, 2), dtype=np.float64)
        )
        self._path_eval_last_target_frame = target_frame

        gt_nearest_idx, gt_nearest_point, gt_nearest_progress_m = eval_nearest_point_on_polyline_with_progress(
            float(body_center_xy_target[0]),
            float(body_center_xy_target[1]),
            gt_midline_target,
        )
        if gt_nearest_idx >= 0:
            body_center_vs_gt_cte, _ = eval_signed_cross_track_error(
                float(body_center_xy_target[0]),
                float(body_center_xy_target[1]),
                gt_midline_target,
            )
            row['body_center_vs_gt_cte_m'] = body_center_vs_gt_cte
            row['controller_vs_gt_cte_m'] = body_center_vs_gt_cte
            row['gt_reference_x_m'] = float(gt_nearest_point[0])
            row['gt_reference_y_m'] = float(gt_nearest_point[1])
            row['gt_reference_s_m'] = float(gt_nearest_progress_m)
        if front_axle_xy_target is not None:
            front_axle_vs_gt_cte, _ = eval_signed_cross_track_error(
                float(front_axle_xy_target[0]),
                float(front_axle_xy_target[1]),
                gt_midline_target,
            )
            row['front_axle_vs_gt_cte_m'] = front_axle_vs_gt_cte

        if planner_available:
            body_center_planner_nearest_idx, body_center_planner_nearest_point = eval_nearest_point_on_polyline(
                float(body_center_xy_target[0]),
                float(body_center_xy_target[1]),
                self._path_eval_planner_xy,
            )
            if body_center_planner_nearest_idx >= 0:
                body_center_vs_planner_cte, _ = eval_signed_cross_track_error(
                    float(body_center_xy_target[0]),
                    float(body_center_xy_target[1]),
                    self._path_eval_planner_xy,
                )
                row['body_center_vs_planner_cte_m'] = body_center_vs_planner_cte
                row['controller_vs_planner_cte_m'] = body_center_vs_planner_cte
            planner_reference_point = (
                np.asarray(body_center_planner_nearest_point, dtype=np.float64)
                if body_center_planner_nearest_idx >= 0
                else None
            )
            planner_reference_progress_m = float('nan')
            if body_center_planner_nearest_idx >= 0:
                _idx, _point, planner_reference_progress_m = eval_nearest_point_on_polyline_with_progress(
                    float(body_center_xy_target[0]),
                    float(body_center_xy_target[1]),
                    self._path_eval_planner_xy,
                )
            if front_axle_xy_target is not None:
                front_axle_vs_planner_cte, _ = eval_signed_cross_track_error(
                    float(front_axle_xy_target[0]),
                    float(front_axle_xy_target[1]),
                    self._path_eval_planner_xy,
                )
                row['front_axle_vs_planner_cte_m'] = front_axle_vs_planner_cte
                (
                    front_axle_planner_nearest_idx,
                    front_axle_planner_nearest_point,
                    front_axle_planner_progress_m,
                ) = eval_nearest_point_on_polyline_with_progress(
                    float(front_axle_xy_target[0]),
                    float(front_axle_xy_target[1]),
                    self._path_eval_planner_xy,
                )
                if front_axle_planner_nearest_idx >= 0:
                    planner_reference_point = np.asarray(front_axle_planner_nearest_point, dtype=np.float64)
                    planner_reference_progress_m = float(front_axle_planner_progress_m)

            if planner_reference_point is not None:
                row['planner_reference_x_m'] = float(planner_reference_point[0])
                row['planner_reference_y_m'] = float(planner_reference_point[1])
                row['planner_reference_s_m'] = float(planner_reference_progress_m)
                self._path_eval_reference_trace_points.append(np.asarray(planner_reference_point, dtype=np.float64))
                planner_reference_vs_gt_cte, _ = eval_signed_cross_track_error(
                    float(planner_reference_point[0]),
                    float(planner_reference_point[1]),
                    gt_midline_target,
                )
                row['planner_reference_vs_gt_cte_m'] = planner_reference_vs_gt_cte

            planner_metrics = compare_planner_path_to_gt(
                planner_xy=self._path_eval_planner_xy,
                gt_midline_xy=gt_midline_target,
                vehicle_xy=body_center_xy_target,
                resolution_m=0.5,
            )
            row['planner_vs_gt_cte_rms_m'] = float(planner_metrics.get('planner_vs_gt_cte_rms_m', float('nan')))
            row['planner_vs_gt_cte_p95_m'] = float(planner_metrics.get('planner_vs_gt_cte_p95_m', float('nan')))
            row['planner_vs_gt_cte_max_m'] = float(planner_metrics.get('planner_vs_gt_cte_max_m', float('nan')))
            if (
                math.isfinite(row['front_axle_vs_planner_cte_m'])
                and math.isfinite(row['front_axle_vs_gt_cte_m'])
                and (
                    math.isfinite(row['planner_reference_vs_gt_cte_m'])
                    or math.isfinite(row['planner_vs_gt_cte_rms_m'])
                )
            ):
                row['sample_valid_flag'] = 1.0
                row['status'] = 'ok'
            else:
                row['status'] = 'partial_metrics'
        else:
            row['status'] = 'waiting_for_planner_path'

        if self._path_eval_run_start_sec is None:
            self._path_eval_run_start_sec = now_sec
        self._path_eval_run_last_sec = now_sec
        self._path_eval_csv_writer.writerow(row)
        self._path_eval_flush_counter += 1
        if self._path_eval_flush_counter >= self._path_eval_flush_stride:
            self._path_eval_file_handle.flush()
            self._path_eval_flush_counter = 0

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
                self._finalize_steering_diag_outputs()
                self._finalize_path_tracking_eval_outputs()
                self._finalize_corridor_oscillation_outputs()

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

    def _finalize_path_tracking_eval_outputs(self) -> None:
        if not self._path_tracking_eval_enabled or self._run_session is None:
            return
        if self._path_eval_timer is not None:
            self._path_eval_timer.cancel()
            self._path_eval_timer = None
        if self._path_eval_file_handle is not None:
            self._path_eval_file_handle.flush()
            self._path_eval_file_handle.close()
            self._path_eval_file_handle = None
        self._path_eval_csv_writer = None

        csv_path = self._run_session.logs_path / self._path_tracking_eval_filename
        summary_json = self._run_session.logs_path / self._path_tracking_eval_summary_json
        summary_txt = self._run_session.logs_path / self._path_tracking_eval_summary_txt
        try:
            summary = analyze_path_tracking_csv(csv_path)
            write_path_tracking_summary_files(summary, summary_json, summary_txt)
            self._safe_log_info(
                'Path tracking evaluation summary: '
                f"planner_vs_gt_rms={summary.get('planner_vs_gt_cte_rms_m', float('nan')):.4f} m "
                f"front_axle_vs_planner_rms={summary.get('front_axle_vs_planner_cte_rms_m', float('nan')):.4f} m "
                f"front_axle_vs_gt_rms={summary.get('front_axle_vs_gt_cte_rms_m', float('nan')):.4f} m"
            )
        except Exception as exc:
            self._safe_log_warn(f'Failed path tracking evaluation analysis: {exc}')

        planner_trace_xy = build_stitched_reference_trace(
            np.asarray(self._path_eval_reference_trace_points, dtype=np.float64),
            min_spacing_m=0.1,
        )
        overlay_average_distances: Dict[str, float] = {}
        overlay_track_width_m = float('nan')
        try:
            cte_plot_path = self._run_session.plots_path / 'path_tracking_eval_cte.pdf'
            generated_cte = generate_path_tracking_cte_plot(csv_path, cte_plot_path)
            if generated_cte is not None:
                self._safe_log_info(f'Generated path tracking CTE plot: {generated_cte}')
            else:
                self._safe_log_warn('Path tracking CTE plot skipped: no path tracking evaluation rows')
        except Exception as exc:
            self._safe_log_warn(f'Failed path tracking CTE plot generation: {exc}')

        try:
            overlay_path = self._run_session.plots_path / 'path_tracking_eval_overlay.pdf'
            overlay_segments_xy = None
            overlay_midline_xy = self._path_eval_last_gt_midline_xy
            overlay_blue_xy = self._path_eval_last_gt_left_xy
            overlay_yellow_xy = self._path_eval_last_gt_right_xy
            if self._path_tracking_eval_track_name == 'skidpad':
                overlay_segments_xy = build_skidpad_gt_overlay_segments()
                overlay_midline_xy = np.empty((0, 2), dtype=np.float64)
                overlay_blue_xy = np.empty((0, 2), dtype=np.float64)
                overlay_yellow_xy = np.empty((0, 2), dtype=np.float64)
                target_frame = str(self._path_eval_last_target_frame).strip()
                stamp = self._path_eval_planner_stamp if self._path_eval_planner_stamp is not None else self._path_eval_vehicle_stamp
                if target_frame and stamp is not None:
                    transformed_segments: list[np.ndarray] = []
                    for segment_xy in overlay_segments_xy:
                        transformed_segment = self._path_eval_transform_path_to_frame(
                            segment_xy,
                            source_frame='map',
                            target_frame=target_frame,
                            stamp=stamp,
                        )
                        transformed_segments.append(
                            np.asarray(transformed_segment, dtype=np.float64)
                            if transformed_segment is not None
                            else np.asarray(segment_xy, dtype=np.float64)
                        )
                    overlay_segments_xy = transformed_segments
            overlay_average_distances = compute_path_tracking_overlay_average_distances(
                csv_path,
                gt_midline_xy=overlay_midline_xy,
                planner_trace_xy=planner_trace_xy,
                gt_reference_segments_xy=overlay_segments_xy,
            )
            overlay_track_width_m = _estimate_average_track_width_m(overlay_blue_xy, overlay_yellow_xy)
            skidpad_circle_times = (
                compute_skidpad_circle_times_sec(csv_path)
                if self._path_tracking_eval_track_name == 'skidpad'
                else None
            )
            generated_overlay = generate_path_tracking_overlay_plot(
                csv_path,
                overlay_path,
                gt_midline_xy=overlay_midline_xy,
                gt_left_xy=overlay_blue_xy,
                gt_right_xy=overlay_yellow_xy,
                planner_trace_xy=planner_trace_xy,
                gt_overlay_segments_xy=overlay_segments_xy,
                lap_count=self._path_eval_smalltrack_completed_laps
                if self._path_tracking_eval_track_name == 'smalltrack'
                else None,
                lap_target=self._path_tracking_eval_autostop_laps
                if self._path_tracking_eval_track_name == 'smalltrack'
                and self._path_tracking_eval_autostop_laps > 0
                else None,
                average_lap_time_sec=self._compute_average_lap_time_sec(),
                skidpad_circle_times_sec=skidpad_circle_times,
            )
            if generated_overlay is not None:
                self._safe_log_info(f'Generated path tracking overlay plot: {generated_overlay}')
            else:
                self._safe_log_warn('Path tracking overlay plot skipped: GT midline unavailable')
        except Exception as exc:
            self._safe_log_warn(f'Failed path tracking overlay plot generation: {exc}')

        try:
            completed_laps = (
                self._path_eval_smalltrack_completed_laps
                if self._path_tracking_eval_track_name == 'smalltrack'
                else None
            )
            csv_report_path, jsonl_report_path = write_track_metrics_report(
                session_path=self._run_session.session_path,
                base_path=self._run_session.base_path,
                run_id=self._run_session.run_id,
                completed_laps=completed_laps,
                lap_target=self._path_tracking_eval_autostop_laps,
                overlay_average_distances=overlay_average_distances,
                avg_track_width_m=overlay_track_width_m,
                avg_lap_time_sec=self._compute_average_lap_time_sec(),
            )
            self._safe_log_info(
                f'Updated per-track path tracking metrics: {csv_report_path}, {jsonl_report_path}'
            )
        except Exception as exc:
            self._safe_log_warn(f'Failed per-track path tracking metrics report update: {exc}')

    def _finalize_corridor_oscillation_outputs(self) -> None:
        if self._run_session is None or not self._thesis_diag_enabled:
            return

        thesis_csv_path = self._run_session.logs_path / self._thesis_diag_filename
        if not thesis_csv_path.exists():
            return

        path_tracking_csv_path = None
        if self._path_tracking_eval_enabled:
            candidate = self._run_session.logs_path / self._path_tracking_eval_filename
            if candidate.exists():
                path_tracking_csv_path = candidate

        try:
            summary = analyze_corridor_oscillation(thesis_csv_path, path_tracking_csv_path)
        except Exception as exc:
            self._safe_log_warn(f'Failed corridor oscillation analysis: {exc}')
            return

        if summary.get('sample_count', 0.0) <= 0.0:
            return
        if summary.get('profile_row_count', 0.0) <= 0.0:
            self._safe_log_warn('Corridor oscillation analysis skipped: no corridor profile samples found')
            return

        summary_json = self._run_session.logs_path / 'corridor_oscillation_summary.json'
        summary_txt = self._run_session.logs_path / 'corridor_oscillation_summary.txt'
        try:
            write_corridor_oscillation_summary_files(summary, summary_json, summary_txt)
            self._safe_log_info(
                'Corridor oscillation summary: '
                f"planner_ref_vs_gt_rms={summary.get('planner_reference_vs_gt_cte_rms_m', float('nan')):.4f} m "
                f"buffer_delta_p95={summary.get('buffer_vs_prevalidation_abs_p95_m', float('nan')):.4f} m "
                f"anchor_delta_p95={summary.get('control_vs_buffer_abs_p95_m', float('nan')):.4f} m"
            )
        except Exception as exc:
            self._safe_log_warn(f'Failed corridor oscillation summary write: {exc}')

        try:
            plot_path = self._run_session.plots_path / 'corridor_oscillation_analysis.png'
            generated_path = generate_corridor_oscillation_plot(
                thesis_csv_path,
                path_tracking_csv_path,
                plot_path,
            )
            if generated_path is not None:
                self._safe_log_info(f'Generated corridor oscillation plot: {generated_path}')
            else:
                self._safe_log_warn('Corridor oscillation plot skipped: no thesis diagnostics rows')
        except Exception as exc:
            self._safe_log_warn(f'Failed corridor oscillation plot generation: {exc}')

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

    def _generate_offline_plots(self) -> None:
        """Generate offline plots from logged data."""
        total = 0
        self._safe_log_info("Generating offline plots...")

        try:
            from ..plotting.offline_plotter import OfflinePlotter
            plotter = OfflinePlotter(self._run_session.session_path, output_format='pdf')
            generated = plotter.generate_plots()
            total += len(generated)
        except ImportError as e:
            self._safe_log_warn(f"Could not import offline plotter: {e}")
        except Exception as e:
            self._safe_log_warn(f"Failed to generate vehicle offline plots: {e}")

        try:
            from ..plotting.offline_cone_plotter import OfflineConePlotter
            cone_plotter = OfflineConePlotter(self._run_session.session_path, output_format='pdf')
            generated_paths = cone_plotter.generate_all_range_rmse_plots(delete_legacy_png=True)
            total += len(generated_paths)
            for output_path in generated_paths:
                self._safe_log_info(f"Generated cone range RMSE offline plot: {output_path}")
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

    def _path_eval_lookup_transform(self, *, target_frame: str, source_frame: str, stamp):
        if self._path_eval_tf_buffer is None:
            return None
        timeout = Duration(seconds=float(self._path_tracking_eval_tf_timeout_sec))
        try:
            stamp_time = Time.from_msg(stamp) if stamp is not None else Time()
            return self._path_eval_tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp_time,
                timeout=timeout,
            )
        except Exception:
            pass
        try:
            return self._path_eval_tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=timeout,
            )
        except Exception:
            return None

    def _path_eval_transform_path_to_frame(
        self,
        path_xy: np.ndarray,
        *,
        source_frame: str,
        target_frame: str,
        stamp,
    ) -> Optional[np.ndarray]:
        source_frame = str(source_frame).strip()
        target_frame = str(target_frame).strip()
        if source_frame == target_frame:
            return np.asarray(path_xy, dtype=np.float64)
        if should_assume_identity_transform(source_frame, target_frame):
            self._path_eval_warn_identity_transform(source_frame, target_frame)
            return np.asarray(path_xy, dtype=np.float64)
        transform = self._path_eval_lookup_transform(
            target_frame=target_frame,
            source_frame=source_frame,
            stamp=stamp,
        )
        if transform is None:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = quaternion_to_yaw(
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        )
        return transform_xy_points(
            np.asarray(path_xy, dtype=np.float64),
            translation_xy=np.asarray([float(translation.x), float(translation.y)], dtype=np.float64),
            yaw_rad=yaw,
        )

    def _path_eval_transform_point_to_frame(
        self,
        point_xy: np.ndarray,
        *,
        source_frame: str,
        target_frame: str,
        stamp,
    ) -> Optional[np.ndarray]:
        source_frame = str(source_frame).strip()
        target_frame = str(target_frame).strip()
        if source_frame == target_frame:
            return np.asarray(point_xy, dtype=np.float64)
        if should_assume_identity_transform(source_frame, target_frame):
            self._path_eval_warn_identity_transform(source_frame, target_frame)
            return np.asarray(point_xy, dtype=np.float64)
        transform = self._path_eval_lookup_transform(
            target_frame=target_frame,
            source_frame=source_frame,
            stamp=stamp,
        )
        if transform is None:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = quaternion_to_yaw(
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        )
        return transform_xy_point(
            np.asarray(point_xy, dtype=np.float64),
            translation_xy=np.asarray([float(translation.x), float(translation.y)], dtype=np.float64),
            yaw_rad=yaw,
        )

    def _path_eval_warn_identity_transform(self, source_frame: str, target_frame: str) -> None:
        pair = tuple(sorted((str(source_frame).strip().lower(), str(target_frame).strip().lower())))
        if pair in self._path_eval_identity_warned_pairs:
            return
        self._path_eval_identity_warned_pairs.add(pair)
        self._safe_log_warn(
            'Path tracking evaluation is assuming identity transform between '
            f'{source_frame} and {target_frame}; no TF was provided for this world-frame pair.'
        )

    def _resolve_control_pose_from_odom(
        self,
        msg: Odometry,
        *,
        base_frame: str,
    ) -> Optional[tuple[float, float, float]]:
        pose = msg.pose.pose
        q = pose.orientation
        return self._convert_odom_child_pose_to_base_frame(
            child_frame=str(msg.child_frame_id).strip(),
            base_frame=base_frame,
            tx=float(pose.position.x),
            ty=float(pose.position.y),
            yaw=quaternion_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w)),
        )

    def _path_eval_resolve_control_point_to_frame(
        self,
        *,
        point_xy: np.ndarray,
        source_frame: str,
        child_frame: str,
        yaw_rad: float,
        target_frame: str,
        stamp,
        base_frame: str,
    ) -> Optional[np.ndarray]:
        source_frame = str(source_frame).strip()
        target_frame = str(target_frame).strip()
        resolved_pose = self._convert_odom_child_pose_to_base_frame(
            child_frame=child_frame,
            base_frame=base_frame,
            tx=float(point_xy[0]),
            ty=float(point_xy[1]),
            yaw=float(yaw_rad),
        )
        if resolved_pose is None:
            return None
        return self._path_eval_transform_point_to_frame(
            np.asarray([float(resolved_pose[0]), float(resolved_pose[1])], dtype=np.float64),
            source_frame=source_frame,
            target_frame=target_frame,
            stamp=stamp,
        )

    def _convert_odom_child_pose_to_base_frame(
        self,
        *,
        child_frame: str,
        base_frame: str,
        tx: float,
        ty: float,
        yaw: float,
    ) -> Optional[tuple[float, float, float]]:
        return convert_odom_child_pose_to_base_frame(
            child_frame=child_frame,
            base_frame=base_frame,
            tx=tx,
            ty=ty,
            yaw=yaw,
            wheelbase_m=self._control_reference_wheelbase_m,
            is_alias=self._is_control_frame_alias,
        )

    @staticmethod
    def _control_frame_aliases(frame: str) -> set[str]:
        out: set[str] = set()

        def add(token: str) -> None:
            normalized = str(token).strip().strip('/')
            if normalized:
                out.add(normalized)

        add(frame)
        normalized = str(frame).strip().strip('/').lower()
        if normalized == 'odom':
            add('odom')
        if normalized in {'front_axle', 'base_link', 'base_footprint'}:
            add('front_axle')
            add('base_link')
            add('base_footprint')
        return out

    def _is_control_frame_alias(self, frame_a: str, frame_b: str) -> bool:
        return bool(self._control_frame_aliases(frame_a).intersection(self._control_frame_aliases(frame_b)))

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
