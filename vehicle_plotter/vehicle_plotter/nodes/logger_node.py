#!/usr/bin/env python3
"""
LoggerNode - Logs vehicle state to files.

Subscribes to /vehicle_plotter/state and writes data to
Parquet or CSV files. Supports multi-machine run synchronization
via /run_session topic.

Storage Layout:
    ./multidata/<prefix>_<timestamp>/logs/
        vehicle_state_0000.parquet
        metadata.json

Where <prefix> is 'sim' for simulation or 'jetson' for real hardware.
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
from std_msgs.msg import Float32, Int32, String

from ..core.vehicle_state import VehicleState
from ..core.run_session import RunSession
from ..core.qos_profiles import PLOTTER_QOS, RELIABLE_SENSOR_QOS
from ..logging.log_writer import LogWriter
from ..logging.log_config import LogConfig
from ..logging.steering_diagnostics import (
    analyze_csv,
    heading_error,
    parse_planner_diag,
    signed_cross_track_error,
    write_summary_files,
)


class LoggerNode(Node):
    """
    Data logging node for vehicle state.

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
        adapter (str): Sensor adapter type ('gazebo', 'can', 'vectornav') - determines directory prefix
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
        self.declare_parameter('wait_for_session', True)
        self.declare_parameter('session_timeout_sec', 5.0)
        self.declare_parameter('adapter', 'gazebo')  # Determines directory prefix
        self.declare_parameter('auto_plot_on_shutdown', True)
        self.declare_parameter('cone_eval_topic', '/sim/stereo/eval/cone_depth_per_cone')
        self.declare_parameter('cone_log_suffix', '')
        self.declare_parameter('steering_diag_enabled', False)
        self.declare_parameter('steering_diag_rate_hz', 50.0)
        self.declare_parameter('steering_diag_cmd_topic', '/cmd')
        self.declare_parameter('steering_diag_steering_topic', '/sim/steering_angle')
        self.declare_parameter('steering_diag_joint_states_topic', '/sim/raw/joint_states')
        self.declare_parameter('steering_diag_odom_topic', '/sim/odom')
        self.declare_parameter('steering_diag_path_topic', '/planned_centerline')
        self.declare_parameter('steering_diag_planner_diag_topic', '/delaunay_planner/diagnostics')
        self.declare_parameter('steering_diag_filename', 'steering_tracking_diagnostics.csv')
        self.declare_parameter('steering_diag_summary_json', 'steering_tracking_summary.json')
        self.declare_parameter('steering_diag_summary_txt', 'steering_tracking_summary.txt')

        # Get parameters
        self._log_format = self.get_parameter('format').value
        self._compression = self.get_parameter('compression').value
        base_path_str = self.get_parameter('base_path').value
        self._session_name = self.get_parameter('session_name').value
        self._flush_interval = self.get_parameter('flush_interval_sec').value
        self._buffer_size = self.get_parameter('buffer_size').value
        state_topic = self.get_parameter('state_topic').value
        enable_logging = self.get_parameter('enable_logging').value
        self._wait_for_session = self.get_parameter('wait_for_session').value
        self._session_timeout = self.get_parameter('session_timeout_sec').value
        self._adapter_type = self.get_parameter('adapter').value
        self._auto_plot = self.get_parameter('auto_plot_on_shutdown').value
        self._cone_eval_topic = str(self.get_parameter('cone_eval_topic').value).strip()
        self._cone_metrics_prefix = self._derive_cone_metrics_prefix(self._cone_eval_topic)
        self._cone_log_suffix = self._derive_cone_log_suffix(
            str(self.get_parameter('cone_log_suffix').value).strip(),
            self._cone_eval_topic,
        )
        self._steering_diag_enabled = bool(self.get_parameter('steering_diag_enabled').value)
        self._steering_diag_rate_hz = max(1.0, float(self.get_parameter('steering_diag_rate_hz').value))
        self._steering_diag_cmd_topic = str(self.get_parameter('steering_diag_cmd_topic').value).strip() or '/cmd'
        self._steering_diag_steering_topic = str(
            self.get_parameter('steering_diag_steering_topic').value
        ).strip() or '/sim/steering_angle'
        self._steering_diag_joint_states_topic = str(
            self.get_parameter('steering_diag_joint_states_topic').value
        ).strip() or '/sim/raw/joint_states'
        self._steering_diag_odom_topic = str(self.get_parameter('steering_diag_odom_topic').value).strip() or '/sim/odom'
        self._steering_diag_path_topic = str(self.get_parameter('steering_diag_path_topic').value).strip() or '/planned_centerline'
        self._steering_diag_planner_diag_topic = str(
            self.get_parameter('steering_diag_planner_diag_topic').value
        ).strip() or '/delaunay_planner/diagnostics'
        self._steering_diag_filename = str(self.get_parameter('steering_diag_filename').value).strip() or 'steering_tracking_diagnostics.csv'
        self._steering_diag_summary_json = str(self.get_parameter('steering_diag_summary_json').value).strip() or 'steering_tracking_summary.json'
        self._steering_diag_summary_txt = str(self.get_parameter('steering_diag_summary_txt').value).strip() or 'steering_tracking_summary.txt'

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
        self._run_session: Optional[RunSession] = None
        self.log_writer: Optional[LogWriter] = None
        self._session_initialized = False
        self._buffered_states = []  # Buffer states until session is ready
        self._shutdown_called = False
        self._cone_top_count = 4
        self._cone_avg_stride = 15
        self._cone_avg_counter = 0
        self._cone_total_avg_rmse_sum = 0.0
        self._cone_total_avg_rmse_count = 0
        self._cone_total_avg_mae_sum = 0.0
        self._cone_total_avg_mae_count = 0
        self._cone_latest_metrics: Dict[str, float] = {
            'cone_depth_pairs': float('nan'),
            'cone_depth_yolo_detections': float('nan'),
            'cone_depth_yolo_depth_valid': float('nan'),
            'cone_depth_gt_projected': float('nan'),
            'cone_depth_bbox_matches': float('nan'),
            'cone_depth_cone_id_matches': float('nan'),
            'cone_depth_axis_mae_m': float('nan'),
            'cone_depth_axis_rmse_m': float('nan'),
            'cone_depth_axis_bias_m': float('nan'),
            'cone_depth_range_mae_m': float('nan'),
            'cone_depth_range_rmse_m': float('nan'),
            'cone_depth_sync_dt_ms': float('nan'),
            'yolo_detection_count': float('nan'),
            'yolo_inference_ms': float('nan'),
        }
        self._cone_records: List[Dict[str, float]] = []
        self._cone_range_rmse_samples: List[Dict[str, object]] = []
        self._monocular_fit_samples: List[Dict[str, object]] = []
        self._cone_slot_rows: List[Optional[Dict[str, object]]] = [None] * self._cone_top_count
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
        self._diag_planner_metrics: Dict[str, float] = {
            'centerline_jump_max_m': float('nan'),
            'selected_edge_churn_ratio': float('nan'),
            'tracked_cones_frame_delta_p95_m': float('nan'),
        }
        self._diag_file_handle = None
        self._diag_csv_writer: Optional[csv.DictWriter] = None
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

        # Subscribe to vehicle state
        self.state_sub = self.create_subscription(
            VehicleStateMsg,
            state_topic,
            self.state_callback,
            PLOTTER_QOS,
        )

        # Flush timer (will be created after session init)
        self._flush_timer = None

        # Status timer
        self.status_timer = self.create_timer(10.0, self.status_callback)

        self.get_logger().info(f'LoggerNode started, subscribed to {state_topic}')

        # Ensure shutdown handler runs on ROS shutdown as well
        # NOTE: rclpy.on_shutdown / Node.add_on_shutdown are not available in Humble
        rclpy.get_default_context().on_shutdown(self.shutdown)

    def _setup_cone_subscriptions(self) -> None:
        if not self._cone_eval_topic:
            return
        self._cone_per_cone_sub = self.create_subscription(
            String,
            self._cone_eval_topic,
            self._cone_per_cone_callback,
            10,
        )
        prefix = self._cone_metrics_prefix.rstrip('/')
        self._cone_pairs_sub = self.create_subscription(
            Int32,
            f'{prefix}/cone_depth_pairs',
            self._cone_pairs_callback,
            10,
        )
        self._cone_samples_sub = self.create_subscription(
            String,
            f'{prefix}/cone_depth_samples',
            self._cone_depth_samples_callback,
            10,
        )
        self._cone_mono_fit_samples_sub = self.create_subscription(
            String,
            f'{prefix}/cone_depth_monocular_fit_samples',
            self._cone_monocular_fit_samples_callback,
            10,
        )
        self._cone_yolo_detections_sub = self.create_subscription(
            Int32,
            f'{prefix}/cone_depth_yolo_detections',
            self._cone_yolo_detections_callback,
            10,
        )
        self._cone_yolo_depth_valid_sub = self.create_subscription(
            Int32,
            f'{prefix}/cone_depth_yolo_depth_valid',
            self._cone_yolo_depth_valid_callback,
            10,
        )
        self._cone_gt_projected_sub = self.create_subscription(
            Int32,
            f'{prefix}/cone_depth_gt_projected',
            self._cone_gt_projected_callback,
            10,
        )
        self._cone_bbox_matches_sub = self.create_subscription(
            Int32,
            f'{prefix}/cone_depth_bbox_matches',
            self._cone_bbox_matches_callback,
            10,
        )
        self._cone_id_matches_sub = self.create_subscription(
            Int32,
            f'{prefix}/cone_depth_cone_id_matches',
            self._cone_id_matches_callback,
            10,
        )
        self._cone_axis_mae_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_axis_mae_m',
            self._cone_axis_mae_callback,
            10,
        )
        self._cone_axis_rmse_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_axis_rmse_m',
            self._cone_axis_rmse_callback,
            10,
        )
        self._cone_axis_bias_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_axis_bias_m',
            self._cone_axis_bias_callback,
            10,
        )
        self._cone_range_mae_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_range_mae_m',
            self._cone_range_mae_callback,
            10,
        )
        self._cone_range_rmse_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_range_rmse_m',
            self._cone_range_rmse_callback,
            10,
        )
        self._cone_sync_dt_sub = self.create_subscription(
            Float32,
            f'{prefix}/cone_depth_sync_dt_ms',
            self._cone_sync_dt_callback,
            10,
        )
        self._yolo_detection_count_sub = self.create_subscription(
            Int32,
            f'{prefix}/yolo/detection_count',
            self._yolo_detection_count_callback,
            10,
        )
        self._yolo_inference_ms_sub = self.create_subscription(
            Float32,
            f'{prefix}/yolo/inference_ms',
            self._yolo_inference_ms_callback,
            10,
        )
        self.get_logger().info(
            f'Cone logging enabled: table={self._cone_eval_topic}, metrics_prefix={self._cone_metrics_prefix}'
        )

    def _setup_steering_diag_subscriptions(self) -> None:
        if not self._steering_diag_enabled:
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
            'Steering diagnostics enabled: '
            f'cmd={self._steering_diag_cmd_topic} '
            f'steering={self._steering_diag_steering_topic} '
            f'joint_states={self._steering_diag_joint_states_topic} '
            f'odom={self._steering_diag_odom_topic} '
            f'path={self._steering_diag_path_topic}'
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
        topic = cone_eval_topic.strip()
        suffix = '/cone_depth_per_cone'
        if topic.endswith(suffix):
            return topic[:-len(suffix)]
        return topic

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
    def _safe_mean(values: List[float]) -> float:
        valid = [v for v in values if math.isfinite(v)]
        if not valid:
            return float('nan')
        return float(sum(valid) / len(valid))

    def _parse_cone_rows(self, payload: str) -> List[Dict[str, object]]:
        lines = [line.strip() for line in payload.splitlines() if line.strip()]
        if not lines or lines[0].startswith('no per-cone depth samples'):
            return []

        header_map: Dict[str, int] = {}
        start = 0
        if lines[0].startswith('cone_id,'):
            start = 1
            header_parts = [part.strip() for part in lines[0].split(',')]
            header_map = {name: idx for idx, name in enumerate(header_parts)}
        rows: List[Dict[str, object]] = []
        for line in lines[start:]:
            if line.startswith('...'):
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) < 7:
                continue

            def get_value(key: str, fallback_idx: Optional[int] = None) -> str:
                idx = header_map.get(key)
                if idx is None:
                    idx = fallback_idx
                if idx is None or idx < 0 or idx >= len(parts):
                    return ''
                return parts[idx]
            rows.append(
                {
                    'cone_id': parts[0],
                    'mae': self._parse_float(get_value('axis_mae_m', 3)),
                    'rmse': self._parse_float(get_value('axis_rmse_m', 4)),
                    'rmse_x': self._parse_float(get_value('axis_rmse_x_m')),
                    'rmse_y': self._parse_float(get_value('axis_rmse_y_m')),
                    'dcam_inst': self._parse_float(get_value('dcam_inst')),
                    'dgt_inst': self._parse_float(get_value('dgt_inst')),
                    'dcam': self._parse_float(get_value('dcam', 5)),
                    'dgt': self._parse_float(get_value('dgt', 6)),
                }
            )
        return rows

    @staticmethod
    def _row_sort_key(row: Dict[str, object]):
        dgt_inst = row.get('dgt_inst')
        dcam_inst = row.get('dcam_inst')
        dgt_avg = row.get('dgt')
        dcam_avg = row.get('dcam')
        dgt = (
            float(dgt_inst)
            if isinstance(dgt_inst, (int, float)) and math.isfinite(float(dgt_inst))
            else (
                float(dgt_avg)
                if isinstance(dgt_avg, (int, float)) and math.isfinite(float(dgt_avg))
                else math.inf
            )
        )
        dcam = (
            float(dcam_inst)
            if isinstance(dcam_inst, (int, float)) and math.isfinite(float(dcam_inst))
            else (
                float(dcam_avg)
                if isinstance(dcam_avg, (int, float)) and math.isfinite(float(dcam_avg))
                else math.inf
            )
        )
        cone_id = str(row.get('cone_id', ''))
        return (dgt, dcam, cone_id)

    def _select_slot_rows(self, rows: List[Dict[str, object]]) -> List[Optional[Dict[str, object]]]:
        ranked_rows = sorted(rows, key=self._row_sort_key)
        if not ranked_rows:
            return [None] * self._cone_top_count

        by_id: Dict[str, Dict[str, object]] = {}
        for row in ranked_rows:
            cone_id = str(row.get('cone_id', '')).strip()
            if not cone_id:
                continue
            if cone_id not in by_id:
                by_id[cone_id] = row

        selected: List[Optional[Dict[str, object]]] = [None] * self._cone_top_count
        used_ids = set()

        # Keep previous cone-to-column assignment stable when possible.
        for idx, prev in enumerate(self._cone_slot_rows):
            if prev is None:
                continue
            prev_id = str(prev.get('cone_id', '')).strip()
            if not prev_id:
                continue
            current = by_id.get(prev_id)
            if current is None:
                continue
            selected[idx] = current
            used_ids.add(prev_id)

        for row in ranked_rows:
            cone_id = str(row.get('cone_id', '')).strip()
            if not cone_id or cone_id in used_ids:
                continue
            try:
                target_idx = selected.index(None)
            except ValueError:
                break
            selected[target_idx] = row
            used_ids.add(cone_id)

        self._cone_slot_rows = selected
        return selected

    def _cone_per_cone_callback(self, msg: String) -> None:
        rows = self._parse_cone_rows(msg.data)
        selected = self._select_slot_rows(rows)
        timestamp_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        if timestamp_sec <= 0.0:
            timestamp_sec = time.monotonic()

        record: Dict[str, float] = {
            'timestamp': timestamp_sec,
        }
        for idx in range(self._cone_top_count):
            prefix = f'cone_{idx + 1}'
            row = selected[idx] if idx < len(selected) else None
            rmse = float(row['rmse']) if row is not None and isinstance(row.get('rmse'), (int, float)) else float('nan')
            mae = float(row['mae']) if row is not None and isinstance(row.get('mae'), (int, float)) else float('nan')
            rmse_x = float(row['rmse_x']) if row is not None and isinstance(row.get('rmse_x'), (int, float)) else float('nan')
            rmse_y = float(row['rmse_y']) if row is not None and isinstance(row.get('rmse_y'), (int, float)) else float('nan')
            dcam_inst = float(row['dcam_inst']) if row is not None and isinstance(row.get('dcam_inst'), (int, float)) else float('nan')
            dgt_inst = float(row['dgt_inst']) if row is not None and isinstance(row.get('dgt_inst'), (int, float)) else float('nan')
            dcam_avg = float(row['dcam']) if row is not None and isinstance(row.get('dcam'), (int, float)) else float('nan')
            dgt_avg = float(row['dgt']) if row is not None and isinstance(row.get('dgt'), (int, float)) else float('nan')
            dcam_plot = dcam_inst if math.isfinite(dcam_inst) else dcam_avg
            dgt_plot = dgt_inst if math.isfinite(dgt_inst) else dgt_avg

            record[f'{prefix}_axis_rmse_m'] = rmse
            record[f'{prefix}_axis_mae_m'] = mae
            # Legacy aliases kept for compatibility with older tooling.
            record[f'{prefix}_rmse'] = record[f'{prefix}_axis_rmse_m']
            record[f'{prefix}_mae'] = record[f'{prefix}_axis_mae_m']
            record[f'{prefix}_rmse_x'] = rmse_x
            record[f'{prefix}_rmse_y'] = rmse_y
            record[f'{prefix}_dcam'] = dcam_plot
            record[f'{prefix}_dgt'] = dgt_plot
            record[f'{prefix}_dcam_inst'] = dcam_inst
            record[f'{prefix}_dgt_inst'] = dgt_inst
            record[f'{prefix}_dcam_avg'] = dcam_avg
            record[f'{prefix}_dgt_avg'] = dgt_avg

        avg_rmse_now = self._safe_mean([float(row['rmse']) for row in rows if isinstance(row.get('rmse'), (int, float))])
        avg_mae_now = self._safe_mean([float(row['mae']) for row in rows if isinstance(row.get('mae'), (int, float))])
        self._cone_avg_counter += 1
        if self._cone_avg_counter % self._cone_avg_stride == 0:
            if math.isfinite(avg_rmse_now):
                self._cone_total_avg_rmse_sum += avg_rmse_now
                self._cone_total_avg_rmse_count += 1
            if math.isfinite(avg_mae_now):
                self._cone_total_avg_mae_sum += avg_mae_now
                self._cone_total_avg_mae_count += 1

        record['avg_axis_rmse_m'] = avg_rmse_now
        record['avg_axis_mae_m'] = avg_mae_now
        record['avg_total_axis_rmse_m'] = (
            self._cone_total_avg_rmse_sum / self._cone_total_avg_rmse_count
            if self._cone_total_avg_rmse_count > 0
            else float('nan')
        )
        record['avg_total_axis_mae_m'] = (
            self._cone_total_avg_mae_sum / self._cone_total_avg_mae_count
            if self._cone_total_avg_mae_count > 0
            else float('nan')
        )
        # Legacy aliases kept for compatibility with older tooling.
        record['avg_rmse'] = record['avg_axis_rmse_m']
        record['avg_mae'] = record['avg_axis_mae_m']
        record['avg_total_rmse'] = record['avg_total_axis_rmse_m']
        record['avg_total_mae'] = record['avg_total_axis_mae_m']

        for key, value in self._cone_latest_metrics.items():
            record[key] = value

        self._cone_records.append(record)

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
            self._cone_range_rmse_samples.append(
                {
                    'timestamp': timestamp_sec,
                    'source': source,
                    'gt_range_m': gt_range_m,
                    'error_m': error_m,
                    'predicted_class_id': predicted_class_id,
                    'ground_truth_class_id': ground_truth_class_id,
                }
            )

    def _cone_monocular_fit_samples_callback(self, msg: String) -> None:
        payload = str(msg.data).strip()
        if not payload:
            return
        reader = csv.DictReader(payload.splitlines())
        if reader.fieldnames is None:
            return
        for row in reader:
            if not row:
                continue
            parsed: Dict[str, object] = {}
            for key, value in row.items():
                if key is None:
                    continue
                text = '' if value is None else str(value).strip()
                if text == '':
                    parsed[key] = ''
                    continue
                lowered = text.lower()
                if lowered in {'nan', 'n/a', 'none'}:
                    parsed[key] = ''
                    continue
                try:
                    number = float(text)
                except ValueError:
                    parsed[key] = text
                else:
                    parsed[key] = number if math.isfinite(number) else ''
            if parsed:
                self._monocular_fit_samples.append(parsed)

    def _cone_pairs_callback(self, msg: Int32) -> None:
        self._cone_latest_metrics['cone_depth_pairs'] = float(msg.data)

    def _cone_yolo_detections_callback(self, msg: Int32) -> None:
        self._cone_latest_metrics['cone_depth_yolo_detections'] = float(msg.data)

    def _cone_yolo_depth_valid_callback(self, msg: Int32) -> None:
        self._cone_latest_metrics['cone_depth_yolo_depth_valid'] = float(msg.data)

    def _cone_gt_projected_callback(self, msg: Int32) -> None:
        self._cone_latest_metrics['cone_depth_gt_projected'] = float(msg.data)

    def _cone_bbox_matches_callback(self, msg: Int32) -> None:
        self._cone_latest_metrics['cone_depth_bbox_matches'] = float(msg.data)

    def _cone_id_matches_callback(self, msg: Int32) -> None:
        self._cone_latest_metrics['cone_depth_cone_id_matches'] = float(msg.data)

    def _cone_axis_mae_callback(self, msg: Float32) -> None:
        self._cone_latest_metrics['cone_depth_axis_mae_m'] = float(msg.data)

    def _cone_axis_rmse_callback(self, msg: Float32) -> None:
        self._cone_latest_metrics['cone_depth_axis_rmse_m'] = float(msg.data)

    def _cone_axis_bias_callback(self, msg: Float32) -> None:
        self._cone_latest_metrics['cone_depth_axis_bias_m'] = float(msg.data)

    def _cone_range_mae_callback(self, msg: Float32) -> None:
        self._cone_latest_metrics['cone_depth_range_mae_m'] = float(msg.data)

    def _cone_range_rmse_callback(self, msg: Float32) -> None:
        self._cone_latest_metrics['cone_depth_range_rmse_m'] = float(msg.data)

    def _cone_sync_dt_callback(self, msg: Float32) -> None:
        self._cone_latest_metrics['cone_depth_sync_dt_ms'] = float(msg.data)

    def _yolo_detection_count_callback(self, msg: Int32) -> None:
        self._cone_latest_metrics['yolo_detection_count'] = float(msg.data)

    def _yolo_inference_ms_callback(self, msg: Float32) -> None:
        self._cone_latest_metrics['yolo_inference_ms'] = float(msg.data)

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
        if not self._steering_diag_enabled or self._run_session is None:
            return
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
            'vehicle_x_m',
            'vehicle_y_m',
            'vehicle_yaw_rad',
            'vehicle_yaw_rate_rps',
            'vehicle_speed_mps',
            'centerline_available',
            'centerline_point_count',
            'cte_m',
            'cte_abs_m',
            'heading_error_rad',
            'heading_error_abs_rad',
            'planner_centerline_jump_max_m',
            'planner_selected_edge_churn_ratio',
            'planner_tracked_cones_frame_delta_p95_m',
        ]
        self._diag_csv_writer = csv.DictWriter(self._diag_file_handle, fieldnames=fieldnames)
        self._diag_csv_writer.writeheader()
        self._diag_flush_counter = 0
        self._diag_timer = self.create_timer(1.0 / self._steering_diag_rate_hz, self._steering_diag_sample)
        self.get_logger().info(f'Steering diagnostics CSV: {diag_path}')

    def _steering_diag_sample(self) -> None:
        if self._diag_csv_writer is None or self._diag_file_handle is None:
            return

        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        desired = float(self._diag_desired_steering_rad)
        actual_deg = float(self._diag_actual_steering_deg)
        actual_rad = math.radians(actual_deg) if math.isfinite(actual_deg) else float('nan')
        err = (actual_rad - desired) if math.isfinite(actual_rad) and math.isfinite(desired) else float('nan')
        cmd_age_sec = now_sec - self._diag_cmd_recv_sec if math.isfinite(self._diag_cmd_recv_sec) else float('nan')

        centerline_count = int(self._diag_centerline_xy.shape[0])
        centerline_available = 1 if centerline_count >= 2 else 0
        cte_m = float('nan')
        heading_err = float('nan')
        if centerline_available and math.isfinite(self._diag_vehicle_x_m) and math.isfinite(self._diag_vehicle_y_m):
            cte_m, tangent_yaw = signed_cross_track_error(
                self._diag_vehicle_x_m,
                self._diag_vehicle_y_m,
                self._diag_centerline_xy,
            )
            if math.isfinite(tangent_yaw) and math.isfinite(self._diag_vehicle_yaw_rad):
                heading_err = heading_error(self._diag_vehicle_yaw_rad, tangent_yaw)

        planner_jump = float(self._diag_planner_metrics.get('centerline_jump_max_m', float('nan')))
        planner_churn = float(self._diag_planner_metrics.get('selected_edge_churn_ratio', float('nan')))
        planner_tracked_delta = float(
            self._diag_planner_metrics.get('tracked_cones_frame_delta_p95_m', float('nan'))
        )

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
            'vehicle_x_m': self._diag_vehicle_x_m,
            'vehicle_y_m': self._diag_vehicle_y_m,
            'vehicle_yaw_rad': self._diag_vehicle_yaw_rad,
            'vehicle_yaw_rate_rps': self._diag_vehicle_yaw_rate_rps,
            'vehicle_speed_mps': self._diag_vehicle_speed_mps,
            'centerline_available': centerline_available,
            'centerline_point_count': centerline_count,
            'cte_m': cte_m,
            'cte_abs_m': abs(cte_m) if math.isfinite(cte_m) else float('nan'),
            'heading_error_rad': heading_err,
            'heading_error_abs_rad': abs(heading_err) if math.isfinite(heading_err) else float('nan'),
            'planner_centerline_jump_max_m': planner_jump,
            'planner_selected_edge_churn_ratio': planner_churn,
            'planner_tracked_cones_frame_delta_p95_m': planner_tracked_delta,
        }
        self._diag_csv_writer.writerow(row)
        self._diag_flush_counter += 1
        if self._diag_flush_counter >= self._diag_flush_stride:
            self._diag_file_handle.flush()
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
                self._save_cone_metrics_csv()
                self._save_cone_range_rmse_samples_csv()
                self._save_monocular_fit_samples_csv()
                self._finalize_steering_diag_outputs()

            # Auto-generate plots if enabled
            if self._auto_plot and self._run_session is not None:
                self._generate_offline_plots()

    def _finalize_steering_diag_outputs(self) -> None:
        if not self._steering_diag_enabled or self._run_session is None:
            return
        if self._diag_timer is not None:
            self._diag_timer.cancel()
            self._diag_timer = None
        if self._diag_file_handle is None:
            return
        self._diag_file_handle.flush()
        self._diag_file_handle.close()
        self._diag_file_handle = None
        self._diag_csv_writer = None

        csv_path = self._run_session.logs_path / self._steering_diag_filename
        summary_json = self._run_session.logs_path / self._steering_diag_summary_json
        summary_txt = self._run_session.logs_path / self._steering_diag_summary_txt
        try:
            summary = analyze_csv(csv_path)
            write_summary_files(summary, summary_json, summary_txt)
            self._safe_log_info(
                'Steering diagnostics summary: '
                f"rms={summary.get('steering_error_rms_rad', float('nan')):.4f} rad "
                f"cte_rms={summary.get('cte_rms_m', float('nan')):.4f} m "
                f"lag={summary.get('lag_sec', float('nan')):.4f} s"
            )
        except Exception as exc:
            self._safe_log_warn(f'Failed steering diagnostics analysis: {exc}')

    def _save_cone_metrics_csv(self) -> None:
        if self._run_session is None or not self._cone_records:
            return
        out_path = self._run_session.logs_path / self._cone_output_filename('cone_metrics', 'csv')
        fieldnames = list(self._cone_records[0].keys())
        with open(out_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._cone_records)
        self._safe_log_info(f'Saved cone metrics log: {out_path}')

    def _save_cone_range_rmse_samples_csv(self) -> None:
        if self._run_session is None or not self._cone_range_rmse_samples:
            return
        out_path = self._run_session.logs_path / self._cone_output_filename('cone_range_rmse_samples', 'csv')
        fieldnames = [
            'timestamp',
            'source',
            'gt_range_m',
            'error_m',
            'predicted_class_id',
            'ground_truth_class_id',
        ]
        with open(out_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._cone_range_rmse_samples)
        self._safe_log_info(f'Saved cone range RMSE sample log: {out_path}')

    def _save_monocular_fit_samples_csv(self) -> None:
        if self._run_session is None or not self._monocular_fit_samples:
            return
        out_path = self._run_session.logs_path / self._cone_output_filename('monocular_fit_samples', 'csv')
        fieldnames = list(self._monocular_fit_samples[0].keys())
        with open(out_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._monocular_fit_samples)
        self._safe_log_info(f'Saved monocular fit sample log: {out_path}')

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
                metrics_filename=self._cone_output_filename('cone_metrics', 'csv'),
                range_rmse_filename=self._cone_output_filename('cone_range_rmse_samples', 'csv'),
                output_suffix=self._cone_log_suffix,
            )
            cone_generated = cone_plotter.generate_plot()
            if cone_generated is not None:
                total += 1
                self._safe_log_info(f"Generated cone offline plot: {cone_generated}")
            else:
                self._safe_log_warn("Cone offline plot skipped: no cone metrics data found")
            cone_range_generated = cone_plotter.generate_range_rmse_plot()
            if cone_range_generated is not None:
                total += 1
                self._safe_log_info(f"Generated cone range RMSE offline plot: {cone_range_generated}")
            else:
                self._safe_log_warn("Cone range RMSE offline plot skipped: no cone range samples found")

            if self._cone_log_suffix != 'lidar':
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
                    "Reinstall compatible versions to enable cone_depth_validation.png generation."
                )
        except Exception as e:
            self._safe_log_warn(f"Failed to generate cone offline plots: {e}")

        self._safe_log_info(f"Generated {total} plots in {self._run_session.plots_path}")

    @staticmethod
    def _derive_cone_log_suffix(configured_suffix: str, cone_eval_topic: str) -> str:
        suffix = configured_suffix.strip().lower()
        if suffix:
            return ''.join(ch if (ch.isalnum() or ch in {'_', '-'}) else '_' for ch in suffix).strip('_')
        topic = cone_eval_topic.strip().lower()
        if '/lidar/' in topic:
            return 'lidar'
        return ''

    def _cone_output_filename(self, stem: str, ext: str) -> str:
        clean_stem = stem.strip()
        clean_ext = ext.strip().lstrip('.')
        if self._cone_log_suffix:
            return f'{clean_stem}_{self._cone_log_suffix}.{clean_ext}'
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
