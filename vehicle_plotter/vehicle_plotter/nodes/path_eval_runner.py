"""Path tracking evaluation runtime for logger_node."""

from __future__ import annotations

import csv
import math
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from sim_car.cones.tracking.pose import convert_odom_child_pose_to_base_frame

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
from ..logging.track_metrics_report import write_track_metrics_report
from ..utils.transforms import quaternion_to_yaw


class PathEvalRunner:

    def _init_path_eval_state(self) -> None:
        self._path_eval_tf_buffer = None
        self._path_eval_tf_listener = None
        self._path_eval_latest_gt_msg = None
        self._path_eval_gt_midline_source = None
        self._path_eval_last_gt_midline_xy = np.empty((0, 2), dtype=np.float64)
        self._path_eval_last_gt_left_xy = np.empty((0, 2), dtype=np.float64)
        self._path_eval_last_gt_right_xy = np.empty((0, 2), dtype=np.float64)
        self._path_eval_last_target_frame = ''
        self._path_eval_start_xy = None
        self._path_eval_start_heading_xy = None
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
        self._path_eval_csv_writer = None
        self._path_eval_flush_counter = 0
        self._path_eval_flush_stride = max(1, int(self._path_tracking_eval_rate_hz * 2.0))
        self._path_eval_timer = None
        self._path_eval_identity_warned_pairs: set[tuple[str, str]] = set()
        self._path_eval_smalltrack_gate_source = None
        self._path_eval_smalltrack_lap_counter = None
        self._path_eval_smalltrack_completed_laps = 0
        self._path_eval_smalltrack_lap_times_sec: list[float] = []
        self._path_eval_smalltrack_autostop_triggered = False
        self._path_eval_run_start_sec = None
        self._path_eval_run_last_sec = None
        self._off_track_no_cone_since = None
        self._off_track_autostop_triggered = False

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
