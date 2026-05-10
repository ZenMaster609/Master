"""Diagnostic publishing logic for the tracked-cone planner runtime."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue


class DiagnosticsMixin:
    """Diagnostic publishing, operator state text, and metric formatting for TrackedConePlannerRuntime."""

    def _log_operator_state_transition(
        self,
        *,
        operator_state: str,
        operator_reason: str,
        hold_remaining_s: float,
        selected_chain_length: int,
    ) -> None:
        if operator_state == self._last_operator_state and operator_reason == self._last_operator_reason:
            return
        prev_state = self._last_operator_state or 'none'
        prev_reason = self._last_operator_reason or 'none'
        self.get_logger().info(
            'planner state %s -> %s | reason=%s | prev_reason=%s | hold_remaining=%.2fs | chain=%d | clean_frames=%d/%d'
            % (
                prev_state,
                operator_state,
                operator_reason,
                prev_reason,
                float(hold_remaining_s),
                int(selected_chain_length),
                int(self._hold_clean_frame_count),
                int(self.hold_exit_clean_frames),
            )
        )
        self._last_operator_state = operator_state
        self._last_operator_reason = operator_reason

    @staticmethod
    def _fmt_metric(value: float, suffix: str = '', *, digits: int = 2) -> str:
        value_f = float(value)
        if not math.isfinite(value_f):
            return 'n/a'
        return f'{value_f:.{digits}f}{suffix}'

    def _build_operator_status_text(
        self,
        *,
        operator_state: str,
        operator_reason: str,
        centerline_point_count: int,
        cmd_speed: float,
        cmd_steering: float,
        lookahead: float,
        candidate_diagonal_count: int,
        selected_chain_length: int,
        seed_midpoint_distance_m: float,
        near_field_lateral_max_m: float,
        near_field_midpoint_kink_max_rad: float,
        hold_remaining_s: float,
    ) -> str:
        return '\n'.join(
            [
                f'STATE: {operator_state.upper()}',
                f'REASON: {self._operator_reason_label(operator_reason)}',
                (
                    f'PATH: {int(centerline_point_count)} pts | '
                    f'chain={int(selected_chain_length)} | cand={int(candidate_diagonal_count)}'
                ),
                (
                    f'CMD: v={self._fmt_metric(cmd_speed, " m/s")} | '
                    f'delta={self._fmt_metric(cmd_steering, " rad")} | '
                    f'Ld={self._fmt_metric(lookahead, " m")}'
                ),
                (
                    f'NF: lat={self._fmt_metric(near_field_lateral_max_m, " m")} | '
                    f'kink={self._fmt_metric(near_field_midpoint_kink_max_rad, " rad")} | '
                    f'seed={self._fmt_metric(seed_midpoint_distance_m, " m")}'
                ),
                (
                    f'HOLD: {self._fmt_metric(hold_remaining_s, " s")} left | '
                    f'clean {int(self._hold_clean_frame_count)}/{int(self.hold_exit_clean_frames)}'
                ),
            ]
        )

    def _publish_empty_cycle(
        self,
        *,
        frame_id: str,
        status: str,
        operator_state: str,
        operator_reason: str,
        cmd_speed: float,
        cmd_steering: float,
        lookahead: float,
        zero_cmd_sent_flag: int,
    ) -> None:
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        hold_remaining_s = self._hold_remaining_s(now_sec)
        self._log_operator_state_transition(
            operator_state=operator_state,
            operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            selected_chain_length=0,
        )
        self._publish_diagnostics(
            frame_id=frame_id,
            centerline_point_count=0,
            selected_edge_count=0,
            status=status,
            planner_metrics={
                'candidate_diagonal_count': 0,
                'selected_chain_length': 0,
                'selected_chain_median_width_m': float('nan'),
                'expected_width_prior_m': float('nan'),
                'reject_wrong_side_count': 0,
                'reject_width_count': 0,
                'reject_width_range_count': 0,
                'reject_width_prior_count': 0,
                'reject_orientation_count': 0,
                'reject_progress_count': 0,
                'reject_near_field_continuity_count': 0,
                'reject_midpoint_kink_count': 0,
                'reject_seed_distance_count': 0,
                'near_field_lateral_max_m': float('nan'),
                'near_field_lateral_mean_m': float('nan'),
                'near_field_displacement_max_m': float('nan'),
                'near_field_displacement_mean_m': float('nan'),
                'near_field_midpoint_kink_max_rad': float('nan'),
                'seed_midpoint_distance_m': float('nan'),
                'seed_temporal_offset_m': float('nan'),
                'publish_mode': operator_state if operator_state in {'fresh', 'held'} else 'held',
                'hold_mode_active_flag': 1 if self._hold_mode_active else 0,
                'hold_clean_frame_count': self._hold_clean_frame_count,
                'hold_reason': '',
                'planner_state_code': self._operator_state_code(operator_state),
                'fresh_publish_flag': 1 if operator_state == 'fresh' else 0,
                'held_publish_flag': 1 if operator_state == 'held' else 0,
                'stopped_flag': 1 if operator_state == 'stopped' else 0,
                'waiting_flag': 1 if operator_state == 'waiting' else 0,
                'operator_reason_code': self._operator_reason_code(operator_reason),
                'operator_reason': operator_reason,
                'hold_remaining_s': hold_remaining_s,
                'control_path_point_count': 0,
                'zero_cmd_sent_flag': int(zero_cmd_sent_flag),
                'planner_mode': self._active_planner_mode,
            },
        )
        self._publish_outputs(
            frame_id=frame_id,
            centerline=np.empty((0, 2), dtype=np.float64),
            raw_centerline=np.empty((0, 2), dtype=np.float64),
            raw_midpoint_chain=np.empty((0, 2), dtype=np.float64),
            result=None,
            status=status,
            control_target_frame=None,
            cmd_speed=cmd_speed,
            cmd_steering=cmd_steering,
            lookahead=lookahead,
            operator_state=operator_state,
            operator_reason=operator_reason,
            hold_remaining_s=hold_remaining_s,
            control_path_point_count=0,
            candidate_diagonal_count=0,
            selected_chain_length=0,
            seed_midpoint_distance_m=float('nan'),
            near_field_lateral_max_m=float('nan'),
            near_field_midpoint_kink_max_rad=float('nan'),
        )

    def _publish_diagnostics(
        self,
        *,
        frame_id: str,
        centerline_point_count: int,
        selected_edge_count: int,
        status: str,
        control_debug_metrics: Optional[dict[str, float]] = None,
        planner_metrics: Optional[dict[str, object]] = None,
    ) -> None:
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        diag = DiagnosticStatus()
        diag.name = f'{self._planner_identity.diagnostics_prefix}/stability'
        diag.hardware_id = self._planner_identity.hardware_id
        diag.level = DiagnosticStatus.OK
        diag.message = status
        diag.values = [
            KeyValue(key='centerline_point_count', value=str(int(centerline_point_count))),
            KeyValue(key='selected_edge_count', value=str(int(selected_edge_count))),
        ]
        if planner_metrics is not None:
            for key in sorted(planner_metrics):
                diag.values.append(KeyValue(key=str(key), value=self._diag_value_string(planner_metrics[key])))
        msg.status.append(diag)

        if self.publish_control_debug:
            merged = self._default_control_debug_metrics()
            if control_debug_metrics is not None:
                merged.update(control_debug_metrics)
            control_diag = DiagnosticStatus()
            control_diag.name = f'{self._planner_identity.diagnostics_prefix}/control_debug'
            control_diag.hardware_id = self._planner_identity.hardware_id
            control_diag.level = DiagnosticStatus.OK
            control_diag.message = 'stanley debug signals'
            control_diag.values = [
                KeyValue(key='heading_error_rad', value=f"{merged['heading_error_rad']:.6f}"),
                KeyValue(key='cross_track_error_m', value=f"{merged['cross_track_error_m']:.6f}"),
                KeyValue(key='vehicle_speed_mps', value=f"{merged['vehicle_speed_mps']:.6f}"),
                KeyValue(key='speed_term_mps', value=f"{merged['speed_term_mps']:.6f}"),
                KeyValue(key='heading_contribution_rad', value=f"{merged['heading_contribution_rad']:.6f}"),
                KeyValue(
                    key='cross_track_contribution_rad',
                    value=f"{merged['cross_track_contribution_rad']:.6f}",
                ),
                KeyValue(
                    key='yaw_rate_damping_contribution_rad',
                    value=f"{merged['yaw_rate_damping_contribution_rad']:.6f}",
                ),
                KeyValue(key='yaw_rate_rps', value=f"{merged['yaw_rate_rps']:.6f}"),
                KeyValue(key='raw_steering_cmd_rad', value=f"{merged['raw_steering_cmd_rad']:.6f}"),
                KeyValue(
                    key='steering_after_clamp_rad',
                    value=f"{merged['steering_after_clamp_rad']:.6f}",
                ),
                KeyValue(
                    key='steering_after_filter_rad',
                    value=f"{merged['steering_after_filter_rad']:.6f}",
                ),
                KeyValue(
                    key='steering_after_rate_limit_rad',
                    value=f"{merged['steering_after_rate_limit_rad']:.6f}",
                ),
                KeyValue(key='final_steering_cmd_rad', value=f"{merged['final_steering_cmd_rad']:.6f}"),
                KeyValue(
                    key='steering_saturated_flag',
                    value=str(int(round(merged['steering_saturated_flag']))),
                ),
                KeyValue(key='nearest_path_index', value=str(int(round(merged['nearest_path_index'])))),
                KeyValue(key='heading_path_index', value=str(int(round(merged['heading_path_index'])))),
                KeyValue(
                    key='target_point_x_base_m',
                    value=f"{merged['target_point_x_base_m']:.6f}",
                ),
                KeyValue(
                    key='target_point_y_base_m',
                    value=f"{merged['target_point_y_base_m']:.6f}",
                ),
                KeyValue(
                    key='target_point_x_frame_m',
                    value=f"{merged['target_point_x_frame_m']:.6f}",
                ),
                KeyValue(
                    key='target_point_y_frame_m',
                    value=f"{merged['target_point_y_frame_m']:.6f}",
                ),
            ]
            msg.status.append(control_diag)
        self._diag_pub.publish(msg)

    @staticmethod
    def _default_control_debug_metrics() -> dict[str, float]:
        nan = float('nan')
        return {
            'heading_error_rad': nan,
            'cross_track_error_m': nan,
            'vehicle_speed_mps': nan,
            'speed_term_mps': nan,
            'heading_contribution_rad': nan,
            'cross_track_contribution_rad': nan,
            'yaw_rate_damping_contribution_rad': nan,
            'yaw_rate_rps': nan,
            'raw_steering_cmd_rad': nan,
            'steering_after_clamp_rad': nan,
            'steering_after_filter_rad': nan,
            'steering_after_rate_limit_rad': nan,
            'final_steering_cmd_rad': nan,
            'steering_saturated_flag': 0.0,
            'nearest_path_index': -1.0,
            'heading_path_index': -1.0,
            'target_point_x_base_m': nan,
            'target_point_y_base_m': nan,
            'target_point_x_frame_m': nan,
            'target_point_y_frame_m': nan,
        }

    @staticmethod
    def _diag_value_string(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            value_f = float(value)
            return f'{value_f:.6f}' if math.isfinite(value_f) else 'nan'
        return str(value)


    @staticmethod
    def _diagnostic_path_length_m(centerline: np.ndarray) -> float:
        if centerline.shape[0] < 2:
            return float('nan')
        diffs = np.diff(centerline, axis=0)
        lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        if lengths.size == 0:
            return float('nan')
        return float(np.sum(lengths))

    @staticmethod
    def _path_curvature_abs_p95(centerline: np.ndarray) -> float:
        if centerline.shape[0] < 3:
            return float('nan')
        diffs = np.diff(centerline, axis=0)
        seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
        valid = seg_len > 1e-6
        if np.count_nonzero(valid) < 2:
            return float('nan')
        headings = np.arctan2(diffs[valid, 1], diffs[valid, 0])
        if headings.size < 2:
            return float('nan')
        delta_heading = np.arctan2(
            np.sin(np.diff(headings)),
            np.cos(np.diff(headings)),
        )
        ds = 0.5 * (seg_len[valid][1:] + seg_len[valid][:-1])
        valid_ds = ds > 1e-6
        if not np.any(valid_ds):
            return float('nan')
        curvature = np.abs(delta_heading[valid_ds] / ds[valid_ds])
        if curvature.size == 0:
            return float('nan')
        return float(np.percentile(curvature, 95.0))
