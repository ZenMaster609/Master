"""Operator state/hold-hysteresis logic for the tracked-cone planner runtime."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sim_car.planning.planner_constants import (
    OPERATOR_REASON_CODES as _OPERATOR_REASON_CODES,
    OPERATOR_STATE_CODES as _OPERATOR_STATE_CODES,
)

_OPERATOR_REASON_LABELS = {
    'none': 'fresh path accepted',
    'waiting_for_cones': 'waiting for /tracked_cones',
    'missing_vehicle_pose': 'missing vehicle pose',
    'cone_transform_unavailable': 'cone transform unavailable',
    'no_safe_chain': 'no safe fresh chain',
    'near_field_continuity': 'near-field continuity rejected fresh chain',
    'seed_distance': 'seed midpoint too far ahead',
    'midpoint_kink': 'near-field midpoint kink too large',
    'hysteresis_holding': 'holding previous valid path during hysteresis',
    'holding_previous_valid': 'holding previous valid path',
    'hold_expired_no_path': 'held path expired and no fresh path is available',
    'no_control_path': 'no control path available',
    'controller_compute_failed': 'controller compute failed',
    'stop_if_no_path': 'stop_if_no_path sent zero command',
    'controller_disabled': 'controller disabled',
}

_OPERATOR_STATE_COLORS = {
    'waiting': (0.9, 0.9, 0.9),
    'fresh': (0.2, 1.0, 0.3),
    'held': (1.0, 0.9, 0.2),
    'stopped': (1.0, 0.2, 0.2),
}


class StateMachineMixin:
    """Hold/hysteresis and operator state helpers for TrackedConePlannerRuntime."""

    def _apply_no_path_behavior(self) -> bool:
        if self.stop_if_no_path:
            self._publish_cmd(0.0, 0.0)
            self._last_speed_cmd = 0.0
            self._last_steering_cmd = 0.0
            return True
        if self._last_speed_cmd is not None and self._last_steering_cmd is not None:
            self._publish_cmd(float(self._last_speed_cmd), float(self._last_steering_cmd))
        return False

    def _apply_controller_disabled_behavior(self) -> bool:
        if self.stop_if_no_path:
            self._publish_cmd(0.0, 0.0)
            self._last_speed_cmd = 0.0
            self._last_steering_cmd = 0.0
            return True
        return False

    def _update_committed_near_field(
        self,
        *,
        centerline: np.ndarray,
        selected_edge_churn: float,
        selected_chain_length: int,
        previous_edge_keys: set[tuple[int, int, int, int]],
    ) -> None:
        if not self.enable_committed_near_field or self.commit_plan_horizon_m <= 0.0:
            return
        if centerline.shape[0] < 2 or selected_chain_length <= 0:
            self._commit_stable_frame_count = 0
            return

        stable_enough = (not previous_edge_keys) or (
            selected_edge_churn <= self.commit_update_max_churn_ratio
        )
        if not stable_enough:
            self._commit_stable_frame_count = 0
            return

        self._commit_stable_frame_count += 1
        if self._commit_stable_frame_count < self.commit_stable_frames:
            return

        self._committed_centerline = np.array(centerline, copy=True)

    def _record_valid_plan(
        self,
        *,
        now_sec: float,
        centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        selected_chain_width_median: float,
    ) -> None:
        self._last_valid_centerline = np.array(centerline, copy=True)
        self._last_valid_raw_midpoint_chain = (
            np.array(raw_midpoint_chain, copy=True)
            if raw_midpoint_chain.shape[0] > 0
            else None
        )
        self._last_valid_width_m = (
            float(selected_chain_width_median)
            if math.isfinite(float(selected_chain_width_median))
            else self._last_valid_width_m
        )
        self._last_valid_time_sec = now_sec

    def _advance_hold_hysteresis(self, *, plan_ok: bool) -> bool:
        if not plan_ok:
            self._hold_mode_active = True
            self._hold_clean_frame_count = 0
            return True
        # Re-enter fresh publishing only after consecutive clean frames so borderline cases do not
        # flap between held and fresh controller inputs.
        if not self._hold_mode_active:
            return False
        self._hold_clean_frame_count += 1
        if self._hold_clean_frame_count < self.hold_exit_clean_frames:
            return True
        self._hold_mode_active = False
        self._hold_clean_frame_count = 0
        return False

    def _held_centerline(self, now_sec: float) -> Optional[np.ndarray]:
        if self._last_valid_centerline is None:
            return None
        if self._last_valid_time_sec < 0.0:
            return None
        if (now_sec - self._last_valid_time_sec) > self.hold_last_valid_s:
            return None
        return np.array(self._last_valid_centerline, copy=True)

    def _hold_remaining_s(self, now_sec: float) -> float:
        if self._last_valid_centerline is None or self._last_valid_time_sec < 0.0:
            return 0.0
        return max(0.0, float(self.hold_last_valid_s) - max(0.0, now_sec - self._last_valid_time_sec))

    @staticmethod
    def _operator_state_code(state: str) -> int:
        return int(_OPERATOR_STATE_CODES.get(state, 0))

    @staticmethod
    def _operator_reason_code(reason: str) -> int:
        return int(_OPERATOR_REASON_CODES.get(reason, 0))

    @staticmethod
    def _operator_reason_label(reason: str) -> str:
        return _OPERATOR_REASON_LABELS.get(reason, reason.replace('_', ' '))

    @staticmethod
    def _operator_state_color(state: str) -> tuple[float, float, float]:
        return _OPERATOR_STATE_COLORS.get(state, _OPERATOR_STATE_COLORS['waiting'])

    def _normalize_core_reject_reason(self, result: Optional[object]) -> str:
        if result is None:
            return 'none'
        reject_counts = result.reject_counts or {}
        if int(reject_counts.get('near_field_continuity', 0)) > 0:
            return 'near_field_continuity'
        if int(reject_counts.get('seed_distance', 0)) > 0:
            return 'seed_distance'
        if int(reject_counts.get('midpoint_kink', 0)) > 0:
            return 'midpoint_kink'
        text = (result.reject_reason or result.status or '').strip().lower()
        if text in {
            'no valid diagonal candidates',
            'no safe zig-zag chain',
            'centerline generation failed',
            'dedup removed all selected midpoints',
            'no cones available',
        }:
            return 'no_safe_chain'
        if text.startswith('usable cones below minimum'):
            return 'no_safe_chain'
        return 'none'
