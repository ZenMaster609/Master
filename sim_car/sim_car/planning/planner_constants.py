"""Shared constants for planner runtime and node modules."""

from __future__ import annotations

from vehicle_plotter_msgs.msg import ConeDetection

MSG_TRACK_STATE_TENTATIVE = int(getattr(ConeDetection, "TRACK_STATE_TENTATIVE", 0))
MSG_TRACK_STATE_CONFIRMED = int(getattr(ConeDetection, "TRACK_STATE_CONFIRMED", 1))
MSG_TRACK_STATE_STALE = int(getattr(ConeDetection, "TRACK_STATE_STALE", 2))

# Accept validated jumps only inside the near-field horizon used by all planners.
VALIDATED_JUMP_ACCEPT_HORIZON_M = 3.0
# Reject single-point jumps larger than the existing near-field lateral tolerance.
VALIDATED_JUMP_ACCEPT_LATERAL_MAX_M = 0.45
# Reject sustained jumps larger than the existing mean lateral tolerance.
VALIDATED_JUMP_ACCEPT_LATERAL_MEAN_M = 0.25
# Reject heading changes larger than the existing validated-jump tolerance.
VALIDATED_JUMP_ACCEPT_HEADING_DELTA_RAD = 0.30

# Keep recently passed pairs long enough to avoid flicker at the vehicle origin.
PAIR_PASSED_MARGIN_M = 0.5
# Render shared midpoint/corridor centerline markers at the existing RViz width.
CENTERLINE_MARKER_WIDTH_M = 0.20

OPERATOR_STATE_CODES = {
    "waiting": 0,
    "fresh": 1,
    "held": 2,
    "stopped": 3,
}

OPERATOR_REASON_CODES = {
    "none": 0,
    "waiting_for_cones": 1,
    "missing_vehicle_pose": 2,
    "cone_transform_unavailable": 3,
    "no_safe_chain": 4,
    "near_field_continuity": 5,
    "seed_distance": 6,
    "midpoint_kink": 7,
    "hysteresis_holding": 8,
    "holding_previous_valid": 9,
    "hold_expired_no_path": 10,
    "no_control_path": 11,
    "controller_compute_failed": 12,
    "stop_if_no_path": 13,
    "controller_disabled": 14,
    "missing_gt_midline": 15,
}
