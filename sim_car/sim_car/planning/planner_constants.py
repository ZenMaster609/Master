"""Shared constants, pipeline defaults, runtime types, and planner registry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

# --- Pipeline Defaults ---

ODOM_FRAME_DEFAULT = "odom"  # Shared odometry frame used by simulation and planners.
BASE_FRAME_DEFAULT = "front_axle"  # Controller geometry is referenced at the front axle.
TRACKED_CONES_TOPIC_DEFAULT = "/tracked_cones"  # Cone memory publishes the planner input here.
ODOM_TOPIC_DEFAULT = "/sim/odom"  # Simulation odometry topic used by cone memory and planners.
WHEELBASE_M_DEFAULT = 1.65  # Meters; matches the EUFS vehicle kinematic wheelbase.
TF_TIMEOUT_S_DEFAULT = 0.03  # Seconds; short enough for realtime planning without hiding TF issues.
PLANNER_MAX_RANGE_M_DEFAULT = 25.0  # Meters; matches useful lidar range in the default sim setup.
PLANNER_MIN_CONFIDENCE_DEFAULT = 0.3  # Unitless; filters noisy tracks while keeping sparse layouts.

# --- Runtime Types ---


@dataclass(frozen=True)
class PlannerIdentity:
    node_name: str
    planner_mode: str
    diagnostics_prefix: str
    diagnostics_topic: str

    @property
    def hardware_id(self) -> str:
        return f"sim_car.{self.diagnostics_prefix}"


@dataclass(frozen=True)
class TrackedConePlanningMetadata:
    track_ids: np.ndarray
    track_states: np.ndarray
    track_confidences: np.ndarray


@dataclass(frozen=True)
class TrackedConePlanningFrame:
    points_xy: np.ndarray
    colors: list[str]
    raw_confidences: np.ndarray
    planner_confidences: np.ndarray
    raw_colors: list[str]
    boundary_hints: list[str]
    metadata: TrackedConePlanningMetadata

    @property
    def track_ids(self) -> np.ndarray:
        return self.metadata.track_ids

    @property
    def track_states(self) -> np.ndarray:
        return self.metadata.track_states

    @property
    def track_confidences(self) -> np.ndarray:
        return self.metadata.track_confidences

# --- Planner Registry ---


@dataclass(frozen=True)
class PlannerLaunchSpec:
    name: str
    executable: str
    diagnostics_topic: str
    default_rviz_profile: str
    allowed_tracks: frozenset[str] | None = None


MIGRATED_PLANNERS = frozenset({'midpoint', 'single_boundary', 'corridor'})
CONFIGURED_PLANNERS = frozenset(set(MIGRATED_PLANNERS) | {'linetest'})
SUPPORTED_PLANNERS = frozenset(set(CONFIGURED_PLANNERS) | {'none'})
SUPPORTED_CONTROLLERS = frozenset({'stanley', 'pure_pursuit', 'none'})

PLANNER_REGISTRY: dict[str, PlannerLaunchSpec] = {
    'midpoint': PlannerLaunchSpec(
        name='midpoint',
        executable='midpoint_planner_node',
        diagnostics_topic='/midpoint_planner/diagnostics',
        default_rviz_profile='midpoint',
    ),
    'single_boundary': PlannerLaunchSpec(
        name='single_boundary',
        executable='single_boundary_planner_node',
        diagnostics_topic='/single_boundary_planner/diagnostics',
        default_rviz_profile='single_boundary',
    ),
    'corridor': PlannerLaunchSpec(
        name='corridor',
        executable='corridor_planner_node',
        diagnostics_topic='/corridor_planner/diagnostics',
        default_rviz_profile='corridor',
    ),
    'linetest': PlannerLaunchSpec(
        name='linetest',
        executable='linetest_planner_node',
        diagnostics_topic='/linetest_planner/diagnostics',
        default_rviz_profile='linetest',
        allowed_tracks=frozenset({'acceleration', 'smalltrack'}),
    ),
    'none': PlannerLaunchSpec(
        name='none',
        executable='',
        diagnostics_topic='/midpoint_planner/diagnostics',
        default_rviz_profile='clean',
    ),
}


def get_planner_spec(planner_name: str) -> PlannerLaunchSpec:
    normalized_name = str(planner_name).strip().lower()
    if normalized_name not in SUPPORTED_PLANNERS:
        raise KeyError(normalized_name)
    return PLANNER_REGISTRY[normalized_name]


def planner_allowed_for_track(*, planner_name: str, track_name: str) -> bool:
    spec = get_planner_spec(planner_name)
    if spec.allowed_tracks is None:
        return True
    return str(track_name).strip().lower() in spec.allowed_tracks
