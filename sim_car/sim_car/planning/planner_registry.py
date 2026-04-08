from __future__ import annotations

from dataclasses import dataclass


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
        allowed_tracks=frozenset({'acceleration'}),
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
