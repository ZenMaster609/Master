from __future__ import annotations

from sim_car.planning.tracked_cone_planner_runtime import TrackedConePlannerRuntime


class TrackedConePlannerBase(TrackedConePlannerRuntime):
    """Shared tracked-cone planner runtime used by midpoint/single-boundary planners."""
