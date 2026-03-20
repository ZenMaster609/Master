from __future__ import annotations

from sim_car.planning.delaunay_planner_node import DelaunayPlannerNode


class TrackedConePlannerBase(DelaunayPlannerNode):
    """Shared tracked-cone planner runtime used by midpoint/single-boundary planners."""

