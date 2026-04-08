from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
