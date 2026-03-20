from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerIdentity:
    node_name: str
    planner_mode: str
    diagnostics_prefix: str
    diagnostics_topic: str

    @property
    def hardware_id(self) -> str:
        return f"sim_car.{self.diagnostics_prefix}"

