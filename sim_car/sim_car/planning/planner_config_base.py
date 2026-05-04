"""Shared planner configuration fields."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BasePlannerConfig:
    # --- Filtering ---
    max_cone_range_m: float = 25.0  # Meters; keeps planner inputs inside the useful sensor horizon.
    behind_drop_m: float = 5.0  # Meters; retains a small rear buffer for near-vehicle continuity.
    min_confidence: float = 0.3  # Unitless; accepts moderately stable detections while filtering noise.
    min_required_cones: int = 4  # Count; minimum geometry needed before path generation is meaningful.

    # --- Boundary Chain ---
    min_step_m: float = 0.8  # Meters; rejects duplicate or nearly overlapping cones in a chain.
    max_heading_change_rad: float = 1.0  # Radians; allows normal turns while rejecting abrupt chain flips.
    min_forward_progress_m: float = 0.2  # Meters; prevents chain growth from moving mostly sideways.

    # --- Width Estimation ---
    initial_width_m: float = 3.6  # Meters; nominal Formula Student track width prior.
    min_width_m: float = 2.4  # Meters; lower clamp for plausible track width estimates.
    max_width_m: float = 4.8  # Meters; upper clamp for plausible track width estimates.
    width_filter_alpha: float = 0.15  # Unitless; smooths width updates without ignoring new pairs.
    max_width_delta_per_update_m: float = 0.2  # Meters; limits one-frame width estimate jumps.
    min_trustworthy_pairs: int = 3  # Count; avoids updating width from too little pair evidence.

    # --- Output ---
    path_resolution_m: float = 0.5  # Meters; balances path smoothness with compact output size.
    max_path_length_m: float = 30.0  # Meters; bounds planner output to the controller horizon.

    # --- Validation ---
    jump_check_horizon_m: float = 8.0  # Meters; checks continuity where control response is immediate.
