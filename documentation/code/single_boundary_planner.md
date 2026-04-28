# Single-Boundary Planner Code Map

This page maps the `documentation/single_boundary_planner.md` behavior to the single-boundary planner node and core offset-path algorithm.

## Primary Files

- `sim_car/sim_car/planning/single_boundary_planner_node.py`
- `sim_car/sim_car/planning/single_boundary_planner_core.py`
- `sim_car/sim_car/planning/tracked_cone_planner_contract.py`

## Function Map

### Runtime Flow

- `SingleBoundaryPlannerNode._on_timer` in `sim_car/sim_car/planning/single_boundary_planner_node.py`: top-level planning cycle that runs the core algorithm and manages publish/hold behavior.
- `compute_single_boundary_centerline` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: main algorithm for producing a centerline from one or both visible boundaries.
- `SingleBoundaryPlannerNode._select_candidate_centerline` in `sim_car/sim_car/planning/single_boundary_planner_node.py`: chooses the publishable candidate path from fresh and remembered geometry.

### Cone Filtering

- `_geometry_filter` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: removes cones outside the useful planning envelope.
- `read_migrated_tracked_cone_planner_common_config` in `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: loads the shared filtering and controller settings used by this planner too.

### Boundary Chains

- `_deterministic_order` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: creates a stable candidate ordering.
- `_build_boundary_chain` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: constructs the blue or yellow candidate boundary chain.
- `_candidate_progresses_from_vehicle` and `_candidate_is_shadowed` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: implement forward-progress and shadowing checks during chain growth.

### Fallback Boundary Selection

- `_select_fallback_chain` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: chooses which visible boundary to trust when one-boundary planning is needed.
- `SingleBoundaryPlannerNode._normalize_core_reject_reason` in `sim_car/sim_car/planning/single_boundary_planner_node.py`: turns core status into operator-facing rejection reasons.

### Offset Path Generation

- `_offset_boundary_chain` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: offsets the selected boundary inward by the estimated track width.
- `_inward_normal` and `_estimate_tangents` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: compute the inward offset direction along the boundary chain.

### Track-Width Use

- `update_track_width_estimate` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: updates the planner’s filtered width estimate from trustworthy pair measurements.

### Validation And Hold Behavior

- `_finalize_path` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: smooths, resamples, and validates the candidate path before publication.
- `_path_self_intersects` and `_path_alignment_metrics` in `sim_car/sim_car/planning/single_boundary_planner_core.py`: implement major path validation checks.
- `SingleBoundaryPlannerNode._candidate_path_is_updateable` and `_candidate_transition_metrics` in `sim_car/sim_car/planning/single_boundary_planner_node.py`: decide when a new path may replace the held path.
- `SingleBoundaryPlannerNode._update_midline_buffer` in `sim_car/sim_car/planning/single_boundary_planner_node.py`: maintains recent path memory used for smooth transitions.

## Related Entry Points

- `SingleBoundaryPlannerNode._publish_empty_cycle` in `sim_car/sim_car/planning/single_boundary_planner_node.py`: handles the planner’s no-path cycle output.
- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: launches this planner when `planner:=single_boundary`.
