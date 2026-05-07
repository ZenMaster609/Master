# Midpoint Planner Code Map

This page maps the `documentation/midpoint_planner.md` behavior to the midpoint planner node and its core geometric algorithm.

## Primary Files

- `sim_car/sim_car/planning/tracked_cone_planner_node.py`
- `sim_car/sim_car/planning/midpoint_planner_core.py`
- `sim_car/sim_car/planning/tracked_cone_planner_base.py`
- `sim_car/sim_car/planning/planning_diagnostics.py`
- `sim_car/sim_car/planning/planning_state_machine.py`
- `sim_car/sim_car/planning/planning_visualization.py`
- `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`
- `sim_car/sim_car/planning/tracked_cone_planner_contract.py`

## Function Map

### Runtime Flow

- `MidpointPlannerNode._on_timer` in `sim_car/sim_car/planning/tracked_cone_planner_node.py`: top-level planning cycle that reads tracked cones, runs the core planner, manages hold/update logic, and publishes outputs.
- `compute_midpoint_centerline` in `sim_car/sim_car/planning/midpoint_planner_core.py`: main midpoint-planning algorithm from filtered boundary cones to centerline candidate.
- `MidpointPlannerNode._select_candidate_centerline` in `sim_car/sim_car/planning/tracked_cone_planner_node.py`: chooses between fresh, held, and fallback candidate paths before publication.
- `TrackedConePlannerBase` in `sim_car/sim_car/planning/tracked_cone_planner_base.py`: shared tracked-cone node runtime used for callbacks, TF utilities, controller execution, publishing, and path-memory support.
- `DiagnosticsMixin`, `StateMachineMixin`, and `VisualizationMixin`: shared diagnostic, hold/operator-state, and RViz marker helpers inherited by the base runtime.

### Cone Filtering

- `_geometry_filter` in `sim_car/sim_car/planning/midpoint_planner_core.py`: removes cones that violate range, horizon, behind-car, or finite-geometry constraints.
- `MidpointPlannerNode._tentative_cone_is_usable_for_planning` in `sim_car/sim_car/planning/tracked_cone_planner_node.py`: gate for whether tentative tracks may contribute to planning.
- `read_migrated_tracked_cone_planner_common_config` in `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: loads the shared filtering and planner-common parameters.

### Boundary Chains

- `_deterministic_order` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: sorts cones consistently before chain building.
- `_build_boundary_chain` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: shared wrapper around boundary-chain construction.
- `build_boundary_chain_data`, `grow_boundary_chain_positions`, and `candidate_is_shadowed` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: implement the shared chain-growth, chain-heading progress, and shadowing rules.

### Pair Creation

- `_pair_boundary_chains` in `sim_car/sim_car/planning/midpoint_planner_core.py`: searches for valid left/right boundary pairs.
- `pair_width_in_range`, `inward_distance`, `midpoint_outside_pair_span`, `unknown_partner_check`, and `unknown_partner_within_limits` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: shared pair predicates used inside midpoint pair creation.
- `MidpointPlannerNode._active_pair_memory`, `_pair_geometry_from_memory`, and `_remember_pairs` in `sim_car/sim_car/planning/tracked_cone_planner_node.py`: stabilize pair choices across frames and avoid rapid reassignment.
- `_trim_pairs_by_midpoint_step_length` in `sim_car/sim_car/planning/midpoint_planner_core.py`: removes implausible jumps between consecutive pair-derived midpoints.

### Midpoint Ordering

- `_order_pairs_into_midpoint_chain` in `sim_car/sim_car/planning/midpoint_planner_core.py`: orders accepted pairs into a forward midpoint chain.
- `_midpoint_progress_reference` in `sim_car/sim_car/planning/midpoint_planner_core.py`: computes the progress reference used to order ambiguous nearby midpoints.
- `MidpointPlannerNode._update_midline_buffer` in `sim_car/sim_car/planning/tracked_cone_planner_node.py`: maintains the recent-path buffer used to stabilize updates.

### Width Estimate

- `update_track_width_estimate` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: updates the filtered track-width prior from available pair measurements.

### Centerline Finalization

- `_finalize_path` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: smooths, resamples, and trims the midpoint chain into the published centerline.
- `_resample_path` and `moving_average` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: implement the main path-shaping steps used by `_finalize_path`.
- `MidpointPlannerNode._candidate_path_is_updateable` in `sim_car/sim_car/planning/tracked_cone_planner_node.py`: decides whether a candidate path is stable enough to replace the held path.

## Related Entry Points

- `DiagnosticsMixin._publish_empty_cycle` in `sim_car/sim_car/planning/planning_diagnostics.py`: publishes the planner's no-path/hold behavior.
- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: launches this planner when `planner:=midpoint`.
