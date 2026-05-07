# Corridor Planner Code Map

This page maps the `documentation/corridor_planner.md` behavior to the corridor planner node and the core corridor-construction algorithm.

## Primary Files

- `sim_car/sim_car/planning/tracked_cone_planner_node.py`
- `sim_car/sim_car/planning/corridor_planner_core.py`
- `sim_car/sim_car/planning/tracked_cone_planner_base.py`
- `sim_car/sim_car/planning/planning_diagnostics.py`
- `sim_car/sim_car/planning/planning_state_machine.py`
- `sim_car/sim_car/planning/planning_visualization.py`
- `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`
- `sim_car/sim_car/planning/tracked_cone_planner_contract.py`

## Function Map

### Runtime Flow

- `CorridorPlannerNode._on_timer` in `sim_car/sim_car/planning/tracked_cone_planner_node.py`: top-level planning cycle that runs the corridor core, applies hold/update rules, and publishes outputs.
- `compute_corridor_centerline` in `sim_car/sim_car/planning/corridor_planner_core.py`: main algorithm that builds boundary chains, corridor samples, and a final centerline.
- `CorridorPlannerNode._select_candidate_centerline` in `sim_car/sim_car/planning/tracked_cone_planner_node.py`: chooses the path that will actually be published when fresh and remembered geometry compete.
- `TrackedConePlannerBase` in `sim_car/sim_car/planning/tracked_cone_planner_base.py`: shared tracked-cone node runtime used for callbacks, TF utilities, controller execution, publishing, and path-memory support.
- `DiagnosticsMixin`, `StateMachineMixin`, and `VisualizationMixin`: shared diagnostic, hold/operator-state, and RViz marker helpers inherited by the base runtime.

### Cone Filtering

- `_geometry_filter` in `sim_car/sim_car/planning/corridor_planner_core.py`: drops cones outside the usable planning region.
- `read_migrated_tracked_cone_planner_common_config` in `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: loads the shared tracked-cone planner configuration used by the node.

### Boundary Chain Construction

- `_deterministic_order` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: creates a stable cone ordering before chain construction.
- `_build_boundary_chain` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: shared wrapper around boundary-chain construction.
- `build_boundary_chain_data`, `grow_boundary_chain_positions`, and `candidate_is_shadowed` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: implement the shared chain-growth, chain-heading progress, and shadowing rules.

### Corridor Sampling

- `_build_corridor` in `sim_car/sim_car/planning/corridor_planner_core.py`: samples overlapping left/right boundary geometry into corridor cross-sections.
- `_build_corridor_candidate` in `sim_car/sim_car/planning/corridor_planner_core.py`: evaluates one candidate cross-section series.
- `_corridor_valid_mask`, `_corridor_candidate_score`, and `_longest_valid_slice` in `sim_car/sim_car/planning/corridor_planner_core.py`: keep only the best valid overlapping corridor.

### Centerline Fitting

- `_fit_centerline_from_anchors` in `sim_car/sim_car/planning/corridor_planner_core.py`: converts accepted corridor center anchors into a continuous centerline.
- `_moving_average_points` and `_resample_boundary_by_station` in `sim_car/sim_car/planning/corridor_planner_core.py`: implement the smoothing and resampling used during centerline fitting.
- `_finalize_path` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: produces the final publishable path from the anchor-derived centerline.

### Corridor Membership And Validation

- `_path_violates_corridor` in `sim_car/sim_car/planning/corridor_planner_core.py`: rejects fitted paths that leave the sampled corridor.
- `_near_field_delta_metrics` and `_path_alignment_metrics` in `sim_car/sim_car/planning/corridor_planner_core.py`, plus `path_heading_delta_max` and `path_self_intersects` in `tracked_cone_planner_geometry.py`: implement the main path validation checks.
- `CorridorPlannerNode._candidate_path_is_updateable` and `_candidate_transition_metrics` in `sim_car/sim_car/planning/tracked_cone_planner_node.py`: decide whether a fresh corridor path can replace the stored path.

### Visualization Output

- `VisualizationMixin._build_markers` in `sim_car/sim_car/planning/planning_visualization.py`: publishes corridor boundaries, center anchors, rungs, centerline, lookahead, and status markers.

## Related Entry Points

- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: launches this planner when `planner:=corridor`.
- `update_track_width_estimate` and `pair_width_in_range` in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: shared width helpers used when corridor geometry yields trustworthy width samples.
