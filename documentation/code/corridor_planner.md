# Corridor Planner Code Map

This page maps the `documentation/corridor_planner.md` behavior to the corridor planner node and the core corridor-construction algorithm.

## Primary Files

- `sim_car/sim_car/planning/corridor_planner_node.py`
- `sim_car/sim_car/planning/corridor_planner_core.py`
- `sim_car/sim_car/planning/tracked_cone_planner_contract.py`

## Function Map

### Runtime Flow

- `CorridorPlannerNode._on_timer` in `sim_car/sim_car/planning/corridor_planner_node.py`: top-level planning cycle that runs the corridor core, applies hold/update rules, and publishes outputs.
- `compute_corridor_centerline` in `sim_car/sim_car/planning/corridor_planner_core.py`: main algorithm that builds boundary chains, corridor samples, and a final centerline.
- `CorridorPlannerNode._select_candidate_centerline` in `sim_car/sim_car/planning/corridor_planner_node.py`: chooses the path that will actually be published when fresh and remembered geometry compete.

### Cone Filtering

- `_geometry_filter` in `sim_car/sim_car/planning/corridor_planner_core.py`: drops cones outside the usable planning region.
- `read_migrated_tracked_cone_planner_common_config` in `sim_car/sim_car/planning/tracked_cone_planner_contract.py`: loads the shared tracked-cone planner configuration used by the node.

### Boundary Chain Construction

- `_deterministic_order` in `sim_car/sim_car/planning/corridor_planner_core.py`: creates a stable cone ordering before chain construction.
- `_build_boundary_chain` in `sim_car/sim_car/planning/corridor_planner_core.py`: constructs the blue and yellow boundary chains used by corridor sampling.
- `_candidate_progresses_from_vehicle` and `_candidate_is_shadowed` in `sim_car/sim_car/planning/corridor_planner_core.py`: enforce forward progression and candidate shadowing rules.
- `CorridorPlannerNode._build_cone_audit_entries` in `sim_car/sim_car/planning/corridor_planner_node.py`: builds debug explanations for why cones were used or rejected.

### Corridor Sampling

- `_build_corridor` in `sim_car/sim_car/planning/corridor_planner_core.py`: samples overlapping left/right boundary geometry into corridor cross-sections.
- `_build_corridor_candidate` in `sim_car/sim_car/planning/corridor_planner_core.py`: evaluates one candidate cross-section series.
- `_corridor_valid_mask`, `_corridor_candidate_score`, and `_longest_valid_slice` in `sim_car/sim_car/planning/corridor_planner_core.py`: keep only the best valid overlapping corridor.
- `_corridor_pair_audit_reasons` in `sim_car/sim_car/planning/corridor_planner_core.py`: assigns the pair-audit reject reasons surfaced by the debug markers.

### Centerline Fitting

- `_fit_centerline_from_anchors` in `sim_car/sim_car/planning/corridor_planner_core.py`: converts accepted corridor center anchors into a continuous centerline.
- `_moving_average_points` and `_resample_boundary_by_station` in `sim_car/sim_car/planning/corridor_planner_core.py`: implement the smoothing and resampling used during centerline fitting.
- `_finalize_path` in `sim_car/sim_car/planning/corridor_planner_core.py`: produces the final publishable path from the anchor-derived centerline.

### Corridor Membership And Validation

- `_path_violates_corridor` in `sim_car/sim_car/planning/corridor_planner_core.py`: rejects fitted paths that leave the sampled corridor.
- `_near_field_delta_metrics`, `_path_alignment_metrics`, `_path_heading_delta_max`, and `_path_self_intersects` in `sim_car/sim_car/planning/corridor_planner_core.py`: implement the main path validation checks.
- `CorridorPlannerNode._candidate_path_is_updateable` and `_candidate_transition_metrics` in `sim_car/sim_car/planning/corridor_planner_node.py`: decide whether a fresh corridor path can replace the stored path.

### Debug Output

- `CorridorPlannerNode._build_markers` and `_append_corridor_pair_audit_markers` in `sim_car/sim_car/planning/corridor_planner_node.py`: publish corridor anchors, rungs, and pair-audit debug markers.
- `CorridorPlannerNode._publish_cone_audit_markers` in `sim_car/sim_car/planning/corridor_planner_node.py`: publishes the cone-audit view of boundary usage.

## Related Entry Points

- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: launches this planner when `planner:=corridor`.
- `update_track_width_estimate` in `sim_car/sim_car/planning/corridor_planner_core.py`: shared width-estimate update used when corridor geometry yields trustworthy width samples.
