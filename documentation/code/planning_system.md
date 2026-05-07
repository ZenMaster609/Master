# Planning System Code Map

This page maps `documentation/concepts/planning_system.md` to the refactored planning source files.

## Primary Files

- `sim_car/launch/full_sim_launch.launch.py`
- `sim_car/sim_car/planning/planner_constants.py`
- `sim_car/sim_car/planning/tracked_cone_planner_contract.py`
- `sim_car/sim_car/planning/tracked_cone_planner_base.py`
- `sim_car/sim_car/planning/tracked_cone_planner_node.py`
- `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`
- `sim_car/sim_car/planning/midpoint_planner_core.py`
- `sim_car/sim_car/planning/single_boundary_planner_core.py`
- `sim_car/sim_car/planning/corridor_planner_core.py`
- `sim_car/sim_car/planning/midline_memory.py`
- `sim_car/sim_car/planning/planning_state_machine.py`
- `sim_car/sim_car/planning/planning_diagnostics.py`
- `sim_car/sim_car/planning/planning_visualization.py`
- `sim_car/sim_car/planning/linetest_planner_node.py`
- `sim_car/sim_car/planning/skidpad_router_node.py`
- `sim_car/sim_car/planning/skidpad_router_core.py`

## Launch Selection

- `PLANNER_REGISTRY` in `planner_constants.py`: maps `midpoint`, `single_boundary`, `corridor`, `linetest`, and `none` to executable names, diagnostics topics, default RViz profiles, and allowed tracks.
- `get_planner_spec` in `planner_constants.py`: resolves the launch-time planner name.
- `planner_allowed_for_track` in `planner_constants.py`: rejects invalid combinations such as `linetest` on skidpad.
- `_resolve_launch_selection` in `full_sim_launch.launch.py`: resolves the track, planner, controller, world, spawn defaults, controller overlays, and generated parameter overlay.
- `planner_input_topic` in `full_sim_launch.launch.py`: selects `/tracked_cones`, `/tracked_cones/skidpad_routed`, or direct perception input depending on cone memory, track, and routing choices.

## Shared Constants And Runtime Types

- `planner_constants.py`: holds message track-state codes, operator state/reason code tables, validated-jump limits, shared marker widths, pair-pass margins, topic/frame defaults, and planner launch specs.
- `PlannerIdentity` in `planner_constants.py`: gives each tracked-cone node its node name, planner mode, diagnostics prefix, and diagnostics topic.
- `TrackedConePlanningFrame` in `planner_constants.py`: packages transformed points, colors, confidences, boundary hints, track IDs, track states, and track confidences for core planner calls.

## Parameter Contract And Controllers

- `PUBLIC_TRACKED_CONE_PLANNER_DEFAULTS` in `tracked_cone_planner_contract.py`: shared public parameter surface for tracked-cone planners.
- `declare_tracked_cone_planner_parameters` in `tracked_cone_planner_contract.py`: declares shared parameters on each tracked-cone node.
- `read_migrated_tracked_cone_planner_common_config` in `tracked_cone_planner_contract.py`: reads shared runtime, filtering, controller, speed, debug, and midline-memory settings.
- `planner_algorithm_profile` in `tracked_cone_planner_contract.py`: converts public grouped parameters into algorithm profile values and applies compatibility aliases from `planner.*`.
- `apply_common_config_to_node` in `tracked_cone_planner_contract.py`: writes common config fields onto the node instance.
- `build_stanley_config`, `build_pure_pursuit_config`, and `build_steering_controller` in `tracked_cone_planner_contract.py`: read controller parameter groups and create the selected controller.

## Shared Tracked-Cone Runtime

- `GenericTrackedConePlannerNode` in `tracked_cone_planner_node.py`: shared tracked-cone node skeleton. It declares common parameters, reads common config, calls planner-specific parameter readers, builds the core config, initializes state, and wires ROS interfaces.
- `TrackedConePlannerBase` in `tracked_cone_planner_base.py`: common ROS-facing runtime used by midpoint, single-boundary, and corridor.
- `TrackedConePlannerBase._init_common_ros_interfaces`: creates publishers for `/cmd`, centerline path, RViz markers, optional point arrays, and diagnostics; subscribes to tracked cones and odometry; starts the planning timer.
- `TrackedConePlannerBase._resolve_cone_planning_context`: shared timer preamble. It handles waiting for cones, resolving frames and vehicle pose, transforming cones, fallback to odom, and empty-cycle publication on failure.
- `TrackedConePlannerBase._publish_outputs`: publishes the path, optional point array, and RViz markers.
- `TrackedConePlannerBase._publish_diagnostics`: emits the diagnostic array with planner, control, hold, and operator-state metrics.
- `TrackedConePlannerBase._build_steering_controller`: delegates to the controller factory in the parameter contract.

## Geometry Helpers

- `tracked_cone_planner_geometry.py`: shared geometry utilities for transforms, vehicle-frame conversion, deterministic ordering, filtering, chain growth, path cumulative lengths, resampling, moving average, heading and curvature checks, self-intersection checks, path finalization, and width helpers.
- `build_boundary_chain_data`, `grow_boundary_chain_positions`, and `candidate_is_shadowed`: build side-boundary chains from filtered cones.
- `_finalize_path`: smooths, resamples, and trims candidate paths into publishable centerlines.
- `update_track_width_estimate`: updates the filtered width prior used by midpoint and single-boundary planning.
- `extract_forward_path_from_pose`, `sample_path_at_lengths`, and `resample_to_count`: shared path-memory and control-path sampling helpers.

## Planner Cores

- `MidpointPlannerConfig`, `MidpointPlannerPrior`, `MidpointPlannerResult`, and `compute_midpoint_centerline` in `midpoint_planner_core.py`: two-boundary pairing and midpoint centerline algorithm.
- `SingleBoundaryPlannerConfig`, `SingleBoundaryPlannerPrior`, `SingleBoundaryPlannerResult`, and `compute_single_boundary_centerline` in `single_boundary_planner_core.py`: one-boundary offset fallback algorithm with optional pair support.
- `CorridorPlannerConfig`, `CorridorPlannerPrior`, `CorridorPlannerResult`, and `compute_corridor_centerline` in `corridor_planner_core.py`: strict two-boundary corridor sampling and fitted centerline algorithm.
- `BasePlannerConfig` in `planner_config_base.py`: shared inherited fields for the core config dataclasses.

## Planner Node Classes

- `MidpointPlannerNode` in `tracked_cone_planner_node.py`: declares midpoint-specific filtering, pairing, width, centerline, validation, and debug parameters. Its `_on_timer` builds the planning frame, calls `compute_midpoint_centerline`, updates pair memory and width estimate, updates midline memory, runs control, and publishes outputs.
- `SingleBoundaryPlannerNode` in `tracked_cone_planner_node.py`: declares one-boundary, pairing, offset, width, centerline, and validation parameters. Its `_on_timer` calls `compute_single_boundary_centerline`, manages one-boundary pair memory, selects fresh or held paths, and publishes debug offset geometry.
- `CorridorPlannerNode` in `tracked_cone_planner_node.py`: declares corridor width, resampling, membership, fitting, pair-memory, and validation parameters. Its `_on_timer` calls `compute_corridor_centerline`, merges live and remembered corridor pair geometry, selects the publishable path, and publishes corridor audit markers.
- `main_midpoint`, `main_single_boundary`, and `main_corridor` in `tracked_cone_planner_node.py`: console-script entry points used by the installed executables.

## State, Diagnostics, And Visualization

- `CommittedMidlineMemory` in `midline_memory.py`: stores a committed forward path, blends accepted candidates into it, rejects near-field jumps, and serves held paths while memory is still valid.
- `StateMachineMixin` in `planning_state_machine.py`: owns `waiting`, `fresh`, `held`, and `stopped` behavior, no-path command handling, last-valid hold, hold hysteresis, operator reason labels, and state/reason codes.
- `DiagnosticsMixin` in `planning_diagnostics.py`: formats operator status text and publishes diagnostic metrics for path quality, hold behavior, selected cones, controller output, and failure reasons.
- `VisualizationMixin` in `planning_visualization.py`: builds RViz marker arrays for remembered cones, graph edges, boundaries, pairs, raw paths, final centerline, lookahead target, and status text.

## Linetest Planner

- `LineTestPlannerNode` in `linetest_planner_node.py`: standalone fixed-line planner for controller testing. It declares and reads its own parameters, subscribes to odometry, generates a fixed path or GT-derived path, converts the forward segment into the vehicle frame, runs the shared steering-controller factory, publishes `/cmd`, and handles end-of-line braking.
- `LineTestPlannerNode._on_timer`: main loop for line projection, control-path extraction, controller execution, diagnostics, path publication, and brake publication.
- `LineTestPlannerNode._build_steering_controller`: uses `build_steering_controller` from `tracked_cone_planner_contract.py`.

## Skidpad Router

- `SkidpadRouterNode` in `skidpad_router_node.py`: ROS node that subscribes to `/tracked_cones` and `/sim/odom`, filters cones to the active skidpad or acceleration branch, republishes routed cones, publishes route diagnostics, publishes route markers, and can publish stop/brake commands.
- `SkidpadStateMachine` in `skidpad_router_core.py`: mission-stage state machine for approach, right/left lobes, straight, and parked states.
- `SkidpadStateMachine.route_mask`: masks cone points to the active route branch.
- `detect_stop_line_pair` and `detect_acceleration_stop_row` in `skidpad_router_core.py`: find parking/acceleration stop geometry.
- `boundary_color_from_lateral_y` in `skidpad_router_core.py`: assigns planning-side boundary colors from lateral position for routed cones.

## Where To Edit

- Add a shared tracked-cone parameter: `tracked_cone_planner_contract.py`.
- Add a planner-specific parameter: the relevant class in `tracked_cone_planner_node.py`.
- Change midpoint/single-boundary/corridor algorithm behavior: the matching `*_planner_core.py`.
- Change shared geometry or path validation primitives: `tracked_cone_planner_geometry.py`.
- Change hold behavior or operator states: `planning_state_machine.py` and `planner_constants.py`.
- Change diagnostics fields: `planning_diagnostics.py`.
- Change RViz marker content: `planning_visualization.py`.
- Change launch planner names, allowed tracks, or default diagnostics topics: `planner_constants.py` and `full_sim_launch.launch.py`.
- Change controller parameter loading: `tracked_cone_planner_contract.py`.
- Change fixed-line controller testing behavior: `linetest_planner_node.py`.
- Change skidpad or acceleration cone routing: `skidpad_router_node.py` and `skidpad_router_core.py`.
