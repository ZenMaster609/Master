Phase 1 — Dead safe, do it all at once
Zero logic changes. Pure extraction.


Batch 1A — Extract shared constants
  sim_car/sim_car/planning/planner_constants.py  (new)
    ← _VALIDATED_JUMP_ACCEPT_* from all 4 planner node files
    ← _OPERATOR_STATE/REASON_CODES from runtime + linetest
    ← MSG_TRACK_STATE_* from all 3 nodes + base
    ← _CENTERLINE_MARKER_WIDTH_M, _PAIR_PASSED_MARGIN_M

  Touch: corridor_planner_node.py, midpoint_planner_node.py,
         single_boundary_planner_node.py, linetest_planner_node.py,
         tracked_cone_planner_runtime.py, tracked_cone_planner_base.py

Batch 1B — Consolidate shared geometry utilities
  tracked_cone_planner_geometry.py  (already exists, expand it)
    ← _to_vehicle_frame, _path_curvature_abs_max, _path_heading_delta_max
    ← _moving_average, _path_self_intersects, _segments_intersect
    ← _path_cumulative_lengths
    (delete copies from corridor_, midpoint_, single_boundary_ *_core.py)

  Touch: the 3 *_core.py files + tracked_cone_planner_geometry.py

Batch 1C — Consolidate vehicle_plotter shared stats
  vehicle_plotter/vehicle_plotter/analysis/analysis_utils.py  (new)
    ← CSV time-alignment logic duplicated across:
       steering_diagnostics.py, thesis_controller_diagnostics.py,
       stanley_debug_plots.py
    ← nearest_point_on_polyline, signed_cross_track_error
       (both packages)

  Touch: 3 diagnostics files + path_tracking_eval.py
Each batch is a single PR. Tests should pass unchanged after each.

Phase 2 — Config consolidation
Only dataclass structure changes. No algorithm changes.


Batch 2A — BasePlannerConfig
  sim_car/sim_car/planning/planner_config_base.py  (new)
    Extract ~30 shared fields from CorridorPlannerConfig,
    MidpointPlannerConfig, SingleBoundaryPlannerConfig into
    BasePlannerConfig. Subclass from it.

  Touch: corridor_planner_core.py, midpoint_planner_core.py,
         single_boundary_planner_core.py

Batch 2B — Sensor node factory
  sim_car/sim_car/sensors/simple_sensor_node.py  (new)
    Single parametric class replacing:
    water_temp_in_node.py, water_temp_out_node.py,
    brake_temp_fr_node.py, brake_temp_rl_node.py,
    water_flow_node.py, water_pressure_node.py,
    pitot_pressure_node.py

  Touch: 7 sensor files + nodes.launch.py or equivalent launch wiring

Batch 2C — logger_node.py config extraction
  vehicle_plotter/vehicle_plotter/log_config.py  (already exists, extend)
    Move all 163 declare_parameter + get_parameter calls into a
    declare_and_load_config() helper.
    logger_node.__init__ stores self._cfg, not 163 variables.

  Touch: logger_node.py, log_config.py
Phase 3 — File splitting (highest risk, do one file per session)

Batch 3A — Split tracked_cone_planner_runtime.py (2504 lines)
  Keep in runtime:  ROS wiring only (subs, pubs, timers, callbacks)
  Extract to:
    planning_diagnostics.py   ← diagnostics aggregation logic
    planning_visualization.py ← marker array / rviz publishing
    planning_state_machine.py ← operator state / reason code transitions

  Rule: do NOT touch algorithm logic in this batch.

Batch 3B — Split logger_node.py (2132 lines)
  Keep in node:     ROS wiring + session lifecycle
  Extract to:
    log_diagnostics_runner.py  ← steering/thesis/corridor diagnostics
    path_eval_runner.py        ← path tracking evaluation
    plot_runner.py             ← matplotlib figure generation

  Rule: do NOT touch analysis logic in this batch.
Phase 4 — Algorithm cleanup (one file per session)

One session per *_core.py file:
  - Split functions > 40 lines
  - Replace dict bundles with dataclasses
  - Unify rejection/failure reporting pattern

Order: single_boundary_planner_core.py first (simplest, 1144 lines),
       then midpoint_planner_core.py (1324),
       then corridor_planner_core.py (1390)

Do not touch the corresponding *_node.py in the same session.
Sizing Guide for a Single Session
Change type	Max files to touch	Max new lines to write
Extract constants / move functions	8–10 files	< 200
Consolidate config dataclasses	3–4 files	< 300
Split a large file	1 source + 2–3 new files	< 500
Clean up one algorithm file	1 file	< 600
If a session needs more than that, the batch is too big — split it.

Sequencing Rule
Always do Phase 1 before Phase 3. Splitting large files while they still contain duplicated constants makes the constants harder to find and delete. Clean the duplication first so each file is smaller before you split it.

Using the rules in CLAUDE.md and the plan in REFACTOR_PLAN.md, execute Batch 1A.

- Do not change any algorithm logic, only move/consolidate code
- After each file change, confirm the import chain still resolves
- When finished, list every file modified and every file created
- Run `python -m pytest sim_car/tests/ -x -q` and report the result
- Stop. Do not start Batch 1B.
"Ignore line numbers in the plan — find the relevant code yourself before moving it."
