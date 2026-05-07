# Refactored Architecture

This page summarizes the current refactoring boundaries in `sim_car` and `vehicle_plotter`. It is a reader map: use the behavior pages for what the stack does, and this page for where responsibilities now live.

## Planning Package

The tracked-cone planners are split into four layers:

- Planner cores: `midpoint_planner_core.py`, `single_boundary_planner_core.py`, and `corridor_planner_core.py` contain the algorithm-specific centerline construction. They do not own ROS subscriptions, publishers, or timers.
- Shared geometry and algorithm helpers: `tracked_cone_planner_geometry.py` contains reusable geometry primitives plus the shared cone filtering, deterministic ordering, boundary-chain wrappers, path finalization, and result-field helpers used by the planner cores.
- Shared runtime: `tracked_cone_planner_base.py` owns ROS-facing planner utilities and inherits `DiagnosticsMixin`, `VisualizationMixin`, and `StateMachineMixin`.
- Planner node entry points: `tracked_cone_planner_node.py` now contains `MidpointPlannerNode`, `CorridorPlannerNode`, and `SingleBoundaryPlannerNode`. The installed console scripts keep their old executable names (`midpoint_planner_node`, `corridor_planner_node`, and `single_boundary_planner_node`) but dispatch to this shared module.

Shared planner constants now live in `planner_constants.py`. This includes cone track-state message codes, validated-jump acceptance limits, shared marker widths, pair-pass margins, and operator state/reason codes.

Shared planner config fields now live in `BasePlannerConfig` in `planner_config_base.py`. `CorridorPlannerConfig`, `MidpointPlannerConfig`, and `SingleBoundaryPlannerConfig` subclass it and add only planner-specific fields.

Common default topics and frame names live in `pipeline_defaults.py`. Controller parameter builders live in `controller_config.py`.

The refactor keeps the old grouped planner parameter contract as the public tuning surface. `tracked_cone_planner_contract.py` declares shared groups such as `filtering.*`, `boundary_chain.*`, `width_estimation.*`, `centerline.*`, `validation.*`, `midline_memory.*`, controller groups, and debug flags, while each planner node reads its planner-specific groups into its typed core config.

## Planner Runtime Mixins

The shared tracked-cone runtime is split by concern:

- `planning_state_machine.py`: operator state, hold hysteresis, last-valid-path behavior, and state/reason labels.
- `planning_diagnostics.py`: diagnostic arrays, operator status text, and metric formatting.
- `planning_visualization.py`: RViz marker construction for centerlines, boundaries, pairs, corridor rungs, and status.
- `tracked_cone_planner_base.py`: TF lookup, frame aliases, cone/odom callbacks, controller execution, path buffering, and common node wiring.

The individual planner nodes still keep planner-specific memory and candidate-selection logic when their data shapes differ. For example, midpoint and corridor pair-memory entries are intentionally not merged because they track different metadata.

## Vehicle Plotter

`vehicle_plotter` now separates logging, path evaluation, plotting, and config loading:

- `nodes/logger_node.py`: wires ROS subscriptions, sessions, state logging, cone metrics, off-track autostop, and shutdown.
- `logging/log_config.py`: declares and reads logger parameters in one pass, then builds `LoggerNodeConfig`.
- `nodes/path_eval_runner.py`: owns path-vs-ground-truth sampling, smalltrack lap counting, and path-eval finalization.
- `nodes/plot_runner.py`: owns offline plot generation from a completed run session.
- `logging/path_tracking_eval.py`: contains path-evaluation geometry and summary metrics.
- `logging/path_tracking_eval_plots.py`: renders path tracking CTE and overlay plots.
- `logging/track_metrics_report.py`: writes per-session track metric records.
- `analysis/analysis_utils.py`: shared CSV and path-analysis helpers used by plotting/diagnostic code.

The old standalone steering and thesis-controller diagnostic outputs are no longer active logger artifacts in this branch. Current logger diagnostics are vehicle-state chunks, cone range RMSE samples/plots, path tracking evaluation, track metrics reports, and off-track autostop based on planner diagnostics.
