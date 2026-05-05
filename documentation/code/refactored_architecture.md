# Refactored Architecture Code Map

This page maps the architecture summary in `documentation/refactored_architecture.md` to the files that implement the current split.

## Planner Core And Shared Algorithm Files

- `sim_car/sim_car/planning/planner_config_base.py`: `BasePlannerConfig`, shared config fields inherited by the tracked-cone planner cores.
- `sim_car/sim_car/planning/planner_constants.py`: shared planner constants and operator state/reason code tables.
- `sim_car/sim_car/planning/planner_utils.py`: shared filtering, deterministic ordering, boundary-chain wrappers, path finalization, path metrics, and result-field helpers.
- `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`: reusable geometric primitives such as vehicle-frame transforms, path cumulative lengths, curvature/heading checks, self-intersection checks, boundary-chain growth, tangents, inward normals, and pair predicates.
- `sim_car/sim_car/planning/midpoint_planner_core.py`: midpoint-specific pair search, midpoint ordering, and validation.
- `sim_car/sim_car/planning/single_boundary_planner_core.py`: one-boundary chain selection, inward offset path generation, pair support, and validation.
- `sim_car/sim_car/planning/corridor_planner_core.py`: corridor sampling, rung validation, center-anchor fitting, corridor membership checks, and validation.

## Planner Runtime Files

- `sim_car/sim_car/planning/tracked_cone_planner_base.py`: shared tracked-cone ROS node utilities, callbacks, TF lookup, controller execution, path buffering, and common publishing flow.
- `sim_car/sim_car/planning/planning_state_machine.py`: hold/hysteresis behavior and operator state/reason helpers.
- `sim_car/sim_car/planning/planning_diagnostics.py`: diagnostic publishing and operator status formatting.
- `sim_car/sim_car/planning/planning_visualization.py`: RViz markers for centerlines, boundaries, pairs, corridor audit data, and planner status.
- `sim_car/sim_car/planning/controller_config.py`: Stanley and pure-pursuit config builders used by tracked-cone planners.
- `sim_car/sim_car/planning/pipeline_defaults.py`: shared topic, frame, QoS timeout, and planner input defaults.

## Planner Node Files

- `sim_car/sim_car/planning/midpoint_planner_node.py`: midpoint parameter loading, candidate selection, pair memory, and planner-cycle orchestration.
- `sim_car/sim_car/planning/single_boundary_planner_node.py`: single-boundary parameter loading, candidate selection, pair memory, and planner-cycle orchestration.
- `sim_car/sim_car/planning/corridor_planner_node.py`: corridor parameter loading, candidate selection, pair/corridor audit publishing, and planner-cycle orchestration.
- `sim_car/sim_car/planning/linetest_planner_node.py`: fixed-line controller test planner.

## Vehicle Plotter Files

- `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: logger node wiring, session handling, subscriptions, state logging, cone metrics, off-track autostop, and shutdown.
- `vehicle_plotter/vehicle_plotter/logging/log_config.py`: `LoggerNodeConfig` and one-pass parameter declaration/loading.
- `vehicle_plotter/vehicle_plotter/nodes/path_eval_runner.py`: path-evaluation callbacks, sampling timer, smalltrack lap counting, and final artifact generation.
- `vehicle_plotter/vehicle_plotter/nodes/plot_runner.py`: offline plot generation on shutdown.
- `vehicle_plotter/vehicle_plotter/logging/path_tracking_eval.py`: path-vs-ground-truth computation and summary files.
- `vehicle_plotter/vehicle_plotter/logging/path_tracking_eval_plots.py`: path tracking plot generation.
- `vehicle_plotter/vehicle_plotter/logging/track_metrics_report.py`: per-session track metrics reports.
- `vehicle_plotter/vehicle_plotter/analysis/analysis_utils.py`: shared CSV and path-analysis helpers.
