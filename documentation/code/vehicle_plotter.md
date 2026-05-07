# Vehicle Plotter Code Map

This page maps the `documentation/concepts/vehicle_plotter.md` behavior to the nodes and helpers that create run sessions, aggregate vehicle state, and write logs.

## Primary Files

- `vehicle_plotter/vehicle_plotter/nodes/session_manager_node.py`
- `vehicle_plotter/vehicle_plotter/nodes/plotter_node.py`
- `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`
- `vehicle_plotter/vehicle_plotter/nodes/path_eval_runner.py`
- `vehicle_plotter/vehicle_plotter/nodes/plot_runner.py`
- `vehicle_plotter/vehicle_plotter/core/run_session.py`
- `vehicle_plotter/vehicle_plotter/logging/log_config.py`
- `vehicle_plotter/vehicle_plotter/logging/path_tracking_eval.py`
- `vehicle_plotter/vehicle_plotter/logging/path_tracking_eval_plots.py`
- `vehicle_plotter/vehicle_plotter/logging/track_metrics_report.py`

## Function Map

### Session Management

- `SessionManagerNode` in `vehicle_plotter/vehicle_plotter/nodes/session_manager_node.py`: creates and republishes the shared run session for the rest of the stack.
- `SessionManagerNode._publish_session` in `vehicle_plotter/vehicle_plotter/nodes/session_manager_node.py`: broadcasts the current session on `/run_session`.
- `RunSession.create_new` in `vehicle_plotter/vehicle_plotter/core/run_session.py`: builds a fresh run ID and session directory layout.
- `RunSession.ensure_directories` in `vehicle_plotter/vehicle_plotter/core/run_session.py`: creates `logs/`, `plots/`, and `configs/`.

### State Aggregation And Live Dashboard

- `PlotterNode` in `vehicle_plotter/vehicle_plotter/nodes/plotter_node.py`: owns the live state publisher and dashboard refresh loop.
- `GazeboAdapter.setup_subscriptions` in `vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`: subscribes to measured sim topics for the plotter.
- `GazeboAdapter.compute_state` in `vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`: constructs the aggregated `VehicleState`.
- `PlotterNode._publish_state_callback` in `vehicle_plotter/vehicle_plotter/nodes/plotter_node.py`: publishes `/vehicle_plotter/state`.
- `PlotterNode._refresh_plots_callback` in `vehicle_plotter/vehicle_plotter/nodes/plotter_node.py`: updates the live plot buffers and dashboard outputs.

### Logging And Evaluation Artifacts

- `LoggerNode` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: run-artifact node that wires state logs, cone metrics, path-eval outputs, off-track autostop, and shutdown.
- `declare_and_load_config` and `LoggerNodeConfig` in `vehicle_plotter/vehicle_plotter/logging/log_config.py`: declare logger parameters once and build the typed logger config.
- `LoggerNode.state_callback` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: consumes `VehicleState` messages for logged state data.
- `LoggerNode._initialize_session` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: resolves or creates the active session directory before writing artifacts.
- `LoggerNode._setup_cone_subscriptions` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: subscribes to cone range RMSE sample streams.
- `LoggerNode._setup_path_tracking_eval_subscriptions` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: wires path-evaluation topics to the path-eval runner.
- `PathEvalRunner._path_tracking_eval_sample` and `_finalize_path_tracking_eval_outputs` in `vehicle_plotter/vehicle_plotter/nodes/path_eval_runner.py`: implement path-vs-ground-truth sampling and final reports/plots.
- `PlotRunner._generate_offline_plots` in `vehicle_plotter/vehicle_plotter/nodes/plot_runner.py`: renders end-of-run vehicle plot artifacts.
- `compare_planner_path_to_gt` and `analyze_path_tracking_csv` in `vehicle_plotter/vehicle_plotter/logging/path_tracking_eval.py`: compute path-evaluation metrics and summaries.
- `generate_path_tracking_cte_plot` and `generate_path_tracking_overlay_plot` in `vehicle_plotter/vehicle_plotter/logging/path_tracking_eval_plots.py`: render path tracking plots.
- `write_track_metrics_report` in `vehicle_plotter/vehicle_plotter/logging/track_metrics_report.py`: writes session-level track metrics records.

### Config Snapshots

- `RunArtifactsNode` in `sim_car/sim_car/run_artifacts_node.py`: copies config files and launch parameters into the run session.
- `RunArtifactsNode._copy_config_files` and `_write_launch_parameters` in `sim_car/sim_car/run_artifacts_node.py`: create the `configs/` snapshot.
- `copy_config_snapshot` in `sim_car/sim_car/run_artifacts_node.py`: helper used to copy matching config files.

## Related Entry Points

- `generate_launch_description` in `vehicle_plotter/launch/plotter.launch.py`: launch entry point for the session manager and plotter nodes.
- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: includes the plotter launch on normal full-sim runs.
