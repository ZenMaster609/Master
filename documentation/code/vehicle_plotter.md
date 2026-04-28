# Vehicle Plotter Code Map

This page maps the `documentation/vehicle_plotter.md` behavior to the nodes and helpers that create run sessions, aggregate vehicle state, write logs, and manage rosbags.

## Primary Files

- `vehicle_plotter/vehicle_plotter/nodes/session_manager_node.py`
- `vehicle_plotter/vehicle_plotter/nodes/plotter_node.py`
- `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`
- `vehicle_plotter/vehicle_plotter/core/run_session.py`

## Function Map

### Session Management

- `SessionManagerNode` in `vehicle_plotter/vehicle_plotter/nodes/session_manager_node.py`: creates and republishes the shared run session for the rest of the stack.
- `SessionManagerNode._publish_session` in `vehicle_plotter/vehicle_plotter/nodes/session_manager_node.py`: broadcasts the current session on `/run_session`.
- `RunSession.create_new` in `vehicle_plotter/vehicle_plotter/core/run_session.py`: builds a fresh run ID and session directory layout.
- `RunSession.ensure_directories` in `vehicle_plotter/vehicle_plotter/core/run_session.py`: creates `logs/`, `rosbags/`, `plots/`, and `configs/`.

### State Aggregation And Live Dashboard

- `PlotterNode` in `vehicle_plotter/vehicle_plotter/nodes/plotter_node.py`: owns the live state publisher and dashboard refresh loop.
- `GazeboAdapter.setup_subscriptions` in `vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`: subscribes to measured sim topics for the plotter.
- `GazeboAdapter.compute_state` in `vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`: constructs the aggregated `VehicleState`.
- `PlotterNode._publish_state_callback` in `vehicle_plotter/vehicle_plotter/nodes/plotter_node.py`: publishes `/vehicle_plotter/state`.
- `PlotterNode._refresh_plots_callback` in `vehicle_plotter/vehicle_plotter/nodes/plotter_node.py`: updates the live plot buffers and dashboard outputs.

### Logging And Evaluation Artifacts

- `LoggerNode` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: central run-artifact writer for state logs, diagnostics, path-eval outputs, and offline plots.
- `LoggerNode.state_callback` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: consumes `VehicleState` messages for logged state data.
- `LoggerNode._initialize_session` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: resolves or creates the active session directory before writing artifacts.
- `LoggerNode._setup_steering_diag_subscriptions` and `_steering_diag_sample` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: collect controller-tracking diagnostics.
- `LoggerNode._setup_path_tracking_eval_subscriptions`, `_path_tracking_eval_sample`, and `_finalize_path_tracking_eval_outputs` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: implement path-vs-ground-truth evaluation.
- `LoggerNode._generate_offline_plots` in `vehicle_plotter/vehicle_plotter/nodes/logger_node.py`: renders end-of-run plot artifacts.

### Rosbag Control

- `RosbagControllerNode` in `vehicle_plotter/vehicle_plotter/nodes/rosbag_controller_node.py`: manages the `ros2 bag record` subprocess for the active session.
- `RosbagControllerNode._load_topics_from_config` in `vehicle_plotter/vehicle_plotter/nodes/rosbag_controller_node.py`: reads the configured bag topic sets.
- `RosbagControllerNode._initialize_recording` and `_start_recording` in `vehicle_plotter/vehicle_plotter/nodes/rosbag_controller_node.py`: start recording inside the current session directory.
- `RosbagControllerNode._stop_recording` in `vehicle_plotter/vehicle_plotter/nodes/rosbag_controller_node.py`: shuts the rosbag process down cleanly.

### Config Snapshots

- `RunArtifactsNode` in `sim_car/sim_car/run_artifacts_node.py`: copies config files and launch parameters into the run session.
- `RunArtifactsNode._copy_config_files` and `_write_launch_parameters` in `sim_car/sim_car/run_artifacts_node.py`: create the `configs/` snapshot.
- `copy_config_snapshot` in `sim_car/sim_car/run_artifacts_node.py`: helper used to copy matching config files.

## Related Entry Points

- `generate_launch_description` in `vehicle_plotter/launch/plotter.launch.py`: launch entry point for the session manager, plotter, and rosbag controller nodes.
- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: includes the plotter launch on normal full-sim runs.
