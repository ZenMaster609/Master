# Vehicle Plotter

`vehicle_plotter` is the run-artifact package for the simulation stack. It creates run sessions, publishes aggregated vehicle state, writes logs, starts rosbag recording, and generates cone, controller, and path-evaluation artifacts under `multidata/`.

It does not create sensor values. Sensor values come from Gazebo and `sim_car`; `vehicle_plotter` consumes measured simulation topics plus planner/controller outputs.

## Runtime Pieces

### `session_manager_node`

Creates a `RunSession`, publishes it on `/run_session`, and keeps broadcasting it so other nodes share one run directory.

Default layout:

`multidata/<run_id>/`

Session subdirectories:

- `logs/`: parquet/CSV logs, summaries, and text reports
- `rosbags/`: rosbag recordings
- `plots/`: generated PNG plots
- `configs/`: copied YAML configs and launch-parameter snapshots

`full_sim_launch.launch.py` passes a compact run-id prefix built from track, planner, controller, and LiDAR pipeline. Example:

`small_mid_pp_3d_YYYY-MM-DD_HH-MM-SS`

### `plotter_node`

Subscribes to measured `/sim/...` topics through the Gazebo adapter, synchronizes samples, builds `VehicleState`, publishes `/vehicle_plotter/state`, and updates the live dashboard buffers.

On shutdown it can export:

`plots/virtual_sensors.png`

In the full sim launch, `plotter_node` starts when state logging/dashboard support is enabled. That happens when either `sensor_pipeline:=true` or `logging:=true`.

### `logger_node`

Writes run artifacts. In `full_sim_launch.launch.py`, the direct `logger_node` is always launched so path tracking evaluation and diagnostics can run even when the live sensor dashboard is disabled.

State logging is separate from the logger process:

- `sensor_pipeline:=true` or `logging:=true` enables `/vehicle_plotter/state` subscription and vehicle-state chunks.
- `controller_diagnostics:=true` enables steering-tracking diagnostics.
- `thesis_controller_diagnostics:=true` enables the wider thesis-oriented controller diagnostics.
- `path_tracking_eval:=true` enables path-vs-ground-truth evaluation and is true by default.

The logger waits for `/run_session` when available. If no session arrives before timeout, it creates its own session.

### `rosbag_controller_node`

Starts a `ros2 bag record` subprocess for the active run session when `rosbagging:=true`. It waits for `/run_session` by default and writes to:

`multidata/<run_id>/rosbags/bag`

Topic selection comes from `vehicle_plotter/config/rosbag_topics.yaml` unless the node receives an explicit `topics` parameter. The launch include sets the rosbag controller `mode` parameter to `simulation`, so the recorded set is `common` plus `simulation` from that YAML.

## Full Sim Launch Behavior

`sim_car/launch/full_sim_launch.launch.py` includes `vehicle_plotter/launch/plotter.launch.py` every run. The include starts the session manager, can start `plotter_node`, and can start the rosbag controller.

Important launch flags:

- `sensor_pipeline:=true`: starts raw sensor nodes, `measurement_node`, and `plotter_node`.
- `logging:=true`: enables state logging/dashboard support even without the full sensor pipeline.
- `rosbagging:=true`: enables `rosbag_controller_node`.
- `controller_diagnostics:=true`: writes steering-tracking CSV, summary, and plot artifacts.
- `thesis_controller_diagnostics:=true`: writes thesis controller diagnostics and corridor oscillation artifacts when applicable.
- `path_tracking_eval:=true`: writes path tracking evaluation artifacts; this is the current full-sim default.

The old standalone live cone RMSE and controller-diagnostics windows are not current console scripts. Cone RMSE and controller diagnostics are now logger outputs and generated plots.

## Data Flow

Measured sensor state path:

`raw sensor nodes -> measurement_node -> measured /sim topics -> plotter_node -> /vehicle_plotter/state -> logger_node`

Diagnostics/evaluation path:

`planner/controller/cone evaluator topics -> logger_node -> logs + plots`

Rosbag path:

`/run_session -> rosbag_controller_node -> multidata/<run_id>/rosbags/bag`

When `measure:=true` or `sensor_pipeline:=true`, perception and LiDAR prefixes switch to `/sim/raw/...` before measurement. The plotter dashboard consumes measured `/sim/...` values.

## Main Topics

Inputs used by `plotter_node` through the Gazebo adapter include measured odometry, wheel encoder, suspension, steering, cooling, brake, and pitot topics from `/sim/...`.

Important package topics:

- `/run_session`: `vehicle_plotter_msgs/RunSession`
- `/vehicle_plotter/state`: `vehicle_plotter_msgs/VehicleState`
- `/cmd`: Ackermann command used by diagnostics
- `/planned_centerline`: planner path used by diagnostics and path tracking evaluation
- `/ground_truth/track`: ground-truth cones for path evaluation
- `/sim/stereo/eval` and `/sim/lidar/eval`: cone evaluator prefixes, or `/sim/raw/...` when measurement prefixing is active

## Artifact Outputs

Vehicle-state logging can write:

- `logs/vehicle_state_0000.parquet`
- CSV vehicle-state chunks when `log_format:=csv`
- `logs/metadata.json`
- offline vehicle plots generated on shutdown

Cone RMSE logging can write source-specific files:

- `logs/cone_range_rmse_samples_mono.csv`
- `logs/cone_range_rmse_samples_stereo.csv`
- `logs/cone_range_rmse_samples_lidar.csv`
- `plots/cone_range_rmse_mono.png`
- `plots/cone_range_rmse_stereo.png`
- `plots/cone_range_rmse_lidar.png`

Only sources with received samples produce files.

Controller diagnostics can write:

- `logs/steering_tracking_diagnostics.csv`
- `logs/steering_tracking_summary.json`
- `logs/steering_tracking_summary.txt`
- `plots/stanley_debug_plots.png`

Thesis controller diagnostics can write:

- `logs/thesis_controller_diagnostics.csv`
- `logs/thesis_controller_diagnostics_summary.json`
- `logs/thesis_controller_diagnostics_summary.txt`
- `plots/thesis_controller_diagnostics.png`
- corridor oscillation summaries and plots when corridor diagnostics are present

Path tracking evaluation can write:

- `logs/path_tracking_eval.csv`
- `logs/path_tracking_eval_summary.json`
- `logs/path_tracking_eval_summary.txt`
- `logs/path_tracking_eval_track_metrics.json`
- `logs/path_tracking_eval_track_metrics.txt`
- `plots/path_tracking_eval_cte.png`
- `plots/path_tracking_eval_overlay.png`

## Useful Commands

Build:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select vehicle_plotter vehicle_plotter_msgs sim_car
```

Source:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Launch the full measured sensor dashboard path:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py sensor_pipeline:=true
```

Launch with state logging and rosbagging:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py logging:=true rosbagging:=true
```

Launch with controller diagnostics:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py logging:=true controller_diagnostics:=true
```

Run vehicle_plotter tests:

```bash
cd ~/ros2_ws && colcon test --packages-select vehicle_plotter vehicle_plotter_msgs
cd ~/ros2_ws && colcon test-result --verbose
```

## Prefix Path Warnings

Warnings about missing old packages in `AMENT_PREFIX_PATH` or `CMAKE_PREFIX_PATH` mean the shell has stale sourced workspace paths.

Reset and source again:

```bash
cd ~/ros2_ws && unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
cd ~/ros2_ws && source /opt/ros/humble/setup.bash
cd ~/ros2_ws && source install/setup.bash
```
