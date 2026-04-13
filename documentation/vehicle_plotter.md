# Vehicle Plotter

`vehicle_plotter` owns run sessions, live virtual-sensor plotting, vehicle-state logging, rosbag control, cone RMSE artifacts, controller diagnostics, and path-tracking evaluation.

It does not generate the sensor values. It consumes measured simulation topics and planner/controller outputs, then turns them into runtime dashboards and files under `multidata/`.

## Main Runtime Pieces

### `session_manager_node`

Creates a run session and publishes it on `/run_session`. Other nodes use this to agree on one run directory.

Default layout:

`multidata/<run_id>/`

Session subdirectories:

- `logs/`: parquet/CSV logs and summaries
- `rosbags/`: rosbag recordings
- `plots/`: generated PNG plots
- `configs/`: copied launch/config snapshots

### `plotter_node`

Aggregates measured `/sim/...` topics through the Gazebo adapter, builds `VehicleState`, publishes `/vehicle_plotter/state`, and maintains live dashboard buffers.

On shutdown, it can export:

`plots/virtual_sensors.png`

This node starts only when state logging/dashboard support is enabled by the launch include. In `full_sim_launch.launch.py`, `sensor_pipeline:=true` enables the plotter launch path.

### `logger_node`

Writes run artifacts. It can subscribe to `/vehicle_plotter/state` and write vehicle-state logs in parquet or CSV format. In the full sim launch, it is also started directly for diagnostics and evaluation artifacts.

Current logger responsibilities include:

- `vehicle_state_0000.parquet` or CSV vehicle-state chunks
- `metadata.json`
- cone range-RMSE sample CSVs
- steering tracking diagnostics
- thesis controller diagnostics
- path tracking evaluation CSVs and summaries
- offline plots on shutdown

The logger can run without subscribing to `/vehicle_plotter/state`, which is useful when only controller/path/cone diagnostics are needed.

### `rosbag_controller_node`

Starts rosbag recording for the active run session when rosbagging is enabled. It waits for `/run_session` so the bag lands in the same session directory as the logs and plots.

## Launch Behavior

In `sim_car/launch/full_sim_launch.launch.py`:

- `sensor_pipeline:=true` starts raw sensor nodes, `measurement_node`, and the `vehicle_plotter` launch include.
- `logging:=true` enables the direct full-sim logger node.
- `rosbagging:=true` enables rosbag recording through the plotter launch include.
- `controller_diagnostics:=true` enables steering-tracking diagnostics artifacts.
- `thesis_controller_diagnostics:=true` enables the wider thesis-oriented controller diagnostics.
- `path_tracking_eval:=true` enables path-vs-ground-truth evaluation. This is currently true by default in the full sim launch.

The old standalone live cone RMSE and controller diagnostics window names are not current console scripts. Cone RMSE and controller diagnostics are now handled through evaluator output, logger subscriptions, CSV summaries, and generated plots.

## Sensor Pipeline

The intended sensor data path is:

`raw sensor nodes -> measurement_node -> measured /sim topics -> plotter_node -> /vehicle_plotter/state -> logger_node`

`plotter_node` reads measured topics, not raw topics, when the sensor pipeline is active. This keeps the dashboard aligned with the sensor values the rest of the stack is expected to consume.

## Cone RMSE Artifacts

Camera and LiDAR cone evaluators publish sample streams. `logger_node` subscribes to the configured camera and LiDAR eval prefixes and writes source-specific CSV files on shutdown.

Typical files:

- `logs/cone_range_rmse_samples_mono.csv`
- `logs/cone_range_rmse_samples_stereo.csv`
- `logs/cone_range_rmse_samples_lidar.csv`
- `plots/cone_range_rmse_mono.png`
- `plots/cone_range_rmse_stereo.png`
- `plots/cone_range_rmse_lidar.png`

Only files for sources with samples are produced.

## Controller And Path Evaluation Artifacts

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
- corridor oscillation summaries and plot when applicable

Path tracking evaluation can write:

- `logs/path_tracking_eval.csv`
- `logs/path_tracking_eval_summary.json`
- `logs/path_tracking_eval_summary.txt`
- `plots/path_tracking_eval_cte.png`
- `plots/path_tracking_eval_overlay.png`

## Run IDs

`full_sim_launch.launch.py` builds a compact run prefix from selected track, planner, controller, and LiDAR pipeline. For example, a smalltrack midpoint pure-pursuit run with 3D LiDAR gets a prefix like:

`small_mid_pp_3d`

The session manager adds the timestamp to make the final `run_id`.

## Useful Commands

Build:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select vehicle_plotter vehicle_plotter_msgs sim_car
```

Source:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Launch the full sensor dashboard path:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py sensor_pipeline:=true
```

Launch with logging and rosbagging:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py logging:=true rosbagging:=true
```

## Prefix Path Warnings

Warnings about missing old packages in `AMENT_PREFIX_PATH` or `CMAKE_PREFIX_PATH` mean the shell has stale sourced workspace paths.

Reset and source again:

```bash
cd ~/ros2_ws && unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
cd ~/ros2_ws && source /opt/ros/humble/setup.bash
cd ~/ros2_ws && source install/setup.bash
```
