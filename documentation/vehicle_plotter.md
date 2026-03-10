# Vehicle Plotter

`vehicle_plotter` is the plotting and logging package used by the sim.

## What it does

- Builds the live sensor dashboard for the virtual sensor stack
- Aggregates measured `/sim/*` topics into `vehicle_plotter/state` when sensor plotting is enabled
- Saves one offline dashboard image on shutdown: `multidata/<run_id>/plots/virtual_sensors.png`
- Saves per-plot CSV files under `multidata/<run_id>/logs/`
- Runs the live cone RMSE window
- Runs the live controller diagnostics window
- Logs cone RMSE samples and controller diagnostics artifacts

## Main runtime pieces

- `plotter_node`
  - Used for the sensor dashboard
  - Reads measured sensor topics directly
  - Publishes `vehicle_plotter/state`
  - Saves `virtual_sensors.png` on shutdown
- `cone_rmse_plot_node`
  - Shows camera and lidar cone RMSE live
  - Uses raw evaluator outputs
  - Is not affected by `measurement_node`
- `controller_diagnostics_plot_node`
  - Shows live controller diagnostics
- `logger_node`
  - Writes CSV/parquet/session artifacts
  - Writes cone RMSE sample logs
  - Writes controller diagnostics summaries
- `session_manager_node`
  - Creates the `multidata/<run_id>/` session folders

## Sensor pipeline

The intended sim path is:

`sensor nodes -> measurement_node -> vehicle_plotter`

More explicitly:

- virtual sensor nodes publish `/sim/raw/*`
- `measurement_node` adds noise / latency and republishes to `/sim/*`
- `plotter_node` reads `/sim/*`
- `plotter_node` publishes `vehicle_plotter/state`
- `logger_node` stores artifacts for the run

## Launch behavior

In `sim_car/launch/full_sim_launch.launch.py`:

- `sensor_pipeline:=true`
  - starts the virtual sensor nodes
  - starts `measurement_node`
  - starts the main sensor dashboard in `vehicle_plotter`
- `sensor_pipeline:=false`
  - does not start the main sensor dashboard
  - does not publish `vehicle_plotter/state`

The cone RMSE and controller diagnostics windows are separate from the main sensor dashboard and can run independently.

## Output files

Typical session output:

- `multidata/<run_id>/plots/virtual_sensors.png`
- `multidata/<run_id>/logs/*.csv`
- `multidata/<run_id>/logs/vehicle_state_0000.parquet`
- `multidata/<run_id>/logs/cone_range_rmse_samples.csv`
- `multidata/<run_id>/logs/cone_range_rmse_samples_lidar.csv`

## Common commands

Build:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car vehicle_plotter
```

Source:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Launch with the full sensor pipeline:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py sensor_pipeline:=true
```

## About the prefix path warnings

Warnings like:

- missing `install/vectornav_decoder`
- missing `install/measurement_node`
- missing `install/canbus_decoder`

mean the current shell still has old workspace paths in `AMENT_PREFIX_PATH` or `CMAKE_PREFIX_PATH`.

The current workspace setup is clean. If you see those warnings, reset the shell environment and source the current workspace again:

```bash
cd ~/ros2_ws && unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
cd ~/ros2_ws && source /opt/ros/humble/setup.bash
cd ~/ros2_ws && source install/setup.bash
```
