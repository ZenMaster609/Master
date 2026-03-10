# Launch Files and Arguments

This file lists each launch file in the workspace and the arguments it accepts. Defaults are the current values in the launch files.

## sim_car

### `sim_car/launch/gazebo_sim.launch.py`

- `use_sim_time` (default: `true`)
- `world` (default: `sim_car/worlds/small_track.world`)
- `headless` (default: `false`)
- `update_rate_hz` (default: `100.0`)
- `topic_prefix` (default: `/sim/raw`)

### `sim_car/launch/nodes.launch.py`

- `topic_prefix` (default: `/sim/raw`)

### `sim_car/launch/full_sim_launch.launch.py`

- `headless` (default: `false`)
- `update_rate_hz` (default: `180.0`)
- `camera_rate_hz` (default: `15.0`)
- `perception_rate_hz` (default: `60.0`)
- `planner_rate_hz` (default: `60.0`)
- `world` (default: `sim_car/worlds/small_track.world`)
- `plotting` (default: `false`)
- `logging` (default: `false`)
- `close_plots` (default: `true`)
- `rosbagging` (default: `false`)
- `steering` (default: `false`)
- `use_sim_time` (default: `true`)
- `sensor_pipeline` (default: `false`)
- `measure` (default: `false`)
- `sensor_nodes` (default: `false`)
- `measurement_config` (default: `sim_car/config/sensor_config.yaml`)

## vehicle_plotter

### `vehicle_plotter/launch/plotter.launch.py`

- `output_rate_hz` (default: `50.0`)
- `enable_plot` (default: `true`)
- `plot_rate_hz` (default: `30.0`)
- `dark_mode` (default: `true`)
- `save_plots_on_exit` (default: `false`)
- `save_plot_data_on_exit` (default: `true`)
- `close_plots` (default: `true`)
- `enable_log` (default: `true`)
- `log_format` (default: `parquet`)
- `log_path` (default: empty string)
- `gps_origin_lat` (default: `0.0`)
- `gps_origin_lon` (default: `0.0`)
- `use_sim_time` (default: `true`)
- `enable_rosbag` (default: `true`)

### `vehicle_plotter/launch/offline_replay.launch.py`

- `bag_path` (required)
- `rate` (default: `1.0`)
- `loop` (default: `false`)
- `enable_plot` (default: `true`)
- `enable_log` (default: `false`)
- `log_path` (default: `~/.ros/vehicle_logs`)

### `vehicle_plotter/launch/replay.launch.py`

- `bag_path` (required)
- `rate` (default: `1.0`)
- `enable_plot` (default: `true`)
- `enable_log` (default: `false`)
- `qos_override` (default: `vehicle_plotter/config/replay_qos_override.yaml`)
