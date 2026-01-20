# sim_car Documentation

This document describes how to run the full sim launch, how the sensor pipeline is wired, and how to add or modify sensors and plots for the EUFS sim stack.

## Quick Start: Full Sim Launch

Build and source:

```bash
cd /home/aleks/ros2_ws
colcon build --packages-select sim_car
source install/setup.bash
```

Launch:

```bash
ros2 launch sim_car full_sim_launch.launch.py
```

### Launch Arguments (full_sim_launch.launch.py)

- `headless` (default: `false`): Run Gazebo without GUI.
- `world` (default: `sim_car/worlds/small_track.world`): World to load.
- `control_mode` (default: `auto`): `auto`, `keyboard`, or `none`.
- `linear_speed` (default: `0.5`): Auto mode linear speed (m/s).
- `angular_speed` (default: `1.0`): Auto mode angular speed (rad/s).
- `sensor_mode` (default: `both`): `real`, `virtual`, or `both`.
- `enable_plot` (default: `true`): Enable live plotting.
- `enable_log` (default: `true`): Enable logging.
- `enable_rosbag` (default: `true`): Enable rosbag recording.
- `use_sim_time` (default: `true`): Use sim time from `/clock`.

Examples:

```bash
# Headless sim, no plots
ros2 launch sim_car full_sim_launch.launch.py headless:=true enable_plot:=false

# Only real sensors (no virtual) and no control node
ros2 launch sim_car full_sim_launch.launch.py sensor_mode:=real control_mode:=none
```

## Pipeline Overview

Gazebo publishes `/sim/*` topics. The `sim_car` sensor nodes publish additional derived topics under `/sim/*`. The `vehicle_plotter` package subscribes to those `/sim/*` topics and produces `/vehicle_plotter/state` for plotting/logging.

Key topic flow:

- `/sim/odom`, `/sim/imu`, `/sim/navsat` from Gazebo sensors/plugins
- `/sim/joint_states` from Gazebo joint state plugin
- `/sim/wheel_encoder/velocities`, `/sim/suspension`, `/sim/steering_angle` from `sim_car` nodes
- Virtual sensors: `/sim/cooling/*`, `/sim/brakes/*`, `/sim/pitot/*`
- Plotter output: `/vehicle_plotter/state`

## Where to Put Vehicle Dynamics Parameters

EUFS dynamics parameters live in:

- `Master/sim_car/config/eufs_config.yaml`

This file is passed to the EUFS dynamics plugin via `eufs_car.urdf.xacro` and used at spawn time.

## Editing an Existing Sensor

Typical locations:

- Sensor rate/limits: `Master/sim_car/config/sensor_config.yaml`
- Gazebo sensor definitions: `Master/sim_car/urdf/eufs_car.urdf.xacro`
- Sensor node implementation: `Master/sim_car/sim_car/*.py`
- Plotter adapter/topics: `Master/vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`
- Plot layout: `Master/vehicle_plotter/vehicle_plotter/plotting/plot_config.py`

Steps:

1. Update sensor rate or range in `sensor_config.yaml` if applicable.
2. If the sensor is a Gazebo sensor, edit the `<sensor>` block in `eufs_car.urdf.xacro`.
3. If the sensor is a derived topic, edit the node in `sim_car/sim_car/`.
4. Ensure the topic name matches what the plotter expects (see `gazebo_adapter.py`).
5. Rebuild and relaunch.

## Adding a New Sensor (End-to-End)

### 1) Decide sensor type

- **Gazebo-native sensor**: add a `<gazebo><sensor>` in `eufs_car.urdf.xacro`.
- **Derived sensor**: implement a ROS node in `sim_car/sim_car/` that subscribes to existing topics and publishes a new `/sim/*` topic.

### 2) Add the sensor topic

- If Gazebo sensor: define the topic inside the `<sensor>` block in `eufs_car.urdf.xacro`.
- If derived: create a new node that publishes under `/sim/...` and add it to `Master/sim_car/launch/nodes.launch.py`.

### 3) Add config (optional)

- Add rates/limits to `Master/sim_car/config/sensor_config.yaml`.
- Read the value in `nodes.launch.py` and pass as a parameter to the node.

### 4) Wire into the plotter

- Register the topic in `Master/vehicle_plotter/vehicle_plotter/adapters/gazebo_adapter.py`:
  - Add to `DEFAULT_TOPICS`
  - Add to `SENSOR_RATES` if it should be time-synchronized
  - Create a subscription in `setup_subscriptions`
  - Store data and map to a `VehicleState` field in `compute_state`
- If a new VehicleState field is needed, add it to:
  - `Master/vehicle_plotter_msgs/msg/VehicleState.msg`
  - `Master/vehicle_plotter/vehicle_plotter/core/vehicle_state.py`
  - Any serialization/deserialization logic

### 5) Add plots

- Add a plot definition in `Master/vehicle_plotter/vehicle_plotter/plotting/plot_config.py`.
- Use `XAxisType.TIME` for normal time series or `plot_type="xy"` for trajectories.
- Reference the new `VehicleState` field by name.

### 6) Rebuild

```bash
colcon build --packages-select sim_car vehicle_plotter vehicle_plotter_msgs
source install/setup.bash
```

## Editing or Adding Plots Only

Modify:

- `Master/vehicle_plotter/vehicle_plotter/plotting/plot_config.py`

Then rebuild:

```bash
colcon build --packages-select vehicle_plotter
source install/setup.bash
```

## Common Pitfalls

- If plots show `t=0` or time goes backwards, ensure only one Gazebo instance is running and `/clock` is monotonic.
- If sensor nodes show no data, verify the topic name and namespace (`/sim/...`).
- If wheel/suspension data is missing, check that `/sim/joint_states` is publishing.
