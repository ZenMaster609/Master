# agent.md

This file provides guidance to Codex when working in this repository.

## User Interaction Preferences

Always include copyable command lines prefixed with `cd ~/ros2_ws &&` whenever you mention build, run, launch, source, or test steps. When a rebuild is relevant, also include the appropriate `colcon` rebuild command. When sourcing is needed, also include the appropriate `source` command (often `source install/setup.bash`). If you provide multiple commands, each line must start with `cd ~/ros2_ws &&`.

## ROS2/Gazebo Process Policy

You are authorized to build and run ROS2 projects in this workspace. After collecting whatever data you need, shut down any ROS2 and Gazebo processes you started.

## Workspace Summary

This is a ROS2 multi-package workspace for vehicle simulation, CAN/IMU decoding, and plotting/logging.

Core packages:
- `sim_car`: Gazebo Fortress simulation, virtual sensors, and Ackermann command bridge.
- `measurement_node`: Adds configurable noise/latency/dropout between `/sim/raw/*` and `/sim/*` topics.
- `vehicle_plotter`: Aggregates sensor data into `VehicleState`, plots in real time, logs to disk, and controls rosbag.
- `canbus_decoder`: Decodes raw CAN frames into wheel RPM, suspension, and steering topics.
- `vectornav_decoder`: VN-200 serial decoder (IMU/GPS/INS topics).
- `steering_gui`: RQT GUI for Ackermann commands and brake command.
- `vehicle_plotter_msgs`: `VehicleState` and `RunSession` messages.

EUFS packages are present under `eufs_sim/` and `eufs_msgs/` for compatibility and asset reuse.

## Data Output Paths

`vehicle_plotter` uses run sessions under `multidata/` by default. A session directory contains logs, rosbags, plots, and plot data.

## Common Commands

These are examples only. Remember the `cd ~/ros2_ws &&` prefix rule for any command you provide.

- Build everything:
  - `cd ~/ros2_ws && colcon build --symlink-install`
- Build a single package:
  - `cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car`
- Source the workspace:
  - `cd ~/ros2_ws && source install/setup.bash`
