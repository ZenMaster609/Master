# Workspace Layout

## Top-Level Layout

```
Master/
├── agent.md
├── documentation/                # Canonical docs (this folder)
├── canbus_decoder/               # CAN decoder ROS2 package
├── eufs_remastered/
│   ├── eufs_gz_dynamics/         # Gazebo dynamics plugin (C++)
│   ├── eufs_msgs/                # EUFS message and action definitions
│   ├── gazebo_cone_plugins/      # Gazebo cone plugin (ground-truth/visible/recolor)
│   ├── eufs_models/              # EUFS vehicle model library (C++)
│   └── steering_gui/             # RQT steering GUI plugin
├── measurement_node/             # Measurement noise/latency node
├── sim_car/                      # Main Gazebo simulation package
├── vectornav_decoder/            # VectorNav VN-200 serial decoder
├── vehicle_plotter/              # Plotting/logging/data collection
├── vehicle_plotter_msgs/         # RunSession + VehicleState messages
├── multidata/                    # Default data output directory
├── scripts/                      # Helper scripts (non-ROS)
├── tools/                        # Utility scripts and tooling
├── utilities/                    # Misc utilities
├── requirements.txt
├── Centre of mass calculator 2024 - COG.csv
└── Centre of mass calculator 2024.xlsx
```

## ROS2 Packages in This Workspace

| Package | Path | Type | Purpose |
| --- | --- | --- | --- |
| `canbus_decoder` | `canbus_decoder/` | Python | Decode raw CAN frames into wheel/suspension/steering topics. |
| `eufs_gz_dynamics` | `eufs_remastered/eufs_gz_dynamics/` | C++ | Gazebo dynamics plugin library for EUFS car models. |
| `eufs_msgs` | `eufs_remastered/eufs_msgs/` | Interface | EUFS messages/actions used by EUFS simulation stack. |
| `gazebo_cone_plugins` | `eufs_remastered/gazebo_cone_plugins/` | C++ | Remastered cone plugin: track cones, visible cones, YAML confusion-matrix recoloring. |
| `eufs_models` | `eufs_remastered/eufs_models/` | C++ | EUFS vehicle model library required by `eufs_gz_dynamics`. |
| `measurement_node` | `measurement_node/` | Python | Noise/latency/dropout layer between /sim/raw and /sim. |
| `sim_car` | `sim_car/` | Python | Gazebo Fortress sim, virtual sensors, control bridge. |
| `steering_gui` | `eufs_remastered/steering_gui/` | Python | RQT steering GUI for Ackermann commands. |
| `vectornav_decoder` | `vectornav_decoder/` | Python | VN-200 serial decoder to ROS topics. |
| `vehicle_plotter` | `vehicle_plotter/` | Python | Data collection, plotting, logging, rosbag control. |
| `vehicle_plotter_msgs` | `vehicle_plotter_msgs/` | Interface | `RunSession` and `VehicleState` messages. |

Notes:
- `eufs_car` is not a ROS2 package in this workspace. The `sim_car/urdf/eufs_car.urdf.xacro` file is a model asset and is not documented as a standalone package.
