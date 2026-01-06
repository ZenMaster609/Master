# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROS2 multi-package workspace for vehicle simulation and state estimation:

| Package | Description |
|---------|-------------|
| **sim_car** | Gazebo Fortress car simulation with IMU, GPS, wheel encoders |
| **vehicle_plotter** | Real-time plotting and logging for vehicle state data |
| **vehicle_plotter_msgs** | Custom ROS2 message definitions (VehicleState.msg) |
| **canbus_decoder** | CAN bus decoder for real hardware (wheel speeds, suspension) |

## Workspace Structure

```
master/                         # Workspace root
├── .devcontainer/              # Docker dev container config
├── sim_car/                    # Simulation package
│   ├── sim_car/                # Python nodes
│   ├── launch/                 # Launch files
│   ├── urdf/                   # Robot model
│   └── worlds/                 # Gazebo worlds
├── vehicle_plotter/            # Plotting/logging package
│   ├── vehicle_plotter/        # Python modules
│   │   ├── core/               # Vehicle state, time sync, QoS
│   │   ├── adapters/           # Sensor adapters (Gazebo, CAN, VectorNav)
│   │   ├── nodes/              # ROS2 nodes
│   │   ├── plotting/           # Plot config, manager, backends
│   │   └── logging/            # Log writer, formats
│   ├── launch/                 # Launch files
│   └── config/                 # YAML configs
├── vehicle_plotter_msgs/       # Message definitions
│   └── msg/VehicleState.msg
└── canbus_decoder/             # CAN bus decoder (real hardware only)
    ├── canbus_decoder/         # Python nodes
    └── launch/                 # Launch files
```

## Dev Container (Recommended)

Use VS Code Dev Containers for a ready-to-go development environment:

**Prerequisites:**
- Docker with NVIDIA Container Toolkit
- VS Code with "Dev Containers" extension

**Setup:**
1. Open `master/` folder in VS Code
2. Click "Reopen in Container" when prompted
3. Wait for build (~5 min first time)
4. Start developing!

**Inside container:**
```bash
# Build all packages
cb

# Build specific package
cbs sim_car

# Launch full system (simulation + plotter + logger)
bringup

# Or launch just the plotter
plotter
```

## Build Commands

```bash
# Build all packages (from workspace root ~/ros2_ws)
colcon build --symlink-install

# Build in dependency order
colcon build --symlink-install --packages-select vehicle_plotter_msgs
colcon build --symlink-install --packages-select sim_car
colcon build --symlink-install --packages-select vehicle_plotter

# Source the workspace
source install/setup.bash
```

## Running the System

### Full System (Recommended)
```bash
# Launch everything: Gazebo + sim_car nodes + plotter + logger
ros2 launch vehicle_plotter bringup.launch.py

# With auto control mode (car drives in circles)
ros2 launch vehicle_plotter bringup.launch.py control_mode:=auto

# Headless Gazebo (plotter still shows)
ros2 launch vehicle_plotter bringup.launch.py headless:=true
```

### Components Separately
```bash
# Terminal 1: Gazebo simulation
ros2 launch sim_car gazebo_sim.launch.py

# Terminal 2: Control and sensor nodes
ros2 launch sim_car nodes.launch.py

# Terminal 3: Plotter and logger
ros2 launch vehicle_plotter plotter.launch.py
```

### Plotting/Logging Only
```bash
# Enable plotting, disable logging
ros2 launch vehicle_plotter plotter.launch.py enable_log:=false

# Enable logging, disable plotting
ros2 launch vehicle_plotter plotter.launch.py enable_plot:=false

# Change log format
ros2 launch vehicle_plotter plotter.launch.py log_format:=csv
```

### Offline Replay
```bash
# Replay from rosbag
ros2 launch vehicle_plotter offline_replay.launch.py bag_path:=/path/to/bag
```

### CAN Bus Monitoring (Jetson Only)
```bash
# First, setup the CAN interface
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0

# Install ros2_socketcan (if not already installed)
sudo apt install ros-humble-ros2-socketcan

# Build canbus_decoder package
colcon build --symlink-install --packages-select canbus_decoder
source install/setup.bash

# Run CAN monitor to see raw CAN traffic
ros2 launch canbus_decoder can_monitor.launch.py

# With options
ros2 launch canbus_decoder can_monitor.launch.py can_device:=can0 verbose:=true
ros2 launch canbus_decoder can_monitor.launch.py stats_interval:=10.0
```

## Architecture

### Data Flow
```
Gazebo Sensors → ros_gz_bridge → sim_car nodes → vehicle_plotter
     ↓                                              ↓
  /imu/data                              DataCollectorNode
  /gps/fix                                      ↓
  /odom                              /vehicle_plotter/state
  /wheel_encoder/*                         ↓        ↓
                                    PlotterNode  LoggerNode
                                         ↓           ↓
                                    PyQtGraph   ~/.ros/vehicle_logs/
```

### Key Topics
| Topic | Type | Source |
|-------|------|--------|
| `/cmd_vel` | Twist | control_node |
| `/odom` | Odometry | Gazebo |
| `/imu/data` | Imu | Gazebo |
| `/gps/fix` | NavSatFix | Gazebo |
| `/wheel_encoder/ticks` | Int32MultiArray | wheel_encoder_node |
| `/vehicle_plotter/state` | VehicleState | data_collector_node |

### VehicleState Message
```
# Position (local frame, meters)
float64 x, y

# Velocity (body frame, m/s)
float64 vx, vy

# Orientation
float64 yaw, yaw_rate

# Derived
float64 speed, distance_traveled, slip_longitudinal, slip_lateral

# Encoders [FL, FR, RL, RR]
int32[4] encoder_ticks
float32[4] encoder_velocities

# GPS
float64 gps_latitude, gps_longitude
bool gps_valid

# EKF-ready
float64[36] covariance
string estimation_status  # "raw", "filtered", "predicted"
```

## Configuration

### Plot Configuration (config/default_plots.yaml)
- Position trajectory (X vs Y)
- Velocity vs time (Vx, Vy, Speed)
- Heading vs time (Yaw in degrees)
- Encoder ticks vs distance

### Topic Mappings
- `config/gazebo_topics.yaml` - Gazebo simulation topics
- `config/vectornav_topics.yaml` - VectorNav VN-200 hardware topics

### Logging
- Default format: Parquet (70-90% smaller than CSV)
- Default path: `~/.ros/vehicle_logs/session_YYYYMMDD_HHMMSS/`
- Override: `log_format:=csv`, `log_path:=/custom/path`

## Dependencies

**System:**
- ROS2 Humble
- Gazebo Fortress (Ignition)
- NVIDIA GPU (for Gazebo rendering)

**Python:**
- pyqtgraph, PyQt5 (real-time plotting)
- pyarrow (Parquet logging)
- numpy

## Testing

```bash
# Run tests
colcon test --packages-select sim_car vehicle_plotter

# Monitor topics
ros2 topic list
ros2 topic echo /vehicle_plotter/state

# Check rates
ros2 topic hz /vehicle_plotter/state
```
