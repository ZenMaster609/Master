# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROS2 Gazebo Fortress car simulation with virtual sensors. A 4-wheeled differential drive car with IMU, GPS, and wheel encoders running in Gazebo Fortress with ros_gz bridge.

**Workspace**: `/home/developer/ros2_ws` (package at `src/sim_car`)

## Build Commands

```bash
# Build (from workspace root ~/ros2_ws)
colcon build --packages-select sim_car

# Source the workspace
source install/setup.bash

# Build with symlink for faster iteration
colcon build --packages-select sim_car --symlink-install
```

## Running the Simulation

```bash
# Terminal 1: Launch Gazebo with the car
ros2 launch sim_car gazebo_sim.launch.py

# Terminal 2: Launch control and sensor nodes
ros2 launch sim_car nodes.launch.py

# Automated mode (car drives in circles)
ros2 launch sim_car nodes.launch.py control_mode:=auto
```

## Testing and Debugging

```bash
# Run linting tests (flake8, pep257, copyright)
colcon test --packages-select sim_car
colcon test-result --verbose

# Run individual node for testing
ros2 run sim_car control_node --ros-args -p mode:=auto
ros2 run sim_car sensor_processor
ros2 run sim_car wheel_encoder_node

# List active topics
ros2 topic list

# Monitor topic data
ros2 topic echo /odom
ros2 topic echo /imu
ros2 topic echo /navsat

# Check publish rates
ros2 topic hz /odom
```

## Architecture

### Gazebo Bridge Architecture
- Gazebo Fortress publishes native sensor topics (`/imu`, `/navsat`, `/odom`)
- `ros_gz_bridge` in `gazebo_sim.launch.py` converts Gazebo messages to ROS2 messages
- Bridge mappings: `/cmd_vel`, `/odom`, `/imu`, `/navsat`, `/joint_states`, `/clock`
- Python nodes subscribe using relative topic names (e.g., `imu/data`, `gps/fix`)

### Key Components
- **car.urdf**: Robot model with gz-sim-* plugins (DiffDrive, JointStatePublisher, IMU sensor, NavSat sensor)
- **gazebo_sim.launch.py**: Starts Gazebo, spawns robot, configures ros_gz_bridge
- **nodes.launch.py**: Starts control_node, sensor_processor, wheel_encoder_node

### ROS2 Nodes
| Node | Purpose |
|------|---------|
| control_node | Keyboard/auto control, publishes `/cmd_vel` |
| sensor_processor | Subscribes to all sensors, logs status |
| wheel_encoder_node | Converts joint states to encoder ticks |

### Key Topics
| Topic | Message Type | Source |
|-------|--------------|--------|
| `/cmd_vel` | geometry_msgs/Twist | control_node |
| `/odom` | nav_msgs/Odometry | Gazebo DiffDrive |
| `/imu` | sensor_msgs/Imu | Gazebo via bridge |
| `/navsat` | sensor_msgs/NavSatFix | Gazebo via bridge |
| `/wheel_encoder/velocities` | std_msgs/Float32MultiArray | wheel_encoder_node |

## File Structure

```
sim_car/
├── sim_car/           # Python nodes
│   ├── control_node.py
│   ├── sensor_processor.py
│   └── wheel_encoder_node.py
├── launch/            # Launch files
├── urdf/car.urdf      # Robot model with Gazebo plugins
└── worlds/            # SDF world files
```

## Gazebo Plugins Used

Uses **Ignition Gazebo 6 (Fortress)** naming convention:
- `ignition-gazebo-diff-drive-system`: Differential drive controller
- `ignition-gazebo-joint-state-publisher-system`: Wheel joint states
- Native `imu` and `navsat` sensors (no separate plugin needed)

**Important**: Use `ignition::gazebo::systems::*` namespace, NOT `gz::sim::systems::*` (that's for newer Gazebo Garden/Harmonic).

## Development Environments

Two options for development, each with different trade-offs:

### Option 1: Docker Dev Container (Headless)

Best for: CI/testing, portable environments, headless simulation.

**Limitation**: Gazebo/RViz2 GUI does not work due to OpenGL/X11 forwarding issues.

```bash
# 1. Open in VS Code → "Reopen in Container"
# 2. Build
cb

# 3. Run headless simulation
ros2 launch sim_car gazebo_sim.launch.py headless:=true

# 4. Verify via topics
ros2 topic list
ros2 topic echo /odom --once
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0}}" --once
```

**Claude Code CLI** is pre-installed and authentication persists between rebuilds.

### Option 2: WSL2 or Native Linux (Full GUI)

Best for: Interactive development with Gazebo + RViz2 GUI.

**For WSL2 on Windows** (run once on fresh Ubuntu 22.04):
```bash
# Copy script from Windows
cp /mnt/e/master/sim_car/scripts/setup-wsl2.sh ~
chmod +x ~/setup-wsl2.sh
./setup-wsl2.sh

# After setup, copy project
cp -r /mnt/e/master/sim_car ~/ros2_ws/src/

# Build and run
cb
ros2 launch sim_car gazebo_sim.launch.py
```

**For Native Linux** (run once on fresh Ubuntu 22.04):
```bash
# Clone/copy the project, then run setup script
chmod +x scripts/setup-wsl2.sh
./scripts/setup-wsl2.sh

# Copy project to workspace
cp -r . ~/ros2_ws/src/sim_car

# Build and run
cb
ros2 launch sim_car gazebo_sim.launch.py
```

The script auto-detects WSL2 vs native Linux:
- **WSL2**: Uses OGRE1 renderer (D3D12 compatible)
- **Native Linux**: Uses OGRE2 renderer (full GPU acceleration)

### Environment Portability

To replicate your WSL2 environment on other machines:
```powershell
# Export from configured machine
wsl --export Ubuntu-22.04 D:\ros2-env.tar

# Import on new machine
wsl --import ROS2Dev D:\WSL\ROS2Dev D:\ros2-env.tar
```

Or just run `setup-wsl2.sh` on each machine.

**WARNING - Do NOT modify `.devcontainer/devcontainer.json` to add:**
- X11 mounts (`/tmp/.X11-unix`, `/mnt/wslg`)
- Linux-specific env vars (`XDG_RUNTIME_DIR`, `WAYLAND_DISPLAY`)

These paths don't exist on Windows and will break the container.

## Dependencies

- ROS2 Humble or later
- Gazebo Fortress (via ros_gz packages)
- ros_gz_sim, ros_gz_bridge

## Launch Parameters

**gazebo_sim.launch.py:**
- `use_sim_time`: Use simulation time (default: true)
- `world`: Path to world file (default: test_world.sdf)

**nodes.launch.py:**
- `control_mode`: keyboard or auto (default: keyboard)
- `linear_speed`: m/s (default: 0.5)
- `angular_speed`: rad/s (default: 1.0)
- `publish_rate`: Hz for sensor status (default: 1.0)
