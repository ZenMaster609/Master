# ROS2 Simulated Car with Virtual Sensors

A complete ROS2 (Humble or later) Python project that simulates a car equipped with virtual sensors in Gazebo.

## Features

- **Car Model**: 4-wheeled differential drive car with controllable movement
- **Virtual Sensors**:
  - IMU (Inertial Measurement Unit)
  - GPS (Global Positioning System)
  - Wheel Encoders (all 4 wheels with configurable resolution)
- **Control Modes**:
  - Keyboard control (interactive)
  - Automated control (circular pattern)
- **Sensor Processing**: Real-time sensor data monitoring and processing
- **Dev Container**: Ready-to-use Docker environment with all dependencies

## Quick Start

### Option 1: WSL2 or Native Linux (Recommended for GUI)

Best for interactive development with full Gazebo GUI support.

**Prerequisites:**
- Windows 10/11 with WSL2 and Ubuntu 22.04, OR
- Native Ubuntu 22.04

**Setup (run once):**
```bash
# For WSL2: Copy script from Windows drive
cp /mnt/YOUR_DRIVE/path/to/sim_car/scripts/setup-wsl2.sh ~

# For native Linux: Use script directly from project
# chmod +x scripts/setup-wsl2.sh && cp scripts/setup-wsl2.sh ~

# Run the setup script
chmod +x ~/setup-wsl2.sh
./setup-wsl2.sh

# Copy project and build
cp -r /mnt/YOUR_DRIVE/path/to/sim_car ~/ros2_ws/src/  # WSL2
# OR: cp -r . ~/ros2_ws/src/sim_car  # Native Linux
cb
```

**Run the Simulation:**
```bash
ros2 launch sim_car gazebo_sim.launch.py
```

The script auto-detects WSL2 vs native Linux and configures the appropriate renderer.

### Option 2: Docker Dev Container (Headless Only)

Best for CI/testing or headless simulation. **Note:** Gazebo GUI does not work in Docker on Windows due to OpenGL limitations.

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with NVIDIA Container Toolkit
- [VS Code](https://code.visualstudio.com/) with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**Setup:**
1. Open `sim_car` folder in VS Code
2. Click "Reopen in Container" (or `Ctrl+Shift+P` → "Dev Containers: Reopen in Container")
3. Wait for container to build (~5-10 min first time)

**Run Headless Simulation:**
```bash
# Launch without GUI
ros2 launch sim_car gazebo_sim.launch.py headless:=true

# Verify via topics
ros2 topic echo /odom --once
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0}}" --once
```

## Project Structure

```
sim_car/
├── .devcontainer/          # Dev Container configuration
│   ├── devcontainer.json      # VS Code container settings
│   ├── Dockerfile             # Container image definition
│   └── post-create.sh         # Setup script
├── package.xml              # Package dependencies
├── setup.py                 # Python package setup
├── setup.cfg               # Install configuration
├── resource/               # Package resources
├── sim_car/                # Python source code
│   ├── __init__.py
│   ├── control_node.py        # Car control node
│   ├── sensor_processor.py    # Sensor processing node
│   └── wheel_encoder_node.py  # Wheel encoder processing
├── launch/                 # Launch files
│   ├── gazebo_sim.launch.py    # Gazebo simulation
│   └── nodes.launch.py         # Control & sensor nodes
├── urdf/                   # Robot model
│   └── car.urdf           # Car URDF with sensors
├── worlds/                 # Gazebo worlds
│   └── test_world.sdf     # Test environment (SDF format)
└── README.md              # This file
```

## Prerequisites

> **Note**: If using the Dev Container (recommended), skip to [Quick Start with Dev Container](#quick-start-with-dev-container-recommended) - all dependencies are included.

### System Requirements (Manual Installation)

- Ubuntu 22.04 or later
- ROS2 Humble, Iron, or later
- Gazebo Fortress or Garden
- Python 3.8 or later

### Required ROS2 Packages

```bash
sudo apt update
sudo apt install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    ros-humble-tf2-ros
```

Note: Replace `humble` with your ROS2 distribution name if different.

## Installation

### 1. Create a ROS2 Workspace (if you don't have one)

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 2. Copy the Package

Copy the `sim_car` directory to your workspace's `src` folder:

```bash
# If you're already in the directory containing sim_car:
cp -r sim_car ~/ros2_ws/src/

# Or clone if using git:
# cd ~/ros2_ws/src
# git clone <repository-url>
```

### 3. Build the Package

```bash
cd ~/ros2_ws
colcon build --packages-select sim_car
source install/setup.bash
```

### 4. Verify Installation

```bash
ros2 pkg list | grep sim_car
```

You should see `sim_car` in the output.

## Usage

### Quick Start

**Terminal 1: Launch Gazebo with the car**

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch sim_car gazebo_sim.launch.py
```

Wait for Gazebo to fully load. You should see the car spawn in the simulation.

**Terminal 2: Launch control and sensor nodes**

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch sim_car nodes.launch.py
```

The car is now ready to be controlled!

### Control Modes

#### Keyboard Control (Default)

When running in keyboard mode, use these keys:

**Speed Adjustment:**
- `w` - Increase linear speed
- `x` - Decrease linear speed
- `a` - Increase angular speed
- `d` - Decrease angular speed

**Movement:**
- `i` - Move forward
- `,` - Move backward
- `j` - Turn left
- `l` - Turn right
- `k` - Stop

**Other:**
- `q` - Quit

#### Automated Control

To run in automated mode (car drives in circles):

```bash
ros2 launch sim_car nodes.launch.py control_mode:=auto
```

### Advanced Options

#### Custom Speed Settings

```bash
ros2 launch sim_car nodes.launch.py \
    control_mode:=keyboard \
    linear_speed:=1.0 \
    angular_speed:=1.5
```

#### Custom Sensor Update Rate

```bash
ros2 launch sim_car nodes.launch.py \
    publish_rate:=2.0
```

#### Custom Encoder Resolution

```bash
ros2 launch sim_car nodes.launch.py \
    ticks_per_revolution:=4096
```

#### Custom World File

```bash
ros2 launch sim_car gazebo_sim.launch.py \
    world:=/path/to/your/world.world
```

## ROS2 Topics

The simulation publishes to the following topics:

### Published by Gazebo Plugins

- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands (input)
- `/imu/data` (sensor_msgs/Imu) - IMU data
- `/gps/fix` (sensor_msgs/NavSatFix) - GPS position
- `/odom` (nav_msgs/Odometry) - Odometry data
- `/wheel_encoder/joint_states` (sensor_msgs/JointState) - Raw wheel joint states

### Published by Encoder Node

- `/wheel_encoder/ticks` (std_msgs/Int32MultiArray) - Encoder ticks (FL, FR, RL, RR)
- `/wheel_encoder/velocities` (std_msgs/Float32MultiArray) - Wheel velocities in m/s

### Inspect Topics

```bash
# List all active topics
ros2 topic list

# View GPS data
ros2 topic echo /gps/fix

# View IMU data
ros2 topic echo /imu/data

# View wheel encoder ticks
ros2 topic echo /wheel_encoder/ticks

# View wheel velocities
ros2 topic echo /wheel_encoder/velocities

# Check topic frequency
ros2 topic hz /odom
```

## Visualizing Data

### RViz2 Visualization

```bash
rviz2
```

In RViz2:
1. Set Fixed Frame to `base_link` or `odom`
2. Add displays:
   - **TF**: To see coordinate frames
   - **RobotModel**: To visualize the car

## Troubleshooting

### Gazebo doesn't start

**Issue**: Gazebo fails to launch or crashes.

**Solutions**:
- Ensure Gazebo is properly installed: `gazebo --version`
- Try launching Gazebo standalone first: `gazebo`
- Check if required models are available

### No sensor data

**Issue**: Sensor topics exist but no data is published.

**Solutions**:
- Verify the robot spawned correctly in Gazebo
- Check topic list: `ros2 topic list`
- Ensure Gazebo physics is running (not paused)
- Verify plugin loading: Check Gazebo terminal output for errors

### Keyboard control not working

**Issue**: Keys don't control the car.

**Solutions**:
- Make sure the terminal running the control node has focus
- Check if `/cmd_vel` topic is receiving data: `ros2 topic echo /cmd_vel`
- Verify the control node is running: `ros2 node list`

### Car doesn't move

**Issue**: Commands sent but car doesn't move in Gazebo.

**Solutions**:
- Check if Gazebo physics is paused (click play button)
- Verify differential drive plugin loaded correctly
- Check Gazebo terminal for plugin errors
- Ensure wheels have proper friction (mu1/mu2 in URDF)

### Build errors

**Issue**: `colcon build` fails.

**Solutions**:
- Ensure all dependencies are installed
- Check Python version: `python3 --version` (should be 3.8+)
- Source ROS2: `source /opt/ros/humble/setup.bash`
- Clean build: `rm -rf build install log` then rebuild

## Extending the Project

### Adding New Sensors

1. Add sensor link and joint in `urdf/car.urdf`
2. Add Gazebo sensor plugin in `<gazebo>` tags
3. Update `sensor_processor.py` to subscribe to new topic

### Modifying the World

Edit `worlds/test_world.world` to add:
- New obstacles
- Different terrain
- Lighting changes
- Additional environmental features

### Custom Control Logic

Modify `sim_car/control_node.py`:
- Add new movement patterns
- Implement path planning
- Add obstacle avoidance logic

### Advanced Sensor Processing

Enhance `sim_car/sensor_processor.py`:
- Add computer vision processing
- Implement SLAM algorithms
- Create sensor fusion logic

## Example Commands Summary

```bash
# Build the package
cd ~/ros2_ws
colcon build --packages-select sim_car
source install/setup.bash

# Launch simulation
ros2 launch sim_car gazebo_sim.launch.py

# Launch nodes (keyboard control)
ros2 launch sim_car nodes.launch.py

# Launch nodes (auto control)
ros2 launch sim_car nodes.launch.py control_mode:=auto

# Monitor topics
ros2 topic list
ros2 topic echo /imu/data
ros2 topic hz /odom

# Visualize
rviz2
```

## Technical Details

### Car Specifications

- **Dimensions**: 0.6m (L) × 0.4m (W) × 0.2m (H)
- **Wheel Diameter**: 0.2m
- **Wheel Separation**: 0.45m
- **Mass**: 10 kg (chassis) + 0.5 kg (each wheel)
- **Drive Type**: Differential drive (front wheels)

### Sensor Specifications

**IMU:**
- Update rate: 100 Hz
- Outputs: Orientation, angular velocity, linear acceleration

**GPS:**
- Update rate: 5 Hz
- Noise: ±2m (horizontal and vertical)

**Wheel Encoders:**
- Resolution: 2048 ticks/revolution (configurable)
- Update rate: 50 Hz
- Number of wheels: 4 (FL, FR, RL, RR)
- Outputs: Cumulative ticks, wheel velocities

## License

Apache-2.0

## Support

For issues, questions, or contributions, please refer to the project repository or contact the maintainer.

## References

- [ROS2 Documentation](https://docs.ros.org/en/humble/index.html)
- [Gazebo Documentation](https://gazebosim.org/docs)
- [URDF Tutorials](https://wiki.ros.org/urdf/Tutorials)
- [Gazebo ROS2 Plugins](https://github.com/ros-simulation/gazebo_ros_pkgs)
