# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Install dependencies
sudo apt install -y ros-humble-gazebo-ros-pkgs ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher ros-humble-xacro ros-humble-tf2-ros

# 2. Create workspace and copy package
mkdir -p ~/ros2_ws/src
cp -r sim_car ~/ros2_ws/src/

# 3. Build
cd ~/ros2_ws
colcon build --packages-select sim_car
source install/setup.bash
```

## Run the Simulation (2 terminals)

### Terminal 1: Start Gazebo
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch sim_car gazebo_sim.launch.py
```

### Terminal 2: Start Control Nodes
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch sim_car nodes.launch.py
```

## Keyboard Controls

- `i` - Forward
- `,` - Backward
- `j` - Turn left
- `l` - Turn right
- `k` - Stop
- `w/x` - Increase/decrease speed
- `q` - Quit

## Automated Mode

```bash
ros2 launch sim_car nodes.launch.py control_mode:=auto
```

## View Sensor Data

```bash
# List topics
ros2 topic list

# View GPS
ros2 topic echo /gps/fix

# View IMU
ros2 topic echo /imu/data

# View wheel RPM
ros2 topic echo /sim/wheel_encoder/rpm

# Visualize in RViz2
rviz2
```

## Key Topics

- `/cmd_vel` - Velocity commands
- `/imu/data` - IMU readings
- `/gps/fix` - GPS position
- `/odom` - Odometry
- `/sim/wheel_encoder/rpm` - Wheel speeds in RPM

## Troubleshooting

**Gazebo won't start?**
```bash
gazebo --version  # Check installation
```

**No sensor data?**
```bash
ros2 topic list  # Verify topics exist
ros2 topic hz /odom  # Check publish rate
```

**Car won't move?**
- Ensure Gazebo is not paused (play button)
- Check `/cmd_vel` topic: `ros2 topic echo /cmd_vel`

See README.md for complete documentation.
