# Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         GAZEBO                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    Simulated Car                      │  │
│  │                                                       │  │
│  │           ┌─────┐  ┌─────┐                           │  │
│  │           │ IMU │  │ GPS │                           │  │
│  │           └──┬──┘  └──┬──┘                           │  │
│  │              │        │                               │  │
│  │              └────────┘                               │  │
│  │                   │                                   │  │
│  │            ┌──────▼───────┐                          │  │
│  │            │ Diff Drive   │◄──────── /cmd_vel        │  │
│  │            │   Plugin     │                          │  │
│  │            └──────────────┘                          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ ROS2 Topics
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
┌───▼──────────┐    ┌────────▼─────────┐    ┌────────▼────────┐
│              │    │                  │    │                 │
│  /imu/data   │    │  /gps/fix        │    │  /odom          │
│  (Imu)       │    │  (NavSatFix)     │    │  (Odometry)     │
│              │    │                  │    │                 │
└───┬──────────┘    └────────┬─────────┘    └────────┬────────┘
    │                        │                        │
    └────────────────────────┴────────────────────────┘
                             │
                             │ Subscribe
                             │
                    ┌────────▼─────────┐
                    │                  │
                    │ sensor_processor │
                    │      Node        │
                    │                  │
                    └──────────────────┘
                             │
                             │ Logging & Processing
                             ▼
                         Terminal Output


┌──────────────┐
│   User       │
│   Input      │
└──────┬───────┘
       │
       │ Keyboard / Auto
       │
┌──────▼───────┐
│              │
│ control_node │
│              │
└──────┬───────┘
       │
       │ Publish
       │
┌──────▼───────┐
│   /cmd_vel   │
│   (Twist)    │
└──────────────┘
```

## Component Details

### Gazebo Simulation
- **URDF Model**: Defines car structure and sensors
- **Plugins**: Bridge between Gazebo and ROS2
  - `gz-sim-diff-drive-system`: Converts cmd_vel to wheel motion
  - `gz-sim-joint-state-publisher-system`: Publishes wheel encoder joint states
  - Native `imu` sensor: Publishes IMU data
  - Native `navsat` sensor: Publishes GPS data

### ROS2 Nodes

#### 1. control_node
- **Purpose**: Control car movement
- **Modes**:
  - Keyboard: Interactive control via terminal
  - Auto: Automated circular movement
- **Publishes**: `/cmd_vel` (geometry_msgs/Twist)
- **Parameters**:
  - `mode`: 'keyboard' or 'auto'
  - `linear_speed`: Default forward speed
  - `angular_speed`: Default turning speed

#### 2. sensor_processor
- **Purpose**: Monitor and process sensor data
- **Subscribes**:
  - `/imu/data` - IMU readings
  - `/gps/fix` - GPS position
  - `/odom` - Odometry
  - `/wheel_encoder/velocities` - Wheel velocities in m/s
- **Parameters**:
  - `publish_rate`: Status update frequency

#### 3. wheel_encoder_node
- **Purpose**: Convert wheel joint states to wheel velocities
- **Subscribes**: `/joint_states` - Raw wheel joint data from Gazebo
- **Publishes**: `/wheel_encoder/velocities` - Wheel velocities in m/s [FL, FR, RL, RR]
- **Parameters**:
  - `wheel_radius`: Wheel radius in meters (default: 0.1)
  - `publish_rate`: Update rate in Hz (default: 50)

#### 4. robot_state_publisher
- **Purpose**: Publish robot TF transforms
- **Publishes**: `/tf`, `/tf_static`

#### 5. joint_state_publisher
- **Purpose**: Publish joint states for visualization
- **Publishes**: `/joint_states`

## Data Flow

### Control Flow
```
User → control_node → /cmd_vel → Gazebo Plugin → Car Movement
```

### Sensor Flow
```
Gazebo Sensors → Topics → sensor_processor → Terminal Output
```

### Transform Flow
```
URDF → robot_state_publisher → /tf → RViz/Other Nodes
```

### Encoder Flow
```
Wheel Joints → joint_state_publisher → /joint_states
→ wheel_encoder_node → /wheel_encoder/velocities
→ sensor_processor → Terminal Output
```

## Launch Files

### gazebo_sim.launch.py
Starts:
1. Gazebo with custom world
2. Spawns robot from URDF
3. robot_state_publisher
4. joint_state_publisher

### nodes.launch.py
Starts:
1. control_node (with configurable mode)
2. sensor_processor
3. wheel_encoder_node

## File Structure

```
sim_car/
├── package.xml                 # ROS2 package manifest
├── setup.py                    # Python package setup
├── setup.cfg                   # Install configuration
│
├── sim_car/                    # Python source
│   ├── __init__.py
│   ├── control_node.py         # 150 lines - Car control
│   ├── sensor_processor.py     # 210 lines - Sensor processing
│   └── wheel_encoder_node.py   # 140 lines - Encoder processing
│
├── launch/                     # Launch files
│   ├── gazebo_sim.launch.py    # 80 lines - Gazebo setup
│   └── nodes.launch.py         # 100 lines - Node launcher
│
├── urdf/                       # Robot models
│   └── car.urdf                # 500+ lines - Full robot description
│
├── worlds/                     # Gazebo worlds
│   └── test_world.world        # Test environment with obstacles
│
└── docs/                       # Documentation
    ├── README.md               # Complete documentation
    ├── QUICKSTART.md          # Quick start guide
    └── ARCHITECTURE.md        # This file
```

## Extension Points

### Adding New Sensors
1. Add link/joint in `urdf/car.urdf`
2. Add Gazebo plugin
3. Subscribe in `sensor_processor.py`

### Custom Control
1. Modify `control_node.py`
2. Add new movement algorithms
3. Update launch file parameters

### Advanced Processing
1. Extend `sensor_processor.py`
2. Add OpenCV/PCL processing
3. Implement SLAM/Navigation

### Custom Worlds
1. Create new `.world` file
2. Add obstacles/features
3. Launch with custom world parameter

## Dependencies

### Build Dependencies
- `ament_python` - Build system
- `setuptools` - Python packaging

### Runtime Dependencies
- `rclpy` - ROS2 Python client library
- `ros_gz_sim` - Gazebo Fortress simulation
- `ros_gz_bridge` - Gazebo-ROS2 topic bridge
- `robot_state_publisher` - TF publishing
- `joint_state_publisher` - Joint states
- Message packages:
  - `geometry_msgs`
  - `sensor_msgs`
  - `nav_msgs`
  - `tf2_ros`

## Performance Characteristics

- **Gazebo Update Rate**: 1000 Hz (physics)
- **IMU**: 100 Hz
- **GPS**: 5 Hz
- **Control Loop**: ~10 Hz (keyboard), continuous (auto)
- **Sensor Status**: Configurable (default 1 Hz)

## Notes

- Built for ROS2 Humble or later
- Tested with Gazebo Fortress/Garden
- Pure Python implementation (easy to modify)
- Minimal dependencies (easy to deploy)
- Extensible architecture (easy to enhance)
