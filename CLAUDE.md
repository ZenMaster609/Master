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
| **vectornav_decoder** | VectorNav VN-200 serial decoder for IMU/GPS/INS data |

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
├── canbus_decoder/             # CAN bus decoder (real hardware only)
│   ├── canbus_decoder/         # Python nodes
│   └── launch/                 # Launch files
└── vectornav_decoder/          # VectorNav VN-200 decoder
    ├── vectornav_decoder/      # Python modules
    │   ├── vn200_protocol.py   # Binary protocol parser (NO ROS)
    │   ├── vn200_config.py     # YAML config handling (NO ROS)
    │   ├── vn200_serial.py     # Serial communication (NO ROS)
    │   └── vectornav_decoder_node.py  # ROS2 node
    ├── config/                 # Output configs
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

### CAN Bus Wheel Encoders (Real Hardware)

The canbus_decoder package decodes wheel encoder CAN messages from real vehicle hardware.

#### Hardware Setup (Jetson/Linux)
```bash
# Setup CAN interface (run once per boot)
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0

# Verify CAN interface is up
ip -details link show can0

# Install ros2_socketcan (if not already installed)
sudo apt install ros-humble-ros2-socketcan

# Build canbus_decoder package
colcon build --symlink-install --packages-select canbus_decoder
source install/setup.bash
```

#### CAN Decoder Usage

**Option 1: Full System (Decoder + Plotter)**
```bash
# Terminal 1: CAN decoder (decodes CAN messages to ROS topics)
ros2 launch canbus_decoder can_decoder.launch.py can_device:=can0

# Terminal 2: Vehicle plotter with CAN adapter
ros2 launch vehicle_plotter plotter.launch.py adapter:=can

# Verify data flow
ros2 topic echo /can/wheel_velocities  # [FL, FR, RL, RR] in m/s
ros2 topic echo /vehicle_plotter/state
```

**Option 2: CAN Monitor (Debugging Only)**
```bash
# Monitor raw CAN traffic (for debugging)
ros2 launch canbus_decoder can_monitor.launch.py

# With options
ros2 launch canbus_decoder can_monitor.launch.py can_device:=can0 verbose:=true
ros2 launch canbus_decoder can_monitor.launch.py stats_interval:=10.0
```

#### Testing with Virtual CAN (Development)
```bash
# Setup virtual CAN interface (vcan0)
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Inject test CAN frames
# Front axle (ID 0xB9): FL=10mm/s, FR=20mm/s, steering=0
cansend vcan0 0B9#0A001400323232

# Rear axle (ID 0xBA): RL=30mm/s, RR=40mm/s
cansend vcan0 0BA#1E00280032320000

# Run decoder on vcan0
ros2 launch canbus_decoder can_decoder.launch.py can_device:=vcan0

# Verify output
ros2 topic echo /can/wheel_velocities
```

#### CAN Message Protocol

The system decodes CAN messages from wheel encoder sensors:

**CAN IDs:**
- `0xB9 (185)`: Front axle (FL, FR wheels)
- `0xBA (186)`: Rear axle (RL, RR wheels)

**Byte Layout (8 bytes per message):**
```
Byte 0-1: Wheel 1 velocity (16-bit unsigned, mm/s, little-endian)
Byte 2-3: Wheel 2 velocity (16-bit unsigned, mm/s, little-endian)
Byte 4:   Suspension WSS 1 (8-bit unsigned, mm)
Byte 5:   Suspension WSS 2 (8-bit unsigned, mm)
Byte 6-7: Steering angle (16-bit signed, 0.1 deg units, little-endian)
          ⚠️ Only valid for front axle (ID 185)
```

**Vehicle Wheel Mapping:**
- Front axle: Wheel 1 = FL, Wheel 2 = FR
- Rear axle: Wheel 1 = RL, Wheel 2 = RR

#### CAN System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CAN Bus System Architecture                     │
└─────────────────────────────────────────────────────────────────────┘

Physical CAN Bus (500kbps)
    │
    ├─ ID 185 (Front): FL, FR wheel velocities + steering
    └─ ID 186 (Rear):  RL, RR wheel velocities
           ↓
    SocketCAN (Linux kernel driver)
           ↓
    ros2_socketcan bridge
           ↓
    /can/rx (can_msgs/Frame) ← Raw CAN frames
           ↓
    ┌──────────────────────────────────────────┐
    │      can_decoder_node                    │
    │  ┌────────────────────────────────┐     │
    │  │  can_protocol.py               │     │  ← Protocol definitions
    │  │  - decode_wheel_pair()         │     │  ← Byte-level decoding
    │  │  - WheelPairData dataclass     │     │  ← Unit conversions
    │  └────────────────────────────────┘     │
    │                                          │
    │  - Filters IDs 185/186                  │
    │  - Synchronizes front/rear axles        │
    │  - Converts mm/s → m/s                  │
    │  - Publishes structured topics          │
    └──────────────────────────────────────────┘
           ↓          ↓          ↓
    /can/wheel_velocities    /can/suspension    /can/steering_angle
    (Float32MultiArray)      (Float32MultiArray) (Float32)
     [FL,FR,RL,RR] m/s       [FL,FR,RL,RR] m    radians
           ↓
    ┌──────────────────────────────────────────┐
    │  CANAdapter                              │  ← Adapter pattern
    │  (vehicle_plotter/adapters/can_adapter)  │
    │                                          │
    │  - Subscribes to decoded topics         │
    │  - Registers with TimeSynchronizer      │
    │  - Implements compute_state()           │
    └──────────────────────────────────────────┘
           ↓
    DataCollectorNode
           ↓
    /vehicle_plotter/state (VehicleState)
           ↓          ↓
    PlotterNode   LoggerNode
```

**Component Responsibilities:**

1. **can_protocol.py** (Reusable Library)
   - Pure Python module for CAN message decoding
   - Defines CAN IDs (0xB9, 0xBA) and byte layouts
   - Implements `decode_wheel_pair()` function using `struct.unpack()`
   - Provides dataclasses with unit conversion properties (mm/s → m/s, decideg → rad)
   - **No ROS dependencies** - can be used in non-ROS applications

2. **can_decoder_node.py** (ROS2 Node)
   - ROS2 node that bridges raw CAN to structured topics
   - Subscribes to `/can/rx` (raw CAN frames from ros2_socketcan)
   - Uses can_protocol.py to decode byte data
   - Synchronizes front/rear axle messages
   - Publishes human-readable topics (wheel_velocities, suspension, steering)
   - Handles timeouts and stale data detection
   - Lives in **canbus_decoder** package

3. **can_adapter.py** (Sensor Adapter)
   - Integrates CAN data into vehicle_plotter system
   - Extends SensorAdapterInterface (follows GazeboAdapter pattern)
   - Subscribes to decoded CAN topics from can_decoder_node
   - Implements compute_state() to populate VehicleState message
   - Registers sensors with TimeSynchronizer
   - Lives in **vehicle_plotter** package

**Key Insight:** This separation allows:
- can_protocol.py to be tested independently with unit tests
- can_decoder_node to be reused in other applications beyond vehicle_plotter
- can_adapter.py to focus solely on vehicle_plotter integration

### VectorNav VN-200 IMU/GPS/INS (Real Hardware)

The vectornav_decoder package decodes binary serial data from the VectorNav VN-200 IMU/GPS/INS unit.

#### Hardware Setup (Jetson/Linux)
```bash
# Check serial port
ls /dev/ttyUSB*

# Verify permissions (add user to dialout group if needed)
sudo usermod -a -G dialout $USER

# Build vectornav_decoder package
colcon build --symlink-install --packages-select vectornav_decoder
source install/setup.bash
```

#### VectorNav Decoder Usage

**Option 1: Full System (Decoder + Plotter)**
```bash
# Terminal 1: VectorNav decoder (decodes serial to ROS topics)
ros2 launch vectornav_decoder vectornav_decoder.launch.py serial_port:=/dev/ttyUSB0

# Terminal 2: Vehicle plotter with VectorNav adapter
ros2 launch vehicle_plotter plotter.launch.py adapter:=vectornav

# Verify data flow
ros2 topic echo /vectornav/imu
ros2 topic echo /vectornav/gps
ros2 topic echo /vectornav/ins
ros2 topic echo /vehicle_plotter/state
```

**Option 2: VectorNav Monitor (Debugging Only)**
```bash
# Monitor raw VectorNav data (for debugging)
ros2 launch vectornav_decoder vectornav_monitor.launch.py

# With verbose output (shows all fields)
ros2 launch vectornav_decoder vectornav_monitor.launch.py verbose:=true
```

**Option 3: Custom Configuration**
```bash
# Use high-rate IMU config (400 Hz)
ros2 launch vectornav_decoder vectornav_decoder.launch.py \
    config_file:=/path/to/high_rate_imu.yaml

# Override serial settings
ros2 launch vectornav_decoder vectornav_decoder.launch.py \
    serial_port:=/dev/ttyACM0 baudrate:=115200
```

#### VectorNav Output Configuration

The VN-200 binary output is configurable via YAML. Edit `config/default_output.yaml`:

```yaml
# Output rate: 800 Hz / rate_divisor
output:
  rate_divisor: 4           # 200 Hz

binary_output_1:
  # Group 1: Common (YPR, angular rate, accel)
  group_common:
    enabled: true
    fields:
      - TimeStartup
      - YawPitchRoll
      - AngularRate
      - Accel

  # Group 3: IMU (raw measurements)
  group_imu:
    enabled: true
    fields:
      - UncompAccel
      - UncompGyro
      - Temp

  # Group 4: GPS
  group_gps:
    enabled: true
    fields:
      - NumSats
      - Fix
      - PosLla
      - VelNed
      - PosU
      - VelU

  # Group 6: INS (fused solution)
  group_ins:
    enabled: true
    fields:
      - InsStatus
      - PosLla
      - VelBody
      - VelNed
```

To add or remove fields, simply edit the YAML - no code changes required.

#### VectorNav Binary Protocol

The VN-200 uses a binary streaming protocol:

**Packet Structure:**
```
[Sync 0xFA][Groups 1B][Field1 2B]...[FieldN 2B][Payload][CRC 2B]
```

**Output Groups:**
| Group | Bit | Description |
|-------|-----|-------------|
| Common | 0x01 | General purpose (YPR, velocity, accel) |
| Time | 0x02 | Timestamps |
| IMU | 0x04 | Raw IMU measurements |
| GPS | 0x08 | GNSS receiver data |
| Attitude | 0x10 | Attitude estimation |
| INS | 0x20 | INS solution |

**Key Fields:**
- `YawPitchRoll`: Euler angles (degrees) - 3x float32
- `AngularRate`: Gyro (rad/s) - 3x float32
- `Accel`: Accelerometer (m/s^2) - 3x float32
- `PosLla`: GPS position (lat, lon, alt) - 3x float64
- `VelBody`: Body-frame velocity (m/s) - 3x float32

#### VectorNav System Architecture

```
VN-200 Hardware (921600 baud)
    │
    └─ USB/Serial (/dev/ttyUSB0)
           ↓
    ┌──────────────────────────────────────────┐
    │      vectornav_decoder_node              │
    │  ┌────────────────────────────────┐     │
    │  │  vn200_protocol.py             │     │  ← Binary parsing (NO ROS)
    │  │  - VN200Parser class           │     │  ← CRC validation
    │  │  - Dataclasses for each group  │     │  ← Unit conversions
    │  └────────────────────────────────┘     │
    │  ┌────────────────────────────────┐     │
    │  │  vn200_serial.py               │     │  ← pyserial wrapper
    │  │  - Connection management       │     │  ← Auto-reconnect
    │  │  - ASCII config commands       │     │  ← Thread-safe
    │  └────────────────────────────────┘     │
    │  ┌────────────────────────────────┐     │
    │  │  vn200_config.py               │     │  ← YAML config loader
    │  │  - Field name → bit mapping    │     │  ← Register command builder
    │  └────────────────────────────────┘     │
    │                                          │
    │  - Configures VN-200 on startup         │
    │  - Parses binary packets at 200 Hz      │
    │  - Publishes ROS2 messages              │
    └──────────────────────────────────────────┘
           ↓          ↓          ↓
    /vectornav/imu    /vectornav/gps    /vectornav/ins
    (sensor_msgs/Imu) (NavSatFix)       (nav_msgs/Odometry)
     200 Hz           5 Hz              200 Hz
           ↓
    ┌──────────────────────────────────────────┐
    │  VectorNavAdapter                        │  ← Adapter pattern
    │  (vehicle_plotter/adapters)              │
    │                                          │
    │  - Subscribes to decoded topics         │
    │  - Auto-sets GPS origin                 │
    │  - Converts GPS to local coords         │
    │  - Prefers INS for position             │
    │  - Implements compute_state()           │
    └──────────────────────────────────────────┘
           ↓
    DataCollectorNode
           ↓
    /vehicle_plotter/state (VehicleState)
           ↓          ↓
    PlotterNode   LoggerNode
```

**Component Responsibilities:**

1. **vn200_protocol.py** (Pure Python, NO ROS)
   - Binary packet parsing with CRC-16 validation
   - Byte-level field extraction using `struct`
   - Dataclasses for Common, IMU, GPS, INS data
   - Unit conversion properties (deg→rad)
   - Testable without ROS environment

2. **vn200_serial.py** (Pure Python, NO ROS)
   - pyserial wrapper with reconnection logic
   - ASCII command interface for configuration
   - Background thread for continuous reading
   - Callback mechanism for parsed packets

3. **vn200_config.py** (Pure Python, NO ROS)
   - YAML configuration loading
   - Field name to bit value mapping
   - ASCII register command builder

4. **vectornav_decoder_node.py** (ROS2 Node)
   - Publishes Imu, NavSatFix, Odometry messages
   - Configurable topics and frame IDs
   - Periodic statistics logging

5. **vectornav_adapter.py** (vehicle_plotter integration)
   - Implements SensorAdapterInterface
   - Auto-sets GPS origin from first fix
   - Converts GPS to local coordinates
   - Fuses INS/GPS/IMU into VehicleState

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
| `/can/rx` | Frame | ros2_socketcan |
| `/can/wheel_velocities` | Float32MultiArray | can_decoder_node |
| `/can/suspension` | Float32MultiArray | can_decoder_node |
| `/can/steering_angle` | Float32 | can_decoder_node |
| `/vectornav/imu` | Imu | vectornav_decoder_node |
| `/vectornav/gps` | NavSatFix | vectornav_decoder_node |
| `/vectornav/ins` | Odometry | vectornav_decoder_node |
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
- `config/can_topics.yaml` - CAN bus wheel encoder topics
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
# Run all tests
colcon test --packages-select sim_car vehicle_plotter canbus_decoder vectornav_decoder

# Run CAN protocol unit tests
colcon test --packages-select canbus_decoder
colcon test-result --verbose  # View test results

# Run VectorNav protocol unit tests
colcon test --packages-select vectornav_decoder
colcon test-result --verbose

# Or run directly with pytest
cd canbus_decoder
python3 -m pytest test/test_can_protocol.py -v

cd vectornav_decoder
python3 -m pytest test/test_vn200_protocol.py -v

# Monitor topics
ros2 topic list
ros2 topic echo /vehicle_plotter/state
ros2 topic echo /can/wheel_velocities
ros2 topic echo /vectornav/imu
ros2 topic echo /vectornav/gps

# Check rates
ros2 topic hz /vehicle_plotter/state
ros2 topic hz /can/wheel_velocities
ros2 topic hz /vectornav/imu
```
