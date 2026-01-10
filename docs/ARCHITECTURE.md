# System Architecture

## Package Overview

```mermaid
graph TB
    subgraph "Hardware Layer"
        CAN[CAN Bus<br/>Wheel Encoders]
        VN[VectorNav VN-200<br/>IMU/GPS/INS]
        GZ[Gazebo Fortress<br/>Simulation]
    end

    subgraph "Decoder Packages"
        CD[canbus_decoder<br/>can_decoder_node]
        VD[vectornav_decoder<br/>vectornav_decoder_node]
        SC[sim_car<br/>sensor nodes]
    end

    subgraph "vehicle_plotter Package"
        SM[SessionManager<br/>run_id broadcast]
        DC[DataCollector<br/>adapter pattern]
        PN[PlotterNode<br/>PyQtGraph]
        LN[LoggerNode<br/>Parquet/CSV]
        RB[RosbagController<br/>ros2 bag record]
    end

    subgraph "Storage"
        MD[(multidata/<br/>run_id/)]
        LOGS[logs/]
        BAGS[rosbags/]
        PLOTS[plots/]
        GT[ground_truth/]
    end

    CAN --> CD
    VN --> VD
    GZ --> SC

    CD --> DC
    VD --> DC
    SC --> DC

    SM --> LN
    SM --> PN
    SM --> RB

    DC --> PN
    DC --> LN

    LN --> LOGS
    RB --> BAGS
    PN --> PLOTS
    SM --> GT

    MD --- LOGS
    MD --- BAGS
    MD --- PLOTS
    MD --- GT
```

## Topic Graph

```mermaid
graph LR
    subgraph "Sensor Topics"
        IMU[/vectornav/imu<br/>sensor_msgs/Imu]
        GPS[/vectornav/gps<br/>sensor_msgs/NavSatFix]
        INS[/vectornav/ins<br/>nav_msgs/Odometry]
        WV[/can/wheel_velocities<br/>Float32MultiArray]
        SUS[/can/suspension<br/>Float32MultiArray]
        STR[/can/steering_angle<br/>Float32]
    end

    subgraph "Simulation Topics"
        SIMU[/imu<br/>sensor_msgs/Imu]
        SIMGPS[/navsat<br/>sensor_msgs/NavSatFix]
        ODOM[/odom<br/>nav_msgs/Odometry]
        ENC[/wheel_encoder/*]
    end

    subgraph "Processing"
        DC((DataCollector))
    end

    subgraph "Output Topics"
        VS[/vehicle_plotter/state<br/>VehicleState]
        RS[/run_session<br/>RunSession]
    end

    IMU --> DC
    GPS --> DC
    INS --> DC
    WV --> DC
    SUS --> DC
    STR --> DC

    SIMU --> DC
    SIMGPS --> DC
    ODOM --> DC
    ENC --> DC

    DC --> VS

    SM((SessionManager)) --> RS
```

## Operational Modes

### 1. Simulation Mode
```mermaid
sequenceDiagram
    participant GZ as Gazebo
    participant SC as sim_car
    participant DC as DataCollector
    participant PN as Plotter
    participant LN as Logger
    participant RB as Rosbag

    GZ->>SC: sensor data
    SC->>DC: /imu, /odom, /navsat
    DC->>PN: /vehicle_plotter/state
    DC->>LN: /vehicle_plotter/state
    DC->>RB: record topics
    PN->>PN: display plots
    LN->>LN: write parquet
```

### 2. IRL Jetson Only
```mermaid
sequenceDiagram
    participant HW as Hardware
    participant CD as CAN Decoder
    participant VD as VectorNav
    participant DC as DataCollector
    participant LN as Logger
    participant RB as Rosbag

    HW->>CD: CAN frames
    HW->>VD: serial data
    CD->>DC: /can/*
    VD->>DC: /vectornav/*
    DC->>LN: /vehicle_plotter/state
    DC->>RB: record topics
    Note over LN,RB: Headless - no plotter
```

### 3. IRL with Windows Plotting
```mermaid
sequenceDiagram
    participant HW as Hardware
    participant JET as Jetson
    participant NET as WiFi Network
    participant WIN as Windows

    rect rgb(200, 220, 255)
        Note over JET: Jetson Side
        HW->>JET: sensor data
        JET->>JET: decode + collect
        JET->>JET: log + rosbag
        JET->>NET: publish /run_session
        JET->>NET: publish /vehicle_plotter/state
    end

    rect rgb(255, 220, 200)
        Note over WIN: Windows Side
        NET->>WIN: subscribe topics
        WIN->>WIN: sync run_id
        WIN->>WIN: plot + log + rosbag
        WIN->>WIN: save plots on exit
    end
```

### 4. Virtual CAN Test
```mermaid
sequenceDiagram
    participant VP as VCAN Publisher
    participant CD as CAN Decoder
    participant DC as DataCollector
    participant PN as Plotter

    VP->>VP: generate fake frames
    VP->>CD: /to_can_bus
    CD->>DC: /can/wheel_velocities
    DC->>PN: /vehicle_plotter/state
    PN->>PN: display plots
```

### 5. Rosbag Replay
```mermaid
sequenceDiagram
    participant BAG as Rosbag
    participant DC as DataCollector
    participant PN as Plotter
    participant LN as Logger

    BAG->>BAG: ros2 bag play
    BAG->>DC: original topics
    DC->>PN: /vehicle_plotter/state
    DC->>LN: optional re-log
    PN->>PN: display + save plots
```

## Storage Layout

```
./multidata/
└── 2026-01-09_14-12-33/          # run_id (timestamp)
    ├── linux/                     # Jetson data
    │   ├── logs/
    │   │   ├── vehicle_state_0000.parquet
    │   │   └── metadata.json
    │   ├── rosbags/
    │   │   └── bag/
    │   ├── plots/
    │   │   └── plots_20260109_141530.png
    │   ├── plot_data/
    │   │   ├── trajectory.csv
    │   │   └── velocity.csv
    │   ├── ground_truth/
    │   │   ├── nodes.txt
    │   │   ├── topics.txt
    │   │   └── system_info.json
    │   └── session_info.json
    │
    └── windows/                   # Windows data (same structure)
        ├── logs/
        ├── rosbags/
        ├── plots/
        ├── plot_data/
        ├── ground_truth/
        └── session_info.json
```

## Message Types

### VehicleState.msg
```
float64 timestamp
float64 x, y
float64 vx, vy
float64 yaw, yaw_rate
float64 speed, distance_traveled
int32[4] encoder_ticks
float32[4] encoder_velocities
float64 gps_latitude, gps_longitude
bool gps_valid
string estimation_status
```

### RunSession.msg
```
string run_id
string base_path
string originator_hostname
uint32 ros_domain_id
builtin_interfaces/Time start_time
```
