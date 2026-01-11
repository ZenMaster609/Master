# Usage Guide

Quick reference for running the vehicle plotter system in different modes.

## Prerequisites

```bash
# Build the workspace
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Quick Start Commands

| Mode | Command |
|------|---------|
| Simulation | `ros2 launch vehicle_plotter sim.launch.py` |
| Jetson Only | `ros2 launch vehicle_plotter irl_jetson_only.launch.py` |
| Jetson + Windows | See [Multi-Machine Setup](#multi-machine-jetson--windows) |
| Virtual CAN Test | `ros2 launch vehicle_plotter vcan_test.launch.py` |
| Rosbag Replay | `ros2 launch vehicle_plotter replay.launch.py bag_path:=/path/to/bag` |

---

## Simulation Mode

Full Gazebo simulation with plotting and logging.

```bash
# Default (headless Gazebo, auto control)
ros2 launch vehicle_plotter sim.launch.py

# With Gazebo GUI
ros2 launch vehicle_plotter sim.launch.py headless:=false

# Keyboard control
ros2 launch vehicle_plotter sim.launch.py control_mode:=keyboard

# Disable plotting (logging only)
ros2 launch vehicle_plotter sim.launch.py enable_plot:=false

# Disable rosbag recording
ros2 launch vehicle_plotter sim.launch.py enable_rosbag:=false
```

**Output:**
- Plots: `./multidata/<run_id>/linux/plots/`
- Logs: `./multidata/<run_id>/linux/logs/`
- Rosbags: `./multidata/<run_id>/linux/rosbags/`

---

## IRL Jetson Only

Headless operation on Jetson with real hardware (CAN + VectorNav).

```bash
# Default (both CAN and VectorNav enabled)
ros2 launch vehicle_plotter irl_jetson_only.launch.py

# CAN only (no VectorNav)
ros2 launch vehicle_plotter irl_jetson_only.launch.py enable_vectornav:=false

# VectorNav only (no CAN)
ros2 launch vehicle_plotter irl_jetson_only.launch.py enable_can:=false

# Custom devices
ros2 launch vehicle_plotter irl_jetson_only.launch.py \
    can_device:=can1 serial_port:=/dev/ttyUSB1

# Disable rosbag (logging only)
ros2 launch vehicle_plotter irl_jetson_only.launch.py enable_rosbag:=false
```

**Output:**
- Logs: `./multidata/<run_id>/linux/logs/`
- Rosbags: `./multidata/<run_id>/linux/rosbags/`

---

## Multi-Machine (Jetson + Windows)

Live plotting on Windows while Jetson collects data.

### Step 1: Configure Networking

Both machines must be on the same network with matching DDS config.

```bash
# Edit DDS config with your IP addresses
nano config/dds/cyclonedds_wifi.xml
# Replace JETSON_IP and WINDOWS_IP with actual addresses
```

Set environment variables on **both machines**:

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///path/to/config/dds/cyclonedds_wifi.xml
```

### Step 2: Start Jetson

```bash
# On Jetson
ros2 launch vehicle_plotter irl_jetson_source.launch.py
```

### Step 3: Start Windows

```bash
# On Windows (after Jetson is running)
ros2 launch vehicle_plotter irl_windows_plotter.launch.py
```

Windows will:
- Sync `run_id` from Jetson via `/run_session` topic
- Display live plots
- Record its own rosbag
- Save plots on exit

**Output (both machines use same run_id):**
- Jetson: `./multidata/<run_id>/linux/`
- Windows: `./multidata/<run_id>/windows/`

### Troubleshooting

```bash
# Verify connectivity
ros2 topic list  # Should see topics from other machine

# Check run_session
ros2 topic echo /run_session

# See docs/NETWORKING.md for detailed troubleshooting
```

---

## Virtual CAN Test

Test the full pipeline without hardware using fake CAN data.

### Setup

```bash
# Create virtual CAN interface (once per boot)
./scripts/setup_vcan.sh
```

### Run

```bash
# Default (circle motion pattern)
ros2 launch vehicle_plotter vcan_test.launch.py

# Different motion patterns
ros2 launch vehicle_plotter vcan_test.launch.py mode:=static
ros2 launch vehicle_plotter vcan_test.launch.py mode:=linear
ros2 launch vehicle_plotter vcan_test.launch.py mode:=circle
ros2 launch vehicle_plotter vcan_test.launch.py mode:=random

# Adjust speed
ros2 launch vehicle_plotter vcan_test.launch.py base_velocity_mps:=2.0
```

**Output:**
- Same as simulation mode

---

## Rosbag Replay

Replay recorded data with visualization.

```bash
# Basic replay
ros2 launch vehicle_plotter replay.launch.py \
    bag_path:=/path/to/multidata/2026-01-09_14-12-33/linux/rosbags/bag

# Slow motion (0.5x speed)
ros2 launch vehicle_plotter replay.launch.py \
    bag_path:=/path/to/bag rate:=0.5

# Fast forward (2x speed)
ros2 launch vehicle_plotter replay.launch.py \
    bag_path:=/path/to/bag rate:=2.0

# Re-log data during replay (creates new session)
ros2 launch vehicle_plotter replay.launch.py \
    bag_path:=/path/to/bag enable_log:=true

# Specify adapter type
ros2 launch vehicle_plotter replay.launch.py \
    bag_path:=/path/to/bag adapter:=gazebo
```

---

## Standalone Parquet Plotting (Windows)

Plot logged data without ROS using the standalone script.

### Install Dependencies

```bash
pip install -r tools/requirements_tools.txt
```

### Usage

```bash
# Plot a session directory
python tools/plot_parquet.py /path/to/multidata/2026-01-09_14-12-33/linux/logs

# Plot a single parquet file
python tools/plot_parquet.py /path/to/vehicle_state_0000.parquet

# Save to PNG (no window)
python tools/plot_parquet.py /path/to/session --output report.png

# Interactive mode
python tools/plot_parquet.py /path/to/session --interactive

# Extended report (includes GPS plot)
python tools/plot_parquet.py /path/to/session --extended
```

---

## Ground Truth Capture

Capture system state for debugging and documentation.

```bash
# Capture current ROS state
./tools/capture_ground_truth.sh

# Capture to specific directory
./tools/capture_ground_truth.sh /path/to/output

# Capture with custom run_id
./tools/capture_ground_truth.sh ./multidata/2026-01-09_14-12-33/linux/ground_truth
```

**Output files:**
- `nodes.txt` - Active nodes
- `topics.txt` - Topics with types
- `system_info.json` - Hostname, IPs, env vars

---

## Common Parameters

### All Launch Files

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_log` | `true` | Enable Parquet logging |
| `enable_rosbag` | `true` | Enable rosbag recording |

### Plotter Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_plot` | `true` | Enable live plotting |
| `save_plots_on_exit` | `true` | Save PNG on shutdown |
| `save_plot_data_on_exit` | `true` | Save CSV data on shutdown |
| `update_rate_hz` | `30.0` | Plot refresh rate |

### Hardware Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `can_device` | `can0` | CAN interface |
| `serial_port` | `/dev/ttyUSB0` | VectorNav serial port |
| `output_rate_hz` | `50.0` | Data collector rate |

---

## Monitoring

```bash
# List active topics
ros2 topic list

# Check data flow
ros2 topic hz /vehicle_plotter/state

# View vehicle state
ros2 topic echo /vehicle_plotter/state

# Check run session
ros2 topic echo /run_session
```
