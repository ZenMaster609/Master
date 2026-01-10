# Multi-Machine Networking Setup

Guide for configuring ROS 2 communication between Jetson and Windows over WiFi.

## Prerequisites

- Both machines connected to the same network (WiFi or wired)
- ROS 2 Humble installed on both machines
- Same ROS_DOMAIN_ID on both machines
- Firewall configured to allow UDP traffic (ports 7400-7500)

## Quick Start

### 1. Find IP Addresses

**Jetson (Linux):**
```bash
ip addr show wlan0  # or eth0 for wired
# Look for "inet 192.168.x.x"
```

**Windows:**
```powershell
ipconfig
# Look for "IPv4 Address" under your WiFi adapter
```

### 2. Configure DDS

Edit the CycloneDDS config file to add your IP addresses:

```bash
# On both machines, edit the config:
nano config/dds/cyclonedds_wifi.xml

# Replace JETSON_IP and WINDOWS_IP with actual addresses:
# <Peer address="192.168.1.100"/>  <!-- Jetson -->
# <Peer address="192.168.1.101"/>  <!-- Windows -->
```

### 3. Set Environment Variables

**Jetson:**
```bash
# Add to ~/.bashrc
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/$USER/ros2_ws/src/Master/config/dds/cyclonedds_wifi.xml

# Apply changes
source ~/.bashrc
```

**Windows (PowerShell):**
```powershell
# Set for current session
$env:ROS_DOMAIN_ID = "42"
$env:RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"
$env:CYCLONEDDS_URI = "file:///C:/path/to/cyclonedds_wifi.xml"

# Or add to system environment variables for persistence
```

**Windows (WSL2):**
```bash
# Add to ~/.bashrc
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///mnt/c/path/to/cyclonedds_wifi.xml
```

### 4. Test Communication

**Terminal 1 (Jetson):**
```bash
ros2 topic pub /test std_msgs/String "data: 'hello from jetson'" -r 1
```

**Terminal 2 (Windows):**
```bash
ros2 topic echo /test
# Should see: data: 'hello from jetson'
```

## DDS Configuration Options

### CycloneDDS (Recommended)

Simpler configuration, good default performance.

**Config file:** `config/dds/cyclonedds_wifi.xml`

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///path/to/cyclonedds_wifi.xml
```

### FastDDS

More configuration options, may work better in some network environments.

**Config file:** `config/dds/fastdds_wifi.xml`

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/fastdds_wifi.xml
```

## Firewall Configuration

### Linux (Jetson)
```bash
# Allow ROS 2 DDS ports
sudo ufw allow 7400:7500/udp

# Or disable firewall for testing
sudo ufw disable
```

### Windows
```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "ROS2 DDS" -Direction Inbound -Protocol UDP -LocalPort 7400-7500 -Action Allow
New-NetFirewallRule -DisplayName "ROS2 DDS Out" -Direction Outbound -Protocol UDP -LocalPort 7400-7500 -Action Allow
```

### WSL2 Specific

WSL2 has network isolation that can block multicast. Options:

**Option 1: Mirror mode (Windows 11)**
Add to `%UserProfile%\.wslconfig`:
```ini
[wsl2]
networkingMode=mirrored
```

**Option 2: Port forwarding**
```powershell
# Forward ROS 2 ports
netsh interface portproxy add v4tov4 listenport=7400 listenaddress=0.0.0.0 connectport=7400 connectaddress=(WSL IP)
```

**Option 3: Use native Windows ROS 2**
Install ROS 2 natively on Windows instead of WSL2 for best network compatibility.

## Troubleshooting

### Check Basic Connectivity
```bash
# From Jetson, ping Windows
ping 192.168.1.101

# From Windows, ping Jetson
ping 192.168.1.100
```

### Verify ROS 2 Setup
```bash
# Should show same domain ID on both machines
echo $ROS_DOMAIN_ID

# Check DDS implementation
echo $RMW_IMPLEMENTATION

# Run diagnostics
ros2 doctor --report
```

### Common Issues

**"No topics visible on remote machine"**
- Verify ROS_DOMAIN_ID matches on both machines
- Check firewall allows UDP 7400-7500
- Ensure DDS config has correct peer IPs
- Try disabling multicast: set `AllowMulticast` to `false` in CycloneDDS config

**"Topics visible but no data"**
- Check QoS compatibility between publisher and subscriber
- Verify message types match (same package version)
- Check network latency: `ping -c 10 <remote_ip>`

**"Intermittent connectivity"**
- WiFi may have multicast filtering; use explicit peer list
- Increase lease duration in DDS config
- Check for IP address changes (use static IP if possible)

**"WSL2 can't communicate"**
- WSL2 has NAT by default; use mirrored networking or native Windows ROS 2
- Ensure Windows firewall allows traffic to WSL

### Debug DDS Discovery

**CycloneDDS:**
```bash
# Enable verbose logging
export CYCLONEDDS_URI='<CycloneDDS><Domain><Tracing><Category>discovery</Category></Tracing></Domain></CycloneDDS>'
```

**FastDDS:**
```bash
# Enable debug logging
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/fastdds_debug.xml
# Add <Verbosity>high</Verbosity> to the XML
```

### Network Performance

Check network quality:
```bash
# Measure latency
ping -c 100 192.168.1.100 | tail -1

# Test bandwidth (install iperf3)
# Server (Jetson):
iperf3 -s
# Client (Windows):
iperf3 -c 192.168.1.100
```

For high-frequency data (>100 Hz), consider:
- Wired Ethernet connection
- Dedicated WiFi network (5 GHz preferred)
- Reducing message size or rate

## Running Multi-Machine Setup

### Jetson (Data Source)
```bash
# Source workspace
source ~/ros2_ws/install/setup.bash

# Launch sensor nodes
ros2 launch vehicle_plotter irl_jetson_source.launch.py
```

### Windows (Plotter)
```bash
# Source workspace
source /path/to/ros2_ws/install/setup.bash

# Launch plotter (will sync run_id from Jetson)
ros2 launch vehicle_plotter irl_windows_plotter.launch.py
```

Both machines will use the same `run_id` and save data to their respective `multidata/<run_id>/<os>/` folders.

## Static IP Configuration (Recommended)

For reliable multi-machine operation, use static IPs.

### Jetson (Linux)
Edit `/etc/netplan/01-netcfg.yaml`:
```yaml
network:
  version: 2
  renderer: networkd
  wifis:
    wlan0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
      access-points:
        "YourWiFiName":
          password: "YourPassword"
```

Apply: `sudo netplan apply`

### Windows
1. Open Network Settings
2. Change adapter options
3. Right-click WiFi adapter > Properties
4. Select "Internet Protocol Version 4" > Properties
5. Use static IP: 192.168.1.101, Subnet: 255.255.255.0, Gateway: 192.168.1.1
