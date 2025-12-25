#!/bin/bash
# Post-create script for dev container setup
# Builds all ROS2 packages: sim_car, vehicle_plotter_msgs, vehicle_plotter

set -e

echo "=========================================="
echo "Setting up ROS2 Multi-Package Workspace"
echo "=========================================="

# Source ROS2
source /opt/ros/humble/setup.bash

# Navigate to workspace
cd ~/ros2_ws

# List packages
echo ""
echo "Packages found in workspace:"
ls -1 src/

# Install dependencies via rosdep
echo ""
echo "Installing rosdep dependencies..."
rosdep install --from-paths src --ignore-src -r -y || true

# Build all packages in dependency order
echo ""
echo "Building packages..."

# First build message package (CMake)
echo "  [1/3] Building vehicle_plotter_msgs..."
colcon build --symlink-install --packages-select vehicle_plotter_msgs

# Source to make messages available
source install/setup.bash

# Build sim_car (no deps on vehicle_plotter)
echo "  [2/3] Building sim_car..."
colcon build --symlink-install --packages-select sim_car

# Build vehicle_plotter (depends on vehicle_plotter_msgs)
echo "  [3/3] Building vehicle_plotter..."
colcon build --symlink-install --packages-select vehicle_plotter

# Source the complete workspace
source install/setup.bash

echo ""
echo "=========================================="
echo "Development environment ready!"
echo "=========================================="
echo ""
echo "Packages built:"
echo "  - sim_car            : Gazebo car simulation"
echo "  - vehicle_plotter_msgs: Custom ROS messages"
echo "  - vehicle_plotter    : Real-time plotting & logging"
echo ""
echo "Quick commands:"
echo "  cb           - Build entire workspace"
echo "  cbs pkg      - Build specific package"
echo "  bringup      - Launch full system (simulation + plotter)"
echo "  plotter      - Launch plotter only"
echo ""
echo "Launch simulation only:"
echo "  ros2 launch sim_car gazebo_sim.launch.py"
echo "  ros2 launch sim_car nodes.launch.py"
echo ""
echo "Launch full system (simulation + plotter + logger):"
echo "  ros2 launch vehicle_plotter bringup.launch.py"
echo ""
echo "Launch with auto control:"
echo "  ros2 launch vehicle_plotter bringup.launch.py control_mode:=auto"
echo ""
echo "Plotting only (attach to running simulation):"
echo "  ros2 launch vehicle_plotter plotter.launch.py"
echo ""
echo "Logs are saved to: ~/.ros/vehicle_logs/"
echo ""
