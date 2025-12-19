#!/bin/bash
# Post-create script for dev container setup

set -e

echo "=========================================="
echo "Setting up ROS2 development environment"
echo "=========================================="

# Source ROS2
source /opt/ros/humble/setup.bash

# Navigate to workspace
cd ~/ros2_ws

# Install dependencies via rosdep
echo "Installing rosdep dependencies..."
rosdep install --from-paths src --ignore-src -r -y || true

# Build the workspace
echo "Building workspace..."
colcon build --symlink-install

# Source the workspace
source install/setup.bash

echo ""
echo "=========================================="
echo "Development environment ready!"
echo "=========================================="
echo ""
echo "Quick commands:"
echo "  cb       - Build entire workspace"
echo "  cbs pkg  - Build specific package"
echo "  sb       - Source workspace"
echo "  gs       - Launch Gazebo Sim"
echo ""
echo "Launch simulation:"
echo "  ros2 launch sim_car gazebo_sim.launch.py"
echo ""
echo "Launch control nodes:"
echo "  ros2 launch sim_car nodes.launch.py"
echo ""
