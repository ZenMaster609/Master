#!/bin/bash
# =============================================================================
# Setup Script for ROS2 Humble on NVIDIA Jetson (Headless)
# =============================================================================
# This script sets up a lightweight ROS2 environment for this workspace.
#
# Usage:
#   chmod +x setup-jetson.sh
#   ./setup-jetson.sh
#
# Tested on: Jetson Orin with JetPack 6.x (Ubuntu 22.04)
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  ROS2 Humble Setup for NVIDIA Jetson"
echo "  Workspace Setup"
echo "=============================================="
echo ""

# =============================================================================
# [1/9] Platform Detection
# =============================================================================
echo "[1/9] Detecting platform..."

# Check if running on Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo "  Jetson platform detected"
    cat /etc/nv_tegra_release
else
    echo "  Warning: /etc/nv_tegra_release not found"
    echo "  This script is designed for NVIDIA Jetson"
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Ubuntu version
if ! grep -q "22.04" /etc/os-release; then
    echo "  Warning: This script is designed for Ubuntu 22.04"
    echo "  You appear to be running a different version"
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "  Ubuntu 22.04 detected"

# =============================================================================
# [2/9] Update System Packages
# =============================================================================
echo ""
echo "[2/9] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# =============================================================================
# [3/9] Setup Locale
# =============================================================================
echo ""
echo "[3/9] Setting up locale..."
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# =============================================================================
# [4/9] Install Common Dependencies
# =============================================================================
echo ""
echo "[4/9] Installing common dependencies..."
sudo apt install -y \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    git \
    vim \
    nano \
    htop \
    tmux \
    wget \
    tree \
    bash-completion \
    build-essential \
    cmake \
    python3-pip \
    iproute2

# =============================================================================
# [5/9] Add ROS2 Humble Repository
# =============================================================================
echo ""
echo "[5/9] Adding ROS2 Humble repository..."
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# =============================================================================
# [6/9] Install ROS2 Humble (ros-base - lightweight, no GUI)
# =============================================================================
echo ""
echo "[6/9] Installing ROS2 Humble (ros-base)..."
echo "      (This may take a while...)"
sudo apt update
sudo apt install -y \
    ros-humble-ros-base \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-humble-tf2-ros \
    ros-humble-tf2-tools

# =============================================================================
# [7/9] Install Python Dependencies
# =============================================================================
echo ""
echo "[7/9] Installing Python dependencies..."
pip3 install --user \
    pyserial \
    pyyaml \
    numpy \
    pyarrow

# =============================================================================
# [8/9] Setup User Permissions
# =============================================================================
echo ""
echo "[8/9] Setting up user permissions..."
sudo usermod -a -G dialout $USER
echo "  Added $USER to dialout group (requires re-login)"

# =============================================================================
# Initialize rosdep
# =============================================================================
echo ""
echo "Initializing rosdep..."
sudo rosdep init 2>/dev/null || true
rosdep update

# =============================================================================
# Setup Shell Environment
# =============================================================================
echo ""
echo "Setting up shell environment..."

# Add ROS2 setup to bashrc if not already present
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# =============================================================================
# ROS2 Humble Environment (Jetson)
# =============================================================================
source /opt/ros/humble/setup.bash

# Source workspace if it exists
if [ -f ~/ros2_ws/install/setup.bash ]; then
    source ~/ros2_ws/install/setup.bash
fi

# Colcon build functions with auto-source
cb() {
    cd ~/ros2_ws && colcon build --symlink-install "$@" && source install/setup.bash
}
cbs() {
    cd ~/ros2_ws && colcon build --symlink-install --packages-select "$@" && source install/setup.bash
}

# Aliases
alias sb='source ~/ros2_ws/install/setup.bash'

# ROS2 Domain ID (change if needed for multi-robot setups)
export ROS_DOMAIN_ID=0

# Vehicle log path
export VEHICLE_LOG_PATH=~/.ros/logs

EOF
    echo "  Added ROS2 environment to ~/.bashrc"
fi

# Create workspace and logs directory
mkdir -p ~/ros2_ws/src
mkdir -p ~/.ros/logs
echo "  Created ~/ros2_ws/src and ~/.ros/logs directories"

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "[9/9] Verification:"
echo "  ROS2 version: $(ros2 --version 2>/dev/null || echo 'source bashrc first')"
echo "  User groups:  $(groups $USER | grep -o dialout || echo 'dialout not added yet')"
echo ""
echo "Next steps:"
echo ""
echo "1. Log out and back in (or reboot) for group changes:"
echo "   sudo reboot"
echo ""
echo "2. Clone/copy your project to ~/ros2_ws/src/:"
echo "   cd ~/ros2_ws/src"
echo "   git clone <your-repo-url> Master"
echo ""
echo "3. Build the workspace:"
echo "   cb"
echo ""
echo "Quick commands:"
echo "  cb         - Build entire workspace"
echo "  cbs <pkg>  - Build specific package"
echo "  sb         - Source workspace"
echo ""
echo "Logs: ~/.ros/logs/"
echo ""
