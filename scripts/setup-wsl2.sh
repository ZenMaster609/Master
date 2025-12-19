#!/bin/bash
# =============================================================================
# Setup Script for ROS2 Humble + Gazebo Fortress Development
# =============================================================================
# This script sets up a complete development environment for the sim_car project
# Works on both WSL2 (Windows) and native Ubuntu 22.04
#
# Usage:
#   chmod +x setup-wsl2.sh
#   ./setup-wsl2.sh
#
# For WSL2: Automatically configures OGRE1 renderer for compatibility
# For Native Linux: Uses default OGRE2 renderer with full GPU support
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  ROS2 Humble + Gazebo Fortress Setup"
echo "  For WSL2 Ubuntu 22.04"
echo "=============================================="
echo ""

# Check Ubuntu version
if ! grep -q "22.04" /etc/os-release; then
    echo "Warning: This script is designed for Ubuntu 22.04"
    echo "You appear to be running a different version"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo ""
echo "[1/8] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install locale
echo ""
echo "[2/8] Setting up locale..."
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Install common dependencies
echo ""
echo "[3/8] Installing common dependencies..."
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
    gdb \
    python3-pip

# Add ROS2 repository
echo ""
echo "[4/8] Adding ROS2 Humble repository..."
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Add Gazebo Fortress repository
echo ""
echo "[5/8] Adding Gazebo Fortress repository..."
sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Install ROS2 and Gazebo
echo ""
echo "[6/8] Installing ROS2 Humble and Gazebo Fortress..."
echo "      (This may take a while...)"
sudo apt update
sudo apt install -y \
    ros-humble-desktop \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ignition-fortress \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-image \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    ros-humble-rqt* \
    ros-humble-foxglove-bridge

# Initialize rosdep
echo ""
echo "[7/8] Initializing rosdep..."
sudo rosdep init 2>/dev/null || true
rosdep update

# Install Node.js and Claude Code CLI
echo ""
echo "[8/8] Installing Node.js and Claude Code CLI..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
sudo npm install -g @anthropic-ai/claude-code

# Set up Ignition Gazebo to use OGRE1 (fixes WSL2 rendering issues)
# Only apply this fix on WSL2, native Linux works fine with OGRE2
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo ""
    echo "WSL2 detected - Configuring Ignition Gazebo for WSL2 compatibility..."
    mkdir -p ~/.ignition/gazebo/6
    if [ -f /usr/share/ignition/ignition-gazebo6/gui/gui.config ]; then
        sed 's/<engine>ogre2<\/engine>/<engine>ogre<\/engine>/g' \
            /usr/share/ignition/ignition-gazebo6/gui/gui.config > ~/.ignition/gazebo/6/gui.config
        echo "Configured Gazebo to use OGRE1 renderer (WSL2 compatible)"
    fi
else
    echo ""
    echo "Native Linux detected - using default OGRE2 renderer"
fi

# Set up bashrc
echo ""
echo "Setting up shell environment..."

# Add ROS2 setup to bashrc if not already present
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# =============================================================================
# ROS2 Humble Environment
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
alias gs='ign gazebo'
alias sb='source ~/ros2_ws/install/setup.bash'

# ROS2 Domain ID (change if needed for multi-robot setups)
export ROS_DOMAIN_ID=0

# Gazebo model path
export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:~/ros2_ws/src

EOF
    echo "Added ROS2 environment to ~/.bashrc"
fi

# Create workspace
echo ""
echo "Creating ROS2 workspace..."
mkdir -p ~/ros2_ws/src

# Final message
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Close and reopen your terminal (or run: source ~/.bashrc)"
echo ""
echo "2. Clone/copy your project to ~/ros2_ws/src/:"
echo "   cd ~/ros2_ws/src"
echo "   # If using git:"
echo "   git clone <your-repo-url> sim_car"
echo "   # Or copy from Windows:"
echo "   cp -r /mnt/e/master/sim_car ."
echo ""
echo "3. Build the workspace:"
echo "   cb"
echo ""
echo "4. Launch the simulation:"
echo "   ros2 launch sim_car gazebo_sim.launch.py"
echo ""
echo "Quick commands:"
echo "  cb        - Build entire workspace"
echo "  cbs <pkg> - Build specific package"
echo "  sb        - Source workspace"
echo "  gs        - Launch Gazebo Sim"
echo ""
echo "GPU Note: WSL2 with WSLg should automatically use your NVIDIA GPU."
echo "Check with: glxinfo | grep 'OpenGL renderer'"
echo ""
