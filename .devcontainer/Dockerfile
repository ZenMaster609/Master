# Dev Container for ROS2 Humble + Gazebo Fortress + Vehicle Plotter
# Supports NVIDIA GPUs (3060 Ti, 3080, and other NVIDIA cards)

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ============================================
# System setup and locale
# ============================================
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# ROS2 Humble installation
# ============================================
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ============================================
# Gazebo Fortress repository
# ============================================
RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# ============================================
# Node.js for Claude Code CLI
# ============================================
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -

# ============================================
# Install ROS2, Gazebo, and development tools
# ============================================
RUN apt-get update && apt-get install -y \
    # ROS2 Humble Desktop (includes RViz, rqt, etc.)
    ros-humble-desktop \
    # ROS2 development tools
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    python3-vcstool \
    # Gazebo Fortress
    ignition-fortress \
    # ROS-Gazebo bridge packages
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-image \
    # CycloneDDS for better DDS performance
    ros-humble-rmw-cyclonedds-cpp \
    # Additional ROS2 packages for this project
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    ros-humble-rqt* \
    # Node.js for Claude Code
    nodejs \
    # Development tools
    git \
    vim \
    nano \
    htop \
    tmux \
    wget \
    tree \
    bash-completion \
    # Build essentials
    build-essential \
    cmake \
    gdb \
    # X11 and display utilities for GUI
    x11-apps \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    # NVIDIA OpenGL/EGL libraries for GPU rendering
    libnvidia-gl-535 \
    libnvidia-egl-wayland1 \
    libegl1 \
    libglvnd0 \
    libglx0 \
    # Qt5 for PyQtGraph GUI plotting
    python3-pyqt5 \
    python3-pyqt5.qtopengl \
    libqt5opengl5 \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Install Python packages for vehicle_plotter
# ============================================
RUN pip3 install --no-cache-dir \
    pyqtgraph \
    pyarrow \
    numpy

# ============================================
# Install Claude Code CLI globally
# ============================================
RUN npm install -g @anthropic-ai/claude-code

# ============================================
# Initialize rosdep
# ============================================
RUN rosdep init || true

# ============================================
# Create non-root user for development
# ============================================
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add user to video and render groups for GPU access
    && usermod -aG video $USERNAME \
    && usermod -aG render $USERNAME || true \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Set up user environment
# ============================================
USER $USERNAME
WORKDIR /home/$USERNAME

# Update rosdep as user
RUN rosdep update

# Create ROS2 workspace and Claude config directory
RUN mkdir -p /home/$USERNAME/ros2_ws/src \
    && mkdir -p /home/$USERNAME/.claude \
    && mkdir -p /home/$USERNAME/.ros/vehicle_logs

# Add ROS2 setup to bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "source /home/$USERNAME/ros2_ws/install/setup.bash 2>/dev/null || true" >> /home/$USERNAME/.bashrc \
    && echo "" >> /home/$USERNAME/.bashrc \
    && echo "# Auto-source after build" >> /home/$USERNAME/.bashrc \
    && echo "cb() { cd ~/ros2_ws && colcon build --symlink-install \"\$@\" && source install/setup.bash; }" >> /home/$USERNAME/.bashrc \
    && echo "cbs() { cd ~/ros2_ws && colcon build --symlink-install --packages-select \"\$@\" && source install/setup.bash; }" >> /home/$USERNAME/.bashrc \
    && echo "alias gs='ign gazebo'" >> /home/$USERNAME/.bashrc \
    && echo "" >> /home/$USERNAME/.bashrc \
    && echo "# Vehicle plotter shortcuts" >> /home/$USERNAME/.bashrc \
    && echo "alias plotter='ros2 launch vehicle_plotter plotter.launch.py'" >> /home/$USERNAME/.bashrc \
    && echo "alias bringup='ros2 launch vehicle_plotter bringup.launch.py'" >> /home/$USERNAME/.bashrc

# Set default shell
ENV SHELL=/bin/bash

WORKDIR /home/$USERNAME/ros2_ws
