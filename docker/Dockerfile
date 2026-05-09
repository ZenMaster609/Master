# syntax=docker/dockerfile:1.7

FROM osrf/ros:humble-desktop-full

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV ROS_WS=/home/${USERNAME}/ros2_ws
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/${USERNAME}/.local/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    gnupg \
    libgl1 \
    libglx-mesa0 \
    ignition-tools \
    libignition-gazebo6-dev \
    libignition-gazebo6-plugins \
    libignition-math6-dev \
    libignition-plugin-dev \
    libx11-6 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxext6 \
    libxrender1 \
    libyaml-cpp-dev \
    python3-colcon-common-extensions \
    python3-ignition-gazebo6 \
    python3-matplotlib \
    python3-numpy \
    python3-opencv \
    python3-pandas \
    python3-pip \
    python3-pyqt5 \
    python3-pyqtgraph \
    python3-rosdep \
    python3-venv \
    python3-vcstool \
    python3-yaml \
    ros-${ROS_DISTRO}-ackermann-msgs \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-ros-gz-bridge \
    ros-${ROS_DISTRO}-ros-gz-sim \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-rqt-gui \
    ros-${ROS_DISTRO}-rqt-gui-py \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-xacro \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN if ! getent group "${USER_GID}" >/dev/null; then groupadd --gid "${USER_GID}" "${USERNAME}"; fi \
    && if ! id --user "${USERNAME}" >/dev/null 2>&1; then \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}"; \
    fi \
    && usermod -aG sudo "${USERNAME}" \
    && for device_group in video render dialout; do \
        if getent group "${device_group}" >/dev/null; then usermod -aG "${device_group}" "${USERNAME}"; fi; \
    done \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/90-${USERNAME} \
    && chmod 0440 /etc/sudoers.d/90-${USERNAME}

RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi \
    && rosdep update --rosdistro "${ROS_DISTRO}"

WORKDIR ${ROS_WS}
RUN mkdir -p "${ROS_WS}/src" && chown -R "${USERNAME}:${USER_GID}" "/home/${USERNAME}"

COPY --chown=${USERNAME}:${USER_GID} requirements.user.lock.txt requirements-yolo-pt.lock.txt /tmp/planner-requirements/

USER ${USERNAME}

RUN --mount=type=cache,target=/home/${USERNAME}/.cache/pip,uid=${USER_UID},gid=${USER_GID} \
    python3 -m pip install --user --upgrade pip setuptools wheel \
    && python3 -m pip install --user -r /tmp/planner-requirements/requirements.user.lock.txt

RUN --mount=type=cache,target=/home/${USERNAME}/.cache/pip,uid=${USER_UID},gid=${USER_GID} \
    python3 -m venv --system-site-packages "${ROS_WS}/yolo_pt_venv" \
    && "${ROS_WS}/yolo_pt_venv/bin/python" -m pip install --upgrade pip setuptools wheel \
    && "${ROS_WS}/yolo_pt_venv/bin/python" -m pip install -r /tmp/planner-requirements/requirements-yolo-pt.lock.txt

USER root

COPY docker/runtime-overlays/opencv_local.tar.gz docker/runtime-overlays/cudnn_py.tar.gz /tmp/runtime-overlays/

RUN tar -C "${ROS_WS}" -xzf /tmp/runtime-overlays/opencv_local.tar.gz \
    && tar -C "${ROS_WS}" -xzf /tmp/runtime-overlays/cudnn_py.tar.gz \
    && chown -R "${USERNAME}:${USER_GID}" "${ROS_WS}/opencv_local" "${ROS_WS}/cudnn_py" \
    && rm -rf /tmp/runtime-overlays

COPY --chown=${USERNAME}:${USER_GID} \
    AGENTS.md \
    CLAUDE.md \
    refpr.txt \
    requirements.txt \
    requirements.user.lock.txt \
    requirements-yolo-pt.lock.txt \
    ${ROS_WS}/src/Master/

COPY --chown=${USERNAME}:${USER_GID} .dockerignore compose.yaml compose.dev.yaml ${ROS_WS}/src/Master/
COPY --chown=${USERNAME}:${USER_GID} docker/Dockerfile docker/entrypoint.sh docker/README.md ${ROS_WS}/src/Master/docker/
COPY --chown=${USERNAME}:${USER_GID} documentation ${ROS_WS}/src/Master/documentation
COPY --chown=${USERNAME}:${USER_GID} eufs_remastered ${ROS_WS}/src/Master/eufs_remastered
COPY --chown=${USERNAME}:${USER_GID} scripts ${ROS_WS}/src/Master/scripts
COPY --chown=${USERNAME}:${USER_GID} sim_car ${ROS_WS}/src/Master/sim_car
COPY --chown=${USERNAME}:${USER_GID} tools ${ROS_WS}/src/Master/tools
COPY --chown=${USERNAME}:${USER_GID} utilities ${ROS_WS}/src/Master/utilities
COPY --chown=${USERNAME}:${USER_GID} vehicle_plotter ${ROS_WS}/src/Master/vehicle_plotter
COPY --chown=${USERNAME}:${USER_GID} vehicle_plotter_msgs ${ROS_WS}/src/Master/vehicle_plotter_msgs

RUN rosdep update --rosdistro "${ROS_DISTRO}" \
    && source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && rosdep install --from-paths "${ROS_WS}/src" --ignore-src -r -y --rosdistro "${ROS_DISTRO}"

USER ${USERNAME}

RUN cd "${ROS_WS}" \
    && source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon build --symlink-install

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >>"/home/${USERNAME}/.bashrc" \
    && echo "if [ -f ${ROS_WS}/install/setup.bash ]; then source ${ROS_WS}/install/setup.bash; fi" >>"/home/${USERNAME}/.bashrc" \
    && echo "cd ${ROS_WS}" >>"/home/${USERNAME}/.bashrc"

COPY --chown=${USERNAME}:${USER_GID} docker/entrypoint.sh /usr/local/bin/ros2-planner-entrypoint

USER root
RUN chmod +x /usr/local/bin/ros2-planner-entrypoint
USER ${USERNAME}

WORKDIR ${ROS_WS}
ENTRYPOINT ["/usr/local/bin/ros2-planner-entrypoint"]
CMD ["bash"]
