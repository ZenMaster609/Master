#!/usr/bin/env bash

set -e

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"

if [[ -f "${ROS_WS:-$HOME/ros2_ws}/install/setup.bash" ]]; then
    source "${ROS_WS:-$HOME/ros2_ws}/install/setup.bash"
fi

cd "${ROS_WS:-$HOME/ros2_ws}"

exec "$@"
