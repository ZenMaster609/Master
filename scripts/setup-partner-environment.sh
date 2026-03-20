#!/usr/bin/env bash

set -euo pipefail

WS_ROOT="${WS_ROOT:-$HOME/ros2_ws}"
REPO_ROOT="${REPO_ROOT:-$WS_ROOT/src/Master}"
RUNTIME_ARCHIVE_DIR="${RUNTIME_ARCHIVE_DIR:-$WS_ROOT/runtime-overlays}"
YOLO_VENV="${YOLO_VENV:-$WS_ROOT/yolo_pt_venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

restore_archive_if_needed() {
    local archive_name="$1"
    local target_name="$2"
    local archive_path="${RUNTIME_ARCHIVE_DIR}/${archive_name}"
    local target_path="${WS_ROOT}/${target_name}"

    if [[ -d "${target_path}" ]]; then
        return
    fi

    if [[ -f "${archive_path}" ]]; then
        echo "Restoring ${target_name} from ${archive_path}"
        tar -C "${WS_ROOT}" -xzf "${archive_path}"
    fi
}

if [[ ! -f "${REPO_ROOT}/requirements.user.lock.txt" ]]; then
    echo "Missing ${REPO_ROOT}/requirements.user.lock.txt"
    exit 1
fi

if [[ ! -f "${REPO_ROOT}/requirements-yolo-pt.lock.txt" ]]; then
    echo "Missing ${REPO_ROOT}/requirements-yolo-pt.lock.txt"
    exit 1
fi

restore_archive_if_needed "opencv_local.tar.gz" "opencv_local"
restore_archive_if_needed "cudnn_py.tar.gz" "cudnn_py"

if [[ ! -d "${WS_ROOT}/opencv_local" ]]; then
    echo "Warning: ${WS_ROOT}/opencv_local is missing."
    echo "Exact perception runtime will not match the reference machine."
fi

if [[ ! -d "${WS_ROOT}/cudnn_py" ]]; then
    echo "Warning: ${WS_ROOT}/cudnn_py is missing."
    echo "Custom OpenCV CUDA/cuDNN loading will not match the reference machine."
fi

"${PYTHON_BIN}" -m pip install --user -r "${REPO_ROOT}/requirements.user.lock.txt"

"${PYTHON_BIN}" -m venv --system-site-packages "${YOLO_VENV}"
"${YOLO_VENV}/bin/python" -m pip install --upgrade pip
"${YOLO_VENV}/bin/python" -m pip install -r "${REPO_ROOT}/requirements-yolo-pt.lock.txt"

source /opt/ros/humble/setup.bash
cd "${WS_ROOT}"
colcon build --symlink-install

cat <<EOF
Partner environment setup complete.

Next commands:
cd ~/ros2_ws && source /opt/ros/humble/setup.bash
cd ~/ros2_ws && source install/setup.bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py
EOF
