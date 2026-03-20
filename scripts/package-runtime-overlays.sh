#!/usr/bin/env bash

set -euo pipefail

WS_ROOT="${WS_ROOT:-$HOME/ros2_ws}"
REPO_ROOT="${REPO_ROOT:-$WS_ROOT/src/Master}"
OUT_DIR="${1:-$WS_ROOT/runtime-overlays}"

mkdir -p "${OUT_DIR}"

if [[ ! -d "${WS_ROOT}/opencv_local" ]]; then
    echo "Missing ${WS_ROOT}/opencv_local"
    exit 1
fi

if [[ ! -d "${WS_ROOT}/cudnn_py" ]]; then
    echo "Missing ${WS_ROOT}/cudnn_py"
    exit 1
fi

tar -C "${WS_ROOT}" -czf "${OUT_DIR}/opencv_local.tar.gz" opencv_local
tar -C "${WS_ROOT}" -czf "${OUT_DIR}/cudnn_py.tar.gz" cudnn_py

cp "${REPO_ROOT}/requirements.user.lock.txt" "${OUT_DIR}/"
cp "${REPO_ROOT}/requirements-yolo-pt.lock.txt" "${OUT_DIR}/"

sha256sum \
    "${OUT_DIR}/opencv_local.tar.gz" \
    "${OUT_DIR}/cudnn_py.tar.gz" \
    > "${OUT_DIR}/SHA256SUMS"

if command -v git >/dev/null 2>&1; then
    git -C "${REPO_ROOT}" rev-parse HEAD > "${OUT_DIR}/MASTER_GIT_COMMIT"
fi

cat <<EOF
Runtime overlays packaged in:
  ${OUT_DIR}

Copy this directory to the partner machine at:
  ${WS_ROOT}/runtime-overlays
EOF
