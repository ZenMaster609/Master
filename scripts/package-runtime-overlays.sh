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

CUDA_RUNTIME_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "${CUDA_RUNTIME_DIR}"
}
trap cleanup EXIT

mkdir -p "${CUDA_RUNTIME_DIR}/cuda_runtime/lib"

shopt -s nullglob
cuda_runtime_libs=(
    /usr/lib/x86_64-linux-gnu/libcublas.so*
    /usr/lib/x86_64-linux-gnu/libcublasLt.so*
    /usr/lib/x86_64-linux-gnu/libcudart.so*
    /usr/lib/x86_64-linux-gnu/libcufft.so*
    /usr/lib/x86_64-linux-gnu/libculibos.so*
    /usr/lib/x86_64-linux-gnu/libnpp*.so*
)

if (( ${#cuda_runtime_libs[@]} == 0 )); then
    echo "Missing CUDA runtime libraries under /usr/lib/x86_64-linux-gnu"
    exit 1
fi

cp -a "${cuda_runtime_libs[@]}" "${CUDA_RUNTIME_DIR}/cuda_runtime/lib/"
shopt -u nullglob

tar -C "${WS_ROOT}" -czf "${OUT_DIR}/opencv_local.tar.gz" opencv_local
tar -C "${WS_ROOT}" -czf "${OUT_DIR}/cudnn_py.tar.gz" cudnn_py
tar -C "${CUDA_RUNTIME_DIR}" -czf "${OUT_DIR}/cuda_runtime.tar.gz" cuda_runtime

cp "${REPO_ROOT}/requirements.user.lock.txt" "${OUT_DIR}/"
cp "${REPO_ROOT}/requirements-yolo-pt.lock.txt" "${OUT_DIR}/"

sha256sum \
    "${OUT_DIR}/opencv_local.tar.gz" \
    "${OUT_DIR}/cudnn_py.tar.gz" \
    "${OUT_DIR}/cuda_runtime.tar.gz" \
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
