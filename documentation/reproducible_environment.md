# Reproducible Thesis Environment

This repository currently runs from a mixed native environment, not from a single self-contained virtualenv or container. The working reference machine uses:

- Ubuntu 22.04.5 LTS (`x86_64`)
- ROS 2 Humble
- Python `3.10.12`
- `ros-humble-desktop=0.10.0-1jammy.20260219.051411`
- `ros-humble-ros-gz-sim=0.244.22-1jammy.20260217.054107`
- `ros-humble-ros-gz-bridge=0.244.22-1jammy.20260217.053415`
- `ros-humble-cv-bridge=3.2.1-1jammy.20260217.044540`
- `python3-opencv=4.5.4+dfsg-9ubuntu4`
- `python3-pyqt5=5.15.6+dfsg-1ubuntu3`
- `gz-fortress=1.0.3-2~jammy`
- `gz-tools=1.5.0-1~jammy`
- Custom OpenCV install in `~/ros2_ws/opencv_local` with OpenCV `4.8.1`
- Custom cuDNN/CUDA runtime overlay in `~/ros2_ws/cudnn_py`
- Separate YOLO venv in `~/ros2_ws/yolo_pt_venv`

## What To Freeze

To make another machine behave like the reference machine, keep these four pieces aligned:

1. Same Ubuntu and ROS major versions.
2. Same git commit of this repository.
3. Same Python package versions.
4. Same external runtime overlays: `opencv_local` and `cudnn_py`.

`requirements.txt` is only a loose development dependency file. The exact reference environment is captured in:

- `requirements.user.lock.txt`
- `requirements-yolo-pt.lock.txt`

## Recommended Native Workflow

On the reference machine, package the non-repo runtime overlays:

```bash
cd ~/ros2_ws && ./src/Master/scripts/package-runtime-overlays.sh
```

Copy these to the partner machine:

- The repo at the same git commit
- `~/ros2_ws/runtime-overlays/`

On the partner machine:

```bash
cd ~/ros2_ws && source /opt/ros/humble/setup.bash
cd ~/ros2_ws && rosdep install --from-paths src --ignore-src -r -y
cd ~/ros2_ws && ./src/Master/scripts/setup-partner-environment.sh
cd ~/ros2_ws && source /opt/ros/humble/setup.bash
cd ~/ros2_ws && source install/setup.bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py
```

## Why The Overlay Copy Matters

The launch file currently injects these paths into `perception_node`:

- `~/ros2_ws/opencv_local/lib/python3.10/dist-packages`
- `~/ros2_ws/opencv_local/lib`
- `~/ros2_ws/cudnn_py/nvidia/cudnn/lib`
- `~/ros2_ws/yolo_pt_venv/lib/python3.10/site-packages`

That means a plain `pip install -r requirements.txt` is not enough to reproduce the working setup.

## Stronger Option

If you need exact reproducibility months later, use a Docker/Apptainer image or a full machine image in addition to the files above. Native `apt` + `pip` lockfiles get close, but they are still tied to whatever package versions remain available in the upstream repositories.
