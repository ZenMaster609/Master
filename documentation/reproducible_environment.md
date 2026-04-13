# Reproducible Thesis Environment

This repository currently runs from a mixed native ROS 2 environment, not from one self-contained virtualenv or container. Reproducing it means aligning the OS/ROS install, this repository commit, Python packages, and local runtime overlays used by perception.

## Reference Shape

The reference setup is Ubuntu 22.04 with ROS 2 Humble, Gazebo Fortress, Python 3.10, and local overlays under `~/ros2_ws`.

The important repo-tracked dependency files are:

- `requirements.txt`: loose development/runtime dependency list
- `requirements.user.lock.txt`: lock snapshot for the main user environment
- `requirements-yolo-pt.lock.txt`: lock snapshot for the separate YOLO `.pt` environment

The important local runtime overlays are:

- `~/ros2_ws/opencv_local`
- `~/ros2_ws/cudnn_py`
- `~/ros2_ws/yolo_pt_venv`

The launch file injects these paths into `perception_node`, so a plain `pip install -r requirements.txt` is not enough to reproduce camera perception.

## Runtime Overlay Injection

`sim_car/launch/full_sim_launch.launch.py` prepends these paths for `perception_node`:

- `~/ros2_ws/opencv_local/lib/python3.10/dist-packages`
- `~/ros2_ws/opencv_local/lib`
- `~/ros2_ws/cudnn_py/nvidia/cudnn/lib`
- `~/ros2_ws/yolo_pt_venv/lib/python3.10/site-packages`

This lets the perception node use the custom OpenCV, CUDA/cuDNN runtime pieces, and the YOLO `.pt` Python environment while the rest of the workspace stays in the normal ROS 2 environment.

## What To Keep Fixed

For another machine to behave like the reference machine, keep these aligned:

1. Ubuntu and ROS 2 major versions.
2. Gazebo Fortress / ROS-GZ packages.
3. The exact git commit of this repository.
4. Python package versions from the lockfiles.
5. The runtime overlays listed above.
6. The same YOLO weights under `sim_car/yolo/weights/` or an explicit `yolo_model_path`.

## Native Partner Setup

On the reference machine, package the non-repo runtime overlays:

```bash
cd ~/ros2_ws && ./src/Master/scripts/package-runtime-overlays.sh
```

Copy these to the partner machine:

- this repository at the same commit
- `~/ros2_ws/runtime-overlays/`

On the partner machine:

```bash
cd ~/ros2_ws && source /opt/ros/humble/setup.bash
cd ~/ros2_ws && rosdep install --from-paths src --ignore-src -r -y
cd ~/ros2_ws && ./src/Master/scripts/setup-partner-environment.sh
cd ~/ros2_ws && colcon build --symlink-install
cd ~/ros2_ws && source /opt/ros/humble/setup.bash
cd ~/ros2_ws && source install/setup.bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py
```

## Why This Is Not Fully Frozen

The native setup still depends on apt repositories, ROS package repositories, GPU driver compatibility, and local system libraries. The lockfiles and overlay copy make the setup much closer, but they do not guarantee long-term bit-for-bit reproducibility.

For exact reproduction months later, use a Docker/Apptainer image or full machine image in addition to this repository and the runtime overlays.
