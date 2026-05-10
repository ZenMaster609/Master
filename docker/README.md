# Dockerized ROS2 Planner Environment

This setup builds a Docker image for Linux users with NVIDIA GPU and X11 GUI support. The Dockerfile recreates the ROS 2 Humble / Gazebo Fortress workspace and keeps the existing perception layout: custom OpenCV and cuDNN overlays under `~/ros2_ws`, plus the separate YOLO `.pt` virtualenv.

The image bundles `opencv_local`, `cudnn_py`, and `cuda_runtime` from runtime overlay tarballs. It recreates `yolo_pt_venv` from `requirements-yolo-pt.lock.txt` during the image build. The lockfiles are installed with pip dependency resolution disabled because they are snapshots of an already-working environment. The main ROS Python environment keeps Ubuntu's patched `setuptools`, which ROS Humble's `colcon --symlink-install` expects.

## Build

Package the local binary overlays before building the image:

```bash
cd ~/ros2_ws && ./src/Master/scripts/package-runtime-overlays.sh src/Master/docker/runtime-overlays
```

Build the image:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml build planner
```

Allow X11 windows from the container:

```bash
cd ~/ros2_ws && xhost +local:docker
```

## Run

Open an interactive shell:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner bash
```

Run the full simulator with Gazebo/RViz/RQT windows:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner ros2 launch sim_car full_sim_launch.launch.py
```

Run a headless smoke launch:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner ros2 launch sim_car full_sim_launch.launch.py headless:=true
```

## Development Mode

The development override bind-mounts the host repo into the container and uses container-local `build/`, `install/`, and `log/` volumes. Rebuild the workspace after source changes:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml -f src/Master/compose.dev.yaml run --rm planner colcon build --symlink-install
```

Then run the mounted workspace:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml -f src/Master/compose.dev.yaml run --rm planner ros2 launch sim_car full_sim_launch.launch.py
```

## Validation

Check that ROS sees the expected packages:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner bash -lc 'source install/setup.bash && ros2 pkg list | grep -E "sim_car|vehicle_plotter|eufs_gz_dynamics"'
```

Check the custom OpenCV overlay:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner bash -lc 'PYTHONPATH=/home/ros/ros2_ws/opencv_local/lib/python3.10/dist-packages LD_LIBRARY_PATH=/home/ros/ros2_ws/opencv_local/lib:/home/ros/ros2_ws/cudnn_py/nvidia/cudnn/lib:/home/ros/ros2_ws/cuda_runtime/lib python3 -c "import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())"'
```

Check the YOLO `.pt` virtualenv:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner bash -lc '/home/ros/ros2_ws/yolo_pt_venv/bin/python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"'
```

Run a launch-level smoke test:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner ros2 launch sim_car full_sim_launch.launch.py headless:=true yolo_enabled:=false
```

## Host Requirements

This setup targets Linux hosts with Docker, Docker Compose v2, NVIDIA Container Toolkit, and an NVIDIA driver new enough for the CUDA runtimes used by OpenCV and PyTorch. Windows and macOS Docker setups are intentionally out of scope.
