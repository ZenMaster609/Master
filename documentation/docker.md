# Docker Handoff

This project uses a Docker image built from `src/Master/docker/Dockerfile`. A Docker image is the packaged environment; a container is a running instance of that image.

A GitHub clone is not enough by itself. The custom OpenCV/CUDA runtime files and the prebuilt Docker image are large and are intentionally not tracked in git.

## Host Requirements

The target machine must be Linux with:

- Docker and Docker Compose v2
- NVIDIA driver
- NVIDIA Container Toolkit
- Enough free disk space; 50 GB or more is recommended

Check Docker GPU access:

```bash
cd ~/ros2_ws && docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that command does not show the GPU, fix the host Docker/NVIDIA setup before using this project image.

## Recommended Handoff: One Archive Plus GitHub

This is the intended handoff for the counselor.

Give them:

- The GitHub repository link
- `master-planner-docker-handoff.tar`

The archive is created on the source machine with:

```bash
cd ~/ros2_ws && tar -cf master-planner-docker-handoff.tar docker-handoff
```

In the current handoff, `master-planner-docker-handoff.tar` contains:

```text
docker-handoff/master-planner-humble.tar.gz
docker-handoff/opencv_local.tar.gz
docker-handoff/cudnn_py.tar.gz
docker-handoff/cuda_runtime.tar.gz
docker-handoff/SHA256SUMS
```

On their machine, clone the repository to:

```text
~/ros2_ws/src/Master
```

Place `master-planner-docker-handoff.tar` in:

```text
~/ros2_ws/
```

Extract it:

```bash
cd ~/ros2_ws && tar -xf master-planner-docker-handoff.tar
```

Load the prebuilt Docker image:

```bash
cd ~/ros2_ws && docker load < docker-handoff/master-planner-humble.tar.gz
```

Copy the rebuild overlays into the repository, so future rebuilds work:

```bash
cd ~/ros2_ws && mkdir -p src/Master/docker/runtime-overlays
cd ~/ros2_ws && cp docker-handoff/opencv_local.tar.gz docker-handoff/cudnn_py.tar.gz docker-handoff/cuda_runtime.tar.gz docker-handoff/SHA256SUMS src/Master/docker/runtime-overlays/
```

Validate that Docker sees the image:

```bash
cd ~/ros2_ws && docker images | grep master-planner
```

Run the stack:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=midpoint controller:=pure_pursuit
```

## Option 1: They Build The Image

Use this when the next person may edit or rebuild the stack.

Give them:

- The GitHub repository cloned to `~/ros2_ws/src/Master`
- `opencv_local.tar.gz`
- `cudnn_py.tar.gz`
- `cuda_runtime.tar.gz`

On your machine, generate those overlay archives:

```bash
cd ~/ros2_ws && ./src/Master/scripts/package-runtime-overlays.sh src/Master/docker/runtime-overlays
```

Send these files from `~/ros2_ws/src/Master/docker/runtime-overlays/`:

- `opencv_local.tar.gz`
- `cudnn_py.tar.gz`
- `cuda_runtime.tar.gz`
- `SHA256SUMS`

On their machine, place the three `.tar.gz` files under:

```text
~/ros2_ws/src/Master/docker/runtime-overlays/
```

Then build the image:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml build planner
```

Validate custom OpenCV:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner bash -lc 'PYTHONPATH=/home/ros/ros2_ws/opencv_local/lib/python3.10/dist-packages LD_LIBRARY_PATH=/home/ros/ros2_ws/opencv_local/lib:/home/ros/ros2_ws/cudnn_py/nvidia/cudnn/lib:/home/ros/ros2_ws/cuda_runtime/lib python3 -c "import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())"'
```

Validate YOLO/PyTorch:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner bash -lc 'PYTHONNOUSERSITE=1 PYTHONPATH=/home/ros/ros2_ws/opencv_local/lib/python3.10/dist-packages:/home/ros/ros2_ws/yolo_pt_venv/lib/python3.10/site-packages LD_LIBRARY_PATH=/home/ros/ros2_ws/opencv_local/lib:/home/ros/ros2_ws/cudnn_py/nvidia/cudnn/lib:/home/ros/ros2_ws/cuda_runtime/lib python3 -c "import cv2, torch; from ultralytics import YOLO; print(cv2.__version__, torch.__version__, torch.cuda.is_available())"'
```

Run a shell inside the environment:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner bash
```

Run the stack:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=midpoint controller:=pure_pursuit
```

## Option 2: You Give Them The Finished Image

Use this when the next person mainly needs the stack to run.

After your image is built, export it:

```bash
cd ~/ros2_ws && docker save master-planner:humble | gzip > master-planner-humble.tar.gz
```

Give them:

- `master-planner-humble.tar.gz`
- The GitHub repository cloned to `~/ros2_ws/src/Master`, so they have `compose.yaml`, docs, launch files, and development mode files

On their machine, load the image:

```bash
cd ~/ros2_ws && docker load < master-planner-humble.tar.gz
```

Validate that Docker sees the image:

```bash
cd ~/ros2_ws && docker images | grep master-planner
```

Run a shell inside the environment:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner bash
```

Run the stack:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner ros2 launch sim_car full_sim_launch.launch.py track:=smalltrack planner:=midpoint controller:=pure_pursuit
```

## Recommendation

For a successor, give both:

- The finished image archive from Option 2, or the combined `master-planner-docker-handoff.tar`
- The runtime overlay tarballs from Option 1

The image lets them run immediately. The runtime overlays let them rebuild later without needing to rebuild the custom OpenCV/CUDA setup from source.

If you only send the GitHub link, they will not be able to rebuild the Docker image because the required runtime overlay tarballs are not in git.
