# Docker Handoff

This project uses a Docker image built from `src/Master/docker/Dockerfile`. A Docker image is the packaged environment; a container is a running instance of that image.

There are two good ways to transfer the environment to the next person.

## Option 1: They Build The Image

Use this when the next person may edit or rebuild the stack.

Give them:

- The repository under `~/ros2_ws/src/Master`
- `src/Master/docker/runtime-overlays/opencv_local.tar.gz`
- `src/Master/docker/runtime-overlays/cudnn_py.tar.gz`
- `src/Master/docker/runtime-overlays/cuda_runtime.tar.gz`

If you need to regenerate those overlay files first, run:

```bash
cd ~/ros2_ws && ./src/Master/scripts/package-runtime-overlays.sh src/Master/docker/runtime-overlays
```

On their Linux machine, they build the image:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml build planner
```

Then they can run a shell inside the environment:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner bash
```

## Option 2: You Give Them The Finished Image

Use this when the next person mainly needs the stack to run.

After your image is built, export it:

```bash
cd ~/ros2_ws && docker save master-planner:humble | gzip > master-planner-humble.tar.gz
```

Give them:

- `master-planner-humble.tar.gz`
- The repository under `~/ros2_ws/src/Master`, so they have `compose.yaml`, docs, launch files, and development mode files

On their Linux machine, they load the image:

```bash
cd ~/ros2_ws && docker load < master-planner-humble.tar.gz
```

Then they can run a shell inside the environment:

```bash
cd ~/ros2_ws && docker compose -f src/Master/compose.yaml run --rm planner bash
```

## Recommendation

For a successor, give both:

- The finished image archive from Option 2
- The runtime overlay tarballs from Option 1

The image lets them run immediately. The runtime overlays let them rebuild later without needing to rebuild the custom OpenCV/CUDA setup from source.
