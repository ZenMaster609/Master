# Reproducible Environment Code Map

This page maps the `documentation/concepts/reproducible_environment.md` behavior to the launch files and setup scripts that define the current runtime environment.

## Primary Files

- `sim_car/launch/full_sim_launch.launch.py`
- `scripts/package-runtime-overlays.sh`
- `scripts/setup-partner-environment.sh`
- `docker/Dockerfile`
- `compose.yaml`
- `compose.dev.yaml`
- `requirements*.txt`

## Function Map

### Runtime Overlay Injection

- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: launch entry point that sets up the perception node environment and runtime process graph.
- `_load_control_config` in `sim_car/launch/full_sim_launch.launch.py`: reads optional local control/runtime config values used by the launch file.
- `_resolve_launch_selection` in `sim_car/launch/full_sim_launch.launch.py`: resolves the selected track/planner/controller bundle that the partner environment must match.

### Copied Runtime Context

- `package-runtime-overlays.sh` in `scripts/package-runtime-overlays.sh`: packages the local runtime overlays that are not fully represented in repo-tracked Python requirements.
- `setup-partner-environment.sh` in `scripts/setup-partner-environment.sh`: installs the packaged overlays and partner-machine environment pieces.
- `docker/Dockerfile`: builds the ROS 2 Humble image, restores the runtime overlays, creates the YOLO virtualenv, and builds the workspace.
- `compose.yaml`: defines the Linux/NVIDIA/X11 runtime container.
- `compose.dev.yaml`: bind-mounts the host source tree and keeps build/install/log output in container-local volumes for development.

### Reproducibility Boundaries

- There is no single runtime function that guarantees reproducibility for this topic.
- The effective implementation surface is the combination of:
  - `sim_car/launch/full_sim_launch.launch.py`
  - repo-tracked requirements files
  - overlay packaging/setup scripts
  - Docker/Compose runtime files
  - the selected YOLO weights and local runtime overlay directories

## Related Entry Points

- `requirements.txt`, `requirements.user.lock.txt`, and `requirements-yolo-pt.lock.txt`: repo-tracked dependency references for the documented environment.
- `resolve_yolo_model_path` in `sim_car/sim_car/perception/yolo_runtime.py`: runtime-side path normalization for the YOLO model selected by launch/config.
