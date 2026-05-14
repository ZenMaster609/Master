# Stereo Calibration

This page explains how to reproduce `sim_car/config/stereo_calibration.yaml`.
The current calibration was not produced with checkerboard images. It is a
synthetic calibration derived from the camera model and mounting geometry that
the simulator already defines.

## When This Method Applies

Use this method when the stereo cameras are simulated or otherwise have known
intrinsics and a known rigid mounting transform. A checkerboard calibration is
still the right tool when the real lenses, lens distortion, sensor alignment, or
mounting tolerances are unknown.

For the current simulator setup, the relevant values are:

| Quantity | Value | Source |
| --- | ---: | --- |
| Image width | `1280 px` | `sim_car/urdf/eufs_car.urdf.xacro` camera image block |
| Image height | `720 px` | `sim_car/urdf/eufs_car.urdf.xacro` camera image block |
| Horizontal field of view | `1.919862 rad` | `sim_car/urdf/eufs_car.urdf.xacro` camera block |
| Left camera mount | `xyz="1.71 0.06 0.2"` | `stereo_left_joint` |
| Right camera mount | `xyz="1.71 -0.06 0.2"` | `stereo_right_joint` |
| Baseline | `0.12 m` | Difference between left and right lateral offsets |
| Distortion | `0.0` | Ideal Gazebo camera lens |
| Relative rotation | Identity | Both camera joints use `rpy="0 0 0"` |

## Step 1: Read The Camera Model

Open `sim_car/urdf/eufs_car.urdf.xacro` and find the stereo camera sensors:

```xml
<horizontal_fov>1.919862</horizontal_fov>
<image>
  <width>1280</width>
  <height>720</height>
  <format>R8G8B8</format>
</image>
```

Both left and right cameras must use the same image size and field of view for
the simple rectified stereo model used here.

## Step 2: Compute The Focal Length

Gazebo gives the horizontal field of view. With a pinhole camera and square
pixels, the focal length in pixels is:

```text
fx = width_px / (2 * tan(horizontal_fov_rad / 2))
```

For the current values:

```text
fx = 1280 / (2 * tan(1.919862 / 2))
fx = 448.141402 px
```

The simulator camera uses square pixels, so:

```text
fy = fx = 448.141402 px
```

## Step 3: Choose The Principal Point

The image center is used as the principal point:

```text
cx = (width_px - 1) / 2 = 639.5 px
cy = (height_px - 1) / 2 = 359.5 px
```

That gives this intrinsic matrix for both cameras:

```text
K = [448.141402,   0.0,      639.5,
       0.0,      448.141402, 359.5,
       0.0,        0.0,        1.0]
```

## Step 4: Derive The Stereo Baseline

The camera joints are mounted at the same `x` and `z` coordinates, with opposite
`y` offsets:

```xml
<origin xyz="1.71 0.06 0.2" rpy="0 0 0"/>
<origin xyz="1.71 -0.06 0.2" rpy="0 0 0"/>
```

The baseline is therefore:

```text
baseline_m = abs(0.06 - (-0.06)) = 0.12 m
```

In the OpenCV rectified stereo convention used by `stereo_pipeline.py`, the
right camera is translated by `-baseline` from the left camera along the stereo
image x-axis:

```text
T_left_to_right = [-0.12, 0.0, 0.0]^T
R_left_to_right = identity
```

The sign matters because the right projection matrix stores:

```text
P_right[0, 3] = -fx * baseline_m
```

For the current setup:

```text
P_right[0, 3] = -448.141402 * 0.12 = -53.776968
```

## Step 5: Fill The Calibration YAML

The left camera projection matrix has no x offset:

```text
P_left = [fx, 0,  cx, 0,
          0,  fy, cy, 0,
          0,  0,  1,  0]
```

The right camera projection matrix uses the baseline term:

```text
P_right = [fx, 0,  cx, -fx * baseline_m,
           0,  fy, cy, 0,
           0,  0,  1,  0]
```

Because this is an ideal simulator lens, the distortion model is `plumb_bob`
with all coefficients set to zero:

```text
D = [0, 0, 0, 0, 0]
```

Because both cameras are already aligned, both rectification matrices are
identity matrices.

## Step 6: Fill The Q Matrix

The OpenCV reprojection matrix uses the same focal length, principal point, and
baseline:

```text
Q = [1, 0, 0, -cx,
     0, 1, 0, -cy,
     0, 0, 0,  fx,
     0, 0, 1 / baseline_m, 0]
```

For the current setup:

```text
1 / baseline_m = 1 / 0.12 = 8.333333333
```

This produces the `q_matrix` currently stored in
`sim_car/config/stereo_calibration.yaml`.

## Step 7: Make The File Available To The Perception Node

The full simulation launch file passes the calibration file into
`perception_node`:

```text
calibration_file = sim_car/config/stereo_calibration.yaml
baseline_m = 0.12
```

After changing `sim_car/config/stereo_calibration.yaml`, rebuild or refresh the
installed package share so the launch file can load the updated YAML:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car
```

Then source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```

## Step 8: Validate The Calibration

Launch stereo perception:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py stereo:=true camera_debug:=true
```

Check the startup log from `perception_node`. A successful load prints the
calibration path, focal length, and baseline. For the current calibration it
should report approximately:

```text
fx=448.141px baseline=0.1200m
```

You can also inspect the published camera info:

```bash
cd ~/ros2_ws && ros2 topic echo /sim/stereo/left/camera_info --once
```

```bash
cd ~/ros2_ws && ros2 topic echo /sim/stereo/right/camera_info --once
```

The `k` matrix should match the intrinsic matrix above. The right `p` matrix
should contain the negative baseline term near `-53.776968` in element `p[3]`.

Finally, check the stereo output qualitatively by enabling the debug image and
watching that cone detections stay stable as range changes. For numeric
validation, use the cone evaluator output from the stereo run and compare
`cone_range_rmse_samples_stereo.csv` against ground truth in the generated run
session.

## If The Camera Model Changes

Repeat the same derivation whenever any of these values change:

- image width or height
- horizontal field of view
- left or right camera mount position
- relative camera rotation
- lens distortion model

At minimum, recompute `fx`, `fy`, `cx`, `cy`, the baseline, `P_right[0, 3]`, and
the `Q` matrix. If the cameras are no longer perfectly parallel, replace the
identity relative rotation with the actual left-to-right rotation and let
OpenCV stereo rectification build the final rectification maps at runtime.
