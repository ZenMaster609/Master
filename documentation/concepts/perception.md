# Perception

`sim_car` uses one camera perception node for both monocular and stereo cone detection. The active camera mode changes the ranging method, not the downstream contract: both modes publish `vehicle_plotter_msgs/ConeDetectionArray` and feed the cone evaluator, cone memory, and planners through the same shape of data.

The current launch default is monocular camera perception. Stereo is opt-in with `stereo:=true`.

## Runtime Flow

Typical camera flow:

`camera images -> perception_node -> /sim/stereo/perception/cones_3d -> cone_evaluator_node -> cone_memory_node -> planner`

When `measure:=true` or `sensor_pipeline:=true`, the launch file switches perception inputs and outputs under `/sim/raw/...` so the measurement layer remains the boundary between idealized sim signals and measured `/sim/...` signals:

`/sim/raw/stereo/... -> perception_node -> /sim/raw/stereo/perception/cones_3d`

When measurement is not enabled, perception uses `/sim/...` directly:

`/sim/stereo/... -> perception_node -> /sim/stereo/perception/cones_3d`

The launch file handles this with a shared `topic_prefix`, so the perception node itself does not need separate raw/measured modes.

## YOLO Detection

YOLO provides the semantic cone detections. By default, `full_sim_launch.launch.py` enables YOLO and points at:

`sim_car/yolo/weights/best.pt`

The model path can be changed with `yolo_model_path:=...`. A `.pt` model uses the Ultralytics/Torch path injected through `yolo_ultralytics_pythonpath`; an `.onnx` model can use the OpenCV DNN path. The launch also injects the custom OpenCV and cuDNN library paths used by the reference environment.

YOLO outputs bounding boxes, confidence scores, and class labels. The perception node then attaches a depth estimate to each detection and reconstructs a 3D cone position.

## Monocular Mode

Monocular mode uses only the left camera image. It is selected by default:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py stereo:=false
```

Processing steps:

1. Decode the left image.
2. Run YOLO on the left image.
3. Estimate cone axis depth from bounding-box height with a pinhole model.
4. Reconstruct the cone point from the bbox center, camera intrinsics, and estimated depth.
5. Transform or publish the point in the configured output frame.

The depth equation is:

`Z = fy * H / (h - delta)`

Where:

- `fy` is the vertical focal length in pixels.
- `H` is the assumed real cone height.
- `h` is the detected bounding-box height in pixels.
- `delta` is `monocular_bbox_height_offset_px`.

The active cone height depends on the class:

- `monocular_cone_height_m` for standard cones.
- `monocular_big_cone_height_m` for big orange cones.

`monocular_bbox_height_offset_px` is a manual empirical correction. It is exposed as a launch argument and defaults to the value set in `full_sim_launch.launch.py`.

## Stereo Mode

Stereo mode is enabled with:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py stereo:=true
```

Processing steps:

1. Buffer left and right images.
2. Pair frames within `max_time_diff_sec`.
3. Rectify both images from `stereo_calibration.yaml`.
4. Compute disparity with CUDA StereoBM when available, otherwise CPU StereoSGBM.
5. Convert valid disparity to depth with focal length and baseline.
6. Run YOLO on the rectified left image.
7. Sample median depth from the lower-central part of each bbox.
8. Reconstruct and transform each cone point.

The lower-central bbox crop intentionally avoids much of the background around the cone and biases the range sample toward the cone body/base.

## Output Contract

Both camera modes publish one `ConeDetectionArray`. Each cone includes:

- normalized class label: `blue`, `yellow`, `orange`, `big_orange`, or `unknown`
- confidence
- 3D position
- output frame in the array header

In the full sim launch, camera detections are configured to publish in `front_axle`.

The node deduplicates nearby reconstructed cones using `cone_dedup_radius_m`, which prevents repeated YOLO boxes from becoming repeated planner inputs.

## Cone Evaluation

`cone_evaluator_node` compares predicted cones against `/ground_truth/cones`. The camera evaluator source name is:

- `monocular` when `stereo:=false`
- `stereo` when `stereo:=true`

The evaluator publishes range-error samples under the selected eval prefix. The logger later writes source-specific CSVs such as:

- `cone_range_rmse_samples_mono.csv`
- `cone_range_rmse_samples_stereo.csv`
- `cone_range_rmse_samples_lidar.csv`

## Cone Memory And Planner Use

`cone_memory_node` fuses camera and scan LiDAR cone detections into `/tracked_cones` when `cone_memory_enabled:=true`. The camera contributes class/color information and, depending on the configured range split, can also contribute position.

Important fusion parameters:

- `camera_range_m`: far-band distance where camera position can override LiDAR.
- `prefer_lidar_if_camera_missing_far`: use LiDAR position in the far band if camera position is missing.
- `allow_camera_fallback_near`: allow camera position in the near band when LiDAR position is missing.

If cone memory is disabled, normal track planners can use the camera cone topic directly. On skidpad and acceleration, the planner input still goes through the skidpad router when the selected planner is `midpoint`, `single_boundary`, or `corridor`.

Planner input routing in the full launch is:

- smalltrack with cone memory: `/tracked_cones`
- smalltrack without cone memory: camera cone detections
- skidpad/acceleration with tracked-cone planners: `/tracked_cones/skidpad_routed`
- `linetest`: no cone input

## Useful Commands

Build the relevant packages:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car vehicle_plotter vehicle_plotter_msgs
```

Source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Run the default monocular pipeline:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py stereo:=false
```

Run stereo:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py stereo:=true
```
