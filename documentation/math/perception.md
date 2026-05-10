# Perception Math

## Scope

This page documents the camera perception math used after YOLO detects cone bounding boxes. It covers monocular depth, stereo depth, 2D-to-3D reconstruction, TF transforms, and duplicate cone merging.

The neural-network detector itself is not described here. This page focuses on the geometric and numeric processing around detector output.

## Pipeline Map

1. `sim_car/sim_car/perception/perception_node.py` receives images and camera info, then runs a YOLO backend through `perception/yolo_runtime.py`.
2. In monocular mode, `sim_car/sim_car/perception/detection_depth.py::apply_monocular_depth_to_detections` estimates depth from bounding-box height.
3. In stereo mode, `sim_car/sim_car/perception/stereo_pipeline.py::StereoPipeline.process` rectifies images, computes disparity, and converts disparity to depth. `apply_depth_map_to_detections` samples that depth for each detection.
4. `sim_car/sim_car/perception/cone_geometry.py::reconstruct_cam_point_from_axis` converts image coordinates and depth to a 3D camera-frame point.
5. `sim_car/sim_car/perception/cone_geometry.py::transform_point` applies the ROS TF rotation/translation to the requested output frame.
6. `deduplicate_cone_candidates` merges overlapping reconstructed cones before `PerceptionNode._publish_cone_detections` publishes `ConeDetectionArray`.

## Mathematical Building Blocks

### Camera Intrinsics

`sim_car/sim_car/perception/cone_geometry.py::camera_intrinsics` extracts `fx`, `fy`, `cx`, and `cy` from `CameraInfo.k`. These values are used both for monocular pinhole depth and for reconstructing 3D rays from pixel coordinates.

### Monocular Bounding-Box Depth

`sim_car/sim_car/perception/monocular_depth.py::estimate_axis_depth_from_bbox_height` uses:

```text
axis_depth = fy_px * cone_height_m / (bbox_height_px - bbox_height_offset_px)
```

The code chooses `cone_height_m` or `big_cone_height_m` based on the normalized detector label. This estimates depth along the camera projection axis, which is enough for the downstream cone-memory pipeline to place the detection in 3D before TF conversion.

### Stereo Rectification And Disparity

`sim_car/sim_car/perception/stereo_pipeline.py::StereoPipeline._build_maps` loads calibration matrices, calls OpenCV stereo rectification, and builds remap tables. Rectification makes corresponding epipolar points lie on common image rows, so the disparity matcher can search horizontally.

`StereoPipeline._create_cpu_matcher` configures OpenCV StereoSGBM. The CPU path returns fixed-point disparity, and `_disparity_to_float` divides integer disparity by `16.0`. The CUDA path uses StereoBM when available.

`StereoPipeline._compute_depth` converts valid disparity to depth:

```text
depth = fx * baseline / disparity
```

Depth outside `[min_depth_m, max_depth_m]` is set to NaN, so later sampling ignores invalid or unrealistic geometry.

### Depth Sampling Inside A Detection

`sim_car/sim_car/perception/detection_depth.py::sample_depth_from_bbox` samples the lower central part of the bounding box and uses the median finite depth. This region is biased toward the cone body instead of the background above or beside it.

If the cropped patch has no valid samples, it falls back to `sample_depth`, which takes a small median window around the detection center.

### 2D-To-3D Reconstruction

`sim_car/sim_car/perception/cone_geometry.py::reconstruct_cam_point_from_axis` supports two camera frame conventions:

- `optical_z`: depth is camera `z`; `x = (u - cx) / fx * z`, `y = (v - cy) / fy * z`.
- `forward_x`: depth is camera `x`; lateral and vertical axes are sign-adjusted to match forward-axis frames.

This lets the perception node publish consistent cone points even when the source frame is an optical frame or a forward-facing link frame.

### Quaternion Transform Application

`sim_car/sim_car/perception/cone_geometry.py::transform_point` expands a unit quaternion into a 3x3 rotation matrix and applies:

```text
p_out = R * p_in + t
```

The same explicit matrix pattern appears in several ROS-facing modules because it avoids pulling in a heavier transform dependency for single-point conversions.

### Confidence-Weighted Cone Deduplication

`sim_car/sim_car/perception/cone_geometry.py::deduplicate_cone_candidates` sorts candidates by confidence and range, finds existing compatible candidates inside `dedup_radius_m`, and merges positions with confidence-derived weights. Known colors are not merged across conflicting known labels.

This reduces duplicate camera detections before they enter cone memory.

## Function Reference

| Math operation | Function | Runtime use |
| --- | --- | --- |
| Intrinsic extraction | `sim_car/sim_car/perception/cone_geometry.py::camera_intrinsics` | Supplies pinhole parameters for depth and reconstruction. |
| Monocular depth | `sim_car/sim_car/perception/monocular_depth.py::estimate_axis_depth_from_bbox_height` | Estimates axis depth from cone height in pixels. |
| Monocular depth attachment | `sim_car/sim_car/perception/detection_depth.py::apply_monocular_depth_to_detections` | Adds `depth_m`, `u_center`, and `v_center` to YOLO detections. |
| Stereo frame processing | `sim_car/sim_car/perception/stereo_pipeline.py::StereoPipeline.process` | Runs decode, rectify, disparity, and depth conversion. |
| Depth conversion | `sim_car/sim_car/perception/stereo_pipeline.py::StereoPipeline._compute_depth` | Converts disparity to metric depth and filters invalid values. |
| Baseline/focal resolution | `sim_car/sim_car/perception/stereo_pipeline.py::StereoPipeline._resolve_fx_baseline` | Chooses calibration, config, or `CameraInfo` sources for stereo geometry. |
| Rectification maps | `sim_car/sim_car/perception/stereo_pipeline.py::StereoPipeline._build_maps` | Builds calibrated remap matrices for stereo matching. |
| Stereo depth sampling | `sim_car/sim_car/perception/detection_depth.py::apply_depth_map_to_detections` | Samples per-detection depth from the stereo depth map. |
| BBox patch median | `sim_car/sim_car/perception/detection_depth.py::sample_depth_from_bbox` | Uses robust median depth in the cone body region. |
| 3D reconstruction | `sim_car/sim_car/perception/cone_geometry.py::reconstruct_cam_point_from_axis` | Converts pixel coordinate plus depth into a camera-frame point. |
| TF point transform | `sim_car/sim_car/perception/cone_geometry.py::transform_point` | Converts reconstructed cone points into the output frame. |
| Candidate merge | `sim_car/sim_car/perception/cone_geometry.py::deduplicate_cone_candidates` | Reduces duplicate camera detections before publication. |

## Notes / Limits

- Monocular depth depends directly on assumed cone height and detector box height. Small bounding-box errors become large range errors at distance.
- Stereo depth is only as good as rectification, baseline/focal calibration, and valid disparity. Invalid pixels are propagated as NaN.
- Median patch sampling is robust to sparse invalid pixels but can still fail if the bounding box mostly covers background.
- The perception layer deduplicates camera candidates, but long-term fusion and temporal smoothing are handled by cone memory.
