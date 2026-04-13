# 3D LiDAR

The 3D LiDAR pipeline consumes `sensor_msgs/PointCloud2`, filters the point cloud into cone-like clusters, and publishes planner-facing `ConeDetectionArray` messages with unknown cone color.

It is selected by default with `lidar_pipeline:=pointcloud3d`.

## Runtime Flow

`PointCloud2 -> pointcloud_lidar_node -> filtered point cloud + cluster markers + cone detections`

Main outputs:

- filtered point cloud
- accepted/rejected cluster debug markers
- `/sim/lidar/perception/cones_3d` or `/sim/raw/lidar/perception/cones_3d`

The output detections are normally fused by cone memory with camera detections.

## PointCloud Decoding

The node decodes `x`, `y`, and `z` fields from the incoming `PointCloud2`. Points with non-finite XYZ values are discarded.

The decoder handles row layout and point-field padding directly, so it does not depend on a separate point-cloud helper package.

## Pre-Filtering

The raw cloud is processed through several filters:

1. Downsample with `downsample_stride`.
2. Remove azimuth sectors listed in `azimuth_mask_ranges_deg`.
3. Randomly thin far points with `thinning_start_range_m`, `max_detection_range_m`, and `thinning_keep_ratio_at_max_range`.
4. Transform points into the cone-detection frame.
5. Crop to rectangular ROI.
6. Suppress ground points.

The full sim launch sets the detection frame to `base_footprint` in the node helper, and the pipeline then publishes detections in the configured output frame.

## ROI Crop

The ROI crop keeps points inside:

- `x_min_m` to `x_max_m`
- `y_min_m` to `y_max_m`
- `z_min_m` to `z_max_m`

This removes points behind the car, far outside the track corridor, or above the cone-height region.

## Adaptive Ground Suppression

Ground suppression removes points below an adaptive floor:

`max(ground_base_cutoff_m, ground_range_bias_m + ground_range_slope_m_per_m * max(0, range - 3))`

The floor rises slightly with range. This helps reject flat ground returns while keeping cone body points.

## XY Clustering

The remaining points are clustered in XY. The clustering radius depends on range:

- close range: tighter radius
- mid range: medium radius
- far range: larger radius

This is important because far cones have fewer and less tightly grouped points.

## Range-Adaptive Acceptance

Each cluster is summarized by:

- centroid
- point count
- width
- depth
- height
- min/max range

Acceptance thresholds are adjusted by range. Farther clusters can pass with fewer points and looser size gates. Rejected clusters keep a reason such as:

- `too_few_points`
- `too_many_points`
- `too_narrow`
- `too_wide`
- `too_shallow`
- `too_deep`
- `too_short`
- `too_tall`

The node publishes debug markers for both accepted and rejected clusters, which makes threshold tuning much easier in RViz.

## Output

Accepted clusters become cone detections:

- color: `unknown`
- confidence: `lidar_confidence`
- position: cluster centroid XY, with output Z set to zero

The filtered point cloud is also published for debugging.

## Launch-Specific Effects

When `lidar_pipeline:=pointcloud3d`, the full sim launch lowers cone-memory confirmation and planner confidence thresholds:

- cone memory confirmation: one hit
- planner `filtering.min_confidence`: `0.15`

This reduces planning delay for 3D LiDAR because the point-cloud filter already rejects many non-cone points before publishing detections.

## Strengths And Weaknesses

Strengths:

- uses height and 3D shape, not only scan width
- better debug visibility through rejected cluster reasons
- stronger long-range behavior than the 2D scan pipeline

Weaknesses:

- more parameters
- sensitive to ground suppression and ROI tuning
- still does not classify cone color

## Useful Commands

Run with 3D LiDAR:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py lidar_pipeline:=pointcloud3d
```
