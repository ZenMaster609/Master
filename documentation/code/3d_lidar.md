# 3D LiDAR Code Map

This page maps the `documentation/3d_lidar.md` behavior to the point-cloud pipeline that filters, clusters, and publishes cone detections.

## Primary Files

- `sim_car/sim_car/lidar/pointcloud_lidar_node.py`
- `sim_car/sim_car/lidar/pointcloud_processing.py`
- `sim_car/launch/full_sim_launch.launch.py`

## Function Map

### Runtime Flow

- `PointCloudLidarNode._pointcloud_cb` in `sim_car/sim_car/lidar/pointcloud_lidar_node.py`: top-level callback that decodes, filters, transforms, clusters, and publishes each `PointCloud2`.
- `PointCloudLidarNode._publish_filtered_points` in `sim_car/sim_car/lidar/pointcloud_lidar_node.py`: publishes the debug filtered cloud.
- `PointCloudLidarNode._publish_debug_clusters` in `sim_car/sim_car/lidar/pointcloud_lidar_node.py`: publishes accepted and rejected cluster markers for debugging.
- `PointCloudLidarNode._publish_cone_detections` in `sim_car/sim_car/lidar/pointcloud_lidar_node.py`: publishes planner-facing cone detections.

### PointCloud Decoding

- `pointcloud2_to_xyz_array` in `sim_car/sim_car/lidar/pointcloud_processing.py`: decodes `x`, `y`, and `z` directly from `PointCloud2`, dropping non-finite points.
- `_pointcloud_dtype` in `sim_car/sim_car/lidar/pointcloud_processing.py`: handles point-field layout and padding so decoding works without an external helper package.
- `xyz_array_to_pointcloud2` in `sim_car/sim_car/lidar/pointcloud_processing.py`: rebuilds the filtered debug cloud for publication.

### Pre-Filtering

- `downsample_points` in `sim_car/sim_car/lidar/pointcloud_processing.py`: applies the `downsample_stride` thinning step.
- `apply_azimuth_masks` in `sim_car/sim_car/lidar/pointcloud_processing.py`: removes points inside configured blocked azimuth sectors.
- `apply_range_thinning` in `sim_car/sim_car/lidar/pointcloud_processing.py`: randomly thins far points using the configured start range and keep ratio.
- `PointCloudLidarNode._transform_points` in `sim_car/sim_car/lidar/pointcloud_lidar_node.py`: moves points into the detection frame before ROI and ground filtering.

### ROI Crop

- `crop_points_to_roi` in `sim_car/sim_car/lidar/pointcloud_processing.py`: keeps only points inside the configured XYZ box around the vehicle.

### Adaptive Ground Suppression

- `suppress_ground_points` in `sim_car/sim_car/lidar/pointcloud_processing.py`: removes points under the adaptive ground floor.

### XY Clustering

- `cluster_xy_points_adaptive` in `sim_car/sim_car/lidar/pointcloud_processing.py`: clusters the remaining cloud in XY using range-adaptive neighborhood radii.
- `_cluster_radius_for_range` in `sim_car/sim_car/lidar/pointcloud_processing.py`: expands the cluster radius with range so distant cones still group correctly.

### Range-Adaptive Acceptance

- `detect_cone_like_clusters` in `sim_car/sim_car/lidar/pointcloud_processing.py`: summarizes each cluster and applies range-adjusted cone acceptance thresholds.
- `_acceptance_thresholds_for_range` in `sim_car/sim_car/lidar/pointcloud_processing.py`: computes the active width, depth, height, and point-count thresholds for a given range.
- `_cluster_acceptance_reason` in `sim_car/sim_car/lidar/pointcloud_processing.py`: assigns reject reasons such as `too_few_points` and `too_wide`.
- `PointClusterDetection` in `sim_car/sim_car/lidar/pointcloud_processing.py`: carries accepted or rejected cluster summaries through the pipeline.

### Debug Summaries

- `summarize_clusters_for_debug` in `sim_car/sim_car/lidar/pointcloud_processing.py`: builds the node-facing debug summary data used for markers and stats.
- `summarize_rejection_reasons` in `sim_car/sim_car/lidar/pointcloud_processing.py`: aggregates rejection categories for logging.
- `PointCloudLidarNode._log_detection_stats` in `sim_car/sim_car/lidar/pointcloud_lidar_node.py`: emits throttled per-frame detection statistics.

## Related Entry Points

- `_pointcloud3d_lidar_parameters` in `sim_car/launch/full_sim_launch.launch.py`: launch helper that injects the main 3D LiDAR parameter set.
- `_pointcloud3d_debug_topics` in `sim_car/launch/full_sim_launch.launch.py`: defines the debug point-cloud and marker topics.
- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: enables this pipeline when `lidar_pipeline:=pointcloud3d`.
