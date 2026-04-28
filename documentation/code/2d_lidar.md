# 2D LiDAR Code Map

This page maps the `documentation/2d_lidar.md` behavior to the scan-processing symbols that implement it.

## Primary Files

- `sim_car/sim_car/lidar/lidar_node.py`
- `sim_car/sim_car/lidar/clustering.py`
- `sim_car/launch/full_sim_launch.launch.py`

## Function Map

### Runtime Flow

- `LidarNode._scan_cb` in `sim_car/sim_car/lidar/lidar_node.py`: ROS callback that runs the full scan-to-detections pipeline for each `LaserScan`.
- `LidarNode._extract_detections` in `sim_car/sim_car/lidar/lidar_node.py`: connects range conversion, clustering, cone gating, and deduplication before frame transforms.
- `LidarNode._publish_cone_detections` in `sim_car/sim_car/lidar/lidar_node.py`: publishes planner-facing `ConeDetectionArray` output.

### Range Filtering

- `points_from_ranges` in `sim_car/sim_car/lidar/clustering.py`: converts ordered scan ranges into XY points while enforcing finite values, scan min/max, and configured detection-range limits.
- `LidarNode._read_parameters` in `sim_car/sim_car/lidar/lidar_node.py`: loads `min_detection_range_m` and `max_detection_range_m` used by `points_from_ranges`.

### Scan-Ordered Clustering

- `cluster_points` in `sim_car/sim_car/lidar/clustering.py`: walks scan-ordered points and starts a new cluster when adjacent-point distance exceeds `cluster_jump_threshold_m`.
- `LidarNode._extract_detections` in `sim_car/sim_car/lidar/lidar_node.py`: passes the configured jump threshold into `cluster_points`.

### Cone-Like Cluster Gates

- `detect_cone_candidates` in `sim_car/sim_car/lidar/clustering.py`: summarizes each cluster by count, width, depth, and centroid, then applies the cone acceptance gates.
- `ClusterDetection` in `sim_car/sim_car/lidar/clustering.py`: carries accepted cluster summary data back to the node layer.

### Deduplication

- `LidarNode._deduplicate_xy` in `sim_car/sim_car/lidar/lidar_node.py`: merges nearby accepted centroids within `dedup_radius_m` so one cone does not become several planner detections.

### Frame Transform

- `LidarNode._transform_detections` in `sim_car/sim_car/lidar/lidar_node.py`: transforms accepted detections from the scan frame into the configured cone-detection frame.
- `LidarNode._lookup_transform` in `sim_car/sim_car/lidar/lidar_node.py`: resolves TF lookups and tries source-frame aliases when Gazebo frame names vary.
- `LidarNode._source_frame_candidates` in `sim_car/sim_car/lidar/lidar_node.py`: generates the alias candidates used during TF lookup.
- `LidarNode._transform_point` in `sim_car/sim_car/lidar/lidar_node.py`: applies one transform to one XY detection point.

### Output

- `LidarNode._publish_cone_detections` in `sim_car/sim_car/lidar/lidar_node.py`: stamps output detections as unknown-color planner cones with fixed confidence.
- `LidarNode._warn_throttled` in `sim_car/sim_car/lidar/lidar_node.py`: reports dropped-frame conditions such as missing transforms without spamming logs.

## Related Entry Points

- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: selects this pipeline when `lidar_pipeline:=scan2d`.
- `_lidar_pipeline_match_expr` in `sim_car/launch/full_sim_launch.launch.py`: launch-side switch used to enable the 2D node versus the 3D node.
