# 2D LiDAR

The 2D LiDAR pipeline consumes `sensor_msgs/LaserScan`, extracts cone-like scan clusters, and publishes planner-facing `ConeDetectionArray` messages with unknown cone color.

It is selected with `lidar_pipeline:=scan2d`.

The 2D path is the legacy scan pipeline. The current full-launch default is `lidar_pipeline:=pointcloud3d`.

## Runtime Flow

`LaserScan -> lidar_node -> /sim/lidar/perception/cones_3d -> cone_evaluator_node / cone_memory_node`

When `measure:=true` or `sensor_pipeline:=true`, the topics use `/sim/raw/...` instead.

The 2D LiDAR node does not classify cone color. It only estimates cone positions.

## Range Filtering

The node converts scan ranges into XY points. It keeps only finite ranges inside:

- scan message `range_min` / `range_max`
- `min_detection_range_m`
- `max_detection_range_m`

Each kept point is converted from polar scan coordinates into the scan frame:

`x = range * cos(angle)`

`y = range * sin(angle)`

## Scan-Ordered Clustering

The scan points remain in angular order. The clustering step walks through the ordered points and starts a new cluster when the Euclidean jump from the previous scan point exceeds:

`cluster_jump_threshold_m`

This works because a small cone appears as a compact run of neighboring scan beams, while gaps between objects produce larger jumps.

## Cone-Like Cluster Gates

Each cluster is summarized by:

- point count
- width between first and last point
- range depth from min/max point range
- centroid

The cluster is accepted only if it passes:

- `min_cluster_points`
- `max_cluster_points`
- `min_cluster_width_m`
- `max_cluster_width_m`
- `max_cluster_depth_m`

These gates reject noise, walls, and large objects that do not look like a cone in a 2D scan.

## Deduplication

Accepted cluster centroids are deduplicated with `dedup_radius_m`. Points closer than this radius are merged with a weighted average. The closest detections are processed first, so near detections tend to anchor the merged position.

Deduplication prevents one physical cone from becoming several planner inputs when the scan segmentation splits it.

## Frame Transform

The node transforms detections from the scan frame into `cone_detections_frame`, configured as `front_axle` in the full sim launch.

If the transform is unavailable, the frame is dropped. This is intentional: publishing detections in the wrong frame is worse than skipping one LiDAR frame.

The node tries several source-frame aliases to handle namespaced Gazebo frames.

## Output

Each accepted detection is published as:

- color: `unknown`
- confidence: `0.5`
- position: transformed cone XY with zero Z

Cone memory can fuse these unknown-position detections with camera color/class detections. The planner can also infer unknown cones to blue/yellow by lateral side when configured.

## Strengths And Weaknesses

Strengths:

- simple and cheap
- deterministic cluster behavior
- useful close-range position source

Weaknesses:

- no height information
- no cone color
- sensitive to scan resolution and object shape
- weaker at separating complex nearby objects than the 3D point-cloud pipeline

## Useful Commands

Run with 2D LiDAR:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py lidar_pipeline:=scan2d
```
