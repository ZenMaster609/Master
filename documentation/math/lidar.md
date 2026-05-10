# LiDAR Math

## Scope

This page documents the math used by the 2D LiDAR cone extraction helpers in `sim_car/sim_car/lidar`. The subsystem converts `LaserScan` polar ranges into Cartesian points, groups scan-contiguous points into Euclidean clusters, and filters those clusters into cone-like detections.

## Pipeline Map

1. `sim_car/sim_car/lidar/lidar_node.py` receives `sensor_msgs/LaserScan`.
2. `sim_car/sim_car/lidar/clustering.py::points_from_ranges` converts finite ranges inside the configured detection interval into `(x, y, r)` points.
3. `sim_car/sim_car/lidar/clustering.py::cluster_points` splits scan-ordered points when the Euclidean jump between adjacent valid returns exceeds a threshold.
4. `sim_car/sim_car/lidar/clustering.py::detect_cone_candidates` filters clusters by point count, apparent width, depth, and centroid.
5. `lidar_node.py` publishes detections for downstream cone memory or direct debugging.

## Mathematical Building Blocks

### Polar-To-Cartesian Projection

`points_from_ranges` walks the scan angles from `angle_min_rad` in increments of `angle_increment_rad`. Each finite range `r` that lies inside the intersection of the sensor valid interval and the configured detection interval is converted with:

```text
x = r * cos(angle)
y = r * sin(angle)
```

The original range is kept beside `(x, y)` because later filters use both Euclidean point geometry and radial depth.

### Scan-Order Euclidean Jump Clustering

`cluster_points` assumes the input points remain in scan order. It computes the Euclidean distance between each valid point and the previous valid point:

```text
jump = hypot(x_i - x_{i-1}, y_i - y_{i-1})
```

If `jump > jump_threshold_m`, a new cluster starts. This turns contiguous arcs of range returns into candidate objects without requiring a global clustering algorithm.

### Cone Candidate Width And Depth

`detect_cone_candidates` applies these cluster-level tests:

- point count is inside `[min_cluster_points, max_cluster_points]`.
- endpoint width `hypot(x_last - x_first, y_last - y_first)` is inside the configured width interval.
- radial depth `max(range) - min(range)` is below `max_cluster_depth_m`.
- centroid is the arithmetic mean of all cluster `(x, y)` points.

The endpoint width checks the apparent lateral span of the return. The range depth rejects extended surfaces that have too much radial thickness to be a cone-like object in the 2D scan.

## Function Reference

| Math operation | Function | Runtime use |
| --- | --- | --- |
| Polar range projection | `sim_car/sim_car/lidar/clustering.py::points_from_ranges` | Converts `LaserScan` beams to finite planar points. |
| Adjacent-point clustering | `sim_car/sim_car/lidar/clustering.py::cluster_points` | Splits scan-ordered points into object-sized groups. |
| Cluster filtering | `sim_car/sim_car/lidar/clustering.py::detect_cone_candidates` | Produces cone-like `ClusterDetection` records. |
| LiDAR node output | `sim_car/sim_car/lidar/lidar_node.py::LidarNode` | Wires scan input, clustering parameters, and detection publication. |

## Notes / Limits

- Clustering is scan-contiguous. Two objects that overlap in scan order can merge if no range jump appears between them.
- Width is measured between the first and last point in scan order, not from a fitted shape.
- Depth is radial range spread, not physical object depth in a local object frame.
- The method is intentionally lightweight and deterministic; cone memory performs later fusion, tracking, and deduplication.
