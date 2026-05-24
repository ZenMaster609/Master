# Cone Memory And Evaluation

`sim_car` uses a dedicated cone subsystem to fuse LiDAR and camera cone detections into stable, plannable tracks and to evaluate detection quality against ground truth. The two main runtime nodes are `cone_memory_node` and `cone_evaluator_node`.

## Runtime Flow

Typical cone memory flow:

`lidar cones + camera cones -> cone_memory_node -> /tracked_cones -> planner`

The evaluator runs alongside memory to score detection quality:

`predicted cones + /ground_truth/cones -> cone_evaluator_node -> range-error samples -> logger`

## Cone Memory

`cone_memory_node` maintains a short-term local track list by fusing incoming LiDAR and camera cone detections. It publishes `/tracked_cones` as a `ConeDetectionArray` that downstream planners consume.

### Track States

Each cone track progresses through three states:

- **Tentative**: newly created, not yet confirmed by enough hits.
- **Confirmed**: seen enough times to be trusted for planning.
- **Stale**: not recently updated; retained briefly before deletion.

Confirmed tracks publish with their resolved color. Tentative tracks are not forwarded to planners. Stale confirmed tracks are forwarded by default while they are within `memory.stale_planner_ttl_sec`, controlled by `memory.publish_stale_tracks`.

### Sensor Fusion

LiDAR and camera detections are fused per track. Position source selection depends on range:

- In the near band, LiDAR position is preferred. Camera position may be used as a fallback if LiDAR is missing and `allow_camera_fallback_near` is set.
- In the far band, camera position can override LiDAR. LiDAR position is used as a fallback if camera is missing and `prefer_lidar_if_camera_missing_far` is set.

Color class probabilities are updated in a Bayesian style. Each LiDAR or camera observation shifts the probability mass toward the detected color. The track's published color is the highest-probability class once confirmed.

### Color Resolution

Orange cones and cones with unknown color are resolved to `blue` or `yellow` before planning using a neighbor-hint and lateral-position strategy:

- If an orange cone has enough blue or yellow neighbors nearby, the majority neighbor color is inferred.
- If no neighbors are available, the cone's lateral position relative to the vehicle is used to pick the closer boundary color.
- This boundary color resolution produces the `/tracked_cones` output used by standard track planners.

### Permanent Cone Memory

An optional permanent cone store can accumulate confirmed tracks across time. This allows the system to remember cones that briefly disappear behind occlusions. Permanent memory is enabled separately from the local tracker and is controlled by `permanent_memory.enabled`.

### Visualization

`cone_memory_node` publishes `MarkerArray` topics for RViz inspection:

- **Cone markers** (`/local_cone_map_viz`): cylinders at each tracked cone position. Confirmed cones are larger, tentative and stale cones are smaller or dimmed.
- **ID markers**: floating text labels with track IDs above each cone.
- **Track polylines** (`/cone_memory/believed_track_viz`): line strips tracing the track path on each boundary.

Raw sensor inputs can also be visualized under `/local_cone_map_viz/raw_lidar` and `/local_cone_map_viz/raw_camera`.

### CSV Export

On shutdown, `cone_memory_node` saves track data to CSV. The output directory is resolved from the active `RunSession` path or falls back to the default multidata directory. The CSV contains track positions, labels, confidence values, and centerline estimates.

## Cone Evaluation

`cone_evaluator_node` runs alongside the detection pipeline and compares predicted cone positions against `/ground_truth/cones`. It supports evaluation of multiple detection sources (camera and LiDAR) simultaneously.

### Matching

The evaluator performs a greedy nearest-neighbor match between predicted and ground-truth cones within a configurable distance threshold. Cones outside the threshold are counted as misses. Matched pairs produce an error sample.

Both predicted and ground-truth cones are transformed into a common reference frame using TF before matching.

### Output

Matched pairs are published as signed range-error samples:

- source name (e.g., `monocular`, `stereo`, `lidar`)
- ground-truth range from the vehicle
- `error_m`, the predicted range minus the ground-truth range
- cone class IDs

These samples feed the logging layer, which writes per-source CSVs such as:

- `cone_range_rmse_samples_mono.csv`
- `cone_range_rmse_samples_lidar.csv`

The evaluator publishes to `{eval_prefix}/cone_depth_samples` as CSV-formatted strings for compatibility with the vehicle plotter runtime logger.

## Key Parameters

Important `cone_memory_node` parameters:

- `topics.lidar_cones_topic`, `topics.camera_cones_topic`: input detection topics.
- `topics.tracked_cones_topic`: output topic for planners.
- `memory.confirm_hits`: number of detections required to confirm a track.
- `memory.stale_after_sec`: time without updates before a confirmed track becomes stale.
- `memory.publish_stale_tracks`: whether fresh-enough stale tracks are still planner-forwarded.
- `memory.stale_planner_ttl_sec`: maximum stale-track age for planner forwarding.
- `fusion.camera_range_m`: range boundary between near and far fusion bands.
- `fusion.prefer_lidar_if_camera_missing_far`: use LiDAR position in far band when camera is absent.
- `fusion.allow_camera_fallback_near`: allow camera position in near band when LiDAR is absent.
- `permanent_memory.enabled`: enable the permanent cone store.

## Useful Commands

Build the relevant packages:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car vehicle_plotter vehicle_plotter_msgs
```

Source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Launch with cone memory enabled:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py cone_memory_enabled:=true
```

Inspect tracked cones:

```bash
cd ~/ros2_ws && ros2 topic echo /tracked_cones
```

Inspect visualization in RViz by subscribing to `/local_cone_map_viz` and `/cone_memory/believed_track_viz`.
