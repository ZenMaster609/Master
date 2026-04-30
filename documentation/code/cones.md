# Cones Code Map

This page maps the `documentation/cones.md` behavior to the cone memory node, evaluator node, tracker, fusion helpers, visualization builders, and CSV export logic.

## Primary Files

- `sim_car/sim_car/cones/nodes/memory_node.py`
- `sim_car/sim_car/cones/nodes/evaluator_node.py`
- `sim_car/sim_car/cones/tracking/tracker.py`
- `sim_car/sim_car/cones/tracking/fusion.py`
- `sim_car/sim_car/cones/nodes/_memory_viz.py`
- `sim_car/sim_car/cones/nodes/_memory_csv.py`

## Function Map

### Cone Memory Runtime

- `ConeMemoryNode` in `sim_car/sim_car/cones/nodes/memory_node.py`: top-level node owning all subscriptions, the local tracker, the permanent store, and the publication loop.
- `ConeMemoryNode._declare_parameters` and `ConeMemoryNode._read_parameters` in `sim_car/sim_car/cones/nodes/memory_node.py`: declare and load the full node parameter set at startup.
- `ConeMemoryNode._on_lidar_cones` and `ConeMemoryNode._on_camera_cones` in `sim_car/sim_car/cones/nodes/memory_node.py`: receive incoming LiDAR and camera detections and queue them for the tracker update.
- `ConeMemoryNode._update_tracker` in `sim_car/sim_car/cones/nodes/memory_node.py`: drives a single tracker update cycle fusing the latest LiDAR and camera batches.
- `ConeMemoryNode._publish_tracked_cones` in `sim_car/sim_car/cones/nodes/memory_node.py`: serializes confirmed tracks into a `ConeDetectionArray` and publishes to `/tracked_cones`.
- `ConeMemoryNode._publish_visualization` in `sim_car/sim_car/cones/nodes/memory_node.py`: calls the `_memory_viz` builders and publishes all three `MarkerArray` topics.

### Track Management

- `LocalConeTracker` in `sim_car/sim_car/cones/tracking/tracker.py`: manages the full list of `ConeTrack` objects, handles association, creation, update, and pruning.
- `ConeTrack` in `sim_car/sim_car/cones/tracking/tracker.py`: mutable dataclass for one tracked cone; holds position, class probabilities, state (`TENTATIVE`, `CONFIRMED`, `STALE`), confidence, and timing.
- `TrackUpdate` in `sim_car/sim_car/cones/tracking/tracker.py`: fused observation container passed to the tracker from the memory node each update cycle.
- `SensorDetection` in `sim_car/sim_car/cones/nodes/memory_types.py`: lightweight dataclass holding a single incoming detection's position in odom and base frames, range, color, and confidence.

### Sensor Fusion

- `normalize_color` in `sim_car/sim_car/cones/tracking/fusion.py`: maps raw detection color strings (including `big_orange`) to the standard internal color set.
- `update_class_probs` in `sim_car/sim_car/cones/tracking/fusion.py`: Bayesian-style update of per-track color class probabilities from a new observation.
- `blend_track_confidence` in `sim_car/sim_car/cones/tracking/fusion.py`: bounded running confidence score update used to judge track validity.
- `choose_position_source` in `sim_car/sim_car/cones/tracking/fusion.py`: near/far band policy deciding whether a track update should use the LiDAR or camera position.
- `clamp_camera_range` in `sim_car/sim_car/cones/tracking/fusion.py`: enforces the configured maximum camera range before the position is used.
- `resolve_boundary_colors_for_planning` in `sim_car/sim_car/cones/tracking/fusion.py`: infers `blue` or `yellow` for orange and unknown cones using lateral position and neighbor hints.

### Global / Permanent Memory

- `PermanentConeStore` in `sim_car/sim_car/cones/tracking/global_memory.py`: long-lived accumulator that merges confirmed local tracks into a global store with running-average position filtering.
- `PermaCone` in `sim_car/sim_car/cones/tracking/global_memory.py`: dataclass for a single global-store entry with position, label, confidence, and hit count.

### Pose Helpers

- `base_point_to_odom`, `odom_point_to_base`, and `convert_odom_child_pose_to_base_frame` in `sim_car/sim_car/cones/tracking/pose.py`: frame-conversion utilities used by the memory node during detection ingestion and tracker updates.

### Visualization

- `make_cone_marker` in `sim_car/sim_car/cones/nodes/_memory_viz.py`: builds a cylinder `Marker` for one cone; size and alpha vary by track state.
- `make_id_marker` in `sim_car/sim_car/cones/nodes/_memory_viz.py`: builds a text `Marker` showing the track ID above a cone.
- `make_line_marker` in `sim_car/sim_car/cones/nodes/_memory_viz.py`: builds a `LINE_STRIP` marker for the believed track path on one boundary.
- `make_perma_cone_marker` in `sim_car/sim_car/cones/nodes/_memory_viz.py`: builds markers for cones held in the permanent store.
- `make_raw_sensor_markers` in `sim_car/sim_car/cones/nodes/_memory_viz.py`: builds markers for raw incoming detections in base and odom frames.
- `split_polyline_by_gap` in `sim_car/sim_car/cones/nodes/_memory_viz.py`: splits a track polyline at gaps larger than a threshold to avoid long visual artifacts.

### CSV Export

- `save_track_data_csv` in `sim_car/sim_car/cones/nodes/_memory_csv.py`: writes cone track positions, labels, confidence, and centerline data to CSV on node shutdown.
- `resolve_logs_dir` in `sim_car/sim_car/cones/nodes/_memory_csv.py`: resolves the output directory from the active `RunSession` or falls back to the default multidata path.

### Cone Evaluation

- `ConeEvaluatorNode` in `sim_car/sim_car/cones/nodes/evaluator_node.py`: subscribes to predicted and ground-truth cone topics, performs greedy nearest-neighbor matching, and publishes range-error samples.
- `ConeEvaluatorNode._match_cones` in `sim_car/sim_car/cones/nodes/evaluator_node.py`: implements the distance-threshold greedy matching between predicted and ground-truth cone sets.
- `ConeEvaluatorNode._on_prediction` in `sim_car/sim_car/cones/nodes/evaluator_node.py`: handles incoming predicted cone arrays and queues them against the latest ground truth.

### Evaluation Output

- `ConePlotting2Runtime` in `sim_car/sim_car/cones/plotting/runtime.py`: publishes cone RMSE samples as CSV-formatted strings to `{eval_prefix}/cone_depth_samples`.
- `format_sample_rows` in `sim_car/sim_car/cones/plotting/runtime.py`: formats a batch of range-error samples into the CSV row layout expected by the logger.

## Related Entry Points

- `generate_launch_description` in `sim_car/launch/full_sim_launch.launch.py`: controls `cone_memory_enabled`, evaluation prefix, and input topic wiring for the cone subsystem.
