# Cones Math

## Scope

This page documents the mathematical operations used by the cone memory, tracking, fusion, permanent-memory, and evaluation code. It focuses on what each operation does for the runtime system and where it is implemented.

The cone subsystem receives camera and LiDAR cone detections, expresses them in common frames, pairs detections that describe the same physical cone, updates local tracks, infers planner-facing boundary colors, and publishes `/tracked_cones`.

## Pipeline Map

1. `sim_car/sim_car/cones/nodes/memory_node.py::ConeMemoryNode._lidar_cb` and `ConeMemoryNode._camera_cb` convert incoming detections into `SensorDetection` records.
2. `sim_car/sim_car/cones/tracking/pose.py` handles planar odom/base transforms and optional odometry-lag compensation.
3. `sim_car/sim_car/cones/nodes/memory_node.py::_deduplicate_sensor_detections` merges duplicate detections inside each sensor batch.
4. `sim_car/sim_car/cones/nodes/memory_node.py::_pair_detections` greedily pairs LiDAR and camera detections before track updates.
5. `sim_car/sim_car/cones/tracking/tracker.py::LocalConeTracker.update` associates fused observations with existing tracks, updates position/color/confidence state, and creates new tracks when needed.
6. `sim_car/sim_car/cones/tracking/fusion.py::resolve_boundary_colors_for_planning` maps raw color labels to planner-side blue/yellow/unknown labels.
7. `sim_car/sim_car/cones/tracking/global_memory.py::GlobalConeMemory.update_from_tracks` and `PermanentConeStore` keep long-lived global and cross-lap cone estimates.
8. `sim_car/sim_car/cones/nodes/evaluator_node.py::ConeEvaluatorNode._match_predictions_to_gt` uses the same distance-gated matching idea for prediction-vs-ground-truth evaluation.

## Mathematical Building Blocks

### Planar Frame Transforms

`sim_car/sim_car/cones/tracking/pose.py::odom_point_to_base` and `base_point_to_odom` use the vehicle pose `(tx, ty, yaw)` as a 2D rigid transform. The inverse transform subtracts translation and rotates by `-yaw`; the forward transform rotates by `yaw` and adds translation.

The cone memory uses these transforms to keep two views of each detection:

- odom-frame coordinates for persistent tracking and planner publication.
- base-frame coordinates for range, behind-car pruning, near/far policy, and RViz/debug views.

`convert_odom_child_pose_to_base_frame` also shifts between body-center frames and the configured `front_axle` frame by `0.5 * wheelbase_m`. That keeps cone geometry consistent with the controller reference point.

### Constant-Twist Pose Projection

`sim_car/sim_car/cones/tracking/pose.py::project_planar_pose_constant_twist` projects odometry forward over a bounded delay. For near-zero yaw rate it uses straight-line motion. Otherwise it integrates a constant planar twist using the turn radius `speed / yaw_rate`.

This is not a vehicle dynamics model. It is a short-horizon timestamp compensation step so detections are transformed with a pose closer to the sensor acquisition time.

### Greedy Distance-Gated Pairing

`sim_car/sim_car/cones/nodes/memory_node.py::_pair_detections`, `sim_car/sim_car/cones/tracking/tracker.py::LocalConeTracker._associate`, and `sim_car/sim_car/cones/nodes/evaluator_node.py::ConeEvaluatorNode._match_predictions_to_gt` all build candidate pairs whose Euclidean distance is below a gate. Candidates are sorted by distance and accepted greedily while preventing reuse of either endpoint.

This gives deterministic one-to-one matching without solving a global assignment problem. That is adequate here because the cone spacing and gating radii are chosen so most ambiguous matches are local and sparse.

### Confidence-Weighted Deduplication

`sim_car/sim_car/cones/nodes/memory_node.py::_deduplicate_sensor_detections` sorts detections by range and confidence, finds compatible detections within a radius, and replaces them with a confidence-weighted position average. Color is taken from the higher-confidence detection unless one side is unknown.

`sim_car/sim_car/cones/tracking/global_memory.py::GlobalConeMemory.update_from_tracks` uses a related running average, with `alpha = 1 / max(2, hits + 1)`, so early observations move the global estimate more than later observations. This stabilizes displayed and stored cone positions over repeated passes.

### Track Position Filtering

`sim_car/sim_car/cones/tracking/tracker.py::_update_position` updates a track by exponential smoothing:

```text
new = (1 - alpha) * old + alpha * observation
```

Separate `alpha_lidar` and `alpha_camera` values let the memory trust LiDAR and camera position updates differently. The same update also stores a residual-derived variance proxy, which is useful as a simple track-stability signal.

### Color Belief, Hysteresis, And Decay

`sim_car/sim_car/cones/tracking/fusion.py::update_class_probs` keeps a probability vector over `unknown`, `blue`, `yellow`, and `orange`. New camera labels are blended toward a one-hot observation with strength based on detector confidence. Optional decay pulls old evidence down before normalization.

`update_color_belief` adds hysteresis through `color_switch_margin`. A track only changes stable color if the proposed class beats the current class by enough margin. This prevents color labels from flickering when detector confidence is close between classes.

`blend_track_confidence` maintains a bounded confidence score. Seen tracks are reinforced; missed tracks decay. `age_to_decay` converts elapsed time to an exponential-style decay factor.

### Near/Far Source Selection

`sim_car/sim_car/cones/tracking/fusion.py::choose_position_source` chooses LiDAR or camera positions using a near/far split. `near_band_limit(camera_range_m)` defines where LiDAR should dominate; beyond that, camera position can be used if available.

This policy reflects the simulator stack: LiDAR is preferred for close cone geometry, while camera detections can keep farther cones alive when LiDAR does not provide a useful return.

### Boundary Color Inference

`sim_car/sim_car/cones/tracking/fusion.py::resolve_boundary_colors_for_planning` first transforms cone points into the vehicle frame with `_rotate_into_vehicle`. Known blue/yellow labels pass through. Unknown and orange labels can be assigned from lateral sign, and orange cones can also use nearest known-color neighbors inside a configured radius with a distance margin.

The output is deliberately limited to planner-facing `blue`, `yellow`, or `unknown`, because the planners need boundary side more than raw semantic cone class.

## Function Reference

| Math operation | Function | Runtime use |
| --- | --- | --- |
| Odom to base transform | `sim_car/sim_car/cones/tracking/pose.py::odom_point_to_base` | Converts tracked/global cone coordinates into vehicle-local pruning and planning context. |
| Base to odom transform | `sim_car/sim_car/cones/tracking/pose.py::base_point_to_odom` | Converts incoming base-frame detections to persistent odom-frame tracks. |
| Odometry-lag projection | `sim_car/sim_car/cones/tracking/pose.py::project_planar_pose_constant_twist` | Compensates short odometry delay before frame conversion. |
| Front-axle/body-center shift | `sim_car/sim_car/cones/tracking/pose.py::convert_odom_child_pose_to_base_frame` | Aligns odometry child frame with the configured planner/control base frame. |
| Sensor-batch deduplication | `sim_car/sim_car/cones/nodes/memory_node.py::_deduplicate_sensor_detections` | Merges multiple detections of one cone before LiDAR/camera pairing. |
| LiDAR/camera pairing | `sim_car/sim_car/cones/nodes/memory_node.py::_pair_detections` | Produces fused observations from separate sensor inputs. |
| Track association | `sim_car/sim_car/cones/tracking/tracker.py::LocalConeTracker._associate` | Assigns new observations to existing `ConeTrack` objects. |
| Position EMA | `sim_car/sim_car/cones/tracking/tracker.py::_update_position` | Smooths noisy cone positions while preserving responsiveness. |
| Color probabilities | `sim_car/sim_car/cones/tracking/fusion.py::update_class_probs` | Accumulates detector color evidence over time. |
| Stable color label | `sim_car/sim_car/cones/tracking/fusion.py::update_color_belief` | Prevents rapid color switching from noisy classifications. |
| Track confidence | `sim_car/sim_car/cones/tracking/fusion.py::blend_track_confidence` | Promotes repeatedly observed tracks and decays missed tracks. |
| Boundary inference | `sim_car/sim_car/cones/tracking/fusion.py::resolve_boundary_colors_for_planning` | Produces planner-facing side colors for `/tracked_cones`. |
| Global memory merge | `sim_car/sim_car/cones/tracking/global_memory.py::GlobalConeMemory.update_from_tracks` | Maintains long-lived cone estimates for visualization and belief state. |
| Evaluation matching | `sim_car/sim_car/cones/nodes/evaluator_node.py::ConeEvaluatorNode._match_predictions_to_gt` | Computes matched prediction/ground-truth cone pairs for signed range-error samples. |

## Notes / Limits

- Matching is greedy, not globally optimal. It is intentionally simple and deterministic because the gating radius is the primary ambiguity control.
- Position filtering is an EMA, not a Kalman filter. The code stores useful stability proxies but does not model full covariance.
- Color inference from lateral sign assumes the vehicle is oriented consistently with the track boundary convention: positive lateral `y` maps to blue, negative lateral `y` maps to yellow.
- Constant-twist projection is capped in time. It is only meant to reduce small timestamp errors, not to predict through long dropouts.
