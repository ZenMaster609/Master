# Delaunay Planner Parameters

This document explains every parameter in [`sim_car/config/delaunay_planner.yaml`](/home/aleks/ros2_ws/src/Master/sim_car/config/delaunay_planner.yaml).

The file configures `delaunay_planner_node`, which is launched from [`full_sim_launch.launch.py`](/home/aleks/ros2_ws/src/Master/sim_car/launch/full_sim_launch.launch.py). The node reads these values at startup in [`delaunay_planner_node.py`](/home/aleks/ros2_ws/src/Master/sim_car/sim_car/planning/delaunay_planner_node.py), so YAML edits normally require a relaunch to take effect.

At a high level, the planner does this:

1. Read tracked cones and vehicle pose.
2. Filter cones by geometry, confidence, and color handling rules.
3. Build Delaunay-style edges and pick likely blue-yellow cross edges.
4. Convert those cross edges into midpoints and a resampled centerline.
5. Validate the new centerline against recent history, optionally smooth it over time, and convert it into the controller frame.
6. Run the Stanley steering controller and curvature-based speed command.

The geometry-heavy logic lives mostly in [`delaunay_planner_core.py`](/home/aleks/ros2_ws/src/Master/sim_car/sim_car/planning/delaunay_planner_core.py). The ROS integration, validation, smoothing, diagnostics, and command publishing live mostly in [`delaunay_planner_node.py`](/home/aleks/ros2_ws/src/Master/sim_car/sim_car/planning/delaunay_planner_node.py).

## `frames`

This group defines which coordinate frames the planner uses internally and how long it waits for TF.

- `frames.planning_frame = "odom"`: The preferred frame for planning outputs and internal geometry. If TF into this frame is unavailable, the node falls back to `odom`. This mainly affects ROS integration.
- `frames.odom_frame = "odom"`: The frame name treated as the odometry reference frame. This is used for TF matching, fallback pose lookup, and output publishing. Changing it only makes sense if the rest of the sim uses a different odom frame. This mainly affects ROS integration.
- `frames.base_frame = "front_axle"`: The vehicle frame used as the controller reference. It affects how the planner resolves vehicle pose and how the centerline is transformed into the controller's coordinate system. Changing it without matching TF and vehicle-model assumptions can shift perceived cross-track error. This mainly affects ROS integration and controller behavior.
- `frames.tf_timeout_s = 0.03`: Maximum TF wait time when resolving transforms. Larger values tolerate slower TF updates but increase latency before fallback; smaller values fail faster. This mainly affects ROS integration.

## `topics`

This group defines planner inputs and outputs.

- `topics.tracked_cones_topic = "/tracked_cones"`: Input cone detections used to build the centerline. If the topic becomes noisier or sparser, planner stability will degrade even if the planner code is unchanged. This mainly affects ROS integration.
- `topics.cmd_topic = "/cmd"`: Ackermann command output topic. It does not change the planner geometry, only where the controller output is published. This mainly affects ROS integration.
- `topics.centerline_topic = "/planned_centerline"`: Published `nav_msgs/Path` centerline topic. This is the main path output used by visualizers and diagnostics consumers. This mainly affects ROS integration and observability.
- `topics.viz_topic = "/planner_viz"`: `MarkerArray` output used for planner debug visualization. It affects observability only. This mainly affects debugging.
- `topics.points_topic = "/planned_centerline_points"`: Optional `PoseArray` output for centerline points. It is only populated if the optional point-publishing debug switch is enabled. This mainly affects debugging.
- `topics.odom_topic = "/sim/odom"`: Odometry input used for speed, yaw-rate, and pose fallback when TF is unavailable. If this topic has the wrong frame conventions, both planning and control quality suffer. This mainly affects ROS integration and controller behavior.

## `filtering`

This group controls which cones are considered usable before edge selection.

- `filtering.max_cone_range_m = 25.0`: Maximum radial distance from the vehicle for a cone to remain eligible. Increasing it includes more distant cones and gives longer-range structure, but also lets far noisy detections influence the path. Decreasing it makes the planner more local and often more stable. This mainly affects geometry selection.
- `filtering.behind_drop_m = 5.0`: How far behind the vehicle a cone or midpoint may remain in the candidate set. Increasing it keeps more history behind the car, which can help continuity but can also let stale turn-entry geometry distort the next section. Decreasing it usually makes the planner more forward-looking. This mainly affects geometry selection and path stability.
- `filtering.min_confidence = 0.3`: Minimum cone confidence accepted from the detection/tracking pipeline. Increasing it rejects uncertain detections and usually improves robustness at the cost of fewer usable cones. Decreasing it makes the planner more permissive. This mainly affects geometry selection.
- `filtering.use_unknown_cones = true`: Allows `unknown` cones to be included when there are too few confidently colored cones. Turning it off makes the planner stricter about color certainty, which can improve correctness when color inference is unreliable but can also starve the planner. This mainly affects geometry selection.
- `filtering.infer_unknown_by_side = true`: Re-labels `unknown` cones as left or right boundary cones based on lateral position relative to the car. Turning it on increases usable cone count; turning it off avoids side-inference mistakes. This mainly affects geometry selection and path stability.
- `filtering.infer_orange_by_side = true`: Attempts to reinterpret orange cones as blue or yellow when their lateral position or local neighborhood makes the side clear. Turning it on can recover useful boundary information; turning it off avoids false side assignment for ambiguous orange cones. This mainly affects geometry selection.
- `filtering.include_orange = false`: If enabled, orange cones are treated as directly usable boundary cones without needing reclassification. This can help when orange cones are a meaningful part of the boundary, but it can also contaminate the left-right pairing if orange usage is inconsistent. This mainly affects geometry selection.
- `filtering.orange_min_lateral_m = 0.9`: Lateral-distance threshold used when inferring whether an orange cone clearly belongs to one side. Increasing it makes orange side inference more conservative; decreasing it makes it easier for orange cones near the center to be forced onto one side. This mainly affects geometry selection.
- `filtering.orange_neighbor_radius_m = 3.5`: Neighborhood radius used when orange side assignment falls back to nearby blue/yellow cones. Increasing it allows more distant neighboring cones to influence the decision; decreasing it makes the inference more local. This mainly affects geometry selection.
- `filtering.orange_neighbor_margin_m = 0.75`: Required distance advantage for one boundary color over the other during orange side inference. Increasing it requires clearer evidence before assigning a side; decreasing it makes the orange inference more willing to choose. This mainly affects geometry selection and path stability.
- `filtering.min_colored_cones = 4`: Minimum count of confidently blue/yellow cones required before the planner stops considering `unknown` cones as a fallback. Increasing it makes the planner more likely to pull in unknown cones; decreasing it makes it rely on colored cones sooner. This mainly affects geometry selection.
- `filtering.min_required_cones = 4`: Minimum number of usable cones needed before planning continues at all. Increasing it makes the planner reject sparse scenes rather than guess; decreasing it allows planning from very limited structure. This mainly affects geometry selection and path stability.

## `edge_selection`

This group controls how the planner decides which cone pairs are plausible left-right cross edges.

- `edge_selection.min_cross_edge_m = 0.8`: Minimum allowed length of a boundary-crossing edge. Increasing it rejects very tight pairings that may be duplicate detections or artifacts; decreasing it allows narrower pairings. This mainly affects geometry selection.
- `edge_selection.max_cross_edge_m = 6.0`: Maximum allowed length of a cross edge. Increasing it allows wider pairings and makes it easier for distant or mis-grouped cones to form a midpoint; decreasing it is often useful when the centerline jumps outward before a turn. This mainly affects geometry selection and path stability.
- `edge_selection.cross_edge_lateral_ratio = 0.6`: Minimum fraction of an edge that must lie in the lateral direction when expressed in vehicle coordinates. Higher values require edges to look more side-to-side and reject diagonal pairings; lower values allow more forward-slanted pairings. This is one of the highest-leverage parameters for reducing bad pre-turn cross edges. This mainly affects geometry selection and path stability.
- `edge_selection.min_cross_edges = 3`: Minimum number of valid cross edges required before the planner is satisfied with the current edge set. If fewer are found, the core falls back to nearest blue-yellow pairing logic. Increasing it makes the planner demand a more complete structure before trusting the current graph; decreasing it makes planning easier in sparse scenes. This mainly affects geometry selection and path stability.

## `centerline`

This group controls midpoint cleanup, path resampling, and temporal blending.

- `centerline.min_spacing_m = 0.5`: Minimum spacing between ordered midpoint samples before resampling. Increasing it removes closely spaced midpoints and simplifies the path; decreasing it preserves more raw local detail. This mainly affects geometry selection and path smoothness.
- `centerline.path_resolution_m = 0.5`: Target spacing of the final resampled centerline. Smaller values produce denser paths; larger values produce sparser paths. This affects controller input resolution more than the underlying geometry choice. This mainly affects path representation and controller behavior.
- `centerline.max_path_length_m = 30.0`: Maximum centerline length produced by the core. Increasing it gives the controller more path ahead, but it also lets distant structure influence the path. Decreasing it keeps the plan more local. This mainly affects geometry selection and controller behavior.
- `centerline.enable_temporal_smoothing = true`: Enables blending of the current centerline with the previous one after validation. Turning it off makes the planner fully reactive to each new plan; turning it on reduces frame-to-frame jitter. This mainly affects path stability.
- `centerline.smoothing_alpha = 0.2`: Blend weight used in temporal smoothing. In this implementation, larger values give more weight to the new centerline and smaller values keep more of the previous one. Increasing it makes the path more responsive; decreasing it makes the path more stable but slower to adapt. This mainly affects path stability.

## `runtime`

This group controls how often the node runs and how often repeated warnings are throttled.

- `runtime.publish_rate_hz = 180.0`: Planner timer frequency. Higher values produce more frequent command/path updates and tighter steering-rate limiting steps; lower values reduce compute load but also reduce update frequency. This mainly affects runtime behavior and controller behavior.
- `runtime.log_throttle_s = 1.0`: Minimum time between repeated throttled warnings with the same key. Increasing it makes repeated issues quieter; decreasing it makes repeated issues more visible in logs. This mainly affects observability.

## `control`

This group selects which controller implementation is used after the centerline is generated.

- `control.controller_type = "stanley"`: Steering controller type. The current implementation only supports `stanley`, so this is effectively a fixed selector unless more controllers are added later. This mainly affects controller behavior.

## `stanley`

This group controls the Stanley steering law and the post-processing applied to steering commands.

- `stanley.k_gain = 1.2`: Gain on the cross-track term inside the Stanley controller. Increasing it makes the car correct lateral error more aggressively; decreasing it softens that response. Too high can cause sharper steering reactions to small path offsets. This mainly affects controller behavior.
- `stanley.softening_speed_mps = 0.0`: Speed offset added to the denominator of the Stanley cross-track term. Increasing it reduces low-speed aggressiveness and makes steering less sensitive when the vehicle is slow; decreasing it makes low-speed cross-track corrections stronger. This mainly affects controller behavior.
- `stanley.heading_gain = 1.0`: Multiplier on heading error relative to the path segment. Increasing it prioritizes heading alignment; decreasing it prioritizes the cross-track term more. This mainly affects controller behavior.
- `stanley.lookahead_idx_offset = 0`: Segment index offset used when choosing the path segment for heading estimation. Increasing it makes the controller align to a point farther ahead on the path, which can smooth behavior but may cut corners; decreasing it keeps heading alignment more local. This mainly affects controller behavior.
- `stanley.steering_limit_rad = 0.52`: Hard clamp on steering command magnitude before filtering. Increasing it allows tighter commanded curvature; decreasing it caps steering earlier. This mainly affects controller behavior.
- `stanley.steering_lowpass_alpha = 1.0`: Low-pass blend weight between the newly clamped steering command and the previous steering command. In this implementation, `1.0` means effectively no smoothing and smaller values retain more of the previous command. Decreasing it smooths steering but adds lag. This mainly affects controller behavior.
- `stanley.steering_rate_limit_rad_s = 10.0`: Maximum steering-command slew rate. Increasing it allows faster steering transients; decreasing it forces smoother changes. A value at or below zero disables rate limiting. This mainly affects controller behavior.
- `stanley.use_yaw_rate_damping = true`: Enables subtraction of a yaw-rate damping term from the steering command. Turning it off removes that stabilizing term entirely. This mainly affects controller behavior.
- `stanley.yaw_rate_damping_gain = 0.0`: Gain on the yaw-rate damping term. Increasing it makes the controller push back more strongly against current yaw rate; with the current default, the feature is enabled but inactive because the gain is zero. This mainly affects controller behavior.
- `stanley.wheelbase_m = 1.65`: Wheelbase used when converting steering angle to curvature and when deriving some front-axle pose assumptions. It should match the simulated vehicle reasonably well. This mainly affects controller behavior and ROS integration.
- `stanley.cross_track_deadband_m = 0.0`: Cross-track error deadband before the controller reacts. Increasing it ignores tiny lateral offsets and can reduce steering chatter; decreasing it makes the controller react to even very small offsets. This mainly affects controller behavior.
- `stanley.stop_if_no_path = true`: Behavior when no valid path is available. If true, the node publishes zero speed and zero steering; if false, it reuses the last command when possible. This mainly affects controller behavior and safety.

## `speed_control`

This group controls the simple curvature-based speed target generated from the chosen steering curvature.

- `speed_control.speed_min_mps = 1.0`: Minimum commanded speed after curvature scaling. Increasing it prevents the car from slowing too much in tight turns; decreasing it allows slower corner entry. This mainly affects controller behavior.
- `speed_control.speed_max_mps = 4.0`: Maximum commanded speed in low-curvature sections. Increasing it raises straight-line speed; decreasing it lowers the whole speed envelope. This mainly affects controller behavior.
- `speed_control.curvature_speed_gain = 4.0`: How strongly commanded speed falls as path curvature increases. Increasing it makes the car slow down more aggressively in corners; decreasing it keeps speed higher even on curved paths. This mainly affects controller behavior.
- `speed_control.lowpass_speed_alpha = 0.15`: Blend weight between the new desired speed and the previous speed command. Larger values follow new speed targets faster; smaller values retain more of the previous command and smooth speed transitions. This mainly affects controller behavior.

## `validation`

This group decides whether a newly generated centerline should be accepted, rejected, or temporarily replaced by the last valid one.

- `validation.max_centerline_jump_m = 0.8`: Maximum allowed lateral deviation between the new centerline and the recent valid centerline history over the configured consistency horizon. Increasing it makes the validator more permissive; decreasing it rejects path flips sooner. This is one of the main parameters to tune when the midline jumps outward before a turn. This mainly affects path stability.
- `validation.consistency_horizon_m = 8.0`: Forward distance over which the new centerline is compared against recent valid history. Increasing it makes the validator compare farther ahead; decreasing it focuses the check on the near-term path. This mainly affects path stability.
- `validation.max_history_frames = 36`: Number of previously accepted centerlines kept for validation and stability history. Increasing it gives a longer memory; decreasing it makes the reference history more recent. This mainly affects path stability.
- `validation.hold_last_valid_s = 0.5`: How long the planner may keep publishing the previous valid centerline when the current one is rejected or missing. Increasing it makes the planner ride through brief instability better; decreasing it makes it give up sooner. This mainly affects path stability.
- `validation.min_stable_frames = 3`: Number of accepted centerlines required before the planner considers its history stable enough to trust or hold. Increasing it makes the node wait longer before accepting the history as reliable; decreasing it allows earlier activation. This mainly affects path stability.
- `validation.max_selected_edge_churn_ratio = 0.55`: Maximum allowed frame-to-frame churn in the selected cross-edge set before the new plan is rejected. Increasing it tolerates more graph reconfiguration; decreasing it rejects sudden edge flips more aggressively. This is another high-value parameter for suppressing inconsistent pre-turn path jumps. This mainly affects path stability.

## `diagnostics`

This group controls what planner stability data is published and when warnings are raised.

- `diagnostics.topic = "/delaunay_planner/diagnostics"`: Output topic for `DiagnosticArray` status messages. It does not change planner behavior, only where diagnostics are published. This mainly affects observability.
- `diagnostics.centerline_jump_horizon_m = 8.0`: Horizon used for the live diagnostic jump metric between the current raw centerline and the previous raw centerline. It is separate from the validation horizon and is used for warning/monitoring. This mainly affects observability.
- `diagnostics.edge_quantization_m = 0.05`: Spatial quantization used when converting selected edges into comparable keys for churn measurement. Increasing it makes edge matching more tolerant to small coordinate drift; decreasing it makes churn calculation more exact. This mainly affects observability and path stability metrics.
- `diagnostics.jump_warn_threshold_m = 0.8`: Threshold above which the live centerline jump metric triggers a warning log. Increasing it reduces warnings; decreasing it makes warnings more sensitive. This does not itself reject plans. This mainly affects observability.
- `diagnostics.edge_churn_warn_threshold = 0.55`: Threshold above which selected-edge churn triggers a warning log. Increasing it reduces warnings; decreasing it makes warnings more sensitive. This does not itself reject plans unless the validation limit is also crossed. This mainly affects observability.
- `diagnostics.publish_control_debug = true`: If enabled, the node publishes a second diagnostic status containing detailed Stanley controller internals such as heading error, cross-track error, steering clamp/filter/rate-limit stages, and target-point coordinates. Turning it off reduces debug visibility but does not change control logic. This mainly affects observability.

## `debug`

This group controls RViz/debug outputs published on the planner visualization topics.

- `debug.enable_markers = true`: Master switch for planner `MarkerArray` publishing. Turning it off removes marker output entirely but leaves the actual planner behavior unchanged. This mainly affects debugging.
- `debug.show_raw_cones = false`: Shows the filtered cone set used by the core as gray points. Useful when checking whether the planner is being destabilized by the input cone set. This mainly affects debugging.
- `debug.show_delaunay_edges = true`: Shows all triangulation or fallback graph edges used before cross-edge filtering. Useful for seeing whether the graph itself is reasonable. This mainly affects debugging.
- `debug.show_candidate_edges = true`: Shows geometry-valid candidate cross edges after length and lateral-ratio filtering. Useful for understanding why the planner is considering a strange pairing. This mainly affects debugging.
- `debug.show_selected_edges = true`: Shows the final selected cross edges that generate the path midpoints. If the centerline jumps, this is one of the first marker layers to inspect. This mainly affects debugging.
- `debug.publish_points_topic = false`: Publishes the centerline as a `PoseArray` on the configured points topic in addition to the `Path`. This is extra debug output only. This mainly affects debugging.
- `debug.show_lookahead_point = true`: Shows the controller target point used by Stanley in the visualization frame. This helps separate path-generation problems from controller-targeting problems. This mainly affects debugging.

## Most likely tuning knobs for path jumps

If the centerline occasionally jumps outside the track before a turn, the highest-value controls are usually:

- the jump-rejection threshold in the validation block
- the selected-edge churn limit in the validation block
- the previous-path hold time in the validation block
- how much geometry behind the vehicle is kept
- the maximum allowed cross-edge width
- how strictly cross edges must look lateral instead of diagonal
- how far away cones may still influence the planner

The main tradeoff is responsiveness versus stability. Geometry and validation parameters change which path gets accepted. Stanley parameters mostly change how the car follows the accepted path. If the path itself is visibly wrong in RViz, tune `filtering`, `edge_selection`, `centerline`, and `validation` before touching the controller.

## Minimal commands

Source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Relaunch the sim and planner after changing the YAML:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py
```
