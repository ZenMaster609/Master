# `driving_clean.rviz`

This document explains the line and text overlays shown by [`sim_car/rviz/driving_clean.rviz`](../sim_car/rviz/driving_clean.rviz).

`driving_clean.rviz` and [`sim_car/rviz/planner_debug.rviz`](../sim_car/rviz/planner_debug.rviz) are almost the same. The only relevant difference here is that `planner_debug.rviz` enables `Global Track Hypothesis` on `/cone_memory/believed_track_viz`, while `driving_clean.rviz` disables it. The planner overlays on `/planner_viz` and the final path on `/planned_centerline` are the same in both files.

## What is actually visible in `driving_clean.rviz`

Enabled displays that matter for lines and text:

- `Cone Memory` on `/local_cone_map_viz`
- `Planner Debug` on `/planner_viz`
- `Planned Centerline` on `/planned_centerline`

Enabled but not planner-line/text overlays:

- `TF`
- `Odometry` on `/sim/odom`

Disabled in `driving_clean.rviz`, so not part of the visible explanation:

- `Global Track Hypothesis` on `/cone_memory/believed_track_viz`
- `Raw Lidar Debug`
- `Raw Camera Debug`

## Cone Memory

`Cone Memory` does not draw any lines in the current config. It publishes cone cylinders from [`memory_node.py`](../sim_car/sim_car/cones/nodes/memory_node.py).

Text is also off by default because [`cone_memory.yaml`](../sim_car/config/cone_memory.yaml) sets `enable_id_text: false`.

So in the current `driving_clean.rviz` setup:

- line overlays from `Cone Memory`: not used
- text overlays from `Cone Memory`: not used

## Planner Lines

All of the following come from [`hybrid_boundary_planner_node.py`](../sim_car/sim_car/planning/hybrid_boundary_planner_node.py) and [`hybrid_boundary_planner_core.py`](../sim_car/sim_car/planning/hybrid_boundary_planner_core.py).

### `boundary_left`

Blue line and blue points.

Meaning:

- The ordered left boundary chain built from blue cones.

How it is calculated:

- Cones are first filtered by range, behind-vehicle cutoff, confidence, and color.
- Blue-side cones are ordered with `_build_boundary_chain(...)`.
- The seed is the nearest forward cone with the smallest `x`, then smallest `|y|`.
- The chain is grown greedily by accepting the best next cone that satisfies:
  - enough forward progress
  - step distance within `boundary_chain.min_step_m` to `boundary_chain.max_step_m`
  - heading change below `boundary_chain.max_heading_change_rad`
  - not shadowed by a nearer cone in nearly the same direction

Single-boundary note:

- Still used. Even in forced single-boundary mode, the planner still builds both boundary chains before choosing the side it will offset inward from.

### `boundary_right`

Yellow line and yellow points.

Meaning:

- The ordered right boundary chain built from yellow cones.

How it is calculated:

- Same chain-building logic as `boundary_left`, but using yellow cones.

Single-boundary note:

- Still used for chain construction and fallback-side selection.

### `accepted_pairs`

Green connector lines between left and right cones.

Meaning:

- Accepted left/right cone pairings used by midpoint-mode planning.

How it is calculated:

- `_pair_boundary_chains(...)` walks along the longer boundary as the anchor chain.
- For each anchor cone it looks for an opposite-side partner that passes:
  - valid width range
  - correct inward side
  - forward-progress consistency
  - width-jump consistency against the previous pair
- If enabled, unknown-color cones can be used as pair completion when they are close to the expected partner location.

Single-boundary note:

- Not used for single boundary planner.

### `raw_midpoint_chain`

White line.

Meaning:

- The raw midpoint sequence before path smoothing/resampling.

How it is calculated:

- For each accepted pair, the planner takes the midpoint between the left and right cone positions.
- Those midpoints are stacked in pair order.

Single-boundary note:

- Not used for single boundary planner.

### `raw_offset_path`

Cyan line and cyan points.

Meaning:

- The raw single-boundary candidate path before smoothing/resampling.

How it is calculated:

- If midpoint mode is unavailable, the planner selects one fallback boundary with `_select_fallback_chain(...)`.
- That selection prefers:
  - larger forward extent
  - then longer chain length
  - then smaller mean heading change
- `_offset_boundary_chain(...)` then offsets every boundary point inward by half the current track width estimate.
- The inward direction is the boundary tangent normal, so the path stays parallel to the chosen boundary.

Single-boundary note:

- This is the main raw geometry for forced single-boundary planning.

### `raw_prevalidation_centerline`

Magenta line.

Meaning:

- The planner’s finalized candidate path before the node’s stored-midline/hold logic changes what gets published.

How it is calculated:

- The core picks `raw_curve`:
  - midpoint chain in midpoint mode
  - raw offset path in single-boundary mode
- `_finalize_path(...)` then applies:
  - moving-average smoothing with `centerline.smoothing_window`
  - resampling to `centerline.path_resolution_m`
  - truncation to `centerline.max_path_length_m`
- The result is then validated for:
  - minimum point count
  - minimum forward extent
  - near-field continuity
  - start-heading error
  - heading-delta limit
  - self-intersection

Single-boundary note:

- Used for single boundary planner.

### `centerline` on `/planner_viz`

Red line marker.

Meaning:

- The final path the node is currently publishing after all node-level stability logic.

How it is calculated:

- The node starts from the validated centerline if one exists.
- If validation failed but the planner is in single-boundary mode, it can still soft-accept a finalized `raw_offset_path` as a candidate source.
- That candidate is then passed through the persistent midline buffer:
  - accepted candidates update the stored path
  - rejected candidates can cause the node to keep publishing the stored path for a while
- The path is then trimmed to the forward part in front of the car.
- Near the vehicle it is anchored so the first point is at the vehicle origin and the near segment is pulled toward zero lateral offset.

Single-boundary note:

- Used for single boundary planner.

### `Planned Centerline` on `/planned_centerline`

Red `Path` display.

Meaning:

- The same final centerline as above, but published as `nav_msgs/Path` instead of marker geometry.

How it is calculated:

- The node publishes the final `centerline` array on `/planned_centerline`.
- RViz renders that topic as a `Path`.

Single-boundary note:

- Used for single boundary planner.

## Planner Text

The only planner text visible by default is the status block in marker namespace `status` on `/planner_viz`.

### Status block

This text is built by `_build_operator_status_text(...)` in [`hybrid_boundary_planner_node.py`](../sim_car/sim_car/planning/hybrid_boundary_planner_node.py).

#### `STATE`

Meaning:

- What the node is currently doing with the path it has.

How it is calculated:

- `FRESH`: a current cycle produced a usable path and the node is publishing it.
- `HELD`: the current cycle was not good enough, so the node is still publishing a stored previous valid path.
- `STOPPED`: no usable path is available, or controller/path handling failed, and stop behavior has taken over.
- `WAITING`: early startup or missing required inputs.

#### `MODE`

Meaning:

- Which planner mode is active.

How it is calculated:

- Comes from `result.planner_mode` plus node-level overrides.
- Possible values are mainly `MIDPOINT`, `SINGLE_BOUNDARY`, and `HOLDING_LAST_VALID`.

Single-boundary note:

- In forced single-boundary operation this is effectively `SINGLE_BOUNDARY` when planning normally, or `HOLDING_LAST_VALID` when the node is publishing stored geometry.

#### `REASON`

Meaning:

- Why the current `STATE` happened.

How it is calculated:

- Derived from the core rejection result and node/controller state.
- Examples are things like near-field continuity rejection, midpoint kink rejection, holding previous valid path, hold expired, no safe chain, controller failure, or stop-if-no-path.

#### `TRACKS`

Meaning:

- `remembered`: how many cones are in the latest `/tracked_cones`
- `stale`: how many of those tracks are marked stale

How it is calculated:

- These are counted directly from the latest `ConeDetectionArray`.

Single-boundary note:

- Used for single boundary planner.

#### `BOUNDARIES`

Meaning:

- `L`: left boundary chain length
- `R`: right boundary chain length
- `pairs`: accepted pair count
- `unknown`: accepted unknown-color completion count
- `path`: final published path point count

How it is calculated:

- `L` and `R` come from the boundary-chain builder.
- `pairs` and `unknown` come from `_pair_boundary_chains(...)`.
- `path` is the number of points in the final published centerline.

Single-boundary note:

- `L`, `R`, and `path` are used for single boundary planner.
- `pairs` and `unknown`: not used for single boundary planner.

#### `WIDTH`

Meaning:

- The planner’s current track-width estimate.

How it is calculated:

- The width starts from `width_estimation.initial_width_m`.
- It is only updated from measured pair widths when midpoint mode has enough trustworthy pairs.
- The update uses a clipped low-pass filter in `update_track_width_estimate(...)`.

Single-boundary note:

- Used for single boundary planner because the raw offset path is shifted inward by half this width.
- In forced single-boundary mode it normally stays at the configured prior width, because midpoint pair measurements are not being used to refresh it.

#### `CMD`

Meaning:

- `v`: commanded speed
- `delta`: commanded steering angle
- `Ld`: controller lookahead distance

How it is calculated:

- These come from the active steering controller after it receives the final control path.
- `v` is then shaped by the speed-control logic from path curvature.

Single-boundary note:

- Used for single boundary planner.

#### `NF`

Meaning:

- `lat`: maximum near-field lateral change against the previous path
- `kink`: maximum heading step change along the current candidate path
- `seed`: distance from the vehicle to the first point of the candidate path

How it is calculated:

- `lat` comes from `_near_field_delta_metrics(...)`, which compares current and previous centerlines in vehicle frame after resampling both to 0.25 m stations up to the jump-check horizon.
- `kink` comes from `_path_heading_delta_max(...)`, the maximum absolute change between consecutive segment headings.
- `seed` comes from `_first_point_distance(...)`.

Single-boundary note:

- Used for single boundary planner.

#### `HOLD`

Meaning:

- How long the node can keep using stored valid geometry, whether it is currently doing so, and how many clean frames it has seen toward leaving hold mode.

How it is calculated:

- Remaining time comes from `_hold_remaining_s(...)`.
- `held` is `1` when a stored path is currently being published.
- `clean a/b` is the current clean-frame count against `validation.hold_exit_clean_frames`.

Single-boundary note:

- Used for single boundary planner.

## Important practical note for forced single-boundary mode

If `planner.force_single_boundary: true` in [`hybrid_boundary_planner.yaml`](../sim_car/config/hybrid_boundary_planner.yaml), the overlays that matter most are:

- `boundary_left`
- `boundary_right`
- `raw_offset_path`
- `raw_prevalidation_centerline`
- `centerline`
- `Planned Centerline`
- the `status` text block

These are the planner overlays that still describe the actual path-generation flow.

The midpoint-only overlays are:

- `accepted_pairs`
- `raw_midpoint_chain`
- the `pairs` and `unknown` fields in `BOUNDARIES`

## Small RViz difference from `planner_debug.rviz`

If you open `planner_debug.rviz`, the extra visible layer is `Global Track Hypothesis` on `/cone_memory/believed_track_viz`.

That layer is not part of `driving_clean.rviz`, so it is intentionally not explained in detail here.
