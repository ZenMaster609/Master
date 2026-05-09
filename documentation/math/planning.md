# Planning Math

## Scope

This page documents the mathematical functions and pipelines used by the tracked-cone planners, midline memory, ground-truth midline helpers, and skidpad router. It focuses on geometric construction, path validation, candidate scoring, and path stabilization.

The planning package receives cone tracks in odom/global coordinates, converts them into vehicle-local coordinates, builds a centerline candidate, validates it, stabilizes it over time, and passes it to the selected steering controller.

## Pipeline Map

1. `sim_car/sim_car/planning/tracked_cone_planner_base.py::TrackedConePlannerBase._resolve_cone_planning_context` collects tracked cones, pose, and transforms.
2. Shared geometry in `sim_car/sim_car/planning/tracked_cone_planner_geometry.py` filters cones, orders boundary chains, estimates tangents, computes widths, resamples paths, and validates path shape.
3. One planner core produces a candidate centerline:
   - `midpoint_planner_core.py::compute_midpoint_centerline`
   - `single_boundary_planner_core.py::compute_single_boundary_centerline`
   - `corridor_planner_core.py::compute_corridor_centerline`
4. `sim_car/sim_car/planning/midline_memory.py::CommittedMidlineMemory.update` blends accepted candidates with stored path memory or holds the last valid path.
5. `sim_car/sim_car/planning/tracked_cone_planner_node.py` selects the path source, runs the controller, and publishes path, diagnostics, and markers.
6. Special helpers in `ground_truth_midline.py` and `skidpad_router_core.py` provide debug paths and skidpad mission routing.

## Mathematical Building Blocks

### Vehicle-Frame Conversion

`sim_car/sim_car/planning/tracked_cone_planner_geometry.py::to_vehicle_frame` converts odom/global cone coordinates into local coordinates:

```text
x_local = cos(yaw) * dx + sin(yaw) * dy
y_local = -sin(yaw) * dx + cos(yaw) * dy
```

Most planner tests are easier in this frame: forward extent is `x`, boundary side is the sign of `y`, near-field checks can compare lateral path jumps, and controller paths can be returned directly.

### Cone Filtering And Deterministic Ordering

`_filter_and_order_cones` applies geometric range gates, confidence gates, and color/unknown selection. `_deterministic_order` then sorts by color rank, forward distance, lateral placement, and global coordinates.

The goal is not mathematical optimality. The goal is repeatable candidate generation so identical cone inputs produce identical pair and path selection, which makes tuning and debugging possible.

### Boundary Chain Growth

`build_boundary_chain_data` and `grow_boundary_chain_positions` build ordered side-boundary chains. A seed cone is selected near the vehicle and ahead of it with `select_seed_index`. The chain then greedily extends through remaining cones using:

- step distance inside `[min_step_m, max_step_m]`.
- positive projection onto the current heading.
- bounded heading change via `angle_between`.
- a shadowing test that rejects farther candidates hidden behind a closer candidate along nearly the same ray.

This produces a local ordered boundary without fitting a spline or solving a global graph problem.

### Tangents, Normals, And Width

`estimate_tangents` uses endpoint differences at the ends and centered differences inside the chain. `inward_normal` rotates the tangent toward the track interior based on side color. Width checks use Euclidean distance between paired left/right cones and compare it against configured min/max values.

The inward normal is central to both midpoint and single-boundary planning because it defines where the opposite boundary should be relative to an anchor cone.

### Path Resampling And Smoothing

`path_cumulative_lengths`, `_resample_path`, `sample_path_at_lengths`, and `resample_to_count` convert irregular point chains into approximately uniform station samples. `moving_average` provides local smoothing before final path publication.

Resampling gives later validation and controller stages a consistent path scale. Without it, a dense cone segment and a sparse cone segment would have different influence on heading, curvature, and controller lookahead.

### Heading, Curvature, And Self-Intersection Checks

`path_heading_delta_max` computes the largest wrapped heading change between adjacent path segments. `path_curvature_abs_max` divides heading change by local arc-length spacing, giving a discrete curvature proxy. `path_self_intersects` tests non-adjacent segment pairs with orientation signs in `segments_intersect`.

These tests reject centerlines that are geometrically plausible point averages but too discontinuous or self-crossing for controller use.

### Track Width Estimate

`update_track_width_estimate` clamps the previous width prior, clamps the measured width, limits per-update width delta, blends by `width_filter_alpha`, and clamps again. Midpoint, single-boundary, and corridor planners use this to keep expected track width stable when pair observations are temporarily sparse.

## Midpoint Planner

`sim_car/sim_car/planning/midpoint_planner_core.py::compute_midpoint_centerline` builds a centerline from paired left/right cones.

The key math stages are:

- `_geometry_filter`: keeps finite cones inside range and not too far behind the vehicle.
- `_pair_boundary_chains`: creates candidate right-side partners for each left anchor, including unknown-color completion when enabled.
- `_real_pair_candidate`: checks width, inward projection, raw color compatibility, and midpoint span.
- `_unknown_pair_candidate`: compares an unknown partner to the expected partner location generated from the anchor inward normal and expected width.
- `_pair_candidate_selection_key`: sorts candidates by cost, unknown penalty, width, range, forward progress, lateral magnitude, and track IDs.
- `_select_boundary_pairs`: greedily accepts non-overlapping pairs.
- `_order_pairs_into_midpoint_chain`: orders pair midpoints by a local progress reference and penalizes backward progress and width jumps.
- `_validate_midpoint_centerline`: finalizes the midpoint chain, computes near-field deltas against the previous path, checks heading delta, and returns either an accepted path or a rejection reason.

This planner is strongest when both boundaries are visible and color labels are reliable.

## Single-Boundary Planner

`sim_car/sim_car/planning/single_boundary_planner_core.py::compute_single_boundary_centerline` can produce a path when only one boundary is reliable.

The key math stages are:

- `_select_pairing_anchor`: chooses the longer visible boundary as the pairing anchor.
- `_real_partner_option` and `_unknown_partner_option`: score real or unknown opposite-side candidates with longitudinal offset, width error, and expected-partner distance.
- `_prefer_previous_partner_option`: biases toward previous partner track IDs inside a reassignment margin to reduce pair switching.
- `_select_fallback_chain`: chooses the boundary with larger forward extent, more points, and lower mean heading change.
- `_offset_boundary_chain`: offsets the selected boundary inward by `0.5 * expected_width_m` along the tangent-derived normal.
- `_candidate_path_metrics` and `_validated_centerline`: finalize, resample, and validate the offset path using the same near-field and heading continuity ideas as midpoint planning.

This planner trades geometric certainty for availability: one good boundary is enough to keep the car moving through temporary occlusion or color imbalance.

## Corridor Planner

`sim_car/sim_car/planning/corridor_planner_core.py::compute_corridor_centerline` constructs a corridor from two boundary chains and fits a centerline inside it.

The key math stages are:

- `_make_boundary_chains`: builds left/right chains with corridor-specific step limits and rejection reasons.
- `_resample_boundary_by_station`: resamples each boundary by arc length so left and right stations can be compared.
- `_corridor_valid_mask`: rejects station pairs with invalid coordinates, out-of-range width, behind-car samples, or samples beyond the planning horizon.
- `_fill_small_invalid_gaps` and `_longest_valid_slice`: keep the longest coherent corridor segment while tolerating isolated invalid samples.
- `_corridor_candidate_parts`: computes anchors as `0.5 * (left + right)`.
- `_fit_centerline_from_anchors`: smooths the anchor path, adapts smoothing for tighter turns, and keeps the candidate with acceptable or lowest curvature.
- `_path_violates_corridor`: resamples the centerline to corridor station count and checks deviation against half corridor width plus membership margin.
- `_validate_corridor_path`: applies corridor sample count, finite geometry, forward extent, membership, initial heading, near-field continuity, heading delta, curvature, and self-intersection checks.

This planner is the strictest of the main planners because it requires a valid corridor overlap and rejects paths that exit that corridor.

## Midline Memory

`sim_car/sim_car/planning/midline_memory.py::CommittedMidlineMemory.update` stabilizes accepted planner paths over time.

The key math stages are:

- `_candidate_forward` and `_stored_forward`: project the vehicle onto the candidate/stored path and extract the forward portion.
- `_resample_stations`: sample paths at fixed station spacing up to the configured horizon.
- `_transition_metrics`: compare stored and candidate forward paths by max displacement, lateral max, and lateral mean over near-field horizons.
- `_candidate_can_recover_from_jump`: allows recovery after repeated rejected jumps if the near-field lateral delta becomes small enough.
- `_blend_samples_path_relative`: blends stored and candidate samples in the candidate path's local tangent/normal frame. Longitudinal placement follows the candidate, while lateral motion is limited by near/mid/far alpha and max-shift parameters.

This memory layer makes path publication less sensitive to one-frame cone-pair changes while still allowing the path to migrate when a new candidate remains consistent.

## Ground-Truth And Skidpad Helpers

`sim_car/sim_car/planning/ground_truth_midline.py` provides debug/reference midlines:

- `order_boundary_points` greedily orders boundary cones from a start point using distance, forward cosine, lateral penalty, and heading projection.
- `project_point_to_path_s` and `sample_path_at_lengths` provide arc-length projection and interpolation.
- `_build_midpoint_chain_from_matched_boundaries` uses dynamic programming to match two ordered boundaries while allowing one side to advance without the other.
- `_close_loop_if_needed` closes a loop when endpoint gap is small relative to median segment length.

`sim_car/sim_car/planning/skidpad_router_core.py` provides mission-specific geometry:

- `SkidpadStateMachine.route_mask` selects cones inside active route regions using rectangles, circles, and straight corridors.
- `_update_circle_progress` accumulates wrapped angular progress around a lobe center and arms lap completion after a configured angle.
- `detect_stop_line_forward_distance_m`, `detect_stop_line_pair`, and `detect_acceleration_stop_row` detect forward orange-cone rows using cluster depth, lateral span, side counts, pair distance, and median forward position.

## Function Reference

| Math operation | Function | Runtime use |
| --- | --- | --- |
| Vehicle-frame transform | `sim_car/sim_car/planning/tracked_cone_planner_geometry.py::to_vehicle_frame` | Converts global cone/path geometry into planner-local coordinates. |
| Boundary chain growth | `sim_car/sim_car/planning/tracked_cone_planner_geometry.py::build_boundary_chain_data` | Builds ordered left/right side chains. |
| Tangent estimation | `sim_car/sim_car/planning/tracked_cone_planner_geometry.py::estimate_tangents` | Supplies normals and heading continuity checks. |
| Inward normal | `sim_car/sim_car/planning/tracked_cone_planner_geometry.py::inward_normal` | Defines expected opposite boundary and single-boundary offset direction. |
| Width prior update | `sim_car/sim_car/planning/tracked_cone_planner_geometry.py::update_track_width_estimate` | Stabilizes expected track width across planner cycles. |
| Path resampling | `sim_car/sim_car/planning/tracked_cone_planner_geometry.py::_resample_path` | Produces uniform path stations for validation/control. |
| Heading check | `sim_car/sim_car/planning/tracked_cone_planner_geometry.py::path_heading_delta_max` | Rejects abrupt heading changes. |
| Curvature check | `sim_car/sim_car/planning/tracked_cone_planner_geometry.py::path_curvature_abs_max` | Rejects overly sharp discretized paths. |
| Self-intersection | `sim_car/sim_car/planning/tracked_cone_planner_geometry.py::path_self_intersects` | Rejects crossing centerline geometry. |
| Midpoint centerline | `sim_car/sim_car/planning/midpoint_planner_core.py::compute_midpoint_centerline` | Two-boundary cone pairing and midpoint-chain planning. |
| Single-boundary centerline | `sim_car/sim_car/planning/single_boundary_planner_core.py::compute_single_boundary_centerline` | Offset fallback path from one reliable boundary. |
| Corridor centerline | `sim_car/sim_car/planning/corridor_planner_core.py::compute_corridor_centerline` | Strict valid-corridor centerline construction. |
| Path memory update | `sim_car/sim_car/planning/midline_memory.py::CommittedMidlineMemory.update` | Blends or holds centerlines across frames. |
| GT midline | `sim_car/sim_car/planning/ground_truth_midline.py::build_gt_midline_from_cones` | Builds debug/reference paths from known cone layout. |
| Skidpad routing | `sim_car/sim_car/planning/skidpad_router_core.py::SkidpadStateMachine.route_mask` | Masks tracked cones to the active skidpad branch. |
| Stop-row detection | `sim_car/sim_car/planning/skidpad_router_core.py::detect_acceleration_stop_row` | Detects mission stop geometry from orange cones. |

## Notes / Limits

- The planner cores use deterministic greedy choices in several places. This makes behavior repeatable, but it is not a global optimum over all cone pairings.
- Curvature and heading metrics are discrete path metrics. Their values depend on resampling resolution and smoothing window.
- Unknown-color pair completion is intentionally bounded by width, radial, longitudinal, and consecutive-unknown limits because unknown cones can otherwise produce convincing but wrong geometry.
- Midline memory can hold a path after planner rejection; diagnostics should be checked to distinguish fresh paths from held paths.
