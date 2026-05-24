# Planning System Code Map

This page maps `documentation/concepts/planning_system.md` to the refactored planning source files.

## Primary Files

- `sim_car/launch/full_sim_launch.launch.py`
- `sim_car/sim_car/planning/planner_constants.py`
- `sim_car/sim_car/planning/tracked_cone_planner_contract.py`
- `sim_car/sim_car/planning/tracked_cone_planner_base.py`
- `sim_car/sim_car/planning/tracked_cone_planner_node.py`
- `sim_car/sim_car/planning/tracked_cone_planner_geometry.py`
- `sim_car/sim_car/planning/midpoint_planner_core.py`
- `sim_car/sim_car/planning/single_boundary_planner_core.py`
- `sim_car/sim_car/planning/corridor_planner_core.py`
- `sim_car/sim_car/planning/midline_memory.py`
- `sim_car/sim_car/planning/planning_state_machine.py`
- `sim_car/sim_car/planning/planning_diagnostics.py`
- `sim_car/sim_car/planning/planning_visualization.py`
- `sim_car/sim_car/planning/linetest_planner_node.py`
- `sim_car/sim_car/planning/skidpad_router_node.py`
- `sim_car/sim_car/planning/skidpad_router_core.py`

## Launch Selection

- `PLANNER_REGISTRY` in `planner_constants.py`: maps `midpoint`, `single_boundary`, `corridor`, `linetest`, and `none` to executable names, diagnostics topics, default RViz profiles, and allowed tracks.
- `get_planner_spec` in `planner_constants.py`: resolves the launch-time planner name.
- `planner_allowed_for_track` in `planner_constants.py`: rejects invalid combinations such as `linetest` on skidpad.
- `_resolve_launch_selection` in `full_sim_launch.launch.py`: resolves the track, planner, controller, world, spawn defaults, controller overlays, and generated parameter overlay.
- `planner_input_topic` in `full_sim_launch.launch.py`: selects `/tracked_cones`, `/tracked_cones/skidpad_routed`, or direct perception input depending on cone memory, track, and routing choices.

## Shared Constants And Runtime Types

- `planner_constants.py`: holds message track-state codes, operator state/reason code tables, validated-jump limits, shared marker widths, pair-pass margins, topic/frame defaults, and planner launch specs.
- `PlannerIdentity` in `planner_constants.py`: gives each tracked-cone node its node name, planner mode, diagnostics prefix, and diagnostics topic.
- `TrackedConePlanningFrame` in `planner_constants.py`: packages transformed points, colors, confidences, boundary hints, track IDs, track states, and track confidences for core planner calls.

## Parameter Contract And Controllers

- `PUBLIC_TRACKED_CONE_PLANNER_DEFAULTS` in `tracked_cone_planner_contract.py`: shared public parameter surface for tracked-cone planners.
- `declare_tracked_cone_planner_parameters` in `tracked_cone_planner_contract.py`: declares shared parameters on each tracked-cone node.
- `read_migrated_tracked_cone_planner_common_config` in `tracked_cone_planner_contract.py`: reads shared runtime, filtering, controller, speed, debug, and midline-memory settings.
- `planner_algorithm_profile` in `tracked_cone_planner_contract.py`: converts public grouped parameters into algorithm profile values and applies compatibility aliases from `planner.*`.
- `apply_common_config_to_node` in `tracked_cone_planner_contract.py`: writes common config fields onto the node instance.
- `build_stanley_config`, `build_pure_pursuit_config`, and `build_steering_controller` in `tracked_cone_planner_contract.py`: read controller parameter groups and create the selected controller.

## Shared Tracked-Cone Runtime

- `GenericTrackedConePlannerNode` in `tracked_cone_planner_node.py`: shared tracked-cone node skeleton. It declares common parameters, reads common config, calls planner-specific parameter readers, builds the core config, initializes state, and wires ROS interfaces.
- `TrackedConePlannerBase` in `tracked_cone_planner_base.py`: common ROS-facing runtime used by midpoint, single-boundary, and corridor.
- `TrackedConePlannerBase._init_common_ros_interfaces`: creates publishers for `/cmd`, centerline path, RViz markers, optional point arrays, and diagnostics; subscribes to tracked cones and odometry; starts the planning timer.
- `TrackedConePlannerBase._resolve_cone_planning_context`: shared timer preamble. It handles waiting for cones, resolving frames and vehicle pose, transforming cones, fallback to odom, and empty-cycle publication on failure.
- `TrackedConePlannerBase._publish_outputs`: publishes the path, optional point array, and RViz markers.
- `TrackedConePlannerBase._publish_diagnostics`: emits the diagnostic array with planner, control, hold, and operator-state metrics.
- `TrackedConePlannerBase._build_steering_controller`: delegates to the controller factory in the parameter contract.

## Geometry Helpers

`tracked_cone_planner_geometry.py` is the single canonical home for all pure geometry and algorithm helpers shared across the planning package. There are no duplicate copies of any function or dataclass — all three planner cores and the runtime import from here.

### Shared Dataclasses

- `BoundaryChainData`: the canonical ordered side-boundary chain used by all three planner cores. Fields: `filtered_indices`, `global_points`, `local_points`, `tangents_local`, `mean_heading_change_rad`, `forward_extent_m`, `rejected_reasons_by_filtered_index`. Produced by `build_boundary_chain_data`; the empty sentinel is `empty_boundary_chain_data`.
- `FilteredCones`: the standard cone set after range/confidence/color filtering and deterministic ordering. Fields: `points` (global), `local`, `track_ids`, `colors`, `colored_count`, `raw_colors` (optional, default empty). All three planner cores use this as the input to their main algorithm.
- `BoundaryPair`: one matched left/right cone pair. Fields: `left_filtered_idx`, `right_filtered_idx`, `left_track_id`, `right_track_id`, `left_global`, `right_global`, `left_local`, `right_local`, `width_m`. Computed properties: `midpoint_global` and `midpoint_local` return `0.5 * (left + right)`. Used by midpoint and single-boundary planner cores.
- `NearFieldMetrics`: frozen near-field path-delta measurement. Fields: `lateral_max_m`, `lateral_mean_m`, `displacement_max_m`, `displacement_mean_m` (all default 0.0). Supports `__getitem__` for string-key access used by diagnostics. Produced by `near_field_delta_metrics`.
- `BoundaryChainGrowth`: intermediate result of `grow_boundary_chain_positions`. Fields: `chain_positions`, `heading_changes`, `rejected_reasons_by_filtered_index`.
- `UnknownPartnerCheck`: frozen result of `unknown_partner_check`. Fields: `longitudinal_error_m`, `width_error_m`, `radial_error_m`; `.cost` property sums them.

### Shared Algorithm Functions

- `geometry_filter(local_points, config, *, planning_horizon_m=None, max_lateral_range_m=None)`: base cone range filter. Core mask: finiteness, max range, behind-drop. Optional kwargs add corridor's planning-horizon and lateral-range gates. Corridor calls it with both kwargs via a lambda; midpoint and single-boundary call it without.
- `expected_width_m(prior, config)`: reads `prior.previous_width_m`, falls back to `config.initial_width_m` if None, clamps to `[config.min_width_m, config.max_width_m]`. Duck-typed: any prior with `previous_width_m` and config with width bounds works.
- `near_field_delta_metrics(*, current, previous, vehicle_xy, vehicle_yaw, horizon_m)` → `NearFieldMetrics`: transforms both paths into vehicle frame, resamples both to 0.25 m resolution up to `horizon_m`, computes element-wise lateral and displacement deltas. Returns zero-valued `NearFieldMetrics` when previous is None or either path is too short.
- `local_forward_prefix(path_local, *, horizon_m)` → `np.ndarray`: extracts the forward portion of an already-local-frame path. Drops points with `x < -0.1 m`, resamples at 0.25 m up to `horizon_m`. Used by corridor's `_path_alignment_metrics` and single-boundary's offset path comparison.
- `validate_path(centerline, centerline_local, near_field, heading_delta_max, continuity_threshold_m, reject_counts, config)` → `str`: shared validation sequence used by midpoint and single-boundary cores. Checks: minimum point count, finite geometry, forward extent, start-heading error, near-field lateral jump, heading delta, and self-intersection. Returns the rejection reason string or `""` on success. Increments `reject_counts["near_field_continuity"]` and `reject_counts["midpoint_kink"]` as appropriate. Corridor has its own `_validate_corridor_path` because it has additional corridor-specific checks.
- `pair_segments(pairs)` → `np.ndarray` of shape `(N, 2, 2)`: builds an array of `[left_global, right_global]` endpoint pairs from a list of `BoundaryPair` objects. Used by midpoint and single-boundary to pass pair-edge geometry to the visualization mixin.
- `update_track_width_estimate(previous_width_m, measured_width_m, config)`: clamps prior, clamps measured, limits per-update delta, blends by `width_filter_alpha`, clamps again.

### Chain Building

- `build_boundary_chain_data(*, filtered_points, filtered_local, side_indices, config, collect_rejection_reasons, min_step_m)` → `BoundaryChainData`: top-level boundary chain builder. Selects a seed with `select_seed_index`, calls `grow_boundary_chain_positions`, and materializes with `materialize_boundary_chain_data`.
- `grow_boundary_chain_positions(*, side_local, config, seed_pos, ...)` → `BoundaryChainGrowth`: greedy chain growth using step-distance, forward-projection, heading-change, and shadow-rejection gates.
- `candidate_is_shadowed(current_local, candidate_pos, side_local, remaining)`: rejects a candidate hidden behind a closer cone along nearly the same ray.
- `select_seed_index(side_local)`: selects the first chain cone closest to the vehicle and ahead of it.

### Other Geometry Helpers

- `to_vehicle_frame(points, vehicle_xy, vehicle_yaw)`: converts global cone/path coordinates into vehicle-local coordinates. `x_local = cos(yaw)*dx + sin(yaw)*dy`.
- `estimate_tangents(points)`: endpoint differences at ends, centered differences inside; produces a tangent vector per point.
- `inward_normal(tangent, color)`: rotates the tangent 90° toward the track interior based on cone side color.
- `inward_distance(anchor_local, candidate_local, inward_normal)`: signed projection of the candidate onto the anchor's inward normal; used to check partner placement.
- `pair_width_in_range(width_m, config)` and `width_jump_exceeds(new_m, last_m, max_jump_m)`: gate helpers for pair acceptance.
- `midpoint_outside_pair_span(midpoint_local, left_local, right_local)`: checks if the computed midpoint lies outside the physical extent of the pair.
- `unknown_partner_check(anchor_local, candidate_local, inward_normal, expected_width_m)` → `UnknownPartnerCheck`: measures how far an unknown-color cone deviates from the expected partner location.
- `unknown_partner_within_limits(check, config)`: returns True when the unknown check's error fields are all within config limits.
- `path_heading_delta_max(points)`, `path_curvature_abs_max(points)`, `path_self_intersects(points)`, `segments_intersect(a1, a2, b1, b2)`: path shape validation primitives.
- `moving_average(points, window)`: uniform local smoothing applied before finalization.
- `path_cumulative_lengths(points)`: cumulative arc-length array; base for all resampling.
- `sample_path_at_lengths(points, cum_lengths, samples)`, `extract_forward_path_from_pose`, `resample_to_count`: path-memory and control-path sampling helpers used by midline memory and the node timer.
- `splice_frozen_near_field`, `path_forward_extent_local`: used by midline memory for near-field freezing and extent checks.
- `project_point_to_path_s(path, cum_lengths, point_xy)`: arc-length projection used by midline memory and ground-truth helpers.

## Planner Cores

Each core is a pure Python module with no ROS imports. It receives a `TrackedConePlanningFrame` and returns a typed result dataclass. The node timer in `tracked_cone_planner_node.py` calls the core, then handles state, memory, control, and publishing.

All three cores share the following from `tracked_cone_planner_geometry.py`:
- `FilteredCones`, `BoundaryChainData`, `BoundaryPair`, `NearFieldMetrics` dataclasses
- `geometry_filter`, `expected_width_m`, `near_field_delta_metrics`, `validate_path`, `pair_segments`
- All chain building, tangent, inward-normal, and path-primitive helpers

### Midpoint Planner Core (`midpoint_planner_core.py`)

**What it does:** pairs left and right boundary cones, computes the midpoint of each accepted pair, orders those midpoints forward, and finalizes into a centerline. Strongest when both boundary colors are visible.

**Config — `MidpointPlannerConfig(BasePlannerConfig)`:**
- Pairing: `allow_unknown_pair_completion`, `unknown_pair_search_radius_m`, `unknown_pair_max_longitudinal_error_m`, `unknown_pair_max_width_error_m`, `max_consecutive_unknown_pairs`
- Chain: `max_step_m` (10 m), `min_chain_length` (3)
- Pair gates: `min_pair_width_m` (2.2 m), `max_pair_width_m` (5.5 m), `max_width_jump_m` (0.8 m), `min_pair_count` (3), `pair_reassignment_margin` (0.25), `pair_inward_projection_tolerance_m` (0.15), `pairing_tangent_neighbor_count` (4), `enforce_opposite_color_pairing`, `enforce_geometry_pairing_gate`
- Chain ordering: `midpoint_order_reference_handoff_m` (6 m), `midpoint_order_history_size` (3), `midpoint_order_backtrack_tolerance_m` (0.35 m)
- Validation: `min_path_points` (4), `min_forward_extent_m` (2.0 m), `max_near_field_lateral_jump_m` (0.6 m), `max_near_field_lateral_jump_m_sparse_pairs` (0.9 m), `max_start_heading_error_rad` (1.0 rad), `max_heading_delta_rad` (0.75 rad)
- Inherited from `BasePlannerConfig`: `max_cone_range_m`, `behind_drop_m`, `min_confidence`, `path_resolution_m`, `max_path_length_m`, `initial_width_m`, `min_width_m`, `max_width_m`, `width_filter_alpha`, `max_width_delta_per_update_m`, `jump_check_horizon_m`. `smoothing_window` is defined on `MidpointPlannerConfig`.

**Internal dataclasses:**
- `_BoundaryIndices`: `left`, `right`, `unknown` index arrays after cone color separation.
- `_PairingIndices`: `left`, `right` index arrays passed to pair building; `use_legacy_side_gate` flag.
- `_PairAnchor`: one anchor cone's full context — `filtered_idx`, `track_id`, `global_point`, `local_point`, `tangent`, `raw_color`, `inward_normal`.
- `_PairCandidate`: scored candidate pair — anchor and partner geometries, `width_m`, `cost`, `selection_cost`, `sort_key`.
- `_PairingOutcome`: `pairs` (list of `BoundaryPair`), `candidate_count`, `unknown_pair_count`, `reject_counts`.
- `_MidpointPairingResult`: `candidate_edges`, `pairs`, `midpoint_chain`, `selected_edges`, `selected_pair_track_ids`, `planner_mode`.

**Key functions:**
- `compute_midpoint_centerline(frame, config, prior)` → `MidpointPlannerResult`: top-level entry point. Calls filtering → chain building → pairing → ordering → validation.
- `_pair_boundary_chains(...)`: resolves pairing indices, generates real and unknown candidate partners, scores them, and calls `_select_boundary_pairs`.
- `_real_pair_candidate(anchor, candidate_idx, cones, config, expected_width)` → `_PairCandidate | None`: checks width in range, inward projection, raw color compatibility, midpoint span.
- `_unknown_pair_candidate(anchor, candidate_idx, cones, config, expected_width)` → `_PairCandidate | None`: compares unknown partner to expected location from inward normal and width using `unknown_partner_check`.
- `_pair_candidate_selection_key(candidate, anchor_local)` → `tuple`: sorts candidates by cost, unknown penalty, width, range, forward progress, lateral magnitude, track IDs. Deterministic tie-breaking.
- `_select_boundary_pairs(candidates_by_anchor, ...)`: greedy non-overlapping pair selection — each filtered index used at most once across the full pair set.
- `_order_pairs_into_midpoint_chain(pairs, config, prior_chain)`: orders accepted pairs by a rolling local progress reference; applies `midpoint_order_reference_handoff_m` for smooth handoff and penalizes backward progress and width jumps.
- `_validate_midpoint_centerline(centerline, centerline_local, near_field, ...)`: calls shared `validate_path`; returns accepted path or rejection reason.

**Result — `MidpointPlannerResult`:**
`filtered_points`, `filtered_colors`, `candidate_edges`, `selected_edges`, `selected_pair_track_ids`, `midpoints_raw`, `centerline`, `prevalidation_centerline`, `left_boundary`, `right_boundary`, `used_fallback`, `status`, and diagnostic counters: `candidate_count`, `selected_chain_length`, `selected_chain_width_median`, `expected_width_prior_m`, `near_field_lateral_max_m`, `near_field_lateral_mean_m`, `near_field_displacement_max_m`, `near_field_displacement_mean_m`, `near_field_kink_max_rad`, `seed_midpoint_distance_m`, `seed_temporal_offset_m`, `reject_reason`, `reject_counts`, `planner_mode`, `accepted_pair_count`, `left_chain_length`, `right_chain_length`, `filtered_track_width_m`, `unknown_pair_count`.

---

### Single-Boundary Planner Core (`single_boundary_planner_core.py`)

**What it does:** attempts pairing like midpoint first; when pairing fails or yields too little coverage, falls back to offsetting the best single boundary inward by half the estimated track width. Can keep the car moving when one side is sparse or occluded.

**Config — `SingleBoundaryPlannerConfig(BasePlannerConfig)`:**
- Mirrors midpoint pairing and chain fields; lower validation thresholds: `min_path_points` (2), `min_forward_extent_m` (1.0 m).
- Additional: `max_near_field_lateral_jump_m_single_boundary` (5.0 m) — the offset fallback is allowed a much larger near-field jump than the pair-based path because offset geometry is naturally less stable.
- Near-field horizon is capped at `_MAX_NEAR_FIELD_ALIGNMENT_HORIZON_M = 3.0 m` at the call site (independent of `jump_check_horizon_m`).

**Internal dataclasses:**
- `_BoundaryChains`: `left` (`BoundaryChainData`), `right` (`BoundaryChainData`), `unknown_indices`.
- `_PairingAnchor`: `anchor_chain`, `other_chain`, `anchor_side` string. Holds which boundary is the anchor for the current pairing attempt.
- `_AnchorContext`: per-anchor-point geometry — `local`, `global_point`, `tangent`, `filtered_idx`, `track_id`, `inward_normal`.
- `_PartnerOption`: scored partner candidate — `use_unknown`, `other_pos`, partner geometries, `width_m`, `cost`, `sort_key`.
- `_PairingState`: mutable state threaded through the pairing loop — `pairs`, `next_other_start`, `last_width`, `last_partner_progress`, `used_unknown_indices`, `consecutive_unknown_pairs`, `candidate_count`, `unknown_pair_count`.
- `_PairingResult`: `pairs`, `candidate_count`, `unknown_pair_count`, `measured_width_m`, `reject_counts`.
- `_PairingSearch`: frozen snapshot of pairing context passed to inner helpers to avoid long argument lists.

**Key functions:**
- `compute_single_boundary_centerline(frame, config, prior)` → `SingleBoundaryPlannerResult`: tries pair-based path first; on failure, tries single-boundary offset fallback.
- `_select_pairing_anchor(chains)` → `_PairingAnchor`: chooses the boundary with longer forward extent as the anchor; the shorter boundary is the partner side.
- `_real_partner_option(search, anchor_ctx, other_pos)` and `_unknown_partner_option(search, anchor_ctx, other_pos)` → `_PartnerOption | None`: score real/unknown opposite candidates with longitudinal offset, width error, and expected-partner distance.
- `_prefer_previous_partner_option(options, current_option, preferred_partner_track_id, reassignment_margin)`: biases toward the prior-frame partner track ID within `pair_reassignment_margin` to reduce unnecessary pair switching.
- `_select_fallback_chain(chains, cones, config)`: picks the better single boundary — larger forward extent, more points, lower mean heading change.
- `_offset_boundary_chain(chain, expected_width_m)`: offsets boundary points inward by `_TRACK_HALF_WIDTH_SCALE * expected_width` (0.5×) along the tangent-derived normal; produces a candidate offset centerline.
- Near-field comparison for the fallback path uses `max_near_field_lateral_jump_m_single_boundary` as the continuity threshold, allowing the offset path more lateral variance.

**Result — `SingleBoundaryPlannerResult`:**
Same shape as `MidpointPlannerResult` plus `active_boundary_side` string (which boundary was used as anchor or fallback).

---

### Corridor Planner Core (`corridor_planner_core.py`)

**What it does:** builds a drivable corridor between two boundary chains by resampling them to common arc-length stations, checking each left/right station pair for valid width, finding the longest valid corridor slice, computing corridor anchor midpoints, fitting a smooth centerline through them, and rejecting the path if it leaves the corridor. The strictest of the three planners.

**Config — `CorridorPlannerConfig(BasePlannerConfig)`:**
- Extra filter gates (used in `geometry_filter` lambda): `planning_horizon_m` (25 m), `max_lateral_range_m` (8 m).
- Chain: `max_step_m` (6 m), `max_heading_change_rad` (2.35 rad, wider than midpoint/SB), `min_chain_length` (3). Chain growth uses `CORRIDOR_CHAIN_MIN_STEP_M = 0.35 m` as a minimum step to prevent duplicate stations.
- Corridor construction: `boundary_resample_dx` (0.5 m), `min_corridor_width_m` (2.2 m), `max_corridor_width_m` (8 m), `min_required_corridor_samples` (5), `path_fit_smoothing_window` (5), `membership_margin_m` (0.15 m).
- Validation: `min_path_points` (4), `min_forward_extent_m` (2.0 m), `max_near_field_lateral_jump_m` (0.8 m), `max_heading_delta_rad` (0.75 rad), `max_initial_heading_error_rad` (3π/4 rad), `max_curvature` (0.45 m⁻¹).
- Module-level curvature constants: `CURVATURE_DENSE_SAMPLE_RELAXATION` (1.75×), `CURVATURE_SHARP_TURN_HEADING_DELTA_RAD` (0.45 rad), `CURVATURE_SHARP_TURN_RELAXATION` (1.75×), `CURVATURE_MODERATE_TURN_HEADING_DELTA_RAD` (0.28 rad), `CURVATURE_MODERATE_TURN_RELAXATION` (1.35×), `CURVATURE_SHALLOW_TURN_HEADING_DELTA_RAD` (0.18 rad), `CURVATURE_SHALLOW_TURN_RELAXATION` (1.15×).

**Internal dataclasses:**
- `_BoundaryChains`: `left` (`BoundaryChainData`), `right` (`BoundaryChainData`). No unknown_indices; corridor does not do unknown completion.
- `_CorridorCandidate`: full corridor candidate — `left_local`, `right_local`, `widths_m`, `anchors_local`, `centerline_local`, `width_std_m`, `width_range_m`, `centerline_curvature_abs_max_1pm`, `centerline_heading_delta_max_rad`, `prior_lateral_mean_m`, `prior_lateral_max_m`, `prior_heading_delta_rad`.
- `_CorridorCandidateParts`: valid-slice geometry — `left_valid`, `right_valid`, `widths_valid`, `anchors_local`, `centerline_local`.
- `_BuiltCorridor`: global-frame visualization geometry — `anchors_global`, `widths_m`, `left_global`, `right_global`, `centerline_global`, `rungs_global`.
- `_PathDeltaMetrics`: like `NearFieldMetrics` but adds `heading_delta_rad`. Intentionally kept as a private type because `heading_delta_rad` is used in prior-alignment scoring inside corridor and is not needed by midpoint or single-boundary.
- `_CorridorPathMetrics`: assembled path metrics bundle — `centerline`, `centerline_local`, `prevalidation_centerline`, `near_field` (`_PathDeltaMetrics`), `heading_delta_max_rad`, `curvature_max_1pm`, `seed_distance_m`.
- `_CorridorInputs`: filtered cones (`FilteredCones`) + chains (`_BoundaryChains`) + `reject_counts`. Passed to the main corridor construction function.

**Key functions:**
- `compute_corridor_centerline(frame, config, prior)` → `CorridorPlannerResult`: top-level entry point.
- `_make_boundary_chains(inputs, config)`: calls `_build_boundary_chain` for each side with `min_step_m=CORRIDOR_CHAIN_MIN_STEP_M`.
- `_resample_boundary_by_station(chain, dx)` → `np.ndarray`: resamples a `BoundaryChainData.local_points` by arc length at spacing `dx`. Zero or near-zero total lengths get a single-point fallback.
- `_corridor_valid_mask(left_rs, right_rs, config)` → `np.ndarray[bool]`: station-wise validity — rejects stations with non-finite coords, out-of-range width, behind-car x, or beyond `planning_horizon_m`.
- `_fill_small_invalid_gaps(mask)` and `_longest_valid_slice(mask)` → `np.ndarray[bool]`: bridges gaps up to `MAX_CORRIDOR_GAP_FILL_SAMPLES = 1` sample, then returns the longest contiguous valid run.
- `_corridor_candidate_parts(left_rs, right_rs, valid_mask, config)` → `_CorridorCandidateParts | None`: computes `anchors = 0.5 * (left + right)` at valid stations; returns None if fewer than `min_required_corridor_samples`.
- `_fit_centerline_from_anchors(parts, prior_candidate, config)` → `_CorridorCandidate | None`: moving-average smoothing with `path_fit_smoothing_window`; adapts curvature limit based on observed heading delta (sharp/moderate/shallow/dense constants); keeps candidate with lowest curvature among trials.
- `_path_violates_corridor(centerline_local, parts, config)` → `bool`: resamples the centerline to corridor station count with `resample_to_count`, checks deviation against `0.5 * width + membership_margin_m` at each station.
- `_validate_corridor_path(candidate, prior, config, reject_counts)` → `str`: corridor-specific validation — sample count, finite geometry, forward extent, corridor membership, initial heading, near-field continuity, heading delta, curvature, self-intersection. Returns rejection reason or `""`.
- `_near_field_delta_metrics(current, previous, vehicle_xy, vehicle_yaw)` → `_PathDeltaMetrics`: corridor's own near-field function; uses `NEAR_FIELD_ALIGNMENT_HORIZON_M = 3.0 m` and `PRIOR_ALIGNMENT_HORIZON_M = 4.0 m` for the two comparison horizons and returns the extra `heading_delta_rad` field used in prior scoring. This is the only near-field function not replaced by the shared `near_field_delta_metrics` in geometry.

**Result — `CorridorPlannerResult`:**
Same common fields as the other planners plus corridor-specific fields: `prevalidation_centerline`, `raw_left_chain_points`, `raw_right_chain_points`, `used_left_track_ids`, `used_right_track_ids`, `chain_rejection_reasons_by_track_id`, `corridor_width_min_m`, `corridor_width_max_m`.

---

### How The Three Planner Cores Differ

| Property | Midpoint | Single-Boundary | Corridor |
| --- | --- | --- | --- |
| Both boundaries required | yes (pairs both sides) | no (offsets one side) | yes (corridor spans both) |
| Unknown cone completion | yes | yes | no |
| Pair memory (track ID reuse) | yes | yes | no |
| Width estimate updates | yes | yes (from pairs) | yes |
| Near-field horizon cap | none (8 m default) | capped to 3 m | 3 m / 4 m (two horizons) |
| Geometry filter extras | none | none | planning_horizon + lateral_range |
| Own near-field type | `NearFieldMetrics` (shared) | `NearFieldMetrics` (shared) | `_PathDeltaMetrics` (private, adds heading_delta_rad) |
| Validate path function | `validate_path` (shared) | `validate_path` (shared) | `_validate_corridor_path` (private) |
| `min_path_points` default | 4 | 2 | 4 |
| `min_forward_extent_m` default | 2.0 m | 1.0 m | 2.0 m |

## Planner Node Classes

- `MidpointPlannerNode` in `tracked_cone_planner_node.py`: declares midpoint-specific filtering, pairing, width, centerline, validation, and debug parameters. Its `_on_timer` builds the planning frame, calls `compute_midpoint_centerline`, updates pair memory and width estimate, updates midline memory, runs control through `_run_controller_and_state` and `_dispatch_controller`, and publishes outputs.
- `SingleBoundaryPlannerNode` in `tracked_cone_planner_node.py`: declares one-boundary, pairing, offset, width, centerline, and validation parameters. Its `_on_timer` calls `compute_single_boundary_centerline`, manages one-boundary pair memory, selects fresh or held paths, runs control through `_run_controller` and `_dispatch_controller`, and publishes debug offset geometry.
- `CorridorPlannerNode` in `tracked_cone_planner_node.py`: declares corridor width, resampling, membership, fitting, pair-memory, and validation parameters. Its `_on_timer` calls `compute_corridor_centerline`, merges live and remembered corridor pair geometry, selects the publishable path, runs control through `_run_controller` and `_dispatch_controller`, and publishes corridor audit markers.
- `main_midpoint`, `main_single_boundary`, and `main_corridor` in `tracked_cone_planner_node.py`: console-script entry points used by the installed executables.

## State, Diagnostics, And Visualization

- `CommittedMidlineMemory` in `midline_memory.py`: stores a committed forward path, blends accepted candidates into it, rejects near-field jumps, and serves held paths while memory is still valid.
- `StateMachineMixin` in `planning_state_machine.py`: owns `waiting`, `fresh`, `held`, and `stopped` behavior, no-path command handling, last-valid hold, hold hysteresis, operator reason labels, and state/reason codes.
- `DiagnosticsMixin` in `planning_diagnostics.py`: formats operator status text and publishes diagnostic metrics for path quality, hold behavior, selected cones, controller output, and failure reasons.
- `VisualizationMixin` in `planning_visualization.py`: builds RViz marker arrays for remembered cones, graph edges, boundaries, pairs, raw paths, final centerline, lookahead target, and status text.

## Linetest Planner

- `LineTestPlannerNode` in `linetest_planner_node.py`: standalone fixed-line planner for controller testing. It declares and reads its own parameters, subscribes to odometry, generates a fixed path or GT-derived path, converts the forward segment into the vehicle frame, runs the shared steering-controller factory, publishes `/cmd`, and handles end-of-line braking.
- `LineTestPlannerNode._on_timer`: main loop for line projection, control-path extraction, controller execution, diagnostics, path publication, and brake publication.
- `LineTestPlannerNode._build_steering_controller`: uses `build_steering_controller` from `tracked_cone_planner_contract.py`.

## Skidpad Router

- `SkidpadRouterNode` in `skidpad_router_node.py`: ROS node that subscribes to `/tracked_cones` and `/sim/odom`, filters cones to the active skidpad or acceleration branch, republishes routed cones, publishes route diagnostics, publishes route markers, and can publish stop/brake commands.
- `SkidpadStateMachine` in `skidpad_router_core.py`: mission-stage state machine for approach, right/left lobes, straight, and parked states.
- `SkidpadStateMachine.route_mask`: masks cone points to the active route branch.
- `detect_stop_line_pair` and `detect_acceleration_stop_row` in `skidpad_router_core.py`: find parking/acceleration stop geometry.
- `boundary_color_from_lateral_y` in `skidpad_router_core.py`: assigns planning-side boundary colors from lateral position for routed cones.

## Where To Edit

- Add a shared tracked-cone parameter: `tracked_cone_planner_contract.py`.
- Add a planner-specific parameter: the relevant class in `tracked_cone_planner_node.py`.
- Change midpoint/single-boundary/corridor algorithm behavior: the matching `*_planner_core.py`.
- Change a helper used by two or more planner cores (geometry, chain building, near-field, validation): `tracked_cone_planner_geometry.py`.
- Change shared geometry or path validation primitives: `tracked_cone_planner_geometry.py`.
- Change hold behavior or operator states: `planning_state_machine.py` and `planner_constants.py`.
- Change diagnostics fields: `planning_diagnostics.py`.
- Change RViz marker content: `planning_visualization.py`.
- Change launch planner names, allowed tracks, or default diagnostics topics: `planner_constants.py` and `full_sim_launch.launch.py`.
- Change controller parameter loading: `tracked_cone_planner_contract.py`.
- Change fixed-line controller testing behavior: `linetest_planner_node.py`.
- Change skidpad or acceleration cone routing: `skidpad_router_node.py` and `skidpad_router_core.py`.
