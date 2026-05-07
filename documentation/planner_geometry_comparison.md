# Planner Geometry Comparison

This page explains how the midpoint, single-boundary, and corridor planners build their planner line from tracked cones. It focuses on geometry: where the path points come from, how they are smoothed/resampled, and how the planners reject unsafe or implausible geometry.

This is not a tuning guide or launch guide. For parameters and planner selection, see [Planner Tuning](planner_tuning.md). For deeper function maps, see the pages in [Code Reference](code/README.md).

## Shared Geometry Foundation

All three migrated tracked-cone planners start with the same basic problem: convert a noisy set of visible cones into a short, forward-driving centerline in front of the car.

The common geometric stages are:

1. Convert cones into the vehicle frame, where local `x` means forward and local `y` means left/right.
2. Filter unusable cones by finite coordinates, distance, behind-car distance, confidence, and planner-specific side/color rules.
3. Sort the remaining cones deterministically so repeated frames do not depend on arbitrary array order.
4. Split cones into blue and yellow candidate boundaries.
5. Grow each boundary as a forward chain instead of connecting every nearby cone.
6. Create a raw planner line from those boundary chains.
7. Smooth and/or resample that raw line into evenly spaced path points.
8. Validate the result before it can become a fresh centerline.

The boundary-chain step is the most important shared geometry piece. In `tracked_cone_planner_geometry.py`, `build_boundary_chain_data` starts from the closest useful forward cone on one side, then repeatedly chooses the next plausible cone. A candidate step must be far enough away, not too far away, project forward along the current chain heading, avoid sharp heading changes, and avoid being shadowed by a better nearer cone.

Tangents and inward normals are also shared ideas. `estimate_tangents` estimates the local direction of a boundary chain. `inward_normal` rotates that tangent toward the track interior: blue cones offset inward toward the right side of the track, while yellow cones offset inward toward the left side.

Track-width and pair gates now share small predicate helpers too. `update_track_width_estimate` keeps the filtered width prior consistent across the planners. Pairing code in midpoint and single-boundary uses shared checks such as `pair_width_in_range`, `inward_distance`, `unknown_partner_check`, and `unknown_partner_within_limits`, while each planner still owns the control flow that decides how candidates become a path.

Shared algorithm utilities also live in `tracked_cone_planner_geometry.py`. This includes deterministic cone ordering, common filtering/order extraction, the boundary-chain wrapper used by the cores, moving-average/resampling finalization, path length/forward-extent helpers, and empty-result field filling.

Validation is conceptually shared even when exact limits differ. The planners check for enough points, enough forward extent, finite geometry, near-vehicle heading sanity, jumps from the previous accepted path, excessive kinks, and self-intersection. The path-shape primitives live in `tracked_cone_planner_geometry.py`; each core still decides which rejection thresholds apply. The corridor planner adds corridor-membership and curvature checks because it has an explicit drivable channel.

## Midpoint Planner Geometry

The midpoint planner builds the line from cross-track cone pairs:

`tracked cones -> filtered cones -> left/right boundary chains -> accepted pairs -> midpoint chain -> smoothed/resampled centerline`

In `compute_midpoint_centerline`, the planner first creates left and right chains. It then uses `_pair_boundary_chains` to find blue/yellow or blue/unknown pairs that look like valid track-width cross-sections. A pair is only useful if the width is within limits, the partner is on the inward side of the anchor boundary, the pairing is not a large width jump from nearby pairs, and color/geometry gates allow it.

Each accepted pair creates one raw midpoint:

`midpoint = 0.5 * (left_cone + right_cone)`

Those midpoints are not automatically in driving order. `_order_pairs_into_midpoint_chain` starts near the vehicle and connects nearby midpoints while discouraging backwards progress. This matters on curves and in sparse cone layouts, where the closest pair in Euclidean distance is not always the next pair along the track. `_trim_pairs_by_midpoint_step_length` then cuts the chain if a segment jumps too far.

The raw midpoint chain is finalized through the shared `_finalize_path` helper in `tracked_cone_planner_geometry.py`. It applies a moving-average smoothing window when enabled, then resamples the curve at `centerline.path_resolution_m` up to `centerline.max_path_length_m`.

Midpoint validation rejects paths that are too short, not forward enough, non-finite, heading-flipped near the vehicle, too different from the previous near-field path, too kinked, or self-crossing. If validation fails, the fresh path is rejected instead of publishing arbitrary pair geometry.

## Single-Boundary Planner Geometry

The single-boundary planner builds the line by offsetting one reliable boundary inward:

`tracked cones -> filtered cones -> best boundary chain -> inward half-width offset -> smoothed/resampled centerline`

In `compute_single_boundary_centerline`, the planner still builds both blue and yellow chains and can evaluate pair information for width estimation. The actual path, however, comes from `_select_fallback_chain`, which chooses the more reliable single boundary using forward extent, chain length, heading smoothness, and deterministic side priority.

The pair-support code is intentionally separate from path generation. `_pair_boundary_chains` gathers pair evidence for width estimation and diagnostics, while `_real_partner_option`, `_unknown_partner_option`, and `_boundary_pair_from_option` keep the individual real-cone, unknown-cone, and pair-construction checks readable.

After a boundary is selected, `_offset_boundary_chain` estimates tangents along that chain and offsets every chain point inward by half of the expected track width:

`center_point = boundary_point + inward_normal * (expected_width / 2)`

This is the key individual behavior. Midpoint and corridor planners need both sides to geometrically agree on the center. The single-boundary planner can continue when only one side is visible, but it must trust the filtered track-width prior more heavily.

Finalization is intentionally similar to midpoint planning. The shared `_finalize_path` helper smooths the raw offset path with a moving average, then resamples it at the configured path resolution and length.

Validation also mostly matches midpoint validation: enough path points, enough forward extent, finite geometry, sane starting heading, limited heading delta, near-field continuity, and no self-intersection. The important difference is the continuity threshold. Single-boundary mode allows a larger near-field lateral jump because switching to a clean one-boundary offset can look like a large lateral change even when it is the best available geometry.

## Corridor Planner Geometry

The corridor planner builds an explicit drivable channel before fitting the centerline:

`tracked cones -> filtered colored cones -> left/right boundary chains -> corridor rungs -> center anchors -> fitted/resampled centerline`

In `compute_corridor_centerline`, both boundary chains must be reliable enough. Unlike midpoint and single-boundary planning, the core corridor algorithm does not use unknown cones for pair completion. It wants colored blue/yellow boundaries that overlap as a continuous corridor.

`_build_corridor` resamples both boundary chains and searches for a valid overlap. Each accepted corridor sample creates a rung from left boundary to right boundary, a width, and a center anchor. A candidate rung must have a reasonable width and belong to the best continuous corridor slice. This makes corridor planning stricter than midpoint planning: it is not just asking whether individual pairs are plausible, but whether the pair sequence forms a consistent channel.

The centerline does not simply connect raw midpoint pairs. `_fit_centerline_from_anchors` smooths the accepted center anchors with a window controlled by `corridor.path_fit_smoothing_window`. It adapts smoothing based on heading/curvature so it does not over-flatten sharper valid geometry. The fitted line is then resampled to the normal path resolution and maximum length.

Corridor validation includes the shared checks plus two corridor-specific checks. `_path_violates_corridor` resamples the centerline against the corridor anchors and rejects it if points leave the sampled corridor plus `corridor.membership_margin_m`. The planner also checks maximum curvature, because a smooth-looking line can still bend too sharply for a stable drivable path.

## What Is Identical vs Different

| Area | Shared / identical idea | Midpoint planner | Single-boundary planner | Corridor planner |
| --- | --- | --- | --- | --- |
| Input geometry | Uses tracked cone positions, colors, confidences, vehicle pose, and previous path/width prior. | Can use unknown cones for pair completion. | Can use unknown cones for pair completion and width support. | Uses only planner-facing blue/yellow cones in the core corridor geometry. |
| Vehicle frame | Local `x` is forward and local `y` is lateral. | Same transform idea. | Same transform idea. | Same transform idea, plus stricter horizon/lateral range filtering. |
| Boundary chains | Blue and yellow cones are grown into plausible chains with step, chain-heading progress, heading-change, and shadowing gates. | Chains support pairing but raw path comes from accepted pair midpoints. | Chains support side selection; raw path comes from one selected boundary. | Both chains are required and become the corridor edges. |
| Raw line source | All planners create a raw geometric line before final resampling. | Average each accepted left/right pair. | Offset one boundary inward by half expected width. | Sample a continuous left/right corridor and use rung centers as anchors. |
| Width usage | Width limits reject implausible geometry and previous width can act as a prior. | Pair widths directly create and validate midpoints. | Expected width is critical because it sets the offset distance. | Corridor sample widths define the valid drivable channel. |
| Ordering | Output path must progress forward from near vehicle to far field. | Explicitly orders accepted midpoint pairs and trims large jumps. | Boundary chain order becomes offset path order. | Corridor samples are taken from the best continuous overlap. |
| Smoothing | Geometry can be smoothed before final publication. | Moving-average smoothing of midpoint chain, then resampling. | Moving-average smoothing of offset path, then resampling. | Smooths/fits center anchors, then resamples; smoothing adapts around curvature. |
| Validation | Rejects non-finite, too-short, low-extent, jumpy, kinked, or self-crossing paths. | Standard midpoint continuity and heading/kink checks. | Similar checks, but one-boundary mode permits larger near-field lateral jumps. | Adds corridor membership and curvature checks. |
| Failure behavior | A failed fresh geometry candidate is not blindly published. Node-level path memory/hold behavior can preserve a recent valid path. | Fails when reliable pair chain cannot be built or validation rejects it. | Can still succeed with one good boundary, but fails if no boundary chain is reliable. | Fails when either boundary or the corridor overlap is insufficient. |

## Mental Model

Use this quick distinction when reading the code:

- Midpoint asks: "Which left/right cone pairs define safe cross-track centers?"
- Single-boundary asks: "Which one boundary is trustworthy enough to offset inward?"
- Corridor asks: "Do the two boundaries form a continuous drivable channel, and does the fitted centerline stay inside it?"

The shared code builds trustworthy boundary ingredients. The individual planner logic decides what geometric object is trusted enough to become the planner line: pair midpoints, a one-sided offset, or a sampled corridor.
