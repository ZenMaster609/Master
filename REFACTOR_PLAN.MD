# Refactoring Plan: Planner Nodes and Cores

## Verdict: Is the Line Count Reasonable?

**Partially.** ~700 lines across the 6 planner files are genuine rule violations. The rest is real algorithm complexity.

The geometry extraction from the previous refactoring is **complete and clean** — `tracked_cone_planner_geometry.py` has 29 functions, zero copies remain in any planner file.

However, a second category of duplication was not addressed: **pure algorithm helpers shared across the 3 core files** that belong in `planner_utils.py`. These are identical or near-identical functions that exist in all 3 (or 2) core files.

`single_boundary_planner_node.py` is already **fully compliant** (0 violations). The cores are also individually clean (all functions ≤ 40 lines). The problem is cross-file duplication and 21 oversized functions across the 2 remaining node files.

---

## Track 1 (Higher Priority): Eliminate Duplication — Rules 4 & 15

### 1A: Move duplicate core helpers → `planner_utils.py`

**Identical — direct move:**

| Function | Present In | Est. Lines Saved |
|----------|-----------|-----------------|
| `_path_length` | 2 cores | ~5 |
| `_first_point_distance` | 3 cores | ~12 |
| `_forward_extent_m` | 3 cores | ~22 |
| `_path_start_heading_error` | 3 cores | ~18 |
| `_finalize_path` | 3 cores | ~24 |
| `_resample_path` | 3 cores | ~40 |
| `_deterministic_order` | 3 cores | ~46 |
| `_merge_reject_counts` | 2 cores | ~5 |

**Near-identical — extract with minor parameterization:**

| Function | Present In | Strategy |
|----------|-----------|---------|
| `_filter_and_order_cones` | 3 cores (~70L saved) | Extract shared `_ConeFilterConfig` parameter; 3 versions differ only in config fields read |
| `_build_boundary_chain` | 3 cores (~45L saved) | Midpoint and single_boundary are identical; corridor has one extra check — extract common logic, keep corridor check inline |
| `_empty_result` | 3 cores (~80L saved) | Each returns its own result type; create shared `_fill_empty_result_fields(result_obj)` for common NaN fields, keep type construction in each core |

**Verification after 1A:**
```bash
cd ~/ros2_ws && grep -rn "def _path_length\|def _first_point_distance\|def _forward_extent_m\|def _path_start_heading_error\|def _finalize_path\|def _resample_path\|def _deterministic_order\|def _merge_reject_counts\|def _filter_and_order_cones\|def _build_boundary_chain" src/Master/sim_car/sim_car/planning/
# Each function name must appear in exactly ONE file
```

### 1B: Move duplicate node helpers → `tracked_cone_planner_base.py`

| Function | Present In | Est. Lines Saved | Status |
|----------|-----------|-----------------|--------|
| `_build_stanley_debug_metrics` | 3 nodes | ~56 | Corridor/midpoint IDENTICAL; single_boundary slightly different — move shared version to base |
| `_buffer_and_anchor_centerline` | 2 nodes (corridor + single_boundary) | ~54 | IDENTICAL — move to base |
| `_execute_controller` | 2 nodes (corridor + single_boundary) | ~53 | IDENTICAL — move to base |
| `_apply_hold_logic` | 2 nodes (corridor + single_boundary) | ~40 | Functionally equivalent — extract shared core logic to base, keep planner-specific parts in each node |
| `_build_midline_status` | 2 nodes (corridor + single_boundary) | ~40 | Functionally equivalent — same approach |

**Verification after 1B:**
```bash
cd ~/ros2_ws && grep -rn "def _build_stanley_debug_metrics\|def _execute_controller\|def _buffer_and_anchor_centerline\|def _apply_hold_logic\|def _build_midline_status" src/Master/sim_car/sim_car/planning/
# Each must appear in exactly ONE file
```

**Expected size reduction after Track 1:**
- Each core file: −170 lines (~10%)
- Each node file: −80 lines (~5%)
- Total: ~700 lines eliminated

---

## Track 2 (Lower Priority): Fix Rule 3 Violations — Split Oversized Functions

After Track 1, re-run the AST check. Current state (before deduplication):

- `single_boundary_planner_node.py`: **0 violations** — already clean, do not touch
- `corridor_planner_core.py`, `midpoint_planner_core.py`, `single_boundary_planner_core.py`: **0 violations** — clean
- `corridor_planner_node.py`: **9 violations**
- `midpoint_planner_node.py`: **12 violations**

The `_on_timer` callbacks are **already compliant** (corridor: 49L = within ≤50 timer rule; midpoint: 36L). Do not touch them.

### corridor_planner_node.py — 9 violations

| Lines | Function | Split Strategy |
|-------|----------|---------------|
| 99 | `_candidate_transition_metrics` | Port `single_boundary_planner_node.py` pattern exactly: `_resolve_transition_paths`, `_transition_paths_to_vehicle_frame`, `_summarize_transition_delta`, `_empty_candidate_transition_metrics`. Use local `_TransitionPaths` dataclass (not shared). |
| 95 | `_read_parameters` | Delegate to `_read_runtime_parameters`, `_build_core_config`, `_filtering_config_values`, `_boundary_chain_and_width_config_values`, `_corridor_and_validation_config_values` — match `single_boundary_planner_node.py` structure |
| 86 | `_build_cone_audit_markers` | Extract `_cone_audit_clear_marker`, `_build_cone_audit_main_marker`, `_build_cone_audit_halo_marker`, `_build_cone_audit_label_marker` |
| 82 | `_sort_pair_entries_by_forward_progress` | Extract `_project_pair_midpoints_to_local`, `_select_sort_seed_index` (@staticmethod), `_greedy_chain_sort` (@staticmethod) |
| 79 | `_build_cone_audit_entries` | Extract `_build_cone_audit_context`, `_build_single_cone_audit_entry` |
| 50 | `_select_candidate_centerline` | Extract `_pick_best_candidate` (@staticmethod) |
| 49 | `_pair_entries_from_segments` | Extract `_transform_pair_segment_to_odom` |
| 49 | `_on_timer` | **Already compliant under ≤50 timer rule — do not touch** |
| 48 | `_corridor_analysis_metrics` | Extract `_corridor_analysis_fill_prefix_metrics` (@staticmethod) |

### midpoint_planner_node.py — 12 violations

| Lines | Function | Split Strategy |
|-------|----------|---------------|
| 108 | `_read_parameters` | Delegate to `_read_runtime_parameters`, `_build_core_config`, `_filtering_config_values`, `_boundary_chain_config_values`, `_pairing_and_width_config_values`, `_centerline_and_validation_config_values` |
| 91 | `_remember_pairs` | Extract `_filter_eligible_remembered_pairs`, `_transform_pair_segment_to_odom_with_ids`, `_merge_new_pairs_into_memory` |
| 73 | `_sort_pair_entries_by_forward_progress` | Extract `_compute_local_midpoints`, `_pick_chain_start` (@staticmethod), `_greedy_direction_chain` (@staticmethod) |
| 68 | `_run_planning_pipeline` | Extract `_run_controller_and_state` |
| 65 | `_pair_entries_from_segments` | Extract `_transform_pair_points_to_odom` |
| 56 | `_build_operator_status_text` | Extract `_build_reject_parts_text` |
| 51 | `_publish_cycle` | Extract `_build_publish_diagnostics_call` |
| 51 | `_resolve_publish_mode` | Extract `_apply_held_pair_geometry` |
| 49 | `_resolve_centerline` | Extract `_finalize_centerline_and_cache` |
| 46 | `_resolve_operator_state` | Extract `_apply_operator_state_overrides` |
| 45 | `_run_controller_step` | Extract `_execute_controller_compute` |
| 43 | `_select_candidate_centerline` | Extract `_pick_best_candidate` (@staticmethod) |

**Verification after Track 2:**
```bash
cd ~/ros2_ws && python3 - <<'EOF'
import ast
for fp in [
    "src/Master/sim_car/sim_car/planning/corridor_planner_node.py",
    "src/Master/sim_car/sim_car/planning/midpoint_planner_node.py",
]:
    with open(fp) as f: src = f.read()
    tree = ast.parse(src)
    bad = [(n.end_lineno - n.lineno + 1, n.name, n.lineno)
           for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and (n.end_lineno - n.lineno + 1) > 40]
    print(f"{'VIOLATIONS' if bad else 'OK'}: {fp}")
    for s, n, l in sorted(bad, reverse=True): print(f"  {s}L L{l} {n}")
EOF
```

---

## Key Constraints (Do Not Violate)

1. **`_candidate_transition_metrics` and `_read_parameters`:** `single_boundary_planner_node.py` is the reference pattern — match it exactly.
2. **`_PairMemoryEntry` stays private in each node** — corridor carries `last_valid_sec` (no track IDs), midpoint carries track IDs (no timestamp). Different structures, do not merge.
3. **`_sort_pair_entries_by_forward_progress` is NOT a duplicate** — corridor uses range-scoring tuples, midpoint uses direction-vector greedy chain. Different algorithms, different contracts.
4. **All Track 2 splits stay in the same file** — no new imports, no new files for Track 2.
5. **Do not change algorithm logic during restructuring** — structure only.
6. **Do not touch `_on_timer`** in either file — both are already compliant.

---

## What Will NOT Change (Already Clean)

- `single_boundary_planner_node.py` — 0 function-size violations
- All 3 `*_planner_core.py` files — 0 function-size violations individually
- `tracked_cone_planner_geometry.py` — geometry extraction complete, 0 duplicates

---

## Files Modified

| File | Track | Change |
|------|-------|--------|
| `planner_utils.py` | 1A | Receives ~8 moved functions from all 3 cores |
| `corridor_planner_core.py` | 1A | Remove duplicated helpers, import from planner_utils |
| `midpoint_planner_core.py` | 1A | Remove duplicated helpers, import from planner_utils |
| `single_boundary_planner_core.py` | 1A | Remove duplicated helpers, import from planner_utils |
| `tracked_cone_planner_base.py` | 1B | Receives ~5 moved methods from corridor + single_boundary nodes |
| `corridor_planner_node.py` | 1B + 2 | Remove node duplicates; split 9 oversized functions |
| `midpoint_planner_node.py` | 1B + 2 | Remove node duplicates; split 12 oversized functions |
| `single_boundary_planner_node.py` | 1B | Remove duplicated methods now in base (calls `super` instead) |
