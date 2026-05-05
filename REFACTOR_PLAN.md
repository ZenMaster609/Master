# Refactor Plan — sim_car & vehicle_plotter

---

## DONE

- **Phase 1A** — Shared constants extracted to `planner_constants.py`
- **Phase 1B** — Geometry utilities consolidated in `tracked_cone_planner_geometry.py`
- **Phase 1C** — vehicle_plotter analysis utilities → `analysis/analysis_utils.py`
- **Phase 2A** — `BasePlannerConfig` extracted to `planner_config_base.py`
- **Phase 2B** — Sensor node factory → `simple_sensor_node.py`
- **Phase 2C** — Logger config extraction → `log_config.py`
- **Phase 3 scaffolding** — Mixin files created and wired into `TrackedConePlannerRuntime`:
  - `planning_state_machine.py` (StateMachineMixin) ✓
  - `planning_diagnostics.py` (DiagnosticsMixin) ✓
  - `planning_visualization.py` (VisualizationMixin) ✓
- **Batch 3A-fix** — `_on_timer` in `tracked_cone_planner_runtime.py` refactored to 41-line orchestrator; `_publish_outputs` and `_convert_with_odom_pose_fallback` split; all new helpers ≤40 lines ✓
- **Batch 3B-fix** — `logger_node.py` `__init__` reduced to 42 lines; `_unpack_config()` and `_setup_all_subscriptions()` extracted; `_init_path_eval_state()` moved to `PathEvalRunner` mixin ✓
- **Batch 3D** — `_on_timer` in `midpoint_planner_node.py` reduced to 34 lines; `_blend_midline_samples` (68→32 lines) split into `_project_samples_to_vehicle_frame`, `_apply_lateral_clip`, `_convert_blended_local_to_odom`; all new helpers ≤ 40 lines ✓

---

## WHAT WENT WRONG IN PHASE 3

The mixin files exist and `TrackedConePlannerRuntime` inherits from all three. However,
`_on_timer` in `tracked_cone_planner_runtime.py` was **not split** — it is still ~1000+ lines
and reimplements logic inline instead of delegating to the mixin methods.

The same failure applies to the three `*_planner_node.py` files: `_on_timer` in each is
380–580 lines, mixing ROS wiring, algorithm invocation, diagnostics, and visualization.

**The failure mode to avoid in every remaining batch:**
Creating new files or methods without deleting the originals is not a refactor.
If you extract logic into a method but leave the same logic in `_on_timer`, you have
added code, not restructured it. Every extraction must be followed by deletion of the original.

---

## REMAINING WORK

Execute batches in order. Do not start the next batch until the verification command for the
current batch passes.

---

### Batch 3A-fix — Refactor `_on_timer` in `tracked_cone_planner_runtime.py`

The mixin files already have the right methods. This batch only rewrites `_on_timer`.

**What to do:**
1. Read `_on_timer` in full. Identify every distinct stage it performs.
2. For each stage that already has a corresponding method in DiagnosticsMixin,
   VisualizationMixin, or StateMachineMixin: delete the inline logic from `_on_timer`
   and replace it with a call to `self.<mixin_method>()`.
3. For stages with no existing mixin method: extract them into a new private method
   on the class, ≤ 40 lines each, named after what the stage means.
4. The final `_on_timer` must be ≤ 50 lines and read as a flat sequence of named calls.
5. Also split these oversized helpers in the same file:
   - `_publish_outputs` (currently ~73 lines) → split by concern into ≤ 40 line methods
   - `_convert_with_odom_pose_fallback` (currently ~81 lines) → split by fallback stage

**Verification — run this and confirm the output fits in 50 lines:**
```
awk '/def _on_timer/,/^    def /' sim_car/sim_car/planning/tracked_cone_planner_runtime.py | head -55
```

**Touch:** `tracked_cone_planner_runtime.py` only.
**Do not** modify any mixin file. **Do not** change any algorithm logic.

---

### Batch 3B-fix — Verify and complete `logger_node.py` split

Check whether `logger_node.py` `__init__` and its main callback are ≤ 50 lines.
If not, apply the same orchestrator rule as Batch 3A-fix.
The split files (`log_diagnostics_runner.py`, `path_eval_runner.py`, `plot_runner.py`)
may already exist — verify. If they exist but the node still reimplements their logic
inline, delete the inline copies and call the runner objects instead.

**Verification:**
```
grep -n "def __init__\|def _on_" vehicle_plotter/vehicle_plotter/nodes/logger_node.py
```
Then read each listed function. None may exceed 50 lines.

**Touch:** `logger_node.py` and any existing split files.

---

### Batch 3C — Refactor `_on_timer` in `corridor_planner_node.py`

`_on_timer` in this file is ~580 lines. Apply the same orchestrator rule as Batch 3A-fix.

**Additional rule:** `_build_markers` in `corridor_planner_node.py` must not reimplement
visualization logic already in `VisualizationMixin`. Move any unique logic into
`VisualizationMixin`, then delete `_build_markers` from the node file entirely and call
`self._build_markers()` (the mixin version) instead.

After the split, `corridor_planner_node.py` must contain only:
- Parameter declaration and reading
- ROS subscription/publisher/timer wiring
- `_on_timer` as an orchestrator ≤ 50 lines
All algorithm logic must remain in `corridor_planner_core.py`.

**Verification:**
```
awk '/def _on_timer/,/^    def /' sim_car/sim_car/planning/corridor_planner_node.py | head -55
```

**Touch:** `corridor_planner_node.py` only (plus `planning_visualization.py` if unique
marker logic needs to be moved there first).

---

### Batch 3D — Refactor `_on_timer` in `midpoint_planner_node.py`

Same rules as Batch 3C applied to `midpoint_planner_node.py`.

`_blend_midline_samples` (~70 lines) must also be split into stage methods ≤ 40 lines each,
named by what each stage means (e.g. `_compute_blend_alpha`, `_apply_lateral_clip`).

**Verification:**
```
awk '/def _on_timer/,/^    def /' sim_car/sim_car/planning/midpoint_planner_node.py | head -55
```

**Touch:** `midpoint_planner_node.py` only.

---

### Batch 3E — Refactor `_on_timer` in `single_boundary_planner_node.py`

Same rules as Batch 3C applied to `single_boundary_planner_node.py`.

`_candidate_transition_metrics` (~115 lines) must be split by stage.

**Verification:**
```
awk '/def _on_timer/,/^    def /' sim_car/sim_car/planning/single_boundary_planner_node.py | head -55
```

**Touch:** `single_boundary_planner_node.py` only.

---

### Batch 3F — Resolve `_build_markers` duplication

`tracked_cone_planner_base.py` contains a `_build_markers` method (~250+ lines) that
duplicates `VisualizationMixin._build_markers` in `planning_visualization.py`.

**What to do:**
1. Read both `_build_markers` implementations side by side.
2. Merge any logic unique to the base class version into `VisualizationMixin._build_markers`.
3. Delete `_build_markers` from `tracked_cone_planner_base.py` entirely.
4. Split `VisualizationMixin._build_markers` so no single method exceeds 40 lines.
   Split by marker type, e.g.:
   - `_append_boundary_markers`
   - `_append_path_markers`
   - `_append_cone_markers`
   - `_append_status_marker`
   `_build_markers` itself becomes the ≤ 40 line orchestrator that calls these.

**Verification:**
```
grep "def _build_markers" sim_car/sim_car/planning/tracked_cone_planner_base.py
```
Must return no output.

**Touch:** `tracked_cone_planner_base.py`, `planning_visualization.py`.

---

### Batch 3G — Eliminate per-node utility duplication

Utility functions that appear identically in multiple node files must be consolidated.

**Instance method utilities** (need `self`) — move to `tracked_cone_planner_base.py`,
delete per-file copies:
- `_lookup_transform`, `_lookup_transform_with_alias`
- `_frame_aliases`, `_is_alias`
- `_warn_throttled`

**Pure function utilities** (no `self`) — move to `tracked_cone_planner_geometry.py`,
delete per-file copies:
- `_yaw_from_quat`
- `_odom_point_to_base`, `_base_point_to_odom`
- `_transform_point`

**Algorithm-pure helpers** shared across the three `*_core.py` files — create a new file
`sim_car/sim_car/planning/planner_utils.py` and move there:
- `_clamp`
- `_default_reject_counts` (or equivalent empty-result helpers)
Delete the copies from all three core files and update their imports.

**Verification:**
```
grep -rn "def _yaw_from_quat\|def _clamp\|def _frame_aliases" sim_car/sim_car/planning/
```
Each function name must appear in exactly one file.

**Touch:** `tracked_cone_planner_base.py`, `tracked_cone_planner_geometry.py`,
`corridor_planner_node.py`, `midpoint_planner_node.py`, `single_boundary_planner_node.py`,
`linetest_planner_node.py`, `corridor_planner_core.py`, `midpoint_planner_core.py`,
`single_boundary_planner_core.py`, plus new `planner_utils.py`.

---

### Batch 4A — Algorithm cleanup: `single_boundary_planner_core.py` (~1635 lines)

**Rules:**
- Split every function > 40 lines by stage. Name each piece by what it *means*, not how
  it works. Naming like `_part1` or `_inner` is forbidden.
- Replace any remaining plain-dict bundles passed between functions with `@dataclass`.
- Unify failure reporting: every function that can fail returns `tuple[Result | None, str]`.
  The str is a human-readable rejection reason; empty string on success.
- Do NOT touch `single_boundary_planner_node.py` in this session.
- Do NOT change any algorithm logic.

**Verification:**
```
python3 - <<'EOF'
import ast, sys
src = open("sim_car/sim_car/planning/single_boundary_planner_core.py").read()
tree = ast.parse(src)
bad = [(n.name, n.end_lineno - n.lineno) for n in ast.walk(tree)
       if isinstance(n, ast.FunctionDef) and n.end_lineno - n.lineno > 40]
if bad:
    print("OVER 40 LINES:", bad); sys.exit(1)
print("OK")
EOF
```

---

### Batch 4B — Algorithm cleanup: `midpoint_planner_core.py` (~1920 lines)

Same rules as Batch 4A.

`_pair_boundary_chains` (~275 lines) must be split into named stage functions, e.g.:
- `_score_pair_candidates`
- `_select_pair_per_anchor`
- `_extend_midpoint_chain`

Do NOT touch `midpoint_planner_node.py` in this session.

**Verification:** same script as Batch 4A, applied to `midpoint_planner_core.py`.

---

### Batch 4C — Algorithm cleanup: `corridor_planner_core.py` (~1762 lines)

Same rules as Batch 4A.

`compute_corridor_centerline` must become an orchestrator ≤ 50 lines that calls named
stage functions:
- `_prepare_corridor_inputs`
- `_build_boundary_chains`
- `_score_corridor_candidates`
- `_fit_centerline`
- `_validate_corridor_path`

Do NOT touch `corridor_planner_node.py` in this session.

**Verification:** same script as Batch 4A, applied to `corridor_planner_core.py`.

---

## Sizing Guide

| Change type | Max files to touch | Max new lines to write |
|---|---|---|
| Refactor `_on_timer` (delegate to existing methods) | 1–2 files | < 100 |
| Resolve duplication across node files | 4–6 files | < 150 |
| Split a core algorithm file | 1 file | < 600 |

If a session needs more than that, the batch is too large — split it.

---

## Sequencing Rule

Complete batches in order. Do not start Batch 3C before Batch 3A-fix verification passes.
Do not start Phase 4 before all Phase 3 batches are verified.
