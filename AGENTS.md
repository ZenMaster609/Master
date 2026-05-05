
# AGENTS.md — Refactoring Rules for sim_car & vehicle_plotter

## Goal
Every file should have one responsibility. Every function should fit on screen.
Every number should have a name. No code should exist in two places.

---

## 1. No magic numbers — ever
Replace every bare numeric literal with a named constant, config field, or named variable.
Include the unit in the name. Add a one-line comment explaining why that value was chosen.

  Bad:   `if count < 3` / `cost + 0.05` / `width * 1.75` / `<= 1e-9`
  Good:  `if count < MIN_PAIR_COUNT` / `cost + UNKNOWN_CONE_COST_BIAS` / `width * WIDTH_SCALE` / `<= COLLINEAR_EPSILON`

This applies everywhere: controller thresholds, sensor noise values, plot font sizes, ROS QoS depths.

---

## 2. Constants go in one place — not copied across files
If a constant appears in more than one file, it belongs in a shared module.

  Current violations to fix:
  - `_VALIDATED_JUMP_ACCEPT_*` (4 copies across planner node files) → one location
  - `_OPERATOR_STATE_CODES` / `_OPERATOR_REASON_CODES` (corridor_planner_node + linetest_planner_node + tracked_cone_planner_runtime) → one location
  - `MSG_TRACK_STATE_*` (3 planner nodes + tracked_cone_planner_base) → one location
  - `_CENTERLINE_MARKER_WIDTH_M`, `_PAIR_PASSED_MARGIN_M` (corridor + midpoint nodes) → one location

Never define the same constant in two files. Import it from where it is defined.

---

## 3. Functions ≤ 40 lines
A function that exceeds 40 lines must be split. Name each piece after what it *means*, not how it works.

  Current violations to fix:
  - `_pair_boundary_chains()` in midpoint_planner_core.py (275 lines)
  - `compute_corridor_centerline()` (217 lines)
  - `__init__()` in logger_node.py (163 lines of parameter extraction alone)
  - `_init_common_planner_state()` in tracked_cone_planner_base.py (50+ attribute assignments)

Split by stage: `_score_candidates()`, `_select_best_pair()`, `_fill_unknown_gaps()` — not
`_pair_boundary_chains_part1()`.

---

## 4. No duplicated utility functions across files
These functions exist in multiple files with identical or near-identical implementations.
Move each to one canonical location and delete the copies:

  - `nearest_point_on_polyline` → tracked_cone_planner_geometry.py
  - `signed_cross_track_error` → tracked_cone_planner_geometry.py
  - `path_cumulative_lengths` → tracked_cone_planner_geometry.py
  - `_to_vehicle_frame` → tracked_cone_planner_geometry.py
  - `_path_curvature_abs_max`, `_path_heading_delta_max`, `_moving_average` → tracked_cone_planner_geometry.py
  - `_path_self_intersects`, `_segments_intersect` → tracked_cone_planner_geometry.py
  - Time-series CSV alignment logic → a shared utility in vehicle_plotter (used by steering_diagnostics, thesis_controller_diagnostics, stanley_debug_plots)

---

## 5. One file — one responsibility
A file that mixes concerns must be split. Acceptable single-responsibility scopes:

  - Algorithm / geometry (no ROS, no I/O)
  - ROS node wiring (subscriptions, publishers, timers — delegates to algorithms)
  - Configuration and constants
  - Diagnostics / visualization

  Current violations to fix:
  - `tracked_cone_planner_runtime.py` (2504 lines): state machine + geometry + ROS + diagnostics + visualization + controller → split into at minimum: geometry module, diagnostics module, visualization module, node module
  - `logger_node.py` (2132 lines): logging + diagnostics + path evaluation + plotting + session management → each concern gets its own class/module; the node just wires them together

---

## 6. Config dataclasses: use inheritance, not copy-paste
`CorridorPlannerConfig`, `MidpointPlannerConfig`, and `SingleBoundaryPlannerConfig` each have ~75 fields
with 30+ duplicated fields. Extract shared fields into a `BasePlannerConfig` dataclass and subclass it.

Group fields inside any config with blank lines and section comments:
`# --- Filtering ---`, `# --- Boundary Chain ---`, `# --- Validation ---`, `# --- Output ---`

Each field gets a one-line comment with its unit and why the default was chosen.

---

## 7. ROS parameter extraction: declare once, read once
The `__init__()` of a ROS node must not have 163 `declare_parameter()` calls followed by
163 `get_parameter()` calls. Use a helper that declares and reads in one pass, then builds
a typed config dataclass. The node stores the config object — not 163 instance variables.

  Bad:
    self.declare_parameter('format', 'parquet')
    ...  # 80 more declares
    self._log_format = self.get_parameter('format').value
    ...  # 80 more gets

  Good:
    cfg = declare_and_load_config(self, LogConfig)
    self._cfg = cfg

---

## 8. State variables: group by responsibility, not declaration order
A class that initializes 50+ flat instance variables in `__init__` is not readable.
Group related state into small dataclasses or named inner structs:

  Bad:   self._previous_centerline / self._last_valid_centerline / self._committed_centerline / ...
  Good:  self._path_state = PathState()  # holds previous, last_valid, committed, etc.

Each group should fit in a screen. Naming should make the group's purpose self-evident.

---

## 9. Sensor boilerplate: use a factory, not 8 identical files
`water_temp_in_node.py`, `water_temp_out_node.py`, `brake_temp_fr_node.py`, etc. are all 24-line
copies with different topic strings. Replace with a single parametric node class and one
factory function. Delete the duplicate files.

---

## 10. Consistent naming everywhere
Pick one convention per context and never mix them:

  - Config parameters with namespaces: always use dots (`boundary_chain.max_heading_change_rad`)
  - Config parameters without namespaces: always use underscores (`noise_stddev`)
  - Topic parameters: always suffix with `_topic` — no exceptions
  - Dataclass fields: no prefixes; use descriptive names (`left_boundary`, not `lb`)
  - Functions that return bool: start with `is_` or `has_` or `can_`
  - Private helpers: single underscore prefix `_helper_name`

---

## 11. Intermediate data: typed dataclasses, not dicts
Dict with string keys is unreadable and breaks autocomplete.

  Bad:   candidate = {"anchors_local": ..., "widths_m": ..., "audit_reasons": ...}
  Good:  @dataclass class CorridorCandidate: anchors_local: ... widths_m: ... audit_reasons: ...

Every bundle of data passed between functions gets a dataclass. No plain dicts for structured data.

---

## 12. Consistent failure reporting
Never silently return an empty list or None when something fails. Always surface a reason.
Use one pattern everywhere:

  Functions that can fail:  return tuple[Result | None, str]
  The str is a human-readable rejection reason, empty string on success.

Do not mix: dicts-of-rejection-reasons, reject_counts, silent empty returns, and raised exceptions
for the same class of failure.

---

## What NOT to do during refactoring
- Do not change algorithm logic while restructuring — structure first, logic separately.
- Do not introduce an abstraction unless at least two existing sites already need it.
- Do not add error handling for conditions that cannot happen inside this codebase.
- Do not add comments that explain *what* code does — only *why* a non-obvious decision was made.
- Do not leave TODO comments — fix it now or open a ticket; TODOs rot and mislead.
- Do not add backwards-compatibility shims for code you are deleting.# agent.md

This file provides guidance to Codex when working in this repository.

## User Interaction Preferences

Always include copyable command lines prefixed with `cd ~/ros2_ws &&` whenever you mention build, run, launch, source, or test steps. When a rebuild is relevant, also include the appropriate `colcon` rebuild command. When sourcing is needed, also include the appropriate `source` command (often `source install/setup.bash`). If you provide multiple commands, each line must start with `cd ~/ros2_ws &&`.

If code or ideas are copied/adapted from a specific external source, add a source citation comment at the top of the affected file and explicitly mention that source attribution in the prompt/response to the user.

## ROS2/Gazebo Process Policy
If you ask me to run something to get some data, i will always reply run it yourself, so you might aswell skip the step where you ask and just run it. The only time where i run something and report to you is if theres anything purely visual that only i can see.
You are authorized to build and run ROS2 projects in this workspace. After collecting whatever data you need, shut down any ROS2 and Gazebo processes you started.

## Workspace Summary

This is a ROS2 multi-package workspace for vehicle simulation, CAN/IMU decoding, and plotting/logging.

Core packages:
- `sim_car`: Gazebo Fortress simulation, virtual sensors, the measurement/noise layer, and Ackermann command bridge.
- `vehicle_plotter`: Aggregates sensor data into `VehicleState`, plots in real time, and logs to disk.
- `steering_gui`: RQT GUI for Ackermann commands and brake command.
- `vehicle_plotter_msgs`: `VehicleState` and `RunSession` messages.

`eufs_models` (under `eufs_remastered/`) is required by `eufs_gz_dynamics`. Other `eufs_sim` packages are currently not used by the core pipeline.

## Data Output Paths

`vehicle_plotter` uses run sessions under `multidata/` by default. A session directory contains logs, plots, and plot data.

## Common Commands

These are examples only. Remember the `cd ~/ros2_ws &&` prefix rule for any command you provide.

- Build everything:
  - `cd ~/ros2_ws && colcon build --symlink-install`
- Build a single package:
  - `cd ~/ros2_ws && colcon build --symlink-install --packages-select sim_car`
- Source the workspace:
  - `cd ~/ros2_ws && source install/setup.bash`
