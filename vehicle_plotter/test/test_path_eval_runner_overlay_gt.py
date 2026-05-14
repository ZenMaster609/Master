from __future__ import annotations

import pathlib
import sys

import numpy as np


TEST_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parents[1]
VEHICLE_PLOTTER_ROOT = REPO_ROOT / "vehicle_plotter"
SIM_CAR_ROOT = REPO_ROOT / "sim_car"
for package_root in (VEHICLE_PLOTTER_ROOT, SIM_CAR_ROOT):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from vehicle_plotter.nodes.path_eval_runner import PathEvalRunner


FIRST_GT_MIDLINE_XY = np.asarray(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
    dtype=np.float64,
)
FIRST_GT_LEFT_XY = np.asarray(
    [[0.0, 1.0], [1.0, 1.0]],
    dtype=np.float64,
)
FIRST_GT_RIGHT_XY = np.asarray(
    [[0.0, -1.0], [1.0, -1.0]],
    dtype=np.float64,
)
MOVED_GT_OFFSET_M = (
    100.0  # Simulates later cone movement that must not alter the overlay reference.
)
TARGET_FRAME = "odom"


def _make_runner_with_overlay_state() -> PathEvalRunner:
    runner = PathEvalRunner.__new__(PathEvalRunner)
    runner._path_eval_static_overlay_gt_midline_xy = np.empty((0, 2), dtype=np.float64)
    runner._path_eval_static_overlay_gt_left_xy = np.empty((0, 2), dtype=np.float64)
    runner._path_eval_static_overlay_gt_right_xy = np.empty((0, 2), dtype=np.float64)
    runner._path_eval_static_overlay_target_frame = ""
    runner._path_eval_last_gt_midline_xy = np.empty((0, 2), dtype=np.float64)
    runner._path_eval_last_gt_left_xy = np.empty((0, 2), dtype=np.float64)
    runner._path_eval_last_gt_right_xy = np.empty((0, 2), dtype=np.float64)
    return runner


def test_path_eval_overlay_gt_reference_freezes_first_complete_track_geometry():
    runner = _make_runner_with_overlay_state()
    moved_midline_xy = FIRST_GT_MIDLINE_XY + MOVED_GT_OFFSET_M
    moved_left_xy = FIRST_GT_LEFT_XY + MOVED_GT_OFFSET_M
    moved_right_xy = FIRST_GT_RIGHT_XY + MOVED_GT_OFFSET_M

    runner._path_tracking_eval_capture_static_overlay_gt(
        target_frame=TARGET_FRAME,
        gt_midline_xy=FIRST_GT_MIDLINE_XY,
        gt_left_xy=FIRST_GT_LEFT_XY,
        gt_right_xy=FIRST_GT_RIGHT_XY,
    )
    runner._path_tracking_eval_capture_static_overlay_gt(
        target_frame=TARGET_FRAME,
        gt_midline_xy=moved_midline_xy,
        gt_left_xy=moved_left_xy,
        gt_right_xy=moved_right_xy,
    )

    midline_xy, left_xy, right_xy = runner._path_tracking_eval_overlay_gt_reference()

    assert np.array_equal(midline_xy, FIRST_GT_MIDLINE_XY)
    assert np.array_equal(left_xy, FIRST_GT_LEFT_XY)
    assert np.array_equal(right_xy, FIRST_GT_RIGHT_XY)


def test_path_eval_overlay_gt_reference_falls_back_to_last_gt_until_complete_snapshot_exists():
    runner = _make_runner_with_overlay_state()
    runner._path_eval_last_gt_midline_xy = FIRST_GT_MIDLINE_XY
    runner._path_eval_last_gt_left_xy = FIRST_GT_LEFT_XY
    runner._path_eval_last_gt_right_xy = FIRST_GT_RIGHT_XY

    midline_xy, left_xy, right_xy = runner._path_tracking_eval_overlay_gt_reference()

    assert np.array_equal(midline_xy, FIRST_GT_MIDLINE_XY)
    assert np.array_equal(left_xy, FIRST_GT_LEFT_XY)
    assert np.array_equal(right_xy, FIRST_GT_RIGHT_XY)
