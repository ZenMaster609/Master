from __future__ import annotations

import math
import pathlib
import sys


TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.perception.monocular_depth import estimate_axis_depth_from_bbox_height


def test_estimate_axis_depth_from_bbox_height_valid():
    depth_m = estimate_axis_depth_from_bbox_height(
        fy_px=600.0,
        cone_height_m=0.3034,
        bbox_height_px=100.0,
    )
    assert depth_m is not None
    assert math.isclose(depth_m, 1.8204, rel_tol=1e-6)


def test_estimate_axis_depth_from_bbox_height_invalid_inputs():
    assert estimate_axis_depth_from_bbox_height(0.0, 0.3034, 100.0) is None
    assert estimate_axis_depth_from_bbox_height(600.0, 0.0, 100.0) is None
    assert estimate_axis_depth_from_bbox_height(600.0, 0.3034, 0.0) is None
    assert estimate_axis_depth_from_bbox_height(float('nan'), 0.3034, 100.0) is None


def test_estimate_axis_depth_from_bbox_height_positive_offset_increases_depth():
    base = estimate_axis_depth_from_bbox_height(600.0, 0.3034, 100.0, bbox_height_offset_px=0.0)
    corrected = estimate_axis_depth_from_bbox_height(600.0, 0.3034, 100.0, bbox_height_offset_px=5.0)
    assert base is not None
    assert corrected is not None
    assert corrected > base


def test_estimate_axis_depth_from_bbox_height_invalid_effective_height():
    assert estimate_axis_depth_from_bbox_height(600.0, 0.3034, 5.0, bbox_height_offset_px=4.5) is None
    assert estimate_axis_depth_from_bbox_height(600.0, 0.3034, 5.0, bbox_height_offset_px=5.0) is None
