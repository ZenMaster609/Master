from __future__ import annotations

import math
import pathlib
import sys

import numpy as np


TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.cones.plotting.runtime import format_sample_rows
from sim_car.perception.cone_geometry import (
    deduplicate_cone_candidates,
    normalize_detection_color,
    reconstruct_cam_point_from_axis,
)
from sim_car.perception.detection_depth import apply_monocular_depth_to_detections, sample_depth_from_bbox


def test_apply_monocular_depth_uses_big_orange_height():
    detections = [{'x0': 10.0, 'y0': 20.0, 'x1': 30.0, 'y1': 120.0, 'label': 'big_orange'}]

    apply_monocular_depth_to_detections(
        detections,
        fy_px=600.0,
        cone_height_m=0.3034,
        big_cone_height_m=0.51,
        bbox_height_offset_px=0.0,
        normalize_detection_color=normalize_detection_color,
    )

    assert math.isclose(float(detections[0]['depth_m']), 3.06, rel_tol=1e-6)


def test_sample_depth_from_bbox_prefers_valid_lower_patch():
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    depth[5:9, 3:7] = 4.0

    sampled = sample_depth_from_bbox(depth, 2, 1, 8, 9)

    assert math.isclose(sampled, 4.0, rel_tol=1e-6)


def test_reconstruct_cam_point_forward_x_model():
    point = reconstruct_cam_point_from_axis(
        u=650.0,
        v=360.0,
        axis_depth=5.0,
        fx=500.0,
        fy=500.0,
        cx=640.0,
        cy=360.0,
        model='forward_x',
    )

    assert point is not None
    assert math.isclose(point[0], 5.0, rel_tol=1e-6)
    assert math.isclose(point[1], -0.1, rel_tol=1e-6)
    assert math.isclose(point[2], 0.0, rel_tol=1e-6)


def test_deduplicate_cone_candidates_merges_same_color_cluster():
    merged = deduplicate_cone_candidates(
        [
            (1.0, 1.0, 0.0, 'blue', 0.9),
            (1.1, 1.0, 0.0, 'blue', 0.6),
            (3.0, 3.0, 0.0, 'yellow', 0.8),
        ],
        dedup_radius_m=0.25,
    )

    assert len(merged) == 2
    colors = sorted(item[3] for item in merged)
    assert colors == ['blue', 'yellow']


def test_format_sample_rows_preserves_source_labels():
    payload = format_sample_rows(
        [
            ('monocular', 3.5, -0.2, 0, 1),
            ('stereo', 4.0, 0.1, 1, 1),
        ]
    )

    assert 'source,gt_range_m,error_m,predicted_class_id,ground_truth_class_id' in payload
    assert 'monocular,3.500000,-0.200000,0,1' in payload
    assert 'stereo,4.000000,0.100000,1,1' in payload
