from __future__ import annotations

import math
import pathlib
import sys


TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.lidar.clustering import detect_cone_candidates, points_from_ranges


def test_points_from_ranges_filters_invalid_and_limits():
    ranges = [0.2, float('nan'), 1.0, float('inf'), 4.0]
    points = points_from_ranges(
        ranges,
        angle_min_rad=0.0,
        angle_increment_rad=0.1,
        range_min_m=0.1,
        range_max_m=30.0,
        min_detection_range_m=0.5,
        max_detection_range_m=3.0,
    )
    assert len(points) == 1
    x, y, r = points[0]
    assert math.isclose(r, 1.0, rel_tol=1e-6)
    assert math.isclose(x, math.cos(0.2), rel_tol=1e-6)
    assert math.isclose(y, math.sin(0.2), rel_tol=1e-6)


def test_detect_cone_candidates_splits_clusters_on_large_jump():
    points = [
        (1.00, 0.00, 1.00),
        (1.03, 0.02, 1.03),
        (2.00, 0.90, 2.19),
        (2.05, 0.92, 2.25),
    ]
    detections = detect_cone_candidates(
        points,
        jump_threshold_m=0.2,
        min_cluster_points=2,
        max_cluster_points=6,
        min_cluster_width_m=0.01,
        max_cluster_width_m=0.50,
        max_cluster_depth_m=0.4,
    )
    assert len(detections) == 2
    assert detections[0].point_count == 2
    assert detections[1].point_count == 2


def test_detect_cone_candidates_rejects_wall_like_cluster():
    wall_cluster = [
        (2.0, -1.0, math.hypot(2.0, 1.0)),
        (2.0, -0.5, math.hypot(2.0, 0.5)),
        (2.0, 0.0, 2.0),
        (2.0, 0.5, math.hypot(2.0, 0.5)),
        (2.0, 1.0, math.hypot(2.0, 1.0)),
    ]
    detections = detect_cone_candidates(
        wall_cluster,
        jump_threshold_m=0.6,
        min_cluster_points=2,
        max_cluster_points=10,
        min_cluster_width_m=0.01,
        max_cluster_width_m=0.45,
        max_cluster_depth_m=0.5,
    )
    assert detections == []


def test_detect_cone_candidates_accepts_narrow_cone_like_cluster():
    cluster = [
        (3.00, 0.10, math.hypot(3.00, 0.10)),
        (3.02, 0.12, math.hypot(3.02, 0.12)),
        (3.01, 0.11, math.hypot(3.01, 0.11)),
    ]
    detections = detect_cone_candidates(
        cluster,
        jump_threshold_m=0.2,
        min_cluster_points=2,
        max_cluster_points=10,
        min_cluster_width_m=0.01,
        max_cluster_width_m=0.45,
        max_cluster_depth_m=0.35,
    )
    assert len(detections) == 1
    det = detections[0]
    assert 3.0 < det.x_m < 3.02
    assert 0.10 < det.y_m < 0.12

