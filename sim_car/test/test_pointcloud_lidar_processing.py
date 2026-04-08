from __future__ import annotations

import math
import pathlib
import sys

import numpy as np


TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.lidar.pointcloud_processing import (
    PointClusterDetection,
    apply_azimuth_masks,
    apply_range_thinning,
    crop_points_to_roi,
    detect_cone_like_clusters,
    pointcloud2_to_xyz_array,
    suppress_ground_points,
    summarize_clusters_for_debug,
    summarize_rejection_reasons,
    xyz_array_to_pointcloud2,
)


def test_pointcloud2_round_trip_xyz_decoding() -> None:
    points = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    msg = xyz_array_to_pointcloud2(points, frame_id='lidar_link', stamp=None)
    decoded = pointcloud2_to_xyz_array(msg)
    assert decoded.shape == (2, 3)
    assert np.allclose(decoded, points)


def test_crop_points_to_roi_filters_xy_and_z() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.2],
            [30.0, 0.0, 0.2],
            [1.0, 15.0, 0.2],
            [1.0, 0.0, 0.01],
        ],
        dtype=np.float32,
    )
    cropped = crop_points_to_roi(
        points,
        x_min_m=-3.0,
        x_max_m=25.0,
        y_min_m=-12.0,
        y_max_m=12.0,
        z_min_m=0.03,
        z_max_m=0.80,
    )
    assert cropped.shape == (1, 3)
    assert np.allclose(cropped[0], [0.0, 0.0, 0.2])


def test_detect_cone_like_clusters_accepts_single_cone_cluster() -> None:
    cone = np.asarray(
        [
            [5.00, 0.00, 0.10],
            [5.06, 0.03, 0.18],
            [4.98, -0.03, 0.24],
            [5.02, 0.02, 0.31],
            [5.01, -0.03, 0.36],
        ],
        dtype=np.float32,
    )
    detections = detect_cone_like_clusters(
        cone,
        max_cluster_radius_m=0.20,
        min_cluster_points=3,
        max_cluster_points=80,
        min_cluster_width_m=0.05,
        max_cluster_width_m=0.60,
        min_cluster_depth_m=0.05,
        max_cluster_depth_m=0.60,
        min_cluster_height_m=0.15,
        max_cluster_height_m=0.60,
    )
    assert len(detections) == 1
    detection = detections[0]
    assert math.isclose(detection.x_m, 5.01, rel_tol=0.01)
    assert abs(detection.y_m) < 0.02
    assert detection.point_count == 5


def test_detect_cone_like_clusters_rejects_wall_like_cluster() -> None:
    wall = np.asarray(
        [
            [6.0, -1.0, 0.10],
            [6.0, -0.5, 0.12],
            [6.0, 0.0, 0.15],
            [6.0, 0.5, 0.18],
            [6.0, 1.0, 0.20],
        ],
        dtype=np.float32,
    )
    detections = detect_cone_like_clusters(
        wall,
        max_cluster_radius_m=0.30,
        min_cluster_points=3,
        max_cluster_points=80,
        min_cluster_width_m=0.05,
        max_cluster_width_m=0.60,
        min_cluster_depth_m=0.05,
        max_cluster_depth_m=0.60,
        min_cluster_height_m=0.15,
        max_cluster_height_m=0.60,
    )
    assert detections == []


def test_detect_cone_like_clusters_keeps_two_nearby_cones_separate() -> None:
    left = np.asarray(
        [
            [8.00, -0.30, 0.12],
            [8.06, -0.26, 0.20],
            [7.98, -0.33, 0.32],
        ],
        dtype=np.float32,
    )
    right = np.asarray(
        [
            [8.00, 0.30, 0.12],
            [8.06, 0.34, 0.20],
            [7.98, 0.27, 0.32],
        ],
        dtype=np.float32,
    )
    detections = detect_cone_like_clusters(
        np.vstack((left, right)),
        max_cluster_radius_m=0.20,
        min_cluster_points=3,
        max_cluster_points=80,
        min_cluster_width_m=0.05,
        max_cluster_width_m=0.60,
        min_cluster_depth_m=0.05,
        max_cluster_depth_m=0.60,
        min_cluster_height_m=0.15,
        max_cluster_height_m=0.60,
    )
    assert len(detections) == 2
    ys = sorted(detection.y_m for detection in detections)
    assert ys[0] < -0.2
    assert ys[1] > 0.2


def test_apply_range_thinning_drops_far_points_more_aggressively() -> None:
    near = np.tile(np.asarray([[5.0, 0.0, 0.2]], dtype=np.float32), (200, 1))
    far = np.tile(np.asarray([[20.0, 0.0, 0.2]], dtype=np.float32), (200, 1))
    rng = np.random.default_rng(7)
    thinned = apply_range_thinning(
        np.vstack((near, far)),
        thinning_start_range_m=12.0,
        max_range_m=25.0,
        keep_ratio_at_max_range=0.20,
        rng=rng,
    )
    near_kept = int(np.sum(np.isclose(thinned[:, 0], 5.0)))
    far_kept = int(np.sum(np.isclose(thinned[:, 0], 20.0)))
    assert near_kept > far_kept
    assert near_kept > 150


def test_apply_azimuth_masks_removes_masked_sector_points() -> None:
    points = np.asarray(
        [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.2],
            [-1.0, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    masked = apply_azimuth_masks(points, [-10.0, 10.0])
    assert masked.shape == (2, 3)
    assert not np.any(np.isclose(masked[:, 0], 1.0))


def test_empty_cloud_produces_no_detections() -> None:
    detections = detect_cone_like_clusters(
        np.empty((0, 3), dtype=np.float32),
        max_cluster_radius_m=0.20,
        min_cluster_points=3,
        max_cluster_points=80,
        min_cluster_width_m=0.05,
        max_cluster_width_m=0.60,
        min_cluster_depth_m=0.05,
        max_cluster_depth_m=0.60,
        min_cluster_height_m=0.15,
        max_cluster_height_m=0.60,
    )
    assert detections == []


def test_suppress_ground_points_rejects_flat_ground_across_range() -> None:
    points = np.asarray(
        [
            [2.0, 0.0, 0.03],
            [5.0, 0.0, 0.015],
            [10.0, 0.0, 0.03],
            [15.0, 0.0, 0.05],
        ],
        dtype=np.float32,
    )
    filtered = suppress_ground_points(
        points,
        base_cutoff_m=0.035,
        range_slope_m_per_m=0.004,
        range_bias_m=0.01,
        z_max_m=0.90,
    )
    assert filtered.shape == (0, 3)


def test_suppress_ground_points_preserves_cone_body_points() -> None:
    points = np.asarray(
        [
            [3.0, 0.0, 0.18],
            [8.0, 0.0, 0.25],
            [12.0, 0.0, 0.32],
        ],
        dtype=np.float32,
    )
    filtered = suppress_ground_points(
        points,
        base_cutoff_m=0.035,
        range_slope_m_per_m=0.004,
        range_bias_m=0.01,
        z_max_m=0.90,
    )
    assert filtered.shape == (3, 3)


def test_suppress_ground_points_keeps_cone_points_and_removes_ground_mix() -> None:
    points = np.asarray(
        [
            [4.0, 0.0, 0.02],
            [4.0, 0.1, 0.03],
            [4.1, 0.0, 0.18],
            [4.05, 0.02, 0.26],
        ],
        dtype=np.float32,
    )
    filtered = suppress_ground_points(
        points,
        base_cutoff_m=0.035,
        range_slope_m_per_m=0.004,
        range_bias_m=0.01,
        z_max_m=0.90,
    )
    assert filtered.shape == (2, 3)
    assert np.all(filtered[:, 2] > 0.1)


def test_suppress_ground_points_keeps_low_cone_body_farther_out() -> None:
    points = np.asarray(
        [
            [8.0, 0.0, 0.055],
            [12.0, 0.0, 0.06],
            [16.0, 0.0, 0.065],
        ],
        dtype=np.float32,
    )
    filtered = suppress_ground_points(
        points,
        base_cutoff_m=0.035,
        range_slope_m_per_m=0.004,
        range_bias_m=0.01,
        z_max_m=0.90,
    )
    assert filtered.shape == (3, 3)


def test_summarize_rejection_reasons_counts_rejected_clusters() -> None:
    summaries = [
        PointClusterDetection(
            x_m=6.0,
            y_m=0.0,
            z_m=0.2,
            width_m=0.8,
            depth_m=0.1,
            height_m=0.3,
            point_count=5,
            min_range_m=5.9,
            max_range_m=6.1,
            accepted=False,
            reason='too_wide',
        ),
        PointClusterDetection(
            x_m=7.0,
            y_m=0.0,
            z_m=0.2,
            width_m=0.02,
            depth_m=0.01,
            height_m=0.05,
            point_count=1,
            min_range_m=7.0,
            max_range_m=7.0,
            accepted=False,
            reason='too_few_points',
        ),
        PointClusterDetection(
            x_m=8.0,
            y_m=0.0,
            z_m=0.2,
            width_m=0.2,
            depth_m=0.1,
            height_m=0.3,
            point_count=4,
            min_range_m=8.0,
            max_range_m=8.0,
            accepted=False,
            reason='too_wide',
        ),
        PointClusterDetection(
            x_m=4.0,
            y_m=0.0,
            z_m=0.2,
            width_m=0.2,
            depth_m=0.1,
            height_m=0.3,
            point_count=4,
            min_range_m=4.0,
            max_range_m=4.0,
            accepted=True,
            reason='',
        ),
    ]
    assert summarize_rejection_reasons(summaries) == {'too_wide': 2, 'too_few_points': 1}


def test_range_adaptive_detection_accepts_sparse_mid_range_cluster() -> None:
    cone = np.asarray(
        [
            [7.98, 0.02, 0.14],
            [8.03, -0.01, 0.27],
        ],
        dtype=np.float32,
    )
    detections = detect_cone_like_clusters(
        cone,
        max_cluster_radius_m=0.30,
        min_cluster_points=3,
        max_cluster_points=80,
        min_cluster_width_m=0.05,
        max_cluster_width_m=0.60,
        min_cluster_depth_m=0.05,
        max_cluster_depth_m=0.60,
        min_cluster_height_m=0.15,
        max_cluster_height_m=0.60,
    )
    assert len(detections) == 1
    assert 7.5 < detections[0].x_m < 8.5


def test_range_adaptive_detection_rejects_single_point_sparse_return() -> None:
    cone = np.asarray([[8.0, 0.0, 0.20]], dtype=np.float32)
    detections = detect_cone_like_clusters(
        cone,
        max_cluster_radius_m=0.30,
        min_cluster_points=3,
        max_cluster_points=80,
        min_cluster_width_m=0.05,
        max_cluster_width_m=0.60,
        min_cluster_depth_m=0.05,
        max_cluster_depth_m=0.60,
        min_cluster_height_m=0.15,
        max_cluster_height_m=0.60,
    )
    assert detections == []


def test_summarize_clusters_for_debug_includes_rejected_wall() -> None:
    wall = np.asarray(
        [
            [6.0, -0.4, 0.18],
            [6.0, -0.2, 0.22],
            [6.0, 0.0, 0.24],
            [6.0, 0.2, 0.20],
            [6.0, 0.4, 0.18],
        ],
        dtype=np.float32,
    )
    summaries = summarize_clusters_for_debug(
        wall,
        max_cluster_radius_m=0.30,
        min_cluster_points=3,
        max_cluster_points=80,
        min_cluster_width_m=0.05,
        max_cluster_width_m=0.60,
        min_cluster_depth_m=0.05,
        max_cluster_depth_m=0.60,
        min_cluster_height_m=0.15,
        max_cluster_height_m=0.60,
    )
    assert len(summaries) == 1
    assert not summaries[0].accepted
    assert summaries[0].reason == 'too_wide'
