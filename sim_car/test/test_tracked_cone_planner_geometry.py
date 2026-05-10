from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys

import numpy as np

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.planning.tracked_cone_planner_geometry import (  # noqa: E402
    build_boundary_chain_data,
    candidate_is_shadowed,
    candidate_progresses_from_vehicle,
    inward_normal,
    pair_width_in_range,
    unknown_partner_check,
    unknown_partner_within_limits,
    update_track_width_estimate,
    width_jump_exceeds,
)


@dataclass
class _ChainConfig:
    min_step_m: float = 0.8
    max_step_m: float = 6.0
    max_heading_change_rad: float = 1.0
    min_forward_progress_m: float = 0.2


@dataclass
class _WidthConfig:
    initial_width_m: float = 3.6
    min_width_m: float = 2.4
    max_width_m: float = 4.8
    width_filter_alpha: float = 0.15
    max_width_delta_per_update_m: float = 0.2


def test_candidate_progress_allows_forward_or_outboard_steps():
    assert candidate_progresses_from_vehicle(
        current_local=np.asarray([2.0, 1.0]),
        candidate_local=np.asarray([2.2, 1.0]),
        min_progress_m=0.2,
    )
    assert candidate_progresses_from_vehicle(
        current_local=np.asarray([2.0, 1.0]),
        candidate_local=np.asarray([1.8, 1.2]),
        min_progress_m=0.2,
    )
    assert not candidate_progresses_from_vehicle(
        current_local=np.asarray([2.0, 0.0]),
        candidate_local=np.asarray([1.5, 0.0]),
        min_progress_m=0.2,
    )


def test_candidate_is_shadowed_by_nearer_cone_on_same_ray():
    side_local = np.asarray(
        [
            [2.0, 0.6],
            [4.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert candidate_is_shadowed(
        current_local=np.asarray([0.0, 0.0], dtype=np.float64),
        candidate_pos=1,
        side_local=side_local,
        remaining=[0, 1],
    )


def test_track_width_update_is_clamped_and_rate_limited():
    updated = update_track_width_estimate(3.6, 4.5, _WidthConfig())

    assert updated == 3.63
    assert update_track_width_estimate(None, None, _WidthConfig()) == 3.6


def test_boundary_chain_data_orders_forward_boundary():
    points = np.asarray(
        [
            [2.0, 1.0],
            [4.0, 1.1],
            [6.0, 1.2],
        ],
        dtype=np.float64,
    )
    side_indices = np.asarray([0, 1, 2], dtype=np.int64)

    chain = build_boundary_chain_data(
        filtered_points=points,
        filtered_local=points,
        side_indices=side_indices,
        config=_ChainConfig(),
    )

    assert chain.filtered_indices.tolist() == [0, 1, 2]
    assert chain.local_points.shape == (3, 2)
    assert np.all(np.isfinite(chain.tangents_local))


def test_pair_predicates_cover_width_jump_and_unknown_partner_gate():
    tangent = np.asarray([1.0, 0.0], dtype=np.float64)
    normal = inward_normal(tangent, "blue")
    expected_partner = np.asarray([2.0, 0.0], dtype=np.float64) + (normal * 3.6)
    check = unknown_partner_check(
        partner_local=expected_partner + np.asarray([0.1, 0.05]),
        expected_partner_local=expected_partner,
        anchor_tangent=tangent,
        width_m=3.65,
        expected_width_m=3.6,
    )

    assert pair_width_in_range(3.6, 2.2, 5.5)
    assert width_jump_exceeds(3.6, 4.5, 0.8)
    assert unknown_partner_within_limits(
        check,
        max_longitudinal_error_m=1.5,
        max_width_error_m=0.9,
        search_radius_m=1.25,
    )
