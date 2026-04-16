from __future__ import annotations

import math

import numpy as np
import pytest

from sim_car.planning.midline_memory import (
    CommittedMidlineMemory,
    MidlineCandidate,
    MidlineMemoryConfig,
)


def _memory(**overrides) -> CommittedMidlineMemory:
    params = {
        "station_spacing_m": 1.0,
        "horizon_m": 8.0,
        "near_distance_m": 2.0,
        "mid_distance_m": 5.0,
        "near_alpha": 0.1,
        "mid_alpha": 0.25,
        "far_alpha": 0.5,
        "near_max_lateral_shift_m": 0.10,
        "mid_max_lateral_shift_m": 0.25,
        "far_max_lateral_shift_m": 0.50,
        "candidate_jump_reject_threshold_m": 0.8,
        "hold_last_valid_duration_s": 2.0,
        "min_buffer_confidence": 0.1,
        "candidate_min_extent_m": 1.0,
    }
    params.update(overrides)
    cfg = MidlineMemoryConfig(**params)
    return CommittedMidlineMemory(cfg)


def _candidate(
    path: np.ndarray,
    *,
    updateable: bool = True,
    support_path: np.ndarray | None = None,
    allow_estimation: bool = False,
) -> MidlineCandidate:
    return MidlineCandidate(
        centerline=np.asarray(path, dtype=np.float64),
        source="validated",
        updateable=updateable,
        update_reason="ok" if updateable else "invalid",
        support_path=support_path,
        allow_estimation=allow_estimation,
    )


def test_seeds_from_first_valid_candidate():
    mem = _memory()
    path = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)

    result = mem.update(
        candidate=_candidate(path),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.0,
    )

    assert result.candidate_accepted is True
    assert result.update_mode == "seed"
    assert result.buffer_confidence == pytest.approx(1.0)
    assert result.centerline.shape[0] >= 3
    assert np.allclose(result.centerline[:, 1], 0.0)


def test_rejects_nonfinite_candidate_without_buffer():
    mem = _memory()
    path = np.array([[0.0, 0.0], [math.nan, 0.0]], dtype=np.float64)

    result = mem.update(
        candidate=_candidate(path),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.0,
    )

    assert result.candidate_accepted is False
    assert result.update_mode == "reject"
    assert result.centerline.shape == (0, 2)


def test_near_field_update_is_clamped_instead_of_replaced():
    mem = _memory(near_alpha=1.0, near_max_lateral_shift_m=0.10, candidate_jump_reject_threshold_m=5.0)
    stored = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    candidate = np.array([[0.0, 0.5], [2.0, 0.5], [4.0, 0.5]], dtype=np.float64)
    mem.update(candidate=_candidate(stored), vehicle_xy=(0.0, 0.0), vehicle_yaw=0.0, now_sec=1.0)

    result = mem.update(
        candidate=_candidate(candidate),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.1,
    )

    assert result.update_mode == "blend"
    assert result.candidate_accepted is True
    assert result.centerline[0, 1] == pytest.approx(0.10)
    assert result.centerline[1, 1] == pytest.approx(0.10)


def test_far_field_can_move_more_than_near_field():
    mem = _memory(candidate_jump_reject_threshold_m=5.0)
    stored = np.column_stack((np.arange(0.0, 7.0, 1.0), np.zeros(7)))
    candidate = np.column_stack((np.arange(0.0, 7.0, 1.0), np.ones(7)))
    mem.update(candidate=_candidate(stored), vehicle_xy=(0.0, 0.0), vehicle_yaw=0.0, now_sec=1.0)

    result = mem.update(
        candidate=_candidate(candidate),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.1,
    )

    assert result.centerline[1, 1] < result.centerline[-1, 1]


def test_holds_previous_path_during_large_jump():
    mem = _memory(candidate_jump_reject_threshold_m=0.4)
    stored = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    candidate = np.array([[0.0, 1.2], [2.0, 1.2], [4.0, 1.2]], dtype=np.float64)
    mem.update(candidate=_candidate(stored), vehicle_xy=(0.0, 0.0), vehicle_yaw=0.0, now_sec=1.0)

    result = mem.update(
        candidate=_candidate(candidate),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.1,
    )

    assert result.candidate_accepted is False
    assert result.update_mode == "hold"
    assert result.reason == "candidate_jump_rejected"
    assert np.allclose(result.centerline[:, 1], 0.0)


def test_held_path_expires_after_timeout():
    mem = _memory(hold_last_valid_duration_s=0.5)
    stored = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    mem.update(candidate=_candidate(stored), vehicle_xy=(0.0, 0.0), vehicle_yaw=0.0, now_sec=1.0)

    result = mem.update(
        candidate=_candidate(np.empty((0, 2), dtype=np.float64), updateable=False),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=2.0,
    )

    assert result.update_mode == "reject"
    assert result.centerline.shape == (0, 2)
    assert mem.path is None


def test_curved_path_blends_by_path_relative_station():
    mem = _memory(candidate_jump_reject_threshold_m=5.0, near_alpha=0.2, near_max_lateral_shift_m=0.20)
    theta = np.linspace(0.0, math.pi / 2.0, 8)
    stored = np.column_stack((4.0 * np.sin(theta), 4.0 * (1.0 - np.cos(theta))))
    candidate = np.column_stack((4.2 * np.sin(theta), 4.2 * (1.0 - np.cos(theta))))
    mem.update(candidate=_candidate(stored), vehicle_xy=(0.0, 0.0), vehicle_yaw=0.0, now_sec=1.0)

    result = mem.update(
        candidate=_candidate(candidate),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.1,
    )

    assert result.update_mode == "blend"
    assert np.all(np.isfinite(result.centerline))
    assert result.centerline.shape[0] >= 4
    assert result.near_field_lateral_delta_max_m > 0.0


def test_short_valid_prefix_does_not_append_stored_path():
    mem = _memory(
        candidate_jump_reject_threshold_m=5.0,
        min_estimated_extent_m=6.0,
        max_estimation_extension_m=4.0,
    )
    stored = np.column_stack((np.arange(0.0, 9.0, 1.0), np.zeros(9)))
    live_prefix = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    mem.update(candidate=_candidate(stored), vehicle_xy=(0.0, 0.0), vehicle_yaw=0.0, now_sec=1.0)

    result = mem.update(
        candidate=_candidate(live_prefix, allow_estimation=True),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.1,
    )

    assert result.candidate_accepted is True
    assert result.update_mode == "blend"
    assert result.estimation_mode == "none"
    assert result.estimated_point_count == 0
    assert result.live_prefix_extent_m == pytest.approx(2.0)
    assert result.centerline[-1, 0] == pytest.approx(2.0)


def test_short_valid_prefix_does_not_estimate():
    mem = _memory(
        candidate_jump_reject_threshold_m=5.0,
        min_estimated_extent_m=6.0,
        max_estimation_extension_m=4.0,
    )
    stored = np.column_stack((np.arange(0.0, 9.0, 1.0), np.zeros(9)))
    live_prefix = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    mem.update(candidate=_candidate(stored), vehicle_xy=(0.0, 0.0), vehicle_yaw=0.0, now_sec=1.0)

    result = mem.update(
        candidate=_candidate(live_prefix),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.1,
    )

    assert result.candidate_accepted is True
    assert result.estimation_mode == "none"
    assert result.live_prefix_extent_m == pytest.approx(2.0)
    assert result.estimated_point_count == 0


def test_offset_stored_path_does_not_append_synthetic_tail():
    mem = _memory(
        candidate_jump_reject_threshold_m=5.0,
        min_estimated_extent_m=6.0,
        max_estimation_extension_m=4.0,
        max_tangent_estimation_extension_m=2.0,
        max_estimation_join_lateral_m=0.5,
    )
    stored = np.column_stack((np.arange(0.0, 9.0, 1.0), np.full(9, 3.0)))
    live_prefix = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    mem.update(candidate=_candidate(stored), vehicle_xy=(0.0, 0.0), vehicle_yaw=0.0, now_sec=1.0)

    result = mem.update(
        candidate=_candidate(live_prefix, allow_estimation=True),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.1,
    )

    assert result.candidate_accepted is True
    assert result.update_mode == "blend"
    assert result.estimation_mode == "none"
    assert result.estimated_extent_m == pytest.approx(0.0)
    assert result.centerline[-1, 0] == pytest.approx(2.0)


def test_tangent_estimate_without_memory_is_disabled():
    mem = _memory(
        min_estimated_extent_m=6.0,
        max_estimation_extension_m=4.0,
        max_tangent_estimation_extension_m=1.5,
    )
    live_prefix = np.array([[0.0, 0.0], [2.0, 0.2]], dtype=np.float64)

    result = mem.update(
        candidate=_candidate(live_prefix, allow_estimation=True),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.0,
    )

    assert result.candidate_accepted is True
    assert result.update_mode == "seed"
    assert result.estimation_mode == "none"
    assert result.estimated_extent_m == pytest.approx(0.0)
    assert result.centerline[-1, 0] == pytest.approx(2.0)


def test_invalid_candidate_holds_stored_path_without_estimation_mode():
    mem = _memory(hold_last_valid_duration_s=3.0)
    stored = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    mem.update(candidate=_candidate(stored), vehicle_xy=(0.0, 0.0), vehicle_yaw=0.0, now_sec=1.0)

    result = mem.update(
        candidate=_candidate(np.empty((0, 2), dtype=np.float64), updateable=False),
        vehicle_xy=(0.0, 0.0),
        vehicle_yaw=0.0,
        now_sec=1.1,
    )

    assert result.candidate_accepted is False
    assert result.update_mode == "hold"
    assert result.estimation_mode == "none"
    assert result.estimated_point_count == 0
    assert result.centerline.shape[0] >= 2


def test_same_shared_memory_inputs_do_not_estimate_for_any_planner_source():
    stored = np.column_stack((np.arange(0.0, 9.0, 1.0), np.zeros(9)))
    live_prefix = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    outputs = []
    for source in ("midpoint", "single_boundary", "corridor"):
        mem = _memory(
            candidate_jump_reject_threshold_m=5.0,
            min_estimated_extent_m=6.0,
            max_estimation_extension_m=4.0,
        )
        mem.update(
            candidate=MidlineCandidate(
                centerline=stored,
                source=source,
                updateable=True,
                update_reason="ok",
            ),
            vehicle_xy=(0.0, 0.0),
            vehicle_yaw=0.0,
            now_sec=1.0,
        )
        result = mem.update(
            candidate=MidlineCandidate(
                centerline=live_prefix,
                source=source,
                updateable=True,
                update_reason="ok",
                support_path=live_prefix,
                allow_estimation=True,
            ),
            vehicle_xy=(0.0, 0.0),
            vehicle_yaw=0.0,
            now_sec=1.1,
        )
        outputs.append(result.centerline)
        assert result.estimation_mode == "none"
        assert result.estimated_point_count == 0
        assert result.centerline[-1, 0] == pytest.approx(2.0)

    assert np.allclose(outputs[0], outputs[1])
    assert np.allclose(outputs[1], outputs[2])
