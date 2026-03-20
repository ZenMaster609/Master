from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from builtin_interfaces.msg import Time as TimeMsg

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from sim_car.planning.midpoint_planner_core import MidpointPlannerResult  # noqa: E402
    from sim_car.planning.midpoint_planner_node import MidpointPlannerNode  # noqa: E402
    from sim_car.planning.planner_runtime_types import PlannerIdentity  # noqa: E402
except ImportError as exc:  # pragma: no cover - depends on generated ROS interfaces
    pytest.skip(f"ROS planner node imports unavailable: {exc}", allow_module_level=True)


class _FakePublisher:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, msg: object) -> None:
        self.messages.append(msg)


class _FakeNow:
    def __init__(self, stamp: TimeMsg) -> None:
        self._stamp = stamp
        self.nanoseconds = (int(stamp.sec) * 1_000_000_000) + int(stamp.nanosec)

    def to_msg(self) -> TimeMsg:
        return self._stamp


class _FakeClock:
    def __init__(self, stamp: TimeMsg) -> None:
        self._stamp = stamp

    def now(self) -> _FakeNow:
        return _FakeNow(self._stamp)


def _make_node() -> MidpointPlannerNode:
    node = object.__new__(MidpointPlannerNode)
    node._planner_identity = PlannerIdentity(
        node_name="midpoint_planner_node",
        planner_mode="midpoint",
        diagnostics_prefix="midpoint_planner",
        diagnostics_topic="/midpoint_planner/diagnostics",
    )
    node._diag_pub = _FakePublisher()
    node.publish_control_debug = False
    node.publish_thesis_context = False
    node._hold_mode_active = False
    node._hold_clean_frame_count = 0
    node._active_planner_mode = "midpoint"
    node.get_clock = lambda: _FakeClock(TimeMsg(sec=1, nanosec=0))
    return node


def _sample_result() -> MidpointPlannerResult:
    return MidpointPlannerResult(
        filtered_points=np.empty((0, 2), dtype=np.float64),
        filtered_colors=[],
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.empty((0, 2), dtype=np.int64),
        selected_pair_track_ids=np.empty((0, 2), dtype=np.int64),
        midpoints_raw=np.array([[2.0, 0.0], [4.0, 0.0]], dtype=np.float64),
        centerline=np.array([[2.0, 0.0], [4.0, 0.0]], dtype=np.float64),
        left_boundary=np.empty((0, 2), dtype=np.float64),
        right_boundary=np.empty((0, 2), dtype=np.float64),
        used_fallback=False,
        status="ok",
        planner_mode="midpoint",
    )


def test_publish_diagnostics_uses_midpoint_identity():
    node = _make_node()
    node._publish_diagnostics(
        frame_id="odom",
        centerline_jump_max_m=0.1,
        selected_edge_churn_ratio=0.2,
        tracked_cones_frame_delta_p95_m=0.3,
        centerline_point_count=2,
        selected_edge_count=1,
        status="ok",
        planner_metrics={"planner_mode": "midpoint"},
    )
    msg = node._diag_pub.messages[-1]
    assert msg.status[0].name == "midpoint_planner/stability"


def test_normalize_core_reject_reason_maps_to_no_safe_chain():
    node = _make_node()
    result = _sample_result()
    result.status = "no reliable midpoint chain"
    result.reject_reason = result.status
    assert node._normalize_core_reject_reason(result) == "no_safe_chain"
