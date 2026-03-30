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
    from sim_car.planning.planner_runtime_types import PlannerIdentity  # noqa: E402
    from sim_car.planning.single_boundary_planner_core import SingleBoundaryPlannerResult  # noqa: E402
    from sim_car.planning.single_boundary_planner_node import SingleBoundaryPlannerNode  # noqa: E402
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


def _make_node() -> SingleBoundaryPlannerNode:
    node = object.__new__(SingleBoundaryPlannerNode)
    node._planner_identity = PlannerIdentity(
        node_name="single_boundary_planner_node",
        planner_mode="single_boundary",
        diagnostics_prefix="single_boundary_planner",
        diagnostics_topic="/single_boundary_planner/diagnostics",
    )
    node._diag_pub = _FakePublisher()
    node.publish_control_debug = False
    node.publish_thesis_context = False
    node._hold_mode_active = False
    node._hold_clean_frame_count = 0
    node._active_planner_mode = "single_boundary"
    node.show_raw_cones = False
    node.show_boundary_chains = False
    node.show_pair_lines = False
    node.show_raw_midpoint_chain = False
    node.show_raw_offset_path = False
    node.show_raw_prevalidation_centerline = False
    node.show_lookahead_point = False
    node._current_pair_segments_for_viz = None
    node._last_viz_left_boundary = None
    node._last_viz_right_boundary = None
    node._last_viz_raw_offset_path = None
    node.get_clock = lambda: _FakeClock(TimeMsg(sec=1, nanosec=0))
    return node


def _sample_result() -> SingleBoundaryPlannerResult:
    return SingleBoundaryPlannerResult(
        filtered_points=np.empty((0, 2), dtype=np.float64),
        filtered_colors=[],
        triangulation_edges=np.empty((0, 2), dtype=np.int64),
        candidate_edges=np.empty((0, 2), dtype=np.int64),
        selected_edges=np.empty((0, 2), dtype=np.int64),
        selected_pair_track_ids=np.empty((0, 2), dtype=np.int64),
        midpoints_raw=np.empty((0, 2), dtype=np.float64),
        centerline=np.array([[2.0, 0.1], [4.0, 0.2]], dtype=np.float64),
        left_boundary=np.array([[2.0, 1.8], [4.0, 2.0]], dtype=np.float64),
        right_boundary=np.empty((0, 2), dtype=np.float64),
        used_fallback=True,
        status="ok",
        planner_mode="single_boundary",
        active_boundary_side="blue",
        raw_offset_path=np.array([[2.0, 0.1], [4.0, 0.2]], dtype=np.float64),
    )


def _marker_map(marker_array) -> dict[str, object]:
    return {
        marker.ns: marker
        for marker in marker_array.markers
        if getattr(marker, "ns", "")
    }


def _marker_xy(marker) -> np.ndarray:
    return np.asarray([[point.x, point.y] for point in marker.points], dtype=np.float64)


def test_publish_diagnostics_uses_single_boundary_identity():
    node = _make_node()
    node._publish_diagnostics(
        frame_id="odom",
        centerline_jump_max_m=0.1,
        selected_edge_churn_ratio=0.2,
        tracked_cones_frame_delta_p95_m=0.3,
        centerline_point_count=2,
        selected_edge_count=0,
        status="ok",
        planner_metrics={"planner_mode": "single_boundary"},
    )
    msg = node._diag_pub.messages[-1]
    assert msg.status[0].name == "single_boundary_planner/stability"


def test_single_boundary_reject_reason_maps_to_no_safe_chain():
    node = _make_node()
    result = _sample_result()
    result.status = "no reliable boundary chain"
    result.reject_reason = result.status
    assert node._normalize_core_reject_reason(result) == "no_safe_chain"


def test_build_markers_show_remembered_cones_instead_of_filtered_subset():
    node = _make_node()
    node.show_raw_cones = True
    result = _sample_result()
    result.filtered_points = np.array([[9.0, 9.0]], dtype=np.float64)
    node._update_remembered_cone_viz(
        points_xy=np.array([[1.0, 1.0], [2.0, -2.0]], dtype=np.float64),
        colors=["blue", "yellow"],
    )

    markers = node._build_markers(
        now=TimeMsg(sec=1, nanosec=0),
        frame_id="odom",
        result=result,
        centerline=result.centerline,
        raw_centerline=result.centerline,
        raw_midpoint_chain=result.midpoints_raw,
        status="ok",
        operator_state="fresh",
        control_target_frame=None,
    )
    by_ns = _marker_map(markers)
    assert "remembered_cones" in by_ns
    assert np.allclose(
        _marker_xy(by_ns["remembered_cones"]),
        np.array([[1.0, 1.0], [2.0, -2.0]], dtype=np.float64),
    )
