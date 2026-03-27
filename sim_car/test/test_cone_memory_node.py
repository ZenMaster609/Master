from __future__ import annotations

import pathlib
import sys

import pytest
from builtin_interfaces.msg import Time as TimeMsg

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from sim_car.cones.nodes.memory_node import ConeMemoryNode  # noqa: E402
    from sim_car.cones.tracking.tracker import ConeTrack  # noqa: E402
except ImportError as exc:  # pragma: no cover - depends on generated ROS interfaces
    pytest.skip(f"ROS cone memory node imports unavailable: {exc}", allow_module_level=True)


class _FakePublisher:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, msg: object) -> None:
        self.messages.append(msg)


class _FakeNow:
    def __init__(self, stamp: TimeMsg) -> None:
        self._stamp = stamp

    def to_msg(self) -> TimeMsg:
        return self._stamp


def _track(track_id: int, x: float, y: float, label: str) -> ConeTrack:
    probs = [0.0, 0.0, 0.0, 0.0]
    if label == 'orange':
        probs[3] = 1.0
    elif label == 'yellow':
        probs[2] = 1.0
    elif label == 'blue':
        probs[1] = 1.0
    else:
        probs[0] = 1.0
    return ConeTrack(
        track_id=track_id,
        x=x,
        y=y,
        z=0.0,
        variance=0.1,
        class_probs=probs,
        stable_label=label,
        stable_color_confidence=0.95,
        track_confidence=0.9,
        first_seen_sec=1.0,
        last_seen_sec=2.0,
        seen_count=4,
        consecutive_seen_count=4,
        track_state=1,
        camera_class_confirmed=True,
    )


def test_publish_tracked_cones_keeps_raw_color_and_sets_boundary_color():
    node = object.__new__(ConeMemoryNode)
    node._tracked_pub = _FakePublisher()
    node._last_tracker_update_stamp = None
    node.odom_frame = 'odom'

    now = _FakeNow(TimeMsg(sec=2, nanosec=0))
    tracks = [
        _track(1, 3.0, 1.0, 'orange'),
        _track(2, 3.0, -1.0, 'orange'),
    ]

    node._publish_tracked_cones(
        tracks,
        now,
        vehicle_x=0.0,
        vehicle_y=0.0,
        heading_x=1.0,
        heading_y=0.0,
    )

    msg = node._tracked_pub.messages[-1]
    assert [cone.color for cone in msg.cones] == ['orange', 'orange']
    assert [cone.boundary_color for cone in msg.cones] == ['blue', 'yellow']
