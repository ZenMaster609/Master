from __future__ import annotations

import math
import pathlib
import sys

import pytest

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.cones.tracking.pose import project_planar_pose_constant_twist  # noqa: E402


def test_project_planar_pose_constant_twist_zero_and_negative_delay_leave_pose_unchanged():
    pose = (1.0, 2.0, 0.3)

    assert project_planar_pose_constant_twist(
        pose,
        speed_mps=3.0,
        yaw_rate_rps=0.4,
        delay_s=0.0,
    ) == pytest.approx(pose)
    assert project_planar_pose_constant_twist(
        pose,
        speed_mps=3.0,
        yaw_rate_rps=0.4,
        delay_s=-0.2,
    ) == pytest.approx(pose)


def test_project_planar_pose_constant_twist_straight_motion():
    projected = project_planar_pose_constant_twist(
        (1.0, 2.0, 0.0),
        speed_mps=3.0,
        yaw_rate_rps=0.0,
        delay_s=0.04,
    )

    assert projected == pytest.approx((1.12, 2.0, 0.0), abs=1e-12)


def test_project_planar_pose_constant_twist_near_zero_yaw_rate_uses_straight_motion():
    projected = project_planar_pose_constant_twist(
        (0.0, 0.0, math.pi / 2.0),
        speed_mps=2.0,
        yaw_rate_rps=1e-12,
        delay_s=0.1,
    )

    assert projected[0] == pytest.approx(0.0, abs=1e-12)
    assert projected[1] == pytest.approx(0.2, abs=1e-12)
    assert projected[2] == pytest.approx(math.pi / 2.0, abs=1e-12)


def test_project_planar_pose_constant_twist_turning_motion():
    projected = project_planar_pose_constant_twist(
        (0.0, 0.0, 0.0),
        speed_mps=2.0,
        yaw_rate_rps=0.5,
        delay_s=0.2,
        max_delay_s=1.0,
    )

    assert projected[0] == pytest.approx((2.0 / 0.5) * math.sin(0.1), abs=1e-12)
    assert projected[1] == pytest.approx(-(2.0 / 0.5) * (math.cos(0.1) - 1.0), abs=1e-12)
    assert projected[2] == pytest.approx(0.1, abs=1e-12)


def test_project_planar_pose_constant_twist_caps_delay():
    projected = project_planar_pose_constant_twist(
        (0.0, 0.0, 0.0),
        speed_mps=2.0,
        yaw_rate_rps=0.0,
        delay_s=1.0,
        max_delay_s=0.15,
    )

    assert projected == pytest.approx((0.3, 0.0, 0.0), abs=1e-12)
