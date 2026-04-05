from __future__ import annotations

import pathlib
import sys

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.sensors.steering_convention import steering_joint_mean_to_deg


def test_steering_joint_mean_to_deg_flips_joint_sign_by_default():
    assert steering_joint_mean_to_deg(0.1, 0.1) < 0.0


def test_steering_joint_mean_to_deg_supports_positive_sign_override():
    assert steering_joint_mean_to_deg(0.1, 0.1, sign=1.0) > 0.0
