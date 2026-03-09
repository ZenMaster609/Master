from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

TEST_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sim_car.controllers.factory import create_steering_controller
from sim_car.controllers.pure_pursuit_controller import PurePursuitConfig, PurePursuitController
from sim_car.controllers.stanley_controller import StanleyConfig, StanleyController


def test_pure_pursuit_returns_valid_output_on_straight_path():
    config = PurePursuitConfig()
    controller = PurePursuitController(config=config, publish_rate_hz=20.0)
    control_path = np.array([[0.0, 0.0], [6.0, 0.0], [12.0, 0.0]], dtype=np.float64)

    output = controller.compute(control_path=control_path, speed_mps=2.5, yaw_rate_rps=0.0)

    assert np.isfinite(output.steering_rad)
    assert np.isfinite(output.kappa)
    assert output.target_point_base.shape == (2,)
    assert output.lookahead_m >= config.lookahead_min_m
    assert abs(output.steering_rad) < 1e-3


def test_pure_pursuit_curved_path_produces_turn_command():
    controller = PurePursuitController(config=PurePursuitConfig(), publish_rate_hz=20.0)
    control_path = np.array(
        [[0.0, 0.0], [4.0, 0.4], [8.0, 1.1], [12.0, 2.1]],
        dtype=np.float64,
    )

    output = controller.compute(control_path=control_path, speed_mps=3.0, yaw_rate_rps=0.0)

    assert output.steering_rad > 0.0
    assert output.kappa > 0.0
    assert np.isfinite(output.lookahead_m)


def test_stanley_sign_behavior_for_left_and_right_offsets():
    config = StanleyConfig(
        heading_gain=0.0,
        k_gain=1.5,
        steering_lowpass_alpha=1.0,
        steering_rate_limit_rad_s=0.0,
        softening_speed_mps=0.5,
    )
    left_controller = StanleyController(config=config, publish_rate_hz=20.0)
    right_controller = StanleyController(config=config, publish_rate_hz=20.0)

    left_path = np.array([[2.0, 1.0], [12.0, 1.0]], dtype=np.float64)
    right_path = np.array([[2.0, -1.0], [12.0, -1.0]], dtype=np.float64)

    left_output = left_controller.compute(control_path=left_path, speed_mps=3.0, yaw_rate_rps=0.0)
    right_output = right_controller.compute(control_path=right_path, speed_mps=3.0, yaw_rate_rps=0.0)

    assert np.isfinite(left_output.steering_rad)
    assert np.isfinite(right_output.steering_rad)
    assert left_output.steering_rad > 0.0
    assert right_output.steering_rad < 0.0


def test_stanley_debug_payload_exposes_stage_outputs_without_changing_command():
    config = StanleyConfig(
        heading_gain=1.0,
        k_gain=1.4,
        softening_speed_mps=0.5,
        steering_lowpass_alpha=0.5,
        steering_rate_limit_rad_s=0.4,
        steering_limit_rad=0.52,
        yaw_rate_damping_gain=0.2,
        use_yaw_rate_damping=True,
    )
    controller = StanleyController(config=config, publish_rate_hz=20.0)
    left_path = np.array([[1.0, 1.0], [12.0, 1.0]], dtype=np.float64)
    right_path = np.array([[1.0, -1.0], [12.0, -1.0]], dtype=np.float64)

    first = controller.compute(control_path=left_path, speed_mps=2.0, yaw_rate_rps=0.3)
    second = controller.compute(control_path=right_path, speed_mps=2.0, yaw_rate_rps=0.3)

    debug = second.stanley_debug
    assert debug is not None
    assert np.isfinite(debug.raw_steering_cmd_rad)
    assert np.isfinite(debug.steering_after_clamp_rad)
    assert np.isfinite(debug.steering_after_filter_rad)
    assert np.isfinite(debug.steering_after_rate_limit_rad)
    assert np.isfinite(debug.final_steering_cmd_rad)
    assert second.steering_rad == pytest.approx(debug.final_steering_cmd_rad)
    max_step = config.steering_rate_limit_rad_s / 20.0
    assert abs(debug.steering_after_rate_limit_rad - first.steering_rad) <= (max_step + 1e-6)
    expected_filtered = (config.steering_lowpass_alpha * debug.steering_after_clamp_rad) + (
        (1.0 - config.steering_lowpass_alpha) * first.steering_rad
    )
    assert debug.steering_after_filter_rad == pytest.approx(expected_filtered, abs=1e-9)
    assert debug.nearest_path_index >= 0
    assert debug.heading_path_index >= debug.nearest_path_index
    assert debug.target_point_y_base_m < 0.0


def test_controller_selection_factory_chooses_correct_module():
    pure_pursuit = create_steering_controller(
        controller_type='pure_pursuit',
        pure_pursuit_config=PurePursuitConfig(),
        stanley_config=StanleyConfig(),
        publish_rate_hz=20.0,
    )
    stanley = create_steering_controller(
        controller_type='stanley',
        pure_pursuit_config=PurePursuitConfig(),
        stanley_config=StanleyConfig(),
        publish_rate_hz=20.0,
    )

    assert isinstance(pure_pursuit, PurePursuitController)
    assert isinstance(stanley, StanleyController)


def test_invalid_controller_type_fails_fast():
    with pytest.raises(ValueError, match='Unsupported control.controller_type'):
        create_steering_controller(
            controller_type='not_a_controller',
            pure_pursuit_config=PurePursuitConfig(),
            stanley_config=StanleyConfig(),
            publish_rate_hz=20.0,
        )
