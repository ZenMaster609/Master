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


def test_controller_selection_factory_chooses_stanley_module():
    stanley = create_steering_controller(
        controller_type='stanley',
        stanley_config=StanleyConfig(),
        publish_rate_hz=20.0,
    )

    assert isinstance(stanley, StanleyController)


def test_controller_selection_factory_chooses_pure_pursuit_module():
    pure_pursuit = create_steering_controller(
        controller_type='pure_pursuit',
        stanley_config=StanleyConfig(),
        pure_pursuit_config=PurePursuitConfig(),
        publish_rate_hz=20.0,
    )

    assert isinstance(pure_pursuit, PurePursuitController)


def test_pure_pursuit_sign_behavior_for_left_and_right_offsets():
    config = PurePursuitConfig(
        lookahead_m=2.0,
        min_lookahead_m=2.0,
        max_lookahead_m=2.0,
        steering_lowpass_alpha=1.0,
        steering_rate_limit_rad_s=0.0,
    )
    left_controller = PurePursuitController(config=config, publish_rate_hz=20.0)
    right_controller = PurePursuitController(config=config, publish_rate_hz=20.0)

    left_path = np.array([[2.0, 1.0], [12.0, 1.0]], dtype=np.float64)
    right_path = np.array([[2.0, -1.0], [12.0, -1.0]], dtype=np.float64)

    left_output = left_controller.compute(control_path=left_path, speed_mps=3.0, yaw_rate_rps=0.0)
    right_output = right_controller.compute(control_path=right_path, speed_mps=3.0, yaw_rate_rps=0.0)

    assert left_output.steering_rad > 0.0
    assert right_output.steering_rad < 0.0


def test_pure_pursuit_lookahead_scales_with_speed_and_clamps():
    config = PurePursuitConfig(
        lookahead_m=2.0,
        min_lookahead_m=1.5,
        max_lookahead_m=4.0,
        lookahead_gain=1.0,
        steering_lowpass_alpha=1.0,
        steering_rate_limit_rad_s=0.0,
    )
    controller = PurePursuitController(config=config, publish_rate_hz=20.0)
    path = np.array([[0.0, 0.0], [20.0, 0.0]], dtype=np.float64)

    slow = controller.compute(control_path=path, speed_mps=0.0, yaw_rate_rps=0.0)
    fast = controller.compute(control_path=path, speed_mps=10.0, yaw_rate_rps=0.0)

    assert slow.steering_rad == pytest.approx(0.0, abs=1e-9)
    assert fast.steering_rad == pytest.approx(0.0, abs=1e-9)
    assert slow.lookahead_m == pytest.approx(2.0, abs=1e-9)
    assert fast.lookahead_m == pytest.approx(4.0, abs=1e-9)
    assert slow.target_point_base[0] == pytest.approx(2.0, abs=1e-9)
    assert fast.target_point_base[0] == pytest.approx(4.0, abs=1e-9)


def test_pure_pursuit_short_path_uses_last_point():
    controller = PurePursuitController(
        config=PurePursuitConfig(
            lookahead_m=5.0,
            min_lookahead_m=5.0,
            max_lookahead_m=5.0,
            steering_lowpass_alpha=1.0,
            steering_rate_limit_rad_s=0.0,
        ),
        publish_rate_hz=20.0,
    )
    short_path = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    output = controller.compute(control_path=short_path, speed_mps=0.0, yaw_rate_rps=0.0)

    assert np.allclose(output.target_point_base, np.array([1.0, 0.0], dtype=np.float64))
    assert output.lookahead_m == pytest.approx(1.0, abs=1e-9)


def test_pure_pursuit_filter_and_rate_limit_match_configuration():
    config = PurePursuitConfig(
        lookahead_m=2.0,
        min_lookahead_m=2.0,
        max_lookahead_m=2.0,
        steering_limit_rad=0.52,
        steering_lowpass_alpha=0.5,
        steering_rate_limit_rad_s=0.4,
    )
    controller = PurePursuitController(config=config, publish_rate_hz=20.0)
    raw_controller = PurePursuitController(
        config=PurePursuitConfig(
            lookahead_m=2.0,
            min_lookahead_m=2.0,
            max_lookahead_m=2.0,
            steering_limit_rad=0.52,
            steering_lowpass_alpha=1.0,
            steering_rate_limit_rad_s=0.0,
        ),
        publish_rate_hz=20.0,
    )
    left_path = np.array([[2.0, 1.0], [12.0, 1.0]], dtype=np.float64)
    right_path = np.array([[2.0, -1.0], [12.0, -1.0]], dtype=np.float64)

    first = controller.compute(control_path=left_path, speed_mps=3.0, yaw_rate_rps=0.0)
    raw_second = raw_controller.compute(control_path=right_path, speed_mps=3.0, yaw_rate_rps=0.0)
    second = controller.compute(control_path=right_path, speed_mps=3.0, yaw_rate_rps=0.0)

    expected_filtered = (config.steering_lowpass_alpha * raw_second.steering_rad) + (
        (1.0 - config.steering_lowpass_alpha) * first.steering_rad
    )
    max_step = config.steering_rate_limit_rad_s / 20.0
    expected_final = float(
        np.clip(expected_filtered, first.steering_rad - max_step, first.steering_rad + max_step)
    )

    assert second.steering_rad == pytest.approx(expected_final, abs=1e-9)
    assert abs(second.steering_rad - first.steering_rad) <= (max_step + 1e-9)


def test_invalid_controller_type_fails_fast():
    with pytest.raises(ValueError, match='Unsupported control.controller_type'):
        create_steering_controller(
            controller_type='not_a_controller',
            stanley_config=StanleyConfig(),
            publish_rate_hz=20.0,
        )
