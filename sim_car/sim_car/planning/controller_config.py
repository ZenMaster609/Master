from __future__ import annotations

from typing import Any

import numpy as np

from sim_car.controllers.factory import create_steering_controller
from sim_car.controllers.pure_pursuit_controller import PurePursuitConfig
from sim_car.controllers.stanley_controller import StanleyConfig


def build_stanley_config(node: Any) -> StanleyConfig:
    return StanleyConfig(
        k_gain=max(0.0, float(node.get_parameter('stanley.k_gain').value)),
        softening_speed_mps=max(0.0, float(node.get_parameter('stanley.softening_speed_mps').value)),
        heading_gain=float(node.get_parameter('stanley.heading_gain').value),
        lookahead_idx_offset=max(0, int(node.get_parameter('stanley.lookahead_idx_offset').value)),
        steering_limit_rad=max(0.01, float(node.get_parameter('stanley.steering_limit_rad').value)),
        steering_lowpass_alpha=float(
            np.clip(float(node.get_parameter('stanley.steering_lowpass_alpha').value), 0.0, 1.0)
        ),
        steering_rate_limit_rad_s=max(
            0.0,
            float(node.get_parameter('stanley.steering_rate_limit_rad_s').value),
        ),
        use_yaw_rate_damping=bool(node.get_parameter('stanley.use_yaw_rate_damping').value),
        yaw_rate_damping_gain=max(
            0.0,
            float(node.get_parameter('stanley.yaw_rate_damping_gain').value),
        ),
        wheelbase_m=max(0.1, float(node.get_parameter('stanley.wheelbase_m').value)),
        cross_track_deadband_m=max(
            0.0,
            float(node.get_parameter('stanley.cross_track_deadband_m').value),
        ),
    )


def build_pure_pursuit_config(node: Any) -> PurePursuitConfig:
    config = PurePursuitConfig(
        lookahead_m=max(0.0, float(node.get_parameter('pure_pursuit.lookahead_m').value)),
        min_lookahead_m=max(0.01, float(node.get_parameter('pure_pursuit.min_lookahead_m').value)),
        max_lookahead_m=max(
            0.01,
            float(node.get_parameter('pure_pursuit.max_lookahead_m').value),
        ),
        lookahead_gain=max(0.0, float(node.get_parameter('pure_pursuit.lookahead_gain').value)),
        steering_limit_rad=max(
            0.01,
            float(node.get_parameter('pure_pursuit.steering_limit_rad').value),
        ),
        steering_lowpass_alpha=float(
            np.clip(float(node.get_parameter('pure_pursuit.steering_lowpass_alpha').value), 0.0, 1.0)
        ),
        steering_rate_limit_rad_s=max(
            0.0,
            float(node.get_parameter('pure_pursuit.steering_rate_limit_rad_s').value),
        ),
        wheelbase_m=max(0.1, float(node.get_parameter('pure_pursuit.wheelbase_m').value)),
    )
    if config.max_lookahead_m < config.min_lookahead_m:
        return PurePursuitConfig(
            lookahead_m=config.lookahead_m,
            min_lookahead_m=config.min_lookahead_m,
            max_lookahead_m=config.min_lookahead_m,
            lookahead_gain=config.lookahead_gain,
            steering_limit_rad=config.steering_limit_rad,
            steering_lowpass_alpha=config.steering_lowpass_alpha,
            steering_rate_limit_rad_s=config.steering_rate_limit_rad_s,
            wheelbase_m=config.wheelbase_m,
        )
    return config


def build_steering_controller(
    *,
    node: Any,
    controller_type: str,
    publish_rate_hz: float,
):
    return create_steering_controller(
        controller_type=controller_type,
        stanley_config=build_stanley_config(node),
        pure_pursuit_config=build_pure_pursuit_config(node),
        publish_rate_hz=publish_rate_hz,
    )


def build_stanley_config_for_stage(node: Any, stage: int) -> StanleyConfig:
    ns = f'stanley_{stage}'
    return StanleyConfig(
        k_gain=max(0.0, float(node.get_parameter(f'{ns}.k_gain').value)),
        softening_speed_mps=max(0.0, float(node.get_parameter(f'{ns}.softening_speed_mps').value)),
        heading_gain=float(node.get_parameter(f'{ns}.heading_gain').value),
        lookahead_idx_offset=max(0, int(node.get_parameter(f'{ns}.lookahead_idx_offset').value)),
        steering_limit_rad=max(0.01, float(node.get_parameter(f'{ns}.steering_limit_rad').value)),
        steering_lowpass_alpha=float(
            np.clip(float(node.get_parameter(f'{ns}.steering_lowpass_alpha').value), 0.0, 1.0)
        ),
        steering_rate_limit_rad_s=max(
            0.0,
            float(node.get_parameter(f'{ns}.steering_rate_limit_rad_s').value),
        ),
        use_yaw_rate_damping=bool(node.get_parameter(f'{ns}.use_yaw_rate_damping').value),
        yaw_rate_damping_gain=max(
            0.0,
            float(node.get_parameter(f'{ns}.yaw_rate_damping_gain').value),
        ),
        wheelbase_m=max(0.1, float(node.get_parameter(f'{ns}.wheelbase_m').value)),
        cross_track_deadband_m=max(
            0.0,
            float(node.get_parameter(f'{ns}.cross_track_deadband_m').value),
        ),
    )


def build_pure_pursuit_config_for_stage(node: Any, stage: int) -> PurePursuitConfig:
    ns = f'pure_pursuit_{stage}'
    config = PurePursuitConfig(
        lookahead_m=max(0.0, float(node.get_parameter(f'{ns}.lookahead_m').value)),
        min_lookahead_m=max(0.01, float(node.get_parameter(f'{ns}.min_lookahead_m').value)),
        max_lookahead_m=max(
            0.01,
            float(node.get_parameter(f'{ns}.max_lookahead_m').value),
        ),
        lookahead_gain=max(0.0, float(node.get_parameter(f'{ns}.lookahead_gain').value)),
        steering_limit_rad=max(
            0.01,
            float(node.get_parameter(f'{ns}.steering_limit_rad').value),
        ),
        steering_lowpass_alpha=float(
            np.clip(float(node.get_parameter(f'{ns}.steering_lowpass_alpha').value), 0.0, 1.0)
        ),
        steering_rate_limit_rad_s=max(
            0.0,
            float(node.get_parameter(f'{ns}.steering_rate_limit_rad_s').value),
        ),
        wheelbase_m=max(0.1, float(node.get_parameter(f'{ns}.wheelbase_m').value)),
    )
    if config.max_lookahead_m < config.min_lookahead_m:
        return PurePursuitConfig(
            lookahead_m=config.lookahead_m,
            min_lookahead_m=config.min_lookahead_m,
            max_lookahead_m=config.min_lookahead_m,
            lookahead_gain=config.lookahead_gain,
            steering_limit_rad=config.steering_limit_rad,
            steering_lowpass_alpha=config.steering_lowpass_alpha,
            steering_rate_limit_rad_s=config.steering_rate_limit_rad_s,
            wheelbase_m=config.wheelbase_m,
        )
    return config


def build_steering_controller_for_stage(
    *,
    node: Any,
    controller_type: str,
    publish_rate_hz: float,
    stage: int,
):
    return create_steering_controller(
        controller_type=controller_type,
        stanley_config=build_stanley_config_for_stage(node, stage),
        pure_pursuit_config=build_pure_pursuit_config_for_stage(node, stage),
        publish_rate_hz=publish_rate_hz,
    )
