from __future__ import annotations

from typing import Any

import numpy as np

from sim_car.controllers.factory import create_steering_controller
from sim_car.controllers.pure_pursuit_controller import PurePursuitConfig
from sim_car.controllers.stanley_controller import StanleyConfig
from sim_car.planning.pipeline_defaults import WHEELBASE_M_DEFAULT


def _read_param(node: Any, name: str, default: object) -> object:
    try:
        return node.get_parameter(name).value
    except Exception:
        return default


def _node_wheelbase_m(node: Any) -> float:
    return max(0.1, float(getattr(node, "vehicle_wheelbase_m", WHEELBASE_M_DEFAULT)))


def build_stanley_config(node: Any) -> StanleyConfig:
    defaults = StanleyConfig(wheelbase_m=_node_wheelbase_m(node))
    return StanleyConfig(
        k_gain=max(0.0, float(_read_param(node, 'stanley.k_gain', defaults.k_gain))),
        softening_speed_mps=max(
            0.0,
            float(_read_param(node, 'stanley.softening_speed_mps', defaults.softening_speed_mps)),
        ),
        heading_gain=float(_read_param(node, 'stanley.heading_gain', defaults.heading_gain)),
        lookahead_idx_offset=max(
            0,
            int(_read_param(node, 'stanley.lookahead_idx_offset', defaults.lookahead_idx_offset)),
        ),
        steering_limit_rad=max(
            0.01,
            float(_read_param(node, 'stanley.steering_limit_rad', defaults.steering_limit_rad)),
        ),
        steering_lowpass_alpha=float(
            np.clip(
                float(_read_param(node, 'stanley.steering_lowpass_alpha', defaults.steering_lowpass_alpha)),
                0.0,
                1.0,
            )
        ),
        steering_rate_limit_rad_s=max(
            0.0,
            float(_read_param(node, 'stanley.steering_rate_limit_rad_s', defaults.steering_rate_limit_rad_s)),
        ),
        use_yaw_rate_damping=bool(
            _read_param(node, 'stanley.use_yaw_rate_damping', defaults.use_yaw_rate_damping)
        ),
        yaw_rate_damping_gain=max(
            0.0,
            float(_read_param(node, 'stanley.yaw_rate_damping_gain', defaults.yaw_rate_damping_gain)),
        ),
        wheelbase_m=max(0.1, float(_read_param(node, 'stanley.wheelbase_m', defaults.wheelbase_m))),
        cross_track_deadband_m=max(
            0.0,
            float(_read_param(node, 'stanley.cross_track_deadband_m', defaults.cross_track_deadband_m)),
        ),
    )


def build_pure_pursuit_config(node: Any) -> PurePursuitConfig:
    defaults = PurePursuitConfig(wheelbase_m=_node_wheelbase_m(node))
    config = PurePursuitConfig(
        lookahead_m=max(0.0, float(_read_param(node, 'pure_pursuit.lookahead_m', defaults.lookahead_m))),
        min_lookahead_m=max(
            0.01,
            float(_read_param(node, 'pure_pursuit.min_lookahead_m', defaults.min_lookahead_m)),
        ),
        max_lookahead_m=max(
            0.01,
            float(_read_param(node, 'pure_pursuit.max_lookahead_m', defaults.max_lookahead_m)),
        ),
        lookahead_gain=max(
            0.0,
            float(_read_param(node, 'pure_pursuit.lookahead_gain', defaults.lookahead_gain)),
        ),
        steering_limit_rad=max(
            0.01,
            float(_read_param(node, 'pure_pursuit.steering_limit_rad', defaults.steering_limit_rad)),
        ),
        steering_lowpass_alpha=float(
            np.clip(
                float(_read_param(node, 'pure_pursuit.steering_lowpass_alpha', defaults.steering_lowpass_alpha)),
                0.0,
                1.0,
            )
        ),
        steering_rate_limit_rad_s=max(
            0.0,
            float(_read_param(node, 'pure_pursuit.steering_rate_limit_rad_s', defaults.steering_rate_limit_rad_s)),
        ),
        wheelbase_m=max(0.1, float(_read_param(node, 'pure_pursuit.wheelbase_m', defaults.wheelbase_m))),
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
