"""Factory helpers for selecting a steering controller implementation."""

from __future__ import annotations

from sim_car.controllers.base import SteeringController
from sim_car.controllers.pure_pursuit_controller import PurePursuitConfig, PurePursuitController
from sim_car.controllers.stanley_controller import StanleyConfig, StanleyController


def create_steering_controller(
    *,
    controller_type: str,
    stanley_config: StanleyConfig,
    pure_pursuit_config: PurePursuitConfig | None = None,
    publish_rate_hz: float,
) -> SteeringController:
    """Instantiate the steering controller selected in parameters."""

    normalized = str(controller_type).strip().lower()
    if normalized == 'stanley':
        return StanleyController(
            config=stanley_config,
            publish_rate_hz=publish_rate_hz,
        )
    if normalized == 'pure_pursuit':
        if pure_pursuit_config is None:
            raise ValueError('pure_pursuit_config is required for pure_pursuit controller')
        return PurePursuitController(
            config=pure_pursuit_config,
            publish_rate_hz=publish_rate_hz,
        )
    raise ValueError(
        "Unsupported control.controller_type '%s'. Supported values: stanley, pure_pursuit"
        % controller_type
    )
