"""Parametric single-topic virtual sensor nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .virtual_sensors_base import VirtualSensorNodeBase, spin_node
from .virtual_sensors_model import VirtualSensorsModel


SensorReader = Callable[[VirtualSensorsModel], float]


@dataclass(frozen=True)
class SimpleSensorConfig:
    node_name: str
    publish_topic: str
    noise_param: str
    noise_default: float
    read_value: SensorReader
    needs_brake_cmd: bool = False


class SimpleSensorNode(VirtualSensorNodeBase):
    def __init__(self, config: SimpleSensorConfig):
        self._simple_sensor_config = config
        super().__init__(
            node_name=config.node_name,
            publish_topic=config.publish_topic,
            noise_param=config.noise_param,
            noise_default=config.noise_default,
            needs_brake_cmd=config.needs_brake_cmd,
        )

    def compute_value(self) -> float:
        return self._simple_sensor_config.read_value(self.model)


def create_simple_sensor_node(config: SimpleSensorConfig) -> type[SimpleSensorNode]:
    class ConfiguredSimpleSensorNode(SimpleSensorNode):
        def __init__(self):
            super().__init__(config)

    ConfiguredSimpleSensorNode.__name__ = "".join(
        part.capitalize() for part in config.node_name.split("_")
    )
    return ConfiguredSimpleSensorNode


WATER_TEMP_IN_CONFIG = SimpleSensorConfig(
    node_name="water_temp_in_node",
    publish_topic="/sim/raw/cooling/water_temp_in",
    noise_param="noise_temp",
    noise_default=0.3,
    read_value=lambda model: model.water_temp_in,
)
WaterTempInNode = create_simple_sensor_node(WATER_TEMP_IN_CONFIG)

WATER_TEMP_OUT_CONFIG = SimpleSensorConfig(
    node_name="water_temp_out_node",
    publish_topic="/sim/raw/cooling/water_temp_out",
    noise_param="noise_temp",
    noise_default=0.3,
    read_value=lambda model: model.water_temp_out,
)
WaterTempOutNode = create_simple_sensor_node(WATER_TEMP_OUT_CONFIG)

BRAKE_TEMP_FR_CONFIG = SimpleSensorConfig(
    node_name="brake_temp_fr_node",
    publish_topic="/sim/raw/brakes/temp_fr",
    noise_param="noise_brake_temp",
    noise_default=1.0,
    read_value=lambda model: model.brake_temp_fr,
    needs_brake_cmd=True,
)
BrakeTempFrNode = create_simple_sensor_node(BRAKE_TEMP_FR_CONFIG)

BRAKE_TEMP_RL_CONFIG = SimpleSensorConfig(
    node_name="brake_temp_rl_node",
    publish_topic="/sim/raw/brakes/temp_rl",
    noise_param="noise_brake_temp",
    noise_default=1.0,
    read_value=lambda model: model.brake_temp_rl,
    needs_brake_cmd=True,
)
BrakeTempRlNode = create_simple_sensor_node(BRAKE_TEMP_RL_CONFIG)

WATER_FLOW_CONFIG = SimpleSensorConfig(
    node_name="water_flow_node",
    publish_topic="/sim/raw/cooling/water_flow",
    noise_param="noise_flow",
    noise_default=0.5,
    read_value=lambda model: model.compute_water_flow(),
)
WaterFlowNode = create_simple_sensor_node(WATER_FLOW_CONFIG)

WATER_PRESSURE_CONFIG = SimpleSensorConfig(
    node_name="water_pressure_node",
    publish_topic="/sim/raw/cooling/water_pressure",
    noise_param="noise_pressure",
    noise_default=0.02,
    read_value=lambda model: model.compute_water_pressure(),
)
WaterPressureNode = create_simple_sensor_node(WATER_PRESSURE_CONFIG)

PITOT_DYNAMIC_PRESSURE_CONFIG = SimpleSensorConfig(
    node_name="pitot_dynamic_pressure_node",
    publish_topic="/sim/raw/pitot/dynamic_pressure",
    noise_param="noise_pitot",
    noise_default=2.0,
    read_value=lambda model: model.compute_pitot_pressure(),
)
PitotDynamicPressureNode = create_simple_sensor_node(PITOT_DYNAMIC_PRESSURE_CONFIG)


def main_water_temp_in(args=None) -> None:
    spin_node(WaterTempInNode, args=args)


def main_water_temp_out(args=None) -> None:
    spin_node(WaterTempOutNode, args=args)


def main_brake_temp_fr(args=None) -> None:
    spin_node(BrakeTempFrNode, args=args)


def main_brake_temp_rl(args=None) -> None:
    spin_node(BrakeTempRlNode, args=args)


def main_water_flow(args=None) -> None:
    spin_node(WaterFlowNode, args=args)


def main_water_pressure(args=None) -> None:
    spin_node(WaterPressureNode, args=args)


def main_pitot_dynamic_pressure(args=None) -> None:
    spin_node(PitotDynamicPressureNode, args=args)
