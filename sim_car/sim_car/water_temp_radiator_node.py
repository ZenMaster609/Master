"""Virtual radiator temperature sensor node."""

from .virtual_sensors_base import VirtualSensorNodeBase, spin_node


class WaterTempRadiatorNode(VirtualSensorNodeBase):
    def __init__(self):
        super().__init__(
            node_name='water_temp_radiator_node',
            publish_topic='/sim/raw/cooling/water_temp_radiator',
            noise_param='noise_temp',
            noise_default=0.3,
        )

    def compute_value(self) -> float:
        return self.model.water_temp_radiator


def main(args=None):
    spin_node(WaterTempRadiatorNode, args=args)


if __name__ == '__main__':
    main()
