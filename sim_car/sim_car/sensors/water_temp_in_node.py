"""Virtual water inlet temperature sensor node."""

from .virtual_sensors_base import VirtualSensorNodeBase, spin_node


class WaterTempInNode(VirtualSensorNodeBase):
    def __init__(self):
        super().__init__(
            node_name='water_temp_in_node',
            publish_topic='/sim/raw/cooling/water_temp_in',
            noise_param='noise_temp',
            noise_default=0.3,
        )

    def compute_value(self) -> float:
        return self.model.water_temp_in


def main(args=None):
    spin_node(WaterTempInNode, args=args)


if __name__ == '__main__':
    main()
