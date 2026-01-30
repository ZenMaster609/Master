"""Virtual water pressure sensor node."""

from .virtual_sensors_base import VirtualSensorNodeBase, spin_node


class WaterPressureNode(VirtualSensorNodeBase):
    def __init__(self):
        super().__init__(
            node_name='water_pressure_node',
            publish_topic='/sim/cooling/water_pressure',
            noise_param='noise_pressure',
            noise_default=0.02,
        )

    def compute_value(self) -> float:
        return self.model.compute_water_pressure()


def main(args=None):
    spin_node(WaterPressureNode, args=args)


if __name__ == '__main__':
    main()
