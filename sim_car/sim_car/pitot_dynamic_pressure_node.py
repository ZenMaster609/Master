"""Virtual pitot dynamic pressure sensor node."""

from .virtual_sensors_base import VirtualSensorNodeBase, spin_node


class PitotDynamicPressureNode(VirtualSensorNodeBase):
    def __init__(self):
        super().__init__(
            node_name='pitot_dynamic_pressure_node',
            publish_topic='/sim/pitot/dynamic_pressure',
            noise_param='noise_pitot',
            noise_default=2.0,
        )

    def compute_value(self) -> float:
        return self.model.compute_pitot_pressure()


def main(args=None):
    spin_node(PitotDynamicPressureNode, args=args)


if __name__ == '__main__':
    main()
