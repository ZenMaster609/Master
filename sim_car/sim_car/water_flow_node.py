"""Virtual water flow sensor node."""

from .virtual_sensors_base import VirtualSensorNodeBase, spin_node


class WaterFlowNode(VirtualSensorNodeBase):
    def __init__(self):
        super().__init__(
            node_name='water_flow_node',
            publish_topic='/sim/raw/cooling/water_flow',
            noise_param='noise_flow',
            noise_default=0.5,
        )

    def compute_value(self) -> float:
        return self.model.compute_water_flow()


def main(args=None):
    spin_node(WaterFlowNode, args=args)


if __name__ == '__main__':
    main()
