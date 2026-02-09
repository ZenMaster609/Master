"""Virtual front-right brake temperature sensor node."""

from .virtual_sensors_base import VirtualSensorNodeBase, spin_node


class BrakeTempFrNode(VirtualSensorNodeBase):
    def __init__(self):
        super().__init__(
            node_name='brake_temp_fr_node',
            publish_topic='/sim/raw/brakes/temp_fr',
            noise_param='noise_brake_temp',
            noise_default=1.0,
            needs_brake_cmd=True,
        )

    def compute_value(self) -> float:
        return self.model.brake_temp_fr


def main(args=None):
    spin_node(BrakeTempFrNode, args=args)


if __name__ == '__main__':
    main()
