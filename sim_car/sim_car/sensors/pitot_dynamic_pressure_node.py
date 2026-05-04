"""Virtual pitot dynamic pressure sensor node."""

from .simple_sensor_node import PitotDynamicPressureNode, main_pitot_dynamic_pressure


def main(args=None):
    main_pitot_dynamic_pressure(args=args)


if __name__ == '__main__':
    main()
