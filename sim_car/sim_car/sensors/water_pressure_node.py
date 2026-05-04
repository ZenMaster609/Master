"""Virtual water pressure sensor node."""

from .simple_sensor_node import WaterPressureNode, main_water_pressure


def main(args=None):
    main_water_pressure(args=args)


if __name__ == '__main__':
    main()
