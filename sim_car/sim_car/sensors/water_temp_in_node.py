"""Virtual water inlet temperature sensor node."""

from .simple_sensor_node import WaterTempInNode, main_water_temp_in


def main(args=None):
    main_water_temp_in(args=args)


if __name__ == '__main__':
    main()
