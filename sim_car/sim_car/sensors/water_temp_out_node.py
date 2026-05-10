"""Virtual water outlet temperature sensor node."""

from .simple_sensor_node import WaterTempOutNode, main_water_temp_out


def main(args=None):
    main_water_temp_out(args=args)


if __name__ == '__main__':
    main()
