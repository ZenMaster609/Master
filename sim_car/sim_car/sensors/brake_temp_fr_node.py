"""Virtual front-right brake temperature sensor node."""

from .simple_sensor_node import BrakeTempFrNode, main_brake_temp_fr


def main(args=None):
    main_brake_temp_fr(args=args)


if __name__ == '__main__':
    main()
