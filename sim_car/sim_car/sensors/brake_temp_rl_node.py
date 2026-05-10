"""Virtual rear-left brake temperature sensor node."""

from .simple_sensor_node import BrakeTempRlNode, main_brake_temp_rl


def main(args=None):
    main_brake_temp_rl(args=args)


if __name__ == '__main__':
    main()
