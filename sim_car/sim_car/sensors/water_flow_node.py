"""Virtual water flow sensor node."""

from .simple_sensor_node import WaterFlowNode, main_water_flow


def main(args=None):
    main_water_flow(args=args)


if __name__ == '__main__':
    main()
