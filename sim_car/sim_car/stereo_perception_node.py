"""Stereo-only wrapper for PerceptionNode."""

from sim_car.perception_node import main as perception_main


def main(args=None):
    perception_main(args=args, force_stereo=True)


if __name__ == '__main__':
    main()
