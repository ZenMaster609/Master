#!/usr/bin/env python3
"""
VCANPublisherNode - Publishes fake CAN frames for testing.

Generates realistic wheel encoder CAN frames and publishes them
to the /to_can_bus topic for testing without real hardware.

Usage:
    # First setup vcan interface
    sudo modprobe vcan
    sudo ip link add dev vcan0 type vcan
    sudo ip link set up vcan0

    # Run the publisher
    ros2 run canbus_decoder vcan_publisher_node

    # With specific mode
    ros2 run canbus_decoder vcan_publisher_node --ros-args -p mode:=circle
"""

import rclpy
from rclpy.node import Node
from can_msgs.msg import Frame
import struct
import math
import time


class VCANPublisherNode(Node):
    """
    Publishes fake CAN frames simulating vehicle wheel encoders.

    Generates CAN frames matching the protocol expected by can_decoder_node:
    - ID 0xB9 (185): Front axle - FL, FR velocities + steering
    - ID 0xBA (186): Rear axle - RL, RR velocities

    Parameters:
        publish_rate_hz (float): Frame publication rate (default: 100)
        mode (str): Motion pattern - 'static', 'linear', 'circle', 'random'
        base_velocity_mps (float): Base wheel velocity in m/s (default: 1.0)
    """

    def __init__(self):
        super().__init__('vcan_publisher_node')

        # Declare parameters
        self.declare_parameter('publish_rate_hz', 100.0)
        self.declare_parameter('mode', 'circle')
        self.declare_parameter('base_velocity_mps', 5.0)  # 5 m/s (~18 km/h) for visible test data

        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate_hz').value
        self.mode = self.get_parameter('mode').value
        self.base_velocity = self.get_parameter('base_velocity_mps').value

        self.get_logger().info(f'VCANPublisherNode starting...')
        self.get_logger().info(f'  Mode: {self.mode}')
        self.get_logger().info(f'  Base velocity: {self.base_velocity} m/s')
        self.get_logger().info(f'  Publish rate: {self.publish_rate} Hz')

        # CAN IDs from can_protocol.py
        self.FRONT_AXLE_ID = 0xB9  # 185
        self.REAR_AXLE_ID = 0xBA   # 186

        # State for motion simulation
        self._start_time = time.time()
        self._steering_angle = 0.0  # radians
        self._msg_count = 0

        # Publisher
        self.can_pub = self.create_publisher(
            Frame,
            '/to_can_bus',
            10,
        )

        # Timer
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.timer_callback,
        )

        self.get_logger().info('VCANPublisherNode started, publishing to /to_can_bus')

    def timer_callback(self) -> None:
        """Generate and publish CAN frames."""
        elapsed = time.time() - self._start_time

        # Get velocities based on mode
        fl_vel, fr_vel, rl_vel, rr_vel, steering = self._get_velocities(elapsed)

        # Publish front axle frame
        front_frame = self._create_wheel_frame(
            self.FRONT_AXLE_ID,
            fl_vel, fr_vel,
            steering_rad=steering,
        )
        self.can_pub.publish(front_frame)

        # Publish rear axle frame
        rear_frame = self._create_wheel_frame(
            self.REAR_AXLE_ID,
            rl_vel, rr_vel,
        )
        self.can_pub.publish(rear_frame)

        # Log periodically (every 2 seconds)
        self._msg_count += 1
        if self._msg_count % (int(self.publish_rate) * 2) == 0:
            self.get_logger().info(
                f'Wheel velocities (m/s): FL={fl_vel:.2f} FR={fr_vel:.2f} RL={rl_vel:.2f} RR={rr_vel:.2f} | Steer={steering:.2f} rad'
            )

    def _get_velocities(self, elapsed: float):
        """Get wheel velocities based on mode."""
        base = self.base_velocity

        if self.mode == 'static':
            return base, base, base, base, 0.0

        elif self.mode == 'linear':
            # Accelerate then cruise
            if elapsed < 5.0:
                speed = base * (elapsed / 5.0)
            else:
                speed = base
            return speed, speed, speed, speed, 0.0

        elif self.mode == 'circle':
            # Drive in circles with differential wheel speeds
            period = 10.0  # seconds per circle
            phase = (elapsed / period) * 2 * math.pi

            # Steering oscillates
            steering = 0.3 * math.sin(phase)  # +/- 0.3 rad (~17 deg)

            # Inner wheels slower, outer wheels faster
            inner_factor = 0.8
            outer_factor = 1.2

            if steering > 0:  # Turning left
                fl = base * inner_factor
                fr = base * outer_factor
                rl = base * inner_factor
                rr = base * outer_factor
            else:  # Turning right
                fl = base * outer_factor
                fr = base * inner_factor
                rl = base * outer_factor
                rr = base * inner_factor

            return fl, fr, rl, rr, steering

        elif self.mode == 'random':
            # Add noise to base velocity
            import random
            noise = lambda: random.uniform(-0.1, 0.1)
            return (
                base + noise(),
                base + noise(),
                base + noise(),
                base + noise(),
                0.05 * math.sin(elapsed),
            )

        else:
            self.get_logger().warning(f'Unknown mode: {self.mode}, using static')
            return base, base, base, base, 0.0

    def _create_wheel_frame(
        self,
        can_id: int,
        vel1_mps: float,
        vel2_mps: float,
        susp1_m: float = 0.05,
        susp2_m: float = 0.05,
        steering_rad: float = None,
    ) -> Frame:
        """
        Create a CAN frame matching the wheel encoder protocol.

        Protocol (8 bytes):
            Byte 0-1: Wheel 1 velocity (16-bit unsigned, mm/s, little-endian)
            Byte 2-3: Wheel 2 velocity (16-bit unsigned, mm/s, little-endian)
            Byte 4:   Suspension WSS 1 (8-bit unsigned, mm)
            Byte 5:   Suspension WSS 2 (8-bit unsigned, mm)
            Byte 6-7: Steering angle (16-bit signed, 0.1 deg units, little-endian)
                      Only valid for front axle (ID 0xB9)
        """
        # Convert velocities to mm/s (unsigned)
        vel1_mms = max(0, int(vel1_mps * 1000))
        vel2_mms = max(0, int(vel2_mps * 1000))

        # Convert suspension to mm (unsigned, clamp to 255)
        susp1_mm = min(255, max(0, int(susp1_m * 1000)))
        susp2_mm = min(255, max(0, int(susp2_m * 1000)))

        # Convert steering to decidegrees (signed)
        if steering_rad is not None:
            steering_decideg = int(math.degrees(steering_rad) * 10)
        else:
            steering_decideg = 0

        # Pack data
        data = struct.pack(
            '<HHBBh',  # Little-endian: 2 uint16, 2 uint8, 1 int16
            vel1_mms,
            vel2_mms,
            susp1_mm,
            susp2_mm,
            steering_decideg,
        )

        # Create Frame message
        frame = Frame()
        frame.header.stamp = self.get_clock().now().to_msg()
        frame.id = can_id
        frame.dlc = 8
        frame.data = list(data)

        return frame


def main(args=None):
    rclpy.init(args=args)

    node = VCANPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('VCANPublisherNode shutting down')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
