#!/usr/bin/env python3

"""
Control node for the simulated car.
Supports keyboard control and automated movement patterns.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select
import time


class CarControlNode(Node):
    def __init__(self):
        super().__init__('car_control_node')

        # Declare parameters
        self.declare_parameter('mode', 'keyboard')  # 'keyboard' or 'auto'
        self.declare_parameter('linear_speed', 2.5)
        self.declare_parameter('angular_speed', 1.0)
        self.declare_parameter('ramp_time', 1.0)  # seconds to reach full input
        self.declare_parameter('hold_timeout', 0.2)  # seconds to treat key as "held"
        self.declare_parameter('loop_hz', 50.0)

        self.mode = self.get_parameter('mode').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.ramp_time = max(0.1, float(self.get_parameter('ramp_time').value))
        self.hold_timeout = max(0.05, float(self.get_parameter('hold_timeout').value))
        self.loop_hz = max(10.0, float(self.get_parameter('loop_hz').value))

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info(f'Car control node started in {self.mode} mode')
        self.get_logger().info(f'Linear speed: {self.linear_speed}, Angular speed: {self.angular_speed}')

        # Check if we have a TTY for keyboard mode
        self._has_tty = sys.stdin.isatty()

        if self.mode == 'keyboard':
            if not self._has_tty:
                self.get_logger().error('=' * 60)
                self.get_logger().error('KEYBOARD MODE REQUIRES A TERMINAL (TTY)')
                self.get_logger().error('=' * 60)
                self.get_logger().error('ros2 launch does not provide keyboard input.')
                self.get_logger().error('')
                self.get_logger().error('To use keyboard control, run in a SEPARATE terminal:')
                self.get_logger().error('  ros2 run sim_car control_node --ros-args -p mode:=keyboard')
                self.get_logger().error('')
                self.get_logger().error('Or use auto mode in launch:')
                self.get_logger().error('  ros2 launch vehicle_plotter bringup.launch.py control_mode:=auto')
                self.get_logger().error('=' * 60)
                # Don't start - let user run keyboard control separately
                raise SystemExit(1)
            else:
                self.get_logger().info('Keyboard controls:')
                self.get_logger().info('  w   : throttle (hold to accelerate)')
                self.get_logger().info('  s   : brake (release to coast)')
                self.get_logger().info('  a/d : steer left/right (hold to increase)')
                self.get_logger().info('  k   : stop (zero throttle + steering)')
                self.get_logger().info('  q   : quit')
                self.run_keyboard_control()
        elif self.mode == 'auto':
            # Automated movement (circle pattern)
            self.timer = self.create_timer(0.1, self.auto_control_callback)

    def auto_control_callback(self):
        """Automated control: drive in a circle"""
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = self.angular_speed * 0.5
        self.cmd_vel_pub.publish(twist)

    def run_keyboard_control(self):
        """Run keyboard control loop"""
        settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setraw(sys.stdin.fileno())

            twist = Twist()
            throttle = 0.0  # 0..1
            steering = 0.0  # -1..1
            throttle_intent = 0.0
            throttle_target = 0.0
            steering_target = 0.0

            last_w_time = 0.0
            last_s_time = 0.0
            last_a_time = 0.0
            last_d_time = 0.0
            last_f_time = 0.0
            last_steer_dir = 0.0
            last_throttle_input = 0.0
            should_quit = False

            last_update = time.monotonic()
            loop_timeout = 1.0 / self.loop_hz
            ramp_rate = 1.0 / self.ramp_time

            while rclpy.ok():
                now = time.monotonic()
                dt = now - last_update
                last_update = now

                # Read all available keys (non-blocking)
                while select.select([sys.stdin], [], [], 0.0)[0]:
                    key = sys.stdin.read(1)

                    if key == 'q':
                        self.get_logger().info('Quitting...')
                        should_quit = True
                        break
                    elif key == 'w':
                        last_w_time = now
                        throttle_intent = 1.0
                        last_throttle_input = now
                    elif key == 's':
                        last_s_time = now
                        throttle_intent = 0.0
                        last_throttle_input = now
                    elif key == 'a':
                        last_a_time = now
                        last_steer_dir = 1.0
                    elif key == 'd':
                        last_d_time = now
                        last_steer_dir = -1.0
                    elif key == 'f':
                        last_f_time = now
                        throttle_intent = 0.0
                        last_throttle_input = now
                    elif key == 'k':
                        throttle = 0.0
                        steering = 0.0
                        throttle_target = 0.0
                        steering_target = 0.0
                        throttle_intent = 0.0
                        self.get_logger().info('Stopped')

                if should_quit or not rclpy.ok():
                    break

                w_active = (now - last_w_time) <= self.hold_timeout
                s_active = (now - last_s_time) <= self.hold_timeout
                f_active = (now - last_f_time) <= self.hold_timeout
                a_active = (now - last_a_time) <= self.hold_timeout
                d_active = (now - last_d_time) <= self.hold_timeout

                steering_active = a_active or d_active

                if s_active or f_active:
                    throttle_intent = 0.0
                    last_throttle_input = now
                elif w_active:
                    throttle_intent = 1.0
                    last_throttle_input = now
                elif (not steering_active and
                        (now - last_throttle_input) > self.hold_timeout):
                    throttle_intent = 0.0

                throttle_target = throttle_intent

                if a_active and d_active:
                    steering_target = last_steer_dir
                elif a_active:
                    steering_target = 1.0
                elif d_active:
                    steering_target = -1.0
                else:
                    steering_target = 0.0

                throttle = self._move_toward(throttle, throttle_target, ramp_rate * dt)
                steering = self._move_toward(steering, steering_target, ramp_rate * dt)

                twist.linear.x = throttle * self.linear_speed
                twist.angular.z = steering * self.angular_speed
                self.cmd_vel_pub.publish(twist)

                # Spin once to process callbacks
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(loop_timeout)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            # Stop the car
            twist = Twist()
            self.cmd_vel_pub.publish(twist)

    @staticmethod
    def _move_toward(value: float, target: float, max_delta: float) -> float:
        if value < target:
            return min(value + max_delta, target)
        if value > target:
            return max(value - max_delta, target)
        return value


def main(args=None):
    rclpy.init(args=args)

    try:
        node = CarControlNode()

        if node.mode == 'auto':
            rclpy.spin(node)
        # For keyboard mode, the node handles its own loop

    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
