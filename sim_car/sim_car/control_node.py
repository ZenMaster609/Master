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


class CarControlNode(Node):
    def __init__(self):
        super().__init__('car_control_node')

        # Declare parameters
        self.declare_parameter('mode', 'keyboard')  # 'keyboard' or 'auto'
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 1.0)

        self.mode = self.get_parameter('mode').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info(f'Car control node started in {self.mode} mode')
        self.get_logger().info(f'Linear speed: {self.linear_speed}, Angular speed: {self.angular_speed}')

        if self.mode == 'keyboard':
            self.get_logger().info('Keyboard controls:')
            self.get_logger().info('  w/x : increase/decrease linear speed')
            self.get_logger().info('  a/d : increase/decrease angular speed')
            self.get_logger().info('  i   : move forward')
            self.get_logger().info('  ,   : move backward')
            self.get_logger().info('  j   : turn left')
            self.get_logger().info('  l   : turn right')
            self.get_logger().info('  k   : stop')
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

            while rclpy.ok():
                # Check if key is pressed
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)

                    if key == 'q':
                        self.get_logger().info('Quitting...')
                        break
                    elif key == 'w':
                        self.linear_speed = min(self.linear_speed + 0.1, 2.0)
                        self.get_logger().info(f'Linear speed: {self.linear_speed:.2f}')
                    elif key == 'x':
                        self.linear_speed = max(self.linear_speed - 0.1, 0.0)
                        self.get_logger().info(f'Linear speed: {self.linear_speed:.2f}')
                    elif key == 'a':
                        self.angular_speed = min(self.angular_speed + 0.1, 2.0)
                        self.get_logger().info(f'Angular speed: {self.angular_speed:.2f}')
                    elif key == 'd':
                        self.angular_speed = max(self.angular_speed - 0.1, 0.0)
                        self.get_logger().info(f'Angular speed: {self.angular_speed:.2f}')
                    elif key == 'i':
                        twist.linear.x = self.linear_speed
                        twist.angular.z = 0.0
                        self.get_logger().info('Moving forward')
                    elif key == ',':
                        twist.linear.x = -self.linear_speed
                        twist.angular.z = 0.0
                        self.get_logger().info('Moving backward')
                    elif key == 'j':
                        twist.linear.x = 0.0
                        twist.angular.z = self.angular_speed
                        self.get_logger().info('Turning left')
                    elif key == 'l':
                        twist.linear.x = 0.0
                        twist.angular.z = -self.angular_speed
                        self.get_logger().info('Turning right')
                    elif key == 'k':
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.get_logger().info('Stopped')

                    self.cmd_vel_pub.publish(twist)

                # Spin once to process callbacks
                rclpy.spin_once(self, timeout_sec=0.0)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            # Stop the car
            twist = Twist()
            self.cmd_vel_pub.publish(twist)


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
