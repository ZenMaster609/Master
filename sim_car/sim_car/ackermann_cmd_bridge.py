#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger


class AckermannCmdBridge(Node):
    def __init__(self):
        super().__init__('ackermann_cmd_bridge')

        self.declare_parameter('input_topic', '/cmd')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('wheelbase', 1.6)
        self.declare_parameter('command_mode', 'velocity')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.command_mode = self.get_parameter('command_mode').get_parameter_value().string_value

        self.publisher = self.create_publisher(Twist, output_topic, 10)
        self.subscription = self.create_subscription(
            AckermannDriveStamped,
            input_topic,
            self._on_cmd,
            10,
        )
        self.command_mode_service = self.create_service(
            Trigger,
            '/race_car_model/command_mode',
            self._on_command_mode,
        )

    def _on_command_mode(self, request, response):
        response.success = True
        response.message = self.command_mode
        return response

    def _on_cmd(self, msg):
        speed = msg.drive.speed
        steering_angle = msg.drive.steering_angle

        twist = Twist()
        twist.linear.x = speed
        if self.wheelbase != 0.0:
            twist.angular.z = speed / self.wheelbase * math.tan(steering_angle)
        else:
            twist.angular.z = 0.0

        self.publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = AckermannCmdBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
