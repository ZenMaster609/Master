#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from std_srvs.srv import Trigger


class AckermannCmdBridge(Node):
    def __init__(self):
        super().__init__('ackermann_cmd_bridge')

        self.declare_parameter('input_topic', '/cmd')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('wheelbase', 1.65)
        self.declare_parameter('command_mode', 'acceleration')
        self.declare_parameter('max_speed', 75.0)
        self.declare_parameter('accel_limit', 12.5)
        self.declare_parameter('brake_decel_limit', 25.0)
        self.declare_parameter('control_rate', 200.0)
        self.declare_parameter('brake_cmd_topic', '/sim/brake_cmd')
        self.declare_parameter('steering_from_desired_speed', False)
        self.declare_parameter('steering_speed_floor', 0.2)
        self.declare_parameter('steering_sign', 1.0)

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.command_mode = self.get_parameter('command_mode').get_parameter_value().string_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.accel_limit = self.get_parameter('accel_limit').get_parameter_value().double_value
        self.brake_decel_limit = self.get_parameter('brake_decel_limit').get_parameter_value().double_value
        self.control_rate = self.get_parameter('control_rate').get_parameter_value().double_value
        self.brake_cmd_topic = self.get_parameter('brake_cmd_topic').get_parameter_value().string_value
        self.steering_from_desired_speed = (
            self.get_parameter('steering_from_desired_speed').get_parameter_value().bool_value
        )
        self.steering_speed_floor = (
            self.get_parameter('steering_speed_floor').get_parameter_value().double_value
        )
        steering_sign_raw = self.get_parameter('steering_sign').get_parameter_value().double_value
        self.steering_sign = -1.0 if steering_sign_raw < 0.0 else 1.0

        self.publisher = self.create_publisher(Twist, output_topic, 10)
        self.subscription = self.create_subscription(
            AckermannDriveStamped,
            input_topic,
            self._on_cmd,
            10,
        )
        self.brake_subscription = self.create_subscription(
            Float32,
            self.brake_cmd_topic,
            self._on_brake_cmd,
            10,
        )
        self.command_mode_service = self.create_service(
            Trigger,
            '/race_car_model/command_mode',
            self._on_command_mode,
        )

        self.desired_speed = 0.0
        self.desired_accel = 0.0
        self.desired_steering = 0.0
        self.current_speed = 0.0
        self.brake_cmd = 0.0
        self.last_update_time = None
        self.last_wall_time = None

        if self.control_rate <= 0:
            self.control_rate = 50.0
        self.control_timer = self.create_timer(1.0 / self.control_rate, self._control_loop)
        self.get_logger().info(
            f'ackermann_cmd_bridge configured mode={self.command_mode} '
            f'wheelbase={self.wheelbase:.3f} steering_sign={self.steering_sign:+.0f}'
        )

    def _on_command_mode(self, request, response):
        response.success = True
        response.message = self.command_mode
        return response

    def _on_cmd(self, msg):
        if self.command_mode == 'acceleration':
            self.desired_accel = float(msg.drive.acceleration)
        else:
            self.desired_speed = float(msg.drive.speed)
        self.desired_speed = max(-self.max_speed, min(self.max_speed, self.desired_speed))
        self.desired_steering = float(msg.drive.steering_angle)

    def _on_brake_cmd(self, msg):
        self.brake_cmd = max(0.0, min(1.0, float(msg.data)))

    def _control_loop(self):
        now_wall = time.monotonic()
        if self.last_wall_time is None:
            dt = 1.0 / self.control_rate
        else:
            dt = max(0.001, now_wall - self.last_wall_time)
        self.last_wall_time = now_wall

        if self.brake_cmd > 0.0:
            max_delta = self.brake_decel_limit * self.brake_cmd * dt
            target_speed = 0.0
            self.current_speed = self._move_toward(self.current_speed, target_speed, max_delta)
        else:
            if self.command_mode == 'acceleration':
                accel_cmd = max(-self.brake_decel_limit, min(self.accel_limit, self.desired_accel))
                self.current_speed = self.current_speed + accel_cmd * dt
            else:
                target_speed = self.desired_speed
                max_delta = self.accel_limit * dt
                self.current_speed = self._move_toward(self.current_speed, target_speed, max_delta)
        self.current_speed = max(-self.max_speed, min(self.max_speed, self.current_speed))
        if self.command_mode == 'acceleration':
            self.desired_speed = self.current_speed

        steering_speed = self.current_speed
        if self.steering_from_desired_speed:
            steering_speed = self.desired_speed
        if abs(steering_speed) < self.steering_speed_floor:
            steering_speed = math.copysign(self.steering_speed_floor, steering_speed or 1.0)

        twist = Twist()
        twist.linear.x = self.current_speed
        if self.wheelbase != 0.0:
            effective_steering = self.steering_sign * self.desired_steering
            twist.angular.z = steering_speed / self.wheelbase * math.tan(effective_steering)
        else:
            twist.angular.z = 0.0

        self.publisher.publish(twist)

    @staticmethod
    def _move_toward(value: float, target: float, max_delta: float) -> float:
        if value < target:
            return min(value + max_delta, target)
        if value > target:
            return max(value - max_delta, target)
        return value


def main(args=None):
    rclpy.init(args=args)
    node = AckermannCmdBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
