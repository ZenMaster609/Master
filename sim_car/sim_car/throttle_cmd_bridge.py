#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from std_srvs.srv import Trigger


class ThrottleCmdBridge(Node):
    def __init__(self):
        super().__init__('throttle_cmd_bridge')

        # Topics
        self.declare_parameter('input_topic', '/cmd')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('brake_cmd_topic', '/sim/brake_cmd')

        # Vehicle geometry
        self.declare_parameter('wheelbase', 1.6)

        # Input mode: 'throttle' (u in [0,1]) or 'accel' (m/s^2)
        self.declare_parameter('input_mode', 'throttle')
        # Service response for steering GUI compatibility
        self.declare_parameter('command_mode', 'throttle')

        # Dynamics limits
        self.declare_parameter('max_speed', 75.0)
        self.declare_parameter('accel_limit', 8.0)
        self.declare_parameter('brake_decel_limit', 25.0)

        # Throttle shaping / lag
        self.declare_parameter('throttle_tau', 0.25)  # seconds
        self.declare_parameter('throttle_rate_up', 2.0)  # 1/s
        self.declare_parameter('throttle_rate_down', 3.0)  # 1/s

        # Speed-dependent accel limit (linear falloff)
        self.declare_parameter('accel_falloff_speed', 30.0)  # m/s
        self.declare_parameter('accel_min_at_vmax', 1.5)  # m/s^2

        # Resistance model: a_resist = sign(v) * (c_rr + c_drag * v^2)
        self.declare_parameter('c_rr', 0.3)    # m/s^2
        self.declare_parameter('c_drag', 0.02)  # 1/m

        # Control loop
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('publish_speed_eps', 0.05)
        self.declare_parameter('publish_steer_eps', 0.02)
        self.declare_parameter('steering_speed_floor', 0.2)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.brake_cmd_topic = self.get_parameter('brake_cmd_topic').value

        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.input_mode = str(self.get_parameter('input_mode').value)
        self.command_mode = str(self.get_parameter('command_mode').value)

        self.max_speed = float(self.get_parameter('max_speed').value)
        self.accel_limit = float(self.get_parameter('accel_limit').value)
        self.brake_decel_limit = float(self.get_parameter('brake_decel_limit').value)

        self.throttle_tau = float(self.get_parameter('throttle_tau').value)
        self.throttle_rate_up = float(self.get_parameter('throttle_rate_up').value)
        self.throttle_rate_down = float(self.get_parameter('throttle_rate_down').value)

        self.accel_falloff_speed = float(self.get_parameter('accel_falloff_speed').value)
        self.accel_min_at_vmax = float(self.get_parameter('accel_min_at_vmax').value)

        self.c_rr = float(self.get_parameter('c_rr').value)
        self.c_drag = float(self.get_parameter('c_drag').value)

        self.control_rate = float(self.get_parameter('control_rate').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.publish_speed_eps = float(self.get_parameter('publish_speed_eps').value)
        self.publish_steer_eps = float(self.get_parameter('publish_steer_eps').value)
        self.steering_speed_floor = float(self.get_parameter('steering_speed_floor').value)

        if self.control_rate <= 0.0:
            self.control_rate = 50.0
        if self.input_mode != 'throttle' and self.command_mode == 'throttle':
            self.get_logger().warn(
                "command_mode is 'throttle' but input_mode is '{}'; "
                "overriding to 'acceleration' for GUI compatibility.".format(self.input_mode)
            )
            self.command_mode = 'acceleration'

        self.pub = self.create_publisher(Twist, self.output_topic, 10)
        self.sub = self.create_subscription(
            AckermannDriveStamped,
            self.input_topic,
            self._on_cmd,
            10,
        )
        self.brake_sub = self.create_subscription(
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

        self.throttle_cmd = 0.0
        self.accel_cmd = 0.0
        self.steering_cmd = 0.0
        self.brake_cmd = 0.0

        self.throttle_state = 0.0
        self.speed_state = 0.0
        self.last_wall_time = None
        self.last_publish_time = None
        self.last_pub_speed = None
        self.last_pub_steer = None

        self.timer = self.create_timer(1.0 / self.control_rate, self._control_loop)

    def _on_cmd(self, msg: AckermannDriveStamped):
        self.steering_cmd = float(msg.drive.steering_angle)
        if self.input_mode == 'accel':
            self.accel_cmd = float(msg.drive.acceleration)
        else:
            self.throttle_cmd = float(msg.drive.acceleration)

    def _on_brake_cmd(self, msg: Float32):
        self.brake_cmd = max(0.0, min(1.0, float(msg.data)))

    def _on_command_mode(self, request, response):
        response.success = True
        response.message = self.command_mode
        return response

    def _control_loop(self):
        now = time.monotonic()
        if self.last_wall_time is None:
            dt = 1.0 / self.control_rate
        else:
            dt = max(0.001, now - self.last_wall_time)
        self.last_wall_time = now

        # Filter / rate-limit throttle command
        throttle_target = max(0.0, min(1.0, self.throttle_cmd))
        if self.throttle_tau > 0.0:
            alpha = min(1.0, dt / self.throttle_tau)
            throttle_target = self.throttle_state + alpha * (throttle_target - self.throttle_state)

        rate = self.throttle_rate_up if throttle_target > self.throttle_state else self.throttle_rate_down
        max_step = max(0.0, rate * dt)
        self.throttle_state = self._move_toward(self.throttle_state, throttle_target, max_step)

        # Speed-dependent accel limit (linear falloff)
        if self.accel_falloff_speed <= 0.0:
            accel_cap = self.accel_limit
        else:
            falloff = max(0.0, min(1.0, self.speed_state / self.accel_falloff_speed))
            accel_cap = self.accel_limit - falloff * (self.accel_limit - self.accel_min_at_vmax)
            accel_cap = max(self.accel_min_at_vmax, accel_cap)

        if self.input_mode == 'accel':
            accel_cmd = max(-self.brake_decel_limit, min(self.accel_limit, self.accel_cmd))
        else:
            accel_cmd = self.throttle_state * accel_cap

        # Resistance + braking
        v = self.speed_state
        resist = (self.c_rr + self.c_drag * v * v) * (1.0 if v >= 0.0 else -1.0)
        brake = self.brake_cmd * self.brake_decel_limit

        accel_net = accel_cmd - resist - brake
        accel_net = max(-self.brake_decel_limit, min(self.accel_limit, accel_net))

        self.speed_state = max(0.0, min(self.max_speed, v + accel_net * dt))

        twist = Twist()
        twist.linear.x = self.speed_state

        steering_speed = self.speed_state
        if abs(steering_speed) < self.steering_speed_floor:
            steering_speed = math.copysign(self.steering_speed_floor, steering_speed or 1.0)

        if self.wheelbase != 0.0:
            twist.angular.z = steering_speed / self.wheelbase * math.tan(self.steering_cmd)
        else:
            twist.angular.z = 0.0

        now = time.monotonic()
        min_dt = 1.0 / self.publish_rate if self.publish_rate > 0.0 else 0.0
        if self.last_publish_time is None:
            should_publish = True
        else:
            too_soon = (now - self.last_publish_time) < min_dt
            speed_changed = (
                self.last_pub_speed is None
                or abs(twist.linear.x - self.last_pub_speed) >= self.publish_speed_eps
            )
            steer_changed = (
                self.last_pub_steer is None
                or abs(twist.angular.z - self.last_pub_steer) >= self.publish_steer_eps
            )
            should_publish = (not too_soon) and (speed_changed or steer_changed)

        if should_publish:
            self.pub.publish(twist)
            self.last_publish_time = now
            self.last_pub_speed = twist.linear.x
            self.last_pub_steer = twist.angular.z

    @staticmethod
    def _move_toward(value: float, target: float, max_delta: float) -> float:
        if value < target:
            return min(value + max_delta, target)
        if value > target:
            return max(value - max_delta, target)
        return value


def main(args=None):
    rclpy.init(args=args)
    node = ThrottleCmdBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
