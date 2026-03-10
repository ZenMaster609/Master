"""Base class for single-topic virtual sensor nodes."""

import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from .virtual_sensors_model import VirtualSensorsModel
from .topic_utils import apply_topic_prefix


class VirtualSensorNodeBase(Node):
    """Base node for a single virtual sensor output."""

    def __init__(
        self,
        *,
        node_name: str,
        publish_topic: str,
        noise_param: str,
        noise_default: float,
        publish_rate_default: float = 10.0,
        needs_brake_cmd: bool = False,
    ):
        super().__init__(node_name)

        self.declare_parameter('publish_rate', publish_rate_default)
        self.declare_parameter('ambient_temp', 25.0)
        self.declare_parameter(noise_param, noise_default)
        self.declare_parameter('topic_prefix', '/sim/raw')
        if needs_brake_cmd:
            self.declare_parameter('brake_cmd_topic', '/sim/brake_cmd')

        self.publish_rate = float(self.get_parameter('publish_rate').value)
        if self.publish_rate <= 0.0:
            self.get_logger().warn(
                f'publish_rate <= 0 for {node_name}; falling back to {publish_rate_default}Hz'
            )
            self.publish_rate = publish_rate_default if publish_rate_default > 0.0 else 10.0

        self.ambient_temp = float(self.get_parameter('ambient_temp').value)
        self.noise_stddev = float(self.get_parameter(noise_param).value)
        self.topic_prefix = str(self.get_parameter('topic_prefix').value)

        self.model = VirtualSensorsModel(ambient_temp=self.ambient_temp)

        publish_topic = apply_topic_prefix(publish_topic, self.topic_prefix)
        self.publisher = self.create_publisher(Float32, publish_topic, 10)
        self.odom_sub = self.create_subscription(
            Odometry, apply_topic_prefix('/sim/raw/odom', self.topic_prefix), self.odom_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        if needs_brake_cmd:
            brake_topic = self.get_parameter('brake_cmd_topic').value
            self.brake_cmd_sub = self.create_subscription(
                Float32, brake_topic, self.brake_cmd_callback, 10
            )
        else:
            self.brake_cmd_sub = None

        self.timer = self.create_timer(1.0 / self.publish_rate, self._on_timer)

        self.get_logger().info(
            f'{node_name} publishing {publish_topic} at {self.publish_rate}Hz'
        )

    def odom_callback(self, msg: Odometry) -> None:
        twist = msg.twist.twist
        self.model.update_vehicle_speed(twist.linear.x, twist.linear.y)

    def cmd_vel_callback(self, msg: Twist) -> None:
        self.model.update_cmd_vel(msg.linear.x)

    def brake_cmd_callback(self, msg: Float32) -> None:
        self.model.update_brake_cmd(msg.data)

    def _on_timer(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        self.model.step(now, 1.0 / self.publish_rate)
        value = self.compute_value()
        self._publish_float(value, self.noise_stddev)

    def compute_value(self) -> float:
        raise NotImplementedError

    def _publish_float(self, value: float, noise_stddev: float) -> None:
        if noise_stddev > 0.0:
            value += random.gauss(0.0, noise_stddev)
        msg = Float32()
        msg.data = float(value)
        self.publisher.publish(msg)


def spin_node(node_factory, args=None) -> None:
    rclpy.init(args=args)
    node = node_factory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
