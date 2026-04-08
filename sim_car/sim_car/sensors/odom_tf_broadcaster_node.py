#!/usr/bin/env python3
"""Broadcast odom -> base transform from nav_msgs/Odometry."""

from __future__ import annotations

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class OdomTfBroadcasterNode(Node):
    """Republish /sim/odom as a TF transform for RViz and TF consumers."""

    def __init__(self) -> None:
        super().__init__('odom_tf_broadcaster_node')
        self.declare_parameter('odom_topic', '/sim/odom')
        self.declare_parameter('default_frame_id', 'odom')
        self.declare_parameter('default_child_frame_id', 'base_footprint')

        self.odom_topic = str(self.get_parameter('odom_topic').value).strip() or '/sim/odom'
        self.default_frame_id = str(self.get_parameter('default_frame_id').value).strip() or 'odom'
        self.default_child_frame_id = str(self.get_parameter('default_child_frame_id').value).strip() or 'base_footprint'

        self._broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos_profile_sensor_data)
        self.get_logger().info(f'odom_tf_broadcaster_node ready odom_topic={self.odom_topic}')

    def _odom_cb(self, msg: Odometry) -> None:
        transform = TransformStamped()
        transform.header = msg.header
        transform.header.frame_id = str(msg.header.frame_id).strip() or self.default_frame_id
        transform.child_frame_id = str(msg.child_frame_id).strip() or self.default_child_frame_id
        transform.transform.translation.x = float(msg.pose.pose.position.x)
        transform.transform.translation.y = float(msg.pose.pose.position.y)
        transform.transform.translation.z = float(msg.pose.pose.position.z)
        transform.transform.rotation = msg.pose.pose.orientation
        self._broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomTfBroadcasterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
