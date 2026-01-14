"""
Steering Angle Sensor Node for Formula Student Car.

Reads steering joint positions from /sim/joint_states and publishes
the average steering angle in degrees with configurable noise and latency.
"""

import math
import random
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32


class SteeringSensorNode(Node):
    """
    Publishes steering angle with configurable noise and latency.

    Parameters:
        noise_stddev (float): Gaussian noise standard deviation (degrees)
        latency_ms (float): Artificial latency in milliseconds
        publish_rate (float): Output rate in Hz

    Subscribes:
        /sim/joint_states (sensor_msgs/JointState)

    Publishes:
        /sim/steering_angle (std_msgs/Float32) - steering angle in degrees
    """

    # Joint names for steering
    STEERING_JOINTS = ['steering_fl_joint', 'steering_fr_joint']

    def __init__(self):
        super().__init__('steering_sensor_node')

        # Declare parameters
        self.declare_parameter('noise_stddev', 0.1)  # degrees
        self.declare_parameter('latency_ms', 5.0)  # milliseconds
        self.declare_parameter('publish_rate', 100.0)  # Hz
        self.declare_parameter('bias', 0.0)  # degrees
        self.declare_parameter('dropout_probability', 0.0)  # 0-1

        # Get parameters
        self.noise_stddev = self.get_parameter('noise_stddev').value
        self.latency_ms = self.get_parameter('latency_ms').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.bias = self.get_parameter('bias').value
        self.dropout_probability = self.get_parameter('dropout_probability').value

        # Latency buffer (stores (timestamp, value) pairs)
        # Buffer size based on latency and publish rate
        buffer_size = max(1, int(self.latency_ms / 1000.0 * self.publish_rate) + 1)
        self.latency_buffer = deque(maxlen=buffer_size)

        # Store latest steering positions (radians)
        self.steering_fl = 0.0
        self.steering_fr = 0.0
        self.joint_state_received = False

        # Publisher
        self.steering_pub = self.create_publisher(
            Float32, '/sim/steering_angle', 10)

        # Subscriber
        self.joint_state_sub = self.create_subscription(
            JointState, '/sim/joint_states', self.joint_state_callback, 10)

        # Timer for publishing at fixed rate
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_steering)

        self.get_logger().info(
            f'Steering sensor initialized: '
            f'noise={self.noise_stddev}deg, latency={self.latency_ms}ms, '
            f'rate={self.publish_rate}Hz, bias={self.bias}deg'
        )

    def joint_state_callback(self, msg: JointState):
        """Extract steering joint positions from joint state message."""
        for joint_name in self.STEERING_JOINTS:
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                if idx < len(msg.position):
                    if joint_name == 'steering_fl_joint':
                        self.steering_fl = msg.position[idx]
                    elif joint_name == 'steering_fr_joint':
                        self.steering_fr = msg.position[idx]

        self.joint_state_received = True

    def publish_steering(self):
        """Publish steering angle with noise and latency."""
        if not self.joint_state_received:
            return

        # Get current time
        now = self.get_clock().now().nanoseconds / 1e9

        # Compute average steering angle (use average of both front wheels)
        # Convert from radians to degrees
        avg_steering_rad = (self.steering_fl + self.steering_fr) / 2.0
        avg_steering_deg = math.degrees(avg_steering_rad)

        # Add to latency buffer
        self.latency_buffer.append((now, avg_steering_deg))

        # Check for dropout
        if random.random() < self.dropout_probability:
            return

        # Get delayed value from buffer
        target_time = now - (self.latency_ms / 1000.0)

        # Find the value closest to target_time
        delayed_value = avg_steering_deg  # Fallback to current
        for timestamp, value in self.latency_buffer:
            if timestamp <= target_time:
                delayed_value = value
            else:
                break

        # Add bias
        delayed_value += self.bias

        # Add Gaussian noise
        if self.noise_stddev > 0:
            delayed_value += random.gauss(0, self.noise_stddev)

        # Publish
        msg = Float32()
        msg.data = float(delayed_value)
        self.steering_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SteeringSensorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
