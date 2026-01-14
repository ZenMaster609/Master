"""
Suspension Sensor Node for Formula Student Car.

Reads suspension joint positions from /sim/joint_states and publishes
them as displacement values in millimeters with configurable noise and bias.
"""

import random
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


class SuspensionSensorNode(Node):
    """
    Publishes suspension displacement with configurable noise and bias.

    Parameters:
        noise_stddev (float): Gaussian noise standard deviation (mm)
        bias_fl (float): Front left bias offset (mm)
        bias_fr (float): Front right bias offset (mm)
        bias_rl (float): Rear left bias offset (mm)
        bias_rr (float): Rear right bias offset (mm)
        publish_rate (float): Output rate in Hz

    Subscribes:
        /sim/joint_states (sensor_msgs/JointState)

    Publishes:
        /sim/suspension (std_msgs/Float32MultiArray) - [FL, FR, RL, RR] in mm
    """

    # Joint names in the URDF
    SUSPENSION_JOINTS = [
        'suspension_fl_joint',
        'suspension_fr_joint',
        'suspension_rl_joint',
        'suspension_rr_joint',
    ]

    def __init__(self):
        super().__init__('suspension_sensor_node')

        # Declare parameters
        self.declare_parameter('noise_stddev', 0.5)  # mm
        self.declare_parameter('bias_fl', 0.0)  # mm
        self.declare_parameter('bias_fr', 0.0)  # mm
        self.declare_parameter('bias_rl', 0.0)  # mm
        self.declare_parameter('bias_rr', 0.0)  # mm
        self.declare_parameter('publish_rate', 100.0)  # Hz
        self.declare_parameter('dropout_probability', 0.0)  # 0-1, chance of missing data

        # Get parameters
        self.noise_stddev = self.get_parameter('noise_stddev').value
        self.biases = [
            self.get_parameter('bias_fl').value,
            self.get_parameter('bias_fr').value,
            self.get_parameter('bias_rl').value,
            self.get_parameter('bias_rr').value,
        ]
        self.publish_rate = self.get_parameter('publish_rate').value
        self.dropout_probability = self.get_parameter('dropout_probability').value

        # Store latest joint positions (meters, from Gazebo)
        self.suspension_positions = [0.0, 0.0, 0.0, 0.0]  # FL, FR, RL, RR
        self.joint_state_received = False

        # Publisher
        self.suspension_pub = self.create_publisher(
            Float32MultiArray, '/sim/suspension', 10)

        # Subscriber to joint states
        self.joint_state_sub = self.create_subscription(
            JointState, '/sim/joint_states', self.joint_state_callback, 10)

        # Timer for publishing at fixed rate
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_suspension)

        self.get_logger().info(
            f'Suspension sensor initialized: '
            f'noise={self.noise_stddev}mm, rate={self.publish_rate}Hz, '
            f'biases=[{self.biases[0]:.1f}, {self.biases[1]:.1f}, '
            f'{self.biases[2]:.1f}, {self.biases[3]:.1f}]mm'
        )

    def joint_state_callback(self, msg: JointState):
        """Extract suspension joint positions from joint state message."""
        for i, joint_name in enumerate(self.SUSPENSION_JOINTS):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                if idx < len(msg.position):
                    # Position is in meters from Gazebo
                    self.suspension_positions[i] = msg.position[idx]

        self.joint_state_received = True

    def publish_suspension(self):
        """Publish suspension displacement with noise and bias."""
        if not self.joint_state_received:
            return

        # Simulate dropout
        if random.random() < self.dropout_probability:
            return

        # Convert to mm and add noise/bias
        displacements_mm = []
        for i in range(4):
            # Convert meters to mm
            displacement_mm = self.suspension_positions[i] * 1000.0

            # Add bias
            displacement_mm += self.biases[i]

            # Add Gaussian noise
            if self.noise_stddev > 0:
                displacement_mm += random.gauss(0, self.noise_stddev)

            displacements_mm.append(float(displacement_mm))

        # Publish
        msg = Float32MultiArray()
        msg.data = displacements_mm
        self.suspension_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SuspensionSensorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
