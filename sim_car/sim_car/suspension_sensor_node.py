"""
Suspension Sensor Node for Formula Student Car.

Reads suspension joint positions from /sim/raw/joint_states and publishes
them as displacement values in millimeters with configurable noise and bias.
"""

import random
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from .topic_utils import apply_topic_prefix


class SuspensionSensorNode(Node):
    """
    Publishes suspension displacement with configurable noise and bias.

    Parameters:
        mode (str): 'joint_states' or 'synthetic'
        noise_stddev (float): Gaussian noise standard deviation (mm)
        bias_fl (float): Front left bias offset (mm)
        bias_fr (float): Front right bias offset (mm)
        bias_rl (float): Rear left bias offset (mm)
        bias_rr (float): Rear right bias offset (mm)
        publish_rate (float): Output rate in Hz
        static_mm (float): Static ride height offset (mm) for synthetic mode
        pitch_gain (float): Pitch gain (mm per m/s^2) for synthetic mode
        roll_gain (float): Roll gain (mm per m/s^2) for synthetic mode
        filter_tau_sec (float): Low-pass filter time constant, 0 disables

    Subscribes:
        /sim/raw/joint_states (sensor_msgs/JointState)
        /sim/raw/odom (nav_msgs/Odometry) [synthetic mode]

    Publishes:
        /sim/raw/suspension (std_msgs/Float32MultiArray) - [FL, FR, RL, RR] in mm
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
        self.declare_parameter('mode', 'synthetic')
        self.declare_parameter('noise_stddev', 0.5)  # mm
        self.declare_parameter('bias_fl', 0.0)  # mm
        self.declare_parameter('bias_fr', 0.0)  # mm
        self.declare_parameter('bias_rl', 0.0)  # mm
        self.declare_parameter('bias_rr', 0.0)  # mm
        self.declare_parameter('publish_rate', 100.0)  # Hz
        self.declare_parameter('dropout_probability', 0.0)  # 0-1, chance of missing data
        self.declare_parameter('static_mm', 20.0)
        self.declare_parameter('pitch_gain', 4.0)  # mm per m/s^2
        self.declare_parameter('roll_gain', 3.0)  # mm per m/s^2
        self.declare_parameter('filter_tau_sec', 0.0)
        self.declare_parameter('topic_prefix', '/sim/raw')

        # Get parameters
        self.mode = self.get_parameter('mode').value
        self.noise_stddev = self.get_parameter('noise_stddev').value
        self.biases = [
            self.get_parameter('bias_fl').value,
            self.get_parameter('bias_fr').value,
            self.get_parameter('bias_rl').value,
            self.get_parameter('bias_rr').value,
        ]
        self.publish_rate = self.get_parameter('publish_rate').value
        self.dropout_probability = self.get_parameter('dropout_probability').value
        self.static_mm = float(self.get_parameter('static_mm').value)
        self.pitch_gain = float(self.get_parameter('pitch_gain').value)
        self.roll_gain = float(self.get_parameter('roll_gain').value)
        self.filter_tau_sec = float(self.get_parameter('filter_tau_sec').value)
        self.topic_prefix = str(self.get_parameter('topic_prefix').value)

        # Store latest joint positions (meters, from Gazebo)
        self.suspension_positions = [0.0, 0.0, 0.0, 0.0]  # FL, FR, RL, RR
        self.joint_state_received = False
        self.odom_received = False
        self._last_speed = None
        self._last_time = None
        self._filtered_mm = None

        # Publisher
        suspension_topic = apply_topic_prefix('/sim/raw/suspension', self.topic_prefix)
        self.suspension_pub = self.create_publisher(
            Float32MultiArray, suspension_topic, 10)

        # Subscriber to joint states
        joint_states_topic = apply_topic_prefix('/sim/raw/joint_states', self.topic_prefix)
        self.joint_state_sub = self.create_subscription(
            JointState, joint_states_topic, self.joint_state_callback, 10)

        # Subscriber to odometry (synthetic mode)
        odom_topic = apply_topic_prefix('/sim/raw/odom', self.topic_prefix)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)

        # Timer for publishing at fixed rate
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_suspension)

        self.get_logger().info(
            f'Suspension sensor initialized: '
            f'mode={self.mode}, noise={self.noise_stddev}mm, rate={self.publish_rate}Hz, '
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

    def odom_callback(self, msg: Odometry):
        """Store latest odometry for synthetic suspension model."""
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            now = self.get_clock().now().nanoseconds * 1e-9

        speed = msg.twist.twist.linear.x
        yaw_rate = msg.twist.twist.angular.z

        self._current_speed = speed
        self._current_yaw_rate = yaw_rate
        self._current_time = now
        self.odom_received = True

    def publish_suspension(self):
        """Publish suspension displacement with noise and bias."""
        if self.mode == 'joint_states':
            if not self.joint_state_received:
                return
            # Convert to mm
            base_mm = [pos * 1000.0 for pos in self.suspension_positions]
        else:
            if not self.odom_received:
                return
            base_mm = self._compute_synthetic_mm()

        # Simulate dropout
        if random.random() < self.dropout_probability:
            return

        # Add bias/noise
        displacements_mm = []
        for i in range(4):
            displacement_mm = base_mm[i] + self.biases[i]
            # Add Gaussian noise
            if self.noise_stddev > 0:
                displacement_mm += random.gauss(0, self.noise_stddev)

            displacements_mm.append(float(displacement_mm))

        # Publish
        msg = Float32MultiArray()
        msg.data = displacements_mm
        self.suspension_pub.publish(msg)

    def _compute_synthetic_mm(self):
        """Compute synthetic suspension travel from odom signals."""
        now = self._current_time
        speed = self._current_speed
        yaw_rate = self._current_yaw_rate

        if self._last_time is None:
            self._last_time = now
            self._last_speed = speed
            raw_mm = [self.static_mm] * 4
            self._filtered_mm = raw_mm
            return raw_mm

        dt = max(1e-4, now - self._last_time)
        a_long = (speed - self._last_speed) / dt
        a_lat = speed * yaw_rate

        fl = self.static_mm + (-self.pitch_gain * a_long) + (self.roll_gain * a_lat)
        fr = self.static_mm + (-self.pitch_gain * a_long) + (-self.roll_gain * a_lat)
        rl = self.static_mm + (self.pitch_gain * a_long) + (self.roll_gain * a_lat)
        rr = self.static_mm + (self.pitch_gain * a_long) + (-self.roll_gain * a_lat)
        raw_mm = [fl, fr, rl, rr]

        self._last_time = now
        self._last_speed = speed

        if self.filter_tau_sec <= 0.0:
            self._filtered_mm = raw_mm
            return raw_mm

        alpha = dt / (self.filter_tau_sec + dt)
        if self._filtered_mm is None:
            self._filtered_mm = raw_mm
            return raw_mm
        self._filtered_mm = [
            (1.0 - alpha) * self._filtered_mm[i] + alpha * raw_mm[i]
            for i in range(4)
        ]
        return self._filtered_mm


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
