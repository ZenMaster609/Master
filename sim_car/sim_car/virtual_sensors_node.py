"""
Virtual Sensors Node for Formula Student Car.

Simulates various sensors that don't have physical representation in Gazebo:
- Cooling system (water pressure, flow, temperatures)
- Brake temperatures (front-right, rear-left)
- Pitot tube (dynamic pressure / airflow)

All models are physics-inspired approximations driven by vehicle speed
and control inputs.
"""

import random

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

from .virtual_sensors_model import VirtualSensorsModel


class VirtualSensorsNode(Node):
    """
    Simulates cooling, brake, and pitot tube sensors.

    Parameters:
        publish_rate (float): Output rate in Hz (default: 10)
        ambient_temp (float): Ambient temperature in Celsius
        noise_* (float): Noise standard deviation for each sensor

    Subscribes:
        /sim/odom (nav_msgs/Odometry): For vehicle speed
        /cmd_vel (geometry_msgs/Twist): For throttle proxy (and brake fallback)
        /sim/brake_cmd (std_msgs/Float32): For explicit braking command (0..1)

    Publishes:
        /sim/cooling/water_pressure (Float32) - bar
        /sim/cooling/water_flow (Float32) - L/min
        /sim/cooling/water_temp_in (Float32) - Celsius
        /sim/cooling/water_temp_out (Float32) - Celsius
        /sim/cooling/water_temp_radiator (Float32) - Celsius
        /sim/brakes/temp_fr (Float32) - Celsius
        /sim/brakes/temp_rl (Float32) - Celsius
        /sim/pitot/dynamic_pressure (Float32) - Pa
    """

    def __init__(self):
        super().__init__('virtual_sensors_node')

        # Declare parameters
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('ambient_temp', 25.0)  # Celsius

        # Per-sensor publish rates (Hz)
        self.declare_parameter('publish_rate_water_pressure', self.get_parameter('publish_rate').value)
        self.declare_parameter('publish_rate_water_flow', self.get_parameter('publish_rate').value)
        self.declare_parameter('publish_rate_water_temp_in', self.get_parameter('publish_rate').value)
        self.declare_parameter('publish_rate_water_temp_out', self.get_parameter('publish_rate').value)
        self.declare_parameter('publish_rate_water_temp_radiator', self.get_parameter('publish_rate').value)
        self.declare_parameter('publish_rate_brake_temp_fr', self.get_parameter('publish_rate').value)
        self.declare_parameter('publish_rate_brake_temp_rl', self.get_parameter('publish_rate').value)
        self.declare_parameter('publish_rate_pitot_dynamic_pressure', self.get_parameter('publish_rate').value)
        self.declare_parameter('brake_cmd_topic', '/sim/brake_cmd')

        # Noise parameters
        self.declare_parameter('noise_pressure', 0.02)  # bar
        self.declare_parameter('noise_flow', 0.5)  # L/min
        self.declare_parameter('noise_temp', 0.3)  # Celsius
        self.declare_parameter('noise_brake_temp', 1.0)  # Celsius
        self.declare_parameter('noise_pitot', 2.0)  # Pa

        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.ambient_temp = self.get_parameter('ambient_temp').value
        self.noise_pressure = self.get_parameter('noise_pressure').value
        self.noise_flow = self.get_parameter('noise_flow').value
        self.noise_temp = self.get_parameter('noise_temp').value
        self.noise_brake_temp = self.get_parameter('noise_brake_temp').value
        self.noise_pitot = self.get_parameter('noise_pitot').value

        self.publish_rates = {
            'water_pressure': self.get_parameter('publish_rate_water_pressure').value,
            'water_flow': self.get_parameter('publish_rate_water_flow').value,
            'water_temp_in': self.get_parameter('publish_rate_water_temp_in').value,
            'water_temp_out': self.get_parameter('publish_rate_water_temp_out').value,
            'water_temp_radiator': self.get_parameter('publish_rate_water_temp_radiator').value,
            'brake_temp_fr': self.get_parameter('publish_rate_brake_temp_fr').value,
            'brake_temp_rl': self.get_parameter('publish_rate_brake_temp_rl').value,
            'pitot_dynamic_pressure': self.get_parameter('publish_rate_pitot_dynamic_pressure').value,
        }
        self.update_rate_hz = max(self.publish_rates.values()) if self.publish_rates else self.publish_rate
        if self.update_rate_hz <= 0:
            self.update_rate_hz = self.publish_rate

        self.model = VirtualSensorsModel(ambient_temp=self.ambient_temp)

        # Time tracking
        self.last_publish_times = {}

        # Publishers - Cooling
        self.pressure_pub = self.create_publisher(
            Float32, '/sim/cooling/water_pressure', 10)
        self.flow_pub = self.create_publisher(
            Float32, '/sim/cooling/water_flow', 10)
        self.temp_in_pub = self.create_publisher(
            Float32, '/sim/cooling/water_temp_in', 10)
        self.temp_out_pub = self.create_publisher(
            Float32, '/sim/cooling/water_temp_out', 10)
        self.temp_rad_pub = self.create_publisher(
            Float32, '/sim/cooling/water_temp_radiator', 10)

        # Publishers - Brakes
        self.brake_fr_pub = self.create_publisher(
            Float32, '/sim/brakes/temp_fr', 10)
        self.brake_rl_pub = self.create_publisher(
            Float32, '/sim/brakes/temp_rl', 10)

        # Publishers - Pitot
        self.pitot_pub = self.create_publisher(
            Float32, '/sim/pitot/dynamic_pressure', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/sim/odom', self.odom_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        brake_topic = self.get_parameter('brake_cmd_topic').value
        self.brake_cmd_sub = self.create_subscription(
            Float32, brake_topic, self.brake_cmd_callback, 10)

        # Timer for publishing
        timer_period = 1.0 / self.update_rate_hz
        self.timer = self.create_timer(timer_period, self.update_and_publish)

        self.get_logger().info(
            f'Virtual sensors initialized: '
            f'update_rate={self.update_rate_hz}Hz, ambient={self.ambient_temp}C'
        )

    def odom_callback(self, msg: Odometry):
        """Extract vehicle speed from odometry."""
        twist = msg.twist.twist
        self.model.update_vehicle_speed(twist.linear.x, twist.linear.y)

    def cmd_vel_callback(self, msg: Twist):
        """Extract throttle/brake proxy from velocity command."""
        self.model.update_cmd_vel(msg.linear.x)

    def brake_cmd_callback(self, msg: Float32):
        """Explicit braking command (0..1)."""
        self.model.update_brake_cmd(msg.data)

    def update_and_publish(self):
        """Update thermal models and publish all sensor values."""
        now = self.get_clock().now().nanoseconds / 1e9
        self.model.step(now, 1.0 / self.update_rate_hz)

        # Publish cooling sensors
        if self._should_publish('water_pressure', now):
            self._publish_float(self.pressure_pub, self.model.compute_water_pressure(),
                                self.noise_pressure)
        if self._should_publish('water_flow', now):
            self._publish_float(self.flow_pub, self.model.compute_water_flow(),
                                self.noise_flow)
        if self._should_publish('water_temp_in', now):
            self._publish_float(self.temp_in_pub, self.model.water_temp_in, self.noise_temp)
        if self._should_publish('water_temp_out', now):
            self._publish_float(self.temp_out_pub, self.model.water_temp_out, self.noise_temp)
        if self._should_publish('water_temp_radiator', now):
            self._publish_float(self.temp_rad_pub, self.model.water_temp_radiator, self.noise_temp)

        # Publish brake temps
        if self._should_publish('brake_temp_fr', now):
            self._publish_float(self.brake_fr_pub, self.model.brake_temp_fr, self.noise_brake_temp)
        if self._should_publish('brake_temp_rl', now):
            self._publish_float(self.brake_rl_pub, self.model.brake_temp_rl, self.noise_brake_temp)

        # Publish pitot
        if self._should_publish('pitot_dynamic_pressure', now):
            self._publish_float(self.pitot_pub, self.model.compute_pitot_pressure(),
                                self.noise_pitot)

    def _publish_float(self, publisher, value: float, noise_stddev: float):
        """Publish a Float32 with optional noise."""
        if noise_stddev > 0:
            value += random.gauss(0, noise_stddev)
        msg = Float32()
        msg.data = float(value)
        publisher.publish(msg)

    def _should_publish(self, sensor_name: str, now: float) -> bool:
        rate = self.publish_rates.get(sensor_name, 0.0)
        if rate <= 0:
            return False
        last_time = self.last_publish_times.get(sensor_name)
        if last_time is None or (now - last_time) >= (1.0 / rate):
            self.last_publish_times[sensor_name] = now
            return True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = VirtualSensorsNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
