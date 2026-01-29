"""
Virtual Sensors Node for Formula Student Car.

Simulates various sensors that don't have physical representation in Gazebo:
- Cooling system (water pressure, flow, temperatures)
- Brake temperatures (front-right, rear-left)
- Pitot tube (dynamic pressure / airflow)

All models are physics-inspired approximations driven by vehicle speed
and control inputs.
"""

import math
import random
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


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

    # Air density at sea level (kg/m^3)
    AIR_DENSITY = 1.225

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

        # State variables
        self.vehicle_speed = 0.0  # m/s
        self.throttle = 0.0  # 0-1 proxy
        self.braking = 0.0  # 0-1 proxy (negative cmd_vel.linear.x)
        self.brake_cmd = 0.0  # 0-1 explicit brake command
        self.prev_speed = 0.0

        # Thermal state (temperatures)
        self.water_temp_in = self.ambient_temp
        self.water_temp_out = self.ambient_temp
        self.water_temp_radiator = self.ambient_temp
        self.brake_temp_fr = self.ambient_temp
        self.brake_temp_rl = self.ambient_temp

        # Time tracking
        self.last_update_time = None
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
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.vehicle_speed = math.sqrt(vx**2 + vy**2)

    def cmd_vel_callback(self, msg: Twist):
        """Extract throttle/brake proxy from velocity command."""
        # Positive linear.x = throttle, negative = braking
        if msg.linear.x > 0:
            self.throttle = min(msg.linear.x / 10.0, 1.0)  # Normalize to 0-1
            self.braking = 0.0
        else:
            self.throttle = 0.0
            self.braking = min(abs(msg.linear.x) / 5.0, 1.0)  # Normalize to 0-1

    def brake_cmd_callback(self, msg: Float32):
        """Explicit braking command (0..1)."""
        self.brake_cmd = max(0.0, min(1.0, float(msg.data)))

    def update_and_publish(self):
        """Update thermal models and publish all sensor values."""
        # Get time delta
        now = self.get_clock().now().nanoseconds / 1e9
        if self.last_update_time is None:
            dt = 1.0 / self.update_rate_hz
        else:
            dt = now - self.last_update_time
        self.last_update_time = now

        # Clamp dt to avoid instability
        dt = min(dt, 0.5)

        # Update thermal models
        self._update_cooling_system(dt)
        self._update_brake_temps(dt)

        # Publish cooling sensors
        if self._should_publish('water_pressure', now):
            self._publish_float(self.pressure_pub, self._compute_water_pressure(),
                                self.noise_pressure)
        if self._should_publish('water_flow', now):
            self._publish_float(self.flow_pub, self._compute_water_flow(),
                                self.noise_flow)
        if self._should_publish('water_temp_in', now):
            self._publish_float(self.temp_in_pub, self.water_temp_in, self.noise_temp)
        if self._should_publish('water_temp_out', now):
            self._publish_float(self.temp_out_pub, self.water_temp_out, self.noise_temp)
        if self._should_publish('water_temp_radiator', now):
            self._publish_float(self.temp_rad_pub, self.water_temp_radiator, self.noise_temp)

        # Publish brake temps
        if self._should_publish('brake_temp_fr', now):
            self._publish_float(self.brake_fr_pub, self.brake_temp_fr, self.noise_brake_temp)
        if self._should_publish('brake_temp_rl', now):
            self._publish_float(self.brake_rl_pub, self.brake_temp_rl, self.noise_brake_temp)

        # Publish pitot
        if self._should_publish('pitot_dynamic_pressure', now):
            self._publish_float(self.pitot_pub, self._compute_pitot_pressure(),
                                self.noise_pitot)

        # Store for deceleration calculation
        self.prev_speed = self.vehicle_speed

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

    def _compute_water_pressure(self) -> float:
        """
        Compute water pump pressure.

        Model: Base pressure + speed-dependent component (pump driven by engine)
        """
        # Base pressure at idle
        base_pressure = 1.5  # bar

        # Speed-dependent (pump speed ~ engine RPM ~ wheel speed)
        speed_factor = 0.05 * self.vehicle_speed  # bar per m/s

        # Throttle adds some pressure (higher RPM)
        throttle_factor = 0.3 * self.throttle

        return base_pressure + speed_factor + throttle_factor

    def _compute_water_flow(self) -> float:
        """
        Compute coolant flow rate.

        Model: Base flow + speed-dependent component
        """
        # Base flow at idle
        base_flow = 30.0  # L/min

        # Speed-dependent
        speed_factor = 2.0 * self.vehicle_speed  # L/min per m/s

        # Throttle increases flow
        throttle_factor = 15.0 * self.throttle

        return base_flow + speed_factor + throttle_factor

    def _update_cooling_system(self, dt: float):
        """
        Update cooling system temperatures using simple thermal model.

        Heat sources:
        - Engine (proxy: throttle * speed)

        Heat sinks:
        - Radiator (proportional to airflow ~ speed)
        - Ambient (slow convection)
        """
        # Time constants
        tau_heat = 30.0  # seconds to heat up
        tau_cool = 60.0  # seconds to cool down

        # Heat generation (throttle * speed proxy)
        heat_generation = self.throttle * self.vehicle_speed * 2.0  # degrees/s

        # Radiator cooling (airflow ~ speed)
        radiator_cooling = 0.1 * self.vehicle_speed * (self.water_temp_radiator - self.ambient_temp)

        # Ambient cooling (slow)
        ambient_cooling_rate = 0.01  # 1/s

        # Update water_temp_out (hottest point, after engine)
        target_out = self.ambient_temp + 60.0 + 20.0 * self.throttle
        self.water_temp_out += dt * (
            heat_generation
            - (self.water_temp_out - self.water_temp_in) / tau_cool
        )
        self.water_temp_out = max(self.ambient_temp, min(120.0, self.water_temp_out))

        # Update water_temp_radiator (after radiator, before pump)
        self.water_temp_radiator += dt * (
            (self.water_temp_out - self.water_temp_radiator) / tau_heat
            - radiator_cooling / 10.0
            - ambient_cooling_rate * (self.water_temp_radiator - self.ambient_temp)
        )
        self.water_temp_radiator = max(self.ambient_temp, min(110.0, self.water_temp_radiator))

        # Update water_temp_in (after radiator, coolest point)
        self.water_temp_in += dt * (
            (self.water_temp_radiator - self.water_temp_in) / tau_heat
            - ambient_cooling_rate * (self.water_temp_in - self.ambient_temp)
        )
        self.water_temp_in = max(self.ambient_temp, min(100.0, self.water_temp_in))

    def _update_brake_temps(self, dt: float):
        """
        Update brake temperatures.

        Heat sources:
        - Braking (kinetic energy -> heat)

        Heat sinks:
        - Airflow (proportional to speed)
        - Ambient (slow convection)
        """
        # Use explicit brake command when available, fallback to cmd_vel proxy
        brake_input = max(self.brake_cmd, self.braking)

        # Compute deceleration proxy
        decel = max(0, (self.prev_speed - self.vehicle_speed) / max(dt, 0.001))
        speed_for_heating = max(self.vehicle_speed, self.prev_speed)
        effective_brake = brake_input if (decel > 0.0 and speed_for_heating > 0.2) else 0.0

        # Heat generation from braking (kinetic energy dissipation)
        # Q ~ v * decel (simplified power dissipation)
        heat_fr = speed_for_heating * decel * 0.5 * effective_brake  # Front takes more braking
        heat_rl = speed_for_heating * decel * 0.3 * effective_brake  # Rear takes less

        # Add explicit braking input only when decelerating
        heat_fr += effective_brake * 30.0
        heat_rl += effective_brake * 20.0

        # Cooling from airflow
        airflow_cooling = 0.02 * self.vehicle_speed

        # Ambient cooling (convection)
        ambient_cooling = 0.005

        # Update front-right brake
        self.brake_temp_fr += dt * (
            heat_fr
            - airflow_cooling * (self.brake_temp_fr - self.ambient_temp)
            - ambient_cooling * (self.brake_temp_fr - self.ambient_temp)
        )
        self.brake_temp_fr = max(self.ambient_temp, min(400.0, self.brake_temp_fr))

        # Update rear-left brake
        self.brake_temp_rl += dt * (
            heat_rl
            - airflow_cooling * (self.brake_temp_rl - self.ambient_temp)
            - ambient_cooling * (self.brake_temp_rl - self.ambient_temp)
        )
        self.brake_temp_rl = max(self.ambient_temp, min(400.0, self.brake_temp_rl))

    def _compute_pitot_pressure(self) -> float:
        """
        Compute pitot tube dynamic pressure.

        q = 0.5 * rho * v^2

        Where:
        - q = dynamic pressure (Pa)
        - rho = air density (~1.225 kg/m^3 at sea level)
        - v = airspeed (m/s)

        We assume airspeed = vehicle speed (no wind).
        """
        # Dynamic pressure formula
        q = 0.5 * self.AIR_DENSITY * self.vehicle_speed**2

        # Add low-speed bias (sensor offset)
        if self.vehicle_speed < 1.0:
            q += random.uniform(-2, 2)  # Low-speed noise

        return max(0, q)


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
