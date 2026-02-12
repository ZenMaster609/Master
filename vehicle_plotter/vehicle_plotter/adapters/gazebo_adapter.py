"""
GazeboAdapter - Sensor adapter for Gazebo Fortress simulation.

Subscribes to Gazebo-bridged sensor topics and converts them
to unified VehicleState messages.
"""

from typing import Optional, Dict, Any
import math

from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Float32

from ..core.sensor_interfaces import SensorAdapterInterface
from ..core.vehicle_state import VehicleState
from ..core.time_sync import TimeSynchronizer, interpolate_odometry
from ..core.qos_profiles import SENSOR_QOS, RELIABLE_SENSOR_QOS
from ..utils.transforms import (
    quaternion_to_yaw,
    gps_to_local,
    set_gps_origin,
    get_gps_origin,
)


class GazeboAdapter(SensorAdapterInterface):
    """
    Sensor adapter for Gazebo Fortress simulation.

    Subscribes to:
    - /imu/data (sensor_msgs/Imu) - IMU data at ~100Hz
    - /gps/fix (sensor_msgs/NavSatFix) - GPS data at ~5Hz
    - /odom (nav_msgs/Odometry) - Odometry at ~50Hz
    - /wheel_encoder/rpm (std_msgs/Float32MultiArray) - Wheel speeds in RPM
    - /wheel_encoder/speed_mm_s (std_msgs/Float32MultiArray) - Wheel speeds in mm/s
    - /wheel_encoder/angle_accum (std_msgs/Float32MultiArray) - Angle accumulation per wheel (rad)

    Topic names are configurable via ROS parameters.
    """

    # Default topic names (can be remapped via launch file)
    # All simulation topics use /sim/ namespace
    DEFAULT_TOPICS = {
        'imu': '/sim/imu',                              # From ros_gz_bridge
        'gps': '/sim/navsat',                           # From ros_gz_bridge
        'odom': '/sim/odom',                            # From ros_gz_bridge
        'encoder_velocities': '/sim/wheel_encoder/rpm',  # From wheel_encoder_node
        'encoder_speeds_mm_s': '/sim/wheel_encoder/speed_mm_s',  # From wheel_encoder_node
        'encoder_angle_accum': '/sim/wheel_encoder/angle_accum',  # From wheel_encoder_node
        'suspension': '/sim/suspension',                # From suspension_sensor_node
        'steering_angle': '/sim/steering_angle',        # From steering_sensor_node
        # Virtual sensors from virtual_sensors_node
        'water_pressure': '/sim/cooling/water_pressure',
        'water_flow': '/sim/cooling/water_flow',
        'water_temp_in': '/sim/cooling/water_temp_in',
        'water_temp_out': '/sim/cooling/water_temp_out',
        'water_temp_radiator': '/sim/cooling/water_temp_radiator',
        'brake_temp_fr': '/sim/brakes/temp_fr',
        'brake_temp_rl': '/sim/brakes/temp_rl',
        'pitot_pressure': '/sim/pitot/dynamic_pressure',
    }

    # Sensor rates (Hz) - used for time sync configuration
    SENSOR_RATES = {
        'imu': 100.0,
        'gps': 5.0,
        'odom': 50.0,
        'encoder_velocities': 50.0,
        'encoder_speeds_mm_s': 50.0,
        'encoder_angle_accum': 50.0,
        'suspension': 100.0,
        'steering_angle': 100.0,
        'water_pressure': 10.0,
        'water_flow': 10.0,
        'water_temp_in': 10.0,
        'water_temp_out': 10.0,
        'water_temp_radiator': 10.0,
        'brake_temp_fr': 10.0,
        'brake_temp_rl': 10.0,
        'pitot_pressure': 10.0,
    }

    def __init__(
        self,
        node: Node,
        synchronizer: TimeSynchronizer,
        topics: Optional[Dict[str, str]] = None,
        auto_set_gps_origin: bool = True,
    ):
        """
        Initialize the Gazebo adapter.

        Args:
            node: ROS 2 node for subscriptions
            synchronizer: Time synchronizer
            topics: Optional topic name overrides
            auto_set_gps_origin: If True, set GPS origin from first GPS message
        """
        super().__init__(node, synchronizer, adapter_name="gazebo")

        self.topics = {**self.DEFAULT_TOPICS, **(topics or {})}
        self.auto_set_gps_origin = auto_set_gps_origin
        self._gps_origin_set = False
        self.wheel_radius = self.node.declare_parameter('wheel_radius', 0.23).value

        # Latest raw sensor data (for debugging)
        self._last_imu: Optional[Imu] = None
        self._last_gps: Optional[NavSatFix] = None
        self._last_odom: Optional[Odometry] = None
        self._last_encoder_velocities: Optional[Float32MultiArray] = None
        self._last_encoder_speeds_mm_s: Optional[Float32MultiArray] = None
        self._last_encoder_angle_accum: Optional[Float32MultiArray] = None
        self._last_suspension: Optional[Float32MultiArray] = None
        self._last_steering_angle: Optional[Float32] = None
        self._last_water_pressure: Optional[Float32] = None
        self._last_water_flow: Optional[Float32] = None
        self._last_water_temp_in: Optional[Float32] = None
        self._last_water_temp_out: Optional[Float32] = None
        self._last_water_temp_radiator: Optional[Float32] = None
        self._last_brake_temp_fr: Optional[Float32] = None
        self._last_brake_temp_rl: Optional[Float32] = None
        self._last_pitot_pressure: Optional[Float32] = None
        self._dr_x = 0.0
        self._dr_y = 0.0
        self._dr_vx = 0.0
        self._dr_vy = 0.0
        self._last_dr_time: Optional[float] = None

        # Set up subscriptions
        self.setup_subscriptions()

    def setup_subscriptions(self) -> None:
        """Create ROS 2 subscriptions for Gazebo sensor topics."""
        # Register sensors with synchronizer
        self.synchronizer.add_sensor(
            'imu',
            rate_hz=self.SENSOR_RATES['imu'],
            required=False,  # Can operate without IMU
            interpolate=True,
        )
        self.synchronizer.add_sensor(
            'odom',
            rate_hz=self.SENSOR_RATES['odom'],
            required=True,
            interpolate=True,
            interpolator=interpolate_odometry,
        )
        self.synchronizer.add_sensor(
            'gps',
            rate_hz=self.SENSOR_RATES['gps'],
            required=False,
            interpolate=False,  # GPS too slow for interpolation
            max_age_sec=1.0,
        )
        self.synchronizer.add_sensor(
            'encoder_velocities',
            rate_hz=self.SENSOR_RATES['encoder_velocities'],
            required=False,
            interpolate=False,
        )
        self.synchronizer.add_sensor(
            'encoder_speeds_mm_s',
            rate_hz=self.SENSOR_RATES['encoder_speeds_mm_s'],
            required=False,
            interpolate=False,
        )

        # Create subscriptions
        self._imu_sub = self.node.create_subscription(
            Imu,
            self.topics['imu'],
            self._imu_callback,
            SENSOR_QOS,
        )

        self._gps_sub = self.node.create_subscription(
            NavSatFix,
            self.topics['gps'],
            self._gps_callback,
            RELIABLE_SENSOR_QOS,
        )

        self._odom_sub = self.node.create_subscription(
            Odometry,
            self.topics['odom'],
            self._odom_callback,
            SENSOR_QOS,
        )

        self._encoder_velocities_sub = self.node.create_subscription(
            Float32MultiArray,
            self.topics['encoder_velocities'],
            self._encoder_velocities_callback,
            SENSOR_QOS,
        )

        self._encoder_speeds_mm_s_sub = self.node.create_subscription(
            Float32MultiArray,
            self.topics['encoder_speeds_mm_s'],
            self._encoder_speeds_mm_s_callback,
            SENSOR_QOS,
        )

        self._encoder_angle_accum_sub = self.node.create_subscription(
            Float32MultiArray,
            self.topics['encoder_angle_accum'],
            self._encoder_angle_accum_callback,
            SENSOR_QOS,
        )

        # Suspension sensor
        self._suspension_sub = self.node.create_subscription(
            Float32MultiArray,
            self.topics['suspension'],
            self._suspension_callback,
            SENSOR_QOS,
        )

        # Steering angle sensor
        self._steering_angle_sub = self.node.create_subscription(
            Float32,
            self.topics['steering_angle'],
            self._steering_angle_callback,
            SENSOR_QOS,
        )

        # Virtual sensors - Cooling
        self._water_pressure_sub = self.node.create_subscription(
            Float32,
            self.topics['water_pressure'],
            self._water_pressure_callback,
            SENSOR_QOS,
        )
        self._water_flow_sub = self.node.create_subscription(
            Float32,
            self.topics['water_flow'],
            self._water_flow_callback,
            SENSOR_QOS,
        )
        self._water_temp_in_sub = self.node.create_subscription(
            Float32,
            self.topics['water_temp_in'],
            self._water_temp_in_callback,
            SENSOR_QOS,
        )
        self._water_temp_out_sub = self.node.create_subscription(
            Float32,
            self.topics['water_temp_out'],
            self._water_temp_out_callback,
            SENSOR_QOS,
        )
        self._water_temp_radiator_sub = self.node.create_subscription(
            Float32,
            self.topics['water_temp_radiator'],
            self._water_temp_radiator_callback,
            SENSOR_QOS,
        )

        # Virtual sensors - Brakes
        self._brake_temp_fr_sub = self.node.create_subscription(
            Float32,
            self.topics['brake_temp_fr'],
            self._brake_temp_fr_callback,
            SENSOR_QOS,
        )
        self._brake_temp_rl_sub = self.node.create_subscription(
            Float32,
            self.topics['brake_temp_rl'],
            self._brake_temp_rl_callback,
            SENSOR_QOS,
        )

        # Virtual sensors - Pitot
        self._pitot_pressure_sub = self.node.create_subscription(
            Float32,
            self.topics['pitot_pressure'],
            self._pitot_pressure_callback,
            SENSOR_QOS,
        )

        self.log_info("Subscribed to Gazebo topics:")
        for name, topic in self.topics.items():
            self.log_info(f"  {name}: {topic}")

    def _get_timestamp(self, header) -> float:
        """Extract timestamp from ROS header."""
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def _imu_callback(self, msg: Imu) -> None:
        """IMU callback - add to synchronizer."""
        timestamp = self._get_timestamp(msg.header)
        self.synchronizer.add_sample('imu', timestamp, msg)
        self._last_imu = msg

    def _gps_callback(self, msg: NavSatFix) -> None:
        """GPS callback - set origin if needed, add to synchronizer."""
        # Auto-set GPS origin from first valid message
        if self.auto_set_gps_origin and not self._gps_origin_set:
            if msg.status.status >= 0:  # STATUS_FIX or better
                set_gps_origin(msg.latitude, msg.longitude, msg.altitude)
                self._gps_origin_set = True
                self.log_info(f"GPS origin set: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}")

        timestamp = self._get_timestamp(msg.header)
        self.synchronizer.add_sample('gps', timestamp, msg)
        self._last_gps = msg

    def _odom_callback(self, msg: Odometry) -> None:
        """Odometry callback - add to synchronizer."""
        timestamp = self._get_timestamp(msg.header)
        self.synchronizer.add_sample('odom', timestamp, msg)
        self._last_odom = msg

        # Log first odom message for debugging
        if not hasattr(self, '_first_odom_logged'):
            self._first_odom_logged = True
            self.log_info(f"First odom received: timestamp={timestamp:.2f}s, pos=({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})")

    def _encoder_velocities_callback(self, msg: Float32MultiArray) -> None:
        """Encoder RPM callback - add to synchronizer."""
        # Use latest odom timestamp to stay synchronized
        if self._last_odom is not None:
            timestamp = self._get_timestamp(self._last_odom.header)
        else:
            timestamp = 0.0
        self.synchronizer.add_sample('encoder_velocities', timestamp, msg)
        self._last_encoder_velocities = msg

    def _encoder_angle_accum_callback(self, msg: Float32MultiArray) -> None:
        """Encoder angle accumulation callback."""
        self._last_encoder_angle_accum = msg

    def _encoder_speeds_mm_s_callback(self, msg: Float32MultiArray) -> None:
        """Encoder linear speed callback."""
        if self._last_odom is not None:
            timestamp = self._get_timestamp(self._last_odom.header)
        else:
            timestamp = 0.0
        self.synchronizer.add_sample('encoder_speeds_mm_s', timestamp, msg)
        self._last_encoder_speeds_mm_s = msg

    def _suspension_callback(self, msg: Float32MultiArray) -> None:
        """Suspension displacement callback."""
        self._last_suspension = msg

    def _steering_angle_callback(self, msg: Float32) -> None:
        """Steering angle callback."""
        self._last_steering_angle = msg

    def _water_pressure_callback(self, msg: Float32) -> None:
        """Water pressure callback."""
        self._last_water_pressure = msg

    def _water_flow_callback(self, msg: Float32) -> None:
        """Water flow callback."""
        self._last_water_flow = msg

    def _water_temp_in_callback(self, msg: Float32) -> None:
        """Water temp in callback."""
        self._last_water_temp_in = msg

    def _water_temp_out_callback(self, msg: Float32) -> None:
        """Water temp out callback."""
        self._last_water_temp_out = msg

    def _water_temp_radiator_callback(self, msg: Float32) -> None:
        """Water temp radiator callback."""
        self._last_water_temp_radiator = msg

    def _brake_temp_fr_callback(self, msg: Float32) -> None:
        """Brake temp FR callback."""
        self._last_brake_temp_fr = msg

    def _brake_temp_rl_callback(self, msg: Float32) -> None:
        """Brake temp RL callback."""
        self._last_brake_temp_rl = msg

    def _pitot_pressure_callback(self, msg: Float32) -> None:
        """Pitot pressure callback."""
        self._last_pitot_pressure = msg

    def compute_state(
        self,
        synced_data: Dict[str, Any],
        prev_state: Optional[VehicleState],
    ) -> VehicleState:
        """
        Convert synchronized sensor data to VehicleState.

        Primary source: Odometry (position, velocity, orientation)
        Secondary: IMU (yaw rate), GPS (lat/lon reference), Encoders
        """
        state = VehicleState()
        state.source_adapter = self.adapter_name

        # Get current time
        odom: Optional[Odometry] = synced_data.get('odom')
        if odom is None:
            # Fallback to current time if no odom
            state.timestamp = self.node.get_clock().now().nanoseconds * 1e-9
        else:
            state.timestamp = self._get_timestamp(odom.header)

        # Position from odometry
        if odom is not None:
            state.x = odom.pose.pose.position.x
            state.y = odom.pose.pose.position.y

            # Velocity from odometry (body frame)
            state.vx = odom.twist.twist.linear.x
            state.vy = odom.twist.twist.linear.y

            # Yaw from odometry quaternion
            q = odom.pose.pose.orientation
            state.yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

            # Yaw rate from odometry
            state.yaw_rate = odom.twist.twist.angular.z

        # Refine yaw rate from IMU if available
        imu: Optional[Imu] = synced_data.get('imu')
        if imu is not None:
            # IMU provides more accurate angular velocity
            state.yaw_rate = imu.angular_velocity.z

        # GPS data (for reference)
        gps: Optional[NavSatFix] = synced_data.get('gps')
        if gps is not None:
            state.gps_latitude = gps.latitude
            state.gps_longitude = gps.longitude
            state.gps_altitude = gps.altitude
            state.gps_valid = gps.status.status >= 0
            if state.gps_valid and get_gps_origin() is not None:
                try:
                    state.gps_local_x, state.gps_local_y = gps_to_local(
                        gps.latitude, gps.longitude
                    )
                except ValueError:
                    pass

        # Encoder RPM
        encoder_velocities: Optional[Float32MultiArray] = synced_data.get('encoder_velocities')
        if encoder_velocities is not None and len(encoder_velocities.data) >= 4:
            state.encoder_velocities = list(encoder_velocities.data[:4])

        encoder_speeds_mm_s: Optional[Float32MultiArray] = synced_data.get('encoder_speeds_mm_s')
        if encoder_speeds_mm_s is not None and len(encoder_speeds_mm_s.data) >= 4:
            state.encoder_speeds_mm_s = list(encoder_speeds_mm_s.data[:4])
        elif self._last_encoder_speeds_mm_s is not None and len(self._last_encoder_speeds_mm_s.data) >= 4:
            state.encoder_speeds_mm_s = list(self._last_encoder_speeds_mm_s.data[:4])

        if self._last_encoder_angle_accum is not None and len(self._last_encoder_angle_accum.data) >= 4:
            state.encoder_angle_accum = list(self._last_encoder_angle_accum.data[:4])

        # Compute derived quantities
        state.update_speed()

        # IMU-derived yaw and dead-reckoning position
        if imu is not None:
            q = imu.orientation
            state.imu_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
            if self._last_dr_time is None:
                self._last_dr_time = state.timestamp
            dt = max(0.0, state.timestamp - self._last_dr_time)
            if 0.0 < dt <= 0.2:
                ax = float(imu.linear_acceleration.x)
                ay = float(imu.linear_acceleration.y)
                cy = math.cos(state.imu_yaw)
                sy = math.sin(state.imu_yaw)
                ax_world = cy * ax - sy * ay
                ay_world = sy * ax + cy * ay
                self._dr_vx += ax_world * dt
                self._dr_vy += ay_world * dt
                self._dr_x += self._dr_vx * dt
                self._dr_y += self._dr_vy * dt
            self._last_dr_time = state.timestamp

        state.dr_x = self._dr_x
        state.dr_y = self._dr_y
        state.imu_vx = self._dr_vx
        state.imu_vy = self._dr_vy

        # Compute slip if we have wheel RPM (convert to m/s for slip calc)
        if state.encoder_velocities and state.speed > 0.01 and self.wheel_radius > 0.0:
            avg_wheel_rpm = sum(state.encoder_velocities) / 4.0
            avg_wheel_mps = (avg_wheel_rpm * 2.0 * math.pi / 60.0) * self.wheel_radius
            max_vel = max(abs(avg_wheel_mps), state.speed)
            if max_vel > 0.01:
                state.slip_longitudinal = (avg_wheel_mps - state.speed) / max_vel

            if abs(state.vx) > 0.01:
                state.slip_lateral = math.atan2(state.vy, state.vx)

        # Suspension displacements (convert from mm to meters)
        if self._last_suspension is not None and len(self._last_suspension.data) >= 4:
            # Suspension node publishes in mm, convert to meters for VehicleState
            state.suspension = [v / 1000.0 for v in self._last_suspension.data[:4]]

        # Steering angle (already in degrees from steering_sensor_node)
        if self._last_steering_angle is not None:
            state.steering_angle = self._last_steering_angle.data
            state.steering_valid = True

        # Cooling system
        if self._last_water_pressure is not None:
            state.water_pressure = self._last_water_pressure.data
        if self._last_water_flow is not None:
            state.water_flow = self._last_water_flow.data
        if self._last_water_temp_in is not None:
            state.water_temp_in = self._last_water_temp_in.data
        if self._last_water_temp_out is not None:
            state.water_temp_out = self._last_water_temp_out.data
        if self._last_water_temp_radiator is not None:
            state.water_temp_radiator = self._last_water_temp_radiator.data

        # Brake temperatures
        if self._last_brake_temp_fr is not None:
            state.brake_temp_fr = self._last_brake_temp_fr.data
        if self._last_brake_temp_rl is not None:
            state.brake_temp_rl = self._last_brake_temp_rl.data

        # Pitot tube
        if self._last_pitot_pressure is not None:
            state.pitot_dynamic_pressure = self._last_pitot_pressure.data

        return state

    def get_last_raw_data(self) -> Dict[str, Any]:
        """Return the last raw sensor messages (for debugging)."""
        return {
            'imu': self._last_imu,
            'gps': self._last_gps,
            'odom': self._last_odom,
            'encoder_velocities': self._last_encoder_velocities,
            'encoder_speeds_mm_s': self._last_encoder_speeds_mm_s,
            'encoder_angle_accum': self._last_encoder_angle_accum,
            'suspension': self._last_suspension,
            'steering_angle': self._last_steering_angle,
            'water_pressure': self._last_water_pressure,
            'water_flow': self._last_water_flow,
            'water_temp_in': self._last_water_temp_in,
            'water_temp_out': self._last_water_temp_out,
            'water_temp_radiator': self._last_water_temp_radiator,
            'brake_temp_fr': self._last_brake_temp_fr,
            'brake_temp_rl': self._last_brake_temp_rl,
            'pitot_pressure': self._last_pitot_pressure,
        }
