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
from std_msgs.msg import Float32MultiArray

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
    - /wheel_encoder/velocities (std_msgs/Float32MultiArray) - Wheel velocities in m/s

    Topic names are configurable via ROS parameters.
    """

    # Default topic names (can be remapped via launch file)
    # These match the ros_gz_bridge topic names from gazebo_sim.launch.py
    DEFAULT_TOPICS = {
        'imu': '/imu',                              # From ros_gz_bridge
        'gps': '/navsat',                           # From ros_gz_bridge
        'odom': '/odom',                            # From ros_gz_bridge
        'encoder_velocities': '/wheel_encoder/velocities',  # From wheel_encoder_node
    }

    # Sensor rates (Hz) - used for time sync configuration
    SENSOR_RATES = {
        'imu': 100.0,
        'gps': 5.0,
        'odom': 50.0,
        'encoder_velocities': 50.0,
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

        # Latest raw sensor data (for debugging)
        self._last_imu: Optional[Imu] = None
        self._last_gps: Optional[NavSatFix] = None
        self._last_odom: Optional[Odometry] = None
        self._last_encoder_velocities: Optional[Float32MultiArray] = None

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

        self.log_info(f"Subscribed to Gazebo topics:")
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
        """Encoder velocities callback - add to synchronizer."""
        # Use latest odom timestamp to stay synchronized
        if self._last_odom is not None:
            timestamp = self._get_timestamp(self._last_odom.header)
        else:
            timestamp = 0.0
        self.synchronizer.add_sample('encoder_velocities', timestamp, msg)
        self._last_encoder_velocities = msg

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

        # Encoder velocities
        encoder_velocities: Optional[Float32MultiArray] = synced_data.get('encoder_velocities')
        if encoder_velocities is not None and len(encoder_velocities.data) >= 4:
            state.encoder_velocities = list(encoder_velocities.data[:4])

        # Compute derived quantities
        state.update_speed()

        # Compute slip if we have wheel velocities
        if state.encoder_velocities and state.speed > 0.01:
            avg_wheel_vel = sum(state.encoder_velocities) / 4.0
            max_vel = max(abs(avg_wheel_vel), state.speed)
            if max_vel > 0.01:
                state.slip_longitudinal = (avg_wheel_vel - state.speed) / max_vel

            if abs(state.vx) > 0.01:
                state.slip_lateral = math.atan2(state.vy, state.vx)

        return state

    def get_last_raw_data(self) -> Dict[str, Any]:
        """Return the last raw sensor messages (for debugging)."""
        return {
            'imu': self._last_imu,
            'gps': self._last_gps,
            'odom': self._last_odom,
            'encoder_velocities': self._last_encoder_velocities,
        }
