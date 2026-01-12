"""
VectorNavAdapter - Sensor adapter for VectorNav VN-200 INS/GPS.

Subscribes to decoded VectorNav topics from the vectornav_decoder package
and converts them to unified VehicleState messages.
"""

from typing import Optional, Dict, Any
import math

from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry

from ..core.sensor_interfaces import SensorAdapterInterface
from ..core.vehicle_state import VehicleState
from ..core.time_sync import TimeSynchronizer
from ..core.qos_profiles import SENSOR_QOS, RELIABLE_SENSOR_QOS
from ..utils.transforms import (
    quaternion_to_yaw,
    gps_to_local,
    set_gps_origin,
    get_gps_origin,
)


class VectorNavAdapter(SensorAdapterInterface):
    """
    Sensor adapter for VectorNav VN-200 INS.

    Subscribes to:
    - /vectornav/imu (sensor_msgs/Imu) - IMU at ~200 Hz
    - /vectornav/gps (sensor_msgs/NavSatFix) - GPS at ~5 Hz
    - /vectornav/ins (nav_msgs/Odometry) - INS solution at ~200 Hz

    Provides full vehicle state including:
    - Position from INS (or GPS + dead reckoning)
    - Velocity from INS body frame
    - Orientation from IMU
    - Yaw rate from IMU angular velocity

    Topic names are configurable via the topics parameter.
    """

    DEFAULT_TOPICS = {
        'imu': '/vectornav/imu',
        'gps': '/vectornav/gps',
        'ins': '/vectornav/ins',
    }

    SENSOR_RATES = {
        'imu': 200.0,
        'gps': 5.0,
        'ins': 200.0,
    }

    def __init__(
        self,
        node: Node,
        synchronizer: TimeSynchronizer,
        topics: Optional[Dict[str, str]] = None,
        auto_set_gps_origin: bool = True,
        prefer_ins: bool = True,
    ):
        """
        Initialize VectorNav adapter.

        Args:
            node: ROS 2 node
            synchronizer: Time synchronizer
            topics: Topic name overrides
            auto_set_gps_origin: Set GPS origin from first fix
            prefer_ins: Prefer INS solution over raw GPS
        """
        super().__init__(node, synchronizer, adapter_name="vectornav")

        self.topics = {**self.DEFAULT_TOPICS, **(topics or {})}
        self.auto_set_gps_origin = auto_set_gps_origin
        self.prefer_ins = prefer_ins
        self._gps_origin_set = False

        # Latest raw data (for debugging)
        self._last_imu: Optional[Imu] = None
        self._last_gps: Optional[NavSatFix] = None
        self._last_ins: Optional[Odometry] = None

        # Dead-reckoning state (IMU integration)
        self._dr_x: float = 0.0
        self._dr_y: float = 0.0
        self._dr_vx: float = 0.0
        self._dr_vy: float = 0.0
        self._last_imu_time: Optional[float] = None

        self.setup_subscriptions()

    def setup_subscriptions(self) -> None:
        """Create ROS 2 subscriptions for VectorNav topics."""
        # Register sensors with synchronizer
        self.synchronizer.add_sensor(
            'imu',
            rate_hz=self.SENSOR_RATES['imu'],
            required=True,  # IMU is primary data source
            interpolate=True,
        )
        self.synchronizer.add_sensor(
            'gps',
            rate_hz=self.SENSOR_RATES['gps'],
            required=False,
            interpolate=False,
            max_age_sec=1.0,
        )
        self.synchronizer.add_sensor(
            'ins',
            rate_hz=self.SENSOR_RATES['ins'],
            required=False,  # Can operate with IMU only
            interpolate=True,
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

        self._ins_sub = self.node.create_subscription(
            Odometry,
            self.topics['ins'],
            self._ins_callback,
            SENSOR_QOS,
        )

        self.log_info("Subscribed to VectorNav topics:")
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

        # Log first message
        if not hasattr(self, '_first_imu_logged'):
            self._first_imu_logged = True
            self.log_info(f"First IMU received at t={timestamp:.2f}")

    def _gps_callback(self, msg: NavSatFix) -> None:
        """GPS callback - add to synchronizer and set origin if needed."""
        # Auto-set GPS origin from first valid fix
        if self.auto_set_gps_origin and not self._gps_origin_set:
            if msg.status.status >= 0:  # Has fix
                set_gps_origin(msg.latitude, msg.longitude, msg.altitude)
                self._gps_origin_set = True
                self.log_info(
                    f"GPS origin set: {msg.latitude:.6f}, {msg.longitude:.6f}"
                )

        timestamp = self._get_timestamp(msg.header)
        self.synchronizer.add_sample('gps', timestamp, msg)
        self._last_gps = msg

        # Log first message
        if not hasattr(self, '_first_gps_logged'):
            self._first_gps_logged = True
            self.log_info(
                f"First GPS received: {msg.latitude:.6f}, {msg.longitude:.6f}, "
                f"fix={msg.status.status}"
            )

    def _ins_callback(self, msg: Odometry) -> None:
        """INS callback - add to synchronizer."""
        timestamp = self._get_timestamp(msg.header)
        self.synchronizer.add_sample('ins', timestamp, msg)
        self._last_ins = msg

        # Log first message
        if not hasattr(self, '_first_ins_logged'):
            self._first_ins_logged = True
            self.log_info(f"First INS received at t={timestamp:.2f}")

    def compute_state(
        self,
        synced_data: Dict[str, Any],
        prev_state: Optional[VehicleState],
    ) -> VehicleState:
        """
        Convert synchronized VectorNav data to VehicleState.

        Priority:
        1. INS for position/velocity (if available and prefer_ins=True)
        2. GPS for position (fallback)
        3. IMU for orientation and yaw rate
        """
        state = VehicleState()
        state.source_adapter = self.adapter_name

        imu: Optional[Imu] = synced_data.get('imu')
        gps: Optional[NavSatFix] = synced_data.get('gps')
        ins: Optional[Odometry] = synced_data.get('ins')

        # Timestamp from IMU (highest rate)
        if imu is not None:
            state.timestamp = self._get_timestamp(imu.header)
        elif ins is not None:
            state.timestamp = self._get_timestamp(ins.header)
        else:
            state.timestamp = self.node.get_clock().now().nanoseconds * 1e-9

        # Position: prefer INS, fall back to GPS
        position_set = False
        if self.prefer_ins and ins is not None:
            # INS odometry stores GPS coords in position (from vectornav_decoder_node)
            # Convert lat/lon to local frame if origin is set
            if self._gps_origin_set:
                try:
                    # The decoder stores: x=longitude, y=latitude, z=altitude
                    x, y = gps_to_local(
                        ins.pose.pose.position.y,  # latitude
                        ins.pose.pose.position.x,  # longitude
                    )
                    state.x = x
                    state.y = y
                    position_set = True
                except ValueError:
                    pass  # Origin not set yet

        if not position_set and gps is not None and gps.status.status >= 0:
            if self._gps_origin_set:
                try:
                    x, y = gps_to_local(gps.latitude, gps.longitude)
                    state.x = x
                    state.y = y
                    position_set = True
                except ValueError:
                    pass

            # Store raw GPS
            state.gps_latitude = gps.latitude
            state.gps_longitude = gps.longitude
            state.gps_valid = True

        # Velocity: from INS body frame
        if ins is not None:
            state.vx = ins.twist.twist.linear.x  # Forward velocity
            state.vy = ins.twist.twist.linear.y  # Lateral velocity

        # Orientation: from IMU quaternion
        if imu is not None:
            q = imu.orientation
            # Check if quaternion is valid (not all zeros)
            if q.w != 0 or q.x != 0 or q.y != 0 or q.z != 0:
                state.yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
            # Yaw rate from angular velocity
            state.yaw_rate = imu.angular_velocity.z

        # Compute derived quantities
        state.speed = math.sqrt(state.vx ** 2 + state.vy ** 2)

        # INS position (explicit copy for plotting)
        state.ins_x = state.x
        state.ins_y = state.y

        # GPS to local coordinates
        if gps is not None and gps.status.status >= 0 and self._gps_origin_set:
            try:
                gps_x, gps_y = gps_to_local(gps.latitude, gps.longitude)
                state.gps_local_x = gps_x
                state.gps_local_y = gps_y
            except ValueError:
                pass  # Origin not set yet

        # Dead-reckoning from IMU (integrate acceleration)
        if imu is not None:
            current_time = state.timestamp
            if self._last_imu_time is not None:
                dt = current_time - self._last_imu_time
                if 0 < dt < 0.1:  # Sanity check: reasonable time delta
                    # Get linear acceleration in body frame
                    ax = imu.linear_acceleration.x
                    ay = imu.linear_acceleration.y

                    # Integrate to velocity (body frame)
                    self._dr_vx += ax * dt
                    self._dr_vy += ay * dt

                    # Rotate body-frame velocity to world-frame using current yaw
                    cos_yaw = math.cos(state.yaw)
                    sin_yaw = math.sin(state.yaw)
                    world_vx = self._dr_vx * cos_yaw - self._dr_vy * sin_yaw
                    world_vy = self._dr_vx * sin_yaw + self._dr_vy * cos_yaw

                    # Integrate position (world frame)
                    self._dr_x += world_vx * dt
                    self._dr_y += world_vy * dt

            self._last_imu_time = current_time

        state.dr_x = self._dr_x
        state.dr_y = self._dr_y

        # Distance traveled (accumulated)
        if prev_state is not None:
            dt = state.timestamp - prev_state.timestamp
            if dt > 0 and dt < 0.1:  # Reasonable time delta
                state.distance_traveled = prev_state.distance_traveled + state.speed * dt
            else:
                state.distance_traveled = prev_state.distance_traveled

        return state

    def get_last_raw_data(self) -> Dict[str, Any]:
        """Return last raw sensor messages (for debugging)."""
        return {
            'imu': self._last_imu,
            'gps': self._last_gps,
            'ins': self._last_ins,
        }
