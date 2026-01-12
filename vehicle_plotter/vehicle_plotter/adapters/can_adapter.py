"""
CANAdapter - Sensor adapter for CAN bus wheel encoders.

Subscribes to decoded CAN topics from the canbus_decoder package
and converts them to unified VehicleState messages.
"""

from typing import Optional, Dict, Any

from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray

from ..core.sensor_interfaces import SensorAdapterInterface
from ..core.vehicle_state import VehicleState
from ..core.time_sync import TimeSynchronizer
from ..core.qos_profiles import SENSOR_QOS


class CANAdapter(SensorAdapterInterface):
    """
    Sensor adapter for CAN bus wheel encoders.

    Subscribes to:
    - /can/wheel_velocities (std_msgs/Float32MultiArray) - [FL, FR, RL, RR] wheel velocities in m/s
    - /can/suspension (std_msgs/Float32MultiArray) - [FL, FR, RL, RR] suspension displacements in m
    - /can/steering_angle (std_msgs/Float32) - Steering angle in radians

    Topic names are configurable via ROS parameters.

    Note: CAN encoder data provides wheel velocities and steering but not
    position or orientation (no IMU/GPS). Those fields remain at zero.
    """

    # Default topic names (can be remapped via launch file)
    DEFAULT_TOPICS = {
        'wheel_velocities': '/can/wheel_velocities',
        'suspension': '/can/suspension',
        'steering_angle': '/can/steering_angle',
    }

    # Sensor rates (Hz) - CAN bus at 50Hz typical
    SENSOR_RATES = {
        'wheel_velocities': 50.0,
        'suspension': 50.0,
        'steering_angle': 50.0,
    }

    def __init__(
        self,
        node: Node,
        synchronizer: TimeSynchronizer,
        topics: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the CAN adapter.

        Args:
            node: ROS 2 node for subscriptions
            synchronizer: Time synchronizer
            topics: Optional topic name overrides
        """
        super().__init__(node, synchronizer, adapter_name="can")

        self.topics = {**self.DEFAULT_TOPICS, **(topics or {})}

        # Latest raw sensor data (for debugging)
        self._last_wheel_velocities: Optional[Float32MultiArray] = None
        self._last_suspension: Optional[Float32MultiArray] = None
        self._last_steering_angle: Optional[Float32] = None

        # Set up subscriptions
        self.setup_subscriptions()

    def setup_subscriptions(self) -> None:
        """Create ROS 2 subscriptions for CAN decoder topics."""
        # Register sensors with synchronizer
        self.synchronizer.add_sensor(
            'wheel_velocities',
            rate_hz=self.SENSOR_RATES['wheel_velocities'],
            required=True,  # Need wheel velocities at minimum
            interpolate=False,  # Discrete encoder data, no interpolation
        )
        self.synchronizer.add_sensor(
            'suspension',
            rate_hz=self.SENSOR_RATES['suspension'],
            required=False,
            interpolate=False,
        )
        self.synchronizer.add_sensor(
            'steering_angle',
            rate_hz=self.SENSOR_RATES['steering_angle'],
            required=False,
            interpolate=False,
        )

        # Create subscriptions
        self._wheel_vel_sub = self.node.create_subscription(
            Float32MultiArray,
            self.topics['wheel_velocities'],
            self._wheel_velocities_callback,
            SENSOR_QOS,
        )

        self._suspension_sub = self.node.create_subscription(
            Float32MultiArray,
            self.topics['suspension'],
            self._suspension_callback,
            SENSOR_QOS,
        )

        self._steering_sub = self.node.create_subscription(
            Float32,
            self.topics['steering_angle'],
            self._steering_angle_callback,
            SENSOR_QOS,
        )

        self.log_info(f"Subscribed to CAN topics:")
        for name, topic in self.topics.items():
            self.log_info(f"  {name}: {topic}")

    def _get_timestamp(self) -> float:
        """
        Get current timestamp.

        Note: CAN messages from can_decoder_node don't have headers,
        so we use ROS node time.
        """
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _wheel_velocities_callback(self, msg: Float32MultiArray) -> None:
        """Wheel velocities callback - add to synchronizer."""
        timestamp = self._get_timestamp()
        self.synchronizer.add_sample('wheel_velocities', timestamp, msg)
        self._last_wheel_velocities = msg

        # Log first message for debugging
        if not hasattr(self, '_first_wheel_vel_logged'):
            self._first_wheel_vel_logged = True
            self.log_info(f"First wheel velocities received: {msg.data}")

    def _suspension_callback(self, msg: Float32MultiArray) -> None:
        """Suspension callback - add to synchronizer."""
        timestamp = self._get_timestamp()
        self.synchronizer.add_sample('suspension', timestamp, msg)
        self._last_suspension = msg

    def _steering_angle_callback(self, msg: Float32) -> None:
        """Steering angle callback - add to synchronizer."""
        timestamp = self._get_timestamp()
        self.synchronizer.add_sample('steering_angle', timestamp, msg)
        self._last_steering_angle = msg

    def compute_state(
        self,
        synced_data: Dict[str, Any],
        prev_state: Optional[VehicleState],
    ) -> VehicleState:
        """
        Convert synchronized CAN data to VehicleState.

        Primary source: Wheel velocities (encoder_velocities)
        Secondary: Suspension, steering angle
        Missing: Position, orientation, GPS (CAN has no IMU/GPS)
        """
        state = VehicleState()
        state.source_adapter = self.adapter_name

        # Get current time
        state.timestamp = self._get_timestamp()

        # Wheel velocities (primary CAN data)
        wheel_velocities: Optional[Float32MultiArray] = synced_data.get('wheel_velocities')
        if wheel_velocities is not None and len(wheel_velocities.data) >= 4:
            state.encoder_velocities = list(wheel_velocities.data[:4])  # [FL, FR, RL, RR]

            # Compute average speed from wheel velocities
            avg_wheel_speed = sum(state.encoder_velocities) / 4.0
            state.speed = abs(avg_wheel_speed)

        # Suspension data
        suspension: Optional[Float32MultiArray] = synced_data.get('suspension')
        if suspension is not None and len(suspension.data) >= 4:
            state.suspension = list(suspension.data[:4])  # [FL, FR, RL, RR]

        # Steering angle (not part of standard VehicleState, but available)
        steering_angle: Optional[Float32] = synced_data.get('steering_angle')
        if steering_angle is not None:
            # For now, steering angle is captured but not stored in VehicleState
            # Future: Could extend VehicleState with steering_angle field
            pass

        # CAN data does not provide:
        # - Position (x, y) - would need IMU/GPS or odometry integration
        # - Orientation (yaw, yaw_rate) - would need IMU
        # - GPS data - would need GPS receiver
        # These fields remain at their default values (0.0)

        # Compute distance traveled if we have previous state
        if prev_state is not None and state.speed > 0:
            dt = state.timestamp - prev_state.timestamp
            if dt > 0 and dt < 1.0:  # Reasonable time delta
                state.distance_traveled = prev_state.distance_traveled + state.speed * dt

        return state

    def get_last_raw_data(self) -> Dict[str, Any]:
        """Return the last raw sensor messages (for debugging)."""
        return {
            'wheel_velocities': self._last_wheel_velocities,
            'suspension': self._last_suspension,
            'steering_angle': self._last_steering_angle,
        }
