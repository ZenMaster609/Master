"""
TimeSynchronizer - Buffer-based approximate time synchronization.

Handles multi-rate sensor data synchronization with interpolation
for smooth output at a fixed rate.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Deque, Callable
from collections import deque
import threading


@dataclass
class SensorSample:
    """A timestamped sensor reading."""
    timestamp: float      # ROS time in seconds
    data: Any            # The sensor message/data
    source: str          # Sensor identifier


@dataclass
class SensorConfig:
    """Configuration for a sensor source."""
    name: str
    rate_hz: float
    required: bool = True
    interpolate: bool = True
    max_age_sec: float = 0.5  # Maximum age before considered stale


class TimeSynchronizer:
    """
    Buffer-based approximate time synchronizer for multi-rate sensors.

    Strategy:
    1. Maintain rolling buffers for each sensor
    2. On each output tick, find the best-matching samples
    3. For slower sensors (GPS), use most recent valid reading
    4. For faster sensors (IMU), interpolate or use nearest

    This approach is suitable for pre-EKF data collection where
    exact synchronization is less critical than for real-time control.

    Example:
        sync = TimeSynchronizer(output_rate_hz=50.0)

        # Configure sensors
        sync.add_sensor('imu', rate_hz=100, required=True, interpolate=True)
        sync.add_sensor('gps', rate_hz=5, required=False, interpolate=False)
        sync.add_sensor('odom', rate_hz=50, required=True, interpolate=True)

        # In callbacks, add samples
        sync.add_sample('imu', timestamp, imu_msg)
        sync.add_sample('gps', timestamp, gps_msg)

        # At output rate, get synchronized data
        synced = sync.get_synchronized(current_time)
        if synced is not None:
            # Process synchronized data
            pass
    """

    def __init__(
        self,
        output_rate_hz: float = 50.0,
        buffer_duration_sec: float = 0.2,
        default_interpolate: bool = True,
    ):
        """
        Initialize the time synchronizer.

        Args:
            output_rate_hz: Target output rate in Hz
            buffer_duration_sec: How long to buffer samples (seconds)
            default_interpolate: Default interpolation setting for sensors
        """
        self.output_rate_hz = output_rate_hz
        self.buffer_duration_sec = buffer_duration_sec
        self.default_interpolate = default_interpolate

        self._buffers: Dict[str, Deque[SensorSample]] = {}
        self._configs: Dict[str, SensorConfig] = {}
        self._interpolators: Dict[str, Callable] = {}
        self._lock = threading.Lock()

        # Statistics
        self._sample_counts: Dict[str, int] = {}
        self._last_sync_time: float = 0.0

    def add_sensor(
        self,
        name: str,
        rate_hz: float,
        required: bool = True,
        interpolate: bool = True,
        max_age_sec: float = 0.5,
        interpolator: Optional[Callable] = None,
    ) -> None:
        """
        Add a sensor source to the synchronizer.

        Args:
            name: Unique sensor identifier
            rate_hz: Expected update rate in Hz
            required: If True, synchronization fails without this sensor
            interpolate: If True, interpolate between samples
            max_age_sec: Maximum age before data is considered stale
            interpolator: Custom interpolation function (a, b, alpha) -> interpolated
        """
        with self._lock:
            self._configs[name] = SensorConfig(
                name=name,
                rate_hz=rate_hz,
                required=required,
                interpolate=interpolate,
                max_age_sec=max_age_sec,
            )

            # Calculate buffer size based on rate and duration
            buffer_size = max(100, int(rate_hz * self.buffer_duration_sec * 2))
            self._buffers[name] = deque(maxlen=buffer_size)
            self._sample_counts[name] = 0

            if interpolator is not None:
                self._interpolators[name] = interpolator

    def add_sample(self, source: str, timestamp: float, data: Any) -> None:
        """
        Add a new sample to the appropriate buffer.

        Args:
            source: Sensor identifier (must be registered with add_sensor)
            timestamp: ROS time in seconds
            data: The sensor message/data
        """
        with self._lock:
            if source not in self._buffers:
                # Auto-register with defaults if not configured
                self._configs[source] = SensorConfig(
                    name=source,
                    rate_hz=50.0,
                    required=False,
                    interpolate=self.default_interpolate,
                )
                self._buffers[source] = deque(maxlen=200)
                self._sample_counts[source] = 0

            self._buffers[source].append(SensorSample(timestamp, data, source))
            self._sample_counts[source] += 1

            # Log first sample for debugging
            if self._sample_counts[source] == 1:
                config = self._configs.get(source)
                import sys
                print(f"[TimeSynchronizer] First sample for '{source}': timestamp={timestamp:.2f}s, required={config.required if config else 'N/A'}", flush=True)
                sys.stdout.flush()

            # Prune old samples
            cutoff = timestamp - self.buffer_duration_sec
            while self._buffers[source] and self._buffers[source][0].timestamp < cutoff:
                self._buffers[source].popleft()

    def get_synchronized(self, target_time: float) -> Optional[Dict[str, Any]]:
        """
        Get synchronized sensor readings for the target time.

        Args:
            target_time: Target timestamp in seconds

        Returns:
            Dictionary mapping sensor names to data, or None if required sensors missing
        """
        with self._lock:
            result = {}

            for name, config in self._configs.items():
                buffer = self._buffers.get(name)

                if not buffer or len(buffer) == 0:
                    if config.required:
                        return None
                    result[name] = None
                    continue

                # Check if data is too old
                newest = buffer[-1]
                if target_time - newest.timestamp > config.max_age_sec:
                    if config.required:
                        return None
                    result[name] = None
                    continue

                # Get synchronized data
                if config.interpolate:
                    result[name] = self._interpolate(name, buffer, target_time)
                else:
                    result[name] = self._nearest(buffer, target_time)

            self._last_sync_time = target_time
            return result

    def _nearest(self, buffer: Deque[SensorSample], target: float) -> Any:
        """Find nearest sample to target time."""
        if not buffer:
            return None

        best = min(buffer, key=lambda s: abs(s.timestamp - target))
        return best.data

    def _interpolate(self, name: str, buffer: Deque[SensorSample], target: float) -> Any:
        """
        Linear interpolation between two nearest samples.

        For complex types, uses custom interpolator if provided,
        otherwise falls back to nearest.
        """
        if not buffer:
            return None

        # Find bracketing samples
        before = None
        after = None

        for sample in buffer:
            if sample.timestamp <= target:
                before = sample
            elif sample.timestamp > target and after is None:
                after = sample
                break

        # Edge cases
        if before is None:
            return buffer[0].data if buffer else None
        if after is None:
            return before.data

        # Compute interpolation factor
        dt = after.timestamp - before.timestamp
        if dt < 1e-9:
            return before.data

        alpha = (target - before.timestamp) / dt

        # Use custom interpolator if available
        if name in self._interpolators:
            return self._interpolators[name](before.data, after.data, alpha)

        # Default: return nearest (no interpolation for complex types)
        return before.data if alpha < 0.5 else after.data

    def get_latest_time(self) -> Optional[float]:
        """
        Get the latest timestamp from any required sensor buffer.

        Returns:
            The most recent timestamp from required sensors, or None if no data.
        """
        with self._lock:
            latest = None
            for name, config in self._configs.items():
                if not config.required:
                    continue
                buffer = self._buffers.get(name)
                if buffer and len(buffer) > 0:
                    newest = buffer[-1].timestamp
                    if latest is None or newest < latest:
                        # Use the oldest "newest" among required sensors
                        # This ensures all required sensors have data up to this time
                        latest = newest
            return latest

    def get_buffer_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information about all buffers.

        Returns:
            Dictionary with buffer statistics
        """
        with self._lock:
            status = {}
            for name, buffer in self._buffers.items():
                config = self._configs.get(name)
                status[name] = {
                    'count': len(buffer),
                    'samples_received': self._sample_counts.get(name, 0),
                    'required': config.required if config else False,
                    'interpolate': config.interpolate if config else False,
                    'oldest': buffer[0].timestamp if buffer else None,
                    'newest': buffer[-1].timestamp if buffer else None,
                }
            return status

    def clear(self) -> None:
        """Clear all buffers."""
        with self._lock:
            for buffer in self._buffers.values():
                buffer.clear()
            self._sample_counts = {k: 0 for k in self._sample_counts}


# Interpolation helpers for common message types

def interpolate_float(a: float, b: float, alpha: float) -> float:
    """Linear interpolation for scalar values."""
    return a + alpha * (b - a)


def interpolate_odometry(odom_a, odom_b, alpha: float):
    """
    Interpolate between two Odometry messages.

    Interpolates position and velocity linearly.
    For orientation, uses nearest (proper SLERP is complex).
    """
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Point, Quaternion, Twist, Vector3

    result = Odometry()
    result.header = odom_b.header  # Use newer header
    result.child_frame_id = odom_b.child_frame_id

    # Interpolate position
    result.pose.pose.position = Point(
        x=interpolate_float(odom_a.pose.pose.position.x, odom_b.pose.pose.position.x, alpha),
        y=interpolate_float(odom_a.pose.pose.position.y, odom_b.pose.pose.position.y, alpha),
        z=interpolate_float(odom_a.pose.pose.position.z, odom_b.pose.pose.position.z, alpha),
    )

    # Use nearest orientation (proper interpolation would use SLERP)
    result.pose.pose.orientation = odom_a.pose.pose.orientation if alpha < 0.5 else odom_b.pose.pose.orientation

    # Interpolate velocities
    result.twist.twist.linear = Vector3(
        x=interpolate_float(odom_a.twist.twist.linear.x, odom_b.twist.twist.linear.x, alpha),
        y=interpolate_float(odom_a.twist.twist.linear.y, odom_b.twist.twist.linear.y, alpha),
        z=interpolate_float(odom_a.twist.twist.linear.z, odom_b.twist.twist.linear.z, alpha),
    )
    result.twist.twist.angular = Vector3(
        x=interpolate_float(odom_a.twist.twist.angular.x, odom_b.twist.twist.angular.x, alpha),
        y=interpolate_float(odom_a.twist.twist.angular.y, odom_b.twist.twist.angular.y, alpha),
        z=interpolate_float(odom_a.twist.twist.angular.z, odom_b.twist.twist.angular.z, alpha),
    )

    return result
