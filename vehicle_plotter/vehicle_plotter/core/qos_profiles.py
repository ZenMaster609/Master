"""
QoS (Quality of Service) profile definitions for vehicle_plotter.

These profiles ensure reliable communication between nodes and
compatibility with rosbag2 recording.
"""

from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


# For high-rate sensor data (IMU ~100Hz, Odom ~50Hz)
# Best effort to avoid blocking on slow subscribers
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# For slower, important sensor data (GPS ~5Hz)
# Reliable to ensure no samples are lost
RELIABLE_SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=50,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# For control commands (/cmd_vel)
# Reliable to ensure commands reach the controller
CONTROL_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# For internal vehicle_plotter topics (/vehicle_plotter/state)
# Transient local so late subscribers get recent data
PLOTTER_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=100,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

# For system clock topic (when using sim time)
CLOCK_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# For replay from rosbags recorded with transient_local durability
# Compatible with bags that used RELIABLE + TRANSIENT_LOCAL
REPLAY_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


def get_qos_for_topic(topic_name: str) -> QoSProfile:
    """
    Get appropriate QoS profile for a given topic name.

    Args:
        topic_name: The ROS topic name

    Returns:
        Appropriate QoSProfile for the topic
    """
    topic_lower = topic_name.lower()

    # GPS topics - reliable
    if 'gps' in topic_lower or 'navsat' in topic_lower or 'fix' in topic_lower:
        return RELIABLE_SENSOR_QOS

    # Control topics - reliable
    if 'cmd_vel' in topic_lower or 'command' in topic_lower:
        return CONTROL_QOS

    # Plotter internal topics - reliable with transient local
    if 'vehicle_plotter' in topic_lower:
        return PLOTTER_QOS

    # Clock topic
    if topic_lower == '/clock':
        return CLOCK_QOS

    # Default to sensor QoS for IMU, odom, etc.
    return SENSOR_QOS
