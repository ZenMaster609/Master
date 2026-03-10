"""
Abstract sensor adapter interfaces.

Defines the interface that all sensor adapters must implement,
allowing the data collector to work with the simulation backend.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from rclpy.node import Node

from .vehicle_state import VehicleState
from .time_sync import TimeSynchronizer


class SensorAdapterInterface(ABC):
    """
    Abstract base class for sensor adapters.

    Sensor adapters handle the specifics of subscribing to sensor
    topics and converting sensor data to VehicleState. Different
    adapters exist for simulation backends.

    Subclasses must implement:
    - setup_subscriptions(): Create ROS subscriptions
    - compute_state(): Convert synchronized sensor data to VehicleState
    """

    def __init__(
        self,
        node: Node,
        synchronizer: TimeSynchronizer,
        adapter_name: str = "unknown",
    ):
        """
        Initialize the sensor adapter.

        Args:
            node: ROS 2 node for creating subscriptions
            synchronizer: Time synchronizer for multi-rate data
            adapter_name: Name identifier for this adapter
        """
        self.node = node
        self.synchronizer = synchronizer
        self.adapter_name = adapter_name
        self._logger = node.get_logger()

    @abstractmethod
    def setup_subscriptions(self) -> None:
        """
        Create ROS 2 subscriptions for all sensor topics.

        Called once during initialization. Must register all sensors
        with the synchronizer via synchronizer.add_sensor().
        """
        pass

    @abstractmethod
    def compute_state(
        self,
        synced_data: Dict[str, Any],
        prev_state: Optional[VehicleState],
    ) -> VehicleState:
        """
        Convert synchronized sensor data to VehicleState.

        Args:
            synced_data: Dictionary of sensor data from synchronizer
            prev_state: Previous state (for computing derivatives)

        Returns:
            New VehicleState computed from sensor data
        """
        pass

    def get_adapter_name(self) -> str:
        """Return the adapter name."""
        return self.adapter_name

    def log_info(self, message: str) -> None:
        """Log an info message."""
        self._logger.info(f"[{self.adapter_name}] {message}")

    def log_warn(self, message: str) -> None:
        """Log a warning message."""
        self._logger.warn(f"[{self.adapter_name}] {message}")

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(f"[{self.adapter_name}] {message}")
