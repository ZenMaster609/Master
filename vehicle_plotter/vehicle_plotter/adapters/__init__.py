"""Sensor adapters for different hardware/simulation backends."""

from .gazebo_adapter import GazeboAdapter
from .can_adapter import CANAdapter

__all__ = [
    'GazeboAdapter',
    'CANAdapter',
]
