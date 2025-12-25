"""Sensor adapters for different hardware/simulation backends."""

from .gazebo_adapter import GazeboAdapter

__all__ = [
    'GazeboAdapter',
]
