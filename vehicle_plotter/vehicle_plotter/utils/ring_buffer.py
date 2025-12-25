"""
RingBuffer - Efficient circular buffer for real-time data.

Provides O(1) append and efficient numpy array conversion for plotting.
"""

from typing import List, Optional, Any
from collections import deque
import numpy as np


class RingBuffer:
    """
    Efficient circular buffer for time-series data.

    Optimized for:
    - O(1) append operations
    - Efficient conversion to numpy arrays
    - Thread-safe (single writer, multiple readers)

    Example:
        buffer = RingBuffer(maxlen=1000)
        buffer.push(1.0)
        buffer.push(2.0)

        data = buffer.to_numpy()  # [1.0, 2.0]
        latest = buffer.latest(10)  # Last 10 values
    """

    def __init__(self, maxlen: int = 1000, dtype: type = float):
        """
        Initialize the ring buffer.

        Args:
            maxlen: Maximum number of elements to store
            dtype: Data type for numpy conversion
        """
        self.maxlen = maxlen
        self.dtype = dtype
        self._buffer: deque = deque(maxlen=maxlen)

    def push(self, value: Any) -> None:
        """
        Add a value to the buffer.

        If buffer is full, oldest value is discarded.

        Args:
            value: Value to add
        """
        self._buffer.append(value)

    def extend(self, values: List[Any]) -> None:
        """
        Add multiple values to the buffer.

        Args:
            values: List of values to add
        """
        self._buffer.extend(values)

    def clear(self) -> None:
        """Clear all values from the buffer."""
        self._buffer.clear()

    def resize(self, new_maxlen: int) -> None:
        """
        Resize the buffer.

        If new size is smaller, oldest values are discarded.

        Args:
            new_maxlen: New maximum length
        """
        if new_maxlen < self.maxlen:
            # Keep most recent values
            keep = list(self._buffer)[-new_maxlen:]
            self._buffer = deque(keep, maxlen=new_maxlen)
        else:
            # Create new buffer with larger size
            old_data = list(self._buffer)
            self._buffer = deque(old_data, maxlen=new_maxlen)

        self.maxlen = new_maxlen

    def to_list(self) -> List[Any]:
        """
        Convert buffer to a list.

        Returns:
            List of all values in order (oldest first)
        """
        return list(self._buffer)

    def to_numpy(self) -> np.ndarray:
        """
        Convert buffer to numpy array.

        Returns:
            Numpy array of all values
        """
        if len(self._buffer) == 0:
            return np.array([], dtype=self.dtype)
        return np.array(self._buffer, dtype=self.dtype)

    def latest(self, n: int) -> List[Any]:
        """
        Get the N most recent values.

        Args:
            n: Number of values to return

        Returns:
            List of most recent N values
        """
        if n >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-n:]

    def oldest(self, n: int) -> List[Any]:
        """
        Get the N oldest values.

        Args:
            n: Number of values to return

        Returns:
            List of oldest N values
        """
        if n >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[:n]

    @property
    def last(self) -> Optional[Any]:
        """Get the most recent value, or None if empty."""
        if len(self._buffer) == 0:
            return None
        return self._buffer[-1]

    @property
    def first(self) -> Optional[Any]:
        """Get the oldest value, or None if empty."""
        if len(self._buffer) == 0:
            return None
        return self._buffer[0]

    def __len__(self) -> int:
        """Return the number of values in the buffer."""
        return len(self._buffer)

    def __bool__(self) -> bool:
        """Return True if buffer is not empty."""
        return len(self._buffer) > 0

    def __getitem__(self, idx: int) -> Any:
        """Get value by index."""
        return self._buffer[idx]

    def __iter__(self):
        """Iterate over values."""
        return iter(self._buffer)


class MultiChannelRingBuffer:
    """
    Ring buffer for multiple synchronized channels.

    All channels share the same time index and are pushed together.

    Example:
        buffer = MultiChannelRingBuffer(['x', 'y', 'vx', 'vy'], maxlen=1000)
        buffer.push({'x': 1.0, 'y': 2.0, 'vx': 0.5, 'vy': 0.1})

        x_data = buffer.get_channel('x')  # numpy array
        all_data = buffer.to_dict()  # {'x': [...], 'y': [...], ...}
    """

    def __init__(self, channels: List[str], maxlen: int = 1000, dtype: type = float):
        """
        Initialize multi-channel ring buffer.

        Args:
            channels: List of channel names
            maxlen: Maximum number of samples per channel
            dtype: Data type for numpy conversion
        """
        self.channels = channels
        self.maxlen = maxlen
        self.dtype = dtype

        self._buffers = {
            name: RingBuffer(maxlen=maxlen, dtype=dtype)
            for name in channels
        }

    def push(self, values: dict) -> None:
        """
        Add values for all channels.

        Args:
            values: Dictionary mapping channel names to values
        """
        for name in self.channels:
            value = values.get(name, 0.0)
            self._buffers[name].push(value)

    def get_channel(self, name: str) -> np.ndarray:
        """
        Get numpy array for a single channel.

        Args:
            name: Channel name

        Returns:
            Numpy array of channel values
        """
        return self._buffers[name].to_numpy()

    def to_dict(self) -> dict:
        """
        Get all channels as numpy arrays.

        Returns:
            Dictionary mapping channel names to numpy arrays
        """
        return {
            name: buffer.to_numpy()
            for name, buffer in self._buffers.items()
        }

    def to_lists(self) -> dict:
        """
        Get all channels as lists.

        Returns:
            Dictionary mapping channel names to lists
        """
        return {
            name: buffer.to_list()
            for name, buffer in self._buffers.items()
        }

    def clear(self) -> None:
        """Clear all channels."""
        for buffer in self._buffers.values():
            buffer.clear()

    def __len__(self) -> int:
        """Return number of samples (same for all channels)."""
        if self.channels:
            return len(self._buffers[self.channels[0]])
        return 0
