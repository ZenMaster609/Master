"""
Abstract base class for log format writers.

Defines the interface that all format writers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path


class FormatWriter(ABC):
    """
    Abstract base class for log format writers.

    Implementations include:
    - ParquetWriter: Columnar format, excellent for analysis
    - CSVWriter: Human-readable, universal compatibility
    - HDF5Writer: Hierarchical, good for complex data (optional)
    """

    @abstractmethod
    def open(self, path: Path, schema: Dict[str, type]) -> None:
        """
        Open file for writing with given schema.

        Args:
            path: Output file path
            schema: Dictionary mapping column names to Python types
        """
        pass

    @abstractmethod
    def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """
        Write a batch of records.

        Args:
            records: List of dictionaries, each representing one row
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush buffered data to disk."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close file and finalize."""
        pass

    @property
    @abstractmethod
    def bytes_written(self) -> int:
        """Return total bytes written."""
        pass

    @property
    @abstractmethod
    def records_written(self) -> int:
        """Return total records written."""
        pass
