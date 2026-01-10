"""
LogWriter - Handles buffered writing of vehicle state to files.

Provides automatic buffering, file management, and session tracking.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import threading

from .log_config import LogConfig
from .formats.base_format import FormatWriter
from .formats.parquet_writer import ParquetWriter
from .formats.csv_writer import CSVWriter
from ..core.vehicle_state import VehicleState


class LogWriter:
    """
    Buffered log writer for vehicle state data.

    Features:
    - Configurable output format (Parquet, CSV)
    - Buffered writes for performance
    - Automatic file splitting on size limit
    - Thread-safe operation
    - Graceful shutdown with flush

    Example:
        config = LogConfig(format='parquet')
        writer = LogWriter(config)

        # In ROS callback:
        writer.write(vehicle_state)

        # On shutdown:
        writer.close()
    """

    def __init__(self, config: LogConfig):
        """
        Initialize the log writer.

        Args:
            config: Logging configuration
        """
        self.config = config
        self._lock = threading.Lock()

        # Create session directory
        self._session_path = self._create_session_directory()

        # Initialize format writer
        self._writer: FormatWriter = self._create_writer()

        # Write buffer
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_count = 0

        # File management
        self._file_index = 0
        self._current_file: Optional[Path] = None

        # Statistics
        self._total_records = 0
        self._session_start = datetime.now()

        # Open first file
        self._open_new_file()

        # Write session metadata
        self._write_metadata()

    def _create_session_directory(self) -> Path:
        """Create session directory.

        If session_name is empty string, writes directly to base_path.
        If session_name is None, generates a timestamped session name.
        Otherwise uses the provided session_name.
        """
        if self.config.session_name == '':
            # Empty string = use base_path directly (for RunSession integration)
            session_path = self.config.base_path
        elif self.config.session_name:
            # Explicit session name provided
            session_path = self.config.base_path / self.config.session_name
        else:
            # None = auto-generate timestamped name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{timestamp}"
            session_path = self.config.base_path / session_name

        session_path.mkdir(parents=True, exist_ok=True)

        return session_path

    def _create_writer(self) -> FormatWriter:
        """Create format writer based on config."""
        if self.config.format == "parquet":
            writer = ParquetWriter(compression=self.config.compression)
            if not writer.is_available():
                print("WARNING: PyArrow not available, falling back to CSV")
                return CSVWriter()
            return writer
        elif self.config.format == "csv":
            return CSVWriter()
        else:
            raise ValueError(f"Unknown format: {self.config.format}")

    def _open_new_file(self) -> None:
        """Open a new data file."""
        if self._current_file and self._writer:
            self._writer.close()

        filename = f"vehicle_state_{self._file_index:04d}.{self.config.format}"
        self._current_file = self._session_path / filename

        # Ensure session directory exists
        self._session_path.mkdir(parents=True, exist_ok=True)

        # Get schema from config
        schema = self.config.get_signal_schema()
        self._writer.open(self._current_file, schema)

        self._file_index += 1

    def _write_metadata(self) -> None:
        """Write session metadata file."""
        metadata = {
            'session_name': self._session_path.name,
            'created_at': self._session_start.isoformat(),
            'config': {
                'format': self.config.format,
                'compression': self.config.compression,
                'signals': self.config.signals,
                'flush_interval_sec': self.config.flush_interval_sec,
            },
            'files': [],
        }

        metadata_path = self._session_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def write(self, state: VehicleState) -> None:
        """
        Write a vehicle state to the log.

        Buffers data and flushes periodically.

        Args:
            state: Vehicle state to log
        """
        with self._lock:
            # Convert state to dict with configured signals only
            record = state.to_dict()
            filtered_record = {
                key: record.get(key)
                for key in self.config.signals
                if key in record
            }

            self._buffer.append(filtered_record)
            self._buffer_count += 1
            self._total_records += 1

            # Check if we should flush
            if self._buffer_count >= self.config.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffer to disk."""
        if not self._buffer:
            return

        # Check file size limit
        if self._writer.bytes_written > self.config.max_file_size_mb * 1024 * 1024:
            self._open_new_file()

        # Write batch
        self._writer.write_batch(self._buffer)
        self._buffer.clear()
        self._buffer_count = 0

    def flush(self) -> None:
        """Force flush buffer to disk."""
        with self._lock:
            self._flush_buffer()
            self._writer.flush()

    def close(self) -> None:
        """Close the log writer and finalize files."""
        with self._lock:
            self._flush_buffer()
            self._writer.close()

            # Update metadata with file list
            self._finalize_metadata()

    def _finalize_metadata(self) -> None:
        """Update metadata with final file information."""
        metadata_path = self._session_path / "metadata.json"

        if not metadata_path.exists():
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # List all data files
        pattern = f"vehicle_state_*.{self.config.format}"
        data_files = sorted(self._session_path.glob(pattern))

        metadata['files'] = [f.name for f in data_files]
        metadata['closed_at'] = datetime.now().isoformat()
        metadata['total_files'] = len(data_files)
        metadata['total_records'] = self._total_records

        # Calculate duration
        duration = datetime.now() - self._session_start
        metadata['duration_seconds'] = duration.total_seconds()

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @property
    def session_path(self) -> Path:
        """Return the session directory path."""
        return self._session_path

    @property
    def total_records(self) -> int:
        """Return total records written."""
        return self._total_records

    def get_status(self) -> Dict[str, Any]:
        """Get current status information."""
        with self._lock:
            return {
                'session_path': str(self._session_path),
                'current_file': str(self._current_file) if self._current_file else None,
                'file_index': self._file_index,
                'buffer_count': self._buffer_count,
                'total_records': self._total_records,
                'bytes_written': self._writer.bytes_written if self._writer else 0,
            }
