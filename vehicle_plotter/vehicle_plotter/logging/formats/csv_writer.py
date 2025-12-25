"""
CSV format writer.

Simple, human-readable format for debugging and compatibility.
"""

import csv
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base_format import FormatWriter


class CSVWriter(FormatWriter):
    """
    CSV format writer.

    Features:
    - Human-readable output
    - Universal compatibility
    - No additional dependencies
    - Good for debugging

    Limitations:
    - Larger file size than compressed formats
    - Slower reads for large datasets
    - No type enforcement
    """

    def __init__(self):
        self._path: Optional[Path] = None
        self._file = None
        self._writer = None
        self._schema: Dict[str, type] = {}
        self._columns: List[str] = []
        self._bytes_written = 0
        self._records_written = 0

    def open(self, path: Path, schema: Dict[str, type]) -> None:
        """Open CSV file for writing."""
        self._path = path
        self._schema = schema
        self._columns = list(schema.keys())

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Open file and write header
        self._file = open(path, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self._columns)
        self._writer.writeheader()

        self._bytes_written = 0
        self._records_written = 0

    def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write a batch of records to CSV."""
        if self._writer is None:
            raise RuntimeError("File not open. Call open() first.")

        for record in records:
            # Filter to only include configured columns
            filtered = {k: record.get(k, '') for k in self._columns}
            self._writer.writerow(filtered)
            self._records_written += 1

    def flush(self) -> None:
        """Flush to disk."""
        if self._file is not None:
            self._file.flush()
            # Update bytes written
            self._bytes_written = self._path.stat().st_size if self._path.exists() else 0

    def close(self) -> None:
        """Close the CSV file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

            # Final bytes count
            if self._path and self._path.exists():
                self._bytes_written = self._path.stat().st_size

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    @property
    def records_written(self) -> int:
        return self._records_written
