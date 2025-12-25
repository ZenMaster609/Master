"""
Parquet format writer.

Columnar format with compression, excellent for time-series analysis.
"""

from typing import List, Dict, Any, Optional, Literal
from pathlib import Path

from .base_format import FormatWriter


class ParquetWriter(FormatWriter):
    """
    Apache Parquet format writer.

    Features:
    - Columnar storage (efficient for analysis)
    - Built-in compression (70-90% smaller than CSV)
    - Fast reads (10-100x faster than CSV)
    - Strong typing with schema enforcement
    - Native support in pandas, polars, DuckDB

    Requires: pip install pyarrow
    """

    def __init__(self, compression: Literal["snappy", "gzip", "zstd", "none"] = "snappy"):
        """
        Initialize Parquet writer.

        Args:
            compression: Compression algorithm to use
        """
        self._compression = compression if compression != "none" else None
        self._path: Optional[Path] = None
        self._schema: Dict[str, type] = {}
        self._pa_schema = None
        self._writer = None
        self._buffer: List[Dict[str, Any]] = []
        self._bytes_written = 0
        self._records_written = 0

        # Check if pyarrow is available
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Check if pyarrow is available."""
        try:
            import pyarrow
            import pyarrow.parquet
            return True
        except ImportError:
            return False

    def _python_to_arrow_type(self, py_type: type):
        """Convert Python type to PyArrow type."""
        import pyarrow as pa

        type_map = {
            float: pa.float64(),
            int: pa.int64(),
            str: pa.string(),
            bool: pa.bool_(),
        }
        return type_map.get(py_type, pa.string())

    def open(self, path: Path, schema: Dict[str, type]) -> None:
        """Open Parquet file for writing."""
        if not self._available:
            raise RuntimeError("PyArrow not available. Install with: pip install pyarrow")

        import pyarrow as pa
        import pyarrow.parquet as pq

        self._path = path
        self._schema = schema

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build PyArrow schema
        fields = [
            (name, self._python_to_arrow_type(py_type))
            for name, py_type in schema.items()
        ]
        self._pa_schema = pa.schema(fields)

        # Create writer
        self._writer = pq.ParquetWriter(
            str(path),
            self._pa_schema,
            compression=self._compression,
        )

        self._buffer = []
        self._bytes_written = 0
        self._records_written = 0

    def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write a batch of records to Parquet."""
        if self._writer is None:
            raise RuntimeError("File not open. Call open() first.")

        import pyarrow as pa

        # Prepare columnar data
        columns = {name: [] for name in self._schema.keys()}

        for record in records:
            for name in self._schema.keys():
                value = record.get(name)
                # Handle None values
                if value is None:
                    if self._schema[name] == float:
                        value = float('nan')
                    elif self._schema[name] == int:
                        value = 0
                    elif self._schema[name] == bool:
                        value = False
                    else:
                        value = ""
                columns[name].append(value)

        # Create PyArrow table
        arrays = [pa.array(columns[name]) for name in self._schema.keys()]
        table = pa.Table.from_arrays(arrays, schema=self._pa_schema)

        # Write table
        self._writer.write_table(table)
        self._records_written += len(records)

    def flush(self) -> None:
        """Flush is automatic with Parquet row groups."""
        # Update bytes written estimate
        if self._path and self._path.exists():
            self._bytes_written = self._path.stat().st_size

    def close(self) -> None:
        """Close the Parquet file."""
        if self._writer is not None:
            self._writer.close()
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

    def is_available(self) -> bool:
        """Check if Parquet writing is available."""
        return self._available
