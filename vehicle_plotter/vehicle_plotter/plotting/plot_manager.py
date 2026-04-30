"""
PlotManager - Manages multiple real-time plots with configurable layout.

Provides a high-level interface for:
- Adding/removing plots dynamically
- Pushing vehicle state data to plots
- Handling different X-axis types
- Exporting plot images and data
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import csv

from .plot_config import PlotConfig, PlotLayoutConfig, XAxisType, get_default_plots
from .matplotlib_fonts import apply_serif_font_preferences
from .backends.base_backend import PlotBackend, DummyBackend
from .backends.pyqtgraph_backend import PyQtGraphBackend
from ..core.vehicle_state import VehicleState
from ..utils.ring_buffer import RingBuffer


STATIC_DASHBOARD_TICK_FONTSIZE = 15.0
STATIC_DASHBOARD_TITLE_FONTSIZE = 15.0
STATIC_DASHBOARD_Y_LABEL_FONTSIZE = 15.0
STATIC_DASHBOARD_BOTTOM_X_LABEL_FONTSIZE = 15.0
STATIC_DASHBOARD_LEGEND_FONTSIZE = 15.0


class PlotManager:
    """
    Manages real-time plotting of vehicle state data.

    Features:
    - Configurable plot layout (grid of subplots)
    - Dynamic add/remove plots
    - Multiple X-axis types (time, distance, yaw, encoder RPM)
    - Thread-safe data updates
    - Backend abstraction (PyQtGraph primary, matplotlib fallback)

    Example:
        from vehicle_plotter.plotting import PlotManager, get_default_plots

        # Create manager with default plots
        manager = PlotManager(layout_config=get_default_plots())

        # In ROS callback:
        manager.push_state(vehicle_state)

        # In timer callback:
        if not manager.refresh():
            # Window was closed
            pass
    """

    def __init__(
        self,
        layout_config: Optional[PlotLayoutConfig] = None,
        backend: str = "pyqtgraph",
        enable_gui: bool = True,
    ):
        """
        Initialize the plot manager.

        Args:
            layout_config: Plot layout configuration (uses default if None)
            backend: Backend to use ('pyqtgraph', 'matplotlib', 'dummy')
            enable_gui: If False, uses dummy backend (for headless)
        """
        self._lock = threading.Lock()
        self.layout = layout_config or get_default_plots()

        # Initialize backend
        if not enable_gui:
            self._backend: PlotBackend = DummyBackend()
        elif backend == "pyqtgraph":
            pg_backend = PyQtGraphBackend()
            if pg_backend.is_available():
                self._backend = pg_backend
            else:
                print("WARNING: PyQtGraph not available, using dummy backend")
                self._backend = DummyBackend()
        elif backend == "dummy":
            self._backend = DummyBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Data buffers for each plot (keyed by plot name)
        self._buffers: Dict[str, Dict[str, RingBuffer]] = {}

        # Plot configurations (keyed by plot name)
        self._configs: Dict[str, PlotConfig] = {}

        # X-axis value generators
        self._x_generators = {
            XAxisType.TIME: self._get_time_value,
            XAxisType.DISTANCE: self._get_distance_value,
            XAxisType.YAW: self._get_yaw_value,
            XAxisType.ENCODER_TOTAL: self._get_encoder_total_value,
            XAxisType.ENCODER_FL: self._get_encoder_fl_value,
        }

        # Track time offset for relative time display
        self._time_offset: Optional[float] = None

        # Initialize backend with layout
        self._backend.initialize(self.layout)

        # Add initial plots from config
        for plot_config in self.layout.plots:
            self.add_plot(plot_config)

    def add_plot(self, config: PlotConfig) -> None:
        """Add a new plot to the layout."""
        with self._lock:
            if config.name in self._configs:
                raise ValueError(f"Plot '{config.name}' already exists")

            # Store config
            self._configs[config.name] = config

            # Create data buffers
            self._buffers[config.name] = {
                'x': RingBuffer(config.buffer_size),
            }
            for series in config.series:
                self._buffers[config.name][series.name] = RingBuffer(config.buffer_size)

            # Add to backend
            self._backend.add_plot(config)

    def remove_plot(self, name: str) -> None:
        """Remove a plot from the layout."""
        with self._lock:
            if name not in self._configs:
                return

            del self._configs[name]
            del self._buffers[name]
            self._backend.remove_plot(name)

    def update_plot(self, name: str, **kwargs) -> None:
        """Update plot configuration."""
        with self._lock:
            if name not in self._configs:
                return

            config = self._configs[name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # Resize buffer if needed
            if 'buffer_size' in kwargs:
                for buffer in self._buffers[name].values():
                    buffer.resize(kwargs['buffer_size'])

    def push_state(self, state: VehicleState) -> None:
        """
        Push new vehicle state to all plots.

        Called from data collector node callback.

        Args:
            state: Current vehicle state
        """
        # Set time offset on first state
        if self._time_offset is None:
            self._time_offset = state.timestamp

        with self._lock:
            for name, config in self._configs.items():
                buffers = self._buffers[name]

                # Get X value based on axis type
                if config.x_axis is not None and config.x_axis.variable:
                    # Custom X axis from VehicleState attribute
                    x_val = self._get_attribute(state, config.x_axis.variable)
                    if x_val is None:
                        x_val = 0.0
                    x_val = x_val * config.x_axis.scale + config.x_axis.offset
                    if config.x_axis.transform:
                        x_val = config.x_axis.transform(x_val)
                else:
                    # Standard X axis type
                    x_val = self._x_generators[config.x_axis_type](state)

                buffers['x'].push(x_val)

                # Get Y values for each series
                for series in config.series:
                    y_val = self._get_attribute(state, series.variable)
                    if y_val is not None:
                        y_val = y_val * series.scale + series.offset
                        if series.transform:
                            y_val = series.transform(y_val)
                    else:
                        y_val = 0.0
                    buffers[series.name].push(y_val)

    def refresh(self) -> bool:
        """
        Refresh all plots with current buffer data.

        Should be called at the desired update rate (e.g., 30 Hz).

        Returns:
            False if window was closed, True otherwise
        """
        with self._lock:
            for name, config in self._configs.items():
                buffers = self._buffers[name]

                x_data = buffers['x'].to_list()
                y_data = {
                    series.name: buffers[series.name].to_list()
                    for series in config.series
                }

                self._backend.update_data(name, x_data, y_data)

        return self._backend.process_events()

    def close(self) -> None:
        """Close the plot window and clean up."""
        self._backend.close()

    def export(self, path: str, dpi: int = 150) -> None:
        """Export current view to image file."""
        self._backend.export_figure(path, dpi)

    def export_static_dashboard(self, path: str, dpi: int = 150) -> None:
        """Export the buffered plots as one static dashboard image/PDF."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        apply_serif_font_preferences()

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(
            figsize=(
                max(8.0, self.layout.window_size[0] / 100.0),
                max(6.0, self.layout.window_size[1] / 100.0),
            )
        )
        grid = fig.add_gridspec(self.layout.rows, self.layout.cols, hspace=0.35, wspace=0.25)

        with self._lock:
            for config in self.layout.plots:
                ax = fig.add_subplot(
                    grid[
                        config.row: config.row + config.row_span,
                        config.col: config.col + config.col_span,
                    ]
                )
                buffers = self._buffers.get(config.name)
                if buffers is None:
                    continue

                x_data = buffers['x'].to_list()
                if not x_data:
                    ax.set_title(config.name, fontsize=STATIC_DASHBOARD_TITLE_FONTSIZE)
                    ax.grid(config.show_grid, alpha=0.3)
                    ax.set_xlabel(self._x_axis_label(config))
                    ax.set_ylabel(self._y_axis_label(config))
                    self._format_static_dashboard_axis_text(ax, config)
                    continue

                for series in config.series:
                    y_data = buffers[series.name].to_list()
                    min_len = min(len(x_data), len(y_data))
                    if min_len <= 0:
                        continue

                    style = {
                        'solid': '-',
                        'dashed': '--',
                        'dotted': ':',
                    }.get(series.line_style, '-')
                    color = None if series.color == 'auto' else series.color
                    ax.plot(
                        x_data[:min_len],
                        y_data[:min_len],
                        style,
                        label=series.name,
                        color=color,
                        linewidth=series.line_width,
                    )

                ax.set_title(config.name, fontsize=STATIC_DASHBOARD_TITLE_FONTSIZE)
                ax.grid(config.show_grid, alpha=0.3)
                ax.set_xlabel(self._x_axis_label(config))
                ax.set_ylabel(self._y_axis_label(config))
                self._format_static_dashboard_axis_text(ax, config)

                if config.plot_type == 'xy':
                    ax.set_aspect('equal', adjustable='datalim')

                if config.x_axis is not None and config.x_axis.limits:
                    ax.set_xlim(*config.x_axis.limits)
                if config.y_axis is not None and config.y_axis.limits:
                    ax.set_ylim(*config.y_axis.limits)

                if config.show_legend:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(loc='best', fontsize=STATIC_DASHBOARD_LEGEND_FONTSIZE)

        fig.tight_layout()
        fig.savefig(path_obj, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    def _format_static_dashboard_axis_text(self, ax, config: PlotConfig) -> None:
        ax.tick_params(
            axis='both',
            which='both',
            labelsize=STATIC_DASHBOARD_TICK_FONTSIZE,
        )
        ax.yaxis.label.set_size(STATIC_DASHBOARD_Y_LABEL_FONTSIZE)

        is_bottom_row = config.row + config.row_span >= self.layout.rows
        if is_bottom_row:
            ax.xaxis.label.set_size(STATIC_DASHBOARD_BOTTOM_X_LABEL_FONTSIZE)
        else:
            ax.set_xlabel('')

    def export_data(self, output_dir: Path) -> Dict[str, Path]:
        """
        Export plot buffer data to CSV files.

        Creates one CSV file per plot containing all series data.

        Args:
            output_dir: Directory to save CSV files

        Returns:
            Dictionary mapping plot names to saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        with self._lock:
            for name, config in self._configs.items():
                buffers = self._buffers[name]

                # Get X and Y data
                x_data = buffers['x'].to_list()
                if not x_data:
                    continue

                # Build rows
                rows = []
                header = ['x'] + [s.name for s in config.series]

                for i, x_val in enumerate(x_data):
                    row = [x_val]
                    for series in config.series:
                        series_data = buffers[series.name].to_list()
                        if i < len(series_data):
                            row.append(series_data[i])
                        else:
                            row.append('')
                    rows.append(row)

                # Write CSV
                safe_name = name.replace(' ', '_').replace('/', '_').lower()
                csv_path = output_dir / f"{safe_name}.csv"

                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)

                exported_files[name] = csv_path

        return exported_files

    def get_plot_names(self) -> List[str]:
        """Get list of current plot names."""
        with self._lock:
            return list(self._configs.keys())

    def clear_buffers(self) -> None:
        """Clear all data buffers."""
        with self._lock:
            for buffers in self._buffers.values():
                for buffer in buffers.values():
                    buffer.clear()
        self._time_offset = None

    # X-axis value generators
    def _get_time_value(self, state: VehicleState) -> float:
        """Get relative time since start."""
        if self._time_offset is None:
            return 0.0
        return state.timestamp - self._time_offset

    def _get_distance_value(self, state: VehicleState) -> float:
        return state.distance_traveled

    def _get_yaw_value(self, state: VehicleState) -> float:
        return state.yaw

    def _get_encoder_total_value(self, state: VehicleState) -> float:
        return float(sum(state.encoder_velocities))

    def _get_encoder_fl_value(self, state: VehicleState) -> float:
        return float(state.encoder_velocities[0]) if state.encoder_velocities else 0.0

    def _get_attribute(self, state: VehicleState, variable: str) -> Optional[float]:
        """
        Get attribute from VehicleState, supporting indexed access.

        Examples:
            "x" -> state.x
            "vx" -> state.vx
            "encoder_velocities[0]" -> state.encoder_velocities[0]
            "encoder_velocities[2]" -> state.encoder_velocities[2]
        """
        try:
            if '[' in variable:
                # Indexed access: encoder_velocities[0]
                attr_name, idx_str = variable.split('[')
                idx = int(idx_str.rstrip(']'))
                attr_value = getattr(state, attr_name, None)
                if attr_value is not None and len(attr_value) > idx:
                    return float(attr_value[idx])
                return None
            else:
                value = getattr(state, variable, None)
                if value is not None:
                    return float(value)
                return None
        except (ValueError, TypeError, IndexError):
            return None

    @staticmethod
    def _format_axis_label(label: str, unit: str) -> str:
        if label and unit:
            return f"{label} ({unit})"
        return label or unit or ""

    def _x_axis_label(self, config: PlotConfig) -> str:
        if config.x_axis is not None:
            return self._format_axis_label(config.x_axis.label, config.x_axis.unit)

        if config.x_axis_type == XAxisType.TIME:
            return "Time (s)"
        if config.x_axis_type == XAxisType.DISTANCE:
            return "Distance (m)"
        if config.x_axis_type == XAxisType.YAW:
            return "Yaw (rad)"
        if config.x_axis_type == XAxisType.ENCODER_TOTAL:
            return "Encoder Total (RPM)"
        if config.x_axis_type == XAxisType.ENCODER_FL:
            return "Encoder FL (RPM)"
        return "X"

    def _y_axis_label(self, config: PlotConfig) -> str:
        if config.y_axis is not None:
            return self._format_axis_label(config.y_axis.label, config.y_axis.unit)
        return "Value"
