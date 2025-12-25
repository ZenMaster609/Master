"""
PyQtGraph plotting backend.

High-performance real-time plotting using PyQtGraph.
Supports 60+ Hz update rates with interactive pan/zoom.
"""

from typing import Dict, List, Optional
import numpy as np

from .base_backend import PlotBackend
from ..plot_config import PlotConfig, PlotLayoutConfig


class PyQtGraphBackend(PlotBackend):
    """
    PyQtGraph-based plotting backend.

    Features:
    - 60+ Hz update capability
    - Interactive pan/zoom
    - Low CPU usage via OpenGL acceleration
    - Works in Docker/WSL via X11
    """

    def __init__(self):
        self._app = None
        self._win = None
        self._plots: Dict[str, dict] = {}
        self._layout: Optional[PlotLayoutConfig] = None
        self._is_closed = False

        # Import check
        self._pg = None
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Check if PyQtGraph is available and display is accessible."""
        import os

        # Check if we have a display
        display = os.environ.get('DISPLAY', '')
        if not display:
            print("WARNING: No DISPLAY environment variable set, GUI disabled")
            return False

        # Check if running in headless mode
        qt_platform = os.environ.get('QT_QPA_PLATFORM', '')
        if qt_platform == 'offscreen':
            print("WARNING: QT_QPA_PLATFORM=offscreen, GUI disabled")
            return False

        # Test if X11 display is actually reachable
        if not self._test_x11_connection(display):
            print(f"WARNING: Cannot connect to X11 display '{display}', GUI disabled")
            # Set offscreen mode to prevent Qt crash
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            return False

        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtWidgets
            self._pg = pg
            return True
        except ImportError:
            return False

    def _test_x11_connection(self, display: str) -> bool:
        """Test if we can actually connect to the X11 display."""
        import socket

        try:
            # Parse display string (e.g., "host.docker.internal:0" or ":0")
            if ':' in display:
                host_part, display_num = display.rsplit(':', 1)
                host = host_part if host_part else 'localhost'
                port = 6000 + int(display_num.split('.')[0])
            else:
                return False

            # Try to connect to X11 port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def is_available(self) -> bool:
        return self._available

    def initialize(self, layout: PlotLayoutConfig) -> None:
        """Initialize the PyQtGraph window."""
        if not self._available:
            raise RuntimeError("PyQtGraph not available. Install with: pip install pyqtgraph PyQt5")

        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets

        try:
            # Create application if needed
            self._app = QtWidgets.QApplication.instance()
            if self._app is None:
                self._app = QtWidgets.QApplication([])

            # Configure PyQtGraph
            if layout.dark_mode:
                pg.setConfigOption('background', '#1e1e1e')
                pg.setConfigOption('foreground', '#d4d4d4')
            else:
                pg.setConfigOption('background', 'w')
                pg.setConfigOption('foreground', 'k')

            pg.setConfigOption('antialias', True)

            # Create main window
            self._win = pg.GraphicsLayoutWidget(title=layout.window_title)
            self._win.resize(*layout.window_size)
            self._win.show()

            self._layout = layout
            self._is_closed = False

            # Connect close event
            self._win.closeEvent = self._on_close

        except Exception as e:
            # Display not available (common in Docker/headless)
            print(f"WARNING: Could not initialize PyQtGraph display: {e}")
            print("Falling back to headless mode (no GUI)")
            self._available = False
            self._is_closed = True
            self._win = None

    def _on_close(self, event):
        """Handle window close event."""
        self._is_closed = True
        event.accept()

    def add_plot(self, config: PlotConfig) -> str:
        """Add a new plot to the layout."""
        if not self._available or self._win is None:
            return config.name

        import pyqtgraph as pg

        # Create plot item at specified grid position
        plot_item = self._win.addPlot(
            row=config.row,
            col=config.col,
            rowspan=config.row_span,
            colspan=config.col_span,
            title=config.name,
        )

        # Configure plot
        plot_item.showGrid(x=config.show_grid, y=config.show_grid)

        if config.show_legend:
            plot_item.addLegend()

        # Set axis labels
        if config.x_axis is not None:
            label = config.x_axis.label
            if config.x_axis.unit:
                label += f" ({config.x_axis.unit})"
            plot_item.setLabel('bottom', label)
        elif config.plot_type == "timeseries":
            plot_item.setLabel('bottom', 'Time (s)')

        if config.y_axis is not None:
            label = config.y_axis.label
            if config.y_axis.unit:
                label += f" ({config.y_axis.unit})"
            plot_item.setLabel('left', label)

            # Set fixed limits if specified
            if config.y_axis.limits:
                plot_item.setYRange(*config.y_axis.limits)
                plot_item.enableAutoRange(axis='y', enable=config.y_axis.auto_scale)

        # Create curves for each series
        curves = {}
        colors = self._get_colors(len(config.series))

        for i, series in enumerate(config.series):
            color = series.color if series.color != "auto" else colors[i]
            pen = self._create_pen(color, series.line_width, series.line_style)

            curve = plot_item.plot(
                [],
                [],
                name=series.name,
                pen=pen,
                symbol=series.marker,
            )
            curves[series.name] = curve

        # Store plot info
        self._plots[config.name] = {
            'item': plot_item,
            'curves': curves,
            'config': config,
        }

        return config.name

    def _create_pen(self, color: str, width: float, style: str):
        """Create a pen with the specified properties."""
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore

        style_map = {
            'solid': QtCore.Qt.PenStyle.SolidLine,
            'dashed': QtCore.Qt.PenStyle.DashLine,
            'dotted': QtCore.Qt.PenStyle.DotLine,
        }

        return pg.mkPen(color=color, width=width, style=style_map.get(style, QtCore.Qt.PenStyle.SolidLine))

    def _get_colors(self, n: int) -> List[str]:
        """Get n distinct colors for series."""
        # Default color palette (colorblind-friendly)
        palette = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf',  # cyan
        ]
        return [palette[i % len(palette)] for i in range(n)]

    def remove_plot(self, plot_id: str) -> None:
        """Remove a plot from the layout."""
        if plot_id in self._plots:
            plot_info = self._plots[plot_id]
            self._win.removeItem(plot_info['item'])
            del self._plots[plot_id]

    def update_data(
        self,
        plot_id: str,
        x_data: List[float],
        y_data: Dict[str, List[float]],
    ) -> None:
        """Update plot data."""
        if plot_id not in self._plots:
            return

        plot_info = self._plots[plot_id]
        curves = plot_info['curves']

        # Convert to numpy for performance
        x_arr = np.array(x_data) if x_data else np.array([])

        for name, curve in curves.items():
            if name in y_data:
                y_arr = np.array(y_data[name]) if y_data[name] else np.array([])

                # Ensure same length
                min_len = min(len(x_arr), len(y_arr))
                if min_len > 0:
                    curve.setData(x_arr[:min_len], y_arr[:min_len])

    def process_events(self) -> bool:
        """Process Qt events, return False if window closed."""
        if self._is_closed:
            return False

        if self._app is not None:
            self._app.processEvents()

        return not self._is_closed

    def close(self) -> None:
        """Close the window."""
        if self._win is not None:
            self._win.close()
            self._win = None
        self._is_closed = True

    def export_figure(self, path: str, dpi: int = 150) -> None:
        """Export current view to image file."""
        if self._win is None:
            return

        import pyqtgraph.exporters as exporters

        # Use ImageExporter for raster formats
        if path.endswith('.png') or path.endswith('.jpg'):
            exporter = exporters.ImageExporter(self._win.scene())
            exporter.parameters()['width'] = int(self._layout.window_size[0] * dpi / 96)
            exporter.export(path)
        elif path.endswith('.svg'):
            exporter = exporters.SVGExporter(self._win.scene())
            exporter.export(path)
