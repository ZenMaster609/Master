"""
Abstract base class for plotting backends.

Defines the interface that all plotting backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..plot_config import PlotConfig, PlotLayoutConfig


class PlotBackend(ABC):
    """
    Abstract base class for plotting backends.

    Implementations include:
    - PyQtGraphBackend: High-performance real-time plotting
    - MatplotlibBackend: Publication-quality static plots (fallback)
    """

    @abstractmethod
    def initialize(self, layout: PlotLayoutConfig) -> None:
        """
        Initialize the plotting window/figure.

        Args:
            layout: Layout configuration for all plots
        """
        pass

    @abstractmethod
    def add_plot(self, config: PlotConfig) -> str:
        """
        Add a new plot to the layout.

        Args:
            config: Plot configuration

        Returns:
            Plot identifier string
        """
        pass

    @abstractmethod
    def remove_plot(self, plot_id: str) -> None:
        """
        Remove a plot from the layout.

        Args:
            plot_id: Plot identifier
        """
        pass

    @abstractmethod
    def update_data(
        self,
        plot_id: str,
        x_data: List[float],
        y_data: Dict[str, List[float]],
    ) -> None:
        """
        Update plot data.

        Args:
            plot_id: Plot identifier
            x_data: X-axis values
            y_data: Dictionary mapping series names to Y values
        """
        pass

    @abstractmethod
    def process_events(self) -> bool:
        """
        Process GUI events.

        Returns:
            False if window was closed, True otherwise
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources and close windows."""
        pass

    @abstractmethod
    def export_figure(self, path: str, dpi: int = 150) -> None:
        """
        Export current view to image file.

        Args:
            path: Output file path
            dpi: Image resolution
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available.

        Returns:
            True if backend dependencies are installed
        """
        pass


class DummyBackend(PlotBackend):
    """
    Dummy backend for headless operation.

    Does nothing but satisfies the interface.
    """

    def __init__(self):
        self._initialized = False

    def initialize(self, layout: PlotLayoutConfig) -> None:
        self._initialized = True

    def add_plot(self, config: PlotConfig) -> str:
        return config.name

    def remove_plot(self, plot_id: str) -> None:
        pass

    def update_data(
        self,
        plot_id: str,
        x_data: List[float],
        y_data: Dict[str, List[float]],
    ) -> None:
        pass

    def process_events(self) -> bool:
        return True

    def close(self) -> None:
        self._initialized = False

    def export_figure(self, path: str, dpi: int = 150) -> None:
        pass

    def is_available(self) -> bool:
        return True
