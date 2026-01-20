"""Plotting subsystem for vehicle_plotter."""

from .plot_config import (
    PlotConfig,
    PlotLayoutConfig,
    SeriesConfig,
    AxisConfig,
    XAxisType,
    get_default_plots,
    get_virtual_sensor_plots,
)
from .plot_manager import PlotManager
from .plot_definitions import (
    PlotDefinition,
    XYPlotDefinition,
    TimeSeriesPlotDefinition,
    MultiPanelPlotDefinition,
    get_all_plot_definitions,
)
# Note: OfflinePlotter not imported here to avoid pandas dependency at load time
# Import directly from .offline_plotter when needed

__all__ = [
    'PlotConfig',
    'PlotLayoutConfig',
    'SeriesConfig',
    'AxisConfig',
    'XAxisType',
    'PlotManager',
    'PlotDefinition',
    'XYPlotDefinition',
    'TimeSeriesPlotDefinition',
    'MultiPanelPlotDefinition',
    'get_all_plot_definitions',
    'get_default_plots',
    'get_virtual_sensor_plots',
]
