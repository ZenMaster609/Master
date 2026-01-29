"""
Plot definitions for offline plotter.

Each PlotDefinition encapsulates the logic to create a specific plot type.
Adding new plots is as simple as creating a new PlotDefinition instance
and adding it to get_all_plot_definitions().

Plot Types:
- XYPlotDefinition: X-Y scatter/line plots (position trajectories)
- TimeSeriesPlotDefinition: Single-panel time series
- MultiPanelPlotDefinition: Multi-panel grid (wheel RPM, suspension)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

# Defer matplotlib import to avoid issues with headless mode
# Import happens inside create_plot methods


@dataclass
class PlotDefinition(ABC):
    """Base class for plot definitions."""

    name: str
    filename: str
    title_template: str  # e.g., "Position Trajectory ({source})"
    required_columns: List[str]

    def has_required_data(self, df) -> bool:
        """
        Check if dataframe has required columns with non-null data.

        Args:
            df: Pandas DataFrame with logged data

        Returns:
            True if all required columns exist and have at least some non-null values
        """
        for col in self.required_columns:
            if col not in df.columns:
                return False
            # Check if column has any non-zero, non-null values
            if df[col].isna().all():
                return False
            # For numeric columns, also check if all values are zero
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if (df[col] == 0).all():
                    return False
        return True

    @abstractmethod
    def create_plot(self, df, source_label: str) -> Tuple:
        """
        Create the plot and return figure and axes.

        Args:
            df: Pandas DataFrame with logged data
            source_label: 'Simulation' or 'Real Vehicle' for title

        Returns:
            Tuple of (figure, axes)
        """
        pass


@dataclass
class XYPlotDefinition(PlotDefinition):
    """X-Y scatter/line plot (e.g., position trajectory)."""

    x_column: str = ""
    y_column: str = ""
    x_label: str = "X"
    y_label: str = "Y"
    color: str = "#1f77b4"
    equal_aspect: bool = True
    line_width: float = 1.5

    def create_plot(self, df, source_label: str) -> Tuple:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        x_data = df[self.x_column].values
        y_data = df[self.y_column].values

        ax.plot(x_data, y_data, color=self.color, linewidth=self.line_width)

        # Mark start and end points
        if len(x_data) > 0:
            ax.scatter([x_data[0]], [y_data[0]], color='green', s=100, zorder=5, label='Start')
            ax.scatter([x_data[-1]], [y_data[-1]], color='red', s=100, zorder=5, label='End')
            ax.legend()

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(self.title_template.format(source=source_label))
        ax.grid(True, alpha=0.3)

        if self.equal_aspect:
            ax.set_aspect('equal')

        fig.tight_layout()
        return fig, ax


@dataclass
class TimeSeriesPlotDefinition(PlotDefinition):
    """Time series plot with multiple series."""

    series: List[Tuple[str, str, str]] = field(default_factory=list)  # [(column, label, color), ...]
    y_label: str = "Value"
    time_column: str = "timestamp"

    def create_plot(self, df, source_label: str) -> Tuple:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Convert timestamp to relative time
        t = df[self.time_column].values
        if len(t) > 0:
            t = t - t[0]  # Relative time from start

        for col, label, color in self.series:
            if col in df.columns and not df[col].isna().all():
                ax.plot(t, df[col].values, label=label, color=color, linewidth=1.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(self.y_label)
        ax.set_title(self.title_template.format(source=source_label))
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig, ax


@dataclass
class MultiPanelPlotDefinition(PlotDefinition):
    """Multiple subplots (e.g., 4 wheel RPM traces)."""

    subplots: List[Tuple[str, str, str]] = field(default_factory=list)  # [(column, title, color), ...]
    y_label: str = "Value"
    time_column: str = "timestamp"
    ncols: int = 2
    share_y: bool = True

    def create_plot(self, df, source_label: str) -> Tuple:
        import matplotlib.pyplot as plt

        n = len(self.subplots)
        nrows = (n + self.ncols - 1) // self.ncols

        fig, axes = plt.subplots(nrows, self.ncols, figsize=(12, 4 * nrows),
                                  sharey=self.share_y)
        axes = np.atleast_1d(axes).flatten()

        # Convert timestamp to relative time
        t = df[self.time_column].values
        if len(t) > 0:
            t = t - t[0]

        for i, (col, title, color) in enumerate(self.subplots):
            ax = axes[i]
            if col in df.columns and not df[col].isna().all():
                ax.plot(t, df[col].values, color=color, linewidth=1.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(self.y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(self.title_template.format(source=source_label), fontsize=14, y=1.02)
        fig.tight_layout()

        return fig, axes[0]


def get_all_plot_definitions() -> List[PlotDefinition]:
    """
    Return all available plot definitions.

    To add a new plot, simply add a new PlotDefinition instance to this list.
    """
    return [
        # Position from INS (fused solution)
        XYPlotDefinition(
            name="INS Position",
            filename="position_ins",
            title_template="INS Position Trajectory ({source})",
            required_columns=['ins_x', 'ins_y'],
            x_column='ins_x',
            y_column='ins_y',
            x_label='X (m)',
            y_label='Y (m)',
            color='#1f77b4',  # Blue
        ),

        # Position from Dead Reckoning (IMU integration only)
        XYPlotDefinition(
            name="Dead Reckoning Position",
            filename="position_dr",
            title_template="IMU Dead Reckoning Position ({source})",
            required_columns=['dr_x', 'dr_y'],
            x_column='dr_x',
            y_column='dr_y',
            x_label='X (m)',
            y_label='Y (m)',
            color='#ff7f0e',  # Orange
        ),

        # Position from GPS (converted to local coords)
        XYPlotDefinition(
            name="GPS Position (Local)",
            filename="position_gps",
            title_template="GPS Position (Local Coords) ({source})",
            required_columns=['gps_local_x', 'gps_local_y'],
            x_column='gps_local_x',
            y_column='gps_local_y',
            x_label='X (m)',
            y_label='Y (m)',
            color='#2ca02c',  # Green
        ),

        # Legacy position (x, y) - works with all adapters
        XYPlotDefinition(
            name="Position Trajectory",
            filename="position_xy",
            title_template="Position Trajectory ({source})",
            required_columns=['x', 'y'],
            x_column='x',
            y_column='y',
            x_label='X (m)',
            y_label='Y (m)',
            color='#9467bd',  # Purple
        ),

        # Encoder RPM (matches live plot naming)
        MultiPanelPlotDefinition(
            name="Encoder RPM",
            filename="encoder_rpm",
            title_template="Encoder RPM ({source})",
            required_columns=['encoder_fl', 'encoder_fr', 'encoder_rl', 'encoder_rr'],
            subplots=[
                ('encoder_fl', 'Front Left', '#2ca02c'),
                ('encoder_fr', 'Front Right', '#d62728'),
                ('encoder_rl', 'Rear Left', '#9467bd'),
                ('encoder_rr', 'Rear Right', '#8c564b'),
            ],
            y_label='RPM',
        ),

        # Suspension (CAN data)
        MultiPanelPlotDefinition(
            name="Suspension",
            filename="suspension",
            title_template="Suspension Displacement ({source})",
            required_columns=['suspension_fl', 'suspension_fr', 'suspension_rl', 'suspension_rr'],
            subplots=[
                ('suspension_fl', 'Front Left', '#2ca02c'),
                ('suspension_fr', 'Front Right', '#d62728'),
                ('suspension_rl', 'Rear Left', '#9467bd'),
                ('suspension_rr', 'Rear Right', '#8c564b'),
            ],
            y_label='Displacement (m)',
        ),

        # Velocity time series
        TimeSeriesPlotDefinition(
            name="Velocity",
            filename="velocity",
            title_template="Velocity ({source})",
            required_columns=['vx', 'vy', 'speed'],
            series=[
                ('vx', 'Vx (forward)', '#1f77b4'),
                ('vy', 'Vy (lateral)', '#ff7f0e'),
                ('speed', 'Speed', '#2ca02c'),
            ],
            y_label='Velocity (m/s)',
        ),

        # Heading (yaw) time series
        TimeSeriesPlotDefinition(
            name="Heading",
            filename="heading",
            title_template="Heading ({source})",
            required_columns=['yaw'],
            series=[
                ('yaw', 'Yaw', '#1f77b4'),
            ],
            y_label='Yaw (rad)',
        ),

        # Virtual sensors: cooling pressure/flow
        TimeSeriesPlotDefinition(
            name="Cooling Pressure/Flow",
            filename="cooling_pressure_flow",
            title_template="Cooling Pressure/Flow ({source})",
            required_columns=['water_pressure', 'water_flow'],
            series=[
                ('water_pressure', 'Pressure', '#1f77b4'),
                ('water_flow', 'Flow', '#2ca02c'),
            ],
            y_label='Cooling (bar / L/min)',
        ),

        # Virtual sensors: cooling temperatures
        TimeSeriesPlotDefinition(
            name="Cooling Temperatures",
            filename="cooling_temperatures",
            title_template="Cooling Temperatures ({source})",
            required_columns=['water_temp_in', 'water_temp_out', 'water_temp_radiator'],
            series=[
                ('water_temp_in', 'In', '#ff7f0e'),
                ('water_temp_out', 'Out', '#d62728'),
                ('water_temp_radiator', 'Radiator', '#9467bd'),
            ],
            y_label='Temperature (C)',
        ),

        # Virtual sensors: brake temps
        TimeSeriesPlotDefinition(
            name="Brake Temps",
            filename="brake_temps",
            title_template="Brake Temperatures ({source})",
            required_columns=['brake_temp_fr', 'brake_temp_rl'],
            series=[
                ('brake_temp_fr', 'Front Right', '#d62728'),
                ('brake_temp_rl', 'Rear Left', '#8c564b'),
            ],
            y_label='Temperature (C)',
        ),

        # Virtual sensors: pitot dynamic pressure
        TimeSeriesPlotDefinition(
            name="Pitot Dynamic Pressure",
            filename="pitot_dynamic_pressure",
            title_template="Pitot Dynamic Pressure ({source})",
            required_columns=['pitot_dynamic_pressure'],
            series=[
                ('pitot_dynamic_pressure', 'Pitot', '#17becf'),
            ],
            y_label='Pressure (Pa)',
        ),
    ]


def get_plot_definition_by_name(name: str) -> Optional[PlotDefinition]:
    """Get a specific plot definition by name."""
    for plot_def in get_all_plot_definitions():
        if plot_def.name == name:
            return plot_def
    return None
