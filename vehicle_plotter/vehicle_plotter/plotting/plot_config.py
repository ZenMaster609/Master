"""
Plot configuration dataclasses.

Provides a highly modular, declarative way to configure plots.
Easy to add/remove plots and change X/Y variables.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Literal
from enum import Enum


class XAxisType(Enum):
    """Supported X-axis types for plots."""
    TIME = "time"                    # Elapsed time in seconds
    DISTANCE = "distance"            # Cumulative distance traveled
    YAW = "yaw"                      # Yaw angle (radians)
    ENCODER_TOTAL = "encoder_total"  # Sum of all encoder RPM values
    ENCODER_FL = "encoder_fl"        # Front-left encoder RPM
    CUSTOM = "custom"                # User-defined via x_axis


@dataclass
class AxisConfig:
    """Configuration for a single axis."""
    label: str                              # Axis label for display
    variable: str = ""                      # Attribute name on VehicleState (optional for y_axis)
    unit: str = ""                          # Unit string (e.g., "m", "rad/s")
    scale: float = 1.0                      # Multiply value by this factor
    offset: float = 0.0                     # Add this after scaling
    transform: Optional[Callable[[float], float]] = None  # Custom transform
    limits: Optional[tuple] = None          # Fixed axis limits (min, max)
    auto_scale: bool = True                 # Auto-scale axis to data


@dataclass
class SeriesConfig:
    """Configuration for a single data series in a plot."""
    name: str                               # Legend name
    variable: str                           # Attribute name on VehicleState
    color: str = "auto"                     # Color (hex, name, or "auto")
    line_width: float = 1.5
    line_style: Literal["solid", "dashed", "dotted"] = "solid"
    marker: Optional[str] = None            # Marker style (None, "o", "x", etc.)
    scale: float = 1.0
    offset: float = 0.0
    transform: Optional[Callable[[float], float]] = None


@dataclass
class PlotConfig:
    """
    Complete configuration for a single plot.

    Examples:
        # Position plot (x vs y)
        PlotConfig(
            name="Position (X vs Y)",
            x_axis=AxisConfig(label="X Position", variable="x", unit="m"),
            series=[
                SeriesConfig(name="Y Position", variable="y")
            ],
            plot_type="xy",
        )

        # Time series plot (velocity vs time)
        PlotConfig(
            name="Velocity",
            x_axis_type=XAxisType.TIME,
            series=[
                SeriesConfig(name="Vx", variable="vx", color="blue"),
                SeriesConfig(name="Vy", variable="vy", color="red"),
            ],
            plot_type="timeseries",
        )
    """
    name: str                               # Plot title/identifier
    series: List[SeriesConfig]              # Data series to plot

    # X-axis configuration (mutually exclusive options)
    x_axis_type: XAxisType = XAxisType.TIME
    x_axis: Optional[AxisConfig] = None     # Custom X-axis (for XY plots)

    # Y-axis configuration
    y_axis: Optional[AxisConfig] = None     # Y-axis settings (limits, label)

    # Plot type
    plot_type: Literal["timeseries", "xy", "histogram"] = "timeseries"

    # Display options
    buffer_size: int = 1000                 # Max points to display
    update_rate_hz: float = 30.0            # Plot update rate
    show_legend: bool = True
    show_grid: bool = True

    # Layout hints
    row: int = 0                            # Grid position (row)
    col: int = 0                            # Grid position (column)
    row_span: int = 1
    col_span: int = 1


@dataclass
class PlotLayoutConfig:
    """Configuration for the overall plot layout."""
    plots: List[PlotConfig] = field(default_factory=list)
    rows: int = 2
    cols: int = 2
    window_title: str = "Vehicle Plotter"
    window_size: tuple = (1200, 800)
    dark_mode: bool = True
    update_rate_hz: float = 30.0            # Global update rate


# Pre-defined plot configurations
def get_default_plots() -> PlotLayoutConfig:
    """Get the default plot layout configuration."""
    import math

    return PlotLayoutConfig(
        rows=3,
        cols=2,
        window_size=(1200, 1000),
        plots=[
            # Top-left: Position trajectory
            PlotConfig(
                name="Position Trajectory",
                plot_type="xy",
                x_axis=AxisConfig(label="X", variable="x", unit="m"),
                series=[SeriesConfig(name="Y", variable="y", color="#1f77b4")],
                y_axis=AxisConfig(label="Y", variable="y", unit="m"),
                row=0, col=0,
                show_grid=True,
                buffer_size=2000,
            ),

            # Top-right: Velocity vs time
            PlotConfig(
                name="Velocity",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="Vx", variable="vx", color="#1f77b4"),
                    SeriesConfig(name="Vy", variable="vy", color="#ff7f0e"),
                    SeriesConfig(name="Speed", variable="speed", color="#2ca02c"),
                ],
                y_axis=AxisConfig(label="Velocity", unit="m/s"),
                row=0, col=1,
                buffer_size=1000,
            ),

            # Bottom-left: Yaw angle vs time
            PlotConfig(
                name="Heading",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(
                        name="Yaw",
                        variable="yaw",
                        color="#d62728",
                        transform=lambda x: x * 180 / math.pi,  # rad to deg
                    ),
                ],
                y_axis=AxisConfig(label="Yaw", unit="deg"),
                row=1, col=0,
                buffer_size=1000,
            ),

            # Bottom-right: Encoder RPM vs time
            PlotConfig(
                name="Encoder RPM",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="FL", variable="encoder_velocities[0]", color="#2ca02c"),
                    SeriesConfig(name="FR", variable="encoder_velocities[1]", color="#d62728"),
                    SeriesConfig(name="RL", variable="encoder_velocities[2]", color="#9467bd"),
                    SeriesConfig(name="RR", variable="encoder_velocities[3]", color="#8c564b"),
                ],
                y_axis=AxisConfig(label="Wheel RPM", unit="RPM"),
                row=1, col=1,
                buffer_size=1000,
            ),

            # Bottom row: Suspension travel vs time
            PlotConfig(
                name="Suspension",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="FL", variable="suspension[0]", color="#2ca02c"),
                    SeriesConfig(name="FR", variable="suspension[1]", color="#d62728"),
                    SeriesConfig(name="RL", variable="suspension[2]", color="#9467bd"),
                    SeriesConfig(name="RR", variable="suspension[3]", color="#8c564b"),
                ],
                y_axis=AxisConfig(label="Suspension", unit="m"),
                row=2, col=0,
                buffer_size=1000,
            ),

            PlotConfig(
                name="Encoder Angle Accum",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="FL", variable="encoder_angle_accum[0]", color="#2ca02c"),
                    SeriesConfig(name="FR", variable="encoder_angle_accum[1]", color="#d62728"),
                    SeriesConfig(name="RL", variable="encoder_angle_accum[2]", color="#9467bd"),
                    SeriesConfig(name="RR", variable="encoder_angle_accum[3]", color="#8c564b"),
                ],
                y_axis=AxisConfig(label="Angle Accum", unit="rad"),
                row=2, col=1,
                buffer_size=1000,
            ),
        ],
    )


def get_virtual_sensor_plots() -> PlotLayoutConfig:
    """Get a plot layout for virtual sensor channels."""
    return PlotLayoutConfig(
        rows=2,
        cols=2,
        window_title="Virtual Sensors",
        window_size=(1200, 900),
        plots=[
            PlotConfig(
                name="Cooling Pressure/Flow",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="Pressure", variable="water_pressure", color="#1f77b4"),
                    SeriesConfig(name="Flow", variable="water_flow", color="#2ca02c"),
                ],
                y_axis=AxisConfig(label="Cooling", unit="bar / L/min"),
                row=0, col=0,
                buffer_size=1000,
            ),
            PlotConfig(
                name="Cooling Temperatures",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="In", variable="water_temp_in", color="#ff7f0e"),
                    SeriesConfig(name="Out", variable="water_temp_out", color="#d62728"),
                    SeriesConfig(name="Radiator", variable="water_temp_radiator", color="#9467bd"),
                ],
                y_axis=AxisConfig(label="Temperature", unit="C"),
                row=0, col=1,
                buffer_size=1000,
            ),
            PlotConfig(
                name="Brake Temps",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="Front Right", variable="brake_temp_fr", color="#d62728"),
                    SeriesConfig(name="Rear Left", variable="brake_temp_rl", color="#8c564b"),
                ],
                y_axis=AxisConfig(label="Temperature", unit="C"),
                row=1, col=0,
                buffer_size=1000,
            ),
            PlotConfig(
                name="Pitot Dynamic Pressure",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="Pitot", variable="pitot_dynamic_pressure", color="#17becf"),
                ],
                y_axis=AxisConfig(label="Pressure", unit="Pa"),
                row=1, col=1,
                buffer_size=1000,
            ),
        ],
    )


def get_slip_plots() -> PlotLayoutConfig:
    """Get a plot layout focused on slip analysis."""
    return PlotLayoutConfig(
        rows=2,
        cols=2,
        window_title="Slip Analysis",
        plots=[
            PlotConfig(
                name="Longitudinal Slip",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="Slip Long", variable="slip_longitudinal", color="#1f77b4"),
                ],
                y_axis=AxisConfig(label="Slip Ratio", limits=(-1.0, 1.0)),
                row=0, col=0,
            ),
            PlotConfig(
                name="Lateral Slip",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="Slip Lat", variable="slip_lateral", color="#ff7f0e"),
                ],
                y_axis=AxisConfig(label="Slip Angle (rad)"),
                row=0, col=1,
            ),
            PlotConfig(
                name="Wheel RPM",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="FL", variable="encoder_velocities[0]", color="#2ca02c"),
                    SeriesConfig(name="FR", variable="encoder_velocities[1]", color="#d62728"),
                    SeriesConfig(name="RL", variable="encoder_velocities[2]", color="#9467bd"),
                    SeriesConfig(name="RR", variable="encoder_velocities[3]", color="#8c564b"),
                ],
                y_axis=AxisConfig(label="Wheel RPM"),
                row=1, col=0,
            ),
            PlotConfig(
                name="Speed Comparison",
                plot_type="timeseries",
                x_axis_type=XAxisType.TIME,
                series=[
                    SeriesConfig(name="Body Speed", variable="speed", color="#1f77b4"),
                    SeriesConfig(name="Avg Wheel", variable="avg_wheel_speed", color="#ff7f0e"),
                ],
                y_axis=AxisConfig(label="Speed (m/s)"),
                row=1, col=1,
            ),
        ],
    )
