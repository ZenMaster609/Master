"""
Offline headless plotter for vehicle state data.

Reads Parquet log files and generates publication-quality plots
as PNG/PDF without requiring a display (uses matplotlib Agg backend).

Usage:
    python -m vehicle_plotter.plotting.offline_plotter /path/to/session
    python -m vehicle_plotter.plotting.offline_plotter /path/to/session --format pdf
    python -m vehicle_plotter.plotting.offline_plotter /path/to/session --dpi 300

This module can be:
1. Called from logger_node on shutdown (auto-plot feature)
2. Run as a standalone script for re-processing old data
"""

# Set headless backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from ..core.vehicle_state import VehicleState
from .plot_config import (
    AxisConfig,
    PlotConfig,
    PlotLayoutConfig,
    VELOCITY_AXIS_LIMITS_MPS,
    get_all_plots,
)
from .plot_manager import PlotManager
from .plot_definitions import (
    PlotDefinition,
    get_all_plot_definitions,
)


WHEEL_SPEED_MIN_MPS = 0.0  # Wheel linear speed is shown as a non-negative magnitude in exports.
WHEEL_SPEED_MAX_MPS = 8.0  # Matches the current simulated dashboard range for wheel speed inspection.
MILLIMETERS_PER_SECOND_TO_METERS_PER_SECOND_SCALE = (
    0.001  # Converts logged wheel speed to meters per second.
)
MOVEMENT_DASHBOARD_PLOT_INDICES = [0, 1, 2, 3]  # Selects movement plots from the live dashboard layout.
MECHANICAL_DASHBOARD_PLOT_INDICES = [4, 5, 6, 7]  # Selects mechanical plots from the live dashboard layout.
MOVEMENT_VELOCITY_PLOT_INDEX = 1  # Velocity is the top-right movement subplot in the 2x2 export layout.
MOVEMENT_WHEEL_SPEED_PLOT_INDEX = (
    3  # Wheel speed is the bottom-right movement subplot in the 2x2 export layout.
)


class OfflinePlotter:
    """
    Generates plots from logged vehicle state data.

    Reads Parquet files and produces static plots using matplotlib's
    Agg backend (no display required). Designed for both:
    - Automatic plot generation on logger shutdown
    - Manual re-processing of old data

    Attributes:
        session_path: Path to session directory containing logs/
        output_format: Image format ('png', 'pdf', 'svg')
        dpi: Resolution for raster formats
    """

    def __init__(
        self,
        session_path: Path,
        output_format: str = 'pdf',
        dpi: int = 150,
    ):
        """
        Initialize the offline plotter.

        Args:
            session_path: Path to session directory (e.g., multidata/sim_2026-01-12_14-00-00/)
            output_format: Image format ('png', 'pdf', 'svg')
            dpi: Resolution for raster formats (default 150)
        """
        self.session_path = Path(session_path)
        self.output_format = output_format
        self.dpi = dpi
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load all parquet files from session logs directory.

        Handles both new format (multidata/<prefix>_<timestamp>/logs/)
        and old format (multidata/<timestamp>/<os>/logs/).

        Returns:
            Concatenated DataFrame from all parquet files

        Raises:
            FileNotFoundError: If no parquet files found
        """
        # Try new format first: session_path/logs/
        logs_path = self.session_path / 'logs'

        if not logs_path.exists():
            # Try old format: session_path might already be the OS folder
            # Check if parquet files are directly in session_path
            if list(self.session_path.glob('vehicle_state_*.parquet')):
                logs_path = self.session_path
            else:
                # Try looking in linux/ or windows/ subfolders (old format)
                for os_folder in ['linux', 'windows', 'macos']:
                    candidate = self.session_path / os_folder / 'logs'
                    if candidate.exists():
                        logs_path = candidate
                        break

        parquet_files = sorted(logs_path.glob('vehicle_state_*.parquet'))
        csv_files = sorted(logs_path.glob('vehicle_state_*.csv'))

        if parquet_files:
            # Load and concatenate all parquet files
            dfs = []
            for f in parquet_files:
                df = pd.read_parquet(f)
                dfs.append(df)
        elif csv_files:
            # Fallback to CSV logs when parquet/pyarrow is unavailable
            dfs = []
            for f in csv_files:
                df = pd.read_csv(f)
                dfs.append(df)
        else:
            raise FileNotFoundError(
                f"No parquet or csv files found in {logs_path}"
            )

        self.data = pd.concat(dfs, ignore_index=True)

        # Load metadata if available
        metadata_path = logs_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        return self.data

    def detect_data_source(self) -> str:
        """
        Detect whether data is from simulation.

        Detection order:
        1. Check run_id prefix (sim_)
        2. Check source_adapter column in data
        3. Default to 'unknown'

        Returns:
            'simulation' or 'unknown'
        """
        run_id = self.session_path.name

        # Check prefix (new format)
        if run_id.startswith('sim_'):
            return 'simulation'
        # Check source_adapter column (works with old format)
        if self.data is not None and 'source_adapter' in self.data.columns:
            source = self.data['source_adapter'].iloc[0] if len(self.data) > 0 else None
            if source == 'gazebo':
                return 'simulation'

        return 'unknown'

    def get_source_label(self) -> str:
        """Get human-readable source label for plot titles."""
        source = self.detect_data_source()
        if source == 'simulation':
            return 'Simulation'
        return 'Unknown Source'

    def generate_plots(
        self,
        plot_defs: Optional[List[PlotDefinition]] = None,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        Generate all configured plots.

        Args:
            plot_defs: List of plot definitions (uses defaults if None)
            output_dir: Output directory (uses session_path/plots/ if None)

        Returns:
            List of paths to generated plot files
        """
        import matplotlib.pyplot as plt

        if self.data is None:
            self.load_data()

        if plot_defs is None:
            plot_defs = get_all_plot_definitions()

        if output_dir is None:
            output_dir = self.session_path / 'plots'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        source_label = self.get_source_label()
        generated_files = []

        for plot_def in plot_defs:
            # Check if required columns exist with valid data
            if not plot_def.has_required_data(self.data):
                print(f"Skipping {plot_def.name}: missing or empty required columns")
                continue

            try:
                fig, ax = plot_def.create_plot(self.data, source_label)

                filename = f"{plot_def.filename}.{self.output_format}"
                filepath = output_dir / filename

                fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)

                generated_files.append(filepath)
                print(f"Generated: {filepath}")

            except Exception as e:
                print(f"Error generating {plot_def.name}: {e}")
                continue

        try:
            dashboard_paths = self.generate_split_dashboards(output_dir)
            generated_files.extend(dashboard_paths)
            for dashboard_path in dashboard_paths:
                print(f"Generated: {dashboard_path}")
        except Exception as e:
            print(f"Error generating split dashboards: {e}")
        for stale_dashboard in output_dir.glob("all_plots.*"):
            stale_dashboard.unlink()
        for stale_position_ins in output_dir.glob("position_ins.*"):
            stale_position_ins.unlink()

        return generated_files

    def generate_split_dashboards(self, output_dir: Path) -> List[Path]:
        """Render the live sensor dashboard as movement/mechanical 2x2 PDFs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = [
            (
                output_dir / f"movement.{self.output_format}",
                self._build_movement_layout(),
            ),
            (
                output_dir / f"mechanical.{self.output_format}",
                self._build_mechanical_layout(),
            ),
        ]

        generated: List[Path] = []
        for output_path, layout in outputs:
            generated.append(self._export_dashboard(output_path, layout_config=layout))

            if self.output_format != 'png':
                legacy_png = output_path.with_suffix('.png')
                if legacy_png.exists():
                    legacy_png.unlink()

        return generated

    def generate_combined_dashboard(self, output_path: Path) -> Path:
        """Render a static dashboard layout from logged state data."""
        return self._export_dashboard(output_path, layout_config=get_all_plots())

    def _export_dashboard(
        self,
        output_path: Path,
        *,
        layout_config: PlotLayoutConfig,
    ) -> Path:
        """Export a specific dashboard layout to a static file."""
        if self.data is None:
            self.load_data()

        self._expand_dashboard_buffers(layout_config)
        plot_manager = PlotManager(
            layout_config=layout_config,
            enable_gui=False,
        )
        try:
            for row in self.data.to_dict(orient='records'):
                plot_manager.push_state(self._vehicle_state_from_row(row))
            plot_manager.export_static_dashboard(str(output_path), dpi=self.dpi)
        finally:
            plot_manager.close()

        return output_path

    def _build_movement_layout(self) -> PlotLayoutConfig:
        """Build the movement dashboard with wheel speed shown in m/s."""
        layout = self._build_dashboard_layout(
            MOVEMENT_DASHBOARD_PLOT_INDICES,
            window_title="Movement",
        )
        self._configure_velocity_components_plot(layout.plots[MOVEMENT_VELOCITY_PLOT_INDEX])
        wheel_speed_plot = layout.plots[MOVEMENT_WHEEL_SPEED_PLOT_INDEX]
        wheel_speed_plot.name = "Wheel Speed (m/s)"
        wheel_speed_plot.y_axis.label = "Wheel Speed"
        wheel_speed_plot.y_axis.unit = "m/s"
        wheel_speed_plot.y_axis.limits = (WHEEL_SPEED_MIN_MPS, WHEEL_SPEED_MAX_MPS)
        for series in wheel_speed_plot.series:
            series.scale = MILLIMETERS_PER_SECOND_TO_METERS_PER_SECOND_SCALE
        return layout

    def _build_mechanical_layout(self) -> PlotLayoutConfig:
        """Build the mechanical dashboard."""
        return self._build_dashboard_layout(
            MECHANICAL_DASHBOARD_PLOT_INDICES,
            window_title="Mechanical",
        )

    def _expand_dashboard_buffers(self, layout_config: PlotLayoutConfig) -> None:
        """Resize offline buffers so PDF exports include the full session."""
        row_count = len(self.data) if self.data is not None else 0
        for plot_config in layout_config.plots:
            plot_config.buffer_size = max(plot_config.buffer_size, row_count)

    def _configure_velocity_components_plot(self, plot_config: PlotConfig) -> None:
        """Show body-frame velocity components in the movement dashboard export."""
        plot_config.name = "Velocity"
        plot_config.series = [
            series for series in plot_config.series
            if series.variable in {"vx", "vy"}
        ]
        plot_config.y_axis = AxisConfig(
            label="Velocity",
            unit="m/s",
            limits=VELOCITY_AXIS_LIMITS_MPS,
            auto_scale=False,
        )

    def _build_dashboard_layout(
        self,
        plot_indices: List[int],
        *,
        window_title: str,
    ) -> PlotLayoutConfig:
        """Clone selected dashboard plots into a 2x2 export layout."""
        base_layout = get_all_plots()
        cloned_plots = []
        for position, plot_index in enumerate(plot_indices):
            plot_config = deepcopy(base_layout.plots[plot_index])
            plot_config.row = position // 2
            plot_config.col = position % 2
            plot_config.row_span = 1
            plot_config.col_span = 1
            cloned_plots.append(plot_config)

        return PlotLayoutConfig(
            plots=cloned_plots,
            rows=2,
            cols=2,
            window_title=window_title,
            window_size=(1650, 900),
            dark_mode=base_layout.dark_mode,
            update_rate_hz=base_layout.update_rate_hz,
        )

    def _vehicle_state_from_row(self, row: Dict[str, Any]) -> VehicleState:
        """Map a logged row back into VehicleState for dashboard export."""
        return VehicleState(
            timestamp=self._float_value(row, 'timestamp'),
            x=self._float_value(row, 'x'),
            y=self._float_value(row, 'y'),
            vx=self._float_value(row, 'vx'),
            vy=self._float_value(row, 'vy'),
            yaw=self._float_value(row, 'yaw'),
            yaw_rate=self._float_value(row, 'yaw_rate'),
            raw_x=self._float_value(row, 'raw_x'),
            raw_y=self._float_value(row, 'raw_y'),
            raw_vx=self._float_value(row, 'raw_vx'),
            raw_vy=self._float_value(row, 'raw_vy'),
            raw_yaw=self._float_value(row, 'raw_yaw'),
            raw_speed=self._float_value(row, 'raw_speed'),
            imu_vx=self._float_value(row, 'imu_vx'),
            imu_vy=self._float_value(row, 'imu_vy'),
            imu_yaw=self._float_value(row, 'imu_yaw'),
            speed=self._float_value(row, 'speed'),
            distance_traveled=self._float_value(row, 'distance_traveled'),
            slip_longitudinal=self._float_value(row, 'slip_longitudinal'),
            slip_lateral=self._float_value(row, 'slip_lateral'),
            encoder_velocities=[
                self._float_value(row, 'encoder_fl'),
                self._float_value(row, 'encoder_fr'),
                self._float_value(row, 'encoder_rl'),
                self._float_value(row, 'encoder_rr'),
            ],
            # Legacy sessions may only have wheel RPM columns; fall back so the
            # combined dashboard still shows the wheel-speed panel.
            encoder_speeds_mm_s=[
                self._float_value(row, 'encoder_speed_fl', 'vel_fl', 'encoder_fl'),
                self._float_value(row, 'encoder_speed_fr', 'vel_fr', 'encoder_fr'),
                self._float_value(row, 'encoder_speed_rl', 'vel_rl', 'encoder_rl'),
                self._float_value(row, 'encoder_speed_rr', 'vel_rr', 'encoder_rr'),
            ],
            gps_latitude=self._float_value(row, 'gps_latitude'),
            gps_longitude=self._float_value(row, 'gps_longitude'),
            gps_altitude=self._float_value(row, 'gps_altitude'),
            gps_valid=self._bool_value(row, 'gps_valid'),
            gps_local_x=self._float_value(row, 'gps_local_x'),
            gps_local_y=self._float_value(row, 'gps_local_y'),
            ins_x=self._float_value(row, 'ins_x'),
            ins_y=self._float_value(row, 'ins_y'),
            dr_x=self._float_value(row, 'dr_x'),
            dr_y=self._float_value(row, 'dr_y'),
            suspension=[
                self._float_value(row, 'suspension_fl'),
                self._float_value(row, 'suspension_fr'),
                self._float_value(row, 'suspension_rl'),
                self._float_value(row, 'suspension_rr'),
            ],
            steering_angle=self._float_value(row, 'steering_angle'),
            steering_valid=self._bool_value(row, 'steering_valid'),
            water_pressure=self._float_value(row, 'water_pressure'),
            water_flow=self._float_value(row, 'water_flow'),
            water_temp_in=self._float_value(row, 'water_temp_in'),
            water_temp_out=self._float_value(row, 'water_temp_out'),
            water_temp_radiator=self._float_value(row, 'water_temp_radiator'),
            brake_temp_fr=self._float_value(row, 'brake_temp_fr'),
            brake_temp_rl=self._float_value(row, 'brake_temp_rl'),
            pitot_dynamic_pressure=self._float_value(row, 'pitot_dynamic_pressure'),
            estimation_status=self._string_value(row, 'estimation_status', default='raw'),
            source_adapter=self._string_value(row, 'source_adapter', default='unknown'),
        )

    @staticmethod
    def _float_value(row: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
        for key in keys:
            value = row.get(key)
            if value is None or pd.isna(value):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return default

    @staticmethod
    def _bool_value(row: Dict[str, Any], key: str, default: bool = False) -> bool:
        value = row.get(key)
        if value is None or pd.isna(value):
            return default
        return bool(value)

    @staticmethod
    def _string_value(row: Dict[str, Any], key: str, default: str = '') -> str:
        value = row.get(key)
        if value is None:
            return default
        return str(value)

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the loaded data."""
        if self.data is None:
            self.load_data()

        summary = {
            'total_records': len(self.data),
            'columns': list(self.data.columns),
            'source': self.detect_data_source(),
        }

        # Add time range if timestamp available
        if 'timestamp' in self.data.columns and len(self.data) > 0:
            summary['duration_sec'] = self.data['timestamp'].iloc[-1] - self.data['timestamp'].iloc[0]

        # Check which plot types have data
        summary['available_plots'] = []
        for plot_def in get_all_plot_definitions():
            if plot_def.has_required_data(self.data):
                summary['available_plots'].append(plot_def.name)

        return summary


def main():
    """CLI entry point for offline plotter."""
    parser = argparse.ArgumentParser(
        description='Generate plots from vehicle state logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s ./multidata/sim_2026-01-12_14-00-00/
    %(prog)s ./multidata/sim_2026-01-12_14-00-00/ --format pdf
    %(prog)s ./multidata/2026-01-12_14-00-00/linux/ --dpi 300
        """
    )
    parser.add_argument('session_path', help='Path to session directory')
    parser.add_argument('--format', '-f', choices=['png', 'pdf', 'svg'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('--dpi', '-d', type=int, default=150,
                        help='DPI for raster formats (default: 150)')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots')
    parser.add_argument('--summary', '-s', action='store_true',
                        help='Print data summary and exit')

    args = parser.parse_args()

    plotter = OfflinePlotter(
        session_path=Path(args.session_path),
        output_format=args.format,
        dpi=args.dpi,
    )

    if args.summary:
        summary = plotter.get_data_summary()
        print("\nData Summary:")
        print(f"  Records: {summary['total_records']}")
        print(f"  Source: {summary['source']}")
        if 'duration_sec' in summary:
            print(f"  Duration: {summary['duration_sec']:.1f} seconds")
        print(f"  Available plots: {', '.join(summary['available_plots'])}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    generated = plotter.generate_plots(output_dir=output_dir)

    print(f"\nGenerated {len(generated)} plots")


if __name__ == '__main__':
    main()
