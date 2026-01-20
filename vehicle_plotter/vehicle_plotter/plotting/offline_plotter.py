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
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from .plot_definitions import PlotDefinition, get_all_plot_definitions


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
        output_format: str = 'png',
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
        Detect whether data is from simulation or real vehicle.

        Detection order:
        1. Check run_id prefix (sim_ or jetson_)
        2. Check source_adapter column in data
        3. Default to 'unknown'

        Returns:
            'simulation', 'real_vehicle', or 'unknown'
        """
        run_id = self.session_path.name

        # Check prefix (new format)
        if run_id.startswith('sim_'):
            return 'simulation'
        elif run_id.startswith('jetson_'):
            return 'real_vehicle'

        # Check source_adapter column (works with old format)
        if self.data is not None and 'source_adapter' in self.data.columns:
            source = self.data['source_adapter'].iloc[0] if len(self.data) > 0 else None
            if source == 'gazebo':
                return 'simulation'
            elif source in ('can', 'vectornav'):
                return 'real_vehicle'

        return 'unknown'

    def get_source_label(self) -> str:
        """Get human-readable source label for plot titles."""
        source = self.detect_data_source()
        if source == 'simulation':
            return 'Simulation'
        elif source == 'real_vehicle':
            return 'Real Vehicle'
        else:
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

        return generated_files

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
    %(prog)s ./multidata/jetson_2026-01-12_14-00-00/ --format pdf
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
