#!/usr/bin/env python3
"""
Standalone Parquet plotting script (no ROS required).

Reads vehicle state logs saved by vehicle_plotter and generates plots.
Works on Windows, Linux, and Mac without any ROS installation.

Usage:
    python plot_parquet.py /path/to/session_directory
    python plot_parquet.py /path/to/file.parquet
    python plot_parquet.py /path/to/session --output report.png
    python plot_parquet.py /path/to/session --interactive

Examples:
    # Plot from a session directory (loads all parquet files)
    python plot_parquet.py ~/.ros/vehicle_logs/session_20260109_141233

    # Plot a single parquet file
    python plot_parquet.py vehicle_state_0000.parquet

    # Save to PNG instead of showing
    python plot_parquet.py ./session --output vehicle_report.png

    # Interactive mode (show plot window)
    python plot_parquet.py ./session --interactive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def load_parquet_file(file_path: Path) -> pd.DataFrame:
    """Load a single parquet file into a DataFrame."""
    table = pq.read_table(file_path)
    return table.to_pandas()


def load_session(session_path: Path) -> pd.DataFrame:
    """
    Load all parquet files from a session directory.

    Args:
        session_path: Path to session directory or single parquet file

    Returns:
        Combined DataFrame with all data, sorted by timestamp
    """
    session_path = Path(session_path)

    if session_path.is_file():
        # Single file
        if session_path.suffix == '.parquet':
            return load_parquet_file(session_path)
        else:
            raise ValueError(f"Unsupported file type: {session_path.suffix}")

    if session_path.is_dir():
        # Session directory - find all parquet files
        parquet_files = sorted(session_path.glob("*.parquet"))

        if not parquet_files:
            # Try looking in logs subdirectory (multidata structure)
            parquet_files = sorted(session_path.glob("logs/*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {session_path}")

        # Load and concatenate all files
        dfs = []
        for pf in parquet_files:
            df = load_parquet_file(pf)
            dfs.append(df)
            print(f"  Loaded {pf.name}: {len(df)} records")

        combined = pd.concat(dfs, ignore_index=True)

        # Sort by timestamp
        if 'timestamp' in combined.columns:
            combined = combined.sort_values('timestamp').reset_index(drop=True)

        return combined

    raise FileNotFoundError(f"Path not found: {session_path}")


def load_metadata(session_path: Path) -> Optional[Dict[str, Any]]:
    """Load session metadata if available."""
    session_path = Path(session_path)

    if session_path.is_file():
        session_path = session_path.parent

    metadata_file = session_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None


def plot_trajectory(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot XY trajectory."""
    if 'x' not in df.columns or 'y' not in df.columns:
        ax.text(0.5, 0.5, 'No position data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Trajectory')
        return

    # Color by time for visual progression
    if 'timestamp' in df.columns:
        t = df['timestamp'] - df['timestamp'].iloc[0]
        scatter = ax.scatter(df['x'], df['y'], c=t, cmap='viridis', s=1, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Time (s)')
    else:
        ax.plot(df['x'], df['y'], 'b-', linewidth=0.5, alpha=0.7)

    # Mark start and end
    ax.plot(df['x'].iloc[0], df['y'].iloc[0], 'go', markersize=10, label='Start')
    ax.plot(df['x'].iloc[-1], df['y'].iloc[-1], 'ro', markersize=10, label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory')
    ax.axis('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_velocity(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot velocity components vs time."""
    if 'timestamp' not in df.columns:
        ax.text(0.5, 0.5, 'No timestamp data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Velocity')
        return

    t = df['timestamp'] - df['timestamp'].iloc[0]

    plotted = False
    if 'vx' in df.columns:
        ax.plot(t, df['vx'], 'b-', linewidth=0.8, label='Vx (forward)', alpha=0.8)
        plotted = True
    if 'vy' in df.columns:
        ax.plot(t, df['vy'], 'g-', linewidth=0.8, label='Vy (lateral)', alpha=0.8)
        plotted = True
    if 'speed' in df.columns:
        ax.plot(t, df['speed'], 'r-', linewidth=1.2, label='Speed', alpha=0.9)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, 'No velocity data', ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_heading(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot heading (yaw) vs time."""
    if 'timestamp' not in df.columns or 'yaw' not in df.columns:
        ax.text(0.5, 0.5, 'No heading data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Heading')
        return

    t = df['timestamp'] - df['timestamp'].iloc[0]
    yaw_deg = np.degrees(df['yaw'])

    ax.plot(t, yaw_deg, 'purple', linewidth=0.8, alpha=0.8)

    if 'yaw_rate' in df.columns:
        ax2 = ax.twinx()
        yaw_rate_deg = np.degrees(df['yaw_rate'])
        ax2.plot(t, yaw_rate_deg, 'orange', linewidth=0.6, alpha=0.6, label='Yaw rate')
        ax2.set_ylabel('Yaw Rate (deg/s)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw (deg)', color='purple')
    ax.tick_params(axis='y', labelcolor='purple')
    ax.set_title('Heading')
    ax.grid(True, alpha=0.3)


def plot_encoders(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot wheel encoder data vs distance or time."""
    # Check for encoder columns
    encoder_cols = ['encoder_fl', 'encoder_fr', 'encoder_rl', 'encoder_rr']
    velocity_cols = ['vel_fl', 'vel_fr', 'vel_rl', 'vel_rr']

    has_encoders = all(col in df.columns for col in encoder_cols)
    has_velocities = all(col in df.columns for col in velocity_cols)

    if not has_encoders and not has_velocities:
        ax.text(0.5, 0.5, 'No encoder data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Wheel Encoders')
        return

    # Use distance as x-axis if available, otherwise time
    if 'distance_traveled' in df.columns and df['distance_traveled'].max() > 0:
        x = df['distance_traveled']
        x_label = 'Distance (m)'
    elif 'timestamp' in df.columns:
        x = df['timestamp'] - df['timestamp'].iloc[0]
        x_label = 'Time (s)'
    else:
        x = np.arange(len(df))
        x_label = 'Sample'

    colors = ['blue', 'green', 'red', 'orange']
    labels = ['FL', 'FR', 'RL', 'RR']

    # Plot velocities if available (more useful than raw ticks)
    if has_velocities:
        for col, color, label in zip(velocity_cols, colors, labels):
            ax.plot(x, df[col], color=color, linewidth=0.6, alpha=0.7, label=label)
        ax.set_ylabel('Wheel Velocity (m/s)')
    else:
        for col, color, label in zip(encoder_cols, colors, labels):
            ax.plot(x, df[col], color=color, linewidth=0.6, alpha=0.7, label=label)
        ax.set_ylabel('Encoder Ticks')

    ax.set_xlabel(x_label)
    ax.set_title('Wheel Encoders')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)


def plot_gps(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot GPS coordinates if available."""
    if 'gps_latitude' not in df.columns or 'gps_longitude' not in df.columns:
        ax.text(0.5, 0.5, 'No GPS data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GPS')
        return

    # Filter valid GPS points
    if 'gps_valid' in df.columns:
        valid = df[df['gps_valid'] == True]
    else:
        valid = df[(df['gps_latitude'] != 0) | (df['gps_longitude'] != 0)]

    if len(valid) == 0:
        ax.text(0.5, 0.5, 'No valid GPS fixes', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GPS')
        return

    ax.scatter(valid['gps_longitude'], valid['gps_latitude'], s=1, alpha=0.5)
    ax.plot(valid['gps_longitude'].iloc[0], valid['gps_latitude'].iloc[0], 'go', markersize=8, label='Start')
    ax.plot(valid['gps_longitude'].iloc[-1], valid['gps_latitude'].iloc[-1], 'ro', markersize=8, label='End')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('GPS Track')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def create_report(df: pd.DataFrame, title: str = "Vehicle State Report") -> Figure:
    """Create a 2x2 plot report."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Top row: Trajectory and Velocity
    plot_trajectory(df, axes[0, 0])
    plot_velocity(df, axes[0, 1])

    # Bottom row: Heading and Encoders
    plot_heading(df, axes[1, 0])
    plot_encoders(df, axes[1, 1])

    # Add summary stats
    duration = 0
    if 'timestamp' in df.columns:
        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]

    distance = 0
    if 'distance_traveled' in df.columns:
        distance = df['distance_traveled'].iloc[-1]

    avg_speed = 0
    if 'speed' in df.columns:
        avg_speed = df['speed'].mean()

    stats_text = f"Records: {len(df):,} | Duration: {duration:.1f}s | Distance: {distance:.1f}m | Avg Speed: {avg_speed:.2f}m/s"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig


def create_extended_report(df: pd.DataFrame, title: str = "Vehicle State Report") -> Figure:
    """Create a 2x3 plot report with GPS."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Row 1: Trajectory, Velocity, Heading
    plot_trajectory(df, axes[0, 0])
    plot_velocity(df, axes[0, 1])
    plot_heading(df, axes[0, 2])

    # Row 2: Encoders, GPS, Stats
    plot_encoders(df, axes[1, 0])
    plot_gps(df, axes[1, 1])

    # Summary stats in last panel
    axes[1, 2].axis('off')

    stats_lines = []
    stats_lines.append(f"Total Records: {len(df):,}")

    if 'timestamp' in df.columns:
        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        stats_lines.append(f"Duration: {duration:.1f} seconds")

    if 'distance_traveled' in df.columns:
        distance = df['distance_traveled'].iloc[-1]
        stats_lines.append(f"Distance: {distance:.1f} m")

    if 'speed' in df.columns:
        stats_lines.append(f"Avg Speed: {df['speed'].mean():.2f} m/s")
        stats_lines.append(f"Max Speed: {df['speed'].max():.2f} m/s")

    if 'x' in df.columns and 'y' in df.columns:
        x_range = df['x'].max() - df['x'].min()
        y_range = df['y'].max() - df['y'].min()
        stats_lines.append(f"X Range: {x_range:.1f} m")
        stats_lines.append(f"Y Range: {y_range:.1f} m")

    if 'gps_valid' in df.columns:
        valid_pct = df['gps_valid'].mean() * 100
        stats_lines.append(f"GPS Valid: {valid_pct:.1f}%")

    if 'source_adapter' in df.columns:
        adapter = df['source_adapter'].iloc[0] if len(df) > 0 else 'unknown'
        stats_lines.append(f"Source: {adapter}")

    stats_text = '\n'.join(stats_lines)
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 2].set_title('Session Statistics')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot vehicle state data from Parquet log files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'path',
        type=str,
        help='Path to session directory or single parquet file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (PNG, PDF, SVG). If not specified, shows interactive window.'
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Force interactive mode (show window even if output specified)'
    )
    parser.add_argument(
        '--extended',
        action='store_true',
        help='Create extended 2x3 report with GPS plot'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved images (default: 150)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Custom title for the report'
    )

    args = parser.parse_args()

    # Load data
    path = Path(args.path)
    print(f"Loading data from: {path}")

    try:
        df = load_session(path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df):,} records")

    # Load metadata for title
    metadata = load_metadata(path)
    if args.title:
        title = args.title
    elif metadata and 'session_name' in metadata:
        title = f"Vehicle State: {metadata['session_name']}"
    else:
        title = f"Vehicle State: {path.name}"

    # Create report
    if args.extended:
        fig = create_extended_report(df, title)
    else:
        fig = create_report(df, title)

    # Save or show
    if args.output:
        output_path = Path(args.output)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if args.interactive or not args.output:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    main()
