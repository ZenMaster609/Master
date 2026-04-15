"""CSV export helpers for cone memory track data.

These functions are pure file-writing logic with no Node dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from sim_car.cones.tracking.tracker import ConeTrack


def save_track_data_csv(
    csv_path: Path,
    tracks: list[ConeTrack],
    center: list[tuple[float, float]],
    logger,
) -> None:
    """Write cone track and centerline data to *csv_path* in CSV format."""
    try:
        with csv_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['section', 'id', 'x', 'y', 'z', 'label', 'seen_count', 'track_confidence'])
            for track in tracks:
                label, _color_conf = track.class_label()
                writer.writerow([
                    'remembered_track',
                    track.track_id,
                    f'{track.x:.6f}',
                    f'{track.y:.6f}',
                    f'{track.z:.6f}',
                    label,
                    track.seen_count,
                    f'{track.track_confidence:.6f}',
                ])
            for idx, (x, y) in enumerate(center):
                writer.writerow(['centerline', idx, f'{x:.6f}', f'{y:.6f}', '0.0', '', '', ''])
        logger.info(f'saved cone memory track data: {csv_path}')
    except Exception as exc:  # pylint: disable=broad-except
        logger.warn(f'failed to save cone memory track data csv: {exc}')


def resolve_logs_dir(run_id: str, base_path: str, logger) -> Optional[Path]:
    """Return the logs output directory for the current run, or None if unavailable."""
    if run_id:
        root = Path(base_path).expanduser() if base_path else _default_multidata_root()
        return root / run_id / 'logs'

    root = _default_multidata_root()
    if not root.exists():
        return None

    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime) / 'logs'


def _default_multidata_root() -> Path:
    candidate = Path.home() / 'ros2_ws' / 'src' / 'Master' / 'multidata'
    if candidate.exists():
        return candidate
    return Path.cwd() / 'multidata'
