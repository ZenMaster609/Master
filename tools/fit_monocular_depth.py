#!/usr/bin/env python3
"""Fit monocular depth correction parameters from logged per-detection samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


DEFAULT_RANGE_BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0))


@dataclass
class FitRow:
    timestamp: float
    session_source: str
    u_center_px: float
    v_center_px: float
    bbox_height_px: float
    bbox_width_px: float
    fy_px: float
    fx_px: float
    cx_px: float
    cy_px: float
    est_axis_depth_m: float
    gt_axis_depth_m: float
    axis_error_m: float
    gt_range_m: float
    gt_x_cam_m: float
    gt_y_cam_m: float
    gt_z_cam_m: float
    est_x_cam_m: float
    est_y_cam_m: float
    est_z_cam_m: float
    error_xy_m: float
    cone_color: str
    predicted_class_id: Optional[int]
    ground_truth_class_id: Optional[int]
    cone_id: str
    projection_model: str


@dataclass
class MetricSummary:
    samples: int
    axis_bias_m: float
    axis_rmse_m: float
    xy_rmse_m: float


def _parse_float(value: str) -> float:
    text = str(value).strip()
    if not text:
        return float('nan')
    try:
        return float(text)
    except ValueError:
        return float('nan')


def _parse_optional_int(value: str) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _resolve_csv_path(path_like: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [
            path / 'logs' / 'monocular_fit_samples.csv',
            path / 'monocular_fit_samples.csv',
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f'Could not resolve monocular fit CSV from {path_like}')


def load_rows(paths: Iterable[str]) -> list[FitRow]:
    rows: list[FitRow] = []
    for path_like in paths:
        csv_path = _resolve_csv_path(path_like)
        with open(csv_path, 'r', newline='') as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row = FitRow(
                    timestamp=_parse_float(raw.get('timestamp', '')),
                    session_source=str(raw.get('session_source', '')).strip(),
                    u_center_px=_parse_float(raw.get('u_center_px', '')),
                    v_center_px=_parse_float(raw.get('v_center_px', '')),
                    bbox_height_px=_parse_float(raw.get('bbox_height_px', '')),
                    bbox_width_px=_parse_float(raw.get('bbox_width_px', '')),
                    fy_px=_parse_float(raw.get('fy_px', '')),
                    fx_px=_parse_float(raw.get('fx_px', '')),
                    cx_px=_parse_float(raw.get('cx_px', '')),
                    cy_px=_parse_float(raw.get('cy_px', '')),
                    est_axis_depth_m=_parse_float(raw.get('est_axis_depth_m', '')),
                    gt_axis_depth_m=_parse_float(raw.get('gt_axis_depth_m', '')),
                    axis_error_m=_parse_float(raw.get('axis_error_m', '')),
                    gt_range_m=_parse_float(raw.get('gt_range_m', '')),
                    gt_x_cam_m=_parse_float(raw.get('gt_x_cam_m', '')),
                    gt_y_cam_m=_parse_float(raw.get('gt_y_cam_m', '')),
                    gt_z_cam_m=_parse_float(raw.get('gt_z_cam_m', '')),
                    est_x_cam_m=_parse_float(raw.get('est_x_cam_m', '')),
                    est_y_cam_m=_parse_float(raw.get('est_y_cam_m', '')),
                    est_z_cam_m=_parse_float(raw.get('est_z_cam_m', '')),
                    error_xy_m=_parse_float(raw.get('error_xy_m', '')),
                    cone_color=str(raw.get('cone_color', '')).strip(),
                    predicted_class_id=_parse_optional_int(raw.get('predicted_class_id', '')),
                    ground_truth_class_id=_parse_optional_int(raw.get('ground_truth_class_id', '')),
                    cone_id=str(raw.get('cone_id', '')).strip(),
                    projection_model=str(raw.get('projection_model', 'optical_z')).strip() or 'optical_z',
                )
                rows.append(row)
    return rows


def filter_rows(
    rows: Iterable[FitRow],
    min_range: Optional[float] = None,
    max_range: Optional[float] = None,
    outlier_threshold: Optional[float] = None,
) -> list[FitRow]:
    filtered: list[FitRow] = []
    for row in rows:
        if row.session_source and row.session_source.lower() != 'monocular':
            continue
        if not (
            math.isfinite(row.fy_px)
            and math.isfinite(row.bbox_height_px)
            and math.isfinite(row.gt_axis_depth_m)
            and math.isfinite(row.gt_range_m)
            and math.isfinite(row.fx_px)
            and math.isfinite(row.cx_px)
            and math.isfinite(row.cy_px)
            and math.isfinite(row.u_center_px)
            and math.isfinite(row.v_center_px)
            and math.isfinite(row.gt_x_cam_m)
            and math.isfinite(row.gt_y_cam_m)
            and math.isfinite(row.gt_z_cam_m)
        ):
            continue
        if row.bbox_height_px <= 1.0 or row.fy_px <= 0.0 or row.fx_px == 0.0:
            continue
        if min_range is not None and row.gt_range_m < min_range:
            continue
        if max_range is not None and row.gt_range_m > max_range:
            continue
        if outlier_threshold is not None:
            baseline_err = row.est_axis_depth_m - row.gt_axis_depth_m
            if math.isfinite(baseline_err) and abs(baseline_err) > outlier_threshold:
                continue
        filtered.append(row)
    return filtered


def predict_axis_depth(
    row: FitRow,
    cone_height_m: float,
    bbox_height_offset_px: float,
    scale: float = 1.0,
) -> float:
    effective_height_px = row.bbox_height_px - bbox_height_offset_px
    if effective_height_px <= 1.0:
        return float('nan')
    return float((row.fy_px * cone_height_m * scale) / effective_height_px)


def reconstruct_xy_error(row: FitRow, axis_depth: float) -> float:
    if not math.isfinite(axis_depth) or axis_depth <= 0.0:
        return float('nan')
    if row.projection_model == 'forward_x':
        est_x = axis_depth
        est_y = -((row.u_center_px - row.cx_px) / row.fx_px) * est_x
    else:
        est_z = axis_depth
        est_x = ((row.u_center_px - row.cx_px) / row.fx_px) * est_z
        est_y = ((row.v_center_px - row.cy_px) / row.fy_px) * est_z
    return float(math.hypot(est_x - row.gt_x_cam_m, est_y - row.gt_y_cam_m))


def summarize_rows(rows: Iterable[FitRow], predictor) -> MetricSummary:
    axis_errors = []
    xy_errors = []
    for row in rows:
        pred = predictor(row)
        if not math.isfinite(pred):
            continue
        axis_errors.append(pred - row.gt_axis_depth_m)
        xy_error = reconstruct_xy_error(row, pred)
        if math.isfinite(xy_error):
            xy_errors.append(xy_error)
    if not axis_errors:
        return MetricSummary(samples=0, axis_bias_m=float('nan'), axis_rmse_m=float('nan'), xy_rmse_m=float('nan'))
    axis_arr = np.asarray(axis_errors, dtype=np.float64)
    xy_arr = np.asarray(xy_errors, dtype=np.float64) if xy_errors else np.asarray([], dtype=np.float64)
    xy_rmse = float(np.sqrt(np.mean(np.square(xy_arr)))) if xy_arr.size > 0 else float('nan')
    return MetricSummary(
        samples=int(axis_arr.size),
        axis_bias_m=float(np.mean(axis_arr)),
        axis_rmse_m=float(np.sqrt(np.mean(np.square(axis_arr)))),
        xy_rmse_m=xy_rmse,
    )


def summarize_bins(rows: Iterable[FitRow], predictor) -> list[dict]:
    results = []
    row_list = list(rows)
    for lo, hi in DEFAULT_RANGE_BINS:
        selected = [row for row in row_list if lo <= row.gt_range_m < hi]
        summary = summarize_rows(selected, predictor)
        results.append(
            {
                'range_min_m': lo,
                'range_max_m': hi,
                'samples': summary.samples,
                'axis_bias_m': summary.axis_bias_m,
                'axis_rmse_m': summary.axis_rmse_m,
                'xy_rmse_m': summary.xy_rmse_m,
            }
        )
    return results


def fit_offset(rows: list[FitRow], cone_height_m: float, allow_scale: bool = False) -> dict:
    if not rows:
        raise ValueError('No valid rows available for fitting')

    bbox = np.asarray([row.bbox_height_px for row in rows], dtype=np.float64)
    fy = np.asarray([row.fy_px for row in rows], dtype=np.float64)
    gt = np.asarray([row.gt_axis_depth_m for row in rows], dtype=np.float64)

    coarse_offsets = np.linspace(-10.0, 30.0, 4001)
    best = None
    for offset in coarse_offsets:
        effective = bbox - offset
        valid = effective > 1.0
        if not np.any(valid):
            continue
        base_pred = (fy[valid] * cone_height_m) / effective[valid]
        if allow_scale:
            denom = float(np.dot(base_pred, base_pred))
            if denom <= 0.0:
                continue
            scale = float(np.dot(gt[valid], base_pred) / denom)
        else:
            scale = 1.0
        pred = scale * base_pred
        mse = float(np.mean(np.square(pred - gt[valid])))
        if best is None or mse < best['mse']:
            best = {'offset_px': float(offset), 'scale': float(scale), 'mse': mse}

    if best is None:
        raise RuntimeError('Failed to find a valid fit')

    refine_offsets = np.linspace(best['offset_px'] - 0.5, best['offset_px'] + 0.5, 2001)
    refined = best
    for offset in refine_offsets:
        effective = bbox - offset
        valid = effective > 1.0
        if not np.any(valid):
            continue
        base_pred = (fy[valid] * cone_height_m) / effective[valid]
        if allow_scale:
            denom = float(np.dot(base_pred, base_pred))
            if denom <= 0.0:
                continue
            scale = float(np.dot(gt[valid], base_pred) / denom)
        else:
            scale = 1.0
        pred = scale * base_pred
        mse = float(np.mean(np.square(pred - gt[valid])))
        if mse < refined['mse']:
            refined = {'offset_px': float(offset), 'scale': float(scale), 'mse': mse}
    return refined


def fit_scale(rows: list[FitRow], cone_height_m: float) -> dict:
    if not rows:
        raise ValueError('No valid rows available for fitting')
    base = np.asarray([predict_axis_depth(row, cone_height_m, 0.0, 1.0) for row in rows], dtype=np.float64)
    gt = np.asarray([row.gt_axis_depth_m for row in rows], dtype=np.float64)
    valid = np.isfinite(base) & np.isfinite(gt)
    if not np.any(valid):
        raise RuntimeError('Failed to build valid scale fit arrays')
    denom = float(np.dot(base[valid], base[valid]))
    if denom <= 0.0:
        raise RuntimeError('Degenerate scale fit')
    scale = float(np.dot(gt[valid], base[valid]) / denom)
    return {'offset_px': 0.0, 'scale': scale, 'mse': float(np.mean(np.square((scale * base[valid]) - gt[valid])))}


def format_metric(value: float) -> str:
    return 'n/a' if not math.isfinite(value) else f'{value:.4f}'


def build_report(rows: list[FitRow], fit_mode: str, cone_height_m: float) -> dict:
    if fit_mode == 'offset':
        fit = fit_offset(rows, cone_height_m, allow_scale=False)
    elif fit_mode == 'scale':
        fit = fit_scale(rows, cone_height_m)
    elif fit_mode == 'both':
        fit = fit_offset(rows, cone_height_m, allow_scale=True)
    else:
        raise ValueError(f'Unsupported fit mode: {fit_mode}')

    baseline = summarize_rows(rows, lambda row: row.est_axis_depth_m)
    corrected = summarize_rows(
        rows,
        lambda row: predict_axis_depth(row, cone_height_m, fit['offset_px'], fit['scale']),
    )
    report = {
        'fit_mode': fit_mode,
        'sample_count': len(rows),
        'fitted_bbox_height_offset_px': fit['offset_px'],
        'fitted_scale': fit['scale'],
        'fitted_cone_height_m': float(cone_height_m * fit['scale']),
        'baseline': asdict(baseline),
        'corrected': asdict(corrected),
        'baseline_bins': summarize_bins(rows, lambda row: row.est_axis_depth_m),
        'corrected_bins': summarize_bins(
            rows,
            lambda row: predict_axis_depth(row, cone_height_m, fit['offset_px'], fit['scale']),
        ),
    }
    return report


def print_report(report: dict) -> None:
    print(f"Samples: {report['sample_count']}")
    print(f"Fit mode: {report['fit_mode']}")
    print(f"Fitted bbox_height_offset_px: {report['fitted_bbox_height_offset_px']:.4f}")
    print(f"Fitted scale: {report['fitted_scale']:.6f}")
    print(f"Fitted cone height (effective): {report['fitted_cone_height_m']:.6f} m")
    print('')
    print('Baseline:')
    print(
        f"  axis_bias_m={format_metric(report['baseline']['axis_bias_m'])} "
        f"axis_rmse_m={format_metric(report['baseline']['axis_rmse_m'])} "
        f"xy_rmse_m={format_metric(report['baseline']['xy_rmse_m'])}"
    )
    print('Corrected:')
    print(
        f"  axis_bias_m={format_metric(report['corrected']['axis_bias_m'])} "
        f"axis_rmse_m={format_metric(report['corrected']['axis_rmse_m'])} "
        f"xy_rmse_m={format_metric(report['corrected']['xy_rmse_m'])}"
    )
    print('')
    print('Per-range bins:')
    for before, after in zip(report['baseline_bins'], report['corrected_bins']):
        lo = before['range_min_m']
        hi = before['range_max_m']
        print(
            f"  [{lo:.0f},{hi:.0f}) n={before['samples']} "
            f"bias {format_metric(before['axis_bias_m'])}->{format_metric(after['axis_bias_m'])} "
            f"rmse {format_metric(before['axis_rmse_m'])}->{format_metric(after['axis_rmse_m'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fit monocular depth correction from logged fit samples.')
    parser.add_argument('paths', nargs='+', help='Session directory, logs directory, or monocular_fit_samples.csv path')
    parser.add_argument('--fit-mode', choices=('offset', 'scale', 'both'), default='offset')
    parser.add_argument('--cone-height-m', type=float, default=0.3034)
    parser.add_argument('--min-range', type=float, default=None)
    parser.add_argument('--max-range', type=float, default=None)
    parser.add_argument('--outlier-threshold', type=float, default=None)
    parser.add_argument('--json-out', type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.paths)
    filtered = filter_rows(
        rows,
        min_range=args.min_range,
        max_range=args.max_range,
        outlier_threshold=args.outlier_threshold,
    )
    if not filtered:
        raise SystemExit('No valid monocular fit rows found after filtering')

    report = build_report(filtered, fit_mode=args.fit_mode, cone_height_m=float(args.cone_height_m))
    print_report(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
