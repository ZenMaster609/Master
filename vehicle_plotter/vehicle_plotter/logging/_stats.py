"""Shared numeric stat helpers for the logging subsystem."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def median_dt(timestamps: np.ndarray) -> float:
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size < 2:
        return float("nan")
    dt = np.diff(np.sort(finite))
    dt = dt[dt > 1e-6]
    if dt.size == 0:
        return float("nan")
    return float(np.median(dt))


def estimate_lag(desired: np.ndarray, actual: np.ndarray, dt: float) -> tuple[int, float]:
    mask = np.isfinite(desired) & np.isfinite(actual)
    if np.count_nonzero(mask) < 8:
        return 0, float("nan")
    x = desired[mask] - np.mean(desired[mask])
    y = actual[mask] - np.mean(actual[mask])
    corr = np.correlate(y, x, mode="full")
    lags = np.arange(-len(x) + 1, len(x), dtype=np.int64)
    best_idx = int(np.argmax(corr))
    lag_samples = int(lags[best_idx])
    lag_sec = float(lag_samples * dt) if math.isfinite(dt) else float("nan")
    return lag_samples, lag_sec


def nanmean(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def nanrms(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(valid * valid)))


def nanmaxabs(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.max(np.abs(valid)))


def nancorr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    av = a[mask]
    bv = b[mask]
    if np.std(av) < 1e-12 or np.std(bv) < 1e-12:
        return float("nan")
    corr = np.corrcoef(av, bv)[0, 1]
    return float(corr) if math.isfinite(float(corr)) else float("nan")
