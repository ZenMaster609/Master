"""Shared algorithm helpers for planner core modules."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

_INITIAL_REJECT_COUNT = 0  # Reject counters start at zero before candidate checks run.


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))


def _default_reject_counts(reason_keys: Iterable[str]) -> dict[str, int]:
    return {str(key): _INITIAL_REJECT_COUNT for key in reason_keys}
