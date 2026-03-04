"""Helpers for monocular cone depth estimation."""

from __future__ import annotations

import math
from typing import Optional


def estimate_axis_depth_from_bbox_height(
    fy_px: float,
    cone_height_m: float,
    bbox_height_px: float,
    bbox_height_offset_px: float = 0.0,
) -> Optional[float]:
    """Estimate axis depth from pinhole geometry using bbox pixel height."""
    if not math.isfinite(fy_px) or fy_px <= 0.0:
        return None
    if not math.isfinite(cone_height_m) or cone_height_m <= 0.0:
        return None
    if not math.isfinite(bbox_height_px) or bbox_height_px <= 0.0:
        return None
    if not math.isfinite(bbox_height_offset_px):
        return None
    effective_height_px = bbox_height_px - bbox_height_offset_px
    if effective_height_px <= 1.0:
        return None
    return float((fy_px * cone_height_m) / effective_height_px)
