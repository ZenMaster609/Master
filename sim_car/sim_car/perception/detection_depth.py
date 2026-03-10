"""Depth assignment helpers for monocular and stereo detections."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .monocular_depth import estimate_axis_depth_from_bbox_height


def apply_monocular_depth_to_detections(
    yolo_detections: list[dict],
    *,
    fy_px: float,
    cone_height_m: float,
    big_cone_height_m: float,
    bbox_height_offset_px: float,
    normalize_detection_color: Callable[[str], str],
) -> None:
    if not yolo_detections:
        return

    for det in yolo_detections:
        x0 = float(det.get('x0', -1.0))
        y0 = float(det.get('y0', -1.0))
        x1 = float(det.get('x1', -1.0))
        y1 = float(det.get('y1', -1.0))
        if x1 <= x0 or y1 <= y0:
            det['depth_m'] = None
            continue

        det['u_center'] = 0.5 * (x0 + x1)
        det['v_center'] = 0.5 * (y0 + y1)
        bbox_height_px = y1 - y0
        det_color = normalize_detection_color(str(det.get('label', '')))
        active_cone_height_m = big_cone_height_m if det_color == 'big_orange' else cone_height_m
        depth_m = estimate_axis_depth_from_bbox_height(
            fy_px=fy_px,
            cone_height_m=active_cone_height_m,
            bbox_height_px=bbox_height_px,
            bbox_height_offset_px=bbox_height_offset_px,
        )
        det['depth_m'] = float(depth_m) if depth_m is not None else None


def apply_depth_map_to_detections(depth: np.ndarray, yolo_detections: list[dict]) -> None:
    if depth is None or depth.size == 0 or not yolo_detections:
        return

    height, width = depth.shape[:2]
    for det in yolo_detections:
        x0 = int(det.get('x0', -1))
        y0 = int(det.get('y0', -1))
        x1 = int(det.get('x1', -1))
        y1 = int(det.get('y1', -1))
        if x1 <= x0 or y1 <= y0:
            det['depth_m'] = None
            continue

        u = max(0.0, min(float(width - 1), 0.5 * float(x0 + x1)))
        v = max(0.0, min(float(height - 1), 0.5 * float(y0 + y1)))
        det['u_center'] = float(u)
        det['v_center'] = float(v)
        est_axis = sample_depth_from_bbox(depth, x0, y0, x1, y1)
        det['depth_m'] = float(est_axis) if np.isfinite(est_axis) else None


def sample_depth_from_bbox(depth: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    height, width = depth.shape[:2]
    if x1 <= x0 or y1 <= y0:
        return float('nan')

    box_w = x1 - x0
    box_h = y1 - y0
    crop_x0 = max(0, x0 + int(round(0.25 * box_w)))
    crop_x1 = min(width, x1 - int(round(0.25 * box_w)))
    crop_y0 = max(0, y0 + int(round(0.45 * box_h)))
    crop_y1 = min(height, y0 + int(round(0.95 * box_h)))
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return sample_depth(depth, 0.5 * float(x0 + x1), 0.5 * float(y0 + y1), radius_px=2)

    patch = depth[crop_y0:crop_y1, crop_x0:crop_x1]
    valid = patch[np.isfinite(patch)]
    if valid.size > 0:
        return float(np.median(valid))
    return sample_depth(depth, 0.5 * float(x0 + x1), 0.5 * float(y0 + y1), radius_px=2)


def sample_depth(depth: np.ndarray, u: float, v: float, radius_px: int) -> float:
    u_i = int(round(u))
    v_i = int(round(v))
    height, width = depth.shape[:2]
    if u_i < 0 or v_i < 0 or u_i >= width or v_i >= height:
        return float('nan')

    if radius_px <= 0:
        value = float(depth[v_i, u_i])
        return value if np.isfinite(value) else float('nan')

    u0 = max(0, u_i - radius_px)
    u1 = min(width, u_i + radius_px + 1)
    v0 = max(0, v_i - radius_px)
    v1 = min(height, v_i + radius_px + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch)]
    if valid.size == 0:
        return float('nan')
    return float(np.median(valid))
