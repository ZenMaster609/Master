"""Debug image rendering for camera detections."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def build_camera_debug_image(
    image: np.ndarray | None,
    yolo_detections: list[dict],
    *,
    scale: float = 0.5,
    mono: bool = True,
) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None

    if image.ndim == 2:
        debug_image = image.copy()
    elif image.ndim == 3 and image.shape[2] == 1:
        debug_image = image[:, :, 0].copy()
    elif image.ndim == 3 and image.shape[2] >= 3:
        debug_image = image[:, :, :3].copy() if not mono else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        return None

    if scale < 0.999:
        debug_image = cv2.resize(
            debug_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    _draw_yolo_debug_overlays(debug_image, yolo_detections, scale=scale)
    if mono:
        if debug_image.ndim == 3 and debug_image.shape[2] >= 3:
            return cv2.cvtColor(debug_image[:, :, :3], cv2.COLOR_BGR2GRAY)
        return debug_image
    if debug_image.ndim == 2:
        return cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
    return debug_image


def _draw_yolo_debug_overlays(image: np.ndarray, yolo_detections: list[dict], *, scale: float) -> None:
    if image is None or not yolo_detections:
        return

    height, width = image.shape[:2]
    is_color = image.ndim == 3 and image.shape[2] >= 3
    line_color = (255, 255, 255) if is_color else 255
    text_color = (255, 255, 255) if is_color else 255
    text_bg_color = (0, 0, 0) if is_color else 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.40
    thickness = 1

    for det in yolo_detections:
        x0 = int(round(float(det.get('x0', -1)) * scale))
        y0 = int(round(float(det.get('y0', -1)) * scale))
        x1 = int(round(float(det.get('x1', -1)) * scale))
        y1 = int(round(float(det.get('y1', -1)) * scale))
        if x1 <= x0 or y1 <= y0:
            continue

        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        cv2.rectangle(image, (x0, y0), (x1, y1), line_color, 1)

        label = str(det.get('label', '')).strip()
        if not label:
            continue
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = max(0, min(width - text_w - 3, x0))
        text_y = max(text_h + baseline + 2, y0 - 2)
        box_y0 = max(0, text_y - text_h - baseline - 2)
        box_y1 = min(height - 1, text_y + 1)
        box_x1 = min(width - 1, text_x + text_w + 2)
        cv2.rectangle(image, (text_x, box_y0), (box_x1, box_y1), text_bg_color, -1)
        cv2.putText(image, label, (text_x + 1, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
