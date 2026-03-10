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
        debug_image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
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
    _draw_debug_status_text(debug_image, yolo_detection_count=len(yolo_detections))

    if mono:
        return debug_image
    return cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)


def _draw_yolo_debug_overlays(image: np.ndarray, yolo_detections: list[dict], *, scale: float) -> None:
    if image is None or not yolo_detections:
        return

    height, width = image.shape[:2]
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
        cv2.rectangle(image, (x0, y0), (x1, y1), 255, 1)

        label = str(det.get('label', '')).strip()
        if not label:
            continue
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = max(0, min(width - text_w - 3, x0))
        text_y = max(text_h + baseline + 2, y0 - 2)
        box_y0 = max(0, text_y - text_h - baseline - 2)
        box_y1 = min(height - 1, text_y + 1)
        box_x1 = min(width - 1, text_x + text_w + 2)
        cv2.rectangle(image, (text_x, box_y0), (box_x1, box_y1), 0, -1)
        cv2.putText(image, label, (text_x + 1, text_y), font, font_scale, 255, thickness, cv2.LINE_AA)


def _draw_debug_status_text(image: np.ndarray, yolo_detection_count: int) -> None:
    text = f'yolo={int(yolo_detection_count)}'
    cv2.rectangle(image, (6, 6), (120, 26), 0, -1)
    cv2.rectangle(image, (6, 6), (120, 26), 180, 1)
    cv2.putText(image, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 255, 1, cv2.LINE_AA)
