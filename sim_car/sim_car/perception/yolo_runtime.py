"""Runtime helpers for selecting and running YOLO backends."""

from __future__ import annotations

import os

import numpy as np

from .yolo_onnx import YoloOnnxDetector
from .yolo_pt import YoloPtDetector


def init_yolo_detector(
    logger,
    *,
    enabled: bool,
    model_path: str,
    input_size: int,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    class_names: list[str],
    prefer_cuda: bool,
):
    if not enabled:
        return None

    resolved_model_path = resolve_yolo_model_path(model_path)
    if not resolved_model_path or not os.path.exists(resolved_model_path):
        logger.warn(f'YOLO model not found (yolo_model_path="{model_path}"); disabling YOLO.')
        return None

    try:
        if resolved_model_path.lower().endswith('.onnx'):
            return YoloOnnxDetector(
                logger=logger,
                model_path=resolved_model_path,
                input_size=input_size,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                class_names=class_names,
                prefer_cuda=prefer_cuda,
            )
        return YoloPtDetector(
            logger=logger,
            model_path=resolved_model_path,
            input_size=input_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_names=class_names,
            prefer_cuda=prefer_cuda,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warn(f'Failed to initialize YOLO detector ({exc}); disabling YOLO.')
        return None


def run_yolo(detector, image: np.ndarray, logger) -> tuple[list[dict], float]:
    if detector is None:
        return [], 0.0

    try:
        detections, infer_ms = detector.detect(image)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warn(f'YOLO inference failed ({exc})')
        return [], 0.0

    overlays = []
    for det in detections:
        overlays.append(
            {
                'x0': int(det.x0),
                'y0': int(det.y0),
                'x1': int(det.x1),
                'y1': int(det.y1),
                'confidence': float(det.confidence),
                'class_id': int(det.class_id),
                'label': str(det.label),
            }
        )
    return overlays, float(infer_ms)


def resolve_yolo_model_path(path: str) -> str:
    candidate = str(path).strip()
    if not candidate:
        return ''
    if os.path.isabs(candidate):
        return candidate
    direct = os.path.abspath(candidate)
    if os.path.exists(direct):
        return direct
    workspace_relative = os.path.abspath(os.path.join(os.path.expanduser('~/ros2_ws'), candidate))
    if os.path.exists(workspace_relative):
        return workspace_relative
    return direct
