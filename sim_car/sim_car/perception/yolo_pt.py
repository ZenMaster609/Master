"""PyTorch (.pt) YOLO inference helper using Ultralytics."""

from dataclasses import dataclass
import time
from typing import Optional

import cv2
import numpy as np

from .yolo_onnx import YoloDetection


@dataclass(frozen=True)
class _PtBackendDeps:
    torch: object
    yolo_cls: object


class YoloPtDetector:
    """Runs YOLO .pt inference through Ultralytics and returns detections."""

    def __init__(
        self,
        logger,
        model_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        class_names: Optional[list[str]] = None,
        prefer_cuda: bool = True,
    ):
        self._logger = logger
        self._model_path = str(model_path)
        self._input_size = max(64, int(input_size))
        self._conf_threshold = float(conf_threshold)
        self._iou_threshold = float(iou_threshold)
        self._max_detections = max(1, int(max_detections))
        self._class_names = list(class_names) if class_names else []
        self._prefer_cuda = bool(prefer_cuda)

        deps = self._load_deps()
        self._torch = deps.torch
        self._device = 'cuda:0' if self._prefer_cuda else 'cpu'
        if self._prefer_cuda and not bool(self._torch.cuda.is_available()):
            raise RuntimeError('Ultralytics .pt requested CUDA-only execution, but torch CUDA is unavailable.')
        self._model = deps.yolo_cls(self._model_path)
        self._backend = f'ultralytics-pt-{self._device}'
        self._logger.info(
            f'YOLO PT enabled: model={self._model_path} backend={self._backend} '
            f'input={self._input_size} conf={self._conf_threshold:.2f} iou={self._iou_threshold:.2f}'
        )

    @property
    def backend(self) -> str:
        """Current backend string."""
        return self._backend

    @staticmethod
    def _load_deps() -> _PtBackendDeps:
        try:
            import torch  # type: ignore
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                'Ultralytics .pt backend unavailable. Install ultralytics+torch in a separate venv and '
                'expose only that venv site-packages via launch PYTHONPATH.'
            ) from exc
        return _PtBackendDeps(torch=torch, yolo_cls=YOLO)

    def detect(self, image: np.ndarray) -> tuple[list[YoloDetection], float]:
        """Run one inference pass on a single grayscale or BGR frame."""
        t0 = time.perf_counter()
        if image is None or image.size == 0:
            return [], 0.0

        if image.ndim == 2:
            source = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            source = image
        elif image.ndim == 3 and image.shape[2] == 1:
            source = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            return [], 0.0

        results = self._model.predict(
            source=source,
            imgsz=self._input_size,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            max_det=self._max_detections,
            device=self._device,
            verbose=False,
        )
        if not results:
            return [], (time.perf_counter() - t0) * 1000.0

        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) <= 0:
            return [], (time.perf_counter() - t0) * 1000.0

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()
        names = getattr(result, 'names', None) or {}

        detections: list[YoloDetection] = []
        for i in range(len(xyxy)):
            x0, y0, x1, y1 = xyxy[i].tolist()
            class_id = int(cls[i])
            if 0 <= class_id < len(self._class_names):
                label = str(self._class_names[class_id])
            else:
                label = str(names.get(class_id, f'class_{class_id}'))
            detections.append(
                YoloDetection(
                    x0=int(round(x0)),
                    y0=int(round(y0)),
                    x1=int(round(x1)),
                    y1=int(round(y1)),
                    confidence=float(conf[i]),
                    class_id=class_id,
                    label=label,
                )
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return detections, elapsed_ms
