"""ONNX YOLO inference helper with OpenCV-DNN/ONNXRuntime backends."""

from dataclasses import dataclass
import time
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pylint: disable=broad-except
    ort = None


@dataclass(frozen=True)
class YoloDetection:
    """One detected object in source-image pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float
    class_id: int
    label: str


class YoloOnnxDetector:
    """Runs ONNX YOLO inference and returns post-processed detections."""

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

        self._backend = ''
        self._cv2_net = None
        self._ort_session = None
        self._ort_input_name = ''
        self._init_backend()
        self._logger.info(
            f'YOLO ONNX enabled: model={self._model_path} backend={self._backend} '
            f'input={self._input_size} conf={self._conf_threshold:.2f} iou={self._iou_threshold:.2f}'
        )

    @property
    def backend(self) -> str:
        """Current backend string."""
        return self._backend

    def _init_backend(self):
        cv2_has_dnn = hasattr(cv2, 'dnn') and hasattr(cv2.dnn, 'readNetFromONNX')
        if cv2_has_dnn:
            try:
                self._cv2_net = cv2.dnn.readNetFromONNX(self._model_path)
                if self._prefer_cuda:
                    try:
                        self._cv2_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self._cv2_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        self._backend = 'opencv-dnn-cuda'
                    except Exception as exc:  # pylint: disable=broad-except
                        raise RuntimeError(
                            f'YOLO ONNX OpenCV CUDA unavailable ({exc}); '
                            'configured for CUDA-only execution.'
                        ) from exc
                else:
                    self._cv2_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self._cv2_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self._backend = 'opencv-dnn-cpu'
                return
            except Exception as exc:  # pylint: disable=broad-except
                if self._prefer_cuda:
                    raise RuntimeError(
                        f'YOLO ONNX OpenCV-DNN init failed ({exc}); '
                        'configured for OpenCV-DNN CUDA only.'
                    ) from exc
                self._logger.warn(f'YOLO ONNX OpenCV-DNN init failed ({exc}); trying ONNXRuntime backend.')
                self._cv2_net = None

        if self._prefer_cuda:
            raise RuntimeError(
                'OpenCV DNN ONNX backend is not available, but CUDA-only YOLO was requested.'
            )

        if ort is None:
            raise RuntimeError(
                'No ONNX backend available: OpenCV lacks cv2.dnn and onnxruntime is not installed.'
            )

        providers = ['CPUExecutionProvider']
        if self._prefer_cuda:
            available = set(ort.get_available_providers())
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._ort_session = ort.InferenceSession(self._model_path, providers=providers)
        self._ort_input_name = self._ort_session.get_inputs()[0].name
        active_provider = self._ort_session.get_providers()[0] if self._ort_session.get_providers() else 'unknown'
        self._backend = f'onnxruntime-{active_provider}'

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

        input_image, scale, pad_x, pad_y = self._letterbox(source, self._input_size)
        raw_output = self._run_backend(input_image)
        detections = self._decode_output(
            raw_output=raw_output,
            src_width=int(source.shape[1]),
            src_height=int(source.shape[0]),
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return detections, elapsed_ms

    def _run_backend(self, input_image: np.ndarray):
        if self._cv2_net is not None:
            blob = cv2.dnn.blobFromImage(
                input_image,
                scalefactor=1.0 / 255.0,
                size=(self._input_size, self._input_size),
                mean=(0.0, 0.0, 0.0),
                swapRB=True,
                crop=False,
            )
            self._cv2_net.setInput(blob)
            return self._cv2_net.forward()

        if self._ort_session is None:
            raise RuntimeError('YOLO backend was not initialized.')

        tensor = np.transpose(input_image, (2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        outputs = self._ort_session.run(None, {self._ort_input_name: tensor})
        if not outputs:
            return np.empty((0,), dtype=np.float32)
        return outputs[0]

    @staticmethod
    def _letterbox(image: np.ndarray, target_size: int) -> tuple[np.ndarray, float, int, int]:
        src_h, src_w = image.shape[:2]
        scale = min(float(target_size) / max(1, src_w), float(target_size) / max(1, src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    def _decode_output(
        self,
        raw_output,
        src_width: int,
        src_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[YoloDetection]:
        predictions = np.array(raw_output)
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]

        if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
            # Typical YOLOv8 export shape: (84, 8400) -> transpose to (8400, 84)
            predictions = predictions.T
        if predictions.ndim != 2 or predictions.shape[1] < 5:
            return []

        boxes_xywh: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        for row in predictions:
            cx, cy, bw, bh = row[0:4]
            if bw <= 1e-3 or bh <= 1e-3:
                continue

            class_scores = row[4:]
            if class_scores.size <= 0:
                continue

            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            # Also support obj*cls style heads when present.
            if row.shape[0] >= 6:
                obj_conf = float(row[4])
                cls_scores_obj = row[5:]
                if cls_scores_obj.size > 0:
                    obj_class_id = int(np.argmax(cls_scores_obj))
                    obj_style_conf = obj_conf * float(cls_scores_obj[obj_class_id])
                    if obj_style_conf > confidence:
                        confidence = obj_style_conf
                        class_id = obj_class_id

            if confidence < self._conf_threshold:
                continue

            x = float(cx - (bw * 0.5))
            y = float(cy - (bh * 0.5))
            boxes_xywh.append([x, y, float(bw), float(bh)])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes_xywh:
            return []

        indices = self._nms_indices(boxes_xywh, scores, self._iou_threshold)
        if not indices:
            return []

        detections: list[YoloDetection] = []
        for idx in indices[: self._max_detections]:
            x, y, w, h = boxes_xywh[int(idx)]
            x0 = int(round((x - float(pad_x)) / max(1e-6, scale)))
            y0 = int(round((y - float(pad_y)) / max(1e-6, scale)))
            x1 = int(round((x + w - float(pad_x)) / max(1e-6, scale)))
            y1 = int(round((y + h - float(pad_y)) / max(1e-6, scale)))

            x0 = max(0, min(src_width - 1, x0))
            y0 = max(0, min(src_height - 1, y0))
            x1 = max(0, min(src_width - 1, x1))
            y1 = max(0, min(src_height - 1, y1))
            if x1 <= x0 or y1 <= y0:
                continue

            class_id = int(class_ids[int(idx)])
            if 0 <= class_id < len(self._class_names):
                label = str(self._class_names[class_id])
            else:
                label = f'class_{class_id}'

            detections.append(
                YoloDetection(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    confidence=float(scores[int(idx)]),
                    class_id=class_id,
                    label=label,
                )
            )
        return detections

    @staticmethod
    def _nms_indices(boxes_xywh: list[list[float]], scores: list[float], iou_threshold: float) -> list[int]:
        if not boxes_xywh:
            return []

        x0 = np.array([b[0] for b in boxes_xywh], dtype=np.float32)
        y0 = np.array([b[1] for b in boxes_xywh], dtype=np.float32)
        x1 = np.array([b[0] + b[2] for b in boxes_xywh], dtype=np.float32)
        y1 = np.array([b[1] + b[3] for b in boxes_xywh], dtype=np.float32)
        areas = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
        order = np.argsort(np.array(scores, dtype=np.float32))[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            others = order[1:]
            xx0 = np.maximum(x0[i], x0[others])
            yy0 = np.maximum(y0[i], y0[others])
            xx1 = np.minimum(x1[i], x1[others])
            yy1 = np.minimum(y1[i], y1[others])

            inter_w = np.maximum(0.0, xx1 - xx0)
            inter_h = np.maximum(0.0, yy1 - yy0)
            intersection = inter_w * inter_h
            union = areas[i] + areas[others] - intersection
            iou = np.where(union > 0.0, intersection / union, 0.0)
            order = others[iou <= float(iou_threshold)]

        return keep
