"""Camera cone detection node (stereo or mono) without GT evaluation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
import threading
import time
from typing import Deque, Optional

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, Int32, Float32
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray

from sim_car.perception import (
    CameraDebugPublisher,
    StereoPipeline,
    StereoPipelineConfig,
    YoloOnnxDetector,
    YoloPtDetector,
    estimate_axis_depth_from_bbox_height,
)


@dataclass
class FramePacket:
    msg: Image
    pair_time_sec: float


class PerceptionNode(Node):
    """Detect cones from stereo or mono camera and publish ConeDetectionArray."""

    def __init__(self, *, force_stereo: Optional[bool] = None):
        super().__init__('perception_node')
        self._declare_parameters()
        self._read_parameters()
        if force_stereo is not None:
            self.stereo_enabled = bool(force_stereo)

        self._left_info: Optional[CameraInfo] = None
        self._right_info: Optional[CameraInfo] = None

        self._left_queue: Deque[FramePacket] = deque()
        self._right_queue: Deque[FramePacket] = deque()
        self._queue_lock = threading.Lock()
        self._queue_cv = threading.Condition(self._queue_lock)
        self._running = True

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_cache_lock = threading.Lock()
        self._tf_cache = {}
        self._last_throttled_log_sec: dict[str, float] = {}
        self._pipeline: Optional[StereoPipeline] = None
        if self.stereo_enabled:
            self._pipeline = StereoPipeline(
                logger=self.get_logger(),
                config=StereoPipelineConfig(
                    calibration_file=self.calibration_file,
                    prefer_cuda=self.prefer_cuda,
                    min_disparity=self.min_disparity,
                    num_disparities=self.num_disparities,
                    block_size=self.block_size,
                    uniqueness_ratio=self.uniqueness_ratio,
                    speckle_window_size=self.speckle_window_size,
                    speckle_range=self.speckle_range,
                    disp12_max_diff=self.disp12_max_diff,
                    pre_filter_cap=self.pre_filter_cap,
                    baseline_m=self.baseline_m,
                    focal_length_px=self.focal_length_px,
                    disparity_valid_threshold=self.disparity_valid_threshold,
                    min_depth_m=self.min_depth_m,
                    max_depth_m=self.max_depth_m,
                ),
            )

        self._camera_debug = CameraDebugPublisher(
            node=self,
            enabled=self.camera_debug,
            topic=self.camera_debug_topic,
            publish_every_n=self.camera_debug_n_frames,
        )
        self._yolo_detector = self._init_yolo_detector()
        self._yolo_backend = self._yolo_detector.backend if self._yolo_detector is not None else 'disabled'

        prefix = self.eval_topic_prefix.rstrip('/')
        self._yolo_count_pub = self.create_publisher(Int32, f'{prefix}/yolo/detection_count', 10)
        self._yolo_infer_ms_pub = self.create_publisher(Float32, f'{prefix}/yolo/inference_ms', 10)
        self._cone_detections_pub = self.create_publisher(ConeDetectionArray, self.cone_detections_topic, 10)

        self.create_subscription(Image, self.left_image_topic, self._left_image_cb, 10)
        self.create_subscription(CameraInfo, self.left_camera_info_topic, self._left_info_cb, 10)
        if self.stereo_enabled:
            self.create_subscription(Image, self.right_image_topic, self._right_image_cb, 10)
            self.create_subscription(CameraInfo, self.right_camera_info_topic, self._right_info_cb, 10)

        self._worker_thread: Optional[threading.Thread] = None
        if self.stereo_enabled:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

        self.get_logger().info(
            'perception_node ready: '
            f'stereo_enabled={self.stereo_enabled} left={self.left_image_topic} right={self.right_image_topic} '
            f'yolo_enabled={self.yolo_enabled} yolo_backend={self._yolo_backend} '
            f'cone_detections_topic={self.cone_detections_topic} cone_detections_frame={self.cone_detections_frame}'
        )

    def _declare_parameters(self):
        self.declare_parameter('left_image_topic', '/sim/raw/stereo/left/image_raw')
        self.declare_parameter('right_image_topic', '/sim/raw/stereo/right/image_raw')
        self.declare_parameter('left_camera_info_topic', '/sim/raw/stereo/left/camera_info')
        self.declare_parameter('right_camera_info_topic', '/sim/raw/stereo/right/camera_info')
        self.declare_parameter('stereo_enabled', True)
        self.declare_parameter('monocular_cone_height_m', 0.3034)
        self.declare_parameter('monocular_big_cone_height_m', 0.51)
        self.declare_parameter('monocular_bbox_height_offset_px', 0.0)

        self.declare_parameter('calibration_file', '')
        self.declare_parameter('max_time_diff_sec', 0.08)
        self.declare_parameter('queue_size', 30)
        self.declare_parameter('prefer_cuda', True)

        self.declare_parameter('min_disparity', 0)
        self.declare_parameter('num_disparities', 192)
        self.declare_parameter('block_size', 7)
        self.declare_parameter('uniqueness_ratio', 10)
        self.declare_parameter('speckle_window_size', 100)
        self.declare_parameter('speckle_range', 2)
        self.declare_parameter('disp12_max_diff', 1)
        self.declare_parameter('pre_filter_cap', 31)

        self.declare_parameter('baseline_m', 0.12)
        self.declare_parameter('focal_length_px', 0.0)
        self.declare_parameter('disparity_valid_threshold', 0.1)
        self.declare_parameter('min_depth_m', 0.3)
        self.declare_parameter('max_depth_m', 30.0)

        self.declare_parameter('eval_topic_prefix', '/sim/raw/stereo/eval')
        self.declare_parameter('camera_debug', True)
        self.declare_parameter('camera_debug_topic', '/sim/raw/stereo/camera_debug')
        self.declare_parameter('camera_debug_n_frames', 30)
        self.declare_parameter('camera_debug_scale', 0.5)
        self.declare_parameter('camera_debug_mono', True)

        self.declare_parameter('yolo_enabled', False)
        self.declare_parameter('yolo_model_path', '')
        self.declare_parameter('yolo_input_size', 960)
        self.declare_parameter('yolo_conf_threshold', 0.25)
        self.declare_parameter('yolo_iou_threshold', 0.45)
        self.declare_parameter('yolo_max_detections', 100)
        self.declare_parameter('yolo_prefer_cuda', True)
        self.declare_parameter('yolo_class_names', [])

        self.declare_parameter('cone_detections_topic', '/sim/raw/stereo/perception/cones_3d')
        self.declare_parameter('cone_detections_frame', 'base_footprint')
        self.declare_parameter('cone_dedup_radius_m', 0.85)
        self.declare_parameter('cone_eval_tf_timeout_sec', 0.0)

    def _read_parameters(self):
        self.left_image_topic = str(self.get_parameter('left_image_topic').value)
        self.right_image_topic = str(self.get_parameter('right_image_topic').value)
        self.left_camera_info_topic = str(self.get_parameter('left_camera_info_topic').value)
        self.right_camera_info_topic = str(self.get_parameter('right_camera_info_topic').value)
        self.stereo_enabled = bool(self.get_parameter('stereo_enabled').value)

        self.monocular_cone_height_m = max(1e-6, float(self.get_parameter('monocular_cone_height_m').value))
        self.monocular_big_cone_height_m = max(
            1e-6,
            float(self.get_parameter('monocular_big_cone_height_m').value),
        )
        self.monocular_bbox_height_offset_px = float(self.get_parameter('monocular_bbox_height_offset_px').value)

        self.calibration_file = str(self.get_parameter('calibration_file').value)
        self.max_time_diff_sec = max(0.0, float(self.get_parameter('max_time_diff_sec').value))
        self.queue_size = max(5, int(self.get_parameter('queue_size').value))
        self.prefer_cuda = bool(self.get_parameter('prefer_cuda').value)

        self.min_disparity = int(self.get_parameter('min_disparity').value)
        self.num_disparities = int(self.get_parameter('num_disparities').value)
        self.block_size = int(self.get_parameter('block_size').value)
        self.uniqueness_ratio = int(self.get_parameter('uniqueness_ratio').value)
        self.speckle_window_size = int(self.get_parameter('speckle_window_size').value)
        self.speckle_range = int(self.get_parameter('speckle_range').value)
        self.disp12_max_diff = int(self.get_parameter('disp12_max_diff').value)
        self.pre_filter_cap = int(self.get_parameter('pre_filter_cap').value)

        self.baseline_m = float(self.get_parameter('baseline_m').value)
        self.focal_length_px = float(self.get_parameter('focal_length_px').value)
        self.disparity_valid_threshold = float(self.get_parameter('disparity_valid_threshold').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)

        self.eval_topic_prefix = str(self.get_parameter('eval_topic_prefix').value)
        self.camera_debug = self._sanitize_camera_debug(self.get_parameter('camera_debug').value)
        self.camera_debug_topic = str(self.get_parameter('camera_debug_topic').value)
        self.camera_debug_n_frames = max(1, int(self.get_parameter('camera_debug_n_frames').value))
        self.camera_debug_scale = float(np.clip(float(self.get_parameter('camera_debug_scale').value), 0.1, 1.0))
        self.camera_debug_mono = bool(self.get_parameter('camera_debug_mono').value)

        self.yolo_enabled = bool(self.get_parameter('yolo_enabled').value)
        self.yolo_model_path = str(self.get_parameter('yolo_model_path').value)
        self.yolo_input_size = max(64, int(self.get_parameter('yolo_input_size').value))
        self.yolo_conf_threshold = float(self.get_parameter('yolo_conf_threshold').value)
        self.yolo_iou_threshold = float(self.get_parameter('yolo_iou_threshold').value)
        self.yolo_max_detections = max(1, int(self.get_parameter('yolo_max_detections').value))
        self.yolo_prefer_cuda = bool(self.get_parameter('yolo_prefer_cuda').value)
        self.yolo_class_names = self._sanitize_yolo_class_names(self.get_parameter('yolo_class_names').value)

        self.cone_detections_topic = str(self.get_parameter('cone_detections_topic').value)
        self.cone_detections_frame = str(self.get_parameter('cone_detections_frame').value).strip() or 'base_footprint'
        self.cone_dedup_radius_m = max(0.01, float(self.get_parameter('cone_dedup_radius_m').value))
        self.cone_eval_tf_timeout_sec = max(0.0, float(self.get_parameter('cone_eval_tf_timeout_sec').value))

    def _left_info_cb(self, msg: CameraInfo):
        self._left_info = msg

    def _right_info_cb(self, msg: CameraInfo):
        self._right_info = msg

    def _left_image_cb(self, msg: Image):
        if self.stereo_enabled:
            self._enqueue_frame(msg, side='left')
            return
        self._process_monocular_frame(msg)

    def _right_image_cb(self, msg: Image):
        self._enqueue_frame(msg, side='right')

    def _enqueue_frame(self, msg: Image, side: str):
        pair_time_sec = self._stamp_to_sec(msg)
        if pair_time_sec <= 0.0:
            pair_time_sec = time.monotonic()
        packet = FramePacket(msg=msg, pair_time_sec=pair_time_sec)

        with self._queue_cv:
            queue = self._left_queue if side == 'left' else self._right_queue
            queue.append(packet)
            while len(queue) > self.queue_size:
                queue.popleft()
            self._queue_cv.notify()

    def _pair_frames(self):
        while self._left_queue and self._right_queue:
            left_head = self._left_queue[0]
            best_idx = -1
            best_dt = None
            for idx, right_packet in enumerate(self._right_queue):
                dt = abs(left_head.pair_time_sec - right_packet.pair_time_sec)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_idx = idx

            if best_dt is not None and best_dt <= self.max_time_diff_sec:
                left_packet = self._left_queue.popleft()
                right_packet = self._right_queue[best_idx]
                del self._right_queue[best_idx]
                return left_packet, right_packet

            left_t = left_head.pair_time_sec
            right_oldest_t = self._right_queue[0].pair_time_sec
            right_newest_t = self._right_queue[-1].pair_time_sec
            if right_newest_t < left_t - self.max_time_diff_sec:
                self._right_queue.popleft()
                continue
            if left_t < right_oldest_t - self.max_time_diff_sec:
                self._left_queue.popleft()
                continue
            if left_t <= right_oldest_t:
                self._left_queue.popleft()
            else:
                self._right_queue.popleft()
        return None

    def _worker_loop(self):
        while True:
            with self._queue_cv:
                pair = None
                while self._running and pair is None:
                    pair = self._pair_frames()
                    if pair is None:
                        self._queue_cv.wait(timeout=0.05)
                if not self._running:
                    return

            left_packet, right_packet = pair
            if self._pipeline is None:
                continue
            output = self._pipeline.process(
                left_msg=left_packet.msg,
                right_msg=right_packet.msg,
                left_info=self._left_info,
                right_info=self._right_info,
            )
            if output is None:
                continue

            eval_header = self._common_header(left_packet.msg.header, right_packet.msg.header)
            left_info = self._pipeline.build_rectified_left_camera_info(self._left_info)

            stereo_debug_image = output.left_rect_color
            if stereo_debug_image is None:
                stereo_debug_image = output.left_rect
            yolo_input_image = output.left_rect_color
            if yolo_input_image is None:
                yolo_input_image = cv2.cvtColor(output.left_rect, cv2.COLOR_GRAY2BGR)

            yolo_detections, infer_ms = self._run_yolo(yolo_input_image)
            self._yolo_count_pub.publish(Int32(data=len(yolo_detections)))
            self._yolo_infer_ms_pub.publish(Float32(data=float(infer_ms)))
            self._apply_depth_map_to_detections(output.depth, yolo_detections)
            self._publish_cone_detections(yolo_detections=yolo_detections, left_info=left_info, eval_header=eval_header)

            if self._camera_debug.should_publish():
                debug_image = self._build_camera_debug_image(stereo_debug_image, yolo_detections)
                if debug_image is not None:
                    encoding = 'mono8' if self.camera_debug_mono else 'bgr8'
                    self._camera_debug.publish_image(eval_header, debug_image, encoding)

    def _process_monocular_frame(self, msg: Image) -> None:
        left_bgr = StereoPipeline._decode_to_bgr(msg)
        if left_bgr is None:
            return

        eval_header = Header()
        eval_header.stamp = msg.header.stamp
        eval_header.frame_id = msg.header.frame_id

        yolo_detections, infer_ms = self._run_yolo(left_bgr)
        self._yolo_count_pub.publish(Int32(data=len(yolo_detections)))
        self._yolo_infer_ms_pub.publish(Float32(data=float(infer_ms)))
        self._apply_monocular_depth_to_detections(yolo_detections, self._left_info)
        self._publish_cone_detections(yolo_detections=yolo_detections, left_info=self._left_info, eval_header=eval_header)

        if self._camera_debug.should_publish():
            debug_image = self._build_camera_debug_image(left_bgr, yolo_detections)
            if debug_image is not None:
                encoding = 'mono8' if self.camera_debug_mono else 'bgr8'
                self._camera_debug.publish_image(eval_header, debug_image, encoding)

    def _build_camera_debug_image(self, image: np.ndarray | None, yolo_detections: list[dict]) -> Optional[np.ndarray]:
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

        if self.camera_debug_scale < 0.999:
            debug_image = cv2.resize(
                debug_image,
                None,
                fx=self.camera_debug_scale,
                fy=self.camera_debug_scale,
                interpolation=cv2.INTER_AREA,
            )

        self._draw_yolo_debug_overlays(debug_image, yolo_detections)
        self._draw_debug_status_text(debug_image, yolo_detection_count=len(yolo_detections))

        if self.camera_debug_mono:
            return debug_image
        return cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)

    def _apply_monocular_depth_to_detections(self, yolo_detections: list[dict], left_info: Optional[CameraInfo]) -> None:
        if not yolo_detections:
            return

        _fx, fy, _cx, _cy = self._camera_intrinsics(left_info) if left_info is not None else (0.0, 0.0, 0.0, 0.0)
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
            det_color = self._normalize_detection_color(str(det.get('label', '')))
            cone_height_m = (
                self.monocular_big_cone_height_m if det_color == 'big_orange' else self.monocular_cone_height_m
            )
            depth_m = estimate_axis_depth_from_bbox_height(
                fy_px=fy,
                cone_height_m=cone_height_m,
                bbox_height_px=bbox_height_px,
                bbox_height_offset_px=self.monocular_bbox_height_offset_px,
            )
            det['depth_m'] = float(depth_m) if depth_m is not None else None

    def _apply_depth_map_to_detections(self, depth: np.ndarray, yolo_detections: list[dict]) -> None:
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
            est_axis = self._sample_depth_from_bbox(depth, x0, y0, x1, y1)
            det['depth_m'] = float(est_axis) if np.isfinite(est_axis) else None

    @staticmethod
    def _sample_depth_from_bbox(depth: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
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
            return PerceptionNode._sample_depth(depth, 0.5 * float(x0 + x1), 0.5 * float(y0 + y1), radius_px=2)

        patch = depth[crop_y0:crop_y1, crop_x0:crop_x1]
        valid = patch[np.isfinite(patch)]
        if valid.size > 0:
            return float(np.median(valid))
        return PerceptionNode._sample_depth(depth, 0.5 * float(x0 + x1), 0.5 * float(y0 + y1), radius_px=2)

    @staticmethod
    def _sample_depth(depth: np.ndarray, u: float, v: float, radius_px: int) -> float:
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

    def _draw_yolo_debug_overlays(self, image: np.ndarray, yolo_detections: list[dict]) -> None:
        if image is None or not yolo_detections:
            return

        height, width = image.shape[:2]
        scale = self.camera_debug_scale
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

    @staticmethod
    def _draw_debug_status_text(image: np.ndarray, yolo_detection_count: int) -> None:
        text = f'yolo={int(yolo_detection_count)}'
        cv2.rectangle(image, (6, 6), (120, 26), 0, -1)
        cv2.rectangle(image, (6, 6), (120, 26), 180, 1)
        cv2.putText(image, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 255, 1, cv2.LINE_AA)

    def _init_yolo_detector(self):
        if not self.yolo_enabled:
            return None
        model_path = self._resolve_yolo_model_path(self.yolo_model_path)
        if not model_path or not os.path.exists(model_path):
            self.get_logger().warn(f'YOLO model not found (yolo_model_path="{self.yolo_model_path}"); disabling YOLO.')
            return None

        try:
            if model_path.lower().endswith('.onnx'):
                return YoloOnnxDetector(
                    logger=self.get_logger(),
                    model_path=model_path,
                    input_size=self.yolo_input_size,
                    conf_threshold=self.yolo_conf_threshold,
                    iou_threshold=self.yolo_iou_threshold,
                    max_detections=self.yolo_max_detections,
                    class_names=self.yolo_class_names,
                    prefer_cuda=self.yolo_prefer_cuda,
                )
            return YoloPtDetector(
                logger=self.get_logger(),
                model_path=model_path,
                input_size=self.yolo_input_size,
                conf_threshold=self.yolo_conf_threshold,
                iou_threshold=self.yolo_iou_threshold,
                max_detections=self.yolo_max_detections,
                class_names=self.yolo_class_names,
                prefer_cuda=self.yolo_prefer_cuda,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f'Failed to initialize YOLO detector ({exc}); disabling YOLO.')
            return None

    def _run_yolo(self, left_rect: np.ndarray) -> tuple[list[dict], float]:
        if self._yolo_detector is None:
            return [], 0.0
        try:
            detections, infer_ms = self._yolo_detector.detect(left_rect)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f'YOLO inference failed ({exc})')
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

    def _publish_cone_detections(
        self,
        yolo_detections: list[dict],
        left_info: Optional[CameraInfo],
        eval_header: Header,
    ) -> None:
        msg = ConeDetectionArray()
        msg.header.stamp = eval_header.stamp
        msg.header.frame_id = self.cone_detections_frame

        if not yolo_detections or left_info is None:
            self._cone_detections_pub.publish(msg)
            return

        fx, fy, cx, cy = self._camera_intrinsics(left_info)
        if fx <= 0.0 or fy <= 0.0:
            self._cone_detections_pub.publish(msg)
            return

        camera_frame = str(left_info.header.frame_id).strip() or str(eval_header.frame_id).strip()
        if not camera_frame:
            self._cone_detections_pub.publish(msg)
            return

        projection_model = self._projection_model_for_frame(camera_frame)
        cam_to_output = None
        output_frame = self.cone_detections_frame
        transform_source_frame = camera_frame
        if camera_frame != self.cone_detections_frame:
            target_candidates = [self.cone_detections_frame]
            namespaced_frame = self._resolve_namespaced_output_frame(
                camera_frame=camera_frame,
                requested_frame=self.cone_detections_frame,
            )
            if namespaced_frame and namespaced_frame not in target_candidates:
                target_candidates.append(namespaced_frame)

            source_candidates = self._cone_output_source_frame_candidates(
                source_frame=camera_frame,
                requested_output_frame=self.cone_detections_frame,
            )

            for target_candidate in target_candidates:
                if cam_to_output is not None:
                    break
                for source_candidate in source_candidates:
                    candidate = self._lookup_transform(target_candidate, source_candidate, eval_header.stamp)
                    if candidate is None:
                        continue
                    cam_to_output = candidate
                    output_frame = target_candidate
                    transform_source_frame = source_candidate
                    msg.header.frame_id = output_frame
                    break

            if cam_to_output is None:
                output_frame = camera_frame
                msg.header.frame_id = output_frame
                self._warn_throttled(
                    'cone_frame_fallback',
                    f'cone detections transform unavailable {camera_frame}->{self.cone_detections_frame}; '
                    f'publishing in source frame "{output_frame}"',
                )

        reconstruction_model = self._projection_model_for_frame(transform_source_frame)
        cone_candidates: list[tuple[float, float, float, str, float]] = []

        for det in yolo_detections:
            axis_depth = det.get('depth_m')
            if axis_depth is None or not np.isfinite(float(axis_depth)):
                continue

            u_center = det.get('u_center')
            v_center = det.get('v_center')
            if u_center is None or v_center is None:
                x0 = float(det.get('x0', -1))
                y0 = float(det.get('y0', -1))
                x1 = float(det.get('x1', -1))
                y1 = float(det.get('y1', -1))
                if x1 <= x0 or y1 <= y0:
                    continue
                u_center = 0.5 * (x0 + x1)
                v_center = 0.5 * (y0 + y1)

            cam_point = self._reconstruct_cam_point_from_axis(
                u=float(u_center),
                v=float(v_center),
                axis_depth=float(axis_depth),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                model=reconstruction_model,
            )
            if cam_point is None:
                continue

            x_out, y_out, z_out = cam_point
            if cam_to_output is not None:
                x_out, y_out, z_out = self._transform_point(cam_to_output, x_out, y_out, z_out)

            confidence = det.get('confidence')
            cone_candidates.append(
                (
                    float(x_out),
                    float(y_out),
                    float(z_out),
                    self._normalize_detection_color(str(det.get('label', ''))),
                    float(max(0.0, min(1.0, float(confidence)))) if confidence is not None else 0.0,
                )
            )

        for x_out, y_out, z_out, color, confidence in self._deduplicate_cone_candidates(
            cone_candidates,
            self.cone_dedup_radius_m,
        ):
            cone = ConeDetection()
            cone.color = color
            cone.confidence = confidence
            cone.position.x = x_out
            cone.position.y = y_out
            cone.position.z = z_out
            msg.cones.append(cone)

        self._cone_detections_pub.publish(msg)

    @staticmethod
    def _deduplicate_cone_candidates(
        candidates: list[tuple[float, float, float, str, float]],
        dedup_radius_m: float,
    ) -> list[tuple[float, float, float, str, float]]:
        if len(candidates) <= 1:
            return list(candidates)

        radius_sq = float(dedup_radius_m) * float(dedup_radius_m)
        merged: list[tuple[float, float, float, str, float, float]] = []

        for x, y, z, color, confidence in sorted(
            candidates,
            key=lambda item: (-item[4], (item[0] * item[0]) + (item[1] * item[1])),
        ):
            best_idx = -1
            best_dist_sq = float('inf')
            for idx, (mx, my, _mz, mcolor, _mconf, _weight) in enumerate(merged):
                if color != mcolor and color != 'unknown' and mcolor != 'unknown':
                    continue
                dx = x - mx
                dy = y - my
                dist_sq = (dx * dx) + (dy * dy)
                if dist_sq <= radius_sq and dist_sq < best_dist_sq:
                    best_idx = idx
                    best_dist_sq = dist_sq

            weight = max(0.1, confidence)
            if best_idx < 0:
                merged.append((x, y, z, color, confidence, weight))
                continue

            mx, my, mz, mcolor, mconf, mweight = merged[best_idx]
            total_weight = mweight + weight
            merged[best_idx] = (
                ((mx * mweight) + (x * weight)) / total_weight,
                ((my * mweight) + (y * weight)) / total_weight,
                ((mz * mweight) + (z * weight)) / total_weight,
                color if confidence >= mconf else mcolor,
                max(mconf, confidence),
                total_weight,
            )

        return [(x, y, z, color, confidence) for x, y, z, color, confidence, _weight in merged]

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp):
        cache_key = (target_frame, source_frame)
        query_time = Time.from_msg(stamp)
        timeout = Duration(seconds=float(self.cone_eval_tf_timeout_sec))

        try:
            transform = self._tf_buffer.lookup_transform(target_frame, source_frame, query_time, timeout=timeout)
            with self._tf_cache_lock:
                self._tf_cache[cache_key] = transform
            return transform
        except TransformException:
            pass

        try:
            transform = self._tf_buffer.lookup_transform(target_frame, source_frame, Time(), timeout=timeout)
            with self._tf_cache_lock:
                self._tf_cache[cache_key] = transform
            return transform
        except TransformException:
            pass

        with self._tf_cache_lock:
            return self._tf_cache.get(cache_key)

    @staticmethod
    def _camera_intrinsics(camera_info: CameraInfo):
        fx = float(camera_info.k[0]) if len(camera_info.k) >= 1 else 0.0
        fy = float(camera_info.k[4]) if len(camera_info.k) >= 5 else 0.0
        cx = float(camera_info.k[2]) if len(camera_info.k) >= 3 else 0.0
        cy = float(camera_info.k[5]) if len(camera_info.k) >= 6 else 0.0
        return fx, fy, cx, cy

    @staticmethod
    def _transform_point(transform, x: float, y: float, z: float):
        t = transform.transform.translation
        q = transform.transform.rotation
        qx = float(q.x)
        qy = float(q.y)
        qz = float(q.z)
        qw = float(q.w)

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)
        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)

        tx = float(t.x)
        ty = float(t.y)
        tz = float(t.z)

        px = (r00 * x) + (r01 * y) + (r02 * z) + tx
        py = (r10 * x) + (r11 * y) + (r12 * z) + ty
        pz = (r20 * x) + (r21 * y) + (r22 * z) + tz
        return px, py, pz

    @staticmethod
    def _reconstruct_cam_point_from_axis(
        u: float,
        v: float,
        axis_depth: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        model: str,
    ):
        if not math.isfinite(axis_depth) or axis_depth <= 0.0 or fx == 0.0 or fy == 0.0:
            return None
        if model == 'forward_x':
            x_cam = axis_depth
            y_cam = -((u - cx) / fx) * x_cam
            z_cam = -((v - cy) / fy) * x_cam
            return float(x_cam), float(y_cam), float(z_cam)
        z_cam = axis_depth
        x_cam = ((u - cx) / fx) * z_cam
        y_cam = ((v - cy) / fy) * z_cam
        return float(x_cam), float(y_cam), float(z_cam)

    def _warn_throttled(self, key: str, message: str) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get(key, -1.0)
        if (now_sec - last_sec) >= 1.0:
            self.get_logger().warn(message)
            self._last_throttled_log_sec[key] = now_sec

    @staticmethod
    def _sanitize_camera_debug(value) -> bool:
        token = str(value).strip().lower()
        if token in {'', 'false', '0', 'off', 'none', 'no'}:
            return False
        return token in {'true', '1', 'on', 'yes', 'rect_left', 'left_rect', 'depth', 'disparity', 'yolo'}

    @staticmethod
    def _sanitize_yolo_class_names(value) -> list[str]:
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        if not text:
            return []
        return [token.strip() for token in text.split(',') if token.strip()]

    @staticmethod
    def _resolve_yolo_model_path(path: str) -> str:
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

    @staticmethod
    def _stamp_to_sec(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _projection_model_for_frame(frame_id: str) -> str:
        frame = str(frame_id).strip().lower()
        if 'optical' in frame or frame.endswith('_camera'):
            return 'optical_z'
        if frame.endswith('_link'):
            return 'forward_x'
        return 'optical_z'

    @staticmethod
    def _normalize_detection_color(label: str) -> str:
        token = str(label).strip().lower().replace('-', '_').replace(' ', '_')
        if (
            'big_orange' in token
            or 'large_orange' in token
            or (('big' in token or 'large' in token) and 'orange' in token)
        ):
            return 'big_orange'
        if 'orange' in token:
            return 'orange'
        if 'yellow' in token:
            return 'yellow'
        if 'blue' in token:
            return 'blue'
        return 'unknown'

    @staticmethod
    def _resolve_namespaced_output_frame(camera_frame: str, requested_frame: str) -> str:
        requested = str(requested_frame).strip().strip('/')
        source = str(camera_frame).strip().strip('/')
        if not requested or not source or '/' in requested:
            return ''
        marker = f'/{requested}/'
        source_with_slashes = f'/{source}/'
        idx = source_with_slashes.find(marker)
        if idx < 0:
            return ''
        prefix = source_with_slashes[1:idx].strip('/')
        if not prefix:
            return requested
        return f'{prefix}/{requested}'

    @classmethod
    def _cone_output_source_frame_candidates(cls, source_frame: str, requested_output_frame: str) -> list[str]:
        source = str(source_frame).strip().strip('/')
        if not source:
            return []

        candidates: list[str] = [source]
        if '/' in source:
            parts = [p for p in source.split('/') if p]
            if parts:
                leaf = parts[-1]
                if leaf not in candidates:
                    candidates.append(leaf)
                namespace_leaf = f'{parts[0]}/{leaf}'
                if namespace_leaf not in candidates:
                    candidates.append(namespace_leaf)

            requested = str(requested_output_frame).strip().strip('/')
            marker = f'/{requested}/' if requested else ''
            source_with_slashes = f'/{source}/'
            if marker and marker in source_with_slashes:
                idx = source_with_slashes.find(marker)
                prefix = source_with_slashes[1:idx].strip('/')
                suffix_start = idx + len(marker)
                suffix = source_with_slashes[suffix_start:-1].strip('/')
                if prefix and suffix:
                    prefixed_suffix = f'{prefix}/{suffix}'
                    if prefixed_suffix not in candidates:
                        candidates.append(prefixed_suffix)

        expanded = list(candidates)
        for token in candidates:
            if token.endswith('_camera'):
                link_token = token[:-7] + '_link'
                if link_token not in expanded:
                    expanded.append(link_token)
        return expanded

    @staticmethod
    def _max_stamp(a, b):
        if (int(a.sec), int(a.nanosec)) >= (int(b.sec), int(b.nanosec)):
            return a
        return b

    def _common_header(self, left_header: Header, right_header: Header) -> Header:
        header = Header()
        header.frame_id = left_header.frame_id or right_header.frame_id
        header.stamp = self._max_stamp(left_header.stamp, right_header.stamp)
        return header

    def destroy_node(self):
        with self._queue_cv:
            self._running = False
            self._queue_cv.notify_all()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None, *, force_stereo: Optional[bool] = None):
    rclpy.init(args=args)
    node = PerceptionNode(force_stereo=force_stereo)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
