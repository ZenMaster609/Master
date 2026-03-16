"""Camera cone detection node for stereo and monocular perception."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray

from .cone_geometry import (
    camera_intrinsics,
    cone_output_source_frame_candidates,
    deduplicate_cone_candidates,
    normalize_detection_color,
    projection_model_for_frame,
    reconstruct_cam_point_from_axis,
    resolve_namespaced_output_frame,
    transform_point,
)
from .debug_render import build_camera_debug_image
from .debug_view import CameraDebugPublisher
from .detection_depth import apply_depth_map_to_detections, apply_monocular_depth_to_detections
from .perception_params import declare_parameters, load_parameters
from .stereo_pipeline import StereoPipeline, StereoPipelineConfig
from .yolo_runtime import init_yolo_detector, run_yolo


_DEBUG_IMAGE_SCALE = 0.5
_DEBUG_IMAGE_MONO = False


@dataclass
class FramePacket:
    msg: Image
    pair_time_sec: float


class PerceptionNode(Node):
    """Detect cones from stereo or mono camera and publish ConeDetectionArray."""

    def __init__(self):
        super().__init__('perception_node')
        declare_parameters(self)
        self._config = load_parameters(self)

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
        if self._config.stereo_enabled:
            self._pipeline = StereoPipeline(
                logger=self.get_logger(),
                config=StereoPipelineConfig(
                    calibration_file=self._config.calibration_file,
                    prefer_cuda=self._config.prefer_cuda,
                    min_disparity=self._config.min_disparity,
                    num_disparities=self._config.num_disparities,
                    block_size=self._config.block_size,
                    uniqueness_ratio=self._config.uniqueness_ratio,
                    speckle_window_size=self._config.speckle_window_size,
                    speckle_range=self._config.speckle_range,
                    disp12_max_diff=self._config.disp12_max_diff,
                    pre_filter_cap=self._config.pre_filter_cap,
                    baseline_m=self._config.baseline_m,
                    focal_length_px=self._config.focal_length_px,
                    disparity_valid_threshold=self._config.disparity_valid_threshold,
                    min_depth_m=self._config.min_depth_m,
                    max_depth_m=self._config.max_depth_m,
                ),
            )

        self._camera_debug = CameraDebugPublisher(
            node=self,
            enabled=self._config.camera_debug,
            topic=self._config.camera_debug_topic,
            publish_every_n=self._config.camera_debug_n_frames,
        )
        self._yolo_detector = init_yolo_detector(
            self.get_logger(),
            enabled=self._config.yolo_enabled,
            model_path=self._config.yolo_model_path,
            input_size=self._config.yolo_input_size,
            conf_threshold=self._config.yolo_conf_threshold,
            iou_threshold=self._config.yolo_iou_threshold,
            max_detections=self._config.yolo_max_detections,
            class_names=self._config.yolo_class_names,
            prefer_cuda=self._config.yolo_prefer_cuda,
        )
        self._yolo_backend = self._yolo_detector.backend if self._yolo_detector is not None else 'disabled'

        self._cone_detections_pub = self.create_publisher(
            ConeDetectionArray,
            self._config.cone_detections_topic,
            10,
        )

        self.create_subscription(Image, self._config.left_image_topic, self._left_image_cb, 10)
        self.create_subscription(CameraInfo, self._config.left_camera_info_topic, self._left_info_cb, 10)
        if self._config.stereo_enabled:
            self.create_subscription(Image, self._config.right_image_topic, self._right_image_cb, 10)
            self.create_subscription(CameraInfo, self._config.right_camera_info_topic, self._right_info_cb, 10)

        self._worker_thread: Optional[threading.Thread] = None
        if self._config.stereo_enabled:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

        self.get_logger().info(
            'perception_node ready: '
            f'stereo_enabled={self._config.stereo_enabled} '
            f'left={self._config.left_image_topic} right={self._config.right_image_topic} '
            f'yolo_enabled={self._config.yolo_enabled} yolo_backend={self._yolo_backend} '
            f'cone_detections_topic={self._config.cone_detections_topic} '
            f'cone_detections_frame={self._config.cone_detections_frame}'
        )

    def _left_info_cb(self, msg: CameraInfo) -> None:
        self._left_info = msg

    def _right_info_cb(self, msg: CameraInfo) -> None:
        self._right_info = msg

    def _left_image_cb(self, msg: Image) -> None:
        if self._config.stereo_enabled:
            self._enqueue_frame(msg, side='left')
            return
        self._process_monocular_frame(msg)

    def _right_image_cb(self, msg: Image) -> None:
        self._enqueue_frame(msg, side='right')

    def _enqueue_frame(self, msg: Image, side: str) -> None:
        pair_time_sec = self._stamp_to_sec(msg)
        if pair_time_sec <= 0.0:
            pair_time_sec = time.monotonic()
        packet = FramePacket(msg=msg, pair_time_sec=pair_time_sec)

        with self._queue_cv:
            queue = self._left_queue if side == 'left' else self._right_queue
            queue.append(packet)
            while len(queue) > self._config.queue_size:
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

            if best_dt is not None and best_dt <= self._config.max_time_diff_sec:
                left_packet = self._left_queue.popleft()
                right_packet = self._right_queue[best_idx]
                del self._right_queue[best_idx]
                return left_packet, right_packet

            left_t = left_head.pair_time_sec
            right_oldest_t = self._right_queue[0].pair_time_sec
            right_newest_t = self._right_queue[-1].pair_time_sec
            if right_newest_t < left_t - self._config.max_time_diff_sec:
                self._right_queue.popleft()
                continue
            if left_t < right_oldest_t - self._config.max_time_diff_sec:
                self._left_queue.popleft()
                continue
            if left_t <= right_oldest_t:
                self._left_queue.popleft()
            else:
                self._right_queue.popleft()
        return None

    def _worker_loop(self) -> None:
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

            yolo_detections, _infer_ms = run_yolo(self._yolo_detector, yolo_input_image, self.get_logger())
            apply_depth_map_to_detections(output.depth, yolo_detections)
            self._publish_cone_detections(
                yolo_detections=yolo_detections,
                left_info=left_info,
                eval_header=eval_header,
            )
            self._publish_debug_image(eval_header, stereo_debug_image, yolo_detections)

    def _process_monocular_frame(self, msg: Image) -> None:
        left_bgr = StereoPipeline._decode_to_bgr(msg)
        if left_bgr is None:
            return

        eval_header = Header()
        eval_header.stamp = msg.header.stamp
        eval_header.frame_id = msg.header.frame_id

        yolo_detections, _infer_ms = run_yolo(self._yolo_detector, left_bgr, self.get_logger())
        _fx, fy, _cx, _cy = camera_intrinsics(self._left_info) if self._left_info is not None else (0.0, 0.0, 0.0, 0.0)
        apply_monocular_depth_to_detections(
            yolo_detections,
            fy_px=fy,
            cone_height_m=self._config.monocular_cone_height_m,
            big_cone_height_m=self._config.monocular_big_cone_height_m,
            bbox_height_offset_px=self._config.monocular_bbox_height_offset_px,
            normalize_detection_color=normalize_detection_color,
        )
        self._publish_cone_detections(
            yolo_detections=yolo_detections,
            left_info=self._left_info,
            eval_header=eval_header,
        )
        self._publish_debug_image(eval_header, left_bgr, yolo_detections)

    def _publish_debug_image(self, header: Header, source_image: np.ndarray | None, detections: list[dict]) -> None:
        if not self._camera_debug.should_publish():
            return
        debug_image = build_camera_debug_image(
            source_image,
            detections,
            scale=_DEBUG_IMAGE_SCALE,
            mono=_DEBUG_IMAGE_MONO,
        )
        if debug_image is None:
            return
        encoding = 'mono8' if _DEBUG_IMAGE_MONO else 'bgr8'
        self._camera_debug.publish_image(header, debug_image, encoding)

    def _publish_cone_detections(
        self,
        *,
        yolo_detections: list[dict],
        left_info: Optional[CameraInfo],
        eval_header: Header,
    ) -> None:
        msg = ConeDetectionArray()
        msg.header.stamp = eval_header.stamp
        msg.header.frame_id = self._config.cone_detections_frame

        if not yolo_detections or left_info is None:
            self._cone_detections_pub.publish(msg)
            return

        fx, fy, cx, cy = camera_intrinsics(left_info)
        if fx <= 0.0 or fy <= 0.0:
            self._cone_detections_pub.publish(msg)
            return

        camera_frame = str(left_info.header.frame_id).strip() or str(eval_header.frame_id).strip()
        if not camera_frame:
            self._cone_detections_pub.publish(msg)
            return

        cam_to_output = None
        output_frame = self._config.cone_detections_frame
        transform_source_frame = camera_frame
        if camera_frame != self._config.cone_detections_frame:
            target_candidates = [self._config.cone_detections_frame]
            namespaced_frame = resolve_namespaced_output_frame(
                camera_frame=camera_frame,
                requested_frame=self._config.cone_detections_frame,
            )
            if namespaced_frame and namespaced_frame not in target_candidates:
                target_candidates.append(namespaced_frame)

            source_candidates = cone_output_source_frame_candidates(
                source_frame=camera_frame,
                requested_output_frame=self._config.cone_detections_frame,
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
                    f'cone detections transform unavailable {camera_frame}->{self._config.cone_detections_frame}; '
                    f'publishing in source frame "{output_frame}"',
                )

        reconstruction_model = projection_model_for_frame(transform_source_frame)
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

            cam_point = reconstruct_cam_point_from_axis(
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
                x_out, y_out, z_out = transform_point(cam_to_output, x_out, y_out, z_out)

            confidence = det.get('confidence')
            cone_candidates.append(
                (
                    float(x_out),
                    float(y_out),
                    float(z_out),
                    normalize_detection_color(str(det.get('label', ''))),
                    float(max(0.0, min(1.0, float(confidence)))) if confidence is not None else 0.0,
                )
            )

        for x_out, y_out, z_out, color, confidence in deduplicate_cone_candidates(
            cone_candidates,
            self._config.cone_dedup_radius_m,
        ):
            cone = ConeDetection()
            cone.color = color
            cone.confidence = confidence
            cone.position.x = x_out
            cone.position.y = y_out
            cone.position.z = z_out
            msg.cones.append(cone)

        self._cone_detections_pub.publish(msg)

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp):
        cache_key = (target_frame, source_frame)
        query_time = Time.from_msg(stamp)
        timeout = Duration(seconds=float(self._config.cone_eval_tf_timeout_sec))

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

    def _warn_throttled(self, key: str, message: str) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get(key, -1.0)
        if (now_sec - last_sec) >= 1.0:
            self.get_logger().warn(message)
            self._last_throttled_log_sec[key] = now_sec

    @staticmethod
    def _stamp_to_sec(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

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


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
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
