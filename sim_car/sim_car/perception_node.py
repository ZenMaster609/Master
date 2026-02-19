"""ROS2 node that runs stereo perception and evaluation in one process."""

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Deque, Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from eufs_msgs.msg import ConeArrayWithCovariance
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Header, Int32, String
from tf2_ros import Buffer, TransformException, TransformListener

from sim_car.perception import (
    CameraDebugPublisher,
    PerfLogger,
    StereoEvaluator,
    StereoPipeline,
    StereoPipelineConfig,
)


@dataclass
class FramePacket:
    """Queued frame packet used by pairer worker."""

    msg: Image
    pair_time_sec: float


@dataclass
class ConePacket:
    """Queued cone packet used for time matching with stereo frames."""

    msg: ConeArrayWithCovariance
    stamp_sec: float


@dataclass
class ConeDepthMetrics:
    """Latest cone-vs-depth validation metrics."""

    pairs: int = 0
    axis_mae_m: Optional[float] = None
    axis_rmse_m: Optional[float] = None
    axis_bias_m: Optional[float] = None
    range_mae_m: Optional[float] = None
    range_rmse_m: Optional[float] = None
    sync_dt_ms: Optional[float] = None


@dataclass(frozen=True)
class TrackConeRef:
    """Track cone reference used to assign stable per-cone IDs."""

    cone_id: str
    color: str
    x: float
    y: float


@dataclass
class ConeRunningStats:
    """Running error statistics for one cone ID."""

    color: str
    samples: int = 0
    abs_err_sum: float = 0.0
    sq_err_sum: float = 0.0
    cam_depth_sum: float = 0.0
    gt_depth_sum: float = 0.0

    def add(self, err_m: float, cam_depth_m: float, gt_depth_m: float):
        self.samples += 1
        self.abs_err_sum += abs(err_m)
        self.sq_err_sum += err_m * err_m
        self.cam_depth_sum += cam_depth_m
        self.gt_depth_sum += gt_depth_m

    def mae(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return self.abs_err_sum / float(self.samples)

    def rmse(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return math.sqrt(self.sq_err_sum / float(self.samples))

    def dcam(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return self.cam_depth_sum / float(self.samples)

    def dgt(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return self.gt_depth_sum / float(self.samples)


class PerceptionNode(Node):
    """Subscribes stereo topics, computes disparity/depth, and publishes eval metrics."""

    CAMERA_DEBUG_EVERY_N = 30

    def __init__(self):
        super().__init__('perception_node')

        self._declare_parameters()
        self._read_parameters()

        self._left_info: Optional[CameraInfo] = None
        self._right_info: Optional[CameraInfo] = None

        self._left_queue: Deque[FramePacket] = deque()
        self._right_queue: Deque[FramePacket] = deque()
        self._queue_lock = threading.Lock()
        self._queue_cv = threading.Condition(self._queue_lock)
        self._running = True

        self._cone_queue: Deque[ConePacket] = deque()
        self._cone_lock = threading.Lock()
        self._latest_cone_metrics = ConeDepthMetrics()
        self._track_lock = threading.Lock()
        self._track_cones = []
        self._track_frame_id = 'map'
        self._odom_lock = threading.Lock()
        self._latest_odom = None
        self._cone_stats_lock = threading.Lock()
        self._cone_stats = {}

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

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
        self._evaluator = StereoEvaluator(
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            disparity_valid_threshold=self.disparity_valid_threshold,
            orb_features=self.eval_orb_features,
            max_matches=self.eval_max_matches,
            match_ratio_test=self.eval_match_ratio_test,
        )
        self._perf = PerfLogger(self, eval_topic_prefix=self.eval_topic_prefix)
        sanitized_disparities = max(16, (self.num_disparities // 16) * 16)
        self._camera_debug = CameraDebugPublisher(
            node=self,
            mode=self.camera_debug,
            topic=self.camera_debug_topic,
            publish_every_n=self.CAMERA_DEBUG_EVERY_N,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            max_disparity=float(sanitized_disparities),
            disparity_valid_threshold=self.disparity_valid_threshold,
        )

        prefix = self.eval_topic_prefix.rstrip('/')
        self._cone_pairs_pub = self.create_publisher(Int32, f'{prefix}/cone_depth_pairs', 10)
        self._cone_axis_mae_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_axis_mae_m', 10)
        self._cone_axis_rmse_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_axis_rmse_m', 10)
        self._cone_axis_bias_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_axis_bias_m', 10)
        self._cone_range_mae_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_range_mae_m', 10)
        self._cone_range_rmse_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_range_rmse_m', 10)
        self._cone_sync_dt_pub = self.create_publisher(Float32, f'{prefix}/cone_depth_sync_dt_ms', 10)
        self._cone_per_cone_pub = self.create_publisher(String, f'{prefix}/cone_depth_per_cone', 10)

        self.create_subscription(Image, self.left_image_topic, self._left_image_cb, 10)
        self.create_subscription(Image, self.right_image_topic, self._right_image_cb, 10)
        self.create_subscription(CameraInfo, self.left_camera_info_topic, self._left_info_cb, 10)
        self.create_subscription(CameraInfo, self.right_camera_info_topic, self._right_info_cb, 10)
        self.create_subscription(ConeArrayWithCovariance, self.ground_truth_cones_topic, self._cone_gt_cb, 10)
        self.create_subscription(ConeArrayWithCovariance, self.ground_truth_track_topic, self._track_gt_cb, 10)
        self.create_subscription(Odometry, self.cone_eval_odom_topic, self._odom_cb, 10)

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        if self.perf_log_hz > 0.0:
            self.create_timer(1.0 / self.perf_log_hz, self._perf_timer_cb)

        self.get_logger().info(
            'perception_node ready: '
            f'left={self.left_image_topic} right={self.right_image_topic} '
            f'eval_prefix={self.eval_topic_prefix} perf_log_hz={self.perf_log_hz:.2f} '
            f'camera_debug={self.camera_debug} cones={self.ground_truth_cones_topic} '
            f'track={self.ground_truth_track_topic} odom={self.cone_eval_odom_topic}'
        )

    def _declare_parameters(self):
        self.declare_parameter('left_image_topic', '/sim/raw/stereo/left/image_raw')
        self.declare_parameter('right_image_topic', '/sim/raw/stereo/right/image_raw')
        self.declare_parameter('left_camera_info_topic', '/sim/raw/stereo/left/camera_info')
        self.declare_parameter('right_camera_info_topic', '/sim/raw/stereo/right/camera_info')

        self.declare_parameter('calibration_file', '')
        self.declare_parameter('max_time_diff_sec', 0.08)
        self.declare_parameter('queue_size', 30)
        self.declare_parameter('perf_log_hz', 1.0)
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
        self.declare_parameter('eval_orb_features', 700)
        self.declare_parameter('eval_max_matches', 200)
        self.declare_parameter('eval_match_ratio_test', 0.75)
        self.declare_parameter('camera_debug', 'none')
        self.declare_parameter('camera_debug_topic', '/sim/raw/stereo/camera_debug')

        self.declare_parameter('ground_truth_cones_topic', '/ground_truth/cones')
        self.declare_parameter('ground_truth_track_topic', '/ground_truth/track')
        self.declare_parameter('cone_eval_sync_slop_sec', 0.10)
        self.declare_parameter('cone_eval_target_frame', '')
        self.declare_parameter('cone_eval_fallback_frame', 'stereo_left_link')
        self.declare_parameter('cone_eval_projection_model', 'auto')
        self.declare_parameter('cone_eval_pixel_radius', 2)
        self.declare_parameter('cone_eval_include_unknown', False)
        self.declare_parameter('cone_eval_track_match_threshold_m', 0.75)
        self.declare_parameter('cone_eval_per_cone_max_rows', 120)
        self.declare_parameter('cone_eval_odom_topic', '/sim/odom')

    def _read_parameters(self):
        self.left_image_topic = str(self.get_parameter('left_image_topic').value)
        self.right_image_topic = str(self.get_parameter('right_image_topic').value)
        self.left_camera_info_topic = str(self.get_parameter('left_camera_info_topic').value)
        self.right_camera_info_topic = str(self.get_parameter('right_camera_info_topic').value)

        self.calibration_file = str(self.get_parameter('calibration_file').value)
        self.max_time_diff_sec = max(0.0, float(self.get_parameter('max_time_diff_sec').value))
        self.queue_size = max(5, int(self.get_parameter('queue_size').value))
        self.perf_log_hz = max(0.0, float(self.get_parameter('perf_log_hz').value))
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
        self.eval_orb_features = int(self.get_parameter('eval_orb_features').value)
        self.eval_max_matches = int(self.get_parameter('eval_max_matches').value)
        self.eval_match_ratio_test = float(self.get_parameter('eval_match_ratio_test').value)
        self.camera_debug = self._sanitize_camera_debug(self.get_parameter('camera_debug').value)
        self.camera_debug_topic = str(self.get_parameter('camera_debug_topic').value)

        self.ground_truth_cones_topic = str(self.get_parameter('ground_truth_cones_topic').value)
        self.ground_truth_track_topic = str(self.get_parameter('ground_truth_track_topic').value)
        self.cone_eval_sync_slop_sec = max(0.01, float(self.get_parameter('cone_eval_sync_slop_sec').value))
        self.cone_eval_target_frame = str(self.get_parameter('cone_eval_target_frame').value)
        self.cone_eval_fallback_frame = str(self.get_parameter('cone_eval_fallback_frame').value)
        self.cone_eval_projection_model = self._sanitize_projection_model(
            self.get_parameter('cone_eval_projection_model').value
        )
        self.cone_eval_pixel_radius = max(0, int(self.get_parameter('cone_eval_pixel_radius').value))
        self.cone_eval_include_unknown = bool(self.get_parameter('cone_eval_include_unknown').value)
        self.cone_eval_track_match_threshold_m = max(
            0.1, float(self.get_parameter('cone_eval_track_match_threshold_m').value)
        )
        self.cone_eval_per_cone_max_rows = max(1, int(self.get_parameter('cone_eval_per_cone_max_rows').value))
        self.cone_eval_odom_topic = str(self.get_parameter('cone_eval_odom_topic').value)

    def _left_info_cb(self, msg: CameraInfo):
        self._left_info = msg

    def _right_info_cb(self, msg: CameraInfo):
        self._right_info = msg

    def _left_image_cb(self, msg: Image):
        self._enqueue_frame(msg, side='left')

    def _right_image_cb(self, msg: Image):
        self._enqueue_frame(msg, side='right')

    def _cone_gt_cb(self, msg: ConeArrayWithCovariance):
        stamp_sec = self._stamp_msg_to_sec(msg.header.stamp)
        with self._cone_lock:
            self._cone_queue.append(ConePacket(msg=msg, stamp_sec=stamp_sec))
            while len(self._cone_queue) > 200:
                self._cone_queue.popleft()

    def _track_gt_cb(self, msg: ConeArrayWithCovariance):
        refs = []
        refs.extend(self._build_track_refs(msg.blue_cones, 'blue'))
        refs.extend(self._build_track_refs(msg.yellow_cones, 'yellow'))
        refs.extend(self._build_track_refs(msg.orange_cones, 'orange'))
        refs.extend(self._build_track_refs(msg.big_orange_cones, 'big_orange'))
        refs.extend(self._build_track_refs(msg.unknown_color_cones, 'unknown'))
        frame_id = str(msg.header.frame_id).strip() or 'map'
        with self._track_lock:
            self._track_cones = refs
            self._track_frame_id = frame_id

    def _odom_cb(self, msg: Odometry):
        with self._odom_lock:
            self._latest_odom = msg

    def _enqueue_frame(self, msg: Image, side: str):
        pair_time_sec = self._stamp_to_sec(msg)
        if pair_time_sec <= 0.0:
            pair_time_sec = time.monotonic()

        packet = FramePacket(msg=msg, pair_time_sec=pair_time_sec)
        with self._queue_cv:
            queue = self._left_queue if side == 'left' else self._right_queue
            queue.append(packet)
            self._perf.count_incoming(side=side, queue_len=len(queue))

            while len(queue) > self.queue_size:
                queue.popleft()
                self._perf.count_drop(side=side)

            self._queue_cv.notify()

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
            output = self._pipeline.process(
                left_msg=left_packet.msg,
                right_msg=right_packet.msg,
                left_info=self._left_info,
                right_info=self._right_info,
            )
            if output is None:
                continue

            self._evaluator.update(
                left_rect_gray=output.left_rect,
                right_rect_gray=output.right_rect,
                disparity=output.disparity,
                depth=output.depth,
            )
            self._perf.record_processed(timings_ms=output.timings_ms, backend=output.backend)

            eval_header = self._common_header(left_packet.msg.header, right_packet.msg.header)
            self._latest_cone_metrics, cone_overlays = self._evaluate_cone_depth(
                depth=output.depth,
                left_info=self._left_info,
                eval_header=eval_header,
            )

            self._camera_debug.maybe_publish(
                header=eval_header,
                disparity=output.disparity,
                depth=output.depth,
                left_rect=output.left_rect,
                cone_overlays=cone_overlays,
            )

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
                self._perf.count_pair(best_dt)
                return left_packet, right_packet

            left_t = left_head.pair_time_sec
            right_oldest_t = self._right_queue[0].pair_time_sec
            right_newest_t = self._right_queue[-1].pair_time_sec

            if right_newest_t < left_t - self.max_time_diff_sec:
                self._right_queue.popleft()
                self._perf.count_drop(side='right')
                continue
            if left_t < right_oldest_t - self.max_time_diff_sec:
                self._left_queue.popleft()
                self._perf.count_drop(side='left')
                continue

            if left_t <= right_oldest_t:
                self._left_queue.popleft()
                self._perf.count_drop(side='left')
            else:
                self._right_queue.popleft()
                self._perf.count_drop(side='right')
        return None

    def _evaluate_cone_depth(
        self,
        depth: np.ndarray,
        left_info: Optional[CameraInfo],
        eval_header: Header,
    ) -> tuple[ConeDepthMetrics, list[dict]]:
        metrics = ConeDepthMetrics()
        overlays = []
        if left_info is None:
            self._publish_cone_metrics(metrics)
            return metrics, overlays

        height, width = depth.shape[:2]
        fx, fy, cx, cy = self._camera_intrinsics(left_info)
        if fx <= 0.0 or fy <= 0.0:
            self._publish_cone_metrics(metrics)
            return metrics, overlays

        frame_stamp_sec = self._stamp_msg_to_sec(eval_header.stamp)
        cone_packet = self._get_nearest_cone_packet(frame_stamp_sec)
        if cone_packet is None:
            self._publish_cone_metrics(metrics)
            return metrics, overlays

        metrics.sync_dt_ms = abs(frame_stamp_sec - cone_packet.stamp_sec) * 1000.0
        if metrics.sync_dt_ms > self.cone_eval_sync_slop_sec * 1000.0:
            self._publish_cone_metrics(metrics)
            return metrics, overlays

        src_frame = cone_packet.msg.header.frame_id
        if not src_frame:
            self._publish_cone_metrics(metrics)
            return metrics, overlays

        transform_bundle = self._resolve_transform_and_projection(
            source_frame=src_frame,
            left_info_frame=left_info.header.frame_id,
            stamp=eval_header.stamp,
        )
        if transform_bundle is None:
            self._publish_cone_metrics(metrics)
            return metrics, overlays
        transform, projection_model = transform_bundle
        with self._track_lock:
            track_frame_id = self._track_frame_id
        track_transform = None
        track_coord_mode = 'none'
        if track_frame_id == src_frame:
            track_coord_mode = 'identity'
        else:
            track_transform = self._lookup_transform(
                target_frame=track_frame_id,
                source_frame=src_frame,
                stamp=eval_header.stamp,
            )
            if track_transform is not None:
                track_coord_mode = 'transform'
            elif (
                track_frame_id in {'map', 'odom'}
                and self._is_base_footprint_frame(src_frame)
            ):
                track_coord_mode = 'odom_fallback'

        errors_axis = []
        errors_range = []

        for cone_color, cone in self._iter_cones(cone_packet.msg):
            x_cam, y_cam, z_cam = self._transform_point(transform, cone.point.x, cone.point.y, cone.point.z)
            projection = self._project_to_pixel(
                x_cam=x_cam,
                y_cam=y_cam,
                z_cam=z_cam,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                model=projection_model,
            )
            if projection is None:
                continue
            u, v, gt_axis = projection
            if u < 0.0 or v < 0.0 or u >= float(width) or v >= float(height):
                continue

            est_axis = self._sample_depth(depth, u, v, self.cone_eval_pixel_radius)
            if not np.isfinite(est_axis):
                continue

            err_axis = float(est_axis - gt_axis)
            errors_axis.append(err_axis)
            cone_id = None
            if track_coord_mode == 'transform':
                x_map, y_map, _ = self._transform_point(
                    track_transform,
                    cone.point.x,
                    cone.point.y,
                    cone.point.z,
                )
                cone_id = self._match_cone_id(
                    x_map=x_map,
                    y_map=y_map,
                    color=cone_color,
                )
            elif track_coord_mode == 'identity':
                cone_id = self._match_cone_id(
                    x_map=float(cone.point.x),
                    y_map=float(cone.point.y),
                    color=cone_color,
                )
            elif track_coord_mode == 'odom_fallback':
                fallback_point = self._point_to_track_with_odom_fallback(
                    x=float(cone.point.x),
                    y=float(cone.point.y),
                    z=float(cone.point.z),
                )
                if fallback_point is not None:
                    x_map, y_map, _ = fallback_point
                    cone_id = self._match_cone_id(
                        x_map=x_map,
                        y_map=y_map,
                        color=cone_color,
                    )
            if cone_id is not None:
                self._update_cone_stats(
                    cone_id=cone_id,
                    color=cone_color,
                    axis_err_m=err_axis,
                    dcam_m=float(est_axis),
                    dgt_m=float(gt_axis),
                )
                samples, mae, rmse, dcam, dgt = self._get_cone_stat_values(cone_id)
                if mae is not None and rmse is not None and dcam is not None and dgt is not None:
                    overlays.append(
                        {
                            'u': float(u),
                            'v': float(v),
                            'color': cone_color,
                            'sort_depth': float(dgt),
                            'label': (
                                f'{cone_id} '
                                f'MAE:{mae:.2f}m '
                                f'RMSE:{rmse:.2f}m '
                                f'\n'
                                f'dcam:{dcam:.2f}m '
                                f'dgt:{dgt:.2f}m'
                            ),
                        }
                    )

            gt_range = math.sqrt((x_cam * x_cam) + (y_cam * y_cam) + (z_cam * z_cam))
            ray_scale = math.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
            est_range = float(est_axis) * ray_scale
            errors_range.append(est_range - gt_range)

        metrics.pairs = len(errors_axis)
        if metrics.pairs > 0:
            axis_arr = np.asarray(errors_axis, dtype=np.float32)
            range_arr = np.asarray(errors_range, dtype=np.float32)
            metrics.axis_mae_m = float(np.mean(np.abs(axis_arr)))
            metrics.axis_rmse_m = float(np.sqrt(np.mean(np.square(axis_arr))))
            metrics.axis_bias_m = float(np.mean(axis_arr))
            metrics.range_mae_m = float(np.mean(np.abs(range_arr)))
            metrics.range_rmse_m = float(np.sqrt(np.mean(np.square(range_arr))))

        overlays = self._assign_overlay_placements(overlays, cx)
        self._publish_cone_metrics(metrics)
        return metrics, overlays

    @staticmethod
    def _assign_overlay_placements(overlays: list[dict], image_cx: float) -> list[dict]:
        if not overlays:
            return overlays

        placement_cycle = ('bottom', 'left', 'right', 'top')
        left = []
        right = []
        for overlay in overlays:
            depth = float(overlay.get('sort_depth', float('inf')))
            u = float(overlay.get('u', 0.0))
            if u < image_cx:
                left.append((depth, overlay))
            else:
                right.append((depth, overlay))

        left.sort(key=lambda item: item[0])
        right.sort(key=lambda item: item[0])

        for idx in range(max(len(left), len(right))):
            placement = placement_cycle[idx % len(placement_cycle)]
            if idx < len(left):
                left[idx][1]['placement'] = PerceptionNode._side_safe_placement(
                    placement=placement,
                    side='left',
                )
            if idx < len(right):
                right[idx][1]['placement'] = PerceptionNode._side_safe_placement(
                    placement=placement,
                    side='right',
                )

        for overlay in overlays:
            overlay.pop('sort_depth', None)
            if 'placement' not in overlay:
                overlay['placement'] = 'right'
        return overlays

    @staticmethod
    def _side_safe_placement(placement: str, side: str) -> str:
        p = str(placement).strip().lower()
        s = str(side).strip().lower()
        if s == 'left' and p == 'right':
            return 'top'
        if s == 'right' and p == 'left':
            return 'top'
        return p

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp):
        query_time = Time.from_msg(stamp)
        try:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                query_time,
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            try:
                return self._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
            except TransformException as exc:
                self.get_logger().debug(
                    f'cone depth eval transform failed {source_frame}->{target_frame}: {exc}'
                )
                return None

    def _resolve_transform_and_projection(self, source_frame: str, left_info_frame: str, stamp):
        explicit_target = self.cone_eval_target_frame.strip()
        fallback_target = self.cone_eval_fallback_frame.strip()

        candidates = []
        if explicit_target:
            candidates.append(explicit_target)
        if left_info_frame:
            candidates.append(left_info_frame)
        if fallback_target:
            candidates.append(fallback_target)

        seen = set()
        unique_candidates = []
        for frame in candidates:
            if frame in seen:
                continue
            seen.add(frame)
            unique_candidates.append(frame)

        for frame in unique_candidates:
            transform = self._lookup_transform(
                target_frame=frame,
                source_frame=source_frame,
                stamp=stamp,
            )
            if transform is None:
                continue

            model = self.cone_eval_projection_model
            if model == 'auto':
                frame_lower = frame.lower()
                if 'optical' in frame_lower or frame_lower.endswith('_camera'):
                    model = 'optical_z'
                elif frame_lower.endswith('_link'):
                    model = 'forward_x'
                else:
                    model = 'optical_z'
            return transform, model
        return None

    @staticmethod
    def _project_to_pixel(
        x_cam: float,
        y_cam: float,
        z_cam: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        model: str,
    ):
        if model == 'forward_x':
            if x_cam <= 0.0:
                return None
            u = (fx * (-y_cam / x_cam)) + cx
            v = (fy * (-z_cam / x_cam)) + cy
            return float(u), float(v), float(x_cam)

        # optical_z
        if z_cam <= 0.0:
            return None
        u = (fx * (x_cam / z_cam)) + cx
        v = (fy * (y_cam / z_cam)) + cy
        return float(u), float(v), float(z_cam)

    @staticmethod
    def _build_track_refs(cones, color: str):
        refs = []
        ordered = sorted(cones, key=lambda c: (float(c.point.x), float(c.point.y)))
        for idx, cone in enumerate(ordered):
            refs.append(
                TrackConeRef(
                    cone_id=f'{color}_{idx:03d}',
                    color=color,
                    x=float(cone.point.x),
                    y=float(cone.point.y),
                )
            )
        return refs

    def _match_cone_id(self, x_map: float, y_map: float, color: str) -> Optional[str]:
        with self._track_lock:
            track_refs = self._track_cones
        if not track_refs:
            return None

        best_id = None
        best_dist_sq = None
        threshold_sq = self.cone_eval_track_match_threshold_m * self.cone_eval_track_match_threshold_m
        for ref in track_refs:
            if color != 'unknown' and ref.color != color:
                continue
            dx = ref.x - x_map
            dy = ref.y - y_map
            dist_sq = (dx * dx) + (dy * dy)
            if dist_sq > threshold_sq:
                continue
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_id = ref.cone_id
        return best_id

    def _update_cone_stats(
        self,
        cone_id: str,
        color: str,
        axis_err_m: float,
        dcam_m: float,
        dgt_m: float,
    ):
        with self._cone_stats_lock:
            stats = self._cone_stats.get(cone_id)
            if stats is None:
                stats = ConeRunningStats(color=color)
                self._cone_stats[cone_id] = stats
            stats.add(
                err_m=axis_err_m,
                cam_depth_m=dcam_m,
                gt_depth_m=dgt_m,
            )

    def _get_cone_stat_values(self, cone_id: str):
        with self._cone_stats_lock:
            stats = self._cone_stats.get(cone_id)
            if stats is None:
                return 0, None, None, None, None
            return stats.samples, stats.mae(), stats.rmse(), stats.dcam(), stats.dgt()

    @staticmethod
    def _is_base_footprint_frame(frame_id: str) -> bool:
        frame = frame_id.strip()
        return frame == 'base_footprint' or frame.endswith('/base_footprint')

    def _point_to_track_with_odom_fallback(self, x: float, y: float, z: float):
        with self._odom_lock:
            odom = self._latest_odom
        if odom is None:
            return None

        pose = odom.pose.pose
        return self._transform_point_from_pose(
            pose,
            x,
            y,
            z,
        )

    def _per_cone_table_text(self) -> str:
        with self._cone_stats_lock:
            items = sorted(
                (
                    (
                        cone_id,
                        stats.color,
                        stats.samples,
                        stats.mae(),
                        stats.rmse(),
                        stats.dcam(),
                        stats.dgt(),
                    )
                    for cone_id, stats in self._cone_stats.items()
                ),
                key=lambda item: (-item[2], item[0]),
            )

        if not items:
            return 'no per-cone depth samples yet'

        lines = ['cone_id,color,samples,axis_mae_m,axis_rmse_m,dcam,dgt']
        for idx, (cone_id, color, samples, mae, rmse, dcam, dgt) in enumerate(items):
            if idx >= self.cone_eval_per_cone_max_rows:
                remaining = len(items) - idx
                lines.append(f'... ({remaining} more cones)')
                break
            mae_str = 'n/a' if mae is None else f'{mae:.4f}'
            rmse_str = 'n/a' if rmse is None else f'{rmse:.4f}'
            dcam_str = 'n/a' if dcam is None else f'{dcam:.4f}'
            dgt_str = 'n/a' if dgt is None else f'{dgt:.4f}'
            lines.append(f'{cone_id},{color},{samples},{mae_str},{rmse_str},{dcam_str},{dgt_str}')
        return '\n'.join(lines)

    def _get_nearest_cone_packet(self, target_stamp_sec: float) -> Optional[ConePacket]:
        with self._cone_lock:
            if not self._cone_queue:
                return None

            while len(self._cone_queue) > 2:
                if self._cone_queue[1].stamp_sec < target_stamp_sec - self.cone_eval_sync_slop_sec:
                    self._cone_queue.popleft()
                else:
                    break

            best_packet = None
            best_dt = None
            for packet in self._cone_queue:
                dt = abs(packet.stamp_sec - target_stamp_sec)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_packet = packet
            return best_packet

    def _publish_cone_metrics(self, metrics: ConeDepthMetrics):
        self._cone_pairs_pub.publish(Int32(data=int(metrics.pairs)))
        if metrics.axis_mae_m is not None:
            self._cone_axis_mae_pub.publish(Float32(data=float(metrics.axis_mae_m)))
        if metrics.axis_rmse_m is not None:
            self._cone_axis_rmse_pub.publish(Float32(data=float(metrics.axis_rmse_m)))
        if metrics.axis_bias_m is not None:
            self._cone_axis_bias_pub.publish(Float32(data=float(metrics.axis_bias_m)))
        if metrics.range_mae_m is not None:
            self._cone_range_mae_pub.publish(Float32(data=float(metrics.range_mae_m)))
        if metrics.range_rmse_m is not None:
            self._cone_range_rmse_pub.publish(Float32(data=float(metrics.range_rmse_m)))
        if metrics.sync_dt_ms is not None:
            self._cone_sync_dt_pub.publish(Float32(data=float(metrics.sync_dt_ms)))

    @staticmethod
    def _camera_intrinsics(camera_info: CameraInfo):
        fx = float(camera_info.k[0]) if len(camera_info.k) >= 1 else 0.0
        fy = float(camera_info.k[4]) if len(camera_info.k) >= 5 else 0.0
        cx = float(camera_info.k[2]) if len(camera_info.k) >= 3 else 0.0
        cy = float(camera_info.k[5]) if len(camera_info.k) >= 6 else 0.0
        return fx, fy, cx, cy

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
    def _transform_point_from_pose(pose, x: float, y: float, z: float):
        q = pose.orientation

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

        t = pose.position
        tx = float(t.x)
        ty = float(t.y)
        tz = float(t.z)

        px = (r00 * x) + (r01 * y) + (r02 * z) + tx
        py = (r10 * x) + (r11 * y) + (r12 * z) + ty
        pz = (r20 * x) + (r21 * y) + (r22 * z) + tz
        return px, py, pz

    def _iter_cones(self, msg: ConeArrayWithCovariance):
        for cone in msg.blue_cones:
            yield 'blue', cone
        for cone in msg.yellow_cones:
            yield 'yellow', cone
        for cone in msg.orange_cones:
            yield 'orange', cone
        for cone in msg.big_orange_cones:
            yield 'big_orange', cone
        if self.cone_eval_include_unknown:
            for cone in msg.unknown_color_cones:
                yield 'unknown', cone

    def _perf_timer_cb(self):
        self._perf.log_and_publish(self._evaluator.snapshot())
        per_cone_table = self._per_cone_table_text()
        self._cone_per_cone_pub.publish(String(data=per_cone_table))
        with self._cone_stats_lock:
            tracked_cones = len(self._cone_stats)
        m = self._latest_cone_metrics
        self.get_logger().info(
            'cone depth eval '
            f'pairs={m.pairs} '
            f'axis_mae={self._fmt_opt(m.axis_mae_m)} '
            f'axis_rmse={self._fmt_opt(m.axis_rmse_m)} '
            f'axis_bias={self._fmt_opt(m.axis_bias_m)} '
            f'range_mae={self._fmt_opt(m.range_mae_m)} '
            f'range_rmse={self._fmt_opt(m.range_rmse_m)} '
            f'sync_dt_ms={self._fmt_opt(m.sync_dt_ms)} '
            f'per_cone_tracked={tracked_cones}'
        )

    @staticmethod
    def _fmt_opt(value: Optional[float]) -> str:
        if value is None:
            return 'n/a'
        return f'{value:.4f}'

    @staticmethod
    def _stamp_to_sec(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _stamp_msg_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _sanitize_camera_debug(value) -> str:
        mode = str(value).strip().lower()
        if mode == 'rect_left':
            mode = 'left_rect'
        if mode in {'disparity', 'depth', 'left_rect', 'none'}:
            return mode
        return 'none'

    @staticmethod
    def _sanitize_projection_model(value) -> str:
        mode = str(value).strip().lower()
        if mode in {'auto', 'optical_z', 'forward_x'}:
            return mode
        return 'auto'

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
        if self._worker_thread.is_alive():
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
