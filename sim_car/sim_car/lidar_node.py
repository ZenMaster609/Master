"""ROS2 node for LiDAR cone detection and per-cone RMSE evaluation."""

from collections import deque
import math
import threading
import time
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from eufs_msgs.msg import ConeArrayWithCovariance
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32, String
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray

from sim_car.lidar.clustering import detect_cone_candidates, points_from_ranges
from sim_car.perception.range_rmse_analyzer import RangeRMSEAnalyzer
from sim_car.perception.range_rmse_live_plot import RangeRMSELivePlot


@dataclass
class ConePacket:
    """Timestamped visible cone message."""

    msg: ConeArrayWithCovariance
    stamp_sec: float


@dataclass
class OdomPacket:
    """Timestamped odometry sample."""

    msg: Odometry
    stamp_sec: float


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
    sq_err_x_sum: float = 0.0
    sq_err_y_sum: float = 0.0
    pred_range_sum: float = 0.0
    gt_range_sum: float = 0.0

    def add(self, err_m: float, err_x_m: float, err_y_m: float, pred_range_m: float, gt_range_m: float) -> None:
        self.samples += 1
        self.abs_err_sum += abs(err_m)
        self.sq_err_sum += err_m * err_m
        self.sq_err_x_sum += err_x_m * err_x_m
        self.sq_err_y_sum += err_y_m * err_y_m
        self.pred_range_sum += pred_range_m
        self.gt_range_sum += gt_range_m

    def mae(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return self.abs_err_sum / float(self.samples)

    def rmse(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return math.sqrt(self.sq_err_sum / float(self.samples))

    def rmse_x(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return math.sqrt(self.sq_err_x_sum / float(self.samples))

    def rmse_y(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return math.sqrt(self.sq_err_y_sum / float(self.samples))

    def pred_avg(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return self.pred_range_sum / float(self.samples)

    def gt_avg(self) -> Optional[float]:
        if self.samples <= 0:
            return None
        return self.gt_range_sum / float(self.samples)


@dataclass
class VisibleConeMetrics:
    """Per-frame cone metrics for currently visible/matched cones."""

    cone_id: str
    color: str
    samples: int
    mae: Optional[float]
    rmse: Optional[float]
    rmse_x: Optional[float]
    rmse_y: Optional[float]
    pred_inst: float
    gt_inst: float
    pred_avg: float
    gt_avg: float


@dataclass
class ConeDepthMetrics:
    """Latest LiDAR cone-vs-ground-truth metrics."""

    pairs: int = 0
    axis_mae_m: Optional[float] = None
    axis_rmse_m: Optional[float] = None
    axis_bias_m: Optional[float] = None
    range_mae_m: Optional[float] = None
    range_rmse_m: Optional[float] = None
    sync_dt_ms: Optional[float] = None


class LidarNode(Node):
    """LiDAR-only cone detector with GT-backed RMSE evaluation."""

    _CONE_CLASS_NAME_TO_ID = {
        'blue': 0,
        'yellow': 1,
        'orange': 2,
        'big_orange': 3,
        'unknown': 4,
    }

    def __init__(self) -> None:
        super().__init__('lidar_node')
        self._declare_parameters()
        self._read_parameters()

        self._cone_queue: Deque[ConePacket] = deque()
        self._cone_lock = threading.Lock()
        self._odom_queue: Deque[OdomPacket] = deque()
        self._odom_lock = threading.Lock()
        self._track_refs: list[TrackConeRef] = []
        self._track_lock = threading.Lock()
        self._cone_stats: dict[str, ConeRunningStats] = {}
        self._cone_stats_lock = threading.Lock()
        self._latest_visible_cones: list[VisibleConeMetrics] = []
        self._visible_cones_lock = threading.Lock()
        self._latest_metrics = ConeDepthMetrics()
        self._last_throttled_log_sec: dict[str, float] = {}

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._range_rmse_analyzer: Optional[RangeRMSEAnalyzer] = None
        self._range_rmse_plot: Optional[RangeRMSELivePlot] = None
        if self.cone_plotting_2:
            self._range_rmse_analyzer = RangeRMSEAnalyzer(range_min_m=0.0, range_max_m=20.0, bin_width_m=1.0)
            try:
                self._range_rmse_plot = RangeRMSELivePlot(range_min_m=0.0, range_max_m=20.0, bin_width_m=1.0)
                self.create_timer(0.2, self._update_range_rmse_plot)
            except Exception as exc:  # pylint: disable=broad-except
                self._range_rmse_plot = None
                self.get_logger().warn(
                    f'Failed to initialize lidar cone_plotting_2 window ({exc}); disabling live range plot.'
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
        self._cone_sample_pub = self.create_publisher(String, f'{prefix}/cone_depth_samples', 10)
        self._cone_detections_pub = self.create_publisher(ConeDetectionArray, self.cone_detections_topic, 10)

        self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10)
        self.create_subscription(ConeArrayWithCovariance, self.ground_truth_cones_topic, self._cone_gt_cb, 10)
        self.create_subscription(ConeArrayWithCovariance, self.ground_truth_track_topic, self._track_gt_cb, 10)
        self.create_subscription(Odometry, self.cone_eval_odom_topic, self._odom_cb, 10)

        if self.perf_log_hz > 0.0:
            self.create_timer(1.0 / self.perf_log_hz, self._perf_timer_cb)

        self.get_logger().info(
            'lidar_node ready: '
            f'scan={self.scan_topic} detections={self.cone_detections_topic} '
            f'eval_prefix={self.eval_topic_prefix} gt_cones={self.ground_truth_cones_topic} '
            f'gt_track={self.ground_truth_track_topic} odom={self.cone_eval_odom_topic} '
            f'plotting2={self.cone_plotting_2}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('scan_topic', '/sim/raw/lidar')
        self.declare_parameter('eval_topic_prefix', '/sim/raw/lidar/eval')
        self.declare_parameter('cone_detections_topic', '/sim/raw/lidar/perception/cones_3d')
        self.declare_parameter('ground_truth_cones_topic', '/ground_truth/cones')
        self.declare_parameter('ground_truth_track_topic', '/ground_truth/track')
        self.declare_parameter('cone_eval_odom_topic', '/sim/odom')
        self.declare_parameter('lidar_frame', 'lidar_link')
        self.declare_parameter('cone_detections_frame', 'base_footprint')
        self.declare_parameter('min_detection_range_m', 0.5)
        self.declare_parameter('max_detection_range_m', 20.0)
        self.declare_parameter('cluster_jump_threshold_m', 0.18)
        self.declare_parameter('min_cluster_points', 2)
        self.declare_parameter('max_cluster_points', 12)
        self.declare_parameter('min_cluster_width_m', 0.03)
        self.declare_parameter('max_cluster_width_m', 0.45)
        self.declare_parameter('max_cluster_depth_m', 0.35)
        self.declare_parameter('cone_eval_sync_slop_sec', 0.10)
        self.declare_parameter('eval_match_threshold_m', 0.75)
        self.declare_parameter('cone_eval_track_match_threshold_m', 1.5)
        self.declare_parameter('cone_eval_track_match_relaxed_threshold_m', 3.0)
        self.declare_parameter('cone_eval_per_cone_max_rows', 120)
        self.declare_parameter('cone_eval_tf_timeout_sec', 0.0)
        self.declare_parameter('cone_plotting_2', False)
        self.declare_parameter('perf_log_hz', 0.0)

    def _read_parameters(self) -> None:
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.eval_topic_prefix = str(self.get_parameter('eval_topic_prefix').value)
        self.cone_detections_topic = str(self.get_parameter('cone_detections_topic').value)
        self.ground_truth_cones_topic = str(self.get_parameter('ground_truth_cones_topic').value)
        self.ground_truth_track_topic = str(self.get_parameter('ground_truth_track_topic').value)
        self.cone_eval_odom_topic = str(self.get_parameter('cone_eval_odom_topic').value)
        self.lidar_frame = str(self.get_parameter('lidar_frame').value).strip() or 'lidar_link'
        self.cone_detections_frame = str(self.get_parameter('cone_detections_frame').value).strip() or 'base_footprint'
        self.min_detection_range_m = max(0.01, float(self.get_parameter('min_detection_range_m').value))
        self.max_detection_range_m = max(
            self.min_detection_range_m + 0.1, float(self.get_parameter('max_detection_range_m').value)
        )
        self.cluster_jump_threshold_m = max(0.01, float(self.get_parameter('cluster_jump_threshold_m').value))
        self.min_cluster_points = max(1, int(self.get_parameter('min_cluster_points').value))
        self.max_cluster_points = max(self.min_cluster_points, int(self.get_parameter('max_cluster_points').value))
        self.min_cluster_width_m = max(0.0, float(self.get_parameter('min_cluster_width_m').value))
        self.max_cluster_width_m = max(
            self.min_cluster_width_m + 1e-6, float(self.get_parameter('max_cluster_width_m').value)
        )
        self.max_cluster_depth_m = max(0.0, float(self.get_parameter('max_cluster_depth_m').value))
        self.cone_eval_sync_slop_sec = max(0.01, float(self.get_parameter('cone_eval_sync_slop_sec').value))
        self.eval_match_threshold_m = max(0.05, float(self.get_parameter('eval_match_threshold_m').value))
        self.cone_eval_track_match_threshold_m = max(
            0.1, float(self.get_parameter('cone_eval_track_match_threshold_m').value)
        )
        self.cone_eval_track_match_relaxed_threshold_m = max(
            self.cone_eval_track_match_threshold_m,
            float(self.get_parameter('cone_eval_track_match_relaxed_threshold_m').value),
        )
        self.cone_eval_per_cone_max_rows = max(1, int(self.get_parameter('cone_eval_per_cone_max_rows').value))
        self.cone_eval_tf_timeout_sec = max(0.0, float(self.get_parameter('cone_eval_tf_timeout_sec').value))
        self.cone_plotting_2 = bool(self.get_parameter('cone_plotting_2').value)
        self.perf_log_hz = max(0.0, float(self.get_parameter('perf_log_hz').value))

    def _scan_cb(self, msg: LaserScan) -> None:
        stamp_sec = self._stamp_msg_to_sec(msg.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = time.monotonic()

        detections_xy = self._extract_detections(msg)
        detections_out = self._transform_detections(
            detections_xy=detections_xy,
            source_frame=str(msg.header.frame_id).strip() or self.lidar_frame,
            stamp=msg.header.stamp,
        )
        self._publish_cone_detections(detections_out, stamp=msg.header.stamp)

        metrics, visible_rows, sample_rows = self._evaluate_frame(
            detections_out=detections_out,
            target_stamp_sec=stamp_sec,
        )
        self._latest_metrics = metrics
        self._set_latest_visible_cones(visible_rows)
        self._publish_cone_metrics(metrics)
        self._publish_per_cone_table()
        self._publish_range_rmse_samples(sample_rows)

    def _extract_detections(self, scan: LaserScan) -> list[tuple[float, float]]:
        points = points_from_ranges(
            scan.ranges,
            angle_min_rad=float(scan.angle_min),
            angle_increment_rad=float(scan.angle_increment),
            range_min_m=float(scan.range_min),
            range_max_m=float(scan.range_max),
            min_detection_range_m=self.min_detection_range_m,
            max_detection_range_m=self.max_detection_range_m,
        )
        clusters = detect_cone_candidates(
            points,
            jump_threshold_m=self.cluster_jump_threshold_m,
            min_cluster_points=self.min_cluster_points,
            max_cluster_points=self.max_cluster_points,
            min_cluster_width_m=self.min_cluster_width_m,
            max_cluster_width_m=self.max_cluster_width_m,
            max_cluster_depth_m=self.max_cluster_depth_m,
        )
        return [(cluster.x_m, cluster.y_m) for cluster in clusters]

    def _transform_detections(
        self,
        *,
        detections_xy: list[tuple[float, float]],
        source_frame: str,
        stamp,
    ) -> list[tuple[float, float]]:
        if not detections_xy:
            return []
        if source_frame == self.cone_detections_frame:
            return detections_xy
        transform = None
        resolved_source = source_frame
        for source_candidate in self._source_frame_candidates(source_frame):
            transform = self._lookup_transform(
                target_frame=self.cone_detections_frame,
                source_frame=source_candidate,
                stamp=stamp,
            )
            if transform is not None:
                resolved_source = source_candidate
                break
        if transform is None:
            self._warn_throttled(
                'lidar_tf_missing',
                f'lidar detection transform unavailable {source_frame}->{self.cone_detections_frame}; dropping frame',
            )
            return []
        if resolved_source != source_frame:
            self._warn_throttled(
                'lidar_tf_alias',
                f'lidar detection frame alias {source_frame}->{resolved_source} for transform to {self.cone_detections_frame}',
            )
        transformed = []
        for x, y in detections_xy:
            xx, yy, _ = self._transform_point(transform, x, y, 0.0)
            transformed.append((xx, yy))
        return transformed

    def _publish_cone_detections(self, detections_out: list[tuple[float, float]], stamp) -> None:
        msg = ConeDetectionArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.cone_detections_frame
        for x, y in detections_out:
            cone = ConeDetection()
            cone.color = 'unknown'
            cone.confidence = 0.5
            cone.position.x = float(x)
            cone.position.y = float(y)
            cone.position.z = 0.0
            msg.cones.append(cone)
        self._cone_detections_pub.publish(msg)

    def _evaluate_frame(
        self,
        *,
        detections_out: list[tuple[float, float]],
        target_stamp_sec: float,
    ) -> tuple[ConeDepthMetrics, list[VisibleConeMetrics], list[tuple[str, float, float, Optional[int], Optional[int]]]]:
        packet = self._get_nearest_cone_packet(target_stamp_sec)
        if packet is None:
            return ConeDepthMetrics(pairs=0), [], []

        sync_dt_ms = abs(target_stamp_sec - packet.stamp_sec) * 1000.0
        metrics = ConeDepthMetrics(pairs=0, sync_dt_ms=sync_dt_ms)
        if sync_dt_ms > self.cone_eval_sync_slop_sec * 1000.0:
            return metrics, [], []

        gt_frame = str(packet.msg.header.frame_id).strip() or 'base_footprint'
        gt_points = list(self._iter_gt_points(packet.msg))
        gt_in_output = self._transform_gt_points(gt_points, source_frame=gt_frame, stamp=packet.msg.header.stamp)
        if gt_in_output is None:
            self._warn_throttled(
                'lidar_gt_tf_missing',
                f'ground truth transform unavailable {gt_frame}->{self.cone_detections_frame}; skipping eval frame',
            )
            return metrics, [], []

        matches = self._match_predictions_to_gt(
            detections=detections_out,
            gt_points=gt_in_output,
            threshold_m=self.eval_match_threshold_m,
        )
        if not matches:
            return metrics, [], []

        axis_errors = []
        range_errors = []
        err_x_vals = []
        err_y_vals = []
        visible_rows = []
        sample_rows = []

        for pred_idx, gt_idx, _ in matches:
            pred_x, pred_y = detections_out[pred_idx]
            gt_color, gt_x, gt_y = gt_in_output[gt_idx]
            pred_range = math.hypot(pred_x, pred_y)
            gt_range = math.hypot(gt_x, gt_y)
            axis_err = pred_range - gt_range
            err_x = pred_x - gt_x
            err_y = pred_y - gt_y

            cone_id = self._assign_cone_id(
                gt_x=gt_x,
                gt_y=gt_y,
                color=gt_color,
                target_stamp_sec=target_stamp_sec,
                fallback_index=gt_idx,
            )

            self._update_cone_stats(
                cone_id=cone_id,
                color=gt_color,
                axis_err_m=axis_err,
                err_x_m=err_x,
                err_y_m=err_y,
                pred_range_m=pred_range,
                gt_range_m=gt_range,
            )
            samples, mae, rmse, rmse_x, rmse_y, pred_avg, gt_avg = self._get_cone_stat_values(cone_id)
            visible_rows.append(
                VisibleConeMetrics(
                    cone_id=cone_id,
                    color=gt_color,
                    samples=samples,
                    mae=mae,
                    rmse=rmse,
                    rmse_x=rmse_x,
                    rmse_y=rmse_y,
                    pred_inst=pred_range,
                    gt_inst=gt_range,
                    pred_avg=pred_avg if pred_avg is not None else float('nan'),
                    gt_avg=gt_avg if gt_avg is not None else float('nan'),
                )
            )

            gt_class_id = self._cone_class_name_to_id(gt_color)
            sample_rows.append(('lidar', gt_range, axis_err, None, gt_class_id))
            self._record_range_rmse_sample('lidar', gt_range, axis_err, predicted_class_id=None, ground_truth_class_id=gt_class_id)

            axis_errors.append(axis_err)
            range_errors.append(abs(pred_range - gt_range))
            err_x_vals.append(err_x)
            err_y_vals.append(err_y)

        axis_arr = np.asarray(axis_errors, dtype=np.float64)
        range_arr = np.asarray(range_errors, dtype=np.float64)
        err_x_arr = np.asarray(err_x_vals, dtype=np.float64)
        err_y_arr = np.asarray(err_y_vals, dtype=np.float64)
        metrics.pairs = int(len(matches))
        metrics.axis_mae_m = float(np.mean(np.abs(axis_arr)))
        metrics.axis_rmse_m = float(np.sqrt(np.mean(np.square(axis_arr))))
        metrics.axis_bias_m = float(np.mean(axis_arr))
        metrics.range_mae_m = float(np.mean(range_arr))
        metrics.range_rmse_m = float(np.sqrt(np.mean(np.square(range_arr))))

        return metrics, visible_rows, sample_rows

    def _cone_gt_cb(self, msg: ConeArrayWithCovariance) -> None:
        stamp_sec = self._stamp_msg_to_sec(msg.header.stamp)
        with self._cone_lock:
            self._cone_queue.append(ConePacket(msg=msg, stamp_sec=stamp_sec))
            while len(self._cone_queue) > 200:
                self._cone_queue.popleft()

    def _track_gt_cb(self, msg: ConeArrayWithCovariance) -> None:
        refs = []
        refs.extend(self._build_track_refs(msg.blue_cones, 'blue'))
        refs.extend(self._build_track_refs(msg.yellow_cones, 'yellow'))
        refs.extend(self._build_track_refs(msg.orange_cones, 'orange'))
        refs.extend(self._build_track_refs(msg.big_orange_cones, 'big_orange'))
        refs.extend(self._build_track_refs(msg.unknown_color_cones, 'unknown'))
        with self._track_lock:
            self._track_refs = refs

    def _odom_cb(self, msg: Odometry) -> None:
        stamp_sec = self._stamp_msg_to_sec(msg.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = time.monotonic()
        with self._odom_lock:
            self._odom_queue.append(OdomPacket(msg=msg, stamp_sec=stamp_sec))
            while len(self._odom_queue) > 400:
                self._odom_queue.popleft()

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

    @staticmethod
    def _iter_gt_points(msg: ConeArrayWithCovariance):
        for cone in msg.blue_cones:
            yield 'blue', float(cone.point.x), float(cone.point.y)
        for cone in msg.yellow_cones:
            yield 'yellow', float(cone.point.x), float(cone.point.y)
        for cone in msg.orange_cones:
            yield 'orange', float(cone.point.x), float(cone.point.y)
        for cone in msg.big_orange_cones:
            yield 'big_orange', float(cone.point.x), float(cone.point.y)
        for cone in msg.unknown_color_cones:
            yield 'unknown', float(cone.point.x), float(cone.point.y)

    def _transform_gt_points(
        self,
        gt_points: list[tuple[str, float, float]],
        *,
        source_frame: str,
        stamp,
    ) -> Optional[list[tuple[str, float, float]]]:
        if source_frame == self.cone_detections_frame:
            return gt_points
        transform = self._lookup_transform(
            target_frame=self.cone_detections_frame,
            source_frame=source_frame,
            stamp=stamp,
        )
        if transform is None:
            return None
        out = []
        for color, x, y in gt_points:
            xx, yy, _ = self._transform_point(transform, x, y, 0.0)
            out.append((color, xx, yy))
        return out

    @staticmethod
    def _match_predictions_to_gt(
        *,
        detections: list[tuple[float, float]],
        gt_points: list[tuple[str, float, float]],
        threshold_m: float,
    ) -> list[tuple[int, int, float]]:
        if not detections or not gt_points:
            return []
        threshold_sq = max(0.0, threshold_m) * max(0.0, threshold_m)
        candidates = []
        for i, (px, py) in enumerate(detections):
            for j, (_, gx, gy) in enumerate(gt_points):
                dx = px - gx
                dy = py - gy
                dist_sq = dx * dx + dy * dy
                if dist_sq <= threshold_sq:
                    candidates.append((dist_sq, i, j))
        candidates.sort(key=lambda item: item[0])
        used_pred = set()
        used_gt = set()
        matches = []
        for dist_sq, i, j in candidates:
            if i in used_pred or j in used_gt:
                continue
            used_pred.add(i)
            used_gt.add(j)
            matches.append((i, j, math.sqrt(dist_sq)))
        return matches

    def _assign_cone_id(
        self,
        *,
        gt_x: float,
        gt_y: float,
        color: str,
        target_stamp_sec: float,
        fallback_index: int,
    ) -> str:
        map_point = self._point_to_track_with_odom_fallback(gt_x, gt_y, 0.0, target_stamp_sec=target_stamp_sec)
        if map_point is None:
            return f'{color}_temp_{fallback_index:03d}'
        map_x, map_y, _ = map_point
        cone_id = self._match_cone_id(map_x, map_y, color)
        if cone_id is None:
            return f'{color}_temp_{fallback_index:03d}'
        return cone_id

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
            track_refs = list(self._track_refs)
        if not track_refs:
            return None
        strict = self._find_nearest_track_ref(
            track_refs=track_refs,
            x_map=x_map,
            y_map=y_map,
            color=color,
            threshold_m=self.cone_eval_track_match_threshold_m,
            allow_color_mismatch=False,
            require_unambiguous=False,
        )
        if strict is not None:
            return strict
        relaxed = self._find_nearest_track_ref(
            track_refs=track_refs,
            x_map=x_map,
            y_map=y_map,
            color=color,
            threshold_m=self.cone_eval_track_match_relaxed_threshold_m,
            allow_color_mismatch=False,
            require_unambiguous=True,
        )
        if relaxed is not None:
            return relaxed
        if color != 'unknown':
            return self._find_nearest_track_ref(
                track_refs=track_refs,
                x_map=x_map,
                y_map=y_map,
                color=color,
                threshold_m=self.cone_eval_track_match_relaxed_threshold_m,
                allow_color_mismatch=True,
                require_unambiguous=True,
            )
        return None

    @staticmethod
    def _find_nearest_track_ref(
        *,
        track_refs: list[TrackConeRef],
        x_map: float,
        y_map: float,
        color: str,
        threshold_m: float,
        allow_color_mismatch: bool,
        require_unambiguous: bool,
    ) -> Optional[str]:
        best_id = None
        best_dist_sq = None
        second_best_dist_sq = None
        threshold_sq = max(0.0, threshold_m) * max(0.0, threshold_m)
        for ref in track_refs:
            if (not allow_color_mismatch) and color != 'unknown' and ref.color != color:
                continue
            dx = ref.x - x_map
            dy = ref.y - y_map
            dist_sq = dx * dx + dy * dy
            if dist_sq > threshold_sq:
                continue
            if best_dist_sq is None or dist_sq < best_dist_sq:
                if best_dist_sq is not None:
                    second_best_dist_sq = best_dist_sq
                best_dist_sq = dist_sq
                best_id = ref.cone_id
            elif second_best_dist_sq is None or dist_sq < second_best_dist_sq:
                second_best_dist_sq = dist_sq
        if best_id is None:
            return None
        if not require_unambiguous:
            return best_id
        if second_best_dist_sq is None:
            return best_id
        best_d = math.sqrt(best_dist_sq)
        second_d = math.sqrt(second_best_dist_sq)
        ratio = second_d / max(best_d, 1e-6)
        if ratio < 1.15 and (second_d - best_d) < 0.35:
            return None
        return best_id

    def _point_to_track_with_odom_fallback(
        self,
        x: float,
        y: float,
        z: float,
        *,
        target_stamp_sec: Optional[float] = None,
    ) -> Optional[tuple[float, float, float]]:
        with self._odom_lock:
            nearest = self._nearest_odom_locked(target_stamp_sec)
        if nearest is None:
            return None
        pose = nearest.msg.pose.pose
        return self._transform_point_from_pose(pose, x, y, z)

    def _nearest_odom_locked(self, target_stamp_sec: Optional[float]) -> Optional[OdomPacket]:
        if not self._odom_queue:
            return None
        if target_stamp_sec is None or not math.isfinite(target_stamp_sec):
            return self._odom_queue[-1]
        best = None
        best_dt = None
        for packet in self._odom_queue:
            dt = abs(packet.stamp_sec - target_stamp_sec)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = packet
        if best is None:
            return self._odom_queue[-1]
        if best_dt is not None and best_dt > 0.25:
            return self._odom_queue[-1]
        return best

    def _update_cone_stats(
        self,
        *,
        cone_id: str,
        color: str,
        axis_err_m: float,
        err_x_m: float,
        err_y_m: float,
        pred_range_m: float,
        gt_range_m: float,
    ) -> None:
        with self._cone_stats_lock:
            stats = self._cone_stats.get(cone_id)
            if stats is None:
                stats = ConeRunningStats(color=color)
                self._cone_stats[cone_id] = stats
            stats.add(
                err_m=axis_err_m,
                err_x_m=err_x_m,
                err_y_m=err_y_m,
                pred_range_m=pred_range_m,
                gt_range_m=gt_range_m,
            )

    def _get_cone_stat_values(self, cone_id: str):
        with self._cone_stats_lock:
            stats = self._cone_stats.get(cone_id)
            if stats is None:
                return 0, None, None, None, None, None, None
            return (
                stats.samples,
                stats.mae(),
                stats.rmse(),
                stats.rmse_x(),
                stats.rmse_y(),
                stats.pred_avg(),
                stats.gt_avg(),
            )

    def _set_latest_visible_cones(self, rows: list[VisibleConeMetrics]) -> None:
        if not rows:
            with self._visible_cones_lock:
                self._latest_visible_cones = []
            return
        best_by_id = {}
        for row in rows:
            prev = best_by_id.get(row.cone_id)
            if prev is None or row.gt_inst < prev.gt_inst:
                best_by_id[row.cone_id] = row
        ordered = sorted(best_by_id.values(), key=lambda item: (item.gt_inst, item.pred_inst, item.cone_id))
        with self._visible_cones_lock:
            self._latest_visible_cones = ordered

    def _per_cone_table_text(self) -> str:
        with self._visible_cones_lock:
            rows = list(self._latest_visible_cones)
        if not rows:
            return 'no per-cone depth samples yet'
        lines = [
            'cone_id,color,samples,axis_mae_m,axis_rmse_m,axis_rmse_x_m,axis_rmse_y_m,'
            'dcam_inst,dgt_inst,dcam,dgt'
        ]
        for idx, item in enumerate(rows):
            if idx >= self.cone_eval_per_cone_max_rows:
                remaining = len(rows) - idx
                lines.append(f'... ({remaining} more cones)')
                break
            mae = 'n/a' if item.mae is None else f'{item.mae:.4f}'
            rmse = 'n/a' if item.rmse is None else f'{item.rmse:.4f}'
            rmse_x = 'n/a' if item.rmse_x is None else f'{item.rmse_x:.4f}'
            rmse_y = 'n/a' if item.rmse_y is None else f'{item.rmse_y:.4f}'
            lines.append(
                f'{item.cone_id},{item.color},{item.samples},{mae},{rmse},{rmse_x},{rmse_y},'
                f'{item.pred_inst:.4f},{item.gt_inst:.4f},{item.pred_avg:.4f},{item.gt_avg:.4f}'
            )
        return '\n'.join(lines)

    def _publish_per_cone_table(self) -> None:
        self._cone_per_cone_pub.publish(String(data=self._per_cone_table_text()))

    def _publish_cone_metrics(self, metrics: ConeDepthMetrics) -> None:
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

    def _publish_range_rmse_samples(
        self,
        samples: list[tuple[str, float, float, Optional[int], Optional[int]]],
    ) -> None:
        if not samples:
            return
        lines = ['source,gt_range_m,error_m,predicted_class_id,ground_truth_class_id']
        for source, gt_range_m, error_m, predicted_class_id, ground_truth_class_id in samples:
            if not (math.isfinite(gt_range_m) and math.isfinite(error_m)):
                continue
            predicted_str = '' if predicted_class_id is None else str(int(predicted_class_id))
            gt_str = '' if ground_truth_class_id is None else str(int(ground_truth_class_id))
            lines.append(f'{source},{gt_range_m:.6f},{error_m:.6f},{predicted_str},{gt_str}')
        if len(lines) > 1:
            self._cone_sample_pub.publish(String(data='\n'.join(lines)))

    def _record_range_rmse_sample(
        self,
        source: str,
        gt_range_m: float,
        error_m: float,
        *,
        predicted_class_id: Optional[int],
        ground_truth_class_id: Optional[int],
    ) -> None:
        if self._range_rmse_analyzer is None:
            return
        self._range_rmse_analyzer.add_sample(
            source=source,
            gt_range_m=gt_range_m,
            error_m=error_m,
            predicted_class_id=predicted_class_id,
            ground_truth_class_id=ground_truth_class_id,
        )

    def _update_range_rmse_plot(self) -> None:
        if self._range_rmse_analyzer is None or self._range_rmse_plot is None:
            return
        stats = self._range_rmse_analyzer.compute_binned_rmse()
        is_open = self._range_rmse_plot.update(stats)
        if not is_open:
            self._range_rmse_plot = None

    def _perf_timer_cb(self) -> None:
        self._publish_per_cone_table()
        m = self._latest_metrics
        self.get_logger().info(
            'lidar cone eval '
            f'pairs={m.pairs} axis_rmse={self._fmt_opt(m.axis_rmse_m)} '
            f'range_rmse={self._fmt_opt(m.range_rmse_m)} sync_dt_ms={self._fmt_opt(m.sync_dt_ms)}'
        )

    @staticmethod
    def _fmt_opt(value: Optional[float]) -> str:
        if value is None:
            return 'n/a'
        return f'{value:.3f}'

    def _lookup_transform(self, *, target_frame: str, source_frame: str, stamp):
        timeout = Duration(seconds=float(self.cone_eval_tf_timeout_sec))
        target = target_frame.strip()
        source = source_frame.strip()
        if not target or not source:
            return None
        try:
            stamp_time = Time.from_msg(stamp)
            return self._tf_buffer.lookup_transform(target, source, stamp_time, timeout=timeout)
        except (TransformException, ValueError):
            pass
        try:
            return self._tf_buffer.lookup_transform(target, source, Time(), timeout=timeout)
        except TransformException:
            return None

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

    def _warn_throttled(self, key: str, message: str) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get(key, -1.0)
        if (now_sec - last_sec) >= 1.0:
            self.get_logger().warn(message)
            self._last_throttled_log_sec[key] = now_sec

    @staticmethod
    def _stamp_msg_to_sec(stamp) -> float:
        return float(stamp.sec) + (float(stamp.nanosec) * 1e-9)

    @classmethod
    def _cone_class_name_to_id(cls, name: str) -> Optional[int]:
        return cls._CONE_CLASS_NAME_TO_ID.get(name.strip().lower())

    def shutdown(self) -> None:
        if self._range_rmse_plot is not None:
            try:
                self._range_rmse_plot.close()
            except Exception:  # pylint: disable=broad-except
                pass

    def _source_frame_candidates(self, source_frame: str) -> list[str]:
        source = source_frame.strip()
        if not source:
            return [self.lidar_frame]

        candidates: list[str] = []

        def add(frame: str) -> None:
            f = frame.strip()
            if f and f not in candidates:
                candidates.append(f)

        add(source)
        add(self.lidar_frame)

        leaf = source.split('/')[-1]
        if leaf:
            add(leaf)
            if leaf == 'front_lidar':
                add('lidar_link')

        if '/' in source:
            prefix = source.rsplit('/', 1)[0]
            add(f'{prefix}/lidar_link')
            if leaf == 'front_lidar':
                add(f'{prefix}/lidar_link')

        return candidates


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
