"""PointCloud2-based lidar cone detection node."""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.lidar.pointcloud_processing import (
    apply_azimuth_masks,
    apply_range_thinning,
    crop_points_to_roi,
    downsample_points,
    pointcloud2_to_xyz_array,
    summarize_rejection_reasons,
    summarize_clusters_for_debug,
    suppress_ground_points,
    xyz_array_to_pointcloud2,
)


class PointCloudLidarNode(Node):
    """LiDAR cone detector that consumes PointCloud2 and publishes ConeDetectionArray."""

    def __init__(self) -> None:
        super().__init__('pointcloud_lidar_node')
        self._declare_parameters()
        self._read_parameters()

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._last_throttled_log_sec: dict[str, float] = {}
        self._rng = np.random.default_rng(self.random_seed)

        self._filtered_points_pub = self.create_publisher(PointCloud2, self.filtered_pointcloud_topic, qos_profile_sensor_data)
        self._cone_detections_pub = self.create_publisher(ConeDetectionArray, self.cone_detections_topic, 10)
        self._debug_clusters_pub = self.create_publisher(MarkerArray, self.debug_clusters_topic, 10)
        self.create_subscription(PointCloud2, self.pointcloud_topic, self._pointcloud_cb, qos_profile_sensor_data)

        self.get_logger().info(
            'pointcloud_lidar_node ready: '
            f'points={self.pointcloud_topic} filtered={self.filtered_pointcloud_topic} '
            f'detections={self.cone_detections_topic} detections_frame={self.cone_detections_frame}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('pointcloud_topic', '/sim/raw/lidar/points')
        self.declare_parameter('filtered_pointcloud_topic', '/sim/raw/lidar/points_filtered')
        self.declare_parameter('cone_detections_topic', '/sim/raw/lidar/perception/cones_3d')
        self.declare_parameter('debug_clusters_topic', '/sim/raw/lidar/debug/cone_candidates')
        self.declare_parameter('lidar_frame', 'lidar_link')
        self.declare_parameter('cone_detections_frame', 'base_footprint')
        self.declare_parameter('x_min_m', -3.0)
        self.declare_parameter('x_max_m', 25.0)
        self.declare_parameter('y_min_m', -12.0)
        self.declare_parameter('y_max_m', 12.0)
        self.declare_parameter('ground_base_cutoff_m', 0.035)
        self.declare_parameter('ground_range_bias_m', 0.01)
        self.declare_parameter('ground_range_slope_m_per_m', 0.004)
        self.declare_parameter('z_max_m', 0.90)
        self.declare_parameter('cluster_radius_m', 0.30)
        self.declare_parameter('min_cluster_points', 3)
        self.declare_parameter('max_cluster_points', 80)
        self.declare_parameter('min_cluster_width_m', 0.05)
        self.declare_parameter('max_cluster_width_m', 0.60)
        self.declare_parameter('min_cluster_depth_m', 0.05)
        self.declare_parameter('max_cluster_depth_m', 0.60)
        self.declare_parameter('min_cluster_height_m', 0.15)
        self.declare_parameter('max_cluster_height_m', 0.60)
        self.declare_parameter('frame_dropout_probability', 0.0)
        self.declare_parameter('thinning_start_range_m', 12.0)
        self.declare_parameter('thinning_keep_ratio_at_max_range', 1.0)
        self.declare_parameter('max_detection_range_m', 25.0)
        self.declare_parameter('azimuth_mask_ranges_deg', [])
        self.declare_parameter('downsample_stride', 1)
        self.declare_parameter('lidar_confidence', 0.5)
        self.declare_parameter('cone_eval_tf_timeout_sec', 0.0)
        self.declare_parameter('random_seed', 0)

    def _read_parameters(self) -> None:
        self.pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        self.filtered_pointcloud_topic = str(self.get_parameter('filtered_pointcloud_topic').value)
        self.cone_detections_topic = str(self.get_parameter('cone_detections_topic').value)
        self.debug_clusters_topic = str(self.get_parameter('debug_clusters_topic').value)
        self.lidar_frame = str(self.get_parameter('lidar_frame').value).strip() or 'lidar_link'
        self.cone_detections_frame = str(self.get_parameter('cone_detections_frame').value).strip() or 'base_footprint'
        self.x_min_m = float(self.get_parameter('x_min_m').value)
        self.x_max_m = float(self.get_parameter('x_max_m').value)
        self.y_min_m = float(self.get_parameter('y_min_m').value)
        self.y_max_m = float(self.get_parameter('y_max_m').value)
        self.ground_base_cutoff_m = float(self.get_parameter('ground_base_cutoff_m').value)
        self.ground_range_bias_m = float(self.get_parameter('ground_range_bias_m').value)
        self.ground_range_slope_m_per_m = float(self.get_parameter('ground_range_slope_m_per_m').value)
        self.z_max_m = float(self.get_parameter('z_max_m').value)
        self.cluster_radius_m = max(0.01, float(self.get_parameter('cluster_radius_m').value))
        self.min_cluster_points = max(1, int(self.get_parameter('min_cluster_points').value))
        self.max_cluster_points = max(self.min_cluster_points, int(self.get_parameter('max_cluster_points').value))
        self.min_cluster_width_m = max(0.0, float(self.get_parameter('min_cluster_width_m').value))
        self.max_cluster_width_m = max(self.min_cluster_width_m + 1e-6, float(self.get_parameter('max_cluster_width_m').value))
        self.min_cluster_depth_m = max(0.0, float(self.get_parameter('min_cluster_depth_m').value))
        self.max_cluster_depth_m = max(self.min_cluster_depth_m + 1e-6, float(self.get_parameter('max_cluster_depth_m').value))
        self.min_cluster_height_m = max(0.0, float(self.get_parameter('min_cluster_height_m').value))
        self.max_cluster_height_m = max(self.min_cluster_height_m + 1e-6, float(self.get_parameter('max_cluster_height_m').value))
        self.frame_dropout_probability = min(1.0, max(0.0, float(self.get_parameter('frame_dropout_probability').value)))
        self.thinning_start_range_m = max(0.0, float(self.get_parameter('thinning_start_range_m').value))
        self.thinning_keep_ratio_at_max_range = min(
            1.0,
            max(0.0, float(self.get_parameter('thinning_keep_ratio_at_max_range').value)),
        )
        self.max_detection_range_m = max(self.thinning_start_range_m + 1e-6, float(self.get_parameter('max_detection_range_m').value))
        mask_ranges = self.get_parameter('azimuth_mask_ranges_deg').value
        self.azimuth_mask_ranges_deg = [float(v) for v in (mask_ranges or [])]
        self.downsample_stride = max(1, int(self.get_parameter('downsample_stride').value))
        self.lidar_confidence = min(1.0, max(0.0, float(self.get_parameter('lidar_confidence').value)))
        self.cone_eval_tf_timeout_sec = max(0.0, float(self.get_parameter('cone_eval_tf_timeout_sec').value))
        self.random_seed = int(self.get_parameter('random_seed').value)

    def _pointcloud_cb(self, msg: PointCloud2) -> None:
        if self.frame_dropout_probability > 0.0 and self._rng.random() < self.frame_dropout_probability:
            self._publish_filtered_points(np.empty((0, 3), dtype=np.float32), stamp=msg.header.stamp)
            self._publish_debug_clusters([], stamp=msg.header.stamp)
            self._publish_cone_detections([], stamp=msg.header.stamp)
            return

        source_frame = str(msg.header.frame_id).strip() or self.lidar_frame
        points = pointcloud2_to_xyz_array(msg)
        if points.shape[0] == 0:
            self._publish_filtered_points(points, stamp=msg.header.stamp)
            self._publish_debug_clusters([], stamp=msg.header.stamp)
            self._publish_cone_detections([], stamp=msg.header.stamp)
            return

        raw_points_count = int(points.shape[0])
        points = downsample_points(points, self.downsample_stride)
        points = apply_azimuth_masks(points, self.azimuth_mask_ranges_deg)
        points = apply_range_thinning(
            points,
            thinning_start_range_m=self.thinning_start_range_m,
            max_range_m=self.max_detection_range_m,
            keep_ratio_at_max_range=self.thinning_keep_ratio_at_max_range,
            rng=self._rng,
        )
        points = self._transform_points(points_xyz=points, source_frame=source_frame, stamp=msg.header.stamp)
        roi_points = crop_points_to_roi(
            points,
            x_min_m=self.x_min_m,
            x_max_m=self.x_max_m,
            y_min_m=self.y_min_m,
            y_max_m=self.y_max_m,
            z_min_m=-10.0,
            z_max_m=self.z_max_m,
        )
        filtered_points = suppress_ground_points(
            roi_points,
            base_cutoff_m=self.ground_base_cutoff_m,
            range_slope_m_per_m=self.ground_range_slope_m_per_m,
            range_bias_m=self.ground_range_bias_m,
            z_max_m=self.z_max_m,
        )
        self._publish_filtered_points(filtered_points, stamp=msg.header.stamp)

        cluster_summaries = summarize_clusters_for_debug(
            filtered_points,
            max_cluster_radius_m=self.cluster_radius_m,
            min_cluster_points=self.min_cluster_points,
            max_cluster_points=self.max_cluster_points,
            min_cluster_width_m=self.min_cluster_width_m,
            max_cluster_width_m=self.max_cluster_width_m,
            min_cluster_depth_m=self.min_cluster_depth_m,
            max_cluster_depth_m=self.max_cluster_depth_m,
            min_cluster_height_m=self.min_cluster_height_m,
            max_cluster_height_m=self.max_cluster_height_m,
        )
        self._publish_debug_clusters(cluster_summaries, stamp=msg.header.stamp)
        detections = [summary for summary in cluster_summaries if summary.accepted]
        self._log_detection_stats(
            raw_points_count=raw_points_count,
            roi_points_count=int(roi_points.shape[0]),
            filtered_points_count=int(filtered_points.shape[0]),
            cluster_summaries=cluster_summaries,
        )
        self._publish_cone_detections(detections, stamp=msg.header.stamp)

    def _transform_points(self, *, points_xyz: np.ndarray, source_frame: str, stamp) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        if points.shape[0] == 0 or source_frame == self.cone_detections_frame:
            return points

        transform = None
        for source_candidate in self._source_frame_candidates(source_frame):
            transform = self._lookup_transform(
                target_frame=self.cone_detections_frame,
                source_frame=source_candidate,
                stamp=stamp,
            )
            if transform is not None:
                break

        if transform is None:
            self._warn_throttled(
                'lidar_tf_missing',
                f'lidar point transform unavailable {source_frame}->{self.cone_detections_frame}; dropping frame',
            )
            return np.empty((0, 3), dtype=np.float32)
        return self._transform_points_numpy(transform, points)

    def _publish_filtered_points(self, points_xyz: np.ndarray, *, stamp) -> None:
        msg = xyz_array_to_pointcloud2(points_xyz, frame_id=self.cone_detections_frame, stamp=stamp)
        self._filtered_points_pub.publish(msg)

    def _publish_debug_clusters(self, clusters, *, stamp) -> None:
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.header.stamp = stamp
        delete_all.header.frame_id = self.cone_detections_frame
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        for idx, cluster in enumerate(clusters):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.cone_detections_frame
            marker.ns = 'accepted_clusters' if cluster.accepted else 'rejected_clusters'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(cluster.x_m)
            marker.pose.position.y = float(cluster.y_m)
            marker.pose.position.z = float(cluster.z_m)
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(0.08, float(cluster.width_m))
            marker.scale.y = max(0.08, float(cluster.depth_m))
            marker.scale.z = max(0.08, float(cluster.height_m))
            if cluster.accepted:
                marker.color.r = 0.1
                marker.color.g = 0.9
                marker.color.b = 0.2
                marker.color.a = 0.75
            else:
                marker.color.r = 0.95
                marker.color.g = 0.2
                marker.color.b = 0.2
                marker.color.a = 0.45
            markers.markers.append(marker)

            text = Marker()
            text.header.stamp = stamp
            text.header.frame_id = self.cone_detections_frame
            text.ns = 'cluster_labels'
            text.id = 10000 + idx
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(cluster.x_m)
            text.pose.position.y = float(cluster.y_m)
            text.pose.position.z = float(cluster.z_m + 0.25)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.12
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.9
            status = 'ok' if cluster.accepted else cluster.reason
            text.text = f'{status} n={cluster.point_count}'
            markers.markers.append(text)

        self._debug_clusters_pub.publish(markers)

    def _publish_cone_detections(self, detections, *, stamp) -> None:
        msg = ConeDetectionArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.cone_detections_frame
        for detection in detections:
            cone = ConeDetection()
            cone.color = 'unknown'
            cone.confidence = float(self.lidar_confidence)
            cone.position.x = float(detection.x_m)
            cone.position.y = float(detection.y_m)
            cone.position.z = 0.0
            msg.cones.append(cone)
        self._cone_detections_pub.publish(msg)

    def _lookup_transform(self, *, target_frame: str, source_frame: str, stamp):
        timeout = Duration(seconds=float(self.cone_eval_tf_timeout_sec))
        try:
            stamp_time = Time.from_msg(stamp)
            return self._tf_buffer.lookup_transform(target_frame, source_frame, stamp_time, timeout=timeout)
        except (TransformException, ValueError):
            pass
        try:
            return self._tf_buffer.lookup_transform(target_frame, source_frame, Time(), timeout=timeout)
        except TransformException:
            return None

    @staticmethod
    def _transform_points_numpy(transform, points_xyz: np.ndarray) -> np.ndarray:
        q = transform.transform.rotation
        t = transform.transform.translation
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
        rotation = np.asarray(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )
        translation = np.asarray([float(t.x), float(t.y), float(t.z)], dtype=np.float32)
        return (points_xyz @ rotation.T) + translation

    def _warn_throttled(self, key: str, message: str) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get(key, -1.0)
        if (now_sec - last_sec) >= 1.0:
            self.get_logger().warn(message)
            self._last_throttled_log_sec[key] = now_sec

    def _log_detection_stats(
        self,
        *,
        raw_points_count: int,
        roi_points_count: int,
        filtered_points_count: int,
        cluster_summaries,
    ) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get('lidar_stats', -1.0)
        if (now_sec - last_sec) < 1.0:
            return

        accepted = [cluster for cluster in cluster_summaries if cluster.accepted]
        ranges = [float(np.hypot(cluster.x_m, cluster.y_m)) for cluster in accepted]
        histogram = {
            '<5m': sum(1 for value in ranges if value < 5.0),
            '5-10m': sum(1 for value in ranges if 5.0 <= value < 10.0),
            '10m+': sum(1 for value in ranges if value >= 10.0),
        }
        rejection_reasons = summarize_rejection_reasons(cluster_summaries)
        self.get_logger().info(
            'lidar stats '
            f'raw={raw_points_count} roi={roi_points_count} ground_suppressed={filtered_points_count} '
            f'clusters={len(cluster_summaries)} accepted={len(accepted)} '
            f'range_hist={histogram} rejected={rejection_reasons}'
        )
        self._last_throttled_log_sec['lidar_stats'] = now_sec

    def _source_frame_candidates(self, source_frame: str) -> list[str]:
        source = source_frame.strip()
        if not source:
            return [self.lidar_frame]
        candidates: list[str] = []

        def add(name: str) -> None:
            normalized = name.strip()
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        add(source)
        add(self.lidar_frame)
        if source.startswith('/'):
            add(source[1:])
        else:
            add(f'/{source}')
        if '/' in source:
            prefix, _, suffix = source.partition('/')
            if suffix:
                add(suffix)
                if not suffix.endswith('lidar_link'):
                    add(f'{prefix}/lidar_link')
            if prefix:
                add(f'{prefix}/lidar_link')
        return candidates or [self.lidar_frame]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PointCloudLidarNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
