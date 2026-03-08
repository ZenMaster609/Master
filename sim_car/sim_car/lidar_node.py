"""LiDAR cone detection node without GT evaluation."""

from __future__ import annotations

import time

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetection, ConeDetectionArray

from sim_car.lidar.clustering import detect_cone_candidates, points_from_ranges


class LidarNode(Node):
    """LiDAR-only cone detector that publishes ConeDetectionArray for planning."""

    def __init__(self) -> None:
        super().__init__('lidar_node')
        self._declare_parameters()
        self._read_parameters()

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._last_throttled_log_sec: dict[str, float] = {}
        self._cone_detections_pub = self.create_publisher(ConeDetectionArray, self.cone_detections_topic, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10)

        self.get_logger().info(
            'lidar_node ready: '
            f'scan={self.scan_topic} detections={self.cone_detections_topic} '
            f'detections_frame={self.cone_detections_frame}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('scan_topic', '/sim/raw/lidar')
        self.declare_parameter('cone_detections_topic', '/sim/raw/lidar/perception/cones_3d')
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
        self.declare_parameter('dedup_radius_m', 0.85)
        self.declare_parameter('cone_eval_tf_timeout_sec', 0.0)

    def _read_parameters(self) -> None:
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.cone_detections_topic = str(self.get_parameter('cone_detections_topic').value)
        self.lidar_frame = str(self.get_parameter('lidar_frame').value).strip() or 'lidar_link'
        self.cone_detections_frame = str(self.get_parameter('cone_detections_frame').value).strip() or 'base_footprint'
        self.min_detection_range_m = max(0.01, float(self.get_parameter('min_detection_range_m').value))
        self.max_detection_range_m = max(
            self.min_detection_range_m + 0.1,
            float(self.get_parameter('max_detection_range_m').value),
        )
        self.cluster_jump_threshold_m = max(0.01, float(self.get_parameter('cluster_jump_threshold_m').value))
        self.min_cluster_points = max(1, int(self.get_parameter('min_cluster_points').value))
        self.max_cluster_points = max(self.min_cluster_points, int(self.get_parameter('max_cluster_points').value))
        self.min_cluster_width_m = max(0.0, float(self.get_parameter('min_cluster_width_m').value))
        self.max_cluster_width_m = max(self.min_cluster_width_m + 1e-6, float(self.get_parameter('max_cluster_width_m').value))
        self.max_cluster_depth_m = max(0.0, float(self.get_parameter('max_cluster_depth_m').value))
        self.dedup_radius_m = max(0.01, float(self.get_parameter('dedup_radius_m').value))
        self.cone_eval_tf_timeout_sec = max(0.0, float(self.get_parameter('cone_eval_tf_timeout_sec').value))

    def _scan_cb(self, msg: LaserScan) -> None:
        detections_xy = self._extract_detections(msg)
        detections_out = self._transform_detections(
            detections_xy=detections_xy,
            source_frame=str(msg.header.frame_id).strip() or self.lidar_frame,
            stamp=msg.header.stamp,
        )
        self._publish_cone_detections(detections_out, stamp=msg.header.stamp)

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
        return self._deduplicate_xy([(cluster.x_m, cluster.y_m) for cluster in clusters], self.dedup_radius_m)

    def _transform_detections(self, *, detections_xy: list[tuple[float, float]], source_frame: str, stamp) -> list[tuple[float, float]]:
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

    def _warn_throttled(self, key: str, message: str) -> None:
        now_sec = time.monotonic()
        last_sec = self._last_throttled_log_sec.get(key, -1.0)
        if (now_sec - last_sec) >= 1.0:
            self.get_logger().warn(message)
            self._last_throttled_log_sec[key] = now_sec

    @staticmethod
    def _deduplicate_xy(points: list[tuple[float, float]], dedup_radius_m: float) -> list[tuple[float, float]]:
        if len(points) <= 1:
            return list(points)

        radius_sq = float(dedup_radius_m) * float(dedup_radius_m)
        merged: list[tuple[float, float, float]] = []

        for x, y in sorted(points, key=lambda item: (item[0] * item[0]) + (item[1] * item[1])):
            best_idx = -1
            best_dist_sq = float('inf')
            for idx, (mx, my, weight) in enumerate(merged):
                dx = x - mx
                dy = y - my
                dist_sq = (dx * dx) + (dy * dy)
                if dist_sq <= radius_sq and dist_sq < best_dist_sq:
                    best_idx = idx
                    best_dist_sq = dist_sq

            if best_idx < 0:
                merged.append((float(x), float(y), 1.0))
                continue

            mx, my, weight = merged[best_idx]
            new_weight = weight + 1.0
            merged[best_idx] = (
                ((mx * weight) + float(x)) / new_weight,
                ((my * weight) + float(y)) / new_weight,
                new_weight,
            )

        return [(x, y) for x, y, _weight in merged]

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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
