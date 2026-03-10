"""Evaluate predicted cone detections against GT and feed cone_plotting_2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import warnings

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from eufs_msgs.msg import ConeArrayWithCovariance
from tf2_ros import Buffer, TransformException, TransformListener
from vehicle_plotter_msgs.msg import ConeDetectionArray

from sim_car.cones.plotting.runtime import ConePlotting2Runtime

# This node does not use 3D plots; suppress non-fatal mixed-site-packages warning.
warnings.filterwarnings(
    'ignore',
    message=r'Unable to import Axes3D\..*',
    category=UserWarning,
    module=r'matplotlib\.projections',
)
# Numpy subnormal warnings are environment-specific and non-fatal for this node.
warnings.filterwarnings(
    'ignore',
    message=r'The value of the smallest subnormal for <class \'.*\'?> type is zero\.',
    category=UserWarning,
    module=r'numpy\.core\.getlimits',
)


@dataclass
class ConePacket:
    msg: ConeArrayWithCovariance
    stamp_sec: float


class ConeEvaluatorNode(Node):
    """Standalone evaluator node for cone_plotting_2 inputs and plotting."""

    _CONE_CLASS_NAME_TO_ID = {
        'blue': 0,
        'yellow': 1,
        'orange': 2,
        'big_orange': 3,
        'unknown': 4,
    }

    def __init__(self) -> None:
        super().__init__('cone_evaluator_node')
        self._declare_parameters()
        self._read_parameters()

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._cone_queue: deque[ConePacket] = deque()
        self._cone_lock = threading.Lock()

        self._runtime = ConePlotting2Runtime(
            self,
            eval_topic_prefix=self.eval_topic_prefix,
            enabled=self.cone_plotting_2,
            enable_live_plot=self.cone_plotting_2_live_plot,
        )

        self.create_subscription(ConeDetectionArray, self.predicted_cones_topic, self._predicted_cb, 10)
        self.create_subscription(ConeArrayWithCovariance, self.ground_truth_cones_topic, self._gt_cb, 10)

        self.get_logger().info(
            'cone_evaluator_node ready: '
            f'predicted={self.predicted_cones_topic} gt={self.ground_truth_cones_topic} '
            f'eval_prefix={self.eval_topic_prefix} source={self.source_name} plotting2={self.cone_plotting_2}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('predicted_cones_topic', '/sim/raw/stereo/perception/cones_3d')
        self.declare_parameter('ground_truth_cones_topic', '/ground_truth/cones')
        self.declare_parameter('eval_topic_prefix', '/sim/raw/stereo/eval')
        self.declare_parameter('source_name', 'stereo')
        self.declare_parameter('cone_plotting_2', False)
        self.declare_parameter('cone_plotting_2_live_plot', True)
        self.declare_parameter('cone_eval_sync_slop_sec', 0.10)
        self.declare_parameter('eval_match_threshold_m', 0.75)
        self.declare_parameter('cone_eval_tf_timeout_sec', 0.0)
        self.declare_parameter('cone_eval_include_unknown', False)

    def _read_parameters(self) -> None:
        self.predicted_cones_topic = str(self.get_parameter('predicted_cones_topic').value)
        self.ground_truth_cones_topic = str(self.get_parameter('ground_truth_cones_topic').value)
        self.eval_topic_prefix = str(self.get_parameter('eval_topic_prefix').value)
        self.source_name = str(self.get_parameter('source_name').value).strip().lower() or 'stereo'
        self.cone_plotting_2 = bool(self.get_parameter('cone_plotting_2').value)
        self.cone_plotting_2_live_plot = bool(self.get_parameter('cone_plotting_2_live_plot').value)
        self.cone_eval_sync_slop_sec = max(0.01, float(self.get_parameter('cone_eval_sync_slop_sec').value))
        self.eval_match_threshold_m = max(0.05, float(self.get_parameter('eval_match_threshold_m').value))
        self.cone_eval_tf_timeout_sec = max(0.0, float(self.get_parameter('cone_eval_tf_timeout_sec').value))
        self.cone_eval_include_unknown = bool(self.get_parameter('cone_eval_include_unknown').value)

    def _gt_cb(self, msg: ConeArrayWithCovariance) -> None:
        stamp_sec = self._stamp_msg_to_sec(msg.header.stamp)
        with self._cone_lock:
            self._cone_queue.append(ConePacket(msg=msg, stamp_sec=stamp_sec))
            while len(self._cone_queue) > 300:
                self._cone_queue.popleft()

    def _predicted_cb(self, msg: ConeDetectionArray) -> None:
        if not self._runtime.enabled:
            return

        pred_stamp_sec = self._stamp_msg_to_sec(msg.header.stamp)
        if pred_stamp_sec <= 0.0:
            return

        gt_packet = self._nearest_gt_packet(pred_stamp_sec)
        if gt_packet is None:
            return

        sync_dt = abs(pred_stamp_sec - gt_packet.stamp_sec)
        if sync_dt > self.cone_eval_sync_slop_sec:
            return

        target_frame = str(msg.header.frame_id).strip()
        if not target_frame:
            return

        gt_frame = str(gt_packet.msg.header.frame_id).strip() or 'map'
        gt_points = self._iter_gt_points(gt_packet.msg)
        transformed_gt = self._transform_gt_points(
            gt_points=gt_points,
            source_frame=gt_frame,
            target_frame=target_frame,
            stamp=msg.header.stamp,
        )
        if transformed_gt is None or not transformed_gt:
            return

        predictions = []
        for cone in msg.cones:
            predictions.append((float(cone.position.x), float(cone.position.y), self._cone_class_name_to_id(cone.color)))

        if not predictions:
            return

        matches = self._match_predictions_to_gt(predictions=predictions, gt_points=transformed_gt)
        if not matches:
            return

        sample_rows: list[tuple[str, float, float, int, int]] = []
        for pred_idx, gt_idx in matches:
            pred_x, pred_y, pred_class_id = predictions[pred_idx]
            gt_color, gt_x, gt_y = transformed_gt[gt_idx]
            gt_class_id = self._cone_class_name_to_id(gt_color)

            pred_range = math.hypot(pred_x, pred_y)
            gt_range = math.hypot(gt_x, gt_y)
            error_m = pred_range - gt_range

            self._runtime.record_sample(
                source=self.source_name,
                gt_range_m=gt_range,
                error_m=error_m,
                predicted_class_id=pred_class_id,
                ground_truth_class_id=gt_class_id,
            )
            sample_rows.append((self.source_name, gt_range, error_m, pred_class_id, gt_class_id))

        self._runtime.publish_sample_rows(sample_rows)

    def _nearest_gt_packet(self, target_stamp_sec: float) -> ConePacket | None:
        with self._cone_lock:
            if not self._cone_queue:
                return None
            while len(self._cone_queue) > 1 and self._cone_queue[1].stamp_sec < target_stamp_sec - self.cone_eval_sync_slop_sec:
                self._cone_queue.popleft()

            best_packet = None
            best_dt = None
            for packet in self._cone_queue:
                dt = abs(packet.stamp_sec - target_stamp_sec)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_packet = packet
            return best_packet

    def _iter_gt_points(self, msg: ConeArrayWithCovariance) -> list[tuple[str, float, float, float]]:
        points = []
        for cone in msg.blue_cones:
            points.append(('blue', float(cone.point.x), float(cone.point.y), float(cone.point.z)))
        for cone in msg.yellow_cones:
            points.append(('yellow', float(cone.point.x), float(cone.point.y), float(cone.point.z)))
        for cone in msg.orange_cones:
            points.append(('orange', float(cone.point.x), float(cone.point.y), float(cone.point.z)))
        for cone in msg.big_orange_cones:
            points.append(('big_orange', float(cone.point.x), float(cone.point.y), float(cone.point.z)))
        if self.cone_eval_include_unknown:
            for cone in msg.unknown_color_cones:
                points.append(('unknown', float(cone.point.x), float(cone.point.y), float(cone.point.z)))
        return points

    def _transform_gt_points(self, *, gt_points: list[tuple[str, float, float, float]], source_frame: str, target_frame: str, stamp):
        if source_frame == target_frame:
            return [(color, x, y) for color, x, y, _ in gt_points]

        transform = self._lookup_transform(target_frame=target_frame, source_frame=source_frame, stamp=stamp)
        if transform is None:
            return None

        out = []
        for color, x, y, z in gt_points:
            xx, yy, _zz = self._transform_point(transform, x, y, z)
            out.append((color, xx, yy))
        return out

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

    def _match_predictions_to_gt(self, *, predictions: list[tuple[float, float, int]], gt_points: list[tuple[str, float, float]]):
        candidates: list[tuple[float, int, int]] = []
        for pred_idx, (pred_x, pred_y, _pred_class) in enumerate(predictions):
            for gt_idx, (_gt_color, gt_x, gt_y) in enumerate(gt_points):
                dist = math.hypot(pred_x - gt_x, pred_y - gt_y)
                if dist <= self.eval_match_threshold_m:
                    candidates.append((dist, pred_idx, gt_idx))

        candidates.sort(key=lambda row: row[0])
        used_pred: set[int] = set()
        used_gt: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _dist, pred_idx, gt_idx in candidates:
            if pred_idx in used_pred or gt_idx in used_gt:
                continue
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
            matches.append((pred_idx, gt_idx))
        return matches

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
    def _stamp_msg_to_sec(stamp) -> float:
        return float(stamp.sec) + (float(stamp.nanosec) * 1e-9)

    @classmethod
    def _cone_class_name_to_id(cls, name: str) -> int:
        token = str(name).strip().lower().replace('-', '_').replace(' ', '_')
        if (
            'big_orange' in token
            or 'large_orange' in token
            or (('big' in token or 'large' in token) and 'orange' in token)
        ):
            token = 'big_orange'
        elif 'orange' in token:
            token = 'orange'
        elif 'yellow' in token:
            token = 'yellow'
        elif 'blue' in token:
            token = 'blue'
        else:
            token = 'unknown'
        return int(cls._CONE_CLASS_NAME_TO_ID[token])

    def destroy_node(self):
        self._runtime.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConeEvaluatorNode()
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
