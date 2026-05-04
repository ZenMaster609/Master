"""RViz marker construction for the tracked-cone planner runtime."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from sim_car.cones.tracking.fusion import normalize_color
from sim_car.planning.planning_state_machine import _OPERATOR_STATE_COLORS
from sim_car.planning.triangulation_planner_core import CoreResult


class VisualizationMixin:
    """RViz marker construction and cone visualization helpers for TrackedConePlannerRuntime."""

    def _build_markers(
        self,
        *,
        now,
        frame_id: str,
        result: Optional[CoreResult],
        centerline: np.ndarray,
        raw_centerline: np.ndarray,
        raw_midpoint_chain: np.ndarray,
        status: str,
        operator_state: str,
        control_target_frame: Optional[np.ndarray],
    ) -> MarkerArray:
        arr = MarkerArray()

        clear = Marker()
        clear.header.frame_id = frame_id
        clear.header.stamp = now
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        marker_id = 1
        if result is None:
            marker_id = self._append_status_marker(
                arr,
                marker_id,
                frame_id,
                now,
                status,
                operator_state=operator_state,
            )
            return arr

        if self.show_raw_cones:
            marker_id = self._append_remembered_cone_marker(
                arr,
                marker_id,
                frame_id,
                now,
            )

        if self.show_triangulation_edges:
            arr.markers.append(
                self._make_edge_list_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns='triangulation_edges',
                    points=result.filtered_points,
                    edges=result.triangulation_edges,
                    color=(0.3, 0.7, 1.0, 0.35),
                    width=0.03,
                )
            )
            marker_id += 1

        if self.show_candidate_edges:
            arr.markers.append(
                self._make_edge_list_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns='candidate_cross_edges',
                    points=result.filtered_points,
                    edges=result.candidate_edges,
                    color=(1.0, 0.6, 0.1, 0.8),
                    width=0.06,
                )
            )
            marker_id += 1

        if self.show_selected_edges:
            arr.markers.append(
                self._make_edge_list_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns='selected_cross_edges',
                    points=result.filtered_points,
                    edges=result.selected_edges,
                    color=(0.2, 1.0, 0.3, 0.95),
                    width=0.08,
                )
            )
            marker_id += 1

        if self.show_raw_midpoint_chain:
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns='raw_midpoint_chain',
                    points=raw_midpoint_chain,
                    color=(1.0, 1.0, 1.0, 0.95),
                    width=0.06,
                    z_offset=0.03,
                )
            )
            marker_id += 1

        if self.show_raw_prevalidation_centerline:
            arr.markers.append(
                self._make_line_strip_marker(
                    frame_id=frame_id,
                    stamp=now,
                    marker_id=marker_id,
                    ns='raw_prevalidation_centerline',
                    points=raw_centerline,
                    color=(1.0, 0.15, 0.85, 0.9),
                    width=0.07,
                    z_offset=0.05,
                )
            )
            marker_id += 1

        arr.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns='boundary_left',
                points=result.left_boundary,
                color=(0.2, 0.45, 1.0, 0.95),
                width=0.07,
                z_offset=0.02,
            )
        )
        marker_id += 1

        arr.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns='boundary_right',
                points=result.right_boundary,
                color=(1.0, 0.9, 0.2, 0.95),
                width=0.07,
                z_offset=0.02,
            )
        )
        marker_id += 1

        arr.markers.append(
            self._make_line_strip_marker(
                frame_id=frame_id,
                stamp=now,
                marker_id=marker_id,
                ns='centerline',
                points=centerline,
                color=(0.95, 0.15, 0.15, 1.0),
                width=0.09,
                z_offset=0.07,
            )
        )
        marker_id += 1

        if self.show_lookahead_point and control_target_frame is not None:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = now
            marker.ns = 'lookahead'
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.position.x = float(control_target_frame[0])
            marker.pose.position.y = float(control_target_frame[1])
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            arr.markers.append(marker)
            marker_id += 1

        self._append_status_marker(
            arr,
            marker_id,
            frame_id,
            now,
            status,
            operator_state=operator_state,
        )
        return arr

    @staticmethod
    def _path_point_yaw(path_xy: np.ndarray, idx: int) -> float:
        if path_xy.shape[0] < 2:
            return 0.0
        if idx == path_xy.shape[0] - 1:
            dx, dy = path_xy[idx] - path_xy[idx - 1]
        else:
            dx, dy = path_xy[idx + 1] - path_xy[idx]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return float(math.atan2(dy, dx))

    @staticmethod
    def _make_points_marker(
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        color: tuple[float, float, float, float],
        scale: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.pose.orientation.w = 1.0
        for x, y in points:
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = 0.03
            marker.points.append(pt)
        return marker

    def _update_remembered_cone_viz(
        self,
        *,
        points_xy: np.ndarray,
        colors: list[str],
    ) -> None:
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            self._remembered_cone_viz_points = np.empty((0, 2), dtype=np.float64)
            self._remembered_cone_viz_colors = []
            return

        normalized_colors = [normalize_color(color) for color in colors]
        if len(normalized_colors) < points.shape[0]:
            normalized_colors.extend(['unknown'] * (points.shape[0] - len(normalized_colors)))
        elif len(normalized_colors) > points.shape[0]:
            normalized_colors = normalized_colors[: points.shape[0]]

        finite_mask = np.all(np.isfinite(points), axis=1)
        self._remembered_cone_viz_points = np.array(points[finite_mask], copy=True)
        self._remembered_cone_viz_colors = [
            color
            for color, keep in zip(normalized_colors, finite_mask)
            if bool(keep)
        ]

    def _append_remembered_cone_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
    ) -> int:
        points = np.asarray(
            getattr(self, '_remembered_cone_viz_points', np.empty((0, 2), dtype=np.float64)),
            dtype=np.float64,
        )
        colors = list(getattr(self, '_remembered_cone_viz_colors', []))
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            return marker_id
        if len(colors) < points.shape[0]:
            colors.extend(['unknown'] * (points.shape[0] - len(colors)))
        elif len(colors) > points.shape[0]:
            colors = colors[: points.shape[0]]

        markers.markers.append(
            self._make_colored_points_marker(
                frame_id=frame_id,
                stamp=stamp,
                marker_id=marker_id,
                ns='remembered_cones',
                points=points,
                colors=colors,
                scale=0.18,
            )
        )
        return marker_id + 1

    @classmethod
    def _make_colored_points_marker(
        cls,
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        colors: list[str],
        scale: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        for (x, y), color_name in zip(np.asarray(points, dtype=np.float64), colors):
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = 0.03
            marker.points.append(pt)

            rgba = ColorRGBA()
            rgba.r, rgba.g, rgba.b, rgba.a = cls._cone_marker_rgba(color_name)
            marker.colors.append(rgba)
        return marker

    @staticmethod
    def _cone_marker_rgba(color_name: str) -> tuple[float, float, float, float]:
        color = normalize_color(color_name)
        if color == 'blue':
            return 0.2, 0.55, 1.0, 0.95
        if color == 'yellow':
            return 1.0, 0.92, 0.25, 0.95
        if color == 'orange':
            return 1.0, 0.55, 0.15, 0.95
        return 0.8, 0.8, 0.8, 0.80

    @staticmethod
    def _make_edge_list_marker(
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        edges: np.ndarray,
        color: tuple[float, float, float, float],
        width: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.pose.orientation.w = 1.0

        for edge in edges:
            a = int(edge[0])
            b = int(edge[1])
            if a < 0 or b < 0 or a >= points.shape[0] or b >= points.shape[0]:
                continue
            p0 = Point()
            p0.x = float(points[a, 0])
            p0.y = float(points[a, 1])
            p0.z = 0.02
            p1 = Point()
            p1.x = float(points[b, 0])
            p1.y = float(points[b, 1])
            p1.z = 0.02
            marker.points.append(p0)
            marker.points.append(p1)
        return marker

    @staticmethod
    def _make_line_strip_marker(
        *,
        frame_id: str,
        stamp,
        marker_id: int,
        ns: str,
        points: np.ndarray,
        color: tuple[float, float, float, float],
        width: float,
        z_offset: float = 0.02,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.pose.orientation.w = 1.0
        for x, y in points:
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = float(z_offset)
            marker.points.append(pt)
        return marker

    @staticmethod
    def _append_status_marker(
        markers: MarkerArray,
        marker_id: int,
        frame_id: str,
        stamp,
        status: str,
        *,
        operator_state: str,
    ) -> int:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'status'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 0.32
        marker.color.a = 1.0
        color = _OPERATOR_STATE_COLORS.get(operator_state, _OPERATOR_STATE_COLORS['waiting'])
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.pose.position.x = 1.5
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.35
        marker.pose.orientation.w = 1.0
        marker.text = status
        markers.markers.append(marker)
        return marker_id + 1
