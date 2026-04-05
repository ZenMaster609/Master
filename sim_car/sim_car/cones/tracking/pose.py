"""Shared planar pose helpers for cone tracking and reprojection."""

from __future__ import annotations

import math
from typing import Callable, Optional

PlanarPose = tuple[float, float, float]
FrameAliasMatcher = Callable[[str, str], bool]


def _frame_token(frame: str) -> tuple[str, str]:
    normalized = str(frame).strip().strip("/").lower()
    leaf = normalized.split("/")[-1] if normalized else ""
    return normalized, leaf


def odom_point_to_base(x_odom: float, y_odom: float, pose: PlanarPose) -> tuple[float, float]:
    tx, ty, yaw = pose
    dx = x_odom - tx
    dy = y_odom - ty
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x_base = cos_y * dx + sin_y * dy
    y_base = -sin_y * dx + cos_y * dy
    return x_base, y_base


def base_point_to_odom(
    x_base: float,
    y_base: float,
    z_base: float,
    pose: Optional[PlanarPose],
) -> tuple[float, float, float]:
    if pose is None:
        return x_base, y_base, z_base
    tx, ty, yaw = pose
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x_odom = tx + cos_y * x_base - sin_y * y_base
    y_odom = ty + sin_y * x_base + cos_y * y_base
    return x_odom, y_odom, z_base


def convert_odom_child_pose_to_base_frame(
    *,
    child_frame: str,
    base_frame: str,
    tx: float,
    ty: float,
    yaw: float,
    wheelbase_m: float,
    is_alias: FrameAliasMatcher,
) -> Optional[PlanarPose]:
    child_token, child_leaf = _frame_token(child_frame)
    base_token, base_leaf = _frame_token(base_frame)

    child_is_body_center = child_leaf in {"base_footprint", "base_link"}
    base_is_body_center = base_leaf in {"base_footprint", "base_link"}
    child_is_front_axle = child_leaf == "front_axle"
    base_is_front_axle = base_leaf == "front_axle"

    if child_token and child_token == base_token:
        return tx, ty, yaw

    if child_is_body_center and base_is_body_center:
        return tx, ty, yaw

    if child_is_front_axle and base_is_front_axle:
        return tx, ty, yaw

    if child_is_body_center and base_is_front_axle:
        x_base, y_base, _z_base = base_point_to_odom(0.5 * wheelbase_m, 0.0, 0.0, (tx, ty, yaw))
        return x_base, y_base, yaw

    if child_is_front_axle and base_is_body_center:
        x_base, y_base, _z_base = base_point_to_odom(-0.5 * wheelbase_m, 0.0, 0.0, (tx, ty, yaw))
        return x_base, y_base, yaw

    if is_alias(child_frame, base_frame):
        return tx, ty, yaw

    return None
