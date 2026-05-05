"""Shared defaults for the cone-memory to planner pipeline."""

from __future__ import annotations

ODOM_FRAME_DEFAULT = "odom"  # Shared odometry frame used by simulation and planners.
BASE_FRAME_DEFAULT = "front_axle"  # Controller geometry is referenced at the front axle.
TRACKED_CONES_TOPIC_DEFAULT = "/tracked_cones"  # Cone memory publishes the planner input here.
ODOM_TOPIC_DEFAULT = "/sim/odom"  # Simulation odometry topic used by cone memory and planners.
WHEELBASE_M_DEFAULT = 1.65  # Meters; matches the EUFS vehicle kinematic wheelbase.
TF_TIMEOUT_S_DEFAULT = 0.03  # Seconds; short enough for realtime planning without hiding TF issues.
PLANNER_MAX_RANGE_M_DEFAULT = 25.0  # Meters; matches useful lidar range in the default sim setup.
PLANNER_MIN_CONFIDENCE_DEFAULT = 0.3  # Unitless; filters noisy tracks while keeping sparse layouts.
