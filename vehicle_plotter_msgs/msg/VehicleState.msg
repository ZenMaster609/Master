# VehicleState.msg
# Canonical vehicle state representation for plotting, logging, and EKF integration
#
# This message provides a unified state that all sensor adapters produce
# and all consumers (plotter, logger, EKF) expect.

# Header with timestamp and frame
std_msgs/Header header

# Position (local frame, meters)
# Origin is set at GPS origin or simulation start position
float64 x
float64 y

# Velocity (body frame, m/s)
# vx: forward velocity, vy: lateral velocity
float64 vx
float64 vy

# Orientation
float64 yaw           # radians, range: -pi to pi
float64 yaw_rate      # rad/s, positive = counter-clockwise

# Raw odometry (unprocessed /sim/raw/odom)
float64 raw_x
float64 raw_y
float64 raw_vx
float64 raw_vy
float64 raw_yaw
float64 raw_speed

# IMU-derived orientation (yaw) and dead-reckoned velocity
float64 imu_vx         # m/s, IMU-only integration
float64 imu_vy         # m/s, IMU-only integration
float64 imu_yaw        # radians, yaw from IMU orientation

# Derived quantities
float64 speed                 # sqrt(vx^2 + vy^2), m/s
float64 distance_traveled     # cumulative distance, meters

# Slip estimates (dimensionless ratios, optional)
# slip_longitudinal: (wheel_vel - body_vel) / max(wheel_vel, body_vel)
# slip_lateral: atan2(vy, vx) - sideslip angle
float64 slip_longitudinal
float64 slip_lateral

# Wheel encoder speeds [FL, FR, RL, RR] in RPM
float32[4] encoder_velocities

# Wheel encoder linear speeds [FL, FR, RL, RR] in mm/s
float32[4] encoder_speeds_mm_s

# Wheel encoder angle accumulation [FL, FR, RL, RR] in radians (per RPM window)
float32[4] encoder_angle_accum

# GPS data (raw, for reference)
float64 gps_latitude      # degrees
float64 gps_longitude     # degrees
float64 gps_altitude      # meters above WGS84 ellipsoid
bool gps_valid            # true if GPS fix is valid

# GPS converted to local coordinates (same frame as x,y)
float64 gps_local_x       # meters, local frame
float64 gps_local_y       # meters, local frame

# INS fused position (explicit naming, same as x,y when from INS)
float64 ins_x             # meters, local frame
float64 ins_y             # meters, local frame

# Dead-reckoning position (IMU-only integration, no GPS corrections)
float64 dr_x              # meters, local frame
float64 dr_y              # meters, local frame

# Suspension displacements [FL, FR, RL, RR] in meters
float32[4] suspension

# Steering angle (degrees)
float32 steering_angle
bool steering_valid

# Cooling system
float32 water_pressure        # bar
float32 water_flow            # L/min
float32 water_temp_in         # celsius
float32 water_temp_out        # celsius
float32 water_temp_radiator   # celsius

# Brake temperatures
float32 brake_temp_fr         # celsius
float32 brake_temp_rl         # celsius

# Pitot tube
float32 pitot_dynamic_pressure  # Pa

# EKF/UKF readiness fields
# covariance: 6x6 state covariance matrix (x, y, vx, vy, yaw, yaw_rate)
# Stored as row-major: [P00, P01, ..., P05, P10, P11, ..., P55]
float64[36] covariance

# Estimation status
# "raw" = direct sensor measurements, no filtering
# "predicted" = EKF prediction step only
# "filtered" = EKF update step applied
string estimation_status

# Source adapter identifier (for debugging)
string source_adapter
