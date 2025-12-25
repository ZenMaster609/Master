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

# Derived quantities
float64 speed                 # sqrt(vx^2 + vy^2), m/s
float64 distance_traveled     # cumulative distance, meters

# Slip estimates (dimensionless ratios, optional)
# slip_longitudinal: (wheel_vel - body_vel) / max(wheel_vel, body_vel)
# slip_lateral: atan2(vy, vx) - sideslip angle
float64 slip_longitudinal
float64 slip_lateral

# Wheel encoders [FL, FR, RL, RR]
int32[4] encoder_ticks
float32[4] encoder_velocities   # m/s at wheel circumference

# GPS data (raw, for reference)
float64 gps_latitude      # degrees
float64 gps_longitude     # degrees
float64 gps_altitude      # meters above WGS84 ellipsoid
bool gps_valid            # true if GPS fix is valid

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
