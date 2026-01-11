# Run session identifier for multi-machine synchronization
#
# Published by the data source machine (e.g., Jetson) at session start.
# Subscriber machines (e.g., Windows plotter) use this to sync run_id.

# Unique session identifier (format: YYYY-MM-DD_HH-MM-SS)
string run_id

# Base path for multidata storage (informational)
string base_path

# Hostname of the machine that generated this run_id
string originator_hostname

# ROS_DOMAIN_ID for verification
uint32 ros_domain_id

# Session start time
builtin_interfaces/Time start_time
