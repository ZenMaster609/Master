# Custom Messages

## vehicle_plotter_msgs

### `vehicle_plotter_msgs/msg/VehicleState.msg`

- Canonical vehicle state used by data collection, plotting, and logging.
- Fields include position, velocity, orientation, wheel encoder data, suspension, steering, cooling, brakes, pitot, GPS/INS, covariance, and source adapter.

### `vehicle_plotter_msgs/msg/RunSession.msg`

- Session metadata for synchronizing logging and plotting across machines.
- Fields: `run_id`, `base_path`, `originator_hostname`, `ros_domain_id`, `start_time`.

## eufs_msgs

- Message definitions used by EUFS tooling. These are not authored in this repository but are included for compatibility with EUFS simulation components.
- Message directory: `eufs_msgs/msg/`.
- Action directory: `eufs_msgs/action/`.
