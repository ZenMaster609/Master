# eufs_msgs

This package is intentionally trimmed to the EUFS interfaces used by the
remastered simulation stack.

## Messages

| Name | Description |
| ---- | ---- |
| ConeArrayWithCovariance.msg | Ground-truth cone arrays published by the Gazebo cone plugin. |
| ConeWithCovariance.msg | A cone point and covariance used inside `ConeArrayWithCovariance`. |
| WheelSpeeds.msg | Wheel-speed helper type required by `eufs_models`. |
