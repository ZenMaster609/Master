# Steering GUI Code Map

This page maps the `documentation/concepts/steering_gui.md` behavior to the RQT plugin implementation.

## Primary Files

- `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`

## Function Map

### Plugin Initialization

- `EUFSRobotSteeringGUI` in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: main RQT plugin class; sets up the Qt widget, creates publishers and subscriptions, and wires slider signals to command publishing.
- `EUFSRobotSteeringGUI.__init__` in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: initializes the plugin, loads the UI, and sets default parameter values for command topic and steering range.

### Command Publishing

- `EUFSRobotSteeringGUI._publish_command` (or equivalent slot) in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: builds an `AckermannDriveStamped` message from the current slider value and publishes it to the configured drive command topic.
- `EUFSRobotSteeringGUI._publish_brake` (or equivalent) in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: publishes a `Float32` brake value to the separate brake command topic.

### Keyboard Shortcuts

- Keyboard event handlers in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: intercept key presses to allow steering adjustment without using the slider directly; typically increment/decrement the steering angle by a fixed step.

### Vehicle State Feedback

- State subscription callback in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: subscribes to the vehicle state topic and updates displayed feedback values (speed, steering angle) in the GUI panel.

### Topic Configuration

- Topic name handling in `eufs_remastered/steering_gui/src/steering_gui/EUFSRobotSteeringGUI.py`: allows the operator to change the drive command topic at runtime so the GUI can target different vehicle namespaces.

## Related Entry Points

- `eufs_remastered/steering_gui/package.xml` and `setup.py`: register the plugin with the RQT plugin system so it appears in the Plugins menu.
- `eufs_remastered/steering_gui/plugin.xml`: RQT plugin descriptor file mapping the GUI class to the plugin entry point.
- `eufs_remastered/eufs_gz_dynamics/src/eufs_gz_dynamics.cpp`: subscribes to the `AckermannDriveStamped` topic published by the GUI; the topic name must match.
