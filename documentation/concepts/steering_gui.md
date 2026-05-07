# Steering GUI

`steering_gui` is an RQT plugin that provides a manual steering control panel for the simulated vehicle. It is intended for development and debugging: a developer can drive the simulated car by hand to test perception, planning, or control behavior without writing a custom command publisher.

## Purpose

During simulation development it is often useful to manually steer the vehicle to put it in specific configurations or to quickly verify that the control stack is receiving and responding to commands. The steering GUI provides a graphical slider interface for this purpose inside the standard RQT tool window.

## Interface

The plugin adds a dockable RQT panel containing:

- **Steering slider**: controls the steering angle sent to the vehicle. The range and resolution match the configured Ackermann steering limits.
- **Keyboard shortcuts**: allow quick control from the keyboard without interacting with the slider directly.
- **Topic configuration**: the command topic can be changed at runtime to target different namespaces or vehicle instances.

## Output Topics

The GUI publishes two topics:

- **Ackermann drive command** (`AckermannDriveStamped`): the primary steering and speed command. Published to the configured command topic (default matches the `eufs_gz_dynamics` subscription).
- **Brake command** (`Float32`): brake value published to a separate brake topic. Allows the GUI to command braking independently of the drive command.

## Feedback

The plugin subscribes to the vehicle state topic to display current speed or steering angle as feedback. This lets the operator confirm that commands are being received and that the vehicle is responding.

## Usage

The steering GUI is launched as part of the standard RQT session:

```bash
rqt --standalone steering_gui
```

Or it can be added as a panel in a running RQT window via Plugins menu.

## Useful Commands

Build the plugin:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select steering_gui
```

Source the workspace:

```bash
cd ~/ros2_ws && source install/setup.bash
```

Launch as a standalone RQT panel:

```bash
rqt --standalone steering_gui
```
