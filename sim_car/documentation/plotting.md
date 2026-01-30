# Plotting (vehicle_plotter)

This document describes the plotting pipeline and where to configure plots.
Plotting is driven by `vehicle_plotter` and consumes `/vehicle_plotter/state`.

---

## Pipeline

1. **Gazebo + sim_car sensors** publish `/sim/*` topics.
2. **vehicle_plotter data_collector** subscribes to `/sim/*` and publishes `/vehicle_plotter/state`.
3. **vehicle_plotter plotter** renders all plots in a single window.

The plotter window layout is defined in:
- `vehicle_plotter/vehicle_plotter/plotting/plot_config.py` (`get_all_plots()`)

---

## Running the Plotter

Build and source:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select vehicle_plotter
cd ~/ros2_ws && source install/setup.bash
```

Launch via the full sim bringup:

```bash
cd ~/ros2_ws && ros2 launch sim_car full_sim_launch.launch.py
```

Or launch the plotter directly:

```bash
cd ~/ros2_ws && ros2 launch vehicle_plotter plotter.launch.py
```

---

## Editing Plots

Update `vehicle_plotter/vehicle_plotter/plotting/plot_config.py`:
- Add/remove plots in `get_all_plots()`.
- Each plot uses a `VehicleState` field as the series source.
- Use `plot_type="timeseries"` for time-series plots, or `plot_type="xy"` for trajectories.

After changes:

```bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select vehicle_plotter
cd ~/ros2_ws && source install/setup.bash
```

---

## Notes

- The plotter subscribes to `/vehicle_plotter/state` only.
- If a field is missing, ensure the Gazebo adapter maps the topic and the field exists in `VehicleState`.
