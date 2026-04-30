# EUFS Models Code Map

This page maps the `documentation/eufs_models.md` behavior to the vehicle dynamics model library headers and source files.

## Primary Files

- `eufs_remastered/eufs_models/include/eufs_models/eufs_models.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/vehicle_model.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/dynamic_bicycle.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/point_mass.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/vehicle_param.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/vehicle_state.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/vehicle_input.hpp`
- `eufs_remastered/eufs_models/include/eufs_models/noise.hpp`

## Function Map

### Model Interface

- `VehicleModelBase` in `eufs_remastered/eufs_models/include/eufs_models/vehicle_model.hpp`: abstract base class defining the shared interface for all vehicle dynamics models; any concrete model must implement the integration step.
- `VehicleModelBase::updateState` (or equivalent pure virtual) in `eufs_remastered/eufs_models/include/eufs_models/vehicle_model.hpp`: overridden by each model to advance the vehicle state by one timestep given a control input.

### Dynamic Bicycle Model

- `DynamicBicycle` in `eufs_remastered/eufs_models/include/eufs_models/dynamic_bicycle.hpp`: concrete bicycle model class using Ackermann kinematics and tire force equations.
- `DynamicBicycle` implementation in `eufs_remastered/eufs_models/src/dynamic_bicycle.cpp`: ODE integration loop; reads vehicle parameters to compute lateral and longitudinal forces and advances position, velocity, and orientation.

### Point-Mass Model

- `PointMass` in `eufs_remastered/eufs_models/include/eufs_models/point_mass.hpp`: simplified dynamics model treating the vehicle as a single mass.
- `PointMass` implementation in `eufs_remastered/eufs_models/src/point_mass.cpp`: integration logic for the point-mass equations of motion.

### Vehicle Parameters

- `VehicleParam` in `eufs_remastered/eufs_models/include/eufs_models/vehicle_param.hpp`: YAML-based parameter struct; loading function populates inertia, kinematics, tire, aerodynamics, dynamics, and input-range sections from the configured file path.

### State And Input

- `VehicleState` in `eufs_remastered/eufs_models/include/eufs_models/vehicle_state.hpp`: container for full vehicle state (position, orientation, linear/angular velocity, wheel speeds, steering); updated in-place by each integration step.
- `Input` in `eufs_remastered/eufs_models/include/eufs_models/vehicle_input.hpp`: control input struct holding throttle, braking, and steering values consumed by model integration.

### Noise Model

- `NoiseModel` (or equivalent) in `eufs_remastered/eufs_models/include/eufs_models/noise.hpp`: applies Gaussian noise to vehicle state components before publication; standard deviations are loaded from `eufs_remastered/eufs_models/config/noise.yaml`.

### Umbrella Header

- `eufs_remastered/eufs_models/include/eufs_models/eufs_models.hpp`: convenience header that includes all model types; `eufs_gz_dynamics` includes this single header to access the full library.

## Related Entry Points

- `eufs_remastered/eufs_models/config/noise.yaml`: noise standard deviation values for each state component (all currently zero).
- `eufs_remastered/eufs_models/CMakeLists.txt`: build configuration; exports the library for use by `eufs_gz_dynamics`.
