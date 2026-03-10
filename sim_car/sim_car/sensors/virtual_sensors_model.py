"""
Shared virtual sensor models for cooling, brakes, and pitot.

ROS-agnostic so it can be reused by multiple nodes.
"""

import math
import random


class VirtualSensorsModel:
    """Stateful model for virtual sensors."""

    AIR_DENSITY = 1.225

    def __init__(self, ambient_temp: float = 25.0):
        self.ambient_temp = float(ambient_temp)

        # Inputs
        self.vehicle_speed = 0.0  # m/s
        self.throttle = 0.0  # 0..1
        self.braking = 0.0  # 0..1 (proxy from cmd_vel)
        self.brake_cmd = 0.0  # 0..1 explicit

        # State
        self.prev_speed = 0.0
        self.water_temp_in = self.ambient_temp
        self.water_temp_out = self.ambient_temp
        self.water_temp_radiator = self.ambient_temp
        self.brake_temp_fr = self.ambient_temp
        self.brake_temp_rl = self.ambient_temp

        self.last_update_time = None

    def update_vehicle_speed(self, vx: float, vy: float) -> None:
        self.vehicle_speed = math.sqrt(vx * vx + vy * vy)

    def update_cmd_vel(self, linear_x: float) -> None:
        if linear_x > 0:
            self.throttle = min(linear_x / 10.0, 1.0)
            self.braking = 0.0
        else:
            self.throttle = 0.0
            self.braking = min(abs(linear_x) / 5.0, 1.0)

    def update_brake_cmd(self, brake_cmd: float) -> None:
        self.brake_cmd = max(0.0, min(1.0, float(brake_cmd)))

    def step(self, now_sec: float, default_dt: float) -> None:
        if self.last_update_time is None:
            dt = default_dt
        else:
            dt = now_sec - self.last_update_time
        self.last_update_time = now_sec

        if dt < 0.0:
            dt = default_dt
        dt = min(dt, 0.5)

        self._update_cooling_system(dt)
        self._update_brake_temps(dt)
        self.prev_speed = self.vehicle_speed

    def compute_water_pressure(self) -> float:
        base_pressure = 1.5  # bar
        speed_factor = 0.05 * self.vehicle_speed
        throttle_factor = 0.3 * self.throttle
        return base_pressure + speed_factor + throttle_factor

    def compute_water_flow(self) -> float:
        base_flow = 30.0  # L/min
        speed_factor = 2.0 * self.vehicle_speed
        throttle_factor = 15.0 * self.throttle
        return base_flow + speed_factor + throttle_factor

    def compute_pitot_pressure(self) -> float:
        q = 0.5 * self.AIR_DENSITY * self.vehicle_speed ** 2
        if self.vehicle_speed < 1.0:
            q += random.uniform(-2.0, 2.0)
        return max(0.0, q)

    def _update_cooling_system(self, dt: float) -> None:
        tau_heat = 30.0
        tau_cool = 60.0

        heat_generation = self.throttle * self.vehicle_speed * 2.0
        radiator_cooling = 0.1 * self.vehicle_speed * (self.water_temp_radiator - self.ambient_temp)
        ambient_cooling_rate = 0.01

        self.water_temp_out += dt * (
            heat_generation
            - (self.water_temp_out - self.water_temp_in) / tau_cool
        )
        self.water_temp_out = max(self.ambient_temp, min(120.0, self.water_temp_out))

        self.water_temp_radiator += dt * (
            (self.water_temp_out - self.water_temp_radiator) / tau_heat
            - radiator_cooling / 10.0
            - ambient_cooling_rate * (self.water_temp_radiator - self.ambient_temp)
        )
        self.water_temp_radiator = max(self.ambient_temp, min(110.0, self.water_temp_radiator))

        self.water_temp_in += dt * (
            (self.water_temp_radiator - self.water_temp_in) / tau_heat
            - ambient_cooling_rate * (self.water_temp_in - self.ambient_temp)
        )
        self.water_temp_in = max(self.ambient_temp, min(100.0, self.water_temp_in))

    def _update_brake_temps(self, dt: float) -> None:
        brake_input = max(self.brake_cmd, self.braking)
        decel = max(0.0, (self.prev_speed - self.vehicle_speed) / max(dt, 0.001))
        speed_for_heating = max(self.vehicle_speed, self.prev_speed)
        effective_brake = brake_input if (decel > 0.0 and speed_for_heating > 0.2) else 0.0

        heat_fr = speed_for_heating * decel * 0.5 * effective_brake
        heat_rl = speed_for_heating * decel * 0.3 * effective_brake

        heat_fr += effective_brake * 30.0
        heat_rl += effective_brake * 20.0

        airflow_cooling = 0.02 * self.vehicle_speed
        ambient_cooling = 0.005

        self.brake_temp_fr += dt * (
            heat_fr
            - airflow_cooling * (self.brake_temp_fr - self.ambient_temp)
            - ambient_cooling * (self.brake_temp_fr - self.ambient_temp)
        )
        self.brake_temp_fr = max(self.ambient_temp, min(400.0, self.brake_temp_fr))

        self.brake_temp_rl += dt * (
            heat_rl
            - airflow_cooling * (self.brake_temp_rl - self.ambient_temp)
            - ambient_cooling * (self.brake_temp_rl - self.ambient_temp)
        )
        self.brake_temp_rl = max(self.ambient_temp, min(400.0, self.brake_temp_rl))
