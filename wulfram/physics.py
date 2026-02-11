"""
Direct-impulse physics model for vehicle heading.

The Wulfram2 client uses a direct torque-impulse system for tank yaw rotation,
NOT springs. The spring system only handles suspension and pitch/roll terrain
following — it explicitly ZEROES yaw torque output (Physics.c:2584).

Per-frame cycle:
  1. Torque accumulator zeroed (entity[0x48-0x50] cleared after previous frame)
  2. Yaw torque = lateral_mobility * turn_adjust * raw_input  (Vehicles.c:1193)
     - raw_input = -button_normalized(1), clamped [-1,1]  (no piecewise curve!)
  3. Physics substep (Physics.c:4502-4562, damped mode):
     a. heading += angular_velocity * dt  (explicit Euler, heading first)
     b. angular_velocity += (torque - angular_velocity * damp_coeff) * dt
  4. Torque accumulator zeroed for next frame (Physics.c:6169-6174)

At steady state: angular_velocity = torque / damp_coeff

Client references:
  - Vehicles.c:1193: entity[0x50] += lateral_mobility * turn_adjust * raw_input
  - Vehicles.c:1229: dampening = turn_rate * 1.4137167 (stored but not the damp_coeff)
  - Physics.c:4502-4562: Physics_substep_integrate (substep with damped mode)
  - Physics.c:4414-4494: Matrix3_integrate_angular (orientation + ang_vel update)
  - Physics.c:6169-6174: entity[0x48-0x50] zeroed after physics step
  - Physics.c:2582-2584: Spring system zeroes yaw torque output
"""

import math


class VehiclePhysics:
    """Per-vehicle direct-impulse physics for heading.

    Matches the client's actual yaw physics pipeline:
      torque = lateral_mobility * turn_adjust * raw_input  (per frame)
      heading += angular_velocity * dt  (explicit Euler)
      angular_velocity += (torque - angular_velocity * damp_coeff) * dt

    Parameters are tunable at runtime via control commands.
    """

    def __init__(self, damp_coeff: float = 1.0):
        self._heading = 0.0
        self._angular_velocity = 0.0

        # Damping coefficient for angular velocity.
        # From Physics_substep_integrate damped mode: entity->0xbc->+4->+0x7c
        # At steady state: ang_vel_ss = torque / damp_coeff
        # With torque=4.5 and damp_coeff=1.0: ang_vel_ss = 4.5 rad/s (258 deg/s)
        self.damp_coeff = damp_coeff

    def step(self, torque: float, dt: float):
        """Advance physics by dt seconds.

        torque: pre-computed yaw torque = lateral_mobility * turn_adjust * raw_input.
                This mimics entity[0x50] which is zeroed each frame and refilled.
        dt: timestep in seconds (1/tick_rate, typically 1/30).

        Integration order matches client (explicit Euler):
          Matrix3_integrate_angular rotates orientation by ang_vel*dt FIRST,
          then updates ang_vel by (torque - ang_vel * damp_coeff) * dt.
        """
        # 1. Integrate heading with CURRENT angular velocity (before update)
        self._heading += self._angular_velocity * dt

        # 2. Update angular velocity with damping
        # Physics_substep_integrate: effective = torque - ang_vel * damp_coeff
        # Matrix3_integrate_angular: ang_vel += effective * dt
        self._angular_velocity += (torque - self._angular_velocity * self.damp_coeff) * dt

        # Wrap to [-pi, pi]
        if self._heading > math.pi:
            self._heading -= 2 * math.pi * ((self._heading + math.pi) // (2 * math.pi))
        elif self._heading < -math.pi:
            self._heading += 2 * math.pi * ((-self._heading + math.pi) // (2 * math.pi))

    @property
    def heading(self) -> float:
        return self._heading

    @heading.setter
    def heading(self, val: float):
        self._heading = val

    @property
    def angular_velocity(self) -> float:
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, val: float):
        self._angular_velocity = val

    def reset(self):
        """Reset all state to zero."""
        self._heading = 0.0
        self._angular_velocity = 0.0
