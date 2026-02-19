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
import struct


def _f32(v: float) -> float:
    """Round-trip a float through float32 to match client precision."""
    return struct.unpack('<f', struct.pack('<f', v))[0]


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
        # Minimum angular velocity threshold — zero out when coasting below this
        # Client likely has a similar cutoff to prevent infinite drift from exponential decay
        self.ang_vel_cutoff = 0.01  # ~0.6 deg/s

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

        # 3. Zero angular velocity when below threshold (client does this to stop drift)
        # Without this, exponential decay never reaches zero and heading drifts forever
        if abs(torque) < 0.001 and abs(self._angular_velocity) < self.ang_vel_cutoff:
            self._angular_velocity = 0.0

        # Wrap to [-pi, pi]
        if self._heading > math.pi:
            self._heading -= 2 * math.pi * ((self._heading + math.pi) // (2 * math.pi))
        elif self._heading < -math.pi:
            self._heading += 2 * math.pi * ((-self._heading + math.pi) // (2 * math.pi))

    def step_f32(self, torque: float, dt: float):
        """Like step(), but quantize results to float32 after each substep.

        Matches the client's float32 storage for heading and angular_velocity.
        """
        self.step(torque, dt)
        self._heading = _f32(self._heading)
        self._angular_velocity = _f32(self._angular_velocity)

    def step_client_substeps(self, torque: float, frame_dt: float, use_f32: bool = False):
        """Step physics using the client's exact two-level substep algorithm.

        Matches decompile EXACTLY:
          Outer (GUESS5_GameSim_substep_update):
            - elapsed clamped to 550ms max
            - num_substeps = elapsed_ms // 110 + 1, capped at 5
            - per_substep = elapsed_ms // num_substeps (INTEGER division)
            - last substep gets the remainder
          Inner (GUESS4_Physics_substep_integrate):
            - inner_max = 0.04s (40ms)
            - if dt > 0.08s: inner_max = dt * 0.5
            - subtraction loop: step inner_max until remainder, last gets remainder

        Args:
            torque: pre-computed yaw torque (turn_adjust * raw_input)
            frame_dt: the client's actual frame duration in seconds
            use_f32: if True, quantize heading/ang_vel to float32 after each inner step
        """
        step_fn = self.step_f32 if use_f32 else self.step

        # Convert to integer ms, matching client's integer arithmetic
        elapsed_ms = int(frame_dt * 1000.0)

        # Clamp to 550ms max (0x226)
        if elapsed_ms > 550:
            elapsed_ms = 550

        # Outer substep count: elapsed_ms // 110 + 1, capped at 5
        outer_count = min(elapsed_ms // 110 + 1, 5)
        # Per-substep dt in integer ms (floor division)
        outer_dt_ms = elapsed_ms // outer_count
        remaining_ms = elapsed_ms

        for i in range(outer_count):
            # Last substep gets remainder, others get floor-divided dt
            if i == outer_count - 1:
                this_outer_ms = remaining_ms
            else:
                this_outer_ms = outer_dt_ms
                remaining_ms -= outer_dt_ms

            # Convert to seconds for inner substep
            this_outer_s = this_outer_ms / 1000.0

            # Inner substep (GUESS4_Physics_substep_integrate)
            # Subtraction loop with fixed inner_max, last step gets remainder
            inner_max_s = 0.04  # 40ms default
            if this_outer_s > 0.08:
                inner_max_s = this_outer_s * 0.5

            inner_remaining = this_outer_s
            while inner_remaining > 0.0:
                if inner_remaining <= inner_max_s:
                    step_fn(torque, inner_remaining)
                    inner_remaining = 0.0
                else:
                    step_fn(torque, inner_max_s)
                    inner_remaining -= inner_max_s

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
