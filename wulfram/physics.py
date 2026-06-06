"""
Rotation-matrix physics model for vehicle heading.

Matches the Wulfram2 client's EXACT angular integration pipeline from the azurefishy
decompile. The client uses a 3x3 rotation matrix (stored as doubles) with axis-angle
Rodrigues construction, float32 matrix multiply, and atan2 euler extraction.

Per-frame cycle (from decompile):
  1. Torque accumulator zeroed (entity[0x48-0x50] cleared after previous frame)
  2. Yaw torque = turn_mobility * (float)turn_adjust * yaw_axis
     (TankVehicle_apply_physics, Vehicles.c:1336; torque add pattern Vehicles.c:1425)
     - turn_adjust read as double, cast to float32 BEFORE multiply
  3. Entity_get_rotation_matrix: dirty-check euler vs prev_euler, rebuild if changed
     (GUESS6_Entity_get_rotation_matrix, Entities/Core.c:656)
  4. Physics_substep_integrate_angular (Physics.c:5264):
     - Substep loop: max 40ms or dt*0.5 if dt > 80ms (Physics.c:5298-5300)
     - Matrix3_integrate_angular per substep (Physics.c:5169):
       a. scaled_omega = ang_vel * (float)dt
       b. delta_R = Matrix3_from_axis_angle(scaled_omega)  [Rodrigues, float32]
       c. new_R = delta_R * old_R  [9 dot products, float32 intermediates]
       d. Store new_R as doubles
       e. Extract euler angles via atan2
       f. Normalize euler to [0, 2*pi] via iterative add/subtract
       g. ang_vel += effective_accel * (float)dt
  5. Save euler to prev_euler cache (entity+0xA0)
  6. Torque accumulator zeroed for next frame

Client references (azurefishy-src file:line; verified 2026-06-01):
  - GUESS3_Matrix3_from_axis_angle      System/Core/Math.c:2774   Rodrigues formula, float32
  - GUESS5_Matrix3_integrate_angular    Game/Simulation/Physics.c:5169  Full angular integration
  - GUESS5_Physics_substep_integrate_angular  Game/Simulation/Physics.c:5264  Substep loop + damping
  - GUESS6_GameSim_substep_update       Game/Simulation/Physics.c:1974  Outer frame-delta splitter
  - GUESS5_Matrix3_from_euler_xyz       System/Core/Math.c:606    Euler→matrix rebuild
  - GUESS3_Math_normalize_angle_radians System/Core/Math.c:2744   Iterative [0, 2*pi]
  - GUESS2_Vec3_normalize_safe          System/Core/Math.c:2358   Euler extraction via atan2
  - GUESS6_Entity_get_rotation_matrix   Game/Simulation/Entities/Core.c:656  Dirty-check matrix accessor

NOTE: Earlier docstrings cited raw Ghidra addresses (0x004f1150, ...) and line numbers
(28804-28877, 25514-25574) from a superseded flat decompile dump no longer in the tree.
Those have been repointed to the organized azurefishy-src source above.
"""

import math

# Single shared sim kernel (CH1): the rotation/attitude primitives live in
# exactly one place — shared/wulfram2_protocol/sim_kernel/rotation.py. This
# module is a thin adapter that imports them under the legacy private names the
# rest of the server uses, so server and client can never silently diverge.
from wulfram2_protocol.sim_kernel import (  # noqa: F401  (backend gated by WULFRAM_NATIVE_KERNEL)
    F32_TWO_PI,
    extract_euler_angles as _extract_euler_angles,
    f32 as _f32,
    matrix3_from_axis_angle as _matrix3_from_axis_angle,
    matrix3_from_euler_xyz as _matrix3_from_euler_xyz,
    normalize_angle_client as _normalize_angle_client,
)

TWO_PI = 2.0 * math.pi


def _normalize_angle_0_2pi(angle: float) -> float:
    """Normalize angle to [0, 2*pi] — float64 version for step() backward compat."""
    angle = angle % TWO_PI
    if angle < 0.0:
        angle += TWO_PI
    return angle


class VehiclePhysics:
    """Per-vehicle rotation-matrix physics matching the client's full pipeline.

    Maintains a 3x3 rotation matrix (9 doubles) + euler angles (3 float32) + angular
    velocity. Each substep does: Rodrigues construction → matrix multiply → euler
    extraction → angle normalization → velocity update.

    Heading = euler[2] (Z component = entity+0x38).
    """

    def __init__(self, damp_coeff: float = 1.0):
        self._angular_velocity = 0.0

        # Damping coefficient for angular velocity.
        # Runtime value = 2.0 for tanks (from decompile entity->0xbc->+4->+0x7c)
        self.damp_coeff = damp_coeff

        # 9-element rotation matrix, row-major, stored as doubles (client entity+0x58)
        self._matrix = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        # Euler angles [X=roll, Y=pitch, Z=heading], float32, [0, 2*pi]
        # Matches client entity+0x30/+0x34/+0x38
        self._euler = [0.0, 0.0, 0.0]

        # Cached prev euler for dirty-check (client entity+0xA0/+0xA4/+0xA8)
        # When euler != prev_euler, matrix is rebuilt from euler on next step
        self._prev_euler = [0.0, 0.0, 0.0]

    def _maybe_rebuild_matrix(self):
        """Dirty-check: if euler changed externally, rebuild matrix from euler.

        Matches client's GUESS6_Entity_get_rotation_matrix (Entities/Core.c:656):
          if (cached_euler != current_euler) → Matrix3_from_euler_xyz()
        """
        if (self._euler[0] != self._prev_euler[0] or
                self._euler[1] != self._prev_euler[1] or
                self._euler[2] != self._prev_euler[2]):
            self._matrix = _matrix3_from_euler_xyz(
                self._euler[0], self._euler[1], self._euler[2]
            )

    def step(self, torque: float, dt: float):
        """Advance physics by dt seconds (float64 mode, backward compat).

        Uses scalar heading integration — NOT the rotation matrix path.
        Kept for non-f32 testing and backward compatibility.
        """
        h = self._euler[2]
        ang_vel = self._angular_velocity

        h += ang_vel * dt
        ang_vel += (torque - ang_vel * self.damp_coeff) * dt

        self._angular_velocity = ang_vel
        self._euler[2] = _normalize_angle_0_2pi(h)
        # Invalidate matrix dirty-check (don't update prev_euler)

    def step_f32(self, torque: float, dt: float):
        """Step with full rotation matrix pipeline matching client's x86 code.

        Matches GUESS5_Matrix3_integrate_angular (Game/Simulation/Physics.c:5169):
          1. scaled_omega = ang_vel * (float)dt
          2. delta_R = Matrix3_from_axis_angle(scaled_omega)
          3. new_R = delta_R * old_R  (float32 dot products)
          4. Store new_R as doubles
          5. Extract euler via atan2
          6. Normalize euler to [0, 2*pi]
          7. ang_vel += effective_accel * (float)dt
        """
        # Dirty-check: rebuild matrix from euler if changed externally
        self._maybe_rebuild_matrix()

        fdt = _f32(dt)
        ang_vel = _f32(self._angular_velocity)
        f_torque = _f32(torque)
        f_damp = _f32(self.damp_coeff)

        # 1. Build scaled_omega = ang_vel * (float)dt
        # For pure yaw (no pitch/roll): only Z component is nonzero
        omega_x = 0.0
        omega_y = 0.0
        omega_z = _f32(ang_vel * fdt)

        # 2. Build incremental rotation matrix via Rodrigues formula
        delta = _matrix3_from_axis_angle(omega_x, omega_y, omega_z)

        # 3. Matrix multiply: new_matrix = delta * old_matrix
        # Each element = dot(delta_row_i, old_row_j), all in float32
        # Read old matrix elements as (float)double
        M = self._matrix
        d = delta
        fM = [_f32(M[i]) for i in range(9)]

        # Row 0 of result (delta row 0 dotted with old rows 0,1,2)
        # Dot product order from decompile: c2*M2 + c0*M0 + c1*M1 → (a+b)+c
        r00 = _f32(_f32(_f32(d[2] * fM[2]) + _f32(d[0] * fM[0])) + _f32(d[1] * fM[1]))
        r01 = _f32(_f32(_f32(d[2] * fM[5]) + _f32(d[0] * fM[3])) + _f32(d[1] * fM[4]))
        r02 = _f32(_f32(_f32(d[2] * fM[8]) + _f32(d[0] * fM[6])) + _f32(d[1] * fM[7]))

        # Row 1 of result (delta row 1)
        r10 = _f32(_f32(_f32(d[5] * fM[2]) + _f32(d[3] * fM[0])) + _f32(d[4] * fM[1]))
        r11 = _f32(_f32(_f32(d[5] * fM[5]) + _f32(d[3] * fM[3])) + _f32(d[4] * fM[4]))
        r12 = _f32(_f32(_f32(d[5] * fM[8]) + _f32(d[3] * fM[6])) + _f32(d[4] * fM[7]))

        # Row 2 of result (delta row 2)
        r20 = _f32(_f32(_f32(d[8] * fM[2]) + _f32(d[6] * fM[0])) + _f32(d[7] * fM[1]))
        r21 = _f32(_f32(_f32(d[8] * fM[5]) + _f32(d[6] * fM[3])) + _f32(d[7] * fM[4]))
        r22 = _f32(_f32(_f32(d[8] * fM[8]) + _f32(d[6] * fM[6])) + _f32(d[7] * fM[7]))

        # 4. Write result back as doubles (matching client's (double)result cast)
        # Write-back mapping from decompile (interleaved):
        #   M[0]=r00  M[3]=r01  M[6]=r02
        #   M[1]=r10  M[4]=r11  M[7]=r12
        #   M[2]=r20  M[5]=r21  M[8]=r22
        self._matrix[0] = float(r00)
        self._matrix[3] = float(r01)
        self._matrix[6] = float(r02)
        self._matrix[1] = float(r10)
        self._matrix[4] = float(r11)
        self._matrix[7] = float(r12)
        self._matrix[2] = float(r20)
        self._matrix[5] = float(r21)
        self._matrix[8] = float(r22)

        # 5. Extract euler angles from rotation matrix
        euler_x, euler_y, euler_z = _extract_euler_angles(self._matrix)

        # 6. Normalize each euler angle to [0, 2*pi]
        self._euler[0] = _normalize_angle_client(euler_x)
        self._euler[1] = _normalize_angle_client(euler_y)
        self._euler[2] = _normalize_angle_client(euler_z)

        # 7. Update prev_euler (dirty-check: next step won't rebuild matrix)
        self._prev_euler[0] = self._euler[0]
        self._prev_euler[1] = self._euler[1]
        self._prev_euler[2] = self._euler[2]

        # 8. Compute effective angular acceleration and update velocity
        # Client: damping_torque = -ang_vel * damp_coeff (Physics.c:5169 integrator;
        #         damping mode in Physics.c:5264 substep, coeff at physics_config+0x7C)
        #         effective_accel = angular_acceleration + damping_torque
        #         ang_vel += effective_accel * (float)dt
        damping = _f32(_f32(-ang_vel) * f_damp)
        effective = _f32(f_torque + damping)
        ang_vel = _f32(ang_vel + _f32(effective * fdt))

        self._angular_velocity = ang_vel

    def step_client_substeps(self, torque: float, frame_dt: float, use_f32: bool = False):
        """Step physics using the client's two-level substep algorithm.

        Outer: GUESS6_GameSim_substep_update (Game/Simulation/Physics.c:1974)
          - elapsed clamped to 550ms (0x226), substep_count = elapsed/110 (0x6E) + 1,
            capped at 5; uniform time_per_step with the remainder on the last step
          - Each outer substep: TankVehicle_apply_physics() (Vehicles.c:1336, re-adds
            torque) via vehicle-client VTable+0x0C, then physics tick
          - Torque ACCUMULATES across outer substeps (entity[0x50] += torque each time)

        Inner: GUESS5_Physics_substep_integrate_angular (Game/Simulation/Physics.c:5264)
          - max substep = 0.04s (40ms), or dt*0.5 if dt > 0.08s (Physics.c:5298-5300)
          - Subtraction loop until remainder

        Args:
            torque: base torque for ONE application (turn_adjust * raw_input)
            frame_dt: the client's frame duration in seconds
            use_f32: if True, use rotation matrix pipeline (matching client)
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
            # Client: TankVehicle_apply_physics() ADDS torque each outer substep
            # entity[0x50] += torque (accumulated, NOT zeroed between outer substeps)
            # So substep i sees (i+1) * torque
            accumulated_torque = torque * (i + 1)

            # Last substep gets remainder, others get floor-divided dt
            if i == outer_count - 1:
                this_outer_ms = remaining_ms
            else:
                this_outer_ms = outer_dt_ms
                remaining_ms -= outer_dt_ms

            # Convert to seconds for inner substep
            this_outer_s = this_outer_ms / 1000.0

            # Inner substep (GUESS5_Physics_substep_integrate_angular)
            inner_max_s = 0.04  # 40ms default
            if this_outer_s > 0.08:
                inner_max_s = this_outer_s * 0.5

            inner_remaining = this_outer_s
            while inner_remaining > 0.0:
                if inner_remaining <= inner_max_s:
                    step_fn(accumulated_torque, inner_remaining)
                    inner_remaining = 0.0
                else:
                    step_fn(accumulated_torque, inner_max_s)
                    inner_remaining -= inner_max_s

    @property
    def heading(self) -> float:
        """Return heading converted to [-pi, pi] for server code compatibility.

        Heading is euler[2] (Z component = entity+0x38), stored in [0, 2*pi].
        """
        h = self._euler[2]
        if h > math.pi:
            h -= TWO_PI
        return h

    @property
    def heading_raw(self) -> float:
        """Return internal heading in [0, 2*pi] (matches client representation)."""
        return self._euler[2]

    @property
    def rotation(self) -> tuple[float, float, float]:
        """Return the current XYZ Euler body rotation."""
        return (self._euler[0], self._euler[1], self._euler[2])

    @heading.setter
    def heading(self, val: float):
        """Set heading. Does NOT update prev_euler — triggers dirty-check matrix rebuild.

        This matches client behavior: external euler writes (spawn, server correction)
        leave prev_euler stale, causing Entity_get_rotation_matrix to rebuild.
        """
        self._euler[2] = val

    @property
    def angular_velocity(self) -> float:
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, val: float):
        self._angular_velocity = val

    def reset(self):
        """Reset all state to zero/identity."""
        self._angular_velocity = 0.0
        self._matrix = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        self._euler = [0.0, 0.0, 0.0]
        self._prev_euler = [0.0, 0.0, 0.0]
