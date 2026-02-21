"""
Rotation-matrix physics model for vehicle heading.

Matches the Wulfram2 client's EXACT angular integration pipeline from the azurefishy
decompile. The client uses a 3x3 rotation matrix (stored as doubles) with axis-angle
Rodrigues construction, float32 matrix multiply, and atan2 euler extraction.

Per-frame cycle (from decompile):
  1. Torque accumulator zeroed (entity[0x48-0x50] cleared after previous frame)
  2. Yaw torque = turn_mobility * (float)turn_adjust * yaw_axis  (line 23103)
     - turn_adjust read as double, cast to float32 BEFORE multiply
  3. Entity_get_rotation_matrix: dirty-check euler vs prev_euler, rebuild if changed
  4. Physics_substep_integrate_angular (lines 28804-28877):
     - Substep loop: max 40ms or dt*0.5 if dt > 80ms
     - Matrix3_integrate_angular per substep (lines 28704-28796):
       a. scaled_omega = ang_vel * (float)dt
       b. delta_R = Matrix3_from_axis_angle(scaled_omega)  [Rodrigues, float32]
       c. new_R = delta_R * old_R  [9 dot products, float32 intermediates]
       d. Store new_R as doubles
       e. Extract euler angles via atan2
       f. Normalize euler to [0, 2*pi] via iterative add/subtract
       g. ang_vel += effective_accel * (float)dt
  5. Save euler to prev_euler cache (entity+0xA0)
  6. Torque accumulator zeroed for next frame

Client references (from azurefishy decompile):
  - GUESS3_Matrix3_from_axis_angle (addr 0x004f1150): Rodrigues formula, float32
  - GUESS5_Matrix3_integrate_angular (addr 0x004f12c0): Full angular integration
  - GUESS5_Physics_substep_integrate_angular (addr 0x004f14a0): Substep loop + damping
  - GUESS5_Matrix3_from_euler_xyz (addr 0x004350c0): Euler→matrix rebuild
  - GUESS3_Math_normalize_angle_radians (addr 0x004f0da0): Iterative [0, 2*pi]
  - GUESS2_Vec3_normalize_safe (addr 0x004e1c00): Euler extraction via atan2
"""

import math
import struct

TWO_PI = 2.0 * math.pi


def _f32(v: float) -> float:
    """Round-trip a float through float32 to match client precision."""
    return struct.unpack('<f', struct.pack('<f', v))[0]


# Float32 2*pi constant from decompile (exact value used in Math_normalize_angle_radians)
F32_TWO_PI = _f32(6.2831855)


def _normalize_angle_client(angle: float) -> float:
    """Normalize angle to [0, 2*pi] matching client's Math_normalize_angle_radians.

    From decompile (addr 0x004f0da0):
      - Safety clamp: |angle| > 20000 → 0.0
      - Iterative add/subtract of 6.2831855f (float32 2*pi)
      - All operations in float32
    """
    angle = _f32(angle)
    if angle > 20000.0 or angle < -20000.0:
        return 0.0
    while angle < 0.0:
        angle = _f32(angle + F32_TWO_PI)
    while angle > F32_TWO_PI:
        angle = _f32(angle - F32_TWO_PI)
    return angle


def _normalize_angle_0_2pi(angle: float) -> float:
    """Normalize angle to [0, 2*pi] — float64 version for step() backward compat."""
    angle = angle % TWO_PI
    if angle < 0.0:
        angle += TWO_PI
    return angle


def _matrix3_from_euler_xyz(ex: float, ey: float, ez: float) -> list:
    """Build 3x3 rotation matrix from euler XYZ angles.

    Matches GUESS5_Matrix3_from_euler_xyz (addr 0x004350c0).
    XYZ intrinsic rotation order: R = Rz * Ry * Rx.

    Input euler angles: X=roll, Y=pitch, Z=heading.
    Output: 9-element list (row-major doubles), matching client entity+0x58.

    Trig computed as extended precision (Python float64 ≈ x87 float80),
    intermediates as float64 (matching decompile's (float10)(double) casts),
    results stored as double.
    """
    cx = math.cos(ex)
    sx = math.sin(ex)
    cy = math.cos(ey)
    sy = math.sin(ey)
    cz = math.cos(ez)
    sz = math.sin(ez)

    return [
        cz * cy,                        # M[0]
        cz * sy * sx - sz * cx,         # M[1]
        sz * sx + cz * cx * sy,         # M[2]
        sz * cy,                        # M[3]
        sy * sx * sz + cz * cx,         # M[4]
        sy * cx * sz - cz * sx,         # M[5]
        -sy,                            # M[6]
        cy * sx,                        # M[7]
        cx * cy,                        # M[8]
    ]


def _matrix3_from_axis_angle(omega_x: float, omega_y: float, omega_z: float) -> list:
    """Build 3x3 rotation matrix from axis-angle vector via Rodrigues formula.

    Matches GUESS3_Matrix3_from_axis_angle (addr 0x004f1150).
    Input: axis-angle vector (3 float32 values). Angle = ||vector||, axis = vector/||vector||.
    Output: 9-element list (float32 values), row-major.

    All arithmetic in float32 except sqrt/cos/sin which use extended precision
    (x87 FPU) then cast to float32.
    """
    # Compute angle = length of axis-angle vector
    # Client uses x87 sqrt (extended precision), casts to float32
    angle_sq = omega_x * omega_x + omega_y * omega_y + omega_z * omega_z
    angle_f64 = math.sqrt(angle_sq)  # extended precision (Python float64 ≈ x87 float80)
    angle = _f32(angle_f64)

    # Identity threshold: angle < 1e-05 (from decompile)
    if angle < 1e-05:
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    # Normalize axis
    inv_len = _f32(1.0 / angle)
    nx = _f32(omega_x * inv_len)
    ny = _f32(omega_y * inv_len)
    nz = _f32(omega_z * inv_len)

    # cos/sin via x87 extended precision, cast to float32
    # Client passes the angle (still on FPU stack from sqrt) to cos/sin
    c = _f32(math.cos(angle_f64))
    s = _f32(math.sin(angle_f64))
    t = _f32(1.0 - c)  # 1 - cos(angle)

    # Rodrigues formula: R = cos*I + (1-cos)*n*nT + sin*skew(n)
    t_nx = _f32(nx * t)
    t_ny = _f32(ny * t)
    t_nz = _f32(nz * t)

    return [
        _f32(_f32(nx * t_nx) + c),         # R[0] = nx*nx*(1-cos) + cos
        _f32(_f32(ny * t_nx) + _f32(nz * s)),  # R[1] = ny*nx*(1-cos) + nz*sin
        _f32(_f32(t_nx * nz) - _f32(ny * s)),  # R[2] = nx*nz*(1-cos) - ny*sin
        _f32(_f32(t_ny * nx) - _f32(nz * s)),  # R[3] = nx*ny*(1-cos) - nz*sin
        _f32(_f32(ny * t_ny) + c),         # R[4] = ny*ny*(1-cos) + cos
        _f32(_f32(t_ny * nz) + _f32(nx * s)),  # R[5] = nz*ny*(1-cos) + nx*sin
        _f32(_f32(nx * t_nz) + _f32(ny * s)),  # R[6] = nx*nz*(1-cos) + ny*sin
        _f32(_f32(ny * t_nz) - _f32(nx * s)),  # R[7] = ny*nz*(1-cos) - nx*sin
        _f32(_f32(nz * t_nz) + c),         # R[8] = nz*nz*(1-cos) + cos
    ]


def _extract_euler_angles(m: list) -> tuple:
    """Extract XYZ euler angles from 3x3 rotation matrix.

    Matches GUESS2_Vec3_normalize_safe (addr 0x004e1c00) which is actually
    euler extraction via atan2, despite its misleading name. The atan2 arguments
    come from the rotation matrix elements on the FPU stack.

    Standard XYZ euler extraction from row-major matrix:
      euler_x (roll)  = atan2(M[7], M[8])           = atan2(R[2][1], R[2][2])
      euler_y (pitch) = atan2(-M[6], sqrt(M[0]^2 + M[1]^2))  = atan2(-R[2][0], ...)
      euler_z (heading) = atan2(M[3], M[0])          = atan2(R[1][0], R[0][0])

    Gimbal lock: when sqrt(M[0]^2 + M[1]^2) <= 2^(-19), roll is set to 0.

    Returns: (euler_x, euler_y, euler_z) as float32 values.
    """
    # Read matrix elements as float32 (matching client's (float) casts)
    fm0 = _f32(m[0])
    fm1 = _f32(m[1])
    fm3 = _f32(m[3])
    fm6 = _f32(m[6])
    fm7 = _f32(m[7])
    fm8 = _f32(m[8])

    # Gimbal lock check: sqrt(M[0][0]^2 + M[1][0]^2)
    # Client computes this on the FPU stack (extended precision)
    gimbal = math.sqrt(fm0 * fm0 + fm1 * fm1)

    if gimbal <= 1.9073486328125e-06:  # 2^(-19), from decompile
        # Degenerate case: near gimbal lock
        # pitch and yaw extracted, roll set to 0
        euler_x = _f32(math.atan2(fm7, fm8))
        euler_y = _f32(math.atan2(-fm6, gimbal))
        euler_z = 0.0
    else:
        # Normal case: all 3 euler angles
        euler_x = _f32(math.atan2(fm7, fm8))
        euler_y = _f32(float(math.atan2(-fm6, gimbal)))  # extra double cast per decompile
        euler_z = _f32(math.atan2(fm3, fm0))

    return (euler_x, euler_y, euler_z)


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

        Matches client's Entity_get_rotation_matrix (line 18028):
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

        Matches GUESS5_Matrix3_integrate_angular (addr 0x004f12c0):
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
        # Client: damping_torque = -ang_vel * damp_coeff (line 28861)
        #         effective_accel = angular_acceleration + damping_torque (line 28864)
        #         ang_vel += effective_accel * (float)dt
        damping = _f32(_f32(-ang_vel) * f_damp)
        effective = _f32(f_torque + damping)
        ang_vel = _f32(ang_vel + _f32(effective * fdt))

        self._angular_velocity = ang_vel

    def step_client_substeps(self, torque: float, frame_dt: float, use_f32: bool = False):
        """Step physics using the client's two-level substep algorithm.

        Outer: GUESS5_GameSim_substep_update (lines 25514-25574)
          - elapsed clamped to 550ms, split into 1-5 substeps of <=110ms each
          - Each outer substep: TankVehicle_apply_physics() (re-adds torque) + physics tick
          - Torque ACCUMULATES across outer substeps (entity[0x50] += torque each time)

        Inner: GUESS5_Physics_substep_integrate_angular (lines 28804-28877)
          - max substep = 0.04s (40ms), or dt*0.5 if dt > 0.08s
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
