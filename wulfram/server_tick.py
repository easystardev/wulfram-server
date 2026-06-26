"""TickMixin -- per-tick physics: tank surface/attitude sampling, turn torque,
input decode, and jump-jet stepping, extracted verbatim from WulframServer
(server.py decomposition, step 5). Method-only mixin; shares state via `self`.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from .client import ClientContext
from .physics import _extract_euler_angles, _matrix3_from_euler_xyz, _normalize_angle_client
from .weapons import BehaviorSlot, EntityType, VEHICLE_PHYSICS_CONFIGS
from wulfram2_protocol.entities import (
    JUMP_JET_CONFIGS,
    JUMP_JET_SPAWN_LOCKOUT,
    rigid_body_point_velocity,
    terrain_aligned_basis,
    tank_altitude_mobility_factor,
    tank_body_matrix_with_heading,
    tank_fuel_mobility_factor,
    tank_hover_clearance_target,
    tank_spring_attitude_step,
    tank_spring_average_clearance,
    tank_spring_force_attitude_step,
    tank_suspension_local_sample_offsets,
    tank_suspension_world_sample_offsets,
)


class TickMixin:
    @staticmethod
    def _piecewise_interpolate(samples: list, t: float) -> float:
        """Piecewise-linear interpolation matching the client's steering curve.

        ``samples`` contains N evenly-spaced output values over the domain
        [0.0, 1.0].  ``t`` should be in [0, 1].  Returns the interpolated
        output value (0.0-1.0).
        """
        n = len(samples)
        if n < 2:
            return t
        t = max(0.0, min(1.0, t))
        # Map t into the sample index space.
        idx_f = t * (n - 1)
        idx_lo = int(idx_f)
        if idx_lo >= n - 1:
            return samples[-1]
        frac = idx_f - idx_lo
        return samples[idx_lo] + (samples[idx_lo + 1] - samples[idx_lo]) * frac

    @staticmethod
    def _tank_low_speed_mobility_factor(current_speed: float, speed_threshold: float) -> float:
        """Compatibility wrapper for the earlier speed-based interpretation."""
        return tank_fuel_mobility_factor(current_speed, speed_threshold)

    def _tank_altitude_mobility(self, ctx: ClientContext) -> float:
        """Approximate the OG tank altitude penalty from current terrain clearance."""
        if ctx.entity_type != EntityType.TANK or self.terrain is None:
            return 1.0
        _avg_up, clearance_ratio = self._sample_tank_surface_state(ctx)
        return tank_altitude_mobility_factor(clearance_ratio)

    def _tank_hover_clearance_target(self, ctx: ClientContext) -> float:
        veh_config = VEHICLE_PHYSICS_CONFIGS.get(ctx.entity_type)
        max_altitude = veh_config.max_altitude if veh_config else 3.25
        return tank_hover_clearance_target(
            getattr(self, "tank_spring_base_offset", 0.0),
            max_altitude,
        )

    def _tank_terrain_contact_vector(self, ctx: ClientContext) -> tuple[float, float]:
        """Approximate the OG spring contact direction from sampled terrain normals."""
        if ctx.entity_type != EntityType.TANK or self.terrain is None:
            return (0.0, 0.0)
        avg_up, _clearance_ratio = self._sample_tank_surface_state(ctx)
        return (avg_up[0], avg_up[1])

    def _sample_tank_surface_state(
        self,
        ctx: ClientContext,
        heading: float | None = None,
    ) -> tuple[tuple[float, float, float], float]:
        """Approximate spring world-state from four tank-footprint terrain samples."""
        if ctx.entity_type != EntityType.TANK or self.terrain is None:
            ctx.debug_last_spring_state = {}
            return (0.0, 0.0, 1.0), 1.0

        if heading is None:
            heading = ctx.player_heading
        try:
            base_pos = (
                float(ctx.player_pos[0]),
                float(ctx.player_pos[1]),
                float(ctx.player_pos[2]),
            )
            base_vel = (
                float(ctx.player_vel[0]),
                float(ctx.player_vel[1]),
                float(ctx.player_vel[2]),
            )
            heading = float(heading)
        except (TypeError, ValueError, OverflowError, IndexError):
            ctx.debug_last_spring_state = {
                "source": "Spring_update_world_state",
                "invalid_state": True,
                "invalid_reason": "non_numeric_pose",
            }
            return (0.0, 0.0, 1.0), 1.0
        if not all(math.isfinite(value) for value in (*base_pos, *base_vel, heading)):
            ctx.debug_last_spring_state = {
                "source": "Spring_update_world_state",
                "invalid_state": True,
                "invalid_reason": "nonfinite_pose",
                "position": base_pos,
                "velocity": base_vel,
                "heading": heading,
            }
            return (0.0, 0.0, 1.0), 1.0

        local_offsets = tank_suspension_local_sample_offsets(
            longitudinal=self._TANK_RADIUS * 0.85,
            lateral=self._TANK_RADIUS * 0.55,
            local_offsets=getattr(self, "tank_spring_sample_local_offsets", None),
        )
        body_matrix = None
        if getattr(self, "terrain_pitch_enabled", False):
            try:
                body_matrix = tuple(
                    float(v)
                    for v in tuple(getattr(ctx, "spring_body_matrix", ()) or ())[:9]
                )
            except (TypeError, ValueError):
                body_matrix = ()
            if len(body_matrix) != 9:
                body_matrix = _matrix3_from_euler_xyz(
                    float(ctx.player_pose.get("roll", 0.0) or 0.0),
                    float(ctx.player_pose.get("pitch", 0.0) or 0.0),
                    heading,
                )
            else:
                body_matrix = tank_body_matrix_with_heading(
                    body_matrix,
                    heading,
                    fallback_roll=float(ctx.player_pose.get("roll", 0.0) or 0.0),
                    fallback_pitch=float(ctx.player_pose.get("pitch", 0.0) or 0.0),
                )
        offsets = tank_suspension_world_sample_offsets(
            heading,
            longitudinal=self._TANK_RADIUS * 0.85,
            lateral=self._TANK_RADIUS * 0.55,
            local_offsets=local_offsets,
            rotation_matrix=body_matrix,
        )
        sum_up_x = 0.0
        sum_up_y = 0.0
        sum_up_z = 0.0
        sum_clearance = 0.0
        samples = []
        body_ang_vel = getattr(ctx, "spring_body_ang_vel", (0.0, 0.0)) or (0.0, 0.0)
        try:
            roll_velocity = float(body_ang_vel[0])  # type: ignore[index]
            pitch_velocity = float(body_ang_vel[1])  # type: ignore[index]
        except (TypeError, IndexError, ValueError):
            roll_velocity = 0.0
            pitch_velocity = 0.0
        try:
            yaw_velocity = float(getattr(ctx, "angular_vel_yaw", 0.0) or 0.0)
        except (TypeError, ValueError):
            yaw_velocity = 0.0

        for (local_x, local_y), (dx, dy, dz) in zip(local_offsets, offsets):
            sx = base_pos[0] + dx
            sy = base_pos[1] + dy
            sz = base_pos[2] + dz
            sample_height_normal = getattr(self.terrain, "sample_height_normal", None)
            if callable(sample_height_normal):
                raw_ground_z, sample_up = sample_height_normal(sx, sy)
            else:
                raw_ground_z = self.terrain.get_height(sx, sy)
                dh_dx, dh_dy = self.terrain.get_slope(sx, sy)
                mag_sq = dh_dx * dh_dx + dh_dy * dh_dy + 1.0
                if mag_sq <= 1e-10:
                    sample_up = (0.0, 0.0, 1.0)
                else:
                    inv_mag = 1.0 / math.sqrt(mag_sq)
                    sample_up = (-dh_dx * inv_mag, -dh_dy * inv_mag, inv_mag)
            clearance = sz - raw_ground_z
            sum_up_x += sample_up[0]
            sum_up_y += sample_up[1]
            sum_up_z += sample_up[2]
            sum_clearance += clearance
            point_velocity = rigid_body_point_velocity(
                base_pos,
                base_vel,
                (roll_velocity, pitch_velocity, yaw_velocity),
                (sx, sy, sz),
                rotation_matrix=body_matrix,
            )
            samples.append(
                {
                    "local_offset": [round(float(local_x), 5), round(float(local_y), 5)],
                    "spring_normal": [0.0, 0.0, -1.0],
                    "world_offset": [round(float(dx), 5), round(float(dy), 5)],
                    "world_offset_z": round(float(dz), 5),
                    "sample_xy": [round(float(sx), 5), round(float(sy), 5)],
                    "sample_z": round(float(sz), 5),
                    "raw_ground_z": round(float(raw_ground_z), 5),
                    "clearance": round(float(clearance), 5),
                    "point_velocity": [round(float(v), 5) for v in point_velocity],
                    "point_velocity_z": round(float(point_velocity[2]), 5),
                    "point_velocity_source": "RigidBody_compute_point_velocity",
                    "normal": [round(float(v), 6) for v in sample_up],
                }
            )

        inv_count = 1.0 / float(len(offsets))
        avg_up_x = sum_up_x * inv_count
        avg_up_y = sum_up_y * inv_count
        avg_up_z = sum_up_z * inv_count
        avg_mag_sq = avg_up_x * avg_up_x + avg_up_y * avg_up_y + avg_up_z * avg_up_z
        if avg_mag_sq <= 1e-10:
            avg_up = (0.0, 0.0, 1.0)
        else:
            inv_avg_mag = 1.0 / math.sqrt(avg_mag_sq)
            avg_up = (avg_up_x * inv_avg_mag, avg_up_y * inv_avg_mag, avg_up_z * inv_avg_mag)

        target_clearance = self._tank_hover_clearance_target(ctx)
        average_clearance = tank_spring_average_clearance(sum_clearance, len(offsets))
        clearance_ratio = average_clearance / target_clearance
        ctx.debug_last_spring_state = {
            "source": "Spring_update_world_state",
            "point_count": len(offsets),
            "clearance_denominator": max(1, len(offsets) - 1),
            "height_sum": round(float(sum_clearance), 5),
            "average_clearance": round(float(average_clearance), 5),
            "target_clearance": round(float(target_clearance), 5),
            "clearance_ratio": round(float(clearance_ratio), 6),
            "avg_normal": [round(float(v), 6) for v in avg_up],
            "rotation_source": "body_matrix" if body_matrix is not None else "heading_flat",
            "body_matrix": (
                [round(float(v), 8) for v in body_matrix]
                if body_matrix is not None
                else None
            ),
            "samples": samples,
        }
        return avg_up, clearance_ratio

    def _update_player_surface_attitude(
        self,
        ctx: ClientContext,
        heading: float | None = None,
        dt: float | None = None,
        snap: bool = False,
        suspension_lift: float | None = None,
        suspension_point_forces: Sequence[float] | None = None,
        suspension_point_blend_factors: Sequence[float] | None = None,
        spring_state_override: Mapping[str, object] | None = None,
    ) -> dict:
        """Update replicated tank body roll/pitch from the spring response path.

        OG keeps the tank's yaw/input heading separate from the active softbody
        surface normal. `Spring_compute_suspension_forces` then contributes
        pitch/roll torque rather than snapping Euler angles directly, so keep a
        small X/Y angular-velocity state for body attitude while the full
        per-point force curve is being ported.
        """
        if heading is None:
            heading = ctx.player_heading
        if dt is None:
            snap = True
            dt = 1.0 / float(getattr(self, "tick_rate_hz", 30.0) or 30.0)

        if (
            ctx.entity_type != EntityType.TANK
            or self.terrain is None
            or self.up_axis != "z"
            or not getattr(self, "terrain_pitch_enabled", False)
        ):
            ctx.player_pose["roll"] = 0.0
            ctx.player_pose["pitch"] = 0.0
            ctx.player_pose["yaw"] = -ctx.player_heading
            ctx.spring_body_ang_vel = (0.0, 0.0)
            ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, ctx.player_heading)
            return {
                "source": "flat",
                "rotation": (0.0, 0.0, ctx.player_heading),
                "up": (0.0, 0.0, 1.0),
                "matrix": ctx.spring_body_matrix,
                "target_rotation": (0.0, 0.0, ctx.player_heading),
                "angular_velocity": (0.0, 0.0),
            }

        spring_state_for_attitude = (
            spring_state_override
            if isinstance(spring_state_override, Mapping)
            else None
        )
        if spring_state_for_attitude is not None:
            raw_up = spring_state_for_attitude.get("avg_normal")
            if isinstance(raw_up, (list, tuple)) and len(raw_up) >= 3:
                try:
                    avg_up = (
                        float(raw_up[0]),
                        float(raw_up[1]),
                        float(raw_up[2]),
                    )
                except (TypeError, ValueError):
                    avg_up = (0.0, 0.0, 1.0)
            else:
                avg_up = (0.0, 0.0, 1.0)
            try:
                _clearance_ratio = float(
                    spring_state_for_attitude.get("clearance_ratio", 1.0)
                )
            except (TypeError, ValueError):
                _clearance_ratio = 1.0
        else:
            avg_up, _clearance_ratio = self._sample_tank_surface_state(ctx, heading)
        if abs(avg_up[2]) > 1e-6:
            dh_dx = -avg_up[0] / avg_up[2]
            dh_dy = -avg_up[1] / avg_up[2]
        else:
            dh_dx, dh_dy = self.terrain.get_slope(ctx.player_pos[0], ctx.player_pos[1])

        forward, right, up = terrain_aligned_basis(dh_dx, dh_dy, heading)
        matrix = [
            forward[0], right[0], up[0],
            forward[1], right[1], up[1],
            forward[2], right[2], up[2],
        ]
        target_roll, target_pitch, _yaw_from_matrix = _extract_euler_angles(matrix)
        target_roll = _normalize_angle_client(target_roll)
        target_pitch = _normalize_angle_client(target_pitch)
        # CH2 slope-attitude diagnosis: stash the live terrain-normal target + the
        # gradient it was derived from so the `attitude` control command can show
        # where the forward/pitch component diverges (read-only debug).
        ctx.debug_attitude_target = (
            target_roll, target_pitch, self.tank_spring_attitude_model, dh_dx, dh_dy,
        )
        if snap:
            roll = target_roll
            pitch = target_pitch
            step = None
            ctx.spring_body_ang_vel = (0.0, 0.0)
            matrix = _matrix3_from_euler_xyz(roll, pitch, heading)
        else:
            body_vel = getattr(ctx, "spring_body_ang_vel", (0.0, 0.0)) or (0.0, 0.0)
            veh_cfg = VEHICLE_PHYSICS_CONFIGS.get(ctx.entity_type)
            spring_state = (
                spring_state_for_attitude
                if spring_state_for_attitude is not None
                else getattr(ctx, "debug_last_spring_state", {}) or {}
            )
            samples = spring_state.get("samples") if isinstance(spring_state, dict) else None
            source_matrix = (
                spring_state.get("body_matrix")
                if isinstance(spring_state, dict)
                else None
            )
            damping = getattr(
                self,
                "tank_spring_attitude_damping",
                veh_cfg.angular_damping if veh_cfg else 2.0,
            )
            if (
                getattr(self, "tank_spring_attitude_model", "force") == "force"
                and suspension_lift is not None
                and isinstance(samples, (list, tuple))
                and samples
            ):
                step = tank_spring_force_attitude_step(
                    float(ctx.player_pose.get("roll", 0.0) or 0.0),
                    float(ctx.player_pose.get("pitch", 0.0) or 0.0),
                    heading,
                    samples,
                    float(body_vel[0]),
                    float(body_vel[1]),
                    float(dt),
                    float(suspension_lift),
                    damping=damping,
                    point_forces=suspension_point_forces,
                    point_blend_factors=suspension_point_blend_factors,
                    integration_model=getattr(
                        self,
                        "tank_spring_attitude_integration",
                        "decompile_accel",
                    ),
                    rotation_matrix=source_matrix,
                )
            else:
                step = tank_spring_attitude_step(
                    float(ctx.player_pose.get("roll", 0.0) or 0.0),
                    float(ctx.player_pose.get("pitch", 0.0) or 0.0),
                    target_roll,
                    target_pitch,
                    float(body_vel[0]),
                    float(body_vel[1]),
                    float(dt),
                    stiffness=getattr(self, "tank_spring_attitude_stiffness", 40.0),
                    damping=damping,
                )
            roll = _normalize_angle_client(step.roll)
            pitch = _normalize_angle_client(step.pitch)
            ctx.spring_body_ang_vel = (step.roll_velocity, step.pitch_velocity)
            if hasattr(step, "rotation_matrix") and getattr(step, "rotation_matrix"):
                matrix = tuple(float(v) for v in step.rotation_matrix)
            else:
                matrix = _matrix3_from_euler_xyz(roll, pitch, heading)
            up = (matrix[2], matrix[5], matrix[8])
        ctx.player_pose["roll"] = roll
        ctx.player_pose["pitch"] = pitch
        ctx.player_pose["yaw"] = -ctx.player_heading
        ctx.spring_body_matrix = tuple(float(v) for v in matrix)
        debug = {
            "target": (target_roll, target_pitch, ctx.player_heading),
            "angular_velocity": ctx.spring_body_ang_vel,
            "spring_state_source": (
                "force_sample"
                if spring_state_for_attitude is not None
                else "resampled"
            ),
        }
        if step is not None:
            debug["model"] = "force" if hasattr(step, "point_forces") else "target"
            debug["torque"] = (step.roll_torque, step.pitch_torque)
            debug["damping"] = step.damping
            debug["dt"] = step.dt
            if hasattr(step, "point_forces"):
                debug.update(
                    {
                        "local_torque": (step.local_torque_x, step.local_torque_y),
                        "point_forces": step.point_forces,
                        "total_lift": step.total_lift,
                        "torque_scale": step.torque_scale,
                        "torque_model": step.torque_model,
                        "torque_force_scales": step.torque_force_scales,
                        "integration_model": step.integration_model,
                        "angular_velocity_before": step.angular_velocity_before,
                        "spring_angular_delta": step.spring_angular_delta,
                        "angular_velocity_after_spring": step.angular_velocity_after_spring,
                        "angular_velocity_after_damping": step.angular_velocity_after_damping,
                        "rotation_matrix": step.rotation_matrix,
                    }
                )
            else:
                debug.update(
                    {
                        "error": (step.roll_error, step.pitch_error),
                        "stiffness": step.stiffness,
                    }
                )
        return {
            "source": "terrain_surface",
            "rotation": (roll, pitch, ctx.player_heading),
            "up": up,
            "matrix": ctx.spring_body_matrix,
            "target_rotation": (target_roll, target_pitch, ctx.player_heading),
            "angular_velocity": ctx.spring_body_ang_vel,
            "spring_attitude": debug,
        }

    def _normalize_turn_input_value(self, ctx: ClientContext, turn_val: float) -> float:
        """Normalize a raw TURNING slot value to signed yaw input in [-1, 1]."""
        if turn_val > 1.5 or turn_val < -1.5:
            scale = getattr(ctx.weapon_system, "control_max", 1000.0) or 1000.0
            turn_input = max(-1.0, min(1.0, turn_val / scale))
        else:
            turn_input = max(-1.0, min(1.0, turn_val))

        if abs(turn_input) < self.turn_deadzone:
            turn_input = 0.0

        # Apply turn_sign to match client negation:
        # Client: controller[0x74] = -button_normalized(1)
        # Our turn_sign = -1.0 achieves the same inversion.
        return self.turn_sign * turn_input

    def _compute_turn_torque(self, ctx: ClientContext, raw_input: float) -> float:
        """Compute yaw torque from normalized raw input using client-equivalent f32 math."""
        # Client: entity[0x50] += turn_mobility * (float)turn_adjust * yaw_axis
        # turn_adjust is read as double then cast to float32 before multiply.
        from .physics import _f32

        veh_cfg = VEHICLE_PHYSICS_CONFIGS.get(ctx.entity_type)
        turn_adj = veh_cfg.turn_adjust if veh_cfg else self.turn_adjust
        _f32_turn_adjust = _f32(float(turn_adj))
        torque = _f32(_f32_turn_adjust * _f32(raw_input))
        return _f32(torque * _f32(self._tank_altitude_mobility(ctx)))

    def _sync_heading_physics_to_context(self, ctx: ClientContext, physics) -> None:
        """Copy yaw physics into the context without flattening spring body pose.

        The current `VehiclePhysics` model only integrates yaw torque. Tank
        pitch/roll is produced by the spring/softbody attitude path in
        `_update_player_position`, so copying `physics.rotation[0:2]` here would
        erase the previous spring-derived body matrix before the next
        `Spring_update_world_state` sample.
        """
        ctx.player_heading = physics.heading
        ctx.angular_vel_yaw = physics.angular_velocity
        ctx.player_yaw = -ctx.player_heading
        ctx.player_pose["yaw"] = -ctx.player_heading
        ctx.spring_body_matrix = tank_body_matrix_with_heading(
            getattr(ctx, "spring_body_matrix", None),
            ctx.player_heading,
            fallback_roll=float(ctx.player_pose.get("roll", 0.0) or 0.0),
            fallback_pitch=float(ctx.player_pose.get("pitch", 0.0) or 0.0),
        )

    def _get_raw_turn_input(self, ctx: ClientContext) -> float:
        """Get normalized turning input [-1, 1] with deadzone and sign applied.

        Returns the raw turning input for direct-impulse yaw physics.
        The client uses raw input directly for yaw (Vehicles.c:1193),
        NOT the piecewise curve (which only feeds the spring system for
        pitch/roll terrain following).

        The torque is computed externally: torque = raw_input * turn_adjust.
        """
        if ctx.injected_turn is not None:
            return self.turn_sign * ctx.injected_turn

        turn_val = ctx.weapon_system.behavior_slots[BehaviorSlot.TURNING]
        return self._normalize_turn_input_value(ctx, turn_val)

    def _normalize_behavior_axis_value(self, ctx: ClientContext, val: float) -> float:
        """Normalize a network behavior slot axis into [-1, 1]."""
        if val > 1.5 or val < -1.5:
            scale = getattr(ctx.weapon_system, "control_max", 1000.0) or 1000.0
            return max(-1.0, min(1.0, val / scale))
        return max(-1.0, min(1.0, val))

    def _decode_network_strafe_input(self, ctx: ClientContext, strafe_val: float) -> float:
        """Decode OG slot-3 semantics into world-space strafe.

        `Tank_read_control_inputs` negates the button-normalized slot-3 input
        before the tank controller consumes it. The OG client therefore sends
        rightward strafe as a negative slot value and leftward strafe as a
        positive slot value on the wire. Convert that back into world-space
        strafe here so negative = left and positive = right in simulation.
        """
        return self.strafe_sign * self._normalize_behavior_axis_value(ctx, strafe_val)

    def _get_jumpjet_input(self, ctx: ClientContext) -> float:
        """Get digital jumpjet action input from OG behavior slot 4."""
        injected = getattr(ctx, "injected_jumpjet", None)
        if injected is not None:
            return 1.0 if float(injected) >= 0.5 else 0.0
        # Backward compatibility for older tests/control scripts that used the
        # upward-thrust override before slot 4 was split out as jumpjet.
        injected = getattr(ctx, "injected_thrust", None)
        if injected is not None:
            return 1.0 if float(injected) >= 0.5 else 0.0
        if getattr(ctx, "weapon_system", None) is None:
            return 0.0
        return 1.0 if ctx.weapon_system.behavior_slots[BehaviorSlot.JUMPJET] >= 0.5 else 0.0

    def _reset_jump_jet_state(self, ctx: ClientContext) -> None:
        """Reset fixed-step jump-jet prediction state on spawn/respawn."""
        ctx.jump_prev_thrust_input = 0.0
        ctx.jump_cooldown_remaining = 0.0
        ctx.jump_spawn_lockout = JUMP_JET_SPAWN_LOCKOUT
        if getattr(ctx, "jump_jet_system", None) is not None:
            try:
                ctx.jump_jet_system.reset_player(ctx.session.player_id or ctx.entity_id)
            except Exception:
                pass

    def _jump_jet_direction_vector(
        self,
        ctx: ClientContext,
        *,
        vertical_idx: int,
    ) -> tuple[float, float, float]:
        """Return the world-space direction for a jumpjet impulse."""
        world_up = (0.0, 0.0, 1.0) if vertical_idx == 2 else (0.0, 1.0, 0.0)
        if (
            getattr(self, "jump_jet_direction", "body") != "body"
            or vertical_idx != 2
            or ctx.entity_type != EntityType.TANK
        ):
            return world_up

        try:
            matrix = tank_body_matrix_with_heading(
                getattr(ctx, "spring_body_matrix", None),
                float(getattr(ctx, "player_heading", 0.0) or 0.0),
                fallback_roll=float((getattr(ctx, "player_pose", {}) or {}).get("roll", 0.0) or 0.0),
                fallback_pitch=float((getattr(ctx, "player_pose", {}) or {}).get("pitch", 0.0) or 0.0),
            )
            body_up = (float(matrix[2]), float(matrix[5]), float(matrix[8]))
        except (TypeError, ValueError, IndexError):
            return world_up

        mag_sq = body_up[0] * body_up[0] + body_up[1] * body_up[1] + body_up[2] * body_up[2]
        if not math.isfinite(mag_sq) or mag_sq <= 1e-10:
            return world_up
        inv_mag = 1.0 / math.sqrt(mag_sq)
        direction = (body_up[0] * inv_mag, body_up[1] * inv_mag, body_up[2] * inv_mag)
        if direction[2] <= 0.05:
            return world_up
        return direction

    def _apply_jump_jets_fixed_step(
        self,
        ctx: ClientContext,
        *,
        dt: float,
        jumpjet_input: float,
        current_altitude: float,
        current_vel_up: float,
        direction: tuple[float, float, float],
        vertical_idx: int,
    ) -> tuple[bool, float, tuple[float, float, float]]:
        """Apply opt-in custom jump jets in the deterministic movement frame."""
        ctx.jump_cooldown_remaining = max(
            0.0,
            float(getattr(ctx, "jump_cooldown_remaining", 0.0)) - dt,
        )
        ctx.jump_spawn_lockout = max(
            0.0,
            float(getattr(ctx, "jump_spawn_lockout", 0.0)) - dt,
        )

        impulse = 0.0
        fired = False
        impulse_vector = (0.0, 0.0, 0.0)
        cfg = JUMP_JET_CONFIGS.get(ctx.entity_type) if getattr(self, "jump_jets_enabled", False) else None
        if cfg is not None and ctx.jump_spawn_lockout <= 0.0:
            rising_edge = ctx.jump_prev_thrust_input < 0.5 and jumpjet_input >= 0.5
            if (
                rising_edge
                and ctx.jump_cooldown_remaining <= 0.0
                and current_altitude < cfg.max_altitude
                and ctx.player_energy >= cfg.fuel_cost
            ):
                impulse = cfg.impulse
                impulse_vector = (
                    impulse * direction[0],
                    impulse * direction[1],
                    impulse * direction[2],
                )
                current_vel_up += impulse_vector[vertical_idx]
                ctx.jump_cooldown_remaining = cfg.cooldown
                fired = True
                if cfg.fuel_cost > 0.0:
                    self._consume_player_energy(ctx, cfg.fuel_cost)
                player_id = ctx.session.player_id or ctx.entity_id
                self._on_jump_jet_triggered(ctx, player_id, impulse, current_vel_up)

        ctx.jump_prev_thrust_input = jumpjet_input
        return fired, impulse, impulse_vector
