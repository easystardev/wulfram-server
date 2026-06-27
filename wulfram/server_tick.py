"""TickMixin -- per-tick physics: tank surface/attitude sampling, turn torque,
input decode, and jump-jet stepping, extracted verbatim from WulframServer
(server.py decomposition, step 5). Method-only mixin; shares state via `self`.
"""
from __future__ import annotations

import math
import os
import time
from typing import Mapping, Optional, Sequence

from .client import ClientContext
from .physics import _extract_euler_angles, _matrix3_from_euler_xyz, _normalize_angle_client
from .weapons import BehaviorSlot, EntityType, VEHICLE_PHYSICS_CONFIGS
from .world_collision import TerrainContact
from .packets import get_ticks
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
    entity_interpolate_toward_target_decision,
    mesh_aabb_half_extents_from_vertices,
    resolve_iterative_terrain_start_contact,
    solve_static_terrain_constraint,
    OG_PHYSICS_TIMESTEP_FACTOR,
    OG_TANK_SPRING_STRETCH_SPEED_DENOMINATOR,
    tank_body_matrix_drive_basis,
    tank_slope_mobility_factor,
    tank_softbody_control_slot_value,
    tank_softbody_horizontal_damping,
    tank_softbody_suspension_force,
    tank_spring_scalar_stretch_ratio,
    tank_suspension_lift_accel,
    tank_terrain_contact_coupling,
    vehicle_runtime_speed,
)


from .packets import build_chat_message


# --- Numeric-validation helpers hoisted verbatim from the nested closures of
# _resolve_entity_world_collision (server_tick.py decomposition). They are PURE (no
# self/method-local capture), so lifting them to module scope is behavior-identical:
# the method's existing unqualified call sites resolve to these by Python scoping with
# zero call-site edits. (sane_position_triplet stays nested -- it needs self.world_bound.)
def finite_values(values) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError, OverflowError):
        return False


def finite_triplet(value):
    if value is None:
        return None
    try:
        if len(value) < 3:
            return None
        result = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, OverflowError):
        return None
    return result if finite_values(result) else None


def sane_velocity_triplet(value):
    result = finite_triplet(value)
    if result is None:
        return None
    if any(abs(component) > 10000.0 for component in result):
        return None
    return result


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

    def _get_entity_world_half_extents(self, ctx: ClientContext) -> tuple[float, float, float]:
        team_id = ctx.session.team_id or 1
        cache_key = (ctx.entity_type, team_id)
        cached = self._entity_collision_extents_cache.get(cache_key)
        if cached is not None:
            return cached

        half_extents = (self._TANK_RADIUS, self._TANK_RADIUS, self._TANK_RADIUS)
        model_names = self._ENTITY_WORLD_MODEL_NAMES.get(ctx.entity_type)
        if model_names and self._building_collision.available:
            model_name = self._select_team_model_name(model_names, team_id)
            model = self._building_collision.models.get(model_name)
            mesh = getattr(model, "collision_mesh", None) if model is not None else None
            vertices = getattr(mesh, "vertices", None) if mesh is not None else None
            if vertices:
                xs = [v.x for v in vertices]
                ys = [v.y for v in vertices]
                zs = [v.z for v in vertices]
                half_extents = (
                    max(self._TANK_RADIUS, max(abs(min(xs)), abs(max(xs)))),
                    max(self._TANK_RADIUS, max(abs(min(ys)), abs(max(ys)))),
                    max(abs(min(zs)), abs(max(zs))),
                )

        self._entity_collision_extents_cache[cache_key] = half_extents
        return half_extents

    def _get_entity_dirty_threshold_sq(
        self,
        ctx: ClientContext,
        fallback_half_extents: tuple[float, float, float],
    ) -> float:
        if not hasattr(self, "_entity_dirty_threshold_sq_cache"):
            self._entity_dirty_threshold_sq_cache = {}
        team_id = ctx.session.team_id or 1
        cache_key = (ctx.entity_type, team_id)
        cached = self._entity_dirty_threshold_sq_cache.get(cache_key)
        if cached is not None:
            return cached

        min_half_extent = min(fallback_half_extents)
        model_names = self._ENTITY_WORLD_MODEL_NAMES.get(ctx.entity_type)
        building_collision = getattr(self, "_building_collision", None)
        if model_names and building_collision is not None and building_collision.available:
            model_name = self._select_team_model_name(model_names, team_id)
            model = building_collision.models.get(model_name)
            mesh = getattr(model, "collision_mesh", None) if model is not None else None
            vertices = getattr(mesh, "vertices", None) if mesh is not None else None
            if vertices:
                min_half_extent = min(
                    max(abs(v.x) for v in vertices),
                    max(abs(v.y) for v in vertices),
                    max(abs(v.z) for v in vertices),
                )

        threshold_sq = (min_half_extent * 0.8) * (min_half_extent * 0.8)
        self._entity_dirty_threshold_sq_cache[cache_key] = threshold_sq
        return threshold_sq

    @staticmethod
    def _get_static_separation_from_contact(
        entity_pos: tuple[float, float, float],
        contact_point: tuple[float, float, float],
    ) -> float:
        distance = math.sqrt(
            (contact_point[0] - entity_pos[0]) * (contact_point[0] - entity_pos[0]) +
            (contact_point[1] - entity_pos[1]) * (contact_point[1] - entity_pos[1]) +
            (contact_point[2] - entity_pos[2]) * (contact_point[2] - entity_pos[2])
        )
        separation = distance * 0.03
        if separation > 0.5:
            return 0.5
        if separation <= 0.01:
            return 0.01
        return separation

    @staticmethod
    def _is_pathological_dirty_bounds_contact(
        entity_pos: tuple[float, float, float],
        contact,
        bounding_radius: float,
    ) -> bool:
        contact_distance = math.sqrt(
            (contact.position[0] - entity_pos[0]) * (contact.position[0] - entity_pos[0]) +
            (contact.position[1] - entity_pos[1]) * (contact.position[1] - entity_pos[1]) +
            (contact.position[2] - entity_pos[2]) * (contact.position[2] - entity_pos[2])
        )
        if contact_distance > max(bounding_radius * 1.5, 8.0):
            return True
        if contact.normal[2] <= 0.0:
            return True
        normal_z = contact.normal[2]
        penetration_limit = max(bounding_radius * 1.25, 8.0)
        return normal_z < 0.1 and contact.penetration > penetration_limit

    def _cached_mesh_aabb_half_extents(self, vertices):
        """mesh_aabb_half_extents_from_vertices is a pure function of the (static)
        collision-model vertices but was recomputed every physics step (~525 vertex
        reads/call, a large fraction of the rough-cell collision cost). The model
        vertices are held by _entity_collision_model_cache, so id(vertices) is a stable
        cache key. Parity: returns the identical value, just memoised."""
        if vertices is None:
            return None
        cache = getattr(self, "_mesh_aabb_half_extents_cache", None)
        if cache is None:
            cache = {}
            self._mesh_aabb_half_extents_cache = cache
        key = id(vertices)
        cached = cache.get(key)
        if cached is None:
            cached = (mesh_aabb_half_extents_from_vertices(vertices), True)
            cache[key] = cached
        return cached[0]

    def _resolve_entity_world_collision(
        self,
        ctx,
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        *,
        pre_pos=None,
        pre_vel=None,
        dt=None,
    ):
        if self._terrain_grid_collision is None:
            return px, py, pz, vx, vy, vz
        if not getattr(self, "entity_terrain_collision_enabled", True):
            ctx.world_collision_bounds_dirty = False
            return px, py, pz, vx, vy, vz
        if (
            ctx.ground_level_override is not None
            and not getattr(self, "terrain_collision_with_ground_override", False)
        ):
            return px, py, pz, vx, vy, vz
        if pre_pos is None:
            pre_pos = getattr(ctx, "_world_collision_step_pre_pos", None)
        if pre_vel is None:
            pre_vel = getattr(ctx, "_world_collision_step_pre_vel", None)
        if dt is None:
            dt = getattr(ctx, "_world_collision_step_dt", None)

        # finite_values / finite_triplet / sane_velocity_triplet hoisted to module
        # scope (top of file); only sane_position_triplet stays nested (needs world_bound).
        def sane_position_triplet(value):
            result = finite_triplet(value)
            if result is None:
                return None
            pos_limit = max(float(getattr(self, "world_bound", 8192.0) or 8192.0) * 4.0, 32768.0)
            if any(abs(component) > pos_limit for component in result):
                return None
            return result

        def finish_result(px_out, py_out, pz_out, vx_out, vy_out, vz_out, *, reason="terrain_motion_nonfinite_output"):
            if (
                sane_position_triplet((px_out, py_out, pz_out)) is not None
                and sane_velocity_triplet((vx_out, vy_out, vz_out)) is not None
            ):
                return px_out, py_out, pz_out, vx_out, vy_out, vz_out

            fallback_pos = (
                sane_position_triplet(pre_pos)
                or sane_position_triplet(getattr(ctx, "player_pos", None))
                or (0.0, 0.0, float(getattr(self, "ground_level", 0.0) or 0.0))
            )
            fallback_vel = (
                sane_velocity_triplet(pre_vel)
                or sane_velocity_triplet(getattr(ctx, "player_vel", None))
                or (0.0, 0.0, 0.0)
            )
            ctx.debug_last_collision = {
                "kind": reason,
                "bad_pos": (px_out, py_out, pz_out),
                "bad_vel": (vx_out, vy_out, vz_out),
                "fallback_pos": fallback_pos,
                "fallback_vel": fallback_vel,
            }
            ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
            ctx.world_collision_bounds_dirty = False
            ctx.world_collision_ref_pos = fallback_pos
            return (
                fallback_pos[0],
                fallback_pos[1],
                fallback_pos[2],
                fallback_vel[0],
                fallback_vel[1],
                fallback_vel[2],
            )

        if (
            sane_position_triplet((px, py, pz)) is None
            or sane_velocity_triplet((vx, vy, vz)) is None
        ):
            return finish_result(px, py, pz, vx, vy, vz, reason="terrain_motion_nonfinite_input")

        half_extents = self._get_entity_world_half_extents(ctx)
        heading = ctx.player_heading
        anchor = [px, py, pz]
        reference_pos = getattr(ctx, "world_collision_ref_pos", None) or ctx.player_pos
        origin_mode = (
            os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", "lift").strip().lower()
        )
        contact_response = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", "auto").strip().lower()
        )
        ctx_entity_type = getattr(ctx, "entity_type", EntityType.TANK)
        if not isinstance(ctx_entity_type, EntityType):
            try:
                ctx_entity_type = EntityType(int(ctx_entity_type))
            except (TypeError, ValueError):
                ctx_entity_type = EntityType.TANK
        tank_clean_pair_solver_enabled = (
            os.environ.get("WULFRAM_TANK_CLEAN_TERRAIN_PAIR_SOLVER", "1")
            .strip()
            .lower()
            not in {"0", "false", "off", "no", "disabled", "legacy"}
        )
        try:
            tank_clean_pair_solver_max_depth = float(
                os.environ.get("WULFRAM_TANK_CLEAN_TERRAIN_MAX_DEPTH", "10.0")
            )
        except ValueError:
            tank_clean_pair_solver_max_depth = 10.0
        pair_solver_response = (
            contact_response in {"pair", "solver", "constraint"}
            or (
                contact_response == "auto"
                and origin_mode in {"entity", "origin", "raw"}
            )
            or (
                contact_response == "auto"
                and ctx_entity_type == EntityType.TANK
                and tank_clean_pair_solver_enabled
            )
        )
        contact_timing_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING", "auto").strip().lower()
        )
        endpoint_vel_for_timing = (vx, vy, vz)
        timing_ready = (
            pre_pos is not None
            and pre_vel is not None
            and dt is not None
            and float(dt) > 0.0
        )
        timed_pair_response = pair_solver_response and timing_ready and (
            contact_timing_mode in {"1", "true", "on", "pair", "solver", "sweep", "toi", "probe", "bucket", "loop"}
            or (
                contact_timing_mode == "auto"
                and origin_mode in {"entity", "origin", "raw"}
            )
        )
        default_contact_iterations = (
            1
            if contact_timing_mode in {"probe", "sweep", "toi"}
            else 8
        )
        try:
            contact_iteration_limit = int(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS",
                    str(default_contact_iterations),
                )
            )
        except ValueError:
            contact_iteration_limit = default_contact_iterations
        contact_iteration_limit = max(1, min(30, contact_iteration_limit))
        contact_sweep_scan_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN", "0")
            .strip()
            .lower()
        )
        contact_sweep_scan_enabled = contact_sweep_scan_mode in {
            "1",
            "true",
            "on",
            "yes",
            "scan",
            "bucket",
            "decompile",
        }
        try:
            contact_sweep_scan_steps = int(
                os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS", "30")
            )
        except ValueError:
            contact_sweep_scan_steps = 30
        contact_sweep_scan_steps = max(1, min(30, contact_sweep_scan_steps))
        start_iterative_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_START_ITERATIVE", "0").strip().lower()
        )
        start_iterative_enabled = start_iterative_mode in {
            "1",
            "true",
            "on",
            "yes",
            "iterative",
            "decompile",
        }
        try:
            start_iterative_limit = int(
                os.environ.get("WULFRAM_ENTITY_TERRAIN_START_ITERATIVE_LIMIT", "40")
            )
        except ValueError:
            start_iterative_limit = 40
        start_iterative_limit = max(1, min(200, start_iterative_limit))
        start_time_clamp_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_START_TIME_CLAMP", "0").strip().lower()
        )
        start_time_clamp_enabled = start_time_clamp_mode in {
            "1",
            "true",
            "on",
            "yes",
            "clamp",
            "decompile",
        }
        collision_model = self._get_entity_world_collision_model(ctx)
        if collision_model is not None:
            vertices, cbsp_tree, bounding_radius, z_lift = collision_model
            inertia_half_extents = (
                self._cached_mesh_aabb_half_extents(vertices) or half_extents
            )
        else:
            vertices = None
            cbsp_tree = None
            bounding_radius = math.sqrt(
                half_extents[0] * half_extents[0] +
                half_extents[1] * half_extents[1] +
                half_extents[2] * half_extents[2]
            )
            z_lift = half_extents[2]
            inertia_half_extents = half_extents
        terrain_collision_shape = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE", "model")
            .strip()
            .lower()
        )
        if terrain_collision_shape in {"box", "obb", "hull", "aabb", "decompile"}:
            terrain_collision_shape = "box"
        elif terrain_collision_shape in {
            "entity_box",
            "raw_box",
            "origin_box",
            "decompile_entity",
        }:
            terrain_collision_shape = "entity_box"
        else:
            terrain_collision_shape = "model"
        body_ang_vel = getattr(ctx, "spring_body_ang_vel", (0.0, 0.0)) or (0.0, 0.0)
        contact_angular_velocity = [
            float(body_ang_vel[0]) if len(body_ang_vel) > 0 else 0.0,
            float(body_ang_vel[1]) if len(body_ang_vel) > 1 else 0.0,
            float(getattr(ctx, "angular_vel_yaw", 0.0) or 0.0),
        ]
        model_contact_rotation_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_MODEL_CONTACT_ROTATION", "heading")
            .strip()
            .lower()
        )
        if model_contact_rotation_mode in {
            "1",
            "true",
            "on",
            "yes",
            "body",
            "matrix",
            "spring",
            "decompile",
            "full",
        }:
            model_contact_rotation_matrix = tank_body_matrix_with_heading(
                getattr(ctx, "spring_body_matrix", None),
                heading,
                fallback_roll=float((getattr(ctx, "player_pose", {}) or {}).get("roll", 0.0) or 0.0),
                fallback_pitch=float((getattr(ctx, "player_pose", {}) or {}).get("pitch", 0.0) or 0.0),
            )
            model_contact_rotation_source = "body_matrix"
        else:
            model_contact_rotation_matrix = None
            model_contact_rotation_source = "heading_only"
        model_contact_selection = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_MODEL_CONTACT_SELECTION", "first")
            .strip()
            .lower()
        )
        dirty_model_center_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_DIRTY_MODEL_CENTER", "lift")
            .strip()
            .lower()
        )
        dirty_bounds_box_fallback_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_FALLBACK", "0")
            .strip()
            .lower()
        )
        dirty_bounds_box_fallback_enabled = dirty_bounds_box_fallback_mode in {
            "1",
            "true",
            "on",
            "yes",
            "box",
            "aabb",
            "decompile",
        }
        dirty_bounds_box_shape = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_SHAPE", "inertia")
            .strip()
            .lower()
        )
        dirty_bounds_box_center_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_CENTER", "raw")
            .strip()
            .lower()
        )
        dirty_miss_refresh_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_DIRTY_MISS_REFRESH", "1")
            .strip()
            .lower()
        )
        dirty_miss_refresh_enabled = dirty_miss_refresh_mode not in {
            "0",
            "false",
            "off",
            "no",
            "hold",
            "preserve",
        }
        dirty_reference_pair_probe_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_PROBE", "0")
            .strip()
            .lower()
        )
        dirty_reference_pair_probe_enabled = dirty_reference_pair_probe_mode in {
            "1",
            "true",
            "on",
            "yes",
            "probe",
            "reference",
            "dirty_reference",
            "decompile",
        }
        dirty_reference_pair_response_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_RESPONSE", "0")
            .strip()
            .lower()
        )
        dirty_reference_pair_response_enabled = dirty_reference_pair_response_mode in {
            "1",
            "true",
            "on",
            "yes",
            "probe",
            "apply",
            "contact",
            "response",
            "decompile",
        }
        dirty_reference_pair_response_apply_enabled = (
            dirty_reference_pair_response_enabled
            and dirty_reference_pair_response_mode
            in {"apply", "contact", "response", "decompile"}
        )
        try:
            dirty_reference_pair_response_max_distance = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_RESPONSE_MAX_DISTANCE",
                    "0",
                )
            )
        except (TypeError, ValueError):
            dirty_reference_pair_response_max_distance = 0.0
        if dirty_reference_pair_response_max_distance < 0.0:
            dirty_reference_pair_response_max_distance = 0.0
        if dirty_reference_pair_response_enabled:
            dirty_reference_pair_probe_enabled = True
        dirty_bounds_safe_response_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_SAFE_RESPONSE", "0")
            .strip()
            .lower()
        )
        dirty_bounds_safe_response_enabled = dirty_bounds_safe_response_mode in {
            "1",
            "true",
            "on",
            "yes",
            "safe",
            "safety",
            "limited",
            "decompile",
        }

        def box_collision_z_lift() -> float:
            if terrain_collision_shape == "entity_box":
                return 0.0
            return z_lift if collision_model is not None else half_extents[2]

        def dirty_bounds_box_half_extents():
            if dirty_bounds_box_shape in {
                "radius",
                "radius_cube",
                "bounding_radius",
                "sphere_aabb",
                "decompile_radius",
            }:
                return (
                    (bounding_radius, bounding_radius, bounding_radius),
                    "bounding_radius",
                )
            if dirty_bounds_box_shape in {
                "fallback",
                "fallback_half_extents",
                "tank",
                "tank_radius",
                "legacy",
            }:
                return half_extents, "fallback_half_extents"
            return inertia_half_extents, "inertia_half_extents"

        def dirty_bounds_box_center():
            if dirty_bounds_box_center_mode in {
                "lift",
                "lifted",
                "model",
                "collision",
                "z_lift",
            }:
                z_offset = box_collision_z_lift()
                center_mode = "lift"
            else:
                z_offset = 0.0
                center_mode = "raw"
            center = (anchor[0], anchor[1], anchor[2] + z_offset)
            return center, center_mode, z_offset

        def contact_debug_fields(contact):
            return {
                "contact_sector_index": getattr(contact, "sector_index", None),
                "contact_cell": getattr(contact, "cell", None),
                "contact_normal_source": getattr(contact, "normal_source", None),
                "contact_cbsp_split_normal": getattr(contact, "cbsp_split_normal", None),
                "contact_terrain_face_normal": getattr(contact, "terrain_face_normal", None),
                "contact_mesh_face_normal": getattr(contact, "mesh_face_normal", None),
                "contact_entity_radial_normal": getattr(contact, "entity_radial_normal", None),
                "contact_cbsp_store_normal0": getattr(contact, "cbsp_store_normal0", None),
                "contact_cbsp_store_normal1": getattr(contact, "cbsp_store_normal1", None),
                "contact_cbsp_record_hit_source": getattr(contact, "cbsp_record_hit_source", None),
                "contact_cbsp_mesh_triangle_indices": getattr(contact, "cbsp_mesh_triangle_indices", None),
                "contact_cbsp_guess7_order": getattr(contact, "cbsp_guess7_order", None),
                "contact_cbsp_guess7_terms": getattr(contact, "cbsp_guess7_terms", None),
                "contact_cbsp_edge_hit_kind": getattr(contact, "cbsp_edge_hit_kind", None),
                "contact_cbsp_edge_t": getattr(contact, "cbsp_edge_t", None),
                "contact_cbsp_node_index": getattr(contact, "cbsp_node_index", None),
                "contact_cbsp_node_depth": getattr(contact, "cbsp_node_depth", None),
                "contact_cbsp_node_mesh_normal_angle_deg": getattr(
                    contact,
                    "cbsp_node_mesh_normal_angle_deg",
                    None,
                ),
            }

        contact_probe_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_PROBE", "1").strip().lower()
        )
        contact_probe_enabled = contact_probe_mode not in {
            "0",
            "false",
            "off",
            "no",
            "disabled",
        }
        reference_pose_probe_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_PROBE", "0")
            .strip()
            .lower()
        )
        reference_pose_probe_enabled = reference_pose_probe_mode in {
            "1",
            "true",
            "on",
            "yes",
            "reference",
            "pre",
            "midpoint",
            "decompile",
        }
        reference_pose_contact_response_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_CONTACT", "0")
            .strip()
            .lower()
        )
        reference_pose_contact_response_enabled = (
            reference_pose_contact_response_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "apply",
                "contact",
                "pair",
                "response",
                "decompile",
            }
        )
        if reference_pose_contact_response_enabled:
            reference_pose_probe_enabled = True
        reference_pose_pair_response_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_PAIR_RESPONSE", "0")
            .strip()
            .lower()
        )
        reference_pose_pair_response_enabled = reference_pose_pair_response_mode in {
            "1",
            "true",
            "on",
            "yes",
            "probe",
            "apply",
            "contact",
            "response",
            "decompile",
        }
        reference_pose_pair_response_apply_enabled = (
            reference_pose_pair_response_enabled
            and reference_pose_pair_response_mode
            in {"apply", "contact", "response", "decompile"}
        )
        try:
            reference_pose_pair_response_max_distance = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_PAIR_RESPONSE_MAX_DISTANCE",
                    "0",
                )
            )
        except (TypeError, ValueError):
            reference_pose_pair_response_max_distance = 0.0
        if reference_pose_pair_response_max_distance < 0.0:
            reference_pose_pair_response_max_distance = 0.0
        if reference_pose_pair_response_enabled:
            reference_pose_probe_enabled = True
        reference_pose_contact_order = tuple(
            label.strip()
            for label in os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_CONTACT_ORDER",
                (
                    "pre_to_current_75,pre_to_current_50,pre_to_current_25,"
                    "pre_pos,dirty_reference_pos,world_collision_ref_pos"
                ),
            ).split(",")
            if label.strip()
        )
        raw_fallback_env = os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK")
        raw_fallback_mode = (raw_fallback_env if raw_fallback_env is not None else "0").strip().lower()
        raw_fallback_enabled = raw_fallback_mode in {
            "1",
            "true",
            "on",
            "yes",
            "raw",
            "fallback",
            "decompile",
        }
        raw_fallback_timed_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_TIMED_FALLBACK", "0")
            .strip()
            .lower()
        )
        raw_fallback_timed_enabled = raw_fallback_enabled and raw_fallback_timed_mode in {
            "1",
            "true",
            "on",
            "yes",
            "timed",
            "sweep",
            "bucket",
            "decompile",
        }
        try:
            raw_fallback_min_depth = float(
                os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_DEPTH", "2.0")
            )
        except ValueError:
            raw_fallback_min_depth = 2.0
        try:
            raw_fallback_max_depth = float(
                os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_DEPTH", "8.0")
            )
        except ValueError:
            raw_fallback_max_depth = 8.0
        try:
            raw_fallback_min_normal_z = float(
                os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_NORMAL_Z", "0.5")
            )
        except ValueError:
            raw_fallback_min_normal_z = 0.5
        try:
            raw_fallback_min_speed = float(
                os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_SPEED", "5.0")
            )
        except ValueError:
            raw_fallback_min_speed = 5.0
        try:
            raw_fallback_max_velocity_delta = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
                    "20.0",
                )
            )
        except ValueError:
            raw_fallback_max_velocity_delta = 20.0
        try:
            raw_fallback_max_speed = float(
                os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED", "200.0")
            )
        except ValueError:
            raw_fallback_max_speed = 200.0
        try:
            raw_fallback_max_angular_delta = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
                    "0.5",
                )
            )
        except ValueError:
            raw_fallback_max_angular_delta = 0.5
        raw_fallback_projection_order = os.environ.get(
            "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_PROJECTION_ORDER",
            "opposite_if_separating",
        )
        raw_fallback_normal_source = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE", "mesh")
            .strip()
            .lower()
        )
        raw_fallback_delta_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE", "solver")
            .strip()
            .lower()
        )
        raw_fallback_angular_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE", "solver")
            .strip()
            .lower()
        )
        raw_fallback_vertical_delta_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_VERTICAL_DELTA_MODE",
                "component",
            )
            .strip()
            .lower()
        )
        raw_fallback_closing_only = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY", "1")
            .strip()
            .lower()
            not in {"0", "false", "off", "no", "disabled"}
        )
        raw_fallback_friction = None
        raw_fallback_friction_env = os.environ.get(
            "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION"
        )
        if raw_fallback_friction_env not in (None, "", "default"):
            try:
                raw_fallback_friction = max(0.0, float(raw_fallback_friction_env))
            except ValueError:
                raw_fallback_friction = None
        tank_raw_fallback_env = os.environ.get("WULFRAM_TANK_RAW_ORIGIN_FALLBACK")
        tank_raw_fallback_mode = (
            tank_raw_fallback_env if tank_raw_fallback_env is not None else "1"
        ).strip().lower()
        tank_raw_fallback_auto_enabled = (
            ctx_entity_type == EntityType.TANK
            and contact_response == "auto"
            and origin_mode not in {"entity", "origin", "raw"}
            and raw_fallback_env is None
            and tank_raw_fallback_mode
            not in {"0", "false", "off", "no", "disabled", "legacy"}
        )
        tank_raw_fallback_normal_source = (
            os.environ.get("WULFRAM_TANK_RAW_ORIGIN_NORMAL_SOURCE", "terrain_face")
            .strip()
            .lower()
        )
        tank_raw_fallback_delta_normal_mode = (
            os.environ.get("WULFRAM_TANK_RAW_ORIGIN_DELTA_NORMAL", "horizontal_face")
            .strip()
            .lower()
        )
        try:
            tank_raw_fallback_min_depth = float(
                os.environ.get("WULFRAM_TANK_RAW_ORIGIN_MIN_DEPTH", "2.0")
            )
        except ValueError:
            tank_raw_fallback_min_depth = 2.0
        try:
            tank_raw_fallback_max_depth = float(
                os.environ.get("WULFRAM_TANK_RAW_ORIGIN_MAX_DEPTH", "10.0")
            )
        except ValueError:
            tank_raw_fallback_max_depth = 10.0
        try:
            tank_raw_fallback_min_normal_z = float(
                os.environ.get("WULFRAM_TANK_RAW_ORIGIN_MIN_NORMAL_Z", "0.4")
            )
        except ValueError:
            tank_raw_fallback_min_normal_z = 0.4
        try:
            tank_raw_fallback_min_face_normal_z = float(
                os.environ.get("WULFRAM_TANK_RAW_ORIGIN_MIN_FACE_NORMAL_Z", "0.4")
            )
        except ValueError:
            tank_raw_fallback_min_face_normal_z = 0.4
        try:
            tank_raw_fallback_min_speed = float(
                os.environ.get("WULFRAM_TANK_RAW_ORIGIN_MIN_SPEED", "5.0")
            )
        except ValueError:
            tank_raw_fallback_min_speed = 5.0
        try:
            tank_raw_fallback_max_velocity_delta = float(
                os.environ.get("WULFRAM_TANK_RAW_ORIGIN_MAX_VELOCITY_DELTA", "18.0")
            )
        except ValueError:
            tank_raw_fallback_max_velocity_delta = 18.0
        try:
            tank_raw_fallback_max_vertical_delta = float(
                os.environ.get("WULFRAM_TANK_RAW_ORIGIN_MAX_VERTICAL_DELTA", "8.0")
            )
        except ValueError:
            tank_raw_fallback_max_vertical_delta = 8.0
        try:
            tank_raw_fallback_max_speed = float(
                os.environ.get("WULFRAM_TANK_RAW_ORIGIN_MAX_SPEED", "200.0")
            )
        except ValueError:
            tank_raw_fallback_max_speed = 200.0
        try:
            tank_raw_fallback_max_angular_delta = float(
                os.environ.get("WULFRAM_TANK_RAW_ORIGIN_MAX_ANGULAR_DELTA", "0.0")
            )
        except ValueError:
            tank_raw_fallback_max_angular_delta = 0.0
        tank_raw_fallback_projection_order = os.environ.get(
            "WULFRAM_TANK_RAW_ORIGIN_PROJECTION_ORDER",
            "opposite_if_separating",
        )
        tank_raw_fallback_delta_mode = (
            os.environ.get("WULFRAM_TANK_RAW_ORIGIN_DELTA_MODE", "closing_velocity")
            .strip()
            .lower()
        )
        tank_raw_fallback_angular_mode = (
            os.environ.get("WULFRAM_TANK_RAW_ORIGIN_ANGULAR_MODE", "preserve")
            .strip()
            .lower()
        )
        tank_raw_fallback_vertical_delta_mode = (
            os.environ.get("WULFRAM_TANK_RAW_ORIGIN_VERTICAL_DELTA_MODE", "component")
            .strip()
            .lower()
        )
        tank_raw_fallback_closing_only = (
            os.environ.get("WULFRAM_TANK_RAW_ORIGIN_CLOSING_ONLY", "1")
            .strip()
            .lower()
            not in {"0", "false", "off", "no", "disabled"}
        )
        tank_clean_face_fallback_mode = (
            os.environ.get("WULFRAM_TANK_CLEAN_TERRAIN_FACE_FALLBACK", "1")
            .strip()
            .lower()
        )
        tank_clean_face_fallback_enabled = (
            ctx_entity_type == EntityType.TANK
            and contact_response == "auto"
            and origin_mode not in {"entity", "origin", "raw"}
            and tank_clean_face_fallback_mode
            not in {"0", "false", "off", "no", "disabled", "legacy"}
        )
        tank_face_fallback_latch_enabled = (
            os.environ.get("WULFRAM_TANK_FACE_FALLBACK_LATCH", "0")
            .strip()
            .lower()
            not in {"0", "false", "off", "no", "disabled", "legacy"}
        )
        tank_clean_face_fallback_delta_mode = (
            os.environ.get(
                "WULFRAM_TANK_CLEAN_TERRAIN_FACE_FALLBACK_DELTA_MODE",
                "center_closing_velocity",
            )
            .strip()
            .lower()
        )
        tank_clean_face_fallback_delta_normal_mode = (
            os.environ.get(
                "WULFRAM_TANK_CLEAN_TERRAIN_FACE_FALLBACK_DELTA_NORMAL",
                "horizontal_face",
            )
            .strip()
            .lower()
        )
        try:
            tank_clean_face_fallback_max_contact_normal_z = float(
                os.environ.get(
                    "WULFRAM_TANK_CLEAN_TERRAIN_FACE_FALLBACK_MAX_CONTACT_NORMAL_Z",
                    "0.2",
                )
            )
        except ValueError:
            tank_clean_face_fallback_max_contact_normal_z = 0.2
        try:
            tank_clean_face_fallback_min_face_normal_z = float(
                os.environ.get(
                    "WULFRAM_TANK_CLEAN_TERRAIN_FACE_FALLBACK_MIN_FACE_NORMAL_Z",
                    str(tank_raw_fallback_min_face_normal_z),
                )
            )
        except ValueError:
            tank_clean_face_fallback_min_face_normal_z = tank_raw_fallback_min_face_normal_z
        pair_record_contact_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", "1")
            .strip()
            .lower()
        )
        pair_record_contact_enabled = pair_record_contact_mode not in {
            "0",
            "false",
            "off",
            "no",
            "disabled",
        }
        pair_record_contact_selection = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
                "upward_min_depth",
            )
            .strip()
            .lower()
        )
        pair_record_bounds_sat_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_BOUNDS_SAT", "0")
            .strip()
            .lower()
        )
        pair_record_bounds_sat_enabled = pair_record_bounds_sat_mode in {
            "1",
            "true",
            "on",
            "yes",
            "probe",
            "report",
            "readonly",
            "apply",
            "contact",
            "decompile",
        }
        pair_record_bounds_sat_apply_enabled = (
            pair_record_bounds_sat_enabled
            and pair_record_bounds_sat_mode
            in {"1", "true", "on", "yes", "apply", "contact", "decompile"}
        )
        try:
            pair_record_contact_min_depth = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MIN_DEPTH",
                    str(self._PENETRATION_SLOP_DEFAULT),
                )
            )
        except ValueError:
            pair_record_contact_min_depth = self._PENETRATION_SLOP_DEFAULT
        try:
            pair_record_contact_max_depth = float(
                os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_DEPTH", "10.0")
            )
        except ValueError:
            pair_record_contact_max_depth = 10.0
        try:
            pair_record_contact_min_normal_z = float(
                os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MIN_NORMAL_Z", "0.5")
            )
        except ValueError:
            pair_record_contact_min_normal_z = 0.5
        try:
            pair_record_contact_min_face_normal_z = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MIN_FACE_NORMAL_Z",
                    "0.4",
                )
            )
        except ValueError:
            pair_record_contact_min_face_normal_z = 0.4
        pair_record_contact_normal_source = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE", "mesh")
            .strip()
            .lower()
        )
        pair_record_contact_delta_normal_source = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
                "entity_radial_terrain_face_blend",
            )
            .strip()
            .lower()
        )
        pair_record_contact_projection_order = os.environ.get(
            "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PROJECTION_ORDER",
            "opposite_if_separating",
        )
        pair_record_contact_response_profile = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_RESPONSE_PROFILE",
                "safety_limited",
            )
            .strip()
            .lower()
        )
        if pair_record_contact_response_profile in {
            "",
            "default",
            "safe",
            "safety",
            "safety_limited",
            "limited",
        }:
            pair_record_contact_response_profile = "safety_limited"
        pair_record_decompile_linear_solver = (
            pair_record_contact_response_profile
            in {
                "decompile_linear_solver",
                "decompile_solver_linear",
                "raw_solver_linear",
                "solver_linear",
            }
        )
        pair_record_contact_delta_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
                (
                    "solver_vector"
                    if pair_record_decompile_linear_solver
                    else "closing_velocity"
                ),
            )
            .strip()
            .lower()
        )
        pair_record_contact_angular_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE", "preserve")
            .strip()
            .lower()
        )
        pair_record_contact_vertical_delta_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE",
                "scale",
            )
            .strip()
            .lower()
        )
        pair_record_contact_closing_only = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CLOSING_ONLY", "1")
            .strip()
            .lower()
            not in {"0", "false", "off", "no", "disabled"}
        )
        try:
            pair_record_contact_max_velocity_delta = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
                    "0.0" if pair_record_decompile_linear_solver else "3.0",
                )
            )
        except ValueError:
            pair_record_contact_max_velocity_delta = 3.0
        try:
            pair_record_contact_max_vertical_delta = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
                    "0.0" if pair_record_decompile_linear_solver else "1.0",
                )
            )
        except ValueError:
            pair_record_contact_max_vertical_delta = 1.0
        try:
            pair_record_contact_max_speed = float(
                os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_SPEED", "200.0")
            )
        except ValueError:
            pair_record_contact_max_speed = 200.0
        try:
            pair_record_contact_max_angular_delta = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_ANGULAR_DELTA",
                    "0.5",
                )
            )
        except ValueError:
            pair_record_contact_max_angular_delta = 0.5
        pair_record_cached_contact_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CACHE_CONTACT", "0")
            .strip()
            .lower()
        )
        pair_record_cached_contact_enabled = (
            pair_record_contact_enabled
            and pair_record_cached_contact_mode
            in {"1", "true", "on", "yes", "cache", "cached", "decompile"}
        )
        try:
            pair_record_cached_contact_max_age_steps = int(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CACHE_MAX_AGE_STEPS",
                    "8",
                )
            )
        except ValueError:
            pair_record_cached_contact_max_age_steps = 8
        pair_record_cached_contact_max_age_steps = max(
            1,
            min(60, pair_record_cached_contact_max_age_steps),
        )
        try:
            pair_record_cached_contact_max_distance = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CACHE_MAX_DISTANCE",
                    "6.0",
                )
            )
        except ValueError:
            pair_record_cached_contact_max_distance = 6.0
        try:
            pair_record_cached_contact_max_ref_distance = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CACHE_MAX_REF_DISTANCE",
                    "8.0",
                )
            )
        except ValueError:
            pair_record_cached_contact_max_ref_distance = 8.0
        try:
            pair_record_cache_step = (
                int(getattr(ctx, "terrain_pair_record_contact_cache_step", 0) or 0)
                + 1
            )
        except (TypeError, ValueError, OverflowError):
            pair_record_cache_step = 1
        ctx.terrain_pair_record_contact_cache_step = pair_record_cache_step
        pair_record_timed_contact_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT", "0")
            .strip()
            .lower()
        )
        pair_record_timed_contact_enabled = (
            pair_record_contact_enabled
            and timing_ready
            and pair_record_timed_contact_mode
            not in {"0", "false", "off", "no", "disabled"}
            and contact_timing_mode
            in {
                "auto",
                "1",
                "true",
                "on",
                "pair",
                "solver",
                "sweep",
                "toi",
                "probe",
                "bucket",
                "loop",
            }
        )
        pair_record_timed_sweep_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_SWEEP", "1")
            .strip()
            .lower()
        )
        pair_record_timed_sweep_enabled = (
            pair_record_timed_contact_enabled
            and pair_record_timed_sweep_mode
            not in {"0", "false", "off", "no", "disabled"}
        )
        if pair_record_timed_contact_enabled:
            timed_pair_response = True
            if (
                contact_timing_mode == "auto"
                and "WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS" not in os.environ
            ):
                contact_iteration_limit = 1
            if (
                pair_record_timed_sweep_enabled
                and "WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN" not in os.environ
            ):
                contact_sweep_scan_enabled = True
        pair_record_continue_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING", "0")
            .strip()
            .lower()
        )
        pair_record_continue_remaining_enabled = (
            pair_record_contact_enabled
            and timing_ready
            and pair_record_continue_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "continue",
                "remaining",
                "bucket",
                "decompile",
            }
        )
        pair_record_schedule_probe_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_PROBE", "0")
            .strip()
            .lower()
        )
        pair_record_schedule_probe_enabled = (
            pair_record_continue_remaining_enabled
            or (
                pair_record_contact_enabled
                and timing_ready
                and pair_record_schedule_probe_mode
                in {
                    "1",
                    "true",
                    "on",
                    "yes",
                    "probe",
                    "schedule",
                    "timing",
                    "bucket",
                    "decompile",
                }
            )
        )
        pair_record_spatial_ref_schedule_probe_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SPATIAL_REF_SCHEDULE_PROBE",
                "0",
            )
            .strip()
            .lower()
        )
        pair_record_spatial_ref_schedule_probe_enabled = (
            pair_record_contact_enabled
            and timing_ready
            and pair_record_spatial_ref_schedule_probe_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "probe",
                "spatial",
                "reference",
                "decompile",
            }
        )
        pair_record_frame_phase_probe_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_FRAME_PHASE_PROBE",
                "0",
            )
            .strip()
            .lower()
        )
        pair_record_frame_phase_probe_enabled = (
            pair_record_contact_enabled
            and timing_ready
            and pair_record_frame_phase_probe_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "probe",
                "frame",
                "phase",
                "cbsp",
                "report_first",
                "decompile",
            }
        )
        selected_row_phase_trace_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_SELECTED_ROW_PHASE_TRACE",
                "0",
            )
            .strip()
            .lower()
        )
        selected_row_phase_trace_enabled = (
            pair_record_contact_enabled
            and selected_row_phase_trace_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "trace",
                "selected",
                "selected_row",
                "phase",
                "decompile",
            }
        )
        pair_record_frame_phase_resolve_preview_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_FRAME_PHASE_RESOLVE_PREVIEW",
                "all",
            )
            .strip()
            .lower()
        )
        pair_record_frame_phase_resolve_preview_all = (
            pair_record_frame_phase_resolve_preview_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "all",
                "always",
            }
        )
        pair_record_frame_phase_resolve_preview_accepted = (
            pair_record_frame_phase_resolve_preview_all
            or pair_record_frame_phase_resolve_preview_mode
            in {
                "accepted",
                "accepted_only",
                "contact",
                "contacts",
            }
        )
        pair_record_schedule_response_probe_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_RESPONSE_PROBE",
                "0",
            )
            .strip()
            .lower()
        )
        pair_record_schedule_response_probe_enabled = (
            pair_record_schedule_probe_enabled
            and pair_record_schedule_response_probe_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "probe",
                "response",
                "schedule_response",
                "decompile",
            }
        )
        pair_record_deferred_prestep_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP",
                "0",
            )
            .strip()
            .lower()
        )
        pair_record_deferred_prestep_enabled = (
            pair_record_contact_enabled
            and timing_ready
            and pair_record_deferred_prestep_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "pre",
                "prestep",
                "deferred",
                "decompile",
            }
        )
        pair_record_deferred_prestep_probe_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP_PROBE",
                "0",
            )
            .strip()
            .lower()
        )
        pair_record_deferred_prestep_probe_enabled = (
            pair_record_contact_enabled
            and timing_ready
            and pair_record_deferred_prestep_probe_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "probe",
                "pre",
                "prestep",
                "decompile",
            }
        )
        try:
            pair_record_deferred_prestep_max_distance = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP_MAX_DISTANCE",
                    "3.0",
                )
            )
        except ValueError:
            pair_record_deferred_prestep_max_distance = 3.0
        if not math.isfinite(pair_record_deferred_prestep_max_distance):
            pair_record_deferred_prestep_max_distance = 3.0
        pair_record_phase_lookahead_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD",
                "0",
            )
            .strip()
            .lower()
        )
        pair_record_phase_lookahead_enabled = (
            pair_record_contact_enabled
            and timing_ready
            and pair_record_phase_lookahead_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "probe",
                "apply",
                "queue",
                "queued",
                "schedule",
                "defer",
                "deferred",
                "contact",
                "lookahead",
                "phase",
                "decompile",
            }
        )
        pair_record_phase_lookahead_apply_enabled = (
            pair_record_phase_lookahead_enabled
            and pair_record_phase_lookahead_mode
            in {
                "apply",
                "contact",
                "response",
                "resolve",
                "decompile",
            }
        )
        pair_record_phase_lookahead_queue_enabled = (
            pair_record_phase_lookahead_enabled
            and pair_record_phase_lookahead_mode
            in {"queue", "queued", "schedule", "defer", "deferred"}
        )
        try:
            pair_record_phase_lookahead_max_time = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_TIME",
                    "0.12",
                )
            )
        except ValueError:
            pair_record_phase_lookahead_max_time = 0.12
        if not math.isfinite(pair_record_phase_lookahead_max_time):
            pair_record_phase_lookahead_max_time = 0.12
        pair_record_phase_lookahead_max_time = max(
            0.0,
            min(0.5, pair_record_phase_lookahead_max_time),
        )
        try:
            pair_record_phase_lookahead_steps = int(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_STEPS",
                    "12",
                )
            )
        except ValueError:
            pair_record_phase_lookahead_steps = 12
        pair_record_phase_lookahead_steps = max(
            1,
            min(60, pair_record_phase_lookahead_steps),
        )
        try:
            pair_record_phase_lookahead_max_distance = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_DISTANCE",
                    "3.0",
                )
            )
        except ValueError:
            pair_record_phase_lookahead_max_distance = 3.0
        if not math.isfinite(pair_record_phase_lookahead_max_distance):
            pair_record_phase_lookahead_max_distance = 3.0
        pair_record_phase_lookahead_max_distance = max(
            0.0,
            pair_record_phase_lookahead_max_distance,
        )
        pair_record_phase_lookahead_accel_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_ACCEL",
                "constant_velocity",
            )
            .strip()
            .lower()
        )
        pair_record_phase_backtrack_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK",
                "0",
            )
            .strip()
            .lower()
        )
        pair_record_phase_backtrack_enabled = (
            pair_record_contact_enabled
            and timing_ready
            and pair_record_phase_backtrack_mode
            in {
                "1",
                "true",
                "on",
                "yes",
                "probe",
                "apply",
                "response",
                "resolve",
                "backtrack",
                "phase",
                "decompile",
            }
        )
        pair_record_phase_backtrack_apply_enabled = (
            pair_record_phase_backtrack_enabled
            and pair_record_phase_backtrack_mode
            in {"apply", "response", "resolve", "contact"}
        )
        try:
            pair_record_phase_backtrack_max_time = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_MAX_TIME",
                    "0.18",
                )
            )
        except ValueError:
            pair_record_phase_backtrack_max_time = 0.18
        if not math.isfinite(pair_record_phase_backtrack_max_time):
            pair_record_phase_backtrack_max_time = 0.18
        pair_record_phase_backtrack_max_time = max(
            0.0,
            min(0.5, pair_record_phase_backtrack_max_time),
        )
        try:
            pair_record_phase_backtrack_steps = int(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_STEPS",
                    "12",
                )
            )
        except ValueError:
            pair_record_phase_backtrack_steps = 12
        pair_record_phase_backtrack_steps = max(
            1,
            min(60, pair_record_phase_backtrack_steps),
        )
        try:
            pair_record_phase_backtrack_max_distance = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_MAX_DISTANCE",
                    "4.0",
                )
            )
        except ValueError:
            pair_record_phase_backtrack_max_distance = 4.0
        if not math.isfinite(pair_record_phase_backtrack_max_distance):
            pair_record_phase_backtrack_max_distance = 4.0
        pair_record_phase_backtrack_max_distance = max(
            0.0,
            pair_record_phase_backtrack_max_distance,
        )
        pair_record_phase_backtrack_accel_mode = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_ACCEL",
                "constant_velocity",
            )
            .strip()
            .lower()
        )
        pair_record_phase_backtrack_source = (
            os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_SOURCE",
                "pre",
            )
            .strip()
            .lower()
        )
        raycast_fallback_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_RAYCAST_FALLBACK", "0")
            .strip()
            .lower()
        )
        raycast_fallback_enabled = raycast_fallback_mode in {
            "1",
            "true",
            "on",
            "yes",
            "raycast",
            "capsule",
            "decompile",
        }
        raycast_fallback_timed_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_RAYCAST_TIMED_FALLBACK", "0")
            .strip()
            .lower()
        )
        raycast_fallback_timed_enabled = raycast_fallback_enabled and raycast_fallback_timed_mode in {
            "1",
            "true",
            "on",
            "yes",
            "timed",
            "sweep",
            "bucket",
            "decompile",
        }
        try:
            raycast_fallback_min_penetration = float(
                os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_RAYCAST_MIN_PENETRATION",
                    str(self._PENETRATION_SLOP_DEFAULT),
                )
            )
        except ValueError:
            raycast_fallback_min_penetration = self._PENETRATION_SLOP_DEFAULT
        dirty_dispatch_debug = {}

        def probe_contact_fields(contact, *, center, z_lift_used):
            if contact is None:
                return None
            return {
                "point": getattr(contact, "position", None),
                "normal": getattr(contact, "normal", None),
                "depth": getattr(contact, "penetration", None),
                "contact_sector_index": getattr(contact, "sector_index", None),
                "contact_cell": getattr(contact, "cell", None),
                "contact_normal_source": getattr(contact, "normal_source", None),
                "contact_cbsp_split_normal": getattr(contact, "cbsp_split_normal", None),
                "contact_terrain_face_normal": getattr(contact, "terrain_face_normal", None),
                "contact_mesh_face_normal": getattr(contact, "mesh_face_normal", None),
                "contact_entity_radial_normal": getattr(contact, "entity_radial_normal", None),
                "contact_cbsp_store_normal0": getattr(contact, "cbsp_store_normal0", None),
                "contact_cbsp_store_normal1": getattr(contact, "cbsp_store_normal1", None),
                "contact_cbsp_record_hit_source": getattr(contact, "cbsp_record_hit_source", None),
                "contact_cbsp_mesh_triangle_indices": getattr(contact, "cbsp_mesh_triangle_indices", None),
                "contact_cbsp_guess7_order": getattr(contact, "cbsp_guess7_order", None),
                "contact_cbsp_guess7_terms": getattr(contact, "cbsp_guess7_terms", None),
                "contact_cbsp_edge_hit_kind": getattr(contact, "cbsp_edge_hit_kind", None),
                "contact_cbsp_edge_t": getattr(contact, "cbsp_edge_t", None),
                "contact_cbsp_node_index": getattr(contact, "cbsp_node_index", None),
                "contact_cbsp_node_depth": getattr(contact, "cbsp_node_depth", None),
                "contact_cbsp_node_mesh_normal_angle_deg": getattr(
                    contact,
                    "cbsp_node_mesh_normal_angle_deg",
                    None,
                ),
                "model_center": center,
                "z_lift": z_lift_used,
            }

        def raycast_probe_fields(probe):
            if not isinstance(probe, dict):
                return None
            contact = probe.get("contact")
            out = {
                "enabled": probe.get("enabled"),
                "reject": probe.get("reject"),
                "ray_start": probe.get("ray_start"),
                "ray_end": probe.get("ray_end"),
                "ray_length": probe.get("ray_length"),
                "hit_position": probe.get("hit_position"),
                "hit_distance": probe.get("hit_distance"),
                "contact": probe_contact_fields(
                    contact,
                    center=probe.get("ray_end"),
                    z_lift_used=0.0,
                ),
            }
            return {key: value for key, value in out.items() if value not in ({}, None)}

        def sample_raw_origin_contact_at(pos, *, contact_selection=None):
            if (
                collision_model is None
                or vertices is None
                or cbsp_tree is None
                or origin_mode in {"entity", "origin", "raw"}
                or not finite_values((*pos, heading))
            ):
                return None, None, None
            raw_center = (pos[0], pos[1], pos[2])
            selected_contact_selection = (
                model_contact_selection
                if contact_selection is None
                else str(contact_selection or "first").strip().lower()
            )
            try:
                raw_contact = self._terrain_grid_collision.test_model_collision(
                    raw_center,
                    heading,
                    vertices,
                    cbsp_tree,
                    bounding_radius,
                    rotation_matrix=model_contact_rotation_matrix,
                    contact_selection=selected_contact_selection,
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                return None, None, str(exc)
            raw_bounds_contact = None
            if raw_contact is None:
                bounds_probe = getattr(
                    self._terrain_grid_collision,
                    "test_model_bounds_contact",
                    None,
                )
                if callable(bounds_probe):
                    try:
                        raw_bounds_contact = bounds_probe(
                            raw_center,
                            raw_center,
                            heading,
                            vertices,
                            cbsp_tree,
                            bounding_radius,
                            rotation_matrix=model_contact_rotation_matrix,
                            contact_selection=selected_contact_selection,
                        )
                    except Exception:
                        raw_bounds_contact = None
            if (
                raw_contact is None
                and raw_bounds_contact is None
                and pair_record_bounds_sat_enabled
            ):
                bounds_sat_probe = getattr(
                    self._terrain_grid_collision,
                    "test_box_bounds_contact",
                    None,
                )
                if callable(bounds_sat_probe):
                    try:
                        raw_bounds_contact = bounds_sat_probe(
                            raw_center,
                            raw_center,
                            inertia_half_extents,
                            heading,
                            bounding_radius,
                            rotation_matrix=model_contact_rotation_matrix,
                            contact_selection=selected_contact_selection,
                        )
                    except Exception:
                        raw_bounds_contact = None
            return raw_contact, raw_bounds_contact, None

        def raw_origin_fallback_reject_reason(
            raw_contact,
            *,
            velocity=None,
            enabled=None,
            min_depth=None,
            max_depth=None,
            min_normal_z=None,
            min_speed=None,
        ):
            fallback_enabled = raw_fallback_enabled if enabled is None else bool(enabled)
            if not fallback_enabled:
                return "disabled"
            if raw_contact is None:
                return "no_raw_origin_contact"
            depth_threshold = (
                raw_fallback_min_depth if min_depth is None else float(min_depth)
            )
            max_depth_threshold = (
                raw_fallback_max_depth if max_depth is None else float(max_depth)
            )
            normal_z_threshold = (
                raw_fallback_min_normal_z
                if min_normal_z is None
                else float(min_normal_z)
            )
            speed_threshold = (
                raw_fallback_min_speed if min_speed is None else float(min_speed)
            )
            try:
                depth = float(raw_contact.penetration)
                normal_z = float(raw_contact.normal[2])
                speed_vel = velocity if velocity is not None else (vx, vy, vz)
                speed = math.sqrt(
                    float(speed_vel[0]) * float(speed_vel[0])
                    + float(speed_vel[1]) * float(speed_vel[1])
                    + float(speed_vel[2]) * float(speed_vel[2])
                )
            except (TypeError, ValueError, OverflowError, IndexError):
                return "nonfinite_raw_origin_contact"
            if depth <= self._PENETRATION_SLOP_DEFAULT:
                return "below_slop"
            if depth < depth_threshold:
                return "below_min_depth"
            if depth > max_depth_threshold:
                return "above_max_depth"
            if normal_z < normal_z_threshold:
                return "normal_z_below_min"
            if speed < speed_threshold:
                return "speed_below_min"
            return ""

        def raw_origin_contact_for_fallback(raw_contact, *, normal_source=None):
            if raw_contact is None:
                return None
            selected_normal_source = raw_fallback_normal_source if normal_source is None else normal_source
            def normalized_oriented_normal(value):
                try:
                    normal = (
                        float(value[0]),
                        float(value[1]),
                        float(value[2]),
                    )
                    normal_len = math.sqrt(
                        normal[0] * normal[0]
                        + normal[1] * normal[1]
                        + normal[2] * normal[2]
                    )
                except (TypeError, ValueError, OverflowError, IndexError):
                    return None
                if normal_len <= 1e-8:
                    return None
                normal = (
                    normal[0] / normal_len,
                    normal[1] / normal_len,
                    normal[2] / normal_len,
                )
                if normal[2] < 0.0:
                    normal = (-normal[0], -normal[1], -normal[2])
                return normal

            def contact_with_normal(normal, normal_source):
                return TerrainContact(
                    position=raw_contact.position,
                    normal=normal,
                    penetration=raw_contact.penetration,
                    sector_index=raw_contact.sector_index,
                    cell=raw_contact.cell,
                    normal_source=normal_source,
                    cbsp_split_normal=getattr(raw_contact, "cbsp_split_normal", None),
                    terrain_face_normal=getattr(raw_contact, "terrain_face_normal", None),
                    mesh_face_normal=getattr(raw_contact, "mesh_face_normal", None),
                    entity_radial_normal=getattr(raw_contact, "entity_radial_normal", None),
                )

            contact_face_modes = {
                "face",
                "triangle",
                "terrain_face",
                "contact_face",
                "decompile_face",
                "terrain_triangle_contact",
            }
            entity_radial_modes = {
                "entity_radial",
                "radial",
                "contact_to_body",
                "body_from_contact",
                "decompile_context",
                "decompile_cbsp_context",
            }
            radial_face_blend_modes = {
                "entity_radial_terrain_face_blend",
                "radial_face_blend",
                "decompile_context_face_blend",
            }
            radial_face_forward_up_modes = {
                "entity_radial_terrain_face_forward_up",
                "radial_face_forward_up",
                "decompile_context_face_forward_up",
            }
            sampled_terrain_modes = {"terrain", "sampled_terrain", "heightfield"}
            if selected_normal_source in radial_face_blend_modes | radial_face_forward_up_modes:
                radial_normal = normalized_oriented_normal(
                    getattr(raw_contact, "entity_radial_normal", None)
                )
                face_normal = normalized_oriented_normal(
                    getattr(raw_contact, "terrain_face_normal", None)
                )
                try:
                    radial_weight = float(
                        os.environ.get(
                            "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ENTITY_RADIAL_WEIGHT",
                            "1.0",
                        )
                    )
                except (TypeError, ValueError, OverflowError):
                    radial_weight = 1.0
                try:
                    face_weight = float(
                        os.environ.get(
                            "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_TERRAIN_FACE_WEIGHT",
                            "1.0",
                        )
                    )
                except (TypeError, ValueError, OverflowError):
                    face_weight = 1.0
                if radial_normal is not None and face_normal is not None:
                    blend_normal = normalized_oriented_normal(
                        (
                            radial_normal[0] * radial_weight + face_normal[0] * face_weight,
                            radial_normal[1] * radial_weight + face_normal[1] * face_weight,
                            radial_normal[2] * radial_weight + face_normal[2] * face_weight,
                        )
                    )
                    if (
                        blend_normal is not None
                        and selected_normal_source in radial_face_forward_up_modes
                    ):
                        right = (-math.sin(heading), math.cos(heading), 0.0)
                        right_component = (
                            blend_normal[0] * right[0]
                            + blend_normal[1] * right[1]
                            + blend_normal[2] * right[2]
                        )
                        blend_normal = normalized_oriented_normal(
                            (
                                blend_normal[0] - right[0] * right_component,
                                blend_normal[1] - right[1] * right_component,
                                blend_normal[2] - right[2] * right_component,
                            )
                        )
                    if blend_normal is not None:
                        normal_source = (
                            "entity_radial_terrain_face_forward_up"
                            if selected_normal_source in radial_face_forward_up_modes
                            else "entity_radial_terrain_face_blend"
                        )
                        return contact_with_normal(blend_normal, normal_source)
                return raw_contact
            if selected_normal_source in entity_radial_modes:
                radial_normal = normalized_oriented_normal(
                    getattr(raw_contact, "entity_radial_normal", None)
                )
                if radial_normal is not None:
                    return contact_with_normal(radial_normal, "entity_radial")
                return raw_contact
            if selected_normal_source in contact_face_modes:
                face_normal = normalized_oriented_normal(
                    getattr(raw_contact, "terrain_face_normal", None)
                )
                if face_normal is not None:
                    return contact_with_normal(face_normal, "terrain_triangle_contact_face")
                if selected_normal_source not in sampled_terrain_modes:
                    return raw_contact
            if selected_normal_source not in sampled_terrain_modes:
                return raw_contact
            terrain = getattr(self, "terrain", None)
            if terrain is None:
                return raw_contact
            try:
                _height, terrain_normal = terrain.sample_height_normal(
                    float(raw_contact.position[0]),
                    float(raw_contact.position[1]),
                )
            except (TypeError, ValueError, OverflowError):
                return raw_contact
            if not finite_values(terrain_normal):
                return raw_contact
            return TerrainContact(
                position=raw_contact.position,
                normal=terrain_normal,
                penetration=raw_contact.penetration,
                sector_index=raw_contact.sector_index,
                cell=raw_contact.cell,
                normal_source="terrain_triangle",
                cbsp_split_normal=getattr(raw_contact, "cbsp_split_normal", None),
                terrain_face_normal=getattr(raw_contact, "terrain_face_normal", None),
                mesh_face_normal=getattr(raw_contact, "mesh_face_normal", None),
                entity_radial_normal=getattr(raw_contact, "entity_radial_normal", None),
            )

        def tank_raw_origin_fallback_enabled_for(raw_contact) -> bool:
            if not tank_raw_fallback_auto_enabled or raw_contact is None:
                return False
            face_normal = getattr(raw_contact, "terrain_face_normal", None)
            try:
                return float(face_normal[2]) >= tank_raw_fallback_min_face_normal_z
            except (TypeError, ValueError, OverflowError, IndexError):
                return False

        def raw_origin_fallback_probe_for(raw_contact, *, velocity=None):
            tank_auto_raw = tank_raw_origin_fallback_enabled_for(raw_contact)
            fallback_contact = raw_origin_contact_for_fallback(
                raw_contact,
                normal_source=(
                    tank_raw_fallback_normal_source
                    if tank_auto_raw and not raw_fallback_enabled
                    else None
                ),
            )
            reject = raw_origin_fallback_reject_reason(
                fallback_contact,
                velocity=velocity,
                enabled=(raw_fallback_enabled or tank_auto_raw),
                min_depth=(
                    tank_raw_fallback_min_depth
                    if tank_auto_raw and not raw_fallback_enabled
                    else None
                ),
                max_depth=(
                    tank_raw_fallback_max_depth
                    if tank_auto_raw and not raw_fallback_enabled
                    else None
                ),
                min_normal_z=(
                    tank_raw_fallback_min_normal_z
                    if tank_auto_raw and not raw_fallback_enabled
                    else None
                ),
                min_speed=(
                    tank_raw_fallback_min_speed
                    if tank_auto_raw and not raw_fallback_enabled
                    else None
                ),
            )
            return fallback_contact, reject, tank_auto_raw

        def tank_clean_face_fallback_contact_for(contact):
            if not tank_clean_face_fallback_enabled:
                return None, "disabled"
            if contact is None:
                return None, "no_contact"
            try:
                depth = float(contact.penetration)
                normal_z = float(contact.normal[2])
                face_normal = getattr(contact, "terrain_face_normal", None)
                face_normal_z = float(face_normal[2])
            except (TypeError, ValueError, OverflowError, IndexError):
                return None, "nonfinite_contact"
            if depth <= self._PENETRATION_SLOP_DEFAULT:
                return None, "below_slop"
            if (
                tank_clean_pair_solver_max_depth > 0.0
                and depth > tank_clean_pair_solver_max_depth
            ):
                return None, "above_max_depth"
            if normal_z > tank_clean_face_fallback_max_contact_normal_z:
                return None, "contact_normal_z_above_max"
            if face_normal_z < tank_clean_face_fallback_min_face_normal_z:
                return None, "terrain_face_normal_z_below_min"
            face_contact = raw_origin_contact_for_fallback(
                contact,
                normal_source=tank_raw_fallback_normal_source,
            )
            if face_contact is None:
                return None, "missing_face_contact"
            try:
                face_contact_normal_z = float(face_contact.normal[2])
            except (TypeError, ValueError, OverflowError, IndexError):
                return None, "nonfinite_face_contact"
            if face_contact_normal_z < tank_raw_fallback_min_normal_z:
                return None, "face_contact_normal_z_below_min"
            return face_contact, ""

        def tank_face_fallback_delta_normal_for(contact, mode):
            normal = getattr(contact, "normal", None)
            source = getattr(contact, "normal_source", None)
            if mode not in {"horizontal", "horizontal_face", "xy", "flat"}:
                return normal, source
            try:
                nx = float(normal[0])
                ny = float(normal[1])
            except (TypeError, ValueError, OverflowError, IndexError):
                return normal, source
            mag_xy = math.sqrt(nx * nx + ny * ny)
            if not math.isfinite(mag_xy) or mag_xy <= 1e-9:
                return normal, source
            return (
                (nx / mag_xy, ny / mag_xy, 0.0),
                f"{source}_horizontal" if source else "horizontal_face",
            )

        def tank_clean_face_fallback_delta_normal_for(contact):
            return tank_face_fallback_delta_normal_for(
                contact,
                tank_clean_face_fallback_delta_normal_mode,
            )

        def tank_raw_fallback_delta_normal_for(contact):
            return tank_face_fallback_delta_normal_for(
                contact,
                tank_raw_fallback_delta_normal_mode,
            )

        def tank_face_fallback_latch_key(contact):
            if not tank_face_fallback_latch_enabled or contact is None:
                return None
            try:
                cell = tuple(int(round(float(value))) for value in contact.cell)
            except (TypeError, ValueError, OverflowError):
                cell = None
            try:
                normal = tuple(round(float(value), 2) for value in contact.normal[:3])
            except (TypeError, ValueError, OverflowError, IndexError):
                return None
            return (cell, normal)

        def tank_face_fallback_latch_matches(contact) -> bool:
            key = tank_face_fallback_latch_key(contact)
            if key is None:
                return False
            latch = getattr(ctx, "tank_face_fallback_latch", None)
            return isinstance(latch, dict) and latch.get("key") == key

        def tank_face_fallback_latch_record(contact, *, source: str) -> None:
            key = tank_face_fallback_latch_key(contact)
            if key is None:
                return
            try:
                depth = float(contact.penetration)
            except (TypeError, ValueError, OverflowError):
                depth = None
            ctx.tank_face_fallback_latch = {
                "key": key,
                "source": source,
                "depth": depth,
                "cell": getattr(contact, "cell", None),
                "normal": contact.normal,
            }

        def tank_face_fallback_latch_clear(reason: str) -> None:
            if getattr(ctx, "tank_face_fallback_latch", None) is not None:
                ctx.tank_face_fallback_latch = {
                    "cleared": True,
                    "reason": reason,
                }

        def tank_face_fallback_latched_debug(contact, *, kind: str) -> dict:
            return {
                "kind": kind,
                "point": contact.position,
                "normal": contact.normal,
                "depth": contact.penetration,
                "terrain_collision_shape": terrain_collision_shape,
                **contact_debug_fields(contact),
                "tank_face_fallback_latched": True,
                "tank_face_fallback_latch": getattr(
                    ctx,
                    "tank_face_fallback_latch",
                    None,
                ),
                "velocity_before": (vx, vy, vz),
                "velocity_after": (vx, vy, vz),
                "angular_velocity_before": tuple(contact_angular_velocity),
                "angular_velocity_after": tuple(contact_angular_velocity),
                "response": "terrain_face_fallback_latched",
            }

        def sample_raw_origin_fallback_contact_at(pos, *, velocity=None):
            raw_contact, raw_bounds_contact, raw_error = sample_raw_origin_contact_at(pos)
            fallback_contact, reject, tank_auto_raw = raw_origin_fallback_probe_for(
                raw_contact,
                velocity=velocity,
            )
            return {
                "contact": fallback_contact,
                "raw_contact": fallback_contact,
                "raw_bounds_contact": raw_bounds_contact,
                "raw_error": raw_error,
                "reject": reject,
                "tank_raw_origin_fallback": tank_auto_raw,
            }

        def pair_record_contact_reject_reason(contact, *, velocity=None):
            if not pair_record_contact_enabled:
                return "disabled"
            if raw_fallback_enabled:
                return "raw_origin_fallback_enabled"
            if terrain_collision_shape != "model" or collision_model is None:
                return "non_model_collision_shape"
            if contact is None:
                return "no_raw_origin_contact"
            face_normal = getattr(contact, "terrain_face_normal", None)
            try:
                face_normal_z = float(face_normal[2])
            except (TypeError, ValueError, OverflowError, IndexError):
                return "missing_terrain_face_normal"
            if face_normal_z < pair_record_contact_min_face_normal_z:
                return "terrain_face_normal_z_below_min"
            return raw_origin_fallback_reject_reason(
                contact,
                velocity=velocity,
                enabled=True,
                min_depth=pair_record_contact_min_depth,
                max_depth=pair_record_contact_max_depth,
                min_normal_z=pair_record_contact_min_normal_z,
                min_speed=0.0,
            )

        def sample_pair_record_contact_at(pos, *, velocity=None, contact_selection=None):
            raw_contact, raw_bounds_contact, raw_error = sample_raw_origin_contact_at(pos)
            pair_raw_contact = raw_contact
            pair_raw_bounds_contact = raw_bounds_contact
            pair_raw_error = raw_error
            selected_pair_record_contact_selection = (
                pair_record_contact_selection
                if contact_selection is None
                else str(contact_selection or "").strip().lower()
            )
            if (
                raw_error is None
                and selected_pair_record_contact_selection
                and selected_pair_record_contact_selection != model_contact_selection
            ):
                pair_raw_contact, pair_raw_bounds_contact, pair_raw_error = (
                    sample_raw_origin_contact_at(
                        pos,
                        contact_selection=selected_pair_record_contact_selection,
                    )
                )
                if pair_raw_error is not None:
                    pair_raw_contact = raw_contact
                    pair_raw_bounds_contact = raw_bounds_contact
                    pair_raw_error = raw_error
            pair_contact_source = pair_raw_contact
            if pair_contact_source is None and pair_record_bounds_sat_apply_enabled:
                pair_contact_source = pair_raw_bounds_contact
            pair_contact = raw_origin_contact_for_fallback(
                pair_contact_source,
                normal_source=pair_record_contact_normal_source,
            )
            pair_delta_contact = raw_origin_contact_for_fallback(
                pair_contact_source,
                normal_source=pair_record_contact_delta_normal_source,
            )
            reject = pair_record_contact_reject_reason(
                pair_contact,
                velocity=velocity,
            )
            return {
                "contact": pair_contact,
                "delta_contact": pair_delta_contact,
                "raw_contact": raw_contact,
                "raw_bounds_contact": raw_bounds_contact,
                "raw_error": raw_error,
                "selected_raw_contact": pair_raw_contact,
                "selected_raw_bounds_contact": pair_raw_bounds_contact,
                "selected_pair_contact_source": (
                    getattr(pair_contact_source, "normal_source", None)
                    if pair_contact_source is not None
                    else None
                ),
                "selected_raw_error": pair_raw_error,
                "contact_selection": selected_pair_record_contact_selection,
                "reject": reject,
            }

        def record_dirty_reference_pair_probe():
            if not dirty_reference_pair_probe_enabled:
                return None
            dirty_dispatch_debug["dirty_reference_pair_probe_enabled"] = True
            probe = {
                "enabled": True,
                "dirty_bounds_active": bool(
                    dirty_dispatch_debug.get("dirty_bounds_active")
                ),
                "dirty_bounds_xy_overlap": dirty_dispatch_debug.get(
                    "dirty_bounds_xy_overlap"
                ),
            }
            dirty_dispatch_debug["dirty_reference_pair_probe"] = probe
            if not pair_record_contact_enabled:
                probe["reject"] = "pair_record_contact_disabled"
                return None
            if not dirty_dispatch_debug.get("dirty_bounds_active"):
                probe["reject"] = "dirty_bounds_inactive"
                return None
            if dirty_dispatch_debug.get("dirty_bounds_xy_overlap") is not True:
                probe["reject"] = "dirty_bounds_no_xy_overlap"
                return None

            current_pos = finite_triplet((anchor[0], anchor[1], anchor[2]))
            reference = finite_triplet(reference_pos)
            candidates = []
            seen = set()

            def add_candidate(label, pos):
                candidate = finite_triplet(pos)
                if candidate is None:
                    return
                key = (
                    round(candidate[0], 4),
                    round(candidate[1], 4),
                    round(candidate[2], 4),
                )
                if key in seen:
                    return
                seen.add(key)
                candidates.append((label, candidate))

            add_candidate("dirty_reference_pos", reference)
            if reference is not None and current_pos is not None:
                add_candidate(
                    "dirty_midpoint_pos",
                    (
                        reference[0] + (current_pos[0] - reference[0]) * 0.5,
                        reference[1] + (current_pos[1] - reference[1]) * 0.5,
                        reference[2] + (current_pos[2] - reference[2]) * 0.5,
                    ),
                )
            add_candidate("dirty_current_pos", current_pos)

            results = {}
            accept_labels = []
            accepted_pairs = []
            for label, candidate in candidates:
                pair_probe = sample_pair_record_contact_at(
                    candidate,
                    velocity=(vx, vy, vz),
                )
                pair_contact = pair_probe.get("contact")
                pair_delta_contact = pair_probe.get("delta_contact")
                reject = pair_probe.get("reject")
                accepted = pair_contact is not None and reject == ""
                if accepted:
                    accept_labels.append(label)
                    accepted_pairs.append({
                        "label": label,
                        "pos": candidate,
                        "velocity": (vx, vy, vz),
                        "probe": pair_probe,
                        "contact": pair_contact,
                        "delta_contact": pair_delta_contact,
                    })
                item = {
                    "pos": candidate,
                    "reject": reject,
                    "selected_raw_error": pair_probe.get("selected_raw_error"),
                    "contact": probe_contact_fields(
                        pair_contact,
                        center=candidate,
                        z_lift_used=0.0,
                    ),
                    "delta_contact": probe_contact_fields(
                        pair_delta_contact,
                        center=candidate,
                        z_lift_used=0.0,
                    ),
                    "accepted": accepted,
                }
                results[label] = {
                    key: value
                    for key, value in item.items()
                    if value not in ({}, None)
                }
            probe.update({
                "reject": "" if accept_labels else "no_pair_record_contact",
                "accept_labels": accept_labels,
                "selected_label": accepted_pairs[0]["label"] if accepted_pairs else None,
                "results": results,
            })
            return accepted_pairs[0] if accepted_pairs else None

        def record_pair_record_contact_cache(
            contact,
            *,
            delta_contact=None,
            pos=None,
            velocity=None,
            source: str,
        ) -> None:
            if not pair_record_cached_contact_enabled or contact is None:
                return
            ctx.terrain_pair_record_contact_cache = {
                "contact": contact,
                "delta_contact": delta_contact,
                "pos": finite_triplet(pos) or (anchor[0], anchor[1], anchor[2]),
                "velocity": finite_triplet(velocity) or (vx, vy, vz),
                "reference_pos": finite_triplet(reference_pos),
                "step": pair_record_cache_step,
                "source": source,
            }

        def cached_pair_record_contact_probe(pos, *, velocity=None):
            if not pair_record_cached_contact_enabled:
                return {
                    "contact": None,
                    "delta_contact": None,
                    "reject": "disabled",
                    "cache": None,
                }
            cache = getattr(ctx, "terrain_pair_record_contact_cache", None)
            if not isinstance(cache, dict):
                return {
                    "contact": None,
                    "delta_contact": None,
                    "reject": "no_cached_pair_record_contact",
                    "cache": None,
                }
            contact = cache.get("contact")
            if contact is None:
                return {
                    "contact": None,
                    "delta_contact": None,
                    "reject": "no_cached_pair_record_contact",
                    "cache": cache,
                }
            try:
                cache_step = int(cache.get("step", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                cache_step = 0
            age_steps = max(0, pair_record_cache_step - cache_step)
            if age_steps > pair_record_cached_contact_max_age_steps:
                return {
                    "contact": None,
                    "delta_contact": None,
                    "reject": "cached_pair_record_contact_too_old",
                    "age_steps": age_steps,
                    "cache": cache,
                }
            current = finite_triplet(pos)
            cached_pos = finite_triplet(cache.get("pos"))
            if current is None or cached_pos is None:
                return {
                    "contact": None,
                    "delta_contact": None,
                    "reject": "cached_pair_record_contact_nonfinite_pos",
                    "age_steps": age_steps,
                    "cache": cache,
                }
            distance = math.sqrt(
                (current[0] - cached_pos[0]) * (current[0] - cached_pos[0])
                + (current[1] - cached_pos[1]) * (current[1] - cached_pos[1])
                + (current[2] - cached_pos[2]) * (current[2] - cached_pos[2])
            )
            if distance > pair_record_cached_contact_max_distance:
                return {
                    "contact": None,
                    "delta_contact": None,
                    "reject": "cached_pair_record_contact_too_far",
                    "age_steps": age_steps,
                    "distance": distance,
                    "cache": cache,
                }
            cached_ref = finite_triplet(cache.get("reference_pos"))
            current_ref = finite_triplet(reference_pos)
            ref_distance = None
            if cached_ref is not None and current_ref is not None:
                ref_distance = math.sqrt(
                    (current_ref[0] - cached_ref[0])
                    * (current_ref[0] - cached_ref[0])
                    + (current_ref[1] - cached_ref[1])
                    * (current_ref[1] - cached_ref[1])
                    + (current_ref[2] - cached_ref[2])
                    * (current_ref[2] - cached_ref[2])
                )
                if ref_distance > pair_record_cached_contact_max_ref_distance:
                    return {
                        "contact": None,
                        "delta_contact": None,
                        "reject": "cached_pair_record_reference_too_far",
                        "age_steps": age_steps,
                        "distance": distance,
                        "reference_distance": ref_distance,
                        "cache": cache,
                    }
            reject = pair_record_contact_reject_reason(contact, velocity=velocity)
            if reject:
                return {
                    "contact": None,
                    "delta_contact": cache.get("delta_contact"),
                    "reject": f"cached_pair_record_{reject}",
                    "age_steps": age_steps,
                    "distance": distance,
                    "reference_distance": ref_distance,
                    "cache": cache,
                }
            if contact.penetration <= self._PENETRATION_SLOP_DEFAULT:
                return {
                    "contact": None,
                    "delta_contact": cache.get("delta_contact"),
                    "reject": "cached_pair_record_below_slop",
                    "age_steps": age_steps,
                    "distance": distance,
                    "reference_distance": ref_distance,
                    "cache": cache,
                }
            return {
                "contact": contact,
                "delta_contact": cache.get("delta_contact"),
                "reject": "",
                "age_steps": age_steps,
                "distance": distance,
                "reference_distance": ref_distance,
                "source": cache.get("source"),
                "cache": cache,
            }

        def sample_raycast_fallback_contact_at(pos, *, reference=None, velocity=None):
            raycast_fn_local = getattr(self._terrain_grid_collision, "raycast", None)
            reference_candidate = (
                finite_triplet(reference)
                or finite_triplet(reference_pos)
                or finite_triplet(pre_pos)
                or finite_triplet(getattr(ctx, "world_collision_ref_pos", None))
            )
            position = finite_triplet(pos)
            if not raycast_fallback_enabled:
                return {
                    "enabled": False,
                    "reject": "disabled",
                    "contact": None,
                    "ray_start": reference_candidate,
                    "ray_end": position,
                }
            if not callable(raycast_fn_local):
                return {
                    "enabled": True,
                    "reject": "raycast_unavailable",
                    "contact": None,
                    "ray_start": reference_candidate,
                    "ray_end": position,
                }
            if reference_candidate is None or position is None:
                return {
                    "enabled": True,
                    "reject": "nonfinite_raycast_input",
                    "contact": None,
                    "ray_start": reference_candidate,
                    "ray_end": position,
                }
            ray_dir = (
                position[0] - reference_candidate[0],
                position[1] - reference_candidate[1],
                position[2] - reference_candidate[2],
            )
            ray_len = math.sqrt(
                ray_dir[0] * ray_dir[0] +
                ray_dir[1] * ray_dir[1] +
                ray_dir[2] * ray_dir[2]
            )
            try:
                terrain_hit = raycast_fn_local(reference_candidate, position)
            except Exception as exc:  # pragma: no cover - diagnostic only
                return {
                    "enabled": True,
                    "reject": "raycast_error",
                    "error": str(exc),
                    "contact": None,
                    "ray_start": reference_candidate,
                    "ray_end": position,
                    "ray_length": ray_len,
                }
            if terrain_hit is None:
                return {
                    "enabled": True,
                    "reject": "no_terrain_raycast_hit",
                    "contact": None,
                    "ray_start": reference_candidate,
                    "ray_end": position,
                    "ray_length": ray_len,
                }
            if ray_len <= 0.001:
                contact_point = terrain_hit.position
                penetration = max(float(bounding_radius), self._PENETRATION_SLOP_DEFAULT * 2.0)
            else:
                ray_scale = float(bounding_radius) / max(ray_len, 1e-9)
                scaled_dir = (
                    ray_dir[0] * ray_scale,
                    ray_dir[1] * ray_scale,
                    ray_dir[2] * ray_scale,
                )
                contact_point = (
                    position[0] + scaled_dir[0],
                    position[1] + scaled_dir[1],
                    position[2] + scaled_dir[2],
                )
                hit_distance = float(getattr(terrain_hit, "distance", ray_len) or ray_len)
                penetration = max(
                    self._PENETRATION_SLOP_DEFAULT * 2.0,
                    float(bounding_radius) - hit_distance,
                )
            normal = (
                float(terrain_hit.normal[0]),
                float(terrain_hit.normal[1]),
                float(terrain_hit.normal[2]),
            )
            if not finite_values((*contact_point, *normal, penetration)):
                return {
                    "enabled": True,
                    "reject": "nonfinite_raycast_contact",
                    "contact": None,
                    "ray_start": reference_candidate,
                    "ray_end": position,
                    "ray_length": ray_len,
                    "hit_position": getattr(terrain_hit, "position", None),
                    "hit_distance": getattr(terrain_hit, "distance", None),
                }
            contact = TerrainContact(
                position=contact_point,
                normal=normal,
                penetration=penetration,
                sector_index=terrain_hit.sector_index,
                cell=terrain_hit.cell,
                normal_source="terrain_capsule_raycast",
            )
            reject = ""
            if penetration <= max(self._PENETRATION_SLOP_DEFAULT, raycast_fallback_min_penetration):
                reject = "below_min_penetration"
            speed_vel = velocity if velocity is not None else (vx, vy, vz)
            try:
                speed = math.sqrt(
                    float(speed_vel[0]) * float(speed_vel[0])
                    + float(speed_vel[1]) * float(speed_vel[1])
                    + float(speed_vel[2]) * float(speed_vel[2])
                )
            except (TypeError, ValueError, OverflowError, IndexError):
                speed = math.inf
            if reject == "" and speed < raw_fallback_min_speed:
                reject = "speed_below_min"
            return {
                "enabled": True,
                "reject": reject,
                "contact": contact,
                "ray_start": reference_candidate,
                "ray_end": position,
                "ray_length": ray_len,
                "hit_position": terrain_hit.position,
                "hit_distance": getattr(terrain_hit, "distance", None),
            }

        def reference_pose_candidates(pos):
            current = finite_triplet(pos)
            if current is None:
                return []
            candidates = []
            seen_positions: set[tuple[float, float, float]] = set()

            def add_candidate(label, value):
                candidate = finite_triplet(value)
                if candidate is None:
                    return
                rounded = (
                    round(candidate[0], 4),
                    round(candidate[1], 4),
                    round(candidate[2], 4),
                )
                if rounded in seen_positions:
                    return
                seen_positions.add(rounded)
                candidates.append((label, candidate))

            def add_midpoint(label, start, end, fraction):
                start_pos = finite_triplet(start)
                end_pos = finite_triplet(end)
                if start_pos is None or end_pos is None:
                    return
                add_candidate(
                    label,
                    (
                        start_pos[0] + (end_pos[0] - start_pos[0]) * fraction,
                        start_pos[1] + (end_pos[1] - start_pos[1]) * fraction,
                        start_pos[2] + (end_pos[2] - start_pos[2]) * fraction,
                    ),
                )

            pre_step_pos = finite_triplet(pre_pos)
            dirty_reference_pos = finite_triplet(reference_pos)
            world_reference_pos = finite_triplet(
                getattr(ctx, "world_collision_ref_pos", None)
            )
            add_candidate("current", current)
            add_candidate("pre_pos", pre_step_pos)
            add_candidate("dirty_reference_pos", dirty_reference_pos)
            add_candidate("world_collision_ref_pos", world_reference_pos)
            add_midpoint("pre_to_current_25", pre_step_pos, current, 0.25)
            add_midpoint("pre_to_current_50", pre_step_pos, current, 0.50)
            add_midpoint("pre_to_current_75", pre_step_pos, current, 0.75)
            add_midpoint("dirty_reference_to_current_50", dirty_reference_pos, current, 0.50)
            return candidates

        def reference_pose_candidate_velocity(label, candidate, *, fallback=None):
            current_vel = finite_triplet((vx, vy, vz))
            fallback_vel = finite_triplet(fallback) or current_vel
            start_vel = finite_triplet(pre_vel)
            if current_vel is None:
                return None, None, "nonfinite_current_velocity"
            if start_vel is None:
                return fallback_vel, None, "fallback_no_prestep_velocity"

            def lerp_velocity(fraction, source):
                fraction = max(0.0, min(1.0, float(fraction)))
                return (
                    start_vel[0] + (current_vel[0] - start_vel[0]) * fraction,
                    start_vel[1] + (current_vel[1] - start_vel[1]) * fraction,
                    start_vel[2] + (current_vel[2] - start_vel[2]) * fraction,
                ), fraction, source

            if label == "pre_pos":
                return start_vel, 0.0, "pre_step_velocity"
            if label == "current":
                return current_vel, 1.0, "current_velocity"
            if label.startswith("pre_to_current_"):
                try:
                    fraction = float(label.rsplit("_", 1)[-1])
                    if fraction > 1.0:
                        fraction /= 100.0
                    return lerp_velocity(fraction, "pre_to_current_fraction")
                except ValueError:
                    pass
            if label == "dirty_reference_to_current_50":
                return lerp_velocity(0.5, "dirty_reference_to_current_fraction")

            start_pos = finite_triplet(pre_pos)
            end_pos = finite_triplet((anchor[0], anchor[1], anchor[2]))
            candidate_pos = finite_triplet(candidate)
            if start_pos is not None and end_pos is not None and candidate_pos is not None:
                step_delta = (
                    end_pos[0] - start_pos[0],
                    end_pos[1] - start_pos[1],
                    end_pos[2] - start_pos[2],
                )
                step_len_sq = (
                    step_delta[0] * step_delta[0]
                    + step_delta[1] * step_delta[1]
                    + step_delta[2] * step_delta[2]
                )
                if step_len_sq > 1e-8:
                    raw_fraction = (
                        (candidate_pos[0] - start_pos[0]) * step_delta[0]
                        + (candidate_pos[1] - start_pos[1]) * step_delta[1]
                        + (candidate_pos[2] - start_pos[2]) * step_delta[2]
                    ) / step_len_sq
                    return lerp_velocity(
                        raw_fraction,
                        (
                            "projected_step_fraction"
                            if 0.0 <= raw_fraction <= 1.0
                            else "projected_step_fraction_clamped"
                        ),
                    )
            return fallback_vel, None, "fallback_velocity"

        def sample_reference_pose_contacts(pos, *, velocity=None):
            """Record contacts at decompile-style collision/reference poses."""

            output = {}
            for label, candidate in reference_pose_candidates(pos):
                candidate_velocity, velocity_fraction, velocity_source = (
                    reference_pose_candidate_velocity(
                        label,
                        candidate,
                        fallback=velocity,
                    )
                )
                lifted_contact = None
                lifted_error = None
                try:
                    lifted_contact = sample_contact_at(candidate)
                except Exception as exc:  # pragma: no cover - diagnostic only
                    lifted_error = str(exc)
                raw_contact, raw_bounds_contact, raw_error = sample_raw_origin_contact_at(
                    candidate
                )
                raw_fallback_contact, raw_reject, raw_tank_auto = raw_origin_fallback_probe_for(
                    raw_contact,
                    velocity=candidate_velocity,
                )
                pair_probe = sample_pair_record_contact_at(
                    candidate,
                    velocity=candidate_velocity,
                )
                pair_contact = pair_probe.get("contact")
                pair_delta_contact = pair_probe.get("delta_contact")
                pair_reject = pair_probe.get("reject")
                raw_center = (candidate[0], candidate[1], candidate[2])
                lifted_center = (candidate[0], candidate[1], candidate[2] + z_lift)
                contact_any = (
                    lifted_contact is not None
                    or raw_fallback_contact is not None
                    or raw_bounds_contact is not None
                    or pair_contact is not None
                )
                if (
                    not contact_any
                    and lifted_error is None
                    and raw_error is None
                    and pair_probe.get("selected_raw_error") is None
                ):
                    continue
                item = {
                    "pos": candidate,
                    "velocity": candidate_velocity,
                    "velocity_fraction": velocity_fraction,
                    "velocity_source": velocity_source,
                    "lifted_contact": probe_contact_fields(
                        lifted_contact,
                        center=lifted_center,
                        z_lift_used=z_lift,
                    ),
                    "lifted_error": lifted_error,
                    "raw_origin_contact": probe_contact_fields(
                        raw_fallback_contact,
                        center=raw_center,
                        z_lift_used=0.0,
                    ),
                    "raw_origin_bounds_contact": probe_contact_fields(
                        raw_bounds_contact,
                        center=raw_center,
                        z_lift_used=0.0,
                    ),
                    "raw_error": raw_error,
                    "raw_origin_fallback_reject": raw_reject,
                    "tank_raw_origin_fallback": raw_tank_auto,
                    "pair_record_contact": probe_contact_fields(
                        pair_contact,
                        center=raw_center,
                        z_lift_used=0.0,
                    ),
                    "pair_record_delta_contact": probe_contact_fields(
                        pair_delta_contact,
                        center=raw_center,
                        z_lift_used=0.0,
                    ),
                    "pair_record_contact_reject": pair_reject,
                    "pair_record_selected_raw_error": pair_probe.get(
                        "selected_raw_error"
                    ),
                    "lifted_contact_any": lifted_contact is not None,
                    "raw_contact_any": (
                        raw_fallback_contact is not None
                        or raw_bounds_contact is not None
                    ),
                    "pair_record_contact_any": pair_contact is not None,
                    "pair_record_contact_accept": (
                        pair_contact is not None and pair_reject == ""
                    ),
                    "contact_any": contact_any,
                }
                output[label] = {
                    key: value
                    for key, value in item.items()
                    if value not in ({}, None)
                }
            return output

        def sample_frame_phase_report_first_probe(pos, *, velocity=None):
            """Read-only report-first CBSP probe across the server frame phase."""

            if not pair_record_frame_phase_probe_enabled:
                return None
            frame_dt = max(0.0, float(dt or 0.0))
            bucket_count = 30

            def bucket_for_fraction(fraction):
                if fraction is None:
                    return None, None, None
                try:
                    clamped = max(0.0, min(1.0, float(fraction)))
                except (TypeError, ValueError, OverflowError):
                    return None, None, None
                if frame_dt <= 0.0:
                    return None, None, None
                time_s = frame_dt * clamped
                bucket_index = int((time_s / frame_dt) * float(bucket_count))
                bucket_index = max(0, min(bucket_count - 1, bucket_index))
                bucket_width = frame_dt / float(bucket_count)
                return (
                    time_s,
                    bucket_index,
                    (bucket_width * float(bucket_index), bucket_width * float(bucket_index + 1)),
                )

            selections = (
                "cbsp_mesh_edge_terrain_plane_probe",
                "cbsp_mesh_edge_terrain_plane_traversal_probe",
            )
            frame_candidates = []
            for label, candidate in reference_pose_candidates(pos):
                candidate_velocity, velocity_fraction, velocity_source = (
                    reference_pose_candidate_velocity(
                        label,
                        candidate,
                        fallback=velocity,
                    )
                )
                frame_candidates.append(
                    (
                        label,
                        candidate,
                        candidate_velocity,
                        velocity_fraction,
                        velocity_source,
                    )
                )

            pre_step_frame_pos = finite_triplet(pre_pos)
            current_frame_pos = finite_triplet(pos)
            pre_step_frame_vel = finite_triplet(pre_vel)
            current_frame_vel = finite_triplet((vx, vy, vz))

            def triplet_delta(start, end):
                start_triplet = finite_triplet(start)
                end_triplet = finite_triplet(end)
                if start_triplet is None or end_triplet is None:
                    return None
                return (
                    end_triplet[0] - start_triplet[0],
                    end_triplet[1] - start_triplet[1],
                    end_triplet[2] - start_triplet[2],
                )

            def triplet_span(delta):
                delta_triplet = finite_triplet(delta)
                if delta_triplet is None:
                    return None
                return math.sqrt(
                    delta_triplet[0] * delta_triplet[0]
                    + delta_triplet[1] * delta_triplet[1]
                    + delta_triplet[2] * delta_triplet[2]
                )

            frame_pose_delta = triplet_delta(pre_step_frame_pos, current_frame_pos)
            frame_pose_span_u = triplet_span(frame_pose_delta)
            frame_velocity_delta = triplet_delta(pre_step_frame_vel, current_frame_vel)
            frame_velocity_span_u = triplet_span(frame_velocity_delta)
            frame_pose_velocity_integrated_end = None
            if (
                frame_dt > 0.0
                and pre_step_frame_pos is not None
                and pre_step_frame_vel is not None
                and current_frame_vel is not None
            ):
                frame_pose_velocity_integrated_end = (
                    pre_step_frame_pos[0]
                    + 0.5
                    * (pre_step_frame_vel[0] + current_frame_vel[0])
                    * frame_dt,
                    pre_step_frame_pos[1]
                    + 0.5
                    * (pre_step_frame_vel[1] + current_frame_vel[1])
                    * frame_dt,
                    pre_step_frame_pos[2]
                    + 0.5
                    * (pre_step_frame_vel[2] + current_frame_vel[2])
                    * frame_dt,
                )
            frame_pose_integrated_delta_from_source_end = triplet_delta(
                frame_pose_velocity_integrated_end,
                current_frame_pos,
            )
            frame_pose_integrated_error_u = triplet_span(
                frame_pose_integrated_delta_from_source_end
            )
            if frame_pose_span_u is None:
                frame_pose_span_verdict = "frame_pose_unavailable"
            elif bucket_count > 1 and frame_pose_span_u <= 0.001:
                frame_pose_span_verdict = "frame_pose_static"
            else:
                frame_pose_span_verdict = "frame_pose_varies"
            if frame_pose_integrated_error_u is None:
                frame_pose_motion_consistency_verdict = (
                    "frame_pose_motion_consistency_unavailable"
                )
            elif frame_pose_integrated_error_u <= 0.1:
                frame_pose_motion_consistency_verdict = (
                    "frame_pose_matches_velocity_integration"
                )
            else:
                frame_pose_motion_consistency_verdict = (
                    "frame_pose_differs_from_velocity_integration"
                )

            def integrate_frame_phase_state(start_pos, start_vel, elapsed_s):
                start_position = finite_triplet(start_pos)
                start_velocity = finite_triplet(start_vel)
                if start_position is None or start_velocity is None:
                    return None, None
                if (
                    frame_dt > 0.0
                    and pre_step_frame_vel is not None
                    and current_frame_vel is not None
                ):
                    acc = (
                        (current_frame_vel[0] - pre_step_frame_vel[0]) / frame_dt,
                        (current_frame_vel[1] - pre_step_frame_vel[1]) / frame_dt,
                        (current_frame_vel[2] - pre_step_frame_vel[2]) / frame_dt,
                    )
                else:
                    acc = (0.0, 0.0, 0.0)
                elapsed = max(0.0, float(elapsed_s or 0.0))
                return (
                    (
                        start_position[0]
                        + start_velocity[0] * elapsed
                        + 0.5 * acc[0] * elapsed * elapsed,
                        start_position[1]
                        + start_velocity[1] * elapsed
                        + 0.5 * acc[1] * elapsed * elapsed,
                        start_position[2]
                        + start_velocity[2] * elapsed
                        + 0.5 * acc[2] * elapsed * elapsed,
                    ),
                    (
                        start_velocity[0] + acc[0] * elapsed,
                        start_velocity[1] + acc[1] * elapsed,
                        start_velocity[2] + acc[2] * elapsed,
                    ),
                )

            def preview_frame_phase_resolve_model(
                *,
                label,
                selection,
                candidate,
                candidate_velocity,
                time_s,
                bucket_index,
                pair_contact,
                response_preview,
            ):
                collision_time = None
                if time_s is not None:
                    try:
                        collision_time = max(0.0, min(frame_dt, float(time_s)))
                    except (TypeError, ValueError, OverflowError):
                        collision_time = None
                remaining_time = (
                    max(0.0, frame_dt - collision_time)
                    if collision_time is not None
                    else None
                )
                retest_probe = None
                retest_contact = None
                retest_reject = "nonfinite_retest_pose"
                if current_frame_pos is not None:
                    retest_probe = sample_pair_record_contact_at(
                        current_frame_pos,
                        velocity=current_frame_vel,
                        contact_selection=selection,
                    )
                    retest_contact = retest_probe.get("contact")
                    retest_reject = retest_probe.get("reject")
                retest_accept = (
                    retest_probe is not None
                    and retest_probe.get("selected_raw_error") is None
                    and retest_contact is not None
                    and retest_reject == ""
                    and retest_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                )
                remaining_endpoint_pos = None
                remaining_endpoint_vel = None
                no_response_endpoint_pos = None
                no_response_endpoint_vel = None
                if remaining_time is not None:
                    no_response_endpoint_pos, no_response_endpoint_vel = (
                        integrate_frame_phase_state(
                            candidate,
                            candidate_velocity,
                            remaining_time,
                        )
                    )
                    if isinstance(response_preview, dict):
                        remaining_start_pos = (
                            finite_triplet(response_preview.get("post_contact_pos"))
                            or finite_triplet(candidate)
                        )
                        remaining_start_vel = (
                            finite_triplet(response_preview.get("post_contact_vel"))
                            or finite_triplet(candidate_velocity)
                        )
                        if (
                            response_preview.get("applied") is True
                            and remaining_start_pos is not None
                            and remaining_start_vel is not None
                        ):
                            remaining_endpoint_pos, remaining_endpoint_vel = (
                                integrate_frame_phase_state(
                                    remaining_start_pos,
                                    remaining_start_vel,
                                    remaining_time,
                                )
                            )

                preview = {
                    "enabled": True,
                    "runtime_default": "off",
                    "decompile_source": (
                        "GUESS6_CollisionPairPool_process_all passes full "
                        "frame_dt into GUESS6_CollisionPair_resolve; buckets "
                        "order pairs only, and resolve continues entities for "
                        "max(frame_dt - collision_time, 0)."
                    ),
                    "bucket_order_only": True,
                    "bucket_center_probe_is_diagnostic": True,
                    "label": label,
                    "contact_selection": selection,
                    "collision_time_s": collision_time,
                    "frame_dt_s": frame_dt,
                    "collision_pair_bucket": bucket_index,
                    "remaining_after_resolve_s": remaining_time,
                    "collision_time_pose": candidate,
                    "collision_time_velocity": candidate_velocity,
                    "resolve_retest_pose": current_frame_pos,
                    "resolve_retest_velocity": current_frame_vel,
                    "resolve_retest_reject": retest_reject,
                    "resolve_retest_accept": retest_accept,
                    "resolve_retest_contact": probe_contact_fields(
                        retest_contact,
                        center=current_frame_pos,
                        z_lift_used=0.0,
                    ),
                    "response_preview_applied": (
                        response_preview.get("applied")
                        if isinstance(response_preview, dict)
                        else None
                    ),
                    "remaining_endpoint_without_response_pos": (
                        no_response_endpoint_pos
                    ),
                    "remaining_endpoint_without_response_vel": (
                        no_response_endpoint_vel
                    ),
                    "post_response_remaining_endpoint_pos": remaining_endpoint_pos,
                    "post_response_remaining_endpoint_vel": remaining_endpoint_vel,
                    "candidate_had_contact": pair_contact is not None,
                }
                return {
                    key: value
                    for key, value in preview.items()
                    if value not in ({}, None)
                }

            if (
                frame_dt > 0.0
                and pre_step_frame_pos is not None
                and current_frame_pos is not None
            ):
                for bucket_index in range(bucket_count):
                    fraction = (float(bucket_index) + 0.5) / float(bucket_count)
                    label = f"pre_to_current_bucket_{bucket_index:02d}_center"
                    candidate = (
                        pre_step_frame_pos[0]
                        + (current_frame_pos[0] - pre_step_frame_pos[0]) * fraction,
                        pre_step_frame_pos[1]
                        + (current_frame_pos[1] - pre_step_frame_pos[1]) * fraction,
                        pre_step_frame_pos[2]
                        + (current_frame_pos[2] - pre_step_frame_pos[2]) * fraction,
                    )
                    if pre_step_frame_vel is not None and current_frame_vel is not None:
                        candidate_velocity = (
                            pre_step_frame_vel[0]
                            + (current_frame_vel[0] - pre_step_frame_vel[0])
                            * fraction,
                            pre_step_frame_vel[1]
                            + (current_frame_vel[1] - pre_step_frame_vel[1])
                            * fraction,
                            pre_step_frame_vel[2]
                            + (current_frame_vel[2] - pre_step_frame_vel[2])
                            * fraction,
                        )
                        velocity_source = "pre_to_current_bucket_center_fraction"
                    else:
                        candidate_velocity = velocity
                        velocity_source = "fallback_bucket_center_velocity"
                    frame_candidates.append(
                        (
                            label,
                            candidate,
                            candidate_velocity,
                            fraction,
                            velocity_source,
                        )
                    )

            results = {}
            accepted_rows = []
            for (
                label,
                candidate,
                candidate_velocity,
                velocity_fraction,
                velocity_source,
            ) in frame_candidates:
                time_s, bucket_index, bucket_bounds = bucket_for_fraction(
                    velocity_fraction
                )
                selection_results = {}
                for selection in selections:
                    pair_probe = sample_pair_record_contact_at(
                        candidate,
                        velocity=candidate_velocity,
                        contact_selection=selection,
                    )
                    pair_contact = pair_probe.get("contact")
                    pair_delta_contact = pair_probe.get("delta_contact")
                    reject = pair_probe.get("reject")
                    accepted = (
                        pair_probe.get("selected_raw_error") is None
                        and pair_contact is not None
                        and reject == ""
                        and pair_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                    )
                    response_preview = None
                    if accepted:
                        response_preview = preview_frame_phase_pair_response(
                            label=label,
                            pos=candidate,
                            velocity=candidate_velocity,
                            velocity_fraction=velocity_fraction,
                            velocity_source=velocity_source,
                            contact=pair_contact,
                            delta_contact=pair_delta_contact,
                        )
                    resolve_phase_preview = None
                    if pair_record_frame_phase_resolve_preview_all or (
                        accepted and pair_record_frame_phase_resolve_preview_accepted
                    ):
                        resolve_phase_preview = preview_frame_phase_resolve_model(
                            label=label,
                            selection=selection,
                            candidate=candidate,
                            candidate_velocity=candidate_velocity,
                            time_s=time_s,
                            bucket_index=bucket_index,
                            pair_contact=pair_contact,
                            response_preview=response_preview,
                        )
                    row = {
                        "label": label,
                        "pos": candidate,
                        "velocity": candidate_velocity,
                        "velocity_fraction": velocity_fraction,
                        "velocity_source": velocity_source,
                        "server_report_time_s": time_s,
                        "bucket_count": bucket_count,
                        "bucket_index": bucket_index,
                        "bucket_start_s": (
                            None if bucket_bounds is None else bucket_bounds[0]
                        ),
                        "bucket_end_s": (
                            None if bucket_bounds is None else bucket_bounds[1]
                        ),
                        "contact_selection": selection,
                        "reject": reject,
                        "selected_raw_error": pair_probe.get("selected_raw_error"),
                        "selected_pair_contact_source": pair_probe.get(
                            "selected_pair_contact_source"
                        ),
                        "contact": probe_contact_fields(
                            pair_contact,
                            center=candidate,
                            z_lift_used=0.0,
                        ),
                        "delta_contact": probe_contact_fields(
                            pair_delta_contact,
                            center=candidate,
                            z_lift_used=0.0,
                        ),
                        "accepted": accepted,
                        "response_preview": response_preview,
                        "resolve_phase_preview": resolve_phase_preview,
                    }
                    row = {
                        key: value
                        for key, value in row.items()
                        if value not in ({}, None)
                    }
                    selection_results[selection] = row
                    if accepted:
                        accepted_rows.append(row)
                results[label] = selection_results
            return {
                "enabled": True,
                "runtime_default": "off",
                "decompile_source": (
                    "Report-first GUESS4_CBSP_edge_triangle_intersect selections "
                    "sampled at server frame reference poses and 30 "
                    "CollisionPair_record bucket centers before applying any terrain "
                    "response. Resolve-phase previews are report-only: bucket "
                    "centers are diagnostic, while OG resolve retests at the "
                    "post-step pose and continues the remaining frame time."
                ),
                "frame_dt_s": frame_dt,
                "frame_pose_start": pre_step_frame_pos,
                "frame_pose_end": current_frame_pos,
                "frame_pose_delta": frame_pose_delta,
                "frame_pose_span_u": frame_pose_span_u,
                "frame_pose_span_verdict": frame_pose_span_verdict,
                "frame_velocity_start": pre_step_frame_vel,
                "frame_velocity_end": current_frame_vel,
                "frame_velocity_delta": frame_velocity_delta,
                "frame_velocity_span_u": frame_velocity_span_u,
                "frame_pose_velocity_integrated_end": (
                    frame_pose_velocity_integrated_end
                ),
                "frame_pose_integrated_delta_from_source_end": (
                    frame_pose_integrated_delta_from_source_end
                ),
                "frame_pose_integrated_error_u": frame_pose_integrated_error_u,
                "frame_pose_motion_consistency_verdict": (
                    frame_pose_motion_consistency_verdict
                ),
                "bucket_count": bucket_count,
                "bucket_center_sample_count": bucket_count,
                "resolve_preview_mode": (
                    pair_record_frame_phase_resolve_preview_mode
                ),
                "accepted_count": len(accepted_rows),
                "first_accepted": accepted_rows[0] if accepted_rows else None,
                "results": results,
            }

        def select_reference_pose_pair_record_contact(pos, *, velocity=None):
            if not (
                reference_pose_contact_response_enabled
                or reference_pose_pair_response_enabled
            ):
                return None
            if not pair_record_contact_enabled:
                return None
            current_pos = finite_triplet(pos)
            candidates = dict(reference_pose_candidates(pos))
            for label in reference_pose_contact_order:
                candidate = candidates.get(label)
                if candidate is None:
                    continue
                if (
                    reference_pose_pair_response_enabled
                    and reference_pose_pair_response_max_distance > 0.0
                    and current_pos is not None
                ):
                    dx = candidate[0] - current_pos[0]
                    dy = candidate[1] - current_pos[1]
                    dz = candidate[2] - current_pos[2]
                    if (
                        math.sqrt(dx * dx + dy * dy + dz * dz)
                        > reference_pose_pair_response_max_distance
                    ):
                        continue
                candidate_velocity, velocity_fraction, velocity_source = (
                    reference_pose_candidate_velocity(
                        label,
                        candidate,
                        fallback=velocity,
                    )
                )
                pair_probe = sample_pair_record_contact_at(
                    candidate,
                    velocity=candidate_velocity,
                )
                pair_contact = pair_probe.get("contact")
                if (
                    pair_probe.get("selected_raw_error") is None
                    and pair_contact is not None
                    and pair_probe.get("reject") == ""
                    and pair_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                ):
                    return {
                        "label": label,
                        "pos": candidate,
                        "velocity": candidate_velocity,
                        "velocity_fraction": velocity_fraction,
                        "velocity_source": velocity_source,
                        "probe": pair_probe,
                        "contact": pair_contact,
                        "delta_contact": pair_probe.get("delta_contact"),
                    }
            return None

        def update_contact_probe(
            pos,
            lifted_contact,
            *,
            reason,
            raw_contact=None,
            raw_bounds_contact=None,
            raw_error=None,
            raw_fallback_reject=None,
            tank_raw_origin_fallback=False,
            pair_record_contact=None,
            pair_record_delta_contact=None,
            pair_record_contact_reject=None,
            pair_record_raw_contact=None,
            pair_record_selected_raw_contact=None,
            pair_record_selected_raw_bounds_contact=None,
            pair_record_selected_raw_error=None,
            pair_record_selected_pair_contact_source=None,
            raycast_probe=None,
        ):
            ctx.debug_last_terrain_contact_probe = {}
            if (
                not contact_probe_enabled
                or collision_model is None
                or vertices is None
                or cbsp_tree is None
            ):
                return
            if not finite_values((*pos, heading)):
                ctx.debug_last_terrain_contact_probe = {
                    "reason": "nonfinite_probe_position",
                    "origin_mode": origin_mode,
                    "probe_enabled": True,
                }
                return
            lifted_center = (pos[0], pos[1], pos[2] + z_lift)
            raw_center = (pos[0], pos[1], pos[2])
            if (
                lifted_contact is None
                and raw_contact is None
                and raw_bounds_contact is None
                and raw_error is None
            ):
                raw_contact, raw_bounds_contact, raw_error = sample_raw_origin_contact_at(pos)
                if raw_error is not None:
                    ctx.debug_last_terrain_contact_probe = {
                        "reason": "raw_origin_probe_error",
                        "error": str(raw_error),
                        "origin_mode": origin_mode,
                        "probe_enabled": True,
                    }
                    return

            if lifted_contact is not None:
                probe_reason = "lifted_contact"
            elif isinstance(raycast_probe, dict) and raycast_probe.get("contact") is not None:
                probe_reason = "lifted_clear_raycast_contact"
            elif raw_contact is not None:
                probe_reason = "lifted_clear_raw_origin_contact"
            elif raw_bounds_contact is not None:
                probe_reason = "lifted_clear_raw_origin_bounds_contact"
            else:
                probe_reason = reason
            reference_pose_contacts = (
                {}
                if (
                    not reference_pose_probe_enabled
                    or probe_reason == "lifted_contact"
                    or raw_error == "dirty_bounds_contact"
                )
                else sample_reference_pose_contacts(pos, velocity=(vx, vy, vz))
            )
            frame_phase_probe = sample_frame_phase_report_first_probe(
                pos,
                velocity=(vx, vy, vz),
            )

            def selected_row_contact_phase_trace():
                if not selected_row_phase_trace_enabled:
                    return None
                controller = getattr(ctx, "debug_last_controller_step", None)
                if not isinstance(controller, dict):
                    controller = {}
                trace = {
                    "enabled": True,
                    "runtime_default": "off",
                    "mode": selected_row_phase_trace_mode,
                    "decompile_source": (
                        "Selected-row phase trace for the server pose/contact "
                        "context that feeds CollisionPair_record. It is "
                        "report-only and does not change contact selection or "
                        "response."
                    ),
                    "controller_time": controller.get("controller_time"),
                    "controller_tick": controller.get("controller_tick"),
                    "controller_physics_step_count": controller.get(
                        "controller_physics_step_count"
                    ),
                    "action_packet_count_at_controller": controller.get(
                        "action_packet_count_at_controller"
                    ),
                    "last_action_client_tick_at_controller": controller.get(
                        "last_action_client_tick_at_controller"
                    ),
                    "last_action_age_s_at_controller": controller.get(
                        "last_action_age_s_at_controller"
                    ),
                    "movement_history_latest_nonzero_age_s_at_controller": (
                        controller.get(
                            "movement_history_latest_nonzero_age_s_at_controller"
                        )
                    ),
                    "movement_history_latest_nonzero_fwd_at_controller": (
                        controller.get(
                            "movement_history_latest_nonzero_fwd_at_controller"
                        )
                    ),
                    "frame_dt_s": dt,
                    "pre_step_pos": pre_pos,
                    "pre_step_vel": pre_vel,
                    "selected_pos": pos,
                    "selected_vel": (vx, vy, vz),
                    "dirty_reference_pos": dirty_dispatch_debug.get(
                        "dirty_reference_pos"
                    ),
                    "dirty_current_pos": dirty_dispatch_debug.get(
                        "dirty_current_pos"
                    ),
                    "heading": heading,
                    "body_matrix": model_contact_rotation_matrix,
                    "model_contact_rotation_source": model_contact_rotation_source,
                    "model_contact_rotation_mode": model_contact_rotation_mode,
                    "model_contact_selection": model_contact_selection,
                    "pair_record_contact_selection": pair_record_contact_selection,
                    "pair_record_contact_reject": pair_record_contact_reject,
                    "pair_record_selected_raw_error": (
                        pair_record_selected_raw_error
                    ),
                    "pair_record_selected_pair_contact_source": (
                        pair_record_selected_pair_contact_source
                    ),
                    "pair_record_contact_source_family": (
                        getattr(pair_record_contact, "cbsp_record_hit_source", None)
                        if pair_record_contact is not None
                        else None
                    ),
                    "pair_record_selected_raw_source_family": (
                        getattr(
                            pair_record_selected_raw_contact,
                            "cbsp_record_hit_source",
                            None,
                        )
                        if pair_record_selected_raw_contact is not None
                        else None
                    ),
                    "pair_record_raw_contact": probe_contact_fields(
                        pair_record_raw_contact,
                        center=raw_center,
                        z_lift_used=0.0,
                    ),
                    "pair_record_selected_raw_contact": probe_contact_fields(
                        pair_record_selected_raw_contact,
                        center=raw_center,
                        z_lift_used=0.0,
                    ),
                    "pair_record_selected_raw_bounds_contact": probe_contact_fields(
                        pair_record_selected_raw_bounds_contact,
                        center=raw_center,
                        z_lift_used=0.0,
                    ),
                    "pair_record_contact": probe_contact_fields(
                        pair_record_contact,
                        center=raw_center,
                        z_lift_used=0.0,
                    ),
                    "pair_record_delta_contact": probe_contact_fields(
                        pair_record_delta_contact,
                        center=raw_center,
                        z_lift_used=0.0,
                    ),
                    "selected_row_contact_accept": (
                        pair_record_contact is not None
                        and pair_record_contact_reject == ""
                    ),
                    "selected_row_lifted_contact": lifted_contact is not None,
                    "selected_row_raw_contact": raw_contact is not None,
                    "selected_row_raw_bounds_contact": raw_bounds_contact is not None,
                    "selected_row_raycast_contact": (
                        isinstance(raycast_probe, dict)
                        and raycast_probe.get("contact") is not None
                    ),
                    "selected_row_probe_reason": probe_reason,
                    "frame_phase_probe_enabled": (
                        pair_record_frame_phase_probe_enabled
                    ),
                    "frame_phase_probe_result_count": (
                        len(frame_phase_probe.get("results", {}))
                        if isinstance(frame_phase_probe, dict)
                        else 0
                    ),
                    "frame_phase_probe_accepted_count": (
                        frame_phase_probe.get("accepted_count")
                        if isinstance(frame_phase_probe, dict)
                        else None
                    ),
                }
                return {
                    key: value
                    for key, value in trace.items()
                    if value not in ({}, None)
                }

            selected_phase_trace = selected_row_contact_phase_trace()
            ctx.debug_last_terrain_contact_probe = {
                "reason": probe_reason,
                "origin_mode": origin_mode,
                "contact_response": contact_response,
                "contact_timing_mode": contact_timing_mode,
                "terrain_collision_shape": terrain_collision_shape,
                "model_contact_rotation_source": model_contact_rotation_source,
                "model_contact_rotation_mode": model_contact_rotation_mode,
                "model_contact_selection": model_contact_selection,
                "probe_enabled": True,
                "reference_pose_probe_enabled": reference_pose_probe_enabled,
                "reference_pose_contact_response_enabled": (
                    reference_pose_contact_response_enabled
                ),
                "reference_pose_pair_response_enabled": (
                    reference_pose_pair_response_enabled
                ),
                "reference_pose_pair_response_apply_enabled": (
                    reference_pose_pair_response_apply_enabled
                ),
                "reference_pose_pair_response": dirty_dispatch_debug.get(
                    "reference_pose_pair_response"
                ),
                "reference_pose_contact_order": reference_pose_contact_order,
                "position": pos,
                "velocity": (vx, vy, vz),
                "pre_step_pos": pre_pos,
                "pre_step_vel": pre_vel,
                "step_dt": dt,
                "heading": heading,
                "model_z_lift": z_lift,
                "bounding_radius": bounding_radius,
                "dirty_bounds_active": dirty_dispatch_debug.get("dirty_bounds_active"),
                "dirty_threshold_sq": dirty_dispatch_debug.get("dirty_threshold_sq"),
                "dirty_displacement_sq": dirty_dispatch_debug.get("dirty_displacement_sq"),
                "dirty_reference_pos": dirty_dispatch_debug.get("dirty_reference_pos"),
                "dirty_current_pos": dirty_dispatch_debug.get("dirty_current_pos"),
                "dirty_miss_refresh_enabled": dirty_dispatch_debug.get(
                    "dirty_miss_refresh_enabled"
                ),
                "dirty_miss_ref_action": dirty_dispatch_debug.get("dirty_miss_ref_action"),
                "dirty_miss_reason": dirty_dispatch_debug.get("dirty_miss_reason"),
                "dirty_model_center_mode": dirty_dispatch_debug.get("dirty_model_center_mode"),
                "dirty_model_bounds_center": dirty_dispatch_debug.get(
                    "dirty_model_bounds_center"
                ),
                "dirty_model_collision_center": dirty_dispatch_debug.get(
                    "dirty_model_collision_center"
                ),
                "dirty_bounds_aabb_min": dirty_dispatch_debug.get(
                    "dirty_bounds_aabb_min"
                ),
                "dirty_bounds_aabb_max": dirty_dispatch_debug.get(
                    "dirty_bounds_aabb_max"
                ),
                "dirty_bounds_xy_overlap": dirty_dispatch_debug.get(
                    "dirty_bounds_xy_overlap"
                ),
                "dirty_bounds_xy_overlap_error": dirty_dispatch_debug.get(
                    "dirty_bounds_xy_overlap_error"
                ),
                "dirty_bounds_box_fallback_enabled": dirty_dispatch_debug.get(
                    "dirty_bounds_box_fallback_enabled"
                ),
                "dirty_bounds_box_fallback_attempted": dirty_dispatch_debug.get(
                    "dirty_bounds_box_fallback_attempted"
                ),
                "dirty_bounds_box_fallback_applied": dirty_dispatch_debug.get(
                    "dirty_bounds_box_fallback_applied"
                ),
                "dirty_bounds_box_shape": dirty_dispatch_debug.get(
                    "dirty_bounds_box_shape"
                ),
                "dirty_bounds_safe_response_enabled": dirty_dispatch_debug.get(
                    "dirty_bounds_safe_response_enabled",
                    dirty_bounds_safe_response_enabled,
                ),
                "dirty_bounds_box_half_extents_source": dirty_dispatch_debug.get(
                    "dirty_bounds_box_half_extents_source"
                ),
                "dirty_bounds_box_half_extents": dirty_dispatch_debug.get(
                    "dirty_bounds_box_half_extents"
                ),
                "dirty_bounds_box_center_mode": dirty_dispatch_debug.get(
                    "dirty_bounds_box_center_mode"
                ),
                "dirty_bounds_box_z_offset": dirty_dispatch_debug.get(
                    "dirty_bounds_box_z_offset"
                ),
                "dirty_bounds_box_bounds_center": dirty_dispatch_debug.get(
                    "dirty_bounds_box_bounds_center"
                ),
                "dirty_bounds_box_collision_center": dirty_dispatch_debug.get(
                    "dirty_bounds_box_collision_center"
                ),
                "dirty_bounds_box_contact": dirty_dispatch_debug.get(
                    "dirty_bounds_box_contact"
                ),
                "dirty_bounds_box_reject": dirty_dispatch_debug.get(
                    "dirty_bounds_box_reject"
                ),
                "dirty_reference_pair_probe_enabled": (
                    dirty_dispatch_debug.get(
                        "dirty_reference_pair_probe_enabled",
                        dirty_reference_pair_probe_enabled,
                    )
                ),
                "dirty_reference_pair_probe": dirty_dispatch_debug.get(
                    "dirty_reference_pair_probe"
                ),
                "dirty_reference_pair_response_enabled": (
                    dirty_dispatch_debug.get(
                        "dirty_reference_pair_response_enabled",
                        dirty_reference_pair_response_enabled,
                    )
                ),
                "dirty_reference_pair_response_apply_enabled": (
                    dirty_dispatch_debug.get(
                        "dirty_reference_pair_response_apply_enabled",
                        dirty_reference_pair_response_apply_enabled,
                    )
                ),
                "dirty_reference_pair_response": dirty_dispatch_debug.get(
                    "dirty_reference_pair_response"
                ),
                "dirty_raycast_reject": dirty_dispatch_debug.get("dirty_raycast_reject"),
                "dirty_raycast_start": dirty_dispatch_debug.get("dirty_raycast_start"),
                "dirty_raycast_end": dirty_dispatch_debug.get("dirty_raycast_end"),
                "dirty_raycast_length": dirty_dispatch_debug.get("dirty_raycast_length"),
                "dirty_raycast_hit_position": dirty_dispatch_debug.get(
                    "dirty_raycast_hit_position"
                ),
                "dirty_raycast_hit_cell": dirty_dispatch_debug.get("dirty_raycast_hit_cell"),
                "raw_origin_fallback_enabled": bool(
                    raw_fallback_enabled or tank_raw_origin_fallback
                ),
                "tank_raw_origin_fallback": bool(tank_raw_origin_fallback),
                "raw_origin_fallback_reject": raw_fallback_reject,
                "pair_record_contact_enabled": pair_record_contact_enabled,
                "pair_record_bounds_sat_enabled": pair_record_bounds_sat_enabled,
                "pair_record_bounds_sat_apply_enabled": (
                    pair_record_bounds_sat_apply_enabled
                ),
                "pair_record_contact_response_profile": (
                    pair_record_contact_response_profile
                ),
                "pair_record_contact_reject": pair_record_contact_reject,
                "pair_record_contact_selection": pair_record_contact_selection,
                "pair_record_selected_raw_error": pair_record_selected_raw_error,
                "pair_record_selected_pair_contact_source": pair_record_selected_pair_contact_source,
                "pair_record_contact_normal_source": pair_record_contact_normal_source,
                "pair_record_contact_delta_normal_source": pair_record_contact_delta_normal_source,
                "pair_record_contact_vertical_delta_mode": pair_record_contact_vertical_delta_mode,
                "pair_record_contact_max_velocity_delta": pair_record_contact_max_velocity_delta,
                "pair_record_contact_max_vertical_delta": pair_record_contact_max_vertical_delta,
                "pair_record_cached_contact_enabled": pair_record_cached_contact_enabled,
                "pair_record_cached_contact_max_age_steps": (
                    pair_record_cached_contact_max_age_steps
                ),
                "pair_record_cached_contact_max_distance": (
                    pair_record_cached_contact_max_distance
                ),
                "pair_record_cached_contact_max_ref_distance": (
                    pair_record_cached_contact_max_ref_distance
                ),
                "pair_record_timed_contact_enabled": pair_record_timed_contact_enabled,
                "pair_record_timed_sweep_enabled": pair_record_timed_sweep_enabled,
                "pair_record_schedule_probe_enabled": pair_record_schedule_probe_enabled,
                "pair_record_spatial_ref_schedule_probe_enabled": (
                    pair_record_spatial_ref_schedule_probe_enabled
                ),
                "pair_record_frame_phase_probe_enabled": (
                    pair_record_frame_phase_probe_enabled
                ),
                "pair_record_frame_phase_probe": frame_phase_probe,
                "selected_row_phase_trace_enabled": (
                    selected_row_phase_trace_enabled
                ),
                "selected_row_phase_trace": selected_phase_trace,
                "pair_record_schedule_response_probe_enabled": (
                    pair_record_schedule_response_probe_enabled
                ),
                "pair_record_continue_remaining_enabled": pair_record_continue_remaining_enabled,
                "pair_record_deferred_prestep_enabled": (
                    pair_record_deferred_prestep_enabled
                ),
                "pair_record_deferred_prestep_probe_enabled": (
                    pair_record_deferred_prestep_probe_enabled
                ),
                "pair_record_deferred_prestep_max_distance": (
                    pair_record_deferred_prestep_max_distance
                ),
                "pair_record_phase_lookahead_enabled": (
                    pair_record_phase_lookahead_enabled
                ),
                "pair_record_phase_lookahead_apply_enabled": (
                    pair_record_phase_lookahead_apply_enabled
                ),
                "pair_record_phase_lookahead_queue_enabled": (
                    pair_record_phase_lookahead_queue_enabled
                ),
                "pair_record_phase_lookahead_mode": pair_record_phase_lookahead_mode,
                "pair_record_phase_lookahead_max_time_s": (
                    pair_record_phase_lookahead_max_time
                ),
                "pair_record_phase_lookahead_steps": (
                    pair_record_phase_lookahead_steps
                ),
                "pair_record_phase_lookahead_max_distance": (
                    pair_record_phase_lookahead_max_distance
                ),
                "pair_record_phase_lookahead_accel_mode": (
                    pair_record_phase_lookahead_accel_mode
                ),
                "pair_record_phase_backtrack_enabled": (
                    pair_record_phase_backtrack_enabled
                ),
                "pair_record_phase_backtrack_apply_enabled": (
                    pair_record_phase_backtrack_apply_enabled
                ),
                "pair_record_phase_backtrack_mode": (
                    pair_record_phase_backtrack_mode
                ),
                "pair_record_phase_backtrack_max_time_s": (
                    pair_record_phase_backtrack_max_time
                ),
                "pair_record_phase_backtrack_steps": (
                    pair_record_phase_backtrack_steps
                ),
                "pair_record_phase_backtrack_max_distance": (
                    pair_record_phase_backtrack_max_distance
                ),
                "pair_record_phase_backtrack_accel_mode": (
                    pair_record_phase_backtrack_accel_mode
                ),
                "pair_record_phase_backtrack_source": (
                    pair_record_phase_backtrack_source
                ),
                "timed_pair_response": timed_pair_response,
                "timing_ready": timing_ready,
                "contact_sweep_scan_enabled": contact_sweep_scan_enabled,
                "contact_sweep_scan_steps": contact_sweep_scan_steps,
                "raycast_fallback_enabled": raycast_fallback_enabled,
                "raycast_timed_fallback_enabled": raycast_fallback_timed_enabled,
                "raycast_fallback_reject": (
                    raycast_probe.get("reject")
                    if isinstance(raycast_probe, dict)
                    else None
                ),
                "raycast_fallback_probe": raycast_probe_fields(raycast_probe),
                "lifted_contact": probe_contact_fields(
                    lifted_contact,
                    center=lifted_center,
                    z_lift_used=z_lift,
                ),
                "raw_origin_contact": probe_contact_fields(
                    raw_contact,
                    center=raw_center,
                    z_lift_used=0.0,
                ),
                "pair_record_raw_contact": probe_contact_fields(
                    pair_record_raw_contact,
                    center=raw_center,
                    z_lift_used=0.0,
                ),
                "pair_record_selected_raw_contact": probe_contact_fields(
                    pair_record_selected_raw_contact,
                    center=raw_center,
                    z_lift_used=0.0,
                ),
                "pair_record_selected_raw_bounds_contact": probe_contact_fields(
                    pair_record_selected_raw_bounds_contact,
                    center=raw_center,
                    z_lift_used=0.0,
                ),
                "pair_record_contact": probe_contact_fields(
                    pair_record_contact,
                    center=raw_center,
                    z_lift_used=0.0,
                ),
                "pair_record_delta_contact": probe_contact_fields(
                    pair_record_delta_contact,
                    center=raw_center,
                    z_lift_used=0.0,
                ),
                "raw_origin_bounds_contact": probe_contact_fields(
                    raw_bounds_contact,
                    center=raw_center,
                    z_lift_used=0.0,
                ),
                "reference_pose_contacts": reference_pose_contacts,
            }

        def sample_contact_at(pos):
            if not finite_values((*pos, heading)):
                return None
            if collision_model is not None and terrain_collision_shape == "model":
                model_center = (pos[0], pos[1], pos[2] + z_lift)
                return self._terrain_grid_collision.test_model_collision(
                    model_center,
                    heading,
                    vertices,
                    cbsp_tree,
                    bounding_radius,
                    rotation_matrix=model_contact_rotation_matrix,
                    contact_selection=model_contact_selection,
                )
            box_center = (pos[0], pos[1], pos[2] + box_collision_z_lift())
            return self._terrain_grid_collision.test_box_collision(
                box_center,
                inertia_half_extents,
                heading,
            )

        def sample_contact():
            return sample_contact_at((anchor[0], anchor[1], anchor[2]))

        def apply_pair_solver_contact(
            contact,
            *,
            projection_order_override=None,
            friction_override=None,
        ):
            nonlocal vx, vy, vz
            try:
                correction_cap = float(
                    os.environ.get(
                        "WULFRAM_ENTITY_CONTACT_POSITION_CORRECTION_CAP",
                        str(self._PENETRATION_SLOP_DEFAULT),
                    )
                )
            except ValueError:
                correction_cap = self._PENETRATION_SLOP_DEFAULT
            try:
                constraint_iterations = int(
                    os.environ.get("WULFRAM_ENTITY_TERRAIN_CONSTRAINT_ITERATIONS", "100")
                )
            except ValueError:
                constraint_iterations = 100
            try:
                restitution_fraction = float(
                    os.environ.get("WULFRAM_ENTITY_TERRAIN_RESTITUTION_FRACTION", "0.1")
                )
            except ValueError:
                restitution_fraction = 0.1
            before_vel = (vx, vy, vz)
            before_ang = tuple(contact_angular_velocity)
            entity_type = ctx.entity_type
            if not isinstance(entity_type, EntityType):
                try:
                    entity_type = EntityType(int(entity_type))
                except (TypeError, ValueError):
                    entity_type = EntityType.TANK
            collision_config = self._ENTITY_COLLISION_TABLE.get(
                entity_type,
                self._ENTITY_COLLISION_DEFAULT,
            )
            projection_order = projection_order_override
            if projection_order is None:
                projection_order = os.environ.get("WULFRAM_ENTITY_TERRAIN_PROJECTION_ORDER")
            if (
                projection_order is None
                and entity_type == EntityType.TANK
                and tank_clean_pair_solver_enabled
                and contact_response == "auto"
                and origin_mode not in {"entity", "origin", "raw"}
            ):
                projection_order = os.environ.get(
                    "WULFRAM_TANK_CLEAN_TERRAIN_PROJECTION_ORDER",
                    "opposite_if_separating",
                )
            if projection_order is None:
                projection_order = "body_minus_world"
            constraint_kwargs = dict(
                position=(anchor[0], anchor[1], anchor[2]),
                velocity=(vx, vy, vz),
                angular_velocity=tuple(contact_angular_velocity),
                contact_point=contact.position,
                contact_normal=contact.normal,
                penetration=contact.penetration,
                half_extents=half_extents,
                inertia_half_extents=inertia_half_extents,
                mass=collision_config["mass"],
                friction=(
                    collision_config["friction"]
                    if friction_override is None
                    else friction_override
                ),
                body_should_sleep=bool(getattr(ctx, "rigid_body_should_sleep", False)),
                body_is_sleeping=bool(getattr(ctx, "rigid_body_sleeping", False)),
                slop=self._PENETRATION_SLOP_DEFAULT,
                correction_cap=correction_cap,
                constraint_iterations=constraint_iterations,
                solver_variant=os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_CONSTRAINT_SOLVER",
                    "constraint",
                ),
                restitution_fraction=restitution_fraction,
                projection_order=projection_order,
            )
            contact_rotation_frame = os.environ.get(
                "WULFRAM_ENTITY_CONTACT_ROTATION_FRAME",
                "decompile",
            ).strip().lower()
            if contact_rotation_frame not in {
                "0",
                "false",
                "off",
                "no",
                "identity",
                "legacy",
            }:
                constraint_kwargs["body_rotation"] = (
                    float((getattr(ctx, "player_pose", {}) or {}).get("roll", 0.0) or 0.0),
                    float((getattr(ctx, "player_pose", {}) or {}).get("pitch", 0.0) or 0.0),
                    float(getattr(ctx, "player_heading", heading) or 0.0),
                )
                constraint_kwargs["rotation_matrix"] = getattr(
                    ctx,
                    "spring_body_matrix",
                    None,
                )
            constraint_retest = os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_CONSTRAINT_RETEST",
                "0",
            ).strip().lower()
            if constraint_retest in {"1", "true", "on", "yes", "retest", "decompile"}:
                constraint_kwargs["enable_inactive_retest"] = True
                try:
                    constraint_kwargs["inactive_retest_bias"] = float(
                        os.environ.get(
                            "WULFRAM_ENTITY_TERRAIN_CONSTRAINT_RETEST_BIAS",
                            "0.1",
                        )
                    )
                except ValueError:
                    constraint_kwargs["inactive_retest_bias"] = 0.1
            static_target_mode = os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_STATIC_TARGET_SEPARATION",
                "0",
            ).strip().lower()
            if static_target_mode in {
                "1",
                "true",
                "on",
                "yes",
                "decompile",
                "static",
                "target",
            }:
                constraint_kwargs["target_separation"] = (
                    self._get_static_separation_from_contact(
                        (anchor[0], anchor[1], anchor[2]),
                        contact.position,
                    )
                )
            result = solve_static_terrain_constraint(**constraint_kwargs)
            if not finite_values((*result.position, *result.velocity, *result.angular_velocity)):
                return {
                    "response": "terrain_pair_solver_nonfinite_rejected",
                    "velocity_before": before_vel,
                    "velocity_after": before_vel,
                    "angular_velocity_before": before_ang,
                    "angular_velocity_after": before_ang,
                    "bad_position": result.position,
                    "bad_velocity": result.velocity,
                    "bad_angular_velocity": result.angular_velocity,
                }
            result_debug = dict(result.debug)
            yaw_feedback_mode = os.environ.get(
                "WULFRAM_ENTITY_TERRAIN_CONTACT_YAW_FEEDBACK",
                "0",
            ).strip().lower()
            yaw_feedback_enabled = yaw_feedback_mode in {
                "1",
                "true",
                "on",
                "yes",
                "decompile",
                "legacy",
            }
            raw_contact_angular_velocity = tuple(result.angular_velocity)
            applied_contact_angular_velocity = (
                raw_contact_angular_velocity
                if yaw_feedback_enabled
                else (
                    raw_contact_angular_velocity[0],
                    raw_contact_angular_velocity[1],
                    before_ang[2],
                )
            )
            current_tick = int(getattr(getattr(ctx, "session", None), "tick", 0) or getattr(ctx, "last_client_tick", 0) or 0)
            interp_decision = entity_interpolate_toward_target_decision(
                current_position=result.position,
                target_position=getattr(ctx, "rigid_body_target_pos", None),
                current_rotation=(
                    float((getattr(ctx, "player_pose", {}) or {}).get("roll", 0.0) or 0.0),
                    float((getattr(ctx, "player_pose", {}) or {}).get("pitch", 0.0) or 0.0),
                    float(getattr(ctx, "player_heading", heading) or 0.0),
                ),
                target_rotation=getattr(ctx, "rigid_body_target_rot", None),
                tolerance=float(getattr(ctx, "rigid_body_interp_tolerance", self._PENETRATION_SLOP_DEFAULT) or self._PENETRATION_SLOP_DEFAULT),
                combined_radius=bounding_radius,
                current_tick=current_tick,
                last_interp_tick=int(getattr(ctx, "rigid_body_last_interp_tick", 0) or 0),
                delta_seconds=1.0 / max(float(getattr(self, "tick_rate_hz", 30.0) or 30.0), 1e-6),
                wake_override=bool(getattr(ctx, "rigid_body_should_sleep", False)),
            )
            anchor[0], anchor[1], anchor[2] = result.position
            vx, vy, vz = result.velocity
            contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = applied_contact_angular_velocity
            if interp_decision.update_last_interp_tick:
                ctx.rigid_body_last_interp_tick = current_tick
            ctx.spring_body_ang_vel = (contact_angular_velocity[0], contact_angular_velocity[1])
            ctx.angular_vel_yaw = contact_angular_velocity[2]
            after_vel = (vx, vy, vz)
            result_debug["constraint_angular_velocity_after_raw"] = raw_contact_angular_velocity
            result_debug["constraint_angular_delta_raw"] = (
                raw_contact_angular_velocity[0] - before_ang[0],
                raw_contact_angular_velocity[1] - before_ang[1],
                raw_contact_angular_velocity[2] - before_ang[2],
            )
            result_debug["contact_yaw_feedback_enabled"] = yaw_feedback_enabled
            result_debug["contact_yaw_delta_suppressed"] = (
                0.0
                if yaw_feedback_enabled
                else raw_contact_angular_velocity[2] - before_ang[2]
            )
            result_debug["angular_velocity_after"] = tuple(contact_angular_velocity)
            result_debug["angular_delta"] = (
                contact_angular_velocity[0] - before_ang[0],
                contact_angular_velocity[1] - before_ang[1],
                contact_angular_velocity[2] - before_ang[2],
            )
            return {
                "velocity_before": before_vel,
                "velocity_after": after_vel,
                "angular_velocity_before": before_ang,
                "angular_velocity_after": tuple(contact_angular_velocity),
                **dict(interp_decision.debug),
                "interpolation_reset_physics": interp_decision.reset_physics,
                "interpolation_wake": interp_decision.wake,
                "interpolation_update_last_interp_tick": interp_decision.update_last_interp_tick,
                **result_debug,
            }

        def apply_contact(
            contact,
            *,
            force_pair_solver=False,
            projection_order_override=None,
            friction_override=None,
        ):
            nonlocal vx, vy, vz
            if pair_solver_response or force_pair_solver:
                if (
                    not force_pair_solver
                    and contact_response == "auto"
                    and ctx_entity_type == EntityType.TANK
                    and tank_clean_pair_solver_enabled
                    and origin_mode not in {"entity", "origin", "raw"}
                    and tank_clean_pair_solver_max_depth > 0.0
                    and float(contact.penetration) > tank_clean_pair_solver_max_depth
                ):
                    return {
                        "response": "terrain_contact_constraint_solver_depth_rejected",
                        "terrain_contact_depth_rejected": True,
                        "terrain_contact_depth": float(contact.penetration),
                        "terrain_contact_max_depth": tank_clean_pair_solver_max_depth,
                        "velocity_before": (vx, vy, vz),
                        "velocity_after": (vx, vy, vz),
                        "angular_velocity_before": tuple(contact_angular_velocity),
                        "angular_velocity_after": tuple(contact_angular_velocity),
                    }
                return apply_pair_solver_contact(
                    contact,
                    projection_order_override=projection_order_override,
                    friction_override=friction_override,
                )
            push = contact.penetration + self._get_static_separation_from_contact(
                (anchor[0], anchor[1], anchor[2]),
                contact.position,
            )
            anchor[0] += contact.normal[0] * push
            anchor[1] += contact.normal[1] * push
            anchor[2] += contact.normal[2] * push
            vel_dot = (
                vx * contact.normal[0] +
                vy * contact.normal[1] +
                vz * contact.normal[2]
            )
            if vel_dot < 0.0:
                vx -= contact.normal[0] * vel_dot
                vy -= contact.normal[1] * vel_dot
                vz -= contact.normal[2] * vel_dot
            return {
                "response": "terrain_legacy_projection",
                "position_correction": push,
                "normal_velocity_before": vel_dot,
            }

        def apply_raw_origin_fallback_contact(
            contact,
            *,
            projection_order=None,
            friction=None,
            delta_mode=None,
            delta_normal=None,
            delta_normal_source=None,
            angular_mode=None,
            closing_only=None,
            max_velocity_delta=None,
            max_vertical_delta=None,
            vertical_delta_mode=None,
            max_speed=None,
            max_angular_delta=None,
        ):
            nonlocal vx, vy, vz
            local_projection_order = (
                raw_fallback_projection_order
                if projection_order is None
                else projection_order
            )
            local_friction = raw_fallback_friction if friction is None else friction
            local_delta_mode = raw_fallback_delta_mode if delta_mode is None else delta_mode
            local_angular_mode = (
                raw_fallback_angular_mode if angular_mode is None else angular_mode
            )
            local_closing_only = (
                raw_fallback_closing_only if closing_only is None else closing_only
            )
            local_max_velocity_delta = (
                raw_fallback_max_velocity_delta
                if max_velocity_delta is None
                else max_velocity_delta
            )
            local_max_vertical_delta = max_vertical_delta
            local_vertical_delta_mode = (
                raw_fallback_vertical_delta_mode
                if vertical_delta_mode is None
                else vertical_delta_mode
            )
            local_max_speed = raw_fallback_max_speed if max_speed is None else max_speed
            local_max_angular_delta = (
                raw_fallback_max_angular_delta
                if max_angular_delta is None
                else max_angular_delta
            )
            before_pos = (anchor[0], anchor[1], anchor[2])
            before_vel = (vx, vy, vz)
            before_ang = tuple(contact_angular_velocity)
            response_debug = apply_contact(
                contact,
                force_pair_solver=True,
                projection_order_override=local_projection_order,
                friction_override=local_friction,
            ) or {}
            raw_after_pos = (anchor[0], anchor[1], anchor[2])
            raw_after_vel = (vx, vy, vz)
            raw_after_ang = tuple(contact_angular_velocity)
            if not finite_values((*raw_after_pos, *raw_after_vel, *raw_after_ang)):
                anchor[0], anchor[1], anchor[2] = before_pos
                vx, vy, vz = before_vel
                contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = before_ang
                ctx.spring_body_ang_vel = (before_ang[0], before_ang[1])
                ctx.angular_vel_yaw = before_ang[2]
                response_debug.update({
                    "response": "terrain_raw_origin_fallback_nonfinite_rejected",
                    "raw_origin_fallback_safety_rejected": True,
                    "raw_origin_fallback_position_after_unclamped": raw_after_pos,
                    "raw_origin_fallback_velocity_after_unclamped": raw_after_vel,
                    "raw_origin_fallback_angular_velocity_after_unclamped": raw_after_ang,
                    "velocity_after": before_vel,
                    "angular_velocity_after": before_ang,
                })
                return response_debug, False

            def vec_mag(values):
                return math.sqrt(sum(float(value) * float(value) for value in values))

            safe_vel = raw_after_vel
            raw_delta = (
                raw_after_vel[0] - before_vel[0],
                raw_after_vel[1] - before_vel[1],
                raw_after_vel[2] - before_vel[2],
            )
            solver_velocity_delta_unprojected = raw_delta
            solver_velocity_delta_mag_unprojected = vec_mag(
                solver_velocity_delta_unprojected
            )
            normal_delta_projected = False
            before_normal_speed = None
            before_center_normal_speed = None
            before_normal_speed_source = ""
            normal_delta_skip_reason = ""
            delta_projection_normal = None
            if local_delta_mode in {
                "normal",
                "normal_only",
                "contact_normal",
                "closing",
                "closing_velocity",
                "projection_speed",
                "target_speed",
                "decompile_projection",
                "center_closing",
                "center_closing_velocity",
                "center_projection_speed",
            }:
                normal = None
                try:
                    delta_projection_normal = (
                        delta_normal if delta_normal is not None else contact.normal
                    )
                    normal_mag = vec_mag(delta_projection_normal)
                    if normal_mag > 1e-9:
                        normal = (
                            float(delta_projection_normal[0]) / normal_mag,
                            float(delta_projection_normal[1]) / normal_mag,
                            float(delta_projection_normal[2]) / normal_mag,
                        )
                except (TypeError, ValueError, OverflowError, IndexError):
                    normal = None
                if normal is not None:
                    before_center_normal_speed = (
                        before_vel[0] * normal[0]
                        + before_vel[1] * normal[1]
                        + before_vel[2] * normal[2]
                    )
                    before_normal_speed = before_center_normal_speed
                    before_normal_speed_source = "center_velocity"
                    if local_delta_mode not in {
                        "center_closing",
                        "center_closing_velocity",
                        "center_projection_speed",
                    }:
                        for speed_key in (
                            "constraint_selected_separation_speed_before",
                            "point_normal_velocity_before",
                        ):
                            try:
                                candidate_speed = float(response_debug.get(speed_key))
                            except (TypeError, ValueError, OverflowError):
                                candidate_speed = None
                            if candidate_speed is not None and math.isfinite(candidate_speed):
                                before_normal_speed = candidate_speed
                                before_normal_speed_source = speed_key
                                break
                    if local_delta_mode in {
                        "closing",
                        "closing_velocity",
                        "projection_speed",
                        "target_speed",
                        "decompile_projection",
                        "center_closing",
                        "center_closing_velocity",
                        "center_projection_speed",
                    }:
                        try:
                            target_separation_speed = float(
                                response_debug.get(
                                    "target_separation",
                                    self._PENETRATION_SLOP_DEFAULT,
                                )
                            )
                        except (TypeError, ValueError, OverflowError):
                            target_separation_speed = self._PENETRATION_SLOP_DEFAULT
                        if not math.isfinite(target_separation_speed):
                            target_separation_speed = self._PENETRATION_SLOP_DEFAULT
                        normal_component = max(
                            0.0,
                            target_separation_speed - before_normal_speed,
                        )
                    else:
                        normal_component = (
                            raw_delta[0] * normal[0]
                            + raw_delta[1] * normal[1]
                            + raw_delta[2] * normal[2]
                        )
                    if (
                        normal_component > 0.0
                        and (
                            not local_closing_only
                            or before_normal_speed < 0.0
                        )
                    ):
                        raw_delta = (
                            normal[0] * normal_component,
                            normal[1] * normal_component,
                            normal[2] * normal_component,
                        )
                        safe_vel = (
                            before_vel[0] + raw_delta[0],
                            before_vel[1] + raw_delta[1],
                            before_vel[2] + raw_delta[2],
                        )
                    else:
                        if local_closing_only and before_normal_speed >= 0.0:
                            normal_delta_skip_reason = "separating_before_velocity"
                        elif normal_component <= 0.0:
                            normal_delta_skip_reason = "nonpositive_solver_normal_delta"
                        else:
                            normal_delta_skip_reason = "separating_before_velocity"
                        raw_delta = (0.0, 0.0, 0.0)
                        safe_vel = before_vel
                    normal_delta_projected = True
            raw_delta_mag = vec_mag(raw_delta)
            velocity_delta_clamped = False
            if (
                math.isfinite(local_max_velocity_delta)
                and local_max_velocity_delta > 0.0
                and raw_delta_mag > local_max_velocity_delta
            ):
                scale = local_max_velocity_delta / max(raw_delta_mag, 1e-9)
                safe_vel = (
                    before_vel[0] + raw_delta[0] * scale,
                    before_vel[1] + raw_delta[1] * scale,
                    before_vel[2] + raw_delta[2] * scale,
                )
                velocity_delta_clamped = True

            vertical_delta_clamped = False
            if local_max_vertical_delta is not None:
                try:
                    local_max_vertical_delta_value = float(local_max_vertical_delta)
                except (TypeError, ValueError, OverflowError):
                    local_max_vertical_delta_value = 0.0
                if (
                    math.isfinite(local_max_vertical_delta_value)
                    and local_max_vertical_delta_value > 0.0
                ):
                    current_delta_z = safe_vel[2] - before_vel[2]
                    if current_delta_z > local_max_vertical_delta_value:
                        current_delta = (
                            safe_vel[0] - before_vel[0],
                            safe_vel[1] - before_vel[1],
                            current_delta_z,
                        )
                        if (
                            local_vertical_delta_mode
                            in {"scale", "preserve_direction", "direction", "normal_scale"}
                            and current_delta_z > 1e-9
                        ):
                            scale = local_max_vertical_delta_value / current_delta_z
                            safe_vel = (
                                before_vel[0] + current_delta[0] * scale,
                                before_vel[1] + current_delta[1] * scale,
                                before_vel[2] + current_delta[2] * scale,
                            )
                        else:
                            safe_vel = (
                                safe_vel[0],
                                safe_vel[1],
                                before_vel[2] + local_max_vertical_delta_value,
                            )
                        vertical_delta_clamped = True
            else:
                local_max_vertical_delta_value = None

            speed_clamped = False
            safe_speed = vec_mag(safe_vel)
            if (
                math.isfinite(local_max_speed)
                and local_max_speed > 0.0
                and safe_speed > local_max_speed
            ):
                scale = local_max_speed / max(safe_speed, 1e-9)
                safe_vel = (
                    safe_vel[0] * scale,
                    safe_vel[1] * scale,
                    safe_vel[2] * scale,
                )
                speed_clamped = True

            safe_ang = raw_after_ang
            raw_ang_delta = (
                raw_after_ang[0] - before_ang[0],
                raw_after_ang[1] - before_ang[1],
                raw_after_ang[2] - before_ang[2],
            )
            solver_angular_delta_unprojected = raw_ang_delta
            raw_ang_delta_mag = vec_mag(raw_ang_delta)
            solver_angular_delta_mag_unprojected = raw_ang_delta_mag
            angular_delta_clamped = False
            angular_delta_preserved = False
            if local_angular_mode in {"preserve", "none", "linear", "linear_only", "off"}:
                safe_ang = before_ang
                angular_delta_preserved = True
            elif (
                local_angular_mode == "auto"
                and local_delta_mode in {"normal", "normal_only", "contact_normal"}
            ):
                safe_ang = before_ang
                angular_delta_preserved = True
            elif (
                math.isfinite(local_max_angular_delta)
                and local_max_angular_delta > 0.0
                and raw_ang_delta_mag > local_max_angular_delta
            ):
                scale = local_max_angular_delta / max(raw_ang_delta_mag, 1e-9)
                safe_ang = (
                    before_ang[0] + raw_ang_delta[0] * scale,
                    before_ang[1] + raw_ang_delta[1] * scale,
                    before_ang[2] + raw_ang_delta[2] * scale,
                )
                angular_delta_clamped = True

            vx, vy, vz = safe_vel
            contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = safe_ang
            ctx.spring_body_ang_vel = (safe_ang[0], safe_ang[1])
            ctx.angular_vel_yaw = safe_ang[2]
            final_delta = (
                safe_vel[0] - before_vel[0],
                safe_vel[1] - before_vel[1],
                safe_vel[2] - before_vel[2],
            )
            final_ang_delta = (
                safe_ang[0] - before_ang[0],
                safe_ang[1] - before_ang[1],
                safe_ang[2] - before_ang[2],
            )
            response_debug.update({
                "raw_origin_fallback_safety_rejected": False,
                "raw_origin_fallback_velocity_safety_max_delta": local_max_velocity_delta,
                "raw_origin_fallback_velocity_safety_max_vertical_delta": local_max_vertical_delta_value,
                "raw_origin_fallback_velocity_safety_max_speed": local_max_speed,
                "raw_origin_fallback_angular_safety_max_delta": local_max_angular_delta,
                "raw_origin_fallback_friction_override": local_friction,
                "raw_origin_fallback_delta_mode": local_delta_mode,
                "raw_origin_fallback_delta_normal": delta_projection_normal,
                "raw_origin_fallback_delta_normal_source": delta_normal_source,
                "raw_origin_fallback_angular_mode": local_angular_mode,
                "raw_origin_fallback_vertical_delta_mode": local_vertical_delta_mode,
                "raw_origin_fallback_closing_only": local_closing_only,
                "raw_origin_fallback_before_normal_speed": before_normal_speed,
                "raw_origin_fallback_before_normal_speed_source": before_normal_speed_source,
                "raw_origin_fallback_before_center_normal_speed": before_center_normal_speed,
                "raw_origin_fallback_normal_delta_skip_reason": normal_delta_skip_reason,
                "raw_origin_fallback_normal_delta_projected": normal_delta_projected,
                "raw_origin_fallback_velocity_after_unclamped": raw_after_vel,
                "raw_origin_fallback_solver_velocity_delta_unprojected": (
                    solver_velocity_delta_unprojected
                ),
                "raw_origin_fallback_solver_velocity_delta_mag_unprojected": (
                    solver_velocity_delta_mag_unprojected
                ),
                "raw_origin_fallback_velocity_delta_unclamped": raw_delta,
                "raw_origin_fallback_velocity_delta_mag_unclamped": raw_delta_mag,
                "raw_origin_fallback_velocity_delta_clamped": velocity_delta_clamped,
                "raw_origin_fallback_vertical_delta_clamped": vertical_delta_clamped,
                "raw_origin_fallback_speed_clamped": speed_clamped,
                "raw_origin_fallback_angular_velocity_after_unclamped": raw_after_ang,
                "raw_origin_fallback_solver_angular_delta_unprojected": (
                    solver_angular_delta_unprojected
                ),
                "raw_origin_fallback_solver_angular_delta_mag_unprojected": (
                    solver_angular_delta_mag_unprojected
                ),
                "raw_origin_fallback_angular_delta_unclamped": raw_ang_delta,
                "raw_origin_fallback_angular_delta_mag_unclamped": raw_ang_delta_mag,
                "raw_origin_fallback_angular_delta_clamped": angular_delta_clamped,
                "raw_origin_fallback_angular_preserved": angular_delta_preserved,
                "raw_origin_fallback_velocity_delta_after_safety": final_delta,
                "raw_origin_fallback_velocity_delta_mag_after_safety": vec_mag(final_delta),
                "raw_origin_fallback_angular_delta_after_safety": final_ang_delta,
                "raw_origin_fallback_angular_delta_mag_after_safety": vec_mag(final_ang_delta),
                "velocity_after": safe_vel,
                "angular_velocity_after": safe_ang,
                "angular_delta": final_ang_delta,
            })
            return response_debug, True

        def preview_frame_phase_pair_response(
            *,
            label,
            pos,
            velocity,
            velocity_fraction,
            velocity_source,
            contact,
            delta_contact=None,
        ):
            """Dry-run the pair-record response for an accepted frame-phase row."""

            nonlocal vx, vy, vz
            response_record = {
                "enabled": True,
                "runtime_default": "off",
                "apply_enabled": False,
                "label": label,
                "velocity_fraction": velocity_fraction,
                "velocity_source": velocity_source,
            }
            if contact is None:
                response_record["reject"] = "no_pair_record_contact"
                response_record["applied"] = False
                return response_record

            current_pos = (anchor[0], anchor[1], anchor[2])
            current_vel = (vx, vy, vz)
            current_ang = tuple(contact_angular_velocity)
            pair_pos = finite_triplet(pos) or current_pos
            pair_vel = finite_triplet(velocity) or current_vel
            pair_dx = pair_pos[0] - current_pos[0]
            pair_dy = pair_pos[1] - current_pos[1]
            pair_dz = pair_pos[2] - current_pos[2]
            response_record.update({
                "pos": pair_pos,
                "velocity_before": pair_vel,
                "current_pos": current_pos,
                "current_distance": math.sqrt(
                    pair_dx * pair_dx + pair_dy * pair_dy + pair_dz * pair_dz
                ),
                "current_xy_distance": math.sqrt(pair_dx * pair_dx + pair_dy * pair_dy),
                "current_z_delta": pair_dz,
                "contact": probe_contact_fields(
                    contact,
                    center=pair_pos,
                    z_lift_used=0.0,
                ),
            })

            saved_body_ang_vel = getattr(ctx, "spring_body_ang_vel", None)
            saved_yaw = getattr(ctx, "angular_vel_yaw", None)
            saved_last_interp_tick = getattr(ctx, "rigid_body_last_interp_tick", None)
            response_debug = {}
            applied = False
            post_contact_pos = None
            post_contact_vel = None
            post_contact_ang = None
            try:
                anchor[0], anchor[1], anchor[2] = pair_pos
                vx, vy, vz = pair_vel
                response_debug, applied = apply_raw_origin_fallback_contact(
                    contact,
                    projection_order=pair_record_contact_projection_order,
                    delta_mode=pair_record_contact_delta_mode,
                    delta_normal=(
                        None
                        if delta_contact is None
                        else delta_contact.normal
                    ),
                    delta_normal_source=(
                        None
                        if delta_contact is None
                        else getattr(delta_contact, "normal_source", None)
                    ),
                    angular_mode=pair_record_contact_angular_mode,
                    closing_only=pair_record_contact_closing_only,
                    max_velocity_delta=pair_record_contact_max_velocity_delta,
                    max_vertical_delta=pair_record_contact_max_vertical_delta,
                    vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                    max_speed=pair_record_contact_max_speed,
                    max_angular_delta=pair_record_contact_max_angular_delta,
                    friction=0.0,
                )
                post_contact_pos = (anchor[0], anchor[1], anchor[2])
                post_contact_vel = (vx, vy, vz)
                post_contact_ang = tuple(contact_angular_velocity)
            finally:
                anchor[0], anchor[1], anchor[2] = current_pos
                vx, vy, vz = current_vel
                contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = current_ang
                ctx.spring_body_ang_vel = saved_body_ang_vel
                ctx.angular_vel_yaw = saved_yaw
                ctx.rigid_body_last_interp_tick = saved_last_interp_tick

            velocity_delta = None
            angular_delta = None
            if post_contact_vel is not None:
                velocity_delta = (
                    post_contact_vel[0] - pair_vel[0],
                    post_contact_vel[1] - pair_vel[1],
                    post_contact_vel[2] - pair_vel[2],
                )
            if post_contact_ang is not None:
                angular_delta = (
                    post_contact_ang[0] - current_ang[0],
                    post_contact_ang[1] - current_ang[1],
                    post_contact_ang[2] - current_ang[2],
                )
            response_record.update({
                "reject": "" if applied else "response_not_applied",
                "applied": applied,
                "post_contact_pos": post_contact_pos,
                "post_contact_vel": post_contact_vel,
                "post_contact_ang": post_contact_ang,
                "velocity_delta": velocity_delta,
                "angular_delta": angular_delta,
                "preserved_position": True,
            })
            for key in (
                "response",
                "target_separation",
                "constraint_selected_separation_speed_before",
                "point_normal_velocity_before",
                "raw_origin_fallback_velocity_delta_clamped",
                "raw_origin_fallback_vertical_delta_clamped",
                "raw_origin_fallback_angular_preserved",
                "raw_origin_fallback_delta_mode",
                "raw_origin_fallback_delta_normal",
                "raw_origin_fallback_delta_normal_source",
                "raw_origin_fallback_velocity_after_unclamped",
                "raw_origin_fallback_velocity_delta_unclamped",
                "raw_origin_fallback_velocity_delta_after_safety",
                "raw_origin_fallback_angular_velocity_after_unclamped",
                "raw_origin_fallback_angular_delta_after_safety",
            ):
                if key in response_debug:
                    response_record[key] = response_debug.get(key)
            return response_record

        def resolve_reference_pose_pair_response(reference_pair):
            nonlocal vx, vy, vz
            if not reference_pose_pair_response_enabled:
                return False
            response_record = {
                "enabled": True,
                "apply_enabled": reference_pose_pair_response_apply_enabled,
            }
            dirty_dispatch_debug["reference_pose_pair_response"] = response_record
            if not isinstance(reference_pair, dict):
                response_record["reject"] = "no_reference_pose_pair_contact"
                return False
            contact = reference_pair.get("contact")
            if contact is None:
                response_record["reject"] = "no_reference_pose_pair_contact"
                return False

            current_pos = (anchor[0], anchor[1], anchor[2])
            current_vel = (vx, vy, vz)
            current_ang = tuple(contact_angular_velocity)
            pair_pos = finite_triplet(reference_pair.get("pos")) or current_pos
            pair_vel = finite_triplet(reference_pair.get("velocity")) or current_vel
            pair_delta_contact = reference_pair.get("delta_contact")
            pair_dx = pair_pos[0] - current_pos[0]
            pair_dy = pair_pos[1] - current_pos[1]
            pair_dz = pair_pos[2] - current_pos[2]
            pair_current_xy_distance = math.sqrt(pair_dx * pair_dx + pair_dy * pair_dy)
            pair_current_distance = math.sqrt(
                pair_dx * pair_dx + pair_dy * pair_dy + pair_dz * pair_dz
            )
            response_record.update({
                "max_distance": reference_pose_pair_response_max_distance,
                "current_pos": current_pos,
                "current_distance": pair_current_distance,
                "current_xy_distance": pair_current_xy_distance,
                "current_z_delta": pair_dz,
            })
            if (
                reference_pose_pair_response_max_distance > 0.0
                and pair_current_distance
                > reference_pose_pair_response_max_distance
            ):
                response_record["reject"] = "reference_pose_pair_response_too_far"
                response_record["applied"] = False
                response_record["label"] = reference_pair.get("label")
                response_record["pos"] = pair_pos
                response_record["velocity_before"] = pair_vel
                response_record["velocity_fraction"] = reference_pair.get(
                    "velocity_fraction"
                )
                response_record["velocity_source"] = reference_pair.get(
                    "velocity_source"
                )
                return False
            saved_body_ang_vel = getattr(ctx, "spring_body_ang_vel", None)
            saved_yaw = getattr(ctx, "angular_vel_yaw", None)
            saved_last_interp_tick = getattr(ctx, "rigid_body_last_interp_tick", None)
            response_debug = {}
            applied = False
            post_contact_pos = None
            post_contact_vel = None
            post_contact_ang = None
            try:
                anchor[0], anchor[1], anchor[2] = pair_pos
                vx, vy, vz = pair_vel
                response_debug, applied = apply_raw_origin_fallback_contact(
                    contact,
                    projection_order=pair_record_contact_projection_order,
                    delta_mode=pair_record_contact_delta_mode,
                    delta_normal=(
                        None
                        if pair_delta_contact is None
                        else pair_delta_contact.normal
                    ),
                    delta_normal_source=(
                        None
                        if pair_delta_contact is None
                        else getattr(pair_delta_contact, "normal_source", None)
                    ),
                    angular_mode=pair_record_contact_angular_mode,
                    closing_only=pair_record_contact_closing_only,
                    max_velocity_delta=pair_record_contact_max_velocity_delta,
                    max_vertical_delta=pair_record_contact_max_vertical_delta,
                    vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                    max_speed=pair_record_contact_max_speed,
                    max_angular_delta=pair_record_contact_max_angular_delta,
                    friction=0.0,
                )
                post_contact_pos = (anchor[0], anchor[1], anchor[2])
                post_contact_vel = (vx, vy, vz)
                post_contact_ang = tuple(contact_angular_velocity)
            finally:
                anchor[0], anchor[1], anchor[2] = current_pos
                vx, vy, vz = current_vel
                contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = current_ang
                ctx.spring_body_ang_vel = saved_body_ang_vel
                ctx.angular_vel_yaw = saved_yaw
                ctx.rigid_body_last_interp_tick = saved_last_interp_tick

            velocity_delta = None
            angular_delta = None
            if post_contact_vel is not None:
                velocity_delta = (
                    post_contact_vel[0] - pair_vel[0],
                    post_contact_vel[1] - pair_vel[1],
                    post_contact_vel[2] - pair_vel[2],
                )
            if post_contact_ang is not None:
                angular_delta = (
                    post_contact_ang[0] - current_ang[0],
                    post_contact_ang[1] - current_ang[1],
                    post_contact_ang[2] - current_ang[2],
                )
            response_record.update({
                "reject": "" if applied else "response_not_applied",
                "applied": applied,
                "label": reference_pair.get("label"),
                "pos": pair_pos,
                "velocity_before": pair_vel,
                "velocity_fraction": reference_pair.get("velocity_fraction"),
                "velocity_source": reference_pair.get("velocity_source"),
                "post_contact_pos": post_contact_pos,
                "post_contact_vel": post_contact_vel,
                "post_contact_ang": post_contact_ang,
                "velocity_delta": velocity_delta,
                "angular_delta": angular_delta,
                "preserved_position": True,
                "contact": probe_contact_fields(
                    contact,
                    center=pair_pos,
                    z_lift_used=0.0,
                ),
            })
            for key in (
                "response",
                "target_separation",
                "constraint_selected_separation_speed_before",
                "point_normal_velocity_before",
                "raw_origin_fallback_velocity_delta_clamped",
                "raw_origin_fallback_vertical_delta_clamped",
                "raw_origin_fallback_angular_preserved",
                "raw_origin_fallback_delta_mode",
                "raw_origin_fallback_delta_normal",
                "raw_origin_fallback_delta_normal_source",
                "raw_origin_fallback_velocity_after_unclamped",
                "raw_origin_fallback_velocity_delta_unclamped",
                "raw_origin_fallback_velocity_delta_after_safety",
                "raw_origin_fallback_angular_velocity_after_unclamped",
                "raw_origin_fallback_angular_delta_after_safety",
            ):
                if key in response_debug:
                    response_record[key] = response_debug.get(key)
            if not applied or not reference_pose_pair_response_apply_enabled:
                return False

            final_vel = (
                current_vel[0] + (velocity_delta[0] if velocity_delta else 0.0),
                current_vel[1] + (velocity_delta[1] if velocity_delta else 0.0),
                current_vel[2] + (velocity_delta[2] if velocity_delta else 0.0),
            )
            if not finite_values((*current_pos, *final_vel)):
                response_record["reject"] = "nonfinite_final_state"
                response_record["applied"] = False
                return False
            vx, vy, vz = final_vel
            if post_contact_ang is not None:
                contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = post_contact_ang
                ctx.spring_body_ang_vel = (
                    contact_angular_velocity[0],
                    contact_angular_velocity[1],
                )
                ctx.angular_vel_yaw = contact_angular_velocity[2]
            response_record["final_vel"] = final_vel
            response_record["applied_to_current_state"] = True
            response_debug = dict(response_debug or {})
            response_debug.update({
                "reference_pose_pair_response": True,
                "reference_pose_pair_response_label": reference_pair.get("label"),
                "reference_pose_pair_response_pos": pair_pos,
                "reference_pose_pair_response_preserved_position": True,
                "reference_pose_pair_response_max_distance": (
                    reference_pose_pair_response_max_distance
                ),
                "reference_pose_pair_response_current_distance": (
                    pair_current_distance
                ),
                "reference_pose_pair_response_current_xy_distance": (
                    pair_current_xy_distance
                ),
                "reference_pose_pair_response_current_z_delta": pair_dz,
                "reference_pose_pair_response_velocity_delta": velocity_delta,
                "reference_pose_pair_response_final_vel": final_vel,
                "reference_pose_pair_response_velocity_fraction": (
                    reference_pair.get("velocity_fraction")
                ),
                "reference_pose_pair_response_velocity_source": (
                    reference_pair.get("velocity_source")
                ),
                "velocity_before": current_vel,
                "velocity_after": final_vel,
            })
            ctx.debug_last_collision = {
                "kind": "terrain_reference_pose_pair_response",
                "point": contact.position,
                "normal": contact.normal,
                "depth": contact.penetration,
                **contact_debug_fields(contact),
                "pair_record_contact": True,
                "pair_record_contact_reason": "reference_pose_pair_response",
                "pair_record_contact_reject": "",
                "pair_record_contact_enabled": pair_record_contact_enabled,
                "pair_record_contact_response_profile": (
                    pair_record_contact_response_profile
                ),
                "pair_record_contact_selection": pair_record_contact_selection,
                "pair_record_contact_normal_source": pair_record_contact_normal_source,
                "pair_record_contact_delta_normal_source": (
                    pair_record_contact_delta_normal_source
                ),
                **response_debug,
            }
            ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
            return True

        def resolve_dirty_reference_pair_response(reference_pair):
            nonlocal vx, vy, vz
            if not dirty_reference_pair_response_enabled:
                return False
            response_record = {
                "enabled": True,
                "apply_enabled": dirty_reference_pair_response_apply_enabled,
            }
            dirty_dispatch_debug["dirty_reference_pair_response"] = response_record
            if not isinstance(reference_pair, dict):
                response_record["reject"] = "no_dirty_reference_pair_contact"
                return False
            contact = reference_pair.get("contact")
            if contact is None:
                response_record["reject"] = "no_dirty_reference_pair_contact"
                return False

            current_pos = (anchor[0], anchor[1], anchor[2])
            current_vel = (vx, vy, vz)
            current_ang = tuple(contact_angular_velocity)
            pair_pos = finite_triplet(reference_pair.get("pos")) or current_pos
            pair_vel = finite_triplet(reference_pair.get("velocity")) or current_vel
            pair_delta_contact = reference_pair.get("delta_contact")
            pair_dx = pair_pos[0] - current_pos[0]
            pair_dy = pair_pos[1] - current_pos[1]
            pair_dz = pair_pos[2] - current_pos[2]
            pair_current_xy_distance = math.sqrt(pair_dx * pair_dx + pair_dy * pair_dy)
            pair_current_distance = math.sqrt(
                pair_dx * pair_dx + pair_dy * pair_dy + pair_dz * pair_dz
            )
            response_record.update({
                "max_distance": dirty_reference_pair_response_max_distance,
                "current_pos": current_pos,
                "current_distance": pair_current_distance,
                "current_xy_distance": pair_current_xy_distance,
                "current_z_delta": pair_dz,
            })
            if (
                dirty_reference_pair_response_max_distance > 0.0
                and pair_current_distance
                > dirty_reference_pair_response_max_distance
            ):
                response_record["reject"] = (
                    "dirty_reference_pair_response_too_far"
                )
                response_record["applied"] = False
                response_record["label"] = reference_pair.get("label")
                response_record["pos"] = pair_pos
                response_record["velocity_before"] = pair_vel
                return False
            saved_body_ang_vel = getattr(ctx, "spring_body_ang_vel", None)
            saved_yaw = getattr(ctx, "angular_vel_yaw", None)
            saved_last_interp_tick = getattr(ctx, "rigid_body_last_interp_tick", None)
            response_debug = {}
            applied = False
            post_contact_pos = None
            post_contact_vel = None
            post_contact_ang = None
            try:
                anchor[0], anchor[1], anchor[2] = pair_pos
                vx, vy, vz = pair_vel
                response_debug, applied = apply_raw_origin_fallback_contact(
                    contact,
                    projection_order=pair_record_contact_projection_order,
                    delta_mode=pair_record_contact_delta_mode,
                    delta_normal=(
                        None
                        if pair_delta_contact is None
                        else pair_delta_contact.normal
                    ),
                    delta_normal_source=(
                        None
                        if pair_delta_contact is None
                        else getattr(pair_delta_contact, "normal_source", None)
                    ),
                    angular_mode=pair_record_contact_angular_mode,
                    closing_only=pair_record_contact_closing_only,
                    max_velocity_delta=pair_record_contact_max_velocity_delta,
                    max_vertical_delta=pair_record_contact_max_vertical_delta,
                    vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                    max_speed=pair_record_contact_max_speed,
                    max_angular_delta=pair_record_contact_max_angular_delta,
                    friction=0.0,
                )
                post_contact_pos = (anchor[0], anchor[1], anchor[2])
                post_contact_vel = (vx, vy, vz)
                post_contact_ang = tuple(contact_angular_velocity)
            finally:
                anchor[0], anchor[1], anchor[2] = current_pos
                vx, vy, vz = current_vel
                contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = current_ang
                ctx.spring_body_ang_vel = saved_body_ang_vel
                ctx.angular_vel_yaw = saved_yaw
                ctx.rigid_body_last_interp_tick = saved_last_interp_tick

            velocity_delta = None
            angular_delta = None
            if post_contact_vel is not None:
                velocity_delta = (
                    post_contact_vel[0] - pair_vel[0],
                    post_contact_vel[1] - pair_vel[1],
                    post_contact_vel[2] - pair_vel[2],
                )
            if post_contact_ang is not None:
                angular_delta = (
                    post_contact_ang[0] - current_ang[0],
                    post_contact_ang[1] - current_ang[1],
                    post_contact_ang[2] - current_ang[2],
                )
            response_record.update({
                "reject": "" if applied else "response_not_applied",
                "applied": applied,
                "label": reference_pair.get("label"),
                "pos": pair_pos,
                "velocity_before": pair_vel,
                "post_contact_pos": post_contact_pos,
                "post_contact_vel": post_contact_vel,
                "post_contact_ang": post_contact_ang,
                "velocity_delta": velocity_delta,
                "angular_delta": angular_delta,
                "preserved_position": True,
                "contact": probe_contact_fields(
                    contact,
                    center=pair_pos,
                    z_lift_used=0.0,
                ),
            })
            for key in (
                "response",
                "target_separation",
                "constraint_selected_separation_speed_before",
                "point_normal_velocity_before",
                "raw_origin_fallback_velocity_delta_clamped",
                "raw_origin_fallback_vertical_delta_clamped",
                "raw_origin_fallback_angular_preserved",
                "raw_origin_fallback_delta_mode",
                "raw_origin_fallback_delta_normal",
                "raw_origin_fallback_delta_normal_source",
                "raw_origin_fallback_velocity_after_unclamped",
                "raw_origin_fallback_velocity_delta_unclamped",
                "raw_origin_fallback_velocity_delta_after_safety",
                "raw_origin_fallback_angular_velocity_after_unclamped",
                "raw_origin_fallback_angular_delta_after_safety",
            ):
                if key in response_debug:
                    response_record[key] = response_debug.get(key)
            if not applied or not dirty_reference_pair_response_apply_enabled:
                return False

            final_vel = (
                current_vel[0] + (velocity_delta[0] if velocity_delta else 0.0),
                current_vel[1] + (velocity_delta[1] if velocity_delta else 0.0),
                current_vel[2] + (velocity_delta[2] if velocity_delta else 0.0),
            )
            if not finite_values((*current_pos, *final_vel)):
                response_record["reject"] = "nonfinite_final_state"
                response_record["applied"] = False
                return False
            vx, vy, vz = final_vel
            if post_contact_ang is not None:
                contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = post_contact_ang
                ctx.spring_body_ang_vel = (
                    contact_angular_velocity[0],
                    contact_angular_velocity[1],
                )
                ctx.angular_vel_yaw = contact_angular_velocity[2]
            response_record["final_vel"] = final_vel
            response_record["applied_to_current_state"] = True
            response_debug = dict(response_debug or {})
            response_debug.update({
                "dirty_reference_pair_response": True,
                "dirty_reference_pair_response_label": reference_pair.get("label"),
                "dirty_reference_pair_response_pos": pair_pos,
                "dirty_reference_pair_response_preserved_position": True,
                "dirty_reference_pair_response_max_distance": (
                    dirty_reference_pair_response_max_distance
                ),
                "dirty_reference_pair_response_current_distance": (
                    pair_current_distance
                ),
                "dirty_reference_pair_response_current_xy_distance": (
                    pair_current_xy_distance
                ),
                "dirty_reference_pair_response_current_z_delta": pair_dz,
                "dirty_reference_pair_response_velocity_delta": velocity_delta,
                "dirty_reference_pair_response_final_vel": final_vel,
                "velocity_before": current_vel,
                "velocity_after": final_vel,
            })
            ctx.debug_last_collision = {
                "kind": "terrain_dirty_reference_pair_response",
                "point": contact.position,
                "normal": contact.normal,
                "depth": contact.penetration,
                **contact_debug_fields(contact),
                "pair_record_contact": True,
                "pair_record_contact_reason": "dirty_reference_pair_response",
                "pair_record_contact_reject": "",
                "pair_record_contact_enabled": pair_record_contact_enabled,
                "pair_record_contact_response_profile": (
                    pair_record_contact_response_profile
                ),
                "pair_record_contact_selection": pair_record_contact_selection,
                "pair_record_contact_normal_source": pair_record_contact_normal_source,
                "pair_record_contact_delta_normal_source": (
                    pair_record_contact_delta_normal_source
                ),
                **response_debug,
            }
            ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
            return True

        def apply_iterative_start_contact(contact):
            before_pos = (anchor[0], anchor[1], anchor[2])
            before_vel = (vx, vy, vz)
            result = resolve_iterative_terrain_start_contact(
                position=before_pos,
                contact_normal=contact.normal,
                sample_contact=sample_contact_at,
                slop=self._PENETRATION_SLOP_DEFAULT,
                max_iterations=start_iterative_limit,
                use_vertical_fallback=True,
            )
            anchor[0], anchor[1], anchor[2] = result.position
            return {
                "velocity_before": before_vel,
                "velocity_after": before_vel,
                "position_before_iterative": before_pos,
                "position_after_iterative": result.position,
                **dict(result.debug),
            }

        def apply_dirty_bounds_contact(contact):
            nonlocal vx, vy, vz
            if pair_solver_response and dirty_bounds_safe_response_enabled:
                response_debug, applied_dirty_safety = apply_raw_origin_fallback_contact(
                    contact,
                    projection_order=pair_record_contact_projection_order,
                    delta_mode=pair_record_contact_delta_mode,
                    delta_normal=contact.normal,
                    delta_normal_source=getattr(contact, "normal_source", None),
                    angular_mode=pair_record_contact_angular_mode,
                    closing_only=pair_record_contact_closing_only,
                    max_velocity_delta=pair_record_contact_max_velocity_delta,
                    max_vertical_delta=pair_record_contact_max_vertical_delta,
                    vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                    max_speed=pair_record_contact_max_speed,
                    max_angular_delta=pair_record_contact_max_angular_delta,
                    friction=0.0,
                )
                ctx.debug_last_collision = {
                    "kind": (
                        "terrain_dirty_bounds_safety_limited"
                        if applied_dirty_safety
                        else "terrain_dirty_bounds_safety_rejected"
                    ),
                    "point": contact.position,
                    "normal": contact.normal,
                    "depth": contact.penetration,
                    "terrain_collision_shape": terrain_collision_shape,
                    **contact_debug_fields(contact),
                    "dirty_bounds_safe_response": True,
                    "dirty_bounds_safe_response_enabled": (
                        dirty_bounds_safe_response_enabled
                    ),
                    "dirty_bounds_safe_response_applied": applied_dirty_safety,
                    "dirty_bounds_safe_response_profile": (
                        pair_record_contact_response_profile
                    ),
                    "dirty_bounds_safe_response_source": (
                        dirty_dispatch_debug.get("dirty_bounds_contact_source")
                    ),
                    "detail": f"reference={reference_pos!r}",
                    **(response_debug or {}),
                }
                ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                return
            if pair_solver_response:
                response_debug = apply_pair_solver_contact(contact)
                ctx.debug_last_collision = {
                    "kind": "terrain_dirty_bounds",
                    "point": contact.position,
                    "normal": contact.normal,
                    "depth": contact.penetration,
                    "terrain_collision_shape": terrain_collision_shape,
                    **contact_debug_fields(contact),
                    "detail": f"reference={reference_pos!r}",
                    **(response_debug or {}),
                }
                ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                return
            separation = self._get_static_separation_from_contact(
                (anchor[0], anchor[1], anchor[2]),
                contact.position,
            )
            anchor[0] = contact.position[0] + contact.normal[0] * (bounding_radius + separation)
            anchor[1] = contact.position[1] + contact.normal[1] * (bounding_radius + separation)
            anchor[2] = contact.position[2] + contact.normal[2] * (bounding_radius + separation)
            vel_dot = (
                vx * contact.normal[0] +
                vy * contact.normal[1] +
                vz * contact.normal[2]
            )
            if vel_dot < 0.0:
                vx -= contact.normal[0] * vel_dot
                vy -= contact.normal[1] * vel_dot
                vz -= contact.normal[2] * vel_dot
            response_debug = {
                "response": "terrain_dirty_bounds_radius_projection",
                "position_correction": bounding_radius + separation,
                "normal_velocity_before": vel_dot,
            }
            ctx.debug_last_motion_collision = {
                "kind": "terrain_clean_contact",
                "point": contact.position,
                "normal": contact.normal,
                "depth": contact.penetration,
                "terrain_collision_shape": terrain_collision_shape,
                **contact_debug_fields(contact),
            }
            ctx.debug_last_collision = {
                "kind": "terrain_dirty_bounds",
                "point": contact.position,
                "normal": contact.normal,
                "depth": contact.penetration,
                "terrain_collision_shape": terrain_collision_shape,
                **contact_debug_fields(contact),
                "detail": f"reference={reference_pos!r}",
                **response_debug,
            }
            ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)

        def motion_state_at(start_pos, start_vel, acc, elapsed_s, max_elapsed):
            t = max(0.0, min(float(max_elapsed), float(elapsed_s)))
            return (
                (
                    start_pos[0] + start_vel[0] * t + 0.5 * acc[0] * t * t,
                    start_pos[1] + start_vel[1] * t + 0.5 * acc[1] * t * t,
                    start_pos[2] + start_vel[2] * t + 0.5 * acc[2] * t * t,
                ),
                (
                    start_vel[0] + acc[0] * t,
                    start_vel[1] + acc[1] * t,
                    start_vel[2] + acc[2] * t,
                ),
            )

        def timing_acceleration():
            frame_dt = max(float(dt), 1e-9)
            acc = (
                (endpoint_vel_for_timing[0] - pre_vel[0]) / frame_dt,
                (endpoint_vel_for_timing[1] - pre_vel[1]) / frame_dt,
                (endpoint_vel_for_timing[2] - pre_vel[2]) / frame_dt,
            )
            return acc

        def estimate_direct_pair_record_contact_timing():
            if not pair_record_schedule_probe_enabled:
                return None
            frame_dt = max(0.0, float(dt))
            if frame_dt <= 0.0:
                return None
            acc = timing_acceleration()

            def candidate_at(elapsed_s):
                pos, velocity = motion_state_at(
                    tuple(pre_pos),
                    tuple(pre_vel),
                    acc,
                    elapsed_s,
                    frame_dt,
                )
                pair_probe = sample_pair_record_contact_at(pos, velocity=velocity)
                pair_contact = pair_probe.get("contact")
                if (
                    pair_probe.get("selected_raw_error") is None
                    and pair_contact is not None
                    and pair_probe.get("reject") == ""
                    and pair_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                ):
                    return {
                        "pos": pos,
                        "velocity": velocity,
                        "probe": pair_probe,
                        "contact": pair_contact,
                        "delta_contact": pair_probe.get("delta_contact"),
                    }
                return None

            start_candidate = candidate_at(0.0)
            if start_candidate is not None:
                return {
                    "collision_time_s": 0.0,
                    "remaining_time_s": frame_dt,
                    "collision_at_start": True,
                    "sweep_iterations": 0,
                    "sweep_clear_count": 0,
                    "sweep_contact_count": 2,
                    "contact_sweep_scan": False,
                    "contact_sweep_scan_steps": 0,
                    "contact_sweep_scan_hit_time_s": None,
                    **start_candidate,
                }

            end_candidate = candidate_at(frame_dt)
            contact_sweep_scan = False
            contact_sweep_scan_hit_time = None
            scan_steps_used = 0
            lo = 0.0
            if end_candidate is None:
                found_time = None
                found_candidate = None
                prev_time = 0.0
                scan_steps_used = max(1, int(contact_sweep_scan_steps))
                for scan_index in range(1, scan_steps_used + 1):
                    scan_time = frame_dt * (
                        float(scan_index) / float(scan_steps_used + 1)
                    )
                    scan_candidate = candidate_at(scan_time)
                    if scan_candidate is not None:
                        found_time = scan_time
                        found_candidate = scan_candidate
                        break
                    prev_time = scan_time
                if found_time is None or found_candidate is None:
                    return None
                contact_sweep_scan = True
                contact_sweep_scan_hit_time = found_time
                lo = prev_time
                hi = found_time
                best = found_candidate
                clear_count = max(1, int(round(prev_time > 0.0)))
                contact_count = 1
            else:
                hi = frame_dt
                best = end_candidate
                clear_count = 1
                contact_count = 1
            iterations = 0
            while hi - lo > 0.0025 and iterations < 24:
                mid = (lo + hi) * 0.5
                mid_candidate = candidate_at(mid)
                iterations += 1
                if mid_candidate is None:
                    lo = mid
                    clear_count += 1
                else:
                    hi = mid
                    best = mid_candidate
                    contact_count += 1
            return {
                "collision_time_s": hi,
                "remaining_time_s": max(0.0, frame_dt - hi),
                "collision_at_start": False,
                "sweep_iterations": iterations,
                "sweep_clear_count": clear_count,
                "sweep_contact_count": contact_count,
                "contact_sweep_scan": contact_sweep_scan,
                "contact_sweep_scan_steps": scan_steps_used,
                "contact_sweep_scan_hit_time_s": contact_sweep_scan_hit_time,
                **best,
            }

        def estimate_spatial_ref_pair_record_contact_timing():
            if not pair_record_spatial_ref_schedule_probe_enabled:
                return None
            frame_dt = max(0.0, float(dt))
            if frame_dt <= 0.0:
                return {
                    "probe_result": "invalid_step_dt",
                    "step_dt_s": frame_dt,
                }
            start_pos = (
                finite_triplet(reference_pos)
                or finite_triplet(getattr(ctx, "world_collision_ref_pos", None))
            )
            end_pos = finite_triplet((anchor[0], anchor[1], anchor[2]))
            start_vel = finite_triplet(pre_vel) or finite_triplet((vx, vy, vz))
            end_vel = finite_triplet((vx, vy, vz))
            if (
                start_pos is None
                or end_pos is None
                or start_vel is None
                or end_vel is None
            ):
                return {
                    "probe_result": "nonfinite_spatial_ref_state",
                    "step_dt_s": frame_dt,
                    "ref_pos": start_pos,
                    "current_pos": end_pos,
                }
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            dz = end_pos[2] - start_pos[2]
            ref_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            ref_xy_distance = math.sqrt(dx * dx + dy * dy)
            acc = (
                (end_vel[0] - start_vel[0]) / frame_dt,
                (end_vel[1] - start_vel[1]) / frame_dt,
                (end_vel[2] - start_vel[2]) / frame_dt,
            )

            def candidate_at(elapsed_s):
                pos, velocity = motion_state_at(
                    start_pos,
                    start_vel,
                    acc,
                    elapsed_s,
                    frame_dt,
                )
                pair_probe = sample_pair_record_contact_at(pos, velocity=velocity)
                pair_contact = pair_probe.get("contact")
                if (
                    pair_probe.get("selected_raw_error") is None
                    and pair_contact is not None
                    and pair_probe.get("reject") == ""
                    and pair_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                ):
                    return {
                        "pos": pos,
                        "velocity": velocity,
                        "probe": pair_probe,
                        "contact": pair_contact,
                        "delta_contact": pair_probe.get("delta_contact"),
                    }
                return None

            base = {
                "probe_result": "no_interval_contact",
                "step_dt_s": frame_dt,
                "ref_pos": start_pos,
                "current_pos": end_pos,
                "ref_to_current_distance": ref_distance,
                "ref_to_current_xy_distance": ref_xy_distance,
                "ref_to_current_z_delta": dz,
                "start_velocity": start_vel,
                "end_velocity": end_vel,
            }
            start_candidate = candidate_at(0.0)
            if start_candidate is not None:
                base.update({
                    "probe_result": "interval_contact",
                    "collision_time_s": 0.0,
                    "remaining_time_s": frame_dt,
                    "collision_at_start": True,
                    "sweep_iterations": 0,
                    "sweep_clear_count": 0,
                    "sweep_contact_count": 2,
                    "contact_sweep_scan": False,
                    "contact_sweep_scan_steps": 0,
                    "contact_sweep_scan_hit_time_s": None,
                    **start_candidate,
                })
                return base

            end_candidate = candidate_at(frame_dt)
            contact_sweep_scan = False
            contact_sweep_scan_hit_time = None
            scan_steps_used = 0
            lo = 0.0
            if end_candidate is None:
                found_time = None
                found_candidate = None
                prev_time = 0.0
                scan_steps_used = max(1, int(contact_sweep_scan_steps))
                for scan_index in range(1, scan_steps_used + 1):
                    scan_time = frame_dt * (
                        float(scan_index) / float(scan_steps_used + 1)
                    )
                    scan_candidate = candidate_at(scan_time)
                    if scan_candidate is not None:
                        found_time = scan_time
                        found_candidate = scan_candidate
                        break
                    prev_time = scan_time
                if found_time is None or found_candidate is None:
                    return base
                contact_sweep_scan = True
                contact_sweep_scan_hit_time = found_time
                lo = prev_time
                hi = found_time
                best = found_candidate
                clear_count = max(1, int(round(prev_time > 0.0)))
                contact_count = 1
            else:
                hi = frame_dt
                best = end_candidate
                clear_count = 1
                contact_count = 1
            iterations = 0
            while hi - lo > 0.0025 and iterations < 24:
                mid = (lo + hi) * 0.5
                mid_candidate = candidate_at(mid)
                iterations += 1
                if mid_candidate is None:
                    lo = mid
                    clear_count += 1
                else:
                    hi = mid
                    best = mid_candidate
                    contact_count += 1
            base.update({
                "probe_result": "interval_contact",
                "collision_time_s": hi,
                "remaining_time_s": max(0.0, frame_dt - hi),
                "collision_at_start": False,
                "sweep_iterations": iterations,
                "sweep_clear_count": clear_count,
                "sweep_contact_count": contact_count,
                "contact_sweep_scan": contact_sweep_scan,
                "contact_sweep_scan_steps": scan_steps_used,
                "contact_sweep_scan_hit_time_s": contact_sweep_scan_hit_time,
                **best,
            })
            return base

        def probe_direct_pair_record_contact_response(pair_timing):
            nonlocal vx, vy, vz
            if not pair_record_schedule_response_probe_enabled:
                return {}
            if pair_timing is None:
                return {
                    "pair_record_schedule_response_probe_enabled": True,
                    "pair_record_schedule_response_probe_result": (
                        "no_interval_contact"
                    ),
                }
            saved_pos = (anchor[0], anchor[1], anchor[2])
            saved_vel = (vx, vy, vz)
            saved_ang = tuple(contact_angular_velocity)
            saved_body_ang_vel = getattr(ctx, "spring_body_ang_vel", None)
            saved_yaw = getattr(ctx, "angular_vel_yaw", None)
            saved_last_interp_tick = getattr(ctx, "rigid_body_last_interp_tick", None)
            response_debug = {}
            applied = False
            post_contact_pos = None
            post_contact_vel = None
            endpoint_pos = None
            endpoint_vel = None
            try:
                anchor[0], anchor[1], anchor[2] = pair_timing["pos"]
                vx, vy, vz = pair_timing["velocity"]
                response_debug, applied = apply_raw_origin_fallback_contact(
                    pair_timing["contact"],
                    projection_order=pair_record_contact_projection_order,
                    delta_mode=pair_record_contact_delta_mode,
                    delta_normal=(
                        None
                        if pair_timing.get("delta_contact") is None
                        else pair_timing["delta_contact"].normal
                    ),
                    delta_normal_source=(
                        None
                        if pair_timing.get("delta_contact") is None
                        else getattr(
                            pair_timing["delta_contact"],
                            "normal_source",
                            None,
                        )
                    ),
                    angular_mode=pair_record_contact_angular_mode,
                    closing_only=pair_record_contact_closing_only,
                    max_velocity_delta=pair_record_contact_max_velocity_delta,
                    max_vertical_delta=pair_record_contact_max_vertical_delta,
                    vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                    max_speed=pair_record_contact_max_speed,
                    max_angular_delta=pair_record_contact_max_angular_delta,
                )
                post_contact_pos = (anchor[0], anchor[1], anchor[2])
                post_contact_vel = (vx, vy, vz)
                remaining_after_contact = max(
                    0.0,
                    float(pair_timing.get("remaining_time_s") or 0.0),
                )
                if applied and remaining_after_contact > 0.0:
                    endpoint_pos, endpoint_vel = motion_state_at(
                        post_contact_pos,
                        post_contact_vel,
                        timing_acceleration(),
                        remaining_after_contact,
                        remaining_after_contact,
                    )
            finally:
                anchor[0], anchor[1], anchor[2] = saved_pos
                vx, vy, vz = saved_vel
                contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = saved_ang
                ctx.spring_body_ang_vel = saved_body_ang_vel
                ctx.angular_vel_yaw = saved_yaw
                ctx.rigid_body_last_interp_tick = saved_last_interp_tick

            response_debug = dict(response_debug or {})
            result = {
                "pair_record_schedule_response_probe_enabled": True,
                "pair_record_schedule_response_probe_result": (
                    "applied" if applied else "not_applied"
                ),
                "pair_record_schedule_response_probe_applied": applied,
                "pair_record_schedule_response_probe_remaining_time_s": (
                    pair_timing.get("remaining_time_s")
                ),
                "pair_record_schedule_response_probe_contact_pos": (
                    pair_timing.get("pos")
                ),
                "pair_record_schedule_response_probe_vel_before": (
                    pair_timing.get("velocity")
                ),
                "pair_record_schedule_response_probe_post_contact_pos": (
                    post_contact_pos
                ),
                "pair_record_schedule_response_probe_post_contact_vel": (
                    post_contact_vel
                ),
                "pair_record_schedule_response_probe_endpoint_pos": endpoint_pos,
                "pair_record_schedule_response_probe_endpoint_vel": endpoint_vel,
            }
            response_fields = (
                "response",
                "target_separation",
                "constraint_selected_separation_speed_before",
                "point_normal_velocity_before",
                "raw_origin_fallback_safety_rejected",
                "raw_origin_fallback_delta_mode",
                "raw_origin_fallback_delta_normal",
                "raw_origin_fallback_delta_normal_source",
                "raw_origin_fallback_angular_mode",
                "raw_origin_fallback_vertical_delta_mode",
                "raw_origin_fallback_closing_only",
                "raw_origin_fallback_before_normal_speed",
                "raw_origin_fallback_before_normal_speed_source",
                "raw_origin_fallback_before_center_normal_speed",
                "raw_origin_fallback_normal_delta_skip_reason",
                "raw_origin_fallback_normal_delta_projected",
                "raw_origin_fallback_velocity_after_unclamped",
                "raw_origin_fallback_solver_velocity_delta_unprojected",
                "raw_origin_fallback_solver_velocity_delta_mag_unprojected",
                "raw_origin_fallback_velocity_delta_unclamped",
                "raw_origin_fallback_velocity_delta_mag_unclamped",
                "raw_origin_fallback_velocity_delta_clamped",
                "raw_origin_fallback_vertical_delta_clamped",
                "raw_origin_fallback_speed_clamped",
                "raw_origin_fallback_angular_velocity_after_unclamped",
                "raw_origin_fallback_solver_angular_delta_unprojected",
                "raw_origin_fallback_solver_angular_delta_mag_unprojected",
                "raw_origin_fallback_angular_delta_unclamped",
                "raw_origin_fallback_angular_delta_mag_unclamped",
                "raw_origin_fallback_angular_delta_clamped",
                "raw_origin_fallback_angular_preserved",
                "raw_origin_fallback_velocity_delta_after_safety",
                "raw_origin_fallback_velocity_delta_mag_after_safety",
                "raw_origin_fallback_angular_delta_after_safety",
                "raw_origin_fallback_angular_delta_mag_after_safety",
                "velocity_after",
                "angular_velocity_after",
                "angular_delta",
            )
            for key in response_fields:
                if key not in response_debug:
                    continue
                suffix = (
                    key[len("raw_origin_fallback_") :]
                    if key.startswith("raw_origin_fallback_")
                    else key
                )
                result[f"pair_record_schedule_response_probe_{suffix}"] = (
                    response_debug.get(key)
                )
            return result

        def estimate_deferred_prestep_pair_record_contact(current_reject, current_raw_error):
            if not (
                pair_record_deferred_prestep_enabled
                or pair_record_deferred_prestep_probe_enabled
            ):
                return {"contact": None, "reject": "disabled"}
            if current_raw_error is not None:
                return {"contact": None, "reject": "current_raw_origin_error"}
            if current_reject != "no_raw_origin_contact":
                return {
                    "contact": None,
                    "reject": "current_pair_contact_not_clear",
                }
            start_pos = finite_triplet(pre_pos)
            start_vel = finite_triplet(pre_vel)
            current_pos = finite_triplet((anchor[0], anchor[1], anchor[2]))
            if start_pos is None or start_vel is None or current_pos is None:
                return {"contact": None, "reject": "nonfinite_prestep_state"}
            distance = math.sqrt(
                (current_pos[0] - start_pos[0]) * (current_pos[0] - start_pos[0])
                + (current_pos[1] - start_pos[1]) * (current_pos[1] - start_pos[1])
                + (current_pos[2] - start_pos[2]) * (current_pos[2] - start_pos[2])
            )
            if (
                pair_record_deferred_prestep_max_distance > 0.0
                and distance > pair_record_deferred_prestep_max_distance
            ):
                return {
                    "contact": None,
                    "reject": "prestep_pair_record_too_far",
                    "distance": distance,
                }
            pair_probe = sample_pair_record_contact_at(start_pos, velocity=start_vel)
            pair_contact = pair_probe.get("contact")
            reject = pair_probe.get("reject")
            if (
                pair_probe.get("selected_raw_error") is not None
                or pair_contact is None
                or reject != ""
                or pair_contact.penetration <= self._PENETRATION_SLOP_DEFAULT
            ):
                return {
                    "contact": None,
                    "reject": (
                        f"prestep_pair_record_{reject}"
                        if reject
                        else "no_prestep_pair_record_contact"
                    ),
                    "distance": distance,
                    "probe": pair_probe,
                }
            return {
                "contact": pair_contact,
                "delta_contact": pair_probe.get("delta_contact"),
                "pos": start_pos,
                "velocity": start_vel,
                "distance": distance,
                "probe": pair_probe,
                "reject": "",
            }

        phase_lookahead_queue_attr = "terrain_pair_record_phase_lookahead_queue"

        def estimate_phase_lookahead_pair_record_contact(current_reject, current_raw_error):
            if not pair_record_phase_lookahead_enabled:
                return {"contact": None, "reject": "disabled"}
            if current_raw_error is not None:
                return {"contact": None, "reject": "current_raw_origin_error"}
            if current_reject != "no_raw_origin_contact":
                return {
                    "contact": None,
                    "reject": "current_pair_contact_not_clear",
                }
            current_pos = finite_triplet((anchor[0], anchor[1], anchor[2]))
            current_vel = finite_triplet((vx, vy, vz))
            if current_pos is None or current_vel is None:
                return {"contact": None, "reject": "nonfinite_current_state"}
            speed_sq = (
                current_vel[0] * current_vel[0]
                + current_vel[1] * current_vel[1]
                + current_vel[2] * current_vel[2]
            )
            if speed_sq <= 1e-8:
                return {"contact": None, "reject": "stationary_current_velocity"}
            max_time = max(0.0, float(pair_record_phase_lookahead_max_time))
            if max_time <= 0.0:
                return {"contact": None, "reject": "zero_lookahead_window"}
            if pair_record_phase_lookahead_accel_mode in {
                "frame",
                "frame_accel",
                "frame_acceleration",
                "accel",
                "acceleration",
                "decompile",
            }:
                acc = timing_acceleration()
                acceleration_source = "frame_acceleration"
            else:
                acc = (0.0, 0.0, 0.0)
                acceleration_source = "constant_velocity"

            def state_at(elapsed_s):
                return motion_state_at(
                    current_pos,
                    current_vel,
                    acc,
                    elapsed_s,
                    max_time,
                )

            def distance_from_current(pos):
                return math.sqrt(
                    (pos[0] - current_pos[0]) * (pos[0] - current_pos[0])
                    + (pos[1] - current_pos[1]) * (pos[1] - current_pos[1])
                    + (pos[2] - current_pos[2]) * (pos[2] - current_pos[2])
                )

            def candidate_at(elapsed_s):
                pos, velocity = state_at(elapsed_s)
                distance = distance_from_current(pos)
                if (
                    pair_record_phase_lookahead_max_distance > 0.0
                    and distance > pair_record_phase_lookahead_max_distance
                ):
                    return {
                        "contact": None,
                        "reject": "phase_lookahead_too_far",
                        "elapsed_s": elapsed_s,
                        "distance": distance,
                    }
                pair_probe = sample_pair_record_contact_at(pos, velocity=velocity)
                pair_contact = pair_probe.get("contact")
                reject = pair_probe.get("reject")
                if (
                    pair_probe.get("selected_raw_error") is None
                    and pair_contact is not None
                    and reject == ""
                    and pair_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                ):
                    return {
                        "pos": pos,
                        "velocity": velocity,
                        "distance": distance,
                        "probe": pair_probe,
                        "contact": pair_contact,
                        "delta_contact": pair_probe.get("delta_contact"),
                        "reject": "",
                    }
                return {
                    "contact": None,
                    "reject": (
                        f"phase_lookahead_{reject}"
                        if reject
                        else "phase_lookahead_no_pair_record_contact"
                    ),
                    "elapsed_s": elapsed_s,
                    "distance": distance,
                    "probe": pair_probe,
                }

            scan_steps = max(1, int(pair_record_phase_lookahead_steps))
            previous_clear_time = 0.0
            previous_clear = candidate_at(0.0)
            found_time = None
            found = None
            scan_clear_count = 0
            scan_contact_count = 0
            last_reject = previous_clear.get("reject")
            for scan_index in range(1, scan_steps + 1):
                scan_time = max_time * (float(scan_index) / float(scan_steps))
                candidate = candidate_at(scan_time)
                if candidate.get("contact") is not None:
                    found_time = scan_time
                    found = candidate
                    scan_contact_count += 1
                    break
                previous_clear_time = scan_time
                scan_clear_count += 1
                last_reject = candidate.get("reject")
            if found_time is None or found is None:
                return {
                    "contact": None,
                    "reject": last_reject or "phase_lookahead_no_pair_record_contact",
                    "scan_steps": scan_steps,
                    "scan_clear_count": scan_clear_count,
                    "scan_contact_count": scan_contact_count,
                    "acceleration_source": acceleration_source,
                }

            lo = previous_clear_time
            hi = found_time
            best = found
            iterations = 0
            while hi - lo > 0.0025 and iterations < 24:
                mid = (lo + hi) * 0.5
                candidate = candidate_at(mid)
                iterations += 1
                if candidate.get("contact") is not None:
                    hi = mid
                    best = candidate
                    scan_contact_count += 1
                else:
                    lo = mid
                    scan_clear_count += 1
                    last_reject = candidate.get("reject")

            return {
                "contact": best["contact"],
                "delta_contact": best.get("delta_contact"),
                "pos": best["pos"],
                "velocity": best["velocity"],
                "distance": best.get("distance"),
                "collision_time_s": hi,
                "remaining_time_s": max(0.0, max_time - hi),
                "scan_steps": scan_steps,
                "scan_clear_count": scan_clear_count,
                "scan_contact_count": scan_contact_count,
                "sweep_iterations": iterations,
                "hit_time_s": found_time,
                "last_clear_time_s": previous_clear_time,
                "acceleration_source": acceleration_source,
                "probe": best.get("probe"),
                "reject": "",
            }

        def apply_queued_phase_lookahead_pair_record_contact(queue_record):
            nonlocal vx, vy, vz
            if not pair_record_phase_lookahead_queue_enabled:
                return False, {"reject": "phase_lookahead_queue_disabled"}
            if not isinstance(queue_record, dict):
                return False, {"reject": "phase_lookahead_queue_empty"}
            contact = queue_record.get("contact")
            if contact is None:
                setattr(ctx, phase_lookahead_queue_attr, None)
                return False, {"reject": "phase_lookahead_queue_missing_contact"}
            try:
                time_to_contact = float(queue_record.get("time_to_contact_s") or 0.0)
            except (TypeError, ValueError, OverflowError):
                time_to_contact = 0.0
            frame_dt = max(0.0, float(dt or 0.0))
            if frame_dt <= 0.0:
                setattr(ctx, phase_lookahead_queue_attr, None)
                return False, {"reject": "phase_lookahead_queue_zero_dt"}
            if time_to_contact > frame_dt:
                queue_record = dict(queue_record)
                queue_record["time_to_contact_s"] = max(0.0, time_to_contact - frame_dt)
                queue_record["age_steps"] = int(queue_record.get("age_steps") or 0) + 1
                setattr(ctx, phase_lookahead_queue_attr, queue_record)
                return False, {
                    "queued": True,
                    "pending": True,
                    "time_to_contact_s": queue_record["time_to_contact_s"],
                    "age_steps": queue_record["age_steps"],
                    "collision_time_s": queue_record.get("collision_time_s"),
                    "distance": queue_record.get("distance"),
                }

            start_pos = finite_triplet(pre_pos) or finite_triplet(
                (anchor[0], anchor[1], anchor[2])
            )
            start_vel = finite_triplet(pre_vel) or finite_triplet((vx, vy, vz))
            if start_pos is None or start_vel is None:
                setattr(ctx, phase_lookahead_queue_attr, None)
                return False, {"reject": "phase_lookahead_queue_nonfinite_start"}

            collision_time = max(0.0, min(frame_dt, time_to_contact))
            contact_pos, contact_vel = motion_state_at(
                start_pos,
                start_vel,
                timing_acceleration(),
                collision_time,
                frame_dt,
            )
            endpoint_pos = (anchor[0], anchor[1], anchor[2])
            endpoint_vel = (vx, vy, vz)
            endpoint_ang = tuple(contact_angular_velocity)
            saved_body_ang_vel = getattr(ctx, "spring_body_ang_vel", None)
            saved_yaw = getattr(ctx, "angular_vel_yaw", None)
            try:
                anchor[0], anchor[1], anchor[2] = contact_pos
                vx, vy, vz = contact_vel
                delta_contact = queue_record.get("delta_contact")
                response_debug, applied = apply_raw_origin_fallback_contact(
                    contact,
                    projection_order=pair_record_contact_projection_order,
                    delta_mode=pair_record_contact_delta_mode,
                    delta_normal=(
                        None if delta_contact is None else delta_contact.normal
                    ),
                    delta_normal_source=(
                        None
                        if delta_contact is None
                        else getattr(delta_contact, "normal_source", None)
                    ),
                    angular_mode=pair_record_contact_angular_mode,
                    closing_only=pair_record_contact_closing_only,
                    max_velocity_delta=pair_record_contact_max_velocity_delta,
                    max_vertical_delta=pair_record_contact_max_vertical_delta,
                    vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                    max_speed=pair_record_contact_max_speed,
                    max_angular_delta=pair_record_contact_max_angular_delta,
                )
                if not applied:
                    anchor[0], anchor[1], anchor[2] = endpoint_pos
                    vx, vy, vz = endpoint_vel
                    contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = endpoint_ang
                    ctx.spring_body_ang_vel = saved_body_ang_vel
                    ctx.angular_vel_yaw = saved_yaw
                    setattr(ctx, phase_lookahead_queue_attr, None)
                    return False, {
                        "reject": "phase_lookahead_queue_response_not_applied",
                        **dict(response_debug or {}),
                    }

                post_contact_pos = (anchor[0], anchor[1], anchor[2])
                post_contact_vel = (vx, vy, vz)
                remaining_after_contact = max(0.0, frame_dt - collision_time)
                if remaining_after_contact > 0.0:
                    final_pos, final_vel = motion_state_at(
                        post_contact_pos,
                        post_contact_vel,
                        timing_acceleration(),
                        remaining_after_contact,
                        remaining_after_contact,
                    )
                    anchor[0], anchor[1], anchor[2] = final_pos
                    vx, vy, vz = final_vel
                response_debug = dict(response_debug or {})
                response_debug.update({
                    "phase_lookahead_queued_pair_record_contact": True,
                    "pair_record_phase_lookahead_queue_enabled": True,
                    "pair_record_phase_lookahead_queued_collision_time_s": (
                        collision_time
                    ),
                    "pair_record_phase_lookahead_queued_remaining_time_s": (
                        remaining_after_contact
                    ),
                    "pair_record_phase_lookahead_queued_original_collision_time_s": (
                        queue_record.get("collision_time_s")
                    ),
                    "pair_record_phase_lookahead_queued_age_steps": (
                        queue_record.get("age_steps")
                    ),
                    "pair_record_phase_lookahead_queued_distance": (
                        queue_record.get("distance")
                    ),
                    "pair_record_phase_lookahead_queued_contact_pos": contact_pos,
                    "pair_record_phase_lookahead_queued_contact_vel_before": (
                        contact_vel
                    ),
                    "pair_record_phase_lookahead_queued_post_contact_pos": (
                        post_contact_pos
                    ),
                    "pair_record_phase_lookahead_queued_post_contact_vel": (
                        post_contact_vel
                    ),
                    "pair_record_phase_lookahead_queued_endpoint_pos": endpoint_pos,
                    "pair_record_phase_lookahead_queued_endpoint_vel": endpoint_vel,
                    "velocity_after": (vx, vy, vz),
                })
                setattr(ctx, phase_lookahead_queue_attr, None)
                return True, response_debug
            except Exception:
                anchor[0], anchor[1], anchor[2] = endpoint_pos
                vx, vy, vz = endpoint_vel
                contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = endpoint_ang
                ctx.spring_body_ang_vel = saved_body_ang_vel
                ctx.angular_vel_yaw = saved_yaw
                setattr(ctx, phase_lookahead_queue_attr, None)
                raise

        def estimate_phase_backtrack_pair_record_contact(current_reject, current_raw_error):
            if not pair_record_phase_backtrack_enabled:
                return {"contact": None, "reject": "disabled"}
            if current_raw_error is not None:
                return {"contact": None, "reject": "current_raw_origin_error"}
            if current_reject != "no_raw_origin_contact":
                return {
                    "contact": None,
                    "reject": "current_pair_contact_not_clear",
                }

            source = pair_record_phase_backtrack_source
            if source in {"current", "endpoint", "post", "poststep"}:
                base_source = "current"
                base_pos = finite_triplet((anchor[0], anchor[1], anchor[2]))
                base_vel = finite_triplet((vx, vy, vz))
            elif source in {"ref", "reference", "dirty_reference", "world_ref"}:
                base_source = "dirty_reference"
                base_pos = finite_triplet(reference_pos)
                base_vel = finite_triplet(pre_vel) or finite_triplet((vx, vy, vz))
            else:
                base_source = "pre"
                base_pos = finite_triplet(pre_pos)
                base_vel = finite_triplet(pre_vel)
                if base_pos is None or base_vel is None:
                    base_source = "current"
                    base_pos = finite_triplet((anchor[0], anchor[1], anchor[2]))
                    base_vel = finite_triplet((vx, vy, vz))
            if base_pos is None or base_vel is None:
                return {
                    "contact": None,
                    "reject": "nonfinite_backtrack_state",
                    "source": base_source,
                }
            speed_sq = (
                base_vel[0] * base_vel[0]
                + base_vel[1] * base_vel[1]
                + base_vel[2] * base_vel[2]
            )
            if speed_sq <= 1e-8:
                return {
                    "contact": None,
                    "reject": "stationary_backtrack_velocity",
                    "source": base_source,
                }
            max_time = max(0.0, float(pair_record_phase_backtrack_max_time))
            if max_time <= 0.0:
                return {
                    "contact": None,
                    "reject": "zero_backtrack_window",
                    "source": base_source,
                }
            if pair_record_phase_backtrack_accel_mode in {
                "frame",
                "frame_accel",
                "frame_acceleration",
                "accel",
                "acceleration",
                "decompile",
            }:
                acc = timing_acceleration()
                acceleration_source = "frame_acceleration"
            else:
                acc = (0.0, 0.0, 0.0)
                acceleration_source = "constant_velocity"

            def state_at(elapsed_s):
                elapsed_s = max(0.0, float(elapsed_s))
                return (
                    (
                        base_pos[0] - base_vel[0] * elapsed_s + 0.5 * acc[0] * elapsed_s * elapsed_s,
                        base_pos[1] - base_vel[1] * elapsed_s + 0.5 * acc[1] * elapsed_s * elapsed_s,
                        base_pos[2] - base_vel[2] * elapsed_s + 0.5 * acc[2] * elapsed_s * elapsed_s,
                    ),
                    (
                        base_vel[0] - acc[0] * elapsed_s,
                        base_vel[1] - acc[1] * elapsed_s,
                        base_vel[2] - acc[2] * elapsed_s,
                    ),
                )

            def distance_from_base(pos):
                return math.sqrt(
                    (pos[0] - base_pos[0]) * (pos[0] - base_pos[0])
                    + (pos[1] - base_pos[1]) * (pos[1] - base_pos[1])
                    + (pos[2] - base_pos[2]) * (pos[2] - base_pos[2])
                )

            def candidate_at(elapsed_s):
                pos, velocity = state_at(elapsed_s)
                distance = distance_from_base(pos)
                if (
                    pair_record_phase_backtrack_max_distance > 0.0
                    and distance > pair_record_phase_backtrack_max_distance
                ):
                    return {
                        "contact": None,
                        "reject": "phase_backtrack_too_far",
                        "elapsed_s": elapsed_s,
                        "distance": distance,
                    }
                pair_probe = sample_pair_record_contact_at(pos, velocity=velocity)
                pair_contact = pair_probe.get("contact")
                reject = pair_probe.get("reject")
                if (
                    pair_probe.get("selected_raw_error") is None
                    and pair_contact is not None
                    and reject == ""
                    and pair_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                ):
                    return {
                        "pos": pos,
                        "velocity": velocity,
                        "distance": distance,
                        "probe": pair_probe,
                        "contact": pair_contact,
                        "delta_contact": pair_probe.get("delta_contact"),
                        "reject": "",
                    }
                return {
                    "contact": None,
                    "reject": (
                        f"phase_backtrack_{reject}"
                        if reject
                        else "phase_backtrack_no_pair_record_contact"
                    ),
                    "elapsed_s": elapsed_s,
                    "distance": distance,
                    "probe": pair_probe,
                }

            scan_steps = max(1, int(pair_record_phase_backtrack_steps))
            previous_clear_time = 0.0
            previous_clear = candidate_at(0.0)
            if previous_clear.get("contact") is not None:
                return {
                    "contact": previous_clear["contact"],
                    "delta_contact": previous_clear.get("delta_contact"),
                    "pos": previous_clear["pos"],
                    "velocity": previous_clear["velocity"],
                    "distance": previous_clear.get("distance"),
                    "backtrack_time_s": 0.0,
                    "scan_steps": scan_steps,
                    "scan_clear_count": 0,
                    "scan_contact_count": 1,
                    "sweep_iterations": 0,
                    "source": base_source,
                    "base_pos": base_pos,
                    "base_vel": base_vel,
                    "acceleration_source": acceleration_source,
                    "probe": previous_clear.get("probe"),
                    "reject": "",
                }

            found_time = None
            found = None
            scan_clear_count = 1
            scan_contact_count = 0
            last_reject = previous_clear.get("reject")
            for scan_index in range(1, scan_steps + 1):
                scan_time = max_time * (float(scan_index) / float(scan_steps))
                candidate = candidate_at(scan_time)
                if candidate.get("contact") is not None:
                    found_time = scan_time
                    found = candidate
                    scan_contact_count += 1
                    break
                previous_clear_time = scan_time
                scan_clear_count += 1
                last_reject = candidate.get("reject")
            if found_time is None or found is None:
                return {
                    "contact": None,
                    "reject": last_reject or "phase_backtrack_no_pair_record_contact",
                    "scan_steps": scan_steps,
                    "scan_clear_count": scan_clear_count,
                    "scan_contact_count": scan_contact_count,
                    "source": base_source,
                    "base_pos": base_pos,
                    "base_vel": base_vel,
                    "acceleration_source": acceleration_source,
                }

            lo = previous_clear_time
            hi = found_time
            best = found
            iterations = 0
            while hi - lo > 0.0025 and iterations < 24:
                mid = (lo + hi) * 0.5
                candidate = candidate_at(mid)
                iterations += 1
                if candidate.get("contact") is not None:
                    hi = mid
                    best = candidate
                    scan_contact_count += 1
                else:
                    lo = mid
                    scan_clear_count += 1
                    last_reject = candidate.get("reject")

            return {
                "contact": best["contact"],
                "delta_contact": best.get("delta_contact"),
                "pos": best["pos"],
                "velocity": best["velocity"],
                "distance": best.get("distance"),
                "backtrack_time_s": hi,
                "hit_time_s": found_time,
                "last_clear_time_s": previous_clear_time,
                "scan_steps": scan_steps,
                "scan_clear_count": scan_clear_count,
                "scan_contact_count": scan_contact_count,
                "sweep_iterations": iterations,
                "source": base_source,
                "base_pos": base_pos,
                "base_vel": base_vel,
                "acceleration_source": acceleration_source,
                "probe": best.get("probe"),
                "reject": "",
            }

        def estimate_timed_contact_from(start_pos, start_vel, acc, remaining_time):
            if not timed_pair_response:
                return None
            remaining_time = max(0.0, float(remaining_time))
            if remaining_time <= 0.0:
                return None

            def timed_contact_candidate_at(pos, velocity):
                lifted_contact = sample_contact_at(pos)
                if (
                    lifted_contact is not None
                    and lifted_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                ):
                    return {
                        "contact": lifted_contact,
                        "raw_origin_fallback": False,
                    }
                pair_probe = None
                if pair_record_timed_contact_enabled:
                    pair_probe = sample_pair_record_contact_at(pos, velocity=velocity)
                    pair_contact = pair_probe.get("contact")
                    pair_delta_contact = pair_probe.get("delta_contact")
                    if (
                        pair_probe.get("selected_raw_error") is None
                        and pair_contact is not None
                        and pair_probe.get("reject") == ""
                        and pair_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                    ):
                        return {
                            "contact": pair_contact,
                            "raw_origin_fallback": False,
                            "pair_record_contact": True,
                            "pair_record_timed_contact": True,
                            "pair_record_contact_reject": "",
                            "pair_record_contact_reason": (
                                "timed_lifted_clear_pair_record_contact"
                                if lifted_contact is None
                                else "timed_lifted_below_slop_pair_record_contact"
                            ),
                            "pair_record_contact_selection": pair_record_contact_selection,
                            "pair_record_delta_normal": (
                                None
                                if pair_delta_contact is None
                                else pair_delta_contact.normal
                            ),
                            "pair_record_delta_normal_source": (
                                None
                                if pair_delta_contact is None
                                else getattr(pair_delta_contact, "normal_source", None)
                            ),
                            "pair_record_raw_normal": getattr(
                                pair_probe.get("raw_contact"),
                                "normal",
                                None,
                            ),
                            "pair_record_selected_raw_normal": getattr(
                                pair_probe.get("selected_raw_contact"),
                                "normal",
                                None,
                            ),
                        }
                if not raw_fallback_timed_enabled:
                    if raycast_fallback_timed_enabled:
                        ray_probe = sample_raycast_fallback_contact_at(
                            pos,
                            velocity=velocity,
                            reference=reference_pos,
                        )
                        ray_contact = ray_probe.get("contact")
                        if (
                            ray_contact is not None
                            and ray_probe.get("reject") == ""
                            and ray_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                        ):
                            return {
                                "contact": ray_contact,
                                "raw_origin_fallback": False,
                                "raycast_fallback": True,
                                "raycast_fallback_reject": "",
                                "raycast_fallback_probe_reason": (
                                    "timed_lifted_clear_raycast_contact"
                                    if lifted_contact is None
                                    else "timed_lifted_below_slop_raycast_contact"
                                ),
                            }
                    return {
                        "contact": lifted_contact,
                        "raw_origin_fallback": False,
                        "pair_record_contact_reject": (
                            None if pair_probe is None else pair_probe.get("reject")
                        ),
                        "pair_record_contact_reason": (
                            None
                            if pair_probe is None
                            else (
                                "timed_lifted_clear_pair_record_rejected"
                                if lifted_contact is None
                                else "timed_lifted_below_slop_pair_record_rejected"
                            )
                        ),
                    }
                raw_probe = sample_raw_origin_fallback_contact_at(pos, velocity=velocity)
                raw_contact = raw_probe.get("contact")
                if (
                    raw_probe.get("raw_error") is None
                    and raw_contact is not None
                    and raw_probe.get("reject") == ""
                    and raw_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                ):
                    return {
                        "contact": raw_contact,
                        "raw_origin_fallback": True,
                        "raw_origin_fallback_reject": "",
                        "raw_origin_fallback_probe_reason": (
                            "timed_lifted_clear_raw_origin_contact"
                            if lifted_contact is None
                            else "timed_lifted_below_slop_raw_origin_contact"
                        ),
                    }
                ray_probe = sample_raycast_fallback_contact_at(
                    pos,
                    velocity=velocity,
                    reference=reference_pos,
                )
                ray_contact = ray_probe.get("contact")
                if (
                    raycast_fallback_timed_enabled
                    and ray_contact is not None
                    and ray_probe.get("reject") == ""
                    and ray_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                ):
                    return {
                        "contact": ray_contact,
                        "raw_origin_fallback": False,
                        "raycast_fallback": True,
                        "raycast_fallback_reject": "",
                        "raycast_fallback_probe_reason": (
                            "timed_lifted_clear_raycast_contact"
                            if lifted_contact is None
                            else "timed_lifted_below_slop_raycast_contact"
                        ),
                    }
                return {
                    "contact": lifted_contact,
                    "raw_origin_fallback": False,
                    "raw_origin_fallback_reject": raw_probe.get("reject"),
                    "raw_origin_fallback_probe_reason": (
                        "timed_lifted_clear_raw_origin_rejected"
                        if lifted_contact is None
                        else "timed_lifted_below_slop_raw_origin_rejected"
                    ),
                    "pair_record_contact_reject": (
                        None if pair_probe is None else pair_probe.get("reject")
                    ),
                    "pair_record_contact_reason": (
                        None
                        if pair_probe is None
                        else (
                            "timed_lifted_clear_pair_record_rejected"
                            if lifted_contact is None
                            else "timed_lifted_below_slop_pair_record_rejected"
                        )
                    ),
                    "raycast_fallback_reject": ray_probe.get("reject"),
                }

            start_candidate = timed_contact_candidate_at(start_pos, start_vel)
            start_contact = start_candidate.get("contact")
            if (
                start_contact is not None
                and start_contact.penetration > self._PENETRATION_SLOP_DEFAULT
            ):
                collision_time = min(remaining_time, 0.005) if start_time_clamp_enabled else 0.0
                contact = start_contact
                contact_candidate = start_candidate
                if collision_time > 0.0:
                    contact_pos, _contact_vel = motion_state_at(
                        start_pos,
                        start_vel,
                        acc,
                        collision_time,
                        remaining_time,
                    )
                    contact_candidate_at_time = timed_contact_candidate_at(
                        contact_pos,
                        _contact_vel,
                    )
                    contact_at_time = contact_candidate_at_time.get("contact")
                    if (
                        contact_at_time is not None
                        and contact_at_time.penetration > self._PENETRATION_SLOP_DEFAULT
                    ):
                        contact = contact_at_time
                        contact_candidate = contact_candidate_at_time
                return {
                    "collision_time_s": collision_time,
                    "contact": contact,
                    "sweep_iterations": 0,
                    "sweep_clear_count": 0,
                    "sweep_contact_count": 2 if contact is not start_contact else 1,
                    "collision_at_start": True,
                    "start_time_clamped": collision_time > 0.0,
                    "raw_origin_fallback": bool(
                        contact_candidate.get("raw_origin_fallback")
                    ),
                    "raw_origin_fallback_reject": contact_candidate.get(
                        "raw_origin_fallback_reject"
                    ),
                    "raw_origin_fallback_probe_reason": contact_candidate.get(
                        "raw_origin_fallback_probe_reason"
                    ),
                    "raycast_fallback": bool(
                        contact_candidate.get("raycast_fallback")
                    ),
                    "raycast_fallback_reject": contact_candidate.get(
                        "raycast_fallback_reject"
                    ),
                    "raycast_fallback_probe_reason": contact_candidate.get(
                        "raycast_fallback_probe_reason"
                    ),
                    "pair_record_contact": bool(
                        contact_candidate.get("pair_record_contact")
                    ),
                    "pair_record_timed_contact": bool(
                        contact_candidate.get("pair_record_timed_contact")
                    ),
                    "pair_record_contact_reject": contact_candidate.get(
                        "pair_record_contact_reject"
                    ),
                    "pair_record_contact_reason": contact_candidate.get(
                        "pair_record_contact_reason"
                    ),
                    "pair_record_contact_selection": contact_candidate.get(
                        "pair_record_contact_selection"
                    ),
                    "pair_record_delta_normal": contact_candidate.get(
                        "pair_record_delta_normal"
                    ),
                    "pair_record_delta_normal_source": contact_candidate.get(
                        "pair_record_delta_normal_source"
                    ),
                    "pair_record_raw_normal": contact_candidate.get(
                        "pair_record_raw_normal"
                    ),
                    "pair_record_selected_raw_normal": contact_candidate.get(
                        "pair_record_selected_raw_normal"
                    ),
                }

            end_pos, end_vel = motion_state_at(start_pos, start_vel, acc, remaining_time, remaining_time)
            end_candidate = timed_contact_candidate_at(end_pos, end_vel)
            end_contact = end_candidate.get("contact")
            if (
                end_contact is None
                or end_contact.penetration <= self._PENETRATION_SLOP_DEFAULT
            ):
                if not contact_sweep_scan_enabled:
                    return None
                prev_time = 0.0
                found_time = None
                found_candidate = None
                for scan_index in range(1, contact_sweep_scan_steps + 1):
                    scan_time = remaining_time * (
                        float(scan_index) / float(contact_sweep_scan_steps + 1)
                    )
                    scan_pos, scan_vel = motion_state_at(
                        start_pos,
                        start_vel,
                        acc,
                        scan_time,
                        remaining_time,
                    )
                    scan_candidate = timed_contact_candidate_at(scan_pos, scan_vel)
                    scan_contact = scan_candidate.get("contact")
                    if (
                        scan_contact is not None
                        and scan_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                    ):
                        found_time = scan_time
                        found_candidate = scan_candidate
                        break
                    prev_time = scan_time
                if found_time is None or found_candidate is None:
                    return None

                lo = prev_time
                hi = found_time
                contact = found_candidate["contact"]
                contact_candidate = found_candidate
                iterations = 0
                clear_count = max(0, int(round(prev_time > 0.0)))
                contact_count = 1
                while hi - lo > 0.0025 and iterations < 24:
                    mid = (lo + hi) * 0.5
                    mid_pos, mid_vel = motion_state_at(
                        start_pos,
                        start_vel,
                        acc,
                        mid,
                        remaining_time,
                    )
                    mid_candidate = timed_contact_candidate_at(mid_pos, mid_vel)
                    mid_contact = mid_candidate.get("contact")
                    iterations += 1
                    if (
                        mid_contact is not None
                        and mid_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                    ):
                        hi = mid
                        contact = mid_contact
                        contact_candidate = mid_candidate
                        contact_count += 1
                    else:
                        lo = mid
                        clear_count += 1

                return {
                    "collision_time_s": hi,
                    "contact": contact,
                    "sweep_iterations": iterations,
                    "sweep_clear_count": clear_count,
                    "sweep_contact_count": contact_count,
                    "collision_at_start": False,
                    "contact_sweep_scan": True,
                    "contact_sweep_scan_steps": contact_sweep_scan_steps,
                    "contact_sweep_scan_hit_time_s": found_time,
                    "raw_origin_fallback": bool(
                        contact_candidate.get("raw_origin_fallback")
                    ),
                    "raw_origin_fallback_reject": contact_candidate.get(
                        "raw_origin_fallback_reject"
                    ),
                    "raw_origin_fallback_probe_reason": contact_candidate.get(
                        "raw_origin_fallback_probe_reason"
                    ),
                    "raycast_fallback": bool(
                        contact_candidate.get("raycast_fallback")
                    ),
                    "raycast_fallback_reject": contact_candidate.get(
                        "raycast_fallback_reject"
                    ),
                    "raycast_fallback_probe_reason": contact_candidate.get(
                        "raycast_fallback_probe_reason"
                    ),
                    "pair_record_contact": bool(
                        contact_candidate.get("pair_record_contact")
                    ),
                    "pair_record_timed_contact": bool(
                        contact_candidate.get("pair_record_timed_contact")
                    ),
                    "pair_record_contact_reject": contact_candidate.get(
                        "pair_record_contact_reject"
                    ),
                    "pair_record_contact_reason": contact_candidate.get(
                        "pair_record_contact_reason"
                    ),
                    "pair_record_contact_selection": contact_candidate.get(
                        "pair_record_contact_selection"
                    ),
                    "pair_record_delta_normal": contact_candidate.get(
                        "pair_record_delta_normal"
                    ),
                    "pair_record_delta_normal_source": contact_candidate.get(
                        "pair_record_delta_normal_source"
                    ),
                    "pair_record_raw_normal": contact_candidate.get(
                        "pair_record_raw_normal"
                    ),
                    "pair_record_selected_raw_normal": contact_candidate.get(
                        "pair_record_selected_raw_normal"
                    ),
                }

            lo = 0.0
            hi = remaining_time
            contact = end_contact
            contact_candidate = end_candidate
            iterations = 0
            clear_count = 0
            contact_count = 1
            while hi - lo > 0.0025 and iterations < 24:
                mid = (lo + hi) * 0.5
                mid_pos, mid_vel = motion_state_at(start_pos, start_vel, acc, mid, remaining_time)
                mid_candidate = timed_contact_candidate_at(mid_pos, mid_vel)
                mid_contact = mid_candidate.get("contact")
                iterations += 1
                if (
                    mid_contact is not None
                    and mid_contact.penetration > self._PENETRATION_SLOP_DEFAULT
                ):
                    hi = mid
                    contact = mid_contact
                    contact_candidate = mid_candidate
                    contact_count += 1
                else:
                    lo = mid
                    clear_count += 1

            return {
                "collision_time_s": hi,
                "contact": contact,
                "sweep_iterations": iterations,
                "sweep_clear_count": clear_count,
                "sweep_contact_count": contact_count,
                "collision_at_start": False,
                "contact_sweep_scan": False,
                "contact_sweep_scan_steps": contact_sweep_scan_steps,
                "raw_origin_fallback": bool(
                    contact_candidate.get("raw_origin_fallback")
                ),
                "raw_origin_fallback_reject": contact_candidate.get(
                    "raw_origin_fallback_reject"
                ),
                "raw_origin_fallback_probe_reason": contact_candidate.get(
                    "raw_origin_fallback_probe_reason"
                ),
                "raycast_fallback": bool(
                    contact_candidate.get("raycast_fallback")
                ),
                "raycast_fallback_reject": contact_candidate.get(
                    "raycast_fallback_reject"
                ),
                "raycast_fallback_probe_reason": contact_candidate.get(
                    "raycast_fallback_probe_reason"
                ),
                "pair_record_contact": bool(
                    contact_candidate.get("pair_record_contact")
                ),
                "pair_record_timed_contact": bool(
                    contact_candidate.get("pair_record_timed_contact")
                ),
                "pair_record_contact_reject": contact_candidate.get(
                    "pair_record_contact_reject"
                ),
                "pair_record_contact_reason": contact_candidate.get(
                    "pair_record_contact_reason"
                ),
                "pair_record_contact_selection": contact_candidate.get(
                    "pair_record_contact_selection"
                ),
                "pair_record_delta_normal": contact_candidate.get(
                    "pair_record_delta_normal"
                ),
                "pair_record_delta_normal_source": contact_candidate.get(
                    "pair_record_delta_normal_source"
                ),
                "pair_record_raw_normal": contact_candidate.get(
                    "pair_record_raw_normal"
                ),
                "pair_record_selected_raw_normal": contact_candidate.get(
                    "pair_record_selected_raw_normal"
                ),
            }

        def resolve_timed_pair_contact():
            nonlocal vx, vy, vz
            frame_dt = float(dt)
            acc = timing_acceleration()
            elapsed = 0.0
            current_pos = tuple(pre_pos)
            current_vel = tuple(pre_vel)
            contact_events = []
            response_debug = None
            contact = None
            contact_pos = None
            contact_vel = None

            for iteration_index in range(contact_iteration_limit):
                remaining = max(0.0, frame_dt - elapsed)
                if remaining <= 0.0:
                    break

                timed_contact = estimate_timed_contact_from(current_pos, current_vel, acc, remaining)
                if timed_contact is None:
                    final_pos, final_vel = motion_state_at(
                        current_pos,
                        current_vel,
                        acc,
                        remaining,
                        remaining,
                    )
                    current_pos = final_pos
                    current_vel = final_vel
                    elapsed = frame_dt
                    break

                collision_time = timed_contact["collision_time_s"]
                contact = timed_contact["contact"]
                contact_pos, contact_vel = motion_state_at(
                    current_pos,
                    current_vel,
                    acc,
                    collision_time,
                    remaining,
                )
                anchor[0], anchor[1], anchor[2] = contact_pos
                vx, vy, vz = contact_vel
                if timed_contact["collision_at_start"] and start_iterative_enabled:
                    response_debug = apply_iterative_start_contact(contact)
                elif timed_contact.get("pair_record_contact"):
                    response_debug, applied_pair_record_contact = (
                        apply_raw_origin_fallback_contact(
                            contact,
                            projection_order=pair_record_contact_projection_order,
                            delta_mode=pair_record_contact_delta_mode,
                            delta_normal=timed_contact.get("pair_record_delta_normal"),
                            delta_normal_source=timed_contact.get(
                                "pair_record_delta_normal_source"
                            ),
                            angular_mode=pair_record_contact_angular_mode,
                            closing_only=pair_record_contact_closing_only,
                            max_velocity_delta=pair_record_contact_max_velocity_delta,
                            max_vertical_delta=pair_record_contact_max_vertical_delta,
                            vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                            max_speed=pair_record_contact_max_speed,
                            max_angular_delta=pair_record_contact_max_angular_delta,
                        )
                    )
                    if not applied_pair_record_contact:
                        return False
                    if response_debug is not None:
                        response_debug.update({
                            "pair_record_contact": True,
                            "pair_record_timed_contact": True,
                            "pair_record_contact_reason": timed_contact.get(
                                "pair_record_contact_reason"
                            ),
                            "pair_record_contact_reject": timed_contact.get(
                                "pair_record_contact_reject"
                            ),
                            "pair_record_contact_enabled": pair_record_contact_enabled,
                            "pair_record_contact_response_profile": (
                                pair_record_contact_response_profile
                            ),
                            "pair_record_contact_selection": pair_record_contact_selection,
                            "pair_record_contact_normal_source": pair_record_contact_normal_source,
                            "pair_record_contact_delta_normal_source": pair_record_contact_delta_normal_source,
                            "pair_record_delta_normal": timed_contact.get(
                                "pair_record_delta_normal"
                            ),
                            "pair_record_delta_normal_source": timed_contact.get(
                                "pair_record_delta_normal_source"
                            ),
                            "pair_record_contact_projection_order": pair_record_contact_projection_order,
                            "pair_record_contact_delta_mode": pair_record_contact_delta_mode,
                            "pair_record_contact_angular_mode": pair_record_contact_angular_mode,
                            "pair_record_contact_vertical_delta_mode": pair_record_contact_vertical_delta_mode,
                            "pair_record_contact_closing_only": pair_record_contact_closing_only,
                            "pair_record_contact_max_velocity_delta": pair_record_contact_max_velocity_delta,
                            "pair_record_contact_max_vertical_delta": pair_record_contact_max_vertical_delta,
                            "pair_record_contact_max_speed": pair_record_contact_max_speed,
                            "pair_record_contact_max_angular_delta": pair_record_contact_max_angular_delta,
                            "pair_record_raw_normal": timed_contact.get(
                                "pair_record_raw_normal"
                            ),
                            "pair_record_selected_raw_normal": timed_contact.get(
                                "pair_record_selected_raw_normal"
                            ),
                        })
                elif timed_contact.get("raw_origin_fallback"):
                    response_debug, applied_raw_fallback = apply_raw_origin_fallback_contact(
                        contact
                    )
                    if not applied_raw_fallback:
                        return False
                elif timed_contact.get("raycast_fallback"):
                    response_debug, applied_raycast_fallback = apply_raw_origin_fallback_contact(
                        contact
                    )
                    if not applied_raycast_fallback:
                        return False
                    if response_debug is not None:
                        response_debug.update({
                            "raycast_fallback": True,
                            "terrain_raycast_fallback": True,
                            "raycast_fallback_reject": timed_contact.get(
                                "raycast_fallback_reject"
                            ),
                            "raycast_fallback_probe_reason": timed_contact.get(
                                "raycast_fallback_probe_reason"
                            ),
                        })
                else:
                    response_debug = apply_contact(contact)
                if timed_contact.get("raw_origin_fallback") and response_debug is not None:
                    response_debug.update({
                        "raw_origin_fallback": True,
                        "raw_origin_timed_fallback": True,
                        "raw_origin_fallback_reject": timed_contact.get(
                            "raw_origin_fallback_reject"
                        ),
                        "raw_origin_fallback_probe_reason": timed_contact.get(
                            "raw_origin_fallback_probe_reason"
                        ),
                    })

                elapsed += collision_time
                current_pos = (anchor[0], anchor[1], anchor[2])
                current_vel = (vx, vy, vz)
                event_debug = {
                    "iteration": iteration_index + 1,
                    "collision_time_s": collision_time,
                    "elapsed_s": elapsed,
                    "remaining_time_s": max(0.0, frame_dt - elapsed),
                    "sweep_iterations": timed_contact["sweep_iterations"],
                    "sweep_clear_count": timed_contact["sweep_clear_count"],
                    "sweep_contact_count": timed_contact["sweep_contact_count"],
                    "collision_at_start": timed_contact["collision_at_start"],
                    "contact_sweep_scan": bool(timed_contact.get("contact_sweep_scan")),
                    "contact_sweep_scan_steps": timed_contact.get(
                        "contact_sweep_scan_steps"
                    ),
                    "contact_sweep_scan_hit_time_s": timed_contact.get(
                        "contact_sweep_scan_hit_time_s"
                    ),
                    "start_time_clamped": timed_contact.get("start_time_clamped", False),
                    "raw_origin_fallback": bool(timed_contact.get("raw_origin_fallback")),
                    "raw_origin_timed_fallback": bool(timed_contact.get("raw_origin_fallback")),
                    "raw_origin_fallback_reject": timed_contact.get(
                        "raw_origin_fallback_reject"
                    ),
                    "raw_origin_fallback_probe_reason": timed_contact.get(
                        "raw_origin_fallback_probe_reason"
                    ),
                    "raycast_fallback": bool(timed_contact.get("raycast_fallback")),
                    "terrain_raycast_fallback": bool(timed_contact.get("raycast_fallback")),
                    "raycast_fallback_reject": timed_contact.get(
                        "raycast_fallback_reject"
                    ),
                    "raycast_fallback_probe_reason": timed_contact.get(
                        "raycast_fallback_probe_reason"
                    ),
                    "pair_record_contact": bool(timed_contact.get("pair_record_contact")),
                    "pair_record_timed_contact": bool(
                        timed_contact.get("pair_record_timed_contact")
                    ),
                    "pair_record_contact_reject": timed_contact.get(
                        "pair_record_contact_reject"
                    ),
                    "pair_record_contact_reason": timed_contact.get(
                        "pair_record_contact_reason"
                    ),
                    "pair_record_contact_selection": timed_contact.get(
                        "pair_record_contact_selection"
                    ),
                    "pair_record_delta_normal": timed_contact.get(
                        "pair_record_delta_normal"
                    ),
                    "pair_record_delta_normal_source": timed_contact.get(
                        "pair_record_delta_normal_source"
                    ),
                    "depth": contact.penetration,
                    "normal": contact.normal,
                    "point": contact.position,
                    **contact_debug_fields(contact),
                }
                if response_debug:
                    event_debug.update({
                        "normal_velocity_before": response_debug.get("normal_velocity_before"),
                        "response": response_debug.get("response"),
                        "iterative_separation_model": response_debug.get("iterative_separation_model"),
                        "iterative_cleared": response_debug.get("iterative_cleared"),
                        "iterative_iterations": response_debug.get("iterative_iterations"),
                        "iterative_position_delta": response_debug.get("iterative_position_delta"),
                        "iterative_position_delta_mag": response_debug.get("iterative_position_delta_mag"),
                        "iterative_final_penetration": response_debug.get("iterative_final_penetration"),
                        "point_normal_velocity_before": response_debug.get("point_normal_velocity_before"),
                        "point_normal_velocity_after": response_debug.get("point_normal_velocity_after"),
                        "normal_delta": response_debug.get("normal_delta"),
                        "position_correction": response_debug.get("position_correction"),
                        "constraint_pair_order": response_debug.get("constraint_pair_order"),
                        "constraint_record_order": response_debug.get("constraint_record_order"),
                        "constraint_record_order_source": response_debug.get("constraint_record_order_source"),
                        "constraint_projection_model": response_debug.get("constraint_projection_model"),
                        "constraint_solver_variant": response_debug.get("constraint_solver_variant"),
                        "constraint_iteration_limit": response_debug.get("constraint_iteration_limit"),
                        "constraint_min_correction_initial": response_debug.get("constraint_min_correction_initial"),
                        "constraint_min_correction_increment": response_debug.get("constraint_min_correction_increment"),
                        "constraint_progressive_scaling": response_debug.get("constraint_progressive_scaling"),
                        "constraint_projection_order": response_debug.get("constraint_projection_order"),
                        "constraint_projection_speed_source": response_debug.get("constraint_projection_speed_source"),
                        "constraint_primary_projection_speed_source": response_debug.get("constraint_primary_projection_speed_source"),
                        "constraint_world_point_velocity_before": response_debug.get("constraint_world_point_velocity_before"),
                        "constraint_body_point_velocity_before": response_debug.get("constraint_body_point_velocity_before"),
                        "constraint_relative_velocity_before": response_debug.get("constraint_relative_velocity_before"),
                        "constraint_opposite_relative_velocity_before": response_debug.get("constraint_opposite_relative_velocity_before"),
                        "constraint_normal_used_for_projection": response_debug.get("constraint_normal_used_for_projection"),
                        "constraint_body_minus_world_speed_before": response_debug.get("constraint_body_minus_world_speed_before"),
                        "constraint_world_minus_body_speed_before": response_debug.get("constraint_world_minus_body_speed_before"),
                        "constraint_selected_separation_speed_before": response_debug.get("constraint_selected_separation_speed_before"),
                        "constraint_separation_speed_before": response_debug.get("constraint_separation_speed_before"),
                        "constraint_opposite_separation_speed_before": response_debug.get("constraint_opposite_separation_speed_before"),
                        "normal_impulse_body_sign": response_debug.get("normal_impulse_body_sign"),
                        "normal_impulse_world_sign": response_debug.get("normal_impulse_world_sign"),
                        "normal_impulse_body_direction": response_debug.get("normal_impulse_body_direction"),
                        "effective_mass_normal": response_debug.get("effective_mass_normal"),
                        "inertia_model": response_debug.get("inertia_model"),
                        "inertia_diagonal": response_debug.get("inertia_diagonal"),
                        "primary_normal_iterations": response_debug.get("primary_normal_iterations"),
                        "primary_start_separation_speed": response_debug.get("primary_start_separation_speed"),
                        "primary_final_separation_speed": response_debug.get("primary_final_separation_speed"),
                        "inactive_retest_enabled": response_debug.get("inactive_retest_enabled"),
                        "inactive_retest_applied": response_debug.get("inactive_retest_applied"),
                        "inactive_retest_iterations": response_debug.get("inactive_retest_iterations"),
                        "inactive_retest_start_separation_speed": response_debug.get("inactive_retest_start_separation_speed"),
                        "inactive_retest_target_separation": response_debug.get("inactive_retest_target_separation"),
                        "inactive_retest_final_separation_speed": response_debug.get("inactive_retest_final_separation_speed"),
                        "normal_impulse": response_debug.get("normal_impulse"),
                        "normal_iterations": response_debug.get("normal_iterations"),
                        "friction_model": response_debug.get("friction_model"),
                        "pair_friction_coeff": response_debug.get("pair_friction_coeff"),
                        "terrain_friction_coeff": response_debug.get("terrain_friction_coeff"),
                        "body_should_sleep": response_debug.get("body_should_sleep"),
                        "body_is_sleeping": response_debug.get("body_is_sleeping"),
                        "constraint_frozen": response_debug.get("constraint_frozen"),
                        "effective_mass_sleep_scale": response_debug.get("effective_mass_sleep_scale"),
                        "impulse_sleep_scale": response_debug.get("impulse_sleep_scale"),
                        "friction_skip_reason": response_debug.get("friction_skip_reason"),
                        "entity_interpolation_model": response_debug.get("entity_interpolation_model"),
                        "interpolation_action": response_debug.get("interpolation_action"),
                        "interpolation_reset_physics": response_debug.get("interpolation_reset_physics"),
                        "interpolation_wake": response_debug.get("interpolation_wake"),
                        "interpolation_update_last_interp_tick": response_debug.get("interpolation_update_last_interp_tick"),
                        "friction_impulse": response_debug.get("friction_impulse"),
                        "friction_iterations": response_debug.get("friction_iterations"),
                        "restitution_impulse": response_debug.get("restitution_impulse"),
                        "velocity_before": response_debug.get("velocity_before"),
                        "velocity_after": response_debug.get("velocity_after"),
                        "angular_velocity_before": response_debug.get("angular_velocity_before"),
                        "angular_velocity_after": response_debug.get("angular_velocity_after"),
                        "angular_delta": response_debug.get("angular_delta"),
                    })
                contact_events.append(event_debug)

                if contact_iteration_limit == 1:
                    remaining_after_contact = max(0.0, frame_dt - elapsed)
                    if remaining_after_contact > 0.0:
                        final_pos, final_vel = motion_state_at(
                            current_pos,
                            current_vel,
                            acc,
                            remaining_after_contact,
                            remaining_after_contact,
                        )
                        current_pos = final_pos
                        current_vel = final_vel
                        elapsed = frame_dt
                    break

            if not contact_events:
                return False

            if elapsed < frame_dt:
                remaining_after_contacts = frame_dt - elapsed
                final_pos, final_vel = motion_state_at(
                    current_pos,
                    current_vel,
                    acc,
                    remaining_after_contacts,
                    remaining_after_contacts,
                )
                current_pos = final_pos
                current_vel = final_vel
                elapsed = frame_dt

            anchor[0], anchor[1], anchor[2] = current_pos
            vx, vy, vz = current_vel
            remaining = max(0.0, frame_dt - elapsed)

            ctx.debug_last_collision = {
                "kind": (
                    "terrain_pair_record_timed_contact"
                    if contact_events[0].get("pair_record_timed_contact")
                    else "terrain_clean_contact"
                ),
                "point": contact.position,
                "normal": contact.normal,
                "depth": contact.penetration,
                "terrain_collision_shape": terrain_collision_shape,
                **contact_debug_fields(contact),
                "timing_response": (
                    "terrain_contact_pair_toi_single_step"
                    if contact_iteration_limit == 1
                    else "terrain_contact_pair_bucketed_step"
                ),
                "contact_timing_mode": contact_timing_mode,
                "collision_time_s": contact_events[0]["collision_time_s"],
                "remaining_time_s": contact_events[0]["remaining_time_s"],
                "final_remaining_time_s": remaining,
                "contact_iteration_limit": contact_iteration_limit,
                "contact_iteration_count": len(contact_events),
                "contact_sweep_scan_enabled": contact_sweep_scan_enabled,
                "contact_sweep_scan_steps": contact_sweep_scan_steps,
                "contact_sweep_scan_event_count": sum(
                    1 for event in contact_events if event.get("contact_sweep_scan")
                ),
                "contact_sweep_scan_hit_time_s": contact_events[0].get(
                    "contact_sweep_scan_hit_time_s"
                ),
                "raw_origin_timed_fallback_enabled": raw_fallback_timed_enabled,
                "raw_origin_timed_fallback_event_count": sum(
                    1 for event in contact_events if event.get("raw_origin_timed_fallback")
                ),
                "pair_record_timed_contact_enabled": pair_record_timed_contact_enabled,
                "pair_record_timed_sweep_enabled": pair_record_timed_sweep_enabled,
                "pair_record_timed_contact_event_count": sum(
                    1 for event in contact_events if event.get("pair_record_timed_contact")
                ),
                "raycast_timed_fallback_enabled": raycast_fallback_timed_enabled,
                "raycast_timed_fallback_event_count": sum(
                    1 for event in contact_events if event.get("terrain_raycast_fallback")
                ),
                "sweep_iterations": sum(event["sweep_iterations"] for event in contact_events),
                "sweep_clear_count": sum(event["sweep_clear_count"] for event in contact_events),
                "sweep_contact_count": sum(event["sweep_contact_count"] for event in contact_events),
                "collision_at_start": contact_events[0]["collision_at_start"],
                "contact_events": contact_events[:8],
                "pre_step_pos": pre_pos,
                "pre_step_vel": pre_vel,
                "contact_pos_at_time": contact_pos,
                "contact_vel_before": contact_vel,
                **(response_debug or {}),
            }
            ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
            return True

        def resolve_single_contact():
            nonlocal vx, vy, vz
            if timed_pair_response and resolve_timed_pair_contact():
                return True
            contact = sample_contact()
            if contact is None or contact.penetration <= self._PENETRATION_SLOP_DEFAULT:
                raw_contact, raw_bounds_contact, raw_error = sample_raw_origin_contact_at(
                    (anchor[0], anchor[1], anchor[2])
                )
                pair_raw_contact = raw_contact
                pair_raw_bounds_contact = raw_bounds_contact
                pair_raw_error = raw_error
                if (
                    pair_record_contact_enabled
                    and raw_error is None
                    and pair_record_contact_selection
                    and pair_record_contact_selection != model_contact_selection
                ):
                    pair_raw_contact, pair_raw_bounds_contact, pair_raw_error = (
                        sample_raw_origin_contact_at(
                            (anchor[0], anchor[1], anchor[2]),
                            contact_selection=pair_record_contact_selection,
                        )
                    )
                    if pair_raw_error is not None:
                        pair_raw_contact = raw_contact
                        pair_raw_bounds_contact = raw_bounds_contact
                raw_fallback_contact, raw_fallback_reject, tank_raw_fallback = (
                    raw_origin_fallback_probe_for(raw_contact, velocity=(vx, vy, vz))
                )
                if raw_fallback_contact is None or raw_fallback_reject in {
                    "no_raw_origin_contact",
                    "above_max_depth",
                    "normal_z_below_min",
                }:
                    tank_face_fallback_latch_clear(raw_fallback_reject or "no_contact")
                pair_contact_source = pair_raw_contact
                if pair_contact_source is None and pair_record_bounds_sat_apply_enabled:
                    pair_contact_source = pair_raw_bounds_contact
                pair_contact = raw_origin_contact_for_fallback(
                    pair_contact_source,
                    normal_source=pair_record_contact_normal_source,
                )
                pair_delta_contact = raw_origin_contact_for_fallback(
                    pair_contact_source,
                    normal_source=pair_record_contact_delta_normal_source,
                )
                pair_contact_reject = pair_record_contact_reject_reason(pair_contact)
                raycast_probe = sample_raycast_fallback_contact_at(
                    (anchor[0], anchor[1], anchor[2]),
                    velocity=(vx, vy, vz),
                    reference=reference_pos,
                )
                update_contact_probe(
                    (anchor[0], anchor[1], anchor[2]),
                    contact,
                    reason=(
                        "lifted_clear"
                        if contact is None
                        else "lifted_contact_below_slop"
                    ),
                    raw_contact=raw_fallback_contact,
                    raw_bounds_contact=raw_bounds_contact,
                    raw_error=raw_error,
                    raw_fallback_reject=raw_fallback_reject,
                    tank_raw_origin_fallback=tank_raw_fallback,
                    pair_record_contact=pair_contact,
                    pair_record_delta_contact=pair_delta_contact,
                    pair_record_contact_reject=pair_contact_reject,
                    pair_record_raw_contact=raw_contact,
                    pair_record_selected_raw_contact=pair_raw_contact,
                    pair_record_selected_raw_bounds_contact=pair_raw_bounds_contact,
                    pair_record_selected_raw_error=pair_raw_error,
                    pair_record_selected_pair_contact_source=(
                        getattr(pair_contact_source, "normal_source", None)
                        if pair_contact_source is not None
                        else None
                    ),
                    raycast_probe=raycast_probe,
                )
                direct_pair_schedule_probe = estimate_direct_pair_record_contact_timing()
                spatial_ref_schedule_probe = (
                    estimate_spatial_ref_pair_record_contact_timing()
                )
                direct_pair_timing = (
                    direct_pair_schedule_probe
                    if pair_record_continue_remaining_enabled
                    else None
                )
                if pair_record_spatial_ref_schedule_probe_enabled:
                    probe_debug = getattr(ctx, "debug_last_terrain_contact_probe", None)
                    if isinstance(probe_debug, dict):
                        spatial_probe = spatial_ref_schedule_probe or {
                            "probe_result": "no_interval_contact",
                            "step_dt_s": max(0.0, float(dt)),
                        }
                        spatial_result = spatial_probe.get(
                            "probe_result", "no_interval_contact"
                        )
                        collision_time_s = spatial_probe.get("collision_time_s")
                        frame_dt = max(0.0, float(dt))
                        bucket_index = None
                        bucket_count = 30
                        bucket_start_s = None
                        bucket_end_s = None
                        bucket_rate_hz = None
                        try:
                            if (
                                frame_dt > 0.0
                                and collision_time_s is not None
                                and math.isfinite(float(collision_time_s))
                            ):
                                bucket_rate_hz = float(bucket_count) / frame_dt
                                bucket_index = max(
                                    0,
                                    min(
                                        bucket_count - 1,
                                        int(
                                            (float(collision_time_s) / frame_dt)
                                            * float(bucket_count)
                                        ),
                                    ),
                                )
                                bucket_width_s = frame_dt / float(bucket_count)
                                bucket_start_s = bucket_width_s * float(bucket_index)
                                bucket_end_s = bucket_width_s * float(bucket_index + 1)
                        except (TypeError, ValueError, OverflowError):
                            bucket_index = None
                            bucket_start_s = None
                            bucket_end_s = None
                            bucket_rate_hz = None
                        probe_debug.update({
                            "pair_record_spatial_ref_schedule_probe_enabled": True,
                            "pair_record_spatial_ref_schedule_probe_result": (
                                spatial_result
                            ),
                            "pair_record_spatial_ref_schedule_decompile_pool_model": (
                                "CollisionPairPool_30_bucket_spatial_ref"
                            ),
                            "pair_record_spatial_ref_schedule_ref_pos": (
                                spatial_probe.get("ref_pos")
                            ),
                            "pair_record_spatial_ref_schedule_current_pos": (
                                spatial_probe.get("current_pos")
                            ),
                            "pair_record_spatial_ref_schedule_ref_to_current_distance": (
                                spatial_probe.get("ref_to_current_distance")
                            ),
                            "pair_record_spatial_ref_schedule_ref_to_current_xy_distance": (
                                spatial_probe.get("ref_to_current_xy_distance")
                            ),
                            "pair_record_spatial_ref_schedule_ref_to_current_z_delta": (
                                spatial_probe.get("ref_to_current_z_delta")
                            ),
                            "pair_record_spatial_ref_schedule_collision_time_s": (
                                collision_time_s
                            ),
                            "pair_record_spatial_ref_schedule_step_dt_s": (
                                spatial_probe.get("step_dt_s", frame_dt)
                            ),
                            "pair_record_spatial_ref_schedule_remaining_time_s": (
                                spatial_probe.get("remaining_time_s")
                            ),
                            "pair_record_spatial_ref_schedule_bucket_count": (
                                bucket_count
                            ),
                            "pair_record_spatial_ref_schedule_bucket_rate_hz": (
                                bucket_rate_hz
                            ),
                            "pair_record_spatial_ref_schedule_bucket_index": (
                                bucket_index
                            ),
                            "pair_record_spatial_ref_schedule_bucket_start_s": (
                                bucket_start_s
                            ),
                            "pair_record_spatial_ref_schedule_bucket_end_s": (
                                bucket_end_s
                            ),
                            "pair_record_spatial_ref_schedule_collision_at_start": (
                                spatial_probe.get("collision_at_start")
                            ),
                            "pair_record_spatial_ref_schedule_sweep_iterations": (
                                spatial_probe.get("sweep_iterations")
                            ),
                            "pair_record_spatial_ref_schedule_sweep_clear_count": (
                                spatial_probe.get("sweep_clear_count")
                            ),
                            "pair_record_spatial_ref_schedule_sweep_contact_count": (
                                spatial_probe.get("sweep_contact_count")
                            ),
                            "pair_record_spatial_ref_schedule_contact_sweep_scan": (
                                spatial_probe.get("contact_sweep_scan")
                            ),
                            "pair_record_spatial_ref_schedule_contact_sweep_scan_steps": (
                                spatial_probe.get("contact_sweep_scan_steps")
                            ),
                            "pair_record_spatial_ref_schedule_contact_sweep_scan_hit_time_s": (
                                spatial_probe.get("contact_sweep_scan_hit_time_s")
                            ),
                            "pair_record_spatial_ref_schedule_contact": (
                                probe_contact_fields(
                                    spatial_probe.get("contact"),
                                    center=spatial_probe.get("pos"),
                                    z_lift_used=0.0,
                                )
                            ),
                            "pair_record_spatial_ref_schedule_delta_contact": (
                                probe_contact_fields(
                                    spatial_probe.get("delta_contact"),
                                    center=spatial_probe.get("pos"),
                                    z_lift_used=0.0,
                                )
                            ),
                            "pair_record_spatial_ref_schedule_contact_pos": (
                                spatial_probe.get("pos")
                            ),
                            "pair_record_spatial_ref_schedule_contact_vel_before": (
                                spatial_probe.get("velocity")
                            ),
                        })
                if pair_record_schedule_probe_enabled:
                    probe_debug = getattr(ctx, "debug_last_terrain_contact_probe", None)
                    if isinstance(probe_debug, dict):
                        if direct_pair_schedule_probe is None:
                            schedule_update = {
                                "pair_record_schedule_probe_result": "no_interval_contact",
                            }
                            if pair_record_schedule_response_probe_enabled:
                                schedule_update.update(
                                    probe_direct_pair_record_contact_response(None)
                                )
                            if pair_record_continue_remaining_enabled:
                                schedule_update[
                                    "pair_record_continue_probe_result"
                                ] = "no_interval_contact"
                            probe_debug.update(schedule_update)
                            selected_trace = probe_debug.get(
                                "selected_row_phase_trace"
                            )
                            if isinstance(selected_trace, dict):
                                selected_trace.update({
                                    "pair_record_schedule_probe_result": (
                                        schedule_update.get(
                                            "pair_record_schedule_probe_result"
                                        )
                                    ),
                                    "pair_record_schedule_response_probe_enabled": (
                                        pair_record_schedule_response_probe_enabled
                                    ),
                                })
                        else:
                            collision_time_s = direct_pair_schedule_probe.get(
                                "collision_time_s"
                            )
                            remaining_time_s = direct_pair_schedule_probe.get(
                                "remaining_time_s"
                            )
                            frame_dt = max(0.0, float(dt))
                            bucket_index = None
                            bucket_count = 30
                            bucket_start_s = None
                            bucket_end_s = None
                            bucket_rate_hz = None
                            try:
                                if (
                                    frame_dt > 0.0
                                    and collision_time_s is not None
                                    and math.isfinite(float(collision_time_s))
                                ):
                                    bucket_rate_hz = float(bucket_count) / frame_dt
                                    bucket_index = max(
                                        0,
                                        min(
                                            bucket_count - 1,
                                            int(
                                                (
                                                    float(collision_time_s)
                                                    / frame_dt
                                                )
                                                * float(bucket_count)
                                            ),
                                        ),
                                    )
                                    bucket_width_s = frame_dt / float(bucket_count)
                                    bucket_start_s = bucket_width_s * float(bucket_index)
                                    bucket_end_s = bucket_width_s * float(bucket_index + 1)
                            except (TypeError, ValueError, OverflowError):
                                bucket_index = None
                                bucket_start_s = None
                                bucket_end_s = None
                                bucket_rate_hz = None
                            schedule_update = {
                                "pair_record_schedule_probe_result": "interval_contact",
                                "pair_record_schedule_decompile_pool_model": (
                                    "CollisionPairPool_30_bucket_time_order"
                                ),
                                "pair_record_schedule_collision_time_s": collision_time_s,
                                "pair_record_schedule_step_dt_s": frame_dt,
                                "pair_record_schedule_remaining_time_s": remaining_time_s,
                                "pair_record_schedule_resolve_remaining_to_tick_end_s": (
                                    remaining_time_s
                                ),
                                "pair_record_schedule_bucket_count": bucket_count,
                                "pair_record_schedule_bucket_rate_hz": bucket_rate_hz,
                                "pair_record_schedule_bucket_index": bucket_index,
                                "pair_record_schedule_bucket_start_s": bucket_start_s,
                                "pair_record_schedule_bucket_end_s": bucket_end_s,
                                "pair_record_schedule_collision_at_start": direct_pair_schedule_probe.get(
                                    "collision_at_start"
                                ),
                                "pair_record_schedule_sweep_iterations": direct_pair_schedule_probe.get(
                                    "sweep_iterations"
                                ),
                                "pair_record_schedule_sweep_clear_count": direct_pair_schedule_probe.get(
                                    "sweep_clear_count"
                                ),
                                "pair_record_schedule_sweep_contact_count": direct_pair_schedule_probe.get(
                                    "sweep_contact_count"
                                ),
                                "pair_record_schedule_contact_sweep_scan": direct_pair_schedule_probe.get(
                                    "contact_sweep_scan"
                                ),
                                "pair_record_schedule_contact_sweep_scan_steps": direct_pair_schedule_probe.get(
                                    "contact_sweep_scan_steps"
                                ),
                                "pair_record_schedule_contact_sweep_scan_hit_time_s": direct_pair_schedule_probe.get(
                                    "contact_sweep_scan_hit_time_s"
                                ),
                                "pair_record_schedule_contact": probe_contact_fields(
                                    direct_pair_schedule_probe.get("contact"),
                                    center=direct_pair_schedule_probe.get("pos"),
                                    z_lift_used=0.0,
                                ),
                                "pair_record_schedule_delta_contact": probe_contact_fields(
                                    direct_pair_schedule_probe.get("delta_contact"),
                                    center=direct_pair_schedule_probe.get("pos"),
                                    z_lift_used=0.0,
                                ),
                                "pair_record_schedule_contact_pos": direct_pair_schedule_probe.get(
                                    "pos"
                                ),
                                "pair_record_schedule_contact_vel_before": direct_pair_schedule_probe.get(
                                    "velocity"
                                ),
                            }
                            if pair_record_schedule_response_probe_enabled:
                                schedule_update.update(
                                    probe_direct_pair_record_contact_response(
                                        direct_pair_schedule_probe
                                    )
                                )
                            if pair_record_continue_remaining_enabled:
                                schedule_update.update({
                                    "pair_record_continue_probe_result": "interval_contact",
                                    "pair_record_continue_collision_time_s": direct_pair_schedule_probe.get(
                                        "collision_time_s"
                                    ),
                                    "pair_record_continue_remaining_time_s": direct_pair_schedule_probe.get(
                                        "remaining_time_s"
                                    ),
                                    "pair_record_continue_collision_at_start": direct_pair_schedule_probe.get(
                                        "collision_at_start"
                                    ),
                                    "pair_record_continue_sweep_iterations": direct_pair_schedule_probe.get(
                                        "sweep_iterations"
                                    ),
                                    "pair_record_continue_sweep_clear_count": direct_pair_schedule_probe.get(
                                        "sweep_clear_count"
                                    ),
                                    "pair_record_continue_sweep_contact_count": direct_pair_schedule_probe.get(
                                        "sweep_contact_count"
                                    ),
                                    "pair_record_continue_contact_sweep_scan": direct_pair_schedule_probe.get(
                                        "contact_sweep_scan"
                                    ),
                                    "pair_record_continue_contact_sweep_scan_steps": direct_pair_schedule_probe.get(
                                        "contact_sweep_scan_steps"
                                    ),
                                    "pair_record_continue_contact_sweep_scan_hit_time_s": direct_pair_schedule_probe.get(
                                        "contact_sweep_scan_hit_time_s"
                                    ),
                                })
                            probe_debug.update(schedule_update)
                            selected_trace = probe_debug.get(
                                "selected_row_phase_trace"
                            )
                            if isinstance(selected_trace, dict):
                                schedule_contact = direct_pair_schedule_probe.get(
                                    "contact"
                                )
                                selected_trace.update({
                                    "pair_record_schedule_probe_result": (
                                        schedule_update.get(
                                            "pair_record_schedule_probe_result"
                                        )
                                    ),
                                    "pair_record_schedule_collision_time_s": (
                                        collision_time_s
                                    ),
                                    "pair_record_schedule_step_dt_s": frame_dt,
                                    "pair_record_schedule_remaining_time_s": (
                                        remaining_time_s
                                    ),
                                    "pair_record_schedule_bucket_count": bucket_count,
                                    "pair_record_schedule_bucket_index": bucket_index,
                                    "pair_record_schedule_bucket_start_s": (
                                        bucket_start_s
                                    ),
                                    "pair_record_schedule_bucket_end_s": (
                                        bucket_end_s
                                    ),
                                    "pair_record_schedule_collision_at_start": (
                                        direct_pair_schedule_probe.get(
                                            "collision_at_start"
                                        )
                                    ),
                                    "pair_record_schedule_contact_source_family": (
                                        getattr(
                                            schedule_contact,
                                            "cbsp_record_hit_source",
                                            None,
                                        )
                                        if schedule_contact is not None
                                        else None
                                    ),
                                    "pair_record_schedule_contact": (
                                        probe_contact_fields(
                                            schedule_contact,
                                            center=direct_pair_schedule_probe.get(
                                                "pos"
                                            ),
                                            z_lift_used=0.0,
                                        )
                                    ),
                                    "pair_record_schedule_contact_pos": (
                                        direct_pair_schedule_probe.get("pos")
                                    ),
                                    "pair_record_schedule_contact_vel_before": (
                                        direct_pair_schedule_probe.get("velocity")
                                    ),
                                    "pair_record_schedule_response_probe_enabled": (
                                        pair_record_schedule_response_probe_enabled
                                    ),
                                    "pair_record_schedule_response_probe_result": (
                                        schedule_update.get(
                                            "pair_record_schedule_response_probe_result"
                                        )
                                    ),
                                    "pair_record_schedule_response_probe_applied": (
                                        schedule_update.get(
                                            "pair_record_schedule_response_probe_applied"
                                        )
                                    ),
                                    "pair_record_schedule_response_probe_velocity_delta": (
                                        schedule_update.get(
                                            "pair_record_schedule_response_probe_velocity_delta"
                                        )
                                    ),
                                    "pair_record_schedule_response_probe_angular_delta": (
                                        schedule_update.get(
                                            "pair_record_schedule_response_probe_angular_delta"
                                        )
                                    ),
                                    "pair_record_schedule_response_probe_post_contact_pos": (
                                        schedule_update.get(
                                            "pair_record_schedule_response_probe_post_contact_pos"
                                        )
                                    ),
                                    "pair_record_schedule_response_probe_endpoint_pos": (
                                        schedule_update.get(
                                            "pair_record_schedule_response_probe_endpoint_pos"
                                        )
                                    ),
                                })
                if (
                    direct_pair_timing is not None
                    or (
                        pair_raw_error is None
                        and pair_contact is not None
                        and pair_contact_reject == ""
                    )
                ):
                    if direct_pair_timing is not None:
                        pair_contact = direct_pair_timing["contact"]
                        pair_delta_contact = direct_pair_timing.get("delta_contact")
                        contact_pos = direct_pair_timing["pos"]
                        contact_vel = direct_pair_timing["velocity"]
                        anchor[0], anchor[1], anchor[2] = contact_pos
                        vx, vy, vz = contact_vel
                    response_debug, applied_pair_record_contact = apply_raw_origin_fallback_contact(
                        pair_contact,
                        projection_order=pair_record_contact_projection_order,
                        delta_mode=pair_record_contact_delta_mode,
                        delta_normal=(
                            None
                            if pair_delta_contact is None
                            else pair_delta_contact.normal
                        ),
                        delta_normal_source=(
                            None
                            if pair_delta_contact is None
                            else getattr(pair_delta_contact, "normal_source", None)
                        ),
                        angular_mode=pair_record_contact_angular_mode,
                        closing_only=pair_record_contact_closing_only,
                        max_velocity_delta=pair_record_contact_max_velocity_delta,
                        max_vertical_delta=pair_record_contact_max_vertical_delta,
                        vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                        max_speed=pair_record_contact_max_speed,
                        max_angular_delta=pair_record_contact_max_angular_delta,
                    )
                    if not applied_pair_record_contact:
                        return False
                    if pair_record_phase_lookahead_queue_enabled:
                        setattr(ctx, phase_lookahead_queue_attr, None)
                    if direct_pair_timing is not None:
                        response_debug = dict(response_debug or {})
                        post_contact_pos = (anchor[0], anchor[1], anchor[2])
                        post_contact_vel = (vx, vy, vz)
                        remaining_after_contact = float(
                            direct_pair_timing.get("remaining_time_s") or 0.0
                        )
                        if remaining_after_contact > 0.0:
                            final_pos, final_vel = motion_state_at(
                                post_contact_pos,
                                post_contact_vel,
                                timing_acceleration(),
                                remaining_after_contact,
                                remaining_after_contact,
                            )
                            anchor[0], anchor[1], anchor[2] = final_pos
                            vx, vy, vz = final_vel
                        direct_event_debug = {
                            "iteration": 1,
                            "collision_time_s": direct_pair_timing.get(
                                "collision_time_s"
                            ),
                            "remaining_time_s": direct_pair_timing.get(
                                "remaining_time_s"
                            ),
                            "sweep_iterations": direct_pair_timing.get(
                                "sweep_iterations"
                            ),
                            "sweep_clear_count": direct_pair_timing.get(
                                "sweep_clear_count"
                            ),
                            "sweep_contact_count": direct_pair_timing.get(
                                "sweep_contact_count"
                            ),
                            "contact_sweep_scan": direct_pair_timing.get(
                                "contact_sweep_scan"
                            ),
                            "contact_sweep_scan_steps": direct_pair_timing.get(
                                "contact_sweep_scan_steps"
                            ),
                            "contact_sweep_scan_hit_time_s": direct_pair_timing.get(
                                "contact_sweep_scan_hit_time_s"
                            ),
                            "collision_at_start": direct_pair_timing.get(
                                "collision_at_start"
                            ),
                            "pair_record_contact": True,
                            "pair_record_continued_contact": True,
                            "point": pair_contact.position,
                            "normal": pair_contact.normal,
                            "depth": pair_contact.penetration,
                            **contact_debug_fields(pair_contact),
                            **response_debug,
                        }
                        response_debug.update({
                            "pair_record_continued_contact": True,
                            "pair_record_continue_remaining_enabled": (
                                pair_record_continue_remaining_enabled
                            ),
                            "pair_record_continue_collision_time_s": (
                                direct_pair_timing.get("collision_time_s")
                            ),
                            "pair_record_continue_remaining_time_s": (
                                direct_pair_timing.get("remaining_time_s")
                            ),
                            "pair_record_continue_collision_at_start": (
                                direct_pair_timing.get("collision_at_start")
                            ),
                            "pair_record_continue_sweep_iterations": (
                                direct_pair_timing.get("sweep_iterations")
                            ),
                            "pair_record_continue_sweep_clear_count": (
                                direct_pair_timing.get("sweep_clear_count")
                            ),
                            "pair_record_continue_sweep_contact_count": (
                                direct_pair_timing.get("sweep_contact_count")
                            ),
                            "pair_record_continue_contact_sweep_scan": (
                                direct_pair_timing.get("contact_sweep_scan")
                            ),
                            "pair_record_continue_contact_sweep_scan_steps": (
                                direct_pair_timing.get("contact_sweep_scan_steps")
                            ),
                            "pair_record_continue_contact_sweep_scan_hit_time_s": (
                                direct_pair_timing.get("contact_sweep_scan_hit_time_s")
                            ),
                            "pair_record_continue_contact_pos": (
                                direct_pair_timing.get("pos")
                            ),
                            "pair_record_continue_contact_vel_before": (
                                direct_pair_timing.get("velocity")
                            ),
                            "pair_record_continue_post_contact_pos": post_contact_pos,
                            "pair_record_continue_post_contact_vel": post_contact_vel,
                            "velocity_after": (vx, vy, vz),
                            "contact_events": [direct_event_debug],
                        })
                    record_pair_record_contact_cache(
                        pair_contact,
                        delta_contact=pair_delta_contact,
                        pos=(anchor[0], anchor[1], anchor[2]),
                        velocity=(vx, vy, vz),
                        source=(
                            "continued_pair_record_contact"
                            if direct_pair_timing is not None
                            else "pair_record_contact"
                        ),
                    )
                    ctx.debug_last_collision = {
                        "kind": (
                            "terrain_pair_record_continued_contact"
                            if direct_pair_timing is not None
                            else "terrain_pair_record_contact"
                        ),
                        "point": pair_contact.position,
                        "normal": pair_contact.normal,
                        "depth": pair_contact.penetration,
                        **contact_debug_fields(pair_contact),
                        "lifted_contact_missing": contact is None,
                        "lifted_contact_depth": (
                            None if contact is None else contact.penetration
                        ),
                        "pair_record_contact": True,
                        "pair_record_contact_reason": (
                            (
                                "lifted_clear_raw_origin_bounds_contact"
                                if pair_raw_contact is None
                                and pair_raw_bounds_contact is not None
                                else "lifted_clear_raw_origin_contact"
                            )
                            if contact is None
                            else (
                                "lifted_below_slop_raw_origin_bounds_contact"
                                if pair_raw_contact is None
                                and pair_raw_bounds_contact is not None
                                else "lifted_below_slop_raw_origin_contact"
                            )
                        ),
                        "pair_record_contact_reject": pair_contact_reject,
                        "pair_record_contact_enabled": pair_record_contact_enabled,
                        "pair_record_bounds_sat_enabled": pair_record_bounds_sat_enabled,
                        "pair_record_bounds_sat_apply_enabled": (
                            pair_record_bounds_sat_apply_enabled
                        ),
                        "pair_record_contact_response_profile": (
                            pair_record_contact_response_profile
                        ),
                        "pair_record_contact_selection": pair_record_contact_selection,
                        "pair_record_contact_normal_source": pair_record_contact_normal_source,
                        "pair_record_contact_delta_normal_source": pair_record_contact_delta_normal_source,
                        "pair_record_solver_normal_source": getattr(
                            pair_contact,
                            "normal_source",
                            None,
                        ),
                        "pair_record_delta_normal_source": (
                            None
                            if pair_delta_contact is None
                            else getattr(pair_delta_contact, "normal_source", None)
                        ),
                        "pair_record_delta_normal": (
                            None if pair_delta_contact is None else pair_delta_contact.normal
                        ),
                        "pair_record_contact_projection_order": pair_record_contact_projection_order,
                        "pair_record_contact_delta_mode": pair_record_contact_delta_mode,
                        "pair_record_contact_angular_mode": pair_record_contact_angular_mode,
                        "pair_record_contact_vertical_delta_mode": pair_record_contact_vertical_delta_mode,
                        "pair_record_contact_closing_only": pair_record_contact_closing_only,
                        "pair_record_contact_min_depth": pair_record_contact_min_depth,
                        "pair_record_contact_max_depth": pair_record_contact_max_depth,
                        "pair_record_contact_min_normal_z": pair_record_contact_min_normal_z,
                        "pair_record_contact_min_face_normal_z": pair_record_contact_min_face_normal_z,
                        "pair_record_contact_max_velocity_delta": pair_record_contact_max_velocity_delta,
                        "pair_record_contact_max_vertical_delta": pair_record_contact_max_vertical_delta,
                        "pair_record_contact_max_speed": pair_record_contact_max_speed,
                        "pair_record_contact_max_angular_delta": pair_record_contact_max_angular_delta,
                        "pair_record_raw_normal": getattr(raw_contact, "normal", None),
                        "pair_record_selected_raw_normal": getattr(
                            pair_raw_contact,
                            "normal",
                            None,
                        ),
                        "pair_record_selected_raw_bounds_normal": getattr(
                            pair_raw_bounds_contact,
                            "normal",
                            None,
                        ),
                        "pair_record_terrain_face_normal": getattr(
                            pair_raw_contact or pair_raw_bounds_contact,
                            "terrain_face_normal",
                            None,
                        ),
                        "pair_record_mesh_face_normal": getattr(
                            pair_raw_contact or pair_raw_bounds_contact,
                            "mesh_face_normal",
                            None,
                        ),
                        "pair_record_entity_radial_normal": getattr(
                            pair_raw_contact or pair_raw_bounds_contact,
                            "entity_radial_normal",
                            None,
                        ),
                        **(response_debug or {}),
                    }
                    ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                    return True
                queued_phase_lookahead = getattr(
                    ctx,
                    phase_lookahead_queue_attr,
                    None,
                )
                queued_phase_applied = False
                queued_phase_debug = {}
                if pair_record_phase_lookahead_queue_enabled:
                    queued_phase_applied, queued_phase_debug = (
                        apply_queued_phase_lookahead_pair_record_contact(
                            queued_phase_lookahead
                        )
                    )
                    probe_debug = getattr(ctx, "debug_last_terrain_contact_probe", None)
                    if isinstance(probe_debug, dict) and queued_phase_debug:
                        probe_debug.update({
                            "pair_record_phase_lookahead_queue_enabled": True,
                            "pair_record_phase_lookahead_queue_pending": (
                                queued_phase_debug.get("pending")
                            ),
                            "pair_record_phase_lookahead_queue_reject": (
                                queued_phase_debug.get("reject")
                            ),
                            "pair_record_phase_lookahead_queue_time_to_contact_s": (
                                queued_phase_debug.get("time_to_contact_s")
                            ),
                            "pair_record_phase_lookahead_queue_age_steps": (
                                queued_phase_debug.get("age_steps")
                            ),
                            "pair_record_phase_lookahead_queue_collision_time_s": (
                                queued_phase_debug.get("collision_time_s")
                            ),
                            "pair_record_phase_lookahead_queue_distance": (
                                queued_phase_debug.get("distance")
                            ),
                        })
                    if queued_phase_applied:
                        queued_contact = (
                            queued_phase_lookahead.get("contact")
                            if isinstance(queued_phase_lookahead, dict)
                            else None
                        )
                        queued_delta_contact = (
                            queued_phase_lookahead.get("delta_contact")
                            if isinstance(queued_phase_lookahead, dict)
                            else None
                        )
                        event_debug = {
                            "iteration": 1,
                            "pair_record_contact": True,
                            "phase_lookahead_queued_pair_record_contact": True,
                            "point": (
                                None if queued_contact is None else queued_contact.position
                            ),
                            "normal": (
                                None if queued_contact is None else queued_contact.normal
                            ),
                            "depth": (
                                None
                                if queued_contact is None
                                else queued_contact.penetration
                            ),
                            **contact_debug_fields(queued_contact),
                            **queued_phase_debug,
                        }
                        record_pair_record_contact_cache(
                            queued_contact,
                            delta_contact=queued_delta_contact,
                            pos=(anchor[0], anchor[1], anchor[2]),
                            velocity=(vx, vy, vz),
                            source="phase_lookahead_queued_pair_record_contact",
                        )
                        queued_probe = (
                            queued_phase_lookahead.get("probe") or {}
                            if isinstance(queued_phase_lookahead, dict)
                            else {}
                        )
                        queued_raw_contact = queued_probe.get("raw_contact")
                        queued_selected_raw_contact = queued_probe.get(
                            "selected_raw_contact"
                        )
                        ctx.debug_last_collision = {
                            "kind": "terrain_phase_lookahead_queued_pair_record_contact",
                            "point": (
                                None if queued_contact is None else queued_contact.position
                            ),
                            "normal": (
                                None if queued_contact is None else queued_contact.normal
                            ),
                            "depth": (
                                None
                                if queued_contact is None
                                else queued_contact.penetration
                            ),
                            **contact_debug_fields(queued_contact),
                            "lifted_contact_missing": contact is None,
                            "lifted_contact_depth": (
                                None if contact is None else contact.penetration
                            ),
                            "pair_record_contact": True,
                            "phase_lookahead_queued_pair_record_contact": True,
                            "pair_record_contact_reason": (
                                "phase_lookahead_queued_pair_record_contact"
                            ),
                            "pair_record_contact_reject": pair_contact_reject,
                            "pair_record_contact_enabled": pair_record_contact_enabled,
                            "pair_record_contact_response_profile": (
                                pair_record_contact_response_profile
                            ),
                            "pair_record_contact_selection": (
                                pair_record_contact_selection
                            ),
                            "pair_record_contact_normal_source": (
                                pair_record_contact_normal_source
                            ),
                            "pair_record_contact_delta_normal_source": (
                                pair_record_contact_delta_normal_source
                            ),
                            "pair_record_solver_normal_source": getattr(
                                queued_contact,
                                "normal_source",
                                None,
                            ),
                            "pair_record_delta_normal_source": (
                                None
                                if queued_delta_contact is None
                                else getattr(
                                    queued_delta_contact,
                                    "normal_source",
                                    None,
                                )
                            ),
                            "pair_record_delta_normal": (
                                None
                                if queued_delta_contact is None
                                else queued_delta_contact.normal
                            ),
                            "pair_record_contact_projection_order": (
                                pair_record_contact_projection_order
                            ),
                            "pair_record_contact_delta_mode": (
                                pair_record_contact_delta_mode
                            ),
                            "pair_record_contact_angular_mode": (
                                pair_record_contact_angular_mode
                            ),
                            "pair_record_contact_vertical_delta_mode": (
                                pair_record_contact_vertical_delta_mode
                            ),
                            "pair_record_contact_closing_only": (
                                pair_record_contact_closing_only
                            ),
                            "pair_record_contact_min_depth": pair_record_contact_min_depth,
                            "pair_record_contact_max_depth": pair_record_contact_max_depth,
                            "pair_record_contact_min_normal_z": (
                                pair_record_contact_min_normal_z
                            ),
                            "pair_record_contact_min_face_normal_z": (
                                pair_record_contact_min_face_normal_z
                            ),
                            "pair_record_contact_max_velocity_delta": (
                                pair_record_contact_max_velocity_delta
                            ),
                            "pair_record_contact_max_vertical_delta": (
                                pair_record_contact_max_vertical_delta
                            ),
                            "pair_record_contact_max_speed": (
                                pair_record_contact_max_speed
                            ),
                            "pair_record_contact_max_angular_delta": (
                                pair_record_contact_max_angular_delta
                            ),
                            "pair_record_raw_normal": getattr(
                                queued_raw_contact,
                                "normal",
                                None,
                            ),
                            "pair_record_selected_raw_normal": getattr(
                                queued_selected_raw_contact,
                                "normal",
                                None,
                            ),
                            "pair_record_terrain_face_normal": getattr(
                                queued_selected_raw_contact,
                                "terrain_face_normal",
                                None,
                            ),
                            "pair_record_mesh_face_normal": getattr(
                                queued_selected_raw_contact,
                                "mesh_face_normal",
                                None,
                            ),
                            "pair_record_entity_radial_normal": getattr(
                                queued_selected_raw_contact,
                                "entity_radial_normal",
                                None,
                            ),
                            "contact_events": [event_debug],
                            **queued_phase_debug,
                        }
                        ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                        return True

                phase_lookahead_pair = estimate_phase_lookahead_pair_record_contact(
                    pair_contact_reject,
                    pair_raw_error,
                )
                if pair_record_phase_lookahead_enabled:
                    probe_debug = getattr(ctx, "debug_last_terrain_contact_probe", None)
                    if isinstance(probe_debug, dict):
                        phase_probe = phase_lookahead_pair.get("probe") or {}
                        phase_contact = phase_lookahead_pair.get("contact")
                        phase_delta_contact = phase_lookahead_pair.get("delta_contact")
                        probe_debug.update({
                            "pair_record_phase_lookahead_enabled": (
                                pair_record_phase_lookahead_enabled
                            ),
                            "pair_record_phase_lookahead_apply_enabled": (
                                pair_record_phase_lookahead_apply_enabled
                            ),
                            "pair_record_phase_lookahead_queue_enabled": (
                                pair_record_phase_lookahead_queue_enabled
                            ),
                            "pair_record_phase_lookahead_mode": (
                                pair_record_phase_lookahead_mode
                            ),
                            "pair_record_phase_lookahead_reject": (
                                phase_lookahead_pair.get("reject")
                            ),
                            "pair_record_phase_lookahead_collision_time_s": (
                                phase_lookahead_pair.get("collision_time_s")
                            ),
                            "pair_record_phase_lookahead_hit_time_s": (
                                phase_lookahead_pair.get("hit_time_s")
                            ),
                            "pair_record_phase_lookahead_distance": (
                                phase_lookahead_pair.get("distance")
                            ),
                            "pair_record_phase_lookahead_scan_steps": (
                                phase_lookahead_pair.get("scan_steps")
                            ),
                            "pair_record_phase_lookahead_sweep_iterations": (
                                phase_lookahead_pair.get("sweep_iterations")
                            ),
                            "pair_record_phase_lookahead_scan_clear_count": (
                                phase_lookahead_pair.get("scan_clear_count")
                            ),
                            "pair_record_phase_lookahead_scan_contact_count": (
                                phase_lookahead_pair.get("scan_contact_count")
                            ),
                            "pair_record_phase_lookahead_pos": (
                                phase_lookahead_pair.get("pos")
                            ),
                            "pair_record_phase_lookahead_velocity": (
                                phase_lookahead_pair.get("velocity")
                            ),
                            "pair_record_phase_lookahead_acceleration_source": (
                                phase_lookahead_pair.get("acceleration_source")
                            ),
                            "pair_record_phase_lookahead_contact": (
                                probe_contact_fields(
                                    phase_contact,
                                    center=phase_lookahead_pair.get("pos"),
                                    z_lift_used=0.0,
                                )
                            ),
                            "pair_record_phase_lookahead_delta_contact": (
                                probe_contact_fields(
                                    phase_delta_contact,
                                    center=phase_lookahead_pair.get("pos"),
                                    z_lift_used=0.0,
                                )
                            ),
                            "pair_record_phase_lookahead_selected_raw_error": (
                                phase_probe.get("selected_raw_error")
                            ),
                        })
                if (
                    pair_record_phase_lookahead_queue_enabled
                    and phase_lookahead_pair.get("contact") is not None
                    and not isinstance(
                        getattr(ctx, phase_lookahead_queue_attr, None),
                        dict,
                    )
                ):
                    queued_collision_time = max(
                        0.0,
                        float(phase_lookahead_pair.get("collision_time_s") or 0.0),
                    )
                    setattr(
                        ctx,
                        phase_lookahead_queue_attr,
                        {
                            "time_to_contact_s": queued_collision_time,
                            "collision_time_s": queued_collision_time,
                            "hit_time_s": phase_lookahead_pair.get("hit_time_s"),
                            "distance": phase_lookahead_pair.get("distance"),
                            "scan_steps": phase_lookahead_pair.get("scan_steps"),
                            "sweep_iterations": phase_lookahead_pair.get(
                                "sweep_iterations"
                            ),
                            "scan_clear_count": phase_lookahead_pair.get(
                                "scan_clear_count"
                            ),
                            "scan_contact_count": phase_lookahead_pair.get(
                                "scan_contact_count"
                            ),
                            "acceleration_source": phase_lookahead_pair.get(
                                "acceleration_source"
                            ),
                            "pos": phase_lookahead_pair.get("pos"),
                            "velocity": phase_lookahead_pair.get("velocity"),
                            "contact": phase_lookahead_pair.get("contact"),
                            "delta_contact": phase_lookahead_pair.get(
                                "delta_contact"
                            ),
                            "probe": phase_lookahead_pair.get("probe"),
                            "age_steps": 0,
                        },
                    )
                    probe_debug = getattr(ctx, "debug_last_terrain_contact_probe", None)
                    if isinstance(probe_debug, dict):
                        probe_debug.update({
                            "pair_record_phase_lookahead_queue_stored": True,
                            "pair_record_phase_lookahead_queue_time_to_contact_s": (
                                queued_collision_time
                            ),
                            "pair_record_phase_lookahead_queue_distance": (
                                phase_lookahead_pair.get("distance")
                            ),
                        })
                if (
                    pair_record_phase_lookahead_apply_enabled
                    and phase_lookahead_pair.get("contact") is not None
                ):
                    phase_pair_contact = phase_lookahead_pair["contact"]
                    phase_pair_delta_contact = phase_lookahead_pair.get(
                        "delta_contact"
                    )
                    current_pos_before = (anchor[0], anchor[1], anchor[2])
                    current_vel_before = (vx, vy, vz)
                    current_ang_before = tuple(contact_angular_velocity)
                    phase_pos = phase_lookahead_pair["pos"]
                    phase_vel = phase_lookahead_pair["velocity"]
                    anchor[0], anchor[1], anchor[2] = phase_pos
                    vx, vy, vz = phase_vel
                    response_debug, applied_phase_pair_contact = (
                        apply_raw_origin_fallback_contact(
                            phase_pair_contact,
                            projection_order=pair_record_contact_projection_order,
                            delta_mode=pair_record_contact_delta_mode,
                            delta_normal=(
                                None
                                if phase_pair_delta_contact is None
                                else phase_pair_delta_contact.normal
                            ),
                            delta_normal_source=(
                                None
                                if phase_pair_delta_contact is None
                                else getattr(
                                    phase_pair_delta_contact,
                                    "normal_source",
                                    None,
                                )
                            ),
                            angular_mode=pair_record_contact_angular_mode,
                            closing_only=pair_record_contact_closing_only,
                            max_velocity_delta=pair_record_contact_max_velocity_delta,
                            max_vertical_delta=pair_record_contact_max_vertical_delta,
                            vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                            max_speed=pair_record_contact_max_speed,
                            max_angular_delta=pair_record_contact_max_angular_delta,
                        )
                    )
                    phase_post_contact_pos = (anchor[0], anchor[1], anchor[2])
                    phase_post_contact_vel = (vx, vy, vz)
                    phase_post_contact_ang = tuple(contact_angular_velocity)
                    if not applied_phase_pair_contact:
                        anchor[0], anchor[1], anchor[2] = current_pos_before
                        vx, vy, vz = current_vel_before
                        contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = current_ang_before
                        ctx.spring_body_ang_vel = (
                            current_ang_before[0],
                            current_ang_before[1],
                        )
                        ctx.angular_vel_yaw = current_ang_before[2]
                        return False
                    phase_velocity_delta = (
                        phase_post_contact_vel[0] - phase_vel[0],
                        phase_post_contact_vel[1] - phase_vel[1],
                        phase_post_contact_vel[2] - phase_vel[2],
                    )
                    phase_angular_delta = (
                        phase_post_contact_ang[0] - current_ang_before[0],
                        phase_post_contact_ang[1] - current_ang_before[1],
                        phase_post_contact_ang[2] - current_ang_before[2],
                    )
                    final_vel = (
                        current_vel_before[0] + phase_velocity_delta[0],
                        current_vel_before[1] + phase_velocity_delta[1],
                        current_vel_before[2] + phase_velocity_delta[2],
                    )
                    anchor[0], anchor[1], anchor[2] = current_pos_before
                    vx, vy, vz = final_vel
                    contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = phase_post_contact_ang
                    ctx.spring_body_ang_vel = (
                        contact_angular_velocity[0],
                        contact_angular_velocity[1],
                    )
                    ctx.angular_vel_yaw = contact_angular_velocity[2]
                    response_debug = dict(response_debug or {})
                    response_debug.update({
                        "pair_record_phase_lookahead_contact": True,
                        "phase_lookahead_pair_record_contact": True,
                        "pair_record_phase_lookahead_apply_enabled": (
                            pair_record_phase_lookahead_apply_enabled
                        ),
                        "pair_record_phase_lookahead_collision_time_s": (
                            phase_lookahead_pair.get("collision_time_s")
                        ),
                        "pair_record_phase_lookahead_hit_time_s": (
                            phase_lookahead_pair.get("hit_time_s")
                        ),
                        "pair_record_phase_lookahead_distance": (
                            phase_lookahead_pair.get("distance")
                        ),
                        "pair_record_phase_lookahead_scan_steps": (
                            phase_lookahead_pair.get("scan_steps")
                        ),
                        "pair_record_phase_lookahead_sweep_iterations": (
                            phase_lookahead_pair.get("sweep_iterations")
                        ),
                        "pair_record_phase_lookahead_scan_clear_count": (
                            phase_lookahead_pair.get("scan_clear_count")
                        ),
                        "pair_record_phase_lookahead_scan_contact_count": (
                            phase_lookahead_pair.get("scan_contact_count")
                        ),
                        "pair_record_phase_lookahead_acceleration_source": (
                            phase_lookahead_pair.get("acceleration_source")
                        ),
                        "pair_record_phase_lookahead_current_pos": (
                            current_pos_before
                        ),
                        "pair_record_phase_lookahead_current_vel": (
                            current_vel_before
                        ),
                        "pair_record_phase_lookahead_contact_pos": phase_pos,
                        "pair_record_phase_lookahead_contact_vel_before": phase_vel,
                        "pair_record_phase_lookahead_post_contact_pos": (
                            phase_post_contact_pos
                        ),
                        "pair_record_phase_lookahead_post_contact_vel": (
                            phase_post_contact_vel
                        ),
                        "pair_record_phase_lookahead_velocity_delta": (
                            phase_velocity_delta
                        ),
                        "pair_record_phase_lookahead_angular_delta": (
                            phase_angular_delta
                        ),
                        "pair_record_phase_lookahead_preserved_position": True,
                        "velocity_before": current_vel_before,
                        "velocity_after": final_vel,
                    })
                    phase_probe = phase_lookahead_pair.get("probe") or {}
                    phase_raw_contact = phase_probe.get("raw_contact")
                    phase_selected_raw_contact = phase_probe.get(
                        "selected_raw_contact"
                    )
                    ctx.debug_last_collision = {
                        "kind": "terrain_phase_lookahead_pair_record_contact",
                        "point": phase_pair_contact.position,
                        "normal": phase_pair_contact.normal,
                        "depth": phase_pair_contact.penetration,
                        **contact_debug_fields(phase_pair_contact),
                        "lifted_contact_missing": contact is None,
                        "lifted_contact_depth": (
                            None if contact is None else contact.penetration
                        ),
                        "pair_record_contact": True,
                        "pair_record_contact_reason": (
                            "phase_lookahead_pair_record_contact"
                        ),
                        "pair_record_contact_reject": "",
                        "pair_record_contact_enabled": pair_record_contact_enabled,
                        "pair_record_contact_response_profile": (
                            pair_record_contact_response_profile
                        ),
                        "pair_record_contact_selection": pair_record_contact_selection,
                        "pair_record_contact_normal_source": pair_record_contact_normal_source,
                        "pair_record_contact_delta_normal_source": pair_record_contact_delta_normal_source,
                        "pair_record_solver_normal_source": getattr(
                            phase_pair_contact,
                            "normal_source",
                            None,
                        ),
                        "pair_record_delta_normal_source": (
                            None
                            if phase_pair_delta_contact is None
                            else getattr(
                                phase_pair_delta_contact,
                                "normal_source",
                                None,
                            )
                        ),
                        "pair_record_delta_normal": (
                            None
                            if phase_pair_delta_contact is None
                            else phase_pair_delta_contact.normal
                        ),
                        "pair_record_contact_projection_order": pair_record_contact_projection_order,
                        "pair_record_contact_delta_mode": pair_record_contact_delta_mode,
                        "pair_record_contact_angular_mode": pair_record_contact_angular_mode,
                        "pair_record_contact_vertical_delta_mode": pair_record_contact_vertical_delta_mode,
                        "pair_record_contact_closing_only": pair_record_contact_closing_only,
                        "pair_record_contact_min_depth": pair_record_contact_min_depth,
                        "pair_record_contact_max_depth": pair_record_contact_max_depth,
                        "pair_record_contact_min_normal_z": pair_record_contact_min_normal_z,
                        "pair_record_contact_min_face_normal_z": pair_record_contact_min_face_normal_z,
                        "pair_record_contact_max_velocity_delta": pair_record_contact_max_velocity_delta,
                        "pair_record_contact_max_vertical_delta": pair_record_contact_max_vertical_delta,
                        "pair_record_contact_max_speed": pair_record_contact_max_speed,
                        "pair_record_contact_max_angular_delta": pair_record_contact_max_angular_delta,
                        "pair_record_raw_normal": getattr(
                            phase_raw_contact,
                            "normal",
                            None,
                        ),
                        "pair_record_selected_raw_normal": getattr(
                            phase_selected_raw_contact,
                            "normal",
                            None,
                        ),
                        "pair_record_terrain_face_normal": getattr(
                            phase_selected_raw_contact,
                            "terrain_face_normal",
                            None,
                        ),
                        "pair_record_mesh_face_normal": getattr(
                            phase_selected_raw_contact,
                            "mesh_face_normal",
                            None,
                        ),
                        "pair_record_entity_radial_normal": getattr(
                            phase_selected_raw_contact,
                            "entity_radial_normal",
                            None,
                        ),
                        **response_debug,
                    }
                    ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                    return True
                phase_backtrack_pair = estimate_phase_backtrack_pair_record_contact(
                    pair_contact_reject,
                    pair_raw_error,
                )
                if pair_record_phase_backtrack_enabled:
                    probe_debug = getattr(ctx, "debug_last_terrain_contact_probe", None)
                    if isinstance(probe_debug, dict):
                        backtrack_probe = phase_backtrack_pair.get("probe") or {}
                        backtrack_contact = phase_backtrack_pair.get("contact")
                        backtrack_delta_contact = phase_backtrack_pair.get(
                            "delta_contact"
                        )
                        probe_debug.update({
                            "pair_record_phase_backtrack_enabled": (
                                pair_record_phase_backtrack_enabled
                            ),
                            "pair_record_phase_backtrack_apply_enabled": (
                                pair_record_phase_backtrack_apply_enabled
                            ),
                            "pair_record_phase_backtrack_mode": (
                                pair_record_phase_backtrack_mode
                            ),
                            "pair_record_phase_backtrack_reject": (
                                phase_backtrack_pair.get("reject")
                            ),
                            "pair_record_phase_backtrack_time_s": (
                                phase_backtrack_pair.get("backtrack_time_s")
                            ),
                            "pair_record_phase_backtrack_hit_time_s": (
                                phase_backtrack_pair.get("hit_time_s")
                            ),
                            "pair_record_phase_backtrack_distance": (
                                phase_backtrack_pair.get("distance")
                            ),
                            "pair_record_phase_backtrack_scan_steps": (
                                phase_backtrack_pair.get("scan_steps")
                            ),
                            "pair_record_phase_backtrack_sweep_iterations": (
                                phase_backtrack_pair.get("sweep_iterations")
                            ),
                            "pair_record_phase_backtrack_scan_clear_count": (
                                phase_backtrack_pair.get("scan_clear_count")
                            ),
                            "pair_record_phase_backtrack_scan_contact_count": (
                                phase_backtrack_pair.get("scan_contact_count")
                            ),
                            "pair_record_phase_backtrack_source": (
                                phase_backtrack_pair.get("source")
                            ),
                            "pair_record_phase_backtrack_base_pos": (
                                phase_backtrack_pair.get("base_pos")
                            ),
                            "pair_record_phase_backtrack_base_vel": (
                                phase_backtrack_pair.get("base_vel")
                            ),
                            "pair_record_phase_backtrack_pos": (
                                phase_backtrack_pair.get("pos")
                            ),
                            "pair_record_phase_backtrack_velocity": (
                                phase_backtrack_pair.get("velocity")
                            ),
                            "pair_record_phase_backtrack_acceleration_source": (
                                phase_backtrack_pair.get("acceleration_source")
                            ),
                            "pair_record_phase_backtrack_contact": (
                                probe_contact_fields(
                                    backtrack_contact,
                                    center=phase_backtrack_pair.get("pos"),
                                    z_lift_used=0.0,
                                )
                            ),
                            "pair_record_phase_backtrack_delta_contact": (
                                probe_contact_fields(
                                    backtrack_delta_contact,
                                    center=phase_backtrack_pair.get("pos"),
                                    z_lift_used=0.0,
                                )
                            ),
                            "pair_record_phase_backtrack_selected_raw_error": (
                                backtrack_probe.get("selected_raw_error")
                            ),
                        })
                if (
                    pair_record_phase_backtrack_apply_enabled
                    and phase_backtrack_pair.get("contact") is not None
                ):
                    backtrack_probe = phase_backtrack_pair.get("probe") or {}
                    backtrack_pair_contact = phase_backtrack_pair["contact"]
                    backtrack_pair_delta_contact = phase_backtrack_pair.get(
                        "delta_contact"
                    )
                    current_pos = (anchor[0], anchor[1], anchor[2])
                    current_vel = (vx, vy, vz)
                    contact_pos = phase_backtrack_pair["pos"]
                    contact_vel = phase_backtrack_pair["velocity"]
                    anchor[0], anchor[1], anchor[2] = contact_pos
                    vx, vy, vz = contact_vel
                    response_debug, applied_backtrack_pair_contact = (
                        apply_raw_origin_fallback_contact(
                            backtrack_pair_contact,
                            projection_order=pair_record_contact_projection_order,
                            delta_mode=pair_record_contact_delta_mode,
                            delta_normal=(
                                None
                                if backtrack_pair_delta_contact is None
                                else backtrack_pair_delta_contact.normal
                            ),
                            delta_normal_source=(
                                None
                                if backtrack_pair_delta_contact is None
                                else getattr(
                                    backtrack_pair_delta_contact,
                                    "normal_source",
                                    None,
                                )
                            ),
                            angular_mode=pair_record_contact_angular_mode,
                            closing_only=pair_record_contact_closing_only,
                            max_velocity_delta=pair_record_contact_max_velocity_delta,
                            max_vertical_delta=pair_record_contact_max_vertical_delta,
                            vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                            max_speed=pair_record_contact_max_speed,
                            max_angular_delta=pair_record_contact_max_angular_delta,
                        )
                    )
                    if not applied_backtrack_pair_contact:
                        return False
                    response_debug = dict(response_debug or {})
                    post_contact_pos = (anchor[0], anchor[1], anchor[2])
                    post_contact_vel = (vx, vy, vz)
                    remaining_after_contact = max(
                        0.0,
                        float(phase_backtrack_pair.get("backtrack_time_s") or 0.0),
                    )
                    replay_acc = (
                        timing_acceleration()
                        if phase_backtrack_pair.get("acceleration_source")
                        == "frame_acceleration"
                        else (0.0, 0.0, 0.0)
                    )
                    if remaining_after_contact > 0.0:
                        final_pos, final_vel = motion_state_at(
                            post_contact_pos,
                            post_contact_vel,
                            replay_acc,
                            remaining_after_contact,
                            remaining_after_contact,
                        )
                        anchor[0], anchor[1], anchor[2] = final_pos
                        vx, vy, vz = final_vel
                    event_debug = {
                        "iteration": 1,
                        "pair_record_contact": True,
                        "phase_backtrack_pair_record_contact": True,
                        "pair_record_phase_backtrack_time_s": (
                            phase_backtrack_pair.get("backtrack_time_s")
                        ),
                        "pair_record_phase_backtrack_hit_time_s": (
                            phase_backtrack_pair.get("hit_time_s")
                        ),
                        "pair_record_phase_backtrack_distance": (
                            phase_backtrack_pair.get("distance")
                        ),
                        "pair_record_phase_backtrack_scan_steps": (
                            phase_backtrack_pair.get("scan_steps")
                        ),
                        "pair_record_phase_backtrack_sweep_iterations": (
                            phase_backtrack_pair.get("sweep_iterations")
                        ),
                        "pair_record_phase_backtrack_scan_clear_count": (
                            phase_backtrack_pair.get("scan_clear_count")
                        ),
                        "pair_record_phase_backtrack_scan_contact_count": (
                            phase_backtrack_pair.get("scan_contact_count")
                        ),
                        "pair_record_phase_backtrack_source": (
                            phase_backtrack_pair.get("source")
                        ),
                        "point": backtrack_pair_contact.position,
                        "normal": backtrack_pair_contact.normal,
                        "depth": backtrack_pair_contact.penetration,
                        **contact_debug_fields(backtrack_pair_contact),
                        **response_debug,
                    }
                    response_debug.update({
                        "phase_backtrack_pair_record_contact": True,
                        "pair_record_phase_backtrack_apply_enabled": (
                            pair_record_phase_backtrack_apply_enabled
                        ),
                        "pair_record_phase_backtrack_time_s": (
                            phase_backtrack_pair.get("backtrack_time_s")
                        ),
                        "pair_record_phase_backtrack_hit_time_s": (
                            phase_backtrack_pair.get("hit_time_s")
                        ),
                        "pair_record_phase_backtrack_distance": (
                            phase_backtrack_pair.get("distance")
                        ),
                        "pair_record_phase_backtrack_scan_steps": (
                            phase_backtrack_pair.get("scan_steps")
                        ),
                        "pair_record_phase_backtrack_sweep_iterations": (
                            phase_backtrack_pair.get("sweep_iterations")
                        ),
                        "pair_record_phase_backtrack_scan_clear_count": (
                            phase_backtrack_pair.get("scan_clear_count")
                        ),
                        "pair_record_phase_backtrack_scan_contact_count": (
                            phase_backtrack_pair.get("scan_contact_count")
                        ),
                        "pair_record_phase_backtrack_source": (
                            phase_backtrack_pair.get("source")
                        ),
                        "pair_record_phase_backtrack_base_pos": (
                            phase_backtrack_pair.get("base_pos")
                        ),
                        "pair_record_phase_backtrack_base_vel": (
                            phase_backtrack_pair.get("base_vel")
                        ),
                        "pair_record_phase_backtrack_contact_pos": contact_pos,
                        "pair_record_phase_backtrack_contact_vel_before": contact_vel,
                        "pair_record_phase_backtrack_current_pos": current_pos,
                        "pair_record_phase_backtrack_current_vel": current_vel,
                        "pair_record_phase_backtrack_post_contact_pos": (
                            post_contact_pos
                        ),
                        "pair_record_phase_backtrack_post_contact_vel": (
                            post_contact_vel
                        ),
                        "pair_record_phase_backtrack_endpoint_pos": (
                            anchor[0],
                            anchor[1],
                            anchor[2],
                        ),
                        "pair_record_phase_backtrack_endpoint_vel": (vx, vy, vz),
                        "pair_record_phase_backtrack_replayed_position": True,
                        "pair_record_phase_backtrack_replay_acceleration_source": (
                            phase_backtrack_pair.get("acceleration_source")
                        ),
                        "velocity_after": (vx, vy, vz),
                        "contact_events": [event_debug],
                    })
                    record_pair_record_contact_cache(
                        backtrack_pair_contact,
                        delta_contact=backtrack_pair_delta_contact,
                        pos=(anchor[0], anchor[1], anchor[2]),
                        velocity=(vx, vy, vz),
                        source="phase_backtrack_pair_record_contact",
                    )
                    ctx.debug_last_collision = {
                        "kind": "terrain_phase_backtrack_pair_record_contact",
                        "point": backtrack_pair_contact.position,
                        "normal": backtrack_pair_contact.normal,
                        "depth": backtrack_pair_contact.penetration,
                        **contact_debug_fields(backtrack_pair_contact),
                        "lifted_contact_missing": contact is None,
                        "lifted_contact_depth": (
                            None if contact is None else contact.penetration
                        ),
                        "pair_record_contact": True,
                        "phase_backtrack_pair_record_contact": True,
                        "pair_record_contact_reason": (
                            "phase_backtrack_pair_record_contact"
                        ),
                        "pair_record_contact_reject": pair_contact_reject,
                        "pair_record_contact_enabled": pair_record_contact_enabled,
                        "pair_record_contact_response_profile": (
                            pair_record_contact_response_profile
                        ),
                        "pair_record_contact_selection": pair_record_contact_selection,
                        "pair_record_contact_normal_source": (
                            pair_record_contact_normal_source
                        ),
                        "pair_record_contact_delta_normal_source": (
                            pair_record_contact_delta_normal_source
                        ),
                        "pair_record_solver_normal_source": getattr(
                            backtrack_pair_contact,
                            "normal_source",
                            None,
                        ),
                        "pair_record_delta_normal_source": (
                            None
                            if backtrack_pair_delta_contact is None
                            else getattr(
                                backtrack_pair_delta_contact,
                                "normal_source",
                                None,
                            )
                        ),
                        "pair_record_delta_normal": (
                            None
                            if backtrack_pair_delta_contact is None
                            else backtrack_pair_delta_contact.normal
                        ),
                        "pair_record_contact_projection_order": (
                            pair_record_contact_projection_order
                        ),
                        "pair_record_contact_delta_mode": (
                            pair_record_contact_delta_mode
                        ),
                        "pair_record_contact_angular_mode": (
                            pair_record_contact_angular_mode
                        ),
                        "pair_record_contact_vertical_delta_mode": (
                            pair_record_contact_vertical_delta_mode
                        ),
                        "pair_record_contact_closing_only": (
                            pair_record_contact_closing_only
                        ),
                        "pair_record_contact_min_depth": pair_record_contact_min_depth,
                        "pair_record_contact_max_depth": pair_record_contact_max_depth,
                        "pair_record_contact_min_normal_z": (
                            pair_record_contact_min_normal_z
                        ),
                        "pair_record_contact_min_face_normal_z": (
                            pair_record_contact_min_face_normal_z
                        ),
                        "pair_record_contact_max_velocity_delta": (
                            pair_record_contact_max_velocity_delta
                        ),
                        "pair_record_contact_max_vertical_delta": (
                            pair_record_contact_max_vertical_delta
                        ),
                        "pair_record_contact_max_speed": (
                            pair_record_contact_max_speed
                        ),
                        "pair_record_contact_max_angular_delta": (
                            pair_record_contact_max_angular_delta
                        ),
                        "pair_record_raw_normal": getattr(raw_contact, "normal", None),
                        "pair_record_selected_raw_normal": getattr(
                            backtrack_probe.get("selected_raw_contact"),
                            "normal",
                            None,
                        ),
                        "pair_record_terrain_face_normal": getattr(
                            backtrack_probe.get("selected_raw_contact"),
                            "terrain_face_normal",
                            None,
                        ),
                        "pair_record_mesh_face_normal": getattr(
                            backtrack_probe.get("selected_raw_contact"),
                            "mesh_face_normal",
                            None,
                        ),
                        "pair_record_entity_radial_normal": getattr(
                            backtrack_probe.get("selected_raw_contact"),
                            "entity_radial_normal",
                            None,
                        ),
                        **response_debug,
                    }
                    ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                    return True
                deferred_prestep_pair = estimate_deferred_prestep_pair_record_contact(
                    pair_contact_reject,
                    pair_raw_error,
                )
                if (
                    pair_record_deferred_prestep_enabled
                    or pair_record_deferred_prestep_probe_enabled
                ):
                    probe_debug = getattr(ctx, "debug_last_terrain_contact_probe", None)
                    if isinstance(probe_debug, dict):
                        deferred_probe = deferred_prestep_pair.get("probe") or {}
                        deferred_contact = deferred_prestep_pair.get("contact")
                        deferred_delta_contact = deferred_prestep_pair.get(
                            "delta_contact"
                        )
                        probe_debug.update({
                            "pair_record_deferred_prestep_enabled": (
                                pair_record_deferred_prestep_enabled
                            ),
                            "pair_record_deferred_prestep_probe_enabled": (
                                pair_record_deferred_prestep_probe_enabled
                            ),
                            "pair_record_deferred_prestep_reject": (
                                deferred_prestep_pair.get("reject")
                            ),
                            "pair_record_deferred_prestep_distance": (
                                deferred_prestep_pair.get("distance")
                            ),
                            "pair_record_deferred_prestep_pos": (
                                deferred_prestep_pair.get("pos")
                            ),
                            "pair_record_deferred_prestep_contact": (
                                probe_contact_fields(
                                    deferred_contact,
                                    center=deferred_prestep_pair.get("pos"),
                                    z_lift_used=0.0,
                                )
                            ),
                            "pair_record_deferred_prestep_delta_contact": (
                                probe_contact_fields(
                                    deferred_delta_contact,
                                    center=deferred_prestep_pair.get("pos"),
                                    z_lift_used=0.0,
                                )
                            ),
                            "pair_record_deferred_prestep_selected_raw_error": (
                                deferred_probe.get("selected_raw_error")
                            ),
                        })
                if (
                    pair_record_deferred_prestep_enabled
                    and deferred_prestep_pair.get("contact") is not None
                ):
                    deferred_pair_contact = deferred_prestep_pair["contact"]
                    deferred_pair_delta_contact = deferred_prestep_pair.get(
                        "delta_contact"
                    )
                    endpoint_pos = (anchor[0], anchor[1], anchor[2])
                    endpoint_vel = (vx, vy, vz)
                    endpoint_ang = tuple(contact_angular_velocity)
                    anchor[0], anchor[1], anchor[2] = deferred_prestep_pair["pos"]
                    vx, vy, vz = deferred_prestep_pair["velocity"]
                    response_debug, applied_deferred_pair_contact = (
                        apply_raw_origin_fallback_contact(
                            deferred_pair_contact,
                            projection_order=pair_record_contact_projection_order,
                            delta_mode=pair_record_contact_delta_mode,
                            delta_normal=(
                                None
                                if deferred_pair_delta_contact is None
                                else deferred_pair_delta_contact.normal
                            ),
                            delta_normal_source=(
                                None
                                if deferred_pair_delta_contact is None
                                else getattr(
                                    deferred_pair_delta_contact,
                                    "normal_source",
                                    None,
                                )
                            ),
                            angular_mode=pair_record_contact_angular_mode,
                            closing_only=pair_record_contact_closing_only,
                            max_velocity_delta=pair_record_contact_max_velocity_delta,
                            max_vertical_delta=pair_record_contact_max_vertical_delta,
                            vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                            max_speed=pair_record_contact_max_speed,
                            max_angular_delta=pair_record_contact_max_angular_delta,
                        )
                    )
                    if not applied_deferred_pair_contact:
                        anchor[0], anchor[1], anchor[2] = endpoint_pos
                        vx, vy, vz = endpoint_vel
                        contact_angular_velocity[0], contact_angular_velocity[1], contact_angular_velocity[2] = endpoint_ang
                        ctx.spring_body_ang_vel = (endpoint_ang[0], endpoint_ang[1])
                        ctx.angular_vel_yaw = endpoint_ang[2]
                    else:
                        response_debug = dict(response_debug or {})
                        post_contact_pos = (anchor[0], anchor[1], anchor[2])
                        post_contact_vel = (vx, vy, vz)
                        remaining_after_contact = max(0.0, float(dt or 0.0))
                        if remaining_after_contact > 0.0:
                            final_pos, final_vel = motion_state_at(
                                post_contact_pos,
                                post_contact_vel,
                                timing_acceleration(),
                                remaining_after_contact,
                                remaining_after_contact,
                            )
                            anchor[0], anchor[1], anchor[2] = final_pos
                            vx, vy, vz = final_vel
                        event_debug = {
                            "iteration": 1,
                            "collision_time_s": 0.0,
                            "remaining_time_s": remaining_after_contact,
                            "collision_at_start": True,
                            "pair_record_contact": True,
                            "pair_record_deferred_prestep_contact": True,
                            "point": deferred_pair_contact.position,
                            "normal": deferred_pair_contact.normal,
                            "depth": deferred_pair_contact.penetration,
                            **contact_debug_fields(deferred_pair_contact),
                            **response_debug,
                        }
                        response_debug.update({
                            "deferred_prestep_pair_record_contact": True,
                            "pair_record_deferred_prestep_enabled": (
                                pair_record_deferred_prestep_enabled
                            ),
                            "pair_record_deferred_prestep_distance": (
                                deferred_prestep_pair.get("distance")
                            ),
                            "pair_record_deferred_prestep_max_distance": (
                                pair_record_deferred_prestep_max_distance
                            ),
                            "pair_record_deferred_prestep_pos": (
                                deferred_prestep_pair.get("pos")
                            ),
                            "pair_record_deferred_prestep_vel_before": (
                                deferred_prestep_pair.get("velocity")
                            ),
                            "pair_record_deferred_prestep_endpoint_pos": endpoint_pos,
                            "pair_record_deferred_prestep_endpoint_vel": endpoint_vel,
                            "pair_record_deferred_prestep_post_contact_pos": (
                                post_contact_pos
                            ),
                            "pair_record_deferred_prestep_post_contact_vel": (
                                post_contact_vel
                            ),
                            "pair_record_deferred_prestep_remaining_time_s": (
                                remaining_after_contact
                            ),
                            "velocity_after": (vx, vy, vz),
                            "contact_events": [event_debug],
                        })
                        deferred_pair_probe = (
                            deferred_prestep_pair.get("probe") or {}
                        )
                        deferred_raw_contact = deferred_pair_probe.get("raw_contact")
                        deferred_selected_raw_contact = deferred_pair_probe.get(
                            "selected_raw_contact"
                        )
                        ctx.debug_last_collision = {
                            "kind": "terrain_deferred_prestep_pair_record_contact",
                            "point": deferred_pair_contact.position,
                            "normal": deferred_pair_contact.normal,
                            "depth": deferred_pair_contact.penetration,
                            **contact_debug_fields(deferred_pair_contact),
                            "lifted_contact_missing": contact is None,
                            "lifted_contact_depth": (
                                None if contact is None else contact.penetration
                            ),
                            "pair_record_contact": True,
                            "pair_record_contact_reason": (
                                "deferred_prestep_pair_record_contact"
                            ),
                            "pair_record_contact_reject": "",
                            "pair_record_contact_enabled": pair_record_contact_enabled,
                            "pair_record_contact_response_profile": (
                                pair_record_contact_response_profile
                            ),
                            "pair_record_contact_selection": pair_record_contact_selection,
                            "pair_record_contact_normal_source": (
                                pair_record_contact_normal_source
                            ),
                            "pair_record_contact_delta_normal_source": (
                                pair_record_contact_delta_normal_source
                            ),
                            "pair_record_solver_normal_source": getattr(
                                deferred_pair_contact,
                                "normal_source",
                                None,
                            ),
                            "pair_record_delta_normal_source": (
                                None
                                if deferred_pair_delta_contact is None
                                else getattr(
                                    deferred_pair_delta_contact,
                                    "normal_source",
                                    None,
                                )
                            ),
                            "pair_record_delta_normal": (
                                None
                                if deferred_pair_delta_contact is None
                                else deferred_pair_delta_contact.normal
                            ),
                            "pair_record_contact_projection_order": pair_record_contact_projection_order,
                            "pair_record_contact_delta_mode": pair_record_contact_delta_mode,
                            "pair_record_contact_angular_mode": pair_record_contact_angular_mode,
                            "pair_record_contact_vertical_delta_mode": pair_record_contact_vertical_delta_mode,
                            "pair_record_contact_closing_only": pair_record_contact_closing_only,
                            "pair_record_contact_min_depth": pair_record_contact_min_depth,
                            "pair_record_contact_max_depth": pair_record_contact_max_depth,
                            "pair_record_contact_min_normal_z": pair_record_contact_min_normal_z,
                            "pair_record_contact_min_face_normal_z": pair_record_contact_min_face_normal_z,
                            "pair_record_contact_max_velocity_delta": pair_record_contact_max_velocity_delta,
                            "pair_record_contact_max_vertical_delta": pair_record_contact_max_vertical_delta,
                            "pair_record_contact_max_speed": pair_record_contact_max_speed,
                            "pair_record_contact_max_angular_delta": pair_record_contact_max_angular_delta,
                            "pair_record_raw_normal": getattr(
                                deferred_raw_contact,
                                "normal",
                                None,
                            ),
                            "pair_record_selected_raw_normal": getattr(
                                deferred_selected_raw_contact,
                                "normal",
                                None,
                            ),
                            "pair_record_terrain_face_normal": getattr(
                                deferred_selected_raw_contact,
                                "terrain_face_normal",
                                None,
                            ),
                            "pair_record_mesh_face_normal": getattr(
                                deferred_selected_raw_contact,
                                "mesh_face_normal",
                                None,
                            ),
                            "pair_record_entity_radial_normal": getattr(
                                deferred_selected_raw_contact,
                                "entity_radial_normal",
                                None,
                            ),
                            **response_debug,
                        }
                        ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                        return True
                reference_pair = select_reference_pose_pair_record_contact(
                    (anchor[0], anchor[1], anchor[2]),
                    velocity=(vx, vy, vz),
                )
                if resolve_reference_pose_pair_response(reference_pair):
                    probe_debug = getattr(ctx, "debug_last_terrain_contact_probe", None)
                    if isinstance(probe_debug, dict):
                        probe_debug["reference_pose_pair_response_enabled"] = (
                            reference_pose_pair_response_enabled
                        )
                        probe_debug["reference_pose_pair_response_apply_enabled"] = (
                            reference_pose_pair_response_apply_enabled
                        )
                        probe_debug["reference_pose_pair_response"] = (
                            dirty_dispatch_debug.get("reference_pose_pair_response")
                        )
                    return True
                if reference_pair is not None and reference_pose_contact_response_enabled:
                    reference_pair_contact = reference_pair["contact"]
                    reference_pair_delta_contact = reference_pair.get("delta_contact")
                    response_debug, applied_reference_pair_contact = (
                        apply_raw_origin_fallback_contact(
                            reference_pair_contact,
                            projection_order=pair_record_contact_projection_order,
                            delta_mode=pair_record_contact_delta_mode,
                            delta_normal=(
                                None
                                if reference_pair_delta_contact is None
                                else reference_pair_delta_contact.normal
                            ),
                            delta_normal_source=(
                                None
                                if reference_pair_delta_contact is None
                                else getattr(
                                    reference_pair_delta_contact,
                                    "normal_source",
                                    None,
                                )
                            ),
                            angular_mode=pair_record_contact_angular_mode,
                            closing_only=pair_record_contact_closing_only,
                            max_velocity_delta=pair_record_contact_max_velocity_delta,
                            max_vertical_delta=pair_record_contact_max_vertical_delta,
                            vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                            max_speed=pair_record_contact_max_speed,
                            max_angular_delta=pair_record_contact_max_angular_delta,
                        )
                    )
                    if not applied_reference_pair_contact:
                        return False
                    reference_pair_probe = reference_pair.get("probe") or {}
                    reference_raw_contact = reference_pair_probe.get("raw_contact")
                    reference_selected_raw_contact = reference_pair_probe.get(
                        "selected_raw_contact"
                    )
                    ctx.debug_last_collision = {
                        "kind": "terrain_reference_pose_pair_record_contact",
                        "point": reference_pair_contact.position,
                        "normal": reference_pair_contact.normal,
                        "depth": reference_pair_contact.penetration,
                        **contact_debug_fields(reference_pair_contact),
                        "lifted_contact_missing": contact is None,
                        "lifted_contact_depth": (
                            None if contact is None else contact.penetration
                        ),
                        "reference_pose_pair_record_contact": True,
                        "reference_pose_contact_response_enabled": (
                            reference_pose_contact_response_enabled
                        ),
                        "reference_pose_contact_label": reference_pair["label"],
                        "reference_pose_contact_pos": reference_pair["pos"],
                        "reference_pose_current_pair_reject": pair_contact_reject,
                        "pair_record_contact": True,
                        "pair_record_contact_reason": (
                            "reference_pose_pair_record_contact"
                        ),
                        "pair_record_contact_reject": "",
                        "pair_record_contact_enabled": pair_record_contact_enabled,
                        "pair_record_contact_response_profile": (
                            pair_record_contact_response_profile
                        ),
                        "pair_record_contact_selection": pair_record_contact_selection,
                        "pair_record_contact_normal_source": pair_record_contact_normal_source,
                        "pair_record_contact_delta_normal_source": pair_record_contact_delta_normal_source,
                        "pair_record_solver_normal_source": getattr(
                            reference_pair_contact,
                            "normal_source",
                            None,
                        ),
                        "pair_record_delta_normal_source": (
                            None
                            if reference_pair_delta_contact is None
                            else getattr(
                                reference_pair_delta_contact,
                                "normal_source",
                                None,
                            )
                        ),
                        "pair_record_delta_normal": (
                            None
                            if reference_pair_delta_contact is None
                            else reference_pair_delta_contact.normal
                        ),
                        "pair_record_contact_projection_order": pair_record_contact_projection_order,
                        "pair_record_contact_delta_mode": pair_record_contact_delta_mode,
                        "pair_record_contact_angular_mode": pair_record_contact_angular_mode,
                        "pair_record_contact_vertical_delta_mode": pair_record_contact_vertical_delta_mode,
                        "pair_record_contact_closing_only": pair_record_contact_closing_only,
                        "pair_record_contact_min_depth": pair_record_contact_min_depth,
                        "pair_record_contact_max_depth": pair_record_contact_max_depth,
                        "pair_record_contact_min_normal_z": pair_record_contact_min_normal_z,
                        "pair_record_contact_min_face_normal_z": pair_record_contact_min_face_normal_z,
                        "pair_record_contact_max_velocity_delta": pair_record_contact_max_velocity_delta,
                        "pair_record_contact_max_vertical_delta": pair_record_contact_max_vertical_delta,
                        "pair_record_contact_max_speed": pair_record_contact_max_speed,
                        "pair_record_contact_max_angular_delta": pair_record_contact_max_angular_delta,
                        "pair_record_raw_normal": getattr(
                            reference_raw_contact,
                            "normal",
                            None,
                        ),
                        "pair_record_selected_raw_normal": getattr(
                            reference_selected_raw_contact,
                            "normal",
                            None,
                        ),
                        "pair_record_terrain_face_normal": getattr(
                            reference_selected_raw_contact,
                            "terrain_face_normal",
                            None,
                        ),
                        "pair_record_mesh_face_normal": getattr(
                            reference_selected_raw_contact,
                            "mesh_face_normal",
                            None,
                        ),
                        "pair_record_entity_radial_normal": getattr(
                            reference_selected_raw_contact,
                            "entity_radial_normal",
                            None,
                        ),
                        **(response_debug or {}),
                    }
                    ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                    return True
                cached_pair = None
                if (
                    pair_contact_reject == "no_raw_origin_contact"
                    and raw_error is None
                ):
                    cached_pair = cached_pair_record_contact_probe(
                        (anchor[0], anchor[1], anchor[2]),
                        velocity=(vx, vy, vz),
                    )
                    probe_debug = getattr(ctx, "debug_last_terrain_contact_probe", None)
                    if isinstance(probe_debug, dict):
                        probe_debug.update({
                            "pair_record_cached_contact_enabled": (
                                pair_record_cached_contact_enabled
                            ),
                            "pair_record_cached_contact_reject": cached_pair.get(
                                "reject"
                            ),
                            "pair_record_cached_contact_age_steps": cached_pair.get(
                                "age_steps"
                            ),
                            "pair_record_cached_contact_distance": cached_pair.get(
                                "distance"
                            ),
                            "pair_record_cached_contact_reference_distance": (
                                cached_pair.get("reference_distance")
                            ),
                            "pair_record_cached_contact_source": cached_pair.get(
                                "source"
                            ),
                        })
                if cached_pair is not None and cached_pair.get("contact") is not None:
                    cached_pair_contact = cached_pair["contact"]
                    cached_pair_delta_contact = cached_pair.get("delta_contact")
                    response_debug, applied_cached_pair_contact = (
                        apply_raw_origin_fallback_contact(
                            cached_pair_contact,
                            projection_order=pair_record_contact_projection_order,
                            delta_mode=pair_record_contact_delta_mode,
                            delta_normal=(
                                None
                                if cached_pair_delta_contact is None
                                else cached_pair_delta_contact.normal
                            ),
                            delta_normal_source=(
                                None
                                if cached_pair_delta_contact is None
                                else getattr(
                                    cached_pair_delta_contact,
                                    "normal_source",
                                    None,
                                )
                            ),
                            angular_mode=pair_record_contact_angular_mode,
                            closing_only=pair_record_contact_closing_only,
                            max_velocity_delta=pair_record_contact_max_velocity_delta,
                            max_vertical_delta=pair_record_contact_max_vertical_delta,
                            vertical_delta_mode=pair_record_contact_vertical_delta_mode,
                            max_speed=pair_record_contact_max_speed,
                            max_angular_delta=pair_record_contact_max_angular_delta,
                        )
                    )
                    if not applied_cached_pair_contact:
                        return False
                    ctx.debug_last_collision = {
                        "kind": "terrain_cached_pair_record_contact",
                        "point": cached_pair_contact.position,
                        "normal": cached_pair_contact.normal,
                        "depth": cached_pair_contact.penetration,
                        **contact_debug_fields(cached_pair_contact),
                        "lifted_contact_missing": contact is None,
                        "lifted_contact_depth": (
                            None if contact is None else contact.penetration
                        ),
                        "cached_pair_record_contact": True,
                        "pair_record_cached_contact_enabled": (
                            pair_record_cached_contact_enabled
                        ),
                        "pair_record_cached_contact_age_steps": cached_pair.get(
                            "age_steps"
                        ),
                        "pair_record_cached_contact_distance": cached_pair.get(
                            "distance"
                        ),
                        "pair_record_cached_contact_reference_distance": (
                            cached_pair.get("reference_distance")
                        ),
                        "pair_record_cached_contact_source": cached_pair.get(
                            "source"
                        ),
                        "pair_record_contact": True,
                        "pair_record_contact_reason": "cached_pair_record_contact",
                        "pair_record_contact_reject": "",
                        "pair_record_contact_enabled": pair_record_contact_enabled,
                        "pair_record_contact_response_profile": (
                            pair_record_contact_response_profile
                        ),
                        "pair_record_contact_selection": pair_record_contact_selection,
                        "pair_record_contact_normal_source": pair_record_contact_normal_source,
                        "pair_record_contact_delta_normal_source": pair_record_contact_delta_normal_source,
                        "pair_record_solver_normal_source": getattr(
                            cached_pair_contact,
                            "normal_source",
                            None,
                        ),
                        "pair_record_delta_normal_source": (
                            None
                            if cached_pair_delta_contact is None
                            else getattr(
                                cached_pair_delta_contact,
                                "normal_source",
                                None,
                            )
                        ),
                        "pair_record_delta_normal": (
                            None
                            if cached_pair_delta_contact is None
                            else cached_pair_delta_contact.normal
                        ),
                        "pair_record_contact_projection_order": pair_record_contact_projection_order,
                        "pair_record_contact_delta_mode": pair_record_contact_delta_mode,
                        "pair_record_contact_angular_mode": pair_record_contact_angular_mode,
                        "pair_record_contact_vertical_delta_mode": pair_record_contact_vertical_delta_mode,
                        "pair_record_contact_closing_only": pair_record_contact_closing_only,
                        "pair_record_contact_min_depth": pair_record_contact_min_depth,
                        "pair_record_contact_max_depth": pair_record_contact_max_depth,
                        "pair_record_contact_min_normal_z": pair_record_contact_min_normal_z,
                        "pair_record_contact_min_face_normal_z": pair_record_contact_min_face_normal_z,
                        "pair_record_contact_max_velocity_delta": pair_record_contact_max_velocity_delta,
                        "pair_record_contact_max_vertical_delta": pair_record_contact_max_vertical_delta,
                        "pair_record_contact_max_speed": pair_record_contact_max_speed,
                        "pair_record_contact_max_angular_delta": pair_record_contact_max_angular_delta,
                        **(response_debug or {}),
                    }
                    ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                    return True
                if raw_error is None and raw_fallback_contact is not None and raw_fallback_reject == "":
                    if tank_raw_fallback and tank_face_fallback_latch_matches(
                        raw_fallback_contact
                    ):
                        ctx.debug_last_collision = tank_face_fallback_latched_debug(
                            raw_fallback_contact,
                            kind="terrain_raw_origin_fallback_latched",
                        )
                        ctx.debug_last_collision.update({
                            "raw_origin_fallback": True,
                            "tank_raw_origin_fallback": True,
                            "raw_origin_fallback_reject": raw_fallback_reject,
                        })
                        ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                        return True
                    tank_raw_delta_normal = None
                    tank_raw_delta_normal_source = None
                    if tank_raw_fallback and not raw_fallback_enabled:
                        tank_raw_delta_normal, tank_raw_delta_normal_source = (
                            tank_raw_fallback_delta_normal_for(raw_fallback_contact)
                        )
                    response_debug, applied_raw_fallback = apply_raw_origin_fallback_contact(
                        raw_fallback_contact,
                        projection_order=(
                            tank_raw_fallback_projection_order
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                        delta_mode=(
                            tank_raw_fallback_delta_mode
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                        delta_normal=tank_raw_delta_normal,
                        delta_normal_source=tank_raw_delta_normal_source,
                        angular_mode=(
                            tank_raw_fallback_angular_mode
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                        closing_only=(
                            tank_raw_fallback_closing_only
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                        max_velocity_delta=(
                            tank_raw_fallback_max_velocity_delta
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                        max_vertical_delta=(
                            tank_raw_fallback_max_vertical_delta
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                        vertical_delta_mode=(
                            tank_raw_fallback_vertical_delta_mode
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                        max_speed=(
                            tank_raw_fallback_max_speed
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                        max_angular_delta=(
                            tank_raw_fallback_max_angular_delta
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                        friction=(
                            0.0
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                    )
                    if not applied_raw_fallback:
                        return False
                    if tank_raw_fallback:
                        tank_face_fallback_latch_record(
                            raw_fallback_contact,
                            source="raw_origin",
                        )
                    raw_debug_projection_order = (
                        tank_raw_fallback_projection_order
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_projection_order
                    )
                    raw_debug_normal_source = (
                        tank_raw_fallback_normal_source
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_normal_source
                    )
                    raw_debug_min_depth = (
                        tank_raw_fallback_min_depth
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_min_depth
                    )
                    raw_debug_max_depth = (
                        tank_raw_fallback_max_depth
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_max_depth
                    )
                    raw_debug_min_normal_z = (
                        tank_raw_fallback_min_normal_z
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_min_normal_z
                    )
                    raw_debug_min_speed = (
                        tank_raw_fallback_min_speed
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_min_speed
                    )
                    raw_debug_max_velocity_delta = (
                        tank_raw_fallback_max_velocity_delta
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_max_velocity_delta
                    )
                    raw_debug_max_speed = (
                        tank_raw_fallback_max_speed
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_max_speed
                    )
                    raw_debug_max_angular_delta = (
                        tank_raw_fallback_max_angular_delta
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_max_angular_delta
                    )
                    raw_debug_angular_mode = (
                        tank_raw_fallback_angular_mode
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_angular_mode
                    )
                    raw_debug_closing_only = (
                        tank_raw_fallback_closing_only
                        if tank_raw_fallback and not raw_fallback_enabled
                        else raw_fallback_closing_only
                    )
                    ctx.debug_last_collision = {
                        "kind": "terrain_raw_origin_fallback_contact",
                        "point": raw_fallback_contact.position,
                        "normal": raw_fallback_contact.normal,
                        "depth": raw_fallback_contact.penetration,
                        **contact_debug_fields(raw_fallback_contact),
                        "lifted_contact_missing": contact is None,
                        "lifted_contact_depth": (
                            None if contact is None else contact.penetration
                        ),
                        "raw_origin_fallback": True,
                        "tank_raw_origin_fallback": bool(tank_raw_fallback),
                        "raw_origin_fallback_reject": raw_fallback_reject,
                        "raw_origin_fallback_projection_order": raw_debug_projection_order,
                        "raw_origin_fallback_normal_source": raw_debug_normal_source,
                        "raw_origin_fallback_min_depth": raw_debug_min_depth,
                        "raw_origin_fallback_max_depth": raw_debug_max_depth,
                        "raw_origin_fallback_min_normal_z": raw_debug_min_normal_z,
                        "raw_origin_fallback_min_speed": raw_debug_min_speed,
                        "raw_origin_fallback_max_velocity_delta": raw_debug_max_velocity_delta,
                        "raw_origin_fallback_max_speed": raw_debug_max_speed,
                        "raw_origin_fallback_max_angular_delta": raw_debug_max_angular_delta,
                        "raw_origin_fallback_angular_mode": raw_debug_angular_mode,
                        "raw_origin_fallback_closing_only": raw_debug_closing_only,
                        "tank_raw_origin_fallback_delta_normal_mode": (
                            tank_raw_fallback_delta_normal_mode
                            if tank_raw_fallback and not raw_fallback_enabled
                            else None
                        ),
                        **(response_debug or {}),
                    }
                    ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                    return True
                raycast_contact = raycast_probe.get("contact") if isinstance(raycast_probe, dict) else None
                if (
                    raycast_fallback_enabled
                    and raycast_contact is not None
                    and raycast_probe.get("reject") == ""
                ):
                    response_debug, applied_raycast_fallback = apply_raw_origin_fallback_contact(raycast_contact)
                    if not applied_raycast_fallback:
                        return False
                    ctx.debug_last_collision = {
                        "kind": "terrain_raycast_fallback_contact",
                        "point": raycast_contact.position,
                        "normal": raycast_contact.normal,
                        "depth": raycast_contact.penetration,
                        **contact_debug_fields(raycast_contact),
                        "lifted_contact_missing": contact is None,
                        "lifted_contact_depth": (
                            None if contact is None else contact.penetration
                        ),
                        "raycast_fallback": True,
                        "terrain_raycast_fallback": True,
                        "raycast_fallback_reject": raycast_probe.get("reject"),
                        "raycast_fallback_probe_reason": (
                            "lifted_clear_raycast_contact"
                            if contact is None
                            else "lifted_below_slop_raycast_contact"
                        ),
                        "raycast_fallback_ray_start": raycast_probe.get("ray_start"),
                        "raycast_fallback_ray_end": raycast_probe.get("ray_end"),
                        "raycast_fallback_ray_length": raycast_probe.get("ray_length"),
                        "raycast_fallback_hit_position": raycast_probe.get("hit_position"),
                        "raycast_fallback_hit_distance": raycast_probe.get("hit_distance"),
                        **(response_debug or {}),
                    }
                    ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                    return True
                return False
            update_contact_probe(
                (anchor[0], anchor[1], anchor[2]),
                contact,
                reason="lifted_contact",
            )
            tank_face_contact, tank_face_reject = tank_clean_face_fallback_contact_for(
                contact
            )
            if tank_face_contact is None:
                tank_face_fallback_latch_clear(tank_face_reject)
            if tank_face_contact is not None and tank_face_reject == "":
                if tank_face_fallback_latch_matches(tank_face_contact):
                    ctx.debug_last_collision = tank_face_fallback_latched_debug(
                        tank_face_contact,
                        kind="terrain_clean_face_fallback_latched",
                    )
                    ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                    return True
                tank_face_delta_normal, tank_face_delta_normal_source = (
                    tank_clean_face_fallback_delta_normal_for(tank_face_contact)
                )
                response_debug, applied_tank_face_fallback = (
                    apply_raw_origin_fallback_contact(
                        tank_face_contact,
                        projection_order=tank_raw_fallback_projection_order,
                        delta_mode=tank_clean_face_fallback_delta_mode,
                        delta_normal=tank_face_delta_normal,
                        delta_normal_source=tank_face_delta_normal_source,
                        angular_mode=tank_raw_fallback_angular_mode,
                        closing_only=tank_raw_fallback_closing_only,
                        max_velocity_delta=tank_raw_fallback_max_velocity_delta,
                        max_vertical_delta=tank_raw_fallback_max_vertical_delta,
                        vertical_delta_mode=tank_raw_fallback_vertical_delta_mode,
                        max_speed=tank_raw_fallback_max_speed,
                        max_angular_delta=tank_raw_fallback_max_angular_delta,
                        friction=0.0,
                    )
                )
                if not applied_tank_face_fallback:
                    return False
                tank_face_fallback_latch_record(
                    tank_face_contact,
                    source="clean_face",
                )
                ctx.debug_last_collision = {
                    "kind": "terrain_clean_face_fallback_contact",
                    "point": tank_face_contact.position,
                    "normal": tank_face_contact.normal,
                    "depth": tank_face_contact.penetration,
                    "terrain_collision_shape": terrain_collision_shape,
                    **contact_debug_fields(tank_face_contact),
                    "tank_clean_face_fallback": True,
                    "tank_clean_face_fallback_reject": tank_face_reject,
                    "tank_clean_face_fallback_max_contact_normal_z": (
                        tank_clean_face_fallback_max_contact_normal_z
                    ),
                    "tank_clean_face_fallback_min_face_normal_z": (
                        tank_clean_face_fallback_min_face_normal_z
                    ),
                    "clean_contact_original_normal": contact.normal,
                    "clean_contact_original_normal_source": getattr(
                        contact,
                        "normal_source",
                        None,
                    ),
                    "clean_contact_original_depth": contact.penetration,
                    "tank_clean_face_fallback_delta_mode": (
                        tank_clean_face_fallback_delta_mode
                    ),
                    "tank_clean_face_fallback_delta_normal_mode": (
                        tank_clean_face_fallback_delta_normal_mode
                    ),
                    **(response_debug or {}),
                }
                ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                return True
            response_debug = apply_contact(contact)
            ctx.debug_last_collision = {
                "kind": "terrain_clean_contact",
                "point": contact.position,
                "normal": contact.normal,
                "depth": contact.penetration,
                "terrain_collision_shape": terrain_collision_shape,
                "tank_clean_face_fallback_reject": tank_face_reject,
                **contact_debug_fields(contact),
                **(response_debug or {}),
            }
            ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
            return True

        def resolve_dirty_contact():
            contact = sample_contact()
            if contact is None:
                return False
            apply_dirty_bounds_contact(contact)
            return True

        dirty_threshold_sq = self._get_entity_dirty_threshold_sq(ctx, half_extents)
        displacement_sq = (
            (anchor[0] - reference_pos[0]) * (anchor[0] - reference_pos[0]) +
            (anchor[1] - reference_pos[1]) * (anchor[1] - reference_pos[1]) +
            (anchor[2] - reference_pos[2]) * (anchor[2] - reference_pos[2])
        )
        raycast_fn = getattr(self._terrain_grid_collision, "raycast", None)
        ctx.world_collision_bounds_dirty = bool(
            callable(raycast_fn) and dirty_threshold_sq > 0.0 and displacement_sq >= dirty_threshold_sq
        )
        dirty_dispatch_debug.update({
            "dirty_bounds_active": bool(ctx.world_collision_bounds_dirty),
            "dirty_threshold_sq": dirty_threshold_sq,
            "dirty_displacement_sq": displacement_sq,
            "dirty_reference_pos": reference_pos,
            "dirty_current_pos": (anchor[0], anchor[1], anchor[2]),
            "dirty_miss_refresh_enabled": dirty_miss_refresh_enabled,
            "dirty_model_center_mode": dirty_model_center_mode,
            "dirty_bounds_box_fallback_enabled": dirty_bounds_box_fallback_enabled,
            "dirty_bounds_box_shape": dirty_bounds_box_shape,
            "dirty_bounds_box_center_mode": dirty_bounds_box_center_mode,
            "dirty_reference_pair_response_enabled": (
                dirty_reference_pair_response_enabled
            ),
            "dirty_reference_pair_response_apply_enabled": (
                dirty_reference_pair_response_apply_enabled
            ),
            "dirty_bounds_safe_response_enabled": dirty_bounds_safe_response_enabled,
        })
        dirty_bounds_overlap_fn = getattr(
            self._terrain_grid_collision,
            "test_bounds_intersection",
            None,
        )
        dirty_aabb_min = None
        dirty_aabb_max = None
        dirty_bounds_xy_overlap = None
        if ctx.world_collision_bounds_dirty and callable(dirty_bounds_overlap_fn):
            dirty_aabb_min = (
                anchor[0] - bounding_radius,
                anchor[1] - bounding_radius,
                anchor[2] - bounding_radius,
            )
            dirty_aabb_max = (
                anchor[0] + bounding_radius,
                anchor[1] + bounding_radius,
                anchor[2] + bounding_radius,
            )
            try:
                dirty_bounds_xy_overlap = bool(
                    dirty_bounds_overlap_fn(dirty_aabb_min, dirty_aabb_max)
                )
            except Exception as exc:
                dirty_dispatch_debug["dirty_bounds_xy_overlap_error"] = str(exc)
                if collision_model is None or terrain_collision_shape != "model":
                    raise
            dirty_dispatch_debug.update({
                "dirty_bounds_aabb_min": dirty_aabb_min,
                "dirty_bounds_aabb_max": dirty_aabb_max,
                "dirty_bounds_xy_overlap": dirty_bounds_xy_overlap,
            })
        if ctx.world_collision_bounds_dirty:
            dirty_contact_fn = None
            dirty_contact_args = None
            dirty_contact_source = None
            if collision_model is not None and terrain_collision_shape == "model":
                dirty_contact_fn = getattr(self._terrain_grid_collision, "test_model_bounds_contact", None)
                if callable(dirty_contact_fn):
                    dirty_collision_center = (
                        (anchor[0], anchor[1], anchor[2])
                        if dirty_model_center_mode in {
                            "raw",
                            "entity",
                            "origin",
                            "decompile",
                            "physics_state",
                        }
                        else (anchor[0], anchor[1], anchor[2] + z_lift)
                    )
                    dirty_contact_args = (
                        (anchor[0], anchor[1], anchor[2]),
                        dirty_collision_center,
                        heading,
                        vertices,
                        cbsp_tree,
                        bounding_radius,
                    )
                    dirty_dispatch_debug.update({
                        "dirty_model_bounds_center": dirty_contact_args[0],
                        "dirty_model_collision_center": dirty_contact_args[1],
                    })
                    dirty_contact_source = "model_bounds"
            else:
                dirty_contact_fn = getattr(self._terrain_grid_collision, "test_box_bounds_contact", None)
                if callable(dirty_contact_fn):
                    box_z_lift = box_collision_z_lift()
                    box_center = (
                        anchor[0],
                        anchor[1],
                        anchor[2] + box_z_lift,
                    )
                    dirty_contact_args = (
                        box_center,
                        box_center,
                        inertia_half_extents,
                        heading,
                        bounding_radius,
                    )
                    dirty_dispatch_debug.update({
                        "dirty_model_bounds_center": dirty_contact_args[0],
                        "dirty_model_collision_center": dirty_contact_args[1],
                    })
                    dirty_contact_source = "box_bounds"

            if dirty_contact_args is not None:
                if collision_model is not None and terrain_collision_shape == "model":
                    contact = dirty_contact_fn(
                        *dirty_contact_args,
                        rotation_matrix=model_contact_rotation_matrix,
                        contact_selection=model_contact_selection,
                    )
                else:
                    contact = dirty_contact_fn(*dirty_contact_args)
                if (
                    contact is None
                    and dirty_contact_source == "model_bounds"
                    and dirty_bounds_box_fallback_enabled
                ):
                    box_contact_fn = getattr(
                        self._terrain_grid_collision,
                        "test_box_bounds_contact",
                        None,
                    )
                    dirty_dispatch_debug["dirty_bounds_box_fallback_attempted"] = True
                    if callable(box_contact_fn):
                        box_half_extents, box_half_extents_source = (
                            dirty_bounds_box_half_extents()
                        )
                        box_center, box_center_mode, box_z_offset = (
                            dirty_bounds_box_center()
                        )
                        box_args = (
                            (anchor[0], anchor[1], anchor[2]),
                            box_center,
                            box_half_extents,
                            heading,
                            bounding_radius,
                        )
                        dirty_dispatch_debug.update({
                            "dirty_bounds_box_half_extents_source": (
                                box_half_extents_source
                            ),
                            "dirty_bounds_box_half_extents": box_half_extents,
                            "dirty_bounds_box_center_mode": box_center_mode,
                            "dirty_bounds_box_z_offset": box_z_offset,
                            "dirty_bounds_box_bounds_center": box_args[0],
                            "dirty_bounds_box_collision_center": box_args[1],
                        })
                        contact = box_contact_fn(
                            *box_args,
                            rotation_matrix=model_contact_rotation_matrix,
                            contact_selection=pair_record_contact_selection,
                        )
                        if contact is not None:
                            dirty_contact_source = "box_bounds_fallback"
                            dirty_contact_args = box_args
                            dirty_dispatch_debug[
                                "dirty_bounds_box_fallback_applied"
                            ] = True
                            dirty_dispatch_debug[
                                "dirty_bounds_box_contact"
                            ] = probe_contact_fields(
                                contact,
                                center=box_center,
                                z_lift_used=box_z_offset,
                            )
                        else:
                            dirty_dispatch_debug[
                                "dirty_bounds_box_reject"
                            ] = "no_box_bounds_contact"
                    else:
                        dirty_dispatch_debug[
                            "dirty_bounds_box_reject"
                        ] = "no_box_bounds_contact_fn"
                if contact is not None:
                    dirty_dispatch_debug["dirty_bounds_contact_source"] = (
                        dirty_contact_source
                    )
                    if self._is_pathological_dirty_bounds_contact((anchor[0], anchor[1], anchor[2]), contact, bounding_radius):
                        ctx.debug_last_collision = {
                            "kind": "terrain_dirty_bounds_filtered",
                            "point": contact.position,
                            "normal": contact.normal,
                            "depth": contact.penetration,
                            **contact_debug_fields(contact),
                            "dirty_model_center_mode": dirty_model_center_mode,
                            "dirty_bounds_center": dirty_contact_args[0],
                            "dirty_collision_center": dirty_contact_args[1],
                            "dirty_bounds_contact_source": dirty_contact_source,
                            "dirty_bounds_box_fallback_enabled": (
                                dirty_bounds_box_fallback_enabled
                            ),
                            "dirty_bounds_box_fallback_attempted": (
                                dirty_dispatch_debug.get(
                                    "dirty_bounds_box_fallback_attempted"
                                )
                            ),
                            "dirty_bounds_box_fallback_applied": (
                                dirty_dispatch_debug.get(
                                    "dirty_bounds_box_fallback_applied"
                                )
                            ),
                            "dirty_bounds_box_shape": dirty_dispatch_debug.get(
                                "dirty_bounds_box_shape"
                            ),
                            "dirty_bounds_box_half_extents_source": (
                                dirty_dispatch_debug.get(
                                    "dirty_bounds_box_half_extents_source"
                                )
                            ),
                            "dirty_bounds_box_center_mode": (
                                dirty_dispatch_debug.get(
                                    "dirty_bounds_box_center_mode"
                                )
                            ),
                            "detail": f"reference={reference_pos!r}",
                            "dirty_threshold_sq": dirty_threshold_sq,
                            "dirty_displacement_sq": displacement_sq,
                        }
                        dirty_dispatch_debug["dirty_miss_reason"] = "dirty_bounds_filtered"
                    else:
                        apply_dirty_bounds_contact(contact)
                        ctx.debug_last_collision.update({
                            "dirty_model_center_mode": dirty_model_center_mode,
                            "dirty_bounds_center": dirty_contact_args[0],
                            "dirty_collision_center": dirty_contact_args[1],
                            "dirty_bounds_contact_source": dirty_contact_source,
                            "dirty_bounds_box_fallback_enabled": (
                                dirty_bounds_box_fallback_enabled
                            ),
                            "dirty_bounds_box_fallback_attempted": (
                                dirty_dispatch_debug.get(
                                    "dirty_bounds_box_fallback_attempted"
                                )
                            ),
                            "dirty_bounds_box_fallback_applied": (
                                dirty_dispatch_debug.get(
                                    "dirty_bounds_box_fallback_applied"
                                )
                            ),
                            "dirty_bounds_box_shape": dirty_dispatch_debug.get(
                                "dirty_bounds_box_shape"
                            ),
                            "dirty_bounds_box_half_extents_source": (
                                dirty_dispatch_debug.get(
                                    "dirty_bounds_box_half_extents_source"
                                )
                            ),
                            "dirty_bounds_box_center_mode": (
                                dirty_dispatch_debug.get(
                                    "dirty_bounds_box_center_mode"
                                )
                            ),
                            "model_contact_selection": model_contact_selection,
                            "dirty_threshold_sq": dirty_threshold_sq,
                            "dirty_displacement_sq": displacement_sq,
                        })
                        ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                        update_contact_probe(
                            (anchor[0], anchor[1], anchor[2]),
                            None,
                            reason=f"{dirty_contact_source}_contact",
                            raw_error="dirty_bounds_contact",
                        )
                        ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
                        return finish_result(anchor[0], anchor[1], anchor[2], vx, vy, vz)
                else:
                    if dirty_dispatch_debug.get("dirty_bounds_box_reject"):
                        dirty_dispatch_debug[
                            "dirty_miss_reason"
                        ] = "dirty_bounds_clear_box_clear"
                    else:
                        dirty_dispatch_debug["dirty_miss_reason"] = "dirty_bounds_clear"
            else:
                if callable(dirty_bounds_overlap_fn):
                    if dirty_bounds_xy_overlap:
                        if resolve_dirty_contact():
                            ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
                            return finish_result(anchor[0], anchor[1], anchor[2], vx, vy, vz)
                        dirty_dispatch_debug[
                            "dirty_miss_reason"
                        ] = "dirty_bounds_overlap_clean_contact_clear"
                    else:
                        dirty_dispatch_debug[
                            "dirty_miss_reason"
                        ] = "dirty_bounds_overlap_clear"
                elif resolve_dirty_contact():
                    ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
                    return finish_result(anchor[0], anchor[1], anchor[2], vx, vy, vz)
                else:
                    dirty_dispatch_debug["dirty_miss_reason"] = "dirty_clean_contact_clear"
            dirty_reference_pair = record_dirty_reference_pair_probe()
            if resolve_dirty_reference_pair_response(dirty_reference_pair):
                update_contact_probe(
                    (anchor[0], anchor[1], anchor[2]),
                    None,
                    reason="dirty_reference_pair_response",
                    raw_error="dirty_bounds_contact",
                )
                ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
                return finish_result(anchor[0], anchor[1], anchor[2], vx, vy, vz)
            terrain_hit = raycast_fn(reference_pos, (anchor[0], anchor[1], anchor[2]))
            ray_dir = (
                anchor[0] - reference_pos[0],
                anchor[1] - reference_pos[1],
                anchor[2] - reference_pos[2],
            )
            ray_dir_len = math.sqrt(
                ray_dir[0] * ray_dir[0] +
                ray_dir[1] * ray_dir[1] +
                ray_dir[2] * ray_dir[2]
            )
            dirty_dispatch_debug.update({
                "dirty_raycast_start": reference_pos,
                "dirty_raycast_end": (anchor[0], anchor[1], anchor[2]),
                "dirty_raycast_length": ray_dir_len,
            })
            if terrain_hit is not None:
                dirty_dispatch_debug.update({
                    "dirty_raycast_hit_position": terrain_hit.position,
                    "dirty_raycast_hit_cell": terrain_hit.cell,
                })
                contact_normal = (
                    terrain_hit.normal[0],
                    terrain_hit.normal[1],
                    terrain_hit.normal[2],
                )
                if ray_dir_len <= 0.001:
                    contact_point = terrain_hit.position
                    separation = self._get_static_separation_from_contact(
                        (anchor[0], anchor[1], anchor[2]),
                        contact_point,
                    )
                    anchor[0] = terrain_hit.position[0] + contact_normal[0] * (bounding_radius + separation)
                    anchor[1] = terrain_hit.position[1] + contact_normal[1] * (bounding_radius + separation)
                    anchor[2] = terrain_hit.position[2] + contact_normal[2] * (bounding_radius + separation)
                else:
                    ray_scale = bounding_radius / ray_dir_len
                    scaled_dir = (
                        ray_dir[0] * ray_scale,
                        ray_dir[1] * ray_scale,
                        ray_dir[2] * ray_scale,
                    )
                    contact_point = (
                        anchor[0] + scaled_dir[0],
                        anchor[1] + scaled_dir[1],
                        anchor[2] + scaled_dir[2],
                    )
                    separation = self._get_static_separation_from_contact(
                        (anchor[0], anchor[1], anchor[2]),
                        contact_point,
                    )
                    anchor[0] = terrain_hit.position[0] - scaled_dir[0] + contact_normal[0] * separation
                    anchor[1] = terrain_hit.position[1] - scaled_dir[1] + contact_normal[1] * separation
                    anchor[2] = terrain_hit.position[2] - scaled_dir[2] + contact_normal[2] * separation

                vel_dot = (
                    vx * contact_normal[0] +
                    vy * contact_normal[1] +
                    vz * contact_normal[2]
                )
                if vel_dot < 0.0:
                    vx -= contact_normal[0] * vel_dot
                    vy -= contact_normal[1] * vel_dot
                    vz -= contact_normal[2] * vel_dot
                ctx.debug_last_collision = {
                    "kind": "terrain_dirty_raycast",
                    "point": terrain_hit.position,
                    "normal": contact_normal,
                    "depth": separation,
                    "contact_sector_index": terrain_hit.sector_index,
                    "contact_cell": terrain_hit.cell,
                    "detail": f"reference={reference_pos!r}",
                    "dirty_threshold_sq": dirty_threshold_sq,
                    "dirty_displacement_sq": displacement_sq,
                    "dirty_raycast_length": ray_dir_len,
                }
                ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
                return finish_result(anchor[0], anchor[1], anchor[2], vx, vy, vz)
            dirty_dispatch_debug["dirty_raycast_reject"] = "no_terrain_raycast_hit"
            if dirty_miss_refresh_enabled:
                ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
                dirty_dispatch_debug["dirty_miss_ref_action"] = "refreshed"
            else:
                dirty_dispatch_debug["dirty_miss_ref_action"] = "preserved"
            # Dirty terrain dispatch can miss when the lifted model is clear and
            # the center-to-center ray remains above the height field. Keep the
            # default reference refresh, but allow an opt-in hold probe to test
            # whether the OG path keeps the secondary spatial point long enough
            # for a later dirty ray to span the contact.

        clean_contact_resolved = resolve_single_contact()
        # Keep the dirty-reference anchored to the latest clean-resolution pose.
        # Otherwise repeated flat-ground clamp contacts accumulate displacement
        # against an old reference and falsely enter the dirty terrain-ray path.
        # A lifted clear must not refresh this reference: OG's bounds-dirty flag
        # compares current position against the last recorded physics position.
        if clean_contact_resolved:
            ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])

        return finish_result(anchor[0], anchor[1], anchor[2], vx, vy, vz)

    def _resolve_entity_entity_collisions(self, ctx: ClientContext):
        """Resolve entity-entity collisions using impulse-based sphere-sphere detection.

        Decompile: Physics.c — time-bucketed deferred collision pairs with impulse dynamics.
        Simplified here to per-tick sphere overlap + impulse response using the verified
        collision table (exe VA 0x5730C0). Each entity independently resolves its own
        mass-proportional share of the collision (position and velocity).

        Response: J = -(1 + e) * v_rel·n / (1/m_a + 1/m_b)
        where e = avg(elasticity_a, elasticity_b), n = collision normal.
        Position separation split by inverse mass ratio.
        """
        if not ctx.session or not ctx.session.in_game:
            return

        pos_a = ctx.player_pos
        vel_a = ctx.player_vel
        radius_a = self._TANK_RADIUS
        col_a = self._ENTITY_COLLISION_TABLE.get(ctx.entity_type, self._ENTITY_COLLISION_DEFAULT)
        mass_a = col_a["mass"]

        for other in self._snapshot_in_game_clients():
            if other is ctx:
                continue
            if not other.session or not other.session.in_game:
                continue

            pos_b = other.player_pos
            radius_b = self._TANK_RADIUS

            # Sphere-sphere overlap test (XY plane + Z)
            dx = pos_a[0] - pos_b[0]
            dy = pos_a[1] - pos_b[1]
            dz = pos_a[2] - pos_b[2]
            dist_sq = dx * dx + dy * dy + dz * dz
            combined_radius = radius_a + radius_b
            if dist_sq >= combined_radius * combined_radius:
                continue

            dist = math.sqrt(dist_sq)
            if dist < 0.001:
                # Perfectly overlapping — use arbitrary separation direction
                dx, dy, dz = 1.0, 0.0, 0.0
                dist = 0.001

            # Collision normal: A -> B direction (pushes A away from B)
            inv_dist = 1.0 / dist
            nx = dx * inv_dist
            ny = dy * inv_dist
            nz = dz * inv_dist

            penetration = combined_radius - dist

            col_b = self._ENTITY_COLLISION_TABLE.get(other.entity_type, self._ENTITY_COLLISION_DEFAULT)
            mass_b = col_b["mass"]
            elasticity = (col_a["elasticity"] + col_b["elasticity"]) * 0.5

            vel_b = other.player_vel

            # Relative velocity along collision normal
            rel_vx = vel_a[0] - vel_b[0]
            rel_vy = vel_a[1] - vel_b[1]
            rel_vz = vel_a[2] - vel_b[2]
            v_rel_n = rel_vx * nx + rel_vy * ny + rel_vz * nz

            # Only resolve if entities are approaching (negative = separating)
            # But always do position separation
            inv_mass_sum = 1.0 / mass_a + 1.0 / mass_b

            if v_rel_n > 0.0:
                # Impulse magnitude (decompile: J = -(1+e) * v_rel·n / (1/m_a + 1/m_b))
                j = -(1.0 + elasticity) * v_rel_n / inv_mass_sum

                # Apply impulse to this entity only (A gets pushed along +normal)
                impulse_a = j / mass_a
                new_vx = vel_a[0] + nx * impulse_a
                new_vy = vel_a[1] + ny * impulse_a
                new_vz = vel_a[2] + nz * impulse_a

                # Safety cap: decompile caps impulse magnitude at 200.0
                speed_sq = new_vx * new_vx + new_vy * new_vy + new_vz * new_vz
                if speed_sq > 200.0 * 200.0:
                    scale = 200.0 / math.sqrt(speed_sq)
                    new_vx *= scale
                    new_vy *= scale
                    new_vz *= scale

                vel_a = (new_vx, new_vy, new_vz)

            # Position separation: push this entity's share of the penetration
            # Each entity gets pushed proportional to inverse mass
            share_a = (1.0 / mass_a) / inv_mass_sum
            push = penetration * share_a + 0.1  # small separation buffer
            new_x = pos_a[0] + nx * push
            new_y = pos_a[1] + ny * push
            new_z = pos_a[2] + nz * push

            # Clamp Z to terrain
            if self.up_axis == "z" and self.terrain:
                terrain_z = self._terrain_physics_ground_z_at(new_x, new_y)
                if new_z < terrain_z:
                    new_z = terrain_z

            pos_a = (new_x, new_y, new_z)

            ctx.debug_last_collision = {
                "kind": "entity_entity",
                "other_id": other.client_id,
                "penetration": penetration,
                "normal": (nx, ny, nz),
                "dist": dist,
            }

        # Write back if changed
        if pos_a is not ctx.player_pos:
            ctx.player_pos = pos_a
            ctx.player_vel = vel_a
            ctx.player_pose["pos"] = pos_a
            ctx.player_pose["vel"] = vel_a

    def _get_entity_world_collision_model(self, ctx: ClientContext):
        team_id = ctx.session.team_id or 1
        origin_mode = (
            os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", "lift").strip().lower()
        )
        model_variant = (
            os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_VARIANT", "primary")
            .strip()
            .lower()
        )
        model_name_override = os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_NAME", "").strip()
        cache_key = (
            ctx.entity_type,
            team_id,
            origin_mode,
            model_variant,
            model_name_override,
        )
        if cache_key in self._entity_collision_model_cache:
            return self._entity_collision_model_cache[cache_key]

        model_names = self._ENTITY_WORLD_MODEL_NAMES.get(ctx.entity_type)
        if not model_names or not self._building_collision.available:
            self._entity_collision_model_cache[cache_key] = None
            return None

        model_name = model_name_override or self._select_team_model_name(model_names, team_id)
        if model_variant in {"simplified", "simple", "s", "_s"} and model_name:
            model_name = model_name if model_name.endswith("_s") else f"{model_name}_s"
        model = self._building_collision.models.get(model_name)
        mesh = getattr(model, "collision_mesh", None) if model is not None else None
        vertices = getattr(mesh, "vertices", None) if mesh is not None else None
        cbsp_tree = getattr(model, "cbsp_tree", None) if model is not None else None
        if not vertices or cbsp_tree is None or not cbsp_tree.nodes:
            self._entity_collision_model_cache[cache_key] = None
            return None

        root = cbsp_tree.root
        bounding_radius = root.radius if root is not None else 0.0
        min_z = min(getattr(vertex, "z", 0.0) for vertex in vertices)
        if origin_mode in {"entity", "origin", "raw"}:
            # Experimental decompile-backed transform: terrain contact tests
            # use entity.pos directly. Keep this opt-in until live rough-terrain
            # gates prove the pair/timed response path stable.
            z_lift = 0.0
        else:
            z_lift = max(0.0, -float(min_z))
        result = (vertices, cbsp_tree, bounding_radius, z_lift)
        self._entity_collision_model_cache[cache_key] = result
        return result

    def _record_client_action_telemetry(
        self,
        ctx: ClientContext,
        packet_type: str,
        client_tick: int,
    ) -> None:
        """Record decoded client input so live control-plane checks are unambiguous."""
        if ctx is None or ctx.weapon_system is None:
            return

        now = time.monotonic()
        ws = ctx.weapon_system
        turn_input = self._get_raw_turn_input(ctx)
        fwd_input = self._normalize_behavior_axis_value(
            ctx,
            ws.behavior_slots[BehaviorSlot.MOVING_FORWARD],
        )
        strafe_input = self._decode_network_strafe_input(
            ctx,
            ws.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS],
        )
        fire_input = ws.behavior_slots[BehaviorSlot.FIRE]
        thrust_input = tank_softbody_control_slot_value(ws.behavior_slots)
        jumpjet_input = ws.behavior_slots[BehaviorSlot.JUMPJET]
        active_slots = {
            str(idx): float(value)
            for idx, value in enumerate(ws.behavior_slots)
            if abs(float(value)) > 0.001
        }

        ctx.action_packet_count += 1
        if packet_type == "ACTION_UPDATE":
            ctx.action_update_count += 1
        elif packet_type == "ACTION_DUMP":
            ctx.action_dump_count += 1
        ctx.last_action_packet_time = now
        ctx.last_action_packet_type = packet_type
        ctx.last_action_packet_client_tick = client_tick
        history = getattr(ctx, "movement_input_history", None)
        if history is not None:
            history.append(
                {
                    "time": now,
                    "fwd": float(fwd_input),
                    "strafe": float(strafe_input),
                    "packet_type": packet_type,
                    "client_tick": int(client_tick or 0),
                    "action_sequence": int(ctx.action_packet_count),
                }
            )
        ctx.last_decoded_input = {
            "turn": float(turn_input),
            "fwd": float(fwd_input),
            "strafe": float(strafe_input),
            "fire": float(fire_input),
            "thrust": float(thrust_input),
            "jumpjet": float(jumpjet_input),
            "active_slots": active_slots,
        }
        if abs(fwd_input) > 0.05 or abs(strafe_input) > 0.05:
            ctx.nonzero_move_input_count += 1
            ctx.last_nonzero_move_input_time = now

    def _select_delayed_movement_input(
        self,
        ctx: ClientContext,
        *,
        current_fwd: float,
        current_strafe: float,
        delay_s: float,
    ) -> tuple[float, float, str]:
        """Replay remote OG movement slots at the phase the local client applies."""
        delay = max(0.0, float(delay_s))
        now = time.monotonic()
        debug = {
            "delay_s": delay,
            "current_fwd": float(current_fwd),
            "current_strafe": float(current_strafe),
            "target_age_s": delay,
        }
        if delay <= 0.0:
            ctx.debug_last_movement_input_selection = {
                **debug,
                "source": "current_slots",
                "history_len": len(getattr(ctx, "movement_input_history", []) or []),
                "selected_age_s": 0.0,
            }
            return float(current_fwd), float(current_strafe), "current_slots"
        history = getattr(ctx, "movement_input_history", None)
        if not history:
            ctx.debug_last_movement_input_selection = {
                **debug,
                "source": "current_slots_no_history",
                "history_len": 0,
                "selected_age_s": 0.0,
            }
            return float(current_fwd), float(current_strafe), "current_slots_no_history"
        # Snapshot the deque: the UDP receive thread appends ACTION_UPDATE movement
        # samples to `movement_input_history` concurrently with this tick-thread
        # iteration. GOAL 8's position sub-stepping calls this up to 3x per tick,
        # widening that window — without a copy, a concurrent append raises
        # "deque mutated during iteration" and crashes the client's tick (seen live
        # on DO under real network jitter). A snapshot is also correct: selection
        # must be consistent within one frame.
        history = list(history)

        def entry_tick(entry) -> int | None:
            try:
                tick = int(entry.get("client_tick", 0) or 0)
            except (TypeError, ValueError, AttributeError, OverflowError):
                return None
            return tick if tick > 0 else None

        def build_tick_context(enabled: bool) -> dict:
            if not enabled:
                return {"debug": {}}
            try:
                server_tick_now = int(get_ticks()) & 0xFFFFFFFF
            except (TypeError, ValueError, OverflowError):
                server_tick_now = 0
            tick_offset = getattr(ctx, "tick_offset", None)
            current_client_tick = int(getattr(ctx, "last_client_tick", 0) or 0)
            if tick_offset is not None:
                try:
                    current_client_tick = (
                        server_tick_now + int(tick_offset)
                    ) & 0xFFFFFFFF
                except (TypeError, ValueError, OverflowError):
                    current_client_tick = int(
                        getattr(ctx, "last_client_tick", 0) or 0
                    )
            if current_client_tick <= 0:
                return {
                    "debug": {
                        "tick_probe_enabled": True,
                        "tick_probe_reject": "no_current_client_tick",
                        "tick_probe_server_tick": server_tick_now,
                        "tick_probe_tick_offset": tick_offset,
                        "tick_probe_last_client_tick": int(
                            getattr(ctx, "last_client_tick", 0) or 0
                        ),
                    },
                    "reject": "no_current_client_tick",
                }
            delay_ms = max(0, int(round(delay * 1000.0)))
            target_client_tick = (current_client_tick - delay_ms) & 0xFFFFFFFF

            before = None
            nearest = None
            nearest_abs_delta = None
            for entry in reversed(history):
                tick = entry_tick(entry)
                if tick is None:
                    continue
                delta_ms = self._tick_delta_signed(tick, target_client_tick)
                abs_delta = abs(delta_ms)
                if nearest_abs_delta is None or abs_delta < nearest_abs_delta:
                    nearest = entry
                    nearest_abs_delta = abs_delta
                if before is None and delta_ms <= 0:
                    before = entry
            latest = history[-1] if history else {}
            latest_tick = entry_tick(latest) if isinstance(latest, dict) else None

            def fields(prefix: str, entry) -> dict:
                tick = entry_tick(entry) if isinstance(entry, dict) else None
                if tick is None:
                    return {f"{prefix}_found": False}
                delta_ms = self._tick_delta_signed(tick, target_client_tick)
                return {
                    f"{prefix}_found": True,
                    f"{prefix}_client_tick": tick,
                    f"{prefix}_target_error_ms": delta_ms,
                    f"{prefix}_abs_target_error_ms": abs(delta_ms),
                    f"{prefix}_future_of_target": delta_ms > 0,
                    f"{prefix}_fwd": float(entry.get("fwd", 0.0)),
                    f"{prefix}_strafe": float(entry.get("strafe", 0.0)),
                }

            out = {
                "tick_probe_enabled": True,
                "tick_probe_reject": "",
                "tick_probe_server_tick": server_tick_now,
                "tick_probe_tick_offset": tick_offset,
                "tick_probe_current_client_tick": current_client_tick,
                "tick_probe_last_client_tick": int(
                    getattr(ctx, "last_client_tick", 0) or 0
                ),
                "tick_probe_delay_ms": delay_ms,
                "tick_probe_target_client_tick": target_client_tick,
                "tick_probe_latest_client_tick": latest_tick,
            }
            out.update(fields("tick_probe_before", before))
            out.update(fields("tick_probe_nearest", nearest))
            return {
                "debug": out,
                "before": before,
                "nearest": nearest,
                "target_client_tick": target_client_tick,
            }

        target_time = now - delay
        selection_policy = getattr(
            self,
            "remote_og_movement_input_selection",
            "latest_before_target",
        )
        if selection_policy not in {
            "latest_before_target",
            "nearest_to_target",
            "bounded_after_target",
            "latest_before_tick_target",
            "nearest_tick_target",
        }:
            selection_policy = "latest_before_target"
        tick_selection_enabled = selection_policy in {
            "latest_before_tick_target",
            "nearest_tick_target",
        }
        tick_context = build_tick_context(
            bool(getattr(self, "remote_og_movement_input_tick_probe", False))
            or tick_selection_enabled
        )
        tick_probe = tick_context.get("debug") if isinstance(tick_context, dict) else {}
        before = None
        after = None
        nearest = None
        nearest_error = None
        for entry in history:
            try:
                sample_time = float(entry.get("time", 0.0))
            except (TypeError, ValueError, AttributeError):
                continue
            error = abs(sample_time - target_time)
            if nearest_error is None or error < nearest_error:
                nearest = entry
                nearest_error = error
            if sample_time <= target_time:
                before = entry
            elif after is None:
                after = entry

        def entry_axis_delta(entry, fwd, strafe) -> float | None:
            if not isinstance(entry, dict):
                return None
            try:
                entry_fwd = float(entry.get("fwd", 0.0))
                entry_strafe = float(entry.get("strafe", 0.0))
            except (TypeError, ValueError, AttributeError, OverflowError):
                return None
            return abs(entry_fwd - float(fwd)) + abs(entry_strafe - float(strafe))

        def entry_target_error(entry) -> float | None:
            if not isinstance(entry, dict):
                return None
            try:
                sample_time = float(entry.get("time", 0.0))
            except (TypeError, ValueError, AttributeError, OverflowError):
                return None
            return sample_time - target_time if sample_time > 0.0 else None

        def entry_has_movement(entry, *, threshold: float = 0.05) -> bool:
            if not isinstance(entry, dict):
                return False
            try:
                entry_fwd = float(entry.get("fwd", 0.0))
                entry_strafe = float(entry.get("strafe", 0.0))
            except (TypeError, ValueError, AttributeError, OverflowError):
                return False
            return abs(entry_fwd) > threshold or abs(entry_strafe) > threshold

        def movement_time_probe_fields(prefix: str, entry) -> dict:
            if not isinstance(entry, dict):
                return {f"{prefix}_found": False}
            try:
                sample_time = float(entry.get("time", 0.0))
            except (TypeError, ValueError, AttributeError):
                return {f"{prefix}_found": False}
            target_error = sample_time - target_time
            return {
                f"{prefix}_found": True,
                f"{prefix}_age_s": (now - sample_time) if sample_time > 0.0 else None,
                f"{prefix}_target_error_s": target_error,
                f"{prefix}_abs_target_error_s": abs(target_error),
                f"{prefix}_future_of_target": target_error > 0.0,
                f"{prefix}_fwd": float(entry.get("fwd", 0.0)),
                f"{prefix}_strafe": float(entry.get("strafe", 0.0)),
                f"{prefix}_client_tick": int(entry.get("client_tick", 0) or 0),
            }

        def movement_history_window_fields(selected_entry) -> dict:
            window_s = 0.75
            max_entries = 16
            candidates = []
            for index, entry in enumerate(history):
                if not isinstance(entry, dict):
                    continue
                try:
                    sample_time = float(entry.get("time", 0.0))
                except (TypeError, ValueError, AttributeError, OverflowError):
                    continue
                if sample_time <= 0.0:
                    continue
                target_error = sample_time - target_time
                if abs(target_error) > window_s:
                    continue
                try:
                    entry_fwd = float(entry.get("fwd", 0.0))
                    entry_strafe = float(entry.get("strafe", 0.0))
                except (TypeError, ValueError, AttributeError, OverflowError):
                    entry_fwd = 0.0
                    entry_strafe = 0.0
                candidates.append(
                    {
                        "index": index,
                        "time": sample_time,
                        "target_error_s": target_error,
                        "abs_target_error_s": abs(target_error),
                        "age_s": now - sample_time,
                        "fwd": entry_fwd,
                        "strafe": entry_strafe,
                        "client_tick": int(entry.get("client_tick", 0) or 0),
                        "packet_type": entry.get("packet_type"),
                        "selected": entry is selected_entry,
                    }
                )
            total = len(candidates)
            if total > max_entries:
                selected_index = None
                for idx, item in enumerate(candidates):
                    if item.get("selected"):
                        selected_index = idx
                        break
                if selected_index is None:
                    keep = sorted(
                        candidates,
                        key=lambda item: (
                            float(item.get("abs_target_error_s", 999.0)),
                            int(item.get("index", 0)),
                        ),
                    )[:max_entries]
                    keep_indices = {int(item["index"]) for item in keep}
                else:
                    half = max_entries // 2
                    start = max(0, selected_index - half)
                    end = min(total, start + max_entries)
                    start = max(0, end - max_entries)
                    keep_indices = {
                        int(item["index"]) for item in candidates[start:end]
                    }
                candidates = [
                    item for item in candidates if int(item["index"]) in keep_indices
                ]
            compact = []
            for item in sorted(candidates, key=lambda row: int(row.get("index", 0))):
                compact.append(
                    {
                        "index": int(item["index"]),
                        "age_s": float(item["age_s"]),
                        "target_error_s": float(item["target_error_s"]),
                        "abs_target_error_s": float(item["abs_target_error_s"]),
                        "future_of_target": float(item["target_error_s"]) > 0.0,
                        "fwd": float(item["fwd"]),
                        "strafe": float(item["strafe"]),
                        "client_tick": int(item["client_tick"]),
                        "packet_type": item.get("packet_type"),
                        "selected": bool(item.get("selected")),
                    }
                )
            return {
                "movement_history_window_s": window_s,
                "movement_history_window_count": len(compact),
                "movement_history_window_total_count": total,
                "movement_history_window_truncated": total > max_entries,
                "movement_history_window": compact,
            }

        nonzero_before = None
        nonzero_after = None
        nonzero_nearest = None
        nonzero_nearest_error = None
        for entry in history:
            if not entry_has_movement(entry):
                continue
            try:
                sample_time = float(entry.get("time", 0.0))
            except (TypeError, ValueError, AttributeError, OverflowError):
                continue
            error = abs(sample_time - target_time)
            if nonzero_nearest_error is None or error < nonzero_nearest_error:
                nonzero_nearest = entry
                nonzero_nearest_error = error
            if sample_time <= target_time:
                nonzero_before = entry
            elif nonzero_after is None:
                nonzero_after = entry

        selected = None
        bounded_after_applied = False
        bounded_after_reason = ""
        bounded_after_max = max(
            0.0,
            float(getattr(self, "remote_og_movement_input_after_max", 0.20) or 0.0),
        )
        nonzero_after_target_error = entry_target_error(nonzero_after)
        nonzero_after_within_bounded_after_max = (
            nonzero_after_target_error is not None
            and nonzero_after_target_error <= bounded_after_max
        )
        if selection_policy == "nearest_to_target":
            selected = nearest
        elif selection_policy == "bounded_after_target":
            selected = before
            after_target_error = entry_target_error(after)
            before_current_delta = entry_axis_delta(before, current_fwd, current_strafe)
            after_current_delta = entry_axis_delta(after, current_fwd, current_strafe)
            after_is_bounded = (
                after_target_error is not None
                and after_target_error >= 0.0
                and after_target_error <= bounded_after_max
            )
            after_moves_toward_current = (
                after_current_delta is not None
                and (
                    before_current_delta is None
                    or after_current_delta + 1e-6 < before_current_delta
                )
            )
            if after_is_bounded and after_moves_toward_current:
                selected = after
                bounded_after_applied = True
                bounded_after_reason = "after_sample_within_bound_matches_current_input"
            else:
                if after is None:
                    bounded_after_reason = "no_after_sample"
                elif not after_is_bounded:
                    bounded_after_reason = "after_sample_outside_bound"
                elif not after_moves_toward_current:
                    bounded_after_reason = "after_sample_not_closer_to_current_input"
        elif selection_policy == "latest_before_tick_target":
            selected = tick_context.get("before") if isinstance(tick_context, dict) else None
        elif selection_policy == "nearest_tick_target":
            selected = tick_context.get("nearest") if isinstance(tick_context, dict) else None
        else:
            selected = before

        if selected is None:
            # The first nonzero packet can arrive before OG local physics has
            # consumed the key event. Hold neutral during that short replay
            # window instead of advancing the server early.
            latest = history[-1] if history else {}
            latest_time = 0.0
            try:
                latest_time = float(latest.get("time", 0.0))
            except (TypeError, ValueError, AttributeError):
                latest_time = 0.0
            ctx.debug_last_movement_input_selection = {
                **debug,
                **tick_probe,
                **movement_time_probe_fields("time_probe_before", before),
                **movement_time_probe_fields("time_probe_after", after),
                **movement_time_probe_fields("time_probe_nearest", nearest),
                **movement_time_probe_fields(
                    "time_probe_nonzero_before", nonzero_before
                ),
                **movement_time_probe_fields(
                    "time_probe_nonzero_after", nonzero_after
                ),
                **movement_time_probe_fields(
                    "time_probe_nonzero_nearest", nonzero_nearest
                ),
                **movement_history_window_fields(None),
                "source": "delayed_remote_og_pre_history_zero",
                "selection_policy": selection_policy,
                "bounded_after_max_s": bounded_after_max,
                "bounded_after_applied": bounded_after_applied,
                "bounded_after_reason": bounded_after_reason,
                "nonzero_after_within_bounded_after_max": (
                    nonzero_after_within_bounded_after_max
                ),
                "history_len": len(history),
                "target_time": target_time,
                "latest_age_s": (now - latest_time) if latest_time > 0.0 else None,
                "latest_fwd": float(latest.get("fwd", 0.0)) if isinstance(latest, dict) else 0.0,
                "latest_strafe": float(latest.get("strafe", 0.0)) if isinstance(latest, dict) else 0.0,
                "latest_client_tick": int(latest.get("client_tick", 0) or 0) if isinstance(latest, dict) else 0,
            }
            return 0.0, 0.0, "delayed_remote_og_pre_history_zero"

        try:
            fwd = float(selected.get("fwd", 0.0))
        except (TypeError, ValueError, AttributeError):
            fwd = 0.0
        try:
            strafe = float(selected.get("strafe", 0.0))
        except (TypeError, ValueError, AttributeError):
            strafe = 0.0
        selected_time = 0.0
        try:
            selected_time = float(selected.get("time", 0.0))
        except (TypeError, ValueError, AttributeError):
            selected_time = 0.0
        selected_target_error = selected_time - target_time if selected_time > 0.0 else None
        stale_clamp = max(
            0.0,
            float(getattr(self, "remote_og_movement_input_stale_clamp", 0.0) or 0.0),
        )
        stale_clamp_applied = False
        stale_clamp_reason = ""
        if (
            stale_clamp > 0.0
            and selected_target_error is not None
            and selected_target_error < -stale_clamp
        ):
            current_changed = (
                abs(float(current_fwd) - fwd) > 0.05
                or abs(float(current_strafe) - strafe) > 0.05
            )
            if current_changed:
                fwd = float(current_fwd)
                strafe = float(current_strafe)
                stale_clamp_applied = True
                stale_clamp_reason = "selected_sample_older_than_target_and_current_changed"
        selected_interval_end_time = None
        if selected is before and isinstance(after, dict):
            try:
                selected_interval_end_time = float(after.get("time", 0.0))
            except (TypeError, ValueError, AttributeError):
                selected_interval_end_time = None
        latest = history[-1] if history else {}
        latest_time = 0.0
        try:
            latest_time = float(latest.get("time", 0.0))
        except (TypeError, ValueError, AttributeError):
            latest_time = 0.0
        ctx.debug_last_movement_input_selection = {
            **debug,
            **tick_probe,
            **movement_time_probe_fields("time_probe_before", before),
            **movement_time_probe_fields("time_probe_after", after),
            **movement_time_probe_fields("time_probe_nearest", nearest),
            **movement_time_probe_fields("time_probe_nonzero_before", nonzero_before),
            **movement_time_probe_fields("time_probe_nonzero_after", nonzero_after),
            **movement_time_probe_fields("time_probe_nonzero_nearest", nonzero_nearest),
            **movement_history_window_fields(selected),
            "source": "delayed_remote_og_action_history",
            "selection_policy": selection_policy,
            "bounded_after_max_s": bounded_after_max,
            "bounded_after_applied": bounded_after_applied,
            "bounded_after_reason": bounded_after_reason,
            "nonzero_after_within_bounded_after_max": (
                nonzero_after_within_bounded_after_max
            ),
            "history_len": len(history),
            "target_time": target_time,
            "selected_age_s": (now - selected_time) if selected_time > 0.0 else None,
            "selected_target_error_s": selected_target_error,
            "selected_abs_target_error_s": (
                abs(selected_target_error) if selected_target_error is not None else None
            ),
            "selected_future_of_target": (
                selected_time > target_time if selected_time > 0.0 else None
            ),
            "selected_original_fwd": float(selected.get("fwd", 0.0)),
            "selected_original_strafe": float(selected.get("strafe", 0.0)),
            "selected_tick_target_error_ms": (
                self._tick_delta_signed(
                    int(selected.get("client_tick", 0) or 0),
                    int(tick_context.get("target_client_tick", 0) or 0),
                )
                if tick_selection_enabled
                and isinstance(tick_context, dict)
                and tick_context.get("target_client_tick") is not None
                and int(selected.get("client_tick", 0) or 0) > 0
                else None
            ),
            "selected_tick_abs_target_error_ms": (
                abs(
                    self._tick_delta_signed(
                        int(selected.get("client_tick", 0) or 0),
                        int(tick_context.get("target_client_tick", 0) or 0),
                    )
                )
                if tick_selection_enabled
                and isinstance(tick_context, dict)
                and tick_context.get("target_client_tick") is not None
                and int(selected.get("client_tick", 0) or 0) > 0
                else None
            ),
            "stale_clamp_s": stale_clamp,
            "stale_clamp_applied": stale_clamp_applied,
            "stale_clamp_reason": stale_clamp_reason,
            "selected_fwd": fwd,
            "selected_strafe": strafe,
            "selected_client_tick": int(selected.get("client_tick", 0) or 0),
            "selected_interval_end_age_s": (
                (now - selected_interval_end_time)
                if selected_interval_end_time is not None
                and selected_interval_end_time > 0.0
                else None
            ),
            "selected_interval_end_target_error_s": (
                (selected_interval_end_time - target_time)
                if selected_interval_end_time is not None
                else None
            ),
            "selected_interval_contains_target": (
                selected_time <= target_time
                and (
                    selected_interval_end_time is None
                    or target_time < selected_interval_end_time
                )
                if selected_time > 0.0
                else None
            ),
            "latest_age_s": (now - latest_time) if latest_time > 0.0 else None,
            "latest_fwd": float(latest.get("fwd", 0.0)) if isinstance(latest, dict) else 0.0,
            "latest_strafe": float(latest.get("strafe", 0.0)) if isinstance(latest, dict) else 0.0,
            "latest_client_tick": int(latest.get("client_tick", 0) or 0) if isinstance(latest, dict) else 0,
        }
        source = (
            "delayed_remote_og_action_history_stale_clamped"
            if stale_clamp_applied
            else "delayed_remote_og_action_history"
        )
        return fwd, strafe, source

    def _update_player_aim(self, ctx: ClientContext):
        """Update aim yaw/pitch from viewpoint or slot inputs (if enabled)."""
        now = time.monotonic()
        dt = now - ctx.last_aim_update
        ctx.last_aim_update = now
        if dt <= 0:
            dt = 1.0 / 60.0

        # If we recently received viewpoint info, keep it as the aim source.
        if ctx.player_aim_source == "viewpoint":
            if (now - ctx.player_aim_time) < self.viewpoint_timeout:
                return

        if not self.use_slot_aim:
            ctx.player_aim_yaw = ctx.player_heading
            ctx.player_aim_pitch = 0.0
            ctx.player_aim_source = "input"
            ctx.player_aim_time = 0.0
            return

        # Slot 6/7 are potential aim axes (empirical).
        def _normalize_axis(val: float) -> float:
            if val > 1.5 or val < -1.5:
                scale = getattr(ctx.weapon_system, "control_max", 1000.0) or 1000.0
                return max(-1.0, min(1.0, val / scale))
            return max(-1.0, min(1.0, val))

        slot6_val = _normalize_axis(ctx.weapon_system.behavior_slots[BehaviorSlot.SLOT6])
        slot7_val = _normalize_axis(ctx.weapon_system.behavior_slots[BehaviorSlot.SLOT7])

        if abs(slot6_val) > 0.01 or abs(slot7_val) > 0.01:
            # Integrate aim from slot inputs.
            ctx.player_aim_yaw += slot6_val * self.aim_turn_adjust * dt
            ctx.player_aim_pitch += slot7_val * self.aim_pitch_adjust * dt
            # Clamp pitch to reasonable range.
            max_pitch = math.radians(75.0)
            if ctx.player_aim_pitch > max_pitch:
                ctx.player_aim_pitch = max_pitch
            if ctx.player_aim_pitch < -max_pitch:
                ctx.player_aim_pitch = -max_pitch
            ctx.player_aim_source = "slot"
            ctx.player_aim_time = now
            return

        # Hold last slot-based aim briefly to avoid snapping back mid-fire.
        if ctx.player_aim_source == "slot" and (now - ctx.player_aim_time) < self.aim_hold_time:
            return

        # No aim input this tick - fall back to body heading.
        ctx.player_aim_yaw = ctx.player_heading
        ctx.player_aim_pitch = 0.0
        ctx.player_aim_source = "input"
        ctx.player_aim_time = 0.0

    def _update_player_position_stepped(self, ctx: ClientContext, move_dt: float,
                                        heading_override: float = None):
        """GOAL 8: integrate position/suspension in OG-sized inner substeps while the
        tank is near-stationary, to stabilize the idle vertical hover spring.

        The client's GameSim_substep_update (azurefishy-src Physics.c:1974) advances
        BOTH the angular and the rigid-body position integration in inner substeps
        (~40ms). The server's angular path (`step_client_substeps`) already inner-
        substeps, but `_update_player_position` was called ONCE with the full outer
        client-frame dt (~84ms on the OG WARP client). At that coarse dt the nonlinear
        tank hover spring overshoots: a ~0.95u displacement (the OG spawn-height vs the
        server spring-equilibrium mismatch) rings ~2u and takes seconds to settle —
        the live DO idle-Z bounce (`z=4.09→2.08→3.11`, `divergence_accum=0.000u`).
        Splitting into <=`goal8_substep_cap_s` chunks (default = server tick period)
        reproduces the OG inner-substep behavior: the SAME displacement rings ~0.41u
        and settles fast.

        SCOPE: only substep when the tank is near-stationary (horizontal speed below
        `goal8_substep_speed_max`). The idle Z bounce is a stationary/spawn problem; an
        actively driving/coasting tank carries real horizontal speed where the legacy
        single full-dt step already matches the client's prediction, and substepping
        there would perturb the coast velocity-decay trajectory the client reconciles
        against (a larger transient heading spike at turn→coast). Heading is held
        across the substeps (it was already advanced by the angular path this frame),
        so GOAL-7 angular parity is untouched; the py client (stepped at <=1/30 dt)
        takes the single-step path unchanged, so the GOAL-6 drift baseline is intact.
        """
        if move_dt <= 1e-6:
            return
        cap = float(getattr(self, "goal8_substep_cap_s", 1.0 / 30.0) or (1.0 / 30.0))
        if getattr(self, "goal8_legacy", False) or move_dt <= cap + 1e-6 or cap <= 0.0:
            self._update_player_position(ctx, dt_override=move_dt, heading_override=heading_override)
            return
        vx, vy = ctx.player_vel[0], ctx.player_vel[1]
        speed_max = float(getattr(self, "goal8_substep_speed_max", 0.5) or 0.0)
        if speed_max > 0.0 and (vx * vx + vy * vy) > speed_max * speed_max:
            # Moving: keep the legacy single-step horizontal/coast trajectory.
            self._update_player_position(ctx, dt_override=move_dt, heading_override=heading_override)
            return
        n = int(math.ceil(move_dt / cap))
        inner = move_dt / n
        for _ in range(n):
            self._update_player_position(ctx, dt_override=inner, heading_override=heading_override)

    def _update_player_position(self, ctx: ClientContext, dt_override: float = 0.0, heading_override: float = None):
        """
        Simulate player position using damped persistent velocity model.

        Matches client's RigidBody_integrate_position (Game/Simulation/Physics.c:6032;
        core math Vec3_integrate_motion at Physics.c:5124). Verified bit-exact 2026-06-01:
          1. Vehicle controller computes per-frame impulse (zeroed each frame)
          2. effective_acc = impulse - vel * linear_damp  (damped mode, PhysicsStateFlags+3;
             decompile reads single linear_damping from PhysicsConfig+0x78)
          3. pos += vel * dt + 0.5 * effective_acc * dtÂ²  (semi-implicit Euler)
          4. vel += effective_acc * dt  (velocity persists across frames)
        ORDER (verified faithful 2026-06-01, tick loop ~21586): heading is stepped FIRST
        (step_client_substeps) then this runs, matching decompile RigidBody_step (Physics.c:6088,
        angular before position). heading_override passes the OLD (pre-integration) heading so
        thrust direction uses it — also faithful, since decompile TankVehicle_apply_physics
        accumulates the impulse (entity+0x24) BEFORE RigidBody_step runs.
        NOTE: linear_damp is hardcoded 1.5 (env-overridable), not parsed from BEHAVIOR; the
        decompile reads per-entity PhysicsConfig+0x78. 1.5 is OG-tank-verified; non-tank may differ.

        Entity layout:
          entity[0x0c] = position (persistent)
          entity[0x18] = velocity (persistent, damped)
          entity[0x24] = impulse accumulator (zeroed after physics step)

        Steady state: vel = impulse / linear_damp

        Current OG tank memory shows the active PhysicsConfig linear damping
        coefficient is 1.5 with linear damping enabled. The server still keeps
        drive/coast env overrides so we can A/B older empirical assumptions.
        """
        import math

        # Default-off sub-phase timing (collision vs building vs attitude) to localize
        # the rough-cell controller-cadence cost. Gated by WULFRAM_UPP_PHASE_TIMING.
        _upp_timing = os.environ.get("WULFRAM_UPP_PHASE_TIMING", "0") == "1"
        if _upp_timing:
            ctx._upp_collision_ms = 0.0
            ctx._upp_building_ms = 0.0
            ctx._upp_attitude_ms = 0.0

        self._repair_recent_control_pose_jump(ctx, "controller_tick_pre")
        dt = dt_override if dt_override > 0 else 1.0 / self.tick_rate_hz

        # PHYSICS SANITY GUARD: a spring/suspension instability on rough or steep
        # terrain can drive a tank's velocity (then position) to inf/NaN. A NaN
        # then crashes the per-tick suspension sample EVERY tick — caught by the
        # tick loop, but it spams a traceback ~10x/s and freezes that tank — and a
        # NaN position would poison replication to every other client. Reset a
        # non-finite tank state to a finite, safe value so it self-heals once
        # instead of recurring forever. This fires ONLY on already-broken
        # (non-finite) state, so it cannot affect normal finite physics parity.
        # Surfaced by the A3 multi-client soak (2026-06-01).
        if not all(math.isfinite(v) for v in ctx.player_vel):
            print(f"[PHYSICS] Client {ctx.client_id}: non-finite velocity "
                  f"{tuple(ctx.player_vel)} -> reset to 0")
            ctx.player_vel = [0.0, 0.0, 0.0]
        if not all(math.isfinite(p) for p in ctx.player_pos):
            fx = ctx.player_pos[0] if math.isfinite(ctx.player_pos[0]) else 5050.0
            fy = ctx.player_pos[1] if math.isfinite(ctx.player_pos[1]) else 5000.0
            safe_z = 5.0
            if self.terrain is not None:
                try:
                    safe_z = float(self.terrain.get_height(fx, fy)) + 5.0
                except Exception:
                    safe_z = 5.0
            print(f"[PHYSICS] Client {ctx.client_id}: non-finite position -> "
                  f"clamp to ({fx:.1f},{fy:.1f},{safe_z:.1f})")
            ctx.player_pos = [fx, fy, safe_z]
            ctx.player_vel = [0.0, 0.0, 0.0]

        # Read movement input (slot 2 = forward, slot 3 = strafe)
        raw_throttle_input = 0.0
        raw_strafe_input = 0.0
        movement_input_delay_s = 0.0
        movement_input_source = "current_slots"
        if ctx.injected_input is not None:
            throttle_input, strafe_input = ctx.injected_input
            raw_throttle_input, raw_strafe_input = throttle_input, strafe_input
            movement_input_source = "injected"
        else:
            throttle_val = ctx.weapon_system.behavior_slots[BehaviorSlot.MOVING_FORWARD]
            strafe_val = ctx.weapon_system.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS]
            raw_throttle_input = self._normalize_behavior_axis_value(ctx, throttle_val)
            raw_strafe_input = self._decode_network_strafe_input(ctx, strafe_val)
            movement_input_delay_s = self._remote_og_movement_input_delay_for_ctx(ctx)
            throttle_input, strafe_input, movement_input_source = self._select_delayed_movement_input(
                ctx,
                current_fwd=raw_throttle_input,
                current_strafe=raw_strafe_input,
                delay_s=movement_input_delay_s,
            )

        if abs(throttle_input) < 0.05:
            throttle_input = 0.0
        if abs(strafe_input) < 0.05:
            strafe_input = 0.0

        # Per-vehicle-type physics from shared config (decompile-verified)
        veh_config = VEHICLE_PHYSICS_CONFIGS.get(ctx.entity_type)
        move_adjust = veh_config.move_adjust if veh_config else 85.0
        strafe_adjust = veh_config.strafe_adjust if veh_config else 69.7
        low_fuel_level = veh_config.low_fuel_level if veh_config else 2000.0
        max_fuel = veh_config.max_fuel if veh_config else 33000.0
        has_input = abs(throttle_input) > 0.0 or abs(strafe_input) > 0.0
        linear_damp = self.linear_damp_driving if has_input else self.linear_damp_coasting
        vel_x, vel_y, vel_z = ctx.player_vel

        # Decompile-backed flat-ground mobility gate from Tank_compute_mobility_factors:
        # forward mobility ramps from 0.4 at rest toward 1.0 as current speed rises.
        if ctx.entity_type == EntityType.TANK:
            current_speed = ctx.player_speed
            if current_speed <= 0.0:
                current_speed = vehicle_runtime_speed(
                    vel_x,
                    vel_y,
                    vel_z,
                    up_axis=self.up_axis,
                )
            current_fuel = float(getattr(ctx, "player_fuel", max_fuel))
            forward_mobility = tank_fuel_mobility_factor(
                current_fuel,
                low_fuel_level,
            )
            turn_mobility = self._tank_altitude_mobility(ctx)
            forward_mobility *= turn_mobility

            # Decompile slope mobility (Vehicles.c:1148-1161)
            if self.terrain and self.terrain_pitch_enabled:
                _heading = heading_override if heading_override is not None else ctx.player_heading
                avg_up_s, _ = self._sample_tank_surface_state(ctx, _heading)
                if abs(avg_up_s[2]) > 1e-6:
                    slope_dx = -avg_up_s[0] / avg_up_s[2]
                    slope_dy = -avg_up_s[1] / avg_up_s[2]
                else:
                    slope_dx, slope_dy = self.terrain.get_slope(
                        ctx.player_pos[0], ctx.player_pos[1])
                cos_y = math.cos(_heading)
                sin_y = math.sin(_heading)
                slope_fwd = slope_dx * cos_y + slope_dy * sin_y
                max_vel = veh_config.max_velocity if veh_config else 80.0
                slope_factor = tank_slope_mobility_factor(
                    slope_fwd, throttle_input, max_vel)
                forward_mobility *= slope_factor
        else:
            current_speed = vehicle_runtime_speed(
                vel_x,
                vel_y,
                vel_z,
                up_axis=self.up_axis,
            )
            forward_mobility = 1.0
            turn_mobility = 1.0

        yaw = heading_override if heading_override is not None else ctx.player_heading
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        avg_up = (0.0, 0.0, 1.0)
        _clearance_ratio = 1.0
        dh_dx = 0.0
        dh_dy = 0.0

        if self.up_axis == "z":
            if self.terrain and self.terrain_pitch_enabled:
                avg_up, _clearance_ratio = self._sample_tank_surface_state(ctx, yaw)
                if abs(avg_up[2]) > 1e-6:
                    dh_dx = -avg_up[0] / avg_up[2]
                    dh_dy = -avg_up[1] / avg_up[2]
                else:
                    dh_dx, dh_dy = self.terrain.get_slope(ctx.player_pos[0], ctx.player_pos[1])
                if getattr(self, "tank_drive_terrain_aligned", False):
                    forward, right, _up = terrain_aligned_basis(dh_dx, dh_dy, yaw)
                    drive_basis_source = "terrain_aligned"
                elif getattr(self, "tank_drive_body_matrix", True):
                    forward, right = tank_body_matrix_drive_basis(
                        yaw,
                        roll=float(ctx.player_pose.get("roll", 0.0) or 0.0),
                        pitch=float(ctx.player_pose.get("pitch", 0.0) or 0.0),
                        rotation_matrix=getattr(ctx, "spring_body_matrix", None),
                    )
                    drive_basis_source = "entity_body_matrix"
                else:
                    # Explicit debug fallback for isolating body-pose drive
                    # effects against the older horizontal approximation.
                    forward = (cos_yaw, sin_yaw, 0.0)
                    right = (-sin_yaw, cos_yaw, 0.0)
                    drive_basis_source = "entity_yaw_flat"
            else:
                forward = (cos_yaw, sin_yaw, 0.0)
                right = (-sin_yaw, cos_yaw, 0.0)
                drive_basis_source = "flat"
            vertical_idx = 2
        else:
            forward = (cos_yaw, 0.0, sin_yaw)
            right = (-sin_yaw, 0.0, cos_yaw)
            drive_basis_source = "y_up"
            vertical_idx = 1

        # Per-frame impulse (like entity[0x24], zeroed each frame by controller)
        fwd_impulse = throttle_input * move_adjust * forward_mobility
        strafe_impulse = (
            strafe_input * strafe_adjust * forward_mobility * turn_mobility
        )

        impulse_x = forward[0] * fwd_impulse + right[0] * strafe_impulse
        impulse_y = forward[1] * fwd_impulse + right[1] * strafe_impulse
        impulse_z = forward[2] * fwd_impulse + right[2] * strafe_impulse
        drive_impulse_uncapped = (impulse_x, impulse_y, impulse_z)

        # TankVehicle_apply_physics clamps the movement vector against the same
        # move_adjust scalar used to build forward motion, not the separate
        # max_velocity field.
        move_cap = move_adjust
        move_mag = math.sqrt(
            impulse_x * impulse_x + impulse_y * impulse_y + impulse_z * impulse_z
        )
        if move_mag > move_cap and move_mag > 0.0:
            scale = move_cap / move_mag
            impulse_x *= scale
            impulse_y *= scale
            impulse_z *= scale
        drive_impulse_capped = (impulse_x, impulse_y, impulse_z)

        contact_x = 0.0
        contact_y = 0.0
        terrain_contact_impulse = (0.0, 0.0, 0.0)
        if (
            ctx.entity_type == EntityType.TANK
            and getattr(self, "tank_terrain_contact_coupling_enabled", False)
        ):
            contact_x, contact_y = self._tank_terrain_contact_vector(ctx)
            pre_contact_x, pre_contact_y, pre_contact_z = impulse_x, impulse_y, impulse_z
            impulse_x, impulse_y, _terrain_speed = tank_terrain_contact_coupling(
                impulse_x,
                impulse_y,
                contact_x,
                contact_y,
            )
            terrain_contact_impulse = (
                impulse_x - pre_contact_x,
                impulse_y - pre_contact_y,
                impulse_z - pre_contact_z,
            )
        else:
            _terrain_speed = 0.0
        tank_vehicle_impulse = (impulse_x, impulse_y, impulse_z)
        softbody_scalar_stretch_source = "off"
        softbody_scalar_stretch_speed = 0.0
        softbody_scalar_stretch_denominator = float(
            getattr(
                self,
                "tank_softbody_scalar_stretch_denominator",
                OG_TANK_SPRING_STRETCH_SPEED_DENOMINATOR,
            )
            or OG_TANK_SPRING_STRETCH_SPEED_DENOMINATOR
        )
        softbody_scalar_stretch_ratio = 0.0
        configured_stretch_source = str(
            getattr(self, "tank_softbody_scalar_stretch_source", "entity_velocity")
            or "entity_velocity"
        ).strip().lower()
        if configured_stretch_source in {"0", "false", "off", "no"}:
            configured_stretch_source = "off"
        if configured_stretch_source in {"entity_velocity", "velocity"}:
            softbody_scalar_stretch_source = "entity_velocity"
            softbody_scalar_stretch_speed = math.hypot(vel_x, vel_y)
            softbody_scalar_stretch_ratio = tank_spring_scalar_stretch_ratio(
                vel_x,
                vel_y,
                speed_denominator=softbody_scalar_stretch_denominator,
            )
        elif configured_stretch_source == "tank_vehicle_impulse":
            softbody_scalar_stretch_source = "tank_vehicle_impulse"
            softbody_scalar_stretch_speed = math.hypot(
                tank_vehicle_impulse[0],
                tank_vehicle_impulse[1],
            )
            softbody_scalar_stretch_ratio = tank_spring_scalar_stretch_ratio(
                tank_vehicle_impulse[0],
                tank_vehicle_impulse[1],
                speed_denominator=softbody_scalar_stretch_denominator,
            )

        # Add gravity to vertical impulse (matches GUESS3_Transform_accelerate_z)
        gravity = self.gravity
        terrain_ground_level = None
        if self.terrain and self.up_axis == "z":
            terrain_ground_level = self._terrain_physics_ground_z_at(
                ctx.player_pos[0],
                ctx.player_pos[1],
            )
        ground_override_ref_pos = getattr(ctx, "world_collision_ref_pos", None)
        ground_override_released = False
        ground_override_release_reason = ""
        ground_override_ref_terrain_level = getattr(ctx, "ground_override_ref_terrain_level", None)
        ground_override_terrain_change = None
        if ctx.ground_level_override is not None and terrain_ground_level is not None:
            release_distance = max(0.0, getattr(self, "ground_override_release_distance", 24.0))
            release_height = max(0.0, getattr(self, "ground_override_release_height", 4.0))
            terrain_release_distance = max(
                0.0,
                getattr(self, "ground_override_release_terrain_distance", 4.0),
            )
            terrain_release_height = max(
                0.0,
                getattr(self, "ground_override_release_terrain_height", 0.75),
            )
            moved_far = False
            moved_for_terrain_change = False
            if ground_override_ref_pos is not None:
                dx_ref = ctx.player_pos[0] - ground_override_ref_pos[0]
                dy_ref = ctx.player_pos[1] - ground_override_ref_pos[1]
                dist_sq_ref = dx_ref * dx_ref + dy_ref * dy_ref
                moved_far = (
                    release_distance > 0.0
                    and dist_sq_ref >= release_distance * release_distance
                )
                moved_for_terrain_change = (
                    terrain_release_distance > 0.0
                    and dist_sq_ref >= terrain_release_distance * terrain_release_distance
                )
            terrain_delta = abs(float(ctx.ground_level_override) - terrain_ground_level)
            if ground_override_ref_terrain_level is not None:
                ground_override_terrain_change = abs(
                    terrain_ground_level - float(ground_override_ref_terrain_level)
                )
            terrain_changed_under_anchor = (
                ground_override_terrain_change is not None
                and terrain_release_height > 0.0
                and moved_for_terrain_change
                and ground_override_terrain_change >= terrain_release_height
            )
            release_by_height = release_height > 0.0 and terrain_delta >= release_height
            if moved_far or release_by_height or terrain_changed_under_anchor:
                ctx.ground_level_override = None
                ground_override_released = True
                ctx.ground_override_ref_terrain_level = None
                if moved_far:
                    ground_override_release_reason = "distance"
                elif terrain_changed_under_anchor:
                    ground_override_release_reason = "terrain_change"
                else:
                    ground_override_release_reason = "height"
        if (
            ctx.ground_level_override is not None
            and terrain_ground_level is not None
            and ctx.entity_type == EntityType.TANK
            and getattr(self, "tank_suspension_enabled", False)
            and getattr(self, "tank_suspension_model", "softbody") != "compact"
        ):
            ctx.ground_level_override = None
            ground_override_released = True
            ground_override_release_reason = "softbody_suspension"
            ctx.ground_override_ref_terrain_level = None
        use_ground_override = ctx.ground_level_override is not None
        if use_ground_override:
            ground_level = ctx.ground_level_override
            ground_level_source = "override"
        elif terrain_ground_level is not None:
            ground_level = terrain_ground_level
            ground_level_source = "terrain"
        else:
            ground_level = self.ground_level
            ground_level_source = "default"
        horizontal_damp = linear_damp
        tank_ground_contact_damp = 0.0

        jumpjet_input = self._get_jumpjet_input(ctx)
        jump_jet_direction = self._jump_jet_direction_vector(ctx, vertical_idx=vertical_idx)
        jump_jet_velocity_delta = (0.0, 0.0, 0.0)
        if vertical_idx == 2:
            jump_altitude = ctx.player_pos[2] - ground_level if ground_level is not None else ctx.player_pos[2]
            jump_jet_fired, jump_jet_impulse, jump_jet_velocity_delta = self._apply_jump_jets_fixed_step(
                ctx,
                dt=dt,
                jumpjet_input=jumpjet_input,
                current_altitude=jump_altitude,
                current_vel_up=vel_z,
                direction=jump_jet_direction,
                vertical_idx=vertical_idx,
            )
            vel_x += jump_jet_velocity_delta[0]
            vel_y += jump_jet_velocity_delta[1]
            vel_z += jump_jet_velocity_delta[2]
        else:
            jump_altitude = ctx.player_pos[1] - ground_level if ground_level is not None else ctx.player_pos[1]
            jump_jet_fired, jump_jet_impulse, jump_jet_velocity_delta = self._apply_jump_jets_fixed_step(
                ctx,
                dt=dt,
                jumpjet_input=jumpjet_input,
                current_altitude=jump_altitude,
                current_vel_up=vel_y,
                direction=jump_jet_direction,
                vertical_idx=vertical_idx,
            )
            vel_x += jump_jet_velocity_delta[0]
            vel_y += jump_jet_velocity_delta[1]
            vel_z += jump_jet_velocity_delta[2]

        gravity_impulse = (0.0, 0.0, 0.0)
        suspension_impulse = (0.0, 0.0, 0.0)
        pre_ground_vertical_impulse = None
        vertical_ground_cancelled = False
        suspension_lift = 0.0
        suspension_clearance = None
        suspension_target_clearance = None
        suspension_model = None
        suspension_softbody = None
        spring_state_for_attitude = None
        if (
            getattr(self, "tank_suspension_enabled", False)
            and ctx.entity_type == EntityType.TANK
            and self.terrain is not None
            and self.up_axis == "z"
            and not use_ground_override
        ):
            if not self.terrain_pitch_enabled:
                avg_up, _clearance_ratio = self._sample_tank_surface_state(ctx, yaw)
            spring_state = getattr(ctx, "debug_last_spring_state", {}) or {}
            if isinstance(spring_state, dict) and spring_state:
                spring_state_for_attitude = dict(spring_state)
            legacy_target_clearance = self._tank_hover_clearance_target(ctx)
            try:
                suspension_clearance = float(spring_state.get("average_clearance"))
            except (TypeError, ValueError):
                suspension_clearance = _clearance_ratio * legacy_target_clearance

            if getattr(self, "tank_suspension_model", "softbody") == "compact":
                suspension_model = "compact_legacy"
                suspension_target_clearance = legacy_target_clearance
                suspension_lift = tank_suspension_lift_accel(
                    suspension_clearance,
                    suspension_target_clearance,
                    vel_z,
                    stiffness=getattr(self, "tank_suspension_stiffness", 40.0),
                    damping=getattr(self, "tank_suspension_damping", 1.5),
                    lift_cap=getattr(self, "tank_suspension_lift_cap", 120.0),
                )
            else:
                veh_cfg = VEHICLE_PHYSICS_CONFIGS.get(ctx.entity_type)
                slot5 = (
                    self._normalize_behavior_axis_value(
                        ctx,
                        tank_softbody_control_slot_value(ctx.weapon_system.behavior_slots),
                    )
                    if getattr(ctx, "weapon_system", None) is not None
                    else 0.0
                )
                suspension_softbody = tank_softbody_suspension_force(
                    suspension_clearance,
                    vel_z,
                    slot5,
                    samples=(
                        spring_state.get("samples")
                        if isinstance(spring_state, dict)
                        else None
                    ),
                    use_per_point_lift=getattr(
                        self,
                        "tank_softbody_per_point_force",
                        False,
                    ),
                    use_piecewise_height_factor=getattr(
                        self,
                        "tank_softbody_piecewise_height",
                        False,
                    ),
                    use_decompile_piecewise_force=getattr(
                        self,
                        "tank_softbody_decompile_piecewise_force",
                        False,
                    ),
                    gravity=gravity,
                    physics_timestep_factor=(
                        OG_PHYSICS_TIMESTEP_FACTOR if gravity < 0.0 else 0.0
                    ),
                    max_altitude=veh_cfg.max_altitude if veh_cfg else 3.25,
                    gravity_pct=veh_cfg.gravity_pct if veh_cfg else 1.0,
                    damping=getattr(self, "tank_suspension_damping", 6.0),
                    scalar_stretch_ratio=softbody_scalar_stretch_ratio,
                    scalar_stretch_source=softbody_scalar_stretch_source,
                    scalar_stretch_speed=softbody_scalar_stretch_speed,
                    scalar_stretch_denominator=softbody_scalar_stretch_denominator,
                )
                suspension_model = suspension_softbody.model
                suspension_target_clearance = suspension_softbody.target_average_height
                suspension_lift = suspension_softbody.lift_accel
                if gravity < 0.0:
                    gravity = -abs(suspension_softbody.support_accel)

        if (
            ctx.entity_type == EntityType.TANK
            and self.up_axis == "z"
            and self.terrain is not None
            and not use_ground_override
        ):
            configured_contact_damp = max(
                0.0,
                float(getattr(self, "tank_ground_contact_damp", 0.0) or 0.0),
            )
            if suspension_softbody is not None:
                horizontal_damp, tank_ground_contact_damp = tank_softbody_horizontal_damping(
                    linear_damp,
                    configured_contact_damp,
                    suspension_softbody.slot5,
                )
            elif configured_contact_damp > 0.0:
                tank_ground_contact_damp = configured_contact_damp
                horizontal_damp = max(linear_damp, configured_contact_damp)

        # Gravity and ground collision use terrain-aware ground_level (computed above).
        if vertical_idx == 2:
            gravity_impulse = (0.0, 0.0, gravity)
            suspension_impulse = (0.0, 0.0, suspension_lift)
            impulse_z += gravity  # gravity is negative
            impulse_z += suspension_lift
            pre_ground_vertical_impulse = impulse_z
            if ctx.player_pos[2] <= ground_level and ctx.player_vel[2] + impulse_z * dt < 0:
                vertical_ground_cancelled = True
                impulse_z = 0.0
        else:
            gravity_impulse = (0.0, gravity, 0.0)
            impulse_y += gravity
            pre_ground_vertical_impulse = impulse_y
            if ctx.player_pos[1] <= ground_level and ctx.player_vel[1] + gravity * dt < 0:
                vertical_ground_cancelled = True
                impulse_y = 0.0

        # Damped effective acceleration: acc = impulse - vel * linear_damp
        # (from RigidBody_integrate_position damped mode, Game/Simulation/Physics.c:6032;
        #  decompile reads linear_damping from PhysicsConfig+0x78. Verified 2026-06-01.)
        acc_x = impulse_x - vel_x * horizontal_damp
        acc_y = impulse_y - vel_y * horizontal_damp
        acc_z = impulse_z - vel_z * linear_damp

        # Ground collision: zero vertical acc+vel when on ground and pushing down
        if vertical_idx == 2:
            if ctx.player_pos[2] <= ground_level and (vel_z + acc_z * dt) < 0:
                acc_z = -vel_z / dt if dt > 0 else 0.0  # bring vel to zero
        else:
            if ctx.player_pos[1] <= ground_level and (vel_y + acc_y * dt) < 0:
                acc_y = -vel_y / dt if dt > 0 else 0.0

        pre_pos = ctx.player_pos
        pre_vel = (vel_x, vel_y, vel_z)

        # Verlet position step + float32 quantize, via the shared sim kernel
        # (integrate_verlet): pos += vel*dt + 0.5*acc*dt²; vel += acc*dt; pos uses
        # the OLD velocity. Semi-implicit Euler from Vec3_integrate_motion
        # (Game/Simulation/Physics.c:5124, called by RigidBody_integrate_position;
        # bit-exact verified 2026-06-01). Always f32 — the rounding point now lives
        # in the kernel (client stores pos/vel as float32).
        from .physics import _integrate_verlet
        (new_x, new_y, new_z), (new_vel_x, new_vel_y, new_vel_z) = _integrate_verlet(
            ctx.player_pos, (vel_x, vel_y, vel_z), (acc_x, acc_y, acc_z), dt
        )

        # Clamp to world bounds
        if self.up_axis == "z":
            new_x = max(-self.world_bound, min(self.world_bound, new_x))
            new_y = max(-self.world_bound, min(self.world_bound, new_y))
            # Clamp Z to terrain height at NEW position (not pre-integration)
            if self.terrain and not use_ground_override:
                terrain_z = self._terrain_physics_ground_z_at(new_x, new_y)
                if new_z < terrain_z:
                    new_z = terrain_z
                    if new_vel_z < 0:
                        new_vel_z = 0.0
            else:
                if new_z < ground_level:
                    new_z = ground_level
                    if new_vel_z < 0:
                        new_vel_z = 0.0
        else:
            new_x = max(-self.world_bound, min(self.world_bound, new_x))
            new_z = max(-self.world_bound, min(self.world_bound, new_z))
            new_y = max(ground_level, new_y)

        ctx.debug_last_collision = {}
        ctx.debug_last_motion_collision = {}
        ctx.debug_last_terrain_contact_probe = {}
        jump_jet_collision_guard_debug = {}
        tank_terrain_projection_guard_debug = {}
        ctx.rigid_body_target_pos = (new_x, new_y, new_z)
        ctx.rigid_body_target_rot = (
            float((getattr(ctx, "player_pose", {}) or {}).get("roll", 0.0) or 0.0),
            float((getattr(ctx, "player_pose", {}) or {}).get("pitch", 0.0) or 0.0),
            float(heading_override if heading_override is not None else ctx.player_heading),
        )
        ctx.rigid_body_interp_tolerance = float(
            os.environ.get("WULFRAM_ENTITY_INTERPOLATION_TOLERANCE", "0.003")
        )

        # Decompile-shaped terrain/world contact pass before static blockers.
        pre_world_collision_pos = (new_x, new_y, new_z)
        pre_world_collision_vel = (new_vel_x, new_vel_y, new_vel_z)
        ctx._world_collision_step_pre_pos = pre_pos
        ctx._world_collision_step_pre_vel = pre_vel
        ctx._world_collision_step_dt = dt
        _upp_t0 = time.perf_counter() if _upp_timing else 0.0
        # Per-step collision wall-clock budget. DEFAULT OFF (0) now that the CBSP hot
        # path is fast enough (the rough-H180 deep-stuck physics step resolves in ~4.6 ms
        # median / 7.5 ms total after the inlining + separating-plane early-reject +
        # static-geometry caches; see docs/goal-runs/2026-05-31-collision-perf-port.md),
        # so collision runs UNBUDGETED and deterministic by default. Set
        # WULFRAM_ENTITY_TERRAIN_CONTACT_TIME_BUDGET_MS to a positive millisecond value to
        # re-enable it as a safety cap (it then truncates any step's contact resolution at
        # that wall-clock deadline, trading determinism for a bounded worst case).
        tgc = getattr(self, "_terrain_grid_collision", None)
        if tgc is not None:
            try:
                _coll_budget_ms = float(
                    os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_TIME_BUDGET_MS", "0")
                    or 0.0
                )
            except ValueError:
                _coll_budget_ms = 0.0
            tgc._query_deadline = (
                time.perf_counter() + _coll_budget_ms / 1000.0
                if _coll_budget_ms > 0.0
                else None
            )
        try:
            new_x, new_y, new_z, new_vel_x, new_vel_y, new_vel_z = self._resolve_entity_world_collision(
                ctx, new_x, new_y, new_z, new_vel_x, new_vel_y, new_vel_z
            )
        finally:
            if tgc is not None:
                tgc._query_deadline = None
            if _upp_timing:
                ctx._upp_collision_ms = (time.perf_counter() - _upp_t0) * 1000.0
            for attr_name in (
                "_world_collision_step_pre_pos",
                "_world_collision_step_pre_vel",
                "_world_collision_step_dt",
            ):
                if hasattr(ctx, attr_name):
                    delattr(ctx, attr_name)

        if (
            self.up_axis == "z"
            and ctx.entity_type == EntityType.TANK
            and not use_ground_override
            and getattr(self, "jump_jet_collision_guard", True)
            and float(getattr(ctx, "jump_cooldown_remaining", 0.0) or 0.0) > 0.0
        ):
            raw_world_collision_pos = (new_x, new_y, new_z)
            raw_world_collision_vel = (new_vel_x, new_vel_y, new_vel_z)
            collision_dx = raw_world_collision_pos[0] - pre_world_collision_pos[0]
            collision_dy = raw_world_collision_pos[1] - pre_world_collision_pos[1]
            collision_xy = math.sqrt(collision_dx * collision_dx + collision_dy * collision_dy)
            collision_z_pop = raw_world_collision_pos[2] - pre_world_collision_pos[2]
            max_xy = float(getattr(self, "jump_jet_collision_guard_xy", 1.0) or 0.0)
            max_z_pop = float(getattr(self, "jump_jet_collision_guard_zpop", 2.0) or 0.0)
            guard_applies = (
                (max_xy > 0.0 and collision_xy > max_xy)
                or (max_z_pop > 0.0 and collision_z_pop > max_z_pop)
            )
            if guard_applies:
                if self.terrain is not None:
                    landing_ground = self._terrain_physics_ground_z_at(
                        pre_world_collision_pos[0],
                        pre_world_collision_pos[1],
                    )
                else:
                    landing_ground = ground_level
                landing_clearance = float(
                    getattr(self, "jump_jet_landing_clearance", 1.85) or 0.0
                )
                landing_floor = float(landing_ground) + max(0.0, landing_clearance)
                landing_floor_applied = pre_world_collision_pos[2] < landing_floor
                new_x = pre_world_collision_pos[0]
                new_y = pre_world_collision_pos[1]
                new_z = landing_floor if landing_floor_applied else pre_world_collision_pos[2]
                new_vel_x = pre_world_collision_vel[0]
                new_vel_y = pre_world_collision_vel[1]
                new_vel_z = (
                    max(0.0, pre_world_collision_vel[2])
                    if landing_floor_applied
                    else pre_world_collision_vel[2]
                )
                ctx.world_collision_ref_pos = (new_x, new_y, new_z)
                ctx.world_collision_bounds_dirty = False
                jump_jet_collision_guard_debug = {
                    "applied": True,
                    "reason": "jumpjet_large_world_collision_projection",
                    "collision_xy": collision_xy,
                    "collision_z_pop": collision_z_pop,
                    "max_xy": max_xy,
                    "max_z_pop": max_z_pop,
                    "landing_ground": landing_ground,
                    "landing_floor": landing_floor,
                    "landing_floor_applied": landing_floor_applied,
                    "pre_world_collision_pos": pre_world_collision_pos,
                    "pre_world_collision_vel": pre_world_collision_vel,
                    "raw_world_collision_pos": raw_world_collision_pos,
                    "raw_world_collision_vel": raw_world_collision_vel,
                    "guarded_pos": (new_x, new_y, new_z),
                    "guarded_vel": (new_vel_x, new_vel_y, new_vel_z),
                    "raw_motion_collision": dict(getattr(ctx, "debug_last_motion_collision", {}) or {}),
                }
            elif collision_xy > 0.0 or collision_z_pop > 0.0:
                jump_jet_collision_guard_debug = {
                    "applied": False,
                    "collision_xy": collision_xy,
                    "collision_z_pop": collision_z_pop,
                    "max_xy": max_xy,
                    "max_z_pop": max_z_pop,
                }

        if (
            self.up_axis == "z"
            and ctx.entity_type == EntityType.TANK
            and not use_ground_override
            and getattr(self, "tank_terrain_projection_guard", False)
            and float(getattr(ctx, "jump_cooldown_remaining", 0.0) or 0.0) <= 0.0
        ):
            raw_world_collision_pos = (new_x, new_y, new_z)
            raw_world_collision_vel = (new_vel_x, new_vel_y, new_vel_z)
            collision_dx = raw_world_collision_pos[0] - pre_world_collision_pos[0]
            collision_dy = raw_world_collision_pos[1] - pre_world_collision_pos[1]
            collision_xy = math.sqrt(collision_dx * collision_dx + collision_dy * collision_dy)
            collision_z_pop = raw_world_collision_pos[2] - pre_world_collision_pos[2]
            max_xy = float(getattr(self, "tank_terrain_projection_guard_xy", 1.0) or 0.0)
            max_z_pop = float(getattr(self, "tank_terrain_projection_guard_zpop", 2.0) or 0.0)
            if self.terrain is not None:
                projection_ground = self._terrain_physics_ground_z_at(
                    pre_world_collision_pos[0],
                    pre_world_collision_pos[1],
                )
            else:
                projection_ground = ground_level
            projection_clearance = pre_world_collision_pos[2] - float(projection_ground)
            min_clearance = float(
                getattr(self, "tank_terrain_projection_guard_min_clearance", 0.5) or 0.0
            )
            guard_applies = (
                projection_clearance >= min_clearance
                and (
                    (max_xy > 0.0 and collision_xy > max_xy)
                    or (max_z_pop > 0.0 and collision_z_pop > max_z_pop)
                )
            )
            if guard_applies:
                new_x, new_y, new_z = pre_world_collision_pos
                new_vel_x, new_vel_y, new_vel_z = pre_world_collision_vel
                ctx.world_collision_ref_pos = (new_x, new_y, new_z)
                ctx.world_collision_bounds_dirty = False
                tank_terrain_projection_guard_debug = {
                    "applied": True,
                    "reason": "tank_large_terrain_projection_while_clear",
                    "collision_xy": collision_xy,
                    "collision_z_pop": collision_z_pop,
                    "max_xy": max_xy,
                    "max_z_pop": max_z_pop,
                    "projection_ground": projection_ground,
                    "projection_clearance": projection_clearance,
                    "min_clearance": min_clearance,
                    "pre_world_collision_pos": pre_world_collision_pos,
                    "pre_world_collision_vel": pre_world_collision_vel,
                    "raw_world_collision_pos": raw_world_collision_pos,
                    "raw_world_collision_vel": raw_world_collision_vel,
                    "guarded_pos": (new_x, new_y, new_z),
                    "guarded_vel": (new_vel_x, new_vel_y, new_vel_z),
                    "raw_motion_collision": dict(getattr(ctx, "debug_last_motion_collision", {}) or {}),
                }
            elif collision_xy > 0.0 or collision_z_pop > 0.0:
                tank_terrain_projection_guard_debug = {
                    "applied": False,
                    "collision_xy": collision_xy,
                    "collision_z_pop": collision_z_pop,
                    "max_xy": max_xy,
                    "max_z_pop": max_z_pop,
                    "projection_ground": projection_ground,
                    "projection_clearance": projection_clearance,
                    "min_clearance": min_clearance,
                }

        # Building AABB collision (matching client-side)
        _upp_tb = time.perf_counter() if _upp_timing else 0.0
        new_x, new_y, new_vel_x, new_vel_y = self._check_building_collisions(
            ctx, new_x, new_y, new_z, new_vel_x, new_vel_y)
        if _upp_timing:
            ctx._upp_building_ms = (time.perf_counter() - _upp_tb) * 1000.0

        # Terrain/world contact response can still push the tank back below the
        # terrain plane after the initial post-integrate clamp. Keep the final
        # authoritative pose on or above terrain before replication.
        final_ground_clamped = False
        # GOAL 2 gate: the terrain-Z safety clamp is the canonical server-only,
        # client-unpredictable displacement (server clamps Z to terrain; the OG
        # client uses a spring-damper). The magnitude pushed up here is the
        # divergence signal the reactive correction gate accumulates.
        ground_clamp_dz = 0.0
        if self.up_axis == "z":
            if self.terrain and not use_ground_override:
                terrain_z = self._terrain_physics_ground_z_at(new_x, new_y)
            else:
                terrain_z = ground_level
            if new_z < terrain_z:
                ground_clamp_dz = terrain_z - new_z
                new_z = terrain_z
                final_ground_clamped = True
                if new_vel_z < 0.0:
                    new_vel_z = 0.0
        else:
            if new_y < ground_level:
                ground_clamp_dz = ground_level - new_y
                new_y = ground_level
                final_ground_clamped = True
                if new_vel_y < 0.0:
                    new_vel_y = 0.0
        self._accumulate_correction_divergence(ctx, ground_clamp_dz)
        if final_ground_clamped:
            ctx.world_collision_ref_pos = (new_x, new_y, new_z)

        old_pos = ctx.player_pos
        # ROOT PHYSICS SANITY CLAMP: a spring/suspension instability on rough or
        # steep terrain can grow the integrated velocity past float32 range (->
        # inf via _f32) and then position to inf/NaN. A non-finite pos/vel then
        # crashes EVERY downstream consumer of it — the turn-torque terrain
        # sample, the suspension curve, and the wire serializers — on both the
        # per-client tick loop and the shared UDP thread. Commit ONLY finite,
        # magnitude-bounded state so a diverging tank self-heals instead of
        # poisoning the whole server. The speed cap (~40x max tank speed) and the
        # non-finite resets fire only on already-broken physics, so they never
        # change normal finite play. Surfaced by the A3 soak (2026-06-01).
        _MAX_TANK_SPEED = 1000.0
        if not math.isfinite(new_vel_x):
            new_vel_x = 0.0
        if not math.isfinite(new_vel_y):
            new_vel_y = 0.0
        if not math.isfinite(new_vel_z):
            new_vel_z = 0.0
        new_vel_x = max(-_MAX_TANK_SPEED, min(_MAX_TANK_SPEED, new_vel_x))
        new_vel_y = max(-_MAX_TANK_SPEED, min(_MAX_TANK_SPEED, new_vel_y))
        new_vel_z = max(-_MAX_TANK_SPEED, min(_MAX_TANK_SPEED, new_vel_z))
        if not (math.isfinite(new_x) and math.isfinite(new_y) and math.isfinite(new_z)):
            try:
                fx = old_pos[0] if math.isfinite(old_pos[0]) else 5050.0
                fy = old_pos[1] if math.isfinite(old_pos[1]) else 5000.0
                fz = old_pos[2] if math.isfinite(old_pos[2]) else 50.0
            except Exception:
                fx, fy, fz = 5050.0, 5000.0, 50.0
            print(f"[PHYSICS] Client {ctx.client_id}: non-finite integrated position "
                  f"({new_x},{new_y},{new_z}) -> hold ({fx:.1f},{fy:.1f},{fz:.1f}), zero vel")
            new_x, new_y, new_z = fx, fy, fz
            new_vel_x = new_vel_y = new_vel_z = 0.0
        ctx.player_pos = (new_x, new_y, new_z)
        ctx.player_vel = (new_vel_x, new_vel_y, new_vel_z)
        ctx.player_speed = vehicle_runtime_speed(
            new_vel_x,
            new_vel_y,
            new_vel_z,
            up_axis=self.up_axis,
        )
        ctx.player_pose["pos"] = ctx.player_pos
        ctx.player_pose["vel"] = ctx.player_vel
        # GOAL 8 (2026-06-04): per-step suspension/Z instrumentation. Default OFF.
        # Surfaces the softbody-spring state that drives the idle Z limit-cycle so
        # the bounce can be characterized empirically rather than guessed.
        if getattr(self, "goal8_zdebug", False) and ctx.entity_type == EntityType.TANK:
            tk = getattr(getattr(ctx, "session", None), "tick", 0) or 0
            if tk % int(getattr(self, "goal8_zdebug_every", 15) or 15) == 0:
                print(
                    f"[ZDBG] c{ctx.client_id} t{tk} dt={dt*1000:.1f}ms "
                    f"z={new_z:.4f} vz={new_vel_z:.4f} clamp={ground_clamp_dz:.4f} "
                    f"susp_lift={float(suspension_lift):.3f} "
                    f"susp_clr={('%.3f' % suspension_clearance) if suspension_clearance is not None else 'None'} "
                    f"susp_tgt={('%.3f' % suspension_target_clearance) if suspension_target_clearance is not None else 'None'} "
                    f"model={suspension_model} ovr={int(use_ground_override)} "
                    f"rel={ground_override_release_reason or '-'} grnd={ground_level:.3f}"
                )
        if getattr(ctx, "vehicle_physics", None) is not None:
            ctx.vehicle_physics.angular_velocity = ctx.angular_vel_yaw
        controller_now = time.monotonic()
        movement_history = getattr(ctx, "movement_input_history", None)
        try:
            movement_history_len = len(movement_history or [])
        except TypeError:
            movement_history_len = 0
        latest_history = None
        if movement_history:
            try:
                latest_history = list(movement_history)[-1]
            except (IndexError, TypeError):
                latest_history = None
        if not isinstance(latest_history, dict):
            latest_history = {}
        latest_nonzero_history = None
        if movement_history:
            try:
                for entry in reversed(list(movement_history)):
                    if not isinstance(entry, dict):
                        continue
                    try:
                        entry_fwd = float(entry.get("fwd", 0.0) or 0.0)
                        entry_strafe = float(entry.get("strafe", 0.0) or 0.0)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if abs(entry_fwd) > 0.05 or abs(entry_strafe) > 0.05:
                        latest_nonzero_history = entry
                        break
            except TypeError:
                latest_nonzero_history = None
        if not isinstance(latest_nonzero_history, dict):
            latest_nonzero_history = {}
        latest_history_time = 0.0
        try:
            latest_history_time = float(latest_history.get("time", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            latest_history_time = 0.0
        latest_nonzero_history_time = 0.0
        try:
            latest_nonzero_history_time = float(
                latest_nonzero_history.get("time", 0.0) or 0.0
            )
        except (TypeError, ValueError, OverflowError):
            latest_nonzero_history_time = 0.0
        last_action_time = float(getattr(ctx, "last_action_packet_time", 0.0) or 0.0)
        last_nonzero_time = float(
            getattr(ctx, "last_nonzero_move_input_time", 0.0) or 0.0
        )
        last_decoded_input = getattr(ctx, "last_decoded_input", {}) or {}
        if not isinstance(last_decoded_input, dict):
            last_decoded_input = {}
        try:
            last_decoded_fwd = float(last_decoded_input.get("fwd", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            last_decoded_fwd = 0.0
        try:
            last_decoded_strafe = float(last_decoded_input.get("strafe", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            last_decoded_strafe = 0.0

        ctx.debug_last_controller_step = {
            "controller_time": controller_now,
            "controller_tick": ctx.session.tick if ctx.session else None,
            "controller_physics_step_count": getattr(ctx, "physics_step_count", 0),
            "action_packet_count_at_controller": getattr(
                ctx, "action_packet_count", 0
            ),
            "action_update_count_at_controller": getattr(
                ctx, "action_update_count", 0
            ),
            "action_dump_count_at_controller": getattr(ctx, "action_dump_count", 0),
            "last_action_packet_type_at_controller": getattr(
                ctx, "last_action_packet_type", ""
            ),
            "last_action_client_tick_at_controller": getattr(
                ctx, "last_action_packet_client_tick", 0
            ),
            "last_action_age_s_at_controller": (
                max(0.0, controller_now - last_action_time)
                if last_action_time > 0.0
                else None
            ),
            "last_nonzero_move_input_age_s_at_controller": (
                max(0.0, controller_now - last_nonzero_time)
                if last_nonzero_time > 0.0
                else None
            ),
            "movement_history_len_at_controller": movement_history_len,
            "movement_history_latest_age_s_at_controller": (
                max(0.0, controller_now - latest_history_time)
                if latest_history_time > 0.0
                else None
            ),
            "movement_history_latest_fwd_at_controller": float(
                latest_history.get("fwd", 0.0) or 0.0
            ),
            "movement_history_latest_strafe_at_controller": float(
                latest_history.get("strafe", 0.0) or 0.0
            ),
            "movement_history_latest_client_tick_at_controller": int(
                latest_history.get("client_tick", 0) or 0
            ),
            "movement_history_latest_packet_type_at_controller": latest_history.get(
                "packet_type", ""
            ),
            "movement_history_latest_action_sequence_at_controller": int(
                latest_history.get("action_sequence", 0) or 0
            ),
            "movement_history_latest_nonzero_age_s_at_controller": (
                max(0.0, controller_now - latest_nonzero_history_time)
                if latest_nonzero_history_time > 0.0
                else None
            ),
            "movement_history_latest_nonzero_fwd_at_controller": float(
                latest_nonzero_history.get("fwd", 0.0) or 0.0
            ),
            "movement_history_latest_nonzero_strafe_at_controller": float(
                latest_nonzero_history.get("strafe", 0.0) or 0.0
            ),
            "movement_history_latest_nonzero_client_tick_at_controller": int(
                latest_nonzero_history.get("client_tick", 0) or 0
            ),
            "movement_history_latest_nonzero_packet_type_at_controller": (
                latest_nonzero_history.get("packet_type", "")
            ),
            "movement_history_latest_nonzero_action_sequence_at_controller": int(
                latest_nonzero_history.get("action_sequence", 0) or 0
            ),
            "last_decoded_forward_input_at_controller": last_decoded_fwd,
            "last_decoded_strafe_input_at_controller": last_decoded_strafe,
            "turn_input": (
                self.turn_sign * ctx.injected_turn
                if ctx.injected_turn is not None
                else (
                    self._normalize_turn_input_value(
                        ctx,
                        ctx.weapon_system.behavior_slots[BehaviorSlot.TURNING],
                    )
                    if getattr(ctx, "weapon_system", None) is not None
                    else 0.0
                )
            ),
            "forward_input": throttle_input,
            "strafe_input": strafe_input,
            "raw_forward_input_current": raw_throttle_input,
            "raw_strafe_input_current": raw_strafe_input,
            "movement_input_source": movement_input_source,
            "movement_input_delay_s": movement_input_delay_s,
            "movement_input_selection": dict(
                getattr(ctx, "debug_last_movement_input_selection", {}) or {}
            ),
            "thrust_input": (
                self._normalize_behavior_axis_value(
                    ctx,
                    tank_softbody_control_slot_value(ctx.weapon_system.behavior_slots),
                )
                if getattr(ctx, "weapon_system", None) is not None
                else 0.0
            ),
            "jumpjet_input": jumpjet_input,
            "pre_pos": pre_pos,
            "pre_vel": pre_vel,
            "old_heading": yaw,
            "new_heading": ctx.player_heading,
            "yaw_angular_velocity": ctx.angular_vel_yaw,
            "vehicle_physics_angular_velocity": (
                ctx.vehicle_physics.angular_velocity
                if getattr(ctx, "vehicle_physics", None) is not None
                else None
            ),
            "current_speed": current_speed,
            "current_fuel": current_fuel if ctx.entity_type == EntityType.TANK else None,
            "forward_mobility": forward_mobility,
            "turn_mobility": turn_mobility,
            "terrain_up": avg_up if self.terrain and self.up_axis == "z" and self.terrain_pitch_enabled else (0.0, 0.0, 1.0),
            "terrain_clearance_ratio": _clearance_ratio if self.terrain and self.up_axis == "z" and self.terrain_pitch_enabled else 1.0,
            "spring_state": dict(getattr(ctx, "debug_last_spring_state", {}) or {}),
            "terrain_gradient": (dh_dx, dh_dy) if self.terrain and self.up_axis == "z" and self.terrain_pitch_enabled else (0.0, 0.0),
            "terrain_contact": (contact_x, contact_y) if ctx.entity_type == EntityType.TANK else (0.0, 0.0),
            "terrain_contact_coupling_enabled": (
                bool(getattr(self, "tank_terrain_contact_coupling_enabled", False))
                if ctx.entity_type == EntityType.TANK
                else False
            ),
            "drive_basis_source": drive_basis_source,
            "basis_forward": forward,
            "basis_right": right,
            "raw_impulse": (fwd_impulse, strafe_impulse),
            "drive_impulse_uncapped": drive_impulse_uncapped,
            "drive_impulse_capped": drive_impulse_capped,
            "terrain_contact_impulse": terrain_contact_impulse,
            "tank_vehicle_impulse": tank_vehicle_impulse,
            "gravity_impulse": gravity_impulse,
            "suspension_impulse": suspension_impulse,
            "pre_ground_vertical_impulse": pre_ground_vertical_impulse,
            "vertical_ground_cancelled": vertical_ground_cancelled,
            "move_impulse": (impulse_x, impulse_y, impulse_z),
            "ground_level": ground_level,
            "ground_level_source": ground_level_source,
            "terrain_ground_level": terrain_ground_level,
            "jump_jet_fired": jump_jet_fired,
            "jump_jet_impulse": jump_jet_impulse,
            "jump_jet_direction": jump_jet_direction,
            "jump_jet_direction_mode": getattr(self, "jump_jet_direction", "body"),
            "jump_jet_velocity_delta": jump_jet_velocity_delta,
            "jump_jet_altitude": jump_altitude,
            "jump_cooldown_remaining": ctx.jump_cooldown_remaining,
            "jump_spawn_lockout": ctx.jump_spawn_lockout,
            "suspension_lift": suspension_lift,
            "suspension_clearance": suspension_clearance,
            "suspension_target_clearance": suspension_target_clearance,
            "suspension_model": suspension_model,
            "softbody_target_average_height": (
                None if suspension_softbody is None else suspension_softbody.target_average_height
            ),
            "softbody_height_error": (
                None if suspension_softbody is None else suspension_softbody.height_error
            ),
            "softbody_height_ratio": (
                None if suspension_softbody is None else suspension_softbody.height_ratio
            ),
            "softbody_vehicle_throttle": (
                None if suspension_softbody is None else suspension_softbody.vehicle_throttle
            ),
            "softbody_stiffness": (
                None if suspension_softbody is None else suspension_softbody.softbody_stiffness
            ),
            "softbody_response_scale": (
                None if suspension_softbody is None else suspension_softbody.response_scale
            ),
            "softbody_support_accel": (
                None if suspension_softbody is None else suspension_softbody.support_accel
            ),
            "softbody_force_curve_input": (
                None if suspension_softbody is None else suspension_softbody.force_curve_input
            ),
            "softbody_force_bias_accel": (
                None if suspension_softbody is None else suspension_softbody.force_bias_accel
            ),
            "softbody_height_response_accel": (
                None if suspension_softbody is None else suspension_softbody.height_response_accel
            ),
            "softbody_damping_accel": (
                None if suspension_softbody is None else suspension_softbody.damping_accel
            ),
            "softbody_point_count": (
                None if suspension_softbody is None else suspension_softbody.point_count
            ),
            "softbody_point_forces": (
                None if suspension_softbody is None else suspension_softbody.point_forces
            ),
            "softbody_point_vertical_forces": (
                None if suspension_softbody is None else suspension_softbody.point_vertical_forces
            ),
            "softbody_point_clearances": (
                None if suspension_softbody is None else suspension_softbody.point_clearances
            ),
            "softbody_point_height_errors": (
                None if suspension_softbody is None else suspension_softbody.point_height_errors
            ),
            "softbody_point_normal_z": (
                None if suspension_softbody is None else suspension_softbody.point_normal_z
            ),
            "softbody_point_force_curve_inputs": (
                None if suspension_softbody is None else suspension_softbody.point_force_curve_inputs
            ),
            "softbody_point_height_curve_factors": (
                None if suspension_softbody is None else suspension_softbody.point_height_curve_factors
            ),
            "softbody_point_blend_factors": (
                None if suspension_softbody is None else suspension_softbody.point_blend_factors
            ),
            "softbody_point_shear_corrections": (
                None if suspension_softbody is None else suspension_softbody.point_shear_corrections
            ),
            "softbody_point_velocity_z": (
                None if suspension_softbody is None else suspension_softbody.point_velocity_z
            ),
            "softbody_point_decompile_force_magnitudes": (
                None if suspension_softbody is None else suspension_softbody.point_decompile_force_magnitudes
            ),
            "softbody_point_decompile_react_blends": (
                None if suspension_softbody is None else suspension_softbody.point_decompile_react_blends
            ),
            "softbody_point_decompile_fast_reacts": (
                None if suspension_softbody is None else suspension_softbody.point_decompile_fast_reacts
            ),
            "softbody_point_decompile_slow_reacts": (
                None if suspension_softbody is None else suspension_softbody.point_decompile_slow_reacts
            ),
            "softbody_scalar_stretch_ratio": (
                None if suspension_softbody is None else suspension_softbody.scalar_stretch_ratio
            ),
            "softbody_scalar_stretch_source": (
                None if suspension_softbody is None else suspension_softbody.scalar_stretch_source
            ),
            "softbody_scalar_stretch_speed": (
                None if suspension_softbody is None else suspension_softbody.scalar_stretch_speed
            ),
            "softbody_scalar_stretch_denominator": (
                None if suspension_softbody is None else suspension_softbody.scalar_stretch_denominator
            ),
            "ground_level_override": ctx.ground_level_override,
            "ground_override_ref_pos": ground_override_ref_pos,
            "ground_override_ref_terrain_level": ground_override_ref_terrain_level,
            "ground_override_terrain_change": ground_override_terrain_change,
            "ground_override_released": ground_override_released,
            "ground_override_release_reason": ground_override_release_reason,
            "linear_damp": linear_damp,
            "horizontal_damp": horizontal_damp,
            "tank_ground_contact_damp": tank_ground_contact_damp,
            "acceleration": (acc_x, acc_y, acc_z),
            "world_collision_ref_pos": getattr(ctx, "world_collision_ref_pos", None),
            "world_collision_bounds_dirty": bool(getattr(ctx, "world_collision_bounds_dirty", False)),
            "motion_collision": dict(getattr(ctx, "debug_last_motion_collision", {}) or {}),
            "jump_jet_collision_guard": jump_jet_collision_guard_debug,
            "tank_terrain_projection_guard": tank_terrain_projection_guard_debug,
            "terrain_contact_probe": dict(getattr(ctx, "debug_last_terrain_contact_probe", {}) or {}),
            "pos": ctx.player_pos,
            "vel": ctx.player_vel,
        }
        _upp_ta = time.perf_counter() if _upp_timing else 0.0
        body_attitude = self._update_player_surface_attitude(
            ctx,
            ctx.player_heading,
            dt=dt,
            suspension_lift=suspension_lift,
            suspension_point_forces=(
                suspension_softbody.point_forces if suspension_softbody is not None else None
            ),
            suspension_point_blend_factors=(
                suspension_softbody.point_blend_factors if suspension_softbody is not None else None
            ),
            spring_state_override=spring_state_for_attitude,
        )
        if _upp_timing:
            ctx._upp_attitude_ms = (time.perf_counter() - _upp_ta) * 1000.0
        ctx.debug_last_controller_step["body_rotation_source"] = body_attitude["source"]
        ctx.debug_last_controller_step["body_rotation"] = body_attitude["rotation"]
        ctx.debug_last_controller_step["body_up"] = body_attitude["up"]
        ctx.debug_last_controller_step["body_matrix"] = body_attitude.get("matrix")
        ctx.debug_last_controller_step["body_target_rotation"] = body_attitude.get("target_rotation")
        ctx.debug_last_controller_step["body_angular_velocity"] = body_attitude.get("angular_velocity")
        ctx.debug_last_controller_step["spring_attitude"] = body_attitude.get("spring_attitude")

        # Log position changes periodically
        dist = math.sqrt(
            (new_x - old_pos[0]) ** 2 +
            (new_y - old_pos[1]) ** 2 +
            (new_z - old_pos[2]) ** 2
        )
        if dist > 10.0:
            print(f"[POS] Client {ctx.client_id} at ({new_x:.1f}, {new_y:.1f}, {new_z:.1f}) yaw={math.degrees(yaw):.1f} deg")


    # Building AABB half-extents matching client-side table
    _BUILDING_HALF_EXTENTS = {
        25: (12.0, 12.0), 26: (8.0, 8.0), 27: (6.0, 6.0), 28: (5.0, 5.0),
        29: (10.0, 10.0), 30: (7.0, 7.0), 31: (6.0, 6.0), 32: (8.0, 8.0),
        33: (5.0, 5.0), 34: (4.0, 4.0), 35: (6.0, 6.0), 36: (5.0, 5.0),
        37: (7.0, 7.0),
    }
    _TANK_RADIUS = 4.0
    _BUILDING_HALF_HEIGHT = 20.0
    # Decompile: Physics.c:5380 — penetration slop thresholds
    _PENETRATION_SLOP_SLEEPING = 0.001
    _PENETRATION_SLOP_DEFAULT = 0.005

    # Entity collision table — from exe VA 0x5730C0, stride 0x28 (40 bytes)
    # Format: {mass, elasticity, friction, restitution}
    # See docs/decompile-findings-2026-03-16.md §1
    _ENTITY_COLLISION_TABLE: dict = {
        EntityType.TANK:             {"mass": 6700.0,  "elasticity": 0.40, "friction": 0.20, "restitution": 2.00},
        EntityType.SCOUT:            {"mass": 6700.0,  "elasticity": 0.50, "friction": 0.20, "restitution": 2.00},
        EntityType.ASSAULT_PLATFORM: {"mass": 19000.0, "elasticity": 0.10, "friction": 0.20, "restitution": 2.00},
        EntityType.BOMBER:           {"mass": 5700.0,  "elasticity": 0.10, "friction": 0.20, "restitution": 2.00},
        EntityType.TRANSPORT:        {"mass": 6700.0,  "elasticity": 0.10, "friction": 0.20, "restitution": 2.00},
    }
    _ENTITY_COLLISION_DEFAULT = {"mass": 6700.0, "elasticity": 0.40, "friction": 0.20, "restitution": 2.00}

    _ENTITY_WORLD_MODEL_NAMES = {
        EntityType.TANK: ("tank_1", "tank_2"),
        EntityType.SCOUT: ("scout_1", "scout_2"),
    }
    _PROJECTILE_MODEL_NAMES = {
        EntityType.FLAK_SHELL: ("flak_shell",),
        EntityType.PULSE_SHELL: ("pulse_shell",),
        EntityType.SHORT_MISSILE: ("s_missile_1", "s_missile_2"),
        EntityType.HUNTER: ("missile_1", "missile_2"),
        EntityType.HEAVY_MISSILE: ("p_rocket_1", "p_rocket_2"),
        EntityType.MINE: ("mine",),
        EntityType.TORPEDO: ("torpedo",),
        EntityType.PIERCER: ("rocket_1", "rocket_2"),
        EntityType.THUMPER: ("p_rocket_1", "p_rocket_2"),
    }

    @staticmethod
    def _select_team_model_name(model_names, team_id: int) -> Optional[str]:
        if not model_names:
            return None
        if len(model_names) == 1:
            return model_names[0]
        if team_id == 1:
            return model_names[1]
        return model_names[0]

    def _process_jump_jets(self, ctx: ClientContext, addr: tuple):
        """
        Process jump jet input.
        Called after decoding ACTION_DUMP or ACTION_UPDATE.
        """
        # Jump jets are now applied in _update_player_position() so the server
        # and Python prediction use the same fixed-step rising-edge model.
        # Keep this packet-arrival hook as a no-op compatibility shim.
        return

    def _on_jump_jet_triggered(self, ctx: ClientContext, player_id: int, impulse: float, new_vel_z: float):
        """Callback when a jump jet is triggered."""
        print(f"[JUMP] Jump triggered for player {player_id}: impulse={impulse}, vel_up={new_vel_z:.1f}")
        # Loopback fork retired (2026-06-02): the burst queues for every client.
        burst_count = int(getattr(self, "jump_jet_correction_burst_count", 0) or 0)
        if burst_count > 0:
            # The original Tank controller has no local jumpjet impulse, so OG
            # clients need a short authoritative burst to make the custom
            # server-side hop visible instead of waiting for sparse organic
            # STATE_REQUEST replies.
            ctx.force_correction_once = True
            ctx.correction_burst_remaining = max(
                int(getattr(ctx, "correction_burst_remaining", 0) or 0),
                burst_count - 1,
            )
            ctx.correction_burst_interval_s = float(
                getattr(self, "jump_jet_correction_burst_interval", 0.05) or 0.05
            )
            ctx.last_correction_send = 0.0
            print(
                f"[JUMP] Queued correction burst x{burst_count} "
                f"@ {ctx.correction_burst_interval_s:.2f}s for client {ctx.client_id}"
            )

        # Send visual/audio feedback via chat — debug clients only.
        # OG client crashes on unexpected COMM_MESSAGE during spawn.
        if ctx.tcp_handler and self._debug_comm_allowed_for_client(ctx):
            msg = build_chat_message("*WHOOSH*", source_id=player_id)
            ctx.tcp_handler.send(msg)
