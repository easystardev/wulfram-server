"""
Weapon system for Wulfram 2 server.
Handles client input packets and spawns projectiles.
"""

import struct
import time
import math
import os
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Callable, List

from .codec import BitReader
from .packets import (
    VEC_POS_MAX, VEC_POS_RANGE,
    VEC_VEL_MAX, VEC_VEL_RANGE,
    VEC_ROT_MAX, VEC_ROT_RANGE,
)
from wulfram2_protocol.entities import (  # noqa: F401 — re-export for existing importers
    BehaviorSlot,
    EntityType,
    WeaponType,
    WEAPON_NAMES,
    TANK_WEAPON_SLOTS,
    Projectile,
)


class WeaponSystem:
    """
    Handles weapon firing and projectile management.
    """

    def __init__(self):
        self.behavior_slots: List[float] = [0.0] * 22  # Current behavior values
        # Sub-tick input timing: records when the TURNING slot last changed
        # so the tick loop can split the physics step at the transition point.
        self.turn_input_change_time: float = 0.0  # monotonic time of last turn input change
        self.turn_input_prev_value: float = 0.0    # value before the change
        self.turn_input_change_client_tick: int = 0  # client tick (ms) at last TURNING change
        self.prev_fire_state: float = 0.0
        self.fire_cooldown: float = 0.0  # Time until can fire again
        self.last_update_time: float = time.monotonic()
        self.projectiles: List[Projectile] = []
        self.next_entity_id: int = 1000  # Start projectile IDs high

        # Callbacks
        self.on_chain_gun_fire: Optional[Callable] = None
        self.on_projectile_spawn: Optional[Callable] = None

        # Weapon configs
        self.chain_gun_cooldown: float = 0.1  # 100ms between shots
        self.pulse_cannon_cooldown: float = 0.5  # 500ms between shots

        # === EXPERIMENTAL: Projectile velocity settings ===
        # Try different values to see what client displays correctly
        # Options: 50, 75, 100, 150, 200 (units per second)
        self.pulse_shell_speed: float = 75.0  # Medium velocity

        # Coordinate system config (defaults to z-up). Override with WULFRAM_UP_AXIS.
        self.up_axis = os.environ.get("WULFRAM_UP_AXIS", "z").lower()
        # Pitch is often noisy; keep disabled unless explicitly enabled.
        self.use_pitch = os.environ.get("WULFRAM_USE_PITCH", "0") == "1"
        # Aim offsets/inversions for tuning projectile direction.
        try:
            self.aim_yaw_offset = math.radians(float(os.environ.get("WULFRAM_AIM_YAW_OFFSET_DEG", "0.0")))
        except ValueError:
            self.aim_yaw_offset = 0.0
        try:
            self.aim_pitch_offset = math.radians(float(os.environ.get("WULFRAM_AIM_PITCH_OFFSET_DEG", "0.0")))
        except ValueError:
            self.aim_pitch_offset = 0.0
        self.aim_yaw_invert = os.environ.get("WULFRAM_AIM_YAW_INVERT", "0") == "1"
        self.aim_pitch_invert = os.environ.get("WULFRAM_AIM_PITCH_INVERT", "0") == "1"
        # Constant heading offset (radians) applied to yaw before projectile direction
        # calculation. Compensates for any reference frame mismatch between the
        # server's tracked heading (starts at 0.0) and the client's actual facing.
        # Tune empirically: fire at spawn, observe direction, adjust offset.
        try:
            self.heading_offset = math.radians(float(os.environ.get("WULFRAM_HEADING_OFFSET_DEG", "0.0")))
        except ValueError:
            self.heading_offset = 0.0

        # Projectile spawn tuning via hardpoints (shape data).
        # Hardpoint data is useful for tuning but not yet stable across maps/models.
        # Default to simple forward offset from authoritative server pose.
        self.projectile_spawn_mode = os.environ.get("WULFRAM_PROJECTILE_SPAWN_MODE", "offset").lower()
        self.projectile_hardpoint_name = os.environ.get("WULFRAM_PROJECTILE_HARDPOINT", "gun").strip()
        try:
            self.projectile_spawn_offset = float(os.environ.get("WULFRAM_PROJECTILE_SPAWN_OFFSET", "2.0"))
        except ValueError:
            self.projectile_spawn_offset = 2.0
        try:
            self.projectile_barrel_right = float(os.environ.get("WULFRAM_PROJECTILE_BARREL_RIGHT", "0.0"))
        except ValueError:
            self.projectile_barrel_right = 0.0
        try:
            self.projectile_barrel_up = float(os.environ.get("WULFRAM_PROJECTILE_BARREL_UP", "0.2"))
        except ValueError:
            self.projectile_barrel_up = 0.2
        try:
            self.shape_coord_scale = float(os.environ.get("WULFRAM_SHAPE_COORD_SCALE", "4096.0"))
        except ValueError:
            self.shape_coord_scale = 4096.0
        self.hardpoint_order = os.environ.get("WULFRAM_HARDPOINT_ORDER", "zxy").lower()
        self.player_shape_override = os.environ.get("WULFRAM_PLAYER_SHAPE", "").strip()
        try:
            self.projectile_muzzle_push = float(os.environ.get("WULFRAM_PROJECTILE_MUZZLE_PUSH", "0.0"))
        except ValueError:
            self.projectile_muzzle_push = 0.0
        # Hardpoint origin mode: center (default) or base/min_y.
        self.hardpoint_origin_mode = os.environ.get("WULFRAM_HARDPOINT_ORIGIN", "center").lower()
        # Optional hardpoint axis tuning (sign flips / swap) for empirical alignment.
        try:
            self.hardpoint_forward_sign = float(os.environ.get("WULFRAM_HARDPOINT_FORWARD_SIGN", "1.0"))
        except ValueError:
            self.hardpoint_forward_sign = 1.0
        try:
            self.hardpoint_right_sign = float(os.environ.get("WULFRAM_HARDPOINT_RIGHT_SIGN", "1.0"))
        except ValueError:
            self.hardpoint_right_sign = 1.0
        try:
            self.hardpoint_up_sign = float(os.environ.get("WULFRAM_HARDPOINT_UP_SIGN", "1.0"))
        except ValueError:
            self.hardpoint_up_sign = 1.0
        self.hardpoint_swap_fr = os.environ.get("WULFRAM_HARDPOINT_SWAP_FR", "0") == "1"
        self._hardpoint_cache = {}
        self._shape_bounds_cache = {}

        # Quantizer config for behavior inputs (from TRANSLATION packet)
        # Defaults match server/wulfram/packets.py scalar configs.
        self.control_bits = int(os.environ.get("WULFRAM_CONTROL_BITS", "16"))
        self.control_max = float(os.environ.get("WULFRAM_CONTROL_MAX", "1000.0"))
        self.control_range = float(os.environ.get("WULFRAM_CONTROL_RANGE", "2000.0"))
        self.zoom_bits = int(os.environ.get("WULFRAM_ZOOM_BITS", str(self.control_bits)))
        self.zoom_max = float(os.environ.get("WULFRAM_ZOOM_MAX", str(self.control_max)))
        self.zoom_range = float(os.environ.get("WULFRAM_ZOOM_RANGE", str(self.control_range)))
        # Slot index bit width (ACTION_UPDATE uses quantizer index 15 from TRANSLATION; default 16 bits)
        self.slot_index_bits = int(os.environ.get("WULFRAM_SLOT_INDEX_BITS", "16"))
        self.debug_inputs = os.environ.get("WULFRAM_DEBUG_INPUT", "0") == "1"
        self.debug_input_time = os.environ.get("WULFRAM_DEBUG_INPUT_TIME", "0") == "1"

        # Player state (set by server)
        self.player_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.player_rot: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.player_team: int = 1
        self.player_id: int = 1337

        # Current weapon (from slot 4)
        # Default to pulse cannon for testing projectile visibility
        self.current_weapon: int = 4  # WeaponType.PULSE_CANNON

    def decode_action_dump(self, data: bytes) -> bool:
        """
        Decode ACTION_DUMP packet (0x09).
        Format: [opcode:1] [tick:4] [frame:4] [slot_values:bit-packed]

        Slot encoding (from decompilation):
        - Slot 5 (UPWARD_THRUST) uses zoom quantizer (index 10)
        - Slots 1-3 and 6-7 use control quantizer (index 11)
        - All other slots use 1 bit (boolean)

        Returns True if successfully decoded.
        """
        if len(data) < 9:
            return False
        try:
            # Header: opcode + tick + frame (bit-aligned)
            reader = BitReader(data[1:])
            tick = reader.read_bits(32)
            frame = reader.read_bits(32)

            # Debug: capture raw values before decoding
            raw_values = {}

            # Slot indices 1..21 (slot 0 is not transmitted)
            for slot_idx in range(1, 22):
                try:
                    # Track raw bits for debugging control slots
                    start_byte = reader.byte_pos
                    start_bit = reader.bit_pos
                    value, raw_val = self._read_behavior_value_debug(slot_idx, reader)
                    raw_values[slot_idx] = raw_val
                except ValueError:
                    break
                if 0 <= slot_idx < len(self.behavior_slots):
                    # Never let ACTION_DUMP set analog control slots to non-zero.
                    # Only ACTION_UPDATE should set these — the client encodes
                    # with off-by-one raw values between the two packet types
                    # (e.g. raw 32750 vs 32751 for MOVING_FORWARD), and
                    # ACTION_DUMP can race ahead of ACTION_UPDATE at key press.
                    if slot_idx in (BehaviorSlot.TURNING, BehaviorSlot.MOVING_FORWARD,
                                    BehaviorSlot.MOVING_SIDEWAYS) and abs(value) > 0.01:
                        continue
                    self.behavior_slots[slot_idx] = value

            # Debug: log movement-related slots with RAW values
            turn = self.behavior_slots[BehaviorSlot.TURNING]
            fwd = self.behavior_slots[BehaviorSlot.MOVING_FORWARD]
            strafe = self.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS]
            thrust = self.behavior_slots[BehaviorSlot.UPWARD_THRUST]
            fire = self.behavior_slots[BehaviorSlot.FIRE]

            # Show raw bits for movement slots (1=turn, 2=fwd, 3=strafe)
            raw_turn = raw_values.get(1, 0)
            raw_fwd = raw_values.get(2, 0)
            raw_strafe = raw_values.get(3, 0)
            raw_thrust = raw_values.get(5, 0)
            raw_s6 = raw_values.get(6, 0)
            raw_s7 = raw_values.get(7, 0)

            # Always log raw data and decoded values for debugging
            print(f"[ACTION_DUMP] tick={tick} frame={frame}")
            s6 = self.behavior_slots[BehaviorSlot.SLOT6]
            s7 = self.behavior_slots[BehaviorSlot.SLOT7]
            print(
                f"[ACTION_DUMP] raw_hex={data[9:17].hex()} | "
                f"turn: raw={raw_turn} val={turn:.3f} | fwd: raw={raw_fwd} val={fwd:.3f} | "
                f"s6: raw={raw_s6} val={s6:.3f} | s7: raw={raw_s7} val={s7:.3f}"
            )
            print(
                f"[ACTION_DUMP] strafe: raw={raw_strafe} val={strafe:.3f} | "
                f"thrust: raw={raw_thrust} val={thrust:.3f} | fire={fire:.1f}"
            )
            return True
        except Exception as e:
            print(f"[WEAPON] Error decoding ACTION_DUMP: {e}")
            import traceback
            traceback.print_exc()
            return False

    def decode_action_update(self, data: bytes) -> bool:
        """
        Decode ACTION_UPDATE packet (0x0A).

        Format: [opcode][count:u8][tick:u32][frame:u32] then bit-packed updates.
        Slot index bit width comes from TRANSLATION quantizer 15 (default 16 bits).

        Returns True if successfully decoded.
        """
        if len(data) < 10:
            return False

        reader = BitReader(data[1:])
        try:
            count = reader.read_bits(8)
            tick = reader.read_bits(32)
            frame = reader.read_bits(32)
        except ValueError:
            return False

        updated_any = False
        changed = []
        max_updates = min(count, 64)

        try:
            for _ in range(max_updates):
                slot_idx = reader.read_bits(self.slot_index_bits)
                value = self._read_behavior_value_update(slot_idx, reader)
                if 0 <= slot_idx < len(self.behavior_slots):
                    old_val = self.behavior_slots[slot_idx]
                    self.behavior_slots[slot_idx] = value
                    updated_any = True
                    if abs(value - old_val) > 0.01:
                        changed.append((slot_idx, value))
                    if slot_idx == BehaviorSlot.TURNING and abs(value - old_val) > 0.01:
                        self.turn_input_prev_value = old_val
                        self.turn_input_change_time = time.monotonic()
                        self.turn_input_change_client_tick = tick
                    if slot_idx == BehaviorSlot.FIRE and abs(value - old_val) > 0.01:
                        print(f"[WEAPON] FIRE: {old_val:.3f} -> {value:.3f}")
        except ValueError:
            # Partial packet - still apply what we decoded so far.
            pass

        if updated_any:
            # Always log ACTION_UPDATE changes for debugging
            if changed:
                print(f"[ACTION_UPDATE] tick={tick} frame={frame} changes={changed}")
            return True

        return False

    def _decode_quantized(self, raw: int, bits: int, max_val: float, range_val: float) -> float:
        """Decode quantized integer to float using ValueQuantizer formula."""
        if raw == 0 or bits <= 0:
            return 0.0
        if range_val <= 0.0:
            return 0.0
        denom = (1 << bits) - 2
        if denom <= 0:
            return 0.0
        value = max_val - ((raw - 1) * range_val) / denom
        min_val = max_val - range_val
        if value > max_val:
            value = max_val
        if value < min_val:
            value = min_val
        return value

    def _read_behavior_value(self, slot_idx: int, reader: BitReader) -> float:
        """Read a behavior slot value from the bitstream."""
        if slot_idx == BehaviorSlot.UPWARD_THRUST:
            raw = reader.read_bits(self.zoom_bits)
            return self._decode_quantized(raw, self.zoom_bits, self.zoom_max, self.zoom_range)
        if slot_idx in (BehaviorSlot.UNUSED0, BehaviorSlot.TURNING, BehaviorSlot.MOVING_FORWARD,
                        BehaviorSlot.MOVING_SIDEWAYS, BehaviorSlot.SLOT6, BehaviorSlot.SLOT7):
            raw = reader.read_bits(self.control_bits)
            return self._decode_quantized(raw, self.control_bits, self.control_max, self.control_range)
        # Other slots are binary
        raw = reader.read_bits(1)
        return 1.0 if raw else 0.0

    def _read_behavior_value_update(self, slot_idx: int, reader: BitReader) -> float:
        """
        Read a behavior slot value from ACTION_UPDATE.
        Digital actions are 1-bit values in a continuous bitstream.
        """
        if slot_idx == BehaviorSlot.UPWARD_THRUST:
            raw = reader.read_bits(self.zoom_bits)
            return self._decode_quantized(raw, self.zoom_bits, self.zoom_max, self.zoom_range)
        if slot_idx in (BehaviorSlot.UNUSED0, BehaviorSlot.TURNING, BehaviorSlot.MOVING_FORWARD,
                        BehaviorSlot.MOVING_SIDEWAYS, BehaviorSlot.SLOT6, BehaviorSlot.SLOT7):
            raw = reader.read_bits(self.control_bits)
            return self._decode_quantized(raw, self.control_bits, self.control_max, self.control_range)
        # Other slots are binary (1 bit)
        raw = reader.read_bits(1)
        return 1.0 if raw else 0.0

    def _read_behavior_value_debug(self, slot_idx: int, reader: BitReader) -> tuple:
        """Read a behavior slot value from the bitstream, returning (decoded_value, raw_bits)."""
        if slot_idx == BehaviorSlot.UPWARD_THRUST:
            raw = reader.read_bits(self.zoom_bits)
            return self._decode_quantized(raw, self.zoom_bits, self.zoom_max, self.zoom_range), raw
        if slot_idx in (BehaviorSlot.UNUSED0, BehaviorSlot.TURNING, BehaviorSlot.MOVING_FORWARD,
                        BehaviorSlot.MOVING_SIDEWAYS, BehaviorSlot.SLOT6, BehaviorSlot.SLOT7):
            raw = reader.read_bits(self.control_bits)
            return self._decode_quantized(raw, self.control_bits, self.control_max, self.control_range), raw
        # Other slots are binary
        raw = reader.read_bits(1)
        return (1.0 if raw else 0.0), raw

    def update(self, dt: float = None) -> List[Projectile]:
        """
        Process weapon state changes and return any new projectiles.
        Call this after decoding input packets.
        """
        now = time.monotonic()
        if dt is None:
            dt = now - self.last_update_time
        self.last_update_time = now

        # Update cooldowns
        self.fire_cooldown = max(0.0, self.fire_cooldown - dt)

        # DISABLED: Reading weapon from slot 4 during testing
        # The client sends weapon slot 0 (chaingun), which overwrites our forced
        # pulse cannon setting. Skip this until weapon switching is fully implemented.
        #
        # # Update current weapon from slot 4 (5-bit quantized, so 0-31 range)
        # weapon_val = self.behavior_slots[BehaviorSlot.WEAPON_SELECT]
        # # Convert normalized 0-1 back to weapon slot index (0-31)
        # weapon_slot = int(weapon_val * 31 + 0.5)  # Round to nearest
        # if weapon_slot != self.current_weapon:
        #     old_name = WEAPON_NAMES.get(self.current_weapon, f"Slot {self.current_weapon}")
        #     new_name = WEAPON_NAMES.get(weapon_slot, f"Slot {weapon_slot}")
        #     print(f"[WEAPON] Weapon changed: {old_name} -> {new_name} (raw={weapon_val:.3f})")
        #     self.current_weapon = weapon_slot

        new_projectiles = []

        # Check fire state (behavior slot FIRE)
        fire_val = self.behavior_slots[BehaviorSlot.FIRE]

        # Debug: log when fire value changes significantly
        if abs(fire_val - self.prev_fire_state) > 0.01:
            print(f"[WEAPON] Fire state changed: {self.prev_fire_state:.3f} -> {fire_val:.3f}")

        # Use higher threshold - fire button sends ~0.06 when pressed
        # Threshold of 0.05 filters more noise while still catching real presses
        fire_threshold = 0.05
        fire_pressed = fire_val > fire_threshold
        prev_pressed = self.prev_fire_state > fire_threshold

        # EDGE DETECTION: Only fire on RISING EDGE (transition from not-pressed to pressed)
        # This prevents continuous firing from held buttons or noise
        rising_edge = fire_pressed and not prev_pressed

        # Fire trigger: rising edge AND cooldown ready
        fire_trigger = rising_edge and self.fire_cooldown <= 0

        if fire_trigger:
            weapon_name = WEAPON_NAMES.get(self.current_weapon, f"Slot {self.current_weapon}")

            if self.current_weapon == WeaponType.CHAIN_GUN:
                # Chain gun: instant hit, no projectile
                self._fire_chain_gun()
                self.fire_cooldown = self.chain_gun_cooldown

            elif self.current_weapon == WeaponType.PULSE_CANNON:
                # Pulse cannon: energy projectile
                proj = self._fire_pulse_cannon()
                if proj:
                    new_projectiles.append(proj)
                    self.projectiles.append(proj)
                self.fire_cooldown = self.pulse_cannon_cooldown

            elif self.current_weapon in TANK_WEAPON_SLOTS:
                # Other Tank weapons - log and send feedback
                print(f"[WEAPON] {weapon_name} fired! pos={self.player_pos}")
                if self.on_chain_gun_fire:
                    # Use chain gun callback for now to send feedback
                    self.on_chain_gun_fire(
                        pos=self.player_pos,
                        rot=self.player_rot,
                        team=self.player_team,
                        weapon_name=weapon_name
                    )
                self.fire_cooldown = 0.5  # Generic cooldown

            else:
                # Invalid weapon slot
                print(f"[WEAPON] Fire attempt with invalid slot {self.current_weapon}")
                self.fire_cooldown = 0.5

        self.prev_fire_state = fire_val

        # Update projectiles (remove expired)
        self.projectiles = [p for p in self.projectiles
                          if now - p.spawn_time < p.lifetime]

        return new_projectiles

    def _fire_chain_gun(self):
        """Fire chain gun (instant hit weapon)."""
        print(f"[WEAPON] Chain gun fired! pos={self.player_pos}")

        if self.on_chain_gun_fire:
            self.on_chain_gun_fire(
                pos=self.player_pos,
                rot=self.player_rot,
                team=self.player_team
            )

    def _fire_pulse_cannon(self) -> Optional[Projectile]:
        """Fire pulse cannon (spawns projectile entity)."""
        # Use player's facing direction from yaw (+ optional pitch)
        # Rotation order: (roll, pitch, yaw)
        pitch = self.player_rot[1]
        yaw = self.player_rot[2]
        print(f"[WEAPON] Pulse cannon fired! pos={self.player_pos} yaw={math.degrees(yaw):.1f}deg")

        if not self.use_pitch:
            pitch = 0.0

        # Apply aim tuning (invert + offset + heading reference frame offset)
        if self.aim_yaw_invert:
            yaw = -yaw
        if self.aim_pitch_invert:
            pitch = -pitch
        yaw += self.aim_yaw_offset + self.heading_offset
        pitch += self.aim_pitch_offset

        if self.up_axis == "z":
            # Z-up: X/Y horizontal, Z vertical
            # Client rotation matrix M[6] = -sin(euler_Y) for forward Z component
            fwd_x = math.cos(pitch) * math.cos(yaw)
            fwd_y = math.cos(pitch) * math.sin(yaw)
            fwd_z = -math.sin(pitch)
        else:
            # Y-up: X/Z horizontal, Y vertical (default)
            fwd_x = math.cos(pitch) * math.cos(yaw)
            fwd_y = -math.sin(pitch)
            fwd_z = math.cos(pitch) * math.sin(yaw)

        vel_x = self.pulse_shell_speed * fwd_x
        vel_y = self.pulse_shell_speed * fwd_y
        vel_z = self.pulse_shell_speed * fwd_z

        spawn_x, spawn_y, spawn_z = self.player_pos
        spawn_offset = 0.0
        hardpoint_shape = None
        hardpoint_raw = None
        hardpoint_local = None

        spawn_mode = self.projectile_spawn_mode
        if spawn_mode == "hardpoint":
            shape_name = self.player_shape_override or (f"tank_{self.player_team}" if self.player_team in (1, 2) else "tank_1")
            hp = self._get_shape_hardpoint(shape_name, self.projectile_hardpoint_name)
            if hp is not None:
                hardpoint_shape = shape_name
                hardpoint_raw, hardpoint_local = hp
                # Model axes from shapes: X=right, Y=up, Z=forward.
                model_x, model_y, model_z = hardpoint_local
                if self.hardpoint_origin_mode in ("base", "min_y"):
                    bounds = self._get_shape_bounds(shape_name)
                    if bounds is not None:
                        min_y = bounds.get("min_y", 0.0)
                        model_y -= min_y
                # Map model axes into world forward/right/up so offsets track tank pose.
                hp_forward = model_z
                hp_right = model_x
                hp_up = model_y
                if self.hardpoint_swap_fr:
                    hp_forward, hp_right = hp_right, hp_forward
                hp_forward *= self.hardpoint_forward_sign
                hp_right *= self.hardpoint_right_sign
                hp_up *= self.hardpoint_up_sign
                cy = math.cos(yaw)
                sy = math.sin(yaw)
                if self.up_axis == "z":
                    # World Z-up: rotate in X/Y plane (forward/right).
                    rot_x = hp_forward * cy - hp_right * sy
                    rot_y = hp_forward * sy + hp_right * cy
                    rot_z = hp_up
                else:
                    # World Y-up: rotate in X/Z plane (forward/right).
                    rot_x = hp_forward * cy - hp_right * sy
                    rot_y = hp_up
                    rot_z = hp_forward * sy + hp_right * cy
                spawn_x += rot_x
                spawn_y += rot_y
                spawn_z += rot_z

        if hardpoint_shape is None:
            if spawn_mode == "center":
                spawn_offset = 0.0
            else:
                # Default forward offset if not using hardpoints.
                spawn_offset = self.projectile_spawn_offset
                spawn_x += spawn_offset * fwd_x
                spawn_y += spawn_offset * fwd_y
                spawn_z += spawn_offset * fwd_z
            # Barrel right/up offsets rotate with heading
            barrel_right = self.projectile_barrel_right
            barrel_up = self.projectile_barrel_up
            if barrel_right != 0.0 or barrel_up != 0.0:
                cy = math.cos(yaw)
                sy = math.sin(yaw)
                if self.up_axis == "z":
                    # Z-up: right is perpendicular to forward in XY plane, up is Z
                    spawn_x += barrel_right * (-sy)
                    spawn_y += barrel_right * cy
                    spawn_z += barrel_up
                else:
                    # Y-up: right is perpendicular to forward in XZ plane, up is Y
                    spawn_x += barrel_right * (-sy)
                    spawn_y += barrel_up
                    spawn_z += barrel_right * cy

        if self.projectile_muzzle_push:
            spawn_x += self.projectile_muzzle_push * fwd_x
            spawn_y += self.projectile_muzzle_push * fwd_y
            spawn_z += self.projectile_muzzle_push * fwd_z

        debug_context = {
            "player_pos": self.player_pos,
            "player_rot": self.player_rot,
            "aim_source": getattr(self, "projectile_aim_source", None),
            "yaw": yaw,
            "pitch": pitch,
            "roll": self.player_rot[0],
            "use_pitch": self.use_pitch,
            "up_axis": self.up_axis,
            "aim_yaw_offset_deg": math.degrees(self.aim_yaw_offset),
            "aim_pitch_offset_deg": math.degrees(self.aim_pitch_offset),
            "heading_offset_deg": math.degrees(self.heading_offset),
            "aim_yaw_invert": self.aim_yaw_invert,
            "aim_pitch_invert": self.aim_pitch_invert,
            "forward": (fwd_x, fwd_y, fwd_z),
            "spawn_mode": spawn_mode,
            "spawn_offset": spawn_offset,
            "barrel_right": self.projectile_barrel_right,
            "barrel_up": self.projectile_barrel_up,
            "spawn_pos": (spawn_x, spawn_y, spawn_z),
            "vel": (vel_x, vel_y, vel_z),
            "speed": self.pulse_shell_speed,
            "hardpoint_shape": hardpoint_shape,
            "hardpoint_raw": hardpoint_raw,
            "hardpoint_local": hardpoint_local,
            "hardpoint_order": self.hardpoint_order,
            "hardpoint_model": hardpoint_local,
            "hardpoint_world_offset": (rot_x, rot_y, rot_z) if hardpoint_shape else None,
            "hardpoint_origin_mode": self.hardpoint_origin_mode,
            "hardpoint_forward_sign": self.hardpoint_forward_sign,
            "hardpoint_right_sign": self.hardpoint_right_sign,
            "hardpoint_up_sign": self.hardpoint_up_sign,
            "hardpoint_swap_fr": self.hardpoint_swap_fr,
            "shape_scale": self.shape_coord_scale,
            "muzzle_push": self.projectile_muzzle_push,
        }

        entity_id = self.next_entity_id
        self.next_entity_id += 1

        proj = Projectile(
            entity_id=entity_id,
            entity_type=EntityType.PULSE_SHELL,
            owner_id=self.player_id,
            team=self.player_team,
            pos=(spawn_x, spawn_y, spawn_z),
            vel=(vel_x, vel_y, vel_z),
            spawn_time=time.monotonic(),
            lifetime=5.0,
            debug_context=debug_context
        )

        if self.on_projectile_spawn:
            self.on_projectile_spawn(proj)

        return proj

    def set_player_state(self, pos: Tuple[float, float, float],
                        rot: Tuple[float, float, float],
                        team: int, player_id: int):
        """Update player state for weapon calculations."""
        self.player_pos = pos
        self.player_rot = rot
        self.player_team = team
        self.player_id = player_id

    def _get_shape_hardpoint(self, shape_name: str, hardpoint_name: str):
        """Return ((raw_x,raw_y,raw_z), (local_x,local_y,local_z)) for a named hardpoint."""
        if not shape_name or not hardpoint_name:
            return None

        cache_key = (shape_name, hardpoint_name, self.shape_coord_scale, self.hardpoint_order)
        if cache_key in self._hardpoint_cache:
            return self._hardpoint_cache[cache_key]

        shapes_zip = os.environ.get("WULFRAM_SHAPES_ZIP", "").strip()
        if shapes_zip:
            zip_path = Path(shapes_zip)
        else:
            root = Path(__file__).resolve().parents[2]
            zip_path = root / "slurpysoft-wulfram" / "data" / "shapes.zip"
            if not zip_path.exists():
                zip_path = root / "wulfram2-extracted" / "disk1" / "data" / "shapes.zip"

        if not zip_path.exists():
            self._hardpoint_cache[cache_key] = None
            return None

        name_bytes = (hardpoint_name + "\x00").encode("ascii", errors="ignore")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if shape_name not in zf.namelist():
                    self._hardpoint_cache[cache_key] = None
                    return None
                data = zf.read(shape_name)
        except Exception:
            self._hardpoint_cache[cache_key] = None
            return None

        idx = data.find(name_bytes)
        if idx < 0 or idx + len(name_bytes) + 12 > len(data):
            self._hardpoint_cache[cache_key] = None
            return None

        raw = struct.unpack("<iii", data[idx + len(name_bytes): idx + len(name_bytes) + 12])
        order = self.hardpoint_order
        if len(order) != 3 or set(order) != {"x", "y", "z"}:
            order = "zxy"

        mapping = {order[0]: raw[0], order[1]: raw[1], order[2]: raw[2]}
        local = (
            mapping["x"] / self.shape_coord_scale,
            mapping["y"] / self.shape_coord_scale,
            mapping["z"] / self.shape_coord_scale,
        )
        result = (raw, local)
        self._hardpoint_cache[cache_key] = result
        return result

    def _get_shape_bounds(self, shape_name: str):
        """Return cached model bounds in model coordinates (x/y/z)."""
        if shape_name in self._shape_bounds_cache:
            return self._shape_bounds_cache[shape_name]

        shapes_zip = os.environ.get("WULFRAM_SHAPES_ZIP", "").strip()
        if shapes_zip:
            zip_path = Path(shapes_zip)
        else:
            root = Path(__file__).resolve().parents[2]
            zip_path = root / "slurpysoft-wulfram" / "data" / "shapes.zip"
            if not zip_path.exists():
                zip_path = root / "wulfram2-extracted" / "disk1" / "data" / "shapes.zip"

        if not zip_path.exists():
            self._shape_bounds_cache[shape_name] = None
            return None

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if shape_name not in zf.namelist():
                    self._shape_bounds_cache[shape_name] = None
                    return None
                data = zf.read(shape_name)
        except Exception:
            self._shape_bounds_cache[shape_name] = None
            return None

        # Parse header to reach vertex data.
        pos = data.find(b"\x00")
        if pos < 0:
            self._shape_bounds_cache[shape_name] = None
            return None
        pos += 1
        if pos + 2 > len(data):
            self._shape_bounds_cache[shape_name] = None
            return None
        tex_count = struct.unpack("<H", data[pos:pos + 2])[0]
        pos += 2
        for _ in range(tex_count):
            end = data.find(b"\x00", pos)
            if end < 0:
                self._shape_bounds_cache[shape_name] = None
                return None
            pos = end + 1
        if pos + 2 > len(data):
            self._shape_bounds_cache[shape_name] = None
            return None
        vert_count = struct.unpack("<H", data[pos:pos + 2])[0]
        pos += 2
        if vert_count <= 0:
            self._shape_bounds_cache[shape_name] = None
            return None

        min_x = min_y = min_z = None
        max_x = max_y = max_z = None
        for _ in range(vert_count):
            if pos + 12 > len(data):
                break
            z_raw, x_raw, y_raw = struct.unpack("<iii", data[pos:pos + 12])
            pos += 12
            x = x_raw / self.shape_coord_scale
            y = y_raw / self.shape_coord_scale
            z = z_raw / self.shape_coord_scale
            if min_x is None:
                min_x = max_x = x
                min_y = max_y = y
                min_z = max_z = z
            else:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_z = min(min_z, z)
                max_z = max(max_z, z)

        if min_x is None:
            self._shape_bounds_cache[shape_name] = None
            return None

        bounds = {
            "min_x": min_x, "max_x": max_x,
            "min_y": min_y, "max_y": max_y,
            "min_z": min_z, "max_z": max_z,
        }
        self._shape_bounds_cache[shape_name] = bounds
        return bounds

def build_projectile_spawn_packet(
    proj: Projectile,
    tick: int,
    *,
    include_local_state: bool = False,
    weapon_id: int = 0,
    health: float = 1.0,
    fuel: float = 1.0,
    entity_config: int = 0,
    is_static: bool = True,
) -> bytes:
    """
    Build UPDATE_ARRAY packet to spawn a projectile entity.

    Format based on wulf-forge's working implementation:
    - 4 bytes: tick (sequence_id)
    - 1 bit: has_local_stats (0)
    - 8 bits: entity_count
    - Per entity:
      - 32 bits: net_id (OID)
      - 1 bit: is_manned
      - 10 bits: update_mask (DEFINITION | POS = 0b0000000011)
      - 16 bits: bank_selector (0 for bank 0)
      - If DEFINITION bit:
        - 8 bits: unit_type
        - 8 bits: team_id
        - 8 bits: team_id again
        - 1 bit: is_teleport (spawn snap/teleport; should be 1 on entity create)
      - If POS bit:
        - 4 bits: precision_header (3 = max quality)
        - 16 bits each: x, y, z compressed
    """
    from .codec import BitWriter
    from .packets import _write_local_player_state

    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    # Local stats section (optional)
    _write_local_player_state(
        bw,
        include_local_state,
        weapon_id=weapon_id,
        health=health,
        fuel=fuel,
    )

    # Entity count: 8 bits
    bw.write_bits(8, 1)  # 1 entity

    # --- Entity data ---
    # OID: 32 bits
    bw.write_bits(32, proj.entity_id)

    # is_manned: 1 bit (projectiles are not manned)
    bw.write_bits(1, 0)

    # Update mask: 10 bits
    # Bit 0 = DEFINITION (create entity)
    # Bit 1 = POS (position data)
    # Wulf-forge-style creation uses DEFINITION | POS only.
    update_mask = 0b0000000011  # DEFINITION | POS
    bw.write_bits(10, update_mask)

    # Bank selector: 16 bits (0 for bank 0)
    bw.write_bits(16, 0)

    # --- DEFINITION block (bit 0 set) ---
    # Decomp: unit_type, team_id, team_id, is_static (wulf-forge default)
    # NOTE: "is_static" behaves like a spawn-time snap/teleport flag. Wulf-forge sets it
    # to 1 on entity creation; leaving it 0 for moving entities can destabilize the client.
    bw.write_bits(8, proj.entity_type & 0xFF)
    config_val = entity_config if entity_config not in (None, 0) else proj.team
    bw.write_bits(8, config_val & 0xFF)
    bw.write_bits(8, proj.team & 0xFF)
    bw.write_bits(1, 1 if is_static else 0)

    # --- POS block (bit 1 set) ---
    # Position compression uses TranslationConfig from wulf-forge:
    # - header_bits: 4 (precision selector)
    # - max_total_bits: 16
    # - max_value: 4096.0, range: 8192.0 (so -4096 to +4096)
    #
    # With header=4, total=16: base_bits = 16 - 4 + 1 = 13
    # At priority 3 (max): current_bits = 13 + 3 = 16
    # denom = (1 << 16) - 2 = 65534
    # raw = ((max - val) * denom / range) + 1

    def compress_value(val: float, max_val: float, range_val: float, total_bits: int = 16) -> int:
        """Compress value using wulf-forge's algorithm.

        Args:
            val: Value to compress
            max_val: Maximum value in range
            range_val: Total range (max - min)
            total_bits: Number of bits to use (16 for position, 14 for velocity)
        """
        min_val = max_val - range_val

        # Special case: exactly 0
        if val == 0.0:
            return 0

        # Clamp to valid range
        if val > max_val:
            val = max_val
        if val < min_val:
            val = min_val

        # Denominator based on bit count
        denom = (1 << total_bits) - 2

        # Inverse quantization: raw = ((max - val) * denom / range) + 1
        delta = max_val - val
        scaled = (delta * denom) / range_val
        raw = int(scaled) + 1

        return raw

    def decode_value(raw: int, max_val: float, range_val: float, total_bits: int) -> float:
        """Decode quantized integer to float using ValueQuantizer formula."""
        if raw == 0:
            return 0.0
        denom = (1 << total_bits) - 2
        return max_val - ((raw - 1) * range_val) / denom

    debug_quant = os.environ.get("WULFRAM_DEBUG_PROJECTILE_QUANT", "0") == "1"

    # Position: 4-bit header + 16 bits × 3
    # wulf-forge VEC_POS: max=8192, range=16384 (covers ±8192)
    bw.write_bits(4, 15)  # priority=15 for 16 bits per component
    pos_raw = []
    pos_dec = []
    for v in proj.pos:
        compressed = compress_value(v, VEC_POS_MAX, VEC_POS_RANGE)
        pos_raw.append(compressed)
        if debug_quant:
            pos_dec.append(decode_value(compressed, VEC_POS_MAX, VEC_POS_RANGE, 16))
        bw.write_bits(16, compressed)

    if debug_quant:
        pos_fmt = ", ".join(f"{v:.2f}" for v in proj.pos)
        pos_dec_fmt = ", ".join(f"{v:.2f}" for v in pos_dec)
        print(
            f"[PROJ-QUANT] spawn id={proj.entity_id} tick={tick} "
            f"pos=({pos_fmt}) raw={pos_raw} dec=({pos_dec_fmt})"
        )

    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_projectile_update_packet(
    proj: Projectile,
    tick: int,
    dt: float,
    *,
    include_local_state: bool = False,
    weapon_id: int = 0,
    health: float = 1.0,
    fuel: float = 1.0,
) -> bytes:
    """
    Build UPDATE_ARRAY packet to update a projectile's position.
    Unlike spawn packet, this only includes POS (no DEFINITION).

    Args:
        proj: The projectile to update
        tick: Current game tick
        dt: Delta time since last update (for position calculation)
    """
    from .codec import BitWriter
    from .packets import _write_local_player_state

    # Update projectile position based on velocity
    new_pos = (
        proj.pos[0] + proj.vel[0] * dt,
        proj.pos[1] + proj.vel[1] * dt,
        proj.pos[2] + proj.vel[2] * dt,
    )
    # Update the projectile's stored position
    proj.pos = new_pos

    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    # Local stats (optional)
    _write_local_player_state(
        bw,
        include_local_state,
        weapon_id=weapon_id,
        health=health,
        fuel=fuel,
    )

    # Entity count: 1
    bw.write_bits(8, 1)

    # Entity header
    bw.write_bits(32, proj.entity_id)
    bw.write_bits(1, 0)  # is_manned = False

    # Update mask: POS | VEL | ROT
    # Client REQUIRES both position AND rotation for non-static entities!
    # From decompiled code: if rotation is missing, update is rejected with exit(0)
    # Bit 1 = POS, Bit 2 = VEL, Bit 3 = ROT
    update_mask = 0b0000001110  # POS | VEL | ROT
    bw.write_bits(10, update_mask)

    # Bank selector: ALWAYS written (client always reads it regardless of DEFINITION bit)
    # Value 0 selects bank 0 (quantizer array index 16)
    bw.write_bits(16, 0)

    # Position compression
    def compress_value(val: float, max_val: float, range_val: float, total_bits: int = 16) -> int:
        min_val = max_val - range_val
        if val == 0.0:
            return 0
        if val > max_val:
            val = max_val
        if val < min_val:
            val = min_val
        denom = (1 << total_bits) - 2
        delta = max_val - val
        scaled = (delta * denom) / range_val
        return int(scaled) + 1

    # Position: 4-bit header + 16 bits × 3
    # wulf-forge VEC_POS: max=8192, range=16384
    bw.write_bits(4, 15)
    for v in proj.pos:
        compressed = compress_value(v, VEC_POS_MAX, VEC_POS_RANGE, total_bits=16)
        bw.write_bits(16, compressed)

    # Velocity: 4-bit header + 16 bits × 3
    # wulf-forge VEC_VEL: max=1000, range=2000
    bw.write_bits(4, 15)
    for v in proj.vel:
        compressed = compress_value(v, VEC_VEL_MAX, VEC_VEL_RANGE, total_bits=16)
        bw.write_bits(16, compressed)

    # Rotation: 4-bit header + 16 bits × 3 (REQUIRED for non-static entities!)
    # wulf-forge VEC_ROT: max=6.3, range=12.6
    rot_from_vel = os.environ.get("WULFRAM_PROJECTILE_ROT_FROM_VEL", "1") == "1"
    if rot_from_vel:
        vx, vy, vz = proj.vel
        up_axis = os.environ.get("WULFRAM_UP_AXIS", "z").lower()
        if up_axis == "z":
            yaw = math.atan2(vy, vx) if (vx != 0.0 or vy != 0.0) else 0.0
            horiz = math.hypot(vx, vy)
            pitch = math.atan2(vz, horiz) if horiz != 0.0 else 0.0
        else:
            yaw = math.atan2(vz, vx) if (vx != 0.0 or vz != 0.0) else 0.0
            horiz = math.hypot(vx, vz)
            pitch = math.atan2(vy, horiz) if horiz != 0.0 else 0.0
        rot = (0.0, pitch, yaw)
    else:
        rot = (0.0, 0.0, 0.0)

    bw.write_bits(4, 15)
    for v in rot:
        compressed = compress_value(v, VEC_ROT_MAX, VEC_ROT_RANGE)
        bw.write_bits(16, compressed)

    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_projectile_destroy_packet(entity_id: int, tick: int) -> bytes:
    """
    Build UPDATE_ARRAY packet to destroy/remove a projectile entity.
    Sets the entity as "dead" or removes it from the client's entity table.

    Note: The exact mechanism for entity destruction is not yet verified.
    This may need adjustment based on client behavior.
    """
    from .codec import BitWriter

    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    # Local stats: no
    bw.write_bits(1, 0)

    # Entity count: 1
    bw.write_bits(8, 1)

    # Entity header
    bw.write_bits(32, entity_id)
    bw.write_bits(1, 0)  # is_manned = False

    # Update mask: 0 (no data, just presence in packet)
    # This might signal removal, or we may need a different approach
    update_mask = 0b0000000000
    bw.write_bits(10, update_mask)

    # Bank selector: ALWAYS written (client always reads it regardless of DEFINITION bit)
    bw.write_bits(16, 0)

    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_birth_notice_packet(proj: Projectile) -> bytes:
    """
    Build BIRTH_NOTICE packet (0x1E) to announce projectile spawn.
    This is an alternative to UPDATE_ARRAY for entity creation.
    """
    # BIRTH_NOTICE format (from decompilation):
    # - u16 entity_id (slot)
    # - u8 entity_type
    # - u8 team
    # - position data...

    packet = bytearray()
    packet.append(0x1E)  # BIRTH_NOTICE opcode

    # Entity slot (16 bits)
    packet.extend(struct.pack(">H", proj.entity_id & 0xFFFF))

    # Entity type (8 bits)
    packet.append(proj.entity_type & 0xFF)

    # Team (8 bits)
    packet.append(proj.team & 0xFF)

    # Owner ID (32 bits) - who fired it
    packet.extend(struct.pack(">I", proj.owner_id))

    # Position (3x float32 as fixed16 or raw)
    for v in proj.pos:
        # Use 16.16 fixed point
        fixed = int(v * 65536.0)
        packet.extend(struct.pack(">i", fixed))

    # Velocity (3x float32 as fixed16)
    for v in proj.vel:
        fixed = int(v * 65536.0)
        packet.extend(struct.pack(">i", fixed))

    return bytes(packet)
