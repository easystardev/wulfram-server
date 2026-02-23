"""
Packet definitions and builder functions.
Pure functions that return packet payloads - no I/O.

Protocol constants, packet types, and shared encoding helpers are imported
from the shared wulfram2_protocol module. This file contains server-side
packet builder functions.
"""

import struct
import time
import math
import os
from typing import Optional, Tuple, List

from wulfram2_protocol.codec import BitWriter, pack_fixed16, frame_packet
from wulfram2_protocol.packets import (  # noqa: F401 — re-export for existing importers
    PacketType,
    PACKET_NAMES,
    get_packet_name,
    BEHAVIOR_HEADER_SIZE,
    BEHAVIOR_WEAPON_UNITS,
    BEHAVIOR_WEAPON_SLOTS,
    BEHAVIOR_WEAPON_SLOT_SIZE,
    VEC_POS_MAX, VEC_POS_RANGE,
    VEC_VEL_MAX, VEC_VEL_RANGE,
    VEC_ROT_MAX, VEC_ROT_RANGE,
    VEC_SPIN_MAX, VEC_SPIN_RANGE,
    HEALTH_MAX, HEALTH_RANGE,
    ENERGY_MAX, ENERGY_RANGE,
    HEALTH_NORMALIZED,
    HEALTH_RAW_MODE,
    ENTITY_VITALS_MODE,
    LOCAL_STATE_TURRET_HEADER_BITS,
    LOCAL_STATE_TURRET_PRIORITY,
    compress_value,
    compress_position,
    compress_rotation,
    encode_health_bits,
    write_local_player_state,
    write_update_array_entity,
    _read_float_env,
)

# Server-private aliases for underscore-prefixed callers in this file
_compress_value = compress_value
_compress_wulfforge = compress_value  # same algorithm
_compress_position = compress_position
_compress_rotation = compress_rotation
_encode_health_bits = encode_health_bits
_write_local_player_state = write_local_player_state
_write_update_array_entity = write_update_array_entity

# Server tick clock - ticks relative to server start (matches wulf-forge)
_SERVER_START = time.monotonic()

# Behavior packet feature toggles (wulf-forge-inspired).
BEHAVIOR_THRUSTERS = os.environ.get("WULFRAM_BEHAVIOR_THRUSTERS", "1") == "1"
BEHAVIOR_ACTIVE_EXTRAS = os.environ.get("WULFRAM_BEHAVIOR_ACTIVE_EXTRAS", "1") == "1"

# Behavior packet physics defaults.
BEHAVIOR_GROUND_FRICTION = _read_float_env("WULFRAM_BEHAVIOR_GROUND_FRICTION", 0.8)
BEHAVIOR_TURN_RATE = _read_float_env("WULFRAM_BEHAVIOR_TURN_RATE", 0.05)
BEHAVIOR_SUSPENSION_DAMPENING = _read_float_env("WULFRAM_BEHAVIOR_SUSP_DAMPENING", 1.3)
BEHAVIOR_MAX_ALTITUDE = _read_float_env("WULFRAM_BEHAVIOR_MAX_ALTITUDE", 3.25)
BEHAVIOR_GRAVITY_PCT = _read_float_env("WULFRAM_BEHAVIOR_GRAVITY_PCT", 1.0)


def get_ticks() -> int:
    """Get current tick count (ms since server start), matching wulf-forge."""
    return int((time.monotonic() - _SERVER_START) * 1000) & 0xFFFFFFFF


# ============ Packet Builders ============

def build_hello_udp_config(server_ip: str, server_port: int) -> bytes:
    """Build HELLO packet with UDP config (subcmd 1)."""
    subcmd = b'\x01'
    ip_bytes = (server_ip + '\x00').encode('ascii')
    udp_config = struct.pack(">H", server_port)  # UDP port (2627 = 0x0A43)
    udp_config += struct.pack(">H", 1)  # Count (number of IPs)
    udp_config += struct.pack(">H", len(ip_bytes))  # IP string length
    udp_config += ip_bytes
    return b'\x13' + subcmd + udp_config


def build_hello_version(version: int = 0x4E89) -> bytes:
    """Build HELLO packet with version."""
    # SubCmd 0x00 + Version as signed 32-bit int
    return b'\x13\x00' + struct.pack(">i", version)


def build_hello_session_key(session_key: str) -> bytes:
    """Build HELLO packet with session key."""
    key_bytes = (session_key + '\x00').encode('ascii')
    return b'\x13\x02' + struct.pack(">H", len(key_bytes)) + key_bytes


def build_hello_verified() -> bytes:
    """Build HELLO packet with verified subcmd (no payload)."""
    return b'\x13\x03'


def build_ping_request() -> bytes:
    """Build PING_REQUEST packet (0x0B) with current timestamp."""
    ticks = int(time.time() * 1000) & 0xFFFFFFFF
    return b'\x0B' + struct.pack(">I", ticks)


def build_identified_udp() -> bytes:
    """Build IDENTIFIED_UDP packet."""
    return b'\x4D'


def build_login_status(code: int, is_donor: bool = False) -> bytes:
    """Build LOGIN_STATUS packet."""
    return b'\x22' + (b'\x01' if is_donor else b'\x00') + struct.pack("B", code)


def build_player(entity_id: int, spectator: bool = True) -> bytes:
    """Build PLAYER packet (0x17)."""
    return b'\x17' + struct.pack(">I", entity_id) + (b'\x01' if spectator else b'\x00')


def build_team_info() -> bytes:
    """Build TEAM_INFO packet with two teams."""
    def pack_string(s: str) -> bytes:
        encoded = s.encode('ascii') + b'\x00'
        return struct.pack(">H", len(encoded)) + encoded

    payload = b'\x28'
    # Team 1 (Crimson Federation)
    payload += struct.pack("B", 1)
    payload += pack_string("Crimson Federation")
    payload += pack_string("Red Team")
    payload += pack_string("Crimson Base")
    payload += pack_string("The red team.")
    payload += pack_string("Azure Alliance Wins!")

    # Team 2 (Azure Alliance)
    payload += struct.pack("B", 2)
    payload += pack_string("Azure Alliance")
    payload += pack_string("Blue Team")
    payload += pack_string("Crimson Base")
    payload += pack_string("The blue team.")
    payload += pack_string("Crimson Federation Wins!")

    return payload


def build_world_stats(
    map_name: str = "crossroads",
    grid_rows: int = 1,
    grid_cols: int = 1,
    scale: float = 1.0,
) -> bytes:
    """Build WORLD_STATS packet."""
    map_bytes = (map_name + '\x00').encode('ascii')
    payload = b'\x16'
    payload += struct.pack(">H", len(map_bytes)) + map_bytes
    payload += struct.pack("B", grid_rows & 0xFF)
    payload += struct.pack("B", grid_cols & 0xFF)
    payload += pack_fixed16(scale)
    return payload


def build_bps_response(rate_index: int, approved: bool = True) -> bytes:
    """Build BPS response packet."""
    return b'\x4E' + struct.pack(">I", rate_index) + (b'\x01' if approved else b'\x00')


def build_reincarnate(code: int, message: str) -> bytes:
    """Build REINCARNATE packet."""
    msg_bytes = (message + '\x00').encode('ascii')
    return b'\x25' + struct.pack("B", code) + struct.pack(">H", len(msg_bytes)) + msg_bytes


def build_add_to_roster(player_id: int, entity_id: int, name: str, team: int,
                        clan: str = "", kills: int = 0, deaths: int = 0) -> bytes:
    """Build ADD_TO_ROSTER packet (0x1A).

    Format per decompile (GUESS5_PacketHandler_ADD_TO_ROSTER):
      u32 player_oid, u32 team_id, u16 kills, u16 deaths,
      string name, string clan, u16 stat_a, u16 stat_b,
      fixed16 ping, u32 flags
    """
    name_bytes = (name + '\x00').encode('ascii')
    clan_bytes = (clan + '\x00').encode('ascii')
    payload = b'\x1A'
    payload += struct.pack(">I", player_id)
    payload += struct.pack(">I", team & 0xFFFFFFFF)
    payload += struct.pack(">H", kills & 0xFFFF)
    payload += struct.pack(">H", deaths & 0xFFFF)
    payload += struct.pack(">H", len(name_bytes)) + name_bytes
    payload += struct.pack(">H", len(clan_bytes)) + clan_bytes
    payload += struct.pack(">H", 0)       # stat_a
    payload += struct.pack(">H", 0)       # stat_b
    payload += pack_fixed16(0.0)          # ping
    payload += struct.pack(">I", 0)       # flags
    return payload


def build_update_stats(player_id: int, entity_id: int, kills: int = 0,
                       deaths: int = 0, team_id: int = 0) -> bytes:
    """Build UPDATE_STATS packet (0x1C).

    Format per decompile (GUESS5_PacketHandler_UPDATE_STATS):
      u32 player_oid, u32 entity_oid, u16 kills, u16 deaths,
      u16 assists, u16 damage_dealt, u16 team_id,
      fixed16 ping, fixed16 score, u32 flags
    """
    payload = b'\x1C'
    payload += struct.pack(">I", player_id)
    payload += struct.pack(">I", entity_id)
    payload += struct.pack(">H", kills & 0xFFFF)
    payload += struct.pack(">H", deaths & 0xFFFF)
    payload += struct.pack(">H", 0)           # assists
    payload += struct.pack(">H", 0)           # damage_dealt
    payload += struct.pack(">H", team_id & 0xFFFF)
    payload += pack_fixed16(0.0)              # ping
    payload += pack_fixed16(0.0)              # score
    payload += struct.pack(">I", 0)           # flags
    return payload


def build_birth_notice(entity_id: int, owner_entity_id: Optional[int] = None) -> bytes:
    """Build BIRTH_NOTICE packet."""
    if owner_entity_id is None:
        owner_entity_id = entity_id
    return b'\x1E' + struct.pack(">I", entity_id) + struct.pack(">I", owner_entity_id)


def build_delete_object(tick: int, entity_ids: list, with_effects: bool = False) -> bytes:
    """Build DELETE_OBJECT packet (0x15)."""
    payload = struct.pack(">I", tick)
    payload += struct.pack("B", len(entity_ids))
    effects_byte = 1 if with_effects else 0
    for eid in entity_ids:
        payload += struct.pack(">I", eid)
        payload += struct.pack("B", effects_byte)
    return b'\x15' + payload


def build_game_clock(time_ms: int = 0, running: bool = True, round_time_ms: int = 30000) -> bytes:
    """Build GAME_CLOCK packet (0x2F)."""
    ticks = int(time.time() * 1000) & 0xFFFFFFFF
    payload = b'\x2F'
    payload += struct.pack(">I", ticks)
    payload += b'\x01' if running else b'\x00'
    payload += struct.pack(">I", 1)
    payload += struct.pack(">I", round_time_ms)
    return payload


def build_motd(message: str = "Welcome to Wulfram!") -> bytes:
    """Build MOTD packet (0x23)."""
    msg_bytes = (message + '\x00').encode('ascii')
    return b'\x23' + struct.pack(">H", len(msg_bytes)) + msg_bytes


def build_chat_message(message: str, source_id: int = 0, target_id: int = 0) -> bytes:
    """Build COMM_MESSAGE (chat) packet."""
    payload = b'\x1F'
    payload += struct.pack(">H", 0)
    payload += struct.pack(">I", target_id)
    payload += struct.pack(">H", 0)
    payload += struct.pack(">I", source_id)
    msg_bytes = (message + '\x00').encode('ascii')
    payload += struct.pack(">H", len(msg_bytes)) + msg_bytes
    return payload


def build_player_info(entity_oid: int, vehicle_type: int, pos: Tuple[float, float, float],
                      rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                      *, include_local_state: bool = False,
                      weapon_id: int = 0,
                      health: float = 1.0,
                      fuel: float = 1.0,
                      properties: int = 0,
                      frame_id: int | None = None,
                      ammo_count_bits: int = 0,
                      ammo_count: int = 0,
                      primary_turret_bits: int = 0,
                      primary_turret_angle: float = 0.0,
                      secondary_turret_bits: int = 0,
                      secondary_turret_angle: float = 0.0,
                      turret_max: float = 6.3,
                      turret_range: float = 12.6) -> bytes:
    """Build PLAYER_INFO packet (0x18) for spawning the local player's vehicle."""
    bw = BitWriter()
    bw.write_bits(32, entity_oid)

    _write_local_player_state(
        bw,
        include_local_state,
        weapon_id=weapon_id,
        health=health,
        fuel=fuel,
        ammo_count_bits=ammo_count_bits,
        ammo_count=ammo_count,
        primary_turret_bits=primary_turret_bits,
        primary_turret_angle=primary_turret_angle,
        secondary_turret_bits=secondary_turret_bits,
        secondary_turret_angle=secondary_turret_angle,
        turret_max=turret_max,
        turret_range=turret_range,
        include_ammo_turrets=include_local_state,
    )

    bw.write_bits(32, vehicle_type)
    actual_frame_id = frame_id if frame_id is not None else entity_oid
    bw.write_bits(32, actual_frame_id & 0xFFFFFFFF)
    bw.write_bits(8, properties & 0xFF)

    x, y, z = pos
    bw.write_bits(32, int(x * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(y * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(z * 65536.0) & 0xFFFFFFFF)

    rx, ry, rz = rot
    bw.write_bits(32, int(rx * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(ry * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(rz * 65536.0) & 0xFFFFFFFF)

    return b'\x18' + bw.get_bytes()


def build_udp_tank_packet_wf(
    net_id: int,
    unit_type: int,
    team_id: int,
    pos: Tuple[float, float, float],
    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    tick: Optional[int] = None,
    include_vitals: bool = False,
    weapon_id: int = 0,
    health: float = 1.0,
    energy: float = 1.0,
    health_mult_bits: Optional[int] = None,
    energy_mult_bits: Optional[int] = None,
    include_firing_mask: bool = False,
    firing_mask_13bits: int = 0,
    include_extras: bool = False,
    extra_a_bits: int = 1,
    extra_b_bits: int = 1,
) -> bytes:
    """Build a UDP TANK packet (0x18) using the Wulf-Forge bit layout."""
    if tick is None:
        tick = get_ticks()

    bw = BitWriter()
    bw.write_bits(32, tick)

    bw.write_bits(1, 1 if include_vitals else 0)
    if include_vitals:
        if health_mult_bits is None:
            health_mult_bits = _encode_health_bits(health, total_bits=10)
        if energy_mult_bits is None:
            energy_mult_bits = _encode_health_bits(energy, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE)
        bw.write_bits(5, weapon_id & 0x1F)
        bw.write_bits(10, health_mult_bits & 0x3FF)
        bw.write_bits(10, energy_mult_bits & 0x3FF)
        if include_firing_mask:
            bw.write_bits(13, firing_mask_13bits & 0x1FFF)
        if include_extras:
            bw.write_bits(8, extra_a_bits & 0xFF)
            bw.write_bits(8, extra_b_bits & 0xFF)

    bw.write_bits(32, unit_type & 0xFFFFFFFF)
    bw.write_bits(32, net_id & 0xFFFFFFFF)
    bw.write_bits(8, team_id & 0xFF)

    for value in pos:
        bw.write_bits(32, int(value * 65536.0) & 0xFFFFFFFF)
    for value in rot:
        bw.write_bits(32, int(value * 65536.0) & 0xFFFFFFFF)

    return b'\x18' + bw.get_bytes()


def build_tank(entity_type: int, oid: int, pos: Tuple[float, float, float],
               rot: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> bytes:
    """Build TANK/PLAYER_INFO packet for spawning vehicles (legacy)."""
    return build_player_info(oid, entity_type, pos, rot=rot)


def build_tank_packet(net_id: int, unit_type: int, pos: Tuple[float, float, float],
                      rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                      flags: int = 1, include_vitals: bool = True,
                      health: float = 1.0, energy: float = 1.0) -> bytes:
    """Build TankPacket (0x18) matching Wulf-Forge's format exactly."""
    import time
    ticks = int(time.monotonic() * 1000) & 0xFFFFFFFF

    bw = BitWriter()
    bw.write_bits(32, ticks)
    bw.write_bits(1, 1 if include_vitals else 0)

    if include_vitals:
        bw.write_bits(5, 0)
        bw.write_bits(10, _encode_health_bits(health, total_bits=10))
        bw.write_bits(10, _encode_health_bits(energy, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE))

    bw.write_bits(32, unit_type)
    bw.write_bits(32, net_id)
    bw.write_bits(8, flags)

    x, y, z = pos
    bw.write_bits(32, int(x * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(y * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(z * 65536.0) & 0xFFFFFFFF)

    rx, ry, rz = rot
    bw.write_bits(32, int(rx * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(ry * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(rz * 65536.0) & 0xFFFFFFFF)

    return b'\x18' + bw.get_bytes()


def build_update_array_empty(tick: int = 0) -> bytes:
    """Build empty UPDATE_ARRAY packet."""
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()
    bw.write_bits(1, 0)
    bw.write_bits(8, 0)
    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_update_array_heartbeat(tick: int, entity_id: int, include_health: bool = False,
                                 entity_type_index: int = 0,
                                 weapon_id: int = 0,
                                 health: float = 1.0,
                                 fuel: float = 1.0,
                                 ammo_count_bits: int = 0,
                                 ammo_count: int = 0,
                                 primary_turret_bits: int = 0,
                                 primary_turret_angle: float = 0.0,
                                 secondary_turret_bits: int = 0,
                                 secondary_turret_angle: float = 0.0,
                                 turret_max: float = 6.3,
                                 turret_range: float = 12.6,
                                 is_view_update: bool = False,
                                 rot: tuple = None,
                                 pos: tuple = None) -> bytes:
    """Build UPDATE_ARRAY/VIEW_UPDATE with entity heartbeat and optional health data."""
    if is_view_update:
        timestamp = get_ticks()
        header = b'\x0F' + struct.pack(">I", timestamp) + struct.pack(">I", tick)
    else:
        header = b'\x0E' + struct.pack(">I", tick)

    bw = BitWriter()

    _write_local_player_state(
        bw,
        include_health,
        weapon_id=weapon_id,
        health=health,
        fuel=fuel,
        ammo_count_bits=ammo_count_bits,
        ammo_count=ammo_count,
        primary_turret_bits=primary_turret_bits,
        primary_turret_angle=primary_turret_angle,
        secondary_turret_bits=secondary_turret_bits,
        secondary_turret_angle=secondary_turret_angle,
        turret_max=turret_max,
        turret_range=turret_range,
        include_ammo_turrets=include_health,
    )

    DUMMY_ENTITY_ID = 0xFFFFFFFE
    has_player_entity = rot is not None or pos is not None
    entity_count = 2 if has_player_entity else 1
    bw.write_bits(8, entity_count)

    bw.write_bits(32, DUMMY_ENTITY_ID)
    bw.write_bits(1, 1)
    bw.write_bits(10, 0)
    bw.write_bits(16, 0)

    if has_player_entity:
        mask = 0
        if pos is not None:
            mask |= (1 << 1)  # bit 1 = position
        if rot is not None:
            mask |= (1 << 3)  # bit 3 = rotation
        bw.write_bits(32, entity_id)
        bw.write_bits(1, 1)
        bw.write_bits(10, mask)
        bw.write_bits(16, 0)

        if pos is not None:
            bw.write_bits(4, 15)
            for v in pos:
                bw.write_bits(16, _compress_value(v, VEC_POS_MAX, VEC_POS_RANGE, total_bits=16))

        if rot is not None:
            bw.write_bits(4, 15)
            for v in rot:
                bw.write_bits(16, _compress_value(v, 6.3, 12.6, total_bits=16))

    return header + bw.get_bytes()


def build_update_array_player_update(tick: int, entity_id: int,
                                     pos: Tuple[float, float, float],
                                     vel: Tuple[float, float, float],
                                     rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                     *,
                                     include_pos: bool = True,
                                     include_vel: bool = True,
                                     include_rot: bool = True,
                                     include_spin: bool = False,
                                     spin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                     include_local_state: bool = True,
                                     include_entity_vitals: bool = False,
                                     is_manned: bool = True,
                                     weapon_id: int = 0,
                                     health: float = 1.0,
                                     fuel: float = 1.0,
                                     speed_scale: float = 1.0,
                                     ammo_count_bits: int = 0,
                                     ammo_count: int = 0,
                                     primary_turret_bits: int = 0,
                                     primary_turret_angle: float = 0.0,
                                     secondary_turret_bits: int = 0,
                                     secondary_turret_angle: float = 0.0,
                                     turret_max: float = 6.3,
                                     turret_range: float = 12.6) -> bytes:
    """Build UPDATE_ARRAY packet with player position/velocity updates."""
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    _write_local_player_state(
        bw,
        include_local_state,
        weapon_id=weapon_id,
        health=health,
        fuel=fuel,
        ammo_count_bits=ammo_count_bits,
        ammo_count=ammo_count,
        primary_turret_bits=primary_turret_bits,
        primary_turret_angle=primary_turret_angle,
        secondary_turret_bits=secondary_turret_bits,
        secondary_turret_angle=secondary_turret_angle,
        turret_max=turret_max,
        turret_range=turret_range,
    )

    bw.write_bits(8, 1)
    bw.write_bits(32, entity_id)
    bw.write_bits(1, 1 if is_manned else 0)

    update_mask = 0
    if include_pos:
        update_mask |= (1 << 1)
    if include_vel:
        update_mask |= (1 << 2)
    if include_rot:
        update_mask |= (1 << 3)
    if include_spin:
        update_mask |= (1 << 4)
    if include_entity_vitals:
        update_mask |= (1 << 5)
        update_mask |= (1 << 7)
    bw.write_bits(10, update_mask)
    bw.write_bits(16, 0)

    if include_pos:
        bw.write_bits(4, 15)
        for v in pos:
            bw.write_bits(16, _compress_value(v, VEC_POS_MAX, VEC_POS_RANGE, total_bits=16))

    if include_vel:
        bw.write_bits(4, 15)
        for v in vel:
            bw.write_bits(16, _compress_value(v, VEC_VEL_MAX, VEC_VEL_RANGE, total_bits=16))

    if include_rot:
        bw.write_bits(4, 15)
        for v in rot:
            bw.write_bits(16, _compress_value(v, 6.3, 12.6, total_bits=16))

    if include_spin:
        bw.write_bits(4, 15)
        for v in spin:
            bw.write_bits(16, _compress_value(v, VEC_SPIN_MAX, VEC_SPIN_RANGE, total_bits=16))

    if include_entity_vitals:
        if ENTITY_VITALS_MODE in ("health", "vitals"):
            bw.write_bits(10, _encode_health_bits(speed_scale, total_bits=10))
            bw.write_bits(10, _encode_health_bits(fuel, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE))
        else:
            bw.write_bits(10, _compress_value(speed_scale, 1.0, 1.0, total_bits=10))
            bw.write_bits(10, _compress_value(fuel, 1.0, 1.0, total_bits=10))

    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_view_update_player_update(tick: int, entity_id: int,
                                    pos: Tuple[float, float, float],
                                    vel: Tuple[float, float, float],
                                    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                    *,
                                    include_pos: bool = True,
                                    include_vel: bool = True,
                                    include_rot: bool = True,
                                    include_local_state: bool = True,
                                    include_entity_vitals: bool = False,
                                    is_manned: bool = True,
                                    weapon_id: int = 0,
                                    health: float = 1.0,
                                    fuel: float = 1.0,
                                    speed_scale: float = 1.0,
                                    ammo_count_bits: int = 0,
                                    ammo_count: int = 0,
                                    primary_turret_bits: int = 0,
                                    primary_turret_angle: float = 0.0,
                                    secondary_turret_bits: int = 0,
                                    secondary_turret_angle: float = 0.0,
                                    turret_max: float = 6.3,
                                    turret_range: float = 12.6,
                                    timestamp: Optional[int] = None) -> bytes:
    """Build VIEW_UPDATE packet (0x0F) with player position/velocity updates."""
    if timestamp is None:
        timestamp = get_ticks()
    header = struct.pack(">I", timestamp) + struct.pack(">I", tick)
    bw = BitWriter()

    _write_local_player_state(
        bw,
        include_local_state,
        weapon_id=weapon_id,
        health=health,
        fuel=fuel,
        ammo_count_bits=ammo_count_bits,
        ammo_count=ammo_count,
        primary_turret_bits=primary_turret_bits,
        primary_turret_angle=primary_turret_angle,
        secondary_turret_bits=secondary_turret_bits,
        secondary_turret_angle=secondary_turret_angle,
        turret_max=turret_max,
        turret_range=turret_range,
    )

    bw.write_bits(8, 1)
    bw.write_bits(32, entity_id)
    bw.write_bits(1, 1 if is_manned else 0)

    update_mask = 0
    if include_pos:
        update_mask |= (1 << 1)
    if include_vel:
        update_mask |= (1 << 2)
    if include_rot:
        update_mask |= (1 << 3)
    if include_entity_vitals:
        update_mask |= (1 << 5)
        update_mask |= (1 << 7)
    bw.write_bits(10, update_mask)
    bw.write_bits(16, 0)

    if include_pos:
        bw.write_bits(4, 15)
        for v in pos:
            bw.write_bits(16, _compress_value(v, VEC_POS_MAX, VEC_POS_RANGE, total_bits=16))

    if include_vel:
        bw.write_bits(4, 15)
        for v in vel:
            bw.write_bits(16, _compress_value(v, VEC_VEL_MAX, VEC_VEL_RANGE, total_bits=16))

    if include_rot:
        bw.write_bits(4, 15)
        for v in rot:
            bw.write_bits(16, _compress_value(v, 6.3, 12.6, total_bits=16))

    if include_entity_vitals:
        if ENTITY_VITALS_MODE in ("health", "vitals"):
            bw.write_bits(10, _encode_health_bits(speed_scale, total_bits=10))
            bw.write_bits(10, _encode_health_bits(fuel, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE))
        else:
            bw.write_bits(10, _compress_value(speed_scale, 1.0, 1.0, total_bits=10))
            bw.write_bits(10, _compress_value(fuel, 1.0, 1.0, total_bits=10))

    return b'\x0F' + header + bw.get_bytes()


def build_update_array_multi(tick: int,
                             *,
                             include_local_state: bool,
                             weapon_id: int = 0,
                             health: float = 1.0,
                             fuel: float = 1.0,
                             ammo_count_bits: int = 0,
                             ammo_count: int = 0,
                             primary_turret_bits: int = 0,
                             primary_turret_angle: float = 0.0,
                             secondary_turret_bits: int = 0,
                             secondary_turret_angle: float = 0.0,
                             turret_max: float = 6.3,
                             turret_range: float = 12.6,
                             entities: Optional[list] = None) -> bytes:
    """Build UPDATE_ARRAY packet with multiple entity updates."""
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    _write_local_player_state(
        bw,
        include_local_state,
        weapon_id=weapon_id,
        health=health,
        fuel=fuel,
        ammo_count_bits=ammo_count_bits,
        ammo_count=ammo_count,
        primary_turret_bits=primary_turret_bits,
        primary_turret_angle=primary_turret_angle,
        secondary_turret_bits=secondary_turret_bits,
        secondary_turret_angle=secondary_turret_angle,
        turret_max=turret_max,
        turret_range=turret_range,
        include_ammo_turrets=include_local_state,
    )

    entities = entities or []
    bw.write_bits(8, len(entities))
    for ent in entities:
        _write_update_array_entity(bw, **ent)

    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_view_update_multi(tick: int,
                            *,
                            include_local_state: bool,
                            weapon_id: int = 0,
                            health: float = 1.0,
                            fuel: float = 1.0,
                            ammo_count_bits: int = 0,
                            ammo_count: int = 0,
                            primary_turret_bits: int = 0,
                            primary_turret_angle: float = 0.0,
                            secondary_turret_bits: int = 0,
                            secondary_turret_angle: float = 0.0,
                            turret_max: float = 6.3,
                            turret_range: float = 12.6,
                            entities: Optional[list] = None,
                            timestamp: Optional[int] = None) -> bytes:
    """Build VIEW_UPDATE (0x0F) packet with a timestamp + update array payload."""
    if timestamp is None:
        timestamp = get_ticks()
    header = struct.pack(">I", timestamp) + struct.pack(">I", tick)
    bw = BitWriter()

    _write_local_player_state(
        bw,
        include_local_state,
        weapon_id=weapon_id,
        health=health,
        fuel=fuel,
        ammo_count_bits=ammo_count_bits,
        ammo_count=ammo_count,
        primary_turret_bits=primary_turret_bits,
        primary_turret_angle=primary_turret_angle,
        secondary_turret_bits=secondary_turret_bits,
        secondary_turret_angle=secondary_turret_angle,
        turret_max=turret_max,
        turret_range=turret_range,
        include_ammo_turrets=include_local_state,
    )

    entities = entities or []
    bw.write_bits(8, len(entities))
    for ent in entities:
        _write_update_array_entity(bw, **ent)

    return b'\x0F' + header + bw.get_bytes()


def build_update_array_create_tank(tick: int, entity_id: int, entity_type: int, team: int,
                                    pos: Tuple[float, float, float], behavior_type: int = 0,
                                    include_interp: bool = False, interp_bits: int = 16,
                                    include_health: bool = True,
                                    include_entity_vitals: bool = False,
                                    health: float = 1.0,
                                    fuel: float = 1.0,
                                    is_manned: bool = True,
                                    weapon_id: int = 2) -> bytes:
    """Build UPDATE_ARRAY that creates a tank entity with position inline."""
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    _write_local_player_state(bw, include_health, weapon_id=weapon_id, health=health, fuel=fuel, include_ammo_turrets=False)
    bw.write_bits(8, 1)

    bw.write_bits(32, entity_id)
    bw.write_bits(1, 1 if is_manned else 0)

    presence_flags = 0b0000001011
    if include_entity_vitals:
        presence_flags |= (1 << 5)
        presence_flags |= (1 << 7)
    bw.write_bits(10, presence_flags)
    bw.write_bits(16, 0)

    bw.write_bits(8, entity_type & 0xFF)
    config_val = behavior_type if behavior_type else team
    bw.write_bits(8, config_val & 0xFF)
    bw.write_bits(8, team & 0xFF)
    bw.write_bits(1, 0)

    bw.write_bits(4, 15)
    for coord in pos:
        _, quantized = _compress_position(coord)
        bw.write_bits(16, quantized)

    bw.write_bits(4, 15)
    for _ in range(3):
        _, quantized = _compress_rotation(0.0)
        bw.write_bits(16, quantized)

    if include_entity_vitals:
        bw.write_bits(10, _encode_health_bits(health, total_bits=10))
        bw.write_bits(10, _encode_health_bits(fuel, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE))

    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_update_array_teleport(tick: int, entity_id: int,
                                pos: Tuple[float, float, float],
                                rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                include_health: bool = True,
                                weapon_id: int = 2,
                                health: float = 1.0,
                                fuel: float = 1.0) -> bytes:
    """Build UPDATE_ARRAY that teleports an existing entity to a new position."""
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    _write_local_player_state(bw, include_health, weapon_id=weapon_id,
                              health=health, fuel=fuel,
                              include_ammo_turrets=False)

    bw.write_bits(8, 1)
    bw.write_bits(32, entity_id)
    bw.write_bits(1, 1)
    bw.write_bits(10, 0b0000001010)
    bw.write_bits(16, 0)

    bw.write_bits(4, 15)
    for coord in pos:
        _, quantized = _compress_position(coord)
        bw.write_bits(16, quantized)

    bw.write_bits(4, 15)
    for val in rot:
        _, quantized = _compress_rotation(val)
        bw.write_bits(16, quantized)

    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_update_array_spawn_points(tick: int, spawn_points: list) -> bytes:
    """Build UPDATE_ARRAY with spawn point entities (type 27 = Repair Pad)."""
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    bw.write_bits(1, 0)
    bw.write_bits(8, len(spawn_points))

    for sp in spawn_points:
        oid = sp['oid']
        team = sp['team']
        config = sp.get('config', team)
        x, y, z = sp.get('x', 100.0), sp.get('y', 10.0), sp.get('z', 100.0)

        bw.write_bits(32, oid)
        bw.write_bits(1, 0)
        bw.write_bits(10, 0x03)
        bw.write_bits(16, 0)

        bw.write_bits(8, 27)
        bw.write_bits(8, config & 0xFF)
        bw.write_bits(8, team & 0xFF)
        bw.write_bits(1, 0)  # Must be 0 — matches create_tank; 1 causes bitstream shift crash

        bw.write_bits(4, 15)
        for coord in (x, y, z):
            quantized = _compress_wulfforge(coord, max_val=VEC_POS_MAX, range_val=VEC_POS_RANGE, total_bits=16)
            bw.write_bits(16, quantized)

    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_view_update_spawn_points(tick: int,
                                   spawn_points: list,
                                   *,
                                   include_local_state: bool = False,
                                   weapon_id: int = 0,
                                   health: float = 1.0,
                                   fuel: float = 1.0,
                                   timestamp: Optional[int] = None) -> bytes:
    """Build VIEW_UPDATE (0x0F) with spawn point entities."""
    if timestamp is None:
        timestamp = get_ticks()
    header = struct.pack(">I", timestamp) + struct.pack(">I", tick)
    bw = BitWriter()

    _write_local_player_state(
        bw,
        include_local_state,
        weapon_id=weapon_id,
        health=health,
        fuel=fuel,
        include_ammo_turrets=include_local_state,
    )

    bw.write_bits(8, len(spawn_points))

    for sp in spawn_points:
        oid = sp['oid']
        team = sp['team']
        config = sp.get('config', team)
        x, y, z = sp.get('x', 100.0), sp.get('y', 10.0), sp.get('z', 100.0)

        bw.write_bits(32, oid)
        bw.write_bits(1, 0)
        bw.write_bits(10, 0x03)
        bw.write_bits(16, 0)

        bw.write_bits(8, 27)
        bw.write_bits(8, config & 0xFF)
        bw.write_bits(8, team & 0xFF)
        bw.write_bits(1, 0)  # Must be 0 — matches create_tank; 1 causes bitstream shift crash

        bw.write_bits(4, 15)
        for coord in (x, y, z):
            quantized = _compress_wulfforge(coord, max_val=VEC_POS_MAX, range_val=VEC_POS_RANGE, total_bits=16)
            bw.write_bits(16, quantized)

    return b'\x0F' + header + bw.get_bytes()


def build_behavior_packet() -> bytes:
    """Build BEHAVIOR packet (0x24) with game parameters."""
    payload = bytearray()
    behavior_log = os.environ.get("WULFRAM_BEHAVIOR_LOG", "0") == "1"

    # Section 1: Header (95 bytes)
    payload.append(0x00)
    payload += pack_fixed16(5.0)
    payload += pack_fixed16(10.0)
    payload += pack_fixed16(10.0)
    payload += pack_fixed16(10.0)
    payload += pack_fixed16(10.0)

    payload += struct.pack(">I", 20)
    payload += struct.pack(">I", 25000)
    payload += struct.pack(">I", 35000)

    payload += pack_fixed16(100.0)
    payload += struct.pack(">I", 1)
    payload += struct.pack(">I", 1)
    payload += pack_fixed16(1.0)

    for _ in range(11):
        payload += pack_fixed16(1.0)

    payload.append(0x01)
    payload.append(0x01)

    assert len(payload) == 95, f"Section 1 should be 95 bytes, got {len(payload)}"

    # Section 2: Weapons (2340 bytes)
    enable_slot0 = os.environ.get("WULFRAM_BEHAVIOR_SLOT0", "0") == "1"
    TANK_SLOT_CONFIG = {
        0:  [1, 0, 0, 0, 0],
    } if enable_slot0 else {}

    for _unit in range(4):
        for _slot in range(13):
            if _unit == 0 and _slot in TANK_SLOT_CONFIG:
                flags = TANK_SLOT_CONFIG[_slot]
                payload += bytes(flags)
            else:
                payload += b'\x00\x00\x00\x00\x00'
            payload += pack_fixed16(1.0)
            payload += struct.pack(">I", 0) * 5
            payload += pack_fixed16(100.0)
            payload += pack_fixed16(1000.0)
            payload += pack_fixed16(500.0)
            payload += pack_fixed16(1.0)

    assert len(payload) == 95 + 2340, f"After Section 2: expected 2435, got {len(payload)}"

    # Section 3: Unit Stats (468 bytes)
    for _ in range(39):
        payload += pack_fixed16(1.0)
        payload += pack_fixed16(100.0)
        payload += struct.pack(">I", 100)

    assert len(payload) == 95 + 2340 + 468, f"After Section 3: expected 2903, got {len(payload)}"

    # Section 4: Vehicle Physics (72 bytes)
    for _ in range(2):
        payload += pack_fixed16(20.0)
        payload += pack_fixed16(4.0)
        payload += struct.pack(">I", 700)
        payload += struct.pack(">I", 550)
        payload += pack_fixed16(BEHAVIOR_GROUND_FRICTION)
        payload += pack_fixed16(BEHAVIOR_TURN_RATE)
        payload += pack_fixed16(BEHAVIOR_SUSPENSION_DAMPENING)
        payload += struct.pack(">I", 0)
        payload += struct.pack(">I", 33000)

    assert len(payload) == 95 + 2340 + 468 + 72, f"After Section 4: expected 2975, got {len(payload)}"

    # Section 5: Hardpoints
    section5_start = len(payload)

    def _write_hardpoint_block(count: int, is_thruster: bool) -> None:
        payload.extend(struct.pack(">I", count))
        if count > 0:
            for i in range(count):
                if is_thruster:
                    wing_width = 2.0
                    lateral_bias = 0.0
                    forward_bias = 0.0
                    if i % 2 == 1:
                        x_pos = wing_width + lateral_bias
                    else:
                        x_pos = -wing_width + lateral_bias
                    y_pos = forward_bias
                    z_pos = -0.5
                    nx, ny, nz = 0.0, 0.0, -0.75
                else:
                    x_pos = 1.5 if (i % 2 == 1) else -1.5
                    y_pos = 2.0
                    z_pos = 0.5
                    nx, ny, nz = 0.0, 1.0, 0.0

                payload.extend(pack_fixed16(float(x_pos)))
                payload.extend(pack_fixed16(float(y_pos)))
                payload.extend(pack_fixed16(float(z_pos)))
                payload.extend(pack_fixed16(float(nx)))
                payload.extend(pack_fixed16(float(ny)))
                payload.extend(pack_fixed16(float(nz)))
                payload.extend(struct.pack(">I", 0))

        if is_thruster:
            payload.extend(pack_fixed16(-5.0))
        else:
            payload.extend(pack_fixed16(0.0))

    if BEHAVIOR_THRUSTERS:
        _write_hardpoint_block(2, True)
        _write_hardpoint_block(2, True)
        _write_hardpoint_block(2, True)
        _write_hardpoint_block(2, True)
    else:
        for _ in range(4):
            _write_hardpoint_block(0, False)

    section5_size = len(payload) - section5_start
    assert len(payload) == 2975 + section5_size, f"After Section 5: expected {2975 + section5_size}, got {len(payload)}"

    # Section 6: Active Vehicle Physics
    section6_start = len(payload)
    for i in range(3):
        if not BEHAVIOR_ACTIVE_EXTRAS:
            payload += pack_fixed16(4.5)
            payload += pack_fixed16(85.0)
            payload += pack_fixed16(69.7)
            payload += pack_fixed16(80.0)
            payload += pack_fixed16(2000.0)
            payload += pack_fixed16(BEHAVIOR_MAX_ALTITUDE)
            payload += pack_fixed16(BEHAVIOR_GRAVITY_PCT)
            continue

        if i == 0:
            payload += pack_fixed16(4.5)
            payload += pack_fixed16(85.0)
            payload += pack_fixed16(69.7)
            payload += pack_fixed16(80.0)
            payload += pack_fixed16(2000.0)
            payload += pack_fixed16(BEHAVIOR_MAX_ALTITUDE)
            payload += pack_fixed16(BEHAVIOR_GRAVITY_PCT)
        elif i == 1:
            payload += pack_fixed16(4.5)
            payload += pack_fixed16(85.0)
            payload += pack_fixed16(38.0)
            payload += pack_fixed16(72.0)
            payload += pack_fixed16(85.0)
            payload += pack_fixed16(2000.0)
            payload += pack_fixed16(4.9)
            payload += pack_fixed16(3.5)
            payload += pack_fixed16(BEHAVIOR_GRAVITY_PCT)
        else:
            payload += pack_fixed16(-2.5132741233144)
            payload += pack_fixed16(2.35619449060725)
            payload += pack_fixed16(80.0)
            payload += pack_fixed16(45.0)
            payload += pack_fixed16(0.5)
            payload += pack_fixed16(70.0)
            payload += pack_fixed16(110.0)
            payload += pack_fixed16(340.0)
            payload += pack_fixed16(1000.0)
            payload += pack_fixed16(1800.0)
            payload += pack_fixed16(2000.0)

    section6_size = len(payload) - section6_start
    expected_section6 = 84 if not BEHAVIOR_ACTIVE_EXTRAS else (7 + 9 + 11) * 4
    assert section6_size == expected_section6, f"Section 6 size mismatch: got {section6_size}, expected {expected_section6}"
    current_size = 2975 + section5_size + section6_size
    assert len(payload) == current_size, f"After Section 6: expected {current_size}, got {len(payload)}"

    # Padding to target size (3116)
    target_size = 3116
    packet = bytearray()
    packet.append(0x24)
    packet += payload

    padding_needed = target_size - len(packet)
    if padding_needed > 0:
        packet += b'\x00' * padding_needed
    if behavior_log:
        print(
            "[BEHAVIOR] "
            f"thrusters={int(BEHAVIOR_THRUSTERS)} extras={int(BEHAVIOR_ACTIVE_EXTRAS)} "
            f"section5={section5_size} section6={section6_size} "
            f"payload_len={len(payload)} packet_len={len(packet)}"
        )
    return bytes(packet)


def get_behavior_weapon_capability_counts(packet: Optional[bytes] = None) -> List[Tuple[int, int, int, int]]:
    """Return per-weapon-type capability counts derived from the BEHAVIOR packet."""
    if packet is None:
        packet = build_behavior_packet()
    if not packet:
        return []

    offset = 0
    if packet[0] == PacketType.BEHAVIOR:
        offset = 1

    offset += BEHAVIOR_HEADER_SIZE
    counts: List[Tuple[int, int, int, int]] = []

    for _unit in range(BEHAVIOR_WEAPON_UNITS):
        ammo = fire = active = cooldown = 0
        for _slot in range(BEHAVIOR_WEAPON_SLOTS):
            if offset + 5 > len(packet):
                return counts
            enabled = packet[offset] != 0
            if enabled:
                if packet[offset + 1]:
                    ammo += 1
                if packet[offset + 2]:
                    fire += 1
                if packet[offset + 3]:
                    active += 1
                if packet[offset + 4]:
                    cooldown += 1
            offset += BEHAVIOR_WEAPON_SLOT_SIZE
        counts.append((ammo, fire, active, cooldown))

    return counts


def build_translation_packet() -> bytes:
    """Build TRANSLATION/quantizer packet (0x32)."""
    payload = bytearray()
    payload.append(0x32)

    def _write_string(text: str) -> bytes:
        raw = (text + "\x00").encode("ascii")
        return struct.pack(">H", len(raw)) + raw

    def _write_entry(fixed_bits: int, max_total_bits: int, max_str: str, range_str: str) -> None:
        payload.extend(struct.pack(">I", fixed_bits))
        payload.extend(struct.pack(">I", 0))
        payload.extend(struct.pack(">I", max_total_bits))
        payload.extend(_write_string(max_str))
        payload.extend(_write_string(range_str))

    scalar_configs = [(16, 0, "1000.0", "2000.0") for _ in range(16)]
    # Slots 1-4: input axes — must match server decode (control_bits=16, max=1000, range=2000)
    # OG client uses these to configure ValueQuantizer for ACTION_UPDATE encoding
    # Slot 1 (weapon type): must be 5 bits to match write_local_player_state's hardcoded
    # 5-bit weapon field.  Client reads weapon using this quantizer from TRANSLATION.
    # Mismatch (16 vs 5) causes bitstream shift → crash in Render_prepare_frame.
    scalar_configs[1] = (5, 0, "1000.0", "2000.0")
    # Slot 2 (entity_type in UPDATE_ARRAY definition block): client reads via
    # g_network_quantizer_array[0x20] = entry 2.  Must be 8 bits to match
    # the 8-bit entity_type written by build_update_array_create_tank and
    # build_update_array_spawn_points.
    scalar_configs[2] = (8, 0, "1.0", "1.0")
    # Slot 3 (parent_id / team_id in definition block): client reads via
    # g_network_quantizer_array[0x30] = entry 3.  Used TWICE (parent + team),
    # each 8 bits.
    scalar_configs[3] = (8, 0, "1.0", "1.0")
    scalar_configs[5] = (10, 0, f"{HEALTH_MAX}", f"{HEALTH_RANGE}")
    scalar_configs[8] = (10, 0, f"{ENERGY_MAX}", f"{ENERGY_RANGE}")
    scalar_configs[13] = (8, 0, "1.0", "1.0")
    scalar_configs[14] = (8, 0, "1.0", "1.0")

    for cfg in scalar_configs:
        _write_entry(*cfg)

    vector_templates = [
        (4, 16, f"{VEC_POS_MAX}", f"{VEC_POS_RANGE}"),
        (4, 16, f"{VEC_VEL_MAX}", f"{VEC_VEL_RANGE}"),
        (4, 16, f"{VEC_ROT_MAX}", f"{VEC_ROT_RANGE}"),
        (4, 16, f"{VEC_SPIN_MAX}", f"{VEC_SPIN_RANGE}"),
    ]

    for _ in range(3):
        for cfg in vector_templates:
            _write_entry(*cfg)

    return bytes(payload)


# --- TRANSIENT_ARRAY (0x0D) — Remote FX Events ---
# Simplified format: we control both server and client, so use fixed-width
# fields rather than the original bitstream format.

# FX event types (subset of decompile's 40 types)
FX_CHAIN_GUN_FIRE = 0
FX_PULSE_FIRE = 1
FX_FLAK_FIRE = 2
FX_MISSILE_FIRE = 3
FX_TURRET_FIRE = 5
FX_IMPACT_GENERIC = 9
FX_IMPACT_VEHICLE = 10
FX_IMPACT_BUILDING = 11
FX_IMPACT_TERRAIN = 12


def build_transient_array(events: list) -> bytes:
    """Build TRANSIENT_ARRAY (0x0D) packet with FX events.

    Each event is a dict with:
        type: int (FX_* constant)
        pos: optional (x, y, z) tuple
        entity_id: optional int (source entity)

    Wire format (simplified):
        u8  opcode (0x0D)
        u8  count
        per event:
            u8  fx_type
            u8  flags (bit 0 = has_pos, bit 1 = has_entity)
            [3×f32 pos]       (if has_pos)
            [u32 entity_id]   (if has_entity)
    """
    if not events:
        return b''

    count = min(len(events), 255)
    buf = bytearray()
    buf.append(0x0D)
    buf.append(count)

    for ev in events[:count]:
        fx_type = ev.get('type', 0)
        pos = ev.get('pos')
        eid = ev.get('entity_id', 0)

        flags = 0
        if pos is not None:
            flags |= 0x01
        if eid:
            flags |= 0x02

        buf.append(fx_type & 0xFF)
        buf.append(flags)

        if pos is not None:
            buf.extend(struct.pack('>3f', pos[0], pos[1], pos[2]))
        if eid:
            buf.extend(struct.pack('>I', eid))

    return bytes(buf)
