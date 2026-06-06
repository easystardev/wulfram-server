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
from wulfram2_protocol.packets import (  # noqa: F401 - re-export for existing importers
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

# Behavior packet feature toggles.
BEHAVIOR_ACTIVE_EXTRAS = os.environ.get("WULFRAM_BEHAVIOR_ACTIVE_EXTRAS", "1") == "1"

# Behavior packet physics defaults.
BEHAVIOR_GROUND_FRICTION = _read_float_env("WULFRAM_BEHAVIOR_GROUND_FRICTION", 0.8)
BEHAVIOR_TURN_RATE = _read_float_env("WULFRAM_BEHAVIOR_TURN_RATE", 0.05)
BEHAVIOR_SUSPENSION_DAMPENING = _read_float_env("WULFRAM_BEHAVIOR_SUSP_DAMPENING", 1.3)
BEHAVIOR_MAX_ALTITUDE = _read_float_env("WULFRAM_BEHAVIOR_MAX_ALTITUDE", 3.25)
BEHAVIOR_GRAVITY_PCT = _read_float_env("WULFRAM_BEHAVIOR_GRAVITY_PCT", 1.0)
BEHAVIOR_SPRING_STATES = (
    os.environ.get(
        "WULFRAM_BEHAVIOR_SPRING_STATES",
        os.environ.get("WULFRAM_BEHAVIOR_THRUSTERS", "1"),
    ) == "1"
)
BEHAVIOR_TANK_SPRING_LONGITUDINAL = _read_float_env("WULFRAM_BEHAVIOR_TANK_SPRING_LONGITUDINAL", 3.4)
BEHAVIOR_TANK_SPRING_LATERAL = _read_float_env("WULFRAM_BEHAVIOR_TANK_SPRING_LATERAL", 2.2)


def get_behavior_tank_spring_local_offsets() -> Tuple[Tuple[float, float], ...]:
    """Return the tank-local XY offsets emitted in BEHAVIOR Section 5."""
    return (
        (BEHAVIOR_TANK_SPRING_LONGITUDINAL, BEHAVIOR_TANK_SPRING_LATERAL),
        (BEHAVIOR_TANK_SPRING_LONGITUDINAL, -BEHAVIOR_TANK_SPRING_LATERAL),
        (-BEHAVIOR_TANK_SPRING_LONGITUDINAL, BEHAVIOR_TANK_SPRING_LATERAL),
        (-BEHAVIOR_TANK_SPRING_LONGITUDINAL, -BEHAVIOR_TANK_SPRING_LATERAL),
    )


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


def build_ping_request(request_id: Optional[int] = None) -> bytes:
    """Build the server-side PING_REQUEST packet (0x0B).

    OG client disassembly:
    - inbound 0x0B handler reads a single u32
    - no second u32 is consumed on the server->client path

    Wire shape for server->client 0x0B:
    [opcode:1][request_id:u32_be]
    """
    if request_id is None:
        request_id = int(time.time() * 1000) & 0xFFFFFFFF
    return b'\x0B' + struct.pack(">I", request_id & 0xFFFFFFFF)


def build_ping_reply(request_id: int) -> bytes:
    """Build the server-side 0x0C ping reply.

    OG client inbound 0x0C handler reads exactly one u32 before finalizing the
    packet, so the server->client reply must be 5 bytes:
    [opcode:1][request_id:u32_be]
    """
    return b'\x0C' + struct.pack(">I", request_id & 0xFFFFFFFF)


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
    """Build TEAM_INFO packet with the empirical OG/wulf-forge string set.

    `azurefishy-src` shows TEAM_INFO as:
      team_id + 5 strings per team

    The exact string semantics are still partly inferred, so prefer the
    captured payload values that are known to keep the original client alive.
    """
    def pack_string(s: str) -> bytes:
        encoded = s.encode('ascii') + b'\x00'
        return struct.pack(">H", len(encoded)) + encoded

    payload = b'\x28'
    # Team 1
    payload += struct.pack("B", 1)
    payload += pack_string("Crimson_Federation")
    payload += pack_string("Crimson Federation")
    payload += pack_string("Crimson Base")
    payload += pack_string("The red team.")
    payload += pack_string("Crimson Federation Wins!")

    # Team 2
    payload += struct.pack("B", 2)
    payload += pack_string("Azure_Alliance")
    payload += pack_string("Azure Alliance")
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

    Wire field order, read by GUESS5_PacketHandler_ADD_TO_ROSTER (Handlers.c:622):
      u32 f1, u32 f2, u16 f3, u16 f4, string name, string clan,
      u16 f5, u16 f6, fixed16 ping, u32 flags.

    GOAL 3 (verified empirically 2026-06-02 by raw-injecting variants and reading
    the OG P-key scoreboard): the handler passes these POSITIONALLY to
    PlayerEntry_create (Social.c:5199), whose stored offsets are what the
    scoreboard render + team-count actually read. The decompiler's positional arg
    order is REAL (not a Ghidra artifact). The resulting field->offset->meaning is:

      f1 (u32) -> +0x0C player_id        (scoreboard record[3])
      f2 (u32) -> +0x10 clan_id          (NOT a team field; unused for display)
      f3 (u16) -> +0x08 team_id          <-- the SCOREBOARD TEAM FILTER (1=red/2=blue)
      f4 (u16) -> +0x14 stats_flags
      name     -> +0x00 name             (record[0])
      clan     -> +0x04 clan_name        (record[1])
      f5 (u16) -> +0x18 kills            (record[6], displayed)
      f6 (u16) -> +0x1C deaths           (record[7], displayed)

    The row renders only when +0x08 == team_info+0x18 (hardcoded 1/2). Previously
    this packed `team` into f2 and `kills` into f3, so every entry's filter-team
    became `kills` (0) -> matched neither column -> the roster stayed empty (and
    the top team-count read 0/0). Putting `team` in f3 makes the player appear on
    the correct team; kills/deaths now go in f5/f6.
    """
    name_bytes = (name + '\x00').encode('ascii')
    clan_bytes = (clan + '\x00').encode('ascii')
    payload = b'\x1A'
    payload += struct.pack(">I", player_id)        # f1 -> +0x0C player_id
    payload += struct.pack(">I", 0)                # f2 -> +0x10 clan_id (unused)
    payload += struct.pack(">H", team & 0xFFFF)    # f3 -> +0x08 team filter (1/2)
    payload += struct.pack(">H", 0)                # f4 -> +0x14 stats_flags
    payload += struct.pack(">H", len(name_bytes)) + name_bytes
    payload += struct.pack(">H", len(clan_bytes)) + clan_bytes
    payload += struct.pack(">H", kills & 0xFFFF)   # f5 -> +0x18 kills (displayed)
    payload += struct.pack(">H", deaths & 0xFFFF)  # f6 -> +0x1C deaths (displayed)
    payload += pack_fixed16(0.0)                   # ping
    payload += struct.pack(">I", 0)                # flags
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


def build_update_stats_team_first(player_id: int, entity_id: int, team_id: int = 0,
                                  deaths: int = 0) -> bytes:
    """Build the empirical OG team-switch UPDATE_STATS variant.

    Archived OG-facing traces place the selected team in the first u16 after
    the two u32 identifiers. Keep the canonical team field populated too so
    Python-client parsing remains intelligible when this probe path is used.
    """
    payload = b'\x1C'
    payload += struct.pack(">I", player_id)
    payload += struct.pack(">I", entity_id)
    payload += struct.pack(">H", team_id & 0xFFFF)  # empirical team/status field
    payload += struct.pack(">H", deaths & 0xFFFF)
    payload += struct.pack(">H", 0)
    payload += struct.pack(">H", 0)
    payload += struct.pack(">H", team_id & 0xFFFF)  # canonical team field
    payload += pack_fixed16(0.0)
    payload += pack_fixed16(0.0)
    payload += struct.pack(">I", 0)
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
    """Build COMM_MESSAGE (0x1F).

    Decompile (Social.c, PacketHandler_COMM_MESSAGE):
      u16 sender_mode, u32 sender_id, u16 target_mode, u32 target_id, string
    """
    payload = b'\x1F'
    payload += struct.pack(">H", 0)           # sender_mode
    payload += struct.pack(">I", source_id)   # sender_id
    payload += struct.pack(">H", 0)           # target_mode
    payload += struct.pack(">I", target_id)   # target_id
    msg_bytes = (message + '\x00').encode('ascii')
    payload += struct.pack(">H", len(msg_bytes)) + msg_bytes
    return payload


def build_ship_status(ship_id: int, team_id: int, name: str = "Supply Ship") -> bytes:
    """Build SHIP_STATUS (0x27).

    Decompile (Network/Handlers.c, PacketHandler_SHIP_STATUS):
      u32 ship_oid, u16 team_id, string ship_name
    """
    name_bytes = (name + '\x00').encode('ascii', errors='ignore')
    payload = b'\x27'
    payload += struct.pack(">I", ship_id & 0xFFFFFFFF)
    payload += struct.pack(">H", team_id & 0xFFFF)
    payload += struct.pack(">H", len(name_bytes)) + name_bytes
    return payload


def build_carrying_info(
    player_id: int,
    *,
    cargo_type: int = 0,
    has_uplink: bool = False,
    cargo_count: int = 0,
) -> bytes:
    """Build CARRYING_INFO (0x29).

    Decompile (Network/Handlers.c, PacketHandler_CARRYING_INFO):
      u32 player_oid, u8 cargo_type, u8 has_uplink, u8 cargo_count
    """
    return (
        b'\x29'
        + struct.pack(">I", player_id & 0xFFFFFFFF)
        + struct.pack("BBB", cargo_type & 0xFF, 1 if has_uplink else 0, cargo_count & 0xFF)
    )


def build_uplink_info(team_index: int, holder_entity_id: int, state: int) -> bytes:
    """Build UPLINK_INFO (0x2A).

    Decompile cross-check:
      PacketHandler_UPLINK_INFO reads u8, u32, u32 and passes them to
      Common_TeamUpdateUplinkState(team, arg2, arg3). Team.c stores arg2 as
      TeamData+0x08 holder entity and arg3 as TeamData+0x04 uplink state.

    Wire body:
      u8 team_index, u32 holder_entity_oid, u32 uplink_state
    """
    return (
        b'\x2A'
        + struct.pack("B", team_index & 0xFF)
        + struct.pack(">I", holder_entity_id & 0xFFFFFFFF)
        + struct.pack(">I", state & 0xFFFFFFFF)
    )


def build_supply_ship_info(
    ship_id: int,
    *,
    shield_pct: int = 100,
    status_template: int = 0,
    cargo_slots: tuple[int, int, int, int] | list[int] = (40, 40, 40, 40),
    cargo_times: tuple[int, int, int, int] | list[int] = (0, 0, 0, 0),
    build_mode: int = 3,
) -> bytes:
    """Build SUPPLY_SHIP_INFO (0x2D).

    Direct disasm of PacketHandler_SUPPLY_SHIP_INFO at 0x0046e590:
      u32 ship_oid
      u32 shield_pct
      u32 status_template_index
      repeat 4: u32 cargo_entity_type, u32 cargo_build_time
      u8 build_mode

    The handler finds the existing SHIP_STATUS marker by OID, copies the
    4 cargo types, 4 timers, and build mode into marker+0x0c, then writes
    shield_pct to marker+0x14 and status_template_index to marker+0x18.
    """
    slots = list(cargo_slots)[:4]
    times = list(cargo_times)[:4]
    while len(slots) < 4:
        slots.append(40)
    while len(times) < 4:
        times.append(0)

    payload = b"\x2D"
    payload += struct.pack(">I", ship_id & 0xFFFFFFFF)
    payload += struct.pack(">I", int(shield_pct) & 0xFFFFFFFF)
    payload += struct.pack(">I", int(status_template) & 0xFFFFFFFF)
    for cargo_type, cargo_time in zip(slots, times):
        payload += struct.pack(">I", int(cargo_type) & 0xFFFFFFFF)
        payload += struct.pack(">I", int(cargo_time) & 0xFFFFFFFF)
    payload += struct.pack("B", int(build_mode) & 0xFF)
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
                                 include_entities: bool = True,
                                 use_local_entity_when_no_transform: bool = False,
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

    if not include_entities:
        bw.write_bits(8, 0)
        return header + bw.get_bytes()

    has_player_entity = rot is not None or pos is not None
    entity_count = 2 if has_player_entity else 1
    bw.write_bits(8, entity_count)

    if use_local_entity_when_no_transform and not has_player_entity:
        bw.write_bits(32, entity_id)
        bw.write_bits(1, 1)
        bw.write_bits(10, 0)
        bw.write_bits(16, 0)
        return header + bw.get_bytes()

    DUMMY_ENTITY_ID = 0xFFFFFFFE
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
    """Build VIEW_UPDATE packet (0x0F) with player position/velocity updates.

    The OG client's `Entity_apply_network_transform` (Replication.c:683-693)
    calls `_exit(0)` on a non-static entity whose interp record is missing
    either a position or rotation flag. Any VIEW_UPDATE that sets position
    must also set rotation for a manned tank. Clamp here defensively.
    """
    if include_pos and not include_rot:
        include_rot = True
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
    """Build VIEW_UPDATE (0x0F) packet with a timestamp + update array payload.

    The OG client's `Entity_apply_network_transform` (Replication.c:683-693)
    calls `_exit(0)` on a non-static entity whose interp record has position
    without rotation. Clamp per-entity here so a mis-configured caller can't
    crash the OG session.
    """
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
        ent = dict(ent)
        if ent.get("include_pos") and not ent.get("include_rot"):
            ent["include_rot"] = True
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
                                    is_static: bool = False,
                                    weapon_id: int = 2,
                                    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                    ammo_count_bits: int = 0,
                                    ammo_count: int = 0,
                                    primary_turret_bits: int = 0,
                                    primary_turret_angle: float = 0.0,
                                    secondary_turret_bits: int = 0,
                                    secondary_turret_angle: float = 0.0,
                                    turret_max: float = 6.3,
                                    turret_range: float = 12.6) -> bytes:
    """Build UPDATE_ARRAY that creates a tank entity with position inline."""
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    _write_local_player_state(bw, include_health, weapon_id=weapon_id, health=health, fuel=fuel,
                              ammo_count_bits=ammo_count_bits, ammo_count=ammo_count,
                              primary_turret_bits=primary_turret_bits, primary_turret_angle=primary_turret_angle,
                              secondary_turret_bits=secondary_turret_bits, secondary_turret_angle=secondary_turret_angle,
                              turret_max=turret_max, turret_range=turret_range)
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
    bw.write_bits(1, 1 if is_static else 0)

    bw.write_bits(4, 15)
    for coord in pos:
        _, quantized = _compress_position(coord)
        bw.write_bits(16, quantized)

    bw.write_bits(4, 15)
    for v in rot:
        _, quantized = _compress_rotation(v)
        bw.write_bits(16, quantized)

    if include_entity_vitals:
        bw.write_bits(10, _encode_health_bits(health, total_bits=10))
        bw.write_bits(10, _encode_health_bits(fuel, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE))

    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_view_update_create_tank(tick: int, entity_id: int, entity_type: int, team: int,
                                  pos: Tuple[float, float, float], behavior_type: int = 0,
                                  include_interp: bool = False, interp_bits: int = 16,
                                  include_health: bool = True,
                                  include_entity_vitals: bool = False,
                                  health: float = 1.0,
                                  fuel: float = 1.0,
                                  is_manned: bool = True,
                                  is_static: bool = False,
                                  weapon_id: int = 2,
                                  rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                  ammo_count_bits: int = 0,
                                  ammo_count: int = 0,
                                  primary_turret_bits: int = 0,
                                  primary_turret_angle: float = 0.0,
                                  secondary_turret_bits: int = 0,
                                  secondary_turret_angle: float = 0.0,
                                  turret_max: float = 6.3,
                                  turret_range: float = 12.6,
                                  timestamp: Optional[int] = None) -> bytes:
    """Build VIEW_UPDATE carrying the same definition-bearing tank shape.

    OG's local correction path appears to gate transform application on the
    entity-definition bit in the same UPDATE_ARRAY payload. Keep this wrapper
    byte-identical to build_update_array_create_tank after the replay timestamp.
    """
    if timestamp is None:
        timestamp = get_ticks()
    update = build_update_array_create_tank(
        tick=tick,
        entity_id=entity_id,
        entity_type=entity_type,
        team=team,
        pos=pos,
        behavior_type=behavior_type,
        include_interp=include_interp,
        interp_bits=interp_bits,
        include_health=include_health,
        include_entity_vitals=include_entity_vitals,
        health=health,
        fuel=fuel,
        is_manned=is_manned,
        is_static=is_static,
        weapon_id=weapon_id,
        rot=rot,
        ammo_count_bits=ammo_count_bits,
        ammo_count=ammo_count,
        primary_turret_bits=primary_turret_bits,
        primary_turret_angle=primary_turret_angle,
        secondary_turret_bits=secondary_turret_bits,
        secondary_turret_angle=secondary_turret_angle,
        turret_max=turret_max,
        turret_range=turret_range,
    )
    return b'\x0F' + struct.pack(">I", timestamp) + update[1:]


def build_update_array_teleport(tick: int, entity_id: int,
                                pos: Tuple[float, float, float],
                                rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                include_health: bool = True,
                                weapon_id: int = 2,
                                health: float = 1.0,
                                fuel: float = 1.0,
                                ammo_count_bits: int = 0,
                                ammo_count: int = 0,
                                primary_turret_bits: int = 0,
                                primary_turret_angle: float = 0.0,
                                secondary_turret_bits: int = 0,
                                secondary_turret_angle: float = 0.0,
                                turret_max: float = 6.3,
                                turret_range: float = 12.6) -> bytes:
    """Build UPDATE_ARRAY that teleports an existing entity to a new position."""
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    _write_local_player_state(bw, include_health, weapon_id=weapon_id,
                              health=health, fuel=fuel,
                              ammo_count_bits=ammo_count_bits, ammo_count=ammo_count,
                              primary_turret_bits=primary_turret_bits, primary_turret_angle=primary_turret_angle,
                              secondary_turret_bits=secondary_turret_bits, secondary_turret_angle=secondary_turret_angle,
                              turret_max=turret_max, turret_range=turret_range)

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
    """Build UPDATE_ARRAY with repair-pad spawn entities.

    Replication.c reads two independent spawn fields for create-bit entities:
    `field_ec` then `field_f0`. For repair pads, `field_f0` is the visible/team
    owner field, while `field_ec` is a separate variant/config value.
    """
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()
    include_rot = os.environ.get("WULFRAM_SPAWN_POINTS_INCLUDE_ROT", "0") == "1"
    update_mask = 0x0B if include_rot else 0x03

    bw.write_bits(1, 0)
    bw.write_bits(8, len(spawn_points))

    for sp in spawn_points:
        oid = sp['oid']
        team = sp['team']
        variant = sp.get('variant', sp.get('config', 1))
        x, y, z = sp.get('x', 100.0), sp.get('y', 10.0), sp.get('z', 100.0)
        rot = sp.get('rot', (0.0, 0.0, 0.0))

        bw.write_bits(32, oid)
        bw.write_bits(1, 0)
        bw.write_bits(10, update_mask)
        bw.write_bits(16, 0)

        bw.write_bits(8, 27)
        bw.write_bits(8, variant & 0xFF)
        bw.write_bits(8, team & 0xFF)
        # Repair-pad creates need the active/transform-applied branch on the OG
        # client; leaving this cleared creates the entity but skips transform.
        bw.write_bits(1, 1)

        bw.write_bits(4, 15)
        for coord in (x, y, z):
            quantized = _compress_wulfforge(coord, max_val=VEC_POS_MAX, range_val=VEC_POS_RANGE, total_bits=16)
            bw.write_bits(16, quantized)

        if include_rot:
            bw.write_bits(4, 15)
            for angle in rot:
                _, quantized = _compress_rotation(angle)
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
    """Build VIEW_UPDATE (0x0F) with repair-pad spawn entities."""
    if timestamp is None:
        timestamp = get_ticks()
    header = struct.pack(">I", timestamp) + struct.pack(">I", tick)
    bw = BitWriter()
    include_rot = os.environ.get("WULFRAM_SPAWN_POINTS_INCLUDE_ROT", "0") == "1"
    update_mask = 0x0B if include_rot else 0x03

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
        variant = sp.get('variant', sp.get('config', 1))
        x, y, z = sp.get('x', 100.0), sp.get('y', 10.0), sp.get('z', 100.0)
        rot = sp.get('rot', (0.0, 0.0, 0.0))

        bw.write_bits(32, oid)
        bw.write_bits(1, 0)
        bw.write_bits(10, update_mask)
        bw.write_bits(16, 0)

        bw.write_bits(8, 27)
        bw.write_bits(8, variant & 0xFF)
        bw.write_bits(8, team & 0xFF)
        bw.write_bits(1, 1)

        bw.write_bits(4, 15)
        for coord in (x, y, z):
            quantized = _compress_wulfforge(coord, max_val=VEC_POS_MAX, range_val=VEC_POS_RANGE, total_bits=16)
            bw.write_bits(16, quantized)

        if include_rot:
            bw.write_bits(4, 15)
            for angle in rot:
                _, quantized = _compress_rotation(angle)
                bw.write_bits(16, quantized)

    return b'\x0F' + header + bw.get_bytes()


def build_behavior_packet() -> bytes:
    """Build BEHAVIOR packet (0x24) with game parameters."""
    payload = bytearray()
    behavior_log = os.environ.get("WULFRAM_BEHAVIOR_LOG", "0") == "1"

    # Section 1: Header (95 bytes)
    spawn_enabled = os.environ.get("WULFRAM_BEHAVIOR_SPAWN_ENABLED", "1").strip().lower()
    payload.append(0x00 if spawn_enabled in ("0", "false", "off", "no") else 0x01)
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
    # Per slot: 5 capability bytes → AmmoSlotDef offsets +0x10 through +0x14
    #   [0] +0x10: gate flag — must be 1 for slot to participate in capability counting
    #   [1] +0x11: ammo_count cap — if set, pool[0] increments → ammo bits in local_player_state
    #   [2] +0x12: reload_state cap — if set, pool[4] increments → reload bits in entity state
    #   [3] +0x13: active_flag cap — if set, pool[8] increments → active-flags bitmask
    #              (controls WeaponCooldown_update_all auto-fire: active=1 → cooldown ticks → fire)
    #   [4] +0x14: cooldown_cap — if set, pool[12] increments → cooldown timestamp bits
    # NOTE: +0x00 (enabled) is always 1 from AmmoSlotDef_init, NOT part of the packet.
    # Keep ammo_count/reload/cooldown caps at 0 to avoid extra local_player_state bits.
    # Slot indices match WeaponType enum: 0=ChainGun, 4=Pulse, 5=Flak, etc.
    TANK_SLOT_CONFIG = {
        0:  [1, 0, 0, 1, 0],   # Chain Gun
        4:  [1, 0, 0, 1, 0],   # Pulse Cannon
        5:  [1, 0, 0, 1, 0],   # Flak
        6:  [1, 0, 0, 1, 0],   # Guided Missile
        7:  [1, 0, 0, 1, 0],   # Hunter Seeker
        8:  [1, 0, 0, 1, 0],   # Mine
        9:  [1, 0, 0, 1, 0],   # Thumper
        10: [1, 0, 0, 1, 0],   # Mortar
        11: [1, 0, 0, 1, 0],   # Piercer
    }
    # Fire rate (ms) per slot → AmmoSlotDef +0x20. Controls cooldown reset in
    # WeaponCooldown_update_all(). Default 1000ms from AmmoSlotDef_init.
    # 0 = fire every frame (BAD).
    TANK_FIRE_RATE_MS = {
        0:  100,    # Chain Gun — rapid fire
        4:  500,    # Pulse Cannon
        5:  200,    # Flak
        6:  1500,   # Guided Missile
        7:  2000,   # Hunter Seeker
        8:  3000,   # Mine
        9:  500,    # Thumper
        10: 1000,   # Mortar
        11: 500,    # Piercer
    }

    for _unit in range(4):
        for _slot in range(13):
            if _unit == 0 and _slot in TANK_SLOT_CONFIG:
                flags = TANK_SLOT_CONFIG[_slot]
                payload += bytes(flags)
            else:
                payload += b'\x00\x00\x00\x00\x00'
            payload += pack_fixed16(1.0)                            # +0x18: accuracy (cos)
            fire_rate = TANK_FIRE_RATE_MS.get(_slot, 1000) if _unit == 0 else 1000
            payload += struct.pack(">I", fire_rate)                 # +0x20: fire_rate_ms
            payload += struct.pack(">I", 0) * 4                    # +0x24..+0x30: params
            payload += pack_fixed16(100.0)                          # +0x38: param_double_0
            payload += pack_fixed16(1000.0)                         # +0x40: min_range (250.0 default)
            payload += pack_fixed16(500.0)                          # +0x48: max_range (450.0 default)
            payload += pack_fixed16(1.0)                            # +0x50: spread (0.08 default)

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
        payload += struct.pack(">I", 33000)  # max_fuel (NOT mass — mass is in collision table: 6700)

    assert len(payload) == 95 + 2340 + 468 + 72, f"After Section 4: expected 2975, got {len(payload)}"

    # Section 5: Spring states
    section5_start = len(payload)

    def _write_spring_state(local_offsets: Tuple[Tuple[float, float], ...]) -> None:
        # Spring_read_from_stream reads u32 count, then position Vec3,
        # normal Vec3, pinned flag per point, followed by one fixed16 scalar.
        # The decompile calls the second Vec3 "velocity" in the stream reader,
        # but Spring_compute_suspension_forces uses it as the point normal. The
        # allocator default is (0, 0, -1); writing zero normals leaves OG with a
        # live softbody state that cannot generate visible vertical spring force.
        payload.extend(struct.pack(">I", len(local_offsets)))
        for x_pos, y_pos in local_offsets:
            payload.extend(pack_fixed16(float(x_pos)))
            payload.extend(pack_fixed16(float(y_pos)))
            payload.extend(pack_fixed16(0.0))
            payload.extend(pack_fixed16(0.0))
            payload.extend(pack_fixed16(0.0))
            payload.extend(pack_fixed16(-1.0))
            payload.extend(struct.pack(">I", 0))
        payload.extend(pack_fixed16(0.0))

    if BEHAVIOR_SPRING_STATES:
        tank_offsets = get_behavior_tank_spring_local_offsets()
        _write_spring_state(tank_offsets)
        _write_spring_state(tank_offsets)
        _write_spring_state(tank_offsets)
        _write_spring_state(tank_offsets)
    else:
        for _ in range(4):
            _write_spring_state(())

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
            f"spring_states={int(BEHAVIOR_SPRING_STATES)} extras={int(BEHAVIOR_ACTIVE_EXTRAS)} "
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
    from wulfram2_protocol.quantizers import QUANTIZER_TABLE

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

    for cfg in sorted(QUANTIZER_TABLE, key=lambda q: q.index):
        _write_entry(
            cfg.fixed_bits,
            cfg.total_bits,
            f"{cfg.max_value}",
            f"{cfg.range_value}",
        )

    return bytes(payload)


# --- TRANSIENT_ARRAY (0x0D) - Remote FX Events ---
# Decompile-backed quantized bitstream format (0x0046CA60).

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

# Legacy DETACHED bit widths (WULFRAM_TRANSIENT_LEGACY=1): the pre-CH4 form that
# hardcoded the widths in packets.py instead of sourcing them from the live
# TRANSLATION/quantizer table. They currently EQUAL the table values, so the A/B
# is byte-neutral today; the gate exists so a TRANSLATION-table width change is
# followed by the live (table-sourced) path, not silently by these constants.
#   type:      quantizer entry 12 (scalar12), bits=16
#   entity_id: quantizer entry 6  (scalar6),  bits=16
#   position:  quantizer entry 16 (pos_bank0), per-component value width=16
_TRANSIENT_TYPE_BITS = 16
_TRANSIENT_ENTITY_BITS = 16
_TRANSIENT_POS_BITS = 16

# Quantizer-table indices for the TRANSIENT_ARRAY fields (decompile-backed).
_QIDX_TRANSIENT_TYPE = 12
_QIDX_TRANSIENT_ENTITY = 6
_QIDX_TRANSIENT_POS = 16


def _transient_legacy_default() -> bool:
    return os.environ.get("WULFRAM_TRANSIENT_LEGACY", "0") == "1"


def _transient_field_params(legacy: bool):
    """Resolve (type_bits, entity_bits, pos_bits, pos_max, pos_range) for TRANSIENT.

    CH4 fidelity debt: the canonical path sources every width/range from the
    shared quantizer table (the same table the server advertises in TRANSLATION),
    so encode/decode follow the table rather than packets.py defaults. The legacy
    gate restores the detached hardcoded widths for A/B.
    """
    if legacy:
        return (_TRANSIENT_TYPE_BITS, _TRANSIENT_ENTITY_BITS, _TRANSIENT_POS_BITS,
                VEC_POS_MAX, VEC_POS_RANGE)
    from wulfram2_protocol.quantizers import get_quantizer
    qt = get_quantizer(_QIDX_TRANSIENT_TYPE)
    qe = get_quantizer(_QIDX_TRANSIENT_ENTITY)
    qp = get_quantizer(_QIDX_TRANSIENT_POS)
    return (qt.fixed_bits, qe.fixed_bits, qp.total_bits, qp.max_value, qp.range_value)


def build_transient_array(events: list, *, legacy: bool = None) -> bytes:
    """Build TRANSIENT_ARRAY (0x0D) packet using decompile-backed quantized bitstream.

    Decompile: GUESS6_PacketHandler_TRANSIENT_ARRAY (0x0046CA60)

    Each event is a dict with:
        type: int (FX_* constant)
        pos: optional (x, y, z) tuple
        entity_id: optional int (source entity)

    Wire format (quantized bitstream after opcode byte):
        8 bits: count
        per event:
            <type_bits>: fx_type     (quantizer index 12)
            1 bit:       has_pos
            if has_pos:  3 × <pos_bits> quantized via the table's pos max/range
            1 bit:       has_entity
            if has_entity: <entity_bits>: entity_id (quantizer index 6)

    Field widths are sourced from the shared quantizer table (CH4); set
    WULFRAM_TRANSIENT_LEGACY=1 (or pass legacy=True) for the detached defaults.
    """
    from wulfram2_protocol.codec import BitWriter, quantize_float

    if not events:
        return b''

    if legacy is None:
        legacy = _transient_legacy_default()
    type_bits, entity_bits, pos_bits, pos_max, pos_range = _transient_field_params(legacy)

    count = min(len(events), 255)
    bw = BitWriter()
    bw.write_bits(8, count)

    for ev in events[:count]:
        fx_type = ev.get('type', 0)
        pos = ev.get('pos')
        eid = ev.get('entity_id', 0)

        bw.write_bits(type_bits, fx_type & ((1 << type_bits) - 1))

        if pos is not None:
            bw.write_bits(1, 1)  # has_pos = 1
            for v in pos:
                raw = quantize_float(float(v), pos_max, pos_range, pos_bits)
                bw.write_bits(pos_bits, raw)
        else:
            bw.write_bits(1, 0)  # has_pos = 0

        if eid:
            bw.write_bits(1, 1)  # has_entity = 1
            bw.write_bits(entity_bits, eid & ((1 << entity_bits) - 1))
        else:
            bw.write_bits(1, 0)  # has_entity = 0

    # Opcode byte + bitstream payload
    return bytes([0x0D]) + bw.get_bytes()


def decode_transient_array(packet: bytes, *, legacy: bool = None) -> list:
    """Decode a TRANSIENT_ARRAY (0x0D) packet, sourcing field widths from the
    same quantizer table the encoder used (CH4 round-trip). Returns the event
    list: [{type, pos|None, entity_id}]. Inverse of build_transient_array."""
    from wulfram2_protocol.codec import BitReader, dequantize_float

    if not packet or packet[0] != 0x0D:
        return []
    if legacy is None:
        legacy = _transient_legacy_default()
    type_bits, entity_bits, pos_bits, pos_max, pos_range = _transient_field_params(legacy)

    br = BitReader(packet[1:])
    count = br.read_bits(8)
    events = []
    for _ in range(count):
        fx_type = br.read_bits(type_bits)
        ev = {"type": fx_type, "pos": None, "entity_id": 0}
        if br.read_bits(1):
            ev["pos"] = tuple(
                dequantize_float(br.read_bits(pos_bits), pos_max, pos_range, pos_bits)
                for _ in range(3)
            )
        if br.read_bits(1):
            ev["entity_id"] = br.read_bits(entity_bits)
        events.append(ev)
    return events
