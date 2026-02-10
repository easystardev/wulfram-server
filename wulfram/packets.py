"""
Packet definitions and builder functions.
Pure functions that return packet payloads - no I/O.
"""

import struct
import time
import math
import os
from typing import Optional, Tuple, List
from .codec import BitWriter, pack_fixed16, frame_packet


# Server tick clock - ticks relative to server start (matches wulf-forge)
_SERVER_START = time.monotonic()

# Behavior packet feature toggles (wulf-forge-inspired).
BEHAVIOR_THRUSTERS = os.environ.get("WULFRAM_BEHAVIOR_THRUSTERS", "1") == "1"
BEHAVIOR_ACTIVE_EXTRAS = os.environ.get("WULFRAM_BEHAVIOR_ACTIVE_EXTRAS", "1") == "1"
_RAW_HEALTH_MODE = os.environ.get("WULFRAM_HEALTH_RAW_MODE", "wulf").strip().lower()
_ALLOW_LINEAR_HEALTH = os.environ.get("WULFRAM_ALLOW_LINEAR_HEALTH", "0") == "1"
if _RAW_HEALTH_MODE in ("linear", "lin") and not _ALLOW_LINEAR_HEALTH:
    # Linear encoding makes 1.0 -> 0x3FF, which the client decodes as *zero* health.
    # Only allow linear when explicitly opted-in.
    print("[WARN] WULFRAM_HEALTH_RAW_MODE=linear ignored; forcing wulf encoding. Set WULFRAM_ALLOW_LINEAR_HEALTH=1 to override.")
    HEALTH_RAW_MODE = "wulf"
elif _RAW_HEALTH_MODE:
    HEALTH_RAW_MODE = _RAW_HEALTH_MODE
else:
    HEALTH_RAW_MODE = "wulf"

def _read_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)

# Health/energy quantizer scaling. Default is normalized 0..1.
HEALTH_MAX = _read_float_env("WULFRAM_HEALTH_MAX", 1.0)
HEALTH_RANGE = _read_float_env("WULFRAM_HEALTH_RANGE", HEALTH_MAX)
ENERGY_MAX = _read_float_env("WULFRAM_ENERGY_MAX", HEALTH_MAX)
ENERGY_RANGE = _read_float_env("WULFRAM_ENERGY_RANGE", ENERGY_MAX)
HEALTH_NORMALIZED = os.environ.get("WULFRAM_HEALTH_NORMALIZED", "1") == "1"
ENTITY_VITALS_MODE = os.environ.get("WULFRAM_ENTITY_VITALS_MODE", "health").strip().lower()
# Behavior packet physics defaults.
# Keep these aligned with wulf-forge packets.toml to avoid client-side
# divergence while we stabilize spawn + controls.
BEHAVIOR_GROUND_FRICTION = _read_float_env("WULFRAM_BEHAVIOR_GROUND_FRICTION", 0.8)
BEHAVIOR_TURN_RATE = _read_float_env("WULFRAM_BEHAVIOR_TURN_RATE", 0.05)
BEHAVIOR_SUSPENSION_DAMPENING = _read_float_env("WULFRAM_BEHAVIOR_SUSP_DAMPENING", 1.3)
BEHAVIOR_MAX_ALTITUDE = _read_float_env("WULFRAM_BEHAVIOR_MAX_ALTITUDE", 3.25)
BEHAVIOR_GRAVITY_PCT = _read_float_env("WULFRAM_BEHAVIOR_GRAVITY_PCT", 1.0)
# Vector quantizer defaults (match wulf-forge unless overridden).
VEC_POS_MAX = _read_float_env("WULFRAM_VEC_POS_MAX", 8192.0)
VEC_POS_RANGE = _read_float_env("WULFRAM_VEC_POS_RANGE", 16384.0)
VEC_VEL_MAX = _read_float_env("WULFRAM_VEC_VEL_MAX", 1000.0)
VEC_VEL_RANGE = _read_float_env("WULFRAM_VEC_VEL_RANGE", 2000.0)
VEC_ROT_MAX = _read_float_env("WULFRAM_VEC_ROT_MAX", 6.3)
VEC_ROT_RANGE = _read_float_env("WULFRAM_VEC_ROT_RANGE", 12.6)
VEC_SPIN_MAX = _read_float_env("WULFRAM_VEC_SPIN_MAX", 200.0)
VEC_SPIN_RANGE = _read_float_env("WULFRAM_VEC_SPIN_RANGE", 400.0)
# Turret angles in UPDATE_ARRAY local-state may use a dynamic quantizer header.
# If the client expects a header (e.g., rot quantizer), set bits/priority here.
LOCAL_STATE_TURRET_HEADER_BITS = int(os.environ.get("WULFRAM_LOCAL_STATE_TURRET_HEADER_BITS", "0"))
LOCAL_STATE_TURRET_PRIORITY = int(os.environ.get("WULFRAM_LOCAL_STATE_TURRET_PRIORITY", "15"))

def get_ticks() -> int:
    """Get current tick count (ms since server start), matching wulf-forge."""
    return int((time.monotonic() - _SERVER_START) * 1000) & 0xFFFFFFFF


# Packet type constants
class PacketType:
    # Core protocol
    HELLO = 0x13
    PLAYER = 0x17
    PLAYER_INFO = 0x18  # Spawns local player vehicle
    TANK = 0x18  # Legacy alias
    ADD_TO_ROSTER = 0x1A
    UPDATE_STATS = 0x1C
    BIRTH_NOTICE = 0x1E
    GAME_CLOCK = 0x2F
    COMM_MESSAGE = 0x1F
    LOGIN_STATUS = 0x22
    LOGIN_REQUEST = 0x21
    BEHAVIOR = 0x24
    REINCARNATE = 0x25
    TEAM_INFO = 0x28
    TRANSLATION = 0x32
    WANT_UPDATES = 0x39
    IDENTIFIED_UDP = 0x4D
    BPS = 0x4E
    UPDATE_ARRAY = 0x0E
    WORLD_STATS = 0x16
    PING_REQUEST = 0x0B
    VIEW_UPDATE = 0x0F


# BEHAVIOR packet layout (used to derive weapon slot capability counts)
BEHAVIOR_HEADER_SIZE = 95
BEHAVIOR_WEAPON_UNITS = 4
BEHAVIOR_WEAPON_SLOTS = 13
BEHAVIOR_WEAPON_SLOT_SIZE = 45


# Packet name lookup for logging
PACKET_NAMES = {
    0x08: "HELLO_ACK",
    0x09: "ACTION_DUMP",
    0x0A: "ACTION_UPDATE",
    0x0B: "PING_REQUEST",
    0x0F: "VIEW_UPDATE",
    0x0E: "UPDATE_ARRAY",
    0x13: "HELLO",
    0x16: "WORLD_STATS",
    0x17: "PLAYER",
    0x18: "TANK",
    0x19: "TANK_RESEND",
    0x1A: "ADD_TO_ROSTER",
    0x1C: "UPDATE_STATS",
    0x1E: "BIRTH_NOTICE",
    0x2F: "GAME_CLOCK",
    0x1F: "COMM_MESSAGE",
    0x20: "CHAT",
    0x21: "LOGIN_REQUEST",
    0x22: "LOGIN_STATUS",
    0x24: "BEHAVIOR",
    0x25: "REINCARNATE",
    0x28: "TEAM_INFO",
    0x32: "TRANSLATION",
    0x33: "TRANSLATION_ACK",
    0x35: "VIEWPOINT_INFO",
    0x39: "WANT_UPDATES",
    0x3a: "BEACON_REQ",
    0x4D: "IDENTIFIED_UDP",
    0x4E: "BPS",
    0x4F: "KUDOS",
    0x54: "VOICE_DATA",
}


def get_packet_name(pkt_type: int) -> str:
    return PACKET_NAMES.get(pkt_type, f"UNKNOWN_{pkt_type:02X}")


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
    """
    Build PLAYER packet (0x17).

    Sets g_local_player_id and g_player_is_spectator in client.
    spectator=True (0x01) means player is in Entry Map/team selection.
    spectator=False (0x00) means player is actively playing.
    """
    return b'\x17' + struct.pack(">I", entity_id) + (b'\x01' if spectator else b'\x00')


def build_team_info() -> bytes:
    """Build TEAM_INFO packet with two teams."""
    def pack_string(s: str) -> bytes:
        encoded = s.encode('ascii') + b'\x00'
        return struct.pack(">H", len(encoded)) + encoded

    payload = b'\x28'
    # Team 1 (Crimson Federation)
    payload += struct.pack("B", 1)  # Team ID (1 byte)
    payload += pack_string("Crimson Federation")
    payload += pack_string("Red Team")
    payload += pack_string("Crimson Base")
    payload += pack_string("The red team.")
    payload += pack_string("Azure Alliance Wins!")

    # Team 2 (Azure Alliance)
    payload += struct.pack("B", 2)  # Team ID (1 byte)
    payload += pack_string("Azure Alliance")
    payload += pack_string("Blue Team")
    payload += pack_string("Crimson Base")
    payload += pack_string("The blue team.")
    payload += pack_string("Crimson Federation Wins!")

    return payload


def build_world_stats(
    map_name: str = "crossroads",  # Wulf-forge default; override via WULFRAM_MAP_NAME
    grid_rows: int = 1,       # Match wulf-forge (flag byte)
    grid_cols: int = 1,       # Match wulf-forge (map_id byte)
    scale: float = 1.0,
) -> bytes:
    """Build WORLD_STATS packet.

    WORLD_STATS payload (from decomp):
    - map name (length-prefixed string, includes null)
    - grid_rows (u8)
    - grid_cols (u8)
    - scale (fixed16.16)
    """
    # Map name is length-prefixed; include trailing null in the length.
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


def build_add_to_roster(player_id: int, entity_id: int, name: str, team: int, clan: str = "") -> bytes:
    """
    Build ADD_TO_ROSTER packet.

    Format aligned to current Wulf-Forge:
    - u32 account_id (player_id)
    - u32 unknown (currently 0)
    - u16 team
    - u16 unknown/deaths
    - string name
    - string nametag/clan
    - u16 kills2
    - u16 deaths2
    - fixed16.16 score (4 bytes)
    - u32 unknown
    """
    name_bytes = (name + '\x00').encode('ascii')
    clan_bytes = (clan + '\x00').encode('ascii')
    payload = b'\x1A'
    payload += struct.pack(">I", player_id)
    payload += struct.pack(">I", 0)             # unknown
    payload += struct.pack(">H", team & 0xFFFF) # team
    payload += struct.pack(">H", 2)             # unknown/deaths
    payload += struct.pack(">H", len(name_bytes)) + name_bytes
    payload += struct.pack(">H", len(clan_bytes)) + clan_bytes
    payload += struct.pack(">H", 2)             # kills2 (u16)
    payload += struct.pack(">H", 2)             # deaths2 (u16)
    payload += pack_fixed16(6.9)                # Score as fixed16.16 (4 bytes)
    payload += struct.pack(">I", 2)             # unknown
    return payload


def build_update_stats(account_id: int, team_id: int) -> bytes:
    """
    Build UPDATE_STATS packet.

    Format from decompiled GUESS3_PacketHandler_UPDATE_STATS:
    - u32 account_id
    - u32 stat_type
    - u16 team_id
    - u16 flags
    - u16 rank
    - u16 kills
    - u16 deaths
    - fixed16 score1 (4 bytes - read by Stream_read_fixed16_double)
    - fixed16 score2 (4 bytes - read by Stream_read_fixed16_double)
    - u32 something
    """
    payload = b'\x1C'
    payload += struct.pack(">I", account_id)
    payload += struct.pack(">I", 6)       # Stat type
    payload += struct.pack(">H", team_id)
    payload += struct.pack(">H", 0x21)    # Flags
    payload += struct.pack(">H", 3)       # Rank
    payload += struct.pack(">H", 5)       # Kills
    payload += struct.pack(">H", 9)       # Deaths
    payload += pack_fixed16(1.0)          # Score1 - wulf-forge uses 1.0
    payload += pack_fixed16(1.0)          # Score2 - wulf-forge uses 1.0
    payload += struct.pack(">I", 10)      # Unknown u32 - wulf-forge uses 10 (0x0A)
    return payload


def build_birth_notice(entity_id: int, owner_entity_id: Optional[int] = None) -> bytes:
    """Build BIRTH_NOTICE packet."""
    if owner_entity_id is None:
        owner_entity_id = entity_id
    return b'\x1E' + struct.pack(">I", entity_id) + struct.pack(">I", owner_entity_id)


def build_game_clock(time_ms: int = 0, running: bool = True, round_time_ms: int = 30000) -> bytes:
    """Build GAME_CLOCK packet (0x2F).

    Wulf-forge format:
    - int32 ticks (current time in ms)
    - byte is_active (0x01 = running)
    - int32 phase (0 = push, 1 = glimpse)
    - int32 duration_ms (length of push/glimpse in ms)
    """
    ticks = int(time.time() * 1000) & 0xFFFFFFFF
    payload = b'\x2F'
    payload += struct.pack(">I", ticks)
    payload += b'\x01' if running else b'\x00'
    payload += struct.pack(">I", 1)  # Phase = 1 (glimpse)
    payload += struct.pack(">I", round_time_ms)  # Duration
    return payload


def build_motd(message: str = "Welcome to Wulfram!") -> bytes:
    """Build MOTD packet (0x23).

    Format: [0x23] [length-prefixed string]
    """
    msg_bytes = (message + '\x00').encode('ascii')
    return b'\x23' + struct.pack(">H", len(msg_bytes)) + msg_bytes


def build_chat_message(message: str, source_id: int = 0, target_id: int = 0) -> bytes:
    """
    Build COMM_MESSAGE (chat) packet.
    Structure: [1F] [H target_type] [I target_id] [H source_type] [I source_id] [String]
    """
    payload = b'\x1F'
    payload += struct.pack(">H", 0)  # target_type (0 = global/system)
    payload += struct.pack(">I", target_id)
    payload += struct.pack(">H", 0)  # source_type
    payload += struct.pack(">I", source_id)
    # String: [H length] [bytes with null]
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
                      ammo_count_bits: int = 0,
                      ammo_count: int = 0,
                      primary_turret_bits: int = 0,
                      primary_turret_angle: float = 0.0,
                      secondary_turret_bits: int = 0,
                      secondary_turret_angle: float = 0.0,
                      turret_max: float = 6.3,
                      turret_range: float = 12.6) -> bytes:
    """
    Build PLAYER_INFO packet (0x18) for spawning the local player's vehicle.

    Format from decomp GUESS3_PacketHandler_PLAYER_INFO:
    - u32 entity_oid (32 bits)
    - 1 bit flag (1 = include local player state)
    - u32 vehicle_type (32 bits, 0-4 valid)
    - u32 frame_id (32 bits)
    - u8 properties (8 bits)
    - 3x fixed16 position (96 bits)
    - 3x fixed16 rotation (96 bits)

    Client uses bit-based reading (read_u32 = read_bits(32)), so we use BitWriter.
    """
    bw = BitWriter()

    # Entity OID (32 bits)
    bw.write_bits(32, entity_oid)

    # Local player state flag and optional state block
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

    # Vehicle type (32 bits) - 0-4 are valid tank types
    bw.write_bits(32, vehicle_type)

    # Frame ID (32 bits)
    bw.write_bits(32, entity_oid)  # Use OID as frame ID

    # Properties / config byte (8 bits)
    bw.write_bits(8, properties & 0xFF)

    # Position (3x fixed16.16 = 3x 32 bits)
    x, y, z = pos
    bw.write_bits(32, int(x * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(y * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(z * 65536.0) & 0xFFFFFFFF)

    # Rotation (3x fixed16.16 = 3x 32 bits)
    rx, ry, rz = rot
    bw.write_bits(32, int(rx * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(ry * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(rz * 65536.0) & 0xFFFFFFFF)

    # Opcode is not part of the bitstream; prepend it separately.
    return b'\x18' + bw.get_bytes()


def build_udp_tank_packet_wf(
    net_id: int,
    unit_type: int,
    team_id: int,
    pos: Tuple[float, float, float],
    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    tick: Optional[int] = None,
    include_vitals: bool = False,  # Try False: skip local_state quantizer reads entirely
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
    """
    Build a UDP TANK packet (0x18) using the Wulf-Forge bit layout.

    Layout (bit-packed, not byte-aligned after vitals):
    - u32 tick
    - 1 bit: include vitals
    - [vitals bits...]
    - u32 unit_type
    - u32 net_id
    - u8 team_id
    - 3x fixed16 position
    - 3x fixed16 rotation
    """
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
    # Use new PLAYER_INFO format
    return build_player_info(oid, entity_type, pos, rot=rot)


def build_tank_packet(net_id: int, unit_type: int, pos: Tuple[float, float, float],
                      rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                      flags: int = 1, include_vitals: bool = True,
                      health: float = 1.0, energy: float = 1.0) -> bytes:
    """
    Build TankPacket (0x18) matching Wulf-Forge's format exactly.

    Format (from Wulf-Forge network/packets/tank.py):
    - int32 ticks (frame time)
    - 1 bit vitals flag
    - if vitals: weapon_id (5 bits), health (10 bits), energy (10 bits)
    - int32 unit_type
    - int32 net_id
    - byte flags
    - 3x fixed16 position
    - 3x fixed16 rotation
    """
    import time
    ticks = int(time.monotonic() * 1000) & 0xFFFFFFFF

    bw = BitWriter()

    # Frame tick (32 bits) - NOTE: Wulf-Forge uses byte-aligned int32
    bw.write_bits(32, ticks)

    # Vitals flag (1 bit)
    bw.write_bits(1, 1 if include_vitals else 0)

    if include_vitals:
        # Weapon ID (5 bits)
        bw.write_bits(5, 0)
        # Health/Energy (10 bits) - encoding depends on HEALTH_RAW_MODE
        bw.write_bits(10, _encode_health_bits(health, total_bits=10))
        bw.write_bits(10, _encode_health_bits(energy, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE))

    # Unit type (32 bits)
    bw.write_bits(32, unit_type)

    # Net ID (32 bits)
    bw.write_bits(32, net_id)

    # Flags (8 bits)
    bw.write_bits(8, flags)

    # Position (3x fixed16.16)
    x, y, z = pos
    bw.write_bits(32, int(x * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(y * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(z * 65536.0) & 0xFFFFFFFF)

    # Rotation (3x fixed16.16)
    rx, ry, rz = rot
    bw.write_bits(32, int(rx * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(ry * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(rz * 65536.0) & 0xFFFFFFFF)

    return b'\x18' + bw.get_bytes()


def build_update_array_empty(tick: int = 0) -> bytes:
    """Build empty UPDATE_ARRAY packet."""
    # Tick is read as byte-aligned u32 by client, NOT from bitstream
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()
    bw.write_bits(1, 0)      # Local player state flag = 0
    bw.write_bits(8, 0)      # Zero entries
    return b'\x0E' + tick_bytes + bw.get_bytes()


def _write_local_player_state(bw: BitWriter, include: bool,
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
                              include_ammo_turrets: bool = True) -> None:
    """
    Write local player state block used by UPDATE_ARRAY and PLAYER_INFO.

    Format (from decomp):
    - 1 bit flag
    - weapon type (quantizer index 1 => 5 bits in our TRANSLATION). This is the
      weapon definition index (entity type), not the current weapon slot.
    - health (quantizer index 5 => 10 bits)
    - fuel/energy (quantizer index 8 => 10 bits)
    - ammo bitmask bits (size comes from ammo slot state pool; 0 for our current config)
    - optional primary/secondary turret angles (bit width depends on weapon def flags)

    AzureFishy decomp notes:
    - WeaponDef_init_by_entity_type sets +0x170 for Tank (entity type 0) => primary turret angle REQUIRED.
    - Scout (entity type 1) sets +0x68 => secondary turret angle REQUIRED.
    """
    if not include:
        bw.write_bits(1, 0)
        return

    bw.write_bits(1, 1)
    bw.write_bits(5, weapon_id & 0x1F)
    bw.write_bits(10, _encode_health_bits(health, total_bits=10))
    bw.write_bits(10, _encode_health_bits(fuel, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE))
    # Ammo slot bitmask: only write if behavior-derived bit count is nonzero.
    # The client uses ammo_slot_state_pool[weapon_type].active_bits to decide
    # how many bits to read. Writing bits when the count is zero desyncs.
    if include_ammo_turrets and ammo_count_bits > 0:
        bw.write_bits(ammo_count_bits, ammo_count & ((1 << ammo_count_bits) - 1))

    # Turret angles (if weapon def flags indicate they exist)
    if include_ammo_turrets and primary_turret_bits:
        if LOCAL_STATE_TURRET_HEADER_BITS > 0:
            bw.write_bits(LOCAL_STATE_TURRET_HEADER_BITS, LOCAL_STATE_TURRET_PRIORITY & ((1 << LOCAL_STATE_TURRET_HEADER_BITS) - 1))
        bw.write_bits(primary_turret_bits, _compress_value(primary_turret_angle, turret_max, turret_range, total_bits=primary_turret_bits))
    if include_ammo_turrets and secondary_turret_bits:
        if LOCAL_STATE_TURRET_HEADER_BITS > 0:
            bw.write_bits(LOCAL_STATE_TURRET_HEADER_BITS, LOCAL_STATE_TURRET_PRIORITY & ((1 << LOCAL_STATE_TURRET_HEADER_BITS) - 1))
        bw.write_bits(secondary_turret_bits, _compress_value(secondary_turret_angle, turret_max, turret_range, total_bits=secondary_turret_bits))


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
                                 rot: tuple = None) -> bytes:
    """Build UPDATE_ARRAY/VIEW_UPDATE with entity heartbeat and optional health data.

    When include_health=True, sends local player state with full health/energy.
    When is_view_update=True, uses 0x0F (VIEW_UPDATE) format with timestamp+tick
    header, matching wulf-forge's primary health delivery mechanism.

    - Entity: net_id(32) + is_manned(1) + mask(10) + bank_selector(16)
    """
    if is_view_update:
        # VIEW_UPDATE (0x0F): timestamp(4) + tick(4) + bitstream
        # wulf-forge uses this format for health updates in its main loop
        timestamp = get_ticks()
        header = b'\x0F' + struct.pack(">I", timestamp) + struct.pack(">I", tick)
    else:
        # UPDATE_ARRAY (0x0E): tick(4) + bitstream
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

    # entity_count = 1 is critical: entity_count > 0 causes
    # Network_record_update_stats(packet_tick) to be called, whose return
    # value flows to sync_local_player's EAX (tick guard parameter).
    # With entity_count=0, EAX = return of Network_update_latency_stats()
    # (a small value), which fails the tick guard when g_last_input_apply_tick
    # was set by client input processing to the current game tick (~27000+).
    # With entity_count=1, EAX = packet tick → guard passes.
    #
    # IMPORTANT: Use a dummy entity_id (not the player's) so OIDTable_lookup
    # returns NULL and entity processing goes to g_null_entity_stub.
    # This avoids two problems with using the real entity_id:
    #   - mask=0 (no rotation): Replication.c:1163 zeros entity rotation/velocity
    #   - mask with rotation: overrides client rotation to server heading (diverges)
    # With a dummy ID, the null stub absorbs all writes harmlessly while the
    # tick guard still passes (entity_count > 0).
    DUMMY_ENTITY_ID = 0xFFFFFFFE
    bw.write_bits(8, 1)              # 1 entity
    bw.write_bits(32, DUMMY_ENTITY_ID)  # dummy net_id (not in OIDTable)
    bw.write_bits(1, 1)              # is_manned = True
    bw.write_bits(10, 0)             # mask=0, no data fields
    bw.write_bits(16, 0)             # bank_selector = 0
    return header + bw.get_bytes()


def _compress_value(val: float, max_val: float, range_val: float, total_bits: int = 16) -> int:
    """Compress a value using the same inverse quantization as wulf-forge."""
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


def _encode_health_bits(value: float, total_bits: int = 10, *, max_val: Optional[float] = None,
                        range_val: Optional[float] = None) -> int:
    """
    Encode health/energy for local-state + TankPacket vitals.

    Modes:
    - "linear": value 0..1 mapped to [0, 2^bits-1]
    - "wulf": wulf-forge quantizer (raw=1 -> max)
    """
    if value is None:
        value = 0.0
    max_val = HEALTH_MAX if max_val is None else max_val
    range_val = HEALTH_RANGE if range_val is None else range_val
    value = float(value)
    if HEALTH_NORMALIZED and max_val > 0:
        value = value * max_val
    if max_val > 0:
        value = max(0.0, min(max_val, value))
    else:
        value = max(0.0, value)
    if HEALTH_RAW_MODE in ("wulf", "wulfforge"):
        return _compress_value(value, max_val, range_val, total_bits=total_bits)
    denom = (1 << total_bits) - 1
    if denom <= 0:
        return 0
    if max_val <= 0:
        return 0
    scaled = value / max_val
    return int(round(scaled * denom)) & denom


def build_update_array_player_update(tick: int, entity_id: int,
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
                                     turret_range: float = 12.6) -> bytes:
    """
    Build UPDATE_ARRAY packet with player position/velocity updates.

    Uses wulf-forge bit layout:
    - Optional local stats (health/energy)
    - 1 entity with POS + VEL + ROT updates
    """
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

    bw.write_bits(8, 1)           # 1 entity
    bw.write_bits(32, entity_id)  # OID
    bw.write_bits(1, 1 if is_manned else 0)  # is_manned (player vehicle)

    # Update mask: POS | VEL | ROT (+ SPEED_SCALE/FUEL)
    update_mask = 0
    if include_pos:
        update_mask |= (1 << 1)
    if include_vel:
        update_mask |= (1 << 2)
    if include_rot:
        update_mask |= (1 << 3)
    if include_entity_vitals:
        update_mask |= (1 << 5)  # speed scale
        update_mask |= (1 << 7)  # fuel
    bw.write_bits(10, update_mask)

    # Bank selector
    bw.write_bits(16, 0)

    if include_pos:
        # Position vector (bank 0, 16-bit precision) - must match TRANSLATION VEC_POS config.
        bw.write_bits(4, 15)
        for v in pos:
            bw.write_bits(16, _compress_value(v, VEC_POS_MAX, VEC_POS_RANGE, total_bits=16))

    if include_vel:
        # Velocity vector (bank 0, 16-bit precision) - must match TRANSLATION VEC_VEL config.
        bw.write_bits(4, 15)
        for v in vel:
            bw.write_bits(16, _compress_value(v, VEC_VEL_MAX, VEC_VEL_RANGE, total_bits=16))

    if include_rot:
        # Rotation vector (bank 0, 16-bit precision) - wulf-forge VEC_ROT config
        # max=6.3, range=12.6 (covers ±2π radians)
        bw.write_bits(4, 15)
        for v in rot:
            bw.write_bits(16, _compress_value(v, 6.3, 12.6, total_bits=16))

    if include_entity_vitals:
        if ENTITY_VITALS_MODE in ("health", "vitals"):
            bw.write_bits(10, _encode_health_bits(speed_scale, total_bits=10))
            bw.write_bits(10, _encode_health_bits(fuel, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE))
        else:
            bw.write_bits(10, _compress_value(speed_scale, 1.0, 1.0, total_bits=10))  # speed scale
            bw.write_bits(10, _compress_value(fuel, 1.0, 1.0, total_bits=10))         # fuel fraction

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
    """
    Build VIEW_UPDATE packet (0x0F) with player position/velocity updates.

    Mirrors build_update_array_player_update but includes the VIEW_UPDATE
    header (timestamp + tick) used by the client for interpolation.
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
    )

    bw.write_bits(8, 1)           # 1 entity
    bw.write_bits(32, entity_id)  # OID
    bw.write_bits(1, 1 if is_manned else 0)  # is_manned (player vehicle)

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


def _write_update_array_entity(bw: "BitWriter",
                               *,
                               entity_id: int,
                               is_manned: bool,
                               pos: Tuple[float, float, float],
                               vel: Tuple[float, float, float],
                               rot: Tuple[float, float, float],
                               include_pos: bool,
                               include_vel: bool,
                               include_rot: bool,
                               include_entity_vitals: bool = False,
                               speed_scale: float = 1.0,
                               fuel: float = 1.0) -> None:
    """Write a single entity update block to an UPDATE_ARRAY bitstream."""
    bw.write_bits(32, entity_id)  # OID
    bw.write_bits(1, 1 if is_manned else 0)

    update_mask = 0
    if include_pos:
        update_mask |= (1 << 1)
    if include_vel:
        update_mask |= (1 << 2)
    if include_rot:
        update_mask |= (1 << 3)
    if include_entity_vitals:
        update_mask |= (1 << 5)  # speed scale
        update_mask |= (1 << 7)  # fuel
    bw.write_bits(10, update_mask)

    bw.write_bits(16, 0)  # Bank selector

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
            bw.write_bits(16, _compress_value(v, VEC_ROT_MAX, VEC_ROT_RANGE, total_bits=16))

    if include_entity_vitals:
        if ENTITY_VITALS_MODE in ("health", "vitals"):
            bw.write_bits(10, _encode_health_bits(speed_scale, total_bits=10))
            bw.write_bits(10, _encode_health_bits(fuel, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE))
        else:
            bw.write_bits(10, _compress_value(speed_scale, 1.0, 1.0, total_bits=10))
            bw.write_bits(10, _compress_value(fuel, 1.0, 1.0, total_bits=10))


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


def _compress_position(value: float, max_val: float = VEC_POS_MAX, range_val: float = VEC_POS_RANGE,
                       total_bits: int = 16) -> Tuple[int, int]:
    """
    Compress a position value using Wulf-Forge's quantization scheme.
    Returns (header, quantized_value) where header is always 15 (max precision for 4-bit header).

    Matches Wulf-Forge VEC_POS: max=8192, range=16384, total_bits=16.
    """
    return (15, _compress_wulfforge(value, max_val=max_val, range_val=range_val, total_bits=total_bits))


def _compress_wulfforge(value: float, max_val: float, range_val: float, total_bits: int) -> int:
    """
    Compress a float using Wulf-Forge's exact quantization formula.
    Matches the C++ 'Unpack_Float_From_Int' inverse.

    Formula: raw_val = ((max_val - value) * denom / range) + 1
    Where denom = (1 << total_bits) - 2

    Returns the raw quantized integer.
    """
    # Special case: zero
    if value == 0.0:
        return 0

    min_val = max_val - range_val

    # Clamp
    if value > max_val:
        value = max_val
    if value < min_val:
        value = min_val

    # Calculate denominator (max steps)
    denom = (1 << total_bits) - 2
    if denom <= 0:
        denom = 1

    if range_val == 0:
        return 1

    # Wulf-forge formula: work from max_value DOWN
    delta = max_val - value
    scaled = (delta * denom) / range_val
    raw_val = int(scaled) + 1

    return raw_val


def _compress_rotation(value: float, max_val: float = VEC_ROT_MAX, range_val: float = VEC_ROT_RANGE,
                       total_bits: int = 16) -> Tuple[int, int]:
    """
    Compress a rotation value using Wulf-Forge's quantization scheme.
    Matches VEC_ROT: max=6.3, range=12.6, total_bits=16.
    """
    return (15, _compress_wulfforge(value, max_val=max_val, range_val=range_val, total_bits=total_bits))


def build_update_array_create_tank(tick: int, entity_id: int, entity_type: int, team: int,
                                    pos: Tuple[float, float, float], behavior_type: int = 0,
                                    include_interp: bool = False, interp_bits: int = 16,
                                    include_health: bool = True,
                                    include_entity_vitals: bool = False,
                                    health: float = 1.0,
                                    fuel: float = 1.0,
                                    is_manned: bool = True,
                                    weapon_id: int = 2) -> bytes:
    """
    Build UPDATE_ARRAY that creates a tank entity with position inline.

    Matches Wulf-Forge's format exactly (from update_array.py):
    - Entity header: net_id(32) + is_manned(1) + mask(10) + bank_selector(16)
    - Creation: unit_type(8) + team(8) + team_again(8) + teleport(1)
    - Position/rotation vectors with 4-bit header + data bits

    Presence flags (10 bits, LSB first):
    - Bit 0 (0x001): Entity creation data (DEFINITION)
    - Bit 1 (0x002): Position vectors (POS)
    - Bit 2 (0x004): Velocity vectors (VEL)
    - Bit 3 (0x008): Rotation vectors (ROT)
    """
    # Tick is read as byte-aligned u32 by client, NOT from bitstream
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    # Header (bitstream starts here)
    _write_local_player_state(bw, include_health, weapon_id=weapon_id, health=health, fuel=fuel, include_ammo_turrets=False)
    bw.write_bits(8, 1)            # 1 entity

    # Entity header (matches wulf-forge update_array.py EntitySerializer.serialize)
    bw.write_bits(32, entity_id)   # net_id (OID)
    bw.write_bits(1, 1 if is_manned else 0)  # is_manned

    # Presence flags: bit 0 (creation) + bit 1 (position) + bit 3 (rotation)
    # Binary: 0b0000001011 = 11 decimal
    presence_flags = 0b0000001011
    if include_entity_vitals:
        # Wulf-forge marks HEALTH (bit 5); include ENERGY (bit 7) as a stable pair.
        presence_flags |= (1 << 5)
        presence_flags |= (1 << 7)
    bw.write_bits(10, presence_flags)

    # CRITICAL: Bank selector (16 bits) - wulf-forge puts this AFTER presence flags!
    # We were MISSING this! Selects quantizer bank (0 = bank 0 = indices 16-19)
    bw.write_bits(16, 0)           # Bank selector = 0

    # --- CREATION BLOCK (Bit 0 / DEFINITION) ---
    # Wulf-forge uses 8-bit fields for unit type and team IDs (quantizer[2] and [3]).
    # The second team byte is usually the same as team_id; keep behavior_type only when non-zero.
    bw.write_bits(8, entity_type & 0xFF)   # unit_type (8 bits, quantizer[2])
    config_val = behavior_type if behavior_type else team
    bw.write_bits(8, config_val & 0xFF)    # team_id / entity_config
    bw.write_bits(8, team & 0xFF)          # team_id (quantizer[3])
    bw.write_bits(1, 1)                    # is_static = True (force position)

    # --- POSITION VECTORS (Bit 1 / POS) ---
    # Each component: 4-bit header + 16 bits of data at max priority.
    bw.write_bits(4, 15)           # Position header (priority = 15)
    for coord in pos:
        _, quantized = _compress_position(coord)
        bw.write_bits(16, quantized)

    # --- ROTATION VECTORS (Bit 3 / ROT) ---
    # Header ONCE, then 3 values (16 bits each)
    bw.write_bits(4, 15)           # Rotation header (priority = 15)
    for _ in range(3):
        _, quantized = _compress_rotation(0.0)
        bw.write_bits(16, quantized)

    # --- ENTITY VITALS (Bits 5 & 7) ---
    if include_entity_vitals:
        bw.write_bits(10, _encode_health_bits(health, total_bits=10))
        bw.write_bits(10, _encode_health_bits(fuel, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE))

    return b'\x0E' + tick_bytes + bw.get_bytes()


def build_update_array_spawn_points(tick: int, spawn_points: list) -> bytes:
    """
    Build UPDATE_ARRAY with spawn point entities (type 27 = Repair Pad).

    spawn_points: list of dicts with keys:
        - oid: unique entity ID
        - team: team ID (1 or 2)
        - x, y, z: position coordinates

    UPDATE_ARRAY format for entity creation (matching wulf-forge):
    - u32 tick (byte-aligned, NOT in bitstream)
    - 1 bit local player state flag (0 = skip)
    - u8 entity count
    - For each entity:
        - u32 OID (net_id)
        - 1 bit is_manned
        - 10 bits presence flags (bit 0 = DEFINITION, bit 1 = POS)
        - 16 bits bank selector
        - If bit 0 (DEFINITION):
            - unit_type (8 bits) - 27 for Repair Pad
            - team_id (8 bits)
            - team_id_also (8 bits)
            - is_teleport (1 bit)
        - If bit 1 (POS):
            - 4-bit header (priority)
            - 3x position values (variable bits based on priority)
    """
    # Tick is read as byte-aligned u32 by client, NOT from bitstream
    tick_bytes = struct.pack(">I", tick)
    bw = BitWriter()

    # Header (bitstream starts here)
    bw.write_bits(1, 0)                        # Local player state flag = 0 (skip)
    bw.write_bits(8, len(spawn_points))        # Entity count

    for sp in spawn_points:
        oid = sp['oid']
        team = sp['team']
        config = sp.get('config', team)
        x, y, z = sp.get('x', 100.0), sp.get('y', 10.0), sp.get('z', 100.0)

        # Entity header (matches wulf-forge format)
        bw.write_bits(32, oid)                 # net_id (OID)
        bw.write_bits(1, 0)                    # is_manned = False (static building)

        # Presence flags: bit 0 = DEFINITION, bit 1 = POS
        # Binary: 0b0000000011 = 3
        bw.write_bits(10, 0x03)                # DEFINITION | POS

        # Bank selector (16 bits) - selects quantizer bank 0
        bw.write_bits(16, 0)

        # --- DEFINITION block (bit 0) ---
        # Unit type 27 = Repair Pad (spawn point)
        bw.write_bits(8, 27)                   # unit_type = 27 (Repair Pad)
        bw.write_bits(8, config & 0xFF)        # entity_config / variant
        bw.write_bits(8, team & 0xFF)          # team_id
        bw.write_bits(1, 1)                    # is_static = True (repair pad)

        # --- POSITION block (bit 1) ---
        # Wulf-forge VEC_POS config: head=4, total=16, max=8192, range=16384
        # precision_base_bits = 16 - (1 << 4) + 1 = 1
        # With priority 15: total_bits = 1 + 15 = 16 bits per component
        #
        # Write 4-bit header ONCE with max priority (15), then 3x 16-bit values
        bw.write_bits(4, 15)                   # Position header (priority = 15)

        # Position values using wulf-forge exact formula
        # Range: -8192 to +8192 (max=8192, range=16384)
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
        bw.write_bits(10, 0x03)  # DEFINITION | POS
        bw.write_bits(16, 0)     # Bank selector

        bw.write_bits(8, 27)          # unit_type = Repair Pad
        bw.write_bits(8, config & 0xFF)
        bw.write_bits(8, team & 0xFF)
        bw.write_bits(1, 1)           # spawn snap

        bw.write_bits(4, 15)
        for coord in (x, y, z):
            quantized = _compress_wulfforge(coord, max_val=VEC_POS_MAX, range_val=VEC_POS_RANGE, total_bits=16)
            bw.write_bits(16, quantized)

    return b'\x0F' + header + bw.get_bytes()


def build_behavior_packet() -> bytes:
    """
    Build BEHAVIOR packet (0x24) with game parameters.

    Structure matching wulf-forge exactly:
    - Section 1: Header (95 bytes)
    - Section 2: Weapons (4 units × 13 slots × 45 bytes = 2340 bytes)
    - Section 3: Unit stats (39 units × 12 bytes = 468 bytes)
    - Section 4: Vehicle Physics (2 vehicles × 36 bytes = 72 bytes)
    - Section 5: Hardpoints (4 blocks with counts + trailing fixed16)
    - Section 6: Active Vehicle Physics (3 vehicles × 28 bytes = 84 bytes)
    - Padding to target_size (3116 bytes)
    """
    payload = bytearray()
    behavior_log = os.environ.get("WULFRAM_BEHAVIOR_LOG", "0") == "1"

    # ========== SECTION 1: Header (95 bytes) ==========
    # Matches wulf-forge BehaviorHeader defaults
    payload.append(0x00)                  # spawn_related (wulf-forge default: 0)
    payload += pack_fixed16(5.0)          # timeout
    payload += pack_fixed16(10.0)         # dbl_6792F8 (wulf-forge default)
    payload += pack_fixed16(10.0)         # velocity_q (wulf-forge default)
    payload += pack_fixed16(10.0)         # dbl_679308 (wulf-forge default)
    payload += pack_fixed16(10.0)         # dbl_679310 (wulf-forge default)

    payload += struct.pack(">I", 20)      # total_team_size
    payload += struct.pack(">I", 25000)   # glimpse_ms
    payload += struct.pack(">I", 35000)   # push_ms

    payload += pack_fixed16(100.0)        # dbl_5738B8 - GRAVITY constant (100.0 for faster fall)
    payload += struct.pack(">I", 1)       # dword_6791B8
    payload += struct.pack(">I", 1)       # dword_6791BC
    payload += pack_fixed16(1.0)          # max_pulse_charge

    # 11x fixed16 values (unk11 - all 1.0 in wulf-forge)
    for _ in range(11):
        payload += pack_fixed16(1.0)

    payload.append(0x01)  # flag1
    payload.append(0x01)  # flag2

    assert len(payload) == 95, f"Section 1 should be 95 bytes, got {len(payload)}"

    # ========== SECTION 2: Weapons (2340 bytes) ==========
    # 4 units × 13 slots × 45 bytes each
    #
    # Tank weapon config from decompiled GUESS4_WeaponDef_init_by_entity_type:
    # Pre-switch common init sets capability flags, then Tank case enables slots.
    #
    # Bool bytes: [enabled, ammo_capable, fire_capable, active_capable, cooldown_capable]
    #
    # IMPORTANT: Setting capability flags to 1 makes the client expect extra bits in
    # UPDATE_ARRAY for ammo/fire/active/cooldown state. Since we don't send those bits,
    # enabling capability flags causes bitstream desync and crashes.
    #
    # Safe config: enabled=1, all other flags=0 for all slots
    # This allows weapon cycling without requiring ammo data in packets.
    # WEAPON SLOT BUG: Only slot 0 works. Enabling ANY other slot (alone or combined
    # with slot 0) causes client crash in entity destructor (GUESS3_List_free_nodes_only)
    # at entity+0xE8 with corrupted pointer 0xFFFFFFFE.
    #
    # This appears to be a client-side initialization bug where non-slot-0 weapons
    # don't properly initialize entity data structures. The chain gun (slot 0) works
    # because it has special handling in the client code.
    #
    # Attempts that didn't fix the crash:
    # - Setting all capability flags to 0 (only enabled=1)
    # - Zeroing all data for disabled slots
    # - Enabling only slot 4 (without slot 0)
    # - Enabling slots 0 and 1 (consecutive)
    enable_slot0 = os.environ.get("WULFRAM_BEHAVIOR_SLOT0", "0") == "1"
    TANK_SLOT_CONFIG = {
        # slot: [enabled, ammo_capable, fire_capable, active_capable, cooldown_capable]
        0:  [1, 0, 0, 0, 0],  # Slot 0: Chain gun (optional)
    } if enable_slot0 else {}

    for _unit in range(4):
        for _slot in range(13):
            if _unit == 0 and _slot in TANK_SLOT_CONFIG:
                flags = TANK_SLOT_CONFIG[_slot]
                payload += bytes(flags)
            else:
                payload += b'\x00\x00\x00\x00\x00'
            # Weapon parameters (same for all slots)
            payload += pack_fixed16(1.0)  # targeting cone
            payload += struct.pack(">I", 0) * 5  # 5 ints
            payload += pack_fixed16(100.0)
            payload += pack_fixed16(1000.0)
            payload += pack_fixed16(500.0)
            payload += pack_fixed16(1.0)

    assert len(payload) == 95 + 2340, f"After Section 2: expected 2435, got {len(payload)}"

    # ========== SECTION 3: Unit Stats (468 bytes) ==========
    # 39 units × 12 bytes: [scale: fixed16][regen: fixed16][max_health: u32]
    for _ in range(39):
        payload += pack_fixed16(1.0)      # scale
        payload += pack_fixed16(100.0)    # regen_or_health_related
        payload += struct.pack(">I", 100) # max_health

    assert len(payload) == 95 + 2340 + 468, f"After Section 3: expected 2903, got {len(payload)}"

    # ========== SECTION 4: Vehicle Physics (72 bytes) ==========
    # 2 vehicles × 36 bytes each (matching wulf-forge vehicle_physics_count=2)
    # Format: speed(f16) + accel(f16) + torque(i32) + stiffness(i32) +
    #         friction(f16) + turn_rate(f16) + dampening(f16) + unk(i32) + mass(i32)
    for _ in range(2):
        payload += pack_fixed16(20.0)     # speed
        payload += pack_fixed16(4.0)      # accel
        payload += struct.pack(">I", 700) # engine_torque
        payload += struct.pack(">I", 550) # suspension_stiffness
        payload += pack_fixed16(BEHAVIOR_GROUND_FRICTION)      # ground_friction (wulf-forge packets.toml)
        payload += pack_fixed16(BEHAVIOR_TURN_RATE)            # turn_rate (wulf-forge packets.toml)
        payload += pack_fixed16(BEHAVIOR_SUSPENSION_DAMPENING) # suspension_dampening (wulf-forge packets.toml)
        payload += struct.pack(">I", 0)   # unknown_int_30
        payload += struct.pack(">I", 33000)  # mass

    assert len(payload) == 95 + 2340 + 468 + 72, f"After Section 4: expected 2975, got {len(payload)}"

    # ========== SECTION 5: Hardpoints ==========
    # 4 hardpoint blocks, each with count + optional data + trailing fixed16
    # wulf-forge now emits thruster hardpoints for tank/scout (count=2).
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
        # Match current wulf-forge behavior packet layout:
        # send thruster hardpoints for all four team/vehicle variants.
        _write_hardpoint_block(2, True)   # Red tank
        _write_hardpoint_block(2, True)   # Blue tank
        _write_hardpoint_block(2, True)   # Red scout
        _write_hardpoint_block(2, True)   # Blue scout
    else:
        for _ in range(4):
            _write_hardpoint_block(0, False)

    section5_size = len(payload) - section5_start
    assert len(payload) == 2975 + section5_size, f"After Section 5: expected {2975 + section5_size}, got {len(payload)}"

    # ========== SECTION 6: Active Vehicle Physics ==========
    # Vehicle-specific sizes (tank=7, scout=9, bomber=11 fixed16 values).
    section6_start = len(payload)
    for i in range(3):
        if not BEHAVIOR_ACTIVE_EXTRAS:
            payload += pack_fixed16(4.5)      # turn_adjust
            payload += pack_fixed16(85.0)     # move_adjust
            payload += pack_fixed16(69.7)     # strafe_adjust
            payload += pack_fixed16(80.0)     # max_velocity
            payload += pack_fixed16(2000.0)   # low_fuel_level
            payload += pack_fixed16(BEHAVIOR_MAX_ALTITUDE)  # max_altitude (wulf-forge packets.toml)
            payload += pack_fixed16(BEHAVIOR_GRAVITY_PCT)   # gravity_pct (wulf-forge packets.toml)
            continue

        if i == 0:
            payload += pack_fixed16(4.5)      # turn_adjust
            payload += pack_fixed16(85.0)     # move_adjust
            payload += pack_fixed16(69.7)     # strafe_adjust
            payload += pack_fixed16(80.0)     # max_velocity
            payload += pack_fixed16(2000.0)   # low_fuel_level
            payload += pack_fixed16(BEHAVIOR_MAX_ALTITUDE)  # max_altitude (wulf-forge packets.toml)
            payload += pack_fixed16(BEHAVIOR_GRAVITY_PCT)   # gravity_pct (wulf-forge packets.toml)
        elif i == 1:
            payload += pack_fixed16(4.5)      # turn_adjust
            payload += pack_fixed16(85.0)     # move_forward_adjust
            payload += pack_fixed16(38.0)     # move_backward_adjust
            payload += pack_fixed16(72.0)     # strafe_adjust
            payload += pack_fixed16(85.0)     # max_velocity
            payload += pack_fixed16(2000.0)   # low_fuel_level
            payload += pack_fixed16(4.9)      # max_altitude
            payload += pack_fixed16(3.5)      # max_speed_height_pickup
            payload += pack_fixed16(BEHAVIOR_GRAVITY_PCT)   # gravity_pct (wulf-forge packets.toml)
        else:
            payload += pack_fixed16(-2.5132741233144)  # ax_mag
            payload += pack_fixed16(2.35619449060725)  # ay_mag
            payload += pack_fixed16(80.0)              # forward_mag
            payload += pack_fixed16(45.0)              # low_airspeed
            payload += pack_fixed16(0.5)               # angfac
            payload += pack_fixed16(70.0)              # turn_low
            payload += pack_fixed16(110.0)             # turn_high
            payload += pack_fixed16(340.0)             # turn_zero
            payload += pack_fixed16(1000.0)            # very_high
            payload += pack_fixed16(1800.0)            # ceiling
            payload += pack_fixed16(2000.0)            # low_fuel_level

    section6_size = len(payload) - section6_start
    expected_section6 = 84 if not BEHAVIOR_ACTIVE_EXTRAS else (7 + 9 + 11) * 4
    assert section6_size == expected_section6, f"Section 6 size mismatch: got {section6_size}, expected {expected_section6}"
    current_size = 2975 + section5_size + section6_size
    assert len(payload) == current_size, f"After Section 6: expected {current_size}, got {len(payload)}"

    # ========== PADDING to target size (3116) ==========
    target_size = 3116
    packet = bytearray()
    packet.append(0x24)  # Packet type
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
    """
    Return per-weapon-type capability counts derived from the BEHAVIOR packet.

    Each entry is (ammo_capable_count, fire_capable_count, active_capable_count, cooldown_capable_count)
    across enabled slots for that weapon type.
    """
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
    """
    Build TRANSLATION/quantizer packet (0x32).
    Initializes the quantizer array used by UPDATE_ARRAY.
    28 quantizers, no count field - client expects exactly 28.
    Each quantizer: 3x u32 + 2x strings (max/range as float strings).
    Format matches Wulf-Forge/RE docs:
      - fixed_bits (precision header bits)
      - padding (u32)
      - max_total_bits (u32)
      - max_value (string)
      - range_value (string)
    """
    payload = bytearray()
    payload.append(0x32)  # Packet type

    def _write_string(text: str) -> bytes:
        raw = (text + "\x00").encode("ascii")
        return struct.pack(">H", len(raw)) + raw

    def _write_entry(fixed_bits: int, max_total_bits: int, max_str: str, range_str: str) -> None:
        payload.extend(struct.pack(">I", fixed_bits))
        payload.extend(struct.pack(">I", 0))  # padding
        payload.extend(struct.pack(">I", max_total_bits))
        payload.extend(_write_string(max_str))
        payload.extend(_write_string(range_str))

    # Scalars (0-15) - MUST match wulf-forge's translation_config.py!
    scalar_configs = [(16, 0, "1000.0", "2000.0") for _ in range(16)]
    scalar_configs[1] = (5, 0, "0.0", "0.0")   # weapon id (5 bits)
    scalar_configs[2] = (8, 0, "0.0", "0.0")   # unit type (8 bits) - wulf-forge uses 8!
    scalar_configs[3] = (8, 0, "0.0", "0.0")   # team id (8 bits) - wulf-forge uses 8!
    scalar_configs[4] = (8, 0, "0.0", "0.0")   # unit type in cargo (8 bits)
    scalar_configs[5] = (10, 0, f"{HEALTH_MAX}", f"{HEALTH_RANGE}")  # health (10 bits)
    scalar_configs[8] = (10, 0, f"{ENERGY_MAX}", f"{ENERGY_RANGE}")  # energy (10 bits)
    scalar_configs[13] = (8, 0, "1.0", "1.0")  # extra A
    scalar_configs[14] = (8, 0, "1.0", "1.0")  # extra B

    for cfg in scalar_configs:
        _write_entry(*cfg)

    # Vectors (16-27): 3 banks of 4 vectors
    # Format: (header_bits, total_bits, max_str, range_str)
    # MUST MATCH what we actually write in UPDATE_ARRAY packets!
    # Using 16 bits for all vectors (matching wulf-forge's actual encoding)
    vector_templates = [
        (4, 16, f"{VEC_POS_MAX}", f"{VEC_POS_RANGE}"),   # position
        (4, 16, f"{VEC_VEL_MAX}", f"{VEC_VEL_RANGE}"),   # velocity
        (4, 16, f"{VEC_ROT_MAX}", f"{VEC_ROT_RANGE}"),   # rotation
        (4, 16, f"{VEC_SPIN_MAX}", f"{VEC_SPIN_RANGE}"), # spin
    ]

    for _ in range(3):
        for cfg in vector_templates:
            _write_entry(*cfg)

    return bytes(payload)
