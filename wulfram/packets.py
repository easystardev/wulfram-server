"""
Packet definitions and builder functions.
Pure functions that return packet payloads - no I/O.
"""

import struct
import time
import math
from typing import Optional, Tuple, List
from .codec import BitWriter, pack_fixed16, frame_packet


# Server tick clock - ticks relative to server start (matches wulf-forge)
_SERVER_START = time.monotonic()

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
    # Code is sent directly - donor flag handled elsewhere
    return b'\x22\x01' + struct.pack("B", code)


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
    map_name: str = "bpass",  # Match wulf-forge default
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

    Format from Wulf-Forge (working):
    - u32 account_id (player_id)
    - u32 team (NOT entity_id! Wulf-Forge puts team here as u32)
    - u16 kills
    - u16 deaths
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
    payload += struct.pack(">I", team)          # Wulf-Forge puts team in second u32!
    payload += struct.pack(">H", 0)             # kills (u16)
    payload += struct.pack(">H", 0)             # deaths (u16)
    payload += struct.pack(">H", len(name_bytes)) + name_bytes
    payload += struct.pack(">H", len(clan_bytes)) + clan_bytes
    payload += struct.pack(">H", 0)             # kills2 (u16)
    payload += struct.pack(">H", 0)             # deaths2 (u16)
    payload += pack_fixed16(0.0)                # Score as fixed16.16 (4 bytes)
    payload += struct.pack(">I", 0)             # unknown
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
                      vel: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                      *, include_local_state: bool = False,
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
    - 3x fixed16 velocity (96 bits)

    Client uses bit-based reading (read_u32 = read_bits(32)), so we use BitWriter.
    """
    bw = BitWriter()

    # Packet type (8 bits)
    bw.write_bits(8, 0x18)

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
    )

    # Vehicle type (32 bits) - 0-4 are valid tank types
    bw.write_bits(32, vehicle_type)

    # Frame ID (32 bits)
    bw.write_bits(32, entity_oid)  # Use OID as frame ID

    # Properties (8 bits)
    bw.write_bits(8, 0)

    # Position (3x fixed16.16 = 3x 32 bits)
    x, y, z = pos
    bw.write_bits(32, int(x * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(y * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(z * 65536.0) & 0xFFFFFFFF)

    # Velocity (3x fixed16.16 = 3x 32 bits)
    vx, vy, vz = vel
    bw.write_bits(32, int(vx * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(vy * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(vz * 65536.0) & 0xFFFFFFFF)

    return bw.get_bytes()


def build_udp_tank_packet_wf(
    net_id: int,
    unit_type: int,
    team_id: int,
    pos: Tuple[float, float, float],
    vel: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    tick: Optional[int] = None,
    include_vitals: bool = False,  # Try False: skip local_state quantizer reads entirely
    weapon_id: int = 0,
    health_mult_bits: int = 1,  # Wulf-forge default (not 1023!)
    energy_mult_bits: int = 1,  # Wulf-forge default (not 1023!)
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
    - 3x fixed16 velocity
    """
    if tick is None:
        tick = get_ticks()

    bw = BitWriter()
    bw.write_bits(32, tick)

    bw.write_bits(1, 1 if include_vitals else 0)
    if include_vitals:
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
    for value in vel:
        bw.write_bits(32, int(value * 65536.0) & 0xFFFFFFFF)

    return b'\x18' + bw.get_bytes()


def build_tank(entity_type: int, oid: int, pos: Tuple[float, float, float],
               rot: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> bytes:
    """Build TANK/PLAYER_INFO packet for spawning vehicles (legacy)."""
    # Use new PLAYER_INFO format
    return build_player_info(oid, entity_type, pos)


def build_tank_packet(net_id: int, unit_type: int, pos: Tuple[float, float, float],
                      vel: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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
    - 3x fixed16 velocity
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
        # Health multiplier (10 bits) - uses quantizer encoding
        # Wulf-forge uses raw value 1 = 100% health (NOT health * 1023!)
        # The quantizer decodes: value = max - ((raw-1) * range) / denom
        # So raw=1 -> max, raw=1023 -> min
        bw.write_bits(10, 1)   # Health = 100% (raw value 1)
        # Energy multiplier (10 bits) - same quantizer encoding
        bw.write_bits(10, 1)   # Energy = 100% (raw value 1)

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

    # Velocity (3x fixed16.16)
    vx, vy, vz = vel
    bw.write_bits(32, int(vx * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(vy * 65536.0) & 0xFFFFFFFF)
    bw.write_bits(32, int(vz * 65536.0) & 0xFFFFFFFF)

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
                              turret_range: float = 12.6) -> None:
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
    """
    if not include:
        bw.write_bits(1, 0)
        return

    bw.write_bits(1, 1)
    bw.write_bits(5, weapon_id & 0x1F)
    bw.write_bits(10, _compress_value(health, 1.0, 1.0, total_bits=10))
    bw.write_bits(10, _compress_value(fuel, 1.0, 1.0, total_bits=10))
    if ammo_count_bits:
        bw.write_bits(ammo_count_bits, ammo_count & ((1 << ammo_count_bits) - 1))

    # Turret angles (if weapon def flags indicate they exist)
    if primary_turret_bits:
        bw.write_bits(primary_turret_bits, _compress_value(primary_turret_angle, turret_max, turret_range, total_bits=primary_turret_bits))
    if secondary_turret_bits:
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
                                 turret_range: float = 12.6) -> bytes:
    """Build UPDATE_ARRAY with entity heartbeat and optional health data.

    When include_health=True, sends local player state with full health/energy.

    - Entity: net_id(32) + is_manned(1) + mask(10) + bank_selector(16)

    NO ammo, NO turret angles - those were causing crashes!
    """
    # Tick is read as byte-aligned u32 by client, NOT from bitstream
    tick_bytes = struct.pack(">I", tick)
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
    )

    # Include 1 entity (the player) - client may ignore health when entity_count=0
    bw.write_bits(8, 1)           # 1 entity
    bw.write_bits(32, entity_id)  # OID
    bw.write_bits(1, 1)           # is_manned = True (local player)
    bw.write_bits(10, 0)          # Update mask = 0 (no updates, just heartbeat)
    bw.write_bits(16, 0)          # Bank selector = 0
    return b'\x0E' + tick_bytes + bw.get_bytes()


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


def build_update_array_player_update(tick: int, entity_id: int,
                                     pos: Tuple[float, float, float],
                                     vel: Tuple[float, float, float],
                                     rot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                     *,
                                     include_local_state: bool = True,
                                     include_entity_vitals: bool = False,
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
    bw.write_bits(1, 1)           # is_manned = True (player vehicle)

    # Update mask: POS | VEL | ROT (+ SPEED_SCALE/FUEL)
    update_mask = 0b0000001110
    if include_entity_vitals:
        update_mask |= (1 << 5)  # speed scale
        update_mask |= (1 << 7)  # fuel
    bw.write_bits(10, update_mask)

    # Bank selector
    bw.write_bits(16, 0)

    # Position vector (bank 0, 16-bit precision) - must match wulf-forge VEC_POS config
    # max=8192, range=16384 (min=-8192)
    bw.write_bits(4, 15)
    for v in pos:
        bw.write_bits(16, _compress_value(v, 8192.0, 16384.0, total_bits=16))

    # Velocity vector (bank 0, 16-bit precision) - wulf-forge VEC_VEL config
    # max=200, range=400 (min=-200)
    bw.write_bits(4, 15)
    for v in vel:
        bw.write_bits(16, _compress_value(v, 200.0, 400.0, total_bits=16))

    # Rotation vector (bank 0, 16-bit precision) - wulf-forge VEC_ROT config
    # max=6.3, range=12.6 (covers ±2π radians)
    bw.write_bits(4, 15)
    for v in rot:
        bw.write_bits(16, _compress_value(v, 6.3, 12.6, total_bits=16))

    if include_entity_vitals:
        bw.write_bits(10, _compress_value(speed_scale, 1.0, 1.0, total_bits=10))  # speed scale
        bw.write_bits(10, _compress_value(fuel, 1.0, 1.0, total_bits=10))         # fuel fraction

    return b'\x0E' + tick_bytes + bw.get_bytes()


def _compress_position(value: float, max_val: float = 8192.0, range_val: float = 16384.0,
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


def _compress_rotation(value: float, max_val: float = 6.3, range_val: float = 12.6,
                       total_bits: int = 16) -> Tuple[int, int]:
    """
    Compress a rotation value using Wulf-Forge's quantization scheme.
    Matches VEC_ROT: max=6.3, range=12.6, total_bits=16.
    """
    return (15, _compress_wulfforge(value, max_val=max_val, range_val=range_val, total_bits=total_bits))


def build_update_array_create_tank(tick: int, entity_id: int, entity_type: int, team: int,
                                    pos: Tuple[float, float, float], behavior_type: int = 0,
                                    include_interp: bool = False, interp_bits: int = 16,
                                    include_health: bool = True) -> bytes:
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
    _write_local_player_state(bw, include_health)
    bw.write_bits(8, 1)            # 1 entity

    # Entity header (matches wulf-forge update_array.py EntitySerializer.serialize)
    bw.write_bits(32, entity_id)   # net_id (OID)
    bw.write_bits(1, 1)            # is_manned = True

    # Presence flags: bit 0 (creation) + bit 1 (position) + bit 3 (rotation)
    # Binary: 0b0000001011 = 11 decimal
    presence_flags = 0b0000001011
    bw.write_bits(10, presence_flags)

    # CRITICAL: Bank selector (16 bits) - wulf-forge puts this AFTER presence flags!
    # We were MISSING this! Selects quantizer bank (0 = bank 0 = indices 16-19)
    bw.write_bits(16, 0)           # Bank selector = 0

    # --- CREATION BLOCK (Bit 0 / DEFINITION) ---
    # Wulf-forge uses 8-bit fields for unit type and team IDs (quantizer[2] and [3])
    bw.write_bits(8, entity_type & 0xFF)   # unit_type (8 bits, quantizer[2])
    bw.write_bits(8, team & 0xFF)          # team_id (8 bits, quantizer[3])
    bw.write_bits(8, team & 0xFF)          # team_id_also/state (8 bits, quantizer[3])
    bw.write_bits(1, 1)                    # is_teleport_or_snap = True (force position)

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
        bw.write_bits(8, team & 0xFF)          # team_id
        bw.write_bits(8, team & 0xFF)          # team_id_also/state
        bw.write_bits(1, 1)                    # is_teleport_or_snap = True

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
            quantized = _compress_wulfforge(coord, max_val=8192.0, range_val=16384.0, total_bits=16)
            bw.write_bits(16, quantized)

    return b'\x0E' + tick_bytes + bw.get_bytes()


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

    # ========== SECTION 1: Header (95 bytes) ==========
    # Matches wulf-forge BehaviorHeader defaults
    payload.append(0x00)                  # spawn_related (wulf-forge default: 0)
    payload += pack_fixed16(5.0)          # timeout
    payload += pack_fixed16(100.0)        # dbl_6792F8
    payload += pack_fixed16(100.0)        # velocity_q
    payload += pack_fixed16(100.0)        # dbl_679308
    payload += pack_fixed16(100.0)        # dbl_679310

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
    TANK_SLOT_CONFIG = {
        # slot: [enabled, ammo_capable, fire_capable, active_capable, cooldown_capable]
        0:  [1, 0, 0, 0, 0],  # Slot 0: Chain gun - ONLY WORKING SLOT
    }

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
        payload += pack_fixed16(0.35)     # ground_friction (reduced for smoother sliding)
        payload += pack_fixed16(0.4)      # turn_rate (increased for responsive steering)
        payload += pack_fixed16(2.0)      # suspension_dampening
        payload += struct.pack(">I", 0)   # unknown_int_30
        payload += struct.pack(">I", 33000)  # mass

    assert len(payload) == 95 + 2340 + 468 + 72, f"After Section 4: expected 2975, got {len(payload)}"

    # ========== SECTION 5: Hardpoints ==========
    # 4 hardpoint blocks, each with count + optional data + trailing fixed16
    # wulf-forge uses count=0 for all blocks, so just count(4 bytes) + fixed16(4 bytes) each
    for _ in range(4):
        payload += struct.pack(">I", 0)   # count = 0
        payload += pack_fixed16(0.0)      # trailing fixed16

    section5_size = 4 * 8  # 4 blocks × (4 + 4) = 32 bytes
    assert len(payload) == 2975 + section5_size, f"After Section 5: expected {2975 + section5_size}, got {len(payload)}"

    # ========== SECTION 6: Active Vehicle Physics (84 bytes) ==========
    # 3 vehicles × 28 bytes each (matching wulf-forge active_vehicles_count=3)
    # Format: 7 fixed16 values × 4 bytes = 28 bytes per vehicle
    # CRITICAL: gravity_pct enables physics!
    for _ in range(3):
        payload += pack_fixed16(4.5)      # turn_adjust
        payload += pack_fixed16(85.0)     # move_adjust
        payload += pack_fixed16(69.7)     # strafe_adjust
        payload += pack_fixed16(80.0)     # max_velocity
        payload += pack_fixed16(2000.0)   # low_fuel_level
        payload += pack_fixed16(5.0)      # max_altitude (wulf-forge default)
        payload += pack_fixed16(0.5)      # gravity_pct - ENABLES PHYSICS!

    section6_size = 3 * 28  # 84 bytes
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
    scalar_configs[5] = (10, 0, "1.0", "1.0")  # health (10 bits)
    scalar_configs[8] = (10, 0, "1.0", "1.0")  # energy (10 bits)
    scalar_configs[13] = (8, 0, "1.0", "1.0")  # extra A
    scalar_configs[14] = (8, 0, "1.0", "1.0")  # extra B

    for cfg in scalar_configs:
        _write_entry(*cfg)

    # Vectors (16-27): 3 banks of 4 vectors
    # Format: (header_bits, total_bits, max_str, range_str)
    # MUST MATCH what we actually write in UPDATE_ARRAY packets!
    # Using 16 bits for all vectors (matching wulf-forge's actual encoding)
    vector_templates = [
        (4, 16, "8192.0", "16384.0"),  # position: 16 bits, range -8192 to +8192
        (4, 16, "200.0", "400.0"),     # velocity: 16 bits, range -200 to +200
        (4, 16, "6.3", "12.6"),        # rotation: 16 bits, range -6.3 to +6.3
        (4, 16, "10.0", "20.0"),       # spin: 16 bits, range -10 to +10
    ]

    for _ in range(3):
        for cfg in vector_templates:
            _write_entry(*cfg)

    return bytes(payload)
