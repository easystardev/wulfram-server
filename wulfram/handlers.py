"""
Packet handlers extracted from server.py.
Contains TCP and UDP packet handling logic.
"""

import struct
import time
from typing import TYPE_CHECKING, Optional, Tuple

from .session import Phase, FEATURES
from .packets import (
    PacketType, build_hello_version, build_hello_session_key,
    build_login_status, build_player, build_team_info,
    build_world_stats, build_bps_response, build_chat_message,
    build_add_to_roster, build_update_stats, build_tank_packet,
    build_update_array_empty, build_update_array_spawn_points,
    build_behavior_packet, build_translation_packet, build_game_clock,
    build_motd,
)

if TYPE_CHECKING:
    from .server import WulframServer
    from .client import ClientContext


def decode_lp_string(data: bytes, offset: int) -> Tuple[str, int]:
    """Decode a length-prefixed string. Returns (text, new_offset)."""
    if offset + 2 > len(data):
        return "", offset
    length = struct.unpack(">H", data[offset:offset + 2])[0]
    start = offset + 2
    end = min(start + length, len(data))
    raw = data[start:end]
    text = raw.rstrip(b"\x00").decode("ascii", errors="ignore")
    return text, end


# ============ TCP Handlers ============

def handle_hello(server: "WulframServer", ctx: "ClientContext", packet: bytes):
    """Handle HELLO packets during login."""
    if len(packet) < 2:
        return

    subcmd = packet[1]

    if subcmd == 0x00 and len(packet) >= 6:
        # Subcmd 0 = Version check
        version = struct.unpack(">I", packet[2:6])[0]
        print(f"[LOGIN] Client {ctx.client_id} version: 0x{version:04X}")
        ctx.tcp_handler.send(build_hello_version(version))

    elif subcmd == 0x02:
        # Subcmd 2 = Session key request
        print(f"[LOGIN] Client {ctx.client_id}: Session key requested")
        ctx.tcp_handler.send(build_hello_session_key("WulframSessionKey123"))


def handle_login_request(server: "WulframServer", ctx: "ClientContext", packet: bytes):
    """Handle LOGIN_REQUEST packet."""
    if len(packet) < 2:
        return

    session = ctx.session
    if session.login_complete:
        print(f"[LOGIN] Client {ctx.client_id}: Ignoring LOGIN_REQUEST after login complete")
        return

    sub_type = packet[1]
    offset = 2
    username, offset = decode_lp_string(packet, offset)
    password, offset = decode_lp_string(packet, offset)

    if username:
        session.username = username

    # Parse optional game-service extra data (subtypes 2/3/4)
    extra_count = None
    if sub_type in (0x02, 0x03, 0x04) and offset + 4 <= len(packet):
        extra_count = struct.unpack(">I", packet[offset:offset + 4])[0]
        offset += 4
        for _ in range(extra_count):
            _, offset = decode_lp_string(packet, offset)
        _, offset = decode_lp_string(packet, offset)  # server name

    if extra_count is None:
        print(f"[LOGIN] Client {ctx.client_id}: sub_type=0x{sub_type:02X} user={username}")
    else:
        print(f"[LOGIN] Client {ctx.client_id}: sub_type=0x{sub_type:02X} user={username} extra_count={extra_count}")

    # Game-service login: sub_type 0x01 -> request continuation (status 2)
    if sub_type == 0x01:
        if not session.login_game_service_requested:
            session.login_game_service_requested = True
            ctx.tcp_handler.send(build_login_status(2))
        else:
            ctx.tcp_handler.send(build_login_status(2))
        return

    # Normal login: sub_type 0x00 -> request password (status 1)
    if sub_type == 0x00 and not session.login_password_requested:
        session.login_password_requested = True
        ctx.tcp_handler.send(build_login_status(1))
        return

    # Final login success
    if sub_type in (0x00, 0x02, 0x03, 0x04) or session.login_game_service_requested:
        print(f"[LOGIN] Client {ctx.client_id}: Login successful!")
        session.login_complete = True
        session.transition_to(Phase.TEAM_SELECT)
        send_initial_game_data(server, ctx)
        return

    print(f"[LOGIN] Client {ctx.client_id}: Unhandled sub_type 0x{sub_type:02X}")


def send_initial_game_data(server: "WulframServer", ctx: "ClientContext"):
    """Send packets needed for team selection screen."""
    session = ctx.session
    tcp = ctx.tcp_handler

    # Team info
    tcp.send(build_team_info())

    # LOGIN_STATUS(8) after TEAM_INFO
    tcp.send(build_login_status(8, is_donor=True))

    # PLAYER with spectator=False
    if session.player_id == 0:
        session.player_id = ctx.entity_id
        tcp.send(build_player(entity_id=session.player_id, spectator=False))

    # GameClock
    tcp.send(build_game_clock())

    # MOTD
    tcp.send(build_motd("Welcome to Wulfram!"))

    # Behavior packet
    if FEATURES.send_behavior_packet and not session.behavior_sent:
        tcp.send(build_behavior_packet())
        session.behavior_sent = True

    # TRANSLATION
    if FEATURES.send_translation_packet:
        tcp.send(build_translation_packet())

    # ADD_TO_ROSTER
    if not session.roster_sent:
        name = session.username or f"Player{ctx.client_id}"
        tcp.send(build_add_to_roster(
            player_id=session.player_id,
            entity_id=session.player_id,
            name=name,
            team=1
        ))
        session.roster_sent = True


def handle_bps(server: "WulframServer", ctx: "ClientContext", packet: bytes):
    """Handle BPS (bandwidth/rate) packet."""
    session = ctx.session
    tcp = ctx.tcp_handler

    if len(packet) >= 5:
        rate_index = struct.unpack(">I", packet[1:5])[0]
        print(f"[GAME] Client {ctx.client_id}: BPS request: rate={rate_index}")
        tcp.send(build_bps_response(rate_index, approved=True))

        # Send WORLD_STATS after BPS if not yet sent
        if (FEATURES.send_world_stats_on_login and not session.world_stats_sent
                and session.phase == Phase.TEAM_SELECT):
            time.sleep(0.2)
            print(f"[GAME] Client {ctx.client_id}: Sending WORLD_STATS after BPS request")
            tcp.send(build_world_stats())
            session.world_stats_sent = True


def handle_want_updates(server: "WulframServer", ctx: "ClientContext", packet: bytes):
    """Handle WANT_UPDATES - client is ready for game data."""
    session = ctx.session
    tcp = ctx.tcp_handler
    now = time.monotonic()

    session.want_updates_received = True
    session.want_updates_time = now

    # Guard against duplicate WANT_UPDATES
    if session.in_game:
        print(f"[GAME] Client {ctx.client_id}: Ignoring duplicate WANT_UPDATES (already in game)")
        session.want_updates_handled = True
        session.want_updates_handled_time = now
        return

    if session.suppress_want_updates_payload:
        print(f"[GAME] Client {ctx.client_id}: WANT_UPDATES payload suppressed")
        session.want_updates_handled = True
        session.want_updates_handled_time = now
        return

    since_connect = now - session.connected_at
    print(f"[GAME] Client {ctx.client_id}: ready for updates (t+{since_connect:.2f}s)")

    # Start ping loop
    server._start_ping_loop(ctx)

    # Send welcome chat
    tcp.send(build_chat_message("System: Welcome to Wulfram!"))

    # Send empty update array
    if FEATURES.send_update_array_empty:
        tcp.send(build_update_array_empty())

    # Send spawn points
    if FEATURES.send_spawn_points:
        spawn_points = [
            {'oid': 5001, 'team': 1, 'x': 50.0, 'y': 10.0, 'z': 50.0},
            {'oid': 5002, 'team': 1, 'x': 60.0, 'y': 10.0, 'z': 50.0},
            {'oid': 5003, 'team': 2, 'x': 150.0, 'y': 10.0, 'z': 150.0},
            {'oid': 5004, 'team': 2, 'x': 160.0, 'y': 10.0, 'z': 150.0},
        ]
        if hasattr(server, "_to_client_pos"):
            for sp in spawn_points:
                x, y, z = server._to_client_pos((sp["x"], sp["y"], sp["z"]))
                sp["x"], sp["y"], sp["z"] = x, y, z
        print(f"[GAME] Client {ctx.client_id}: Sending {len(spawn_points)} spawn points")
        tcp.send(build_update_array_spawn_points(0, spawn_points))

    # Spawn if team was selected
    if session.pending_spawn_team_id:
        server._auto_join_team(ctx, session.pending_spawn_team_id)
        session.pending_spawn_team_id = 0
        session.want_updates_handled = True
        session.want_updates_handled_time = now
        return

    session.want_updates_handled = True
    session.want_updates_handled_time = now

    # Auto-join team
    if FEATURES.auto_join_team:
        team = session.team_id or session.pending_spawn_team_id or 1
        session.delayed_spawn_time = now + 3.0
        session.delayed_spawn_team = team
        print(f"[GAME] Client {ctx.client_id}: Scheduled auto-spawn in 3 seconds for team {team}")
    else:
        print(f"[GAME] Client {ctx.client_id}: Auto-spawn disabled")


def handle_reincarnate_tcp(server: "WulframServer", ctx: "ClientContext", packet: bytes):
    """Handle TCP REINCARNATE - player wants to spawn."""
    session = ctx.session
    tcp = ctx.tcp_handler

    team_id = 2
    if len(packet) >= 2:
        sub_type = packet[1]
        if sub_type == 0x01:
            team_id = 1
        elif sub_type == 0x03:
            team_id = 2

    print(f"[GAME] Client {ctx.client_id}: Spawn request for team {team_id}")

    session.pending_spawn_team_id = team_id
    session.team_id = team_id

    # Send UPDATE_STATS
    tcp.send(build_update_stats(
        account_id=session.player_id or ctx.entity_id,
        team_id=team_id
    ))

    # Send WORLD_STATS if not sent
    if not session.world_stats_sent:
        tcp.send(build_world_stats())
        session.world_stats_sent = True


# ============ UDP Handlers ============

def handle_udp_d_handshake(server: "WulframServer", ctx: Optional["ClientContext"], data: bytes, addr: tuple):
    """Handle UDP D_HANDSHAKE (0x03) and respond with stream definitions."""
    if len(data) < 13:
        return

    timestamp = struct.unpack(">I", data[1:5])[0]
    conn_id = struct.unpack(">I", data[5:9])[0]
    stream_count = struct.unpack(">I", data[9:13])[0]
    print(f"[UDP] D_HANDSHAKE time={timestamp} id={conn_id} streams={stream_count}")

    if ctx:
        ctx.session.udp_d_handshake_received = True
        ctx.session.udp_addr = addr
        # Register UDP address for this client
        server.udp_addr_to_client[addr] = ctx

    # ACK
    ack = b'\x02' + b'\x00' + struct.pack(">I", int(time.monotonic() * 1000) & 0xFFFFFFFF)
    server.udp_handler.send_to(ack, addr)
    print(f"[UDP SEND] D_ACK to {addr}")

    # Stream definitions
    def _pack_lp_string(text: str) -> bytes:
        raw = (text + '\x00').encode('ascii', errors='ignore')
        return struct.pack(">H", len(raw)) + raw

    payload = bytearray()
    payload += struct.pack(">I", int(time.monotonic() * 1000) & 0xFFFFFFFF)
    player_id = ctx.session.player_id if ctx else 1001
    payload += struct.pack(">I", player_id or 1001)
    payload += struct.pack(">I", 4)

    for name, sid in (("Unreliable", 0), ("Reliable", 1), ("Stream 2", 2), ("Game Data", 3)):
        payload += _pack_lp_string(name)
        payload += struct.pack(">I", 1)
        payload += struct.pack(">I", sid)

    payload += struct.pack(">I", 4)
    for sid in (0, 1, 2, 3):
        payload += struct.pack(">I", sid)
        payload += struct.pack(">I", 1)

    server.udp_handler.send_to(b'\x03' + bytes(payload), addr)
    print(f"[UDP SEND] D_STREAM_DEFS to {addr}")

    # Unpause streams
    for sid in (1, 3):
        server.udp_handler.send_to(b'\x04' + struct.pack(">BH", sid, 1), addr)
        print(f"[UDP SEND] D_UNPAUSE stream={sid} to {addr}")


def handle_udp_chat(server: "WulframServer", ctx: Optional["ClientContext"], data: bytes, addr: tuple):
    """Handle UDP COMM_REQ (0x20) for /s spawn."""
    if len(data) < 9:
        return

    seq_num = struct.unpack(">H", data[1:3])[0]
    length = struct.unpack(">H", data[3:5])[0]
    source = struct.unpack(">H", data[5:7])[0]
    msg, _ = decode_lp_string(data, 9)

    print(f"[UDP] COMM_REQ seq={seq_num} len={length} source={source} msg='{msg}'")

    # ACK
    send_udp_ack(server, ctx, addr, 0x20, seq_num)

    if ctx is None:
        return

    cmd = msg.strip()
    if cmd.lower().startswith("/s "):
        cmd = cmd[3:].strip()

    if cmd.lower() == "spawn":
        net_id = ctx.session.player_id or ctx.entity_id
        team_id = ctx.session.team_id or 2
        server._spawn_wf_style(ctx, team_id=team_id, net_id=net_id)

    elif cmd.lower() in ("fire", "pulse", "pew"):
        # Test command: spawn a pulse shell projectile
        _spawn_test_projectile(server, ctx, addr)


def handle_udp_reincarnate(server: "WulframServer", ctx: Optional["ClientContext"], data: bytes, addr: tuple):
    """Handle UDP REINCARNATE (0x25)."""
    if ctx is None:
        print(f"[UDP] REINCARNATE from unknown client at {addr}")
        return

    session = ctx.session

    if len(data) < 6:
        print(f"[UDP] REINCARNATE too short (len={len(data)})")
        return

    seq_num = struct.unpack(">H", data[1:3])[0]
    length = struct.unpack(">H", data[3:5])[0]
    payload = data[5:]

    # ACK immediately
    send_udp_ack(server, ctx, addr, 0x25, seq_num)

    if len(payload) < 1:
        print(f"[UDP] REINCARNATE seq={seq_num} len={length} - empty payload")
        return

    subtype = payload[0]

    if subtype == 0x00:
        # Type 0: Spawn at spawn point
        if len(payload) >= 9:
            spawn_point_id = struct.unpack(">I", payload[1:5])[0]
            vehicle_type = struct.unpack(">I", payload[5:9])[0]
            print(f"[UDP] REINCARNATE (spawn) seq={seq_num} spawn_point={spawn_point_id} vehicle={vehicle_type}")
            handle_spawn_at_point(server, ctx, spawn_point_id, vehicle_type, addr)
        else:
            print(f"[UDP] REINCARNATE (spawn) seq={seq_num} - incomplete payload")

    elif subtype == 0x01:
        # Type 1: Team switch
        if len(payload) >= 5:
            team_id = struct.unpack(">I", payload[1:5])[0]
            print(f"[UDP] REINCARNATE (team_switch) seq={seq_num} team={team_id}")
            handle_team_switch(server, ctx, team_id, addr)
        else:
            print(f"[UDP] REINCARNATE (team_switch) seq={seq_num} - incomplete payload")

    else:
        # Unknown subtype - try fallback
        print(f"[UDP] REINCARNATE seq={seq_num} len={length} unknown_subtype=0x{subtype:02X}")
        if len(payload) >= 5:
            team_id = struct.unpack(">I", payload[1:5])[0]
            if team_id in (1, 2):
                print(f"[UDP] REINCARNATE (fallback) team={team_id}")
                handle_team_switch(server, ctx, team_id, addr)


def handle_team_switch(server: "WulframServer", ctx: "ClientContext", team_id: int, addr: tuple):
    """Handle team switch/spawn request."""
    session = ctx.session

    if team_id not in (1, 2):
        print(f"[UDP] Invalid team_id {team_id}")
        return

    print(f"[UDP] Client {ctx.client_id}: Team switch to team {team_id}")
    session.team_id = team_id
    session.pending_spawn_team_id = team_id

    if server.udp_handler and addr:
        # UPDATE_STATS
        update_stats_pkt = build_update_stats(
            account_id=session.player_id or ctx.entity_id,
            team_id=team_id
        )
        server.udp_handler.send_to(update_stats_pkt, addr)

        # REINCARNATE code=0x11
        reincarnate_pkt = b'\x25\x11\x00' + struct.pack("B", team_id) + b'\x00'
        server.udp_handler.send_to(reincarnate_pkt, addr)
        print(f"[UDP] Sent REINCARNATE code=0x11 for team {team_id}")

    print(f"[UDP] Team {team_id} selected - use /s spawn to spawn")


def handle_spawn_at_point(server: "WulframServer", ctx: "ClientContext", spawn_point_id: int, vehicle_type: int, addr: tuple):
    """Handle spawn at a specific spawn point."""
    print(f"[SPAWN] Client {ctx.client_id}: Spawn at point {spawn_point_id} vehicle={vehicle_type}")

    spawn_positions = {
        5001: (50.0, 10.0, 50.0),
        5002: (60.0, 10.0, 50.0),
        5003: (150.0, 10.0, 150.0),
        5004: (160.0, 10.0, 150.0),
    }

    pos = spawn_positions.get(spawn_point_id, (100.0, 10.0, 100.0))

    if spawn_point_id in (5001, 5002):
        team_id = 1
    elif spawn_point_id in (5003, 5004):
        team_id = 2
    else:
        team_id = ctx.session.team_id or 2

    server._spawn_wf_style(ctx, team_id=team_id, pos=pos)


def send_udp_ack(server: "WulframServer", ctx: Optional["ClientContext"], addr: tuple, packet_id: int, seq_num: int, subcmd: int = 1):
    """Send a Wulf-Forge style UDP ACK (0x02)."""
    if not server.udp_handler:
        return
    if ctx:
        ctx.session.udp_outgoing_seq = (ctx.session.udp_outgoing_seq + 1) & 0xFFFF
        our_seq = ctx.session.udp_outgoing_seq
    else:
        our_seq = 0
    payload = struct.pack(">HHBBH", our_seq, 9, subcmd, packet_id & 0xFF, seq_num & 0xFFFF)
    server.udp_handler.send_to(b'\x02' + payload, addr)


def _spawn_test_projectile(server: "WulframServer", ctx: "ClientContext", addr: tuple):
    """Spawn a test pulse shell projectile via /s fire command."""
    from .weapons import Projectile, EntityType, build_projectile_spawn_packet
    from .packets import get_ticks

    # Get player position (or use default test position)
    pos = ctx.player_pos
    player_id = ctx.session.player_id or ctx.entity_id
    team_id = ctx.session.team_id or 1

    # Generate projectile entity ID
    if not hasattr(server, '_test_projectile_id'):
        server._test_projectile_id = 2000
    server._test_projectile_id += 1

    # Create projectile moving forward (positive X direction)
    proj = Projectile(
        entity_id=server._test_projectile_id,
        entity_type=EntityType.PULSE_SHELL,
        owner_id=player_id,
        team=team_id,
        pos=(pos[0] + 5.0, pos[1], pos[2]),  # Slightly in front of player
        vel=(50.0, 0.0, 0.0),  # Moving forward at 50 units/sec
        spawn_time=time.monotonic(),
        lifetime=5.0
    )

    # Build and send spawn packet
    tick = get_ticks()
    packet = build_projectile_spawn_packet(proj, tick)

    print(f"[TEST] Client {ctx.client_id}: Spawning pulse shell id={proj.entity_id} pos={proj.pos} vel={proj.vel}")
    print(f"[TEST] Packet ({len(packet)} bytes): {packet.hex()}")

    if server.udp_handler and addr:
        server.udp_handler.send_to(packet, addr)
        print(f"[TEST] Sent projectile spawn to {addr}")

        # Also send chat feedback
        msg = build_chat_message("*PEW* (test projectile)", source_id=player_id)
        ctx.tcp_handler.send(msg)
