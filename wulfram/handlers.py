"""
Packet handlers extracted from server.py.
Contains TCP and UDP packet handling logic.
"""

import ipaddress
import os
import struct
import time
import secrets
from typing import TYPE_CHECKING, Optional, Tuple

from .session import Phase, FEATURES
from .packets import (
    PacketType, build_hello_version, build_hello_session_key,
    build_login_status, build_player, build_team_info,
    build_world_stats, build_bps_response, build_chat_message,
    build_add_to_roster, build_update_stats, build_update_stats_team_first, build_tank_packet,
    build_update_array_empty, build_update_array_spawn_points, get_ticks,
    build_behavior_packet, build_translation_packet, build_game_clock,
    build_ping_request,
    build_motd, build_reincarnate,
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


_OG_D_HANDSHAKE_STREAMS = (
    ("Beacon Stream", (0x3A, 0x3B)),
    ("Squad Things", (0x42, 0x46, 0x49, 0x4A)),
)

_OG_D_HANDSHAKE_PRIVATE_MODES = (
    (0x19, 1),
    (0x20, 3),
    (0x25, 3),
    (0x26, 1),
    (0x2B, 1),
    (0x2E, 3),
    (0x33, 3),
    (0x35, 1),
    (0x3A, 3),
    (0x3B, 3),
    (0x42, 3),
    (0x46, 3),
    (0x49, 3),
    (0x4A, 3),
    (0x4F, 3),
)


def _pack_lp_string(text: str) -> bytes:
    raw = (text + "\x00").encode("ascii", errors="ignore")
    return struct.pack(">H", len(raw)) + raw


def _parse_empirical_client_d_handshake(data: bytes) -> Optional[dict]:
    """Parse the OG-style client D_HANDSHAKE seen in live captures."""
    if len(data) < 9 or data[0] != 0x03:
        return None

    try:
        offset = 1
        sequence = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        stream_count = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        if stream_count > 64:
            return None

        streams = []
        for _ in range(stream_count):
            name, offset = decode_lp_string(data, offset)
            if not name or offset + 4 > len(data):
                return None
            index_count = struct.unpack_from(">I", data, offset)[0]
            offset += 4
            if index_count > 256:
                return None
            indices = []
            for _ in range(index_count):
                if offset + 4 > len(data):
                    return None
                indices.append(struct.unpack_from(">I", data, offset)[0])
                offset += 4
            streams.append((name, tuple(indices)))

        if offset + 4 > len(data):
            return None
        private_count = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        if private_count > 256:
            return None

        private_modes = []
        for _ in range(private_count):
            if offset + 8 > len(data):
                return None
            packet_id, delivery_mode = struct.unpack_from(">II", data, offset)
            offset += 8
            private_modes.append((packet_id, delivery_mode))

        if offset != len(data):
            return None

        return {
            "kind": "empirical",
            "sequence": sequence,
            "session_id": 0,
            "streams": tuple(streams),
            "private_modes": tuple(private_modes),
        }
    except struct.error:
        return None


def _parse_legacy_client_d_handshake(data: bytes) -> Optional[dict]:
    """Parse the older simplified Python client D_HANDSHAKE."""
    if len(data) < 13 or data[0] != 0x03:
        return None
    try:
        sequence, session_id, stream_count = struct.unpack_from(">III", data, 1)
    except struct.error:
        return None
    if stream_count > 64:
        return None
    return {
        "kind": "legacy",
        "sequence": sequence,
        "session_id": session_id,
        "streams": (),
        "private_modes": (),
        "stream_count": stream_count,
    }


def _build_server_d_handshake(ctx: Optional["ClientContext"]) -> bytes:
    """Build a server D_HANDSHAKE with empirical OG stream/private mappings."""
    sequence = int(time.monotonic() * 1000) & 0xFFFFFFFF
    session_id = 1
    if ctx is not None:
        session_id = ctx.session.player_id or ctx.client_id or 1

    payload = bytearray()
    payload += struct.pack(">I", sequence)
    payload += struct.pack(">I", session_id)
    payload += struct.pack(">I", len(_OG_D_HANDSHAKE_STREAMS))

    for name, indices in _OG_D_HANDSHAKE_STREAMS:
        payload += _pack_lp_string(name)
        payload += struct.pack(">I", len(indices))
        for channel_id in indices:
            payload += struct.pack(">I", channel_id)

    payload += struct.pack(">I", len(_OG_D_HANDSHAKE_PRIVATE_MODES))
    for packet_id, delivery_mode in _OG_D_HANDSHAKE_PRIVATE_MODES:
        payload += struct.pack(">II", packet_id, delivery_mode)

    return b"\x03" + bytes(payload)


def _is_loopback_client(ctx: "ClientContext") -> bool:
    """Return True when the TCP peer is loopback/local-only."""
    addr = getattr(ctx, "client_addr", None)
    if not addr:
        return False
    host = str(addr[0]).split("%", 1)[0]
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _get_login_bootstrap_mode(ctx: "ClientContext") -> str:
    """Resolve login bootstrap mode.

    Modes:
      - minimal: current Python-client-oriented bootstrap
      - og: capture/decompile-aligned bootstrap for the original client
      - hybrid: minimal for loopback clients, og for remote clients
    """
    mode = os.environ.get("WULFRAM_LOGIN_BOOTSTRAP", "hybrid").strip().lower()
    if mode not in ("minimal", "og", "hybrid"):
        mode = "hybrid"
    if mode == "hybrid":
        return "minimal" if _is_loopback_client(ctx) else "og"
    return mode


def _send_minimal_login_bootstrap(server: "WulframServer", ctx: "ClientContext") -> None:
    """Send the current Python-client-oriented login bootstrap."""
    session = ctx.session
    tcp = ctx.tcp_handler

    tcp.send(build_login_status(8, is_donor=True))

    if FEATURES.send_behavior_packet and not session.behavior_sent:
        tcp.send(build_behavior_packet())
        session.behavior_sent = True

    tcp.send(build_team_info())

    if FEATURES.send_player_on_login and session.player_id == 0:
        session.player_id = ctx.entity_id
        tcp.send(build_player(entity_id=session.player_id, spectator=True))

    if getattr(server, "send_game_clock_on_login", False):
        tcp.send(build_game_clock())

    if getattr(server, "send_request_start_on_login", False):
        tcp.send(build_motd("Welcome to Wulfram!"))

    if getattr(server, "send_roster_on_login", False) and not session.roster_sent:
        name = session.username or f"Player{ctx.client_id}"
        tcp.send(build_add_to_roster(
            player_id=session.player_id,
            entity_id=session.player_id,
            name=name,
            team=session.team_id if session.team_id else 0
        ))
        session.roster_sent = True


def _send_og_login_bootstrap(server: "WulframServer", ctx: "ClientContext") -> None:
    """Send the OG login bootstrap up to the team-select screen.

    Keep this deliberately narrow.  The original client needs the spectator
    PLAYER identity to finish the "processing player map" step, but it runs
    disconnect/map cleanup paths from later gameplay bootstrap packets.
    Sending GAME_CLOCK, roster, translation, or WORLD_STATS before team
    selection can leave it half-spawned or bounce it back to the address
    screen when the user clicks a team.
    """
    session = ctx.session
    tcp = ctx.tcp_handler
    if session.player_id == 0:
        session.player_id = ctx.entity_id

    tcp.send(build_team_info())
    tcp.send(build_login_status(8, is_donor=True))
    tcp.send(build_player(entity_id=session.player_id, spectator=True))


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
        # Subcmd 2 = Session key request — generate unique key per client
        key = f"WS-{ctx.client_id}-{secrets.token_hex(8)}"
        ctx.session.session_key = key
        server.session_key_to_client[key] = ctx
        print(f"[LOGIN] Client {ctx.client_id}: Session key generated: {key}")
        ctx.tcp_handler.send(build_hello_session_key(key))


def handle_login_request(server: "WulframServer", ctx: "ClientContext", packet: bytes):
    """Handle LOGIN_REQUEST packet."""
    if len(packet) < 2:
        return

    session = ctx.session
    if session.login_complete:
        # OG game-service flow: client sends LOGIN_REQUEST after bootstrap.
        # Respond with LOGIN_STATUS(8) so the client knows login succeeded.
        # Without this response, the client times out → Protocol Mismatch.
        print(f"[LOGIN] Client {ctx.client_id}: Late LOGIN_REQUEST after login complete → sending LOGIN_STATUS(8)")
        ctx.tcp_handler.send(build_login_status(8, is_donor=True))
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
    """Send packets needed for team selection screen.

    Bootstrap mode is selected by WULFRAM_LOGIN_BOOTSTRAP:
      - minimal: current Python-client path
      - og: empirical/decompile-aligned original-client path
      - hybrid: minimal for loopback, og for remote peers
    """
    mode = _get_login_bootstrap_mode(ctx)
    print(f"[LOGIN] Client {ctx.client_id}: bootstrap={mode} addr={ctx.client_addr[0]}")
    if mode == "og":
        _send_og_login_bootstrap(server, ctx)
    else:
        _send_minimal_login_bootstrap(server, ctx)


def _broadcast_update_stats(server: "WulframServer", account_id: int, team_id: int) -> int:
    """Broadcast UPDATE_STATS to all connected clients with a known transport."""
    packet = build_update_stats(player_id=account_id, entity_id=account_id, team_id=team_id)
    sent = 0
    lock = getattr(server, "clients_lock", None)
    clients = getattr(server, "clients", {})
    if lock is not None:
        with lock:
            targets = list(clients.values())
    else:
        targets = list(clients.values())

    for target in targets:
        try:
            udp_addr = getattr(target.session, "udp_addr", None)
            if server.udp_handler and udp_addr:
                server.udp_handler.send_to(packet, udp_addr)
                sent += 1
            elif target.tcp_handler:
                target.tcp_handler.send(packet)
                sent += 1
        except Exception as ex:
            print(
                f"[STATS] Broadcast UPDATE_STATS failed for "
                f"client {getattr(target, 'client_id', '?')}: {ex}"
            )
    return sent


def _safe_tcp_send(ctx: "ClientContext", payload: bytes, label: str = "") -> bool:
    """
    Best-effort TCP send for paths that may run from UDP/tick threads.
    Returns False when the client socket is gone instead of raising.
    """
    tcp = getattr(ctx, "tcp_handler", None)
    if not tcp:
        return False
    sock = getattr(tcp, "sock", None)
    try:
        if sock is None or sock.fileno() < 0:
            print(f"[TCP-SEND] Skip {label or 'packet'}: socket unavailable for client {ctx.client_id}")
            return False
    except Exception:
        print(f"[TCP-SEND] Skip {label or 'packet'}: socket state check failed for client {ctx.client_id}")
        return False

    try:
        tcp.send(payload)
        return True
    except Exception as ex:
        print(f"[TCP-SEND] Failed {label or 'packet'} for client {ctx.client_id}: {ex}")
        return False


def _schedule_team_select_spawn(server: "WulframServer", ctx: "ClientContext", team_id: int, reason: str) -> None:
    """Schedule/queue auto-spawn for legacy team-select flow."""
    session = ctx.session
    if not getattr(server, "spawn_on_team_select", False):
        session.pending_spawn_team_id = 0
        return

    now = time.monotonic()
    wants_updates = session.want_updates_received or session.want_updates_handled
    if wants_updates:
        session.pending_spawn_team_id = 0
        session.delayed_spawn_team = team_id
        session.delayed_spawn_time = now + server.spawn_delay_seconds
        print(
            f"[SPAWN] Client {ctx.client_id}: Scheduled delayed spawn in "
            f"{server.spawn_delay_seconds:.1f}s for team {team_id} ({reason})"
        )
        return

    session.pending_spawn_team_id = team_id
    print(
        f"[SPAWN] Client {ctx.client_id}: Deferred spawn for team {team_id} "
        f"until WANT_UPDATES ({reason})"
    )


def handle_bps(server: "WulframServer", ctx: "ClientContext", packet: bytes):
    """Handle BPS (bandwidth/rate) packet."""
    session = ctx.session
    tcp = ctx.tcp_handler

    if len(packet) >= 5:
        rate_index = struct.unpack(">I", packet[1:5])[0]
        print(f"[GAME] Client {ctx.client_id}: BPS request: rate={rate_index}")
        tcp.send(build_bps_response(rate_index, approved=True))

        if _get_login_bootstrap_mode(ctx) == "og" and not session.bootstrap_ping_sent:
            tcp.send(build_ping_request(get_ticks()))
            session.bootstrap_ping_sent = True

        if session.phase != Phase.TEAM_SELECT:
            return

        # Canonical flow does not require WORLD_STATS during team select.
        # Keep this as an explicit fallback only.
        if FEATURES.send_world_stats_on_login and not session.world_stats_sent:
            time.sleep(0.15)
            print(f"[GAME] Client {ctx.client_id}: Sending WORLD_STATS after BPS request")
            tcp.send(server.build_world_stats_packet())
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

    if os.environ.get("WULFRAM_SEND_WELCOME_CHAT_ON_WANT_UPDATES", "0").strip().lower() in ("1", "true", "on", "yes"):
        tcp.send(build_chat_message("System: Welcome to Wulfram!"))

    # NOTE: VIEW_UPDATE initial snapshot disabled over TCP — OG client's
    # undecompiled handler causes TCP stream desync -> MAX_STREAM_DATA crash.
    # Re-enable when OG handler wire format is verified via Ghidra.

    # OG-safe team-select bootstrap: send config once after WANT_UPDATES, then
    # prime UPDATE_ARRAY and wait for TRANSLATION_ACK before spawn points.
    if FEATURES.send_behavior_packet and not session.behavior_sent:
        tcp.send(build_behavior_packet())
        session.behavior_sent = True

    if FEATURES.send_translation_packet and not session.translation_sent:
        tcp.send(build_translation_packet())
        session.translation_sent = True

    # Remote OG clients are fragile around stray UPDATE_ARRAY packets on the
    # TCP stream during WANT_UPDATES/bootstrap. Keep the legacy empty priming
    # packet only for loopback probes; remote clients continue over UDP once
    # bootstrap is actually ready.
    if _is_loopback_client(ctx):
        tcp.send(build_update_array_empty())
    else:
        print(
            f"[GAME] Client {ctx.client_id}: Suppressing empty TCP UPDATE_ARRAY "
            "during WANT_UPDATES for remote bootstrap"
        )

    # Spawn points are deferred until the client finishes UDP bootstrap AND has
    # acknowledged TRANSLATION so quantizers are definitely active.
    if FEATURES.send_spawn_points:
        if session.udp_d_handshake_received and session.translation_ack_received:
            _send_spawn_points_for_client(server, ctx)
        else:
            session.pending_spawn_points = True
            print(
                f"[GAME] Client {ctx.client_id}: Deferring spawn points until "
                "UDP D_HANDSHAKE + TRANSLATION_ACK complete"
            )

    # Optional legacy path: spawn directly from team-select state.
    if session.pending_spawn_team_id:
        if getattr(server, "spawn_on_team_select", False):
            team = session.pending_spawn_team_id
            session.pending_spawn_team_id = 0
            session.delayed_spawn_time = now + server.spawn_delay_seconds
            session.delayed_spawn_team = team
            print(
                f"[GAME] Client {ctx.client_id}: Scheduled spawn in "
                f"{server.spawn_delay_seconds:.1f}s for team {team} (team select)"
            )
            session.want_updates_handled = True
            session.want_updates_handled_time = now
            return
        print(
            f"[GAME] Client {ctx.client_id}: pending team-select spawn ignored "
            "(spawn_on_team_select=0)"
        )
        session.pending_spawn_team_id = 0

    session.want_updates_handled = True
    session.want_updates_handled_time = now

    # Auto-join team (alternate between team 1 and 2 for multiplayer)
    if FEATURES.auto_join_team:
        # Alternate by client_id: odd=team1, even=team2
        team = 1 + ((ctx.client_id - 1) % 2)  # client 1 -> team 1, client 2 -> team 2
        spawn_delay_seconds = getattr(server, "spawn_delay_seconds", 6.0)
        session.delayed_spawn_time = now + spawn_delay_seconds
        session.delayed_spawn_team = team
        print(
            f"[GAME] Client {ctx.client_id}: Scheduled auto-spawn in "
            f"{spawn_delay_seconds:.1f}s for team {team}"
        )
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

    session.team_id = team_id
    if session.player_id == 0:
        session.player_id = ctx.entity_id
    _schedule_team_select_spawn(server, ctx, team_id, reason="tcp_reincarnate")
    _send_team_switch_roster(server, ctx, team_id)
    _send_team_switch_update_stats(server, ctx, team_id)

    # Mirror wulf-forge: acknowledge team switch/spawn intent with REINCARNATE code 0x11.
    if getattr(server, "team_switch_send_reincarnate", True):
        tcp.send(build_reincarnate(0x11, ""))

    _send_post_reincarnate_entry_packets(server, ctx)


def _send_post_reincarnate_entry_packets(server: "WulframServer", ctx: "ClientContext") -> None:
    """Send the canonical team-entry packets after REINCARNATE."""
    session = ctx.session
    if session.player_id == 0:
        session.player_id = ctx.entity_id
    if not getattr(server, "team_switch_send_entry_packets", True):
        print(
            f"[GAME] Client {ctx.client_id}: Post-team-switch entry packets disabled "
            "(WULFRAM_TEAM_SWITCH_ENTRY_PACKETS=0)"
        )
        return

    _safe_tcp_send(
        ctx,
        build_player(entity_id=session.player_id, spectator=False),
        label="post_reincarnate_player",
    )
    _safe_tcp_send(ctx, build_game_clock(), label="post_reincarnate_game_clock")

    if not session.roster_sent:
        name = session.username or f"Player{ctx.client_id}"
        _safe_tcp_send(
            ctx,
            build_add_to_roster(
                player_id=session.player_id,
                entity_id=session.player_id,
                name=name,
                team=session.team_id if session.team_id else 0,
            ),
            label="post_reincarnate_add_to_roster",
        )
        session.roster_sent = True

    if not session.world_stats_sent:
        _safe_tcp_send(
            ctx,
            server.build_world_stats_packet(),
            label="post_reincarnate_world_stats",
        )
        session.world_stats_sent = True


def _send_team_switch_roster(
    server: "WulframServer",
    ctx: "ClientContext",
    team_id: int,
) -> None:
    """Optionally reassert the local player's roster entry after team select."""
    if not getattr(server, "team_switch_send_roster", False):
        return
    session = ctx.session
    player_id = session.player_id or ctx.entity_id
    name = session.username or f"Player{ctx.client_id}"
    if _safe_tcp_send(
        ctx,
        build_add_to_roster(
            player_id=player_id,
            entity_id=player_id,
            name=name,
            team=team_id,
        ),
        label="team_switch_add_to_roster",
    ):
        session.roster_sent = True
        print(f"[TCP] Team {team_id} roster reasserted for client {ctx.client_id}")


def _send_team_switch_update_stats(
    server: "WulframServer",
    ctx: "ClientContext",
    team_id: int,
    addr: Optional[tuple] = None,
) -> None:
    """Send the roster stats/team update that OG sees during team switch."""
    if not getattr(server, "team_switch_send_update_stats", True):
        return

    session = ctx.session
    player_id = session.player_id or ctx.entity_id
    variant = getattr(server, "team_switch_update_stats_variant", "canonical")
    if variant == "team_first":
        packet = build_update_stats_team_first(
            player_id=player_id,
            entity_id=player_id,
            team_id=team_id,
        )
    else:
        packet = build_update_stats(player_id=player_id, entity_id=player_id, team_id=team_id)
    target_addr = addr or session.udp_addr
    transport = getattr(server, "team_switch_update_stats_transport", "udp")

    if transport in ("udp", "auto") and server.udp_handler and target_addr:
        try:
            server.udp_handler.send_to(packet, target_addr)
            print(f"[UDP] Team {team_id} UPDATE_STATS sent to {target_addr}")
            return
        except Exception as ex:
            print(f"[UDP] Failed to send team-switch UPDATE_STATS: {ex}")
            if transport == "udp":
                return

    if transport in ("tcp", "auto") and ctx.tcp_handler and _safe_tcp_send(
        ctx,
        packet,
        label="team_switch_update_stats",
    ):
        print(f"[TCP] Team {team_id} UPDATE_STATS sent (fallback)")


# ============ UDP Handlers ============

def handle_udp_d_handshake(server: "WulframServer", ctx: Optional["ClientContext"], data: bytes, addr: tuple):
    """Handle UDP D_HANDSHAKE (0x03) and respond with OG-shaped stream metadata."""
    parsed = _parse_empirical_client_d_handshake(data)
    if parsed is None:
        parsed = _parse_legacy_client_d_handshake(data)
    if parsed is None:
        print(f"[UDP] D_HANDSHAKE malformed from {addr}: len={len(data)} data={data.hex()}")
        return

    stream_count = len(parsed["streams"]) if parsed["streams"] else parsed.get("stream_count", 0)
    print(
        f"[UDP] D_HANDSHAKE[{parsed['kind']}] seq={parsed['sequence']} "
        f"id={parsed['session_id']} streams={stream_count} "
        f"private={len(parsed['private_modes'])}"
    )

    # If ctx is None, allow only a unique, safe recovery candidate.
    if ctx is None:
        ctx = server._recover_udp_client(addr, allow_handshake=True)

    if ctx:
        ctx.session.udp_d_handshake_received = True
        ctx.session.udp_verified = True
        server._bind_udp_client(ctx, addr, reason="d_handshake")
        print(f"[UDP] Registered client {ctx.client_id} with UDP addr {addr}")

    # ACK
    ack = b'\x02' + b'\x00' + struct.pack(">I", int(time.monotonic() * 1000) & 0xFFFFFFFF)
    server.udp_handler.send_to(ack, addr)
    print(f"[UDP SEND] D_ACK to {addr}")

    server_handshake = _build_server_d_handshake(ctx)
    server.udp_handler.send_to(server_handshake, addr)
    print(
        f"[UDP SEND] D_HANDSHAKE seq={struct.unpack('>I', server_handshake[1:5])[0]} "
        f"to {addr}"
    )

    if (
        ctx
        and ctx.session.pending_spawn_points
        and not ctx.session.spawn_points_sent
        and ctx.session.translation_ack_received
    ):
        _send_spawn_points_for_client(server, ctx)


def _send_spawn_points_for_client(server: "WulframServer", ctx: "ClientContext") -> None:
    session = ctx.session
    if session.spawn_points_sent:
        return
    if not session.translation_ack_received:
        session.pending_spawn_points = True
        print(f"[GAME] Client {ctx.client_id}: Waiting for TRANSLATION_ACK before spawn points")
        return

    spawn_points = server.get_spawn_points()
    if hasattr(server, "_to_client_pos"):
        for sp in spawn_points:
            x, y, z = server._to_client_pos((sp["x"], sp["y"], sp["z"]))
            sp["x"], sp["y"], sp["z"] = x, y, z

    print(f"[GAME] Client {ctx.client_id}: Sending {len(spawn_points)} spawn points")
    tick = get_ticks()
    spawn_pkt = build_update_array_spawn_points(tick, spawn_points)
    spawn_transport = os.environ.get("WULFRAM_SPAWN_POINTS_TRANSPORT", "tcp").strip().lower()
    # Remote OG clients are far more sensitive to UPDATE_ARRAY over TCP than
    # loopback Python probes. We already defer spawn points until UDP bootstrap
    # is complete, so prefer the UDP path for non-loopback clients.
    if not _is_loopback_client(ctx) and spawn_transport in ("tcp", "both"):
        spawn_transport = "udp"

    if spawn_transport in ("tcp", "both"):
        ctx.tcp_handler.send(spawn_pkt)
    if spawn_transport in ("udp", "both"):
        if server.udp_handler and session.udp_addr:
            server.udp_handler.send_to(spawn_pkt, session.udp_addr)
        else:
            print(
                f"[GAME] Client {ctx.client_id}: UDP not ready, keeping "
                "spawn-point UPDATE_ARRAY deferred"
            )
            session.pending_spawn_points = True
            return

    session.pending_spawn_points = False
    session.spawn_points_sent = True


def recent_control_pose_spawn_block(ctx: "ClientContext", now: Optional[float] = None) -> dict:
    """Return whether a spawn path should be blocked after a control pose reset."""
    try:
        block_s = float(os.environ.get("WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S", "45.0"))
    except (TypeError, ValueError):
        block_s = 45.0
    reset_time = float(getattr(ctx, "control_pose_reset_time", 0.0) or 0.0)
    now = time.monotonic() if now is None else float(now)
    age_s = now - reset_time if reset_time > 0.0 else None
    blocked = bool(block_s > 0.0 and age_s is not None and age_s < block_s)
    return {
        "blocked": blocked,
        "age_s": age_s,
        "block_s": block_s,
        "reset_pos": list(getattr(ctx, "control_pose_reset_pos", []) or []),
    }


def handle_udp_chat(server: "WulframServer", ctx: Optional["ClientContext"], data: bytes, addr: tuple):
    """Handle UDP COMM_REQ (0x20) for /s spawn and uplink commands."""
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

    comm_handler = getattr(server, "_handle_comm_message_request", None)
    if callable(comm_handler):
        event = comm_handler(
            ctx,
            data,
            transport="udp",
            body=data[5:],
            addr=addr,
            sequence=seq_num,
        )
        if event.get("handled"):
            return

    cmd = msg.strip()
    if cmd.lower().startswith("/s "):
        cmd = cmd[3:].strip()

    if cmd.lower() == "spawn":
        net_id = ctx.session.player_id or ctx.entity_id
        team_id = ctx.session.team_id or 1
        # Align with wulf-forge /s spawn behavior: fixed debug spawn, no map-spawn lookup.
        if server.up_axis == "z":
            spawn_pos = (100.0, 100.0, server.spawn_height)
        else:
            spawn_pos = (100.0, server.spawn_height, 100.0)
        server._spawn_wf_style(ctx, team_id=team_id, net_id=net_id, pos=spawn_pos)

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
        # Type 0: explicit spawn request. Different clients/builds can append
        # extra fields, so decode conservatively and resolve the spawn point
        # from any known spawn oid in the payload words.
        words = []
        for off in range(1, len(payload), 4):
            if off + 4 > len(payload):
                break
            words.append(struct.unpack(">I", payload[off:off + 4])[0])

        spawn_points = server.get_spawn_points()
        spawn_ids = {int(sp.get("oid", 0)) for sp in spawn_points}
        spawn_point_id = 0
        vehicle_type = 0
        team_hint = ctx.session.team_id or 1

        if words:
            if len(words) >= 2:
                first, second = words[0], words[1]
                if first in spawn_ids:
                    spawn_point_id = first
                    vehicle_type = second
                elif second in spawn_ids:
                    # Some variants place unit/vehicle id first and spawn oid second.
                    spawn_point_id = second
                    vehicle_type = first
                if first in (1, 2):
                    team_hint = first
            if not spawn_point_id:
                for w in words:
                    if w in spawn_ids:
                        spawn_point_id = w
                        break
            if not spawn_point_id and len(words) > 1 and words[1] in (1, 2):
                team_hint = words[1]
            if not spawn_point_id and len(words) >= 2:
                vehicle_type = words[1]

        if not spawn_point_id and spawn_points:
            # Last-resort fallback: pick a spawn point for the current/hinted team.
            picked = next((sp for sp in spawn_points if sp.get("team") == team_hint), spawn_points[0])
            spawn_point_id = int(picked.get("oid", 0))
            print(
                f"[UDP] REINCARNATE (spawn) seq={seq_num} words={words} "
                f"fallback_spawn_point={spawn_point_id} team_hint={team_hint}"
            )

        if spawn_point_id:
            print(
                f"[UDP] REINCARNATE (spawn) seq={seq_num} "
                f"spawn_point={spawn_point_id} vehicle={vehicle_type} words={words}"
            )
            handle_spawn_at_point(server, ctx, spawn_point_id, vehicle_type, addr)
        else:
            print(
                f"[UDP] REINCARNATE (spawn) seq={seq_num} "
                f"unable_to_resolve_spawn words={words} payload_hex={payload.hex()}"
            )

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
    if session.player_id == 0:
        session.player_id = ctx.entity_id
    _schedule_team_select_spawn(server, ctx, team_id, reason="udp_team_switch")
    _send_team_switch_roster(server, ctx, team_id)
    _send_team_switch_update_stats(server, ctx, team_id, addr)

    # Wulf-forge-style ACK: REINCARNATE(code=17) should be seen on UDP for
    # reliable entry-map -> world transition. Fall back to TCP only when UDP
    # address is not yet known.
    if getattr(server, "team_switch_send_reincarnate", True):
        rein = build_reincarnate(0x11, "")
        sent_rein = False
        if server.udp_handler and addr:
            try:
                server.udp_handler.send_to(rein, addr)
                sent_rein = True
                print(f"[UDP] Team {team_id} switch acked with REINCARNATE 0x11 (UDP)")
            except Exception as ex:
                print(f"[UDP] Failed to send team-switch REINCARNATE over UDP: {ex}")
        if not sent_rein and ctx.tcp_handler:
            if _safe_tcp_send(ctx, rein, label="team_switch_reincarnate_ack"):
                print(f"[TCP] Team {team_id} switch acked with REINCARNATE 0x11 (fallback)")

    _send_post_reincarnate_entry_packets(server, ctx)

    if getattr(server, "spawn_on_team_select", False):
        print(
            f"[UDP] Team {team_id} selected - spawn_on_team_select=1 "
            "(legacy auto-spawn path)"
        )
    else:
        # Explicit spawn-point flow: if the client never sends subtype-0 REINCARNATE,
        # schedule a bounded fallback spawn to avoid getting stuck in entry-map UI.
        force_after = float(getattr(server, "spawn_force_after", 0.0) or 0.0)
        if force_after > 0.0:
            now = time.monotonic()
            base = session.want_updates_time if session.want_updates_time > 0.0 else now
            session.delayed_spawn_team = team_id
            session.delayed_spawn_time = max(base + force_after, now + 0.5)
            session.spawn_wait_logged = False
            wait_s = max(0.0, session.delayed_spawn_time - now)
            anchor = "WANT_UPDATES" if session.want_updates_time > 0.0 else "team-switch"
            print(
                f"[UDP] Team {team_id} explicit-spawn fallback scheduled in "
                f"{wait_s:.1f}s (anchor={anchor})"
            )
        print(
            f"[UDP] Team {team_id} selected - waiting for explicit spawn-point packet"
        )


def handle_spawn_at_point(server: "WulframServer", ctx: "ClientContext", spawn_point_id: int, vehicle_type: int, addr: tuple):
    """Handle spawn at a specific spawn point."""
    session = ctx.session
    now = time.monotonic()
    in_game = session.in_game or session.phase == Phase.IN_GAME
    spawn_override = False

    # Some clients can remain on entry-map UI after auto-spawn and then send an
    # explicit spawn-point click. Allow a guarded override for that case.
    if in_game:
        if not getattr(server, "spawn_allow_point_override", False):
            print(
                f"[SPAWN] Client {ctx.client_id}: Ignoring duplicate spawn request "
                f"while IN_GAME (point={spawn_point_id} vehicle={vehicle_type})"
            )
            _safe_tcp_send(ctx, build_reincarnate(0x11, "Already spawned"), label="spawn_duplicate_reincarnate")
            _safe_tcp_send(
                ctx,
                build_player(session.player_id or ctx.entity_id, spectator=False),
                label="spawn_duplicate_player_active",
            )
            return

        control_pose_block = recent_control_pose_spawn_block(ctx, now=now)
        if control_pose_block["blocked"]:
            print(
                f"[SPAWN] Client {ctx.client_id}: Ignoring IN_GAME spawn override "
                f"after recent control pose reset "
                f"(age={control_pose_block['age_s']:.2f}s < block={control_pose_block['block_s']:.2f}s) "
                f"point={spawn_point_id} vehicle={vehicle_type}"
            )
            return

        try:
            min_interval = float(getattr(server, "spawn_point_override_min_interval", 0.0))
        except (TypeError, ValueError):
            min_interval = 0.0
        elapsed = now - float(session.last_spawn_time or 0.0)
        if min_interval > 0.0 and elapsed < min_interval:
            print(
                f"[SPAWN] Client {ctx.client_id}: Ignoring IN_GAME spawn override "
                f"(elapsed={elapsed:.2f}s < min={min_interval:.2f}s) "
                f"point={spawn_point_id} vehicle={vehicle_type}"
            )
            return

        spawn_override = True
        print(
            f"[SPAWN] Client {ctx.client_id}: Applying IN_GAME spawn-point override "
            f"(point={spawn_point_id} vehicle={vehicle_type}, elapsed={elapsed:.2f}s)"
        )

    # If a spawn is already being processed, ignore retries to avoid re-entry races.
    if session.phase == Phase.SPAWNING and not spawn_override:
        print(
            f"[SPAWN] Client {ctx.client_id}: Ignoring duplicate spawn request "
            f"while SPAWNING (point={spawn_point_id} vehicle={vehicle_type})"
        )
        return

    # Explicit spawn-point packet won; cancel any delayed fallback spawn from team-switch.
    session.pending_spawn_team_id = 0
    session.delayed_spawn_team = 0
    session.delayed_spawn_time = 0.0
    session.spawn_wait_logged = False

    if not spawn_override:
        session.transition_to(Phase.SPAWNING)
    print(f"[SPAWN] Client {ctx.client_id}: Spawn at point {spawn_point_id} vehicle={vehicle_type}")
    spawn_points = server.get_spawn_points()
    selected = None
    for sp in spawn_points:
        if sp.get("oid") == spawn_point_id:
            selected = sp
            break

    if selected:
        team_id = selected.get("team", ctx.session.team_id or 2)
        requested_pos = (selected["x"], selected["y"], selected["z"])
    else:
        team_id = ctx.session.team_id or 2
        requested_pos = None

    if requested_pos is not None:
        pos = server._resolve_spawn_pos(team_id, explicit_pos=requested_pos)
        print(
            f"[SPAWN] Client {ctx.client_id}: honoring spawn-point {spawn_point_id} "
            f"pos={pos}"
        )
    elif server._get_configured_default_spawn_pos() is not None:
        pos = server._resolve_spawn_pos(team_id)
        print(
            f"[SPAWN] Client {ctx.client_id}: overriding spawn-point {spawn_point_id} "
            f"with default flat spawn pos={pos}"
        )
    else:
        pos = server._resolve_spawn_pos(team_id, explicit_pos=requested_pos)

    server._spawn_wf_style(ctx, team_id=team_id, pos=pos)


def send_udp_ack(server: "WulframServer", ctx: Optional["ClientContext"], addr: tuple, packet_id: int, seq_num: int, subcmd: int = 1):
    """Send an OG-shaped UDP D_ACK (0x02)."""
    if not server.udp_handler:
        return
    if subcmd == 1:
        payload = bytes((0x02, 0x01, packet_id & 0xFF)) + struct.pack(">H", seq_num & 0xFFFF)
    elif subcmd == 2:
        timestamp = int(time.monotonic() * 1000) & 0xFFFFFFFF
        payload = bytes((0x02, 0x02)) + struct.pack(">I", timestamp) + bytes((packet_id & 0xFF,)) + struct.pack(">H", seq_num & 0xFFFF)
    else:
        timestamp = int(time.monotonic() * 1000) & 0xFFFFFFFF
        payload = bytes((0x02, 0x00)) + struct.pack(">I", timestamp)
    server.udp_handler.send_to(payload, addr)


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
