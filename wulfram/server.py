"""
Main server: Orchestrates the protocol flow using layered components.

NOTE: Use manage_server.py to start/stop the server instead of running this directly.
This avoids orphaned processes and provides clean shutdown handling.

    python server/manage_server.py start
    python server/manage_server.py stop
    python server/manage_server.py restart
"""

import math
import os
import socket
import struct
import threading
import time
from typing import Optional, Dict

from .session import Session, Phase, FEATURES
from .transport import TCPHandler, UDPHandler, PacketLogger, print_packet
from .codec import BitReader
from .control import ControlServer
from .weapons import WeaponSystem, build_projectile_spawn_packet, EntityType, BehaviorSlot
from .jump_jets import JumpJetSystem
from .client import ClientContext
from .packets import (
    PacketType, get_packet_name, get_ticks,
    build_hello_udp_config, build_hello_verified,
    build_identified_udp, build_login_status, build_tank_packet,
    build_udp_tank_packet_wf, build_update_array_heartbeat,
    build_chat_message, build_add_to_roster, build_player_info,
    build_birth_notice, build_game_clock, build_reincarnate,
    build_update_array_create_tank, build_update_array_player_update,
    get_behavior_weapon_capability_counts,
)
from . import handlers


class WulframServer:
    """
    Wulfram2 game server emulator with multi-client support.

    Each client runs in its own thread with its own ClientContext.
    The UDP handler is shared across all clients.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 2627):
        self.host = host
        self.port = port
        self.logger = PacketLogger()
        self.udp_handler: Optional[UDPHandler] = None
        self.running = False
        self.control_server = ControlServer(port=port + 1)  # Control on port+1 (2628)

        # Multi-client management
        self.clients: Dict[int, ClientContext] = {}
        self.clients_lock = threading.Lock()
        self.next_client_id = 1
        self.next_entity_id = 1000

        # UDP address to client mapping for packet routing
        self.udp_addr_to_client: Dict[tuple, ClientContext] = {}

        # Coordinate system config (defaults to z-up).
        self.up_axis = os.environ.get("WULFRAM_UP_AXIS", "z").lower()
        if self.up_axis not in ("y", "z"):
            self.up_axis = "z"
        try:
            self.spawn_height = float(os.environ.get("WULFRAM_SPAWN_HEIGHT", "20.0"))
        except ValueError:
            self.spawn_height = 20.0
        # Client/world offset for position alignment (default to 0 for pure server-space).
        if self.up_axis == "z":
            try:
                self.pos_offset = float(os.environ.get("WULFRAM_POS_OFFSET_Z", "0.0"))
            except ValueError:
                self.pos_offset = 0.0
        else:
            try:
                self.pos_offset = float(os.environ.get("WULFRAM_POS_OFFSET_Y", "0.0"))
            except ValueError:
                self.pos_offset = 0.0
        # Server-authoritative physics: send position/velocity updates to client
        # Goal: Server controls movement, client renders server state
        # NOTE: Tick loop causes client input freeze - disabled for now (see _spawn_wf_style)
        self.send_player_updates = os.environ.get("WULFRAM_SEND_PLAYER_UPDATES", "1") == "1"
        self.send_updates_tcp = os.environ.get("WULFRAM_UPDATE_TCP", "1") == "1"
        self.send_updates_udp = os.environ.get("WULFRAM_UPDATE_UDP", "1") == "1"
        # Local stats in projectile packets can desync the UPDATE_ARRAY bitstream if ammo/turret bits are missing.
        # Default to disabled unless explicitly enabled.
        self.projectile_local_stats = os.environ.get("WULFRAM_PROJECTILE_LOCAL_STATS", "0") == "1"
        self.debug_projectiles = (
            os.environ.get("WULFRAM_DEBUG_PROJECTILES", "0") == "1"
            or os.environ.get("WULFRAM_DEBUG_AIM", "0") == "1"
        )
        try:
            self.projectile_config = int(os.environ.get("WULFRAM_PROJECTILE_CONFIG", "0"))
        except ValueError:
            self.projectile_config = 0
        self.projectile_static = os.environ.get("WULFRAM_PROJECTILE_STATIC", "0") == "1"
        try:
            self.viewpoint_timeout = float(os.environ.get("WULFRAM_VIEWPOINT_TIMEOUT", "1.0"))
        except ValueError:
            self.viewpoint_timeout = 1.0
        try:
            self.multi_spawn_offset = float(os.environ.get("WULFRAM_MULTI_SPAWN_OFFSET", "120.0"))
        except ValueError:
            self.multi_spawn_offset = 120.0
        try:
            self.weapon_id = int(os.environ.get("WULFRAM_WEAPON_ID", "0"))
        except ValueError:
            self.weapon_id = 0
        if self.weapon_id < 0 or self.weapon_id > 31:
            print(f"[WEAPON] Invalid weapon_id={self.weapon_id}, defaulting to 0")
            self.weapon_id = 0
        self.projectile_aim_source = os.environ.get("WULFRAM_PROJECTILE_AIM_SOURCE", "auto").lower()
        # TankPacket vitals are REQUIRED to keep the HUD health/fuel stable.
        # If disabled, the client drifts into red-screen health after ~5-10s.
        # Only turn off when UPDATE_ARRAY local-state is proven stable long-term.
        self.tank_vitals = os.environ.get("WULFRAM_TANK_VITALS", "1") == "1"
        # Periodically refresh TankPacket vitals to keep health stable in multi-client.
        self.tank_vitals_heartbeat = os.environ.get("WULFRAM_TANK_VITALS_HEARTBEAT", "1") == "1"
        try:
            self.tank_vitals_interval = float(os.environ.get("WULFRAM_TANK_VITALS_INTERVAL", "1.0"))
        except ValueError:
            self.tank_vitals_interval = 1.0
        # Local-player state in UPDATE_ARRAY has caused HUD/health issues in multi-client.
        # Keep it opt-in; rely on TankPacket vitals heartbeat by default.
        self.update_local_state = os.environ.get("WULFRAM_UPDATE_LOCAL_STATE", "0") == "1"
        self.player_info_local_state = os.environ.get("WULFRAM_PLAYER_INFO_LOCAL_STATE", "0") == "1"
        # Local-state weapon type (entity type index). Default is 0 (tank).
        # Override if needed for testing.
        try:
            self.local_state_weapon_type = int(os.environ.get("WULFRAM_LOCAL_STATE_WEAPON_TYPE", "0"))
        except ValueError:
            self.local_state_weapon_type = 0
        # Local-state ammo bitmask parameters (active slot flags). Defaults match our BEHAVIOR config (no active flags).
        try:
            self.local_state_ammo_bits = int(os.environ.get("WULFRAM_LOCAL_STATE_AMMO_BITS", "0"))
        except ValueError:
            self.local_state_ammo_bits = 0
        try:
            self.local_state_ammo_mask = int(os.environ.get("WULFRAM_LOCAL_STATE_AMMO_MASK", "0"))
        except ValueError:
            self.local_state_ammo_mask = 0
        self.local_state_ammo_from_behavior = os.environ.get("WULFRAM_LOCAL_STATE_AMMO_FROM_BEHAVIOR", "0") == "1"
        self.local_state_ammo_override = (
            "WULFRAM_LOCAL_STATE_AMMO_BITS" in os.environ
            or "WULFRAM_LOCAL_STATE_AMMO_MASK" in os.environ
        )
        # Turret angle bits for local state (if weapon def flags require them).
        try:
            self.local_state_turret_bits = int(os.environ.get("WULFRAM_LOCAL_STATE_TURRET_BITS", "16"))
        except ValueError:
            self.local_state_turret_bits = 16
        try:
            self.local_state_turret_max = float(os.environ.get("WULFRAM_LOCAL_STATE_TURRET_MAX", "6.3"))
        except ValueError:
            self.local_state_turret_max = 6.3
        try:
            self.local_state_turret_range = float(os.environ.get("WULFRAM_LOCAL_STATE_TURRET_RANGE", "12.6"))
        except ValueError:
            self.local_state_turret_range = 12.6
        self.local_state_primary_override = os.environ.get("WULFRAM_LOCAL_STATE_PRIMARY_TURRET", "").strip()
        self.local_state_secondary_override = os.environ.get("WULFRAM_LOCAL_STATE_SECONDARY_TURRET", "").strip()

        # Derive per-weapon capability counts from the BEHAVIOR packet.
        # Used to size local-state ammo/active-slot bitmasks correctly.
        self.behavior_weapon_caps = get_behavior_weapon_capability_counts()
        if not self.behavior_weapon_caps:
            self.behavior_weapon_caps = [(0, 0, 0, 0)] * 4
        if self.update_local_state:
            print(f"[LOCAL-STATE] Weapon capability counts (ammo/fire/active/cooldown): {self.behavior_weapon_caps}")
        if not self.tank_vitals:
            print("[WARN] WULFRAM_TANK_VITALS=0 can cause red health overlay after ~5-10s.")
        print(
            "[CONFIG] update_local_state="
            f"{int(self.update_local_state)} player_info_local_state={int(self.player_info_local_state)} "
            f"tank_vitals={int(self.tank_vitals)} local_state_weapon_type={self.local_state_weapon_type} "
            f"ammo_from_behavior={int(self.local_state_ammo_from_behavior)} "
            f"vitals_heartbeat={int(self.tank_vitals_heartbeat)}"
        )

        # Aim/movement configuration (shared across clients)
        self.use_slot_aim = os.environ.get("WULFRAM_USE_SLOT_AIM", "0") == "1"
        try:
            self.aim_turn_adjust = float(os.environ.get("WULFRAM_AIM_TURN_ADJUST", "4.5"))
        except ValueError:
            self.aim_turn_adjust = 4.5
        try:
            self.aim_pitch_adjust = float(os.environ.get("WULFRAM_AIM_PITCH_ADJUST", "2.5"))
        except ValueError:
            self.aim_pitch_adjust = 2.5
        try:
            self.turn_adjust = float(os.environ.get("WULFRAM_TURN_ADJUST", "4.5"))
        except ValueError:
            self.turn_adjust = 4.5
        try:
            self.turn_deadzone = float(os.environ.get("WULFRAM_TURN_DEADZONE", "0.05"))
        except ValueError:
            self.turn_deadzone = 0.05
        try:
            self.aim_hold_time = float(os.environ.get("WULFRAM_AIM_HOLD", "0.4"))
        except ValueError:
            self.aim_hold_time = 0.4

        self.estimated_speed = 15.0  # Units per second (tunable)

    def _sync_tick_offset(self, ctx: ClientContext, client_tick: int) -> None:
        """Align server ticks to client tick domain for UPDATE_ARRAY gating."""
        if client_tick <= 0:
            return
        server_tick = get_ticks()
        new_offset = client_tick - server_tick
        if ctx.tick_offset is None or abs(new_offset - ctx.tick_offset) > 5000:
            print(f"[TICK] Sync tick offset: client={client_tick} server={server_tick} offset={new_offset}")
        ctx.tick_offset = new_offset
        ctx.last_client_tick = client_tick

    def _get_network_tick(self, ctx: ClientContext) -> int:
        """Return a monotonic tick aligned to the client tick domain when possible."""
        tick = get_ticks()
        if ctx.tick_offset is not None:
            tick = tick + ctx.tick_offset
        if ctx.last_client_tick and tick < ctx.last_client_tick:
            tick = ctx.last_client_tick
        if ctx.last_sent_tick and tick <= ctx.last_sent_tick:
            tick = ctx.last_sent_tick + 1
        ctx.last_sent_tick = tick & 0xFFFFFFFF
        return ctx.last_sent_tick

    def _get_local_state_weapon_type(self, ctx: ClientContext) -> int:
        """Return weapon type index used by local player state (entity type index, not weapon slot)."""
        if self.local_state_weapon_type:
            return self.local_state_weapon_type
        if getattr(ctx, "entity_type", None) is not None:
            return int(ctx.entity_type)
        return 0

    def _get_local_state_ammo_bits(self, ctx: ClientContext) -> tuple:
        """Return (ammo_bits, ammo_mask) for local player state."""
        if self.local_state_ammo_override:
            return self.local_state_ammo_bits, self.local_state_ammo_mask

        if not self.local_state_ammo_from_behavior:
            return 0, 0

        weapon_type = self._get_local_state_weapon_type(ctx)
        active_bits = 0
        if 0 <= weapon_type < len(self.behavior_weapon_caps):
            active_bits = self.behavior_weapon_caps[weapon_type][2]
        active_mask = (1 << active_bits) - 1 if active_bits > 0 else 0
        return active_bits, active_mask

    def _get_local_state_turret_bits(self, ctx: ClientContext) -> tuple:
        """
        Return (primary_bits, primary_angle, secondary_bits, secondary_angle).
        Flags are inferred from entity_type unless overrides are provided.
        """
        weapon_type = self._get_local_state_weapon_type(ctx)

        # Default turret bits OFF. Our inferred turret flags have caused bitstream
        # misalignment and client crashes; only enable via explicit overrides.
        primary_flag = False
        secondary_flag = False

        if self.local_state_primary_override:
            primary_flag = self.local_state_primary_override not in ("0", "false", "False")
        if self.local_state_secondary_override:
            secondary_flag = self.local_state_secondary_override not in ("0", "false", "False")

        if primary_flag:
            primary_bits = max(0, self.local_state_turret_bits)
            primary_angle = ctx.player_aim_yaw if ctx else 0.0
        else:
            primary_bits = 0
            primary_angle = 0.0

        if secondary_flag:
            secondary_bits = max(0, self.local_state_turret_bits)
            secondary_angle = ctx.player_aim_yaw if ctx else 0.0
        else:
            secondary_bits = 0
            secondary_angle = 0.0

        return primary_bits, primary_angle, secondary_bits, secondary_angle

    def _to_client_pos(self, pos: tuple) -> tuple:
        """Apply configured world offset when sending positions to the client."""
        if self.up_axis == "z":
            return (pos[0], pos[1], pos[2] + self.pos_offset)
        return (pos[0], pos[1] + self.pos_offset, pos[2])

    def _snapshot_clients(self):
        """Return a snapshot list of all clients (thread-safe)."""
        with self.clients_lock:
            return list(self.clients.values())

    def _snapshot_in_game_clients(self):
        """Return a snapshot list of in-game clients (thread-safe)."""
        return [c for c in self._snapshot_clients() if c.session and c.session.in_game]

    def _send_packet_to_client(self, ctx: ClientContext, payload: bytes, *, prefer_tcp: bool = True) -> None:
        """Send payload to a client, preferring TCP and falling back to UDP."""
        sent = False
        if prefer_tcp and ctx.tcp_handler:
            try:
                ctx.tcp_handler.send(payload, log=False)
                sent = True
            except Exception as tcp_err:
                print(f"[MULTI] Client {ctx.client_id}: TCP send failed ({tcp_err})")
        if not sent and self.udp_handler and ctx.session.udp_addr:
            self.udp_handler.send_to(payload, ctx.session.udp_addr)

    def _send_roster_entry(self, target_ctx: ClientContext, player_ctx: ClientContext) -> None:
        """Send ADD_TO_ROSTER for player_ctx to target_ctx (once)."""
        if not target_ctx.tcp_handler:
            return
        player_id = player_ctx.session.player_id or player_ctx.entity_id
        if player_id in target_ctx.known_roster_ids:
            return
        name = player_ctx.session.username or f"Player{player_ctx.client_id}"
        team = player_ctx.session.team_id or 1
        target_ctx.tcp_handler.send(build_add_to_roster(
            player_id=player_id,
            entity_id=player_id,
            name=name,
            team=team,
        ))
        target_ctx.known_roster_ids.add(player_id)
        print(f"[MULTI] Sent roster {name} (id={player_id}) -> client {target_ctx.client_id}")

    def _send_entity_create(self, target_ctx: ClientContext, player_ctx: ClientContext) -> None:
        """Send UPDATE_ARRAY entity creation for player_ctx to target_ctx (once)."""
        if not target_ctx.session.translation_ack_received:
            return
        entity_id = player_ctx.session.entity_id or player_ctx.entity_id
        if entity_id in target_ctx.known_entity_ids:
            return
        team = player_ctx.session.team_id or 1
        pos = self._to_client_pos(player_ctx.player_pos)
        tick = self._get_network_tick(target_ctx)
        packet = build_update_array_create_tank(
            tick=tick,
            entity_id=entity_id,
            entity_type=player_ctx.entity_type,
            team=team,
            pos=pos,
            include_health=False,
            is_manned=False,
        )
        self._send_packet_to_client(target_ctx, packet, prefer_tcp=True)
        target_ctx.known_entity_ids.add(entity_id)
        print(f"[MULTI] Sent entity create id={entity_id} -> client {target_ctx.client_id}")

    def _ensure_multiplayer_visibility(self, ctx: ClientContext) -> None:
        """Ensure ctx sees other players and vice versa once translation is ready."""
        if not ctx.session.translation_ack_received:
            return
        for other in self._snapshot_in_game_clients():
            if other is ctx:
                continue
            # Always try to exchange roster entries (idempotent).
            self._send_roster_entry(ctx, other)
            if other.session.translation_ack_received:
                self._send_roster_entry(other, ctx)
            # Entity creation requires target translation ack.
            self._send_entity_create(ctx, other)
            if other.session.translation_ack_received:
                self._send_entity_create(other, ctx)

    def _sync_clients_on_spawn(self, ctx: ClientContext) -> None:
        """Ensure new spawns are visible to all in-game clients."""
        others = [c for c in self._snapshot_in_game_clients() if c is not ctx]
        for other in others:
            # Roster entries both directions
            self._send_roster_entry(other, ctx)
            self._send_roster_entry(ctx, other)
            # Entity creation both directions
            self._send_entity_create(other, ctx)
            self._send_entity_create(ctx, other)

    def _send_remote_player_updates(self, ctx: ClientContext, tick: int, *, prefer_tcp: bool = True) -> None:
        """Send other players' transforms to a client."""
        for other in self._snapshot_in_game_clients():
            if other is ctx:
                continue
            entity_id = other.session.entity_id or other.entity_id
            if entity_id not in ctx.known_entity_ids:
                continue
            send_pos = self._to_client_pos(other.player_pos)
            payload = build_update_array_player_update(
                tick,
                entity_id,
                pos=send_pos,
                vel=other.player_vel,
                rot=(
                    other.player_pose.get("roll", 0.0),
                    0.0,
                    other.player_yaw,
                ),
                include_local_state=False,
                include_entity_vitals=False,
                is_manned=False,
            )
            self._send_packet_to_client(ctx, payload, prefer_tcp=prefer_tcp)

    def _create_client_context(self, client_addr: tuple) -> ClientContext:
        """Create a new ClientContext with unique IDs and initialized systems."""
        with self.clients_lock:
            client_id = self.next_client_id
            self.next_client_id += 1
            entity_id = self.next_entity_id
            self.next_entity_id += 1

        # Create session for this client
        session = Session()

        # Initialize player pose based on up axis config
        if self.up_axis == "z":
            init_pos = (100.0, 100.0, self.spawn_height)
        else:
            init_pos = (100.0, 5.0, 100.0)

        # Create context
        ctx = ClientContext(
            client_id=client_id,
            client_addr=client_addr,
            session=session,
            entity_id=entity_id,
            player_pos=init_pos,
        )
        ctx.entity_type = 0

        # Update pose dict
        ctx.player_pose["pos"] = init_pos

        # Create weapon system for this client
        ctx.weapon_system = WeaponSystem()
        # Use a per-client projectile ID range to avoid colliding with player OIDs.
        # Keep IDs within 16-bit-ish limits (some client arrays appear indexed by OID).
        # Collisions (e.g., projectile id == player id) can crash the client.
        ctx.weapon_system.next_entity_id = max(ctx.weapon_system.next_entity_id, 20000 + (client_id * 1000))
        ctx.weapon_system.on_chain_gun_fire = lambda pos, rot, team, name=None: self._on_chain_gun_fire(ctx, pos, rot, team, name)
        ctx.weapon_system.on_projectile_spawn = lambda proj: self._on_projectile_spawn(ctx, proj)

        # Create jump jet system for this client
        ctx.jump_jet_system = JumpJetSystem()
        ctx.jump_jet_system.on_jump = lambda pid, imp, vel: self._on_jump_jet_triggered(ctx, pid, imp, vel)

        return ctx

    def start(self):
        """Start the server."""
        print(f"[SERVER] Starting on {self.host}:{self.port}")

        # Check for wulf-forge compatibility mode
        if os.environ.get('WULFRAM_WULFFORGE_MODE') == '1':
            FEATURES.set_wulfforge_mode(True)

        FEATURES.log_state()

        # Create TCP socket
        tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp_sock.bind((self.host, self.port))
        tcp_sock.listen(5)  # Allow multiple pending connections

        # Create UDP socket
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_sock.bind(("0.0.0.0", self.port))
        udp_sock.settimeout(0.1)

        self.udp_handler = UDPHandler(udp_sock, self.logger)

        # Start UDP listener thread (set running first to avoid race)
        self.running = True
        udp_thread = threading.Thread(target=self._udp_loop, daemon=True)
        udp_thread.start()

        # Start control server for packet injection
        self.control_server.server = self
        self.control_server.start()

        print(f"[SERVER] Listening on {self.host}:{self.port} (multi-client enabled)")

        try:
            while self.running:
                tcp_sock.settimeout(1.0)
                try:
                    client_sock, client_addr = tcp_sock.accept()
                    print(f"[SERVER] Client connected from {client_addr}")

                    # Spawn a new thread for this client (non-blocking)
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_sock, client_addr),
                        daemon=True
                    )
                    client_thread.start()

                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("\n[SERVER] Shutting down...")
        finally:
            self.running = False
            # Clean up all client contexts
            with self.clients_lock:
                for ctx in self.clients.values():
                    ctx.running = False
                self.clients.clear()
            tcp_sock.close()
            udp_sock.close()

    def _udp_loop(self):
        """Handle incoming UDP packets, routing to correct client context."""
        while self.running:
            data, addr = self.udp_handler.recv_from()
            if data is None:
                continue

            if len(data) < 1:
                continue

            # Find or identify client for this UDP address
            ctx = self.udp_addr_to_client.get(addr)

            # Debug: log raw datagrams that might contain ACTION packets
            if len(data) > 5 and 0x09 in data:
                print(f"[UDP-RAW] Datagram with 0x09: len={len(data)} data={data[:40].hex()}")
            # Debug: log raw datagrams that might contain VIEWPOINT_INFO (0x35)
            if 0x35 in data:
                indices = [i for i, b in enumerate(data) if b == 0x35]
                if indices:
                    idx = indices[0]
                    start = max(0, idx - 8)
                    end = min(len(data), idx + 24)
                    snippet = data[start:end].hex()
                    print(f"[UDP-RAW] Datagram with 0x35: len={len(data)} idxs={indices} snippet={snippet}")
                    print(f"[UDP-RAW] Datagram with 0x35 full={data.hex()}")
            # If 0x35 appears inside a 0x10 wrapper, dump full hex for analysis.
            if data[0] == 0x10 and 0x35 in data:
                print(f"[UDP-RAW] 0x10 wrapper: len={len(data)} hex={data.hex()}")

            # Parse multiple packets from a single UDP datagram
            for packet in self._parse_udp_datagram(data, ctx):
                self._handle_single_udp_packet(ctx, packet, addr)

    def _ping_loop(self, ctx: ClientContext):
        """Send periodic ping requests to keep connection alive (like wulf-forge)."""
        from .packets import build_ping_request
        while ctx.running and not ctx.ping_stop_event.wait(2.0):
            if ctx.tcp_handler:
                try:
                    ctx.tcp_handler.send(build_ping_request())
                except Exception:
                    break

    def _start_ping_loop(self, ctx: ClientContext):
        """Start the ping loop thread for a client."""
        if ctx.ping_thread is not None:
            return  # Already running
        ctx.ping_stop_event.clear()
        ctx.ping_thread = threading.Thread(target=self._ping_loop, args=(ctx,), daemon=True)
        ctx.ping_thread.start()

    def _stop_ping_loop(self, ctx: ClientContext):
        """Stop the ping loop thread for a client."""
        ctx.ping_stop_event.set()
        ctx.ping_thread = None

    def _parse_udp_datagram(self, data: bytes, ctx: Optional[ClientContext] = None):
        """
        Parse a UDP datagram into individual packets.
        Packets can be batched together in a single datagram.

        IMPORTANT: 0x10 is a container packet that wraps other packets.
        We need to extract the inner packets, especially 0x35 VIEWPOINT_INFO.
        """
        def _extract_viewpoint_payloads(raw: bytes):
            """Extract synthetic VIEWPOINT_INFO packets from compressed payloads."""
            payloads = []
            for pos in range(len(raw) - 2):
                if raw[pos] == 0x00 and raw[pos + 1] == 0x00 and raw[pos + 2] == 0x14:
                    payload = raw[pos + 3:pos + 18]
                    if len(payload) < 15:
                        continue
                    # Signature check for viewpoint payload (empirical).
                    if payload[0] == 0xE4 and payload[1] == 0x00 and payload[5] == 0x80 and payload[6] == 0x00 and payload[7] == 0x02 and payload[14] == 0x00:
                        synthetic = b'\x35\x00\x00\x00\x14' + payload
                        payloads.append((pos, synthetic))
            return payloads

        def _action_dump_len() -> int:
            # ACTION_DUMP: opcode + tick(32) + frame(32) + slots 1..21 (bit-packed)
            # Use ctx's weapon system if available, else use default values
            if ctx and ctx.weapon_system:
                control_bits = ctx.weapon_system.control_bits
                zoom_bits = ctx.weapon_system.zoom_bits
            else:
                control_bits = 10
                zoom_bits = 5
            control_slot_count = 5  # slots 1,2,3,6,7
            other_slot_count = 21 - control_slot_count - 1  # minus slot 5 (upward_thrust uses zoom quantizer)
            bits = 64 + zoom_bits + control_slot_count * control_bits + other_slot_count
            return 1 + ((bits + 7) // 8)

        def _action_update_len(start: int) -> int | None:
            remaining = len(data) - start
            if remaining <= 0:
                return None
            # ACTION_UPDATE: opcode + count(8) + tick(32) + frame(32) + bit-packed slot/value pairs
            if remaining < 10:
                return None
            reader = BitReader(data[start + 1:])
            # Use ctx's weapon system if available
            if ctx and ctx.weapon_system:
                ws = ctx.weapon_system
            else:
                # Create a temporary weapon system for parsing
                ws = WeaponSystem()
            try:
                count = reader.read_bits(8)
                reader.read_bits(32)
                reader.read_bits(32)
                max_updates = min(count, 64)
                for _ in range(max_updates):
                    slot_idx = reader.read_bits(ws.slot_index_bits)
                    if slot_idx == BehaviorSlot.UPWARD_THRUST:
                        reader.read_bits(ws.zoom_bits)
                    elif slot_idx in (BehaviorSlot.UNUSED0, BehaviorSlot.TURNING, BehaviorSlot.MOVING_FORWARD,
                                      BehaviorSlot.MOVING_SIDEWAYS, BehaviorSlot.SLOT6, BehaviorSlot.SLOT7):
                        reader.read_bits(ws.control_bits)
                    else:
                        reader.read_bits(1)
            except ValueError:
                return None
            bitpos = reader.byte_pos * 8 + reader.bit_pos
            pkt_len = 1 + ((bitpos + 7) // 8)
            if 0 < pkt_len <= remaining:
                return pkt_len
            return None

        # First pass: extract viewpoint payloads from any UDP datagram.
        extracted_viewpoints = _extract_viewpoint_payloads(data)
        for pos, pkt in extracted_viewpoints:
            print(f"[VIEWPOINT-EXTRACT] payload at offset {pos} len={len(pkt)}")
            yield pkt

        cursor = 0
        while cursor < len(data):
            if cursor >= len(data):
                break

            pkt_type = data[cursor]

            # 0x02 appears to be a Wulf-Forge UDP wrapper that carries
            # input/viewpoint payloads after a fixed 10-byte header.
            # Empirically, payloads start at offset +10 and include 0x09/0x0A/0x10/0x40.
            if pkt_type == 0x02:
                remaining = len(data) - cursor
                if remaining >= 11:
                    payload = data[cursor + 10:]
                    # If payload has any non-zero bytes, parse it like a normal datagram.
                    if payload and any(payload):
                        for inner in self._parse_udp_datagram(payload):
                            yield inner
                # Consume the wrapper
                break

            # 0x10 CONTAINER packet - wraps other packets (including 0x35 VIEWPOINT_INFO)
            # The actual format needs empirical discovery - search for 0x35 in the data
            if pkt_type == 0x10:
                # 0x10 appears to carry compressed viewpoint payloads.
                raw = data[cursor:]
                extracted = len(extracted_viewpoints)
                # Debug: look for literal 0x35 packets inside the container as a fallback.
                if extracted == 0:
                    full_hex = data[cursor:].hex()
                    if '35' in full_hex:
                        positions = [i for i, b in enumerate(raw) if b == 0x35]
                        for pos in positions:
                            start = max(0, pos - 4)
                            end = min(len(raw), pos + 20)
                            context = raw[start:end].hex()
                            print(f"[0x10-SCAN] Found 0x35 at offset {pos}: ...{context}...")
                            if pos + 5 <= len(raw):
                                potential_pkt = raw[pos:]
                                if len(potential_pkt) >= 5 and potential_pkt[3:5] == b"\x00\x14":
                                    pkt_len = 20
                                    if pos + pkt_len <= len(raw):
                                        print(f"[0x10-EXTRACT] Extracting 0x35 at offset {pos}, declared len={pkt_len}")
                                        yield potential_pkt[:pkt_len]

                # Consume entire 0x10 packet
                break

            # Reliable stream packets with length at bytes 3-4
            if pkt_type in (0x20, 0x25, 0x33, 0x35, 0x3a):
                if cursor + 5 > len(data):
                    yield data[cursor:]
                    break
                pkt_len = struct.unpack(">H", data[cursor+3:cursor+5])[0]
                if pkt_len < 5:
                    pkt_len = len(data) - cursor
                end = cursor + pkt_len
                if end > len(data):
                    end = len(data)
                yield data[cursor:end]
                cursor = end

            # Fixed-size packets
            elif pkt_type == 0x40:  # INPUT_FEEDBACK - 5 bytes (opcode + u32 frame)
                pkt_len = 5
                yield data[cursor:cursor+pkt_len]
                cursor += pkt_len

            elif pkt_type == 0x0B:  # PING - 9 bytes typically
                pkt_len = min(9, len(data) - cursor)
                yield data[cursor:cursor+pkt_len]
                cursor += pkt_len

            elif pkt_type == 0x2E:  # WEAPON_DEMAND - ~10 bytes
                pkt_len = min(10, len(data) - cursor)
                yield data[cursor:cursor+pkt_len]
                cursor += pkt_len

            elif pkt_type == 0x0C:  # Unknown - seems to be 9 bytes
                pkt_len = min(9, len(data) - cursor)
                yield data[cursor:cursor+pkt_len]
                cursor += pkt_len

            elif pkt_type == 0x09:  # ACTION_DUMP - variable, has length info
                pkt_len = _action_dump_len()
                end = cursor + pkt_len
                if end > len(data):
                    end = len(data)
                yield data[cursor:end]
                cursor = end

            elif pkt_type == 0x0A:  # ACTION_UPDATE - variable
                pkt_len = _action_update_len(cursor)
                if pkt_len:
                    end = cursor + pkt_len
                    yield data[cursor:end]
                    cursor = end
                else:
                    end = cursor + 1
                    while end < len(data) and data[end] not in (0x09, 0x0A, 0x0B, 0x0C, 0x25, 0x33, 0x35, 0x3a, 0x40):
                        end += 1
                    yield data[cursor:end]
                    cursor = end

            else:
                # Unknown packet - consume rest of datagram
                yield data[cursor:]
                break

    def _handle_single_udp_packet(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle a single UDP packet (after parsing from datagram)."""
        if not data:
            return

        pkt_type = data[0]

        # Update ctx if we found it, or try to find it from addr
        if ctx is None:
            ctx = self.udp_addr_to_client.get(addr)

        # For some packet types (HELLO, D_HANDSHAKE), we may not have ctx yet
        # These packets help identify/register the client

        # Diagnostic: log all reliable stream packets to debug 0x35 visibility
        if pkt_type in (0x33, 0x35, 0x25, 0x20, 0x3a):
            print(f"[UDP-RELIABLE] 0x{pkt_type:02X} len={len(data)} head={data[:min(16,len(data))].hex()}")

        # Handle UDP HELLO (session key verification)
        if pkt_type == 0x08:
            # D_Protocol carrier vs simple HELLO_ACK
            if len(data) >= 2 and 0x01 <= data[1] <= 0x04:
                d_type = data[1]
                if d_type == 0x03:
                    # Treat payload as D_HANDSHAKE (strip carrier byte)
                    self._handle_udp_d_handshake(ctx, data[1:], addr)
                elif d_type == 0x02:
                    print(f"[UDP] D_ACK from {addr} (len={len(data)})")
                else:
                    print(f"[UDP] D_Protocol type=0x{d_type:02X} from {addr} (len={len(data)})")
            else:
                # HELLO_ACK - this registers the UDP address for a client
                text = data[1:].decode('ascii', errors='ignore').strip('\x00')
                print(f"[UDP] HELLO_ACK from {addr}: '{text}'")

                # If ctx is None, find a client waiting for UDP verification
                if ctx is None:
                    from .session import Phase
                    with self.clients_lock:
                        for c in self.clients.values():
                            if c.session and c.session.phase == Phase.HANDSHAKE and not c.session.udp_verified:
                                ctx = c
                                print(f"[UDP] Matched HELLO_ACK to client {ctx.client_id} in HANDSHAKE state")
                                break

                if ctx:
                    ctx.session.udp_addr = addr
                    ctx.session.udp_verified = True
                    # Register this UDP address for the client
                    self.udp_addr_to_client[addr] = ctx
                    print(f"[UDP] Registered client {ctx.client_id} with UDP addr {addr}")

        elif pkt_type == 0x13:
            # Session key
            if len(data) > 5:
                key = data[5:].decode('ascii', errors='ignore').strip('\x00')
                print(f"[UDP] Session key from {addr}: '{key}'")

        elif pkt_type == 0x33:
            # TRANSLATION_ACK (client -> server, reliable stream packet)
            # MUST ACK this or client stalls reliable-send window and won't send 0x35!
            if ctx:
                ctx.session.translation_ack_received = True
                ctx.session.translation_ack_time = time.monotonic()
                ctx.session.last_udp_activity = time.monotonic()
            print(f"[UDP] TRANSLATION_ACK from {addr} (len={len(data)})")
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x33, seq_num)
                print(f"[UDP] Sent ACK for 0x33 seq={seq_num}")

        elif pkt_type == 0x03:
            # D_HANDSHAKE (Wulf-Forge UDP stream init)
            self._handle_udp_d_handshake(ctx, data, addr)

        elif pkt_type == 0x20:
            # COMM_REQ (chat/system commands) - used by Wulf-Forge for /s spawn
            self._handle_udp_chat(ctx, data, addr)

        elif pkt_type == 0x25:
            # REINCARNATE over UDP (Wulf-Forge style)
            self._handle_udp_reincarnate(ctx, data, addr)

        elif pkt_type == 0x35:
            # VIEWPOINT_INFO - client sends camera/view position and orientation
            # This is the ACTUAL player pose, not reconstructed from inputs!
            print(f"[UDP] VIEWPOINT_INFO 0x35 len={len(data)} data={data[:32].hex()}...")
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x35, seq_num)
                self._handle_viewpoint_info(ctx, data, addr)

        elif pkt_type == 0x3a:
            # BEACON_REQ - client requests beacon info
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x3a, seq_num)

        elif pkt_type == 0x09:
            # ACTION_DUMP - full behavior slot dump (includes fire state)
            print(f"[UDP] ACTION_DUMP received: len={len(data)} data={data[:24].hex()}")
            self._handle_action_dump(ctx, data, addr)

        elif pkt_type == 0x0A:
            # ACTION_UPDATE - incremental behavior slot updates
            self._handle_action_update(ctx, data, addr)

        elif pkt_type == 0x40:
            # INPUT_FEEDBACK - player input state (movement, firing)
            self._handle_input_feedback(ctx, data, addr)

        elif pkt_type == 0x2E:
            # WEAPON_DEMAND - fire button pressed
            print(f"[UDP] WEAPON_DEMAND from {addr} (len={len(data)}) data={data.hex()}")
            self._handle_weapon_demand(ctx, data, addr)

        elif pkt_type == 0x0C:
            # STATE_REQUEST - may contain state/position info
            print(f"[UDP] STATE_REQUEST 0x0C len={len(data)} data={data.hex()}")
            self._handle_state_request(ctx, data, addr)

        else:
            print(f"[UDP] Packet 0x{pkt_type:02X} from {addr} (len={len(data)})")

        # Update last activity time if we have ctx
        if ctx:
            ctx.session.last_udp_activity = time.monotonic()

    def _handle_udp_d_handshake(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle UDP D_HANDSHAKE (0x03) and respond with stream definitions."""
        handlers.handle_udp_d_handshake(self, ctx, data, addr)

    def _send_udp_ack(self, ctx: Optional[ClientContext], addr: tuple, packet_id: int, seq_num: int, subcmd: int = 1):
        """Send a Wulf-Forge style UDP ACK (0x02)."""
        handlers.send_udp_ack(self, ctx, addr, packet_id, seq_num, subcmd)

    def _send_udp_wf_spawn(self, ctx: ClientContext, addr: tuple, net_id: int, team_id: int,
                            unit_type: int = 0,
                            pos: tuple = (100.0, 100.0, 100.0),
                            vel: tuple = (0.0, 0.0, 0.0)):
        """Send a Wulf-Forge style UDP TANK (0x18) packet to spawn a unit."""
        if not self.udp_handler:
            return
        payload = build_udp_tank_packet_wf(
            net_id=net_id,
            unit_type=unit_type,
            team_id=team_id,
            pos=pos,
            vel=vel,
            include_vitals=self.tank_vitals,
            weapon_id=self.weapon_id,
            health_mult_bits=1,
            energy_mult_bits=1,
        )
        self.udp_handler.send_to(payload, addr)

    def _spawn_wf_style(self, ctx: ClientContext, team_id: int, net_id: Optional[int] = None,
                         unit_type: int = 0,
                         pos: tuple = (100.0, 100.0, 100.0),
                         vel: tuple = (0.0, 0.0, 0.0),
                         announce: bool = True):
        """
        Spawn using Wulf-Forge's simple approach: just send TankPacket.

        Wulf-Forge's /s spawn only sends a single TankPacket (0x18) -
        no UPDATE_ARRAY, no separate PLAYER_INFO, no entity pre-creation.
        """
        net_id = net_id or (ctx.session.player_id or ctx.entity_id)
        ctx.session.player_id = net_id
        ctx.session.team_id = team_id
        ctx.entity_type = unit_type

        # Reset position tracking to spawn location (near ground)
        if self.up_axis == "z":
            spawn_pos = (pos[0], pos[1], self.spawn_height)
        else:
            spawn_pos = (pos[0], 5.0, pos[2])

        # Offset spawn to avoid overlapping tanks in multi-client tests.
        if ctx.client_id > 1 and self.multi_spawn_offset:
            spawn_pos = (spawn_pos[0] + (ctx.client_id - 1) * self.multi_spawn_offset, spawn_pos[1], spawn_pos[2])
        ctx.player_pos = spawn_pos
        ctx.player_pose["pos"] = spawn_pos
        ctx.player_yaw = 0.0
        ctx.player_heading = 0.0
        ctx.last_action_dump_time = time.monotonic()  # Reset timer for position tracking

        print(f"[SPAWN] Wulf-Forge style: client={ctx.client_id} net_id={net_id} team={team_id} pos={spawn_pos}")

        if not ctx.tcp_handler:
            print("[SPAWN] ERROR: No TCP handler")
            return

        # Roster (already sent during login, but ensure it's there for name display)
        if not ctx.session.roster_sent:
            name = ctx.session.username or f"Player{ctx.client_id}"
            print(f"[SPAWN] Sending ADD_TO_ROSTER for {name}")
            ctx.tcp_handler.send(build_add_to_roster(
                player_id=net_id, entity_id=net_id, name=name, team=team_id
            ))
            ctx.session.roster_sent = True

        # NOTE: Wulf-forge capture shows /s spawn ONLY sends TankPacket + CommMessage
        # It does NOT send UPDATE_STATS or REINCARNATE before TankPacket!
        # Those are only sent in response to team switch requests.

        # Send TankPacket (vitals only if TRANSLATION has been applied).
        include_spawn_vitals = self.tank_vitals and ctx.session.translation_ack_received
        print(f"[SPAWN] Sending UDP TankPacket (vitals={int(include_spawn_vitals)})")
        send_pos = self._to_client_pos(spawn_pos)
        tank_packet = build_udp_tank_packet_wf(
            net_id=net_id,
            unit_type=unit_type,
            team_id=team_id,
            pos=send_pos,
            vel=vel,
            include_vitals=include_spawn_vitals,
            weapon_id=self.weapon_id,
            health_mult_bits=1,   # Wulf-forge uses 1
            energy_mult_bits=1,   # Wulf-forge uses 1
        )
        # HEX DUMP for comparison with wulf-forge
        print(f"[TANK-HEX] len={len(tank_packet)} hex={tank_packet.hex().upper()}")
        # Send over UDP (matching wulf-forge behavior)
        if self.udp_handler and ctx.session.udp_addr:
            self.udp_handler.send_to(tank_packet, ctx.session.udp_addr)
            print(f"[SPAWN] Sent UDP TankPacket to {ctx.session.udp_addr}")
        else:
            # Fallback to TCP if no UDP address
            ctx.tcp_handler.send(tank_packet)
            print(f"[SPAWN] Sent TCP TankPacket (no UDP addr)")

        # Wulf-forge sends CommMessage over UDP after TankPacket (same transport!)
        if announce and self.udp_handler and ctx.session.udp_addr:
            comm_pkt = build_chat_message("Spawning in...", source_id=net_id)
            self.udp_handler.send_to(comm_pkt, ctx.session.udp_addr)
            print(f"[SPAWN] Sent UDP CommMessage to {ctx.session.udp_addr}")

        # NOTE: We rely on TankPacket (UDP) to create the entity.
        # UPDATE_ARRAY_CREATE_TANK causes crash when sent AFTER TankPacket.
        # For now, skip UPDATE_ARRAY and try PLAYER_INFO alone.

        # Ensure TRANSLATION has been applied before sending any local-state data.
        if not ctx.session.translation_ack_received:
            wait_until = time.monotonic() + 2.0
            while not ctx.session.translation_ack_received and time.monotonic() < wait_until:
                time.sleep(0.05)
            if not ctx.session.translation_ack_received:
                print("[SPAWN] WARNING: TRANSLATION_ACK not received before PLAYER_INFO")
        # If we spawned without vitals, send a vitals refresh once TRANSLATION is ready.
        if self.tank_vitals and not include_spawn_vitals and ctx.session.translation_ack_received:
            vitals_packet = build_udp_tank_packet_wf(
                net_id=net_id,
                unit_type=unit_type,
                team_id=team_id,
                pos=send_pos,
                vel=vel,
                include_vitals=True,
                weapon_id=self.weapon_id,
                health_mult_bits=1,
                energy_mult_bits=1,
            )
            if self.udp_handler and ctx.session.udp_addr:
                self.udp_handler.send_to(vitals_packet, ctx.session.udp_addr)
                print(f"[SPAWN] Sent UDP TankPacket vitals refresh to {ctx.session.udp_addr}")
            else:
                ctx.tcp_handler.send(vitals_packet)
                print("[SPAWN] Sent TCP TankPacket vitals refresh (no UDP addr)")

        # PLAYER_INFO tells the client "this is your controllable entity"
        # Without this, client won't send VIEWPOINT_INFO (0x35)
        weapon_type = self._get_local_state_weapon_type(ctx)
        ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
        pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
        if self.player_info_local_state:
            print(
                "[LOCAL-STATE] PLAYER_INFO "
                f"weapon={weapon_type} ammo_bits={ammo_bits} "
                f"pt_bits={pt_bits} st_bits={st_bits}"
            )
        player_info_pkt = build_player_info(
            entity_oid=net_id,
            vehicle_type=unit_type,
            pos=send_pos,
            include_local_state=self.player_info_local_state,
            weapon_id=weapon_type,
            health=1.0,
            fuel=1.0,
            ammo_count_bits=ammo_bits,
            ammo_count=ammo_mask,
            primary_turret_bits=pt_bits,
            primary_turret_angle=pt_angle,
            secondary_turret_bits=st_bits,
            secondary_turret_angle=st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
        )
        ctx.tcp_handler.send(player_info_pkt)
        print(f"[SPAWN] Sent TCP PLAYER_INFO: entity_oid={net_id} vehicle={unit_type}")

        # Complete spawn sequence (matching spawn_full)
        ctx.tcp_handler.send(build_game_clock())
        ctx.tcp_handler.send(build_reincarnate(0x11, "Spawn success"))
        ctx.tcp_handler.send(build_birth_notice(net_id))
        print(f"[SPAWN] Sent GAME_CLOCK, REINCARNATE(0x11), BIRTH_NOTICE")

        # Enter game mode and start tick loop for UPDATE_ARRAY
        ctx.session.entity_id = net_id
        ctx.session.in_game = True
        ctx.session.transition_to(Phase.IN_GAME)

        # Sync roster/entity visibility with other in-game clients.
        self._sync_clients_on_spawn(ctx)

        if FEATURES.tick_loop_enabled and (ctx.tick_thread is None or not ctx.tick_thread.is_alive()):
            ctx.tick_thread = threading.Thread(target=self._tick_loop, args=(ctx,), daemon=True)
            ctx.tick_thread.start()
            print(f"[SPAWN] Started tick loop (local_state_updates={int(self.update_local_state)})")

    def _spawn_wf_minimal(self, ctx: ClientContext, team_id: int, net_id: int, addr: tuple):
        """
        Absolutely minimal spawn - just TankPacket (wulf-forge style).

        Uses include_vitals per WULFRAM_TANK_VITALS (defaults off while investigating).
        TRANSLATION quantizers define 5/10/10 bits which matches TankPacket format.
        """
        print(f"[SPAWN] Minimal WF: client={ctx.client_id} net_id={net_id} team={team_id}")
        ctx.entity_type = 0

        if self.up_axis == "z":
            spawn_pos = (100.0, 100.0, self.spawn_height)
        else:
            spawn_pos = (100.0, 5.0, 100.0)

        if ctx.client_id > 1 and self.multi_spawn_offset:
            spawn_pos = (spawn_pos[0] + (ctx.client_id - 1) * self.multi_spawn_offset, spawn_pos[1], spawn_pos[2])

        send_pos = self._to_client_pos(spawn_pos)
        tank_packet = build_udp_tank_packet_wf(
            net_id=net_id,
            unit_type=0,
            team_id=team_id,
            pos=send_pos,
            vel=(0.0, 0.0, 0.0),
            include_vitals=self.tank_vitals,
            weapon_id=self.weapon_id,
            health_mult_bits=1,
            energy_mult_bits=1,
        )

        if self.udp_handler:
            self.udp_handler.send_to(tank_packet, addr)
            print(f"[SPAWN] Sent minimal TankPacket (vitals={int(self.tank_vitals)}) to {addr}")

    def _handle_udp_chat(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle UDP COMM_REQ (0x20) for /s spawn."""
        handlers.handle_udp_chat(self, ctx, data, addr)

    def _handle_udp_reincarnate(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle UDP REINCARNATE (0x25)."""
        handlers.handle_udp_reincarnate(self, ctx, data, addr)

    def _handle_team_switch(self, ctx: ClientContext, team_id: int, addr: tuple):
        """Handle team switch/spawn request from REINCARNATE."""
        handlers.handle_team_switch(self, ctx, team_id, addr)

    def _spawn_at_point(self, ctx: ClientContext, spawn_point_id: int, vehicle_type: int, addr: tuple):
        """Handle spawn at a specific spawn point."""
        handlers.handle_spawn_at_point(self, ctx, spawn_point_id, vehicle_type, addr)

    def _handle_client(self, sock: socket.socket, addr):
        """Handle a single client connection in its own thread."""
        # Create client context with unique IDs
        ctx = self._create_client_context(addr)
        ctx.session.transition_to(Phase.HANDSHAKE)

        # Note: Don't set timeout here - it interferes with login flow
        # Set timeout later when entering game loop
        ctx.tcp_handler = TCPHandler(sock, self.logger)

        # Register client in global dict
        with self.clients_lock:
            self.clients[ctx.client_id] = ctx

        print(f"[SERVER] Client {ctx.client_id} assigned entity_id={ctx.entity_id}")

        try:
            # Phase 1: Handshake
            self._do_handshake(ctx)

            # Phase 2: Login
            self._do_login(ctx)

            # Phase 3: Team Select / Game Loop
            self._game_loop(ctx)

        except Exception as e:
            print(f"[SERVER] Client {ctx.client_id} error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            ctx.running = False
            ctx.session.in_game = False
            ctx.session.reset()

            # Remove from client tracking
            with self.clients_lock:
                if ctx.client_id in self.clients:
                    del self.clients[ctx.client_id]

            # Remove UDP address mapping
            if ctx.session.udp_addr and ctx.session.udp_addr in self.udp_addr_to_client:
                del self.udp_addr_to_client[ctx.session.udp_addr]

            sock.close()
            print(f"[SERVER] Client {ctx.client_id} disconnected")

    def _do_handshake(self, ctx: ClientContext):
        """Perform initial handshake."""
        time.sleep(0.5)  # Let client initialize

        # Send UDP config
        ctx.tcp_handler.send(build_hello_udp_config(self.host, self.port))

        # Wait for UDP verification
        timeout = time.monotonic() + 5.0
        while not ctx.session.udp_verified and time.monotonic() < timeout:
            time.sleep(0.1)

        if ctx.session.udp_verified:
            print(f"[HANDSHAKE] Client {ctx.client_id} UDP verified")
            ctx.tcp_handler.send(build_identified_udp())
        else:
            print(f"[HANDSHAKE] Client {ctx.client_id} UDP verification timeout")

        # Mirror Wulf-Forge: send HELLO verified (subcmd 0x03) after UDP setup
        ctx.tcp_handler.send(build_hello_verified())

        ctx.session.transition_to(Phase.LOGIN)

    def _do_login(self, ctx: ClientContext):
        """Handle login sequence."""
        # Request username
        ctx.tcp_handler.send(build_login_status(5))  # Code 5 = request handle

        # Wait for packets
        while ctx.session.phase == Phase.LOGIN:
            packet = ctx.tcp_handler.recv()
            if packet is None:
                raise ConnectionError("Client disconnected during login")

            if len(packet) < 1:
                continue

            pkt_type = packet[0]
            print_packet("RECV", pkt_type, packet)

            if pkt_type == PacketType.HELLO:
                self._handle_hello(ctx, packet)
            elif pkt_type == PacketType.LOGIN_REQUEST:
                self._handle_login_request(ctx, packet)
            elif pkt_type == 0x54:
                # Voice data request - ignore
                pass
            else:
                print(f"[LOGIN] Ignoring packet 0x{pkt_type:02X}")

    def _handle_hello(self, ctx: ClientContext, packet: bytes):
        """Handle HELLO packets during login."""
        handlers.handle_hello(self, ctx, packet)

    @staticmethod
    def _decode_lp_string(data: bytes, offset: int):
        """Decode a length-prefixed string. Returns (text, new_offset)."""
        return handlers.decode_lp_string(data, offset)

    def _handle_login_request(self, ctx: ClientContext, packet: bytes):
        """Handle LOGIN_REQUEST packet."""
        handlers.handle_login_request(self, ctx, packet)

    def _send_initial_game_data(self, ctx: ClientContext):
        """Send packets needed for team selection screen."""
        handlers.send_initial_game_data(self, ctx)

    def _game_loop(self, ctx: ClientContext):
        """Main game packet loop."""
        # Set socket timeout for delayed spawn checking (recv returns None on timeout)
        ctx.tcp_handler.sock.settimeout(0.5)

        # Track last activity for dead connection detection
        # UDP packets count as activity since client sends TRANSLATION_ACK continuously
        last_activity = time.monotonic()
        INACTIVITY_TIMEOUT = 10.0  # Disconnect if no TCP/UDP activity for 10 seconds

        while ctx.running and ctx.session.phase in [Phase.TEAM_SELECT, Phase.SPAWNING, Phase.IN_GAME]:
            # Check for delayed spawn
            if ctx.session.delayed_spawn_team and ctx.session.delayed_spawn_time:
                if time.monotonic() >= ctx.session.delayed_spawn_time:
                    team = ctx.session.delayed_spawn_team
                    ctx.session.delayed_spawn_team = 0
                    ctx.session.delayed_spawn_time = 0
                    print(f"[GAME] Client {ctx.client_id}: Executing delayed spawn for team {team}")
                    self._auto_join_team(ctx, team)

            packet = ctx.tcp_handler.recv()
            if packet is None:
                # Timeout - check for inactivity (dead connection)
                # Also check UDP activity since client sends TRANSLATION_ACKs
                if ctx.session.last_udp_activity:
                    last_activity = max(last_activity, ctx.session.last_udp_activity)

                if time.monotonic() - last_activity > INACTIVITY_TIMEOUT:
                    print(f"[GAME] Client {ctx.client_id} inactive for {INACTIVITY_TIMEOUT}s - disconnecting")
                    break

                # Also try getpeername as backup check
                try:
                    ctx.tcp_handler.sock.getpeername()
                    continue  # Socket still connected, just timeout
                except:
                    break  # Socket disconnected

            if len(packet) < 1:
                continue

            pkt_type = packet[0]
            print_packet("RECV", pkt_type, packet)

            if pkt_type == PacketType.BPS:
                self._handle_bps(ctx, packet)
            elif pkt_type == PacketType.LOGIN_REQUEST:
                # Ignore late login packets once we've entered team select
                if not ctx.session.login_complete:
                    self._handle_login_request(ctx, packet)
                else:
                    print("[GAME] Ignoring LOGIN_REQUEST after login complete")
            elif pkt_type == PacketType.WANT_UPDATES:
                self._handle_want_updates(ctx, packet)
            elif pkt_type == PacketType.REINCARNATE:
                self._handle_reincarnate(ctx, packet)
            elif pkt_type == 0x33:
                ctx.session.translation_ack_received = True
                ctx.session.translation_ack_time = time.monotonic()
                print("[GAME] Translation ACK received")
            elif pkt_type == 0x19:
                # Tank resend request (client didn't accept PLAYER_INFO)
                print("[GAME] TANK_RESEND_REQUEST received - resending TankPacket")
                entity_id = ctx.session.entity_id or ctx.entity_id
                tank_pos = (100.0, 15.0, self.spawn_height) if self.up_axis == "z" else (100.0, 15.0, 100.0)
                send_pos = self._to_client_pos(tank_pos)
                ctx.tcp_handler.send(build_tank_packet(
                    net_id=entity_id,
                    unit_type=0,
                    pos=send_pos,
                    vel=(0.0, 0.0, 0.0),
                    flags=1,
                    include_vitals=self.tank_vitals,
                    health=1.0,
                    energy=1.0
                ))
            else:
                print(f"[GAME] Unhandled packet 0x{pkt_type:02X}")

    def _handle_bps(self, ctx: ClientContext, packet: bytes):
        """Handle BPS (bandwidth/rate) packet."""
        handlers.handle_bps(self, ctx, packet)

    def _handle_want_updates(self, ctx: ClientContext, packet: bytes):
        """Handle WANT_UPDATES - client is ready for game data."""
        handlers.handle_want_updates(self, ctx, packet)

    def _auto_join_team(self, ctx: ClientContext, team_id: int):
        """Auto-spawn after WANT_UPDATES using Wulf-Forge-style UDP TANK."""
        print(f"[GAME] Client {ctx.client_id}: Auto-spawn (WF) on team {team_id}")
        self._spawn_wf_style(ctx, team_id=team_id, net_id=ctx.session.player_id or ctx.entity_id)

    def _handle_reincarnate(self, ctx: ClientContext, packet: bytes):
        """Handle REINCARNATE - player wants to spawn."""
        handlers.handle_reincarnate_tcp(self, ctx, packet)

    def _send_tank(self, ctx: ClientContext, entity_id: int = None):
        """Send TankPacket (0x18) to spawn the player's tank."""
        if entity_id is None:
            entity_id = ctx.entity_id
        if self.up_axis == "z":
            spawn_pos = (100.0, 100.0, self.spawn_height)
        else:
            spawn_pos = (100.0, 5.0, 100.0)
        send_pos = self._to_client_pos(spawn_pos)
        ctx.tcp_handler.send(build_tank_packet(
            net_id=entity_id,
            unit_type=0,
            pos=send_pos,
            vel=(0.0, 0.0, 0.0),
            flags=1,
            include_vitals=self.tank_vitals,
            health=1.0,
            energy=1.0
        ))

    # ============ Pose Tracking (from client packets) ============

    def _handle_viewpoint_info(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle VIEWPOINT_INFO (0x35) - contains camera/view orientation (not position).

        Discovered format:
        - Byte 0: opcode (0x35)
        - Bytes 1-2: sequence number (for ACK)
        - Bytes 3-10: pitch angle as big-endian double (radians)
        - Bytes 11-18: yaw angle as big-endian double (radians)

        This provides the aim/view direction, not world position.
        Position comes from entity state/TankPacket.
        """
        if ctx is None:
            return

        # Log all VIEWPOINT packets for format analysis
        print(f"[UDP] VIEWPOINT_INFO 0x35 len={len(data)} data={data.hex()[:40]}...")

        if len(data) < 20:
            # Short packet - might be different subtype, log for analysis
            if len(data) >= 5:
                print(f"[VIEWPOINT-SHORT] len={len(data)} bytes={data.hex()}")
            return

        seq_num = struct.unpack(">H", data[1:3])[0]
        # Empirical format (20 bytes total):
        # [0]=opcode(0x35) [1:3]=seq [3:5]=len(0x0014)
        # payload (15 bytes) appears to include 24-bit pitch/yaw fields.
        payload = data[5:]

        if len(payload) >= 14:
            try:
                # 24-bit fixed values (0..2^24) scaled to degrees.
                pitch_raw = (payload[2] << 16) | (payload[3] << 8) | payload[4]
                yaw_raw = (payload[11] << 16) | (payload[12] << 8) | payload[13]
                pitch_deg = (pitch_raw / (1 << 24)) * 360.0
                yaw_deg = (yaw_raw / (1 << 24)) * 360.0 - 180.0
                if pitch_deg > 180.0:
                    pitch_deg -= 360.0

                pitch = math.radians(pitch_deg)
                yaw = math.radians(yaw_deg)

                # Update aim pose (view/turret) with view angles
                old_yaw = ctx.player_aim_yaw
                old_pitch = ctx.player_aim_pitch
                ctx.player_aim_pitch = pitch
                ctx.player_aim_yaw = yaw
                ctx.player_aim_source = "viewpoint"
                ctx.player_aim_time = time.monotonic()

                # Track viewpoint packet count
                ctx.viewpoint_count = ctx.viewpoint_count + 1

                # Log ALL viewpoint packets for debugging
                old_yaw_deg = math.degrees(old_yaw)
                print(
                    f"[VIEWPOINT #{ctx.viewpoint_count}] pitch={pitch_deg:.1f} "
                    f"yaw={yaw_deg:.1f} (was {old_yaw_deg:.1f})"
                )
                return
            except (IndexError, ValueError, struct.error) as e:
                print(f"[VIEWPOINT-ERR] Failed to decode: {e}")

        # Fallback: attempt double decode if payload is larger in other variants.
        if len(payload) >= 17:
            try:
                pitch = struct.unpack(">d", payload[0:8])[0]
                yaw = struct.unpack(">d", payload[8:16])[0]

                old_yaw = ctx.player_aim_yaw
                ctx.player_aim_pitch = pitch
                ctx.player_aim_yaw = yaw
                ctx.player_aim_source = "viewpoint"
                ctx.player_aim_time = time.monotonic()

                ctx.viewpoint_count = ctx.viewpoint_count + 1
                print(
                    f"[VIEWPOINT #{ctx.viewpoint_count}] pitch={math.degrees(pitch):.1f} "
                    f"yaw={math.degrees(yaw):.1f} (was {math.degrees(old_yaw):.1f})"
                )
            except struct.error as e:
                print(f"[VIEWPOINT-ERR] Failed to decode (double): {e}")

    def _handle_state_request(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle STATE_REQUEST (0x0C) - may contain state/position info.
        Format: [opcode:1] [tick:4] [a:2] [b:2] (9 bytes typical)
        """
        if len(data) < 5:
            return

        # Decode fields
        tick = struct.unpack(">I", data[1:5])[0] if len(data) >= 5 else 0
        a = struct.unpack(">H", data[5:7])[0] if len(data) >= 7 else 0
        b = struct.unpack(">H", data[7:9])[0] if len(data) >= 9 else 0

        # Log occasionally to avoid spam
        if tick % 1000 == 0:
            print(f"[0x0C] STATE_REQUEST tick={tick} a={a} b={b} len={len(data)}")

    # ============ Weapon System Handlers ============

    def _handle_action_dump(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle ACTION_DUMP packet (0x09).
        Contains all behavior slot values including fire state.
        Format: [opcode:1] [tick:4] [frame:4] [slot_data:bit-packed]
        """
        if ctx is None:
            return

        if len(data) >= 5:
            try:
                self._sync_tick_offset(ctx, struct.unpack(">I", data[1:5])[0])
            except struct.error:
                pass
        if ctx.weapon_system.decode_action_dump(data):
            ctx.last_action_dump_time = time.monotonic()
            self._update_player_aim(ctx)
            # Update weapon system with current pose
            ctx.weapon_system.player_id = ctx.session.player_id or ctx.entity_id
            ctx.weapon_system.player_team = ctx.session.team_id or 2
            # Use server-tracked position for projectile spawns.
            ctx.weapon_system.player_pos = ctx.player_pos
            # Aim rotation (viewpoint when available) order: (roll, pitch, yaw)
            aim_pitch, aim_yaw, aim_src = self._get_aim_rotation(ctx)
            aim_override = "auto"
            if self.projectile_aim_source == "body":
                aim_pitch = 0.0
                aim_yaw = ctx.player_heading
                aim_override = "body"
            elif self.projectile_aim_source == "viewpoint":
                aim_pitch = ctx.player_aim_pitch
                aim_yaw = ctx.player_aim_yaw
                aim_override = "viewpoint"
            ctx.weapon_system.player_rot = (
                ctx.player_pose.get("roll", 0.0),
                aim_pitch,
                aim_yaw,
            )
            ctx.weapon_system.projectile_aim_source = aim_override if aim_override != "auto" else aim_src
            # Force pulse cannon for testing
            ctx.weapon_system.current_weapon = 4  # PULSE_CANNON

            # Weapon spawning with wulf-forge encoding
            new_projectiles = ctx.weapon_system.update()
            if new_projectiles:
                yaw_deg = math.degrees(ctx.player_yaw)
                print(f"[WEAPON-FIRE] Firing {len(new_projectiles)} proj at yaw={yaw_deg:.1f} pos={ctx.player_pos}")
            for proj in new_projectiles:
                self._spawn_moving_projectile(ctx, proj, addr)

            # Process jump jets (slot 5 = upward thrust)
            self._process_jump_jets(ctx, addr)

    def _handle_action_update(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle ACTION_UPDATE packet (0x0A).
        Contains incremental behavior slot updates.
        """
        if ctx is None:
            return

        print(f"[UDP] ACTION_UPDATE received: len={len(data)} data={data.hex()}")
        if len(data) >= 6:
            try:
                self._sync_tick_offset(ctx, struct.unpack(">I", data[2:6])[0])
            except struct.error:
                pass
        if ctx.weapon_system.decode_action_update(data):
            ctx.last_action_dump_time = time.monotonic()
            self._update_player_aim(ctx)
            # Yaw is tracked via VIEWPOINT_INFO when available; otherwise input-based fallback is used.
            # Position is simulated in the tick loop from behavior slots.

            # Update weapon system with current pose
            ctx.weapon_system.player_id = ctx.session.player_id or ctx.entity_id
            ctx.weapon_system.player_team = ctx.session.team_id or 2
            # Use server-tracked position for projectile spawns.
            ctx.weapon_system.player_pos = ctx.player_pos
            # Aim rotation (viewpoint when available) order: (roll, pitch, yaw)
            aim_pitch, aim_yaw, aim_src = self._get_aim_rotation(ctx)
            aim_override = "auto"
            if self.projectile_aim_source == "body":
                aim_pitch = 0.0
                aim_yaw = ctx.player_heading
                aim_override = "body"
            elif self.projectile_aim_source == "viewpoint":
                aim_pitch = ctx.player_aim_pitch
                aim_yaw = ctx.player_aim_yaw
                aim_override = "viewpoint"
            ctx.weapon_system.player_rot = (
                ctx.player_pose.get("roll", 0.0),
                aim_pitch,
                aim_yaw,
            )
            ctx.weapon_system.projectile_aim_source = aim_override if aim_override != "auto" else aim_src
            # Force pulse cannon for testing
            ctx.weapon_system.current_weapon = 4  # PULSE_CANNON

            # Weapon spawning with wulf-forge encoding
            new_projectiles = ctx.weapon_system.update()
            if new_projectiles:
                yaw_deg = math.degrees(ctx.player_yaw)
                print(f"[WEAPON-FIRE] Firing {len(new_projectiles)} proj at yaw={yaw_deg:.1f} pos={ctx.player_pos}")
            for proj in new_projectiles:
                self._spawn_moving_projectile(ctx, proj, addr)

            # Process jump jets (slot 5 = upward thrust)
            self._process_jump_jets(ctx, addr)

    def _handle_weapon_demand(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle WEAPON_DEMAND packet (0x2E) - weapon selection/action.
        Format: [opcode:1] [mode:1] [slot:4] [param:4]
        Mode values:
          1 = Cycle weapon forward
          2 = Cycle weapon backward
          3 = Buy/request ammo
          4 = Sell/drop ammo
        """
        if ctx is None or len(data) < 2:
            return

        # Parse the packet
        mode = data[1] if len(data) > 1 else 0
        slot = struct.unpack(">I", data[2:6])[0] if len(data) >= 6 else 0
        param = struct.unpack(">i", data[6:10])[0] if len(data) >= 10 else 0

        print(f"[WEAPON] WEAPON_DEMAND mode={mode} slot={slot} param={param}")

        # Handle weapon cycling
        if mode == 1:  # Cycle forward
            ctx.weapon_system.current_weapon = (ctx.weapon_system.current_weapon + 1) % 13
            print(f"[WEAPON] Cycled forward to weapon slot {ctx.weapon_system.current_weapon}")
        elif mode == 2:  # Cycle backward
            ctx.weapon_system.current_weapon = (ctx.weapon_system.current_weapon - 1) % 13
            print(f"[WEAPON] Cycled backward to weapon slot {ctx.weapon_system.current_weapon}")
        elif slot != ctx.weapon_system.current_weapon:
            # Direct weapon selection via slot parameter
            ctx.weapon_system.current_weapon = slot
            print(f"[WEAPON] Selected weapon slot {slot}")

        # Send visual feedback
        if ctx.tcp_handler:
            weapon_names = {0: "Chain Gun", 1: "Pulse Cannon", 2: "Mortar", 3: "Missile",
                           4: "Hunter", 5: "Heavy Missile", 6: "Mine", 7: "Piercer"}
            weapon = weapon_names.get(ctx.weapon_system.current_weapon, f"Weapon {ctx.weapon_system.current_weapon}")
            msg = build_chat_message(f"[{weapon}]", source_id=ctx.session.player_id or ctx.entity_id)
            ctx.tcp_handler.send(msg)

    def _handle_input_feedback(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle INPUT_FEEDBACK packet (0x40).
        Contains player input state - frame counter and possibly input flags.
        """
        if len(data) >= 5:
            frame_counter = struct.unpack(">I", data[1:5])[0]
            # Only log occasionally to reduce spam
            if frame_counter % 100 == 0:
                print(f"[INPUT] INPUT_FEEDBACK frame={frame_counter} len={len(data)}")
            # Additional data might contain input flags
            if len(data) > 5:
                extra = data[5:]
                if extra and extra != b'\x00' * len(extra):
                    print(f"[INPUT] Extra data: {extra.hex()}")

    def _on_chain_gun_fire(self, ctx: ClientContext, pos: tuple, rot: tuple, team: int, weapon_name: str = None):
        """Callback when weapon fires (instant hit or placeholder for projectiles)."""
        weapon_name = weapon_name or "Chain Gun"
        print(f"[WEAPON] {weapon_name} fired! pos={pos}")
        # Send chat message to confirm firing
        if ctx.tcp_handler:
            if weapon_name == "Chain Gun":
                msg = build_chat_message("*ratatatat*", source_id=ctx.session.player_id or ctx.entity_id)
            else:
                # Other weapons get descriptive feedback
                msg = build_chat_message(f"*{weapon_name.lower()} fired*", source_id=ctx.session.player_id or ctx.entity_id)
            ctx.tcp_handler.send(msg)

    def _on_projectile_spawn(self, ctx: ClientContext, proj):
        """Callback when a projectile is spawned."""
        print(f"[WEAPON] Projectile spawned: id={proj.entity_id} type={proj.entity_type.name}")

        # NOTE: Do NOT send spawn here to avoid duplicate spawns.
        # Spawn is handled by _spawn_moving_projectile to prevent TCP/UDP reorders.

        # Send chat message for visual confirmation
        if ctx.tcp_handler:
            from .packets import build_chat_message
            msg = build_chat_message("*PEW*", source_id=ctx.session.player_id or ctx.entity_id)
            ctx.tcp_handler.send(msg)

    def _send_projectile_spawn(self, ctx: ClientContext, proj, addr: tuple):
        """Send packet to spawn a projectile entity."""
        # Build UPDATE_ARRAY packet to create the projectile entity
        tick = self._get_network_tick(ctx)
        packet_local = build_projectile_spawn_packet(
            proj,
            tick,
            include_local_state=self.projectile_local_stats,
            weapon_id=self.weapon_id,
            health=1.0,
            fuel=1.0,
            entity_config=self.projectile_config,
            is_static=self.projectile_static,
        )
        packet_remote = packet_local
        if self.projectile_local_stats:
            packet_remote = build_projectile_spawn_packet(
                proj,
                tick,
                include_local_state=False,
                weapon_id=self.weapon_id,
                health=1.0,
                fuel=1.0,
                entity_config=self.projectile_config,
                is_static=self.projectile_static,
            )

        # Send via UDP
        sent_count = 0
        if self.udp_handler:
            for target in self._snapshot_in_game_clients():
                if not target.session.udp_addr or not target.session.translation_ack_received:
                    continue
                packet = packet_local if target is ctx else packet_remote
                self.udp_handler.send_to(packet, target.session.udp_addr)
                sent_count += 1
        if sent_count:
            print(f"[WEAPON] Sent projectile spawn via UDP: id={proj.entity_id} targets={sent_count}")
        if self.debug_projectiles:
            print(f"[PROJ-SPAWN] id={proj.entity_id} type={proj.entity_type} config={self.projectile_config} static={int(self.projectile_static)}")
            if os.environ.get("WULFRAM_DEBUG_PROJECTILE_HEX", "0") == "1":
                print(f"[PROJ-HEX] id={proj.entity_id} len={len(packet_local)} hex={packet_local.hex()}")

        # Send chat message with projectile position for debugging
        pos_msg = f"FIRE! pos=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f}) vel=({proj.vel[0]:.1f},{proj.vel[1]:.1f},{proj.vel[2]:.1f})"
        chat_packet = build_chat_message(pos_msg, source_id=ctx.session.player_id or ctx.entity_id)
        if self.udp_handler and addr:
            self.udp_handler.send_to(chat_packet, addr)
        print(f"[WEAPON] {pos_msg}")
        if self.debug_projectiles and proj.debug_context:
            hp_shape = proj.debug_context.get("hardpoint_shape")
            hp_raw = proj.debug_context.get("hardpoint_raw")
            hp_local = proj.debug_context.get("hardpoint_local")
            hp_order = proj.debug_context.get("hardpoint_order")
            hp_scale = proj.debug_context.get("shape_scale")
            hp_world = proj.debug_context.get("hardpoint_world_offset")
            muzzle_push = proj.debug_context.get("muzzle_push")
            hp_origin = proj.debug_context.get("hardpoint_origin_mode")
            hp_fsign = proj.debug_context.get("hardpoint_forward_sign")
            hp_rsign = proj.debug_context.get("hardpoint_right_sign")
            hp_usign = proj.debug_context.get("hardpoint_up_sign")
            hp_swap = proj.debug_context.get("hardpoint_swap_fr")
            if hp_shape and hp_raw and hp_local:
                raw_fmt = f"({hp_raw[0]},{hp_raw[1]},{hp_raw[2]})"
                loc_fmt = f"({hp_local[0]:.2f},{hp_local[1]:.2f},{hp_local[2]:.2f})"
                world_fmt = "(n/a)"
                if hp_world:
                    world_fmt = f"({hp_world[0]:.2f},{hp_world[1]:.2f},{hp_world[2]:.2f})"
                print(
                    f"[HARDPOINT] shape={hp_shape} name={ctx.weapon_system.projectile_hardpoint_name} "
                    f"raw={raw_fmt} order={hp_order} scale={hp_scale} local={loc_fmt} "
                    f"world={world_fmt} origin={hp_origin} "
                    f"f_sign={hp_fsign} r_sign={hp_rsign} u_sign={hp_usign} swap_fr={int(bool(hp_swap))} "
                    f"muzzle_push={muzzle_push}"
                )

    def _update_player_heading(self, ctx: ClientContext):
        """Track player movement heading from TURNING input (slot 1)."""
        import math

        def _normalize_axis(val: float) -> float:
            # Inputs decoded from ACTION_* packets appear to already be in -1..1.
            # Only scale down if we ever see large quantized values (e.g., +/-1000).
            if val > 1.5 or val < -1.5:
                scale = getattr(ctx.weapon_system, "control_max", 1000.0) or 1000.0
                return max(-1.0, min(1.0, val / scale))
            return max(-1.0, min(1.0, val))

        # Slot 1 is TURNING (mouse/left-right) per empirical input mapping.
        turn_val = ctx.weapon_system.behavior_slots[BehaviorSlot.TURNING]
        turn_input = _normalize_axis(turn_val)
        if abs(turn_input) < self.turn_deadzone:
            turn_input = 0.0

        # Turn physics from BEHAVIOR packet (approximate)
        turn_adjust = self.turn_adjust

        # Use actual time delta for turn integration (clamped).
        now = time.monotonic()
        dt = now - ctx.last_heading_update
        ctx.last_heading_update = now
        dt = min(dt, 0.1)  # Clamp to avoid huge jumps

        # Apply turn rate to movement heading (not aim yaw)
        ctx.player_heading += turn_input * turn_adjust * dt

        # Keep heading in -pi to pi range
        while ctx.player_heading > math.pi:
            ctx.player_heading -= 2 * math.pi
        while ctx.player_heading < -math.pi:
            ctx.player_heading += 2 * math.pi

        # Body yaw always follows movement heading.
        ctx.player_yaw = ctx.player_heading
        ctx.player_pose["yaw"] = ctx.player_heading

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

    def _update_player_position(self, ctx: ClientContext):
        """
        Simulate player position based on movement input (slot 2 forward, slot 3 strafe).
        Uses actual time delta for accurate physics simulation.
        """
        import math

        def _normalize_axis(val: float) -> float:
            # Inputs decoded from ACTION_* packets appear to already be in -1..1.
            # Only scale down if we ever see large quantized values (e.g., +/-1000).
            if val > 1.5 or val < -1.5:
                scale = getattr(ctx.weapon_system, "control_max", 1000.0) or 1000.0
                return max(-1.0, min(1.0, val / scale))
            return max(-1.0, min(1.0, val))

        # Calculate actual time delta
        now = time.monotonic()
        dt = now - ctx.last_position_update
        ctx.last_position_update = now

        # Clamp dt to reasonable range (avoid huge jumps)
        dt = min(dt, 0.1)  # Max 100ms

        # Slot 2 = moving_forward, slot 3 = moving_sideways (strafe)
        throttle_val = ctx.weapon_system.behavior_slots[BehaviorSlot.MOVING_FORWARD]
        strafe_val = ctx.weapon_system.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS]
        throttle_input = _normalize_axis(throttle_val)
        strafe_input = _normalize_axis(strafe_val)

        # Deadzone
        if abs(throttle_input) < 0.05:
            throttle_input = 0.0
        if abs(strafe_input) < 0.05:
            strafe_input = 0.0

        # Tank physics from BEHAVIOR packet (approximate values)
        move_adjust = 85.0   # Forward/back speed scaling
        strafe_adjust = 69.7  # Lateral speed scaling
        max_velocity = 80.0  # Speed cap

        yaw = ctx.player_heading
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        if self.up_axis == "z":
            forward = (cos_yaw, sin_yaw, 0.0)
            right = (-sin_yaw, cos_yaw, 0.0)
            vertical_idx = 2
        else:
            forward = (cos_yaw, 0.0, sin_yaw)
            right = (-sin_yaw, 0.0, cos_yaw)
            vertical_idx = 1

        fwd_speed = throttle_input * move_adjust
        strafe_speed = strafe_input * strafe_adjust

        vx = forward[0] * fwd_speed + right[0] * strafe_speed
        vy = forward[1] * fwd_speed + right[1] * strafe_speed
        vz = forward[2] * fwd_speed + right[2] * strafe_speed

        # Preserve vertical velocity component (jump jets) and apply gravity
        gravity = -50.0  # Downward acceleration (units/s^2)
        ground_level = 5.0  # Approximate ground height

        if vertical_idx == 2:
            # Z is up
            current_z = ctx.player_pos[2]
            current_vz = ctx.player_vel[2]

            # Apply gravity to vertical velocity
            new_vz = current_vz + gravity * dt

            # Ground collision - stop falling if at ground level
            if current_z <= ground_level and new_vz < 0:
                new_vz = 0.0

            vz = new_vz
        else:
            # Y is up
            current_y = ctx.player_pos[1]
            current_vy = ctx.player_vel[1]

            new_vy = current_vy + gravity * dt
            if current_y <= ground_level and new_vy < 0:
                new_vy = 0.0

            vy = new_vy

        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed > max_velocity and speed > 0.0:
            scale = max_velocity / speed
            vx *= scale
            vy *= scale
            vz *= scale

        x, y, z = ctx.player_pos
        new_x = x + vx * dt
        new_y = y + vy * dt
        new_z = z + vz * dt

        # Clamp to reasonable world bounds
        if self.up_axis == "z":
            new_x = max(-4000.0, min(4000.0, new_x))
            new_y = max(-4000.0, min(4000.0, new_y))
            # Clamp to ground level
            new_z = max(ground_level, new_z)
        else:
            new_x = max(-4000.0, min(4000.0, new_x))
            new_z = max(-4000.0, min(4000.0, new_z))
            # Clamp to ground level
            new_y = max(ground_level, new_y)

        old_pos = ctx.player_pos
        ctx.player_pos = (new_x, new_y, new_z)
        ctx.player_vel = (vx, vy, vz)
        ctx.player_pose["pos"] = ctx.player_pos
        ctx.player_pose["vel"] = ctx.player_vel

        # Log position changes periodically
        dist = math.sqrt(
            (new_x - old_pos[0]) ** 2 +
            (new_y - old_pos[1]) ** 2 +
            (new_z - old_pos[2]) ** 2
        )
        if dist > 10.0:
            print(f"[POS] Client {ctx.client_id} at ({new_x:.1f}, {new_y:.1f}, {new_z:.1f}) yaw={math.degrees(yaw):.1f} deg")


    def _spawn_moving_projectile(self, ctx: ClientContext, proj, addr: tuple):
        """Spawn a projectile and start sending movement updates."""
        import time
        from .weapons import build_projectile_update_packet

        # Convert to client/world coordinates once so spawn + updates stay aligned.
        server_pos = proj.pos
        if self.debug_projectiles:
            self._log_projectile_aim(ctx, proj, server_pos)
        proj.pos = self._to_client_pos(server_pos)

        # Send initial spawn packet
        self._send_projectile_spawn(ctx, proj, addr)

        # Start background thread for movement updates
        def update_loop():
            # === Projectile update mode ===
            # Mode 0: No updates (let client simulate from spawn velocity)
            # Mode 1: Low rate updates (5 Hz)
            # Mode 2: Medium rate updates (15 Hz)
            # Mode 3: High rate updates (30 Hz)
            update_mode = 2  # 15 Hz updates

            if update_mode == 0:
                # No updates - just wait for lifetime then clean up
                time.sleep(proj.lifetime)
                print(f"[PROJ] id={proj.entity_id} expired (no-update mode)")
                with ctx.projectile_lock:
                    if proj in ctx.active_projectiles:
                        ctx.active_projectiles.remove(proj)
                return

            update_rate = {1: 5.0, 2: 15.0, 3: 30.0}.get(update_mode, 15.0)
            dt = 1.0 / update_rate
            duration = proj.lifetime

            updates = int(duration * update_rate)
            for i in range(updates):
                time.sleep(dt)

                # Check if projectile still active
                with ctx.projectile_lock:
                    if proj not in ctx.active_projectiles:
                        break

                # Send position update
                tick = self._get_network_tick(ctx)
                update_pkt_local = build_projectile_update_packet(
                    proj,
                    tick,
                    dt,
                    include_local_state=self.projectile_local_stats,
                    weapon_id=self.weapon_id,
                    health=1.0,
                    fuel=1.0,
                )
                update_pkt_remote = update_pkt_local
                if self.projectile_local_stats:
                    update_pkt_remote = build_projectile_update_packet(
                        proj,
                        tick,
                        dt,
                        include_local_state=False,
                        weapon_id=self.weapon_id,
                        health=1.0,
                        fuel=1.0,
                    )

                if self.udp_handler:
                    for target in self._snapshot_in_game_clients():
                        if not target.session.udp_addr or not target.session.translation_ack_received:
                            continue
                        pkt = update_pkt_local if target is ctx else update_pkt_remote
                        self.udp_handler.send_to(pkt, target.session.udp_addr)

                if i % 15 == 0:  # Log every 0.5 sec at 30Hz
                    print(f"[PROJ] id={proj.entity_id} pos=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f}) vel=({proj.vel[0]:.0f},{proj.vel[1]:.0f},{proj.vel[2]:.0f}) tick={tick}")

            # Remove from active list when done
            with ctx.projectile_lock:
                if proj in ctx.active_projectiles:
                    ctx.active_projectiles.remove(proj)
            print(f"[PROJ] id={proj.entity_id} expired")

        # Add to active list and start thread
        with ctx.projectile_lock:
            ctx.active_projectiles.append(proj)

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

    def _get_aim_rotation(self, ctx: ClientContext) -> tuple:
        """Return (pitch, yaw, source) for aiming/projectiles."""
        now = time.monotonic()
        aim_recent = (now - ctx.player_aim_time) < self.viewpoint_timeout
        if ctx.player_aim_source == "viewpoint" and aim_recent:
            return ctx.player_aim_pitch, ctx.player_aim_yaw, "viewpoint"
        if ctx.player_aim_source == "slot" and (now - ctx.player_aim_time) < self.aim_hold_time:
            return ctx.player_aim_pitch, ctx.player_aim_yaw, "slot"
        return 0.0, ctx.player_heading, "input"

    def _log_projectile_aim(self, ctx: ClientContext, proj, server_pos: tuple):
        """Log detailed aim/pose context for projectile alignment debugging."""
        def _fmt_vec(vec: Optional[tuple]) -> str:
            if not vec:
                return "(none)"
            return f"({vec[0]:.2f},{vec[1]:.2f},{vec[2]:.2f})"

        def _wrap_angle(angle: float) -> float:
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
            return angle

        pitch, yaw, source = self._get_aim_rotation(ctx)
        viewpoint_recent = (source == "viewpoint")
        roll = ctx.player_pose.get("roll", 0.0)
        heading = ctx.player_heading
        delta = _wrap_angle(yaw - heading)

        slots = ctx.weapon_system.behavior_slots
        turn = slots[BehaviorSlot.TURNING]
        fwd_input = slots[BehaviorSlot.MOVING_FORWARD]
        strafe = slots[BehaviorSlot.MOVING_SIDEWAYS]
        thrust = slots[BehaviorSlot.UPWARD_THRUST]
        slot6 = slots[BehaviorSlot.SLOT6]
        slot7 = slots[BehaviorSlot.SLOT7]
        fire = slots[BehaviorSlot.FIRE]

        player_pos = ctx.player_pos
        player_pos_client = self._to_client_pos(player_pos)
        proj_pos_client = self._to_client_pos(server_pos)

        print(
            f"[PROJ-AIM] id={proj.entity_id} src={source} vp_recent={int(viewpoint_recent)} "
            f"up={self.up_axis} use_pitch={int(ctx.weapon_system.use_pitch)} "
            f"pos_offset={self.pos_offset:.1f}"
        )
        print(
            f"[PROJ-AIM] heading={math.degrees(heading):.1f} yaw={math.degrees(yaw):.1f} "
            f"pitch={math.degrees(pitch):.1f} roll={math.degrees(roll):.1f} "
            f"delta={math.degrees(delta):.1f}"
        )
        print(
            f"[PROJ-AIM] player_srv={_fmt_vec(player_pos)} player_cli={_fmt_vec(player_pos_client)} "
            f"proj_srv={_fmt_vec(server_pos)} proj_cli={_fmt_vec(proj_pos_client)}"
        )
        print(f"[PROJ-AIM] vel={_fmt_vec(proj.vel)}")

        debug = getattr(proj, "debug_context", None) or {}
        fwd_vec = None
        spawn_offset = 0.0
        if debug:
            fwd_vec = debug.get("forward")
            speed = debug.get("speed", 0.0)
            spawn_offset = debug.get("spawn_offset", 0.0)
            aim_yaw = debug.get("yaw", yaw)
            aim_pitch = debug.get("pitch", pitch)
            aim_source = debug.get("aim_source")
            aim_yaw_offset = debug.get("aim_yaw_offset_deg", 0.0)
            aim_pitch_offset = debug.get("aim_pitch_offset_deg", 0.0)
            aim_yaw_invert = int(bool(debug.get("aim_yaw_invert", False)))
            aim_pitch_invert = int(bool(debug.get("aim_pitch_invert", False)))
            print(
                f"[PROJ-AIM] aim_used yaw={math.degrees(aim_yaw):.1f} "
                f"pitch={math.degrees(aim_pitch):.1f} fwd={_fmt_vec(fwd_vec)} "
                f"speed={speed:.1f} spawn_offset={spawn_offset:.1f}"
            )
            if aim_source:
                print(f"[PROJ-AIM] aim_source={aim_source}")
            # Offset decomposition vs forward/right/up for quick mirror diagnosis.
            if proj_pos_client and player_pos_client:
                dx = proj_pos_client[0] - player_pos_client[0]
                dy = proj_pos_client[1] - player_pos_client[1]
                dz = proj_pos_client[2] - player_pos_client[2]
                if self.up_axis == "z":
                    right = (-math.sin(aim_yaw), math.cos(aim_yaw), 0.0)
                    up = (0.0, 0.0, 1.0)
                else:
                    right = (-math.sin(aim_yaw), 0.0, math.cos(aim_yaw))
                    up = (0.0, 1.0, 0.0)
                fwd_dir = fwd_vec if fwd_vec else (0.0, 0.0, 0.0)
                dot_fwd = dx * fwd_dir[0] + dy * fwd_dir[1] + dz * fwd_dir[2]
                dot_right = dx * right[0] + dy * right[1] + dz * right[2]
                dot_up = dx * up[0] + dy * up[1] + dz * up[2]
                print(
                    f"[PROJ-AIM] offset dfwd={dot_fwd:.2f} dright={dot_right:.2f} dup={dot_up:.2f}"
                )
            print(
                f"[PROJ-AIM] aim_tune yaw_off={aim_yaw_offset:.1f} pitch_off={aim_pitch_offset:.1f} "
                f"yaw_inv={aim_yaw_invert} pitch_inv={aim_pitch_invert}"
            )

        # Compare against last sent player state (client-facing) to detect desync.
        last = ctx.last_sent_player_state
        if last:
            sent_pos = last.get("pos")
            sent_rot = last.get("rot")
            dt = time.monotonic() - last.get("time", 0.0)
            if sent_pos and fwd_vec:
                exp_x = sent_pos[0] + spawn_offset * fwd_vec[0]
                exp_y = sent_pos[1] + spawn_offset * fwd_vec[1]
                exp_z = sent_pos[2] + spawn_offset * fwd_vec[2]
                dx = proj_pos_client[0] - exp_x
                dy = proj_pos_client[1] - exp_y
                dz = proj_pos_client[2] - exp_z
                err = math.sqrt(dx * dx + dy * dy + dz * dz)
                print(
                    f"[PROJ-AIM] sent_pos=({_fmt_vec(sent_pos)}) sent_rot=({_fmt_vec(sent_rot)}) "
                    f"dt={dt:.3f}s exp_spawn=({exp_x:.2f},{exp_y:.2f},{exp_z:.2f}) "
                    f"err={err:.2f}"
                )

        print(
            f"[PROJ-AIM] input turn={turn:.3f} fwd={fwd_input:.3f} strafe={strafe:.3f} "
            f"thrust={thrust:.3f} s6={slot6:.3f} s7={slot7:.3f} fire={fire:.3f}"
        )

    # ============ Jump Jet System Handlers ============

    def _process_jump_jets(self, ctx: ClientContext, addr: tuple):
        """
        Process jump jet input from behavior slot 5.
        Called after decoding ACTION_DUMP or ACTION_UPDATE.
        """
        # Get slot 5 value (upward thrust / Q/Z key)
        slot5_value = ctx.weapon_system.behavior_slots[BehaviorSlot.UPWARD_THRUST]

        # Get player/entity info
        player_id = ctx.session.player_id or ctx.entity_id
        entity_type = 0  # Tank (could track per-player vehicle type)

        if self.up_axis == "z":
            current_pos_up = ctx.player_pos[2]
            current_vel_up = ctx.player_vel[2]
        else:
            current_pos_up = ctx.player_pos[1]
            current_vel_up = ctx.player_vel[1]

        # Process jump jet input
        impulse, cooldown_ready = ctx.jump_jet_system.process_input(
            player_id=player_id,
            slot5_value=slot5_value,
            entity_type=entity_type,
            current_pos_z=current_pos_up,
            current_vel_z=current_vel_up,
            current_energy=ctx.player_energy
        )

        if impulse > 0:
            # Apply vertical impulse to player velocity
            if self.up_axis == "z":
                ctx.player_vel = (
                    ctx.player_vel[0],
                    ctx.player_vel[1],
                    ctx.player_vel[2] + impulse
                )
            else:
                ctx.player_vel = (
                    ctx.player_vel[0],
                    ctx.player_vel[1] + impulse,
                    ctx.player_vel[2]
                )

            # Consume fuel (if using energy system)
            fuel_cost = ctx.jump_jet_system.get_fuel_cost(entity_type)
            if fuel_cost > 0:
                ctx.player_energy = max(0.0, ctx.player_energy - fuel_cost)

            # Send position/velocity update to client
            self._send_jump_velocity_update(ctx, addr)

    def _on_jump_jet_triggered(self, ctx: ClientContext, player_id: int, impulse: float, new_vel_z: float):
        """Callback when a jump jet is triggered."""
        print(f"[JUMP] Jump triggered for player {player_id}: impulse={impulse}, vel_z={new_vel_z:.1f}")

        # Send visual/audio feedback via chat (until we implement proper effects)
        if ctx.tcp_handler:
            msg = build_chat_message("*WHOOSH*", source_id=player_id)
            ctx.tcp_handler.send(msg)

    def _send_jump_velocity_update(self, ctx: ClientContext, addr: tuple):
        """Send velocity update to client after jump."""
        from .packets import get_ticks

        # Build UPDATE_ARRAY with new velocity
        # Using the same format as the heartbeat but with velocity data
        packet = self._build_velocity_update_packet(ctx)

        # Send via UDP for low latency
        if self.udp_handler and addr:
            self.udp_handler.send_to(packet, addr)
            vel_up = ctx.player_vel[2] if self.up_axis == "z" else ctx.player_vel[1]
            print(f"[JUMP] Sent velocity update: vel_up={vel_up:.1f} to {addr}")

        # Also send via TCP for reliability
        if ctx.tcp_handler:
            ctx.tcp_handler.send(packet)

    def _build_velocity_update_packet(self, ctx: ClientContext) -> bytes:
        """Build UPDATE_ARRAY packet with current velocity."""
        from .packets import _compress_value
        from .codec import BitWriter

        tick = self._get_network_tick(ctx)
        tick_bytes = struct.pack(">I", tick)

        bw = BitWriter()

        # CRITICAL: Must have local stats flag first!
        bw.write_bits(1, 0)  # No local stats

        # Header: 1 entity
        bw.write_bits(8, 1)

        # Entity OID
        entity_id = ctx.session.entity_id or ctx.entity_id
        bw.write_bits(32, entity_id)

        # Is manned = True (player vehicle)
        bw.write_bits(1, 1)

        # Presence flags: bit 2 = velocity update
        # bits: 0=creation, 1=position, 2=velocity, 3=rotation, etc.
        presence_flags = 0b0000000100  # Just velocity (bit 2)
        bw.write_bits(10, presence_flags)

        # Bank selector
        bw.write_bits(16, 0)

        # Velocity vector using wulf-forge format (4-bit header + 16-bit values)
        # VEC_VEL: max=200, range=400
        bw.write_bits(4, 15)  # Header (max precision)
        for v in ctx.player_vel:
            bw.write_bits(16, _compress_value(v, 200.0, 400.0, total_bits=16))

        return b'\x0E' + tick_bytes + bw.get_bytes()

    def _tick_loop(self, ctx: ClientContext):
        """Game tick loop - sends UPDATE_ARRAY periodically."""
        print(f"[TICK] Starting tick loop for client {ctx.client_id}")
        tcp_failed = False
        tick_start_time = time.monotonic()
        logged_wait_translation = False
        logged_wait_client_tick = False
        grace_period_logged = False
        grace_period_end = None  # Set when client is ready

        # Desync detection state
        last_position = ctx.player_pos
        last_position_change_time = time.monotonic()
        last_input_time = time.monotonic()
        desync_warned = False

        while ctx.running and ctx.session.in_game:
            try:
                ctx.session.tick += 1

                # Wait until client has quantizers and at least one input tick.
                # This avoids mis-decoding local stats before TRANSLATION is applied.
                if not ctx.session.translation_ack_received:
                    if not logged_wait_translation and time.monotonic() - tick_start_time < 5.0:
                        print(f"[TICK] Client {ctx.client_id}: Waiting for TRANSLATION_ACK before sending UPDATE_ARRAY")
                        logged_wait_translation = True
                    time.sleep(0.05)
                    continue

                if ctx.last_client_tick <= 0:
                    if not logged_wait_client_tick and time.monotonic() - tick_start_time < 5.0:
                        print(f"[TICK] Client {ctx.client_id}: Waiting for client input tick before sending UPDATE_ARRAY")
                        logged_wait_client_tick = True
                    time.sleep(0.01)
                    continue

                # Grace period: delay UPDATE_ARRAY sends for 3 seconds after client is ready
                # This lets the client establish local control before server authority kicks in
                if grace_period_end is None:
                    grace_period_end = time.monotonic() + 3.0
                    print(f"[TICK] Client {ctx.client_id}: Starting 3s grace period before sending position updates")
                if time.monotonic() < grace_period_end:
                    if not grace_period_logged:
                        remaining = grace_period_end - time.monotonic()
                        print(f"[TICK] Client {ctx.client_id}: Grace period ({remaining:.1f}s remaining)")
                        grace_period_logged = True
                    time.sleep(0.05)
                    continue

                # Once translation is ready, make sure this client sees others and vice versa.
                self._ensure_multiplayer_visibility(ctx)

                # Update simulated heading/position from inputs
                self._update_player_heading(ctx)
                self._update_player_aim(ctx)
                self._update_player_position(ctx)

                # Desync detection: track position changes and input
                now = time.monotonic()
                pos_changed = (
                    abs(ctx.player_pos[0] - last_position[0]) > 0.1 or
                    abs(ctx.player_pos[1] - last_position[1]) > 0.1 or
                    abs(ctx.player_pos[2] - last_position[2]) > 0.1
                )
                if pos_changed:
                    last_position = ctx.player_pos
                    last_position_change_time = now
                    desync_warned = False

                # Check for non-zero input in behavior slots (only movement, not thrust)
                fwd_input = ctx.weapon_system.behavior_slots[BehaviorSlot.MOVING_FORWARD]
                strafe_input = ctx.weapon_system.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS]
                thrust_input = ctx.weapon_system.behavior_slots[BehaviorSlot.UPWARD_THRUST]
                # Only consider actual movement input (fwd/strafe), not thrust which may be constant
                has_movement_input = abs(fwd_input) > 0.05 or abs(strafe_input) > 0.05
                if has_movement_input:
                    last_input_time = now

                # Warn if position stuck but receiving movement input
                stuck_duration = now - last_position_change_time
                movement_input_active = (now - last_input_time) < 2.0
                if stuck_duration > 5.0 and movement_input_active and not desync_warned:
                    print(f"[DESYNC] Client {ctx.client_id}: Position stuck for {stuck_duration:.1f}s but movement input active!")
                    print(f"[DESYNC]   pos={ctx.player_pos} vel={ctx.player_vel}")
                    print(f"[DESYNC]   fwd={fwd_input:.3f} strafe={strafe_input:.3f}")
                    desync_warned = True

                # Periodic status for debugging (every 30 seconds)
                if ctx.session.tick % 900 == 0:  # ~30 seconds at 30Hz
                    input_status = "IDLE" if (abs(fwd_input) < 0.01 and abs(strafe_input) < 0.01) else "ACTIVE"
                    print(f"[STATUS] Client {ctx.client_id}: pos={ctx.player_pos} input={input_status}(fwd={fwd_input:.2f},strafe={strafe_input:.2f}) stuck={stuck_duration:.0f}s")

                tick = self._get_network_tick(ctx)
                # Debug: log tick value periodically
                if ctx.session.tick % 300 == 0:
                    print(f"[TICK-DEBUG] Client {ctx.client_id}: network_tick={tick} client_tick={ctx.last_client_tick} offset={ctx.tick_offset}")
                # Full position updates (server authoritative)
                send_full_update = True

                if send_full_update:
                    # Send UPDATE_ARRAY with position/velocity
                    send_pos = self._to_client_pos(ctx.player_pos)
                    weapon_type = self._get_local_state_weapon_type(ctx)
                    ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
                    pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
                    payload = build_update_array_player_update(
                        tick,
                        ctx.session.entity_id,
                        pos=send_pos,
                        vel=ctx.player_vel,
                        # Rotation vector order follows wulf-forge: (roll, pitch, yaw)
                        rot=(
                            ctx.player_pose.get("roll", 0.0),
                            0.0,
                            ctx.player_yaw,
                        ),
                        include_local_state=self.update_local_state,
                        include_entity_vitals=False,
                        weapon_id=weapon_type,
                        health=1.0,
                        fuel=1.0,
                        speed_scale=1.0,
                        ammo_count_bits=ammo_bits,
                        ammo_count=ammo_mask,
                        primary_turret_bits=pt_bits,
                        primary_turret_angle=pt_angle,
                        secondary_turret_bits=st_bits,
                        secondary_turret_angle=st_angle,
                        turret_max=self.local_state_turret_max,
                        turret_range=self.local_state_turret_range,
                    )
                else:
                    # Heartbeat only (health/energy) - no position to avoid rubber-banding
                    weapon_type = self._get_local_state_weapon_type(ctx)
                    ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
                    pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
                    payload = build_update_array_heartbeat(
                        tick,
                        ctx.session.entity_id,
                        include_health=self.update_local_state,
                        weapon_id=weapon_type,
                        health=1.0,
                        fuel=1.0,
                        ammo_count_bits=ammo_bits,
                        ammo_count=ammo_mask,
                        primary_turret_bits=pt_bits,
                        primary_turret_angle=pt_angle,
                        secondary_turret_bits=st_bits,
                        secondary_turret_angle=st_angle,
                        turret_max=self.local_state_turret_max,
                        turret_range=self.local_state_turret_range,
                    )

                # Try TCP first, fall back to UDP if TCP fails
                # Client may close TCP after spawn and use UDP only
                if self.send_updates_tcp and not tcp_failed and ctx.tcp_handler:
                    try:
                        ctx.tcp_handler.send(payload, log=False)
                    except Exception as tcp_err:
                        print(f"[TICK] Client {ctx.client_id}: TCP failed ({tcp_err}), switching to UDP-only")
                        tcp_failed = True

                # Always send via UDP as well for reliability
                if self.send_updates_udp and self.udp_handler and ctx.session.udp_addr:
                    self.udp_handler.send_to(payload, ctx.session.udp_addr)

                # Periodic TankPacket vitals refresh to stabilize HUD health/energy.
                if self.tank_vitals and self.tank_vitals_heartbeat and ctx.session.udp_addr:
                    now = time.monotonic()
                    if (now - ctx.last_vitals_send) >= self.tank_vitals_interval:
                        ctx.last_vitals_send = now
                        vitals_packet = build_udp_tank_packet_wf(
                            net_id=ctx.session.entity_id,
                            unit_type=ctx.entity_type,
                            team_id=ctx.session.team_id or 1,
                            pos=self._to_client_pos(ctx.player_pos),
                            vel=ctx.player_vel,
                            include_vitals=True,
                            weapon_id=self.weapon_id,
                            health_mult_bits=1,
                            energy_mult_bits=1,
                        )
                        if self.udp_handler and ctx.session.udp_addr:
                            self.udp_handler.send_to(vitals_packet, ctx.session.udp_addr)

                # Send other players' transforms to this client (multiplayer visibility).
                if self.send_player_updates:
                    self._send_remote_player_updates(
                        ctx,
                        tick,
                        prefer_tcp=(self.send_updates_tcp and not tcp_failed),
                    )

                # Track last sent player state for projectile alignment diagnostics.
                # Use player_pos if send_pos not set (heartbeat-only mode)
                track_pos = send_pos if send_full_update else self._to_client_pos(ctx.player_pos)
                ctx.last_sent_player_state = {
                    "time": time.monotonic(),
                    "tick": tick,
                    "pos": track_pos,
                    "rot": (
                        ctx.player_pose.get("roll", 0.0),
                        0.0,
                        ctx.player_yaw,
                    ),
                    "vel": ctx.player_vel,
                }

                # Log every 10 ticks to trace movement + health sends
                if ctx.session.tick % 10 == 0:
                    px, py, pz = ctx.player_pos
                    vx, vy, vz = ctx.player_vel
                    yaw_deg = math.degrees(ctx.player_yaw)
                    print(f"[TICK] Client {ctx.client_id}: pos=({px:.2f},{py:.2f},{pz:.2f}) vel=({vx:.2f},{vy:.2f},{vz:.2f}) yaw={yaw_deg:.1f}")
                    aim_recent = (time.monotonic() - ctx.player_aim_time) < self.viewpoint_timeout
                    if not aim_recent:
                        print(f"[VIEWPOINT-INPUT] yaw={yaw_deg:.1f}")
                    pkt_type = "FULL" if send_full_update else "BEAT"
                    udp_addr = ctx.session.udp_addr if ctx.session.udp_addr else "NO_ADDR"
                    # Verify health encoding in payload
                    health_hex = payload[5:8].hex() if len(payload) > 7 else "??"
                    print(f"[TICK-HEALTH] t={ctx.session.tick} type={pkt_type} udp={udp_addr} health_bytes={health_hex}")

                time.sleep(0.033)  # 30 Hz (smoother updates)

            except Exception as e:
                print(f"[TICK] Client {ctx.client_id} Error: {e}")
                import traceback
                traceback.print_exc()
                # Don't break - try to continue even with errors
                time.sleep(0.1)

        ctx.tick_thread = None
        print(f"[TICK] Tick loop ended for client {ctx.client_id}")


def main():
    """Entry point."""
    server = WulframServer()
    server.start()


if __name__ == "__main__":
    main()
