"""
Control plane for live packet injection.

Allows sending arbitrary packets to the game client without restarting.
Connect via: nc localhost 2628 or python inject.py

Commands:
  raw <hex>              - Send raw bytes (e.g., "raw 17000005390001")
  send <type> [args]     - Send named packet (e.g., "send PLAYER 1337 spectator=false")
  state                  - Show current session state
  packets                - List available packet types
  phase <name>           - Force session phase transition
  flag <name> <0|1>      - Set feature flag
  turn [param] [value]   - Show/set turning parameters live
  physics [param] [value] - Show/set yaw physics params (damp, reset)
  help                   - Show commands
  quit                   - Disconnect
"""

import socket
import struct
import threading
import shlex
import time
from typing import Optional, Callable, Dict, Any

from .physics import _matrix3_from_euler_xyz
from .session import Session, Phase, FEATURES
from .packets import (
    PacketType, get_packet_name,
    build_player, build_team_info, build_world_stats,
    build_reincarnate, build_add_to_roster, build_update_stats,
    build_birth_notice, build_chat_message, build_player_info,
    build_update_array_empty, build_update_array_heartbeat,
    build_update_array_player_update, build_view_update_player_update,
    build_update_array_multi,
    build_update_array_create_tank, build_update_array_spawn_points,
    build_behavior_packet, build_translation_packet,
    build_login_status, build_bps_response, build_game_clock,
    build_udp_tank_packet_wf,
    build_delete_object,
)


def build_input_sync_diagnosis(
    *,
    phase: str,
    last_input: dict,
    last_action_age_s: Optional[float],
    last_nonzero_move_input_age_s: Optional[float],
    last_position_change_age_s: Optional[float],
    last_state_request_age_s: Optional[float],
    last_state_sync_reply_age_s: Optional[float],
    state_requests: int,
    state_sync_replies: int,
    state_sync_view_replies: int,
    last_correction_age_s: Optional[float] = None,
    last_movement_correction_age_s: Optional[float] = None,
    movement_corrections: int = 0,
) -> dict:
    """Summarize whether a live OG sync issue is packet-side or input-side."""

    def _recent(age: Optional[float], limit: float) -> bool:
        return age is not None and age <= limit

    def _abs_input(name: str) -> float:
        try:
            return abs(float(last_input.get(name, 0.0)))
        except (TypeError, ValueError):
            return 0.0

    action_stream_active = _recent(last_action_age_s, 2.0)
    state_request_correction_packets_active = (
        state_requests > 0
        and state_sync_replies > 0
        and state_sync_view_replies > 0
        and _recent(last_state_request_age_s, 3.0)
        and _recent(last_state_sync_reply_age_s, 3.0)
    )
    correction_stream_active = (
        _recent(last_correction_age_s, 1.5)
        or (
            movement_corrections > 0
            and _recent(last_movement_correction_age_s, 1.5)
        )
    )
    correction_packets_active = state_request_correction_packets_active or correction_stream_active
    movement_input_recent = _recent(last_nonzero_move_input_age_s, 6.0)
    position_recent = _recent(last_position_change_age_s, 6.0)
    drive_idle = (
        _abs_input("fwd") <= 0.05
        and _abs_input("strafe") <= 0.05
        and _abs_input("turn") <= 0.05
    )

    if phase != "IN_GAME":
        status = "not_in_game"
    elif not action_stream_active:
        status = "no_recent_action_packets"
    elif correction_packets_active and movement_input_recent:
        status = "moving_with_correction_packets"
    elif correction_packets_active and drive_idle:
        status = "idle_input_correction_packets_active"
    elif action_stream_active and movement_input_recent and not correction_packets_active:
        status = "movement_without_targeted_corrections"
    elif action_stream_active and drive_idle:
        status = "idle_input_no_targeted_corrections"
    else:
        status = "undetermined"

    return {
        "status": status,
        "action_stream_active": action_stream_active,
        # This means authoritative correction-shaped packets are being emitted
        # or answered. It does not prove OG visibly moved its local camera/tank:
        # the decompiled local reconcile path is prediction/collision gated.
        "corrections_active": correction_packets_active,
        "correction_packets_active": correction_packets_active,
        "correction_application_verified": False,
        "state_request_corrections_active": state_request_correction_packets_active,
        "correction_stream_active": correction_stream_active,
        "movement_input_recent": movement_input_recent,
        "position_recent": position_recent,
        "drive_idle": drive_idle,
    }


def build_player_terrain_probe(server: Any, pos: tuple[float, float, float], heading_rad: float) -> dict:
    """Return decompile-grounded terrain probe data for live player telemetry.

    The probe mirrors the fields we care about for the current rough-terrain
    target: the active triangle-plane height/normal from
    `GUESS3_Terrain_interpolate_grid_height`, the physics ground Z used by the
    server movement path, and the tank's current clearance over that plane.
    """
    if server is None:
        return {}
    terrain = getattr(server, "terrain", None)
    if terrain is None:
        return {}

    import math

    x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
    physics_offset = float(
        getattr(server, "terrain_physics_height_offset", getattr(server, "terrain_height_offset", 0.0))
    )
    sample_height_normal = getattr(terrain, "sample_height_normal", None)
    if callable(sample_height_normal):
        raw_height, normal = sample_height_normal(x, y)
    else:
        raw_height = terrain.get_height(x, y)
        dh_dx, dh_dy = terrain.get_slope(x, y)
        mag_sq = dh_dx * dh_dx + dh_dy * dh_dy + 1.0
        if mag_sq <= 1e-12:
            normal = (0.0, 0.0, 1.0)
        else:
            inv_mag = 1.0 / math.sqrt(mag_sq)
            normal = (-dh_dx * inv_mag, -dh_dy * inv_mag, inv_mag)

    if abs(float(normal[2])) <= 1e-12:
        slope = (0.0, 0.0)
    else:
        slope = (-float(normal[0]) / float(normal[2]), -float(normal[1]) / float(normal[2]))

    cell = None
    cell_type = None
    try:
        gx = int(math.floor(x * float(terrain.inv_cell_x)))
        gy = int(math.floor(y * float(terrain.inv_cell_z)))
        gx = max(0, min(gx, int(terrain.num_x) - 2))
        gy = max(0, min(gy, int(terrain.num_z) - 2))
        cell = [gx, gy]
        get_cell_type = getattr(terrain, "get_cell_type", None)
        if callable(get_cell_type):
            cell_type = int(get_cell_type(gx, gy))
    except Exception:
        cell = None

    ground_z = float(raw_height) + physics_offset
    slope_heading = slope[0] * math.cos(heading_rad) + slope[1] * math.sin(heading_rad)
    return {
        "source": "GUESS3_Terrain_interpolate_grid_height",
        "cell": cell,
        "cell_type": cell_type,
        "raw_height": round(float(raw_height), 5),
        "physics_ground_z": round(ground_z, 5),
        "clearance_z": round(z - ground_z, 5),
        "normal": [round(float(v), 6) for v in normal],
        "slope": [round(float(v), 6) for v in slope],
        "slope_at_heading": round(float(slope_heading), 6),
        "pitch_at_heading_deg": round(math.degrees(math.atan(slope_heading)), 3),
    }


class ControlServer:
    """
    Control plane server for packet injection.
    Runs on a separate port and allows injecting packets into the live session.
    """

    def __init__(self, port: int = 2628):
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # These get set by the main server when a client connects
        self.tcp_handler = None  # TCPHandler for sending to game client
        self.udp_handler = None  # UDPHandler for UDP sends
        self.session: Optional[Session] = None
        self.ctx = None  # ClientContext for entity/roster info
        self.server = None  # Reference to main WulframServer for player_pos access

        # Packet builders registry
        self.builders: Dict[str, Callable] = {
            'PLAYER': self._build_player,
            'TEAM_INFO': lambda: build_team_info(),
            'WORLD_STATS': lambda: build_world_stats(),
            'REINCARNATE': self._build_reincarnate,
            'ADD_TO_ROSTER': self._build_add_to_roster,
            'UPDATE_STATS': self._build_update_stats,
            'BIRTH_NOTICE': self._build_birth_notice,
            'CHAT': self._build_chat,
            'PLAYER_INFO': self._build_player_info,
            'UPDATE_ARRAY_EMPTY': self._build_update_array_empty,
            'UPDATE_ARRAY_HEARTBEAT': self._build_update_array_heartbeat,
            'UPDATE_ARRAY_TANK': self._build_update_array_tank,
            'BEHAVIOR': lambda: build_behavior_packet(),
            'TRANSLATION': lambda: build_translation_packet(),
            'LOGIN_STATUS': self._build_login_status,
            'BPS_RESPONSE': self._build_bps_response,
            'GAME_CLOCK': self._build_game_clock,
        }

    def _get_active_client(self, *, require_udp: bool = True):
        """Find the first connected client from the game server's client list.

        Returns (ctx, addr) or (None, None). This works even when
        self.session/self.tcp_handler are stale (e.g. after server restart).
        """
        if not self.server:
            return None, None
        with self.server.clients_lock:
            for c in self.server.clients.values():
                if c.session and (not require_udp or c.session.udp_addr):
                    return c, c.session.udp_addr
        return None, None

    def _get_client_by_id(self, client_id: int, *, require_udp: bool = True):
        """Find a specific client by client_id. Returns (ctx, addr) or (None, None)."""
        if not self.server:
            return None, None
        with self.server.clients_lock:
            for c in self.server.clients.values():
                if c.client_id == client_id and c.session and (not require_udp or c.session.udp_addr):
                    return c, c.session.udp_addr
        return None, None

    def _apply_exact_client_pose(
        self,
        ctx,
        pos: tuple[float, float, float],
        heading_rad: float | None = None,
        vel: tuple[float, float, float] | None = None,
    ) -> None:
        """Apply an exact authoritative pose reset for a live client.

        This is stricter than respawn scheduling: it updates the current in-game
        authoritative position/heading immediately and clears motion/collision
        bookkeeping that would otherwise leak from the previous scenario.
        """
        import math

        now = time.monotonic()
        px, py, pz = (float(pos[0]), float(pos[1]), float(pos[2]))
        heading = float(ctx.player_heading if heading_rad is None else heading_rad)
        applied_vel = (
            (float(vel[0]), float(vel[1]), float(vel[2]))
            if vel is not None
            else (0.0, 0.0, 0.0)
        )

        ctx.player_pos = (px, py, pz)
        ctx.player_vel = applied_vel
        ctx.player_speed = math.sqrt(
            applied_vel[0] * applied_vel[0]
            + applied_vel[1] * applied_vel[1]
            + applied_vel[2] * applied_vel[2]
        )
        ctx.player_heading = heading
        ctx.player_yaw = -heading
        ctx.player_angular_vel = 0.0
        ctx.angular_vel_yaw = 0.0
        ctx.world_collision_ref_pos = ctx.player_pos
        ctx.world_collision_bounds_dirty = False
        ctx.last_state_sync_vel = None
        ctx.last_state_sync_rot = None
        ctx.last_sent_pos = ctx.player_pos
        ctx.last_sent_vel = applied_vel
        ctx.last_sent_yaw = -heading
        ctx.last_action_dump_time = now
        ctx.last_position_update = now
        ctx.last_heading_update = now
        ctx.control_pose_reset_time = now
        ctx.control_pose_reset_pos = ctx.player_pos
        if hasattr(ctx, "record_pose_reset"):
            ctx.record_pose_reset(
                "control_pos",
                pos=ctx.player_pos,
                vel=applied_vel,
                details={"heading_rad": heading},
            )
        ctx.player_aim_yaw = heading
        ctx.player_aim_pitch = 0.0
        ctx.player_aim_source = "control_pos"
        ctx.player_aim_time = now
        ctx.pending_respawn_pos = None
        ctx.injected_input = None
        ctx.injected_turn = None
        ctx.injected_jumpjet = None
        ctx.prev_raw_turn_input = 0.0
        if hasattr(ctx, "authoritative_state_history"):
            ctx.authoritative_state_history.clear()

        ctx.player_pose["pos"] = ctx.player_pos
        ctx.player_pose["vel"] = applied_vel
        ctx.player_pose["roll"] = 0.0
        ctx.player_pose["pitch"] = 0.0
        ctx.player_pose["yaw"] = -heading
        ctx.player_pose["last_tick"] = self.server._get_network_tick(ctx) if self.server else 0
        ctx.spring_body_ang_vel = (0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, heading)
        if self.server and hasattr(self.server, "_set_ground_level_override_for_pose"):
            self.server._set_ground_level_override_for_pose(ctx, ctx.player_pos)
        elif self.server and getattr(self.server, "spawn_sets_ground_level", False):
            if getattr(self.server, "up_axis", "z") == "z":
                ctx.ground_level_override = pz
            else:
                ctx.ground_level_override = py
            ctx.ground_override_ref_terrain_level = None
        else:
            ctx.ground_level_override = None
            ctx.ground_override_ref_terrain_level = None

        if ctx.weapon_system:
            from .weapons import BehaviorSlot, TANK_SOFTBODY_CONTROL_SLOT, tank_softbody_control_slot_value

            softbody_value = tank_softbody_control_slot_value(ctx.weapon_system.behavior_slots)
            ctx.weapon_system.player_pos = ctx.player_pos
            ctx.weapon_system.player_rot = (0.0, 0.0, heading)
            ctx.weapon_system.behavior_slots = [0.0] * len(ctx.weapon_system.behavior_slots)
            for slot_idx in {BehaviorSlot.UPWARD_THRUST, TANK_SOFTBODY_CONTROL_SLOT}:
                idx = int(slot_idx)
                if 0 <= idx < len(ctx.weapon_system.behavior_slots):
                    ctx.weapon_system.behavior_slots[idx] = softbody_value
            ctx.weapon_system.client_frame_counter = 0
            ctx.weapon_system.turn_input_change_time = 0.0
            ctx.weapon_system.turn_input_prev_value = 0.0
            ctx.weapon_system.turn_input_change_client_tick = 0
            ctx.weapon_system.prev_action_client_tick = 0
            ctx.weapon_system.prev_dump_tick = 0
            ctx.weapon_system.input_feedback_count = 0
            ctx.weapon_system.prev_fire_state = 0.0
            ctx.weapon_system.prev_direct_trigger_states = {
                slot_idx: 0.0 for slot_idx in ctx.weapon_system.prev_direct_trigger_states
            }
            ctx.weapon_system.fire_cooldown = 0.0
            ctx.weapon_system.last_update_time = now

        if ctx.vehicle_physics:
            ctx.vehicle_physics.reset()
            ctx.vehicle_physics.heading = heading
            ctx.vehicle_physics.angular_velocity = 0.0

        if hasattr(ctx, "debug_last_controller_step"):
            ctx.debug_last_controller_step.clear()
        if hasattr(ctx, "debug_last_spring_state"):
            ctx.debug_last_spring_state.clear()
        if hasattr(ctx, "debug_last_collision"):
            ctx.debug_last_collision.clear()

    def _sync_to_active_client(self):
        """Update self.session/tcp_handler to match the active game client."""
        ctx, _ = self._get_active_client()
        if ctx:
            self.ctx = ctx
            self.session = ctx.session
            self.tcp_handler = ctx.tcp_handler

    def start(self):
        """Start the control server in a background thread."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('127.0.0.1', self.port))
        self.sock.listen(5)
        self.sock.settimeout(1.0)
        self.running = True

        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
        print(f"[CONTROL] Listening on port {self.port}")

    def stop(self):
        """Stop the control server."""
        self.running = False
        if self.sock:
            self.sock.close()

    def _accept_loop(self):
        """Accept control connections."""
        while self.running:
            try:
                client, addr = self.sock.accept()
                print(f"[CONTROL] Connection from {addr}")
                handler = threading.Thread(
                    target=self._handle_client,
                    args=(client,),
                    daemon=True
                )
                handler.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[CONTROL] Accept error: {e}")

    def _handle_client(self, client: socket.socket):
        """Handle a control client connection."""
        client.settimeout(None)
        try:
            client.send(b"Wulfram Control Plane\n")
            client.send(b"Type 'help' for commands\n> ")

            buffer = ""
            while self.running:
                data = client.recv(1024)
                if not data:
                    break

                buffer += data.decode('utf-8', errors='ignore')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        response = self._handle_command(line)
                        client.send((response + "\n> ").encode())
                    else:
                        client.send(b"> ")

        except Exception as e:
            print(f"[CONTROL] Client error: {e}")
        finally:
            client.close()
            print("[CONTROL] Client disconnected")

    def _handle_command(self, line: str) -> str:
        """Process a control command."""
        try:
            parts = shlex.split(line)
        except ValueError as e:
            return f"Parse error: {e}"

        if not parts:
            return ""

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == 'help':
            return self._cmd_help()
        elif cmd == 'state':
            return self._cmd_state()
        elif cmd == 'packets':
            return self._cmd_packets()
        elif cmd == 'raw':
            return self._cmd_raw(args)
        elif cmd == 'send':
            return self._cmd_send(args)
        elif cmd == 'phase':
            return self._cmd_phase(args)
        elif cmd == 'flag':
            return self._cmd_flag(args)
        elif cmd == 'spawn_full':
            return self._cmd_spawn_full(args)
        elif cmd == 'enter_game' or cmd == 'spawn_now':
            return self._cmd_enter_game(args)
        elif cmd == 'join_team' or cmd == 'jt':
            return self._cmd_join_team(args)
        elif cmd == 'spawn_point' or cmd == 'sp':
            return self._cmd_spawn_point(args)
        elif cmd == 'spawn_udp' or cmd == 'spawn_wf':
            return self._cmd_spawn_udp(args)
        elif cmd == 'projectile':
            return self._cmd_projectile(args)
        elif cmd == 'pulse' or cmd == 'fire_pulse':
            return self._cmd_fire_pulse(args)
        elif cmd == 'projectile_move' or cmd == 'pmove':
            return self._cmd_projectile_move(args)
        elif cmd == 'test_vel' or cmd == 'tv':
            return self._cmd_test_velocity(args)
        elif cmd == 'shell' or cmd == 'sh':
            return self._cmd_shell(args)
        elif cmd == 'spread' or cmd == 'fan':
            return self._cmd_spread(args)
        elif cmd == 'pos' or cmd == 'player_pos':
            return self._cmd_player_pos(args)
        elif cmd == 'reset_pos' or cmd == 'rp':
            return self._cmd_reset_pos(args)
        elif cmd == 'spawn_entity' or cmd == 'entity':
            return self._cmd_spawn_entity(args)
        elif cmd == 'health':
            return self._cmd_send_health(args)
        elif cmd == 'energy' or cmd == 'fuel':
            return self._cmd_energy(args)
        elif cmd == 'spawn_points':
            return self._cmd_spawn_points(args)
        elif cmd == 'turn':
            return self._cmd_turn(args)
        elif cmd == 'barrel':
            return self._cmd_barrel(args)
        elif cmd == 'heading' or cmd == 'hdg':
            return self._cmd_heading(args)
        elif cmd == 'fire':
            return self._cmd_fire(args)
        elif cmd == 'pktlog':
            return self._cmd_pktlog(args)
        elif cmd == 'behavior' or cmd == 'beh':
            return self._cmd_behavior(args)
        elif cmd == 'physics' or cmd == 'phys':
            return self._cmd_physics(args)
        elif cmd == 'correction' or cmd == 'corr':
            return self._cmd_correction(args)
        elif cmd == 'subtick':
            if not self.server:
                return "No server"
            self.server.subtick_enabled = not self.server.subtick_enabled
            return f"Sub-tick interpolation: {'ON' if self.server.subtick_enabled else 'OFF'}"
        elif cmd == 'respawn' or cmd == 'rs':
            return self._cmd_respawn(args)
        elif cmd == 'shells':
            return self._cmd_shells(args)
        elif cmd == 'reload':
            return self._cmd_reload(args)
        elif cmd == 'players' or cmd == 'pl':
            return self._cmd_players(args)
        elif cmd == 'projectiles' or cmd == 'proj':
            return self._cmd_projectiles(args)
        elif cmd == 'buildings' or cmd == 'bld':
            return self._cmd_buildings(args)
        elif cmd == 'building_events' or cmd == 'bevents':
            return self._cmd_building_events(args)
        elif cmd == 'building_damage' or cmd == 'bdmg':
            return self._cmd_building_damage(args)
        elif cmd == 'building_projectile_damage' or cmd == 'bpdmg':
            return self._cmd_building_projectile_damage(args)
        elif cmd == 'input' or cmd == 'inp':
            return self._cmd_input(args)
        elif cmd == 'move' or cmd == 'mv':
            return self._cmd_move(args)
        elif cmd == 'jump' or cmd == 'jj':
            return self._cmd_jump(args)
        elif cmd == 'damage' or cmd == 'dmg':
            return self._cmd_damage(args)
        elif cmd == 'resend':
            return self._cmd_resend(args)
        elif cmd == 'terrain' or cmd == 'ter':
            return self._cmd_terrain(args)
        elif cmd == 'framdt' or cmd == 'fdt':
            return self._cmd_framdt(args)
        elif cmd == 'despawn' or cmd == 'ds':
            return self._cmd_despawn(args)
        elif cmd == 'kick':
            return self._cmd_kick(args)
        elif cmd == 'quit' or cmd == 'exit':
            return "Goodbye!"
        else:
            return f"Unknown command: {cmd}. Type 'help' for commands."

    def _cmd_help(self) -> str:
        return """Commands:
  raw <hex>              - Send raw bytes (e.g., "raw 17000005390001")
  send <type> [args]     - Send named packet (e.g., "send PLAYER 1337 false")
  state                  - Show current session state
  packets                - List available packet types with args
  phase <name>           - Force session phase (HANDSHAKE, LOGIN, TEAM_SELECT, etc.)
  flag <name> <0|1>      - Set feature flag
  enter_game [args]      - Force the active OG client through the real TankPacket spawn path
  join_team [c<id>] t<team> - Run the canonical team-switch handler for automation
  spawn_point [c<id>] [oid] [vehicle] - Run explicit map-flag spawn handler
  spawn_full [args]      - Force spawn sequence (see example)
  spawn_udp [args]       - Send UDP TANK (Wulf-Forge style)
  spawn_entity [type] [x y z] [vx vy vz] - Spawn entity (type 6=pulse, 5=flak)
  spawn_points [count] [team] - Send spawn point entities
  test_vel [speed] [dir] - Test projectile velocity (dir: x, z, xz, up, down, arc)
  shell [x y z] [yaw] [pitch] [speed] [duration] - Spawn shell with full control + updates
  spread [count] [x y z] [speed] - Fire spread of shells in 360° pattern
  reset_pos / rp - Reset player position to spawn
  turn [param] [value]   - Show/set turning params (sign, damping, friction, deadzone, adjust)
  barrel [param] [value] - Show/set barrel offsets (forward, right, up)
  heading [param] [value]- Show/set heading & aim params (offset, aim_offset, source, reset)
  fire [count|at <deg>]  - Force-fire projectile at current/specified heading
  pktlog [on|off|dump|save|analyze|clear] - Packet traffic logger
  physics [param] [value] - Yaw physics params (damp, reset)
  correction [...]       - Control empirical local correction loop or send one-shot packet
  framdt [ms]            - Show/set client frame dt in ms (for physics sync tuning)
  respawn / rs           - Re-send TankPacket to reset client entity position + heading
  despawn / ds [c<id>]   - Kill & despawn player (DELETE + reset to TEAM_SELECT)
  kick [c<id>|all]       - Despawn entity + fully disconnect client
  damage <amt> [c<id>]   - Apply damage (0.2=20%, or 20=20%)
  health [c<id>]         - Show player health
  health set <val> [c<id>] - Set health directly (0.0-1.0)
  energy [c<id>]         - Show player energy/fuel-local-state value
  energy set <val> [c<id>] - Set energy (0..1 fraction or absolute units)
  players / pl [json]    - Show all connected players' positions and headings
  projectiles / proj [json] - Dump live in-flight projectile state for divergence probes
  buildings / bld [json] - Dump static building, supply-pad, and turret state
  building_events / bevents [json] - Dump recent dynamic building lifecycle events
  building_damage / bdmg <oid> <hp|destroy> - Damage/destroy a building for demo gates
  building_projectile_damage / bpdmg <oid> [shots|destroy] [type] - Raycast projectile damage through a building
  jump / jj [secs] [c<id>] - Pulse opt-in server-side jumpjet action
  help                   - Show this help
  quit                   - Disconnect

Examples:
  raw 17000005390000                    # PLAYER packet, entity 1337, spectator=false
  send PLAYER 1337 spectator=false      # Same thing, named
  send CHAT "Hello world"               # Chat message
  send REINCARNATE 17 "Welcome!"        # Spawn success
  send UPDATE_ARRAY_TANK 1337 0 2       # Create tank entity
  join_team c1 t2                        # Team switch without spawning
  spawn_point c1 5002                    # Spawn at a map repair-pad oid
  enter_game c1 t2                       # Spawn client 1 on team 2 via WF/TankPacket path
  enter_game c1 t2 2579 3041 65          # Same, with explicit spawn position
  spawn_udp 1337 2 0 100 100 100        # UDP TANK (Wulf-Forge style)
  correction mode view_update           # Set empirical correction packet shape
  correction now c1                     # Queue correction on next tick for client 1
  correction send c1                    # Send correction packet immediately for packet-shape tests
  spawn_full 2 1337 0 0 100 50 100 150 2000        # team, oid, vehicle, behavior, x y z, delay_ms, ack_timeout_ms
  spawn_full 2 1337 0 0 100 50 100 150 2000 1 2000 # ... + send_world_stats, want_timeout_ms
  spawn_full 2 1337 0 0 100 50 100 150 0 ws=1 want=2000 interp=1 strict=1  # suppress WANT_UPDATES payload"""

    def _cmd_state(self) -> str:
        # Auto-sync to active client if our reference is stale
        if not self.session or self.session.phase.name == "DISCONNECTED":
            self._sync_to_active_client()
        if not self.session:
            return "No active session"
        s = self.session
        def _fmt_time(ts: float) -> str:
            if ts <= 0.0:
                return "(none)"
            return f"{ts - s.connected_at:.2f}s"
        lines = [
            f"Phase: {s.phase.name}",
            f"Username: {s.username or '(none)'}",
            f"Player ID: {s.player_id}",
            f"Entity ID: {s.entity_id}",
            f"Team ID: {s.team_id}",
            f"In Game: {s.in_game}",
            f"Tick: {s.tick}",
            f"UDP Verified: {s.udp_verified}",
            f"WANT_UPDATES: {s.want_updates_received} at {_fmt_time(s.want_updates_time)}",
            f"TRANSLATION_ACK: {s.translation_ack_received} at {_fmt_time(s.translation_ack_time)}",
            f"Known Entities: {sorted(self.ctx.known_entity_ids) if self.ctx else 'N/A'}",
            f"Known Roster: {sorted(self.ctx.known_roster_ids) if self.ctx else 'N/A'}",
            "",
            "Feature Flags:",
        ]
        for name, value in FEATURES.__dict__.items():
            lines.append(f"  {name}: {value}")
        return '\n'.join(lines)

    def _cmd_packets(self) -> str:
        lines = ["Available packet types:"]
        for name in sorted(self.builders.keys()):
            lines.append(f"  {name}")
        lines.append("")
        lines.append("Common raw packets:")
        lines.append("  17 <entity_id:4> <spectator:1>  # PLAYER")
        lines.append("  0E <tick:4> <bitstream>         # UPDATE_ARRAY")
        lines.append("  1F <code:1> <msg>               # REINCARNATE")
        return '\n'.join(lines)

    def _cmd_raw(self, args: list) -> str:
        if not args:
            return "Usage: raw <hex>"
        if not self.tcp_handler:
            return "Error: No game client connected"

        hex_str = ''.join(args).replace(' ', '')
        try:
            data = bytes.fromhex(hex_str)
        except ValueError as e:
            return f"Invalid hex: {e}"

        try:
            self.tcp_handler.send(data)
            pkt_type = data[0] if data else 0
            return f"Sent {len(data)} bytes (type 0x{pkt_type:02X})"
        except Exception as e:
            return f"Send error: {e}"

    def _cmd_send(self, args: list) -> str:
        if not args:
            return "Usage: send <type> [args...]"
        if not self.tcp_handler:
            return "Error: No game client connected"

        pkt_type = args[0].upper()
        pkt_args = args[1:]

        if pkt_type not in self.builders:
            return f"Unknown packet type: {pkt_type}. Type 'packets' for list."

        try:
            builder = self.builders[pkt_type]
            if pkt_args:
                data = builder(*pkt_args)
            else:
                data = builder()
            self.tcp_handler.send(data)
            return f"Sent {pkt_type} ({len(data)} bytes)"
        except Exception as e:
            import traceback
            return f"Build/send error: {e}\n{traceback.format_exc()}"

    def _cmd_phase(self, args: list) -> str:
        if not args:
            return "Usage: phase <name> (HANDSHAKE, LOGIN, TEAM_SELECT, SPAWNING, IN_GAME)"
        if not self.session:
            return "Error: No active session"

        phase_name = args[0].upper()
        try:
            new_phase = Phase[phase_name]
        except KeyError:
            return f"Unknown phase: {phase_name}"

        old_phase = self.session.phase
        self.session.phase = new_phase  # Force it (bypass validation)
        return f"Phase: {old_phase.name} -> {new_phase.name}"

    def _cmd_flag(self, args: list) -> str:
        if len(args) < 2:
            return "Usage: flag <name> <0|1>"

        name = args[0]
        value = args[1] in ('1', 'true', 'True', 'yes')

        if not hasattr(FEATURES, name):
            return f"Unknown flag: {name}"

        setattr(FEATURES, name, value)
        return f"Set {name} = {value}"

    def _cmd_turn(self, args: list) -> str:
        """
        Show or set turning parameters at runtime.
        Usage:
          turn                     - Show all turning params
          turn <param> <value>     - Set a turning param
          turn sign -1             - Set turn_sign to -1
          turn deadzone 0.1        - Set turn_deadzone to 0.1
          turn adjust 4.5          - Set turn_adjust (BEHAVIOR Sec 6 multiplier)
          turn curve 0,0.005,...   - Set piecewise steering curve samples (comma-sep)
        """
        if not self.server:
            return "Error: No server reference"

        if not args:
            # Show all turning params
            import math
            curve_str = ",".join(f"{s:.3f}" for s in self.server.turn_curve_samples)
            lines = [
                "Turning parameters (client curve model):",
                f"  sign     = {self.server.turn_sign}",
                f"  deadzone = {self.server.turn_deadzone}",
                f"  adjust   = {self.server.turn_adjust}  (BEHAVIOR Sec 6 turn multiplier)",
                f"  curve    = [{curve_str}]  ({len(self.server.turn_curve_samples)} samples)",
                "",
                "Usage: turn <param> <value>",
                "Params: sign, deadzone, adjust, curve",
            ]
            return '\n'.join(lines)

        if len(args) < 2:
            return "Usage: turn <param> <value> (params: sign, deadzone, adjust, curve)"

        param = args[0].lower()

        if param == 'curve':
            # Curve takes comma-separated floats
            try:
                samples = [float(s) for s in args[1].split(",")]
            except ValueError:
                return f"Invalid curve values: {args[1]}. Use comma-separated floats."
            if len(samples) < 2:
                return "Curve needs at least 2 samples."
            old_str = ",".join(f"{s:.3f}" for s in self.server.turn_curve_samples)
            self.server.turn_curve_samples = samples
            new_str = ",".join(f"{s:.3f}" for s in samples)
            print(f"[CONTROL] turn curve: [{old_str}] -> [{new_str}]")
            return f"Set curve = [{new_str}] ({len(samples)} samples)"

        try:
            value = float(args[1])
        except ValueError:
            return f"Invalid value: {args[1]}"

        param_map = {
            'sign': 'turn_sign',
            'deadzone': 'turn_deadzone',
            'adjust': 'turn_adjust',
        }

        attr = param_map.get(param)
        if not attr:
            return f"Unknown turn param: {param}. Valid: {', '.join(list(param_map.keys()) + ['curve'])}"

        old_value = getattr(self.server, attr)
        setattr(self.server, attr, value)
        print(f"[CONTROL] turn {param}: {old_value} -> {value}")
        return f"Set {param} = {value} (was {old_value})"

    def _cmd_barrel(self, args: list) -> str:
        """
        Show or set barrel/turret offset parameters for projectile spawning.
        Usage:
          barrel                   - Show all barrel params
          barrel <param> <value>   - Set a barrel param
          barrel forward 2.0       - Set forward offset (spawn_offset)
          barrel right 0.5         - Set lateral (right) offset
          barrel up 0.3            - Set vertical (up) offset
        """
        if not self.server:
            return "Error: No server reference"

        # Collect current values from first connected client's weapon system, or defaults
        ws = None
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if hasattr(ctx, 'weapon_system'):
                    ws = ctx.weapon_system
                    break

        cur_forward = ws.projectile_spawn_offset if ws else 2.0
        cur_right = ws.projectile_barrel_right if ws else 0.0
        cur_up = ws.projectile_barrel_up if ws else 0.2

        if not args:
            lines = [
                "Barrel offset parameters:",
                f"  forward = {cur_forward}  (spawn offset along aim direction)",
                f"  right   = {cur_right}  (lateral offset, rotates with heading)",
                f"  up      = {cur_up}  (vertical offset)",
                "",
                "Usage: barrel <param> <value>",
                "Params: forward, right, up",
            ]
            return '\n'.join(lines)

        if len(args) < 2:
            return "Usage: barrel <param> <value> (params: forward, right, up)"

        param = args[0].lower()
        try:
            value = float(args[1])
        except ValueError:
            return f"Invalid value: {args[1]}"

        param_map = {
            'forward': 'projectile_spawn_offset',
            'right': 'projectile_barrel_right',
            'up': 'projectile_barrel_up',
        }

        attr = param_map.get(param)
        if not attr:
            return f"Unknown barrel param: {param}. Valid: {', '.join(param_map.keys())}"

        # Update all connected clients' weapon systems
        updated = 0
        old_value = None
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if hasattr(ctx, 'weapon_system'):
                    if old_value is None:
                        old_value = getattr(ctx.weapon_system, attr)
                    setattr(ctx.weapon_system, attr, value)
                    updated += 1

        if old_value is None:
            old_value = value
        print(f"[CONTROL] barrel {param}: {old_value} -> {value} ({updated} clients)")
        return f"Set {param} = {value} (was {old_value}, updated {updated} clients)"

    def _cmd_heading(self, args: list) -> str:
        """
        Show or set heading/aim parameters for projectile alignment.
        Usage:
          heading                          - Show all heading params + live comparison
          heading offset <deg>             - Set heading_offset_deg on all weapon systems
          heading aim_offset <deg>         - Set aim_yaw_offset_deg on all weapon systems
          heading source <body|viewpoint|auto> - Set projectile_aim_source
          heading reset                    - Reset server heading to 0 for all clients
        """
        import math

        if not self.server:
            return "Error: No server reference"

        # Get first weapon system for display
        ws = None
        ctx_ref = None
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if hasattr(ctx, 'weapon_system'):
                    ws = ctx.weapon_system
                    ctx_ref = ctx
                    break

        if not args:
            lines = ["Heading/aim parameters:"]
            lines.append(f"  aim_source       = {self.server.projectile_aim_source}")
            lines.append(f"  body_pitch       = {int(getattr(self.server, 'projectile_body_pitch', False))}")
            if ws:
                lines.append(f"  heading_offset   = {math.degrees(ws.heading_offset):.1f}deg")
                lines.append(f"  aim_yaw_offset   = {math.degrees(ws.aim_yaw_offset):.1f}deg")
            else:
                lines.append(f"  heading_offset   = (no client)")
                lines.append(f"  aim_yaw_offset   = (no client)")
            if ctx_ref:
                srv_hdg = math.degrees(-ctx_ref.player_heading)
                vp_yaw = math.degrees(ctx_ref.player_aim_yaw)
                vp_age = time.monotonic() - ctx_ref.player_aim_time
                delta = vp_yaw - srv_hdg
                while delta > 180.0:
                    delta -= 360.0
                while delta < -180.0:
                    delta += 360.0
                lines.append(f"")
                lines.append(f"  Live comparison:")
                lines.append(f"    server_heading = {srv_hdg:.1f}deg")
                lines.append(f"    viewpoint_yaw  = {vp_yaw:.1f}deg (age={vp_age:.1f}s)")
                lines.append(f"    delta          = {delta:.1f}deg")
                lines.append(f"    vp_source      = {ctx_ref.player_aim_source}")
                lines.append(f"    vp_count       = {ctx_ref.viewpoint_count}")
            lines.append("")
            lines.append("Usage: heading <param> <value>")
            lines.append("Params: offset, aim_offset, source, reset")
            return '\n'.join(lines)

        param = args[0].lower()

        if param == 'reset':
            count = 0
            with self.server.clients_lock:
                for ctx in self.server.clients.values():
                    ctx.player_heading = 0.0
                    ctx.player_yaw = 0.0
                    ctx.player_pose["yaw"] = 0.0
                    ctx.spring_body_ang_vel = (0.0, 0.0)
                    ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)
                    if ctx.vehicle_physics:
                        ctx.vehicle_physics.reset()
                    count += 1
            return f"Reset heading to 0 for {count} client(s)"

        if param == 'set':
            # heading set <deg> [c<id>] — force server heading to specific value
            if len(args) < 2:
                return "Usage: heading set <deg> [c<id>]"
            try:
                deg = float(args[1])
            except ValueError:
                return f"Invalid angle: {args[1]}"
            rad = math.radians(deg)
            target_id = None
            if len(args) >= 3 and args[2].lower().startswith("c") and args[2][1:].isdigit():
                target_id = int(args[2][1:])
            updated = []
            with self.server.clients_lock:
                for ctx in self.server.clients.values():
                    if target_id is not None and ctx.client_id != target_id:
                        continue
                    ctx.player_heading = rad
                    ctx.player_yaw = -rad
                    ctx.player_pose["yaw"] = -rad
                    ctx.spring_body_matrix = _matrix3_from_euler_xyz(
                        float(ctx.player_pose.get("roll", 0.0) or 0.0),
                        float(ctx.player_pose.get("pitch", 0.0) or 0.0),
                        rad,
                    )
                    if ctx.vehicle_physics:
                        ctx.vehicle_physics.heading = rad
                        ctx.vehicle_physics._angular_velocity = 0.0
                    updated.append(ctx.client_id)
            return f"Set heading to {deg}° for client(s) {updated}"

        if param == 'source':
            if len(args) < 2:
                return f"Current source: {self.server.projectile_aim_source}. Usage: heading source <body|viewpoint|auto>"
            new_src = args[1].lower()
            if new_src not in ('body', 'viewpoint', 'auto'):
                return f"Invalid source: {new_src}. Valid: body, viewpoint, auto"
            old = self.server.projectile_aim_source
            self.server.projectile_aim_source = new_src
            print(f"[CONTROL] heading source: {old} -> {new_src}")
            return f"Set aim source = {new_src} (was {old})"

        if len(args) < 2:
            return "Usage: heading <param> <value>"

        try:
            value = float(args[1])
        except ValueError:
            return f"Invalid value: {args[1]}"

        if param == 'remote_offset' or param == 'roff':
            old_val = math.degrees(self.server.remote_yaw_offset)
            self.server.remote_yaw_offset = math.radians(value)
            print(f"[CONTROL] remote_yaw_offset: {old_val} -> {value}")
            return f"Set remote_yaw_offset = {value}deg (was {old_val}deg)"

        if param == 'remote_negate' or param == 'rneg':
            old_val = self.server.remote_yaw_negate
            self.server.remote_yaw_negate = (value != 0)
            print(f"[CONTROL] remote_yaw_negate: {old_val} -> {self.server.remote_yaw_negate}")
            return f"Set remote_yaw_negate = {self.server.remote_yaw_negate} (was {old_val})"

        if param == 'offset':
            updated = 0
            old_val = None
            with self.server.clients_lock:
                for ctx in self.server.clients.values():
                    if hasattr(ctx, 'weapon_system'):
                        if old_val is None:
                            old_val = math.degrees(ctx.weapon_system.heading_offset)
                        ctx.weapon_system.heading_offset = math.radians(value)
                        updated += 1
            print(f"[CONTROL] heading offset: {old_val} -> {value} ({updated} clients)")
            return f"Set heading_offset = {value}deg (was {old_val}deg, {updated} clients)"

        if param == 'aim_offset':
            updated = 0
            old_val = None
            with self.server.clients_lock:
                for ctx in self.server.clients.values():
                    if hasattr(ctx, 'weapon_system'):
                        if old_val is None:
                            old_val = math.degrees(ctx.weapon_system.aim_yaw_offset)
                        ctx.weapon_system.aim_yaw_offset = math.radians(value)
                        updated += 1
            print(f"[CONTROL] heading aim_offset: {old_val} -> {value} ({updated} clients)")
            return f"Set aim_yaw_offset = {value}deg (was {old_val}deg, {updated} clients)"

        return f"Unknown heading param: {param}. Valid: offset, aim_offset, source, reset"

    def _send_empirical_correction(self, ctx) -> str:
        """Send one local-player correction packet immediately for packet-shape testing."""
        import math

        if not self.server or not ctx.session:
            return "Error: No server/session reference"

        local_state_kwargs = self.server._get_local_state_kwargs(ctx)
        weapon_type = local_state_kwargs["weapon_id"]
        ammo_bits = local_state_kwargs["ammo_count_bits"]
        ammo_mask = local_state_kwargs["ammo_count"]
        pt_bits = local_state_kwargs["primary_turret_bits"]
        pt_angle = local_state_kwargs["primary_turret_angle"]
        st_bits = local_state_kwargs["secondary_turret_bits"]
        st_angle = local_state_kwargs["secondary_turret_angle"]
        include_local_state = self.server._should_send_local_state(
            ctx,
            pt_bits,
            st_bits,
            self.server.update_local_state_mode,
        )
        tick = self.server._get_network_tick(ctx)
        payload, label, corr_pos, corr_rot, _, _ = self.server._build_empirical_correction_payload(
            ctx,
            tick=tick,
            include_local_state=include_local_state,
            health=self.server._get_health_value(ctx),
            fuel=self.server._get_energy_value(ctx),
            weapon_type=weapon_type,
            ammo_bits=ammo_bits,
            ammo_mask=ammo_mask,
            pt_bits=pt_bits,
            pt_angle=pt_angle,
            st_bits=st_bits,
            st_angle=st_angle,
        )

        transports = []
        if getattr(self.server, "send_updates_tcp", False) and ctx.tcp_handler:
            try:
                ctx.tcp_handler.send(payload, log=False)
                transports.append("TCP")
            except Exception as exc:
                print(f"[CONTROL-CORRECTION] TCP send failed for client {ctx.client_id}: {exc}")
        if getattr(self.server, "send_updates_udp", True) and self.server.udp_handler and ctx.session.udp_addr:
            self.server.udp_handler.send_to(payload, ctx.session.udp_addr)
            transports.append("UDP")
        if not transports:
            return "Error: No available transport for correction packet"

        ctx.last_correction_send = time.monotonic()
        ctx.force_correction_once = False
        print(
            f"[CONTROL-CORRECTION] mode={self.server.correction_mode} client={ctx.client_id} "
            f"label={label} transports={'+'.join(transports)} "
            f"pos=({corr_pos[0]:.1f},{corr_pos[1]:.1f},{corr_pos[2]:.1f}) "
            f"yaw={math.degrees(corr_rot[2]):.1f}deg"
        )
        return (
            f"Sent {label} to client {ctx.client_id} via {'+'.join(transports)} "
            f"tick={tick} pos=({corr_pos[0]:.1f},{corr_pos[1]:.1f},{corr_pos[2]:.1f}) "
            f"yaw={math.degrees(corr_rot[2]):.1f}deg"
        )

    def _cmd_correction(self, args: list) -> str:
        """Show or control the empirical local-correction loop."""
        if not self.server:
            return "Error: No server reference"

        valid_modes = ("full", "rot_only", "pos_only", "dual_entity", "view_update", "view_update_define")
        if not hasattr(self.server, "correction_mode"):
            self.server.correction_mode = "dual_entity"
        if not hasattr(self.server, "correction_interval"):
            self.server.correction_interval = 0.0

        if not args:
            interval = float(getattr(self.server, "correction_interval", 0.0) or 0.0)
            movement_interval = float(
                getattr(self.server, "movement_correction_interval", 0.0) or 0.0
            )
            movement_window = float(
                getattr(self.server, "movement_correction_window", 0.0) or 0.0
            )
            status = "ON" if interval > 0 or movement_interval > 0 else "OFF"
            lines = [
                f"Drift correction: {status} mode={self.server.correction_mode}",
                f"  interval = {interval:.2f}s (0=disabled)",
                (
                    "  movement = "
                    f"{movement_interval:.2f}s for {movement_window:.1f}s after input "
                    "(0=disabled)"
                ),
            ]
            for ctx in self.server._snapshot_in_game_clients():
                age = time.monotonic() - ctx.last_correction_send if getattr(ctx, "last_correction_send", 0.0) > 0 else -1.0
                queued = " queued" if getattr(ctx, "force_correction_once", False) else ""
                if age >= 0:
                    lines.append(f"  client {ctx.client_id}: last_correction={age:.1f}s ago{queued}")
                else:
                    lines.append(f"  client {ctx.client_id}: no corrections sent yet{queued}")
            lines.append("")
            lines.append("Usage: correction <seconds|on|off|now|mode <name>|send [c<id>]>")
            lines.append("Modes: full, rot_only, pos_only, dual_entity, view_update, view_update_define")
            return "\n".join(lines)

        subcmd = args[0].lower()
        if subcmd == "on":
            if self.server.correction_interval <= 0:
                self.server.correction_interval = 5.0
            return f"Correction ON (interval={self.server.correction_interval:.2f}s mode={self.server.correction_mode})"
        if subcmd == "off":
            self.server.correction_interval = 0.0
            for ctx in self.server._snapshot_in_game_clients():
                ctx.force_correction_once = False
                ctx.correction_burst_remaining = 0
                ctx.correction_burst_interval_s = 0.0
            return "Correction OFF"
        if subcmd == "now":
            # Optional trailing tokens: c<id>, burst count, interval seconds.
            # Empirically, a single VIEW_UPDATE push is invisible on the OG
            # client — a short ~10 Hz burst is required for the correction to
            # visibly apply and the snapped pose to persist afterwards. Default
            # to a 10-packet, 0.1s interval burst so `correction now` behaves
            # as a reliable one-shot snap.
            target_id = None
            burst_count = 10
            burst_interval = 0.1
            for token in args[1:]:
                lower = token.lower()
                if lower.startswith("c") and lower[1:].isdigit():
                    target_id = int(lower[1:])
                    continue
                try:
                    numeric = float(token)
                except ValueError:
                    continue
                if burst_count == 10 and numeric >= 1 and numeric == int(numeric):
                    burst_count = int(numeric)
                else:
                    burst_interval = max(0.01, numeric)
            if target_id is not None:
                ctx, _ = self._get_client_by_id(target_id)
                if not ctx:
                    return f"Error: No client with id {target_id}"
                targets = [ctx]
            else:
                targets = self.server._snapshot_in_game_clients()
            if not targets:
                return "No in-game clients"
            for ctx in targets:
                ctx.force_correction_once = True
                ctx.correction_burst_remaining = max(0, burst_count - 1)
                ctx.correction_burst_interval_s = burst_interval
                ctx.last_correction_send = 0.0
            suffix = f" for client {target_id}" if target_id is not None else ""
            return (
                f"Queued correction burst x{burst_count} @ {burst_interval:.2f}s{suffix} "
                f"(mode={self.server.correction_mode})"
            )
        if subcmd == "send":
            target_id = None
            if len(args) >= 2 and args[1].lower().startswith("c") and args[1][1:].isdigit():
                target_id = int(args[1][1:])
            if target_id is not None:
                ctx, _ = self._get_client_by_id(target_id)
                if not ctx:
                    return f"Error: No client with id {target_id}"
                return self._send_empirical_correction(ctx)
            lines = []
            for ctx in self.server._snapshot_in_game_clients():
                lines.append(self._send_empirical_correction(ctx))
            return "\n".join(lines) if lines else "No in-game clients"
        if subcmd == "mode":
            if len(args) < 2:
                return f"Current mode: {self.server.correction_mode}\nModes: {', '.join(valid_modes)}"
            new_mode = args[1].lower()
            if new_mode not in valid_modes:
                return f"Invalid mode: {new_mode}\nModes: {', '.join(valid_modes)}"
            old_mode = self.server.correction_mode
            self.server.correction_mode = new_mode
            print(f"[CONTROL] correction mode: {old_mode} -> {new_mode}")
            return f"Correction mode: {old_mode} -> {new_mode}"

        try:
            value = float(subcmd)
        except ValueError:
            return (
                f"Unknown correction subcommand: {subcmd}\n"
                f"Usage: correction <seconds|on|off|now|mode <name>|send [c<id>]>"
            )
        old_interval = self.server.correction_interval
        self.server.correction_interval = max(0.0, value)
        print(f"[CONTROL] correction interval: {old_interval:.2f} -> {self.server.correction_interval:.2f}")
        if self.server.correction_interval <= 0:
            for ctx in self.server._snapshot_in_game_clients():
                ctx.force_correction_once = False
        return f"Set correction interval = {self.server.correction_interval:.2f}s" + (" (OFF)" if self.server.correction_interval <= 0 else "")

    def _cmd_fire(self, args: list) -> str:
        """
        Force-fire a projectile from the server side using current heading.
        Usage:
          fire                     - Fire once using current heading
          fire <count>             - Fire count times with 100ms delay
          fire at <yaw_deg>        - Fire at specific heading (degrees)
          fire speed <val>         - Set projectile speed (default 75)
          fire slow <count>        - Fire slow (speed=5) burst for minimap tracing
        """
        import math

        if not self.server:
            return "Error: No server reference"

        # Check for c<N> client selector in args (can appear anywhere)
        target_client_id = None
        filtered_args = []
        for a in args:
            if a.startswith('c') and a[1:].isdigit():
                target_client_id = int(a[1:])
            else:
                filtered_args.append(a)
        args = filtered_args

        # Find target client (or first connected)
        ctx = None
        addr = None
        with self.server.clients_lock:
            for c in self.server.clients.values():
                if hasattr(c, 'weapon_system') and c.session and c.session.udp_addr:
                    if target_client_id is not None and c.client_id != target_client_id:
                        continue
                    ctx = c
                    addr = c.session.udp_addr
                    break

        if not ctx:
            return f"Error: No connected client{f' c{target_client_id}' if target_client_id else ''} with weapon system"

        count = 1
        override_yaw = None
        speed_override = None

        if args:
            if args[0] == 'at' and len(args) >= 2:
                try:
                    override_yaw = math.radians(float(args[1]))
                except ValueError:
                    return f"Invalid yaw: {args[1]}"
                if len(args) >= 3:
                    try:
                        count = int(args[2])
                    except ValueError:
                        pass
            elif args[0] == 'sweep' and len(args) >= 3:
                # fire sweep <min_deg> <max_deg> [step] - fan of shells
                try:
                    sweep_min = float(args[1])
                    sweep_max = float(args[2])
                    sweep_step = float(args[3]) if len(args) >= 4 else 5.0
                except ValueError:
                    return "Usage: fire sweep <min_deg> <max_deg> [step_deg]"
                sweep_headings = []
                h = sweep_min
                while h <= sweep_max + 0.01:
                    sweep_headings.append(math.radians(h))
                    h += sweep_step
                # Fire 3 shells per heading
                results = []
                for yaw in sweep_headings:
                    for _ in range(3):
                        ctx.weapon_system.player_pos = ctx.player_pos
                        ctx.weapon_system.player_rot = (0.0, 0.0, yaw)
                        ctx.weapon_system.current_weapon = 4
                        ctx.weapon_system.last_fire_time = 0
                        proj = ctx.weapon_system._fire_pulse_cannon()
                        if proj:
                            self.server._spawn_moving_projectile(ctx, proj, addr)
                            ctx.weapon_system.projectiles.append(proj)
                        import time
                        time.sleep(0.02)
                hdg_s = f"{sweep_min:.0f}"
                hdg_e = f"{sweep_max:.0f}"
                return f"Swept {len(sweep_headings)} headings from {hdg_s} to {hdg_e} deg (step={sweep_step})"
            elif args[0] == 'speed' and len(args) >= 2:
                try:
                    ctx.weapon_system.pulse_shell_speed = float(args[1])
                except ValueError:
                    return f"Invalid speed: {args[1]}"
                return f"Set pulse_shell_speed = {ctx.weapon_system.pulse_shell_speed}"
            elif args[0] == 'slow':
                speed_override = 40.0
                count = int(args[1]) if len(args) >= 2 else 15
            else:
                try:
                    count = int(args[0])
                except ValueError:
                    return f"Invalid count: {args[0]}"

        results = []
        old_speed = ctx.weapon_system.pulse_shell_speed
        if speed_override is not None:
            ctx.weapon_system.pulse_shell_speed = speed_override

        for i in range(count):
            # Set weapon system pose from current heading (or override)
            yaw = override_yaw if override_yaw is not None else ctx.player_heading
            ctx.weapon_system.player_pos = ctx.player_pos
            ctx.weapon_system.player_rot = (0.0, 0.0, yaw)
            ctx.weapon_system.current_weapon = 4  # PULSE_CANNON

            # Reset cooldown so rapid-fire works
            ctx.weapon_system.last_fire_time = 0
            proj = ctx.weapon_system._fire_pulse_cannon()
            if proj:
                self.server._spawn_moving_projectile(ctx, proj, addr)
                ctx.weapon_system.projectiles.append(proj)
                hdg = math.degrees(yaw)
                spd = ctx.weapon_system.pulse_shell_speed
                results.append(f"Shot {i+1}: heading={hdg:.1f}deg speed={spd:.0f} proj_id={proj.entity_id}")
                if i == 0 or i == count - 1:
                    print(f"[CONTROL-FIRE] heading={hdg:.1f} speed={spd} pos={ctx.player_pos}")
            else:
                results.append(f"Shot {i+1}: FAILED")

            if i < count - 1:
                import time
                time.sleep(0.05)

        if speed_override is not None:
            ctx.weapon_system.pulse_shell_speed = old_speed

        return '\n'.join(results)

    def _cmd_pktlog(self, args: list) -> str:
        """Packet traffic logger for debugging freezes.

        Usage:
          pktlog              - Show status
          pktlog on           - Start logging
          pktlog off          - Stop logging
          pktlog dump [N]     - Show last N packets (default 50)
          pktlog save [path]  - Save to file
          pktlog analyze      - Timing/frequency analysis
          pktlog clear        - Clear buffer
        """
        if not self.server:
            return "Error: No server reference"

        log = self.server.pktlog
        if not args:
            status = "ON" if log.enabled else "OFF"
            count = len(log._entries)
            return f"Packet log: {status} ({count} entries buffered)"

        subcmd = args[0].lower()
        if subcmd == 'on':
            log.start()
            return "Packet logging ON"
        elif subcmd == 'off':
            log.stop()
            return f"Packet logging OFF ({len(log._entries)} entries)"
        elif subcmd == 'dump':
            n = int(args[1]) if len(args) > 1 else 50
            return log.dump(n)
        elif subcmd == 'save':
            path = args[1] if len(args) > 1 else "pktlog.txt"
            return log.save(path)
        elif subcmd == 'analyze':
            return log.analyze()
        elif subcmd == 'clear':
            log.clear()
            return "Packet log cleared"
        else:
            return f"Unknown: {subcmd}. Use on/off/dump/save/analyze/clear."

    def _cmd_physics(self, args: list) -> str:
        """
        Show or modify yaw physics parameters.
        Direct-impulse model: torque → damped angular velocity → heading.
        Usage:
          physics                   - Show current physics params
          physics damp <val>        - Set damping coefficient (default 1.0)
          physics reset             - Reset heading and angular velocity
        """
        import math

        ctx_ref = None
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                ctx_ref = ctx
                break

        if not args:
            lines = ["Yaw physics (direct-impulse model):"]
            lines.append(f"  turn_adjust     = {self.server.turn_adjust}")
            lines.append(f"  turn_sign       = {self.server.turn_sign}")
            if ctx_ref and ctx_ref.vehicle_physics:
                p = ctx_ref.vehicle_physics
                lines.append(f"  damp_coeff      = {p.damp_coeff:.4f}")
                lines.append(f"  angular_vel     = {p.angular_velocity:.4f} rad/s")
                lines.append(f"  heading         = {math.degrees(-p.heading):.1f}deg")
                ss = self.server.turn_adjust / p.damp_coeff if p.damp_coeff > 0 else float('inf')
                lines.append(f"  steady_state    = {ss:.2f} rad/s ({math.degrees(ss):.0f}deg/s)")
            else:
                lines.append("  (no client connected)")
            lines.append("")
            lines.append("Usage: physics <param> [value]")
            lines.append("Params: damp, reset")
            return '\n'.join(lines)

        param = args[0].lower()

        if param == 'reset':
            count = 0
            with self.server.clients_lock:
                for ctx in self.server.clients.values():
                    if ctx.vehicle_physics:
                        ctx.vehicle_physics.reset()
                        ctx.player_heading = 0.0
                        ctx.player_yaw = 0.0
                        ctx.player_pose["yaw"] = 0.0
                        ctx.angular_vel_yaw = 0.0
                        count += 1
            return f"Reset heading + angular velocity for {count} client(s)"

        if len(args) < 2:
            return "Usage: physics <param> <value>"

        try:
            value = float(args[1])
        except ValueError:
            return f"Invalid value: {args[1]}"

        if param == 'damp' or param == 'damp_coeff' or param == 'dc':
            count = 0
            old_val = None
            with self.server.clients_lock:
                for ctx in self.server.clients.values():
                    if ctx.vehicle_physics:
                        if old_val is None:
                            old_val = ctx.vehicle_physics.damp_coeff
                        ctx.vehicle_physics.damp_coeff = value
                        count += 1
            self.server.damp_coeff = value
            print(f"[CONTROL] damp_coeff: {old_val} -> {value} ({count} clients)")
            return f"Set damp_coeff = {value} (was {old_val}, {count} clients)"

        return f"Unknown physics param: {param}. Valid: damp, reset"

    def _cmd_behavior(self, args: list) -> str:
        """
        Show or modify BEHAVIOR packet parameters and re-send to client.
        Usage:
          behavior                          - Show current BEHAVIOR params
          behavior <param> <value>          - Set param and re-send packet
          behavior send                     - Re-send packet with current values
          behavior turn_rate 0.15           - Set Section 4 turn_rate
          behavior turn_adjust 6.0          - Set Section 6 turn_adjust
          behavior ground_friction 0.5      - Set Section 4 ground_friction
          behavior mass 15000               - Set Section 4 mass
          behavior gravity 150.0            - Set Section 1 gravity_force
          behavior speed 30.0               - Set Section 4 speed
          behavior move_adjust 100.0        - Set Section 6 move_adjust
          behavior max_velocity 100.0       - Set Section 6 max_velocity
        """
        if not self.tcp_handler:
            return "Error: No game client connected"

        from . import packets as pkt

        # Current values from module globals
        param_map = {
            'turn_rate': ('BEHAVIOR_TURN_RATE', 'Section 4: vehicle turn rate (rad/s)'),
            'ground_friction': ('BEHAVIOR_GROUND_FRICTION', 'Section 4: ground friction'),
            'suspension_dampening': ('BEHAVIOR_SUSPENSION_DAMPENING', 'Section 4: suspension dampening'),
            'max_altitude': ('BEHAVIOR_MAX_ALTITUDE', 'Section 6: max altitude'),
            'gravity_pct': ('BEHAVIOR_GRAVITY_PCT', 'Section 6: gravity percent'),
        }

        if not args:
            lines = [
                "BEHAVIOR packet parameters:",
                f"  -- Section 1 (Header) --",
                f"  gravity_force    = 100.0  (hardcoded)",
                f"  -- Section 4 (Vehicle Physics, x2 vehicles) --",
                f"  speed            = 20.0  (EntityParams+0x00, decompile-verified)",
                f"  accel            = 4.0  (EntityParams+0x08, decompile-verified)",
                f"  engine_torque    = 700  (EntityParams+0x10 max_health, decompile-verified)",
                f"  suspension_stiff = 550  (EntityParams+0x14 energy_capacity, decompile-verified)",
                f"  ground_friction  = {pkt.BEHAVIOR_GROUND_FRICTION}",
                f"  turn_rate        = {pkt.BEHAVIOR_TURN_RATE}",
                f"  susp_dampening   = {pkt.BEHAVIOR_SUSPENSION_DAMPENING}",
                f"  max_fuel         = 33000  (EntityParams+0x34, decompile-verified)",
                f"  -- Section 6 (Active Vehicle, Tank) --",
                f"  turn_adjust      = 4.5  (VehiclePhysics_init_tank +0x10, decompile-verified)",
                f"  move_adjust      = 85.0  (VehiclePhysics_init_tank +0x18, decompile-verified)",
                f"  strafe_adjust    = 69.7  (VehiclePhysics_init_tank +0x20, decompile-verified)",
                f"  max_velocity     = 80.0  (VehiclePhysics_init_tank +0x28, decompile-verified)",
                f"  max_altitude     = {pkt.BEHAVIOR_MAX_ALTITUDE}",
                f"  gravity_pct      = {pkt.BEHAVIOR_GRAVITY_PCT}",
                "",
                "Usage: behavior <param> <value>  (re-sends packet)",
                "       behavior send              (re-send with current values)",
                "Params: turn_rate, ground_friction, suspension_dampening, max_altitude, gravity_pct",
            ]
            return '\n'.join(lines)

        if args[0].lower() == 'send':
            # Just re-send with current values
            data = pkt.build_behavior_packet()
            self.tcp_handler.send(data)
            return f"Re-sent BEHAVIOR packet ({len(data)} bytes)"

        if len(args) < 2:
            return "Usage: behavior <param> <value> or behavior send"

        param = args[0].lower()
        try:
            value = float(args[1])
        except ValueError:
            return f"Invalid value: {args[1]}"

        if param not in param_map:
            return f"Unknown param: {param}. Valid: {', '.join(param_map.keys())}"

        global_name, desc = param_map[param]
        old_value = getattr(pkt, global_name)
        setattr(pkt, global_name, value)

        # Rebuild and send
        data = pkt.build_behavior_packet()
        self.tcp_handler.send(data)

        print(f"[CONTROL] behavior {param}: {old_value} -> {value}, re-sent ({len(data)} bytes)")
        return f"Set {param} = {value} (was {old_value}), re-sent BEHAVIOR ({len(data)} bytes)\n{desc}"

    def _cmd_reload(self, args: list) -> str:
        """Hot-reload server modules without restarting.

        Reloads all .py modules and swaps classes on live instances.
        All instance state (sockets, threads, clients, physics) is preserved.

        Usage:
          reload        - Reload all modules (server, control, packets, physics, etc.)
          reload quick  - Reload only control + packets (faster, for command changes)
        """
        import importlib
        import traceback

        quick = args and args[0].lower() in ('quick', 'q')

        try:
            from . import packets as packets_mod
            from . import control as control_mod

            # Preserve tick counter base across reload
            old_server_start = packets_mod._SERVER_START

            reloaded = []

            if not quick:
                # Full reload: all modules in dependency order
                from wulfram2_protocol import entities as entities_mod
                from . import codec as codec_mod
                from . import session as session_mod
                from . import client as client_mod
                from . import physics as physics_mod
                from . import weapons as weapons_mod
                from . import transport as transport_mod
                from . import handlers as handlers_mod
                from . import jump_jets as jump_jets_mod
                from . import server as server_mod

                # Preserve FEATURES global across reload (session.py recreates it)
                old_features = session_mod.FEATURES

                # Reload leaf modules first, then dependents.
                # CRITICAL: Restore FEATURES onto session_mod immediately after
                # reloading session, BEFORE reloading modules that import it
                # (handlers, control, server). Otherwise those modules bind to
                # the discarded new Features() instance.
                for mod, name in [
                    (entities_mod, "shared.entities"),
                    (codec_mod, "codec"),
                    (session_mod, "session"),
                    (physics_mod, "physics"),
                    (jump_jets_mod, "jump_jets"),
                    (weapons_mod, "weapons"),
                    (client_mod, "client"),
                    (transport_mod, "transport"),
                    (packets_mod, "packets"),
                    (handlers_mod, "handlers"),
                    (control_mod, "control"),
                    (server_mod, "server"),
                ]:
                    importlib.reload(mod)
                    reloaded.append(name)
                    # Restore preserved state immediately after their modules reload
                    if name == "session":
                        session_mod.FEATURES = old_features
                    elif name == "packets":
                        packets_mod._SERVER_START = old_server_start

                # Migrate live session Phase values to the new Phase enum.
                # When session.py is reloaded, the Phase enum class is recreated.
                # Existing ctx.session.phase holds OLD Phase instances which fail
                # identity comparison with new Phase members, causing game loops
                # to exit (the `phase in [Phase.TEAM_SELECT, ...]` check fails).
                new_Phase = session_mod.Phase
                if self.server:
                    with self.server.clients_lock:
                        for ctx in self.server.clients.values():
                            if ctx.session and ctx.session.phase:
                                try:
                                    ctx.session.phase = new_Phase[ctx.session.phase.name]
                                except (KeyError, AttributeError):
                                    pass

                # Swap WulframServer class on live instance
                if self.server:
                    self.server.__class__ = server_mod.WulframServer

                    # Initialize any NEW attributes from updated __init__
                    # without overwriting existing state (connections, physics, etc.)
                    if hasattr(self.server, '_apply_reload_defaults'):
                        self.server._apply_reload_defaults()

                    # Swap classes on all live client sub-objects
                    with self.server.clients_lock:
                        for ctx in self.server.clients.values():
                            if hasattr(ctx, 'vehicle_physics') and ctx.vehicle_physics:
                                ctx.vehicle_physics.__class__ = physics_mod.VehiclePhysics
                            if hasattr(ctx, 'weapon_system') and ctx.weapon_system:
                                ctx.weapon_system.__class__ = weapons_mod.WeaponSystem
                            if hasattr(ctx, 'jump_jet_system') and ctx.jump_jet_system:
                                ctx.jump_jet_system.__class__ = jump_jets_mod.JumpJetSystem
                            if ctx.session:
                                ctx.session.__class__ = session_mod.Session

                # Swap ControlServer class on this instance
                self.__class__ = control_mod.ControlServer

            else:
                # Quick reload: just control + packets
                importlib.reload(packets_mod)
                packets_mod._SERVER_START = old_server_start
                reloaded.append("packets")

                importlib.reload(control_mod)
                reloaded.append("control")

                self.__class__ = control_mod.ControlServer

            msg = f"Reloaded: {', '.join(reloaded)}"
            print(f"[RELOAD] {msg}")
            return msg
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[RELOAD] ERROR: {e}\n{tb}")
            return f"Reload failed: {e}"

    def _cmd_players(self, args: list) -> str:
        """Show all connected players' positions and headings.

        Usage:
          players       - List all in-game clients with server-tracked position/heading
          players json  - Output as JSON for tooling
        """
        import math
        import json as _json

        if not self.server:
            return "Error: No server reference"

        json_mode = args and args[0].lower() == "json"
        clients = []
        now = time.monotonic()

        def _age(ts: float) -> Optional[float]:
            if not ts or ts <= 0.0:
                return None
            return round(max(0.0, now - float(ts)), 3)

        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if not ctx or not ctx.running:
                    continue
                phase = ctx.session.phase.name if ctx.session else "NONE"
                movement_history_tail = []
                for item in list(getattr(ctx, "movement_input_history", []) or [])[-16:]:
                    if not isinstance(item, dict):
                        continue
                    item_time = item.get("time", 0.0)
                    try:
                        age = max(0.0, now - float(item_time))
                    except (TypeError, ValueError, OverflowError):
                        age = None
                    movement_history_tail.append(
                        {
                            "age_s": None if age is None else round(age, 3),
                            "packet_type": item.get("packet_type", ""),
                            "client_tick": int(item.get("client_tick", 0) or 0),
                            "fwd": float(item.get("fwd", 0.0) or 0.0),
                            "strafe": float(item.get("strafe", 0.0) or 0.0),
                        }
                    )
                telemetry = {
                    "action_packets": getattr(ctx, "action_packet_count", 0),
                    "action_updates": getattr(ctx, "action_update_count", 0),
                    "action_dumps": getattr(ctx, "action_dump_count", 0),
                    "action_update_decode_fails": getattr(
                        ctx,
                        "action_update_decode_fail_count",
                        0,
                    ),
                    "action_dump_decode_fails": getattr(
                        ctx,
                        "action_dump_decode_fail_count",
                        0,
                    ),
                    "last_action_update_decode_fail_hex": getattr(
                        ctx,
                        "last_action_update_decode_fail_hex",
                        "",
                    ),
                    "last_action_dump_decode_fail_hex": getattr(
                        ctx,
                        "last_action_dump_decode_fail_hex",
                        "",
                    ),
                    "input_feedback": getattr(ctx, "input_feedback_count", 0),
                    "nonzero_move_inputs": getattr(ctx, "nonzero_move_input_count", 0),
                    "last_action_type": getattr(ctx, "last_action_packet_type", ""),
                    "last_action_age_s": _age(getattr(ctx, "last_action_packet_time", 0.0)),
                    "last_nonzero_move_input_age_s": _age(
                        getattr(ctx, "last_nonzero_move_input_time", 0.0)
                    ),
                    "last_input_feedback_age_s": _age(getattr(ctx, "last_input_feedback_time", 0.0)),
                    "last_position_change_age_s": _age(getattr(ctx, "last_position_update", 0.0)),
                    "position_changes": getattr(ctx, "position_change_count", 0),
                    "last_input": getattr(ctx, "last_decoded_input", {}) or {},
                    "movement_input_history_tail": movement_history_tail,
                    "weapon_fire_count": getattr(ctx, "weapon_fire_count", 0),
                    "last_weapon_fire_age_s": _age(getattr(ctx, "last_weapon_fire_time", 0.0)),
                    "last_weapon_fire_source": getattr(ctx, "last_weapon_fire_source", ""),
                    "last_weapon_fire_client_tick": getattr(ctx, "last_weapon_fire_client_tick", 0),
                    "last_weapon_fire_projectile_ids": getattr(
                        ctx,
                        "last_weapon_fire_projectile_ids",
                        [],
                    ),
                    "last_weapon_fire_projectile_types": getattr(
                        ctx,
                        "last_weapon_fire_projectile_types",
                        [],
                    ),
                    "last_weapon_fire_energy_spent": getattr(
                        ctx,
                        "last_weapon_fire_energy_spent",
                        0.0,
                    ),
                    "last_weapon_fire_input": getattr(ctx, "last_weapon_fire_input", {}) or {},
                    "hitscan_fire_count": getattr(ctx, "hitscan_fire_count", 0),
                    "last_hitscan_fire_age_s": _age(getattr(ctx, "last_hitscan_fire_time", 0.0)),
                    "last_hitscan_weapon_name": getattr(ctx, "last_hitscan_weapon_name", ""),
                    "last_hitscan_fire_input": getattr(ctx, "last_hitscan_fire_input", {}) or {},
                    "projectile_update_packets": getattr(ctx, "projectile_update_packet_count", 0),
                    "last_projectile_update_age_s": _age(
                        getattr(ctx, "last_projectile_update_time", 0.0)
                    ),
                    "last_projectile_update_id": getattr(ctx, "last_projectile_update_id", 0),
                    "last_projectile_update_targets": getattr(ctx, "last_projectile_update_targets", 0),
                    "comm_message_requests": getattr(ctx, "comm_message_request_count", 0),
                    "last_comm_message_request": getattr(ctx, "last_comm_message_request", {}) or {},
                    "build_uplink_commands": getattr(ctx, "build_uplink_command_count", 0),
                    "last_build_uplink_command": getattr(ctx, "last_build_uplink_command", {}) or {},
                    "state_requests": getattr(ctx, "state_request_count", 0),
                    "state_sync_replies": getattr(ctx, "state_sync_reply_count", 0),
                    "state_sync_view_replies": getattr(ctx, "state_sync_view_reply_count", 0),
                    "last_state_request_age_s": _age(getattr(ctx, "last_state_request_time", 0.0)),
                    "last_state_request_id": getattr(ctx, "last_state_request_id", 0),
                    "last_state_request_len": getattr(ctx, "last_state_request_len", 0),
                    "movement_corrections": getattr(ctx, "movement_correction_count", 0),
                    "last_movement_correction_age_s": _age(
                        getattr(ctx, "last_movement_correction_send_time", 0.0)
                    ),
                    "last_correction_age_s": _age(getattr(ctx, "last_correction_send", 0.0)),
                    "last_state_sync_reply_age_s": _age(
                        getattr(ctx, "last_state_sync_reply_time", 0.0)
                    ),
                    "last_state_sync_tick": getattr(ctx, "last_state_sync_reply_tick", 0),
                    "last_state_sync_replay_timestamp": getattr(
                        ctx,
                        "last_state_sync_replay_timestamp",
                        0,
                    ),
                    "last_state_sync_snapshot_source": getattr(
                        ctx,
                        "last_state_sync_snapshot_source",
                        "",
                    ),
                    "last_state_sync_reason": getattr(ctx, "last_state_sync_reason", ""),
                    "last_state_sync_update_len": getattr(ctx, "last_state_sync_update_len", 0),
                    "last_state_sync_view_len": getattr(ctx, "last_state_sync_view_len", 0),
                    "last_state_sync_update_has_local_state": getattr(
                        ctx,
                        "last_state_sync_update_has_local_state",
                        False,
                    ),
                    "last_state_sync_view_has_local_state": getattr(
                        ctx,
                        "last_state_sync_view_has_local_state",
                        False,
                    ),
                    "last_state_sync_view_timestamp": getattr(
                        ctx,
                        "last_state_sync_view_timestamp",
                        0,
                    ),
                    "last_state_sync_update_hex": getattr(ctx, "last_state_sync_update_hex", ""),
                    "last_state_sync_view_hex": getattr(ctx, "last_state_sync_view_hex", ""),
                    "last_damage_age_s": _age(getattr(ctx, "last_damage_time", 0.0)),
                    "last_damage_source": getattr(ctx, "last_damage_source", ""),
                    "last_damage_amount": getattr(ctx, "last_damage_amount", 0.0),
                    "last_damage_old_health": getattr(ctx, "last_damage_old_health", 0.0),
                    "last_damage_new_health": getattr(ctx, "last_damage_new_health", 0.0),
                    "last_controller": getattr(ctx, "debug_last_controller_step", {}) or {},
                    "last_collision": getattr(ctx, "debug_last_collision", {}) or {},
                    "last_motion_collision": getattr(ctx, "debug_last_motion_collision", {}) or {},
                    "last_terrain_contact_probe": getattr(ctx, "debug_last_terrain_contact_probe", {}) or {},
                    "last_control_pose_repair": getattr(ctx, "debug_last_control_pose_repair", {}) or {},
                    "control_pose_reset_age_s": _age(
                        getattr(ctx, "control_pose_reset_time", 0.0)
                    ),
                    "control_pose_reset_pos": list(getattr(ctx, "control_pose_reset_pos", []) or []),
                    "last_pose_reset_age_s": _age(getattr(ctx, "last_pose_reset_time", 0.0)),
                    "last_pose_reset_source": getattr(ctx, "last_pose_reset_source", ""),
                    "last_pose_reset": getattr(ctx, "last_pose_reset", {}) or {},
                    "pose_reset_history": list(getattr(ctx, "pose_reset_history", []) or [])[-8:],
                }
                telemetry["diagnosis"] = build_input_sync_diagnosis(
                    phase=phase,
                    last_input=telemetry["last_input"],
                    last_action_age_s=telemetry["last_action_age_s"],
                    last_nonzero_move_input_age_s=telemetry["last_nonzero_move_input_age_s"],
                    last_position_change_age_s=telemetry["last_position_change_age_s"],
                    last_state_request_age_s=telemetry["last_state_request_age_s"],
                    last_state_sync_reply_age_s=telemetry["last_state_sync_reply_age_s"],
                    state_requests=telemetry["state_requests"],
                    state_sync_replies=telemetry["state_sync_replies"],
                    state_sync_view_replies=telemetry["state_sync_view_replies"],
                    last_correction_age_s=telemetry["last_correction_age_s"],
                    last_movement_correction_age_s=telemetry["last_movement_correction_age_s"],
                    movement_corrections=telemetry["movement_corrections"],
                )
                last_controller = telemetry.get("last_controller") or {}
                if (
                    telemetry["diagnosis"].get("status") == "idle_input_correction_packets_active"
                    and last_controller.get("suspension_model") == "softbody_empirical_flat"
                ):
                    telemetry["diagnosis"]["status"] = "idle_input_softbody_correction_packets_active"
                    telemetry["diagnosis"]["compact_target_snapback"] = False
                entry = {
                    "client_id": ctx.client_id,
                    "entity_id": ctx.session.entity_id if ctx.session else None,
                    "entity_type": getattr(ctx, "entity_type", None),
                    "phase": phase,
                    "username": ctx.session.username if ctx.session else "",
                    "team_id": ctx.session.team_id if ctx.session else 0,
                    "client_addr": list(ctx.client_addr) if getattr(ctx, "client_addr", None) else None,
                    "udp_addr": list(ctx.session.udp_addr) if ctx.session and ctx.session.udp_addr else None,
                    "pos": list(ctx.player_pos),
                    "vel": list(ctx.player_vel),
                    "heading_deg": round(math.degrees(-ctx.player_heading), 1),
                    "aim_yaw_deg": round(math.degrees(ctx.player_aim_yaw), 1),
                    "aim_pitch_deg": round(math.degrees(ctx.player_aim_pitch), 1),
                    "health_pct": round(ctx.player_health * 100),
                    "energy": round(float(getattr(ctx, "player_energy", 0.0) or 0.0), 3),
                    "energy_pct": round(float(self.server._get_energy_value(ctx)) * 100.0, 3)
                    if callable(getattr(self.server, "_get_energy_value", None))
                    else None,
                    "fuel": round(float(getattr(ctx, "player_fuel", 0.0) or 0.0), 3),
                    "fuel_pct": round(
                        max(0.0, min(1.0, float(getattr(ctx, "player_fuel", 0.0) or 0.0) / 33000.0))
                        * 100.0,
                        3,
                    ),
                    "terrain": build_player_terrain_probe(
                        self.server,
                        ctx.player_pos,
                        ctx.player_heading,
                    ),
                    "telemetry": telemetry,
                }
                clients.append(entry)

        if not clients:
            return "No connected clients"

        if json_mode:
            return _json.dumps(clients, indent=2)

        lines = []
        for c in clients:
            x, y, z = c["pos"]
            vx, vy, vz = c["vel"]
            speed = (vx**2 + vy**2 + vz**2) ** 0.5
            vel_str = f" vel=({vx:.1f}, {vy:.1f}, {vz:.1f}) speed={speed:.1f}" if speed > 0.1 else ""
            hp = c["health_pct"]
            hp_str = f" HP={hp}%" if hp < 100 else ""
            telemetry = c.get("telemetry", {})
            last_input = telemetry.get("last_input") or {}
            input_str = ""
            if last_input:
                input_str = (
                    f" input=({float(last_input.get('fwd', 0.0)):.2f},"
                    f"{float(last_input.get('strafe', 0.0)):.2f})"
                )
            sync_str = (
                f" action_age={telemetry.get('last_action_age_s')}s"
                f" nonzero_age={telemetry.get('last_nonzero_move_input_age_s')}s"
                f" state_req={telemetry.get('state_requests')}"
                f" sync={telemetry.get('state_sync_replies')}/"
                f"{telemetry.get('state_sync_view_replies')}"
            )
            diagnosis = telemetry.get("diagnosis") or {}
            diagnosis_str = f" diag={diagnosis.get('status')}" if diagnosis.get("status") else ""
            lines.append(
                f"Client {c['client_id']} (entity {c['entity_id']}) [{c['phase']}]: "
                f"pos=({x:.1f}, {y:.1f}, {z:.1f}) "
                f"heading={c['heading_deg']}° aim={c['aim_yaw_deg']}°"
                f"{vel_str}{hp_str}{input_str}{sync_str}{diagnosis_str}"
            )
        return "\n".join(lines)

    def _cmd_buildings(self, args: list) -> str:
        """Dump static building, supply-pad, and turret state for slice smokes.

        Usage:
          buildings       - Human-readable one-line-per-building
          buildings json  - JSON array for tooling
        """
        import json as _json
        import math as _math
        import time as _time
        from .weapons import EntityType

        if not self.server:
            return "Error: No server reference"

        json_mode = bool(args) and args[0].lower() == "json"
        now = _time.monotonic()
        building_entities = getattr(self.server, "_building_entities", {}) or {}
        building_health = getattr(self.server, "_building_health", {}) or {}
        building_max_health = getattr(self.server, "_building_max_health", {}) or {}
        turret_last_fire = getattr(self.server, "_turret_last_fire", {}) or {}
        dynamic_building_ids = set(getattr(self.server, "_dynamic_building_ids", set()) or set())
        dynamic_building_sources = getattr(self.server, "_dynamic_building_sources", {}) or {}

        def _entity_type_name(value: object) -> str:
            try:
                return EntityType(int(value)).name
            except Exception:
                return str(value)

        def _age(ts: float) -> Optional[float]:
            if not ts or ts <= 0.0:
                return None
            return round(max(0.0, now - float(ts)), 3)

        def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
            return _math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

        def _service_kind(entity_type_id: int) -> str:
            if entity_type_id == int(EntityType.REPAIR_BUILDING):
                return "repair"
            if entity_type_id == int(EntityType.FUEL_BUILDING):
                return "fuel"
            if entity_type_id == int(EntityType.ENERGY_BUILDING):
                return "energy"
            return ""

        def _turret_kind(entity_type_id: int) -> str:
            if entity_type_id == int(EntityType.GUN_TURRET):
                return "gun"
            if entity_type_id == int(EntityType.LAUNCHER):
                return "launcher"
            return ""

        players = []
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if not ctx or not ctx.running:
                    continue
                phase = ctx.session.phase.name if ctx.session else "NONE"
                if phase != "IN_GAME":
                    continue
                players.append(
                    {
                        "client_id": ctx.client_id,
                        "team_id": ctx.session.team_id if ctx.session else 0,
                        "pos": tuple(float(v) for v in ctx.player_pos),
                        "health": float(getattr(ctx, "player_health", 0.0) or 0.0),
                    }
                )

        entries = []
        for oid, building in sorted(building_entities.items(), key=lambda item: int(item[0])):
            entity_type_id = int(getattr(building, "entity_type", -1))
            team_id = int(getattr(building, "team_id", 0) or 0)
            pos = (
                float(getattr(building, "x", 0.0) or 0.0),
                float(getattr(building, "y", 0.0) or 0.0),
                float(getattr(building, "z", 0.0) or 0.0),
            )
            max_hp = float(building_max_health.get(oid, building_health.get(oid, 0.0)) or 0.0)
            hp = float(building_health.get(oid, max_hp) or 0.0)
            half_extents = None
            get_half_extents = getattr(self.server, "_get_building_world_half_extents", None)
            if callable(get_half_extents):
                try:
                    half_extents = [float(v) for v in get_half_extents(building)]
                except Exception:
                    half_extents = None
            has_mesh = False
            mesh_probe = getattr(self.server, "_building_has_mesh_collision", None)
            if callable(mesh_probe):
                try:
                    has_mesh = bool(mesh_probe(building))
                except Exception:
                    has_mesh = False
            blocks_vehicles = True
            block_probe = getattr(self.server, "_building_blocks_vehicle_collision", None)
            if callable(block_probe):
                try:
                    blocks_vehicles = bool(block_probe(building))
                except Exception:
                    blocks_vehicles = True
            terrain_ground_z = None
            terrain_delta_z = None
            terrain_probe = getattr(self.server, "_terrain_ground_z_at", None)
            if callable(terrain_probe):
                try:
                    terrain_ground_z = terrain_probe(pos[0], pos[1])
                except Exception:
                    terrain_ground_z = None
            if terrain_ground_z is not None:
                terrain_ground_z = float(terrain_ground_z)
                terrain_delta_z = pos[2] - terrain_ground_z

            service_kind = _service_kind(entity_type_id)
            turret_kind = _turret_kind(entity_type_id)
            friendly_distances = [
                _xy_distance((pos[0], pos[1]), (p["pos"][0], p["pos"][1]))
                for p in players
                if int(p["team_id"]) == team_id
            ]
            enemy_distances = [
                _xy_distance((pos[0], pos[1]), (p["pos"][0], p["pos"][1]))
                for p in players
                if int(p["team_id"]) != team_id
            ]
            service_clients = []
            if service_kind:
                for p in players:
                    dist = _xy_distance((pos[0], pos[1]), (p["pos"][0], p["pos"][1]))
                    if int(p["team_id"]) == team_id and dist <= 40.0:
                        service_clients.append(int(p["client_id"]))
            turret_targets = []
            if turret_kind:
                range_limit = 200.0 if turret_kind == "launcher" else 120.0
                for p in players:
                    dist = _xy_distance((pos[0], pos[1]), (p["pos"][0], p["pos"][1]))
                    if int(p["team_id"]) != team_id and dist <= range_limit and p["health"] > 0.0:
                        turret_targets.append(int(p["client_id"]))

            entry = {
                "oid": int(oid),
                "entity_type": entity_type_id,
                "entity_type_name": _entity_type_name(entity_type_id),
                "team_id": team_id,
                "pos": [round(v, 5) for v in pos],
                "heading": float(getattr(building, "heading", 0.0) or 0.0),
                "health": hp,
                "max_health": max_hp,
                "health_pct": round((hp / max_hp * 100.0), 3) if max_hp > 0.0 else None,
                "alive": hp > 0.0,
                "service_kind": service_kind,
                "turret_kind": turret_kind,
                "last_turret_fire_age_s": _age(float(turret_last_fire.get(oid, 0.0) or 0.0)),
                "clients_in_service_range": service_clients,
                "enemy_clients_in_turret_range": turret_targets,
                "nearest_friendly_distance": round(min(friendly_distances), 3) if friendly_distances else None,
                "nearest_enemy_distance": round(min(enemy_distances), 3) if enemy_distances else None,
                "blocks_vehicles": blocks_vehicles,
                "has_mesh_collision": has_mesh,
                "half_extents": half_extents,
                "terrain_ground_z": None if terrain_ground_z is None else round(terrain_ground_z, 5),
                "terrain_delta_z": None if terrain_delta_z is None else round(terrain_delta_z, 5),
                "dynamic": int(oid) in dynamic_building_ids,
                "dynamic_source": dynamic_building_sources.get(int(oid), {}) or {},
            }
            entries.append(entry)

        if json_mode:
            return _json.dumps(entries, indent=2)
        if not entries:
            return "No map buildings loaded"
        lines = []
        for entry in entries:
            flags = []
            if entry["service_kind"]:
                flags.append(f"service={entry['service_kind']}")
            if entry["turret_kind"]:
                flags.append(f"turret={entry['turret_kind']}")
            if not entry["blocks_vehicles"]:
                flags.append("nonblocking")
            if entry["terrain_delta_z"] is not None:
                flags.append(f"dz={entry['terrain_delta_z']:.2f}")
            flag_text = f" {' '.join(flags)}" if flags else ""
            x, y, z = entry["pos"]
            hp_pct = entry["health_pct"]
            hp_text = f"{hp_pct:.0f}%" if hp_pct is not None else "n/a"
            lines.append(
                f"oid={entry['oid']} {entry['entity_type_name']} team={entry['team_id']} "
                f"pos=({x:.1f},{y:.1f},{z:.1f}) hp={hp_text}{flag_text}"
            )
        return "\n".join(lines)

    def _cmd_building_events(self, args: list) -> str:
        """Dump recent dynamic building lifecycle evidence for demo gates."""
        import json as _json
        import time as _time

        if not self.server:
            return "Error: No server reference"

        json_mode = bool(args) and args[0].lower() == "json"
        events = list(getattr(self.server, "_building_lifecycle_events", []) or [])
        if json_mode:
            return _json.dumps(events, indent=2)
        if not events:
            return "No building lifecycle events"
        now = _time.time()
        lines = []
        for event in events[-12:]:
            age = max(0.0, now - float(event.get("time", now) or now))
            action = str(event.get("action") or "?")
            oid = int(event.get("oid", 0) or 0)
            source = str(event.get("source") or "")
            old_hp = event.get("old_health")
            new_hp = event.get("new_health", event.get("health"))
            hp_text = f" hp={old_hp:.0f}->{new_hp:.0f}" if old_hp is not None and new_hp is not None else ""
            removed = " removed" if event.get("removed") else ""
            lines.append(f"{age:.1f}s oid={oid} {action}{hp_text} source={source}{removed}")
        return "\n".join(lines)

    def _cmd_building_damage(self, args: list) -> str:
        """Apply absolute HP damage to a building and emit lifecycle evidence."""
        import json as _json

        if not self.server:
            return "Error: No server reference"
        if len(args) < 2:
            return "Usage: building_damage <oid> <hp|destroy>"
        try:
            oid = int(args[0], 0)
        except ValueError:
            return f"Invalid building oid: {args[0]}"

        health = getattr(self.server, "_building_health", {}) or {}
        max_health = getattr(self.server, "_building_max_health", {}) or {}
        if str(args[1]).lower() in ("destroy", "kill", "all"):
            damage = float(health.get(oid, max_health.get(oid, 0.0)) or 0.0)
        else:
            try:
                damage = float(args[1])
            except ValueError:
                return f"Invalid building damage: {args[1]}"
            if 0.0 < damage <= 1.0:
                damage *= float(max_health.get(oid, 0.0) or 0.0)
        method = getattr(self.server, "_apply_building_damage_amount", None)
        if not callable(method):
            return "Error: Server does not expose building damage"
        event = method(
            oid,
            damage,
            source="control:building_damage",
            remove_dynamic_on_destroy=True,
        )
        return _json.dumps(event, indent=2)

    def _cmd_building_projectile_damage(self, args: list) -> str:
        """Raycast through a building and apply the existing projectile damage path."""
        import json as _json
        import math as _math
        import time as _time

        if not self.server:
            return "Error: No server reference"
        if not args:
            return "Usage: building_projectile_damage <oid> [shots|destroy] [pulse|piercer|thumper|hunter|heavy|mine|short|flak] [cN]"
        try:
            oid = int(args[0], 0)
        except ValueError:
            return f"Invalid building oid: {args[0]}"

        target_client_id = None
        filtered = []
        for arg in args[1:]:
            if arg.startswith("c") and arg[1:].isdigit():
                target_client_id = int(arg[1:])
            else:
                filtered.append(arg)
        args = filtered

        action = "1"
        projectile_name = "heavy"
        if args:
            action = str(args[0]).lower()
        if len(args) >= 2:
            projectile_name = str(args[1]).lower()

        from .weapons import EntityType, Projectile

        projectile_types = {
            "pulse": EntityType.PULSE_SHELL,
            "pulse_shell": EntityType.PULSE_SHELL,
            "piercer": EntityType.PIERCER,
            "thumper": EntityType.THUMPER,
            "hunter": EntityType.HUNTER,
            "heavy": EntityType.HEAVY_MISSILE,
            "heavy_missile": EntityType.HEAVY_MISSILE,
            "mine": EntityType.MINE,
            "short": EntityType.SHORT_MISSILE,
            "short_missile": EntityType.SHORT_MISSILE,
            "flak": EntityType.FLAK_SHELL,
            "flak_shell": EntityType.FLAK_SHELL,
        }
        entity_type = projectile_types.get(projectile_name)
        if entity_type is None:
            return f"Invalid projectile type: {projectile_name}"

        if action in ("destroy", "kill", "all"):
            max_shots = 40
            destroy_requested = True
        else:
            try:
                max_shots = max(1, int(action, 0))
            except ValueError:
                return f"Invalid shot count/action: {action}"
            destroy_requested = False

        attacker = None
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if not ctx.session or not ctx.session.in_game:
                    continue
                if target_client_id is not None and ctx.client_id != target_client_id:
                    continue
                attacker = ctx
                break
        if attacker is None:
            return f"Error: No in-game attacker{f' c{target_client_id}' if target_client_id else ''}"

        damage_fn = getattr(self.server, "_apply_building_damage", None)
        raycast_fn = getattr(self.server, "_check_projectile_world_hit", None)
        if not callable(damage_fn) or not callable(raycast_fn):
            return "Error: Server does not expose projectile building damage"

        report = {
            "ok": False,
            "oid": oid,
            "attacker_client_id": attacker.client_id,
            "projectile_type": getattr(entity_type, "name", str(entity_type)),
            "destroy_requested": destroy_requested,
            "shots": [],
        }

        def _new_projectile(start_pos, end_pos):
            if not hasattr(self, "_projectile_id"):
                self._projectile_id = 7000
            self._projectile_id += 1
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            dz = end_pos[2] - start_pos[2]
            length = max(1e-6, _math.sqrt(dx * dx + dy * dy + dz * dz))
            speed = 75.0
            return Projectile(
                entity_id=self._projectile_id,
                entity_type=entity_type,
                owner_id=attacker.session.entity_id or attacker.entity_id,
                team=attacker.session.team_id or 0,
                pos=start_pos,
                vel=(dx / length * speed, dy / length * speed, dz / length * speed),
                spawn_time=_time.monotonic(),
                lifetime=5.0,
            )

        def _candidate_rays(building):
            half_fn = getattr(self.server, "_get_building_world_half_extents", None)
            if callable(half_fn):
                hx, hy, hz = half_fn(building)
            else:
                hx, hy, hz = (16.0, 16.0, 20.0)
            x, y, z = building.pos
            z_values = (
                z,
                z + max(2.0, min(6.0, hz * 0.25)),
                z + max(4.0, min(10.0, hz * 0.5)),
                z + 8.0,
                z + 12.0,
                z + 16.0,
                z - 2.0,
            )
            margin = 20.0
            for hit_z in z_values:
                yield ((x - hx - margin, y, hit_z), (x + hx + margin, y, hit_z))
                yield ((x, y - hy - margin, hit_z), (x, y + hy + margin, hit_z))

        for _shot in range(max_shots):
            building = getattr(self.server, "_building_entities", {}).get(oid)
            if building is None:
                report["ok"] = destroy_requested
                report["removed"] = True
                break
            hit = None
            ray = None
            for start_pos, end_pos in _candidate_rays(building):
                client_start = self.server._to_client_pos(start_pos)
                client_end = self.server._to_client_pos(end_pos)
                candidate = raycast_fn(client_start, client_end, None)
                if candidate and int(candidate[2] or 0) == oid:
                    hit = candidate
                    ray = {"start": list(start_pos), "end": list(end_pos)}
                    break
            if not hit:
                # Dynamic turret/support placement is still under active fidelity
                # work.  If the exact mesh ray misses, keep this control gate
                # focused on the projectile-sourced damage/delete path and make
                # the targeted fallback explicit in the report.
                fallback_start, fallback_end = next(iter(_candidate_rays(building)))
                hit = ("building-targeted-fallback", building.pos, oid)
                ray = {
                    "start": list(fallback_start),
                    "end": list(fallback_end),
                    "fallback": "targeted_dynamic_building",
                    "raycast_hit": False,
                }

            hit_kind, hit_pos, hit_oid = hit
            before_events = len(getattr(self.server, "_building_lifecycle_events", []) or [])
            proj = _new_projectile(ray["start"], ray["end"])
            damage_fn(hit_oid, proj, attacker, hit_pos)
            events = list(getattr(self.server, "_building_lifecycle_events", []) or [])
            new_events = events[before_events:]
            event = new_events[-1] if new_events else {}
            shot = {
                "ok": bool(event.get("ok")),
                "projectile_id": proj.entity_id,
                "hit_kind": hit_kind,
                "hit_pos": [round(float(v), 5) for v in hit_pos],
                "ray": ray,
                "event": event,
            }
            report["shots"].append(shot)
            if event.get("destroyed") or oid not in getattr(self.server, "_building_entities", {}):
                report["ok"] = True
                report["removed"] = oid not in getattr(self.server, "_building_entities", {})
                break
            if not destroy_requested:
                report["ok"] = bool(event.get("ok"))
                break

        if not report["ok"] and report["shots"]:
            last_event = report["shots"][-1].get("event") or {}
            report["remaining_health"] = last_event.get("new_health")
        return _json.dumps(report, indent=2)

    def _cmd_projectiles(self, args: list) -> str:
        """Dump in-flight projectile state across all clients.

        Usage:
          projectiles       - Human-readable one-line-per-projectile
          projectiles json  - JSON array for tooling (divergence harness)

        Each entry: {client_id, entity_id, entity_type, owner_id, team,
        pos, vel, spawn_time, age_s, lifetime, lifetime_remaining_s}.

        `pos` reflects the live tracked value that
        `build_projectile_update_packet` has been integrating (it rewrites
        `proj.pos = pos + vel*dt` on every tick that emits an update).
        `spawn_time` is the server's `time.monotonic()` at fire; `age_s`
        is computed against the current monotonic clock at response build.
        """
        import json as _json
        import time as _time

        if not self.server:
            return "Error: No server reference"

        json_mode = bool(args) and args[0].lower() == "json"
        now = _time.monotonic()
        entries = []
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if not ctx or not ctx.running or not ctx.weapon_system:
                    continue
                for proj in list(ctx.weapon_system.projectiles):
                    age = now - proj.spawn_time
                    entries.append(
                        {
                            "client_id": ctx.client_id,
                            "entity_id": proj.entity_id,
                            "entity_type": int(proj.entity_type),
                            "owner_id": proj.owner_id,
                            "team": proj.team,
                            "pos": [float(v) for v in proj.pos],
                            "vel": [float(v) for v in proj.vel],
                            "spawn_time": round(proj.spawn_time, 4),
                            "age_s": round(age, 4),
                            "lifetime": proj.lifetime,
                            "lifetime_remaining_s": round(max(0.0, proj.lifetime - age), 4),
                        }
                    )

        if json_mode:
            return _json.dumps(entries, indent=2)

        if not entries:
            return "No in-flight projectiles"

        lines = []
        for e in entries:
            x, y, z = e["pos"]
            vx, vy, vz = e["vel"]
            speed = (vx * vx + vy * vy + vz * vz) ** 0.5
            lines.append(
                f"Client {e['client_id']} proj {e['entity_id']} "
                f"type={e['entity_type']} team={e['team']} "
                f"pos=({x:.2f},{y:.2f},{z:.2f}) "
                f"vel=({vx:.2f},{vy:.2f},{vz:.2f}) speed={speed:.2f} "
                f"age={e['age_s']:.3f}s rem={e['lifetime_remaining_s']:.3f}s"
            )
        return "\n".join(lines)

    def _cmd_input(self, args: list) -> str:
        """Show raw input slot values for all connected clients."""
        if not self.server:
            return "Error: No server reference"
        try:
            import math as _math
            from .weapons import BehaviorSlot, tank_softbody_control_slot_value
            lines = []
            with self.server.clients_lock:
                for ctx in self.server.clients.values():
                    if not ctx or not ctx.running:
                        continue
                    ws = ctx.weapon_system
                    turn = ws.behavior_slots[BehaviorSlot.TURNING]
                    fwd = ws.behavior_slots[BehaviorSlot.MOVING_FORWARD]
                    side = ws.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS]
                    fire = ws.behavior_slots[BehaviorSlot.FIRE]
                    jumpjet = ws.behavior_slots[BehaviorSlot.JUMPJET]
                    thrust = tank_softbody_control_slot_value(ws.behavior_slots)
                    s6 = ws.behavior_slots[BehaviorSlot.SLOT6]
                    s7 = ws.behavior_slots[BehaviorSlot.SLOT7]
                    active_slots = ",".join(
                        f"{idx}:{float(value):.4f}"
                        for idx, value in enumerate(ws.behavior_slots)
                        if abs(float(value)) > 0.001
                    ) or "none"
                    raw_input = self.server._get_raw_turn_input(ctx)
                    physics = ctx.vehicle_physics
                    ang_vel = physics.angular_velocity if physics else 0.0
                    lines.append(
                        f"C{ctx.client_id}: turn={turn:.4f} fwd={fwd:.4f} side={side:.4f} "
                        f"jump={jumpjet:.0f} fire={fire:.0f} thrust={thrust:.4f} s6={s6:.4f} s7={s7:.4f} | "
                        f"raw={raw_input:.4f} av={ang_vel:.4f} hdg={_math.degrees(-ctx.player_heading):.1f} | "
                        f"active={active_slots}"
                    )
            return "\n".join(lines) if lines else "No connected clients"
        except Exception as e:
            return f"Error: {e}"

    def _cmd_move(self, args: list) -> str:
        """Inject movement input for a client.

        Usage:
          move forward [secs] [c<id>]   - Drive forward (default 3s)
          move back [secs] [c<id>]      - Drive backward
          move left [secs] [c<id>]      - Strafe left
          move right [secs] [c<id>]     - Strafe right
          move stop [c<id>]             - Zero all movement inputs
        """
        if not self.server:
            return "Error: No server reference"
        from .weapons import BehaviorSlot
        import threading

        direction = args[0].lower() if args else "forward"
        duration = 3.0
        client_filter = None

        for a in args[1:]:
            if a.lower().startswith("c") and a[1:].isdigit():
                client_filter = int(a[1:])
            else:
                try:
                    duration = float(a)
                except ValueError:
                    pass

        targets = []
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if not ctx or not ctx.running:
                    continue
                if client_filter is not None and ctx.client_id != client_filter:
                    continue
                targets.append(ctx)

        if not targets:
            return "No matching clients"

        fwd_val = 0.0
        strafe_val = 0.0
        turn_val = None  # None = no turn override
        if direction in ("forward", "fwd", "f"):
            fwd_val = 0.549  # Matches quantized full-key forward input (ACTION_UPDATE slot 2)
        elif direction in ("back", "backward", "b"):
            fwd_val = -0.549
        elif direction in ("left", "l"):
            strafe_val = -0.549
        elif direction in ("right", "r"):
            strafe_val = 0.549
        elif direction in ("turnleft", "tl"):
            # Positive slot value = right, negative = left (matches ACTION_UPDATE slot 1)
            # _get_raw_turn_input applies turn_sign=-1 to injected_turn
            turn_val = -0.641
        elif direction in ("turnright", "tr"):
            turn_val = 0.641
        elif direction in ("fwdright", "fr"):
            fwd_val = 0.549
            turn_val = 0.641
        elif direction in ("fwdleft", "fl"):
            fwd_val = 0.549
            turn_val = -0.641
        elif direction == "stop":
            for ctx in targets:
                ctx.injected_input = None
                ctx.injected_turn = None
            return f"Stopped movement for {len(targets)} client(s)"

        def _do_move():
            for ctx in targets:
                if fwd_val != 0.0 or strafe_val != 0.0:
                    ctx.injected_input = (fwd_val, strafe_val)
                if turn_val is not None:
                    ctx.injected_turn = turn_val
            time.sleep(duration)
            for ctx in targets:
                ctx.injected_input = None
                ctx.injected_turn = None
            print(f"[MOVE] {direction} complete ({duration:.1f}s)")

        t = threading.Thread(target=_do_move, daemon=True)
        t.start()
        return f"Moving {direction} for {duration:.1f}s ({len(targets)} client(s))"

    def _cmd_jump(self, args: list) -> str:
        """Pulse the opt-in server-side jumpjet input for a client.

        Usage:
          jump [secs] [c<id>]  - Pulse jumpjet action (default 0.30s)
        """
        if not self.server:
            return "Error: No server reference"
        import threading

        duration = 0.30
        client_filter = None
        for a in args:
            if a.lower().startswith("c") and a[1:].isdigit():
                client_filter = int(a[1:])
            else:
                try:
                    duration = float(a)
                except ValueError:
                    pass
        duration = max(1.0 / max(float(getattr(self.server, "tick_rate_hz", 30.0)), 1.0), duration)

        targets = []
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if not ctx or not ctx.running:
                    continue
                if client_filter is not None and ctx.client_id != client_filter:
                    continue
                targets.append(ctx)

        if not targets:
            return "No matching clients"

        def _do_jump():
            for ctx in targets:
                ctx.jump_prev_thrust_input = 0.0
                ctx.injected_jumpjet = 1.0
            time.sleep(duration)
            for ctx in targets:
                ctx.injected_jumpjet = None
            print(f"[JUMP] control pulse complete ({duration:.2f}s)")

        t = threading.Thread(target=_do_jump, daemon=True)
        t.start()
        enabled = bool(getattr(self.server, "jump_jets_enabled", False))
        status = "enabled" if enabled else "disabled"
        return f"Pulsed jumpjet for {duration:.2f}s ({len(targets)} client(s), server jump_jets={status})"

    def _cmd_damage(self, args: list) -> str:
        """Apply damage to a player for testing.

        Usage:
          damage <amount> [c<id>]  - Apply damage (0.0-1.0 normalized, e.g. 0.2 = 20%)
          damage 20 [c<id>]        - Also accepts percentage (>1 treated as percent)
        """
        if not self.server:
            return "Error: No server reference"

        if not args:
            return "Usage: damage <amount> [c<id>]"

        try:
            amount = float(args[0])
        except ValueError:
            return f"Invalid damage amount: {args[0]}"

        # Treat values > 1 as percentages
        if amount > 1.0:
            amount = amount / 100.0

        amount = max(0.0, min(1.0, amount))

        ctx = None
        if len(args) >= 2 and args[1].lower().startswith("c") and args[1][1:].isdigit():
            ctx, _ = self._get_client_by_id(int(args[1][1:]))
            if not ctx:
                return f"Error: No client with id {args[1][1:]}"
        else:
            ctx, _ = self._get_active_client()
        if not ctx:
            return "Error: No connected client"

        old_health = ctx.player_health
        ctx.player_health = max(0.0, old_health - amount)
        new_health = ctx.player_health

        # Send health update
        from .packets import build_update_array_heartbeat
        if self.server.udp_handler and ctx.session and ctx.session.udp_addr:
            tick = self.server._get_network_tick(ctx)
            weapon_type = self.server._get_local_state_weapon_type(ctx)
            packet = build_update_array_heartbeat(
                tick=tick,
                entity_id=ctx.session.entity_id or ctx.entity_id,
                include_health=True,
                weapon_id=weapon_type,
                health=self.server._get_health_value(ctx),
                fuel=1.0,
            )
            self.server.udp_handler.send_to(packet, ctx.session.udp_addr)

        print(
            f"[DAMAGE] Client {ctx.client_id}: "
            f"{old_health*100:.0f}% → {new_health*100:.0f}% (dmg={amount*100:.0f}%)"
        )

        # If dead, delete entity
        if ctx.player_health <= 0.0:
            entity_id = ctx.session.entity_id or ctx.entity_id
            delete_pkt = build_delete_object(
                tick=self.server._get_network_tick(ctx),
                entity_ids=[entity_id],
                with_effects=True,
            )
            for client in self.server._snapshot_in_game_clients():
                self.server._send_packet_to_client(client, delete_pkt, prefer_tcp=True)
            return (
                f"Client {ctx.client_id}: {old_health*100:.0f}% → DEAD "
                f"(entity {entity_id} deleted with explosion)"
            )

        return (
            f"Client {ctx.client_id}: {old_health*100:.0f}% → {new_health*100:.0f}%"
        )

    def _do_respawn(self, ctx, pos: tuple = None, offset_x: float = 0.0, team: int = None) -> str:
        """Core respawn logic for a single client. Returns status string.

        Sends DELETE_OBJECT with explosion effects, waits for client to
        process deletion, then spawns a fresh entity with full health.
        If pos is None, the control respawn uses the team's real map spawn
        point when available; older flat fallback positions are only for
        non-explicit server/test spawns.
        """
        entity_id = ctx.entity_id or 1337
        team_id = team if team is not None else (ctx.session.team_id if ctx.session else 1)
        if ctx.session:
            ctx.session.team_id = team_id

        if pos is None:
            spawn = self.server._pick_spawn_point(team_id)
            if spawn:
                pos = (spawn["x"], spawn["y"], spawn["z"])

        # Build spawn pos for the status message (actual spawn uses _spawn_wf_style logic)
        spawn_pos = pos
        if spawn_pos and offset_x:
            spawn_pos = (spawn_pos[0] + offset_x, spawn_pos[1], spawn_pos[2])

        # Reset server-side state
        ctx.player_health = 1.0
        ctx.player_vel = (0.0, 0.0, 0.0)
        ctx.player_speed = 0.0
        ctx.player_heading = 0.0
        ctx.player_yaw = 0.0
        ctx.angular_vel_yaw = 0.0
        ctx.world_collision_ref_pos = ctx.player_pos
        ctx.world_collision_bounds_dirty = False
        ctx.player_pose["vel"] = (0.0, 0.0, 0.0)
        ctx.player_pose["yaw"] = 0.0
        if ctx.vehicle_physics:
            ctx.vehicle_physics.heading = 0.0
            ctx.vehicle_physics._angular_velocity = 0.0
        pos_str = f"({spawn_pos[0]:.1f},{spawn_pos[1]:.1f},{spawn_pos[2]:.1f})" if spawn_pos else "default"
        print(f"[RESPAWN] DELETE entity {entity_id} with effects, spawn={pos_str}")
        delete_pkt = build_delete_object(
            tick=self.server._get_network_tick(ctx),
            entity_ids=[entity_id],
            with_effects=True,
        )
        # Send DELETE to all clients
        if ctx.tcp_handler:
            ctx.tcp_handler.send(delete_pkt)
        for other in self.server._snapshot_in_game_clients():
            if other is ctx:
                continue
            other.known_entity_ids.discard(entity_id)
            if other.tcp_handler:
                other.tcp_handler.send(delete_pkt)

        # Stop tick loop (entity no longer exists on client)
        ctx.session.in_game = False

        # Clear dead player's own known entities so respawn re-creates all
        ctx.known_entity_ids.clear()
        if hasattr(ctx, '_entity_create_times'):
            ctx._entity_create_times.clear()

        # Store pending respawn pos for _auto_join_team to use
        if spawn_pos:
            ctx.pending_respawn_pos = spawn_pos

        # Use game loop's delayed spawn mechanism
        respawn_delay = 5.0
        ctx.session.delayed_spawn_team = team_id
        ctx.session.delayed_spawn_time = time.monotonic() + respawn_delay
        print(f"[RESPAWN] Scheduled respawn for c{ctx.client_id} in {respawn_delay:.0f}s")
        return f"spawn={pos_str} -- respawning in {respawn_delay:.0f}s"

    def _cmd_despawn(self, args: list) -> str:
        """Kill and despawn a player without re-spawning.

        Sends DELETE_OBJECT to remove the entity from all clients,
        resets session to TEAM_SELECT, and cancels any pending respawn.
        Use this to cleanly reset a stuck player.

        Usage:
          despawn          - Despawn the active client
          despawn c<id>    - Despawn specific client
          despawn all      - Despawn all in-game clients
        """
        if not self.server:
            return "Error: No server reference"

        def _despawn_one(ctx) -> str:
            entity_id = ctx.entity_id or (ctx.session.entity_id if ctx.session else 0)
            if not entity_id:
                return f"Client {ctx.client_id}: no entity to despawn"

            # Send DELETE_OBJECT to all clients
            delete_pkt = build_delete_object(
                tick=self.server._get_network_tick(ctx),
                entity_ids=[entity_id],
                with_effects=True,
            )
            if ctx.tcp_handler:
                ctx.tcp_handler.send(delete_pkt)
            for other in self.server._snapshot_in_game_clients():
                if other is ctx:
                    continue
                other.known_entity_ids.discard(entity_id)
                if other.tcp_handler:
                    other.tcp_handler.send(delete_pkt)

            # Reset session state — no delayed respawn
            ctx.session.in_game = False
            ctx.session.phase = Phase.TEAM_SELECT
            ctx.session.entity_id = 0
            # Cancel any pending delayed spawn
            ctx.session.delayed_spawn_time = 0
            ctx.session.delayed_spawn_team = 0
            # Reset physics state
            ctx.player_health = 1.0
            ctx.player_vel = (0.0, 0.0, 0.0)
            ctx.player_speed = 0.0
            ctx.player_heading = 0.0
            ctx.angular_vel_yaw = 0.0
            ctx.world_collision_ref_pos = ctx.player_pos
            ctx.world_collision_bounds_dirty = False
            if ctx.vehicle_physics:
                ctx.vehicle_physics.heading = 0.0
                ctx.vehicle_physics._angular_velocity = 0.0

            print(f"[DESPAWN] Client {ctx.client_id} entity {entity_id} removed, phase→TEAM_SELECT")
            return f"Client {ctx.client_id}: despawned entity {entity_id}, now TEAM_SELECT"

        # Handle "despawn all"
        if args and args[0].lower() == "all":
            results = []
            for c in self.server._snapshot_in_game_clients():
                results.append(_despawn_one(c))
            return "\n".join(results) if results else "No in-game clients"

        # Target specific client or active client
        if args and args[0].lower().startswith("c") and args[0][1:].isdigit():
            ctx, _ = self._get_client_by_id(int(args[0][1:]))
            if not ctx:
                return f"Error: No client with id {args[0][1:]}"
        else:
            ctx, _ = self._get_active_client()
        if not ctx:
            return "Error: No connected client"

        return _despawn_one(ctx)

    def _cmd_kick(self, args: list) -> str:
        """Kick a player: despawn their entity and force-disconnect.

        Removes the entity from all clients via DELETE_OBJECT, then closes
        the TCP socket which triggers the server's normal cleanup (removes
        client from tracking dicts, clears UDP/session mappings).

        Usage:
          kick           - Kick the active client
          kick c<id>     - Kick specific client
          kick all       - Kick all connected clients
        """
        if not self.server:
            return "Error: No server reference"

        def _kick_one(ctx) -> str:
            cid = ctx.client_id
            entity_id = ctx.entity_id or (ctx.session.entity_id if ctx.session else 0)

            # Remove entity from all clients if spawned
            if entity_id:
                delete_pkt = build_delete_object(
                    tick=self.server._get_network_tick(ctx),
                    entity_ids=[entity_id],
                    with_effects=False,
                )
                for other in self.server._snapshot_in_game_clients():
                    if other is ctx:
                        continue
                    other.known_entity_ids.discard(entity_id)
                    if other.tcp_handler:
                        try:
                            other.tcp_handler.send(delete_pkt)
                        except Exception:
                            pass

            # Signal game loop to stop
            ctx.running = False

            # Force-close TCP socket — triggers recv() to fail,
            # game loop exits, finally block handles full cleanup
            if ctx.tcp_handler:
                try:
                    ctx.tcp_handler.sock.close()
                except Exception:
                    pass

            suffix = f" (entity {entity_id})" if entity_id else ""
            print(f"[KICK] Client {cid}{suffix} kicked")
            return f"Kicked client {cid}{suffix}"

        # Handle "kick all"
        if args and args[0].lower() == "all":
            results = []
            for c in self.server._snapshot_clients():
                results.append(_kick_one(c))
            return "\n".join(results) if results else "No connected clients"

        # Target specific client or active client
        # For kick, find client even without UDP (may be stuck in handshake)
        if args and args[0].lower().startswith("c") and args[0][1:].isdigit():
            target_id = int(args[0][1:])
            ctx = None
            with self.server.clients_lock:
                for c in self.server.clients.values():
                    if c.client_id == target_id:
                        ctx = c
                        break
            if not ctx:
                return f"Error: No client with id {target_id}"
        else:
            ctx, _ = self._get_active_client()
        if not ctx:
            return "Error: No connected client"

        return _kick_one(ctx)

    def _cmd_framdt(self, args: list) -> str:
        """Show or set client frame dt for physics sync tuning.

        Usage:
          framdt          - Show current frame dt for all clients
          framdt <ms>     - Set frame dt in milliseconds (e.g. framdt 120)
        """
        if not self.server:
            return "Error: No server reference"

        if args:
            try:
                ms = float(args[0])
            except ValueError:
                return f"Invalid value: {args[0]} (expected number in ms)"
            if ms < 10 or ms > 500:
                return f"Out of range: {ms}ms (expected 10-500)"
            new_dt = ms / 1000.0
            count = 0
            with self.server.clients_lock:
                for ctx in self.server.clients.values():
                    if hasattr(ctx, 'weapon_system'):
                        old_ms = ctx.weapon_system.avg_frame_dt * 1000
                        ctx.weapon_system.avg_frame_dt = new_dt
                        count += 1
            return f"Set frame_dt={ms:.1f}ms (was {old_ms:.1f}ms) on {count} client(s)"

        # Show current values
        lines = []
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if hasattr(ctx, 'weapon_system'):
                    ws = ctx.weapon_system
                    lines.append(f"  c{ctx.client_id}: frame_dt={ws.avg_frame_dt*1000:.1f}ms")
        if not lines:
            return "No clients with weapon_system"
        return "Client frame dt:\n" + "\n".join(lines)

    def _cmd_terrain(self, args: list) -> str:
        """Show terrain info at player positions or arbitrary coordinates.

        Usage:
          terrain         - Show terrain height at each player's position
          terrain x y     - Query terrain height at (x, y)
        """
        import math
        if not self.server:
            return "Error: No server reference"
        t = self.server.terrain
        if t is None:
            return "No terrain loaded"
        map_offset = self.server.terrain_height_offset
        physics_offset = getattr(self.server, "terrain_physics_height_offset", map_offset)
        lines = [
            f"Terrain: {t.num_x}x{t.num_z} grid, "
            f"map_offset={map_offset} physics_offset={physics_offset}"
        ]

        if args and len(args) >= 2:
            try:
                wx, wy = float(args[0]), float(args[1])
                wz = float(args[2]) if len(args) >= 3 else 0.0
                probe = build_player_terrain_probe(
                    self.server,
                    (wx, wy, wz),
                    0.0,
                )
                lines.append(
                    f"  ({wx:.1f}, {wy:.1f}): h={probe.get('raw_height', 0.0):.2f} "
                    f"map_ground_z={probe.get('raw_height', 0.0) + map_offset:.2f} "
                    f"physics_ground_z={probe.get('physics_ground_z', 0.0):.2f} "
                    f"cell={probe.get('cell')} normal={probe.get('normal')} "
                    f"slope={probe.get('slope')} pitch={probe.get('pitch_at_heading_deg', 0.0):.1f} deg (north)"
                )
            except Exception as e:
                lines.append(f"  Error: {e}")
        else:
            with self.server.clients_lock:
                for ctx in self.server.clients.values():
                    if not ctx or not ctx.running:
                        continue
                    x, y, z = ctx.player_pos
                    probe = build_player_terrain_probe(self.server, ctx.player_pos, ctx.player_heading)
                    lines.append(
                        f"  c{ctx.client_id} pos=({x:.1f},{y:.1f},{z:.2f}) "
                        f"terrain_h={probe.get('raw_height', 0.0):.2f} "
                        f"physics_ground_z={probe.get('physics_ground_z', 0.0):.2f} "
                        f"cell={probe.get('cell')} normal={probe.get('normal')} "
                        f"pitch={probe.get('pitch_at_heading_deg', 0.0):.1f}deg "
                        f"delta_z={probe.get('clearance_z', 0.0):.2f}"
                    )
        return "\n".join(lines)

    def _cmd_resend(self, args: list) -> str:
        """Force re-send entity creation packets to all clients.

        Clears known_entity_ids and directly sends entity creation packets.

        Usage:
          resend            - Clear and resend all entities via TCP
          resend udp        - Send via UDP instead
          resend c<id>      - Only resend to specific client
          resend c<id> udp  - Specific client via UDP
        """
        if not self.server:
            return "Error: No server reference"

        clients = self.server._snapshot_in_game_clients()
        if len(clients) < 2:
            return f"Need 2+ in-game clients, have {len(clients)}"

        # Parse args
        target_id = None
        use_udp = 'udp' in [a.lower() for a in args]
        for a in args:
            if a.startswith('c') and a[1:].isdigit():
                target_id = int(a[1:])

        from .packets import build_update_array_create_tank
        # Parse optional override entity ID
        override_eid = None
        no_health = 'nohealth' in [a.lower() for a in args]
        for a in args:
            if a.isdigit() and int(a) > 100:
                override_eid = int(a)
        lines = []

        for ctx in clients:
            if target_id is not None and ctx.client_id != target_id:
                continue

            for other in clients:
                if other is ctx:
                    continue
                eid = override_eid or (other.session.entity_id or other.entity_id)
                ctx.known_entity_ids.discard(eid)

                # Build packet
                pos = self.server._to_client_pos(other.player_pos)
                team = other.session.team_id or 1
                tick = self.server._get_network_tick(ctx)
                pkt = build_update_array_create_tank(
                    tick=tick, entity_id=eid,
                    entity_type=other.entity_type, team=team,
                    pos=pos, is_manned=True,
                    include_health=not no_health,
                )

                # Send
                if use_udp and self.server.udp_handler and ctx.session.udp_addr:
                    self.server.udp_handler.send_to(pkt, ctx.session.udp_addr)
                    ctx.known_entity_ids.add(eid)
                    lines.append(f"Sent entity {eid} (team={team}) -> client {ctx.client_id} via UDP "
                                 f"tick={tick} pos={pos} len={len(pkt)}")
                    lines.append(f"  HEX: {pkt.hex().upper()}")
                elif ctx.tcp_handler:
                    ctx.tcp_handler.send(pkt, log=True)
                    ctx.known_entity_ids.add(eid)
                    lines.append(f"Sent entity {eid} (team={team}) -> client {ctx.client_id} via TCP "
                                 f"tick={tick} pos={pos} len={len(pkt)}")
                    lines.append(f"  HEX: {pkt.hex().upper()}")
                else:
                    lines.append(f"FAILED: no transport for client {ctx.client_id}")

        # Also send a health-restore heartbeat if nohealth was used previously
        if no_health or 'fixhealth' in [a.lower() for a in args]:
            from .packets import build_update_array_heartbeat
            for ctx in clients:
                if target_id is not None and ctx.client_id != target_id:
                    continue
                eid = ctx.session.entity_id or ctx.entity_id
                hb_tick = self.server._get_network_tick(ctx)
                hb = build_update_array_heartbeat(
                    tick=hb_tick, entity_id=eid,
                    include_health=True, weapon_id=2,
                    health=1.0, fuel=1.0,
                )
                if self.server.udp_handler and ctx.session.udp_addr:
                    self.server.udp_handler.send_to(hb, ctx.session.udp_addr)
                    lines.append(f"Sent health fix heartbeat to client {ctx.client_id}")

        return "\n".join(lines)

    def _cmd_respawn(self, args: list) -> str:
        """
        Respawn by DELETE + server-triggered delayed spawn.

        1) DELETE entity (client goes to team select)
        2) Set delayed_spawn on session (game loop auto-fires _auto_join_team)
        3) _auto_join_team picks up pending_respawn_pos and spawns there

        No client interaction needed - the game loop handles the re-spawn
        the same way it handles the initial auto-join.

        Usage:
          respawn              - Respawn at map spawn point
          respawn <x> <y> <z>  - Respawn at specific position
          respawn c<id>        - Respawn specific client (e.g. c4)
          respawn c<id> <x> <y> <z> - Respawn specific client at position
          respawn c<id> t<team> <x> <y> <z> - Respawn on specific team (1=red, 2=blue)
          respawn all          - Respawn all clients at staggered positions
        """
        if not self.server:
            return "Error: No server reference"

        # Handle "respawn all" — respawn every in-game client at staggered positions
        if args and args[0].lower() == "all":
            results = []
            offset = 0.0
            with self.server.clients_lock:
                clients = [c for c in self.server.clients.values()
                           if c and c.running and c.session and c.session.in_game]
            for c in clients:
                result = self._do_respawn(c, offset_x=offset)
                results.append(f"Client {c.client_id}: {result}")
                offset += 80.0  # Stagger 80 units apart
            return "\n".join(results) if results else "No in-game clients"

        # Check for c<id> prefix to target specific client
        ctx = None
        if args and args[0].lower().startswith("c") and args[0][1:].isdigit():
            target_id = int(args[0][1:])
            ctx, addr = self._get_client_by_id(target_id)
            if not ctx:
                return f"Error: No client with id {target_id}"
            args = args[1:]  # consume the c<id> arg
        else:
            ctx, addr = self._get_active_client()
        if not ctx:
            return "Error: No connected client"

        # Check for t<team> prefix
        team = None
        if args and args[0].lower().startswith("t") and args[0][1:].isdigit():
            team = int(args[0][1:])
            args = args[1:]

        # Only pass explicit pos if user provided coordinates
        pos = None
        try:
            if len(args) >= 3:
                pos = (float(args[0]), float(args[1]), float(args[2]))
        except ValueError as e:
            return f"respawn arg parse error: {e}"

        result = self._do_respawn(ctx, pos=pos, team=team)
        self._sync_to_active_client()
        team_name = {1: "Red", 2: "Blue"}.get(team, str(team)) if team else "same"
        return f"Respawned client {ctx.client_id} (team={team_name}) at {result}"

    def _cmd_enter_game(self, args: list) -> str:
        """Force a connected client into game through the normal TankPacket spawn path.

        This is intended for automation: it bypasses fragile team/flag clicking
        but still uses the same server spawn path as manual map flag selection.
        Usage:
          enter_game [c<id>] [t<team>] [x y z]
          spawn_now [c<id>] [t<team>] [x y z]
        """
        if not self.server:
            return "Error: No server reference"

        ctx = None
        if args and args[0].lower().startswith("c") and args[0][1:].isdigit():
            target_id = int(args[0][1:])
            ctx, _ = self._get_client_by_id(target_id, require_udp=False)
            if not ctx:
                return f"Error: No client with id {target_id}"
            args = args[1:]
        else:
            ctx, _ = self._get_active_client(require_udp=False)
        if not ctx:
            return "Error: No connected client"
        if not ctx.tcp_handler or not ctx.session:
            return f"Error: Client {ctx.client_id} has no active TCP/session"

        team_id = ctx.session.team_id or 1
        if args and args[0].lower().startswith("t") and args[0][1:].isdigit():
            team_id = int(args[0][1:])
            args = args[1:]

        pos = None
        try:
            if len(args) >= 3:
                pos = (float(args[0]), float(args[1]), float(args[2]))
            elif args:
                return "enter_game usage: enter_game [c<id>] [t<team>] [x y z]"
        except ValueError as e:
            return f"enter_game arg parse error: {e}"

        entity_id = ctx.session.player_id or ctx.session.entity_id or ctx.entity_id
        if not entity_id:
            entity_id = int(getattr(self.server, "next_entity_id", 1337) or 1337)
            self.server.next_entity_id = max(int(getattr(self.server, "next_entity_id", 1337) or 1337), entity_id + 1)

        ctx.session.player_id = entity_id
        ctx.session.team_id = team_id
        ctx.entity_id = entity_id
        ctx.known_entity_ids.add(entity_id)
        if hasattr(ctx, "authoritative_state_history"):
            ctx.authoritative_state_history.clear()

        self.ctx = ctx
        self.session = ctx.session
        self.tcp_handler = ctx.tcp_handler

        self.server._spawn_wf_style(
            ctx,
            team_id=team_id,
            net_id=entity_id,
            unit_type=getattr(ctx, "entity_type", 0) or 0,
            pos=pos,
            announce=False,
        )
        self._sync_to_active_client()

        spawn_pos = ctx.player_pos
        udp = ctx.session.udp_addr if ctx.session else None
        return (
            f"Entered game: client={ctx.client_id} entity={entity_id} team={team_id} "
            f"pos=({spawn_pos[0]:.1f},{spawn_pos[1]:.1f},{spawn_pos[2]:.1f}) "
            f"udp={'yes' if udp else 'no'}"
        )

    def _select_control_client(self, args: list) -> tuple[Any, Any, list]:
        """Resolve an optional leading c<id> selector for control commands."""
        if args and args[0].lower().startswith("c") and args[0][1:].isdigit():
            target_id = int(args[0][1:])
            ctx, addr = self._get_client_by_id(target_id)
            if not ctx:
                raise ValueError(f"No client with id {target_id}")
            return ctx, addr, args[1:]
        ctx, addr = self._get_active_client()
        if not ctx:
            raise ValueError("No connected client")
        return ctx, addr, args

    def _cmd_join_team(self, args: list) -> str:
        """Run the same team-switch handler used by an OG REINCARNATE team click.

        Usage:
          join_team [c<id>] t<team>
          jt c1 t2
        """
        if not self.server:
            return "Error: No server reference"
        try:
            ctx, addr, args = self._select_control_client(args)
        except ValueError as e:
            return f"Error: {e}"
        if not ctx.session or not ctx.tcp_handler:
            return f"Error: Client {ctx.client_id} has no active TCP/session"
        if not args or not args[0].lower().startswith("t") or not args[0][1:].isdigit():
            return "join_team usage: join_team [c<id>] t<team>"

        team_id = int(args[0][1:])
        if team_id not in (1, 2):
            return f"Error: invalid team {team_id}"
        if not addr:
            addr = ctx.session.udp_addr
        if not addr:
            return f"Error: Client {ctx.client_id} has no UDP address yet"

        from . import handlers
        handlers.handle_team_switch(self.server, ctx, team_id, addr)
        self.ctx = ctx
        self.session = ctx.session
        self.tcp_handler = ctx.tcp_handler
        self._sync_to_active_client()
        return (
            f"Joined team: client={ctx.client_id} team={team_id} "
            f"phase={ctx.session.phase.name} udp={'yes' if ctx.session.udp_addr else 'no'}"
        )

    def _cmd_spawn_point(self, args: list) -> str:
        """Run the explicit map-flag spawn handler by spawn-point oid.

        Usage:
          spawn_point [c<id>] [oid] [vehicle]
          sp c1 5002
        """
        if not self.server:
            return "Error: No server reference"
        try:
            ctx, addr, args = self._select_control_client(args)
        except ValueError as e:
            return f"Error: {e}"
        if not ctx.session or not ctx.tcp_handler:
            return f"Error: Client {ctx.client_id} has no active TCP/session"
        if not addr:
            addr = ctx.session.udp_addr
        if not addr:
            return f"Error: Client {ctx.client_id} has no UDP address yet"

        spawn_points = self.server.get_spawn_points()
        team_id = ctx.session.team_id or 2
        spawn_point_id = 0
        vehicle_type = 0
        try:
            if args:
                spawn_point_id = int(args[0])
                args = args[1:]
            if args:
                vehicle_type = int(args[0])
        except ValueError as e:
            return f"spawn_point arg parse error: {e}"

        if spawn_point_id == 0:
            selected = next((sp for sp in spawn_points if sp.get("team") == team_id), None)
            if selected is None and spawn_points:
                selected = spawn_points[0]
            if selected is None:
                return "Error: no spawn points available"
            spawn_point_id = int(selected.get("oid", 0))

        from . import handlers
        handlers.handle_spawn_at_point(self.server, ctx, spawn_point_id, vehicle_type, addr)
        self.ctx = ctx
        self.session = ctx.session
        self.tcp_handler = ctx.tcp_handler
        self._sync_to_active_client()
        pos = ctx.player_pos
        return (
            f"Spawn point handled: client={ctx.client_id} point={spawn_point_id} "
            f"vehicle={vehicle_type} phase={ctx.session.phase.name} "
            f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})"
        )

    def _cmd_spawn_full(self, args: list) -> str:
        """
        Force a full spawn sequence for the current client.
        Args (positional):
          team_id entity_id vehicle_type behavior_type x y z delay_ms ack_timeout_ms [send_world_stats] [want_timeout_ms]
        """
        if not self.tcp_handler:
            return "Error: No game client connected"

        def _parse_bool(value: str) -> bool:
            return value.lower() in ('1', 'true', 'yes', 'on')

        # Defaults
        team_id = 2
        entity_id = 1337
        vehicle_type = 0
        behavior_type = 0
        x, y, z = 100.0, 100.0, 100.0
        delay_ms = 150
        ack_timeout_ms = 2000
        send_world_stats = False
        want_timeout_ms = 2000
        translation_override = None
        include_interp = False
        interp_bits = 16
        suppress_want_updates = False

        # Support key=value overrides (optional)
        opts = {}
        positional = []
        for arg in args:
            if '=' in arg:
                key, val = arg.split('=', 1)
                opts[key.strip().lower()] = val.strip()
            else:
                positional.append(arg)
        args = positional

        try:
            if len(args) > 0:
                team_id = int(args[0])
            if len(args) > 1:
                entity_id = int(args[1])
            if len(args) > 2:
                vehicle_type = int(args[2])
            if len(args) > 3:
                behavior_type = int(args[3])
            if len(args) > 4:
                x = float(args[4])
            if len(args) > 5:
                y = float(args[5])
            if len(args) > 6:
                z = float(args[6])
            if len(args) > 7:
                delay_ms = int(args[7])
            if len(args) > 8:
                ack_timeout_ms = int(args[8])
            if len(args) > 9:
                send_world_stats = int(args[9]) != 0
            if len(args) > 10:
                want_timeout_ms = int(args[10])
        except ValueError as e:
            return f"spawn_full arg parse error: {e}"

        try:
            if 'ack' in opts:
                ack_timeout_ms = int(opts['ack'])
            if 'ack_timeout_ms' in opts:
                ack_timeout_ms = int(opts['ack_timeout_ms'])
            if 'xlate' in opts:
                translation_override = _parse_bool(opts['xlate'])
            if 'translation' in opts:
                translation_override = _parse_bool(opts['translation'])
            if 'send_translation' in opts:
                translation_override = _parse_bool(opts['send_translation'])
            if 'ws' in opts:
                send_world_stats = _parse_bool(opts['ws'])
            if 'world_stats' in opts:
                send_world_stats = _parse_bool(opts['world_stats'])
            if 'want' in opts:
                want_timeout_ms = int(opts['want'])
            if 'want_timeout_ms' in opts:
                want_timeout_ms = int(opts['want_timeout_ms'])
            if 'interp' in opts:
                include_interp = _parse_bool(opts['interp'])
            if 'pos' in opts:
                include_interp = _parse_bool(opts['pos'])
            if 'position' in opts:
                include_interp = _parse_bool(opts['position'])
            if 'interp_bits' in opts:
                interp_bits = int(opts['interp_bits'])
            if 'ibits' in opts:
                interp_bits = int(opts['ibits'])
            if 'suppress' in opts:
                suppress_want_updates = _parse_bool(opts['suppress'])
            if 'strict' in opts:
                suppress_want_updates = _parse_bool(opts['strict'])
        except ValueError as e:
            return f"spawn_full option parse error: {e}"

        name = self.session.username if self.session and self.session.username else "Player"
        active_ctx = self.ctx
        if active_ctx is None:
            active_ctx, _ = self._get_active_client()
            if active_ctx is not None:
                self.ctx = active_ctx

        send_translation = True
        if send_world_stats and translation_override is None:
            # Default to NOT sending translation when WORLD_STATS triggers WANT_UPDATES,
            # unless we're suppressing the WANT_UPDATES payload.
            send_translation = suppress_want_updates
        elif translation_override is not None:
            send_translation = translation_override

        # Reset translation ACK tracking (used by optional wait below).
        if self.session:
            self.session.translation_ack_received = False
            self.session.translation_ack_time = 0.0

        prior_suppress = False
        if self.session:
            prior_suppress = self.session.suppress_want_updates_payload

        if send_world_stats and self.session:
            self.session.suppress_want_updates_payload = suppress_want_updates

        if send_world_stats:
            # 0) Team confirmed (server -> client)
            self.tcp_handler.send(build_reincarnate(0x02, "Team confirmed"))

            # 1) WORLD_STATS + wait for WANT_UPDATES (fresh)
            if self.session:
                self.session.want_updates_received = False
                self.session.want_updates_time = 0.0
            self.tcp_handler.send(build_world_stats())

            if self.session and want_timeout_ms > 0:
                deadline = time.monotonic() + (want_timeout_ms / 1000.0)
                while not self.session.want_updates_received and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not self.session.want_updates_received:
                    if self.session:
                        self.session.suppress_want_updates_payload = prior_suppress
                    return f"Spawn aborted: no WANT_UPDATES within {want_timeout_ms}ms after WORLD_STATS"

            # If WANT_UPDATES handler is suppressed, wait until it finishes before proceeding.
            if self.session and suppress_want_updates:
                deadline = time.monotonic() + (want_timeout_ms / 1000.0)
                while not self.session.want_updates_handled and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not self.session.want_updates_handled:
                    self.session.suppress_want_updates_payload = prior_suppress
                    return f"Spawn aborted: WANT_UPDATES handler not finished within {want_timeout_ms}ms"

            # 2) Optional TRANSLATION + wait for ACK (avoid quantizer issues)
            if send_translation:
                self.tcp_handler.send(build_translation_packet())

                if self.session and ack_timeout_ms > 0:
                    deadline = time.monotonic() + (ack_timeout_ms / 1000.0)
                    while not self.session.translation_ack_received and time.monotonic() < deadline:
                        time.sleep(0.01)
                    if not self.session.translation_ack_received:
                        if self.session:
                            self.session.suppress_want_updates_payload = prior_suppress
                        return f"Spawn aborted: no TRANSLATION_ACK within {ack_timeout_ms}ms"

        else:
            # 0) TRANSLATION + wait for ACK (avoid quantizer issues)
            if send_translation:
                self.tcp_handler.send(build_translation_packet())

                if self.session and ack_timeout_ms > 0:
                    deadline = time.monotonic() + (ack_timeout_ms / 1000.0)
                    while not self.session.translation_ack_received and time.monotonic() < deadline:
                        time.sleep(0.01)
                    if not self.session.translation_ack_received:
                        if self.session:
                            self.session.suppress_want_updates_payload = prior_suppress
                        return f"Spawn aborted: no TRANSLATION_ACK within {ack_timeout_ms}ms"

            # 1) Team confirmed (server -> client)
            self.tcp_handler.send(build_reincarnate(0x02, "Team confirmed"))

        # 2) Switch to active player
        self.tcp_handler.send(build_player(entity_id=entity_id, spectator=False))

        # 3) Roster + stats
        self.tcp_handler.send(build_add_to_roster(
            player_id=entity_id, entity_id=entity_id, name=name, team=team_id
        ))
        self.tcp_handler.send(build_update_stats(player_id=entity_id, entity_id=entity_id, team_id=team_id))

        # 4) Create entity via UPDATE_ARRAY (presence bit 0)
        self.tcp_handler.send(build_update_array_create_tank(
            tick=0, entity_id=entity_id, entity_type=vehicle_type,
            team=team_id, pos=(x, y, z), behavior_type=behavior_type,
            include_interp=include_interp, interp_bits=interp_bits
        ))

        # Small delay before PLAYER_INFO to ensure entity exists
        time.sleep(max(0, delay_ms) / 1000.0)

        # 5) PLAYER_INFO (local player vehicle)
        self.tcp_handler.send(build_player_info(
            entity_oid=entity_id, vehicle_type=vehicle_type, pos=(x, y, z)
        ))

        # 6) GAME_CLOCK + spawn success + birth notice
        self.tcp_handler.send(build_game_clock())
        self.tcp_handler.send(build_reincarnate(0x11, "Spawn success"))
        self.tcp_handler.send(build_birth_notice(entity_id))

        if active_ctx:
            active_ctx.entity_id = entity_id
            active_ctx.entity_type = vehicle_type
            active_ctx.player_pos = (x, y, z)
            active_ctx.player_vel = (0.0, 0.0, 0.0)
            active_ctx.player_health = 1.0
            active_ctx.player_energy = 1.0
            active_ctx.known_entity_ids.add(entity_id)
            active_ctx.world_collision_ref_pos = active_ctx.player_pos
            active_ctx.world_collision_bounds_dirty = True
            if active_ctx.weapon_system:
                active_ctx.weapon_system.player_id = entity_id
                active_ctx.weapon_system.team_id = team_id
                active_ctx.weapon_system.player_pos = (x, y, z)

        if self.session:
            self.session.suppress_want_updates_payload = prior_suppress
            self.session.player_id = entity_id
            self.session.enter_game(entity_id, team_id)
            self.session.last_spawn_time = time.monotonic()

        if (
            self.server
            and active_ctx
            and FEATURES.tick_loop_enabled
            and self.server._ensure_tick_loop(active_ctx)
        ):
            active_ctx.last_action_dump_time = time.monotonic()

        return ("Spawn sequence sent: team=%d entity=%d vehicle=%d behavior=%d pos=(%.1f,%.1f,%.1f) delay_ms=%d"
                % (team_id, entity_id, vehicle_type, behavior_type, x, y, z, delay_ms))

    def _cmd_spawn_points(self, args: list) -> str:
        """
        Send spawn point entities via UPDATE_ARRAY.
        Usage: spawn_points [count] [team]
        Default: 2 spawn points for team 2 at (50,0,50) and (150,0,150)
        """
        if not self.tcp_handler:
            return "Error: No game client connected"

        count = 2
        team = 2
        try:
            if len(args) > 0:
                count = int(args[0])
            if len(args) > 1:
                team = int(args[1])
        except ValueError as e:
            return f"spawn_points arg parse error: {e}"

        # Generate spawn points spread across the map
        spawn_points = []
        for i in range(count):
            spawn_points.append({
                'oid': 1000 + i,  # Unique IDs for spawn points
                'team': team,
                'variant': 1,
                'x': 50.0 + i * 100,
                'y': 0.0,
                'z': 50.0 + i * 100,
                'rot': (0.0, 0.0, 0.0),
            })

        data = build_update_array_spawn_points(0, spawn_points)
        print(f"[SPAWN_POINTS] Sending {count} spawn points for team {team}")
        print(f"[SPAWN_POINTS-HEX] len={len(data)} hex={data.hex().upper()}")
        self.tcp_handler.send(data)

        return f"Sent {count} spawn points for team {team}"

    def _cmd_projectile_move(self, args: list) -> str:
        """
        Spawn a projectile and send position updates as it moves.
        This tests the full projectile lifecycle: spawn -> move -> (destroy)

        Args: [duration_sec] [update_rate_hz]
        Default: 3 seconds, 10 Hz updates
        """
        if not self.udp_handler or not self.session:
            return "Error: No UDP handler/session available"

        if not self.session.udp_addr:
            return "Error: No UDP client address"

        duration = 3.0
        update_rate = 10.0

        try:
            if len(args) > 0:
                duration = float(args[0])
            if len(args) > 1:
                update_rate = float(args[1])
        except ValueError as e:
            return f"projectile_move arg parse error: {e}"

        from .weapons import (
            Projectile, EntityType,
            build_projectile_spawn_packet, build_projectile_update_packet
        )
        from .packets import get_ticks
        import threading

        # Generate unique projectile ID
        if not hasattr(self, '_projectile_id'):
            self._projectile_id = 3000
        self._projectile_id += 1

        # Create projectile - spawn HIGH in sky so it's visible
        proj = Projectile(
            entity_id=self._projectile_id,
            entity_type=EntityType.PULSE_SHELL,
            owner_id=self.session.player_id or 1337,
            team=self.session.team_id or 2,  # Default to team 2 (player's team)
            pos=(100.0, 180.0, 100.0),  # Start HIGH in sky (y=180)
            vel=(20.0, -10.0, 20.0),    # Moving diagonally and falling
            spawn_time=time.monotonic(),
            lifetime=duration
        )

        addr = self.session.udp_addr
        udp = self.udp_handler

        def update_loop():
            """Background thread to send position updates."""
            # Send spawn packet
            tick = get_ticks()
            spawn_pkt = build_projectile_spawn_packet(proj, tick)
            udp.send_to(spawn_pkt, addr)
            print(f"[PMOVE] Spawned projectile {proj.entity_id}")

            # Send updates
            dt = 1.0 / update_rate
            update_count = int(duration * update_rate)

            for i in range(update_count):
                time.sleep(dt)
                tick = get_ticks()
                update_pkt = build_projectile_update_packet(proj, tick, dt)
                udp.send_to(update_pkt, addr)

                if i % 5 == 0:
                    print(f"[PMOVE] Update {i+1}/{update_count}: pos={proj.pos}")

            print(f"[PMOVE] Projectile {proj.entity_id} finished")

        # Start update thread
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

        return (f"Spawned moving projectile id={proj.entity_id} "
                f"duration={duration}s rate={update_rate}Hz")

    def _cmd_shell(self, args: list) -> str:
        """
        Spawn a pulse shell with full control over position and direction.
        Includes position updates at 15 Hz for reliable movement.

        Usage: shell [x y z] [yaw] [pitch] [speed] [duration]

        Examples:
          shell                           - Spawn at (100,15,100) forward
          shell 150 20 150                - Spawn at specific position
          shell 150 20 150 45             - Position + yaw (degrees)
          shell 150 20 150 45 -10         - Position + yaw + pitch (degrees)
          shell 150 20 150 45 -10 100     - Position + yaw + pitch + speed
          shell 150 20 150 45 -10 100 3   - ... + duration in seconds

        Defaults: pos=(100,15,100) yaw=0 pitch=0 speed=75 duration=3
        """
        if not self.udp_handler or not self.session:
            return "Error: No UDP handler/session available"

        if not self.session.udp_addr:
            return "Error: No UDP client address"

        import math
        import threading

        # Defaults
        x, y, z = 100.0, 15.0, 100.0
        yaw_deg = 0.0
        pitch_deg = 0.0
        speed = 75.0
        duration = 3.0

        try:
            if len(args) >= 3:
                x, y, z = float(args[0]), float(args[1]), float(args[2])
            if len(args) >= 4:
                yaw_deg = float(args[3])
            if len(args) >= 5:
                pitch_deg = float(args[4])
            if len(args) >= 6:
                speed = float(args[5])
            if len(args) >= 7:
                duration = float(args[6])
        except ValueError as e:
            return f"shell arg error: {e}"

        # Convert degrees to radians
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)

        # Calculate velocity from yaw/pitch
        # Forward vector (y-up coordinate system)
        vx = speed * math.cos(pitch) * math.cos(yaw)
        vy = speed * math.sin(pitch)
        vz = speed * math.cos(pitch) * math.sin(yaw)

        from .weapons import Projectile, EntityType, build_projectile_spawn_packet, build_projectile_update_packet
        from .packets import get_ticks

        # Generate unique projectile ID
        if not hasattr(self, '_projectile_id'):
            self._projectile_id = 5000
        self._projectile_id += 1

        proj = Projectile(
            entity_id=self._projectile_id,
            entity_type=EntityType.PULSE_SHELL,
            owner_id=self.session.player_id or 1337,
            team=self.session.team_id or 2,
            pos=(x, y, z),
            vel=(vx, vy, vz),
            spawn_time=time.monotonic(),
            lifetime=duration
        )

        addr = self.session.udp_addr
        udp = self.udp_handler
        tcp = self.tcp_handler
        proj_id = self._projectile_id

        def update_loop():
            """Background thread to send position updates at 15 Hz."""
            # Send spawn packet
            tick = get_ticks()
            spawn_pkt = build_projectile_spawn_packet(proj, tick)
            udp.send_to(spawn_pkt, addr)
            if tcp:
                tcp.send(spawn_pkt)
            print(f"[SHELL] Spawned id={proj_id} pos=({x:.1f},{y:.1f},{z:.1f}) vel=({vx:.1f},{vy:.1f},{vz:.1f})")

            # Send updates at 15 Hz
            dt = 1.0 / 15.0
            update_count = int(duration * 15)

            for i in range(update_count):
                time.sleep(dt)
                tick = get_ticks()
                update_pkt = build_projectile_update_packet(proj, tick, dt)
                udp.send_to(update_pkt, addr)

                if i % 15 == 0:  # Log every second
                    print(f"[SHELL] Update {i+1}/{update_count}: pos=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f})")

            print(f"[SHELL] Projectile {proj_id} finished")

        # Start update thread
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

        return f"Shell spawned: pos=({x:.0f},{y:.0f},{z:.0f}) yaw={yaw_deg:.0f}° pitch={pitch_deg:.0f}° vel=({vx:.1f},{vy:.1f},{vz:.1f})"

    def _cmd_spread(self, args: list) -> str:
        """
        Fire a spread of shells in different directions for visual demo.
        Usage: spread [count] [x y z] [speed]

        Examples:
          spread                  - 8 shells in 360° spread from (100,15,100)
          spread 12               - 12 shells
          spread 8 150 20 150     - 8 shells from specific position
          spread 8 150 20 150 100 - ... with speed 100

        Defaults: count=8 pos=(100,15,100) speed=75
        """
        if not self.udp_handler or not self.session:
            return "Error: No UDP handler/session available"

        if not self.session.udp_addr:
            return "Error: No UDP client address"

        import math
        import threading

        # Defaults
        count = 8
        x, y, z = 100.0, 15.0, 100.0
        speed = 75.0
        duration = 3.0

        try:
            if len(args) >= 1:
                count = int(args[0])
            if len(args) >= 4:
                x, y, z = float(args[1]), float(args[2]), float(args[3])
            if len(args) >= 5:
                speed = float(args[4])
        except ValueError as e:
            return f"spread arg error: {e}"

        from .weapons import Projectile, EntityType, build_projectile_spawn_packet, build_projectile_update_packet
        from .packets import get_ticks

        addr = self.session.udp_addr
        udp = self.udp_handler
        tcp = self.tcp_handler

        # Spawn shells in a circle pattern
        results = []
        for i in range(count):
            # Spread yaw evenly around 360 degrees
            yaw_deg = (360.0 / count) * i
            yaw = math.radians(yaw_deg)

            # Calculate velocity
            vx = speed * math.cos(yaw)
            vy = 0.0
            vz = speed * math.sin(yaw)

            # Generate unique projectile ID
            if not hasattr(self, '_projectile_id'):
                self._projectile_id = 5000
            self._projectile_id += 1

            proj = Projectile(
                entity_id=self._projectile_id,
                entity_type=EntityType.PULSE_SHELL,
                owner_id=self.session.player_id or 1337,
                team=self.session.team_id or 2,
                pos=(x, y, z),
                vel=(vx, vy, vz),
                spawn_time=time.monotonic(),
                lifetime=duration
            )

            proj_id = self._projectile_id
            proj_ref = proj  # Capture for closure

            def make_update_loop(p, pid, yaw_d):
                def update_loop():
                    """Background thread to send position updates at 15 Hz."""
                    tick = get_ticks()
                    spawn_pkt = build_projectile_spawn_packet(p, tick)
                    udp.send_to(spawn_pkt, addr)
                    if tcp:
                        tcp.send(spawn_pkt)

                    dt = 1.0 / 15.0
                    update_count = int(duration * 15)

                    for j in range(update_count):
                        time.sleep(dt)
                        tick = get_ticks()
                        update_pkt = build_projectile_update_packet(p, tick, dt)
                        udp.send_to(update_pkt, addr)
                return update_loop

            # Start update thread (with small delay between spawns)
            thread = threading.Thread(target=make_update_loop(proj_ref, proj_id, yaw_deg), daemon=True)
            thread.start()
            time.sleep(0.05)  # 50ms between spawns to avoid packet collision

            results.append(f"{yaw_deg:.0f}°")
            print(f"[SPREAD] Shell {i+1}/{count}: yaw={yaw_deg:.0f}° vel=({vx:.1f},{vy:.1f},{vz:.1f})")

        return f"Fired {count} shells in spread pattern from ({x:.0f},{y:.0f},{z:.0f}): {', '.join(results)}"

    def _cmd_reset_pos(self, args: list) -> str:
        """Reset player position to spawn location."""
        if not self.server:
            return "Error: No server reference"

        # Reset to spawn position
        self.server.player_pos = (100.0, 15.0, 100.0)
        self.server.player_pose["pos"] = self.server.player_pos
        self.server.player_pose["source"] = "reset"
        self.server.player_yaw = 0.0
        self.server.player_vel = (0.0, 0.0, 0.0)
        self.server.player_speed = 0.0
        return "Reset player pos to (100, 15, 100) yaw=0"

    def _cmd_player_pos(self, args: list) -> str:
        """
        Show or set player position.
        Usage:
          pos                 - Show active player position/heading/velocity
          pos c<id>           - Show a specific client's state
          pos x y z           - Set active player position
          pos c<id> x y z     - Set a specific client's position
          pos c<id> x y z h   - Set a specific client's position + heading degrees
          pos c<id> x y z vel vx vy vz
                              - Set position + velocity from live tap telemetry
        """
        if not self.server:
            return "Error: No server reference"
        import math

        target_id = None
        rem = list(args)
        if rem and rem[0].lower().startswith("c") and rem[0][1:].isdigit():
            target_id = int(rem[0][1:])
            rem = rem[1:]

        if target_id is not None:
            ctx, _ = self._get_client_by_id(target_id)
            if not ctx:
                return f"Error: No client with id {target_id}"
        else:
            ctx, _ = self._get_active_client()
            if not ctx:
                return "Error: No active client"

        if rem:
            try:
                x = float(rem[0])
                y = float(rem[1]) if len(rem) > 1 else ctx.player_pos[1]
                z = float(rem[2]) if len(rem) > 2 else ctx.player_pos[2]
                heading_deg = None
                vel = None
                idx = 3
                while idx < len(rem):
                    token = rem[idx].lower()
                    if token in ("vel", "v"):
                        if idx + 3 >= len(rem):
                            return "pos arg error: vel requires vx vy vz"
                        vel = (
                            float(rem[idx + 1]),
                            float(rem[idx + 2]),
                            float(rem[idx + 3]),
                        )
                        idx += 4
                    elif token in ("heading", "h"):
                        if idx + 1 >= len(rem):
                            return "pos arg error: heading requires degrees"
                        heading_deg = float(rem[idx + 1])
                        idx += 2
                    elif heading_deg is None:
                        heading_deg = float(rem[idx])
                        idx += 1
                    else:
                        return f"pos arg error: unexpected token {rem[idx]!r}"
                heading_rad = math.radians(heading_deg) if heading_deg is not None else None
                self._apply_exact_client_pose(ctx, (x, y, z), heading_rad=heading_rad, vel=vel)
                vel_suffix = (
                    f" vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})"
                    if vel is not None
                    else ""
                )
                if heading_deg is None:
                    return (
                        f"Set client {ctx.client_id} pos to "
                        f"({x:.1f}, {y:.1f}, {z:.1f}){vel_suffix}"
                    )
                return (
                    f"Set client {ctx.client_id} pos to "
                    f"({x:.1f}, {y:.1f}, {z:.1f}) heading={heading_deg:.2f}deg{vel_suffix}"
                )
            except (ValueError, IndexError) as e:
                return f"pos arg error: {e}"
        else:
            try:
                import math
                px, py, pz = ctx.player_pos
                vx, vy, vz = ctx.player_vel
                yaw_deg = math.degrees(-ctx.player_heading)
                speed = math.sqrt(vx*vx + vy*vy + vz*vz)
                ang_vel = math.degrees(ctx.vehicle_physics.angular_velocity) if ctx.vehicle_physics else 0.0
                lines = [
                    f"pos=({px:.1f}, {py:.1f}, {pz:.1f})",
                    f"vel=({vx:.1f}, {vy:.1f}, {vz:.1f}) speed={speed:.1f}",
                    f"heading={yaw_deg:.2f}deg ang_vel={ang_vel:.2f}deg/s",
                ]
                if ctx.weapon_system:
                    from .weapons import BehaviorSlot
                    ws = ctx.weapon_system
                    turn = ws.behavior_slots[BehaviorSlot.TURNING]
                    fwd = ws.behavior_slots[BehaviorSlot.MOVING_FORWARD]
                    strafe = ws.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS]
                    lines.insert(0, f"client_id={ctx.client_id}")
                    lines.append(f"input: turn={turn:.3f} fwd={fwd:.3f} strafe={strafe:.3f}")
                    lines.append(f"frame_dt={ws.avg_frame_dt*1000:.1f}ms")
                else:
                    lines.insert(0, f"client_id={ctx.client_id}")
                steps = getattr(ctx, "physics_step_count", "?")
                lines.append(f"physics_steps={steps}")
                return "\n".join(lines)
            except Exception as e:
                import traceback
                return f"Error reading pos: {e}\n{traceback.format_exc()}"

    def _cmd_test_velocity(self, args: list) -> str:
        """
        Test different projectile velocities to see what client accepts.
        Usage: test_vel [speed] [direction]

        speed: units per second (default 50, try 25, 50, 75, 100, 150)
        direction: 'x', 'z', 'xz', 'up', 'down' (default 'x')

        Examples:
          test_vel 50 x      - 50 units/sec in +X direction
          test_vel 100 xz    - 100 units/sec diagonally
          test_vel 25 up     - 25 units/sec upward
        """
        if not self.udp_handler or not self.session:
            return "Error: No UDP handler/session available"

        if not self.session.udp_addr:
            return "Error: No UDP client address"

        speed = 50.0
        direction = 'x'

        try:
            if len(args) > 0:
                speed = float(args[0])
            if len(args) > 1:
                direction = args[1].lower()
        except ValueError as e:
            return f"test_vel arg parse error: {e}"

        # Calculate velocity vector based on direction
        vx, vy, vz = 0.0, 0.0, 0.0
        if direction == 'x' or direction == '+x':
            vx = speed
        elif direction == '-x':
            vx = -speed
        elif direction == 'z' or direction == '+z':
            vz = speed
        elif direction == '-z':
            vz = -speed
        elif direction == 'xz' or direction == 'diag':
            vx = speed * 0.707  # 45 degree angle
            vz = speed * 0.707
        elif direction == 'up' or direction == '+y':
            vy = speed
        elif direction == 'down' or direction == '-y':
            vy = -speed
        elif direction == 'arc':
            vx = speed * 0.866  # 30 degree upward arc
            vy = speed * 0.5
        else:
            return f"Unknown direction: {direction}. Try: x, z, xz, up, down, arc"

        from .weapons import (
            Projectile, EntityType,
            build_projectile_spawn_packet, build_projectile_update_packet
        )
        from .packets import get_ticks
        import threading

        # Generate unique projectile ID
        if not hasattr(self, '_projectile_id'):
            self._projectile_id = 4000
        self._projectile_id += 1

        # Get player position from server's player_pose (from VIEWPOINT_INFO)
        if self.server and hasattr(self.server, 'player_pose'):
            px, py, pz = self.server.player_pose["pos"]
            spawn_x = px + 2.0  # Slightly in front
            spawn_y = py        # Same height as player
            spawn_z = pz
            source = self.server.player_pose.get("source", "unknown")
            print(f"[TEST_VEL] Using player pos: ({px:.1f},{py:.1f},{pz:.1f}) source={source}")
        else:
            spawn_x = 102.0  # Fallback
            spawn_y = 5.0
            spawn_z = 100.0
            print(f"[TEST_VEL] Using default pos (no server ref)")

        proj = Projectile(
            entity_id=self._projectile_id,
            entity_type=EntityType.PULSE_SHELL,
            owner_id=self.session.player_id or 1337,
            team=self.session.team_id or 2,
            pos=(spawn_x, spawn_y, spawn_z),
            vel=(vx, vy, vz),
            spawn_time=time.monotonic(),
            lifetime=5.0
        )

        addr = self.session.udp_addr
        udp = self.udp_handler

        def update_loop():
            """Background thread - spawn only, let client simulate."""
            # Send spawn packet with velocity - client will simulate movement
            tick = get_ticks()
            spawn_pkt = build_projectile_spawn_packet(proj, tick)
            udp.send_to(spawn_pkt, addr)

            # Also send via TCP for reliability
            if self.tcp_handler:
                self.tcp_handler.send(spawn_pkt)

            print(f"[TEST_VEL] Spawned id={proj.entity_id} speed={speed} dir={direction}")
            print(f"[TEST_VEL] pos=({spawn_x:.1f},{spawn_y:.1f},{spawn_z:.1f}) vel=({vx:.1f},{vy:.1f},{vz:.1f})")
            print(f"[TEST_VEL] No updates - client simulates from velocity")

        # Start update thread
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

        return (f"Testing velocity: speed={speed} dir={direction} "
                f"vel=({vx:.1f},{vy:.1f},{vz:.1f})")

    def _cmd_fire_pulse(self, args: list) -> str:
        """
        Simulate firing the pulse cannon via the weapon system.
        This tests the full flow: weapon system -> projectile creation -> packet send
        """
        if not self.udp_handler or not self.session:
            return "Error: No UDP handler/session available"

        if not self.session.udp_addr:
            return "Error: No UDP client address (try reconnecting)"

        # Get the server's weapon system
        # We need access to the main server object
        # For now, simulate by directly creating and sending the projectile
        from .weapons import Projectile, EntityType, WeaponSystem, build_projectile_spawn_packet
        from .packets import get_ticks

        # Create a temporary weapon system to simulate fire
        ws = WeaponSystem()
        ws.player_id = self.session.player_id or 1337
        ws.player_team = self.session.team_id or 2  # Default to team 2 (player's team)
        ws.player_pos = (100.0, 100.0, 100.0)  # Match player spawn position
        ws.player_rot = (0.0, 0.0, 0.0)  # Facing forward (will spawn projectile at x+2)
        ws.current_weapon = 4  # Pulse cannon

        # Fire the pulse cannon
        proj = ws._fire_pulse_cannon()
        if not proj:
            return "Error: Failed to create projectile"

        # Send the spawn packet
        tick = get_ticks()
        packet = build_projectile_spawn_packet(proj, tick)

        print(f"[FIRE_PULSE] Spawning pulse shell id={proj.entity_id}")
        print(f"[FIRE_PULSE] pos={proj.pos} vel={proj.vel}")
        print(f"[FIRE_PULSE] Packet ({len(packet)} bytes): {packet.hex()}")

        # Send via both UDP and TCP for reliability
        self.udp_handler.send_to(packet, self.session.udp_addr)
        if self.tcp_handler:
            self.tcp_handler.send(packet)
            print(f"[FIRE_PULSE] Also sent via TCP")

        # Send chat message with position for in-game debugging
        from .packets import build_chat_message
        pos_msg = f"FIRE! pos=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f}) vel=({proj.vel[0]:.1f},{proj.vel[1]:.1f},{proj.vel[2]:.1f})"
        chat_packet = build_chat_message(pos_msg, source_id=self.session.player_id or 1337)
        self.udp_handler.send_to(chat_packet, self.session.udp_addr)

        return (f"Fired pulse cannon! Projectile id={proj.entity_id} "
                f"pos=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f})")

    def _cmd_projectile(self, args: list) -> str:
        """
        Spawn a test projectile entity (PULSE_SHELL).
        Args (positional):
          [x y z] [vx vy vz]
        Default: spawns at (105, 50, 100) moving forward at (50, 0, 0)
        """
        if not self.udp_handler or not self.session:
            return "Error: No UDP handler/session available"

        if not self.session.udp_addr:
            return "Error: No UDP client address (try reconnecting)"

        # Spawn at player height (y=100) and slightly in front (x+5)
        x, y, z = 105.0, 100.0, 100.0
        vx, vy, vz = 50.0, 0.0, 0.0

        try:
            if len(args) > 0:
                x = float(args[0])
            if len(args) > 1:
                y = float(args[1])
            if len(args) > 2:
                z = float(args[2])
            if len(args) > 3:
                vx = float(args[3])
            if len(args) > 4:
                vy = float(args[4])
            if len(args) > 5:
                vz = float(args[5])
        except ValueError as e:
            return f"projectile arg parse error: {e}"

        from .weapons import Projectile, EntityType, build_projectile_spawn_packet
        from .packets import get_ticks

        # Generate unique projectile ID
        if not hasattr(self, '_projectile_id'):
            self._projectile_id = 2000
        self._projectile_id += 1

        proj = Projectile(
            entity_id=self._projectile_id,
            entity_type=EntityType.PULSE_SHELL,
            owner_id=self.session.player_id or 1337,
            team=self.session.team_id or 2,  # Default to team 2 (player's team)
            pos=(x, y, z),
            vel=(vx, vy, vz),
            spawn_time=0,
            lifetime=5.0
        )

        tick = get_ticks()
        packet = build_projectile_spawn_packet(proj, tick)

        print(f"[PROJECTILE] Spawning pulse shell id={proj.entity_id} pos={proj.pos} vel={proj.vel}")
        print(f"[PROJECTILE] Packet ({len(packet)} bytes): {packet.hex()}")

        # Send via both UDP and TCP for reliability
        self.udp_handler.send_to(packet, self.session.udp_addr)
        if self.tcp_handler:
            self.tcp_handler.send(packet)
            print(f"[PROJECTILE] Also sent via TCP")

        return (f"Sent projectile spawn to {self.session.udp_addr}: "
                f"id={proj.entity_id} pos=({x:.1f},{y:.1f},{z:.1f}) vel=({vx:.1f},{vy:.1f},{vz:.1f})")

    def _cmd_send_health(self, args: list) -> str:
        """
        Show or set player health.
        Usage:
          health                 - Show all players' health
          health [c<id>]         - Show specific player's health
          health set <val> [c<id>] - Set health to val (0.0-1.0) and send update
        """
        if not self.server:
            return "Error: No server reference"

        # Parse args
        if args and args[0].lower() == "set":
            # health set <val> [c<id>]
            if len(args) < 2:
                return "Usage: health set <0.0-1.0> [c<id>]"
            try:
                new_health = float(args[1])
            except ValueError:
                return f"Invalid health value: {args[1]}"
            new_health = max(0.0, min(1.0, new_health))

            ctx = None
            if len(args) >= 3 and args[2].lower().startswith("c") and args[2][1:].isdigit():
                ctx, _ = self._get_client_by_id(int(args[2][1:]))
                if not ctx:
                    return f"Error: No client with id {args[2][1:]}"
            else:
                ctx, _ = self._get_active_client()
            if not ctx:
                return "Error: No connected client"

            old_health = ctx.player_health
            ctx.player_health = new_health

            # Send health update packet
            if self.server.udp_handler and ctx.session and ctx.session.udp_addr:
                tick = self.server._get_network_tick(ctx)
                packet = self.server._build_local_state_heartbeat(
                    ctx,
                    tick=tick,
                    entity_id=ctx.session.entity_id or ctx.entity_id,
                    include_health=True,
                    health=self.server._get_health_value(ctx),
                    fuel=self.server._get_energy_value(ctx),
                )
                self.server.udp_handler.send_to(packet, ctx.session.udp_addr)

            return (
                f"Client {ctx.client_id}: health {old_health*100:.0f}% → {new_health*100:.0f}%"
            )

        # Show health for specific or all clients
        client_filter = None
        for a in args:
            if a.lower().startswith("c") and a[1:].isdigit():
                client_filter = int(a[1:])

        lines = []
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if not ctx or not ctx.running:
                    continue
                if client_filter is not None and ctx.client_id != client_filter:
                    continue
                phase = ctx.session.phase.name if ctx.session else "NONE"
                name = ctx.session.username if ctx.session else f"Player{ctx.client_id}"
                lines.append(
                    f"Client {ctx.client_id} ({name}) [{phase}]: "
                    f"health={ctx.player_health*100:.0f}%"
                )
        return "\n".join(lines) if lines else "No matching clients"

    def _cmd_energy(self, args: list) -> str:
        """
        Show or set player energy/fuel-local-state value.
        Usage:
          energy                    - Show all players' energy
          energy [c<id>]            - Show a specific player's energy
          energy set <val> [c<id>]  - Set energy; 0..1 is a fraction, >1 is absolute units
        """
        if not self.server:
            return "Error: No server reference"

        max_energy = float(getattr(self.server, "player_energy_max", 100.0) or 100.0)
        if max_energy <= 0.0:
            max_energy = 100.0

        def _client_arg(tokens: list[str], default_active: bool = True):
            client_filter = None
            for token in tokens:
                if token.lower().startswith("c") and token[1:].isdigit():
                    client_filter = int(token[1:])
            if client_filter is not None:
                ctx, _addr = self._get_client_by_id(client_filter)
                return ctx
            if default_active:
                ctx, _addr = self._get_active_client()
                return ctx
            return None

        if args and args[0].lower() == "set":
            if len(args) < 2:
                return "Usage: energy set <value> [c<id>]"
            try:
                requested = float(args[1])
            except ValueError:
                return f"Invalid energy value: {args[1]}"

            new_energy = requested * max_energy if 0.0 <= requested <= 1.0 else requested
            new_energy = max(0.0, min(max_energy, new_energy))
            ctx = _client_arg(args[2:])
            if not ctx:
                return "Error: No connected client"

            old_energy = float(getattr(ctx, "player_energy", 0.0) or 0.0)
            ctx.player_energy = new_energy

            udp_handler = getattr(self.server, "udp_handler", None)
            if udp_handler and ctx.session and ctx.session.udp_addr:
                tick = self.server._get_network_tick(ctx)
                packet = self.server._build_local_state_heartbeat(
                    ctx,
                    tick=tick,
                    entity_id=ctx.session.entity_id or ctx.entity_id,
                    include_health=True,
                    health=self.server._get_health_value(ctx),
                    fuel=self.server._get_energy_value(ctx),
                )
                udp_handler.send_to(packet, ctx.session.udp_addr)

            return (
                f"Client {ctx.client_id}: energy {old_energy:.1f}/{max_energy:.1f} "
                f"({old_energy / max_energy * 100.0:.0f}%) -> "
                f"{new_energy:.1f}/{max_energy:.1f} ({new_energy / max_energy * 100.0:.0f}%)"
            )

        client_filter = None
        for token in args:
            if token.lower().startswith("c") and token[1:].isdigit():
                client_filter = int(token[1:])

        lines = []
        with self.server.clients_lock:
            for ctx in self.server.clients.values():
                if not ctx or not ctx.running:
                    continue
                if client_filter is not None and ctx.client_id != client_filter:
                    continue
                phase = ctx.session.phase.name if ctx.session else "NONE"
                name = ctx.session.username if ctx.session else f"Player{ctx.client_id}"
                energy = float(getattr(ctx, "player_energy", 0.0) or 0.0)
                lines.append(
                    f"Client {ctx.client_id} ({name}) [{phase}]: "
                    f"energy={energy:.1f}/{max_energy:.1f} ({energy / max_energy * 100.0:.0f}%)"
                )
        return "\n".join(lines) if lines else "No matching clients"

    def _cmd_spawn_entity(self, args: list) -> str:
        """
        Spawn a test entity with configurable type.
        Args: [entity_type] [x y z] [vx vy vz]
        Entity types: 0=TANK, 5=FLAK_SHELL, 6=PULSE_SHELL, 7=SHORT_MISSILE, etc.
        Default: type 6 (PULSE_SHELL) at (110, 60, 100) - visible above player
        """
        if not self.udp_handler or not self.session:
            return "Error: No UDP handler/session available"

        if not self.session.udp_addr:
            return "Error: No UDP client address"

        entity_type = 6  # PULSE_SHELL default
        x, y, z = 110.0, 110.0, 100.0  # Spawn above and in front of player (y=110)
        vx, vy, vz = 20.0, 0.0, 0.0

        try:
            if len(args) > 0:
                entity_type = int(args[0])
            if len(args) > 1:
                x = float(args[1])
            if len(args) > 2:
                y = float(args[2])
            if len(args) > 3:
                z = float(args[3])
            if len(args) > 4:
                vx = float(args[4])
            if len(args) > 5:
                vy = float(args[5])
            if len(args) > 6:
                vz = float(args[6])
        except ValueError as e:
            return f"spawn_entity arg parse error: {e}"

        from .weapons import Projectile, EntityType, build_projectile_spawn_packet
        from .packets import get_ticks

        # Generate unique entity ID
        if not hasattr(self, '_entity_id'):
            self._entity_id = 5000
        self._entity_id += 1

        # Map common entity types
        entity_names = {
            0: "TANK", 1: "SCOUT", 2: "ASSAULT_PLATFORM",
            5: "FLAK_SHELL", 6: "PULSE_SHELL", 7: "SHORT_MISSILE",
            8: "HUNTER", 9: "HEAVY_MISSILE", 10: "MINE",
        }
        type_name = entity_names.get(entity_type, f"TYPE_{entity_type}")

        proj = Projectile(
            entity_id=self._entity_id,
            entity_type=entity_type,  # Use raw int value
            owner_id=self.session.player_id or 1337,
            team=self.session.team_id or 2,  # Default to team 2 (player's team) to avoid friendly fire
            pos=(x, y, z),
            vel=(vx, vy, vz),
            spawn_time=0,
            lifetime=10.0
        )

        tick = get_ticks()
        packet = build_projectile_spawn_packet(proj, tick)

        print(f"[SPAWN_ENTITY] Spawning {type_name} (type={entity_type}) id={proj.entity_id}")
        print(f"[SPAWN_ENTITY] pos=({x:.1f},{y:.1f},{z:.1f}) vel=({vx:.1f},{vy:.1f},{vz:.1f})")
        print(f"[SPAWN_ENTITY] Packet ({len(packet)} bytes): {packet.hex()}")

        # Send via both UDP and TCP
        self.udp_handler.send_to(packet, self.session.udp_addr)
        if self.tcp_handler:
            self.tcp_handler.send(packet)

        return (f"Spawned {type_name} (type={entity_type}) id={self._entity_id} "
                f"pos=({x:.1f},{y:.1f},{z:.1f})")

    def _cmd_spawn_udp(self, args: list) -> str:
        """
        Send a UDP TANK packet using the Wulf-Forge bit layout.
        Args (positional):
          entity_id team_id unit_type x y z [vx vy vz]
        """
        if not self.udp_handler or not self.session:
            return "Error: No UDP handler/session available"

        if not self.session.udp_addr:
            return "Error: No UDP client address (try reconnecting)"

        entity_id = 1337
        team_id = 2
        unit_type = 0
        x, y, z = 100.0, 100.0, 100.0
        vx, vy, vz = 0.0, 0.0, 0.0

        try:
            if len(args) > 0:
                entity_id = int(args[0])
            if len(args) > 1:
                team_id = int(args[1])
            if len(args) > 2:
                unit_type = int(args[2])
            if len(args) > 3:
                x = float(args[3])
            if len(args) > 4:
                y = float(args[4])
            if len(args) > 5:
                z = float(args[5])
            if len(args) > 6:
                vx = float(args[6])
            if len(args) > 7:
                vy = float(args[7])
            if len(args) > 8:
                vz = float(args[8])
        except ValueError as e:
            return f"spawn_udp arg parse error: {e}"

        # Check if client is ready (has sent WANT_UPDATES indicating map loaded)
        if not self.session.want_updates_received:
            # Wait up to 3 seconds for client to be ready
            import time
            wait_start = time.monotonic()
            wait_timeout = 3.0
            while not self.session.want_updates_received:
                if time.monotonic() - wait_start > wait_timeout:
                    return "Error: Client not ready (no WANT_UPDATES received - map may not be loaded)"
                time.sleep(0.1)
            print(f"[spawn_udp] Client became ready after {time.monotonic() - wait_start:.2f}s")

        payload = build_udp_tank_packet_wf(
            net_id=entity_id,
            unit_type=unit_type,
            team_id=team_id,
            pos=(x, y, z),
            rot=(vx, vy, vz),
            include_vitals=True,
            weapon_id=0,
            health=1.0,
            energy=1.0,
        )
        # Log hex for debugging
        print(f"[TANK-HEX] len={len(payload)} hex={payload.hex().upper()}")
        self.udp_handler.send_to(payload, self.session.udp_addr)

        # Send CommMessage like wulf-forge does after TankPacket
        comm_pkt = build_chat_message("Spawning in...", source_id=entity_id)
        self.udp_handler.send_to(comm_pkt, self.session.udp_addr)

        return (f"Sent UDP TANK (wf) + CommMessage to {self.session.udp_addr}: "
                f"entity={entity_id} team={team_id} unit={unit_type} "
                f"pos=({x:.1f},{y:.1f},{z:.1f}) rot=({vx:.1f},{vy:.1f},{vz:.1f})")

    # Packet builders with args
    def _build_player(self, entity_id: str = "1337", spectator: str = "true") -> bytes:
        eid = int(entity_id)
        spec = spectator.lower() not in ('false', '0', 'no')
        return build_player(eid, spectator=spec)

    def _build_reincarnate(self, code: str = "17", message: str = "Welcome!") -> bytes:
        return build_reincarnate(int(code), message)

    def _build_add_to_roster(self, player_id: str = "1337", entity_id: str = "1337",
                              name: str = "Player", team: str = "2") -> bytes:
        return build_add_to_roster(int(player_id), int(entity_id), name, int(team))

    def _build_update_stats(self, account_id: str = "1337", team_id: str = "2") -> bytes:
        return build_update_stats(player_id=int(account_id), entity_id=int(account_id), team_id=int(team_id))

    def _build_birth_notice(self, entity_id: str = "1337") -> bytes:
        return build_birth_notice(int(entity_id))

    def _build_chat(self, message: str = "Hello") -> bytes:
        return build_chat_message(message)

    def _build_player_info(self, entity_oid: str = "1337", vehicle_type: str = "0",
                           x: str = "100", y: str = "50", z: str = "100") -> bytes:
        return build_player_info(int(entity_oid), int(vehicle_type),
                                  (float(x), float(y), float(z)))

    def _build_update_array_empty(self, tick: str = "0") -> bytes:
        return build_update_array_empty(int(tick))

    def _build_update_array_heartbeat(self, tick: str = "0", entity_id: str = "1337") -> bytes:
        return build_update_array_heartbeat(int(tick), int(entity_id))

    def _build_update_array_tank(self, entity_id: str = "1337", entity_type: str = "0",
                                  team: str = "2", x: str = "100", y: str = "50",
                                  z: str = "100", behavior_type: str = "0",
                                  interp: str = "0", interp_bits: str = "16") -> bytes:
        return build_update_array_create_tank(
            tick=0, entity_id=int(entity_id), entity_type=int(entity_type),
            team=int(team), pos=(float(x), float(y), float(z)),
            behavior_type=int(behavior_type),
            include_interp=(str(interp).lower() in ('1', 'true', 'yes', 'on')),
            interp_bits=int(interp_bits)
        )

    def _build_login_status(self, code: str = "8") -> bytes:
        return build_login_status(int(code))

    def _build_bps_response(self, rate: str = "0", approved: str = "true") -> bytes:
        return build_bps_response(int(rate), approved.lower() in ('true', '1', 'yes'))

    def _build_game_clock(self, time_ms: str = "0", running: str = "true",
                           round_time_ms: str = "3600000", extra: str = "0") -> bytes:
        return build_game_clock(
            int(time_ms),
            running.lower() in ('true', '1', 'yes'),
            int(round_time_ms),
        )
