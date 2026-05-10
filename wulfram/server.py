"""
Main server: Orchestrates the protocol flow using layered components.

NOTE: Use manage_server.py to start/stop the server instead of running this directly.
This avoids orphaned processes and provides clean shutdown handling.

    python server/manage_server.py start
    python server/manage_server.py stop
    python server/manage_server.py restart
"""

import ipaddress
import json
import math
import os
import secrets
import socket
import struct
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, Sequence

from .session import Session, Phase, FEATURES
from .transport import TCPHandler, UDPHandler, PacketLogger, print_packet
from .codec import BitReader
from .control import ControlServer
from .terrain import Terrain
from .building_collision import BuildingCollisionAssets, BuildingEntity
from .world_collision import TerrainContact, TerrainGridCollision
from .physics import _extract_euler_angles, _matrix3_from_euler_xyz, _normalize_angle_client
from .weapons import (
    WeaponSystem, build_projectile_spawn_packet, EntityType, BehaviorSlot,
    VEHICLE_PHYSICS_CONFIGS, TANK_WEAPON_SLOTS,
    OG_DIRECT_TRIGGER_WEAPON_SLOTS,
)
from .jump_jets import JumpJetSystem
from .client import ClientContext
from wulfram2_protocol.entities import (
    ACTION_ANALOG_SLOTS,
    ACTION_DUMP_CONTROL_SLOTS,
    JUMP_JET_CONFIGS,
    JUMP_JET_SPAWN_LOCKOUT,
    LOCAL_STATE_PRIMARY_TURRET_WEAPON_TYPES,
    LOCAL_STATE_SECONDARY_TURRET_WEAPON_TYPES,
    OG_PHYSICS_TIMESTEP_FACTOR,
    OG_TANK_SPRING_STRETCH_SPEED_DENOMINATOR,
    WEAPON_NAMES,
    tank_altitude_mobility_factor,
    tank_hover_clearance_target,
    tank_slope_mobility_factor,
    tank_spring_average_clearance,
    tank_spring_attitude_step,
    tank_body_matrix_drive_basis,
    tank_body_matrix_with_heading,
    tank_spring_force_attitude_step,
    tank_suspension_local_sample_offsets,
    tank_suspension_world_sample_offsets,
    tank_suspension_lift_accel,
    tank_spring_scalar_stretch_ratio,
    tank_softbody_control_slot_value,
    tank_softbody_horizontal_damping,
    tank_softbody_suspension_force,
    tank_fuel_mobility_factor,
    tank_terrain_contact_coupling,
    terrain_aligned_basis,
    mesh_aabb_half_extents_from_vertices,
    entity_interpolate_toward_target_decision,
    rigid_body_point_velocity,
    resolve_iterative_terrain_start_contact,
    solve_static_terrain_constraint,
    vehicle_runtime_speed,
)


@dataclass(frozen=True)
class _StaticWorldRayNode:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    children: Optional[tuple[Optional["_StaticWorldRayNode"], Optional["_StaticWorldRayNode"], Optional["_StaticWorldRayNode"], Optional["_StaticWorldRayNode"]]]
    building_ids: tuple[int, ...]


_STATIC_WORLD_RAY_STOP = object()

from .packets import (
    PacketType, get_packet_name, get_ticks,
    build_hello_session_key, build_hello_udp_config, build_hello_verified,
    build_ping_reply,
    build_identified_udp, build_login_status, build_tank_packet,
    build_udp_tank_packet_wf, build_update_array_heartbeat,
    build_chat_message, build_add_to_roster, build_update_stats, build_update_stats_team_first, build_player, build_player_info,
    build_birth_notice, build_game_clock, build_reincarnate,
    build_update_array_create_tank, build_update_array_player_update,
    build_view_update_create_tank,
    build_update_array_multi, build_view_update_multi, build_view_update_player_update,
    get_behavior_weapon_capability_counts, build_world_stats,
    build_delete_object,
    build_ship_status, build_carrying_info, build_uplink_info, build_supply_ship_info,
    _encode_health_bits, _compress_value, VEC_VEL_MAX, VEC_VEL_RANGE,
    build_transient_array, FX_CHAIN_GUN_FIRE, FX_PULSE_FIRE,
    FX_FLAK_FIRE, FX_MISSILE_FIRE, FX_IMPACT_VEHICLE,
    FX_IMPACT_BUILDING, FX_IMPACT_TERRAIN,
    get_behavior_tank_spring_local_offsets,
)
from . import handlers
from .pktlog import PacketLog

class WulframServer:
    """
    Wulfram2 game server emulator with multi-client support.

    Each client runs in its own thread with its own ClientContext.
    The UDP handler is shared across all clients.
    """

    def __init__(self, host: str = None, port: int = 2627):
        # Load .env file if present (written by mp_server.ps1 for detached mode)
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())
        self.host = host or os.environ.get("WULFRAM_BIND_ADDR", "0.0.0.0")
        # Address to advertise to clients for UDP (e.g. when binding 0.0.0.0)
        self.public_addr = os.environ.get("WULFRAM_PUBLIC_ADDR", self.host)
        try:
            self.port = int(os.environ.get("WULFRAM_PORT", str(port)))
        except ValueError:
            self.port = port
        self.logger = PacketLogger()
        self.udp_handler: Optional[UDPHandler] = None
        self.running = False
        self.control_server = ControlServer(port=port + 1)  # Control on port+1 (2628)

        # Game clock — tracks elapsed time since first player spawned
        self._game_start_time: float = 0.0  # monotonic time of first spawn (0 = not started)

        # Multi-client management
        self.clients: Dict[int, ClientContext] = {}
        self.clients_lock = threading.Lock()
        self.next_client_id = 1
        # Match wulf-forge default player/entity ID space (config.toml uses 1337).
        # Very low IDs can collide with implicit client/map assumptions.
        try:
            self.next_entity_id = int(os.environ.get("WULFRAM_START_ENTITY_ID", "1337"))
        except ValueError:
            self.next_entity_id = 1337
        if self.next_entity_id <= 0:
            self.next_entity_id = 1337

        # UDP address to client mapping for packet routing
        self.udp_addr_to_client: Dict[tuple, ClientContext] = {}
        # Session key to client mapping for deterministic UDP binding
        self.session_key_to_client: Dict[str, ClientContext] = {}

        # Coordinate system config (defaults to z-up).
        self.up_axis = os.environ.get("WULFRAM_UP_AXIS", "z").lower()
        if self.up_axis not in ("y", "z"):
            self.up_axis = "z"
        spawn_height_env = os.environ.get("WULFRAM_SPAWN_HEIGHT")
        try:
            # Default to ground-height spawn to avoid immediate fall-damage overlays.
            self.spawn_height = float(spawn_height_env) if spawn_height_env is not None else 5.0
        except ValueError:
            self.spawn_height = 5.0
        try:
            self.spawn_delay_seconds = float(os.environ.get("WULFRAM_SPAWN_DELAY", "6.0"))
        except ValueError:
            self.spawn_delay_seconds = 6.0
        # Default to server-side team-select spawn for stability; explicit spawn-point
        # selection is still available by setting WULFRAM_SPAWN_ON_TEAM_SELECT=0.
        self.spawn_on_team_select = os.environ.get("WULFRAM_SPAWN_ON_TEAM_SELECT", "1") == "1"
        self.team_switch_send_reincarnate = os.environ.get("WULFRAM_TEAM_SWITCH_REINCARNATE", "1") == "1"
        self.team_switch_send_update_stats = (
            os.environ.get("WULFRAM_TEAM_SWITCH_UPDATE_STATS", "1").strip().lower()
            not in ("0", "false", "off", "no")
        )
        self.team_switch_update_stats_transport = os.environ.get(
            "WULFRAM_TEAM_SWITCH_UPDATE_STATS_TRANSPORT",
            "udp",
        ).strip().lower()
        if self.team_switch_update_stats_transport not in ("udp", "tcp", "auto"):
            self.team_switch_update_stats_transport = "udp"
        self.team_switch_update_stats_variant = os.environ.get(
            "WULFRAM_TEAM_SWITCH_UPDATE_STATS_VARIANT",
            "canonical",
        ).strip().lower()
        if self.team_switch_update_stats_variant not in ("canonical", "team_first"):
            self.team_switch_update_stats_variant = "canonical"
        self.team_switch_send_roster = (
            os.environ.get("WULFRAM_TEAM_SWITCH_ROSTER", "0").strip().lower()
            not in ("0", "false", "off", "no")
        )
        self.team_switch_send_entry_packets = (
            os.environ.get("WULFRAM_TEAM_SWITCH_ENTRY_PACKETS", "1").strip().lower()
            not in ("0", "false", "off", "no")
        )
        # Allow explicit spawn-point packets to recover clients stuck on entry-map
        # after auto-spawn already marked the session IN_GAME.
        self.spawn_allow_point_override = os.environ.get("WULFRAM_SPAWN_POINT_OVERRIDE", "1") == "1"
        try:
            self.spawn_point_override_min_interval = float(
                os.environ.get("WULFRAM_SPAWN_POINT_OVERRIDE_MIN_INTERVAL", "0.0")
            )
        except ValueError:
            self.spawn_point_override_min_interval = 0.0
        try:
            self.spawn_force_after = float(os.environ.get("WULFRAM_SPAWN_FORCE_AFTER", "12.0"))
        except ValueError:
            self.spawn_force_after = 12.0
        try:
            # Avoid false "crash" diagnosis when scripted input arrives slightly late.
            self.inactivity_timeout = float(os.environ.get("WULFRAM_INACTIVITY_TIMEOUT", "120.0"))
        except ValueError:
            self.inactivity_timeout = 120.0
        try:
            self.remote_idle_timeout = float(os.environ.get("WULFRAM_REMOTE_IDLE_TIMEOUT", "900.0"))
        except ValueError:
            self.remote_idle_timeout = 900.0
        # Keep spawns pinned to ground unless explicitly disabled.
        self.spawn_sets_ground_level = os.environ.get("WULFRAM_SPAWN_SET_GROUND", "1") == "1"
        try:
            self.ground_override_release_distance = float(
                os.environ.get("WULFRAM_GROUND_OVERRIDE_RELEASE_DISTANCE", "24.0")
            )
        except ValueError:
            self.ground_override_release_distance = 24.0
        try:
            self.ground_override_release_height = float(
                os.environ.get("WULFRAM_GROUND_OVERRIDE_RELEASE_HEIGHT", "4.0")
            )
        except ValueError:
            self.ground_override_release_height = 4.0
        try:
            self.ground_override_release_terrain_distance = float(
                os.environ.get("WULFRAM_GROUND_OVERRIDE_RELEASE_TERRAIN_DISTANCE", "4.0")
            )
        except ValueError:
            self.ground_override_release_terrain_distance = 4.0
        try:
            self.ground_override_release_terrain_height = float(
                os.environ.get("WULFRAM_GROUND_OVERRIDE_RELEASE_TERRAIN_HEIGHT", "0.75")
            )
        except ValueError:
            self.ground_override_release_terrain_height = 0.75
        # Spawn packet toggles (useful for crash isolation).
        # Spawn sequence toggles (default to Wulf-Forge minimal behavior).
        self.spawn_send_udp_tank = os.environ.get("WULFRAM_SPAWN_UDP_TANK", "1") == "1"
        spawn_player_info_env = os.environ.get("WULFRAM_SPAWN_PLAYER_INFO")
        self.spawn_send_player_info_explicit = spawn_player_info_env is not None
        if self.spawn_send_player_info_explicit:
            self.spawn_send_player_info = spawn_player_info_env == "1"
        else:
            # AzureFishy decompile shows PLAYER_INFO is the canonical local-player
            # init path. Keep loopback/Python on the existing minimal path by
            # default, but send PLAYER_INFO to remote OG clients unless explicitly
            # disabled.
            self.spawn_send_player_info = False
        # Keep UPDATE_ARRAY pre-create opt-in until it preserves the OG
        # local sync/camera bootstrap. Proven correction sessions can use the
        # local sync state plus server_expected, even when the camera global is 0.
        self.spawn_send_update_array = os.environ.get("WULFRAM_SPAWN_UPDATE_ARRAY", "0") == "1"
        self.spawn_send_game_clock = os.environ.get("WULFRAM_SPAWN_GAME_CLOCK", "1") == "1"
        self.spawn_send_comm_message = (
            os.environ.get("WULFRAM_SPAWN_COMM_MESSAGE", "0").strip().lower()
            not in ("0", "false", "off", "no")
        )
        # Wulf-forge does NOT send REINCARNATE(0x11) during spawn.
        self.spawn_send_reincarnate = os.environ.get("WULFRAM_SPAWN_REINCARNATE", "0") == "1"
        spawn_entry_transition = os.environ.get("WULFRAM_SPAWN_ENTRY_TRANSITION", "off").strip().lower()
        if spawn_entry_transition not in ("0", "1", "false", "true", "off", "on", "auto", "auto-remote"):
            spawn_entry_transition = "off"
        self.spawn_entry_transition = spawn_entry_transition
        self.spawn_send_birth_notice = os.environ.get("WULFRAM_SPAWN_BIRTH_NOTICE", "0") == "1"
        self.spawn_send_player_packet = os.environ.get("WULFRAM_SPAWN_PLAYER_PACKET", "0") == "1"
        self.spawn_player_spectator = os.environ.get("WULFRAM_SPAWN_PLAYER_SPECTATOR", "0") == "1"
        self.player_info_properties_mode = os.environ.get("WULFRAM_PLAYER_INFO_PROPERTIES", "team").strip().lower()
        # Spawn points are required for reliable entry-map progression.
        # Keep ON by default; allow explicit opt-out for protocol experiments.
        spawn_points_env = os.environ.get("WULFRAM_SEND_SPAWN_POINTS", "1")
        FEATURES.send_spawn_points = spawn_points_env.strip().lower() not in ("0", "false", "off", "no")
        behavior_env = os.environ.get("WULFRAM_SEND_BEHAVIOR")
        if behavior_env is not None:
            FEATURES.send_behavior_packet = behavior_env.strip().lower() not in ("0", "false", "off", "no")
        translation_env = os.environ.get("WULFRAM_SEND_TRANSLATION")
        if translation_env is not None:
            FEATURES.send_translation_packet = translation_env.strip().lower() not in ("0", "false", "off", "no")
        world_stats_env = os.environ.get("WULFRAM_SEND_WORLD_STATS_ON_LOGIN")
        if world_stats_env is not None:
            FEATURES.send_world_stats_on_login = world_stats_env.strip().lower() not in ("0", "false", "off", "no")
        game_clock_login_env = os.environ.get("WULFRAM_SEND_GAME_CLOCK_ON_LOGIN")
        self.send_game_clock_on_login = (
            game_clock_login_env is not None
            and game_clock_login_env.strip().lower() not in ("0", "false", "off", "no")
        )
        request_start_login_env = os.environ.get("WULFRAM_SEND_REQUEST_START_ON_LOGIN")
        self.send_request_start_on_login = (
            request_start_login_env is not None
            and request_start_login_env.strip().lower() not in ("0", "false", "off", "no")
        )
        roster_login_env = os.environ.get("WULFRAM_SEND_ROSTER_ON_LOGIN")
        self.send_roster_on_login = (
            roster_login_env is not None
            and roster_login_env.strip().lower() not in ("0", "false", "off", "no")
        )
        tick_loop_env = os.environ.get("WULFRAM_TICK_LOOP_ENABLED")
        if tick_loop_env is not None:
            FEATURES.tick_loop_enabled = tick_loop_env.strip().lower() not in ("0", "false", "off", "no")
        auto_login_env = os.environ.get("WULFRAM_AUTO_LOGIN")
        if auto_login_env is not None:
            FEATURES.auto_login = auto_login_env.strip().lower() not in ("0", "false", "off", "no")
        auto_join_env = os.environ.get("WULFRAM_AUTO_JOIN_TEAM")
        if auto_join_env is not None:
            FEATURES.auto_join_team = auto_join_env.strip().lower() not in ("0", "false", "off", "no")
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
        # Remote player updates (multiplayer position sync) â€” independent of local heartbeat.
        self.send_remote_updates = os.environ.get("WULFRAM_SEND_REMOTE_UPDATES", "1") == "1"
        # T3 OG-participant isolation gates. These default on so normal
        # behavior is unchanged, but live probes can selectively disable
        # target->OG visibility streams without disturbing loopback clients.
        self.og_viewer_roster_entry = os.environ.get("WULFRAM_OG_VIEWER_ROSTER_ENTRY", "1") == "1"
        self.og_viewer_entity_create = os.environ.get("WULFRAM_OG_VIEWER_ENTITY_CREATE", "1") == "1"
        self.og_viewer_remote_updates = os.environ.get("WULFRAM_OG_VIEWER_REMOTE_UPDATES", "1") == "1"
        # Wulf-forge sends UPDATE_ARRAY over UDP only.
        self.send_updates_tcp = os.environ.get("WULFRAM_UPDATE_TCP", "0") == "1"
        self.send_updates_udp = os.environ.get("WULFRAM_UPDATE_UDP", "1") == "1"
        # Local player update mask (align with wulf-forge minimal updates by default).
        # Modes: pos, pos_vel, pos_rot, pos_vel_rot
        # Default to full local transform updates to avoid client-side state divergence.
        self.local_update_mode = os.environ.get("WULFRAM_LOCAL_UPDATE_MODE", "pos_vel_rot").strip().lower()
        # Remote update shape to avoid malformed packets destabilizing HUD state in multi-client.
        # Modes: pos, pos_rot, pos_vel_rot, heartbeat, off (default: pos_rot).
        self.remote_update_mode = os.environ.get("WULFRAM_REMOTE_UPDATE_MODE", "pos_vel_rot").strip().lower()
        # Throttle remote player updates: interval in seconds (0 = every tick).
        # 30Hz tick rate = every tick; lower values reduce packet flood.
        try:
            self.remote_update_interval = float(os.environ.get("WULFRAM_REMOTE_UPDATE_INTERVAL", "0"))
        except ValueError:
            self.remote_update_interval = 0
        # For remote player updates, set is_manned bit (likely required for player vehicles).
        self.remote_update_is_manned = os.environ.get("WULFRAM_REMOTE_IS_MANNED", "0") == "1"
        # Yaw offset for remote entities (degrees). Tunable via control cmd: heading remote_offset <deg>
        self.remote_yaw_offset = math.radians(float(os.environ.get("WULFRAM_REMOTE_YAW_OFFSET_DEG", "0")))
        # Negate remote yaw before sending (applies after offset)
        self.remote_yaw_negate = os.environ.get("WULFRAM_REMOTE_YAW_NEGATE", "0") == "1"
        # Combine local + remote updates into a single UPDATE_ARRAY per tick.
        self.combine_update_arrays = os.environ.get("WULFRAM_COMBINE_UPDATE_ARRAYS", "0") == "1"
        # Local-state in projectile UPDATE_ARRAY packets is fragile and can
        # misalign entity decoding. Keep projectile packets entity-only by
        # default; normal heartbeat UPDATE_ARRAY packets keep HUD vitals stable.
        self.projectile_local_stats = os.environ.get("WULFRAM_PROJECTILE_LOCAL_STATS", "0") == "1"
        self.debug_projectiles = (
            os.environ.get("WULFRAM_DEBUG_PROJECTILES", "0") == "1"
            or os.environ.get("WULFRAM_DEBUG_AIM", "0") == "1"
        )
        try:
            self.projectile_config = int(os.environ.get("WULFRAM_PROJECTILE_CONFIG", "0"))
        except ValueError:
            self.projectile_config = 0
        # Projectile spawn snap/teleport flag (misleadingly named "static" elsewhere).
        # Default ON: wulf-forge sets this bit on entity creation.
        projectile_snap_env = os.environ.get("WULFRAM_PROJECTILE_SPAWN_SNAP")
        if projectile_snap_env is None:
            projectile_snap_env = os.environ.get("WULFRAM_PROJECTILE_STATIC", "1")
        self.projectile_spawn_snap = projectile_snap_env == "1"
        # Backwards-compat alias; "static" here really means spawn snap/teleport.
        self.projectile_static = self.projectile_spawn_snap
        try:
            self.viewpoint_timeout = float(os.environ.get("WULFRAM_VIEWPOINT_TIMEOUT", "1.0"))
        except ValueError:
            self.viewpoint_timeout = 1.0
        self.debug_viewpoint = os.environ.get("WULFRAM_DEBUG_VIEWPOINT", "0") == "1"
        self.debug_udp_raw = os.environ.get("WULFRAM_DEBUG_UDP_RAW", "0") == "1"
        self.weapon_energy_enabled = os.environ.get("WULFRAM_WEAPON_ENERGY_ENABLED", "1") == "1"
        try:
            self.player_energy_max = float(os.environ.get("WULFRAM_PLAYER_ENERGY_MAX", "100.0"))
        except ValueError:
            self.player_energy_max = 100.0
        if self.player_energy_max <= 0.0:
            self.player_energy_max = 100.0
        try:
            self.player_energy_regen = float(os.environ.get("WULFRAM_PLAYER_ENERGY_REGEN", "10.0"))
        except ValueError:
            self.player_energy_regen = 10.0
        if self.player_energy_regen < 0.0:
            self.player_energy_regen = 0.0
        # VIEW_UPDATE is an auxiliary replay/correction path. Primary gameplay
        # replication stays on UPDATE_ARRAY unless explicitly experimenting.
        self.view_update_enabled = os.environ.get("WULFRAM_VIEW_UPDATE", "0") == "1"
        # Periodic auxiliary VIEW_UPDATE loop. Default OFF for stability.
        self.view_update_loop = os.environ.get("WULFRAM_VIEW_UPDATE_LOOP", "0") == "1"
        # Keep local-stats out of VIEW_UPDATE unless explicitly requested.
        self.view_update_local_stats = os.environ.get("WULFRAM_VIEW_UPDATE_LOCAL_STATS", "0") == "1"
        # Wulf-forge marks HEALTH dirty on spawn; include entity vitals by default.
        # This uses update-mask bits 5/7 (health/energy per wulf-forge).
        # Entity vitals (mask bits 5/7) should be sent only when dirty; default off.
        self.view_update_entity_vitals = os.environ.get("WULFRAM_VIEW_ENTITY_VITALS", "0") == "1"
        self.update_entity_vitals = os.environ.get("WULFRAM_UPDATE_ENTITY_VITALS", "0") == "1"
        self.debug_vitals = os.environ.get("WULFRAM_DEBUG_VITALS", "0") == "1"
        self.debug_health_pattern = os.environ.get("WULFRAM_DEBUG_HEALTH_PATTERN", "0") == "1"
        try:
            self.debug_health_value = float(os.environ.get("WULFRAM_DEBUG_HEALTH_VALUE", "1.0"))
        except ValueError:
            self.debug_health_value = 1.0
        try:
            self.debug_health_low = float(os.environ.get("WULFRAM_DEBUG_HEALTH_LOW", "0.2"))
        except ValueError:
            self.debug_health_low = 0.2
        try:
            self.debug_health_period = float(os.environ.get("WULFRAM_DEBUG_HEALTH_PERIOD", "1.0"))
        except ValueError:
            self.debug_health_period = 1.0
        if self.debug_health_pattern:
            print(
                "[WARN] Debug health pattern enabled: "
                f"value={self.debug_health_value} low={self.debug_health_low} "
                f"period={self.debug_health_period}s (HUD will flicker red)"
            )
        try:
            self.view_update_interval = float(os.environ.get("WULFRAM_VIEW_UPDATE_INTERVAL", "0.1"))
        except ValueError:
            self.view_update_interval = 0.1
        try:
            self.update_grace_seconds = float(os.environ.get("WULFRAM_UPDATE_GRACE", "0.0"))
        except ValueError:
            self.update_grace_seconds = 0.0
        # Wulf-forge sends UPDATE_ARRAY immediately; keep OFF by default while aligning.
        self.require_client_tick = os.environ.get("WULFRAM_REQUIRE_CLIENT_TICK", "0") == "1"
        try:
            self.multi_spawn_offset = float(os.environ.get("WULFRAM_MULTI_SPAWN_OFFSET", "0.0"))
        except ValueError:
            self.multi_spawn_offset = 0.0
        try:
            self.weapon_id = int(os.environ.get("WULFRAM_WEAPON_ID", "0"))
        except ValueError:
            self.weapon_id = 0
        if self.weapon_id < 0 or self.weapon_id > 31:
            print(f"[WEAPON] Invalid weapon_id={self.weapon_id}, defaulting to 0")
            self.weapon_id = 0
        # Default to body-aligned firing for stability. Slot/viewpoint aim remains
        # available via env toggles for targeted experiments.
        self.projectile_aim_source = os.environ.get("WULFRAM_PROJECTILE_AIM_SOURCE", "body").lower()
        self.projectiles_enabled = os.environ.get("WULFRAM_PROJECTILES_ENABLED", "1") == "1"
        self.remote_projectiles = os.environ.get("WULFRAM_REMOTE_PROJECTILES", "1") == "1"
        self.remote_combat_observer_packets = (
            os.environ.get("WULFRAM_REMOTE_COMBAT_OBSERVER_PACKETS", "1") == "1"
        )
        # Projectile update mode:
        # 0=no updates after spawn, 1=5Hz, 2=15Hz (default), 3=30Hz
        try:
            self.projectile_update_mode = int(os.environ.get("WULFRAM_PROJECTILE_UPDATE_MODE", "2"))
        except ValueError:
            self.projectile_update_mode = 2
        if self.projectile_update_mode < 0 or self.projectile_update_mode > 3:
            self.projectile_update_mode = 2
        try:
            self.projectile_collision_radius = abs(float(os.environ.get("WULFRAM_PROJECTILE_COLLISION_RADIUS", "2.0")))
        except ValueError:
            self.projectile_collision_radius = 2.0
        if self.projectile_collision_radius < 0.25:
            self.projectile_collision_radius = 0.25
        # Remote OG impact FX are still prone to live D_ERR disconnects.
        # Keep TRANSIENT_ARRAY enabled for loopback/Python validation, but
        # suppress it remotely until the 0x0D path is verified end to end.
        self.remote_transient_fx = os.environ.get("WULFRAM_REMOTE_TRANSIENT_FX", "0") == "1"
        # TankPacket vitals OFF: flag=0 means the client skips the entire
        # local_state read (weapon, health, fuel, ammo, turret).  This ensures
        # unit_type/net_id/team/pos are parsed correctly so the entity is
        # created and the entry map dismisses.
        # Health is instead delivered by the heartbeat UPDATE_ARRAY which
        # includes entity_count=1, calling Network_record_update_stats (sets
        # EAX for tick guard) then sync_local_player (applies health from ESI).
        # The BEHAVIOR packet clears weapon_def[0]+0x170 (turret flag), so the
        # heartbeat local_state has no turret bit misalignment.
        tank_vitals_raw = os.environ.get("WULFRAM_TANK_VITALS", "0").strip().lower()
        self.tank_vitals = tank_vitals_raw in ("1", "true", "on", "yes")
        # Map configuration (WORLD_STATS + spawn points).
        # Default to Wulf-Forge's startup map ("crossroads").
        self.map_name = os.environ.get("WULFRAM_MAP_NAME", "crossroads")
        self.map_config = self._load_map_config()
        map_key = self.map_name.lower()
        map_conf = self.map_config.get(map_key, {}) if isinstance(self.map_config, dict) else {}
        map_conf_has_rows = "grid_rows" in map_conf
        map_conf_has_cols = "grid_cols" in map_conf
        map_rows_env = os.environ.get("WULFRAM_MAP_GRID_ROWS")
        map_cols_env = os.environ.get("WULFRAM_MAP_GRID_COLS")
        map_scale_env = os.environ.get("WULFRAM_MAP_SCALE")
        allow_large_env = os.environ.get("WULFRAM_MAP_ALLOW_LARGE_GRID")
        self.map_allow_large_grid = bool(map_conf.get("allow_large_grid", False))
        if allow_large_env is not None:
            self.map_allow_large_grid = allow_large_env.strip().lower() in ("1", "true", "yes", "on")
        try:
            if map_rows_env is not None:
                self.map_grid_rows = int(map_rows_env)
            elif "grid_rows" in map_conf:
                self.map_grid_rows = int(map_conf["grid_rows"])
            else:
                self.map_grid_rows = 1
        except ValueError:
            self.map_grid_rows = 1
        try:
            if map_cols_env is not None:
                self.map_grid_cols = int(map_cols_env)
            elif "grid_cols" in map_conf:
                self.map_grid_cols = int(map_conf["grid_cols"])
            else:
                self.map_grid_cols = 1
        except ValueError:
            self.map_grid_cols = 1
        try:
            if map_scale_env is not None:
                self.map_scale = float(map_scale_env)
            elif "scale" in map_conf:
                self.map_scale = float(map_conf["scale"])
            else:
                self.map_scale = 1.0
        except ValueError:
            self.map_scale = 1.0
        # Clamp explicit config/env grid sizes if the client treats WORLD_STATS rows/cols as signed.
        if (self.map_grid_rows > 127 or self.map_grid_cols > 127) and not self.map_allow_large_grid:
            clamped_rows = min(self.map_grid_rows, 127)
            clamped_cols = min(self.map_grid_cols, 127)
            if clamped_rows != self.map_grid_rows or clamped_cols != self.map_grid_cols:
                print(
                    "[MAP] Config grid exceeds 127; "
                    f"clamping {self.map_grid_rows}x{self.map_grid_cols} -> {clamped_rows}x{clamped_cols}"
                )
                self.map_grid_rows = clamped_rows
                self.map_grid_cols = clamped_cols
        # Auto-detect grid rows/cols from map land file when not explicitly set.
        needs_rows = map_rows_env is None and not map_conf_has_rows
        needs_cols = map_cols_env is None and not map_conf_has_cols
        if needs_rows or needs_cols:
            land_grid = self._load_map_land_grid()
            if land_grid:
                rows, cols, size_x, size_y = land_grid
                # WORLD_STATS uses u8; client may treat this as signed char.
                # Clamp to 127 to avoid negative values if sign-extended unless explicitly allowed.
                if (rows > 127 or cols > 127) and not self.map_allow_large_grid:
                    clamped_rows = min(rows, 127)
                    clamped_cols = min(cols, 127)
                    print(
                        "[MAP] Land grid exceeds 127; "
                        f"clamping {rows}x{cols} -> {clamped_rows}x{clamped_cols}"
                    )
                    rows, cols = clamped_rows, clamped_cols
                elif rows > 127 or cols > 127:
                    print(
                        "[MAP] Land grid exceeds 127; "
                        f"keeping {rows}x{cols} (allow_large_grid=1)"
                    )
                if needs_rows:
                    self.map_grid_rows = rows
                if needs_cols:
                    self.map_grid_cols = cols
                print(
                    "[MAP] Land grid detected "
                    f"{rows}x{cols} (size={size_x:.1f}x{size_y:.1f})"
                )
        self.map_spawn_points = self._parse_spawn_points_env(os.environ.get("WULFRAM_SPAWN_POINTS", ""))
        default_use_map_spawn = "1" if self.map_name.lower() == "crossroads" else "0"
        self.use_map_spawn_points = os.environ.get("WULFRAM_USE_MAP_SPAWN", default_use_map_spawn) == "1"
        self.align_spawn_points_to_terrain = (
            os.environ.get("WULFRAM_ALIGN_SPAWN_POINTS_TO_TERRAIN", "0") == "1"
        )
        self.align_spawn_pos_to_terrain = (
            os.environ.get("WULFRAM_ALIGN_SPAWN_POS_TO_TERRAIN", "0") == "1"
        )
        self.default_flat_spawn_pos = self._parse_spawn_pos_env(
            os.environ.get("WULFRAM_DEFAULT_FLAT_SPAWN", "")
        )
        if self.default_flat_spawn_pos is None:
            self.default_flat_spawn_pos = self._get_builtin_flat_spawn_pos()
        self.force_default_spawn_pos = os.environ.get("WULFRAM_FORCE_DEFAULT_SPAWN", "1") == "1"
        # Load building entities for collision detection
        self._building_entities = {}
        self._building_health = {}  # oid -> health (1.0 = full, 0.0 = destroyed)
        self._turret_last_fire = {}  # oid -> monotonic time of last fire
        self._dynamic_building_ids: set[int] = set()
        self._dynamic_building_sources: dict[int, dict[str, Any]] = {}
        self._build_uplink_command_events: list[dict[str, Any]] = []
        self._uplink_ships: dict[int, dict[str, Any]] = {}
        self._static_world_raycast_root: Optional[_StaticWorldRayNode] = None
        self._building_collision = BuildingCollisionAssets()
        if self._building_collision.load_default():
            print(
                f"[COLLISION] Loaded {len(self._building_collision.models)} models "
                f"from {self._building_collision.shapes_path}"
            )
        else:
            print(
                "[COLLISION] Mesh building collision unavailable: "
                f"{self._building_collision.last_error}"
            )
        gravity_env = os.environ.get("WULFRAM_GRAVITY")
        try:
            self.gravity = float(gravity_env) if gravity_env is not None else -50.0
        except ValueError:
            self.gravity = -50.0
        ground_level_env = os.environ.get("WULFRAM_GROUND_LEVEL")
        try:
            self.ground_level = float(ground_level_env) if ground_level_env is not None else 5.0
        except ValueError:
            self.ground_level = 5.0
        # Terrain heightmap for dynamic ground level and slope-aware physics.
        self.terrain: Optional[Terrain] = None
        self.terrain_pitch_enabled = os.environ.get("WULFRAM_TERRAIN_PITCH", "1") == "1"
        self.terrain_collision_with_ground_override = (
            os.environ.get("WULFRAM_TERRAIN_COLLISION_WITH_GROUND_OVERRIDE", "0") == "1"
        )
        self.entity_terrain_collision_enabled = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_COLLISION", "1").strip().lower()
            not in {"0", "false", "off", "no"}
        )
        try:
            self.terrain_height_offset = float(os.environ.get("WULFRAM_TERRAIN_HEIGHT_OFFSET", "5.0"))
        except ValueError:
            self.terrain_height_offset = 5.0
        try:
            self.terrain_physics_height_offset = float(
                os.environ.get("WULFRAM_TERRAIN_PHYSICS_HEIGHT_OFFSET", "0.0")
            )
        except ValueError:
            self.terrain_physics_height_offset = 0.0
        try:
            self.tank_spring_base_offset = float(
                os.environ.get("WULFRAM_TANK_SPRING_BASE_OFFSET", "2.0")
            )
        except ValueError:
            self.tank_spring_base_offset = 2.0
        self.tank_drive_terrain_aligned = (
            os.environ.get("WULFRAM_TANK_DRIVE_TERRAIN_ALIGNED", "0") == "1"
        )
        self.tank_drive_body_matrix = (
            os.environ.get("WULFRAM_TANK_DRIVE_BODY_MATRIX", "1").strip().lower()
            not in ("0", "false", "off", "no")
        )
        self.tank_terrain_contact_coupling_enabled = (
            os.environ.get("WULFRAM_TANK_TERRAIN_CONTACT_COUPLING", "1") == "1"
        )
        self.tank_spring_sample_local_offsets = get_behavior_tank_spring_local_offsets()
        # Tank vertical support. Default to the decompile-shaped softbody
        # stand-in; keep the old compact center-lift model available only for
        # explicit comparison/debugging.
        self.tank_suspension_enabled = os.environ.get("WULFRAM_TANK_SUSPENSION", "1") == "1"
        self.tank_suspension_model = os.environ.get(
            "WULFRAM_TANK_SUSPENSION_MODEL",
            "softbody",
        ).strip().lower()
        self.tank_softbody_per_point_force = (
            os.environ.get("WULFRAM_TANK_SOFTBODY_PER_POINT_FORCE", "0").strip().lower()
            in {"1", "true", "on", "yes"}
        )
        self.tank_softbody_piecewise_height = (
            os.environ.get("WULFRAM_TANK_SOFTBODY_PIECEWISE_HEIGHT", "0").strip().lower()
            in {"1", "true", "on", "yes"}
        )
        self.tank_softbody_decompile_piecewise_force = (
            os.environ.get("WULFRAM_TANK_SOFTBODY_DECOMPILE_FORCE", "0").strip().lower()
            in {"1", "true", "on", "yes"}
        )
        self.tank_softbody_scalar_stretch_source = os.environ.get(
            "WULFRAM_TANK_SOFTBODY_SCALAR_STRETCH_SOURCE",
            "entity_velocity",
        ).strip().lower()
        if self.tank_softbody_scalar_stretch_source in {"0", "false", "off", "no"}:
            self.tank_softbody_scalar_stretch_source = "off"
        elif self.tank_softbody_scalar_stretch_source not in {
            "entity_velocity",
            "velocity",
            "tank_vehicle_impulse",
        }:
            self.tank_softbody_scalar_stretch_source = "entity_velocity"
        try:
            self.tank_softbody_scalar_stretch_denominator = float(
                os.environ.get(
                    "WULFRAM_TANK_SOFTBODY_STRETCH_DENOMINATOR",
                    str(OG_TANK_SPRING_STRETCH_SPEED_DENOMINATOR),
                )
            )
        except ValueError:
            self.tank_softbody_scalar_stretch_denominator = (
                OG_TANK_SPRING_STRETCH_SPEED_DENOMINATOR
            )
        try:
            self.tank_suspension_stiffness = float(os.environ.get("WULFRAM_TANK_SUSPENSION_STIFFNESS", "40.0"))
        except ValueError:
            self.tank_suspension_stiffness = 40.0
        try:
            self.tank_suspension_damping = float(os.environ.get("WULFRAM_TANK_SUSPENSION_DAMPING", "1.5"))
        except ValueError:
            self.tank_suspension_damping = 1.5
        try:
            self.tank_suspension_lift_cap = float(os.environ.get("WULFRAM_TANK_SUSPENSION_LIFT_CAP", "120.0"))
        except ValueError:
            self.tank_suspension_lift_cap = 120.0
        try:
            self.tank_spring_attitude_stiffness = float(os.environ.get(
                "WULFRAM_TANK_SPRING_ATTITUDE_STIFFNESS",
                str(self.tank_suspension_stiffness),
            ))
        except ValueError:
            self.tank_spring_attitude_stiffness = self.tank_suspension_stiffness
        try:
            self.tank_spring_attitude_damping = float(os.environ.get(
                "WULFRAM_TANK_SPRING_ATTITUDE_DAMPING",
                str(VEHICLE_PHYSICS_CONFIGS[EntityType.TANK].angular_damping),
            ))
        except ValueError:
            self.tank_spring_attitude_damping = VEHICLE_PHYSICS_CONFIGS[EntityType.TANK].angular_damping
        self.tank_spring_attitude_model = os.environ.get(
            "WULFRAM_TANK_SPRING_ATTITUDE_MODEL",
            "force",
        ).strip().lower()
        if self.tank_spring_attitude_model not in ("force", "target"):
            self.tank_spring_attitude_model = "force"
        self.tank_spring_attitude_integration = os.environ.get(
            "WULFRAM_TANK_SPRING_ATTITUDE_INTEGRATION",
            "decompile_accel",
        ).strip().lower()
        if self.tank_spring_attitude_integration not in (
            "decompile_accel",
            "decompile_impulse",
            "legacy_accel",
        ):
            self.tank_spring_attitude_integration = "decompile_accel"
        self._load_terrain()
        self._terrain_grid_collision: Optional[TerrainGridCollision] = None
        self._entity_collision_extents_cache: Dict[tuple[int, int], tuple[float, float, float]] = {}
        self._entity_collision_model_cache: Dict[tuple[int, int], Optional[tuple[object, object, float, float]]] = {}
        self._entity_dirty_threshold_sq_cache: Dict[tuple[int, int], float] = {}
        if self.terrain is not None:
            self._terrain_grid_collision = TerrainGridCollision(
                self.terrain,
                self.terrain_physics_height_offset,
                model_contact_normal_source=os.environ.get(
                    "WULFRAM_ENTITY_TERRAIN_MODEL_CONTACT_NORMAL",
                    "mesh",
                ),
            )
            print(
                "[COLLISION] Terrain grid collision initialized "
                f"with {self._terrain_grid_collision.sector_count} sectors "
                f"model_contact_normal={self._terrain_grid_collision.model_contact_normal_source}"
            )
        self._load_map_buildings()
        try:
            self._dynamic_building_next_oid = int(os.environ.get("WULFRAM_DYNAMIC_BUILDING_BASE_OID", "30000"), 0)
        except ValueError:
            self._dynamic_building_next_oid = 30000
        if self._building_entities:
            self._dynamic_building_next_oid = max(
                int(self._dynamic_building_next_oid),
                max(int(oid) for oid in self._building_entities) + 1,
            )
        self.build_uplink_mvp = (
            os.environ.get("WULFRAM_BUILD_UPLINK_MVP", "0").strip().lower()
            not in {"0", "false", "off", "no"}
        )
        world_bound_env = os.environ.get("WULFRAM_WORLD_BOUND")
        try:
            # Clamp world X/Y to the protocol position domain by default (VEC_POS max=8192).
            self.world_bound = abs(float(world_bound_env)) if world_bound_env is not None else 8192.0
        except ValueError:
            self.world_bound = 8192.0
        if self.world_bound < 256.0:
            self.world_bound = 256.0
        try:
            # Higher tick rate improves heading integration accuracy.
            # 30 Hz reduces heading drift from ~14 deg to ~5 deg over 2s turns.
            self.tick_rate_hz = float(os.environ.get("WULFRAM_TICK_RATE_HZ", "30.0"))
        except ValueError:
            self.tick_rate_hz = 30.0
        # Keep steady full-rate updates by default; sparse/on-change updates have
        # shown intermittent HUD red-overlay regressions during active gameplay.
        self.update_on_change = os.environ.get("WULFRAM_UPDATE_ON_CHANGE", "0") == "1"
        try:
            self.update_heartbeat_interval = float(os.environ.get("WULFRAM_UPDATE_HEARTBEAT", "3.0"))
        except ValueError:
            self.update_heartbeat_interval = 3.0
        # Solo-local-player keepalive — feeds the OG client's organic
        # STATE_REQUEST trigger at Replication.c:1173-1177. Emits an
        # UPDATE_ARRAY with exactly one entity (the local player, with
        # pos+rot) at a modest cadence so the `entity_count == 1 && final ==
        # local_player` gate fires regularly.
        #
        # DEFAULT OFF: 2026-04-18 live smoke proved this path silently
        # suppresses the forced-correction burst on the OG client — even
        # with a 2.0s interval well below the 1.0s TimeSeries prune window.
        # Keepalive OFF: main-view changed_ratio 59% on a +80u forced shift.
        # Keepalive ON: 0%. The interaction between the 2Hz non-replay
        # UPDATE_ARRAY pipeline and the correction-burst VIEW_UPDATE is not
        # yet pinned — even after aligning the empirical-correction
        # rotation tuple to the shared body-heading convention. See
        # `docs/keepalive-breaks-correction-2026-04-18.md`. The organic
        # STATE_REQUEST loop the keepalive was meant to feed never
        # re-emerged in traces anyway (OG client stops emitting post-spawn
        # for reasons we haven't fully traced), so leaving this off by
        # default has no regression cost.
        self.solo_local_keepalive_enabled = os.environ.get("WULFRAM_SOLO_LOCAL_KEEPALIVE", "0") == "1"
        try:
            self.solo_local_keepalive_interval = float(os.environ.get("WULFRAM_SOLO_LOCAL_KEEPALIVE_INTERVAL", "2.0"))
        except ValueError:
            self.solo_local_keepalive_interval = 2.0
        try:
            self.update_epsilon = float(os.environ.get("WULFRAM_UPDATE_EPSILON", "0.001"))
        except ValueError:
            self.update_epsilon = 0.001
        tank_vitals_heartbeat_env = os.environ.get("WULFRAM_TANK_VITALS_HEARTBEAT")
        try:
            self.tank_vitals_interval = float(os.environ.get("WULFRAM_TANK_VITALS_INTERVAL", "1.0"))
        except ValueError:
            self.tank_vitals_interval = 1.0
        # Local-player state in UPDATE_ARRAY must match client expectations exactly.
        # Wulf-forge mode sends minimal local stats (weapon_id + health/energy).
        # Modes: off, wf, auto, force
        # Default to wulf-forge style local-state vitals to keep HUD health stable.
        update_local_state_raw = os.environ.get("WULFRAM_UPDATE_LOCAL_STATE", "wf").strip().lower()
        if update_local_state_raw in ("1", "true", "on", "yes", "force"):
            self.update_local_state_mode = "force"
        elif update_local_state_raw in ("0", "false", "off", "no"):
            self.update_local_state_mode = "off"
        elif update_local_state_raw in ("wf", "minimal"):
            # Wulf-forge style: include minimal local stats (weapon_id + health/energy).
            self.update_local_state_mode = "wf"
        else:
            self.update_local_state_mode = "auto"
        # Wulf-forge mode sends minimal local stats (weapon_id=0, health, energy) in UPDATE_ARRAY.
        self.update_local_state = self.update_local_state_mode != "off"
        update_packet_raw = os.environ.get("WULFRAM_UPDATE_PACKET", "").strip().lower()
        if not update_packet_raw:
            update_packet_raw = "update"
        if update_packet_raw == "view":
            print(
                "[CONFIG] WULFRAM_UPDATE_PACKET=view requested, but VIEW_UPDATE "
                "is not the canonical replication stream; forcing UPDATE_ARRAY"
            )
            update_packet_raw = "update"
        if update_packet_raw != "update":
            update_packet_raw = "update"
        self.update_packet_type = update_packet_raw
        # Experimental auxiliary heartbeat/correction path. UPDATE_ARRAY remains
        # the canonical gameplay stream for vitals and entity replication.
        self.heartbeat_view_update = os.environ.get("WULFRAM_HEARTBEAT_VIEW_UPDATE", "0") == "1"
        # Jump jets are a custom extension, not part of the OG Tank controller.
        # Keep them opt-in so default server motion stays clone-focused.
        self.jump_jets_enabled = os.environ.get("WULFRAM_JUMP_JETS", "0") == "1"
        self.jump_jet_correction_burst_count = max(
            0,
            int(os.environ.get("WULFRAM_JUMP_JET_CORRECTION_BURST", "12")),
        )
        self.jump_jet_correction_burst_interval = max(
            0.01,
            float(os.environ.get("WULFRAM_JUMP_JET_CORRECTION_INTERVAL", "0.05")),
        )

        print(
            "[CONFIG] spawn_udp_tank="
            f"{int(self.spawn_send_udp_tank)} player_info="
            f"{('explicit-' + str(int(self.spawn_send_player_info))) if self.spawn_send_player_info_explicit else 'auto-remote'} "
            f"game_clock={int(self.spawn_send_game_clock)} comm_message={int(self.spawn_send_comm_message)} "
            f"reincarnate={int(self.spawn_send_reincarnate)} "
            f"entry_transition={self.spawn_entry_transition} "
            f"birth_notice={int(self.spawn_send_birth_notice)} update_array={int(self.spawn_send_update_array)} "
            f"player_packet={int(self.spawn_send_player_packet)} proj_spawn_snap={int(self.projectile_spawn_snap)} "
            f"map={self.map_name} grid={self.map_grid_rows}x{self.map_grid_cols} scale={self.map_scale} "
            f"allow_large_grid={int(self.map_allow_large_grid)} "
            f"spawn_height={self.spawn_height:.1f} ground_level={self.ground_level:.1f} "
            f"world_bound={self.world_bound:.1f} "
            f"spawn_set_ground={int(self.spawn_sets_ground_level)} "
            f"spawn_on_team_select={int(self.spawn_on_team_select)} "
            f"spawn_point_override={int(self.spawn_allow_point_override)} "
            f"team_switch_reincarnate={int(self.team_switch_send_reincarnate)} "
            f"team_switch_roster={int(self.team_switch_send_roster)} "
            f"team_switch_stats={int(self.team_switch_send_update_stats)}:"
            f"{self.team_switch_update_stats_transport}/{self.team_switch_update_stats_variant} "
            f"gravity={self.gravity:.1f} tick_hz={self.tick_rate_hz:.1f} "
            f"update_on_change={int(self.update_on_change)} heartbeat={self.update_heartbeat_interval:.2f}s "
            f"map_spawns={int(self.use_map_spawn_points)} "
            f"spawn_align_terrain={int(self.align_spawn_points_to_terrain)} "
            f"spawn_pos_align_terrain={int(self.align_spawn_pos_to_terrain)} "
            f"update_packet={self.update_packet_type} "
            f"force_default_spawn={int(self.force_default_spawn_pos)} "
            f"default_spawn={self.default_flat_spawn_pos} "
            f"heartbeat_view={int(self.heartbeat_view_update)} jump_jets={int(self.jump_jets_enabled)} "
            f"jump_corr={self.jump_jet_correction_burst_count}@{self.jump_jet_correction_burst_interval:.2f}s "
            f"terrain_collision_override={int(self.terrain_collision_with_ground_override)} "
            f"inactivity_timeout={self.inactivity_timeout:.1f}s"
        )
        if self.spawn_send_player_info_explicit and not self.spawn_send_player_info:
            print(
                "[CONFIG-WARN] WULFRAM_SPAWN_PLAYER_INFO=0 disables canonical "
                "remote OG local-player initialization; manual flag spawns can "
                "leave camera/sync_state unset and corrections invisible."
            )
        if self.spawn_send_update_array:
            print(
                "[CONFIG-WARN] WULFRAM_SPAWN_UPDATE_ARRAY=1 is experimental for "
                "remote OG correction: current probes require a live local "
                "sync/server_expected sink, and pre-create sessions can strand "
                "that bootstrap."
            )
        if self.spawn_send_player_info and not self.spawn_send_update_array:
            print(
                "[CONFIG-WARN] PLAYER_INFO without spawn UPDATE_ARRAY can leave "
                "the OG entity without network sync state; targeted corrections "
                "may stay invisible."
            )

        # PLAYER_INFO triggers local-state sync on the client. For remote/OG
        # clients, default to canonical local-state so hull/fuel/ammo/turret
        # state initialize through the decompile-backed path. Only the spawn
        # TankPacket keeps the short-form-safe weapon id because 0x18 does not
        # carry the extra ammo/turret bits.
        player_info_local_state_env = os.environ.get("WULFRAM_PLAYER_INFO_LOCAL_STATE")
        self.player_info_local_state_explicit = player_info_local_state_env is not None
        player_info_local_state_raw = (
            player_info_local_state_env.strip().lower()
            if player_info_local_state_env is not None
            else "auto-remote"
        )
        if player_info_local_state_raw in ("1", "true", "on", "yes", "force"):
            self.player_info_local_state_mode = "force"
        elif player_info_local_state_raw in ("0", "false", "off", "no"):
            self.player_info_local_state_mode = "off"
        elif player_info_local_state_raw in ("remote", "auto-remote"):
            self.player_info_local_state_mode = "auto-remote"
        else:
            self.player_info_local_state_mode = "auto"
        self.player_info_local_state = self.player_info_local_state_mode != "off"
        # Wulf-forge does NOT send PLAYER(spectator=0) during spawn.
        self.spawn_send_player_active = os.environ.get("WULFRAM_SPAWN_PLAYER_ACTIVE", "0") == "1"
        # Local-state weapon type (entity type index). Default is 0 (Tank).
        # Tank (0): pool_entry[2]=9 â†’ 9 ammo bits + weapon_def+0x170=1 â†’ 16 turret bits.
        #   Required for weapons: sync_local_player calls update_active_flags every heartbeat;
        #   without 9-bit ammo bitmask (all 1s), active flags reset to 0 â†’ fire check fails.
        # Scout (1): weapon_def+0x68=1 â†’ secondary turret bits (untested).
        # AssaultPlatform (2): pool_entry[2]=0 â†’ no ammo bits â†’ weapons can never fire.
        try:
            self.local_state_weapon_type = int(os.environ.get("WULFRAM_LOCAL_STATE_WEAPON_TYPE", "0"))
        except ValueError:
            self.local_state_weapon_type = 0
        # TankPacket spawn vitals are encoded as the short local-state form
        # (weapon + health + energy only). Keep a safe default weapon type here
        # so the client does not expect ammo/turret bits during 0x18 spawn parse.
        try:
            self.spawn_tank_weapon_type = int(os.environ.get("WULFRAM_SPAWN_TANK_WEAPON_TYPE", "2"))
        except ValueError:
            self.spawn_tank_weapon_type = 2
        if self.spawn_tank_weapon_type < 0 or self.spawn_tank_weapon_type > 31:
            self.spawn_tank_weapon_type = 2
        # In WF local-state mode, include turret angles to match client's
        # read_local_player_state expectations.  Tank (weapon_type=0) requires
        # primary turret (weapon_def+0x170=1).  Omitting these bits causes a
        # bitstream desync: the client reads 16 turret bits from the entity
        # section, garbling entity_count and preventing health application.
        self.wf_local_state_turrets = os.environ.get("WULFRAM_WF_LOCAL_TURRETS", "0") == "1"
        # Local-state ammo bitmask parameters (active slot flags). Defaults match our BEHAVIOR config (no active flags).
        try:
            self.local_state_ammo_bits = int(os.environ.get("WULFRAM_LOCAL_STATE_AMMO_BITS", "0"))
        except ValueError:
            self.local_state_ammo_bits = 0
        try:
            self.local_state_ammo_mask = int(os.environ.get("WULFRAM_LOCAL_STATE_AMMO_MASK", "0"))
        except ValueError:
            self.local_state_ammo_mask = 0
        # Local-state ammo bit width should follow BEHAVIOR-derived slot capabilities.
        self.local_state_ammo_from_behavior = os.environ.get("WULFRAM_LOCAL_STATE_AMMO_FROM_BEHAVIOR", "1") == "1"
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
        self.allow_unsafe_local_state = os.environ.get("WULFRAM_ALLOW_UNSAFE_LOCAL_STATE", "0") == "1"

        # Guard against unsafe local-state configs (missing turret bits or unknown weapon type).
        # Local-state only stays enabled when config is known safe or explicitly overridden.
        local_state_safe = False
        if self.local_state_weapon_type == 0:
            local_state_safe = self.local_state_turret_bits > 0 and (
                self.local_state_primary_override not in ("0", "false", "False")
            )
        elif self.local_state_weapon_type == 1:
            local_state_safe = self.local_state_turret_bits > 0 and (
                self.local_state_secondary_override not in ("0", "false", "False")
            )
        # wf mode now sends full local-state (weapon + health + energy + ammo + turret)
        # to enable weapon firing. Keep strict safety checks for auto/force modes.
        if self.update_local_state_mode not in ("off", "wf") and not self.allow_unsafe_local_state and not local_state_safe:
            print(
                "[LOCAL-STATE] Unsafe local-state config; disabling local-state. "
                "Set WULFRAM_ALLOW_UNSAFE_LOCAL_STATE=1 to override."
            )
            self.update_local_state_mode = "off"
            self.update_local_state = False

        # TankPacket vitals heartbeat: wulf-forge NEVER sends repeated 0x18 packets.
        # Each 0x18 triggers Entity_create_from_network or LocalPlayer_initialize,
        # which can destabilize the client. Default OFF to match wulf-forge.
        if tank_vitals_heartbeat_env is None:
            self.tank_vitals_heartbeat = False
        else:
            self.tank_vitals_heartbeat = tank_vitals_heartbeat_env == "1"

        # Keep auxiliary VIEW_UPDATE off unless explicitly enabled.

        # Derive per-weapon capability counts from the BEHAVIOR packet.
        # Used to size local-state ammo/active-slot bitmasks correctly.
        self.behavior_weapon_caps = get_behavior_weapon_capability_counts()
        if not self.behavior_weapon_caps:
            self.behavior_weapon_caps = [(0, 0, 0, 0)] * 4
        if self.update_local_state_mode != "off":
            print(f"[LOCAL-STATE] Weapon capability counts (ammo/fire/active/cooldown): {self.behavior_weapon_caps}")
        if not self.tank_vitals:
            print("[WARN] WULFRAM_TANK_VITALS=0 can cause red health overlay after ~5-10s.")
        print(
            "[CONFIG] update_local_state="
            f"{self.update_local_state_mode} player_info_local_state={self.player_info_local_state_mode} "
            f"tank_vitals={int(self.tank_vitals)} local_state_weapon_type={self.local_state_weapon_type} "
            f"spawn_tank_weapon_type={self.spawn_tank_weapon_type} "
            f"ammo_from_behavior={int(self.local_state_ammo_from_behavior)} "
            f"vitals_heartbeat={int(self.tank_vitals_heartbeat)} wf_local_turrets={int(self.wf_local_state_turrets)} "
            f"local_update_mode={self.local_update_mode} "
            f"remote_update_mode={self.remote_update_mode} remote_interval={self.remote_update_interval:.2f}s "
            f"combine_updates={int(self.combine_update_arrays)}"
        )
        print(
            "[CONFIG] view_update="
            f"{int(self.view_update_enabled)} loop={int(self.view_update_loop)} "
            f"local_stats={int(self.view_update_local_stats)} "
            f"entity_vitals={int(self.view_update_entity_vitals)} "
            f"interval={self.view_update_interval:.2f}s"
        )
        print(f"[CONFIG] update_grace={self.update_grace_seconds:.2f}s require_client_tick={int(self.require_client_tick)}")
        # Tick selection: default to server tick for stability across multiple clients.
        self.use_client_ticks = os.environ.get("WULFRAM_USE_CLIENT_TICKS", "0") == "1"
        print(f"[CONFIG] use_client_ticks={int(self.use_client_ticks)}")
        try:
            self.remote_og_movement_input_delay = float(
                os.environ.get("WULFRAM_REMOTE_OG_MOVEMENT_INPUT_DELAY", "0.20")
            )
        except ValueError:
            self.remote_og_movement_input_delay = 0.20
        if self.remote_og_movement_input_delay < 0.0:
            self.remote_og_movement_input_delay = 0.0
        print(f"[CONFIG] remote_og_movement_input_delay={self.remote_og_movement_input_delay:.2f}s")
        # Aim/movement configuration (shared across clients)
        # Slot-integrated aim is sensitive to noisy axis samples; keep opt-in.
        self.use_slot_aim = os.environ.get("WULFRAM_USE_SLOT_AIM", "0") == "1"
        print(
            "[CONFIG] projectiles_enabled="
            f"{int(self.projectiles_enabled)} projectile_update_mode={self.projectile_update_mode} "
            f"remote_projectiles={int(self.remote_projectiles)} "
            f"remote_combat_observer_packets={int(self.remote_combat_observer_packets)} "
            f"projectile_local_stats={int(self.projectile_local_stats)} "
            f"projectile_spawn_snap={int(self.projectile_spawn_snap)} "
            f"projectile_aim_source={self.projectile_aim_source} "
            f"use_slot_aim={int(self.use_slot_aim)}"
        )
        print(
            "[CONFIG] energy "
            f"weapon_enabled={int(self.weapon_energy_enabled)} "
            f"max={self.player_energy_max:.1f} regen={self.player_energy_regen:.1f}/s "
            f"debug_viewpoint={int(self.debug_viewpoint)} debug_udp_raw={int(self.debug_udp_raw)}"
        )
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
        # (subtick/ticksync/client_frame_physics removed â€” dead scaffolding, never used in tick loop)
        # float32 precision matching: quantize heading/ang_vel to float32 after each step
        self.f32_physics = os.environ.get("WULFRAM_F32_PHYSICS", "1") == "1"
        # (frame_locked mode removed â€” not part of original decompile architecture)
        try:
            self.turn_sign = float(os.environ.get("WULFRAM_TURN_SIGN", "-1.0"))
        except ValueError:
            self.turn_sign = -1.0
        try:
            self.strafe_sign = float(os.environ.get("WULFRAM_STRAFE_SIGN", "-1.0"))
        except ValueError:
            self.strafe_sign = -1.0
        # (yaw_input_compensation removed â€” dead scaffolding, never used in tick loop)
        # Steering curve: piecewise-linear dead-zone curve from client
        # (azurefishy-src Vehicles.c:1030 Piecewise_interpolate).
        # 10 samples over domain 0.0-1.0, looked up with abs(input).
        # Default samples match the client's default curve.
        default_curve = "0.0,0.005,0.01,0.15,0.25,0.4,0.55,0.7,0.85,1.0"
        curve_str = os.environ.get("WULFRAM_TURN_CURVE", default_curve)
        try:
            self.turn_curve_samples = [float(s) for s in curve_str.split(",")]
        except ValueError:
            self.turn_curve_samples = [float(s) for s in default_curve.split(",")]
        if len(self.turn_curve_samples) < 2:
            self.turn_curve_samples = [float(s) for s in default_curve.split(",")]
        # Send server rotation in heartbeat UPDATE_ARRAY to prevent client angular velocity zeroing.
        # Default ON - entity entries with mask=0 (no rotation) cause client to zero angular velocity.
        self.heartbeat_include_rot = os.environ.get("WULFRAM_HEARTBEAT_ROT", "1") == "1"
        self.heartbeat_include_pos = os.environ.get("WULFRAM_HEARTBEAT_POS", "0") == "1"
        # DEBUG_SYNC: send authoritative physics state to client after each step
        # for measuring client-server divergence (opcode 0x60, 49 bytes, ~12Hz).
        # This is a custom debug packet and must never leak to OG clients unless
        # explicitly allowed.
        self.debug_sync = os.environ.get("WULFRAM_DEBUG_SYNC", "0") == "1"
        debug_sync_hosts = os.environ.get("WULFRAM_DEBUG_SYNC_HOSTS", "127.0.0.1,::1").strip()
        self.debug_sync_allow_all = debug_sync_hosts.lower() in ("*", "all", "any")
        self.debug_sync_hosts = {
            self._normalize_debug_host(entry)
            for entry in debug_sync_hosts.split(",")
            if entry.strip()
        }
        self._debug_sync_blocked_clients: set[int] = set()
        # Keep the old server-originated ping loop opt-in only. The OG client
        # already emits its own ping/state timing packets, and extra server
        # pings have proven to be destabilizing during compatibility testing.
        self.server_ping_loop_enabled = os.environ.get("WULFRAM_SERVER_PING_LOOP", "0") == "1"
        # UDP 0x0B -> 0x0C ping replies share the client's timing opcode
        # neighborhood with STATE_REQUEST. Keep replies available for local
        # tools, but make remote OG timing traffic opt-in so correction tests
        # cannot be perturbed by server-originated ping echoes.
        udp_ping_reply_hosts = os.environ.get(
            "WULFRAM_UDP_PING_REPLY_HOSTS",
            "127.0.0.1,::1,loopback",
        ).strip()
        self.udp_ping_reply_allow_all = udp_ping_reply_hosts.lower() in ("*", "all", "any")
        self.udp_ping_reply_hosts = {
            self._normalize_debug_host(entry)
            for entry in udp_ping_reply_hosts.split(",")
            if entry.strip()
        }
        self._udp_ping_reply_blocked_clients: set[int] = set()
        # The decompile-backed STATE_REQUEST path is canonical for targeted
        # local sync. UPDATE_ARRAY remains the primary gameplay stream, but
        # VIEW_UPDATE has a dedicated replay/correction handler on the client,
        # so allow a targeted overlay on this path.
        state_sync_hosts = os.environ.get("WULFRAM_STATE_SYNC_REPLY_HOSTS", "*").strip()
        self.state_sync_reply_allow_all = state_sync_hosts.lower() in ("*", "all", "any")
        self.state_sync_reply_hosts = {
            self._normalize_debug_host(entry)
            for entry in state_sync_hosts.split(",")
            if entry.strip()
        }
        self._state_sync_blocked_clients: set[int] = set()
        self.state_sync_view_mode = os.environ.get(
            "WULFRAM_STATE_SYNC_VIEW_MODE",
            "all",
        ).strip().lower()
        if self.state_sync_view_mode not in ("off", "loopback", "remote", "all"):
            self.state_sync_view_mode = "all"
        self.state_sync_snapshot_mode = os.environ.get(
            "WULFRAM_STATE_SYNC_SNAPSHOT_MODE",
            "remote_live",
        ).strip().lower()
        if self.state_sync_snapshot_mode not in ("remote_live", "live", "history"):
            self.state_sync_snapshot_mode = "remote_live"
        try:
            self.state_sync_correction_burst_count = int(
                os.environ.get("WULFRAM_STATE_SYNC_CORRECTION_BURST", "6")
            )
        except ValueError:
            self.state_sync_correction_burst_count = 6
        if self.state_sync_correction_burst_count < 0:
            self.state_sync_correction_burst_count = 0
        try:
            self.state_sync_correction_burst_interval = float(
                os.environ.get("WULFRAM_STATE_SYNC_CORRECTION_INTERVAL", "0.10")
            )
        except ValueError:
            self.state_sync_correction_burst_interval = 0.10
        if self.state_sync_correction_burst_interval < 0.0:
            self.state_sync_correction_burst_interval = 0.0
        try:
            self.remote_view_update_timestamp_ahead_ms = int(
                os.environ.get("WULFRAM_REMOTE_VIEW_TIMESTAMP_AHEAD_MS", "1000")
            )
        except ValueError:
            self.remote_view_update_timestamp_ahead_ms = 1000
        if self.remote_view_update_timestamp_ahead_ms < 0:
            self.remote_view_update_timestamp_ahead_ms = 0
        # Non-canonical debug/status chat is useful for the Python client, but
        # it should not leak to OG clients while protocol compatibility work is
        # in flight.
        debug_comm_hosts = os.environ.get("WULFRAM_DEBUG_COMM_MESSAGE_HOSTS", "127.0.0.1,::1").strip()
        self.debug_comm_allow_all = debug_comm_hosts.lower() in ("*", "all", "any")
        self.debug_comm_hosts = {
            self._normalize_debug_host(entry)
            for entry in debug_comm_hosts.split(",")
            if entry.strip()
        }
        self._debug_comm_blocked_clients: set[int] = set()
        try:
            self.remote_full_local_state_delay = float(
                os.environ.get("WULFRAM_REMOTE_FULL_LOCAL_STATE_DELAY", "0.75")
            )
        except ValueError:
            self.remote_full_local_state_delay = 0.75
        print(
            "[CONFIG] server_ping_loop="
            f"{int(self.server_ping_loop_enabled)} "
            f"udp_ping_reply_hosts="
            f"{'*' if self.udp_ping_reply_allow_all else ','.join(sorted(self.udp_ping_reply_hosts))} "
            f"state_sync_reply_hosts="
            f"{'*' if self.state_sync_reply_allow_all else ','.join(sorted(self.state_sync_reply_hosts))} "
            f"state_sync_view_mode={self.state_sync_view_mode} "
            f"state_sync_snapshot_mode={self.state_sync_snapshot_mode} "
            f"state_sync_correction_burst={self.state_sync_correction_burst_count}@"
            f"{self.state_sync_correction_burst_interval}s "
            f"remote_view_ts_ahead={self.remote_view_update_timestamp_ahead_ms}ms "
            f"debug_comm_hosts="
            f"{'*' if self.debug_comm_allow_all else ','.join(sorted(self.debug_comm_hosts))}"
        )
        try:
            self.aim_hold_time = float(os.environ.get("WULFRAM_AIM_HOLD", "0.4"))
        except ValueError:
            self.aim_hold_time = 0.4

        # Angular velocity damping coefficient.
        # From Physics_substep_integrate: ang_vel += (torque - ang_vel * damp_coeff) * dt
        # Steady state: ang_vel = torque / damp_coeff = turn_adjust / damp_coeff
        # Binary source: DAT_005730cc+0x1C = 2.0 (Tank entity descriptor, BoundsInfo+0x7C)
        try:
            self.damp_coeff = float(os.environ.get("WULFRAM_DAMP_COEFF", "2.0"))
        except ValueError:
            self.damp_coeff = 2.0

        # Linear velocity damping coefficients.
        # From RigidBody_integrate_position (Physics.c:5120-5134, damped mode):
        #   effective_acc = impulse - vel * linear_damp
        #   pos += vel * dt + 0.5 * effective_acc * dtÂ²
        #   vel += effective_acc * dt
        # Entity velocity at 0x18 is PERSISTENT (not zeroed each frame).
        # Impulse at 0x24 IS zeroed each frame (from vehicle controller).
        #
        # Live OG memory probe, grounded in Physics.c:6036-6073:
        #   entity+0xC0+3 = 1 (linear damping enabled)
        #   entity+0xBC->+4+0x78 = 1.5 (Tank physics config linear damping)
        # Keep separate env knobs for experiments, but default both drive and
        # coast paths to the measured physics-config coefficient.
        try:
            self.linear_damp_driving = float(os.environ.get("WULFRAM_LINEAR_DAMP_DRIVING", "1.5"))
        except ValueError:
            self.linear_damp_driving = 1.5
        try:
            self.linear_damp_coasting = float(os.environ.get("WULFRAM_LINEAR_DAMP_COASTING", "1.5"))
        except ValueError:
            self.linear_damp_coasting = 1.5
        try:
            self.tank_ground_contact_damp = float(
                os.environ.get("WULFRAM_TANK_GROUND_CONTACT_DAMP", "6.0")
            )
        except ValueError:
            self.tank_ground_contact_damp = 6.0
        if self.tank_ground_contact_damp < 0.0:
            self.tank_ground_contact_damp = 0.0

        # Local-player correction mode. UPDATE_ARRAY-based modes
        # (`dual_entity`, `full`, `pos_only`, `rot_only`) do NOT
        # visually reconcile the local player on the OG client: the
        # non-replay UPDATE_ARRAY path is treated as a heartbeat, not
        # a lag-compensation correction, so the client's lockstep
        # physics overwrites the server state on the very next frame.
        # Older audit notes flagged this as "corrections are
        # architecturally impossible", but the 2026-04-18 forced-
        # correction probe landed at 51-65% main-view pixel change
        # using VIEW_UPDATE (opcode 0x0F): OG's VIEW_UPDATE handler
        # routes through `apply_lag_compensation` which DOES write
        # pos/rot directly and zero velocity for a smooth correction.
        #
        # So the default correction mode is `view_update`. The other
        # modes stay available as env overrides for A/B investigation
        # but should not be the default — they were the culprit
        # behind the "corrections not applying" symptom (the forced-
        # correction probe explicitly flips mode while running, but
        # everything else hits the non-functional UPDATE_ARRAY path).
        # See docs/view-update-correction.md.
        try:
            self.correction_interval = float(os.environ.get("WULFRAM_CORRECTION_INTERVAL", "0"))
        except ValueError:
            self.correction_interval = 0.0
        try:
            # Remote OG still needs fresh VIEW_UPDATE wrappers after movement
            # settles, but live traces show position-bearing replay updates
            # reset the local physics path while forward/strafe input is held.
            self.movement_correction_interval = float(
                os.environ.get("WULFRAM_MOVEMENT_CORRECTION_INTERVAL", "0.10")
            )
        except ValueError:
            self.movement_correction_interval = 0.10
        try:
            self.movement_correction_window = float(
                os.environ.get("WULFRAM_MOVEMENT_CORRECTION_WINDOW", "4.0")
            )
        except ValueError:
            self.movement_correction_window = 4.0
        try:
            self.active_input_correction_suppress_window = float(
                os.environ.get("WULFRAM_ACTIVE_INPUT_CORRECTION_SUPPRESS_WINDOW", "1.25")
            )
        except ValueError:
            self.active_input_correction_suppress_window = 1.25
        if self.active_input_correction_suppress_window < 0.0:
            self.active_input_correction_suppress_window = 0.0
        self.correction_mode = os.environ.get("WULFRAM_CORRECTION_MODE", "view_update").strip().lower()
        if self.correction_mode not in ("full", "rot_only", "pos_only", "dual_entity", "view_update", "view_update_define"):
            self.correction_mode = "view_update"

        print(
            f"[CONFIG-HEADING] turn_adjust={self.turn_adjust} turn_sign={self.turn_sign} "
            f"deadzone={self.turn_deadzone} damp_coeff={self.damp_coeff} "
            f"linear_damp=driving:{self.linear_damp_driving}/coast:{self.linear_damp_coasting} "
            f"tank_ground_contact_damp={self.tank_ground_contact_damp} "
            f"tick_rate={self.tick_rate_hz}Hz correction_interval={self.correction_interval}s "
            f"movement_correction={self.movement_correction_interval}s/"
            f"{self.movement_correction_window}s "
            f"active_suppress={self.active_input_correction_suppress_window}s "
            f"correction_mode={self.correction_mode}"
        )

        self.estimated_speed = 15.0  # Units per second (tunable)
        # Packet traffic logger for debugging freezes
        self.pktlog = PacketLog()
        # Ghost rejoin: auto-create session from orphan UDP (e.g., VM snapshot restore)
        self.ghost_rejoin = os.environ.get("WULFRAM_GHOST_REJOIN", "1") == "1"
        self._ghost_rejoin_attempted: set = set()  # Track addrs we already tried

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
        if self.use_client_ticks and ctx.tick_offset is not None:
            tick = tick + ctx.tick_offset
            # Only clamp to client tick when we're in client-tick mode;
            # otherwise the server-domain tick is much smaller than the
            # client's GetTickCount() and clamping causes a massive jump
            # that breaks remote entity interpolation.
            if ctx.last_client_tick and tick < ctx.last_client_tick:
                tick = ctx.last_client_tick
        if ctx.last_sent_tick and tick <= ctx.last_sent_tick:
            tick = ctx.last_sent_tick + 1
        ctx.last_sent_tick = tick & 0xFFFFFFFF
        return ctx.last_sent_tick

    def _fresh_remote_view_update_timestamp(self, ctx: ClientContext, tick: int) -> int:
        """Return a remote OG VIEW_UPDATE timestamp that the client will clamp fresh.

        The OG replay handler clamps future VIEW_UPDATE timestamps down to its
        current tick before storing interp_record+0x08. Live wulftap shows stale
        STATE_REQUEST ids are rejected by UpdateArray_check_eligible before
        NetworkSync_receive_entity_update can store server_expected, so remote
        OG replay wrappers must be current/future while snapshot selection can
        still use the original request id.
        """
        base_tick = int(tick) & 0xFFFFFFFF
        client_tick = int(getattr(ctx, "last_client_tick", 0) or 0) & 0xFFFFFFFF
        if client_tick and self._tick_delta_signed(client_tick, base_tick) > 0:
            base_tick = client_tick
        ahead_ms = int(getattr(self, "remote_view_update_timestamp_ahead_ms", 1000) or 0)
        if ahead_ms < 0:
            ahead_ms = 0
        return (base_tick + ahead_ms) & 0xFFFFFFFF

    @staticmethod
    def _tick_delta_signed(newer: int, older: int) -> int:
        """Return the signed 32-bit tick delta `newer - older`."""
        delta = (int(newer) - int(older)) & 0xFFFFFFFF
        if delta & 0x80000000:
            delta -= 0x100000000
        return delta

    def _record_authoritative_state(self, ctx: ClientContext, *, tick: int) -> None:
        """Cache recent authoritative states for replay-aligned sync replies."""
        history = getattr(ctx, "authoritative_state_history", None)
        if history is None:
            return
        entry = {
            "tick": tick & 0xFFFFFFFF,
            "time": time.monotonic(),
            "pos": self._to_client_pos(ctx.player_pos),
            "vel": tuple(ctx.player_vel),
            "rot": self._local_player_sync_rotation(ctx),
        }
        if history and history[-1]["tick"] == entry["tick"]:
            history[-1] = entry
        else:
            history.append(entry)

    def _select_authoritative_state_snapshot(
        self,
        ctx: ClientContext,
        replay_timestamp: Optional[int],
    ) -> Optional[dict]:
        """Pick the cached authoritative state closest to a replay request tick."""
        if replay_timestamp is None:
            return None
        history = getattr(ctx, "authoritative_state_history", None)
        if not history:
            return None

        # STATE_REQUEST.request_id stays in the client's GetTickCount domain even
        # when normal replication uses server-domain ticks. Replay-aligned sync
        # still needs to pick the authoritative sample nearest that request, so
        # compare against both domains when a stable tick offset is available.
        candidate_ticks = [int(replay_timestamp) & 0xFFFFFFFF]
        if not getattr(self, "use_client_ticks", False):
            tick_offset = getattr(ctx, "tick_offset", None)
            if tick_offset is not None:
                mapped_tick = (candidate_ticks[0] - int(tick_offset)) & 0xFFFFFFFF
                if mapped_tick != candidate_ticks[0]:
                    candidate_ticks.append(mapped_tick)

        best_prior = None
        best_prior_delta = None
        best_abs = None
        best_abs_delta = None

        for entry in history:
            for target_tick in candidate_ticks:
                delta = self._tick_delta_signed(target_tick, entry["tick"])
                abs_delta = abs(delta)
                if delta >= 0 and (
                    best_prior is None or best_prior_delta is None or delta < best_prior_delta
                ):
                    best_prior = entry
                    best_prior_delta = delta
                if best_abs is None or best_abs_delta is None or abs_delta < best_abs_delta:
                    best_abs = entry
                    best_abs_delta = abs_delta

        max_replay_delta = 250
        if best_prior is not None and best_prior_delta is not None and best_prior_delta <= max_replay_delta:
            return best_prior
        if best_abs is not None and best_abs_delta is not None and best_abs_delta <= max_replay_delta:
            return best_abs
        if best_abs is not None and not handlers._is_loopback_client(ctx):
            # OG prediction is more likely to reject a reply that pairs an old
            # replay timestamp with the live/current pose than a coherent cached
            # authoritative sample. Keep remote replies on history even when the
            # request arrived outside our tight replay window.
            return best_abs
        return None

    def _get_local_state_weapon_type(self, ctx: ClientContext) -> int:
        """Return weapon type index used by local player state (entity type index, not weapon slot)."""
        if self.update_local_state_mode == "wf":
            return self.local_state_weapon_type or 0
        if self.local_state_weapon_type:
            return self.local_state_weapon_type
        if getattr(ctx, "entity_type", None) is not None:
            return int(ctx.entity_type)
        return 0

    def _get_spawn_tank_weapon_type(self, ctx: Optional[ClientContext] = None) -> int:
        """Return weapon id used for spawn TankPacket vitals (short local-state parse-safe)."""
        weapon = self.spawn_tank_weapon_type
        if 0 <= weapon <= 31:
            return weapon
        if ctx is not None:
            return self._get_local_state_weapon_type(ctx)
        return 2

    def _wf_minimal_local_state_for_client(self, ctx: ClientContext) -> bool:
        """Return True while a remote OG client still needs the short-form-safe path.

        Spawn-time `TANK` / `PLAYER_INFO` / first heartbeat are still sensitive
        to local-state bit-count mismatches. Keep remote clients on the safe
        short form until they have explicitly requested targeted sync. Promotion
        after that point controls the broader post-spawn sync/heartbeat path,
        but targeted correction packets still keep the safe local-state prefix.
        """
        if self.update_local_state_mode != "wf":
            return False
        if handlers._is_loopback_client(ctx):
            return False
        return not bool(getattr(ctx, "remote_full_local_state_ready", False))

    def _wf_remote_heartbeat_entity_mode(self, ctx: ClientContext) -> bool:
        """Return True when remote `wf` heartbeats should use local-player sync updates.

        Remote OG clients need the short-form spawn-safe path first, then the
        promoted single-local-player update transport once targeted sync is
        active. That promotion no longer implies the expanded ammo/turret
        local-state shape on targeted correction packets.
        """
        if self.update_local_state_mode != "wf":
            return False
        if handlers._is_loopback_client(ctx):
            return False
        return bool(getattr(ctx, "remote_full_local_state_ready", False))

    def _suppress_remote_spawn_safe_heartbeat(self, ctx: ClientContext) -> bool:
        """Suppress periodic remote heartbeats until the OG client leaves spawn-safe mode.

        The short-form minimal heartbeat exists for spawn/bootstrap safety, but
        it is also the narrowest remaining suspect when OG falls back to the
        address screen immediately after join. Keep the one-off spawn heartbeat,
        but do not stream periodic short-form heartbeats before targeted sync
        has promoted the client into the fuller post-spawn update transport.
        """
        return self._wf_minimal_local_state_for_client(ctx)

    def _suppress_remote_spawn_bootstrap_heartbeat(self, ctx: ClientContext) -> bool:
        """Suppress the one-off post-spawn heartbeat while remote OG stays on the minimal path.

        The 10-byte heartbeat is safe enough for steady loopback probes, but on
        the remote OG bootstrap path it still lands in the narrow packet window
        where protocol-mismatch falls back to the address screen. Once the
        client explicitly requests targeted sync we can leave spawn-safe mode
        and rely on the promoted post-spawn transport instead.
        """
        return self._wf_minimal_local_state_for_client(ctx)

    def _resume_remote_full_local_state_after_spawn(
        self,
        ctx: ClientContext,
        *,
        entity_id: int,
        health: float = 1.0,
        fuel: float = 1.0,
        previously_promoted: bool = False,
    ) -> bool:
        """Restore promoted remote local-state sync after a respawn-safe spawn sequence.

        Keep the initial remote spawn packets on the minimal safe path, but do
        not strand a previously-promoted OG client there after a respawn. Once
        the fresh entity exists again, resume the promoted heartbeat shape and
        send one immediate full heartbeat so the client's local timing/correction
        path has authoritative state to lock onto again.
        """
        if not previously_promoted:
            return False
        if self.update_local_state_mode != "wf":
            return False
        if handlers._is_loopback_client(ctx):
            return False

        ctx.remote_full_local_state_ready = True
        ctx._spawn_safe_heartbeat_suppressed_logged = False
        ctx.last_state_sync_send = 0.0

        if not self.udp_handler or not ctx.session or not ctx.session.udp_addr:
            return True

        try:
            hb_tick = self._get_network_tick(ctx)
            hb_packet = self._build_local_state_heartbeat(
                ctx,
                tick=hb_tick,
                entity_id=entity_id,
                include_health=True,
                health=health,
                fuel=fuel,
            )
            self.udp_handler.send_to(hb_packet, ctx.session.udp_addr)
            print(
                f"[SPAWN] Client {ctx.client_id}: restored promoted remote sync "
                "path and sent immediate full heartbeat"
            )
        except Exception as ex:
            print(
                f"[SPAWN] WARNING: Failed to restore promoted remote sync path "
                f"for client {ctx.client_id}: {ex}"
            )
        return True

    def _get_local_state_ammo_bits(self, ctx: ClientContext, *, force_full_remote: bool = False) -> tuple:
        """Return (ammo_bits, ammo_mask) for local player state."""
        if self._wf_minimal_local_state_for_client(ctx) and not force_full_remote:
            return 0, 0
        if self.local_state_ammo_override:
            return self.local_state_ammo_bits, self.local_state_ammo_mask

        if not self.local_state_ammo_from_behavior:
            return 0, 0

        weapon_type = self._get_local_state_weapon_type(ctx)
        active_bits = 0
        if 0 <= weapon_type < len(self.behavior_weapon_caps):
            active_bits = self.behavior_weapon_caps[weapon_type][2]
        # Send 0 mask (no weapons actively firing).  The active-flags bitmask
        # controls AmmoSlotState+0x05 via update_active_flags(); bit=1 causes
        # the client's WeaponCooldown_update_all() to auto-fire that slot.
        # Only set bits when the player is actually firing (TODO).
        active_mask = 0
        return active_bits, active_mask

    def _get_local_state_turret_bits(self, ctx: ClientContext, *, force_full_remote: bool = False) -> tuple:
        """
        Return (primary_bits, primary_angle, secondary_bits, secondary_angle).
        Flags are inferred from entity_type unless overrides are provided.
        """
        if self._wf_minimal_local_state_for_client(ctx) and not force_full_remote:
            return 0, 0.0, 0, 0.0
        weapon_type = self._get_local_state_weapon_type(ctx)

        # Default turret bits based on WeaponDef_init_by_entity_type (azurefishy decomp).
        primary_flag = weapon_type in LOCAL_STATE_PRIMARY_TURRET_WEAPON_TYPES
        secondary_flag = weapon_type in LOCAL_STATE_SECONDARY_TURRET_WEAPON_TYPES

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

    def _get_local_state_kwargs(self, ctx: ClientContext, *, force_full_remote: bool = False) -> dict:
        """Return dict of local_player_state kwargs for any UPDATE_ARRAY builder.

        Every UPDATE_ARRAY with include_local_state=True MUST include the
        correct ammo/turret bit counts matching the BEHAVIOR packet config,
        otherwise the OG client's bitstream reads misalign â†’ protocol mismatch.
        """
        weapon_id = self._get_local_state_weapon_type(ctx)
        ammo_bits, ammo_mask = self._get_local_state_ammo_bits(
            ctx,
            force_full_remote=force_full_remote,
        )
        pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(
            ctx,
            force_full_remote=force_full_remote,
        )
        if self._wf_minimal_local_state_for_client(ctx) and not force_full_remote:
            weapon_id = self._get_spawn_tank_weapon_type(ctx)
        return dict(
            weapon_id=weapon_id,
            health=self._get_health_value(ctx),
            fuel=self._get_energy_value(ctx),
            ammo_count_bits=ammo_bits,
            ammo_count=ammo_mask,
            primary_turret_bits=pt_bits,
            primary_turret_angle=pt_angle,
            secondary_turret_bits=st_bits,
            secondary_turret_angle=st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
        )

    def _get_player_info_local_state_kwargs(self, ctx: ClientContext) -> tuple[bool, dict]:
        """Return (include_local_state, kwargs) for PLAYER_INFO.

        The original client always runs the local-state reader in PLAYER_INFO.
        Spawn TankPacket (0x18) still uses the short-form-safe weapon id because
        that packet only carries weapon+health+fuel, but canonical PLAYER_INFO
        should carry the real tank local-state so the OG HUD/ammo/turret state
        initializes through the decompile-backed path.
        """
        weapon_type = self._get_local_state_weapon_type(ctx)
        ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
        pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
        minimal_remote = self._wf_minimal_local_state_for_client(ctx)
        if minimal_remote:
            weapon_type = self._get_spawn_tank_weapon_type(ctx)

        mode = self.player_info_local_state_mode
        if mode == "off":
            include_local_state = False
        elif mode in ("force", "wf"):
            include_local_state = True
        elif mode == "auto-remote":
            include_local_state = not handlers._is_loopback_client(ctx)
        else:
            include_local_state = self._local_state_payload_is_safe_for_weapon(
                weapon_type,
                pt_bits,
                st_bits,
            )

        kwargs = dict(
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
        return include_local_state, kwargs

    def _local_state_payload_is_safe(self, ctx: ClientContext, primary_bits: int, secondary_bits: int) -> bool:
        """Return True if local-state payload includes required turret angles for this weapon type."""
        weapon_type = self._get_local_state_weapon_type(ctx)
        if weapon_type in LOCAL_STATE_PRIMARY_TURRET_WEAPON_TYPES and primary_bits <= 0:
            return False
        if weapon_type in LOCAL_STATE_SECONDARY_TURRET_WEAPON_TYPES and secondary_bits <= 0:
            return False
        return True

    def _local_state_payload_is_safe_for_weapon(self, weapon_type: int, primary_bits: int, secondary_bits: int) -> bool:
        """Return True if local-state payload includes required turret angles for an explicit weapon type."""
        if weapon_type in LOCAL_STATE_PRIMARY_TURRET_WEAPON_TYPES and primary_bits <= 0:
            return False
        if weapon_type in LOCAL_STATE_SECONDARY_TURRET_WEAPON_TYPES and secondary_bits <= 0:
            return False
        return True

    def _should_send_local_state(self, ctx: ClientContext, primary_bits: int, secondary_bits: int, mode: str) -> bool:
        """Decide whether to include local-state based on mode and required fields."""
        if mode == "off":
            return False
        if not ctx.session or not ctx.session.translation_ack_received:
            return False
        if mode in ("force", "wf"):
            return True
        safe = self._local_state_payload_is_safe(ctx, primary_bits, secondary_bits)
        if not safe and not getattr(ctx, "local_state_warned", False):
            weapon_type = self._get_local_state_weapon_type(ctx)
            print(
                "[LOCAL-STATE] Skipping local-state (auto): "
                f"weapon_type={weapon_type} primary_bits={primary_bits} secondary_bits={secondary_bits}"
            )
            ctx.local_state_warned = True
        return safe

    def _build_local_state_heartbeat(
        self,
        ctx: ClientContext,
        *,
        tick: Optional[int] = None,
        entity_id: Optional[int] = None,
        include_health: bool = True,
        health: Optional[float] = None,
        fuel: Optional[float] = None,
        is_view_update: Optional[bool] = None,
        rot: tuple = None,
        pos: tuple = None,
    ) -> bytes:
        """Build a heartbeat packet with local-state fields aligned to client expectations."""
        if tick is None:
            tick = self._get_network_tick(ctx)
        if entity_id is None:
            entity_id = ctx.session.entity_id or ctx.entity_id
        if health is None:
            health = self._get_health_value(ctx)
        if fuel is None:
            fuel = self._get_energy_value(ctx)
        if is_view_update is None:
            is_view_update = self.heartbeat_view_update

        weapon_type = self._get_local_state_weapon_type(ctx)
        ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
        pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
        minimal_remote = self._wf_minimal_local_state_for_client(ctx)
        if minimal_remote:
            weapon_type = self._get_spawn_tank_weapon_type(ctx)

        include_local_state = False
        if include_health:
            include_local_state = self._should_send_local_state(
                ctx,
                pt_bits,
                st_bits,
                self.update_local_state_mode,
            )

        if not include_local_state:
            ammo_bits = 0
            ammo_mask = 0
            pt_bits = 0
            st_bits = 0
            pt_angle = 0.0
            st_angle = 0.0

        if self._wf_remote_heartbeat_entity_mode(ctx) and not is_view_update:
            remote_pos = pos
            remote_vel = None
            remote_rot = rot
            if remote_rot is None and self.heartbeat_include_rot:
                # Promoted remote heartbeats should keep body rotation live so
                # the OG client does not zero angular velocity, but they should
                # not silently turn into full position corrections unless the
                # caller explicitly requested transform fields.
                remote_rot = self._local_player_sync_rotation(ctx)
            if remote_pos is not None:
                remote_vel = ctx.player_vel
            return self._build_remote_sync_heartbeat_update(
                ctx,
                tick=tick,
                pos=remote_pos,
                vel=remote_vel,
                rot=remote_rot,
                include_vel=remote_vel is not None,
                include_rot=remote_rot is not None,
                include_local_state=include_local_state,
                health=health,
                fuel=fuel,
                weapon_type=weapon_type,
                ammo_bits=ammo_bits,
                ammo_mask=ammo_mask,
                pt_bits=pt_bits,
                pt_angle=pt_angle,
                st_bits=st_bits,
                st_angle=st_angle,
            )

        use_local_entity_when_no_transform = False
        include_entities = True
        if minimal_remote:
            include_entities = False

        return build_update_array_heartbeat(
            tick=tick,
            entity_id=entity_id,
            include_health=include_local_state,
            weapon_id=weapon_type,
            health=health,
            fuel=fuel,
            ammo_count_bits=ammo_bits,
            ammo_count=ammo_mask,
            primary_turret_bits=pt_bits,
            primary_turret_angle=pt_angle,
            secondary_turret_bits=st_bits,
            secondary_turret_angle=st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
            is_view_update=is_view_update,
            include_entities=include_entities,
            use_local_entity_when_no_transform=use_local_entity_when_no_transform,
            rot=rot,
            pos=pos,
        )

    def _build_empirical_correction_payload(
        self,
        ctx: ClientContext,
        *,
        tick: int,
        include_local_state: bool,
        health: float,
        fuel: float,
        weapon_type: int,
        ammo_bits: int,
        ammo_mask: int,
        pt_bits: int,
        pt_angle: float,
        st_bits: int,
        st_angle: float,
    ) -> tuple[bytes, str, tuple[float, float, float], tuple[float, float, float], bool, bool]:
        """Build one of the older empirical local-correction packet shapes.

        The rotation tuple MUST match what every other local-player packet
        path emits (heartbeat, full UPDATE_ARRAY, VIEW_UPDATE loop,
        TankPacket resend, vitals heartbeat, solo-local keepalive). Those
        all use `_local_player_sync_rotation` which returns
        `(roll, pitch, player_heading)` — the body-heading convention.
        This function historically used `(roll, 0.0, player_yaw)` — the
        camera-yaw sign-flipped convention — which diverged from every
        other path and was flagged in the 2026-03-14 audit but missed.
        Aligning to `_local_player_sync_rotation` does not on its own
        restore the correction burst when the keepalive is running, but
        it removes a confounding variable. See
        docs/keepalive-breaks-correction-2026-04-18.md.
        """
        corr_pos = self._to_client_pos(ctx.player_pos)
        corr_rot = self._local_player_sync_rotation(ctx)
        cmode = self.correction_mode
        inc_pos = cmode in ("full", "pos_only", "dual_entity", "view_update", "view_update_define")
        inc_vel = cmode in ("full", "pos_only", "dual_entity", "view_update")
        inc_rot = cmode in ("full", "rot_only", "dual_entity", "view_update", "view_update_define")
        correction_timestamp = None
        if cmode in ("view_update", "view_update_define"):
            if handlers._is_loopback_client(ctx):
                last_request_id = int(getattr(ctx, "last_state_request_id", 0) or 0)
                last_request_time = float(getattr(ctx, "last_state_request_time", 0.0) or 0.0)
                if last_request_id and last_request_time > 0.0 and (time.monotonic() - last_request_time) <= 1.5:
                    correction_timestamp = last_request_id
            else:
                correction_timestamp = self._fresh_remote_view_update_timestamp(ctx, tick)

        if not handlers._is_loopback_client(ctx):
            # OG clients reject the promoted/full local-state form on targeted
            # correction bursts. Keep this aligned with STATE_REQUEST replies:
            # a short local-state prefix plus the transform-bearing entity.
            include_local_state = True
            weapon_type = self._get_spawn_tank_weapon_type(ctx)
            ammo_bits = 0
            ammo_mask = 0
            pt_bits = 0
            pt_angle = 0.0
            st_bits = 0
            st_angle = 0.0

        common_kw = dict(
            include_local_state=include_local_state,
            weapon_id=weapon_type,
            health=health,
            fuel=fuel,
            ammo_count_bits=ammo_bits,
            ammo_count=ammo_mask,
            primary_turret_bits=pt_bits,
            primary_turret_angle=pt_angle,
            secondary_turret_bits=st_bits,
            secondary_turret_angle=st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
        )

        if cmode == "view_update_define":
            team_id = int(getattr(ctx.session, "team_id", 0) or 1)
            entity_type = int(getattr(ctx, "entity_type", EntityType.TANK) or EntityType.TANK)
            payload = build_view_update_create_tank(
                tick=tick,
                entity_id=ctx.session.entity_id,
                entity_type=entity_type,
                team=team_id,
                pos=corr_pos,
                behavior_type=team_id,
                include_health=include_local_state,
                include_entity_vitals=False,
                health=health,
                fuel=fuel,
                is_manned=True,
                weapon_id=weapon_type,
                rot=corr_rot,
                ammo_count_bits=ammo_bits,
                ammo_count=ammo_mask,
                primary_turret_bits=pt_bits,
                primary_turret_angle=pt_angle,
                secondary_turret_bits=st_bits,
                secondary_turret_angle=st_angle,
                turret_max=self.local_state_turret_max,
                turret_range=self.local_state_turret_range,
                timestamp=correction_timestamp,
            )
            label = "CORRECTION(view_update_define)"
        elif cmode == "view_update":
            payload = build_view_update_player_update(
                tick=tick,
                entity_id=ctx.session.entity_id,
                pos=corr_pos,
                vel=ctx.player_vel,
                rot=corr_rot,
                include_pos=inc_pos,
                include_vel=inc_vel,
                include_rot=inc_rot,
                timestamp=correction_timestamp,
                **common_kw,
            )
            label = "CORRECTION(view_update)"
        elif cmode == "dual_entity":
            ent_real = dict(
                entity_id=ctx.session.entity_id,
                is_manned=True,
                pos=corr_pos,
                vel=ctx.player_vel,
                rot=corr_rot,
                include_pos=inc_pos,
                include_vel=inc_vel,
                include_rot=inc_rot,
            )
            ent_dummy = dict(
                entity_id=0xFFFFFFFE,
                is_manned=True,
                pos=(0.0, 0.0, 0.0),
                vel=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0),
                include_pos=False,
                include_vel=False,
                include_rot=False,
            )
            payload = build_update_array_multi(
                tick=tick,
                entities=[ent_real, ent_dummy],
                **common_kw,
            )
            label = "CORRECTION(dual_entity)"
        else:
            payload = build_update_array_player_update(
                tick=tick,
                entity_id=ctx.session.entity_id,
                pos=corr_pos,
                vel=ctx.player_vel,
                rot=corr_rot,
                include_pos=inc_pos,
                include_vel=inc_vel,
                include_rot=inc_rot,
                **common_kw,
            )
            label = f"CORRECTION({cmode})"

        return payload, label, corr_pos, corr_rot, inc_pos, inc_rot

    def _build_remote_sync_heartbeat_update(
        self,
        ctx: ClientContext,
        *,
        tick: int,
        pos: Optional[tuple[float, float, float]] = None,
        vel: Optional[tuple[float, float, float]] = None,
        rot: Optional[tuple[float, float, float]] = None,
        include_vel: bool,
        include_rot: bool,
        include_local_state: bool,
        health: float,
        fuel: float,
        weapon_type: int,
        ammo_bits: int,
        ammo_mask: int,
        pt_bits: int,
        pt_angle: float,
        st_bits: int,
        st_angle: float,
        safe_local_state: bool = True,
    ) -> bytes:
        """Build a safe promoted remote heartbeat/correction update.

        Loopback/Python heartbeats respect the caller's requested transform
        fields. Remote OG local-player updates must not send partial transform
        records: the decompiled transform applier can consume unpopulated
        position fields when rotation is present, poisoning the local entity
        with NaNs. For OG, any transform-bearing heartbeat is expanded to
        position + velocity + rotation.
        """
        if not handlers._is_loopback_client(ctx):
            wants_transform = pos is not None or vel is not None or rot is not None or include_vel or include_rot
            if wants_transform:
                if pos is None and getattr(ctx, "player_pos", None) is not None:
                    pos = self._to_client_pos(ctx.player_pos)
                if pos is not None:
                    if vel is None:
                        vel = ctx.player_vel
                    if rot is None:
                        rot = self._local_player_sync_rotation(ctx)
                    include_vel = True
                    include_rot = True
                else:
                    # The OG local-player transform applier is not safe with a
                    # partial interpolation record. If no finite position is
                    # available, fall back to a local-state heartbeat only.
                    vel = None
                    rot = None
                    include_vel = False
                    include_rot = False

        entity_id = ctx.session.entity_id or ctx.entity_id
        has_pos = pos is not None
        has_vel = include_vel and vel is not None
        has_rot = include_rot and rot is not None
        hb_rot = rot if rot is not None else self._local_player_sync_rotation(ctx)
        hb_pos = pos if pos is not None else self._to_client_pos(ctx.player_pos)
        hb_vel = vel if vel is not None else ctx.player_vel
        local_weapon_type = self._get_spawn_tank_weapon_type(ctx) if safe_local_state else weapon_type
        local_ammo_bits = 0 if safe_local_state else ammo_bits
        local_ammo_mask = 0 if safe_local_state else ammo_mask
        local_pt_bits = 0 if safe_local_state else pt_bits
        local_pt_angle = 0.0 if safe_local_state else pt_angle
        local_st_bits = 0 if safe_local_state else st_bits
        local_st_angle = 0.0 if safe_local_state else st_angle
        return build_update_array_player_update(
            tick=tick,
            entity_id=entity_id,
            pos=hb_pos,
            vel=hb_vel,
            rot=hb_rot,
            include_pos=has_pos,
            include_vel=has_vel,
            include_rot=has_rot,
            include_spin=has_rot,
            spin=(0.0, 0.0, 0.0),
            include_local_state=include_local_state,
            include_entity_vitals=False,
            weapon_id=local_weapon_type,
            health=health,
            fuel=fuel,
            ammo_count_bits=local_ammo_bits,
            ammo_count=local_ammo_mask,
            primary_turret_bits=local_pt_bits,
            primary_turret_angle=local_pt_angle,
            secondary_turret_bits=local_st_bits,
            secondary_turret_angle=local_st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
            is_manned=True,
            speed_scale=1.0,
        )

    def _build_remote_spawn_bootstrap_heartbeat(
        self,
        ctx: ClientContext,
        *,
        tick: int,
        entity_id: Optional[int] = None,
        health: float,
        fuel: float,
    ) -> bytes:
        """Build the one-off post-spawn OG heartbeat on the full-transform shape.

        Fresh remote OG spawns still benefit from a single local-state sync
        packet immediately after Tank/PLAYER_INFO, but the old minimal
        no-entity heartbeat is fragile in that bootstrap window. Reuse the
        promoted remote heartbeat layout; the remote OG guard expands the body
        rotation request into a complete position + velocity + rotation update
        so the client's interpolation record is fully populated.
        """
        if entity_id is None:
            entity_id = ctx.session.entity_id or ctx.entity_id
        saved_entity_id = ctx.session.entity_id
        if saved_entity_id != entity_id:
            ctx.session.entity_id = entity_id
        try:
            return self._build_remote_sync_heartbeat_update(
                ctx,
                tick=tick,
                rot=self._local_player_sync_rotation(ctx),
                include_vel=False,
                include_rot=True,
                include_local_state=True,
                health=health,
                fuel=fuel,
                weapon_type=self._get_spawn_tank_weapon_type(ctx),
                ammo_bits=0,
                ammo_mask=0,
                pt_bits=0,
                pt_angle=0.0,
                st_bits=0,
                st_angle=0.0,
                safe_local_state=True,
            )
        finally:
            ctx.session.entity_id = saved_entity_id

    def _local_player_sync_rotation(self, ctx: ClientContext) -> tuple[float, float, float]:
        """Return the body-space rotation tuple for local-player replication packets.

        The old client stores entity body rotation in entity+0x30/0x34/0x38 and
        targeted correction only became stable once STATE_REQUEST replies used
        that body-heading tuple instead of the client camera-yaw convention.
        Keep the ordinary local-player replication paths on the same body-space
        rotation so UPDATE_ARRAY / VIEW_UPDATE do not fight targeted correction.
        """
        return (
            ctx.player_pose.get("roll", 0.0),
            ctx.player_pose.get("pitch", 0.0),
            ctx.player_heading,
        )

    def _player_body_rotation(
        self,
        ctx: ClientContext,
        *,
        negate_yaw: bool = False,
        yaw_offset: float = 0.0,
    ) -> tuple[float, float, float]:
        """Return the current replicated body rotation tuple for an entity."""
        yaw = -ctx.player_heading if negate_yaw else ctx.player_heading
        return (
            ctx.player_pose.get("roll", 0.0),
            ctx.player_pose.get("pitch", 0.0),
            yaw + yaw_offset,
        )

    def _get_update_array_local_state_for_viewer(self, ctx: ClientContext) -> tuple[bool, dict]:
        """Return viewer-local local_state args for non-heartbeat UPDATE_ARRAY packets.

        The OG client clears its local input/local-state scratch at the start of
        every UPDATE_ARRAY, then runs local-player sync at the end of the
        packet. Remote OG viewers therefore still need a valid local-state
        prefix even on entity-only updates. Promoted remote viewers still
        reject the fully expanded tank local-state on packets that do not also
        carry the local-player sync entity block, so keep these on the same
        short-form-safe shape as the projectile/update-array compatibility path.
        Loopback/Python clients keep the existing entity-only path to preserve
        the currently working decoder behavior.
        """
        if handlers._is_loopback_client(ctx):
            return False, {}

        return True, dict(
            weapon_id=self._get_spawn_tank_weapon_type(ctx),
            health=self._get_health_value(ctx),
            fuel=self._get_energy_value(ctx),
            ammo_count_bits=0,
            ammo_count=0,
            primary_turret_bits=0,
            primary_turret_angle=0.0,
            secondary_turret_bits=0,
            secondary_turret_angle=0.0,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
        )

    def _get_projectile_local_state_for_viewer(self, ctx: ClientContext) -> tuple[bool, dict]:
        """Return viewer-local local_state args for projectile UPDATE_ARRAY packets.

        OG clients still need a valid local-state prefix on projectile packets,
        but the fully expanded local-state form is fragile here because the
        projectile bitstream does not also carry the local-player entity sync
        block that the promoted heartbeat path uses. Keep remote projectile
        packets on the same short-form-safe local-state shape as spawn-time
        heartbeats even after the viewer has been promoted to full sync.
        """
        if handlers._is_loopback_client(ctx):
            return False, {}

        return True, dict(
            weapon_id=self._get_spawn_tank_weapon_type(ctx),
            health=self._get_health_value(ctx),
            fuel=self._get_energy_value(ctx),
            ammo_count_bits=0,
            ammo_count=0,
            primary_turret_bits=0,
            primary_turret_angle=0.0,
            secondary_turret_bits=0,
            secondary_turret_angle=0.0,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
        )

    def _should_send_spawn_player_info(self, ctx: ClientContext) -> bool:
        """Return whether spawn should include canonical PLAYER_INFO for this client."""
        if self.spawn_send_player_info_explicit:
            return self.spawn_send_player_info
        return not handlers._is_loopback_client(ctx)

    def _maybe_send_view_update_loop(
        self,
        ctx: ClientContext,
        *,
        tick: int,
        send_pos: tuple,
        health_val: float,
        fuel_val: float,
        weapon_type: int,
        ammo_bits: int,
        ammo_mask: int,
        pt_bits: int,
        pt_angle: float,
        st_bits: int,
        st_angle: float,
    ) -> None:
        """Optionally send auxiliary VIEW_UPDATE correction/replay packets."""
        if not self.view_update_loop or self.update_packet_type == "view":
            return

        now = time.monotonic()
        if (now - ctx.last_view_update_send) < self.view_update_interval:
            return
        ctx.last_view_update_send = now

        view_local_state = False
        view_ammo_bits = 0
        view_ammo_mask = 0
        view_pt_bits = 0
        view_st_bits = 0
        view_pt_angle = 0.0
        view_st_angle = 0.0

        # Only include local-state in VIEW_UPDATE when explicitly enabled.
        if self.view_update_local_stats:
            view_mode = "wf" if self.update_local_state_mode == "wf" else "auto"
            if self._should_send_local_state(ctx, pt_bits, st_bits, view_mode):
                view_local_state = True
                view_ammo_bits = ammo_bits
                view_ammo_mask = ammo_mask
                view_pt_bits = pt_bits
                view_st_bits = st_bits
                view_pt_angle = pt_angle
                view_st_angle = st_angle

        view_entities = [
            {
                "entity_id": ctx.session.entity_id,
                "is_manned": True,
                "pos": send_pos,
                "vel": ctx.player_vel,
                "rot": self._local_player_sync_rotation(ctx),
                "include_pos": True,
                "include_vel": True,
                "include_rot": True,
                "include_entity_vitals": self.view_update_entity_vitals,
                "speed_scale": 1.0,
                "fuel": fuel_val,
            }
        ]
        view_payload = build_view_update_multi(
            tick,
            include_local_state=view_local_state,
            weapon_id=weapon_type,
            health=health_val,
            fuel=fuel_val,
            ammo_count_bits=view_ammo_bits,
            ammo_count=view_ammo_mask,
            primary_turret_bits=view_pt_bits,
            primary_turret_angle=view_pt_angle,
            secondary_turret_bits=view_st_bits,
            secondary_turret_angle=view_st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
            entities=view_entities,
        )
        if view_local_state:
            self._log_vitals(
                ctx,
                "VIEW_UPDATE",
                include_vitals=True,
                health=health_val,
                energy=fuel_val,
                weapon_id=weapon_type,
                note=f"ammo_bits={view_ammo_bits} pt_bits={view_pt_bits} st_bits={view_st_bits}",
            )
        if self.udp_handler and ctx.session.udp_addr:
            self.udp_handler.send_to(view_payload, ctx.session.udp_addr)
        # Never send VIEW_UPDATE over TCP (can desync OG stream parser).

    def _to_client_pos(self, pos: tuple) -> tuple:
        """Apply configured world offset when sending positions to the client."""
        if self.up_axis == "z":
            return (pos[0], pos[1], pos[2] + self.pos_offset)
        return (pos[0], pos[1] + self.pos_offset, pos[2])

    def _from_client_pos(self, pos: tuple) -> tuple:
        """Undo the configured world offset from a client-facing position."""
        if self.up_axis == "z":
            return (pos[0], pos[1], pos[2] - self.pos_offset)
        return (pos[0], pos[1] - self.pos_offset, pos[2])

    def _extract_update_mask(self, payload: bytes) -> Optional[int]:
        """Decode local entity update_mask from UPDATE_ARRAY payload for diagnostics."""
        if not payload or len(payload) < 8 or payload[0] != 0x0E:
            return None
        try:
            br = BitReader(payload[5:])
            _has_local_state = br.read_bits(1)
            entity_count = br.read_bits(8)
            if entity_count < 1:
                return None
            _entity_id = br.read_bits(32)
            _is_manned = br.read_bits(1)
            return br.read_bits(10)
        except Exception:
            return None

    def _get_health_value(self, ctx: ClientContext = None) -> float:
        """Return health multiplier for outgoing packets (0..1).

        When ctx is provided, incorporates the client's actual health
        (from combat damage) into the returned value.
        """
        value = self.debug_health_value
        if self.debug_health_pattern:
            period = self.debug_health_period if self.debug_health_period > 0 else 1.0
            phase = int(time.monotonic() / period) % 2
            value = self.debug_health_low if phase else self.debug_health_value
        # Apply per-client health (combat damage)
        if ctx is not None:
            value = value * ctx.player_health
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _get_energy_value(self, ctx: Optional[ClientContext]) -> float:
        """Return normalized local-state energy (0..1) from per-client energy pool."""
        if ctx is None:
            return 1.0
        max_energy = self.player_energy_max if self.player_energy_max > 0.0 else 100.0
        value = max(0.0, min(max_energy, ctx.player_energy))
        return value / max_energy

    def _consume_player_energy(self, ctx: Optional[ClientContext], amount: float) -> float:
        """Consume energy from a client and return the amount actually consumed."""
        if ctx is None or amount <= 0.0:
            return 0.0
        if not self.weapon_energy_enabled:
            return 0.0
        available = max(0.0, ctx.player_energy)
        used = min(available, amount)
        ctx.player_energy = available - used
        return used

    def _regen_player_energy(self, ctx: Optional[ClientContext], dt: float) -> None:
        """Regenerate player energy over time."""
        if ctx is None or dt <= 0.0 or self.player_energy_regen <= 0.0:
            return
        max_energy = self.player_energy_max if self.player_energy_max > 0.0 else 100.0
        ctx.player_energy = min(max_energy, ctx.player_energy + (self.player_energy_regen * dt))

    def _log_vitals(self, ctx: ClientContext, source: str, *,
                    include_vitals: bool,
                    health: float,
                    energy: float,
                    weapon_id: int = 0,
                    note: str = "") -> None:
        """Debug helper: trace health/energy bits by packet source."""
        if not self.debug_vitals:
            return
        if include_vitals:
            health_bits = _encode_health_bits(health, total_bits=10)
            energy_bits = _encode_health_bits(energy, total_bits=10)
            msg = (
                f"[VITALS] src={source} client={ctx.client_id} "
                f"health={health:.2f} bits=0x{health_bits:03x} "
                f"energy={energy:.2f} bits=0x{energy_bits:03x} "
                f"weapon={weapon_id}"
            )
        else:
            msg = f"[VITALS] src={source} client={ctx.client_id} include=0 weapon={weapon_id}"
        if note:
            msg += f" {note}"
        print(msg)

    def _snapshot_clients(self):
        """Return a snapshot list of all clients (thread-safe)."""
        with self.clients_lock:
            return list(self.clients.values())

    def _snapshot_in_game_clients(self):
        """Return a snapshot list of in-game clients (thread-safe)."""
        return [c for c in self._snapshot_clients() if c.session and c.session.in_game]

    def _send_packet_to_client(
        self,
        ctx: ClientContext,
        payload: bytes,
        *,
        prefer_tcp: bool = True,
        allow_udp_fallback: bool = True,
    ) -> bool:
        """Send payload to a client, preferring TCP and optionally falling back to UDP."""
        sent = False
        if prefer_tcp and ctx.tcp_handler:
            try:
                ctx.tcp_handler.send(payload, log=False)
                sent = True
            except Exception as tcp_err:
                print(f"[MULTI] Client {ctx.client_id}: TCP send failed ({tcp_err})")
        if not sent and allow_udp_fallback and self.udp_handler and ctx.session.udp_addr:
            try:
                self.udp_handler.send_to(payload, ctx.session.udp_addr)
                sent = True
            except Exception as udp_err:
                print(f"[MULTI] Client {ctx.client_id}: UDP send failed ({udp_err})")
        return sent

    def _send_spawn_create_update_array(self, ctx: ClientContext, payload: bytes) -> bool:
        """Send local spawn pre-creation UPDATE_ARRAY on the safest transport.

        Remote OG clients are still fragile around UPDATE_ARRAY on the TCP
        stream during bootstrap/spawn. Once UDP bootstrap is ready, keep remote
        clients on UDP here and reserve TCP for loopback probes or last-resort
        fallback when UDP is unavailable.
        """
        sent = False
        if self.udp_handler and ctx.session.udp_addr:
            self.udp_handler.send_to(payload, ctx.session.udp_addr)
            sent = True
        if handlers._is_loopback_client(ctx) and ctx.tcp_handler:
            ctx.tcp_handler.send(payload)
            sent = True
        elif not sent and ctx.tcp_handler:
            ctx.tcp_handler.send(payload)
            sent = True
        return sent

    def _decode_comm_message_request_body(self, body: bytes) -> dict[str, Any]:
        """Decode the body shared by TCP and reliable-UDP COMM_MESSAGE_REQUEST.

        Decompile-backed OG uplink commands write:
          u16 message_type, u16 flags_or_target, string command

        Python chat uses the same leading fields with different message types,
        so this decoder stays generic and lets the caller decide semantics.
        """
        decoded: dict[str, Any] = {
            "ok": False,
            "message_type": None,
            "flags_or_target": None,
            "text": "",
            "body_hex": body.hex(),
        }
        if len(body) < 6:
            decoded["error"] = f"body too short: {len(body)}"
            return decoded
        try:
            message_type = struct.unpack_from(">H", body, 0)[0]
            flags_or_target = struct.unpack_from(">H", body, 2)[0]
            text, offset = handlers.decode_lp_string(body, 4)
        except (struct.error, ValueError) as exc:
            decoded["error"] = str(exc)
            return decoded
        decoded.update(
            {
                "ok": True,
                "message_type": message_type,
                "flags_or_target": flags_or_target,
                "text": text,
                "end_offset": offset,
                "trailing_hex": body[offset:].hex() if offset < len(body) else "",
            }
        )
        return decoded

    def _parse_build_uplink_command(self, text: str) -> dict[str, Any]:
        """Parse OG type-2 starship/uplink text commands."""
        import shlex

        result: dict[str, Any] = {"ok": False, "text": text, "action": ""}
        try:
            parts = shlex.split(text or "")
        except ValueError as exc:
            result["error"] = f"shlex: {exc}"
            return result
        if not parts:
            result["error"] = "empty command"
            return result
        action = parts[0].lower()
        result["action"] = action
        result["parts"] = parts

        def _parse_int(value: str, field: str) -> int | None:
            try:
                return int(value, 0)
            except (TypeError, ValueError):
                result["error"] = f"invalid {field}: {value!r}"
                return None

        if action in ("build", "delete"):
            if len(parts) < 4:
                result["error"] = f"{action} requires ship_oid, entity name, and slot"
                return result
            ship_oid = _parse_int(parts[1], "ship_oid")
            slot = _parse_int(parts[3], "slot")
            if ship_oid is None or slot is None:
                return result
            entity_type = self._build_uplink_entity_type_from_name(parts[2])
            result.update(
                {
                    "ok": entity_type is not None,
                    "ship_oid": ship_oid,
                    "entity_name": parts[2],
                    "entity_type": entity_type,
                    "slot": slot,
                }
            )
            if entity_type is None:
                result["error"] = f"unsupported build entity: {parts[2]!r}"
            return result

        if action == "move":
            if len(parts) < 3:
                result["error"] = "move requires ship_oid and cell"
                return result
            ship_oid = _parse_int(parts[1], "ship_oid")
            if ship_oid is None:
                return result
            result.update({"ok": True, "ship_oid": ship_oid, "cell": parts[2]})
            return result

        if action == "bomb":
            if len(parts) < 2:
                result["error"] = "bomb requires ship_oid"
                return result
            ship_oid = _parse_int(parts[1], "ship_oid")
            if ship_oid is None:
                return result
            result.update({"ok": True, "ship_oid": ship_oid})
            return result

        if action == "set":
            if len(parts) < 4:
                result["error"] = "set requires ship_oid, field, and value"
                return result
            ship_oid = _parse_int(parts[1], "ship_oid")
            value = _parse_int(parts[3], "value")
            if ship_oid is None or value is None:
                return result
            result.update({"ok": True, "ship_oid": ship_oid, "field": parts[2], "value": value})
            return result

        result["error"] = f"unsupported action: {action!r}"
        return result

    @staticmethod
    def _build_uplink_entity_type_from_name(name: str) -> Optional[int]:
        key = "".join(ch for ch in str(name or "").lower() if ch.isalnum())
        aliases = {
            "repair": EntityType.REPAIR_BUILDING,
            "repairbuilding": EntityType.REPAIR_BUILDING,
            "repairpad": EntityType.REPAIR_BUILDING,
            "fuel": EntityType.FUEL_BUILDING,
            "fuelbuilding": EntityType.FUEL_BUILDING,
            "refuel": EntityType.FUEL_BUILDING,
            "refuelpad": EntityType.FUEL_BUILDING,
            "energy": EntityType.ENERGY_BUILDING,
            "energybuilding": EntityType.ENERGY_BUILDING,
            "energypad": EntityType.ENERGY_BUILDING,
            "powercell": EntityType.ENERGY_BUILDING,
            "gun": EntityType.GUN_TURRET,
            "turret": EntityType.GUN_TURRET,
            "gunturret": EntityType.GUN_TURRET,
            "gunbuilding": EntityType.GUN_TURRET,
        }
        value = aliases.get(key)
        return int(value) if value is not None else None

    @staticmethod
    def _building_max_health_for_type(entity_type: int) -> float:
        max_health = {
            EntityType.GUN_TURRET: 1200.0,
            EntityType.LAUNCHER: 1200.0,
            EntityType.SENSOR_BUILDING: 1200.0,
            EntityType.FUEL_BUILDING: 2000.0,
            EntityType.REPAIR_BUILDING: 2000.0,
            EntityType.ENERGY_BUILDING: 2000.0,
            EntityType.PAD: 5000.0,
            EntityType.DARK_LIGHT: 800.0,
        }
        try:
            key = EntityType(int(entity_type))
        except ValueError:
            key = int(entity_type)
        return float(max_health.get(key, 2000.0))

    def _allocate_dynamic_building_oid(self) -> int:
        oid = int(getattr(self, "_dynamic_building_next_oid", 30000) or 30000)
        while oid in self._building_entities or oid in self._dynamic_building_ids:
            oid += 1
        self._dynamic_building_next_oid = oid + 1
        return oid

    def _choose_dynamic_building_pos(self, ctx: ClientContext, slot: int) -> tuple[float, float, float]:
        heading = float(getattr(ctx, "player_heading", 0.0) or 0.0)
        base = tuple(float(v) for v in (getattr(ctx, "player_pos", None) or (2600.0, 3040.0, 5.0)))
        distance = 35.0 + max(0, int(slot)) * 12.0
        x = base[0] + math.cos(heading) * distance
        y = base[1] + math.sin(heading) * distance
        ground_z = self._terrain_ground_z_at(x, y)
        if ground_z is None or not math.isfinite(float(ground_z)):
            z = base[2]
        else:
            z = float(ground_z)
        return (x, y, z)

    def _send_dynamic_entity_definition(
        self,
        target_ctx: ClientContext,
        *,
        entity_id: int,
        entity_type: int,
        team_id: int,
        pos: tuple[float, float, float],
        heading: float = 0.0,
        is_static: bool = True,
    ) -> bool:
        if not target_ctx.session or not target_ctx.session.translation_ack_received:
            return False
        tick = self._get_network_tick(target_ctx)
        include_local_state, ls = self._get_update_array_local_state_for_viewer(target_ctx)
        local_state_kwargs = dict(ls)
        local_state_kwargs.setdefault("health", self._get_health_value(target_ctx))
        local_state_kwargs.setdefault("fuel", self._get_energy_value(target_ctx))
        payload = build_update_array_create_tank(
            tick=tick,
            entity_id=entity_id,
            entity_type=entity_type,
            team=team_id,
            pos=self._to_client_pos(pos),
            behavior_type=team_id,
            include_health=include_local_state,
            include_entity_vitals=False,
            is_manned=False,
            is_static=is_static,
            rot=(0.0, 0.0, float(heading)),
            **local_state_kwargs,
        )
        sent = self._send_packet_to_client(target_ctx, payload, prefer_tcp=False)
        if sent:
            target_ctx.known_entity_ids.add(entity_id)
            if self.pktlog.enabled:
                self.pktlog.log(
                    client_id=target_ctx.client_id,
                    label="DYNAMIC_ENTITY_CREATE",
                    tick=tick,
                    payload=payload,
                    transport="UDP",
                    entity_count=1,
                    entity_ids=(entity_id,),
                    mask_bits=(0b1011,),
                    has_local_state=include_local_state,
                    health=self._get_health_value(target_ctx) if include_local_state else -1.0,
                    extra=f"type={entity_type} team={team_id}",
                )
        return sent

    def _broadcast_dynamic_entity_definition(
        self,
        *,
        entity_id: int,
        entity_type: int,
        team_id: int,
        pos: tuple[float, float, float],
        heading: float = 0.0,
        is_static: bool = True,
    ) -> int:
        sent = 0
        for target in self._snapshot_in_game_clients():
            if self._send_dynamic_entity_definition(
                target,
                entity_id=entity_id,
                entity_type=entity_type,
                team_id=team_id,
                pos=pos,
                heading=heading,
                is_static=is_static,
            ):
                sent += 1
        return sent

    def _create_dynamic_building_from_uplink(
        self,
        ctx: ClientContext,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        entity_type = int(command["entity_type"])
        team_id = int(ctx.session.team_id or 1)
        slot = int(command.get("slot", 0) or 0)
        oid = self._allocate_dynamic_building_oid()
        x, y, z = self._choose_dynamic_building_pos(ctx, slot)
        heading = float(getattr(ctx, "player_heading", 0.0) or 0.0)
        building = BuildingEntity(
            x=x,
            y=y,
            z=z,
            entity_type=entity_type,
            team_id=team_id,
            heading=heading,
        )
        max_hp = self._building_max_health_for_type(entity_type)
        self._building_entities[oid] = building
        self._building_health[oid] = max_hp
        self._building_max_health[oid] = max_hp
        self._dynamic_building_ids.add(oid)
        source = {
            "client_id": ctx.client_id,
            "player_entity_id": ctx.session.entity_id or ctx.entity_id,
            "ship_oid": command.get("ship_oid"),
            "slot": slot,
            "command": command,
            "created_at": time.time(),
        }
        self._dynamic_building_sources[oid] = source
        ship = self._uplink_ships.get(team_id) or self._get_or_create_uplink_ship(ctx, team_id)
        cargo = list(ship.get("cargo", [40, 40, 40, 40]))
        if 0 <= slot < len(cargo):
            cargo[slot] = entity_type
        ship["cargo"] = cargo
        ship["last_build_oid"] = oid
        self._broadcast_uplink_ship_info(ship)
        self._rebuild_static_world_raycast_index()
        sent = self._broadcast_dynamic_entity_definition(
            entity_id=oid,
            entity_type=entity_type,
            team_id=team_id,
            pos=building.pos,
            heading=heading,
            is_static=True,
        )
        event = {
            "ok": sent > 0,
            "oid": oid,
            "entity_type": entity_type,
            "entity_type_name": getattr(EntityType(entity_type), "name", str(entity_type)),
            "team_id": team_id,
            "pos": [round(float(v), 5) for v in building.pos],
            "health": max_hp,
            "replication_targets": sent,
        }
        print(
            f"[BUILD-UPLINK] created oid={oid} type={event['entity_type_name']} "
            f"team={team_id} pos=({x:.1f},{y:.1f},{z:.1f}) targets={sent}"
        )
        return event

    def _delete_dynamic_building_from_uplink(
        self,
        ctx: ClientContext,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        entity_type = int(command.get("entity_type", -1) or -1)
        team_id = int(ctx.session.team_id or 1)
        slot = command.get("slot")
        candidates = []
        for oid in sorted(self._dynamic_building_ids):
            building = self._building_entities.get(oid)
            if not building:
                continue
            source = self._dynamic_building_sources.get(oid, {})
            if entity_type >= 0 and int(building.entity_type) != entity_type:
                continue
            if int(building.team_id) != team_id:
                continue
            if slot is not None and source.get("slot") != slot:
                continue
            candidates.append(oid)
        if not candidates:
            return {"ok": False, "error": "no matching dynamic building"}
        oid = candidates[-1]
        self._building_entities.pop(oid, None)
        self._building_health.pop(oid, None)
        self._building_max_health.pop(oid, None)
        self._dynamic_building_ids.discard(oid)
        self._dynamic_building_sources.pop(oid, None)
        ship = self._uplink_ships.get(team_id)
        if ship is not None and slot is not None:
            cargo = list(ship.get("cargo", [40, 40, 40, 40]))
            try:
                slot_index = int(slot)
            except (TypeError, ValueError):
                slot_index = -1
            if 0 <= slot_index < len(cargo):
                cargo[slot_index] = 40
            ship["cargo"] = cargo
            self._broadcast_uplink_ship_info(ship)
        self._rebuild_static_world_raycast_index()
        packet = build_delete_object(get_ticks(), [oid], with_effects=True)
        sent = 0
        for target in self._snapshot_in_game_clients():
            if self._send_packet_to_client(target, packet, prefer_tcp=False):
                target.known_entity_ids.discard(oid)
                sent += 1
        return {"ok": sent > 0, "oid": oid, "replication_targets": sent}

    def _get_or_create_uplink_ship(self, ctx: ClientContext, team_id: int) -> dict[str, Any]:
        ship = self._uplink_ships.get(team_id)
        if ship is not None:
            return ship
        base_pos = tuple(float(v) for v in (ctx.player_pos or (2600.0, 3040.0, 5.0)))
        try:
            offset_x = float(os.environ.get("WULFRAM_UPLINK_SHIP_OFFSET_X", "-450.0"))
            offset_y = float(os.environ.get("WULFRAM_UPLINK_SHIP_OFFSET_Y", "0.0"))
            offset_z = float(os.environ.get("WULFRAM_UPLINK_SHIP_OFFSET_Z", "12.0"))
        except ValueError:
            offset_x = -450.0
            offset_y = 0.0
            offset_z = 12.0
        x = base_pos[0] + offset_x
        y = base_pos[1] + offset_y
        ground_z = self._terrain_ground_z_at(x, y)
        z = (float(ground_z) + offset_z) if ground_z is not None else base_pos[2] + offset_z
        try:
            base_oid = int(os.environ.get("WULFRAM_UPLINK_SHIP_BASE_OID", "29000"), 0)
        except ValueError:
            base_oid = 29000
        ship = {
            "oid": base_oid + int(team_id),
            "team_id": team_id,
            "name": f"Team {team_id} Supply Ship",
            "pos": (x, y, z),
            "heading": 0.0,
            "cargo": [40, 40, 40, 40],
            "cargo_times": [0, 0, 0, 0],
            "build_mode": 3,
            "shield_pct": 100,
            "status_template": 0,
        }
        self._uplink_ships[team_id] = ship
        return ship

    def _build_uplink_ship_info_packet(self, ship: dict[str, Any]) -> bytes:
        return build_supply_ship_info(
            int(ship["oid"]),
            shield_pct=int(ship.get("shield_pct", 100) or 100),
            status_template=int(ship.get("status_template", 0) or 0),
            cargo_slots=list(ship.get("cargo", [40, 40, 40, 40])),
            cargo_times=list(ship.get("cargo_times", [0, 0, 0, 0])),
            build_mode=int(ship.get("build_mode", 3) or 3),
        )

    def _send_uplink_ship_info(self, ctx: ClientContext, ship: dict[str, Any]) -> bool:
        return self._send_packet_to_client(ctx, self._build_uplink_ship_info_packet(ship), prefer_tcp=True)

    def _broadcast_uplink_ship_info(self, ship: dict[str, Any]) -> int:
        sent = 0
        for target in self._snapshot_in_game_clients():
            if self._send_uplink_ship_info(target, ship):
                sent += 1
        return sent

    def _send_existing_build_uplink_entities(self, ctx: ClientContext) -> int:
        sent = 0
        for team_id, ship in sorted(self._uplink_ships.items(), key=lambda item: int(item[0])):
            if self._send_dynamic_entity_definition(
                ctx,
                entity_id=int(ship["oid"]),
                entity_type=int(EntityType.SUPPLY_SHIP),
                team_id=int(team_id),
                pos=ship["pos"],
                heading=float(ship.get("heading", 0.0) or 0.0),
                is_static=True,
            ):
                sent += 1
        for oid in sorted(self._dynamic_building_ids):
            building = self._building_entities.get(oid)
            if not building:
                continue
            if self._send_dynamic_entity_definition(
                ctx,
                entity_id=int(oid),
                entity_type=int(building.entity_type),
                team_id=int(building.team_id),
                pos=building.pos,
                heading=float(getattr(building, "heading", 0.0) or 0.0),
                is_static=True,
            ):
                sent += 1
        return sent

    def _ensure_uplink_mvp_state(self, ctx: ClientContext) -> None:
        """Default-off bootstrap for the OG uplink/build MVP probe."""
        if not getattr(self, "build_uplink_mvp", False):
            return
        if getattr(ctx, "uplink_mvp_bootstrap_sent", False):
            return
        if not ctx.session or not ctx.session.in_game or not ctx.session.translation_ack_received:
            return
        team_id = int(ctx.session.team_id or 1)
        player_oid = int(ctx.session.entity_id or ctx.entity_id)
        ship = self._get_or_create_uplink_ship(ctx, team_id)
        packets = (
            # ADD_TO_ROSTER creates the local PlayerEntry, but the decompile shows
            # UPDATE_STATS is the path that writes g_player_team.  The uplink UI
            # uses that global to find the team supply ship.
            build_update_stats_team_first(player_id=player_oid, entity_id=player_oid, team_id=team_id),
            build_ship_status(int(ship["oid"]), team_id, str(ship["name"])),
            self._build_uplink_ship_info_packet(ship),
            build_carrying_info(player_oid, cargo_type=0, has_uplink=True, cargo_count=0),
            # State 3 is the decompile-labeled "in use" uplink state.
            build_uplink_info(team_id, player_oid, 3),
        )
        sent = 0
        for payload in packets:
            if self._send_packet_to_client(ctx, payload, prefer_tcp=True):
                sent += 1
        dynamic_sent = self._send_existing_build_uplink_entities(ctx)
        ctx.uplink_mvp_bootstrap_sent = sent == len(packets) and dynamic_sent > 0
        print(
            f"[BUILD-UPLINK] bootstrap client={ctx.client_id} team={team_id} "
            f"ship={ship['oid']} player={player_oid} packets={sent}/{len(packets)} "
            f"dynamic_entities={dynamic_sent}"
        )

    def _handle_comm_message_request(
        self,
        ctx: Optional[ClientContext],
        packet: bytes,
        *,
        transport: str,
        body: bytes,
        addr: Optional[tuple] = None,
        sequence: Optional[int] = None,
    ) -> dict[str, Any]:
        decoded = self._decode_comm_message_request_body(body)
        event: dict[str, Any] = {
            "time": time.time(),
            "transport": transport,
            "client_id": getattr(ctx, "client_id", None),
            "addr": list(addr) if addr else None,
            "sequence": sequence,
            "raw_hex": packet.hex(),
            "decoded": decoded,
            "handled": False,
            "mvp_enabled": bool(getattr(self, "build_uplink_mvp", False)),
        }
        if ctx is not None:
            ctx.comm_message_request_count = int(getattr(ctx, "comm_message_request_count", 0) or 0) + 1
            ctx.last_comm_message_request = event
        if not decoded.get("ok"):
            return event

        text = str(decoded.get("text") or "")
        msg_type = int(decoded.get("message_type") or 0)
        if msg_type != 2:
            return event

        command = self._parse_build_uplink_command(text)
        event["build_uplink_command"] = command
        event["handled"] = True
        if ctx is not None:
            ctx.build_uplink_command_count = int(getattr(ctx, "build_uplink_command_count", 0) or 0) + 1
            ctx.last_build_uplink_command = event
        if not getattr(self, "build_uplink_mvp", False):
            event["result"] = {"ok": False, "error": "WULFRAM_BUILD_UPLINK_MVP disabled"}
        elif ctx is None:
            event["result"] = {"ok": False, "error": "unknown client"}
        elif not command.get("ok"):
            event["result"] = {"ok": False, "error": command.get("error", "parse failed")}
        elif command.get("action") == "build":
            event["result"] = self._create_dynamic_building_from_uplink(ctx, command)
        elif command.get("action") == "delete":
            event["result"] = self._delete_dynamic_building_from_uplink(ctx, command)
        elif command.get("action") == "set":
            team_id = int(ctx.session.team_id or 1)
            ship = self._uplink_ships.get(team_id)
            if ship is not None and str(command.get("field", "")).lower() == "build_mode":
                ship["build_mode"] = int(command.get("value", 2) or 2)
                self._broadcast_uplink_ship_info(ship)
            event["result"] = {"ok": True, "noted": True}
        else:
            event["result"] = {"ok": True, "noted": True}

        self._build_uplink_command_events.append(event)
        del self._build_uplink_command_events[:-100]
        print(
            f"[BUILD-UPLINK] c{getattr(ctx, 'client_id', '?')} {transport} "
            f"type=2 text={text!r} result={event.get('result')}"
        )
        return event

    def _handle_tcp_comm_message_request(self, ctx: ClientContext, packet: bytes) -> dict[str, Any]:
        return self._handle_comm_message_request(
            ctx,
            packet,
            transport="tcp",
            body=packet[1:],
            addr=getattr(ctx, "client_addr", None),
        )

    def _og_viewer_replication_enabled(self, target_ctx: ClientContext, stream: str) -> bool:
        """Return whether a replication stream is enabled for an OG/remote viewer."""
        if handlers._is_loopback_client(target_ctx):
            return True
        if stream == "roster":
            return bool(getattr(self, "og_viewer_roster_entry", True))
        if stream == "entity_create":
            return bool(getattr(self, "og_viewer_entity_create", True))
        if stream == "remote_updates":
            return bool(getattr(self, "og_viewer_remote_updates", True))
        return True

    def _send_roster_entry(self, target_ctx: ClientContext, player_ctx: ClientContext) -> None:
        """Send ADD_TO_ROSTER for player_ctx to target_ctx (once)."""
        if not self._og_viewer_replication_enabled(target_ctx, "roster"):
            return
        player_id = player_ctx.session.player_id or player_ctx.entity_id
        if player_id in target_ctx.known_roster_ids:
            return
        name = player_ctx.session.username or f"Player{player_ctx.client_id}"
        team = player_ctx.session.team_id or 1
        payload = build_add_to_roster(
            player_id=player_id,
            entity_id=player_id,
            name=name,
            team=team,
            kills=player_ctx.kills,
            deaths=player_ctx.deaths,
        )
        if not self._send_packet_to_client(
            target_ctx,
            payload,
            prefer_tcp=True,
            allow_udp_fallback=False,
        ):
            return
        target_ctx.known_roster_ids.add(player_id)
        print(f"[MULTI] Sent roster {name} (id={player_id}) -> client {target_ctx.client_id}")

    def _combat_observer_packets_allowed_for_client(
        self,
        ctx: ClientContext,
        *participants: ClientContext,
    ) -> bool:
        """Return whether combat/death TCP packets are safe for this viewer."""
        if handlers._is_loopback_client(ctx):
            return True
        if any(ctx is participant for participant in participants):
            return True
        return getattr(self, "remote_combat_observer_packets", True)

    def _broadcast_player_stats(
        self,
        player_ctx: ClientContext,
        *,
        participants: tuple[ClientContext, ...] = (),
    ) -> None:
        """Send UPDATE_STATS for player_ctx to all connected clients."""
        player_id = player_ctx.session.player_id or player_ctx.entity_id
        entity_id = player_ctx.entity_id
        team = player_ctx.session.team_id or 1
        pkt = build_update_stats(
            player_id=player_id,
            entity_id=entity_id,
            kills=player_ctx.kills,
            deaths=player_ctx.deaths,
            team_id=team,
        )
        for client in self._snapshot_in_game_clients():
            if not self._combat_observer_packets_allowed_for_client(client, *participants):
                continue
            self._send_packet_to_client(
                client,
                pkt,
                prefer_tcp=True,
                allow_udp_fallback=False,
            )

    def _send_entity_create(self, target_ctx: ClientContext, player_ctx: ClientContext, *, is_retry: bool = False) -> None:
        """Announce player_ctx's entity to target_ctx via UPDATE_ARRAY DEFINITION.

        Remote entities are created via UPDATE_ARRAY's fallback path:
          1. OIDTable_lookup(entity_id) â†’ NOT FOUND
          2. Entity_create_from_network() creates entity + sets team/config
          3. Entity_apply_network_transform: entity+0xAC=0 â†’ copies position â†’
             Entity_toggle_static_state() â†’ Entity_set_transform() â†’ visible!

        The client's Replication.c has a global tick guard
        (g_tick_count < g_next_periodic_send_tick) that may block
        Entity_apply_network_transform for newly connected clients.
        To work around this, we resend entity creation packets for a
        retry period after the first attempt, so that one eventually
        arrives when the guard is open.
        """
        if not self._og_viewer_replication_enabled(target_ctx, "entity_create"):
            return
        if not target_ctx.session.translation_ack_received:
            return
        entity_id = player_ctx.session.entity_id or player_ctx.entity_id

        # Track entity creation timing for retry logic.
        if not hasattr(target_ctx, '_entity_create_times'):
            target_ctx._entity_create_times = {}  # entity_id â†’ (first_send_time, last_send_time)

        now = time.monotonic()
        # Fast retries for the first N seconds, then slow periodic re-announce.
        # The client's tick guard (g_tick_count < g_next_periodic_send_tick) can
        # block Entity_apply_network_transform indefinitely.  DEFINITION packets
        # bypass the guard via Entity_create_from_network (entity+0xAC=0 path),
        # so periodic re-announce keeps entities visible even after the guard
        # activates.
        fast_window = float(os.environ.get("WULFRAM_ENTITY_CREATE_RETRY_SECS", "10"))
        fast_interval = float(os.environ.get("WULFRAM_ENTITY_CREATE_RETRY_INTERVAL", "0.5"))
        slow_interval = float(os.environ.get("WULFRAM_ENTITY_REANNOUNCE_INTERVAL", "5.0"))

        if entity_id in target_ctx.known_entity_ids:
            # Already sent at least once.  Check if we should retry.
            times = target_ctx._entity_create_times.get(entity_id)
            if times is None:
                return  # No tracking info â€” legacy, don't retry.
            first_send, last_send = times
            in_fast_window = (now - first_send) <= fast_window
            if not in_fast_window:
                return  # Past fast window â€” entity is created, UPDATE packets handle sync.
                # NOTE: Slow re-announce was REMOVED because build_update_array_create_tank
                # sends hardcoded rotation=(0,0,0) and no velocity, causing the remote entity
                # to visually snap to heading=0 every re-announce interval (glitching).
            if (now - last_send) < fast_interval:
                return  # Too soon since last retry.
            is_retry = True

        pos = self._to_client_pos(player_ctx.player_pos)
        team = player_ctx.session.team_id or 1
        tick = self._get_network_tick(target_ctx)
        rot = self._player_body_rotation(
            player_ctx,
            negate_yaw=self.remote_yaw_negate,
            yaw_offset=self.remote_yaw_offset,
        )

        include_local_state, ls = self._get_update_array_local_state_for_viewer(target_ctx)
        create_pkt = build_update_array_create_tank(
            tick=tick,
            entity_id=entity_id,
            entity_type=player_ctx.entity_type,
            team=team,
            pos=pos,
            is_manned=True,
            rot=rot,
            include_health=include_local_state,
            **ls,
        )
        label = "RETRY" if is_retry else "CREATE"
        print(f"[MULTI] UPDATE_ARRAY DEFINITION {label} id={entity_id} "
              f"type={player_ctx.entity_type} team={team} pos={pos} "
              f"tick={tick} -> client {target_ctx.client_id}")
        if not is_retry:
            print(f"[MULTI-HEX] {create_pkt.hex().upper()}")
        # OG clients are sensitive to UPDATE_ARRAY over TCP. Remote entity
        # creation already has UDP retries, so keep this path UDP-only.
        if not self._send_packet_to_client(target_ctx, create_pkt, prefer_tcp=False):
            return

        if entity_id not in target_ctx.known_entity_ids:
            target_ctx.known_entity_ids.add(entity_id)
            target_ctx._entity_create_times[entity_id] = (now, now)
            print(f"[MULTI] Sent entity create OK -> client {target_ctx.client_id}")
        else:
            first_send = target_ctx._entity_create_times[entity_id][0]
            target_ctx._entity_create_times[entity_id] = (first_send, now)
            elapsed = now - first_send
            phase = "RETRY" if elapsed <= fast_window else "REANNOUNCE"
            if phase == "RETRY" or int(elapsed) % 30 == 0:  # Log retries always, re-announces every 30s
                print(f"[MULTI] Sent entity create {phase} -> client {target_ctx.client_id} ({elapsed:.1f}s since first)")

    def _ensure_multiplayer_visibility(self, ctx: ClientContext) -> None:
        """Ensure ctx sees other players and vice versa once translation is ready."""
        if not ctx.session.translation_ack_received:
            return
        self._ensure_uplink_mvp_state(ctx)
        others = [c for c in self._snapshot_in_game_clients() if c is not ctx]
        if ctx.session.tick % 300 == 0 and others:
            print(f"[MULTI-DBG] client={ctx.client_id} trans_ack={ctx.session.translation_ack_received} "
                  f"others={[(o.client_id, o.session.entity_id or o.entity_id, o.session.translation_ack_received) for o in others]} "
                  f"known={ctx.known_entity_ids}")
        for other in others:
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
        if not self._og_viewer_replication_enabled(ctx, "remote_updates"):
            return
        mode = self.remote_update_mode
        if mode in ("off", "none", "disabled"):
            return
        include_pos = mode not in ("heartbeat", "mask0")
        include_vel = mode in ("pos_vel", "pos_vel_rot", "full", "all")
        include_rot = mode in ("pos_rot", "pos_vel_rot", "full", "all")
        viewer_fuel = self._get_energy_value(ctx)
        others = self._snapshot_in_game_clients()
        if tick % 300 == 0:
            other_ids = [(c.client_id, c.session.entity_id or c.entity_id) for c in others if c is not ctx]
            print(f"[REMOTE-DBG] client={ctx.client_id} tick={tick} mode={mode} "
                  f"others={other_ids} known={ctx.known_entity_ids}")
        for other in others:
            if other is ctx:
                continue
            entity_id = other.session.entity_id or other.entity_id
            if entity_id not in ctx.known_entity_ids:
                continue
            health_val = self._get_health_value(ctx)
            include_local_state, local_state_kwargs = self._get_update_array_local_state_for_viewer(ctx)
            send_pos = self._to_client_pos(other.player_pos)
            payload = build_update_array_player_update(
                tick,
                entity_id,
                pos=send_pos,
                vel=other.player_vel,
                rot=(
                    other.player_pose.get("roll", 0.0),
                    other.player_pose.get("pitch", 0.0),
                    (-other.player_heading if self.remote_yaw_negate else other.player_heading) + self.remote_yaw_offset,
                ),
                include_pos=include_pos,
                include_vel=include_vel,
                include_rot=include_rot,
                include_spin=include_rot,
                spin=(0.0, 0.0, other.angular_vel_yaw),
                include_local_state=include_local_state,
                include_entity_vitals=False,
                is_manned=True,
                **local_state_kwargs,
            )
            ok = self._send_packet_to_client(ctx, payload, prefer_tcp=prefer_tcp)
            if tick % 300 == 0:
                print(f"[REMOTE-DBG] Sent entity={entity_id} -> client={ctx.client_id} "
                      f"pos={send_pos} is_manned=True mode={mode} ok={ok}")

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
        init_pos = self._resolve_spawn_pos(0, allow_map_spawn=False)

        # Create context
        ctx = ClientContext(
            client_id=client_id,
            client_addr=client_addr,
            session=session,
            entity_id=entity_id,
            player_pos=init_pos,
        )
        ctx.entity_type = 0
        ctx.player_energy = self.player_energy_max
        ctx.vehicle_physics.damp_coeff = self.damp_coeff

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
        ctx.jump_jet_system.enabled = self.jump_jets_enabled
        ctx.jump_jet_system.debug = False
        ctx.jump_jet_system.on_jump = lambda pid, imp, vel: self._on_jump_jet_triggered(ctx, pid, imp, vel)
        self._reset_jump_jet_state(ctx)

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
        self.control_server.udp_handler = self.udp_handler
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
            if self.debug_udp_raw and len(data) > 5 and 0x09 in data:
                print(f"[UDP-RAW] Datagram with 0x09: len={len(data)} data={data[:40].hex()}")
            # Debug: log raw datagrams that might contain VIEWPOINT_INFO (0x35)
            if self.debug_udp_raw and 0x35 in data:
                indices = [i for i, b in enumerate(data) if b == 0x35]
                if indices:
                    idx = indices[0]
                    start = max(0, idx - 8)
                    end = min(len(data), idx + 24)
                    snippet = data[start:end].hex()
                    print(f"[UDP-RAW] Datagram with 0x35: len={len(data)} idxs={indices} snippet={snippet}")
                    print(f"[UDP-RAW] Datagram with 0x35 full={data.hex()}")
            # If 0x35 appears inside a 0x10 wrapper, dump full hex for analysis.
            if self.debug_udp_raw and data[0] == 0x10 and 0x35 in data:
                print(f"[UDP-RAW] 0x10 wrapper: len={len(data)} hex={data.hex()}")

            # Parse multiple packets from a single UDP datagram. OG can batch a
            # STATE_REQUEST before the ACTION_UPDATE/ACTION_DUMP that makes W/A/D
            # active; pre-scan the whole datagram so correction suppression sees
            # that input before the first state-sync packet is handled.
            packets = list(self._parse_udp_datagram(data, ctx))
            if ctx is not None:
                ctx._datagram_active_movement_input = (
                    self._udp_packets_have_active_movement_input(ctx, packets)
                )
            try:
                for packet in packets:
                    self._handle_single_udp_packet(ctx, packet, addr)
            finally:
                if ctx is not None:
                    ctx._datagram_active_movement_input = False

    def _udp_packets_have_active_movement_input(
        self,
        ctx: ClientContext,
        packets: list[bytes],
    ) -> bool:
        """Return true if a batched UDP datagram carries active fwd/strafe input."""
        if handlers._is_loopback_client(ctx):
            return False
        if not getattr(ctx, "weapon_system", None):
            return False

        def _clone_weapon_system_for_decode() -> WeaponSystem:
            source = ctx.weapon_system
            clone = WeaponSystem()
            clone.behavior_slots = source.behavior_slots.copy()
            for attr in (
                "control_bits",
                "control_max",
                "control_range",
                "zoom_bits",
                "zoom_max",
                "zoom_range",
                "slot_index_bits",
                "weapon_from_slot4",
            ):
                setattr(clone, attr, getattr(source, attr))
            return clone

        for packet in packets:
            if not packet:
                continue
            pkt_type = packet[0]
            if pkt_type not in (0x09, 0x0A):
                continue
            probe = _clone_weapon_system_for_decode()
            if pkt_type == 0x09:
                decoded = probe.decode_action_dump(packet)
            else:
                decoded = probe.decode_action_update(packet)
            if not decoded:
                continue
            fwd = self._normalize_behavior_axis_value(
                ctx,
                probe.behavior_slots[BehaviorSlot.MOVING_FORWARD],
            )
            if abs(fwd) > 0.05:
                return True
            strafe = self._decode_network_strafe_input(
                ctx,
                probe.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS],
            )
            if abs(strafe) > 0.05:
                return True
        return False

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
        if not self.server_ping_loop_enabled:
            return
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
                control_bits = int(os.environ.get("WULFRAM_CONTROL_BITS", "16"))
                zoom_bits = int(os.environ.get("WULFRAM_ZOOM_BITS", str(control_bits)))

            bits = 64
            for slot_idx in range(1, 22):
                if slot_idx == BehaviorSlot.UPWARD_THRUST:
                    bits += zoom_bits
                elif slot_idx in ACTION_DUMP_CONTROL_SLOTS:
                    bits += control_bits
                else:
                    bits += 1
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
                    elif slot_idx in ACTION_ANALOG_SLOTS:
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
            if self.debug_viewpoint:
                print(f"[VIEWPOINT-EXTRACT] payload at offset {pos} len={len(pkt)}")
            yield pkt

        cursor = 0
        while cursor < len(data):
            if cursor >= len(data):
                break

            pkt_type = data[cursor]

            # D_ACK / control acknowledgments.
            # Common wire forms from decompile + captures:
            # - 0x02 0x00 <u32 timestamp>                     (6 bytes)
            # - 0x02 0x01 <u8 channel> <u16 seq>             (5 bytes)
            # - 0x02 0x02 <u32 timestamp> <u8 ch> <u16 seq>  (9 bytes)
            # Legacy server frame used by _send_udp_ack:
            # - 0x02 <u16 our_seq> 0x0009 <u8 sub> <u8 pkt> <u16 seq> (9 bytes)
            if pkt_type == 0x02:
                remaining = len(data) - cursor

                # D_ACK subtype 1 (channel + seq)
                if remaining >= 5 and data[cursor + 1] == 0x01:
                    yield data[cursor:cursor + 5]
                    cursor += 5
                    continue

                # D_ACK subtype 0 (timestamp-only) OR legacy 9-byte server ACK frame.
                if remaining >= 6 and data[cursor + 1] == 0x00:
                    if remaining >= 9 and data[cursor + 3:cursor + 5] == b"\x00\x09":
                        yield data[cursor:cursor + 9]
                        cursor += 9
                        continue
                    yield data[cursor:cursor + 6]
                    cursor += 6
                    continue

                # D_ACK subtype 2 (timestamp + channel + seq)
                if remaining >= 9 and data[cursor + 1] == 0x02:
                    yield data[cursor:cursor + 9]
                    cursor += 9
                    continue

                # Fallback for older/unknown framing seen in captures.
                # Keep the previous +10 scan as a last resort to salvage
                # inner packets for analysis, then surface the raw frame.
                salvaged = False
                if remaining >= 11:
                    payload = data[cursor + 10:]
                    if payload and any(payload):
                        for inner in self._parse_udp_datagram(payload, ctx):
                            salvaged = True
                            yield inner
                if not salvaged:
                    yield data[cursor:]
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
                            if self.debug_viewpoint:
                                print(f"[0x10-SCAN] Found 0x35 at offset {pos}: ...{context}...")
                            if pos + 5 <= len(raw):
                                potential_pkt = raw[pos:]
                                if len(potential_pkt) >= 5 and potential_pkt[3:5] == b"\x00\x14":
                                    pkt_len = 20
                                    if pos + pkt_len <= len(raw):
                                        if self.debug_viewpoint:
                                            print(f"[0x10-EXTRACT] Extracting 0x35 at offset {pos}, declared len={pkt_len}")
                                        yield potential_pkt[:pkt_len]

                # Consume entire 0x10 packet
                break

            # Reliable stream packets with length at bytes 3-4
            if pkt_type in (0x20, 0x25, 0x26, 0x2B, 0x33, 0x35, 0x3A, 0x3B):
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

            elif pkt_type == 0x04:  # D_SET_START - channel(u8) + sequence(u16)
                pkt_len = min(4, len(data) - cursor)
                yield data[cursor:cursor+pkt_len]
                cursor += pkt_len

            elif pkt_type == 0x0B:  # PING_REQUEST - opcode + u32 request_id + u32 frame_count
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
                    while end < len(data) and data[end] not in (
                        0x02, 0x03, 0x04, 0x09, 0x0A, 0x0B, 0x0C, 0x10,
                        0x20, 0x25, 0x26, 0x2B, 0x2E, 0x33, 0x35, 0x3A, 0x3B, 0x40, 0x49,
                    ):
                        end += 1
                    yield data[cursor:end]
                    cursor = end

            else:
                # Unknown packet - consume rest of datagram
                yield data[cursor:]
                break

    def _handle_udp_d_ack(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle D_ACK/service-layer ACK packet (0x02)."""
        if len(data) < 2:
            return

        # Legacy ACK frame used by our reliable helpers:
        # 0x02 + our_seq(u16) + len(u16=9) + subcmd(u8) + packet_id(u8) + seq(u16)
        if len(data) >= 9 and data[3:5] == b"\x00\x09":
            our_seq = struct.unpack(">H", data[1:3])[0]
            subcmd = data[5]
            packet_id = data[6]
            seq_num = struct.unpack(">H", data[7:9])[0]
            if self.debug_udp_raw:
                print(
                    f"[UDP] ACK_FRAME 0x02 from {addr}: "
                    f"our_seq={our_seq} sub={subcmd} packet=0x{packet_id:02X} seq={seq_num}"
                )
            return

        subtype = data[1]
        if subtype == 0x00 and len(data) >= 6:
            timestamp = struct.unpack(">I", data[2:6])[0]
            if self.debug_udp_raw:
                print(f"[UDP] D_ACK subtype=0 ts={timestamp} from {addr}")
            return

        if subtype == 0x01 and len(data) >= 5:
            channel = data[2]
            seq_num = struct.unpack(">H", data[3:5])[0]
            if self.debug_udp_raw:
                print(f"[UDP] D_ACK subtype=1 channel={channel} seq={seq_num} from {addr}")
            return

        if subtype == 0x02 and len(data) >= 9:
            timestamp = struct.unpack(">I", data[2:6])[0]
            channel = data[6]
            seq_num = struct.unpack(">H", data[7:9])[0]
            if self.debug_udp_raw:
                print(
                    f"[UDP] D_ACK subtype=2 ts={timestamp} "
                    f"channel={channel} seq={seq_num} from {addr}"
                )
            return

        if self.debug_udp_raw:
            print(f"[UDP] D_ACK malformed/unknown from {addr}: len={len(data)} data={data.hex()}")

    def _handle_udp_d_set_start(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle D_SET_START stream control packet (0x04)."""
        if len(data) < 4:
            if self.debug_udp_raw:
                print(f"[UDP] D_SET_START malformed from {addr}: len={len(data)} data={data.hex()}")
            return

        channel = data[1]
        seq_num = struct.unpack(">H", data[2:4])[0]

        # Mirror service-layer behavior: acknowledge with D_ACK subtype 1.
        if self.udp_handler:
            ack = bytes((0x02, 0x01, channel)) + struct.pack(">H", seq_num)
            self.udp_handler.send_to(ack, addr)

        if self.debug_udp_raw:
            print(f"[UDP] D_SET_START channel={channel} seq={seq_num} from {addr}")

    def _handle_single_udp_packet(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle a single UDP packet (after parsing from datagram)."""
        if not data:
            return

        pkt_type = data[0]

        # Update ctx if we found it, or try to find it from addr
        if ctx is None:
            ctx = self.udp_addr_to_client.get(addr)
        if ctx is None and pkt_type not in (0x03, 0x08, 0x13):
            ctx = self._recover_udp_client(addr)

        # For some packet types (HELLO, D_HANDSHAKE), we may not have ctx yet
        # These packets help identify/register the client

        # Diagnostic: log reliable stream packets when debugging packet flow.
        if (
            pkt_type in (0x33, 0x35, 0x25, 0x20, 0x3a)
            and (self.debug_udp_raw or (self.debug_viewpoint and pkt_type == 0x35))
        ):
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
                    self._handle_udp_d_ack(ctx, data[1:], addr)
                elif d_type == 0x04:
                    self._handle_udp_d_set_start(ctx, data[1:], addr)
                else:
                    print(f"[UDP] D_Protocol type=0x{d_type:02X} from {addr} (len={len(data)})")
            else:
                # HELLO_ACK - this registers the UDP address for a client
                text = data[1:].decode('ascii', errors='ignore').strip('\x00')
                print(f"[UDP] HELLO_ACK from {addr}: '{text}'")

                # Try session-key-based match first (deterministic)
                if ctx is None and text:
                    matched = self.session_key_to_client.get(text)
                    if matched:
                        ctx = matched
                        print(f"[UDP] HELLO_ACK matched by session key -> client {ctx.client_id}")

                # Fallback: find a client waiting for UDP verification
                if ctx is None:
                    from .session import Phase
                    with self.clients_lock:
                        for c in self.clients.values():
                            if c.session and c.session.phase == Phase.HANDSHAKE and not c.session.udp_verified:
                                ctx = c
                                print(f"[UDP] Matched HELLO_ACK to client {ctx.client_id} in HANDSHAKE state (heuristic)")
                                break

                if ctx:
                    self._bind_udp_client(ctx, addr, reason="udp_hello_ack")

        elif pkt_type == 0x13:
            # Session key â€” use to deterministically bind UDP addr â†’ client
            # Format: 0x13 + subcmd(1) + length(2) + key_bytes
            if len(data) > 4:
                key = data[4:].decode('ascii', errors='ignore').strip('\x00')
                matched = self.session_key_to_client.get(key)
                if matched:
                    if self._bind_udp_client(matched, addr, reason=f"session_key '{key}'"):
                        ctx = matched
                else:
                    print(f"[UDP] Session key from {addr}: '{key}' (NO MATCH)")

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
            if (
                ctx
                and ctx.session.pending_spawn_points
                and not ctx.session.spawn_points_sent
                and ctx.session.udp_d_handshake_received
            ):
                handlers._send_spawn_points_for_client(self, ctx)

        elif pkt_type == 0x03:
            # D_HANDSHAKE (Wulf-Forge UDP stream init)
            self._handle_udp_d_handshake(ctx, data, addr)

        elif pkt_type == 0x02:
            # D_ACK / reliable ACK control frame
            self._handle_udp_d_ack(ctx, data, addr)

        elif pkt_type == 0x04:
            # D_SET_START stream sequence control
            self._handle_udp_d_set_start(ctx, data, addr)

        elif pkt_type == 0x20:
            # COMM_REQ (chat/system commands) - used by Wulf-Forge for /s spawn
            self._handle_udp_chat(ctx, data, addr)

        elif pkt_type == 0x25:
            # REINCARNATE over UDP (Wulf-Forge style)
            self._handle_udp_reincarnate(ctx, data, addr)

        elif pkt_type == 0x26:
            # RETARGET (reliable stream): acknowledge to prevent resend storms.
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x26, seq_num)
                if self.debug_udp_raw:
                    print(f"[UDP] RETARGET seq={seq_num} from {addr}")

        elif pkt_type == 0x35:
            # VIEWPOINT_INFO - client sends camera/view position and orientation
            # This is the ACTUAL player pose, not reconstructed from inputs!
            if self.debug_viewpoint:
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

        elif pkt_type == 0x3B:
            # BEACON_MODIFY (reliable stream): acknowledge.
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x3B, seq_num)
                if self.debug_udp_raw:
                    print(f"[UDP] BEACON_MODIFY seq={seq_num} from {addr}")

        elif pkt_type == 0x2B:
            # DROP_REQUEST (reliable stream): acknowledge.
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x2B, seq_num)
                if self.debug_udp_raw:
                    print(f"[UDP] DROP_REQUEST seq={seq_num} from {addr}")

        elif pkt_type == 0x09:
            # ACTION_DUMP - full behavior slot dump (includes fire state)
            if ctx is None:
                ctx = self._ghost_rejoin(addr)
            if self.debug_sync:
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

        elif pkt_type == 0x0B:
            # Client ping request: reply on 0x0C with the same request payload.
            if len(data) >= 9 and self.udp_handler and ctx and self._udp_ping_reply_allowed_for_client(ctx):
                request_id, frame_count = struct.unpack(">II", data[1:9])
                self.udp_handler.send_to(build_ping_reply(request_id), addr)
                print(
                    f"[UDP] PING_REPLY 0x0C to {addr} "
                    f"request_id={request_id} frame_count={frame_count}"
                )
            elif len(data) >= 5 and self.udp_handler and ctx and self._udp_ping_reply_allowed_for_client(ctx):
                # Keep a compatibility fallback for older simplified clients.
                request_id = struct.unpack(">I", data[1:5])[0]
                self.udp_handler.send_to(build_ping_reply(request_id), addr)
                print(f"[UDP] PING_REPLY 0x0C to {addr} request_id={request_id} frame_count=0")

        elif pkt_type == 0x0C:
            # STATE_REQUEST - may contain state/position info
            if self.debug_sync:
                print(f"[UDP] STATE_REQUEST 0x0C len={len(data)} data={data.hex()}")
            self._handle_state_request(ctx, data, addr)

        else:
            print(f"[UDP] Packet 0x{pkt_type:02X} from {addr} (len={len(data)})")

        # Track last UDP activity for mapped clients regardless of packet type.
        if ctx:
            ctx.session.last_udp_activity = time.monotonic()

    def _bind_udp_client(self, ctx: ClientContext, addr: tuple, *, reason: str) -> bool:
        """Bind a UDP endpoint to a client without stealing another active mapping."""
        current = self.udp_addr_to_client.get(addr)
        if current is not None and current is not ctx:
            print(
                f"[UDP] Refusing to bind {addr} to client {ctx.client_id}: "
                f"already owned by client {current.client_id} ({reason})"
            )
            return False

        old_addr = ctx.session.udp_addr
        if old_addr and old_addr != addr and self.udp_addr_to_client.get(old_addr) is ctx:
            del self.udp_addr_to_client[old_addr]

        ctx.session.udp_addr = addr
        ctx.session.udp_verified = True
        ctx.session.last_udp_activity = time.monotonic()
        self.udp_addr_to_client[addr] = ctx
        print(f"[UDP] Bound client {ctx.client_id} to {addr} via {reason}")
        return True

    def _recover_udp_client(self, addr: tuple, *, allow_handshake: bool = False) -> Optional[ClientContext]:
        """
        Recover a missing UDP->client mapping when the client reconnects/rebinds.
        This is a safe fallback only: ambiguous same-host candidates are rejected.
        """
        existing = self.udp_addr_to_client.get(addr)
        if existing:
            return existing

        ip = addr[0] if addr else None
        if not ip:
            return None

        candidates: list[ClientContext] = []
        with self.clients_lock:
            for c in self.clients.values():
                if not c or not c.running or not c.session:
                    continue
                if not c.client_addr or c.client_addr[0] != ip:
                    continue
                if c.session.phase == Phase.DISCONNECTED:
                    continue
                if c.session.phase == Phase.HANDSHAKE:
                    if not allow_handshake:
                        continue
                    if c.session.udp_verified or c.session.udp_config_sent_time <= 0.0:
                        continue
                # Skip clients that already have a verified UDP address on a
                # different port -- recovering them would steal the mapping and
                # break the other client's UDP routing (localhost two-client bug).
                if (c.session.udp_verified and c.session.udp_addr
                        and c.session.udp_addr != addr
                        and c.session.udp_addr in self.udp_addr_to_client):
                    continue
                candidates.append(c)

        if not candidates:
            return None
        if len(candidates) != 1:
            cand_ids = ",".join(str(c.client_id) for c in candidates)
            print(f"[UDP] Recover addr {addr}: ambiguous candidates [{cand_ids}]")
            return None

        ctx = candidates[0]
        if self._bind_udp_client(ctx, addr, reason="safe_recovery"):
            return ctx
        return None

    def _ghost_rejoin(self, addr: tuple) -> Optional[ClientContext]:
        """Create a headless client from orphan UDP traffic (VM snapshot restore).

        When ghost_rejoin is enabled and we see game traffic (ACTION_DUMP) from
        an unknown address, create a ClientContext with no TCP, spawn via UDP,
        and start the tick loop.  This lets us restore a VM snapshot and have
        the client seamlessly reconnect without the full login flow.
        """
        if not self.ghost_rejoin:
            return None
        try:
            if ipaddress.ip_address(addr[0]).is_loopback:
                # Loopback/Python probes should use the normal TCP bootstrap.
                # Creating ghost localhost clients lets stale probe sockets
                # reuse the same entity id and poison local authoritative sync.
                return None
        except ValueError:
            pass
        if addr in self._ghost_rejoin_attempted:
            return None
        self._ghost_rejoin_attempted.add(addr)

        print(f"[GHOST] Attempting ghost rejoin for {addr}")

        # Create a minimal session in IN_GAME state.
        session = Session()
        session.phase = Phase.IN_GAME
        session.username = "ghost"
        session.login_complete = True
        session.in_game = True
        session.udp_addr = addr
        session.udp_verified = True
        session.translation_ack_received = True  # Skip translation wait
        session.roster_sent = True  # Skip roster send (needs TCP)
        session.behavior_sent = True
        session.translation_sent = True
        session.want_updates_received = True
        session.want_updates_handled = True

        client_id = self.next_client_id
        self.next_client_id += 1
        entity_id = 1337  # Same as normal to match client's cached OID
        session.player_id = entity_id
        session.entity_id = entity_id
        session.team_id = 1  # Red team

        ctx = ClientContext(
            client_id=client_id,
            client_addr=(addr[0], 0),
            session=session,
            entity_id=entity_id,
        )
        ctx.tcp_handler = None  # No TCP for ghost clients
        ctx.player_energy = self.player_energy_max
        ctx.vehicle_physics.damp_coeff = self.damp_coeff

        # Set up weapon/jump systems.
        ctx.weapon_system = WeaponSystem()
        ctx.weapon_system.next_entity_id = max(ctx.weapon_system.next_entity_id, 20000 + (client_id * 1000))
        ctx.weapon_system.on_chain_gun_fire = lambda pos, rot, team, name=None: self._on_chain_gun_fire(ctx, pos, rot, team, name)
        ctx.weapon_system.on_projectile_spawn = lambda proj: self._on_projectile_spawn(ctx, proj)
        ctx.jump_jet_system = JumpJetSystem()
        ctx.jump_jet_system.enabled = self.jump_jets_enabled
        ctx.jump_jet_system.debug = False
        ctx.jump_jet_system.on_jump = lambda pid, imp, vel: self._on_jump_jet_triggered(ctx, pid, imp, vel)
        self._reset_jump_jet_state(ctx)

        # Register.
        with self.clients_lock:
            self.clients[client_id] = ctx
        self.udp_addr_to_client[addr] = ctx

        # Pick spawn point (same logic as normal spawn).
        spawn_pos = self._resolve_spawn_pos(session.team_id or 2)

        ctx.player_pos = spawn_pos
        ctx.player_pose["pos"] = spawn_pos
        ctx.world_collision_ref_pos = spawn_pos
        ctx.world_collision_bounds_dirty = False
        ctx.last_state_sync_vel = None
        ctx.last_state_sync_rot = None
        ctx.last_correction_send = 0.0
        ctx.force_correction_once = False
        ctx.authoritative_state_history.clear()
        ctx.player_yaw = 0.0
        ctx.player_heading = 0.0
        ctx.angular_vel_yaw = 0.0
        ctx.spring_body_ang_vel = (0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)
        ctx.player_pose["roll"] = 0.0
        ctx.player_pose["pitch"] = 0.0
        ctx.player_pose["yaw"] = 0.0
        ctx.player_speed = 0.0
        ctx.vehicle_physics.reset()
        self._set_ground_level_override_for_pose(ctx, spawn_pos)

        print(f"[GHOST] Created ghost client {client_id} entity={entity_id} pos={spawn_pos}")

        # Send spawn packets via UDP (skip all TCP-dependent parts).
        if self.udp_handler:
            send_pos = self._to_client_pos(spawn_pos)
            ls_weapon = self._get_local_state_weapon_type(ctx)
            spawn_tank_weapon = self._get_spawn_tank_weapon_type(ctx)
            health_val = self._get_health_value(ctx)

            # Pre-creation UPDATE_ARRAY.
            ls = self._get_local_state_kwargs(ctx)
            ua_packet = build_update_array_create_tank(
                tick=self._get_network_tick(ctx),
                entity_id=entity_id,
                entity_type=0,
                team=session.team_id,
                pos=send_pos,
                behavior_type=0,
                include_health=self.update_local_state,
                include_entity_vitals=self.update_entity_vitals,
                is_manned=True,
                **ls,
            )
            self.udp_handler.send_to(ua_packet, addr)
            time.sleep(0.20)

            # TankPacket.
            tank_packet = build_udp_tank_packet_wf(
                net_id=entity_id,
                unit_type=0,
                team_id=session.team_id,
                pos=send_pos,
                rot=self._local_player_sync_rotation(ctx),
                tick=self._get_network_tick(ctx),
                include_vitals=True,
                weapon_id=spawn_tank_weapon,
                health=1.0,
                energy=1.0,
            )
            comm_pkt = build_chat_message("Ghost rejoin", source_id=entity_id)
            self.udp_handler.send_to(comm_pkt, addr)
            self.udp_handler.send_to(tank_packet, addr)
            print(f"[GHOST] Sent spawn packets to {addr}")

            # Heartbeat.
            time.sleep(0.05)
            hb_packet = self._build_local_state_heartbeat(
                ctx,
                tick=self._get_network_tick(ctx),
                entity_id=entity_id,
                include_health=True,
                health=1.0,
                fuel=1.0,
            )
            self.udp_handler.send_to(hb_packet, addr)

        # Start tick loop.
        session.last_spawn_time = time.monotonic()
        ctx.last_action_dump_time = time.monotonic()
        if self._ensure_tick_loop(ctx):
            print(f"[GHOST] Started tick loop for ghost client {client_id}")

        return ctx

    def _handle_udp_d_handshake(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle UDP D_HANDSHAKE (0x03) and respond with stream definitions."""
        handlers.handle_udp_d_handshake(self, ctx, data, addr)

    def _send_udp_ack(self, ctx: Optional[ClientContext], addr: tuple, packet_id: int, seq_num: int, subcmd: int = 1):
        """Send a Wulf-Forge style UDP ACK (0x02)."""
        handlers.send_udp_ack(self, ctx, addr, packet_id, seq_num, subcmd)

    def _send_udp_wf_spawn(self, ctx: ClientContext, addr: tuple, net_id: int, team_id: int,
                            unit_type: int = 0,
                            pos: tuple = (100.0, 100.0, 100.0),
                            rot: tuple = (0.0, 0.0, 0.0)):
        """Send a Wulf-Forge style UDP TANK (0x18) packet to spawn a unit."""
        if not self.udp_handler:
            return
        payload = build_udp_tank_packet_wf(
            net_id=net_id,
            unit_type=unit_type,
            team_id=team_id,
            pos=pos,
            rot=rot,
            include_vitals=self.tank_vitals,
            weapon_id=self.weapon_id,
            health=1.0,
            energy=1.0,
        )
        self._log_vitals(
            ctx,
            "TANK_UDP_SPAWN",
            include_vitals=self.tank_vitals,
            health=1.0,
            energy=1.0,
            weapon_id=self.weapon_id,
            note=f"team={team_id} net_id={net_id}",
        )
        self.udp_handler.send_to(payload, addr)

    def _should_send_spawn_entry_transition(self, ctx: ClientContext) -> bool:
        mode = getattr(self, "spawn_entry_transition", "off")
        if mode in ("1", "true", "on"):
            return True
        if mode in ("0", "false", "off"):
            return False
        return not handlers._is_loopback_client(ctx)

    def _send_spawn_entry_transition(self, ctx: ClientContext, team_id: int, net_id: int) -> None:
        """Reassert the decompile-backed team-entry sequence before OG auto-spawn."""
        if not self._should_send_spawn_entry_transition(ctx):
            return
        if not ctx.tcp_handler:
            return

        session = ctx.session
        session.player_id = session.player_id or net_id
        session.team_id = team_id

        rein = build_reincarnate(0x11, "")
        sent_rein = False
        if self.udp_handler and session.udp_addr:
            try:
                self.udp_handler.send_to(rein, session.udp_addr)
                sent_rein = True
            except Exception as ex:
                print(f"[SPAWN] Failed entry-transition UDP REINCARNATE(0x11): {ex}")
        if not sent_rein:
            ctx.tcp_handler.send(rein)

        ctx.tcp_handler.send(build_player(entity_id=session.player_id, spectator=False))
        ctx.tcp_handler.send(build_game_clock())

        sent_roster = False
        if not session.roster_sent:
            name = session.username or f"Player{ctx.client_id}"
            ctx.tcp_handler.send(build_add_to_roster(
                player_id=session.player_id,
                entity_id=session.player_id,
                name=name,
                team=team_id,
            ))
            session.roster_sent = True
            sent_roster = True

        sent_world_stats = False
        if not session.world_stats_sent:
            ctx.tcp_handler.send(self.build_world_stats_packet())
            session.world_stats_sent = True
            sent_world_stats = True

        print(
            "[SPAWN] Sent entry transition "
            f"(REINCARNATE via {'UDP' if sent_rein else 'TCP'}, PLAYER, GAME_CLOCK"
            f"{', ADD_TO_ROSTER' if sent_roster else ''}"
            f"{', WORLD_STATS' if sent_world_stats else ''})"
        )

    def _spawn_wf_style(self, ctx: ClientContext, team_id: int, net_id: Optional[int] = None,
                         unit_type: int = 0,
                         pos: Optional[tuple] = None,
                         rot: tuple = (0.0, 0.0, 0.0),
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

        # Reset position tracking to spawn location.
        spawn_pos = self._resolve_spawn_pos(team_id, explicit_pos=pos)
        if pos is not None:
            print(f"[SPAWN] Using explicit spawn pos={spawn_pos}")
        else:
            configured_default = self._get_configured_default_spawn_pos()
            if configured_default is not None:
                print(f"[SPAWN] Using default flat spawn pos={spawn_pos}")
            else:
                map_spawn = self._pick_spawn_point(team_id)
                if map_spawn:
                    print(
                        f"[SPAWN] Auto-selected map spawn oid={map_spawn['oid']} "
                        f"team={team_id} pos={spawn_pos}"
                    )

        # Offset spawn to avoid overlapping tanks in multi-client tests. Do not
        # move explicit map-flag spawns by default: the offset can put a clicked
        # repair pad spawn onto unrelated steep terrain, which turns later
        # targeted corrections into bogus authoritative snaps.
        offset_explicit_spawns = (
            os.environ.get("WULFRAM_MULTI_SPAWN_OFFSET_EXPLICIT", "0")
            .strip()
            .lower()
            in ("1", "true", "on", "yes")
        )
        apply_spawn_offset = bool(self.multi_spawn_offset) and (
            pos is None or offset_explicit_spawns
        )
        if apply_spawn_offset:
            # Use 0-based index among active clients (not client_id, which grows unboundedly).
            with self.clients_lock:
                active_ids = sorted(c.client_id for c in self.clients.values() if c and c.running)
            try:
                idx = active_ids.index(ctx.client_id)
            except ValueError:
                idx = 0
            spawn_pos = (spawn_pos[0] + idx * self.multi_spawn_offset, spawn_pos[1], spawn_pos[2])

        # Optional legacy/debug path: force the spawn anchor onto the decoded
        # heightmap. Default off because OG live memory matches raw map-state
        # repair-pad Z more closely than the currently decoded terrain height.
        if self.terrain and self.up_axis == "z" and self.align_spawn_pos_to_terrain:
            terrain_z = (
                self.terrain.get_height(spawn_pos[0], spawn_pos[1])
                + self.terrain_height_offset
            )
            spawn_pos = (spawn_pos[0], spawn_pos[1], terrain_z)

        ctx.player_pos = spawn_pos
        ctx.player_pose["pos"] = spawn_pos
        ctx.world_collision_ref_pos = spawn_pos
        ctx.world_collision_bounds_dirty = False
        ctx.last_state_sync_vel = None
        ctx.last_state_sync_rot = None
        ctx.last_correction_send = 0.0
        ctx.force_correction_once = False
        ctx.authoritative_state_history.clear()

        ctx.player_health = 1.0
        ctx.player_angular_vel = 0.0
        ctx.player_speed = 0.0
        ctx.vehicle_physics.reset()
        # Reset tick-sync transition tracking so respawn gap doesn't cause huge correction
        ctx._turn_transition_client_tick = 0
        ctx._turn_transition_server_tick = 0

        # Spawn heading â€” can't set local client heading (physics overwrites it),
        # but this affects how OTHER clients see your tank at spawn.
        import math
        spawn_heading_env = os.environ.get("WULFRAM_SPAWN_HEADING")
        spawn_yaw = math.radians(float(spawn_heading_env)) if spawn_heading_env else 0.0
        ctx.player_yaw = -spawn_yaw
        ctx.player_heading = spawn_yaw
        ctx.angular_vel_yaw = 0.0
        ctx.spring_body_ang_vel = (0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, spawn_yaw)
        ctx.player_pose["roll"] = 0.0
        ctx.player_pose["pitch"] = 0.0
        ctx.player_pose["yaw"] = -spawn_yaw
        ctx.player_energy = self.player_energy_max
        ctx.vehicle_physics.heading = spawn_yaw
        self._reset_jump_jet_state(ctx)
        ctx.last_action_dump_time = time.monotonic()  # Reset timer for position tracking
        self._set_ground_level_override_for_pose(ctx, spawn_pos)

        print(f"[SPAWN] Wulf-Forge style: client={ctx.client_id} net_id={net_id} team={team_id} pos={spawn_pos}")

        if not ctx.tcp_handler:
            print("[SPAWN] ERROR: No TCP handler")
            return

        self._send_spawn_entry_transition(ctx, team_id=team_id, net_id=net_id)

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

        # Ensure TRANSLATION has been applied before sending TankPacket with vitals.
        if not ctx.session.translation_ack_received:
            wait_until = time.monotonic() + 2.0
            while not ctx.session.translation_ack_received and time.monotonic() < wait_until:
                time.sleep(0.05)
            if not ctx.session.translation_ack_received:
                print("[SPAWN] WARNING: TRANSLATION_ACK not received before TankPacket")

        # vitals=1 on the FIRST TankPacket â€” this writes health=1.0 into
        # the ESI buffer during PLAYER_INFO's read_local_player_state.  The
        # subsequent sync_local_player (the ONLY call that passes the tick
        # guard, because Mode3_init_descriptor â†’ operator_new returns a heap
        # pointer that gets stored in g_last_input_apply_tick, permanently
        # blocking all future heartbeat sync_local_player calls) then writes
        # ESI[0]=1.0 to the HUD health meter.
        #
        # NOTE: previous test showed vitals=1 caused "entry map persistence"
        # but that was WITHOUT UPDATE_ARRAY pre-creation.  With pre-creation,
        # PLAYER_INFO takes the "entity found" code path (LocalPlayer_initialize
        # in the else branch) instead of Entity_create_from_network.
        include_spawn_vitals = True
        # Force the initial spawn packets onto the short-form-safe remote path.
        # If this client had already reached promoted remote sync before a
        # respawn, restore that mode after the fresh entity exists again.
        restore_promoted_remote_sync_after_spawn = (
            self.update_local_state_mode == "wf"
            and not handlers._is_loopback_client(ctx)
            and bool(getattr(ctx, "remote_full_local_state_ready", False))
        )
        ctx.last_client_tick = 0
        ctx.remote_full_local_state_ready = False
        ctx._spawn_safe_heartbeat_suppressed_logged = False
        # Set last_spawn_time BEFORE sending TankPacket so the 2-second
        # jump suppression window covers the entire retransmit period.
        # Without this, a jump ACTION_UPDATE arriving during retransmit
        # sleep bypasses suppression and sends a velocity UPDATE_ARRAY
        # that sets prev_health positive before a retransmit re-creates
        # the entity with health=0 â†’ permanent DeathScreen.
        ctx.session.last_spawn_time = time.monotonic()
        if not self.spawn_send_udp_tank:
            print("[SPAWN] Skipping UDP TankPacket (WULFRAM_SPAWN_UDP_TANK=0)")
        else:
            print(f"[SPAWN] Sending UDP TankPacket (vitals={int(include_spawn_vitals)})")
        send_pos = self._to_client_pos(spawn_pos)
        send_rot = self._local_player_sync_rotation(ctx)
        # Local-state updates may use Tank(0), but spawn TankPacket vitals use a
        # parse-safe weapon id because 0x18 writes only weapon+health+energy bits.
        ls_weapon = self._get_local_state_weapon_type(ctx)
        spawn_tank_weapon = self._get_spawn_tank_weapon_type(ctx)
        # Optional: create the entity via UPDATE_ARRAY before PLAYER_INFO.
        if self.spawn_send_update_array:
            health_val = self._get_health_value(ctx)
            fuel_val = self._get_energy_value(ctx)
            ua_packet = build_update_array_create_tank(
                tick=self._get_network_tick(ctx),
                entity_id=net_id,
                entity_type=unit_type,
                team=team_id,
                pos=send_pos,
                behavior_type=0,
                include_health=False,  # local_player_state ammo/turret bits crash client
                include_entity_vitals=self.update_entity_vitals,
                health=health_val,
                fuel=fuel_val,
                is_manned=True,
                weapon_id=ls_weapon,
            )
            if not self._send_spawn_create_update_array(ctx, ua_packet):
                print("[SPAWN] WARNING: No transport available for pre-creation UPDATE_ARRAY")
            # Delay so client processes entity creation before PLAYER_INFO.
            # The TankPacket's OIDTable_lookup must find the entity created
            # by UPDATE_ARRAY; if the TankPacket arrives first, it takes the
            # Entity_create_from_network path which does NOT call
            # LocalPlayer_initialize â†’ entry map stays.  200ms gives ~6
            # frames at 30fps for the client to process the UPDATE_ARRAY.
            time.sleep(0.20)
        spawn_tick = self._get_network_tick(ctx)
        tank_packet = build_udp_tank_packet_wf(
            net_id=net_id,
            unit_type=unit_type,
            team_id=team_id,
            pos=send_pos,
            rot=send_rot,
            tick=spawn_tick,
            include_vitals=include_spawn_vitals,
            weapon_id=spawn_tank_weapon,
            health=1.0,
            energy=1.0,
        )
        self._log_vitals(
            ctx,
            "TANK_UDP_SPAWN",
            include_vitals=include_spawn_vitals,
            health=1.0,
            energy=1.0,
            weapon_id=spawn_tank_weapon,
            note=f"team={team_id} net_id={net_id}",
        )
        # HEX DUMP for comparison with wulf-forge
        print(f"[TANK-HEX] len={len(tank_packet)} hex={tank_packet.hex().upper()}")

        # Send TankPacket over UDP (matching wulf-forge behavior)
        sent_tank_udp = False
        if self.spawn_send_udp_tank:
            if self.udp_handler and ctx.session.udp_addr:
                self.udp_handler.send_to(tank_packet, ctx.session.udp_addr)
                sent_tank_udp = True
                print(f"[SPAWN] Sent UDP TankPacket to {ctx.session.udp_addr}")
                # Optional TCP backup.  After combat kill + entity DELETE,
                # the client may not process UDP while on team-select overlay.
                # With pre-creation UPDATE_ARRAY, entity exists in OIDTable so
                # TCP PLAYER_INFO takes "entity found" path â†’ LocalPlayer_initialize.
                if os.environ.get("WULFRAM_SPAWN_TCP_TANK_BACKUP", "0").strip().lower() not in ("0", "false", "off", "no"):
                    ctx.tcp_handler.send(tank_packet)
                    print(f"[SPAWN] Also sent TCP TankPacket (backup for respawn)")
                # Resend TankPacket for reliability (UDP can drop packets)
                # Each retransmit triggers Entity_create_from_network (new
                # entity, 0xD0=0).  This is safe as long as no velocity
                # UPDATE_ARRAY sets prev_health positive during the
                # retransmit window.  The jump-suppression timer (set
                # before TankPacket send) prevents that race.
                spawn_retransmits = int(os.environ.get("WULFRAM_SPAWN_RETRANSMIT", "0"))
                if spawn_retransmits > 0:
                    # Retransmits DISABLED by default (0).  Each retransmit
                    # calls LocalPlayer_initialize again; if it arrives during
                    # the spawn transition it re-shows the entry map overlay
                    # (race condition: run 20260209_000347 vs _000127).
                    # Wulf-forge sends only one TankPacket, no retransmits.
                    # The pre-creation UPDATE_ARRAY handles entity existence
                    # and the first TankPacket is reliably received on
                    # localhost.  Keep env var for optional override.
                    for i in range(spawn_retransmits):
                        time.sleep(0.05)
                        retransmit_tick = self._get_network_tick(ctx)
                        retransmit_packet = build_udp_tank_packet_wf(
                            net_id=net_id,
                            unit_type=unit_type,
                            team_id=team_id,
                            pos=send_pos,
                            rot=send_rot,
                            tick=retransmit_tick,
                            include_vitals=True,
                            weapon_id=spawn_tank_weapon,
                            health=1.0,
                            energy=1.0,
                        )
                        self.udp_handler.send_to(retransmit_packet, ctx.session.udp_addr)
                    print(f"[SPAWN] Retransmitted TankPacket {spawn_retransmits}x (vitals=1)")
                # Immediate UPDATE_ARRAY heartbeat with health=1.0.
                # The pre-creation UPDATE_ARRAY set up the OID table entry,
                # and the TankPacket's PLAYER_INFO path found it via
                # OIDTable_lookup â†’ called LocalPlayer_initialize â†’
                # g_local_player_entity is now set.  This heartbeat populates
                # ESI with health=1.0 so sync_local_player writes it to the
                # HUD health meter, clearing the red overlay.
                if os.environ.get("WULFRAM_SPAWN_BOOTSTRAP_HEARTBEAT", "1").strip().lower() in ("0", "false", "off", "no"):
                    print("[SPAWN] Suppressing immediate spawn bootstrap heartbeat UPDATE_ARRAY")
                elif self._suppress_remote_spawn_bootstrap_heartbeat(ctx):
                    if self.update_local_state_mode == "wf" and not handlers._is_loopback_client(ctx):
                        time.sleep(0.05)
                        hb_tick = self._get_network_tick(ctx)
                        hb_packet = self._build_remote_spawn_bootstrap_heartbeat(
                            ctx,
                            tick=hb_tick,
                            entity_id=net_id,
                            health=1.0,
                            fuel=1.0,
                        )
                        self.udp_handler.send_to(hb_packet, ctx.session.udp_addr)
                        print("[SPAWN] Sent remote bootstrap heartbeat UPDATE_ARRAY (full transform safe shape)")
                    else:
                        print("[SPAWN] Suppressing immediate remote bootstrap heartbeat UPDATE_ARRAY")
                else:
                    time.sleep(0.05)
                    hb_tick = self._get_network_tick(ctx)
                    hb_packet = self._build_local_state_heartbeat(
                        ctx,
                        tick=hb_tick,
                        entity_id=net_id,
                        include_health=True,
                        health=1.0,
                        fuel=1.0,
                    )
                    self.udp_handler.send_to(hb_packet, ctx.session.udp_addr)
                    print(f"[SPAWN] Sent immediate heartbeat UPDATE_ARRAY (health=1.0)")
            else:
                # Fallback to TCP if no UDP address
                ctx.tcp_handler.send(tank_packet)
                print(f"[SPAWN] Sent TCP TankPacket (no UDP addr)")
        else:
            # Explicit TCP fallback for spawn-isolation runs.
            ctx.tcp_handler.send(tank_packet)
            print("[SPAWN] Sent TCP TankPacket (WULFRAM_SPAWN_UDP_TANK=0)")

        # Wulf-forge capture order is TankPacket first, then the spawn message.
        if announce and self.spawn_send_comm_message:
            comm_pkt = build_chat_message("Spawning in...", source_id=net_id)
            if self.udp_handler and ctx.session.udp_addr and sent_tank_udp:
                try:
                    self.udp_handler.send_to(comm_pkt, ctx.session.udp_addr)
                    print(f"[SPAWN] Sent UDP CommMessage for spawn to {ctx.session.udp_addr}")
                except Exception as ex:
                    print(f"[SPAWN] Failed to send UDP CommMessage for spawn: {ex}")
                    if ctx.tcp_handler:
                        ctx.tcp_handler.send(comm_pkt)
                        print(f"[SPAWN] Sent TCP CommMessage for spawn (fallback)")
            elif ctx.tcp_handler:
                ctx.tcp_handler.send(comm_pkt)
                print(f"[SPAWN] Sent TCP CommMessage for spawn (no UDP addr)")

        # NOTE: We rely on TankPacket (UDP) to create the entity.
        # UPDATE_ARRAY_CREATE_TANK causes crash when sent AFTER TankPacket.
        # For now, skip UPDATE_ARRAY and try PLAYER_INFO alone.

        # PLAYER_INFO tells the client "this is your controllable entity"
        # Without this, client won't send VIEWPOINT_INFO (0x35)
        include_player_info_state, player_info_state = self._get_player_info_local_state_kwargs(ctx)
        weapon_type = player_info_state["weapon_id"]
        ammo_bits = player_info_state["ammo_count_bits"]
        ammo_mask = player_info_state["ammo_count"]
        pt_bits = player_info_state["primary_turret_bits"]
        pt_angle = player_info_state["primary_turret_angle"]
        st_bits = player_info_state["secondary_turret_bits"]
        st_angle = player_info_state["secondary_turret_angle"]
        if self.player_info_local_state_mode != "off":
            print(
                "[LOCAL-STATE] PLAYER_INFO "
                f"weapon={weapon_type} ammo_bits={ammo_bits} "
                f"pt_bits={pt_bits} st_bits={st_bits}"
            )
        player_info_props = 0
        if self.player_info_properties_mode in ("team", "team_id"):
            player_info_props = team_id
        elif self.player_info_properties_mode:
            try:
                player_info_props = int(self.player_info_properties_mode, 0)
            except ValueError:
                player_info_props = 0
        if self.player_info_properties_mode:
            print(f"[LOCAL-STATE] PLAYER_INFO properties={player_info_props}")
        if include_player_info_state:
            self._log_vitals(
                ctx,
                "PLAYER_INFO",
                include_vitals=True,
                health=1.0,
                energy=1.0,
                weapon_id=weapon_type,
                note=f"ammo_bits={ammo_bits} pt_bits={pt_bits} st_bits={st_bits}",
            )
        elif self.debug_vitals and self.player_info_local_state_mode != "off":
            self._log_vitals(
                ctx,
                "PLAYER_INFO",
                include_vitals=False,
                health=1.0,
                energy=1.0,
                weapon_id=weapon_type,
                note="local_state=0",
            )
        if self.spawn_send_player_packet:
            ctx.tcp_handler.send(build_player(net_id, spectator=self.spawn_player_spectator))
            print(f"[SPAWN] Sent PLAYER spectator={int(self.spawn_player_spectator)}")
        send_spawn_player_info = self._should_send_spawn_player_info(ctx)
        if not send_spawn_player_info:
            reason = (
                f"WULFRAM_SPAWN_PLAYER_INFO={int(self.spawn_send_player_info)}"
                if self.spawn_send_player_info_explicit
                else "loopback default"
            )
            print(f"[SPAWN] Skipping TCP PLAYER_INFO ({reason})")
        else:
            player_info_pkt = build_player_info(
                entity_oid=net_id,
                vehicle_type=unit_type,
                pos=send_pos,
                rot=send_rot,
                include_local_state=include_player_info_state,
                weapon_id=player_info_state["weapon_id"],
                health=player_info_state["health"],
                fuel=player_info_state["fuel"],
                properties=player_info_props,
                ammo_count_bits=player_info_state["ammo_count_bits"],
                ammo_count=player_info_state["ammo_count"],
                primary_turret_bits=player_info_state["primary_turret_bits"],
                primary_turret_angle=player_info_state["primary_turret_angle"],
                secondary_turret_bits=player_info_state["secondary_turret_bits"],
                secondary_turret_angle=player_info_state["secondary_turret_angle"],
                turret_max=player_info_state["turret_max"],
                turret_range=player_info_state["turret_range"],
            )
            ctx.tcp_handler.send(player_info_pkt)
            print(f"[SPAWN] Sent TCP PLAYER_INFO: entity_oid={net_id} vehicle={unit_type}")

        # Track game start time from first spawn
        if self._game_start_time == 0.0:
            self._game_start_time = time.monotonic()

        # Complete spawn sequence (matching spawn_full)
        if self.spawn_send_game_clock:
            ctx.tcp_handler.send(build_game_clock())
        if self.spawn_send_reincarnate:
            rein = build_reincarnate(0x11, "")
            sent_rein = False
            if self.udp_handler and ctx.session.udp_addr:
                try:
                    self.udp_handler.send_to(rein, ctx.session.udp_addr)
                    sent_rein = True
                    print(f"[SPAWN] Sent UDP REINCARNATE(0x11) to {ctx.session.udp_addr}")
                except Exception as ex:
                    print(f"[SPAWN] Failed to send UDP REINCARNATE(0x11): {ex}")
            if not sent_rein:
                ctx.tcp_handler.send(rein)
                print("[SPAWN] Sent TCP REINCARNATE(0x11) (fallback)")
        if self.spawn_send_birth_notice:
            ctx.tcp_handler.send(build_birth_notice(net_id))
        if self.spawn_send_game_clock or self.spawn_send_reincarnate or self.spawn_send_birth_notice:
            print(
                "[SPAWN] Sent "
                f"{'GAME_CLOCK ' if self.spawn_send_game_clock else ''}"
                f"{'REINCARNATE(0x11) ' if self.spawn_send_reincarnate else ''}"
                f"{'BIRTH_NOTICE' if self.spawn_send_birth_notice else ''}"
            )

        # Enter game mode and start tick loop for UPDATE_ARRAY
        was_in_game = ctx.session.in_game or ctx.session.phase == Phase.IN_GAME
        ctx.session.entity_id = net_id
        ctx.session.last_spawn_time = time.monotonic()
        ctx.session.in_game = True
        if not was_in_game:
            ctx.session.transition_to(Phase.IN_GAME)
        else:
            print(f"[SPAWN] Client {ctx.client_id}: refreshed active spawn while already IN_GAME")
        if self.spawn_send_player_active:
            ctx.tcp_handler.send(build_player(net_id, spectator=False))
            print("[SPAWN] Sent PLAYER spectator=0 (post-spawn reassert)")

        # Sync roster/entity visibility with other in-game clients.
        self._sync_clients_on_spawn(ctx)
        self._ensure_uplink_mvp_state(ctx)

        self._resume_remote_full_local_state_after_spawn(
            ctx,
            entity_id=net_id,
            health=1.0,
            fuel=1.0,
            previously_promoted=restore_promoted_remote_sync_after_spawn,
        )

        if self._ensure_tick_loop(ctx):
            print(f"[SPAWN] Started tick loop (local_state_updates={self.update_local_state_mode})")

    def _spawn_wf_minimal(self, ctx: ClientContext, team_id: int, net_id: int, addr: tuple):
        """
        Absolutely minimal spawn - just TankPacket (wulf-forge style).

        Uses include_vitals per WULFRAM_TANK_VITALS (defaults off while investigating).
        TRANSLATION quantizers define 5/10/10 bits which matches TankPacket format.
        """
        print(f"[SPAWN] Minimal WF: client={ctx.client_id} net_id={net_id} team={team_id}")
        ctx.entity_type = 0

        spawn_pos = self._resolve_spawn_pos(team_id)
        configured_default = self._get_configured_default_spawn_pos()
        map_spawn = None if configured_default is not None else self._pick_spawn_point(team_id)
        if configured_default is not None:
            print(f"[SPAWN] Minimal mode default flat spawn pos={spawn_pos}")
        elif map_spawn:
            print(f"[SPAWN] Minimal mode map spawn oid={map_spawn['oid']} team={team_id} pos={spawn_pos}")
        self._set_ground_level_override_for_pose(ctx, spawn_pos)

        # Use 0-based index among active clients (not client_id, which grows unboundedly).
        if self.multi_spawn_offset:
            with self.clients_lock:
                active_ids = sorted(c.client_id for c in self.clients.values() if c and c.running)
            try:
                idx = active_ids.index(ctx.client_id)
            except ValueError:
                idx = 0
            spawn_pos = (spawn_pos[0] + idx * self.multi_spawn_offset, spawn_pos[1], spawn_pos[2])

        send_pos = self._to_client_pos(spawn_pos)
        self._set_ground_level_override_for_pose(ctx, spawn_pos)
        ctx.player_energy = self.player_energy_max
        self._reset_jump_jet_state(ctx)
        tank_packet = build_udp_tank_packet_wf(
            net_id=net_id,
            unit_type=0,
            team_id=team_id,
            pos=send_pos,
            rot=self._local_player_sync_rotation(ctx),
            include_vitals=self.tank_vitals,
            weapon_id=self.weapon_id,
            health=1.0,
            energy=1.0,
        )
        self._log_vitals(
            ctx,
            "TANK_UDP_MINIMAL",
            include_vitals=self.tank_vitals,
            health=1.0,
            energy=1.0,
            weapon_id=self.weapon_id,
            note=f"team={team_id} net_id={net_id}",
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

    def _load_map_config(self) -> dict:
        """Load per-map configuration overrides from map_config.json."""
        config_path = Path(__file__).resolve().parent / "map_config.json"
        if not config_path.exists():
            return {}
        try:
            raw = config_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        normalized = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            normalized[key.lower()] = value
        return normalized

    def _spawn_at_point(self, ctx: ClientContext, spawn_point_id: int, vehicle_type: int, addr: tuple):
        """Handle spawn at a specific spawn point."""
        handlers.handle_spawn_at_point(self, ctx, spawn_point_id, vehicle_type, addr)

    def _parse_spawn_points_env(self, raw: str) -> Optional[list]:
        """Parse WULFRAM_SPAWN_POINTS env var into spawn point dicts.

        Formats:
        - "oid,team,x,y,z"
        - "oid,team,variant,x,y,z"
        - "team,x,y,z"
        """
        raw = raw.strip()
        if not raw:
            return None
        points = []
        for idx, chunk in enumerate(raw.split(";"), start=1):
            item = chunk.strip()
            if not item:
                continue
            parts = [p.strip() for p in item.split(",")]
            if len(parts) not in (4, 5, 6):
                print(f"[MAP] Invalid spawn point entry '{item}' (expected 4, 5, or 6 fields)")
                return None
            try:
                variant = 1
                if len(parts) == 6:
                    oid = int(parts[0], 0)
                    team = int(parts[1], 0)
                    variant = int(parts[2], 0)
                    x, y, z = map(float, parts[3:])
                elif len(parts) == 5:
                    oid = int(parts[0], 0)
                    team = int(parts[1], 0)
                    x, y, z = map(float, parts[2:])
                else:
                    oid = 5000 + idx
                    team = int(parts[0], 0)
                    x, y, z = map(float, parts[1:])
            except ValueError:
                print(f"[MAP] Invalid spawn point values in '{item}'")
                return None
            points.append({
                "oid": oid,
                "team": team,
                "variant": variant,
                "x": x,
                "y": y,
                "z": z,
            })
        return points

    def _load_terrain(self):
        """Load terrain heightmap from game data or env var path."""
        terrain_file = os.environ.get("WULFRAM_TERRAIN_FILE")
        if terrain_file:
            land_path = Path(terrain_file)
        else:
            repo_root = Path(__file__).resolve().parents[2]
            maps_root = repo_root / "slurpysoft-wulfram" / "data" / "maps"
            map_name = self.map_name
            land_path = maps_root / map_name / "land"
            if not land_path.exists() and maps_root.exists():
                for entry in maps_root.iterdir():
                    if entry.is_dir() and entry.name.lower() == map_name.lower():
                        candidate = entry / "land"
                        if candidate.exists():
                            land_path = candidate
                            break
        if land_path.exists():
            try:
                self.terrain = Terrain(str(land_path))
                print(f"[TERRAIN] Pitch in impulse: {'ON' if self.terrain_pitch_enabled else 'OFF'}")
                print(
                    "[TERRAIN] Height offsets: "
                    f"map={self.terrain_height_offset} "
                    f"physics={self.terrain_physics_height_offset} "
                    f"tank_spring_base={self.tank_spring_base_offset} "
                    f"tank_drive_terrain_aligned={int(self.tank_drive_terrain_aligned)} "
                    f"tank_contact_coupling={int(self.tank_terrain_contact_coupling_enabled)}"
                )
            except Exception as e:
                print(f"[TERRAIN] Failed to load {land_path}: {e}")
                self.terrain = None
        else:
            print(f"[TERRAIN] No heightmap found (looked for {land_path})")

    def _load_map_land_grid(self) -> Optional[tuple]:
        """Read grid size (rows x cols) from the map land file."""
        repo_root = Path(__file__).resolve().parents[2]
        maps_root = repo_root / "slurpysoft-wulfram" / "data" / "maps"
        map_name = self.map_name
        land_path = maps_root / map_name / "land"

        if not land_path.exists() and maps_root.exists():
            # Case-insensitive fallback for map directory.
            for entry in maps_root.iterdir():
                if entry.is_dir() and entry.name.lower() == map_name.lower():
                    candidate = entry / "land"
                    if candidate.exists():
                        land_path = candidate
                        break

        if not land_path.exists():
            return None

        try:
            lines = land_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return None

        if not lines:
            return None

        # Expected: first line like "129x129"
        first = lines[0].strip().lower()
        if "x" not in first:
            return None
        try:
            rows_s, cols_s = first.split("x", 1)
            rows = int(rows_s)
            cols = int(cols_s)
        except ValueError:
            return None

        # Optional: second line like "5600.000000x5600.000000"
        size_x = 0.0
        size_y = 0.0
        if len(lines) > 1 and "x" in lines[1]:
            try:
                size_x_s, size_y_s = lines[1].strip().lower().split("x", 1)
                size_x = float(size_x_s)
                size_y = float(size_y_s)
            except ValueError:
                size_x = 0.0
                size_y = 0.0

        return rows, cols, size_x, size_y

    def _terrain_ground_z_at(self, x: float, y: float) -> Optional[float]:
        """Return map-entity ground Z for X/Y when terrain is loaded."""
        if getattr(self, "up_axis", "z") != "z":
            return None
        terrain = getattr(self, "terrain", None)
        if terrain is None:
            return None
        return float(terrain.get_height(x, y)) + float(getattr(self, "terrain_height_offset", 0.0))

    def _terrain_physics_ground_z_at(self, x: float, y: float) -> Optional[float]:
        """Return vehicle-physics ground Z for X/Y when terrain is loaded."""
        if getattr(self, "up_axis", "z") != "z":
            return None
        terrain = getattr(self, "terrain", None)
        if terrain is None:
            return None
        return float(terrain.get_height(x, y)) + float(
            getattr(self, "terrain_physics_height_offset", 0.0)
        )

    def _set_ground_level_override_for_pose(self, ctx: ClientContext, pos: tuple[float, float, float]) -> None:
        """Pin exact spawn/control Z while recording the terrain anchor under it."""
        if not getattr(self, "spawn_sets_ground_level", False):
            ctx.ground_level_override = None
            ctx.ground_override_ref_terrain_level = None
            return
        if (
            getattr(ctx, "entity_type", None) == EntityType.TANK
            and getattr(self, "up_axis", "z") == "z"
            and getattr(self, "terrain", None) is not None
            and getattr(self, "tank_suspension_enabled", False)
            and getattr(self, "tank_suspension_model", "softbody") != "compact"
        ):
            ctx.ground_level_override = None
            ctx.ground_override_ref_terrain_level = None
            return
        if getattr(self, "up_axis", "z") == "z":
            ctx.ground_level_override = float(pos[2])
            ctx.ground_override_ref_terrain_level = self._terrain_physics_ground_z_at(pos[0], pos[1])
        else:
            ctx.ground_level_override = float(pos[1])
            ctx.ground_override_ref_terrain_level = None

    def _align_map_entity_z_to_terrain(
        self,
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, Optional[float], bool]:
        # Collision and vehicle physics run against the physics terrain plane,
        # not the +5 visual/map offset used by some replicated map entities.
        # Preserve raw map-state Z unless it is genuinely below that physics
        # plane; otherwise server-side building collision diverges from the OG
        # and Python clients, which load local map-state blockers at raw Z.
        ground_z = self._terrain_physics_ground_z_at(x, y)
        if ground_z is None or z >= ground_z - 0.01:
            return z, ground_z, False
        return ground_z, ground_z, True

    def _load_map_spawn_points(self) -> Optional[list]:
        """Load repair pad spawn points from the current map state file."""
        if self.map_spawn_points:
            return self.map_spawn_points

        repo_root = Path(__file__).resolve().parents[2]
        maps_root = repo_root / "slurpysoft-wulfram" / "data" / "maps"
        map_name = self.map_name
        state_path = maps_root / map_name / "state"

        if not state_path.exists() and maps_root.exists():
            # Case-insensitive fallback for map directory.
            for entry in maps_root.iterdir():
                if entry.is_dir() and entry.name.lower() == map_name.lower():
                    candidate = entry / "state"
                    if candidate.exists():
                        state_path = candidate
                        break

        if not state_path.exists():
            return None

        points = []
        aligned_count = 0
        try:
            lines = state_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return None

        oid = 5001
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue

            # Handle optional crate prefix: "c r ..."
            data_start = 1
            unit_code = parts[0]
            if unit_code == "c":
                if len(parts) < 2:
                    continue
                unit_code = parts[1]
                data_start = 2

            if unit_code != "r":
                continue

            try:
                team = int(parts[data_start])
                x = float(parts[data_start + 1])
                y = float(parts[data_start + 2])
                raw_z = float(parts[data_start + 3])
            except (ValueError, IndexError):
                continue
            if getattr(self, "align_spawn_points_to_terrain", False):
                z, terrain_z, aligned = self._align_map_entity_z_to_terrain(x, y, raw_z)
                if aligned:
                    aligned_count += 1
            else:
                z, terrain_z, aligned = raw_z, self._terrain_ground_z_at(x, y), False

            variant = 1
            rot = (0.0, 0.0, 0.0)
            if len(parts) > data_start + 6:
                try:
                    rot = (
                        float(parts[data_start + 4]),
                        float(parts[data_start + 5]),
                        float(parts[data_start + 6]),
                    )
                except ValueError:
                    rot = (0.0, 0.0, 0.0)
            if len(parts) > data_start + 7:
                try:
                    variant = int(parts[data_start + 7])
                except ValueError:
                    variant = 1

            point = {
                "oid": oid,
                "team": team,
                "variant": variant,
                "x": x,
                "y": y,
                "z": z,
                "rot": rot,
            }
            if aligned:
                point["raw_z"] = raw_z
                point["terrain_z"] = terrain_z
            points.append(point)
            oid += 1

        if points:
            suffix = f" ({aligned_count} terrain-aligned)" if aligned_count else ""
            print(f"[MAP] Loaded {len(points)} spawn points from {state_path}{suffix}")
            self.map_spawn_points = points

        return self.map_spawn_points

    # Unit code â†’ entity type mapping for building collision
    _BUILDING_UNIT_CODES = {
        'e': 25,  # ENERGY_BUILDING
        'f': 26,  # FUEL_BUILDING
        'r': 27,  # REPAIR_BUILDING
        'S': 28,  # SPECIAL_STRUCTURE
        's': 29,  # SENSOR_BUILDING
        'g': 30,  # GUN_TURRET
        'E': 31,  # ENERGY_STRUCTURE
        'L': 32,  # LAUNCHER
        'p': 33,  # PAD
        'o': 34,  # ORBITAL_BUILDING
        'd': 35,  # DARK_LIGHT
        'b': 36,  # BUILDING
        '*': 37,  # STRUCTURE
    }

    def _load_map_buildings(self):
        """Load building entities from the map state file for collision."""
        repo_root = Path(__file__).resolve().parents[2]
        maps_root = repo_root / "slurpysoft-wulfram" / "data" / "maps"
        map_name = self.map_name
        state_path = maps_root / map_name / "state"

        if not state_path.exists() and maps_root.exists():
            for entry in maps_root.iterdir():
                if entry.is_dir() and entry.name.lower() == map_name.lower():
                    candidate = entry / "state"
                    if candidate.exists():
                        state_path = candidate
                        break

        if not state_path.exists():
            print(f"[MAP] No state file found for buildings: {state_path}")
            self._building_entities = {}
            self._rebuild_static_world_raycast_index()
            return

        try:
            lines = state_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            self._building_entities = {}
            self._rebuild_static_world_raycast_index()
            return

        buildings = {}
        aligned_count = 0
        oid = 10001  # Offset from spawn point OIDs (5001+)
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue

            # Handle optional crate prefix: "c <code> ..."
            data_start = 1
            unit_code = parts[0]
            if unit_code == "c":
                if len(parts) < 2:
                    continue
                unit_code = parts[1]
                data_start = 2

            entity_type = self._BUILDING_UNIT_CODES.get(unit_code)
            if entity_type is None:
                continue

            try:
                team = int(parts[data_start])
                x = float(parts[data_start + 1])
                y = float(parts[data_start + 2])
                raw_z = float(parts[data_start + 3])
            except (ValueError, IndexError):
                continue
            z, _terrain_z, aligned = self._align_map_entity_z_to_terrain(x, y, raw_z)
            if aligned:
                aligned_count += 1

            heading = 0.0
            if len(parts) > data_start + 6:
                try:
                    # Map-state entries carry three Euler values after xyz.
                    # The final angle is the closest match to building yaw.
                    heading = float(parts[data_start + 6])
                except ValueError:
                    heading = 0.0

            buildings[oid] = BuildingEntity(
                x=x,
                y=y,
                z=z,
                entity_type=entity_type,
                team_id=team,
                heading=heading,
            )
            oid += 1

        self._building_entities = buildings

        # Initialize building health from decompile entity health table (VA 0x4E3B00)
        _BUILDING_MAX_HEALTH = {
            EntityType.GUN_TURRET: 1200.0,
            EntityType.LAUNCHER: 1200.0,
            EntityType.SENSOR_BUILDING: 1200.0,
            EntityType.FUEL_BUILDING: 2000.0,
            EntityType.REPAIR_BUILDING: 2000.0,
            EntityType.ENERGY_BUILDING: 2000.0,
            EntityType.PAD: 5000.0,
            EntityType.DARK_LIGHT: 800.0,
        }
        self._building_health = {}
        self._building_max_health = {}
        for oid, b in buildings.items():
            max_hp = _BUILDING_MAX_HEALTH.get(b.entity_type, 2000.0)
            self._building_health[oid] = max_hp
            self._building_max_health[oid] = max_hp

        self._rebuild_static_world_raycast_index()
        if buildings:
            suffix = f" ({aligned_count} terrain-aligned)" if aligned_count else ""
            print(f"[MAP] Loaded {len(buildings)} building entities for collision from {state_path}{suffix}")

    def get_spawn_points(self) -> list:
        """Return spawn point list for current map/config."""
        if self.map_spawn_points is None:
            self._load_map_spawn_points()
        if self.map_spawn_points:
            return [dict(sp) for sp in self.map_spawn_points]
        return [
            {"oid": 5001, "team": 1, "x": 50.0, "y": 10.0, "z": 50.0},
            {"oid": 5002, "team": 1, "x": 60.0, "y": 10.0, "z": 50.0},
            {"oid": 5003, "team": 2, "x": 150.0, "y": 10.0, "z": 150.0},
            {"oid": 5004, "team": 2, "x": 160.0, "y": 10.0, "z": 150.0},
        ]

    def _pick_spawn_point(self, team_id: int) -> Optional[dict]:
        """Pick the first spawn point for the requested team."""
        if not self.use_map_spawn_points:
            return None
        points = self.get_spawn_points()
        for sp in points:
            if sp.get("team") == team_id:
                return sp
        return points[0] if points else None

    def _parse_spawn_pos_env(self, raw: str) -> Optional[tuple[float, float, float]]:
        """Parse `x,y,z` spawn position env/config values."""
        raw = (raw or "").strip()
        if not raw or raw.lower() in ("none", "null"):
            return None
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) < 2:
            return None
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2]) if len(parts) > 2 else getattr(self, "spawn_height", 5.0)
        except ValueError:
            return None
        return (x, y, z)

    def _get_builtin_flat_spawn_pos(self) -> tuple[float, float, float]:
        """Return the built-in flat default spawn for the active map."""
        map_name = getattr(self, "map_name", "crossroads")
        up_axis = getattr(self, "up_axis", "z")
        spawn_height = getattr(self, "spawn_height", 5.0)
        if map_name.lower() == "crossroads":
            if up_axis == "z":
                return (4950.0, 5100.0, spawn_height)
            return (4950.0, spawn_height, 5100.0)
        if up_axis == "z":
            return (100.0, 100.0, spawn_height)
        return (100.0, spawn_height, 100.0)

    def _get_configured_default_spawn_pos(self) -> Optional[tuple[float, float, float]]:
        """Return the configured default spawn that should win over map pads."""
        spawn_override = self._parse_spawn_pos_env(os.environ.get("WULFRAM_SPAWN_POS", ""))
        if spawn_override is not None:
            return spawn_override
        if getattr(self, "force_default_spawn_pos", False):
            default_flat_spawn_pos = getattr(self, "default_flat_spawn_pos", None)
            if default_flat_spawn_pos is not None:
                return default_flat_spawn_pos
        return None

    def _resolve_spawn_pos(
        self,
        team_id: int,
        *,
        explicit_pos: Optional[tuple[float, float, float]] = None,
        allow_map_spawn: bool = True,
    ) -> tuple[float, float, float]:
        """Resolve the spawn position for normal join/respawn flows."""
        if explicit_pos is not None:
            return explicit_pos
        configured_default = self._get_configured_default_spawn_pos()
        if configured_default is not None:
            return configured_default
        if allow_map_spawn:
            map_spawn = self._pick_spawn_point(team_id)
            if map_spawn:
                return (map_spawn["x"], map_spawn["y"], map_spawn["z"])
        return self._get_builtin_flat_spawn_pos()

    def build_world_stats_packet(self) -> bytes:
        """Build WORLD_STATS with the current map configuration."""
        return build_world_stats(
            map_name=self.map_name,
            grid_rows=self.map_grid_rows,
            grid_cols=self.map_grid_cols,
            scale=self.map_scale,
        )

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

        # Wire control server to this client's handlers
        self.control_server.tcp_handler = ctx.tcp_handler
        self.control_server.session = ctx.session

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
            # Preserve UDP addr before session reset so we can clean mapping safely.
            udp_addr = ctx.session.udp_addr
            disconnected_entity_id = ctx.session.entity_id or ctx.entity_id
            was_in_game = bool(ctx.session.in_game and disconnected_entity_id)
            ctx.running = False
            ctx.session.in_game = False

            # Remove from client tracking
            with self.clients_lock:
                if ctx.client_id in self.clients:
                    del self.clients[ctx.client_id]

            if was_in_game:
                self._broadcast_disconnected_player_delete(ctx, disconnected_entity_id)

            # Remove UDP address mapping only if it still points to this context.
            if udp_addr and self.udp_addr_to_client.get(udp_addr) is ctx:
                del self.udp_addr_to_client[udp_addr]

            # Remove session key mapping
            skey = ctx.session.session_key
            if skey and self.session_key_to_client.get(skey) is ctx:
                del self.session_key_to_client[skey]

            ctx.session.reset()

            sock.close()
            print(f"[SERVER] Client {ctx.client_id} disconnected")

    def _broadcast_disconnected_player_delete(self, ctx: ClientContext, entity_id: int) -> None:
        """Tell remaining clients to remove a disconnected player's entity."""
        tick = get_ticks()
        delete_pkt = build_delete_object(tick, [entity_id], with_effects=False)
        sent_count = 0
        for other in self._snapshot_in_game_clients():
            if other is ctx:
                continue
            if entity_id in other.known_entity_ids:
                other.known_entity_ids.discard(entity_id)
            if self._send_packet_to_client(other, delete_pkt, prefer_tcp=True):
                sent_count += 1
        if sent_count:
            print(
                f"[SERVER] Client {ctx.client_id}: deleted disconnected "
                f"entity {entity_id} for {sent_count} client(s)"
            )

    def _do_handshake(self, ctx: ClientContext):
        """Perform initial handshake."""
        time.sleep(0.5)  # Let client initialize

        login_bootstrap_mode = handlers._get_login_bootstrap_mode(ctx)
        use_og_handshake = (login_bootstrap_mode == "og")

        # Loopback/minimal clients still use the early session key path for
        # deterministic UDP binding. OG clients expect HELLO subtype 1 first
        # and only request HELLO subtype 2 later.
        if not use_og_handshake:
            if not ctx.session.session_key:
                ctx.session.session_key = f"WS-{ctx.client_id}-{secrets.token_hex(8)}"
            self.session_key_to_client[ctx.session.session_key] = ctx
            ctx.tcp_handler.send(build_hello_session_key(ctx.session.session_key))

        # Send UDP config - use public_addr, or resolve from client's TCP connection
        udp_addr = self.public_addr
        if udp_addr == "0.0.0.0":
            # Use the local address the client actually connected to
            udp_addr = ctx.tcp_handler.sock.getsockname()[0]
        print(
            f"[HANDSHAKE] Client {ctx.client_id}: mode={'og' if use_og_handshake else 'minimal'} "
            f"UDP config {udp_addr}:{self.port}"
        )
        ctx.tcp_handler.send(build_hello_udp_config(udp_addr, self.port))
        ctx.session.udp_config_sent_time = time.monotonic()

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

        if FEATURES.auto_login:
            # Wait briefly for client's LOGIN_REQUEST which carries the username,
            # then auto-advance to team select regardless.
            # Auto-login: consume LOGIN_REQUEST to extract username, then
            # proceed directly to bootstrap.  HELLO packets are queued
            # and replied to AFTER the bootstrap — the OG client's login
            # state machine expects bootstrap packets (TEAM_INFO etc.)
            # immediately after LOGIN_STATUS(5), not HELLO responses.
            ctx.tcp_handler.sock.settimeout(1.0)
            pending_hellos = []
            try:
                for _ in range(5):
                    packet = ctx.tcp_handler.recv()
                    if packet and len(packet) >= 2 and packet[0] == PacketType.LOGIN_REQUEST:
                        username, _ = handlers.decode_lp_string(packet, 2)
                        if username:
                            ctx.session.username = username
                        break
                    elif packet and len(packet) >= 1 and packet[0] == PacketType.HELLO:
                        pending_hellos.append(packet)
            except Exception:
                pass
            finally:
                ctx.tcp_handler.sock.settimeout(None)
            if not ctx.session.username:
                ctx.session.username = f"Player{ctx.client_id}"
            ctx.session.login_complete = True
            ctx.session.transition_to(Phase.TEAM_SELECT)
            handlers.send_initial_game_data(self, ctx)
            # Now reply to queued HELLOs (after bootstrap, matching old packet order)
            for hello_pkt in pending_hellos:
                self._handle_hello(ctx, hello_pkt)
            print(f"[LOGIN] Client {ctx.client_id}: Auto-login as {ctx.session.username}")
            return

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

    def _effective_inactivity_timeout(self, ctx: ClientContext) -> float:
        """Remote in-game sessions can legitimately go quiet while idle."""
        timeout = self.inactivity_timeout
        addr = ctx.client_addr[0] if ctx.client_addr else ""
        is_remote = addr not in ("127.0.0.1", "::1", "localhost")
        if is_remote:
            timeout = max(timeout, self.remote_idle_timeout)
        return timeout

    def _game_loop(self, ctx: ClientContext):
        """Main game packet loop."""
        # Set socket timeout for delayed spawn checking (recv returns None on timeout)
        ctx.tcp_handler.sock.settimeout(0.5)

        # Track last activity for dead connection detection
        # UDP packets count as activity since client sends TRANSLATION_ACK continuously
        last_activity = time.monotonic()
        while ctx.running and ctx.session.phase in [Phase.TEAM_SELECT, Phase.SPAWNING, Phase.IN_GAME]:
            inactivity_timeout = self._effective_inactivity_timeout(ctx)
            # Check for delayed spawn
            if ctx.session.delayed_spawn_team and ctx.session.delayed_spawn_time:
                now = time.monotonic()
                if now >= ctx.session.delayed_spawn_time:
                    if (not ctx.session.input_ready and ctx.session.want_updates_time
                            and (now - ctx.session.want_updates_time) < self.spawn_force_after):
                        if not getattr(ctx.session, "spawn_wait_logged", False):
                            print(
                                f"[GAME] Client {ctx.client_id}: Delaying spawn until input or "
                                f"{self.spawn_force_after:.1f}s after WANT_UPDATES"
                            )
                            ctx.session.spawn_wait_logged = True
                        ctx.session.delayed_spawn_time = now + 0.5
                        continue
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

                if time.monotonic() - last_activity > inactivity_timeout:
                    print(f"[GAME] Client {ctx.client_id} inactive for {inactivity_timeout:.1f}s - disconnecting")
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
            elif pkt_type == 0x20:
                self._handle_tcp_comm_message_request(ctx, packet)
            elif pkt_type == PacketType.PING_REQUEST:
                # The client echoes server PING_REQUEST packets back on TCP.
                # Treat that as a latency reply, not an unknown gameplay opcode.
                if self.debug_udp_raw:
                    print(f"[GAME] PING_REQUEST reply len={len(packet)} data={packet.hex()}")
            elif pkt_type == 0x19:
                # Tank resend request (client didn't accept PLAYER_INFO)
                print("[GAME] TANK_RESEND_REQUEST received - resending TankPacket")
                entity_id = ctx.session.entity_id or ctx.entity_id
                if ctx.player_pos and len(ctx.player_pos) == 3:
                    tank_pos = ctx.player_pos
                elif self.up_axis == "z":
                    tank_pos = (100.0, 15.0, self.spawn_height)
                else:
                    tank_pos = (100.0, self.spawn_height, 15.0)
                send_pos = self._to_client_pos(tank_pos)
                send_rot = self._local_player_sync_rotation(ctx)
                tank_packet = build_tank_packet(
                    net_id=entity_id,
                    unit_type=0,
                    pos=send_pos,
                    rot=send_rot,
                    flags=1,
                    include_vitals=self.tank_vitals,
                    health=1.0,
                    energy=self._get_energy_value(ctx)
                )
                self._log_vitals(
                    ctx,
                    "TANK_TCP_RESEND",
                    include_vitals=self.tank_vitals,
                    health=1.0,
                    energy=self._get_energy_value(ctx),
                    weapon_id=self.weapon_id,
                    note=f"net_id={entity_id}",
                )
                ctx.tcp_handler.send(tank_packet)
            elif pkt_type == PacketType.HELLO:
                self._handle_hello(ctx, packet)
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
        # Check for pending respawn position (set by respawn command)
        pos = None
        if getattr(ctx, 'pending_respawn_pos', None):
            pos = ctx.pending_respawn_pos
            ctx.pending_respawn_pos = None
            print(f"[SPAWN] Using pending respawn pos={pos}")
        else:
            pos = self._resolve_spawn_pos(team_id)
            configured_default = self._get_configured_default_spawn_pos()
            if configured_default is not None:
                print(f"[SPAWN] Using default flat spawn pos={pos}")
            else:
                spawn = self._pick_spawn_point(team_id)
                if spawn:
                    print(f"[SPAWN] Using spawn point oid={spawn['oid']} team={team_id} pos={pos}")
        self._spawn_wf_style(ctx, team_id=team_id, net_id=ctx.session.player_id or ctx.entity_id, pos=pos)

    def _handle_reincarnate(self, ctx: ClientContext, packet: bytes):
        """Handle REINCARNATE - player wants to spawn."""
        handlers.handle_reincarnate_tcp(self, ctx, packet)

    def _send_tank(self, ctx: ClientContext, entity_id: int = None):
        """Send TankPacket (0x18) to spawn the player's tank."""
        if entity_id is None:
            entity_id = ctx.entity_id
        if ctx.player_pos and len(ctx.player_pos) == 3:
            spawn_pos = ctx.player_pos
        elif self.up_axis == "z":
            spawn_pos = (100.0, 100.0, self.spawn_height)
        else:
            spawn_pos = (100.0, self.spawn_height, 100.0)
        send_pos = self._to_client_pos(spawn_pos)
        # Keep TankPacket resend/reset paths on the same body-space rotation as
        # ordinary UPDATE_ARRAY / VIEW_UPDATE sync so OG clients do not bounce
        # between camera-yaw bootstrap state and body-heading correction state.
        send_rot = self._local_player_sync_rotation(ctx)
        tank_packet = build_tank_packet(
            net_id=entity_id,
            unit_type=0,
            pos=send_pos,
            rot=send_rot,
            flags=1,
            include_vitals=self.tank_vitals,
            health=1.0,
            energy=self._get_energy_value(ctx)
        )
        self._log_vitals(
            ctx,
            "TANK_TCP_SPAWN",
            include_vitals=self.tank_vitals,
            health=1.0,
            energy=self._get_energy_value(ctx),
            weapon_id=self.weapon_id,
            note=f"net_id={entity_id}",
        )
        ctx.tcp_handler.send(tank_packet)

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

        if self.debug_viewpoint:
            print(f"[UDP] VIEWPOINT_INFO 0x35 len={len(data)} data={data.hex()[:40]}...")

        if len(data) < 20:
            # Short packet - might be different subtype, log for analysis
            if self.debug_viewpoint and len(data) >= 5:
                print(f"[VIEWPOINT-SHORT] len={len(data)} bytes={data.hex()}")
            return

        if ctx.session and not ctx.session.input_ready:
            ctx.session.input_ready = True
            ctx.session.input_ready_time = time.monotonic()
            print(f"[GAME] Client {ctx.client_id}: input ready (VIEWPOINT_INFO)")

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
                # Compare server-tracked heading vs client viewpoint
                heading_deg = math.degrees(ctx.player_heading)
                delta_deg = yaw_deg - heading_deg
                # Normalize to -180..180
                while delta_deg > 180.0:
                    delta_deg -= 360.0
                while delta_deg < -180.0:
                    delta_deg += 360.0
                if self.debug_viewpoint:
                    print(
                        f"[VIEWPOINT #{ctx.viewpoint_count}] pitch={pitch_deg:.1f} "
                        f"yaw={yaw_deg:.1f} (was {old_yaw_deg:.1f})"
                    )
                    print(
                        f"[HEADING-COMPARE] server={heading_deg:.1f} client={yaw_deg:.1f} "
                        f"delta={delta_deg:.1f}deg"
                    )
                return
            except (IndexError, ValueError, struct.error) as e:
                if self.debug_viewpoint:
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
                if self.debug_viewpoint:
                    print(
                        f"[VIEWPOINT #{ctx.viewpoint_count}] pitch={math.degrees(pitch):.1f} "
                        f"yaw={math.degrees(yaw):.1f} (was {math.degrees(old_yaw):.1f})"
                    )
            except struct.error as e:
                if self.debug_viewpoint:
                    print(f"[VIEWPOINT-ERR] Failed to decode (double): {e}")

    def _handle_state_request(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle STATE_REQUEST (0x0C) - may contain state/position info.
        Decompile-shaped client payload:
        [opcode:1] [request_id:4] [frame_count:4]
        """
        if ctx is None or len(data) < 5:
            return

        # Decode fields
        request_id = struct.unpack(">I", data[1:5])[0] if len(data) >= 5 else 0
        frame_count = struct.unpack(">I", data[5:9])[0] if len(data) >= 9 else 0
        now = time.monotonic()
        ctx.state_request_count += 1
        ctx.last_state_request_time = now
        ctx.last_state_request_id = request_id
        ctx.last_state_request_frame_count = frame_count
        ctx.last_state_request_len = len(data)

        # Log occasionally to avoid spam
        if request_id % 1000 == 0:
            print(
                f"[0x0C] STATE_REQUEST request_id={request_id} "
                f"frame_count={frame_count} len={len(data)}"
            )

        if not self._state_sync_reply_allowed_for_client(ctx):
            return

        # Remote OG clients still use STATE_REQUEST as the gate out of the
        # fragile spawn/bootstrap window, but the targeted reply itself must
        # stay on the short-form-safe local-state prefix. Promotion here only
        # unlocks the broader post-spawn sync path; it does not justify the
        # expanded ammo/turret local-state on the correction packet itself.
        self._maybe_promote_remote_full_local_state(ctx, reason="state_request")

        if self._remote_movement_input_active(ctx, now=now):
            return

        include_view_update = self._should_send_state_sync_view_update(ctx)
        self._send_state_sync_snapshot(
            ctx,
            reason="state_request",
            include_view_update=include_view_update,
            replay_timestamp=request_id if request_id else None,
        )
        if include_view_update:
            self._queue_state_sync_correction_burst(ctx)

    def _queue_state_sync_correction_burst(self, ctx: ClientContext) -> bool:
        """Queue enough replay updates for OG's local correction to visibly settle."""
        if handlers._is_loopback_client(ctx):
            return False
        if self._remote_movement_input_active(ctx):
            return False
        count = int(getattr(self, "state_sync_correction_burst_count", 0) or 0)
        if count <= 0:
            return False
        interval = float(getattr(self, "state_sync_correction_burst_interval", 0.10) or 0.0)
        if interval < 0.0:
            interval = 0.0
        current_remaining = int(getattr(ctx, "correction_burst_remaining", 0) or 0)
        ctx.correction_burst_remaining = max(current_remaining, count)
        ctx.correction_burst_interval_s = interval
        return True

    def _send_state_sync_snapshot(
        self,
        ctx: ClientContext,
        *,
        reason: str,
        include_view_update: bool = True,
        replay_timestamp: Optional[int] = None,
    ) -> None:
        """Send an on-demand local state snapshot for timing/state resync.

        The decompile-backed STATE_REQUEST (0x0C) path is used for targeted
        local resync. Send the authoritative UPDATE_ARRAY snapshot to any
        compatible client and optionally pair it with a VIEW_UPDATE replay
        snapshot keyed to the caller's request timing.
        """
        if not self.udp_handler or not ctx.session or not ctx.session.udp_addr:
            return

        # Do not fabricate local-player sync packets before the session has an
        # actual spawned in-game entity. Pre-spawn spectator/player IDs are not
        # safe stand-ins for the OG client's local vehicle sync path.
        if not ctx.session.in_game or ctx.session.entity_id == 0:
            if self.debug_udp_raw:
                print(
                    f"[STATE-SYNC] Skipping {reason} for client {ctx.client_id}: "
                    f"in_game={int(ctx.session.in_game)} entity_id={ctx.session.entity_id}"
                )
            return

        entity_id = ctx.session.entity_id
        if entity_id == 0:
            return

        now = time.monotonic()
        if (now - ctx.last_state_sync_send) < 0.10:
            return
        ctx.last_state_sync_send = now

        def _wrap_angle(angle: float) -> float:
            while angle > math.pi:
                angle -= 2.0 * math.pi
            while angle < -math.pi:
                angle += 2.0 * math.pi
            return angle

        tick = self._get_network_tick(ctx)
        snapshot_mode = getattr(self, "state_sync_snapshot_mode", "remote_live")
        use_history_snapshot = snapshot_mode == "history" or (
            snapshot_mode == "remote_live" and handlers._is_loopback_client(ctx)
        )
        # Live OG telemetry shows stale replay-history poses can be accepted
        # under a fresh VIEW_UPDATE timestamp and drag local prediction behind
        # the server during real movement. Keep history available for loopback
        # tooling/A-B runs, but default remote OG correction to current state.
        snapshot = (
            self._select_authoritative_state_snapshot(ctx, replay_timestamp)
            if use_history_snapshot
            else None
        )
        if snapshot is None:
            snapshot_source = "live"
            send_pos = self._to_client_pos(ctx.player_pos)
            sync_vel = ctx.player_vel
            update_rot = (
                ctx.player_pose.get("roll", 0.0),
                ctx.player_pose.get("pitch", 0.0),
                ctx.player_heading,
            )
        else:
            snapshot_source = "history"
            send_pos = snapshot["pos"]
            sync_vel = snapshot["vel"]
            update_rot = snapshot["rot"]
        include_sync_vel = True
        include_sync_rot = True
        # VIEW_UPDATE is a replay/correction wrapper over the same entity
        # transform payload as UPDATE_ARRAY. The OG reconcile path verifies the
        # buffered predicted rotation against that entity/body rotation, not the
        # client camera-yaw convention. Using player_yaw here flips the sign for
        # remote/OG clients and makes targeted corrections fail intermittently.
        view_rot = update_rot
        health_val = self._get_health_value(ctx)
        fuel_val = self._get_energy_value(ctx)
        if handlers._is_loopback_client(ctx):
            local_state_kwargs = self._get_local_state_kwargs(ctx)
            update_include_local_state = self._should_send_local_state(
                ctx,
                local_state_kwargs["primary_turret_bits"],
                local_state_kwargs["secondary_turret_bits"],
                self.update_local_state_mode,
            )
        else:
            # OG targeted sync is still sensitive to the local-state prefix on
            # these packets. Keep STATE_REQUEST replies on the same short-form-
            # safe shape that earlier live traces showed as stable.
            update_include_local_state, local_state_kwargs = self._get_update_array_local_state_for_viewer(ctx)
        weapon_type = local_state_kwargs["weapon_id"]
        ammo_bits = local_state_kwargs["ammo_count_bits"]
        ammo_mask = local_state_kwargs["ammo_count"]
        pt_bits = local_state_kwargs["primary_turret_bits"]
        pt_angle = local_state_kwargs["primary_turret_angle"]
        st_bits = local_state_kwargs["secondary_turret_bits"]
        st_angle = local_state_kwargs["secondary_turret_angle"]
        if update_include_local_state:
            update_ammo_bits = ammo_bits
            update_ammo_mask = ammo_mask
            update_pt_bits = pt_bits
            update_pt_angle = pt_angle
            update_st_bits = st_bits
            update_st_angle = st_angle
        else:
            update_ammo_bits = 0
            update_ammo_mask = 0
            update_pt_bits = 0
            update_pt_angle = 0.0
            update_st_bits = 0
            update_st_angle = 0.0

        if self._wf_remote_heartbeat_entity_mode(ctx):
            update_payload = self._build_remote_sync_heartbeat_update(
                ctx,
                tick=tick,
                pos=send_pos,
                vel=sync_vel,
                rot=update_rot,
                include_vel=include_sync_vel,
                include_rot=include_sync_rot,
                include_local_state=update_include_local_state,
                health=health_val,
                fuel=fuel_val,
                weapon_type=weapon_type,
                ammo_bits=update_ammo_bits,
                ammo_mask=update_ammo_mask,
                pt_bits=update_pt_bits,
                pt_angle=update_pt_angle,
                st_bits=update_st_bits,
                st_angle=update_st_angle,
                safe_local_state=not handlers._is_loopback_client(ctx),
            )
        else:
            update_payload = build_update_array_player_update(
                tick=tick,
                entity_id=entity_id,
                pos=send_pos,
                vel=sync_vel,
                rot=update_rot,
                include_pos=True,
                include_vel=include_sync_vel,
                include_rot=include_sync_rot,
                include_local_state=update_include_local_state,
                include_entity_vitals=self.update_entity_vitals,
                weapon_id=weapon_type,
                health=health_val,
                fuel=fuel_val,
                ammo_count_bits=update_ammo_bits,
                ammo_count=update_ammo_mask,
                primary_turret_bits=update_pt_bits,
                primary_turret_angle=update_pt_angle,
                secondary_turret_bits=update_st_bits,
                secondary_turret_angle=update_st_angle,
                turret_max=self.local_state_turret_max,
                turret_range=self.local_state_turret_range,
                is_manned=True,
                speed_scale=1.0,
            )

        self.udp_handler.send_to(update_payload, ctx.session.udp_addr)
        ctx.state_sync_reply_count += 1
        ctx.last_state_sync_reply_time = now
        ctx.last_state_sync_reply_tick = tick
        ctx.last_state_sync_replay_timestamp = int(replay_timestamp or 0)
        ctx.last_state_sync_snapshot_source = snapshot_source

        view_include_local_state = False
        view_payload = b""
        if include_view_update:
            # VIEW_UPDATE is replay-mode input to the client's prediction path.
            # Loopback/Python clients keep the STATE_REQUEST/replay id for
            # latency correlation. Remote OG clients need a current/future
            # wrapper timestamp because UpdateArray_check_eligible rejects
            # stale interp_record+0x08 values before prediction storage.
            view_timestamp = replay_timestamp
            if not handlers._is_loopback_client(ctx):
                view_timestamp = self._fresh_remote_view_update_timestamp(ctx, tick)
            view_weapon_type = weapon_type
            view_ammo_bits = 0
            view_ammo_mask = 0
            view_pt_bits = 0
            view_pt_angle = 0.0
            view_st_bits = 0
            view_st_angle = 0.0
            if handlers._is_loopback_client(ctx):
                if self.view_update_local_stats:
                    view_mode = "wf" if self.update_local_state_mode == "wf" else "auto"
                    view_include_local_state = self._should_send_local_state(
                        ctx,
                        pt_bits,
                        st_bits,
                        view_mode,
                    )
                    if view_include_local_state:
                        view_ammo_bits = ammo_bits
                        view_ammo_mask = ammo_mask
                        view_pt_bits = pt_bits
                        view_pt_angle = pt_angle
                        view_st_bits = st_bits
                        view_st_angle = st_angle
            else:
                # VIEW_UPDATE shares the same local-state parser as UPDATE_ARRAY.
                # Keep the replay companion on the same short-form-safe prefix as
                # the targeted UPDATE_ARRAY reply; the transform/timestamp payload
                # carries the correction signal, not expanded ammo/turret bits.
                view_include_local_state = update_include_local_state
                view_weapon_type = weapon_type
                view_ammo_bits = update_ammo_bits
                view_ammo_mask = update_ammo_mask
                view_pt_bits = update_pt_bits
                view_pt_angle = update_pt_angle
                view_st_bits = update_st_bits
                view_st_angle = update_st_angle

            view_payload = build_view_update_player_update(
                tick=tick,
                entity_id=entity_id,
                pos=send_pos,
                vel=sync_vel,
                rot=view_rot,
                include_pos=True,
                include_vel=include_sync_vel,
                include_rot=include_sync_rot,
                include_local_state=view_include_local_state,
                include_entity_vitals=self.view_update_entity_vitals,
                weapon_id=view_weapon_type,
                health=health_val,
                fuel=fuel_val,
                ammo_count_bits=view_ammo_bits,
                ammo_count=view_ammo_mask,
                primary_turret_bits=view_pt_bits,
                primary_turret_angle=view_pt_angle,
                secondary_turret_bits=view_st_bits,
                secondary_turret_angle=view_st_angle,
                turret_max=self.local_state_turret_max,
                turret_range=self.local_state_turret_range,
                is_manned=True,
                speed_scale=1.0,
                timestamp=view_timestamp,
            )
            self.udp_handler.send_to(view_payload, ctx.session.udp_addr)
            ctx.state_sync_view_reply_count += 1

        ctx.last_state_sync_reason = reason
        ctx.last_state_sync_update_len = len(update_payload)
        ctx.last_state_sync_view_len = len(view_payload)
        ctx.last_state_sync_update_has_local_state = bool(update_include_local_state)
        ctx.last_state_sync_view_has_local_state = bool(view_include_local_state)
        ctx.last_state_sync_view_timestamp = int(view_timestamp or 0) if include_view_update else 0
        ctx.last_state_sync_update_hex = update_payload[:32].hex()
        ctx.last_state_sync_view_hex = view_payload[:32].hex()

        if self.pktlog.enabled:
            self.pktlog.log(
                client_id=ctx.client_id,
                label="STATE_SYNC_UPDATE",
                tick=tick,
                payload=update_payload,
                transport="UDP",
                entity_count=1,
                entity_ids=(entity_id,),
                mask_bits=(0b1110,),
                has_local_state=update_include_local_state,
                health=health_val,
                extra=(
                    f"reason={reason} replay=0x{int(replay_timestamp or 0):08x} "
                    f"source={snapshot_source}"
                ),
            )
            if include_view_update:
                self.pktlog.log(
                    client_id=ctx.client_id,
                    label="STATE_SYNC_VIEW",
                    tick=tick,
                    payload=view_payload,
                    transport="UDP",
                    entity_count=1,
                    entity_ids=(entity_id,),
                    mask_bits=(0b1110,),
                    has_local_state=view_include_local_state,
                    health=health_val,
                    extra=(
                        f"reason={reason} replay=0x{int(replay_timestamp or 0):08x} "
                        f"view_ts=0x{int(view_timestamp or 0):08x} source={snapshot_source}"
                    ),
                )

        if self.debug_viewpoint or self.debug_udp_raw:
            print(
                f"[STATE-SYNC] client={ctx.client_id} reason={reason} "
                f"tick={tick} pos=({send_pos[0]:.2f},{send_pos[1]:.2f},{send_pos[2]:.2f}) "
                f"view={int(include_view_update)}"
            )

    # ============ Weapon System Handlers ============

    def _log_fire_pose_context(self, ctx: ClientContext, client_tick: int, source: str) -> None:
        """Trace pose choices used for projectile origin diagnostics."""
        if not self.debug_projectiles:
            return

        def _fmt_vec(vec: Optional[tuple]) -> str:
            if not vec:
                return "None"
            return ",".join(f"{float(v):.2f}" for v in vec)

        now = time.monotonic()
        last = ctx.last_sent_player_state or {}
        last_pos = last.get("pos")
        last_tick = last.get("tick")
        last_dt = now - float(last.get("time", now))

        hist = self._select_authoritative_state_snapshot(ctx, client_tick) if client_tick else None
        hist_pos = hist.get("pos") if hist else None
        hist_tick = hist.get("tick") if hist else None
        hist_dt = now - float(hist.get("time", now)) if hist else 0.0

        print(
            f"[FIRE-POSE] src={source} client={ctx.client_id} "
            f"client_tick={client_tick} server_tick={get_ticks()} "
            f"session_tick={ctx.session.tick if ctx.session else 0} "
            f"tick_offset={ctx.tick_offset} "
            f"player_pos=({_fmt_vec(self._to_client_pos(ctx.player_pos))}) "
            f"player_vel=({_fmt_vec(ctx.player_vel)}) "
            f"last_sent_tick={last_tick} last_sent_dt={last_dt:.3f}s "
            f"last_sent_pos=({_fmt_vec(last_pos)}) "
            f"hist_tick={hist_tick} hist_dt={hist_dt:.3f}s "
            f"hist_pos=({_fmt_vec(hist_pos)})"
        )

    def _select_weapon_fire_pose(
        self,
        ctx: ClientContext,
        client_tick: int,
    ) -> tuple[tuple, tuple, str, str]:
        """Return the packet-aligned pose used to spawn a projectile.

        Fire packets arrive after the client has already simulated the input
        frame that produced them. Using the live server pose makes moving shots
        spawn a few units ahead of the client's local muzzle. Reuse the same
        replay/history mapping as STATE_REQUEST so projectile origin and local
        correction agree on the input tick.
        """
        pose_source = "live"
        fire_pos = ctx.player_pos
        body_roll = float(ctx.player_pose.get("roll", 0.0) or 0.0)
        body_pitch = float(ctx.player_pose.get("pitch", 0.0) or 0.0)
        body_yaw = float(ctx.player_heading)

        hist = self._select_authoritative_state_snapshot(ctx, client_tick) if client_tick else None
        if hist is not None:
            hist_pos = hist.get("pos")
            hist_rot = hist.get("rot") or ()
            if hist_pos is not None:
                fire_pos = self._from_client_pos(hist_pos)
                pose_source = "history"
            if len(hist_rot) >= 1:
                body_roll = float(hist_rot[0])
            if len(hist_rot) >= 2:
                body_pitch = float(hist_rot[1])
            if len(hist_rot) >= 3:
                body_yaw = float(hist_rot[2])

        aim_pitch, aim_yaw, aim_src = self._get_aim_rotation(ctx)
        aim_label = aim_src
        if self.projectile_aim_source == "body":
            aim_pitch = 0.0
            aim_yaw = body_yaw
            aim_label = "body"
        elif self.projectile_aim_source == "viewpoint":
            aim_pitch = ctx.player_aim_pitch
            aim_yaw = ctx.player_aim_yaw
            aim_label = "viewpoint"
        elif self.projectile_aim_source == "auto" and aim_src != "viewpoint":
            aim_pitch = body_pitch
            aim_yaw = body_yaw

        return fire_pos, (body_roll, aim_pitch, aim_yaw), aim_label, pose_source

    def _handle_action_dump(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle ACTION_DUMP packet (0x09).
        Contains all behavior slot values including fire state.
        Format: [opcode:1] [tick:4] [frame:4] [slot_data:bit-packed]
        """
        if ctx is None:
            return

        client_tick = 0
        if len(data) >= 5:
            try:
                client_tick = struct.unpack(">I", data[1:5])[0]
                self._sync_tick_offset(ctx, client_tick)
            except struct.error:
                pass
        if ctx.weapon_system.decode_action_dump(data):
            ctx.last_action_dump_time = time.monotonic()
            self._record_client_action_telemetry(ctx, "ACTION_DUMP", client_tick)

            if ctx.session and not ctx.session.input_ready:
                ctx.session.input_ready = True
                ctx.session.input_ready_time = time.monotonic()
                print(f"[GAME] Client {ctx.client_id}: input ready (ACTION_DUMP)")
            self._update_player_aim(ctx)
            # Update weapon system with current pose
            ctx.weapon_system.player_id = ctx.session.player_id or ctx.entity_id
            ctx.weapon_system.player_team = ctx.session.team_id or 2
            fire_pos, fire_rot, aim_src, pose_src = self._select_weapon_fire_pose(ctx, client_tick)
            ctx.weapon_system.player_pos = fire_pos
            ctx.weapon_system.player_rot = fire_rot
            ctx.weapon_system.projectile_aim_source = aim_src

            # Weapon spawning with wulf-forge encoding
            if self.projectiles_enabled:
                energy_available = ctx.player_energy if self.weapon_energy_enabled else None
                new_projectiles, energy_spent = ctx.weapon_system.update(
                    available_energy=energy_available
                )
                if energy_spent > 0.0:
                    self._consume_player_energy(ctx, energy_spent)
                if new_projectiles:
                    yaw_deg = math.degrees(ctx.player_yaw)
                    turn_val = ctx.weapon_system.behavior_slots[BehaviorSlot.TURNING]
                    aim_pitch, aim_yaw, aim_src = self._get_aim_rotation(ctx)
                    vp_yaw = math.degrees(ctx.player_aim_yaw)
                    print(
                        f"[WEAPON-FIRE] Firing {len(new_projectiles)} proj "
                        f"heading={math.degrees(ctx.player_heading):.1f} "
                        f"aim_yaw={math.degrees(fire_rot[2]):.1f} vp_yaw={vp_yaw:.1f} "
                        f"src={aim_src} pose_src={pose_src} "
                        f"turn_slot={turn_val:.3f} pos={ctx.player_pos} "
                        f"fire_pos={fire_pos}"
                    )
                    self._log_fire_pose_context(ctx, client_tick, "ACTION_DUMP")
                    self._record_client_weapon_fire(
                        ctx,
                        "ACTION_DUMP",
                        client_tick,
                        new_projectiles,
                        energy_spent,
                    )
                for proj in new_projectiles:
                    self._spawn_moving_projectile(ctx, proj, addr)

            # Legacy packet-arrival hook; fixed-step jumpjets run in motion tick.
            self._process_jump_jets(ctx, addr)
        else:
            ctx.action_dump_decode_fail_count += 1
            ctx.last_action_dump_decode_fail_hex = data[:48].hex()
            if self.debug_sync:
                print(
                    f"[UDP] ACTION_DUMP decode failed c{ctx.client_id}: "
                    f"len={len(data)} data={ctx.last_action_dump_decode_fail_hex}"
                )

    def _handle_action_update(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle ACTION_UPDATE packet (0x0A).
        Contains incremental behavior slot updates.
        """
        if ctx is None:
            return

        if self.debug_sync:
            print(f"[UDP] ACTION_UPDATE received: len={len(data)} data={data.hex()}")
        client_tick = 0
        if len(data) >= 6:
            try:
                client_tick = struct.unpack(">I", data[2:6])[0]
                self._sync_tick_offset(ctx, client_tick)
            except struct.error:
                pass
        if ctx.weapon_system.decode_action_update(data):
            ctx.last_action_dump_time = time.monotonic()
            self._record_client_action_telemetry(ctx, "ACTION_UPDATE", client_tick)
            if ctx.session and not ctx.session.input_ready:
                ctx.session.input_ready = True
                ctx.session.input_ready_time = time.monotonic()
                print(f"[GAME] Client {ctx.client_id}: input ready (ACTION_UPDATE)")
            # Backdating is intentionally disabled. We apply input on packet arrival
            # and rely on deterministic lockstep physics, matching the original model.

            self._update_player_aim(ctx)
            # Yaw is tracked via VIEWPOINT_INFO when available; otherwise input-based fallback is used.
            # Position is simulated in the tick loop from behavior slots.

            # Update weapon system with current pose
            ctx.weapon_system.player_id = ctx.session.player_id or ctx.entity_id
            ctx.weapon_system.player_team = ctx.session.team_id or 2
            fire_pos, fire_rot, aim_src, pose_src = self._select_weapon_fire_pose(ctx, client_tick)
            ctx.weapon_system.player_pos = fire_pos
            ctx.weapon_system.player_rot = fire_rot
            ctx.weapon_system.projectile_aim_source = aim_src

            # Weapon spawning with wulf-forge encoding
            if self.projectiles_enabled:
                energy_available = ctx.player_energy if self.weapon_energy_enabled else None
                new_projectiles, energy_spent = ctx.weapon_system.update(
                    available_energy=energy_available
                )
                if energy_spent > 0.0:
                    self._consume_player_energy(ctx, energy_spent)
                if new_projectiles:
                    turn_val = ctx.weapon_system.behavior_slots[BehaviorSlot.TURNING]
                    vp_yaw = math.degrees(ctx.player_aim_yaw)
                    print(
                        f"[WEAPON-FIRE] Firing {len(new_projectiles)} proj "
                        f"heading={math.degrees(ctx.player_heading):.1f} "
                        f"aim_yaw={math.degrees(fire_rot[2]):.1f} vp_yaw={vp_yaw:.1f} "
                        f"src={aim_src} pose_src={pose_src} "
                        f"turn_slot={turn_val:.3f} pos={ctx.player_pos} "
                        f"fire_pos={fire_pos}"
                    )
                    self._log_fire_pose_context(ctx, client_tick, "ACTION_UPDATE")
                    self._record_client_weapon_fire(
                        ctx,
                        "ACTION_UPDATE",
                        client_tick,
                        new_projectiles,
                        energy_spent,
                    )
                for proj in new_projectiles:
                    self._spawn_moving_projectile(ctx, proj, addr)

            # Legacy packet-arrival hook; fixed-step jumpjets run in motion tick.
            self._process_jump_jets(ctx, addr)
        else:
            ctx.action_update_decode_fail_count += 1
            ctx.last_action_update_decode_fail_hex = data[:48].hex()
            if self.debug_sync:
                print(
                    f"[UDP] ACTION_UPDATE decode failed c{ctx.client_id}: "
                    f"len={len(data)} data={ctx.last_action_update_decode_fail_hex}"
                )

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
        cycle_slots = sorted(TANK_WEAPON_SLOTS)
        current = ctx.weapon_system.current_weapon
        if current in cycle_slots:
            current_idx = cycle_slots.index(current)
        else:
            current_idx = 0

        if mode == 1:  # Cycle forward
            ctx.weapon_system.current_weapon = cycle_slots[(current_idx + 1) % len(cycle_slots)]
            print(f"[WEAPON] Cycled forward to weapon slot {ctx.weapon_system.current_weapon}")
        elif mode == 2:  # Cycle backward
            ctx.weapon_system.current_weapon = cycle_slots[(current_idx - 1) % len(cycle_slots)]
            print(f"[WEAPON] Cycled backward to weapon slot {ctx.weapon_system.current_weapon}")
        elif slot != ctx.weapon_system.current_weapon:
            # Direct weapon selection via slot parameter
            if slot in TANK_WEAPON_SLOTS:
                ctx.weapon_system.current_weapon = slot
                print(f"[WEAPON] Selected weapon slot {slot}")
            else:
                print(f"[WEAPON] Ignoring invalid weapon slot request: {slot}")

        # Python-client-only debug feedback; not part of OG gameplay traffic.
        if ctx.tcp_handler and self._debug_comm_allowed_for_client(ctx):
            weapon = WEAPON_NAMES.get(ctx.weapon_system.current_weapon, f"Weapon {ctx.weapon_system.current_weapon}")
            msg = build_chat_message(f"[{weapon}]", source_id=ctx.session.player_id or ctx.entity_id)
            ctx.tcp_handler.send(msg)

    def _handle_input_feedback(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle INPUT_FEEDBACK packet (0x40).
        Contains player input state - frame counter and possibly input flags.
        """
        if len(data) >= 5:
            frame_counter = struct.unpack(">I", data[1:5])[0]
            # Count INPUT_FEEDBACK for frame-rate estimation
            if ctx is not None:
                ws = ctx.weapon_system
                ws.on_input_feedback()
                ctx.input_feedback_count += 1
                ctx.last_input_feedback_time = time.monotonic()
            # Additional data might contain input flags
            if len(data) > 5:
                extra = data[5:]
                if extra and extra != b'\x00' * len(extra):
                    print(f"[INPUT] Extra data: {extra.hex()}")

    def _on_chain_gun_fire(self, ctx: ClientContext, pos: tuple, rot: tuple, team: int, weapon_name: str = None):
        """Callback when weapon fires (instant hit or placeholder for projectiles)."""
        weapon_name = weapon_name or "Chain Gun"
        print(f"[WEAPON] {weapon_name} fired! pos={pos}")
        # Python-client-only debug feedback; suppress for OG clients.
        if ctx.tcp_handler and self._debug_comm_allowed_for_client(ctx):
            if weapon_name == "Chain Gun":
                msg = build_chat_message("*ratatatat*", source_id=ctx.session.player_id or ctx.entity_id)
            else:
                # Other weapons get descriptive feedback
                msg = build_chat_message(f"*{weapon_name.lower()} fired*", source_id=ctx.session.player_id or ctx.entity_id)
            ctx.tcp_handler.send(msg)

    def _broadcast_weapon_fire_fx(self, ctx: ClientContext, proj):
        """Broadcast TRANSIENT_ARRAY weapon fire FX to all clients except the firer.

        Uses decompile-backed quantized bitstream format (0x0046CA60).
        Sent via UDP only — FX is cosmetic, loss is acceptable.
        """
        # Map projectile entity types to FX types
        _PROJ_TO_FX = {
            EntityType.FLAK_SHELL: FX_FLAK_FIRE,
            EntityType.PULSE_SHELL: FX_PULSE_FIRE,
            EntityType.HUNTER: FX_MISSILE_FIRE,
            EntityType.PIERCER: FX_PULSE_FIRE,
            EntityType.THUMPER: FX_FLAK_FIRE,
        }
        fx_type = _PROJ_TO_FX.get(proj.entity_type, FX_CHAIN_GUN_FIRE)

        events = [{
            'type': fx_type,
            'pos': proj.pos,
            'entity_id': ctx.entity_id,
        }]
        self._broadcast_transient_fx(events, exclude_client=ctx)

    def _transient_fx_allowed_for_client(self, ctx: ClientContext) -> bool:
        """Return whether cosmetic TRANSIENT_ARRAY FX are currently safe for a client."""
        if handlers._is_loopback_client(ctx):
            return True
        return getattr(self, "remote_transient_fx", False)

    def _projectile_packets_allowed_for_client(self, ctx: ClientContext) -> bool:
        """Return whether projectile entity packets are currently safe for a client."""
        if handlers._is_loopback_client(ctx):
            return True
        return getattr(self, "remote_projectiles", True)

    def _broadcast_transient_fx(self, events: list, *, exclude_client=None) -> bytes:
        """Broadcast cosmetic TRANSIENT_ARRAY FX on the safest currently supported path."""
        pkt = build_transient_array(events)
        if not pkt:
            return b""

        for target in self._snapshot_in_game_clients():
            if target is exclude_client:
                continue
            if not self._transient_fx_allowed_for_client(target):
                continue
            if self.udp_handler and target.session.udp_addr:
                self.udp_handler.send_to(pkt, target.session.udp_addr)
        return pkt

    def _on_projectile_spawn(self, ctx: ClientContext, proj):
        """Callback when a projectile is spawned."""
        print(f"[WEAPON] Projectile spawned: id={proj.entity_id} type={proj.entity_type.name}")

        # NOTE: Do NOT send spawn here to avoid duplicate spawns.
        # Spawn is handled by _spawn_moving_projectile to prevent TCP/UDP reorders.

        # Python-client-only debug feedback; suppress for OG clients.
        if ctx.tcp_handler and self._debug_comm_allowed_for_client(ctx):
            from .packets import build_chat_message
            msg = build_chat_message("*PEW*", source_id=ctx.session.player_id or ctx.entity_id)
            ctx.tcp_handler.send(msg)

    def _send_projectile_spawn(self, ctx: ClientContext, proj, addr: tuple):
        """Send packet to spawn a projectile entity."""
        sent_count = 0
        if self.udp_handler:
            for target in self._snapshot_in_game_clients():
                if not target.session.udp_addr or not target.session.translation_ack_received:
                    continue
                if not self._projectile_packets_allowed_for_client(target):
                    continue
                tick = self._get_network_tick(target)
                include_local_state, local_state_kwargs = self._get_projectile_local_state_for_viewer(target)
                packet = build_projectile_spawn_packet(
                    proj,
                    tick,
                    include_local_state=include_local_state,
                    **local_state_kwargs,
                    entity_config=self.projectile_config,
                    is_static=self.projectile_spawn_snap,
                )
                self.udp_handler.send_to(packet, target.session.udp_addr)
                # UPDATE_ARRAY (0x0E) over TCP crashes OG client (TCP bitstream
                # desync â†’ protocol mismatch). UDP-only is fine for projectiles.
                if self.pktlog.enabled:
                    self.pktlog.log(
                        client_id=target.client_id,
                        label="PROJ_SPAWN",
                        tick=tick,
                        payload=packet,
                        transport="UDP",
                        entity_count=1,
                        entity_ids=(proj.entity_id,),
                        mask_bits=(0b1111,),  # pos+vel+rot+type_info
                        has_local_state=include_local_state,
                        health=self._get_health_value(target) if include_local_state else -1.0,
                        extra=f"proj_type={proj.entity_type}",
                    )
                sent_count += 1
        if sent_count:
            print(f"[WEAPON] Sent projectile spawn via UDP: id={proj.entity_id} targets={sent_count}")
        if self.debug_projectiles:
            print(
                f"[PROJ-SPAWN] id={proj.entity_id} type={proj.entity_type} "
                f"config={self.projectile_config} spawn_snap={int(self.projectile_spawn_snap)}"
            )
            if os.environ.get("WULFRAM_DEBUG_PROJECTILE_HEX", "0") == "1":
                print(f"[PROJ-HEX] id={proj.entity_id} (per-client packets, no single hex to show)")

        # Send chat message with projectile position for debugging (TCP only).
        pos_msg = f"FIRE! pos=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f}) vel=({proj.vel[0]:.1f},{proj.vel[1]:.1f},{proj.vel[2]:.1f})"
        if ctx.tcp_handler and self._debug_comm_allowed_for_client(ctx):
            chat_packet = build_chat_message(pos_msg, source_id=ctx.session.player_id or ctx.entity_id)
            ctx.tcp_handler.send(chat_packet)
        print(f"[WEAPON] {pos_msg}")

        # Belt-and-suspenders: send an immediate heartbeat UPDATE_ARRAY to the
        # firing player so their HUD health stays current even if the projectile
        # local_state bits are somehow missed or arrive out of order.
        if self.udp_handler and ctx.session.udp_addr:
            hb_tick = self._get_network_tick(ctx)
            hb_health = self._get_health_value(ctx)
            hb_packet = self._build_local_state_heartbeat(
                ctx,
                tick=hb_tick,
                entity_id=ctx.session.entity_id,
                include_health=True,
                health=hb_health,
                fuel=self._get_energy_value(ctx),
            )
            self.udp_handler.send_to(hb_packet, ctx.session.udp_addr)
            if self.pktlog.enabled:
                self.pktlog.log(
                    client_id=ctx.client_id,
                    label="PROJ_FIRE_HEARTBEAT",
                    tick=hb_tick,
                    payload=hb_packet,
                    transport="UDP",
                    entity_count=1,
                    entity_ids=(0xFFFFFFFE,),
                    mask_bits=(0,),
                    has_local_state=True,
                    health=hb_health,
                )
            print(f"[PROJ-HEARTBEAT] Sent post-fire heartbeat UPDATE_ARRAY (health={hb_health:.2f})")

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
            sx = ctx.player_pos[0] + dx
            sy = ctx.player_pos[1] + dy
            sz = ctx.player_pos[2] + dz
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
                ctx.player_pos,
                ctx.player_vel,
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

    def _apply_jump_jets_fixed_step(
        self,
        ctx: ClientContext,
        *,
        dt: float,
        jumpjet_input: float,
        current_altitude: float,
        current_vel_up: float,
    ) -> tuple[float, bool, float]:
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
                current_vel_up += impulse
                ctx.jump_cooldown_remaining = cfg.cooldown
                fired = True
                if cfg.fuel_cost > 0.0:
                    self._consume_player_energy(ctx, cfg.fuel_cost)
                player_id = ctx.session.player_id or ctx.entity_id
                self._on_jump_jet_triggered(ctx, player_id, impulse, current_vel_up)

        ctx.jump_prev_thrust_input = jumpjet_input
        return current_vel_up, fired, impulse

    def _record_client_weapon_fire(
        self,
        ctx: ClientContext,
        packet_type: str,
        client_tick: int,
        projectiles,
        energy_spent: float,
    ) -> None:
        """Record client-input-triggered projectile fire for live T3 gates."""
        if ctx is None or ctx.weapon_system is None:
            return
        projectiles = list(projectiles or [])
        if not projectiles:
            return

        now = time.monotonic()
        ws = ctx.weapon_system
        active_slots = {
            str(idx): float(value)
            for idx, value in enumerate(ws.behavior_slots)
            if abs(float(value)) > 0.001
        }
        direct_slots = {
            str(idx): float(ws.behavior_slots[idx])
            for idx in OG_DIRECT_TRIGGER_WEAPON_SLOTS
            if idx < len(ws.behavior_slots) and abs(float(ws.behavior_slots[idx])) > 0.001
        }

        ctx.weapon_fire_count = int(getattr(ctx, "weapon_fire_count", 0) or 0) + len(projectiles)
        ctx.last_weapon_fire_time = now
        ctx.last_weapon_fire_source = packet_type
        ctx.last_weapon_fire_client_tick = int(client_tick or 0)
        ctx.last_weapon_fire_projectile_ids = [int(getattr(proj, "entity_id", 0) or 0) for proj in projectiles]
        ctx.last_weapon_fire_projectile_types = [
            getattr(getattr(proj, "entity_type", None), "name", str(getattr(proj, "entity_type", "")))
            for proj in projectiles
        ]
        ctx.last_weapon_fire_energy_spent = float(energy_spent or 0.0)
        ctx.last_weapon_fire_input = {
            "active_slots": active_slots,
            "direct_slots": direct_slots,
            "fire": float(ws.behavior_slots[BehaviorSlot.FIRE]),
            "thrust": float(tank_softbody_control_slot_value(ws.behavior_slots)),
            "jumpjet": float(ws.behavior_slots[BehaviorSlot.JUMPJET]),
        }

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

    def _remote_og_movement_input_delay_for_ctx(self, ctx: ClientContext) -> float:
        """Return the empirically observed remote OG input replay delay."""
        if ctx is None or ctx.injected_input is not None:
            return 0.0
        if handlers._is_loopback_client(ctx):
            return 0.0
        return max(0.0, float(getattr(self, "remote_og_movement_input_delay", 0.0) or 0.0))

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
        if delay <= 0.0:
            return float(current_fwd), float(current_strafe), "current_slots"
        history = getattr(ctx, "movement_input_history", None)
        if not history:
            return float(current_fwd), float(current_strafe), "current_slots_no_history"

        target_time = time.monotonic() - delay
        selected = None
        for entry in reversed(history):
            try:
                sample_time = float(entry.get("time", 0.0))
            except (TypeError, ValueError, AttributeError):
                continue
            if sample_time <= target_time:
                selected = entry
                break

        if selected is None:
            # The first nonzero packet can arrive before OG local physics has
            # consumed the key event. Hold neutral during that short replay
            # window instead of advancing the server early.
            return 0.0, 0.0, "delayed_remote_og_pre_history_zero"

        try:
            fwd = float(selected.get("fwd", 0.0))
        except (TypeError, ValueError, AttributeError):
            fwd = 0.0
        try:
            strafe = float(selected.get("strafe", 0.0))
        except (TypeError, ValueError, AttributeError):
            strafe = 0.0
        return fwd, strafe, "delayed_remote_og_action_history"

    def _maybe_promote_remote_full_local_state(self, ctx: ClientContext, *, reason: str) -> bool:
        """Leave the spawn-safe minimal remote path once the client is stably in game."""
        if self.update_local_state_mode != "wf":
            return False
        if handlers._is_loopback_client(ctx):
            return False
        if getattr(ctx, "remote_full_local_state_ready", False):
            return False
        if not ctx.session or not ctx.session.in_game:
            return False
        if reason in ("heartbeat", "action_update", "action_dump"):
            return False
        spawn_time = getattr(ctx.session, "last_spawn_time", 0.0) or 0.0
        if spawn_time > 0.0 and (time.monotonic() - spawn_time) < self.remote_full_local_state_delay:
            return False
        ctx.remote_full_local_state_ready = True
        ctx._spawn_safe_heartbeat_suppressed_logged = False
        print(
            f"[LOCAL-STATE] client={ctx.client_id} promoted to full remote sync "
            f"reason={reason}"
        )
        return True

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

    def _update_player_position(self, ctx: ClientContext, dt_override: float = 0.0, heading_override: float = None):
        """
        Simulate player position using damped persistent velocity model.

        Matches client's RigidBody_integrate_position (Physics.c:5101-5136):
          1. Vehicle controller computes per-frame impulse (zeroed each frame)
          2. effective_acc = impulse - vel * linear_damp  (damped mode, flag 0xc0+3)
          3. pos += vel * dt + 0.5 * effective_acc * dtÂ²  (Verlet integration)
          4. vel += effective_acc * dt  (velocity persists across frames)

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

        dt = dt_override if dt_override > 0 else 1.0 / self.tick_rate_hz

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
        if vertical_idx == 2:
            jump_altitude = ctx.player_pos[2] - ground_level if ground_level is not None else ctx.player_pos[2]
            vel_z, jump_jet_fired, jump_jet_impulse = self._apply_jump_jets_fixed_step(
                ctx,
                dt=dt,
                jumpjet_input=jumpjet_input,
                current_altitude=jump_altitude,
                current_vel_up=vel_z,
            )
        else:
            jump_altitude = ctx.player_pos[1] - ground_level if ground_level is not None else ctx.player_pos[1]
            vel_y, jump_jet_fired, jump_jet_impulse = self._apply_jump_jets_fixed_step(
                ctx,
                dt=dt,
                jumpjet_input=jumpjet_input,
                current_altitude=jump_altitude,
                current_vel_up=vel_y,
            )

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
        # (from RigidBody_integrate_position, damped mode at Physics.c:5126-5129)
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

        # Verlet integration: pos += vel * dt + 0.5 * acc * dtÂ²
        # (from Vec3_integrate_motion, Physics.c:4396-4410)
        x, y, z = ctx.player_pos
        half_dt2 = 0.5 * dt * dt
        new_x = x + vel_x * dt + acc_x * half_dt2
        new_y = y + vel_y * dt + acc_y * half_dt2
        new_z = z + vel_z * dt + acc_z * half_dt2

        # Velocity update: vel += acc * dt
        new_vel_x = vel_x + acc_x * dt
        new_vel_y = vel_y + acc_y * dt
        new_vel_z = vel_z + acc_z * dt

        # Float32 quantization: client stores pos/vel as float32
        if self.f32_physics:
            from .physics import _f32
            new_x = _f32(new_x)
            new_y = _f32(new_y)
            new_z = _f32(new_z)
            new_vel_x = _f32(new_vel_x)
            new_vel_y = _f32(new_vel_y)
            new_vel_z = _f32(new_vel_z)

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
        ctx._world_collision_step_pre_pos = pre_pos
        ctx._world_collision_step_pre_vel = pre_vel
        ctx._world_collision_step_dt = dt
        try:
            new_x, new_y, new_z, new_vel_x, new_vel_y, new_vel_z = self._resolve_entity_world_collision(
                ctx, new_x, new_y, new_z, new_vel_x, new_vel_y, new_vel_z
            )
        finally:
            for attr_name in (
                "_world_collision_step_pre_pos",
                "_world_collision_step_pre_vel",
                "_world_collision_step_dt",
            ):
                if hasattr(ctx, attr_name):
                    delattr(ctx, attr_name)

        # Building AABB collision (matching client-side)
        new_x, new_y, new_vel_x, new_vel_y = self._check_building_collisions(
            ctx, new_x, new_y, new_z, new_vel_x, new_vel_y)

        # Terrain/world contact response can still push the tank back below the
        # terrain plane after the initial post-integrate clamp. Keep the final
        # authoritative pose on or above terrain before replication.
        final_ground_clamped = False
        if self.up_axis == "z":
            if self.terrain and not use_ground_override:
                terrain_z = self._terrain_physics_ground_z_at(new_x, new_y)
            else:
                terrain_z = ground_level
            if new_z < terrain_z:
                new_z = terrain_z
                final_ground_clamped = True
                if new_vel_z < 0.0:
                    new_vel_z = 0.0
        else:
            if new_y < ground_level:
                new_y = ground_level
                final_ground_clamped = True
                if new_vel_y < 0.0:
                    new_vel_y = 0.0
        if final_ground_clamped:
            ctx.world_collision_ref_pos = (new_x, new_y, new_z)

        old_pos = ctx.player_pos
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
        if getattr(ctx, "vehicle_physics", None) is not None:
            ctx.vehicle_physics.angular_velocity = ctx.angular_vel_yaw
        ctx.debug_last_controller_step = {
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
            "terrain_contact_probe": dict(getattr(ctx, "debug_last_terrain_contact_probe", {}) or {}),
            "pos": ctx.player_pos,
            "vel": ctx.player_vel,
        }
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

        def sane_position_triplet(value):
            result = finite_triplet(value)
            if result is None:
                return None
            pos_limit = max(float(getattr(self, "world_bound", 8192.0) or 8192.0) * 4.0, 32768.0)
            if any(abs(component) > pos_limit for component in result):
                return None
            return result

        def sane_velocity_triplet(value):
            result = finite_triplet(value)
            if result is None:
                return None
            if any(abs(component) > 10000.0 for component in result):
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
        pair_solver_response = (
            contact_response in {"pair", "solver", "constraint"}
            or (
                contact_response == "auto"
                and origin_mode in {"entity", "origin", "raw"}
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
                mesh_aabb_half_extents_from_vertices(vertices) or half_extents
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

        def box_collision_z_lift() -> float:
            if terrain_collision_shape == "entity_box":
                return 0.0
            return z_lift if collision_model is not None else half_extents[2]

        def contact_debug_fields(contact):
            return {
                "contact_sector_index": getattr(contact, "sector_index", None),
                "contact_cell": getattr(contact, "cell", None),
                "contact_normal_source": getattr(contact, "normal_source", None),
                "contact_cbsp_split_normal": getattr(contact, "cbsp_split_normal", None),
                "contact_terrain_face_normal": getattr(contact, "terrain_face_normal", None),
                "contact_mesh_face_normal": getattr(contact, "mesh_face_normal", None),
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
        raw_fallback_mode = (
            os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", "0")
            .strip()
            .lower()
        )
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

        def sample_raw_origin_contact_at(pos):
            if (
                collision_model is None
                or vertices is None
                or cbsp_tree is None
                or origin_mode in {"entity", "origin", "raw"}
                or not finite_values((*pos, heading))
            ):
                return None, None, None
            raw_center = (pos[0], pos[1], pos[2])
            try:
                raw_contact = self._terrain_grid_collision.test_model_collision(
                    raw_center,
                    heading,
                    vertices,
                    cbsp_tree,
                    bounding_radius,
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
                        )
                    except Exception:
                        raw_bounds_contact = None
            return raw_contact, raw_bounds_contact, None

        def raw_origin_fallback_reject_reason(raw_contact, *, velocity=None):
            if not raw_fallback_enabled:
                return "disabled"
            if raw_contact is None:
                return "no_raw_origin_contact"
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
            if depth < raw_fallback_min_depth:
                return "below_min_depth"
            if depth > raw_fallback_max_depth:
                return "above_max_depth"
            if normal_z < raw_fallback_min_normal_z:
                return "normal_z_below_min"
            if speed < raw_fallback_min_speed:
                return "speed_below_min"
            return ""

        def raw_origin_contact_for_fallback(raw_contact):
            if raw_contact is None:
                return None
            contact_face_modes = {
                "face",
                "triangle",
                "terrain_face",
                "contact_face",
                "decompile_face",
                "terrain_triangle_contact",
            }
            sampled_terrain_modes = {"terrain", "sampled_terrain", "heightfield"}
            if raw_fallback_normal_source in contact_face_modes:
                face_normal = getattr(raw_contact, "terrain_face_normal", None)
                try:
                    face_normal = (
                        float(face_normal[0]),
                        float(face_normal[1]),
                        float(face_normal[2]),
                    )
                    face_len = math.sqrt(
                        face_normal[0] * face_normal[0]
                        + face_normal[1] * face_normal[1]
                        + face_normal[2] * face_normal[2]
                    )
                except (TypeError, ValueError, OverflowError, IndexError):
                    face_normal = None
                    face_len = 0.0
                if face_normal is not None and face_len > 1e-8:
                    face_normal = (
                        face_normal[0] / face_len,
                        face_normal[1] / face_len,
                        face_normal[2] / face_len,
                    )
                    if face_normal[2] < 0.0:
                        face_normal = (
                            -face_normal[0],
                            -face_normal[1],
                            -face_normal[2],
                        )
                    return TerrainContact(
                        position=raw_contact.position,
                        normal=face_normal,
                        penetration=raw_contact.penetration,
                        sector_index=raw_contact.sector_index,
                        cell=raw_contact.cell,
                        normal_source="terrain_triangle_contact_face",
                        cbsp_split_normal=getattr(raw_contact, "cbsp_split_normal", None),
                        terrain_face_normal=getattr(raw_contact, "terrain_face_normal", None),
                        mesh_face_normal=getattr(raw_contact, "mesh_face_normal", None),
                    )
                if raw_fallback_normal_source not in sampled_terrain_modes:
                    return raw_contact
            if raw_fallback_normal_source not in sampled_terrain_modes:
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
            )

        def sample_raw_origin_fallback_contact_at(pos, *, velocity=None):
            raw_contact, raw_bounds_contact, raw_error = sample_raw_origin_contact_at(pos)
            fallback_contact = raw_origin_contact_for_fallback(raw_contact)
            reject = raw_origin_fallback_reject_reason(
                fallback_contact,
                velocity=velocity,
            )
            return {
                "contact": fallback_contact,
                "raw_contact": fallback_contact,
                "raw_bounds_contact": raw_bounds_contact,
                "raw_error": raw_error,
                "reject": reject,
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

        def update_contact_probe(
            pos,
            lifted_contact,
            *,
            reason,
            raw_contact=None,
            raw_bounds_contact=None,
            raw_error=None,
            raw_fallback_reject=None,
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
            ctx.debug_last_terrain_contact_probe = {
                "reason": probe_reason,
                "origin_mode": origin_mode,
                "contact_response": contact_response,
                "contact_timing_mode": contact_timing_mode,
                "terrain_collision_shape": terrain_collision_shape,
                "probe_enabled": True,
                "position": pos,
                "velocity": (vx, vy, vz),
                "heading": heading,
                "model_z_lift": z_lift,
                "bounding_radius": bounding_radius,
                "raw_origin_fallback_enabled": raw_fallback_enabled,
                "raw_origin_fallback_reject": raw_fallback_reject,
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
                "raw_origin_bounds_contact": probe_contact_fields(
                    raw_bounds_contact,
                    center=raw_center,
                    z_lift_used=0.0,
                ),
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
                projection_order=(
                    projection_order_override
                    if projection_order_override is not None
                    else os.environ.get(
                        "WULFRAM_ENTITY_TERRAIN_PROJECTION_ORDER",
                        "body_minus_world",
                    )
                ),
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

        def apply_raw_origin_fallback_contact(contact):
            nonlocal vx, vy, vz
            before_pos = (anchor[0], anchor[1], anchor[2])
            before_vel = (vx, vy, vz)
            before_ang = tuple(contact_angular_velocity)
            response_debug = apply_contact(
                contact,
                force_pair_solver=True,
                projection_order_override=raw_fallback_projection_order,
                friction_override=raw_fallback_friction,
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
            normal_delta_projected = False
            before_normal_speed = None
            before_center_normal_speed = None
            before_normal_speed_source = ""
            normal_delta_skip_reason = ""
            if raw_fallback_delta_mode in {
                "normal",
                "normal_only",
                "contact_normal",
                "closing",
                "closing_velocity",
                "projection_speed",
                "target_speed",
                "decompile_projection",
            }:
                normal = None
                try:
                    normal_mag = vec_mag(contact.normal)
                    if normal_mag > 1e-9:
                        normal = (
                            float(contact.normal[0]) / normal_mag,
                            float(contact.normal[1]) / normal_mag,
                            float(contact.normal[2]) / normal_mag,
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
                    if raw_fallback_delta_mode in {
                        "closing",
                        "closing_velocity",
                        "projection_speed",
                        "target_speed",
                        "decompile_projection",
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
                            not raw_fallback_closing_only
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
                        if normal_component <= 0.0:
                            normal_delta_skip_reason = "nonpositive_solver_normal_delta"
                        else:
                            normal_delta_skip_reason = "separating_before_velocity"
                        raw_delta = (0.0, 0.0, 0.0)
                        safe_vel = before_vel
                    normal_delta_projected = True
            raw_delta_mag = vec_mag(raw_delta)
            velocity_delta_clamped = False
            if (
                math.isfinite(raw_fallback_max_velocity_delta)
                and raw_fallback_max_velocity_delta > 0.0
                and raw_delta_mag > raw_fallback_max_velocity_delta
            ):
                scale = raw_fallback_max_velocity_delta / max(raw_delta_mag, 1e-9)
                safe_vel = (
                    before_vel[0] + raw_delta[0] * scale,
                    before_vel[1] + raw_delta[1] * scale,
                    before_vel[2] + raw_delta[2] * scale,
                )
                velocity_delta_clamped = True

            speed_clamped = False
            safe_speed = vec_mag(safe_vel)
            if (
                math.isfinite(raw_fallback_max_speed)
                and raw_fallback_max_speed > 0.0
                and safe_speed > raw_fallback_max_speed
            ):
                scale = raw_fallback_max_speed / max(safe_speed, 1e-9)
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
            raw_ang_delta_mag = vec_mag(raw_ang_delta)
            angular_delta_clamped = False
            angular_delta_preserved = False
            if raw_fallback_angular_mode in {"preserve", "none", "linear", "linear_only", "off"}:
                safe_ang = before_ang
                angular_delta_preserved = True
            elif (
                raw_fallback_angular_mode == "auto"
                and raw_fallback_delta_mode in {"normal", "normal_only", "contact_normal"}
            ):
                safe_ang = before_ang
                angular_delta_preserved = True
            elif (
                math.isfinite(raw_fallback_max_angular_delta)
                and raw_fallback_max_angular_delta > 0.0
                and raw_ang_delta_mag > raw_fallback_max_angular_delta
            ):
                scale = raw_fallback_max_angular_delta / max(raw_ang_delta_mag, 1e-9)
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
                "raw_origin_fallback_velocity_safety_max_delta": raw_fallback_max_velocity_delta,
                "raw_origin_fallback_velocity_safety_max_speed": raw_fallback_max_speed,
                "raw_origin_fallback_angular_safety_max_delta": raw_fallback_max_angular_delta,
                "raw_origin_fallback_friction_override": raw_fallback_friction,
                "raw_origin_fallback_delta_mode": raw_fallback_delta_mode,
                "raw_origin_fallback_angular_mode": raw_fallback_angular_mode,
                "raw_origin_fallback_closing_only": raw_fallback_closing_only,
                "raw_origin_fallback_before_normal_speed": before_normal_speed,
                "raw_origin_fallback_before_normal_speed_source": before_normal_speed_source,
                "raw_origin_fallback_before_center_normal_speed": before_center_normal_speed,
                "raw_origin_fallback_normal_delta_skip_reason": normal_delta_skip_reason,
                "raw_origin_fallback_normal_delta_projected": normal_delta_projected,
                "raw_origin_fallback_velocity_after_unclamped": raw_after_vel,
                "raw_origin_fallback_velocity_delta_unclamped": raw_delta,
                "raw_origin_fallback_velocity_delta_mag_unclamped": raw_delta_mag,
                "raw_origin_fallback_velocity_delta_clamped": velocity_delta_clamped,
                "raw_origin_fallback_speed_clamped": speed_clamped,
                "raw_origin_fallback_angular_velocity_after_unclamped": raw_after_ang,
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
                "kind": "terrain_clean_contact",
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
            if timed_pair_response and resolve_timed_pair_contact():
                return True
            contact = sample_contact()
            if contact is None or contact.penetration <= self._PENETRATION_SLOP_DEFAULT:
                raw_contact, raw_bounds_contact, raw_error = sample_raw_origin_contact_at(
                    (anchor[0], anchor[1], anchor[2])
                )
                raw_fallback_contact = raw_origin_contact_for_fallback(raw_contact)
                raw_fallback_reject = raw_origin_fallback_reject_reason(raw_fallback_contact)
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
                    raycast_probe=raycast_probe,
                )
                if raw_error is None and raw_fallback_contact is not None and raw_fallback_reject == "":
                    response_debug, applied_raw_fallback = apply_raw_origin_fallback_contact(raw_fallback_contact)
                    if not applied_raw_fallback:
                        return False
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
                        "raw_origin_fallback_projection_order": raw_fallback_projection_order,
                        "raw_origin_fallback_normal_source": raw_fallback_normal_source,
                        "raw_origin_fallback_min_depth": raw_fallback_min_depth,
                        "raw_origin_fallback_max_depth": raw_fallback_max_depth,
                        "raw_origin_fallback_min_normal_z": raw_fallback_min_normal_z,
                        "raw_origin_fallback_min_speed": raw_fallback_min_speed,
                        "raw_origin_fallback_max_velocity_delta": raw_fallback_max_velocity_delta,
                        "raw_origin_fallback_max_speed": raw_fallback_max_speed,
                        "raw_origin_fallback_max_angular_delta": raw_fallback_max_angular_delta,
                        "raw_origin_fallback_angular_mode": raw_fallback_angular_mode,
                        "raw_origin_fallback_closing_only": raw_fallback_closing_only,
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
            response_debug = apply_contact(contact)
            ctx.debug_last_collision = {
                "kind": "terrain_clean_contact",
                "point": contact.position,
                "normal": contact.normal,
                "depth": contact.penetration,
                "terrain_collision_shape": terrain_collision_shape,
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
        if ctx.world_collision_bounds_dirty:
            dirty_contact_fn = None
            dirty_contact_args = None
            if collision_model is not None and terrain_collision_shape == "model":
                dirty_contact_fn = getattr(self._terrain_grid_collision, "test_model_bounds_contact", None)
                if callable(dirty_contact_fn):
                    dirty_contact_args = (
                        (anchor[0], anchor[1], anchor[2]),
                        (anchor[0], anchor[1], anchor[2] + z_lift),
                        heading,
                        vertices,
                        cbsp_tree,
                        bounding_radius,
                    )
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

            if dirty_contact_args is not None:
                contact = dirty_contact_fn(*dirty_contact_args)
                if contact is not None:
                    if self._is_pathological_dirty_bounds_contact((anchor[0], anchor[1], anchor[2]), contact, bounding_radius):
                        ctx.debug_last_collision = {
                            "kind": "terrain_dirty_bounds_filtered",
                            "point": contact.position,
                            "normal": contact.normal,
                            "depth": contact.penetration,
                            **contact_debug_fields(contact),
                            "detail": f"reference={reference_pos!r}",
                        }
                    else:
                        apply_dirty_bounds_contact(contact)
                        ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
                        return finish_result(anchor[0], anchor[1], anchor[2], vx, vy, vz)
            else:
                bounds_overlap_fn = getattr(self._terrain_grid_collision, "test_bounds_intersection", None)
                if callable(bounds_overlap_fn):
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
                    if bounds_overlap_fn(dirty_aabb_min, dirty_aabb_max):
                        if resolve_dirty_contact():
                            ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
                            return finish_result(anchor[0], anchor[1], anchor[2], vx, vy, vz)
                elif resolve_dirty_contact():
                    ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
                    return finish_result(anchor[0], anchor[1], anchor[2], vx, vy, vz)
            terrain_hit = raycast_fn(reference_pos, (anchor[0], anchor[1], anchor[2]))
            if terrain_hit is not None:
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
                }
                ctx.debug_last_motion_collision = dict(ctx.debug_last_collision)
                ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
                return finish_result(anchor[0], anchor[1], anchor[2], vx, vy, vz)
            ctx.world_collision_ref_pos = (anchor[0], anchor[1], anchor[2])
            # Dirty terrain dispatch can miss when the lifted model is clear and
            # the center-to-center ray remains above the height field. Preserve
            # the decompile-style dirty reference refresh, but still fall through
            # to the clean/raw-origin probe so reports can explain the miss and
            # default-off fallback A/Bs are not hidden by the dirty branch.

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
        cache_key = (ctx.entity_type, team_id, origin_mode)
        if cache_key in self._entity_collision_model_cache:
            return self._entity_collision_model_cache[cache_key]

        model_names = self._ENTITY_WORLD_MODEL_NAMES.get(ctx.entity_type)
        if not model_names or not self._building_collision.available:
            self._entity_collision_model_cache[cache_key] = None
            return None

        model_name = self._select_team_model_name(model_names, team_id)
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

    def _get_projectile_collision_radius(self, proj) -> float:
        radius = self.projectile_collision_radius
        model_names = self._PROJECTILE_MODEL_NAMES.get(proj.entity_type)
        if not model_names or not self._building_collision.available:
            return radius

        team_id = getattr(proj, "team", 1)
        model_name = self._select_team_model_name(model_names, team_id)
        model = self._building_collision.models.get(model_name)
        mesh = getattr(model, "collision_mesh", None) if model is not None else None
        vertices = getattr(mesh, "vertices", None) if mesh is not None else None
        if not vertices:
            return radius

        extents = sorted(
            (
                max(abs(v.x) for v in vertices),
                max(abs(v.y) for v in vertices),
                max(abs(v.z) for v in vertices),
            )
        )
        cross_section_radius = max(0.25, extents[1])
        return min(radius, cross_section_radius)

    def _building_has_mesh_collision(self, building) -> bool:
        if not self._building_collision.available:
            return False
        return self._building_collision.has_collision_model(
            int(building.entity_type),
            int(getattr(building, "team_id", 1)),
        )

    def _building_blocks_vehicle_collision(self, building) -> bool:
        """Return whether a map building should block vehicle movement.

        Repair pads are spawn/service pads. Treating their mesh/AABB as a
        solid blocker makes the authoritative tank shove sideways immediately
        after a map-flag spawn, while the OG client drives across the pad.
        """
        if int(getattr(building, "entity_type", -1)) == int(EntityType.REPAIR_BUILDING):
            return (
                os.environ.get("WULFRAM_REPAIR_PAD_BLOCKS_VEHICLES", "0")
                .strip()
                .lower()
                in ("1", "true", "on", "yes")
            )
        return True

    def _get_building_world_half_extents(self, building) -> tuple[float, float, float]:
        hx, hy = self._BUILDING_HALF_EXTENTS.get(building.entity_type, (8.0, 8.0))
        hz = max(hx, hy, self._BUILDING_HALF_HEIGHT)
        if not self._building_has_mesh_collision(building):
            return (hx, hy, hz)

        model_extents = self._building_collision.get_model_half_extents(
            int(building.entity_type),
            int(getattr(building, "team_id", 1)),
        )
        if model_extents is None:
            return (hx, hy, hz)

        local_hx, local_hy, local_hz = model_extents
        heading = float(getattr(building, "heading", 0.0))
        cos_h = abs(math.cos(heading))
        sin_h = abs(math.sin(heading))
        world_hx = local_hx * cos_h + local_hy * sin_h
        world_hy = local_hx * sin_h + local_hy * cos_h
        return (world_hx, world_hy, local_hz)

    def _get_building_quadtree_radius(self, building) -> float:
        if self._building_has_mesh_collision(building):
            radius = self._building_collision.get_model_bounding_radius(
                int(building.entity_type),
                int(getattr(building, "team_id", 1)),
            )
            if radius is not None:
                return radius
        hx, hy, hz = self._get_building_world_half_extents(building)
        return math.sqrt(hx * hx + hy * hy + hz * hz)

    def _rebuild_static_world_raycast_index(self) -> None:
        building_entities = getattr(self, "_building_entities", {}) or {}
        if not building_entities:
            self._static_world_raycast_root = None
            return

        bounds = []
        for eid, building in building_entities.items():
            radius = self._get_building_quadtree_radius(building)
            bounds.append((eid, building.x - radius, building.x + radius, building.y - radius, building.y + radius))

        min_x = min(item[1] for item in bounds)
        max_x = max(item[2] for item in bounds)
        min_y = min(item[3] for item in bounds)
        max_y = max(item[4] for item in bounds)
        if min_x == max_x:
            max_x = min_x + 1.0
        if min_y == max_y:
            max_y = min_y + 1.0

        building_ids = tuple(building_entities.keys())
        self._static_world_raycast_root = self._build_static_world_ray_node(
            building_ids,
            min_x,
            max_x,
            min_y,
            max_y,
            depth=0,
        )

    def _build_static_world_ray_node(
        self,
        building_ids: tuple[int, ...],
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        *,
        depth: int,
    ) -> _StaticWorldRayNode:
        max_leaf_size = 8
        max_depth = 4
        if len(building_ids) <= max_leaf_size or depth >= max_depth:
            return _StaticWorldRayNode(min_x, max_x, min_y, max_y, None, building_ids)

        mid_x = (min_x + max_x) * 0.5
        mid_y = (min_y + max_y) * 0.5
        child_bounds = (
            (mid_x, max_x, mid_y, max_y),
            (mid_x, max_x, min_y, mid_y),
            (min_x, mid_x, mid_y, max_y),
            (min_x, mid_x, min_y, mid_y),
        )
        buckets = [[], [], [], []]
        for eid in building_ids:
            building = self._building_entities.get(eid)
            if building is None:
                continue
            radius = self._get_building_quadtree_radius(building)
            west = (building.x - radius) < mid_x
            east = (building.x + radius) > mid_x
            north = (building.y - radius) < mid_y
            south = (building.y + radius) > mid_y
            if west and north:
                buckets[2].append(eid)
            if west and south:
                buckets[3].append(eid)
            if east and north:
                buckets[0].append(eid)
            if east and south:
                buckets[1].append(eid)

        non_empty = [bucket for bucket in buckets if bucket]
        if len(non_empty) <= 1:
            return _StaticWorldRayNode(min_x, max_x, min_y, max_y, None, building_ids)
        parent_ids = set(building_ids)
        if all(set(bucket) == parent_ids for bucket in non_empty):
            return _StaticWorldRayNode(min_x, max_x, min_y, max_y, None, building_ids)

        children = []
        for quadrant, bucket in enumerate(buckets):
            if not bucket:
                children.append(None)
                continue
            child_min_x, child_max_x, child_min_y, child_max_y = child_bounds[quadrant]
            children.append(
                self._build_static_world_ray_node(
                    tuple(bucket),
                    child_min_x,
                    child_max_x,
                    child_min_y,
                    child_max_y,
                    depth=depth + 1,
                )
            )
        return _StaticWorldRayNode(min_x, max_x, min_y, max_y, tuple(children), ())

    @staticmethod
    def _xy_outside_code(point: tuple[float, float, float], node: _StaticWorldRayNode) -> int:
        code = 0
        if point[0] < node.min_x:
            code |= 1
        if point[1] < node.min_y:
            code |= 2
        if point[0] > node.max_x:
            code |= 4
        if point[1] > node.max_y:
            code |= 8
        return code

    @staticmethod
    def _ray_misses_static_world_node(
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
        node: _StaticWorldRayNode,
    ) -> bool:
        endpoint_code = WulframServer._xy_outside_code(end_pos, node)
        if endpoint_code & WulframServer._xy_outside_code(start_pos, node):
            return True

        line_a = start_pos[0] - end_pos[0]
        line_b = end_pos[1] - start_pos[1]
        line_c = -(line_a * start_pos[1] + start_pos[0] * line_b)
        corners = (
            line_a * node.min_y + line_b * node.min_x + line_c,
            line_a * node.max_y + line_b * node.min_x + line_c,
            line_a * node.min_y + line_b * node.max_x + line_c,
            line_a * node.max_y + line_b * node.max_x + line_c,
        )
        return (
            all(value <= 0.0 for value in corners) or
            all(value >= 0.0 for value in corners)
        )

    @staticmethod
    def _static_world_origin_quadrant(point: tuple[float, float, float], node: _StaticWorldRayNode) -> int:
        quadrant = 0
        mid_x = (node.min_x + node.max_x) * 0.5
        mid_y = (node.min_y + node.max_y) * 0.5
        if point[1] < mid_y:
            quadrant |= 1
        if point[0] < mid_x:
            quadrant |= 2
        return quadrant

    @staticmethod
    def _iter_static_world_quadrants(origin_quadrant: int) -> tuple[int, int, int, int]:
        return (
            origin_quadrant,
            origin_quadrant ^ 0x1,
            origin_quadrant ^ 0x2,
            origin_quadrant ^ 0x3,
        )

    def _point_hits_static_building(
        self,
        building,
        point: tuple[float, float, float],
    ) -> Optional[tuple[str, tuple[float, float, float], float]]:
        half_extents = self._get_building_world_half_extents(building)
        aabb_min = (
            building.x - half_extents[0],
            building.y - half_extents[1],
            building.z - half_extents[2],
        )
        aabb_max = (
            building.x + half_extents[0],
            building.y + half_extents[1],
            building.z + half_extents[2],
        )
        if any(point[idx] < aabb_min[idx] or point[idx] > aabb_max[idx] for idx in range(3)):
            return None

        if self._building_has_mesh_collision(building):
            depth, _ = self._building_collision.test_sphere_collision(building, point, 1e-4)
            if depth <= 0.0:
                return None
            return ("building", point, 0.0)
        return ("building-aabb", point, 0.0)

    def _raycast_static_building_candidate(
        self,
        building,
        eid: int,
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
        *,
        seg_len: float,
    ) -> Optional[tuple[str, tuple[float, float, float], int, float]]:
        direction = (
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1],
            end_pos[2] - start_pos[2],
        )
        if self._building_has_mesh_collision(building):
            bounding_radius = self._building_collision.get_model_bounding_radius(
                building.entity_type,
                getattr(building, "team_id", 1),
            )
            if bounding_radius is None:
                half_extents = self._get_building_world_half_extents(building)
                bounding_radius = math.sqrt(
                    half_extents[0] * half_extents[0] +
                    half_extents[1] * half_extents[1] +
                    half_extents[2] * half_extents[2]
                )
            hit_t = self._segment_sphere_hit_t(
                start_pos,
                end_pos,
                (building.x, building.y, building.z),
                bounding_radius,
            )
            if hit_t is None:
                return None
            hit_position = (
                start_pos[0] + direction[0] * hit_t,
                start_pos[1] + direction[1] * hit_t,
                start_pos[2] + direction[2] * hit_t,
            )
            distance = seg_len * hit_t
            raycast_fn = getattr(self._building_collision, "raycast_segment_collision", None)
            if callable(raycast_fn):
                mesh_hit = raycast_fn(building, start_pos, end_pos)
                if mesh_hit is None:
                    return None
                hit_position, _, distance = mesh_hit
            elif not self._building_collision.test_segment_collision(building, start_pos, end_pos):
                return None
            return ("building", hit_position, eid, distance)

        half_extents = self._get_building_world_half_extents(building)
        aabb_min = (
            building.x - half_extents[0],
            building.y - half_extents[1],
            building.z - half_extents[2],
        )
        aabb_max = (
            building.x + half_extents[0],
            building.y + half_extents[1],
            building.z + half_extents[2],
        )
        hit_t = self._segment_aabb_hit_t(start_pos, end_pos, aabb_min, aabb_max)
        if hit_t is None:
            return None

        hit_position = (
            start_pos[0] + direction[0] * hit_t,
            start_pos[1] + direction[1] * hit_t,
            start_pos[2] + direction[2] * hit_t,
        )
        distance = seg_len * hit_t

        return ("building-aabb", hit_position, eid, distance)

    def _raycast_static_world_leaf(
        self,
        node: _StaticWorldRayNode,
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
    ) -> Optional[tuple[str, tuple[float, float, float], int, float]]:
        direction = (
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1],
            end_pos[2] - start_pos[2],
        )
        seg_len = math.sqrt(
            direction[0] * direction[0] +
            direction[1] * direction[1] +
            direction[2] * direction[2]
        )
        best_hit = None
        best_distance = None
        for eid in node.building_ids:
            building = self._building_entities.get(eid)
            if building is None:
                continue
            hit = self._raycast_static_building_candidate(building, eid, start_pos, end_pos, seg_len=seg_len)
            if hit is None:
                continue
            if best_distance is None or hit[3] < best_distance:
                best_distance = hit[3]
                best_hit = hit
        return best_hit

    def _point_query_static_world(
        self,
        start_pos: tuple[float, float, float],
        node: _StaticWorldRayNode,
    ) -> Optional[tuple[str, tuple[float, float, float], int, float]]:
        current = node
        while current.children is not None:
            quadrant = self._static_world_origin_quadrant(start_pos, current)
            child = current.children[quadrant]
            if child is None:
                break
            current = child

        def _dist_sq(eid: int) -> float:
            building = self._building_entities.get(eid)
            if building is None:
                return float("inf")
            hx, hy, _ = self._get_building_world_half_extents(building)
            dx = 0.0
            if start_pos[0] < building.x - hx:
                dx = (building.x - hx) - start_pos[0]
            elif start_pos[0] > building.x + hx:
                dx = start_pos[0] - (building.x + hx)
            dy = 0.0
            if start_pos[1] < building.y - hy:
                dy = (building.y - hy) - start_pos[1]
            elif start_pos[1] > building.y + hy:
                dy = start_pos[1] - (building.y + hy)
            return dx * dx + dy * dy

        for eid in sorted(current.building_ids, key=_dist_sq):
            building = self._building_entities.get(eid)
            if building is None:
                continue
            point_hit = self._point_hits_static_building(building, start_pos)
            if point_hit is not None:
                hit_kind, hit_position, distance = point_hit
                return (hit_kind, hit_position, eid, distance)
        return None

    @staticmethod
    def _segment_aabb_hit_t(
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
        aabb_min: tuple[float, float, float],
        aabb_max: tuple[float, float, float],
    ) -> Optional[float]:
        direction = (
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1],
            end_pos[2] - start_pos[2],
        )
        t_min = 0.0
        t_max = 1.0
        for axis in range(3):
            origin = start_pos[axis]
            delta = direction[axis]
            axis_min = aabb_min[axis]
            axis_max = aabb_max[axis]
            if abs(delta) <= 1e-8:
                if origin < axis_min or origin > axis_max:
                    return None
                continue
            inv_delta = 1.0 / delta
            t1 = (axis_min - origin) * inv_delta
            t2 = (axis_max - origin) * inv_delta
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2
            if t_min > t_max:
                return None
        return max(0.0, min(1.0, t_min))

    @staticmethod
    def _segment_sphere_hit_t(
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
        sphere_center: tuple[float, float, float],
        sphere_radius: float,
    ) -> Optional[float]:
        direction = (
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1],
            end_pos[2] - start_pos[2],
        )
        origin_to_center = (
            start_pos[0] - sphere_center[0],
            start_pos[1] - sphere_center[1],
            start_pos[2] - sphere_center[2],
        )
        a = (
            direction[0] * direction[0] +
            direction[1] * direction[1] +
            direction[2] * direction[2]
        )
        if a <= 1e-12:
            center_dist_sq = (
                origin_to_center[0] * origin_to_center[0] +
                origin_to_center[1] * origin_to_center[1] +
                origin_to_center[2] * origin_to_center[2]
            )
            return 0.0 if center_dist_sq <= sphere_radius * sphere_radius else None

        b = 2.0 * (
            direction[0] * origin_to_center[0] +
            direction[1] * origin_to_center[1] +
            direction[2] * origin_to_center[2]
        )
        c = (
            origin_to_center[0] * origin_to_center[0] +
            origin_to_center[1] * origin_to_center[1] +
            origin_to_center[2] * origin_to_center[2] -
            sphere_radius * sphere_radius
        )
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t0 = (-b - sqrt_disc) / (2.0 * a)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        if 0.0 <= t0 <= 1.0:
            return t0
        if 0.0 <= t1 <= 1.0:
            return t1
        if c <= 0.0:
            return 0.0
        return None

    def _raycast_static_buildings(
        self,
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
    ) -> Optional[tuple[str, tuple[float, float, float], int, float]]:
        root = getattr(self, "_static_world_raycast_root", None)
        if root is None and getattr(self, "_building_entities", None):
            self._rebuild_static_world_raycast_index()
            root = getattr(self, "_static_world_raycast_root", None)
        if root is None:
            return None

        if abs(end_pos[0] - start_pos[0]) <= 1e-8 and abs(end_pos[1] - start_pos[1]) <= 1e-8:
            return self._point_query_static_world(start_pos, root)

        def traverse(node: _StaticWorldRayNode):
            endpoint_code = self._xy_outside_code(end_pos, node)
            if endpoint_code & self._xy_outside_code(start_pos, node):
                return None
            if self._ray_misses_static_world_node(start_pos, end_pos, node):
                return None
            if node.children is None:
                leaf_hit = self._raycast_static_world_leaf(node, start_pos, end_pos)
                if leaf_hit is not None:
                    return leaf_hit
                if endpoint_code == 0:
                    return _STATIC_WORLD_RAY_STOP
                return None

            origin_quadrant = self._static_world_origin_quadrant(start_pos, node)
            for quadrant in self._iter_static_world_quadrants(origin_quadrant):
                child = node.children[quadrant]
                if child is None:
                    continue
                child_hit = traverse(child)
                if child_hit is _STATIC_WORLD_RAY_STOP:
                    return _STATIC_WORLD_RAY_STOP
                if child_hit is not None:
                    return child_hit
            return None

        hit = traverse(root)
        if hit is _STATIC_WORLD_RAY_STOP:
            return None
        return hit

    def _raycast_world(
        self,
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
    ) -> Optional[tuple[str, tuple[float, float, float], Optional[int]]]:
        terrain_hit = None
        terrain_dist_sq = None
        clipped_end = end_pos
        if self._terrain_grid_collision is not None:
            terrain_hit = self._terrain_grid_collision.raycast(start_pos, end_pos)
            if terrain_hit is not None:
                clipped_end = terrain_hit.position
                terrain_dist_sq = (
                    (terrain_hit.position[0] - start_pos[0]) * (terrain_hit.position[0] - start_pos[0]) +
                    (terrain_hit.position[1] - start_pos[1]) * (terrain_hit.position[1] - start_pos[1]) +
                    (terrain_hit.position[2] - start_pos[2]) * (terrain_hit.position[2] - start_pos[2])
                )

        building_hit = self._raycast_static_buildings(start_pos, clipped_end)
        if building_hit is not None:
            hit_kind, hit_position, hit_id, hit_distance = building_hit
            building_dist_sq = hit_distance * hit_distance
            if terrain_dist_sq is None or building_dist_sq <= terrain_dist_sq:
                return (hit_kind, hit_position, hit_id)

        if terrain_hit is not None:
            return ("terrain", terrain_hit.position, terrain_hit.sector_index)
        return None

    def _check_building_collisions(self, ctx, px, py, pz, vx, vy):
        """Check mesh/AABB collision against static buildings and other tanks."""
        for other_ctx in self._snapshot_in_game_clients():
            if other_ctx is ctx:
                continue
            # Check entity-to-entity blocking (tank vs tank)
            ox, oy, oz = other_ctx.player_pos
            dx, dy = px - ox, py - oy
            dist_sq = dx * dx + dy * dy
            min_dist = self._TANK_RADIUS * 2.0
            if dist_sq < min_dist * min_dist and dist_sq > 0.01:
                dist = math.sqrt(dist_sq)
                overlap = min_dist - dist
                # Decompile: penetration slop gate (Physics.c:5380)
                if overlap <= self._PENETRATION_SLOP_DEFAULT:
                    continue
                nx, ny = dx / dist, dy / dist
                px += nx * overlap * 0.5
                py += ny * overlap * 0.5
                vel_dot = vx * nx + vy * ny
                if vel_dot < 0:
                    vx -= nx * vel_dot
                    vy -= ny * vel_dot
                ctx.debug_last_collision = {
                    "kind": "vehicle_sphere",
                    "point": (px, py, pz),
                    "normal": (nx, ny, 0.0),
                    "depth": overlap,
                    "blocker_pos": (ox, oy, oz),
                    "detail": f"tank-vs-tank blocker client={other_ctx.client_id}",
                }

        # Check buildings loaded from map state file
        building_entities = self._building_entities
        for eid, building in building_entities.items():
            if not self._building_blocks_vehicle_collision(building):
                continue
            has_mesh_model = self._building_has_mesh_collision(building)
            mesh_hit = False
            if has_mesh_model:
                depth, normal = self._building_collision.test_sphere_collision(
                    building,
                    (px, py, pz),
                    self._TANK_RADIUS,
                )
                if depth > self._PENETRATION_SLOP_DEFAULT and normal:
                    separation = self._get_static_separation_from_contact(
                        (px, py, pz), (px + normal[0] * depth, py + normal[1] * depth, pz),
                    )
                    push = depth + separation
                    px += normal[0] * push
                    py += normal[1] * push
                    vel_dot = vx * normal[0] + vy * normal[1]
                    if vel_dot < 0:
                        vx -= normal[0] * vel_dot
                        vy -= normal[1] * vel_dot
                    mesh_hit = True
                    ctx.debug_last_collision = {
                        "kind": "building_mesh",
                        "point": (px, py, pz),
                        "normal": normal,
                        "depth": depth,
                        "blocker_pos": (building.x, building.y, building.z),
                        "entity_type": int(building.entity_type),
                        "team_id": int(getattr(building, "team_id", 1)),
                        "detail": f"eid={eid}",
                    }

            if mesh_hit:
                continue
            if has_mesh_model:
                continue

            bx, by = building.x, building.y
            etype = building.entity_type
            hx, hy = self._BUILDING_HALF_EXTENTS.get(etype, (8.0, 8.0))
            r = self._TANK_RADIUS
            if (px > bx - hx - r and px < bx + hx + r and
                    py > by - hy - r and py < by + hy + r):
                push_xp = (bx + hx + r) - px
                push_xn = px - (bx - hx - r)
                push_yp = (by + hy + r) - py
                push_yn = py - (by - hy - r)
                mp = min(push_xp, push_xn, push_yp, push_yn)
                # Decompile: penetration slop gate (Physics.c:5380)
                if mp <= self._PENETRATION_SLOP_DEFAULT:
                    continue
                if mp == push_xp:
                    px = bx + hx + r
                    if vx < 0: vx = 0.0
                    normal = (1.0, 0.0, 0.0)
                    depth = push_xp
                elif mp == push_xn:
                    px = bx - hx - r
                    if vx > 0: vx = 0.0
                    normal = (-1.0, 0.0, 0.0)
                    depth = push_xn
                elif mp == push_yp:
                    py = by + hy + r
                    if vy < 0: vy = 0.0
                    normal = (0.0, 1.0, 0.0)
                    depth = push_yp
                elif mp == push_yn:
                    py = by - hy - r
                    if vy > 0: vy = 0.0
                    normal = (0.0, -1.0, 0.0)
                    depth = push_yn
                else:
                    normal = (0.0, 0.0, 0.0)
                    depth = 0.0
                ctx.debug_last_collision = {
                    "kind": "building_aabb",
                    "point": (px, py, pz),
                    "normal": normal,
                    "depth": depth,
                    "blocker_pos": (building.x, building.y, building.z),
                    "entity_type": int(building.entity_type),
                    "team_id": int(getattr(building, "team_id", 1)),
                    "detail": f"eid={eid}",
                }

        return px, py, vx, vy

    def _broadcast_projectile_delete(
        self,
        proj,
        tick: int,
        *,
        with_effects: bool,
        reason: str,
    ) -> None:
        """Broadcast projectile deletion and record it in the packet log."""
        delete_pkt = build_delete_object(tick, [proj.entity_id], with_effects=with_effects)
        for client in self._snapshot_in_game_clients():
            if not self._projectile_packets_allowed_for_client(client):
                continue
            self._send_packet_to_client(client, delete_pkt, prefer_tcp=True)
            if self.pktlog.enabled:
                self.pktlog.log(
                    client_id=client.client_id,
                    label="PROJ_DELETE",
                    tick=tick,
                    payload=delete_pkt,
                    transport="TCP",
                    extra=f"proj_id=0x{proj.entity_id:X} reason={reason}",
                )

    def _check_projectile_world_hit(self, start_client_pos: tuple, end_client_pos: tuple, proj=None):
        """Raycast a projectile against terrain first, then static world blockers."""
        start_pos = self._from_client_pos(start_client_pos)
        end_pos = self._from_client_pos(end_client_pos)
        return self._raycast_world(start_pos, end_pos)

    def _send_debug_sync(self, ctx: ClientContext, frame_counter: int,
                         turn_input: float, fwd_input: float, strafe_input: float):
        """Send DEBUG_SYNC packet (0x60) with server's authoritative state.

        49-byte UDP packet sent after each physics step for measuring
        client-server divergence. Opcode 0x60 is unused by the original protocol.
        """
        if not self._debug_sync_allowed_for_client(ctx):
            return
        # DEBUG_SYNC should use the same client-facing position space as
        # UPDATE_ARRAY/VIEW_UPDATE, otherwise the sync harness reports the
        # configured terrain/world offset as fake divergence.
        pos = self._to_client_pos(ctx.player_pos)
        heading = ctx.player_heading
        vel = ctx.player_vel
        ang_vel = ctx.vehicle_physics.angular_velocity if ctx.vehicle_physics else 0.0
        steps = ctx.physics_step_count

        payload = struct.pack(
            ">BII3f1f3f1f3f",
            0x60,
            frame_counter,
            steps,
            pos[0], pos[1], pos[2],
            heading,
            vel[0], vel[1], vel[2],
            ang_vel,
            turn_input, fwd_input, strafe_input,
        )
        if self.udp_handler and ctx.session.udp_addr:
            self.udp_handler.send_to(payload, ctx.session.udp_addr)

    @staticmethod
    def _normalize_debug_host(host: str) -> str:
        host = host.strip().lower()
        if host.startswith("::ffff:"):
            return host[7:]
        return host

    def _debug_sync_allowed_for_client(self, ctx: ClientContext) -> bool:
        if not self.debug_sync:
            return False
        if self.debug_sync_allow_all:
            return True

        addr = ctx.session.udp_addr or ctx.client_addr
        if not addr:
            return False

        host = self._normalize_debug_host(str(addr[0]))
        if host in self.debug_sync_hosts:
            return True

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None

        if ip and ip.is_loopback and (
            "127.0.0.1" in self.debug_sync_hosts
            or "::1" in self.debug_sync_hosts
            or "loopback" in self.debug_sync_hosts
        ):
            return True

        if ctx.client_id not in self._debug_sync_blocked_clients:
            self._debug_sync_blocked_clients.add(ctx.client_id)
            print(
                f"[DEBUG_SYNC] Suppressing custom 0x60 for client {ctx.client_id} "
                f"host={host} allowlist={sorted(self.debug_sync_hosts)}"
            )
        return False

    def _state_sync_reply_allowed_for_client(self, ctx: ClientContext) -> bool:
        if self.state_sync_reply_allow_all:
            return True

        addr = ctx.session.udp_addr or ctx.client_addr
        if not addr:
            return False

        host = self._normalize_debug_host(str(addr[0]))
        if host in self.state_sync_reply_hosts:
            return True

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None

        if ip and ip.is_loopback and (
            "127.0.0.1" in self.state_sync_reply_hosts
            or "::1" in self.state_sync_reply_hosts
            or "loopback" in self.state_sync_reply_hosts
        ):
            return True

        if ctx.client_id not in self._state_sync_blocked_clients:
            self._state_sync_blocked_clients.add(ctx.client_id)
            print(
                f"[STATE-SYNC] Suppressing STATE_REQUEST reply for client {ctx.client_id} "
                f"host={host} allowlist={sorted(self.state_sync_reply_hosts)}"
            )
        return False

    def _should_send_state_sync_view_update(self, ctx: ClientContext) -> bool:
        mode = getattr(self, "state_sync_view_mode", "all")
        is_loopback = handlers._is_loopback_client(ctx)
        if mode == "off":
            return False
        if mode == "loopback":
            return is_loopback
        if mode == "remote":
            return not is_loopback
        return True

    def _remote_movement_input_active(self, ctx: ClientContext, *, now: Optional[float] = None) -> bool:
        """Return true while a remote OG client is actively driving."""
        if handlers._is_loopback_client(ctx):
            return False
        if bool(getattr(ctx, "_datagram_active_movement_input", False)):
            return True
        window = float(
            getattr(self, "active_input_correction_suppress_window", 0.35) or 0.0
        )
        if window <= 0.0:
            return False
        last_action = float(getattr(ctx, "last_action_packet_time", 0.0) or 0.0)
        if last_action <= 0.0:
            return False
        if now is None:
            now = time.monotonic()
        if (now - last_action) > window:
            return False
        decoded = getattr(ctx, "last_decoded_input", None) or {}
        try:
            fwd = float(decoded.get("fwd", 0.0) or 0.0)
            strafe = float(decoded.get("strafe", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        return abs(fwd) > 0.05 or abs(strafe) > 0.05

    def _udp_ping_reply_allowed_for_client(self, ctx: ClientContext) -> bool:
        if self.udp_ping_reply_allow_all:
            return True

        addr = None
        if ctx.session and ctx.session.udp_addr:
            addr = ctx.session.udp_addr
        elif ctx.udp_addr:
            addr = ctx.udp_addr

        if not addr:
            return False

        host = self._normalize_debug_host(str(addr[0]))
        if host in self.udp_ping_reply_hosts:
            return True

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None

        if ip and ip.is_loopback and (
            "127.0.0.1" in self.udp_ping_reply_hosts
            or "::1" in self.udp_ping_reply_hosts
            or "loopback" in self.udp_ping_reply_hosts
        ):
            return True

        if ctx.client_id not in self._udp_ping_reply_blocked_clients:
            self._udp_ping_reply_blocked_clients.add(ctx.client_id)
            print(
                f"[UDP-PING] Suppressing 0x0C ping reply for client {ctx.client_id} "
                f"host={host} allowlist={sorted(self.udp_ping_reply_hosts)}"
            )
        return False

    def _debug_comm_allowed_for_client(self, ctx: ClientContext) -> bool:
        if self.debug_comm_allow_all:
            return True

        addr = ctx.client_addr
        if ctx.session and ctx.session.udp_addr:
            addr = ctx.session.udp_addr
        if not addr:
            return False

        host = self._normalize_debug_host(str(addr[0]))
        if host in self.debug_comm_hosts:
            return True

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None

        if ip and ip.is_loopback and (
            "127.0.0.1" in self.debug_comm_hosts
            or "::1" in self.debug_comm_hosts
            or "loopback" in self.debug_comm_hosts
        ):
            return True

        if ctx.client_id not in self._debug_comm_blocked_clients:
            self._debug_comm_blocked_clients.add(ctx.client_id)
            print(
                f"[DEBUG-COMM] Suppressing non-OG chat for client {ctx.client_id} "
                f"host={host} allowlist={sorted(self.debug_comm_hosts)}"
            )
        return False

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

        # Broadcast TRANSIENT_ARRAY (weapon fire FX) to all other clients
        # DISABLED: 0x0D format crashes OG client (D_ERR at PROTOCOL.CPP:474)
        # self._broadcast_weapon_fire_fx(ctx, proj)

        # Start background thread for movement updates
        def update_loop():
            # === Projectile update mode ===
            # Mode 0: No updates (let client simulate from spawn velocity)
            # Mode 1: Low rate updates (5 Hz)
            # Mode 2: Medium rate updates (15 Hz)
            # Mode 3: High rate updates (30 Hz)
            update_mode = self.projectile_update_mode
            delete_reason = "expired"
            delete_with_effects = False

            if update_mode == 0:
                # No updates - just wait for lifetime then clean up
                time.sleep(proj.lifetime)
                print(f"[PROJ] id={proj.entity_id} expired (no-update mode)")
                with ctx.projectile_lock:
                    if proj in ctx.active_projectiles:
                        ctx.active_projectiles.remove(proj)
                tick = self._get_network_tick(ctx)
                self._broadcast_projectile_delete(
                    proj,
                    tick,
                    with_effects=False,
                    reason="expired",
                )
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

                # Update projectile position for hit detection
                # (build_projectile_update_packet also updates proj.pos,
                #  but we need the position current before hit check)
                prev_client_pos = proj.pos
                proj.pos = (
                    proj.pos[0] + proj.vel[0] * dt,
                    proj.pos[1] + proj.vel[1] * dt,
                    proj.pos[2] + proj.vel[2] * dt,
                )

                world_hit = self._check_projectile_world_hit(prev_client_pos, proj.pos, proj)
                if world_hit:
                    hit_kind, hit_pos, hit_detail = world_hit
                    delete_reason = hit_kind
                    delete_with_effects = False
                    with ctx.projectile_lock:
                        if proj in ctx.active_projectiles:
                            ctx.active_projectiles.remove(proj)

                    # Broadcast impact FX at hit location
                    if hit_kind == "terrain":
                        fx_type = FX_IMPACT_TERRAIN
                        print(
                            f"[PROJ-WORLD] id={proj.entity_id} hit terrain "
                            f"at=({hit_pos[0]:.1f},{hit_pos[1]:.1f},{hit_pos[2]:.1f})"
                        )
                    else:
                        fx_type = FX_IMPACT_BUILDING
                        print(
                            f"[PROJ-WORLD] id={proj.entity_id} hit {hit_kind} "
                            f"target={hit_detail} at=({hit_pos[0]:.1f},{hit_pos[1]:.1f},{hit_pos[2]:.1f})"
                        )
                        # Apply damage to building
                        self._apply_building_damage(hit_detail, proj, ctx, hit_pos)

                    # Send impact FX via TRANSIENT_ARRAY only to viewers that
                    # can safely accept the current 0x0D path.
                    self._broadcast_transient_fx([{
                        'type': fx_type,
                        'pos': self._to_client_pos(hit_pos),
                    }])
                    break

                # Check collision with enemy players
                hit_target = self._check_projectile_hit(proj, ctx)
                if hit_target:
                    try:
                        self._apply_damage(hit_target, proj, ctx)
                    except Exception as dmg_err:
                        print(f"[COMBAT-ERROR] _apply_damage failed: {dmg_err}")
                        import traceback
                        traceback.print_exc()
                    delete_reason = None
                    with ctx.projectile_lock:
                        if proj in ctx.active_projectiles:
                            ctx.active_projectiles.remove(proj)
                    break  # Stop update loop

                # Send position update (dt=0 since we already advanced pos above)
                # Build per-client packets so each viewer gets their OWN health
                # in local_state (not the shooter's health).
                sent_update_count = 0
                if self.udp_handler:
                    for target in self._snapshot_in_game_clients():
                        if not target.session.udp_addr or not target.session.translation_ack_received:
                            continue
                        if not self._projectile_packets_allowed_for_client(target):
                            continue
                        tick = self._get_network_tick(target)
                        include_local_state, local_state_kwargs = self._get_projectile_local_state_for_viewer(target)
                        pkt = build_projectile_update_packet(
                            proj,
                            tick,
                            0.0,  # Position already advanced above
                            include_local_state=include_local_state,
                            **local_state_kwargs,
                            )
                        self.udp_handler.send_to(pkt, target.session.udp_addr)
                        sent_update_count += 1
                        if self.pktlog.enabled:
                            self.pktlog.log(
                                client_id=target.client_id,
                                label="PROJ_UPDATE",
                                tick=tick,
                                payload=pkt,
                                transport="UDP",
                                entity_count=1,
                                entity_ids=(proj.entity_id,),
                                mask_bits=(0b0010,),  # pos only
                                has_local_state=include_local_state,
                                health=self._get_health_value(target) if include_local_state else -1.0,
                            )
                if sent_update_count:
                    ctx.projectile_update_packet_count = (
                        int(getattr(ctx, "projectile_update_packet_count", 0) or 0)
                        + sent_update_count
                    )
                    ctx.last_projectile_update_time = time.monotonic()
                    ctx.last_projectile_update_id = int(getattr(proj, "entity_id", 0) or 0)
                    ctx.last_projectile_update_targets = sent_update_count

                if i % 15 == 0:  # Log every 0.5 sec at 30Hz
                    print(f"[PROJ] id={proj.entity_id} pos=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f}) vel=({proj.vel[0]:.0f},{proj.vel[1]:.0f},{proj.vel[2]:.0f}) tick={tick}")

            # Remove from active list when done
            with ctx.projectile_lock:
                if proj in ctx.active_projectiles:
                    ctx.active_projectiles.remove(proj)
            # Send DELETE_OBJECT so the client removes the shell entity
            if delete_reason is not None:
                tick = self._get_network_tick(ctx)
                self._broadcast_projectile_delete(
                    proj,
                    tick,
                    with_effects=delete_with_effects,
                    reason=delete_reason,
                )
                print(f"[PROJ] id={proj.entity_id} {delete_reason} - DELETE sent")

        # Add to active list and start thread
        with ctx.projectile_lock:
            ctx.active_projectiles.append(proj)

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

    def _check_projectile_hit(self, proj, owner_ctx: ClientContext) -> Optional[ClientContext]:
        """Check if a projectile hit any enemy player (sphere collision).

        Compares projectile position against all in-game players except the owner.
        Both are in client/world coordinates (proj.pos is already converted).
        Returns the hit ClientContext or None.
        """
        hit_radius = 15.0  # Tank is roughly this size
        hit_radius_sq = hit_radius * hit_radius
        for target in self._snapshot_in_game_clients():
            if target is owner_ctx:
                continue
            # Compare in client space (proj.pos already converted)
            target_pos = self._to_client_pos(target.player_pos)
            dx = proj.pos[0] - target_pos[0]
            dy = proj.pos[1] - target_pos[1]
            dz = proj.pos[2] - target_pos[2]
            dist_sq = dx * dx + dy * dy + dz * dz
            dist = math.sqrt(dist_sq)
            if dist < 100:  # Log when close
                print(
                    f"[PROJ-DIST] id={proj.entity_id} -> c{target.client_id} "
                    f"dist={dist:.1f} (hit<{hit_radius:.0f}) "
                    f"proj=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f}) "
                    f"tgt=({target_pos[0]:.1f},{target_pos[1]:.1f},{target_pos[2]:.1f})"
                )
            if dist_sq <= hit_radius_sq:
                print(f"[PROJ-HIT] id={proj.entity_id} HIT c{target.client_id} dist={dist:.1f}")
                return target
        return None

    def _apply_damage(self, target: ClientContext, proj, attacker: ClientContext) -> None:
        """Apply damage from a projectile hit and broadcast effects.

        1. Reduce target health
        2. Send health update to target
        3. Send DELETE_OBJECT for projectile (with explosion FX)
        4. If dead, send DELETE_OBJECT for target entity
        """
        # Guard: ignore hits on already-dead targets (overkill from queued projectiles)
        if target.player_health <= 0.0:
            print(f"[COMBAT] Ignoring hit on already-dead c{target.client_id}")
            # Still delete the projectile
            tick = self._get_network_tick(attacker)
            delete_proj_pkt = build_delete_object(tick, [proj.entity_id], with_effects=True)
            for client in self._snapshot_in_game_clients():
                if not self._projectile_packets_allowed_for_client(client):
                    continue
                self._send_packet_to_client(client, delete_proj_pkt, prefer_tcp=True)
            return

        # Per-weapon damage (fraction of 100 health). Decompile: entity health
        # table at VA 0x4E3B00 has Tank=100hp. These are approximate pending
        # OG server-side weapon damage values.
        _PROJECTILE_DAMAGE = {
            EntityType.PULSE_SHELL: 0.20,    # 20% — 5 hits to kill
            EntityType.PIERCER: 0.30,        # 30% — fast, high damage
            EntityType.THUMPER: 0.35,        # 35% — slow, heavy damage
            EntityType.HUNTER: 0.25,         # 25% — homing missile
            EntityType.HEAVY_MISSILE: 0.50,  # 50% — heavy ordnance
            EntityType.MINE: 0.40,           # 40% — proximity mine
            EntityType.SHORT_MISSILE: 0.15,  # 15% — short range missile
            EntityType.FLAK_SHELL: 0.10,     # 10% — flak
        }
        damage = _PROJECTILE_DAMAGE.get(proj.entity_type, 0.20)
        old_health = target.player_health
        target.player_health = round(max(0.0, old_health - damage), 6)
        new_health = target.player_health
        target.last_damage_time = time.monotonic()
        target.last_damage_source = f"projectile:{getattr(proj.entity_type, 'name', proj.entity_type)}"
        target.last_damage_amount = damage
        target.last_damage_old_health = old_health
        target.last_damage_new_health = new_health

        attacker_name = attacker.session.username or f"Player{attacker.client_id}"
        target_name = target.session.username or f"Player{target.client_id}"
        print(
            f"[COMBAT] {attacker_name} (c{attacker.client_id}) hit {target_name} (c{target.client_id}) "
            f"for {damage*100:.0f}% damage (health: {old_health*100:.0f}% -> {new_health*100:.0f}%)"
        )

        if target.player_health > 0.0 and self.udp_handler and target.session.udp_addr:
            tick = self._get_network_tick(target)
            health_pkt = self._build_local_state_heartbeat(
                target,
                tick=tick,
                entity_id=target.session.entity_id,
                include_health=True,
                health=self._get_health_value(target),
                fuel=self._get_energy_value(target),
            )
            self.udp_handler.send_to(health_pkt, target.session.udp_addr)
            if self.pktlog.enabled:
                self.pktlog.log(
                    client_id=target.client_id,
                    label="DAMAGE_HEARTBEAT",
                    tick=tick,
                    payload=health_pkt,
                    transport="UDP",
                    entity_count=1,
                    entity_ids=(0xFFFFFFFE,),
                    mask_bits=(0,),
                    has_local_state=True,
                    health=self._get_health_value(target),
                    extra=f"dmg={damage}",
                )

        # Broadcast impact FX via TRANSIENT_ARRAY (decompile-backed quantized bitstream)
        impact_events = [{
            'type': FX_IMPACT_VEHICLE,
            'pos': proj.pos,
            'entity_id': target.entity_id,
        }]
        self._broadcast_transient_fx(impact_events)

        # DELETE projectile with explosion effects
        tick = self._get_network_tick(attacker)
        delete_proj_pkt = build_delete_object(tick, [proj.entity_id], with_effects=True)
        for client in self._snapshot_in_game_clients():
            if not self._projectile_packets_allowed_for_client(client):
                continue
            self._send_packet_to_client(client, delete_proj_pkt, prefer_tcp=True)

        # Chat notification
        hit_msg = f"HIT! {target_name} ({new_health*100:.0f}% health)"
        chat_pkt = build_chat_message(hit_msg, source_id=attacker.session.player_id or attacker.entity_id)
        for client in self._snapshot_in_game_clients():
            if client.tcp_handler and self._debug_comm_allowed_for_client(client):
                client.tcp_handler.send(chat_pkt)

        # Send health refresh to ALL surviving clients (attacker etc.)
        # Projectile UPDATE_ARRAY packets include per-viewer health, but once
        # projectiles are gone the viewer gets no more health data. This
        # heartbeat ensures the attacker's HUD doesn't revert to zero.
        for client in self._snapshot_in_game_clients():
            if client is target:
                continue  # Already sent above
            if (
                client is not attacker
                and not handlers._is_loopback_client(client)
            ):
                continue
            if self.udp_handler and client.session.udp_addr:
                c_tick = self._get_network_tick(client)
                c_pkt = self._build_local_state_heartbeat(
                    client,
                    tick=c_tick,
                    entity_id=client.session.entity_id,
                    include_health=True,
                    health=self._get_health_value(client),
                    fuel=self._get_energy_value(client),
                )
                self.udp_handler.send_to(c_pkt, client.session.udp_addr)

        # If target is dead, delete their entity with explosion and schedule respawn
        if target.player_health <= 0.0:
            # Track kill/death stats
            attacker.kills += 1
            target.deaths += 1
            print(f"[COMBAT] {target_name} (c{target.client_id}) DESTROYED by {attacker_name} (c{attacker.client_id})"
                  f" [K:{attacker.kills} D:{target.deaths}]")
            kill_msg = f"KILL! {attacker_name} destroyed {target_name}!"
            kill_chat = build_chat_message(kill_msg, source_id=attacker.session.player_id or attacker.entity_id)
            for client in self._snapshot_in_game_clients():
                if client.tcp_handler and self._debug_comm_allowed_for_client(client):
                    client.tcp_handler.send(kill_chat)

            # Broadcast updated stats for attacker and target
            combat_participants = (attacker, target)
            self._broadcast_player_stats(attacker, participants=combat_participants)
            self._broadcast_player_stats(target, participants=combat_participants)

            target_entity_id = target.session.entity_id or target.entity_id

            # DELETE entity with explosion effects
            tick_del = self._get_network_tick(target)
            del_pkt = build_delete_object(tick_del, [target_entity_id], with_effects=True)
            for client in self._snapshot_in_game_clients():
                if not self._combat_observer_packets_allowed_for_client(client, attacker, target):
                    continue
                self._send_packet_to_client(client, del_pkt, prefer_tcp=True)

            # Stop tick loop (entity no longer exists on client)
            target.session.in_game = False

            # Remove from other clients' known entities so they re-create on respawn
            for other in self._snapshot_in_game_clients():
                if other is not target:
                    other.known_entity_ids.discard(target_entity_id)

            # Clear dead player's own known entities and retry tracking.
            # On respawn, _sync_clients_on_spawn will re-create all entities
            # for this player.  Without this, the stale known_entity_ids +
            # expired _entity_create_times retry window would cause
            # _send_entity_create to skip re-creation after respawn.
            target.known_entity_ids.clear()
            if hasattr(target, '_entity_create_times'):
                target._entity_create_times.clear()

            # Reset server-side state for next spawn
            target.player_health = 1.0
            target.player_vel = (0.0, 0.0, 0.0)
            target.player_speed = 0.0
            target.angular_vel_yaw = 0.0
            target.world_collision_ref_pos = target.player_pos
            target.world_collision_bounds_dirty = False
            if target.vehicle_physics:
                target.vehicle_physics.reset()

            # Use game loop's delayed spawn mechanism (instead of background thread).
            # The game loop checks delayed_spawn_team every 0.5s and calls
            # _auto_join_team -> _spawn_wf_style when the time arrives.
            respawn_delay = 5.0
            target.session.delayed_spawn_team = target.session.team_id or 1
            target.session.delayed_spawn_time = time.monotonic() + respawn_delay
            print(f"[COMBAT] Respawning c{target.client_id} in {respawn_delay:.0f}s via delayed_spawn")

    def _apply_building_damage(self, building_oid, proj, attacker: ClientContext, hit_pos: tuple):
        """Apply damage from a projectile to a building.

        Decompile: buildings have per-type health (entity health table VA 0x4E3B00).
        Damage values are absolute HP, not fractional like player damage.
        """
        if building_oid not in self._building_health:
            return
        if self._building_health[building_oid] <= 0:
            return

        # Damage in absolute HP (buildings have 800-5000 HP)
        _PROJECTILE_BUILDING_DAMAGE = {
            EntityType.PULSE_SHELL: 50.0,
            EntityType.PIERCER: 80.0,
            EntityType.THUMPER: 120.0,
            EntityType.HUNTER: 70.0,
            EntityType.HEAVY_MISSILE: 200.0,
            EntityType.MINE: 150.0,
            EntityType.SHORT_MISSILE: 30.0,
            EntityType.FLAK_SHELL: 20.0,
        }
        damage = _PROJECTILE_BUILDING_DAMAGE.get(proj.entity_type, 50.0)
        old_hp = self._building_health[building_oid]
        self._building_health[building_oid] = max(0.0, old_hp - damage)
        new_hp = self._building_health[building_oid]

        building = self._building_entities.get(building_oid)
        btype_name = building.entity_type.name if building else "UNKNOWN"
        max_hp = self._building_max_health.get(building_oid, 1.0)
        pct = (new_hp / max_hp * 100) if max_hp > 0 else 0

        attacker_name = attacker.session.username or f"Player{attacker.client_id}"
        print(f"[BUILDING] {attacker_name} hit {btype_name} oid={building_oid} "
              f"for {damage:.0f} dmg ({old_hp:.0f} -> {new_hp:.0f} / {max_hp:.0f}, {pct:.0f}%)")

        # Building destroyed
        if new_hp <= 0:
            print(f"[BUILDING] {btype_name} oid={building_oid} DESTROYED by {attacker_name}")
            # Track building kill
            attacker.kills += 1
            self._broadcast_player_stats(attacker, participants=(attacker,))

            # DELETE_OBJECT for building with explosion effects
            tick = self._get_network_tick(attacker)
            del_pkt = build_delete_object(tick, [building_oid], with_effects=True)
            for client in self._snapshot_in_game_clients():
                if not self._combat_observer_packets_allowed_for_client(client, attacker):
                    continue
                self._send_packet_to_client(client, del_pkt, prefer_tcp=True)
            # Chat notification
            from .packets import build_chat_message
            msg = f"DESTROYED! {attacker_name} leveled a {btype_name}!"
            chat_pkt = build_chat_message(msg, source_id=attacker.session.player_id or attacker.entity_id)
            for client in self._snapshot_in_game_clients():
                if (
                    client.tcp_handler
                    and self._combat_observer_packets_allowed_for_client(client, attacker)
                    and self._debug_comm_allowed_for_client(client)
                ):
                    client.tcp_handler.send(chat_pkt)

    def _update_supply_buildings(self, ctx: ClientContext, dt: float):
        """Apply supply building effects to nearby friendly vehicles.

        Decompile: support buildings (repair/fuel/energy) affect friendly vehicles
        within a service radius. Each tick, nearby friendly players receive:
        - REPAIR_BUILDING: health regeneration
        - FUEL_BUILDING: fuel/energy regeneration (maps to energy in our model)
        - ENERGY_BUILDING: energy regeneration (faster than fuel building)

        Service radius ~40 units from building center. Only active if building
        is alive and on the same team as the player.
        """
        if not self._building_entities:
            return

        px, py, pz = ctx.player_pos
        team = ctx.session.team_id

        # Supply building service radius squared (40 units)
        SERVICE_RADIUS_SQ = 40.0 * 40.0

        # Per-tick rates (at 30Hz): health regen = 0.5%/tick = 15%/s
        # Energy regen = 3.0/tick = 90/s (fills 100 energy in ~1.1s)
        REPAIR_RATE = 0.005       # health per tick (normalized 0-1)
        FUEL_RATE = 2.0           # energy per tick (absolute)
        ENERGY_RATE = 3.0         # energy per tick (absolute, faster)

        near_repair = False
        near_fuel = False
        near_energy = False

        for oid, b in self._building_entities.items():
            # Skip destroyed buildings
            if self._building_health.get(oid, 0) <= 0:
                continue
            # Skip enemy buildings
            if b.team_id != team:
                continue

            dx = px - b.x
            dy = py - b.y
            dist_sq = dx * dx + dy * dy

            if dist_sq > SERVICE_RADIUS_SQ:
                continue

            if b.entity_type == EntityType.REPAIR_BUILDING:
                near_repair = True
            elif b.entity_type == EntityType.FUEL_BUILDING:
                near_fuel = True
            elif b.entity_type == EntityType.ENERGY_BUILDING:
                near_energy = True

        if near_repair and ctx.player_health < 1.0:
            ctx.player_health = min(1.0, ctx.player_health + REPAIR_RATE)

        max_energy = self.player_energy_max if self.player_energy_max > 0.0 else 100.0
        if near_fuel and ctx.player_energy < max_energy:
            ctx.player_energy = min(max_energy, ctx.player_energy + FUEL_RATE)
        if near_energy and ctx.player_energy < max_energy:
            ctx.player_energy = min(max_energy, ctx.player_energy + ENERGY_RATE)

    def _update_turret_ai(self):
        """Turret AI: GUN_TURRET and LAUNCHER buildings fire at nearby enemies.

        Hitscan damage with fire FX broadcast. Turrets target the closest
        enemy vehicle within range and deal direct damage on a cooldown.

        GUN_TURRET: range 120u, fire every 2.0s, 8% damage per shot
        LAUNCHER: range 200u, fire every 3.0s, 15% damage per shot
        """
        if not self._building_entities:
            return

        now = time.monotonic()

        _TURRET_CONFIG = {
            EntityType.GUN_TURRET: {
                'range_sq': 120.0 * 120.0,
                'fire_interval': 2.0,
                'damage': 0.08,  # normalized 0-1 (8% per shot)
                'fx_type': FX_PULSE_FIRE,
            },
            EntityType.LAUNCHER: {
                'range_sq': 200.0 * 200.0,
                'fire_interval': 3.0,
                'damage': 0.15,  # normalized 0-1 (15% per shot)
                'fx_type': FX_MISSILE_FIRE,
            },
        }

        in_game = self._snapshot_in_game_clients()
        if not in_game:
            return

        for oid, b in self._building_entities.items():
            config = _TURRET_CONFIG.get(b.entity_type)
            if config is None:
                continue
            # Skip destroyed turrets
            if self._building_health.get(oid, 0) <= 0:
                continue
            # Check fire cooldown
            last_fire = self._turret_last_fire.get(oid, 0.0)
            if (now - last_fire) < config['fire_interval']:
                continue

            # Find closest enemy player
            best_target = None
            best_dist_sq = config['range_sq']
            for client in in_game:
                if client.session.team_id == b.team_id:
                    continue  # same team, skip
                if client.player_health <= 0:
                    continue
                dx = client.player_pos[0] - b.x
                dy = client.player_pos[1] - b.y
                dist_sq = dx * dx + dy * dy
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_target = client

            if best_target is None:
                continue

            # Fire! Apply hitscan damage + FX
            self._turret_last_fire[oid] = now
            spawn_pos = (b.x, b.y, b.z + 3.0)

            # Broadcast fire FX at turret position
            from .packets import build_transient_array
            fx_pkt = build_transient_array([{
                'type': config['fx_type'],
                'pos': self._to_client_pos(spawn_pos),
            }])
            if fx_pkt:
                for client in in_game:
                    if not self._transient_fx_allowed_for_client(client):
                        continue
                    if self.udp_handler and client.session.udp_addr:
                        self.udp_handler.send_to(fx_pkt, client.session.udp_addr)

            # Impact FX at target position
            impact_pkt = build_transient_array([{
                'type': FX_IMPACT_VEHICLE,
                'pos': self._to_client_pos(best_target.player_pos),
            }])
            if impact_pkt:
                for client in in_game:
                    if not self._transient_fx_allowed_for_client(client):
                        continue
                    if self.udp_handler and client.session.udp_addr:
                        self.udp_handler.send_to(impact_pkt, client.session.udp_addr)

            # Apply damage to target
            damage = config['damage']
            old_health = best_target.player_health
            best_target.player_health = max(0.0, old_health - damage)
            target_name = best_target.session.username or f"Player{best_target.client_id}"
            btype_name = getattr(b.entity_type, 'name', str(b.entity_type))
            best_target.last_damage_time = now
            best_target.last_damage_source = f"turret:{btype_name}:oid={oid}"
            best_target.last_damage_amount = damage
            best_target.last_damage_old_health = old_health
            best_target.last_damage_new_health = best_target.player_health
            print(
                f"[TURRET] {btype_name} oid={oid} hit {target_name} "
                f"for {damage*100:.0f}% "
                f"({old_health*100:.0f}% -> {best_target.player_health*100:.0f}%)"
            )

            if best_target.player_health <= 0.0 and old_health > 0.0:
                # Turret killed the player
                best_target.deaths += 1
                self._broadcast_player_stats(best_target, participants=(best_target,))
                print(f"[TURRET] {btype_name} oid={oid} KILLED {target_name}")
                kill_msg = f"{target_name} was destroyed by a {btype_name}!"
                from .packets import build_chat_message
                kill_chat = build_chat_message(kill_msg, source_id=0)
                for client in in_game:
                    if (
                        client.tcp_handler
                        and self._combat_observer_packets_allowed_for_client(client, best_target)
                        and self._debug_comm_allowed_for_client(client)
                    ):
                        try:
                            client.tcp_handler.send(kill_chat)
                        except Exception:
                            pass  # client may have disconnected

                # Death sequence: DELETE with effects + respawn
                target_eid = best_target.session.entity_id or best_target.entity_id
                tick_del = self._get_network_tick(best_target)
                del_pkt = build_delete_object(tick_del, [target_eid], with_effects=True)
                for client in in_game:
                    if not self._combat_observer_packets_allowed_for_client(client, best_target):
                        continue
                    self._send_packet_to_client(client, del_pkt, prefer_tcp=True)
                best_target.session.in_game = False
                for other in in_game:
                    if other is not best_target:
                        other.known_entity_ids.discard(target_eid)
                best_target.known_entity_ids.clear()
                if hasattr(best_target, '_entity_create_times'):
                    best_target._entity_create_times.clear()
                best_target.player_health = 1.0
                best_target.player_vel = (0.0, 0.0, 0.0)
                best_target.player_speed = 0.0
                best_target.angular_vel_yaw = 0.0
                best_target.world_collision_ref_pos = best_target.player_pos
                best_target.world_collision_bounds_dirty = False
                if best_target.vehicle_physics:
                    best_target.vehicle_physics.reset()
                best_target.session.delayed_spawn_team = best_target.session.team_id or 1
                best_target.session.delayed_spawn_time = time.monotonic() + 5.0

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
        thrust = tank_softbody_control_slot_value(slots)
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
        Process jump jet input.
        Called after decoding ACTION_DUMP or ACTION_UPDATE.
        """
        # Jump jets are now applied in _update_player_position() so the server
        # and Python prediction use the same fixed-step rising-edge model.
        # Keep this packet-arrival hook as a no-op compatibility shim.
        return

    def _on_jump_jet_triggered(self, ctx: ClientContext, player_id: int, impulse: float, new_vel_z: float):
        """Callback when a jump jet is triggered."""
        print(f"[JUMP] Jump triggered for player {player_id}: impulse={impulse}, vel_z={new_vel_z:.1f}")
        burst_count = int(getattr(self, "jump_jet_correction_burst_count", 0) or 0)
        if burst_count > 0 and not handlers._is_loopback_client(ctx):
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

        # UPDATE_ARRAY (0x0E) over TCP crashes OG client (TCP bitstream
        # desync â†’ protocol mismatch). UDP-only for velocity updates.

    def _build_velocity_update_packet(self, ctx: ClientContext) -> bytes:
        """Build UPDATE_ARRAY packet with current velocity."""
        from .packets import _compress_value, _write_local_player_state
        from .codec import BitWriter

        tick = self._get_network_tick(ctx)
        tick_bytes = struct.pack(">I", tick)

        bw = BitWriter()

        # CRITICAL: Always include local_state with health=1.0.
        # The client calls sync_local_player after processing EVERY
        # UPDATE_ARRAY.  It reads health from a static buffer filled
        # by read_local_player_state.  If we send flag=0 (no stats),
        # the buffer retains stale/zero data, and sync_local_player
        # zeroes entity health â†’ triggers permanent DeathScreen.
        # MUST include ammo/turret bits matching BEHAVIOR config,
        # otherwise OG client reads past local_state â†’ bitstream
        # misalignment â†’ protocol mismatch crash (especially on TCP).
        ls = self._get_local_state_kwargs(ctx)
        if self.update_local_state_mode == "wf" and not handlers._is_loopback_client(ctx):
            ls["weapon_id"] = self._get_spawn_tank_weapon_type(ctx)
            ls["ammo_count_bits"] = 0
            ls["ammo_count"] = 0
            ls["primary_turret_bits"] = 0
            ls["primary_turret_angle"] = 0.0
            ls["secondary_turret_bits"] = 0
            ls["secondary_turret_angle"] = 0.0
        ls["health"] = 1.0  # Force full health to prevent DeathScreen
        _write_local_player_state(
            bw,
            include=True,
            **ls,
        )

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

        # Velocity vector using TRANSLATION VEC_VEL config (4-bit header + 16-bit values)
        bw.write_bits(4, 15)  # Header (max precision)
        for v in ctx.player_vel:
            bw.write_bits(16, _compress_value(v, VEC_VEL_MAX, VEC_VEL_RANGE, total_bits=16))

        return b'\x0E' + tick_bytes + bw.get_bytes()

    def _apply_reload_defaults(self):
        """Re-read env-var config after hot reload, setting only NEW attributes.

        Called by control.py reload after class swap.  Uses setdefault-style
        logic so existing live state (connections, counters) is never clobbered.
        """
        def _default(attr, val):
            if not hasattr(self, attr):
                setattr(self, attr, val)

        # Re-read .env file so new vars are visible
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ[key.strip()] = val.strip()

        print("[RELOAD] _apply_reload_defaults done")

    def _ensure_tick_loop(self, ctx: ClientContext) -> bool:
        """Start exactly one authoritative tick loop for a client context."""
        if not FEATURES.tick_loop_enabled:
            return False
        with ctx.tick_lock:
            if ctx.tick_thread is not None and ctx.tick_thread.is_alive():
                return False
            thread = threading.Thread(target=self._tick_loop, args=(ctx,), daemon=True)
            ctx.tick_thread = thread
            thread.start()
            return True

    def _tick_loop(self, ctx: ClientContext):
        """Game tick loop - sends UPDATE_ARRAY periodically."""
        current_thread = threading.current_thread()
        with ctx.tick_lock:
            if ctx.tick_thread is not current_thread:
                print(f"[TICK] Stale tick loop rejected for client {ctx.client_id}")
                return
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

        # Wall-clock tick pacing: Windows time.sleep(0.033) often sleeps ~15-21ms,
        # causing the tick loop to run at ~47Hz instead of 30Hz.  Use a monotonic
        # accumulator to guarantee exactly tick_rate_hz ticks per wall-clock second.
        next_tick_time = time.monotonic()
        tick_period = 1.0 / self.tick_rate_hz if self.tick_rate_hz > 0 else 0.1
        # Physics steps once per tick at native 30Hz (no accumulator needed).
        ctx.physics_step_count = 0
        last_physics_wall_time = time.monotonic()
        # (frame_locked mode removed â€” not part of original decompile)

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

                if self.require_client_tick and ctx.last_client_tick <= 0:
                    if not logged_wait_client_tick and time.monotonic() - tick_start_time < 5.0:
                        print(f"[TICK] Client {ctx.client_id}: Waiting for client input tick before sending UPDATE_ARRAY")
                        logged_wait_client_tick = True
                    time.sleep(0.01)
                    continue

                # Optional grace period to delay UPDATE_ARRAY sends after client is ready.
                if self.update_grace_seconds > 0.0:
                    if grace_period_end is None:
                        grace_period_end = time.monotonic() + self.update_grace_seconds
                        print(
                            f"[TICK] Client {ctx.client_id}: Starting {self.update_grace_seconds:.1f}s "
                            "grace period before sending position updates"
                        )
                    if time.monotonic() < grace_period_end:
                        if not grace_period_logged:
                            remaining = grace_period_end - time.monotonic()
                            print(f"[TICK] Client {ctx.client_id}: Grace period ({remaining:.1f}s remaining)")
                            grace_period_logged = True
                        time.sleep(0.05)
                        continue

                # Once translation is ready, make sure this client sees others and vice versa.
                self._ensure_multiplayer_visibility(ctx)

                # Read current input state (needed every tick for transition detection)
                raw_input = self._get_raw_turn_input(ctx)
                prev_input = getattr(ctx, 'prev_raw_turn_input', 0.0)
                torque = self._compute_turn_torque(ctx, raw_input)  # lateral_mobility=1.0

                physics = ctx.vehicle_physics
                ws = ctx.weapon_system
                use_f32 = self.f32_physics

                # Log input transitions (key press/release)
                input_changed = abs(raw_input - prev_input) > 0.001
                now_mono = time.monotonic()
                if input_changed:
                    transition = "PRESS" if abs(raw_input) > abs(prev_input) else "RELEASE"
                    last_transition_time = getattr(ctx, '_yaw_transition_time', now_mono)
                    last_transition_tick = getattr(ctx, '_yaw_transition_tick', ctx.session.tick)
                    elapsed_ms = (now_mono - last_transition_time) * 1000
                    elapsed_ticks = ctx.session.tick - last_transition_tick
                    effective_hz = elapsed_ticks / max(0.001, now_mono - last_transition_time)
                    ctx._yaw_transition_time = now_mono
                    ctx._yaw_transition_tick = ctx.session.tick
                    if self.debug_sync:
                        yaw_msg = (
                            f"[YAW-INPUT] {transition} c{ctx.client_id} "
                            f"input={prev_input:.3f}->{raw_input:.3f} "
                            f"ang_vel={physics.angular_velocity:.4f} "
                            f"heading={math.degrees(physics.heading):.2f}deg "
                            f"t={ctx.session.tick} "
                            f"wall={elapsed_ms:.0f}ms ticks={elapsed_ticks} hz={effective_hz:.1f}"
                        )
                        print(yaw_msg)
                        try:
                            with open(r"C:\Users\wstri\dev\wolfram\yaw_events.log", "a") as _yf:
                                _yf.write(yaw_msg + "\n")
                        except Exception:
                            pass

                ctx.prev_raw_turn_input = raw_input

                # === PHYSICS STEPPING ===
                # Physics steps once per server tick at native 30Hz (dt=1/tick_rate).
                # Matches client's 30Hz accumulator stepping.
                physics_dt = 1.0 / self.tick_rate_hz

                ctx.physics_step_count += 1
                old_heading = ctx.player_heading

                # Live ACTION_UPDATE packets arrive asynchronously relative to the
                # 30 Hz tick loop. If turning changed partway through this wall-clock
                # tick window, split the simulated 30 Hz step so the pre-change
                # slice uses the previous turn input and only the remainder uses
                # the latest input.
                step_wall_now = time.monotonic()
                step_wall_dt = max(1e-6, step_wall_now - last_physics_wall_time)
                transition_time = float(getattr(ws, "turn_input_change_time", 0.0) or 0.0)
                prev_turn_slot = float(getattr(ws, "turn_input_prev_value", 0.0) or 0.0)
                prev_turn_input = self._normalize_turn_input_value(ctx, prev_turn_slot)
                split_turn_step = (
                    transition_time > last_physics_wall_time and
                    transition_time < step_wall_now and
                    abs(prev_turn_input - raw_input) > 0.001
                )
                move_dt = physics_dt
                move_heading = old_heading
                if split_turn_step:
                    pre_ratio = (transition_time - last_physics_wall_time) / step_wall_dt
                    pre_ratio = max(0.0, min(1.0, pre_ratio))
                    pre_dt = physics_dt * pre_ratio
                    post_dt = physics_dt - pre_dt
                    prev_torque = self._compute_turn_torque(ctx, prev_turn_input)
                    if pre_dt > 1e-6:
                        physics.step_client_substeps(prev_torque, pre_dt, use_f32=use_f32)
                        self._sync_heading_physics_to_context(ctx, physics)
                        self._update_player_position(ctx, dt_override=pre_dt, heading_override=old_heading)
                        move_heading = ctx.player_heading
                    if post_dt > 1e-6:
                        physics.step_client_substeps(torque, post_dt, use_f32=use_f32)
                    move_dt = post_dt
                    ws.turn_input_change_time = 0.0
                    if self.debug_sync:
                        print(
                            f"[YAW-SPLIT] c{ctx.client_id} "
                            f"old={prev_turn_input:.3f} new={raw_input:.3f} "
                            f"pre_dt={pre_dt * 1000.0:.1f}ms post_dt={post_dt * 1000.0:.1f}ms"
                        )
                else:
                    physics.step_client_substeps(torque, physics_dt, use_f32=use_f32)
                last_physics_wall_time = step_wall_now

                self._sync_heading_physics_to_context(ctx, physics)

                if move_dt > 1e-6:
                    self._update_player_position(ctx, dt_override=move_dt, heading_override=move_heading)
                self._resolve_entity_entity_collisions(ctx)
                self._update_player_aim(ctx)
                self._regen_player_energy(ctx, physics_dt)
                self._update_supply_buildings(ctx, physics_dt)
                if os.environ.get("WULFRAM_TURRET_AI", "1") == "1":
                    self._update_turret_ai()

                # Send debug sync state for measuring client-server divergence
                if self.debug_sync:
                    fwd_input = self._normalize_behavior_axis_value(
                        ctx,
                        ws.behavior_slots[BehaviorSlot.MOVING_FORWARD],
                    )
                    strafe_input = self._decode_network_strafe_input(
                        ctx,
                        ws.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS],
                    )
                    self._send_debug_sync(ctx, ws.client_frame_counter,
                                          raw_input, fwd_input, strafe_input)

                # YAW-TRACK: disabled to avoid per-tick console I/O slowing tick loop
                # if abs(ctx.angular_vel_yaw) > 0.01:
                #     print(f"[YAW-TRACK] ...")

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
                    ctx.last_position_update = now
                    ctx.position_change_count += 1
                    desync_warned = False

                # Check for non-zero input in behavior slots (only movement, not thrust)
                fwd_input = self._normalize_behavior_axis_value(
                    ctx,
                    ctx.weapon_system.behavior_slots[BehaviorSlot.MOVING_FORWARD],
                )
                strafe_input = self._decode_network_strafe_input(
                    ctx,
                    ctx.weapon_system.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS],
                )
                thrust_input = tank_softbody_control_slot_value(ctx.weapon_system.behavior_slots)
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
                self._record_authoritative_state(ctx, tick=tick)
                # Debug: log tick value periodically
                if ctx.session.tick % 300 == 0:
                    print(f"[TICK-DEBUG] Client {ctx.client_id}: network_tick={tick} client_tick={ctx.last_client_tick} offset={ctx.tick_offset}")
                health_val = self._get_health_value(ctx)
                fuel_val = self._get_energy_value(ctx)
                # Full position updates (server authoritative).
                # Default OFF: wulf-forge does NOT send the local player's position
                # in UPDATE_ARRAY. Sending position overrides the client's own physics
                # and causes underground clipping + red health overlay on hilly terrain
                # (server ground_level is flat, terrain is not).
                send_full_update = os.environ.get("WULFRAM_SEND_FULL_UPDATES", "0") == "1"
                # Default to current position for tracking even if we skip UPDATE_ARRAY.
                send_pos = self._to_client_pos(ctx.player_pos)
                payload: Optional[bytes] = None
                send_payload = False

                self._maybe_promote_remote_full_local_state(ctx, reason="post_spawn")

                send_update = True
                burst_remaining = int(getattr(ctx, "correction_burst_remaining", 0) or 0)
                burst_interval = float(getattr(ctx, "correction_burst_interval_s", 0.0) or 0.0)
                active_movement_correction_suppressed = self._remote_movement_input_active(
                    ctx,
                    now=now,
                )
                burst_due = (
                    burst_remaining > 0
                    and not active_movement_correction_suppressed
                    and (now - ctx.last_correction_send) >= burst_interval
                )
                movement_interval = float(getattr(self, "movement_correction_interval", 0.0) or 0.0)
                movement_window = float(getattr(self, "movement_correction_window", 0.0) or 0.0)
                recent_move_input_time = float(getattr(ctx, "last_nonzero_move_input_time", 0.0) or 0.0)
                movement_correction_recent = (
                    movement_interval > 0
                    and recent_move_input_time > 0.0
                    and (now - recent_move_input_time) <= movement_window
                    and not handlers._is_loopback_client(ctx)
                )
                movement_correction_due = (
                    movement_correction_recent
                    and not active_movement_correction_suppressed
                    and (now - ctx.last_correction_send) >= movement_interval
                )
                interval_correction_due = (
                    self.correction_interval > 0
                    and not active_movement_correction_suppressed
                    and (now - ctx.last_correction_send) >= self.correction_interval
                )
                correction_reason = ""
                if getattr(ctx, "force_correction_once", False):
                    correction_reason = "forced"
                elif burst_due:
                    correction_reason = "burst"
                elif movement_correction_due:
                    correction_reason = "movement"
                elif interval_correction_due:
                    correction_reason = "interval"
                correction_due = (
                    getattr(ctx, "force_correction_once", False)
                    or burst_due
                    or movement_correction_due
                    or interval_correction_due
                )
                if self.update_on_change:
                    pos_changed = any(abs(a - b) > self.update_epsilon for a, b in zip(send_pos, ctx.last_sent_pos))
                    vel_changed = any(abs(a - b) > self.update_epsilon for a, b in zip(ctx.player_vel, ctx.last_sent_vel))
                    yaw_changed = abs(ctx.player_yaw - ctx.last_sent_yaw) > self.update_epsilon
                    heartbeat_due = self.update_heartbeat_interval > 0 and (now - ctx.last_update_send) >= self.update_heartbeat_interval
                    if not (pos_changed or vel_changed or yaw_changed or heartbeat_due):
                        send_update = False
                else:
                    # Throttle heartbeat to configured interval instead of every tick
                    if self.update_heartbeat_interval > 0:
                        if (now - ctx.last_update_send) < self.update_heartbeat_interval:
                            send_update = False

                if self.send_player_updates and send_full_update and send_update:
                    # Send UPDATE_ARRAY with position/velocity
                    send_pos = self._to_client_pos(ctx.player_pos)
                    local_state_kwargs = self._get_local_state_kwargs(ctx)
                    weapon_type = local_state_kwargs["weapon_id"]
                    ammo_bits = local_state_kwargs["ammo_count_bits"]
                    ammo_mask = local_state_kwargs["ammo_count"]
                    pt_bits = local_state_kwargs["primary_turret_bits"]
                    pt_angle = local_state_kwargs["primary_turret_angle"]
                    st_bits = local_state_kwargs["secondary_turret_bits"]
                    st_angle = local_state_kwargs["secondary_turret_angle"]
                    include_local_state = self._should_send_local_state(
                        ctx,
                        pt_bits,
                        st_bits,
                        self.update_local_state_mode,
                    )
                    if not getattr(ctx, "_ammo_turret_logged", False) and include_local_state:
                        print(
                            f"[LOCAL-STATE] weapon_type={weapon_type} "
                            f"ammo_bits={ammo_bits} ammo_mask=0x{ammo_mask:X} "
                            f"pt_bits={pt_bits} st_bits={st_bits}"
                        )
                        ctx._ammo_turret_logged = True
                    include_lpos = True
                    include_lvel = self.local_update_mode in ("pos_vel", "pos_vel_rot")
                    include_lrot = self.local_update_mode in ("pos_rot", "pos_vel_rot")
                    if not getattr(ctx, "_update_mode_logged", False):
                        print(
                            "[UPDATE-MODE] "
                            f"mode={self.local_update_mode!r} "
                            f"include_pos={int(include_lpos)} "
                            f"include_vel={int(include_lvel)} "
                            f"include_rot={int(include_lrot)}"
                        )
                        ctx._update_mode_logged = True
                    if self.combine_update_arrays and self.send_player_updates:
                        entities = [
                            {
                                "entity_id": ctx.session.entity_id,
                                "is_manned": True,
                                "pos": send_pos,
                                "vel": ctx.player_vel,
                                "rot": (
                                    ctx.player_pose.get("roll", 0.0),
                                    ctx.player_pose.get("pitch", 0.0),
                                    ctx.player_heading,  # entity+0x38 convention
                                ),
                                "include_pos": include_lpos,
                                "include_vel": include_lvel,
                                "include_rot": include_lrot,
                                "include_entity_vitals": self.update_entity_vitals,
                                "speed_scale": 1.0,
                                "fuel": fuel_val,
                            }
                        ]
                        mode = self.remote_update_mode
                        include_rpos = mode not in ("heartbeat", "mask0", "off", "none", "disabled")
                        include_rvel = mode in ("pos_vel", "pos_vel_rot", "full", "all")
                        include_rrot = mode in ("pos_rot", "pos_vel_rot", "full", "all")
                        if mode not in ("off", "none", "disabled") and self._og_viewer_replication_enabled(ctx, "remote_updates"):
                            for other in self._snapshot_in_game_clients():
                                if other is ctx:
                                    continue
                                entity_id = other.session.entity_id or other.entity_id
                                if entity_id not in ctx.known_entity_ids:
                                    continue
                                entities.append(
                                    {
                                        "entity_id": entity_id,
                                        "is_manned": self.remote_update_is_manned,
                                        "pos": self._to_client_pos(other.player_pos),
                                        "vel": other.player_vel,
                                        "rot": (
                                            other.player_pose.get("roll", 0.0),
                                            other.player_pose.get("pitch", 0.0),
                                            (-other.player_heading if self.remote_yaw_negate else other.player_heading) + self.remote_yaw_offset,
                                        ),
                                        "include_pos": include_rpos,
                                        "include_vel": include_rvel,
                                        "include_rot": include_rrot,
                                        "include_spin": include_rrot,
                                        "spin": (0.0, 0.0, other.angular_vel_yaw),
                                        "include_entity_vitals": False,
                                    }
                                )
                        if self.update_packet_type == "view":
                            payload = build_view_update_multi(
                                tick,
                                include_local_state=include_local_state,
                                weapon_id=weapon_type,
                                health=health_val,
                                fuel=fuel_val,
                                ammo_count_bits=ammo_bits,
                                ammo_count=ammo_mask,
                                primary_turret_bits=pt_bits,
                                primary_turret_angle=pt_angle,
                                secondary_turret_bits=st_bits,
                                secondary_turret_angle=st_angle,
                                turret_max=self.local_state_turret_max,
                                turret_range=self.local_state_turret_range,
                                entities=entities,
                            )
                        else:
                            payload = build_update_array_multi(
                                tick,
                                include_local_state=include_local_state,
                                weapon_id=weapon_type,
                                health=health_val,
                                fuel=fuel_val,
                                ammo_count_bits=ammo_bits,
                                ammo_count=ammo_mask,
                                primary_turret_bits=pt_bits,
                                primary_turret_angle=pt_angle,
                                secondary_turret_bits=st_bits,
                                secondary_turret_angle=st_angle,
                                turret_max=self.local_state_turret_max,
                                turret_range=self.local_state_turret_range,
                                entities=entities,
                            )
                    else:
                        if self.update_packet_type == "view":
                            payload = build_view_update_player_update(
                                tick,
                                ctx.session.entity_id,
                                pos=send_pos,
                                vel=ctx.player_vel,
                                # Rotation vector order follows wulf-forge: (roll, pitch, yaw)
                                rot=self._local_player_sync_rotation(ctx),
                                include_pos=include_lpos,
                                include_vel=include_lvel,
                                include_rot=include_lrot,
                                include_local_state=include_local_state,
                                include_entity_vitals=self.update_entity_vitals,
                                weapon_id=weapon_type,
                                health=health_val,
                                fuel=fuel_val,
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
                            payload = build_update_array_player_update(
                                tick,
                                ctx.session.entity_id,
                                pos=send_pos,
                                vel=ctx.player_vel,
                                # Rotation vector order follows wulf-forge: (roll, pitch, yaw)
                                rot=self._local_player_sync_rotation(ctx),
                                include_pos=include_lpos,
                                include_vel=include_lvel,
                                include_rot=include_lrot,
                                include_local_state=include_local_state,
                                include_entity_vitals=self.update_entity_vitals,
                                weapon_id=weapon_type,
                                health=health_val,
                                fuel=fuel_val,
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
                    if include_local_state:
                        self._log_vitals(
                            ctx,
                            "UPDATE_ARRAY_FULL",
                            include_vitals=True,
                            health=health_val,
                            energy=fuel_val,
                            weapon_id=weapon_type,
                            note=f"ammo_bits={ammo_bits} pt_bits={pt_bits} st_bits={st_bits}",
                        )
                    elif self.debug_vitals and ctx.session.tick % 300 == 0:
                        self._log_vitals(
                            ctx,
                            "UPDATE_ARRAY_FULL",
                            include_vitals=False,
                            health=health_val,
                            energy=fuel_val,
                            weapon_id=weapon_type,
                            note="local_state=0",
                        )
                    send_payload = True
                    if not correction_due:
                        self._maybe_send_view_update_loop(
                            ctx,
                            tick=tick,
                            send_pos=send_pos,
                            health_val=health_val,
                            fuel_val=fuel_val,
                            weapon_type=weapon_type,
                            ammo_bits=ammo_bits,
                            ammo_mask=ammo_mask,
                            pt_bits=pt_bits,
                            pt_angle=pt_angle,
                            st_bits=st_bits,
                            st_angle=st_angle,
                        )
                elif self.send_player_updates and not send_full_update and (send_update or correction_due):
                    # Heartbeat path. Remote OG clients that have left the
                    # spawn-safe path need a real single-local-player update
                    # shape here; the synthetic mask-0 stub causes the
                    # original client to protocol-mismatch on spawn.

                    if self._suppress_remote_spawn_safe_heartbeat(ctx):
                        if not getattr(ctx, "_spawn_safe_heartbeat_suppressed_logged", False):
                            print(
                                f"[HEARTBEAT] Client {ctx.client_id}: suppressing periodic "
                                "spawn-safe remote heartbeat until targeted sync is active"
                            )
                            ctx._spawn_safe_heartbeat_suppressed_logged = True
                        send_payload = False
                        continue

                    local_state_kwargs = self._get_local_state_kwargs(ctx)
                    weapon_type = local_state_kwargs["weapon_id"]
                    ammo_bits = local_state_kwargs["ammo_count_bits"]
                    ammo_mask = local_state_kwargs["ammo_count"]
                    pt_bits = local_state_kwargs["primary_turret_bits"]
                    pt_angle = local_state_kwargs["primary_turret_angle"]
                    st_bits = local_state_kwargs["secondary_turret_bits"]
                    st_angle = local_state_kwargs["secondary_turret_angle"]
                    include_local_state = self._should_send_local_state(
                        ctx,
                        pt_bits,
                        st_bits,
                        self.update_local_state_mode,
                    )

                    # Heartbeat: dummy-entity packet (health only, no position override).
                    # NOTE: Local player corrections were removed â€” the client runs
                    # lockstep deterministic physics and overwrites any server position/
                    # rotation corrections every frame. See server __init__ for details.
                    if correction_due:
                        payload, pkt_label, corr_pos, corr_rot, inc_pos, inc_rot = (
                            self._build_empirical_correction_payload(
                                ctx,
                                tick=tick,
                                include_local_state=include_local_state,
                                health=health_val,
                                fuel=fuel_val,
                                weapon_type=weapon_type,
                                ammo_bits=ammo_bits,
                                ammo_mask=ammo_mask,
                                pt_bits=pt_bits,
                                pt_angle=pt_angle,
                                st_bits=st_bits,
                                st_angle=st_angle,
                            )
                        )
                        ctx.last_correction_send = now
                        ctx.force_correction_once = False
                        if burst_remaining > 0:
                            ctx.correction_burst_remaining = burst_remaining - 1
                        if correction_reason == "movement":
                            ctx.movement_correction_count = int(
                                getattr(ctx, "movement_correction_count", 0) or 0
                            ) + 1
                            ctx.last_movement_correction_send_time = now
                        log_due = correction_reason != "movement" or (
                            now - float(getattr(ctx, "last_movement_correction_log", 0.0) or 0.0)
                        ) >= 1.0
                        if log_due:
                            if correction_reason == "movement":
                                ctx.last_movement_correction_log = now
                            print(
                                f"[CORRECTION] reason={correction_reason or 'unknown'} "
                                f"mode={self.correction_mode} client={ctx.client_id} "
                                f"pos=({corr_pos[0]:.1f},{corr_pos[1]:.1f},{corr_pos[2]:.1f}) "
                                f"yaw={math.degrees(corr_rot[2]):.1f}deg "
                                f"inc_pos={int(inc_pos)} inc_rot={int(inc_rot)} "
                                f"burst_left={int(getattr(ctx, 'correction_burst_remaining', 0))}"
                            )
                    else:
                        use_view = self.heartbeat_view_update
                        pkt_label = "VIEW_UPDATE_BEAT" if use_view else "UPDATE_ARRAY_BEAT"
                        hb_rot = None
                        if self.heartbeat_include_rot:
                            hb_rot = self._local_player_sync_rotation(ctx)
                        hb_pos = None
                        if self.heartbeat_include_pos:
                            hb_pos = self._to_client_pos(ctx.player_pos)
                        payload = self._build_local_state_heartbeat(
                            ctx,
                            tick=tick,
                            entity_id=ctx.session.entity_id,
                            include_health=include_local_state,
                            health=health_val,
                            fuel=fuel_val,
                            is_view_update=use_view,
                            rot=hb_rot,
                            pos=hb_pos,
                        )

                    if include_local_state:
                        self._log_vitals(
                            ctx,
                        pkt_label,
                        include_vitals=True,
                        health=health_val,
                        energy=fuel_val,
                        weapon_id=weapon_type,
                        note="heartbeat",
                    )
                    elif self.debug_vitals and ctx.session.tick % 300 == 0:
                        self._log_vitals(
                            ctx,
                        pkt_label,
                        include_vitals=False,
                        health=health_val,
                        energy=fuel_val,
                        weapon_id=weapon_type,
                        note="heartbeat local_state=0",
                    )

                    send_payload = True
                    self._maybe_send_view_update_loop(
                        ctx,
                        tick=tick,
                        send_pos=send_pos,
                        health_val=health_val,
                        fuel_val=fuel_val,
                        weapon_type=weapon_type,
                        ammo_bits=ammo_bits,
                        ammo_mask=ammo_mask,
                        pt_bits=pt_bits,
                        pt_angle=pt_angle,
                        st_bits=st_bits,
                        st_angle=st_angle,
                    )

                if self.send_player_updates and send_payload and payload is not None:
                    # Determine transport for logging
                    _transports = []
                    # Try TCP first, fall back to UDP if TCP fails
                    # Client may close TCP after spawn and use UDP only
                    if self.send_updates_tcp and not tcp_failed and ctx.tcp_handler:
                        try:
                            ctx.tcp_handler.send(payload, log=False)
                            _transports.append("TCP")
                        except Exception as tcp_err:
                            print(f"[TICK] Client {ctx.client_id}: TCP failed ({tcp_err}), switching to UDP-only")
                            tcp_failed = True

                    # Always send via UDP as well for reliability
                    if self.send_updates_udp and self.udp_handler and ctx.session.udp_addr:
                        self.udp_handler.send_to(payload, ctx.session.udp_addr)
                        _transports.append("UDP")

                    # Log packet for traffic analysis
                    if self.pktlog.enabled:
                        _log_ents = (0xFFFFFFFE,)
                        _log_masks = (0,)
                        self.pktlog.log(
                            client_id=ctx.client_id,
                            label=pkt_label,
                            tick=tick,
                            payload=payload,
                            transport="+".join(_transports),
                            entity_count=len(_log_ents),
                            entity_ids=_log_ents,
                            mask_bits=_log_masks,
                            has_local_state=include_local_state,
                            health=health_val if include_local_state else -1.0,
                        )

                    ctx.last_update_send = now
                    ctx.last_sent_pos = send_pos
                    ctx.last_sent_vel = ctx.player_vel
                    ctx.last_sent_yaw = ctx.player_yaw

                # Solo-local-player keepalive — feeds the OG client's organic
                # STATE_REQUEST trigger (Replication.c:1173-1177 requires
                # entity_count == 1 && final == local_player). Emits a
                # single-entity UPDATE_ARRAY with the local player's current
                # pos+rot at the configured cadence; no-op if disabled.
                if (
                    self.solo_local_keepalive_enabled
                    and self.solo_local_keepalive_interval > 0
                    and ctx.session.entity_id
                    and ctx.session.udp_addr
                ):
                    keepalive_due = (now - ctx.last_solo_local_keepalive) >= self.solo_local_keepalive_interval
                    if keepalive_due:
                        ctx.last_solo_local_keepalive = now
                        keep_pos = self._to_client_pos(ctx.player_pos)
                        keep_rot = self._local_player_sync_rotation(ctx)
                        keep_local_state = self._should_send_local_state(
                            ctx,
                            0,
                            0,
                            self.update_local_state_mode,
                        )
                        keep_weapon = self._get_local_state_weapon_type(ctx) if keep_local_state else 0
                        keep_ammo_bits, keep_ammo_mask = (
                            self._get_local_state_ammo_bits(ctx) if keep_local_state else (0, 0)
                        )
                        (
                            keep_pt_bits,
                            keep_pt_angle,
                            keep_st_bits,
                            keep_st_angle,
                        ) = (
                            self._get_local_state_turret_bits(ctx)
                            if keep_local_state
                            else (0, 0.0, 0, 0.0)
                        )
                        keepalive_pkt = build_update_array_player_update(
                            tick=tick,
                            entity_id=ctx.session.entity_id,
                            pos=keep_pos,
                            vel=ctx.player_vel,
                            rot=keep_rot,
                            include_pos=True,
                            include_vel=True,
                            include_rot=True,
                            include_local_state=keep_local_state,
                            weapon_id=keep_weapon,
                            health=self._get_health_value(ctx),
                            fuel=self._get_energy_value(ctx),
                            ammo_count_bits=keep_ammo_bits,
                            ammo_count=keep_ammo_mask,
                            primary_turret_bits=keep_pt_bits,
                            primary_turret_angle=keep_pt_angle,
                            secondary_turret_bits=keep_st_bits,
                            secondary_turret_angle=keep_st_angle,
                            turret_max=self.local_state_turret_max,
                            turret_range=self.local_state_turret_range,
                            is_manned=True,
                        )
                        self.udp_handler.send_to(keepalive_pkt, ctx.session.udp_addr)
                        if self.pktlog.enabled:
                            self.pktlog.log(
                                client_id=ctx.client_id,
                                label="SOLO_LOCAL_KEEPALIVE",
                                tick=tick,
                                payload=keepalive_pkt,
                                transport="UDP",
                                entity_count=1,
                                entity_ids=(ctx.session.entity_id,),
                                mask_bits=(0b1010,),
                                has_local_state=keep_local_state,
                                health=self._get_health_value(ctx) if keep_local_state else -1.0,
                            )

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
                            rot=self._local_player_sync_rotation(ctx),
                            tick=tick,
                            include_vitals=True,
                            weapon_id=self.weapon_id,
                            health=health_val,
                            energy=fuel_val,
                        )
                        self._log_vitals(
                            ctx,
                            "TANK_UDP_HEARTBEAT",
                            include_vitals=True,
                            health=health_val,
                            energy=fuel_val,
                            weapon_id=self.weapon_id,
                            note=f"net_id={ctx.session.entity_id}",
                        )
                        if self.udp_handler and ctx.session.udp_addr:
                            self.udp_handler.send_to(vitals_packet, ctx.session.udp_addr)
                            if self.pktlog.enabled:
                                self.pktlog.log(
                                    client_id=ctx.client_id,
                                    label="TANK_VITALS",
                                    tick=tick,
                                    payload=vitals_packet,
                                    transport="UDP",
                                    has_local_state=True,
                                    health=health_val,
                                    extra="TankPacket",
                                )
                            print(
                                "[VITALS] "
                                f"client={ctx.client_id} net_id={ctx.session.entity_id} "
                                f"tick={tick} addr={ctx.session.udp_addr}"
                            )

                # Send other players' transforms to this client (multiplayer visibility).
                remote_due = (
                    self.remote_update_interval <= 0
                    or (now - ctx.last_remote_update_send) >= self.remote_update_interval
                )
                if self.send_remote_updates and not self.combine_update_arrays and remote_due:
                    ctx.last_remote_update_send = now
                    self._send_remote_player_updates(
                        ctx,
                        tick,
                        prefer_tcp=(self.send_updates_tcp and not tcp_failed),
                    )

                # Track last sent player state for projectile alignment diagnostics.
                # Use player_pos if send_pos not set (heartbeat-only mode)
                if payload is not None:
                    track_pos = send_pos if send_full_update else self._to_client_pos(ctx.player_pos)
                    ctx.last_sent_player_state = {
                        "time": time.monotonic(),
                        "tick": tick,
                        "pos": track_pos,
                        "rot": self._local_player_sync_rotation(ctx),
                        "vel": ctx.player_vel,
                    }

                # Log every 10 ticks to trace movement + health sends
                if self.debug_sync and ctx.session.tick % 10 == 0:
                    px, py, pz = ctx.player_pos
                    vx, vy, vz = ctx.player_vel
                    yaw_deg = math.degrees(ctx.player_yaw)
                    print(f"[TICK] Client {ctx.client_id}: pos=({px:.2f},{py:.2f},{pz:.2f}) vel=({vx:.2f},{vy:.2f},{vz:.2f}) yaw={yaw_deg:.1f}")
                    aim_recent = (time.monotonic() - ctx.player_aim_time) < self.viewpoint_timeout
                    if not aim_recent:
                        print(f"[VIEWPOINT-INPUT] yaw={yaw_deg:.1f}")
                    pkt_type = "FULL" if send_full_update else ("VIEW_BEAT" if self.heartbeat_view_update else "BEAT")
                    udp_addr = ctx.session.udp_addr if ctx.session.udp_addr else "NO_ADDR"
                    # Verify health encoding in payload
                    if payload and len(payload) > 7:
                        if payload[0] == 0x0F:
                            health_hex = payload[9:12].hex() if len(payload) > 11 else "??"
                        else:
                            health_hex = payload[5:8].hex()
                    else:
                        health_hex = "??"
                    update_mask = self._extract_update_mask(payload) if payload else None
                    if update_mask is None:
                        mask_note = "?"
                    else:
                        mask_note = f"0x{update_mask:03x}"
                    print(
                        f"[TICK-HEALTH] t={ctx.session.tick} type={pkt_type} "
                        f"udp={udp_addr} mask={mask_note} health_bytes={health_hex}"
                    )

                # Wall-clock pacing: sleep until next tick boundary
                next_tick_time += tick_period
                sleep_dt = next_tick_time - time.monotonic()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                elif sleep_dt < -tick_period:
                    # Fallen behind by more than one tick â€” reset to avoid burst
                    next_tick_time = time.monotonic()

            except Exception as e:
                print(f"[TICK] Client {ctx.client_id} Error: {e}")
                import traceback
                traceback.print_exc()
                # Don't break - try to continue even with errors
                time.sleep(0.1)

        with ctx.tick_lock:
            if ctx.tick_thread is current_thread:
                ctx.tick_thread = None
        print(f"[TICK] Tick loop ended for client {ctx.client_id}")


def main():
    """Entry point."""
    server = WulframServer()
    server.start()


if __name__ == "__main__":
    main()



