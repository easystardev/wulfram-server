"""
Main server: Orchestrates the protocol flow using layered components.

NOTE: Use manage_server.py to start/stop the server instead of running this directly.
This avoids orphaned processes and provides clean shutdown handling.

    python server/manage_server.py start
    python server/manage_server.py stop
    python server/manage_server.py restart
"""

import json
import math
import os
import secrets
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Optional, Dict

from .session import Session, Phase, FEATURES
from .transport import TCPHandler, UDPHandler, PacketLogger, print_packet
from .codec import BitReader
from .control import ControlServer
from .terrain import Terrain
from .building_collision import BuildingCollisionAssets, BuildingEntity
from .world_collision import TerrainGridCollision
from .weapons import (
    WeaponSystem, build_projectile_spawn_packet, EntityType, BehaviorSlot,
    VEHICLE_PHYSICS_CONFIGS, TANK_WEAPON_SLOTS,
)
from .jump_jets import JumpJetSystem
from .client import ClientContext
from wulfram2_protocol.entities import ACTION_ANALOG_SLOTS, ACTION_DUMP_CONTROL_SLOTS, WEAPON_NAMES
from .packets import (
    PacketType, get_packet_name, get_ticks,
    build_hello_session_key, build_hello_udp_config, build_hello_verified,
    build_identified_udp, build_login_status, build_tank_packet,
    build_udp_tank_packet_wf, build_update_array_heartbeat,
    build_chat_message, build_add_to_roster, build_update_stats, build_player, build_player_info,
    build_birth_notice, build_game_clock, build_reincarnate,
    build_update_array_create_tank, build_update_array_player_update,
    build_update_array_multi, build_view_update_multi, build_view_update_player_update,
    get_behavior_weapon_capability_counts, build_world_stats,
    build_delete_object,
    _encode_health_bits, _compress_value, VEC_VEL_MAX, VEC_VEL_RANGE,
    build_transient_array, FX_CHAIN_GUN_FIRE, FX_PULSE_FIRE,
    FX_FLAK_FIRE, FX_MISSILE_FIRE,
)
from . import handlers
from .pktlog import PacketLog

# WeaponDef turret flags from azurefishy decomp (WeaponDef_init_by_entity_type).
# Tank (entity type 0) sets +0x170 (primary turret). Scout (1) sets +0x68 (secondary).
LOCAL_STATE_PRIMARY_TURRET_TYPES = {0}
LOCAL_STATE_SECONDARY_TURRET_TYPES = {1}


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
        # Keep spawns pinned to ground unless explicitly disabled.
        self.spawn_sets_ground_level = os.environ.get("WULFRAM_SPAWN_SET_GROUND", "1") == "1"
        # Spawn packet toggles (useful for crash isolation).
        # Spawn sequence toggles (default to Wulf-Forge minimal behavior).
        self.spawn_send_udp_tank = os.environ.get("WULFRAM_SPAWN_UDP_TANK", "1") == "1"
        spawn_player_info_env = os.environ.get("WULFRAM_SPAWN_PLAYER_INFO")
        if spawn_player_info_env is None:
            # Wulf-forge does NOT send PLAYER_INFO during spawn.
            self.spawn_send_player_info = False
        else:
            self.spawn_send_player_info = spawn_player_info_env == "1"
        # Pre-create entity via UPDATE_ARRAY before TankPacket so OIDTable has the
        # correct OID when PLAYER_INFO processes it.  Without this,
        # Entity_create_from_network in PLAYER_INFO stores a garbage OID (unaff_EBX)
        # â†’ OIDTable_lookup fails on retransmit â†’ LocalPlayer_initialize never fires
        # â†’ g_local_player_entity stays 0 â†’ sync_local_player skips all health writes.
        self.spawn_send_update_array = os.environ.get("WULFRAM_SPAWN_UPDATE_ARRAY", "1") == "1"
        self.spawn_send_game_clock = os.environ.get("WULFRAM_SPAWN_GAME_CLOCK", "0") == "1"
        # Wulf-forge does NOT send REINCARNATE(0x11) during spawn.
        self.spawn_send_reincarnate = os.environ.get("WULFRAM_SPAWN_REINCARNATE", "0") == "1"
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
        # Load building entities for collision detection
        self._building_entities = {}
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
        self._load_map_buildings()
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
        try:
            self.terrain_height_offset = float(os.environ.get("WULFRAM_TERRAIN_HEIGHT_OFFSET", "5.0"))
        except ValueError:
            self.terrain_height_offset = 5.0
        self._load_terrain()
        self._terrain_grid_collision: Optional[TerrainGridCollision] = None
        self._entity_collision_extents_cache: Dict[tuple[int, int], tuple[float, float, float]] = {}
        self._entity_collision_model_cache: Dict[tuple[int, int], Optional[tuple[object, object, float, float]]] = {}
        if self.terrain is not None:
            self._terrain_grid_collision = TerrainGridCollision(
                self.terrain,
                self.terrain_height_offset,
            )
            print(
                "[COLLISION] Terrain grid collision initialized "
                f"with {self._terrain_grid_collision.sector_count} sectors"
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
        # Jump jets are a custom extension and not part of OG client behavior.
        # Keep disabled by default for protocol fidelity.
        self.jump_jets_enabled = os.environ.get("WULFRAM_JUMP_JETS", "0") == "1"

        print(
            "[CONFIG] spawn_udp_tank="
            f"{int(self.spawn_send_udp_tank)} player_info={int(self.spawn_send_player_info)} "
            f"game_clock={int(self.spawn_send_game_clock)} reincarnate={int(self.spawn_send_reincarnate)} "
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
            f"gravity={self.gravity:.1f} tick_hz={self.tick_rate_hz:.1f} "
            f"update_on_change={int(self.update_on_change)} heartbeat={self.update_heartbeat_interval:.2f}s "
            f"map_spawns={int(self.use_map_spawn_points)} update_packet={self.update_packet_type} "
            f"heartbeat_view={int(self.heartbeat_view_update)} jump_jets={int(self.jump_jets_enabled)} "
            f"inactivity_timeout={self.inactivity_timeout:.1f}s"
        )

        # PLAYER_INFO triggers local-state sync on the client; default OFF to avoid
        # bitstream desync until turret/ammo bits are fully validated.
        player_info_local_state_raw = os.environ.get("WULFRAM_PLAYER_INFO_LOCAL_STATE", "off").strip().lower()
        if player_info_local_state_raw in ("1", "true", "on", "yes", "force"):
            self.player_info_local_state_mode = "force"
        elif player_info_local_state_raw in ("0", "false", "off", "no"):
            self.player_info_local_state_mode = "off"
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
        # Aim/movement configuration (shared across clients)
        # Slot-integrated aim is sensitive to noisy axis samples; keep opt-in.
        self.use_slot_aim = os.environ.get("WULFRAM_USE_SLOT_AIM", "0") == "1"
        print(
            "[CONFIG] projectiles_enabled="
            f"{int(self.projectiles_enabled)} projectile_update_mode={self.projectile_update_mode} "
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
        self.debug_sync = os.environ.get("WULFRAM_DEBUG_SYNC", "0") == "1"
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
        # Client uses TWO different damp values (from Tank_read_throttle_input
        # and Tank_compute_mobility_factors in Vehicles.c):
        #   DRIVING (throttle != 0): damp = ground_friction * terrain_scale + world_damp
        #     On flat ground: 0.8 * 1.0 + 0.0 = 0.8
        #   COASTING (throttle == 0): damp = 2.0 (hardcoded at Vehicles.c:932, 0x40000000)
        try:
            self.linear_damp_driving = float(os.environ.get("WULFRAM_LINEAR_DAMP_DRIVING", "0.8"))
        except ValueError:
            self.linear_damp_driving = 0.8
        try:
            self.linear_damp_coasting = float(os.environ.get("WULFRAM_LINEAR_DAMP_COASTING", "2.0"))
        except ValueError:
            self.linear_damp_coasting = 2.0

        # NOTE: Local player corrections (sending UPDATE_ARRAY with the player's own
        # entity ID) do NOT work. The client runs lockstep deterministic physics and
        # overwrites server position/rotation every frame. Reconciliation only triggers
        # on collision (dead on flat ground). Tested exhaustively: dual_entity, single
        # entity, pos-only, rot-only, teleport â€” none visually correct the local player.
        # Server physics match (0.001% position, 0.00Â° heading) is the sync mechanism.

        print(
            f"[CONFIG-HEADING] turn_adjust={self.turn_adjust} turn_sign={self.turn_sign} "
            f"deadzone={self.turn_deadzone} damp_coeff={self.damp_coeff} "
            f"linear_damp=driving:{self.linear_damp_driving}/coast:{self.linear_damp_coasting} "
            f"tick_rate={self.tick_rate_hz}Hz"
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
        # Send 0 mask (no weapons actively firing).  The active-flags bitmask
        # controls AmmoSlotState+0x05 via update_active_flags(); bit=1 causes
        # the client's WeaponCooldown_update_all() to auto-fire that slot.
        # Only set bits when the player is actually firing (TODO).
        active_mask = 0
        return active_bits, active_mask

    def _get_local_state_turret_bits(self, ctx: ClientContext) -> tuple:
        """
        Return (primary_bits, primary_angle, secondary_bits, secondary_angle).
        Flags are inferred from entity_type unless overrides are provided.
        """
        weapon_type = self._get_local_state_weapon_type(ctx)

        # Default turret bits based on WeaponDef_init_by_entity_type (azurefishy decomp).
        primary_flag = weapon_type in LOCAL_STATE_PRIMARY_TURRET_TYPES
        secondary_flag = weapon_type in LOCAL_STATE_SECONDARY_TURRET_TYPES

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

    def _get_local_state_kwargs(self, ctx: ClientContext) -> dict:
        """Return dict of local_player_state kwargs for any UPDATE_ARRAY builder.

        Every UPDATE_ARRAY with include_local_state=True MUST include the
        correct ammo/turret bit counts matching the BEHAVIOR packet config,
        otherwise the OG client's bitstream reads misalign â†’ protocol mismatch.
        """
        ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
        pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
        return dict(
            weapon_id=self._get_local_state_weapon_type(ctx),
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

    def _local_state_payload_is_safe(self, ctx: ClientContext, primary_bits: int, secondary_bits: int) -> bool:
        """Return True if local-state payload includes required turret angles for this weapon type."""
        weapon_type = self._get_local_state_weapon_type(ctx)
        if weapon_type in LOCAL_STATE_PRIMARY_TURRET_TYPES and primary_bits <= 0:
            return False
        if weapon_type in LOCAL_STATE_SECONDARY_TURRET_TYPES and secondary_bits <= 0:
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
            rot=rot,
            pos=pos,
        )

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
                "rot": (
                    ctx.player_pose.get("roll", 0.0),
                    0.0,
                    ctx.player_yaw,
                ),
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

    def _send_packet_to_client(self, ctx: ClientContext, payload: bytes, *, prefer_tcp: bool = True) -> bool:
        """Send payload to a client, preferring TCP and falling back to UDP."""
        sent = False
        if prefer_tcp and ctx.tcp_handler:
            try:
                ctx.tcp_handler.send(payload, log=False)
                sent = True
            except Exception as tcp_err:
                print(f"[MULTI] Client {ctx.client_id}: TCP send failed ({tcp_err})")
        if not sent and self.udp_handler and ctx.session.udp_addr:
            try:
                self.udp_handler.send_to(payload, ctx.session.udp_addr)
                sent = True
            except Exception as udp_err:
                print(f"[MULTI] Client {ctx.client_id}: UDP send failed ({udp_err})")
        return sent

    def _send_roster_entry(self, target_ctx: ClientContext, player_ctx: ClientContext) -> None:
        """Send ADD_TO_ROSTER for player_ctx to target_ctx (once)."""
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
        if not self._send_packet_to_client(target_ctx, payload, prefer_tcp=True):
            return
        target_ctx.known_roster_ids.add(player_id)
        print(f"[MULTI] Sent roster {name} (id={player_id}) -> client {target_ctx.client_id}")

    def _broadcast_player_stats(self, player_ctx: ClientContext) -> None:
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
            self._send_packet_to_client(client, pkt, prefer_tcp=True)

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
        rot = (
            player_ctx.player_pose.get("roll", 0.0),
            0.0,
            (-player_ctx.player_heading if self.remote_yaw_negate else player_ctx.player_heading) + self.remote_yaw_offset,
        )

        ls = self._get_local_state_kwargs(target_ctx)
        create_pkt = build_update_array_create_tank(
            tick=tick,
            entity_id=entity_id,
            entity_type=player_ctx.entity_type,
            team=team,
            pos=pos,
            is_manned=True,
            rot=rot,
            **ls,
        )
        label = "RETRY" if is_retry else "CREATE"
        print(f"[MULTI] UPDATE_ARRAY DEFINITION {label} id={entity_id} "
              f"type={player_ctx.entity_type} team={team} pos={pos} "
              f"tick={tick} -> client {target_ctx.client_id}")
        if not is_retry:
            print(f"[MULTI-HEX] {create_pkt.hex().upper()}")
        if not self._send_packet_to_client(target_ctx, create_pkt, prefer_tcp=True):
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
            health_val = self._get_health_value(ctx)  # VIEWER's health for local_state HUD
            # Extract weapon/ammo/turret data from the REMOTE player, not the viewer.
            weapon_type = self._get_local_state_weapon_type(other)
            ammo_bits, ammo_mask = self._get_local_state_ammo_bits(other)
            pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(other)
            include_local_state = self._should_send_local_state(
                other,
                pt_bits,
                st_bits,
                self.update_local_state_mode,
            )
            send_pos = self._to_client_pos(other.player_pos)
            payload = build_update_array_player_update(
                tick,
                entity_id,
                pos=send_pos,
                vel=other.player_vel,
                rot=(
                    other.player_pose.get("roll", 0.0),
                    0.0,
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
                weapon_id=weapon_type,
                health=health_val,
                fuel=viewer_fuel,
                ammo_count_bits=ammo_bits,
                ammo_count=ammo_mask,
                primary_turret_bits=pt_bits,
                primary_turret_angle=pt_angle,
                secondary_turret_bits=st_bits,
                secondary_turret_angle=st_angle,
                turret_max=self.local_state_turret_max,
                turret_range=self.local_state_turret_range,
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

            elif pkt_type == 0x0B:  # PING_REQUEST - opcode + u32 timestamp
                pkt_len = min(5, len(data) - cursor)
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
            # Client ping request: echo timestamp back as 0x0C (pong).
            if len(data) >= 5 and self.udp_handler:
                ts = struct.unpack(">I", data[1:5])[0]
                self.udp_handler.send_to(b'\x0C' + struct.pack(">I", ts), addr)
                print(f"[UDP] PING_REPLY 0x0C to {addr} ts={ts}")

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

        # Register.
        with self.clients_lock:
            self.clients[client_id] = ctx
        self.udp_addr_to_client[addr] = ctx

        # Pick spawn point (same logic as normal spawn).
        spawn_pos = None
        if self.map_spawn_points:
            team_spawns = [sp for sp in self.map_spawn_points if sp.get("team") == session.team_id]
            if team_spawns:
                spawn_pos = team_spawns[0]["pos"]
        if spawn_pos is None:
            spawn_pos = (100.0, 100.0, self.spawn_height)

        ctx.player_pos = spawn_pos
        ctx.player_pose["pos"] = spawn_pos
        ctx.player_yaw = 0.0
        ctx.player_heading = 0.0
        ctx.angular_vel_yaw = 0.0
        ctx.vehicle_physics.reset()
        if self.spawn_sets_ground_level:
            ctx.ground_level_override = spawn_pos[2] if self.up_axis == "z" else spawn_pos[1]

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
                rot=(0.0, 0.0, 0.0),
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
        if FEATURES.tick_loop_enabled and (ctx.tick_thread is None or not ctx.tick_thread.is_alive()):
            ctx.tick_thread = threading.Thread(target=self._tick_loop, args=(ctx,), daemon=True)
            ctx.tick_thread.start()
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

        # Reset position tracking to spawn location (prefer map-provided coords).
        spawn_override = os.environ.get("WULFRAM_SPAWN_POS")
        if pos is not None:
            spawn_pos = pos
        elif spawn_override:
            parts = [float(x) for x in spawn_override.split(",")]
            spawn_pos = (parts[0], parts[1], parts[2] if len(parts) > 2 else self.spawn_height)
            print(f"[SPAWN] Using spawn point override pos={spawn_pos}")
        else:
            map_spawn = self._pick_spawn_point(team_id)
            if map_spawn:
                spawn_pos = (map_spawn["x"], map_spawn["y"], map_spawn["z"])
                print(f"[SPAWN] Auto-selected map spawn oid={map_spawn['oid']} team={team_id} pos={spawn_pos}")
            elif self.up_axis == "z":
                spawn_pos = (100.0, 100.0, self.spawn_height)
            else:
                spawn_pos = (100.0, self.spawn_height, 100.0)

        # Offset spawn to avoid overlapping tanks in multi-client tests.
        # Use 0-based index among active clients (not client_id, which grows unboundedly).
        if self.multi_spawn_offset:
            with self.clients_lock:
                active_ids = sorted(c.client_id for c in self.clients.values() if c and c.running)
            try:
                idx = active_ids.index(ctx.client_id)
            except ValueError:
                idx = 0
            spawn_pos = (spawn_pos[0] + idx * self.multi_spawn_offset, spawn_pos[1], spawn_pos[2])

        # Adjust spawn Z to terrain height when terrain is loaded.
        if self.terrain and self.up_axis == "z":
            terrain_z = (
                self.terrain.get_height(spawn_pos[0], spawn_pos[1])
                + self.terrain_height_offset
            )
            spawn_pos = (spawn_pos[0], spawn_pos[1], terrain_z)

        ctx.player_pos = spawn_pos
        ctx.player_pose["pos"] = spawn_pos

        ctx.player_angular_vel = 0.0
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
        ctx.player_energy = self.player_energy_max
        ctx.vehicle_physics.heading = spawn_yaw
        ctx.last_action_dump_time = time.monotonic()  # Reset timer for position tracking
        if self.spawn_sets_ground_level:
            if self.up_axis == "z":
                ctx.ground_level_override = spawn_pos[2]
            else:
                ctx.ground_level_override = spawn_pos[1]
        else:
            ctx.ground_level_override = None

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
        send_rot = (
            ctx.player_pose.get("roll", 0.0),
            0.0,
            ctx.player_yaw,
        )
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
            # Always send pre-creation UPDATE_ARRAY via TCP for reliability.
            # After DELETE_OBJECT (respawn), the client may stop processing UDP
            # while on the team-select screen, causing UDP UPDATE_ARRAY to be
            # lost and the subsequent TankPacket to take the wrong code path
            # (Entity_create_from_network instead of LocalPlayer_initialize).
            ctx.tcp_handler.send(ua_packet)
            print("[SPAWN] Sent UPDATE_ARRAY_CREATE_TANK via TCP (reliable)")
            # Also send via UDP as backup (client processes whichever arrives first)
            if self.udp_handler and ctx.session.udp_addr:
                self.udp_handler.send_to(ua_packet, ctx.session.udp_addr)
                print("[SPAWN] Also sent UPDATE_ARRAY_CREATE_TANK via UDP")
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

        # Wulf-forge order: CommMessage FIRST, then TankPacket.
        if announce:
            comm_pkt = build_chat_message("Spawning in...", source_id=net_id)
            if self.udp_handler and ctx.session.udp_addr:
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

        # Send TankPacket over UDP (matching wulf-forge behavior)
        if self.spawn_send_udp_tank:
            if self.udp_handler and ctx.session.udp_addr:
                self.udp_handler.send_to(tank_packet, ctx.session.udp_addr)
                print(f"[SPAWN] Sent UDP TankPacket to {ctx.session.udp_addr}")
                # Also send via TCP as backup.  After combat kill + entity DELETE,
                # the client may not process UDP while on team-select overlay.
                # With pre-creation UPDATE_ARRAY, entity exists in OIDTable so
                # TCP PLAYER_INFO takes "entity found" path â†’ LocalPlayer_initialize.
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

        # NOTE: We rely on TankPacket (UDP) to create the entity.
        # UPDATE_ARRAY_CREATE_TANK causes crash when sent AFTER TankPacket.
        # For now, skip UPDATE_ARRAY and try PLAYER_INFO alone.

        # PLAYER_INFO tells the client "this is your controllable entity"
        # Without this, client won't send VIEWPOINT_INFO (0x35)
        weapon_type = self._get_local_state_weapon_type(ctx)
        ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
        pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
        if self.player_info_local_state_mode != "off":
            print(
                "[LOCAL-STATE] PLAYER_INFO "
                f"weapon={weapon_type} ammo_bits={ammo_bits} "
                f"pt_bits={pt_bits} st_bits={st_bits}"
            )
        include_player_info_state = self._should_send_local_state(
            ctx,
            pt_bits,
            st_bits,
            self.player_info_local_state_mode,
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
        if not self.spawn_send_player_info:
            print("[SPAWN] Skipping TCP PLAYER_INFO (WULFRAM_SPAWN_PLAYER_INFO=0)")
        else:
            player_info_pkt = build_player_info(
                entity_oid=net_id,
                vehicle_type=unit_type,
                pos=send_pos,
                rot=send_rot,
                include_local_state=include_player_info_state,
                weapon_id=weapon_type,
                health=1.0,
                fuel=1.0,
                properties=player_info_props,
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

        if FEATURES.tick_loop_enabled and (ctx.tick_thread is None or not ctx.tick_thread.is_alive()):
            ctx.tick_thread = threading.Thread(target=self._tick_loop, args=(ctx,), daemon=True)
            ctx.tick_thread.start()
            print(f"[SPAWN] Started tick loop (local_state_updates={self.update_local_state_mode})")

    def _spawn_wf_minimal(self, ctx: ClientContext, team_id: int, net_id: int, addr: tuple):
        """
        Absolutely minimal spawn - just TankPacket (wulf-forge style).

        Uses include_vitals per WULFRAM_TANK_VITALS (defaults off while investigating).
        TRANSLATION quantizers define 5/10/10 bits which matches TankPacket format.
        """
        print(f"[SPAWN] Minimal WF: client={ctx.client_id} net_id={net_id} team={team_id}")
        ctx.entity_type = 0

        map_spawn = self._pick_spawn_point(team_id)
        if map_spawn:
            spawn_pos = (map_spawn["x"], map_spawn["y"], map_spawn["z"])
            print(f"[SPAWN] Minimal mode map spawn oid={map_spawn['oid']} team={team_id} pos={spawn_pos}")
        elif self.up_axis == "z":
            spawn_pos = (100.0, 100.0, self.spawn_height)
        else:
            spawn_pos = (100.0, self.spawn_height, 100.0)
        if self.spawn_sets_ground_level:
            if self.up_axis == "z":
                ctx.ground_level_override = spawn_pos[2]
            else:
                ctx.ground_level_override = spawn_pos[1]
        else:
            ctx.ground_level_override = None

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
        if self.spawn_sets_ground_level:
            if self.up_axis == "z":
                ctx.ground_level_override = spawn_pos[2]
            else:
                ctx.ground_level_override = spawn_pos[1]
        else:
            ctx.ground_level_override = None
        ctx.player_energy = self.player_energy_max
        tank_packet = build_udp_tank_packet_wf(
            net_id=net_id,
            unit_type=0,
            team_id=team_id,
            pos=send_pos,
            rot=(0.0, 0.0, 0.0),
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

        Format: "oid,team,x,y,z;team,x,y,z;..." (oid optional).
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
            if len(parts) not in (4, 5):
                print(f"[MAP] Invalid spawn point entry '{item}' (expected 4 or 5 fields)")
                return None
            try:
                if len(parts) == 5:
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
            points.append({"oid": oid, "team": team, "x": x, "y": y, "z": z})
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
                print(f"[TERRAIN] Height offset: {self.terrain_height_offset}")
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
                z = float(parts[data_start + 3])
            except (ValueError, IndexError):
                continue

            points.append({"oid": oid, "team": team, "x": x, "y": y, "z": z})
            oid += 1

        if points:
            print(f"[MAP] Loaded {len(points)} spawn points from {state_path}")
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
            return

        try:
            lines = state_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            self._building_entities = {}
            return

        buildings = {}
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
                z = float(parts[data_start + 3])
            except (ValueError, IndexError):
                continue

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
        if buildings:
            print(f"[MAP] Loaded {len(buildings)} building entities for collision from {state_path}")

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
            ctx.running = False
            ctx.session.in_game = False

            # Remove from client tracking
            with self.clients_lock:
                if ctx.client_id in self.clients:
                    del self.clients[ctx.client_id]

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

    def _do_handshake(self, ctx: ClientContext):
        """Perform initial handshake."""
        time.sleep(0.5)  # Let client initialize

        # Send a per-client session key first so UDP can bind deterministically.
        if not ctx.session.session_key:
            ctx.session.session_key = f"WS-{ctx.client_id}-{secrets.token_hex(8)}"
        self.session_key_to_client[ctx.session.session_key] = ctx
        ctx.tcp_handler.send(build_hello_session_key(ctx.session.session_key))

        # Send UDP config - use public_addr, or resolve from client's TCP connection
        udp_addr = self.public_addr
        if udp_addr == "0.0.0.0":
            # Use the local address the client actually connected to
            udp_addr = ctx.tcp_handler.sock.getsockname()[0]
        print(f"[HANDSHAKE] Client {ctx.client_id} UDP config: {udp_addr}:{self.port}")
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
            ctx.tcp_handler.sock.settimeout(1.0)
            try:
                for _ in range(5):  # read up to 5 packets within timeout
                    packet = ctx.tcp_handler.recv()
                    if packet and len(packet) >= 2 and packet[0] == PacketType.LOGIN_REQUEST:
                        # Extract username directly (offset 2 = length-prefixed string)
                        username, _ = handlers.decode_lp_string(packet, 2)
                        if username:
                            ctx.session.username = username
                        break
                    elif packet and len(packet) >= 1 and packet[0] == PacketType.HELLO:
                        self._handle_hello(ctx, packet)
            except Exception:
                pass  # timeout or read error â€” proceed with default name
            finally:
                ctx.tcp_handler.sock.settimeout(None)
            if not ctx.session.username:
                ctx.session.username = f"Player{ctx.client_id}"
            ctx.session.login_complete = True
            ctx.session.transition_to(Phase.TEAM_SELECT)
            handlers.send_initial_game_data(self, ctx)
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

    def _game_loop(self, ctx: ClientContext):
        """Main game packet loop."""
        # Set socket timeout for delayed spawn checking (recv returns None on timeout)
        ctx.tcp_handler.sock.settimeout(0.5)

        # Track last activity for dead connection detection
        # UDP packets count as activity since client sends TRANSLATION_ACK continuously
        last_activity = time.monotonic()
        inactivity_timeout = self.inactivity_timeout

        while ctx.running and ctx.session.phase in [Phase.TEAM_SELECT, Phase.SPAWNING, Phase.IN_GAME]:
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
                send_rot = (
                    ctx.player_pose.get("roll", 0.0),
                    0.0,
                    ctx.player_yaw,
                )
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
            # Check for env var spawn override first
            spawn_override = os.environ.get("WULFRAM_SPAWN_POS")
            if spawn_override:
                parts = [float(x) for x in spawn_override.split(",")]
                pos = (parts[0], parts[1], parts[2] if len(parts) > 2 else self.spawn_height)
                print(f"[SPAWN] Using env spawn override pos={pos}")
            else:
                spawn = self._pick_spawn_point(team_id)
                if spawn:
                    pos = (spawn["x"], spawn["y"], spawn["z"])
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
        send_rot = (
            ctx.player_pose.get("roll", 0.0),
            0.0,
            ctx.player_yaw,
        )
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

        # Log occasionally to avoid spam
        if request_id % 1000 == 0:
            print(
                f"[0x0C] STATE_REQUEST request_id={request_id} "
                f"frame_count={frame_count} len={len(data)}"
            )

        self._send_state_sync_snapshot(ctx, reason="state_request")

    def _send_state_sync_snapshot(self, ctx: ClientContext, *, reason: str) -> None:
        """Send an on-demand local state snapshot for timing/state resync.

        The original client has an explicit STATE_REQUEST (0x0C) path for
        requesting sync. We respond with the existing gameplay update builders:
        - UPDATE_ARRAY with authoritative local pos/vel/rot + optional local state
        - auxiliary VIEW_UPDATE replay/correction packet with the same transform
        """
        if not self.udp_handler or not ctx.session or not ctx.session.udp_addr:
            return

        entity_id = ctx.session.entity_id or ctx.entity_id
        if entity_id == 0:
            return

        now = time.monotonic()
        if (now - ctx.last_state_sync_send) < 0.10:
            return
        ctx.last_state_sync_send = now

        tick = self._get_network_tick(ctx)
        send_pos = self._to_client_pos(ctx.player_pos)
        update_rot = (
            ctx.player_pose.get("roll", 0.0),
            0.0,
            ctx.player_heading,
        )
        view_rot = (
            ctx.player_pose.get("roll", 0.0),
            0.0,
            ctx.player_yaw,
        )
        health_val = self._get_health_value(ctx)
        fuel_val = self._get_energy_value(ctx)
        weapon_type = self._get_local_state_weapon_type(ctx)
        ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
        pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)

        update_include_local_state = self._should_send_local_state(
            ctx,
            pt_bits,
            st_bits,
            self.update_local_state_mode,
        )
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

        update_payload = build_update_array_player_update(
            tick=tick,
            entity_id=entity_id,
            pos=send_pos,
            vel=ctx.player_vel,
            rot=update_rot,
            include_pos=True,
            include_vel=True,
            include_rot=True,
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

        view_include_local_state = False
        view_ammo_bits = 0
        view_ammo_mask = 0
        view_pt_bits = 0
        view_pt_angle = 0.0
        view_st_bits = 0
        view_st_angle = 0.0
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

        view_payload = build_view_update_player_update(
            tick=tick,
            entity_id=entity_id,
            pos=send_pos,
            vel=ctx.player_vel,
            rot=view_rot,
            include_pos=True,
            include_vel=True,
            include_rot=True,
            include_local_state=view_include_local_state,
            include_entity_vitals=self.view_update_entity_vitals,
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
            is_manned=True,
            speed_scale=1.0,
        )

        self.udp_handler.send_to(update_payload, ctx.session.udp_addr)
        self.udp_handler.send_to(view_payload, ctx.session.udp_addr)

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
            )
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
            )

        if self.debug_viewpoint or self.debug_udp_raw:
            print(
                f"[STATE-SYNC] client={ctx.client_id} reason={reason} "
                f"tick={tick} pos=({send_pos[0]:.2f},{send_pos[1]:.2f},{send_pos[2]:.2f})"
            )

    # ============ Weapon System Handlers ============

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

            if ctx.session and not ctx.session.input_ready:
                ctx.session.input_ready = True
                ctx.session.input_ready_time = time.monotonic()
                print(f"[GAME] Client {ctx.client_id}: input ready (ACTION_DUMP)")
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
            elif self.projectile_aim_source == "auto":
                # Keep dynamic source from _get_aim_rotation().
                aim_override = aim_src
            ctx.weapon_system.player_rot = (
                ctx.player_pose.get("roll", 0.0),
                aim_pitch,
                aim_yaw,
            )
            ctx.weapon_system.projectile_aim_source = aim_override if aim_override != "auto" else aim_src

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
                        f"aim_yaw={math.degrees(aim_yaw):.1f} vp_yaw={vp_yaw:.1f} "
                        f"src={aim_src} turn_slot={turn_val:.3f} pos={ctx.player_pos}"
                    )
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
            elif self.projectile_aim_source == "auto":
                # Keep dynamic source from _get_aim_rotation().
                aim_override = aim_src
            ctx.weapon_system.player_rot = (
                ctx.player_pose.get("roll", 0.0),
                aim_pitch,
                aim_yaw,
            )
            ctx.weapon_system.projectile_aim_source = aim_override if aim_override != "auto" else aim_src

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
                        f"aim_yaw={math.degrees(aim_yaw):.1f} vp_yaw={vp_yaw:.1f} "
                        f"src={aim_override if aim_override != 'auto' else aim_src} "
                        f"turn_slot={turn_val:.3f} pos={ctx.player_pos}"
                    )
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

        # Send visual feedback
        if ctx.tcp_handler:
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

    def _broadcast_weapon_fire_fx(self, ctx: ClientContext, proj):
        """Broadcast TRANSIENT_ARRAY weapon fire FX to all clients except the firer.

        NOTE: TRANSIENT_ARRAY is sent via UDP only. Our simplified wire format
        (raw bytes) doesn't match the OG client's quantized bitstream format,
        causing TCP stream desync -> MAX_STREAM_DATA crash. The Python client
        handles both formats. FX is cosmetic-only so UDP loss is acceptable.
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
        pkt = build_transient_array(events)
        if not pkt:
            return

        # DISABLED: Our simplified TRANSIENT_ARRAY format (raw bytes) doesn't match
        # the OG client's quantized bitstream format. Even on UDP, the OG client
        # may crash parsing garbage entity_ids or positions from misaligned data.
        # Re-enable once build_transient_array uses proper quantized bitstream.
        # for target in self._snapshot_in_game_clients():
        #     if target is ctx:
        #         continue
        #     if self.udp_handler and target.session.udp_addr:
        #         self.udp_handler.send_to(pkt, target.session.udp_addr)
        return  # No-op until TRANSIENT_ARRAY format is fixed

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
        # Keep projectile packets entity-only unless explicitly re-enabled for
        # targeted experiments. Local-state here has historically desynced the
        # client's UPDATE_ARRAY decode and made shells invisible or malformed.
        include_local_state = False
        sent_count = 0
        if self.udp_handler:
            for target in self._snapshot_in_game_clients():
                if not target.session.udp_addr or not target.session.translation_ack_received:
                    continue
                tick = self._get_network_tick(target)
                packet = build_projectile_spawn_packet(
                    proj,
                    tick,
                    include_local_state=include_local_state,
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
                        health=-1.0,
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
        if ctx.tcp_handler:
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
        return _f32(_f32_turn_adjust * _f32(raw_input))

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

        Dual-damp model (from Tank_read_throttle_input / Tank_compute_mobility_factors):
          DRIVING (throttle != 0): linear_damp = ground_friction (0.8 on flat ground)
          COASTING (throttle == 0): linear_damp = 2.0 (hardcoded, Vehicles.c:932)
        """
        import math

        def _normalize_axis(val: float) -> float:
            if val > 1.5 or val < -1.5:
                scale = getattr(ctx.weapon_system, "control_max", 1000.0) or 1000.0
                return max(-1.0, min(1.0, val / scale))
            return max(-1.0, min(1.0, val))

        dt = dt_override if dt_override > 0 else 1.0 / self.tick_rate_hz

        # Read movement input (slot 2 = forward, slot 3 = strafe)
        if ctx.injected_input is not None:
            throttle_input, strafe_input = ctx.injected_input
        else:
            throttle_val = ctx.weapon_system.behavior_slots[BehaviorSlot.MOVING_FORWARD]
            strafe_val = ctx.weapon_system.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS]
            throttle_input = _normalize_axis(throttle_val)
            strafe_input = _normalize_axis(strafe_val)

        if abs(throttle_input) < 0.05:
            throttle_input = 0.0
        if abs(strafe_input) < 0.05:
            strafe_input = 0.0

        # Per-vehicle-type physics from shared config (decompile-verified)
        veh_config = VEHICLE_PHYSICS_CONFIGS.get(ctx.entity_type)
        move_adjust = veh_config.move_adjust if veh_config else 85.0
        strafe_adjust = veh_config.strafe_adjust if veh_config else 69.7
        # Dual-damp: client uses 0.8 when driving, 2.0 when coasting
        # (from Tank_read_throttle_input and Tank_compute_mobility_factors in Vehicles.c)
        has_input = abs(throttle_input) > 0.0 or abs(strafe_input) > 0.0
        linear_damp = self.linear_damp_driving if has_input else self.linear_damp_coasting

        yaw = heading_override if heading_override is not None else ctx.player_heading
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        if self.up_axis == "z":
            # When terrain is loaded, apply pitch to forward vector so impulse
            # on slopes has a vertical component (matching client's
            # TankVehicle_apply_physics which rotates by full 3D orientation).
            if self.terrain and self.terrain_pitch_enabled:
                terrain_pitch = self.terrain.get_pitch_at_heading(
                    ctx.player_pos[0], ctx.player_pos[1], yaw
                )
                cos_pitch = math.cos(terrain_pitch)
                sin_pitch = math.sin(terrain_pitch)
                forward = (cos_pitch * cos_yaw, cos_pitch * sin_yaw, sin_pitch)
            else:
                forward = (cos_yaw, sin_yaw, 0.0)
            right = (-sin_yaw, cos_yaw, 0.0)
            vertical_idx = 2
        else:
            forward = (cos_yaw, 0.0, sin_yaw)
            right = (-sin_yaw, 0.0, cos_yaw)
            vertical_idx = 1

        # Per-frame impulse (like entity[0x24], zeroed each frame by controller)
        fwd_impulse = throttle_input * move_adjust
        strafe_impulse = strafe_input * strafe_adjust

        impulse_x = forward[0] * fwd_impulse + right[0] * strafe_impulse
        impulse_y = forward[1] * fwd_impulse + right[1] * strafe_impulse
        impulse_z = forward[2] * fwd_impulse + right[2] * strafe_impulse

        # Add gravity to vertical impulse (matches GUESS3_Transform_accelerate_z)
        gravity = self.gravity
        if self.terrain and self.up_axis == "z":
            ground_level = (
                self.terrain.get_height(ctx.player_pos[0], ctx.player_pos[1])
                + self.terrain_height_offset
            )
        else:
            ground_level = self.ground_level
            if ctx.ground_level_override is not None:
                ground_level = ctx.ground_level_override

        # Gravity and ground collision use terrain-aware ground_level (computed above).
        if vertical_idx == 2:
            impulse_z += gravity  # gravity is negative
            if ctx.player_pos[2] <= ground_level and ctx.player_vel[2] + gravity * dt < 0:
                impulse_z = 0.0
        else:
            impulse_y += gravity
            if ctx.player_pos[1] <= ground_level and ctx.player_vel[1] + gravity * dt < 0:
                impulse_y = 0.0

        # Current persistent velocity (entity[0x18])
        vel_x, vel_y, vel_z = ctx.player_vel

        # Damped effective acceleration: acc = impulse - vel * linear_damp
        # (from RigidBody_integrate_position, damped mode at Physics.c:5126-5129)
        acc_x = impulse_x - vel_x * linear_damp
        acc_y = impulse_y - vel_y * linear_damp
        acc_z = impulse_z - vel_z * linear_damp

        # Ground collision: zero vertical acc+vel when on ground and pushing down
        if vertical_idx == 2:
            if ctx.player_pos[2] <= ground_level and (vel_z + acc_z * dt) < 0:
                acc_z = -vel_z / dt if dt > 0 else 0.0  # bring vel to zero
        else:
            if ctx.player_pos[1] <= ground_level and (vel_y + acc_y * dt) < 0:
                acc_y = -vel_y / dt if dt > 0 else 0.0

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
            if self.terrain:
                terrain_z = (
                    self.terrain.get_height(new_x, new_y)
                    + self.terrain_height_offset
                )
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

        # Decompile-shaped terrain/world contact pass before static blockers.
        new_x, new_y, new_z, new_vel_x, new_vel_y, new_vel_z = self._resolve_entity_world_collision(
            ctx, new_x, new_y, new_z, new_vel_x, new_vel_y, new_vel_z
        )

        # Building AABB collision (matching client-side)
        new_x, new_y, new_vel_x, new_vel_y = self._check_building_collisions(
            ctx, new_x, new_y, new_z, new_vel_x, new_vel_y)

        old_pos = ctx.player_pos
        ctx.player_pos = (new_x, new_y, new_z)
        ctx.player_vel = (new_vel_x, new_vel_y, new_vel_z)
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


    # Building AABB half-extents matching client-side table
    _BUILDING_HALF_EXTENTS = {
        25: (12.0, 12.0), 26: (8.0, 8.0), 27: (6.0, 6.0), 28: (5.0, 5.0),
        29: (10.0, 10.0), 30: (7.0, 7.0), 31: (6.0, 6.0), 32: (8.0, 8.0),
        33: (5.0, 5.0), 34: (4.0, 4.0), 35: (6.0, 6.0), 36: (5.0, 5.0),
        37: (7.0, 7.0),
    }
    _TANK_RADIUS = 4.0
    _BUILDING_HALF_HEIGHT = 20.0
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
    }

    def _get_entity_world_half_extents(self, ctx: ClientContext) -> tuple[float, float, float]:
        team_id = ctx.session.team_id or 1
        cache_key = (ctx.entity_type, team_id)
        cached = self._entity_collision_extents_cache.get(cache_key)
        if cached is not None:
            return cached

        half_extents = (self._TANK_RADIUS, self._TANK_RADIUS, self._TANK_RADIUS)
        model_names = self._ENTITY_WORLD_MODEL_NAMES.get(ctx.entity_type)
        if model_names and self._building_collision.available:
            model_name = model_names[1] if team_id == 2 and len(model_names) > 1 else model_names[0]
            model = self._building_collision.models.get(model_name)
            mesh = getattr(model, "collision_mesh", None) if model is not None else None
            vertices = getattr(mesh, "vertices", None) if mesh is not None else None
            if vertices:
                xs = [v.x for v in vertices]
                ys = [v.y for v in vertices]
                half_extents = (
                    max(self._TANK_RADIUS, max(abs(min(xs)), abs(max(xs)))),
                    max(self._TANK_RADIUS, max(abs(min(ys)), abs(max(ys)))),
                    self._TANK_RADIUS,
                )

        self._entity_collision_extents_cache[cache_key] = half_extents
        return half_extents

    def _resolve_entity_world_collision(self, ctx, px, py, pz, vx, vy, vz):
        if self._terrain_grid_collision is None:
            return px, py, pz, vx, vy, vz

        half_extents = self._get_entity_world_half_extents(ctx)
        heading = ctx.player_heading
        anchor = [px, py, pz]
        collision_model = self._get_entity_world_collision_model(ctx)
        if collision_model is not None:
            vertices, cbsp_tree, bounding_radius, z_lift = collision_model
        else:
            vertices = None
            cbsp_tree = None
            bounding_radius = 0.0
            z_lift = half_extents[2]

        # Multiple shallow terrain contacts can stack across adjacent cells.
        for _ in range(2):
            box_center = (anchor[0], anchor[1], anchor[2] + half_extents[2])
            contact = self._terrain_grid_collision.test_box_collision(
                box_center,
                half_extents,
                heading,
            )

            if collision_model is not None:
                model_center = (anchor[0], anchor[1], anchor[2] + z_lift)
                mesh_contact = self._terrain_grid_collision.test_model_collision(
                    model_center,
                    heading,
                    vertices,
                    cbsp_tree,
                    bounding_radius,
                )
                if (
                    mesh_contact is not None and
                    contact is not None and
                    contact.penetration > 0.001 and
                    mesh_contact.penetration <= (contact.penetration + 0.5)
                ):
                    contact = mesh_contact
            if contact is None or contact.penetration <= 0.001:
                break
            push = contact.penetration + 0.01
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

        return anchor[0], anchor[1], anchor[2], vx, vy, vz

    def _get_entity_world_collision_model(self, ctx: ClientContext):
        team_id = ctx.session.team_id or 1
        cache_key = (ctx.entity_type, team_id)
        if cache_key in self._entity_collision_model_cache:
            return self._entity_collision_model_cache[cache_key]

        model_names = self._ENTITY_WORLD_MODEL_NAMES.get(ctx.entity_type)
        if not model_names or not self._building_collision.available:
            self._entity_collision_model_cache[cache_key] = None
            return None

        model_name = model_names[1] if team_id == 2 and len(model_names) > 1 else model_names[0]
        model = self._building_collision.models.get(model_name)
        mesh = getattr(model, "collision_mesh", None) if model is not None else None
        vertices = getattr(mesh, "vertices", None) if mesh is not None else None
        cbsp_tree = getattr(model, "cbsp_tree", None) if model is not None else None
        if not vertices or cbsp_tree is None or not cbsp_tree.nodes:
            self._entity_collision_model_cache[cache_key] = None
            return None

        root = cbsp_tree.root
        bounding_radius = root.radius if root is not None else 0.0
        min_z = None
        for vertex in vertices:
            if min_z is None or vertex.z < min_z:
                min_z = vertex.z
        z_lift = max(0.0, -(min_z or 0.0))
        result = (vertices, cbsp_tree, bounding_radius, z_lift)
        self._entity_collision_model_cache[cache_key] = result
        return result

    def _get_projectile_collision_radius(self, proj) -> float:
        radius = self.projectile_collision_radius
        model_names = self._PROJECTILE_MODEL_NAMES.get(proj.entity_type)
        if not model_names or not self._building_collision.available:
            return radius

        team_id = getattr(proj, "team", 1)
        model_name = model_names[1] if team_id == 2 and len(model_names) > 1 else model_names[0]
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
                nx, ny = dx / dist, dy / dist
                px += nx * overlap * 0.5
                py += ny * overlap * 0.5
                vel_dot = vx * nx + vy * ny
                if vel_dot < 0:
                    vx -= nx * vel_dot
                    vy -= ny * vel_dot

        # Check buildings loaded from map state file
        building_entities = self._building_entities
        for eid, building in building_entities.items():
            mesh_hit = False
            if self._building_collision.available:
                depth, normal = self._building_collision.test_sphere_collision(
                    building,
                    (px, py, pz),
                    self._TANK_RADIUS,
                )
                if depth > 0.0 and normal:
                    px += normal[0] * depth
                    py += normal[1] * depth
                    vel_dot = vx * normal[0] + vy * normal[1]
                    if vel_dot < 0:
                        vx -= normal[0] * vel_dot
                        vy -= normal[1] * vel_dot
                    mesh_hit = True

            if mesh_hit:
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
                if mp == push_xp:
                    px = bx + hx + r
                    if vx < 0: vx = 0.0
                elif mp == push_xn:
                    px = bx - hx - r
                    if vx > 0: vx = 0.0
                elif mp == push_yp:
                    py = by + hy + r
                    if vy < 0: vy = 0.0
                elif mp == push_yn:
                    py = by - hy - r
                    if vy > 0: vy = 0.0

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
        """Sweep a projectile segment against terrain and static buildings."""
        start_pos = self._from_client_pos(start_client_pos)
        end_pos = self._from_client_pos(end_client_pos)
        radius = self._get_projectile_collision_radius(proj) if proj is not None else self.projectile_collision_radius

        seg_dx = end_pos[0] - start_pos[0]
        seg_dy = end_pos[1] - start_pos[1]
        seg_dz = end_pos[2] - start_pos[2]
        seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy + seg_dz * seg_dz)
        step_span = max(radius * 0.5, 0.5)
        steps = max(1, int(math.ceil(seg_len / step_span)))

        for step_idx in range(1, steps + 1):
            t = step_idx / steps
            sample_x = start_pos[0] + seg_dx * t
            sample_y = start_pos[1] + seg_dy * t
            sample_z = start_pos[2] + seg_dz * t

            if self._terrain_grid_collision is not None:
                terrain_hit = self._terrain_grid_collision.test_sphere_collision(
                    (sample_x, sample_y, sample_z),
                    radius,
                )
                if terrain_hit is not None:
                    return ("terrain", terrain_hit.position, terrain_hit.sector_index)
            else:
                if self.terrain is not None:
                    ground_z = self.terrain.get_height(sample_x, sample_y) + self.terrain_height_offset
                else:
                    ground_z = self.ground_level
                if sample_z <= ground_z + radius:
                    return ("terrain", (sample_x, sample_y, ground_z), None)

            for eid, building in self._building_entities.items():
                if self._building_collision.available:
                    depth, normal = self._building_collision.test_sphere_collision(
                        building,
                        (sample_x, sample_y, sample_z),
                        radius,
                    )
                    if depth > 0.0 and normal is not None:
                        return ("building", (sample_x, sample_y, sample_z), eid)

                hx, hy = self._BUILDING_HALF_EXTENTS.get(building.entity_type, (8.0, 8.0))
                half_h = max(hx, hy, self._BUILDING_HALF_HEIGHT)
                if (
                    sample_x > building.x - hx - radius and
                    sample_x < building.x + hx + radius and
                    sample_y > building.y - hy - radius and
                    sample_y < building.y + hy + radius and
                    sample_z > building.z - half_h - radius and
                    sample_z < building.z + half_h + radius
                ):
                    return ("building-aabb", (sample_x, sample_y, sample_z), eid)

        return None

    def _send_debug_sync(self, ctx: ClientContext, frame_counter: int,
                         turn_input: float, fwd_input: float, strafe_input: float):
        """Send DEBUG_SYNC packet (0x60) with server's authoritative state.

        49-byte UDP packet sent after each physics step for measuring
        client-server divergence. Opcode 0x60 is unused by the original protocol.
        """
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
        self._broadcast_weapon_fire_fx(ctx, proj)

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
                    delete_with_effects = True
                    with ctx.projectile_lock:
                        if proj in ctx.active_projectiles:
                            ctx.active_projectiles.remove(proj)
                    if hit_kind == "terrain":
                        print(
                            f"[PROJ-WORLD] id={proj.entity_id} hit terrain "
                            f"at=({hit_pos[0]:.1f},{hit_pos[1]:.1f},{hit_pos[2]:.1f})"
                        )
                    else:
                        print(
                            f"[PROJ-WORLD] id={proj.entity_id} hit {hit_kind} "
                            f"target={hit_detail} at=({hit_pos[0]:.1f},{hit_pos[1]:.1f},{hit_pos[2]:.1f})"
                        )
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
                if self.udp_handler:
                    for target in self._snapshot_in_game_clients():
                        if not target.session.udp_addr or not target.session.translation_ack_received:
                            continue
                        tick = self._get_network_tick(target)
                        pkt = build_projectile_update_packet(
                            proj,
                            tick,
                            0.0,  # Position already advanced above
                            include_local_state=False,
                        )
                        self.udp_handler.send_to(pkt, target.session.udp_addr)
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
                                has_local_state=False,
                                health=-1.0,
                            )

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
                self._send_packet_to_client(client, delete_proj_pkt, prefer_tcp=True)
            return

        damage = 0.2  # Pulse shell = 20% per hit (5 hits to kill)
        old_health = target.player_health
        target.player_health = round(max(0.0, old_health - damage), 6)
        new_health = target.player_health

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

        # DELETE projectile with explosion effects
        tick = self._get_network_tick(attacker)
        delete_proj_pkt = build_delete_object(tick, [proj.entity_id], with_effects=True)
        for client in self._snapshot_in_game_clients():
            self._send_packet_to_client(client, delete_proj_pkt, prefer_tcp=True)

        # Chat notification
        hit_msg = f"HIT! {target_name} ({new_health*100:.0f}% health)"
        chat_pkt = build_chat_message(hit_msg, source_id=attacker.session.player_id or attacker.entity_id)
        for client in self._snapshot_in_game_clients():
            if client.tcp_handler:
                client.tcp_handler.send(chat_pkt)

        # Send health refresh to ALL surviving clients (attacker etc.)
        # Projectile UPDATE_ARRAY packets include per-viewer health, but once
        # projectiles are gone the viewer gets no more health data. This
        # heartbeat ensures the attacker's HUD doesn't revert to zero.
        for client in self._snapshot_in_game_clients():
            if client is target:
                continue  # Already sent above
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
                if client.tcp_handler:
                    client.tcp_handler.send(kill_chat)

            # Broadcast updated stats for attacker and target
            self._broadcast_player_stats(attacker)
            self._broadcast_player_stats(target)

            target_entity_id = target.session.entity_id or target.entity_id

            # DELETE entity with explosion effects
            tick_del = self._get_network_tick(target)
            del_pkt = build_delete_object(tick_del, [target_entity_id], with_effects=True)
            for client in self._snapshot_in_game_clients():
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
            target.angular_vel_yaw = 0.0
            if target.vehicle_physics:
                target.vehicle_physics.reset()

            # Use game loop's delayed spawn mechanism (instead of background thread).
            # The game loop checks delayed_spawn_team every 0.5s and calls
            # _auto_join_team -> _spawn_wf_style when the time arrives.
            respawn_delay = 5.0
            target.session.delayed_spawn_team = target.session.team_id or 1
            target.session.delayed_spawn_time = time.monotonic() + respawn_delay
            print(f"[COMBAT] Respawning c{target.client_id} in {respawn_delay:.0f}s via delayed_spawn")

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
        if not self.jump_jets_enabled:
            return

        # Suppress jump jets for the first 2 seconds after spawn to avoid
        # interfering with the client's spawn state machine.
        if hasattr(ctx.session, 'last_spawn_time') and (time.monotonic() - ctx.session.last_spawn_time) < 2.0:
            return

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
                self._consume_player_energy(ctx, fuel_cost)

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

        # Wall-clock tick pacing: Windows time.sleep(0.033) often sleeps ~15-21ms,
        # causing the tick loop to run at ~47Hz instead of 30Hz.  Use a monotonic
        # accumulator to guarantee exactly tick_rate_hz ticks per wall-clock second.
        next_tick_time = time.monotonic()
        tick_period = 1.0 / self.tick_rate_hz if self.tick_rate_hz > 0 else 0.1
        # Physics steps once per tick at native 30Hz (no accumulator needed).
        ctx.physics_step_count = 0
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

                # Decompile-faithful lockstep behavior: apply current input for the
                # full 30Hz step (no intra-tick split/backdating).
                physics.step_client_substeps(torque, physics_dt, use_f32=use_f32)

                ctx.player_heading = physics.heading
                ctx.angular_vel_yaw = physics.angular_velocity
                ctx.player_yaw = -ctx.player_heading
                ctx.player_pose["yaw"] = -ctx.player_heading

                self._update_player_position(ctx, dt_override=physics_dt, heading_override=old_heading)
                self._update_player_aim(ctx)
                self._regen_player_energy(ctx, physics_dt)

                # Send debug sync state for measuring client-server divergence
                if self.debug_sync:
                    fwd_input = ws.behavior_slots[BehaviorSlot.MOVING_FORWARD]
                    strafe_input = ws.behavior_slots[BehaviorSlot.MOVING_SIDEWAYS]
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

                send_update = True
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
                    weapon_type = self._get_local_state_weapon_type(ctx)
                    ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
                    pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
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
                                    0.0,
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
                        if mode not in ("off", "none", "disabled"):
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
                                            0.0,
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
                                rot=(
                                    ctx.player_pose.get("roll", 0.0),
                                    0.0,
                                    ctx.player_yaw,
                                ),
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
                                rot=(
                                    ctx.player_pose.get("roll", 0.0),
                                    0.0,
                                    ctx.player_yaw,
                                ),
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
                elif self.send_player_updates and not send_full_update and send_update:
                    # Heartbeat path: health/energy delivery (no position correction).

                    weapon_type = self._get_local_state_weapon_type(ctx)
                    ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
                    pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
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
                    use_view = self.heartbeat_view_update
                    hb_rot = None
                    if self.heartbeat_include_rot:
                        hb_rot = (
                            ctx.player_pose.get("roll", 0.0),
                            0.0,
                            ctx.player_heading,  # entity+0x38 convention
                        )
                    hb_pos = None
                    if self.heartbeat_include_pos:
                        hb_pos = self._to_client_pos(ctx.player_pos)
                    payload = build_update_array_heartbeat(
                        tick,
                        ctx.session.entity_id,
                        include_health=include_local_state,
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
                        is_view_update=use_view,
                        rot=hb_rot,
                        pos=hb_pos,
                    )
                    pkt_label = "VIEW_UPDATE_BEAT" if use_view else "UPDATE_ARRAY_BEAT"

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
                            rot=(
                                ctx.player_pose.get("roll", 0.0),
                                0.0,
                                ctx.player_yaw,
                            ),
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
                        "rot": (
                            ctx.player_pose.get("roll", 0.0),
                            0.0,
                            ctx.player_yaw,
                        ),
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

        ctx.tick_thread = None
        print(f"[TICK] Tick loop ended for client {ctx.client_id}")


def main():
    """Entry point."""
    server = WulframServer()
    server.start()


if __name__ == "__main__":
    main()


