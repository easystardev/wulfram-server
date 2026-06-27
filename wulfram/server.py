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
from .server_config import ConfigMixin
from .server_raycast import RaycastMixin, _StaticWorldRayNode
from .server_replication import ReplicationMixin
from .server_spawn import SpawnMixin
from .server_combat import CombatMixin
from .server_remote import RemoteSyncMixin
from .server_corrections import CorrectionMixin
from .server_tick import TickMixin
from .server_net import NetMixin
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



from .packets import (
    PacketType, get_packet_name, get_ticks,
    build_hello_session_key, build_hello_udp_config, build_hello_verified,
    build_ping_reply,
    build_identified_udp, build_login_status, build_tank_packet,
    build_udp_tank_packet_wf, build_update_array_heartbeat,
    build_chat_message, build_add_to_roster, build_remove_from_roster, build_update_stats, build_update_stats_team_first, build_player, build_player_info,
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
from . import handlers, build_uplink, building_lifecycle, config as server_config
from .pktlog import PacketLog

class WulframServer(ConfigMixin, RaycastMixin, ReplicationMixin, SpawnMixin, CombatMixin, RemoteSyncMixin, CorrectionMixin, TickMixin, NetMixin):
    """
    Wulfram2 game server emulator with multi-client support.

    Each client runs in its own thread with its own ClientContext.
    The UDP handler is shared across all clients.
    """

    @staticmethod
    def _projectile_body_pitch_enabled_from_env() -> bool:
        """Return the default-on projectile pitch gate, preserving env opt-out."""
        return os.environ.get("WULFRAM_PROJECTILE_BODY_PITCH", "1") != "0"

    def __init__(self, host: str = None, port: int = 2627):
        server_config.configure_core_server(self, host, port)
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

        # UDP address to client mapping for packet routing
        self.udp_addr_to_client: Dict[tuple, ClientContext] = {}
        # Session key to client mapping for deterministic UDP binding
        self.session_key_to_client: Dict[str, ClientContext] = {}
        self._init_spawn_config()
        self._init_replication_config()
        self._init_map_config()
        # Load building entities for collision detection
        self._building_entities = {}
        self._building_health = {}  # oid -> health (1.0 = full, 0.0 = destroyed)
        self._turret_last_fire = {}  # oid -> monotonic time of last fire
        self._dynamic_building_ids: set[int] = set()
        self._dynamic_building_sources: dict[int, dict[str, Any]] = {}
        self._build_uplink_command_events: list[dict[str, Any]] = []
        self._building_lifecycle_events: list[dict[str, Any]] = []
        self._uplink_ships: dict[int, dict[str, Any]] = {}
        # oid -> monotonic completion time for structures still under construction.
        self._building_construction: dict[int, float] = {}
        # oid -> {done, client_id, entity_type, team_id, slot} for structures being
        # deconstructed (timer); removed + cargo-refunded on completion.
        self._building_deconstruction: dict[int, dict[str, Any]] = {}
        self._last_deconstruction_check: float = 0.0
        # oid -> {pos, cargo_type, team_id} for cargo crates dropped by destroyed
        # buildings (pickup-able like supply-ship cargo). Crate oids from 40000+.
        self._dropped_cargo: dict[int, dict[str, Any]] = {}
        self._dropped_cargo_next_oid: int = 40000
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
        self._init_jump_jet_config()
        self._init_tank_terrain_projection_guard_config()

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
            f"jump_dir={self.jump_jet_direction} "
            f"jump_corr={self.jump_jet_correction_burst_count}@{self.jump_jet_correction_burst_interval:.2f}s "
            f"jump_land_guard={int(self.jump_jet_collision_guard)}:{self.jump_jet_collision_guard_xy:.1f}/"
            f"{self.jump_jet_collision_guard_zpop:.1f}/{self.jump_jet_landing_clearance:.2f} "
            f"tank_proj_guard={int(self.tank_terrain_projection_guard)}:"
            f"{self.tank_terrain_projection_guard_xy:.1f}/{self.tank_terrain_projection_guard_zpop:.1f}/"
            f"{self.tank_terrain_projection_guard_min_clearance:.1f} "
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

        self._init_player_info_local_state_config()

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
        self._init_tick_input_config()
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
            f"projectile_body_pitch={int(self.projectile_body_pitch)} "
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
        # (float32 precision matching is now unconditional: the angular pipeline uses
        #  step_f32 and position uses the shared sim kernel's integrate_verlet — both
        #  always quantize to float32, matching the OG client. The old WULFRAM_F32_PHYSICS
        #  off-path was a non-faithful fallback and has been removed.)
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
        # From RigidBody_integrate_position (Game/Simulation/Physics.c:6032, damped mode):
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

        self._init_correction_config()

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
        self._update_sync_jitter_metric(ctx)

    def _update_sync_jitter_metric(self, ctx: ClientContext) -> None:
        """GOAL 8: track the server's own idle position oscillation amplitude.

        The reactive correction gate's only divergence signal is the terrain-Z
        clamp magnitude, which is BLIND to a hover-spring that overshoots ABOVE
        terrain (the DO idle-Z bounce ran at `divergence_accum=0.000u` while Z swung
        ~2u). Because the client runs deterministic prediction, the server never
        receives the client's true pose; the honest server-measurable proxy for the
        client<->server delta is the server's OWN state oscillation while the input
        is idle — the client renders its smooth local spring while the server bounces.
        We keep a short rolling window of recent Z (and XY) and expose peak-to-peak
        jitter in `pos`/`players`/[STATUS] so sync is actually measurable. Telemetry
        only — it does not feed the gate.
        """
        win = getattr(ctx, "_sync_jitter_window", None)
        if win is None:
            win = []
            ctx._sync_jitter_window = win
        px, py, pz = ctx.player_pos
        win.append((px, py, pz))
        if len(win) > 90:  # ~3s at 30Hz
            del win[:-90]
        zs = [p[2] for p in win]
        ctx.sync_z_jitter = (max(zs) - min(zs)) if len(zs) > 1 else 0.0
        xs = [p[0] for p in win]
        ys = [p[1] for p in win]
        ctx.sync_xy_jitter = max(max(xs) - min(xs), max(ys) - min(ys)) if len(zs) > 1 else 0.0

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

    def _should_send_spawn_player_info(self, ctx: ClientContext) -> bool:
        """Return whether spawn should include canonical PLAYER_INFO for this client."""
        if self.spawn_send_player_info_explicit:
            return self.spawn_send_player_info
        return not handlers._is_loopback_client(ctx)

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

    def _snapshot_logged_in_clients(self):
        """Snapshot of clients past login — roster presence is announced here,
        NOT gated on spawn, so players see who is connected before anyone deploys."""
        return [c for c in self._snapshot_clients() if c.session and c.session.login_complete]

    @staticmethod
    def _is_conn_reset(err) -> bool:
        """True when an error means the peer's TCP socket is definitively dead."""
        if isinstance(err, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return True
        s = str(err)
        return ("10054" in s or "10053" in s or "forcibly closed" in s
                or "aborted by the software" in s or "Broken pipe" in s)

    def _reap_dead_client(self, ctx: ClientContext, reason: str = "") -> None:
        """Mark a client whose TCP socket is dead for teardown.

        Setting running=False makes its game loop exit, which runs the normal
        _handle_client cleanup: broadcast DELETE_OBJECT for its entity (removing
        the ghost tank), drop it from the roster/clients, free its UDP mapping.
        Closing the socket unblocks the loop's recv/getpeername immediately.
        Without this, a crashed/reconnected client lingers as a zombie session
        (endless TCP-send-failed spam, duplicate identities, ghost tanks).
        """
        if not getattr(ctx, "running", False):
            return
        print(f"[SERVER] Client {ctx.client_id}: reaping dead TCP session ({reason})")
        ctx.running = False
        try:
            if ctx.tcp_handler and ctx.tcp_handler.sock:
                try:
                    ctx.tcp_handler.sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                ctx.tcp_handler.sock.close()
        except Exception:
            pass

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
                if self._is_conn_reset(tcp_err):
                    # Socket is dead — reap instead of spamming this every tick.
                    self._reap_dead_client(ctx, reason="tcp_send_reset")
                    return False
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

    def _parse_build_uplink_command(self, text: str) -> dict[str, Any]:
        return build_uplink.parse_command(self, text)

    @staticmethod
    def _build_uplink_entity_type_from_name(name: str) -> Optional[int]:
        return build_uplink.entity_type_from_name(name)

    @staticmethod
    def _building_max_health_for_type(entity_type: int) -> float:
        return build_uplink.building_max_health_for_type(entity_type)

    def _allocate_dynamic_building_oid(self) -> int:
        return build_uplink.allocate_dynamic_building_oid(self)

    def _remember_building_lifecycle_event(self, event: dict[str, Any]) -> dict[str, Any]:
        return building_lifecycle.remember_event(self, event)

    def _building_lifecycle_base_event(self, oid: int, action: str) -> dict[str, Any]:
        return building_lifecycle.base_event(self, oid, action)

    def _broadcast_building_delete(
        self,
        oid: int,
        *,
        prefer_tcp: bool = True,
        participants: tuple[ClientContext, ...] | None = None,
    ) -> int:
        return building_lifecycle.broadcast_delete(
            self,
            oid,
            prefer_tcp=prefer_tcp,
            participants=participants,
        )

    def _remove_dynamic_building_record(self, oid: int) -> None:
        building_lifecycle.remove_dynamic_record(self, oid)

    def _choose_dynamic_building_pos(self, ctx: ClientContext, slot: int) -> tuple[float, float, float]:
        return build_uplink.choose_dynamic_building_pos(self, ctx, slot)

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
        return build_uplink.send_dynamic_entity_definition(
            self,
            target_ctx,
            entity_id=entity_id,
            entity_type=entity_type,
            team_id=team_id,
            pos=pos,
            heading=heading,
            is_static=is_static,
        )

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
        return build_uplink.broadcast_dynamic_entity_definition(
            self,
            entity_id=entity_id,
            entity_type=entity_type,
            team_id=team_id,
            pos=pos,
            heading=heading,
            is_static=is_static,
        )

    def _create_dynamic_building_from_uplink(
        self,
        ctx: ClientContext,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        return build_uplink.create_dynamic_building_from_uplink(self, ctx, command)

    def _delete_dynamic_building_from_uplink(
        self,
        ctx: ClientContext,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        return build_uplink.delete_dynamic_building_from_uplink(self, ctx, command)

    def _get_or_create_uplink_ship(self, ctx: ClientContext, team_id: int) -> dict[str, Any]:
        return build_uplink.get_or_create_uplink_ship(self, ctx, team_id)

    def _build_uplink_ship_info_packet(self, ship: dict[str, Any]) -> bytes:
        return build_uplink.build_uplink_ship_info_packet(ship)

    def _send_uplink_ship_info(self, ctx: ClientContext, ship: dict[str, Any]) -> bool:
        return build_uplink.send_uplink_ship_info(self, ctx, ship)

    def _broadcast_uplink_ship_info(self, ship: dict[str, Any]) -> int:
        return build_uplink.broadcast_uplink_ship_info(self, ship)

    def _broadcast_carrying_info(self, ctx: ClientContext) -> int:
        return build_uplink.broadcast_carrying_info(self, ctx)

    def _update_deconstruction(self) -> None:
        """Per-tick economy update: complete due deconstructions (remove + refund) and
        replenish supply-ship cargo. Guarded to run ~once per tick even though it's
        invoked from the per-client building update."""
        now = time.monotonic()
        if now - float(getattr(self, "_last_deconstruction_check", 0.0) or 0.0) < 0.1:
            return
        self._last_deconstruction_check = now
        if getattr(self, "_building_deconstruction", None):
            build_uplink.update_deconstruction(self)
        build_uplink.replenish_ships(self)

    def _building_under_construction(self, oid: int) -> bool:
        """True while a just-built structure is still constructing (lazy-completes).

        Completion is time-based and resolved on access — when the timer elapses the
        oid is dropped from the under-construction set and the building becomes
        functional (service-providing). No separate per-tick loop needed.
        """
        construction = getattr(self, "_building_construction", None)
        if not construction:
            return False
        done = construction.get(int(oid))
        if done is None:
            return False
        if time.monotonic() >= done:
            del construction[int(oid)]
            print(f"[CONSTRUCTION] building oid={oid} complete")
            return False
        return True

    def _set_player_carry(self, ctx: ClientContext, *, cargo_type: int, cargo_count: int, has_uplink: bool) -> dict:
        return build_uplink.set_player_carry(
            self, ctx, cargo_type=cargo_type, cargo_count=cargo_count, has_uplink=has_uplink
        )

    def _send_existing_build_uplink_entities(self, ctx: ClientContext) -> int:
        return build_uplink.send_existing_build_uplink_entities(self, ctx)

    def _ensure_uplink_mvp_state(self, ctx: ClientContext) -> None:
        build_uplink.ensure_uplink_mvp_state(self, ctx)

    def _relay_player_chat(self, ctx: ClientContext, mode: int, target_id: int, text: str) -> dict:
        """Relay a player chat message to the right recipients as COMM_MESSAGE (0x1F).

        OG chat modes (View/Communications/ChatSystem.c): 4=ALL, 3=TEAM,
        5=PLAYER(whisper), 1=SERVER. Two OG-client facts drive recipients:
          - The client SUPPRESSES its own local echo for ALL/TEAM (it expects the
            server to echo those back), so those recipient sets INCLUDE the sender.
            PLAYER echoes locally, so a whisper goes only to the target.
          - Chat_process_message only auto-resolves the sender NAME for type-5; for
            ALL/TEAM the name field stays null. So we prefix "<name>: " into the
            text (a functional MVP; fuller per-mode coloring via the 0x1F
            sender_mode/channel fields is a follow-up).
        """
        text = (text or "").strip()
        if ctx is None or getattr(ctx, "session", None) is None:
            return {"relayed": 0, "mode": mode, "skipped": "no_ctx"}
        if not text or mode == 1:  # empty or SERVER channel -> not a player relay
            return {"relayed": 0, "mode": mode}
        if len(text) > 240:
            text = text[:240]
        sender_id = ctx.session.player_id or ctx.entity_id
        sender_name = ctx.session.username or f"Player{ctx.client_id}"
        sender_team = ctx.session.team_id

        if mode == 5:  # whisper: target only (sender already echoed locally)
            label = f"[whisper] {sender_name}: {text}"
            recipients = [c for c in self._snapshot_clients()
                          if (c.session.player_id or c.entity_id) == target_id and c is not ctx]
        elif mode == 3:  # TEAM (incl sender)
            label = f"[team] {sender_name}: {text}"
            recipients = [c for c in self._snapshot_clients()
                          if c.session.team_id and c.session.team_id == sender_team]
        else:  # ALL (4) / default: broadcast incl sender
            label = f"{sender_name}: {text}"
            recipients = list(self._snapshot_clients())

        pkt = build_chat_message(label, source_id=sender_id)
        sent = 0
        for c in recipients:
            try:
                self._send_packet_to_client(c, pkt, prefer_tcp=True, allow_udp_fallback=True)
                sent += 1
            except Exception:  # noqa: BLE001
                pass
        print(f"[CHAT] c{ctx.client_id} mode={mode} -> {sent} recipient(s): {label!r}")
        return {"relayed": sent, "mode": mode}

    def _broadcast_kill_feed(self, message: str) -> int:
        """Server-generated kill notification, sent as COMM_MESSAGE (0x1F, system
        source_id=0) to all in-game clients. The OG client has no built-in kill
        feed (DEATH_NOTICE 0x1D only drops the victim's cargo), so kill notices
        ride server chat like the OG. Gated by WULFRAM_KILL_FEED (default on)."""
        if not getattr(self, "kill_feed_enabled", True):
            return 0
        pkt = build_chat_message(message, source_id=0)
        sent = 0
        for client in self._snapshot_in_game_clients():
            try:
                self._send_packet_to_client(client, pkt, prefer_tcp=True, allow_udp_fallback=False)
                sent += 1
            except Exception:  # noqa: BLE001
                pass
        print(f"[KILL-FEED] {message!r} -> {sent} client(s)")
        return sent

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
        malformed_count = 0
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
            try:
                packets = list(self._parse_udp_datagram(data, ctx))
            except (ConnectionResetError, ConnectionAbortedError,
                    BrokenPipeError, OSError):
                continue
            except Exception as parse_err:
                # RESILIENCE BOUNDARY (parse): _udp_loop is the SINGLE shared
                # UDP thread serving every client. A malformed datagram must
                # never crash the parser out of this loop, or UDP dies for ALL
                # clients (a one-packet remote DoS on a public port). Garbage is
                # EXPECTED on a public UDP socket (scans/probes), so log
                # concisely and flood-capped (first few + every 1000th), with NO
                # traceback — an unbounded traceback-per-packet is itself a
                # log-flood DoS the a3 soak watches for.
                malformed_count += 1
                if malformed_count <= 5 or malformed_count % 1000 == 0:
                    cid = ctx.client_id if ctx is not None else "?"
                    print(
                        f"[UDP] dropped malformed datagram #{malformed_count} "
                        f"(client {cid}, len={len(data)}): {parse_err!r}"
                    )
                continue
            if ctx is not None:
                ctx._datagram_active_movement_input = (
                    self._udp_packets_have_active_movement_input(ctx, packets)
                )
            try:
                for packet in packets:
                    try:
                        self._handle_single_udp_packet(ctx, packet, addr)
                    except (ConnectionResetError, ConnectionAbortedError,
                            BrokenPipeError, OSError) as conn_err:
                        # EXPECTED disconnect race: a client closed mid-datagram and
                        # a handler tried to TCP/UDP send to its torn-down socket
                        # (e.g. WinError 10054). Benign — log one concise line, no
                        # traceback, and keep serving every other client.
                        cid = ctx.client_id if ctx is not None else "?"
                        ptype = packet[0] if packet else -1
                        print(
                            f"[UDP] Client {cid}: dropped packet "
                            f"(type=0x{ptype:02X}) after connection-closed: {conn_err!r}"
                        )
                    except Exception as packet_err:
                        # RESILIENCE BOUNDARY: _udp_loop is a SINGLE shared thread
                        # serving every client's UDP. A per-packet handler error must
                        # never propagate out of this loop, or the thread dies and
                        # UDP stops for ALL clients (server-wide gameplay wedge,
                        # observed in the A3 soak 2026-06-01 when a killed client
                        # raced _handle_weapon_demand). An UNEXPECTED error still
                        # gets a full traceback so real bugs stay visible.
                        import traceback
                        cid = ctx.client_id if ctx is not None else "?"
                        ptype = packet[0] if packet else -1
                        print(
                            f"[UDP] Client {cid}: dropped packet "
                            f"(type=0x{ptype:02X}) after handler error: {packet_err!r}"
                        )
                        traceback.print_exc()
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
                except Exception as ping_err:
                    # A failed keepalive ping means the socket is gone; reap so the
                    # session is cleaned up instead of lingering as a zombie.
                    if self._is_conn_reset(ping_err):
                        self._reap_dead_client(ctx, reason="ping_send_reset")
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
        if hasattr(ctx, "record_pose_reset"):
            ctx.record_pose_reset(
                "ghost_rejoin",
                pos=spawn_pos,
                vel=(0.0, 0.0, 0.0),
                details={
                    "team_id": session.team_id,
                    "net_id": entity_id,
                    "unit_type": 0,
                    "explicit_pos": False,
                },
            )
        ctx.world_collision_ref_pos = spawn_pos
        ctx.world_collision_bounds_dirty = False
        ctx.last_state_sync_vel = None
        ctx.last_state_sync_rot = None
        ctx.last_correction_send = 0.0
        ctx.force_correction_once = False
        ctx.last_state_request_burst_queue = 0.0
        # A burst queued against the pre-death pose must not drain against the
        # respawn pose (same discontinuity argument as the accumulators below).
        ctx.correction_burst_remaining = 0
        # Gated-rare correction divergence accumulators (GOAL 2). Reset on
        # spawn so the respawn pose discontinuity doesn't read as divergence.
        ctx.divergence_accum_pos = 0.0
        ctx.divergence_accum_heading = 0.0
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

    def _repair_recent_control_pose_jump(self, ctx: ClientContext, source: str) -> bool:
        """Restore an exact control pose if another path silently rewound it."""
        block = handlers.recent_control_pose_spawn_block(ctx)
        if not block["blocked"]:
            return False
        target = getattr(ctx, "control_pose_reset_pos", None)
        if not target or len(target) != 3:
            return False
        try:
            max_distance = float(os.environ.get("WULFRAM_CONTROL_POSE_REPAIR_DISTANCE", "100.0"))
        except (TypeError, ValueError):
            max_distance = 100.0
        if max_distance <= 0.0:
            return False
        old_pos = ctx.player_pos
        dx = float(old_pos[0]) - float(target[0])
        dy = float(old_pos[1]) - float(target[1])
        dz = float(old_pos[2]) - float(target[2])
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance <= max_distance:
            return False

        repaired_pos = (float(target[0]), float(target[1]), float(target[2]))
        repaired_vel = (0.0, 0.0, 0.0)
        ctx.player_pos = repaired_pos
        ctx.player_vel = repaired_vel
        ctx.player_speed = 0.0
        ctx.player_pose["pos"] = repaired_pos
        ctx.player_pose["vel"] = repaired_vel
        ctx.world_collision_ref_pos = repaired_pos
        ctx.world_collision_bounds_dirty = False
        ctx.last_state_sync_vel = None
        ctx.last_state_sync_rot = None
        ctx.debug_last_control_pose_repair = {
            "source": source,
            "old_pos": [float(old_pos[0]), float(old_pos[1]), float(old_pos[2])],
            "target_pos": list(repaired_pos),
            "distance": distance,
            "threshold": max_distance,
            "control_pose_age_s": block["age_s"],
        }
        if hasattr(ctx, "record_pose_reset"):
            ctx.record_pose_reset(
                "control_pose_jump_repair",
                pos=repaired_pos,
                vel=repaired_vel,
                details=ctx.debug_last_control_pose_repair,
            )
        print(
            f"[CONTROL-POSE] Repaired unstamped pose jump for client {ctx.client_id} "
            f"source={source} distance={distance:.2f} old={old_pos} target={repaired_pos}"
        )
        return True

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

    def build_world_stats_packet(self) -> bytes:
        """Build WORLD_STATS with the current map configuration."""
        return build_world_stats(
            map_name=self.map_name,
            grid_rows=self.map_grid_rows,
            grid_cols=self.map_grid_cols,
            scale=self.map_scale,
        )

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

    def _game_loop(self, ctx: ClientContext):
        """Main game packet loop."""
        # Set socket timeout for delayed spawn checking (recv returns None on timeout)
        ctx.tcp_handler.sock.settimeout(0.5)

        # Track last activity for dead connection detection
        # UDP packets count as activity since client sends TRANSLATION_ACK continuously
        last_activity = time.monotonic()
        while ctx.running and ctx.session.phase in [Phase.TEAM_SELECT, Phase.SPAWNING, Phase.IN_GAME]:
            inactivity_timeout = self._effective_inactivity_timeout(ctx)
            # Announce/refresh roster presence so logged-in players see each
            # other in the scoreboard BEFORE anyone spawns (idempotent — a no-op
            # after the first exchange until a team change invalidates it).
            self._broadcast_roster_presence(ctx)
            # Check for delayed spawn
            if ctx.session.delayed_spawn_team and ctx.session.delayed_spawn_time:
                now = time.monotonic()
                if now >= ctx.session.delayed_spawn_time:
                    if (
                        (ctx.session.in_game or ctx.session.phase == Phase.IN_GAME)
                        and handlers.recent_control_pose_spawn_block(ctx, now=now)["blocked"]
                    ):
                        block = handlers.recent_control_pose_spawn_block(ctx, now=now)
                        print(
                            f"[GAME] Client {ctx.client_id}: Clearing delayed spawn for team "
                            f"{ctx.session.delayed_spawn_team} after recent control pose reset "
                            f"(age={block['age_s']:.2f}s < block={block['block_s']:.2f}s)"
                        )
                        ctx.session.delayed_spawn_team = 0
                        ctx.session.delayed_spawn_time = 0
                        continue
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
                except OSError:
                    break  # Socket disconnected (getpeername raises OSError on a
                    # closed/invalid socket). Narrowed from a bare `except:` so a
                    # KeyboardInterrupt/SystemExit during shutdown isn't swallowed.

            if len(packet) < 1:
                continue

            pkt_type = packet[0]
            print_packet("RECV", pkt_type, packet)

            if pkt_type == PacketType.BPS:
                self._handle_bps(ctx, packet)
            elif pkt_type == PacketType.LOGIN_REQUEST:
                # A late LOGIN_REQUEST after team select is a re-login. Route it
                # through the handler so a changed username is HONORED (behavior
                # (c)) instead of dropped; the handler's login_complete branch
                # re-acks with LOGIN_STATUS(8) and applies any new handle.
                self._handle_login_request(ctx, packet)
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
            # Structures under construction or being deconstructed give no service
            if self._building_under_construction(oid) or oid in (getattr(self, "_building_deconstruction", None) or {}):
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

        self._try_cargo_pickup(ctx)
        self._update_deconstruction()

    def _try_cargo_pickup(self, ctx: ClientContext) -> bool:
        """Pick up a cargo box when near the team supply ship and not carrying.

        Construction-economy pickup: drive within `cargo_pickup_range` of the team
        supply ship while empty-handed and you grab a cargo box (default_cargo_type),
        which broadcasts CARRYING_INFO. Build consumes it (see build_require_cargo).
        """
        if int(getattr(ctx, "cargo_type", 0) or 0) != 0 or getattr(ctx, "has_uplink", False):
            return False  # hands full
        session = getattr(ctx, "session", None)
        if session is None or not getattr(session, "in_game", False):
            return False
        try:
            px, py = float(ctx.player_pos[0]), float(ctx.player_pos[1])
        except (TypeError, KeyError, IndexError):
            return False
        rng = float(getattr(self, "cargo_pickup_range", 30.0))
        rng_sq = rng * rng
        # Dropped-crate pickup (cargo from destroyed buildings) -- takes precedence.
        for oid, crate in list((getattr(self, "_dropped_cargo", None) or {}).items()):
            try:
                cx, cy = float(crate["pos"][0]), float(crate["pos"][1])
            except (TypeError, KeyError, IndexError):
                continue
            if (px - cx) ** 2 + (py - cy) ** 2 <= rng_sq:
                ctype = int(crate.get("cargo_type", 0) or 0)
                if ctype == 0:
                    continue
                self._set_player_carry(ctx, cargo_type=ctype, cargo_count=1, has_uplink=False)
                self._broadcast_building_delete(int(oid), prefer_tcp=False)
                self._dropped_cargo.pop(oid, None)
                print(f"[CARGO] Client {ctx.client_id} picked up dropped crate oid={oid} type={ctype}")
                return True
        # Supply-ship pickup.
        ship = self._uplink_ships.get(int(getattr(session, "team_id", 0) or 0))
        if ship is None:
            return False
        try:
            sx, sy = float(ship["pos"][0]), float(ship["pos"][1])
        except (TypeError, KeyError, IndexError):
            return False
        if (px - sx) ** 2 + (py - sy) ** 2 > rng_sq:
            return False
        available = int(ship.get("cargo_available", 0) or 0)
        if available <= 0:
            return False  # ship out of cargo (replenishing)
        cargo_type = int(getattr(self, "default_cargo_type", 0) or 0)
        if cargo_type == 0:
            return False
        self._set_player_carry(ctx, cargo_type=cargo_type, cargo_count=1, has_uplink=False)
        build_uplink.ship_set_cargo_available(self, ship, available - 1)
        if float(ship.get("next_replenish", 0.0) or 0.0) <= 0.0:
            ship["next_replenish"] = time.monotonic() + float(getattr(self, "ship_replenish_s", 0.0) or 0.0)
        print(f"[CARGO] Client {ctx.client_id} picked up cargo type={cargo_type} from supply ship "
              f"{ship.get('oid')} ({available - 1} left)")
        return True

    def _drop_cargo_crate(self, pos, cargo_type: int, team_id: int) -> int:
        """Drop a pickup-able cargo crate (CARGO_BOX) at pos -- e.g. from a destroyed
        building. Returns the crate oid (0 if not dropped)."""
        ctype = int(cargo_type or 0)
        if ctype <= 0 or pos is None:
            return 0
        oid = int(self._dropped_cargo_next_oid)
        self._dropped_cargo_next_oid = oid + 1
        cpos = tuple(float(v) for v in pos)
        self._dropped_cargo[oid] = {"pos": cpos, "cargo_type": ctype, "team_id": int(team_id or 0)}
        try:
            self._broadcast_dynamic_entity_definition(
                entity_id=oid, entity_type=int(EntityType.CARGO_BOX), team_id=int(team_id or 0),
                pos=cpos, heading=0.0, is_static=True)
        except Exception:  # noqa: BLE001 - drop tracking must never break the destroy path
            pass
        print(f"[CARGO] dropped crate oid={oid} type={ctype} at ({cpos[0]:.0f},{cpos[1]:.0f})")
        return oid

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
        if self.update_local_state_mode == "wf" and handlers._is_og_client(ctx):
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
        server_config.load_env_file(overwrite=True)

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

    @staticmethod
    def _advance_tick_pacer(
        next_tick_time: float,
        tick_period: float,
        *,
        now: float,
        max_catchup_steps: int = 5,
    ) -> tuple[float, float]:
        """Advance the wall-clock pacer while preserving a capped physics backlog."""

        next_tick_time += tick_period
        sleep_dt = next_tick_time - now
        if tick_period <= 0.0:
            return next_tick_time, 0.0

        catchup_steps = max(1, int(max_catchup_steps))
        max_backlog = tick_period * float(catchup_steps)
        if sleep_dt < -max_backlog:
            # OG caps large elapsed physics deltas to a small substep batch.
            next_tick_time = now - tick_period * float(catchup_steps - 1)
            sleep_dt = next_tick_time - now
        return next_tick_time, max(0.0, sleep_dt)

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
        # Default-off per-step phase timing probe (physics vs network-send split).
        # Used to localize rough-terrain controller-cadence collapse. Never on by default.
        _phase_timing = os.environ.get("WULFRAM_TICK_PHASE_TIMING", "0") == "1"
        _phase_timing_threshold_ms = float(
            os.environ.get("WULFRAM_TICK_PHASE_TIMING_MS", "40") or 40.0
        )
        _phase_timing_path = os.environ.get(
            "WULFRAM_TICK_PHASE_TIMING_LOG",
            r"C:\Users\wstri\dev\wolfram\tick_phase_timing.log",
        )
        _phase_t0 = 0.0
        _phase_t_upp = 0.0
        _phase_t_phys = 0.0
        # NOTE: use perf_counter (QPC, sub-us) NOT monotonic (GetTickCount64,
        # 15.6 ms resolution on Windows) so the split is not clock-quantized.
        # Physics steps once per tick at native 30Hz (no accumulator needed).
        ctx.physics_step_count = 0
        last_physics_wall_time = time.monotonic()
        # Real-time physics-rate accumulator (perf_counter = sub-us; monotonic is
        # 16ms-coarse on Windows). See GOAL 6: the tick loop free-runs faster than
        # tick_rate_hz under load, so stepping a fixed 1/30 dt every raw tick made
        # the server integrate ~1.05 sim-seconds per real second and over-rotate
        # every turn. The accumulator runs 0/1/2 fixed steps per tick so simulated
        # time tracks wall-clock, matching the client's LocalPhysics.step.
        phys_accum_last = time.perf_counter()
        phys_accumulator = 0.0
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
                # Advance physics at EXACTLY real-time via a wall-clock accumulator
                # stepping fixed 1/tick_rate increments (GOAL 6). The tick loop free-
                # runs faster than tick_rate_hz under load (measured 31.5Hz vs the
                # client's 30Hz); stepping a fixed dt every raw tick over-integrated
                # ~1.05 sim-seconds per real second, so the server out-rotated the
                # client on every turn and compounded a multi-degree heading drift.
                # Run 0/1/2 fixed steps per tick so simulated time == wall-clock time,
                # matching the client's LocalPhysics.step accumulator on both sides.
                physics_dt = 1.0 / self.tick_rate_hz

                # === GOAL 7: per-client frame-rate-matched physics stepping ===
                # The client integrates physics once per RENDER FRAME, subdividing the
                # real frame delta in GUESS6_GameSim_substep_update (azurefishy-src
                # Physics.c:1974): one outer pass of substep_count = delta/110ms + 1
                # (capped 5), each inner-split at 40ms (or dt*0.5 for dt>80ms), with
                # ang_vel += accel*substep_dt (Physics.c:5169/5264). The server must
                # advance simulated time in chunks of THIS client's frame_dt so the
                # outer/inner substep STRUCTURE and f32-quantization boundaries match
                # the client's per-frame integration exactly — not merely the total
                # simulated time. At ~84ms (OG WARP) vs ~33ms (py @30Hz) the resulting
                # rotation differs only ~1.3% (the coarser frame rotates slightly MORE
                # under explicit Euler): the premise's "2.5x over-rotation from step
                # count" does NOT exist because torque AND heading are both dt-scaled.
                # This change aligns that ~1.3% residual to the real per-frame client
                # behavior. GOAL-6's real-time accumulator is preserved (sim-time ==
                # wall-time); only the per-step chunk changes from a fixed 1/30 to the
                # client's actual frame rate. WULFRAM_GOAL7_LEGACY=1 restores 1/30.
                goal7_legacy = os.environ.get("WULFRAM_GOAL7_LEGACY") == "1"
                if goal7_legacy:
                    step_dt = physics_dt
                else:
                    step_dt = ctx.weapon_system.effective_frame_dt(self.tick_rate_hz)
                    # Clamp to the client's own bounds: a ~120fps floor and the 550ms
                    # outer-delta clamp GameSim_substep_update enforces (Physics.c:1974).
                    step_dt = max(1.0 / 120.0, min(step_dt, 0.55))

                _pc_now = time.perf_counter()
                phys_accumulator += max(0.0, _pc_now - phys_accum_last)
                phys_accum_last = _pc_now
                # Cap backlog at the client's 0.55s elapsed clamp (<=5 catch-up steps).
                _max_backlog = 5.0 * step_dt
                if phys_accumulator > _max_backlog:
                    phys_accumulator = _max_backlog
                n_phys_steps = int(phys_accumulator / step_dt)
                phys_accumulator -= n_phys_steps * step_dt
                if os.environ.get("WULFRAM_GOAL6_LEGACY") == "1":
                    # A/B baseline: legacy fixed-dt one-step-per-raw-tick behavior.
                    n_phys_steps = 1
                    phys_accumulator = 0.0
                    step_dt = physics_dt

                if _phase_timing:
                    _phase_t0 = time.perf_counter()

                # Split window uses monotonic (the clock turn_input_change_time is
                # stamped with); it is a coarse sub-tick refinement, kept decoupled
                # from the perf_counter rate accumulator above.
                step_wall_now = time.monotonic()
                step_wall_dt = max(1e-6, step_wall_now - last_physics_wall_time)
                last_physics_wall_time = step_wall_now
                _window_start = step_wall_now - step_wall_dt

                old_heading = ctx.player_heading
                move_dt = 0.0
                move_heading = old_heading
                for _phys_i in range(n_phys_steps):
                    ctx.physics_step_count += 1
                    old_heading = ctx.player_heading

                    # Live ACTION_UPDATE packets arrive asynchronously relative to the
                    # tick loop. If turning changed partway through this tick's wall
                    # window, split the first sub-step so the pre-change slice uses the
                    # previous turn input and the remainder uses the latest input.
                    transition_time = float(getattr(ws, "turn_input_change_time", 0.0) or 0.0)
                    prev_turn_slot = float(getattr(ws, "turn_input_prev_value", 0.0) or 0.0)
                    prev_turn_input = self._normalize_turn_input_value(ctx, prev_turn_slot)
                    split_turn_step = (
                        _phys_i == 0 and
                        transition_time > _window_start and
                        transition_time < step_wall_now and
                        abs(prev_turn_input - raw_input) > 0.001
                    )
                    move_dt = step_dt
                    move_heading = old_heading
                    if split_turn_step:
                        pre_ratio = (transition_time - _window_start) / step_wall_dt
                        pre_ratio = max(0.0, min(1.0, pre_ratio))
                        pre_dt = step_dt * pre_ratio
                        post_dt = step_dt - pre_dt
                        prev_torque = self._compute_turn_torque(ctx, prev_turn_input)
                        if pre_dt > 1e-6:
                            physics.step_client_substeps(prev_torque, pre_dt)
                            self._sync_heading_physics_to_context(ctx, physics)
                            self._update_player_position_stepped(ctx, pre_dt, heading_override=old_heading)
                            move_heading = ctx.player_heading
                        if post_dt > 1e-6:
                            physics.step_client_substeps(torque, post_dt)
                        move_dt = post_dt
                        ws.turn_input_change_time = 0.0
                        if self.debug_sync:
                            print(
                                f"[YAW-SPLIT] c{ctx.client_id} "
                                f"old={prev_turn_input:.3f} new={raw_input:.3f} "
                                f"pre_dt={pre_dt * 1000.0:.1f}ms post_dt={post_dt * 1000.0:.1f}ms"
                            )
                    else:
                        physics.step_client_substeps(torque, step_dt)

                    self._sync_heading_physics_to_context(ctx, physics)
                    if move_dt > 1e-6:
                        self._update_player_position_stepped(ctx, move_dt, heading_override=move_heading)
                if _phase_timing:
                    _phase_t_upp = time.perf_counter()
                self._resolve_entity_entity_collisions(ctx)
                self._update_player_aim(ctx)
                self._regen_player_energy(ctx, physics_dt)
                self._update_supply_buildings(ctx, physics_dt)
                if os.environ.get("WULFRAM_TURRET_AI", "1") == "1":
                    self._update_turret_ai()

                if _phase_timing:
                    _phase_t_phys = time.perf_counter()

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
                    print(
                        f"[STATUS] Client {ctx.client_id}: pos={ctx.player_pos} "
                        f"input={input_status}(fwd={fwd_input:.2f},strafe={strafe_input:.2f}) "
                        f"stuck={stuck_duration:.0f}s "
                        f"corrections={int(getattr(ctx, 'correction_send_count', 0) or 0)} "
                        f"divergence_accum={float(getattr(ctx, 'divergence_accum_pos', 0.0) or 0.0):.3f}u "
                        f"z_jitter={float(getattr(ctx, 'sync_z_jitter', 0.0) or 0.0):.3f}u "
                        f"xy_jitter={float(getattr(ctx, 'sync_xy_jitter', 0.0) or 0.0):.3f}u"
                    )

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
                correction_reason = ""
                force_due = bool(getattr(ctx, "force_correction_once", False))
                if self.correction_gate_enabled:
                    # OG-faithful GATED-RARE reactive correction (GOAL 2, amended
                    # 2026-06-09). No proactive timer streams: a correction emits
                    # only when (a) explicitly forced, (b) a queued reactive burst
                    # is draining (operator `correction now`, jump-jet hop, or the
                    # rate-capped STATE_REQUEST settle burst — see
                    # _maybe_queue_state_request_burst; queue sites are all
                    # event-driven and rate-capped), or (c) the authoritative
                    # state has genuinely DIVERGED (accumulated server-only,
                    # client-unpredictable displacement — terrain-Z clamp /
                    # collision push — since the last correction) beyond a
                    # threshold, and never faster than the hard rate cap. The gate
                    # is IDENTICAL for every client: a zero-latency loopback client
                    # is bounded by the same caps, with no _is_loopback_client fork.
                    accum_pos = float(getattr(ctx, "divergence_accum_pos", 0.0) or 0.0)
                    accum_heading_deg = math.degrees(
                        abs(float(getattr(ctx, "divergence_accum_heading", 0.0) or 0.0))
                    )
                    rate_ok = (now - ctx.last_correction_send) >= self.correction_min_interval
                    diverged = (
                        accum_pos >= self.correction_divergence_pos
                        or accum_heading_deg >= self.correction_divergence_heading_deg
                    )
                    divergence_correction_due = diverged and rate_ok
                    burst_due = self._correction_burst_due(
                        ctx, now, active_movement_correction_suppressed
                    )
                    # Periodic correction: fires on a steady cadence even DURING
                    # movement (the divergence accumulator is blind to the OG's
                    # lateral prediction gap, and burst/divergence are paused while
                    # driving — so without this, drift only clears on a jumpjet
                    # force-correction). Opt-in via WULFRAM_PERIODIC_CORRECTION_S.
                    periodic_due = (
                        self.periodic_correction_interval > 0.0
                        and not handlers._is_loopback_client(ctx)
                        and (now - ctx.last_correction_send) >= self.periodic_correction_interval
                    )
                    if force_due:
                        correction_reason = "forced"
                    elif burst_due:
                        correction_reason = "burst"
                    elif divergence_correction_due:
                        correction_reason = "divergence"
                    elif periodic_due:
                        correction_reason = "periodic"
                    correction_due = (
                        force_due or burst_due or divergence_correction_due or periodic_due
                    )
                else:
                    # Legacy proactive streams — A/B only (WULFRAM_CORRECTION_GATE=0).
                    burst_due = self._correction_burst_due(
                        ctx, now, active_movement_correction_suppressed
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
                    if force_due:
                        correction_reason = "forced"
                    elif burst_due:
                        correction_reason = "burst"
                    elif movement_correction_due:
                        correction_reason = "movement"
                    elif interval_correction_due:
                        correction_reason = "interval"
                    correction_due = (
                        force_due
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
                        # Gate consumed the divergence: clear the accumulators so a
                        # single correction settles the client before the next can fire.
                        ctx.divergence_accum_pos = 0.0
                        ctx.divergence_accum_heading = 0.0
                        ctx.correction_send_count = int(getattr(ctx, "correction_send_count", 0) or 0) + 1
                        # Decrement against the LIVE value, not the tick-entry
                        # snapshot: the UDP thread may have re-queued a fresh
                        # burst between snapshot and here, and a stale-snapshot
                        # write-back would silently cancel it.
                        live_burst_remaining = int(getattr(ctx, "correction_burst_remaining", 0) or 0)
                        if live_burst_remaining > 0:
                            ctx.correction_burst_remaining = live_burst_remaining - 1
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

                if _phase_timing:
                    _phase_t_end = time.perf_counter()
                    _iter_ms = (_phase_t_end - _phase_t0) * 1000.0
                    if _iter_ms >= _phase_timing_threshold_ms:
                        _upp_ms = (_phase_t_upp - _phase_t0) * 1000.0
                        _rest_phys_ms = (_phase_t_phys - _phase_t_upp) * 1000.0
                        _send_ms = (_phase_t_end - _phase_t_phys) * 1000.0
                        _coll = getattr(ctx, "_upp_collision_ms", 0.0)
                        _bld = getattr(ctx, "_upp_building_ms", 0.0)
                        _att = getattr(ctx, "_upp_attitude_ms", 0.0)
                        try:
                            with open(_phase_timing_path, "a") as _ptf:
                                _ptf.write(
                                    f"tick={ctx.session.tick} step={ctx.physics_step_count} "
                                    f"iter_ms={_iter_ms:.2f} upp_ms={_upp_ms:.2f} "
                                    f"collision_ms={_coll:.2f} building_ms={_bld:.2f} "
                                    f"attitude_ms={_att:.2f} "
                                    f"rest_phys_ms={_rest_phys_ms:.2f} send_ms={_send_ms:.2f} "
                                    f"pos=({ctx.player_pos[0]:.1f},{ctx.player_pos[1]:.1f},"
                                    f"{ctx.player_pos[2]:.1f})\n"
                                )
                        except Exception:
                            pass

                # Wall-clock pacing: preserve a capped fixed-step backlog when late.
                next_tick_time, sleep_dt = self._advance_tick_pacer(
                    next_tick_time,
                    tick_period,
                    now=time.monotonic(),
                    max_catchup_steps=5,
                )
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

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
    # Latency: the per-step terrain collision allocates millions of short-lived tuples
    # (CBSP vector math), so generational GC sweeps of the large, permanent startup heap
    # (loaded terrain/collision meshes) cause intermittent multi-10ms pauses on the
    # controller tick. Freeze the startup heap into the permanent generation so GC no
    # longer rescans it, and relax the gen-0 threshold so collections are rarer. Pure
    # latency tuning — no behaviour/parity change. Disable via WULFRAM_GC_TUNE=0.
    if os.environ.get("WULFRAM_GC_TUNE", "1") != "0":
        try:
            import gc

            gc.collect()
            gc.freeze()
            gc.set_threshold(50000, 500, 500)
            print("[GC] froze startup heap, gen0 threshold=50000 (latency tuning)")
        except Exception as exc:
            print(f"[GC] tuning failed: {exc}")
    server.start()


if __name__ == "__main__":
    main()



