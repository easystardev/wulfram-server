"""
Per-client context for multi-client support.

Each connected client gets its own ClientContext with:
- Session state
- TCP handler
- Player pose tracking
- Weapon/jump jet systems
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any

from .physics import VehiclePhysics

if TYPE_CHECKING:
    from .session import Session
    from .transport import TCPHandler
    from .weapons import WeaponSystem
    from .jump_jets import JumpJetSystem


@dataclass
class ClientContext:
    """
    Per-client state for multi-client server support.

    Each connected client has its own ClientContext instance holding
    all state that was previously global in WulframServer.
    """
    # Client identification
    client_id: int
    client_addr: tuple

    # Session and transport
    session: "Session" = None
    tcp_handler: "TCPHandler" = None

    # Entity assignment
    entity_id: int = 0
    entity_type: int = 0

    # Player pose tracking (position, velocity, rotation)
    player_pose: dict = field(default_factory=lambda: {
        "pos": (100.0, 100.0, 20.0),
        "vel": (0.0, 0.0, 0.0),
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "last_tick": 0,
    })

    # Legacy accessors for compatibility
    player_pos: tuple = (100.0, 100.0, 20.0)
    player_vel: tuple = (0.0, 0.0, 0.0)
    player_speed: float = 0.0
    world_collision_ref_pos: Optional[tuple] = None
    world_collision_bounds_dirty: bool = False
    player_yaw: float = 0.0
    player_heading: float = 0.0
    player_angular_vel: float = 0.0
    # Angular velocity from steering response, integrated in tick loop (rad/s)
    angular_vel_yaw: float = 0.0
    # Spring/softbody pitch-roll angular velocity. OG Spring_apply_forces_to_entity
    # contributes X/Y torque separately from yaw steering.
    spring_body_ang_vel: tuple = (0.0, 0.0)
    # Decompile rigid-body sleep flags: +0xAD is transient should-sleep,
    # +0xAE is persistent sleeping. They are fed into contact probes only.
    rigid_body_should_sleep: bool = False
    rigid_body_sleeping: bool = False
    rigid_body_target_pos: Optional[tuple] = None
    rigid_body_target_rot: Optional[tuple] = None
    rigid_body_interp_tolerance: float = 0.003
    rigid_body_last_interp_tick: int = 0
    player_fuel: float = 33000.0
    player_energy: float = 100.0
    player_health: float = 1.0  # Normalized 0.0-1.0, used for HUD health display

    # Direct-impulse physics for heading (matches client's damped torque system)
    vehicle_physics: Optional[VehiclePhysics] = None

    # Aim tracking
    player_aim_yaw: float = 0.0
    player_aim_pitch: float = 0.0
    player_aim_source: str = "init"
    player_aim_time: float = 0.0
    last_aim_update: float = field(default_factory=time.monotonic)
    last_heading_update: float = field(default_factory=time.monotonic)
    last_position_update: float = field(default_factory=time.monotonic)

    # Weapon system (per-client)
    weapon_system: "WeaponSystem" = None

    # Jump jet system (per-client)
    jump_jet_system: "JumpJetSystem" = None
    jump_prev_thrust_input: float = 0.0
    jump_cooldown_remaining: float = 0.0
    jump_spawn_lockout: float = 2.0

    # Input tracking
    last_action_dump_time: float = field(default_factory=time.monotonic)
    last_client_tick: int = 0
    last_heading_client_tick: int = 0  # Client tick at last heading update
    tick_offset: Optional[int] = None
    tick_offset_smooth: Optional[float] = None  # EMA-smoothed offset for backdating
    action_packet_count: int = 0
    action_update_count: int = 0
    action_dump_count: int = 0
    action_update_decode_fail_count: int = 0
    action_dump_decode_fail_count: int = 0
    last_action_update_decode_fail_hex: str = ""
    last_action_dump_decode_fail_hex: str = ""
    last_action_packet_time: float = 0.0
    last_action_packet_type: str = ""
    last_action_packet_client_tick: int = 0
    nonzero_move_input_count: int = 0
    last_nonzero_move_input_time: float = 0.0
    last_decoded_input: dict = field(default_factory=dict)
    movement_input_history: Any = field(default_factory=lambda: deque(maxlen=180))
    input_feedback_count: int = 0
    last_input_feedback_time: float = 0.0
    state_request_count: int = 0
    last_state_request_time: float = 0.0
    last_state_request_id: int = 0
    last_state_request_frame_count: int = 0
    last_state_request_len: int = 0
    state_sync_reply_count: int = 0
    state_sync_view_reply_count: int = 0
    last_state_sync_reply_time: float = 0.0
    last_state_sync_reply_tick: int = 0
    last_state_sync_replay_timestamp: int = 0
    last_state_sync_snapshot_source: str = ""
    last_state_sync_reason: str = ""
    last_state_sync_update_len: int = 0
    last_state_sync_view_len: int = 0
    last_state_sync_update_has_local_state: bool = False
    last_state_sync_view_has_local_state: bool = False
    last_state_sync_view_timestamp: int = 0
    last_state_sync_update_hex: str = ""
    last_state_sync_view_hex: str = ""
    position_change_count: int = 0
    comm_message_request_count: int = 0
    last_comm_message_request: dict = field(default_factory=dict)
    build_uplink_command_count: int = 0
    last_build_uplink_command: dict = field(default_factory=dict)
    uplink_mvp_bootstrap_sent: bool = False

    last_sent_tick: int = 0
    tick_lock: threading.Lock = field(default_factory=threading.Lock)
    last_sent_player_state: Optional[dict] = None
    ground_level_override: Optional[float] = None

    # Active projectiles for this client
    active_projectiles: list = field(default_factory=list)
    projectile_lock: threading.Lock = field(default_factory=threading.Lock)
    weapon_fire_count: int = 0
    last_weapon_fire_time: float = 0.0
    last_weapon_fire_source: str = ""
    last_weapon_fire_client_tick: int = 0
    last_weapon_fire_projectile_ids: list = field(default_factory=list)
    last_weapon_fire_projectile_types: list = field(default_factory=list)
    last_weapon_fire_energy_spent: float = 0.0
    last_weapon_fire_input: dict = field(default_factory=dict)
    projectile_update_packet_count: int = 0
    last_projectile_update_time: float = 0.0
    last_projectile_update_id: int = 0
    last_projectile_update_targets: int = 0

    # Thread management
    tick_thread: Optional[threading.Thread] = None
    ping_thread: Optional[threading.Thread] = None
    ping_stop_event: threading.Event = field(default_factory=threading.Event)
    running: bool = True

    # Viewpoint tracking
    viewpoint_count: int = 0

    # Multiplayer sync tracking (per receiving client)
    known_entity_ids: set = field(default_factory=set)
    known_roster_ids: set = field(default_factory=set)

    # Health/vitals heartbeat tracking
    last_vitals_send: float = field(default_factory=time.monotonic)
    last_view_update_send: float = field(default_factory=time.monotonic)
    last_state_sync_send: float = field(default_factory=time.monotonic)
    # Update throttling (server-authoritative snapshots)
    last_update_send: float = field(default_factory=time.monotonic)
    # Periodic empirical correction (older local-reconcile probe path)
    last_correction_send: float = 0.0
    force_correction_once: bool = False
    # Burst of authoritative corrections (counts down each tick send).
    # Single VIEW_UPDATE pushes are invisible on the OG client; a short burst
    # at ~10 Hz is visibly applied and the new pose persists after the burst.
    correction_burst_remaining: int = 0
    correction_burst_interval_s: float = 0.0
    # Solo-local-player keepalive — satisfies the OG client's organic
    # STATE_REQUEST trigger at Replication.c:1173-1177 which only fires for
    # `entity_count == 1 && final == local_player` packets. Without a steady
    # drip of this shape, the organic correction loop dies after the spawn
    # window.
    last_solo_local_keepalive: float = 0.0
    # Remote player update throttle
    last_remote_update_send: float = 0.0
    # Combat stats
    kills: int = 0
    deaths: int = 0
    last_damage_time: float = 0.0
    last_damage_source: str = ""
    last_damage_amount: float = 0.0
    last_damage_old_health: float = 0.0
    last_damage_new_health: float = 0.0

    # Pending respawn position (set by respawn command, consumed by auto_join_team)
    pending_respawn_pos: Optional[tuple] = None
    # Server-injected movement input override (fwd, strafe) - persists until cleared
    injected_input: Optional[tuple] = None
    # Server-injected turn input override - persists until cleared
    injected_turn: Optional[float] = None
    # Server-injected upward thrust override - persists until cleared
    injected_thrust: Optional[float] = None
    # Server-injected jumpjet action override - persists until cleared
    injected_jumpjet: Optional[float] = None
    # Previous tick's raw turn input — for detecting input transitions (key press/release)
    prev_raw_turn_input: float = 0.0
    last_sent_pos: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))
    last_sent_vel: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))
    last_sent_yaw: float = 0.0
    last_state_sync_vel: Optional[tuple] = None
    last_state_sync_rot: Optional[tuple] = None
    authoritative_state_history: Any = field(default_factory=lambda: deque(maxlen=180))
    debug_last_controller_step: dict = field(default_factory=dict)
    debug_last_spring_state: dict = field(default_factory=dict)
    debug_last_collision: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize time-based fields after creation."""
        now = time.monotonic()
        self.last_aim_update = now
        self.last_heading_update = now
        self.last_position_update = now
        self.last_action_dump_time = now
        self.last_vitals_send = now
        self.last_view_update_send = now
        self.last_state_sync_send = now
        self.last_update_send = now
        self.last_sent_pos = self.player_pos
        self.last_sent_vel = self.player_vel
        self.last_sent_yaw = self.player_yaw
        if self.world_collision_ref_pos is None:
            self.world_collision_ref_pos = self.player_pos
        if self.vehicle_physics is None:
            self.vehicle_physics = VehiclePhysics()

    def reset(self):
        """Reset client state for reconnection."""
        self.running = False
        if self.ping_stop_event:
            self.ping_stop_event.set()
        if self.session:
            self.session.reset()
        self.known_entity_ids.clear()
        self.known_roster_ids.clear()
        self.authoritative_state_history.clear()

    def update_player_pos(self, pos: tuple):
        """Update player position in both pose dict and legacy field."""
        self.player_pos = pos
        self.player_pose["pos"] = pos
        if self.world_collision_ref_pos is None:
            self.world_collision_ref_pos = pos

    def update_player_vel(self, vel: tuple):
        """Update player velocity in both pose dict and legacy field."""
        self.player_vel = vel
        self.player_pose["vel"] = vel
