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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any

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
    player_yaw: float = 0.0
    player_heading: float = 0.0
    player_energy: float = 100.0

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

    # Input tracking
    last_action_dump_time: float = field(default_factory=time.monotonic)
    last_client_tick: int = 0
    tick_offset: Optional[int] = None
    last_sent_tick: int = 0
    last_sent_player_state: Optional[dict] = None

    # Active projectiles for this client
    active_projectiles: list = field(default_factory=list)
    projectile_lock: threading.Lock = field(default_factory=threading.Lock)

    # Thread management
    tick_thread: Optional[threading.Thread] = None
    ping_thread: Optional[threading.Thread] = None
    ping_stop_event: threading.Event = field(default_factory=threading.Event)
    running: bool = True

    # Viewpoint tracking
    viewpoint_count: int = 0

    def __post_init__(self):
        """Initialize time-based fields after creation."""
        now = time.monotonic()
        self.last_aim_update = now
        self.last_heading_update = now
        self.last_position_update = now
        self.last_action_dump_time = now

    def reset(self):
        """Reset client state for reconnection."""
        self.running = False
        if self.ping_stop_event:
            self.ping_stop_event.set()
        if self.session:
            self.session.reset()

    def update_player_pos(self, pos: tuple):
        """Update player position in both pose dict and legacy field."""
        self.player_pos = pos
        self.player_pose["pos"] = pos

    def update_player_vel(self, vel: tuple):
        """Update player velocity in both pose dict and legacy field."""
        self.player_vel = vel
        self.player_pose["vel"] = vel
