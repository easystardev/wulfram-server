"""
Session layer: Per-connection state machine.
Tracks phase transitions: Handshake → Login → TeamSelect → Spawn → InGame
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
import time


class Phase(Enum):
    """Session phases following the Wulfram2 connection lifecycle."""
    DISCONNECTED = auto()
    HANDSHAKE = auto()      # Initial TCP connect, UDP setup
    LOGIN = auto()          # Username/password exchange
    TEAM_SELECT = auto()    # Player choosing team
    SPAWNING = auto()       # Construction timeout, waiting to enter game
    IN_GAME = auto()        # Active gameplay, tick loop running


@dataclass
class Session:
    """
    Per-connection session state.

    This tracks everything about a single client connection:
    - Current protocol phase
    - Player identity and team
    - Entity assignment
    - Tick counter for replication
    """
    # Connection state
    phase: Phase = Phase.DISCONNECTED
    connected_at: float = field(default_factory=time.monotonic)

    # Login state
    username: str = ""
    session_key: str = ""
    login_password_requested: bool = False
    login_game_service_requested: bool = False
    login_complete: bool = False

    # Player state
    player_id: int = 0
    entity_id: int = 0
    team_id: int = 0

    # Game state
    tick: int = 0
    in_game: bool = False
    pending_spawn_team_id: int = 0

    # Packet sequencing
    behavior_sent: bool = False
    translation_sent: bool = False
    roster_sent: bool = False
    world_stats_sent: bool = False
    want_updates_received: bool = False
    want_updates_time: float = 0.0
    want_updates_handled: bool = False

    # Delayed spawn (to let client initialize)
    delayed_spawn_time: float = 0.0
    delayed_spawn_team: int = 0
    want_updates_handled_time: float = 0.0
    suppress_want_updates_payload: bool = False
    input_ready: bool = False
    input_ready_time: float = 0.0

    # UDP tracking
    udp_addr: Optional[tuple] = None
    udp_verified: bool = False
    udp_outgoing_seq: int = 0
    udp_d_handshake_received: bool = False
    translation_ack_received: bool = False
    translation_ack_time: float = 0.0
    last_udp_activity: float = 0.0  # For dead connection detection

    def reset(self):
        """Reset session to initial state."""
        self.phase = Phase.DISCONNECTED
        self.connected_at = time.monotonic()
        self.username = ""
        self.session_key = ""
        self.login_password_requested = False
        self.login_game_service_requested = False
        self.login_complete = False
        self.player_id = 0
        self.entity_id = 0
        self.team_id = 0
        self.tick = 0
        self.in_game = False
        self.pending_spawn_team_id = 0
        self.behavior_sent = False
        self.translation_sent = False
        self.roster_sent = False
        self.world_stats_sent = False
        self.want_updates_received = False
        self.want_updates_time = 0.0
        self.want_updates_handled = False
        self.want_updates_handled_time = 0.0
        self.suppress_want_updates_payload = False
        self.input_ready = False
        self.input_ready_time = 0.0
        self.udp_addr = None
        self.udp_verified = False
        self.udp_outgoing_seq = 0
        self.udp_d_handshake_received = False
        self.translation_ack_received = False
        self.translation_ack_time = 0.0
        self.last_udp_activity = 0.0

    def transition_to(self, new_phase: Phase) -> bool:
        """
        Transition to a new phase.
        Returns True if transition is valid, False otherwise.
        """
        valid_transitions = {
            Phase.DISCONNECTED: [Phase.HANDSHAKE],
            Phase.HANDSHAKE: [Phase.LOGIN, Phase.DISCONNECTED],
            Phase.LOGIN: [Phase.TEAM_SELECT, Phase.DISCONNECTED],
            Phase.TEAM_SELECT: [Phase.SPAWNING, Phase.IN_GAME, Phase.DISCONNECTED],  # Allow direct spawn
            Phase.SPAWNING: [Phase.IN_GAME, Phase.TEAM_SELECT, Phase.DISCONNECTED],
            Phase.IN_GAME: [Phase.TEAM_SELECT, Phase.DISCONNECTED],
        }

        allowed = valid_transitions.get(self.phase, [])
        if new_phase in allowed:
            old_phase = self.phase
            self.phase = new_phase
            print(f"[SESSION] {old_phase.name} -> {new_phase.name}")
            return True
        else:
            print(f"[SESSION] Invalid transition: {self.phase.name} -> {new_phase.name}")
            return False

    def enter_game(self, entity_id: int, team_id: int):
        """Set up session for in-game state."""
        self.entity_id = entity_id
        self.team_id = team_id
        self.in_game = True
        self.tick = 0
        self.transition_to(Phase.IN_GAME)

    def leave_game(self):
        """Clean up game state."""
        self.in_game = False
        self.entity_id = 0
        self.tick = 0


# Feature flags for toggling speculative behavior
@dataclass
class Features:
    """
    Feature toggles for speculative protocol fields.
    Log state at session start for debugging.
    """
    send_load_status: bool = False
    send_behavior_packet: bool = True  # Re-enabled
    send_translation_packet: bool = True  # Required for quantizers
    send_spawn_points: bool = False  # Wulf-forge baseline: no spawn point entities
    send_player_on_login: bool = True
    send_world_stats_on_login: bool = True
    # Default off: auto-spawn can trigger before world collision data is ready and crash the client.
    # Spawn should come from explicit REINCARNATE flow (team switch + spawn point selection).
    auto_join_team: bool = False
    tick_loop_enabled: bool = True  # ENABLED - sends health/energy heartbeat
    send_update_array_empty: bool = False  # DISABLED - heartbeat is preferred
    wulfforge_compat: bool = False  # When True, minimize to wulf-forge behavior

    def set_wulfforge_mode(self, enabled: bool = True):
        """
        Set wulf-forge compatibility mode.

        When enabled, the server behaves like wulf-forge:
        - Sends BEHAVIOR and TRANSLATION during login (required for spawn)
        - Uses simple TankPacket for spawn
        - No tick loop
        """
        self.wulfforge_compat = enabled
        if enabled:
            # Wulf-forge actually DOES send BEHAVIOR and TRANSLATION before spawn
            # See wulf-forge main.py lines 683-686
            self.send_load_status = False
            self.send_behavior_packet = True   # wulf-forge sends this!
            self.send_translation_packet = True  # wulf-forge sends this!
            self.send_spawn_points = False
            self.send_player_on_login = True
            self.send_world_stats_on_login = True
            self.auto_join_team = True
            self.tick_loop_enabled = False  # wulf-forge doesn't have a tick loop
            self.send_update_array_empty = False  # wulf-forge doesn't send UPDATE_ARRAY
            print("[FEATURES] Wulf-forge compatibility mode ENABLED")
        else:
            # Full server mode (our experimental defaults)
            self.send_behavior_packet = True
            self.send_translation_packet = True
            self.tick_loop_enabled = True
            print("[FEATURES] Wulf-forge compatibility mode DISABLED")

    def log_state(self):
        """Log all feature flag states."""
        mode = "WULF-FORGE COMPAT" if self.wulfforge_compat else "FULL"
        print(f"[FEATURES] Current state ({mode}):")
        for field_name, value in self.__dict__.items():
            print(f"  {field_name}: {value}")


# Global feature flags instance
FEATURES = Features()
