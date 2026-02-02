"""
Jump Jets system for Wulfram 2 server.

Implements a custom jump jet mechanic since the original game doesn't have one.
Uses behavior slot 5 (upward thrust / Q/Z keys) with rising edge detection.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Callable


@dataclass
class JumpJetConfig:
    """Configuration for jump jets per vehicle type."""
    impulse: float = 15.0        # Upward velocity impulse (units/second)
    cooldown: float = 3.0        # Seconds before can jump again
    fuel_cost: float = 10.0      # Energy consumed (0 = free)
    max_altitude: float = 50.0   # Maximum height limit
    air_control: float = 0.3     # Movement control while airborne (0-1)


# Per vehicle type configurations
# Vehicle types: 0=Tank, 1=Scout/Medic, 2=Assault, 3=Bomber
JUMP_JET_CONFIGS: Dict[int, JumpJetConfig] = {
    0: JumpJetConfig(impulse=15.0, cooldown=3.0, fuel_cost=10.0),   # Tank - standard
    1: JumpJetConfig(impulse=20.0, cooldown=2.0, fuel_cost=8.0),    # Scout - high, fast
    2: JumpJetConfig(impulse=10.0, cooldown=5.0, fuel_cost=15.0),   # Assault - low, slow
    # Bomber (3+) - no jump jets, they already fly
}


@dataclass
class JumpJetState:
    """Per-player jump jet state."""
    last_jump_time: float = 0.0       # Monotonic time of last jump
    prev_slot5_value: float = 0.0     # For rising edge detection
    is_airborne: bool = False         # Currently in jump
    air_time: float = 0.0             # Time since leaving ground


class JumpJetSystem:
    """
    Handles jump jet mechanics for all players.

    Detects when behavior slot 5 (Q/Z keys / upward thrust) transitions
    from low to high (rising edge) and applies a vertical impulse if
    cooldown has elapsed and altitude allows.
    """

    def __init__(self):
        self.player_states: Dict[int, JumpJetState] = {}
        self.enabled: bool = True

        # Callback when a jump is triggered
        # Signature: on_jump(player_id, impulse, new_velocity_z)
        self.on_jump: Optional[Callable[[int, float, float], None]] = None

        # Debug logging
        self.debug: bool = True

    def get_state(self, player_id: int) -> JumpJetState:
        """Get or create state for a player."""
        if player_id not in self.player_states:
            self.player_states[player_id] = JumpJetState()
        return self.player_states[player_id]

    def process_input(
        self,
        player_id: int,
        slot5_value: float,
        entity_type: int,
        current_pos_z: float,
        current_vel_z: float = 0.0,
        current_energy: float = 100.0
    ) -> Tuple[float, bool]:
        """
        Process input and return (vertical impulse, cooldown_ready).

        Args:
            player_id: Player identifier
            slot5_value: Current value of behavior slot 5 (0.0-1.0)
            entity_type: Vehicle type (0=Tank, 1=Scout, etc.)
            current_pos_z: Current altitude
            current_vel_z: Current vertical velocity
            current_energy: Current energy level

        Returns:
            Tuple of (impulse to apply, whether cooldown is ready)
            Impulse is 0.0 if no jump should occur.
        """
        if not self.enabled:
            return (0.0, True)

        state = self.get_state(player_id)
        config = JUMP_JET_CONFIGS.get(entity_type)

        if config is None:
            # Vehicle type doesn't support jump jets (e.g., Bomber)
            return (0.0, True)

        # Rising edge detection: slot goes from < 0.5 to >= 0.5
        rising_edge = state.prev_slot5_value < 0.5 and slot5_value >= 0.5
        state.prev_slot5_value = slot5_value

        # Check cooldown status (for UI feedback even if not jumping)
        now = time.monotonic()
        cooldown_elapsed = now - state.last_jump_time
        cooldown_ready = cooldown_elapsed >= config.cooldown

        if not rising_edge:
            return (0.0, cooldown_ready)

        # Rising edge detected - check if we can jump
        if self.debug:
            print(f"[JUMP] Rising edge detected for player {player_id}")

        # Cooldown check
        if not cooldown_ready:
            remaining = config.cooldown - cooldown_elapsed
            if self.debug:
                print(f"[JUMP] Cooldown not ready, {remaining:.1f}s remaining")
            return (0.0, False)

        # Altitude check
        if current_pos_z >= config.max_altitude:
            if self.debug:
                print(f"[JUMP] At max altitude ({current_pos_z:.1f} >= {config.max_altitude})")
            return (0.0, True)

        # Energy/fuel check
        if config.fuel_cost > 0 and current_energy < config.fuel_cost:
            if self.debug:
                print(f"[JUMP] Insufficient energy ({current_energy:.1f} < {config.fuel_cost})")
            return (0.0, True)

        # All checks passed - apply jump!
        state.last_jump_time = now
        state.is_airborne = True
        state.air_time = 0.0

        impulse = config.impulse
        new_vel_z = current_vel_z + impulse

        if self.debug:
            print(f"[JUMP] Player {player_id} JUMPED! impulse={impulse}, new_vel_z={new_vel_z:.1f}")

        # Trigger callback
        if self.on_jump:
            self.on_jump(player_id, impulse, new_vel_z)

        return (impulse, True)

    def get_cooldown_remaining(self, player_id: int, entity_type: int) -> float:
        """Get remaining cooldown time for a player."""
        state = self.get_state(player_id)
        config = JUMP_JET_CONFIGS.get(entity_type)

        if config is None:
            return 0.0

        elapsed = time.monotonic() - state.last_jump_time
        remaining = max(0.0, config.cooldown - elapsed)
        return remaining

    def get_fuel_cost(self, entity_type: int) -> float:
        """Get fuel cost for a vehicle type."""
        config = JUMP_JET_CONFIGS.get(entity_type)
        return config.fuel_cost if config else 0.0

    def reset_player(self, player_id: int):
        """Reset state for a player (on death/respawn)."""
        if player_id in self.player_states:
            del self.player_states[player_id]

    def update_airborne_state(self, player_id: int, on_ground: bool, dt: float):
        """
        Update airborne tracking (call each tick).

        Args:
            player_id: Player identifier
            on_ground: Whether player is on ground
            dt: Time delta since last update
        """
        state = self.get_state(player_id)

        if on_ground:
            state.is_airborne = False
            state.air_time = 0.0
        elif state.is_airborne:
            state.air_time += dt
