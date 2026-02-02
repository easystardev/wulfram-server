#!/usr/bin/env python3
"""
Tests for session state machine and phase transitions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from wulfram.session import Phase, Session, Features


def test_initial_state():
    """Session starts in DISCONNECTED phase."""
    session = Session()
    assert session.phase == Phase.DISCONNECTED
    assert session.username == ""
    assert session.player_id == 0
    assert session.entity_id == 0
    assert session.in_game == False
    print("test_initial_state: PASSED")
    return True


def test_valid_transitions():
    """Test all valid phase transitions."""
    session = Session()

    # DISCONNECTED -> HANDSHAKE
    assert session.transition_to(Phase.HANDSHAKE) == True
    assert session.phase == Phase.HANDSHAKE

    # HANDSHAKE -> LOGIN
    assert session.transition_to(Phase.LOGIN) == True
    assert session.phase == Phase.LOGIN

    # LOGIN -> TEAM_SELECT
    assert session.transition_to(Phase.TEAM_SELECT) == True
    assert session.phase == Phase.TEAM_SELECT

    # TEAM_SELECT -> SPAWNING
    assert session.transition_to(Phase.SPAWNING) == True
    assert session.phase == Phase.SPAWNING

    # SPAWNING -> IN_GAME
    assert session.transition_to(Phase.IN_GAME) == True
    assert session.phase == Phase.IN_GAME

    # IN_GAME -> TEAM_SELECT (death/respawn)
    assert session.transition_to(Phase.TEAM_SELECT) == True
    assert session.phase == Phase.TEAM_SELECT

    print("test_valid_transitions: PASSED")
    return True


def test_direct_spawn_transition():
    """TEAM_SELECT can transition directly to IN_GAME (skip SPAWNING)."""
    session = Session()
    session.phase = Phase.TEAM_SELECT

    assert session.transition_to(Phase.IN_GAME) == True
    assert session.phase == Phase.IN_GAME

    print("test_direct_spawn_transition: PASSED")
    return True


def test_invalid_transitions():
    """Invalid transitions should be rejected."""
    session = Session()

    # Can't go from DISCONNECTED to LOGIN
    assert session.transition_to(Phase.LOGIN) == False
    assert session.phase == Phase.DISCONNECTED

    # Can't go from DISCONNECTED to IN_GAME
    assert session.transition_to(Phase.IN_GAME) == False
    assert session.phase == Phase.DISCONNECTED

    session.phase = Phase.HANDSHAKE
    # Can't skip LOGIN
    assert session.transition_to(Phase.TEAM_SELECT) == False
    assert session.phase == Phase.HANDSHAKE

    print("test_invalid_transitions: PASSED")
    return True


def test_disconnect_from_any_phase():
    """Can always transition to DISCONNECTED."""
    for phase in [Phase.HANDSHAKE, Phase.LOGIN, Phase.TEAM_SELECT, Phase.SPAWNING, Phase.IN_GAME]:
        session = Session()
        session.phase = phase
        assert session.transition_to(Phase.DISCONNECTED) == True
        assert session.phase == Phase.DISCONNECTED

    print("test_disconnect_from_any_phase: PASSED")
    return True


def test_enter_game():
    """enter_game() sets entity, team, and transitions to IN_GAME."""
    session = Session()
    session.phase = Phase.TEAM_SELECT

    session.enter_game(entity_id=42, team_id=1)

    assert session.entity_id == 42
    assert session.team_id == 1
    assert session.in_game == True
    assert session.tick == 0
    assert session.phase == Phase.IN_GAME

    print("test_enter_game: PASSED")
    return True


def test_leave_game():
    """leave_game() clears game state."""
    session = Session()
    session.phase = Phase.TEAM_SELECT
    session.enter_game(entity_id=42, team_id=1)

    session.leave_game()

    assert session.in_game == False
    assert session.entity_id == 0
    assert session.tick == 0

    print("test_leave_game: PASSED")
    return True


def test_reset():
    """reset() returns session to initial state."""
    session = Session()
    session.phase = Phase.IN_GAME
    session.username = "testuser"
    session.entity_id = 42
    session.team_id = 1
    session.in_game = True
    session.tick = 100
    session.udp_verified = True

    session.reset()

    assert session.phase == Phase.DISCONNECTED
    assert session.username == ""
    assert session.entity_id == 0
    assert session.team_id == 0
    assert session.in_game == False
    assert session.tick == 0
    assert session.udp_verified == False

    print("test_reset: PASSED")
    return True


def test_features_default():
    """Features have sensible defaults."""
    features = Features()

    assert features.send_behavior_packet == True
    assert features.send_translation_packet == True
    assert features.wulfforge_compat == False

    print("test_features_default: PASSED")
    return True


def test_features_wulfforge_mode():
    """set_wulfforge_mode() configures compatible settings."""
    features = Features()

    features.set_wulfforge_mode(True)

    assert features.wulfforge_compat == True
    assert features.tick_loop_enabled == False
    assert features.send_update_array_empty == False
    assert features.send_behavior_packet == True
    assert features.send_translation_packet == True

    features.set_wulfforge_mode(False)
    assert features.wulfforge_compat == False
    assert features.tick_loop_enabled == True

    print("test_features_wulfforge_mode: PASSED")
    return True


def main():
    print("=" * 60)
    print("Session State Machine Tests")
    print("=" * 60)

    tests = [
        test_initial_state,
        test_valid_transitions,
        test_direct_spawn_transition,
        test_invalid_transitions,
        test_disconnect_from_any_phase,
        test_enter_game,
        test_leave_game,
        test_reset,
        test_features_default,
        test_features_wulfforge_mode,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print()
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {test.__name__}: FAILED - {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
