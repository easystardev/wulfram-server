from types import SimpleNamespace

from wulfram.native_live import NativeLiveWorld


def test_native_jump_bridge_turns_shared_result_into_external_velocity():
    calls = []

    class Server:
        @staticmethod
        def _terrain_physics_ground_z_at(x, y):
            assert (x, y) == (10.0, 20.0)
            return 12.0

        @staticmethod
        def _get_jumpjet_input(ctx):
            return 1.0

        @staticmethod
        def _jump_jet_direction_vector(ctx, *, vertical_idx):
            assert vertical_idx == 2
            return (0.1, 0.2, 0.97)

        @staticmethod
        def _apply_jump_jets_fixed_step(ctx, **values):
            calls.append(values)
            ctx.jump_cooldown_remaining = 3.0
            ctx.jump_spawn_lockout = 0.0
            return True, 112.5, (11.25, 22.5, 109.125)

    service = NativeLiveWorld.__new__(NativeLiveWorld)
    service.server = Server()
    service.step_ms = 20
    ctx = SimpleNamespace(
        player_pos=(10.0, 20.0, 17.0),
        player_vel=(1.0, 2.0, 3.0),
        jump_cooldown_remaining=0.0,
        jump_spawn_lockout=0.0,
    )

    assert service._advance_jump_jet(ctx)
    assert ctx.player_vel == (12.25, 24.5, 112.125)
    assert calls == [{
        "dt": 0.02,
        "jumpjet_input": 1.0,
        "current_altitude": 5.0,
        "current_vel_up": 3.0,
        "direction": (0.1, 0.2, 0.97),
        "vertical_idx": 2,
    }]
    assert ctx.native_jump_jet_step["fired"] is True


def test_native_jump_bridge_is_optional_for_minimal_test_servers():
    service = NativeLiveWorld.__new__(NativeLiveWorld)
    service.server = SimpleNamespace()
    service.step_ms = 20
    assert service._advance_jump_jet(SimpleNamespace()) is False
