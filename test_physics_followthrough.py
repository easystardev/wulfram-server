"""Controller state, real position-step wiring and input-window regressions."""
import math
from collections import deque
from types import SimpleNamespace
import pytest
from test_tank_longitudinal_mobility import make_step
from wulfram.tank_controller import jet_physics
from wulfram.input_window import average_controls
from wulfram.server import WulframServer
from wulfram.weapons import WeaponSystem
from wulfram.packets import BEHAVIOR_GRAVITY, build_behavior_packet
from wulfram2_protocol.codec import unpack_fixed16
from wulfram.physics import VehiclePhysics
from wulfram2_protocol.entities import tank_softbody_suspension_force, tank_spring_piecewise_force_sample


@pytest.mark.parametrize("throttle,active,damping,friction", [
    (0, False, 2, .4), (.824, True, 1.5, .1),
    (.01, True, 1.5, .1), (-1, True, 1.5, .1),
])
def test_jet_switch_and_minimum(throttle, active, damping, friction):
    state = jet_physics(throttle, .05, .2, 1.3, .4)
    assert state.active == active
    assert state.damping == pytest.approx(damping)
    assert state.friction == pytest.approx(friction)
    if throttle != 0 and throttle < .05:
        assert state.throttle == pytest.approx(.05)


def test_drag_uses_general_mobility_not_throttle_strength():
    assert jet_physics(.8, .05, .2, 1.3, .4, .5).damping == pytest.approx(.85)
    assert jet_physics(.05, .05, .2, 1.3, .4).damping == pytest.approx(1.5)
    with pytest.raises(ValueError):
        jet_physics(float("nan"), .05, .2, 1.3, .4)


def configure_jet(throttle, record=None):
    def configure(server, ctx):
        server.tank_jet_physics_enabled = True
        ctx.weapon_system = WeaponSystem()
        ctx.weapon_system.behavior_slots[5] = throttle
        if record is not None:
            def collision(ctx, px, py, pz, vx, vy, vz):
                record.append(ctx.tank_runtime_friction)
                return px, py, pz, vx, vy, vz
            server._resolve_entity_world_collision = collision
    return configure


def test_zero_jet_skips_drive_and_uses_wire_gravity_and_drag():
    ctx = make_step(True, (10, 0, 0), configure=configure_jet(0))
    d = ctx.debug_last_controller_step
    assert d["raw_impulse"] == (0, 0)
    assert d["linear_damp"] == 2
    assert d["gravity_impulse"] == (0, 0, -BEHAVIOR_GRAVITY)
    assert ctx.player_vel[0] == pytest.approx(10 - 20 / 30)
    assert ctx.player_vel[2] == pytest.approx(-BEHAVIOR_GRAVITY / 30)


def test_active_jet_friction_reaches_collision_even_while_coasting():
    observed = []
    ctx = make_step(True, (10, 0, 0), throttle=0, configure=configure_jet(.824, observed))
    packet = build_behavior_packet()
    vehicle_section = 1 + 95 + 2340 + 468
    expected = .2 + unpack_fixed16(packet[vehicle_section + 24:vehicle_section + 28])
    assert ctx.debug_last_controller_step["linear_damp"] == pytest.approx(expected)
    assert observed and observed[0] == pytest.approx(.1)


def test_minimum_jet_throttle_matches_actual_behavior_bytes():
    ctx = make_step(True, (0, 0, 0), configure=configure_jet(.001))
    vehicle_section = 1 + 95 + 2340 + 468
    packet = build_behavior_packet()
    expected = unpack_fixed16(packet[vehicle_section + 20:vehicle_section + 24])
    assert ctx.debug_last_controller_step["jet_physics"]["throttle"] == expected


def test_zero_jet_suppresses_yaw_torque():
    def configure(server, ctx):
        configure_jet(0)(server, ctx)
        assert server._compute_turn_torque(ctx, 1) == 0
    make_step(True, (0, 0, 0), configure=configure)


def event(when, fwd, turn=0, strafe=0, **extra):
    return {"time": when, "fwd": fwd, "turn": turn, "strafe": strafe, **extra}


def test_multiple_transitions_and_quick_tap_survive_averaging():
    history = [event(0, 0), event(.01, 1, 1), event(.03, 0, -1)]
    avg = average_controls(history, 0, .04, {})
    assert avg["fwd"] == pytest.approx(.5)
    assert avg["turn"] == pytest.approx(.25)
    assert history[1]["fwd"] == 1


def test_windows_do_not_double_consume_catchup_edges():
    history = [event(0, 0), event(.02, 1), event(.06, 0)]
    left = average_controls(history, 0, .04, {})
    right = average_controls(history, .04, .08, {})
    assert left["fwd"] == pytest.approx(.5)
    assert right["fwd"] == pytest.approx(.5)
    assert average_controls(history, .08, .12, {})["fwd"] == 0


def test_future_packets_and_boundary_order():
    history = [event(.02, 1, previous={"fwd": -.5}),
               event(.02, -1, action_sequence=2), event(.10, 1)]
    assert average_controls(history, 0, .02, {})["fwd"] == -.5
    assert average_controls(history, .02, .04, {})["fwd"] == -1
    with pytest.raises(ValueError):
        average_controls(history, 1, 1, {})


def test_shadow_mode_observes_without_changing_control_and_stale_guard_holds():
    server = WulframServer.__new__(WulframServer)
    server.input_window_mode = "shadow"
    server._remote_og_movement_input_delay_for_ctx = lambda ctx: 0
    ctx = SimpleNamespace(injected_input=None, injected_turn=None,
        movement_input_history=deque([event(0, 0), event(.02, 1)]),
        last_decoded_input={}, last_action_packet_time=1)
    server._prepare_physics_input_window(ctx, 0, .04)
    assert ctx.physics_input_window is None
    assert ctx.debug_input_window["average"]["fwd"] == pytest.approx(.5)
    server.input_window_mode = "apply"
    server._prepare_physics_input_window(ctx, 0, .04)
    assert ctx.physics_input_window["fwd"] == pytest.approx(.5)
    server.input_stale_timeout_s = .01
    server._prepare_physics_input_window(ctx, 0, .04)
    assert ctx.physics_input_window == {"turn": 0, "fwd": 0, "strafe": 0}


def test_small_averaged_input_is_not_discarded_by_post_average_deadzone():
    def configure(server, ctx):
        ctx.injected_input = None
        ctx.weapon_system = WeaponSystem()
        ctx.physics_input_window = {"fwd": .025, "strafe": 0, "turn": 0}
        server.strafe_sign = -1
    ctx = make_step(True, (0, 0, 0), configure=configure)
    assert ctx.debug_last_controller_step["raw_impulse"][0] == pytest.approx(85 * .025)


@pytest.mark.parametrize("fraction", [0, .5, 1, 1.5])
def test_piecewise_suspension_does_not_apply_gravity_fraction_twice(fraction):
    samples = [{"clearance": 2, "point_velocity_z": 0,
                "spring_normal": (0, 0, -1)} for _ in range(4)]
    result = tank_softbody_suspension_force(8 / 3, 0, .8,
        samples=samples, gravity_pct=fraction, physics_timestep_factor=100,
        use_decompile_piecewise_force=True, use_shear_corrections=False)
    expected = tank_spring_piecewise_force_sample(.8 * 3.25, 2, 0, 0,
        gravity_pct=fraction, physics_timestep_factor=100).force_magnitude
    assert result.point_forces == pytest.approx((expected,) * 4)


@pytest.mark.parametrize("dt,expected_steps", [
    (.084, [.042, .042]), (.110, [.04, .015, .04, .015]),
    (.220, [.04, .033, .04, .033, .04, .034]),
    (.550, [.055, .055] * 5),
])
def test_outer_entity_tick_clears_torque_before_reapplying_controller(dt, expected_steps):
    physics = VehiclePhysics(damp_coeff=2)
    calls = []
    physics.step_f32 = lambda torque, duration: calls.append((torque, duration))
    physics.step_client_substeps(4.5, dt)
    assert [t for t, _ in calls] == [4.5] * len(expected_steps)
    assert [d for _, d in calls] == pytest.approx(expected_steps)


def test_long_frame_matches_independent_fresh_force_steps():
    whole = VehiclePhysics(damp_coeff=2)
    separate = VehiclePhysics(damp_coeff=2)
    whole.step_client_substeps(4.5, .110)
    for dt in (.04, .015, .04, .015):
        separate.step_f32(4.5, dt)
    assert whole._angular_velocity == separate._angular_velocity
    assert whole._matrix == separate._matrix


@pytest.mark.parametrize("dt", [.084, .110, .220, .550])
def test_python_prediction_uses_the_same_fresh_force_contract(dt):
    from client.wulfram_client.simulation.physics import LocalPhysics
    server = VehiclePhysics(damp_coeff=2)
    client = LocalPhysics()
    server.step_client_substeps(4.5, dt)
    client._step_heading_substeps(4.5, dt)
    assert client.ang_vel == server.angular_velocity
