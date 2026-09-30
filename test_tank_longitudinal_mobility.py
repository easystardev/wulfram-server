"""T0 governor semantics and real server position-step integration."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from wulfram.tank_mobility import tank_forward_mobility
from wulfram.client import ClientContext
from wulfram.server import WulframServer
from wulfram.session import Session
from wulfram.weapons import EntityType


@pytest.mark.parametrize("velocity,throttle,expected", [
    ((0, 0, 0), 1, 1), ((80, 0, 0), 1, 1),
    ((88, 0, 0), 1, .5), ((96, 0, 0), 1, 0),
    ((120, 0, 0), 1, 0), ((-88, 0, 0), -1, .5),
    ((96, 0, 0), -1, 1), ((-96, 0, 0), 1, 1),
    ((96, 0, 0), 0, 1), ((0, 120, 0), 1, 1),
])
def test_governor_retains_braking_and_ignores_sideways_speed(velocity, throttle, expected):
    scale, _ = tank_forward_mobility(velocity, (1, 0, 0), throttle, 80, 33000, 2000)
    assert scale == pytest.approx(expected)


def test_fuel_is_a_cap_after_speed_and_altitude():
    # Original 0x004f9790 and the accepted native mobility fixture:
    # 0.5 overspeed * 0.5 altitude = .25, min(.25, .4 fuel) = .25.
    scale, _ = tank_forward_mobility((88, 0, 0), (1, 0, 0), 1, 80, 0, 2000, .5)
    assert scale == pytest.approx(.25)
    scale, _ = tank_forward_mobility((0, 0, 0), (1, 0, 0), 1, 80, 1000, 2000)
    assert scale == pytest.approx(.7)


def make_step(enabled, velocity, throttle=1, heading=0, altitude=1, fuel=33000, configure=None):
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30
    server.linear_damp_driving = server.linear_damp_coasting = 0
    server.turn_deadzone = .05
    server.turn_sign = -1
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.tank_suspension_enabled = False
    server.tank_terrain_contact_coupling_enabled = False
    server.tank_spring_attitude_model = "target"
    server.gravity = server.ground_level = 0
    server.world_bound = 100000
    server.jump_jets_enabled = False
    if enabled is not None:
        server.tank_longitudinal_mobility_enabled = enabled
    server._tank_altitude_mobility = lambda ctx: altitude
    server._resolve_entity_world_collision = lambda ctx, px, py, pz, vx, vy, vz: (px, py, pz, vx, vy, vz)
    server._check_building_collisions = lambda ctx, px, py, pz, vx, vy: (px, py, vx, vy)
    ctx = ClientContext(client_id=1, client_addr=("127.0.0.1", 50000),
                        session=Session(), entity_id=0x14EA)
    ctx.entity_type = EntityType.TANK
    ctx.injected_input = (throttle, 0)
    ctx.player_pos = (100, 200, 20)
    ctx.player_vel = velocity
    ctx.player_heading = heading
    ctx.player_fuel = fuel
    ctx.player_energy = 100
    ctx.ground_level_override = None
    ctx.player_pose = {"roll": 0, "pitch": 0, "yaw": heading,
                       "pos": ctx.player_pos, "vel": velocity}
    if configure is not None:
        configure(server, ctx)
    server._update_player_position(ctx, dt_override=1 / 30)
    return ctx


@pytest.mark.parametrize("heading,velocity", [(0, (88, 0, 0)), (math.pi / 2, (0, 88, 0))])
def test_server_projects_velocity_without_requiring_terrain(heading, velocity):
    ctx = make_step(True, velocity, heading=heading)
    debug = ctx.debug_last_controller_step
    assert debug["mobility_model"] == "longitudinal"
    assert debug["forward_speed"] == pytest.approx(88)
    assert debug["forward_mobility"] == pytest.approx(.5)
    assert debug["raw_impulse"][0] == pytest.approx(42.5)
    component = 0 if heading == 0 else 1
    assert ctx.player_vel[component] == pytest.approx(88 + 42.5 / 30)


def test_server_defaults_to_corrected_model_with_explicit_legacy_fallback():
    absent = make_step(None, (88, 0, 0))
    enabled = make_step(True, (88, 0, 0))
    disabled = make_step(False, (88, 0, 0))
    assert absent.player_pos == enabled.player_pos
    assert absent.player_vel == enabled.player_vel
    assert absent.player_vel != disabled.player_vel
    assert absent.debug_last_controller_step["mobility_model"] == "longitudinal"
    assert disabled.debug_last_controller_step["raw_impulse"][0] == 85


def test_server_braking_and_combined_caps():
    braking = make_step(True, (96, 0, 0), throttle=-1)
    assert braking.debug_last_controller_step["raw_impulse"][0] == -85
    capped = make_step(True, (88, 0, 0), altitude=.5, fuel=0)
    assert capped.debug_last_controller_step["forward_mobility"] == pytest.approx(.25)
