"""Scout (medic) movement config matches the original wulfram2.exe (legacy).

Evidence: VehiclePhysics_init_medic @ 0x004f6690 (9-slot block
turn / move_forward / move_backward / strafe / max_velocity / low_fuel /
max_altitude / max_speed_height_pickup / gravity_perc), MedicVehicle
apply_physics (reverse with move_backward, cap move_forward * 1.25) and
GUESS5_EntityParams_create (scout 0.15 / 1.1 / max_fuel 13000).
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from wulfram2_protocol.entities import (
    VEHICLE_PHYSICS_CONFIGS,
    EntityType,
    vehicle_max_fuel,
)
from wulfram.client import ClientContext
from wulfram.packets import build_behavior_packet
from wulfram.server import WulframServer
from wulfram.session import Session


def test_scout_config_is_original_medic_defaults():
    scout = VEHICLE_PHYSICS_CONFIGS[EntityType.SCOUT]
    assert scout.turn_adjust == 4.5
    assert scout.move_adjust == 85.0
    assert scout.move_backward_adjust == 38.0
    assert scout.strafe_adjust == 72.0
    assert scout.max_velocity == 85.0
    assert scout.low_fuel_level == 2000.0
    assert scout.max_altitude == 4.9
    assert scout.gravity_pct == 1.0
    assert scout.max_fuel == 13000.0
    assert scout.move_cap_scale == 1.25


def test_tank_config_unchanged():
    tank = VEHICLE_PHYSICS_CONFIGS[EntityType.TANK]
    assert (tank.turn_adjust, tank.move_adjust, tank.strafe_adjust, tank.max_velocity,
            tank.low_fuel_level, tank.max_altitude, tank.gravity_pct, tank.max_fuel) == (
        4.5, 85.0, 69.7, 80.0, 2000.0, 3.25, 1.0, 33000.0)
    assert tank.move_backward_adjust is None
    assert tank.move_cap_scale == 1.0


def test_per_type_max_fuel():
    assert vehicle_max_fuel(EntityType.TANK) == 33000.0
    assert vehicle_max_fuel(EntityType.SCOUT) == 13000.0
    assert vehicle_max_fuel(999) == 33000.0
    assert ClientContext(client_id=1, client_addr=("127.0.0.1", 1)).player_fuel == 33000.0


def _parse_behavior():
    try:
        from wulfram_client.network.behavior import parse_behavior
    except ImportError:  # pragma: no cover - client not on PYTHONPATH
        pytest.skip("wulfram_client not importable")
    return parse_behavior(build_behavior_packet())


def test_behavior_section4_scout_fuel_and_jet_params():
    cfg = _parse_behavior()
    tank, scout = cfg.vehicle_physics
    assert tank.type_id2 == 33000
    assert tank.turn_rate == pytest.approx(0.05, abs=1e-4)
    assert tank.suspension_dampening == pytest.approx(1.3, abs=1e-4)
    assert scout.type_id2 == 13000
    assert scout.turn_rate == pytest.approx(0.15, abs=1e-4)
    assert scout.suspension_dampening == pytest.approx(1.1, abs=1e-4)
    assert scout.type_id1 == 0


def test_behavior_section6_scout_nine_slot_layout():
    cfg = _parse_behavior()
    tank, scout = cfg.active_vehicle_physics[0], cfg.active_vehicle_physics[1]
    assert len(scout.values) == 9
    assert scout.is_medic_layout
    assert scout.turn_adjust == pytest.approx(4.5)
    assert scout.move_adjust == pytest.approx(85.0)
    assert scout.move_backward_adjust == pytest.approx(38.0)
    assert scout.strafe_adjust == pytest.approx(72.0)
    assert scout.max_velocity == pytest.approx(85.0)
    assert scout.low_fuel_level == pytest.approx(2000.0)
    assert scout.max_altitude == pytest.approx(4.9, abs=1e-4)
    assert scout.max_speed_height_pickup == pytest.approx(3.5)
    assert scout.gravity_pct == pytest.approx(1.0)
    assert scout.move_cap_scale == 1.25
    # Tank block keeps the 7-slot tank meanings.
    assert not tank.is_medic_layout
    assert tank.move_backward_adjust is None
    assert tank.strafe_adjust == pytest.approx(69.7, abs=1e-4)
    assert tank.max_velocity == pytest.approx(80.0)
    assert tank.low_fuel_level == pytest.approx(2000.0)
    assert tank.gravity_pct == pytest.approx(1.0)
    assert tank.move_cap_scale == 1.0


def make_step(entity_type, throttle, strafe):
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
    server._tank_altitude_mobility = lambda ctx: 1
    server._resolve_entity_world_collision = lambda ctx, px, py, pz, vx, vy, vz: (px, py, pz, vx, vy, vz)
    server._check_building_collisions = lambda ctx, px, py, pz, vx, vy: (px, py, vx, vy)
    ctx = ClientContext(client_id=1, client_addr=("127.0.0.1", 50000),
                        session=Session(), entity_id=0x14EA)
    ctx.entity_type = entity_type
    ctx.injected_input = (throttle, strafe)
    ctx.player_pos = (100, 200, 20)
    ctx.player_vel = (0, 0, 0)
    ctx.player_heading = 0
    ctx.player_fuel = vehicle_max_fuel(entity_type)
    ctx.player_energy = 100
    ctx.ground_level_override = None
    ctx.player_pose = {"roll": 0, "pitch": 0, "yaw": 0,
                       "pos": ctx.player_pos, "vel": (0, 0, 0)}
    server._update_player_position(ctx, dt_override=1 / 30)
    return ctx.debug_last_controller_step


def _mag(v):
    return math.sqrt(sum(c * c for c in v))


def test_server_scout_reverses_at_move_backward():
    step = make_step(EntityType.SCOUT, -1, 0)
    assert step["raw_impulse"][0] == pytest.approx(-38.0)


def test_server_scout_forward_and_strafe():
    assert make_step(EntityType.SCOUT, 1, 0)["raw_impulse"][0] == pytest.approx(85.0)
    assert abs(make_step(EntityType.SCOUT, 0, 1)["raw_impulse"][1]) == pytest.approx(72.0)


def test_server_scout_cap_is_125_percent_of_move_forward():
    step = make_step(EntityType.SCOUT, 1, 1)
    assert _mag(step["drive_impulse_uncapped"]) == pytest.approx(math.hypot(85.0, 72.0))
    assert _mag(step["drive_impulse_capped"]) == pytest.approx(106.25)


def test_server_tank_unchanged_reverse_and_cap():
    assert make_step(EntityType.TANK, -1, 0)["raw_impulse"][0] == pytest.approx(-85.0)
    step = make_step(EntityType.TANK, 1, 1)
    assert _mag(step["drive_impulse_capped"]) == pytest.approx(85.0)
