import os
from types import SimpleNamespace
import pytest
from wulfram.client import ClientContext
from wulfram.session import Session
from wulfram.physics import VehiclePhysics
from wulfram.weapons import WeaponSystem
from wulfram.native_live import NativeLiveWorld
from wulfram.native_physics import NativePhysicsError
from wulfram.building_collision import BuildingEntity


@pytest.fixture
def live():
    executable = os.environ.get("WULFRAM_TEST_NATIVE_WORKER")
    if not executable:
        pytest.skip("native worker required")
    ctx = ClientContext(client_id=1, client_addr=("127.0.0.1", 12345))
    ctx.session = Session()
    ctx.session.enter_game(1001, 1)
    ctx.session.translation_ack_received = True
    ctx.session.last_spawn_time = 1
    ctx.entity_id = 1001
    ctx.entity_type = 0
    ctx.running = True
    ctx.vehicle_physics = VehiclePhysics()
    ctx.weapon_system = WeaponSystem()
    ctx.player_pos = (100, 100, 1000)
    ctx.player_vel = (0, 0, 0)
    clients = [ctx]
    server = SimpleNamespace(map_name="crossroads", _snapshot_in_game_clients=lambda: list(clients),
                             _building_entities={}, _building_health={}, input_stale_timeout_s=0,
                             _normalize_behavior_axis_value=lambda c, v: max(-1, min(1, v / 1000 if abs(v) > 1.5 else v)),
                             _get_raw_turn_input=lambda c: -c.weapon_system.behavior_slots[1])
    service = NativeLiveWorld(server, executable)
    try:
        yield service, ctx, clients, server
    finally:
        service.close()


def test_publish_is_not_a_second_simulation_step(live):
    service, ctx, _, _ = live
    ctx.weapon_system.behavior_slots[5] = 0
    service.step()
    service.publish(ctx)
    first = ctx.player_pos
    for _ in range(10):
        service.publish(ctx)
    assert ctx.player_pos == first
    assert ctx.player_vel[2] == -2
    assert service.frames == 1
    assert ctx.player_pose["pos"] == first


def test_respawn_and_disconnect(live):
    service, ctx, clients, _ = live
    service.step()
    ctx.session.last_spawn_time = 2
    ctx.player_pos = (300, 300, 1000)
    service.publish(ctx)
    assert ctx.player_pos == (300, 300, 1000)
    service.step()
    service.publish(ctx)
    assert ctx.player_pos[0] == 300
    clients.clear()
    service.step()
    assert service.snapshots == {}
    assert service.members == {}


def test_teleport_is_consumed_before_publish(live):
    service, ctx, _, _ = live
    service.step()
    service.publish(ctx)
    ctx.player_pos = (400, 400, 1000)
    service.publish(ctx)
    assert ctx.player_pos == (400, 400, 1000)
    service.step()
    service.publish(ctx)
    assert ctx.player_pos[0] == 400


def test_raw_fuel_and_original_slot_signs(live):
    service, ctx, _, _ = live
    ctx.player_fuel = 33000
    ctx.weapon_system.behavior_slots[1:4] = [.5, 1, -.5]
    assert service.controls(ctx)[1:4] == [.5, 1, -.5]
    service.step()
    service.publish(ctx)
    assert ctx.angular_vel_yaw < 0
    assert ctx.player_vel[0] > 0
    assert ctx.player_vel[1] > 0


def test_building_lifecycle(live):
    service, _, _, server = live
    server._building_entities[10001] = BuildingEntity(100, 100, 20, 26, 1)
    service.step()
    assert 10001 in service.statics
    server._building_health[10001] = 0
    service.step()
    assert service.statics == {}


def test_worker_failure_stops_native_client(live):
    service, ctx, _, _ = live
    service.error = "test worker failure"
    with pytest.raises(NativePhysicsError):
        service.publish(ctx)
    assert not ctx.running
