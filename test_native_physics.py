"""Native integration gates; select a built worker with WULFRAM_TEST_NATIVE_WORKER."""
import os
import sys
from pathlib import Path
import pytest
from wulfram.native_physics import NativeTankWorld, NativePhysicsError
from wulfram.packets import build_behavior_packet

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def world():
    executable = os.environ.get("WULFRAM_TEST_NATIVE_WORKER")
    if not executable:
        pytest.skip("explicit native worker required")
    with NativeTankWorld(executable, ROOT / "slurpysoft-wulfram/data", "crossroads", build_behavior_packet()[1:]) as value:
        yield value


def test_wire_gravity_and_full_state(world):
    assert world.ready["gravity"] == 100
    world.add(7, (100, 100, 1000), rotation=(.1, .2, .3), angular_velocity=(.4, -.2, .1))
    world.input(7, 5, 0, 0)
    initial = world.snapshot()["bodies"][0]
    result = world.advance(20)
    body = result["bodies"][0]
    assert result["tick"] == 20
    assert body["velocity"][2] == pytest.approx(-2)
    assert body["position"][2] == pytest.approx(999.98, abs=.0001)
    assert all(body["rotation"][i] != initial["rotation"][i] for i in range(3))
    assert len(body["matrix"]) == 9
    assert body["force"] == body["torque"] == [0, 0, 0]


def test_all_bodies_step_once(world):
    for entity, x in [(1, 100), (2, 500)]:
        world.add(entity, (x, 100, 1000))
        world.input(entity, 5, 0, 0)
    result = world.advance(20)
    assert [body["velocity"][2] for body in result["bodies"]] == [-2, -2]
    assert result["contacts"] == 0
    world.remove(1)
    assert [b["id"] for b in world.advance(20)["bodies"]] == [2]


def test_repeatable_state_and_long_frame_partition(world):
    world.add(1, (100, 100, 1000), angular_velocity=(.2, .3, .4))
    world.input(1, 5, 0, 0)
    world.advance(220)
    actual = world.snapshot()["bodies"][0]
    world.remove(1)
    world.add(1, (100, 100, 1000), angular_velocity=(.2, .3, .4))
    world.input(1, 5, 0, 220)
    for dt in (73, 73, 74):
        world.advance(dt)
    assert world.snapshot()["bodies"][0] == actual


def test_native_error_closes_bridge(world):
    with pytest.raises(NativePhysicsError, match="unknown tank"):
        world.remove(123)
    with pytest.raises(NativePhysicsError, match="closed"):
        world.snapshot()


def test_bad_inputs_do_not_mutate_world(world):
    before = world.snapshot()
    for bad in (float("nan"), float("inf"), 2):
        with pytest.raises(ValueError):
            world.input(1, 2, bad, 0)
    with pytest.raises(ValueError):
        world.advance(0)
    with pytest.raises(ValueError):
        world.add(1, (0, 0, float("nan")))
    assert world.snapshot() == before


def test_terrain_and_pair_response(world):
    # Below the terrain and overlapping: exercise actual mesh discovery and
    # bounded separation rather than feeding a solver fabricated contacts.
    world.add(1, (100, 100, -100))
    world.add(2, (101, 100, -100))
    result = world.advance(20)
    assert result["pushes"] > 0
    assert all(body["position"][2] > -100 for body in result["bodies"])
    assert result["bodies"][0]["position"] != result["bodies"][1]["position"]


def test_active_controller_retains_three_axis_motion(world):
    world.add(1, (100, 100, 100))
    world.input(1, 2, 1, 0)
    world.input(1, 4, .5, 0)
    world.input(1, 8, .5, 0)
    for _ in range(100):
        result = world.advance(20)
    body = result["bodies"][0]
    assert body["position"][:2] != [100, 100]
    assert any(abs(value) > .001 for value in body["angular_velocity"][:2])
    assert body["force"] == body["torque"] == [0, 0, 0]


def test_airborne_tank_pair_impulse(world):
    world.add(1, (100, 100, 1000), velocity=(30, 0, 0))
    world.add(2, (112, 100, 1000), velocity=(-30, 0, 0))
    for entity in (1, 2):
        world.input(entity, 5, 0, 0)
    pairs = 0
    for _ in range(15):
        result = world.advance(20)
        pairs += result["pairs"]
    assert pairs > 0
    first, second = result["bodies"]
    assert first["position"][0] < second["position"][0]
    # The contact solver arrests closing motion to a tolerance; it need not
    # produce a rebound or exactly zero residual relative velocity.
    assert abs(first["velocity"][0]) < .01
    assert abs(second["velocity"][0]) < .01


def test_rejects_out_of_order_input_fifo(world):
    world.add(1, (100, 100, 1000))
    world.input(1, 2, 1, 100)
    with pytest.raises(NativePhysicsError, match="FIFO"):
        world.input(1, 2, 0, 50)


def test_static_mesh_arrests_tank(world):
    world.add_static(10001, 37, "tank_1", (112, 100, 1000))
    world.add(1, (98, 100, 1000), velocity=(60, 0, 0))
    world.input(1, 5, 0, 0)
    contacts = 0
    for _ in range(10):
        result = world.advance(20)
        contacts += result["pairs"]
    assert contacts > 0
    assert result["bodies"][0]["position"][0] < 110
    assert abs(result["bodies"][0]["velocity"][0]) < 1
    world.remove(10001)


def test_timeout_terminates_worker(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text('import sys, time\nprint(\'{"ready":1,"protocol":1}\', flush=True)\nsys.stdin.readline()\ntime.sleep(60)\n')
    with NativeTankWorld(sys.executable, script, "crossroads", b"", timeout=1) as value:
        process = value._process
        with pytest.raises(NativePhysicsError, match="timeout"):
            value.snapshot()
        assert process.poll() is not None
        with pytest.raises(NativePhysicsError, match="closed"):
            value.snapshot()
