#!/usr/bin/env python3
"""WULFRAM_COMBAT_PROFILE=upstream-2026-10: measured community-server combat.

Covers ranges and dead zones, cadence, flak no-leading and fan, hunter climb /
speed / outrunning, damage scaling, the wire shapes (UPDATE_ARRAY mask 0x12f
and the upstream gun-turret TRANSIENT_ARRAY), BEHAVIOR speed caps and HP, and
that the legacy default is untouched.

Run: uv run python test_combat_profile.py
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE.parent))

from wulfram import combat_profile as cp  # noqa: E402
from wulfram import upstream_combat as uc  # noqa: E402
from wulfram.building_collision import BuildingEntity  # noqa: E402
from wulfram.packets import (  # noqa: E402
    VEC_POS_MAX, VEC_POS_RANGE, VEC_ROT_MAX, VEC_ROT_RANGE, VEC_VEL_MAX, VEC_VEL_RANGE,
    build_behavior_packet, build_transient_array, decode_transient_array,
)
from wulfram.weapons import EntityType, WeaponSystem, WeaponType  # noqa: E402
from wulfram2_protocol.codec import BitReader, dequantize_float  # noqa: E402

UPSTREAM = {cp.PROFILE_ENV: cp.UPSTREAM_2026_10}
GUN_OID, FLAK_OID, LAUNCHER_OID = 101, 102, 103


def make_client(cid, x, y, z=10.0, team=2, entity_type=0, vel=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        client_id=cid, entity_id=1000 + cid, entity_type=entity_type, running=True,
        observer_transition=False, player_health=1.0, player_pos=(float(x), float(y), float(z)),
        player_vel=vel, session=SimpleNamespace(team_id=team, in_game=True, entity_id=1000 + cid,
                                                username=f"p{cid}", udp_addr=None),
    )


class FakeServer:
    """Just enough server surface for UpstreamTurretRuntime."""

    def __init__(self, buildings, clients, los=True):
        self._building_entities = buildings
        self._building_health = {oid: 500.0 for oid in buildings}
        self._building_crate_oids = set()
        self.clients = clients
        self.los = los
        self.hits = []
        self.udp_handler = None
        self._terrain_grid_collision = None

    def _snapshot_in_game_clients(self):
        return list(self.clients)

    def _turret_has_terrain_line_of_sight(self, building, client):
        return self.los

    def _apply_turret_hit(self, target, fraction, *, oid, source_name, now, hp):
        self.hits.append((now, target.client_id, oid, hp, fraction))
        target.player_health = max(0.0, target.player_health - fraction)


def turret(kind_type, x=0.0, y=0.0, z=0.0, team=1):
    return BuildingEntity(x=x, y=y, z=z, entity_type=kind_type, team_id=team, heading=0.0)


def run(runtime, t0, t1, dt=0.01, move=None):
    t = t0
    while t < t1 - 1e-9:
        if move:
            move(t, dt)
        runtime.update(t)
        t += dt
    return t


class ProfileSwitchTests(unittest.TestCase):
    def test_default_is_legacy(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cp.PROFILE_ENV, None)
            self.assertEqual(cp.active_profile(), cp.LEGACY)
            self.assertFalse(cp.is_upstream())

    def test_aliases_and_unknown(self):
        self.assertTrue(cp.is_upstream({cp.PROFILE_ENV: "upstream-2026-10"}))
        self.assertTrue(cp.is_upstream({cp.PROFILE_ENV: " Upstream "}))
        self.assertFalse(cp.is_upstream({cp.PROFILE_ENV: "legacy"}))
        self.assertFalse(cp.is_upstream({cp.PROFILE_ENV: "bogus"}))


class GunTurretTests(unittest.TestCase):
    def _runtime(self, distance, **kw):
        tank = make_client(1, distance, 0.0, **kw)
        server = FakeServer({GUN_OID: turret(EntityType.GUN_TURRET)}, [tank])
        return uc.UpstreamTurretRuntime(server, network=False), server, tank

    def test_range_475_no_minimum(self):
        for distance, expect in ((10.0, True), (300.0, True), (474.0, True), (476.0, False), (600.0, False)):
            rt, server, _ = self._runtime(distance)
            rt.update(0.0)
            self.assertEqual(bool(server.hits), expect, distance)

    def test_cadence_30hp_every_quarter_second(self):
        rt, server, tank = self._runtime(400.0)
        run(rt, 0.0, 2.0, dt=1.0 / 30.0)
        times = [h[0] for h in server.hits]
        self.assertEqual(len(times), 8)  # 0.00 .. 1.75
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertTrue(all(abs(g - 0.25) < 1.0 / 30.0 + 1e-9 for g in gaps), gaps)
        # Phase-locked: no cumulative drift from the 30 Hz update clock.
        self.assertAlmostEqual(times[-1], 1.75, delta=1.0 / 30.0)
        self.assertTrue(all(h[3] == 30.0 for h in server.hits))
        self.assertAlmostEqual(server.hits[0][4], 30.0 / 500.0)  # tank 500 HP

    def test_scout_takes_30_of_330(self):
        rt, server, _ = self._runtime(200.0, entity_type=1)
        rt.update(0.0)
        self.assertAlmostEqual(server.hits[0][4], 30.0 / 330.0)

    def test_dps_120_kills_tank_in_about_4s(self):
        rt, server, tank = self._runtime(300.0)
        run(rt, 0.0, 10.0, dt=0.02)
        total, dead_at = 0.0, None
        for when, _cid, _oid, hp, _fraction in server.hits:
            total += hp
            if total >= 500.0:
                dead_at = when
                break
        self.assertAlmostEqual(dead_at, 4.0, delta=0.05)  # hit 17 at t=4.0: 17 x 30 = 510 >= 500

    def test_terrain_los_blocks(self):
        tank = make_client(1, 300.0, 0.0)
        server = FakeServer({GUN_OID: turret(EntityType.GUN_TURRET)}, [tank], los=False)
        uc.UpstreamTurretRuntime(server, network=False).update(0.0)
        self.assertEqual(server.hits, [])

    def test_friendly_crate_and_neutral_turrets_hold_fire(self):
        tank = make_client(1, 100.0, 0.0, team=1)
        server = FakeServer({GUN_OID: turret(EntityType.GUN_TURRET, team=1)}, [tank])
        uc.UpstreamTurretRuntime(server, network=False).update(0.0)
        self.assertEqual(server.hits, [])
        enemy = make_client(2, 100.0, 0.0, team=2)
        server = FakeServer({GUN_OID: turret(EntityType.GUN_TURRET, team=0)}, [enemy])
        uc.UpstreamTurretRuntime(server, network=False).update(0.0)
        self.assertEqual(server.hits, [])
        server = FakeServer({GUN_OID: turret(EntityType.GUN_TURRET)}, [enemy])
        server._building_crate_oids = {GUN_OID}
        uc.UpstreamTurretRuntime(server, network=False).update(0.0)
        self.assertEqual(server.hits, [])

    def test_nearest_enemy_is_targeted(self):
        far = make_client(1, 450.0, 0.0)
        near = make_client(2, 0.0, 200.0)
        server = FakeServer({GUN_OID: turret(EntityType.GUN_TURRET)}, [far, near])
        uc.UpstreamTurretRuntime(server, network=False).update(0.0)
        self.assertEqual(server.hits[0][1], 2)


class FlakTests(unittest.TestCase):
    def _runtime(self, tank):
        server = FakeServer({FLAK_OID: turret(EntityType.SENSOR_BUILDING)}, [tank])
        return uc.UpstreamTurretRuntime(server, network=False), server

    def test_band_300_to_900(self):
        for distance, expect in ((250.0, 0), (299.0, 0), (301.0, 3), (600.0, 3), (899.0, 3), (905.0, 0)):
            rt, _ = self._runtime(make_client(1, distance, 0.0, z=5.0))
            rt.update(0.0)
            self.assertEqual(len(rt.projectiles), expect, distance)

    def test_volley_every_6s(self):
        tank = make_client(1, 0.0, 3000.0)  # out of range: shells never hit
        rt, _ = self._runtime(tank)
        tank.player_pos = (600.0, 0.0, 5.0)
        launches = []
        orig = rt._launch

        def spy(proj, clients):
            launches.append(proj.spawn_t)
            orig(proj, clients)

        rt._launch = spy

        def dodge(t, dt):  # sidestep so no shell connects
            tank.player_pos = (600.0, 200.0 * math.sin(t * 3.0), 5.0)

        run(rt, 0.0, 18.5, dt=0.05, move=dodge)
        volleys = sorted(set(round(t, 3) for t in launches))
        self.assertEqual(len(launches), 3 * len(volleys))
        self.assertEqual(len(volleys), 4)  # 0, 6, 12, 18
        for a, b in zip(volleys, volleys[1:]):
            self.assertAlmostEqual(b - a, 6.0, delta=0.051)

    def test_no_leading_and_fan(self):
        # Target crossing at 120 u/s: the centre shell still aims at where it IS.
        tank = make_client(1, 600.0, 0.0, z=5.0, vel=(0.0, 120.0, 0.0))
        rt, _ = self._runtime(tank)
        rt.update(0.0)
        shells = sorted(rt.projectiles, key=lambda p: math.atan2(p.vel[1], p.vel[0]))
        self.assertEqual(len(shells), 3)
        centre = shells[1]
        muzzle = (0.0, 0.0, cp.FLAK_MUZZLE_DZ)
        want = cp.direction_to(muzzle, tank.player_pos)
        got = tuple(v / cp.FLAK_SHELL_SPEED for v in centre.vel)
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=6)
        for shell in shells:
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in shell.vel)), 140.0, places=6)
        yaws = [math.degrees(math.atan2(p.vel[1], p.vel[0])) for p in shells]
        self.assertAlmostEqual(yaws[1] - yaws[0], 5.1, places=6)
        self.assertAlmostEqual(yaws[2] - yaws[1], 5.1, places=6)

    def test_hits_stationary_misses_crossing_target(self):
        still = make_client(1, 500.0, 0.0, z=5.0)
        rt, server = self._runtime(still)
        run(rt, 0.0, 5.0, dt=0.05)
        self.assertEqual(len(server.hits), 1)  # first shell to arrive hits; others fanned wide
        self.assertEqual(server.hits[0][3], 37.5)
        self.assertAlmostEqual(server.hits[0][0], (500.0 - 15.0) / 140.0, delta=0.1)

        mover = make_client(1, 500.0, 0.0, z=5.0)
        rt, server = self._runtime(mover)

        def cross(t, dt):
            mover.player_pos = (500.0, 120.0 * t, 5.0)

        run(rt, 0.0, 5.9, dt=0.05, move=cross)
        self.assertEqual(server.hits, [])

    def test_shell_lifetime_7s(self):
        tank = make_client(1, 800.0, 0.0, z=5.0)
        rt, _ = self._runtime(tank)
        rt.update(0.0)
        tank.session.in_game = False  # nothing to hit; nothing to retarget
        run(rt, 0.01, 6.9, dt=0.05)
        self.assertEqual(len(rt.projectiles), 3)
        run(rt, 6.9, 7.2, dt=0.05)
        self.assertEqual(len(rt.projectiles), 0)
        self.assertEqual(rt.stats.expired["flak"], 3)


class LauncherTests(unittest.TestCase):
    def _runtime(self, tank):
        server = FakeServer({LAUNCHER_OID: turret(EntityType.LAUNCHER)}, [tank])
        return uc.UpstreamTurretRuntime(server, network=False), server

    def test_band_900_to_1400(self):
        for distance, expect in ((500.0, 0), (899.0, 0), (901.0, 1), (1399.0, 1), (1405.0, 0)):
            rt, _ = self._runtime(make_client(1, distance, 0.0))
            rt.update(0.0)
            self.assertEqual(len(rt.projectiles), expect, distance)

    def test_climb_speed_and_cadence(self):
        tank = make_client(1, 1000.0, 0.0, z=10.0)
        rt, server = self._runtime(tank)
        rt.update(0.0)
        hunter = rt.projectiles[0]
        self.assertEqual(hunter.pos[2], cp.HUNTER_LAUNCH_DZ)
        speeds, vertical_until = [], None

        def track(t, dt):
            nonlocal vertical_until
            for p in rt.projectiles:
                speeds.append(math.sqrt(sum(v * v for v in p.vel)))
            if hunter in rt.projectiles and hunter.vel == (0.0, 0.0, cp.HUNTER_SPEED):
                vertical_until = t

        run(rt, 0.01, 6.0, dt=0.05, move=track)
        self.assertTrue(all(abs(s - 95.0) < 1e-6 for s in speeds))
        self.assertAlmostEqual(vertical_until, cp.HUNTER_CLIMB_S, delta=0.11)
        launches = rt.stats.shots["launcher"]
        self.assertEqual(launches, 3)  # t = 0, 2, 4

    def test_hits_a_parked_target(self):
        tank = make_client(1, 950.0, 0.0, z=10.0)
        rt, server = self._runtime(tank)
        run(rt, 0.0, 0.5, dt=0.05)
        rt.next_fire[LAUNCHER_OID] = float("inf")  # one hunter only
        run(rt, 0.5, 16.0, dt=0.05)
        self.assertEqual(len(server.hits), 1)
        self.assertEqual(server.hits[0][3], cp.HUNTER_DAMAGE_HP)

    def test_outrun_at_tank_speed(self):
        # A tank driving away at the 120 u/s cap is never caught by a 95 u/s hunter.
        tank = make_client(1, 950.0, 0.0, z=10.0)
        rt, server = self._runtime(tank)

        def flee(t, dt):
            tank.player_pos = (950.0 + 120.0 * t, 0.0, 10.0)

        run(rt, 0.0, 25.0, dt=0.05, move=flee)
        self.assertEqual(server.hits, [])
        self.assertGreaterEqual(rt.stats.shots["launcher"], 1)


def _decode_entity_record(pkt):
    """Strict decode of a one-entity UPDATE_ARRAY without local state (our widths)."""
    assert pkt[0] == 0x0E
    br = BitReader(pkt[5:])
    assert br.read_bits(1) == 0
    assert br.read_bits(8) == 1
    out = {"oid": br.read_bits(32), "manned": br.read_bits(1), "mask": br.read_bits(10), "qidx": br.read_bits(16)}
    m = out["mask"]
    if m & 1:
        out["type"], out["variant"], out["team"], out["static"] = (br.read_bits(8), br.read_bits(8), br.read_bits(8), br.read_bits(1))

    def vec(mx, rg):
        assert br.read_bits(4) == 15
        return tuple(dequantize_float(br.read_bits(16), mx, rg, 16) for _ in range(3))

    if m & 2:
        out["pos"] = vec(VEC_POS_MAX, VEC_POS_RANGE)
    if m & 4:
        out["vel"] = vec(VEC_VEL_MAX, VEC_VEL_RANGE)
    if m & 8:
        out["rot"] = vec(VEC_ROT_MAX, VEC_ROT_RANGE)
    if m & 32:
        out["health"] = dequantize_float(br.read_bits(10), 1.0, 1.0, 10)
    if m & 256:
        out["target"] = br.read_bits(32)
    out["leftover"] = (len(br.data) - br.byte_pos) * 8 - br.bit_pos
    return out


class WireShapeTests(unittest.TestCase):
    def test_flak_spawn_matches_upstream_record(self):
        # Upstream 10-03 sample: oid 388222 mask 0x12f vel (44.98,-132.45,5.55)
        # rot (0.0396, 0, -1.2434) health 1.0 target 383379.
        vel = (44.98, -132.45, 5.55)
        pkt = uc.build_turret_projectile_packet(
            388222, EntityType.FLAK_SHELL, 1, (3554.86, 4581.89, 20.5), vel, 1234, target_oid=383379)
        rec = _decode_entity_record(pkt)
        self.assertEqual(rec["mask"], 0x12F)
        self.assertEqual((rec["type"], rec["variant"], rec["team"], rec["static"], rec["manned"]), (5, 1, 1, 1, 0))
        self.assertEqual(rec["target"], 383379)
        self.assertAlmostEqual(rec["health"], 1.0, places=2)
        for got, want in zip(rec["rot"], (0.0396, 0.0, -1.2434)):
            self.assertAlmostEqual(got, want, places=3)
        for got, want in zip(rec["vel"], vel):
            self.assertAlmostEqual(got, want, places=1)
        self.assertLess(rec["leftover"], 8)

    def test_hunter_climb_rotation_and_steer_mask(self):
        rot = cp.upstream_projectile_rotation((0.0, 0.0, 95.0))
        self.assertAlmostEqual(rot[0], math.pi / 2, places=6)
        self.assertEqual(rot[1:], (0.0, 0.0))
        pkt = uc.build_turret_projectile_packet(5, EntityType.HUNTER, 1, (0, 0, 0), (60.0, -70.0, -13.0), 7,
                                                mask=uc.STEER_MASK)
        rec = _decode_entity_record(pkt)
        self.assertEqual(rec["mask"], 0x00C)
        self.assertNotIn("pos", rec)
        self.assertLess(rec["leftover"], 8)

    def test_gun_turret_transient_is_byte_identical_to_upstream(self):
        # Captured 2026-10-03 t=100.09: turret oid 383799.
        self.assertEqual(uc.build_gun_turret_fire_transient(383799).hex(), "0d0100050002ed9bc00000")
        events = decode_transient_array(uc.build_gun_turret_fire_transient(383799), upstream=True)
        self.assertEqual(events[0]["source_oid"], 383799)
        self.assertEqual(events[0]["type"], 5)

    def test_legacy_transient_bytes_unchanged(self):
        # Events without 'source_oid' keep the frozen-golden encoding.
        legacy = build_transient_array([{"type": 0, "pos": None, "entity_id": 1337}])
        self.assertEqual(legacy.hex(), "0d010000414e40")  # testdata/wire_parity_golden.json fx#1


class DamageModelTests(unittest.TestCase):
    def test_autocannon_curve(self):
        self.assertAlmostEqual(cp.autocannon_dps(0.0), 78.9)
        self.assertAlmostEqual(cp.autocannon_dps(250.0), 51.9)
        self.assertAlmostEqual(cp.autocannon_dps(500.0), 24.9)
        self.assertEqual(cp.autocannon_dps(800.0), 0.0)
        self.assertAlmostEqual(cp.autocannon_shot_hp(400.0, 0.1), 3.57)
        self.assertAlmostEqual(cp.autocannon_shot_hp(0.0, 5.0), 78.9 * cp.AUTOCANNON_MAX_SHOT_DT)

    def test_pulse_splash_monotonic(self):
        self.assertEqual(cp.pulse_splash_hp(0.0), 300.0)
        self.assertEqual(cp.pulse_splash_hp(80.0), 0.0)
        samples = [cp.pulse_splash_hp(d) for d in range(0, 80)]
        self.assertTrue(all(a >= b for a, b in zip(samples, samples[1:])))
        self.assertTrue(130.0 <= cp.pulse_splash_hp(18.0) <= 140.0)
        self.assertTrue(50.0 <= cp.pulse_splash_hp(45.0) <= 80.0)

    def test_hp_table(self):
        self.assertEqual(cp.entity_max_hp(EntityType.TANK), 500)
        self.assertEqual(cp.entity_max_hp(EntityType.SCOUT), 330)
        self.assertEqual(cp.entity_max_hp(EntityType.GUN_TURRET), 525)
        self.assertEqual(cp.entity_max_hp(EntityType.SENSOR_BUILDING), 550)
        self.assertEqual(cp.entity_max_hp(EntityType.LAUNCHER), 500)
        self.assertEqual(cp.entity_max_hp(EntityType.ENERGY_BUILDING), 700)


class WeaponAndBehaviorTests(unittest.TestCase):
    def test_weapon_configs(self):
        legacy = WeaponSystem()
        self.assertEqual(legacy.projectile_configs[WeaponType.PULSE_CANNON]["speed"], 75.0)
        with patch.dict(os.environ, UPSTREAM):
            ws = WeaponSystem()
        pulse = ws.projectile_configs[WeaponType.PULSE_CANNON]
        self.assertEqual((pulse["speed"], pulse["lifetime"], pulse["cooldown"]), (210.0, 6.0, 0.6))
        self.assertEqual(ws.projectile_configs[WeaponType.PIERCER]["speed"], 125.0)
        self.assertEqual(ws.projectile_configs[WeaponType.HUNTER_SEEKER]["speed"], 95.0)
        self.assertEqual(ws.weapon_energy_costs[WeaponType.PULSE_CANNON], 10.0)
        self.assertEqual(ws.pulse_shell_speed, 210.0)
        self.assertEqual(ws.projectile_spawn_offset, 14.0)  # CAP muzzle

    def test_pulse_follows_the_full_hull_orientation(self):
        """CAP (68 shells): launch direction = shooter yaw and pitch (nose-down positive), muzzle
        14 u along that 3-D direction."""
        with patch.dict(os.environ, {**UPSTREAM, "WULFRAM_PROJECTILE_BARREL_UP": "0",
                                     "WULFRAM_PROJECTILE_BARREL_RIGHT": "0"}):
            ws = WeaponSystem()
        ws.player_pos = (100.0, 200.0, 30.0)
        yaw, pitch = math.radians(30.0), math.radians(21.86)  # captured shot 388358: rot[1] 21.86
        ws.player_rot = (0.0, pitch, yaw)
        ws.use_pitch, ws.up_axis = True, "z"
        proj = ws._fire_pulse_cannon()
        v = proj.vel
        elev = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
        self.assertAlmostEqual(elev, -21.86, places=3)
        self.assertAlmostEqual(math.degrees(math.atan2(v[1], v[0])), 30.0, places=3)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in v)), 210.0, places=3)
        off = [proj.pos[i] - ws.player_pos[i] for i in range(3)]
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in off)), 14.0, places=3)

    def test_behavior_speed_caps_and_hp(self):
        from client.wulfram_client.network.behavior import parse_behavior
        legacy = parse_behavior(build_behavior_packet())
        with patch.dict(os.environ, UPSTREAM):
            up = parse_behavior(build_behavior_packet())
        self.assertEqual(legacy.active_vehicle_physics[0].values[3], 80.0)
        self.assertEqual(legacy.entity_types[0].behavior_flags, 100)
        self.assertEqual(up.active_vehicle_physics[0].values[3], 120.0)
        self.assertEqual(up.active_vehicle_physics[1].values[4], 140.0)
        self.assertEqual([e.behavior_flags for e in up.entity_types], list(cp.UPSTREAM_ENTITY_HP))

    def test_tank_governor(self):
        os.environ.pop(cp.PROFILE_ENV, None)
        self.assertEqual(cp.tank_governor_max_velocity(80.0), 80.0)
        with patch.dict(os.environ, UPSTREAM):
            self.assertEqual(cp.tank_governor_max_velocity(80.0), 120.0)


class ServerIntegrationTests(unittest.TestCase):
    """Through the real WulframServer: profile dispatch, legacy untouched, packets."""

    @classmethod
    def setUpClass(cls):
        from wulfram.server import WulframServer
        cls.server = WulframServer(host="127.0.0.1", port=0)

    def _ctx(self, x, y, z=10.0):
        from wulfram.client import ClientContext
        from wulfram.session import Session, Phase
        ctx = ClientContext(client_id=7, client_addr=("127.0.0.1", 50077))
        ctx.session = Session()
        ctx.session.phase = Phase.IN_GAME
        ctx.session.in_game = True
        ctx.session.team_id = 2
        ctx.session.entity_id = 4242
        ctx.session.udp_addr = ("127.0.0.1", 50077)
        ctx.session.translation_ack_received = True
        ctx.running = True
        ctx.entity_type = 0
        ctx.player_health = 1.0
        ctx.player_pos = (float(x), float(y), float(z))
        return ctx

    def _arm(self, building_type, ctx):
        server = self.server
        server._building_entities = {9001: turret(building_type, x=0.0, y=0.0, z=0.0, team=1)}
        server._building_health = {9001: 525.0}
        server._building_crate_oids = set()
        server._turret_last_fire = {}
        server._upstream_turret_runtime = None
        server._snapshot_in_game_clients = lambda: [ctx]
        server._turret_has_terrain_line_of_sight = lambda b, c: True
        sent = []
        server.udp_handler = SimpleNamespace(send_to=lambda pkt, addr: sent.append(pkt))
        return sent

    def test_legacy_gun_turret_unchanged_by_default(self):
        ctx = self._ctx(100.0, 0.0)  # inside the legacy 120-u ring
        self._arm(EntityType.GUN_TURRET, ctx)
        os.environ.pop(cp.PROFILE_ENV, None)
        self.server._update_turret_ai()
        self.assertAlmostEqual(ctx.player_health, 0.92)  # legacy 8 %
        far = self._ctx(400.0, 0.0)
        self._arm(EntityType.GUN_TURRET, far)
        self.server._update_turret_ai()
        self.assertEqual(far.player_health, 1.0)  # legacy range 120 u

    def test_upstream_gun_turret_through_server(self):
        ctx = self._ctx(400.0, 0.0)
        sent = self._arm(EntityType.GUN_TURRET, ctx)
        with patch.dict(os.environ, UPSTREAM):
            self.server._update_turret_ai()
        self.assertAlmostEqual(ctx.player_health, 1.0 - 30.0 / 500.0)
        # TRANSIENT_ARRAY stays behind the existing remote-FX gate by default.
        self.assertNotIn(uc.build_gun_turret_fire_transient(9001), sent)
        ctx2 = self._ctx(400.0, 0.0)
        sent = self._arm(EntityType.GUN_TURRET, ctx2)
        with patch.dict(os.environ, {**UPSTREAM, "WULFRAM_UPSTREAM_TURRET_FX": "1"}):
            self.server._update_turret_ai()
        self.assertIn(uc.build_gun_turret_fire_transient(9001), sent)

    def test_upstream_flak_volley_reaches_the_wire(self):
        ctx = self._ctx(600.0, 0.0, z=5.0)
        sent = self._arm(EntityType.SENSOR_BUILDING, ctx)
        with patch.dict(os.environ, UPSTREAM):
            self.server._update_turret_ai()
        spawns = [p for p in sent if p[0] == 0x0E]
        self.assertEqual(len(spawns), 3)
        runtime = self.server._upstream_turret_runtime
        self.assertEqual(len(runtime.projectiles), 3)
        self.assertTrue(all(uc.OID_BASE <= p.entity_id < uc.OID_BASE + uc.OID_SPAN for p in runtime.projectiles))

    def _duel(self, target_xy):
        attacker = self._ctx(0.0, 0.0)
        attacker.client_id = 8
        attacker.session.team_id = 1
        attacker.session.entity_id = 4343
        attacker.player_heading = 0.0
        target = self._ctx(*target_xy)
        self._arm(EntityType.GUN_TURRET, target)
        self.server._building_entities = {}
        self.server._snapshot_in_game_clients = lambda: [attacker, target]
        return attacker, target

    def test_upstream_pulse_direct_hit_is_300hp(self):
        from wulfram2_protocol.entities import Projectile
        attacker, target = self._duel((50.0, 0.0))
        proj = Projectile(entity_id=55, entity_type=EntityType.PULSE_SHELL, owner_id=1, team=1,
                          pos=(50.0, 0.0, 10.0), vel=(210.0, 0.0, 0.0), spawn_time=0.0, lifetime=6.0)
        with patch.dict(os.environ, UPSTREAM):
            self.server._apply_damage(target, proj, attacker)
        self.assertAlmostEqual(target.player_health, 1.0 - 300.0 / 500.0)
        legacy_target = self._duel((50.0, 0.0))[1]
        os.environ.pop(cp.PROFILE_ENV, None)
        self.server._apply_damage(legacy_target, proj, attacker)
        self.assertAlmostEqual(legacy_target.player_health, 0.8)  # legacy 20 %

    def test_upstream_pulse_splash_vehicle_and_building(self):
        from wulfram2_protocol.entities import Projectile
        attacker, target = self._duel((28.0, 0.0))
        self.server._building_entities = {9100: turret(EntityType.GUN_TURRET, x=10.0, y=45.0, z=10.0, team=2)}
        self.server._building_health = {9100: 525.0}
        self.server._building_max_health = {9100: 525.0}
        proj = Projectile(entity_id=56, entity_type=EntityType.PULSE_SHELL, owner_id=1, team=1,
                          pos=(10.0, 0.0, 10.0), vel=(210.0, 0.0, 0.0), spawn_time=0.0, lifetime=6.0)
        with patch.dict(os.environ, UPSTREAM):
            self.server._apply_upstream_pulse_splash(proj, attacker, (10.0, 0.0, 10.0))
        self.assertAlmostEqual(target.player_health, 1.0 - cp.pulse_splash_hp(18.0) / 500.0, places=5)
        self.assertAlmostEqual(self.server._building_health[9100], 525.0 - cp.pulse_splash_hp(45.0), places=3)

    def test_upstream_autocannon_follows_distance_curve(self):
        attacker, target = self._duel((400.0, 0.0))
        with patch.dict(os.environ, UPSTREAM):
            self.server._fire_upstream_autocannon(attacker, attacker.player_pos, (0.0, 0.0, 0.0))
        expected_hp = cp.autocannon_shot_hp(400.0, 0.1)
        self.assertAlmostEqual(target.player_health, 1.0 - expected_hp / 500.0, places=6)
        off_lane = self._duel((0.0, 400.0))[1]  # 90 deg off the heading: no auto-aim fallback
        with patch.dict(os.environ, UPSTREAM):
            self.server._fire_upstream_autocannon(attacker, attacker.player_pos, (0.0, 0.0, 0.0))
        self.assertEqual(off_lane.player_health, 1.0)

    def test_upstream_autocannon_shoots_down_a_nearer_hunter(self):
        attacker, target = self._duel((400.0, 0.0))
        attacker.session.team_id = 2
        target.session.team_id = 1
        rt = uc.UpstreamTurretRuntime(self.server, network=False)
        hunter = uc.TurretProjectile(
            entity_id=50002, kind=uc.LAUNCHER, entity_type=EntityType.HUNTER, team=1, turret_oid=7,
            turret_name="L", target=attacker, target_oid=4343, pos=(200.0, 3.0, 90.0), vel=(-95.0, 0.0, 0.0),
            spawn_t=0.0, lifetime=20.0, damage_hp=200.0, hit_radius=15.0, hp=cp.HUNTER_HP)
        rt.projectiles.append(hunter)
        self.server._upstream_turret_runtime = rt
        try:
            with patch.dict(os.environ, UPSTREAM):
                self.server._fire_upstream_autocannon(attacker, attacker.player_pos, (0.0, 0.0, 0.0))
            self.assertEqual(target.player_health, 1.0)  # the hunter took the shot
            self.assertAlmostEqual(hunter.hp, cp.HUNTER_HP - cp.autocannon_shot_hp(math.dist((200, 3, 90), (0, 0, 10)), 0.1))
            hunter.hp = 0.5
            attacker._upstream_autocannon_last_shot = None
            with patch.dict(os.environ, UPSTREAM):
                self.server._fire_upstream_autocannon(attacker, attacker.player_pos, (0.0, 0.0, 0.0))
            self.assertNotIn(hunter, rt.projectiles)
            self.assertEqual(rt.stats.shot_down[uc.LAUNCHER], 1)
        finally:
            self.server._upstream_turret_runtime = None


class HunterShootdownTests(unittest.TestCase):
    """CAP: upstream defenders shot launcher hunters down with the autocannon (INF 25 HP)."""

    def _runtime_with_hunter(self, pos):
        server = FakeServer({}, [])
        rt = uc.UpstreamTurretRuntime(server, network=False)
        proj = uc.TurretProjectile(
            entity_id=50001, kind=uc.LAUNCHER, entity_type=EntityType.HUNTER, team=1, turret_oid=7,
            turret_name="L", target=None, target_oid=0, pos=pos, vel=(-95.0, 0.0, 0.0), spawn_t=0.0,
            lifetime=cp.HUNTER_LIFETIME_S, damage_hp=cp.HUNTER_DAMAGE_HP, hit_radius=cp.HUNTER_HIT_RADIUS,
            hp=cp.HUNTER_HP)
        rt.projectiles.append(proj)
        return rt, proj

    def test_hunter_hp_and_lane(self):
        self.assertEqual(cp.HUNTER_HP, 25.0)
        rt, proj = self._runtime_with_hunter((300.0, 5.0, 120.0))
        # wrong team, behind, off-lane, or beyond a nearer vehicle: no effect
        self.assertIsNone(rt.shoot_in_lane(1, (0, 0, 0), 0.0, range_limit=730, lane_radius=12, max_along=730, shot_dt=0.1))
        self.assertIsNone(rt.shoot_in_lane(2, (0, 0, 0), math.pi, range_limit=730, lane_radius=12, max_along=730, shot_dt=0.1))
        self.assertIsNone(rt.shoot_in_lane(2, (0, 0, 0), math.radians(20), range_limit=730, lane_radius=12, max_along=730, shot_dt=0.1))
        self.assertIsNone(rt.shoot_in_lane(2, (0, 0, 0), 0.0, range_limit=730, lane_radius=12, max_along=250, shot_dt=0.1))
        shots = 0
        while proj in rt.projectiles:
            res = rt.shoot_in_lane(2, (0, 0, 0), 0.0, range_limit=730, lane_radius=12, max_along=730, shot_dt=0.1)
            shots += 1
            self.assertAlmostEqual(res["hp"], cp.autocannon_shot_hp(math.dist(proj.pos, (0, 0, 0)), 0.1))
        # 25 HP at ~323 u: DPS 44 -> ~0.57 s of fire, as the captured 0.5-0.9 s drops
        self.assertEqual(shots, math.ceil(25.0 / cp.autocannon_shot_hp(math.dist((300, 5, 120), (0, 0, 0)), 0.1)))
        self.assertEqual(rt.stats.shot_down[uc.LAUNCHER], 1)

    def test_flak_shells_cannot_be_shot(self):
        rt, proj = self._runtime_with_hunter((300.0, 0.0, 10.0))
        proj.hp = 0.0
        self.assertIsNone(rt.shoot_in_lane(2, (0, 0, 0), 0.0, range_limit=730, lane_radius=12, max_along=730, shot_dt=0.1))


class CapturedHunterReplayTests(unittest.TestCase):
    """Replays captured upstream launcher hunters (2026-10-03, map tron) through the runtime
    with the server's real tron terrain. Upstream: 0 hits; the terrain-ended flights must end in
    terrain at the captured time, and the two the target shot down must be stoppable."""

    @classmethod
    def setUpClass(cls):
        import contextlib
        import io
        import json
        from wulfram.server import WulframServer
        with patch.dict(os.environ, {"WULFRAM_MAP_NAME": "tron"}), contextlib.redirect_stdout(io.StringIO()):
            cls.terrain = WulframServer(host="127.0.0.1", port=0)._terrain_grid_collision
        cls.fixture = json.loads((HERE / "testdata" / "upstream_hunter_replays.json").read_text())["hunters"]

    @staticmethod
    def _target_at(track, t):
        i = min(len(track) - 1, max(0, int(t / 0.2)))
        j = min(len(track) - 1, i + 1)
        a = 0.0 if j == i else max(0.0, min(1.0, (t - track[i][0]) / (track[j][0] - track[i][0])))
        return tuple(track[i][k] + a * (track[j][k] - track[i][k]) for k in (1, 2, 3))

    def _replay(self, h, defend_from=None):
        track = h["target"]
        tank = make_client(1, *self._target_at(track, 0.0))
        server = FakeServer({}, [tank])
        server._terrain_grid_collision = self.terrain
        rt = uc.UpstreamTurretRuntime(server, network=False)
        proj = uc.TurretProjectile(
            entity_id=50001, kind=uc.LAUNCHER, entity_type=EntityType.HUNTER, team=1, turret_oid=7,
            turret_name="L", target=tank, target_oid=tank.entity_id, pos=tuple(h["spawn"]),
            vel=(0.0, 0.0, cp.HUNTER_SPEED), spawn_t=0.0, lifetime=cp.HUNTER_LIFETIME_S,
            damage_hp=cp.HUNTER_DAMAGE_HP, hit_radius=cp.HUNTER_HIT_RADIUS, hp=cp.HUNTER_HP)
        rt.projectiles.append(proj)
        rt._last_step = 0.0
        t, closest = 0.0, float("inf")
        while rt.projectiles and t < 30.0:
            t = round(t + uc.STEP_S, 6)
            tank.player_pos = self._target_at(track, min(t, track[-1][0]))
            if defend_from is not None and t >= defend_from and abs(t * 10 - round(t * 10)) < 1e-6:
                yaw = math.atan2(proj.pos[1] - tank.player_pos[1], proj.pos[0] - tank.player_pos[0])
                rt.shoot_in_lane(2, tank.player_pos, yaw, range_limit=cp.AUTOCANNON_MAX_RANGE,
                                 lane_radius=cp.AUTOCANNON_LANE_RADIUS, max_along=1e9, shot_dt=0.1)
            rt._step_projectiles(t, [tank])
            if proj in rt.projectiles:
                closest = min(closest, math.dist(proj.pos, tank.player_pos))
        s = rt.stats
        end = "hit" if server.hits else "terrain" if s.terrain else "shot down" if s.shot_down else "expired"
        return end, t, closest

    def test_terrain_ended_launches_miss_like_upstream(self):
        rows = [h for h in self.fixture if h["observed_end"] == "terrain"]
        self.assertGreaterEqual(len(rows), 8)
        self.assertTrue(any(h["target_speed_med"] == 0.0 for h in rows))  # parked targets too
        self.assertTrue(any(h["target_speed_med"] > 40.0 for h in rows))
        ends = []
        for h in rows:
            end, t, closest = self._replay(h)
            ends.append(end)
            self.assertEqual(end, "terrain", h["oid"])
            self.assertAlmostEqual(t, h["observed_life"], delta=0.5, msg=h["oid"])
            self.assertAlmostEqual(closest, h["observed_closest"], delta=max(15.0, 0.15 * h["observed_closest"]),
                                   msg=h["oid"])
        self.assertEqual(ends.count("hit"), 0)  # upstream: 0 hits

    def test_shot_down_launches_need_the_defence(self):
        for h in (h for h in self.fixture if h["observed_end"] == "shot down"):
            self.assertEqual(self._replay(h)[0], "hit", h["oid"])  # why the target shot them down
            # the target opened fire 8 s after launch (CAP shootdowns: 8.6-12.2 s)
            end, t, _ = self._replay(h, defend_from=8.0)
            self.assertEqual(end, "shot down", h["oid"])
            self.assertLess(t, h["observed_life"] + 1.0, h["oid"])


class StaticBuildingRaycastTests(unittest.TestCase):
    """Regression: the static-building quadtree filed each building in the
    y-mirrored quadrant, so world rays (projectiles, autocannon) never hit a
    static building.  Uses the committed default map (crossroads)."""

    GUN_OID = 10006
    GUN_POS = (5150.2509765625, 4947.73095703125, 18.33122253418)

    @classmethod
    def setUpClass(cls):
        from wulfram.server import WulframServer
        cls.server = WulframServer(host="127.0.0.1", port=0)

    def _brute(self, start, end):
        s = self.server
        seg = math.dist(start, end)
        best = None
        for oid, b in s._building_entities.items():
            hit = s._raycast_static_building_candidate(b, oid, start, end, seg_len=seg)
            if hit is not None and (best is None or hit[3] < best[3]):
                best = hit
        return best

    def test_quadtree_matches_brute_force_for_every_building(self):
        s = self.server
        self.assertGreater(len(s._building_entities), 8)  # the index really splits
        s._rebuild_static_world_raycast_index()
        self.assertIsNotNone(s._static_world_raycast_root.children)
        point_hits = 0
        for oid, b in s._building_entities.items():
            for ang in (0.0, 45.0, 90.0, 135.0):
                c, sn = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                start = (b.x - 60.0 * c, b.y - 60.0 * sn, b.z)
                end = (b.x + 60.0 * c, b.y + 60.0 * sn, b.z)
                tree = s._raycast_static_buildings(start, end)
                brute = self._brute(start, end)
                self.assertEqual(tree and tree[2], brute and brute[2], (oid, ang))
                self.assertIsNotNone(tree, (oid, ang))
            # Point query (zero-length ray) uses the same index; it must agree
            # with a linear scan (a mesh centre may legitimately be empty).
            point = tree[1]  # the surface point the last ray hit
            tree_pt = s._raycast_static_buildings(point, point)
            brute_pt = {o for o, bb in s._building_entities.items()
                        if s._point_hits_static_building(bb, point) is not None}
            if tree_pt is None:
                self.assertEqual(brute_pt, set(), oid)
            else:
                self.assertIn(tree_pt[2], brute_pt, oid)
                point_hits += 1
        self.assertGreater(point_hits, 0)

    def _fly_pulse(self, start, vel, steps=40, rate=15.0):
        """Step a pulse shell like the server's update loop (15 Hz) and return the first world hit."""
        s = self.server
        pos = s._to_client_pos(start)
        for _ in range(steps):
            nxt = tuple(pos[i] + vel[i] / rate for i in range(3))
            hit = s._check_projectile_world_hit(pos, nxt)
            if hit is not None:
                return hit
            pos = nxt
        return None

    def _direct_hit_damage(self, env):
        from wulfram2_protocol.entities import Projectile
        s = self.server
        b = s._building_entities[self.GUN_OID]
        self.assertEqual(int(b.entity_type), int(EntityType.GUN_TURRET))
        self.assertAlmostEqual(b.x, self.GUN_POS[0], places=2)
        self.assertAlmostEqual(b.y, self.GUN_POS[1], places=2)
        start = (b.x - 120.0, b.y, b.z)
        hit = self._fly_pulse(start, (210.0, 0.0, 0.0))
        self.assertIsNotNone(hit)
        self.assertIn(hit[0], {"building", "building-aabb"})
        self.assertEqual(hit[2], self.GUN_OID)
        attacker = SimpleNamespace(client_id=9, session=SimpleNamespace(username="shooter", team_id=1))
        s._building_health[self.GUN_OID] = 525.0
        s._building_max_health[self.GUN_OID] = 525.0
        proj = Projectile(entity_id=57, entity_type=EntityType.PULSE_SHELL, owner_id=9, team=1,
                          pos=hit[1], vel=(210.0, 0.0, 0.0), spawn_time=0.0, lifetime=6.0)
        with patch.dict(os.environ, env):
            if not env:
                os.environ.pop(cp.PROFILE_ENV, None)
            s._apply_building_damage(self.GUN_OID, proj, attacker, hit[1])
        return 525.0 - s._building_health[self.GUN_OID]

    def test_direct_pulse_hit_on_static_gun_turret_upstream(self):
        self.assertAlmostEqual(self._direct_hit_damage(UPSTREAM), cp.PULSE_DIRECT_HP, places=3)

    def test_direct_pulse_hit_on_static_gun_turret_legacy(self):
        self.assertAlmostEqual(self._direct_hit_damage({}), 50.0, places=3)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=1).result
    sys.exit(0 if result.wasSuccessful() else 1)
