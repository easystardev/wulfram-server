"""Upstream-parity turret runtime (``WULFRAM_COMBAT_PROFILE=upstream-2026-10``).

Replaces the legacy placeholder turret AI with the behaviour measured on the
community server (see ``combat_profile`` for every number and its source):

* Gun turret: hitscan, 30 HP every 0.25 s, 0-475 u, terrain LOS.
* Flak turret (SENSOR_BUILDING, save code ``s``): 300-900 u band, a volley of
  three real 140 u/s shells every 6 s, centre shell aimed at the target's
  current position (no leading), sides fanned +-5.1 deg.
* Missile launcher: 900-1400 u band, one real hunter every 2 s; it climbs
  straight up at 95 u/s for 2.15 s, snaps onto the target, then turns slowly.

Wire shape (mirrors the 2026-10-03 capture, decoded with the strict
``ogdecode.py``):

* Shell/hunter creation: UDP UPDATE_ARRAY, mask 0x12f = DEFINITION (type,
  variant 1, team, static 1) + pos + vel + rot (elevation, 0, yaw) + health
  (1.0) + target oid.
* Hunter course changes: mask 0x00c (vel + rot). Upstream also streams these
  (and vel-only 0x004 for flak) at 20 Hz; we only send real changes to avoid the
  per-shell packet flood that hurt OG clients before (constant-velocity flak
  needs none: the client integrates the same straight line).
* Removal: DELETE_OBJECT with the effects byte set, for hits and expiry alike.
* Gun-turret shot: TRANSIENT_ARRAY {type 5, has_pos 0, u32 turret oid,
  has_entity 1, entity 0}, i.e. the upstream event, sent through the existing
  transient-FX gate.

Simulation runs in server coordinates; ``_to_client_pos`` is applied only when
encoding packets.
"""
from __future__ import annotations

import math
import os
import struct
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from wulfram2_protocol.entities import EntityType

from . import combat_profile as cp
from . import tutorial_runtime
from .codec import BitWriter
from .packets import (
    FX_TURRET_FIRE,
    VEC_POS_MAX,
    VEC_POS_RANGE,
    VEC_ROT_MAX,
    VEC_ROT_RANGE,
    VEC_VEL_MAX,
    VEC_VEL_RANGE,
    _encode_health_bits,
    _write_local_player_state,
    build_transient_array,
    compress_value,
)

MASK_DEFINITION = 1 << 0
MASK_POS = 1 << 1
MASK_VEL = 1 << 2
MASK_ROT = 1 << 3
MASK_HEALTH = 1 << 5
MASK_TARGET = 1 << 8
UPSTREAM_SPAWN_MASK = (MASK_DEFINITION | MASK_POS | MASK_VEL | MASK_ROT
                       | MASK_HEALTH | MASK_TARGET)  # 0x12f
BASIC_SPAWN_MASK = MASK_DEFINITION | MASK_POS | MASK_VEL | MASK_ROT  # 0x00f
STEER_MASK = MASK_VEL | MASK_ROT  # 0x00c

STEP_S = 0.05            # projectile integration step (upstream update cadence, CAP)
MAX_CATCH_UP_S = 0.5     # never integrate more than this per update call
STEER_RESEND_DEG = 0.5   # resend a hunter course only when it changed this much
STEER_RESEND_MIN_S = 0.2
OID_BASE = 50000         # turret projectiles: 50000-59999 (players 20000+, buildings 30000+, cargo 40000+)
OID_SPAN = 10000

GUN = "gun"
FLAK = "flak"
LAUNCHER = "launcher"
TURRET_KINDS = {
    EntityType.GUN_TURRET: GUN,
    EntityType.SENSOR_BUILDING: FLAK,  # save code 's' = flak turret (BEH ring 0x1d)
    EntityType.LAUNCHER: LAUNCHER,
}


def build_turret_projectile_packet(
    entity_id: int,
    entity_type: int,
    team: int,
    pos: tuple,
    vel: tuple,
    tick: int,
    *,
    mask: int = UPSTREAM_SPAWN_MASK,
    target_oid: int = 0,
    health: float = 1.0,
    variant: int = 1,
    is_static: bool = True,
    rot: Optional[tuple] = None,
    include_local_state: bool = False,
    **local_state_kwargs: Any,
) -> bytes:
    """UPDATE_ARRAY (0x0E) with one turret projectile record.

    ``pos`` must already be in client coordinates. Field widths follow our own
    TRANSLATION table (the client reads every width from it).
    """
    bw = BitWriter()
    _write_local_player_state(bw, include_local_state, **local_state_kwargs)
    bw.write_bits(8, 1)
    bw.write_bits(32, int(entity_id) & 0xFFFFFFFF)
    bw.write_bits(1, 0)  # not manned
    bw.write_bits(10, int(mask) & 0x3FF)
    bw.write_bits(16, 0)  # quantizer index / bank 0
    if mask & MASK_DEFINITION:
        bw.write_bits(8, int(entity_type) & 0xFF)
        bw.write_bits(8, int(variant) & 0xFF)
        bw.write_bits(8, int(team) & 0xFF)
        bw.write_bits(1, 1 if is_static else 0)
    if mask & MASK_POS:
        bw.write_bits(4, 15)
        for v in pos:
            bw.write_bits(16, compress_value(float(v), VEC_POS_MAX, VEC_POS_RANGE))
    if mask & MASK_VEL:
        bw.write_bits(4, 15)
        for v in vel:
            bw.write_bits(16, compress_value(float(v), VEC_VEL_MAX, VEC_VEL_RANGE))
    if mask & MASK_ROT:
        if rot is None:
            rot = cp.upstream_projectile_rotation(vel)
        bw.write_bits(4, 15)
        for v in rot:
            bw.write_bits(16, compress_value(float(v), VEC_ROT_MAX, VEC_ROT_RANGE))
    if mask & MASK_HEALTH:
        bw.write_bits(10, _encode_health_bits(float(health), total_bits=10))
    if mask & MASK_TARGET:
        bw.write_bits(32, int(target_oid) & 0xFFFFFFFF)
    return b"\x0E" + struct.pack(">I", int(tick) & 0xFFFFFFFF) + bw.get_bytes()


def build_gun_turret_fire_transient(turret_oid: int) -> bytes:
    """The upstream gun-turret shot event (CAP: ``0d 01 0005 ...``)."""
    return build_transient_array([{
        "type": FX_TURRET_FIRE,
        "pos": None,
        "source_oid": int(turret_oid),
        "has_entity": True,
        "entity_id": 0,
    }])


@dataclass
class TurretProjectile:
    entity_id: int
    kind: str                     # FLAK or LAUNCHER
    entity_type: EntityType
    team: int
    turret_oid: int
    turret_name: str
    target: Any                   # ClientContext (or None)
    target_oid: int
    pos: tuple                    # server coordinates
    vel: tuple
    spawn_t: float
    lifetime: float
    damage_hp: float
    hit_radius: float
    snapped: bool = False         # hunters: left the vertical climb
    last_sent_vel: tuple = (0.0, 0.0, 0.0)
    last_sent_t: float = 0.0
    hp: float = 0.0               # > 0: can be shot down (launcher hunters, INF 25 HP)


@dataclass
class EngagementStats:
    shots: Counter = field(default_factory=Counter)        # kind -> shots/shells/hunters
    hits: Counter = field(default_factory=Counter)         # kind -> hits on vehicles
    damage_hp: Counter = field(default_factory=Counter)    # kind -> HP dealt
    expired: Counter = field(default_factory=Counter)      # kind -> lifetime ends
    terrain: Counter = field(default_factory=Counter)      # kind -> terrain impacts
    shot_down: Counter = field(default_factory=Counter)    # kind -> destroyed by player fire

    def as_dict(self) -> dict:
        return {name: dict(getattr(self, name))
                for name in ("shots", "hits", "damage_hp", "expired", "terrain", "shot_down")}


class UpstreamTurretRuntime:
    """Fires defences and flies their projectiles. Driven by ``update(now)``."""

    def __init__(self, server: Any, *, network: bool = True):
        self.server = server
        self.network = network
        self.next_fire: dict[int, float] = {}
        self.projectiles: list[TurretProjectile] = []
        self.stats = EngagementStats()
        self._last_step: Optional[float] = None
        self._next_oid = OID_BASE
        self.require_power = os.environ.get("WULFRAM_TURRET_REQUIRE_POWER", "0") == "1"
        self.neutral_turrets_fire = os.environ.get("WULFRAM_NEUTRAL_TURRETS_FIRE", "0") == "1"
        # The gun-turret shot event is byte-identical to upstream's, but every
        # TRANSIENT_ARRAY stays behind the existing WULFRAM_REMOTE_TRANSIENT_FX
        # gate (0x0D crashed OG clients in the past) unless this opts in.
        self.force_turret_fx = os.environ.get("WULFRAM_UPSTREAM_TURRET_FX", "0") == "1"
        wire = os.environ.get("WULFRAM_TURRET_SHELL_WIRE", "upstream").strip().lower()
        self.spawn_mask = BASIC_SPAWN_MASK if wire == "basic" else UPSTREAM_SPAWN_MASK

    # ------------------------------------------------------------------ util
    def _alloc_oid(self) -> int:
        live = {p.entity_id for p in self.projectiles}
        for _ in range(OID_SPAN):
            oid = self._next_oid
            self._next_oid = OID_BASE + ((self._next_oid - OID_BASE + 1) % OID_SPAN)
            if oid not in live:
                return oid
        raise RuntimeError("turret projectile oid range exhausted")

    @staticmethod
    def _alive(client: Any) -> bool:
        session = getattr(client, "session", None)
        return (
            bool(getattr(client, "running", True))
            and bool(getattr(session, "in_game", True))
            and not bool(getattr(client, "observer_transition", False))
            and float(getattr(client, "player_health", 0.0) or 0.0) > 0.0
        )

    @staticmethod
    def _team(client: Any) -> int:
        session = getattr(client, "session", None)
        return int(getattr(session, "team_id", 0) or 0)

    @staticmethod
    def _client_oid(client: Any) -> int:
        session = getattr(client, "session", None)
        return int(getattr(session, "entity_id", 0) or getattr(client, "entity_id", 0) or 0)

    def _powered(self, oid: int, building: Any) -> bool:
        """Optional (GUESS) power model: a live friendly power cell within 280 u."""
        if not self.require_power:
            return True
        health = getattr(self.server, "_building_health", {}) or {}
        for other_oid, other in (getattr(self.server, "_building_entities", {}) or {}).items():
            if int(getattr(other, "entity_type", -1)) != int(EntityType.ENERGY_BUILDING):
                continue
            if int(getattr(other, "team_id", -1)) != int(building.team_id):
                continue
            if float(health.get(other_oid, 1.0)) <= 0.0:
                continue
            if other_oid in (getattr(self.server, "_building_crate_oids", set()) or set()):
                continue
            if cp.horizontal_distance((other.x, other.y), (building.x, building.y)) <= cp.POWER_CELL_RADIUS:
                return True
        return False

    def _schedule(self, oid: int, now: float, period: float) -> None:
        """Phase-locked cadence while engaging; restart the clock after a pause."""
        previous = self.next_fire.get(oid)
        if previous is not None and now - previous < period:
            self.next_fire[oid] = previous + period
        else:
            self.next_fire[oid] = now + period

    # ---------------------------------------------------------------- update
    def update(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else float(now)
        clients = list(self.server._snapshot_in_game_clients() or [])
        self._fire_turrets(now, clients)
        self._step_projectiles(now, clients)

    def _fire_turrets(self, now: float, clients: list) -> None:
        server = self.server
        buildings = getattr(server, "_building_entities", None) or {}
        if not buildings or not clients:
            return
        health = getattr(server, "_building_health", {}) or {}
        crates = getattr(server, "_building_crate_oids", set()) or set()
        for oid, b in buildings.items():
            kind = TURRET_KINDS.get(getattr(b, "entity_type", None))
            if kind is None:
                continue
            if float(health.get(oid, 0.0)) <= 0.0 or oid in crates:
                continue
            team = int(getattr(b, "team_id", 0) or 0)
            if team == 0 and not self.neutral_turrets_fire:
                continue  # INF: the neutral turrets were never seen firing upstream
            if now < self.next_fire.get(oid, float("-inf")):
                continue
            if not self._powered(oid, b):
                continue
            if kind == GUN:
                lo, hi, los = cp.GUN_TURRET_MIN_RANGE, cp.GUN_TURRET_MAX_RANGE, True
            elif kind == FLAK:
                lo, hi, los = cp.FLAK_MIN_RANGE, cp.FLAK_MAX_RANGE, True
            else:
                lo, hi, los = cp.LAUNCHER_MIN_RANGE, cp.LAUNCHER_MAX_RANGE, False  # GUESS: guided, no LOS
            candidates = [
                (c, c.player_pos) for c in clients
                if self._alive(c) and self._team(c) != team
            ]
            los_fn = (lambda c, _b=b: server._turret_has_terrain_line_of_sight(_b, c)) if los else None
            picked = cp.pick_nearest_target((b.x, b.y), candidates, lo, hi, los_fn)
            if picked is None:
                continue
            target, distance = picked
            if tutorial_runtime.suppress_turret_fire(server, target, int(oid)):
                continue
            name = getattr(getattr(b, "entity_type", None), "name", str(getattr(b, "entity_type", "?")))
            if kind == GUN:
                self._schedule(oid, now, cp.GUN_TURRET_PERIOD_S)
                self._fire_gun(oid, b, name, target, now)
            elif kind == FLAK:
                self._schedule(oid, now, cp.FLAK_VOLLEY_PERIOD_S)
                self._fire_flak(oid, b, name, target, now, clients)
            else:
                self._schedule(oid, now, cp.LAUNCHER_PERIOD_S)
                self._fire_hunter(oid, b, name, target, now, clients)

    # ------------------------------------------------------------ gun turret
    def _fire_gun(self, oid: int, b: Any, name: str, target: Any, now: float) -> None:
        self.stats.shots[GUN] += 1
        if self.network:
            self._broadcast_transient(build_gun_turret_fire_transient(oid))
        hp = cp.GUN_TURRET_DAMAGE_HP
        self._damage(target, hp, oid, name, now, GUN)

    def _broadcast_transient(self, pkt: bytes) -> None:
        server = self.server
        if not pkt or not getattr(server, "udp_handler", None):
            return
        for client in server._snapshot_in_game_clients():
            if not (self.force_turret_fx or server._transient_fx_allowed_for_client(client)):
                continue
            addr = getattr(client.session, "udp_addr", None)
            if addr:
                server.udp_handler.send_to(pkt, addr)

    # ------------------------------------------------------------------ flak
    def _fire_flak(self, oid: int, b: Any, name: str, target: Any, now: float, clients: list) -> None:
        muzzle = (float(b.x), float(b.y), float(b.z) + cp.FLAK_MUZZLE_DZ)
        target_pos = tuple(float(v) for v in target.player_pos[:3])
        velocities = cp.flak_volley_velocities(muzzle, target_pos)
        half = (len(velocities) - 1) / 2.0
        centre_yaw = math.atan2(target_pos[1] - muzzle[1], target_pos[0] - muzzle[0])
        side = (-math.sin(centre_yaw), math.cos(centre_yaw))
        for i, vel in enumerate(velocities):
            offset = (i - half) * cp.FLAK_SHELL_SIDE_SPACING
            pos = (muzzle[0] + side[0] * offset, muzzle[1] + side[1] * offset, muzzle[2])
            self._launch(TurretProjectile(
                entity_id=self._alloc_oid(), kind=FLAK, entity_type=EntityType.FLAK_SHELL,
                team=int(b.team_id), turret_oid=int(oid), turret_name=name,
                target=target, target_oid=self._client_oid(target),
                pos=pos, vel=vel, spawn_t=now, lifetime=cp.FLAK_SHELL_LIFETIME_S,
                damage_hp=cp.FLAK_SHELL_DAMAGE_HP, hit_radius=cp.FLAK_HIT_RADIUS,
            ), clients)

    # -------------------------------------------------------------- launcher
    def _fire_hunter(self, oid: int, b: Any, name: str, target: Any, now: float, clients: list) -> None:
        pos = (float(b.x), float(b.y), float(b.z) + cp.HUNTER_LAUNCH_DZ)
        self._launch(TurretProjectile(
            entity_id=self._alloc_oid(), kind=LAUNCHER, entity_type=EntityType.HUNTER,
            team=int(b.team_id), turret_oid=int(oid), turret_name=name,
            target=target, target_oid=self._client_oid(target),
            pos=pos, vel=(0.0, 0.0, cp.HUNTER_SPEED), spawn_t=now,
            lifetime=cp.HUNTER_LIFETIME_S, damage_hp=cp.HUNTER_DAMAGE_HP,
            hit_radius=cp.HUNTER_HIT_RADIUS, hp=cp.HUNTER_HP,
        ), clients)

    # ------------------------------------------------------------- shootdown
    def shoot_in_lane(self, shooter_team: int, origin: Sequence[float], yaw: float, *,
                      range_limit: float, lane_radius: float, max_along: float,
                      shot_dt: float) -> Optional[dict]:
        """Credit one player autocannon shot to the nearest enemy hunter in the lane.

        CAP: upstream defenders shot launcher hunters down with the autocannon (health fell
        in 0.1 s steps; the shooter's target field held the hunter's oid). The lane is judged
        in the ground plane, like the server's vehicle lane (no aim pitch on the server).
        Only hunters nearer than ``max_along`` (the vehicle in the lane, if any) are eligible.
        """
        fx, fy = math.cos(yaw), math.sin(yaw)
        best = None
        for proj in self.projectiles:
            if proj.hp <= 0.0 or proj.team == shooter_team:
                continue
            rx, ry = proj.pos[0] - origin[0], proj.pos[1] - origin[1]
            along = rx * fx + ry * fy
            if along < 0.0 or along > min(range_limit, max_along):
                continue
            if rx * rx + ry * ry - along * along > lane_radius * lane_radius:
                continue
            if best is None or along < best[0]:
                best = (along, proj)
        if best is None:
            return None
        proj = best[1]
        distance = math.dist(proj.pos, tuple(origin[:3]))
        hp = cp.autocannon_shot_hp(distance, shot_dt)
        proj.hp -= hp
        destroyed = proj.hp <= 0.0
        if destroyed:
            self.stats.shot_down[proj.kind] += 1
            self._remove(proj, "shot down", self.server._snapshot_in_game_clients() if self.network else [])
        return {"oid": proj.entity_id, "hp": hp, "remaining": max(0.0, proj.hp), "destroyed": destroyed}

    # ----------------------------------------------------------- projectiles
    def _launch(self, proj: TurretProjectile, clients: list) -> None:
        self.projectiles.append(proj)
        self.stats.shots[proj.kind] += 1
        proj.last_sent_vel = proj.vel
        proj.last_sent_t = proj.spawn_t
        if self.network:
            self._send_projectile(proj, self.spawn_mask, clients)

    def _send_projectile(self, proj: TurretProjectile, mask: int, clients: list) -> None:
        server = self.server
        udp = getattr(server, "udp_handler", None)
        if udp is None:
            return
        client_pos = server._to_client_pos(proj.pos)
        for viewer in clients:
            session = getattr(viewer, "session", None)
            addr = getattr(session, "udp_addr", None)
            if not addr or not getattr(session, "translation_ack_received", False):
                continue
            if not server._projectile_packets_allowed_for_client(viewer):
                continue
            tick = server._get_network_tick(viewer)
            include_local_state, local_kwargs = server._get_projectile_local_state_for_viewer(viewer)
            rot = None
            if self.spawn_mask == BASIC_SPAWN_MASK:
                from .weapons import _projectile_rotation_from_velocity
                rot = _projectile_rotation_from_velocity(proj.vel)
            pkt = build_turret_projectile_packet(
                proj.entity_id, int(proj.entity_type), proj.team, client_pos, proj.vel, tick,
                mask=mask, target_oid=proj.target_oid, rot=rot,
                include_local_state=include_local_state, **local_kwargs,
            )
            udp.send_to(pkt, addr)

    def _remove(self, proj: TurretProjectile, reason: str, clients: list) -> None:
        if proj in self.projectiles:
            self.projectiles.remove(proj)
        if not self.network:
            return
        server = self.server
        tick = server._get_network_tick(clients[0]) if clients else 0
        server._broadcast_projectile_delete(proj, tick, with_effects=True, reason=reason)

    def _damage(self, target: Any, hp: float, oid: int, name: str, now: float, kind: str) -> None:
        fraction = cp.hp_to_health_fraction(hp, int(getattr(target, "entity_type", 0) or 0))
        self.stats.hits[kind] += 1
        self.stats.damage_hp[kind] += hp
        self.server._apply_turret_hit(target, fraction, oid=int(oid), source_name=name, now=now,
                                      hp=hp)

    def _guide_hunter(self, proj: TurretProjectile, t: float, dt: float) -> None:
        age = t - proj.spawn_t
        if age < cp.HUNTER_CLIMB_S:
            proj.vel = (0.0, 0.0, cp.HUNTER_SPEED)
            return
        target = proj.target
        if target is None or not self._alive(target):
            return  # lost target: fly straight on
        want = cp.direction_to(proj.pos, target.player_pos)
        if not proj.snapped:
            proj.snapped = True  # CAP: an instant snap onto the target bearing
            proj.vel = tuple(w * cp.HUNTER_SPEED for w in want)
            return
        proj.vel = cp.steer_toward(proj.vel, want, math.radians(cp.HUNTER_TURN_RATE_DEG_S) * dt,
                                   cp.HUNTER_SPEED)

    def _step_projectiles(self, now: float, clients: list) -> None:
        if self._last_step is None or not self.projectiles:
            self._last_step = now
            if not self.projectiles:
                return
        elapsed = min(now - self._last_step, MAX_CATCH_UP_S)
        steps = int(elapsed / STEP_S)
        if steps <= 0:
            return
        t = self._last_step
        terrain = getattr(self.server, "_terrain_grid_collision", None)
        for _ in range(steps):
            t += STEP_S
            for proj in list(self.projectiles):
                if proj not in self.projectiles:
                    continue  # shot down by player fire on another thread
                if t <= proj.spawn_t:
                    continue  # launched later in this update; not airborne yet
                if t - proj.spawn_t >= proj.lifetime:
                    self.stats.expired[proj.kind] += 1
                    self._remove(proj, "expired", clients)
                    continue
                if proj.kind == LAUNCHER:
                    self._guide_hunter(proj, t, STEP_S)
                p0 = proj.pos
                p1 = (p0[0] + proj.vel[0] * STEP_S, p0[1] + proj.vel[1] * STEP_S,
                      p0[2] + proj.vel[2] * STEP_S)
                hit = None
                for client in clients:
                    if not self._alive(client) or self._team(client) == proj.team:
                        continue
                    if cp.segment_point_distance(p0, p1, client.player_pos) <= proj.hit_radius:
                        hit = client
                        break
                if hit is not None:
                    proj.pos = p1
                    self._damage(hit, proj.damage_hp, proj.turret_oid, proj.turret_name, t, proj.kind)
                    self._remove(proj, "hit", clients)
                    continue
                if terrain is not None and terrain.raycast(p0, p1) is not None:
                    self.stats.terrain[proj.kind] += 1
                    self._remove(proj, "terrain", clients)
                    continue
                proj.pos = p1
                if proj.kind == LAUNCHER and self.network:
                    turned = math.degrees(cp.angle_between(proj.vel, proj.last_sent_vel))
                    if turned >= STEER_RESEND_DEG and t - proj.last_sent_t >= STEER_RESEND_MIN_S:
                        proj.last_sent_vel = proj.vel
                        proj.last_sent_t = t
                        self._send_projectile(proj, STEER_MASK, clients)
        self._last_step = self._last_step + steps * STEP_S
