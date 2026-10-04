"""Combat profile switch: legacy placeholder combat vs. measured upstream parity.

``WULFRAM_COMBAT_PROFILE`` selects the profile:

* ``legacy`` (default, also ``""``/``default``): the long-standing placeholder
  turret AI, weapon damage and speeds. Nothing in this module changes behaviour
  unless the upstream profile is selected.
* ``upstream-2026-10`` (alias ``upstream``): values measured from the
  wulfram3.com community-server captures of 2026-09-18 and 2026-10-03
  (``artifacts/basewar-2026-10-03/analysis``) and from that server's BEHAVIOR
  packet.

Every constant carries a source tag:

* CAP: measured from the upstream captures (highest authority).
* BEH: sent by the upstream server in BEHAVIOR (0x24).
* INF: inferred from CAP/BEH data.
* GUIDE: 2004 community manual (only where nothing else exists).
* GUESS: modelling assumption; nothing measured.

This module is pure (no server state) so the behaviour can be unit tested.
"""
from __future__ import annotations

import math
import os
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar

from wulfram2_protocol.entities import EntityType

PROFILE_ENV = "WULFRAM_COMBAT_PROFILE"
LEGACY = "legacy"
UPSTREAM_2026_10 = "upstream-2026-10"

_ALIASES = {
    "": LEGACY,
    "0": LEGACY,
    "default": LEGACY,
    "legacy": LEGACY,
    "upstream": UPSTREAM_2026_10,
    "upstream-2026-10": UPSTREAM_2026_10,
}
_warned_unknown: set[str] = set()


def active_profile(env: Optional[Mapping[str, str]] = None) -> str:
    """Return the selected combat profile name (read on every call)."""
    source = os.environ if env is None else env
    raw = str(source.get(PROFILE_ENV, "") or "").strip().lower()
    profile = _ALIASES.get(raw)
    if profile is None:
        if raw not in _warned_unknown:
            _warned_unknown.add(raw)
            print(f"[COMBAT] Unknown {PROFILE_ENV}={raw!r}; using {LEGACY!r}")
        return LEGACY
    return profile


def is_upstream(env: Optional[Mapping[str, str]] = None) -> bool:
    return active_profile(env) == UPSTREAM_2026_10


# ---------------------------------------------------------------------------
# Hit points (BEH Section 3 u32 -> EntityTypeInfo +0x48), indexed by entity type.
# ---------------------------------------------------------------------------
UPSTREAM_ENTITY_HP: Tuple[int, ...] = (
    500, 330, 1000, 1000, 1000,      # 0 tank, 1 scout, 2-4 other vehicles
    50, 50, 1000, 50, 1000,          # 5 flak shell, 6 pulse, 7 short msl, 8 hunter, 9 heavy msl
    50, 50, 50, 50, 1000,            # 10 mine, 11 piercer, 12 thumper, 13 caltrop, 14
    1000, 1000, 50, 1000, 90,        # 15, 16, 17 flare, 18, 19 cargo box
    1000, 1000, 50, 50, 1000,        # 20 uplink, 21, 22 torpedo, 23, 24
    700, 600, 1000, 1000, 550,       # 25 power cell, 26 refuel, 27 repair, 28, 29 flak turret
    525, 1000, 500, 300, 1000,       # 30 gun turret, 31, 32 launcher, 33 skypump, 34
    250, 1000, 1000, 1000,           # 35 darklight, 36-38
)
assert len(UPSTREAM_ENTITY_HP) == 39


def entity_max_hp(entity_type: int, default: float = 500.0) -> float:
    """Upstream max HP for an entity type (BEH)."""
    try:
        index = int(entity_type)
    except (TypeError, ValueError):
        return float(default)
    if 0 <= index < len(UPSTREAM_ENTITY_HP):
        return float(UPSTREAM_ENTITY_HP[index])
    return float(default)


def hp_to_health_fraction(hp: float, entity_type: int) -> float:
    """Convert absolute HP damage to the server's 0..1 player-health scale."""
    max_hp = entity_max_hp(entity_type)
    if max_hp <= 0.0:
        return 0.0
    return float(hp) / max_hp


# ---------------------------------------------------------------------------
# Vehicle speed caps (BEH Section 6, changed 96/106 -> 120/140 between the
# 2026-09-18 and 2026-10-03 captures; the relay 0x0C block mirrors it).
# ---------------------------------------------------------------------------
TANK_MAX_VELOCITY = 120.0   # BEH tank values[3]
SCOUT_MAX_VELOCITY = 140.0  # BEH scout values[4]


def tank_governor_max_velocity(default: float) -> float:
    """Max velocity for the server's tank longitudinal governor."""
    return TANK_MAX_VELOCITY if is_upstream() else float(default)


# ---------------------------------------------------------------------------
# Defences
# ---------------------------------------------------------------------------
GUN_TURRET_MAX_RANGE = 475.0      # BEH 0x006791a4 ring; CAP hits 215-467 u, none beyond
GUN_TURRET_MIN_RANGE = 0.0        # CAP: hits from 215 u; BEH draws one circle only
GUN_TURRET_PERIOD_S = 0.25        # CAP: 30-HP steps at exact 0.25 s spacing
GUN_TURRET_DAMAGE_HP = 30.0       # CAP: 29.8-30.4 HP steps; no falloff 215-467 u

FLAK_MIN_RANGE = 300.0            # BEH 0x00679190 (inner ring); dead zone inside
FLAK_MAX_RANGE = 900.0            # BEH 0x00679194; CAP volleys at 360-865 u
FLAK_VOLLEY_PERIOD_S = 6.0        # CAP: volley gaps 5.96-6.05 s
FLAK_SHELLS_PER_VOLLEY = 3        # CAP
FLAK_FAN_DEG = 5.1                # CAP: centre shell 0.0 deg, sides +-5.1 deg
FLAK_SHELL_SPEED = 140.0          # CAP: all 177 shells
FLAK_SHELL_LIFETIME_S = 7.0       # CAP: 7.0 s (980 u)
FLAK_SHELL_DAMAGE_HP = 37.5       # CAP: 37.2-37.7
FLAK_MUZZLE_DZ = 5.0              # CAP: shells spawn 4-5 u above the turret
FLAK_SHELL_SIDE_SPACING = 1.0     # CAP: volley shells spawn ~1 u apart
# GUESS: proximity/contact radius. The CAP "35.4-37.7 u from the extrapolated
# point" figure is most likely one 0.25 s server step of travel (140 x 0.25 = 35),
# not a fuse radius; we use the server's vehicle hit sphere instead.
FLAK_HIT_RADIUS = 15.0

LAUNCHER_MIN_RANGE = 900.0        # BEH 0x0067919c; CAP 33/33 launches had a target 900.5-1399.6 u away
LAUNCHER_MAX_RANGE = 1400.0       # BEH 0x006791a0
LAUNCHER_PERIOD_S = 2.0           # CAP: interval >= 2 s (gaps 2, 3, 4 s seen); GUESS: fixed 2 s
HUNTER_SPEED = 95.0               # CAP: |v| = 95.0 throughout
HUNTER_LAUNCH_DZ = 10.0           # CAP: hunters spawn at turret z + 10
HUNTER_CLIMB_S = 2.15             # CAP: vertical (0,0,95) for 2.13-2.16 s, then a snap onto the target
HUNTER_TURN_RATE_DEG_S = 3.0      # GUESS from CAP post-snap turn p50 0.3 / p90 2.7 / p99 7.3 deg/s
HUNTER_LIFETIME_S = 20.0          # GUESS cap; CAP lifetimes 9.7-19.3 s (terrain impact ends most)
HUNTER_DAMAGE_HP = 200.0          # GUIDE (unobserved upstream: 0 hits in 79 hunters)
HUNTER_HIT_RADIUS = 15.0          # GUESS: the server's vehicle hit sphere

TURRET_MUZZLE_DZ = 8.0            # our server's gun-turret LOS muzzle height (kept)
POWER_CELL_RADIUS = 280.0         # BEH 0x00679180 (INF: power-cell coverage radius)

# ---------------------------------------------------------------------------
# Player weapons (server-owned numbers; BEHAVIOR carries none of these).
# ---------------------------------------------------------------------------
PULSE_SPEED = 210.0               # CAP (plus the shooter's velocity, CAP)
PULSE_LIFETIME_S = 6.0            # CAP: life caps at ~6.0 s (max 6.06)
PULSE_DIRECT_HP = 300.0           # CAP: 299.9 / 300.4 / 299.9 on tanks; 299.7 on a flak turret
PULSE_REFIRE_S = 0.6              # BEH tank slot 4: 600 ms
PULSE_ENERGY_PCT = 10.0           # CAP: -100/1023 per shell
# INF: piecewise-linear fit to CAP splash samples (centre distances +-10 u):
# ~300 within ~10 u, 130-140 at 10-20 u, 50-80 at 35-55 u, 15-27 at 60-75 u,
# none seen beyond ~75 u. One building hit ~9 u off-centre did 181.
PULSE_SPLASH_TABLE: Tuple[Tuple[float, float], ...] = (
    (0.0, 300.0),
    (8.0, 300.0),
    (12.0, 180.0),
    (18.0, 135.0),
    (45.0, 65.0),
    (67.0, 21.0),
    (76.0, 0.0),
)
PULSE_SPLASH_RADIUS = PULSE_SPLASH_TABLE[-1][0]

PIERCER_SPEED = 125.0             # CAP
PIERCER_LIFETIME_S = 9.6          # CAP: life up to 9.6 s
PIERCER_DAMAGE_HP = 108.0         # CAP: 107.6-108.1 (GUIDE agrees)
PIERCER_REFIRE_S = 2.25           # BEH slot 5
PIERCER_TURN_RATE_DEG_S = 60.0    # GUESS (CAP shows homing, rate not measured)
MISSILE_LOCK_RANGE = 1000.0       # BEH 0x006791ac targeting max distance

PLAYER_HUNTER_SPEED = 95.0        # CAP (plus the shooter's velocity, CAP)
PLAYER_HUNTER_LIFETIME_S = 14.5   # CAP: flight ~10-14.5 s
PLAYER_HUNTER_DAMAGE_HP = 200.0   # GUIDE (unobserved upstream)
PLAYER_HUNTER_REFIRE_S = 1.5      # BEH slot 8

THUMPER_SPEED = 250.0             # CAP: ~240-260 u/s (2 samples)
THUMPER_GRAVITY = 24.0            # CAP: vz slope -23.4 / -24.0
THUMPER_REFIRE_S = 3.0            # BEH slot 6
CALTROP_DAMAGE_HP = 50.0          # CAP: 49.9 / 54.3 on contact
CALTROP_LIFETIME_S = 20.0         # CAP
MINE_DAMAGE_HP = 300.0            # GUIDE (unobserved upstream)
MINE_REFIRE_S = 1.5               # BEH slot 9

AUTOCANNON_DPS_AT_ZERO = 78.9     # CAP fit over 46 single-attacker streams, 18-494 u
AUTOCANNON_DPS_SLOPE = 0.108      # CAP fit (DPS per unit distance)
AUTOCANNON_MAX_RANGE = AUTOCANNON_DPS_AT_ZERO / AUTOCANNON_DPS_SLOPE  # INF: fit reaches 0 at ~730 u
AUTOCANNON_LANE_RADIUS = 12.0     # our server's hitscan lane half-width (kept)
AUTOCANNON_MAX_SHOT_DT = 0.2      # GUESS: cap on the time credited to one shot

ENERGY_REGEN_PCT_S = 2.9          # CAP: +1..2/1023 per 0.05 s
AUTOCANNON_NET_ENERGY_PCT_S = 2.9  # CAP: net -2.9 %/s while firing (regen keeps running)
AUTOCANNON_GROSS_ENERGY_PCT_S = AUTOCANNON_NET_ENERGY_PCT_S + ENERGY_REGEN_PCT_S


def autocannon_dps(distance: float) -> float:
    """Measured autocannon DPS at ``distance`` (CAP linear fit, >= 0)."""
    return max(0.0, AUTOCANNON_DPS_AT_ZERO - AUTOCANNON_DPS_SLOPE * max(0.0, float(distance)))


def autocannon_shot_hp(distance: float, shot_dt: float) -> float:
    """HP one autocannon shot does when shots are ``shot_dt`` seconds apart."""
    dt = min(max(0.0, float(shot_dt)), AUTOCANNON_MAX_SHOT_DT)
    return autocannon_dps(distance) * dt


def pulse_splash_hp(distance: float) -> float:
    """Pulse-shell damage at ``distance`` from the burst centre (INF fit)."""
    d = max(0.0, float(distance))
    table = PULSE_SPLASH_TABLE
    if d >= table[-1][0]:
        return 0.0
    for (d0, h0), (d1, h1) in zip(table, table[1:]):
        if d <= d1:
            if d1 <= d0:
                return h1
            t = (d - d0) / (d1 - d0)
            return h0 + (h1 - h0) * t
    return 0.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
Vec3 = Tuple[float, float, float]
T = TypeVar("T")


def horizontal_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Ground-plane distance; the BEH range rings are drawn on the 2-D map."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def in_engagement_band(distance: float, min_range: float, max_range: float) -> bool:
    """True when ``distance`` lies in the turret's [min, max] annulus."""
    return float(min_range) <= float(distance) <= float(max_range)


def pick_nearest_target(
    origin: Sequence[float],
    candidates: Iterable[Tuple[T, Sequence[float]]],
    min_range: float,
    max_range: float,
    has_line_of_sight: Optional[Callable[[T], bool]] = None,
) -> Optional[Tuple[T, float]]:
    """Nearest candidate inside the band (and in LOS). Returns (key, distance)."""
    best: Optional[Tuple[T, float]] = None
    for key, pos in candidates:
        distance = horizontal_distance(origin, pos)
        if not in_engagement_band(distance, min_range, max_range):
            continue
        if best is not None and distance >= best[1]:
            continue
        if has_line_of_sight is not None and not has_line_of_sight(key):
            continue
        best = (key, distance)
    return best


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)


def _scale(v: Sequence[float], s: float) -> Vec3:
    return (float(v[0]) * s, float(v[1]) * s, float(v[2]) * s)


def direction_to(origin: Sequence[float], target: Sequence[float]) -> Vec3:
    d = (float(target[0]) - float(origin[0]),
         float(target[1]) - float(origin[1]),
         float(target[2]) - float(origin[2]))
    n = _norm(d)
    if n <= 1e-9:
        return (1.0, 0.0, 0.0)
    return _scale(d, 1.0 / n)


def rotate_about_z(v: Sequence[float], angle: float) -> Vec3:
    c, s = math.cos(angle), math.sin(angle)
    return (float(v[0]) * c - float(v[1]) * s, float(v[0]) * s + float(v[1]) * c, float(v[2]))


def flak_volley_velocities(
    muzzle: Sequence[float],
    target_pos: Sequence[float],
    *,
    speed: float = FLAK_SHELL_SPEED,
    fan_deg: float = FLAK_FAN_DEG,
    count: int = FLAK_SHELLS_PER_VOLLEY,
) -> list[Vec3]:
    """Velocities for one flak volley.

    The centre shell aims at the target's *current* position (CAP: 0.0 deg
    error, no leading); the others are fanned horizontally by +-fan_deg.
    """
    centre = direction_to(muzzle, target_pos)
    fan = math.radians(float(fan_deg))
    half = (int(count) - 1) / 2.0
    out = []
    for i in range(int(count)):
        offset = (i - half) * fan
        out.append(_scale(rotate_about_z(centre, offset), float(speed)))
    return out


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)
    return math.acos(max(-1.0, min(1.0, c)))


def steer_toward(vel: Sequence[float], desired_dir: Sequence[float], max_angle: float,
                 speed: Optional[float] = None) -> Vec3:
    """Rotate ``vel`` toward ``desired_dir`` by at most ``max_angle`` radians."""
    spd = _norm(vel) if speed is None else float(speed)
    cur = _scale(vel, 1.0 / _norm(vel)) if _norm(vel) > 1e-9 else direction_to((0, 0, 0), desired_dir)
    want = direction_to((0, 0, 0), desired_dir)
    theta = angle_between(cur, want)
    if theta <= max(0.0, float(max_angle)) or theta <= 1e-9:
        return _scale(want, spd)
    # Slerp by the allowed fraction.
    t = float(max_angle) / theta
    sin_theta = math.sin(theta)
    if abs(sin_theta) <= 1e-9:
        # Opposite directions: rotate about Z.
        return _scale(rotate_about_z(cur, float(max_angle)), spd)
    w0 = math.sin((1.0 - t) * theta) / sin_theta
    w1 = math.sin(t * theta) / sin_theta
    mixed = (cur[0] * w0 + want[0] * w1, cur[1] * w0 + want[1] * w1, cur[2] * w0 + want[2] * w1)
    n = _norm(mixed)
    return _scale(mixed, spd / n if n > 1e-9 else 0.0)


def segment_point_distance(p0: Sequence[float], p1: Sequence[float], c: Sequence[float]) -> float:
    """Closest distance from point ``c`` to segment p0-p1 (swept hit test)."""
    d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    w = (c[0] - p0[0], c[1] - p0[1], c[2] - p0[2])
    dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    t = 0.0 if dd <= 1e-12 else max(0.0, min(1.0, (w[0] * d[0] + w[1] * d[1] + w[2] * d[2]) / dd))
    q = (p0[0] + d[0] * t - c[0], p0[1] + d[1] * t - c[1], p0[2] + d[2] * t - c[2])
    return _norm(q)


def upstream_projectile_rotation(vel: Sequence[float]) -> Vec3:
    """Rotation vector upstream sends for shells/hunters: (elevation, 0, yaw).

    CAP: flak (0.0396, 0, -1.2434) for v=(44.98,-132.45,5.55); hunters
    (1.571, 0, 0) while climbing straight up, then (elev, 0, yaw) of velocity.
    """
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
    horiz = math.hypot(vx, vy)
    elev = math.atan2(vz, horiz) if (horiz > 0.0 or vz != 0.0) else 0.0
    yaw = math.atan2(vy, vx) if horiz > 0.0 else 0.0
    return (elev, 0.0, yaw)
