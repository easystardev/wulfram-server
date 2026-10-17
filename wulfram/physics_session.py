"""s2c opcode 0x56 'V' physics-session message for the native client (community/native).

The native WulframClient.exe run with ``--no-mods`` (no session provider) only
leaves "Loading world" after it has accepted a 0x56 physics session: that is the
sole setter of its readiness flag ``conn+0x2100`` (D:\\work\\upstream-1016\\ghidra\\
RESULT.md, Q1/Q3).  The same message becomes the client's prediction physics
(``gamesession+0x2e8``), so the values matter once a tank spawns.

Wire layout (client parser ``FUN_14014e330`` + ``FUN_14014da90`` +
``FUN_14014de20``; big-endian, byte-aligned, NO trailing bytes allowed; field
names and defaults: D:\\work\\upstream-1016\\ghidra\\RESULT-0x56.md)::

    u8   0x56
    u32  magic            'WPC1' 0x57504331 | 'WPC2' 0x57504332
    u32  tick_ms          1..550; the client converts server ticks to time with it
    str  physics_version  u16 len + bytes; must be exactly "0.7.0-dev" (fatal otherwise)
    str  config_name      not checked
    vehicle x 2           [0] tank, [1] scout: 17 f64 + 6 f32 movement scalars,
                          2 feedback blocks (jet_fast_react / jet_slow_react),
                          5 curves, 10 f32, bool degen_on_slope
    jets  4 x 4 x {6 x f32, u8 bool}   hover points per (unit kind, team) row
    2 x f32               vehicle_fraction tank, scout
    WPC2 only: f32 max_visibility  (0, or 100..1e6)

    curve = u16 n (2..128) + n x f32;  f64 = hi u32, lo u32;  bool = u8 0/1
    every scalar finite and |x| <= 1e12

``CLIENT_DEFAULT_FIELDS`` is the client's own built-in table, bit-exact
(``wpc_client_defaults.json``, recovered by running the client's builders).  It
mostly agrees with our OG tank sim (turn 4.5, move 85, strafe 69.7, jet curves and
react tables bit-identical) but NOT everywhere -- notably gravity (client 160) and
the jump model -- so prediction can still drift from our server; a table built
from our own sim constants is the follow-up.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Iterable, Sequence

OPCODE = 0x56
MAGIC_WPC1 = 0x57504331
MAGIC_WPC2 = 0x57504332
MAGICS = {"WPC1": MAGIC_WPC1, "WPC2": MAGIC_WPC2}
CLIENT_PHYSICS_VERSION = "0.7.0-dev"
DEFAULT_CONFIG_NAME = "wulfram-server"
SCALAR_LIMIT = 1e12
TICK_MS_RANGE = (1, 550)
CURVE_LEN_RANGE = (2, 128)
VISIBILITY_RANGE = (100.0, 1e6)

KINDS = ("f64", "f32", "bool", "curve")

VEHICLES = ("vehicle[0:tank]", "vehicle[1:scout]")
FEEDBACKS = ("jet_fast_react", "jet_slow_react")
WPC2_ONLY = ("max_visibility",)

_DEFAULTS_PATH = Path(__file__).with_name("wpc_client_defaults.json")


def _from_bits(kind: str, bits):
    if kind == "f64":
        return struct.unpack(">d", bytes.fromhex(bits))[0]
    if kind == "f32":
        return struct.unpack(">f", bytes.fromhex(bits))[0]
    if kind == "curve":
        return tuple(struct.unpack(">f", bytes.fromhex(b))[0] for b in bits)
    return int(bits)


def _load_defaults() -> tuple[tuple[str, str, object], ...]:
    data = json.loads(_DEFAULTS_PATH.read_text(encoding="utf-8"))
    return tuple((name, kind, _from_bits(kind, raw)) for name, kind, raw in data["fields"])


# The client's built-in WPC2 body table (see module docstring).
CLIENT_DEFAULT_FIELDS: tuple[tuple[str, str, object], ...] = _load_defaults()


def layout(magic: str = "WPC2") -> list[tuple[str, str]]:
    """Ordered (name, kind) list of the body after the four header fields."""
    if magic not in MAGICS:
        raise ValueError(f"unknown physics session magic {magic!r}")
    return [(n, k) for n, k, _ in fields_for_magic(CLIENT_DEFAULT_FIELDS, magic)]


def fields_for_magic(fields: Sequence[tuple[str, str, object]], magic: str) -> list[tuple[str, str, object]]:
    """Drop the WPC2-only tail when encoding WPC1 from a WPC2 table."""
    if magic == "WPC1":
        return [f for f in fields if f[0] not in WPC2_ONLY]
    return list(fields)


def _check_scalar(name: str, value) -> float:
    value = float(value)
    if not math.isfinite(value) or abs(value) > SCALAR_LIMIT:
        raise ValueError(f"{name}: scalar {value!r} not finite or beyond {SCALAR_LIMIT:g}")
    return value


def _pack_str(text: str) -> bytes:
    raw = text.encode("ascii")
    if len(raw) > 4096:
        raise ValueError("protocol string too long")
    return struct.pack(">H", len(raw)) + raw


def _validate_constraints(values: dict, magic: str) -> None:
    for vehicle in VEHICLES:
        checks = (
            (f"{vehicle}.density", lambda x: x > 0.0),
            (f"{vehicle}.maximum_velocity", lambda x: x > 0.0),
            (f"{vehicle}.maximum_altitude", lambda x: x > 0.0),
            (f"{vehicle}.maximum_fuel", lambda x: x > 0.0),
            (f"{vehicle}.gravity", lambda x: x >= 0.0),
            (f"{vehicle}.max_xy_speed", lambda x: x > 0.0),
        )
        for fb in FEEDBACKS:
            checks += (
                (f"{vehicle}.{fb}.abs_max_error", lambda x: x > 0.0),
                (f"{vehicle}.{fb}.abs_max_prime", lambda x: x > 0.0),
            )
        for name, ok in checks:
            if not ok(float(values[name])):
                raise ValueError(f"{name}={values[name]!r} violates the client's vehicle limits")
    if magic == "WPC2":
        vis = float(values["max_visibility"])
        if vis != 0.0 and not (VISIBILITY_RANGE[0] <= vis <= VISIBILITY_RANGE[1]):
            raise ValueError(f"max_visibility={vis!r} must be 0 or {VISIBILITY_RANGE}")


def encode_physics_session(
    fields: Iterable[tuple[str, str, object]] = CLIENT_DEFAULT_FIELDS,
    *,
    magic: str = "WPC2",
    tick_ms: int,
    physics_version: str = CLIENT_PHYSICS_VERSION,
    config_name: str = DEFAULT_CONFIG_NAME,
) -> bytes:
    """Encode a complete 0x56 message (opcode included) from a field table."""
    if magic not in MAGICS:
        raise ValueError(f"unknown physics session magic {magic!r}")
    tick_ms = int(tick_ms)
    if not TICK_MS_RANGE[0] <= tick_ms <= TICK_MS_RANGE[1]:
        raise ValueError(f"tick_ms {tick_ms} outside {TICK_MS_RANGE}")
    if physics_version != CLIENT_PHYSICS_VERSION:
        raise ValueError(f"physics_version {physics_version!r} is fatal on the client")
    table = fields_for_magic(list(fields), magic)
    expected = layout(magic)
    if [(n, k) for n, k, _ in table] != expected:
        raise ValueError("physics session field table does not match the client layout")

    out = bytearray((OPCODE,))
    out += struct.pack(">II", MAGICS[magic], tick_ms)
    out += _pack_str(physics_version) + _pack_str(config_name)
    values = {}
    for name, kind, value in table:
        if kind == "f64":
            values[name] = _check_scalar(name, value)
            out += struct.pack(">d", values[name])  # == hi u32, lo u32 big-endian
        elif kind == "f32":
            values[name] = _check_scalar(name, value)
            packed = struct.pack(">f", values[name])
            if not math.isfinite(struct.unpack(">f", packed)[0]):
                raise ValueError(f"{name}: not representable as f32")
            out += packed
        elif kind == "bool":
            if value not in (0, 1, False, True):
                raise ValueError(f"{name}: bool must be 0/1")
            out.append(1 if value else 0)
        elif kind == "curve":
            points = [_check_scalar(name, v) for v in value]
            if not CURVE_LEN_RANGE[0] <= len(points) <= CURVE_LEN_RANGE[1]:
                raise ValueError(f"{name}: curve needs {CURVE_LEN_RANGE} points, got {len(points)}")
            out += struct.pack(">H", len(points)) + b"".join(struct.pack(">f", p) for p in points)
        else:
            raise ValueError(f"{name}: unknown kind {kind!r}")
    _validate_constraints(values, magic)
    return bytes(out)


def tick_ms_for_rate(tick_rate_hz: float) -> int:
    """Server tick in whole milliseconds (30 Hz -> 33), clamped to the wire range."""
    try:
        ms = int(round(1000.0 / float(tick_rate_hz)))
    except (TypeError, ValueError, ZeroDivisionError):
        ms = 33
    return max(TICK_MS_RANGE[0], min(TICK_MS_RANGE[1], ms))
