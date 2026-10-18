"""community/native: step native-client tanks with the out-of-process community physics sidecar.

Tag: community/native.  The community WulframClient.exe (``--no-mods``) predicts its
own vehicle with the "0.7.0-dev" physics it receives in the s2c 0x56 session.
Our legacy sim is a different model, so native sessions drift and get snapped
(VM runs 20-25: 95-110 u after a 5 s drive, a ~4 u idle z bounce).  With the
knob on, a native session's tank is instead stepped by a separately run,
separately licensed sidecar process that implements the client's step; this
module is only a protocol client for it (no physics lives here).  OG
(wulfram2.exe) sessions never reach this module's stepping path, knob on or off.

Knobs (parsed by ``configure``; everything is inert unless the first is on):

* ``WULFRAM_NATIVE_COMMUNITY_PHYSICS=1`` -- enable (default off).
* ``WULFRAM_COMMUNITY_PHYSICS_ADDR`` -- sidecar ``host:port`` (default
  ``127.0.0.1:47056``).  The sidecar is launched separately.
* ``WULFRAM_COMMUNITY_PHYSICS_TICK_MS`` -- the fixed step grid (default 40, the
  client's own built-in tick).  Also becomes the 0x56 ``tick_ms`` default while
  the knob is on; an explicit ``WULFRAM_NATIVE_PHYSICS_SESSION_TICK_MS`` still
  wins and is then used for the grid too (the sidecar always steps with the
  tick_ms the session was actually sent).
* ``WULFRAM_COMMUNITY_PHYSICS_TIMEOUT_S`` -- per-request timeout (default 1.0).
* ``WULFRAM_COMMUNITY_PHYSICS_JET`` -- assumed jet level (default: the 0x56
  ``vehicle_fraction`` the client starts with, 0.8).  The native client does not
  put its jet level on the wire (VM runs 22/25: jet 0 and 0.3 give identical
  ACTION streams), so the server cannot follow jet changes.
* ``WULFRAM_COMMUNITY_PHYSICS_STRAFE_SIGN`` -- strafe = sign * slot 3 (default -1,
  same convention as slot 1, which is observed negated: turn +0.5 -> -0.488).
* ``WULFRAM_COMMUNITY_PHYSICS_INPUT_SNAP`` -- snap a decoded axis to the nearest
  multiple of this grid when that multiple lies in the value's quantisation
  cell (default 0.05; 0 disables).  The ACTION quantiser step is 2000/65534, so
  1.0 arrives as 1.007, 0.5 as 0.488, 0.3 as 0.305.
* ``WULFRAM_COMMUNITY_PHYSICS_SPAWN_SETTLE`` -- spawn the tank this far above
  the terrain (default 4.447, the client's hover rest at jet 0.8), so neither
  side has to settle after spawn.
* ``WULFRAM_COMMUNITY_PHYSICS_SCOUT=1`` -- also manage scouts (default tanks only;
  the scout path has no golden yet).
* ``WULFRAM_COMMUNITY_PHYSICS_SYNC=1`` -- step inside the server tick instead of
  on the per-session worker thread (tests; holds FRAME_LOCK during the RPC).

Fail safe: any sidecar error (refused, timeout, protocol error) is logged once
for that session and the session falls back to the legacy sim for the rest of
its life; nothing else changes.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import socket
import struct
import threading
import time
from typing import Optional

from . import native_session

PROTOCOL_VERSION = 1
MAX_FRAME = 1 << 20
DEFAULT_ADDR = "127.0.0.1:47056"
DEFAULT_TICK_MS = 40
DEFAULT_SETTLE = 4.447
MAX_BACKLOG_TICKS = 12          # the client caps its own catch-up at 500 ms
MAX_TICKS_PER_REQUEST = 25
TANK, SCOUT = 0, 1


class SidecarError(Exception):
    pass


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "on", "yes")


def _env_float(name, default):
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _parse_addr(text):
    host, _, port = text.strip().rpartition(":")
    return (host or "127.0.0.1"), int(port)


# --------------------------------------------------------------------- wire
class SidecarLink:
    """One connection = one sidecar world.  Length-prefixed JSON (u32 BE + UTF-8)."""

    def __init__(self, addr, timeout):
        self.addr, self.timeout = addr, timeout
        self.sock = None
        self._rid = itertools.count(1)

    def open(self):
        try:
            self.sock = socket.create_connection(self.addr, timeout=self.timeout)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as exc:
            raise SidecarError(f"connect {self.addr[0]}:{self.addr[1]} failed: {exc}") from None
        info = self.request("hello", protocol=PROTOCOL_VERSION, client="wulfram-server")
        if info.get("protocol") != PROTOCOL_VERSION:
            raise SidecarError(f"sidecar speaks protocol {info.get('protocol')!r}")
        return info

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise SidecarError("sidecar closed the connection")
            buf += chunk
        return buf

    def request(self, op, **fields):
        if self.sock is None:
            raise SidecarError("not connected")
        rid = next(self._rid)
        body = json.dumps(dict(fields, op=op, rid=rid), separators=(",", ":"), allow_nan=False).encode()
        try:
            self.sock.sendall(struct.pack(">I", len(body)) + body)
            (n,) = struct.unpack(">I", self._recv(4))
            if not 0 < n <= MAX_FRAME:
                raise SidecarError(f"bad frame length {n}")
            reply = json.loads(self._recv(n))
        except (OSError, ValueError) as exc:
            raise SidecarError(f"{op}: {type(exc).__name__}: {exc}") from None
        if not isinstance(reply, dict) or reply.get("rid") not in (rid, None):
            raise SidecarError(f"{op}: unexpected reply")
        if not reply.get("ok"):
            err = reply.get("error") or {}
            raise SidecarError(f"{op}: {err.get('code')}: {err.get('message')}")
        return reply

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


def field_table(fields):
    """0x56 (name, kind, value) table -> JSON-safe [name, kind, value] rows (exact floats)."""
    rows = []
    for name, kind, value in fields:
        if kind == "curve":
            value = [float(v) for v in value]
        elif kind == "bool":
            value = 1 if value else 0
        else:
            value = float(value)
        rows.append([name, kind, value])
    return rows


def _table_value(fields, name, default):
    for n, _kind, value in fields:
        if n == name:
            return float(value)
    return default


def _wrap_pi(a):
    a = math.fmod(a, 2.0 * math.pi)
    if a > math.pi:
        a -= 2.0 * math.pi
    elif a <= -math.pi:
        a += 2.0 * math.pi
    return a


# --------------------------------------------------------------------- per session
class Track:
    """Sidecar state for one in-game native session (one generation = one spawn)."""

    def __init__(self, generation, link, body_id, tick_ms, t0):
        self.generation = generation
        self.link = link
        self.body_id = body_id
        self.tick_ms = tick_ms
        self.t0 = t0
        self.ticks = 0
        self.state = None            # latest body state (client frame)
        self.published = None        # (pos, vel) last written into ctx (server frame)
        self.recreate = None         # pending (pos, vel, euler) after a server-side teleport
        self.error = ""
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None


class CommunityPhysics:
    def __init__(self, server, *, addr=None, tick_ms=None, timeout=None, sync=None, clock=time.perf_counter):
        self.server = server
        self.addr = addr or _parse_addr(os.environ.get("WULFRAM_COMMUNITY_PHYSICS_ADDR", DEFAULT_ADDR))
        self.tick_ms = int(tick_ms or _env_float("WULFRAM_COMMUNITY_PHYSICS_TICK_MS", DEFAULT_TICK_MS))
        if not 1 <= self.tick_ms <= 550:
            self.tick_ms = DEFAULT_TICK_MS
        self.timeout = float(timeout or _env_float("WULFRAM_COMMUNITY_PHYSICS_TIMEOUT_S", 1.0))
        self.sync = _env_flag("WULFRAM_COMMUNITY_PHYSICS_SYNC") if sync is None else bool(sync)
        self.clock = clock
        self.jet_override = os.environ.get("WULFRAM_COMMUNITY_PHYSICS_JET", "").strip()
        self.strafe_sign = -1.0 if _env_float("WULFRAM_COMMUNITY_PHYSICS_STRAFE_SIGN", -1.0) < 0 else 1.0
        self.snap_grid = max(0.0, _env_float("WULFRAM_COMMUNITY_PHYSICS_INPUT_SNAP", 0.05))
        self.spawn_settle = _env_float("WULFRAM_COMMUNITY_PHYSICS_SPAWN_SETTLE", DEFAULT_SETTLE)
        self.kinds = (TANK, SCOUT) if _env_flag("WULFRAM_COMMUNITY_PHYSICS_SCOUT") else (TANK,)
        self.tracks = {}             # id(ctx) -> Track
        self.failed = {}             # id(ctx) -> generation that fell back
        self.lock = threading.Lock()

    # ------------------------------------------------------------ classification
    def eligible(self, ctx) -> bool:
        """Native session with a 0x56 physics session, driving a managed vehicle kind."""
        session = getattr(ctx, "session", None)
        if session is None or not getattr(session, "physics_session_sent", False):
            return False
        if int(getattr(ctx, "entity_type", -1)) not in self.kinds:
            return False
        return native_session.is_native_session(ctx)

    @staticmethod
    def generation(ctx):
        return (id(ctx), float(getattr(ctx.session, "last_spawn_time", 0) or 0))

    def manages(self, ctx) -> bool:
        if not (self.eligible(ctx) and ctx.session.in_game and float(getattr(ctx, "player_health", 0) or 0) > 0):
            return False
        return self.failed.get(id(ctx)) != self.generation(ctx)

    # ------------------------------------------------------------ hooks
    def session_tick_ms(self) -> int:
        return self.tick_ms

    def spawn_z(self, ctx, spawn_pos) -> Optional[float]:
        """Server-frame spawn z for an eligible session: terrain + settle height, else None."""
        terrain = getattr(self.server, "terrain", None)
        if terrain is None or getattr(self.server, "up_axis", "z") != "z" or not self.eligible(ctx):
            return None
        client = self.server._to_client_pos(tuple(spawn_pos))
        settled = float(terrain.get_height(client[0], client[1])) + self.spawn_settle
        return float(self.server._from_client_pos((client[0], client[1], settled))[2])

    def advance(self, ctx) -> bool:
        """Called once per server tick for an in-game client.  True = the sidecar owns this
        tank this tick (legacy stepping must be skipped); False = legacy sim, unchanged."""
        if not self.manages(ctx):
            self._drop(ctx)
            return False
        track = self.tracks.get(id(ctx))
        gen = self.generation(ctx)
        if track is not None and (track.generation != gen or track.error):
            if track.error:
                self._fail(ctx, track.error)
                return False
            self._drop(ctx)
            track = None
        if track is None:
            try:
                track = self._open(ctx, gen)
            except SidecarError as exc:
                self._fail(ctx, str(exc))
                return False
        if track.published is not None and (tuple(ctx.player_pos), tuple(ctx.player_vel)) != track.published:
            # The server moved the tank (teleport, impulse, respawn pose): re-seed the body there.
            with track.lock:
                track.recreate = self._pose(ctx)
        if self.sync:
            try:
                self._step_due(ctx, track)
            except SidecarError as exc:
                self._fail(ctx, str(exc))
                return False
        elif track.error:
            self._fail(ctx, track.error)
            return False
        self._publish(ctx, track)
        ctx.community_physics_active = True
        return True

    # ------------------------------------------------------------ internals
    def _pose(self, ctx):
        pos = self.server._to_client_pos(tuple(ctx.player_pos))
        pose = getattr(ctx, "player_pose", {}) or {}
        euler = (float(pose.get("roll", 0.0) or 0.0), float(pose.get("pitch", 0.0) or 0.0),
                 float(ctx.player_heading or 0.0))
        return [float(v) for v in pos], [float(v) for v in ctx.player_vel], [float(v) for v in euler]

    def _session_table(self, ctx):
        from . import physics_session
        fields = getattr(ctx.session, "physics_session_fields", None) or physics_session.CLIENT_DEFAULT_FIELDS
        tick_ms = int(getattr(ctx.session, "physics_session_tick_ms", 0) or 0) or self.tick_ms
        return list(fields), tick_ms

    def _jet(self, fields, kind):
        if self.jet_override:
            try:
                return min(1.0, max(0.0, float(self.jet_override)))
            except ValueError:
                pass
        return _table_value(fields, "vehicle_fraction[%d:%s]" % (kind, "tank" if kind == TANK else "scout"), 0.8)

    def _open(self, ctx, gen):
        """Register a track seeded with the current (spawn) pose.  Sync mode connects inline
        (raises SidecarError); worker mode connects on the worker thread so the tick loop never
        waits on connect/load_map while holding FRAME_LOCK."""
        fields, tick_ms = self._session_table(ctx)
        kind = int(ctx.entity_type)
        body_id = int(getattr(ctx.session, "player_id", 0) or ctx.entity_id or 1)
        track = Track(gen, SidecarLink(self.addr, self.timeout), body_id, tick_ms, None)
        track.kind = kind
        track.jet = self._jet(fields, kind)
        track.fields = fields
        track.ctx = ctx
        track.seed = self._pose(ctx), int(getattr(ctx.session, "team_id", 0) or 0), float(ctx.player_fuel)
        if self.sync:
            self._connect(track)
        self.tracks[id(ctx)] = track
        if not self.sync:
            track.thread = threading.Thread(target=self._worker, args=(track,),
                                            name=f"community-physics-{ctx.client_id}", daemon=True)
            track.thread.start()
        return track

    def _connect(self, track):
        (pos, vel, euler), team, fuel = track.seed
        link = track.link
        try:
            link.open()
            link.request("configure", tick_ms=track.tick_ms, fields=field_table(track.fields))
            link.request("load_map", map=str(self.server.map_name))
            reply = link.request("create_body", id=track.body_id, kind=track.kind, team=team, pos=pos,
                                 vel=vel, euler=euler, fuel=fuel, jet=track.jet)
        except SidecarError:
            link.close()
            raise
        with track.lock:
            track.state = reply["body"]
        track.t0 = self.clock()
        print(f"[COMMUNITY] Client {track.ctx.client_id}: community physics on "
              f"({self.addr[0]}:{self.addr[1]} map={self.server.map_name} tick_ms={track.tick_ms} "
              f"body={track.body_id} {'sync' if self.sync else 'worker'})")

    def _snap(self, value, quantum):
        v = float(value)
        if self.snap_grid > 0.0:
            k = round(v / self.snap_grid)
            c = k * self.snap_grid
            if abs(v - c) <= 0.5 * quantum + 1e-9:
                v = c
        return max(-1.0, min(1.0, v))

    def controls(self, ctx, jet):
        """Native client controls (turn, move, strafe, jet, tilt, roll, jump) from the
        decoded ACTION slots: slot 1 = -turn, 2 = move, 3 = strafe_sign*strafe, 4 = jump."""
        ws = ctx.weapon_system
        slots = list(ws.behavior_slots) + [0.0] * 8
        bits = int(getattr(ws, "control_bits", 16) or 16)
        quantum = float(getattr(ws, "control_range", 2000.0) or 2000.0) / max(1, (1 << bits) - 2)
        turn = self._snap(-slots[1], quantum)
        move = self._snap(slots[2], quantum)
        strafe = self._snap(self.strafe_sign * slots[3], quantum)
        jump = 1.0 if slots[4] > 0.5 else 0.0
        stale = float(getattr(self.server, "input_stale_timeout_s", 0) or 0)
        last = float(getattr(ctx, "last_action_packet_time", 0) or 0)
        if stale > 0 and last > 0 and time.monotonic() - last > stale:
            turn = move = strafe = 0.0
            jump = 0.0
        return [turn, move, strafe, float(jet), 0.0, 0.0, jump]

    def _step_due(self, ctx, track):
        if track.t0 is None:                   # worker still connecting
            return
        with track.lock:
            recreate, track.recreate = track.recreate, None
        if recreate is not None:
            pos, vel, euler = recreate
            track.link.request("remove_body", id=track.body_id)
            reply = track.link.request("create_body", id=track.body_id, kind=track.kind,
                                       team=int(getattr(ctx.session, "team_id", 0) or 0), pos=pos, vel=vel,
                                       euler=euler, fuel=float(ctx.player_fuel), jet=track.jet)
            with track.lock:
                track.state = reply["body"]
        dt = track.tick_ms / 1000.0
        due = int((self.clock() - track.t0) / dt + 1e-9)
        n = due - track.ticks
        if n > MAX_BACKLOG_TICKS:              # never replay a stall with stale held input
            track.ticks = due - MAX_BACKLOG_TICKS
            n = MAX_BACKLOG_TICKS
        while n > 0:
            count = min(n, MAX_TICKS_PER_REQUEST)
            ctl = self.controls(ctx, track.jet)
            reply = track.link.request("step", tick=track.ticks + 1, count=count,
                                       inputs={str(track.body_id): ctl})
            state = reply["bodies"].get(str(track.body_id))
            if state is None:
                raise SidecarError("step: body missing from reply")
            if state.get("faulted"):
                raise SidecarError(f"step: body faulted: {state['faulted']}")
            with track.lock:
                track.state = state
            track.ticks += count
            n -= count

    def _worker(self, track):
        dt = track.tick_ms / 1000.0
        try:
            self._connect(track)
            while not track.stop.is_set():
                self._step_due(track.ctx, track)
                wait = track.t0 + (track.ticks + 1) * dt - self.clock()
                track.stop.wait(max(0.001, min(wait, dt)))
        except SidecarError as exc:
            track.error = str(exc)
        except Exception as exc:  # never let the worker die silently
            track.error = f"{type(exc).__name__}: {exc}"

    def _publish(self, ctx, track):
        with track.lock:
            st = track.state
        if st is None:
            return
        server = self.server
        pos = tuple(server._from_client_pos(tuple(float(v) for v in st["pos"])))
        vel = tuple(float(v) for v in st["vel"])
        m = [float(v) for v in st["matrix"]]
        w = [float(v) for v in st["ang_vel"]]                     # body frame
        ww = (m[0] * w[0] + m[1] * w[1] + m[2] * w[2],            # world frame = M * w
              m[3] * w[0] + m[4] * w[1] + m[5] * w[2],
              m[6] * w[0] + m[7] * w[1] + m[8] * w[2])
        roll, pitch, yaw = (_wrap_pi(float(v)) for v in st["euler"])
        ctx.player_pos, ctx.player_vel = pos, vel
        ctx.player_heading, ctx.player_yaw = yaw, -yaw
        ctx.angular_vel_yaw = ctx.player_angular_vel = ww[2]
        ctx.spring_body_ang_vel = (ww[0], ww[1])
        ctx.spring_body_matrix = tuple(m)
        ctx.player_speed = math.hypot(vel[0], vel[1])
        if isinstance(getattr(ctx, "player_pose", None), dict):
            ctx.player_pose.update(pos=pos, vel=vel, roll=roll, pitch=pitch, yaw=-yaw)
        ctx.player_fuel = float(st["fuel"])
        ctx.rigid_body_sleeping = False
        ctx.rigid_body_target_pos = pos
        ctx.rigid_body_target_rot = (roll, pitch, yaw)
        physics = getattr(ctx, "vehicle_physics", None)
        if physics is not None:
            physics.heading = yaw
            physics.angular_velocity = ww[2]
        track.published = (pos, vel)
        ctx.debug_last_controller_step = {"backend": "community", "tick": track.ticks, "pos": pos, "vel": vel,
                                          "body_rotation": (roll, pitch, yaw), "angular_velocity": ww,
                                          "clock": "fixed_grid", "step_ms": track.tick_ms,
                                          "jets_on": bool(st.get("jets_on")), "contact": bool(st.get("contact"))}

    def _close_track(self, track):
        track.stop.set()
        track.link.close()                     # also unblocks a worker waiting on a reply
        if track.thread is not None and track.thread is not threading.current_thread():
            track.thread.join(timeout=2.0)

    def _drop(self, ctx):
        track = self.tracks.pop(id(ctx), None)
        if track is not None:
            self._close_track(track)
        if getattr(ctx, "community_physics_active", False):
            ctx.community_physics_active = False

    def _fail(self, ctx, reason):
        gen = self.generation(ctx)
        self._drop(ctx)
        if self.failed.get(id(ctx)) != gen:
            print(f"[COMMUNITY] Client {getattr(ctx, 'client_id', '?')}: sidecar unavailable ({reason}); "
                  f"legacy sim for this session")
        self.failed[id(ctx)] = gen

    def status(self):
        return {"backend": "community", "addr": f"{self.addr[0]}:{self.addr[1]}", "tick_ms": self.tick_ms,
                "sessions": len(self.tracks), "fallbacks": len(self.failed), "sync": self.sync}

    def close(self):
        for key in list(self.tracks):
            track = self.tracks.pop(key)
            self._close_track(track)


def configure(server) -> None:
    """(Re)build ``server.community_physics`` from the environment; None when the knob is off."""
    existing = getattr(server, "community_physics", None)
    if not _env_flag("WULFRAM_NATIVE_COMMUNITY_PHYSICS"):
        if existing is not None:
            existing.close()
        server.community_physics = None
        return
    if existing is not None:
        return
    server.community_physics = CommunityPhysics(server)


def session_tick_ms(server) -> int:
    """0x56 tick_ms default contributed by this module (0 = no opinion)."""
    community = getattr(server, "community_physics", None)
    return community.session_tick_ms() if community is not None else 0
