#!/usr/bin/env python3
"""community/native WULFRAM_NATIVE_COMMUNITY_PHYSICS: the sidecar adapter.

* Unit tests against an in-process FAKE sidecar (protocol only, trivial kinematics).
* OG sessions are untouched with the knob off and on.
* One end-to-end test against the REAL community-physics sidecar process (separate,
  AGPL project; launched by the test, skipped unless COMMUNITY_PHYSICS_SIDECAR,
  COMMUNITY_PHYSICS_SHAPES and COMMUNITY_PHYSICS_GOLDEN point at it, the user's
  shapes.zip and a locally generated golden flat__forward.json).

The two ACTION_UPDATE payloads are the native client's own 66-byte restatement
packets from VM run 25 (idle / move=1); they carry input slots only.
Run: uv run --with pytest python -m pytest -q test_community_physics.py
"""
import copy
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram import community_physics as cp  # noqa: E402
from wulfram import handlers, native_session, physics_session  # noqa: E402
from wulfram.native_live import NativeLiveWorld  # noqa: E402
from wulfram.server import WulframServer  # noqa: E402

NATIVE_IDLE = bytes.fromhex(
    "0a15000000020000000000010000000200000003000000040002bffb00030000000380000004000240014000b000600034001c000f00080004400240013000a00054")
NATIVE_FORWARD = bytes.fromhex(
    "0a1500000046000000000001000000027fdf0003000000040002bffb00030000000380000004000240014000b000600034001c000f00080004400240013000a00054")
PHYS_V2 = bytes.fromhex("540014706f727461626c655f706879736963735f76320000010000000001")
KNOBS = ("WULFRAM_NATIVE_COMMUNITY_PHYSICS", "WULFRAM_COMMUNITY_PHYSICS_ADDR", "WULFRAM_COMMUNITY_PHYSICS_TICK_MS",
         "WULFRAM_COMMUNITY_PHYSICS_SYNC", "WULFRAM_COMMUNITY_PHYSICS_JET", "WULFRAM_NATIVE_PHYSICS_SESSION",
         "WULFRAM_NATIVE_PHYSICS_SESSION_TICK_MS", "WULFRAM_NATIVE_NOMODS_PROFILE")


@contextmanager
def env(**values):
    saved = {k: os.environ.get(k) for k in set(KNOBS) | set(values)}
    for k in KNOBS:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class FakeTCP:
    def __init__(self):
        self.sent = []

    def send(self, payload, log=True):
        self.sent.append(bytes(payload))


# --------------------------------------------------------------------- fake sidecar
class FakeSidecar:
    """Speaks the sidecar protocol; body moves +move u along x per tick, yaws by turn."""

    def __init__(self, fail_after_steps=None):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.addr = self.sock.getsockname()
        self.requests = []
        self.connections = 0
        self.fail_after_steps = fail_after_steps
        self.steps = 0
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._conn, args=(conn,), daemon=True).start()

    @staticmethod
    def _read(conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _conn(self, conn):
        bodies, tick = {}, 0
        while True:
            head = self._read(conn, 4)
            if head is None:
                return
            req = json.loads(self._read(conn, struct.unpack(">I", head)[0]))
            self.requests.append(req)
            op, out = req["op"], {"ok": True, "rid": req.get("rid")}
            if op == "hello":
                out.update(protocol=1, tick_ms=40)
            elif op == "create_body":
                bodies[req["id"]] = dict(id=req["id"], pos=list(req["pos"]), vel=list(req.get("vel", [0, 0, 0])),
                                         euler=list(req.get("euler", [0, 0, 0])), ang_vel=[0.0, 0.0, 0.0],
                                         matrix=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], fuel=req.get("fuel", 0.0),
                                         jet=req.get("jet", 0.8), jets_on=True, contact=False)
                out.update(tick=tick, body=bodies[req["id"]])
            elif op == "remove_body":
                bodies.pop(req["id"], None)
            elif op == "step":
                self.steps += 1
                if self.fail_after_steps is not None and self.steps > self.fail_after_steps:
                    out = {"ok": False, "rid": req.get("rid"), "error": {"code": "internal", "message": "boom"}}
                elif req["tick"] != tick + 1:
                    out = {"ok": False, "rid": req.get("rid"), "error": {"code": "tick_mismatch", "message": "x"}}
                else:
                    for _ in range(req.get("count", 1)):
                        tick += 1
                        for bid, ctl in req.get("inputs", {}).items():
                            b = bodies[int(bid)]
                            b["pos"][0] += ctl[1]
                            b["euler"][2] += ctl[0] * 0.1
                            b["vel"] = [ctl[1] * 25.0, 0.0, 0.0]
                    out.update(tick=tick, bodies={str(k): v for k, v in bodies.items()})
            body = json.dumps(out).encode()
            conn.sendall(struct.pack(">I", len(body)) + body)

    def close(self):
        self.sock.close()


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# --------------------------------------------------------------------- contexts
def make_server(**knobs):
    with env(**knobs):
        return WulframServer(host="127.0.0.1", port=0)


def native_ctx(srv, entity_id=4242):
    ctx = srv._create_client_context(("10.10.10.2", 40003))
    ctx.tcp_handler = FakeTCP()
    ctx.entity_id = entity_id
    s = ctx.session
    s.hello_key_pushed = True
    s.udp_identify_kind = native_session.IDENTIFY_KIND_HELLO_KEY
    s.translation_sent = True
    s.translation_ack_received = True
    s.player_id = entity_id
    s.team_id = 1
    return ctx


def og_ctx(srv):
    ctx = srv._create_client_context(("10.10.10.3", 40004))
    ctx.tcp_handler = FakeTCP()
    ctx.entity_id = 4343
    s = ctx.session
    s.udp_identify_kind = native_session.IDENTIFY_KIND_HELLO_THERE
    s.translation_sent = True
    s.translation_ack_received = True
    s.in_game = True
    s.player_id = 4343
    s.team_id = 2
    ctx.player_health = 1.0
    return ctx


def send_phys(server, ctx):
    assert handlers.maybe_send_physics_session(server, ctx, PHYS_V2) is True
    return physics_session_tick(ctx.tcp_handler.sent[-1])


def physics_session_tick(msg):
    assert msg[0] == 0x56
    return struct.unpack(">I", msg[5:9])[0]


def go_in_game(ctx, pos=(5000.0, 5000.0, 16.0)):
    ctx.session.in_game = True
    ctx.session.last_spawn_time = 1.0
    ctx.player_health = 1.0
    ctx.player_pos = pos
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_heading = 0.0
    ctx.player_fuel = 33000.0
    ctx.last_action_packet_time = 0.0


# --------------------------------------------------------------------- knob / 0x56
def test_knob_default_off_and_inert():
    srv = make_server(WULFRAM_NATIVE_PHYSICS_SESSION="1")
    assert srv.community_physics is None
    assert make_server().community_physics is None
    ctx = native_ctx(srv)
    assert send_phys(srv, ctx) == 33                     # unchanged default (server tick)
    assert ctx.session.physics_session_tick_ms == 33
    assert cp.session_tick_ms(srv) == 0


def test_knob_on_sends_tick_ms_40_and_records_table():
    srv = make_server(WULFRAM_NATIVE_COMMUNITY_PHYSICS="1", WULFRAM_NATIVE_PHYSICS_SESSION="1")
    assert isinstance(srv.community_physics, cp.CommunityPhysics) and srv.community_physics.tick_ms == 40
    ctx = native_ctx(srv)
    assert send_phys(srv, ctx) == 40
    assert ctx.session.physics_session_tick_ms == 40
    assert list(ctx.session.physics_session_fields) == list(physics_session.CLIENT_DEFAULT_FIELDS)
    # explicit session tick wins and is what the sidecar will use
    srv = make_server(WULFRAM_NATIVE_COMMUNITY_PHYSICS="1", WULFRAM_NATIVE_PHYSICS_SESSION="1",
                      WULFRAM_NATIVE_PHYSICS_SESSION_TICK_MS="50")
    ctx = native_ctx(srv)
    assert send_phys(srv, ctx) == 50
    assert srv.community_physics._session_table(ctx)[1] == 50


def test_field_table_is_exact_json():
    rows = cp.field_table(physics_session.CLIENT_DEFAULT_FIELDS)
    back = json.loads(json.dumps(rows))
    for (name, kind, value), row in zip(physics_session.CLIENT_DEFAULT_FIELDS, back):
        assert row[0] == name and row[1] == kind
        if kind == "curve":
            assert [struct.pack(">f", v) for v in value] == [struct.pack(">f", v) for v in row[2]]
        elif kind == "f32":
            assert struct.pack(">f", value) == struct.pack(">f", row[2])
        elif kind == "f64":
            assert struct.pack(">d", value) == struct.pack(">d", row[2])


# --------------------------------------------------------------------- OG untouched
OG_FIELDS = ("player_pos", "player_vel", "player_heading", "player_yaw", "player_fuel", "angular_vel_yaw",
             "spring_body_ang_vel", "spring_body_matrix", "player_pose", "player_speed")


def test_og_session_untouched_knob_off_and_on():
    fake = FakeSidecar()
    try:
        for knobs in ({}, {"WULFRAM_NATIVE_COMMUNITY_PHYSICS": "1",
                           "WULFRAM_COMMUNITY_PHYSICS_ADDR": "%s:%d" % fake.addr}):
            srv = make_server(**knobs)
            ctx = og_ctx(srv)
            ctx.session.physics_session_sent = True     # even if it somehow had one
            before = {k: copy.deepcopy(getattr(ctx, k, None)) for k in OG_FIELDS}
            community = srv.community_physics
            if community is not None:
                assert community.eligible(ctx) is False and community.manages(ctx) is False
                for _ in range(5):
                    assert community.advance(ctx) is False
                assert community.spawn_z(ctx, (5000.0, 5000.0, 3.25)) is None
            assert {k: getattr(ctx, k, None) for k in OG_FIELDS} == before
            assert not getattr(ctx, "community_physics_active", False)
            assert NativeLiveWorld.manages(ctx) is True
            # the OG never sends a portable_physics want; any other 0x54 is not consumed
            assert handlers.maybe_send_physics_session(srv, ctx, b"\x54" + b"\x00" * 25) is False
            assert ctx.tcp_handler.sent == []
        assert fake.connections == 0                     # no RPC was ever attempted for the OG
    finally:
        fake.close()


def test_og_spawn_bytes_identical_knob_off_and_on(monkeypatch):
    """Real _spawn_wf_style for an OG session: same pose and same TCP bytes with the knob on."""
    monkeypatch.setattr(time, "monotonic", lambda: 1234.5)   # TankPacket carries a ms clock
    results = []
    for knobs in ({}, {"WULFRAM_NATIVE_COMMUNITY_PHYSICS": "1", "WULFRAM_COMMUNITY_PHYSICS_ADDR": "127.0.0.1:9"}):
        srv = make_server(**knobs)
        ctx = og_ctx(srv)
        ctx.session.in_game = False
        gen = srv._spawn_wf_style(ctx, team_id=2, pos=(5050.0, 4950.0, 5.0))
        if gen is not None:
            for _ in gen:
                pass
        results.append((ctx.player_pos, ctx.tcp_handler.sent))
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]


# --------------------------------------------------------------------- native with the fake sidecar
def managed(fake, **kw):
    srv = make_server(WULFRAM_NATIVE_COMMUNITY_PHYSICS="1", WULFRAM_NATIVE_PHYSICS_SESSION="1",
                      WULFRAM_COMMUNITY_PHYSICS_ADDR="%s:%d" % fake.addr)
    clock = FakeClock()
    srv.community_physics = cp.CommunityPhysics(srv, addr=fake.addr, sync=True, clock=clock, **kw)
    srv.map_name = "golden_flat"
    ctx = native_ctx(srv)
    send_phys(srv, ctx)
    go_in_game(ctx)
    return srv, ctx, clock


def test_native_session_stepped_on_40ms_grid_with_mapped_inputs():
    fake = FakeSidecar()
    try:
        srv, ctx, clock = managed(fake)
        com = srv.community_physics
        assert com.eligible(ctx) and com.manages(ctx)
        assert com.advance(ctx) is True and ctx.community_physics_active
        assert NativeLiveWorld.manages(ctx) is False
        ops = [r["op"] for r in fake.requests]
        assert ops == ["hello", "configure", "load_map", "create_body"]
        cfg = fake.requests[1]
        assert cfg["tick_ms"] == 40 and len(cfg["fields"]) == len(physics_session.CLIENT_DEFAULT_FIELDS)
        assert fake.requests[2]["map"] == "golden_flat"
        assert fake.requests[3]["jet"] == pytest.approx(0.8) and fake.requests[3]["pos"] == [5000.0, 5000.0, 16.0]
        # native wire: move=1 arrives as 1.007 -> snapped to exactly 1.0
        assert ctx.weapon_system.decode_action_update(NATIVE_FORWARD)
        assert ctx.weapon_system.behavior_slots[2] == pytest.approx(1.007, abs=1e-3)
        clock.t += 0.039
        com.advance(ctx)
        assert fake.steps == 0                            # not a full tick yet
        clock.t += 0.001 + 0.080                          # -> 3 ticks due
        com.advance(ctx)
        step = [r for r in fake.requests if r["op"] == "step"][-1]
        assert step["tick"] == 1 and step["count"] == 3
        ctl = step["inputs"][str(ctx.session.player_id)]
        assert ctl[:3] == [0.0, 1.0, 0.0] and ctl[3] == pytest.approx(0.8) and ctl[4:] == [0.0, 0.0, 0.0]
        assert ctx.player_pos == (5003.0, 5000.0, 16.0)
        assert ctx.debug_last_controller_step["backend"] == "community"
        # backlog is capped at 12 ticks (500 ms), like the client
        clock.t += 5.0
        com.advance(ctx)
        last = [r for r in fake.requests if r["op"] == "step"][-1]
        assert last["count"] == 12
        # idle restatement -> move 0
        assert ctx.weapon_system.decode_action_update(NATIVE_IDLE)
        assert com.controls(ctx, 0.8)[:3] == [0.0, 0.0, 0.0]
    finally:
        fake.close()


def test_input_mapping_signs_and_snap():
    fake = FakeSidecar()
    try:
        srv, ctx, _ = managed(fake)
        com = srv.community_physics
        ws = ctx.weapon_system
        q = ws.control_range / ((1 << ws.control_bits) - 2)
        ws.behavior_slots[1] = -16 * q        # native turn +0.5 arrives as -0.488
        ws.behavior_slots[2] = 26 * q         # 0.8 arrives as 0.7935
        ws.behavior_slots[3] = 33 * q         # strafe slot +1.007
        ws.behavior_slots[4] = 1.0
        turn, move, strafe, jet, tilt, roll, jump = com.controls(ctx, 0.8)
        assert (turn, move, strafe, jump) == (0.5, 0.8, -1.0, 1.0)
        ws.behavior_slots[2] = 0.47           # far from a 0.05 multiple: kept
        assert com.controls(ctx, 0.8)[1] == pytest.approx(0.47)
        ctx.last_action_packet_time = time.monotonic() - 10.0
        srv.input_stale_timeout_s = 1.5
        assert com.controls(ctx, 0.8)[:3] == [0.0, 0.0, 0.0]
    finally:
        fake.close()


def test_teleport_reseeds_body_and_respawn_opens_new_world():
    fake = FakeSidecar()
    try:
        srv, ctx, clock = managed(fake)
        com = srv.community_physics
        com.advance(ctx)
        clock.t += 0.04
        com.advance(ctx)
        ctx.player_pos = (100.0, 200.0, 30.0)               # server-side teleport
        clock.t += 0.04
        com.advance(ctx)
        ops = [r["op"] for r in fake.requests]
        assert ops[-3:] == ["remove_body", "create_body", "step"]
        assert fake.requests[-2]["pos"] == [100.0, 200.0, 30.0]
        ctx.session.last_spawn_time = 2.0                   # respawn: new generation
        com.advance(ctx)
        assert fake.connections == 2
    finally:
        fake.close()


def test_unreachable_sidecar_falls_back_once(capsys):
    srv = make_server(WULFRAM_NATIVE_COMMUNITY_PHYSICS="1", WULFRAM_NATIVE_PHYSICS_SESSION="1")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                            # nothing listens there
    srv.community_physics = cp.CommunityPhysics(srv, addr=("127.0.0.1", port), sync=True, timeout=0.5)
    ctx = native_ctx(srv)
    send_phys(srv, ctx)
    go_in_game(ctx)
    before = ctx.player_pos
    for _ in range(3):
        assert srv.community_physics.advance(ctx) is False
    out = capsys.readouterr().out
    assert out.count("sidecar unavailable") == 1 and "legacy sim" in out
    assert ctx.player_pos == before and not getattr(ctx, "community_physics_active", False)
    assert NativeLiveWorld.manages(ctx) is True             # legacy / native worker resumes
    ctx.session.last_spawn_time = 9.0                        # next life retries
    assert srv.community_physics.advance(ctx) is False
    assert capsys.readouterr().out.count("sidecar unavailable") == 1


def test_sidecar_error_mid_session_falls_back(capsys):
    fake = FakeSidecar(fail_after_steps=1)
    try:
        srv, ctx, clock = managed(fake)
        com = srv.community_physics
        assert com.advance(ctx) is True
        clock.t += 0.04
        assert com.advance(ctx) is True
        clock.t += 0.04
        assert com.advance(ctx) is False
        assert com.advance(ctx) is False
        assert capsys.readouterr().out.count("sidecar unavailable") == 1
    finally:
        fake.close()


def test_worker_thread_mode_publishes():
    fake = FakeSidecar()
    try:
        srv, ctx, _ = managed(fake)
        com = cp.CommunityPhysics(srv, addr=fake.addr, sync=False)
        srv.community_physics = com
        ctx.weapon_system.decode_action_update(NATIVE_FORWARD)
        assert com.advance(ctx) is True
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and ctx.player_pos[0] < 5003.0:
            time.sleep(0.05)
            com.advance(ctx)
        assert ctx.player_pos[0] >= 5003.0
        com.close()
    finally:
        fake.close()


def test_worker_mode_unreachable_falls_back(capsys):
    srv = make_server(WULFRAM_NATIVE_COMMUNITY_PHYSICS="1", WULFRAM_NATIVE_PHYSICS_SESSION="1")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    com = cp.CommunityPhysics(srv, addr=("127.0.0.1", port), sync=False, timeout=0.5)
    srv.community_physics = com
    ctx = native_ctx(srv)
    send_phys(srv, ctx)
    go_in_game(ctx)
    before = ctx.player_pos
    assert com.advance(ctx) is True          # connect happens on the worker, not under FRAME_LOCK
    deadline = time.monotonic() + 5.0
    while com.advance(ctx) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert com.advance(ctx) is False and ctx.player_pos == before
    assert capsys.readouterr().out.count("sidecar unavailable") == 1


def test_spawn_z_settles_native_only():
    fake = FakeSidecar()
    try:
        srv, ctx, _ = managed(fake)
        srv.terrain = SimpleNamespace(get_height=lambda x, y: 10.0)
        srv.up_axis = "z"
        assert srv.community_physics.spawn_z(ctx, (5000.0, 5000.0, 3.25)) == pytest.approx(14.447)
        assert srv.community_physics.spawn_z(og_ctx(srv), (5000.0, 5000.0, 3.25)) is None
    finally:
        fake.close()


# --------------------------------------------------------------------- end to end (real sidecar)
SIDECAR = os.environ.get("COMMUNITY_PHYSICS_SIDECAR")          # path of the community-physics project
SHAPES = os.environ.get("COMMUNITY_PHYSICS_SHAPES")
GOLDEN = os.environ.get("COMMUNITY_PHYSICS_GOLDEN")            # dir holding flat__forward.json


def _flat_land(path, n=129, size=10000.0, h=10.0):
    lines = ["%dx%d" % (n, n), "%fx%f" % (size, size)] + ["0 %r" % h] * (n * n)
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.skipif(not (SIDECAR and SHAPES and GOLDEN), reason="real sidecar / shapes / golden not configured")
def test_end_to_end_forward_5s_matches_golden(tmp_path):
    golden_path = Path(GOLDEN) / "flat__forward.json"
    if not golden_path.exists():
        pytest.skip("flat__forward.json absent")
    golden = json.loads(golden_path.read_text())
    maps = tmp_path / "maps" / "golden_flat"
    maps.mkdir(parents=True)
    _flat_land(maps / "land")
    cmd = ["uv", "run", "--no-project", "--python", "3.13", "python", "-m", "community_physics",
           "--addr", "127.0.0.1:0", "--maps", str(tmp_path / "maps"), "--shapes", SHAPES, "--stdin-watch"]
    proc = subprocess.Popen(cmd, cwd=SIDECAR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        assert line.startswith("LISTENING"), line
        host, port = line.split()[1].rsplit(":", 1)
        srv = make_server(WULFRAM_NATIVE_COMMUNITY_PHYSICS="1", WULFRAM_NATIVE_PHYSICS_SESSION="1")
        clock = FakeClock()
        srv.community_physics = cp.CommunityPhysics(srv, addr=(host, int(port)), sync=True, clock=clock,
                                                    timeout=10.0)
        srv.map_name = "golden_flat"
        ctx = native_ctx(srv)
        assert send_phys(srv, ctx) == 40
        go_in_game(ctx, pos=tuple(golden["spawn"]))
        # scripted native session: the client's own ACTION_UPDATE bytes, 50 idle / 125 forward / 50 idle
        script = [NATIVE_IDLE] * 50 + [NATIVE_FORWARD] * 125 + [NATIVE_IDLE] * 50
        assert srv.community_physics.advance(ctx) is True
        traj = [ctx.player_pos]
        for pkt in script:
            assert ctx.weapon_system.decode_action_update(pkt)
            ctx.last_action_packet_time = time.monotonic()
            clock.t += 0.040
            assert srv.community_physics.advance(ctx) is True
            traj.append(ctx.player_pos)
        rows = golden["rows"]
        assert len(traj) == len(rows) == 226
        worst = max(max(abs(a - b) for a, b in zip(p, r["pos"])) for p, r in zip(traj, rows))
        exact = sum(1 for p, r in zip(traj, rows) if list(p) == r["pos"])
        travelled = rows[175]["pos"][0] - rows[50]["pos"][0]
        print(f"e2e forward: rows={len(rows)} exact={exact} worst_pos_err={worst:.3e} "
              f"forward_5s={traj[175][0] - traj[50][0]:.3f}u (golden {travelled:.3f}u)")
        assert exact == len(rows), worst                    # bit-exact against the client's own step
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
