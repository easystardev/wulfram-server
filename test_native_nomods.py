#!/usr/bin/env python3
"""Native WulframClient.exe --no-mods bring-up knobs (community/native, all default off).

Covers server_config.init_native_client_profile and every item it gates:
UDP 0x4D, UDP HELLO 13 03, native UDP ping replies (+ the pre-bind ping crash fix),
WORLD_STATS once in the login burst, short-form D_ACK of the native 0x33 record,
the native server D_HANDSHAKE, one message per UDP datagram, and the 0x56 physics
session (encoder re-parsed with the ghidra RESULT.md rules).  Every item also has
an OG check: the OG-shaped session's bytes are identical with the knobs off and on.
All values are synthetic except the native client's public D_HANDSHAKE table.
Run: uv run --with pytest python -m pytest -q test_native_nomods.py
"""
import math
import os
import hashlib
import struct
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram import handlers, native_session, physics_session  # noqa: E402
from wulfram.server import WulframServer  # noqa: E402
from wulfram.server_config import init_native_client_profile  # noqa: E402
from wulfram.server_net import NetMixin  # noqa: E402
from wulfram.session import Phase, Session  # noqa: E402
from wulfram.transport import UDPHandler  # noqa: E402

NATIVE_D_HANDSHAKE = bytes.fromhex(
    "03" "00000110" "00000000" "00000001"
    "0010" "506c61796572207265717565737473" "00"
    "0000000a" "00000020" "00000025" "00000026" "0000002b" "0000002e"
    "00000033" "0000003b" "00000042" "00000047" "00000049"
    "0000000a"
    "0000002000000003" "0000002500000003" "0000002600000003" "0000002b00000003"
    "0000002e00000003" "0000003300000003" "0000003b00000003" "0000004200000003"
    "0000004700000003" "0000004900000003"
)
SYNTH_CREDENTIAL = "AAAAbbbbCCCCddddEEEE-_09"
NATIVE_KNOBS = (
    "WULFRAM_NATIVE_NOMODS_PROFILE", "WULFRAM_NATIVE_IDENTIFIED_UDP",
    "WULFRAM_NATIVE_HELLO_VERIFIED_UDP", "WULFRAM_NATIVE_UDP_PING_REPLY",
    "WULFRAM_NATIVE_LOGIN_WORLD_STATS", "WULFRAM_NATIVE_SHORT_ACK",
    "WULFRAM_NATIVE_SERVER_D_HANDSHAKE", "WULFRAM_NATIVE_UDP_SINGLE_MESSAGE",
    "WULFRAM_NATIVE_PHYSICS_SESSION", "WULFRAM_NATIVE_PHYSICS_SESSION_NAME",
    "WULFRAM_NATIVE_PHYSICS_SESSION_TICK_MS",
)
WORLD_STATS = b"\x16" + b"\x00" * 8  # stand-in for server.build_world_stats_packet()


def lp(text: str) -> bytes:
    raw = text.encode() + b"\x00"
    return struct.pack(">H", len(raw)) + raw


@contextmanager
def env(**values):
    saved = {k: os.environ.get(k) for k in set(NATIVE_KNOBS) | set(values)}
    for k in NATIVE_KNOBS:
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
    def __init__(self, on_send=None):
        self.sent = []
        self.on_send = on_send
        self.sock = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 26270), settimeout=lambda *_: None)

    def send(self, payload, log=True):
        self.sent.append(bytes(payload))
        if self.on_send:
            self.on_send(bytes(payload))


class FakeUDP:
    def __init__(self):
        self.sent = []

    def send_to(self, data, addr, log=True):
        self.sent.append((bytes(data), addr))


def make_ctx(i=1, *, phase=Phase.LOGIN, host="10.10.10.2"):
    return SimpleNamespace(client_id=i, client_addr=(host, 40000 + i), running=True,
                           entity_id=1337 + i, udp_socket=object(), tcp_handler=FakeTCP(),
                           tick_offset=None,
                           session=Session(phase=phase, udp_config_sent_time=1.0))


def make_net(**knobs):
    """NetMixin-only server with the native profile parsed from ``knobs``."""
    net = NetMixin()
    net.clients = {}
    net.clients_lock = threading.Lock()
    net.udp_addr_to_client = {}
    net.session_key_to_client = {}
    net.udp_handler = FakeUDP()
    net.debug_viewpoint = False
    net.debug_udp_raw = False
    net.use_client_ticks = False
    net.running = True
    net.public_addr = "127.0.0.1"
    net.port = 26270
    net.tick_rate_hz = 30.0
    net.build_world_stats_packet = lambda: WORLD_STATS
    with env(**knobs):
        init_native_client_profile(net)
    return net


ALL_ON = {"WULFRAM_NATIVE_NOMODS_PROFILE": "1"}


def native_identified(ctx, addr=("10.10.10.2", 50724)):
    ctx.session.hello_key_pushed = True
    ctx.session.udp_identify_kind = native_session.IDENTIFY_KIND_HELLO_KEY
    ctx.session.udp_addr = addr
    ctx.session.udp_verified = True
    return ctx


def og_identified(ctx, addr=("10.10.10.3", 50725)):
    ctx.session.udp_identify_kind = native_session.IDENTIFY_KIND_HELLO_THERE
    ctx.session.udp_addr = addr
    ctx.session.udp_verified = True
    return ctx


def with_handshake(ctx, raw):
    parsed = handlers._parse_empirical_client_d_handshake(raw)
    ctx.session.client_d_handshake = {k: parsed[k] for k in ("receive_window", "streams", "private_modes")}
    return ctx


# --------------------------------------------------------------------- config


def test_knobs_default_off_and_umbrella():
    net = make_net()
    for attr in ("native_identified_udp", "native_hello_verified_udp", "native_udp_ping_reply",
                 "native_login_world_stats", "native_short_ack", "native_udp_single_message",
                 "native_physics_session"):
        assert getattr(net, attr) is False, attr
    assert net.native_server_d_handshake == ""
    on = make_net(**ALL_ON)
    assert all(getattr(on, a) for a in ("native_identified_udp", "native_hello_verified_udp",
                                        "native_udp_ping_reply", "native_login_world_stats",
                                        "native_short_ack", "native_udp_single_message",
                                        "native_physics_session"))
    assert on.native_server_d_handshake == "empty"
    partial = make_net(WULFRAM_NATIVE_NOMODS_PROFILE="1", WULFRAM_NATIVE_SHORT_ACK="0",
                       WULFRAM_NATIVE_SERVER_D_HANDSHAKE="og_safe")
    assert partial.native_short_ack is False and partial.native_identified_udp is True
    assert partial.native_server_d_handshake == "og_safe"
    assert make_net(WULFRAM_NATIVE_SERVER_D_HANDSHAKE="1").native_server_d_handshake == "empty"
    assert make_net(WULFRAM_NATIVE_SERVER_D_HANDSHAKE="bogus").native_server_d_handshake == ""


def test_full_server_parses_profile_default_off():
    with env():
        srv = WulframServer(host="127.0.0.1", port=0)
    assert srv.native_physics_session is False and srv.native_server_d_handshake == ""
    with env(**ALL_ON):
        srv = WulframServer(host="127.0.0.1", port=0)
    assert srv.native_physics_session is True and srv.native_udp_single_message is True


def test_native_classification_is_protocol_observed():
    og = og_identified(make_ctx(1))
    assert not native_session.is_native_session(og)
    pushed_only = make_ctx(2)
    pushed_only.session.hello_key_pushed = True  # OG that later bound via 0x08
    pushed_only.session.udp_identify_kind = native_session.IDENTIFY_KIND_HELLO_THERE
    assert not native_session.is_native_session(pushed_only)
    nat = native_identified(make_ctx(3))
    assert native_session.is_native_identify_session(nat) and native_session.is_native_session(nat)
    assert not native_session.is_native_credential_login(nat)
    nat.session.login_shape = "credential_only"
    assert native_session.is_native_credential_login(nat)
    win0 = with_handshake(make_ctx(4), NATIVE_D_HANDSHAKE)
    assert native_session.is_native_session(win0) and handlers.is_native_stream_client(win0)
    og15 = with_handshake(og_identified(make_ctx(5)), handlers._build_server_d_handshake(None))
    assert not native_session.is_native_session(og15)


# ------------------------------------------------------------ item 1: 0x4D on UDP


def _run_handshake(net, ctx, identify):
    net.clients[ctx.client_id] = ctx
    net.udp_handler.open_channel = lambda c: 50001

    def on_send(payload):
        identify(payload)
    ctx.tcp_handler.on_send = on_send
    net._do_handshake(ctx)
    return ctx


def _native_identify(net, ctx, addr=("10.10.10.2", 50724)):
    def identify(payload):
        if payload[:2] == b"\x13\x02":  # pushed key -> client sends 13 01 <key> on UDP
            key = payload[4:].rstrip(b"\x00").decode()
            net._handle_single_udp_packet(ctx, b"\x13\x01" + lp(key), addr)
    return identify


def _og_identify(net, ctx, addr=("10.10.10.3", 50725)):
    def identify(payload):
        if payload[:2] == b"\x13\x01":  # OG answers HELLO1 at once with 0x08 "Hello There"
            text = b"Hello There\x00"
            net._handle_single_udp_packet(ctx, b"\x08" + struct.pack(">H", len(text)) + text, addr)
    return identify


def test_identified_udp_sent_on_udp_for_native_only():
    saved = {k: os.environ.get(k) for k in ("WULFRAM_HELLO_KEY_FALLBACK_S", "WULFRAM_LOGIN_BOOTSTRAP")}
    os.environ["WULFRAM_HELLO_KEY_FALLBACK_S"] = "0.1"
    os.environ["WULFRAM_LOGIN_BOOTSTRAP"] = "hybrid"
    try:
        results = {}
        for label, knobs in (("off", {}), ("on", {"WULFRAM_NATIVE_IDENTIFIED_UDP": "1"}), ("all", ALL_ON)):
            net = make_net(**knobs)
            ctx = make_ctx(1, phase=Phase.HANDSHAKE)
            _run_handshake(net, ctx, _native_identify(net, ctx))
            assert ctx.session.hello_key_pushed and native_session.is_native_identify_session(ctx)
            udp = [d for d, _ in net.udp_handler.sent]
            assert (b"\x4d" in udp) is (label != "off"), (label, udp)
            if label != "off":
                assert net.udp_handler.sent[udp.index(b"\x4d")][1] == ctx.session.udp_addr
            assert b"\x4d" in ctx.tcp_handler.sent  # TCP copy unchanged
            # OG: binds via 0x08 before any key push; identical bytes, never UDP 0x4D.
            net = make_net(**knobs)
            og = make_ctx(2, phase=Phase.HANDSHAKE)
            _run_handshake(net, og, _og_identify(net, og))
            assert not og.session.hello_key_pushed
            assert og.session.udp_identify_kind == native_session.IDENTIFY_KIND_HELLO_THERE
            results[label] = (og.tcp_handler.sent, net.udp_handler.sent)
        assert results["off"] == results["on"] == results["all"]
        assert results["off"][1] == []
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# --------------------------------------------- item 1: login burst 13 03 + WORLD_STATS


def _login(net, ctx, pkt):
    handlers.handle_login_request(net, ctx, pkt)
    return [p[0] for p in ctx.tcp_handler.sent], list(net.udp_handler.sent)


NATIVE_LOGIN = b"\x21\x03" + lp("synthpil") + lp(SYNTH_CREDENTIAL)
MD5 = "0123456789abcdef0123456789abcdef"
OG_LOGIN = (b"\x21\x03" + lp("synthpil") + lp("pw") + struct.pack(">I", 2)
            + lp(MD5) + lp(MD5[::-1]) + lp("server"))


def test_native_login_burst_hello_on_udp_and_world_stats_once():
    net = make_net(**ALL_ON)
    ctx = native_identified(make_ctx())
    ops, udp = _login(net, ctx, NATIVE_LOGIN)
    assert udp == [(b"\x13\x03", ctx.session.udp_addr)]
    assert ops.count(0x16) == 1 and ctx.session.world_stats_sent
    assert ops.index(0x32) < ops.index(0x16), "WORLD_STATS after TRANSLATION, in the burst"
    # the latch keeps every later WORLD_STATS sender quiet (BPS fallback path)
    ctx.tcp_handler.sent.clear()
    handlers.handle_bps(net, ctx, b"\x4e" + struct.pack(">I", 3))
    assert 0x16 not in [p[0] for p in ctx.tcp_handler.sent]
    # knobs off: native burst unchanged from today (no UDP, no WORLD_STATS)
    net = make_net()
    ctx = native_identified(make_ctx())
    ops, udp = _login(net, ctx, NATIVE_LOGIN)
    assert udp == [] and 0x16 not in ops
    # individual knobs
    net = make_net(WULFRAM_NATIVE_HELLO_VERIFIED_UDP="1")
    ops, udp = _login(net, native_identified(make_ctx()), NATIVE_LOGIN)
    assert len(udp) == 1 and 0x16 not in ops
    net = make_net(WULFRAM_NATIVE_LOGIN_WORLD_STATS="1")
    ops, udp = _login(net, native_identified(make_ctx()), NATIVE_LOGIN)
    assert udp == [] and ops.count(0x16) == 1


def test_login_burst_unchanged_for_og_and_non_native_credential_only():
    outputs = []
    for knobs in ({}, ALL_ON):
        net = make_net(**knobs)
        og = og_identified(make_ctx())
        _login(net, og, OG_LOGIN)
        outputs.append((og.tcp_handler.sent, net.udp_handler.sent))
    assert outputs[0] == outputs[1] and outputs[0][1] == []
    # credential-only login whose UDP bound the OG way: not native, no extras
    net = make_net(**ALL_ON)
    ctx = og_identified(make_ctx())
    ops, udp = _login(net, ctx, NATIVE_LOGIN)
    assert udp == [] and 0x16 not in ops


# --------------------------------------------------------------- item 1: pings


def _ping_server(**knobs):
    with env(**knobs):
        srv = WulframServer(host="127.0.0.1", port=0)
    srv.udp_handler = FakeUDP()
    srv.clients = {}
    srv.udp_addr_to_client = {}
    srv.session_key_to_client = {}
    return srv


def test_prebind_ping_no_longer_crashes():
    srv = _ping_server()
    ctx = make_ctx(phase=Phase.HANDSHAKE)  # SimpleNamespace: no ctx.udp_addr attribute at all
    assert srv._udp_ping_reply_allowed_for_client(ctx) is False
    ping = b"\x0b" + struct.pack(">II", 7, 0)
    srv._handle_single_udp_packet(ctx, ping, ("10.10.10.2", 50724))  # used to raise AttributeError
    assert srv.udp_handler.sent == []


def test_native_ping_reply_only_with_knob_and_only_native():
    ping = b"\x0b" + struct.pack(">II", 7, 0)
    for knobs, expect_native in (({}, False), ({"WULFRAM_NATIVE_UDP_PING_REPLY": "1"}, True), (ALL_ON, True)):
        srv = _ping_server(**knobs)
        nat = native_identified(make_ctx(1))
        srv._handle_single_udp_packet(nat, ping, nat.session.udp_addr)
        assert (len(srv.udp_handler.sent) == 1) is expect_native
        if expect_native:
            assert srv.udp_handler.sent[0][0][0] == 0x0C
        srv.udp_handler.sent.clear()
        og = og_identified(make_ctx(2))  # non-loopback OG: still suppressed by the allowlist
        srv._handle_single_udp_packet(og, ping, og.session.udp_addr)
        assert srv.udp_handler.sent == []


# --------------------------------------------------------- item 2: short-form ACK


TRANSLATION_ACK = bytes.fromhex("33" "0001" "0009" "00000001")


def test_native_0x33_record_gets_short_ack():
    net = make_net(WULFRAM_NATIVE_SHORT_ACK="1")
    ctx = with_handshake(native_identified(make_ctx(phase=Phase.TEAM_SELECT)), NATIVE_D_HANDSHAKE)
    assert list(net._parse_udp_datagram(TRANSLATION_ACK, ctx)) == [TRANSLATION_ACK]
    net._handle_single_udp_packet(ctx, TRANSLATION_ACK, ctx.session.udp_addr)
    assert net.udp_handler.sent == [(bytes.fromhex("0201330001"), ctx.session.udp_addr)]
    assert ctx.session.translation_ack_received
    # other "Player requests" records too (0x25 REINCARNATE seq 0x1234)
    handlers.send_udp_ack(net, ctx, ctx.session.udp_addr, 0x25, 0x1234)
    assert net.udp_handler.sent[-1][0] == bytes.fromhex("0201251234")
    # knob off: today's framed upstream form for window-0 clients
    net = make_net()
    net._handle_single_udp_packet(ctx, TRANSLATION_ACK, ctx.session.udp_addr)
    assert net.udp_handler.sent[0][0] == bytes.fromhex("020001000901330001")


def test_og_ack_bytes_identical_with_short_ack_knob():
    og_shake = handlers._build_server_d_handshake(None)
    outputs = []
    for knobs in ({}, ALL_ON):
        net = make_net(**knobs)
        og = with_handshake(og_identified(make_ctx(phase=Phase.TEAM_SELECT)), og_shake)
        net._handle_single_udp_packet(og, TRANSLATION_ACK, og.session.udp_addr)
        handlers.send_udp_ack(net, og, og.session.udp_addr, 0x26, 9)
        outputs.append(net.udp_handler.sent)
    assert outputs[0] == outputs[1] == [(bytes.fromhex("0201330001"), ("10.10.10.3", 50725)),
                                        (bytes.fromhex("0201260009"), ("10.10.10.3", 50725))]


# ------------------------------------------------------- item 3: server D_HANDSHAKE


def parse_native_server_handshake(raw: bytes) -> dict:
    """Client rules (ghidra RESULT.md Q2): exact layout, no tail, route rules."""
    off = 0

    def take(n):
        nonlocal off
        if off + n > len(raw):
            raise ValueError("packet ended inside a field")
        out = raw[off:off + n]
        off += n
        return out

    assert take(1) == b"\x03"
    seq, window, n_groups = struct.unpack(">III", take(12))
    assert n_groups <= 87
    groups, members = [], set()
    for _ in range(n_groups):
        (ln,) = struct.unpack(">H", take(2))
        name = take(ln).rstrip(b"\x00").decode()
        (n,) = struct.unpack(">I", take(4))
        assert n <= 87
        ops = [struct.unpack(">I", take(4))[0] for _ in range(n)]
        assert all(op <= 0x56 for op in ops)
        assert not members & set(ops), "Duplicate stream group member"
        members |= set(ops)
        groups.append((name, ops))
    (n_routes,) = struct.unpack(">I", take(4))
    assert n_routes <= 87
    routes, seen = [], set()
    for _ in range(n_routes):
        op, mode = struct.unpack(">II", take(8))
        assert op <= 0x56 and mode <= 3
        assert op not in seen and mode != 2 and not (op < 5 and mode != 0), "Unsupported stream route"
        seen.add(op)
        routes.append((op, mode))
    assert off == len(raw), "unexpected packet tail"
    return {"seq": seq, "window": window, "groups": groups, "routes": routes}


def test_native_server_d_handshake_empty_and_alone():
    for knobs in ({"WULFRAM_NATIVE_SERVER_D_HANDSHAKE": "empty"}, ALL_ON):
        net = make_net(**knobs)
        ctx = native_identified(make_ctx(phase=Phase.TEAM_SELECT))
        net.clients[1] = ctx
        handlers.handle_udp_d_handshake(net, ctx, NATIVE_D_HANDSHAKE, ctx.session.udp_addr)
        sent = [d for d, _ in net.udp_handler.sent]
        assert len(sent) == 2 and sent[0][:2] == b"\x02\x00" and len(sent[0]) == 6
        shake = sent[1]
        assert len(shake) == 17 and shake[5:] == bytes.fromhex("0000000f" "00000000" "00000000")
        parsed = parse_native_server_handshake(shake)
        assert parsed["groups"] == [] and parsed["routes"] == []
        assert all(len(native_session.split_single_message(d)) == 1 for d in sent)


def test_native_server_d_handshake_og_safe_passes_route_rules():
    net = make_net(WULFRAM_NATIVE_SERVER_D_HANDSHAKE="og_safe")
    ctx = native_identified(make_ctx(phase=Phase.TEAM_SELECT))
    net.clients[1] = ctx
    handlers.handle_udp_d_handshake(net, ctx, NATIVE_D_HANDSHAKE, ctx.session.udp_addr)
    parsed = parse_native_server_handshake(net.udp_handler.sent[1][0])
    assert [name for name, _ in parsed["groups"]] == ["Beacon Stream", "Squad Things"]
    assert parsed["routes"] and all(op >= 5 for op, _ in parsed["routes"])
    # the rule filter itself drops what the client rejects
    saved = handlers._OG_D_HANDSHAKE_PRIVATE_MODES
    handlers._OG_D_HANDSHAKE_PRIVATE_MODES = ((0x01, 1), (0x03, 0), (0x30, 2), (0x40, 3), (0x40, 1))
    try:
        assert handlers._native_d_handshake_routes() == ((0x03, 0), (0x40, 3))
    finally:
        handlers._OG_D_HANDSHAKE_PRIVATE_MODES = saved


def test_og_server_d_handshake_unchanged_with_knob():
    og_client_shake = handlers._build_server_d_handshake(None)  # window 15, OG-shaped
    outputs = []
    for knobs in ({}, ALL_ON, {"WULFRAM_NATIVE_SERVER_D_HANDSHAKE": "og_safe"}):
        net = make_net(**knobs)
        og = og_identified(make_ctx(phase=Phase.TEAM_SELECT))
        net.clients[1] = og
        handlers.handle_udp_d_handshake(net, og, og_client_shake, og.session.udp_addr)
        sent = [d for d, _ in net.udp_handler.sent]
        # strip the time-based ACK timestamp and handshake sequence
        outputs.append((len(sent), sent[0][:2], sent[1][:1] + sent[1][5:]))
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0][2] == og_client_shake[:1] + og_client_shake[5:]


# ------------------------------------------------- item 4: one message per datagram


class FakeSock:
    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append(bytes(data))


def _udp_handler(owner, single):
    sock = FakeSock()
    udp = UDPHandler(sock)
    udp.peer_owner = lambda addr: owner
    udp.native_single_message = single
    owner.udp_socket = None  # use the main socket
    return udp, sock


def test_split_single_message_rules():
    shake = handlers._build_native_server_d_handshake("empty", 1, 15)
    ack = bytes.fromhex("0200" "00000005")
    set_start = bytes.fromhex("04010001")
    assert native_session.split_single_message(ack + shake + set_start + set_start) == [ack, shake, set_start, set_start]
    assert native_session.split_single_message(b"\x4d" + b"\x13\x03") == [b"\x4d", b"\x13\x03"]
    framed = bytes.fromhex("020001000901330001")
    assert native_session.split_single_message(framed) == [framed]
    short = bytes.fromhex("0201330001")
    assert native_session.split_single_message(short + short) == [short, short]
    va = b"\x0e" + b"\x01" * 20  # ordinary message: whole, never cut
    assert native_session.split_single_message(va) == [va]


def test_udp_handler_never_coalesces_for_native_only():
    bundle = bytes.fromhex("020000000005") + handlers._build_native_server_d_handshake("empty", 1, 15)
    nat = with_handshake(native_identified(make_ctx()), NATIVE_D_HANDSHAKE)
    udp, sock = _udp_handler(nat, True)
    udp.send_to(bundle, nat.session.udp_addr)
    assert sock.sent == [bundle[:6], bundle[6:]]
    udp, sock = _udp_handler(nat, False)  # knob off: unchanged
    udp.send_to(bundle, nat.session.udp_addr)
    assert sock.sent == [bundle]
    og = og_identified(make_ctx(2))
    udp, sock = _udp_handler(og, True)  # OG: unchanged even with the knob on
    udp.send_to(bundle, og.session.udp_addr)
    assert sock.sent == [bundle]


def test_native_bringup_sends_one_message_per_datagram():
    net = make_net(**ALL_ON)
    ctx = native_identified(make_ctx())
    net.clients[1] = ctx
    _login(net, ctx, NATIVE_LOGIN)
    handlers.handle_udp_d_handshake(net, ctx, NATIVE_D_HANDSHAKE, ctx.session.udp_addr)
    net._handle_single_udp_packet(ctx, TRANSLATION_ACK, ctx.session.udp_addr)
    sent = [d for d, _ in net.udp_handler.sent]
    assert [d[0] for d in sent] == [0x13, 0x02, 0x03, 0x02]
    assert all(len(native_session.split_single_message(d)) == 1 for d in sent)


# --------------------------------------------------------- item 5: 0x56 encoder


class Reader:
    """Big-endian byte reader mirroring the client's 0x56 field readers."""

    def __init__(self, data):
        self.data, self.off = data, 0

    def take(self, n):
        if self.off + n > len(self.data):
            raise ValueError("packet ended inside a field")
        out = self.data[self.off:self.off + n]
        self.off += n
        return out

    def u8(self):
        return self.take(1)[0]

    def u16(self):
        return struct.unpack(">H", self.take(2))[0]

    def u32(self):
        return struct.unpack(">I", self.take(4))[0]

    def string(self):
        n = self.u16()
        assert n <= 4096, "invalid protocol string length"
        raw = self.take(n)
        return (raw[:-1] if raw.endswith(b"\x00") else raw).decode()

    def f32(self):
        x = struct.unpack(">f", self.take(4))[0]
        assert math.isfinite(x) and abs(x) <= 1e12, "Invalid physics scalar"
        return x

    def f64(self):
        hi, lo = self.u32(), self.u32()
        x = struct.unpack(">d", struct.pack(">II", hi, lo))[0]
        assert math.isfinite(x) and abs(x) <= 1e12, "Invalid physics scalar"
        return x

    def boolean(self):
        b = self.u8()
        assert b <= 1, "Invalid physics boolean"
        return bool(b)

    def curve(self):
        n = self.u16()
        assert 2 <= n <= 128, "Invalid physics curve length"
        return [self.f32() for _ in range(n)]


def parse_physics_session(data: bytes) -> dict:
    """Independent re-implementation of FUN_14014e330 per ghidra RESULT.md Q3."""
    r = Reader(data)
    assert r.u8() == 0x56, "Invalid physics session opcode"
    magic = r.u32()
    assert magic in (0x57504331, 0x57504332), "Invalid physics session version"
    tick_ms = r.u32()
    assert 1 <= tick_ms <= 550, "Invalid physics tick duration"
    version = r.string()
    assert version == "0.7.0-dev", "This server runs physics X"
    name = r.string()
    vehicles = []
    for _ in range(2):
        f64 = [r.f64() for _ in range(17)]
        f32a = [r.f32() for _ in range(6)]
        feedbacks = []
        for _ in range(2):
            f = [r.f32() for _ in range(5)]
            enabled = r.boolean()
            curves = [r.curve() for _ in range(4)]
            assert f[0] > 0 and f[2] > 0, "Invalid physics feedback scale"
            feedbacks.append((f, enabled, curves))
        curves = [r.curve() for _ in range(5)]
        f32b = [r.f32() for _ in range(10)]
        flag = r.boolean()
        assert (f32a[0] > 0 and f64[12] > 0 and f64[4] > 0 and f64[6] > 0
                and f32b[0] > 0 and f64[9] >= 0), "Invalid vehicle physics limits"
        vehicles.append(dict(f64=f64, f32a=f32a, feedbacks=feedbacks, curves=curves, f32b=f32b, flag=flag))
    jets = [[([r.f32() for _ in range(6)], r.boolean()) for _ in range(4)] for _ in range(4)]
    tail = [r.f32(), r.f32()]
    vis = None
    if magic == 0x57504332:
        vis = r.f32()
        assert vis == 0.0 or 100.0 <= vis <= 1e6, "Invalid maximum visibility"
    rest = len(data) - r.off
    assert rest == 0, "unexpected packet tail or non-zero padding"
    return dict(magic=magic, tick_ms=tick_ms, version=version, name=name, vehicles=vehicles,
                jets=jets, tail=tail, max_visibility=vis)


def test_physics_session_encoder_reparses_with_client_rules():
    body = physics_session.encode_physics_session(tick_ms=33, config_name="wulfram-server")
    parsed = parse_physics_session(body)
    assert parsed["magic"] == 0x57504332 and parsed["tick_ms"] == 33
    assert parsed["version"] == "0.7.0-dev" and parsed["name"] == "wulfram-server"
    assert parsed["tail"] == [struct.unpack(">f", struct.pack(">f", 0.8))[0]] * 2
    assert parsed["max_visibility"] == 0.0
    wpc1 = physics_session.encode_physics_session(magic="WPC1", tick_ms=50)
    p1 = parse_physics_session(wpc1)
    assert p1["magic"] == 0x57504331 and p1["max_visibility"] is None and len(wpc1) == len(body) - 4
    assert physics_session.tick_ms_for_rate(30.0) == 33 and physics_session.tick_ms_for_rate(20.0) == 50
    # the table is ordered exactly like the client layout
    assert [(n, k) for n, k, _ in physics_session.CLIENT_DEFAULT_FIELDS] == physics_session.layout("WPC2")
    # the client's own default message (tick 40, its config name), byte-exact vs the RE reference
    ref = physics_session.encode_physics_session(
        tick_ms=40, config_name="wulfram-recovered-vehicle-defaults-v1")
    assert len(ref) == 2657
    assert hashlib.sha256(ref).hexdigest() == (
        "7f10ccad7b93f4371646ba1836dcb880a5b1ff278ab560009f4a5e96487ebf13")


def test_physics_session_encoder_rejects_what_the_client_rejects():
    base = list(physics_session.CLIENT_DEFAULT_FIELDS)

    def with_value(name, value):
        return [(n, k, value if n == name else v) for n, k, v in base]

    bad_tables = (
        with_value("vehicle[0:tank].jet_speed", (1.0,)),            # curve < 2 points
        with_value("vehicle[1:scout].maximum_velocity", 0.0),                # must be > 0
        with_value("vehicle[0:tank].gravity", -1.0),               # must be >= 0
        with_value("vehicle[0:tank].jet_slow_react.abs_max_prime", 0.0),      # feedback scale
        with_value("vehicle[1:scout].max_xy_speed", 0.0),
        with_value("max_visibility", 50.0),              # 0 or 100..1e6
        with_value("vehicle_fraction[0:tank]", float("nan")),
        with_value("vehicle[0:tank].degen_on_slope", 2),
        base[:-1],                                       # layout mismatch
    )
    for table in bad_tables:
        try:
            physics_session.encode_physics_session(table, tick_ms=33)
        except ValueError:
            continue
        raise AssertionError("encoder accepted an invalid table")
    for kwargs in ({"tick_ms": 0}, {"tick_ms": 551}, {"tick_ms": 33, "physics_version": "1.0"}, {"tick_ms": 33, "physics_version": ""},
                   {"tick_ms": 33, "magic": "WPC3"}):
        try:
            physics_session.encode_physics_session(base, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"encoder accepted {kwargs}")


PHYS_V2 = b"\x54" + lp("portable_physics_v2") + lp("") + struct.pack(">I", 1)
PHYS_V1 = b"\x54" + lp("portable_physics_v1") + lp("") + struct.pack(">I", 1)


def test_generic_want_parser():
    assert len(PHYS_V2) == 1 + 2 + 20 + 2 + 1 + 4  # 54 0014 "...\0" 0001 00 00000001
    assert native_session.portable_physics_want(PHYS_V2) == "portable_physics_v2"
    assert native_session.portable_physics_want(PHYS_V1) == "portable_physics_v1"
    assert native_session.portable_physics_want(PHYS_V2 + b"\x00") == ""
    assert native_session.portable_physics_want(b"\x54" + b"\x00" * 25) == ""
    other = b"\x54" + lp("want_voice_data") + lp("") + struct.pack(">I", 1)
    assert native_session.portable_physics_want(other) == ""


def test_physics_session_sent_once_after_translation_on_want_only():
    net = make_net(WULFRAM_NATIVE_PHYSICS_SESSION="1")
    ctx = native_identified(make_ctx())
    _login(net, ctx, NATIVE_LOGIN)
    assert ctx.session.translation_sent
    before = len(ctx.tcp_handler.sent)
    assert handlers.maybe_send_physics_session(net, ctx, PHYS_V2) is True
    assert handlers.maybe_send_physics_session(net, ctx, PHYS_V1) is True
    new = ctx.tcp_handler.sent[before:]
    assert len(new) == 1 and new[0][0] == 0x56
    parsed = parse_physics_session(new[0])
    assert parsed["magic"] == 0x57504332 and parsed["tick_ms"] == 33 and parsed["name"] == "wulfram-server"
    assert net.udp_handler.sent == []  # TCP only
    # v1-only client -> WPC1; name / tick overrides
    net = make_net(WULFRAM_NATIVE_PHYSICS_SESSION="1", WULFRAM_NATIVE_PHYSICS_SESSION_NAME="lab",
                   WULFRAM_NATIVE_PHYSICS_SESSION_TICK_MS="50")
    ctx = native_identified(make_ctx())
    _login(net, ctx, NATIVE_LOGIN)
    handlers.maybe_send_physics_session(net, ctx, PHYS_V1)
    p = parse_physics_session(ctx.tcp_handler.sent[-1])
    assert p["magic"] == 0x57504331 and p["tick_ms"] == 50 and p["name"] == "lab"
    # before TRANSLATION: not answered
    net = make_net(WULFRAM_NATIVE_PHYSICS_SESSION="1")
    ctx = native_identified(make_ctx())
    assert handlers.maybe_send_physics_session(net, ctx, PHYS_V2) is True
    assert ctx.tcp_handler.sent == [] and not ctx.session.physics_session_sent


def test_physics_session_knob_off_and_og_untouched():
    net = make_net()
    ctx = native_identified(make_ctx())
    _login(net, ctx, NATIVE_LOGIN)
    before = list(ctx.tcp_handler.sent)
    assert handlers.maybe_send_physics_session(net, ctx, PHYS_V2) is False
    assert ctx.tcp_handler.sent == before
    # OG 0x54 traffic (not a portable_physics want) is never consumed, knobs on
    net = make_net(**ALL_ON)
    og = og_identified(make_ctx())
    _login(net, og, OG_LOGIN)
    before = list(og.tcp_handler.sent)
    for pkt in (b"\x54" + b"\x00" * 25, b"\x54" + lp("want_voice_data") + lp("") + struct.pack(">I", 1)):
        assert handlers.maybe_send_physics_session(net, og, pkt) is False
    assert og.tcp_handler.sent == before


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
