#!/usr/bin/env python3
"""Accept-both ingest of the upstream native client (WulframClient.exe, 2026-10-16).

Every value here is synthetic except the D_HANDSHAKE stream declaration, which is
the native client's public "Player requests" table (no identity/credential in it).
Run: uv run python test_native_client_ingest.py
"""
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram import handlers, server_net  # noqa: E402
from wulfram.server import WulframServer  # noqa: E402
from wulfram.server_net import NetMixin  # noqa: E402
from wulfram.session import Phase, Session  # noqa: E402
from wulfram.transport import TCPHandler  # noqa: E402
from wulfram.codec import frame_packet  # noqa: E402

# Captured 2026-10-16 native D_HANDSHAKE: window 0, one stream "Player requests"
# {20,25,26,2B,2E,33,3B,42,47,49}, ten private modes all = 3.
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
assert len(NATIVE_D_HANDSHAKE) == 159
NATIVE_OPCODES = {0x20, 0x25, 0x26, 0x2B, 0x2E, 0x33, 0x3B, 0x42, 0x47, 0x49}


def og_d_handshake() -> bytes:
    """An OG-shaped client handshake (window 15) built with the server's own builder."""
    return handlers._build_server_d_handshake(None)


def lp(text: str) -> bytes:
    raw = text.encode() + b"\x00"
    return struct.pack(">H", len(raw)) + raw


SYNTH_CREDENTIAL = "AAAAbbbbCCCCddddEEEE-_09"  # 24 chars, base64url alphabet, synthetic


class FakeTCP:
    def __init__(self):
        self.sent = []
        self.sock = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 26270),
                                    settimeout=lambda *_: None)

    def send(self, payload, log=True):
        self.sent.append(bytes(payload))


class FakeUDP:
    def __init__(self):
        self.sent = []

    def send_to(self, data, addr, log=True):
        self.sent.append((bytes(data), addr))


def make_ctx(i=1, *, phase=Phase.LOGIN, udp_socket=None):
    return SimpleNamespace(client_id=i, client_addr=("127.0.0.1", 40000 + i), running=True,
                           entity_id=1337 + i, udp_socket=udp_socket, tcp_handler=FakeTCP(),
                           session=Session(phase=phase, udp_config_sent_time=1.0))


def make_net():
    net = NetMixin()
    net.clients = {}
    net.clients_lock = threading.Lock()
    net.udp_addr_to_client = {}
    net.session_key_to_client = {}
    net.udp_handler = FakeUDP()
    net.debug_viewpoint = False
    net.debug_udp_raw = False
    return net


# ------------------------------------------------------------------ D_HANDSHAKE


def test_native_d_handshake_window_zero_parses_empirical():
    parsed = handlers._parse_empirical_client_d_handshake(NATIVE_D_HANDSHAKE)
    assert parsed is not None, "window-0 handshake must not fall back to legacy"
    assert parsed["receive_window"] == 0
    assert parsed["consumed"] == len(NATIVE_D_HANDSHAKE)
    assert parsed["streams"] == (("Player requests", tuple(sorted(NATIVE_OPCODES))),) or \
        set(parsed["streams"][0][1]) == NATIVE_OPCODES
    assert {op for op, _ in parsed["private_modes"]} == NATIVE_OPCODES
    assert {mode for _, mode in parsed["private_modes"]} == {3}
    og = handlers._parse_empirical_client_d_handshake(og_d_handshake())
    assert og is not None and og["receive_window"] == 15


def test_parser_yields_whole_native_handshake():
    srv = WulframServer.__new__(WulframServer)
    srv.debug_viewpoint = False
    packets = list(srv._parse_udp_datagram(NATIVE_D_HANDSHAKE))
    assert packets == [NATIVE_D_HANDSHAKE]


def _ctx_with_handshake(raw: bytes):
    ctx = make_ctx(phase=Phase.TEAM_SELECT)
    parsed = handlers._parse_empirical_client_d_handshake(raw)
    ctx.session.client_d_handshake = {k: parsed[k] for k in ("receive_window", "streams", "private_modes")}
    return ctx


def test_native_shape_helpers_and_env():
    native = _ctx_with_handshake(NATIVE_D_HANDSHAKE)
    og = _ctx_with_handshake(og_d_handshake())
    assert handlers.client_declared_stream_opcodes(native) == frozenset(NATIVE_OPCODES)
    assert handlers.is_native_stream_client(native) and not handlers.is_native_stream_client(og)
    assert handlers.use_framed_stream_ack(native) and not handlers.use_framed_stream_ack(og)
    assert handlers.ack_declared_stream_packets_enabled(native)
    assert not handlers.ack_declared_stream_packets_enabled(og)
    os.environ["WULFRAM_UDP_ACK_FORM"] = "short"
    os.environ["WULFRAM_ACK_DECLARED_STREAM_PACKETS"] = "0"
    try:
        assert not handlers.use_framed_stream_ack(native)
        assert not handlers.ack_declared_stream_packets_enabled(native)
    finally:
        del os.environ["WULFRAM_UDP_ACK_FORM"]
        del os.environ["WULFRAM_ACK_DECLARED_STREAM_PACKETS"]


def test_ack_form_native_framed_og_short():
    net = make_net()
    native = _ctx_with_handshake(NATIVE_D_HANDSHAKE)
    og = _ctx_with_handshake(og_d_handshake())
    handlers.send_udp_ack(net, native, ("127.0.0.1", 1), 0x33, 1)
    handlers.send_udp_ack(net, native, ("127.0.0.1", 1), 0x25, 2)
    handlers.send_udp_ack(net, og, ("127.0.0.1", 2), 0x33, 1)
    # Upstream shape: 02 <u16 server seq> 0009 01 <op> <u16 seq>
    assert net.udp_handler.sent[0][0] == bytes.fromhex("020001000901330001")
    assert net.udp_handler.sent[1][0] == bytes.fromhex("020002000901250002")
    assert net.udp_handler.sent[2][0] == bytes.fromhex("0201330001")  # unchanged OG form


def test_declared_stream_packet_split_and_ack_native_only():
    net = make_net()
    native = _ctx_with_handshake(NATIVE_D_HANDSHAKE)
    og = _ctx_with_handshake(og_d_handshake())
    squad = bytes.fromhex("470007000900000001")  # synthetic 0x47, seq 7, len 9
    ping = bytes.fromhex("0b000007d000000000")
    assert list(net._parse_udp_datagram(squad + ping, native)) == [squad, ping]
    assert list(net._parse_udp_datagram(squad + ping, og)) == [squad + ping]  # OG unchanged
    native.session.udp_addr = ("127.0.0.1", 9)
    net._handle_single_udp_packet(native, squad, ("127.0.0.1", 9))
    assert net.udp_handler.sent[-1][0] == bytes.fromhex("0200010009014700" "07")
    before = len(net.udp_handler.sent)
    net._handle_single_udp_packet(og, squad, ("127.0.0.1", 9))
    assert len(net.udp_handler.sent) == before


def test_native_client_promoted_without_state_request():
    srv = WulframServer.__new__(WulframServer)
    srv.update_local_state_mode = "wf"
    srv.remote_full_local_state_delay = 99999.0  # the shipped .env value
    for raw, expect in ((NATIVE_D_HANDSHAKE, True), (og_d_handshake(), False)):
        ctx = _ctx_with_handshake(raw)
        ctx.session.in_game = True
        ctx.session.last_spawn_time = time.monotonic() - 2.0
        assert srv._maybe_promote_remote_full_local_state(ctx, reason="post_spawn") is expect
        assert bool(getattr(ctx, "remote_full_local_state_ready", False)) is expect
    ctx = _ctx_with_handshake(NATIVE_D_HANDSHAKE)
    ctx.session.in_game = True
    ctx.session.last_spawn_time = time.monotonic() - 0.2  # inside the 1.0 s settle
    assert srv._maybe_promote_remote_full_local_state(ctx, reason="post_spawn") is False
    os.environ["WULFRAM_NATIVE_FULL_LOCAL_STATE_DELAY_S"] = "0"
    try:
        ctx.session.last_spawn_time = time.monotonic() - 5.0
        assert srv._maybe_promote_remote_full_local_state(ctx, reason="post_spawn") is False
    finally:
        del os.environ["WULFRAM_NATIVE_FULL_LOCAL_STATE_DELAY_S"]


# ------------------------------------------------------------------ UDP identify


def test_udp_identify_unknown_key_binds_channel_owner_only():
    net = make_net()
    owner = make_ctx(1, phase=Phase.HANDSHAKE, udp_socket=object())
    other = make_ctx(2, phase=Phase.HANDSHAKE, udp_socket=object())
    net.clients = {1: owner, 2: other}
    other.session.session_key = "k-other"
    net.session_key_to_client["k-other"] = other
    ident = b"\x13\x01" + lp("f" * 64)  # synthetic, matches no session
    net._handle_single_udp_packet(owner, ident, ("127.0.0.1", 5001))
    assert owner.session.udp_verified and net.udp_addr_to_client[("127.0.0.1", 5001)] is owner
    # A key naming ANOTHER session on this owner's destination is still refused.
    net._handle_single_udp_packet(owner, b"\x13\x01" + lp("k-other"), ("127.0.0.1", 5002))
    assert ("127.0.0.1", 5002) not in net.udp_addr_to_client and not other.session.udp_verified
    # Shared-port (no dedicated channel) keeps the old NO MATCH behaviour.
    shared = make_ctx(3, phase=Phase.HANDSHAKE)
    net.clients[3] = shared
    net._handle_single_udp_packet(shared, ident, ("127.0.0.1", 5003))
    assert not shared.session.udp_verified


# ------------------------------------------------------------------ login


def test_credential_only_login_accepted_and_shape_recorded():
    net = make_net()
    ctx = make_ctx()
    pkt = b"\x21\x03" + lp("synthpil") + lp(SYNTH_CREDENTIAL)
    assert len(pkt) == 40  # same size as the captured native LOGIN_REQUEST
    handlers.handle_login_request(net, ctx, pkt)
    s = ctx.session
    assert s.login_complete and s.phase == Phase.TEAM_SELECT and s.username == "synthpil"
    assert s.login_shape == "credential_only"
    ops = [p[0] for p in ctx.tcp_handler.sent]
    assert 0x22 in ops and ctx.tcp_handler.sent[ops.index(0x22)][2] == 8
    assert 0x32 in ops, "native login burst carries TRANSLATION (upstream order)"
    assert ops.index(0x24) < ops.index(0x32)


def test_og_game_service_login_keeps_og_bootstrap():
    net = make_net()
    ctx = make_ctx()
    md5 = "0123456789abcdef0123456789abcdef"
    pkt = (b"\x21\x03" + lp("synthpil") + lp("pw") + struct.pack(">I", 2)
           + lp(md5) + lp(md5[::-1]) + lp("server"))
    handlers.handle_login_request(net, ctx, pkt)
    assert ctx.session.login_complete and ctx.session.login_shape == "sub_03"
    assert 0x32 not in [p[0] for p in ctx.tcp_handler.sent], "OG must not get TRANSLATION at login"


def test_credential_validator_hook():
    net = make_net()
    seen = []
    net.login_credential_validator = lambda c, u, cred, sub: seen.append((u, len(cred), sub)) or False
    ctx = make_ctx()
    try:
        handlers.handle_login_request(net, ctx, b"\x21\x03" + lp("synthpil") + lp(SYNTH_CREDENTIAL))
    except ConnectionAbortedError:
        pass
    else:
        raise AssertionError("rejected credential must abort the login")
    assert seen == [("synthpil", 24, 3)] and not ctx.session.login_complete
    net.login_credential_validator = lambda *a: True
    ctx2 = make_ctx(2)
    handlers.handle_login_request(net, ctx2, b"\x21\x03" + lp("synthpil") + lp(SYNTH_CREDENTIAL))
    assert ctx2.session.login_complete


def _socket_ctx():
    a, b = socket.socketpair()
    ctx = make_ctx()
    ctx.tcp_handler = TCPHandler(a)
    return ctx, a, b


def test_unprompted_login_detected_native_and_not_og():
    net = make_net()
    ctx, a, b = _socket_ctx()
    try:
        b.sendall(frame_packet(b"\x13\x00\x00\x00\x4e\x89") + frame_packet(b"\x54" + b"\x00" * 25)
                  + frame_packet(b"\x21\x03" + lp("synthpil") + lp(SYNTH_CREDENTIAL)))
        pending = []
        assert net._collect_unprompted_login(ctx, pending) is True
        assert [p[0] for p in pending] == [0x13, 0x54, 0x21]
    finally:
        a.close(); b.close()
    ctx, a, b = _socket_ctx()
    try:  # OG: HELLO 0 then the HELLO 0x13/02 key request -> stop, prompt as before
        b.sendall(frame_packet(b"\x13\x00\x00\x00\x4e\x89") + frame_packet(b"\x13\x02"))
        pending = []
        t0 = time.monotonic()
        assert net._collect_unprompted_login(ctx, pending) is False
        assert time.monotonic() - t0 < 0.3 and [p[:2] for p in pending] == [b"\x13\x00", b"\x13\x02"]
    finally:
        a.close(); b.close()
    ctx, a, b = _socket_ctx()
    try:  # stock client that waits to be asked: window times out, nothing lost
        pending = []
        assert net._collect_unprompted_login(ctx, pending) is False and pending == []
    finally:
        a.close(); b.close()


def test_og_mode_session_key_fallback_when_no_identify():
    os.environ["WULFRAM_HELLO_KEY_FALLBACK_S"] = "0.2"
    os.environ.setdefault("WULFRAM_LOGIN_BOOTSTRAP", "hybrid")
    try:
        net = make_net()
        net.public_addr = "127.0.0.1"
        net.port = 26270
        ctx = make_ctx(phase=Phase.HANDSHAKE)
        ctx.session.phase = Phase.HANDSHAKE
        net.udp_handler.open_channel = lambda c: 50001

        def fake_send(payload, log=True):
            ctx.tcp_handler.sent.append(bytes(payload))
            if payload[:2] == b"\x13\x02":  # client identifies once it has the key
                ctx.session.udp_verified = True
        ctx.tcp_handler.send = fake_send
        net._do_handshake(ctx)
        heads = [p[:2] for p in ctx.tcp_handler.sent]
        assert heads[0] == b"\x13\x01" and b"\x13\x02" in heads
        key_pkt = ctx.tcp_handler.sent[heads.index(b"\x13\x02")]
        key = key_pkt[4:].rstrip(b"\x00").decode()
        assert len(key) == 64 and net.session_key_to_client[key] is ctx
        assert ctx.session.phase == Phase.LOGIN
    finally:
        del os.environ["WULFRAM_HELLO_KEY_FALLBACK_S"]


# ------------------------------------------------------------------ ACTION tick domain


class FakeClock:
    def __init__(self, start=500_000):
        self.now = start

    def __call__(self):
        return self.now


def _au(tick: int) -> bytes:
    return b"\x0a\x00" + struct.pack(">II", tick, 0)


def test_action_tick_sequence_domain_normalised():
    net = make_net()
    ctx = make_ctx(phase=Phase.IN_GAME)
    clock = FakeClock()
    orig = server_net.get_ticks
    server_net.get_ticks = clock
    try:
        out = None
        for seq in range(1, 30):  # native: +1 per packet, ~4 ms apart
            clock.now += 4
            out = net._normalize_action_tick(ctx, _au(seq), 2)
        assert ctx.action_tick_domain == "sequence"
        assert struct.unpack(">I", out[2:6])[0] == clock.now
        assert out[:2] == b"\x0a\x00" and out[6:] == _au(0)[6:]
        # Same arrival millisecond twice -> still strictly increasing.
        a = struct.unpack(">I", net._normalize_action_tick(ctx, _au(31), 2)[2:6])[0]
        b = struct.unpack(">I", net._normalize_action_tick(ctx, _au(32), 2)[2:6])[0]
        assert b == a + 1
        # ACTION_DUMP header (offset 1) shares the domain.
        dump = b"\x09" + struct.pack(">II", 33, 0) + b"\x00" * 14
        assert struct.unpack(">I", net._normalize_action_tick(ctx, dump, 1)[1:5])[0] > b
    finally:
        server_net.get_ticks = orig


def test_action_tick_clock_domain_untouched():
    net = make_net()
    ctx = make_ctx(phase=Phase.IN_GAME)
    clock = FakeClock()
    orig = server_net.get_ticks
    server_net.get_ticks = clock
    try:
        client_ms = 9_000_000
        for _ in range(30):  # OG: header carries its ms clock, ~10-22 ms per packet
            clock.now += 15
            client_ms += 15
            pkt = _au(client_ms)
            assert net._normalize_action_tick(ctx, pkt, 2) == pkt
        assert ctx.action_tick_domain == "clock"
        os.environ["WULFRAM_ACTION_TICK_NORMALIZE"] = "0"
        ctx2 = make_ctx(2, phase=Phase.IN_GAME)
        try:
            assert net._normalize_action_tick(ctx2, _au(1), 2) == _au(1)
            assert getattr(ctx2, "action_tick_domain", "") == ""
        finally:
            del os.environ["WULFRAM_ACTION_TICK_NORMALIZE"]
    finally:
        server_net.get_ticks = orig


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"{t.__name__}: PASSED")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{t.__name__}: FAILED - {exc!r}")
            failed += 1
    print(f"Results: {len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
