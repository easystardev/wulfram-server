#!/usr/bin/env python3
"""Replay a captured upstream native-client (WulframClient.exe 2026-10-16) C->S stream
into a real server over loopback, and report what the server did with it.

The capture is a ``messages.pkl`` written by the playtest analysis pipeline
(artifacts/playtest-2026-10-16-analysis/pass2_extract.py): a dict with ``t0`` and
``msgs`` = [{t, proto, dir, port, data}], TCP 2627 entries being one framed message
body each, UDP 2627 entries being one datagram each. That pickle is our own local
analysis artifact (trusted input, so ``pickle.load`` is acceptable here).

SECRETS: the capture carries the live login credential, the upstream session token
and the relay HMAC key in clear. This tool never writes or prints them:

* the TCP LOGIN_REQUEST (0x21) is REBUILT from synthetic values of the same lengths;
* the UDP identify (0x13 01 <token>) is REBUILT with the key OUR server handed out;
* nothing from the relay port (38010) is replayed at all;
* the report only carries opcode/length statistics and server log excerpts.

The replay is open loop, gated only at the handful of points where the real client
must have received something first (UDP port, session key, login accept,
TRANSLATION).  Timing inside the in-game stream follows the capture (``--speed``).

Usage (from server/):
    uv run python tools/native_client_replay.py --pkl <messages.pkl> \
        --port 26270 --duration 120 --report out.json --server-log server.log
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
import pickle
import re
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent

# ---------------------------------------------------------------- capture loading


def load_c2s(pkl_path: str, port: int = 2627) -> list[dict]:
    with open(pkl_path, "rb") as fh:
        blob = pickle.load(fh)  # trusted: our own pass2_extract.py output
    t0 = blob["t0"]
    out = []
    for m in blob["msgs"]:
        if m.get("port") != port or m.get("dir") != "c2s" or m.get("bad_frame"):
            continue
        out.append({"t": float(m["t"]) - t0, "proto": m["proto"], "data": bytes(m["data"])})
    out.sort(key=lambda m: m["t"])
    return out


def synthetic_login(original: bytes) -> bytes:
    """Rebuild 0x21/<sub> handle+credential with synthetic values of equal length."""
    sub = original[1]
    off = 2
    fields = []
    while off + 2 <= len(original):
        n = struct.unpack_from(">H", original, off)[0]
        fields.append(n)
        off += 2 + n
    if off != len(original) or not fields:
        raise ValueError("LOGIN_REQUEST is not a pure sequence of LP strings")
    out = bytearray((0x21, sub))
    for i, n in enumerate(fields):
        body_len = max(0, n - 1)  # LP strings are NUL terminated
        if i == 0:
            text = ("synthpilot" * 8)[:body_len]
        else:
            text = base64.urlsafe_b64encode(secrets.token_bytes(48)).decode()[:body_len]
        raw = text.encode("ascii") + (b"\x00" if n else b"")
        out += struct.pack(">H", len(raw)) + raw
    return bytes(out)


def udp_identify(key: str) -> bytes:
    raw = key.encode("ascii") + b"\x00"
    return b"\x13\x01" + struct.pack(">H", len(raw)) + raw


# ---------------------------------------------------------------- process CPU (Windows)


class _FT(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]


def process_cpu_seconds(proc: subprocess.Popen) -> float | None:
    if os.name != "nt":
        return None
    handle = int(proc._handle)  # type: ignore[attr-defined]
    c, e, k, u = _FT(), _FT(), _FT(), _FT()
    ok = ctypes.windll.kernel32.GetProcessTimes(handle, ctypes.byref(c), ctypes.byref(e),
                                                ctypes.byref(k), ctypes.byref(u))
    if not ok:
        return None
    to_s = lambda ft: ((ft.hi << 32) | ft.lo) / 1e7  # noqa: E731
    return to_s(k) + to_s(u)


# ---------------------------------------------------------------- client side


class ReplayClient:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.tcp = socket.create_connection((host, port), timeout=10)
        self.tcp.settimeout(0.2)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind((host, 0))
        self.udp.settimeout(0.2)
        self.udp_port: int | None = None
        self.key: str | None = None
        self.key_time: float | None = None
        self.login_status: list[int] = []
        self.s2c_tcp = Counter()
        self.s2c_udp = Counter()
        self.s2c_udp_bytes = 0
        self.ping_sent: dict[bytes, float] = {}
        self.ping_rtt_ms: list[float] = []
        self.events: list[tuple[float, str]] = []
        self.got_translation = threading.Event()
        self.got_login_ok = threading.Event()
        self.got_udp_port = threading.Event()
        self.got_key = threading.Event()
        self.got_server_handshake = threading.Event()
        self.closed = threading.Event()
        self.running = True
        self.t_start = time.monotonic()
        threading.Thread(target=self._tcp_reader, daemon=True).start()
        threading.Thread(target=self._udp_reader, daemon=True).start()

    def now(self) -> float:
        return time.monotonic() - self.t_start

    def _tcp_reader(self):
        buf = b""
        while self.running:
            try:
                chunk = self.tcp.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                self.events.append((self.now(), "tcp closed by server"))
                self.closed.set()
                break
            buf += chunk
            while len(buf) >= 2:
                n = struct.unpack_from(">H", buf, 0)[0]
                if n < 2 or len(buf) < n:
                    break
                body, buf = buf[2:n], buf[n:]
                self._on_tcp(body)

    def _on_tcp(self, body: bytes):
        if not body:
            return
        op = body[0]
        self.s2c_tcp[op] += 1
        if op == 0x13 and len(body) >= 2:
            if body[1] == 0x01 and len(body) >= 4:
                # HELLO 1: UDP config: 13 01 <u16 port> <u16 count> <u16 len> <ip>.
                self.udp_port = struct.unpack_from(">H", body, 2)[0]
                self.events.append((self.now(), f"HELLO1 udp_port={self.udp_port}"))
                self.got_udp_port.set()
            elif body[1] == 0x02 and len(body) >= 4:
                n = struct.unpack_from(">H", body, 2)[0]
                self.key = body[4:4 + n].rstrip(b"\x00").decode("ascii", "ignore")
                self.key_time = self.now()
                self.events.append((self.now(), f"HELLO2 key len={len(self.key)}"))
                self.got_key.set()
        elif op == 0x22 and len(body) >= 3:
            self.login_status.append(body[2])
            self.events.append((self.now(), f"LOGIN_STATUS {body[1]:02x} {body[2]}"))
            if body[2] == 8:
                self.got_login_ok.set()
        elif op == 0x32:
            self.got_translation.set()

    def _udp_reader(self):
        while self.running:
            try:
                data, _ = self.udp.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                if not self.running:
                    break
                continue
            if not data:
                continue
            self.s2c_udp[data[0]] += 1
            self.s2c_udp_bytes += len(data)
            if data[0] == 0x03:
                self.got_server_handshake.set()
            elif data[0] == 0x0C and len(data) >= 5:
                sent = self.ping_sent.pop(data[1:5], None)
                if sent is not None:
                    self.ping_rtt_ms.append((time.perf_counter() - sent) * 1000.0)

    def send_tcp(self, body: bytes):
        self.tcp.sendall(struct.pack(">H", len(body) + 2) + body)

    def send_udp(self, data: bytes):
        if data[:1] == b"\x0b" and len(data) >= 5:
            self.ping_sent[data[1:5]] = time.perf_counter()
        self.udp.sendto(data, (self.host, self.udp_port))

    def close(self):
        self.running = False
        for s in (self.tcp, self.udp):
            try:
                s.close()
            except OSError:
                pass


def precise_sleep_until(deadline: float):
    while True:
        rem = deadline - time.monotonic()
        if rem <= 0:
            return
        time.sleep(rem - 0.0015 if rem > 0.003 else 0)


def replay(args, msgs, client: ReplayClient, cpu_probe) -> dict:
    stats = {"sent_tcp": Counter(), "sent_udp": Counter(), "skipped": Counter(),
             "gates": {}}
    # Gate 1: the native client sends nothing before HELLO 1 + HELLO 2.
    if not client.got_udp_port.wait(10):
        stats["gates"]["udp_port"] = "TIMEOUT"
        return stats
    if not client.got_key.wait(args.key_wait):
        stats["gates"]["session_key"] = f"TIMEOUT after {args.key_wait}s (native client has no key)"
        if not args.identify_without_key:
            return stats
    stats["gates"]["session_key_at_s"] = client.key_time

    first_game_t = None
    base_t = None  # capture time aligned to wall time
    base_wall = None
    sent_udp_bytes = 0
    end_t = None
    stream_start_cpu = None
    stream_start_wall = None
    for m in msgs:
        data = m["data"]
        op = data[0]
        if m["proto"] == "udp" and op in (0x09, 0x0A) and first_game_t is None:
            first_game_t = m["t"]
            end_t = first_game_t + args.duration
            stream_start_cpu = cpu_probe()
            stream_start_wall = time.monotonic()
            stats["gates"]["stream_start_at_s"] = round(client.now(), 3)
        if end_t is not None and m["t"] > end_t:
            break
        # ---- gating points (the real client waited for the server here) ----
        if m["proto"] == "udp" and op == 0x03:
            client.got_login_ok.wait(5)
            stats["gates"]["login_ok_before_d_handshake"] = client.got_login_ok.is_set()
            base_t = None
        if m["proto"] == "udp" and op == 0x02:
            client.got_server_handshake.wait(3)
            stats["gates"]["server_d_handshake"] = client.got_server_handshake.is_set()
            base_t = None
        if m["proto"] == "udp" and op == 0x33:
            client.got_translation.wait(5)
            stats["gates"]["translation_before_ack"] = client.got_translation.is_set()
            base_t = None
        # ---- timing ----
        if base_t is None:
            base_t, base_wall = m["t"], time.monotonic()
        else:
            precise_sleep_until(base_wall + (m["t"] - base_t) / args.speed)
        if client.closed.is_set():
            stats["gates"]["server_closed_tcp_at_s"] = round(client.now(), 3)
            break
        # ---- substitutions / filters ----
        if m["proto"] == "tcp":
            if op == 0x21:
                data = synthetic_login(data)
            client.send_tcp(data)
            stats["sent_tcp"][op] += 1
            continue
        if op == 0x13:
            if client.key is None:
                stats["skipped"]["udp_identify_no_key"] += 1
                continue
            data = udp_identify(client.key)
        if args.drop_0x10 and op == 0x10:
            stats["skipped"][0x10] += 1
            continue
        if args.thin_action and op == 0x0A:
            stats.setdefault("_thin", 0)
            stats["_thin"] += 1
            if stats["_thin"] % args.thin_action:
                stats["skipped"][0x0A] += 1
                continue
        client.send_udp(data)
        stats["sent_udp"][op] += 1
        sent_udp_bytes += len(data)
    if stream_start_cpu is not None:
        wall = time.monotonic() - stream_start_wall
        cpu = cpu_probe() - stream_start_cpu if cpu_probe() is not None else None
        stats["stream_wall_s"] = round(wall, 3)
        stats["stream_server_cpu_s"] = None if cpu is None else round(cpu, 3)
        stats["stream_server_cpu_per_s"] = None if cpu is None else round(cpu / wall, 4)
    stats["sent_udp_bytes"] = sent_udp_bytes
    return stats


# ---------------------------------------------------------------- server log triage

LOG_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "handler_error": re.compile(r"dropped packet .*handler error|dropped malformed datagram"),
    "udp_unknown_packet": re.compile(r"^\[UDP\] Packet 0x[0-9A-F]{2} from"),
    "d_handshake_kind": re.compile(r"D_HANDSHAKE\[(\w+)\]"),
    "d_handshake_malformed": re.compile(r"D_HANDSHAKE malformed"),
    "session_key_no_match": re.compile(r"Session key from .*NO MATCH"),
    "udp_verification_timeout": re.compile(r"UDP verification timeout"),
    "login_ignored_packet": re.compile(r"\[LOGIN\] Ignoring packet"),
    "login_unhandled_subtype": re.compile(r"Unhandled sub_type"),
    "login_success": re.compile(r"Login successful"),
    "action_decode_failed": re.compile(r"decode failed|Error decoding ACTION"),
    "client_error": re.compile(r"\[SERVER\] Client \d+ error"),
    "tick_resync": re.compile(r"\[TICK\] Sync tick offset"),
    "action_tick_domain": re.compile(r"\[TICK\] action tick domain"),
    "hello_key_fallback": re.compile(r"session key fallback"),
    "unprompted_login": re.compile(r"unprompted LOGIN_REQUEST"),
}


def triage_log(text: str) -> dict:
    out = {}
    lines = text.splitlines()
    for name, rx in LOG_PATTERNS.items():
        hits = [ln for ln in lines if rx.search(ln)]
        out[name] = {"count": len(hits), "examples": hits[:4]}
    kinds = Counter(m.group(1) for ln in lines for m in [LOG_PATTERNS["d_handshake_kind"].search(ln)] if m)
    out["d_handshake_kind"]["by_kind"] = dict(kinds)
    unk = Counter(ln.split()[2] for ln in out["udp_unknown_packet"]["examples"])
    out["udp_unknown_packet"]["by_type"] = dict(unk)
    return out


def control_query(port: int, command: str, timeout: float = 3.0) -> str:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
            s.sendall((command + "\n").encode())
            s.settimeout(timeout)
            chunks = []
            while True:
                try:
                    c = s.recv(65536)
                except socket.timeout:
                    break
                if not c:
                    break
                chunks.append(c)
            return b"".join(chunks).decode("utf-8", "replace")
    except OSError as exc:
        return f"<control query failed: {exc}>"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--port", type=int, default=26270)
    ap.add_argument("--duration", type=float, default=90.0,
                    help="seconds of in-game capture to replay after the first ACTION packet")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--key-wait", type=float, default=6.0)
    ap.add_argument("--identify-without-key", action="store_true")
    ap.add_argument("--drop-0x10", action="store_true")
    ap.add_argument("--thin-action", type=int, default=0,
                    help="send only every Nth ACTION_UPDATE (CPU comparison)")
    ap.add_argument("--idle-measure", type=float, default=5.0,
                    help="seconds of idle server CPU to measure before connecting")
    ap.add_argument("--no-server", action="store_true", help="server already running")
    ap.add_argument("--server-log", default="native_replay_server.log")
    ap.add_argument("--report", default="")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the server")
    args = ap.parse_args(argv)

    msgs = load_c2s(args.pkl)
    proc = None
    log_fh = None
    if not args.no_server:
        env = os.environ.copy()
        env.update({
            "WULFRAM_PORT": str(args.port),
            "WULFRAM_BIND_ADDR": "127.0.0.1",
            "WULFRAM_PUBLIC_ADDR": "127.0.0.1",
            "WULFRAM_PACKET_TRACE": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        })
        for kv in args.env:
            k, _, v = kv.partition("=")
            env[k] = v
        log_fh = open(args.server_log, "w", encoding="utf-8", errors="replace")
        run_dir = Path(args.server_log).resolve().parent
        proc = subprocess.Popen([sys.executable, str(SERVER_DIR / "run_server.py")],
                                cwd=str(run_dir), env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 120
        # Wait for the accept loop's banner (a probe connect would itself be a client).
        while time.monotonic() < deadline:
            text = Path(args.server_log).read_text(encoding="utf-8", errors="replace")
            if "[SERVER] Listening on" in text:
                break
            if proc.poll() is not None:
                print("server exited during startup", file=sys.stderr)
                return 2
            time.sleep(0.25)
        else:
            proc.kill()
            print("server did not start", file=sys.stderr)
            return 2

    cpu_probe = (lambda: process_cpu_seconds(proc)) if proc else (lambda: None)
    report: dict = {"capture_c2s_messages": len(msgs), "args": {
        k: v for k, v in vars(args).items() if k not in ("pkl",)}}
    try:
        time.sleep(1.0)  # let the accept loop settle
        c0, w0 = cpu_probe(), time.monotonic()
        time.sleep(args.idle_measure)
        if c0 is not None:
            report["idle_server_cpu_per_s"] = round((cpu_probe() - c0) / (time.monotonic() - w0), 4)
        client = ReplayClient("127.0.0.1", args.port)
        try:
            stats = replay(args, msgs, client, cpu_probe)
            time.sleep(1.0)
            report["replay"] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
            report["replay"]["sent_tcp"] = {f"0x{k:02X}": v for k, v in stats["sent_tcp"].items()}
            report["replay"]["sent_udp"] = {f"0x{k:02X}": v for k, v in stats["sent_udp"].items()}
            report["replay"]["skipped"] = {str(k): v for k, v in stats["skipped"].items()}
            report["s2c_tcp"] = {f"0x{k:02X}": v for k, v in sorted(client.s2c_tcp.items())}
            report["s2c_udp"] = {f"0x{k:02X}": v for k, v in sorted(client.s2c_udp.items())}
            report["s2c_udp_bytes"] = client.s2c_udp_bytes
            rtt = sorted(client.ping_rtt_ms)
            report["ping"] = {
                "sent": len(rtt) + len(client.ping_sent), "replied": len(rtt),
                "rtt_ms_median": round(rtt[len(rtt) // 2], 2) if rtt else None,
                "rtt_ms_max": round(rtt[-1], 2) if rtt else None,
            }
            report["login_status_seq"] = client.login_status
            report["client_events"] = [f"{t:.3f} {e}" for t, e in client.events[:40]]
            if proc is not None:
                report["control_clients"] = control_query(args.port + 1, "players")[:6000]
        finally:
            client.close()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log_fh:
            log_fh.close()
    if proc is not None:
        text = Path(args.server_log).read_text(encoding="utf-8", errors="replace")
        report["server_log_triage"] = triage_log(text)
    out = json.dumps(report, indent=2, default=str)
    if args.report:
        Path(args.report).write_text(out, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
