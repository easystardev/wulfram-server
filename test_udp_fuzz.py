#!/usr/bin/env python3
"""Malformed-UDP resilience gate for the public game port.

A public UDP socket is scanned and garbage-flooded within minutes of going
live, so a single crafted datagram must never wedge the server. The UDP receive
path is the one shared-thread single-point-of-failure (TCP is per-client
threaded), so this gate covers it at two layers:

  1. parser robustness  -- _parse_udp_datagram must not raise on fuzz input
     (it processes attacker-controlled bytes; it should skip what it can't
     understand, never throw).
  2. loop resilience boundary -- even if the parser DID raise (a future
     regression), _udp_loop must isolate it and keep serving every other
     client, not let the exception kill the shared UDP thread.

Standalone runnable (parity_gate runs it directly):
    uv run python test_udp_fuzz.py
"""
import contextlib
import io
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

# OG-faithful default so construction matches production config.
os.environ.setdefault("WULFRAM_AUTO_LOGIN", "0")

from wulfram.server import WulframServer  # noqa: E402

FUZZ_SEED = 20260622


def _build_server() -> WulframServer:
    """Construct a real, fully-configured server (port=0, no loops started).

    Construction is noisy ([CONFIG]/[TERRAIN]/... banners); swallow it so the
    test output stays readable.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return WulframServer(host="127.0.0.1", port=0)


def _fuzz_datagrams(seed: int):
    """A broad corpus of malformed datagrams: pure random, every opcode with
    short tails, and truncations/garbage-extensions of plausible packet shapes.
    """
    rng = random.Random(seed)
    # pure random datagrams of varied length (incl. empty)
    for _ in range(6000):
        length = rng.randint(0, 600)
        yield bytes(rng.randrange(256) for _ in range(length))
    # every possible first opcode byte with a few tail lengths
    for opcode in range(256):
        for tail in (0, 1, 2, 3, 5, 8, 16, 64):
            yield bytes([opcode]) + bytes(rng.randrange(256) for _ in range(tail))
    # truncations + garbage-extensions of real-ish packet shapes
    shapes = [
        b"\x02\x00" + b"\x00" * 8,          # D_ACK timestamp
        b"\x02\x01\x00\x00\x00",            # D_ACK channel+seq
        b"\x02\x02" + b"\x00" * 7,          # D_ACK ts+ch+seq
        b"\x09" + b"\xff" * 40,             # ACTION_DUMP
        b"\x0a" + b"\xff" * 40,             # ACTION_UPDATE
        b"\x10" + b"\x00" * 30,             # 0x10 wrapper
        b"\x35\x00\x00\x00\x14" + b"\xe4" * 15,  # VIEWPOINT_INFO
        b"\x25" + b"\x00" * 10,             # REINCARNATE
    ]
    for shape in shapes:
        for cut in range(len(shape) + 1):
            yield shape[:cut]
            yield shape[:cut] + bytes(rng.randrange(256) for _ in range(4))


def test_parser_survives_fuzz() -> bool:
    """_parse_udp_datagram must consume any garbage without raising or hanging."""
    server = _build_server()
    total = 0
    raised: list[tuple[str, str]] = []
    for data in _fuzz_datagrams(FUZZ_SEED):
        total += 1
        try:
            list(server._parse_udp_datagram(data, None))
        except BaseException as exc:  # noqa: BLE001 - any escape is a finding
            raised.append((data[:16].hex(), repr(exc)[:140]))
    print(f"  fuzzed {total} malformed datagrams through the parser")
    if raised:
        print(f"  test_parser_survives_fuzz: FAILED - {len(raised)} raised, e.g.:")
        for sample, err in raised[:5]:
            print(f"    {sample}... -> {err}")
        return False
    print("  test_parser_survives_fuzz: PASS (0 raised)")
    return True


class _ScriptedUDP:
    """Fake udp_handler: yields one datagram, then stops the loop."""

    def __init__(self, server: WulframServer, datagram: bytes):
        self._server = server
        self._datagram = datagram
        self._calls = 0

    def recv_from(self):
        self._calls += 1
        if self._calls == 1:
            return self._datagram, ("203.0.113.7", 54321)  # unknown addr -> ctx None
        self._server.running = False
        return None, None


def test_udp_loop_isolates_parse_crash() -> bool:
    """A raising parser must NOT propagate out of _udp_loop (shared-thread SPOF).

    Monkeypatch the parser to throw on the datagram, then run one loop turn and
    assert the loop returns cleanly (the resilience boundary caught it).
    """
    server = _build_server()
    server.running = True
    server.debug_udp_raw = False
    server.udp_addr_to_client = {}
    server.udp_handler = _ScriptedUDP(server, b"\xde\xad\xbe\xef")

    sentinel = {"raised": False}

    def _boom(_data, _ctx=None):
        sentinel["raised"] = True
        raise ValueError("simulated parser crash on malformed datagram")

    server._parse_udp_datagram = _boom  # type: ignore[method-assign]

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            server._udp_loop()
    except BaseException as exc:  # noqa: BLE001
        print(f"  test_udp_loop_isolates_parse_crash: FAILED - escaped: {exc!r}")
        return False

    if not sentinel["raised"]:
        print("  test_udp_loop_isolates_parse_crash: FAILED - parser was never reached")
        return False
    print("  test_udp_loop_isolates_parse_crash: PASS (parser crash isolated, loop survived)")
    return True


def main() -> bool:
    tests = [test_parser_survives_fuzz, test_udp_loop_isolates_parse_crash]
    passed = 0
    failed = 0
    for test in tests:
        print()
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  {test.__name__}: FAILED - {exc}")
            failed += 1
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
