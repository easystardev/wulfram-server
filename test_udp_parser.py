#!/usr/bin/env python3
"""
Regression tests for UDP datagram splitting and packet naming.
"""

import sys
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram.server import WulframServer
from wulfram import handlers
from wulfram.weapons import WeaponSystem, BehaviorSlot
from wulfram2_protocol.packets import get_packet_name


def _make_server_stub() -> WulframServer:
    """Create a minimal server instance for parser-only tests."""
    srv = WulframServer.__new__(WulframServer)
    srv.debug_viewpoint = False
    return srv


def test_parse_d_ack_plus_translation_ack() -> bool:
    """
    0x02 subtype-1 ACK (5 bytes) followed by 0x33 reliable packet.
    Historical captures used this exact packed datagram shape.
    """
    srv = _make_server_stub()
    datagram = bytes.fromhex("0201020001330001000900000001")
    packets = list(srv._parse_udp_datagram(datagram))
    assert len(packets) == 2
    assert packets[0] == bytes.fromhex("0201020001")
    assert packets[1] == bytes.fromhex("330001000900000001")
    print("test_parse_d_ack_plus_translation_ack: PASSED")
    return True


def test_parse_two_d_ack_then_input_feedback() -> bool:
    """Two 0x02 ACK frames followed by INPUT_FEEDBACK must split cleanly."""
    srv = _make_server_stub()
    datagram = bytes.fromhex("020102002a020102002b4000000000")
    packets = list(srv._parse_udp_datagram(datagram))
    assert len(packets) == 3
    assert packets[0] == bytes.fromhex("020102002a")
    assert packets[1] == bytes.fromhex("020102002b")
    assert packets[2] == bytes.fromhex("4000000000")
    print("test_parse_two_d_ack_then_input_feedback: PASSED")
    return True


def test_parse_legacy_ack_frame() -> bool:
    """Legacy 9-byte ACK frame format from _send_udp_ack should remain intact."""
    srv = _make_server_stub()
    datagram = bytes.fromhex("020001000901250001")
    packets = list(srv._parse_udp_datagram(datagram))
    assert len(packets) == 1
    assert packets[0] == datagram
    print("test_parse_legacy_ack_frame: PASSED")
    return True


def test_parse_d_set_start() -> bool:
    """D_SET_START (0x04) should parse as its fixed 4-byte packet."""
    srv = _make_server_stub()
    datagram = bytes.fromhex("04010001")
    packets = list(srv._parse_udp_datagram(datagram))
    assert len(packets) == 1
    assert packets[0] == datagram
    print("test_parse_d_set_start: PASSED")
    return True


def test_parse_empirical_d_handshake() -> bool:
    """Empirical client D_HANDSHAKE with named/private streams should parse cleanly."""
    data = bytes.fromhex(
        "03"
        "005f6504"
        "00000002"
        "000e426561636f6e2053747265616d00"
        "00000002"
        "0000003a0000003b"
        "000d5371756164205468696e677300"
        "00000004"
        "0000004200000046000000490000004a"
        "0000000f"
        "0000001900000001"
        "0000002000000003"
        "0000002500000003"
        "0000002600000001"
        "0000002b00000001"
        "0000002e00000003"
        "0000003300000003"
        "0000003500000001"
        "0000003a00000003"
        "0000003b00000003"
        "0000004200000003"
        "0000004600000003"
        "0000004900000003"
        "0000004a00000003"
        "0000004f00000003"
    )
    parsed = handlers._parse_empirical_client_d_handshake(data)
    assert parsed is not None
    assert parsed["sequence"] == 0x005F6504
    assert parsed["streams"][0] == ("Beacon Stream", (0x3A, 0x3B))
    assert parsed["private_modes"][0] == (0x19, 1)
    assert parsed["private_modes"][-1] == (0x4F, 3)
    print("test_parse_empirical_d_handshake: PASSED")
    return True


def test_build_server_d_handshake() -> bool:
    """Server D_HANDSHAKE should emit the empirical OG mappings."""
    packet = handlers._build_server_d_handshake(None)
    assert packet[0] == 0x03
    _sequence = struct.unpack_from(">I", packet, 1)[0]
    session_id = struct.unpack_from(">I", packet, 5)[0]
    stream_count = struct.unpack_from(">I", packet, 9)[0]
    assert session_id == 1
    assert stream_count == 2
    assert _sequence >= 0
    print("test_build_server_d_handshake: PASSED")
    return True


def test_parse_mixed_input_feedback_plus_action_dump() -> bool:
    """INPUT_FEEDBACK + ACTION_DUMP mixed datagram should split with 23-byte dump."""
    srv = _make_server_stub()
    datagram = bytes.fromhex(
        "4000000000"
        "090006d5b9000000000000000000003ff2800000000000"
    )
    packets = list(srv._parse_udp_datagram(datagram))
    assert len(packets) == 2
    assert packets[0] == bytes.fromhex("4000000000")
    assert packets[1] == bytes.fromhex("090006d5b9000000000000000000003ff2800000000000")
    print("test_parse_mixed_input_feedback_plus_action_dump: PASSED")
    return True


def test_action_dump_slot4_not_control_quantized() -> bool:
    """Slot 4 must not decode as 16-bit analog weapon ID (prevents slot~500 bug)."""
    ws = WeaponSystem()
    pkt = bytes.fromhex("090006d5b9000000000000000000003ff2800000000000")
    assert ws.decode_action_dump(pkt) is True
    assert ws.behavior_slots[BehaviorSlot.WEAPON_SELECT] == 0.0
    assert ws.behavior_slots[BehaviorSlot.FIRE] == 0.0
    print("test_action_dump_slot4_not_control_quantized: PASSED")
    return True


def test_packet_names_cover_previous_unknowns() -> bool:
    """Names should exist for formerly high-volume UNKNOWN opcodes."""
    expected = {
        0x02: "D_ACK",
        0x03: "D_HANDSHAKE",
        0x04: "D_SET_START",
        0x0C: "STATE_REQUEST",
        0x10: "UPDATE_ARRAY_STREAM",
        0x23: "REQUEST_START",
        0x40: "INPUT_FEEDBACK",
        0x60: "DEBUG_SYNC",
    }
    for opcode, name in expected.items():
        assert get_packet_name(opcode) == name
    print("test_packet_names_cover_previous_unknowns: PASSED")
    return True


def main() -> bool:
    tests = [
        test_parse_d_ack_plus_translation_ack,
        test_parse_two_d_ack_then_input_feedback,
        test_parse_legacy_ack_frame,
        test_parse_d_set_start,
        test_parse_empirical_d_handshake,
        test_build_server_d_handshake,
        test_parse_mixed_input_feedback_plus_action_dump,
        test_action_dump_slot4_not_control_quantized,
        test_packet_names_cover_previous_unknowns,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as ex:
            print(f"{test.__name__}: FAILED - {ex}")
            failed += 1

    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
