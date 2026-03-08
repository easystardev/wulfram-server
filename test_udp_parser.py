#!/usr/bin/env python3
"""
Regression tests for UDP datagram splitting and packet naming.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram.server import WulframServer
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
