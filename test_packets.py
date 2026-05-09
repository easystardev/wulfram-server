#!/usr/bin/env python3
"""
Simple test to validate packet encoding matches expected format.
"""

import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "shared"))

from wulfram.packets import (
    build_hello_udp_config,
    build_hello_version,
    build_hello_session_key,
    build_identified_udp,
    build_login_status,
    build_player,
    build_team_info,
    build_world_stats,
    build_bps_response,
    build_chat_message,
    build_ship_status,
    build_carrying_info,
    build_uplink_info,
)
from wulfram.codec import frame_packet


def test_hello_udp_config():
    """Test HELLO UDP config packet."""
    payload = build_hello_udp_config("127.0.0.1", 2627)
    framed = frame_packet(payload)

    # Expected from working listen.py:
    # [SEND] HELLO (TCP)  (0x13) | Len=20  | Body=001413010A430001000A3132372E302E302E3100
    expected = bytes.fromhex("001413010A430001000A3132372E302E302E3100")

    print(f"HELLO_UDP_CONFIG:")
    print(f"  Got:      {framed.hex().upper()}")
    print(f"  Expected: {expected.hex().upper()}")
    print(f"  Match: {framed == expected}")
    return framed == expected


def test_identified_udp():
    """Test IDENTIFIED_UDP packet."""
    payload = build_identified_udp()
    framed = frame_packet(payload)

    # Expected: 00034D
    expected = bytes.fromhex("00034D")

    print(f"IDENTIFIED_UDP:")
    print(f"  Got:      {framed.hex().upper()}")
    print(f"  Expected: {expected.hex().upper()}")
    print(f"  Match: {framed == expected}")
    return framed == expected


def test_login_status_request_handle():
    """Test LOGIN_STATUS requesting handle (code 5)."""
    payload = build_login_status(5)
    framed = frame_packet(payload)

    # Non-success prompts do not set the donor byte.
    expected = bytes.fromhex("0005220005")

    print(f"LOGIN_STATUS (code 5):")
    print(f"  Got:      {framed.hex().upper()}")
    print(f"  Expected: {expected.hex().upper()}")
    print(f"  Match: {framed == expected}")
    return framed == expected


def test_login_status_request_password():
    """Test LOGIN_STATUS requesting password (code 1)."""
    payload = build_login_status(1)
    framed = frame_packet(payload)

    # Non-success prompts do not set the donor byte.
    expected = bytes.fromhex("0005220001")

    print(f"LOGIN_STATUS (code 1):")
    print(f"  Got:      {framed.hex().upper()}")
    print(f"  Expected: {expected.hex().upper()}")
    print(f"  Match: {framed == expected}")
    return framed == expected


def test_login_status_success():
    """Test LOGIN_STATUS success (code 8)."""
    payload = build_login_status(8, is_donor=True)
    framed = frame_packet(payload)

    # Expected: 0005220108
    expected = bytes.fromhex("0005220108")

    print(f"LOGIN_STATUS (code 8):")
    print(f"  Got:      {framed.hex().upper()}")
    print(f"  Expected: {expected.hex().upper()}")
    print(f"  Match: {framed == expected}")
    return framed == expected


def test_hello_version():
    """Test HELLO version packet."""
    payload = build_hello_version(0x4E89)
    framed = frame_packet(payload)

    # Expected: 0008130000004E89
    expected = bytes.fromhex("0008130000004E89")

    print(f"HELLO_VERSION:")
    print(f"  Got:      {framed.hex().upper()}")
    print(f"  Expected: {expected.hex().upper()}")
    print(f"  Match: {framed == expected}")
    return framed == expected


def test_hello_session_key():
    """Test HELLO session key packet."""
    payload = build_hello_session_key("WulframSessionKey123")
    framed = frame_packet(payload)

    # Expected: 001B1302001557756C6672616D53657373696F6E4B657931323300
    expected = bytes.fromhex("001B1302001557756C6672616D53657373696F6E4B657931323300")

    print(f"HELLO_SESSION_KEY:")
    print(f"  Got:      {framed.hex().upper()}")
    print(f"  Expected: {expected.hex().upper()}")
    print(f"  Match: {framed == expected}")
    return framed == expected


def test_player():
    """Test PLAYER packet."""
    payload = build_player(entity_id=0, spectator=False)
    framed = frame_packet(payload)

    # Expected: 0008170000000000
    expected = bytes.fromhex("0008170000000000")

    print(f"PLAYER (entity=0):")
    print(f"  Got:      {framed.hex().upper()}")
    print(f"  Expected: {expected.hex().upper()}")
    print(f"  Match: {framed == expected}")
    return framed == expected


def test_bps_response():
    """Test BPS response packet."""
    payload = build_bps_response(1, approved=True)
    framed = frame_packet(payload)

    # Expected: 00084E0000000101
    expected = bytes.fromhex("00084E0000000101")

    print(f"BPS_RESPONSE:")
    print(f"  Got:      {framed.hex().upper()}")
    print(f"  Expected: {expected.hex().upper()}")
    print(f"  Match: {framed == expected}")
    return framed == expected


def test_team_info():
    """Test TEAM_INFO packet."""
    from wulfram.packets import build_team_info
    payload = build_team_info()
    framed = frame_packet(payload)

    # Keep this aligned with the current empirical string set rather than a
    # stale historical byte count.
    has_crimson = b"Crimson_Federation\x00" in payload
    has_azure = b"Azure_Alliance\x00" in payload

    print(f"TEAM_INFO:")
    print(f"  Got length:      {len(framed)}")
    print(f"  Has Crimson:     {has_crimson}")
    print(f"  Has Azure:       {has_azure}")
    match = payload[0] == 0x28 and has_crimson and has_azure
    print(f"  Match: {match}")
    return match


def test_tank_packet():
    """Test TANK packet structure."""
    from wulfram.packets import build_tank_packet
    payload = build_tank_packet(
        net_id=1337,
        unit_type=0,
        pos=(100.0, 50.0, 100.0),
        flags=1,
        include_vitals=True,
        health=1.0,
        energy=1.0
    )

    print(f"TANK_PACKET:")
    print(f"  Length: {len(payload)}")
    print(f"  Opcode: 0x{payload[0]:02X}")
    # Should start with 0x18 opcode
    match = payload[0] == 0x18 and len(payload) >= 30
    print(f"  Match: {match}")
    return match


def test_udp_tank_packet_wf():
    """Test UDP TANK packet (Wulf-Forge style)."""
    from wulfram.packets import build_udp_tank_packet_wf
    payload = build_udp_tank_packet_wf(
        net_id=1337,
        unit_type=0,
        team_id=1,
        pos=(100.0, 100.0, 100.0),
        tick=1000,
        include_vitals=True,
    )

    print(f"UDP_TANK_PACKET_WF:")
    print(f"  Length: {len(payload)}")
    print(f"  Opcode: 0x{payload[0]:02X}")
    # Should start with 0x18 opcode
    match = payload[0] == 0x18 and len(payload) >= 30
    print(f"  Match: {match}")
    return match


def test_update_array_empty():
    """Test empty UPDATE_ARRAY packet."""
    from wulfram.packets import build_update_array_empty
    payload = build_update_array_empty(tick=100)

    print(f"UPDATE_ARRAY_EMPTY:")
    print(f"  Length: {len(payload)}")
    print(f"  Opcode: 0x{payload[0]:02X}")
    # 0x0E + 4 bytes tick + 2 bits (local_state=0, count=0) padded
    match = payload[0] == 0x0E and len(payload) >= 6
    print(f"  Match: {match}")
    return match


def test_update_array_heartbeat():
    """Test UPDATE_ARRAY heartbeat packet."""
    from wulfram.packets import build_update_array_heartbeat
    payload = build_update_array_heartbeat(tick=100, entity_id=1337, include_health=False)

    print(f"UPDATE_ARRAY_HEARTBEAT:")
    print(f"  Length: {len(payload)}")
    print(f"  Opcode: 0x{payload[0]:02X}")
    match = payload[0] == 0x0E and len(payload) >= 10
    print(f"  Match: {match}")
    return match


def test_behavior_packet():
    """Test BEHAVIOR packet structure and parser round-trip."""
    from wulfram.packets import build_behavior_packet
    from client.wulfram_client.network.behavior import parse_behavior
    payload = build_behavior_packet()
    parsed = parse_behavior(payload)

    print(f"BEHAVIOR_PACKET:")
    print(f"  Length: {len(payload)}")
    print(f"  Opcode: 0x{payload[0]:02X}")
    spring_counts = [len(s.points) for s in parsed.spring_states]
    match = (
        payload[0] == 0x24
        and len(payload) >= 3116
        and len(parsed.weapon_units) == 4
        and spring_counts == [4, 4, 4, 4]
    )
    print(f"  Parsed weapon units: {len(parsed.weapon_units)}")
    print(f"  Spring states: {spring_counts}")
    print(f"  Match: {match}")
    return match


def test_translation_packet():
    """Test TRANSLATION packet structure."""
    from wulfram.packets import build_translation_packet
    payload = build_translation_packet()

    print(f"TRANSLATION_PACKET:")
    print(f"  Length: {len(payload)}")
    print(f"  Opcode: 0x{payload[0]:02X}")
    # 28 quantizers, each with header + strings
    match = payload[0] == 0x32 and len(payload) > 500
    print(f"  Match: {match}")
    return match


def test_chat_message():
    """Test COMM_MESSAGE packet."""
    from wulfram.packets import build_chat_message
    payload = build_chat_message("Hello World!", source_id=1337)

    print(f"CHAT_MESSAGE:")
    print(f"  Length: {len(payload)}")
    print(f"  Opcode: 0x{payload[0]:02X}")
    # 0x1F + headers + message
    match = payload[0] == 0x1F and len(payload) > 15
    print(f"  Match: {match}")
    return match


def test_ship_status():
    """Test SHIP_STATUS packet."""
    payload = build_ship_status(0x12345678, 2, "Supply Ship")

    print(f"SHIP_STATUS:")
    print(f"  Got: {payload.hex().upper()}")
    match = (
        payload[:7] == bytes.fromhex("27123456780002")
        and payload[7:9] == b"\x00\x0c"
        and payload[9:] == b"Supply Ship\x00"
    )
    print(f"  Match: {match}")
    return match


def test_carrying_info():
    """Test CARRYING_INFO packet."""
    payload = build_carrying_info(0x14EA, cargo_type=3, has_uplink=True, cargo_count=4)

    print(f"CARRYING_INFO:")
    print(f"  Got: {payload.hex().upper()}")
    expected = bytes.fromhex("29000014EA030104")
    match = payload == expected
    print(f"  Match: {match}")
    return match


def test_uplink_info():
    """Test UPLINK_INFO packet."""
    payload = build_uplink_info(1, 3, 0x14EA)

    print(f"UPLINK_INFO:")
    print(f"  Got: {payload.hex().upper()}")
    expected = bytes.fromhex("2A0100000003000014EA")
    match = payload == expected
    print(f"  Match: {match}")
    return match


def test_world_stats():
    """Test WORLD_STATS packet."""
    payload = build_world_stats()
    framed = frame_packet(payload)

    print(f"WORLD_STATS:")
    print(f"  Length: {len(framed)}")
    print(f"  Opcode: 0x{payload[0]:02X}")
    # 0x16 + map name + grid + scale
    match = payload[0] == 0x16 and len(payload) > 10
    print(f"  Match: {match}")
    return match


def main():
    print("=" * 60)
    print("Packet Encoding Tests")
    print("=" * 60)

    tests = [
        test_hello_udp_config,
        test_identified_udp,
        test_login_status_request_handle,
        test_login_status_request_password,
        test_login_status_success,
        test_hello_version,
        test_hello_session_key,
        test_player,
        test_bps_response,
        test_team_info,
        test_tank_packet,
        test_udp_tank_packet_wf,
        test_update_array_empty,
        test_update_array_heartbeat,
        test_behavior_packet,
        test_translation_packet,
        test_chat_message,
        test_ship_status,
        test_carrying_info,
        test_uplink_info,
        test_world_stats,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print()
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
