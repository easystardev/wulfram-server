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
    build_supply_ship_info,
    build_update_array_create_tank,
    _compress_position,
)
from wulfram.codec import frame_packet, BitReader


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
    payload = build_uplink_info(1, 0x14EA, 3)

    print(f"UPLINK_INFO:")
    print(f"  Got: {payload.hex().upper()}")
    expected = bytes.fromhex("2A01000014EA00000003")
    match = payload == expected
    print(f"  Match: {match}")
    return match


def test_supply_ship_info():
    """Test SUPPLY_SHIP_INFO packet."""
    payload = build_supply_ship_info(
        0x12345678,
        shield_pct=100,
        status_template=0,
        cargo_slots=[40, 27, 26, 30],
        cargo_times=[0, 3000, 6000, 9000],
        build_mode=3,
    )

    print(f"SUPPLY_SHIP_INFO:")
    print(f"  Got: {payload.hex().upper()}")
    expected = bytes.fromhex(
        "2D"
        "12345678"
        "00000064"
        "00000000"
        "00000028" "00000000"
        "0000001B" "00000BB8"
        "0000001A" "00001770"
        "0000001E" "00002328"
        "03"
    )
    match = payload == expected
    print(f"  Match: {match}")
    return match


def test_cargo_box_create_definition():
    """CARGO_BOX (type 0x13) create must splice the type-specific DEFINITION field
    (16-bit contained building type, network quantizer entry 4) BETWEEN team and the
    static bit, per the OG parser (Replication.c:1002 -> entity+0xDC). Omitting it
    desyncs the bitstream and the OG client drops with PROTOCOL ERROR "got ILLEGAL".

    This mirrors the OG read sequence and asserts the position vector still decodes,
    proving no downstream desync -- the lenient headless decoder did NOT catch this.
    """
    entity_id = 0x0000ABCD
    team = 1
    contained = 0  # Power Cell
    pos = (1234.0, 5678.0, 42.0)
    # include_health=False -> local-state prefix is exactly 1 bit (deterministic decode).
    pkt = build_update_array_create_tank(
        tick=7, entity_id=entity_id, entity_type=0x13, team=team, pos=pos,
        include_health=False, is_manned=False, is_static=True,
        cargo_contained_type=contained,
    )

    # Skip opcode (1 byte) + tick (4 bytes); bit-decode the rest mirroring the OG parser.
    br = BitReader(pkt[5:])
    ok = True
    ok &= br.read_bits(1) == 0            # local-state flag (no stats)
    ok &= br.read_bits(8) == 1            # entity count
    ok &= br.read_bits(32) == entity_id   # OID
    br.read_bits(1)                       # is_manned
    br.read_bits(10)                      # presence flags
    br.read_bits(16)                      # quantizer index field
    etype = br.read_bits(8)
    ok &= etype == 0x13                   # entity type
    br.read_bits(8)                       # config/parent
    ok &= br.read_bits(8) == team         # team
    ok &= br.read_bits(16) == contained   # *** the type-0x13 DEFINITION field ***
    br.read_bits(1)                       # static bit
    ok &= br.read_bits(4) == 15           # position bank selector
    for coord in pos:                     # alignment proof: position survives intact
        _, q = _compress_position(coord)
        ok &= br.read_bits(16) == q

    # A non-0x13 create of otherwise-identical shape must be exactly 16 bits (2 bytes)
    # shorter -- the field is present ONLY for 0x13.
    pkt_plain = build_update_array_create_tank(
        tick=7, entity_id=entity_id, entity_type=0x19, team=team, pos=pos,
        include_health=False, is_manned=False, is_static=True,
    )
    ok &= (len(pkt) - len(pkt_plain)) == 2

    print(f"CARGO_BOX create DEFINITION field: contained={contained} "
          f"len_delta={len(pkt) - len(pkt_plain)}B  Match: {bool(ok)}")
    return bool(ok)


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
        test_supply_ship_info,
        test_cargo_box_create_definition,
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
