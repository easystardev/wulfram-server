#!/usr/bin/env python3
"""
Tests for handler functions extracted from server.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram.handlers import decode_lp_string


def test_decode_lp_string_basic():
    """Test decoding a length-prefixed string."""
    # Length = 6, string = "Hello\x00"
    data = b'\x00\x06Hello\x00'
    text, offset = decode_lp_string(data, 0)
    assert text == "Hello"
    assert offset == 8
    print("test_decode_lp_string_basic: PASSED")
    return True


def test_decode_lp_string_offset():
    """Test decoding with non-zero offset."""
    data = b'PREFIX\x00\x05Test\x00'
    text, offset = decode_lp_string(data, 6)
    assert text == "Test"
    assert offset == 13
    print("test_decode_lp_string_offset: PASSED")
    return True


def test_decode_lp_string_empty():
    """Test decoding empty string."""
    data = b'\x00\x01\x00'  # Length 1, just null terminator
    text, offset = decode_lp_string(data, 0)
    assert text == ""
    assert offset == 3
    print("test_decode_lp_string_empty: PASSED")
    return True


def test_decode_lp_string_truncated():
    """Test handling truncated data."""
    data = b'\x00'  # Only 1 byte, need 2 for length
    text, offset = decode_lp_string(data, 0)
    assert text == ""
    assert offset == 0
    print("test_decode_lp_string_truncated: PASSED")
    return True


def test_handlers_import():
    """Test that all handler functions can be imported."""
    from wulfram.handlers import (
        handle_hello,
        handle_login_request,
        send_initial_game_data,
        handle_bps,
        handle_want_updates,
        handle_reincarnate_tcp,
        handle_udp_d_handshake,
        handle_udp_chat,
        handle_udp_reincarnate,
        handle_team_switch,
        handle_spawn_at_point,
        send_udp_ack,
    )
    print("test_handlers_import: PASSED")
    return True


def main():
    print("=" * 60)
    print("Handler Tests")
    print("=" * 60)

    tests = [
        test_decode_lp_string_basic,
        test_decode_lp_string_offset,
        test_decode_lp_string_empty,
        test_decode_lp_string_truncated,
        test_handlers_import,
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
            print(f"  {test.__name__}: FAILED - {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
