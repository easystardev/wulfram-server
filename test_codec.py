#!/usr/bin/env python3
"""
Tests for codec utilities: BitWriter, BitReader, Fixed16.16, framing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from wulfram.codec import (
    BitWriter, BitReader,
    pack_fixed16, unpack_fixed16,
    frame_packet, unframe_packet,
    format_hex, format_ascii,
)


def test_bitwriter_single_byte():
    """BitWriter can write 8 bits as one byte."""
    writer = BitWriter()
    writer.write_bits(8, 0xAB)
    assert writer.get_bytes() == bytes([0xAB])
    print("test_bitwriter_single_byte: PASSED")
    return True


def test_bitwriter_multiple_bytes():
    """BitWriter can write multiple bytes."""
    writer = BitWriter()
    writer.write_bits(8, 0x12)
    writer.write_bits(8, 0x34)
    writer.write_bits(8, 0x56)
    assert writer.get_bytes() == bytes([0x12, 0x34, 0x56])
    print("test_bitwriter_multiple_bytes: PASSED")
    return True


def test_bitwriter_partial_bits():
    """BitWriter handles partial byte writes with padding."""
    writer = BitWriter()
    writer.write_bits(4, 0xA)  # 1010
    result = writer.get_bytes()
    # 1010 padded to 10100000 = 0xA0
    assert result == bytes([0xA0])
    print("test_bitwriter_partial_bits: PASSED")
    return True


def test_bitwriter_cross_byte_boundary():
    """BitWriter correctly handles writes crossing byte boundaries."""
    writer = BitWriter()
    writer.write_bits(4, 0xA)   # 1010
    writer.write_bits(8, 0xBC)  # 10111100
    result = writer.get_bytes()
    # 1010 1011 | 1100 0000 = 0xAB 0xC0
    assert result == bytes([0xAB, 0xC0])
    print("test_bitwriter_cross_byte_boundary: PASSED")
    return True


def test_bitwriter_16bit():
    """BitWriter handles 16-bit values."""
    writer = BitWriter()
    writer.write_bits(16, 0x1234)
    assert writer.get_bytes() == bytes([0x12, 0x34])
    print("test_bitwriter_16bit: PASSED")
    return True


def test_bitwriter_32bit():
    """BitWriter handles 32-bit values."""
    writer = BitWriter()
    writer.write_bits(32, 0x12345678)
    assert writer.get_bytes() == bytes([0x12, 0x34, 0x56, 0x78])
    print("test_bitwriter_32bit: PASSED")
    return True


def test_bitreader_single_byte():
    """BitReader can read 8 bits."""
    reader = BitReader(bytes([0xAB]))
    assert reader.read_bits(8) == 0xAB
    print("test_bitreader_single_byte: PASSED")
    return True


def test_bitreader_multiple_bytes():
    """BitReader reads sequential bytes."""
    reader = BitReader(bytes([0x12, 0x34, 0x56]))
    assert reader.read_u8() == 0x12
    assert reader.read_u8() == 0x34
    assert reader.read_u8() == 0x56
    print("test_bitreader_multiple_bytes: PASSED")
    return True


def test_bitreader_partial_bits():
    """BitReader reads partial bytes."""
    reader = BitReader(bytes([0xAB]))  # 10101011
    assert reader.read_bits(4) == 0xA  # 1010
    assert reader.read_bits(4) == 0xB  # 1011
    print("test_bitreader_partial_bits: PASSED")
    return True


def test_bitreader_cross_boundary():
    """BitReader handles cross-byte reads."""
    reader = BitReader(bytes([0xAB, 0xCD]))  # 10101011 11001101
    reader.read_bits(4)  # 1010
    value = reader.read_bits(8)  # 10111100 = 0xBC
    assert value == 0xBC
    print("test_bitreader_cross_boundary: PASSED")
    return True


def test_bitreader_u16():
    """BitReader u16 helper works."""
    reader = BitReader(bytes([0x12, 0x34]))
    assert reader.read_u16() == 0x1234
    print("test_bitreader_u16: PASSED")
    return True


def test_bitreader_u32():
    """BitReader u32 helper works."""
    reader = BitReader(bytes([0x12, 0x34, 0x56, 0x78]))
    assert reader.read_u32() == 0x12345678
    print("test_bitreader_u32: PASSED")
    return True


def test_bitreader_out_of_data():
    """BitReader raises on reading past end."""
    reader = BitReader(bytes([0xAB]))
    reader.read_bits(8)
    try:
        reader.read_bits(1)
        print("test_bitreader_out_of_data: FAILED - no exception")
        return False
    except ValueError as e:
        assert "out of data" in str(e)
        print("test_bitreader_out_of_data: PASSED")
        return True


def test_bit_roundtrip():
    """Write and read bits should roundtrip."""
    writer = BitWriter()
    writer.write_bits(5, 0x1F)   # 11111
    writer.write_bits(11, 0x7AB) # 11110101011
    data = writer.get_bytes()

    reader = BitReader(data)
    assert reader.read_bits(5) == 0x1F
    assert reader.read_bits(11) == 0x7AB
    print("test_bit_roundtrip: PASSED")
    return True


def test_fixed16_positive():
    """Fixed16.16 encodes positive values."""
    data = pack_fixed16(1.5)
    value = unpack_fixed16(data)
    assert abs(value - 1.5) < 0.0001
    print("test_fixed16_positive: PASSED")
    return True


def test_fixed16_negative():
    """Fixed16.16 encodes negative values."""
    data = pack_fixed16(-2.25)
    value = unpack_fixed16(data)
    assert abs(value - (-2.25)) < 0.0001
    print("test_fixed16_negative: PASSED")
    return True


def test_fixed16_zero():
    """Fixed16.16 encodes zero."""
    data = pack_fixed16(0.0)
    value = unpack_fixed16(data)
    assert value == 0.0
    print("test_fixed16_zero: PASSED")
    return True


def test_fixed16_large():
    """Fixed16.16 clamps large values."""
    # 32768.0 is the max for signed 16.16
    data = pack_fixed16(100000.0)
    value = unpack_fixed16(data)
    assert value <= 32768.0
    print("test_fixed16_large: PASSED")
    return True


def test_fixed16_known_value():
    """Fixed16.16 produces expected byte pattern."""
    # 1.0 should be 0x00010000
    data = pack_fixed16(1.0)
    assert data == bytes([0x00, 0x01, 0x00, 0x00])
    print("test_fixed16_known_value: PASSED")
    return True


def test_frame_packet():
    """frame_packet adds 2-byte length prefix."""
    payload = bytes([0x13, 0x01, 0x02])
    framed = frame_packet(payload)
    # Length = 3 + 2 = 5 = 0x0005
    assert framed == bytes([0x00, 0x05, 0x13, 0x01, 0x02])
    print("test_frame_packet: PASSED")
    return True


def test_unframe_packet_complete():
    """unframe_packet extracts complete packet."""
    data = bytes([0x00, 0x05, 0x13, 0x01, 0x02])
    body, remaining = unframe_packet(data)
    assert body == bytes([0x13, 0x01, 0x02])
    assert remaining == b""
    print("test_unframe_packet_complete: PASSED")
    return True


def test_unframe_packet_incomplete():
    """unframe_packet returns None for incomplete data."""
    data = bytes([0x00, 0x05, 0x13])  # Claims 5 bytes, only 3
    body, remaining = unframe_packet(data)
    assert body is None
    assert remaining == data
    print("test_unframe_packet_incomplete: PASSED")
    return True


def test_unframe_packet_too_short():
    """unframe_packet handles < 2 bytes."""
    data = bytes([0x00])
    body, remaining = unframe_packet(data)
    assert body is None
    assert remaining == data
    print("test_unframe_packet_too_short: PASSED")
    return True


def test_unframe_packet_multiple():
    """unframe_packet handles multiple packets."""
    data = bytes([0x00, 0x04, 0xAA, 0xBB, 0x00, 0x03, 0xCC])
    body1, remaining = unframe_packet(data)
    assert body1 == bytes([0xAA, 0xBB])
    body2, remaining = unframe_packet(remaining)
    assert body2 == bytes([0xCC])
    assert remaining == b""
    print("test_unframe_packet_multiple: PASSED")
    return True


def test_format_hex():
    """format_hex produces uppercase hex."""
    result = format_hex(bytes([0xab, 0xcd, 0xef]))
    assert result == "ABCDEF"
    print("test_format_hex: PASSED")
    return True


def test_format_hex_truncate():
    """format_hex truncates long data."""
    data = bytes(range(100))
    result = format_hex(data, max_len=10)
    assert len(result) == 13  # 10 + "..."
    assert result.endswith("...")
    print("test_format_hex_truncate: PASSED")
    return True


def test_format_ascii():
    """format_ascii shows printable chars."""
    result = format_ascii(b"Hello\x00World")
    assert result == "Hello.World"
    print("test_format_ascii: PASSED")
    return True


def main():
    print("=" * 60)
    print("Codec Tests")
    print("=" * 60)

    tests = [
        # BitWriter
        test_bitwriter_single_byte,
        test_bitwriter_multiple_bytes,
        test_bitwriter_partial_bits,
        test_bitwriter_cross_byte_boundary,
        test_bitwriter_16bit,
        test_bitwriter_32bit,
        # BitReader
        test_bitreader_single_byte,
        test_bitreader_multiple_bytes,
        test_bitreader_partial_bits,
        test_bitreader_cross_boundary,
        test_bitreader_u16,
        test_bitreader_u32,
        test_bitreader_out_of_data,
        test_bit_roundtrip,
        # Fixed16.16
        test_fixed16_positive,
        test_fixed16_negative,
        test_fixed16_zero,
        test_fixed16_large,
        test_fixed16_known_value,
        # Framing
        test_frame_packet,
        test_unframe_packet_complete,
        test_unframe_packet_incomplete,
        test_unframe_packet_too_short,
        test_unframe_packet_multiple,
        # Formatting
        test_format_hex,
        test_format_hex_truncate,
        test_format_ascii,
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
