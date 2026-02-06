"""
Codec layer: BitWriter/BitReader and packet encoding utilities.
Pure functions - no I/O, no state.
"""

import struct
from typing import Tuple, Optional


class BitWriter:
    """Writes bits into a byte buffer, MSB first (matches wulf-forge)."""

    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bit_index = 0  # 0-7, next bit position

    def write_bits(self, num_bits: int, value: int):
        """Write `num_bits` bits of `value` to the buffer (MSB first).

        Matches wulf-forge's PacketWriter.write_bits exactly.
        """
        # Iterate from most significant bit down to 0
        for i in range(num_bits - 1, -1, -1):
            bit = (value >> i) & 1
            # Place bit at correct position (7 down to 0)
            self.current_byte |= bit << (7 - self.bit_index)
            self.bit_index += 1
            # If byte is full, push to buffer and reset
            if self.bit_index == 8:
                self.buffer.append(self.current_byte)
                self.current_byte = 0
                self.bit_index = 0

    def get_bytes(self) -> bytes:
        """Flush and return the buffer as bytes."""
        if self.bit_index > 0:
            # Push partial byte (already positioned correctly)
            self.buffer.append(self.current_byte)
        return bytes(self.buffer)


class BitReader:
    """Reads bits from a byte buffer, MSB first."""

    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0

    def read_bits(self, num_bits: int) -> int:
        """Read `num_bits` bits from the buffer (MSB first)."""
        value = 0
        for _ in range(num_bits):
            if self.byte_pos >= len(self.data):
                raise ValueError("BitReader: out of data")
            byte = self.data[self.byte_pos]
            bit = (byte >> (7 - self.bit_pos)) & 1
            value = (value << 1) | bit
            self.bit_pos += 1
            if self.bit_pos == 8:
                self.bit_pos = 0
                self.byte_pos += 1
        return value

    def read_u8(self) -> int:
        return self.read_bits(8)

    def read_u16(self) -> int:
        return self.read_bits(16)

    def read_u32(self) -> int:
        return self.read_bits(32)


# Fixed-point 16.16 format utilities
def pack_fixed16(value: float) -> bytes:
    """Pack a float as Fixed16.16 (big-endian)."""
    fixed = int(round(value * 65536.0))
    # Clamp to signed 32-bit range
    fixed = max(-2147483648, min(2147483647, fixed))
    return struct.pack(">i", fixed)


def unpack_fixed16(data: bytes) -> float:
    """Unpack Fixed16.16 to float."""
    fixed = struct.unpack(">i", data)[0]
    return fixed / 65536.0


# Packet framing utilities
def frame_packet(payload: bytes) -> bytes:
    """Add 2-byte length prefix to packet."""
    length = len(payload) + 2  # Include length bytes themselves
    return struct.pack(">H", length) + payload


def unframe_packet(data: bytes) -> Tuple[Optional[bytes], bytes]:
    """
    Extract one framed packet from data.
    Returns (packet_body, remaining_data) or (None, data) if incomplete.
    """
    if len(data) < 2:
        return None, data
    length = struct.unpack(">H", data[:2])[0]
    if len(data) < length:
        return None, data
    return data[2:length], data[length:]


def format_hex(data: bytes, max_len: int = 100) -> str:
    """Format bytes as hex string for logging."""
    hex_str = data.hex().upper()
    if len(hex_str) > max_len:
        return hex_str[:max_len] + "..."
    return hex_str


def format_ascii(data: bytes, max_len: int = 60) -> str:
    """Format bytes as printable ASCII for logging."""
    result = ""
    for b in data[:max_len]:
        if 32 <= b < 127:
            result += chr(b)
        else:
            result += "."
    if len(data) > max_len:
        result += "..."
    return result
