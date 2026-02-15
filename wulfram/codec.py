"""
Codec layer: re-exports from shared wulfram2_protocol.codec.

All codec primitives (BitWriter, BitReader, fixed16, framing, formatting,
quantization) now live in the shared protocol module. This file re-exports
them so existing server imports continue to work.
"""

from wulfram2_protocol.codec import (  # noqa: F401
    BitWriter,
    BitReader,
    pack_fixed16,
    unpack_fixed16,
    frame_packet,
    unframe_packet,
    format_hex,
    format_ascii,
    quantize_float,
    dequantize_float,
)
