"""Native-client (WulframClient.exe --no-mods) session predicates and UDP framing rules.

Tag: community/native.  Everything here is consulted only behind the default-off
knobs parsed by ``server_config.init_native_client_profile``; with every knob off
no caller changes a byte on the wire.

Evidence: D:\\work\\upstream-1016\\ghidra\\RESULT.md (UDP demux ``FUN_140148880``,
readiness flags in ``FUN_1401499b0``/``FUN_14014ace0``) and the VM lane
(D:\\work\\upstream-1016\\vm\\RESULT.md).

Who counts as "native" (protocol-observed, never address- or name-based):

* ``is_native_identify_session``: the server had to push the session key
  unsolicited (HELLO2 fallback, which never fires for the OG client because the
  OG binds within milliseconds via its constant 0x08 "Hello There") AND the UDP
  endpoint was then bound by a ``13 01 <key>`` identify.  Known right after the
  UDP bind, i.e. before login.
* ``is_native_stream_client``: the client's D_HANDSHAKE advertised receive
  window 0 (the OG advertises 15).  Known only after the UDP D_HANDSHAKE.
* ``is_native_session``: either of the two.
"""
from __future__ import annotations

import struct
from typing import Optional

# c2s 0x54 GENERIC "want" commands the native host sends after TRANSLATION when
# no session provider is loaded (VM run 3: ``54 0014 "portable_physics_v2\0"
# 0001 00 00000001``, then the same for ``_v1``).
PORTABLE_PHYSICS_V2 = "portable_physics_v2"
PORTABLE_PHYSICS_V1 = "portable_physics_v1"
PORTABLE_PHYSICS_WANTS = (PORTABLE_PHYSICS_V2, PORTABLE_PHYSICS_V1)

IDENTIFY_KIND_HELLO_KEY = "hello_key"      # 13 01 <key> on UDP (native shape)
IDENTIFY_KIND_HELLO_THERE = "hello_there"  # 08 "Hello There" (OG shape)


def _session(ctx):
    return getattr(ctx, "session", None) if ctx is not None else None


def is_native_stream_client(ctx) -> bool:
    session = _session(ctx)
    shake = getattr(session, "client_d_handshake", None) if session is not None else None
    return bool(shake) and int(shake.get("receive_window", -1)) == 0


def is_native_identify_session(ctx) -> bool:
    session = _session(ctx)
    if session is None:
        return False
    return bool(getattr(session, "hello_key_pushed", False)) and (
        getattr(session, "udp_identify_kind", "") == IDENTIFY_KIND_HELLO_KEY
    )


def is_native_session(ctx) -> bool:
    return is_native_stream_client(ctx) or is_native_identify_session(ctx)


def is_native_credential_login(ctx) -> bool:
    """Native identify AND the credential-only 0x21/03 login shape."""
    session = _session(ctx)
    return (
        session is not None
        and getattr(session, "login_shape", "") == "credential_only"
        and is_native_identify_session(ctx)
    )


# --------------------------------------------------------------------------- 0x54


def parse_generic_want(packet: bytes) -> Optional[tuple[str, str, int]]:
    """Parse c2s 0x54 GENERIC ``54 <str cmd> <str arg> <u32 value>`` exactly.

    Strings are u16 length + bytes with one trailing NUL trimmed (the client's
    own string writer includes the NUL).  Returns None unless the packet is
    consumed exactly, so the OG client's other 0x54 uses never match.
    """
    if len(packet) < 1 + 2 + 2 + 4 or packet[0] != 0x54:
        return None
    offset = 1
    fields = []
    for _ in range(2):
        if offset + 2 > len(packet):
            return None
        (length,) = struct.unpack_from(">H", packet, offset)
        offset += 2
        if offset + length > len(packet):
            return None
        raw = packet[offset:offset + length]
        offset += length
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        try:
            fields.append(raw.decode("ascii"))
        except UnicodeDecodeError:
            return None
    if offset + 4 != len(packet):
        return None
    (value,) = struct.unpack_from(">I", packet, offset)
    return fields[0], fields[1], value


def portable_physics_want(packet: bytes) -> str:
    """The portable_physics command name if ``packet`` is such a want, else ""."""
    parsed = parse_generic_want(packet)
    if parsed is None:
        return ""
    cmd, _arg, value = parsed
    return cmd if cmd in PORTABLE_PHYSICS_WANTS and value == 1 else ""


# ------------------------------------------------------------------ UDP datagrams


def _control_record_len(data: bytes, offset: int) -> Optional[int]:
    """Length of a fixed-size control record the native demux recognises, else None.

    Native ``FUN_140148880`` (route mode 0): ``01`` IGNORE 4 B, ``04`` D_SET_START
    4 B, ``02`` ACK sub 0 = 6 B / sub 1 = 5 B / sub 2 = 9 B, ``4D`` 1 B.  Any other
    record consumes the rest of the datagram.
    """
    remaining = len(data) - offset
    op = data[offset]
    if op == 0x4D:
        return 1
    if op in (0x01, 0x04):
        return 4 if remaining >= 4 else None
    if op == 0x02 and remaining >= 2:
        sub = data[offset + 1]
        size = {0: 6, 1: 5, 2: 9}.get(sub)
        if size is not None and remaining >= size:
            return size
    return None


def _d_handshake_len(data: bytes, offset: int) -> Optional[int]:
    """Exact length of a D_HANDSHAKE record starting at ``offset``, else None."""
    from . import handlers  # local import: handlers imports this module's users

    parsed = handlers._parse_empirical_client_d_handshake(data[offset:], allow_trailing=True)
    return None if parsed is None else int(parsed["consumed"])


def split_single_message(data: bytes) -> list[bytes]:
    """Split a datagram so each piece holds exactly one message (item 4).

    The native client lets a non-control record swallow the rest of its
    datagram and rejects a D_HANDSHAKE with any tail, so for native sessions
    every message goes in its own datagram.  Only boundaries that are
    unambiguous under the client's demux rules are split: leading fixed-size
    control records and a self-delimiting D_HANDSHAKE.  The server's own 9-byte
    "framed" ACK (``02 <u16 seq> 0009 01 <op> <u16 seq>``) is one message and is
    never split.  Anything else is left whole.
    """
    if len(data) == 9 and data[0] == 0x02 and data[3:5] == b"\x00\x09":
        return [data]
    parts: list[bytes] = []
    offset = 0
    while offset < len(data):
        size = _control_record_len(data, offset)
        if size is None and data[offset] == 0x03:
            size = _d_handshake_len(data, offset)
        if size is None or offset + size >= len(data):
            parts.append(data[offset:])
            break
        parts.append(data[offset:offset + size])
        offset += size
    return parts
