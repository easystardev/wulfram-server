"""NetMixin -- TCP connection + handshake/login, UDP datagram parse/dispatch,
comm-message/chat, and action-input packet handlers, extracted verbatim from
WulframServer (server.py decomposition, networking layer). Method-only mixin;
shares state via `self`. The accept/UDP/game/tick LOOPS stay in the core.
"""
from __future__ import annotations

import math
import os
import secrets
import socket
import struct
import time
import traceback
from typing import Any, Optional

from . import build_uplink, handlers
from .client import ClientContext
from .codec import BitReader
from .session import FEATURES, Phase
from .transport import TCPHandler, print_packet
from .weapons import BehaviorSlot, WeaponSystem
from .packets import (
    PacketType,
    build_hello_session_key,
    build_hello_udp_config,
    build_hello_verified,
    build_identified_udp,
    build_login_status,
    build_ping_reply,
)
from wulfram2_protocol.entities import ACTION_ANALOG_SLOTS, ACTION_DUMP_CONTROL_SLOTS


class NetMixin:
    def _decode_comm_message_request_body(self, body: bytes) -> dict[str, Any]:
        return build_uplink.decode_comm_message_request_body(body)

    def _handle_comm_message_request(
        self,
        ctx: Optional[ClientContext],
        packet: bytes,
        *,
        transport: str,
        body: bytes,
        addr: Optional[tuple] = None,
        sequence: Optional[int] = None,
    ) -> dict[str, Any]:
        return build_uplink.handle_comm_message_request(
            self,
            ctx,
            packet,
            transport=transport,
            body=body,
            addr=addr,
            sequence=sequence,
        )

    def _handle_tcp_comm_message_request(self, ctx: ClientContext, packet: bytes) -> dict[str, Any]:
        return self._handle_comm_message_request(
            ctx,
            packet,
            transport="tcp",
            body=packet[1:],
            addr=getattr(ctx, "client_addr", None),
        )

    def _parse_udp_datagram(self, data: bytes, ctx: Optional[ClientContext] = None):
        """
        Parse a UDP datagram into individual packets.
        Packets can be batched together in a single datagram.

        IMPORTANT: 0x10 is a container packet that wraps other packets.
        We need to extract the inner packets, especially 0x35 VIEWPOINT_INFO.
        """
        def _extract_viewpoint_payloads(raw: bytes):
            """Extract synthetic VIEWPOINT_INFO packets from compressed payloads."""
            payloads = []
            for pos in range(len(raw) - 2):
                if raw[pos] == 0x00 and raw[pos + 1] == 0x00 and raw[pos + 2] == 0x14:
                    payload = raw[pos + 3:pos + 18]
                    if len(payload) < 15:
                        continue
                    # Signature check for viewpoint payload (empirical).
                    if payload[0] == 0xE4 and payload[1] == 0x00 and payload[5] == 0x80 and payload[6] == 0x00 and payload[7] == 0x02 and payload[14] == 0x00:
                        synthetic = b'\x35\x00\x00\x00\x14' + payload
                        payloads.append((pos, synthetic))
            return payloads

        def _action_dump_len() -> int:
            # ACTION_DUMP: opcode + tick(32) + frame(32) + slots 1..21 (bit-packed)
            # Use ctx's weapon system if available, else use default values
            if ctx and ctx.weapon_system:
                control_bits = ctx.weapon_system.control_bits
                zoom_bits = ctx.weapon_system.zoom_bits
            else:
                control_bits = int(os.environ.get("WULFRAM_CONTROL_BITS", "16"))
                zoom_bits = int(os.environ.get("WULFRAM_ZOOM_BITS", str(control_bits)))

            bits = 64
            for slot_idx in range(1, 22):
                if slot_idx == BehaviorSlot.UPWARD_THRUST:
                    bits += zoom_bits
                elif slot_idx in ACTION_DUMP_CONTROL_SLOTS:
                    bits += control_bits
                else:
                    bits += 1
            return 1 + ((bits + 7) // 8)

        def _action_update_len(start: int) -> int | None:
            remaining = len(data) - start
            if remaining <= 0:
                return None
            # ACTION_UPDATE: opcode + count(8) + tick(32) + frame(32) + bit-packed slot/value pairs
            if remaining < 10:
                return None
            reader = BitReader(data[start + 1:])
            # Use ctx's weapon system if available
            if ctx and ctx.weapon_system:
                ws = ctx.weapon_system
            else:
                # Create a temporary weapon system for parsing
                ws = WeaponSystem()
            try:
                count = reader.read_bits(8)
                reader.read_bits(32)
                reader.read_bits(32)
                max_updates = min(count, 64)
                for _ in range(max_updates):
                    slot_idx = reader.read_bits(ws.slot_index_bits)
                    if slot_idx == BehaviorSlot.UPWARD_THRUST:
                        reader.read_bits(ws.zoom_bits)
                    elif slot_idx in ACTION_ANALOG_SLOTS:
                        reader.read_bits(ws.control_bits)
                    else:
                        reader.read_bits(1)
            except ValueError:
                return None
            bitpos = reader.byte_pos * 8 + reader.bit_pos
            pkt_len = 1 + ((bitpos + 7) // 8)
            if 0 < pkt_len <= remaining:
                return pkt_len
            return None

        # First pass: extract viewpoint payloads from any UDP datagram.
        extracted_viewpoints = _extract_viewpoint_payloads(data)
        for pos, pkt in extracted_viewpoints:
            if self.debug_viewpoint:
                print(f"[VIEWPOINT-EXTRACT] payload at offset {pos} len={len(pkt)}")
            yield pkt

        cursor = 0
        while cursor < len(data):
            if cursor >= len(data):
                break

            pkt_type = data[cursor]

            # D_ACK / control acknowledgments.
            # Common wire forms from decompile + captures:
            # - 0x02 0x00 <u32 timestamp>                     (6 bytes)
            # - 0x02 0x01 <u8 channel> <u16 seq>             (5 bytes)
            # - 0x02 0x02 <u32 timestamp> <u8 ch> <u16 seq>  (9 bytes)
            # Legacy server frame used by _send_udp_ack:
            # - 0x02 <u16 our_seq> 0x0009 <u8 sub> <u8 pkt> <u16 seq> (9 bytes)
            if pkt_type == 0x02:
                remaining = len(data) - cursor

                # D_ACK subtype 1 (channel + seq)
                if remaining >= 5 and data[cursor + 1] == 0x01:
                    yield data[cursor:cursor + 5]
                    cursor += 5
                    continue

                # D_ACK subtype 0 (timestamp-only) OR legacy 9-byte server ACK frame.
                if remaining >= 6 and data[cursor + 1] == 0x00:
                    if remaining >= 9 and data[cursor + 3:cursor + 5] == b"\x00\x09":
                        yield data[cursor:cursor + 9]
                        cursor += 9
                        continue
                    yield data[cursor:cursor + 6]
                    cursor += 6
                    continue

                # D_ACK subtype 2 (timestamp + channel + seq)
                if remaining >= 9 and data[cursor + 1] == 0x02:
                    yield data[cursor:cursor + 9]
                    cursor += 9
                    continue

                # Fallback for older/unknown framing seen in captures.
                # Keep the previous +10 scan as a last resort to salvage
                # inner packets for analysis, then surface the raw frame.
                salvaged = False
                if remaining >= 11:
                    payload = data[cursor + 10:]
                    if payload and any(payload):
                        for inner in self._parse_udp_datagram(payload, ctx):
                            salvaged = True
                            yield inner
                if not salvaged:
                    yield data[cursor:]
                break

            # 0x10 CONTAINER packet - wraps other packets (including 0x35 VIEWPOINT_INFO)
            # The actual format needs empirical discovery - search for 0x35 in the data
            if pkt_type == 0x10:
                # 0x10 appears to carry compressed viewpoint payloads.
                raw = data[cursor:]
                extracted = len(extracted_viewpoints)
                # Debug: look for literal 0x35 packets inside the container as a fallback.
                if extracted == 0:
                    full_hex = data[cursor:].hex()
                    if '35' in full_hex:
                        positions = [i for i, b in enumerate(raw) if b == 0x35]
                        for pos in positions:
                            start = max(0, pos - 4)
                            end = min(len(raw), pos + 20)
                            context = raw[start:end].hex()
                            if self.debug_viewpoint:
                                print(f"[0x10-SCAN] Found 0x35 at offset {pos}: ...{context}...")
                            if pos + 5 <= len(raw):
                                potential_pkt = raw[pos:]
                                if len(potential_pkt) >= 5 and potential_pkt[3:5] == b"\x00\x14":
                                    pkt_len = 20
                                    if pos + pkt_len <= len(raw):
                                        if self.debug_viewpoint:
                                            print(f"[0x10-EXTRACT] Extracting 0x35 at offset {pos}, declared len={pkt_len}")
                                        yield potential_pkt[:pkt_len]

                # Consume entire 0x10 packet
                break

            # Reliable stream packets with length at bytes 3-4
            if pkt_type in (0x20, 0x25, 0x26, 0x2B, 0x33, 0x35, 0x3A, 0x3B):
                if cursor + 5 > len(data):
                    yield data[cursor:]
                    break
                pkt_len = struct.unpack(">H", data[cursor+3:cursor+5])[0]
                if pkt_len < 5:
                    pkt_len = len(data) - cursor
                end = cursor + pkt_len
                if end > len(data):
                    end = len(data)
                yield data[cursor:end]
                cursor = end

            # Fixed-size packets
            elif pkt_type == 0x40:  # INPUT_FEEDBACK - 5 bytes (opcode + u32 frame)
                pkt_len = 5
                yield data[cursor:cursor+pkt_len]
                cursor += pkt_len

            elif pkt_type == 0x04:  # D_SET_START - channel(u8) + sequence(u16)
                pkt_len = min(4, len(data) - cursor)
                yield data[cursor:cursor+pkt_len]
                cursor += pkt_len

            elif pkt_type == 0x0B:  # PING_REQUEST - opcode + u32 request_id + u32 frame_count
                pkt_len = min(9, len(data) - cursor)
                yield data[cursor:cursor+pkt_len]
                cursor += pkt_len

            elif pkt_type == 0x2E:  # WEAPON_DEMAND - ~10 bytes
                pkt_len = min(10, len(data) - cursor)
                yield data[cursor:cursor+pkt_len]
                cursor += pkt_len

            elif pkt_type == 0x0C:  # Unknown - seems to be 9 bytes
                pkt_len = min(9, len(data) - cursor)
                yield data[cursor:cursor+pkt_len]
                cursor += pkt_len

            elif pkt_type == 0x09:  # ACTION_DUMP - variable, has length info
                pkt_len = _action_dump_len()
                end = cursor + pkt_len
                if end > len(data):
                    end = len(data)
                yield data[cursor:end]
                cursor = end

            elif pkt_type == 0x0A:  # ACTION_UPDATE - variable
                pkt_len = _action_update_len(cursor)
                if pkt_len:
                    end = cursor + pkt_len
                    yield data[cursor:end]
                    cursor = end
                else:
                    end = cursor + 1
                    while end < len(data) and data[end] not in (
                        0x02, 0x03, 0x04, 0x09, 0x0A, 0x0B, 0x0C, 0x10,
                        0x20, 0x25, 0x26, 0x2B, 0x2E, 0x33, 0x35, 0x3A, 0x3B, 0x40, 0x49,
                    ):
                        end += 1
                    yield data[cursor:end]
                    cursor = end

            else:
                # Unknown packet - consume rest of datagram
                yield data[cursor:]
                break

    def _handle_udp_d_ack(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle D_ACK/service-layer ACK packet (0x02)."""
        if len(data) < 2:
            return

        # Legacy ACK frame used by our reliable helpers:
        # 0x02 + our_seq(u16) + len(u16=9) + subcmd(u8) + packet_id(u8) + seq(u16)
        if len(data) >= 9 and data[3:5] == b"\x00\x09":
            our_seq = struct.unpack(">H", data[1:3])[0]
            subcmd = data[5]
            packet_id = data[6]
            seq_num = struct.unpack(">H", data[7:9])[0]
            if self.debug_udp_raw:
                print(
                    f"[UDP] ACK_FRAME 0x02 from {addr}: "
                    f"our_seq={our_seq} sub={subcmd} packet=0x{packet_id:02X} seq={seq_num}"
                )
            return

        subtype = data[1]
        if subtype == 0x00 and len(data) >= 6:
            timestamp = struct.unpack(">I", data[2:6])[0]
            if self.debug_udp_raw:
                print(f"[UDP] D_ACK subtype=0 ts={timestamp} from {addr}")
            return

        if subtype == 0x01 and len(data) >= 5:
            channel = data[2]
            seq_num = struct.unpack(">H", data[3:5])[0]
            if self.debug_udp_raw:
                print(f"[UDP] D_ACK subtype=1 channel={channel} seq={seq_num} from {addr}")
            return

        if subtype == 0x02 and len(data) >= 9:
            timestamp = struct.unpack(">I", data[2:6])[0]
            channel = data[6]
            seq_num = struct.unpack(">H", data[7:9])[0]
            if self.debug_udp_raw:
                print(
                    f"[UDP] D_ACK subtype=2 ts={timestamp} "
                    f"channel={channel} seq={seq_num} from {addr}"
                )
            return

        if self.debug_udp_raw:
            print(f"[UDP] D_ACK malformed/unknown from {addr}: len={len(data)} data={data.hex()}")

    def _handle_udp_d_set_start(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle D_SET_START stream control packet (0x04)."""
        if len(data) < 4:
            if self.debug_udp_raw:
                print(f"[UDP] D_SET_START malformed from {addr}: len={len(data)} data={data.hex()}")
            return

        channel = data[1]
        seq_num = struct.unpack(">H", data[2:4])[0]

        # Mirror service-layer behavior: acknowledge with D_ACK subtype 1.
        if self.udp_handler:
            ack = bytes((0x02, 0x01, channel)) + struct.pack(">H", seq_num)
            self.udp_handler.send_to(ack, addr)

        if self.debug_udp_raw:
            print(f"[UDP] D_SET_START channel={channel} seq={seq_num} from {addr}")

    def _handle_single_udp_packet(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle a single UDP packet (after parsing from datagram)."""
        if not data:
            return

        pkt_type = data[0]

        # Update ctx if we found it, or try to find it from addr
        if ctx is None:
            ctx = self.udp_addr_to_client.get(addr)
        if ctx is None and pkt_type not in (0x03, 0x08, 0x13):
            ctx = self._recover_udp_client(addr)

        # For some packet types (HELLO, D_HANDSHAKE), we may not have ctx yet
        # These packets help identify/register the client

        # Diagnostic: log reliable stream packets when debugging packet flow.
        if (
            pkt_type in (0x33, 0x35, 0x25, 0x20, 0x3a)
            and (self.debug_udp_raw or (self.debug_viewpoint and pkt_type == 0x35))
        ):
            print(f"[UDP-RELIABLE] 0x{pkt_type:02X} len={len(data)} head={data[:min(16,len(data))].hex()}")

        # Handle UDP HELLO (session key verification)
        if pkt_type == 0x08:
            # D_Protocol carrier vs simple HELLO_ACK
            if len(data) >= 2 and 0x01 <= data[1] <= 0x04:
                d_type = data[1]
                if d_type == 0x03:
                    # Treat payload as D_HANDSHAKE (strip carrier byte)
                    self._handle_udp_d_handshake(ctx, data[1:], addr)
                elif d_type == 0x02:
                    self._handle_udp_d_ack(ctx, data[1:], addr)
                elif d_type == 0x04:
                    self._handle_udp_d_set_start(ctx, data[1:], addr)
                else:
                    print(f"[UDP] D_Protocol type=0x{d_type:02X} from {addr} (len={len(data)})")
            else:
                # HELLO_ACK - this registers the UDP address for a client
                text = data[1:].decode('ascii', errors='ignore').strip('\x00')
                print(f"[UDP] HELLO_ACK from {addr}: '{text}'")

                # Try session-key-based match first (deterministic)
                if ctx is None and text:
                    matched = self.session_key_to_client.get(text)
                    if matched:
                        ctx = matched
                        print(f"[UDP] HELLO_ACK matched by session key -> client {ctx.client_id}")

                # Fallback: find a client waiting for UDP verification
                if ctx is None:
                    from .session import Phase
                    with self.clients_lock:
                        for c in self.clients.values():
                            if c.session and c.session.phase == Phase.HANDSHAKE and not c.session.udp_verified:
                                ctx = c
                                print(f"[UDP] Matched HELLO_ACK to client {ctx.client_id} in HANDSHAKE state (heuristic)")
                                break

                if ctx:
                    self._bind_udp_client(ctx, addr, reason="udp_hello_ack")

        elif pkt_type == 0x13:
            # Session key â€” use to deterministically bind UDP addr â†’ client
            # Format: 0x13 + subcmd(1) + length(2) + key_bytes
            if len(data) > 4:
                key = data[4:].decode('ascii', errors='ignore').strip('\x00')
                matched = self.session_key_to_client.get(key)
                if matched:
                    if self._bind_udp_client(matched, addr, reason=f"session_key '{key}'"):
                        ctx = matched
                else:
                    print(f"[UDP] Session key from {addr}: '{key}' (NO MATCH)")

        elif pkt_type == 0x33:
            # TRANSLATION_ACK (client -> server, reliable stream packet)
            # MUST ACK this or client stalls reliable-send window and won't send 0x35!
            if ctx:
                ctx.session.translation_ack_received = True
                ctx.session.translation_ack_time = time.monotonic()
                ctx.session.last_udp_activity = time.monotonic()
            print(f"[UDP] TRANSLATION_ACK from {addr} (len={len(data)})")
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x33, seq_num)
                print(f"[UDP] Sent ACK for 0x33 seq={seq_num}")
            if (
                ctx
                and ctx.session.pending_spawn_points
                and not ctx.session.spawn_points_sent
                and ctx.session.udp_d_handshake_received
            ):
                handlers._send_spawn_points_for_client(self, ctx)

        elif pkt_type == 0x03:
            # D_HANDSHAKE (Wulf-Forge UDP stream init)
            self._handle_udp_d_handshake(ctx, data, addr)

        elif pkt_type == 0x02:
            # D_ACK / reliable ACK control frame
            self._handle_udp_d_ack(ctx, data, addr)

        elif pkt_type == 0x04:
            # D_SET_START stream sequence control
            self._handle_udp_d_set_start(ctx, data, addr)

        elif pkt_type == 0x20:
            # COMM_REQ (chat/system commands) - used by Wulf-Forge for /s spawn
            self._handle_udp_chat(ctx, data, addr)

        elif pkt_type == 0x25:
            # REINCARNATE over UDP (Wulf-Forge style)
            self._handle_udp_reincarnate(ctx, data, addr)

        elif pkt_type == 0x26:
            # RETARGET (reliable stream): acknowledge to prevent resend storms.
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x26, seq_num)
                if self.debug_udp_raw:
                    print(f"[UDP] RETARGET seq={seq_num} from {addr}")

        elif pkt_type == 0x35:
            # VIEWPOINT_INFO - client sends camera/view position and orientation
            # This is the ACTUAL player pose, not reconstructed from inputs!
            if self.debug_viewpoint:
                print(f"[UDP] VIEWPOINT_INFO 0x35 len={len(data)} data={data[:32].hex()}...")
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x35, seq_num)
                self._handle_viewpoint_info(ctx, data, addr)

        elif pkt_type == 0x3a:
            # BEACON_REQ - client requests beacon info
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x3a, seq_num)

        elif pkt_type == 0x3B:
            # BEACON_MODIFY (reliable stream): acknowledge.
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x3B, seq_num)
                if self.debug_udp_raw:
                    print(f"[UDP] BEACON_MODIFY seq={seq_num} from {addr}")

        elif pkt_type == 0x2B:
            # DROP_REQUEST (reliable stream): drop_cargo (`,`) AND deploy_cargo (`.`)
            # both ride this single opcode (the only cargo-action opcode -- Debug.c:309).
            # Wire format (captured live 2026-06-27): opcode(0x2b) + seq(u16) +
            # len(u16=0x0009) + mode(u32), mode 0 = drop, 1 = deploy. No position/type:
            # the server applies it using the player's authoritative pos + carried type.
            if len(data) >= 3:
                seq_num = struct.unpack(">H", data[1:3])[0]
                self._send_udp_ack(ctx, addr, 0x2B, seq_num)
                mode = struct.unpack(">I", data[5:9])[0] if len(data) >= 9 else 0
                if self.debug_udp_raw:
                    print(f"[UDP] DROP_REQUEST seq={seq_num} mode={mode} "
                          f"len={len(data)} data={data.hex()} from {addr}")
                if getattr(self, "cargo_deploy_enabled", True) and ctx is not None:
                    try:
                        self._handle_drop_request(ctx, mode)
                    except Exception as exc:  # noqa: BLE001 - never break the UDP loop
                        print(f"[DROP_REQUEST] handler error: {exc}")

        elif pkt_type == 0x09:
            # ACTION_DUMP - full behavior slot dump (includes fire state)
            if ctx is None:
                ctx = self._ghost_rejoin(addr)
            if self.debug_sync:
                print(f"[UDP] ACTION_DUMP received: len={len(data)} data={data[:24].hex()}")
            self._handle_action_dump(ctx, data, addr)

        elif pkt_type == 0x0A:
            # ACTION_UPDATE - incremental behavior slot updates
            self._handle_action_update(ctx, data, addr)

        elif pkt_type == 0x40:
            # INPUT_FEEDBACK - player input state (movement, firing)
            self._handle_input_feedback(ctx, data, addr)

        elif pkt_type == 0x2E:
            # WEAPON_DEMAND - fire button pressed
            print(f"[UDP] WEAPON_DEMAND from {addr} (len={len(data)}) data={data.hex()}")
            self._handle_weapon_demand(ctx, data, addr)

        elif pkt_type == 0x0B:
            # Client ping request: reply on 0x0C with the same request payload.
            if len(data) >= 9 and self.udp_handler and ctx and self._udp_ping_reply_allowed_for_client(ctx):
                request_id, frame_count = struct.unpack(">II", data[1:9])
                self.udp_handler.send_to(build_ping_reply(request_id), addr)
                print(
                    f"[UDP] PING_REPLY 0x0C to {addr} "
                    f"request_id={request_id} frame_count={frame_count}"
                )
            elif len(data) >= 5 and self.udp_handler and ctx and self._udp_ping_reply_allowed_for_client(ctx):
                # Keep a compatibility fallback for older simplified clients.
                request_id = struct.unpack(">I", data[1:5])[0]
                self.udp_handler.send_to(build_ping_reply(request_id), addr)
                print(f"[UDP] PING_REPLY 0x0C to {addr} request_id={request_id} frame_count=0")

        elif pkt_type == 0x0C:
            # STATE_REQUEST - may contain state/position info
            if self.debug_sync:
                print(f"[UDP] STATE_REQUEST 0x0C len={len(data)} data={data.hex()}")
            self._handle_state_request(ctx, data, addr)

        else:
            print(f"[UDP] Packet 0x{pkt_type:02X} from {addr} (len={len(data)})")

        # Track last UDP activity for mapped clients regardless of packet type.
        if ctx:
            ctx.session.last_udp_activity = time.monotonic()

    def _bind_udp_client(self, ctx: ClientContext, addr: tuple, *, reason: str) -> bool:
        """Bind a UDP endpoint to a client without stealing another active mapping."""
        current = self.udp_addr_to_client.get(addr)
        if current is not None and current is not ctx:
            print(
                f"[UDP] Refusing to bind {addr} to client {ctx.client_id}: "
                f"already owned by client {current.client_id} ({reason})"
            )
            return False

        old_addr = ctx.session.udp_addr
        if old_addr and old_addr != addr and self.udp_addr_to_client.get(old_addr) is ctx:
            del self.udp_addr_to_client[old_addr]

        ctx.session.udp_addr = addr
        ctx.session.udp_verified = True
        ctx.session.last_udp_activity = time.monotonic()
        self.udp_addr_to_client[addr] = ctx
        print(f"[UDP] Bound client {ctx.client_id} to {addr} via {reason}")
        return True

    def _recover_udp_client(self, addr: tuple, *, allow_handshake: bool = False) -> Optional[ClientContext]:
        """
        Recover a missing UDP->client mapping when the client reconnects/rebinds.
        This is a safe fallback only: ambiguous same-host candidates are rejected.
        """
        existing = self.udp_addr_to_client.get(addr)
        if existing:
            return existing

        ip = addr[0] if addr else None
        if not ip:
            return None

        candidates: list[ClientContext] = []
        with self.clients_lock:
            for c in self.clients.values():
                if not c or not c.running or not c.session:
                    continue
                if not c.client_addr or c.client_addr[0] != ip:
                    continue
                if c.session.phase == Phase.DISCONNECTED:
                    continue
                if c.session.phase == Phase.HANDSHAKE:
                    if not allow_handshake:
                        continue
                    if c.session.udp_verified or c.session.udp_config_sent_time <= 0.0:
                        continue
                # Skip clients that already have a verified UDP address on a
                # different port -- recovering them would steal the mapping and
                # break the other client's UDP routing (localhost two-client bug).
                if (c.session.udp_verified and c.session.udp_addr
                        and c.session.udp_addr != addr
                        and c.session.udp_addr in self.udp_addr_to_client):
                    continue
                candidates.append(c)

        if not candidates:
            return None
        if len(candidates) != 1:
            cand_ids = ",".join(str(c.client_id) for c in candidates)
            print(f"[UDP] Recover addr {addr}: ambiguous candidates [{cand_ids}]")
            return None

        ctx = candidates[0]
        if self._bind_udp_client(ctx, addr, reason="safe_recovery"):
            return ctx
        return None

    def _handle_udp_d_handshake(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle UDP D_HANDSHAKE (0x03) and respond with stream definitions."""
        handlers.handle_udp_d_handshake(self, ctx, data, addr)

    def _send_udp_ack(self, ctx: Optional[ClientContext], addr: tuple, packet_id: int, seq_num: int, subcmd: int = 1):
        """Send a Wulf-Forge style UDP ACK (0x02)."""
        handlers.send_udp_ack(self, ctx, addr, packet_id, seq_num, subcmd)

    def _handle_udp_chat(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle UDP COMM_REQ (0x20) for /s spawn."""
        handlers.handle_udp_chat(self, ctx, data, addr)

    def _handle_udp_reincarnate(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """Handle UDP REINCARNATE (0x25)."""
        handlers.handle_udp_reincarnate(self, ctx, data, addr)

    def _handle_team_switch(self, ctx: ClientContext, team_id: int, addr: tuple):
        """Handle team switch/spawn request from REINCARNATE."""
        handlers.handle_team_switch(self, ctx, team_id, addr)

    def _handle_client(self, sock: socket.socket, addr):
        """Handle a single client connection in its own thread."""
        # Create client context with unique IDs
        ctx = self._create_client_context(addr)
        ctx.session.transition_to(Phase.HANDSHAKE)

        # Note: Don't set timeout here - it interferes with login flow
        # Set timeout later when entering game loop
        ctx.tcp_handler = TCPHandler(sock, self.logger)

        # Register client in global dict
        with self.clients_lock:
            self.clients[ctx.client_id] = ctx

        # Wire control server to this client's handlers
        self.control_server.tcp_handler = ctx.tcp_handler
        self.control_server.session = ctx.session

        print(f"[SERVER] Client {ctx.client_id} assigned entity_id={ctx.entity_id}")

        try:
            # Phase 1: Handshake
            self._do_handshake(ctx)

            # Phase 2: Login
            self._do_login(ctx)

            # Phase 3: Team Select / Game Loop
            self._game_loop(ctx)

        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError) as e:
            phase_name = ctx.session.phase.name if ctx.session.phase else "UNKNOWN"
            print(
                f"[SERVER] Client {ctx.client_id} disconnected during "
                f"{phase_name}: {e}"
            )
        except Exception as e:
            print(f"[SERVER] Client {ctx.client_id} error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Preserve UDP addr before session reset so we can clean mapping safely.
            udp_addr = ctx.session.udp_addr
            disconnected_entity_id = ctx.session.entity_id or ctx.entity_id
            was_in_game = bool(ctx.session.in_game and disconnected_entity_id)
            ctx.running = False
            ctx.session.in_game = False

            # Remove from client tracking
            with self.clients_lock:
                if ctx.client_id in self.clients:
                    del self.clients[ctx.client_id]

            if was_in_game:
                self._broadcast_disconnected_player_delete(ctx, disconnected_entity_id)

            # Drop the departed player's scoreboard row on every other client
            # (REMOVE_FROM_ROSTER 0x1B). Roster presence is login-gated, not
            # spawn-gated, so remove for any logged-in client, not just in-game.
            # Runs before session.reset() while player_id is still valid.
            if ctx.session.login_complete:
                self._broadcast_roster_removal(ctx)

            # Remove UDP address mapping only if it still points to this context.
            if udp_addr and self.udp_addr_to_client.get(udp_addr) is ctx:
                del self.udp_addr_to_client[udp_addr]

            # Remove session key mapping
            skey = ctx.session.session_key
            if skey and self.session_key_to_client.get(skey) is ctx:
                del self.session_key_to_client[skey]

            ctx.session.reset()

            sock.close()
            print(f"[SERVER] Client {ctx.client_id} disconnected")

    def _do_handshake(self, ctx: ClientContext):
        """Perform initial handshake."""
        time.sleep(0.5)  # Let client initialize

        login_bootstrap_mode = handlers._get_login_bootstrap_mode(ctx)
        use_og_handshake = (login_bootstrap_mode == "og")

        # Loopback/minimal clients still use the early session key path for
        # deterministic UDP binding. OG clients expect HELLO subtype 1 first
        # and only request HELLO subtype 2 later.
        if not use_og_handshake:
            if not ctx.session.session_key:
                ctx.session.session_key = f"WS-{ctx.client_id}-{secrets.token_hex(8)}"
            self.session_key_to_client[ctx.session.session_key] = ctx
            ctx.tcp_handler.send(build_hello_session_key(ctx.session.session_key))

        # Send UDP config - use public_addr, or resolve from client's TCP connection
        udp_addr = self.public_addr
        if udp_addr == "0.0.0.0":
            # Use the local address the client actually connected to
            udp_addr = ctx.tcp_handler.sock.getsockname()[0]
        # Advertised UDP endpoint overrides (default = no change). Lets the server
        # sit behind a NAT / a latency proxy: the client is told to send UDP to the
        # proxy's host:port instead of the server's own. Universal, no client fork.
        adv_host = os.environ.get("WULFRAM_ADVERTISE_UDP_HOST", "").strip()
        if adv_host:
            udp_addr = adv_host
        try:
            adv_port = int(os.environ.get("WULFRAM_ADVERTISE_UDP_PORT", "") or self.port)
        except ValueError:
            adv_port = self.port
        print(
            f"[HANDSHAKE] Client {ctx.client_id}: mode={'og' if use_og_handshake else 'minimal'} "
            f"UDP config {udp_addr}:{adv_port}"
        )
        ctx.tcp_handler.send(build_hello_udp_config(udp_addr, adv_port))
        ctx.session.udp_config_sent_time = time.monotonic()

        # Wait for UDP verification
        timeout = time.monotonic() + 5.0
        while not ctx.session.udp_verified and time.monotonic() < timeout:
            time.sleep(0.1)

        if ctx.session.udp_verified:
            print(f"[HANDSHAKE] Client {ctx.client_id} UDP verified")
            ctx.tcp_handler.send(build_identified_udp())
        else:
            print(f"[HANDSHAKE] Client {ctx.client_id} UDP verification timeout")

        # Mirror Wulf-Forge: send HELLO verified (subcmd 0x03) after UDP setup
        ctx.tcp_handler.send(build_hello_verified())

        ctx.session.transition_to(Phase.LOGIN)

    def _do_login(self, ctx: ClientContext):
        """Handle login sequence."""
        # Request username
        ctx.tcp_handler.send(build_login_status(5))  # Code 5 = request handle

        if FEATURES.auto_login:
            # Wait briefly for client's LOGIN_REQUEST which carries the username,
            # then auto-advance to team select regardless.
            # Auto-login: consume LOGIN_REQUEST to extract username, then
            # proceed directly to bootstrap.  HELLO packets are queued
            # and replied to AFTER the bootstrap — the OG client's login
            # state machine expects bootstrap packets (TEAM_INFO etc.)
            # immediately after LOGIN_STATUS(5), not HELLO responses.
            ctx.tcp_handler.sock.settimeout(1.0)
            pending_hellos = []
            try:
                for _ in range(5):
                    packet = ctx.tcp_handler.recv()
                    if packet and len(packet) >= 2 and packet[0] == PacketType.LOGIN_REQUEST:
                        username, _ = handlers.decode_lp_string(packet, 2)
                        handlers.apply_submitted_username(ctx.session, username)
                        break
                    elif packet and len(packet) >= 1 and packet[0] == PacketType.HELLO:
                        pending_hellos.append(packet)
            except Exception:
                pass
            finally:
                ctx.tcp_handler.sock.settimeout(None)
            if not ctx.session.username:
                ctx.session.username = f"Player{ctx.client_id}"
            ctx.session.login_complete = True
            ctx.session.transition_to(Phase.TEAM_SELECT)
            handlers.send_initial_game_data(self, ctx)
            # Now reply to queued HELLOs (after bootstrap, matching old packet order)
            for hello_pkt in pending_hellos:
                self._handle_hello(ctx, hello_pkt)
            print(f"[LOGIN] Client {ctx.client_id}: Auto-login as {ctx.session.username}")
            return

        # Wait for packets
        while ctx.session.phase == Phase.LOGIN:
            packet = ctx.tcp_handler.recv()
            if packet is None:
                raise ConnectionError("Client disconnected during login")

            if len(packet) < 1:
                continue

            pkt_type = packet[0]
            print_packet("RECV", pkt_type, packet)

            if pkt_type == PacketType.HELLO:
                self._handle_hello(ctx, packet)
            elif pkt_type == PacketType.LOGIN_REQUEST:
                self._handle_login_request(ctx, packet)
            elif pkt_type == 0x54:
                # Voice data request - ignore
                pass
            else:
                print(f"[LOGIN] Ignoring packet 0x{pkt_type:02X}")

    def _handle_hello(self, ctx: ClientContext, packet: bytes):
        """Handle HELLO packets during login."""
        handlers.handle_hello(self, ctx, packet)

    @staticmethod
    def _decode_lp_string(data: bytes, offset: int):
        """Decode a length-prefixed string. Returns (text, new_offset)."""
        return handlers.decode_lp_string(data, offset)

    def _handle_login_request(self, ctx: ClientContext, packet: bytes):
        """Handle LOGIN_REQUEST packet."""
        handlers.handle_login_request(self, ctx, packet)

    def _send_initial_game_data(self, ctx: ClientContext):
        """Send packets needed for team selection screen."""
        handlers.send_initial_game_data(self, ctx)

    def _effective_inactivity_timeout(self, ctx: ClientContext) -> float:
        """Remote in-game sessions can legitimately go quiet while idle."""
        timeout = self.inactivity_timeout
        addr = ctx.client_addr[0] if ctx.client_addr else ""
        is_remote = addr not in ("127.0.0.1", "::1", "localhost")
        if is_remote:
            timeout = max(timeout, self.remote_idle_timeout)
        return timeout

    def _handle_bps(self, ctx: ClientContext, packet: bytes):
        """Handle BPS (bandwidth/rate) packet."""
        handlers.handle_bps(self, ctx, packet)

    def _handle_want_updates(self, ctx: ClientContext, packet: bytes):
        """Handle WANT_UPDATES - client is ready for game data."""
        handlers.handle_want_updates(self, ctx, packet)

    def _auto_join_team(self, ctx: ClientContext, team_id: int):
        """Auto-spawn after WANT_UPDATES using Wulf-Forge-style UDP TANK."""
        now = time.monotonic()
        if ctx.session and (ctx.session.in_game or ctx.session.phase == Phase.IN_GAME):
            block = handlers.recent_control_pose_spawn_block(ctx, now=now)
            if block["blocked"]:
                print(
                    f"[GAME] Client {ctx.client_id}: Ignoring auto-spawn for team {team_id} "
                    f"after recent control pose reset "
                    f"(age={block['age_s']:.2f}s < block={block['block_s']:.2f}s)"
                )
                ctx.session.delayed_spawn_team = 0
                ctx.session.delayed_spawn_time = 0
                return
        print(f"[GAME] Client {ctx.client_id}: Auto-spawn (WF) on team {team_id}")
        # Check for pending respawn position (set by respawn command)
        pos = None
        if getattr(ctx, 'pending_respawn_pos', None):
            pos = ctx.pending_respawn_pos
            ctx.pending_respawn_pos = None
            print(f"[SPAWN] Using pending respawn pos={pos}")
        else:
            pos = self._resolve_spawn_pos(team_id)
            configured_default = self._get_configured_default_spawn_pos()
            if configured_default is not None:
                print(f"[SPAWN] Using default flat spawn pos={pos}")
            else:
                spawn = self._pick_spawn_point(team_id)
                if spawn:
                    print(f"[SPAWN] Using spawn point oid={spawn['oid']} team={team_id} pos={pos}")
        self._spawn_wf_style(ctx, team_id=team_id, net_id=ctx.session.player_id or ctx.entity_id, pos=pos)

    def _handle_reincarnate(self, ctx: ClientContext, packet: bytes):
        """Handle REINCARNATE - player wants to spawn."""
        handlers.handle_reincarnate_tcp(self, ctx, packet)

    def _handle_viewpoint_info(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle VIEWPOINT_INFO (0x35) - contains camera/view orientation (not position).

        Discovered format:
        - Byte 0: opcode (0x35)
        - Bytes 1-2: sequence number (for ACK)
        - Bytes 3-10: pitch angle as big-endian double (radians)
        - Bytes 11-18: yaw angle as big-endian double (radians)

        This provides the aim/view direction, not world position.
        Position comes from entity state/TankPacket.
        """
        if ctx is None:
            return

        if self.debug_viewpoint:
            print(f"[UDP] VIEWPOINT_INFO 0x35 len={len(data)} data={data.hex()[:40]}...")

        if len(data) < 20:
            # Short packet - might be different subtype, log for analysis
            if self.debug_viewpoint and len(data) >= 5:
                print(f"[VIEWPOINT-SHORT] len={len(data)} bytes={data.hex()}")
            return

        if ctx.session and not ctx.session.input_ready:
            ctx.session.input_ready = True
            ctx.session.input_ready_time = time.monotonic()
            print(f"[GAME] Client {ctx.client_id}: input ready (VIEWPOINT_INFO)")

        seq_num = struct.unpack(">H", data[1:3])[0]
        # Empirical format (20 bytes total):
        # [0]=opcode(0x35) [1:3]=seq [3:5]=len(0x0014)
        # payload (15 bytes) appears to include 24-bit pitch/yaw fields.
        payload = data[5:]

        if len(payload) >= 14:
            try:
                # 24-bit fixed values (0..2^24) scaled to degrees.
                pitch_raw = (payload[2] << 16) | (payload[3] << 8) | payload[4]
                yaw_raw = (payload[11] << 16) | (payload[12] << 8) | payload[13]
                pitch_deg = (pitch_raw / (1 << 24)) * 360.0
                yaw_deg = (yaw_raw / (1 << 24)) * 360.0 - 180.0
                if pitch_deg > 180.0:
                    pitch_deg -= 360.0

                pitch = math.radians(pitch_deg)
                yaw = math.radians(yaw_deg)

                # Update aim pose (view/turret) with view angles
                old_yaw = ctx.player_aim_yaw
                old_pitch = ctx.player_aim_pitch
                ctx.player_aim_pitch = pitch
                ctx.player_aim_yaw = yaw
                ctx.player_aim_source = "viewpoint"
                ctx.player_aim_time = time.monotonic()

                # Track viewpoint packet count
                ctx.viewpoint_count = ctx.viewpoint_count + 1

                # Log ALL viewpoint packets for debugging
                old_yaw_deg = math.degrees(old_yaw)
                # Compare server-tracked heading vs client viewpoint
                heading_deg = math.degrees(ctx.player_heading)
                delta_deg = yaw_deg - heading_deg
                # Normalize to -180..180
                while delta_deg > 180.0:
                    delta_deg -= 360.0
                while delta_deg < -180.0:
                    delta_deg += 360.0
                if self.debug_viewpoint:
                    print(
                        f"[VIEWPOINT #{ctx.viewpoint_count}] pitch={pitch_deg:.1f} "
                        f"yaw={yaw_deg:.1f} (was {old_yaw_deg:.1f})"
                    )
                    print(
                        f"[HEADING-COMPARE] server={heading_deg:.1f} client={yaw_deg:.1f} "
                        f"delta={delta_deg:.1f}deg"
                    )
                return
            except (IndexError, ValueError, struct.error) as e:
                if self.debug_viewpoint:
                    print(f"[VIEWPOINT-ERR] Failed to decode: {e}")

        # Fallback: attempt double decode if payload is larger in other variants.
        if len(payload) >= 17:
            try:
                pitch = struct.unpack(">d", payload[0:8])[0]
                yaw = struct.unpack(">d", payload[8:16])[0]

                old_yaw = ctx.player_aim_yaw
                ctx.player_aim_pitch = pitch
                ctx.player_aim_yaw = yaw
                ctx.player_aim_source = "viewpoint"
                ctx.player_aim_time = time.monotonic()

                ctx.viewpoint_count = ctx.viewpoint_count + 1
                if self.debug_viewpoint:
                    print(
                        f"[VIEWPOINT #{ctx.viewpoint_count}] pitch={math.degrees(pitch):.1f} "
                        f"yaw={math.degrees(yaw):.1f} (was {math.degrees(old_yaw):.1f})"
                    )
            except struct.error as e:
                if self.debug_viewpoint:
                    print(f"[VIEWPOINT-ERR] Failed to decode (double): {e}")

    def _handle_action_dump(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle ACTION_DUMP packet (0x09).
        Contains all behavior slot values including fire state.
        Format: [opcode:1] [tick:4] [frame:4] [slot_data:bit-packed]
        """
        if ctx is None:
            return

        client_tick = 0
        if len(data) >= 5:
            try:
                client_tick = struct.unpack(">I", data[1:5])[0]
                self._sync_tick_offset(ctx, client_tick)
            except struct.error:
                pass
        if ctx.weapon_system.decode_action_dump(data):
            ctx.last_action_dump_time = time.monotonic()
            self._record_client_action_telemetry(ctx, "ACTION_DUMP", client_tick)

            if ctx.session and not ctx.session.input_ready:
                ctx.session.input_ready = True
                ctx.session.input_ready_time = time.monotonic()
                print(f"[GAME] Client {ctx.client_id}: input ready (ACTION_DUMP)")
            self._update_player_aim(ctx)
            # Update weapon system with current pose
            ctx.weapon_system.player_id = ctx.session.player_id or ctx.entity_id
            ctx.weapon_system.player_team = ctx.session.team_id or 2
            fire_pos, fire_rot, aim_src, pose_src = self._select_weapon_fire_pose(ctx, client_tick)
            ctx.weapon_system.player_pos = fire_pos
            ctx.weapon_system.player_rot = fire_rot
            ctx.weapon_system.projectile_aim_source = aim_src
            ctx.weapon_system.use_pitch = bool(getattr(self, "projectile_body_pitch", False))

            # Weapon spawning with wulf-forge encoding
            if self.projectiles_enabled:
                energy_available = ctx.player_energy if self.weapon_energy_enabled else None
                new_projectiles, energy_spent = ctx.weapon_system.update(
                    available_energy=energy_available
                )
                if energy_spent > 0.0:
                    self._consume_player_energy(ctx, energy_spent)
                if new_projectiles:
                    yaw_deg = math.degrees(ctx.player_yaw)
                    turn_val = ctx.weapon_system.behavior_slots[BehaviorSlot.TURNING]
                    aim_pitch, aim_yaw, aim_src = self._get_aim_rotation(ctx)
                    vp_yaw = math.degrees(ctx.player_aim_yaw)
                    print(
                        f"[WEAPON-FIRE] Firing {len(new_projectiles)} proj "
                        f"heading={math.degrees(ctx.player_heading):.1f} "
                        f"aim_yaw={math.degrees(fire_rot[2]):.1f} vp_yaw={vp_yaw:.1f} "
                        f"src={aim_src} pose_src={pose_src} "
                        f"turn_slot={turn_val:.3f} pos={ctx.player_pos} "
                        f"fire_pos={fire_pos}"
                    )
                    self._log_fire_pose_context(ctx, client_tick, "ACTION_DUMP")
                    self._record_client_weapon_fire(
                        ctx,
                        "ACTION_DUMP",
                        client_tick,
                        new_projectiles,
                        energy_spent,
                    )
                for proj in new_projectiles:
                    self._spawn_moving_projectile(ctx, proj, addr)

            # Legacy packet-arrival hook; fixed-step jumpjets run in motion tick.
            self._process_jump_jets(ctx, addr)
        else:
            ctx.action_dump_decode_fail_count += 1
            ctx.last_action_dump_decode_fail_hex = data[:48].hex()
            if self.debug_sync:
                print(
                    f"[UDP] ACTION_DUMP decode failed c{ctx.client_id}: "
                    f"len={len(data)} data={ctx.last_action_dump_decode_fail_hex}"
                )

    def _handle_action_update(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle ACTION_UPDATE packet (0x0A).
        Contains incremental behavior slot updates.
        """
        if ctx is None:
            return

        if self.debug_sync:
            print(f"[UDP] ACTION_UPDATE received: len={len(data)} data={data.hex()}")
        client_tick = 0
        if len(data) >= 6:
            try:
                client_tick = struct.unpack(">I", data[2:6])[0]
                self._sync_tick_offset(ctx, client_tick)
            except struct.error:
                pass
        if ctx.weapon_system.decode_action_update(data):
            ctx.last_action_dump_time = time.monotonic()
            self._record_client_action_telemetry(ctx, "ACTION_UPDATE", client_tick)
            if ctx.session and not ctx.session.input_ready:
                ctx.session.input_ready = True
                ctx.session.input_ready_time = time.monotonic()
                print(f"[GAME] Client {ctx.client_id}: input ready (ACTION_UPDATE)")
            # Backdating is intentionally disabled. We apply input on packet arrival
            # and rely on deterministic lockstep physics, matching the original model.

            self._update_player_aim(ctx)
            # Yaw is tracked via VIEWPOINT_INFO when available; otherwise input-based fallback is used.
            # Position is simulated in the tick loop from behavior slots.

            # Update weapon system with current pose
            ctx.weapon_system.player_id = ctx.session.player_id or ctx.entity_id
            ctx.weapon_system.player_team = ctx.session.team_id or 2
            fire_pos, fire_rot, aim_src, pose_src = self._select_weapon_fire_pose(ctx, client_tick)
            ctx.weapon_system.player_pos = fire_pos
            ctx.weapon_system.player_rot = fire_rot
            ctx.weapon_system.projectile_aim_source = aim_src
            ctx.weapon_system.use_pitch = bool(getattr(self, "projectile_body_pitch", False))

            # Weapon spawning with wulf-forge encoding
            if self.projectiles_enabled:
                energy_available = ctx.player_energy if self.weapon_energy_enabled else None
                new_projectiles, energy_spent = ctx.weapon_system.update(
                    available_energy=energy_available
                )
                if energy_spent > 0.0:
                    self._consume_player_energy(ctx, energy_spent)
                if new_projectiles:
                    turn_val = ctx.weapon_system.behavior_slots[BehaviorSlot.TURNING]
                    vp_yaw = math.degrees(ctx.player_aim_yaw)
                    print(
                        f"[WEAPON-FIRE] Firing {len(new_projectiles)} proj "
                        f"heading={math.degrees(ctx.player_heading):.1f} "
                        f"aim_yaw={math.degrees(fire_rot[2]):.1f} vp_yaw={vp_yaw:.1f} "
                        f"src={aim_src} pose_src={pose_src} "
                        f"turn_slot={turn_val:.3f} pos={ctx.player_pos} "
                        f"fire_pos={fire_pos}"
                    )
                    self._log_fire_pose_context(ctx, client_tick, "ACTION_UPDATE")
                    self._record_client_weapon_fire(
                        ctx,
                        "ACTION_UPDATE",
                        client_tick,
                        new_projectiles,
                        energy_spent,
                    )
                for proj in new_projectiles:
                    self._spawn_moving_projectile(ctx, proj, addr)

            # Legacy packet-arrival hook; fixed-step jumpjets run in motion tick.
            self._process_jump_jets(ctx, addr)
        else:
            ctx.action_update_decode_fail_count += 1
            ctx.last_action_update_decode_fail_hex = data[:48].hex()
            if self.debug_sync:
                print(
                    f"[UDP] ACTION_UPDATE decode failed c{ctx.client_id}: "
                    f"len={len(data)} data={ctx.last_action_update_decode_fail_hex}"
                )

    def _handle_input_feedback(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle INPUT_FEEDBACK packet (0x40).
        Contains player input state - frame counter and possibly input flags.
        """
        if len(data) >= 5:
            frame_counter = struct.unpack(">I", data[1:5])[0]
            # Count INPUT_FEEDBACK for frame-rate estimation
            if ctx is not None:
                ws = ctx.weapon_system
                ws.on_input_feedback()
                ctx.input_feedback_count += 1
                ctx.last_input_feedback_time = time.monotonic()
            # Additional data might contain input flags
            if len(data) > 5:
                extra = data[5:]
                if extra and extra != b'\x00' * len(extra):
                    print(f"[INPUT] Extra data: {extra.hex()}")
