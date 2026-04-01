"""
Transport layer: TCP/UDP socket handling, framing, logging.
Handles packet I/O and trace recording for replay/debugging.
"""

import socket
import struct
import time
import threading
from typing import Optional, Callable, Tuple
from pathlib import Path
from .codec import frame_packet, unframe_packet, format_hex, format_ascii
from .packets import get_packet_name


class PacketLogger:
    """
    Records packet traces for replay testing.
    Format: timestamp, direction, packet_type, length, hex_dump
    """

    def __init__(self, log_dir: str = "traces"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = int(time.time())
        self.trace_file = self.log_dir / f"trace_{self.session_id}.log"
        self.lock = threading.Lock()

    def log_packet(self, direction: str, packet_type: int, data: bytes, protocol: str = "TCP"):
        """Log a packet to the trace file."""
        with self.lock:
            timestamp = time.monotonic()
            pkt_name = get_packet_name(packet_type)
            hex_dump = data.hex()

            line = f"{timestamp:.6f}|{direction}|{protocol}|{pkt_name}|0x{packet_type:02X}|{len(data)}|{hex_dump}\n"

            with open(self.trace_file, "a") as f:
                f.write(line)

    def log_send(self, packet_type: int, data: bytes, protocol: str = "TCP"):
        self.log_packet("SEND", packet_type, data, protocol)

    def log_recv(self, packet_type: int, data: bytes, protocol: str = "TCP"):
        self.log_packet("RECV", packet_type, data, protocol)


class TCPHandler:
    """
    Handles TCP packet framing and I/O.
    """

    def __init__(self, sock: socket.socket, logger: Optional[PacketLogger] = None):
        self.sock = sock
        self.logger = logger
        self.recv_buffer = b""

    # Packet types that can crash OG client when sent over TCP with entity data.
    # 0x0E (UPDATE_ARRAY): complex bitstreams corrupt g_render_context
    # 0x0D (TRANSIENT_ARRAY): raw format ≠ OG quantized bitstream → stream desync
    # Small/empty 0x0E packets (priming, spawn points) are safe — only warn.
    TCP_WARN_PACKETS = {0x0D, 0x0E}

    def send(self, payload: bytes, log: bool = True):
        """Send a framed packet."""
        if len(payload) == 0:
            return

        packet_type = payload[0]
        if packet_type in self.TCP_WARN_PACKETS:
            pkt_name = get_packet_name(packet_type)
            print(f"[TCP-WARN] Sending {pkt_name} (0x{packet_type:02X}) over TCP "
                  f"({len(payload)}B) — may crash OG client. "
                  f"First 16B: {payload[:16].hex()}")
        framed = frame_packet(payload)

        try:
            self.sock.sendall(framed)

            if log:
                pkt_name = get_packet_name(packet_type)
                print(f"[SEND] {pkt_name:12} (0x{packet_type:02X}) | Len={len(framed):<4}")

            if self.logger:
                self.logger.log_send(packet_type, framed)

        except Exception as e:
            print(f"[SEND] Error: {e}")
            raise

    def recv(self) -> Optional[bytes]:
        """
        Receive one complete packet.
        Returns packet body (without length prefix) or None if disconnected.
        """
        while True:
            # Try to extract a packet from buffer
            packet, self.recv_buffer = unframe_packet(self.recv_buffer)
            if packet is not None:
                if len(packet) > 0:
                    packet_type = packet[0]
                    if self.logger:
                        self.logger.log_recv(packet_type, packet)
                return packet

            # Need more data
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    return None  # Disconnected
                self.recv_buffer += chunk
            except socket.timeout:
                return None
            except ConnectionResetError:
                return None


class UDPHandler:
    """
    Handles UDP packet I/O.
    """

    def __init__(self, sock: socket.socket, logger: Optional[PacketLogger] = None):
        self.sock = sock
        self.logger = logger
        self.client_addr: Optional[Tuple[str, int]] = None

    def bind(self, addr: Tuple[str, int]):
        """Bind to address."""
        self.sock.bind(addr)

    def send_to(self, data: bytes, addr: Tuple[str, int], log: bool = True):
        """Send UDP packet to address."""
        if len(data) == 0:
            return

        try:
            self.sock.sendto(data, addr)

            if log and len(data) > 0:
                packet_type = data[0]
                pkt_name = get_packet_name(packet_type)
                print(f"[UDP SEND] {pkt_name} to {addr}")

            if self.logger and len(data) > 0:
                self.logger.log_send(data[0], data, "UDP")

        except Exception as e:
            print(f"[UDP SEND] Error: {e}")

    def recv_from(self, bufsize: int = 4096) -> Tuple[Optional[bytes], Optional[Tuple[str, int]]]:
        """
        Receive UDP packet.
        Returns (data, addr) or (None, None) on error.
        """
        try:
            data, addr = self.sock.recvfrom(bufsize)

            if len(data) > 0:
                packet_type = data[0]
                if self.logger:
                    self.logger.log_recv(packet_type, data, "UDP")

            return data, addr

        except socket.timeout:
            return None, None
        except OSError as e:
            # WinError 10054 is expected when a UDP peer disappears; treat as no-data.
            if getattr(e, "winerror", None) == 10054:
                return None, None
            print(f"[UDP RECV] Error: {e}")
            return None, None
        except Exception as e:
            print(f"[UDP RECV] Error: {e}")
            return None, None


def print_packet(direction: str, packet_type: int, data: bytes, show_hex: bool = True, show_ascii: bool = True):
    """Pretty-print a packet for debugging."""
    pkt_name = get_packet_name(packet_type)
    print(f"[{direction}] {pkt_name:12} (0x{packet_type:02X}) | Len={len(data):<4}", end="")

    if show_hex:
        print(f" | {format_hex(data, 60)}", end="")

    print()

    if show_ascii and len(data) > 0:
        print(f"       Ascii='{format_ascii(data)}'")
        print("-" * 60)
