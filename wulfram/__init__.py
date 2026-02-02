# Wulfram2 Server Package
"""
Layered architecture for Wulfram2 server emulator:
- codec: BitWriter/BitReader, packet encode/decode
- packets: Packet type definitions and builders
- session: Per-connection state machine
- game: Entities, teams, replication
- transport: TCP/UDP socket handling
"""

__version__ = "0.1.0"
