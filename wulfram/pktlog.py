"""
Packet traffic logger for debugging client freezes.

Records every packet sent to clients with timestamps, tick values,
entity data, and thread info. Designed to capture the exact packet
sequence that precedes a client freeze.

Usage via control command:
  pktlog on          - Start logging
  pktlog off         - Stop logging
  pktlog dump [N]    - Show last N entries (default 50)
  pktlog clear       - Clear buffer
  pktlog save [path] - Save to file
  pktlog analyze     - Show timing analysis
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PacketEntry:
    """Single logged packet."""
    timestamp: float          # time.monotonic()
    wall_time: float          # time.time() for human-readable
    client_id: int
    thread_name: str
    label: str                # e.g. "CORRECTION(dual_entity)", "HEARTBEAT", "PROJ_SPAWN"
    tick: int
    payload_size: int
    transport: str            # "TCP", "UDP", "TCP+UDP"
    entity_count: int = 0
    entity_ids: tuple = ()
    mask_bits: tuple = ()     # per-entity mask values
    has_local_state: bool = False
    health: float = -1.0      # -1 = not included
    extra: str = ""           # free-form notes

    def format_short(self, t0: float = 0.0) -> str:
        """One-line summary."""
        dt = self.timestamp - t0 if t0 else self.timestamp
        ents = ",".join(f"0x{e:X}" for e in self.entity_ids)
        masks = ",".join(f"0b{m:010b}" for m in self.mask_bits)
        ls = f" hp={self.health:.2f}" if self.has_local_state else ""
        return (
            f"{dt:8.3f}s c{self.client_id} [{self.thread_name:12s}] "
            f"{self.label:30s} tick={self.tick:6d} {self.payload_size:4d}B "
            f"{self.transport:7s} ent=[{ents}] mask=[{masks}]{ls}"
            f"{' ' + self.extra if self.extra else ''}"
        )


class PacketLog:
    """Thread-safe circular buffer of packet entries."""

    def __init__(self, max_entries: int = 2000):
        self.enabled = False
        self._lock = threading.Lock()
        self._entries: deque[PacketEntry] = deque(maxlen=max_entries)
        self._t0: float = 0.0  # reference time for relative timestamps
        self._counts: dict = {}  # label -> count

    def start(self):
        with self._lock:
            self.enabled = True
            self._t0 = time.monotonic()
            self._counts.clear()
            print("[PKTLOG] Logging started")

    def stop(self):
        with self._lock:
            self.enabled = False
            print(f"[PKTLOG] Logging stopped ({len(self._entries)} entries)")

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._counts.clear()
            self._t0 = time.monotonic()

    def log(
        self,
        client_id: int,
        label: str,
        tick: int,
        payload: bytes,
        transport: str,
        entity_count: int = 0,
        entity_ids: tuple = (),
        mask_bits: tuple = (),
        has_local_state: bool = False,
        health: float = -1.0,
        extra: str = "",
    ):
        if not self.enabled:
            return
        entry = PacketEntry(
            timestamp=time.monotonic(),
            wall_time=time.time(),
            client_id=client_id,
            thread_name=threading.current_thread().name[:12],
            label=label,
            tick=tick,
            payload_size=len(payload) if payload else 0,
            transport=transport,
            entity_count=entity_count,
            entity_ids=tuple(entity_ids),
            mask_bits=tuple(mask_bits),
            has_local_state=has_local_state,
            health=health,
            extra=extra,
        )
        with self._lock:
            self._entries.append(entry)
            self._counts[label] = self._counts.get(label, 0) + 1

    def dump(self, n: int = 50) -> str:
        """Return last N entries as formatted text."""
        with self._lock:
            entries = list(self._entries)[-n:]
            t0 = self._t0
        if not entries:
            return "No packets logged."
        lines = [f"=== Packet Log (last {len(entries)} of {len(self._entries)} total) ==="]
        lines.append(
            f"{'Time':>8s} {'Client':>6s} {'Thread':>12s} "
            f"{'Label':>30s} {'Tick':>6s} {'Size':>5s} "
            f"{'Trans':>7s} Entities"
        )
        lines.append("-" * 120)
        for e in entries:
            lines.append(e.format_short(t0))
        return "\n".join(lines)

    def save(self, path: str = "pktlog.txt") -> str:
        """Save full log to file."""
        with self._lock:
            entries = list(self._entries)
            t0 = self._t0
        with open(path, "w") as f:
            for e in entries:
                f.write(e.format_short(t0) + "\n")
        return f"Saved {len(entries)} entries to {path}"

    def analyze(self) -> str:
        """Timing and frequency analysis."""
        with self._lock:
            entries = list(self._entries)
            counts = dict(self._counts)
            t0 = self._t0
        if not entries:
            return "No packets to analyze."

        lines = ["=== Packet Traffic Analysis ==="]
        duration = entries[-1].timestamp - entries[0].timestamp if len(entries) > 1 else 0
        lines.append(f"Duration: {duration:.1f}s  Packets: {len(entries)}  Rate: {len(entries)/max(duration,0.001):.1f}/s")
        lines.append("")

        # Per-label counts and rates
        lines.append("By type:")
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            rate = count / max(duration, 0.001)
            lines.append(f"  {label:35s} {count:5d} ({rate:.1f}/s)")
        lines.append("")

        # Per-client breakdown
        client_entries: dict[int, list] = {}
        for e in entries:
            client_entries.setdefault(e.client_id, []).append(e)

        for cid, ces in sorted(client_entries.items()):
            lines.append(f"Client {cid}: {len(ces)} packets")
            # Check for tick ordering issues
            last_tick = -1
            tick_reversals = 0
            tick_gaps = []
            for e in ces:
                if e.tick < last_tick:
                    tick_reversals += 1
                elif e.tick > last_tick + 1 and last_tick >= 0:
                    tick_gaps.append((last_tick, e.tick, e.label))
                last_tick = e.tick

            if tick_reversals:
                lines.append(f"  WARNING: {tick_reversals} tick reversals!")
            if tick_gaps:
                lines.append(f"  Tick gaps: {len(tick_gaps)}")
                for old, new, lbl in tick_gaps[:5]:
                    lines.append(f"    {old} -> {new} (gap={new-old}) at {lbl}")

            # Thread interleaving
            threads = set(e.thread_name for e in ces)
            if len(threads) > 1:
                lines.append(f"  Threads: {', '.join(sorted(threads))}")
                # Find interleaving points
                interleaves = 0
                prev_thread = ces[0].thread_name if ces else ""
                for e in ces[1:]:
                    if e.thread_name != prev_thread:
                        interleaves += 1
                    prev_thread = e.thread_name
                lines.append(f"  Thread switches: {interleaves}")

            # Duplicate ticks (same tick from different threads = race condition)
            tick_sources: dict[int, list] = {}
            for e in ces:
                tick_sources.setdefault(e.tick, []).append(e)
            dup_ticks = {t: es for t, es in tick_sources.items() if len(es) > 1}
            if dup_ticks:
                lines.append(f"  Duplicate ticks: {len(dup_ticks)}")
                for t, es in sorted(dup_ticks.items())[:5]:
                    labels = ", ".join(f"{e.label}({e.thread_name})" for e in es)
                    lines.append(f"    tick={t}: {labels}")

            # Rapid bursts (<5ms between packets)
            bursts = 0
            for i in range(1, len(ces)):
                gap = ces[i].timestamp - ces[i-1].timestamp
                if gap < 0.005:
                    bursts += 1
            if bursts:
                lines.append(f"  Rapid bursts (<5ms apart): {bursts}")

            # Timing gaps (>100ms without a packet)
            big_gaps = []
            for i in range(1, len(ces)):
                gap = ces[i].timestamp - ces[i-1].timestamp
                if gap > 0.1:
                    big_gaps.append((gap, ces[i-1].label, ces[i].label))
            if big_gaps:
                lines.append(f"  Gaps >100ms: {len(big_gaps)}")
                for gap, before, after in sorted(big_gaps, reverse=True)[:3]:
                    lines.append(f"    {gap*1000:.0f}ms: {before} -> {after}")

            # Last N packets before any big gap (potential freeze point)
            if big_gaps:
                biggest_gap = max(big_gaps, key=lambda x: x[0])
                lines.append(f"\n  === Packets around largest gap ({biggest_gap[0]*1000:.0f}ms) ===")
                gap_time = None
                for i in range(1, len(ces)):
                    gap = ces[i].timestamp - ces[i-1].timestamp
                    if abs(gap - biggest_gap[0]) < 0.001:
                        gap_time = ces[i-1].timestamp
                        break
                if gap_time:
                    pre = [e for e in ces if gap_time - 2.0 <= e.timestamp <= gap_time]
                    post = [e for e in ces if gap_time < e.timestamp <= gap_time + biggest_gap[0] + 0.5]
                    for e in pre[-10:]:
                        lines.append(f"    {e.format_short(t0)}")
                    lines.append(f"    --- GAP {biggest_gap[0]*1000:.0f}ms ---")
                    for e in post[:5]:
                        lines.append(f"    {e.format_short(t0)}")

        return "\n".join(lines)
