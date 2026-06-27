"""Match flow: running game clock + round end / reset (Phase 3 slice 4).

The OG client drives its HUD round timer off GAME_CLOCK (0x2f): the server sends a
countdown duration and the client counts it down locally (Handlers.c:1121 ->
Countdown_start). Re-sending periodically keeps clients in sync. On expiry the server
announces the result, broadcasts RESET_GAME (0x3f, clears the client warp/map grid),
and starts a fresh round with a new GAME_CLOCK.

CRITICAL wire detail: GAME_CLOCK's active flag is INVERTED -- a running clock sends
active_flag=0 (the client sets g_game_clock_active = (flag == 0)). Handled in
packets.build_game_clock.
"""

from __future__ import annotations

import time
from typing import Any

from .packets import build_chat_message, build_game_clock, build_reset_game

_TEAM_NAMES = {1: "Red", 2: "Blue"}


def _now() -> float:
    return time.monotonic()


def init_match_state(server: object) -> None:
    """Initialize match-flow runtime state (idempotent; safe to call at startup)."""
    server._match_round_start = _now()
    server._match_phase = 1
    server._match_last_clock_broadcast = 0.0


def remaining_ms(server: object) -> int:
    """Milliseconds left in the current round (clamped at 0)."""
    start = float(getattr(server, "_match_round_start", None) or _now())
    elapsed = _now() - start
    remaining = max(0.0, float(getattr(server, "match_round_duration_s", 600.0)) - elapsed)
    return int(remaining * 1000)


def _broadcast(server: object, pkt: bytes) -> int:
    sent = 0
    for client in server._snapshot_in_game_clients():
        try:
            server._send_packet_to_client(client, pkt, prefer_tcp=True, allow_udp_fallback=False)
            sent += 1
        except Exception:  # noqa: BLE001 - one bad client must not break the round
            pass
    return sent


def broadcast_game_clock(server: object, *, running: bool = True) -> int:
    """Send GAME_CLOCK (0x2f) to all in-game clients with the current remaining time."""
    pkt = build_game_clock(
        running=running,
        round_time_ms=remaining_ms(server),
        phase=int(getattr(server, "_match_phase", 1) or 1),
    )
    return _broadcast(server, pkt)


def _team_scores(server: object) -> dict[int, int]:
    """Total kills per team across in-game clients -- the simple time-round win metric."""
    scores: dict[int, int] = {}
    for c in server._snapshot_in_game_clients():
        team = int(getattr(getattr(c, "session", None), "team_id", 0) or 0)
        scores[team] = scores.get(team, 0) + int(getattr(c, "kills", 0) or 0)
    return scores


def winner_message(server: object) -> str:
    """End-of-round announcement: the team with the most kills wins (ties = draw)."""
    scores = {t: s for t, s in _team_scores(server).items() if t}
    if not scores:
        return "Round over!"
    best = max(scores.values())
    leaders = [t for t, s in scores.items() if s == best]
    if best == 0 or len(leaders) != 1:
        return "Round over -- a draw!"
    team = leaders[0]
    return f"Round over -- {_TEAM_NAMES.get(team, f'Team {team}')} wins with {best} kills!"


def end_round(server: object) -> str:
    """Announce the result, RESET_GAME, and start a fresh round. Returns the message."""
    msg = winner_message(server)
    _broadcast(server, build_chat_message(msg, source_id=0))
    _broadcast(server, build_reset_game())
    server._match_round_start = _now()
    server._match_phase = int(getattr(server, "_match_phase", 1) or 1) + 1
    server._match_last_clock_broadcast = 0.0
    broadcast_game_clock(server, running=True)
    print(f"[MATCH] {msg} -> reset; new round (phase {server._match_phase})")
    return msg


def update_match_flow(server: object) -> None:
    """Per-tick (self-throttled): re-sync the clock, and end the round on expiry."""
    if not getattr(server, "match_flow_enabled", False):
        return
    if getattr(server, "_match_round_start", None) is None:
        init_match_state(server)
    if not server._snapshot_in_game_clients():
        return  # no players -> idle (don't burn rounds on an empty server)
    if remaining_ms(server) <= 0:
        end_round(server)
        return
    now = _now()
    interval = float(getattr(server, "match_clock_interval_s", 5.0) or 5.0)
    if now - float(getattr(server, "_match_last_clock_broadcast", 0.0) or 0.0) >= interval:
        server._match_last_clock_broadcast = now
        broadcast_game_clock(server, running=True)
