"""Dock pin: upstream's at-rest local-tank VIEW_UPDATE stream (2026-10-16 capture analysis,
D:\\work\\upstream-1016\\sync\\RESULT.md).

Upstream behaviour, measured on wulfram3.com:
- At touchdown (the same tick a DOCKING stream starts, ~0.55-1.1 s after thrust release) the
  server latches ONE local-tank pose and re-sends it every tick with UPDATE mask bit 9 set and
  velocity/spin zero.
- The burst ends at the first forward/strafe input, with one final velocity/spin-zero record
  and a DOCKING with docked=0.
- Upstream latches the pose the CLIENT uploaded over its relay, so the pin never moves the tank.
  We have no pose uplink from a stock client, so our latched pose is the SERVER's pose: on an
  OG wulfram2.exe a delta under ~2.3 u / 0.5 rad is ignored, above it the client hard-snaps
  with a physics reset (the 2026-06 freeze/NaN territory). Hence opt-in, allowlist only.

The server-side wiring (latching, emitting, DOCKING) calls should_pin() once per tick per
client; this module only holds the policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DockPinInputs:
    """Everything the policy may look at for one client on one tick."""

    enabled_for_client: bool      # WULFRAM_DOCK_PIN on AND client on the allowlist
    grounded: bool                # server physics says the tank is on the ground / a pad
    speed: float                  # server horizontal speed, u/s
    seconds_since_thrust: float   # time since the last forward/strafe/jump input was released
    move_input_active: bool       # forward/strafe/jump held this tick
    pinned: bool                  # a pin burst is already running for this client
    divergence_pos: float         # |latched pin pose - current server pose|, units
    is_native_client: bool        # window-0 / sequence-tick client (snaps to every record)
    # Added by the wiring: the DOCKING target this pin would report -- 0 for a ground dock,
    # else the friendly FUEL/REPAIR pad entity id the grounded tank is on (09-18 upstream
    # pinned OG spawns on fuel pads, target 134403). Lets the policy treat pad docks and
    # ground docks differently. Defaulted so existing constructions stay valid.
    dock_target: int = 0


# Upstream's touchdown lag: the burst starts ~0.55-1.1 s after thrust release.
TOUCHDOWN_DELAY_S = 0.55
# How far the server pose may drift from the latched pin before the burst stops.
# OG: stay under its ~2.3 u ignore band, so a pin is never a hard snap (upstream's pin was
# a no-op on OG too). Native: it snaps to every record, so keep the drift tiny.
MAX_DIVERGENCE_OG = 2.0
MAX_DIVERGENCE_NATIVE = 0.25
# Starting a pin on a native client latches the server pose it will snap to; refuse while
# the tank is still sliding fast enough for that to be visible. OG gets upstream's no-limit.
MAX_START_SPEED_NATIVE = 2.0


def should_pin(s: DockPinInputs) -> bool:
    """Return True to send (or keep sending) the pinned pose this tick, False to end or skip.

    Upstream-faithful start/stop (touchdown after the thrust-release lag; ends only on move
    input), plus two guards because our latched pose is the server's, not the client's:
    a divergence cap that ends the burst before it becomes a correction, and a start-speed
    limit for native clients, which snap to every record.
    """
    if not s.enabled_for_client or s.move_input_active:
        return False
    native = s.is_native_client
    if s.pinned:
        cap = MAX_DIVERGENCE_NATIVE if native else MAX_DIVERGENCE_OG
        return s.divergence_pos <= cap
    if not s.grounded or s.seconds_since_thrust < TOUCHDOWN_DELAY_S:
        return False
    return not native or s.speed <= MAX_START_SPEED_NATIVE
