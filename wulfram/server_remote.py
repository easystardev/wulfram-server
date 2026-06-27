"""RemoteSyncMixin -- remote-entity update sending + remote-input timing,
extracted verbatim from WulframServer (server.py decomposition, step 8).
Method-only mixin; shares state via `self`.
"""
from __future__ import annotations

import time
from typing import Optional

from . import handlers
from .client import ClientContext
from .packets import build_update_array_player_update


class RemoteSyncMixin:
    def _send_remote_player_updates(self, ctx: ClientContext, tick: int, *, prefer_tcp: bool = True) -> None:
        """Send other players' transforms to a client."""
        if not self._og_viewer_replication_enabled(ctx, "remote_updates"):
            return
        mode = self.remote_update_mode
        if mode in ("off", "none", "disabled"):
            return
        include_pos = mode not in ("heartbeat", "mask0")
        include_vel = mode in ("pos_vel", "pos_vel_rot", "full", "all")
        include_rot = mode in ("pos_rot", "pos_vel_rot", "full", "all")
        viewer_fuel = self._get_energy_value(ctx)
        others = self._snapshot_in_game_clients()
        if tick % 300 == 0:
            other_ids = [(c.client_id, c.session.entity_id or c.entity_id) for c in others if c is not ctx]
            print(f"[REMOTE-DBG] client={ctx.client_id} tick={tick} mode={mode} "
                  f"others={other_ids} known={ctx.known_entity_ids}")
        for other in others:
            if other is ctx:
                continue
            entity_id = other.session.entity_id or other.entity_id
            if entity_id not in ctx.known_entity_ids:
                continue
            # Entity vitals (health) for the REMOTE entity, not the viewer.
            # The OG targeting collector (Targeting.c:3744) drops manned
            # vehicles whose health (entity+0xD0) == 0.0, so a remote tank that
            # only gets pos/rot/manned renders but can NEVER be targeted with T.
            # Sending the health vitals delta-bit (mask bit 5, via speed_scale)
            # sets entity+0xD0 = max_health * health on the client, making the
            # remote player targetable. Dead players (health 0) stay untargetable.
            health_val = self._get_health_value(other)
            include_local_state, local_state_kwargs = self._get_update_array_local_state_for_viewer(ctx)
            send_pos = self._to_client_pos(other.player_pos)
            payload = build_update_array_player_update(
                tick,
                entity_id,
                pos=send_pos,
                vel=other.player_vel,
                rot=(
                    other.player_pose.get("roll", 0.0),
                    other.player_pose.get("pitch", 0.0),
                    (-other.player_heading if self.remote_yaw_negate else other.player_heading) + self.remote_yaw_offset,
                ),
                include_pos=include_pos,
                include_vel=include_vel,
                include_rot=include_rot,
                include_spin=include_rot,
                spin=(0.0, 0.0, other.angular_vel_yaw),
                include_local_state=include_local_state,
                include_entity_vitals=getattr(self, "remote_entity_vitals", True),
                speed_scale=health_val,
                is_manned=True,
                **local_state_kwargs,
            )
            ok = self._send_packet_to_client(ctx, payload, prefer_tcp=prefer_tcp)
            if tick % 300 == 0:
                print(f"[REMOTE-DBG] Sent entity={entity_id} -> client={ctx.client_id} "
                      f"pos={send_pos} is_manned=True mode={mode} ok={ok}")

    def _remote_og_movement_input_delay_for_ctx(self, ctx: ClientContext) -> float:
        """Return the empirically observed remote OG input replay delay."""
        if ctx is None or ctx.injected_input is not None:
            return 0.0
        if handlers._is_loopback_client(ctx):
            return 0.0
        return max(0.0, float(getattr(self, "remote_og_movement_input_delay", 0.0) or 0.0))

    def _maybe_promote_remote_full_local_state(self, ctx: ClientContext, *, reason: str) -> bool:
        """Leave the spawn-safe minimal remote path once the client is stably in game."""
        if self.update_local_state_mode != "wf":
            return False
        if handlers._is_loopback_client(ctx):
            return False
        if getattr(ctx, "remote_full_local_state_ready", False):
            return False
        if not ctx.session or not ctx.session.in_game:
            return False
        if reason in ("heartbeat", "action_update", "action_dump"):
            return False
        spawn_time = getattr(ctx.session, "last_spawn_time", 0.0) or 0.0
        if spawn_time > 0.0 and (time.monotonic() - spawn_time) < self.remote_full_local_state_delay:
            return False
        ctx.remote_full_local_state_ready = True
        ctx._spawn_safe_heartbeat_suppressed_logged = False
        print(
            f"[LOCAL-STATE] client={ctx.client_id} promoted to full remote sync "
            f"reason={reason}"
        )
        return True

    def _remote_movement_input_active(self, ctx: ClientContext, *, now: Optional[float] = None) -> bool:
        """Return true while a remote OG client is actively driving.

        FREEZE FIX (2026-06-27): this gates ALL velocity-zeroing corrections (periodic,
        burst, divergence) -- they ride VIEW_UPDATE, which the OG client applies as snap
        pose + ZERO VELOCITY, so firing them mid-drive freezes movement. The OG client
        sends ACTION_UPDATE only on input CHANGE: while a movement key is HELD there are
        no new packets, so the old 0.35s packet-recency window lapsed and reported "not
        moving" -> corrections fired -> persistent freeze (movement only, chat/respawn
        still work). The server keeps applying the held input every tick and
        last_decoded_input reflects the held axes (cleared by the release packet), so we
        key off the HELD state, not packet recency. `_datagram_active_movement_input`
        stays as an instant-on hint.
        """
        if handlers._is_loopback_client(ctx):
            return False
        if bool(getattr(ctx, "_datagram_active_movement_input", False)):
            return True
        decoded = getattr(ctx, "last_decoded_input", None) or {}
        try:
            fwd = float(decoded.get("fwd", 0.0) or 0.0)
            strafe = float(decoded.get("strafe", 0.0) or 0.0)
            turn = float(decoded.get("turn", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        # Held movement axes (forward/strafe/turn) => actively driving, regardless of
        # how long ago the (change-triggered) packet arrived.
        return abs(fwd) > 0.05 or abs(strafe) > 0.05 or abs(turn) > 0.05
