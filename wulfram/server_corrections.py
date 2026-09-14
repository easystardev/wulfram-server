"""CorrectionMixin -- state-request handling, correction-burst gating,
state-sync snapshot, and debug-sync, extracted verbatim from WulframServer
(server.py decomposition, step 6). Method-only mixin; shares state via `self`.
"""
from __future__ import annotations
from .observer_lifecycle import serialized

import ipaddress
import math
import struct
import time
from typing import Optional

from . import handlers
from .client import ClientContext
from .packets import build_update_array_player_update, build_view_update_player_update


from .weapons import EntityType
from .packets import build_update_array_multi, build_view_update_create_tank, build_view_update_multi


class CorrectionMixin:
    def _queue_settled_movement_correction_tail(self, ctx: ClientContext) -> None:
        """Queue the remaining packets in one reliable stopped reconcile burst."""
        count = max(
            1,
            int(getattr(self, "settled_movement_correction_burst_count", 3) or 1),
        )
        if count <= 1:
            return
        remaining = int(getattr(ctx, "correction_burst_remaining", 0) or 0)
        ctx.correction_burst_remaining = max(remaining, count - 1)
        ctx.correction_burst_interval_s = max(
            0.01,
            float(
                getattr(self, "settled_movement_correction_burst_interval", 0.1)
                or 0.1
            ),
        )

    def _settled_movement_correction_due(
        self,
        ctx: ClientContext,
        *,
        now: float,
        movement_suppressed: bool,
    ) -> bool:
        """Return true once for each movement episode after input has settled.

        The OG does not report its predicted position, so the divergence gate
        cannot see lateral drift.  A single stopped VIEW_UPDATE is safe on the
        crashfix14 client and converges that otherwise-unobservable error.  The
        active-input suppression window is load-bearing: never reset physics
        while the client is still driving.
        """
        if not getattr(self, "settled_movement_correction_enabled", False):
            return False
        if movement_suppressed:
            return False
        moved_at = float(getattr(ctx, "last_nonzero_move_input_time", 0.0) or 0.0)
        consumed_at = float(
            getattr(ctx, "last_settled_movement_correction_input_time", 0.0) or 0.0
        )
        if moved_at <= 0.0 or moved_at <= consumed_at:
            return False
        max_speed = max(
            0.0,
            float(getattr(self, "settled_movement_correction_max_speed", 0.5) or 0.0),
        )
        vx, vy, vz = getattr(ctx, "player_vel", (0.0, 0.0, 0.0))
        linear_speed = math.sqrt(float(vx) ** 2 + float(vy) ** 2 + float(vz) ** 2)
        if not math.isfinite(linear_speed) or linear_speed > max_speed:
            return False
        max_angular_deg = max(
            0.0,
            float(
                getattr(
                    self,
                    "settled_movement_correction_max_angular_speed_deg",
                    1.0,
                )
                or 0.0
            ),
        )
        angular_speed = abs(
            float(
                getattr(getattr(ctx, "vehicle_physics", None), "angular_velocity", 0.0)
                or 0.0
            )
        )
        if not math.isfinite(angular_speed) or math.degrees(angular_speed) > max_angular_deg:
            return False
        min_interval = max(0.0, float(getattr(self, "correction_min_interval", 0.0) or 0.0))
        last_sent = float(getattr(ctx, "last_correction_send", 0.0) or 0.0)
        return (now - last_sent) >= min_interval

    @serialized
    def _handle_state_request(self, ctx: Optional[ClientContext], data: bytes, addr: tuple):
        """
        Handle STATE_REQUEST (0x0C) - may contain state/position info.
        Decompile-shaped client payload:
        [opcode:1] [request_id:4] [frame_count:4]
        """
        if ctx is None or len(data) < 5:
            return

        # Decode fields
        request_id = struct.unpack(">I", data[1:5])[0] if len(data) >= 5 else 0
        frame_count = struct.unpack(">I", data[5:9])[0] if len(data) >= 9 else 0
        now = time.monotonic()
        ctx.state_request_count += 1
        ctx.last_state_request_time = now
        ctx.last_state_request_id = request_id
        ctx.last_state_request_frame_count = frame_count
        ctx.last_state_request_len = len(data)

        # Log occasionally to avoid spam
        if request_id % 1000 == 0:
            print(
                f"[0x0C] STATE_REQUEST request_id={request_id} "
                f"frame_count={frame_count} len={len(data)}"
            )

        if not self._state_sync_reply_allowed_for_client(ctx):
            return

        # Remote OG clients still use STATE_REQUEST as the gate out of the
        # fragile spawn/bootstrap window, but the targeted reply itself must
        # stay on the short-form-safe local-state prefix. Promotion here only
        # unlocks the broader post-spawn sync path; it does not justify the
        # expanded ammo/turret local-state on the correction packet itself.
        self._maybe_promote_remote_full_local_state(ctx, reason="state_request")

        if getattr(self, "state_request_replay_reply", False):
            # Stock-protocol path (2026-09-02): one immediate VIEW_UPDATE reply
            # is the client's designed local-tank rubberband channel. See
            # _send_stock_replay_reply for the decompile-backed rationale.
            self._send_stock_replay_reply(ctx, request_id=request_id, now=now)
            return

        if self._remote_movement_input_active(ctx, now=now):
            return

        # GOAL 2 (amended 2026-06-09): under the gated-rare correction model the
        # STATE_REQUEST reply must not flood — on fire the OG client spams
        # STATE_REQUEST, which previously froze a zero-latency loopback client.
        # But the tick-loop divergence gate is fed ONLY by the terrain-Z clamp
        # (structurally zero on flat ground, blind to lateral drift) and
        # STATE_REQUEST carries no client position, so with NO trigger lateral
        # divergence under motion grew unbounded (60u+ observed live 2026-06-09).
        # Middle ground: reply with the plain UPDATE_ARRAY snapshot, and queue
        # the standard ~10Hz settle burst at most once per
        # state_request_burst_min_interval — the queue-time rate cap defuses the
        # flood while restoring a correction path for lateral drift. Legacy
        # uncapped proactive replies remain available via WULFRAM_CORRECTION_GATE=0.
        if self.correction_gate_enabled:
            include_view_update = False
        else:
            include_view_update = self._should_send_state_sync_view_update(ctx)
        self._send_state_sync_snapshot(
            ctx,
            reason="state_request",
            include_view_update=include_view_update,
            replay_timestamp=request_id if request_id else None,
        )
        if include_view_update:
            self._queue_state_sync_correction_burst(ctx)
        elif getattr(self, "state_request_burst_enabled", False):
            self._maybe_queue_state_request_burst(ctx, now=now)

    @serialized
    def _send_stock_replay_reply(
        self, ctx: ClientContext, *, request_id: int, now: Optional[float] = None
    ) -> bool:
        """Answer a STATE_REQUEST with the stock client's local-correction packet.

        Decompile-backed contract (azurefishy-src Game/Session/Network):

        * ``UpdateArray_process_payload`` only fills ``interp_record+0x08`` in
          replay mode (VIEW_UPDATE 0x0F).  ``NetworkSync_receive_entity_update``
          reads exactly that slot for the CAMERA entity, so a plain
          UPDATE_ARRAY can never be a deterministic local-player correction.
        * ``PacketHandler_VIEW_UPDATE`` (0x46cc20) clamps a future timestamp to
          the client's current tick; ``UpdateArray_check_eligible`` then rejects
          the record when ``now - ts`` exceeds the client's smoothed latency by
          more than 1.25 sigma.  With the client's latency statistics at zero
          (live probe 2026-09-02) only ``ts >= now`` survives, hence the
          fresh-ahead timestamp from ``_fresh_remote_view_update_timestamp``.
        * ``receive_entity_update`` dead-reckons the target as
          ``pos + vel * dt`` reading the record's velocity whenever a position
          is present -- so position MUST travel with velocity or the target is
          built from uninitialised heap (the NaN sync target seen on VM2).
        * The pull itself is ``EntityInterp_apply_delta``: each frame the tank
          moves ``frame_ms * 0.0025 * remaining`` toward the target, applied
          on top of local prediction in ``Entity_apply_server_correction``
          (every physics substep), with ``Camera_clamp_network_delta``
          discarding sub-2.3u / sub-0.5rad differences and
          ``EntityPrediction_server_reconcile`` rolling back into-wall snaps.

        One packet per request; the client's request cadence is the rate.
        """
        if not self.udp_handler or not ctx.session or not ctx.session.udp_addr:
            return False
        if not ctx.running or ctx.observer_transition or not ctx.session.in_game or ctx.session.entity_id == 0:
            return False
        if now is None:
            now = time.monotonic()
        min_interval = float(getattr(self, "state_request_replay_min_interval", 0.05) or 0.0)
        if (now - ctx.last_state_sync_send) < min_interval:
            return False
        if not getattr(self, "state_request_replay_during_movement", True):
            if self._remote_movement_input_active(ctx, now=now):
                return False
        if getattr(ctx, "_spawn_settle_suppress_until", 0.0) and now < ctx._spawn_settle_suppress_until:
            return False
        entity_id = ctx.session.entity_id
        tick = self._get_network_tick(ctx)
        vel = tuple(float(v) for v in getattr(ctx, "player_vel", (0.0, 0.0, 0.0)))
        nudge = float(getattr(self, "state_request_replay_vel_nudge", 0.0) or 0.0)
        if nudge:
            # Prediction.c LocalPlayer_validate_prediction: when the client's
            # predicted velocity/rotation EXACTLY equal the server's (true at
            # rest, since receive_entity_update writes entity.vel = server vel)
            # it clears in_progress and removes the tank from the interp apply
            # list, aborting the smooth pull; the reconcile snap only fires when
            # the raycast to the server position HITS something. A sub-quantum
            # vertical velocity nudge keeps the verify failing so a resting
            # residual is still pulled out. Off by default (stock-faithful).
            vel = (vel[0], vel[1], vel[2] + nudge)
        lead = float(getattr(self, "state_request_replay_lead_ticks", 0.0) or 0.0)
        rot = self._local_player_sync_rotation(ctx)
        if lead > 0.0:
            # Optional forward extrapolation (A/B). Live 2026-09-02: with the
            # server already ~1 step ahead of the client's predictor mid-drive,
            # a 1-tick lead made the pull overshoot (client ended ~7u ahead).
            dt_lead = lead / max(1.0, float(getattr(self, "tick_rate_hz", 30.0)))
            px, py, pz = ctx.player_pos
            send_pos = self._to_client_pos((px + vel[0] * dt_lead, py + vel[1] * dt_lead, pz + vel[2] * dt_lead))
            ang = float(getattr(ctx, "angular_vel_yaw", 0.0) or 0.0)
            rot = (rot[0], rot[1], rot[2] + ang * dt_lead)
        else:
            send_pos = self._to_client_pos(ctx.player_pos)
        if not all(math.isfinite(v) for v in (*send_pos, *rot, *vel)):
            return False
        if handlers._is_loopback_client(ctx):
            view_timestamp = int(request_id or 0) & 0xFFFFFFFF
        else:
            view_timestamp = self._fresh_remote_view_update_timestamp(ctx, tick)
        include_local_state, ls = self._get_update_array_local_state_for_viewer(ctx)
        if include_local_state:
            ammo_bits, ammo_mask = ls["ammo_count_bits"], ls["ammo_count"]
            pt_bits, pt_angle = ls["primary_turret_bits"], ls["primary_turret_angle"]
            st_bits, st_angle = ls["secondary_turret_bits"], ls["secondary_turret_angle"]
        else:
            ammo_bits = ammo_mask = pt_bits = st_bits = 0
            pt_angle = st_angle = 0.0
        payload = build_view_update_player_update(
            tick=tick,
            entity_id=entity_id,
            pos=send_pos,
            vel=vel,
            rot=rot,
            include_pos=True,
            include_vel=True,
            include_rot=True,
            include_local_state=include_local_state,
            include_entity_vitals=self.view_update_entity_vitals,
            weapon_id=ls.get("weapon_id", self._get_spawn_tank_weapon_type(ctx)),
            health=self._get_health_value(ctx),
            fuel=self._get_energy_value(ctx),
            ammo_count_bits=ammo_bits,
            ammo_count=ammo_mask,
            primary_turret_bits=pt_bits,
            primary_turret_angle=pt_angle,
            secondary_turret_bits=st_bits,
            secondary_turret_angle=st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
            is_manned=True,
            speed_scale=1.0,
            timestamp=view_timestamp,
        )
        self.udp_handler.send_to(payload, ctx.session.udp_addr)
        ctx.last_state_sync_send = now
        ctx.state_sync_reply_count += 1
        ctx.last_state_sync_reply_time = now
        ctx.last_state_sync_reply_tick = tick
        ctx.last_state_sync_replay_timestamp = int(request_id or 0)
        ctx.last_state_sync_snapshot_source = "stock_replay"
        ctx.correction_send_count = int(getattr(ctx, "correction_send_count", 0) or 0) + 1
        ctx.last_correction_send = now
        if self.debug_sync:
            print(
                f"[STATE-SYNC] stock replay reply c{ctx.client_id} req={request_id} "
                f"ts={view_timestamp} tick={tick} pos=({send_pos[0]:.2f},{send_pos[1]:.2f},{send_pos[2]:.2f}) "
                f"vel=({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}) yaw={rot[2]:.4f}"
            )
        return True

    def _queue_state_sync_correction_burst(self, ctx: ClientContext) -> bool:
        """Queue enough replay updates for OG's local correction to visibly settle."""
        if handlers._is_loopback_client(ctx):
            return False
        if self._remote_movement_input_active(ctx):
            return False
        count = int(getattr(self, "state_sync_correction_burst_count", 0) or 0)
        if count <= 0:
            return False
        interval = float(getattr(self, "state_sync_correction_burst_interval", 0.10) or 0.0)
        if interval < 0.0:
            interval = 0.0
        current_remaining = int(getattr(ctx, "correction_burst_remaining", 0) or 0)
        ctx.correction_burst_remaining = max(current_remaining, count)
        ctx.correction_burst_interval_s = interval
        return True

    def _maybe_queue_state_request_burst(
        self, ctx: ClientContext, now: Optional[float] = None
    ) -> bool:
        """Correction-trigger fix (2026-06-09): queue the settle burst in reply
        to STATE_REQUEST even under the correction gate, rate-capped per client.

        The gate's divergence accumulator only sees server-side terrain-Z
        clamping (lateral divergence on flat ground never trips it) and
        STATE_REQUEST carries no position for the server to measure against,
        so the client's own request cadence is the only divergence-shaped
        signal on the wire. The queue-time rate cap (default 1.75s) is what
        keeps the GOAL-2 fire-spam flood dead — flood safety lives HERE, not
        at drain time. Identical for every client; no client-type fork."""
        if now is None:
            now = time.monotonic()
        last = float(getattr(ctx, "last_state_request_burst_queue", 0.0) or 0.0)
        min_interval = float(
            getattr(self, "state_request_burst_min_interval", 1.75) or 0.0
        )
        if (now - last) < min_interval:
            return False
        if not self._queue_state_sync_correction_burst(ctx):
            return False
        ctx.last_state_request_burst_queue = now
        return True

    def _correction_burst_due(
        self, ctx: ClientContext, now: float, movement_suppressed: bool
    ) -> bool:
        """A queued correction burst is due to emit its next packet.

        Used by BOTH the gated and legacy tick branches (the gated branch
        ignoring queued bursts was the 2026-06-09 'correction now only sends
        one packet' bug): bursts drain at their own interval (~10Hz), pause
        during active movement input, and deliberately bypass
        correction_min_interval — flood safety lives at burst-QUEUE time
        (_maybe_queue_state_request_burst), not at drain time."""
        burst_remaining = int(getattr(ctx, "correction_burst_remaining", 0) or 0)
        if burst_remaining <= 0:
            return False
        if movement_suppressed:
            return False
        burst_interval = float(getattr(ctx, "correction_burst_interval_s", 0.0) or 0.0)
        return (now - ctx.last_correction_send) >= burst_interval

    @serialized
    def _send_state_sync_snapshot(
        self,
        ctx: ClientContext,
        *,
        reason: str,
        include_view_update: bool = True,
        replay_timestamp: Optional[int] = None,
    ) -> None:
        """Send an on-demand local state snapshot for timing/state resync.

        The decompile-backed STATE_REQUEST (0x0C) path is used for targeted
        local resync. Send the authoritative UPDATE_ARRAY snapshot to any
        compatible client and optionally pair it with a VIEW_UPDATE replay
        snapshot keyed to the caller's request timing.
        """
        if not self.udp_handler or not ctx.session or not ctx.session.udp_addr:
            return

        # Do not fabricate local-player sync packets before the session has an
        # actual spawned in-game entity. Pre-spawn spectator/player IDs are not
        # safe stand-ins for the OG client's local vehicle sync path.
        if not ctx.running or ctx.observer_transition or not ctx.session.in_game or ctx.session.entity_id == 0:
            if self.debug_udp_raw:
                print(
                    f"[STATE-SYNC] Skipping {reason} for client {ctx.client_id}: "
                    f"in_game={int(ctx.session.in_game)} entity_id={ctx.session.entity_id}"
                )
            return

        entity_id = ctx.session.entity_id
        if entity_id == 0:
            return

        now = time.monotonic()
        if (now - ctx.last_state_sync_send) < 0.10:
            return
        ctx.last_state_sync_send = now

        def _wrap_angle(angle: float) -> float:
            while angle > math.pi:
                angle -= 2.0 * math.pi
            while angle < -math.pi:
                angle += 2.0 * math.pi
            return angle

        tick = self._get_network_tick(ctx)
        snapshot_mode = getattr(self, "state_sync_snapshot_mode", "remote_live")
        use_history_snapshot = snapshot_mode == "history" or (
            snapshot_mode == "remote_live" and handlers._is_loopback_client(ctx)
        )
        # Live OG telemetry shows stale replay-history poses can be accepted
        # under a fresh VIEW_UPDATE timestamp and drag local prediction behind
        # the server during real movement. Keep history available for loopback
        # tooling/A-B runs, but default remote OG correction to current state.
        snapshot = (
            self._select_authoritative_state_snapshot(ctx, replay_timestamp)
            if use_history_snapshot
            else None
        )
        if snapshot is None:
            snapshot_source = "live"
            # Same prediction-lead extrapolation as the cadence/burst correction
            # path (_build_empirical_correction_payload) so STATE_REQUEST replies
            # target the client's predicted pose, not the 1-tick-stale current one.
            send_pos, update_rot = self._lead_extrapolated_correction_pose(ctx)
            sync_vel = ctx.player_vel
        else:
            snapshot_source = "history"
            send_pos = snapshot["pos"]
            sync_vel = snapshot["vel"]
            update_rot = snapshot["rot"]
        include_sync_vel = True
        include_sync_rot = True
        # VIEW_UPDATE is a replay/correction wrapper over the same entity
        # transform payload as UPDATE_ARRAY. The OG reconcile path verifies the
        # buffered predicted rotation against that entity/body rotation, not the
        # client camera-yaw convention. Using player_yaw here flips the sign for
        # remote/OG clients and makes targeted corrections fail intermittently.
        view_rot = update_rot
        health_val = self._get_health_value(ctx)
        fuel_val = self._get_energy_value(ctx)
        if handlers._is_loopback_client(ctx):
            local_state_kwargs = self._get_local_state_kwargs(ctx)
            update_include_local_state = self._should_send_local_state(
                ctx,
                local_state_kwargs["primary_turret_bits"],
                local_state_kwargs["secondary_turret_bits"],
                self.update_local_state_mode,
            )
        else:
            # OG targeted sync is still sensitive to the local-state prefix on
            # these packets. Keep STATE_REQUEST replies on the same short-form-
            # safe shape that earlier live traces showed as stable.
            update_include_local_state, local_state_kwargs = self._get_update_array_local_state_for_viewer(ctx)
        weapon_type = local_state_kwargs["weapon_id"]
        ammo_bits = local_state_kwargs["ammo_count_bits"]
        ammo_mask = local_state_kwargs["ammo_count"]
        pt_bits = local_state_kwargs["primary_turret_bits"]
        pt_angle = local_state_kwargs["primary_turret_angle"]
        st_bits = local_state_kwargs["secondary_turret_bits"]
        st_angle = local_state_kwargs["secondary_turret_angle"]
        if update_include_local_state:
            update_ammo_bits = ammo_bits
            update_ammo_mask = ammo_mask
            update_pt_bits = pt_bits
            update_pt_angle = pt_angle
            update_st_bits = st_bits
            update_st_angle = st_angle
        else:
            update_ammo_bits = 0
            update_ammo_mask = 0
            update_pt_bits = 0
            update_pt_angle = 0.0
            update_st_bits = 0
            update_st_angle = 0.0

        if self._wf_remote_heartbeat_entity_mode(ctx):
            update_payload = self._build_remote_sync_heartbeat_update(
                ctx,
                tick=tick,
                pos=send_pos,
                vel=sync_vel,
                rot=update_rot,
                include_vel=include_sync_vel,
                include_rot=include_sync_rot,
                include_local_state=update_include_local_state,
                health=health_val,
                fuel=fuel_val,
                weapon_type=weapon_type,
                ammo_bits=update_ammo_bits,
                ammo_mask=update_ammo_mask,
                pt_bits=update_pt_bits,
                pt_angle=update_pt_angle,
                st_bits=update_st_bits,
                st_angle=update_st_angle,
                safe_local_state=not handlers._is_loopback_client(ctx),
            )
        else:
            update_payload = build_update_array_player_update(
                tick=tick,
                entity_id=entity_id,
                pos=send_pos,
                vel=sync_vel,
                rot=update_rot,
                include_pos=True,
                include_vel=include_sync_vel,
                include_rot=include_sync_rot,
                include_local_state=update_include_local_state,
                include_entity_vitals=self.update_entity_vitals,
                weapon_id=weapon_type,
                health=health_val,
                fuel=fuel_val,
                ammo_count_bits=update_ammo_bits,
                ammo_count=update_ammo_mask,
                primary_turret_bits=update_pt_bits,
                primary_turret_angle=update_pt_angle,
                secondary_turret_bits=update_st_bits,
                secondary_turret_angle=update_st_angle,
                turret_max=self.local_state_turret_max,
                turret_range=self.local_state_turret_range,
                is_manned=True,
                speed_scale=1.0,
            )

        self.udp_handler.send_to(update_payload, ctx.session.udp_addr)
        ctx.state_sync_reply_count += 1
        ctx.last_state_sync_reply_time = now
        ctx.last_state_sync_reply_tick = tick
        ctx.last_state_sync_replay_timestamp = int(replay_timestamp or 0)
        ctx.last_state_sync_snapshot_source = snapshot_source

        view_include_local_state = False
        view_payload = b""
        if include_view_update:
            # VIEW_UPDATE is replay-mode input to the client's prediction path.
            # Loopback/Python clients keep the STATE_REQUEST/replay id for
            # latency correlation. Remote OG clients need a current/future
            # wrapper timestamp because UpdateArray_check_eligible rejects
            # stale interp_record+0x08 values before prediction storage.
            view_timestamp = replay_timestamp
            if not handlers._is_loopback_client(ctx):
                view_timestamp = self._fresh_remote_view_update_timestamp(ctx, tick)
            view_weapon_type = weapon_type
            view_ammo_bits = 0
            view_ammo_mask = 0
            view_pt_bits = 0
            view_pt_angle = 0.0
            view_st_bits = 0
            view_st_angle = 0.0
            if handlers._is_loopback_client(ctx):
                if self.view_update_local_stats:
                    view_mode = "wf" if self.update_local_state_mode == "wf" else "auto"
                    view_include_local_state = self._should_send_local_state(
                        ctx,
                        pt_bits,
                        st_bits,
                        view_mode,
                    )
                    if view_include_local_state:
                        view_ammo_bits = ammo_bits
                        view_ammo_mask = ammo_mask
                        view_pt_bits = pt_bits
                        view_pt_angle = pt_angle
                        view_st_bits = st_bits
                        view_st_angle = st_angle
            else:
                # VIEW_UPDATE shares the same local-state parser as UPDATE_ARRAY.
                # Keep the replay companion on the same short-form-safe prefix as
                # the targeted UPDATE_ARRAY reply; the transform/timestamp payload
                # carries the correction signal, not expanded ammo/turret bits.
                view_include_local_state = update_include_local_state
                view_weapon_type = weapon_type
                view_ammo_bits = update_ammo_bits
                view_ammo_mask = update_ammo_mask
                view_pt_bits = update_pt_bits
                view_pt_angle = update_pt_angle
                view_st_bits = update_st_bits
                view_st_angle = update_st_angle

            view_payload = build_view_update_player_update(
                tick=tick,
                entity_id=entity_id,
                pos=send_pos,
                vel=sync_vel,
                rot=view_rot,
                include_pos=True,
                include_vel=include_sync_vel,
                include_rot=include_sync_rot,
                include_local_state=view_include_local_state,
                include_entity_vitals=self.view_update_entity_vitals,
                weapon_id=view_weapon_type,
                health=health_val,
                fuel=fuel_val,
                ammo_count_bits=view_ammo_bits,
                ammo_count=view_ammo_mask,
                primary_turret_bits=view_pt_bits,
                primary_turret_angle=view_pt_angle,
                secondary_turret_bits=view_st_bits,
                secondary_turret_angle=view_st_angle,
                turret_max=self.local_state_turret_max,
                turret_range=self.local_state_turret_range,
                is_manned=True,
                speed_scale=1.0,
                timestamp=view_timestamp,
            )
            self.udp_handler.send_to(view_payload, ctx.session.udp_addr)
            ctx.state_sync_view_reply_count += 1

        ctx.last_state_sync_reason = reason
        ctx.last_state_sync_update_len = len(update_payload)
        ctx.last_state_sync_view_len = len(view_payload)
        ctx.last_state_sync_update_has_local_state = bool(update_include_local_state)
        ctx.last_state_sync_view_has_local_state = bool(view_include_local_state)
        ctx.last_state_sync_view_timestamp = int(view_timestamp or 0) if include_view_update else 0
        ctx.last_state_sync_update_hex = update_payload[:32].hex()
        ctx.last_state_sync_view_hex = view_payload[:32].hex()

        if self.pktlog.enabled:
            self.pktlog.log(
                client_id=ctx.client_id,
                label="STATE_SYNC_UPDATE",
                tick=tick,
                payload=update_payload,
                transport="UDP",
                entity_count=1,
                entity_ids=(entity_id,),
                mask_bits=(0b1110,),
                has_local_state=update_include_local_state,
                health=health_val,
                extra=(
                    f"reason={reason} replay=0x{int(replay_timestamp or 0):08x} "
                    f"source={snapshot_source}"
                ),
            )
            if include_view_update:
                self.pktlog.log(
                    client_id=ctx.client_id,
                    label="STATE_SYNC_VIEW",
                    tick=tick,
                    payload=view_payload,
                    transport="UDP",
                    entity_count=1,
                    entity_ids=(entity_id,),
                    mask_bits=(0b1110,),
                    has_local_state=view_include_local_state,
                    health=health_val,
                    extra=(
                        f"reason={reason} replay=0x{int(replay_timestamp or 0):08x} "
                        f"view_ts=0x{int(view_timestamp or 0):08x} source={snapshot_source}"
                    ),
                )

        if self.debug_viewpoint or self.debug_udp_raw:
            print(
                f"[STATE-SYNC] client={ctx.client_id} reason={reason} "
                f"tick={tick} pos=({send_pos[0]:.2f},{send_pos[1]:.2f},{send_pos[2]:.2f}) "
                f"view={int(include_view_update)}"
            )

    # ============ Weapon System Handlers ============

    def _send_debug_sync(self, ctx: ClientContext, frame_counter: int,
                         turn_input: float, fwd_input: float, strafe_input: float):
        """Send DEBUG_SYNC packet (0x60) with server's authoritative state.

        49-byte UDP packet sent after each physics step for measuring
        client-server divergence. Opcode 0x60 is unused by the original protocol.
        """
        if not self._debug_sync_allowed_for_client(ctx):
            return
        # DEBUG_SYNC should use the same client-facing position space as
        # UPDATE_ARRAY/VIEW_UPDATE, otherwise the sync harness reports the
        # configured terrain/world offset as fake divergence.
        pos = self._to_client_pos(ctx.player_pos)
        heading = ctx.player_heading
        vel = ctx.player_vel
        ang_vel = ctx.vehicle_physics.angular_velocity if ctx.vehicle_physics else 0.0
        steps = ctx.physics_step_count

        payload = struct.pack(
            ">BII3f1f3f1f3f",
            0x60,
            frame_counter,
            steps,
            pos[0], pos[1], pos[2],
            heading,
            vel[0], vel[1], vel[2],
            ang_vel,
            turn_input, fwd_input, strafe_input,
        )
        if self.udp_handler and ctx.session.udp_addr:
            self.udp_handler.send_to(payload, ctx.session.udp_addr)

    @staticmethod
    def _normalize_debug_host(host: str) -> str:
        host = host.strip().lower()
        if host.startswith("::ffff:"):
            return host[7:]
        return host

    def _debug_sync_allowed_for_client(self, ctx: ClientContext) -> bool:
        if not self.debug_sync:
            return False
        if self.debug_sync_allow_all:
            return True

        addr = ctx.session.udp_addr or ctx.client_addr
        if not addr:
            return False

        host = self._normalize_debug_host(str(addr[0]))
        if host in self.debug_sync_hosts:
            return True

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None

        if ip and ip.is_loopback and (
            "127.0.0.1" in self.debug_sync_hosts
            or "::1" in self.debug_sync_hosts
            or "loopback" in self.debug_sync_hosts
        ):
            return True

        if ctx.client_id not in self._debug_sync_blocked_clients:
            self._debug_sync_blocked_clients.add(ctx.client_id)
            print(
                f"[DEBUG_SYNC] Suppressing custom 0x60 for client {ctx.client_id} "
                f"host={host} allowlist={sorted(self.debug_sync_hosts)}"
            )
        return False

    def _state_sync_reply_allowed_for_client(self, ctx: ClientContext) -> bool:
        if self.state_sync_reply_allow_all:
            return True

        addr = ctx.session.udp_addr or ctx.client_addr
        if not addr:
            return False

        host = self._normalize_debug_host(str(addr[0]))
        if host in self.state_sync_reply_hosts:
            return True

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None

        if ip and ip.is_loopback and (
            "127.0.0.1" in self.state_sync_reply_hosts
            or "::1" in self.state_sync_reply_hosts
            or "loopback" in self.state_sync_reply_hosts
        ):
            return True

        if ctx.client_id not in self._state_sync_blocked_clients:
            self._state_sync_blocked_clients.add(ctx.client_id)
            print(
                f"[STATE-SYNC] Suppressing STATE_REQUEST reply for client {ctx.client_id} "
                f"host={host} allowlist={sorted(self.state_sync_reply_hosts)}"
            )
        return False

    def _should_send_state_sync_view_update(self, ctx: ClientContext) -> bool:
        mode = getattr(self, "state_sync_view_mode", "all")
        is_loopback = handlers._is_loopback_client(ctx)
        if mode == "off":
            return False
        if mode == "loopback":
            return is_loopback
        if mode == "remote":
            return not is_loopback
        return True

    def _fresh_remote_view_update_timestamp(self, ctx: ClientContext, tick: int) -> int:
        """Return a remote OG VIEW_UPDATE timestamp that the client will clamp fresh.

        The OG replay handler clamps future VIEW_UPDATE timestamps down to its
        current tick before storing interp_record+0x08. Live wulftap shows stale
        STATE_REQUEST ids are rejected by UpdateArray_check_eligible before
        NetworkSync_receive_entity_update can store server_expected, so remote
        OG replay wrappers must be current/future while snapshot selection can
        still use the original request id.
        """
        base_tick = int(tick) & 0xFFFFFFFF
        client_tick = int(getattr(ctx, "last_client_tick", 0) or 0) & 0xFFFFFFFF
        if client_tick and self._tick_delta_signed(client_tick, base_tick) > 0:
            base_tick = client_tick
        ahead_ms = int(getattr(self, "remote_view_update_timestamp_ahead_ms", 1000) or 0)
        if ahead_ms < 0:
            ahead_ms = 0
        return (base_tick + ahead_ms) & 0xFFFFFFFF

    def _accumulate_correction_divergence(self, ctx: ClientContext, clamp_dz: float) -> None:
        """Accumulate server-only (client-unpredictable) displacement for the
        reactive correction gate (GOAL 2, 2026-06-02).

        `clamp_dz` is the magnitude the terrain-Z safety clamp pushed the tank
        this physics step — the canonical client/server divergence on this map
        (server clamps Z to terrain; the OG client uses a spring-damper). The
        accumulator decays every call so transient float/quantization noise
        bleeds off and never reaches threshold; only a genuine spike (collision/
        teleport-scale push) or sustained divergence (driving across rough
        terrain) trips the gate. Flat open ground keeps this at ~0 because the
        clamp rarely fires there and small dips stay under the noise floor.
        """
        if not getattr(self, "correction_gate_enabled", False):
            return
        accum = float(getattr(ctx, "divergence_accum_pos", 0.0) or 0.0) * self.correction_divergence_decay
        dz = abs(float(clamp_dz or 0.0))
        if dz > self.correction_divergence_floor:
            accum += dz
            if self.correction_gate_debug:
                ctx._divergence_clamp_events = int(getattr(ctx, "_divergence_clamp_events", 0) or 0) + 1
                ctx._divergence_clamp_max = max(
                    float(getattr(ctx, "_divergence_clamp_max", 0.0) or 0.0), dz
                )
        ctx.divergence_accum_pos = accum

    def _lead_extrapolated_correction_pose(self, ctx: ClientContext):
        """Correction-target pose, optionally extrapolated forward by the client's
        prediction lead (WULFRAM_CORRECTION_LEAD_TICKS; see _init_correction_config).

        Returns ``(client_pos, rot)`` in exactly the conventions the no-lead path
        used — `_to_client_pos(player_pos)` and `_local_player_sync_rotation` —
        so lead=0 is byte-identical to the previous behaviour. When lead>0 the
        position is advanced by `player_vel` and the body heading by
        `angular_vel_yaw` (the same rates the integrator uses to advance pos and
        `player_heading`), so the snap lands where the client already predicted.
        At rest both rates are 0, making the correction a no-op.
        """
        rot = self._local_player_sync_rotation(ctx)
        lead = float(getattr(self, "correction_lead_ticks", 0.0) or 0.0)
        if lead <= 0.0:
            return self._to_client_pos(ctx.player_pos), rot
        dt_lead = lead / max(1.0, float(getattr(self, "tick_rate_hz", 30.0)))
        px, py, pz = ctx.player_pos
        vx, vy, vz = getattr(ctx, "player_vel", (0.0, 0.0, 0.0))
        ext_pos = (px + vx * dt_lead, py + vy * dt_lead, pz + vz * dt_lead)
        ang = float(getattr(ctx, "angular_vel_yaw", 0.0) or 0.0)
        roll, pitch, yaw = rot
        ext_rot = (roll, pitch, yaw + ang * dt_lead)
        return self._to_client_pos(ext_pos), ext_rot

    @staticmethod
    def _attitude_delta_rad(a, b) -> float:
        """Largest per-axis wrapped angular difference between two euler triples.

        A conservative proxy for the delta the client computes. The client does it
        properly -- GUESS5_Camera_clamp_network_delta converts both triples through
        0x00415260 and compares via 0x004e1e80 -- so a per-axis maximum can only
        over-estimate the true angle, never under-estimate it. Over-estimating is
        the safe direction here: it suppresses fewer packets than a true angle
        would, so the guard never drops a correction the client would have acted on.
        """
        worst = 0.0
        for left, right in zip(a, b):
            delta = math.fmod(float(left) - float(right), math.tau)
            if delta > math.pi:
                delta -= math.tau
            elif delta < -math.pi:
                delta += math.tau
            worst = max(worst, abs(delta))
        return worst

    def _build_empirical_correction_payload(
        self,
        ctx: ClientContext,
        *,
        tick: int,
        include_local_state: bool,
        health: float,
        fuel: float,
        weapon_type: int,
        ammo_bits: int,
        ammo_mask: int,
        pt_bits: int,
        pt_angle: float,
        st_bits: int,
        st_angle: float,
        allow_suppression: bool = False,
    ) -> tuple[bytes, str, tuple[float, float, float], tuple[float, float, float], bool, bool]:
        """Build one of the older empirical local-correction packet shapes.

        With ``allow_suppression`` the builder may return an EMPTY payload when the
        packet it would produce cannot change the client. Callers must treat an
        empty payload as "do not send". Only the automatic tick path passes it; an
        explicit `correction send` from the control port does not, because an
        operator asking for a packet should get one.

        The rotation tuple MUST match what every other local-player packet
        path emits (heartbeat, full UPDATE_ARRAY, VIEW_UPDATE loop,
        TankPacket resend, vitals heartbeat, solo-local keepalive). Those
        all use `_local_player_sync_rotation` which returns
        `(roll, pitch, player_heading)` — the body-heading convention.
        This function historically used `(roll, 0.0, player_yaw)` — the
        camera-yaw sign-flipped convention — which diverged from every
        other path and was flagged in the 2026-03-14 audit but missed.
        Aligning to `_local_player_sync_rotation` does not on its own
        restore the correction burst when the keepalive is running, but
        it removes a confounding variable. See
        docs/keepalive-breaks-correction-2026-04-18.md.
        """
        self._repair_recent_control_pose_jump(ctx, "correction_payload")
        corr_pos, corr_rot = self._lead_extrapolated_correction_pose(ctx)
        cmode = self.correction_mode
        inc_pos = cmode in ("full", "pos_only", "dual_entity", "view_update", "view_update_define")
        inc_vel = cmode in ("full", "pos_only", "dual_entity", "view_update")
        inc_rot = cmode in ("full", "rot_only", "dual_entity", "view_update", "view_update_define")
        # ROT-ONLY guard: drop pos/vel so a VIEW_UPDATE carries rotation only. Sending BOTH
        # pos+rot makes the OG client sleep the local player (Entity_reset_physics) -> NaN
        # freeze when corrected mid-motion. rot-only avoids reset_physics (zero-velocity branch).
        # cmode stays view_update so the replay timestamp + build_view_update_player_update path
        # are preserved -- still a real correction, just heading-only. See server.__init__.
        if getattr(self, "correction_rot_only", False) and inc_rot:
            inc_pos = False
            inc_vel = False
        # Sub-epsilon attitude guard. The client replaces the wire rotation sample
        # with its own whenever the two already agree -- 0048c126..0048c135 inside
        # GUESS5_Camera_clamp_network_delta, reached from 0047d67e for the local
        # player -- so such a packet provably cannot move entity+0x30. Measured in
        # rebuild analysis/moravec-s1/ingress-capture-v14-shapeB against its
        # 90-degree control. The baseline is the server's own authoritative
        # attitude, which is also its model of the client's; with lead=0 the delta
        # is exactly zero, and with lead>0 it is the extrapolation, which is ~0 at
        # rest. So this suppresses precisely the at-rest no-op the lead docstring
        # already describes as "a no-op".
        suppressed = None
        epsilon = float(getattr(self, "correction_rot_epsilon_rad", 0.0) or 0.0)
        if allow_suppression and inc_rot and epsilon > 0.0:
            baseline = self._local_player_sync_rotation(ctx)
            delta = self._attitude_delta_rad(corr_rot, baseline)
            if delta < epsilon:
                inc_rot = False
                suppressed = f"rot_subepsilon({math.degrees(delta):.4f}deg)"
        if allow_suppression and not (inc_pos or inc_vel or inc_rot):
            # Nothing transform-bearing left. Do NOT fall through to a mask=0
            # entity: a zero-transform entity zeroes the client's angular
            # velocity (see the heartbeat notes), so an "empty correction" is not
            # harmless. Refuse instead and let the caller skip the send.
            return (b"", f"CORRECTION(suppressed:{suppressed or 'empty'})",
                    corr_pos, corr_rot, False, False)
        def _fresh_ts():
            if cmode in ("view_update", "view_update_define"):
                if handlers._is_loopback_client(ctx):
                    last_request_id = int(getattr(ctx, "last_state_request_id", 0) or 0)
                    last_request_time = float(getattr(ctx, "last_state_request_time", 0.0) or 0.0)
                    if last_request_id and last_request_time > 0.0 and (time.monotonic() - last_request_time) <= 1.5:
                        return last_request_id
                    return None
                return self._fresh_remote_view_update_timestamp(ctx, tick)
            return None

        if not handlers._is_loopback_client(ctx):
            # OG clients reject the promoted/full local-state form on targeted
            # correction bursts. Keep this aligned with STATE_REQUEST replies:
            # a short local-state prefix plus the transform-bearing entity.
            include_local_state = True
            weapon_type = self._get_spawn_tank_weapon_type(ctx)
            ammo_bits = 0
            ammo_mask = 0
            pt_bits = 0
            pt_angle = 0.0
            st_bits = 0
            st_angle = 0.0

        common_kw = dict(
            include_local_state=include_local_state,
            weapon_id=weapon_type,
            health=health,
            fuel=fuel,
            ammo_count_bits=ammo_bits,
            ammo_count=ammo_mask,
            primary_turret_bits=pt_bits,
            primary_turret_angle=pt_angle,
            secondary_turret_bits=st_bits,
            secondary_turret_angle=st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
        )

        def _build_one(ip: bool, iv: bool, ir: bool, ts):
            if cmode == "view_update_define":
                team_id = int(getattr(ctx.session, "team_id", 0) or 1)
                entity_type = int(getattr(ctx, "entity_type", EntityType.TANK) or EntityType.TANK)
                payload = build_view_update_create_tank(
                    tick=tick,
                    entity_id=ctx.session.entity_id,
                    entity_type=entity_type,
                    team=team_id,
                    pos=corr_pos,
                    behavior_type=team_id,
                    include_health=include_local_state,
                    include_entity_vitals=False,
                    health=health,
                    fuel=fuel,
                    is_manned=True,
                    weapon_id=weapon_type,
                    rot=corr_rot,
                    ammo_count_bits=ammo_bits,
                    ammo_count=ammo_mask,
                    primary_turret_bits=pt_bits,
                    primary_turret_angle=pt_angle,
                    secondary_turret_bits=st_bits,
                    secondary_turret_angle=st_angle,
                    turret_max=self.local_state_turret_max,
                    turret_range=self.local_state_turret_range,
                    timestamp=ts,
                )
                return payload, "CORRECTION(view_update_define)"
            if cmode == "view_update":
                payload = build_view_update_player_update(
                    tick=tick,
                    entity_id=ctx.session.entity_id,
                    pos=corr_pos,
                    vel=ctx.player_vel,
                    rot=corr_rot,
                    include_pos=ip,
                    include_vel=iv,
                    include_rot=ir,
                    timestamp=ts,
                    **common_kw,
                )
                return payload, "CORRECTION(view_update)"
            if cmode == "dual_entity":
                ent_real = dict(
                    entity_id=ctx.session.entity_id,
                    is_manned=True,
                    pos=corr_pos,
                    vel=ctx.player_vel,
                    rot=corr_rot,
                    include_pos=ip,
                    include_vel=iv,
                    include_rot=ir,
                )
                ent_dummy = dict(
                    entity_id=0xFFFFFFFE,
                    is_manned=True,
                    pos=(0.0, 0.0, 0.0),
                    vel=(0.0, 0.0, 0.0),
                    rot=(0.0, 0.0, 0.0),
                    include_pos=False,
                    include_vel=False,
                    include_rot=False,
                )
                payload = build_update_array_multi(
                    tick=tick,
                    entities=[ent_real, ent_dummy],
                    **common_kw,
                )
                return payload, "CORRECTION(dual_entity)"
            payload = build_update_array_player_update(
                tick=tick,
                entity_id=ctx.session.entity_id,
                pos=corr_pos,
                vel=ctx.player_vel,
                rot=corr_rot,
                include_pos=ip,
                include_vel=iv,
                include_rot=ir,
                **common_kw,
            )
            return payload, f"CORRECTION({cmode})"

        payload, label = _build_one(inc_pos, inc_vel, inc_rot, _fresh_ts())
        return payload, label, corr_pos, corr_rot, inc_pos, inc_rot

    def _maybe_send_view_update_loop(
        self,
        ctx: ClientContext,
        *,
        tick: int,
        send_pos: tuple,
        health_val: float,
        fuel_val: float,
        weapon_type: int,
        ammo_bits: int,
        ammo_mask: int,
        pt_bits: int,
        pt_angle: float,
        st_bits: int,
        st_angle: float,
    ) -> None:
        """Optionally send auxiliary VIEW_UPDATE correction/replay packets."""
        if not self.view_update_loop or self.update_packet_type == "view":
            return

        now = time.monotonic()
        if (now - ctx.last_view_update_send) < self.view_update_interval:
            return
        ctx.last_view_update_send = now

        view_local_state = False
        view_ammo_bits = 0
        view_ammo_mask = 0
        view_pt_bits = 0
        view_st_bits = 0
        view_pt_angle = 0.0
        view_st_angle = 0.0

        # Only include local-state in VIEW_UPDATE when explicitly enabled.
        if self.view_update_local_stats:
            view_mode = "wf" if self.update_local_state_mode == "wf" else "auto"
            if self._should_send_local_state(ctx, pt_bits, st_bits, view_mode):
                view_local_state = True
                view_ammo_bits = ammo_bits
                view_ammo_mask = ammo_mask
                view_pt_bits = pt_bits
                view_st_bits = st_bits
                view_pt_angle = pt_angle
                view_st_angle = st_angle

        view_entities = [
            {
                "entity_id": ctx.session.entity_id,
                "is_manned": True,
                "pos": send_pos,
                "vel": ctx.player_vel,
                "rot": self._local_player_sync_rotation(ctx),
                "include_pos": True,
                "include_vel": True,
                "include_rot": True,
                "include_entity_vitals": self.view_update_entity_vitals,
                "speed_scale": 1.0,
                "fuel": fuel_val,
            }
        ]
        view_payload = build_view_update_multi(
            tick,
            include_local_state=view_local_state,
            weapon_id=weapon_type,
            health=health_val,
            fuel=fuel_val,
            ammo_count_bits=view_ammo_bits,
            ammo_count=view_ammo_mask,
            primary_turret_bits=view_pt_bits,
            primary_turret_angle=view_pt_angle,
            secondary_turret_bits=view_st_bits,
            secondary_turret_angle=view_st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
            entities=view_entities,
        )
        if view_local_state:
            self._log_vitals(
                ctx,
                "VIEW_UPDATE",
                include_vitals=True,
                health=health_val,
                energy=fuel_val,
                weapon_id=weapon_type,
                note=f"ammo_bits={view_ammo_bits} pt_bits={view_pt_bits} st_bits={view_st_bits}",
            )
        if self.udp_handler and ctx.session.udp_addr:
            self.udp_handler.send_to(view_payload, ctx.session.udp_addr)
        # Never send VIEW_UPDATE over TCP (can desync OG stream parser).
