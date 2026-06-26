"""ReplicationMixin -- local-state payload construction for UPDATE_ARRAY
replication, extracted verbatim from WulframServer (server.py decomposition,
step 3). Method-only mixin; shares state via `self`. First chunk of the
replication layer (the local-state heartbeat builders).
"""
from __future__ import annotations

import os
import time
from typing import Optional

from . import handlers
from .client import ClientContext
from .packets import (
    build_update_array_heartbeat,
    build_add_to_roster,
    build_remove_from_roster,
    build_update_stats,
    build_update_array_create_tank,
    build_update_array_player_update,
)
from wulfram2_protocol.entities import (
    LOCAL_STATE_PRIMARY_TURRET_WEAPON_TYPES,
    LOCAL_STATE_SECONDARY_TURRET_WEAPON_TYPES,
)


class ReplicationMixin:
    def _get_local_state_weapon_type(self, ctx: ClientContext) -> int:
        """Return weapon type index used by local player state (entity type index, not weapon slot)."""
        if self.update_local_state_mode == "wf":
            return self.local_state_weapon_type or 0
        if self.local_state_weapon_type:
            return self.local_state_weapon_type
        if getattr(ctx, "entity_type", None) is not None:
            return int(ctx.entity_type)
        return 0

    def _get_spawn_tank_weapon_type(self, ctx: Optional[ClientContext] = None) -> int:
        """Return weapon id used for spawn TankPacket vitals (short local-state parse-safe)."""
        weapon = self.spawn_tank_weapon_type
        if 0 <= weapon <= 31:
            return weapon
        if ctx is not None:
            return self._get_local_state_weapon_type(ctx)
        return 2

    def _wf_minimal_local_state_for_client(self, ctx: ClientContext) -> bool:
        """Return True while a remote OG client still needs the short-form-safe path.

        Spawn-time `TANK` / `PLAYER_INFO` / first heartbeat are still sensitive
        to local-state bit-count mismatches. Keep remote clients on the safe
        short form until they have explicitly requested targeted sync. Promotion
        after that point controls the broader post-spawn sync/heartbeat path,
        but targeted correction packets still keep the safe local-state prefix.
        """
        if self.update_local_state_mode != "wf":
            return False
        if handlers._is_loopback_client(ctx):
            return False
        return not bool(getattr(ctx, "remote_full_local_state_ready", False))

    def _wf_remote_heartbeat_entity_mode(self, ctx: ClientContext) -> bool:
        """Return True when remote `wf` heartbeats should use local-player sync updates.

        Remote OG clients need the short-form spawn-safe path first, then the
        promoted single-local-player update transport once targeted sync is
        active. That promotion no longer implies the expanded ammo/turret
        local-state shape on targeted correction packets.
        """
        if self.update_local_state_mode != "wf":
            return False
        if handlers._is_loopback_client(ctx):
            return False
        return bool(getattr(ctx, "remote_full_local_state_ready", False))

    def _suppress_remote_spawn_safe_heartbeat(self, ctx: ClientContext) -> bool:
        """Suppress periodic remote heartbeats until the OG client leaves spawn-safe mode.

        The short-form minimal heartbeat exists for spawn/bootstrap safety, but
        it is also the narrowest remaining suspect when OG falls back to the
        address screen immediately after join. Keep the one-off spawn heartbeat,
        but do not stream periodic short-form heartbeats before targeted sync
        has promoted the client into the fuller post-spawn update transport.
        """
        return self._wf_minimal_local_state_for_client(ctx)

    def _suppress_remote_spawn_bootstrap_heartbeat(self, ctx: ClientContext) -> bool:
        """Suppress the one-off post-spawn heartbeat while remote OG stays on the minimal path.

        The 10-byte heartbeat is safe enough for steady loopback probes, but on
        the remote OG bootstrap path it still lands in the narrow packet window
        where protocol-mismatch falls back to the address screen. Once the
        client explicitly requests targeted sync we can leave spawn-safe mode
        and rely on the promoted post-spawn transport instead.
        """
        return self._wf_minimal_local_state_for_client(ctx)

    def _resume_remote_full_local_state_after_spawn(
        self,
        ctx: ClientContext,
        *,
        entity_id: int,
        health: float = 1.0,
        fuel: float = 1.0,
        previously_promoted: bool = False,
    ) -> bool:
        """Restore promoted remote local-state sync after a respawn-safe spawn sequence.

        Keep the initial remote spawn packets on the minimal safe path, but do
        not strand a previously-promoted OG client there after a respawn. Once
        the fresh entity exists again, resume the promoted heartbeat shape and
        send one immediate full heartbeat so the client's local timing/correction
        path has authoritative state to lock onto again.
        """
        if not previously_promoted:
            return False
        if self.update_local_state_mode != "wf":
            return False
        if handlers._is_loopback_client(ctx):
            return False

        ctx.remote_full_local_state_ready = True
        ctx._spawn_safe_heartbeat_suppressed_logged = False
        ctx.last_state_sync_send = 0.0

        if not self.udp_handler or not ctx.session or not ctx.session.udp_addr:
            return True

        try:
            hb_tick = self._get_network_tick(ctx)
            hb_packet = self._build_local_state_heartbeat(
                ctx,
                tick=hb_tick,
                entity_id=entity_id,
                include_health=True,
                health=health,
                fuel=fuel,
            )
            self.udp_handler.send_to(hb_packet, ctx.session.udp_addr)
            print(
                f"[SPAWN] Client {ctx.client_id}: restored promoted remote sync "
                "path and sent immediate full heartbeat"
            )
        except Exception as ex:
            print(
                f"[SPAWN] WARNING: Failed to restore promoted remote sync path "
                f"for client {ctx.client_id}: {ex}"
            )
        return True

    def _get_local_state_ammo_bits(self, ctx: ClientContext, *, force_full_remote: bool = False) -> tuple:
        """Return (ammo_bits, ammo_mask) for local player state."""
        if self._wf_minimal_local_state_for_client(ctx) and not force_full_remote:
            return 0, 0
        if self.local_state_ammo_override:
            return self.local_state_ammo_bits, self.local_state_ammo_mask

        if not self.local_state_ammo_from_behavior:
            return 0, 0

        weapon_type = self._get_local_state_weapon_type(ctx)
        active_bits = 0
        if 0 <= weapon_type < len(self.behavior_weapon_caps):
            active_bits = self.behavior_weapon_caps[weapon_type][2]
        # Send 0 mask (no weapons actively firing).  The active-flags bitmask
        # controls AmmoSlotState+0x05 via update_active_flags(); bit=1 causes
        # the client's WeaponCooldown_update_all() to auto-fire that slot.
        # Only set bits when the player is actually firing (TODO).
        active_mask = 0
        return active_bits, active_mask

    def _get_local_state_turret_bits(self, ctx: ClientContext, *, force_full_remote: bool = False) -> tuple:
        """
        Return (primary_bits, primary_angle, secondary_bits, secondary_angle).
        Flags are inferred from entity_type unless overrides are provided.
        """
        if self._wf_minimal_local_state_for_client(ctx) and not force_full_remote:
            return 0, 0.0, 0, 0.0
        weapon_type = self._get_local_state_weapon_type(ctx)

        # Default turret bits based on WeaponDef_init_by_entity_type (azurefishy decomp).
        primary_flag = weapon_type in LOCAL_STATE_PRIMARY_TURRET_WEAPON_TYPES
        secondary_flag = weapon_type in LOCAL_STATE_SECONDARY_TURRET_WEAPON_TYPES

        if self.local_state_primary_override:
            primary_flag = self.local_state_primary_override not in ("0", "false", "False")
        if self.local_state_secondary_override:
            secondary_flag = self.local_state_secondary_override not in ("0", "false", "False")

        if primary_flag:
            primary_bits = max(0, self.local_state_turret_bits)
            primary_angle = ctx.player_aim_yaw if ctx else 0.0
        else:
            primary_bits = 0
            primary_angle = 0.0

        if secondary_flag:
            secondary_bits = max(0, self.local_state_turret_bits)
            secondary_angle = ctx.player_aim_yaw if ctx else 0.0
        else:
            secondary_bits = 0
            secondary_angle = 0.0

        return primary_bits, primary_angle, secondary_bits, secondary_angle

    def _get_local_state_kwargs(self, ctx: ClientContext, *, force_full_remote: bool = False) -> dict:
        """Return dict of local_player_state kwargs for any UPDATE_ARRAY builder.

        Every UPDATE_ARRAY with include_local_state=True MUST include the
        correct ammo/turret bit counts matching the BEHAVIOR packet config,
        otherwise the OG client's bitstream reads misalign â†’ protocol mismatch.
        """
        weapon_id = self._get_local_state_weapon_type(ctx)
        ammo_bits, ammo_mask = self._get_local_state_ammo_bits(
            ctx,
            force_full_remote=force_full_remote,
        )
        pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(
            ctx,
            force_full_remote=force_full_remote,
        )
        if self._wf_minimal_local_state_for_client(ctx) and not force_full_remote:
            weapon_id = self._get_spawn_tank_weapon_type(ctx)
        return dict(
            weapon_id=weapon_id,
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
        )

    def _get_player_info_local_state_kwargs(self, ctx: ClientContext) -> tuple[bool, dict]:
        """Return (include_local_state, kwargs) for PLAYER_INFO.

        The original client always runs the local-state reader in PLAYER_INFO.
        Spawn TankPacket (0x18) still uses the short-form-safe weapon id because
        that packet only carries weapon+health+fuel, but canonical PLAYER_INFO
        should carry the real tank local-state so the OG HUD/ammo/turret state
        initializes through the decompile-backed path.
        """
        weapon_type = self._get_local_state_weapon_type(ctx)
        ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
        pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
        minimal_remote = self._wf_minimal_local_state_for_client(ctx)
        if minimal_remote:
            weapon_type = self._get_spawn_tank_weapon_type(ctx)

        mode = self.player_info_local_state_mode
        if mode == "off":
            include_local_state = False
        elif mode in ("force", "wf"):
            include_local_state = True
        elif mode == "auto-remote":
            # OG clients (incl. loopback play.bat) need local_state; only genuine
            # Python test clients take the entity-only path.
            include_local_state = handlers._is_og_client(ctx)
        else:
            include_local_state = self._local_state_payload_is_safe_for_weapon(
                weapon_type,
                pt_bits,
                st_bits,
            )

        kwargs = dict(
            weapon_id=weapon_type,
            health=1.0,
            fuel=1.0,
            ammo_count_bits=ammo_bits,
            ammo_count=ammo_mask,
            primary_turret_bits=pt_bits,
            primary_turret_angle=pt_angle,
            secondary_turret_bits=st_bits,
            secondary_turret_angle=st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
        )
        return include_local_state, kwargs

    def _local_state_payload_is_safe(self, ctx: ClientContext, primary_bits: int, secondary_bits: int) -> bool:
        """Return True if local-state payload includes required turret angles for this weapon type."""
        weapon_type = self._get_local_state_weapon_type(ctx)
        if weapon_type in LOCAL_STATE_PRIMARY_TURRET_WEAPON_TYPES and primary_bits <= 0:
            return False
        if weapon_type in LOCAL_STATE_SECONDARY_TURRET_WEAPON_TYPES and secondary_bits <= 0:
            return False
        return True

    def _local_state_payload_is_safe_for_weapon(self, weapon_type: int, primary_bits: int, secondary_bits: int) -> bool:
        """Return True if local-state payload includes required turret angles for an explicit weapon type."""
        if weapon_type in LOCAL_STATE_PRIMARY_TURRET_WEAPON_TYPES and primary_bits <= 0:
            return False
        if weapon_type in LOCAL_STATE_SECONDARY_TURRET_WEAPON_TYPES and secondary_bits <= 0:
            return False
        return True

    def _should_send_local_state(self, ctx: ClientContext, primary_bits: int, secondary_bits: int, mode: str) -> bool:
        """Decide whether to include local-state based on mode and required fields."""
        if mode == "off":
            return False
        if not ctx.session or not ctx.session.translation_ack_received:
            return False
        if mode in ("force", "wf"):
            return True
        safe = self._local_state_payload_is_safe(ctx, primary_bits, secondary_bits)
        if not safe and not getattr(ctx, "local_state_warned", False):
            weapon_type = self._get_local_state_weapon_type(ctx)
            print(
                "[LOCAL-STATE] Skipping local-state (auto): "
                f"weapon_type={weapon_type} primary_bits={primary_bits} secondary_bits={secondary_bits}"
            )
            ctx.local_state_warned = True
        return safe

    def _build_local_state_heartbeat(
        self,
        ctx: ClientContext,
        *,
        tick: Optional[int] = None,
        entity_id: Optional[int] = None,
        include_health: bool = True,
        health: Optional[float] = None,
        fuel: Optional[float] = None,
        is_view_update: Optional[bool] = None,
        rot: tuple = None,
        pos: tuple = None,
    ) -> bytes:
        """Build a heartbeat packet with local-state fields aligned to client expectations."""
        if tick is None:
            tick = self._get_network_tick(ctx)
        if entity_id is None:
            entity_id = ctx.session.entity_id or ctx.entity_id
        if health is None:
            health = self._get_health_value(ctx)
        if fuel is None:
            fuel = self._get_energy_value(ctx)
        if is_view_update is None:
            is_view_update = self.heartbeat_view_update

        weapon_type = self._get_local_state_weapon_type(ctx)
        ammo_bits, ammo_mask = self._get_local_state_ammo_bits(ctx)
        pt_bits, pt_angle, st_bits, st_angle = self._get_local_state_turret_bits(ctx)
        minimal_remote = self._wf_minimal_local_state_for_client(ctx)
        if minimal_remote:
            weapon_type = self._get_spawn_tank_weapon_type(ctx)

        include_local_state = False
        if include_health:
            include_local_state = self._should_send_local_state(
                ctx,
                pt_bits,
                st_bits,
                self.update_local_state_mode,
            )

        if not include_local_state:
            ammo_bits = 0
            ammo_mask = 0
            pt_bits = 0
            st_bits = 0
            pt_angle = 0.0
            st_angle = 0.0

        # GOAL 7: the remote-sync entity path sends the OG client its OWN entity record
        # (id, is_manned, mask=0) every heartbeat, which the OG treats as "zero my
        # angular velocity" — stomping its predicted turn ~10x (confirmed: removing the
        # per-player heartbeat made the OG match the server 1.00). In steady state (no
        # transform to deliver) drop that record and send local_state HUD only, so the OG
        # predicts its turn freely; corrections (pos/rot present) still deliver via the
        # entity path. py does not reconcile to this record (sustained drift unchanged).
        _goal7_drop_stomp_entity = (
            os.environ.get("WULFRAM_GOAL7_LEGACY") != "1"
            and rot is None
            and pos is None
        )
        if (
            self._wf_remote_heartbeat_entity_mode(ctx)
            and not is_view_update
            and not _goal7_drop_stomp_entity
        ):
            remote_pos = pos
            remote_vel = None
            remote_rot = rot
            if remote_rot is None and self.heartbeat_include_rot:
                # Promoted remote heartbeats should keep body rotation live so
                # the OG client does not zero angular velocity, but they should
                # not silently turn into full position corrections unless the
                # caller explicitly requested transform fields.
                remote_rot = self._local_player_sync_rotation(ctx)
            if remote_pos is not None:
                remote_vel = ctx.player_vel
            return self._build_remote_sync_heartbeat_update(
                ctx,
                tick=tick,
                pos=remote_pos,
                vel=remote_vel,
                rot=remote_rot,
                include_vel=remote_vel is not None,
                include_rot=remote_rot is not None,
                include_local_state=include_local_state,
                health=health,
                fuel=fuel,
                weapon_type=weapon_type,
                ammo_bits=ammo_bits,
                ammo_mask=ammo_mask,
                pt_bits=pt_bits,
                pt_angle=pt_angle,
                st_bits=st_bits,
                st_angle=st_angle,
            )

        use_local_entity_when_no_transform = False
        include_entities = True
        if minimal_remote:
            include_entities = False

        # GOAL 7: the periodic heartbeat's dummy entity (0xFFFFFFFE, mask=0) makes the
        # OG client ZERO its predicted angular velocity every ~100ms (confirmed live:
        # with the per-player heartbeat off the OG matched the server 1.00 vs ~13°/147°
        # with it on; the server torque path is faithful — Vehicles.c:1425 raw yaw —
        # so the OG's real rate equals the server and the under-rotation was this stomp).
        # Send local_state HUD with ZERO entities (no dummy to stomp) when there is no
        # transform to deliver, so the OG predicts its turn freely. py does not reconcile
        # to a dummy mask=0 entity, and remote-entity visibility uses a separate path, so
        # this is universal (no client fork) and preserves HUD + remotes. Corrections
        # (which carry pos/rot) keep their entity record. A/B: WULFRAM_GOAL7_LEGACY=1.
        _has_transform = (rot is not None) or (pos is not None)
        if os.environ.get("WULFRAM_GOAL7_LEGACY") != "1" and not _has_transform:
            include_entities = False

        return build_update_array_heartbeat(
            tick=tick,
            entity_id=entity_id,
            include_health=include_local_state,
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
            is_view_update=is_view_update,
            include_entities=include_entities,
            use_local_entity_when_no_transform=use_local_entity_when_no_transform,
            rot=rot,
            pos=pos,
        )

    def _og_viewer_replication_enabled(self, target_ctx: ClientContext, stream: str) -> bool:
        """Return whether a replication stream is enabled for an OG/remote viewer."""
        if handlers._is_loopback_client(target_ctx):
            return True
        if stream == "roster":
            return bool(getattr(self, "og_viewer_roster_entry", True))
        if stream == "entity_create":
            return bool(getattr(self, "og_viewer_entity_create", True))
        if stream == "remote_updates":
            return bool(getattr(self, "og_viewer_remote_updates", True))
        return True

    def _send_roster_entry(self, target_ctx: ClientContext, player_ctx: ClientContext) -> None:
        """Send ADD_TO_ROSTER for player_ctx to target_ctx (once)."""
        if not self._og_viewer_replication_enabled(target_ctx, "roster"):
            return
        player_id = player_ctx.session.player_id or player_ctx.entity_id
        if player_id in target_ctx.known_roster_ids:
            return
        name = player_ctx.session.username or f"Player{player_ctx.client_id}"
        team = player_ctx.session.team_id or 1
        payload = build_add_to_roster(
            player_id=player_id,
            entity_id=player_id,
            name=name,
            team=team,
            kills=player_ctx.kills,
            deaths=player_ctx.deaths,
        )
        if not self._send_packet_to_client(
            target_ctx,
            payload,
            prefer_tcp=True,
            allow_udp_fallback=False,
        ):
            return
        target_ctx.known_roster_ids.add(player_id)
        print(f"[MULTI] Sent roster {name} (id={player_id}) -> client {target_ctx.client_id}")

    def _broadcast_roster_presence(self, ctx: ClientContext) -> None:
        """Exchange roster entries between ctx and every other LOGGED-IN client.

        Player presence is announced at login/team-select (not spawn), so each
        client sees who is connected before anyone deploys. Idempotent:
        _send_roster_entry short-circuits on known_roster_ids, so calling this
        every game-loop turn is a no-op after the first exchange (until a team
        change invalidates the cached entry).
        """
        if not ctx.session.login_complete:
            return
        for other in self._snapshot_logged_in_clients():
            if other is ctx:
                continue
            # Only roster a player once they HAVE a team, so ADD_TO_ROSTER
            # carries the correct team from the start and the row never needs a
            # later REMOVE (which would free the name string mid-render -> crash;
            # see _invalidate_roster_for_player). Players still at team-select
            # (team_id 0) appear as soon as they pick a team / spawn.
            if other.session.team_id:
                self._send_roster_entry(ctx, other)
            if ctx.session.team_id:
                self._send_roster_entry(other, ctx)

    def _invalidate_roster_for_player(self, player_ctx: ClientContext) -> None:
        """Update player_ctx's roster TEAM in place after a (re)spawn / team
        change -- via UPDATE_STATS (0x1C), NEVER remove-then-add.

        REMOVE_FROM_ROSTER makes the OG client FREE the player's name/clan
        strings (PlayerEntry_remove, 0x475d40); the HUD then renders the freed
        string -> use-after-free (observed live: APPCRASH 0xc0000005 in
        HUD_render_text_dispatch right after a respawn, both clients down). And
        re-adding duplicates the row (ADD_TO_ROSTER CREATES a PlayerEntry, it
        does not update). UPDATE_STATS updates the EXISTING entry's team in
        place (PlayerEntry_update_stats, 0x475ec0) with no free and no
        duplicate. Sent only to clients that already hold the entry; others get
        a correct-team ADD from the next presence broadcast."""
        player_id = player_ctx.session.player_id or player_ctx.entity_id
        team = player_ctx.session.team_id
        if not player_id or not team:
            return
        payload = build_update_stats(
            player_id=player_id,
            entity_id=player_id,
            kills=player_ctx.kills,
            deaths=player_ctx.deaths,
            team_id=team,
        )
        for c in self._snapshot_clients():
            if player_id in c.known_roster_ids and self._og_viewer_replication_enabled(c, "roster"):
                self._send_packet_to_client(
                    c, payload, prefer_tcp=True, allow_udp_fallback=False
                )

    def _broadcast_roster_removal(self, player_ctx: ClientContext) -> None:
        """Tell every other client to drop player_ctx from the scoreboard
        (REMOVE_FROM_ROSTER 0x1B) on disconnect, instead of leaving a stale row."""
        player_id = player_ctx.session.player_id or player_ctx.entity_id
        if not player_id:
            return
        payload = build_remove_from_roster(player_id)
        for c in self._snapshot_clients():
            if c is player_ctx:
                continue
            if not self._og_viewer_replication_enabled(c, "roster"):
                continue
            c.known_roster_ids.discard(player_id)
            self._send_packet_to_client(c, payload, prefer_tcp=True, allow_udp_fallback=False)

    def _broadcast_player_stats(
        self,
        player_ctx: ClientContext,
        *,
        participants: tuple[ClientContext, ...] = (),
    ) -> None:
        """Send UPDATE_STATS for player_ctx to all connected clients."""
        player_id = player_ctx.session.player_id or player_ctx.entity_id
        entity_id = player_ctx.entity_id
        team = player_ctx.session.team_id or 1
        pkt = build_update_stats(
            player_id=player_id,
            entity_id=entity_id,
            kills=player_ctx.kills,
            deaths=player_ctx.deaths,
            team_id=team,
        )
        for client in self._snapshot_in_game_clients():
            if not self._combat_observer_packets_allowed_for_client(client, *participants):
                continue
            self._send_packet_to_client(
                client,
                pkt,
                prefer_tcp=True,
                allow_udp_fallback=False,
            )

    def _send_entity_create(self, target_ctx: ClientContext, player_ctx: ClientContext, *, is_retry: bool = False) -> None:
        """Announce player_ctx's entity to target_ctx via UPDATE_ARRAY DEFINITION.

        Remote entities are created via UPDATE_ARRAY's fallback path:
          1. OIDTable_lookup(entity_id) â†’ NOT FOUND
          2. Entity_create_from_network() creates entity + sets team/config
          3. Entity_apply_network_transform: entity+0xAC=0 â†’ copies position â†’
             Entity_toggle_static_state() â†’ Entity_set_transform() â†’ visible!

        The client's Replication.c has a global tick guard
        (g_tick_count < g_next_periodic_send_tick) that may block
        Entity_apply_network_transform for newly connected clients.
        To work around this, we resend entity creation packets for a
        retry period after the first attempt, so that one eventually
        arrives when the guard is open.
        """
        if not self._og_viewer_replication_enabled(target_ctx, "entity_create"):
            return
        if not target_ctx.session.translation_ack_received:
            return
        entity_id = player_ctx.session.entity_id or player_ctx.entity_id

        # Track entity creation timing for retry logic.
        if not hasattr(target_ctx, '_entity_create_times'):
            target_ctx._entity_create_times = {}  # entity_id â†’ (first_send_time, last_send_time)

        now = time.monotonic()
        # Fast retries for the first N seconds, then slow periodic re-announce.
        # The client's tick guard (g_tick_count < g_next_periodic_send_tick) can
        # block Entity_apply_network_transform indefinitely.  DEFINITION packets
        # bypass the guard via Entity_create_from_network (entity+0xAC=0 path),
        # so periodic re-announce keeps entities visible even after the guard
        # activates.
        fast_window = float(os.environ.get("WULFRAM_ENTITY_CREATE_RETRY_SECS", "10"))
        fast_interval = float(os.environ.get("WULFRAM_ENTITY_CREATE_RETRY_INTERVAL", "0.5"))
        slow_interval = float(os.environ.get("WULFRAM_ENTITY_REANNOUNCE_INTERVAL", "5.0"))

        if entity_id in target_ctx.known_entity_ids:
            # Already sent at least once.  Check if we should retry.
            times = target_ctx._entity_create_times.get(entity_id)
            if times is None:
                return  # No tracking info â€” legacy, don't retry.
            first_send, last_send = times
            in_fast_window = (now - first_send) <= fast_window
            if not in_fast_window:
                return  # Past fast window â€” entity is created, UPDATE packets handle sync.
                # NOTE: Slow re-announce was REMOVED because build_update_array_create_tank
                # sends hardcoded rotation=(0,0,0) and no velocity, causing the remote entity
                # to visually snap to heading=0 every re-announce interval (glitching).
            if (now - last_send) < fast_interval:
                return  # Too soon since last retry.
            is_retry = True

        pos = self._to_client_pos(player_ctx.player_pos)
        team = player_ctx.session.team_id or 1
        tick = self._get_network_tick(target_ctx)
        rot = self._player_body_rotation(
            player_ctx,
            negate_yaw=self.remote_yaw_negate,
            yaw_offset=self.remote_yaw_offset,
        )

        include_local_state, ls = self._get_update_array_local_state_for_viewer(target_ctx)
        create_pkt = build_update_array_create_tank(
            tick=tick,
            entity_id=entity_id,
            entity_type=player_ctx.entity_type,
            team=team,
            pos=pos,
            is_manned=True,
            rot=rot,
            include_health=include_local_state,
            **ls,
        )
        label = "RETRY" if is_retry else "CREATE"
        print(f"[MULTI] UPDATE_ARRAY DEFINITION {label} id={entity_id} "
              f"type={player_ctx.entity_type} team={team} pos={pos} "
              f"tick={tick} -> client {target_ctx.client_id}")
        if not is_retry:
            print(f"[MULTI-HEX] {create_pkt.hex().upper()}")
        # OG clients are sensitive to UPDATE_ARRAY over TCP. Remote entity
        # creation already has UDP retries, so keep this path UDP-only.
        if not self._send_packet_to_client(target_ctx, create_pkt, prefer_tcp=False):
            return

        if entity_id not in target_ctx.known_entity_ids:
            target_ctx.known_entity_ids.add(entity_id)
            target_ctx._entity_create_times[entity_id] = (now, now)
            print(f"[MULTI] Sent entity create OK -> client {target_ctx.client_id}")
        else:
            first_send = target_ctx._entity_create_times[entity_id][0]
            target_ctx._entity_create_times[entity_id] = (first_send, now)
            elapsed = now - first_send
            phase = "RETRY" if elapsed <= fast_window else "REANNOUNCE"
            if phase == "RETRY" or int(elapsed) % 30 == 0:  # Log retries always, re-announces every 30s
                print(f"[MULTI] Sent entity create {phase} -> client {target_ctx.client_id} ({elapsed:.1f}s since first)")

    def _ensure_multiplayer_visibility(self, ctx: ClientContext) -> None:
        """Ensure ctx sees other players and vice versa once translation is ready."""
        if not ctx.session.translation_ack_received:
            return
        self._ensure_uplink_mvp_state(ctx)
        others = [c for c in self._snapshot_in_game_clients() if c is not ctx]
        if ctx.session.tick % 300 == 0 and others:
            print(f"[MULTI-DBG] client={ctx.client_id} trans_ack={ctx.session.translation_ack_received} "
                  f"others={[(o.client_id, o.session.entity_id or o.entity_id, o.session.translation_ack_received) for o in others]} "
                  f"known={ctx.known_entity_ids}")
        for other in others:
            # Always try to exchange roster entries (idempotent).
            self._send_roster_entry(ctx, other)
            if other.session.translation_ack_received:
                self._send_roster_entry(other, ctx)
            # Entity creation requires target translation ack.
            self._send_entity_create(ctx, other)
            if other.session.translation_ack_received:
                self._send_entity_create(other, ctx)

    def _build_remote_sync_heartbeat_update(
        self,
        ctx: ClientContext,
        *,
        tick: int,
        pos: Optional[tuple[float, float, float]] = None,
        vel: Optional[tuple[float, float, float]] = None,
        rot: Optional[tuple[float, float, float]] = None,
        include_vel: bool,
        include_rot: bool,
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
        safe_local_state: bool = True,
    ) -> bytes:
        """Build a safe promoted remote heartbeat/correction update.

        Loopback/Python heartbeats respect the caller's requested transform
        fields. Remote OG local-player updates must not send partial transform
        records: the decompiled transform applier can consume unpopulated
        position fields when rotation is present, poisoning the local entity
        with NaNs. For OG, any transform-bearing heartbeat is expanded to
        position + velocity + rotation.
        """
        if not handlers._is_loopback_client(ctx):
            wants_transform = pos is not None or vel is not None or rot is not None or include_vel or include_rot
            if wants_transform:
                if pos is None and getattr(ctx, "player_pos", None) is not None:
                    pos = self._to_client_pos(ctx.player_pos)
                if pos is not None:
                    if vel is None:
                        vel = ctx.player_vel
                    if rot is None:
                        rot = self._local_player_sync_rotation(ctx)
                    include_vel = True
                    include_rot = True
                else:
                    # The OG local-player transform applier is not safe with a
                    # partial interpolation record. If no finite position is
                    # available, fall back to a local-state heartbeat only.
                    vel = None
                    rot = None
                    include_vel = False
                    include_rot = False

        entity_id = ctx.session.entity_id or ctx.entity_id
        has_pos = pos is not None
        has_vel = include_vel and vel is not None
        has_rot = include_rot and rot is not None
        hb_rot = rot if rot is not None else self._local_player_sync_rotation(ctx)
        hb_pos = pos if pos is not None else self._to_client_pos(ctx.player_pos)
        hb_vel = vel if vel is not None else ctx.player_vel
        local_weapon_type = self._get_spawn_tank_weapon_type(ctx) if safe_local_state else weapon_type
        local_ammo_bits = 0 if safe_local_state else ammo_bits
        local_ammo_mask = 0 if safe_local_state else ammo_mask
        local_pt_bits = 0 if safe_local_state else pt_bits
        local_pt_angle = 0.0 if safe_local_state else pt_angle
        local_st_bits = 0 if safe_local_state else st_bits
        local_st_angle = 0.0 if safe_local_state else st_angle
        return build_update_array_player_update(
            tick=tick,
            entity_id=entity_id,
            pos=hb_pos,
            vel=hb_vel,
            rot=hb_rot,
            include_pos=has_pos,
            include_vel=has_vel,
            include_rot=has_rot,
            # GOAL 7: optionally carry the real yaw spin in every local heartbeat (not
            # only when rotation is included) — tested as a hypothesis that the OG client
            # zeros its ang_vel on a spin-less local UPDATE_ARRAY ("heartbeat mask=0 zeros
            # angular velocity"). TESTED 2026-06-03: had NO effect on the OG under-rotation
            # (server 150.7° / OG 13.5° unchanged with WULFRAM_HEARTBEAT_SPIN=1), so the
            # stomp hypothesis is NOT the cause; default OFF to keep heartbeat behavior
            # unchanged. Left as an A/B lever.
            include_spin=(has_rot or os.environ.get("WULFRAM_HEARTBEAT_SPIN", "0") == "1"),
            spin=(0.0, 0.0, 0.0) if os.environ.get("WULFRAM_GOAL6_LEGACY") == "1" else (0.0, 0.0, ctx.angular_vel_yaw),
            include_local_state=include_local_state,
            include_entity_vitals=False,
            weapon_id=local_weapon_type,
            health=health,
            fuel=fuel,
            ammo_count_bits=local_ammo_bits,
            ammo_count=local_ammo_mask,
            primary_turret_bits=local_pt_bits,
            primary_turret_angle=local_pt_angle,
            secondary_turret_bits=local_st_bits,
            secondary_turret_angle=local_st_angle,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
            is_manned=True,
            speed_scale=1.0,
        )

    def _build_remote_spawn_bootstrap_heartbeat(
        self,
        ctx: ClientContext,
        *,
        tick: int,
        entity_id: Optional[int] = None,
        health: float,
        fuel: float,
    ) -> bytes:
        """Build the one-off post-spawn OG heartbeat on the full-transform shape.

        Fresh remote OG spawns still benefit from a single local-state sync
        packet immediately after Tank/PLAYER_INFO, but the old minimal
        no-entity heartbeat is fragile in that bootstrap window. Reuse the
        promoted remote heartbeat layout; the remote OG guard expands the body
        rotation request into a complete position + velocity + rotation update
        so the client's interpolation record is fully populated.
        """
        if entity_id is None:
            entity_id = ctx.session.entity_id or ctx.entity_id
        saved_entity_id = ctx.session.entity_id
        if saved_entity_id != entity_id:
            ctx.session.entity_id = entity_id
        try:
            return self._build_remote_sync_heartbeat_update(
                ctx,
                tick=tick,
                rot=self._local_player_sync_rotation(ctx),
                include_vel=False,
                include_rot=True,
                include_local_state=True,
                health=health,
                fuel=fuel,
                weapon_type=self._get_spawn_tank_weapon_type(ctx),
                ammo_bits=0,
                ammo_mask=0,
                pt_bits=0,
                pt_angle=0.0,
                st_bits=0,
                st_angle=0.0,
                safe_local_state=True,
            )
        finally:
            ctx.session.entity_id = saved_entity_id

    def _get_update_array_local_state_for_viewer(self, ctx: ClientContext) -> tuple[bool, dict]:
        """Return viewer-local local_state args for non-heartbeat UPDATE_ARRAY packets.

        The OG client clears its local input/local-state scratch at the start of
        every UPDATE_ARRAY, then runs local-player sync at the end of the
        packet. Remote OG viewers therefore still need a valid local-state
        prefix even on entity-only updates. Promoted remote viewers still
        reject the fully expanded tank local-state on packets that do not also
        carry the local-player sync entity block, so keep these on the same
        short-form-safe shape as the projectile/update-array compatibility path.
        Only genuine Python test clients keep the entity-only path; a real OG
        client (including one on loopback, e.g. play.bat) must get the local-state
        prefix or its end-of-packet local-player sync reads garbage health.
        """
        if not handlers._is_og_client(ctx):
            return False, {}

        return True, dict(
            weapon_id=self._get_spawn_tank_weapon_type(ctx),
            health=self._get_health_value(ctx),
            fuel=self._get_energy_value(ctx),
            ammo_count_bits=0,
            ammo_count=0,
            primary_turret_bits=0,
            primary_turret_angle=0.0,
            secondary_turret_bits=0,
            secondary_turret_angle=0.0,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
        )

    def _get_projectile_local_state_for_viewer(self, ctx: ClientContext) -> tuple[bool, dict]:
        """Return viewer-local local_state args for projectile UPDATE_ARRAY packets.

        OG clients still need a valid local-state prefix on projectile packets,
        but the fully expanded local-state form is fragile here because the
        projectile bitstream does not also carry the local-player entity sync
        block that the promoted heartbeat path uses. Keep remote projectile
        packets on the same short-form-safe local-state shape as spawn-time
        heartbeats even after the viewer has been promoted to full sync.

        Gate on OG-vs-Python (not loopback) so a loopback OG client (play.bat)
        still gets the local-state prefix it needs.
        """
        if not handlers._is_og_client(ctx):
            return False, {}

        return True, dict(
            weapon_id=self._get_spawn_tank_weapon_type(ctx),
            health=self._get_health_value(ctx),
            fuel=self._get_energy_value(ctx),
            ammo_count_bits=0,
            ammo_count=0,
            primary_turret_bits=0,
            primary_turret_angle=0.0,
            secondary_turret_bits=0,
            secondary_turret_angle=0.0,
            turret_max=self.local_state_turret_max,
            turret_range=self.local_state_turret_range,
        )
