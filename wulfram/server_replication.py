"""ReplicationMixin -- local-state payload construction for UPDATE_ARRAY
replication, extracted verbatim from WulframServer (server.py decomposition,
step 3). Method-only mixin; shares state via `self`. First chunk of the
replication layer (the local-state heartbeat builders).
"""
from __future__ import annotations

import os
from typing import Optional

from . import handlers
from .client import ClientContext
from .packets import build_update_array_heartbeat
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
