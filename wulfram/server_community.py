"""CommunityWireMixin -- upstream wulfram3.com (2026-10-16) wire-profile emitters.

Every behaviour here is opt-in (``server_config.init_upstream_1016_profile``); with
the knobs at their defaults nothing in this module puts a byte on the wire and
``wulfram.dock_pin.should_pin`` is never called.

Evidence: D:\\work\\upstream-1016\\sync\\RESULT.md (Q2, Q4) and the fixture
``server/test_data/upstream_wire_1016.json`` (real s2c port-2627 messages).
"""
from __future__ import annotations

import math
import time
from typing import Optional

from . import handlers
from .client import ClientContext
from .dock_pin import DockPinInputs
from . import dock_pin
from .packets import (
    ENERGY_MAX,
    ENERGY_RANGE,
    build_docking,
    build_view_update_multi,
    encode_health_bits,
)


class CommunityWireMixin:
    # ------------------------------------------------------------------ helpers

    def _community_local_state(self, ctx: ClientContext) -> tuple[bool, dict]:
        kwargs = self._get_local_state_kwargs(ctx)
        include = self._should_send_local_state(
            ctx, kwargs["primary_turret_bits"], kwargs["secondary_turret_bits"], self.update_local_state_mode,
        )
        if not include:
            kwargs = dict(kwargs, ammo_count_bits=0, ammo_count=0, primary_turret_bits=0,
                          secondary_turret_bits=0)
        return include, kwargs

    def _community_ammo_record_supported(self, ctx: ClientContext) -> bool:
        """Mask bit 6 is only encodable for the Tank row with 9 active slots (shared encoder)."""
        if int(getattr(ctx, "entity_type", -1)) != 0:
            return False
        caps = getattr(self, "behavior_weapon_caps", None) or []
        return bool(caps) and int(caps[0][2]) == 9

    def _local_vitals_snapshot(self, ctx: ClientContext) -> dict:
        health = float(self._get_health_value(ctx))
        fuel = float(self._get_energy_value(ctx))
        snap = {
            "health": health,
            "fuel": fuel,
            "h": encode_health_bits(health, total_bits=10),
            "f": encode_health_bits(fuel, total_bits=10, max_val=ENERGY_MAX, range_val=ENERGY_RANGE),
            "ammo": None,
            "target": getattr(ctx, "local_target_id", None),
        }
        if self._community_ammo_record_supported(ctx):
            snap["ammo"] = 1 if self._tank_primary_replication_active(ctx) else 0
        return snap

    def _local_vitals_record_fields(self, ctx: ClientContext, *, floor: bool = False) -> tuple[dict, dict, bool]:
        """(entity-record kwargs for mask bits 5-8, snapshot, any_dirty).

        Only the bits whose quantised value changed since the last record sent to
        this client (upstream's dirty-field policy). ``floor`` forces 5+7.
        """
        snap = self._local_vitals_snapshot(ctx)
        last = getattr(ctx, "_local_vu_last", None) or {}
        health_dirty = last.get("h") != snap["h"]
        fuel_dirty = last.get("f") != snap["f"]
        ammo_dirty = snap["ammo"] is not None and last.get("ammo") != snap["ammo"]
        target_dirty = snap["target"] is not None and last.get("target") != snap["target"]
        any_dirty = health_dirty or fuel_dirty or ammo_dirty or target_dirty
        if floor and not any_dirty:
            health_dirty = fuel_dirty = True
        fields = dict(
            include_health=health_dirty,
            include_fuel=fuel_dirty,
            speed_scale=snap["health"],
            fuel=snap["fuel"],
            ammo_unit=0 if ammo_dirty else None,
            ammo_active_bits=9 if ammo_dirty else 0,
            ammo_active_mask=int(snap["ammo"] or 0) if ammo_dirty else 0,
            target_id=snap["target"] if target_dirty else None,
        )
        return fields, snap, any_dirty

    @staticmethod
    def _commit_local_vitals(ctx: ClientContext, snap: dict) -> None:
        ctx._local_vu_last = {"h": snap["h"], "f": snap["f"], "ammo": snap["ammo"], "target": snap["target"]}

    def _local_view_update(self, ctx: ClientContext, tick: int, record: dict) -> bytes:
        include_local_state, ls = self._community_local_state(ctx)
        entity = dict(
            entity_id=int(ctx.session.entity_id or ctx.entity_id),
            is_manned=True,
            pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0),
            include_pos=False, include_vel=False, include_rot=False,
        )
        entity.update(record)
        return build_view_update_multi(
            tick,
            include_local_state=include_local_state,
            entities=[entity],
            timestamp=self._fresh_remote_view_update_timestamp(ctx, tick),
            **ls,
        )

    def _send_udp(self, ctx: ClientContext, payload: bytes) -> bool:
        if not payload or not self.udp_handler or not ctx.session or not ctx.session.udp_addr:
            return False
        self.udp_handler.send_to(payload, ctx.session.udp_addr)
        return True

    def _dock_grounded(self, ctx: ClientContext) -> bool:
        """Server-side 'resting on something' proxy: |vertical velocity| small.

        There is no explicit ground-contact flag on the server context; the tank
        integrator's up-axis velocity is the closest signal.
        """
        vel = tuple(getattr(ctx, "player_vel", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
        up = 2 if getattr(self, "up_axis", "z") == "z" else 1
        try:
            vz = float(vel[up])
        except (IndexError, TypeError, ValueError):
            return False
        return math.isfinite(vz) and abs(vz) <= float(getattr(self, "dock_grounded_vz", 1.0))

    def _horizontal_speed(self, ctx: ClientContext) -> float:
        vel = tuple(getattr(ctx, "player_vel", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
        up = 2 if getattr(self, "up_axis", "z") == "z" else 1
        comps = [float(v) for i, v in enumerate(vel[:3]) if i != up]
        return math.hypot(*comps) if comps else 0.0

    @staticmethod
    def _move_input_active(ctx: ClientContext) -> bool:
        decoded = getattr(ctx, "last_decoded_input", None) or {}
        try:
            return (
                abs(float(decoded.get("fwd", 0.0) or 0.0)) > 0.05
                or abs(float(decoded.get("strafe", 0.0) or 0.0)) > 0.05
                or float(decoded.get("jumpjet", 0.0) or 0.0) > 0.5
                or float(decoded.get("thrust", 0.0) or 0.0) > 0.5
            )
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------ (a) local vitals VIEW_UPDATE

    def _build_community_local_vu(self, ctx: ClientContext, tick: int, now: Optional[float] = None) -> Optional[bytes]:
        """WULFRAM_LOCAL_VU_ENTITY heartbeat; None = nothing to send this tick."""
        if now is None:
            now = time.monotonic()
        if getattr(ctx, "_dock_pin_tick", None) == tick:
            return None  # the pin record already carried this tick's dirty bits
        active = float(getattr(self, "local_state_active_interval_s", 0.05))
        idle = float(getattr(self, "local_state_idle_interval_s", 1.0))
        last = float(getattr(ctx, "_local_vu_last_send", 0.0) or 0.0)
        if last and (now - last) < active:
            return None
        floor = (not last) or (now - last) >= idle
        fields, snap, dirty = self._local_vitals_record_fields(ctx, floor=floor)
        if not dirty and not floor:
            return None
        payload = self._local_view_update(ctx, tick, fields)
        self._commit_local_vitals(ctx, snap)
        ctx._local_vu_last_send = now
        return payload

    # ------------------------------------------------------- (d) ship batches

    def _community_ship_status_step(self, ctx: ClientContext, now: float) -> int:
        interval = float(getattr(self, "ship_status_interval_s", 0.0) or 0.0)
        repeat = float(getattr(self, "ship_status_repeat_s", 0.0) or 0.0)
        if interval > 0 and (now - float(getattr(ctx, "last_ship_status_send", 0.0) or 0.0)) >= interval:
            ctx.last_ship_status_send = now
            ctx._ship_status_repeat_at = (now + repeat) if repeat > 0 else 0.0
            return self._send_periodic_ship_status(ctx)
        repeat_at = float(getattr(ctx, "_ship_status_repeat_at", 0.0) or 0.0)
        if repeat_at and now >= repeat_at:
            ctx._ship_status_repeat_at = 0.0
            return self._send_periodic_ship_status(ctx)
        return 0

    # ------------------------------------------------------ (c) DOCKING stream

    def _community_docking_step(self, ctx: ClientContext, now: float, tick: int) -> None:
        if not getattr(self, "docking_stream", False):
            return
        if getattr(ctx, "_dock_pin_state", None) is not None:
            return  # the pin owns DOCKING while it runs
        pad = getattr(ctx, "supply_pad_oid", None) if self._dock_grounded(ctx) else None
        current = getattr(ctx, "_docking_stream_target", None)
        interval = float(getattr(self, "docking_interval_s", 0.05))
        if pad:
            last = float(getattr(ctx, "_docking_stream_last", 0.0) or 0.0)
            if current != pad or (now - last) >= interval:
                self._send_udp(ctx, build_docking(tick, int(pad), True))
                ctx._docking_stream_target = int(pad)
                ctx._docking_stream_last = now
        elif current is not None:
            self._send_udp(ctx, build_docking(tick, int(current), False))
            ctx._docking_stream_target = None

    # --------------------------------------------------------------- dock pin

    def _dock_pin_client_allowed(self, ctx: ClientContext) -> bool:
        allowed = getattr(self, "dock_pin_clients", frozenset()) or frozenset()
        if not allowed:
            return False
        if "*" in allowed:
            return True
        host = None
        addr = getattr(ctx.session, "udp_addr", None) if ctx.session else None
        if addr:
            host = addr[0]
        if host is None:
            host = getattr(ctx, "addr", (None,))[0] if getattr(ctx, "addr", None) else None
        return host in allowed

    def _dock_pin_step(self, ctx: ClientContext, now: float, tick: int) -> bool:
        """Run the dock pin for one tick; True when a pin record went out this tick."""
        if not getattr(self, "dock_pin_enabled", False):
            return False
        if not ctx.session or not ctx.session.in_game or not ctx.session.entity_id:
            ctx._dock_pin_state = None
            return False
        move = self._move_input_active(ctx)
        if move:
            ctx._dock_pin_last_move_input = now
        last_move = float(getattr(ctx, "_dock_pin_last_move_input", 0.0) or 0.0)
        state = getattr(ctx, "_dock_pin_state", None)
        cur_pos = tuple(self._to_client_pos(ctx.player_pos))
        divergence = math.dist(state["pos"], cur_pos) if state else 0.0
        enabled = self._dock_pin_client_allowed(ctx)
        grounded = self._dock_grounded(ctx)
        target = int(getattr(ctx, "supply_pad_oid", None) or 0) if grounded else 0
        inputs = DockPinInputs(
            enabled_for_client=enabled,
            grounded=grounded,
            speed=self._horizontal_speed(ctx),
            seconds_since_thrust=(now - last_move) if last_move else math.inf,
            move_input_active=move,
            pinned=state is not None,
            divergence_pos=divergence,
            is_native_client=handlers.is_native_stream_client(ctx),
            dock_target=state["target"] if state else target,
        )
        pin = bool(dock_pin.should_pin(inputs)) and enabled
        if pin:
            if state is None:
                state = {
                    "pos": cur_pos,
                    "rot": tuple(self._local_player_sync_rotation(ctx)),
                    "target": target,
                    "last": 0.0,
                }
                ctx._dock_pin_state = state
            if state["last"] and (now - state["last"]) < float(getattr(self, "dock_pin_interval_s", 0.05)):
                ctx._dock_pin_tick = tick  # upstream: at most one local VU per tick, the pin's
                return False
            fields, snap, _dirty = self._local_vitals_record_fields(ctx)
            record = dict(fields, pos=state["pos"], rot=state["rot"], vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0),
                          include_pos=True, include_vel=True, include_rot=True, include_spin=True,
                          include_reset=True)
            self._send_udp(ctx, self._local_view_update(ctx, tick, record))
            self._send_udp(ctx, build_docking(tick, state["target"], True))
            self._commit_local_vitals(ctx, snap)
            state["last"] = now
            ctx._dock_pin_tick = tick
            ctx.dock_pin_records = int(getattr(ctx, "dock_pin_records", 0) or 0) + 1
            return True
        if state is not None:
            fields, snap, _dirty = self._local_vitals_record_fields(ctx)
            record = dict(fields, vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0),
                          include_vel=True, include_spin=True)
            self._send_udp(ctx, self._local_view_update(ctx, tick, record))
            self._send_udp(ctx, build_docking(tick, state["target"], False))
            self._commit_local_vitals(ctx, snap)
            ctx._dock_pin_state = None
            ctx._dock_pin_tick = tick
            return True
        return False

    # ------------------------------------------------------------ tick entry

    def _community_wire_step(self, ctx: ClientContext, now: float, tick: int) -> None:
        """Once per tick-loop iteration, before the local heartbeat decision."""
        self._community_ship_status_step(ctx, now)
        self._dock_pin_step(ctx, now, tick)
        self._community_docking_step(ctx, now, tick)
