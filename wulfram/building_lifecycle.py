"""Dynamic-building lifecycle helpers used by the server facade."""

from __future__ import annotations

import math
import time
from typing import Any

from .packets import build_delete_object, get_ticks
from .weapons import EntityType


def remember_event(server: object, event: dict[str, Any]) -> dict[str, Any]:
    events = getattr(server, "_building_lifecycle_events", None)
    if events is None:
        events = []
        server._building_lifecycle_events = events
    events.append(event)
    del events[:-100]
    return event


def base_event(server: object, oid: int, action: str) -> dict[str, Any]:
    building = server._building_entities.get(oid)
    entity_type = int(getattr(building, "entity_type", -1) or -1) if building else -1
    try:
        entity_type_name = EntityType(entity_type).name
    except ValueError:
        entity_type_name = str(entity_type)
    max_health = float(server._building_max_health.get(oid, 0.0) or 0.0)
    health = float(server._building_health.get(oid, 0.0) or 0.0)
    return {
        "time": time.time(),
        "action": action,
        "oid": int(oid),
        "entity_type": entity_type,
        "entity_type_name": entity_type_name,
        "team_id": int(getattr(building, "team_id", 0) or 0) if building else 0,
        "pos": [round(float(v), 5) for v in building.pos] if building else None,
        "health": health,
        "max_health": max_health,
        "health_pct": round((health / max_health * 100.0), 3) if max_health > 0.0 else None,
        "dynamic": int(oid) in getattr(server, "_dynamic_building_ids", set()),
        "dynamic_source": getattr(server, "_dynamic_building_sources", {}).get(int(oid), {}) or {},
    }


def broadcast_delete(
    server: object,
    oid: int,
    *,
    prefer_tcp: bool = True,
    participants: tuple[Any, ...] | None = None,
) -> int:
    packet = build_delete_object(get_ticks(), [int(oid)], with_effects=True)
    sent = 0
    for target in server._snapshot_in_game_clients():
        if participants is not None and not server._combat_observer_packets_allowed_for_client(target, *participants):
            continue
        if server._send_packet_to_client(target, packet, prefer_tcp=prefer_tcp):
            target.known_entity_ids.discard(int(oid))
            sent += 1
    return sent


def remove_dynamic_record(server: object, oid: int) -> None:
    building = server._building_entities.get(oid)
    source = server._dynamic_building_sources.get(oid, {}) or {}
    team_id = int(getattr(building, "team_id", 0) or 0) if building else 0
    slot = source.get("slot")
    ship = server._uplink_ships.get(team_id) if team_id else None
    if ship is not None and slot is not None:
        cargo = list(ship.get("cargo", [40, 40, 40, 40]))
        try:
            slot_index = int(slot)
        except (TypeError, ValueError):
            slot_index = -1
        if 0 <= slot_index < len(cargo):
            cargo[slot_index] = 40
            ship["cargo"] = cargo
            server._broadcast_uplink_ship_info(ship)
    server._building_entities.pop(oid, None)
    server._building_health.pop(oid, None)
    server._building_max_health.pop(oid, None)
    server._dynamic_building_ids.discard(oid)
    server._dynamic_building_sources.pop(oid, None)
    server._rebuild_static_world_raycast_index()


def apply_damage_amount(
    server: object,
    oid: int,
    damage: float,
    *,
    source: str,
    remove_dynamic_on_destroy: bool = True,
    delete_participants: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Apply absolute building HP damage and record demo/audit evidence."""
    oid = int(oid)
    if oid not in server._building_entities:
        return {"ok": False, "error": "unknown building", "oid": oid}
    if oid not in server._building_health:
        return {"ok": False, "error": "building has no health", "oid": oid}
    try:
        damage = float(damage)
    except (TypeError, ValueError):
        damage = 0.0
    if not math.isfinite(damage) or damage <= 0.0:
        return {"ok": False, "error": "damage must be positive", "oid": oid}

    old_hp = float(server._building_health.get(oid, 0.0) or 0.0)
    if old_hp <= 0.0:
        event = base_event(server, oid, "damage_ignored")
        event.update({"ok": False, "source": source, "reason": "already_destroyed"})
        remember_event(server, event)
        return event

    new_hp = max(0.0, old_hp - damage)
    server._building_health[oid] = new_hp
    event = base_event(server, oid, "destroy" if new_hp <= 0.0 else "damage")
    event.update(
        {
            "ok": True,
            "source": source,
            "damage": damage,
            "old_health": old_hp,
            "new_health": new_hp,
            "destroyed": new_hp <= 0.0,
            "delete_sent": 0,
            "removed": False,
        }
    )
    if new_hp <= 0.0:
        event["delete_sent"] = broadcast_delete(
            server,
            oid,
            prefer_tcp=True,
            participants=delete_participants,
        )
        if remove_dynamic_on_destroy and oid in server._dynamic_building_ids:
            remove_dynamic_record(server, oid)
            event["removed"] = True
    remember_event(server, event)
    return event

