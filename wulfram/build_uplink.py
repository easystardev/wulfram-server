"""Build/Uplink dynamic entity helpers used by WulframServer."""

from __future__ import annotations

import math
import os
import shlex
import struct
import time
from typing import Any, Optional

from . import handlers
from .building_collision import BuildingEntity
from .packets import (
    build_carrying_info,
    build_ship_status,
    build_supply_ship_info,
    build_update_array_create_tank,
    build_update_stats_team_first,
    build_uplink_info,
    get_ticks,
)
from .weapons import EntityType


def decode_comm_message_request_body(body: bytes) -> dict[str, Any]:
    """Decode the body shared by TCP and reliable-UDP COMM_MESSAGE_REQUEST."""
    decoded: dict[str, Any] = {
        "ok": False,
        "message_type": None,
        "flags_or_target": None,
        "text": "",
        "body_hex": body.hex(),
    }
    if len(body) < 6:
        decoded["error"] = f"body too short: {len(body)}"
        return decoded
    try:
        message_type = struct.unpack_from(">H", body, 0)[0]
        flags_or_target = struct.unpack_from(">H", body, 2)[0]
        text, offset = handlers.decode_lp_string(body, 4)
    except (struct.error, ValueError) as exc:
        decoded["error"] = str(exc)
        return decoded
    decoded.update(
        {
            "ok": True,
            "message_type": message_type,
            "flags_or_target": flags_or_target,
            "text": text,
            "end_offset": offset,
            "trailing_hex": body[offset:].hex() if offset < len(body) else "",
        }
    )
    return decoded


def parse_command(server: object, text: str) -> dict[str, Any]:
    """Parse OG type-2 starship/uplink text commands."""
    result: dict[str, Any] = {"ok": False, "text": text, "action": ""}
    try:
        parts = shlex.split(text or "")
    except ValueError as exc:
        result["error"] = f"shlex: {exc}"
        return result
    if not parts:
        result["error"] = "empty command"
        return result
    action = parts[0].lower()
    result["action"] = action
    result["parts"] = parts

    def _parse_int(value: str, field: str) -> int | None:
        try:
            return int(value, 0)
        except (TypeError, ValueError):
            result["error"] = f"invalid {field}: {value!r}"
            return None

    if action in ("build", "delete"):
        if len(parts) < 4:
            result["error"] = f"{action} requires ship_oid, entity name, and slot"
            return result
        ship_oid = _parse_int(parts[1], "ship_oid")
        slot = _parse_int(parts[3], "slot")
        if ship_oid is None or slot is None:
            return result
        entity_type = entity_type_from_name(parts[2])
        result.update(
            {
                "ok": entity_type is not None,
                "ship_oid": ship_oid,
                "entity_name": parts[2],
                "entity_type": entity_type,
                "slot": slot,
            }
        )
        if entity_type is None:
            result["error"] = f"unsupported build entity: {parts[2]!r}"
        return result

    if action == "move":
        if len(parts) < 3:
            result["error"] = "move requires ship_oid and cell"
            return result
        ship_oid = _parse_int(parts[1], "ship_oid")
        if ship_oid is None:
            return result
        result.update({"ok": True, "ship_oid": ship_oid, "cell": parts[2]})
        return result

    if action == "bomb":
        if len(parts) < 2:
            result["error"] = "bomb requires ship_oid"
            return result
        ship_oid = _parse_int(parts[1], "ship_oid")
        if ship_oid is None:
            return result
        result.update({"ok": True, "ship_oid": ship_oid})
        return result

    if action == "set":
        if len(parts) < 4:
            result["error"] = "set requires ship_oid, field, and value"
            return result
        ship_oid = _parse_int(parts[1], "ship_oid")
        value = _parse_int(parts[3], "value")
        if ship_oid is None or value is None:
            return result
        result.update({"ok": True, "ship_oid": ship_oid, "field": parts[2], "value": value})
        return result

    result["error"] = f"unsupported action: {action!r}"
    return result


def entity_type_from_name(name: str) -> Optional[int]:
    key = "".join(ch for ch in str(name or "").lower() if ch.isalnum())
    aliases = {
        "repair": EntityType.REPAIR_BUILDING,
        "repairbuilding": EntityType.REPAIR_BUILDING,
        "repairpad": EntityType.REPAIR_BUILDING,
        "fuel": EntityType.FUEL_BUILDING,
        "fuelbuilding": EntityType.FUEL_BUILDING,
        "refuel": EntityType.FUEL_BUILDING,
        "refuelpad": EntityType.FUEL_BUILDING,
        "energy": EntityType.ENERGY_BUILDING,
        "energybuilding": EntityType.ENERGY_BUILDING,
        "energypad": EntityType.ENERGY_BUILDING,
        "powercell": EntityType.ENERGY_BUILDING,
        "gun": EntityType.GUN_TURRET,
        "turret": EntityType.GUN_TURRET,
        "gunturret": EntityType.GUN_TURRET,
        "gunbuilding": EntityType.GUN_TURRET,
    }
    value = aliases.get(key)
    return int(value) if value is not None else None


def building_max_health_for_type(entity_type: int) -> float:
    max_health = {
        EntityType.GUN_TURRET: 1200.0,
        EntityType.LAUNCHER: 1200.0,
        EntityType.SENSOR_BUILDING: 1200.0,
        EntityType.FUEL_BUILDING: 2000.0,
        EntityType.REPAIR_BUILDING: 2000.0,
        EntityType.ENERGY_BUILDING: 2000.0,
        EntityType.PAD: 5000.0,
        EntityType.DARK_LIGHT: 800.0,
    }
    try:
        key = EntityType(int(entity_type))
    except ValueError:
        key = int(entity_type)
    return float(max_health.get(key, 2000.0))


def allocate_dynamic_building_oid(server: object) -> int:
    oid = int(getattr(server, "_dynamic_building_next_oid", 30000) or 30000)
    while oid in server._building_entities or oid in server._dynamic_building_ids:
        oid += 1
    server._dynamic_building_next_oid = oid + 1
    return oid


def choose_dynamic_building_pos(server: object, ctx: object, slot: int) -> tuple[float, float, float]:
    heading = float(getattr(ctx, "player_heading", 0.0) or 0.0)
    base = tuple(float(v) for v in (getattr(ctx, "player_pos", None) or (2600.0, 3040.0, 5.0)))
    distance = 35.0 + max(0, int(slot)) * 12.0
    x = base[0] + math.cos(heading) * distance
    y = base[1] + math.sin(heading) * distance
    ground_z = server._terrain_ground_z_at(x, y)
    if ground_z is None or not math.isfinite(float(ground_z)):
        z = base[2]
    else:
        z = float(ground_z)
    return (x, y, z)


def _building_terrain_conform_rot(server: object, pos, heading: float):
    """Rotation tuple (roll, pitch, yaw) for a replicated building.

    Default is flat (roll=pitch=0, yaw=heading). When
    `building_terrain_conform` is on, pitch follows the terrain slope along the
    building's facing and roll follows the slope across it, so a pad lies ON a
    slope instead of cutting through it (the "floats into terrain" report).
    Sign of roll is verified live against the OG client; flip via the negation
    here if it tilts the wrong way.
    """
    yaw = float(heading)
    if not getattr(server, "building_terrain_conform", False):
        return (0.0, 0.0, yaw)
    terrain = getattr(server, "terrain", None)
    if terrain is None:
        return (0.0, 0.0, yaw)
    try:
        x, y = float(pos[0]), float(pos[1])
        pitch = terrain.get_pitch_at_heading(x, y, yaw)
        # Roll = slope across the facing (to the building's right = yaw - 90deg).
        roll = -terrain.get_pitch_at_heading(x, y, yaw - math.pi / 2.0)
    except Exception:  # noqa: BLE001 - terrain probe must never break replication
        return (0.0, 0.0, yaw)
    return (roll, pitch, yaw)


def send_dynamic_entity_definition(
    server: object,
    target_ctx: object,
    *,
    entity_id: int,
    entity_type: int,
    team_id: int,
    pos: tuple[float, float, float],
    heading: float = 0.0,
    is_static: bool = True,
) -> bool:
    if not target_ctx.session or not target_ctx.session.translation_ack_received:
        return False
    tick = server._get_network_tick(target_ctx)
    include_local_state, ls = server._get_update_array_local_state_for_viewer(target_ctx)
    local_state_kwargs = dict(ls)
    local_state_kwargs.setdefault("health", server._get_health_value(target_ctx))
    local_state_kwargs.setdefault("fuel", server._get_energy_value(target_ctx))
    rot = _building_terrain_conform_rot(server, pos, heading)
    payload = build_update_array_create_tank(
        tick=tick,
        entity_id=entity_id,
        entity_type=entity_type,
        team=team_id,
        pos=server._to_client_pos(pos),
        behavior_type=team_id,
        include_health=include_local_state,
        include_entity_vitals=False,
        is_manned=False,
        is_static=is_static,
        rot=rot,
        **local_state_kwargs,
    )
    sent = server._send_packet_to_client(target_ctx, payload, prefer_tcp=False)
    if sent:
        target_ctx.known_entity_ids.add(entity_id)
        if server.pktlog.enabled:
            server.pktlog.log(
                client_id=target_ctx.client_id,
                label="DYNAMIC_ENTITY_CREATE",
                tick=tick,
                payload=payload,
                transport="UDP",
                entity_count=1,
                entity_ids=(entity_id,),
                mask_bits=(0b1011,),
                has_local_state=include_local_state,
                health=server._get_health_value(target_ctx) if include_local_state else -1.0,
                extra=f"type={entity_type} team={team_id}",
            )
    return sent


def broadcast_dynamic_entity_definition(
    server: object,
    *,
    entity_id: int,
    entity_type: int,
    team_id: int,
    pos: tuple[float, float, float],
    heading: float = 0.0,
    is_static: bool = True,
) -> int:
    sent = 0
    for target in server._snapshot_in_game_clients():
        if server._send_dynamic_entity_definition(
            target,
            entity_id=entity_id,
            entity_type=entity_type,
            team_id=team_id,
            pos=pos,
            heading=heading,
            is_static=is_static,
        ):
            sent += 1
    return sent


def create_dynamic_building_from_uplink(server: object, ctx: object, command: dict[str, Any]) -> dict[str, Any]:
    entity_type = int(command["entity_type"])
    team_id = int(ctx.session.team_id or 1)
    slot = int(command.get("slot", 0) or 0)
    oid = server._allocate_dynamic_building_oid()
    x, y, z = server._choose_dynamic_building_pos(ctx, slot)
    heading = float(getattr(ctx, "player_heading", 0.0) or 0.0)
    building = BuildingEntity(x=x, y=y, z=z, entity_type=entity_type, team_id=team_id, heading=heading)
    max_hp = server._building_max_health_for_type(entity_type)
    server._building_entities[oid] = building
    server._building_health[oid] = max_hp
    server._building_max_health[oid] = max_hp
    server._dynamic_building_ids.add(oid)
    source = {
        "client_id": ctx.client_id,
        "player_entity_id": ctx.session.entity_id or ctx.entity_id,
        "ship_oid": command.get("ship_oid"),
        "slot": slot,
        "command": command,
        "created_at": time.time(),
    }
    server._dynamic_building_sources[oid] = source
    ship = server._uplink_ships.get(team_id) or server._get_or_create_uplink_ship(ctx, team_id)
    cargo = list(ship.get("cargo", [40, 40, 40, 40]))
    if 0 <= slot < len(cargo):
        cargo[slot] = entity_type
    ship["cargo"] = cargo
    ship["last_build_oid"] = oid
    server._broadcast_uplink_ship_info(ship)
    server._rebuild_static_world_raycast_index()
    sent = server._broadcast_dynamic_entity_definition(
        entity_id=oid,
        entity_type=entity_type,
        team_id=team_id,
        pos=building.pos,
        heading=heading,
        is_static=True,
    )
    event = {
        "ok": sent > 0,
        "oid": oid,
        "entity_type": entity_type,
        "entity_type_name": getattr(EntityType(entity_type), "name", str(entity_type)),
        "team_id": team_id,
        "pos": [round(float(v), 5) for v in building.pos],
        "health": max_hp,
        "replication_targets": sent,
    }
    print(
        f"[BUILD-UPLINK] created oid={oid} type={event['entity_type_name']} "
        f"team={team_id} pos=({x:.1f},{y:.1f},{z:.1f}) targets={sent}"
    )
    lifecycle = server._building_lifecycle_base_event(oid, "create")
    lifecycle.update({"ok": sent > 0, "source": "uplink_build", "replication_targets": sent})
    server._remember_building_lifecycle_event(lifecycle)
    return event


def delete_dynamic_building_from_uplink(server: object, ctx: object, command: dict[str, Any]) -> dict[str, Any]:
    entity_type = int(command.get("entity_type", -1) or -1)
    team_id = int(ctx.session.team_id or 1)
    slot = command.get("slot")
    candidates = []
    for oid in sorted(server._dynamic_building_ids):
        building = server._building_entities.get(oid)
        if not building:
            continue
        source = server._dynamic_building_sources.get(oid, {})
        if entity_type >= 0 and int(building.entity_type) != entity_type:
            continue
        if int(building.team_id) != team_id:
            continue
        if slot is not None and source.get("slot") != slot:
            continue
        candidates.append(oid)
    if not candidates:
        return {"ok": False, "error": "no matching dynamic building"}
    oid = candidates[-1]
    lifecycle = server._building_lifecycle_base_event(oid, "delete")
    ship = server._uplink_ships.get(team_id)
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
    sent = server._broadcast_building_delete(oid, prefer_tcp=False)
    server._remove_dynamic_building_record(oid)
    lifecycle.update({"ok": sent > 0, "source": "uplink_delete", "delete_sent": sent, "removed": True})
    server._remember_building_lifecycle_event(lifecycle)
    return {"ok": sent > 0, "oid": oid, "replication_targets": sent}


def get_or_create_uplink_ship(server: object, ctx: object, team_id: int) -> dict[str, Any]:
    ship = server._uplink_ships.get(team_id)
    if ship is not None:
        return ship
    base_pos = tuple(float(v) for v in (ctx.player_pos or (2600.0, 3040.0, 5.0)))
    try:
        offset_x = float(os.environ.get("WULFRAM_UPLINK_SHIP_OFFSET_X", "-450.0"))
        offset_y = float(os.environ.get("WULFRAM_UPLINK_SHIP_OFFSET_Y", "0.0"))
        offset_z = float(os.environ.get("WULFRAM_UPLINK_SHIP_OFFSET_Z", "12.0"))
    except ValueError:
        offset_x = -450.0
        offset_y = 0.0
        offset_z = 12.0
    x = base_pos[0] + offset_x
    y = base_pos[1] + offset_y
    ground_z = server._terrain_ground_z_at(x, y)
    z = (float(ground_z) + offset_z) if ground_z is not None else base_pos[2] + offset_z
    try:
        base_oid = int(os.environ.get("WULFRAM_UPLINK_SHIP_BASE_OID", "29000"), 0)
    except ValueError:
        base_oid = 29000
    ship = {
        "oid": base_oid + int(team_id),
        "team_id": team_id,
        "name": f"Team {team_id} Supply Ship",
        "pos": (x, y, z),
        "heading": 0.0,
        "cargo": [40, 40, 40, 40],
        "cargo_times": [0, 0, 0, 0],
        "build_mode": 3,
        "shield_pct": 100,
        "status_template": 0,
    }
    server._uplink_ships[team_id] = ship
    return ship


def build_uplink_ship_info_packet(ship: dict[str, Any]) -> bytes:
    return build_supply_ship_info(
        int(ship["oid"]),
        shield_pct=int(ship.get("shield_pct", 100) or 100),
        status_template=int(ship.get("status_template", 0) or 0),
        cargo_slots=list(ship.get("cargo", [40, 40, 40, 40])),
        cargo_times=list(ship.get("cargo_times", [0, 0, 0, 0])),
        build_mode=int(ship.get("build_mode", 3) or 3),
    )


def send_uplink_ship_info(server: object, ctx: object, ship: dict[str, Any]) -> bool:
    return server._send_packet_to_client(ctx, server._build_uplink_ship_info_packet(ship), prefer_tcp=True)


def broadcast_uplink_ship_info(server: object, ship: dict[str, Any]) -> int:
    sent = 0
    for target in server._snapshot_in_game_clients():
        if server._send_uplink_ship_info(target, ship):
            sent += 1
    return sent


def send_existing_build_uplink_entities(server: object, ctx: object) -> int:
    sent = 0
    for team_id, ship in sorted(server._uplink_ships.items(), key=lambda item: int(item[0])):
        if server._send_dynamic_entity_definition(
            ctx,
            entity_id=int(ship["oid"]),
            entity_type=int(EntityType.SUPPLY_SHIP),
            team_id=int(team_id),
            pos=ship["pos"],
            heading=float(ship.get("heading", 0.0) or 0.0),
            is_static=True,
        ):
            sent += 1
    for oid in sorted(server._dynamic_building_ids):
        building = server._building_entities.get(oid)
        if not building:
            continue
        if server._send_dynamic_entity_definition(
            ctx,
            entity_id=int(oid),
            entity_type=int(building.entity_type),
            team_id=int(building.team_id),
            pos=building.pos,
            heading=float(getattr(building, "heading", 0.0) or 0.0),
            is_static=True,
        ):
            sent += 1
    return sent


def ensure_uplink_mvp_state(server: object, ctx: object) -> None:
    """Default-off bootstrap for the OG uplink/build MVP probe."""
    if not getattr(server, "build_uplink_mvp", False):
        return
    if getattr(ctx, "uplink_mvp_bootstrap_sent", False):
        return
    if not ctx.session or not ctx.session.in_game or not ctx.session.translation_ack_received:
        return
    team_id = int(ctx.session.team_id or 1)
    player_oid = int(ctx.session.entity_id or ctx.entity_id)
    ship = server._get_or_create_uplink_ship(ctx, team_id)
    packets = (
        build_update_stats_team_first(player_id=player_oid, entity_id=player_oid, team_id=team_id),
        build_ship_status(int(ship["oid"]), team_id, str(ship["name"])),
        server._build_uplink_ship_info_packet(ship),
        build_carrying_info(player_oid, cargo_type=0, has_uplink=True, cargo_count=0),
        build_uplink_info(team_id, player_oid, 3),
    )
    sent = 0
    for payload in packets:
        if server._send_packet_to_client(ctx, payload, prefer_tcp=True):
            sent += 1
    dynamic_sent = server._send_existing_build_uplink_entities(ctx)
    ctx.uplink_mvp_bootstrap_sent = sent == len(packets) and dynamic_sent > 0
    print(
        f"[BUILD-UPLINK] bootstrap client={ctx.client_id} team={team_id} "
        f"ship={ship['oid']} player={player_oid} packets={sent}/{len(packets)} "
        f"dynamic_entities={dynamic_sent}"
    )


def handle_comm_message_request(
    server: object,
    ctx: Optional[object],
    packet: bytes,
    *,
    transport: str,
    body: bytes,
    addr: Optional[tuple] = None,
    sequence: Optional[int] = None,
) -> dict[str, Any]:
    decoded = server._decode_comm_message_request_body(body)
    event: dict[str, Any] = {
        "time": time.time(),
        "transport": transport,
        "client_id": getattr(ctx, "client_id", None),
        "addr": list(addr) if addr else None,
        "sequence": sequence,
        "raw_hex": packet.hex(),
        "decoded": decoded,
        "handled": False,
        "mvp_enabled": bool(getattr(server, "build_uplink_mvp", False)),
    }
    if ctx is not None:
        ctx.comm_message_request_count = int(getattr(ctx, "comm_message_request_count", 0) or 0) + 1
        ctx.last_comm_message_request = event
    if not decoded.get("ok"):
        return event

    text = str(decoded.get("text") or "")

    # Player self-commands (respawn/help) ride ordinary chat text, so they work
    # over any transport (TCP / UDP / reliable-UDP) and any send mode — unlike a
    # leading '/', which the OG client eats as a whisper-destination selector and
    # never sends. Checked before the build-uplink type-2 gate and independent of
    # the MVP flag so respawn always works.
    if handlers.dispatch_player_chat_command(server, ctx, text):
        event["handled"] = True
        event["player_command"] = text
        if ctx is not None:
            ctx.last_player_chat_command = text
        print(
            f"[CHAT-CMD] c{getattr(ctx, 'client_id', '?')} {transport} "
            f"text={text!r} -> handled"
        )
        return event

    msg_type = int(decoded.get("message_type") or 0)
    if msg_type != 2:
        # Normal player chat (ALL/TEAM/whisper). msg_type 2 is reserved here for the
        # build-uplink/starship command channel (handled below); everything else is
        # relayed to other players as COMM_MESSAGE (0x1F).
        relay = server._relay_player_chat(
            ctx, msg_type, int(decoded.get("flags_or_target") or 0), text
        )
        event["handled"] = True
        event["chat_relay"] = relay
        return event

    command = server._parse_build_uplink_command(text)
    event["build_uplink_command"] = command
    event["handled"] = True
    if ctx is not None:
        ctx.build_uplink_command_count = int(getattr(ctx, "build_uplink_command_count", 0) or 0) + 1
        ctx.last_build_uplink_command = event
    if not getattr(server, "build_uplink_mvp", False):
        event["result"] = {"ok": False, "error": "WULFRAM_BUILD_UPLINK_MVP disabled"}
    elif ctx is None:
        event["result"] = {"ok": False, "error": "unknown client"}
    elif not command.get("ok"):
        event["result"] = {"ok": False, "error": command.get("error", "parse failed")}
    elif command.get("action") == "build":
        event["result"] = server._create_dynamic_building_from_uplink(ctx, command)
    elif command.get("action") == "delete":
        event["result"] = server._delete_dynamic_building_from_uplink(ctx, command)
    elif command.get("action") == "set":
        team_id = int(ctx.session.team_id or 1)
        ship = server._uplink_ships.get(team_id)
        if ship is not None and str(command.get("field", "")).lower() == "build_mode":
            ship["build_mode"] = int(command.get("value", 2) or 2)
            server._broadcast_uplink_ship_info(ship)
        event["result"] = {"ok": True, "noted": True}
    else:
        event["result"] = {"ok": True, "noted": True}

    server._build_uplink_command_events.append(event)
    del server._build_uplink_command_events[:-100]
    print(
        f"[BUILD-UPLINK] c{getattr(ctx, 'client_id', '?')} {transport} "
        f"type=2 text={text!r} result={event.get('result')}"
    )
    return event

