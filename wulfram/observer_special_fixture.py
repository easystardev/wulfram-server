"""Exact entity-definition fixture for the isolated original-client observer.

The legacy ``spawn_entity`` control serializes its argument through the projectile
layout.  Cargo has an extra definition field and uplink must use the ordinary
entity-definition layout, so that command cannot ground their collision handlers.
This extension deliberately supports only the two special types needed by the observer.
"""
from __future__ import annotations

import json
import time

from .observer_lifecycle import serialized
from .packets import build_uplink_info


SUPPORTED_TYPES = {19, 20}


@serialized
def spawn_definition(control, args: list[str]) -> str:
    if len(args) != 6 or not args[5].startswith("c"):
        return "Error: expected observer_special <19|20> <x> <y> <z> <team> c<id>"
    try:
        entity_type = int(args[0])
        pos = tuple(float(value) for value in args[1:4])
        team_id = int(args[4])
        target_client_id = int(args[5][1:])
    except ValueError as exc:
        return f"Error: observer_special argument parse failed: {exc}"
    if entity_type not in SUPPORTED_TYPES:
        return f"Error: observer_special type {entity_type} is outside {sorted(SUPPORTED_TYPES)}"
    if team_id not in (1, 2):
        return "Error: observer_special team must be 1 or 2"

    server = control.server
    if (server is None or server.port != 2727 or control.port != 2728
            or not server.running):
        return "Error: isolated 2727/2728 endpoint required"
    clients = [client for client in server._snapshot_clients() if client.running]
    if len(clients) != 1:
        return "Error: isolated observer requires exactly one connected client"
    matches = [ctx for ctx in clients if int(ctx.client_id) == target_client_id]
    if len(matches) != 1:
        return "Error: explicit connected observer client required"
    ctx = matches[0]
    if (not ctx.running or not ctx.session or not ctx.session.in_game or
            int(ctx.session.team_id or 0) not in (1, 2)):
        return "Error: stable spawned observer client required"

    local_team = int(ctx.session.team_id or 0)
    local_entity_id = int(ctx.session.player_id or ctx.session.entity_id or ctx.entity_id or 0)
    if local_team not in (1, 2) or local_entity_id <= 0:
        return "Error: observer client has no stable local team/entity identity"

    # Force the documented local-invalid handler branch without changing the
    # original process: the inactive local uplink-holder state makes 0044608b
    # return one. A short isolated pause lets the TCP gate precede the UDP
    # definition at the client.
    gate_payload = build_uplink_info(local_team, local_entity_id, 5)
    gate_sent = server._send_packet_to_client(ctx, gate_payload, prefer_tcp=True)
    if not gate_sent:
        return "Error: failed to send observer special gate"
    time.sleep(0.2)

    known_ids = {entity_id for client in clients for entity_id in client.known_entity_ids}
    candidate = max(5000, int(getattr(control, "_entity_id", 5000) or 5000)) + 1
    while candidate in known_ids:
        candidate += 1
    control._entity_id = candidate

    # This is the receiver's ownership bit, despite the historical `is_static`
    # builder name. Zero makes 00419a70 initialize entity +0xfc to -1. The live
    # fixture is positioned above the spawn pad so that its first independently
    # owned contact is the requested tank/special pair rather than pad geometry.
    created = server._send_dynamic_entity_definition(
        ctx,
        entity_id=candidate,
        entity_type=entity_type,
        team_id=team_id,
        pos=pos,
        heading=0.0,
        is_static=False,
        cargo_contained_type=0 if entity_type == 19 else None,
    )
    result = {
        "ok": bool(created),
        "entity_id": candidate,
        "entity_type": entity_type,
        "team_id": team_id,
        "pos": list(pos),
        "owned_by_local_bit": False,
        "cargo_contained_type": 0 if entity_type == 19 else None,
        "gate": {
            "sent": True,
            "team_id": local_team,
            "holder_entity_id": local_entity_id,
            "state": 5,
        },
        "replication_targets": int(bool(created)),
    }
    return json.dumps(result, separators=(",", ":"))
