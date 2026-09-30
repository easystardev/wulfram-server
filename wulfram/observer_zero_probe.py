"""Uninstalled, default-off control extension for the original observer probe.

Integration contract: control.py dispatches `observer_zero` to this function;
the existing raw/debug guard remains untouched. This module is experimental.
"""
import os
import json
from wulfram.observer_lifecycle import serialized
from wulfram.packets import build_update_array_create_tank, build_update_array_player_update


@serialized
def send_observer_zero_probe(control, args):
    if os.environ.get('WULFRAM_OBSERVER_ZERO_PROBE') != '1':
        return 'Error: observer zero probe disabled'
    if args not in (['create'], ['equal'], ['positive'], ['stale']):
        return 'Error: expected create/equal/positive/stale'
    server = control.server
    if not server or server.port != 2727 or control.port != 2728 or not server.running:
        return 'Error: isolated 2727 endpoint required'
    with server.clients_lock:
        return _send_locked(control, args)


def _send_locked(control, args):
    server = control.server
    clients = list(server.clients.values())
    if len(clients) != 1:
        return 'Error: exclusive client required'
    ctx = clients[0]
    s = ctx.session
    if (not ctx.running or not s or s.in_game or s.phase.name != 'TEAM_SELECT'
            or ctx.observer_transition or ctx.tick_offset is not None
            or not s.login_complete or not s.want_updates_received
            or not s.translation_ack_received):
        return 'Error: never-spawned ready observer required'
    if (ctx.tcp_handler is None or ctx.tcp_handler.closed
            or ctx.tcp_handler is not control.tcp_handler):
        return 'Error: control target changed'
    token = (id(ctx), ctx.client_id, s.world_epoch, s.local_epoch)
    previous = getattr(control, '_observer_zero_probe', None)
    stage = args[0]
    expected = {'create': None, 'equal': 'create', 'positive': 'equal', 'stale': 'positive'}[stage]
    if expected is None:
        if previous is not None and previous[0] == token:
            return 'Error: probe already started in this session'
    elif previous != (token, expected):
        return 'Error: probe stage or session mismatch'
    oid = 1900000123
    ids = set(ctx.known_entity_ids)
    ids.update(getattr(o, field, 0) for o in (ctx, s) for field in ('entity_id', 'player_id'))
    ids.update(p.entity_id for p in ctx.active_projectiles)
    for name in ('_building_entities', '_dynamic_building_ids', '_dynamic_building_sources',
                 '_building_construction', '_building_deconstruction', '_dropped_cargo'):
        ids.update(getattr(server, name, {}))
    ids.update(ship['oid'] for ship in getattr(server, '_uplink_ships', {}).values())
    ids.update(p['oid'] for p in server.get_spawn_points())
    counters = {name: getattr(server, name, 0) for name in ('next_entity_id',
                '_test_projectile_id', '_dynamic_building_next_oid', '_dropped_cargo_next_oid')}
    counters['weapon_next_entity_id'] = getattr(ctx.weapon_system, 'next_entity_id', 0)
    counters.update({name: getattr(control, name, 0) for name in ('_projectile_id', '_entity_id')})
    if oid in ids or any(int(v) >= oid for v in counters.values()):
        return 'Error: probe OID already known'
    if stage == 'create':
        packet = build_update_array_create_tank(0, oid, 0, 2, (1000, 1000, 100),
                                                include_health=False, include_interp=True)
    else:
        tick, x = {'equal': (0, 1100), 'positive': (1, 1200), 'stale': (0, 1300)}[stage]
        packet = build_update_array_player_update(tick, oid, (x, 1000, 100), (0, 0, 0),
                                                 include_local_state=False)
    # A sendall exception can follow partial transmission. Refuse retry in this
    # session rather than risk duplicating an uncertain stage.
    control._observer_zero_probe = (token, 'sending')
    ctx.tcp_handler.send(packet)
    control._observer_zero_probe = (token, stage)
    return ('Sent observer probe ' + stage + ' ' + packet.hex() + ' scan='
            + json.dumps({'known_ids': sorted(ids), 'allocator_counters': counters}))
