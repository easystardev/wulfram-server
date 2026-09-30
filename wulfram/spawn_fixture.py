"""Default-off, one-shot placement for isolated normal-spawn experiments."""
import math
import os
import time

from .observer_lifecycle import FRAME_LOCK


def queue_placement(control, args):
    if os.environ.get('WULFRAM_SPAWN_FIXTURE') != '1':
        return 'Error: spawn fixture disabled'
    try:
        if len(args) != 4 or not args[0].startswith('c'):
            raise ValueError('spawn_fixture c<ID> x y z')
        client_id = int(args[0][1:])
        position = tuple(float(v) for v in args[1:])
        if not all(math.isfinite(v) and -8192 < v < 8192 for v in position):
            raise ValueError('Position outside finite world bounds')
    except ValueError as exc:
        return f'Error: {exc}'
    with FRAME_LOCK:
        ctx, _ = control._get_client_by_id(client_id, require_udp=False)
        if not ctx or not ctx.running or not ctx.session:
            return 'Error: missing active client'
        session = ctx.session
        if session.in_game or ctx.observer_transition or not session.team_id or not session.login_complete:
            return 'Error: placement requires a ready unspawned client'
        pending = getattr(ctx, '_spawn_fixture', None)
        if pending is not None:
            old_session, world, local, team, deadline, _ = pending
            if (old_session is session and world == session.world_epoch
                    and local == session.local_epoch and team == session.team_id
                    and time.monotonic() <= deadline):
                return 'Error: placement already queued'
        ctx._spawn_fixture = (session, session.world_epoch, session.local_epoch,
                              session.team_id, time.monotonic() + 30, position)
        return f'Queued spawn fixture: client={client_id} pos={position}'


def consume_placement(ctx, team_id):
    """Called inside spawn_transition, after its sole local-epoch increment."""
    with FRAME_LOCK:
        pending = getattr(ctx, '_spawn_fixture', None)
        ctx._spawn_fixture = None
        if pending is None:
            return None
        session, world, local, team, deadline, position = pending
        if (ctx.running and ctx.session is session and session.world_epoch == world
                and session.local_epoch == local + 1 and team_id == team
                and time.monotonic() <= deadline):
            return position
        return None
