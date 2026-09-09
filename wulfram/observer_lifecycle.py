"""Isolated observer candidate: coarse serialization, explicit transition epochs.

The lock intentionally spans a local frame and its sends, not waits/joins. It
is process-wide to avoid A->B/B->A victim-lock deadlocks in turret damage.
"""
from functools import wraps
from threading import RLock
import time

FRAME_LOCK = RLock()


def serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with FRAME_LOCK:
            return function(*args, **kwargs)
    return wrapped


def world_reset(function):
    @wraps(function)
    def wrapped(session, *args, **kwargs):
        with FRAME_LOCK:
            session.world_epoch += 1
            session.local_epoch += 1
            return function(session, *args, **kwargs)
    return wrapped


def local_death(function):
    @wraps(function)
    def wrapped(server, ctx, *args, **kwargs):
        with FRAME_LOCK:
            ctx.session.local_epoch += 1
            return function(server, ctx, *args, **kwargs)
    return wrapped


def spawn_transition(function):
    """Run spawn's yielded waits unlocked; reject invalidated continuation.

Only this wrapper owns transition reservation. Nested spawn attempts are refused.
An epoch change during a wait leaves the newer lifecycle untouched.
"""
    @wraps(function)
    def wrapped(server, ctx, *args, **kwargs):
        with FRAME_LOCK:
            if not ctx.running or ctx.observer_transition:
                return False
            ctx.session.local_epoch += 1
            token = (ctx.session.world_epoch, ctx.session.local_epoch)
            ctx.observer_transition = True
            generator = function(server, ctx, *args, **kwargs)
        try:
            while True:
                with FRAME_LOCK:
                    if not ctx.running or token != (ctx.session.world_epoch, ctx.session.local_epoch):
                        generator.close()
                        return False
                    try:
                        delay = next(generator)
                    except StopIteration as result:
                        return result.value
                time.sleep(delay)
        except Exception:
            with FRAME_LOCK:
                if token == (ctx.session.world_epoch, ctx.session.local_epoch):
                    ctx.session.in_game = False
                    ctx.session.local_epoch += 1
            raise
        finally:
            with FRAME_LOCK:
                # The reservation belongs to this invocation even if invalidated.
                ctx.observer_transition = False
    return wrapped


def reject_active_debug(function):
    """Raw state/packet controls have no transactional world-reset contract."""
    @wraps(function)
    def wrapped(control, *args, **kwargs):
        server = control.server
        if server and any(c.tick_thread and c.tick_thread.is_alive() for c in server._snapshot_clients()):
            return "Error: raw lifecycle/debug control unavailable with active observer worker"
        return function(control, *args, **kwargs)
    return wrapped
