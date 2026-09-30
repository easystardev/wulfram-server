"""Bootstrap an unset client clock from the original transport handshake ACK.

This is a one-way receipt estimate, not an RTT measurement. ACTION owns subsequent
clock refinement. Both helpers serialize with simulation/lifecycle operations.
"""
import time
from wulfram.observer_lifecycle import serialized
from wulfram.packets import get_ticks


def _owned(server, ctx, addr):
    return (ctx is not None and server.running and ctx.running
            and server.clients.get(ctx.client_id) is ctx
            and ctx.session is not None and ctx.session.udp_verified
            and ctx.session.udp_addr == addr
            and server.udp_addr_to_client.get(addr) is ctx)


@serialized
def record_handshake(server, ctx, addr, client_tick):
    with server.clients_lock:
        if not _owned(server, ctx, addr) or ctx.tick_offset is not None:
            return
        ctx.transport_clock_pending = (ctx.session, ctx.session.world_epoch,
                                       addr, int(client_tick) & 0xffffffff, time.monotonic())


@serialized
def bootstrap_ack(server, ctx, data, addr):
    if len(data) != 6 or data[:2] != b'\x02\x00' or not server.use_client_ticks:
        return False
    with server.clients_lock:
        if not _owned(server, ctx, addr) or ctx.tick_offset is not None:
            return False
        s = ctx.session
        pending = getattr(ctx, 'transport_clock_pending', None)
        if (not s.udp_d_handshake_received or not pending or pending[0] is not s
                or pending[1] != s.world_epoch or pending[2] != addr):
            return False
        tick = int.from_bytes(data[2:6], 'big')
        age = time.monotonic() - pending[4]
        # The ACK is newly sampled after the client's handshake. Reject reordered
        # pre-handshake samples and old associations; differences use u32 serial
        # arithmetic, including a handshake/ACK crossing the wrap boundary.
        if tick == 0 or not 0 <= age <= 30 or ((tick - pending[3]) & 0xffffffff) > 30000:
            return False
        received = get_ticks()
        ctx.tick_offset = tick - received
        ctx.last_client_tick = tick
        ctx.tick_alignment_source = 'transport_handshake_ack'
        ctx.transport_clock_bootstrap = dict(client_tick=tick, server_receive_tick=received,
                                             handshake_tick=pending[3], receive_age_seconds=age,
                                             world_epoch=s.world_epoch)
        ctx.transport_clock_pending = None
        print(f'[TICK] Transport ACK bootstrap client={ctx.client_id} tick={tick} server={received}')
        return True
