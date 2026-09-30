"""UDP receive scheduling must not starve newly advertised client channels."""

import socket
from types import SimpleNamespace

from wulfram.transport import UDPHandler


def test_busy_shared_socket_does_not_starve_owned_channels():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as shared:
        shared.bind(("127.0.0.1", 0))
        handler = UDPHandler(shared)
        owners = [SimpleNamespace(udp_socket=None, session=SimpleNamespace(udp_verified=False)) for _ in range(2)]
        peers = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(3)]
        try:
            ports = [shared.getsockname()[1], *(handler.open_channel(owner) for owner in owners)]
            for peer in peers:
                peer.bind(("127.0.0.1", 0))

            # Keep the first-listed shared socket busy while both owned sockets
            # have admission datagrams waiting. A ready[0] policy reads only the
            # shared packets; round-robin must service each owner within 3 reads.
            for payload in (b"shared-1", b"shared-2", b"shared-3"):
                peers[0].sendto(payload, ("127.0.0.1", ports[0]))
            peers[1].sendto(b"owner-1", ("127.0.0.1", ports[1]))
            peers[2].sendto(b"owner-2", ("127.0.0.1", ports[2]))

            observed = []
            for _ in range(3):
                payload, _ = handler.recv_from()
                observed.append((payload, handler.received_owner))

            assert [owner for _, owner in observed] == [*owners, None]
            assert [payload for payload, _ in observed] == [b"owner-1", b"owner-2", b"shared-1"]
        finally:
            for owner in owners:
                handler.close_channel(owner)
            for peer in peers:
                peer.close()
