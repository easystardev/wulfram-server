"""Original constant identify, independent ports, adverse order and cleanup gates."""
import socket
import threading
from types import SimpleNamespace
import pytest
from wulfram.server_net import NetMixin
from wulfram.session import Session, Phase
from wulfram.transport import UDPHandler

IDENTIFY = b'\x08\x00\x0cHello There\x00'


def context(i, ip='127.0.0.1'):
    return SimpleNamespace(client_id=i, client_addr=(ip, 40000+i), running=True,
                           udp_socket=None, session=Session(phase=Phase.HANDSHAKE,
                                                           udp_config_sent_time=1))


def server():
    value = NetMixin()
    value.clients = {}
    value.clients_lock = threading.Lock()
    value.udp_addr_to_client = {}
    value.session_key_to_client = {}
    return value


@pytest.mark.parametrize('tcp_order', [(1,2), (2,1)])
@pytest.mark.parametrize('udp_order', [(1,2), (2,1)])
def test_owned_destination_order_matrix_and_reconnect(tcp_order, udp_order):
    value = server()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as shared:
        shared.bind(('127.0.0.1',0))
        transport = UDPHandler(shared)
        transport.peer_owner = value.udp_addr_to_client.get
        for generation in range(2):
            clients = {i: context(i) for i in tcp_order}
            value.clients.update(clients)
            peers = {i: socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for i in tcp_order}
            try:
                ports = {i: transport.open_channel(clients[i]) for i in tcp_order}
                assert len(set(ports.values())) == 2
                for peer in peers.values():
                    peer.bind(('127.0.0.1',0)); peer.settimeout(1)
                for i in udp_order:
                    peers[i].sendto(IDENTIFY, ('127.0.0.1',ports[i]))
                    payload, source = transport.recv_from()
                    assert transport.received_owner is clients[i]
                    value._handle_single_udp_packet(transport.received_owner, payload, source)
                    assert value.udp_addr_to_client[source] is clients[i]
                    assert clients[i].session.udp_addr == peers[i].getsockname()
                    transport.send_to(bytes([i]), clients[i].session.udp_addr)
                    reply, origin = peers[i].recvfrom(100)
                    assert reply == bytes([i]) and origin[1] == ports[i]
            finally:
                for i, c in clients.items():
                    c.running = False
                    transport.close_channel(c)
                    value.clients.pop(i)
                    addr = c.session.udp_addr
                    if value.udp_addr_to_client.get(addr) is c:
                        del value.udp_addr_to_client[addr]
                    peers[i].close()
                assert not transport.channels and not value.udp_addr_to_client
            # A departed context cannot reclaim an endpoint after reconnect.
            assert not value._bind_udp_client(clients[1], ('127.0.0.1',12345), reason='late')


def test_shared_identify_rejects_ambiguity_and_other_ip():
    value = server(); a,b = context(1),context(2)
    value.clients = {1:a,2:b}
    for port in (50001,50002):
        value._handle_single_udp_packet(None, IDENTIFY, ('127.0.0.1',port))
    assert not value.udp_addr_to_client
    b.running = False
    value._handle_single_udp_packet(None, IDENTIFY, ('127.0.0.2',50003))
    assert not value.udp_addr_to_client
    value._handle_single_udp_packet(None, IDENTIFY, ('127.0.0.1',50001))
    assert value.udp_addr_to_client[('127.0.0.1',50001)] is a


def test_binding_does_not_steal_or_accept_stale_context():
    value = server(); a,b = context(1),context(2); value.clients={1:a,2:b}
    addr=('127.0.0.1',50001)
    assert value._bind_udp_client(a,addr,reason='first')
    assert not value._bind_udp_client(b,addr,reason='crossed')
    assert not b.session.udp_verified
    replacement=context(1);value.clients[1]=replacement
    assert not value._bind_udp_client(a,('127.0.0.1',50002),reason='stale')


def test_unverified_handshake_cannot_advance_to_login(monkeypatch):
    import wulfram.server_net as network
    value=server(); c=context(1); value.clients={1:c}; sent=[]
    c.tcp_handler=SimpleNamespace(send=sent.append)
    value.public_addr='127.0.0.1'; value.port=2627
    value.udp_handler=SimpleNamespace(open_channel=lambda owner: 51000)
    monkeypatch.setenv('WULFRAM_LOGIN_BOOTSTRAP','og')
    monkeypatch.delenv('WULFRAM_ADVERTISE_UDP_HOST',raising=False)
    monkeypatch.delenv('WULFRAM_ADVERTISE_UDP_PORT',raising=False)
    ticks=iter(range(0,100,2))
    monkeypatch.setattr(network.time,'monotonic',lambda: next(ticks))
    monkeypatch.setattr(network.time,'sleep',lambda seconds: None)
    with pytest.raises(ConnectionAbortedError,match='ownership'):
        value._do_handshake(c)
    assert c.session.phase is Phase.HANDSHAKE
    assert len(sent)==1 and sent[0][:2]==b'\x13\x01'
