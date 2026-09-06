import socket
import unittest
import threading
from types import SimpleNamespace
from wulfram.transport import TCPHandler
from wulfram.server import WulframServer


class TransportEofTests(unittest.TestCase):
    def test_timeout_then_framed_data_then_eof(self):
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        left.settimeout(.02)
        handler = TCPHandler(left)
        self.assertIsNone(handler.recv())
        self.assertFalse(handler.closed)
        right.sendall(bytes.fromhex('00070b12345678'))
        right.shutdown(socket.SHUT_WR)
        self.assertEqual(handler.recv(), bytes.fromhex('0b12345678'))
        self.assertIsNone(handler.recv())
        self.assertTrue(handler.closed)
        self.assertIsNone(handler.recv())

    def test_local_closed_socket(self):
        left, right = socket.socketpair()
        self.addCleanup(right.close)
        handler = TCPHandler(left)
        left.close()
        self.assertIsNone(handler.recv())
        self.assertTrue(handler.closed)

    def test_ping_wakes_after_disconnect_without_sending(self):
        ctx = SimpleNamespace(running=True)
        class WakeAfterClose:
            def wait(self, timeout):
                ctx.running = False
                return False
        ctx.ping_stop_event = WakeAfterClose()
        ctx.tcp_handler = SimpleNamespace(send=lambda packet: self.fail('late ping'))
        WulframServer._ping_loop(SimpleNamespace(ping_interval_s=.01), ctx)

    def test_stop_ping_joins_worker(self):
        stop = threading.Event()
        worker = threading.Thread(target=lambda: stop.wait(2))
        ctx = SimpleNamespace(ping_stop_event=stop, ping_thread=worker)
        worker.start()
        WulframServer._stop_ping_loop(None, ctx)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(ctx.ping_thread)


if __name__ == '__main__':
    unittest.main()
