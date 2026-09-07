import unittest
from types import SimpleNamespace
from unittest.mock import patch

from wulfram.server import WulframServer


class NetworkTickBootstrapTests(unittest.TestCase):
    @staticmethod
    def server(*, use_client_ticks):
        server = object.__new__(WulframServer)
        server.use_client_ticks = use_client_ticks
        return server

    @staticmethod
    def context(**overrides):
        values = {
            "tick_offset": None,
            "last_client_tick": 0,
            "last_sent_tick": 0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_pre_input_bootstrap_is_zero_and_does_not_seed_monotonic_tick(self):
        server = self.server(use_client_ticks=True)
        ctx = self.context()
        with patch("wulfram.server.get_ticks", return_value=9_000_000) as ticks:
            self.assertEqual(server._get_network_tick(ctx), 0)
            self.assertEqual(server._get_network_tick(ctx), 0)
        ticks.assert_not_called()
        self.assertEqual(ctx.last_sent_tick, 0)
        self.assertIsNone(ctx.tick_offset)

    def test_first_client_tick_can_sync_far_below_server_uptime(self):
        server = self.server(use_client_ticks=True)
        ctx = self.context()
        with patch("wulfram.server.get_ticks", side_effect=[9_000_000, 9_000_025]):
            server._sync_tick_offset(ctx, 1_200)
            self.assertEqual(ctx.tick_offset, -8_998_800)
            self.assertEqual(ctx.last_client_tick, 1_200)
            self.assertEqual(server._get_network_tick(ctx), 1_225)
        self.assertEqual(ctx.last_sent_tick, 1_225)

    def test_aligned_ticks_progress_and_clamp_monotonically(self):
        server = self.server(use_client_ticks=True)
        ctx = self.context(
            tick_offset=-999_000,
            last_client_tick=1_000,
            last_sent_tick=1_000,
        )
        with patch("wulfram.server.get_ticks", return_value=999_900):
            # Aligned 900 clamps to the last observed client tick, then advances
            # past the already-published tick.
            self.assertEqual(server._get_network_tick(ctx), 1_001)
        with patch("wulfram.server.get_ticks", return_value=1_000_000):
            self.assertEqual(server._get_network_tick(ctx), 1_002)
        ctx.last_client_tick = 1_050
        with patch("wulfram.server.get_ticks", return_value=1_000_000):
            self.assertEqual(server._get_network_tick(ctx), 1_050)
        with patch("wulfram.server.get_ticks", return_value=1_000_100):
            self.assertEqual(server._get_network_tick(ctx), 1_100)

    def test_server_tick_domain_ignores_client_alignment_fields(self):
        server = self.server(use_client_ticks=False)
        ctx = self.context(
            tick_offset=-8_000_000,
            last_client_tick=123_456,
        )
        with patch("wulfram.server.get_ticks", return_value=5_000):
            self.assertEqual(server._get_network_tick(ctx), 5_000)
        with patch("wulfram.server.get_ticks", return_value=4_999):
            self.assertEqual(server._get_network_tick(ctx), 5_001)
        self.assertEqual(ctx.last_sent_tick, 5_001)

    def test_zero_client_tick_does_not_establish_an_offset(self):
        server = self.server(use_client_ticks=True)
        ctx = self.context()
        with patch("wulfram.server.get_ticks") as ticks:
            server._sync_tick_offset(ctx, 0)
        ticks.assert_not_called()
        self.assertIsNone(ctx.tick_offset)
        self.assertEqual(ctx.last_client_tick, 0)


if __name__ == "__main__":
    unittest.main()
