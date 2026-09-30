import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from wulfram.spawn_fixture import queue_placement, consume_placement


class SpawnFixtureTest(unittest.TestCase):
    def setUp(self):
        self.session = SimpleNamespace(in_game=False, team_id=1, world_epoch=3, local_epoch=7, login_complete=True)
        self.ctx = SimpleNamespace(session=self.session, running=True, observer_transition=False,
                                   observer_login_complete=True)
        self.control = SimpleNamespace(_get_client_by_id=lambda *a, **k: (self.ctx, None))

    def queue(self):
        with patch.dict(os.environ, WULFRAM_SPAWN_FIXTURE='1'):
            return queue_placement(self.control, ['c1', '2520', '2400', '100'])

    def test_disabled_and_invalid_input(self):
        with patch.dict(os.environ, WULFRAM_SPAWN_FIXTURE='0'):
            self.assertIn('disabled', queue_placement(self.control, []))
        with patch.dict(os.environ, WULFRAM_SPAWN_FIXTURE='1'):
            self.assertIn('Error', queue_placement(self.control, ['c1', 'nan', '0', '0']))

    def test_exact_next_transition_once(self):
        self.assertIn('Queued', self.queue())
        self.assertIn('already queued', self.queue())
        self.session.local_epoch += 1
        self.assertEqual(consume_placement(self.ctx, 1), (2520, 2400, 100))
        self.assertIsNone(consume_placement(self.ctx, 1))

    def test_invalidated_or_expired_never_applies(self):
        for change in ('world', 'local', 'session', 'team', 'disconnect', 'expiry'):
            with self.subTest(change=change):
                self.setUp()
                with patch('wulfram.spawn_fixture.time.monotonic', return_value=100):
                    self.assertIn('Queued', self.queue())
                self.session.local_epoch += 1
                team = 1
                if change == 'world': self.session.world_epoch += 1
                if change == 'local': self.session.local_epoch += 1
                if change == 'session': self.ctx.session = SimpleNamespace()
                if change == 'team': team = 2
                if change == 'disconnect': self.ctx.running = False
                with patch('wulfram.spawn_fixture.time.monotonic', return_value=131 if change == 'expiry' else 101):
                    self.assertIsNone(consume_placement(self.ctx, team))
                self.assertIsNone(self.ctx._spawn_fixture)

    def test_active_or_transitioning_refused(self):
        self.session.in_game = True
        self.assertIn('Error', self.queue())
        self.session.in_game = False
        self.ctx.observer_transition = True
        self.assertIn('Error', self.queue())

    def test_expired_queue_can_be_replaced(self):
        with patch('wulfram.spawn_fixture.time.monotonic', return_value=100):
            self.assertIn('Queued', self.queue())
        with patch('wulfram.spawn_fixture.time.monotonic', return_value=131):
            self.assertIn('Queued', self.queue())
            self.session.local_epoch += 1
            self.assertEqual(consume_placement(self.ctx, 1), (2520, 2400, 100))


if __name__ == '__main__':
    unittest.main()
