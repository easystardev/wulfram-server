from __future__ import annotations

import unittest
from types import SimpleNamespace

from wulfram import tutorial_runtime


def definition():
    return tutorial_runtime.parse_definition({
        "format": "wulfram-tutorial",
        "version": 1,
        "map": "test-map",
        "teams": [1],
        "lessons": [
            {
                "id": "one",
                "prompt": "Reach one.",
                "trigger": {"type": "enter-region", "region": {"min": [10, 10, 0], "max": [20, 20, 30]}},
                "onComplete": {"message": "One done."},
            },
            {
                "id": "two",
                "prompt": "Reach two.",
                "trigger": {"type": "enter-region", "region": {"min": [30, 30, 0], "max": [40, 40, 30]}},
                "onComplete": {"message": "Two done."},
            },
        ],
    }, expected_map="test-map")


def one_lesson(trigger):
    return tutorial_runtime.parse_definition({
        "format": "wulfram-tutorial",
        "version": 1,
        "map": "test-map",
        "teams": [1],
        "lessons": [{
            "id": "exercise",
            "prompt": "Perform the exercise.",
            "hint": "Use the marked bays.",
            "trigger": trigger,
            "onComplete": {"message": "Exercise done."},
        }],
    }, expected_map="test-map")


class TutorialRuntimeTest(unittest.TestCase):
    def test_definition_rejects_non_ascii_chat_text(self):
        raw = {
            "format": "wulfram-tutorial", "version": 1, "map": "test-map",
            "lessons": [{
                "id": "ascii-gate",
                "prompt": "Move beside -- not on top.",
                "hint": "This em dash is invalid: \u2014",
                "trigger": {
                    "type": "self-confirm",
                    "region": {"min": [0, 0, 0], "max": [1, 1, 1]},
                },
            }],
        }
        with self.assertRaisesRegex(ValueError, "ASCII"):
            tutorial_runtime.parse_definition(raw, expected_map="test-map")

    def make_server(self):
        sent = []
        server = SimpleNamespace(tutorial_definition=definition(), _tutorial_events=[], _tutorial_hit_events=[])
        server._send_packet_to_client = lambda ctx, packet, **kwargs: sent.append(packet) or True
        return server, sent

    def make_client(self):
        session = SimpleNamespace(in_game=True, team_id=1, local_epoch=2)
        return SimpleNamespace(
            client_id=7,
            session=session,
            player_pos=(0, 0, 3),
            player_vel=(0, 0, 0),
            player_heading=0.0,
            last_decoded_input={},
        )

    def make_ready(self, server, ctx, now=10.0):
        tutorial_runtime.update_player(server, ctx, now=now)
        ctx.tutorial_state["ready_at"] = now
        ctx.tutorial_state["prompt_retry_at"] = now
        tutorial_runtime.update_player(server, ctx, now=now)

    def test_ordered_regions_advance_and_finish(self):
        server, sent = self.make_server()
        ctx = self.make_client()
        self.make_ready(server, ctx)
        self.assertEqual(ctx.tutorial_state["index"], 0)
        ctx.player_pos = (15, 15, 3)
        tutorial_runtime.update_player(server, ctx, now=10.1)
        self.assertEqual(ctx.tutorial_state["index"], 1)
        ctx.player_pos = (35, 35, 3)
        tutorial_runtime.update_player(server, ctx, now=11.4)
        self.assertEqual(ctx.tutorial_state["index"], 2)
        self.assertTrue(ctx.tutorial_state["complete"])
        self.assertEqual([event["event"] for event in server._tutorial_events], ["prompt", "complete", "prompt", "complete"])
        self.assertEqual(len(sent), 5)

    def test_failed_prompt_does_not_activate_or_advance_and_retries(self):
        server, sent = self.make_server()
        attempts = 0

        def flaky(ctx, packet, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return False
            sent.append(packet)
            return True

        server._send_packet_to_client = flaky
        ctx = self.make_client()
        ctx.player_pos = (15, 15, 3)
        tutorial_runtime.update_player(server, ctx, now=10.0)
        ctx.tutorial_state["ready_at"] = 10.0
        ctx.tutorial_state["prompt_retry_at"] = 10.0
        tutorial_runtime.update_player(server, ctx, now=10.0)
        self.assertEqual(ctx.tutorial_state["index"], 0)
        self.assertFalse(ctx.tutorial_state["prompt_sent"])
        self.assertEqual(server._tutorial_events, [])
        tutorial_runtime.update_player(server, ctx, now=11.0)
        self.assertEqual(ctx.tutorial_state["index"], 1)
        self.assertFalse(ctx.tutorial_state["complete"])
        self.assertEqual(
            [event["event"] for event in server._tutorial_events],
            ["prompt", "complete"],
        )

    def test_stop_requires_entry_and_continuous_low_speed_dwell(self):
        server, _ = self.make_server()
        server.tutorial_definition = one_lesson({
            "type": "stop-in-region",
            "region": {"min": [10, 10, 0], "max": [20, 20, 30]},
            "maxHorizontalSpeed": 0.8,
            "dwellSeconds": 1.5,
            "requireEntry": True,
        })
        ctx = self.make_client()
        self.make_ready(server, ctx)
        ctx.player_pos = (15, 15, 3)
        ctx.player_vel = (2, 0, 0)
        tutorial_runtime.update_player(server, ctx, now=10.1)
        ctx.player_vel = (0, 0, 0)
        tutorial_runtime.update_player(server, ctx, now=10.2)
        tutorial_runtime.update_player(server, ctx, now=11.6)
        self.assertFalse(ctx.tutorial_state["complete"])
        tutorial_runtime.update_player(server, ctx, now=11.7)
        self.assertTrue(ctx.tutorial_state["complete"])

        # Reset while already in the bay: occupancy alone must not count as entry.
        tutorial_runtime.reset_player(server, ctx, announce=False)
        ctx.tutorial_state["ready_at"] = 20.0
        ctx.tutorial_state["prompt_retry_at"] = 20.0
        tutorial_runtime.update_player(server, ctx, now=20.0)
        tutorial_runtime.update_player(server, ctx, now=22.0)
        self.assertFalse(ctx.tutorial_state["complete"])
        ctx.player_pos = (0, 0, 3)
        tutorial_runtime.update_player(server, ctx, now=22.1)
        ctx.player_pos = (15, 15, 3)
        tutorial_runtime.update_player(server, ctx, now=22.2)
        tutorial_runtime.update_player(server, ctx, now=23.7)
        self.assertTrue(ctx.tutorial_state["complete"])

    def test_strafe_requires_ordered_input_distance_and_heading(self):
        trigger = {
            "type": "strafe-to-region",
            "startRegion": {"min": [-5, -5, 0], "max": [5, 5, 30]},
            "targetRegion": {"min": [-5, 195, 0], "max": [5, 205, 30]},
            "minStrafeInput": 0.25,
            "minStrafeSeconds": 1.0,
            "minLateralDistance": 150,
            "maxHeadingChangeDegrees": 8,
            "maxForwardInput": 0.15,
        }
        server, _ = self.make_server()
        server.tutorial_definition = one_lesson(trigger)
        ctx = self.make_client()
        self.make_ready(server, ctx)
        tutorial_runtime.update_player(server, ctx, now=10.1)  # arm in start bay
        ctx.last_decoded_input = {"strafe": 0.8, "fwd": 0.0}
        for tick in (10.35, 10.60, 10.85, 11.10):
            ctx.player_pos = (0, (tick - 10.1) * 100, 3)
            tutorial_runtime.update_player(server, ctx, now=tick)
        ctx.player_pos = (0, 200, 3)
        tutorial_runtime.update_player(server, ctx, now=11.35)
        self.assertTrue(ctx.tutorial_state["complete"])

    def test_strafe_rejects_stale_target_forward_and_heading_evidence(self):
        trigger = {
            "type": "strafe-to-region",
            "startRegion": {"min": [-5, -5, 0], "max": [5, 5, 30]},
            "targetRegion": {"min": [-5, 195, 0], "max": [5, 205, 30]},
            "minStrafeInput": 0.25,
            "minStrafeSeconds": 0.5,
            "minLateralDistance": 150,
            "maxHeadingChangeDegrees": 8,
            "maxForwardInput": 0.15,
        }
        for violation in ("stale", "forward", "heading"):
            server, _ = self.make_server()
            server.tutorial_definition = one_lesson(trigger)
            ctx = self.make_client()
            if violation == "stale":
                ctx.player_pos = (0, 200, 3)
            self.make_ready(server, ctx)
            if violation == "stale":
                tutorial_runtime.update_player(server, ctx, now=11.0)
                self.assertFalse(ctx.tutorial_state["complete"])
                continue
            tutorial_runtime.update_player(server, ctx, now=10.1)
            ctx.last_decoded_input = {"strafe": 0.8, "fwd": 0.5 if violation == "forward" else 0.0}
            ctx.player_heading = 0.25 if violation == "heading" else 0.0
            for tick in (10.35, 10.60, 10.85):
                ctx.player_pos = (0, min(200, (tick - 10.1) * 270), 3)
                tutorial_runtime.update_player(server, ctx, now=tick)
            ctx.player_pos = (0, 200, 3)
            tutorial_runtime.update_player(server, ctx, now=11.1)
            self.assertFalse(ctx.tutorial_state["complete"], violation)

    def test_wrong_team_does_not_start(self):
        server, sent = self.make_server()
        ctx = self.make_client()
        ctx.session.team_id = 2
        tutorial_runtime.update_player(server, ctx, now=10.0)
        self.assertFalse(hasattr(ctx, "tutorial_state"))
        self.assertEqual(sent, [])

    def test_progress_is_isolated_per_player(self):
        server, _ = self.make_server()
        first = self.make_client()
        second = self.make_client()
        second.client_id = 8
        self.make_ready(server, first)
        self.make_ready(server, second)
        first.player_pos = (15, 15, 3)
        tutorial_runtime.update_player(server, first, now=10.1)
        self.assertEqual(first.tutorial_state["index"], 1)
        self.assertEqual(second.tutorial_state["index"], 0)
        self.assertFalse(second.tutorial_state["complete"])

    def test_team_change_clears_evidence_before_reentry(self):
        server, _ = self.make_server()
        ctx = self.make_client()
        self.make_ready(server, ctx)
        ctx.player_pos = (15, 15, 3)
        tutorial_runtime.update_player(server, ctx, now=10.1)
        self.assertEqual(ctx.tutorial_state["index"], 1)
        ctx.session.team_id = 2
        tutorial_runtime.update_player(server, ctx, now=10.2)
        ctx.session.team_id = 1
        tutorial_runtime.update_player(server, ctx, now=10.3)
        self.assertEqual(ctx.tutorial_state["index"], 0)
        self.assertFalse(ctx.tutorial_state["complete"])

    def test_new_spawn_epoch_clears_completed_and_partial_evidence(self):
        server, _ = self.make_server()
        ctx = self.make_client()
        self.make_ready(server, ctx)
        ctx.player_pos = (15, 15, 3)
        tutorial_runtime.update_player(server, ctx, now=10.1)
        self.assertEqual(ctx.tutorial_state["index"], 1)
        ctx.tutorial_state["evidence"]["strafe_seconds"] = 0.75

        ctx.session.local_epoch = 3
        ctx.player_pos = (0, 0, 3)
        tutorial_runtime.update_player(server, ctx, now=10.2)
        self.assertEqual(ctx.tutorial_state["index"], 0)
        self.assertEqual(ctx.tutorial_state["spawn_epoch"], 3)
        self.assertEqual(ctx.tutorial_state["evidence"]["strafe_seconds"], 0.0)
        self.assertFalse(ctx.tutorial_state["complete"])

    def test_reset_replays_from_first_lesson(self):
        server, sent = self.make_server()
        ctx = self.make_client()
        ctx.tutorial_state = {
            "index": 2,
            "prompt_sent": True,
            "started_at": 1.0,
            "ready_at": 1.0,
            "spawn_epoch": 2,
            "complete": True,
        }
        tutorial_runtime.reset_player(server, ctx, announce=True)
        self.assertEqual(ctx.tutorial_state["index"], 0)
        self.assertFalse(ctx.tutorial_state["complete"])
        self.assertFalse(ctx.tutorial_state["evidence"]["entered"])
        self.assertEqual(ctx.tutorial_state["evidence"]["strafe_seconds"], 0.0)
        self.assertEqual(len(sent), 1)

    def combat_definition(self, trigger):
        return tutorial_runtime.parse_definition({
            "format": "wulfram-tutorial", "version": 1, "map": "test-map",
            "teams": [1],
            "targets": [{
                "id": "dummy", "sourceEntityId": "dummy-source", "token": "r",
                "team": 2, "position": [100, 0, 5], "protected": True,
            }],
            "lessons": [{
                "id": "combat", "prompt": "Do the combat step.", "trigger": trigger,
                "onComplete": {"message": "Combat done."},
            }],
        }, expected_map="test-map")

    def test_self_confirm_requires_prompt_and_region(self):
        server, _ = self.make_server()
        server.tutorial_definition = self.combat_definition({
            "type": "self-confirm", "region": {"min": [10, -5, 0], "max": [20, 5, 30]},
        })
        ctx = self.make_client()
        self.assertIn("waiting", tutorial_runtime.confirm_player(server, ctx))
        self.make_ready(server, ctx)
        self.assertIn("marked firing line", tutorial_runtime.confirm_player(server, ctx))
        ctx.player_pos = (15, 0, 3)
        self.assertEqual(tutorial_runtime.confirm_player(server, ctx), "Orientation confirmed.")
        tutorial_runtime.update_player(server, ctx, now=10.1)
        self.assertTrue(ctx.tutorial_state["complete"])

    def test_hit_target_rejects_old_wrong_and_low_strafe_events(self):
        server, _ = self.make_server()
        server.tutorial_definition = self.combat_definition({
            "type": "hit-target", "target": "dummy", "weapon": "PULSE_SHELL", "minStrafeInput": 0.25,
        })
        server.tutorial_definition["targets"]["dummy"]["runtime_oid"] = 10002
        ctx = self.make_client()
        tutorial_runtime.record_hit(server, ctx, target_kind="building", target_id=10002, weapon="PULSE_SHELL")
        self.make_ready(server, ctx)
        self.assertFalse(ctx.tutorial_state["complete"], "pre-lesson hit must not count")

        tutorial_runtime.record_hit(server, ctx, target_kind="building", target_id=10003, weapon="PULSE_SHELL")
        tutorial_runtime.record_hit(server, ctx, target_kind="building", target_id=10002, weapon="CHAIN GUN")
        tutorial_runtime.record_hit(server, ctx, target_kind="building", target_id=10002, weapon="PULSE_SHELL")
        tutorial_runtime.update_player(server, ctx, now=10.1)
        self.assertFalse(ctx.tutorial_state["complete"], "wrong target/weapon and no-strafe hits must not count")

        ctx.last_decoded_input = {"strafe": 0.5}
        tutorial_runtime.record_hit(server, ctx, target_kind="building", target_id=10002, weapon="PULSE_SHELL")
        tutorial_runtime.update_player(server, ctx, now=10.2)
        self.assertTrue(ctx.tutorial_state["complete"])

    def make_scenario_server(self):
        server, sent = self.make_server()
        server.tutorial_definition = tutorial_runtime.parse_definition({
            "format": "wulfram-tutorial", "version": 1, "map": "test-map",
            "teams": [1],
            "targets": [{
                "id": "objective", "sourceEntityId": "objective", "token": "e",
                "team": 2, "position": [100, 0, 13.5], "networkVisible": False,
            }],
            "scenario": {
                "kind": "free-skirmish", "objectiveTarget": "objective",
                "objectiveHealth": 200, "lossOnPlayerDeath": True,
                "lossMessage": "Scenario lost: tank destroyed.",
            },
            "lessons": [{
                "id": "destroy-objective", "prompt": "Destroy the objective.",
                "hint": "Use either approach.",
                "trigger": {"type": "destroy-target", "target": "objective"},
                "onComplete": {"message": "Scenario won."},
            }],
        }, expected_map="test-map")
        server._building_entities = {
            10002: SimpleNamespace(entity_type=25, team_id=2, x=100, y=0, z=13.5),
        }
        server._building_health = {10002: 800.0}
        server._building_max_health = {10002: 800.0}
        server._snapshot_in_game_clients = lambda: ()
        server._broadcast_building_delete = lambda oid, **kwargs: 0
        tutorial_runtime.resolve_targets(server)
        return server, sent

    def test_destroy_target_requires_fresh_exact_attributed_destruction(self):
        server, _ = self.make_scenario_server()
        ctx = self.make_client()
        tutorial_runtime.record_hit(
            server, ctx, target_kind="building", target_id=10002,
            weapon="PULSE_SHELL", destroyed=True,
        )
        self.make_ready(server, ctx)
        self.assertFalse(ctx.tutorial_state["complete"], "pre-prompt destruction must not count")

        other = self.make_client()
        other.client_id = 8
        tutorial_runtime.record_hit(
            server, ctx, target_kind="building", target_id=10002,
            weapon="PULSE_SHELL", destroyed=False,
        )
        tutorial_runtime.record_hit(
            server, other, target_kind="building", target_id=10002,
            weapon="PULSE_SHELL", destroyed=True,
        )
        tutorial_runtime.record_hit(
            server, ctx, target_kind="building", target_id=10003,
            weapon="PULSE_SHELL", destroyed=True,
        )
        tutorial_runtime.update_player(server, ctx, now=10.1)
        self.assertFalse(ctx.tutorial_state["complete"])

        tutorial_runtime.record_hit(
            server, ctx, target_kind="building", target_id=10002,
            weapon="PULSE_SHELL", destroyed=True,
        )
        tutorial_runtime.update_player(server, ctx, now=10.2)
        self.assertTrue(ctx.tutorial_state["complete"])
        self.assertEqual(ctx.tutorial_last_scenario_result, "win")
        self.assertFalse(tutorial_runtime.record_player_death(server, ctx))
        self.assertEqual(ctx.tutorial_last_scenario_result, "win")
        self.assertEqual(
            [event["event"] for event in server._tutorial_events].count("scenario_loss"), 0,
        )

    def test_scenario_death_loss_and_reset_restore_objective(self):
        server, _ = self.make_scenario_server()
        ctx = self.make_client()
        ctx.known_entity_ids = {10002}
        ctx._entity_create_times = {10002: 1.0}
        ctx.tutorial_network_target_oids = {10002}
        server._snapshot_in_game_clients = lambda: (ctx,)
        deleted = []
        server._broadcast_building_delete = lambda oid, **kwargs: deleted.append(oid) or 1
        self.make_ready(server, ctx)

        self.assertTrue(tutorial_runtime.record_player_death(server, ctx))
        self.assertEqual(ctx.tutorial_last_scenario_result, "loss")
        self.assertEqual(server._tutorial_events[-1]["reason"], "player_death")

        server._building_health[10002] = 50.0
        tutorial_runtime.reset_player(server, ctx, announce=False)
        self.assertEqual(server._building_health[10002], 200.0)
        self.assertEqual(server._building_max_health[10002], 200.0)
        self.assertEqual(deleted, [10002])
        self.assertNotIn(10002, ctx.known_entity_ids)
        self.assertNotIn(10002, ctx.tutorial_network_target_oids)
        self.assertIsNone(ctx.tutorial_last_scenario_result)

    def test_scenario_threat_waits_for_gate_and_stays_quiet_after_result(self):
        parsed = tutorial_runtime.parse_definition({
            "format": "wulfram-tutorial", "version": 1, "map": "test-map",
            "teams": [1],
            "targets": [
                {"id": "turret", "sourceEntityId": "turret", "token": "g",
                 "team": 2, "position": [200, 0, 5]},
                {"id": "objective", "sourceEntityId": "objective", "token": "e",
                 "team": 2, "position": [240, 0, 13.5]},
            ],
            "scenario": {
                "kind": "free-skirmish", "objectiveTarget": "objective",
                "objectiveHealth": 200, "threatTarget": "turret",
                "threatActivationLesson": "cross-gate", "quietAfterResult": True,
            },
            "lessons": [
                {"id": "practice", "prompt": "Practice first.",
                 "trigger": {"type": "hit-target", "target": "objective", "weapon": "CHAIN GUN"},
                 "onComplete": {"message": "Practice done."}},
                {"id": "cross-gate", "prompt": "Cross when ready.",
                 "trigger": {"type": "enter-region", "region": {"min": [10, -5, 0], "max": [20, 5, 30]}},
                 "onComplete": {"message": "Threat active."}},
                {"id": "destroy", "prompt": "Destroy it.",
                 "trigger": {"type": "destroy-target", "target": "objective"},
                 "onComplete": {"message": "Won."}},
            ],
        }, expected_map="test-map")
        parsed["targets"]["turret"]["runtime_oid"] = 10001
        parsed["targets"]["objective"]["runtime_oid"] = 10002
        server, _ = self.make_server()
        server.tutorial_definition = parsed
        ctx = self.make_client()
        ctx.tutorial_state = {"index": 0, "complete": False}
        ctx.tutorial_last_scenario_result = None
        self.assertTrue(tutorial_runtime.suppress_turret_fire(server, ctx, 10001))
        ctx.tutorial_state["index"] = 1
        self.assertTrue(tutorial_runtime.suppress_turret_fire(server, ctx, 10001))
        ctx.tutorial_state["index"] = 2
        self.assertFalse(tutorial_runtime.suppress_turret_fire(server, ctx, 10001))
        ctx.tutorial_state["complete"] = True
        self.assertTrue(tutorial_runtime.suppress_turret_fire(server, ctx, 10001))
        self.assertFalse(tutorial_runtime.suppress_turret_fire(server, ctx, 99999))

    def test_scenario_target_health_feedback_miss_throttle_and_reset(self):
        server, sent = self.make_scenario_server()
        target = server.tutorial_definition["targets"]["objective"]
        target["label"] = "Command Power Cell"
        ctx = self.make_client()
        ctx.known_entity_ids = {10002}
        ctx._entity_create_times = {10002: 1.0}
        ctx.tutorial_network_target_oids = {10002}
        server._snapshot_in_game_clients = lambda: (ctx,)
        ctx.tutorial_state = {"index": 0, "complete": False}
        tutorial_runtime.record_hit(
            server, ctx, target_kind="building", target_id=10002,
            weapon="CHAIN GUN", old_health=200, new_health=180, max_health=200,
        )
        self.assertEqual(len(sent), 1, "a real scenario hit should send remaining-health feedback")
        self.assertTrue(tutorial_runtime.record_weapon_miss(
            server, ctx, weapon="CHAIN GUN", reason="none",
        ))
        self.assertFalse(tutorial_runtime.record_weapon_miss(
            server, ctx, weapon="CHAIN GUN", reason="none",
        ), "rapid miss feedback must be throttled")
        server._building_health[10002] = 50.0
        tutorial_runtime.reset_player(server, ctx, announce=False)
        self.assertEqual(server._building_health[10002], 200.0)

    def test_turret_damage_requires_fresh_exact_source_and_survival(self):
        server, _ = self.make_server()
        server.tutorial_definition = self.combat_definition({
            "type": "turret-damage", "target": "dummy", "minDamage": 0.08,
            "ceaseFireOnComplete": True,
        })
        server.tutorial_definition["targets"]["dummy"]["runtime_oid"] = 10002
        ctx = self.make_client()
        ctx.player_health = 1.0
        ctx.last_damage_time = 9.0
        ctx.last_damage_source = "turret:GUN_TURRET:oid=10002"
        ctx.last_damage_amount = 0.08
        self.make_ready(server, ctx)
        self.assertFalse(ctx.tutorial_state["complete"], "pre-prompt turret damage must not count")
        activated = float(ctx.tutorial_state["evidence"]["activated_at"])

        ctx.last_damage_time = activated + 0.1
        ctx.last_damage_source = "turret:GUN_TURRET:oid=10003"
        tutorial_runtime.update_player(server, ctx, now=10.1)
        self.assertFalse(ctx.tutorial_state["complete"], "a different turret must not count")

        ctx.last_damage_time = activated + 0.2
        ctx.last_damage_source = "turret:GUN_TURRET:oid=10002"
        ctx.last_damage_amount = 0.04
        tutorial_runtime.update_player(server, ctx, now=10.2)
        self.assertFalse(ctx.tutorial_state["complete"], "sub-threshold damage must not count")

        ctx.last_damage_time = activated + 0.3
        ctx.last_damage_amount = 0.08
        ctx.player_health = 0.0
        tutorial_runtime.update_player(server, ctx, now=10.3)
        self.assertFalse(ctx.tutorial_state["complete"], "a destroyed learner must not pass")

        ctx.player_health = 0.92
        tutorial_runtime.update_player(server, ctx, now=10.4)
        self.assertTrue(ctx.tutorial_state["complete"])
        self.assertTrue(tutorial_runtime.suppress_turret_fire(server, ctx, 10002))
        self.assertFalse(tutorial_runtime.suppress_turret_fire(server, ctx, 10003))

        tutorial_runtime.reset_player(server, ctx, announce=False)
        self.assertFalse(
            tutorial_runtime.suppress_turret_fire(server, ctx, 10002),
            "reset must re-arm the scored tutorial turret",
        )

    def test_target_resolution_is_exact_and_protection_is_explicit(self):
        server, _ = self.make_server()
        server.tutorial_definition = self.combat_definition({
            "type": "hit-target", "target": "dummy", "weapon": "CHAIN GUN",
        })
        server._building_entities = {
            10001: SimpleNamespace(entity_type=27, team_id=1, x=0, y=0, z=5),
            10002: SimpleNamespace(entity_type=27, team_id=2, x=100, y=0, z=5),
        }
        tutorial_runtime.resolve_targets(server)
        self.assertEqual(server.tutorial_definition["targets"]["dummy"]["runtime_oid"], 10002)
        self.assertTrue(tutorial_runtime.is_protected_target(server, 10002))
        self.assertFalse(tutorial_runtime.is_protected_target(server, 10001))

    def test_network_visible_target_uses_authoritative_static_building(self):
        parsed = self.combat_definition({
            "type": "hit-target", "target": "dummy", "weapon": "CHAIN GUN",
        })
        parsed["targets"]["dummy"]["network_visible"] = True
        parsed["targets"]["dummy"]["runtime_oid"] = 10002
        server, _ = self.make_server()
        server.tutorial_definition = parsed
        server._building_entities = {
            10002: SimpleNamespace(
                entity_type=27, team_id=2, x=100, y=0, z=5, heading=1.25,
            ),
        }
        ctx = self.make_client()
        ctx.session.translation_ack_received = True
        calls = []
        from unittest.mock import patch
        with patch("wulfram.build_uplink.send_dynamic_entity_definition") as send:
            send.side_effect = lambda *args, **kwargs: calls.append(kwargs) or True
            self.assertTrue(tutorial_runtime._ensure_network_targets(server, ctx))
            self.assertTrue(tutorial_runtime._ensure_network_targets(server, ctx))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["entity_id"], 10002)
        self.assertEqual(calls[0]["pos"], (100.0, 0.0, 5.0))
        self.assertEqual(calls[0]["heading"], 1.25)

    def service_definition(self):
        result = tutorial_runtime.parse_definition({
            "format": "wulfram-tutorial", "version": 1, "map": "test-map", "teams": [1],
            "targets": [
                {"id": "repair", "sourceEntityId": "repair", "token": "r", "team": 1,
                 "position": [0, 0, 5]},
                {"id": "fuel", "sourceEntityId": "fuel", "token": "f", "team": 1,
                 "position": [0, 100, 5]},
            ],
            "lessons": [{
                "id": "services", "prompt": "Recover.",
                "trigger": {
                    "type": "service-recovery", "repairTarget": "repair", "fuelTarget": "fuel",
                    "minHealthRestored": 0.2, "minEnergyRestored": 40,
                    "fixtureHealth": 0.65, "fixtureEnergy": 15,
                },
                "onComplete": {"message": "Recovered."},
            }],
        }, expected_map="test-map")
        result["targets"]["repair"]["runtime_oid"] = 10001
        result["targets"]["fuel"]["runtime_oid"] = 10002
        return result

    def test_service_recovery_requires_attributed_post_prompt_events(self):
        server, _ = self.make_server()
        server.tutorial_definition = self.service_definition()
        server._tutorial_service_events = []
        ctx = self.make_client()
        ctx.player_health = 1.0
        ctx.player_energy = 100.0

        # Old evidence and a failed prompt cannot apply the training fixture.
        tutorial_runtime.record_service(server, ctx, service="repair", target_id=10001, delta=0.5)
        attempts = 0
        def flaky(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            return attempts > 1
        server._send_packet_to_client = flaky
        tutorial_runtime.update_player(server, ctx, now=10.0)
        ctx.tutorial_state["ready_at"] = 10.0
        ctx.tutorial_state["prompt_retry_at"] = 10.0
        tutorial_runtime.update_player(server, ctx, now=10.0)
        self.assertEqual((ctx.player_health, ctx.player_energy), (1.0, 100.0))
        tutorial_runtime.update_player(server, ctx, now=11.0)
        self.assertEqual((ctx.player_health, ctx.player_energy), (0.65, 15.0))
        self.assertTrue(tutorial_runtime.suppress_passive_energy(server, ctx))

        # Passive changes and service from the wrong buildings do not count.
        ctx.player_health = 0.95
        ctx.player_energy = 95.0
        tutorial_runtime.record_service(server, ctx, service="repair", target_id=999, delta=0.3)
        tutorial_runtime.record_service(server, ctx, service="fuel", target_id=999, delta=80)
        tutorial_runtime.update_player(server, ctx, now=11.1)
        self.assertFalse(ctx.tutorial_state["complete"])

        tutorial_runtime.record_service(server, ctx, service="repair", target_id=10001, delta=0.2)
        tutorial_runtime.record_service(server, ctx, service="fuel", target_id=10002, delta=40)
        tutorial_runtime.update_player(server, ctx, now=11.2)
        self.assertTrue(ctx.tutorial_state["complete"])
        self.assertFalse(tutorial_runtime.suppress_passive_energy(server, ctx))

    def test_retry_restores_then_reapplies_service_fixture(self):
        server, _ = self.make_server()
        server.tutorial_definition = self.service_definition()
        server._tutorial_service_events = []
        ctx = self.make_client()
        ctx.player_health = 1.0
        ctx.player_energy = 100.0
        self.make_ready(server, ctx)
        self.assertEqual((ctx.player_health, ctx.player_energy), (0.65, 15.0))
        tutorial_runtime.retry_player(server, ctx)
        self.assertEqual((ctx.player_health, ctx.player_energy), (1.0, 100.0))
        ctx.tutorial_state["ready_at"] = 11.0
        ctx.tutorial_state["prompt_retry_at"] = 11.0
        tutorial_runtime.update_player(server, ctx, now=11.0)
        self.assertEqual((ctx.player_health, ctx.player_energy), (0.65, 15.0))

    def logistics_definition(self):
        return tutorial_runtime.parse_definition({
            "format": "wulfram-tutorial", "version": 1, "map": "test-map",
            "teams": [1], "maxConcurrentLearners": 1,
            "lessons": [
                {
                    "id": "pickup", "prompt": "Pick it up.",
                    "trigger": {"type": "cargo-pickup", "binding": "power-crate",
                                "cargoType": 25, "seedPosition": [20, 0, 5], "team": 1},
                    "onComplete": {"message": "Picked up."},
                },
                {
                    "id": "carry", "prompt": "Carry it.",
                    "trigger": {"type": "cargo-carry-to-region", "binding": "power-crate",
                                "cargoType": 25,
                                "region": {"min": [90, -10, 0], "max": [110, 10, 30]},
                                "requireEntry": True},
                    "onComplete": {"message": "Carried."},
                },
                {
                    "id": "drop", "prompt": "Drop it.",
                    "trigger": {"type": "cargo-drop", "binding": "power-crate",
                                "bindAs": "dropped-power", "cargoType": 25,
                                "region": {"min": [90, -10, 0], "max": [120, 10, 30]}},
                    "onComplete": {"message": "Dropped."},
                },
                {
                    "id": "recover", "prompt": "Recover it.",
                    "trigger": {"type": "cargo-recover", "binding": "dropped-power",
                                "cargoType": 25, "minAwayDistance": 40},
                    "onComplete": {"message": "Recovered."},
                },
            ],
        }, expected_map="test-map")

    def make_logistics_server(self):
        server, sent = self.make_server()
        server.tutorial_definition = self.logistics_definition()
        server._tutorial_logistics_events = []
        server._dropped_cargo = {}
        server._next_crate = 40000
        def drop(pos, cargo_type, team):
            oid = server._next_crate
            server._next_crate += 1
            server._dropped_cargo[oid] = {
                "pos": tuple(pos), "cargo_type": cargo_type, "team_id": team,
            }
            return oid
        server._drop_cargo_crate = drop
        server._broadcast_building_delete = lambda *args, **kwargs: 1
        server._set_player_carry = lambda ctx, **values: [setattr(ctx, key, value) for key, value in values.items()]
        return server, sent

    def ready_current(self, server, ctx, now):
        ctx.tutorial_state["ready_at"] = now
        ctx.tutorial_state["prompt_retry_at"] = now
        tutorial_runtime.update_player(server, ctx, now=now)

    def test_logistics_chain_requires_exact_owned_crate_drop_and_recovery(self):
        server, _ = self.make_logistics_server()
        ctx = self.make_client()
        ctx.cargo_type = 0
        ctx.cargo_count = 0
        ctx.has_uplink = False
        self.make_ready(server, ctx)
        seeded = ctx.tutorial_state["bindings"]["power-crate"]
        self.assertEqual(seeded["oid"], 40000)

        # Wrong OID/type and another player's evidence cannot advance pickup.
        other = self.make_client(); other.client_id = 8
        tutorial_runtime.record_logistics_event(
            server, other, action="cargo_pickup", oid=40000, entity_type=25,
            position=(20, 0, 5), fixture_binding="power-crate",
        )
        tutorial_runtime.record_logistics_event(
            server, ctx, action="cargo_pickup", oid=49999, entity_type=25,
            position=(20, 0, 5), fixture_binding="power-crate",
        )
        tutorial_runtime.update_player(server, ctx, now=10.1)
        self.assertEqual(ctx.tutorial_state["index"], 0)

        ctx.cargo_type = 25; ctx.cargo_count = 1
        ctx.tutorial_fixture_cargo = {"tutorial_map": "test-map", "tutorial_binding": "power-crate"}
        tutorial_runtime.record_logistics_event(
            server, ctx, action="cargo_pickup", oid=40000, entity_type=25,
            position=(20, 0, 5), fixture_binding="power-crate",
        )
        tutorial_runtime.update_player(server, ctx, now=10.2)
        self.assertEqual(ctx.tutorial_state["index"], 1)

        self.ready_current(server, ctx, 11.5)
        ctx.player_pos = (100, 0, 3)
        tutorial_runtime.update_player(server, ctx, now=11.6)
        self.assertEqual(ctx.tutorial_state["index"], 2)

        self.ready_current(server, ctx, 12.9)
        tutorial_runtime.record_logistics_event(
            server, ctx, action="cargo_drop", oid=40001, entity_type=25,
            position=(150, 0, 5), fixture_binding="power-crate",
        )
        tutorial_runtime.update_player(server, ctx, now=13.0)
        self.assertEqual(ctx.tutorial_state["index"], 2, "drop outside staging must not count")
        tutorial_runtime.record_logistics_event(
            server, ctx, action="cargo_drop", oid=40002, entity_type=25,
            position=(112, 0, 5), fixture_binding="power-crate",
        )
        tutorial_runtime.update_player(server, ctx, now=13.1)
        self.assertEqual(ctx.tutorial_state["bindings"]["dropped-power"]["oid"], 40002)
        self.assertEqual(ctx.tutorial_state["index"], 3)

        self.ready_current(server, ctx, 14.4)
        tutorial_runtime.record_logistics_event(
            server, ctx, action="cargo_pickup", oid=40002, entity_type=25,
            position=(112, 0, 5), fixture_binding="power-crate",
        )
        tutorial_runtime.update_player(server, ctx, now=14.5)
        self.assertFalse(ctx.tutorial_state["complete"], "pickup without moving away must not count")
        ctx.player_pos = (160, 0, 3)
        tutorial_runtime.update_player(server, ctx, now=14.6)
        tutorial_runtime.record_logistics_event(
            server, ctx, action="cargo_pickup", oid=40002, entity_type=25,
            position=(112, 0, 5), fixture_binding="power-crate",
        )
        tutorial_runtime.update_player(server, ctx, now=14.7)
        self.assertTrue(ctx.tutorial_state["complete"])

    def test_failed_cargo_prompt_does_not_seed_and_reset_cleans_fixture(self):
        server, _ = self.make_logistics_server()
        ctx = self.make_client()
        ctx.cargo_type = 0; ctx.cargo_count = 0; ctx.has_uplink = False
        server._send_packet_to_client = lambda *args, **kwargs: False
        tutorial_runtime.update_player(server, ctx, now=10.0)
        ctx.tutorial_state["ready_at"] = 10.0
        ctx.tutorial_state["prompt_retry_at"] = 10.0
        tutorial_runtime.update_player(server, ctx, now=10.0)
        self.assertEqual(server._dropped_cargo, {})
        server._send_packet_to_client = lambda *args, **kwargs: True
        ctx.tutorial_state["prompt_retry_at"] = 11.0
        tutorial_runtime.update_player(server, ctx, now=11.0)
        self.assertEqual(len(server._dropped_cargo), 1)
        ctx.cargo_type = 25; ctx.cargo_count = 1
        ctx.tutorial_fixture_cargo = {"tutorial_map": "test-map", "tutorial_binding": "power-crate"}
        tutorial_runtime.reset_player(server, ctx, announce=False)
        self.assertEqual(server._dropped_cargo, {})
        self.assertEqual((ctx.cargo_type, ctx.cargo_count), (0, 0))

    def test_reset_forgets_native_map_fixture_without_broadcasting_delete(self):
        server, _ = self.make_logistics_server()
        deleted = []
        server._broadcast_building_delete = (
            lambda oid, **kwargs: deleted.append(oid) or 1
        )
        ctx = self.make_client()
        ctx.cargo_type = 0; ctx.cargo_count = 0; ctx.has_uplink = False
        self.make_ready(server, ctx)
        seeded = ctx.tutorial_state["bindings"]["power-crate"]
        crate = server._dropped_cargo[seeded["oid"]]
        crate["tutorial_native_map"] = True
        tutorial_runtime.reset_player(server, ctx, announce=False)
        self.assertEqual(server._dropped_cargo, {})
        self.assertEqual(deleted, [])

    def test_deploy_and_service_require_exact_bound_building(self):
        server, _ = self.make_logistics_server()
        server.tutorial_definition = tutorial_runtime.parse_definition({
            "format": "wulfram-tutorial", "version": 1, "map": "test-map", "teams": [1],
            "lessons": [
                {
                    "id": "deploy", "prompt": "Deploy it.",
                    "trigger": {
                        "type": "cargo-deploy", "binding": "power-crate",
                        "bindAs": "deployed-power", "cargoType": 25,
                        "region": {"min": [90, -10, 0], "max": [120, 10, 30]},
                    },
                    "onComplete": {"message": "Deployed."},
                },
                {
                    "id": "service", "prompt": "Use its service.",
                    "trigger": {
                        "type": "deployed-service", "binding": "deployed-power",
                        "service": "energy", "minRestored": 6, "fixtureEnergy": 10,
                    },
                    "onComplete": {"message": "Serviced."},
                },
            ],
        }, expected_map="test-map")
        server._tutorial_service_events = []
        ctx = self.make_client()
        ctx.player_energy = 100.0
        ctx.player_health = 1.0
        ctx.tutorial_fixture_cargo = {
            "tutorial_map": "test-map", "tutorial_binding": "power-crate",
            "tutorial_owner_client_id": ctx.client_id,
        }
        tutorial_runtime.record_logistics_event(
            server, ctx, action="cargo_deploy", oid=29999, entity_type=25,
            position=(105, 0, 5), fixture_binding="power-crate",
        )
        self.make_ready(server, ctx)
        self.assertEqual(ctx.tutorial_state["index"], 0, "pre-prompt deployment must be ignored")

        # Wrong binding and an out-of-region deployment cannot satisfy the step.
        tutorial_runtime.record_logistics_event(
            server, ctx, action="cargo_deploy", oid=30000, entity_type=25,
            position=(100, 0, 5), fixture_binding="other-crate",
        )
        tutorial_runtime.record_logistics_event(
            server, ctx, action="cargo_deploy", oid=30001, entity_type=25,
            position=(150, 0, 5), fixture_binding="power-crate",
        )
        tutorial_runtime.update_player(server, ctx, now=10.1)
        self.assertEqual(ctx.tutorial_state["index"], 0)

        tutorial_runtime.record_logistics_event(
            server, ctx, action="cargo_deploy", oid=30002, entity_type=25,
            position=(105, 0, 5), fixture_binding="power-crate",
        )
        tutorial_runtime.update_player(server, ctx, now=10.2)
        self.assertEqual(ctx.tutorial_state["bindings"]["deployed-power"]["oid"], 30002)
        self.assertEqual(ctx.tutorial_state["index"], 1)

        self.ready_current(server, ctx, 11.5)
        self.assertEqual(ctx.player_energy, 10.0)
        tutorial_runtime.record_service(server, ctx, service="energy", target_id=39999, delta=20)
        tutorial_runtime.update_player(server, ctx, now=11.6)
        self.assertFalse(ctx.tutorial_state["complete"], "another power cell must not count")
        tutorial_runtime.record_service(server, ctx, service="energy", target_id=30002, delta=6)
        tutorial_runtime.update_player(server, ctx, now=11.7)
        self.assertTrue(ctx.tutorial_state["complete"])

    def test_reset_removes_only_owned_tutorial_deployments(self):
        server, _ = self.make_logistics_server()
        server._dynamic_building_sources = {
            30000: {"tutorial_fixture": {
                "tutorial_map": "test-map", "tutorial_owner_client_id": 7,
            }},
            30001: {"tutorial_fixture": {
                "tutorial_map": "test-map", "tutorial_owner_client_id": 8,
            }},
            30002: {"source": "normal"},
        }
        removed = []
        server._remove_dynamic_building_record = lambda oid: (
            removed.append(oid), server._dynamic_building_sources.pop(oid, None)
        )
        ctx = self.make_client()
        ctx.cargo_type = 0; ctx.cargo_count = 0; ctx.has_uplink = False
        tutorial_runtime.reset_player(server, ctx, announce=False)
        self.assertEqual(removed, [30000])
        self.assertEqual(set(server._dynamic_building_sources), {30001, 30002})

    def test_disconnect_cleanup_removes_owned_fixture_and_evidence(self):
        server, _ = self.make_logistics_server()
        server._dynamic_building_sources = {
            30000: {"tutorial_fixture": {
                "tutorial_map": "test-map", "tutorial_owner_client_id": 7,
            }},
        }
        removed = []
        server._remove_dynamic_building_record = lambda oid: (
            removed.append(oid), server._dynamic_building_sources.pop(oid, None)
        )
        ctx = self.make_client()
        ctx.cargo_type = 25; ctx.cargo_count = 1; ctx.has_uplink = False
        ctx.tutorial_fixture_cargo = {
            "tutorial_map": "test-map", "tutorial_binding": "power-crate",
        }
        ctx.tutorial_state = {"evidence": {"fixture_applied": False}}
        tutorial_runtime.cleanup_player(server, ctx)
        self.assertEqual((ctx.cargo_type, ctx.cargo_count), (0, 0))
        self.assertEqual(removed, [30000])
        self.assertIsNone(ctx.tutorial_state)


if __name__ == "__main__":
    unittest.main()
