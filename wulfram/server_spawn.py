"""SpawnMixin -- spawn-point selection and position resolution, extracted
verbatim from WulframServer (server.py decomposition, step 4). Method-only
mixin; shares state via `self`. Imports only stdlib -> leaf, no cycle.
"""
from __future__ import annotations

import os
from typing import Optional

from .client import ClientContext
from .session import Phase
from .packets import build_delete_object


class SpawnMixin:
    def get_spawn_points(self) -> list:
        """Return spawn point list for current map/config."""
        if self.map_spawn_points is None:
            self._load_map_spawn_points()
        if self.map_spawn_points:
            return [dict(sp) for sp in self.map_spawn_points]
        return [
            {"oid": 5001, "team": 1, "x": 50.0, "y": 10.0, "z": 50.0},
            {"oid": 5002, "team": 1, "x": 60.0, "y": 10.0, "z": 50.0},
            {"oid": 5003, "team": 2, "x": 150.0, "y": 10.0, "z": 150.0},
            {"oid": 5004, "team": 2, "x": 160.0, "y": 10.0, "z": 150.0},
        ]

    def _pick_spawn_point(self, team_id: int) -> Optional[dict]:
        """Pick the first spawn point for the requested team."""
        if not self.use_map_spawn_points:
            return None
        points = self.get_spawn_points()
        for sp in points:
            if sp.get("team") == team_id:
                return sp
        return points[0] if points else None

    def _parse_spawn_pos_env(self, raw: str) -> Optional[tuple[float, float, float]]:
        """Parse `x,y,z` spawn position env/config values."""
        raw = (raw or "").strip()
        if not raw or raw.lower() in ("none", "null"):
            return None
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) < 2:
            return None
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2]) if len(parts) > 2 else getattr(self, "spawn_height", 5.0)
        except ValueError:
            return None
        return (x, y, z)

    def _get_builtin_flat_spawn_pos(self) -> tuple[float, float, float]:
        """Return the built-in flat default spawn for the active map."""
        map_name = getattr(self, "map_name", "crossroads")
        up_axis = getattr(self, "up_axis", "z")
        spawn_height = getattr(self, "spawn_height", 5.0)
        if map_name.lower() == "crossroads":
            if up_axis == "z":
                return (4950.0, 5100.0, spawn_height)
            return (4950.0, spawn_height, 5100.0)
        if up_axis == "z":
            return (100.0, 100.0, spawn_height)
        return (100.0, spawn_height, 100.0)

    def _get_configured_default_spawn_pos(self) -> Optional[tuple[float, float, float]]:
        """Return the configured default spawn that should win over map pads."""
        spawn_override = self._parse_spawn_pos_env(os.environ.get("WULFRAM_SPAWN_POS", ""))
        if spawn_override is not None:
            return spawn_override
        if getattr(self, "force_default_spawn_pos", False):
            default_flat_spawn_pos = getattr(self, "default_flat_spawn_pos", None)
            if default_flat_spawn_pos is not None:
                return default_flat_spawn_pos
        return None

    def _resolve_spawn_pos(
        self,
        team_id: int,
        *,
        explicit_pos: Optional[tuple[float, float, float]] = None,
        allow_map_spawn: bool = True,
    ) -> tuple[float, float, float]:
        """Resolve the spawn position for normal join/respawn flows."""
        if explicit_pos is not None:
            return explicit_pos
        configured_default = self._get_configured_default_spawn_pos()
        if configured_default is not None:
            return configured_default
        if allow_map_spawn:
            map_spawn = self._pick_spawn_point(team_id)
            if map_spawn:
                return (map_spawn["x"], map_spawn["y"], map_spawn["z"])
        return self._get_builtin_flat_spawn_pos()

    def _enter_death_deploy_state(self, target: ClientContext) -> int:
        """Put a just-killed client into the death/deploy screen WITHOUT auto-spawning.

        GOAL 4: combat respawn used to reuse the *initial-deploy* auto-flow that we
        already retired for fresh spawns -- it set
        ``delayed_spawn_team = team_id or 1`` + ``delayed_spawn_time = now+5s``, which
        the tick loop fired via ``_auto_join_team`` with no death countdown and no
        flag-click. Two failures fell out of that:
          1. Instant respawn (no death screen) because the timer just re-spawned you.
          2. "Neutral/wrong team" because ``team_id or 1`` silently coerces a 0/missing
             team to red(1) -- and that same coercion also leaks into the UPDATE_STATS
             broadcast (``_broadcast_player_stats`` sends ``team_id or 1``), which writes
             ``PlayerEntry+0x08`` on every client (Social.c:5501) and would re-corrupt
             the GOAL 3 roster fix.

        We now mirror the explicit flag-click deploy model: cancel any auto-spawn timer,
        PRESERVE the player's team across death, and leave IN_GAME for TEAM_SELECT so the
        next REINCARNATE 0x00 (map-flag click -> ``handle_spawn_at_point``) is treated as
        a fresh deploy rather than the "Already spawned" IN_GAME branch. No
        ``_is_loopback_client`` fork (exit condition e): identical for every client.

        Returns the preserved team_id (for logging/assertions).
        """
        sess = target.session
        # PRESERVE team across death. Never zero it, never default it here -- the
        # redeploy must land on the SAME team. handle_spawn_at_point reads the clicked
        # flag's team and falls back to this preserved value.
        preserved_team = sess.team_id
        # Belt-and-suspenders: make sure no stale auto-spawn timer can fire. THIS is the
        # line that used to schedule the instant respawn; we explicitly clear it instead.
        sess.delayed_spawn_team = 0
        sess.delayed_spawn_time = 0.0
        sess.pending_spawn_team_id = 0
        # Re-roster on redeploy. The OG client empties its in-game player list when it
        # drops to the team-select/deploy screen on death (the deploy screen shows
        # "0 players"), so the surviving roster entry is gone. ADD_TO_ROSTER is gated by
        # session.roster_sent (sent once per session, server.py ~4507) -> without this
        # reset the redeploy never re-asserts the entry and the P-scoreboard stays empty
        # after respawn (looks like a team-corruption regression but is really a
        # roster-resend gap). Clearing the flag makes the next _spawn_wf_style re-send it.
        sess.roster_sent = False
        # Leave IN_GAME -> death/deploy. in_game bool is already cleared by the caller;
        # move the phase enum too so server state is consistent and the next flag-click
        # spawn is not rejected as a duplicate.
        sess.in_game = False
        if sess.phase == Phase.IN_GAME:
            sess.transition_to(Phase.TEAM_SELECT)
        # NOTE: death leaves the OG client's g_player_team==0, which makes mode-3
        # (TabMenu_initialize_team, Screens.c:1314) HIDE the ENTRY-MAP/deploy tab
        # and force SWITCH-TEAM -- the player "loses their team" and must re-pick
        # instead of seeing the deploy countdown. A trailing UPDATE_STATS for the
        # preserved team (player_id = real id AND 0) does NOT re-set g_player_team
        # post-death (tested 2026-06-24): the client doesn't apply UPDATE_STATS ->
        # g_player_team off the in-game path (or the death DELETE-clear races it).
        # The clear is a side effect of our DELETE_OBJECT clearing the local-player
        # region; OG-faithful "keep team on death" likely needs a death signal that
        # does NOT delete/clear the local-player entity. Left as a TODO.
        # NOTE: an unsolicited server-sent team-confirm here (REINCARNATE 0x11 +
        # roster + UPDATE_STATS for the preserved team) BOUNCES the OG client to
        # the title/login screen (tested 2026-06-23, WULFRAM_DEATH_REINCARNATE_ENTRY)
        # -- the client only accepts that handshake when IT initiated the team
        # click. So death->entry-map cannot be routed from the server this way.
        print(
            f"[COMBAT] c{target.client_id} -> death/deploy "
            f"(team={preserved_team} preserved, awaiting flag-click redeploy; no auto-spawn)"
        )
        return preserved_team

    def _kill_player_for_deploy(self, target: ClientContext, *, attacker: ClientContext = None,
                                reason: str = "control") -> None:
        """Full death sequence ending in the GOAL-4 death/deploy state (no auto-spawn).

        Mirrors the combat-kill cleanup (stats, DELETE_OBJECT with explosion, entity
        bookkeeping, server-state reset) and then hands off to
        ``_enter_death_deploy_state`` so the victim must redeploy via an explicit
        flag-click on their preserved team. Used by the ``damage``/``dmg`` control
        command so a control-port kill exercises the SAME path as a real projectile
        kill (otherwise a control-kill would leave the session IN_GAME and the redeploy
        guard would reject the flag-click as a duplicate spawn).
        """
        target.deaths += 1
        participants = (attacker, target) if attacker is not None else (target,)
        if attacker is not None:
            attacker.kills += 1
            self._broadcast_player_stats(attacker, participants=participants)
            a_name = attacker.session.username or f"Player{attacker.client_id}"
            t_name = target.session.username or f"Player{target.client_id}"
            self._broadcast_kill_feed(f"{a_name} destroyed {t_name}")
        self._broadcast_player_stats(target, participants=participants)

        target_entity_id = target.session.entity_id or target.entity_id
        tick_del = self._get_network_tick(target)
        del_pkt = build_delete_object(tick_del, [target_entity_id], with_effects=True)
        # NOTE: the dying client receiving its OWN DELETE_OBJECT is what triggers the
        # death/deploy screen (Entity_delete(local) -> mode 3), but that SAME code
        # clears the local-player state incl g_player_team (-> SWITCH-TEAM not the
        # deploy tab). Skipping the self-DELETE (tested 2026-06-24) keeps the team
        # but leaves the player stuck IN-GAME with no death UI -- there is no
        # death-without-delete path in the OG client. "Keep team on death" needs a
        # client-side re-assert of g_player_team after the delete, not a server tweak.
        for client in self._snapshot_in_game_clients():
            if not self._combat_observer_packets_allowed_for_client(client, *participants):
                continue
            self._send_packet_to_client(client, del_pkt, prefer_tcp=True)

        target.session.in_game = False
        for other in self._snapshot_in_game_clients():
            if other is not target:
                other.known_entity_ids.discard(target_entity_id)
        target.known_entity_ids.clear()
        if hasattr(target, '_entity_create_times'):
            target._entity_create_times.clear()

        target.player_health = 1.0
        target.player_vel = (0.0, 0.0, 0.0)
        target.player_speed = 0.0
        target.angular_vel_yaw = 0.0
        target.world_collision_ref_pos = target.player_pos
        target.world_collision_bounds_dirty = False
        if target.vehicle_physics:
            target.vehicle_physics.reset()

        print(f"[COMBAT] c{target.client_id} killed ({reason}) [D:{target.deaths}]")
        self._enter_death_deploy_state(target)
