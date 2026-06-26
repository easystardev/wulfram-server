"""SpawnMixin -- spawn-point resolution, spawn execution (wf-style/minimal),
entry-transition, and death/deploy, extracted verbatim from WulframServer
(server.py decomposition, step 4). Method-only mixin; shares state via `self`.
"""
from __future__ import annotations

import math
import os
import time
from typing import Optional

from . import handlers
from .client import ClientContext
from .physics import _matrix3_from_euler_xyz
from .session import Phase
from .packets import (
    build_delete_object,
    build_add_to_roster,
    build_birth_notice,
    build_chat_message,
    build_game_clock,
    build_player,
    build_player_info,
    build_reincarnate,
    build_udp_tank_packet_wf,
    build_update_array_create_tank,
)


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

    def _send_udp_wf_spawn(self, ctx: ClientContext, addr: tuple, net_id: int, team_id: int,
                            unit_type: int = 0,
                            pos: tuple = (100.0, 100.0, 100.0),
                            rot: tuple = (0.0, 0.0, 0.0)):
        """Send a Wulf-Forge style UDP TANK (0x18) packet to spawn a unit."""
        if not self.udp_handler:
            return
        payload = build_udp_tank_packet_wf(
            net_id=net_id,
            unit_type=unit_type,
            team_id=team_id,
            pos=pos,
            rot=rot,
            include_vitals=self.tank_vitals,
            weapon_id=self.weapon_id,
            health=1.0,
            energy=1.0,
        )
        self._log_vitals(
            ctx,
            "TANK_UDP_SPAWN",
            include_vitals=self.tank_vitals,
            health=1.0,
            energy=1.0,
            weapon_id=self.weapon_id,
            note=f"team={team_id} net_id={net_id}",
        )
        self.udp_handler.send_to(payload, addr)

    def _should_send_spawn_entry_transition(self, ctx: ClientContext) -> bool:
        mode = getattr(self, "spawn_entry_transition", "off")
        if mode in ("1", "true", "on"):
            return True
        if mode in ("0", "false", "off"):
            return False
        return not handlers._is_loopback_client(ctx)

    def _send_spawn_entry_transition(self, ctx: ClientContext, team_id: int, net_id: int) -> None:
        """Reassert the decompile-backed team-entry sequence before OG auto-spawn."""
        if not self._should_send_spawn_entry_transition(ctx):
            return
        if not ctx.tcp_handler:
            return

        session = ctx.session
        session.player_id = session.player_id or net_id
        session.team_id = team_id

        rein = build_reincarnate(0x11, "")
        sent_rein = False
        if self.udp_handler and session.udp_addr:
            try:
                self.udp_handler.send_to(rein, session.udp_addr)
                sent_rein = True
            except Exception as ex:
                print(f"[SPAWN] Failed entry-transition UDP REINCARNATE(0x11): {ex}")
        if not sent_rein:
            ctx.tcp_handler.send(rein)

        ctx.tcp_handler.send(build_player(entity_id=session.player_id, spectator=False))
        ctx.tcp_handler.send(build_game_clock())

        sent_roster = False
        if not session.roster_sent:
            name = session.username or f"Player{ctx.client_id}"
            ctx.tcp_handler.send(build_add_to_roster(
                player_id=session.player_id,
                entity_id=session.player_id,
                name=name,
                team=team_id,
            ))
            session.roster_sent = True
            sent_roster = True

        sent_world_stats = False
        if not session.world_stats_sent:
            ctx.tcp_handler.send(self.build_world_stats_packet())
            session.world_stats_sent = True
            sent_world_stats = True

        print(
            "[SPAWN] Sent entry transition "
            f"(REINCARNATE via {'UDP' if sent_rein else 'TCP'}, PLAYER, GAME_CLOCK"
            f"{', ADD_TO_ROSTER' if sent_roster else ''}"
            f"{', WORLD_STATS' if sent_world_stats else ''})"
        )

    def _separate_from_live_tanks(self, ctx: ClientContext, pos: tuple, min_sep: float = 30.0) -> tuple:
        """Return a spawn position clear of every other live tank.

        Returns ``pos`` unchanged when nothing else is in-game or the point is
        already clear (so single-client and distinct-flag spawns are untouched).
        Otherwise spirals outward deterministically (no RNG) until clear.
        """
        if not pos or len(pos) < 3:
            return pos
        others = []
        for other in self._snapshot_in_game_clients():
            if other is ctx:
                continue
            op = getattr(other, "player_pos", None)
            if op and len(op) >= 2:
                others.append(op)
        if not others:
            return pos

        def _clear(px: float, py: float) -> bool:
            return all(
                (px - ox) ** 2 + (py - oy) ** 2 >= min_sep * min_sep
                for ox, oy, *_ in others
            )

        x, y, z = pos[0], pos[1], pos[2]
        if _clear(x, y):
            return pos
        for ring in range(1, 6):
            for k in range(8):
                ang = (k / 8.0) * 2.0 * math.pi
                px = x + math.cos(ang) * min_sep * ring
                py = y + math.sin(ang) * min_sep * ring
                if _clear(px, py):
                    print(
                        f"[SPAWN] Client {ctx.client_id}: separated spawn "
                        f"({x:.1f},{y:.1f}) -> ({px:.1f},{py:.1f}) to avoid stacking"
                    )
                    return (px, py, z)
        print(f"[SPAWN] Client {ctx.client_id}: could not find clear spawn near ({x:.1f},{y:.1f})")
        return pos

    def _spawn_wf_style(self, ctx: ClientContext, team_id: int, net_id: Optional[int] = None,
                         unit_type: int = 0,
                         pos: Optional[tuple] = None,
                         rot: tuple = (0.0, 0.0, 0.0),
                         announce: bool = True):
        """
        Spawn using Wulf-Forge's simple approach: just send TankPacket.

        Wulf-Forge's /s spawn only sends a single TankPacket (0x18) -
        no UPDATE_ARRAY, no separate PLAYER_INFO, no entity pre-creation.
        """
        net_id = net_id or (ctx.session.player_id or ctx.entity_id)
        ctx.session.player_id = net_id
        ctx.session.team_id = team_id
        # Team is now known/changed -> force every viewer to re-send this player's
        # roster entry with the correct team on the next presence broadcast
        # (entries are otherwise cached once per session via known_roster_ids).
        self._invalidate_roster_for_player(ctx)
        ctx.entity_type = unit_type

        # Reset position tracking to spawn location.
        spawn_pos = self._resolve_spawn_pos(team_id, explicit_pos=pos)
        if pos is not None:
            print(f"[SPAWN] Using explicit spawn pos={spawn_pos}")
        else:
            configured_default = self._get_configured_default_spawn_pos()
            if configured_default is not None:
                print(f"[SPAWN] Using default flat spawn pos={spawn_pos}")
            else:
                map_spawn = self._pick_spawn_point(team_id)
                if map_spawn:
                    print(
                        f"[SPAWN] Auto-selected map spawn oid={map_spawn['oid']} "
                        f"team={team_id} pos={spawn_pos}"
                    )

        # Offset spawn to avoid overlapping tanks in multi-client tests. Do not
        # move explicit map-flag spawns by default: the offset can put a clicked
        # repair pad spawn onto unrelated steep terrain, which turns later
        # targeted corrections into bogus authoritative snaps.
        offset_explicit_spawns = (
            os.environ.get("WULFRAM_MULTI_SPAWN_OFFSET_EXPLICIT", "0")
            .strip()
            .lower()
            in ("1", "true", "on", "yes")
        )
        apply_spawn_offset = bool(self.multi_spawn_offset) and (
            pos is None or offset_explicit_spawns
        )
        if apply_spawn_offset:
            # Use 0-based index among active clients (not client_id, which grows unboundedly).
            with self.clients_lock:
                active_ids = sorted(c.client_id for c in self.clients.values() if c and c.running)
            try:
                idx = active_ids.index(ctx.client_id)
            except ValueError:
                idx = 0
            spawn_pos = (spawn_pos[0] + idx * self.multi_spawn_offset, spawn_pos[1], spawn_pos[2])

        # Optional legacy/debug path: force the spawn anchor onto the decoded
        # heightmap. Default off because OG live memory matches raw map-state
        # repair-pad Z more closely than the currently decoded terrain height.
        if self.terrain and self.up_axis == "z" and self.align_spawn_pos_to_terrain:
            terrain_z = (
                self.terrain.get_height(spawn_pos[0], spawn_pos[1])
                + self.terrain_height_offset
            )
            spawn_pos = (spawn_pos[0], spawn_pos[1], terrain_z)

        # Anti-stack safety: never place a tank on top of another live tank. Two
        # tanks at the same point collide on arrival -> instant death, and the
        # spawning client never coexists with the other, which silently breaks
        # roster + remote-entity replication (the "can't see each other" symptom).
        # Only nudges on actual overlap, so single-client parity spawns and
        # distinct map-flag spawns are unaffected.
        spawn_pos = self._separate_from_live_tanks(ctx, spawn_pos)

        # Terrain-safe height (upward-only): never let a spawn sit below the map
        # surface, which causes continuous collision damage on arrival.
        if self.up_axis == "z":
            ground_z = self._terrain_ground_z_at(spawn_pos[0], spawn_pos[1])
            if ground_z is not None and spawn_pos[2] < ground_z:
                print(f"[SPAWN] Raising spawn Z {spawn_pos[2]:.1f} -> {ground_z:.1f} (terrain-safe)")
                spawn_pos = (spawn_pos[0], spawn_pos[1], ground_z)

        ctx.player_pos = spawn_pos
        ctx.player_pose["pos"] = spawn_pos
        if hasattr(ctx, "record_pose_reset"):
            ctx.record_pose_reset(
                "spawn_wf_style",
                pos=spawn_pos,
                vel=(0.0, 0.0, 0.0),
                details={
                    "team_id": team_id,
                    "net_id": net_id,
                    "unit_type": unit_type,
                    "explicit_pos": pos is not None,
                },
            )
        ctx.world_collision_ref_pos = spawn_pos
        ctx.world_collision_bounds_dirty = False
        ctx.last_state_sync_vel = None
        ctx.last_state_sync_rot = None
        ctx.last_correction_send = 0.0
        ctx.force_correction_once = False
        ctx.last_state_request_burst_queue = 0.0
        # A burst queued against the pre-death pose must not drain against the
        # respawn pose (same discontinuity argument as the accumulators below).
        ctx.correction_burst_remaining = 0
        # Gated-rare correction divergence accumulators (GOAL 2). Reset on
        # spawn so the respawn pose discontinuity doesn't read as divergence.
        ctx.divergence_accum_pos = 0.0
        ctx.divergence_accum_heading = 0.0
        ctx.authoritative_state_history.clear()

        ctx.player_health = 1.0
        ctx.player_angular_vel = 0.0
        ctx.player_speed = 0.0
        ctx.vehicle_physics.reset()
        # Reset tick-sync transition tracking so respawn gap doesn't cause huge correction
        ctx._turn_transition_client_tick = 0
        ctx._turn_transition_server_tick = 0

        # Spawn heading â€” can't set local client heading (physics overwrites it),
        # but this affects how OTHER clients see your tank at spawn.
        import math
        spawn_heading_env = os.environ.get("WULFRAM_SPAWN_HEADING")
        spawn_yaw = math.radians(float(spawn_heading_env)) if spawn_heading_env else 0.0
        ctx.player_yaw = -spawn_yaw
        ctx.player_heading = spawn_yaw
        ctx.angular_vel_yaw = 0.0
        ctx.spring_body_ang_vel = (0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, spawn_yaw)
        ctx.player_pose["roll"] = 0.0
        ctx.player_pose["pitch"] = 0.0
        ctx.player_pose["yaw"] = -spawn_yaw
        ctx.player_energy = self.player_energy_max
        ctx.vehicle_physics.heading = spawn_yaw
        self._reset_jump_jet_state(ctx)
        ctx.last_action_dump_time = time.monotonic()  # Reset timer for position tracking
        self._set_ground_level_override_for_pose(ctx, spawn_pos)

        print(f"[SPAWN] Wulf-Forge style: client={ctx.client_id} net_id={net_id} team={team_id} pos={spawn_pos}")

        if not ctx.tcp_handler:
            print("[SPAWN] ERROR: No TCP handler")
            return

        self._send_spawn_entry_transition(ctx, team_id=team_id, net_id=net_id)

        # Roster (already sent during login, but ensure it's there for name display)
        if not ctx.session.roster_sent:
            name = ctx.session.username or f"Player{ctx.client_id}"
            print(f"[SPAWN] Sending ADD_TO_ROSTER for {name}")
            ctx.tcp_handler.send(build_add_to_roster(
                player_id=net_id, entity_id=net_id, name=name, team=team_id
            ))
            ctx.session.roster_sent = True

        # NOTE: Wulf-forge capture shows /s spawn ONLY sends TankPacket + CommMessage
        # It does NOT send UPDATE_STATS or REINCARNATE before TankPacket!
        # Those are only sent in response to team switch requests.

        # Ensure TRANSLATION has been applied before sending TankPacket with vitals.
        if not ctx.session.translation_ack_received:
            wait_until = time.monotonic() + 2.0
            while not ctx.session.translation_ack_received and time.monotonic() < wait_until:
                time.sleep(0.05)
            if not ctx.session.translation_ack_received:
                print("[SPAWN] WARNING: TRANSLATION_ACK not received before TankPacket")

        # vitals=1 on the FIRST TankPacket â€” this writes health=1.0 into
        # the ESI buffer during PLAYER_INFO's read_local_player_state.  The
        # subsequent sync_local_player (the ONLY call that passes the tick
        # guard, because Mode3_init_descriptor â†’ operator_new returns a heap
        # pointer that gets stored in g_last_input_apply_tick, permanently
        # blocking all future heartbeat sync_local_player calls) then writes
        # ESI[0]=1.0 to the HUD health meter.
        #
        # NOTE: previous test showed vitals=1 caused "entry map persistence"
        # but that was WITHOUT UPDATE_ARRAY pre-creation.  With pre-creation,
        # PLAYER_INFO takes the "entity found" code path (LocalPlayer_initialize
        # in the else branch) instead of Entity_create_from_network.
        include_spawn_vitals = True
        # Force the initial spawn packets onto the short-form-safe remote path.
        # If this client had already reached promoted remote sync before a
        # respawn, restore that mode after the fresh entity exists again.
        restore_promoted_remote_sync_after_spawn = (
            self.update_local_state_mode == "wf"
            and not handlers._is_loopback_client(ctx)
            and bool(getattr(ctx, "remote_full_local_state_ready", False))
        )
        ctx.last_client_tick = 0
        ctx.remote_full_local_state_ready = False
        ctx._spawn_safe_heartbeat_suppressed_logged = False
        # Set last_spawn_time BEFORE sending TankPacket so the 2-second
        # jump suppression window covers the entire retransmit period.
        # Without this, a jump ACTION_UPDATE arriving during retransmit
        # sleep bypasses suppression and sends a velocity UPDATE_ARRAY
        # that sets prev_health positive before a retransmit re-creates
        # the entity with health=0 â†’ permanent DeathScreen.
        ctx.session.last_spawn_time = time.monotonic()
        if not self.spawn_send_udp_tank:
            print("[SPAWN] Skipping UDP TankPacket (WULFRAM_SPAWN_UDP_TANK=0)")
        else:
            print(f"[SPAWN] Sending UDP TankPacket (vitals={int(include_spawn_vitals)})")
        send_pos = self._to_client_pos(spawn_pos)
        send_rot = self._local_player_sync_rotation(ctx)
        # Local-state updates may use Tank(0), but spawn TankPacket vitals use a
        # parse-safe weapon id because 0x18 writes only weapon+health+energy bits.
        ls_weapon = self._get_local_state_weapon_type(ctx)
        spawn_tank_weapon = self._get_spawn_tank_weapon_type(ctx)
        # Optional: create the entity via UPDATE_ARRAY before PLAYER_INFO.
        if self.spawn_send_update_array:
            health_val = self._get_health_value(ctx)
            fuel_val = self._get_energy_value(ctx)
            ua_packet = build_update_array_create_tank(
                tick=self._get_network_tick(ctx),
                entity_id=net_id,
                entity_type=unit_type,
                team=team_id,
                pos=send_pos,
                behavior_type=0,
                include_health=False,  # local_player_state ammo/turret bits crash client
                include_entity_vitals=self.update_entity_vitals,
                health=health_val,
                fuel=fuel_val,
                is_manned=True,
                weapon_id=ls_weapon,
            )
            if not self._send_spawn_create_update_array(ctx, ua_packet):
                print("[SPAWN] WARNING: No transport available for pre-creation UPDATE_ARRAY")
            # Delay so client processes entity creation before PLAYER_INFO.
            # The TankPacket's OIDTable_lookup must find the entity created
            # by UPDATE_ARRAY; if the TankPacket arrives first, it takes the
            # Entity_create_from_network path which does NOT call
            # LocalPlayer_initialize â†’ entry map stays.  200ms gives ~6
            # frames at 30fps for the client to process the UPDATE_ARRAY.
            time.sleep(0.20)
        spawn_tick = self._get_network_tick(ctx)
        tank_packet = build_udp_tank_packet_wf(
            net_id=net_id,
            unit_type=unit_type,
            team_id=team_id,
            pos=send_pos,
            rot=send_rot,
            tick=spawn_tick,
            include_vitals=include_spawn_vitals,
            weapon_id=spawn_tank_weapon,
            health=1.0,
            energy=1.0,
        )
        self._log_vitals(
            ctx,
            "TANK_UDP_SPAWN",
            include_vitals=include_spawn_vitals,
            health=1.0,
            energy=1.0,
            weapon_id=spawn_tank_weapon,
            note=f"team={team_id} net_id={net_id}",
        )
        # HEX DUMP for comparison with wulf-forge
        print(f"[TANK-HEX] len={len(tank_packet)} hex={tank_packet.hex().upper()}")

        # Send TankPacket over UDP (matching wulf-forge behavior)
        sent_tank_udp = False
        if self.spawn_send_udp_tank:
            if self.udp_handler and ctx.session.udp_addr:
                self.udp_handler.send_to(tank_packet, ctx.session.udp_addr)
                sent_tank_udp = True
                print(f"[SPAWN] Sent UDP TankPacket to {ctx.session.udp_addr}")
                # Optional TCP backup.  After combat kill + entity DELETE,
                # the client may not process UDP while on team-select overlay.
                # With pre-creation UPDATE_ARRAY, entity exists in OIDTable so
                # TCP PLAYER_INFO takes "entity found" path â†’ LocalPlayer_initialize.
                if os.environ.get("WULFRAM_SPAWN_TCP_TANK_BACKUP", "0").strip().lower() not in ("0", "false", "off", "no"):
                    ctx.tcp_handler.send(tank_packet)
                    print(f"[SPAWN] Also sent TCP TankPacket (backup for respawn)")
                # Resend TankPacket for reliability (UDP can drop packets)
                # Each retransmit triggers Entity_create_from_network (new
                # entity, 0xD0=0).  This is safe as long as no velocity
                # UPDATE_ARRAY sets prev_health positive during the
                # retransmit window.  The jump-suppression timer (set
                # before TankPacket send) prevents that race.
                spawn_retransmits = int(os.environ.get("WULFRAM_SPAWN_RETRANSMIT", "0"))
                if spawn_retransmits > 0:
                    # Retransmits DISABLED by default (0).  Each retransmit
                    # calls LocalPlayer_initialize again; if it arrives during
                    # the spawn transition it re-shows the entry map overlay
                    # (race condition: run 20260209_000347 vs _000127).
                    # Wulf-forge sends only one TankPacket, no retransmits.
                    # The pre-creation UPDATE_ARRAY handles entity existence
                    # and the first TankPacket is reliably received on
                    # localhost.  Keep env var for optional override.
                    for i in range(spawn_retransmits):
                        time.sleep(0.05)
                        retransmit_tick = self._get_network_tick(ctx)
                        retransmit_packet = build_udp_tank_packet_wf(
                            net_id=net_id,
                            unit_type=unit_type,
                            team_id=team_id,
                            pos=send_pos,
                            rot=send_rot,
                            tick=retransmit_tick,
                            include_vitals=True,
                            weapon_id=spawn_tank_weapon,
                            health=1.0,
                            energy=1.0,
                        )
                        self.udp_handler.send_to(retransmit_packet, ctx.session.udp_addr)
                    print(f"[SPAWN] Retransmitted TankPacket {spawn_retransmits}x (vitals=1)")
                # Immediate UPDATE_ARRAY heartbeat with health=1.0.
                # The pre-creation UPDATE_ARRAY set up the OID table entry,
                # and the TankPacket's PLAYER_INFO path found it via
                # OIDTable_lookup â†’ called LocalPlayer_initialize â†’
                # g_local_player_entity is now set.  This heartbeat populates
                # ESI with health=1.0 so sync_local_player writes it to the
                # HUD health meter, clearing the red overlay.
                if os.environ.get("WULFRAM_SPAWN_BOOTSTRAP_HEARTBEAT", "1").strip().lower() in ("0", "false", "off", "no"):
                    print("[SPAWN] Suppressing immediate spawn bootstrap heartbeat UPDATE_ARRAY")
                elif self._suppress_remote_spawn_bootstrap_heartbeat(ctx):
                    if self.update_local_state_mode == "wf" and not handlers._is_loopback_client(ctx):
                        time.sleep(0.05)
                        hb_tick = self._get_network_tick(ctx)
                        hb_packet = self._build_remote_spawn_bootstrap_heartbeat(
                            ctx,
                            tick=hb_tick,
                            entity_id=net_id,
                            health=1.0,
                            fuel=1.0,
                        )
                        self.udp_handler.send_to(hb_packet, ctx.session.udp_addr)
                        print("[SPAWN] Sent remote bootstrap heartbeat UPDATE_ARRAY (full transform safe shape)")
                    else:
                        print("[SPAWN] Suppressing immediate remote bootstrap heartbeat UPDATE_ARRAY")
                else:
                    time.sleep(0.05)
                    hb_tick = self._get_network_tick(ctx)
                    hb_packet = self._build_local_state_heartbeat(
                        ctx,
                        tick=hb_tick,
                        entity_id=net_id,
                        include_health=True,
                        health=1.0,
                        fuel=1.0,
                    )
                    self.udp_handler.send_to(hb_packet, ctx.session.udp_addr)
                    print(f"[SPAWN] Sent immediate heartbeat UPDATE_ARRAY (health=1.0)")
            else:
                # Fallback to TCP if no UDP address
                ctx.tcp_handler.send(tank_packet)
                print(f"[SPAWN] Sent TCP TankPacket (no UDP addr)")
        else:
            # Explicit TCP fallback for spawn-isolation runs.
            ctx.tcp_handler.send(tank_packet)
            print("[SPAWN] Sent TCP TankPacket (WULFRAM_SPAWN_UDP_TANK=0)")

        # Wulf-forge capture order is TankPacket first, then the spawn message.
        if announce and self.spawn_send_comm_message:
            comm_pkt = build_chat_message("Spawning in...", source_id=net_id)
            if self.udp_handler and ctx.session.udp_addr and sent_tank_udp:
                try:
                    self.udp_handler.send_to(comm_pkt, ctx.session.udp_addr)
                    print(f"[SPAWN] Sent UDP CommMessage for spawn to {ctx.session.udp_addr}")
                except Exception as ex:
                    print(f"[SPAWN] Failed to send UDP CommMessage for spawn: {ex}")
                    if ctx.tcp_handler:
                        ctx.tcp_handler.send(comm_pkt)
                        print(f"[SPAWN] Sent TCP CommMessage for spawn (fallback)")
            elif ctx.tcp_handler:
                ctx.tcp_handler.send(comm_pkt)
                print(f"[SPAWN] Sent TCP CommMessage for spawn (no UDP addr)")

        # NOTE: We rely on TankPacket (UDP) to create the entity.
        # UPDATE_ARRAY_CREATE_TANK causes crash when sent AFTER TankPacket.
        # For now, skip UPDATE_ARRAY and try PLAYER_INFO alone.

        # PLAYER_INFO tells the client "this is your controllable entity"
        # Without this, client won't send VIEWPOINT_INFO (0x35)
        include_player_info_state, player_info_state = self._get_player_info_local_state_kwargs(ctx)
        weapon_type = player_info_state["weapon_id"]
        ammo_bits = player_info_state["ammo_count_bits"]
        ammo_mask = player_info_state["ammo_count"]
        pt_bits = player_info_state["primary_turret_bits"]
        pt_angle = player_info_state["primary_turret_angle"]
        st_bits = player_info_state["secondary_turret_bits"]
        st_angle = player_info_state["secondary_turret_angle"]
        if self.player_info_local_state_mode != "off":
            print(
                "[LOCAL-STATE] PLAYER_INFO "
                f"weapon={weapon_type} ammo_bits={ammo_bits} "
                f"pt_bits={pt_bits} st_bits={st_bits}"
            )
        player_info_props = 0
        if self.player_info_properties_mode in ("team", "team_id"):
            player_info_props = team_id
        elif self.player_info_properties_mode:
            try:
                player_info_props = int(self.player_info_properties_mode, 0)
            except ValueError:
                player_info_props = 0
        if self.player_info_properties_mode:
            print(f"[LOCAL-STATE] PLAYER_INFO properties={player_info_props}")
        if include_player_info_state:
            self._log_vitals(
                ctx,
                "PLAYER_INFO",
                include_vitals=True,
                health=1.0,
                energy=1.0,
                weapon_id=weapon_type,
                note=f"ammo_bits={ammo_bits} pt_bits={pt_bits} st_bits={st_bits}",
            )
        elif self.debug_vitals and self.player_info_local_state_mode != "off":
            self._log_vitals(
                ctx,
                "PLAYER_INFO",
                include_vitals=False,
                health=1.0,
                energy=1.0,
                weapon_id=weapon_type,
                note="local_state=0",
            )
        if self.spawn_send_player_packet:
            ctx.tcp_handler.send(build_player(net_id, spectator=self.spawn_player_spectator))
            print(f"[SPAWN] Sent PLAYER spectator={int(self.spawn_player_spectator)}")
        send_spawn_player_info = self._should_send_spawn_player_info(ctx)
        if not send_spawn_player_info:
            reason = (
                f"WULFRAM_SPAWN_PLAYER_INFO={int(self.spawn_send_player_info)}"
                if self.spawn_send_player_info_explicit
                else "loopback default"
            )
            print(f"[SPAWN] Skipping TCP PLAYER_INFO ({reason})")
        else:
            player_info_pkt = build_player_info(
                entity_oid=net_id,
                vehicle_type=unit_type,
                pos=send_pos,
                rot=send_rot,
                include_local_state=include_player_info_state,
                weapon_id=player_info_state["weapon_id"],
                health=player_info_state["health"],
                fuel=player_info_state["fuel"],
                properties=player_info_props,
                ammo_count_bits=player_info_state["ammo_count_bits"],
                ammo_count=player_info_state["ammo_count"],
                primary_turret_bits=player_info_state["primary_turret_bits"],
                primary_turret_angle=player_info_state["primary_turret_angle"],
                secondary_turret_bits=player_info_state["secondary_turret_bits"],
                secondary_turret_angle=player_info_state["secondary_turret_angle"],
                turret_max=player_info_state["turret_max"],
                turret_range=player_info_state["turret_range"],
            )
            ctx.tcp_handler.send(player_info_pkt)
            print(f"[SPAWN] Sent TCP PLAYER_INFO: entity_oid={net_id} vehicle={unit_type}")

        # Track game start time from first spawn
        if self._game_start_time == 0.0:
            self._game_start_time = time.monotonic()

        # Complete spawn sequence (matching spawn_full)
        if self.spawn_send_game_clock:
            ctx.tcp_handler.send(build_game_clock())
        if self.spawn_send_reincarnate:
            rein = build_reincarnate(0x11, "")
            sent_rein = False
            if self.udp_handler and ctx.session.udp_addr:
                try:
                    self.udp_handler.send_to(rein, ctx.session.udp_addr)
                    sent_rein = True
                    print(f"[SPAWN] Sent UDP REINCARNATE(0x11) to {ctx.session.udp_addr}")
                except Exception as ex:
                    print(f"[SPAWN] Failed to send UDP REINCARNATE(0x11): {ex}")
            if not sent_rein:
                ctx.tcp_handler.send(rein)
                print("[SPAWN] Sent TCP REINCARNATE(0x11) (fallback)")
        if self.spawn_send_birth_notice:
            ctx.tcp_handler.send(build_birth_notice(net_id))
        if self.spawn_send_game_clock or self.spawn_send_reincarnate or self.spawn_send_birth_notice:
            print(
                "[SPAWN] Sent "
                f"{'GAME_CLOCK ' if self.spawn_send_game_clock else ''}"
                f"{'REINCARNATE(0x11) ' if self.spawn_send_reincarnate else ''}"
                f"{'BIRTH_NOTICE' if self.spawn_send_birth_notice else ''}"
            )

        # Enter game mode and start tick loop for UPDATE_ARRAY
        was_in_game = ctx.session.in_game or ctx.session.phase == Phase.IN_GAME
        ctx.session.entity_id = net_id
        ctx.session.last_spawn_time = time.monotonic()
        ctx.session.in_game = True
        if not was_in_game:
            ctx.session.transition_to(Phase.IN_GAME)
        else:
            print(f"[SPAWN] Client {ctx.client_id}: refreshed active spawn while already IN_GAME")
        if self.spawn_send_player_active:
            ctx.tcp_handler.send(build_player(net_id, spectator=False))
            print("[SPAWN] Sent PLAYER spectator=0 (post-spawn reassert)")

        # Sync roster/entity visibility with other in-game clients.
        self._sync_clients_on_spawn(ctx)
        self._ensure_uplink_mvp_state(ctx)

        self._resume_remote_full_local_state_after_spawn(
            ctx,
            entity_id=net_id,
            health=1.0,
            fuel=1.0,
            previously_promoted=restore_promoted_remote_sync_after_spawn,
        )

        if self._ensure_tick_loop(ctx):
            print(f"[SPAWN] Started tick loop (local_state_updates={self.update_local_state_mode})")

    def _spawn_wf_minimal(self, ctx: ClientContext, team_id: int, net_id: int, addr: tuple):
        """
        Absolutely minimal spawn - just TankPacket (wulf-forge style).

        Uses include_vitals per WULFRAM_TANK_VITALS (defaults off while investigating).
        TRANSLATION quantizers define 5/10/10 bits which matches TankPacket format.
        """
        print(f"[SPAWN] Minimal WF: client={ctx.client_id} net_id={net_id} team={team_id}")
        ctx.entity_type = 0

        spawn_pos = self._resolve_spawn_pos(team_id)
        configured_default = self._get_configured_default_spawn_pos()
        map_spawn = None if configured_default is not None else self._pick_spawn_point(team_id)
        if configured_default is not None:
            print(f"[SPAWN] Minimal mode default flat spawn pos={spawn_pos}")
        elif map_spawn:
            print(f"[SPAWN] Minimal mode map spawn oid={map_spawn['oid']} team={team_id} pos={spawn_pos}")
        self._set_ground_level_override_for_pose(ctx, spawn_pos)

        # Use 0-based index among active clients (not client_id, which grows unboundedly).
        if self.multi_spawn_offset:
            with self.clients_lock:
                active_ids = sorted(c.client_id for c in self.clients.values() if c and c.running)
            try:
                idx = active_ids.index(ctx.client_id)
            except ValueError:
                idx = 0
            spawn_pos = (spawn_pos[0] + idx * self.multi_spawn_offset, spawn_pos[1], spawn_pos[2])

        send_pos = self._to_client_pos(spawn_pos)
        self._set_ground_level_override_for_pose(ctx, spawn_pos)
        ctx.player_energy = self.player_energy_max
        self._reset_jump_jet_state(ctx)
        tank_packet = build_udp_tank_packet_wf(
            net_id=net_id,
            unit_type=0,
            team_id=team_id,
            pos=send_pos,
            rot=self._local_player_sync_rotation(ctx),
            include_vitals=self.tank_vitals,
            weapon_id=self.weapon_id,
            health=1.0,
            energy=1.0,
        )
        self._log_vitals(
            ctx,
            "TANK_UDP_MINIMAL",
            include_vitals=self.tank_vitals,
            health=1.0,
            energy=1.0,
            weapon_id=self.weapon_id,
            note=f"team={team_id} net_id={net_id}",
        )

        if self.udp_handler:
            self.udp_handler.send_to(tank_packet, addr)
            print(f"[SPAWN] Sent minimal TankPacket (vitals={int(self.tank_vitals)}) to {addr}")
