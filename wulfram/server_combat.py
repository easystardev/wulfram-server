"""CombatMixin -- projectile movement, hit detection, damage application,
extracted verbatim from WulframServer (server.py decomposition, step 7).
Method-only mixin; shares state via `self`.
"""
from __future__ import annotations

import math
import threading
import time
import traceback
from typing import Optional

from . import handlers
from .client import ClientContext
from .weapons import EntityType, build_projectile_update_packet
from .packets import (
    build_chat_message,
    build_delete_object,
    FX_IMPACT_BUILDING,
    FX_IMPACT_TERRAIN,
    FX_IMPACT_VEHICLE,
)


class CombatMixin:
    def _spawn_moving_projectile(self, ctx: ClientContext, proj, addr: tuple):
        """Spawn a projectile and start sending movement updates."""
        import time
        from .weapons import build_projectile_update_packet

        # Convert to client/world coordinates once so spawn + updates stay aligned.
        server_pos = proj.pos
        if self.debug_projectiles:
            self._log_projectile_aim(ctx, proj, server_pos)
        proj.pos = self._to_client_pos(server_pos)

        # Send initial spawn packet
        self._send_projectile_spawn(ctx, proj, addr)

        # Broadcast TRANSIENT_ARRAY (weapon fire FX) to all other clients
        # DISABLED: 0x0D format crashes OG client (D_ERR at PROTOCOL.CPP:474)
        # self._broadcast_weapon_fire_fx(ctx, proj)

        # Start background thread for movement updates
        def update_loop():
            # === Projectile update mode ===
            # Mode 0: No updates (let client simulate from spawn velocity)
            # Mode 1: Low rate updates (5 Hz)
            # Mode 2: Medium rate updates (15 Hz)
            # Mode 3: High rate updates (30 Hz)
            update_mode = self.projectile_update_mode
            delete_reason = "expired"
            delete_with_effects = False

            if update_mode == 0:
                # No updates - just wait for lifetime then clean up
                time.sleep(proj.lifetime)
                print(f"[PROJ] id={proj.entity_id} expired (no-update mode)")
                with ctx.projectile_lock:
                    if proj in ctx.active_projectiles:
                        ctx.active_projectiles.remove(proj)
                tick = self._get_network_tick(ctx)
                self._broadcast_projectile_delete(
                    proj,
                    tick,
                    with_effects=False,
                    reason="expired",
                )
                return

            update_rate = {1: 5.0, 2: 15.0, 3: 30.0}.get(update_mode, 15.0)
            dt = 1.0 / update_rate
            duration = proj.lifetime

            updates = int(duration * update_rate)
            for i in range(updates):
                time.sleep(dt)

                # Check if projectile still active
                with ctx.projectile_lock:
                    if proj not in ctx.active_projectiles:
                        break

                if proj.entity_type == EntityType.CALTROP:
                    self._steer_caltrop_projectile(proj, ctx, dt)

                # Update projectile position for hit detection
                # (build_projectile_update_packet also updates proj.pos,
                #  but we need the position current before hit check)
                prev_client_pos = proj.pos
                proj.pos = (
                    proj.pos[0] + proj.vel[0] * dt,
                    proj.pos[1] + proj.vel[1] * dt,
                    proj.pos[2] + proj.vel[2] * dt,
                )

                world_hit = self._check_projectile_world_hit(prev_client_pos, proj.pos, proj)
                if world_hit:
                    hit_kind, hit_pos, hit_detail = world_hit
                    delete_reason = hit_kind
                    delete_with_effects = False
                    with ctx.projectile_lock:
                        if proj in ctx.active_projectiles:
                            ctx.active_projectiles.remove(proj)

                    # Broadcast impact FX at hit location
                    if hit_kind == "terrain":
                        fx_type = FX_IMPACT_TERRAIN
                        print(
                            f"[PROJ-WORLD] id={proj.entity_id} hit terrain "
                            f"at=({hit_pos[0]:.1f},{hit_pos[1]:.1f},{hit_pos[2]:.1f})"
                        )
                    else:
                        fx_type = FX_IMPACT_BUILDING
                        print(
                            f"[PROJ-WORLD] id={proj.entity_id} hit {hit_kind} "
                            f"target={hit_detail} at=({hit_pos[0]:.1f},{hit_pos[1]:.1f},{hit_pos[2]:.1f})"
                        )
                        # Apply damage to building
                        self._apply_building_damage(hit_detail, proj, ctx, hit_pos)

                    # Send impact FX via TRANSIENT_ARRAY only to viewers that
                    # can safely accept the current 0x0D path.
                    self._broadcast_transient_fx([{
                        'type': fx_type,
                        'pos': self._to_client_pos(hit_pos),
                    }])
                    break

                # Check collision with enemy players
                hit_target = self._check_projectile_hit(proj, ctx)
                if hit_target:
                    try:
                        self._apply_damage(hit_target, proj, ctx)
                    except Exception as dmg_err:
                        print(f"[COMBAT-ERROR] _apply_damage failed: {dmg_err}")
                        import traceback
                        traceback.print_exc()
                    delete_reason = None
                    with ctx.projectile_lock:
                        if proj in ctx.active_projectiles:
                            ctx.active_projectiles.remove(proj)
                    break  # Stop update loop

                # Send position update (dt=0 since we already advanced pos above)
                # Build per-client packets so each viewer gets their OWN health
                # in local_state (not the shooter's health).
                sent_update_count = 0
                if self.udp_handler:
                    for target in self._snapshot_in_game_clients():
                        if not target.session.udp_addr or not target.session.translation_ack_received:
                            continue
                        if not self._projectile_packets_allowed_for_client(target):
                            continue
                        tick = self._get_network_tick(target)
                        include_local_state, local_state_kwargs = self._get_projectile_local_state_for_viewer(target)
                        pkt = build_projectile_update_packet(
                            proj,
                            tick,
                            0.0,  # Position already advanced above
                            include_local_state=include_local_state,
                            **local_state_kwargs,
                            )
                        self.udp_handler.send_to(pkt, target.session.udp_addr)
                        sent_update_count += 1
                        if self.pktlog.enabled:
                            self.pktlog.log(
                                client_id=target.client_id,
                                label="PROJ_UPDATE",
                                tick=tick,
                                payload=pkt,
                                transport="UDP",
                                entity_count=1,
                                entity_ids=(proj.entity_id,),
                                mask_bits=(0b0010,),  # pos only
                                has_local_state=include_local_state,
                                health=self._get_health_value(target) if include_local_state else -1.0,
                            )
                if sent_update_count:
                    ctx.projectile_update_packet_count = (
                        int(getattr(ctx, "projectile_update_packet_count", 0) or 0)
                        + sent_update_count
                    )
                    ctx.last_projectile_update_time = time.monotonic()
                    ctx.last_projectile_update_id = int(getattr(proj, "entity_id", 0) or 0)
                    ctx.last_projectile_update_targets = sent_update_count

                if i % 15 == 0:  # Log every 0.5 sec at 30Hz
                    print(f"[PROJ] id={proj.entity_id} pos=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f}) vel=({proj.vel[0]:.0f},{proj.vel[1]:.0f},{proj.vel[2]:.0f}) tick={tick}")

            # Remove from active list when done
            with ctx.projectile_lock:
                if proj in ctx.active_projectiles:
                    ctx.active_projectiles.remove(proj)
            # Send DELETE_OBJECT so the client removes the shell entity
            if delete_reason is not None:
                tick = self._get_network_tick(ctx)
                self._broadcast_projectile_delete(
                    proj,
                    tick,
                    with_effects=delete_with_effects,
                    reason=delete_reason,
                )
                print(f"[PROJ] id={proj.entity_id} {delete_reason} - DELETE sent")

        # Add to active list and start thread
        with ctx.projectile_lock:
            ctx.active_projectiles.append(proj)

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

    def _check_projectile_hit(self, proj, owner_ctx: ClientContext) -> Optional[ClientContext]:
        """Check if a projectile hit any enemy player (sphere collision).

        Compares projectile position against all in-game players except the owner.
        Both are in client/world coordinates (proj.pos is already converted).
        Returns the hit ClientContext or None.
        """
        hit_radius = 15.0  # Tank is roughly this size
        hit_radius_sq = hit_radius * hit_radius
        for target in self._snapshot_in_game_clients():
            if target is owner_ctx:
                continue
            # Compare in client space (proj.pos already converted)
            target_pos = self._to_client_pos(target.player_pos)
            dx = proj.pos[0] - target_pos[0]
            dy = proj.pos[1] - target_pos[1]
            dz = proj.pos[2] - target_pos[2]
            dist_sq = dx * dx + dy * dy + dz * dz
            dist = math.sqrt(dist_sq)
            if dist < 100:  # Log when close
                print(
                    f"[PROJ-DIST] id={proj.entity_id} -> c{target.client_id} "
                    f"dist={dist:.1f} (hit<{hit_radius:.0f}) "
                    f"proj=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f}) "
                    f"tgt=({target_pos[0]:.1f},{target_pos[1]:.1f},{target_pos[2]:.1f})"
                )
            if dist_sq <= hit_radius_sq:
                print(f"[PROJ-HIT] id={proj.entity_id} HIT c{target.client_id} dist={dist:.1f}")
                return target
        return None

    def _steer_caltrop_projectile(
        self,
        proj,
        owner_ctx: ClientContext,
        dt: float,
        *,
        range_limit: float = 200.0,
        speed: float = 80.0,
    ) -> Optional[ClientContext]:
        """Nudge Caltrops toward the nearest visible tank inside the OG help range."""
        try:
            origin = tuple(float(v) for v in proj.pos[:3])
        except (TypeError, ValueError):
            return None
        best: tuple[float, ClientContext, tuple[float, float, float]] | None = None
        range_sq = range_limit * range_limit
        for target in self._snapshot_in_game_clients():
            if target is owner_ctx:
                continue
            try:
                target_pos = self._to_client_pos(target.player_pos)
                rel = (
                    float(target_pos[0]) - origin[0],
                    float(target_pos[1]) - origin[1],
                    float(target_pos[2]) - origin[2],
                )
            except (TypeError, ValueError, IndexError):
                continue
            dist_sq = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]
            if dist_sq <= 1e-6 or dist_sq > range_sq:
                continue
            if best is None or dist_sq < best[0]:
                best = (dist_sq, target, rel)
        if best is None:
            return None

        dist = math.sqrt(best[0])
        direction = (best[2][0] / dist, best[2][1] / dist, best[2][2] / dist)
        desired = (direction[0] * speed, direction[1] * speed, direction[2] * speed)
        # Caltrops should visibly home, but avoid frame-to-frame right-angle
        # snaps in UPDATE_ARRAY rotation by blending toward the desired vector.
        blend = min(1.0, max(0.0, dt * 6.0))
        proj.vel = (
            proj.vel[0] + (desired[0] - proj.vel[0]) * blend,
            proj.vel[1] + (desired[1] - proj.vel[1]) * blend,
            proj.vel[2] + (desired[2] - proj.vel[2]) * blend,
        )
        return best[1]

    def _apply_damage(self, target: ClientContext, proj, attacker: ClientContext) -> None:
        """Apply damage from a projectile hit and broadcast effects.

        1. Reduce target health
        2. Send health update to target
        3. Send DELETE_OBJECT for projectile (with explosion FX)
        4. If dead, send DELETE_OBJECT for target entity
        """
        # Guard: ignore hits on already-dead targets (overkill from queued projectiles)
        if target.player_health <= 0.0:
            print(f"[COMBAT] Ignoring hit on already-dead c{target.client_id}")
            # Still delete the projectile
            tick = self._get_network_tick(attacker)
            delete_proj_pkt = build_delete_object(tick, [proj.entity_id], with_effects=True)
            for client in self._snapshot_in_game_clients():
                if not self._projectile_packets_allowed_for_client(client):
                    continue
                self._send_packet_to_client(client, delete_proj_pkt, prefer_tcp=True)
            return

        # Per-weapon damage (fraction of 100 health). Decompile: entity health
        # table at VA 0x4E3B00 has Tank=100hp. These are approximate pending
        # OG server-side weapon damage values.
        _PROJECTILE_DAMAGE = {
            EntityType.PULSE_SHELL: 0.20,    # 20% — 5 hits to kill
            EntityType.PIERCER: 0.30,        # 30% — fast, high damage
            EntityType.THUMPER: 0.35,        # 35% — slow, heavy damage
            EntityType.HUNTER: 0.25,         # 25% — homing missile
            EntityType.HEAVY_MISSILE: 0.50,  # 50% — heavy ordnance
            EntityType.MINE: 0.40,           # 40% — proximity mine
            EntityType.CALTROP: 0.10,        # 10% — light homing bomblet
            EntityType.SHORT_MISSILE: 0.15,  # 15% — short range missile
            EntityType.FLAK_SHELL: 0.10,     # 10% — flak
        }
        damage = _PROJECTILE_DAMAGE.get(proj.entity_type, 0.20)
        old_health = target.player_health
        target.player_health = round(max(0.0, old_health - damage), 6)
        new_health = target.player_health
        target.last_damage_time = time.monotonic()
        target.last_damage_source = f"projectile:{getattr(proj.entity_type, 'name', proj.entity_type)}"
        target.last_damage_amount = damage
        target.last_damage_old_health = old_health
        target.last_damage_new_health = new_health

        attacker_name = attacker.session.username or f"Player{attacker.client_id}"
        target_name = target.session.username or f"Player{target.client_id}"
        print(
            f"[COMBAT] {attacker_name} (c{attacker.client_id}) hit {target_name} (c{target.client_id}) "
            f"for {damage*100:.0f}% damage (health: {old_health*100:.0f}% -> {new_health*100:.0f}%)"
        )

        if target.player_health > 0.0 and self.udp_handler and target.session.udp_addr:
            tick = self._get_network_tick(target)
            health_pkt = self._build_local_state_heartbeat(
                target,
                tick=tick,
                entity_id=target.session.entity_id,
                include_health=True,
                health=self._get_health_value(target),
                fuel=self._get_energy_value(target),
            )
            self.udp_handler.send_to(health_pkt, target.session.udp_addr)
            if self.pktlog.enabled:
                self.pktlog.log(
                    client_id=target.client_id,
                    label="DAMAGE_HEARTBEAT",
                    tick=tick,
                    payload=health_pkt,
                    transport="UDP",
                    entity_count=1,
                    entity_ids=(0xFFFFFFFE,),
                    mask_bits=(0,),
                    has_local_state=True,
                    health=self._get_health_value(target),
                    extra=f"dmg={damage}",
                )

        # Broadcast impact FX via TRANSIENT_ARRAY (decompile-backed quantized bitstream)
        impact_events = [{
            'type': FX_IMPACT_VEHICLE,
            'pos': proj.pos,
            'entity_id': target.entity_id,
        }]
        self._broadcast_transient_fx(impact_events)

        # DELETE projectile with explosion effects
        tick = self._get_network_tick(attacker)
        delete_proj_pkt = build_delete_object(tick, [proj.entity_id], with_effects=True)
        for client in self._snapshot_in_game_clients():
            if not self._projectile_packets_allowed_for_client(client):
                continue
            self._send_packet_to_client(client, delete_proj_pkt, prefer_tcp=True)

        # Chat notification
        hit_msg = f"HIT! {target_name} ({new_health*100:.0f}% health)"
        chat_pkt = build_chat_message(hit_msg, source_id=attacker.session.player_id or attacker.entity_id)
        for client in self._snapshot_in_game_clients():
            if client.tcp_handler and self._debug_comm_allowed_for_client(client):
                # Crash-safe send: a client killed mid-combat leaves a dead socket;
                # a raw .send() here re-raises (WinError 10054) and kills this
                # per-projectile thread (leaking the projectile + spamming a
                # traceback). Route through the helper, which logs and skips.
                # (A3 soak, 2026-06-01.)
                self._send_packet_to_client(client, chat_pkt, prefer_tcp=True,
                                            allow_udp_fallback=False)

        # Send health refresh to ALL surviving clients (attacker etc.)
        # Projectile UPDATE_ARRAY packets include per-viewer health, but once
        # projectiles are gone the viewer gets no more health data. This
        # heartbeat ensures the attacker's HUD doesn't revert to zero.
        for client in self._snapshot_in_game_clients():
            if client is target:
                continue  # Already sent above
            if (
                client is not attacker
                and not handlers._is_loopback_client(client)
            ):
                continue
            if self.udp_handler and client.session.udp_addr:
                c_tick = self._get_network_tick(client)
                c_pkt = self._build_local_state_heartbeat(
                    client,
                    tick=c_tick,
                    entity_id=client.session.entity_id,
                    include_health=True,
                    health=self._get_health_value(client),
                    fuel=self._get_energy_value(client),
                )
                self.udp_handler.send_to(c_pkt, client.session.udp_addr)

        # If target is dead, delete their entity with explosion and schedule respawn
        if target.player_health <= 0.0:
            # Track kill/death stats
            attacker.kills += 1
            target.deaths += 1
            print(f"[COMBAT] {target_name} (c{target.client_id}) DESTROYED by {attacker_name} (c{attacker.client_id})"
                  f" [K:{attacker.kills} D:{target.deaths}]")
            self._broadcast_kill_feed(f"{attacker_name} destroyed {target_name}")

            # Broadcast updated stats for attacker and target
            combat_participants = (attacker, target)
            self._broadcast_player_stats(attacker, participants=combat_participants)
            self._broadcast_player_stats(target, participants=combat_participants)

            target_entity_id = target.session.entity_id or target.entity_id

            # DELETE entity with explosion effects
            tick_del = self._get_network_tick(target)
            del_pkt = build_delete_object(tick_del, [target_entity_id], with_effects=True)
            for client in self._snapshot_in_game_clients():
                if not self._combat_observer_packets_allowed_for_client(client, attacker, target):
                    continue
                self._send_packet_to_client(client, del_pkt, prefer_tcp=True)

            # Stop tick loop (entity no longer exists on client)
            target.session.in_game = False

            # Remove from other clients' known entities so they re-create on respawn
            for other in self._snapshot_in_game_clients():
                if other is not target:
                    other.known_entity_ids.discard(target_entity_id)

            # Clear dead player's own known entities and retry tracking.
            # On respawn, _sync_clients_on_spawn will re-create all entities
            # for this player.  Without this, the stale known_entity_ids +
            # expired _entity_create_times retry window would cause
            # _send_entity_create to skip re-creation after respawn.
            target.known_entity_ids.clear()
            if hasattr(target, '_entity_create_times'):
                target._entity_create_times.clear()

            # Reset server-side state for next spawn
            target.player_health = 1.0
            target.player_vel = (0.0, 0.0, 0.0)
            target.player_speed = 0.0
            target.angular_vel_yaw = 0.0
            target.world_collision_ref_pos = target.player_pos
            target.world_collision_bounds_dirty = False
            if target.vehicle_physics:
                target.vehicle_physics.reset()

            # GOAL 4: death/deploy state, no auto-spawn. Player redeploys on flag-click
            # (REINCARNATE 0x00 -> handle_spawn_at_point) on their preserved team. This
            # replaces the old delayed_spawn auto-flow (instant respawn + team_id-or-1
            # neutral-team coercion).
            self._enter_death_deploy_state(target)

    def _apply_building_damage(self, building_oid, proj, attacker: ClientContext, hit_pos: tuple):
        """Apply damage from a projectile to a building.

        Decompile: buildings have per-type health (entity health table VA 0x4E3B00).
        Damage values are absolute HP, not fractional like player damage.
        """
        if building_oid not in self._building_health:
            return
        if self._building_health[building_oid] <= 0:
            return

        # Damage in absolute HP (buildings have 800-5000 HP)
        _PROJECTILE_BUILDING_DAMAGE = {
            EntityType.PULSE_SHELL: 50.0,
            EntityType.PIERCER: 80.0,
            EntityType.THUMPER: 120.0,
            EntityType.HUNTER: 70.0,
            EntityType.HEAVY_MISSILE: 200.0,
            EntityType.MINE: 150.0,
            EntityType.CALTROP: 25.0,
            EntityType.SHORT_MISSILE: 30.0,
            EntityType.FLAK_SHELL: 20.0,
        }
        damage = _PROJECTILE_BUILDING_DAMAGE.get(proj.entity_type, 50.0)
        attacker_name = attacker.session.username or f"Player{attacker.client_id}"
        event = self._apply_building_damage_amount(
            int(building_oid),
            damage,
            source=f"projectile:{int(getattr(proj, 'entity_id', 0) or 0)}:{attacker_name}",
            remove_dynamic_on_destroy=True,
            delete_participants=(attacker,),
        )
        if not event.get("ok"):
            return
        old_hp = float(event.get("old_health", 0.0) or 0.0)
        new_hp = float(event.get("new_health", 0.0) or 0.0)
        max_hp = float(event.get("max_health", 1.0) or 1.0)
        pct = (new_hp / max_hp * 100) if max_hp > 0 else 0.0
        btype_name = str(event.get("entity_type_name") or "UNKNOWN")

        print(
            f"[BUILDING] {attacker_name} hit {btype_name} oid={building_oid} "
            f"for {damage:.0f} dmg ({old_hp:.0f} -> {new_hp:.0f} / {max_hp:.0f}, {pct:.0f}%)"
        )

        # Building destroyed
        if new_hp <= 0:
            print(f"[BUILDING] {btype_name} oid={building_oid} DESTROYED by {attacker_name}")
            # Track building kill
            attacker.kills += 1
            self._broadcast_player_stats(attacker, participants=(attacker,))
            # Chat notification
            from .packets import build_chat_message
            msg = f"DESTROYED! {attacker_name} leveled a {btype_name}!"
            chat_pkt = build_chat_message(msg, source_id=attacker.session.player_id or attacker.entity_id)
            for client in self._snapshot_in_game_clients():
                if (
                    client.tcp_handler
                    and self._combat_observer_packets_allowed_for_client(client, attacker)
                    and self._debug_comm_allowed_for_client(client)
                ):
                    client.tcp_handler.send(chat_pkt)
