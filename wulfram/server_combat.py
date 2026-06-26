"""CombatMixin -- projectile movement, hit detection, damage application,
extracted verbatim from WulframServer (server.py decomposition, step 7).
Method-only mixin; shares state via `self`.
"""
from __future__ import annotations

import math
import os
import threading
import time
import traceback
from typing import Optional

from . import handlers
from .client import ClientContext
from .weapons import (
    EntityType,
    BehaviorSlot,
    OG_DIRECT_TRIGGER_WEAPON_SLOTS,
    build_projectile_spawn_packet,
    build_projectile_update_packet,
)
from .packets import (
    build_chat_message,
    build_delete_object,
    FX_IMPACT_BUILDING,
    FX_IMPACT_TERRAIN,
    FX_IMPACT_VEHICLE,
    FX_MISSILE_FIRE,
    FX_PULSE_FIRE,
    FX_CHAIN_GUN_FIRE,
    FX_FLAK_FIRE,
    build_transient_array,
)
from wulfram2_protocol.entities import tank_softbody_control_slot_value


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

    def _get_projectile_collision_radius(self, proj) -> float:
        radius = self.projectile_collision_radius
        model_names = self._PROJECTILE_MODEL_NAMES.get(proj.entity_type)
        if not model_names or not self._building_collision.available:
            return radius

        team_id = getattr(proj, "team", 1)
        model_name = self._select_team_model_name(model_names, team_id)
        model = self._building_collision.models.get(model_name)
        mesh = getattr(model, "collision_mesh", None) if model is not None else None
        vertices = getattr(mesh, "vertices", None) if mesh is not None else None
        if not vertices:
            return radius

        extents = sorted(
            (
                max(abs(v.x) for v in vertices),
                max(abs(v.y) for v in vertices),
                max(abs(v.z) for v in vertices),
            )
        )
        cross_section_radius = max(0.25, extents[1])
        return min(radius, cross_section_radius)

    def _check_building_collisions(self, ctx, px, py, pz, vx, vy):
        """Check mesh/AABB collision against static buildings and other tanks."""
        for other_ctx in self._snapshot_in_game_clients():
            if other_ctx is ctx:
                continue
            # Check entity-to-entity blocking (tank vs tank)
            ox, oy, oz = other_ctx.player_pos
            dx, dy = px - ox, py - oy
            dist_sq = dx * dx + dy * dy
            min_dist = self._TANK_RADIUS * 2.0
            if dist_sq < min_dist * min_dist and dist_sq > 0.01:
                dist = math.sqrt(dist_sq)
                overlap = min_dist - dist
                # Decompile: penetration slop gate (Physics.c:5380)
                if overlap <= self._PENETRATION_SLOP_DEFAULT:
                    continue
                nx, ny = dx / dist, dy / dist
                px += nx * overlap * 0.5
                py += ny * overlap * 0.5
                vel_dot = vx * nx + vy * ny
                if vel_dot < 0:
                    vx -= nx * vel_dot
                    vy -= ny * vel_dot
                ctx.debug_last_collision = {
                    "kind": "vehicle_sphere",
                    "point": (px, py, pz),
                    "normal": (nx, ny, 0.0),
                    "depth": overlap,
                    "blocker_pos": (ox, oy, oz),
                    "detail": f"tank-vs-tank blocker client={other_ctx.client_id}",
                }

        # Check buildings loaded from map state file
        building_entities = self._building_entities
        for eid, building in building_entities.items():
            if not self._building_blocks_vehicle_collision(building):
                continue
            has_mesh_model = self._building_has_mesh_collision(building)
            mesh_hit = False
            if has_mesh_model:
                depth, normal = self._building_collision.test_sphere_collision(
                    building,
                    (px, py, pz),
                    self._TANK_RADIUS,
                )
                if depth > self._PENETRATION_SLOP_DEFAULT and normal:
                    separation = self._get_static_separation_from_contact(
                        (px, py, pz), (px + normal[0] * depth, py + normal[1] * depth, pz),
                    )
                    push = depth + separation
                    px += normal[0] * push
                    py += normal[1] * push
                    vel_dot = vx * normal[0] + vy * normal[1]
                    if vel_dot < 0:
                        vx -= normal[0] * vel_dot
                        vy -= normal[1] * vel_dot
                    mesh_hit = True
                    ctx.debug_last_collision = {
                        "kind": "building_mesh",
                        "point": (px, py, pz),
                        "normal": normal,
                        "depth": depth,
                        "blocker_pos": (building.x, building.y, building.z),
                        "entity_type": int(building.entity_type),
                        "team_id": int(getattr(building, "team_id", 1)),
                        "detail": f"eid={eid}",
                    }

            if mesh_hit:
                continue
            if has_mesh_model:
                continue

            bx, by = building.x, building.y
            etype = building.entity_type
            hx, hy = self._BUILDING_HALF_EXTENTS.get(etype, (8.0, 8.0))
            r = self._TANK_RADIUS
            if (px > bx - hx - r and px < bx + hx + r and
                    py > by - hy - r and py < by + hy + r):
                push_xp = (bx + hx + r) - px
                push_xn = px - (bx - hx - r)
                push_yp = (by + hy + r) - py
                push_yn = py - (by - hy - r)
                mp = min(push_xp, push_xn, push_yp, push_yn)
                # Decompile: penetration slop gate (Physics.c:5380)
                if mp <= self._PENETRATION_SLOP_DEFAULT:
                    continue
                if mp == push_xp:
                    px = bx + hx + r
                    if vx < 0: vx = 0.0
                    normal = (1.0, 0.0, 0.0)
                    depth = push_xp
                elif mp == push_xn:
                    px = bx - hx - r
                    if vx > 0: vx = 0.0
                    normal = (-1.0, 0.0, 0.0)
                    depth = push_xn
                elif mp == push_yp:
                    py = by + hy + r
                    if vy < 0: vy = 0.0
                    normal = (0.0, 1.0, 0.0)
                    depth = push_yp
                elif mp == push_yn:
                    py = by - hy - r
                    if vy > 0: vy = 0.0
                    normal = (0.0, -1.0, 0.0)
                    depth = push_yn
                else:
                    normal = (0.0, 0.0, 0.0)
                    depth = 0.0
                ctx.debug_last_collision = {
                    "kind": "building_aabb",
                    "point": (px, py, pz),
                    "normal": normal,
                    "depth": depth,
                    "blocker_pos": (building.x, building.y, building.z),
                    "entity_type": int(building.entity_type),
                    "team_id": int(getattr(building, "team_id", 1)),
                    "detail": f"eid={eid}",
                }

        return px, py, vx, vy

    def _broadcast_projectile_delete(
        self,
        proj,
        tick: int,
        *,
        with_effects: bool,
        reason: str,
    ) -> None:
        """Broadcast projectile deletion and record it in the packet log."""
        delete_pkt = build_delete_object(tick, [proj.entity_id], with_effects=with_effects)
        for client in self._snapshot_in_game_clients():
            if not self._projectile_packets_allowed_for_client(client):
                continue
            self._send_packet_to_client(client, delete_pkt, prefer_tcp=True)
            if self.pktlog.enabled:
                self.pktlog.log(
                    client_id=client.client_id,
                    label="PROJ_DELETE",
                    tick=tick,
                    payload=delete_pkt,
                    transport="TCP",
                    extra=f"proj_id=0x{proj.entity_id:X} reason={reason}",
                )

    def _check_projectile_world_hit(self, start_client_pos: tuple, end_client_pos: tuple, proj=None):
        """Raycast a projectile against terrain first, then static world blockers."""
        start_pos = self._from_client_pos(start_client_pos)
        end_pos = self._from_client_pos(end_client_pos)
        return self._raycast_world(start_pos, end_pos)

    def _update_turret_ai(self):
        """Turret AI: GUN_TURRET and LAUNCHER buildings fire at nearby enemies.

        Hitscan damage with fire FX broadcast. Turrets target the closest
        enemy vehicle within range and deal direct damage on a cooldown.

        GUN_TURRET: range 120u, fire every 2.0s, 8% damage per shot
        LAUNCHER: range 200u, fire every 3.0s, 15% damage per shot
        """
        if not self._building_entities:
            return

        now = time.monotonic()

        _TURRET_CONFIG = {
            EntityType.GUN_TURRET: {
                'range_sq': 120.0 * 120.0,
                'fire_interval': 2.0,
                'damage': 0.08,  # normalized 0-1 (8% per shot)
                'fx_type': FX_PULSE_FIRE,
            },
            EntityType.LAUNCHER: {
                'range_sq': 200.0 * 200.0,
                'fire_interval': 3.0,
                'damage': 0.15,  # normalized 0-1 (15% per shot)
                'fx_type': FX_MISSILE_FIRE,
            },
        }

        in_game = self._snapshot_in_game_clients()
        if not in_game:
            return

        for oid, b in self._building_entities.items():
            config = _TURRET_CONFIG.get(b.entity_type)
            if config is None:
                continue
            # Skip destroyed turrets
            if self._building_health.get(oid, 0) <= 0:
                continue
            # Check fire cooldown
            last_fire = self._turret_last_fire.get(oid, 0.0)
            if (now - last_fire) < config['fire_interval']:
                continue

            # Find closest enemy player
            best_target = None
            best_dist_sq = config['range_sq']
            for client in in_game:
                if client.session.team_id == b.team_id:
                    continue  # same team, skip
                if client.player_health <= 0:
                    continue
                dx = client.player_pos[0] - b.x
                dy = client.player_pos[1] - b.y
                dist_sq = dx * dx + dy * dy
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_target = client

            if best_target is None:
                continue

            # Fire! Apply hitscan damage + FX
            self._turret_last_fire[oid] = now
            spawn_pos = (b.x, b.y, b.z + 3.0)

            # Broadcast fire FX at turret position
            from .packets import build_transient_array
            fx_pkt = build_transient_array([{
                'type': config['fx_type'],
                'pos': self._to_client_pos(spawn_pos),
            }])
            if fx_pkt:
                for client in in_game:
                    if not self._transient_fx_allowed_for_client(client):
                        continue
                    if self.udp_handler and client.session.udp_addr:
                        self.udp_handler.send_to(fx_pkt, client.session.udp_addr)

            # Impact FX at target position
            impact_pkt = build_transient_array([{
                'type': FX_IMPACT_VEHICLE,
                'pos': self._to_client_pos(best_target.player_pos),
            }])
            if impact_pkt:
                for client in in_game:
                    if not self._transient_fx_allowed_for_client(client):
                        continue
                    if self.udp_handler and client.session.udp_addr:
                        self.udp_handler.send_to(impact_pkt, client.session.udp_addr)

            # Apply damage to target
            damage = config['damage']
            old_health = best_target.player_health
            best_target.player_health = max(0.0, old_health - damage)
            target_name = best_target.session.username or f"Player{best_target.client_id}"
            btype_name = getattr(b.entity_type, 'name', str(b.entity_type))
            best_target.last_damage_time = now
            best_target.last_damage_source = f"turret:{btype_name}:oid={oid}"
            best_target.last_damage_amount = damage
            best_target.last_damage_old_health = old_health
            best_target.last_damage_new_health = best_target.player_health
            print(
                f"[TURRET] {btype_name} oid={oid} hit {target_name} "
                f"for {damage*100:.0f}% "
                f"({old_health*100:.0f}% -> {best_target.player_health*100:.0f}%)"
            )

            if best_target.player_health <= 0.0 and old_health > 0.0:
                # Turret killed the player
                best_target.deaths += 1
                self._broadcast_player_stats(best_target, participants=(best_target,))
                print(f"[TURRET] {btype_name} oid={oid} KILLED {target_name}")
                self._broadcast_kill_feed(f"{target_name} was destroyed by a {btype_name}")

                # Death sequence: DELETE with effects + respawn
                target_eid = best_target.session.entity_id or best_target.entity_id
                tick_del = self._get_network_tick(best_target)
                del_pkt = build_delete_object(tick_del, [target_eid], with_effects=True)
                for client in in_game:
                    if not self._combat_observer_packets_allowed_for_client(client, best_target):
                        continue
                    self._send_packet_to_client(client, del_pkt, prefer_tcp=True)
                best_target.session.in_game = False
                for other in in_game:
                    if other is not best_target:
                        other.known_entity_ids.discard(target_eid)
                best_target.known_entity_ids.clear()
                if hasattr(best_target, '_entity_create_times'):
                    best_target._entity_create_times.clear()
                best_target.player_health = 1.0
                best_target.player_vel = (0.0, 0.0, 0.0)
                best_target.player_speed = 0.0
                best_target.angular_vel_yaw = 0.0
                best_target.world_collision_ref_pos = best_target.player_pos
                best_target.world_collision_bounds_dirty = False
                if best_target.vehicle_physics:
                    best_target.vehicle_physics.reset()
                # GOAL 4: death/deploy state, no auto-spawn (turret kill). Redeploy on
                # flag-click on the preserved team.
                self._enter_death_deploy_state(best_target)

    def _get_aim_rotation(self, ctx: ClientContext) -> tuple:
        """Return (pitch, yaw, source) for aiming/projectiles."""
        now = time.monotonic()
        aim_recent = (now - ctx.player_aim_time) < self.viewpoint_timeout
        if ctx.player_aim_source == "viewpoint" and aim_recent:
            return ctx.player_aim_pitch, ctx.player_aim_yaw, "viewpoint"
        if ctx.player_aim_source == "slot" and (now - ctx.player_aim_time) < self.aim_hold_time:
            return ctx.player_aim_pitch, ctx.player_aim_yaw, "slot"
        return 0.0, ctx.player_heading, "input"

    def _log_projectile_aim(self, ctx: ClientContext, proj, server_pos: tuple):
        """Log detailed aim/pose context for projectile alignment debugging."""
        def _fmt_vec(vec: Optional[tuple]) -> str:
            if not vec:
                return "(none)"
            return f"({vec[0]:.2f},{vec[1]:.2f},{vec[2]:.2f})"

        def _wrap_angle(angle: float) -> float:
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
            return angle

        pitch, yaw, source = self._get_aim_rotation(ctx)
        viewpoint_recent = (source == "viewpoint")
        roll = ctx.player_pose.get("roll", 0.0)
        heading = ctx.player_heading
        delta = _wrap_angle(yaw - heading)

        slots = ctx.weapon_system.behavior_slots
        turn = slots[BehaviorSlot.TURNING]
        fwd_input = slots[BehaviorSlot.MOVING_FORWARD]
        strafe = slots[BehaviorSlot.MOVING_SIDEWAYS]
        thrust = tank_softbody_control_slot_value(slots)
        slot6 = slots[BehaviorSlot.SLOT6]
        slot7 = slots[BehaviorSlot.SLOT7]
        fire = slots[BehaviorSlot.FIRE]

        player_pos = ctx.player_pos
        player_pos_client = self._to_client_pos(player_pos)
        proj_pos_client = self._to_client_pos(server_pos)

        print(
            f"[PROJ-AIM] id={proj.entity_id} src={source} vp_recent={int(viewpoint_recent)} "
            f"up={self.up_axis} use_pitch={int(ctx.weapon_system.use_pitch)} "
            f"pos_offset={self.pos_offset:.1f}"
        )
        print(
            f"[PROJ-AIM] heading={math.degrees(heading):.1f} yaw={math.degrees(yaw):.1f} "
            f"pitch={math.degrees(pitch):.1f} roll={math.degrees(roll):.1f} "
            f"delta={math.degrees(delta):.1f}"
        )
        print(
            f"[PROJ-AIM] player_srv={_fmt_vec(player_pos)} player_cli={_fmt_vec(player_pos_client)} "
            f"proj_srv={_fmt_vec(server_pos)} proj_cli={_fmt_vec(proj_pos_client)}"
        )
        print(f"[PROJ-AIM] vel={_fmt_vec(proj.vel)}")

        debug = getattr(proj, "debug_context", None) or {}
        fwd_vec = None
        spawn_offset = 0.0
        if debug:
            fwd_vec = debug.get("forward")
            speed = debug.get("speed", 0.0)
            spawn_offset = debug.get("spawn_offset", 0.0)
            aim_yaw = debug.get("yaw", yaw)
            aim_pitch = debug.get("pitch", pitch)
            aim_source = debug.get("aim_source")
            aim_yaw_offset = debug.get("aim_yaw_offset_deg", 0.0)
            aim_pitch_offset = debug.get("aim_pitch_offset_deg", 0.0)
            aim_yaw_invert = int(bool(debug.get("aim_yaw_invert", False)))
            aim_pitch_invert = int(bool(debug.get("aim_pitch_invert", False)))
            print(
                f"[PROJ-AIM] aim_used yaw={math.degrees(aim_yaw):.1f} "
                f"pitch={math.degrees(aim_pitch):.1f} fwd={_fmt_vec(fwd_vec)} "
                f"speed={speed:.1f} spawn_offset={spawn_offset:.1f}"
            )
            if aim_source:
                print(f"[PROJ-AIM] aim_source={aim_source}")
            # Offset decomposition vs forward/right/up for quick mirror diagnosis.
            if proj_pos_client and player_pos_client:
                dx = proj_pos_client[0] - player_pos_client[0]
                dy = proj_pos_client[1] - player_pos_client[1]
                dz = proj_pos_client[2] - player_pos_client[2]
                if self.up_axis == "z":
                    right = (-math.sin(aim_yaw), math.cos(aim_yaw), 0.0)
                    up = (0.0, 0.0, 1.0)
                else:
                    right = (-math.sin(aim_yaw), 0.0, math.cos(aim_yaw))
                    up = (0.0, 1.0, 0.0)
                fwd_dir = fwd_vec if fwd_vec else (0.0, 0.0, 0.0)
                dot_fwd = dx * fwd_dir[0] + dy * fwd_dir[1] + dz * fwd_dir[2]
                dot_right = dx * right[0] + dy * right[1] + dz * right[2]
                dot_up = dx * up[0] + dy * up[1] + dz * up[2]
                print(
                    f"[PROJ-AIM] offset dfwd={dot_fwd:.2f} dright={dot_right:.2f} dup={dot_up:.2f}"
                )
            print(
                f"[PROJ-AIM] aim_tune yaw_off={aim_yaw_offset:.1f} pitch_off={aim_pitch_offset:.1f} "
                f"yaw_inv={aim_yaw_invert} pitch_inv={aim_pitch_invert}"
            )

        # Compare against last sent player state (client-facing) to detect desync.
        last = ctx.last_sent_player_state
        if last:
            sent_pos = last.get("pos")
            sent_rot = last.get("rot")
            dt = time.monotonic() - last.get("time", 0.0)
            if sent_pos and fwd_vec:
                exp_x = sent_pos[0] + spawn_offset * fwd_vec[0]
                exp_y = sent_pos[1] + spawn_offset * fwd_vec[1]
                exp_z = sent_pos[2] + spawn_offset * fwd_vec[2]
                dx = proj_pos_client[0] - exp_x
                dy = proj_pos_client[1] - exp_y
                dz = proj_pos_client[2] - exp_z
                err = math.sqrt(dx * dx + dy * dy + dz * dz)
                print(
                    f"[PROJ-AIM] sent_pos=({_fmt_vec(sent_pos)}) sent_rot=({_fmt_vec(sent_rot)}) "
                    f"dt={dt:.3f}s exp_spawn=({exp_x:.2f},{exp_y:.2f},{exp_z:.2f}) "
                    f"err={err:.2f}"
                )

        print(
            f"[PROJ-AIM] input turn={turn:.3f} fwd={fwd_input:.3f} strafe={strafe:.3f} "
            f"thrust={thrust:.3f} s6={slot6:.3f} s7={slot7:.3f} fire={fire:.3f}"
        )

    # ============ Jump Jet System Handlers ============

    def _on_chain_gun_fire(self, ctx: ClientContext, pos: tuple, rot: tuple, team: int, weapon_name: str = None):
        """Callback when weapon fires (instant hit or placeholder for projectiles)."""
        weapon_name = weapon_name or "Chain Gun"
        print(f"[WEAPON] {weapon_name} fired! pos={pos}")
        if ctx is not None and ctx.weapon_system is not None:
            ws = ctx.weapon_system
            active_slots = {
                str(idx): float(value)
                for idx, value in enumerate(ws.behavior_slots)
                if abs(float(value)) > 0.001
            }
            direct_slots = {
                str(idx): float(ws.behavior_slots[idx])
                for idx in OG_DIRECT_TRIGGER_WEAPON_SLOTS
                if idx < len(ws.behavior_slots) and abs(float(ws.behavior_slots[idx])) > 0.001
            }
            ctx.hitscan_fire_count = int(getattr(ctx, "hitscan_fire_count", 0) or 0) + 1
            ctx.last_hitscan_fire_time = time.monotonic()
            ctx.last_hitscan_weapon_name = str(weapon_name)
            ctx.last_hitscan_fire_input = {
                "active_slots": active_slots,
                "direct_slots": direct_slots,
                "fire": float(ws.behavior_slots[BehaviorSlot.FIRE]),
                "thrust": float(tank_softbody_control_slot_value(ws.behavior_slots)),
                "jumpjet": float(ws.behavior_slots[BehaviorSlot.JUMPJET]),
            }
        if ctx is not None and weapon_name == "Chain Gun":
            target = self._find_chain_gun_target(ctx, pos, rot)
            if target is not None:
                self._apply_hitscan_damage(target, ctx, pos, weapon_name)
        # Python-client-only debug feedback; suppress for OG clients.
        if ctx is not None and ctx.tcp_handler and self._debug_comm_allowed_for_client(ctx):
            if weapon_name == "Chain Gun":
                msg = build_chat_message("*ratatatat*", source_id=ctx.session.player_id or ctx.entity_id)
            else:
                # Other weapons get descriptive feedback
                msg = build_chat_message(f"*{weapon_name.lower()} fired*", source_id=ctx.session.player_id or ctx.entity_id)
            ctx.tcp_handler.send(msg)

    def _find_chain_gun_target(
        self,
        attacker: ClientContext,
        pos: tuple,
        rot: tuple,
        *,
        range_limit: float = 120.0,
        hit_radius: float = 12.0,
    ) -> ClientContext | None:
        """Return the closest in-game target inside the current Chain Gun lane."""
        try:
            origin = tuple(float(v) for v in pos[:3])
        except (TypeError, ValueError):
            origin = tuple(float(v) for v in attacker.player_pos[:3])
        try:
            yaw = float(getattr(attacker, "player_heading", 0.0) or 0.0)
        except (TypeError, ValueError):
            try:
                yaw = float(rot[2])
            except (TypeError, ValueError, IndexError):
                yaw = 0.0

        forward = (math.cos(yaw), math.sin(yaw), 0.0)
        best: tuple[float, ClientContext] | None = None
        nearest: tuple[float, ClientContext] | None = None
        for target in self._snapshot_in_game_clients():
            if target is attacker:
                continue
            try:
                target_pos = tuple(float(v) for v in target.player_pos[:3])
            except (TypeError, ValueError):
                continue
            rel = (
                target_pos[0] - origin[0],
                target_pos[1] - origin[1],
                target_pos[2] - origin[2],
            )
            distance_sq = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]
            if distance_sq <= range_limit * range_limit:
                distance = math.sqrt(distance_sq)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, target)
            along = rel[0] * forward[0] + rel[1] * forward[1] + rel[2] * forward[2]
            if along < 0.0 or along > range_limit:
                continue
            lateral_sq = max(0.0, distance_sq - along * along)
            if lateral_sq > hit_radius * hit_radius:
                continue
            if best is None or along < best[0]:
                best = (along, target)
        return best[1] if best is not None else (nearest[1] if nearest is not None else None)

    def _apply_hitscan_damage(
        self,
        target: ClientContext,
        attacker: ClientContext,
        hit_pos: tuple,
        weapon_name: str,
    ) -> None:
        """Apply controlled-lane hitscan damage without projectile delete traffic."""
        if target.player_health <= 0.0:
            return

        damage = 0.20
        old_health = target.player_health
        target.player_health = round(max(0.0, old_health - damage), 6)
        new_health = target.player_health
        target.last_damage_time = time.monotonic()
        target.last_damage_source = f"hitscan:{weapon_name}"
        target.last_damage_amount = damage
        target.last_damage_old_health = old_health
        target.last_damage_new_health = new_health

        attacker_name = attacker.session.username or f"Player{attacker.client_id}"
        target_name = target.session.username or f"Player{target.client_id}"
        print(
            f"[COMBAT] {attacker_name} (c{attacker.client_id}) hit {target_name} (c{target.client_id}) "
            f"with {weapon_name} for {damage*100:.0f}% damage "
            f"(health: {old_health*100:.0f}% -> {new_health*100:.0f}%)"
        )

        if target.player_health > 0.0:
            return

        attacker.kills += 1
        target.deaths += 1
        print(
            f"[COMBAT] {target_name} (c{target.client_id}) DESTROYED by {attacker_name} "
            f"(c{attacker.client_id}) [K:{attacker.kills} D:{target.deaths}]"
        )
        combat_participants = (attacker, target)
        self._broadcast_player_stats(attacker, participants=combat_participants)
        self._broadcast_player_stats(target, participants=combat_participants)
        self._broadcast_kill_feed(f"{attacker_name} destroyed {target_name}")

        target_entity_id = target.session.entity_id or target.entity_id
        tick_del = self._get_network_tick(target)
        del_pkt = build_delete_object(tick_del, [target_entity_id], with_effects=True)
        for client in self._snapshot_in_game_clients():
            if not self._combat_observer_packets_allowed_for_client(client, attacker, target):
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

        # GOAL 4: death/deploy state, no auto-spawn. Player redeploys on flag-click
        # (REINCARNATE 0x00 -> handle_spawn_at_point) on their preserved team.
        self._enter_death_deploy_state(target)

    def _broadcast_weapon_fire_fx(self, ctx: ClientContext, proj):
        """Broadcast TRANSIENT_ARRAY weapon fire FX to all clients except the firer.

        Uses decompile-backed quantized bitstream format (0x0046CA60).
        Sent via UDP only — FX is cosmetic, loss is acceptable.
        """
        # Map projectile entity types to FX types
        _PROJ_TO_FX = {
            EntityType.FLAK_SHELL: FX_FLAK_FIRE,
            EntityType.PULSE_SHELL: FX_PULSE_FIRE,
            EntityType.HUNTER: FX_MISSILE_FIRE,
            EntityType.PIERCER: FX_PULSE_FIRE,
            EntityType.THUMPER: FX_FLAK_FIRE,
        }
        fx_type = _PROJ_TO_FX.get(proj.entity_type, FX_CHAIN_GUN_FIRE)

        events = [{
            'type': fx_type,
            'pos': proj.pos,
            'entity_id': ctx.entity_id,
        }]
        self._broadcast_transient_fx(events, exclude_client=ctx)

    def _transient_fx_allowed_for_client(self, ctx: ClientContext) -> bool:
        """Return whether cosmetic TRANSIENT_ARRAY FX are currently safe for a client."""
        if handlers._is_loopback_client(ctx):
            return True
        return getattr(self, "remote_transient_fx", False)

    def _projectile_packets_allowed_for_client(self, ctx: ClientContext) -> bool:
        """Return whether projectile entity packets are currently safe for a client."""
        if handlers._is_loopback_client(ctx):
            return True
        return getattr(self, "remote_projectiles", True)

    def _broadcast_transient_fx(self, events: list, *, exclude_client=None) -> bytes:
        """Broadcast cosmetic TRANSIENT_ARRAY FX on the safest currently supported path."""
        pkt = build_transient_array(events)
        if not pkt:
            return b""

        for target in self._snapshot_in_game_clients():
            if target is exclude_client:
                continue
            if not self._transient_fx_allowed_for_client(target):
                continue
            if self.udp_handler and target.session.udp_addr:
                self.udp_handler.send_to(pkt, target.session.udp_addr)
        return pkt

    def _on_projectile_spawn(self, ctx: ClientContext, proj):
        """Callback when a projectile is spawned."""
        print(f"[WEAPON] Projectile spawned: id={proj.entity_id} type={proj.entity_type.name}")

        # NOTE: Do NOT send spawn here to avoid duplicate spawns.
        # Spawn is handled by _spawn_moving_projectile to prevent TCP/UDP reorders.

        # Python-client-only debug feedback; suppress for OG clients.
        if ctx.tcp_handler and self._debug_comm_allowed_for_client(ctx):
            from .packets import build_chat_message
            msg = build_chat_message("*PEW*", source_id=ctx.session.player_id or ctx.entity_id)
            ctx.tcp_handler.send(msg)

    def _send_projectile_spawn(self, ctx: ClientContext, proj, addr: tuple):
        """Send packet to spawn a projectile entity."""
        sent_count = 0
        if self.udp_handler:
            for target in self._snapshot_in_game_clients():
                if not target.session.udp_addr or not target.session.translation_ack_received:
                    continue
                if not self._projectile_packets_allowed_for_client(target):
                    continue
                tick = self._get_network_tick(target)
                include_local_state, local_state_kwargs = self._get_projectile_local_state_for_viewer(target)
                packet = build_projectile_spawn_packet(
                    proj,
                    tick,
                    include_local_state=include_local_state,
                    **local_state_kwargs,
                    entity_config=self.projectile_config,
                    is_static=self.projectile_spawn_snap,
                )
                self.udp_handler.send_to(packet, target.session.udp_addr)
                # UPDATE_ARRAY (0x0E) over TCP crashes OG client (TCP bitstream
                # desync â†’ protocol mismatch). UDP-only is fine for projectiles.
                if self.pktlog.enabled:
                    self.pktlog.log(
                        client_id=target.client_id,
                        label="PROJ_SPAWN",
                        tick=tick,
                        payload=packet,
                        transport="UDP",
                        entity_count=1,
                        entity_ids=(proj.entity_id,),
                        mask_bits=(0b1111,),  # pos+vel+rot+type_info
                        has_local_state=include_local_state,
                        health=self._get_health_value(target) if include_local_state else -1.0,
                        extra=f"proj_type={proj.entity_type}",
                    )
                sent_count += 1
        if sent_count:
            print(f"[WEAPON] Sent projectile spawn via UDP: id={proj.entity_id} targets={sent_count}")
        if self.debug_projectiles:
            print(
                f"[PROJ-SPAWN] id={proj.entity_id} type={proj.entity_type} "
                f"config={self.projectile_config} spawn_snap={int(self.projectile_spawn_snap)}"
            )
            if os.environ.get("WULFRAM_DEBUG_PROJECTILE_HEX", "0") == "1":
                print(f"[PROJ-HEX] id={proj.entity_id} (per-client packets, no single hex to show)")

        # Send chat message with projectile position for debugging (TCP only).
        pos_msg = f"FIRE! pos=({proj.pos[0]:.1f},{proj.pos[1]:.1f},{proj.pos[2]:.1f}) vel=({proj.vel[0]:.1f},{proj.vel[1]:.1f},{proj.vel[2]:.1f})"
        if ctx.tcp_handler and self._debug_comm_allowed_for_client(ctx):
            chat_packet = build_chat_message(pos_msg, source_id=ctx.session.player_id or ctx.entity_id)
            ctx.tcp_handler.send(chat_packet)
        print(f"[WEAPON] {pos_msg}")

        # Belt-and-suspenders: send an immediate heartbeat UPDATE_ARRAY to the
        # firing player so their HUD health stays current even if the projectile
        # local_state bits are somehow missed or arrive out of order.
        if self.udp_handler and ctx.session.udp_addr:
            hb_tick = self._get_network_tick(ctx)
            hb_health = self._get_health_value(ctx)
            hb_packet = self._build_local_state_heartbeat(
                ctx,
                tick=hb_tick,
                entity_id=ctx.session.entity_id,
                include_health=True,
                health=hb_health,
                fuel=self._get_energy_value(ctx),
            )
            self.udp_handler.send_to(hb_packet, ctx.session.udp_addr)
            if self.pktlog.enabled:
                self.pktlog.log(
                    client_id=ctx.client_id,
                    label="PROJ_FIRE_HEARTBEAT",
                    tick=hb_tick,
                    payload=hb_packet,
                    transport="UDP",
                    entity_count=1,
                    entity_ids=(0xFFFFFFFE,),
                    mask_bits=(0,),
                    has_local_state=True,
                    health=hb_health,
                )
            print(f"[PROJ-HEARTBEAT] Sent post-fire heartbeat UPDATE_ARRAY (health={hb_health:.2f})")

        if self.debug_projectiles and proj.debug_context:
            hp_shape = proj.debug_context.get("hardpoint_shape")
            hp_raw = proj.debug_context.get("hardpoint_raw")
            hp_local = proj.debug_context.get("hardpoint_local")
            hp_order = proj.debug_context.get("hardpoint_order")
            hp_scale = proj.debug_context.get("shape_scale")
            hp_world = proj.debug_context.get("hardpoint_world_offset")
            muzzle_push = proj.debug_context.get("muzzle_push")
            hp_origin = proj.debug_context.get("hardpoint_origin_mode")
            hp_fsign = proj.debug_context.get("hardpoint_forward_sign")
            hp_rsign = proj.debug_context.get("hardpoint_right_sign")
            hp_usign = proj.debug_context.get("hardpoint_up_sign")
            hp_swap = proj.debug_context.get("hardpoint_swap_fr")
            if hp_shape and hp_raw and hp_local:
                raw_fmt = f"({hp_raw[0]},{hp_raw[1]},{hp_raw[2]})"
                loc_fmt = f"({hp_local[0]:.2f},{hp_local[1]:.2f},{hp_local[2]:.2f})"
                world_fmt = "(n/a)"
                if hp_world:
                    world_fmt = f"({hp_world[0]:.2f},{hp_world[1]:.2f},{hp_world[2]:.2f})"
                print(
                    f"[HARDPOINT] shape={hp_shape} name={ctx.weapon_system.projectile_hardpoint_name} "
                    f"raw={raw_fmt} order={hp_order} scale={hp_scale} local={loc_fmt} "
                    f"world={world_fmt} origin={hp_origin} "
                    f"f_sign={hp_fsign} r_sign={hp_rsign} u_sign={hp_usign} swap_fr={int(bool(hp_swap))} "
                    f"muzzle_push={muzzle_push}"
                )
