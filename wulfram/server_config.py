"""ConfigMixin — env-parsing config init extracted from WulframServer.__init__.

Part of the server.py decomposition (docs/refactor/server-py-decomposition-plan.md).
Method-only mixin: no __init__ of its own, shares all state via `self`. Each
`_init_*` method holds a coherent config block moved VERBATIM out of the core
`__init__` and is invoked in sequence from it. Pure move — no logic changes.
"""
from __future__ import annotations

import os


class ConfigMixin:
    """Config / env-parsing init methods for WulframServer (method-only mixin)."""

    def _init_jump_jet_config(self):
        # Jump jets are a custom extension driven by OG slot 4. Keep the env
        # override, but make the playable clone default match the promoted
        # Crossroads demo slice.
        self.jump_jets_enabled = os.environ.get("WULFRAM_JUMP_JETS", "1") == "1"
        self.jump_jet_direction = os.environ.get("WULFRAM_JUMP_JET_DIRECTION", "body").strip().lower()
        if self.jump_jet_direction not in ("body", "world"):
            self.jump_jet_direction = "body"
        self.jump_jet_correction_burst_count = max(
            0,
            int(os.environ.get("WULFRAM_JUMP_JET_CORRECTION_BURST", "12")),
        )
        self.jump_jet_correction_burst_interval = max(
            0.01,
            float(os.environ.get("WULFRAM_JUMP_JET_CORRECTION_INTERVAL", "0.05")),
        )
        self.jump_jet_collision_guard = (
            os.environ.get("WULFRAM_JUMP_JET_COLLISION_GUARD", "1")
            .strip()
            .lower()
            not in ("0", "false", "off", "no")
        )
        self.jump_jet_collision_guard_xy = max(
            0.0,
            float(os.environ.get("WULFRAM_JUMP_JET_COLLISION_GUARD_XY", "1.0")),
        )
        self.jump_jet_collision_guard_zpop = max(
            0.0,
            float(os.environ.get("WULFRAM_JUMP_JET_COLLISION_GUARD_ZPOP", "2.0")),
        )
        self.jump_jet_landing_clearance = max(
            0.0,
            float(os.environ.get("WULFRAM_JUMP_JET_LANDING_CLEARANCE", "1.85")),
        )

    def _init_tank_terrain_projection_guard_config(self):
        self.tank_terrain_projection_guard = (
            os.environ.get("WULFRAM_TANK_TERRAIN_PROJECTION_GUARD", "1")
            .strip()
            .lower()
            not in ("0", "false", "off", "no")
        )
        self.tank_terrain_projection_guard_xy = max(
            0.0,
            float(os.environ.get("WULFRAM_TANK_TERRAIN_PROJECTION_GUARD_XY", "1.0")),
        )
        self.tank_terrain_projection_guard_zpop = max(
            0.0,
            float(os.environ.get("WULFRAM_TANK_TERRAIN_PROJECTION_GUARD_ZPOP", "2.0")),
        )
        self.tank_terrain_projection_guard_min_clearance = max(
            0.0,
            float(os.environ.get("WULFRAM_TANK_TERRAIN_PROJECTION_GUARD_MIN_CLEARANCE", "0.0")),
        )

    def _init_player_info_local_state_config(self):
        # PLAYER_INFO triggers local-state sync on the client. For remote/OG
        # clients, default to canonical local-state so hull/fuel/ammo/turret
        # state initialize through the decompile-backed path. Only the spawn
        # TankPacket keeps the short-form-safe weapon id because 0x18 does not
        # carry the extra ammo/turret bits.
        player_info_local_state_env = os.environ.get("WULFRAM_PLAYER_INFO_LOCAL_STATE")
        self.player_info_local_state_explicit = player_info_local_state_env is not None
        player_info_local_state_raw = (
            player_info_local_state_env.strip().lower()
            if player_info_local_state_env is not None
            else "auto-remote"
        )
        if player_info_local_state_raw in ("1", "true", "on", "yes", "force"):
            self.player_info_local_state_mode = "force"
        elif player_info_local_state_raw in ("0", "false", "off", "no"):
            self.player_info_local_state_mode = "off"
        elif player_info_local_state_raw in ("remote", "auto-remote"):
            self.player_info_local_state_mode = "auto-remote"
        else:
            self.player_info_local_state_mode = "auto"
        self.player_info_local_state = self.player_info_local_state_mode != "off"
        # Wulf-forge does NOT send PLAYER(spectator=0) during spawn.
        # KEEP THIS 0 FOR OG CLIENTS. =1 sends build_player(spectator=False) AFTER the
        # IN_GAME transition (see _spawn_wf_style ~4618). On the OG client a PLAYER (0x17)
        # packet re-runs LocalPlayer_initialize, which tears down the in-game view and
        # bounces the client straight back to team-select ("flicker -> back to map, can't
        # spawn"), orphaning the just-spawned tank (looks like a "duplicate spectator copy
        # of yourself"). Empirically confirmed 2026-06-01 on the VM OG client: sending this
        # one packet to an in-game client kicked it to team-select; =0 spawns stay in-game.
        # It was added to try to fix the duplicate-spectator artifact but did NOT fix it and
        # caused the worse bounce. Do not re-enable for real clients.
        self.spawn_send_player_active = os.environ.get("WULFRAM_SPAWN_PLAYER_ACTIVE", "0") == "1"
        # Local-state weapon type (entity type index). Default is 0 (Tank).
        # Tank (0): pool_entry[2]=9 â†’ 9 ammo bits + weapon_def+0x170=1 â†’ 16 turret bits.
        #   Required for weapons: sync_local_player calls update_active_flags every heartbeat;
        #   without 9-bit ammo bitmask (all 1s), active flags reset to 0 â†’ fire check fails.
        # Scout (1): weapon_def+0x68=1 â†’ secondary turret bits (untested).
        # AssaultPlatform (2): pool_entry[2]=0 â†’ no ammo bits â†’ weapons can never fire.
        try:
            self.local_state_weapon_type = int(os.environ.get("WULFRAM_LOCAL_STATE_WEAPON_TYPE", "0"))
        except ValueError:
            self.local_state_weapon_type = 0
        # TankPacket spawn vitals are encoded as the short local-state form
        # (weapon + health + energy only). Keep a safe default weapon type here
        # so the client does not expect ammo/turret bits during 0x18 spawn parse.
        try:
            self.spawn_tank_weapon_type = int(os.environ.get("WULFRAM_SPAWN_TANK_WEAPON_TYPE", "2"))
        except ValueError:
            self.spawn_tank_weapon_type = 2
        if self.spawn_tank_weapon_type < 0 or self.spawn_tank_weapon_type > 31:
            self.spawn_tank_weapon_type = 2
        # In WF local-state mode, include turret angles to match client's
        # read_local_player_state expectations.  Tank (weapon_type=0) requires
        # primary turret (weapon_def+0x170=1).  Omitting these bits causes a
        # bitstream desync: the client reads 16 turret bits from the entity
        # section, garbling entity_count and preventing health application.
        self.wf_local_state_turrets = os.environ.get("WULFRAM_WF_LOCAL_TURRETS", "0") == "1"
        # Local-state ammo bitmask parameters (active slot flags). Defaults match our BEHAVIOR config (no active flags).
        try:
            self.local_state_ammo_bits = int(os.environ.get("WULFRAM_LOCAL_STATE_AMMO_BITS", "0"))
        except ValueError:
            self.local_state_ammo_bits = 0
        try:
            self.local_state_ammo_mask = int(os.environ.get("WULFRAM_LOCAL_STATE_AMMO_MASK", "0"))
        except ValueError:
            self.local_state_ammo_mask = 0
        # Local-state ammo bit width should follow BEHAVIOR-derived slot capabilities.
        self.local_state_ammo_from_behavior = os.environ.get("WULFRAM_LOCAL_STATE_AMMO_FROM_BEHAVIOR", "1") == "1"
        self.local_state_ammo_override = (
            "WULFRAM_LOCAL_STATE_AMMO_BITS" in os.environ
            or "WULFRAM_LOCAL_STATE_AMMO_MASK" in os.environ
        )
        # Turret angle bits for local state (if weapon def flags require them).
        try:
            self.local_state_turret_bits = int(os.environ.get("WULFRAM_LOCAL_STATE_TURRET_BITS", "16"))
        except ValueError:
            self.local_state_turret_bits = 16
        try:
            self.local_state_turret_max = float(os.environ.get("WULFRAM_LOCAL_STATE_TURRET_MAX", "6.3"))
        except ValueError:
            self.local_state_turret_max = 6.3
        try:
            self.local_state_turret_range = float(os.environ.get("WULFRAM_LOCAL_STATE_TURRET_RANGE", "12.6"))
        except ValueError:
            self.local_state_turret_range = 12.6
        self.local_state_primary_override = os.environ.get("WULFRAM_LOCAL_STATE_PRIMARY_TURRET", "").strip()
        self.local_state_secondary_override = os.environ.get("WULFRAM_LOCAL_STATE_SECONDARY_TURRET", "").strip()
        self.allow_unsafe_local_state = os.environ.get("WULFRAM_ALLOW_UNSAFE_LOCAL_STATE", "0") == "1"

        # Guard against unsafe local-state configs (missing turret bits or unknown weapon type).
        # Local-state only stays enabled when config is known safe or explicitly overridden.
        local_state_safe = False
        if self.local_state_weapon_type == 0:
            local_state_safe = self.local_state_turret_bits > 0 and (
                self.local_state_primary_override not in ("0", "false", "False")
            )
        elif self.local_state_weapon_type == 1:
            local_state_safe = self.local_state_turret_bits > 0 and (
                self.local_state_secondary_override not in ("0", "false", "False")
            )
        # wf mode now sends full local-state (weapon + health + energy + ammo + turret)
        # to enable weapon firing. Keep strict safety checks for auto/force modes.
        if self.update_local_state_mode not in ("off", "wf") and not self.allow_unsafe_local_state and not local_state_safe:
            print(
                "[LOCAL-STATE] Unsafe local-state config; disabling local-state. "
                "Set WULFRAM_ALLOW_UNSAFE_LOCAL_STATE=1 to override."
            )
            self.update_local_state_mode = "off"
            self.update_local_state = False

    def _init_correction_config(self):
        # ── OG-faithful GATED-RARE correction gate (GOAL 2, 2026-06-02) ──────
        # The OG client corrects RARELY and reactively. The proactive streams
        # above (spawn burst, movement-correction interval, STATE_REQUEST replies)
        # each camera-clamp/zero-velocity on the OG client; on a zero-latency
        # loopback client that flood freezes rendering. When this gate is enabled
        # (default), those proactive timers no longer emit VIEW_UPDATE corrections.
        # Instead a single correction fires only when the authoritative state has
        # genuinely DIVERGED beyond a threshold (accumulated server-only,
        # non-input-predictable displacement — terrain-Z clamp / collision push —
        # since the last correction), and corrections are hard-rate-limited so a
        # zero-latency client cannot be flooded even under rapid STATE_REQUEST/fire.
        # The gate is IDENTICAL for every client; the loopback fix falls out of the
        # rate cap, not a client-type branch.
        self.correction_gate_enabled = os.environ.get("WULFRAM_CORRECTION_GATE", "1").strip().lower() not in ("0", "off", "false", "no")
        try:
            # Accumulated divergence (units) required to admit a correction.
            self.correction_divergence_pos = float(os.environ.get("WULFRAM_CORRECTION_DIVERGENCE_POS", "1.5"))
        except ValueError:
            self.correction_divergence_pos = 1.5
        if self.correction_divergence_pos <= 0.0:
            self.correction_divergence_pos = 1.5
        try:
            # Accumulated heading divergence (degrees) required to admit a correction.
            self.correction_divergence_heading_deg = float(os.environ.get("WULFRAM_CORRECTION_DIVERGENCE_HEADING", "12.0"))
        except ValueError:
            self.correction_divergence_heading_deg = 12.0
        try:
            # Hard floor between any two corrections to a client (rate cap, seconds).
            self.correction_min_interval = float(os.environ.get("WULFRAM_CORRECTION_MIN_INTERVAL", "0.2"))
        except ValueError:
            self.correction_min_interval = 0.2
        if self.correction_min_interval < 0.0:
            self.correction_min_interval = 0.0
        # Periodic correction (2026-06-17): opt-in steady-cadence correction that
        # fires DURING movement (like the jumpjet force-correction), so divergence
        # is nudged out continuously instead of accumulating into a big jumpjet-
        # triggerable rubber-band. 0 = off (default). Unlike burst/divergence
        # corrections it is NOT held back by the movement-suppress pause; each
        # emit is a single authoritative snapshot, so a small interval keeps the
        # per-correction nudge tiny (smooth) while bounding max drift.
        try:
            self.periodic_correction_interval = float(os.environ.get("WULFRAM_PERIODIC_CORRECTION_S", "0"))
        except ValueError:
            self.periodic_correction_interval = 0.0
        if self.periodic_correction_interval < 0.0:
            self.periodic_correction_interval = 0.0
        try:
            # Per-tick noise floor: server-only displacement below this is treated
            # as float/quantization noise and ignored (keeps flat ground at ~0).
            self.correction_divergence_floor = float(os.environ.get("WULFRAM_CORRECTION_DIVERGENCE_FLOOR", "0.02"))
        except ValueError:
            self.correction_divergence_floor = 0.02
        if self.correction_divergence_floor < 0.0:
            self.correction_divergence_floor = 0.0
        try:
            # Per-tick multiplicative decay of the divergence accumulator so
            # transient/slow noise bleeds off and never reaches threshold; only
            # genuine spikes or sustained divergence trip the gate.
            self.correction_divergence_decay = float(os.environ.get("WULFRAM_CORRECTION_DIVERGENCE_DECAY", "0.98"))
        except ValueError:
            self.correction_divergence_decay = 0.98
        self.correction_divergence_decay = min(1.0, max(0.0, self.correction_divergence_decay))
        # Debug counters for empirical flat-ground verification of the gate.
        self.correction_gate_debug = os.environ.get("WULFRAM_CORRECTION_GATE_DEBUG", "0").strip().lower() in ("1", "on", "true", "yes")
        # ── Correction-trigger fix (2026-06-09) ─────────────────────────────
        # The divergence accumulator above is fed ONLY by the terrain-Z clamp,
        # so it is structurally blind to LATERAL divergence (zero on flat
        # ground), and STATE_REQUEST carries no client position — the server
        # cannot measure lateral drift from the wire. The OG client's own
        # STATE_REQUEST cadence is the only divergence-shaped signal available,
        # so when enabled (default) a STATE_REQUEST queues the standard ~10Hz
        # settle burst even under the correction gate, capped to one burst per
        # state_request_burst_min_interval so fire-spam STATE_REQUESTs cannot
        # reproduce the GOAL-2 flood/freeze. Identical for every client.
        self.state_request_burst_enabled = os.environ.get("WULFRAM_STATE_REQUEST_BURST", "1").strip().lower() not in ("0", "off", "false", "no")
        try:
            self.state_request_burst_min_interval = float(os.environ.get("WULFRAM_STATE_REQUEST_BURST_MIN_INTERVAL", "1.75"))
        except ValueError:
            self.state_request_burst_min_interval = 1.75
        if self.state_request_burst_min_interval < 0.0:
            self.state_request_burst_min_interval = 0.0
        # GOAL 8: per-step suspension/Z instrumentation (default OFF).
        self.goal8_zdebug = os.environ.get("WULFRAM_GOAL8_ZDEBUG", "0").strip().lower() in ("1", "on", "true", "yes")
        try:
            self.goal8_zdebug_every = int(os.environ.get("WULFRAM_GOAL8_ZDEBUG_EVERY", "15") or 15)
        except ValueError:
            self.goal8_zdebug_every = 15
        # GOAL 8 (2026-06-04): sub-step the position/suspension integration to the OG
        # inner-substep size so the vertical hover spring matches the client's
        # GameSim_substep_update (Physics.c:1974, ~40ms inner) instead of overshooting
        # when integrated at the coarse outer client-frame dt (~84ms on OG WARP). The
        # coarse step makes the nonlinear hover spring ring ~2u on any displacement
        # (e.g. the spawn-height vs equilibrium mismatch) — the live DO idle Z bounce.
        # Legacy=1 restores the single full-dt step (A/B). Default = fixed.
        self.goal8_legacy = os.environ.get("WULFRAM_GOAL8_LEGACY", "0").strip().lower() in ("1", "on", "true", "yes")
        try:
            # Inner sub-step cap (seconds). Default = server tick period; empirically
            # a 0.95u spawn displacement rings ~0.46u at 33ms vs ~2.08u at 84ms.
            self.goal8_substep_cap_s = float(os.environ.get("WULFRAM_GOAL8_SUBSTEP_CAP_MS", "")) / 1000.0
        except ValueError:
            self.goal8_substep_cap_s = 1.0 / max(1.0, self.tick_rate_hz)
        if self.goal8_substep_cap_s <= 0.0:
            self.goal8_substep_cap_s = 1.0 / max(1.0, self.tick_rate_hz)
        try:
            # Only substep the vertical-stabilizing path when the tank is near-
            # stationary (horizontal speed below this). Active driving/coasting keeps
            # the legacy single full-dt step so the coast trajectory the client
            # reconciles against is unchanged. 0 disables the speed gate (always sub).
            self.goal8_substep_speed_max = float(os.environ.get("WULFRAM_GOAL8_SUBSTEP_SPEED_MAX", "0.5"))
        except ValueError:
            self.goal8_substep_speed_max = 0.5

        print(
            f"[CONFIG-CORRECTION-GATE] enabled={int(self.correction_gate_enabled)} "
            f"divergence_pos={self.correction_divergence_pos}u "
            f"divergence_heading={self.correction_divergence_heading_deg}deg "
            f"min_interval={self.correction_min_interval}s "
            f"floor={self.correction_divergence_floor}u decay={self.correction_divergence_decay} "
            f"state_request_burst={int(self.state_request_burst_enabled)} "
            f"burst_queue_min_interval={self.state_request_burst_min_interval}s"
        )

    def _init_spawn_config(self):
        spawn_height_env = os.environ.get("WULFRAM_SPAWN_HEIGHT")
        try:
            # Default to ground-height spawn to avoid immediate fall-damage overlays.
            self.spawn_height = float(spawn_height_env) if spawn_height_env is not None else 5.0
        except ValueError:
            self.spawn_height = 5.0
        try:
            self.spawn_delay_seconds = float(os.environ.get("WULFRAM_SPAWN_DELAY", "6.0"))
        except ValueError:
            self.spawn_delay_seconds = 6.0
        # Spawn is driven by the explicit map-flag click (REINCARNATE subtype 0x00 ->
        # handlers.handle_spawn_at_point). Team-select no longer auto-spawns. Combat
        # respawn still uses the delayed_spawn mechanism in the game loop.
        self.team_switch_send_reincarnate = os.environ.get("WULFRAM_TEAM_SWITCH_REINCARNATE", "1") == "1"
        self.team_switch_send_update_stats = (
            os.environ.get("WULFRAM_TEAM_SWITCH_UPDATE_STATS", "1").strip().lower()
            not in ("0", "false", "off", "no")
        )
        self.team_switch_update_stats_transport = os.environ.get(
            "WULFRAM_TEAM_SWITCH_UPDATE_STATS_TRANSPORT",
            "udp",
        ).strip().lower()
        if self.team_switch_update_stats_transport not in ("udp", "tcp", "auto"):
            self.team_switch_update_stats_transport = "udp"
        self.team_switch_update_stats_variant = os.environ.get(
            "WULFRAM_TEAM_SWITCH_UPDATE_STATS_VARIANT",
            "canonical",
        ).strip().lower()
        if self.team_switch_update_stats_variant not in ("canonical", "team_first"):
            self.team_switch_update_stats_variant = "canonical"
        self.team_switch_send_roster = (
            os.environ.get("WULFRAM_TEAM_SWITCH_ROSTER", "0").strip().lower()
            not in ("0", "false", "off", "no")
        )
        self.team_switch_send_entry_packets = (
            os.environ.get("WULFRAM_TEAM_SWITCH_ENTRY_PACKETS", "1").strip().lower()
            not in ("0", "false", "off", "no")
        )
        # Allow explicit spawn-point packets to recover clients stuck on entry-map
        # after auto-spawn already marked the session IN_GAME.
        self.spawn_allow_point_override = os.environ.get("WULFRAM_SPAWN_POINT_OVERRIDE", "1") == "1"
        try:
            self.spawn_point_override_min_interval = float(
                os.environ.get("WULFRAM_SPAWN_POINT_OVERRIDE_MIN_INTERVAL", "0.0")
            )
        except ValueError:
            self.spawn_point_override_min_interval = 0.0
        try:
            self.spawn_force_after = float(os.environ.get("WULFRAM_SPAWN_FORCE_AFTER", "12.0"))
        except ValueError:
            self.spawn_force_after = 12.0
        try:
            # Avoid false "crash" diagnosis when scripted input arrives slightly late.
            self.inactivity_timeout = float(os.environ.get("WULFRAM_INACTIVITY_TIMEOUT", "120.0"))
        except ValueError:
            self.inactivity_timeout = 120.0
        try:
            self.remote_idle_timeout = float(os.environ.get("WULFRAM_REMOTE_IDLE_TIMEOUT", "900.0"))
        except ValueError:
            self.remote_idle_timeout = 900.0
        # Keep spawns pinned to ground unless explicitly disabled.
        self.spawn_sets_ground_level = os.environ.get("WULFRAM_SPAWN_SET_GROUND", "1") == "1"
        try:
            self.ground_override_release_distance = float(
                os.environ.get("WULFRAM_GROUND_OVERRIDE_RELEASE_DISTANCE", "24.0")
            )
        except ValueError:
            self.ground_override_release_distance = 24.0
        try:
            self.ground_override_release_height = float(
                os.environ.get("WULFRAM_GROUND_OVERRIDE_RELEASE_HEIGHT", "4.0")
            )
        except ValueError:
            self.ground_override_release_height = 4.0
        try:
            self.ground_override_release_terrain_distance = float(
                os.environ.get("WULFRAM_GROUND_OVERRIDE_RELEASE_TERRAIN_DISTANCE", "4.0")
            )
        except ValueError:
            self.ground_override_release_terrain_distance = 4.0
        try:
            self.ground_override_release_terrain_height = float(
                os.environ.get("WULFRAM_GROUND_OVERRIDE_RELEASE_TERRAIN_HEIGHT", "0.75")
            )
        except ValueError:
            self.ground_override_release_terrain_height = 0.75
        # Spawn packet toggles (useful for crash isolation).
        # Spawn sequence toggles (default to Wulf-Forge minimal behavior).
        self.spawn_send_udp_tank = os.environ.get("WULFRAM_SPAWN_UDP_TANK", "1") == "1"
        spawn_player_info_env = os.environ.get("WULFRAM_SPAWN_PLAYER_INFO")
        self.spawn_send_player_info_explicit = spawn_player_info_env is not None
        if self.spawn_send_player_info_explicit:
            self.spawn_send_player_info = spawn_player_info_env == "1"
        else:
            # AzureFishy decompile shows PLAYER_INFO is the canonical local-player
            # init path. Keep loopback/Python on the existing minimal path by
            # default, but send PLAYER_INFO to remote OG clients unless explicitly
            # disabled.
            self.spawn_send_player_info = False
        # Keep UPDATE_ARRAY pre-create opt-in until it preserves the OG
        # local sync/camera bootstrap. Proven correction sessions can use the
        # local sync state plus server_expected, even when the camera global is 0.
        self.spawn_send_update_array = os.environ.get("WULFRAM_SPAWN_UPDATE_ARRAY", "0") == "1"
        self.spawn_send_game_clock = os.environ.get("WULFRAM_SPAWN_GAME_CLOCK", "1") == "1"
        self.spawn_send_comm_message = (
            os.environ.get("WULFRAM_SPAWN_COMM_MESSAGE", "0").strip().lower()
            not in ("0", "false", "off", "no")
        )
        # Wulf-forge does NOT send REINCARNATE(0x11) during spawn.
        self.spawn_send_reincarnate = os.environ.get("WULFRAM_SPAWN_REINCARNATE", "0") == "1"
        spawn_entry_transition = os.environ.get("WULFRAM_SPAWN_ENTRY_TRANSITION", "off").strip().lower()
        if spawn_entry_transition not in ("0", "1", "false", "true", "off", "on", "auto", "auto-remote"):
            spawn_entry_transition = "off"
        self.spawn_entry_transition = spawn_entry_transition
        self.spawn_send_birth_notice = os.environ.get("WULFRAM_SPAWN_BIRTH_NOTICE", "0") == "1"
        self.spawn_send_player_packet = os.environ.get("WULFRAM_SPAWN_PLAYER_PACKET", "0") == "1"
        self.spawn_player_spectator = os.environ.get("WULFRAM_SPAWN_PLAYER_SPECTATOR", "0") == "1"
        self.player_info_properties_mode = os.environ.get("WULFRAM_PLAYER_INFO_PROPERTIES", "team").strip().lower()
