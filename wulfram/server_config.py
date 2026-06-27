"""ConfigMixin — env-parsing config init extracted from WulframServer.__init__.

Part of the server.py decomposition (docs/refactor/server-py-decomposition-plan.md).
Method-only mixin: no __init__ of its own, shares all state via `self`. Each
`_init_*` method holds a coherent config block moved VERBATIM out of the core
`__init__` and is invoked in sequence from it. Pure move — no logic changes.
"""
from __future__ import annotations

import os
import math

from .session import FEATURES


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
        # Prediction-lead extrapolation (2026-06-26). Characterized via
        # tools/wulftap_turn_capture.py + analyze_turn_capture.py: the OG client
        # runs deterministic prediction ~1 server tick AHEAD of the server's
        # confirmed authoritative pose. The yaw/position gap through a turn holds
        # a CONSTANT ~0.5-1.0 tick of motion (mean 0.73, std 0.24) — it does NOT
        # grow with turn angle, so the turn RATE is lockstep and the residual is
        # a pure phase lead, not a sim divergence. An open-loop correction that
        # snaps the client to the *current* authoritative pose therefore yanks it
        # ~1 tick BACKWARD on every cadence = the visible jerk. Extrapolating the
        # correction TARGET forward by this many ticks — using the very
        # velocity/angular-velocity the integrator advances pos/heading with —
        # lands the snap where the client already predicted: a near no-op when
        # prediction is correct (and exactly a no-op at rest, since vel=angvel=0),
        # while genuine divergence still snaps. 0 = legacy (target current pose).
        try:
            self.correction_lead_ticks = float(os.environ.get("WULFRAM_CORRECTION_LEAD_TICKS", "1.0"))
        except ValueError:
            self.correction_lead_ticks = 1.0
        if self.correction_lead_ticks < 0.0:
            self.correction_lead_ticks = 0.0
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
            f"lead_ticks={self.correction_lead_ticks} "
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

    def _init_replication_config(self):
        # Spawn points are required for reliable entry-map progression.
        # Keep ON by default; allow explicit opt-out for protocol experiments.
        spawn_points_env = os.environ.get("WULFRAM_SEND_SPAWN_POINTS", "1")
        FEATURES.send_spawn_points = spawn_points_env.strip().lower() not in ("0", "false", "off", "no")
        behavior_env = os.environ.get("WULFRAM_SEND_BEHAVIOR")
        if behavior_env is not None:
            FEATURES.send_behavior_packet = behavior_env.strip().lower() not in ("0", "false", "off", "no")
        translation_env = os.environ.get("WULFRAM_SEND_TRANSLATION")
        if translation_env is not None:
            FEATURES.send_translation_packet = translation_env.strip().lower() not in ("0", "false", "off", "no")
        world_stats_env = os.environ.get("WULFRAM_SEND_WORLD_STATS_ON_LOGIN")
        if world_stats_env is not None:
            FEATURES.send_world_stats_on_login = world_stats_env.strip().lower() not in ("0", "false", "off", "no")
        game_clock_login_env = os.environ.get("WULFRAM_SEND_GAME_CLOCK_ON_LOGIN")
        self.send_game_clock_on_login = (
            game_clock_login_env is not None
            and game_clock_login_env.strip().lower() not in ("0", "false", "off", "no")
        )
        request_start_login_env = os.environ.get("WULFRAM_SEND_REQUEST_START_ON_LOGIN")
        self.send_request_start_on_login = (
            request_start_login_env is not None
            and request_start_login_env.strip().lower() not in ("0", "false", "off", "no")
        )
        roster_login_env = os.environ.get("WULFRAM_SEND_ROSTER_ON_LOGIN")
        self.send_roster_on_login = (
            roster_login_env is not None
            and roster_login_env.strip().lower() not in ("0", "false", "off", "no")
        )
        tick_loop_env = os.environ.get("WULFRAM_TICK_LOOP_ENABLED")
        if tick_loop_env is not None:
            FEATURES.tick_loop_enabled = tick_loop_env.strip().lower() not in ("0", "false", "off", "no")
        auto_login_env = os.environ.get("WULFRAM_AUTO_LOGIN")
        if auto_login_env is not None:
            FEATURES.auto_login = auto_login_env.strip().lower() not in ("0", "false", "off", "no")
        # WULFRAM_AUTO_JOIN_TEAM removed: auto-join/auto-spawn caused client crashes and
        # spawn-under-terrain bugs. Clients spawn via the explicit spawn-point flow only.
        # Client/world offset for position alignment (default to 0 for pure server-space).
        if self.up_axis == "z":
            try:
                self.pos_offset = float(os.environ.get("WULFRAM_POS_OFFSET_Z", "0.0"))
            except ValueError:
                self.pos_offset = 0.0
        else:
            try:
                self.pos_offset = float(os.environ.get("WULFRAM_POS_OFFSET_Y", "0.0"))
            except ValueError:
                self.pos_offset = 0.0
        # Server-authoritative physics: send position/velocity updates to client
        # Goal: Server controls movement, client renders server state
        # NOTE: Tick loop causes client input freeze - disabled for now (see _spawn_wf_style)
        self.send_player_updates = os.environ.get("WULFRAM_SEND_PLAYER_UPDATES", "1") == "1"
        # Remote player updates (multiplayer position sync) â€” independent of local heartbeat.
        self.send_remote_updates = os.environ.get("WULFRAM_SEND_REMOTE_UPDATES", "1") == "1"
        # T3 OG-participant isolation gates. These default on so normal
        # behavior is unchanged, but live probes can selectively disable
        # target->OG visibility streams without disturbing loopback clients.
        self.og_viewer_roster_entry = os.environ.get("WULFRAM_OG_VIEWER_ROSTER_ENTRY", "1") == "1"
        self.og_viewer_entity_create = os.environ.get("WULFRAM_OG_VIEWER_ENTITY_CREATE", "1") == "1"
        self.og_viewer_remote_updates = os.environ.get("WULFRAM_OG_VIEWER_REMOTE_UPDATES", "1") == "1"
        # Wulf-forge sends UPDATE_ARRAY over UDP only.
        self.send_updates_tcp = os.environ.get("WULFRAM_UPDATE_TCP", "0") == "1"
        self.send_updates_udp = os.environ.get("WULFRAM_UPDATE_UDP", "1") == "1"
        # Local player update mask (align with wulf-forge minimal updates by default).
        # Modes: pos, pos_vel, pos_rot, pos_vel_rot
        # Default to full local transform updates to avoid client-side state divergence.
        self.local_update_mode = os.environ.get("WULFRAM_LOCAL_UPDATE_MODE", "pos_vel_rot").strip().lower()
        # Remote update shape to avoid malformed packets destabilizing HUD state in multi-client.
        # Modes: pos, pos_rot, pos_vel_rot, heartbeat, off (default: pos_rot).
        self.remote_update_mode = os.environ.get("WULFRAM_REMOTE_UPDATE_MODE", "pos_vel_rot").strip().lower()
        # Health vitals delta-bit on remote-player UPDATE_ARRAY (makes the remote
        # tank targetable: entity+0xD0>0). Gated so we can A/B it against a
        # suspected render-context freeze. Default ON.
        self.remote_entity_vitals = os.environ.get("WULFRAM_REMOTE_ENTITY_VITALS", "1") == "1"
        # Server-generated kill-feed chat notifications (the OG client has no built-in
        # kill feed; kill notices ride server chat). Default on; A/B off.
        self.kill_feed_enabled = os.environ.get("WULFRAM_KILL_FEED", "1") == "1"
        # Honor the client's selected vehicle at spawn (Tank/Scout/Bomber). Default
        # on; off forces every spawn to a Tank (legacy behaviour).
        self.vehicle_select_enabled = os.environ.get("WULFRAM_VEHICLE_SELECT", "1") == "1"
        # Throttle remote player updates: interval in seconds (0 = every tick).
        # 30Hz tick rate = every tick; lower values reduce packet flood.
        try:
            self.remote_update_interval = float(os.environ.get("WULFRAM_REMOTE_UPDATE_INTERVAL", "0"))
        except ValueError:
            self.remote_update_interval = 0
        # For remote player updates, set is_manned bit (likely required for player vehicles).
        self.remote_update_is_manned = os.environ.get("WULFRAM_REMOTE_IS_MANNED", "0") == "1"
        # Yaw offset for remote entities (degrees). Tunable via control cmd: heading remote_offset <deg>
        self.remote_yaw_offset = math.radians(float(os.environ.get("WULFRAM_REMOTE_YAW_OFFSET_DEG", "0")))
        # Negate remote yaw before sending (applies after offset)
        self.remote_yaw_negate = os.environ.get("WULFRAM_REMOTE_YAW_NEGATE", "0") == "1"
        # Combine local + remote updates into a single UPDATE_ARRAY per tick.
        self.combine_update_arrays = os.environ.get("WULFRAM_COMBINE_UPDATE_ARRAYS", "0") == "1"
        # Local-state in projectile UPDATE_ARRAY packets is fragile and can
        # misalign entity decoding. Keep projectile packets entity-only by
        # default; normal heartbeat UPDATE_ARRAY packets keep HUD vitals stable.
        self.projectile_local_stats = os.environ.get("WULFRAM_PROJECTILE_LOCAL_STATS", "0") == "1"
        self.debug_projectiles = (
            os.environ.get("WULFRAM_DEBUG_PROJECTILES", "0") == "1"
            or os.environ.get("WULFRAM_DEBUG_AIM", "0") == "1"
        )
        try:
            self.projectile_config = int(os.environ.get("WULFRAM_PROJECTILE_CONFIG", "0"))
        except ValueError:
            self.projectile_config = 0
        # Projectile spawn snap/teleport flag (misleadingly named "static" elsewhere).
        # Default ON: wulf-forge sets this bit on entity creation.
        projectile_snap_env = os.environ.get("WULFRAM_PROJECTILE_SPAWN_SNAP")
        if projectile_snap_env is None:
            projectile_snap_env = os.environ.get("WULFRAM_PROJECTILE_STATIC", "1")
        self.projectile_spawn_snap = projectile_snap_env == "1"
        # Backwards-compat alias; "static" here really means spawn snap/teleport.
        self.projectile_static = self.projectile_spawn_snap
        try:
            self.viewpoint_timeout = float(os.environ.get("WULFRAM_VIEWPOINT_TIMEOUT", "1.0"))
        except ValueError:
            self.viewpoint_timeout = 1.0
        self.debug_viewpoint = os.environ.get("WULFRAM_DEBUG_VIEWPOINT", "0") == "1"
        self.debug_udp_raw = os.environ.get("WULFRAM_DEBUG_UDP_RAW", "0") == "1"
        self.weapon_energy_enabled = os.environ.get("WULFRAM_WEAPON_ENERGY_ENABLED", "1") == "1"
        try:
            self.player_energy_max = float(os.environ.get("WULFRAM_PLAYER_ENERGY_MAX", "100.0"))
        except ValueError:
            self.player_energy_max = 100.0
        if self.player_energy_max <= 0.0:
            self.player_energy_max = 100.0
        try:
            self.player_energy_regen = float(os.environ.get("WULFRAM_PLAYER_ENERGY_REGEN", "10.0"))
        except ValueError:
            self.player_energy_regen = 10.0
        if self.player_energy_regen < 0.0:
            self.player_energy_regen = 0.0
        # VIEW_UPDATE is an auxiliary replay/correction path. Primary gameplay
        # replication stays on UPDATE_ARRAY unless explicitly experimenting.
        self.view_update_enabled = os.environ.get("WULFRAM_VIEW_UPDATE", "0") == "1"
        # Periodic auxiliary VIEW_UPDATE loop. Default OFF for stability.
        self.view_update_loop = os.environ.get("WULFRAM_VIEW_UPDATE_LOOP", "0") == "1"
        # Keep local-stats out of VIEW_UPDATE unless explicitly requested.
        self.view_update_local_stats = os.environ.get("WULFRAM_VIEW_UPDATE_LOCAL_STATS", "0") == "1"
        # Wulf-forge marks HEALTH dirty on spawn; include entity vitals by default.
        # This uses update-mask bits 5/7 (health/energy per wulf-forge).
        # Entity vitals (mask bits 5/7) should be sent only when dirty; default off.
        self.view_update_entity_vitals = os.environ.get("WULFRAM_VIEW_ENTITY_VITALS", "0") == "1"
        self.update_entity_vitals = os.environ.get("WULFRAM_UPDATE_ENTITY_VITALS", "0") == "1"
        self.debug_vitals = os.environ.get("WULFRAM_DEBUG_VITALS", "0") == "1"
        self.debug_health_pattern = os.environ.get("WULFRAM_DEBUG_HEALTH_PATTERN", "0") == "1"
        try:
            self.debug_health_value = float(os.environ.get("WULFRAM_DEBUG_HEALTH_VALUE", "1.0"))
        except ValueError:
            self.debug_health_value = 1.0
        try:
            self.debug_health_low = float(os.environ.get("WULFRAM_DEBUG_HEALTH_LOW", "0.2"))
        except ValueError:
            self.debug_health_low = 0.2
        try:
            self.debug_health_period = float(os.environ.get("WULFRAM_DEBUG_HEALTH_PERIOD", "1.0"))
        except ValueError:
            self.debug_health_period = 1.0
        if self.debug_health_pattern:
            print(
                "[WARN] Debug health pattern enabled: "
                f"value={self.debug_health_value} low={self.debug_health_low} "
                f"period={self.debug_health_period}s (HUD will flicker red)"
            )
        try:
            self.view_update_interval = float(os.environ.get("WULFRAM_VIEW_UPDATE_INTERVAL", "0.1"))
        except ValueError:
            self.view_update_interval = 0.1
        try:
            self.update_grace_seconds = float(os.environ.get("WULFRAM_UPDATE_GRACE", "0.0"))
        except ValueError:
            self.update_grace_seconds = 0.0
        # Wulf-forge sends UPDATE_ARRAY immediately; keep OFF by default while aligning.
        self.require_client_tick = os.environ.get("WULFRAM_REQUIRE_CLIENT_TICK", "0") == "1"
        try:
            self.multi_spawn_offset = float(os.environ.get("WULFRAM_MULTI_SPAWN_OFFSET", "0.0"))
        except ValueError:
            self.multi_spawn_offset = 0.0
        try:
            self.weapon_id = int(os.environ.get("WULFRAM_WEAPON_ID", "0"))
        except ValueError:
            self.weapon_id = 0
        if self.weapon_id < 0 or self.weapon_id > 31:
            print(f"[WEAPON] Invalid weapon_id={self.weapon_id}, defaulting to 0")
            self.weapon_id = 0
        # Default to body-aligned firing for stability. Slot/viewpoint aim remains
        # available via env toggles for targeted experiments.
        self.projectile_aim_source = os.environ.get("WULFRAM_PROJECTILE_AIM_SOURCE", "body").lower()
        # Decompile-backed firing math uses the entity rotation matrix
        # (entity +0x30/+0x34/+0x38), but the playable slice keeps the previous
        # yaw-only body source unless this live OG gate flag is enabled.
        self.projectile_body_pitch = self._projectile_body_pitch_enabled_from_env()
        # Terrain-conform replicated (player-built) buildings: a flat pad placed at
        # its center terrain-Z cuts through any slope (uphill half sinks, downhill
        # half floats) = the "pad floats into terrain" report. When on, the dynamic
        # building's replicated rotation gets terrain pitch/roll so it lies on the
        # slope. Static MAP buildings are client-local-rendered and unaffected. 0 = off.
        self.building_terrain_conform = os.environ.get("WULFRAM_BUILDING_TERRAIN_CONFORM", "1") != "0"
        # Cargo economy (construction Phase 1). Pickup: when a player drives within
        # cargo_pickup_range of the team supply ship and isn't carrying, they pick up
        # a cargo box (default_cargo_type). build_require_cargo gates whether a build
        # consumes carried cargo (default off so dynbuild/testing builds freely; on =
        # real economy where you build what you carry).
        try:
            self.cargo_pickup_range = float(os.environ.get("WULFRAM_CARGO_PICKUP_RANGE", "30.0"))
        except ValueError:
            self.cargo_pickup_range = 30.0
        from .weapons import EntityType
        try:
            self.default_cargo_type = int(os.environ.get("WULFRAM_DEFAULT_CARGO_TYPE", str(int(EntityType.REPAIR_BUILDING))))
        except (ValueError, TypeError):
            self.default_cargo_type = int(EntityType.REPAIR_BUILDING)
        self.build_require_cargo = os.environ.get("WULFRAM_BUILD_REQUIRE_CARGO", "0").strip().lower() in ("1", "on", "true", "yes")
        # Broadcasting a CARGO_BOX (type 0x13) entity currently CRASHES the OG client
        # (missing the type-0x13 DEFINITION field -> bitstream desync -> PROTOCOL ERROR).
        # Off until build_update_array_create_tank writes that field (Phase 3).
        self.cargo_box_entity_enabled = os.environ.get("WULFRAM_CARGO_BOX_ENTITY", "0").strip().lower() in ("1", "on", "true", "yes")
        # Supply-ship cargo as a finite, replenishing resource (the OG cargo_times
        # mechanism; DOCKING 0x38 has no OG client handler). The ship holds up to
        # ship_cargo_capacity boxes; each pickup depletes one; one box restocks every
        # ship_replenish_s seconds. So cargo isn't infinite.
        try:
            self.ship_cargo_capacity = max(1, int(os.environ.get("WULFRAM_SHIP_CARGO_CAPACITY", "3")))
        except ValueError:
            self.ship_cargo_capacity = 3
        try:
            self.ship_replenish_s = max(0.0, float(os.environ.get("WULFRAM_SHIP_REPLENISH_S", "12")))
        except ValueError:
            self.ship_replenish_s = 12.0
        # Construction timer: a just-built structure is "under construction" for this
        # many seconds (provides no service until complete). 0 = instant (default).
        try:
            self.construction_timeout = max(0.0, float(os.environ.get("WULFRAM_CONSTRUCTION_TIMEOUT_S", "0")))
        except ValueError:
            self.construction_timeout = 0.0
        # Deconstruction timer: a deconstruct takes this many seconds (building stays,
        # no service, then is removed + cargo refunded). 0 = instant (default).
        try:
            self.deconstruction_timeout = max(0.0, float(os.environ.get("WULFRAM_DECONSTRUCTION_TIMEOUT_S", "0")))
        except ValueError:
            self.deconstruction_timeout = 0.0
        self.projectiles_enabled = os.environ.get("WULFRAM_PROJECTILES_ENABLED", "1") == "1"
        self.remote_projectiles = os.environ.get("WULFRAM_REMOTE_PROJECTILES", "1") == "1"
        self.remote_combat_observer_packets = (
            os.environ.get("WULFRAM_REMOTE_COMBAT_OBSERVER_PACKETS", "1") == "1"
        )
        # Projectile update mode:
        # 0=no updates after spawn, 1=5Hz, 2=15Hz (default), 3=30Hz
        try:
            self.projectile_update_mode = int(os.environ.get("WULFRAM_PROJECTILE_UPDATE_MODE", "2"))
        except ValueError:
            self.projectile_update_mode = 2
        if self.projectile_update_mode < 0 or self.projectile_update_mode > 3:
            self.projectile_update_mode = 2
        try:
            self.projectile_collision_radius = abs(float(os.environ.get("WULFRAM_PROJECTILE_COLLISION_RADIUS", "2.0")))
        except ValueError:
            self.projectile_collision_radius = 2.0
        if self.projectile_collision_radius < 0.25:
            self.projectile_collision_radius = 0.25
        # Remote OG impact FX are still prone to live D_ERR disconnects.
        # Keep TRANSIENT_ARRAY enabled for loopback/Python validation, but
        # suppress it remotely until the 0x0D path is verified end to end.
        self.remote_transient_fx = os.environ.get("WULFRAM_REMOTE_TRANSIENT_FX", "0") == "1"
        # TankPacket vitals OFF: flag=0 means the client skips the entire
        # local_state read (weapon, health, fuel, ammo, turret).  This ensures
        # unit_type/net_id/team/pos are parsed correctly so the entity is
        # created and the entry map dismisses.
        # Health is instead delivered by the heartbeat UPDATE_ARRAY which
        # includes entity_count=1, calling Network_record_update_stats (sets
        # EAX for tick guard) then sync_local_player (applies health from ESI).
        # The BEHAVIOR packet clears weapon_def[0]+0x170 (turret flag), so the
        # heartbeat local_state has no turret bit misalignment.
        tank_vitals_raw = os.environ.get("WULFRAM_TANK_VITALS", "0").strip().lower()
        self.tank_vitals = tank_vitals_raw in ("1", "true", "on", "yes")


    def _init_map_config(self):
        # Map configuration (WORLD_STATS + spawn points).
        # Default to Wulf-Forge's startup map ("crossroads").
        self.map_name = os.environ.get("WULFRAM_MAP_NAME", "crossroads")
        self.map_config = self._load_map_config()
        map_key = self.map_name.lower()
        map_conf = self.map_config.get(map_key, {}) if isinstance(self.map_config, dict) else {}
        map_conf_has_rows = "grid_rows" in map_conf
        map_conf_has_cols = "grid_cols" in map_conf
        map_rows_env = os.environ.get("WULFRAM_MAP_GRID_ROWS")
        map_cols_env = os.environ.get("WULFRAM_MAP_GRID_COLS")
        map_scale_env = os.environ.get("WULFRAM_MAP_SCALE")
        allow_large_env = os.environ.get("WULFRAM_MAP_ALLOW_LARGE_GRID")
        self.map_allow_large_grid = bool(map_conf.get("allow_large_grid", False))
        if allow_large_env is not None:
            self.map_allow_large_grid = allow_large_env.strip().lower() in ("1", "true", "yes", "on")
        try:
            if map_rows_env is not None:
                self.map_grid_rows = int(map_rows_env)
            elif "grid_rows" in map_conf:
                self.map_grid_rows = int(map_conf["grid_rows"])
            else:
                self.map_grid_rows = 1
        except ValueError:
            self.map_grid_rows = 1
        try:
            if map_cols_env is not None:
                self.map_grid_cols = int(map_cols_env)
            elif "grid_cols" in map_conf:
                self.map_grid_cols = int(map_conf["grid_cols"])
            else:
                self.map_grid_cols = 1
        except ValueError:
            self.map_grid_cols = 1
        try:
            if map_scale_env is not None:
                self.map_scale = float(map_scale_env)
            elif "scale" in map_conf:
                self.map_scale = float(map_conf["scale"])
            else:
                self.map_scale = 1.0
        except ValueError:
            self.map_scale = 1.0
        # Clamp explicit config/env grid sizes if the client treats WORLD_STATS rows/cols as signed.
        if (self.map_grid_rows > 127 or self.map_grid_cols > 127) and not self.map_allow_large_grid:
            clamped_rows = min(self.map_grid_rows, 127)
            clamped_cols = min(self.map_grid_cols, 127)
            if clamped_rows != self.map_grid_rows or clamped_cols != self.map_grid_cols:
                print(
                    "[MAP] Config grid exceeds 127; "
                    f"clamping {self.map_grid_rows}x{self.map_grid_cols} -> {clamped_rows}x{clamped_cols}"
                )
                self.map_grid_rows = clamped_rows
                self.map_grid_cols = clamped_cols
        # Auto-detect grid rows/cols from map land file when not explicitly set.
        needs_rows = map_rows_env is None and not map_conf_has_rows
        needs_cols = map_cols_env is None and not map_conf_has_cols
        if needs_rows or needs_cols:
            land_grid = self._load_map_land_grid()
            if land_grid:
                rows, cols, size_x, size_y = land_grid
                # WORLD_STATS uses u8; client may treat this as signed char.
                # Clamp to 127 to avoid negative values if sign-extended unless explicitly allowed.
                if (rows > 127 or cols > 127) and not self.map_allow_large_grid:
                    clamped_rows = min(rows, 127)
                    clamped_cols = min(cols, 127)
                    print(
                        "[MAP] Land grid exceeds 127; "
                        f"clamping {rows}x{cols} -> {clamped_rows}x{clamped_cols}"
                    )
                    rows, cols = clamped_rows, clamped_cols
                elif rows > 127 or cols > 127:
                    print(
                        "[MAP] Land grid exceeds 127; "
                        f"keeping {rows}x{cols} (allow_large_grid=1)"
                    )
                if needs_rows:
                    self.map_grid_rows = rows
                if needs_cols:
                    self.map_grid_cols = cols
                print(
                    "[MAP] Land grid detected "
                    f"{rows}x{cols} (size={size_x:.1f}x{size_y:.1f})"
                )
        self.map_spawn_points = self._parse_spawn_points_env(os.environ.get("WULFRAM_SPAWN_POINTS", ""))
        default_use_map_spawn = "1" if self.map_name.lower() == "crossroads" else "0"
        self.use_map_spawn_points = os.environ.get("WULFRAM_USE_MAP_SPAWN", default_use_map_spawn) == "1"
        self.align_spawn_points_to_terrain = (
            os.environ.get("WULFRAM_ALIGN_SPAWN_POINTS_TO_TERRAIN", "0") == "1"
        )
        self.align_spawn_pos_to_terrain = (
            os.environ.get("WULFRAM_ALIGN_SPAWN_POS_TO_TERRAIN", "0") == "1"
        )
        self.default_flat_spawn_pos = self._parse_spawn_pos_env(
            os.environ.get("WULFRAM_DEFAULT_FLAT_SPAWN", "")
        )
        if self.default_flat_spawn_pos is None:
            self.default_flat_spawn_pos = self._get_builtin_flat_spawn_pos()
        self.force_default_spawn_pos = os.environ.get("WULFRAM_FORCE_DEFAULT_SPAWN", "1") == "1"


    def _init_tick_input_config(self):
        # Tick selection: default to server tick for stability across multiple clients.
        self.use_client_ticks = os.environ.get("WULFRAM_USE_CLIENT_TICKS", "0") == "1"
        print(f"[CONFIG] use_client_ticks={int(self.use_client_ticks)}")
        try:
            self.remote_og_movement_input_delay = float(
                os.environ.get("WULFRAM_REMOTE_OG_MOVEMENT_INPUT_DELAY", "0.20")
            )
        except ValueError:
            self.remote_og_movement_input_delay = 0.20
        if self.remote_og_movement_input_delay < 0.0:
            self.remote_og_movement_input_delay = 0.0
        try:
            self.remote_og_movement_input_stale_clamp = float(
                os.environ.get("WULFRAM_REMOTE_OG_MOVEMENT_INPUT_STALE_CLAMP", "0")
            )
        except ValueError:
            self.remote_og_movement_input_stale_clamp = 0.0
        if self.remote_og_movement_input_stale_clamp < 0.0:
            self.remote_og_movement_input_stale_clamp = 0.0
        try:
            self.remote_og_movement_input_after_max = float(
                os.environ.get("WULFRAM_REMOTE_OG_MOVEMENT_INPUT_AFTER_MAX", "0.20")
            )
        except ValueError:
            self.remote_og_movement_input_after_max = 0.20
        if self.remote_og_movement_input_after_max < 0.0:
            self.remote_og_movement_input_after_max = 0.0
        self.remote_og_movement_input_selection = (
            os.environ.get(
                "WULFRAM_REMOTE_OG_MOVEMENT_INPUT_SELECTION",
                "latest_before_target",
            )
            .strip()
            .lower()
        )
        if self.remote_og_movement_input_selection in {
            "",
            "default",
            "before",
            "before_target",
            "latest_before",
            "latest_before_target",
            "floor",
        }:
            self.remote_og_movement_input_selection = "latest_before_target"
        elif self.remote_og_movement_input_selection in {
            "nearest",
            "nearest_to_target",
            "closest",
            "closest_to_target",
        }:
            self.remote_og_movement_input_selection = "nearest_to_target"
        elif self.remote_og_movement_input_selection in {
            "after",
            "after_target",
            "bounded_after",
            "bounded_after_target",
            "first_after",
            "first_after_target",
        }:
            self.remote_og_movement_input_selection = "bounded_after_target"
        elif self.remote_og_movement_input_selection in {
            "tick_before",
            "tick_before_target",
            "latest_before_tick",
            "latest_before_tick_target",
            "client_tick_before",
            "client_tick_before_target",
        }:
            self.remote_og_movement_input_selection = "latest_before_tick_target"
        elif self.remote_og_movement_input_selection in {
            "tick_nearest",
            "tick_nearest_target",
            "nearest_tick",
            "nearest_tick_target",
            "client_tick_nearest",
            "client_tick_nearest_target",
        }:
            self.remote_og_movement_input_selection = "nearest_tick_target"
        else:
            self.remote_og_movement_input_selection = "latest_before_target"
        print(f"[CONFIG] remote_og_movement_input_delay={self.remote_og_movement_input_delay:.2f}s")
        print(
            "[CONFIG] remote_og_movement_input_selection="
            f"{self.remote_og_movement_input_selection} "
            f"stale_clamp={self.remote_og_movement_input_stale_clamp:.2f}s "
            f"after_max={self.remote_og_movement_input_after_max:.2f}s"
        )
        tick_probe_mode = (
            os.environ.get("WULFRAM_REMOTE_OG_MOVEMENT_INPUT_TICK_PROBE", "0")
            .strip()
            .lower()
        )
        self.remote_og_movement_input_tick_probe = tick_probe_mode in {
            "1",
            "true",
            "on",
            "yes",
            "probe",
            "tick",
            "client_tick",
        }
        if self.remote_og_movement_input_tick_probe:
            print("[CONFIG] remote_og_movement_input_tick_probe=1")
