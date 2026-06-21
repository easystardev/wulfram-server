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
