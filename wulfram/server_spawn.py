"""SpawnMixin -- spawn-point selection and position resolution, extracted
verbatim from WulframServer (server.py decomposition, step 4). Method-only
mixin; shares state via `self`. Imports only stdlib -> leaf, no cycle.
"""
from __future__ import annotations

import os
from typing import Optional


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
