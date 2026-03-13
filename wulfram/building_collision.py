"""
Server-side building collision helpers.

Reuses the client's model and CBSP collision loaders so the server can resolve
building blockers against actual collision meshes instead of AABB-only shells.
Falls back cleanly when assets are unavailable.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "shared")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from client.wulfram_client.data.models import load_all_models
    from client.wulfram_client.simulation.collision import CollisionMeshCache
except Exception as exc:  # pragma: no cover - environment-dependent import path
    load_all_models = None
    CollisionMeshCache = None
    _IMPORT_ERROR: Optional[Exception] = exc
else:
    _IMPORT_ERROR = None

from wulfram2_protocol.entities import EntityType


BUILDING_MODEL_NAMES = {
    EntityType.FUEL_BUILDING: ("refuel_1", "refuel_2"),
    EntityType.REPAIR_BUILDING: ("repair_1", "repair_2"),
    EntityType.SENSOR_BUILDING: ("flak_turret_1", "flak_turret_2"),
    EntityType.GUN_TURRET: ("gun_turret_1", "gun_turret_2"),
    EntityType.LAUNCHER: ("missile_launcher_1", "missile_launcher_2"),
    EntityType.PAD: ("skypump_1", "skypump_2"),
    EntityType.DARK_LIGHT: ("darklight_1", "darklight_2"),
    EntityType.ENERGY_BUILDING: ("energy_1", "energy_2"),
}


@dataclass(frozen=True)
class BuildingEntity:
    x: float
    y: float
    z: float
    entity_type: int
    team_id: int = 1
    heading: float = 0.0

    @property
    def pos(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class BuildingCollisionAssets:
    """Loads building collision meshes and resolves sphere-vs-building tests."""

    def __init__(self):
        self.models = {}
        self._registry = SimpleNamespace(models=self.models)
        self._cache = CollisionMeshCache() if CollisionMeshCache is not None else None
        self._half_extents_cache: dict[tuple[int, int], Optional[tuple[float, float, float]]] = {}
        self.available = False
        self.last_error = str(_IMPORT_ERROR) if _IMPORT_ERROR else ""
        self.shapes_path: Optional[Path] = None

    def load_default(self):
        """Load all models from the default shapes archive if possible."""
        if load_all_models is None or self._cache is None:
            if not self.last_error:
                self.last_error = "client model/collision loaders unavailable"
            return False

        zip_path = self._resolve_shapes_zip()
        if zip_path is None:
            self.last_error = "shapes.zip not found"
            return False

        try:
            self.models = load_all_models(zip_path)
            self._registry.models = self.models
            self._half_extents_cache.clear()
            self.available = bool(self.models)
            self.shapes_path = zip_path
            if not self.available:
                self.last_error = "shapes.zip loaded but no models were parsed"
            else:
                self.last_error = ""
            return self.available
        except Exception as exc:  # pragma: no cover - depends on external assets
            self.available = False
            self.last_error = str(exc)
            return False

    def test_sphere_collision(
        self,
        building: BuildingEntity,
        sphere_pos: tuple[float, float, float],
        sphere_radius: float,
    ) -> tuple[float, Optional[tuple[float, float, float]]]:
        """Return penetration depth and normal for a sphere-vs-building test."""
        if not self.available or self._cache is None:
            return (0.0, None)

        model_name = self.get_model_name(building.entity_type, building.team_id)
        if not model_name:
            return (0.0, None)

        return self._cache.test_sphere_collision(
            model_name,
            self._registry,
            sphere_pos,
            building.pos,
            building.heading,
            sphere_radius,
        )

    def has_collision_model(self, entity_type: int, team_id: int) -> bool:
        """Return True when this building type has a real collision mesh loaded."""
        if not self.available or self._cache is None:
            return False

        model_name = self.get_model_name(entity_type, team_id)
        if not model_name:
            return False

        model = self.models.get(model_name)
        mesh = getattr(model, "collision_mesh", None) if model is not None else None
        vertices = getattr(mesh, "vertices", None) if mesh is not None else None
        return bool(vertices)

    def get_model_half_extents(self, entity_type: int, team_id: int) -> Optional[tuple[float, float, float]]:
        """Return model-space half-extents for a loaded collision mesh."""
        cache_key = (int(entity_type), int(team_id))
        if cache_key in self._half_extents_cache:
            return self._half_extents_cache[cache_key]

        if not self.available or self._cache is None:
            self._half_extents_cache[cache_key] = None
            return None

        model_name = self.get_model_name(entity_type, team_id)
        if not model_name:
            self._half_extents_cache[cache_key] = None
            return None

        model = self.models.get(model_name)
        mesh = getattr(model, "collision_mesh", None) if model is not None else None
        vertices = getattr(mesh, "vertices", None) if mesh is not None else None
        if not vertices:
            self._half_extents_cache[cache_key] = None
            return None

        extents = (
            max(abs(v.x) for v in vertices),
            max(abs(v.y) for v in vertices),
            max(abs(v.z) for v in vertices),
        )
        self._half_extents_cache[cache_key] = extents
        return extents

    @staticmethod
    def get_model_name(entity_type: int, team_id: int) -> Optional[str]:
        names = BUILDING_MODEL_NAMES.get(entity_type)
        if names is None:
            return None
        if len(names) == 1:
            return names[0]
        if team_id == 2 and len(names) > 1:
            return names[1]
        return names[0]

    def test_segment_collision(
        self,
        building: BuildingEntity,
        start_world_pos: tuple[float, float, float],
        end_world_pos: tuple[float, float, float],
    ) -> bool:
        """Return True when a world-space segment intersects a building mesh."""
        if not self.available or self._cache is None:
            return False

        model_name = self.get_model_name(building.entity_type, building.team_id)
        if not model_name:
            return False

        return self._cache.test_segment_collision(
            model_name,
            self._registry,
            start_world_pos,
            end_world_pos,
            building.pos,
            building.heading,
        )

    def raycast_segment_collision(
        self,
        building: BuildingEntity,
        start_world_pos: tuple[float, float, float],
        end_world_pos: tuple[float, float, float],
    ) -> Optional[tuple[tuple[float, float, float], tuple[float, float, float], float]]:
        """Return closest world-space segment hit on a building mesh, or None."""
        if not self.available or self._cache is None:
            return None

        model_name = self.get_model_name(building.entity_type, building.team_id)
        if not model_name:
            return None

        return self._cache.raycast_segment_collision(
            model_name,
            self._registry,
            start_world_pos,
            end_world_pos,
            building.pos,
            building.heading,
        )

    @staticmethod
    def _resolve_shapes_zip() -> Optional[Path]:
        shapes_zip_raw = os.environ.get("WULFRAM_SHAPES_ZIP", "").strip()
        if shapes_zip_raw:
            shapes_zip = Path(shapes_zip_raw)
            if shapes_zip.exists():
                return shapes_zip

        candidates = (
            _REPO_ROOT / "slurpysoft-wulfram" / "data" / "shapes.zip",
            _REPO_ROOT / "wulfram2-extracted" / "disk1" / "data" / "shapes.zip",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
