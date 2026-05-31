"""
Decompile-shaped terrain-grid world collision helpers.

This module starts moving the server toward the original
`Collision_test_entity_world` / `TerrainGrid_test_CBSP_collision` structure:

- world space is partitioned into a 3x3 terrain-sector grid
- collision queries classify AABBs into sector ranges
- each sector traverses overlapping terrain quads
- quads are split with the original alternating diagonal parity

The current live consumers are projectile-vs-terrain collision and the
server-side entity-vs-world terrain resolver. Tank/entity contact response
still keeps inferred pieces around contact recording and response ordering.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


def _clamp(value: int, lo: int, hi: int) -> int:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _dot3(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale3(v, scale: float):
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def _cross3(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length3(v) -> float:
    return math.sqrt(_dot3(v, v))


def _normalize3(v):
    length = _length3(v)
    if length <= 1e-8:
        return None
    inv = 1.0 / length
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def _angle_between3(a, b) -> float | None:
    len_a = _length3(a)
    len_b = _length3(b)
    if len_a <= 1e-8 or len_b <= 1e-8:
        return None
    dot = _dot3(a, b) / (len_a * len_b)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _signbit(value: float) -> bool:
    return math.copysign(1.0, value) < 0.0


def _closest_point_on_triangle(point, v0, v1, v2):
    """Return the closest point on a triangle to `point`."""
    ab = _sub3(v1, v0)
    ac = _sub3(v2, v0)
    ap = _sub3(point, v0)

    d1 = _dot3(ab, ap)
    d2 = _dot3(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return v0

    bp = _sub3(point, v1)
    d3 = _dot3(ab, bp)
    d4 = _dot3(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return v1

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return _add3(v0, _scale3(ab, v))

    cp = _sub3(point, v2)
    d5 = _dot3(ab, cp)
    d6 = _dot3(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return v2

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return _add3(v0, _scale3(ac, w))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return _add3(v1, _scale3(_sub3(v2, v1), w))

    denom = 1.0 / (va + vb + vc)
    v_param = vb * denom
    w_param = vc * denom
    return _add3(v0, _add3(_scale3(ab, v_param), _scale3(ac, w_param)))


@dataclass(frozen=True)
class TerrainSector:
    row: int
    col: int
    index: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    origin_x: float
    origin_y: float
    step_x: float
    step_y: float


@dataclass(frozen=True)
class TerrainHit:
    position: tuple[float, float, float]
    sector_index: int
    cell: tuple[int, int]


@dataclass(frozen=True)
class TerrainContact:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    penetration: float
    sector_index: int
    cell: tuple[int, int]
    normal_source: str = "unknown"
    cbsp_split_normal: Optional[tuple[float, float, float]] = None
    terrain_face_normal: Optional[tuple[float, float, float]] = None
    mesh_face_normal: Optional[tuple[float, float, float]] = None
    entity_radial_normal: Optional[tuple[float, float, float]] = None
    cbsp_store_normal0: Optional[tuple[float, float, float]] = None
    cbsp_store_normal1: Optional[tuple[float, float, float]] = None
    cbsp_record_hit_source: Optional[str] = None
    cbsp_mesh_triangle_indices: Optional[tuple[int, int, int]] = None
    cbsp_guess7_order: Optional[tuple[int, int, int]] = None
    cbsp_guess7_terms: Optional[tuple[float, float, float]] = None
    cbsp_edge_hit_kind: Optional[str] = None
    cbsp_edge_t: Optional[float] = None
    cbsp_node_index: Optional[int] = None
    cbsp_node_depth: Optional[int] = None
    cbsp_node_mesh_normal_angle_deg: Optional[float] = None


@dataclass(frozen=True)
class _CBSPContact:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    penetration: float
    cbsp_split_normal: Optional[tuple[float, float, float]] = None
    terrain_face_normal: Optional[tuple[float, float, float]] = None
    mesh_face_normal: Optional[tuple[float, float, float]] = None
    entity_radial_normal: Optional[tuple[float, float, float]] = None
    store_normal0: Optional[tuple[float, float, float]] = None
    store_normal1: Optional[tuple[float, float, float]] = None
    record_hit_source: Optional[str] = None
    mesh_triangle_indices: Optional[tuple[int, int, int]] = None
    guess7_order: Optional[tuple[int, int, int]] = None
    guess7_terms: Optional[tuple[float, float, float]] = None
    edge_hit_kind: Optional[str] = None
    edge_t: Optional[float] = None
    node_index: Optional[int] = None
    node_depth: Optional[int] = None
    node_mesh_normal_angle_deg: Optional[float] = None

    def __iter__(self):
        yield self.position
        yield self.normal
        yield self.penetration


@dataclass(frozen=True)
class TerrainRaycastHit:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    sector_index: int
    cell: tuple[int, int]
    distance: float


class TerrainGridCollision:
    """3x3 terrain-sector collision traversal modeled after BSP.c."""

    def __init__(
        self,
        terrain,
        height_offset: float,
        sector_rows: int = 3,
        sector_cols: int = 3,
        model_contact_normal_source: str = "mesh",
    ):
        self.terrain = terrain
        self.height_offset = height_offset
        self.sector_rows = sector_rows
        self.sector_cols = sector_cols
        # Per-step collision wall-clock budget (set by the server each physics step).
        self._query_deadline = None
        self._query_deadline_hits = 0
        # Cache of precomputed per-CBSP-node mesh triangles (vertices/normal/plane-d).
        # The entity collision mesh is static, so these are recomputed-once not per query.
        self._node_mesh_cache: dict = {}
        normal_source = str(model_contact_normal_source or "mesh").strip().lower()
        self.model_contact_normal_source = (
            "terrain"
            if normal_source in {"terrain", "face", "triangle"}
            else "entity_radial"
            if normal_source
            in {
                "entity_radial",
                "radial",
                "contact_to_body",
                "body_from_contact",
                "decompile_context",
                "decompile_cbsp_context",
            }
            else "mesh"
        )
        self.cell_x = terrain.cell_x
        self.cell_y = terrain.cell_z
        self.world_w = terrain.world_w
        self.world_h = terrain.world_h
        self.max_cell_x = max(0, terrain.num_x - 2)
        self.max_cell_y = max(0, terrain.num_z - 2)
        self.sectors = self._build_sectors()

    @property
    def sector_count(self) -> int:
        return len(self.sectors)

    @staticmethod
    def _orient_terrain_normal(normal: tuple[float, float, float]) -> tuple[float, float, float]:
        """Terrain is a height field, so contact normals should not push entities downward."""
        if normal[2] < 0.0:
            return (-normal[0], -normal[1], -normal[2])
        return normal

    def _contact_normal_for_model_hit(
        self,
        tri_world,
        normal_local,
        contact_point_local,
        cos_h: float,
        sin_h: float,
        rotation_matrix=None,
    ) -> tuple[tuple[float, float, float], str]:
        if self.model_contact_normal_source == "entity_radial":
            radial_local = (
                -float(contact_point_local[0]),
                -float(contact_point_local[1]),
                -float(contact_point_local[2]),
            )
            radial_world = _normalize3(
                self._local_to_world_dir(radial_local, cos_h, sin_h, rotation_matrix)
            )
            if radial_world is not None:
                return self._orient_terrain_normal(radial_world), "entity_radial"
        if self.model_contact_normal_source == "mesh":
            normal_world = self._local_to_world_dir(normal_local, cos_h, sin_h, rotation_matrix)
            return self._orient_terrain_normal(normal_world), "entity_cbsp_split"
        # The terrain-face path is a default-off rough-terrain A/B probe.
        terrain_normal = _normalize3(
            _cross3(
                _sub3(tri_world[1], tri_world[0]),
                _sub3(tri_world[2], tri_world[0]),
            )
        )
        if terrain_normal is None:
            normal_world = self._local_to_world_dir(normal_local, cos_h, sin_h)
            return self._orient_terrain_normal(normal_world), "entity_cbsp_split_fallback"
        return self._orient_terrain_normal(terrain_normal), "terrain_triangle"

    def _contact_normal_metadata_for_model_hit(
        self,
        tri_world,
        mesh_contact,
        normal_local,
        contact_point_local,
        cos_h: float,
        sin_h: float,
        rotation_matrix=None,
    ) -> dict[str, tuple[float, float, float] | None]:
        def local_normal_to_world(value, *, orient: bool = True):
            if value is None:
                return None
            try:
                if len(value) < 3:
                    return None
                local = (float(value[0]), float(value[1]), float(value[2]))
            except (TypeError, ValueError, OverflowError):
                return None
            world = self._local_to_world_dir(local, cos_h, sin_h, rotation_matrix)
            return self._orient_terrain_normal(world) if orient else world

        cbsp_split_normal = local_normal_to_world(
            getattr(mesh_contact, "cbsp_split_normal", normal_local)
        )
        terrain_face_normal = local_normal_to_world(
            getattr(mesh_contact, "terrain_face_normal", None)
        )
        if terrain_face_normal is None:
            terrain_face_normal = _normalize3(
                _cross3(
                    _sub3(tri_world[1], tri_world[0]),
                    _sub3(tri_world[2], tri_world[0]),
                )
            )
            if terrain_face_normal is not None:
                terrain_face_normal = self._orient_terrain_normal(terrain_face_normal)

        entity_radial_normal = _normalize3(
            self._local_to_world_dir(
                (
                    -float(contact_point_local[0]),
                    -float(contact_point_local[1]),
                    -float(contact_point_local[2]),
                ),
                cos_h,
                sin_h,
                rotation_matrix,
            )
        )
        if entity_radial_normal is not None:
            entity_radial_normal = self._orient_terrain_normal(entity_radial_normal)

        return {
            "cbsp_split_normal": cbsp_split_normal,
            "terrain_face_normal": terrain_face_normal,
            "mesh_face_normal": local_normal_to_world(
                getattr(mesh_contact, "mesh_face_normal", None)
            ),
            "entity_radial_normal": entity_radial_normal,
            "cbsp_store_normal0": local_normal_to_world(
                getattr(mesh_contact, "store_normal0", None),
                orient=False,
            ),
            "cbsp_store_normal1": local_normal_to_world(
                getattr(mesh_contact, "store_normal1", None),
                orient=False,
            ),
        }

    @staticmethod
    def _all_finite(values) -> bool:
        try:
            return all(math.isfinite(float(value)) for value in values)
        except (TypeError, ValueError, OverflowError):
            return False

    def coords_to_index(self, row: int, col: int) -> int:
        return row * self.sector_cols + col

    def classify_cell(self, wx: float, wy: float) -> tuple[int, int]:
        """Classify world-space XY into the top-level 3x3 terrain-sector grid."""
        if self.world_w <= 0.0:
            row = 0
        elif not math.isfinite(float(wx)):
            row = self.sector_rows - 1 if wx > 0.0 else 0
        else:
            row = int(math.floor(wx / (self.world_w / self.sector_rows)))
        if self.world_h <= 0.0:
            col = 0
        elif not math.isfinite(float(wy)):
            col = self.sector_cols - 1 if wy > 0.0 else 0
        else:
            col = int(math.floor(wy / (self.world_h / self.sector_cols)))
        return (
            _clamp(row, 0, self.sector_rows - 1),
            _clamp(col, 0, self.sector_cols - 1),
        )

    def test_sphere_collision(
        self,
        center: tuple[float, float, float],
        radius: float,
    ) -> Optional[TerrainHit]:
        """Test a sphere against sectorized terrain triangles."""
        if not self._all_finite((*center, radius)):
            return None
        aabb_min = (center[0] - radius, center[1] - radius, center[2] - radius)
        aabb_max = (center[0] + radius, center[1] + radius, center[2] + radius)

        for sector in self._iter_aabb_sectors(aabb_min, aabb_max):
            for cell_x, cell_y in self._iter_sector_cells(aabb_min, aabb_max, sector):
                for tri in self._iter_cell_triangles(cell_x, cell_y):
                    if not self._triangle_overlaps_aabb(tri, aabb_min, aabb_max):
                        continue
                    closest = _closest_point_on_triangle(center, tri[0], tri[1], tri[2])
                    delta = _sub3(center, closest)
                    if _dot3(delta, delta) < radius * radius:
                        return TerrainHit(
                            position=closest,
                            sector_index=sector.index,
                            cell=(cell_x, cell_y),
                        )
        return None

    def test_bounds_intersection(
        self,
        aabb_min: tuple[float, float, float],
        aabb_max: tuple[float, float, float],
    ) -> bool:
        """Test whether any terrain triangle overlaps XY bounds, mirroring the dirty-path broadphase."""
        if not self._all_finite((*aabb_min, *aabb_max)):
            return False
        for sector in self._iter_aabb_sectors(aabb_min, aabb_max):
            for cell_x, cell_y in self._iter_sector_cells(aabb_min, aabb_max, sector):
                for tri in self._iter_cell_triangles(cell_x, cell_y):
                    if self._triangle_overlaps_xy_bounds(
                        tri,
                        aabb_min[0],
                        aabb_min[1],
                        aabb_max[0],
                        aabb_max[1],
                    ):
                        return True
        return False

    def test_box_collision(
        self,
        center: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        heading: float,
    ) -> Optional[TerrainContact]:
        """Test a heading-rotated entity box against sectorized terrain triangles."""
        if not self._all_finite((*center, *half_extents, heading)):
            return None
        radius = math.sqrt(
            half_extents[0] * half_extents[0] +
            half_extents[1] * half_extents[1] +
            half_extents[2] * half_extents[2]
        )
        aabb_min = (center[0] - radius, center[1] - radius, center[2] - radius)
        aabb_max = (center[0] + radius, center[1] + radius, center[2] + radius)
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        for sector in self._iter_aabb_sectors(aabb_min, aabb_max):
            for cell_x, cell_y in self._iter_sector_cells(aabb_min, aabb_max, sector):
                for tri_world in self._iter_cell_triangles(cell_x, cell_y):
                    if not self._triangle_overlaps_xy_bounds(
                        tri_world,
                        aabb_min[0],
                        aabb_min[1],
                        aabb_max[0],
                        aabb_max[1],
                    ):
                        continue
                    tri_local = tuple(
                        self._world_to_local_box(vertex, center, cos_h, sin_h)
                        for vertex in tri_world
                    )
                    sat_contact = self._triangle_box_contact(tri_local, half_extents)
                    if sat_contact is None:
                        continue
                    axis_local, penetration = sat_contact
                    axis_world = self._local_to_world_dir(axis_local, cos_h, sin_h)
                    closest_local = _closest_point_on_triangle((0.0, 0.0, 0.0), *tri_local)
                    closest_world = self._local_to_world_point(closest_local, center, cos_h, sin_h)
                    contact = TerrainContact(
                        position=closest_world,
                        normal=self._orient_terrain_normal(axis_world),
                        penetration=penetration,
                        sector_index=sector.index,
                        cell=(cell_x, cell_y),
                    )
                    return contact

        return None

    def test_box_bounds_contact(
        self,
        bounds_center: tuple[float, float, float],
        collision_center: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        heading: float,
        bounding_radius: Optional[float] = None,
        rotation_matrix=None,
        contact_selection: str = "first",
    ) -> Optional[TerrainContact]:
        """Return the first terrain contact found while scanning bounds-overlapping cells."""
        finite_values = (*bounds_center, *collision_center, *half_extents, heading)
        if bounding_radius is not None:
            finite_values = (*finite_values, bounding_radius)
        if not self._all_finite(finite_values):
            return None
        model_matrix = self._coerce_rotation_matrix(rotation_matrix)
        selection = str(contact_selection or "first").strip().lower()
        collect_candidates = selection in {
            "upward",
            "upward_min_depth",
            "upward_shallow",
            "min_depth_upward",
            "shallow_upward",
            "cbsp_record_hit_strict_probe",
            "cbsp_record_hit_guess7_order_probe",
            "cbsp_node_plane_vertex_probe",
            "cbsp_node_plane_vertex_traversal_probe",
            "cbsp_mesh_edge_terrain_plane_probe",
            "cbsp_mesh_edge_terrain_plane_traversal_probe",
            "cbsp_mesh_edge_endpoint_terrain_plane_probe",
            "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
            "cbsp_mesh_vertex_probe",
            "cbsp_mesh_vertex_traversal_probe",
        }
        best_contact = None
        best_score = None
        radius = bounding_radius
        if radius is None:
            radius = math.sqrt(
                half_extents[0] * half_extents[0] +
                half_extents[1] * half_extents[1] +
                half_extents[2] * half_extents[2]
            )
        aabb_min = (
            bounds_center[0] - radius,
            bounds_center[1] - radius,
            bounds_center[2] - radius,
        )
        aabb_max = (
            bounds_center[0] + radius,
            bounds_center[1] + radius,
            bounds_center[2] + radius,
        )
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        for sector in self._iter_aabb_sectors(aabb_min, aabb_max):
            for cell_x, cell_y in self._iter_sector_cells(aabb_min, aabb_max, sector):
                for tri_world in self._iter_cell_triangles(cell_x, cell_y):
                    tri_local = tuple(
                        self._world_to_local_box(
                            vertex,
                            collision_center,
                            cos_h,
                            sin_h,
                            model_matrix,
                        )
                        for vertex in tri_world
                    )
                    sat_contact = self._triangle_box_contact(tri_local, half_extents)
                    if sat_contact is None:
                        continue
                    axis_local, penetration = sat_contact
                    axis_world = self._local_to_world_dir(axis_local, cos_h, sin_h, model_matrix)
                    closest_local = _closest_point_on_triangle((0.0, 0.0, 0.0), *tri_local)
                    closest_world = self._local_to_world_point(
                        closest_local,
                        collision_center,
                        cos_h,
                        sin_h,
                        model_matrix,
                    )
                    terrain_face_normal = _normalize3(
                        _cross3(
                            _sub3(tri_world[1], tri_world[0]),
                            _sub3(tri_world[2], tri_world[0]),
                        )
                    )
                    if terrain_face_normal is not None:
                        terrain_face_normal = self._orient_terrain_normal(terrain_face_normal)
                    entity_radial_normal = _normalize3(
                        self._local_to_world_dir(
                            (
                                -float(closest_local[0]),
                                -float(closest_local[1]),
                                -float(closest_local[2]),
                            ),
                            cos_h,
                            sin_h,
                            model_matrix,
                        )
                    )
                    if entity_radial_normal is not None:
                        entity_radial_normal = self._orient_terrain_normal(
                            entity_radial_normal
                        )
                    contact = TerrainContact(
                        position=closest_world,
                        normal=self._orient_terrain_normal(axis_world),
                        penetration=penetration,
                        sector_index=sector.index,
                        cell=(cell_x, cell_y),
                        normal_source="terrain_bounds_sat",
                        terrain_face_normal=terrain_face_normal,
                        entity_radial_normal=entity_radial_normal,
                    )
                    if collect_candidates:
                        score = self._model_contact_selection_score(contact, selection)
                        if score is not None and (best_score is None or score < best_score):
                            best_score = score
                            best_contact = contact
                        continue
                    return contact
        return best_contact

    def raycast(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> Optional[TerrainRaycastHit]:
        """Raycast a segment against terrain using patch sweep + per-patch DDA traversal."""
        if not self._all_finite((*start, *end)):
            return None
        for sector in self._iter_ray_sectors(start, end):
            hit = self._raycast_sector_cells(start, end, sector)
            if hit is not None:
                return hit
        return None

    def test_model_collision(
        self,
        center: tuple[float, float, float],
        heading: float,
        vertices,
        cbsp_tree,
        bounding_radius: float,
        rotation_matrix=None,
        contact_selection: str = "first",
    ) -> Optional[TerrainContact]:
        """Test sectorized terrain triangles against an entity collision mesh."""
        # Per-step collision wall-clock budget. The swept-TOI + multi-hypothesis
        # contact resolution issues dozens of these full mesh-vs-terrain CBSP queries
        # per physics step; on a thrust-loaded tank grinding rough terrain that costs
        # 100-1500 ms/step in CPython and collapses the 30 Hz controller cadence to
        # ~3-8 Hz (the OG client runs the same physics at ~40 Hz). Once the step's
        # deadline passes, further queries report "no contact" so the resolution
        # finishes with what it already found and the tick stays real-time; the next
        # tick re-resolves. Never triggers in normal play / unit tests (a single query
        # is sub-millisecond). Set via WULFRAM_ENTITY_TERRAIN_CONTACT_TIME_BUDGET_MS=0.
        # See docs/goal-runs/2026-05-30-controller-cadence-rootcause.md.
        _deadline = getattr(self, "_query_deadline", None)
        if _deadline is not None and time.perf_counter() > _deadline:
            self._query_deadline_hits = getattr(self, "_query_deadline_hits", 0) + 1
            return None
        if not self._all_finite((*center, heading, bounding_radius)):
            return None
        model_matrix = self._coerce_rotation_matrix(rotation_matrix)
        selection = str(contact_selection or "first").strip().lower()
        collect_candidates = selection in {
            "upward",
            "upward_min_depth",
            "upward_shallow",
            "min_depth_upward",
            "shallow_upward",
            "cbsp_mesh_edge_terrain_plane_probe",
            "cbsp_mesh_edge_terrain_plane_traversal_probe",
            "cbsp_mesh_edge_endpoint_terrain_plane_probe",
            "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
            "cbsp_node_plane_vertex_probe",
            "cbsp_node_plane_vertex_traversal_probe",
        }
        best_contact = None
        best_score = None
        aabb_min = (center[0] - bounding_radius, center[1] - bounding_radius, center[2] - bounding_radius)
        aabb_max = (center[0] + bounding_radius, center[1] + bounding_radius, center[2] + bounding_radius)
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        for sector in self._iter_aabb_sectors(aabb_min, aabb_max):
            for cell_x, cell_y in self._iter_sector_cells(aabb_min, aabb_max, sector):
                for tri_world in self._iter_cell_triangles(cell_x, cell_y):
                    if not self._triangle_overlaps_aabb(tri_world, aabb_min, aabb_max):
                        continue
                    tri_local = tuple(
                        self._world_to_local_box(vertex, center, cos_h, sin_h, model_matrix)
                        for vertex in tri_world
                    )
                    if selection in {"cbsp_mesh_vertex_probe", "cbsp_mesh_vertex_traversal_probe"}:
                        mesh_contact = self._triangle_cbsp_mesh_vertex_contact(
                            tri_local,
                            vertices,
                            cbsp_tree,
                            traversal_order=selection == "cbsp_mesh_vertex_traversal_probe",
                        )
                    elif selection in {
                        "cbsp_node_plane_vertex_probe",
                        "cbsp_node_plane_vertex_traversal_probe",
                    }:
                        mesh_contact = self._triangle_cbsp_node_plane_vertex_contact(
                            tri_local,
                            vertices,
                            cbsp_tree,
                            traversal_order=(
                                selection == "cbsp_node_plane_vertex_traversal_probe"
                            ),
                        )
                    elif selection in {
                        "cbsp_mesh_edge_terrain_plane_probe",
                        "cbsp_mesh_edge_terrain_plane_traversal_probe",
                        "cbsp_mesh_edge_endpoint_terrain_plane_probe",
                        "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
                    }:
                        mesh_contact = self._triangle_cbsp_mesh_edge_terrain_plane_contact(
                            tri_local,
                            vertices,
                            cbsp_tree,
                            traversal_order=(
                                selection
                                in {
                                    "cbsp_mesh_edge_terrain_plane_traversal_probe",
                                    "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
                                }
                            ),
                            endpoint_only=(
                                selection
                                in {
                                    "cbsp_mesh_edge_endpoint_terrain_plane_probe",
                                    "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
                                }
                            ),
                            prefer_deep_endpoint=(
                                selection
                                in {
                                    "cbsp_mesh_edge_endpoint_terrain_plane_probe",
                                    "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
                                }
                            ),
                        )
                    else:
                        if selection in {
                            "cbsp_record_hit_strict_probe",
                            "cbsp_record_hit_guess7_order_probe",
                        }:
                            mesh_contact = self._triangle_cbsp_contact(
                                tri_local,
                                vertices,
                                cbsp_tree,
                                bounding_radius,
                                include_heuristic_fallbacks=False,
                                point_inside_mode=(
                                    "guess7_order_probe"
                                    if selection == "cbsp_record_hit_guess7_order_probe"
                                    else "edge_walk"
                                ),
                            )
                        else:
                            mesh_contact = self._triangle_cbsp_contact(
                                tri_local,
                                vertices,
                                cbsp_tree,
                                bounding_radius,
                            )
                    if mesh_contact is None:
                        continue
                    contact_point_local, normal_local, penetration = mesh_contact
                    normal_world, normal_source = self._contact_normal_for_model_hit(
                        tri_world,
                        normal_local,
                        contact_point_local,
                        cos_h,
                        sin_h,
                        model_matrix,
                    )
                    normal_metadata = self._contact_normal_metadata_for_model_hit(
                        tri_world,
                        mesh_contact,
                        normal_local,
                        contact_point_local,
                        cos_h,
                        sin_h,
                        model_matrix,
                    )
                    contact_world = self._local_to_world_point(
                        contact_point_local,
                        center,
                        cos_h,
                        sin_h,
                        model_matrix,
                    )
                    contact = TerrainContact(
                        position=contact_world,
                        normal=normal_world,
                        penetration=penetration,
                        sector_index=sector.index,
                        cell=(cell_x, cell_y),
                        normal_source=normal_source,
                        cbsp_record_hit_source=getattr(
                            mesh_contact,
                            "record_hit_source",
                            None,
                        ),
                        cbsp_mesh_triangle_indices=getattr(
                            mesh_contact,
                            "mesh_triangle_indices",
                            None,
                        ),
                        cbsp_guess7_order=getattr(
                            mesh_contact,
                            "guess7_order",
                            None,
                        ),
                        cbsp_guess7_terms=getattr(
                            mesh_contact,
                            "guess7_terms",
                            None,
                        ),
                        cbsp_edge_hit_kind=getattr(
                            mesh_contact,
                            "edge_hit_kind",
                            None,
                        ),
                        cbsp_edge_t=getattr(
                            mesh_contact,
                            "edge_t",
                            None,
                        ),
                        cbsp_node_index=getattr(
                            mesh_contact,
                            "node_index",
                            None,
                        ),
                        cbsp_node_depth=getattr(
                            mesh_contact,
                            "node_depth",
                            None,
                        ),
                        cbsp_node_mesh_normal_angle_deg=getattr(
                            mesh_contact,
                            "node_mesh_normal_angle_deg",
                            None,
                        ),
                        **normal_metadata,
                    )
                    if collect_candidates:
                        score = self._model_contact_selection_score(contact, selection)
                        if score is not None and (best_score is None or score < best_score):
                            best_score = score
                            best_contact = contact
                        continue
                    return contact

        return best_contact

    def test_model_bounds_contact(
        self,
        bounds_center: tuple[float, float, float],
        collision_center: tuple[float, float, float],
        heading: float,
        vertices,
        cbsp_tree,
        bounding_radius: float,
        rotation_matrix=None,
        contact_selection: str = "first",
    ) -> Optional[TerrainContact]:
        """Return the first model/terrain contact found while scanning bounds-overlapping cells."""
        if not self._all_finite((*bounds_center, *collision_center, heading, bounding_radius)):
            return None
        model_matrix = self._coerce_rotation_matrix(rotation_matrix)
        selection = str(contact_selection or "first").strip().lower()
        collect_candidates = selection in {
            "upward",
            "upward_min_depth",
            "upward_shallow",
            "min_depth_upward",
            "shallow_upward",
            "cbsp_record_hit_strict_probe",
            "cbsp_record_hit_guess7_order_probe",
            "cbsp_mesh_edge_terrain_plane_probe",
            "cbsp_mesh_edge_terrain_plane_traversal_probe",
            "cbsp_mesh_edge_endpoint_terrain_plane_probe",
            "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
            "cbsp_node_plane_vertex_probe",
            "cbsp_node_plane_vertex_traversal_probe",
            "cbsp_mesh_vertex_probe",
            "cbsp_mesh_vertex_traversal_probe",
        }
        best_contact = None
        best_score = None
        aabb_min = (
            bounds_center[0] - bounding_radius,
            bounds_center[1] - bounding_radius,
            bounds_center[2] - bounding_radius,
        )
        aabb_max = (
            bounds_center[0] + bounding_radius,
            bounds_center[1] + bounding_radius,
            bounds_center[2] + bounding_radius,
        )
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        for sector in self._iter_aabb_sectors(aabb_min, aabb_max):
            for cell_x, cell_y in self._iter_sector_cells(aabb_min, aabb_max, sector):
                for tri_world in self._iter_cell_triangles(cell_x, cell_y):
                    tri_local = tuple(
                        self._world_to_local_box(
                            vertex,
                            collision_center,
                            cos_h,
                            sin_h,
                            model_matrix,
                        )
                        for vertex in tri_world
                    )
                    if selection in {"cbsp_mesh_vertex_probe", "cbsp_mesh_vertex_traversal_probe"}:
                        mesh_contact = self._triangle_cbsp_mesh_vertex_contact(
                            tri_local,
                            vertices,
                            cbsp_tree,
                            traversal_order=selection == "cbsp_mesh_vertex_traversal_probe",
                        )
                    elif selection in {
                        "cbsp_node_plane_vertex_probe",
                        "cbsp_node_plane_vertex_traversal_probe",
                    }:
                        mesh_contact = self._triangle_cbsp_node_plane_vertex_contact(
                            tri_local,
                            vertices,
                            cbsp_tree,
                            traversal_order=(
                                selection == "cbsp_node_plane_vertex_traversal_probe"
                            ),
                        )
                    elif selection in {
                        "cbsp_mesh_edge_terrain_plane_probe",
                        "cbsp_mesh_edge_terrain_plane_traversal_probe",
                        "cbsp_mesh_edge_endpoint_terrain_plane_probe",
                        "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
                    }:
                        mesh_contact = self._triangle_cbsp_mesh_edge_terrain_plane_contact(
                            tri_local,
                            vertices,
                            cbsp_tree,
                            traversal_order=(
                                selection
                                in {
                                    "cbsp_mesh_edge_terrain_plane_traversal_probe",
                                    "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
                                }
                            ),
                            endpoint_only=(
                                selection
                                in {
                                    "cbsp_mesh_edge_endpoint_terrain_plane_probe",
                                    "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
                                }
                            ),
                            prefer_deep_endpoint=(
                                selection
                                in {
                                    "cbsp_mesh_edge_endpoint_terrain_plane_probe",
                                    "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
                                }
                            ),
                        )
                    else:
                        if selection in {
                            "cbsp_record_hit_strict_probe",
                            "cbsp_record_hit_guess7_order_probe",
                        }:
                            mesh_contact = self._triangle_cbsp_contact(
                                tri_local,
                                vertices,
                                cbsp_tree,
                                bounding_radius,
                                include_heuristic_fallbacks=False,
                                point_inside_mode=(
                                    "guess7_order_probe"
                                    if selection == "cbsp_record_hit_guess7_order_probe"
                                    else "edge_walk"
                                ),
                            )
                        else:
                            mesh_contact = self._triangle_cbsp_contact(
                                tri_local,
                                vertices,
                                cbsp_tree,
                                bounding_radius,
                            )
                    if mesh_contact is None:
                        continue
                    contact_point_local, normal_local, penetration = mesh_contact
                    normal_world, normal_source = self._contact_normal_for_model_hit(
                        tri_world,
                        normal_local,
                        contact_point_local,
                        cos_h,
                        sin_h,
                        model_matrix,
                    )
                    normal_metadata = self._contact_normal_metadata_for_model_hit(
                        tri_world,
                        mesh_contact,
                        normal_local,
                        contact_point_local,
                        cos_h,
                        sin_h,
                        model_matrix,
                    )
                    contact_world = self._local_to_world_point(
                        contact_point_local,
                        collision_center,
                        cos_h,
                        sin_h,
                        model_matrix,
                    )
                    contact = TerrainContact(
                        position=contact_world,
                        normal=normal_world,
                        penetration=penetration,
                        sector_index=sector.index,
                        cell=(cell_x, cell_y),
                        normal_source=normal_source,
                        cbsp_record_hit_source=getattr(
                            mesh_contact,
                            "record_hit_source",
                            None,
                        ),
                        cbsp_mesh_triangle_indices=getattr(
                            mesh_contact,
                            "mesh_triangle_indices",
                            None,
                        ),
                        cbsp_guess7_order=getattr(
                            mesh_contact,
                            "guess7_order",
                            None,
                        ),
                        cbsp_guess7_terms=getattr(
                            mesh_contact,
                            "guess7_terms",
                            None,
                        ),
                        cbsp_edge_hit_kind=getattr(
                            mesh_contact,
                            "edge_hit_kind",
                            None,
                        ),
                        cbsp_edge_t=getattr(
                            mesh_contact,
                            "edge_t",
                            None,
                        ),
                        cbsp_node_index=getattr(
                            mesh_contact,
                            "node_index",
                            None,
                        ),
                        cbsp_node_depth=getattr(
                            mesh_contact,
                            "node_depth",
                            None,
                        ),
                        cbsp_node_mesh_normal_angle_deg=getattr(
                            mesh_contact,
                            "node_mesh_normal_angle_deg",
                            None,
                        ),
                        **normal_metadata,
                    )
                    if collect_candidates:
                        score = self._model_contact_selection_score(contact, selection)
                        if score is not None and (best_score is None or score < best_score):
                            best_score = score
                            best_contact = contact
                        continue
                    return contact

        return best_contact

    def _build_sectors(self) -> tuple[TerrainSector, ...]:
        row_ranges = self._split_axis(self.max_cell_x + 1, self.sector_rows)
        col_ranges = self._split_axis(self.max_cell_y + 1, self.sector_cols)
        sectors = []
        for row, (row_start, row_end) in enumerate(row_ranges):
            for col, (col_start, col_end) in enumerate(col_ranges):
                sectors.append(
                    TerrainSector(
                        row=row,
                        col=col,
                        index=self.coords_to_index(row, col),
                        row_start=row_start,
                        row_end=row_end,
                        col_start=col_start,
                        col_end=col_end,
                        origin_x=row_start * self.cell_x,
                        origin_y=col_start * self.cell_y,
                        step_x=self.cell_x,
                        step_y=self.cell_y,
                    )
                )
        return tuple(sectors)

    @staticmethod
    def _split_axis(cell_count: int, bucket_count: int) -> list[tuple[int, int]]:
        edges = [0]
        for idx in range(1, bucket_count):
            edges.append(int(round(cell_count * idx / bucket_count)))
        edges.append(cell_count)
        return [(edges[idx], edges[idx + 1]) for idx in range(bucket_count)]

    def _iter_aabb_sectors(self, aabb_min, aabb_max):
        row_min, col_min = self.classify_cell(aabb_min[0], aabb_min[1])
        row_max, col_max = self.classify_cell(aabb_max[0], aabb_max[1])
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                yield self.sectors[self.coords_to_index(row, col)]

    def _iter_ray_sectors(self, start, end):
        start_row, start_col = self.classify_cell(start[0], start[1])
        end_row, end_col = self.classify_cell(end[0], end[1])
        step_row = 1 if end_row >= start_row else -1
        step_col = 1 if end_col >= start_col else -1
        row_stop = end_row + step_row
        col_stop = end_col + step_col
        for row in range(start_row, row_stop, step_row):
            for col in range(start_col, col_stop, step_col):
                yield self.sectors[self.coords_to_index(row, col)]

    def _world_to_grid_clamped(self, wx: float, wy: float, sector: TerrainSector) -> tuple[int, int]:
        gx = int(math.floor(wx / self.cell_x)) if self.cell_x > 0.0 else 0
        gy = int(math.floor(wy / self.cell_y)) if self.cell_y > 0.0 else 0
        gx = _clamp(gx, sector.row_start, max(sector.row_end - 1, sector.row_start))
        gy = _clamp(gy, sector.col_start, max(sector.col_end - 1, sector.col_start))
        gx = _clamp(gx, 0, self.max_cell_x)
        gy = _clamp(gy, 0, self.max_cell_y)
        return gx, gy

    def _iter_sector_cells(self, aabb_min, aabb_max, sector: TerrainSector):
        row_min, col_min = self._world_to_grid_clamped(aabb_min[0], aabb_min[1], sector)
        row_max, col_max = self._world_to_grid_clamped(aabb_max[0], aabb_max[1], sector)
        for cell_x in range(row_min, row_max + 1):
            for cell_y in range(col_min, col_max + 1):
                yield cell_x, cell_y

    def _raycast_sector_cells(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        sector: TerrainSector,
    ) -> Optional[TerrainRaycastHit]:
        start_cell_x, start_cell_y = self._world_to_grid_clamped(start[0], start[1], sector)
        end_cell_x, end_cell_y = self._world_to_grid_clamped(end[0], end[1], sector)

        current_cell_x = start_cell_x
        current_cell_y = start_cell_y
        x_boundary_world = current_cell_x * self.cell_x
        y_boundary_world = current_cell_y * self.cell_y
        axis_flags = 0

        if start_cell_x < end_cell_x:
            step_x = 1
            x_boundary_world += self.cell_x
            signed_cell_x = self.cell_x
            axis_flags = 2
        else:
            step_x = -1
            signed_cell_x = -self.cell_x

        if start_cell_y < end_cell_y:
            step_y = 1
            y_boundary_world += self.cell_y
            signed_cell_y = self.cell_y
            axis_flags += 1
        else:
            step_y = -1
            signed_cell_y = -self.cell_y

        x_stop = end_cell_x + step_x
        y_stop = end_cell_y + step_y
        line_a = start[0] - end[0]
        line_b = end[1] - start[1]
        line_c = -(line_a * start[1] + start[0] * line_b)

        hit = self._raycast_cell_triangles(start, end, sector, current_cell_x, current_cell_y)
        while True:
            if hit is not None:
                return hit

            line_side = line_b * x_boundary_world + line_a * y_boundary_world + line_c
            if axis_flags in (0, 3):
                step_x_next = line_side <= 0.0
            else:
                step_x_next = line_side > 0.0

            if step_x_next:
                current_cell_x += step_x
                if current_cell_x == x_stop:
                    return None
                x_boundary_world += signed_cell_x
            else:
                current_cell_y += step_y
                if current_cell_y == y_stop:
                    return None
                y_boundary_world += signed_cell_y

            hit = self._raycast_cell_triangles(start, end, sector, current_cell_x, current_cell_y)

    def _raycast_cell_triangles(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        sector: TerrainSector,
        cell_x: int,
        cell_y: int,
    ) -> Optional[TerrainRaycastHit]:
        for tri in self._iter_cell_triangles(cell_x, cell_y):
            hit_point = self._segment_triangle_intersection(start, end, tri)
            if hit_point is None:
                continue
            normal = _normalize3(_cross3(_sub3(tri[1], tri[0]), _sub3(tri[2], tri[0])))
            if normal is None:
                continue
            normal = self._orient_terrain_normal(normal)
            delta = _sub3(hit_point, start)
            return TerrainRaycastHit(
                position=hit_point,
                normal=normal,
                sector_index=sector.index,
                cell=(cell_x, cell_y),
                distance=math.sqrt(_dot3(delta, delta)),
            )
        return None

    def _iter_cell_triangles(self, cell_x: int, cell_y: int):
        # Per-step collision wall-clock budget (see test_model_collision). This is the
        # common chokepoint every collision query iterates through, so checking the
        # deadline here bounds even a single query that straddles many terrain cells
        # (the rough-cell deep-contact case): once the deadline passes the generator
        # yields no further triangles, so the resolution finishes with what it found.
        _deadline = getattr(self, "_query_deadline", None)
        if _deadline is not None and time.perf_counter() > _deadline:
            self._query_deadline_hits = getattr(self, "_query_deadline_hits", 0) + 1
            return
        x0 = cell_x * self.cell_x
        x1 = (cell_x + 1) * self.cell_x
        y0 = cell_y * self.cell_y
        y1 = (cell_y + 1) * self.cell_y

        h00 = self.terrain._get_raw_height(cell_x, cell_y) + self.height_offset
        h10 = self.terrain._get_raw_height(cell_x + 1, cell_y) + self.height_offset
        h01 = self.terrain._get_raw_height(cell_x, cell_y + 1) + self.height_offset
        h11 = self.terrain._get_raw_height(cell_x + 1, cell_y + 1) + self.height_offset

        v00 = (x0, y0, h00)
        v10 = (x1, y0, h10)
        v01 = (x0, y1, h01)
        v11 = (x1, y1, h11)

        # Decompile parity/order from Collision_test_quad_triangles:
        #   diagonal_flag = (~col ^ row) & 1
        #   flag 0 -> (v01,v00,v10) then (v10,v11,v01)
        #   flag 1 -> (v01,v00,v11) then (v00,v10,v11)
        if ((cell_x + cell_y) & 1) == 1:
            yield (v01, v00, v10)
            yield (v10, v11, v01)
        else:
            yield (v01, v00, v11)
            yield (v00, v10, v11)

    @staticmethod
    def _coerce_rotation_matrix(rotation_matrix):
        try:
            matrix = tuple(float(v) for v in tuple(rotation_matrix or ())[:9])
        except (TypeError, ValueError, OverflowError):
            return None
        if len(matrix) != 9:
            return None
        return matrix

    @staticmethod
    def _model_contact_selection_score(contact: TerrainContact, selection: str):
        try:
            normal_z = float(contact.normal[2])
            penetration = float(contact.penetration)
        except (TypeError, ValueError, OverflowError, IndexError):
            return None
        if not math.isfinite(normal_z) or not math.isfinite(penetration):
            return None
        if "upward" in selection and normal_z < 0.5:
            return None
        if selection in {
            "cbsp_mesh_edge_terrain_plane_probe",
            "cbsp_mesh_edge_terrain_plane_traversal_probe",
            "cbsp_mesh_edge_endpoint_terrain_plane_probe",
            "cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe",
            "cbsp_mesh_vertex_probe",
            "cbsp_mesh_vertex_traversal_probe",
            "cbsp_node_plane_vertex_probe",
            "cbsp_node_plane_vertex_traversal_probe",
            "cbsp_record_hit_guess7_order_probe",
        }:
            return (penetration, -normal_z)
        if "min_depth" in selection or "shallow" in selection or selection == "upward":
            return (penetration, -normal_z)
        return (-normal_z, penetration)

    @staticmethod
    def _world_to_local_box(point, center, cos_h: float, sin_h: float, rotation_matrix=None):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        dz = point[2] - center[2]
        if rotation_matrix is not None:
            return (
                dx * rotation_matrix[0] + dy * rotation_matrix[3] + dz * rotation_matrix[6],
                dx * rotation_matrix[1] + dy * rotation_matrix[4] + dz * rotation_matrix[7],
                dx * rotation_matrix[2] + dy * rotation_matrix[5] + dz * rotation_matrix[8],
            )
        return (
            dx * cos_h + dy * sin_h,
            -dx * sin_h + dy * cos_h,
            dz,
        )

    @staticmethod
    def _local_to_world_dir(direction, cos_h: float, sin_h: float, rotation_matrix=None):
        if rotation_matrix is not None:
            return (
                direction[0] * rotation_matrix[0]
                + direction[1] * rotation_matrix[1]
                + direction[2] * rotation_matrix[2],
                direction[0] * rotation_matrix[3]
                + direction[1] * rotation_matrix[4]
                + direction[2] * rotation_matrix[5],
                direction[0] * rotation_matrix[6]
                + direction[1] * rotation_matrix[7]
                + direction[2] * rotation_matrix[8],
            )
        return (
            direction[0] * cos_h - direction[1] * sin_h,
            direction[0] * sin_h + direction[1] * cos_h,
            direction[2],
        )

    @staticmethod
    def _local_to_world_point(point, center, cos_h: float, sin_h: float, rotation_matrix=None):
        if rotation_matrix is not None:
            return (
                center[0]
                + point[0] * rotation_matrix[0]
                + point[1] * rotation_matrix[1]
                + point[2] * rotation_matrix[2],
                center[1]
                + point[0] * rotation_matrix[3]
                + point[1] * rotation_matrix[4]
                + point[2] * rotation_matrix[5],
                center[2]
                + point[0] * rotation_matrix[6]
                + point[1] * rotation_matrix[7]
                + point[2] * rotation_matrix[8],
            )
        return (
            center[0] + point[0] * cos_h - point[1] * sin_h,
            center[1] + point[0] * sin_h + point[1] * cos_h,
            center[2] + point[2],
        )

    @staticmethod
    def _box_axis_radius(axis, half_extents) -> float:
        return (
            abs(axis[0]) * half_extents[0] +
            abs(axis[1]) * half_extents[1] +
            abs(axis[2]) * half_extents[2]
        )

    def _node_mesh_triangles(self, node, vertices):
        """Precompute (and cache) the static per-CBSP-leaf-node mesh triangles with their
        face normal and plane-d. These depend only on the (static) entity collision mesh,
        but the scalar path rebuilt them on every query (cross/normalize/dot + vertex
        attribute access for every mesh triangle of every visited node). id(node) is a
        stable key because the cbsp_tree/vertices come from the cached collision model.
        Degenerate (zero-normal) triangles are skipped here exactly as the scalar leaf
        loop skipped them (they can never produce a contact). Parity-preserving."""
        cache = self._node_mesh_cache
        cached = cache.get(id(node))
        if cached is not None:
            return cached
        nverts = len(vertices)
        result = []
        for mesh_tri_indices in node.triangles:
            i0, i1, i2 = mesh_tri_indices
            if i0 >= nverts or i1 >= nverts or i2 >= nverts:
                continue
            v0 = vertices[i0]; v1 = vertices[i1]; v2 = vertices[i2]
            mesh_tri = (
                (v0.x, v0.y, v0.z),
                (v1.x, v1.y, v1.z),
                (v2.x, v2.y, v2.z),
            )
            mesh_normal_raw = _cross3(_sub3(mesh_tri[1], mesh_tri[0]), _sub3(mesh_tri[2], mesh_tri[0]))
            mesh_normal = _normalize3(mesh_normal_raw)
            if mesh_normal is None:
                continue
            mesh_plane_d = -_dot3(mesh_normal, mesh_tri[0])
            result.append((mesh_tri_indices, mesh_tri, mesh_normal, mesh_plane_d))
        cache[id(node)] = result
        return result

    def _triangle_cbsp_contact(
        self,
        tri_local,
        vertices,
        cbsp_tree,
        bounding_radius: float,
        *,
        include_heuristic_fallbacks: bool = True,
        point_inside_mode: str = "edge_walk",
    ):
        tri_normal_raw = _cross3(_sub3(tri_local[1], tri_local[0]), _sub3(tri_local[2], tri_local[0]))
        tri_normal = _normalize3(tri_normal_raw)
        if tri_normal is None or cbsp_tree is None or not getattr(cbsp_tree, "nodes", None):
            return None

        tri_center = (
            (tri_local[0][0] + tri_local[1][0] + tri_local[2][0]) / 3.0,
            (tri_local[0][1] + tri_local[1][1] + tri_local[2][1]) / 3.0,
            (tri_local[0][2] + tri_local[1][2] + tri_local[2][2]) / 3.0,
        )
        if _dot3(tri_normal, tri_center) > 0.0:
            tri_normal_raw = (-tri_normal_raw[0], -tri_normal_raw[1], -tri_normal_raw[2])
            tri_normal = (-tri_normal[0], -tri_normal[1], -tri_normal[2])

        tri_aabb_min, tri_aabb_max = self._triangle_bounds(tri_local)
        root = cbsp_tree.root
        if root is None:
            return None

        plane_d = -_dot3(tri_normal, tri_local[0])
        if plane_d > bounding_radius + 1e-4:
            return None

        tri_normal_len_sq = _dot3(tri_normal_raw, tri_normal_raw)
        if tri_normal_len_sq <= 1e-10:
            return None

        tri_edges = (
            (tri_local[0], tri_local[1]),
            (tri_local[0], tri_local[2]),
            (tri_local[1], tri_local[2]),
        )

        def record_hit(
            hit_point,
            mesh_tri,
            hit_normal,
            mesh_normal=None,
            *,
            source: str = "unknown",
            mesh_tri_indices=None,
        ):
            penetration = self._estimate_triangle_penetration(tri_local, mesh_tri, hit_normal)
            return _CBSPContact(
                position=hit_point,
                normal=hit_normal,
                penetration=penetration,
                cbsp_split_normal=hit_normal,
                terrain_face_normal=tri_normal,
                mesh_face_normal=mesh_normal,
                store_normal0=tri_normal,
                store_normal1=hit_normal,
                record_hit_source=source,
                mesh_triangle_indices=(
                    tuple(int(index) for index in mesh_tri_indices)
                    if mesh_tri_indices is not None
                    else None
                ),
            )

        _copysign = math.copysign

        def signbits_differ(value_a: float, value_b: float) -> bool:
            # Inlined _signbit (bit-identical): removes ~1.2M nested-call frames per 600
            # CBSP queries in the edge-straddle tests, keeping copysign sign-bit semantics
            # (handles -0.0) exactly as _signbit. See test_collision_parity.py.
            return (_copysign(1.0, value_a) < 0.0) != (_copysign(1.0, value_b) < 0.0)

        def edge_crosses_plane(dist_a: float, dist_b: float) -> bool:
            return signbits_differ(dist_a, dist_b)

        def lerp_clip(vert_a, vert_b, dist_a: float, dist_b: float):
            denom = dist_a - dist_b
            if abs(denom) <= 1e-8:
                return None
            lerp_t = dist_a / denom
            return (
                vert_a[0] + (vert_b[0] - vert_a[0]) * lerp_t,
                vert_a[1] + (vert_b[1] - vert_a[1]) * lerp_t,
                vert_a[2] + (vert_b[2] - vert_a[2]) * lerp_t,
            )

        def node_support_vertex(node, direction, reference_sign_value: float):
            ref_negative = _signbit(reference_sign_value)
            return (
                node.center.x + (-node.half_extent_x if _signbit(direction[0]) == ref_negative else node.half_extent_x),
                node.center.y + (-node.half_extent_y if _signbit(direction[1]) == ref_negative else node.half_extent_y),
                node.center.z + (-node.half_extent_z if _signbit(direction[2]) == ref_negative else node.half_extent_z),
            )

        def point_inside_mesh_triangle(point, mesh_tri, mesh_normal) -> bool:
            if point_inside_mode == "guess7_order_probe":
                return self._guess7_point_in_triangle_any_order(point, mesh_tri)
            return self._point_in_triangle(point, mesh_tri, mesh_normal)

        def leaf_test_triangles(node):
            node_hit_normal = _normalize3(
                (node.split_normal.x, node.split_normal.y, node.split_normal.z)
            )
            if node_hit_normal is None:
                return None
            clip_points = []
            for (vert_a, vert_b), (dist_a, dist_b) in (
                ((tri_local[0], tri_local[1]), (node_dist_a, node_dist_b)),
                ((tri_local[0], tri_local[2]), (node_dist_a, node_dist_c)),
                ((tri_local[1], tri_local[2]), (node_dist_b, node_dist_c)),
            ):
                if edge_crosses_plane(dist_a, dist_b):
                    clip_point = lerp_clip(vert_a, vert_b, dist_a, dist_b)
                    if clip_point is not None:
                        clip_points.append(clip_point)
            if len(clip_points) > 2:
                clip_points = clip_points[:2]

            for mesh_tri_indices, mesh_tri, mesh_normal, mesh_plane_d in self._node_mesh_triangles(node, vertices):
                if not self._triangle_overlaps_aabb(mesh_tri, tri_aabb_min, tri_aabb_max):
                    continue

                for clip_index, clip_point in enumerate(clip_points):
                    if point_inside_mesh_triangle(clip_point, mesh_tri, mesh_normal):
                        return record_hit(
                            clip_point,
                            mesh_tri,
                            node_hit_normal,
                            mesh_normal,
                            source=(
                                f"cbsp_leaf_clip_point_guess7_order_{clip_index}"
                                if point_inside_mode == "guess7_order_probe"
                                else
                                "cbsp_leaf_clip_point"
                                if include_heuristic_fallbacks
                                else f"cbsp_leaf_clip_point_{clip_index}"
                            ),
                            mesh_tri_indices=mesh_tri_indices,
                        )

                for edge_name, (edge_start, edge_end) in (
                    ("edge_ab", tri_edges[0]),
                    ("edge_ac", tri_edges[1]),
                    ("edge_bc", tri_edges[2]),
                ):
                    dist_start = _dot3(mesh_normal, edge_start) + mesh_plane_d
                    dist_end = _dot3(mesh_normal, edge_end) + mesh_plane_d
                    if not edge_crosses_plane(dist_start, dist_end):
                        continue
                    hit_point = lerp_clip(edge_start, edge_end, dist_start, dist_end)
                    if hit_point is None:
                        continue
                    if point_inside_mesh_triangle(hit_point, mesh_tri, mesh_normal):
                        return record_hit(
                            hit_point,
                            mesh_tri,
                            node_hit_normal,
                            mesh_normal,
                            source=(
                                f"cbsp_edge_triangle_intersect_guess7_order_{edge_name}"
                                if point_inside_mode == "guess7_order_probe"
                                else f"cbsp_edge_triangle_intersect_{edge_name}"
                            ),
                            mesh_tri_indices=mesh_tri_indices,
                        )

                if not include_heuristic_fallbacks:
                    continue

                vertex_contact = self._terrain_triangle_vertex_contact(tri_local, mesh_tri, tri_normal)
                if vertex_contact is not None:
                    hit_point, _ = vertex_contact
                    return record_hit(
                        hit_point,
                        mesh_tri,
                        node_hit_normal,
                        mesh_normal,
                        source="cbsp_vertex_contact",
                        mesh_tri_indices=mesh_tri_indices,
                    )
                hit_point = self._triangles_intersection_point(tri_local, mesh_tri)
                if hit_point is not None:
                    return record_hit(
                        hit_point,
                        mesh_tri,
                        node_hit_normal,
                        mesh_normal,
                        source="cbsp_triangle_intersection",
                        mesh_tri_indices=mesh_tri_indices,
                    )
            return None

        def traverse(node_index: int):
            nonlocal best_contact
            if node_index < 0:
                return

            node = cbsp_tree.nodes[node_index]
            center_delta = (
                node.center.x - tri_local[0][0],
                node.center.y - tri_local[0][1],
                node.center.z - tri_local[0][2],
            )
            proj_dist = _dot3(center_delta, tri_normal_raw)
            if (proj_dist * proj_dist) > (node.radius * node.radius * tri_normal_len_sq + 1e-6):
                return

            support_point = node_support_vertex(node, tri_normal_raw, proj_dist)
            support_proj = _dot3(_sub3(support_point, tri_local[0]), tri_normal_raw)
            if not signbits_differ(support_proj, proj_dist):
                return

            split_normal = (node.split_normal.x, node.split_normal.y, node.split_normal.z)
            split_len_sq = _dot3(split_normal, split_normal)
            if split_len_sq <= 1e-10:
                return

            nonlocal node_dist_a, node_dist_b, node_dist_c
            node_dist_a = _dot3(split_normal, tri_local[0]) + node.split_plane_d
            node_dist_b = _dot3(split_normal, tri_local[1]) + node.split_plane_d
            node_dist_c = _dot3(split_normal, tri_local[2]) + node.split_plane_d
            straddles = (
                edge_crosses_plane(node_dist_a, node_dist_b) or
                edge_crosses_plane(node_dist_a, node_dist_c) or
                edge_crosses_plane(node_dist_b, node_dist_c)
            )

            if straddles:
                leaf_hit = leaf_test_triangles(node)
                if leaf_hit is not None:
                    best_contact = leaf_hit
                    return
                if node.child_pos >= 0:
                    traverse(node.child_pos)
                    if best_contact is not None:
                        return
                if node.child_neg >= 0:
                    traverse(node.child_neg)
                return

            if not _signbit(node_dist_a):
                traverse(node.child_pos)
            else:
                traverse(node.child_neg)

        node_dist_a = 0.0
        node_dist_b = 0.0
        node_dist_c = 0.0
        best_contact = None
        traverse(cbsp_tree.root_index)
        return best_contact

    def _triangle_cbsp_mesh_vertex_contact(
        self,
        tri_local,
        vertices,
        cbsp_tree,
        *,
        traversal_order: bool = False,
    ):
        """Report-only probe for CBSP mesh vertices embedded in a terrain triangle."""
        tri_normal_raw = _cross3(_sub3(tri_local[1], tri_local[0]), _sub3(tri_local[2], tri_local[0]))
        tri_normal = _normalize3(tri_normal_raw)
        if tri_normal is None or cbsp_tree is None or not getattr(cbsp_tree, "nodes", None):
            return None

        tri_center = (
            (tri_local[0][0] + tri_local[1][0] + tri_local[2][0]) / 3.0,
            (tri_local[0][1] + tri_local[1][1] + tri_local[2][1]) / 3.0,
            (tri_local[0][2] + tri_local[1][2] + tri_local[2][2]) / 3.0,
        )
        if _dot3(tri_normal, tri_center) > 0.0:
            tri_normal_raw = (-tri_normal_raw[0], -tri_normal_raw[1], -tri_normal_raw[2])
            tri_normal = (-tri_normal[0], -tri_normal[1], -tri_normal[2])

        tri_aabb_min, tri_aabb_max = self._triangle_bounds(tri_local)

        def make_contact(vertex, mesh_tri, mesh_normal, mesh_tri_indices, signed_plane_distance):
            return _CBSPContact(
                position=vertex,
                normal=mesh_normal,
                penetration=max(0.01, abs(signed_plane_distance)),
                cbsp_split_normal=mesh_normal,
                terrain_face_normal=tri_normal,
                mesh_face_normal=mesh_normal,
                store_normal0=tri_normal,
                store_normal1=mesh_normal,
                record_hit_source=(
                    "cbsp_mesh_vertex_inside_terrain_traversal_probe"
                    if traversal_order
                    else "cbsp_mesh_vertex_inside_terrain_probe"
                ),
                mesh_triangle_indices=tuple(int(index) for index in mesh_tri_indices),
            )

        def first_contact_in_node(node):
            for mesh_tri_indices in getattr(node, "triangles", None) or ():
                i0, i1, i2 = mesh_tri_indices
                if i0 >= len(vertices) or i1 >= len(vertices) or i2 >= len(vertices):
                    continue
                mesh_tri = (
                    (vertices[i0].x, vertices[i0].y, vertices[i0].z),
                    (vertices[i1].x, vertices[i1].y, vertices[i1].z),
                    (vertices[i2].x, vertices[i2].y, vertices[i2].z),
                )
                if not self._triangle_overlaps_aabb(mesh_tri, tri_aabb_min, tri_aabb_max):
                    continue
                mesh_normal = _normalize3(_cross3(_sub3(mesh_tri[1], mesh_tri[0]), _sub3(mesh_tri[2], mesh_tri[0])))
                if mesh_normal is None:
                    continue
                for vertex in mesh_tri:
                    if not self._point_in_triangle(vertex, tri_local, tri_normal, eps=1e-4):
                        continue
                    signed_plane_distance = _dot3(tri_normal, _sub3(vertex, tri_local[0]))
                    return make_contact(
                        vertex,
                        mesh_tri,
                        mesh_normal,
                        mesh_tri_indices,
                        signed_plane_distance,
                    )
            return None

        if traversal_order:
            tri_normal_len_sq = _dot3(tri_normal_raw, tri_normal_raw)
            if tri_normal_len_sq <= 1e-10:
                return None

            def signbits_differ(value_a: float, value_b: float) -> bool:
                return _signbit(value_a) != _signbit(value_b)

            def node_support_vertex(node, direction, reference_sign_value: float):
                ref_negative = _signbit(reference_sign_value)
                return (
                    node.center.x + (-node.half_extent_x if _signbit(direction[0]) == ref_negative else node.half_extent_x),
                    node.center.y + (-node.half_extent_y if _signbit(direction[1]) == ref_negative else node.half_extent_y),
                    node.center.z + (-node.half_extent_z if _signbit(direction[2]) == ref_negative else node.half_extent_z),
                )

            def traverse(node_index: int):
                if node_index < 0:
                    return None
                node = cbsp_tree.nodes[node_index]
                center_delta = (
                    node.center.x - tri_local[0][0],
                    node.center.y - tri_local[0][1],
                    node.center.z - tri_local[0][2],
                )
                proj_dist = _dot3(center_delta, tri_normal_raw)
                if (proj_dist * proj_dist) > (node.radius * node.radius * tri_normal_len_sq + 1e-6):
                    return None
                support_point = node_support_vertex(node, tri_normal_raw, proj_dist)
                support_proj = _dot3(_sub3(support_point, tri_local[0]), tri_normal_raw)
                if not signbits_differ(support_proj, proj_dist):
                    return None

                split_normal = (node.split_normal.x, node.split_normal.y, node.split_normal.z)
                if _dot3(split_normal, split_normal) <= 1e-10:
                    return None
                dist_a = _dot3(split_normal, tri_local[0]) + node.split_plane_d
                dist_b = _dot3(split_normal, tri_local[1]) + node.split_plane_d
                dist_c = _dot3(split_normal, tri_local[2]) + node.split_plane_d
                straddles = (
                    signbits_differ(dist_a, dist_b) or
                    signbits_differ(dist_a, dist_c) or
                    signbits_differ(dist_b, dist_c)
                )
                if straddles:
                    leaf_hit = first_contact_in_node(node)
                    if leaf_hit is not None:
                        return leaf_hit
                    return traverse(node.child_pos) or traverse(node.child_neg)
                if not _signbit(dist_a):
                    return traverse(node.child_pos)
                return traverse(node.child_neg)

            return traverse(cbsp_tree.root_index)

        best_contact = None
        best_score = None
        for node_index, node in enumerate(cbsp_tree.nodes):
            for triangle_order, mesh_tri_indices in enumerate(getattr(node, "triangles", None) or ()):
                i0, i1, i2 = mesh_tri_indices
                if i0 >= len(vertices) or i1 >= len(vertices) or i2 >= len(vertices):
                    continue
                mesh_tri = (
                    (vertices[i0].x, vertices[i0].y, vertices[i0].z),
                    (vertices[i1].x, vertices[i1].y, vertices[i1].z),
                    (vertices[i2].x, vertices[i2].y, vertices[i2].z),
                )
                if not self._triangle_overlaps_aabb(mesh_tri, tri_aabb_min, tri_aabb_max):
                    continue
                mesh_normal = _normalize3(_cross3(_sub3(mesh_tri[1], mesh_tri[0]), _sub3(mesh_tri[2], mesh_tri[0])))
                if mesh_normal is None:
                    continue
                for vertex_order, vertex in enumerate(mesh_tri):
                    if not self._point_in_triangle(vertex, tri_local, tri_normal, eps=1e-4):
                        continue
                    signed_plane_distance = _dot3(tri_normal, _sub3(vertex, tri_local[0]))
                    score = (
                        abs(signed_plane_distance),
                        node_index,
                        triangle_order,
                        vertex_order,
                    )
                    if best_score is not None and score >= best_score:
                        continue
                    best_score = score
                    best_contact = make_contact(
                        vertex,
                        mesh_tri,
                        mesh_normal,
                        mesh_tri_indices,
                        signed_plane_distance,
                    )
        return best_contact

    def _triangle_cbsp_mesh_edge_terrain_plane_contact(
        self,
        tri_local,
        vertices,
        cbsp_tree,
        *,
        traversal_order: bool = False,
        endpoint_only: bool = False,
        prefer_deep_endpoint: bool = False,
    ):
        """Report-only probe for mesh edges crossing the terrain triangle plane."""
        tri_normal_raw = _cross3(_sub3(tri_local[1], tri_local[0]), _sub3(tri_local[2], tri_local[0]))
        tri_normal = _normalize3(tri_normal_raw)
        if tri_normal is None or cbsp_tree is None or not getattr(cbsp_tree, "nodes", None):
            return None

        tri_center = (
            (tri_local[0][0] + tri_local[1][0] + tri_local[2][0]) / 3.0,
            (tri_local[0][1] + tri_local[1][1] + tri_local[2][1]) / 3.0,
            (tri_local[0][2] + tri_local[1][2] + tri_local[2][2]) / 3.0,
        )
        if _dot3(tri_normal, tri_center) > 0.0:
            tri_normal_raw = (-tri_normal_raw[0], -tri_normal_raw[1], -tri_normal_raw[2])
            tri_normal = (-tri_normal[0], -tri_normal[1], -tri_normal[2])

        tri_aabb_min, tri_aabb_max = self._triangle_bounds(tri_local)

        def signbits_differ(value_a: float, value_b: float) -> bool:
            return _signbit(value_a) != _signbit(value_b)

        def node_depths():
            depths = {}
            stack = [(getattr(cbsp_tree, "root_index", 0), 0)]
            while stack:
                node_index, depth = stack.pop()
                if node_index < 0 or node_index in depths or node_index >= len(cbsp_tree.nodes):
                    continue
                depths[node_index] = depth
                node = cbsp_tree.nodes[node_index]
                stack.append((getattr(node, "child_neg", -1), depth + 1))
                stack.append((getattr(node, "child_pos", -1), depth + 1))
            return depths

        depth_by_node = node_depths() if prefer_deep_endpoint else {}

        def make_contact(
            point,
            node_normal,
            mesh_tri,
            mesh_normal,
            mesh_tri_indices,
            edge_name: str,
            *,
            guess7_terms=None,
            edge_hit_kind: str = "crossing",
            edge_t: float | None = None,
            node_index: int | None = None,
            node_depth: int | None = None,
            node_mesh_normal_angle_deg: float | None = None,
        ):
            return _CBSPContact(
                position=point,
                normal=node_normal,
                penetration=0.01,
                cbsp_split_normal=node_normal,
                terrain_face_normal=tri_normal,
                mesh_face_normal=mesh_normal,
                store_normal0=tri_normal,
                store_normal1=node_normal,
                record_hit_source=(
                    (
                        f"cbsp_mesh_edge_endpoint_terrain_plane_traversal_probe_{edge_name}"
                        if traversal_order
                        else f"cbsp_mesh_edge_endpoint_terrain_plane_probe_{edge_name}"
                    )
                    if endpoint_only
                    else (
                        f"cbsp_mesh_edge_terrain_plane_traversal_probe_{edge_name}"
                        if traversal_order
                        else f"cbsp_mesh_edge_terrain_plane_probe_{edge_name}"
                    )
                ),
                mesh_triangle_indices=tuple(int(index) for index in mesh_tri_indices),
                guess7_order=(1, 0, 2) if guess7_terms is not None else None,
                guess7_terms=(
                    tuple(float(term) for term in guess7_terms)
                    if guess7_terms is not None
                    else None
                ),
                edge_hit_kind=edge_hit_kind,
                edge_t=edge_t,
                node_index=node_index,
                node_depth=node_depth,
                node_mesh_normal_angle_deg=node_mesh_normal_angle_deg,
            )

        def edge_plane_hit_point(start, end, dist_start: float, dist_end: float):
            endpoint_epsilon = 1e-2 if endpoint_only else 1e-3
            if abs(dist_start) <= endpoint_epsilon:
                return "start_on_terrain_plane", 0.0, start
            if abs(dist_end) <= endpoint_epsilon:
                return "end_on_terrain_plane", 1.0, end
            if endpoint_only or not signbits_differ(dist_start, dist_end):
                return None
            denom = dist_start - dist_end
            if abs(denom) <= 1e-8:
                return None
            t = dist_start / denom
            if t < -1e-6 or t > 1.0 + 1e-6:
                return None
            point = (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
                start[2] + (end[2] - start[2]) * t,
            )
            return "crossing", t, point

        def first_contact_in_node(node, node_index: int | None = None):
            node_normal = _normalize3(
                (node.split_normal.x, node.split_normal.y, node.split_normal.z)
            )
            if node_normal is None:
                return None
            node_depth = (
                depth_by_node.get(node_index)
                if node_index is not None
                else None
            )
            for mesh_tri_indices in getattr(node, "triangles", None) or ():
                i0, i1, i2 = mesh_tri_indices
                if i0 >= len(vertices) or i1 >= len(vertices) or i2 >= len(vertices):
                    continue
                mesh_tri = (
                    (vertices[i0].x, vertices[i0].y, vertices[i0].z),
                    (vertices[i1].x, vertices[i1].y, vertices[i1].z),
                    (vertices[i2].x, vertices[i2].y, vertices[i2].z),
                )
                if not self._triangle_overlaps_aabb(mesh_tri, tri_aabb_min, tri_aabb_max):
                    continue
                mesh_normal = _normalize3(_cross3(_sub3(mesh_tri[1], mesh_tri[0]), _sub3(mesh_tri[2], mesh_tri[0])))
                if mesh_normal is None:
                    continue
                node_mesh_normal_angle_deg = _angle_between3(node_normal, mesh_normal)
                terrain_distances = [
                    _dot3(tri_normal, _sub3(vertex, tri_local[0]))
                    for vertex in mesh_tri
                ]
                for edge_name, start_idx, end_idx in (
                    ("ab", 0, 1),
                    ("bc", 1, 2),
                    ("ca", 2, 0),
                ):
                    dist_start = terrain_distances[start_idx]
                    dist_end = terrain_distances[end_idx]
                    start = mesh_tri[start_idx]
                    end = mesh_tri[end_idx]
                    edge_hit = edge_plane_hit_point(
                        start,
                        end,
                        dist_start,
                        dist_end,
                    )
                    if edge_hit is None:
                        continue
                    edge_hit_kind, edge_t, point = edge_hit
                    guess7_terms = self._guess7_point_in_triangle_edge_intersect_terms(
                        point,
                        tri_local,
                        tri_normal,
                    )
                    if not all(term >= 0.0 for term in guess7_terms):
                        continue
                    return make_contact(
                        point,
                        node_normal,
                        mesh_tri,
                        mesh_normal,
                        mesh_tri_indices,
                        edge_name,
                        guess7_terms=guess7_terms,
                        edge_hit_kind=edge_hit_kind,
                        edge_t=edge_t,
                        node_index=node_index,
                        node_depth=node_depth,
                        node_mesh_normal_angle_deg=node_mesh_normal_angle_deg,
                    )
            return None

        if traversal_order:
            tri_normal_len_sq = _dot3(tri_normal_raw, tri_normal_raw)
            if tri_normal_len_sq <= 1e-10:
                return None

            def node_support_vertex(node, direction, reference_sign_value: float):
                ref_negative = _signbit(reference_sign_value)
                return (
                    node.center.x + (-node.half_extent_x if _signbit(direction[0]) == ref_negative else node.half_extent_x),
                    node.center.y + (-node.half_extent_y if _signbit(direction[1]) == ref_negative else node.half_extent_y),
                    node.center.z + (-node.half_extent_z if _signbit(direction[2]) == ref_negative else node.half_extent_z),
                )

            def traverse(node_index: int):
                if node_index < 0:
                    return None
                node = cbsp_tree.nodes[node_index]
                center_delta = (
                    node.center.x - tri_local[0][0],
                    node.center.y - tri_local[0][1],
                    node.center.z - tri_local[0][2],
                )
                proj_dist = _dot3(center_delta, tri_normal_raw)
                if (proj_dist * proj_dist) > (node.radius * node.radius * tri_normal_len_sq + 1e-6):
                    return None
                support_point = node_support_vertex(node, tri_normal_raw, proj_dist)
                support_proj = _dot3(_sub3(support_point, tri_local[0]), tri_normal_raw)
                if not signbits_differ(support_proj, proj_dist):
                    return None

                split_normal = (node.split_normal.x, node.split_normal.y, node.split_normal.z)
                if _dot3(split_normal, split_normal) <= 1e-10:
                    return None
                dist_a = _dot3(split_normal, tri_local[0]) + node.split_plane_d
                dist_b = _dot3(split_normal, tri_local[1]) + node.split_plane_d
                dist_c = _dot3(split_normal, tri_local[2]) + node.split_plane_d
                straddles = (
                    signbits_differ(dist_a, dist_b) or
                    signbits_differ(dist_a, dist_c) or
                    signbits_differ(dist_b, dist_c)
                )
                if straddles:
                    leaf_hit = first_contact_in_node(node, node_index)
                    if leaf_hit is not None:
                        return leaf_hit
                    return traverse(node.child_pos) or traverse(node.child_neg)
                if not _signbit(dist_a):
                    return traverse(node.child_pos)
                return traverse(node.child_neg)

            return traverse(cbsp_tree.root_index)

        best_contact = None
        best_score = None
        for node_index, node in enumerate(cbsp_tree.nodes):
            for triangle_order, mesh_tri_indices in enumerate(getattr(node, "triangles", None) or ()):
                contact = first_contact_in_node(
                    type(
                        "_SingleTriangleNode",
                        (),
                        {
                            "triangles": (mesh_tri_indices,),
                            "split_normal": node.split_normal,
                        },
                    )(),
                    node_index,
                )
                if contact is None:
                    continue
                score = (
                    (
                        float(getattr(contact, "node_mesh_normal_angle_deg", None))
                        if getattr(contact, "node_mesh_normal_angle_deg", None) is not None
                        else 999.0
                    ),
                    -int(getattr(contact, "node_depth", None) or 0),
                    node_index,
                    triangle_order,
                ) if endpoint_only and prefer_deep_endpoint else (
                    node_index,
                    triangle_order,
                )
                if best_score is not None and score >= best_score:
                    continue
                best_score = score
                best_contact = contact
        return best_contact

    def _triangle_cbsp_node_plane_vertex_contact(
        self,
        tri_local,
        vertices,
        cbsp_tree,
        *,
        traversal_order: bool = False,
    ):
        """Report-only probe for near-plane mesh vertices using the CBSP node normal."""
        tri_normal_raw = _cross3(_sub3(tri_local[1], tri_local[0]), _sub3(tri_local[2], tri_local[0]))
        tri_normal = _normalize3(tri_normal_raw)
        if tri_normal is None or cbsp_tree is None or not getattr(cbsp_tree, "nodes", None):
            return None

        tri_center = (
            (tri_local[0][0] + tri_local[1][0] + tri_local[2][0]) / 3.0,
            (tri_local[0][1] + tri_local[1][1] + tri_local[2][1]) / 3.0,
            (tri_local[0][2] + tri_local[1][2] + tri_local[2][2]) / 3.0,
        )
        if _dot3(tri_normal, tri_center) > 0.0:
            tri_normal_raw = (-tri_normal_raw[0], -tri_normal_raw[1], -tri_normal_raw[2])
            tri_normal = (-tri_normal[0], -tri_normal[1], -tri_normal[2])

        tri_aabb_min, tri_aabb_max = self._triangle_bounds(tri_local)

        def make_contact(vertex, node_normal, mesh_tri_indices, signed_plane_distance):
            projected = (
                vertex[0] - tri_normal[0] * signed_plane_distance,
                vertex[1] - tri_normal[1] * signed_plane_distance,
                vertex[2] - tri_normal[2] * signed_plane_distance,
            )
            return _CBSPContact(
                position=projected,
                normal=node_normal,
                penetration=max(0.01, abs(signed_plane_distance)),
                cbsp_split_normal=node_normal,
                terrain_face_normal=tri_normal,
                mesh_face_normal=node_normal,
                store_normal0=tri_normal,
                store_normal1=node_normal,
                record_hit_source=(
                    "cbsp_node_plane_vertex_traversal_probe"
                    if traversal_order
                    else "cbsp_node_plane_vertex_probe"
                ),
                mesh_triangle_indices=tuple(int(index) for index in mesh_tri_indices),
            )

        def first_contact_in_node(node):
            node_normal = _normalize3(
                (node.split_normal.x, node.split_normal.y, node.split_normal.z)
            )
            if node_normal is None:
                return None
            best_contact = None
            best_score = None
            for triangle_order, mesh_tri_indices in enumerate(getattr(node, "triangles", None) or ()):
                i0, i1, i2 = mesh_tri_indices
                if i0 >= len(vertices) or i1 >= len(vertices) or i2 >= len(vertices):
                    continue
                mesh_tri = (
                    (vertices[i0].x, vertices[i0].y, vertices[i0].z),
                    (vertices[i1].x, vertices[i1].y, vertices[i1].z),
                    (vertices[i2].x, vertices[i2].y, vertices[i2].z),
                )
                if not self._triangle_overlaps_aabb(mesh_tri, tri_aabb_min, tri_aabb_max):
                    continue
                for vertex_order, vertex in enumerate(mesh_tri):
                    if not self._point_in_triangle(vertex, tri_local, tri_normal, eps=1e-4):
                        continue
                    node_plane_distance = abs(
                        _dot3(node_normal, vertex) + node.split_plane_d
                    )
                    if node_plane_distance > 1e-3:
                        continue
                    signed_plane_distance = _dot3(tri_normal, _sub3(vertex, tri_local[0]))
                    score = (
                        abs(signed_plane_distance),
                        triangle_order,
                        vertex_order,
                    )
                    if best_score is not None and score >= best_score:
                        continue
                    best_score = score
                    best_contact = make_contact(
                        vertex,
                        node_normal,
                        mesh_tri_indices,
                        signed_plane_distance,
                    )
            return best_contact

        if traversal_order:
            tri_normal_len_sq = _dot3(tri_normal_raw, tri_normal_raw)
            if tri_normal_len_sq <= 1e-10:
                return None

            def signbits_differ(value_a: float, value_b: float) -> bool:
                return _signbit(value_a) != _signbit(value_b)

            def node_support_vertex(node, direction, reference_sign_value: float):
                ref_negative = _signbit(reference_sign_value)
                return (
                    node.center.x + (-node.half_extent_x if _signbit(direction[0]) == ref_negative else node.half_extent_x),
                    node.center.y + (-node.half_extent_y if _signbit(direction[1]) == ref_negative else node.half_extent_y),
                    node.center.z + (-node.half_extent_z if _signbit(direction[2]) == ref_negative else node.half_extent_z),
                )

            def traverse(node_index: int):
                if node_index < 0:
                    return None
                node = cbsp_tree.nodes[node_index]
                center_delta = (
                    node.center.x - tri_local[0][0],
                    node.center.y - tri_local[0][1],
                    node.center.z - tri_local[0][2],
                )
                proj_dist = _dot3(center_delta, tri_normal_raw)
                if (proj_dist * proj_dist) > (node.radius * node.radius * tri_normal_len_sq + 1e-6):
                    return None
                support_point = node_support_vertex(node, tri_normal_raw, proj_dist)
                support_proj = _dot3(_sub3(support_point, tri_local[0]), tri_normal_raw)
                if not signbits_differ(support_proj, proj_dist):
                    return None

                split_normal = (node.split_normal.x, node.split_normal.y, node.split_normal.z)
                if _dot3(split_normal, split_normal) <= 1e-10:
                    return None
                dist_a = _dot3(split_normal, tri_local[0]) + node.split_plane_d
                dist_b = _dot3(split_normal, tri_local[1]) + node.split_plane_d
                dist_c = _dot3(split_normal, tri_local[2]) + node.split_plane_d
                straddles = (
                    signbits_differ(dist_a, dist_b) or
                    signbits_differ(dist_a, dist_c) or
                    signbits_differ(dist_b, dist_c)
                )
                if straddles:
                    leaf_hit = first_contact_in_node(node)
                    if leaf_hit is not None:
                        return leaf_hit
                    return traverse(node.child_pos) or traverse(node.child_neg)
                if not _signbit(dist_a):
                    return traverse(node.child_pos)
                return traverse(node.child_neg)

            return traverse(cbsp_tree.root_index)

        best_contact = None
        best_score = None
        for node_index, node in enumerate(cbsp_tree.nodes):
            contact = first_contact_in_node(node)
            if contact is None:
                continue
            score = (
                float(contact.penetration),
                node_index,
            )
            if best_score is not None and score >= best_score:
                continue
            best_score = score
            best_contact = contact
        return best_contact

    def _triangle_box_contact(self, tri_local, half_extents):
        v0, v1, v2 = tri_local
        e0 = _sub3(v1, v0)
        e1 = _sub3(v2, v1)
        e2 = _sub3(v0, v2)
        tri_normal = _cross3(e0, _sub3(v2, v0))
        tri_center = (
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        )

        axes = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            tri_normal,
            _cross3(e0, (1.0, 0.0, 0.0)),
            _cross3(e0, (0.0, 1.0, 0.0)),
            _cross3(e0, (0.0, 0.0, 1.0)),
            _cross3(e1, (1.0, 0.0, 0.0)),
            _cross3(e1, (0.0, 1.0, 0.0)),
            _cross3(e1, (0.0, 0.0, 1.0)),
            _cross3(e2, (1.0, 0.0, 0.0)),
            _cross3(e2, (0.0, 1.0, 0.0)),
            _cross3(e2, (0.0, 0.0, 1.0)),
        ]

        best_axis = None
        best_overlap = None
        for axis in axes:
            axis_len = _length3(axis)
            if axis_len <= 1e-8:
                continue
            tri_proj = (_dot3(axis, v0), _dot3(axis, v1), _dot3(axis, v2))
            tri_min = min(tri_proj)
            tri_max = max(tri_proj)
            box_radius = self._box_axis_radius(axis, half_extents)
            if tri_min > box_radius or tri_max < -box_radius:
                return None

            overlap = min(box_radius - tri_min, tri_max + box_radius) / axis_len
            axis_dir = (axis[0] / axis_len, axis[1] / axis_len, axis[2] / axis_len)
            if _dot3(axis_dir, tri_center) > 0.0:
                axis_dir = (-axis_dir[0], -axis_dir[1], -axis_dir[2])
            if best_overlap is None or overlap < best_overlap:
                best_overlap = overlap
                best_axis = axis_dir

        if best_axis is None or best_overlap is None:
            return None
        return best_axis, best_overlap

    @staticmethod
    def _triangle_bounds(tri):
        mins = (
            min(tri[0][0], tri[1][0], tri[2][0]),
            min(tri[0][1], tri[1][1], tri[2][1]),
            min(tri[0][2], tri[1][2], tri[2][2]),
        )
        maxs = (
            max(tri[0][0], tri[1][0], tri[2][0]),
            max(tri[0][1], tri[1][1], tri[2][1]),
            max(tri[0][2], tri[1][2], tri[2][2]),
        )
        return mins, maxs

    @staticmethod
    def _segment_triangle_intersection(p0, p1, tri, eps: float = 1e-6):
        # Inlined Moeller-Trumbore (bit-identical to the _sub3/_cross3/_dot3 form):
        # eliminates ~9 scalar-helper calls/invocation (this is ~3-4 of every collision
        # query in the rough-cell hot path). See test_collision_parity.py.
        t0 = tri[0]
        t0x = t0[0]; t0y = t0[1]; t0z = t0[2]
        t1 = tri[1]; t2 = tri[2]
        e1x = t1[0] - t0x; e1y = t1[1] - t0y; e1z = t1[2] - t0z
        e2x = t2[0] - t0x; e2y = t2[1] - t0y; e2z = t2[2] - t0z
        p0x = p0[0]; p0y = p0[1]; p0z = p0[2]
        dx = p1[0] - p0x; dy = p1[1] - p0y; dz = p1[2] - p0z
        # h = cross(direction, edge2)
        hx = dy * e2z - dz * e2y
        hy = dz * e2x - dx * e2z
        hz = dx * e2y - dy * e2x
        det = e1x * hx + e1y * hy + e1z * hz
        if -eps < det < eps:
            return None
        inv_det = 1.0 / det
        sx = p0x - t0x; sy = p0y - t0y; sz = p0z - t0z
        u = inv_det * (sx * hx + sy * hy + sz * hz)
        if u < -eps or u > 1.0 + eps:
            return None
        # q = cross(s, edge1)
        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x
        v = inv_det * (dx * qx + dy * qy + dz * qz)
        if v < -eps or (u + v) > 1.0 + eps:
            return None
        t = inv_det * (e2x * qx + e2y * qy + e2z * qz)
        if t < -eps or t > 1.0 + eps:
            return None
        return (p0x + dx * t, p0y + dy * t, p0z + dz * t)

    @staticmethod
    def _point_plane_distance(point, tri, normal) -> float:
        return _dot3(normal, _sub3(point, tri[0]))

    @staticmethod
    def _point_in_triangle(point, tri, normal, eps: float = 1e-5) -> bool:
        # Inlined (bit-identical to the _sub3/_cross3/_dot3 form): for each edge,
        # dot(normal, cross(edge, point-start)) must be >= -eps. ~12 scalar-helper
        # calls/invocation removed; this is one of the hottest leaves. See
        # test_collision_parity.py.
        nx = normal[0]; ny = normal[1]; nz = normal[2]
        px = point[0]; py = point[1]; pz = point[2]
        neg_eps = -eps
        for idx in range(3):
            start = tri[idx]
            end = tri[(idx + 1) % 3]
            sx = start[0]; sy = start[1]; sz = start[2]
            edx = end[0] - sx; edy = end[1] - sy; edz = end[2] - sz
            tpx = px - sx; tpy = py - sy; tpz = pz - sz
            # cross(edge, to_point)
            cx = edy * tpz - edz * tpy
            cy = edz * tpx - edx * tpz
            cz = edx * tpy - edy * tpx
            if nx * cx + ny * cy + nz * cz < neg_eps:
                return False
        return True

    @staticmethod
    def _guess7_point_in_triangle_terms(point, tri, normal):
        vertex_a, vertex_b, reference_vertex = tri
        return (
            _dot3(
                normal,
                _cross3(_sub3(vertex_a, vertex_b), _sub3(point, vertex_b)),
            ),
            _dot3(
                normal,
                _cross3(_sub3(reference_vertex, vertex_a), _sub3(point, vertex_a)),
            ),
            _dot3(
                normal,
                _cross3(_sub3(vertex_b, reference_vertex), _sub3(point, reference_vertex)),
            ),
        )

    @classmethod
    def _guess7_point_in_triangle_any_order(cls, point, tri, normal=None) -> bool:
        for order in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ):
            ordered = (tri[order[0]], tri[order[1]], tri[order[2]])
            order_normal = normal or _normalize3(
                _cross3(_sub3(ordered[1], ordered[0]), _sub3(ordered[2], ordered[0]))
            )
            if order_normal is None:
                continue
            terms = cls._guess7_point_in_triangle_terms(point, ordered, order_normal)
            if terms[0] >= 0.0 and terms[1] >= 0.0 and terms[2] >= 0.0:
                return True
        return False

    @classmethod
    def _guess7_point_in_triangle_edge_intersect_order(cls, point, tri, normal) -> bool:
        terms = cls._guess7_point_in_triangle_edge_intersect_terms(point, tri, normal)
        return terms[0] >= 0.0 and terms[1] >= 0.0 and terms[2] >= 0.0

    @classmethod
    def _guess7_point_in_triangle_edge_intersect_terms(cls, point, tri, normal):
        ordered = (tri[1], tri[0], tri[2])
        return cls._guess7_point_in_triangle_terms(point, ordered, normal)

    def _triangles_intersection_point(self, tri_a, tri_b):
        # Parity-preserving separating-plane early-reject. If every vertex of one triangle
        # is strictly on one side of the OTHER triangle's plane, no edge of either triangle
        # can cross the other (so all 6 segment-triangle tests below would miss). In that
        # case skip the 6 segment tests — but still run the coplanar vertex checks exactly
        # as before, so the result is identical to the original for every pair (separated
        # pairs returned via the segments-miss + coplanar path anyway). This skips the
        # expensive segment sweep for the many near-but-not-crossing pairs in deep contact.
        # Strict (> 0 / < 0) so a vertex on the plane never triggers a skip. The two face
        # normals are computed up front here instead of only in the coplanar tail.
        # See test_collision_parity.py.
        normal_b = _normalize3(_cross3(_sub3(tri_b[1], tri_b[0]), _sub3(tri_b[2], tri_b[0])))
        normal_a = _normalize3(_cross3(_sub3(tri_a[1], tri_a[0]), _sub3(tri_a[2], tri_a[0])))
        separated = False
        if normal_b is not None:
            db = -_dot3(normal_b, tri_b[0])
            a0 = _dot3(normal_b, tri_a[0]) + db
            a1 = _dot3(normal_b, tri_a[1]) + db
            a2 = _dot3(normal_b, tri_a[2]) + db
            if (a0 > 0.0 and a1 > 0.0 and a2 > 0.0) or (a0 < 0.0 and a1 < 0.0 and a2 < 0.0):
                separated = True
        if not separated and normal_a is not None:
            da = -_dot3(normal_a, tri_a[0])
            b0 = _dot3(normal_a, tri_b[0]) + da
            b1 = _dot3(normal_a, tri_b[1]) + da
            b2 = _dot3(normal_a, tri_b[2]) + da
            if (b0 > 0.0 and b1 > 0.0 and b2 > 0.0) or (b0 < 0.0 and b1 < 0.0 and b2 < 0.0):
                separated = True

        if not separated:
            for idx in range(3):
                hit = self._segment_triangle_intersection(tri_a[idx], tri_a[(idx + 1) % 3], tri_b)
                if hit is not None:
                    return hit
            for idx in range(3):
                hit = self._segment_triangle_intersection(tri_b[idx], tri_b[(idx + 1) % 3], tri_a)
                if hit is not None:
                    return hit

        if normal_b is not None and abs(self._point_plane_distance(tri_a[0], tri_b, normal_b)) <= 1e-5:
            if self._point_in_triangle(tri_a[0], tri_b, normal_b):
                return tri_a[0]
        if normal_a is not None and abs(self._point_plane_distance(tri_b[0], tri_a, normal_a)) <= 1e-5:
            if self._point_in_triangle(tri_b[0], tri_a, normal_a):
                return tri_b[0]
        return None

    def _terrain_triangle_vertex_contact(self, terrain_tri, mesh_tri, terrain_normal):
        # Inlined _dot3/_sub3 (bit-identical). Runs per mesh triangle on the steep-cell
        # fallback path. See test_collision_parity.py.
        p0 = terrain_tri[0]
        p0x = p0[0]; p0y = p0[1]; p0z = p0[2]
        nx = terrain_normal[0]; ny = terrain_normal[1]; nz = terrain_normal[2]
        for vertex in mesh_tri:
            vx = vertex[0]; vy = vertex[1]; vz = vertex[2]
            penetration = -(nx * (vx - p0x) + ny * (vy - p0y) + nz * (vz - p0z))
            if penetration <= 0.0:
                continue
            projected = (
                vx + nx * penetration,
                vy + ny * penetration,
                vz + nz * penetration,
            )
            if self._point_in_triangle(projected, terrain_tri, terrain_normal):
                return projected, penetration
        return None

    @staticmethod
    def _estimate_triangle_penetration(tri_a, tri_b, normal) -> float:
        plane_vertex = tri_a[0]
        depths = []
        for vertex in tri_b:
            depth = -_dot3(normal, _sub3(vertex, plane_vertex))
            if depth > 0.0:
                depths.append(depth)
        if depths:
            return max(depths)

        mesh_normal = _normalize3(_cross3(_sub3(tri_b[1], tri_b[0]), _sub3(tri_b[2], tri_b[0])))
        if mesh_normal is not None:
            plane_vertex = tri_b[0]
            for vertex in tri_a:
                depth = _dot3(mesh_normal, _sub3(vertex, plane_vertex))
                if depth < 0.0:
                    depths.append(-depth)
        if depths:
            return max(depths)
        return 0.01

    @staticmethod
    def _triangle_overlaps_aabb(tri, aabb_min, aabb_max) -> bool:
        for axis in range(3):
            tri_min = min(tri[0][axis], tri[1][axis], tri[2][axis])
            tri_max = max(tri[0][axis], tri[1][axis], tri[2][axis])
            if tri_max < aabb_min[axis] or tri_min > aabb_max[axis]:
                return False
        return True

    @staticmethod
    def _triangle_overlaps_xy_bounds(
        tri,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> bool:
        tri_min_x = min(tri[0][0], tri[1][0], tri[2][0])
        tri_max_x = max(tri[0][0], tri[1][0], tri[2][0])
        tri_min_y = min(tri[0][1], tri[1][1], tri[2][1])
        tri_max_y = max(tri[0][1], tri[1][1], tri[2][1])
        return not (
            tri_max_x < min_x or
            tri_min_x > max_x or
            tri_max_y < min_y or
            tri_min_y > max_y
        )
