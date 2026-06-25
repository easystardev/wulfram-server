"""RaycastMixin -- static-world raycast + building-collision geometry, extracted
verbatim from WulframServer (server.py decomposition, step 2).

Method-only mixin: shares all state via `self`. The quadtree node dataclass and
the traversal sentinel are module-level. Imports only stdlib (no dependency on
server.py) so there is no import cycle.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

from .weapons import EntityType


@dataclass(frozen=True)
class _StaticWorldRayNode:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    children: Optional[tuple[Optional["_StaticWorldRayNode"], Optional["_StaticWorldRayNode"], Optional["_StaticWorldRayNode"], Optional["_StaticWorldRayNode"]]]
    building_ids: tuple[int, ...]


_STATIC_WORLD_RAY_STOP = object()


class RaycastMixin:
    def _building_has_mesh_collision(self, building) -> bool:
        if not self._building_collision.available:
            return False
        return self._building_collision.has_collision_model(
            int(building.entity_type),
            int(getattr(building, "team_id", 1)),
        )

    def _building_blocks_vehicle_collision(self, building) -> bool:
        """Return whether a map building should block vehicle movement.

        Repair pads are spawn/service pads. Treating their mesh/AABB as a
        solid blocker makes the authoritative tank shove sideways immediately
        after a map-flag spawn, while the OG client drives across the pad.
        """
        if int(getattr(building, "entity_type", -1)) == int(EntityType.REPAIR_BUILDING):
            return (
                os.environ.get("WULFRAM_REPAIR_PAD_BLOCKS_VEHICLES", "0")
                .strip()
                .lower()
                in ("1", "true", "on", "yes")
            )
        return True

    def _get_building_world_half_extents(self, building) -> tuple[float, float, float]:
        hx, hy = self._BUILDING_HALF_EXTENTS.get(building.entity_type, (8.0, 8.0))
        hz = max(hx, hy, self._BUILDING_HALF_HEIGHT)
        if not self._building_has_mesh_collision(building):
            return (hx, hy, hz)

        model_extents = self._building_collision.get_model_half_extents(
            int(building.entity_type),
            int(getattr(building, "team_id", 1)),
        )
        if model_extents is None:
            return (hx, hy, hz)

        local_hx, local_hy, local_hz = model_extents
        heading = float(getattr(building, "heading", 0.0))
        cos_h = abs(math.cos(heading))
        sin_h = abs(math.sin(heading))
        world_hx = local_hx * cos_h + local_hy * sin_h
        world_hy = local_hx * sin_h + local_hy * cos_h
        return (world_hx, world_hy, local_hz)

    def _get_building_quadtree_radius(self, building) -> float:
        if self._building_has_mesh_collision(building):
            radius = self._building_collision.get_model_bounding_radius(
                int(building.entity_type),
                int(getattr(building, "team_id", 1)),
            )
            if radius is not None:
                return radius
        hx, hy, hz = self._get_building_world_half_extents(building)
        return math.sqrt(hx * hx + hy * hy + hz * hz)

    def _rebuild_static_world_raycast_index(self) -> None:
        building_entities = getattr(self, "_building_entities", {}) or {}
        if not building_entities:
            self._static_world_raycast_root = None
            return

        bounds = []
        for eid, building in building_entities.items():
            radius = self._get_building_quadtree_radius(building)
            bounds.append((eid, building.x - radius, building.x + radius, building.y - radius, building.y + radius))

        min_x = min(item[1] for item in bounds)
        max_x = max(item[2] for item in bounds)
        min_y = min(item[3] for item in bounds)
        max_y = max(item[4] for item in bounds)
        if min_x == max_x:
            max_x = min_x + 1.0
        if min_y == max_y:
            max_y = min_y + 1.0

        building_ids = tuple(building_entities.keys())
        self._static_world_raycast_root = self._build_static_world_ray_node(
            building_ids,
            min_x,
            max_x,
            min_y,
            max_y,
            depth=0,
        )

    def _build_static_world_ray_node(
        self,
        building_ids: tuple[int, ...],
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        *,
        depth: int,
    ) -> _StaticWorldRayNode:
        max_leaf_size = 8
        max_depth = 4
        if len(building_ids) <= max_leaf_size or depth >= max_depth:
            return _StaticWorldRayNode(min_x, max_x, min_y, max_y, None, building_ids)

        mid_x = (min_x + max_x) * 0.5
        mid_y = (min_y + max_y) * 0.5
        child_bounds = (
            (mid_x, max_x, mid_y, max_y),
            (mid_x, max_x, min_y, mid_y),
            (min_x, mid_x, mid_y, max_y),
            (min_x, mid_x, min_y, mid_y),
        )
        buckets = [[], [], [], []]
        for eid in building_ids:
            building = self._building_entities.get(eid)
            if building is None:
                continue
            radius = self._get_building_quadtree_radius(building)
            west = (building.x - radius) < mid_x
            east = (building.x + radius) > mid_x
            north = (building.y - radius) < mid_y
            south = (building.y + radius) > mid_y
            if west and north:
                buckets[2].append(eid)
            if west and south:
                buckets[3].append(eid)
            if east and north:
                buckets[0].append(eid)
            if east and south:
                buckets[1].append(eid)

        non_empty = [bucket for bucket in buckets if bucket]
        if len(non_empty) <= 1:
            return _StaticWorldRayNode(min_x, max_x, min_y, max_y, None, building_ids)
        parent_ids = set(building_ids)
        if all(set(bucket) == parent_ids for bucket in non_empty):
            return _StaticWorldRayNode(min_x, max_x, min_y, max_y, None, building_ids)

        children = []
        for quadrant, bucket in enumerate(buckets):
            if not bucket:
                children.append(None)
                continue
            child_min_x, child_max_x, child_min_y, child_max_y = child_bounds[quadrant]
            children.append(
                self._build_static_world_ray_node(
                    tuple(bucket),
                    child_min_x,
                    child_max_x,
                    child_min_y,
                    child_max_y,
                    depth=depth + 1,
                )
            )
        return _StaticWorldRayNode(min_x, max_x, min_y, max_y, tuple(children), ())

    @staticmethod
    def _xy_outside_code(point: tuple[float, float, float], node: _StaticWorldRayNode) -> int:
        code = 0
        if point[0] < node.min_x:
            code |= 1
        if point[1] < node.min_y:
            code |= 2
        if point[0] > node.max_x:
            code |= 4
        if point[1] > node.max_y:
            code |= 8
        return code

    @staticmethod
    def _ray_misses_static_world_node(
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
        node: _StaticWorldRayNode,
    ) -> bool:
        endpoint_code = RaycastMixin._xy_outside_code(end_pos, node)
        if endpoint_code & RaycastMixin._xy_outside_code(start_pos, node):
            return True

        line_a = start_pos[0] - end_pos[0]
        line_b = end_pos[1] - start_pos[1]
        line_c = -(line_a * start_pos[1] + start_pos[0] * line_b)
        corners = (
            line_a * node.min_y + line_b * node.min_x + line_c,
            line_a * node.max_y + line_b * node.min_x + line_c,
            line_a * node.min_y + line_b * node.max_x + line_c,
            line_a * node.max_y + line_b * node.max_x + line_c,
        )
        return (
            all(value <= 0.0 for value in corners) or
            all(value >= 0.0 for value in corners)
        )

    @staticmethod
    def _static_world_origin_quadrant(point: tuple[float, float, float], node: _StaticWorldRayNode) -> int:
        quadrant = 0
        mid_x = (node.min_x + node.max_x) * 0.5
        mid_y = (node.min_y + node.max_y) * 0.5
        if point[1] < mid_y:
            quadrant |= 1
        if point[0] < mid_x:
            quadrant |= 2
        return quadrant

    @staticmethod
    def _iter_static_world_quadrants(origin_quadrant: int) -> tuple[int, int, int, int]:
        return (
            origin_quadrant,
            origin_quadrant ^ 0x1,
            origin_quadrant ^ 0x2,
            origin_quadrant ^ 0x3,
        )

    def _point_hits_static_building(
        self,
        building,
        point: tuple[float, float, float],
    ) -> Optional[tuple[str, tuple[float, float, float], float]]:
        half_extents = self._get_building_world_half_extents(building)
        aabb_min = (
            building.x - half_extents[0],
            building.y - half_extents[1],
            building.z - half_extents[2],
        )
        aabb_max = (
            building.x + half_extents[0],
            building.y + half_extents[1],
            building.z + half_extents[2],
        )
        if any(point[idx] < aabb_min[idx] or point[idx] > aabb_max[idx] for idx in range(3)):
            return None

        if self._building_has_mesh_collision(building):
            depth, _ = self._building_collision.test_sphere_collision(building, point, 1e-4)
            if depth <= 0.0:
                return None
            return ("building", point, 0.0)
        return ("building-aabb", point, 0.0)

    def _raycast_static_building_candidate(
        self,
        building,
        eid: int,
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
        *,
        seg_len: float,
    ) -> Optional[tuple[str, tuple[float, float, float], int, float]]:
        direction = (
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1],
            end_pos[2] - start_pos[2],
        )
        if self._building_has_mesh_collision(building):
            bounding_radius = self._building_collision.get_model_bounding_radius(
                building.entity_type,
                getattr(building, "team_id", 1),
            )
            if bounding_radius is None:
                half_extents = self._get_building_world_half_extents(building)
                bounding_radius = math.sqrt(
                    half_extents[0] * half_extents[0] +
                    half_extents[1] * half_extents[1] +
                    half_extents[2] * half_extents[2]
                )
            hit_t = self._segment_sphere_hit_t(
                start_pos,
                end_pos,
                (building.x, building.y, building.z),
                bounding_radius,
            )
            if hit_t is None:
                return None
            hit_position = (
                start_pos[0] + direction[0] * hit_t,
                start_pos[1] + direction[1] * hit_t,
                start_pos[2] + direction[2] * hit_t,
            )
            distance = seg_len * hit_t
            raycast_fn = getattr(self._building_collision, "raycast_segment_collision", None)
            if callable(raycast_fn):
                mesh_hit = raycast_fn(building, start_pos, end_pos)
                if mesh_hit is None:
                    return None
                hit_position, _, distance = mesh_hit
            elif not self._building_collision.test_segment_collision(building, start_pos, end_pos):
                return None
            return ("building", hit_position, eid, distance)

        half_extents = self._get_building_world_half_extents(building)
        aabb_min = (
            building.x - half_extents[0],
            building.y - half_extents[1],
            building.z - half_extents[2],
        )
        aabb_max = (
            building.x + half_extents[0],
            building.y + half_extents[1],
            building.z + half_extents[2],
        )
        hit_t = self._segment_aabb_hit_t(start_pos, end_pos, aabb_min, aabb_max)
        if hit_t is None:
            return None

        hit_position = (
            start_pos[0] + direction[0] * hit_t,
            start_pos[1] + direction[1] * hit_t,
            start_pos[2] + direction[2] * hit_t,
        )
        distance = seg_len * hit_t

        return ("building-aabb", hit_position, eid, distance)

    def _raycast_static_world_leaf(
        self,
        node: _StaticWorldRayNode,
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
    ) -> Optional[tuple[str, tuple[float, float, float], int, float]]:
        direction = (
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1],
            end_pos[2] - start_pos[2],
        )
        seg_len = math.sqrt(
            direction[0] * direction[0] +
            direction[1] * direction[1] +
            direction[2] * direction[2]
        )
        best_hit = None
        best_distance = None
        for eid in node.building_ids:
            building = self._building_entities.get(eid)
            if building is None:
                continue
            hit = self._raycast_static_building_candidate(building, eid, start_pos, end_pos, seg_len=seg_len)
            if hit is None:
                continue
            if best_distance is None or hit[3] < best_distance:
                best_distance = hit[3]
                best_hit = hit
        return best_hit

    def _point_query_static_world(
        self,
        start_pos: tuple[float, float, float],
        node: _StaticWorldRayNode,
    ) -> Optional[tuple[str, tuple[float, float, float], int, float]]:
        current = node
        while current.children is not None:
            quadrant = self._static_world_origin_quadrant(start_pos, current)
            child = current.children[quadrant]
            if child is None:
                break
            current = child

        def _dist_sq(eid: int) -> float:
            building = self._building_entities.get(eid)
            if building is None:
                return float("inf")
            hx, hy, _ = self._get_building_world_half_extents(building)
            dx = 0.0
            if start_pos[0] < building.x - hx:
                dx = (building.x - hx) - start_pos[0]
            elif start_pos[0] > building.x + hx:
                dx = start_pos[0] - (building.x + hx)
            dy = 0.0
            if start_pos[1] < building.y - hy:
                dy = (building.y - hy) - start_pos[1]
            elif start_pos[1] > building.y + hy:
                dy = start_pos[1] - (building.y + hy)
            return dx * dx + dy * dy

        for eid in sorted(current.building_ids, key=_dist_sq):
            building = self._building_entities.get(eid)
            if building is None:
                continue
            point_hit = self._point_hits_static_building(building, start_pos)
            if point_hit is not None:
                hit_kind, hit_position, distance = point_hit
                return (hit_kind, hit_position, eid, distance)
        return None

    @staticmethod
    def _segment_aabb_hit_t(
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
        aabb_min: tuple[float, float, float],
        aabb_max: tuple[float, float, float],
    ) -> Optional[float]:
        direction = (
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1],
            end_pos[2] - start_pos[2],
        )
        t_min = 0.0
        t_max = 1.0
        for axis in range(3):
            origin = start_pos[axis]
            delta = direction[axis]
            axis_min = aabb_min[axis]
            axis_max = aabb_max[axis]
            if abs(delta) <= 1e-8:
                if origin < axis_min or origin > axis_max:
                    return None
                continue
            inv_delta = 1.0 / delta
            t1 = (axis_min - origin) * inv_delta
            t2 = (axis_max - origin) * inv_delta
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2
            if t_min > t_max:
                return None
        return max(0.0, min(1.0, t_min))

    @staticmethod
    def _segment_sphere_hit_t(
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
        sphere_center: tuple[float, float, float],
        sphere_radius: float,
    ) -> Optional[float]:
        direction = (
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1],
            end_pos[2] - start_pos[2],
        )
        origin_to_center = (
            start_pos[0] - sphere_center[0],
            start_pos[1] - sphere_center[1],
            start_pos[2] - sphere_center[2],
        )
        a = (
            direction[0] * direction[0] +
            direction[1] * direction[1] +
            direction[2] * direction[2]
        )
        if a <= 1e-12:
            center_dist_sq = (
                origin_to_center[0] * origin_to_center[0] +
                origin_to_center[1] * origin_to_center[1] +
                origin_to_center[2] * origin_to_center[2]
            )
            return 0.0 if center_dist_sq <= sphere_radius * sphere_radius else None

        b = 2.0 * (
            direction[0] * origin_to_center[0] +
            direction[1] * origin_to_center[1] +
            direction[2] * origin_to_center[2]
        )
        c = (
            origin_to_center[0] * origin_to_center[0] +
            origin_to_center[1] * origin_to_center[1] +
            origin_to_center[2] * origin_to_center[2] -
            sphere_radius * sphere_radius
        )
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t0 = (-b - sqrt_disc) / (2.0 * a)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        if 0.0 <= t0 <= 1.0:
            return t0
        if 0.0 <= t1 <= 1.0:
            return t1
        if c <= 0.0:
            return 0.0
        return None

    def _raycast_static_buildings(
        self,
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
    ) -> Optional[tuple[str, tuple[float, float, float], int, float]]:
        root = getattr(self, "_static_world_raycast_root", None)
        if root is None and getattr(self, "_building_entities", None):
            self._rebuild_static_world_raycast_index()
            root = getattr(self, "_static_world_raycast_root", None)
        if root is None:
            return None

        if abs(end_pos[0] - start_pos[0]) <= 1e-8 and abs(end_pos[1] - start_pos[1]) <= 1e-8:
            return self._point_query_static_world(start_pos, root)

        def traverse(node: _StaticWorldRayNode):
            endpoint_code = self._xy_outside_code(end_pos, node)
            if endpoint_code & self._xy_outside_code(start_pos, node):
                return None
            if self._ray_misses_static_world_node(start_pos, end_pos, node):
                return None
            if node.children is None:
                leaf_hit = self._raycast_static_world_leaf(node, start_pos, end_pos)
                if leaf_hit is not None:
                    return leaf_hit
                if endpoint_code == 0:
                    return _STATIC_WORLD_RAY_STOP
                return None

            origin_quadrant = self._static_world_origin_quadrant(start_pos, node)
            for quadrant in self._iter_static_world_quadrants(origin_quadrant):
                child = node.children[quadrant]
                if child is None:
                    continue
                child_hit = traverse(child)
                if child_hit is _STATIC_WORLD_RAY_STOP:
                    return _STATIC_WORLD_RAY_STOP
                if child_hit is not None:
                    return child_hit
            return None

        hit = traverse(root)
        if hit is _STATIC_WORLD_RAY_STOP:
            return None
        return hit

    def _raycast_world(
        self,
        start_pos: tuple[float, float, float],
        end_pos: tuple[float, float, float],
    ) -> Optional[tuple[str, tuple[float, float, float], Optional[int]]]:
        terrain_hit = None
        terrain_dist_sq = None
        clipped_end = end_pos
        if self._terrain_grid_collision is not None:
            terrain_hit = self._terrain_grid_collision.raycast(start_pos, end_pos)
            if terrain_hit is not None:
                clipped_end = terrain_hit.position
                terrain_dist_sq = (
                    (terrain_hit.position[0] - start_pos[0]) * (terrain_hit.position[0] - start_pos[0]) +
                    (terrain_hit.position[1] - start_pos[1]) * (terrain_hit.position[1] - start_pos[1]) +
                    (terrain_hit.position[2] - start_pos[2]) * (terrain_hit.position[2] - start_pos[2])
                )

        building_hit = self._raycast_static_buildings(start_pos, clipped_end)
        if building_hit is not None:
            hit_kind, hit_position, hit_id, hit_distance = building_hit
            building_dist_sq = hit_distance * hit_distance
            if terrain_dist_sq is None or building_dist_sq <= terrain_dist_sq:
                return (hit_kind, hit_position, hit_id)

        if terrain_hit is not None:
            return ("terrain", terrain_hit.position, terrain_hit.sector_index)
        return None
