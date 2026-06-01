"""Terrain heightmap loader for Wulfram server.

Loads the game's 'land' file format and provides height/slope queries
for terrain-aware physics. Matches the client's alternating triangle-plane
height query (GUESS3_Terrain_interpolate_grid_height in World.c).

File format (from GUESS3_Terrain_load_heightmap_file, World.c:9002):
  Line 1: "129x129" (grid dimensions: num_x x num_z)
  Line 2: "5600.000000x5600.000000" (world dimensions)
  Lines 3+: "texture_id height" per vertex, sequential:
    outer loop z=0..num_z-1, inner loop x=0..num_x-1
  Storage order matches client: heights[x][z]

Coordinate mapping (client Y-up -> server Z-up):
  Server X -> Client X (heightmap grid x-axis)
  Server Y -> Client Z (heightmap grid z-axis)
  Server Z -> Client Y (height/vertical)
"""

import math


class Terrain:
    """Heightmap terrain with decompile-matched triangle height queries."""

    def __init__(self, land_file_path: str):
        with open(land_file_path, "r") as f:
            lines = f.readlines()

        # Header line 1: grid dimensions "NxN"
        dims = lines[0].strip().split("x")
        self.num_x = int(dims[0])
        self.num_z = int(dims[1])

        # Header line 2: world dimensions "WxW"
        world = lines[1].strip().split("x")
        self.world_w = float(world[0])
        self.world_h = float(world[1])

        # Cell sizes (128 cells between 129 vertices)
        self.cell_x = self.world_w / (self.num_x - 1)
        self.cell_z = self.world_h / (self.num_z - 1)
        self.inv_cell_x = 1.0 / self.cell_x
        self.inv_cell_z = 1.0 / self.cell_z

        # Parse heights and cell types into flat arrays: [x * num_z + z]
        # File order: outer loop z, inner loop x (matching client loader)
        total = self.num_x * self.num_z
        self._heights = [0.0] * total
        self._cell_types = [0] * total  # per-vertex texture ID from land file

        line_idx = 2
        for gz in range(self.num_z):
            for gx in range(self.num_x):
                if line_idx < len(lines):
                    parts = lines[line_idx].strip().split()
                    if len(parts) >= 2:
                        self._cell_types[gx * self.num_z + gz] = int(parts[0])
                        self._heights[gx * self.num_z + gz] = float(parts[1])
                line_idx += 1

        h_min = min(self._heights)
        h_max = max(self._heights)
        print(
            f"[TERRAIN] Loaded {land_file_path}: "
            f"{self.num_x}x{self.num_z} grid, "
            f"{self.world_w}x{self.world_h} world units, "
            f"cell={self.cell_x:.1f}u, "
            f"height range [{h_min:.1f}, {h_max:.1f}]"
        )

    def _get_raw_height(self, gx: int, gz: int) -> float:
        """Get height at integer grid coordinates (clamped)."""
        gx = max(0, min(gx, self.num_x - 1))
        gz = max(0, min(gz, self.num_z - 1))
        return self._heights[gx * self.num_z + gz]

    def get_cell_type(self, gx: int, gz: int) -> int:
        """Get cell texture type at grid coords (clamped)."""
        gx = max(0, min(gx, self.num_x - 1))
        gz = max(0, min(gz, self.num_z - 1))
        return self._cell_types[gx * self.num_z + gz]

    @staticmethod
    def _height_on_triangle(
        wx: float,
        wy: float,
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        c: tuple[float, float, float],
    ) -> tuple[float, tuple[float, float, float]]:
        """Return the plane height and positive-Z normal for one terrain triangle."""
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        nx = ab[1] * ac[2] - ab[2] * ac[1]
        ny = ab[2] * ac[0] - ab[0] * ac[2]
        nz = ab[0] * ac[1] - ab[1] * ac[0]
        if nz < 0.0:
            nx, ny, nz = -nx, -ny, -nz
        mag_sq = nx * nx + ny * ny + nz * nz
        if mag_sq <= 1e-12 or abs(nz) <= 1e-12:
            return a[2], (0.0, 0.0, 1.0)

        # Plane equation through a: n dot ((x, y, z) - a) = 0.
        height = a[2] - (nx * (wx - a[0]) + ny * (wy - a[1])) / nz
        inv_mag = 1.0 / math.sqrt(mag_sq)
        return height, (nx * inv_mag, ny * inv_mag, nz * inv_mag)

    def sample_height_normal(self, wx: float, wy: float) -> tuple[float, tuple[float, float, float]]:
        """Get raw terrain height and normal at server coords (wx=X, wy=Y).

        `GUESS3_Terrain_interpolate_grid_height` picks one of two triangles per
        cell using `((~col ^ row) & 1)`, then computes height from that triangle
        plane. This intentionally differs from bilinear interpolation on rough
        cells and keeps spring/clamp sampling aligned with terrain collision.
        """
        if not (math.isfinite(wx) and math.isfinite(wy)):
            # TOTAL PRIMITIVE GUARD: never let a non-finite sample coordinate reach
            # int(math.floor(NaN)) below and crash a caller's thread (tick loop,
            # UDP loop, control plane). Return flat ground + up normal. Fires only
            # on already-broken (non-finite) input. (A3 soak, 2026-06-01.)
            return 0.0, (0.0, 0.0, 1.0)
        gx = int(math.floor(wx * self.inv_cell_x))
        gz = int(math.floor(wy * self.inv_cell_z))
        gx = max(0, min(gx, self.num_x - 2))
        gz = max(0, min(gz, self.num_z - 2))

        x0 = gx * self.cell_x
        x1 = (gx + 1) * self.cell_x
        y0 = gz * self.cell_z
        y1 = (gz + 1) * self.cell_z

        h00 = self._heights[gx * self.num_z + gz]
        h10 = self._heights[(gx + 1) * self.num_z + gz]
        h01 = self._heights[gx * self.num_z + gz + 1]
        h11 = self._heights[(gx + 1) * self.num_z + gz + 1]

        v00 = (x0, y0, h00)
        v10 = (x1, y0, h10)
        v01 = (x0, y1, h01)
        v11 = (x1, y1, h11)

        if ((~gx ^ gz) & 1) == 0:
            diagonal_test = (wy - y0) - (x1 - wx)
            if diagonal_test < 0.0:
                return self._height_on_triangle(wx, wy, v01, v00, v10)
            return self._height_on_triangle(wx, wy, v10, v11, v01)

        diagonal_test = (wy - y0) - (wx - x0)
        if diagonal_test < 0.0:
            return self._height_on_triangle(wx, wy, v00, v10, v11)
        return self._height_on_triangle(wx, wy, v01, v00, v11)

    def get_height(self, wx: float, wy: float) -> float:
        """Get raw terrain height at server coords (wx=X, wy=Y)."""
        height, _normal = self.sample_height_normal(wx, wy)
        return height

    def get_slope(self, wx: float, wy: float) -> tuple:
        """Get terrain slope at server coords from the active terrain triangle.

        Returns (dh_dx, dh_dy): height gradient in server X and Y directions.
        Units: rise per unit of run (dimensionless).
        """
        _height, normal = self.sample_height_normal(wx, wy)
        if abs(normal[2]) <= 1e-12:
            return (0.0, 0.0)
        return (-normal[0] / normal[2], -normal[1] / normal[2])

    def get_pitch_at_heading(self, wx: float, wy: float, heading: float) -> float:
        """Get terrain pitch angle (radians) along a heading direction.

        heading: server yaw in radians (0 = +X, positive = CCW)
        Returns pitch in radians (positive = uphill).
        """
        dh_dx, dh_dy = self.get_slope(wx, wy)
        # Project slope onto heading direction
        slope_along_heading = dh_dx * math.cos(heading) + dh_dy * math.sin(heading)
        return math.atan(slope_along_heading)
