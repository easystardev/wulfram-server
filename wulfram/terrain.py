"""Terrain heightmap loader for Wulfram server.

Loads the game's 'land' file format and provides height/slope queries
for terrain-aware physics. Matches the client's bilinear interpolation
(HeightmapGrid_sample_bilinear in World.c:8744).

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
    """Heightmap terrain with bilinear height interpolation and slope queries."""

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

        # Parse heights into flat array: heights[x * num_z + z]
        # File order: outer loop z, inner loop x (matching client loader)
        total = self.num_x * self.num_z
        self._heights = [0.0] * total

        line_idx = 2
        for gz in range(self.num_z):
            for gx in range(self.num_x):
                if line_idx < len(lines):
                    parts = lines[line_idx].strip().split()
                    if len(parts) >= 2:
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

    def get_height(self, wx: float, wy: float) -> float:
        """Get terrain height at server coords (wx=X, wy=Y) via bilinear interpolation.

        Matches client's HeightmapGrid_sample_bilinear (World.c:8744).
        Returns raw heightmap value (add offset externally for server Z).
        """
        # Convert world coords to fractional grid indices
        fx = wx * self.inv_cell_x
        fy = wy * self.inv_cell_z

        # Integer grid cell
        gx = int(math.floor(fx))
        gz = int(math.floor(fy))

        # Fractional part within cell
        tx = fx - gx
        tz = fy - gz

        # Clamp grid indices
        gx = max(0, min(gx, self.num_x - 2))
        gz = max(0, min(gz, self.num_z - 2))

        # Four corner heights
        h00 = self._heights[gx * self.num_z + gz]
        h10 = self._heights[(gx + 1) * self.num_z + gz]
        h01 = self._heights[gx * self.num_z + gz + 1]
        h11 = self._heights[(gx + 1) * self.num_z + gz + 1]

        # Bilinear interpolation (matching client order: lerp X first, then Z)
        h_bottom = h00 + tx * (h10 - h00)  # lerp along X at z=gz
        h_top = h01 + tx * (h11 - h01)     # lerp along X at z=gz+1
        return h_bottom + tz * (h_top - h_bottom)  # lerp along Z

    def get_slope(self, wx: float, wy: float) -> tuple:
        """Get terrain slope at server coords via central differences.

        Returns (dh_dx, dh_dy): height gradient in server X and Y directions.
        Units: rise per unit of run (dimensionless).
        """
        # Use half-cell offsets for central differences
        dx = self.cell_x * 0.5
        dy = self.cell_z * 0.5

        h_xp = self.get_height(wx + dx, wy)
        h_xn = self.get_height(wx - dx, wy)
        h_yp = self.get_height(wx, wy + dy)
        h_yn = self.get_height(wx, wy - dy)

        dh_dx = (h_xp - h_xn) / (2.0 * dx)
        dh_dy = (h_yp - h_yn) / (2.0 * dy)
        return (dh_dx, dh_dy)

    def get_pitch_at_heading(self, wx: float, wy: float, heading: float) -> float:
        """Get terrain pitch angle (radians) along a heading direction.

        heading: server yaw in radians (0 = +X, positive = CCW)
        Returns pitch in radians (positive = uphill).
        """
        dh_dx, dh_dy = self.get_slope(wx, wy)
        # Project slope onto heading direction
        slope_along_heading = dh_dx * math.cos(heading) + dh_dy * math.sin(heading)
        return math.atan(slope_along_heading)
