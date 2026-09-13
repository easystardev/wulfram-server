import math
import os
import unittest
from unittest.mock import patch

from wulfram.weapons import _projectile_rotation_from_velocity


def _normalized(values):
    length = math.sqrt(sum(value * value for value in values))
    return tuple(value / length for value in values)


def _model_forward(rotation, up_axis):
    _, pitch, yaw = rotation
    if up_axis == "z":
        return (
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
        )
    return (
        math.cos(pitch) * math.cos(yaw),
        -math.sin(pitch),
        math.cos(pitch) * math.sin(yaw),
    )


class ProjectileOrientationTests(unittest.TestCase):
    def test_rotation_model_forward_matches_velocity(self):
        cases = {
            "z": (
                (75.0, 0.0, -2.0),
                (0.0, 75.0, 12.0),
                (-75.0, 0.0, -18.0),
                (0.0, -75.0, 9.0),
                (0.0, 0.0, -75.0),
            ),
            "y": (
                (75.0, -2.0, 0.0),
                (0.0, 12.0, 75.0),
                (-75.0, -18.0, 0.0),
                (0.0, 9.0, -75.0),
                (0.0, -75.0, 0.0),
            ),
        }
        for up_axis, velocities in cases.items():
            with self.subTest(up_axis=up_axis), patch.dict(
                os.environ, {"WULFRAM_UP_AXIS": up_axis}, clear=False
            ):
                for velocity in velocities:
                    rotation = _projectile_rotation_from_velocity(velocity)
                    expected = _normalized(velocity)
                    actual = _model_forward(rotation, up_axis)
                    for got, want in zip(actual, expected):
                        self.assertAlmostEqual(got, want, places=12)


if __name__ == "__main__":
    unittest.main()
