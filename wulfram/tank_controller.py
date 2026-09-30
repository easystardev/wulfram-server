"""Source-backed tank controller state at 004f9700/004f9790.

Configuration inputs are explicit: vehicle BEHAVIOR supplies minimum throttle
and additional drag; the original tank descriptor supplies base drag/friction.
"""
from dataclasses import dataclass
import math
from .physics import _f32


@dataclass(frozen=True)
class JetPhysics:
    throttle: float
    damping: float
    friction: float

    @property
    def active(self):
        return self.throttle != 0.0


def jet_physics(throttle, minimum, base_damping, additional_damping,
                base_friction, general_mobility=1.0):
    values = (throttle, minimum, base_damping, additional_damping,
              base_friction, general_mobility)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite tank physics configuration")
    throttle = _f32(throttle)
    if throttle != 0 and throttle < minimum:
        throttle = _f32(minimum)
    if throttle == 0:
        return JetPhysics(throttle, 2.0, _f32(base_friction))
    return JetPhysics(throttle, _f32(_f32(base_damping) +
                      _f32(additional_damping) * _f32(general_mobility)), _f32(.1))
