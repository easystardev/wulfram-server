"""Longitudinal governor recovered at 0x004f9790.

Cross-checked with Baffler Physics Core 5f7ed88, world.cpp:2860-2874,
and rebuild/analysis/tank-mobility. This is a semantic model, not an
x87 bit-exact implementation. The server uses this by default; the legacy
path remains available for matched A/B captures.
"""

from wulfram2_protocol.entities import tank_fuel_mobility_factor


def tank_forward_mobility(
    velocity, forward, throttle, max_velocity, fuel, low_fuel_level,
    altitude_factor=1.0,
):
    """Return (forward scale, signed forward speed).

    Overspeed only limits input accelerating in the direction of travel.
    Apply the altitude multiplier before the low-fuel cap (minimum, not
    multiplication). Velocity is entity+0x18; entity+0xD4 is fuel.
    """
    speed = ((velocity[2] * forward[2] + velocity[0] * forward[0])
             + velocity[1] * forward[1])
    scale = 1.0
    same_direction = (speed > 0 and throttle > 0) or (speed < 0 and throttle < 0)
    if max_velocity > 0 and same_direction and abs(speed) > max_velocity:
        excess = min((abs(speed) - max_velocity) / max_velocity, 0.2)
        scale = (0.2 - excess) / 0.2
    return min(scale * altitude_factor,
               tank_fuel_mobility_factor(fuel, low_fuel_level)), speed
