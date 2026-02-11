"""
Spring-damper physics model for vehicle heading.

The Wulfram2 client uses a two-stage spring-based physics system for tank rotation:

  Stage 1: Spring internal displacement
    Input → Piecewise curve → Spring TARGET (softbody[0x23], clamped [-1,1])
    Spring dynamics: F = -k*(displacement - target) - c*velocity
    → displacement oscillates toward target

  Stage 2: Displacement → torque → angular velocity → heading
    torque = displacement * torque_scale
    angular_velocity += torque           (impulse, not force*dt)
    angular_velocity *= (1 - dampening)  (per-frame rotation dampening)
    heading += angular_velocity * dt

Client references:
  - Vehicles.c:1015-1040 (GUESS2_Tank_apply_steering_response) — sets softbody[0x23]
  - Vehicles.c:1225 (rotation dampening = turn_rate * 1.4137167 ≈ 0.0707)
  - Physics.c:1934-1947 (SpringParam_init_uniform, stiffness=40)
  - Physics.c:2473-2586 (GUESS4_Spring_simulate_step)
  - Physics.c:2594-2618 (GUESS4_Spring_apply_forces_to_entity) — adds torque to ang_vel
"""

import math


class SpringDamper1D:
    """Second-order spring-damper for spring internal displacement.

    Models the spring's own position/velocity, separate from entity heading.
    Target is set from steering response (clamped [-1,1]).
    Output (value) is the spring displacement, converted to torque externally.
    """

    def __init__(self, stiffness: float = 40.0, damping: float = 4.0):
        self.k = stiffness       # Spring stiffness
        self.c = damping          # Spring damping
        self.value = 0.0          # Spring displacement (internal state, NOT heading)
        self.velocity = 0.0       # Spring velocity (internal)
        self.target = 0.0         # Target displacement (from steering, [-1,1])

    def step(self, dt: float):
        """Integrate one timestep using semi-implicit Euler."""
        error = self.value - self.target
        acceleration = -self.k * error - self.c * self.velocity
        self.velocity += acceleration * dt
        self.value += self.velocity * dt


class VehiclePhysics:
    """Per-vehicle two-stage spring-damper physics.

    Stage 1: Spring internal displacement tracks steering target
    Stage 2: Spring displacement → torque → angular velocity → heading

    Matches the client's actual physics pipeline from Vehicles.c / Physics.c.
    All parameters are tunable at runtime via control commands.
    """

    def __init__(
        self,
        spring_stiffness: float = 40.0,
        spring_damping: float = 4.0,
        torque_scale: float = 0.342,
        rotation_dampening: float = 0.0707,
    ):
        # Stage 1: Spring internal state
        self.spring = SpringDamper1D(stiffness=spring_stiffness, damping=spring_damping)

        # Stage 2: Entity rotational state
        self._heading = 0.0
        self._angular_velocity = 0.0

        # Torque scale: converts spring displacement to angular velocity impulse.
        # Derived from lever arm in Spring_apply_forces_to_entity.
        # Calibrated so that steady-state turn rate ≈ turn_adjust (4.5 rad/s).
        # Formula: torque_scale = desired_vel * dampening / (1 - dampening)
        #        = 4.5 * 0.0707 / 0.9293 ≈ 0.342
        self.torque_scale = torque_scale

        # Per-frame rotation dampening (Vehicles.c:1225: turn_rate * 1.4137167)
        self.rotation_dampening = rotation_dampening

    def step(self, dt: float):
        """Advance physics by dt seconds.

        1. Step spring (displacement tracks target)
        2. Apply spring displacement as torque impulse to angular velocity
        3. Apply rotation dampening
        4. Integrate heading
        """
        # Stage 1: Spring dynamics
        self.spring.step(dt)

        # Stage 2: Displacement → torque → angular velocity → heading
        # Client does: entity[ang_vel] += torque (impulse, not force*dt)
        self._angular_velocity += self.spring.value * self.torque_scale

        # Per-frame dampening (client: ang_vel *= (1 - dampening))
        self._angular_velocity *= (1.0 - self.rotation_dampening)

        # Integrate heading
        self._heading += self._angular_velocity * dt

        # Wrap to [-pi, pi]
        while self._heading > math.pi:
            self._heading -= 2 * math.pi
        while self._heading < -math.pi:
            self._heading += 2 * math.pi

    @property
    def heading(self) -> float:
        return self._heading

    @heading.setter
    def heading(self, val: float):
        self._heading = val

    @property
    def angular_velocity(self) -> float:
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, val: float):
        self._angular_velocity = val

    def reset(self):
        """Reset all state to zero."""
        self.spring.value = 0.0
        self.spring.velocity = 0.0
        self.spring.target = 0.0
        self._heading = 0.0
        self._angular_velocity = 0.0
