"""Conserved fluid momentum and paired impulses at rigid interfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from .engine import AdaptiveWorld, FieldSpec
from .world import Position


@dataclass(frozen=True)
class RigidBody:
    center_m: Position
    mass_kg: float
    momentum_kg_m_s: Position = (0.0, 0.0, 0.0)
    angular_momentum_kg_m2_s: Position = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("body mass must be positive and finite")
        for name in ("center_m", "momentum_kg_m_s", "angular_momentum_kg_m2_s"):
            values = tuple(getattr(self, name))
            if len(values) != 3 or not all(map(isfinite, values)):
                raise ValueError(f"{name} must contain three finite components")
            object.__setattr__(self, name, values)


class Momentum:
    """Optional mass and signed momentum fields transported by prescribed face velocities."""

    def __init__(self, world: AdaptiveWorld, *, material: int, mass_density_kg_m3: float,
                 prefix: str = "fluid") -> None:
        if not isfinite(mass_density_kg_m3) or mass_density_kg_m3 <= 0:
            raise ValueError("fluid mass density must be positive and finite")
        self.world = world
        self.mass_field = f"{prefix}.mass"
        self.components = tuple(f"{prefix}.momentum_{axis}" for axis in "xyz")
        if set((self.mass_field, *self.components)) & world.field_specs.keys():
            raise ValueError(f"momentum fields already exist for {prefix}")
        world.add_field(self.mass_field, FieldSpec(advected=True, material=material),
                        initial_density=mass_density_kg_m3)
        for name in self.components:
            world.add_field(name, FieldSpec(advected=True, material=material, nonnegative=False))
        self._bodies: dict[str, RigidBody] = {}

    def add_body(self, name: str, body: RigidBody) -> None:
        if name in self._bodies:
            raise ValueError(f"body already exists: {name}")
        self._bodies[name] = body

    def body(self, name: str) -> RigidBody:
        return self._bodies[name]

    def exchange_impulse(self, name: str, liquid_position_m: Position, impulse_on_body_N_s: Position,
                         *, contact_m: Position | None = None) -> None:
        """Apply a supplied impulse to the rigid body and its opposite to the sampled liquid cell."""
        impulse = np.asarray(impulse_on_body_N_s, dtype=np.float64)
        contact = np.asarray(liquid_position_m if contact_m is None else contact_m, dtype=np.float64)
        if any(vector.shape != (3,) or not np.isfinite(vector).all() for vector in (impulse, contact)):
            raise ValueError("impulse and contact must have three finite components")
        body = self._bodies[name]
        updated = replace(
            body,
            momentum_kg_m_s=tuple(np.asarray(body.momentum_kg_m_s) + impulse),
            angular_momentum_kg_m2_s=tuple(np.asarray(body.angular_momentum_kg_m2_s)
                                            + np.cross(contact - body.center_m, impulse)),
        )
        self.world.add_amounts(liquid_position_m, dict(zip(self.components, map(float, -impulse))))
        self._bodies[name] = updated
