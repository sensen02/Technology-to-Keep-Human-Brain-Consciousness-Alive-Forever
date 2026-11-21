"""Integrate the injury field over the actual intervals crossed by a neural path."""

from dataclasses import dataclass
from math import exp

import numpy as np

from .tissue_geometry import tissue_path_intervals


@dataclass(frozen=True)
class ConductionDamage:
    length_m: float
    injury_integral_m: float = 0.0
    block_at_m: float | None = None

    @property
    def gain(self):
        if self.block_at_m is not None:
            return 0.0
        # Engineering prior: e-fold loss across 20 µm of unit-equivalent injury.
        return exp(-self.injury_integral_m / 20e-6)

    def delay_s(self, speed_m_s):
        return (self.length_m + self.injury_integral_m) / speed_m_s


def path_damage(world, points, field, *, blocked_material=None, allowed_materials=None):
    """Full-failure voxel intersections block; partial damage integrates by arc length."""
    length = injury = 0.0
    for a, b in zip(points, points[1:]):
        a, b = np.asarray(a), np.asarray(b)
        try:
            for midpoint, segment_m in tissue_path_intervals(world, a, b):
                material = world.material_at(midpoint) if (allowed_materials is not None
                                                           or blocked_material is not None) else None
                if ((blocked_material is not None and material == blocked_material)
                        or (allowed_materials is not None and material not in allowed_materials)):
                    return ConductionDamage(length, injury, length)
                damage = float(np.clip(world.sample(field, midpoint), 0.0, 1.0)) if field is not None else 0.0
                if damage >= 1.0:
                    return ConductionDamage(length, injury, length)
                injury += segment_m * damage
                length += segment_m
        except ValueError:
            return ConductionDamage(length, injury, length)
    return ConductionDamage(length, injury)
