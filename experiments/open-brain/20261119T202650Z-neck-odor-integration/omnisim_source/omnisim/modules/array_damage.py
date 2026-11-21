"""Reduced plane-strain insertion mechanics for parallel rigid shafts."""

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class RadialInsertion:
    """Rounded probe opening transverse, incompressible neo-Hookean slices.

    Each slice starts with a regularized core. The mechanical work is evaluated
    along the radial contact solution. Axial traction and interaction between
    shafts are outside this reduced model.
    """

    radius_m: float = 2.0e-6
    core_m: float = 0.2e-6
    shear_Pa: float = 3e3
    onset_Pa: float = 400
    failure_Pa: float = 850

    def __post_init__(self):
        values = (self.radius_m, self.core_m, self.shear_Pa, self.onset_Pa, self.failure_Pa)
        if (not all(isfinite(value) and value > 0 for value in values)
                or self.core_m >= self.radius_m or self.onset_Pa >= self.failure_Pa):
            raise ValueError("invalid radial insertion parameters")

    def strain_energy(self, radius_m, travel_after_tip_m):
        """Undamaged strain energy at a material's pre-insertion radius."""
        offset = np.clip(travel_after_tip_m, 0, self.radius_m)
        contact2 = offset * (2 * self.radius_m - offset)
        q = np.maximum(contact2 - self.core_m**2, 0) / np.maximum(radius_m, self.core_m)**2
        # The principal stretches are sqrt(1+q), 1/sqrt(1+q), 1.
        return (self.shear_Pa / 2) * q**2 / (1 + q)

    def spatial_strain_energy(self, radius_m, travel_after_tip_m):
        """Evaluate the same work at an occupied tissue voxel in the deformed slice."""
        offset = np.clip(travel_after_tip_m, 0, self.radius_m)
        expansion = np.maximum(offset * (2 * self.radius_m - offset) - self.core_m**2, 0)
        reference2 = np.maximum(np.square(radius_m) - expansion, self.core_m**2)
        return self.strain_energy(np.sqrt(reference2), travel_after_tip_m)

    def spatial_threshold_radius_m(self, energy_Pa):
        """Threshold radius in spatial coordinates at full shaft engagement."""
        reference = self._threshold(energy_Pa)[1]
        return sqrt(reference**2 + self.radius_m**2 - self.core_m**2)

    def _threshold(self, energy_Pa):
        if not isfinite(energy_Pa) or energy_Pa <= 0:
            raise ValueError("strain energy threshold must be finite and positive")
        k = 2 * energy_Pa / self.shear_Pa
        q = (k + sqrt(k * (k + 4))) / 2
        radius = sqrt((self.radius_m**2 - self.core_m**2) / q)
        return q, radius

    def failure_radius_m(self):
        """Reference radius at full shaft engagement, derived from strain energy."""
        return self._threshold(self.failure_Pa)[1]

    def first_depths(self, start, end, axes, energy_Pa):
        """First along-shaft load station at which each segment reaches the threshold."""
        start, end, axes = (np.asarray(v, dtype=float) for v in (start, end, axes))
        q, radius = self._threshold(energy_Pa)
        first = np.full(len(start), np.inf)
        if radius < self.core_m:
            return first
        low, high = np.minimum(start, end), np.maximum(start, end)
        epsilon = 64 * np.finfo(float).eps

        def depth(t, radial, tangent, origin, rate):
            r2 = np.square(radial + t[:, None] * tangent).sum(axis=1)
            contact2 = self.core_m**2 + q * np.maximum(self.core_m**2, r2)
            lag = self.radius_m - np.sqrt(np.maximum(self.radius_m**2 - contact2, 0))
            return origin + t * rate + lag

        for a, b in axes:
            length = np.linalg.norm(b - a)
            if length <= 0:
                raise ValueError("every shaft axis must have positive length")
            direction = (b - a) / length
            possible = np.all((low <= np.maximum(a, b) + radius)
                              & (high >= np.minimum(a, b) - radius), axis=1)
            ids = np.flatnonzero(possible)
            p = start[ids] - a
            d = end[ids] - start[ids]
            at, dz = p @ direction, d @ direction
            radial, tangent = p - at[:, None] * direction, d - dz[:, None] * direction
            speed2 = np.square(tangent).sum(axis=1)
            varying = speed2 > epsilon * radius**2
            t_min = np.divide(-np.einsum("ij,ij->i", radial, tangent), speed2,
                              out=np.zeros(len(ids)), where=varying)
            closest2 = np.square(radial + t_min[:, None] * tangent).sum(axis=1)
            width2 = np.divide(radius**2 - closest2, speed2, out=np.full(len(ids), np.inf), where=varying)
            half_width = np.sqrt(np.maximum(0, width2))
            begin, finish = np.maximum(0, t_min - half_width), np.minimum(1, t_min + half_width)
            moving = np.abs(dz) > epsilon * length
            crossing0 = np.divide(-at, dz, out=np.full(len(ids), -np.inf), where=moving)
            crossing1 = np.divide(length - at, dz, out=np.full(len(ids), np.inf), where=moving)
            begin = np.maximum(begin, np.minimum(crossing0, crossing1))
            finish = np.minimum(finish, np.maximum(crossing0, crossing1))
            valid = ((closest2 <= radius**2) & (begin <= finish)
                     & (moving | ((at >= 0) & (at <= length))))
            lo, hi = begin[valid], finish[valid]
            rp, dr, z0, vz = radial[valid], tangent[valid], at[valid], dz[valid]
            endpoints = np.minimum(depth(lo, rp, dr, z0, vz), depth(hi, rp, dr, z0, vz))
            # On this feasible interval the load-station function is convex.
            for _ in range(28):
                t1, t2 = (2 * lo + hi) / 3, (lo + 2 * hi) / 3
                earlier = depth(t1, rp, dr, z0, vz) <= depth(t2, rp, dr, z0, vz)
                lo, hi = np.where(earlier, lo, t1), np.where(earlier, t2, hi)
            arrival = np.minimum(endpoints, depth((lo + hi) / 2, rp, dr, z0, vz))
            arrival = np.where(arrival <= length, arrival, np.inf)
            targets = ids[valid]
            first[targets] = np.minimum(first[targets], arrival)
        return first
