"""Conservative, no-flux diffusion on the shared grid."""

from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np

from .world import World


def adjacent_cells() -> Iterator[tuple[tuple[slice, ...], tuple[slice, ...]]]:
    for axis in range(3):
        low = [slice(None)] * 3
        high = low.copy()
        low[axis] = slice(None, -1)
        high[axis] = slice(1, None)
        yield tuple(low), tuple(high)


@dataclass(frozen=True)
class Diffusion:
    field: str
    diffusivity_m2_s: float
    solid_fraction_field: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.diffusivity_m2_s) or self.diffusivity_m2_s < 0:
            raise ValueError("diffusivity_m2_s must be finite and nonnegative")
        if self.field == self.solid_fraction_field:
            raise ValueError("a diffusion field cannot also represent the solid geometry")

    def attach(self, world: World, name: str) -> None:
        world.fields[self.field]
        if self.solid_fraction_field is not None:
            world.fields[self.solid_fraction_field]

    def step(self, world: World, name: str, dt_s: float) -> None:
        if self.diffusivity_m2_s == 0:
            return
        values = world.fields[self.field]
        ratio = self.diffusivity_m2_s * dt_s / world.grid.spacing_m ** 2
        # The three-dimensional explicit stencil requires 6 * ratio <= 1.
        substeps = max(1, ceil(6 * ratio / 0.9))
        ratio /= substeps
        neighbors = tuple(adjacent_cells())
        open_cells = None
        if self.solid_fraction_field is not None:
            open_cells = world.fields[self.solid_fraction_field] == 0
        change = np.empty_like(values)
        for _ in range(substeps):
            change.fill(0)
            for low, high in neighbors:
                transfer = ratio * (values[high] - values[low])
                if open_cells is not None:
                    transfer *= open_cells[low] & open_cells[high]
                change[low] += transfer
                change[high] -= transfer
            values += change
