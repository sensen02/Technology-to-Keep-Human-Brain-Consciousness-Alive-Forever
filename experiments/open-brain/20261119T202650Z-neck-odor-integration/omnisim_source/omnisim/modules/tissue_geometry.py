"""Geometry shared by modules that move through the tissue voxels."""

from math import inf

import numpy as np

from .tissue import TISSUE


def move_toward(start, target, distance):
    delta = target - start
    length = np.linalg.norm(delta)
    return target.copy() if length <= distance else start + delta * (distance / length)


def tissue_path_intervals(world, start, end):
    """Yield interior points and lengths of positive-volume ray intervals."""
    delta = end - start
    length = float(np.linalg.norm(delta))
    if not length:
        return
    scale = float(np.max(np.abs(delta)))
    nudge = 1e-10 * world.grid.spacing_m / scale
    t = 0.0
    while t < 1.0:
        point = start + min(t + nudge, (t + 1) / 2) * delta
        key, cell = world.cell_location(point)
        grid = world.block_grid(key)
        crossing = min(((grid.origin_m[axis] + (cell[axis] + (d > 0)) * grid.spacing_m - start[axis]) / d
                        for axis, d in enumerate(delta) if d), default=inf)
        # Face ties may round to the current coordinate. Advance into the next cell.
        if crossing <= t:
            t += nudge
            continue
        stop = min(crossing, 1.0)
        yield start + ((t + stop) / 2) * delta, (stop - t) * length
        t = stop


def tissue_path_clear(world, start, end, axon_material=None):
    """Walk the currently active Cartesian cells intersected by a segment."""
    allowed = (TISSUE,) if axon_material is None else (TISSUE, axon_material)
    try:
        return (world.material_at(start) in allowed and world.material_at(end) in allowed
                and all(world.material_at(point) in allowed
                        for point, _ in tissue_path_intervals(world, start, end)))
    except ValueError:
        return False
