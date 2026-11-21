"""Compact batches of small face fragments on a fixed mesh topology."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .mesh import Block, BlockKey, Face, Side
from .world import Field, Position

_MAX_SUBFACES = (16 * 1024**2) // (3 * np.dtype(np.int32).itemsize)
_FRAGMENT_SIZE_LIMIT = 64
_MIN_FRAGMENT_COUNT = 32


@dataclass(frozen=True)
class FaceBatch:
    axis: int
    area_m2: float
    distance_m: float
    low: NDArray[np.int32]
    high: NDArray[np.int32]
    material: NDArray[np.int32]

    def accumulate(self, density: Field, change: Field, velocity: float, diffusivity: float,
                   material: int | None, dt_s: float) -> None:
        left, right = density[self.low], density[self.high]
        flux = velocity * (left if velocity >= 0 else right)
        if diffusivity:
            flux += (diffusivity / self.distance_m) * (left - right)
        if material is not None:
            flux *= self.material == material
        transfer = flux * self.area_m2 * dt_s
        # Multiple fine subfaces can contribute to one coarse cell.
        np.add.at(change, self.low, -transfer)
        np.add.at(change, self.high, transfer)


@dataclass(frozen=True)
class FaceBatches:
    regions: dict[BlockKey, slice]
    cells: int
    batches: tuple[FaceBatch, ...]
    remaining: tuple[Face, ...]

    def accumulate(self, density: Field, change: Field, velocity: Position, diffusivity: float,
                   material: int | None, dt_s: float) -> None:
        for batch in self.batches:
            speed = velocity[batch.axis]
            if diffusivity or speed:
                batch.accumulate(density, change, speed, diffusivity, material, dt_s)


def _flat_indices(side: Side, block: Block, offset: int) -> NDArray[np.int32]:
    # Generate the needed indices directly, without a mesh-sized index array.
    shape = block.materials.shape
    ranges = [np.arange(*index.indices(size), dtype=np.int32) if isinstance(index, slice)
              else np.asarray([index % size], dtype=np.int32) for size, index in zip(shape, side.index)]
    x, y, z = np.ix_(*ranges)
    cells = (x * shape[1] + y) * shape[2] + z + offset
    if side.coarse:
        for axis, index in enumerate(side.index):
            if isinstance(index, slice):
                cells = np.repeat(cells, 2, axis=axis)
    return cells.ravel()


def build_face_batches(blocks: dict[BlockKey, Block], faces: list[Face]) -> FaceBatches | None:
    regions = {}
    cells = 0
    for key, block in blocks.items():
        end = cells + block.materials.size
        regions[key] = slice(cells, end)
        cells = end
    if cells > np.iinfo(np.int32).max:
        return None

    groups = {}
    remaining = []
    patch_count, subfaces = 0, 0
    for face in faces:
        side = face.high if face.low.coarse else face.low
        size = blocks[side.block].materials[side.index].size
        if size >= _FRAGMENT_SIZE_LIMIT or subfaces + size > _MAX_SUBFACES:
            remaining.append(face)
            continue
        groups.setdefault((face.axis, face.area_m2, face.center_distance_m), []).append((face, size))
        patch_count += 1
        subfaces += size
    if patch_count < _MIN_FRAGMENT_COUNT:
        return None

    batches = []
    for (axis, area, distance), entries in groups.items():
        size = sum(n for _, n in entries)
        low = np.empty(size, dtype=np.int32)
        high = np.empty_like(low)
        material = np.empty_like(low)
        start = 0
        for face, count in entries:
            region = slice(start, start + count)
            left, right = blocks[face.low.block], blocks[face.high.block]
            low[region] = _flat_indices(face.low, left, regions[left.key].start)
            high[region] = _flat_indices(face.high, right, regions[right.key].start)
            left_material = face.low.read(left.materials).ravel()
            right_material = face.high.read(right.materials).ravel()
            material[region] = np.where(left_material == right_material, left_material, -1)
            start += count
        batches.append(FaceBatch(axis, area, distance, low, high, material))
    return FaceBatches(regions, cells, tuple(batches), tuple(remaining))
