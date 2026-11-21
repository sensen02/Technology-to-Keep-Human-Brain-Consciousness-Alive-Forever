"""Block-local storage and shared faces for synchronous two-level finite volumes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np
from numpy.typing import NDArray

from .world import Cell, Field

MaterialArray = NDArray[np.int32]
Index = tuple[int | slice, ...]
BlockKey = tuple[int, ...]


def subdivide(values: NDArray) -> NDArray:
    for axis in range(3):
        values = np.repeat(values, 2, axis=axis)
    return values


def restrict(values: Field) -> Field:
    x, y, z = values.shape
    return values.reshape(x // 2, 2, y // 2, 2, z // 2, 2).mean(axis=(1, 3, 5))


@dataclass
class Block:
    key: BlockKey
    start: Cell
    level: int
    materials: MaterialArray
    fields: dict[str, Field]

    @property
    def base_shape(self) -> Cell:
        return tuple(n >> self.level for n in self.materials.shape)

    @property
    def region(self) -> tuple[slice, ...]:
        return tuple(slice(i, i + n) for i, n in zip(self.start, self.base_shape))


@dataclass(frozen=True)
class Side:
    block: BlockKey
    index: Index
    coarse: bool = False

    def read(self, values: NDArray) -> NDArray:
        view = values[self.index]
        if self.coarse:
            view = np.repeat(np.repeat(view, 2, axis=0), 2, axis=1)
        return view

    def add(self, target: NDArray, change: NDArray) -> None:
        if self.coarse:
            x, y = change.shape
            change = change.reshape(x // 2, 2, y // 2, 2).sum(axis=(1, 3))
        target[self.index] += change


@dataclass(frozen=True)
class Face:
    low: Side
    high: Side
    axis: int
    area_m2: float
    center_distance_m: float


def _plane(axis: int, position: int | slice) -> Index:
    index: list[int | slice] = [slice(None)] * 3
    index[axis] = position
    return tuple(index)


def partition(mask: NDArray[np.bool_]) -> list[tuple[tuple[slice, ...], int]]:
    """Pack selected and unselected base cells into disjoint rectangular arrays."""
    pending = [tuple(slice(0, n) for n in mask.shape)]
    regions = []
    while pending:
        region = pending.pop()
        part = mask[region]
        if not part.any() or part.all():
            regions.append((region, int(part.flat[0])))
            continue
        # Prefer planes separating the most unlike neighbors so thin layers stay contiguous.
        best = (-1.0, -1.0)
        for axis, length in enumerate(part.shape):
            if length == 1:
                continue
            changes = part[_plane(axis, slice(1, None))] != part[_plane(axis, slice(None, -1))]
            fractions = changes.mean(axis=tuple(i for i in range(3) if i != axis))
            for offset, fraction in enumerate(fractions, start=1):
                score = (fraction, -abs(offset / length - 0.5))
                if score > best:
                    best, split_axis, split_offset = score, axis, offset
        left, right = list(region), list(region)
        position = region[split_axis].start + split_offset
        left[split_axis] = slice(region[split_axis].start, position)
        right[split_axis] = slice(position, region[split_axis].stop)
        pending.extend((tuple(right), tuple(left)))
    return regions


def build_faces(
    tiles: dict[Cell, tuple[Block, ...]], shape: Cell, tile_size: int,
    spacing_m: float, periodic: tuple[bool, bool, bool],
) -> list[Face]:
    return list(FaceGroups(tiles, shape, tile_size, spacing_m, periodic))


class FaceGroups:
    """Cache face patches by the base tile owning their low endpoint."""

    def __init__(self, tiles, shape, tile_size, spacing_m, periodic, previous=None):
        self.tiles = tiles
        self.ownership = {}
        self.groups = {}
        changed = {key for key, blocks in tiles.items() if previous is None or blocks is not previous.tiles[key]}
        dirty = changed.copy()
        # Cross-tile faces are emitted by the tile on the low side, including periodic wraps.
        for key in changed:
            for axis in range(3):
                neighbor = list(key)
                neighbor[axis] -= 1
                if neighbor[axis] < 0 and periodic[axis]:
                    neighbor[axis] = (shape[axis] - 1) // tile_size
                dirty.add(tuple(neighbor))
        for key, blocks in tiles.items():
            if len(blocks) > 1:
                self.ownership[key] = (_ownership(key, blocks, shape, tile_size) if key in changed
                                       else previous.ownership[key])
        for key, blocks in tiles.items():
            self.groups[key] = (_build_tile_faces(blocks, tiles, self.ownership, shape, tile_size, spacing_m, periodic)
                                if key in dirty else previous.groups[key])

    def __iter__(self):
        return chain.from_iterable(self.groups.values())

    def __len__(self):
        return sum(map(len, self.groups.values()))


def _ownership(tile, blocks, shape, tile_size):
    # Only mixed tiles require an ownership map. Uniform tiles resolve directly.
    start = tuple(i * tile_size for i in tile)
    labels = np.empty(tuple(min(tile_size, n - i) for n, i in zip(shape, start)), dtype=np.int32)
    for index, block in enumerate(blocks):
        region = tuple(slice(i.start - o, i.stop - o) for i, o in zip(block.region, start))
        labels[region] = index
    return labels


def _build_tile_faces(blocks, tiles, ownership, shape, tile_size, spacing_m, periodic):
    faces = []
    for block in blocks:
        width = spacing_m / (2 ** block.level)
        region = block.region
        for axis, size in enumerate(block.materials.shape):
            if size > 1:
                faces.append(Face(Side(block.key, _plane(axis, slice(None, -1))),
                                  Side(block.key, _plane(axis, slice(1, None))), axis, width ** 2, width))
            neighbor_cell = list(block.start)
            neighbor_cell[axis] = region[axis].stop
            if neighbor_cell[axis] == shape[axis]:
                if not periodic[axis]:
                    continue
                neighbor_cell[axis] = 0
            tile = tuple(i // tile_size for i in neighbor_cell)
            neighbors = tiles[tile]
            if tile in ownership:
                start = tuple(i * tile_size for i in tile)
                plane = [slice(r.start - o, r.stop - o) for r, o in zip(region, start)]
                plane[axis] = neighbor_cell[axis] - start[axis]
                neighbors = tuple(neighbors[index] for index in np.unique(ownership[tile][tuple(plane)]))
            for other in neighbors:
                other_width = spacing_m / (2 ** other.level)
                low_index, high_index = [], []
                for dimension, (a, b) in enumerate(zip(region, other.region)):
                    if dimension == axis:
                        low_index.append(-1)
                        high_index.append(0)
                        continue
                    begin, end = max(a.start, b.start), min(a.stop, b.stop)
                    low_index.append(slice((begin - a.start) << block.level, (end - a.start) << block.level))
                    high_index.append(slice((begin - b.start) << other.level, (end - b.start) << other.level))
                faces.append(Face(
                    Side(block.key, tuple(low_index), coarse=width > other_width),
                    Side(other.key, tuple(high_index), coarse=other_width > width),
                    axis, min(width, other_width) ** 2, 0.5 * (width + other_width),
                ))
    return faces
