"""Conservative three-dimensional world with adaptive block storage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from math import ceil, inf, isfinite
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import ArrayLike

from .flux import FaceBatches, build_face_batches
from .mesh import Block, BlockKey, Face, FaceGroups, build_faces, partition, restrict, subdivide
from .transport import adjacent_cells
from .world import Cell, Event, Field, Grid, Position

if TYPE_CHECKING:
    from .chemistry import BulkReaction, SurfaceDissolution


@dataclass(frozen=True)
class _Topology:
    tiles: dict[Cell, tuple[Block, ...]]
    blocks: dict[BlockKey, Block]
    faces: list[Face] | FaceGroups
    refinement: np.ndarray
    transport_batches: FaceBatches | None = None


@dataclass(frozen=True)
class PorousDiffusion:
    """Accessible volume and tortuosity in one material, surrounded by free fluid."""

    material: int
    volume_fraction: float
    tortuosity: float
    excluded_material_from: int | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.material, Integral) or not 0 <= self.material <= np.iinfo(np.int32).max
                or not isfinite(self.volume_fraction) or not 0 < self.volume_fraction <= 1
                or not isfinite(self.tortuosity) or self.tortuosity < 1):
            raise ValueError("invalid porous diffusion medium")
        if not isfinite(self.resistance_multiplier):
            raise ValueError("invalid porous diffusion resistance")
        excluded = self.excluded_material_from
        if excluded is not None and (not isinstance(excluded, Integral)
                                     or not self.material < excluded < np.iinfo(np.int32).max):
            raise ValueError("invalid excluded diffusion material")

    @property
    def resistance_multiplier(self) -> float:
        return self.tortuosity * self.tortuosity / self.volume_fraction


@dataclass(frozen=True)
class FieldSpec:
    """Transport properties for a cell-integrated scalar, in SI units."""

    diffusivity_m2_s: float = 0.0
    advected: bool = False
    material: int | None = None
    nonnegative: bool = True
    elements: Mapping[str, int] | None = None
    porous: PorousDiffusion | None = None
    isolated_materials_from: int | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.diffusivity_m2_s) or self.diffusivity_m2_s < 0:
            raise ValueError("diffusivity must be finite and nonnegative")
        if self.material is not None and (not isinstance(self.material, Integral)
                                          or not 0 <= self.material <= np.iinfo(np.int32).max):
            raise ValueError("material ID must be a nonnegative 32-bit integer")
        if self.elements is not None:
            if not self.elements or any(not element or not isinstance(count, Integral) or count <= 0
                                        for element, count in self.elements.items()):
                raise ValueError("element counts must be positive integers")
            object.__setattr__(self, "elements", MappingProxyType(dict(self.elements)))
        if self.porous is not None and (not isinstance(self.porous, PorousDiffusion) or self.advected
                                        or self.material is not None):
            raise ValueError("porous transport requires diffusion without advection or a separate material restriction")
        isolated = self.isolated_materials_from
        if isolated is not None and (not isinstance(isolated, Integral) or not 0 <= isolated < np.iinfo(np.int32).max
                                     or self.material is not None or self.porous is not None):
            raise ValueError("invalid isolated material range")

    def available_in(self, material):
        if self.material is not None:
            return material == self.material
        if self.isolated_materials_from is not None:
            return material >= self.isolated_materials_from
        if self.porous is not None and self.porous.excluded_material_from is not None:
            return material < self.porous.excluded_material_from
        return True


@dataclass(frozen=True)
class Tolerance:
    absolute: float
    relative: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.absolute) or self.absolute <= 0:
            raise ValueError("absolute tolerance must be positive and finite")
        if not isfinite(self.relative) or self.relative < 0:
            raise ValueError("relative tolerance must be nonnegative and finite")


class AdaptiveModule(Protocol):
    def attach(self, world: AdaptiveWorld, name: str) -> None: ...

    def step(self, world: AdaptiveWorld, name: str, dt_s: float) -> None: ...


class AdaptiveWorld:
    _solver_source = "solver"

    def __init__(
        self,
        grid: Grid,
        *,
        block_cells: int = 8,
        materials: ArrayLike = 0,
        fine_materials: Mapping[Cell, ArrayLike] | None = None,
        periodic: tuple[bool, bool, bool] = (False, False, False),
    ) -> None:
        if not isinstance(block_cells, Integral) or block_cells < 1:
            raise ValueError("block_cells must be a positive integer")
        if len(periodic) != 3:
            raise ValueError("periodic must contain three axis flags")
        material_data = np.asarray(materials)
        if (material_data.dtype.kind not in "iu" or np.any(material_data < 0)
                or np.any(material_data > np.iinfo(np.int32).max)):
            raise ValueError("materials must contain nonnegative 32-bit integer IDs")
        self._materials = np.array(np.broadcast_to(material_data, grid.shape), dtype=np.int32)
        self.grid = grid
        self._block_cells = int(block_cells)
        counts = tuple((n + block_cells - 1) // block_cells for n in grid.shape)
        self._periodic = tuple(periodic)
        self._root_regions: dict[Cell, tuple[slice, ...]] = {}
        blocks: dict[BlockKey, Block] = {}
        for key in product(*(range(n) for n in counts)):
            start = tuple(k * block_cells for k in key)
            region = tuple(slice(s, min(s + block_cells, n)) for s, n in zip(start, grid.shape))
            self._root_regions[key] = region
            blocks[key] = Block(key, start, 0, self._materials[region], {})
        tiles = {key: (block,) for key, block in blocks.items()}
        self._topology = _Topology(tiles, blocks, self._build_faces(tiles), np.zeros(grid.shape, dtype=bool))
        self._mesh_revision = 0
        self._specs: dict[str, FieldSpec] = {}
        self.field_specs = MappingProxyType(self._specs)
        self._chemistry: list[BulkReaction | SurfaceDissolution] = []
        self._velocity_m_s: Position = (0.0, 0.0, 0.0)
        self._time_s = 0.0
        self._failed = False
        self._stepping = False
        self._modules: dict[str, AdaptiveModule] = {}
        self._events: list[Event] = []
        self._quiet_checks = np.zeros(grid.shape, dtype=np.uint16)
        self._fine_materials: dict[Cell, dict[Cell, np.ndarray]] = {}
        self._geometry_pins: tuple[Cell, ...] = tuple(fine_materials or ())
        if self._geometry_pins:
            selected = np.zeros(grid.shape, dtype=bool)
            self._mark_cells(selected, self._geometry_pins)
            for cell in self._geometry_pins:
                children = np.asarray(fine_materials[cell])
                if (children.shape != (2, 2, 2) or children.dtype.kind not in "iu"
                        or np.any(children < 0) or np.any(children > np.iinfo(np.int32).max)):
                    raise ValueError("fine materials require eight nonnegative 32-bit child IDs per base cell")
                key = tuple(i // self._block_cells for i in cell)
                self._fine_materials.setdefault(key, {})[cell] = children.astype(np.int32)
            self._regrid(selected)

    @property
    def _tiles(self):
        return self._topology.tiles

    @property
    def _blocks(self):
        return self._topology.blocks

    @property
    def _faces(self):
        return self._topology.faces

    @property
    def _refinement(self):
        return self._topology.refinement

    @property
    def _transport_batches(self):
        return self._topology.transport_batches

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def refined_blocks(self) -> frozenset[BlockKey]:
        return frozenset(key for key, block in self._blocks.items() if block.level)

    @property
    def block_keys(self) -> tuple[BlockKey, ...]:
        return tuple(self._blocks)

    @property
    def mesh_revision(self) -> int:
        """Increase whenever a replacement topology is published."""
        return self._mesh_revision

    def refinement_mask(self) -> np.ndarray:
        """Return a copy marking the base cells currently replaced by eight children."""
        return self._refinement.copy()

    @property
    def active_cells(self) -> int:
        return sum(block.materials.size for block in self._blocks.values())

    def _volume(self, block: Block) -> float:
        return (self.grid.spacing_m / (2 ** block.level)) ** 3

    def add_field(self, name: str, spec: FieldSpec = FieldSpec(), *, initial_density: ArrayLike = 0) -> None:
        """Initialize from a scalar or an array on the base grid; store integrals."""
        if name in self.field_specs:
            raise ValueError(f"field already exists: {name}")
        density = np.broadcast_to(np.asarray(initial_density, dtype=np.float64), self.grid.shape)
        if not np.isfinite(density).all() or (spec.nonnegative and np.any(density < 0)):
            raise ValueError(f"invalid initial density for {name}")
        amounts = {}
        for key, block in self._blocks.items():
            local = density[block.region]
            if block.level:
                local = subdivide(local)
            available = spec.available_in(block.materials)
            if available is not True:
                local = np.where(available, local, 0.0)
            amounts[key] = local * self._volume(block)
            if not np.isfinite(amounts[key]).all():
                raise ValueError(f"initial amount overflows for {name}")
        for key, values in amounts.items():
            self._blocks[key].fields[name] = values
        self._specs[name] = spec

    def _cell_at(self, position_m: Position) -> tuple[Block, Cell]:
        coarse = self.grid.cell_at(position_m)
        key = tuple(i // self._block_cells for i in coarse)
        neighbors = self._tiles[key]
        block = neighbors[0] if len(neighbors) == 1 else next(
            leaf for leaf in neighbors if all(r.start <= i < r.stop for r, i in zip(leaf.region, coarse))
        )
        local = tuple(i - start for i, start in zip(coarse, block.start))
        if block.level:
            fine = tuple(int(2 * ((x - origin) / self.grid.spacing_m)) - 2 * i
                         for i, x, origin in zip(coarse, position_m, self.grid.origin_m))
            local = tuple(2 * i + offset for i, offset in zip(local, fine))
        return block, local

    def block_at(self, position_m: Position) -> BlockKey:
        return self._cell_at(position_m)[0].key

    def cell_location(self, position_m: Position) -> tuple[BlockKey, Cell]:
        """Return active block and local indices valid for the current mesh revision."""
        block, local = self._cell_at(position_m)
        return block.key, local

    def material_at(self, position_m: Position) -> int:
        block, cell = self._cell_at(position_m)
        return int(block.materials[cell])

    def sample(self, name: str, position_m: Position) -> float:
        """Sample the density in the finest active cell at this world position."""
        block, cell = self._cell_at(position_m)
        return float(block.fields[name][cell] / self._volume(block))

    def sample_concentration(self, name: str, position_m: Position) -> float:
        """Sample concentration per accessible volume for a dissolved field."""
        medium = self.field_specs[name].porous
        density = self.sample(name, position_m)
        if medium is not None and self.material_at(position_m) == medium.material:
            return density / medium.volume_fraction
        return density

    def total(self, name: str) -> float:
        if name not in self.field_specs:
            raise KeyError(name)
        return sum(float(block.fields[name].sum()) for block in self._blocks.values())

    def block_densities(self, name: str, key: BlockKey) -> Field:
        block = self._blocks[key]
        return block.fields[name] / self._volume(block)

    def _densities_to_amounts(self, name: str, block: Block, density: ArrayLike) -> Field:
        spec = self.field_specs[name]
        amounts = np.broadcast_to(np.asarray(density, dtype=np.float64), block.materials.shape) * self._volume(block)
        if not np.isfinite(amounts).all() or (spec.nonnegative and np.any(amounts < 0)):
            raise ValueError(f"invalid densities for {name}")
        available = spec.available_in(block.materials)
        if available is not True:
            amounts = np.where(available, amounts, 0.0)
        return amounts

    def set_block_densities(self, name: str, key: BlockKey, density: ArrayLike) -> None:
        """Replace one block's field with densities on its current active cells."""
        block = self._blocks[key]
        block.fields[name][:] = self._densities_to_amounts(name, block, density)

    def block_grid(self, key: BlockKey) -> Grid:
        """Return the current shape, cell spacing and world origin of an active block."""
        block = self._blocks[key]
        origin = tuple(start + i * self.grid.spacing_m for start, i in zip(self.grid.origin_m, block.start))
        return Grid(block.materials.shape, self.grid.spacing_m / (2 ** block.level), origin)

    def blocks_overlapping(self, lower_m: Position, upper_m: Position) -> Iterable[BlockKey]:
        """Visit the active blocks with positive-volume intersection with a world-space box."""
        lower, upper = np.asarray(lower_m, dtype=float), np.asarray(upper_m, dtype=float)
        if (lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite([lower, upper]))
                or np.any(lower >= upper)):
            raise ValueError("query box must have finite, increasing bounds")
        origin = np.asarray(self.grid.origin_m)
        root_h = self.grid.spacing_m * self._block_cells
        root_end = np.asarray([(n + self._block_cells - 1) // self._block_cells for n in self.grid.shape])
        lo = np.clip(np.floor((lower - origin) / root_h), 0, root_end).astype(int)
        hi = np.clip(np.ceil((upper - origin) / root_h), 0, root_end).astype(int)
        for root in product(*(range(a, b) for a, b in zip(lo, hi))):
            for block in self._tiles[root]:
                grid = self.block_grid(block.key)
                edge = np.asarray(grid.origin_m) + np.asarray(grid.shape) * grid.spacing_m
                if np.all(lower < edge) and np.all(upper > grid.origin_m):
                    yield block.key

    def block_materials(self, key: BlockKey) -> np.ndarray:
        return self._blocks[key].materials.copy()

    def add_amounts(self, position_m: Position, changes: Mapping[str, float]) -> None:
        """Apply a validated group of cell-integrated sources at one position."""
        block, cell = self._cell_at(position_m)
        next_values = {}
        for name, change in changes.items():
            spec = self.field_specs[name]
            if not spec.available_in(block.materials[cell]):
                raise ValueError(f"{name} is unavailable in the target material")
            next_value = float(block.fields[name][cell]) + change
            if not isfinite(next_value) or (spec.nonnegative and next_value < 0):
                raise ValueError(f"source would make {name} invalid")
            next_values[name] = next_value
        for name, value in next_values.items():
            block.fields[name][cell] = value

    def add_reaction(self, rule: BulkReaction | SurfaceDissolution) -> None:
        rule.validate(self.field_specs)
        self._chemistry.append(rule)

    def set_velocity(self, velocity_m_s: Position) -> None:
        """Prescribe an axis-aligned face velocity vector in SI units."""
        if len(velocity_m_s) != 3 or not all(map(isfinite, velocity_m_s)):
            raise ValueError("velocity must have three finite components")
        self._velocity_m_s = tuple(velocity_m_s)

    def _record(self, source: str, kind: str, **data: str | float | int) -> None:
        self.emit(source, kind, **data)

    def emit(self, source: str, kind: str, *, time_s: float | None = None, **data: str | float | int) -> None:
        self._events.append(Event(self.time_s if time_s is None else time_s, source, kind, data))

    def drain_events(self) -> list[Event]:
        events, self._events = self._events, []
        return sorted(events, key=lambda event: event.time_s)

    def _check_ready(self) -> None:
        if self._failed:
            raise RuntimeError("a previous simulation step failed")
        if self._stepping:
            raise RuntimeError("the world is already stepping")

    def add_module(self, name: str, module: AdaptiveModule) -> None:
        """Attach a module between steps. Modules advance in registration order."""
        self._check_ready()
        if name in self._modules:
            raise ValueError(f"module already exists: {name}")
        module.attach(self, name)
        self._modules[name] = module

    def remove_module(self, name: str) -> AdaptiveModule:
        self._check_ready()
        return self._modules.pop(name)

    def _mark_cells(self, mask: np.ndarray, cells: Iterable[Cell]) -> None:
        for cell in cells:
            if len(cell) != 3 or any(not isinstance(i, Integral) or not 0 <= i < n
                                     for i, n in zip(cell, self.grid.shape)):
                raise ValueError(f"refinement refers to an invalid base cell: {cell}")
            mask[cell] = True

    def _mark_blocks(self, mask: np.ndarray, keys: Iterable[BlockKey]) -> None:
        for key in keys:
            if key in self._root_regions:
                mask[self._root_regions[key]] = True
            elif key in self._blocks:
                mask[self._blocks[key].region] = True
            else:
                raise ValueError(f"refinement refers to an unknown block: {key}")

    def regrid(self, refined: Iterable[BlockKey]) -> None:
        """Manually refine whole blocks, e.g. for a uniform fine reference run."""
        selected = np.zeros_like(self._refinement)
        self._mark_blocks(selected, refined)
        self._mark_cells(selected, self._geometry_pins)
        self._regrid(selected)

    def regrid_cells(self, refined: Iterable[Cell]) -> None:
        """Refine exactly the selected base cells, independent of storage blocks."""
        selected = np.zeros_like(self._refinement)
        self._mark_cells(selected, refined)
        self._mark_cells(selected, self._geometry_pins)
        self._regrid(selected)

    def _regrid(self, selected: np.ndarray) -> None:
        changed = selected ^ self._refinement
        if not changed.any():
            return
        self._publish_regrid(self._prepare_regrid(selected, changed), changed)

    def _build_faces(self, tiles, previous=None):
        return build_faces(tiles, self.grid.shape, self._block_cells, self.grid.spacing_m, self._periodic)

    def _prepare_regrid(self, selected: np.ndarray, changed: np.ndarray) -> _Topology:
        updated = self._tiles.copy()
        try:
            with np.errstate(over="raise", invalid="raise"):
                for key, region in self._root_regions.items():
                    if changed[region].any():
                        updated[key] = self._rebuild_tile(key, region, selected[region])
            faces = self._build_faces(updated, self._faces)
            blocks = {block.key: block for tile in updated.values() for block in tile}
            batches = self._make_transport_batches(updated, blocks, faces)
        except MemoryError:
            self._record("mesh", "regrid_out_of_memory", requested_cells=int(selected.sum()))
            raise
        except FloatingPointError as error:
            self._record("mesh", "regrid_invalid_amount", message=str(error))
            raise
        return _Topology(updated, blocks, faces, selected, batches)

    def _publish_regrid(self, topology: _Topology, changed: np.ndarray) -> None:
        previous = self._refinement
        self._topology = topology
        self._mesh_revision += 1
        self._quiet_checks[changed] = 0
        self._record("mesh", "regrid", refined=int(np.count_nonzero(changed & topology.refinement)),
                     coarsened=int(np.count_nonzero(changed & previous)), active_cells=self.active_cells)

    def _make_transport_batches(self, tiles: dict[Cell, tuple[Block, ...]], blocks: dict[BlockKey, Block],
                                faces: list[Face]) -> FaceBatches | None:
        if any(len(tile) > 1 for tile in tiles.values()):
            try:
                return build_face_batches(blocks, faces)
            except MemoryError:
                # The cache is optional; the topology can still use the face slices.
                return None
        return None

    def _tile_layout(self, key: Cell, region: tuple[slice, ...], selected: np.ndarray) -> list[Block]:
        parts = partition(selected)
        new_blocks = []
        origin = tuple(r.start for r in region)
        for local, level in parts:
            start = tuple(a + s.start for a, s in zip(origin, local))
            absolute = tuple(slice(a + s.start, a + s.stop) for a, s in zip(origin, local))
            material = self._materials[absolute]
            if level:
                material = subdivide(material)
                for cell, children in self._fine_materials.get(key, {}).items():
                    if all(r.start <= i < r.stop for r, i in zip(absolute, cell)):
                        child_region = tuple(slice(2 * (i - s), 2 * (i - s) + 2) for i, s in zip(cell, start))
                        material[child_region] = children
            block_key = key if len(parts) == 1 else (*key, *start)
            new_blocks.append(Block(block_key, start, level, material, {}))
        return new_blocks

    def _rebuild_tile(self, key: Cell, region: tuple[slice, ...], selected: np.ndarray) -> tuple[Block, ...]:
        new_blocks = self._tile_layout(key, region, selected)
        origin = tuple(r.start for r in region)
        old_blocks = self._tiles[key]
        if len(old_blocks) == len(new_blocks) == 1:
            old, new = old_blocks[0], new_blocks[0]
            new.fields = {name: subdivide(amounts / 8) if new.level else restrict(amounts) * 8
                          for name, amounts in old.fields.items()}
            return (new,)

        def fine_region(block: Block) -> tuple[slice, ...]:
            return tuple(slice(2 * (r.start - o), 2 * (r.stop - o)) for r, o in zip(block.region, origin))

        old_regions = [(block, fine_region(block)) for block in old_blocks]
        new_regions = [(block, fine_region(block)) for block in new_blocks]
        if self.field_specs:
            scratch = np.empty(tuple(2 * n for n in selected.shape))
            for name in self.field_specs:
                for block, area in old_regions:
                    values = block.fields[name]
                    scratch[area] = values if block.level else subdivide(values / 8)
                for block, area in new_regions:
                    values = scratch[area]
                    block.fields[name] = values.copy() if block.level else restrict(values) * 8
        return tuple(new_blocks)

    def adapt(
        self,
        tolerances: Mapping[str, Tolerance],
        *,
        pinned_cells: Iterable[Cell] = (),
        pinned_blocks: Iterable[BlockKey] = (),
        buffer_cells: int = 0,
        coarsen_after: int = 3,
        max_refined_cells: int | None = None,
        max_refined_blocks: int | None = None,
    ) -> None:
        """Select base cells by local projection loss, independent of storage geometry."""
        if not tolerances or not tolerances.keys() <= self.field_specs.keys():
            raise ValueError("adapt requires tolerances for existing fields")
        if not isinstance(coarsen_after, Integral) or not 1 <= coarsen_after <= 65535:
            raise ValueError("coarsen_after must be in [1, 65535]")
        if not isinstance(buffer_cells, Integral) or buffer_cells < 0:
            raise ValueError("buffer_cells must be a nonnegative integer")
        pinned = np.zeros_like(self._refinement)
        self._mark_cells(pinned, pinned_cells)
        self._mark_blocks(pinned, pinned_blocks)
        self._mark_cells(pinned, self._geometry_pins)
        pinned_count = int(pinned.sum())
        if max_refined_cells is not None and (not isinstance(max_refined_cells, Integral)
                                              or max_refined_cells < pinned_count):
            raise ValueError("refinement budget must accommodate all pinned cells")

        pinned_tiles = set()
        if max_refined_blocks is not None:
            pinned_tiles = {key for key, region in self._root_regions.items() if pinned[region].any()}
            if not isinstance(max_refined_blocks, Integral) or max_refined_blocks < len(pinned_tiles):
                raise ValueError("refinement budget must accommodate all pinned tiles")

        scores, detail = self._projection_scores(tolerances)
        quiet = np.where(self._refinement & (scores < 0.5), self._quiet_checks + 1, 0)
        marked = pinned | (scores > 1)
        pairs = tuple(self._neighbor_regions())
        for _ in range(buffer_cells):
            expanded = marked.copy()
            for low, high in pairs:
                expanded[low] |= marked[high]
                expanded[high] |= marked[low]
            marked = expanded
        chosen = marked | (self._refinement & (quiet < coarsen_after))
        requested = int(chosen.sum())

        if max_refined_blocks is not None:
            ranked = [(float(scores[region].max(where=chosen[region], initial=0)), key)
                      for key, region in self._root_regions.items()
                      if key not in pinned_tiles and chosen[region].any()]
            ranked.sort(key=lambda item: (-item[0], item[1]))
            for _, key in ranked[max_refined_blocks - len(pinned_tiles):]:
                chosen[self._root_regions[key]] = False
        if max_refined_cells is not None and chosen.sum() > max_refined_cells:
            candidates = np.flatnonzero(chosen & ~pinned)
            ranked = np.argsort(-scores.ravel()[candidates], kind="stable")
            chosen.flat[candidates[ranked[max_refined_cells - pinned_count:]]] = False

        at_limit = chosen & self._refinement & (detail > 1)
        self._regrid(chosen)
        self._quiet_checks = np.where(chosen & ~marked, quiet, 0)
        if requested > int(chosen.sum()):
            self._record("mesh", "refinement_budget", skipped_cells=requested - int(chosen.sum()))
        if at_limit.any():
            self._record("mesh", "refinement_level_limit", cells=int(at_limit.sum()),
                         highest_score=float(detail.max(where=at_limit, initial=0)))

    def _neighbor_regions(self):
        for axis, (low, high) in enumerate(adjacent_cells()):
            if self.grid.shape[axis] == 1:
                continue
            yield low, high
            if self._periodic[axis]:
                last, first = list(low), list(high)
                last[axis], first[axis] = -1, 0
                yield tuple(last), tuple(first)

    def _projection_scores(self, tolerances: Mapping[str, Tolerance]) -> tuple[Field, Field]:
        scores = np.zeros(self.grid.shape)
        detail = np.zeros_like(scores)
        for name, tol in tolerances.items():
            spec = self.field_specs[name]
            material, porous = spec.material, spec.porous
            base_density = np.empty_like(scores)
            presence = np.empty(self.grid.shape, dtype=bool) if self._fine_materials and material is not None else None
            for block in self._blocks.values():
                density = block.fields[name] / self._volume(block)
                if porous is not None:
                    density /= np.where(block.materials == porous.material, porous.volume_fraction, 1.0)
                region = block.region
                local_presence = block.materials == material if presence is not None else None
                if block.level:
                    averaged = restrict(density)
                    if local_presence is not None:
                        fraction = restrict(local_presence)
                        np.divide(averaged, fraction, out=averaged, where=fraction > 0)
                    reconstructed = subdivide(averaged)
                    error = np.abs(density - reconstructed) / (
                        tol.absolute + tol.relative * np.maximum(np.abs(density), np.abs(reconstructed)))
                    if local_presence is not None:
                        error *= local_presence
                    x, y, z = averaged.shape
                    loss = error.reshape(x, 2, y, 2, z, 2).max(axis=(1, 3, 5))
                    np.maximum(detail[region], loss, out=detail[region])
                    density = averaged
                if presence is not None:
                    presence[region] = fraction > 0 if block.level else local_presence
                base_density[region] = density
            # Project each adjacent pair to its mean. Each mark refers to the two base cells alone.
            for low, high in self._neighbor_regions():
                left, right = base_density[low], base_density[high]
                loss = (0.5 * np.abs(left - right) /
                        (tol.absolute + tol.relative * np.maximum(np.abs(left), np.abs(right))))
                if material is not None:
                    if presence is None:
                        loss *= (self._materials[low] == material) & (self._materials[high] == material)
                    else:
                        loss *= presence[low] & presence[high]
                np.maximum(scores[low], loss, out=scores[low])
                np.maximum(scores[high], loss, out=scores[high])
        return np.maximum(scores, detail, out=scores), detail

    def _transport_step_limit(self) -> float:
        spacing = self.grid.spacing_m / (2 if self._refinement.any() else 1)
        velocity_sum = sum(map(abs, self._velocity_m_s))
        inverse_step = max(((12 if spec.porous else 6) * spec.diffusivity_m2_s / spacing ** 2
                            + (velocity_sum / spacing if spec.advected else 0.0)
                            for spec in self.field_specs.values()), default=0.0)
        return 0.8 / inverse_step if inverse_step else inf

    def step(self, dt_s: float) -> None:
        end = self.time_s + dt_s
        self._check_ready()
        if not isfinite(end) or end <= self.time_s:
            raise ValueError("dt_s must advance the finite world time")
        source = self._solver_source
        self._stepping = True
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                for source, module in self._modules.items():
                    module.step(self, source, dt_s)
                source = self._solver_source
                self._advance_physics(dt_s, end)
        except Exception as error:
            self._failed = True
            self._record(source, "simulation_error", error=type(error).__name__, message=str(error))
            raise
        finally:
            self._stepping = False

    def _advance_physics(self, dt_s: float, end: float) -> None:
        start = self.time_s
        count = max(1, ceil(dt_s / self._transport_step_limit()))
        duration = dt_s / count
        for index in range(1, count + 1):
            for rule in self._chemistry:
                rule.advance(self, duration)
            for name, spec in self.field_specs.items():
                if spec.diffusivity_m2_s or (spec.advected and any(self._velocity_m_s)):
                    self._transport(name, spec, duration)
            self._time_s = end if index == count else start + index * duration

    def _transport(self, name: str, spec: FieldSpec, dt_s: float) -> None:
        plan = None if spec.porous or spec.isolated_materials_from is not None else self._transport_batches
        speeds = self._velocity_m_s if spec.advected else (0.0, 0.0, 0.0)
        if plan is None:
            densities = {key: block.fields[name] / self._volume(block) for key, block in self._blocks.items()}
            changes = {key: np.zeros_like(density) for key, density in densities.items()}
            remaining = self._faces
        else:
            density = np.empty(plan.cells)
            change_buffer = np.zeros_like(density)
            densities, changes = {}, {}
            for key, block in self._blocks.items():
                region = plan.regions[key]
                values = block.fields[name]
                densities[key] = density[region].reshape(values.shape)
                np.divide(values, self._volume(block), out=densities[key])
                changes[key] = change_buffer[region].reshape(values.shape)
            plan.accumulate(density, change_buffer, speeds, spec.diffusivity_m2_s, spec.material, dt_s)
            remaining = plan.remaining
        for face in remaining:
            velocity = speeds[face.axis]
            if not (spec.diffusivity_m2_s or velocity):
                continue
            low, high = face.low, face.high
            low_density, high_density = low.read(densities[low.block]), high.read(densities[high.block])
            transported = low_density if velocity >= 0 else high_density
            flux = velocity * transported
            if spec.diffusivity_m2_s:
                if spec.porous is None:
                    flux += spec.diffusivity_m2_s / face.center_distance_m * (low_density - high_density)
                else:
                    flux = self._porous_flux(spec, face, low_density, high_density)
            if spec.material is not None:
                flux *= ((low.read(self._blocks[low.block].materials) == spec.material)
                         & (high.read(self._blocks[high.block].materials) == spec.material))
            if spec.isolated_materials_from is not None:
                left_material = low.read(self._blocks[low.block].materials)
                right_material = high.read(self._blocks[high.block].materials)
                flux *= (left_material >= spec.isolated_materials_from) & (left_material == right_material)
            transfer = flux * face.area_m2 * dt_s
            low.add(changes[low.block], -transfer)
            high.add(changes[high.block], transfer)
        for key, change in changes.items():
            self._blocks[key].fields[name] += change

    def _porous_flux(self, spec: FieldSpec, face: Face, left: Field, right: Field) -> Field:
        porous = spec.porous
        low, high = face.low, face.high
        left_block, right_block = self._blocks[low.block], self._blocks[high.block]
        left_porous = low.read(left_block.materials) == porous.material
        right_porous = high.read(right_block.materials) == porous.material
        left_fraction = np.where(left_porous, porous.volume_fraction, 1.0)
        right_fraction = np.where(right_porous, porous.volume_fraction, 1.0)
        resistance = porous.resistance_multiplier
        half_base = self.grid.spacing_m / 2
        distance = (half_base / 2 ** left_block.level * np.where(left_porous, resistance, 1.0)
                    + half_base / 2 ** right_block.level * np.where(right_porous, resistance, 1.0))
        flux = (spec.diffusivity_m2_s / distance) * (left / left_fraction - right / right_fraction)
        if porous.excluded_material_from is not None:
            flux *= ((low.read(left_block.materials) < porous.excluded_material_from)
                     & (high.read(right_block.materials) < porous.excluded_material_from))
        return flux
