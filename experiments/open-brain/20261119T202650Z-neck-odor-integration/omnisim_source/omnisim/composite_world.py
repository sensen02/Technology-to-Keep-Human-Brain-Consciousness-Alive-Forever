"""Fixed multilevel Cartesian boxes using the existing HIP finite-volume kernels."""

from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from .chemistry import SurfaceDissolution
from .engine import FieldSpec, _Topology
from .gpu_world import GpuWorld, _GpuState
from .mesh import partition
from .native_transport import FrozenMesh
from .world import Grid


@dataclass
class _Box:
    key: tuple
    level: int
    lower: tuple
    stride: int
    materials: np.ndarray
    fields: dict = field(default_factory=dict)

    @property
    def upper(self):
        return tuple(a + self.stride * n for a, n in zip(self.lower, self.materials.shape))


class CompositeGpuWorld(GpuWorld):
    """Use static, disjoint boxes with shared finite-volume faces across levels."""

    def __init__(self, grid, materials, *, levels, select=None, materialize=None, max_cells=4_000_000,
                 leaf_limit=None, stream_initial=False, archived_tiles=None, **gpu):
        if (not levels or levels[0] != 0 or any(not isinstance(level, int) for level in levels)
                or any(a >= b for a, b in zip(levels, levels[1:]))):
            raise ValueError("levels must be strictly increasing integers starting at zero")
        if max_cells < 1:
            raise ValueError("max_cells must be positive")
        if leaf_limit is not None and (not isinstance(leaf_limit, int) or leaf_limit < 1):
            raise ValueError("leaf_limit must be a positive integer")
        if (archived_tiles is None) == (select is None):
            raise ValueError("select a layout or supply archived tiles")
        super().__init__(grid, materials=materials, **gpu)
        self._scale = 1 << levels[-1]
        self._levels, self._max_cells = tuple(levels), max_cells
        self.finest_spacing_m = grid.spacing_m / self._scale
        per_root, lookup, boxes = {key: [] for key in self._root_regions}, {}, {}
        refined = np.zeros(grid.shape, dtype=bool)
        count = 0

        def register(root, lo, level, stride, local):
            nonlocal count
            count += local.size
            if count > max_cells:
                raise MemoryError(f"composite topology exceeds {max_cells:,} active cells before HIP allocation")
            key = (*root, level, *lo)
            if key in boxes:
                raise ValueError("duplicate composite tile")
            block = _Box(key, level, lo, stride, local)
            boxes[key] = block
            per_root[root].append(block)

        def store(root, lo, size, level, stride):
            indices = np.ix_(*[(a + stride * np.arange(n)) // self._scale for a, n in zip(lo, size)])
            local = self._materials[indices]
            if materialize is not None:
                local = np.asarray(materialize(self._local_grid(lo, size, stride), local))
                if (local.shape != size or local.dtype.kind not in "iu" or np.any(local < 0)
                        or np.any(local > np.iinfo(np.int32).max)):
                    raise ValueError("invalid composite material array")
                local = local.astype(np.int32, copy=False)
            register(root, lo, level, stride, local)

        def fill(root, lower, shape, stage):
            level = levels[stage]
            stride = self._scale >> level
            subgrid = self._local_grid(lower, shape, stride)
            parts = (partition(np.broadcast_to(np.asarray(select(subgrid, levels[stage + 1]), dtype=bool), shape))
                     if stage + 1 < len(levels) else [(tuple(slice(0, n) for n in shape), 0)])
            for region, split in parts:
                lo = tuple(a + stride * r.start for a, r in zip(lower, region))
                size = tuple(r.stop - r.start for r in region)
                if split:
                    ratio = 1 << (levels[stage + 1] - level)
                    if not level:
                        coarse = tuple(slice(a // self._scale, a // self._scale + n) for a, n in zip(lo, size))
                        refined[coarse] = True
                    fill(root, lo, tuple(n * ratio for n in size), stage + 1)
                    continue
                if leaf_limit is None or max(size) <= leaf_limit:
                    store(root, lo, size, level, stride)
                    continue
                tiling = tuple((n + leaf_limit - 1) // leaf_limit for n in size)
                for tile in np.ndindex(tiling):
                    offset = tuple(i * leaf_limit for i in tile)
                    sub_lo = tuple(a + stride * i for a, i in zip(lo, offset))
                    sub_size = tuple(min(leaf_limit, n - i) for n, i in zip(size, offset))
                    store(root, sub_lo, sub_size, level, stride)

        if archived_tiles is None:
            for root, region in self._root_regions.items():
                lower = tuple(r.start * self._scale for r in region)
                fill(root, lower, tuple(r.stop - r.start for r in region), 0)
        else:
            for level, lo, local in archived_tiles:
                if (level not in levels or len(lo) != 3 or local.ndim != 3 or not local.size
                        or local.dtype != np.int32 or np.any(local < 0)):
                    raise ValueError("invalid archived tile")
                stride = self._scale >> level
                root = tuple(i // (self._scale * self._block_cells) for i in lo)
                region = self._root_regions.get(root)
                upper = tuple(a + stride * n for a, n in zip(lo, local.shape))
                if region is None or any(a % stride or b > r.stop * self._scale
                                         for a, b, r in zip(lo, upper, region)):
                    raise ValueError("archived tile exceeds its root")
                register(root, lo, level, stride, local)
                if level:
                    cells = tuple(slice(a // self._scale, (b - 1) // self._scale + 1) for a, b in zip(lo, upper))
                    refined[cells] = True
        for root, region in self._root_regions.items():
            group = per_root[root]
            lower = np.asarray([b.lower for b in group])
            upper = np.asarray([b.upper for b in group])
            if archived_tiles is not None:
                volume = sum(b.materials.size * b.stride**3 for b in group)
                expected = self._scale**3
                for axis in region:
                    expected *= axis.stop - axis.start
                if volume != expected:
                    raise ValueError("archived tiles do not cover their root volume")
                # The correct volume would still hide holes if some boxes overlapped.
                for start in range(0, len(group), 32):
                    region_ids = slice(start, start + 32)
                    overlap = np.all((lower[region_ids, None] < upper)
                                     & (upper[region_ids, None] > lower), axis=2)
                    if not np.all(overlap.sum(axis=1) == 1):
                        raise ValueError("archived tiles overlap")
            lookup[root] = (group, lower, upper)
        self._lookup = lookup
        self._actual_count = count
        self._stream_initial = stream_initial
        self._initial_density = {}
        self._equilibrium = {}
        self._material_volume = {}
        self._topology = _Topology({k: tuple(v) for k, v in per_root.items()}, boxes, object(), refined)
        self._material_spacing = {}
        for box in boxes.values():
            h = box.stride * self.finest_spacing_m
            ids, counts = np.unique(box.materials, return_counts=True)
            for material_id, cells in zip(ids, counts):
                self._material_spacing[material_id] = min(h, self._material_spacing.get(material_id, h))
                self._material_volume[material_id] = self._material_volume.get(material_id, 0.0) + cells * h**3

    @property
    def active_cells(self):
        return self._actual_count

    def _local_grid(self, lower, shape, stride):
        start = tuple(x + i * self.finest_spacing_m for x, i in zip(self.grid.origin_m, lower))
        return Grid(shape, self.finest_spacing_m * stride, start)

    def block_grid(self, key):
        box = self._blocks[key]
        return self._local_grid(box.lower, box.materials.shape, box.stride)

    def _cell_at(self, position_m):
        base = self.grid.cell_at(position_m)
        root = tuple(i // self._block_cells for i in base)
        point = (np.asarray(position_m) - self.grid.origin_m) / self.finest_spacing_m
        point = np.clip(point, np.asarray(base) * self._scale,
                        np.nextafter((np.asarray(base) + 1) * self._scale, -np.inf))
        group, lo, hi = self._lookup[root]
        index = np.flatnonzero(np.all((point >= lo) & (point < hi), axis=1)).item()
        block = group[index]
        local = tuple(np.floor((point - lo[index]) / block.stride).astype(int))
        return block, local

    def add_field(self, name, spec=FieldSpec(), *, initial_density=0):
        self._restore_host()
        if name in self.field_specs:
            raise ValueError(f"field already exists: {name}")
        density = np.broadcast_to(np.asarray(initial_density, dtype=np.float64), self.grid.shape)
        if not np.isfinite(density).all() or (spec.nonnegative and np.any(density < 0)):
            raise ValueError(f"invalid initial density for {name}")
        if self._stream_initial:
            if np.max(np.abs(density)) > np.finfo(float).max / max(1.0, self.grid.cell_volume_m3):
                raise ValueError(f"initial density overflows for {name}")
            self._specs[name] = spec
            self._initial_density[name] = density.copy()
            return
        self._specs[name] = spec
        try:
            values = {}
            for key, block in self._blocks.items():
                indices = np.ix_(*[(a + block.stride * np.arange(n)) // self._scale
                                    for a, n in zip(block.lower, block.materials.shape)])
                values[key] = self._densities_to_amounts(name, block, density[indices])
        except BaseException:
            del self._specs[name]
            raise
        for key, block in self._blocks.items():
            block.fields[name] = values[key]

    def _initial_block(self, name, block):
        if name in block.fields:
            return block.fields[name]
        # Split the substitution into block-local batches to keep host setup bounded.
        indices = np.ix_(*[(a + block.stride * np.arange(n)) // self._scale
                            for a, n in zip(block.lower, block.materials.shape)])
        return self._densities_to_amounts(name, block, self._initial_density[name][indices])

    def _upload_initial_fields(self, state, groups):
        if not self._initial_density:
            super()._upload_initial_fields(state, groups)
            return
        for key, block in self._blocks.items():
            for first, *others in groups:
                values = np.ascontiguousarray(self._initial_block(first, block)).ravel()
                if others:
                    values = values.copy()
                    for name in others:
                        values += self._initial_block(name, block).ravel()
                state.write(first, state.mesh._regions[key], values)

    def initialize_device(self):
        """Upload the registered dynamic fields before running or sampling the GPU world."""
        self._check_ready()
        if self._device_field_specs():
            self._activate()

    def add_reaction(self, rule):
        if isinstance(rule, SurfaceDissolution):
            raise NotImplementedError("surface dissolution plans are not built for composite grids")
        if hasattr(rule, "reactants") and self._equilibrium.keys() & (rule.reactants.keys() | rule.products.keys()):
            raise ValueError("an equilibrium field cannot participate in reactions")
        super().add_reaction(rule)

    def add_equilibrium_field(self, name, spec, concentration):
        """Keep a prescribed resting solution by material without allocating per-voxel state."""
        self._check_open()
        if name in self.field_specs:
            raise ValueError(f"field already exists: {name}")
        if spec.diffusivity_m2_s or spec.advected or not isfinite(concentration) or concentration < 0:
            raise ValueError("an equilibrium field requires a nonnegative concentration and no transport")
        medium = spec.porous
        densities = {material: concentration * (medium.volume_fraction if medium and material == medium.material else 1)
                     if spec.available_in(material) else 0.0 for material in self._material_volume}
        self._equilibrium[name] = densities
        self._specs[name] = spec

    def _device_field_specs(self):
        return {name: spec for name, spec in self.field_specs.items() if name not in self._equilibrium}

    def sample(self, name, position_m):
        if name not in self._equilibrium:
            if self._gpu is None and name in self._initial_density:
                self._check_open()
                block, cell = self._cell_at(position_m)
                return float(self._initial_block(name, block)[cell] / self._volume(block))
            return super().sample(name, position_m)
        self._check_open()
        return float(self._equilibrium[name][self.material_at(position_m)])

    def block_amounts(self, name, key):
        if name not in self._equilibrium:
            if self._gpu is None and name in self._initial_density:
                self._check_open()
                return self._initial_block(name, self._blocks[key]).copy()
            return super().block_amounts(name, key)
        self._check_open()
        block = self._blocks[key]
        ids, inverse = np.unique(block.materials, return_inverse=True)
        values = np.array([self._equilibrium[name][material] for material in ids])
        return values[inverse].reshape(block.materials.shape) * self._volume(block)

    def total(self, name):
        if name not in self._equilibrium:
            if self._gpu is None and name in self._initial_density:
                self._check_open()
                return sum(float(self._initial_block(name, block).sum()) for block in self._blocks.values())
            return super().total(name)
        self._check_open()
        return float(sum(volume * self._equilibrium[name][material]
                         for material, volume in self._material_volume.items()))

    def set_block_densities(self, name, key, density):
        if name in self._equilibrium:
            raise ValueError("an equilibrium field is read-only")
        if self._gpu is None and name in self._initial_density:
            self._check_open()
            block = self._blocks[key]
            block.fields[name] = self._densities_to_amounts(name, block, density)
            return
        super().set_block_densities(name, key, density)

    def add_amounts(self, position_m, changes):
        if self._equilibrium.keys() & changes.keys():
            raise ValueError("an equilibrium field is read-only")
        if self._gpu is None and self._initial_density.keys() & changes.keys():
            self._check_open()
            block, _ = self._cell_at(position_m)
            for name in (self._initial_density.keys() & changes.keys()) - block.fields.keys():
                block.fields[name] = self._initial_block(name, block)
        super().add_amounts(position_m, changes)

    def _advance_physics(self, dt_s, end):
        if len(self.field_specs) == len(self._equilibrium):
            self._time_s = end
            return
        super()._advance_physics(dt_s, end)

    def _regrid(self, selected):
        raise NotImplementedError("this composite topology is fixed after initialization")

    @staticmethod
    def _group_cells(array, factor):
        shape = tuple(part for size in array.shape for part in (size // factor, factor))
        return array.reshape(shape)

    def _pinned_tile(self, key):
        return False

    def adapt(self, regions, *, coarsen_outside=True):
        """Change one level per leaf around world-space boxes, preserving each material's amount.

        Mixed material voxels cannot merge. Both device meshes must fit the budget until
        the transfer succeeds. On failure the current topology and fields stay available.
        """
        self._check_ready()
        selected = {key for lower, upper in regions for key in self.blocks_overlapping(lower, upper)}
        previous = tuple(self._blocks.values())
        target = []
        count = 0
        for block in previous:
            level, material = block.level, block.materials
            stage = self._levels.index(level)
            factor = 1
            if not self._pinned_tile(block.key):
                if block.key in selected and stage + 1 < len(self._levels):
                    level = self._levels[stage + 1]
                    factor = 1 << (level - block.level)
                elif block.key not in selected and coarsen_outside and stage:
                    parent = self._levels[stage - 1]
                    ratio = 1 << (level - parent)
                    if not any(i % (block.stride * ratio) for i in block.lower) and not any(
                            n % ratio for n in material.shape):
                        grouped = self._group_cells(material, ratio)
                        lower = grouped.min(axis=(1, 3, 5))
                        if np.array_equal(lower, grouped.max(axis=(1, 3, 5))):
                            level, material = parent, lower
            count += material.size * factor**3
            if count > self._max_cells:
                raise MemoryError("adapted composite topology exceeds its cell limit")
            target.append((level, block.lower, material, factor))
        if all(level == old.level for (level, *_), old in zip(target, previous)):
            return False
        specs = self._device_field_specs()
        if specs:
            self.initialize_device()
            available = self.gpu_budget_bytes - self.resident_gpu_bytes
            # The actual topology and solver scratch are accounted for by HIP allocation.
            if count * (4 + 8 * self.gpu_field_buffers) > available:
                raise MemoryError("composite remap needs headroom for both device meshes")

        def tiles():
            for level, lower, material, factor in target:
                if factor > 1:
                    for axis in range(3):
                        material = material.repeat(factor, axis=axis)
                yield level, lower, material

        candidate = CompositeGpuWorld(self.grid, self._materials, levels=self._levels, archived_tiles=tiles(),
                                      max_cells=self._max_cells, block_cells=self._block_cells,
                                      build=self.gpu_build, device=self.gpu_device,
                                      reserve_bytes=self.gpu_reserve_bytes, pack_material_fields=self._pack_material_fields)
        replacement = None
        if specs:
            candidate._specs.update(specs)
            candidate._chemistry.extend(self._chemistry)
            try:
                replacement = _GpuState(candidate, available, upload=False)
                for old, new in zip(previous, candidate._blocks.values()):
                    mapped = {}
                    for name in specs:
                        amounts = self.block_amounts(name, old.key)
                        if new.level > old.level:
                            ratio = 1 << (new.level - old.level)
                            amounts /= ratio**3
                            for axis in range(3):
                                amounts = amounts.repeat(ratio, axis=axis)
                        elif new.level < old.level:
                            ratio = 1 << (old.level - new.level)
                            amounts = self._group_cells(amounts, ratio).sum(axis=(1, 3, 5))
                        field_id = replacement.ids[name]
                        if field_id in mapped:
                            mapped[field_id][1][:] += amounts
                        else:
                            mapped[field_id] = (name, amounts)
                    for name, amounts in mapped.values():
                        replacement.write(name, replacement.mesh._regions[new.key], np.ascontiguousarray(amounts).ravel())
            except BaseException:
                if replacement is not None:
                    replacement.close()
                raise
        if self._gpu is not None:
            self._peak_gpu_bytes = max(self.peak_gpu_bytes, self.resident_gpu_bytes + replacement.allocated_bytes)
            self._gpu.close()
        self._gpu = replacement
        self._lookup, self._topology = candidate._lookup, candidate._topology
        self._material_spacing, self._material_volume = candidate._material_spacing, candidate._material_volume
        self._actual_count = candidate._actual_count
        self._mesh_revision += 1
        self._record("mesh", "regrid", active_cells=self.active_cells)
        return True

    def _transport_step_limit(self):
        velocity = sum(map(abs, self._velocity_m_s))
        inverse = 0.0
        for spec in self.field_specs.values():
            if not spec.diffusivity_m2_s and not spec.advected:
                continue
            for material, h in self._material_spacing.items():
                if not spec.available_in(material):
                    continue
                effective_d = spec.diffusivity_m2_s
                if spec.porous and material == spec.porous.material:
                    effective_d /= spec.porous.tortuosity ** 2
                # The conductance of a face is bounded by its own half-cell resistance.
                local = 12 * effective_d / h**2 + (velocity / h if spec.advected else 0)
                inverse = max(inverse, local)
        return 0.8 / inverse if inverse else float("inf")

    def _device_mesh(self, topology, library):
        if not hasattr(library, "omni_composite_transport_supported"):
            raise RuntimeError("rebuild the HIP library for composite grids")
        return FrozenMesh(self, topology=topology, explicit_faces=self._pack_faces)

    def _pack_faces(self, offsets):
        blocks = list(self._blocks.values())
        lowers, uppers = np.asarray([b.lower for b in blocks]), np.asarray([b.upper for b in blocks])
        edges, geometries, geometry_ids = [], [], {}
        for axis in range(3):
            tangent = [i for i in range(3) if i != axis]
            starts, ends = {}, {}
            for i, (low, high) in enumerate(zip(lowers[:, axis], uppers[:, axis])):
                starts.setdefault(low, []).append(i)
                ends.setdefault(high, []).append(i)
            for plane, left in ends.items():
                right = starts.get(plane)
                if not right:
                    continue
                right = np.asarray(right)
                for i in left:
                    crossing = ((lowers[right][:, tangent] < uppers[i, tangent])
                                & (uppers[right][:, tangent] > lowers[i, tangent])).all(axis=1)
                    for j in right[crossing]:
                        a, b = blocks[i], blocks[j]
                        start, stop = np.maximum(lowers[i], lowers[j]), np.minimum(uppers[i], uppers[j])
                        stride = min(a.stride, b.stride)
                        ranges = [np.arange(start[k], stop[k], stride) if k != axis else None for k in range(3)]
                        dimensions = (axis, a.level, b.level)
                        gid = geometry_ids.setdefault(dimensions, len(geometry_ids))
                        if gid == len(geometries):
                            area = (stride * self.finest_spacing_m) ** 2
                            distance = (a.stride + b.stride) * self.finest_spacing_m / 2
                            geometries.append((axis, self._volume(a), self._volume(b), area, distance))
                        atoms = []
                        for box, inward in ((a, a.materials.shape[axis] - 1), (b, 0)):
                            coordinates = [((r - box.lower[k]) // box.stride if k != axis else np.asarray([inward]))
                                           for k, r in enumerate(ranges)]
                            xyz = np.ix_(*coordinates)
                            ny, nz = box.materials.shape[1:]
                            atoms.append(((xyz[0] * ny + xyz[1]) * nz + xyz[2] + offsets[box.key]).ravel())
                        pairs = np.empty((len(atoms[0]), 3), dtype=np.int32)
                        pairs[:, 0], pairs[:, 1], pairs[:, 2] = *atoms, gid
                        edges.append(pairs)
        return (np.concatenate(edges) if edges else np.empty((0, 3), dtype=np.int32)), geometries
