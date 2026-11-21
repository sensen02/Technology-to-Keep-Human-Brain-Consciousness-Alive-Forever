"""GPU-resident finite volumes with CPU-owned adaptive topology."""

from __future__ import annotations

import ctypes as ct
from math import ceil, isfinite, prod
from pathlib import Path

import numpy as np

from .chemistry import BulkReaction, SurfaceDissolution
from .engine import AdaptiveWorld, FieldSpec
from .mesh import FaceGroups
from .native_transport import DEFAULT_BUILD, FrozenMesh, HipTransport, Parameters, _load, _require_porous_diffusion

_SURFACE_FACE = np.dtype([('solid', 'i4'), ('liquid', 'i4'), ('area', 'f8')], align=True)
_SURFACE_CELL = np.dtype([('cell', 'i4'), ('liquid', 'i4'), ('volume', 'f8')], align=True)
_SCORE_FIELD = np.dtype([('field', 'i4'), ('material', 'i4'), ('absolute', 'f8'), ('relative', 'f8')], align=True)
_POROUS_SCORE_FIELD = np.dtype(_SCORE_FIELD.descr + [('porous_material', 'i4'), ('porous_fraction', 'f8')], align=True)


def _surface_plan(world, mesh, library, rule):
    solid = world.field_specs[rule.solid_species].material
    liquid = world.field_specs[rule.liquid_species].material
    pairs = mesh._surface_pairs(library, solid, liquid)
    n = len(pairs)
    if not n:
        return None
    endpoints = np.concatenate((pairs['solid'], pairs['liquid']))
    order = np.argsort(endpoints, kind='stable')
    sorted_cells = endpoints[order]
    starts = np.r_[0, np.flatnonzero(sorted_cells[:-1] != sorted_cells[1:]) + 1]
    first = order[starts]
    inverse = np.empty(len(endpoints), dtype=np.int32)
    inverse[order] = np.repeat(np.arange(len(starts), dtype=np.int32), np.diff(starts, append=len(endpoints)))
    rows = np.empty(len(starts), dtype=_SURFACE_CELL)
    is_liquid = first >= n
    rows['cell'], rows['liquid'] = sorted_cells[starts], is_liquid
    rows['volume'] = 0
    rows['volume'][is_liquid] = pairs['volume'][first[is_liquid] - n]
    faces = np.empty(n, dtype=_SURFACE_FACE)
    faces['solid'], faces['liquid'], faces['area'] = inverse[:n], inverse[n:], pairs['area']
    offsets = np.r_[starts, len(endpoints)].astype(np.int32)
    incidences = (order % n).astype(np.int32)
    return faces, rows, offsets, incidences


def _remap_plan(old, new, old_tiles, new_tiles, library):
    spans = []
    src = dst = 0
    for key, previous in old_tiles.items():
        current = new_tiles[key]
        spans.append((src, src + len(previous), dst, dst + len(current), previous is current))
        src += len(previous)
        dst += len(current)
    spans = np.asarray(spans, dtype=np.int32)
    counts = (ct.c_int32 * 2)()
    pack = library.omni_build_remap
    pack.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_int32))
    args = (old._locations.ctypes.data, new._locations.ctypes.data, spans.ctypes.data, len(spans))
    if pack(*args, None, None, counts):
        raise ValueError('remap exceeds the GPU index range')
    boxes = np.empty((counts[0], 13), dtype=np.int32)
    chunks = np.empty((counts[1], 2), dtype=np.int32)
    pack(*args, boxes.ctypes.data, chunks.ctypes.data, counts)
    return boxes, chunks


def _field_groups(specs, pack_material_fields):
    """Reuse a slot only across fields defined on pairwise disjoint materials."""
    groups, material_slots, count_by_material = [], [], {}
    for name, spec in specs.items():
        material = spec.material
        if material is None or not pack_material_fields:
            groups.append([name])
            continue
        rank = count_by_material.get(material, 0)
        count_by_material[material] = rank + 1
        if rank == len(material_slots):
            material_slots.append(len(groups))
            groups.append([])
        groups[material_slots[rank]].append(name)
    return groups


class _GpuState(HipTransport):
    def __init__(self, world, budget, upload, topology=None):
        groups = _field_groups(world._device_field_specs(), world._pack_material_fields)
        self.field_buffers = len(groups)
        self.ids = {name: i for i, names in enumerate(groups) for name in names}
        self.shared_materials = {name: world.field_specs[name].material for names in groups
                                 if len(names) > 1 for name in names}
        library = _load(world.gpu_build, 'hip')
        if ((self.shared_materials or world._fine_materials)
                and not hasattr(library, 'omni_hip_total_material')):
            raise RuntimeError('rebuild the HIP library for packed fields or fine material geometry')
        if any(spec.porous for spec in world.field_specs.values()):
            _require_porous_diffusion(library)
        if (any(spec.isolated_materials_from is not None
                or (spec.porous and spec.porous.excluded_material_from is not None)
                for spec in world.field_specs.values())
                and not hasattr(library, "omni_composite_transport_supported")):
            raise RuntimeError("rebuild the HIP library for composite material transport")
        mesh = world._device_mesh(topology, library)
        super().__init__(mesh, world.gpu_build, device=world.gpu_device, budget_bytes=budget,
                         reserve_bytes=world.gpu_reserve_bytes, _fields=self.field_buffers, _library=library)
        self.mesh._release_uploaded_buffers()
        c = ct.c_void_p
        i = ct.c_int
        d = ct.c_double
        s = ct.POINTER(ct.c_size_t)
        lib = self._library
        lib.omni_hip_upload_field.argtypes = lib.omni_hip_download_field.argtypes = (c, i, i, i, c)
        lib.omni_hip_total.argtypes = (c, i, ct.POINTER(d))
        if self.shared_materials:
            lib.omni_hip_total_material.argtypes = (c, i, i, ct.POINTER(d))
        lib.omni_hip_run_field.argtypes = (c, i, ct.POINTER(Parameters), i, ct.POINTER(ct.c_float))
        lib.omni_hip_add_surface.argtypes = (c, c, i, c, i, c, c, s)
        lib.omni_hip_surface.argtypes = (c, i, i, i, d, d, d)
        lib.omni_hip_add_bulk.argtypes = (c, c, i, s)
        lib.omni_hip_bulk.argtypes = (c, i, i, d, d)
        lib.omni_hip_finish.argtypes = (c, ct.POINTER(i))
        lib.omni_hip_remap.argtypes = (c, c, c, i, c, i, s)
        lib.omni_hip_scores.argtypes = (c, c, c, c, i, c, c, s)
        self.rules = []
        try:
            if upload:
                world._upload_initial_fields(self, groups)
            surface_id = bulk_id = 0
            for rule in world._chemistry:
                if isinstance(rule, BulkReaction):
                    fields = dict.fromkeys((*rule.reactants, *rule.products))
                    terms = np.asarray([(self.ids[field], rule.reactants.get(field, 0),
                                         rule.products.get(field, 0) - rule.reactants.get(field, 0))
                                        for field in fields], dtype=np.int32)
                    if rule.rate_constant and np.any(terms[:, 2] < 0):
                        allocated = ct.c_size_t()
                        self._check(lib.omni_hip_add_bulk(self._handle, terms.ctypes.data, len(terms),
                                                         ct.byref(allocated)), 'add bulk reaction')
                        self.allocated_bytes = allocated.value
                        self.rules.append((rule, bulk_id))
                        bulk_id += 1
                elif isinstance(rule, SurfaceDissolution):
                    plan = _surface_plan(world, self.mesh, lib, rule) if rule.rate_m_s and rule.saturation_mol_m3 else None
                    if plan is None:
                        continue
                    faces, rows, offsets, incidences = plan
                    allocated = ct.c_size_t()
                    self._check(lib.omni_hip_add_surface(self._handle, faces.ctypes.data, len(faces),
                                                        rows.ctypes.data, len(rows), offsets.ctypes.data,
                                                        incidences.ctypes.data, ct.byref(allocated)), 'add interface')
                    self.allocated_bytes = allocated.value
                    self.rules.append((rule, surface_id))
                    surface_id += 1
                else:
                    raise TypeError(f'unsupported GPU reaction: {type(rule).__name__}')
        except BaseException:
            self.close()
            raise
        self.mesh._surface_inputs = None
        self.peak_bytes = self.allocated_bytes

    def write(self, name, start, values):
        self._check(self._library.omni_hip_upload_field(self._handle, self.ids[name], start, len(values),
                                                       values.ctypes.data), f'upload {name}')

    def read(self, name, start, count):
        result = np.empty(count, dtype=np.float64)
        self._check(self._library.omni_hip_download_field(self._handle, self.ids[name], start, count,
                                                         result.ctypes.data), f'download {name}')
        return result

    def total(self, name):
        result = ct.c_double()
        material = self.shared_materials.get(name)
        if material is None:
            code = self._library.omni_hip_total(self._handle, self.ids[name], ct.byref(result))
        else:
            code = self._library.omni_hip_total_material(self._handle, self.ids[name], material, ct.byref(result))
        self._check(code, f'sum {name}')
        return result.value

    def step(self, world, dt_s, count):
        lib, handle = self._library, self._handle
        transport = [(self.ids[name], Parameters.for_field(spec, world._velocity_m_s, dt_s))
                     for name, spec in world.field_specs.items()
                     if spec.diffusivity_m2_s or (spec.advected and any(world._velocity_m_s))]
        for _ in range(count):
            for rule, plan in self.rules:
                if isinstance(rule, BulkReaction):
                    code = lib.omni_hip_bulk(handle, plan, rule.material, rule.rate_constant, dt_s)
                else:
                    code = lib.omni_hip_surface(handle, plan, self.ids[rule.solid_species], self.ids[rule.liquid_species],
                                                rule.rate_m_s, rule.saturation_mol_m3, dt_s)
                self._check(code, 'reaction')
            for field, params in transport:
                self._check(lib.omni_hip_run_field(handle, field, ct.byref(params), 1, None), 'transport')
        error = ct.c_int()
        self._check(lib.omni_hip_finish(handle, ct.byref(error)), 'finish physics step')
        if error.value:
            raise FloatingPointError('GPU physics produced an invalid amount or exceeded the reaction substep budget')

    def remap(self, previous, old_tiles, new_tiles):
        boxes, chunks = _remap_plan(previous.mesh, self.mesh, old_tiles, new_tiles, self._library)
        peak = ct.c_size_t()
        self._check(self._library.omni_hip_remap(self._handle, previous._handle, boxes.ctypes.data, len(boxes),
                                                chunks.ctypes.data, len(chunks), ct.byref(peak)), 'remap')
        self.peak_bytes = max(peak.value, previous.peak_bytes, self.peak_bytes)

    def scores(self, world, tolerances):
        tiles = np.asarray([(*world._blocks[key].start, world._blocks[key].level) for key in self.mesh._keys],
                           dtype=np.int32)
        period = sum(1 << i for i, enabled in enumerate(world._periodic) if enabled)
        base = np.asarray((*world.grid.shape, period, prod(world.grid.shape)), dtype=np.int32)
        fields = []
        porous_supported = hasattr(self._library, 'omni_porous_diffusion_supported')
        for name, tolerance in tolerances.items():
            spec = world.field_specs[name]
            material, porous = spec.material, spec.porous
            descriptor = (self.ids[name], -1 if material is None else material, tolerance.absolute, tolerance.relative)
            if porous_supported:
                descriptor += (porous.material, porous.volume_fraction) if porous else (-1, 0.0)
            fields.append(descriptor)
        descriptors = np.asarray(fields, dtype=_POROUS_SCORE_FIELD if porous_supported else _SCORE_FIELD)
        scores, detail = (np.empty(world.grid.shape, dtype=np.float64) for _ in range(2))
        peak = ct.c_size_t()
        code = self._library.omni_hip_scores(self._handle, tiles.ctypes.data, base.ctypes.data,
                                            descriptors.ctypes.data, len(fields), scores.ctypes.data,
                                            detail.ctypes.data, ct.byref(peak))
        if code:
            reason = self._library.omni_hip_error(code).decode()
            raise RuntimeError(f'HIP refinement scoring: {reason} ({code})')
        self.peak_bytes = max(self.peak_bytes, peak.value)
        return scores, detail


class GpuWorld(AdaptiveWorld):
    """Use HIP for state evolution, scoring, and remapping, with host mesh decisions."""

    _solver_source = "hip"

    def __init__(self, grid, *, build=DEFAULT_BUILD, device=0, budget_bytes=512 * 2**20,
                 reserve_bytes=1024 * 2**20, pack_material_fields=False, **mesh_options):
        self._gpu: _GpuState | None = None
        self._closed = False
        self._peak_gpu_bytes = 0
        self._pack_material_fields = pack_material_fields
        self.gpu_build = Path(build)
        self.gpu_device = device
        self.gpu_budget_bytes = budget_bytes
        self.gpu_reserve_bytes = reserve_bytes
        super().__init__(grid, **mesh_options)

    @property
    def peak_gpu_bytes(self):
        return max(self._peak_gpu_bytes, self._gpu.peak_bytes if self._gpu else 0)

    @property
    def resident_gpu_bytes(self):
        return self._gpu.allocated_bytes if self._gpu else 0

    @property
    def gpu_field_buffers(self):
        return self._gpu.field_buffers if self._gpu else 0

    def close(self):
        if self._gpu is not None:
            self._peak_gpu_bytes = self.peak_gpu_bytes
            self._gpu.close()
            self._gpu = None
        self._closed = True

    def _check_open(self):
        if self._closed or self._failed or (self._gpu is not None and not self._gpu._handle):
            raise RuntimeError('GPU world is closed or a prior step failed')

    def _check_ready(self):
        self._check_open()
        super()._check_ready()

    def _activate(self):
        if self._gpu is None:
            self._gpu = _GpuState(self, self.gpu_budget_bytes, upload=True)
            for block in self._blocks.values():
                block.fields.clear()

    def _device_mesh(self, topology, library):
        return FrozenMesh(self, topology=topology, library=library)

    def _device_field_specs(self):
        return self.field_specs

    def _upload_initial_fields(self, state, groups):
        for first, *others in groups:
            packed = state.mesh.pack(self, first)
            for name in others:
                packed += state.mesh.pack(self, name)
            state.write(first, 0, packed)

    def _restore_host(self):
        self._check_open()
        if self._gpu is None:
            return
        for name in self._gpu.ids:
            values = self._gpu.read(name, 0, self.active_cells)
            material = self._gpu.shared_materials.get(name)
            for key, block in self._blocks.items():
                start = self._gpu.mesh._regions[key]
                local = values[start:start + block.materials.size].reshape(block.materials.shape)
                block.fields[name] = np.where(block.materials == material, local, 0.0) if material is not None else local
        self._peak_gpu_bytes = self.peak_gpu_bytes
        self._gpu.close()
        self._gpu = None

    def add_field(self, name, spec=FieldSpec(), *, initial_density=0):
        self._restore_host()
        super().add_field(name, spec, initial_density=initial_density)

    def add_reaction(self, rule):
        if not isinstance(rule, (BulkReaction, SurfaceDissolution)):
            raise TypeError(f'unsupported GPU reaction: {type(rule).__name__}')
        self._restore_host()
        super().add_reaction(rule)

    def _index(self, block, cell):
        return self._gpu.mesh._regions[block.key] + int(np.ravel_multi_index(cell, block.materials.shape))

    def sample(self, name, position_m):
        self._check_open()
        if self._gpu is None:
            return super().sample(name, position_m)
        block, cell = self._cell_at(position_m)
        material = self._gpu.shared_materials.get(name)
        if material is not None and block.materials[cell] != material:
            return 0.0
        return float(self._gpu.read(name, self._index(block, cell), 1)[0] / self._volume(block))

    def total(self, name):
        self._check_open()
        if self._gpu is None:
            return super().total(name)
        return self._gpu.total(name)

    def block_amounts(self, name, key):
        self._check_open()
        block = self._blocks[key]
        if self._gpu is None:
            return block.fields[name].copy()
        start = self._gpu.mesh._regions[key]
        local = self._gpu.read(name, start, block.materials.size).reshape(block.materials.shape)
        material = self._gpu.shared_materials.get(name)
        return np.where(block.materials == material, local, 0.0) if material is not None else local

    def block_densities(self, name, key):
        return self.block_amounts(name, key) / self._volume(self._blocks[key])

    def set_block_densities(self, name, key, density):
        self._check_open()
        if self._gpu is None:
            return super().set_block_densities(name, key, density)
        block = self._blocks[key]
        values = np.ascontiguousarray(self._densities_to_amounts(name, block, density)).ravel()
        start = self._gpu.mesh._regions[key]
        material = self._gpu.shared_materials.get(name)
        if material is not None:
            packed = self._gpu.read(name, start, block.materials.size)
            np.copyto(packed, values, where=block.materials.ravel() == material)
            values = packed
        self._gpu.write(name, start, values)

    def add_amounts(self, position_m, changes):
        self._check_open()
        if self._gpu is None:
            return super().add_amounts(position_m, changes)
        block, cell = self._cell_at(position_m)
        index = self._index(block, cell)
        values = {}
        for name, change in changes.items():
            spec = self.field_specs[name]
            if not spec.available_in(block.materials[cell]):
                raise ValueError(f'{name} is unavailable in the target material')
            value = float(self._gpu.read(name, index, 1)[0]) + change
            if not isfinite(value) or (spec.nonnegative and value < 0):
                raise ValueError(f'source would make {name} invalid')
            values[name] = np.asarray([value], dtype=np.float64)
        for name, value in values.items():
            self._gpu.write(name, index, value)

    def _advance_physics(self, dt_s, end):
        if not self.field_specs:
            self._time_s = end
            return
        self._activate()
        count = max(1, ceil(dt_s / self._transport_step_limit()))
        self._gpu.step(self, dt_s / count, count)
        self._time_s = end

    def _make_transport_batches(self, tiles, blocks, faces):
        return None  # CPU transport batches are unused in this backend.

    def _projection_scores(self, tolerances):
        if self._gpu is None:
            return super()._projection_scores(tolerances)
        return self._gpu.scores(self, tolerances)

    def _rebuild_tile(self, key, region, selected):
        if self._gpu is None:
            return super()._rebuild_tile(key, region, selected)
        return tuple(self._tile_layout(key, region, selected))

    def _build_faces(self, tiles, previous=None):
        return FaceGroups(tiles, self.grid.shape, self._block_cells, self.grid.spacing_m, self._periodic, previous)

    def _regrid(self, selected):
        self._check_open()
        if not self.field_specs:
            return super()._regrid(selected)
        changed = selected ^ self._refinement
        if not changed.any():
            return
        replacement = None
        try:
            self._activate()
            previous = self._gpu
            candidate = self._prepare_regrid(selected, changed)
            budget = self.gpu_budget_bytes - previous.allocated_bytes
            replacement = _GpuState(self, budget, upload=False, topology=candidate)
            replacement.remap(previous, self._tiles, candidate.tiles)
        except BaseException as error:
            if replacement is not None:
                replacement.close()
            self._record('hip', 'regrid_aborted', requested_cells=int(selected.sum()), message=str(error))
            raise
        self._gpu = replacement
        try:
            self._publish_regrid(candidate, changed)
        finally:
            previous.close()
