"""Native finite-volume transport on a frozen AdaptiveWorld mesh."""

from __future__ import annotations

import ctypes as ct
from os import getpid
from math import isfinite
from pathlib import Path

import numpy as np

from .engine import AdaptiveWorld, FieldSpec
from .flux import _flat_indices

_TILE = np.dtype([(name, 'i4') for name in ('offset', 'nx', 'ny', 'nz')]
                 + [(name, 'f8') for name in ('volume', 'area', 'distance')], align=True)
_GEOMETRY = np.dtype([('axis', 'i4')]
                     + [(name, 'f8') for name in ('low_volume', 'high_volume', 'area', 'distance')], align=True)
_SURFACE_PAIR = np.dtype([('solid', 'i4'), ('liquid', 'i4'), ('area', 'f8'), ('volume', 'f8')], align=True)
_CHUNK_SIZE = 128  # Number of cells in a native execution chunk.
_INT_LIMIT = np.iinfo(np.int32).max
DEFAULT_BUILD = Path(__file__).resolve().parent.parent / 'build' / 'native'


def process_vram_bytes() -> int | None:
    process = Path('/sys/class/kfd/kfd/proc') / str(getpid())
    try:
        files = list(process.glob('vram_*'))
        return sum(int(counter.read_text()) for counter in files) if files else None
    except (OSError, ValueError):
        return None


class _Mesh(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in
                ('tiles', 'chunks', 'materials', 'faces', 'geometries', 'cells', 'offsets', 'incidences')]
    _fields_ += [(name, ct.c_int32) for name in
                 ('tile_count', 'chunk_count', 'cell_count', 'face_count', 'geometry_count', 'boundary_count')]


class Parameters(ct.Structure):
    _fields_ = [('velocity', ct.c_double * 3), ('diffusivity', ct.c_double),
                ('dt', ct.c_double), ('material', ct.c_int32), ('porous_material', ct.c_int32),
                ('porous_fraction', ct.c_double), ('porous_resistance', ct.c_double)]

    @classmethod
    def for_field(cls, spec: FieldSpec, velocity: tuple[float, float, float], dt_s: float) -> Parameters:
        if len(velocity) != 3 or not all(map(isfinite, velocity)) or not isfinite(dt_s) or dt_s <= 0:
            raise ValueError('expected a finite velocity and positive finite step')
        speed = velocity if spec.advected else (0, 0, 0)
        porous = spec.porous
        material = -1 if spec.material is None else spec.material
        if porous is not None and porous.excluded_material_from is not None:
            material = -2 - porous.excluded_material_from
        if spec.isolated_materials_from is not None:
            material = -2 - spec.isolated_materials_from
        return cls((ct.c_double * 3)(*speed), spec.diffusivity_m2_s, dt_s,
                   material,
                   porous.material if porous else -1, porous.volume_fraction if porous else 0,
                   porous.resistance_multiplier if porous else 1)


def _require_porous_diffusion(library):
    if not hasattr(library, 'omni_porous_diffusion_supported'):
        raise RuntimeError('rebuild the native library for porous diffusion')


def _check_transport_parameters(library, params):
    if params.porous_fraction:
        _require_porous_diffusion(library)
    if params.material <= -2 and not hasattr(library, "omni_composite_transport_supported"):
        raise RuntimeError("rebuild the native library for composite material transport")


class FrozenMesh:
    """Pack cells by block. Only inter-block and periodic faces are explicit."""

    def __init__(self, world: AdaptiveWorld, *, topology=None, library=None, explicit_faces=None):
        topology = world._topology if topology is None else topology
        blocks = topology.blocks
        if sum(block.materials.size for block in blocks.values()) >= _INT_LIMIT - _CHUNK_SIZE:
            raise ValueError('mesh exceeds the signed 32-bit native cell index range')
        self._topology = topology.faces
        self._keys = tuple(blocks)
        regions, tiles, chunks = {}, [], []
        offset = 0
        for tile, (key, block) in enumerate(blocks.items()):
            spacing = world.grid.spacing_m / (2 ** block.level)
            block_count = block.materials.size
            regions[key] = offset
            tiles.append((offset, *block.materials.shape, spacing ** 3, spacing ** 2, spacing))
            chunks.extend((tile, start) for start in range(0, block_count, _CHUNK_SIZE))
            offset += block_count
        self._regions = regions
        self._tile_ids = {key: index for index, key in enumerate(self._keys)}
        materials = np.concatenate([block.materials.ravel() for block in blocks.values()])
        packed_tiles = np.asarray(tiles, dtype=_TILE)
        if explicit_faces is not None:
            edges, geometries = explicit_faces(self._regions)
        elif library is None:
            edges, geometries = self._reference_faces(world, blocks, regions)
        else:
            self._locations = np.asarray([(*block.start, *block.base_shape, block.level)
                                          for block in blocks.values()], dtype=np.int32)
            edges, geometries, patches = self._native_faces(world, blocks, packed_tiles, library)
            self._surface_inputs = (packed_tiles, materials, patches)
        if len(edges) > _INT_LIMIT // 2:
            raise ValueError('mesh exceeds the signed 32-bit native face index range')
        face_ids = np.arange(1, len(edges) + 1, dtype=np.int32)
        endpoints = np.concatenate((edges[:, 0], edges[:, 1]))
        order = np.argsort(endpoints, kind='stable')
        cells, starts = np.unique(endpoints[order], return_index=True)
        offsets = np.empty(len(starts) + 1, dtype=np.int32)
        offsets[:-1], offsets[-1] = starts, len(endpoints)
        incidences = np.concatenate((-face_ids, face_ids))[order]
        # Keep the storage alive for the entire lifetime of the C view.
        self._arrays = (packed_tiles, np.asarray(chunks, dtype=np.int32), materials, edges,
                        np.asarray(geometries, dtype=_GEOMETRY), cells, offsets, incidences)
        self._topology_bytes = sum(array.nbytes for array in self._arrays)
        self._view = _Mesh(*(array.ctypes.data for array in self._arrays), len(tiles), len(chunks),
                           offset, len(edges), len(geometries), len(cells))

    def _reference_faces(self, world, blocks, regions):
        geometry_ids, faces = {}, []
        for face in self._topology:
            if isinstance(face.low.index[face.axis], slice):
                continue  # Regular interior face.
            low, high = blocks[face.low.block], blocks[face.high.block]
            geometry = (face.axis, world._volume(low), world._volume(high), face.area_m2, face.center_distance_m)
            index = geometry_ids.setdefault(geometry, len(geometry_ids))
            left = _flat_indices(face.low, low, regions[low.key])
            right = _flat_indices(face.high, high, regions[high.key])
            if low is high:
                active = left != right
                left, right = left[active], right[active]
            edges = np.empty((len(left), 3), dtype=np.int32)
            edges[:, 0], edges[:, 1], edges[:, 2] = left, right, index
            faces.append(edges)
        return (np.concatenate(faces) if faces else np.empty((0, 3), dtype=np.int32)), list(geometry_ids)

    def _native_faces(self, world, blocks, tiles, library):
        geometry_ids, geometries, patches = {}, [], []
        dimensions = {key: block.materials.shape for key, block in blocks.items()}
        ids = self._tile_ids
        for face in self._topology:
            low, high = face.low, face.high
            low_shape, high_shape = dimensions[low.block], dimensions[high.block]
            low_start = tuple(index.start or 0 if isinstance(index, slice) else index % size
                              for index, size in zip(low.index, low_shape))
            high_start = tuple(index.start or 0 if isinstance(index, slice) else index % size
                               for index, size in zip(high.index, high_shape))
            fine, shape = (high, high_shape) if low.coarse else (low, low_shape)
            extent = tuple(len(range(*index.indices(size))) if isinstance(index, slice) else 1
                           for index, size in zip(fine.index, shape))
            geometry = -1
            low_id, high_id = ids[low.block], ids[high.block]
            if not isinstance(low.index[face.axis], slice):
                a, b = blocks[low.block], blocks[high.block]
                key = (face.axis, a.level, b.level)
                geometry = geometry_ids.setdefault(key, len(geometries))
                if geometry == len(geometries):
                    geometries.append((face.axis, world._volume(a), world._volume(b),
                                       face.area_m2, face.center_distance_m))
            patches.append((low_id, high_id, *low_start, *high_start, *extent, low.coarse, high.coarse, geometry))
        patches = np.asarray(patches, dtype=np.int32)
        expand = library.omni_expand_faces
        expand.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p)
        expand.restype = ct.c_int64
        args = (tiles.ctypes.data, patches.ctypes.data, len(patches))
        count = expand(*args, None)
        if count > _INT_LIMIT // 2:
            raise ValueError('mesh exceeds the signed 32-bit native face index range')
        edges = np.empty((count, 3), dtype=np.int32)
        expand(*args, edges.ctypes.data)
        return edges, geometries, patches

    def _surface_pairs(self, library, solid, liquid):
        scan = library.omni_scan_surface
        scan.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p)
        scan.restype = ct.c_int64
        tiles, materials, patches = self._surface_inputs
        args = (tiles.ctypes.data, patches.ctypes.data, len(patches), materials.ctypes.data, solid, liquid)
        count = scan(*args, None)
        if count > _INT_LIMIT // 2:
            raise ValueError('interface exceeds the GPU index range')
        pairs = np.empty(count, dtype=_SURFACE_PAIR)
        if count:
            scan(*args, pairs.ctypes.data)
        return pairs

    def _release_uploaded_buffers(self) -> None:
        for pointer, _ in _Mesh._fields_[:8]:
            setattr(self._view, pointer, None)
        self._arrays = ()

    @property
    def cells(self) -> int:
        return self._view.cell_count

    @property
    def boundary_faces(self) -> int:
        return self._view.face_count

    @property
    def topology_bytes(self) -> int:
        return self._topology_bytes

    def pack(self, world: AdaptiveWorld, field: str) -> np.ndarray:
        if world._faces is not self._topology:
            raise ValueError('world topology changed after the native mesh was frozen')
        return np.concatenate([world._blocks[key].fields[field].ravel() for key in self._keys])

    def _state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64, order='C')
        if state.shape != (self.cells,) or not np.isfinite(state).all():
            raise ValueError('expected one finite FP64 amount per active cell')
        return state


def _load(build: Path, name: str) -> ct.CDLL:
    path = (build / f'libomnisim_{name}.so').resolve()
    if not path.is_file():
        raise FileNotFoundError(f'native backend has not been built: {path}')
    return ct.CDLL(str(path))


def _count(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or not 0 < value <= _INT_LIMIT:
        raise ValueError(f'{name} must be a positive 32-bit integer')
    return int(value)


class CpuTransport:
    def __init__(self, mesh: FrozenMesh, build: Path = DEFAULT_BUILD):
        self.mesh = mesh
        self._library = _load(Path(build), 'cpu')
        self._run = self._library.omni_cpu_run
        self._run.argtypes = (ct.POINTER(_Mesh), ct.POINTER(Parameters), ct.c_void_p, ct.c_void_p,
                              ct.c_void_p, ct.c_int, ct.c_int)
        self._run.restype = None

    def run(self, initial: np.ndarray, params: Parameters, steps: int, threads: int = 1) -> np.ndarray:
        _check_transport_parameters(self._library, params)
        steps, threads = _count(steps, 'steps'), _count(threads, 'threads')
        current = self.mesh._state(initial).copy()
        output = np.empty_like(current)
        flux = np.empty(self.mesh.boundary_faces, dtype=np.float64)
        self._run(ct.byref(self.mesh._view), ct.byref(params), current.ctypes.data,
                  output.ctypes.data, flux.ctypes.data, steps, threads)
        return output if steps % 2 else current


class HipTransport:
    """Own device state and topology. Close the context after use."""

    def __init__(self, mesh: FrozenMesh, build: Path = DEFAULT_BUILD, *, device: int = 0,
                 budget_bytes: int = 512 * 2**20, reserve_bytes: int = 1024 * 2**20, _fields: int = 1, _library=None):
        if min(device, budget_bytes, reserve_bytes) < 0:
            raise ValueError('device and memory limits must be nonnegative')
        self.mesh = mesh
        self._uploaded = False
        self._handle = ct.c_void_p()
        self._library = _load(Path(build), 'hip') if _library is None else _library
        library = self._library
        library.omni_hip_error.argtypes = (ct.c_int,)
        library.omni_hip_error.restype = ct.c_char_p
        library.omni_hip_create_world.argtypes = (ct.POINTER(_Mesh), ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_int,
                                                   ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_size_t))
        library.omni_hip_upload.argtypes = library.omni_hip_download.argtypes = (ct.c_void_p, ct.c_void_p)
        library.omni_hip_run.argtypes = (ct.c_void_p, ct.POINTER(Parameters), ct.c_int, ct.POINTER(ct.c_float))
        library.omni_hip_destroy.argtypes = (ct.c_void_p,)
        library.omni_hip_destroy.restype = None
        allocated = ct.c_size_t()
        code = library.omni_hip_create_world(ct.byref(mesh._view), device, budget_bytes, reserve_bytes,
                                              _count(_fields, 'field count'), ct.byref(self._handle), ct.byref(allocated))
        self.allocated_bytes = allocated.value
        self._check(code, f'create (plan {allocated.value} B, budget {budget_bytes} B, reserve {reserve_bytes} B)')

    def _check(self, code: int, operation: str) -> None:
        if code:
            self.close()
            reason = self._library.omni_hip_error(code).decode()
            raise RuntimeError(f'HIP {operation}: {reason} ({code})')

    def _require_state(self) -> None:
        if not self._handle or not self._uploaded:
            raise RuntimeError('upload a state to an open HIP context first')

    def upload(self, initial: np.ndarray) -> None:
        if not self._handle:
            raise RuntimeError('HIP context is closed')
        state = self.mesh._state(initial)
        self._check(self._library.omni_hip_upload(self._handle, state.ctypes.data), 'upload')
        self._uploaded = True

    def advance(self, params: Parameters, steps: int) -> float:
        self._require_state()
        _check_transport_parameters(self._library, params)
        elapsed = ct.c_float()
        self._check(self._library.omni_hip_run(self._handle, ct.byref(params), _count(steps, 'steps'),
                                              ct.byref(elapsed)), 'advance')
        return elapsed.value / 1000

    def download(self) -> np.ndarray:
        self._require_state()
        state = np.empty(self.mesh.cells, dtype=np.float64)
        self._check(self._library.omni_hip_download(self._handle, state.ctypes.data), 'download')
        return state

    def close(self) -> None:
        if self._handle:
            self._library.omni_hip_destroy(self._handle)
            self._handle = ct.c_void_p()

    def __enter__(self) -> HipTransport:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
