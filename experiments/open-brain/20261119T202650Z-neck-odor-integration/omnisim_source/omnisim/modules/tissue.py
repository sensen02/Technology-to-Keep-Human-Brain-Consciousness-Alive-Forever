"""Spatial anatomy and extracellular fluid for a synthetic cortical volume."""

from dataclasses import dataclass
from itertools import product
from math import ceil, cos, isfinite, radians, sin, sqrt, tan
from numbers import Integral

import numpy as np

from ..engine import AdaptiveWorld, FieldSpec, PorousDiffusion
from ..world import Grid, Position

TISSUE, BATH = 0, 1
EXCITATORY, INHIBITORY, ASTROCYTE, MICROGLIA, OLIGODENDROCYTE = range(5)
CELL_TYPES = ("excitatory", "inhibitory", "astrocyte", "microglia", "oligodendrocyte")


@dataclass(frozen=True)
class ImplantLayout:
    """Parallel axes inside one rectangular footprint normal to the shaft direction."""

    count: int = 64
    columns: int | None = None
    normal_depth_m: float = 320e-6
    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0
    array_cross_section_m2: float = (120e-6) ** 2
    cross_section_aspect: float = 1.0

    def __post_init__(self):
        if not isinstance(self.count, Integral) or self.count < 1:
            raise ValueError("axis count must be a positive integer")
        if self.columns is not None and (not isinstance(self.columns, Integral) or not 1 <= self.columns <= self.count):
            raise ValueError("columns must be between one and the axis count")
        dimensions = (self.normal_depth_m, self.array_cross_section_m2, self.cross_section_aspect)
        if (not all(isfinite(value) and value > 0 for value in dimensions)
                or not 0 <= self.tilt_deg < 90 or not isfinite(self.azimuth_deg)):
            raise ValueError("invalid reference axis dimensions or angles")

    @property
    def cross_section_size_m(self) -> tuple[float, float]:
        side = sqrt(self.array_cross_section_m2)
        aspect = sqrt(self.cross_section_aspect)
        return side * aspect, side / aspect

    @property
    def array_shape(self) -> tuple[int, int]:
        columns = self.columns or ceil(sqrt(self.count))
        return columns, (self.count + columns - 1) // columns

    @property
    def pitch_m(self) -> tuple[float, float]:
        return tuple(size / n for size, n in zip(self.cross_section_size_m, self.array_shape))

    def cross_section_centers_m(self) -> np.ndarray:
        """Place one shaft center in each occupied slot of the transverse footprint."""
        columns, _ = self.array_shape
        index = np.arange(self.count)
        slots = np.column_stack((index % columns, index // columns))
        return (slots + 0.5) * self.pitch_m - np.asarray(self.cross_section_size_m) / 2

    def _tilt_direction(self) -> np.ndarray:
        azimuth = radians(self.azimuth_deg)
        return np.array((cos(azimuth), sin(azimuth)))

    def _surface_points_m(self, points: np.ndarray) -> np.ndarray:
        direction = self._tilt_direction()
        # Intersect the tilted parallel axes with z=0. The dimension along tilt stretches by sec(theta).
        stretch = 1 / cos(radians(self.tilt_deg)) - 1
        xy = points + (points @ direction * stretch)[:, None] * direction
        return np.column_stack((xy, np.zeros(len(xy))))

    def surface_footprint_m(self) -> np.ndarray:
        """Corners of the full array outline on the tissue surface in perimeter order."""
        corners = np.array(((-1, -1), (1, -1), (1, 1), (-1, 1))) * self.cross_section_size_m / 2
        return self._surface_points_m(corners)

    def axes_m(self) -> np.ndarray:
        """Return entry and tip coordinates with array shape (count, 2, 3)."""
        entry = self._surface_points_m(self.cross_section_centers_m())
        lateral = self.normal_depth_m * tan(radians(self.tilt_deg))
        displacement = np.r_[lateral * self._tilt_direction(), self.normal_depth_m]
        return np.stack((entry, entry + displacement), axis=1)


@dataclass(frozen=True)
class TissueLayer:
    name: str
    bottom_fraction: float
    density_per_mm3: tuple[float, float, float, float, float]


# Sparse synthetic cortical template. Density, branching and connectivity are configurable hypotheses.
LAYERS = (
    TissueLayer("superficial", 1 / 6, (6_000, 4_000, 10_000, 3_000, 2_000)),
    TissueLayer("middle", 7 / 12, (48_000, 12_000, 12_000, 4_000, 5_000)),
    TissueLayer("deep", 1.0, (36_000, 9_000, 12_000, 4_000, 9_000)),
)


@dataclass(frozen=True)
class TissueFluid:
    """Ion amounts are per voxel; concentration is per accessible liquid volume."""

    volume_fraction: float = 0.20
    tortuosity: float = 1.60
    # Name, dissolved concentration (mol/m³, numerically mM), free diffusivity (m²/s).
    SOLUTES = (("Na", 145.0, 1.33e-9), ("K", 3.0, 1.96e-9), ("Cl", 126.0, 2.03e-9),
               ("HCO3", 26.0, 1.18e-9), ("Ca", 1.2, 0.79e-9), ("Mg", 0.8, 0.71e-9),
               ("glucose", 2.5, 0.67e-9), ("O2", 0.035, 2.0e-9))
    COMPOSITION = {"HCO3": {"H": 1, "C": 1, "O": 3}, "glucose": {"C": 6, "H": 12, "O": 6}, "O2": {"O": 2}}

    @staticmethod
    def field(solute: str) -> str:
        return f"ecs.{solute}"

    def attach_fields(self, world: AdaptiveWorld, materials: np.ndarray, *, excluded_material_from=None,
                      active_solutes=None, compact_equilibrium=False) -> None:
        medium = PorousDiffusion(TISSUE, self.volume_fraction, self.tortuosity, excluded_material_from)
        fraction = self.fraction(materials)
        for solute, concentration, diffusivity in self.SOLUTES:
            elements = self.COMPOSITION.get(solute, {solute: 1})
            active = active_solutes is None or solute in active_solutes
            spec = FieldSpec(diffusivity_m2_s=diffusivity if active else 0, porous=medium, elements=elements)
            if compact_equilibrium and not active:
                world.add_equilibrium_field(self.field(solute), spec, concentration)
            else:
                world.add_field(self.field(solute), spec, initial_density=fraction * concentration)

    def fraction(self, materials):
        return np.where(materials == TISSUE, self.volume_fraction, 1.0)

    def sample(self, world: AdaptiveWorld, solute: str, position_m: Position) -> float:
        return world.sample_concentration(self.field(solute), position_m)

    def block_concentration(self, world: AdaptiveWorld, solute: str, key) -> np.ndarray:
        return world.block_densities(self.field(solute), key) / self.fraction(world.block_materials(key))


def tissue_grid(layouts: tuple[ImplantLayout, ...], *, spacing_m=40e-6, margin_m=120e-6,
                tissue_depth_m=480e-6, bath_depth_m=80e-6) -> tuple[Grid, np.ndarray]:
    """Fit every reference layout into the same Cartesian world."""
    if not layouts:
        raise ValueError("at least one layout is required")
    if (not all(isfinite(value) and value > 0 for value in
                (spacing_m, margin_m, tissue_depth_m, bath_depth_m))
            or any(layout.normal_depth_m >= tissue_depth_m for layout in layouts)):
        raise ValueError("grid dimensions must be positive and tips must lie within the tissue")
    outlines = []
    for layout in layouts:
        entry = layout.surface_footprint_m()
        axis = layout.axes_m()[0]
        outlines.extend((entry, entry + (axis[1] - axis[0])))
    points = np.concatenate(outlines)
    start = np.floor((points[:, :2].min(axis=0) - margin_m) / spacing_m)
    end = np.ceil((points[:, :2].max(axis=0) + margin_m) / spacing_m)
    depth_cells = round(tissue_depth_m / spacing_m, 12)
    bath_cells = ceil(round(bath_depth_m / spacing_m, 12))
    shape = (*map(int, end - start), ceil(depth_cells) + bath_cells)
    origin = (float(start[0] * spacing_m), float(start[1] * spacing_m), -bath_cells * spacing_m)
    grid = Grid(shape, spacing_m, origin)
    material_z = np.where(np.arange(shape[2]) < bath_cells, BATH, TISSUE)
    return grid, np.broadcast_to(material_z, shape)


@dataclass(frozen=True)
class Synapses:
    pre: np.ndarray
    post: np.ndarray
    position_m: np.ndarray
    conductance_S: np.ndarray
    delay_s: np.ndarray


@dataclass(frozen=True)
class Anatomy:
    """Catalog with stable row IDs; neuron IDs are the leading rows of the catalog."""

    position_m: np.ndarray
    cell_type: np.ndarray
    layer: np.ndarray
    process_tips_m: np.ndarray
    axon_branch_m: np.ndarray
    synapses: Synapses

    @property
    def neuron_count(self):
        return len(self.axon_branch_m)


def _somata(rng, lower, upper, densities, scale):
    counts = np.round(np.prod((upper - lower) * 1e3) * scale * np.asarray(densities)).astype(int)
    kinds = np.repeat(np.arange(len(CELL_TYPES), dtype=np.int8), counts)
    n = len(kinds)
    if n == 0:
        return np.empty((0, 3)), kinds
    # Jitter local lattice sites without sharing a site between cell types.
    width = (np.prod(upper - lower) / n) ** (1 / 3)
    shape = np.ceil((upper - lower) / width).astype(int)
    ids = rng.choice(int(np.prod(shape)), n, replace=False)
    sites = np.column_stack(np.unravel_index(ids, shape))
    points = lower + (sites + rng.uniform(0.23, 0.77, (n, 3))) * (upper - lower) / shape
    return points, kinds


def _connect(rng, soma, kinds, tips, branches, connections, reach_m):
    bins = np.floor(soma / reach_m).astype(int)
    members = {}
    for index, cell in enumerate(bins):
        members.setdefault(tuple(cell), []).append(index)
    offsets = np.asarray(tuple(product((-1, 0, 1), repeat=3)))
    pre, post = [], []
    if connections:
        for cell, sources in members.items():
            neighbors = np.fromiter((i for nearby in offsets + cell
                                     for i in members.get(tuple(nearby), ())), dtype=np.intp)
            for source in sources:
                reach = reach_m if kinds[source] == EXCITATORY else 0.75 * reach_m
                # Local connectivity is sampled within a finite axonal reach.
                distance2 = np.square(soma[neighbors] - soma[source]).sum(axis=1)
                eligible = neighbors[(distance2 <= reach ** 2) & (neighbors != source)]
                targets = rng.choice(eligible, min(connections, len(eligible)), replace=False)
                pre.extend([source] * len(targets))
                post.extend(targets)
    pre, post = np.asarray(pre, dtype=np.intp), np.asarray(post, dtype=np.intp)
    dendrite = rng.integers(tips.shape[1], size=len(pre))
    contact = soma[post] + rng.uniform(0.45, 0.95, (len(pre), 1)) * (tips[post, dendrite] - soma[post])
    path = np.linalg.norm(soma[pre] - branches[pre], axis=1) + np.linalg.norm(branches[pre] - contact, axis=1)
    conductance = np.where(kinds[pre] == EXCITATORY, 1.8e-9, 3.5e-9)
    return Synapses(pre, post, contact, conductance, 0.5e-3 + path / 0.35)


def build_anatomy(grid: Grid, *, seed=6, density_scale=1.0, connections=6, reach_m=125e-6,
                  layers=LAYERS) -> Anatomy:
    if (not isfinite(density_scale) or density_scale <= 0
            or not isinstance(connections, Integral) or connections < 0 or not isfinite(reach_m) or reach_m <= 0):
        raise ValueError("invalid density or connection parameters")
    boundaries = np.asarray((0.0, *(layer.bottom_fraction for layer in layers)))
    if boundaries[-1] != 1.0 or not np.all(np.diff(boundaries) > 0):
        raise ValueError("layer boundaries must increase from zero to one")
    for layer in layers:
        if (len(layer.density_per_mm3) != len(CELL_TYPES)
                or not all(isfinite(value) and value >= 0 for value in layer.density_per_mm3)):
            raise ValueError("layer density must be supplied for each cell type")
    rng = np.random.default_rng(seed)
    lower = np.asarray(grid.origin_m)
    upper = lower + grid.spacing_m * np.asarray(grid.shape)
    points, types, memberships = [], [], []
    for index, (a, b, layer) in enumerate(zip(boundaries, boundaries[1:], layers)):
        xy = lower[:2], upper[:2]
        positions, kinds = _somata(rng, np.r_[xy[0], a * upper[2]], np.r_[xy[1], b * upper[2]],
                                  layer.density_per_mm3, density_scale)
        points.append(positions)
        types.append(kinds)
        memberships.append(np.full(len(kinds), index, dtype=np.int16))
    kinds = np.concatenate(types)
    order = np.argsort(kinds, kind="stable")
    soma = np.concatenate(points)[order]
    kinds = kinds[order]
    membership = np.concatenate(memberships)[order]
    n = int(np.count_nonzero(kinds <= INHIBITORY))
    direction = rng.normal(size=(len(soma), 4, 3))
    direction /= np.linalg.norm(direction, axis=2, keepdims=True)
    lengths = np.asarray((55, 37, 32, 26, 20))[kinds, None, None] * 1e-6
    tips = soma[:, None] + direction * lengths * rng.uniform(0.45, 1.0, (len(soma), 4, 1))
    pyramidal = kinds == EXCITATORY
    tips[pyramidal, 0, 2] = soma[pyramidal, 2] - 1.5 * lengths[pyramidal, 0, 0]
    tissue_lower = np.r_[lower[:2], 0.0]
    tissue_upper = upper - grid.spacing_m * 1e-9
    tips = np.clip(tips, tissue_lower, tissue_upper)
    axon = soma[:n] + direction[:n, -1] * np.asarray((14, 14, 24)) * 1e-6
    axon = np.clip(axon, tissue_lower, tissue_upper)
    synapses = _connect(rng, soma[:n], kinds[:n], tips[:n], axon, connections, reach_m)
    return Anatomy(soma, kinds, membership, tips, axon, synapses)
