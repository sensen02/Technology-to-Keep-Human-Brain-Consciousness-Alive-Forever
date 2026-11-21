"""Conservative exchange between point membranes and the shared extracellular fields."""

from dataclasses import dataclass
from math import expm1, isfinite

import numpy as np

from .tissue import TISSUE, TissueFluid

POTASSIUM, CHLORIDE = TissueFluid.field("K"), TissueFluid.field("Cl")


@dataclass(frozen=True)
class IonFeedbackConfig:
    """Effective paired-salt balance for the reduced membrane model, in SI units."""

    release_mol_per_spike: float = 8e-18
    reserve_mol: float = 5e-16
    exchange_half_width_m: float = 4e-6
    recovery_tau_s: float = 0.25
    recovery_interval_s: float = 0.002
    glial_exchange_volume_m3: float = 2e-17
    glial_capacity_mol: float = 2e-16
    glial_tau_s: float = 0.06

    def __post_init__(self):
        values = (self.release_mol_per_spike, self.reserve_mol, self.exchange_half_width_m,
                  self.recovery_tau_s, self.recovery_interval_s, self.glial_exchange_volume_m3,
                  self.glial_capacity_mol, self.glial_tau_s)
        if not all(isfinite(value) and value > 0 for value in values):
            raise ValueError("ion feedback parameters must be finite and positive")


class SpikeIonExchange:
    """Track neuronal withdrawal from a finite pool and exchange on intersecting tissue voxels."""

    def __init__(self, positions_m, fluid, config):
        self._positions = positions_m
        self._rest_density = 3.0 * fluid.volume_fraction
        self._config = config
        self._withdrawal = np.zeros(len(positions_m))
        self._spikes = {}
        self._recovering = set()
        self._last_recovery_s = None
        self.released_mol = 0.0
        self.recovered_mol = 0.0

    @property
    def deficit_mol(self):
        return float(self._withdrawal.sum())

    def record(self, neuron):
        self._spikes[neuron] = self._spikes.get(neuron, 0) + 1

    def _stencil(self, world, neuron, blocks):
        center = self._positions[neuron]
        half = self._config.exchange_half_width_m
        lower, upper = center - half, center + half
        for key in world.blocks_overlapping(lower, upper):
            if key not in blocks:
                blocks[key] = (world.block_grid(key), world.block_materials(key),
                               world.block_densities(POTASSIUM, key), world.block_densities(CHLORIDE, key))
            grid, material, _, _ = blocks[key]
            h = grid.spacing_m
            origin = np.asarray(grid.origin_m)
            first = np.maximum(0, np.floor((lower - origin) / h).astype(int))
            last = np.minimum(grid.shape, np.ceil((upper - origin) / h).astype(int))
            region = tuple(slice(i, j) for i, j in zip(first, last))
            lengths = []
            for lo, hi, i, j, x in zip(lower, upper, first, last, origin):
                left = x + h * np.arange(i, j)
                lengths.append(np.clip(np.minimum(left + h, hi) - np.maximum(left, lo), 0, h))
            volume = lengths[0][:, None, None] * lengths[1][None, :, None] * lengths[2][None, None, :]
            volume *= material[region] == TISSUE
            if np.any(volume):
                yield key, region, volume

    def _transfer(self, world, neuron, amount, blocks, dirty):
        """Return the amount actually moved, positive for secretion and negative for uptake."""
        regions = []
        for key, region, weight in self._stencil(world, neuron, blocks):
            _, _, k, cl = blocks[key]
            if amount < 0:
                weight *= np.minimum(np.maximum(k[region] - self._rest_density, 0), cl[region])
            regions.append((key, region, weight))
        capacity = sum(weight.sum() for _, _, weight in regions)
        if capacity <= 0:
            return 0.0
        transfer = max(amount, -capacity)
        for key, region, weight in regions:
            grid, _, k, cl = blocks[key]
            delta = (transfer / capacity) * weight / grid.cell_volume_m3
            k[region] += delta
            cl[region] += delta
            dirty.add(key)
        return float(transfer)

    def advance(self, world, end_s, inactive):
        config = self._config
        if self._last_recovery_s is None:
            self._last_recovery_s = world.time_s
        elapsed = end_s - self._last_recovery_s
        blocks, dirty = {}, set()
        if elapsed >= config.recovery_interval_s:
            fraction = -expm1(-elapsed / config.recovery_tau_s)
            for neuron in sorted(self._recovering):
                if inactive[neuron]:
                    self._recovering.remove(neuron)
                    continue
                n = self._withdrawal[neuron]
                # Finish the final numerical residue while retaining exact accounting.
                request = n if n <= 1e-10 * config.release_mol_per_spike else n * fraction
                moved = self._transfer(world, neuron, -request, blocks, dirty)
                self._withdrawal[neuron] += moved
                self.recovered_mol -= moved
                if self._withdrawal[neuron] == 0:
                    self._recovering.remove(neuron)
            self._last_recovery_s = end_s
        for neuron, count in sorted(self._spikes.items()):
            available = config.reserve_mol - self._withdrawal[neuron]
            request = min(available, count * config.release_mol_per_spike)
            if request <= 0:
                continue
            moved = self._transfer(world, neuron, request, blocks, dirty)
            if moved:
                self._withdrawal[neuron] += moved
                self.released_mol += moved
                self._recovering.add(neuron)
        for key in sorted(dirty):
            _, _, k, cl = blocks[key]
            world.set_block_densities(POTASSIUM, key, k)
            world.set_block_densities(CHLORIDE, key, cl)
        self._spikes.clear()
