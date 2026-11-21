"""Element-balanced volume reactions and conservative solid-to-liquid transfer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .engine import AdaptiveWorld, FieldSpec
from .mesh import BlockKey, Side
from .world import Field


def _elements(specs: Mapping[str, FieldSpec], species: Mapping[str, int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for name, coeff in species.items():
        spec = specs[name]
        if not spec.nonnegative or spec.elements is None:
            raise ValueError(f"reaction species {name} requires a nonnegative field and element counts")
        for element, amount in spec.elements.items():
            counts[element] += int(coeff) * int(amount)
    return counts


@dataclass(frozen=True)
class BulkReaction:
    """An irreversible mass-action reaction on a single material."""

    reactants: Mapping[str, int]
    products: Mapping[str, int]
    rate_constant: float
    material: int

    def __post_init__(self) -> None:
        for key in ("reactants", "products"):
            species = dict(getattr(self, key))
            if not species or any(not isinstance(n, Integral) or n < 1 for n in species.values()):
                raise ValueError("reaction coefficients must be positive integers")
            object.__setattr__(self, key, MappingProxyType(species))
        if not isfinite(self.rate_constant) or self.rate_constant < 0:
            raise ValueError("rate constant must be finite and nonnegative")

    def validate(self, specs: Mapping[str, FieldSpec]) -> None:
        if _elements(specs, self.reactants) != _elements(specs, self.products):
            raise ValueError("reaction is not element-balanced")
        for name in self.reactants.keys() | self.products.keys():
            if specs[name].material != self.material:
                raise ValueError(f"{name} must be specific to reaction material {self.material}")

    def advance(self, world: AdaptiveWorld, dt_s: float) -> None:
        if self.rate_constant == 0:
            return
        coefficients = {name: self.products.get(name, 0) - self.reactants.get(name, 0)
                        for name in self.reactants.keys() | self.products.keys()}
        consumed = [(name, -coeff) for name, coeff in coefficients.items() if coeff < 0]
        if not consumed:
            return
        for block in world._blocks.values():
            if not np.any(block.materials == self.material):
                continue
            volume = world._volume(block)
            time_left = dt_s
            for _ in range(4096):
                rate = np.full(block.materials.shape, self.rate_constant * volume)
                for name, order in self.reactants.items():
                    rate *= (block.fields[name] / volume) ** order
                if not np.any(rate > 0):
                    break
                # Keep every explicit stage well inside each reactant's inventory limit.
                step = time_left
                time_to_spend = np.empty_like(rate)
                active = rate > 0
                for name, order in consumed:
                    np.divide(block.fields[name] / order, rate, out=time_to_spend, where=active)
                    step = min(step, 0.2 * float(np.min(time_to_spend, where=active, initial=np.inf)))
                if step == 0 or time_left - step == time_left:
                    raise FloatingPointError("reaction requires a smaller world step")
                extent = rate * step
                for name, coeff in coefficients.items():
                    if coeff:
                        block.fields[name] += coeff * extent
                if step == time_left:
                    break
                time_left -= step
            else:
                raise FloatingPointError("reaction exceeded the local explicit substep budget")


@dataclass(frozen=True)
class SurfaceDissolution:
    """Transfer a soluble solid component into liquid across fixed, grid-aligned faces."""

    solid_species: str
    liquid_species: str
    rate_m_s: float
    saturation_mol_m3: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) and value >= 0 for value in (self.rate_m_s, self.saturation_mol_m3)):
            raise ValueError("dissolution rate and saturation must be finite and nonnegative")

    def validate(self, specs: Mapping[str, FieldSpec]) -> None:
        solid, liquid = specs[self.solid_species], specs[self.liquid_species]
        if solid.material is None or liquid.material is None or solid.material == liquid.material:
            raise ValueError("dissolution requires two distinct materials")
        if _elements(specs, {self.solid_species: 1}) != _elements(specs, {self.liquid_species: 1}):
            raise ValueError("dissolution must transfer the same element inventory per mole")

    def advance(self, world: AdaptiveWorld, dt_s: float) -> None:
        if self.rate_m_s == 0 or self.saturation_mol_m3 == 0:
            return
        solid_material = world.field_specs[self.solid_species].material
        liquid_material = world.field_specs[self.liquid_species].material
        solid_demand: dict[BlockKey, Field] = {}
        liquid_demand: dict[BlockKey, Field] = {}
        transfers: list[tuple[Side, Side, Field]] = []
        for face in world._faces:
            low_material = face.low.read(world._blocks[face.low.block].materials)
            high_material = face.high.read(world._blocks[face.high.block].materials)
            orientations = ((face.low, face.high, low_material, high_material),
                            (face.high, face.low, high_material, low_material))
            for solid_side, liquid_side, solid_ids, liquid_ids in orientations:
                interface = (solid_ids == solid_material) & (liquid_ids == liquid_material)
                if not interface.any():
                    continue
                solid_block = world._blocks[solid_side.block]
                liquid_block = world._blocks[liquid_side.block]
                concentration = (liquid_side.read(liquid_block.fields[self.liquid_species])
                                 / world._volume(liquid_block))
                requested = np.maximum(self.saturation_mol_m3 - concentration, 0)
                requested *= interface * (self.rate_m_s * face.area_m2 * dt_s)
                if solid_side.block not in solid_demand:
                    solid_demand[solid_side.block] = np.zeros(solid_block.materials.shape)
                if liquid_side.block not in liquid_demand:
                    liquid_demand[liquid_side.block] = np.zeros(liquid_block.materials.shape)
                solid_side.add(solid_demand[solid_side.block], requested)
                liquid_side.add(liquid_demand[liquid_side.block], requested)
                transfers.append((solid_side, liquid_side, requested))

        solid_scale, liquid_scale = {}, {}
        for key, demand in solid_demand.items():
            available = world._blocks[key].fields[self.solid_species]
            solid_scale[key] = self._limit(demand, available)
            demand.fill(0)
        for key, demand in liquid_demand.items():
            block = world._blocks[key]
            capacity = np.maximum(self.saturation_mol_m3 * world._volume(block)
                                  - block.fields[self.liquid_species], 0)
            liquid_scale[key] = self._limit(demand, capacity)
            demand.fill(0)

        for solid, liquid, amount in transfers:
            amount *= np.minimum(solid.read(solid_scale[solid.block]), liquid.read(liquid_scale[liquid.block]))
            solid.add(solid_demand[solid.block], amount)
            liquid.add(liquid_demand[liquid.block], amount)
        for key, consumed in solid_demand.items():
            world._blocks[key].fields[self.solid_species] -= consumed
        for key, produced in liquid_demand.items():
            world._blocks[key].fields[self.liquid_species] += produced

    @staticmethod
    def _limit(demand: Field, available: Field) -> Field:
        scale = np.ones_like(demand)
        limited = (demand > 0) & (demand >= available)
        # Allow for rounding when several positive face transfers are summed again.
        safe_inventory = available * (1 - 16 * np.finfo(np.float64).eps)
        np.divide(safe_inventory, demand, out=scale, where=limited)
        return scale
