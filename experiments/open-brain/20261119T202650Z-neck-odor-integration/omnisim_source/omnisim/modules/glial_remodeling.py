"""Slow glial dynamics coupled to tissue, sparse debris and optional implant surfaces."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from math import exp, expm1, isfinite, pi

import numpy as np

from .tissue import ASTROCYTE, MICROGLIA, TissueFluid
from .tissue_geometry import move_toward, tissue_path_clear

_NEIGHBORS = tuple(product((-1, 0, 1), repeat=3))
_DIRECTIONS = np.vstack((np.eye(3), -np.eye(3)))


@dataclass(frozen=True)
class GlialRemodelingConfig:
    update_interval_s: float = 30.0
    response_tau_s: float = 60.0
    damage_decay_s: float = 86_400.0
    sensing_radius_m: float = 18e-6
    activation_threshold: float = .12
    process_speed_m_s: float = .8e-6 / 60
    process_reach_m: float = 18e-6
    migration_delay_s: float = 6 * 3600.0
    migration_speed_m_s: float = .2e-6 / 60
    separation_m: float = 5e-6
    clearance_radius_m: float = 2e-6
    engulfment_m3_s: float = 1e-20
    engulfment_capacity_m3: float = 50e-18
    surface_offset_m: float = .85e-6
    contact_radius_m: float = 2e-6
    attachment_tau_s: float = 300.0
    detachment_tau_s: float = 3600.0
    basal_division_s_inv: float = 0.0
    injury_division_s_inv: float = 1 / 43_200.0
    glucose_required_mol_m3: float = .5
    oxygen_required_mol_m3: float = .01

    def __post_init__(self):
        nonnegative = ('migration_delay_s', 'migration_speed_m_s', 'basal_division_s_inv', 'injury_division_s_inv')
        for name, value in vars(self).items():
            minimum = 0 if name in nonnegative else np.finfo(float).tiny
            if not isfinite(value) or value < minimum:
                raise ValueError(f'invalid glial parameter {name}')
        if self.activation_threshold > 1:
            raise ValueError('glial activation threshold cannot exceed one')


@dataclass
class GlialSurface:
    position_m: np.ndarray
    normal: np.ndarray
    area_m2: np.ndarray
    coverage: np.ndarray

    def __post_init__(self):
        n = len(self.area_m2)
        if (self.position_m.shape != (n, 3) or self.normal.shape != (n, 3)
                or self.area_m2.shape != (n,) or self.coverage.shape != (n,)
                or not np.isfinite(self.position_m).all() or not np.isfinite(self.area_m2).all()
                or not np.all(self.area_m2 > 0) or not np.allclose(np.linalg.norm(self.normal, axis=1), 1)
                or not np.all((0 <= self.coverage) & (self.coverage <= 1))
                or self.coverage.dtype.kind != 'f' or not self.coverage.flags.writeable):
            raise ValueError('invalid glial surface patches')


class _PointIndex:
    def __init__(self, points, radius):
        self.points, self.radius = points, radius
        self.bins = defaultdict(list)
        for i, point in enumerate(points):
            self.bins[self._key(point)].append(i)

    def _key(self, point):
        return tuple(np.floor(point / self.radius).astype(int))

    def nearby(self, point):
        cell = self._key(point)
        ids = [idx for shift in _NEIGHBORS
               for idx in self.bins.get(tuple(i + j for i, j in zip(cell, shift)), ())]
        if not ids:
            return np.empty(0, dtype=int), np.empty(0)
        distance = np.linalg.norm(self.points[ids] - point, axis=1)
        order = np.argsort(distance, kind='stable')
        order = order[distance[order] <= self.radius]
        return np.asarray(ids)[order], distance[order]


class GlialRemodeling:
    """The tissue module owns cell IDs and salt pools; this module owns slow history."""

    def __init__(self, tissue, config=GlialRemodelingConfig(), *, surface=None, debris=()):
        self.tissue, self.config, self.surface = tissue, config, surface
        rows = np.asarray(debris, dtype=float).reshape((-1, 4))
        if not np.isfinite(rows).all() or np.any(rows[:, 3] <= 0):
            raise ValueError('debris requires finite positions and positive volumes')
        self.debris_m3 = rows[:, 3].copy()
        self._original_debris_m3 = self.debris_m3.copy()
        self._debris = _PointIndex(rows[:, :3], config.sensing_radius_m)
        self._patches = (None if surface is None else _PointIndex(
            surface.position_m + surface.normal * config.surface_offset_m, config.process_reach_m))
        self.ticks = 0
        self.migrations = 0
        self.births = 0

    def attach(self, world, name):
        if self.tissue._world is not world or self.tissue.glial_remodeling is not None:
            raise ValueError('attach the tissue module first and use one glial remodeling module per population')
        if ((self.config.basal_division_s_inv or self.config.injury_division_s_inv)
                and not {TissueFluid.field('glucose'), TissueFluid.field('O2')} <= world.field_specs.keys()):
            raise ValueError('glial division requires extracellular glucose and oxygen fields')
        self.tissue.glial_remodeling = self
        self._start_s = world.time_s
        n = self.tissue.anatomy.neuron_count
        self.tip_m = self.tissue.anatomy.position_m[n:].copy()
        self.activation = np.zeros(len(self.tip_m))
        self.engulfed_m3 = np.zeros(len(self.tip_m))
        self._exposure_s = np.zeros(len(self.tip_m))
        self._cycle = np.zeros(len(self.tip_m))
        self.parent_ids = np.full(len(self.tip_m), -1, dtype=int)
        self.birth_time_s = np.full(len(self.tip_m), self._start_s)

    def step(self, world, name, dt_s):
        interval = self.config.update_interval_s
        # Only the fast modules and finite-volume fields run at every transport step.
        while (when := self._start_s + (self.ticks + 1) * interval) <= world.time_s + dt_s:
            self._advance(world, name, when, interval)
            self.ticks += 1

    def _target(self, world, soma, debris_ids):
        if self._patches is not None:
            ids, _ = self._patches.nearby(soma)
            for patch in ids:
                toward = self._patches.points[patch]
                if tissue_path_clear(world, soma, toward):
                    return toward, int(patch)
        for site in debris_ids:
            toward = self._debris.points[site]
            if self.debris_m3[site] > 0 and tissue_path_clear(world, soma, toward):
                return toward, -1
        return soma, -1

    def _engulf(self, world, idx, tip, debris_ids, dt):
        p = self.config
        remaining = min(p.engulfment_m3_s * dt * self.activation[idx],
                        p.engulfment_capacity_m3 - self.engulfed_m3[idx])
        for site in debris_ids:
            if remaining <= 0:
                break
            point = self._debris.points[site]
            if (self.debris_m3[site] > 0 and np.linalg.norm(point - tip) <= p.clearance_radius_m
                    and tissue_path_clear(world, tip, point)):
                moved = min(remaining, self.debris_m3[site])
                self.debris_m3[site] -= moved
                self.engulfed_m3[idx] += moved
                remaining -= moved

    def _advance(self, world, name, time_s, dt):
        tissue, p = self.tissue, self.config
        anatomy = tissue.anatomy
        n = anatomy.neuron_count
        xyz = anatomy.position_m.copy()
        kinds, alive = anatomy.cell_type[n:], tissue.glial_alive
        dt_activation = -expm1(-dt / p.response_tau_s)
        damage_factor = exp(-(time_s - self._start_s) / p.damage_decay_s)
        occupancy = tissue.living_somata
        footprints = None if self.surface is None else np.ones_like(self.surface.coverage)
        moved_soma = False
        responders = np.flatnonzero(alive & ((kinds == ASTROCYTE) | (kinds == MICROGLIA)))
        for i in responders:
            cell = n + i
            soma = xyz[cell].copy()
            ids, distances = self._debris.nearby(soma)
            debris_cue = np.max(self.debris_m3[ids] / self._original_debris_m3[ids] *
                                np.exp(-distances / p.sensing_radius_m), initial=0.0)
            injury = 0.0 if tissue.damage_field is None else world.sample(tissue.damage_field, soma)
            cue = max(float(debris_cue), float(np.clip(injury, 0, 1)) * damage_factor)
            self.activation[i] += (cue - self.activation[i]) * dt_activation
            if self.activation[i] >= p.activation_threshold:
                self._exposure_s[i] += dt
                target, patch = self._target(world, soma, ids)
            else:
                self._exposure_s[i] = 0.0
                target, patch = soma, -1
            tip = move_toward(self.tip_m[i], target, p.process_speed_m_s * dt)
            if (not np.array_equal(tip, self.tip_m[i]) and np.linalg.norm(tip - soma) <= p.process_reach_m
                    and tissue_path_clear(world, self.tip_m[i], tip) and tissue_path_clear(world, soma, tip)):
                self.tip_m[i] = tip
            tip = self.tip_m[i]
            if patch >= 0 and np.array_equal(tip, target):
                fraction = min(1.0, pi * p.contact_radius_m**2 / self.surface.area_m2[patch])
                footprints[patch] *= 1 - fraction
            if kinds[i] == MICROGLIA:
                self._engulf(world, i, tip, ids, dt)
                if self._exposure_s[i] and self._exposure_s[i] >= p.migration_delay_s:
                    next_soma = move_toward(soma, target, p.migration_speed_m_s * dt)
                    if not np.array_equal(next_soma, soma) and self._can_place(world, xyz, occupancy, cell, next_soma):
                        xyz[cell] = next_soma
                        self.migrations += 1
                        moved_soma = True
                        world.emit(name, 'glia_migrated', time_s=time_s, cell=int(cell),
                                   x_m=float(next_soma[0]), y_m=float(next_soma[1]), z_m=float(next_soma[2]))
        if self.surface is not None:
            coverage = self.surface.coverage
            delta = (1 - footprints) - coverage
            tau = np.where(delta > 0, p.attachment_tau_s, p.detachment_tau_s)
            coverage[:] = np.clip(coverage + delta * -np.expm1(-dt / tau), 0, 1)
        self._divide(world, name, time_s, dt, xyz, occupancy, moved_soma)

    def _can_place(self, world, xyz, occupancy, cell, point):
        soma = xyz[cell]
        if not tissue_path_clear(world, soma, point):
            return False
        distance2 = np.square(xyz - point).sum(axis=1)
        distance2[cell] = np.inf
        return not np.any(distance2[occupancy] < self.config.separation_m**2)

    def _nutrients(self, world, point):
        p = self.config
        return (world.sample_concentration(TissueFluid.field('glucose'), point) >= p.glucose_required_mol_m3
                and world.sample_concentration(TissueFluid.field('O2'), point) >= p.oxygen_required_mol_m3)

    def _divide(self, world, name, time_s, dt, xyz, occupancy, moved_soma):
        tissue, p = self.tissue, self.config
        n = tissue.anatomy.neuron_count
        # A new cell must be supplied with new biomass before it can divide again.
        astrocytes = tissue.glial_alive & (tissue.anatomy.cell_type[n:] == ASTROCYTE)
        eligible = astrocytes & (tissue.glial_volume_fraction >= 1)
        rate = p.basal_division_s_inv + p.injury_division_s_inv * self.activation
        self._cycle[eligible] += dt * rate[eligible]
        parents, offspring = [], []
        for i in np.flatnonzero(eligible & (self._cycle >= 1)):
            cell = n + i
            soma = xyz[cell]
            if not self._nutrients(world, soma):
                continue
            for direction in _DIRECTIONS:
                point = soma + direction * (1.05 * p.separation_m)
                if (not self._can_place(world, xyz, occupancy, cell, point) or not self._nutrients(world, point)
                        or any(np.linalg.norm(point - other) < p.separation_m for other in offspring)):
                    continue
                parents.append(int(i))
                offspring.append(point)
                self._cycle[i] = 0
                break
        if parents or moved_soma:
            positions = np.vstack((xyz[n:], offspring)) if parents else xyz[n:]
            tissue.update_glial_population(positions, parents)
        if parents:
            first = len(self.tip_m)
            self.tip_m = np.vstack((self.tip_m, offspring))
            self.engulfed_m3[parents] *= .5
            self.engulfed_m3 = np.r_[self.engulfed_m3, self.engulfed_m3[parents]]
            self.activation = np.r_[self.activation, np.zeros(len(parents))]
            self._cycle = np.r_[self._cycle, np.zeros(len(parents))]
            self._exposure_s = np.r_[self._exposure_s, np.zeros(len(parents))]
            self.parent_ids = np.r_[self.parent_ids, np.asarray(parents) + n]
            self.birth_time_s = np.r_[self.birth_time_s, np.full(len(parents), time_s)]
            self.births += len(parents)
            for offset, parent in enumerate(parents):
                world.emit(name, 'glia_divided', time_s=time_s, parent=parent + n, cell=first + offset + n)

    def summary(self):
        area = 0.0 if self.surface is None else float(self.surface.area_m2 @ self.surface.coverage)
        return dict(active_cells=int(self.tissue.glial_alive.sum()), divisions=self.births,
                    soma_moves=self.migrations, covered_area_m2=area, ticks=self.ticks,
                    initial_debris_m3=float(self._original_debris_m3.sum()),
                    remaining_debris_m3=float(self.debris_m3.sum()), engulfed_m3=float(self.engulfed_m3.sum()))
