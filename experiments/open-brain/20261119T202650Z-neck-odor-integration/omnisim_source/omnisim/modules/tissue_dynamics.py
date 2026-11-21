"""Conductance-based population and local glial response on the Cartesian world."""

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from math import expm1, isfinite, log

import numpy as np

from .conduction_damage import path_damage
from .electrophysiology import FARADAY_C_MOL, GAS_CONSTANT_J_MOL_K
from .ion_feedback import CHLORIDE, POTASSIUM, IonFeedbackConfig, SpikeIonExchange
from .synaptic_plasticity import PlasticityConfig, SynapticPlasticity
from .tissue import ASTROCYTE, EXCITATORY, MICROGLIA, TISSUE, Anatomy, TissueFluid

REST_V, THRESHOLD_V = -0.065, -0.050
MEMBRANE_F, LEAK_S = 120e-12, 8e-9
EXCITATORY_TAU_S, INHIBITORY_TAU_S = 0.003, 0.008
THERMAL_V = GAS_CONSTANT_J_MOL_K * 310.15 / FARADAY_C_MOL


@dataclass(frozen=True)
class _BlockSites:
    cells: np.ndarray
    local: tuple[np.ndarray, ...]
    astrocytes: np.ndarray
    microglia: np.ndarray


class TissueDynamics:
    """Reduced point membranes, delayed synapses and a reversible glial salt buffer."""

    def __init__(self, anatomy: Anatomy, fluid: TissueFluid, *, damage_field: str | None = None,
                 buffer_glia: bool = True, plasticity: PlasticityConfig | None = None,
                 first_axon_material: int | None = None, disconnected_axons=None,
                 ion_feedback: IonFeedbackConfig | None = None, blocked_axon_material=None):
        self.anatomy, self.fluid = anatomy, fluid
        self.damage_field = damage_field
        self.blocked_axon_material = blocked_axon_material
        self.buffer_glia = buffer_glia
        n = anatomy.neuron_count
        self._ion_config = ion_feedback
        self.ion_exchange = (None if ion_feedback is None else
                             SpikeIonExchange(anatomy.position_m[:n], fluid, ion_feedback))
        self.voltage_V = np.full(n, REST_V)
        self.excitatory_S = np.zeros(n)
        self.inhibitory_S = np.zeros(n)
        self._rest_V = np.full(n, REST_V)
        self._inhibitory_reversal_V = np.full(n, -THERMAL_V * log(126 / 8))
        self._drive_A = np.zeros(n)
        self._refractory_until = np.zeros(n)
        self._world = None
        self._revision = -1
        self._sites = {}
        self._pending = []
        self._glial_K_mol = np.zeros(len(anatomy.position_m) - n)
        self._glial_volume_fraction = np.ones_like(self._glial_K_mol)
        self.glial_remodeling = None
        self.microglia_activation = np.zeros(len(anatomy.position_m) - n)
        synapses = anatomy.synapses
        delays = np.asarray(synapses.delay_s, dtype=float)
        if not np.all(np.isfinite(delays) & (delays > 0)):
            raise ValueError("synaptic delays must be finite and positive")
        self._max_step_s = float(delays.min(initial=0.2e-3))
        if plasticity is None:
            self._outgoing = np.argsort(synapses.pre, kind="stable")
            self._outgoing_offsets = np.r_[0, np.cumsum(np.bincount(synapses.pre, minlength=n))]
        self._inactive_neurons = np.zeros(n, dtype=bool)
        self._inactive_glia = np.zeros(len(anatomy.position_m) - n, dtype=bool)
        self._disconnected_synapses = np.zeros(len(synapses.pre), dtype=bool)
        self.deliveries = np.zeros(2, dtype=np.int64)
        self._excitatory = anatomy.cell_type[:n] == EXCITATORY
        self.plasticity = (None if plasticity is None else SynapticPlasticity(
            anatomy, plasticity, first_axon_material=first_axon_material, disconnected_axons=disconnected_axons))

    @property
    def glial_K_mol(self):
        return float(self._glial_K_mol.sum())

    @property
    def glial_volume_fraction(self):
        """Per-cell volume relative to its undivided initial volume."""
        return self._glial_volume_fraction.copy()

    @property
    def glial_alive(self):
        return ~self._inactive_glia

    @property
    def living_somata(self):
        return ~np.r_[self._inactive_neurons, self._inactive_glia]

    def update_glial_population(self, positions_m, parents):
        """Move existing glia and append daughters without renumbering any existing cells."""
        anatomy = self.anatomy
        n = anatomy.neuron_count
        count = len(self._inactive_glia)
        parents = np.asarray(parents, dtype=np.intp)
        positions = np.asarray(positions_m, dtype=float)
        if (parents.ndim != 1 or np.any((parents < 0) | (parents >= count))
                or len(np.unique(parents)) != len(parents) or positions.shape != (count + len(parents), 3)
                or not np.isfinite(positions).all() or self._inactive_glia[parents].any()):
            raise ValueError("invalid glial population update")
        # Translate existing branches with their soma. A daughter starts at its own soma.
        tips = anatomy.process_tips_m.copy()
        tips[n:] += (positions[:count] - anatomy.position_m[n:])[:, None, :]
        children = np.broadcast_to(positions[count:, None, :], (len(parents), tips.shape[1], 3))
        self.anatomy = replace(anatomy, position_m=np.concatenate((anatomy.position_m[:n], positions)),
                               process_tips_m=np.concatenate((tips, children)),
                               cell_type=np.r_[anatomy.cell_type, anatomy.cell_type[n + parents]],
                               layer=np.r_[anatomy.layer, anatomy.layer[n + parents]])
        self._glial_K_mol[parents] *= .5
        self._glial_volume_fraction[parents] *= .5
        self._glial_K_mol = np.r_[self._glial_K_mol, self._glial_K_mol[parents]]
        self._glial_volume_fraction = np.r_[self._glial_volume_fraction, self._glial_volume_fraction[parents]]
        self._inactive_glia = np.r_[self._inactive_glia, np.zeros(len(parents), dtype=bool)]
        self.microglia_activation = np.r_[self.microglia_activation, np.zeros(len(parents))]
        self._revision = -1

    @property
    def intracellular_salt_change_mol(self):
        """Change in either tracked ion across the membrane pools since initialization."""
        deficit = self.ion_exchange.deficit_mol if self.ion_exchange is not None else 0.0
        return self.glial_K_mol - deficit

    def set_current(self, neuron: int, current_A: float):
        if not 0 <= neuron < self.anatomy.neuron_count or not isfinite(current_A):
            raise ValueError("invalid current or neuron ID")
        self._drive_A[neuron] = current_A

    def apply_lesion(self, *, neurons, synapses, glia=None):
        """Disable failed somata and suppress future delivery through failed connections."""
        failed_neurons = np.asarray(neurons, dtype=bool)
        failed_synapses = np.asarray(synapses, dtype=bool)
        if (failed_neurons.shape != self._inactive_neurons.shape
                or failed_synapses.shape != self._disconnected_synapses.shape):
            raise ValueError("lesion masks must match the registered anatomy")
        if glia is not None:
            failed_glia = np.asarray(glia, dtype=bool)
            if failed_glia.shape != self._inactive_glia.shape:
                raise ValueError("glial lesion mask must match the registered anatomy")
            self._inactive_glia |= failed_glia
            self.microglia_activation[failed_glia] = 0
        self._revision = -1
        self._inactive_neurons |= failed_neurons
        self._disconnected_synapses |= failed_synapses
        if self.plasticity is not None:
            self.plasticity.apply_lesion(failed_neurons, failed_synapses)
        self.voltage_V[failed_neurons] = REST_V
        self.excitatory_S[failed_neurons] = 0
        self.inhibitory_S[failed_neurons] = 0

    def attach(self, world, name):
        if self._world is not None:
            raise RuntimeError("tissue dynamics is already attached")
        required = {POTASSIUM, CHLORIDE}
        if self.damage_field is not None:
            required.add(self.damage_field)
        if not required <= world.field_specs.keys():
            raise ValueError("tissue dynamics requires extracellular ions and its configured injury field")
        medium = world.field_specs[POTASSIUM].porous
        if (medium is None or medium != world.field_specs[CHLORIDE].porous
                or medium.material != TISSUE or medium.volume_fraction != self.fluid.volume_fraction):
            raise ValueError("extracellular ion fields must share the tissue fluid volume")
        self._world = world
        if self.plasticity is not None:
            self.plasticity.attach(world)

    def _map_sites(self, world):
        groups = {}
        alive = np.flatnonzero(self.living_somata)
        for cell in alive:
            position = self.anatomy.position_m[cell]
            key, local = world.cell_location(position)
            groups.setdefault(key, []).append((cell, local))
        sites = {}
        for key, members in groups.items():
            cells, local = zip(*members)
            cells = np.asarray(cells, dtype=np.intp)
            local = tuple(np.asarray(local, dtype=np.intp).T)
            if np.any(world.block_materials(key)[local] != TISSUE):
                raise ValueError("cell somata must lie in tissue")
            kind = self.anatomy.cell_type[cells]
            sites[key] = _BlockSites(cells, local, np.flatnonzero(kind == ASTROCYTE),
                                     np.flatnonzero(kind == MICROGLIA))
        self._sites, self._revision = sites, world.mesh_revision

    def _buffer(self, site, potassium, chloride, volume, dt_s):
        local = tuple(axis[site.astrocytes] for axis in site.local)
        ids = site.cells[site.astrocytes] - self.anatomy.neuron_count
        old = self._glial_K_mol[ids]
        cells, inverse = np.unique(np.ravel_multi_index(local, potassium.shape), return_inverse=True)
        k, cl = potassium.ravel(), chloride.ravel()
        # Effective KCl storage with equal counterion exchange and no net creation of either solute.
        config = self._ion_config
        reference_volume = (config.glial_exchange_volume_m3 if config is not None else
                            self._world.grid.cell_volume_m3 * self.fluid.volume_fraction * 0.15)
        target = np.maximum(k[cells] / self.fluid.volume_fraction - 3.0, 0) * reference_volume
        if config is not None:
            target = np.minimum(target, config.glial_capacity_mol)
        tau = config.glial_tau_s if config is not None else 0.06
        requested = (target[inverse] * self._glial_volume_fraction[ids] - old) * -expm1(-dt_s / tau)
        release = np.minimum(requested, 0.0)
        uptake = np.maximum(requested, 0.0)
        demand = np.bincount(inverse, weights=uptake, minlength=len(cells))
        restored = -np.bincount(inverse, weights=release, minlength=len(cells))
        available = np.minimum(np.maximum(k[cells] - 3.0 * self.fluid.volume_fraction, 0), cl[cells]) * volume
        use = np.minimum(demand, restored + available)
        accepted = np.divide(use, demand, out=np.zeros_like(use), where=demand > 0)
        transfer = use - restored
        k[cells] -= transfer / volume
        cl[cells] -= transfer / volume
        self._glial_K_mol[ids] = old + release + uptake * accepted[inverse]
        return bool(np.any(transfer))

    def _update_environment(self, world, dt_s):
        if world.mesh_revision != self._revision:
            self._map_sites(world)
        alpha = self.fluid.volume_fraction
        n = self.anatomy.neuron_count
        for key, site in self._sites.items():
            neuronal = site.cells < n
            update_glia = self.buffer_glia and len(site.astrocytes)
            if neuronal.any() or update_glia:
                k = world.block_densities(POTASSIUM, key)
                cl = world.block_densities(CHLORIDE, key)
                if update_glia and self._buffer(site, k, cl, world.block_grid(key).cell_volume_m3, dt_s):
                    world.set_block_densities(POTASSIUM, key, k)
                    world.set_block_densities(CHLORIDE, key, cl)
                neurons = site.cells[neuronal]
                local = tuple(axis[neuronal] for axis in site.local)
                # Fraction of the K reversal shift approximates its contribution to resting voltage.
                k_out = np.maximum(k[local] / alpha, 1e-12)
                cl_out = np.maximum(cl[local] / alpha, 1e-12)
                self._rest_V[neurons] = REST_V + 0.25 * THERMAL_V * np.log(k_out / 3.0)
                self._inhibitory_reversal_V[neurons] = -THERMAL_V * np.log(cl_out / 8.0)
            if self.damage_field is not None and len(site.microglia):
                ids = site.cells[site.microglia] - n
                local = tuple(axis[site.microglia] for axis in site.local)
                injury = np.clip(world.block_densities(self.damage_field, key)[local], 0.0, 1.0)
                self.microglia_activation[ids] += (injury - self.microglia_activation[ids]) * -expm1(-dt_s / 0.1)

    def _deliver(self, time_s):
        s = self.anatomy.synapses
        while self._pending and self._pending[0][0] <= time_s:
            arrival_s, edge = heappop(self._pending)
            if self.plasticity is None:
                pre, post = s.pre[edge], s.post[edge]
                if self._disconnected_synapses[edge] or self._inactive_neurons[pre] or self._inactive_neurons[post]:
                    continue
                weight = s.conductance_S[edge]
            else:
                contact = self.plasticity.contacts[edge]
                if contact.phase != "active":
                    continue
                pre, post = contact.pre, contact.post
            gain, _ = self._damage_on_edge(edge, pre, post)
            if not gain:
                continue
            if self.plasticity is not None:
                weight = self.plasticity.arrived(edge, arrival_s)
            weight *= gain
            if self._excitatory[pre]:
                self.excitatory_S[post] += weight
                self.deliveries[0] += 1
            else:
                self.inhibitory_S[post] += weight
                self.deliveries[1] += 1

    def _damage_on_edge(self, edge, pre, post):
        if self.damage_field is None:
            return 1.0, 0.0
        a = self.anatomy
        contact = (a.synapses.position_m[edge] if self.plasticity is None else
                   self.plasticity.contacts[edge].position_m)
        route = (a.position_m[pre], a.axon_branch_m[pre], contact, a.position_m[post])
        injury = path_damage(self._world, route, self.damage_field, blocked_material=self.blocked_axon_material)
        if injury.block_at_m is not None:
            return 0.0, 0.0
        synapse_gain = 1 - float(np.clip(self._world.sample(self.damage_field, contact), 0, 1))
        return injury.gain * synapse_gain, injury.injury_integral_m / .35

    def _queue_edge(self, edge, time_s, pre, post, delay):
        gain, extra_s = self._damage_on_edge(edge, pre, post)
        if gain:
            heappush(self._pending, (time_s + delay + extra_s, edge))

    def _advance_membranes(self, world, name, start_s, end_s):
        duration = end_s - start_s
        e, i = duration / EXCITATORY_TAU_S, duration / INHIBITORY_TAU_S
        mean_e = self.excitatory_S * (-expm1(-e) / e)
        mean_i = self.inhibitory_S * (-expm1(-i) / i)
        conductance = LEAK_S + mean_e + mean_i
        steady = (LEAK_S * self._rest_V + mean_i * self._inhibitory_reversal_V + self._drive_A) / conductance
        start = np.minimum(np.maximum(start_s, self._refractory_until), end_s)
        rate = conductance / MEMBRANE_F
        initial = self.voltage_V
        next_voltage = steady + (initial - steady) * np.exp(-rate * (end_s - start))
        next_voltage[self._inactive_neurons] = REST_V
        fired = np.flatnonzero((initial < THRESHOLD_V) & (next_voltage >= THRESHOLD_V))
        crossing = start[fired] - np.log(
            (THRESHOLD_V - steady[fired]) / (initial[fired] - steady[fired])) / rate[fired]
        next_voltage[fired] = REST_V
        self.voltage_V[:] = next_voltage
        self._refractory_until[fired] = crossing + 2e-3
        self.excitatory_S *= np.exp(-e)
        self.inhibitory_S *= np.exp(-i)
        synapses = self.anatomy.synapses
        if self.plasticity is not None and len(fired) > 1:
            order = np.argsort(crossing, kind="stable")
            fired, crossing = fired[order], crossing[order]
        for neuron, time_s in zip(fired, crossing):
            world.emit(name, "spike", time_s=float(time_s), neuron=int(neuron))
            if self.ion_exchange is not None:
                self.ion_exchange.record(int(neuron))
            if self.plasticity is None:
                edges = self._outgoing[self._outgoing_offsets[neuron]:self._outgoing_offsets[neuron + 1]]
                for edge in edges:
                    if not self._disconnected_synapses[edge] and not self._inactive_neurons[synapses.post[edge]]:
                        self._queue_edge(edge, time_s, neuron, synapses.post[edge], synapses.delay_s[edge])
            else:
                self.plasticity.spiked(neuron, time_s)
                for edge in self.plasticity.outgoing(neuron):
                    contact = self.plasticity.contacts[edge]
                    self._queue_edge(edge, time_s, neuron, contact.post, contact.delay_s)

    def step(self, world, name, dt_s):
        if world is not self._world:
            raise RuntimeError("tissue dynamics is attached to a different world")
        self._update_environment(world, dt_s)
        time_s, end_s = world.time_s, world.time_s + dt_s
        self._deliver(time_s)
        while time_s < end_s:
            next_s = min(end_s, time_s + self._max_step_s)
            if self._pending:
                next_s = min(next_s, self._pending[0][0])
            if self.plasticity is not None:
                next_s = min(next_s, self.plasticity.next_update_s)
            self._advance_membranes(world, name, time_s, next_s)
            time_s = next_s
            self._deliver(time_s)
            if self.plasticity is not None and time_s == self.plasticity.next_update_s:
                self.plasticity.advance_structure(world, name, time_s)
        if self.ion_exchange is not None:
            self.ion_exchange.advance(world, end_s, self._inactive_neurons)
