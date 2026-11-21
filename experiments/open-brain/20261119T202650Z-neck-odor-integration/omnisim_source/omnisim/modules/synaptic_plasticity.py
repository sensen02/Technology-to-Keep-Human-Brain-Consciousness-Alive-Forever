"""Event-driven STDP and local structural changes on the tissue contact graph."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from math import ceil, exp, floor, inf, isfinite

import numpy as np

from .tissue import EXCITATORY
from .tissue_geometry import move_toward, tissue_path_clear

_NEIGHBORS = tuple(product((-1, 0, 1), repeat=3))


@dataclass(frozen=True)
class PlasticityConfig:
    # Pair window and amplitudes are an excitatory model, independent of structural time.
    tau_plus_s: float = 0.020
    tau_minus_s: float = 0.020
    potentiation: float = 0.02
    depression: float = 0.025
    max_weight_ratio: float = 2.0
    structural_step_s: float = 1.0
    activity_tau_s: float = 20.0
    growth_activity_hz: float = 1.0
    max_contacts_per_axon: int = 12
    reach_m: float = 2e-6
    growth_m_s: float = 0.1e-6 / 60
    slide_m_s: float = 0.02e-6 / 60
    max_slide_m: float = 1e-6
    maturation_s: float = 3600.0
    prune_below_ratio: float = 0.20
    prune_after_s: float = 3600.0

    def __post_init__(self):
        positive = (self.tau_plus_s, self.tau_minus_s, self.structural_step_s, self.activity_tau_s,
                    self.reach_m, self.growth_m_s, self.maturation_s, self.prune_after_s)
        if (any(not isfinite(x) or x <= 0 for x in positive)
                or any(not 0 <= x <= 1 for x in (self.potentiation, self.depression, self.prune_below_ratio))
                or any(not isfinite(x) or x < 0 for x in (self.growth_activity_hz, self.slide_m_s, self.max_slide_m))
                or not isfinite(self.max_weight_ratio) or self.max_weight_ratio <= 1
                or not isinstance(self.max_contacts_per_axon, int) or self.max_contacts_per_axon < 0):
            raise ValueError("invalid synaptic plasticity parameters")


@dataclass
class SynapticContact:
    """The ID is its row in contacts. The row persists after pruning or injury."""

    pre: int
    post: int
    dendrite: int
    position_m: np.ndarray
    terminal_m: np.ndarray
    origin_m: np.ndarray
    reference_S: float
    weight_S: float
    delay_s: float
    phase: str = "active"
    active_since_s: float = 0.0
    stable_since_s: float | None = None
    weak_since_s: float | None = None
    last_arrival_s: float = -inf


def _project(point, start, end):
    delta = end - start
    length2 = float(delta @ delta)
    fraction = float(np.clip((point - start) @ delta / length2, 0, 1)) if length2 else 0.0
    return start + delta * fraction


class SynapticPlasticity:
    def __init__(self, anatomy, config, *, first_axon_material=None, disconnected_axons=None):
        self.anatomy, self.config = anatomy, config
        self._first_axon_material = first_axon_material
        neurons = anatomy.neuron_count
        self._inactive = np.zeros(neurons, dtype=bool)
        self._disconnected_axons = (np.zeros(neurons, dtype=bool) if disconnected_axons is None
                                   else np.asarray(disconnected_axons, dtype=bool).copy())
        if self._disconnected_axons.shape != (neurons,):
            raise ValueError("axon mask must match the anatomy")
        self._excitatory = anatomy.cell_type[:neurons] == EXCITATORY
        self._last_post_s = np.full(neurons, -inf)
        self._activity = np.zeros(neurons)
        self._outgoing = [{} for _ in range(neurons)]
        self._incoming = [{} for _ in range(neurons)]
        self._axon_contacts = np.zeros(neurons, dtype=np.intp)
        self._pairs = defaultdict(int)
        self.contacts: list[SynapticContact] = []
        self._changing = {}
        self._weak = {}
        self._mobile = {}
        self._bins = None
        self.next_update_s = inf
        s = anatomy.synapses
        self._original_count = len(s.pre)
        for pre, post, position, weight, delay in zip(s.pre, s.post, s.position_m, s.conductance_S, s.delay_s):
            pre, post = int(pre), int(post)
            soma = anatomy.position_m[post]
            tips = anatomy.process_tips_m[post]
            segment = tips - soma
            length2 = np.einsum("ij,ij->i", segment, segment)
            fraction = np.divide(segment @ (position - soma), length2, out=np.zeros(len(tips)), where=length2 > 0)
            distances = np.linalg.norm(soma + np.clip(fraction, 0, 1)[:, None] * segment - position, axis=1)
            branch = int(distances.argmin()) if len(tips) else -1
            self._append(SynapticContact(pre, post, branch, position.copy(), position.copy(), position.copy(),
                                         float(weight), float(weight), float(delay)))

    def attach(self, world):
        self.next_update_s = world.time_s + self.config.structural_step_s

    def outgoing(self, neuron):
        return self._outgoing[neuron]

    def _append(self, contact):
        edge = len(self.contacts)
        self.contacts.append(contact)
        self._pairs[contact.pre, contact.post] += 1
        self._axon_contacts[contact.pre] += 1
        if contact.phase == "active":
            self._activate(edge)
        else:
            self._changing[edge] = None
        return edge

    def _activate(self, edge):
        c = self.contacts[edge]
        c.phase = "active"
        self._changing.pop(edge, None)
        self._outgoing[c.pre][edge] = None
        self._incoming[c.post][edge] = None
        self._track_weak(edge, c.active_since_s)

    def _deactivate(self, edge):
        c = self.contacts[edge]
        self._outgoing[c.pre].pop(edge, None)
        self._incoming[c.post].pop(edge, None)
        self._weak.pop(edge, None)
        self._mobile.pop(edge, None)

    def _retire(self, edge, phase):
        c = self.contacts[edge]
        self._deactivate(edge)
        c.phase = phase
        self._changing.pop(edge, None)
        self._axon_contacts[c.pre] -= 1
        pair = c.pre, c.post
        self._pairs[pair] -= 1
        if not self._pairs[pair]:
            del self._pairs[pair]

    def apply_lesion(self, neurons, synapses):
        self._inactive |= neurons
        for edge, c in enumerate(self.contacts):
            if c.phase in ("removed", "lesioned"):
                continue
            if self._inactive[c.pre] or self._inactive[c.post] or (edge < self._original_count and synapses[edge]):
                self._retire(edge, "lesioned")

    def _track_weak(self, edge, time_s):
        c = self.contacts[edge]
        if c.weight_S < c.reference_S * self.config.prune_below_ratio:
            if c.weak_since_s is None:
                c.weak_since_s = time_s
                self._weak[edge] = None
        elif c.weak_since_s is not None:
            c.weak_since_s = None
            self._weak.pop(edge, None)

    def arrived(self, edge, time_s):
        c = self.contacts[edge]
        if c.phase != "active":
            return 0.0
        delivered = c.weight_S
        if self._excitatory[c.pre] and self._excitatory[c.post]:
            last = self._last_post_s[c.post]
            if c.active_since_s <= last < time_s:
                c.weight_S *= 1 - self.config.depression * exp((last - time_s) / self.config.tau_minus_s)
                self._track_weak(edge, time_s)
            c.last_arrival_s = time_s
        return delivered

    def spiked(self, neuron, time_s):
        last = self._last_post_s[neuron]
        tau = self.config.activity_tau_s
        self._activity[neuron] = self._activity[neuron] * exp((last - time_s) / tau) + 1 / tau
        self._last_post_s[neuron] = time_s
        if not self._excitatory[neuron]:
            return
        for edge in self._incoming[neuron]:
            c = self.contacts[edge]
            if self._excitatory[c.pre] and c.active_since_s <= c.last_arrival_s < time_s:
                fraction = self.config.potentiation * exp((c.last_arrival_s - time_s) / self.config.tau_plus_s)
                gap = c.reference_S * self.config.max_weight_ratio - c.weight_S
                c.weight_S += fraction * gap
                if fraction and self.config.slide_m_s and self.config.max_slide_m and c.dendrite >= 0:
                    self._mobile[edge] = None
                self._track_weak(edge, time_s)

    def _emit(self, world, name, kind, edge, time_s):
        c = self.contacts[edge]
        x, y, z = c.terminal_m
        world.emit(name, kind, time_s=time_s, edge=edge, pre=c.pre, post=c.post,
                   x_m=float(x), y_m=float(y), z_m=float(z), conductance_S=c.weight_S)

    def _axon_material(self, pre):
        return None if self._first_axon_material is None else self._first_axon_material + pre

    def _dendrite(self, c):
        return self.anatomy.position_m[c.post], self.anatomy.process_tips_m[c.post, c.dendrite]

    def _slide(self, world, edge, distance):
        c = self.contacts[edge]
        soma, tip = self._dendrite(c)
        best = _project(self.anatomy.axon_branch_m[c.pre], soma, tip)
        bound = move_toward(c.origin_m, best, self.config.max_slide_m)
        new = move_toward(c.position_m, bound, distance)
        if np.linalg.norm(new - c.position_m) < 1e-12:
            return False
        anchor = self.anatomy.axon_branch_m[c.pre]
        material = self._axon_material(c.pre)
        if not (tissue_path_clear(world, c.position_m, new, material) and tissue_path_clear(world, anchor, new, material)
                and tissue_path_clear(world, soma, new, material)):
            return False
        # Conduction time changes for future spikes; already queued arrivals retain their time.
        delta = np.linalg.norm(new - anchor) - np.linalg.norm(c.position_m - anchor)
        c.delay_s = max(0.5e-3, c.delay_s + delta / 0.35)
        c.position_m = c.terminal_m = new
        return True

    def _bin(self, point):
        return tuple(floor((x - origin) / self._bin_width) for x, origin in zip(point, self._bin_origin))

    def _build_index(self, world):
        self._bin_origin = world.grid.origin_m
        self._bin_width = 2 * self.config.reach_m
        self._bins = defaultdict(list)
        # Sparse Cartesian index of branch segments in world coordinates. It survives regridding.
        for post in np.flatnonzero(self._excitatory & ~self._inactive):
            start = self.anatomy.position_m[post]
            for branch, tip in enumerate(self.anatomy.process_tips_m[post]):
                count = max(1, ceil(np.linalg.norm(tip - start) / self.config.reach_m))
                keys = {self._bin(start + (tip - start) * (i / count)) for i in range(count + 1)}
                for key in keys:
                    self._bins[key].append((int(post), branch))

    def _propose(self, world, pre, active):
        source = self.anatomy.axon_branch_m[pre]
        material = self._axon_material(pre)
        key = self._bin(source)
        candidates = {entry for offset in _NEIGHBORS for entry in self._bins.get(
            tuple(a + b for a, b in zip(key, offset)), ())}
        eligible = []
        for post, branch in sorted(candidates):
            if pre == post or not active[post] or (pre, post) in self._pairs:
                continue
            soma, tip = self.anatomy.position_m[post], self.anatomy.process_tips_m[post, branch]
            point = _project(source, soma, tip)
            gap = float(np.linalg.norm(point - source))
            if gap <= self.config.reach_m:
                eligible.append((gap, post, branch, point))
        if not eligible or not tissue_path_clear(world, self.anatomy.position_m[pre], source, material):
            return None
        for gap, post, branch, point in sorted(eligible, key=lambda item: item[:3]):
            if (tissue_path_clear(world, source, point, material)
                    and tissue_path_clear(world, self.anatomy.position_m[post], point, material)):
                path = np.linalg.norm(source - self.anatomy.position_m[pre]) + gap
                return SynapticContact(pre, post, branch, point, source.copy(), source.copy(),
                                       1.8e-9, 0.0, float(0.5e-3 + path / .35), phase="growing")
        return None

    def advance_structure(self, world, name, time_s):
        p = self.config
        dt = p.structural_step_s
        self.next_update_s = time_s + dt
        rates = self._activity * np.exp((self._last_post_s - time_s) / p.activity_tau_s)
        active = (rates >= p.growth_activity_hz) & ~self._inactive & self._excitatory
        for edge in tuple(self._changing):
            c = self.contacts[edge]
            if c.phase == "retracting":
                c.terminal_m = move_toward(c.terminal_m, c.origin_m, p.growth_m_s * dt)
                if np.array_equal(c.terminal_m, c.origin_m):
                    self._retire(edge, "removed")
                    self._emit(world, name, "synapse_removed", edge, time_s)
                else:
                    self._emit(world, name, "synapse_retracted", edge, time_s)
                continue
            new = move_toward(c.terminal_m, c.position_m, p.growth_m_s * dt)
            if not active[c.pre] or not active[c.post] or not tissue_path_clear(world, c.terminal_m, new, self._axon_material(c.pre)):
                c.phase = "retracting"
                self._emit(world, name, "synapse_retracting", edge, time_s)
                continue
            if not np.array_equal(c.terminal_m, new):
                c.terminal_m = new
                self._emit(world, name, "synapse_extended", edge, time_s)
            if np.array_equal(c.terminal_m, c.position_m):
                if c.stable_since_s is None:
                    c.stable_since_s = time_s
                if time_s - c.stable_since_s >= p.maturation_s:
                    c.active_since_s = time_s
                    c.origin_m = c.position_m.copy()
                    c.weight_S = c.reference_S * 0.5
                    self._activate(edge)
                    self._emit(world, name, "synapse_matured", edge, time_s)
        for edge in tuple(self._weak):
            c = self.contacts[edge]
            if time_s - c.weak_since_s >= p.prune_after_s:
                self._deactivate(edge)
                c.phase = "retracting"
                c.origin_m = move_toward(c.terminal_m, self.anatomy.axon_branch_m[c.pre], p.reach_m)
                self._changing[edge] = None
                self._emit(world, name, "synapse_retracting", edge, time_s)
        for edge in tuple(self._mobile):
            c = self.contacts[edge]
            if active[c.pre] and active[c.post] and self._slide(world, edge, p.slide_m_s * dt):
                self._emit(world, name, "synapse_moved", edge, time_s)
            else:
                self._mobile.pop(edge, None)
        eligible = active & ~self._disconnected_axons & (self._axon_contacts < p.max_contacts_per_axon)
        if np.any(eligible) and self._bins is None:
            self._build_index(world)
        for pre in np.flatnonzero(eligible):
            contact = self._propose(world, int(pre), active)
            if contact is not None:
                edge = self._append(contact)
                self._emit(world, name, "synapse_growing", edge, time_s)
