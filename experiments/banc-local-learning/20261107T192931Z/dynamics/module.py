"""Versioned conductance-adaptation hypothesis, not measured fly physiology.

Reuse frozen original identity/topology loader and point parameters. Replace only
adaptation-current subtraction with an explicit nonnegative conductance at -80mV.
No voltage clipping, hidden drive, new synapses or changes to the original engine.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import importlib.util
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/dynamics/module.py')
OLD_SHA256 = '3443853d8ebe725f8fedcb2e022b53ca4ed470d87f06a8d3ed71fa049f150676'
if hashlib.sha256(OLD.read_bytes()).hexdigest() != OLD_SHA256:
    raise ValueError('frozen original identity loader changed')
_NAME = '_banc_frozen_local_185910Z'
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, OLD)
    old = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = old
    spec.loader.exec_module(old)
old = sys.modules[_NAME]
ConductanceParams = old.ConductanceParams


@dataclass(frozen=True)
class AdaptationConductanceParams:
    """Separate typed unit contract; all values are illustrative, fixed hypothesis."""
    reversal_mV: float = -80.0
    tau_ms: float = 100.0
    spike_increment_uS: float = 0.002 / 15.0

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('adaptation conductance parameters must be finite')
        if self.tau_ms <= 0 or self.spike_increment_uS < 0:
            raise ValueError('invalid adaptation conductance parameters')


class AdaptationConductanceNetwork(old.ConductanceNetwork):
    """Original connectivity/state convention, new explicit conductance mechanism.

    p retains immutable source ConductanceParams for membrane/receptor constants;
    p.adaptation_increment_nA is historical metadata and is NOT used in dynamics.
    Active adaptation state g_adapt has units uS, never aliased as adaptation nA.
    """
    def __init__(self, n_neurons, dt_ms=.5):
        super().__init__(n_neurons, ConductanceParams(), dt_ms=dt_ms,
                         voltage_method='backward_euler')
        self.adaptation_params = AdaptationConductanceParams()
        self.adaptation_params.validate()
        del self.adaptation
        self.g_adapt = np.zeros(self.n, dtype=float)
        self.last_spikes = np.zeros(self.n, dtype=bool)

    def step(self, current_nA=0.0):
        I = self._vector(current_nA, 'current', nonnegative=True)
        p, a, dt = self.p, self.adaptation_params, self.dt
        if not all(np.isfinite(g).all() and np.all(g >= 0) for g in (self.ge, self.gi, self.g_adapt)):
            raise ValueError('nonnegative finite conductance state required')
        arrivals = self._ring[self._ring_i].copy()
        self._ring[self._ring_i] = False
        self.ge += self.We @ arrivals
        self.gi += self.Wi @ arrivals
        active = self.t_ms + 1e-10 >= self.refractory_until_ms
        # Convex combination of previous V and four reversal potentials, plus
        # nonnegative current divided by positive denominator. No voltage clip.
        G = p.leak_uS + self.ge + self.gi + self.g_adapt
        rhs = (p.capacitance_nF / dt * self.v + p.leak_uS * p.rest_mV
               + self.ge * p.exc_reversal_mV + self.gi * p.inh_reversal_mV
               + self.g_adapt * a.reversal_mV + I)
        self.v[active] = (rhs / (p.capacitance_nF / dt + G))[active]
        self.v[~active] = p.reset_mV
        spiked = active & (self.v >= p.threshold_mV + self.threshold_offset_mV)
        self.steps += 1
        self.t_ms = self.steps * dt
        self.v[spiked] = p.reset_mV
        self.refractory_until_ms[spiked] = self.t_ms + p.refractory_ms
        self.spike_count += spiked
        self.g_adapt *= np.exp(-dt / a.tau_ms)
        self.g_adapt[spiked] += a.spike_increment_uS
        self.ge *= np.exp(-dt / p.exc_tau_ms)
        self.gi *= np.exp(-dt / p.inh_tau_ms)
        self._ring[self._ring_i] = spiked
        self._ring_i = (self._ring_i + 1) % len(self._ring)
        self.last_spikes = spiked.copy()
        return spiked


class LocalCircuit(old.LocalCircuit):
    """Same 20k identity/323 motor API; negative external currents rejected.

    Fixed .02uS incoming total; dt .5ms or preregistered refinement .25ms only.
    Lower-bound proof assumes reset-origin state and nonnegative currents and g;
    public net mutation outside that domain invalidates the proof.
    """
    def __init__(self, *, total_conductance_uS=.02, dt_ms=.5):
        if total_conductance_uS != .02:
            raise ValueError('this hypothesis has fixed .02uS, no gain search')
        super().__init__(total_conductance_uS=.02, dt_ms=dt_ms)

    def reset(self):
        self.net = AdaptationConductanceNetwork(20000, dt_ms=self.dt_ms)
        self.net.set_connectivity(self._We, self._Wi)
        self.filtered_rates_hz = np.zeros(20000, dtype=float)
        return self

    def advance(self, sensory_current_nA, descending_current_nA, duration_ms=5.):
        s = np.asarray(sensory_current_nA, dtype=float)
        d = np.asarray(descending_current_nA, dtype=float)
        for x, mask, role in ((s, self._sensory_mask, 'sensory'), (d, self._descending_mask, 'descending')):
            if x.shape != (20000,) or not np.isfinite(x).all() or np.any(x < 0):
                raise ValueError(role + ' current must be finite nonnegative (20000,) nA')
            if np.any(x[~mask] != 0):
                raise ValueError(role + ' current outside allowed identities')
        ratio = float(duration_ms) / self.dt_ms
        if not np.isfinite(ratio) or ratio <= 0 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
            raise ValueError('duration_ms must be a positive integer number of steps')
        current = s + d
        if not np.isfinite(current).all():
            raise ValueError('summed current is not finite')
        decay = np.exp(-self.dt_ms / self.filter_tau_ms)
        for _ in range(int(round(ratio))):
            spikes = self.net.step(current)
            self.filtered_rates_hz *= decay
            self.filtered_rates_hz += (1000./self.filter_tau_ms)*spikes
            if not all(np.isfinite(x).all() for x in (self.net.v, self.net.ge, self.net.gi, self.net.g_adapt, self.filtered_rates_hz)):
                raise FloatingPointError('nonfinite neural state')
        return self.filtered_rates_hz[self.mn_local].copy()
