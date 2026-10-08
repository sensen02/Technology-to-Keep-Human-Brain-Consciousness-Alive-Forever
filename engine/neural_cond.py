"""Conductance point tier, explicit units and illustrative parameters.

C[nF] dV[mV]/dt[ms] = g[uS]*(E-V)[mV] + I[nA] - adaptation[nA].
Backward Euler voltage with exact exponential conductance/adaptation decay.
Threshold/reset and adaptation are reductions, NOT calibrated fly physiology.
Matrices carry nonnegative conductance increments, indexed (post, pre).
No transmitter-to-sign inference: caller must supply receptor assumptions.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ConductanceParams:
    capacitance_nF: float = 0.02
    leak_uS: float = 0.001
    rest_mV: float = -65.0
    threshold_mV: float = -50.0
    reset_mV: float = -65.0
    exc_reversal_mV: float = 0.0
    inh_reversal_mV: float = -70.0
    exc_tau_ms: float = 3.0
    inh_tau_ms: float = 8.0
    adaptation_tau_ms: float = 100.0
    adaptation_increment_nA: float = 0.002
    refractory_ms: float = 2.0
    delay_ms: float = 2.0

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('parameters must be finite')
        for k in ('capacitance_nF', 'leak_uS', 'exc_tau_ms', 'inh_tau_ms',
                  'adaptation_tau_ms', 'delay_ms'):
            if getattr(self, k) <= 0:
                raise ValueError(k + ' must be positive')
        if self.refractory_ms < 0 or self.adaptation_increment_nA < 0:
            raise ValueError('negative refractory period or adaptation increment')
        if self.reset_mV >= self.threshold_mV:
            raise ValueError('reset must be below threshold')

    def provenance(self):
        return {k: {'value': v, 'status': 'illustrative',
                    'scope': 'uncalibrated point reduction, not fly measurement'}
                for k, v in asdict(self).items()}


class ConductanceNetwork:
    def __init__(self, n_neurons, params=None, dt_ms=0.1):
        if int(n_neurons) != n_neurons or n_neurons < 1:
            raise ValueError('positive integer neuron count required')
        self.n = int(n_neurons)
        self.p = params or ConductanceParams()
        self.p.validate()
        self.dt = float(dt_ms)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('positive finite dt required')
        d = self.p.delay_ms / self.dt
        if d < 1 or not np.isclose(d, round(d), rtol=0, atol=1e-9):
            raise ValueError('delay must be an integer number of steps, at least one')
        self.delay_steps = int(round(d))
        self._ring = np.zeros((self.delay_steps + 1, self.n), dtype=bool)
        self._ring_i = 0
        self.v = np.full(self.n, self.p.rest_mV)
        self.ge = np.zeros(self.n)
        self.gi = np.zeros(self.n)
        self.adaptation = np.zeros(self.n)
        self.refractory_until_ms = np.zeros(self.n)
        self.spike_count = np.zeros(self.n, dtype=np.int64)
        self.We = sparse.csr_matrix((self.n, self.n))
        self.Wi = sparse.csr_matrix((self.n, self.n))
        self.t_ms = 0.0
        self.steps = 0

    def set_connectivity(self, exc_uS, inh_uS):
        matrices = []
        for w in (exc_uS, inh_uS):
            w = sparse.csr_matrix(w, dtype=float, copy=True)
            w.sum_duplicates()
            if w.shape != (self.n, self.n):
                raise ValueError('connectivity shape mismatch')
            if np.any(~np.isfinite(w.data)) or np.any(w.data < 0):
                raise ValueError('conductance increments must be finite and nonnegative')
            matrices.append(w)
        self.We, self.Wi = matrices
        return self

    def _vector(self, x, name, nonnegative=False):
        a = np.broadcast_to(np.asarray(x, float), (self.n,))
        if not np.all(np.isfinite(a)) or (nonnegative and np.any(a < 0)):
            raise ValueError(name + ' invalid')
        return a

    def step(self, current_nA=0.0, exc_events_uS=0.0, inh_events_uS=0.0,
             enabled=None):
        """Events arrive at interval START, returned spikes at interval END.

        A spike emitted at t arrives at t+delay. External increments are immediate.
        ``enabled`` assigns point-tier ownership; disabled cells do not spike or
        integrate. Incoming events to disabled targets are discarded, so a hybrid
        coordinator must route those edges to their detailed owner instead.
        Disabled voltage/adaptation are frozen. This is a fixed-ownership API,
        not a state-transfer protocol for dynamic coarse/fine switching.
        """
        I = self._vector(current_nA, 'current')
        ee = self._vector(exc_events_uS, 'exc events', True)
        ei = self._vector(inh_events_uS, 'inh events', True)
        own = np.ones(self.n, bool) if enabled is None else np.asarray(enabled)
        if own.shape != (self.n,) or own.dtype != np.dtype(bool):
            raise ValueError('ownership mask must be a boolean vector of neuron count')
        p, dt = self.p, self.dt
        # Slot at step k contains spikes emitted at end of step k-delay-1.
        # Reuse the consumed slot in a delay+1 ring: end->start delay is exact.
        arrivals = self._ring[self._ring_i].copy()
        self._ring[self._ring_i] = False
        self.ge += self.We @ arrivals + ee
        self.gi += self.Wi @ arrivals + ei
        self.ge[~own] = 0
        self.gi[~own] = 0
        active = own & (self.t_ms + 1e-10 >= self.refractory_until_ms)
        G = p.leak_uS + self.ge + self.gi
        rhs = (p.capacitance_nF / dt * self.v + p.leak_uS * p.rest_mV
               + self.ge * p.exc_reversal_mV + self.gi * p.inh_reversal_mV
               + I - self.adaptation)
        self.v[active] = (rhs / (p.capacitance_nF / dt + G))[active]
        self.v[own & ~active] = p.reset_mV
        spiked = active & (self.v >= p.threshold_mV)
        self.steps += 1
        self.t_ms = self.steps * dt
        self.v[spiked] = p.reset_mV
        self.refractory_until_ms[spiked] = self.t_ms + p.refractory_ms
        self.spike_count += spiked
        self.adaptation[own] *= np.exp(-dt / p.adaptation_tau_ms)
        self.adaptation[spiked] += p.adaptation_increment_nA
        self.ge *= np.exp(-dt / p.exc_tau_ms)
        self.gi *= np.exp(-dt / p.inh_tau_ms)
        # Delay >= one step; next slot is consumed after delay full intervals.
        # An extra slot is needed to distinguish start from end timestamps.
        self._ring[self._ring_i] = spiked
        self._ring_i = (self._ring_i + 1) % len(self._ring)
        return spiked

    def run(self, duration_ms, current_nA=0.0, probe_indices=()):
        ratio = duration_ms / self.dt
        if not np.isfinite(ratio) or ratio <= 0 or not np.isclose(ratio, round(ratio)):
            raise ValueError('duration must be positive integer steps')
        probes = np.asarray(probe_indices, dtype=int)
        if probes.ndim != 1 or np.any(probes < 0) or np.any(probes >= self.n):
            raise ValueError('invalid probes')
        n = int(round(ratio))
        rates = np.empty(n)
        volts = np.empty((n, len(probes)))
        start = self.t_ms
        for k in range(n):
            drive = current_nA(self.t_ms) if callable(current_nA) else current_nA
            s = self.step(drive)
            rates[k] = s.mean() * 1000 / self.dt
            volts[k] = self.v[probes]
        return {'times_ms': start + np.arange(1, n+1) * self.dt,
                'rate_hz': rates, 'probe_mV': volts,
                'mean_rate_hz': float(rates.mean()),
                'parameter_provenance': self.p.provenance()}
