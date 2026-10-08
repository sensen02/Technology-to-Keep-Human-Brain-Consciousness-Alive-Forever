"""Dedicated histamine conductance channel on top of ConductanceNetwork.

Motivation: `engine.graded_vision.GradedVision` reserves the parent network's
whole INHIBITORY conductance state for histamine, because ConductanceNetwork
offers only exc/inh channels. This module adds a THIRD, independent conductance
state `g_hist` with its own reversal and tau, advanced by the same backward-Euler
voltage update and exact exponential decay as the parent, so a graded
photoreceptor projection no longer has to borrow the inhibitory channel.

Scope and honesty
-----------------
* This is a bounded point-tier reduction. It is NOT fly-calibrated physiology.
  E_hist/tau_hist and every inherited parameter are ILLUSTRATIVE.
* `HistamineParams` is a strict superset of `ConductanceParams`; the inherited
  parameter set, its provenance text and the parent's numerics are unchanged.
* Zero-histamine equivalence is exact and enforced by test: with histamine
  events identically zero the voltage path is arithmetically the parent's
  (one extra zero addend), so v/ge/gi/adaptation/spike_count are bit-identical.
* Connectivity is caller-supplied. `set_histamine_connectivity` refuses edges
  that would also exist in the exc/inh matrices, so one spike cannot be counted
  twice through two channels.
* No transmitter-to-sign inference lives here; the caller must supply receptor
  assumptions (see engine.multirate_scenario for the explicit BANC mapping).
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy import sparse
from .neural_cond import ConductanceNetwork, ConductanceParams


@dataclass(frozen=True)
class HistamineParams(ConductanceParams):
    """ConductanceParams extended with one dedicated histamine channel.

    hist_reversal_mV / hist_tau_ms are the ONLY new degrees of freedom. The
    default reversal matches the chloride-style hyperpolarizing hypothesis used
    by engine.graded_vision (E_hist = -70 mV), which is a stated HYPOTHESIS and
    not a measurement of the fly photoreceptor-to-target synapse.
    """
    hist_reversal_mV: float = -70.0
    hist_tau_ms: float = 8.0

    def validate(self):
        super().validate()
        if not np.isfinite(self.hist_reversal_mV) or not np.isfinite(self.hist_tau_ms):
            raise ValueError('histamine parameters must be finite')
        if self.hist_tau_ms <= 0:
            raise ValueError('hist_tau_ms must be positive')

    def provenance(self):
        out = super().provenance()
        out.update({k: {'value': v, 'status': 'illustrative-hypothesis',
                        'scope': 'dedicated histamine channel; uncalibrated'}
                    for k, v in (('hist_reversal_mV', self.hist_reversal_mV),
                                 ('hist_tau_ms', self.hist_tau_ms))})
        return out


class HistamineNetwork(ConductanceNetwork):
    """ConductanceNetwork + independent `g_hist` state (parent source untouched).

    Event convention is the parent's: delivered at the START of an interval,
    returned spikes at the END. `hist_events_uS` increments g_hist immediately
    (no delay); recurrent histamine edges live in `W_hist` and use the parent's
    delay ring exactly like `We`/`Wi`.
    """

    def __init__(self, n_neurons, params=None, dt_ms=0.1):
        if params is None:
            params = HistamineParams()
        elif not isinstance(params, HistamineParams):
            raise TypeError('HistamineNetwork requires HistamineParams')
        super().__init__(n_neurons, params, dt_ms)
        self.g_hist = np.zeros(self.n)
        self.W_hist = sparse.csr_matrix((self.n, self.n))
        self.hist_events_delivered_uS = np.zeros(self.n)
        self.hist_increment_total_uS = 0.0

    def set_histamine_connectivity(self, hist_uS):
        """Set the recurrent histamine matrix; must not overlap We/Wi (no double count)."""
        w = sparse.csr_matrix(hist_uS, dtype=float, copy=True)
        w.sum_duplicates()
        if w.shape != (self.n, self.n):
            raise ValueError('connectivity shape mismatch')
        if np.any(~np.isfinite(w.data)) or np.any(w.data < 0):
            raise ValueError('conductance increments must be finite and nonnegative')
        w.eliminate_zeros()
        pattern = w.astype(bool)
        overlap = pattern.multiply(self.We.astype(bool)) + pattern.multiply(self.Wi.astype(bool))
        if overlap.nnz:
            raise ValueError('histamine edges overlap exc/inh edges: double counting')
        self.W_hist = w
        return self

    def step(self, current_nA=0.0, exc_events_uS=0.0, inh_events_uS=0.0,
             hist_events_uS=0.0, enabled=None):
        """One backward-Euler interval with three conductance channels.

        Voltage path (identical form to ConductanceNetwork plus one zero-safe
        histamine addend):

            G   = leak + ge + gi + g_hist
            rhs = C/dt*v + leak*rest + ge*Ee + gi*Ei + g_hist*Ehist + I - adapt
        """
        I = self._vector(current_nA, 'current')
        ee = self._vector(exc_events_uS, 'exc events', True)
        ei = self._vector(inh_events_uS, 'inh events', True)
        eh = self._vector(hist_events_uS, 'hist events', True)
        own = np.ones(self.n, bool) if enabled is None else np.asarray(enabled)
        if own.shape != (self.n,) or own.dtype != np.dtype(bool):
            raise ValueError('ownership mask must be a boolean vector of neuron count')
        p, dt = self.p, self.dt
        arrivals = self._ring[self._ring_i].copy()
        self._ring[self._ring_i] = False
        self.ge += self.We @ arrivals + ee
        self.gi += self.Wi @ arrivals + ei
        self.g_hist += self.W_hist @ arrivals + eh
        self.hist_events_delivered_uS = eh.copy()
        self.hist_increment_total_uS += float(eh.sum())
        self.ge[~own] = 0
        self.gi[~own] = 0
        self.g_hist[~own] = 0
        active = own & (self.t_ms + 1e-10 >= self.refractory_until_ms)
        G = p.leak_uS + self.ge + self.gi + self.g_hist
        rhs = (p.capacitance_nF / dt * self.v + p.leak_uS * p.rest_mV
               + self.ge * p.exc_reversal_mV + self.gi * p.inh_reversal_mV
               + self.g_hist * p.hist_reversal_mV
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
        self.g_hist *= np.exp(-dt / p.hist_tau_ms)
        self._ring[self._ring_i] = spiked
        self._ring_i = (self._ring_i + 1) % len(self._ring)
        return spiked

    def extra_state_hash(self):
        """Stable digest of the histamine-specific state (channel + matrices)."""
        h = 0
        for arr in (self.g_hist, self.hist_events_delivered_uS):
            h = (h * 1000003) ^ hash(np.asarray(arr, float).tobytes())
        return (h * 1000003) ^ hash(float(self.hist_increment_total_uS))


class GradedHistamineDriver:
    """Thin adapter making GradedVision's apply_to contract usable here.

    GradedVision.apply_to requires a network whose INHIBITORY channel matches
    its E_hist/tau and whose Wi and We[:,visual] are empty. This adapter
    presents exactly those attributes, but they are read-only VIEWS of the
    dedicated histamine channel: increments go to `hist_events_uS`, never to
    `gi`. Nothing in engine/graded_vision.py is modified or reimplemented; the
    graded release computation (photoreceptor adaptation, pure-histamine edge
    scaling, exact exponential increment) is GradedVision's own `step()`.
    """

    def __init__(self, net, visual_indices):
        if not isinstance(net, HistamineNetwork):
            raise TypeError('a HistamineNetwork is required')
        visual = np.asarray(visual_indices)
        if visual.ndim != 1 or not np.issubdtype(visual.dtype, np.integer):
            raise ValueError('visual indices must be a 1-D integer array')
        self.net = net
        self.p = _HistView(net.p)
        self.n = net.n
        self.dt = net.dt
        self.Wi = net.Wi
        self.We = net.We
        self._visual = visual
        self.last_increment_uS = np.zeros(net.n)

    def apply_to(self, grad, light, dt_ms=None):
        """Run grad.step() and route the increment into the histamine channel."""
        increment = grad.step(light, dt_ms)
        self.net.step(hist_events_uS=increment)
        self.last_increment_uS = np.asarray(increment, float)
        return self.last_increment_uS


class _HistView:
    """Read-only parameter view exposing the histamine channel as inh_reversal/tau."""

    def __init__(self, params):
        self._p = params

    @property
    def inh_reversal_mV(self):
        return self._p.hist_reversal_mV

    @property
    def inh_tau_ms(self):
        return self._p.hist_tau_ms


def histamine_channel_matches(grad, params):
    """True when a GradedVision instance's E_hist/tau equal a HistamineParams."""
    return (np.isclose(grad.e_hist, params.hist_reversal_mV)
            and np.isclose(grad.tau, params.hist_tau_ms))


def decay_factors(params, dt_ms):
    """Exact per-step decay factors (diagnostics; not used in the hot path)."""
    return {'exc': float(np.exp(-dt_ms / params.exc_tau_ms)),
            'inh': float(np.exp(-dt_ms / params.inh_tau_ms)),
            'hist': float(np.exp(-dt_ms / params.hist_tau_ms))}


def zero_histamine_equivalence_check(n_neurons, params, dt_ms, steps,
                                     seed=0, exc_uS=None, inh_uS=None):
    """Run a parent and a HistamineNetwork side by side with zero histamine.

    Returns dict with bit-equality flags for v/ge/gi/adaptation/spike_count.
    Intended for tests and demos; the networks are stepped in lockstep with
    identical external currents/events.
    """
    if not isinstance(params, HistamineParams):
        raise TypeError('HistamineParams required to build the parent parameter set')
    parent_params = replace(params)
    parent = ConductanceNetwork(n_neurons, parent_params, dt_ms)
    child = HistamineNetwork(n_neurons, params, dt_ms)
    if exc_uS is not None and inh_uS is not None:
        parent.set_connectivity(exc_uS, inh_uS)
        child.set_connectivity(exc_uS, inh_uS)
    rng = np.random.default_rng(seed)
    bit = {'v': True, 'ge': True, 'gi': True, 'adaptation': True, 'spike_count': True}
    for k in range(int(steps)):
        current = float(rng.normal(0, 1e-4))
        exc = 0.0 if k % 5 else 1e-4
        inh = 0.0 if k % 7 else 5e-5
        parent.step(current, exc, inh)
        child.step(current, exc, inh, 0.0)
        bit['v'] &= np.array_equal(parent.v.view(np.uint8), child.v.view(np.uint8))
        bit['ge'] &= np.array_equal(parent.ge.view(np.uint8), child.ge.view(np.uint8))
        bit['gi'] &= np.array_equal(parent.gi.view(np.uint8), child.gi.view(np.uint8))
        bit['adaptation'] &= np.array_equal(parent.adaptation.view(np.uint8),
                                            child.adaptation.view(np.uint8))
        bit['spike_count'] &= np.array_equal(parent.spike_count, child.spike_count)
    bit['all_equal'] = all(bit.values())
    bit['g_hist_all_zero'] = bool(np.all(child.g_hist == 0.)
                                  and child.hist_increment_total_uS == 0.)
    bit['parent_spikes'] = int(parent.spike_count.sum())
    bit['child_spikes'] = int(child.spike_count.sum())
    return bit


def parameter_provenance():
    """Module-level provenance block for reports."""
    p = HistamineParams()
    return {'status': 'illustrative point reduction; NOT fly-calibrated physiology',
            'inherited_parameters': asdict(ConductanceParams()),
            'histamine_channel': {'hist_reversal_mV': p.hist_reversal_mV,
                                  'hist_tau_ms': p.hist_tau_ms,
                                  'status': 'HYPOTHESIS: hyperpolarizing histamine channel, unmeasured'},
            'zero_histamine_equivalence': 'bit-exact with ConductanceNetwork by construction and test'}
