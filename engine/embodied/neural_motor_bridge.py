"""Engineered, literature-informed spike-to-intent bridge (not anatomical reconstruction).

The hypothesis is bilateral descending activity modulates steering. Rate gains,
filter time constant, sign and locomotion permission are DECLARED engineering
choices, not inferred synapses or a validated DNa01/DNa02 spike-to-velocity law.
Caller must supply explicit neuron identity/provenance; no automatic type guesses.
No baseline walking is hidden here: locomotion permission is an explicit input.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class BridgeConfig:
    filter_tau_s: float = 0.05
    steering_rate_scale_hz: float = 100.0
    max_turn: float = 0.8
    baseline_left_hz: float = 0.0
    baseline_right_hz: float = 0.0
    left_minus_right_sign: float = 1.0

    def validate(self):
        values = list(asdict(self).values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError('configuration must be finite')
        if self.filter_tau_s <= 0 or self.steering_rate_scale_hz <= 0:
            raise ValueError('time constant and rate scale must be positive')
        if not 0 <= self.max_turn <= 1 or self.left_minus_right_sign not in (-1, 1):
            raise ValueError('invalid turn amplitude or sign')
        return self


class SpikeIntentBridge:
    """Consume chronological binary spikes; output bounded engineering intent.

Rates are exponential causal filters of per-neuron bin rate. Updating bin by bin
makes chunking invariant. A cut clears filter state to prevent old activity leaking
through the severed route. No fitting on the evaluated test episode occurs.
"""
    def __init__(self, n_neurons, left_indices, right_indices, *, provenance, config=None):
        self.n = int(n_neurons)
        self.left = np.asarray(left_indices, dtype=int)
        self.right = np.asarray(right_indices, dtype=int)
        if self.n <= 0 or not provenance:
            raise ValueError('positive neuron count and explicit provenance required')
        for group in (self.left, self.right):
            if group.ndim != 1 or not group.size or len(np.unique(group)) != group.size:
                raise ValueError('groups must be nonempty distinct one-dimensional indices')
            if np.any(group < 0) or np.any(group >= self.n):
                raise ValueError('group index outside network')
        if np.intersect1d(self.left, self.right).size:
            raise ValueError('left/right groups overlap')
        self.provenance = str(provenance)
        self.cfg = (config or BridgeConfig()).validate()
        self.reset()

    def reset(self):
        self.rates = np.zeros(2)
        self.elapsed_s = 0.0

    def advance(self, spikes, *, spike_dt_s, locomotion_permission=0.0, connected=True):
        sp = np.asarray(spikes)
        if sp.ndim != 2 or sp.shape[1] != self.n or sp.shape[0] == 0:
            raise ValueError('spikes must be nonempty [time,neuron] matrix')
        if not np.all((sp == 0) | (sp == 1)):
            raise ValueError('spikes must be binary and finite')
        if not math.isfinite(spike_dt_s) or spike_dt_s <= 0:
            raise ValueError('positive finite spike timestep required')
        if not math.isfinite(locomotion_permission) or not 0 <= locomotion_permission <= 1:
            raise ValueError('explicit locomotion permission must lie in [0,1]')
        decay = math.exp(-spike_dt_s / self.cfg.filter_tau_s)
        for row in sp:
            if connected:
                observed = np.array([row[self.left].mean(), row[self.right].mean()]) / spike_dt_s
                self.rates = decay * self.rates + (1 - decay) * observed
            else:
                self.rates[:] = 0
            self.elapsed_s += spike_dt_s
        delta = ((self.rates[0] - self.cfg.baseline_left_hz)
                 - (self.rates[1] - self.cfg.baseline_right_hz))
        raw = self.cfg.left_minus_right_sign * delta / self.cfg.steering_rate_scale_hz
        turn = float(np.clip(raw, -self.cfg.max_turn, self.cfg.max_turn)) if connected else 0.0
        # Complete bridge cut removes both drive and steering; silence only affects
        # steering when caller explicitly supplies external locomotion permission.
        drive = float(locomotion_permission) if connected else 0.0
        return {'forward': drive, 'turn': drive * turn,
                'rate_left_hz': float(self.rates[0]), 'rate_right_hz': float(self.rates[1]),
                'saturated': bool(connected and abs(raw) > self.cfg.max_turn),
                'elapsed_s': self.elapsed_s, 'connected': bool(connected),
                'locomotion_source': 'EXTERNAL_DECLARED_PERMISSION_NOT_NEURAL_INITIATION',
                'identity_provenance': self.provenance}
