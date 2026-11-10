"""Versioned adaptation-scale hypothesis: per-neuron spike-increment multiplier.

New file, new timestamp directory. Only the per-spike adaptation increment of the
selected neurons is multiplied; every other equation, parameter, state and input
stays in the frozen 192931Z core. scale==1 everywhere must reproduce that core
bit-exactly, which the runner checks before any grid run.

This is an uncalibrated engineering hypothesis, not measured fly physiology, not
an external oscillator and not a new synapse.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys
import numpy as np

FROZEN = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py')
FROZEN_SHA256 = 'c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250'
if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA256:
    raise ValueError('frozen adaptation core changed')
_NAME = '_adaptation_scale_frozen_192931Z'
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, FROZEN)
    frozen = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = frozen
    spec.loader.exec_module(frozen)
frozen = sys.modules[_NAME]


class ScaledAdaptationNetwork(frozen.AdaptationConductanceNetwork):
    """Frozen network plus a fixed per-neuron multiplier on the spike increment."""

    def __init__(self, n_neurons, dt_ms=.5, increment_scale=None):
        super().__init__(n_neurons, dt_ms=dt_ms)
        if increment_scale is None:
            increment_scale = np.ones(n_neurons)
        scale = np.asarray(increment_scale, dtype=float)
        if scale.shape != (n_neurons,) or not np.isfinite(scale).all() or np.any(scale < 1.):
            raise ValueError('increment scale must be finite (n,) with scale >= 1')
        self.increment_scale = scale
        self.increment_scale.flags.writeable = False

    def step(self, current_nA=0.0):
        spiked = super().step(current_nA)
        extra = self.increment_scale - 1.
        if extra.any():
            self.g_adapt[spiked] += self.adaptation_params.spike_increment_uS * extra[spiked]
        return spiked


class ScaledLocalCircuit(frozen.LocalCircuit):
    """Same identity/edge/input API as the frozen circuit, scaled adaptation only.

    Gained adaptation still decays with the frozen tau and stays nonnegative, so the
    documented -80mV lower bound argument is unchanged.
    """

    def __init__(self, increment_scale=None, **kwargs):
        self.increment_scale = None if increment_scale is None else np.asarray(increment_scale, float)
        super().__init__(**kwargs)

    def reset(self):
        self.net = ScaledAdaptationNetwork(20000, dt_ms=self.dt_ms, increment_scale=self.increment_scale)
        self.net.set_connectivity(self._We, self._Wi)
        self.filtered_rates_hz = np.zeros(20000, dtype=np.float64)
        return self
