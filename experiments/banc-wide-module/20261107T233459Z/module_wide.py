"""Whole-BANC strict-transmitter module: same construction rule as the frozen 20k core.

Every edge whose presynaptic cell has a single verified transmitter among
acetylcholine / gaba / glutamate is kept; raw source synapse counts are multiplied by
0.020 uS and divided by the target neuron's total incoming raw count (the frozen
convention), so each target with in-edges receives exactly 0.020 uS across both
receptor types. Excitatory = acetylcholine, inhibitory = gaba/glutamate. Adaptation
conductance, delays, thresholds and time constants come unchanged from the frozen
192931Z core.

No edge is invented, no sign is assigned from measurement, and the whole module is an
uncalibrated engineering reconstruction of a connectome subset, not a fly brain.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys
import numpy as np
from scipy import sparse

CACHE = Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz')
FROZEN = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py')
FROZEN_SHA256 = 'c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250'
CAND = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MODULE_CANDIDATE.npz')
TOTAL_US = .02
PURE = ('acetylcholine', 'gaba', 'glutamate')

if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA256:
    raise ValueError('frozen adaptation core changed')
_NAME = '_wide_frozen_192931Z'
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, FROZEN)
    frozen = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = frozen
    spec.loader.exec_module(frozen)
frozen = sys.modules[_NAME]


def load_cache():
    with np.load(CACHE, allow_pickle=False) as z:
        return dict(root_ids=z['root_ids'].copy(), pre=z['pre'].copy(), post=z['post'].copy(),
                    syn=z['syn'].astype(np.float64), nt=z['ann_nt_verified'].copy(),
                    super_class=z['ann_super_class'].copy(), nerve=z['ann_nerve'].copy())


def strict_edges(cache):
    keep = np.isin(cache['nt'][cache['pre']], PURE)
    return keep


def build(cache=None, cell_mask=None):
    """Return (We, Wi, stats). cell_mask optionally restricts to a subset of cells (guard)."""
    cache = cache or load_cache()
    keep = strict_edges(cache)
    pre, post = cache['pre'][keep].astype(np.int64), cache['post'][keep].astype(np.int64)
    syn = cache['syn'][keep]
    nt = cache['nt'][pre]
    if cell_mask is None:
        n = cache['root_ids'].size
    else:
        n = int(cell_mask.sum())
        remap = np.full(cache['root_ids'].size, -1, dtype=np.int64)
        remap[cell_mask] = np.arange(n)
        inside = (remap[pre] >= 0) & (remap[post] >= 0)
        pre, post, syn, nt = remap[pre[inside]], remap[post[inside]], syn[inside], nt[inside]
    incoming = np.bincount(post, weights=syn, minlength=n)
    if (incoming[post] <= 0).any():
        raise ValueError('target without incoming weight')
    weight = TOTAL_US * syn / incoming[post]
    exc = nt == 'acetylcholine'
    We = sparse.csr_matrix((weight[exc], (post[exc], pre[exc])), shape=(n, n))
    Wi = sparse.csr_matrix((weight[~exc], (post[~exc], pre[~exc])), shape=(n, n))
    row_sums = np.asarray((We + Wi).sum(axis=1)).ravel()
    nz = row_sums > 0
    stats = dict(cells=int(n), strict_edges=int(pre.size), exc_edges=int(exc.sum()), inh_edges=int((~exc).sum()),
                 total_conductance_uS=float(weight.sum()), targets_with_in_edges=int(nz.sum()),
                 row_sum_max_abs_deviation_uS=float(np.max(np.abs(row_sums[nz] - TOTAL_US))))
    return We, Wi, stats


class WideCircuit:
    """Frozen equations, whole-cache identity ordering."""

    def __init__(self, We, Wi, dt_ms=.5):
        self.dt_ms = float(dt_ms)
        self.filter_tau_ms = 20.0
        self.net = frozen.AdaptationConductanceNetwork(We.shape[0], dt_ms=self.dt_ms)
        self.net.set_connectivity(We, Wi)
        n = We.shape[0]
        self.filtered_rates_hz = np.zeros(n, dtype=float)
        self._sensory_mask = np.zeros(n, dtype=bool)
        self._descending_mask = np.zeros(n, dtype=bool)

    def advance(self, sensory_current_nA, descending_current_nA, duration_ms=5.):
        s = np.asarray(sensory_current_nA, dtype=float)
        d = np.asarray(descending_current_nA, dtype=float)
        n = self.net.n
        for x, mask, role in ((s, self._sensory_mask, 'sensory'), (d, self._descending_mask, 'descending')):
            if x.shape != (n,) or not np.isfinite(x).all() or np.any(x < 0):
                raise ValueError(role + ' current must be finite nonnegative (n,) nA')
            if np.any(x[~mask] != 0):
                raise ValueError(role + ' current outside allowed identities')
        ratio = float(duration_ms) / self.dt_ms
        if not np.isfinite(ratio) or ratio <= 0 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
            raise ValueError('duration_ms must be a positive integer number of steps')
        current = s + d
        decay = np.exp(-self.dt_ms / self.filter_tau_ms)
        for _ in range(int(round(ratio))):
            spikes = self.net.step(current)
            self.filtered_rates_hz *= decay
            self.filtered_rates_hz += (1000. / self.filter_tau_ms) * spikes
            if not all(np.isfinite(x).all() for x in (self.net.v, self.net.ge, self.net.gi, self.net.g_adapt, self.filtered_rates_hz)):
                raise FloatingPointError('nonfinite neural state')
        return self.filtered_rates_hz


def local_to_cache(cache_local_indices, cand):
    """Map 20k-local indices to whole-cache indices."""
    return np.asarray(cand['cache_indices'])[np.asarray(cache_local_indices)]
