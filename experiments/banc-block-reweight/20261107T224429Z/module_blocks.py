"""Versioned block-reweight hypothesis: three bounded scalars on EXISTING leg-pair edges.

Rebuilds the strict source matrices exactly as the frozen 192931Z core does
(excitatory = acetylcholine, inhibitory = gaba/glutamate, per-row 0.020 uS total),
except that each edge's raw synapse count is multiplied by the scalar of its
leg-pair class and every postsynaptic row is renormalised back to 0.020 uS
(proposal V1). All scalars 1 must reproduce the frozen matrices exactly.

No edge is created or removed; no input, time constant, threshold or sign changes.
These are uncalibrated engineering weights, not measured synaptic strengths.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
from scipy import sparse

FROZEN = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py')
FROZEN_SHA256 = 'c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250'
if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA256:
    raise ValueError('frozen adaptation core changed')
_NAME = '_block_reweight_frozen_192931Z'
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, FROZEN)
    frozen = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = frozen
    spec.loader.exec_module(frozen)
frozen = sys.modules[_NAME]

SOURCE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface')
POOLS = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/leg-pools/POOLS_AND_REACH.npz')
LEG_INDEX = {'lf': 0, 'lh': 1, 'lm': 2, 'rf': 3, 'rh': 4, 'rm': 5}
CONTRA_SAME_ROW = {frozenset(('lf', 'rf')), frozenset(('lm', 'rm')), frozenset(('lh', 'rh'))}
FRONT_HIND = {frozenset(('lf', 'lh')), frozenset(('rf', 'rh'))}
TRIPOD_DIAGONAL = {frozenset(('lf', 'rh')), frozenset(('rf', 'lh'))}
TOTAL_US = .02


def edge_classes():
    """Per-strict-edge class id: 0 contra-same-row, 1 front-hind, 2 tripod-diagonal, 3 other."""
    with np.load(POOLS, allow_pickle=False) as z:
        bucket = z['bucket_id'].astype(np.int64)
    leg_of_bucket = {0: 'lf', 1: 'lh', 2: 'lm', 3: 'rf', 4: 'rh', 5: 'rm'}
    with np.load(SOURCE / 'LOCAL_MODULE_CANDIDATE.npz', allow_pickle=False) as z:
        pre, post, syn, nt = (z[k] for k in ('pre_local', 'post_local', 'synapse_counts', 'source_verified_nt'))
        eligible = np.isin(nt, ['acetylcholine', 'gaba', 'glutamate'])
    pre, post, syn, nt = pre[eligible], post[eligible], syn[eligible].astype(np.float64), nt[eligible]
    bp, bq = bucket[pre], bucket[post]
    legp = np.array([leg_of_bucket.get(int(b), '') for b in bp], dtype=object)
    legq = np.array([leg_of_bucket.get(int(b), '') for b in bq], dtype=object)
    cls = np.full(pre.size, 3, dtype=np.int64)
    pair_known = (legp != '') & (legq != '') & (legp != legq)
    for i in np.flatnonzero(pair_known):
        s = frozenset((legp[i], legq[i]))
        if s in CONTRA_SAME_ROW: cls[i] = 0
        elif s in FRONT_HIND: cls[i] = 1
        elif s in TRIPOD_DIAGONAL: cls[i] = 2
    return pre, post, syn, nt, cls


def build_matrices(scalars):
    """Return (We, Wi, stats) with per-row 0.020 uS renormalisation (proposal V1)."""
    pre, post, syn, nt, cls = edge_classes()
    s = np.array([scalars['contra_same_row'], scalars['front_hind'], scalars['tripod_diagonal'], 1.], float)
    if not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError('scalars must be finite and positive')
    raw = syn * s[cls]
    incoming = np.bincount(post, weights=raw, minlength=20000)
    rows = incoming[post]
    if (rows <= 0).any():
        raise ValueError('row without incoming edges')
    weight = TOTAL_US * raw / rows
    exc = nt == 'acetylcholine'
    We = sparse.csr_matrix((weight[exc], (post[exc], pre[exc])), shape=(20000, 20000))
    Wi = sparse.csr_matrix((weight[~exc], (post[~exc], pre[~exc])), shape=(20000, 20000))
    row_sums = np.asarray((We + Wi).sum(axis=1)).ravel()
    nonzero_rows = row_sums > 0
    stats = dict(class_edge_counts={k: int((cls == i).sum()) for i, k in enumerate(('contra_same_row', 'front_hind', 'tripod_diagonal', 'other'))},
                 class_conductance_uS={k: float(weight[cls == i].sum()) for i, k in enumerate(('contra_same_row', 'front_hind', 'tripod_diagonal', 'other'))},
                 total_conductance_uS=float(weight.sum()),
                 row_sum_max_abs_deviation_uS=float(np.max(np.abs(row_sums[nonzero_rows] - TOTAL_US))),
                 rows_with_incoming=int(nonzero_rows.sum()))
    return We, Wi, stats


class BlockLocalCircuit(frozen.LocalCircuit):
    """Frozen circuit with re-weighted existing edges only."""

    def __init__(self, scalars=None, **kwargs):
        self.scalars = dict(scalars or dict(contra_same_row=1., front_hind=1., tripod_diagonal=1.))
        super().__init__(**kwargs)

    def reset(self):
        self.net = frozen.AdaptationConductanceNetwork(20000, dt_ms=self.dt_ms)
        self.net.set_connectivity(self._We, self._Wi)
        self.filtered_rates_hz = np.zeros(20000, dtype=np.float64)
        return self


def apply_scalars(circuit, scalars):
    """Replace the circuit matrices in place after construction, then reset."""
    We, Wi, stats = build_matrices(scalars)
    circuit._We, circuit._Wi = We, Wi
    circuit.scalars = dict(scalars)
    circuit.reset()
    return stats
