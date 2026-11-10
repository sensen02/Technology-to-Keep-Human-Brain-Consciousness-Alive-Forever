"""Half-centre hypothesis: reciprocally re-weight EXISTING A<->B tripod-group edges.

Only four existing edge classes change (A->B inhibitory, B->A inhibitory, A->B
excitatory, B->A excitatory) with per-row renormalisation to the frozen 0.020 uS.
No edge is created or removed. Uncalibrated engineering weights, not measurements.
"""
from pathlib import Path
import numpy as np
from scipy import sparse

import module_blocks as mb

POOLS = mb.POOLS
SOURCE = mb.SOURCE
LEG_BUCKET = {0: 'lf', 1: 'lh', 2: 'lm', 3: 'rf', 4: 'rh', 5: 'rm'}
GROUP_A = {'lf', 'rm', 'lh'}
GROUP_B = {'rf', 'lm', 'rh'}
TOTAL_US = mb.TOTAL_US


def neuron_group():
    """Per-neuron group tag: 'A', 'B' or '' for untagged."""
    with np.load(POOLS, allow_pickle=False) as z:
        bucket = z['bucket_id'].astype(np.int64)
        mn_local = z['mn_local'].astype(np.int64)
        leg_of_mn = z['leg_of_mn'].astype(object)
    group = np.array([''] * bucket.size, dtype=object)
    for b, leg in LEG_BUCKET.items():
        mask = bucket == b
        group[mask] = 'A' if leg in GROUP_A else 'B'
    legs = np.array([str(x) for x in leg_of_mn], dtype=object)
    for neuron, leg in zip(mn_local, legs):
        group[int(neuron)] = 'A' if leg in GROUP_A else 'B'
    return group


def class_ids():
    """0 AB_inh, 1 AB_exc, 2 BA_inh, 3 BA_exc, 4 other."""
    with np.load(SOURCE / 'LOCAL_MODULE_CANDIDATE.npz', allow_pickle=False) as z:
        pre, post, syn, nt = (z[k] for k in ('pre_local', 'post_local', 'synapse_counts', 'source_verified_nt'))
        eligible = np.isin(nt, ['acetylcholine', 'gaba', 'glutamate'])
    pre, post, syn, nt = pre[eligible], post[eligible], syn[eligible].astype(np.float64), nt[eligible]
    group = neuron_group()
    gp, gq = group[pre], group[post]
    exc = nt == 'acetylcholine'
    cls = np.full(pre.size, 4, dtype=np.int64)
    ab = (gp == 'A') & (gq == 'B')
    ba = (gp == 'B') & (gq == 'A')
    cls[ab & ~exc] = 0
    cls[ab & exc] = 1
    cls[ba & ~exc] = 2
    cls[ba & exc] = 3
    return pre, post, syn, nt, cls


def build(scalars):
    """Row-renormalised matrices with only the four A<->B classes re-weighted."""
    pre, post, syn, nt, cls = class_ids()
    s = np.array([scalars['ab_inh'], scalars['ab_exc'], scalars['ba_inh'], scalars['ba_exc'], 1.], float)
    if not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError('scalars must be finite and positive')
    raw = syn * s[cls]
    incoming = np.bincount(post, weights=raw, minlength=20000)
    if (incoming[post] <= 0).any():
        raise ValueError('row without incoming edges')
    weight = TOTAL_US * raw / incoming[post]
    exc = nt == 'acetylcholine'
    We = sparse.csr_matrix((weight[exc], (post[exc], pre[exc])), shape=(20000, 20000))
    Wi = sparse.csr_matrix((weight[~exc], (post[~exc], pre[~exc])), shape=(20000, 20000))
    names = ('cross_AB_inh', 'cross_AB_exc', 'cross_BA_inh', 'cross_BA_exc', 'other')
    details = dict(
        class_edge_counts={n: int((cls == i).sum()) for i, n in enumerate(names)},
        class_conductance_uS={n: float(weight[cls == i].sum()) for i, n in enumerate(names)},
        row_sums=(np.asarray((We + Wi).sum(axis=1)).ravel()))
    return We, Wi, details
