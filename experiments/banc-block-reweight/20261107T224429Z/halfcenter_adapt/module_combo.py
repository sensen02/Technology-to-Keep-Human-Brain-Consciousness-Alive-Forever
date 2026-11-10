"""Half-centre plus premotor-adaptation hypothesis: existing edges re-weighted,
adaptation increment scaled on the six leg-specific premotor pools only.

Motor neurons keep the frozen adaptation (they are strict graph sinks, so scaling
their adaptation cannot change inter-leg coupling). Uncalibrated engineering
weights and gains, not measurements.
"""
from pathlib import Path
import numpy as np

import importlib.util
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mb = _load('_ca_module_blocks', _ROOT / 'module_blocks.py')
hc = _load('_ca_module_halfcenter', _ROOT / 'halfcenter' / 'module_halfcenter.py')

POOLS = mb.POOLS
POOL_BUCKETS = range(0, 6)


def pool_neurons():
    with np.load(POOLS, allow_pickle=False) as z:
        bucket = z['bucket_id'].astype(np.int64)
    mask = np.zeros(bucket.size, dtype=bool)
    for b in POOL_BUCKETS:
        mask |= bucket == b
    return mask


def build(scalars):
    """Return (We, Wi, increment_scale) for the combined hypothesis."""
    pre, post, syn, nt, cls = hc.class_ids()
    s = np.array([scalars['ab_inh'], scalars['ab_exc'], scalars['ba_inh'], scalars['ba_exc'], 1.], float)
    if not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError('scalars must be finite and positive')
    raw = syn * s[cls]
    incoming = np.bincount(post, weights=raw, minlength=20000)
    if (incoming[post] <= 0).any():
        raise ValueError('row without incoming edges')
    weight = .02 * raw / incoming[post]
    exc = nt == 'acetylcholine'
    from scipy import sparse
    We = sparse.csr_matrix((weight[exc], (post[exc], pre[exc])), shape=(20000, 20000))
    Wi = sparse.csr_matrix((weight[~exc], (post[~exc], pre[~exc])), shape=(20000, 20000))
    scale = np.ones(20000)
    scale[pool_neurons()] = float(scalars['pool_adapt'])
    if not np.isfinite(scale).all() or (scale < 1.).any():
        raise ValueError('adaptation scale must be finite and >= 1')
    names = ('cross_AB_inh', 'cross_AB_exc', 'cross_BA_inh', 'cross_BA_exc', 'other')
    details = dict(class_edge_counts={n: int((cls == i).sum()) for i, n in enumerate(names)},
                   class_conductance_uS={n: float(weight[cls == i].sum()) for i, n in enumerate(names)},
                   pool_neurons=int(scale.size - np.count_nonzero(scale == 1.)))
    return We, Wi, scale, details
