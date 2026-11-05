"""Build the three networks that Route B compares, and prove what each one preserves.

Route B asks a question the project has never answered cleanly: does the SPECIFIC wiring of this
sampled BANC graph contribute anything to the task, or would any recurrent network of the same size
and activity do as well? External evidence points at "no" - the one game project that ran a
parameter-matched control (Fly Dino) scored identically with and without the connectome - and this
project's own history points the same way, since its biggest single gain came from a learned
replacement layer rather than from the connectome.

The three arms differ ONLY in the recurrent matrices We and Wi:

  A  real        the archived rebuilt BANC weights, unchanged
  B  rewired     the same neuron count, the same edge count, the exact same in degree and out degree
                 for every one of the 12,000 cells, and the exact same weight multiset, but the
                 pairs are re-paired at random. This keeps everything about the graph except which
                 neuron talks to which.
  C  uniform     the same neuron count, the same edge count and the same weight multiset, but the
                 pairs are placed uniformly at random with no degree control at all.

B is the control that matters. If A and B score the same, the identity of the connections is
irrelevant and only the degree sequence and weight distribution are doing work. If A beats B, the
specific wiring carries information.

Everything downstream of the matrices is untouched: the sensory encoding, the pools, the per cell
readout, the gains, the replacement layer and the body are all identical across arms, and so is the
timestep, which this project has now shown matters at the 0.03 level.

Usage: build arms into <OUT>/ARMS/{A,B,C}/ each holding ACTUAL_We.npz and ACTUAL_Wi.npz, plus
ARM_VERIFICATION.json. Then point a trainer's CONNECTIVITY_SET at one arm.
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'tools'))
REBUILD = Path('/home/sensen/Desktop/engine修复验收/s5-bristle-edge-rebuild/20261104T050000Z')
O = Path(os.environ.get('AB_OUT', '/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z'))
SEED = int(os.environ.get('AB_SEED', '20261108'))
SWEEPS = int(os.environ.get('AB_SWEEPS', '12'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False,
                                  default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def coo_of(m):
    coo = m.tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float64)


def maslov_sneppen(pre, post, n, rng, sweeps=12, log=None):
    """Randomise the pairing while preserving every node's in degree and out degree exactly.

    Each move picks two edges (a->b) and (c->d) and proposes (a->d) and (c->b). Every node keeps its
    number of edges, because each edge still has one head and one tail and no tail or head is added
    or removed. The move is taken only when neither new pair already exists, so the graph never
    grows a duplicate and the edge count is constant.

    A fixed number of proposals is made instead of looping until no duplicate remains, because the
    number of proposals is what a reader can check: sweeps * edge_count, and the function always
    returns. The weight multiset is then permuted across the slots, so the weights are the same
    values with no relation to the new positions.
    """
    size = pre.size
    key = pre * n + post
    occupied = set(key.tolist())
    proposals = int(sweeps * size)
    accepted = 0
    rejected_duplicate = 0
    rejected_degenerate = 0
    i = rng.integers(0, size, size=proposals)
    j = rng.integers(0, size, size=proposals)
    for k in range(proposals):
        a_idx, c_idx = int(i[k]), int(j[k])
        if a_idx == c_idx:
            rejected_degenerate += 1
            continue
        a, b = int(pre[a_idx]), int(post[a_idx])
        c, d = int(pre[c_idx]), int(post[c_idx])
        if a == c or b == d:
            rejected_degenerate += 1
            continue
        k1, k2 = a * n + d, c * n + b
        kold1, kold2 = a * n + b, c * n + d
        if k1 in occupied or k2 in occupied:
            rejected_duplicate += 1
            continue
        occupied.discard(kold1); occupied.discard(kold2)
        occupied.add(k1); occupied.add(k2)
        post[a_idx], post[c_idx] = d, b
        key[a_idx], key[c_idx] = k1, k2
        accepted += 1
    if log is not None:
        log('    accepted %d of %d proposals (%.1f%%), rejected %d duplicates and %d degenerate'
            % (accepted, proposals, 100.0 * accepted / proposals, rejected_duplicate, rejected_degenerate))
    assert len(occupied) == size, 'a duplicate pair appeared'
    return pre, post, accepted


def degree_preserving(n, row, col, data, rng):
    """Same in/out degree per node and same weight multiset, pairing randomised."""
    pre = col.astype(np.int64).copy()
    post = row.astype(np.int64).copy()
    print('  degree_preserving: %d edges, randomising the pairing...' % pre.size, flush=True)
    pre, post, accepted = maslov_sneppen(pre, post, n, rng, sweeps=SWEEPS,
                                         log=lambda m: print(m, flush=True))
    w = data[rng.permutation(data.size)]      # exact same multiset, uncorrelated with position
    # The engine's convention is csr((weight, (row, col))) with row = postsynaptic and col =
    # presynaptic, so the pair is returned as (post, pre) in that order. Returning (pre, post)
    # transposes the graph, which silently swaps the in degree and out degree sequences - the first
    # build did exactly that and produced a control whose degrees were mirrored rather than matched.
    return post, pre, w, accepted


def uniform(n, count, data, rng):
    """Same edge count and same weight multiset, pairs drawn uniformly with no degree control."""
    total = n * n
    chosen = np.empty(0, dtype=np.int64)
    while chosen.size < count:
        want = max(2048, int((count - chosen.size) * 1.06) + 8)
        chosen = np.unique(np.concatenate([chosen, rng.integers(0, total, size=want)]))
    chosen = chosen[:count]                     # still strictly increasing, hence distinct
    pre = chosen // n
    post = chosen % n
    w = data[rng.permutation(data.size)]        # exact same multiset, uncorrelated with position
    assert w.size == count
    return post, pre, w, 0                       # (row, col) order, see the note in degree_preserving


def stats(m, n):
    coo = m.tocoo()
    indeg = np.bincount(coo.row, minlength=n)
    outdeg = np.bincount(coo.col, minlength=n)
    return dict(nnz=int(m.nnz), weight_sum=float(coo.data.sum()),
                weight_min=float(coo.data.min()) if m.nnz else None,
                weight_max=float(coo.data.max()) if m.nnz else None,
                indeg_mean=float(indeg.mean()), indeg_max=int(indeg.max()),
                indeg_zero=int((indeg == 0).sum()),
                outdeg_mean=float(outdeg.mean()), outdeg_max=int(outdeg.max()),
                outdeg_zero=int((outdeg == 0).sum()),
                weight_sha=hashlib.sha256(np.sort(coo.data).tobytes()).hexdigest())


def main():
    rng = np.random.default_rng(SEED)
    we = sparse.load_npz(REBUILD / 'REBUILT_We.npz').tocsr()
    wi = sparse.load_npz(REBUILD / 'REBUILT_Wi.npz').tocsr()
    assert we.shape == wi.shape and we.shape[0] == we.shape[1]
    n = we.shape[0]
    arms = {'A': {}, 'B': {}, 'C': {}}
    for name, mat, kind in (('We', we, 'e'), ('Wi', wi, 'i')):
        row, col, data = coo_of(mat)
        arms['A'][name] = (row, col, data, 0)
        assert (data > 0).all(), name + ' must stay a positive weight matrix'
        if kind == 'e' or kind == 'i':
            arms['B'][name] = degree_preserving(n, row, col, data, rng)
            arms['C'][name] = uniform(n, data.size, data, rng)
    report = dict(
        package='s5-route-b-ablation',
        source=str(REBUILD),
        source_sha256={'We': sha(REBUILD / 'REBUILT_We.npz'), 'Wi': sha(REBUILD / 'REBUILT_Wi.npz')},
        n=n, seed=SEED,
        arms={}, what_each_arm_preserves={
            'A': 'nothing is changed; this is the graph the project has always used',
            'B': 'neuron count, edge count, per node in degree, per node out degree, exact weight multiset; the pairing is random',
            'C': 'neuron count, edge count, exact weight multiset; the degrees are whatever random placement gives'})
    for arm in ('A', 'B', 'C'):
        d = O / 'ARMS' / arm
        d.mkdir(parents=True, exist_ok=True)
        entry = {}
        for name in ('We', 'Wi'):
            row, col, data, iters = arms[arm][name]
            m = sparse.csr_matrix((data, (row, col)), shape=(n, n))
            sparse.save_npz(d / ('ACTUAL_' + name + '.npz'), m)
            entry[name] = stats(m, n)
            entry[name]['swaps_accepted'] = int(iters)
        report['arms'][arm] = entry
    # Full degree sequences, not just their summaries: arm B must match A for every one of the
    # 12,000 cells on both in and out degree, which is the whole point of the control.
    full = {}
    for name in ('We', 'Wi'):
        seq = {}
        for arm in ('A', 'B', 'C'):
            m = sparse.load_npz(O / 'ARMS' / arm / ('ACTUAL_' + name + '.npz')).tocoo()
            seq[arm] = dict(indeg=np.bincount(m.row, minlength=n), outdeg=np.bincount(m.col, minlength=n))
        for arm in ('B', 'C'):
            full['%s_%s' % (arm, name)] = dict(
                indeg_vector_identical=bool(np.array_equal(seq['A']['indeg'], seq[arm]['indeg'])),
                outdeg_vector_identical=bool(np.array_equal(seq['A']['outdeg'], seq[arm]['outdeg'])),
                indeg_differing_cells=int((seq['A']['indeg'] != seq[arm]['indeg']).sum()),
                outdeg_differing_cells=int((seq['A']['outdeg'] != seq[arm]['outdeg']).sum()))
    report['full_degree_sequences'] = full
    # the comparisons that decide whether the controls are honest
    checks = {}
    for name in ('We', 'Wi'):
        a = report['arms']['A'][name]
        for arm in ('B', 'C'):
            b = report['arms'][arm][name]
            checks['%s_%s' % (arm, name)] = dict(
                nnz_equal=bool(a['nnz'] == b['nnz']),
                weight_multiset_equal=bool(a['weight_sha'] == b['weight_sha']),
                weight_sum_equal=bool(abs(a['weight_sum'] - b['weight_sum']) < 1e-9 * max(1.0, abs(a['weight_sum']))),
                indeg_sequence_equal=bool(a['indeg_mean'] == b['indeg_mean'] and a['indeg_max'] == b['indeg_max']
                                          and a['indeg_zero'] == b['indeg_zero']),
                outdeg_sequence_equal=bool(a['outdeg_mean'] == b['outdeg_mean'] and a['outdeg_max'] == b['outdeg_max']
                                           and a['outdeg_zero'] == b['outdeg_zero']))
    report['control_checks'] = checks
    report['honest_notes'] = [
        'the degree checks compare summary statistics, not the full degree vectors; the full sequences are preserved by construction for B and are recorded in ARM_VERIFICATION.json only as summaries',
        'arm C is expected to change activity drastically; that is what "no structure" means, and it is why B, not C, is the informative control',
        're-pairing can create self loops; they are permitted in every arm, including A, so they are not a confound',
        'these matrices are the archived rebuilt weights, which this project verified are the built weights times 0.2; the arms are built from the same source so the scaling is common to all three']
    dump(O / 'ARM_VERIFICATION.json', report)
    print(json.dumps({a: {k: {kk: vv for kk, vv in v.items() if kk != 'weight_sha'} for k, v in report['arms'][a].items()}
                      for a in ('A', 'B', 'C')}, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
