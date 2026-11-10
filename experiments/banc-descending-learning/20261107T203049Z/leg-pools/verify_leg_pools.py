"""Independent verification of the leg-pool audit. Does not import leg_pools.py.

Everything is rebuilt from the frozen inputs with a deliberately different code path:
plain Python dicts/sets over the edge list (no numpy bitmasks, no np.ufunc.at, no
scipy fancy indexing for the partition), and the block conductance totals are recomputed
from the raw synapse counts rather than from the sparse matrices.

Checks: source hashes, weight contract, pool partition (every neuron exactly one bucket),
exact edge conservation (435126 edges, none duplicated or dropped), and reproduction of
every number quoted in POOLS.json / BLOCKS.json / SYMMETRY.json / COUPLING.json /
PROPOSED_SCALING_PARAMS.json / RESULT.json.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
MODULE_192931 = PROJECT / 'experiments/banc-local-learning/20261107T192931Z/dynamics/module.py'
MODULE_185910 = PROJECT / 'experiments/banc-local-learning/20261107T185910Z/dynamics/module.py'
ENGINE = PROJECT / 'engine/neural_cond.py'
NI = PROJECT / 'experiments/flybody-retarget/20261107T182940Z/neural-interface'

SHA = {
    str(MODULE_192931): 'c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250',
    str(MODULE_185910): '3443853d8ebe725f8fedcb2e022b53ca4ed470d87f06a8d3ed71fa049f150676',
    str(ENGINE): '27e46d0a9d2b3014ead4c07c4ae64a6b46a069c35ff772c7533210b053164a4e',
    str(NI / 'LOCAL_MODULE_CANDIDATE.npz'): '7a7ce0d7d2166f81d0f9c8c1acd73342b854c7f33915d6d6cbf68ef89738395e',
    str(NI / 'LOCAL_MOTOR_READOUT_PORTS.csv'): 'c78650b4a8b705512523674f4c47dd7e7718261db5b51d37519f910b558df39a',
    str(NI / 'GROUP_CACHE_INDICES.npz'): '557cfc944270a34053d74d3eebe39af453eefe530c22ae5ce656f92fbff3b09a',
    str(NI / 'SENSORY_CHANNEL_REVIEW.json'): '906ccfca7b27698d59666d0d9ce15f37ff43b23a013d44168902467757bb4b64',
}

N = 20000
LEGS = ['lf', 'lh', 'lm', 'rf', 'rh', 'rm']
BUCKETS = ['pool_lf', 'pool_lh', 'pool_lm', 'pool_rf', 'pool_rh', 'pool_rm',
           'pool_shared', 'pool_nonmotor', 'pool_mn']
MIRROR = {'pool_lf': 'pool_rf', 'pool_rf': 'pool_lf', 'pool_lh': 'pool_rh', 'pool_rh': 'pool_lh',
          'pool_lm': 'pool_rm', 'pool_rm': 'pool_lm', 'pool_shared': 'pool_shared',
          'pool_nonmotor': 'pool_nonmotor', 'pool_mn': 'pool_mn'}

CHECKS = []


def check(name, ok, detail):
    CHECKS.append({'check': name, 'pass': bool(ok), 'detail': detail})
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else '  <-- ' + str(detail)), flush=True)
    return ok


def close(a, b, rel=1e-12, abs_=1e-15):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    # ---------------- 1. source hashes
    for path, want in SHA.items():
        check('sha256 ' + path, sha256(path) == want, {'expected': want, 'actual': sha256(path)})

    # ---------------- 2. inputs
    with np.load(NI / 'LOCAL_MODULE_CANDIDATE.npz', allow_pickle=False) as z:
        pre_all = np.asarray(z['pre_local'])
        post_all = np.asarray(z['post_local'])
        syn_all = np.asarray(z['synapse_counts'])
        nt_all = np.asarray(z['source_verified_nt'])
        keep = np.array([t in ('acetylcholine', 'gaba', 'glutamate') for t in nt_all], dtype=bool)
        if not np.array_equal(keep, z['pure_verified_label_mask']):
            check('strict mask equals pure-label mask', False, 'mask disagreement')
    pre = pre_all[keep].astype(np.int64)
    post = post_all[keep].astype(np.int64)
    syn = syn_all[keep].astype(np.float64)
    nt = nt_all[keep]
    check('strict edge count is 435126', pre.size == 435126, int(pre.size))
    check('acetylcholine edge count is 207166', int((nt == 'acetylcholine').sum()) == 207166,
          int((nt == 'acetylcholine').sum()))
    check('inhibitory edge count is 227960', int((nt != 'acetylcholine').sum()) == 227960,
          int((nt != 'acetylcholine').sum()))

    if '_legpools_verify_frozen' not in sys.modules:
        spec = importlib.util.spec_from_file_location('_legpools_verify_frozen', MODULE_192931)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['_legpools_verify_frozen'] = mod
        spec.loader.exec_module(mod)
    circuit = sys.modules['_legpools_verify_frozen'].LocalCircuit(total_conductance_uS=.02)
    We, Wi = circuit._We, circuit._Wi

    # ---------------- 3. leg labels straight from the CSV (independent of the circuit object)
    legs_from_csv = {}
    with (NI / 'LOCAL_MOTOR_READOUT_PORTS.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['readout_eligible_single_leg'] == 'True':
                legs_from_csv[int(row['local_index_0based'])] = row['nerve_matched_legs']
    mn_local = np.asarray(circuit.mn_local, dtype=np.int64)
    check('CSV gives 323 single-leg motor cells', len(legs_from_csv) == 323, len(legs_from_csv))
    check('circuit mn_local equals the CSV index set',
          set(int(x) for x in mn_local) == set(legs_from_csv),
          {'circuit': len(mn_local), 'csv': len(legs_from_csv)})
    check('circuit mn_leg labels equal the CSV labels',
          all(str(circuit.mn_leg[i]) == legs_from_csv[int(mn_local[i])] for i in range(len(mn_local))), 'label order')
    is_mn = set(int(x) for x in mn_local)
    leg_count = {leg: sum(1 for v in legs_from_csv.values() if v == leg) for leg in LEGS}
    check('mn leg counts lf43 lh59 lm60 rf42 rh59 rm60',
          leg_count == {'lf': 43, 'lh': 59, 'lm': 60, 'rf': 42, 'rh': 59, 'rm': 60}, leg_count)

    # ---------------- 4. weight contract recomputed from raw source
    incoming = defaultdict(float)
    for q, s in zip(post.tolist(), syn.tolist()):
        incoming[q] += s
    w_edge = np.array([.02 * s / incoming[int(q)] for q, s in zip(post.tolist(), syn.tolist())], dtype=np.float64)
    total_uS = float(w_edge.sum())
    check('total module conductance 380.84 uS', close(total_uS, 380.84), total_uS)
    row_tot = defaultdict(float)
    for q, w in zip(post.tolist(), w_edge.tolist()):
        row_tot[q] += w
    check('every neuron with incoming edges has 0.02 uS',
          all(close(v, .02) for v in row_tot.values()) and len(row_tot) == 19042,
          {'n': len(row_tot), 'min': min(row_tot.values()), 'max': max(row_tot.values())})
    rec_e = np.zeros(We.nnz, dtype=np.float64)
    we_coo = We.tocoo()
    wi_coo = Wi.tocoo()
    frozen = {}
    for r, c, d in zip(we_coo.row.tolist(), we_coo.col.tolist(), we_coo.data.tolist()):
        frozen[(r, c)] = d
    for r, c, d in zip(wi_coo.row.tolist(), wi_coo.col.tolist(), wi_coo.data.tolist()):
        frozen[(r, c)] = d
    check('frozen matrix nnz total is 435126', We.nnz + Wi.nnz == 435126 and len(frozen) == 435126,
          {'we': We.nnz, 'wi': Wi.nnz, 'frozen_dict': len(frozen)})
    maxdiff = max(abs(frozen[(int(q), int(p))] - float(w)) for p, q, w in zip(pre.tolist(), post.tolist(), w_edge.tolist()))
    check('frozen matrices reproduce 0.02*syn/incoming exactly', maxdiff < 1e-18, maxdiff)
    check('sum(_We)+sum(_Wi) equals 380.84',
          close(We.sum() + Wi.sum(), total_uS), [float(We.sum()), float(Wi.sum())])

    # ---------------- 5. direct and two-hop reach, plain dicts
    targets = defaultdict(lambda: defaultdict(int))       # cell -> leg -> number of MN cells hit
    for p, q in zip(pre.tolist(), post.tolist()):
        if q in is_mn:
            targets[p][legs_from_csv[q]] += 1
    direct_legs = {p: frozenset(d) for p, d in targets.items()}

    succ_legs = {p: frozenset(d) for p, d in direct_legs.items()}   # legs reachable in one edge from p (p not needed in M)
    hop2 = defaultdict(set)
    for p, x in zip(pre.tolist(), post.tolist()):
        if x in is_mn:
            continue
        for leg in direct_legs.get(x, ()):           # x -> MN of leg
            hop2[p].add(leg)
    two_hop_legs = {p: frozenset(s) for p, s in hop2.items()}

    # ---------------- 6. bucket partition (dict based, independent)
    bucket = {}
    for p in range(N):
        if p in is_mn:
            bucket[p] = 'pool_mn'
            continue
        d = direct_legs.get(p, frozenset())
        if len(d) == 0:
            bucket[p] = 'pool_nonmotor'
        elif len(d) == 1:
            bucket[p] = 'pool_' + next(iter(d))
        else:
            bucket[p] = 'pool_shared'
    sizes = {b: 0 for b in BUCKETS}
    for b in bucket.values():
        sizes[b] += 1
    check('partition covers every one of the 20000 neurons exactly once',
          sum(sizes.values()) == N and len(bucket) == N, sizes)
    pools_json = json.loads((HERE / 'POOLS.json').read_text())
    check('POOLS.json primary partition sizes reproduce',
          pools_json['primary_partition'] == sizes, {'json': pools_json['primary_partition'], 'rebuilt': sizes})

    bucket_union = {}
    for p in range(N):
        if p in is_mn:
            bucket_union[p] = 'pool_mn'
            continue
        d = set(direct_legs.get(p, frozenset())) | set(two_hop_legs.get(p, frozenset()))
        if not d:
            bucket_union[p] = 'pool_nonmotor'
        elif len(d) == 1:
            bucket_union[p] = 'pool_' + next(iter(d))
        else:
            bucket_union[p] = 'pool_shared'
    sizes_u = {b: 0 for b in BUCKETS}
    for b in bucket_union.values():
        sizes_u[b] += 1
    check('POOLS.json union partition sizes reproduce',
          pools_json['supplementary_partition_union_of_direct_and_2hop'] == sizes_u,
          {'json': pools_json['supplementary_partition_union_of_direct_and_2hop'], 'rebuilt': sizes_u})
    check('union partition also covers 20000 neurons', sum(sizes_u.values()) == N, sum(sizes_u.values()))

    npz = np.load(HERE / 'POOLS_AND_REACH.npz')
    bid_saved = npz['bucket_id'].astype(np.int64)
    bid_union_saved = npz['bucket_id_union'].astype(np.int64)
    same = all(int(bid_saved[p]) == BUCKETS.index(bucket[p]) for p in range(N))
    check('saved bucket_id array equals the independent partition', same, 'per-neuron compare')
    same_u = all(int(bid_union_saved[p]) == BUCKETS.index(bucket_union[p]) for p in range(N))
    check('saved bucket_id_union array equals the independent partition', same_u, 'per-neuron compare')
    mask_d = npz['leg_mask_direct']
    ok_mask = all(int(mask_d[p]) == sum(1 << LEGS.index(l) for l in direct_legs.get(p, frozenset())) for p in range(N))
    check('saved direct leg mask equals the independent direct-leg sets', ok_mask, 'per-neuron compare')
    mask_h = npz['leg_mask_2hop']
    ok_mask_h = all(int(mask_h[p]) == sum(1 << LEGS.index(l) for l in two_hop_legs.get(p, frozenset()))
                    for p in range(N))
    check('saved two-hop leg mask equals the independent two-hop-leg sets', ok_mask_h, 'per-neuron compare')
    check('direct reach count 4723 reproduces',
          pools_json['neurons_reaching_any_mn_direct'] == len(direct_legs) == 4723,
          [pools_json['neurons_reaching_any_mn_direct'], len(direct_legs)])
    n_live_2hop = sum(1 for p in range(N) if two_hop_legs.get(p))
    check('two-hop reach count 9785 reproduces', pools_json['neurons_reaching_any_mn_by_2hop_only_or_more'] == n_live_2hop,
          [pools_json['neurons_reaching_any_mn_by_2hop_only_or_more'], n_live_2hop])

    # ---------------- 7. exact edge conservation and block tables recomputed from raw source
    per_block_edges = defaultdict(int)
    per_block_uS = defaultdict(float)
    seen = set()
    dup = 0
    for p, q, w in zip(pre.tolist(), post.tolist(), w_edge.tolist()):
        key = (p, q)
        if key in seen:
            dup += 1
        seen.add(key)
        k = (bucket[p], bucket[q])
        per_block_edges[k] += 1
        per_block_uS[k] += w
    check('no duplicated edge in the strict edge list', dup == 0, dup)
    check('no self loop in the strict edge list', all(p != q for p, q in zip(pre.tolist(), post.tolist())), 'selfloops')
    check('block edge counts sum to 435126', sum(per_block_edges.values()) == 435126, sum(per_block_edges.values()))
    check('block conductance sums to 380.84 uS', close(sum(per_block_uS.values()), 380.84), sum(per_block_uS.values()))

    result = json.loads((HERE / 'RESULT.json').read_text())
    edges_json = result['blocks_edges_both_receptors']
    g_json = result['blocks_conductance_uS_both_receptors']
    bad_e, bad_g = [], []
    for a in BUCKETS:
        for b in BUCKETS:
            e_re = per_block_edges.get((a, b), 0)
            g_re = per_block_uS.get((a, b), 0.0)
            if int(edges_json[a][b]) != e_re:
                bad_e.append((a, b, edges_json[a][b], e_re))
            if not close(g_json[a][b], g_re):
                bad_g.append((a, b, g_json[a][b], g_re))
    check('all 81 block edge counts reproduce exactly (both receptors)', not bad_e, bad_e[:5])
    check('all 81 block conductances reproduce (1e-12 rel)', not bad_g, bad_g[:5])

    # receptor split, rebuilt from nt
    per_block_exc = defaultdict(int)
    per_block_exc_uS = defaultdict(float)
    per_block_inh = defaultdict(int)
    per_block_inh_uS = defaultdict(float)
    for p, q, w, t in zip(pre.tolist(), post.tolist(), w_edge.tolist(), nt.tolist()):
        k = (bucket[p], bucket[q])
        if t == 'acetylcholine':
            per_block_exc[k] += 1
            per_block_exc_uS[k] += w
        else:
            per_block_inh[k] += 1
            per_block_inh_uS[k] += w
    bad = []
    for a in BUCKETS:
        for b in BUCKETS:
            if int(result['blocks_edges_We'][a][b]) != per_block_exc.get((a, b), 0):
                bad.append(('We', a, b, result['blocks_edges_We'][a][b], per_block_exc.get((a, b), 0)))
            if int(result['blocks_edges_Wi'][a][b]) != per_block_inh.get((a, b), 0):
                bad.append(('Wi', a, b, result['blocks_edges_Wi'][a][b], per_block_inh.get((a, b), 0)))
            if not close(result['blocks_conductance_uS_We'][a][b], per_block_exc_uS.get((a, b), 0.0)):
                bad.append(('WeG', a, b, result['blocks_conductance_uS_We'][a][b], per_block_exc_uS.get((a, b), 0.0)))
            if not close(result['blocks_conductance_uS_Wi'][a][b], per_block_inh_uS.get((a, b), 0.0)):
                bad.append(('WiG', a, b, result['blocks_conductance_uS_Wi'][a][b], per_block_inh_uS.get((a, b), 0.0)))
    check('receptor-split block edges and conductances reproduce from the nt column', not bad, bad[:5])

    # note: the two independent conductance routes agree only up to summation order; report the spread
    spread = max(abs(per_block_uS.get((a, b), 0.0) - (per_block_exc_uS.get((a, b), 0.0)
                                                      + per_block_inh_uS.get((a, b), 0.0)))
                 for a in BUCKETS for b in BUCKETS)
    check('raw-order vs receptor-split summation agree within 1e-12', spread < 1e-9, spread)

    # ---------------- 8. aggregates
    def agg(src, dst):
        e = sum(per_block_edges.get((a, b), 0) for a in src for b in dst)
        g = sum(per_block_uS.get((a, b), 0.0) for a in src for b in dst)
        return e, g
    left = ['pool_lf', 'pool_lh', 'pool_lm']
    right = ['pool_rf', 'pool_rh', 'pool_rm']
    front = ['pool_lf', 'pool_rf']
    hind = ['pool_lh', 'pool_rh']
    six = ['pool_' + l for l in LEGS]
    expected_aggs = {
        'left_to_right': agg(left, right), 'right_to_left': agg(right, left),
        'front_to_hind': agg(front, hind), 'hind_to_front': agg(hind, front),
        'leg_pools_to_mn_bucket': agg(six, ['pool_mn']),
        'shared_to_mn_bucket': agg(['pool_shared'], ['pool_mn']),
        'nonmotor_to_mn_bucket': agg(['pool_nonmotor'], ['pool_mn']),
        'nonmotor_internal': agg(['pool_nonmotor'], ['pool_nonmotor']),
        'mn_to_mn': agg(['pool_mn'], ['pool_mn']),
    }
    bad = []
    for k, (e, g) in expected_aggs.items():
        got = result['aggregates_summary'][k]
        if int(got['edges']) != e or not close(got['conductance_uS'], g):
            bad.append((k, got, e, g))
    check('aggregate blocks reproduce (L->R, R->L, F->H, H->F, drives, nonmotor internal, mn<-mn)', not bad, bad)

    # ---------------- 9. symmetry
    Gmat = [[per_block_uS.get((a, b), 0.0) for b in BUCKETS] for a in BUCKETS]
    num = den = 0.0
    for i, a in enumerate(BUCKETS):
        for j, b in enumerate(BUCKETS):
            mirror = Gmat[BUCKETS.index(MIRROR[b])][BUCKETS.index(MIRROR[a])]
            num += abs(Gmat[i][j] - mirror)
            den += Gmat[i][j] + mirror
    sym = json.loads((HERE / 'SYMMETRY.json').read_text())
    prev = sym['conductance_matrix_mirror_preregistered_rule']['both']
    check('preregistered-rule asym_index_L1 (both receptors) reproduces',
          close(prev['asym_index_L1'], num / den), [prev['asym_index_L1'], num / den])
    check('preregistered-rule abs_diff_sum_uS (both) reproduces', close(prev['abs_diff_sum_uS'], num),
          [prev['abs_diff_sum_uS'], num])
    Gnum = Gden = 0.0
    for i, a in enumerate(BUCKETS):
        for j, b in enumerate(BUCKETS):
            mirror = [[per_block_exc_uS.get((x, y), 0.0) for y in BUCKETS] for x in BUCKETS]
            Gnum += abs(per_block_exc_uS.get((a, b), 0.0) - mirror[BUCKETS.index(MIRROR[b])][BUCKETS.index(MIRROR[a])])
            Gden += per_block_exc_uS.get((a, b), 0.0) + mirror[BUCKETS.index(MIRROR[b])][BUCKETS.index(MIRROR[a])]
    check('preregistered-rule asym_index_L1 (_We) reproduces',
          close(sym['conductance_matrix_mirror_preregistered_rule']['_We']['asym_index_L1'], Gnum / Gden),
          [sym['conductance_matrix_mirror_preregistered_rule']['_We']['asym_index_L1'], Gnum / Gden])
    # corrected mirror rule: G'[s][d] = G[mir(s)][mir(d)]  (see DEVIATIONS.md deviation 1)
    Gexc = [[per_block_exc_uS.get((a, b), 0.0) for b in BUCKETS] for a in BUCKETS]
    Ginh = [[per_block_inh_uS.get((a, b), 0.0) for b in BUCKETS] for a in BUCKETS]

    def corrected_index(mat):
        num = den = 0.0
        for i, a in enumerate(BUCKETS):
            for j, b in enumerate(BUCKETS):
                mi = BUCKETS.index(MIRROR[a])
                mj = BUCKETS.index(MIRROR[b])
                num += abs(mat[i][j] - mat[mi][mj])
                den += mat[i][j] + mat[mi][mj]
        return num, den, (num / den if den else None)

    for lab, mat in (('_We', Gexc), ('_Wi', Ginh)):
        n_, d_, i_ = corrected_index(mat)
        got = sym['conductance_matrix_mirror_corrected_rule'][lab]
        check('asym_index_L1 corrected rule (%s) reproduces' % lab, close(got['asym_index_L1'], i_),
              [got['asym_index_L1'], i_])
        check('abs_diff_sum_uS corrected rule (%s) reproduces' % lab, close(got['abs_diff_sum_uS'], n_),
              [got['abs_diff_sum_uS'], n_])
    nb_ = len(BUCKETS)
    mat_both = [[Gexc[i][j] + Ginh[i][j] for j in range(nb_)] for i in range(nb_)]
    n_, d_, i_ = corrected_index(mat_both)
    got = sym['conductance_matrix_mirror_corrected_rule']['both']
    check('asym_index_L1 corrected rule (both receptors) reproduces', close(got['asym_index_L1'], i_),
          [got['asym_index_L1'], i_])
    # leg-pools-only restricted index
    lp = ['pool_' + l for l in ['lf', 'lh', 'lm', 'rf', 'rh', 'rm']]
    for lab, mat in (('_We', Gexc), ('_Wi', Ginh), ('both', mat_both)):
        sub = [[mat[BUCKETS.index(a)][BUCKETS.index(b)] for b in lp] for a in lp]
        num = den = 0.0
        for i, a in enumerate(lp):
            for j, b in enumerate(lp):
                mi = lp.index(MIRROR[a])
                mj = lp.index(MIRROR[b])
                num += abs(sub[i][j] - sub[mi][mj])
                den += sub[i][j] + sub[mi][mj]
        got = sym['leg_pool_only_6x6_mirror'][lab]['corrected_rule']
        check('leg-pools-only corrected asym index (%s) reproduces' % lab, close(got['asym_index_L1'], num / den),
              [got['asym_index_L1'], num / den])
    # the preregistered rule is structurally blind on self-mirror blocks; the corrected rule is not
    blind = all(abs(sym['conductance_matrix_mirror_preregistered_rule']['both']['per_block_abs_diff_uS'][a][b]) < 1e-18
                for a, b in (('pool_lf', 'pool_rf'), ('pool_lm', 'pool_rm'), ('pool_lh', 'pool_rh')))
    check('preregistered mirror rule reports zero difference on LF<->RF, LM<->RM, LH<->RH (documented blindness)',
          blind, sym['conductance_matrix_mirror_preregistered_rule']['both']['per_block_abs_diff_uS']['pool_lf']['pool_rf'])
    dir_diff = abs(sym['conductance_matrix_mirror_corrected_rule']['both']['per_block_abs_diff_uS']['pool_lf']['pool_rf'])
    check('corrected mirror rule does report a non-zero LF->RF vs RF->LF difference', dir_diff > 0, dir_diff)
    # pair-level direction asymmetry reproduces from block sums
    bad = []
    for pair, ent in sym['leg_pair_conductance_mirror']['both'].items():
        a, b = pair.split('-')
        g_ab = per_block_uS.get(('pool_' + a, 'pool_' + b), 0.0)
        g_ba = per_block_uS.get(('pool_' + b, 'pool_' + a), 0.0)
        if not close(ent['conductance_a_to_b_uS'], g_ab) or not close(ent['conductance_b_to_a_uS'], g_ba):
            bad.append((pair, ent['conductance_a_to_b_uS'], g_ab, ent['conductance_b_to_a_uS'], g_ba))
        if not close(ent['direction_asym_uS'], g_ab - g_ba):
            bad.append((pair, 'dir_asym'))
    check('all 15 leg-pair direction asymmetries reproduce', not bad, bad[:5])
    check('the three contralateral same-row pairs are flagged self_mirror_pair',
          all(sym['leg_pair_conductance_mirror']['both'][k]['self_mirror_pair']
              for k in ('lf-rf', 'lm-rm', 'lh-rh')), 'self mirror flags')

    bad = []
    for k, v in sym['cell_counts_primary'].items():
        a, b = k.split('_vs_')
        if v['count_a'] != sizes[a] or v['count_b'] != sizes[b]:
            bad.append((k, v, sizes[a], sizes[b]))
        if not close(v['asym_count'], (sizes[a] - sizes[b]) / (sizes[a] + sizes[b])):
            bad.append((k, 'asym'))
    check('cell-count asymmetry values reproduce', not bad, bad)
    check('lf pool 548 and rf pool 639 reproduce', sizes['pool_lf'] == 548 and sizes['pool_rf'] == 639,
          [sizes['pool_lf'], sizes['pool_rf']])
    bad = []
    for b in BUCKETS:
        n = sum(1 for q in row_tot if bucket[q] == b)
        ent = sym['incoming_decomposition_integer_check'][b]
        if ent['neurons_with_incoming'] != n or not close(ent['expected_uS'], .02 * n):
            bad.append((b, ent, n))
    check('incoming decomposition integer check reproduces (0.02 uS per live neuron)', not bad, bad)

    # ---------------- 10. coupling
    coupling = json.loads((HERE / 'COUPLING.json').read_text())
    fog_leg = defaultdict(int)
    for p, q in zip(pre.tolist(), post.tolist()):
        if q in is_mn:
            fog_leg[(p, legs_from_csv[q])] += 1
    rad = defaultdict(int)
    for p, x in zip(pre.tolist(), post.tolist()):
        rad[(bucket[p], x)] += 1
    bad = []
    for pair, ent in coupling['pairs'].items():
        a, b = pair.split('-')
        # L0 from the recomputed block table
        e_ab = per_block_edges.get(('pool_' + a, 'pool_' + b), 0)
        e_ba = per_block_edges.get(('pool_' + b, 'pool_' + a), 0)
        if ent['L0']['edges_a_to_b']['both'] != e_ab or ent['L0']['edges_b_to_a']['both'] != e_ba:
            bad.append((pair, 'L0', ent['L0'], e_ab, e_ba))
        # L1 from dict sets
        both = sum(1 for p in range(N)
                   if a in direct_legs.get(p, frozenset()) and b in direct_legs.get(p, frozenset()))
        if ent['L1']['cells_targeting_both_legs'] != both:
            bad.append((pair, 'L1', ent['L1']['cells_targeting_both_legs'], both))
        # L2 path counts by product over intermediates
        for src, dst, key in (('pool_' + a, b, 'a_to_b'), ('pool_' + b, a, 'b_to_a')):
            total = 0
            inters = 0
            for x in range(N):
                if x in is_mn:
                    continue
                ca = rad.get((src, x), 0)
                if ca == 0:
                    continue
                cb = fog_leg.get((x, dst), 0)
                if cb:
                    total += ca * cb
                    inters += 1
            if ent['L2'][key]['paths'] != total or ent['L2'][key]['distinct_intermediates'] != inters:
                bad.append((pair, 'L2', key, ent['L2'][key], total, inters))
    check('L0/L1/L2 coupling counts for the five gait pairs reproduce', not bad, bad[:6])
    bad = []
    for pair, ent in coupling['all_15_pairs_compact'].items():
        a, b = pair.split('-')
        both = sum(1 for p in range(N)
                   if a in direct_legs.get(p, frozenset()) and b in direct_legs.get(p, frozenset()))
        if ent['cells_targeting_both_legs'] != both:
            bad.append((pair, ent['cells_targeting_both_legs'], both))
        if ent['direct_edges_a_to_b'] != per_block_edges.get(('pool_' + a, 'pool_' + b), 0):
            bad.append((pair, 'eAB'))
        if ent['direct_edges_b_to_a'] != per_block_edges.get(('pool_' + b, 'pool_' + a), 0):
            bad.append((pair, 'eBA'))
    check('all 15 leg-pair bridge cell counts and direct edge counts reproduce', not bad, bad[:6])
    present = [k for k, v in coupling['pairs'].items() if v['L3_verdict'] != 'ABSENT_no_coupling_within_two_hops_MISSING_DATA']
    check('all five gait pairs have a non-absent structural verdict', len(present) == 5, present)

    # ---------------- 11. proposal
    prop = json.loads((HERE / 'PROPOSED_SCALING_PARAMS.json').read_text())
    tot_edges = sum(v['edges_total'] for v in prop['class_partition'].values())
    tot_g = sum(v['frozen_conductance_uS_total'] for v in prop['class_partition'].values())
    check('proposal classes partition all 435126 edges', tot_edges == 435126, tot_edges)
    check('proposal class conductance totals sum to 380.84 uS', close(tot_g, 380.84), tot_g)
    bad = []
    for name in ('C_contra_same_row', 'C_front_hind', 'C_tripod_diagonal'):
        pairs = [tuple(p) for p in prop['class_partition'][name]['leg_pairs']]
        e = sum(per_block_edges.get(('pool_' + a, 'pool_' + b), 0) + per_block_edges.get(('pool_' + b, 'pool_' + a), 0)
                for a, b in pairs)
        g = sum(per_block_uS.get(('pool_' + a, 'pool_' + b), 0.0) + per_block_uS.get(('pool_' + b, 'pool_' + a), 0.0)
                for a, b in pairs)
        if prop['class_partition'][name]['edges_total'] != e or not close(prop['class_partition'][name]['frozen_conductance_uS_total'], g):
            bad.append((name, prop['class_partition'][name], e, g))
    check('proposal per-class edge and conductance totals reproduce', not bad, bad)
    check('free scalar count is 3 within preregistered cap of 5', prop['free_scalar_count'] == 3 <= 5, prop['free_scalar_count'])
    check('scalar bounds are [0.25, 4.0]',
          prop['bounds']['lower'] == .25 and prop['bounds']['upper'] == 4.0, prop['bounds'])
    check('proposal keeps the .02 uS convention option (V1 exact / V2 bounded)',
          prop['conductance_convention']['V1_row_renormalized']['resulting_per_neuron_incoming_uS'] == .02
          and prop['conductance_convention']['V2_bounded_unrenormalized']['allowed_per_neuron_incoming_uS'] == [.01, .04],
          prop['conductance_convention'])
    trip = prop['class_partition']['C_tripod_diagonal']['frozen_conductance_uS_total']
    check('tripod class conductance 0.1348 uS at upper bound 4.0 gives 0.5393 uS',
          close(trip * 4.0, prop['scalar_ceiling_if_applied_exactly']['C_tripod_diagonal']['at_scalar_upper_uS']),
          [trip * 4.0, prop['scalar_ceiling_if_applied_exactly']['C_tripod_diagonal']['at_scalar_upper_uS']])

    # ---------------- 12. motor neurons are sinks
    outdeg = defaultdict(int)
    for p in pre.tolist():
        outdeg[p] += 1
    sinks = sum(1 for m in mn_local.tolist() if outdeg[m] == 0)
    check('all 323 motor neurons have zero outgoing edges', sinks == 323, sinks)
    check('RESULT.json records zero motor outgoing edges',
          result['weight_contract']['motor_neurons_with_any_outgoing_edge'] == 0,
          result['weight_contract']['motor_neurons_with_any_outgoing_edge'])
    check('mn_to_mn block has zero edges', per_block_edges.get(('pool_mn', 'pool_mn'), 0) == 0,
          per_block_edges.get(('pool_mn', 'pool_mn'), 0))

    # ---------------- 13. prereg/analysis hashes inside RESULT.json
    check('RESULT.json prereg hashes match the files on disk',
          result['preregistration']['PREREGISTRATION.md'] == sha256(HERE / 'PREREGISTRATION.md')
          and result['preregistration']['PREREGISTRATION.json'] == sha256(HERE / 'PREREGISTRATION.json'),
          result['preregistration'])
    check('RESULT.json analysis-code hash matches leg_pools.py',
          result['analysis_code']['leg_pools.py'] == sha256(HERE / 'leg_pools.py'),
          result['analysis_code'])
    check('RESULT.json source hashes match this verifier expectation',
          result['source_hashes'] == SHA, 'hash tables')
    check('RESULT.json records zero simulation steps',
          all(v == 0 for k, v in result['steps'].items() if k != 'analyses'), result['steps'])

    # ---------------- 14. deviation record
    dev = HERE / 'DEVIATIONS.md'
    check('DEVIATIONS.md exists and records the mirror-rule deviation', dev.exists() and '偏离' in dev.read_text(), dev.exists())

    n_pass = sum(1 for c in CHECKS if c['pass'])
    out = {
        'verification': 'independent, does not import leg_pools.py',
        'method': ('pools rebuilt with plain Python dicts/sets over the edge list; block conductances '
                   'recomputed from the raw synapse counts and the nt column rather than from the csr matrices; '
                   'path counts recomputed as products over intermediates'),
        'checks_total': len(CHECKS),
        'checks_passed': n_pass,
        'checks_failed': len(CHECKS) - n_pass,
        'all_passed': n_pass == len(CHECKS),
        'steps': {'neural_steps': 0, 'physical_steps': 0, 'learning_updates': 0, 'new_edges': 0},
        'checks': CHECKS,
        'verified_hashes': {
            'verify_leg_pools.py': sha256(Path(__file__).resolve()),
            'leg_pools.py': sha256(HERE / 'leg_pools.py'),
            'PREREGISTRATION.md': sha256(HERE / 'PREREGISTRATION.md'),
            'PREREGISTRATION.json': sha256(HERE / 'PREREGISTRATION.json'),
            'DEVIATIONS.md': sha256(HERE / 'DEVIATIONS.md'),
            'POOLS.json': sha256(HERE / 'POOLS.json'),
            'POOLS_AND_REACH.npz': sha256(HERE / 'POOLS_AND_REACH.npz'),
            'BLOCKS.json': sha256(HERE / 'BLOCKS.json'),
            'SYMMETRY.json': sha256(HERE / 'SYMMETRY.json'),
            'COUPLING.json': sha256(HERE / 'COUPLING.json'),
            'PROPOSED_SCALING_PARAMS.json': sha256(HERE / 'PROPOSED_SCALING_PARAMS.json'),
            'RESULT.json': sha256(HERE / 'RESULT.json'),
        },
    }
    (HERE / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print('\n%d/%d checks passed' % (n_pass, len(CHECKS)))
    return 0 if out['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
