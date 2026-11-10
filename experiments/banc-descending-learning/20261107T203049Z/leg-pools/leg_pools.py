"""Leg-specific premotor pool audit of the frozen strict subgraph (offline, 0 dynamics steps).

Reads the frozen 19:29:31Z module to obtain the authoritative `_We`/`_Wi` (uS), reads the
frozen 18:29:40Z neural-interface for edge identity/synapse counts/leg labels, and derives:
  * per-neuron direct and two-hop MN-leg reach
  * a 9-bucket partition of all 20,000 neurons
  * 9x9 conductance/edge blocks for both receptor types (real uS, not gains)
  * aggregate within / L->R / F->H / leg-pair views and the premotor drive matrix
  * bilateral mirror-symmetry indices (exact integers + raw uS)
  * exact 1-hop / 2-hop coupling counts for the five gait-relevant leg pairs
  * a bounded rescaling proposal (design document, machine readable)

No simulation, no learning, no new edges, no writes to any frozen file.
Definitions and decision rules are fixed in PREREGISTRATION.md / PREREGISTRATION.json,
which this script hashes so post-hoc edits are detectable.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import scipy
from scipy import sparse

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
MODULE_192931 = PROJECT / 'experiments/banc-local-learning/20261107T192931Z/dynamics/module.py'
MODULE_185910 = PROJECT / 'experiments/banc-local-learning/20261107T185910Z/dynamics/module.py'
ENGINE = PROJECT / 'engine/neural_cond.py'
NI = PROJECT / 'experiments/flybody-retarget/20261107T182940Z/neural-interface'

EXPECTED_SHA256 = {
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
LEG_BIT = {leg: 1 << i for i, leg in enumerate(LEGS)}
BUCKETS = ['pool_lf', 'pool_lh', 'pool_lm', 'pool_rf', 'pool_rh', 'pool_rm',
           'pool_shared', 'pool_nonmotor', 'pool_mn']
BIDX = {b: i for i, b in enumerate(BUCKETS)}
NB = len(BUCKETS)
MIRROR = {'pool_lf': 'pool_rf', 'pool_rf': 'pool_lf',
          'pool_lh': 'pool_rh', 'pool_rh': 'pool_lh',
          'pool_lm': 'pool_rm', 'pool_rm': 'pool_lm',
          'pool_shared': 'pool_shared', 'pool_nonmotor': 'pool_nonmotor',
          'pool_mn': 'pool_mn'}
ASYM_TOL = 0.05
GAIT_PAIRS = [('lf', 'rf'), ('lm', 'rm'), ('lh', 'rh'), ('lf', 'rh'), ('rf', 'lh')]
IPSILATERAL_PAIRS = [('lf', 'lm'), ('lm', 'lh'), ('lf', 'lh'), ('rf', 'rm'), ('rm', 'rh'), ('rf', 'rh')]
CONTRA_SAME_ROW = [('lf', 'rf'), ('lm', 'rm'), ('lh', 'rh')]
CONTRA_CROSS_ROW = [('lf', 'rm'), ('lf', 'rh'), ('lh', 'rm'), ('lh', 'rf'), ('lm', 'rf'), ('lm', 'rh')]
TRIPOD = [('lf', 'rh'), ('rf', 'lh')]
ALL_PAIRS = [tuple(p) for p in IPSILATERAL_PAIRS + CONTRA_SAME_ROW + CONTRA_CROSS_ROW]
if len(ALL_PAIRS) != 15 or len(set(frozenset(p) for p in ALL_PAIRS)) != 15:
    raise SystemExit('ABORT: leg-pair grouping is not a partition of the 15 unordered pairs')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def log(msg):
    print('[leg-pools] ' + msg, flush=True)


def check_sources():
    got, bad = {}, []
    for path, want in EXPECTED_SHA256.items():
        have = sha256(path)
        got[path] = have
        if have != want:
            bad.append({'path': path, 'expected': want, 'actual': have})
    if bad:
        raise SystemExit('ABORT: frozen source hash mismatch ' + json.dumps(bad, indent=2))
    log('source hashes verified (%d files)' % len(got))
    return got


def load_frozen():
    if '_legpools_frozen_module' not in sys.modules:
        spec = importlib.util.spec_from_file_location('_legpools_frozen_module', MODULE_192931)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['_legpools_frozen_module'] = mod
        spec.loader.exec_module(mod)
    mod = sys.modules['_legpools_frozen_module']
    return mod.LocalCircuit(total_conductance_uS=.02)


def load_edges():
    with np.load(NI / 'LOCAL_MODULE_CANDIDATE.npz', allow_pickle=False) as z:
        pre_all = z['pre_local']
        post_all = z['post_local']
        syn_all = z['synapse_counts']
        nt_all = z['source_verified_nt']
        rows_all = z['source_edge_rows']
        root_ids = z['root_ids'].copy()
        cache_idx = z['cache_indices'].copy()
        eligible = np.isin(nt_all, ['acetylcholine', 'gaba', 'glutamate'])
        if not np.array_equal(eligible, z['pure_verified_label_mask']):
            raise SystemExit('ABORT: strict label mask disagrees with pure source labels')
        strict_rows = z['strict_source_edge_rows'].copy()
    if int(eligible.sum()) != 435126:
        raise SystemExit('ABORT: strict edge count changed')
    return {
        'pre': pre_all[eligible].astype(np.int64),
        'post': post_all[eligible].astype(np.int64),
        'syn': syn_all[eligible].astype(np.float64),
        'nt': nt_all[eligible],
        'source_edge_rows': rows_all[eligible].astype(np.int64),
        'root_ids': root_ids,
        'cache_indices': cache_idx,
        'strict_source_edge_rows': strict_rows,
    }


def leg_label_arrays(circuit):
    mn_local = np.asarray(circuit.mn_local, dtype=np.int64)
    mn_leg_str = np.asarray([str(x) for x in circuit.mn_leg])
    leg_of_mn = np.array([LEGS.index(x) for x in mn_leg_str], dtype=np.int64)
    leg_code = np.full(N, -1, dtype=np.int64)
    leg_code[mn_local] = leg_of_mn
    is_mn = np.zeros(N, dtype=bool)
    is_mn[mn_local] = True
    return mn_local, mn_leg_str, leg_of_mn, leg_code, is_mn


def leg_masks(pre, post, leg_code, is_mn):
    """Direct leg-reach bitmask, 2-hop bitmask, per-leg direct targets and 2-hop path counts."""
    is_mn_edge = is_mn[post]
    e_pre, e_post = pre[is_mn_edge], post[is_mn_edge]
    lg = leg_code[e_post]

    # per (cell, leg): number of MN cells of that leg hit by one existing edge
    targets = np.zeros((N, len(LEGS)), dtype=np.int32)
    np.add.at(targets, (e_pre, lg), 1)
    mask_direct = np.zeros(N, dtype=np.uint8)
    np.bitwise_or.at(mask_direct, e_pre, (1 << lg).astype(np.uint8))

    # two-hop: p -> x -> q through an intermediate x that is not a MN
    step1 = ~is_mn[post]
    s1_pre, s1_mid = pre[step1], post[step1]
    mask_2hop = np.zeros(N, dtype=np.uint8)
    counts2 = np.zeros((N, len(LEGS)), dtype=np.int32)
    for li in range(len(LEGS)):
        w = targets[:, li]
        hit = w[s1_mid] > 0
        if hit.any():
            np.bitwise_or.at(mask_2hop, s1_pre[hit], np.uint8(1 << li))
        contrib = w[s1_mid].astype(np.int64)
        if contrib.any():
            np.add.at(counts2[:, li], s1_pre, contrib)
    return mask_direct, targets, mask_2hop, counts2


def bucket_ids(mask):
    """Preregistered order: empty -> nonmotor; single leg -> pool_<leg>; >=2 legs -> shared."""
    popcount = np.array([bin(int(m)).count('1') for m in range(256)], dtype=np.int64)
    bid = np.full(N, BIDX['pool_nonmotor'], dtype=np.int64)
    for i, leg in enumerate(LEGS):
        bid[mask == (1 << i)] = BIDX['pool_' + leg]
    bid[popcount[mask] >= 2] = BIDX['pool_shared']
    return bid, popcount[mask]


def block_tables(X, bid_src, bid_dst):
    """Exact per-(src,dst) bucket block statistics for one sparse matrix X (uS)."""
    x = X.tocoo()
    dst = x.row.astype(np.int64)
    src = x.col.astype(np.int64)
    data = np.asarray(x.data, dtype=np.float64)
    pair = bid_src[src] * NB + bid_dst[dst]
    order = np.argsort(pair, kind='stable')
    pair_s, data_s = pair[order], data[order]
    uniq, starts, counts = np.unique(pair_s, return_index=True, return_counts=True)

    gsum = np.zeros(NB * NB, dtype=np.float64)
    edges = np.zeros(NB * NB, dtype=np.int64)
    amax = np.zeros(NB * NB, dtype=np.float64)
    amin = np.zeros(NB * NB, dtype=np.float64)
    amed = np.zeros(NB * NB, dtype=np.float64)
    q = np.zeros((NB * NB, 4), dtype=np.float64)  # q05, q25, q75, q95
    for u, st, cn in zip(uniq, starts, counts):
        seg = data_s[st:st + cn]
        edges[u] = cn
        gsum[u] = seg.sum()
        amax[u] = seg.max()
        amin[u] = seg.min()
        amed[u] = float(np.quantile(seg, .5, method='linear'))
        q[u] = np.quantile(seg, [.05, .25, .75, .95], method='linear')
    return {'edges': edges, 'gsum': gsum, 'max': amax, 'min': amin, 'median': amed,
            'q05': q[:, 0], 'q25': q[:, 1], 'q75': q[:, 2], 'q95': q[:, 3],
            'src': src, 'dst': dst, 'nnz': int(data.size)}


def syn_tables(pre, post, syn, bid_src, bid_dst):
    pair = bid_src[pre] * NB + bid_dst[post]
    return (np.bincount(pair, minlength=NB * NB).astype(np.int64),
            np.bincount(pair, weights=syn, minlength=NB * NB))


def strength_label(g, T):
    if g == 0:
        return 'absent'
    if g <= 1e-4 * T:
        return 'trace'
    if g <= 1e-3 * T:
        return 'very_weak'
    if g <= 1e-2 * T:
        return 'weak'
    if g <= 1e-1 * T:
        return 'moderate'
    return 'strong'


def main():
    source_hashes = check_sources()
    circuit = load_frozen()
    e = load_edges()
    pre, post, syn = e['pre'], e['post'], e['syn']
    mn_local, mn_leg_str, leg_of_mn, leg_code, is_mn = leg_label_arrays(circuit)
    We, Wi = circuit._We, circuit._Wi

    if We.nnz + Wi.nnz != pre.size or We.nnz + Wi.nnz != 435126:
        raise SystemExit('ABORT: frozen matrix nnz total %d != 435126' % (We.nnz + Wi.nnz))

    # ---------- weight contract re-derived from the raw source, compared to the frozen matrices
    incoming_raw = np.bincount(post, weights=syn, minlength=N)
    w_from_raw = .02 * syn / incoming_raw[post]
    exc_edge = e['nt'] == 'acetylcholine'
    recon_e = sparse.csr_matrix((w_from_raw[exc_edge], (post[exc_edge], pre[exc_edge])), shape=(N, N))
    recon_i = sparse.csr_matrix((w_from_raw[~exc_edge], (post[~exc_edge], pre[~exc_edge])), shape=(N, N))
    we_coo, wi_coo = We.tocoo(), Wi.tocoo()
    key_raw = np.sort(post * N + pre)
    key_frozen = np.sort(np.concatenate([we_coo.row.astype(np.int64) * N + we_coo.col.astype(np.int64),
                                         wi_coo.row.astype(np.int64) * N + wi_coo.col.astype(np.int64)]))
    pairs_identical = bool(key_raw.size == key_frozen.size and np.array_equal(key_raw, key_frozen))
    if not pairs_identical:
        raise SystemExit('ABORT: frozen (post,pre) pair multiset differs from the strict source edge list')
    wsum_we = float(We.sum())
    wsum_wi = float(Wi.sum())
    row_tot = np.asarray(We.sum(1)).ravel() + np.asarray(Wi.sum(1)).ravel()
    live = incoming_raw > 0
    weight_contract = {
        'max_abs_diff_We_vs_source_formula_uS': float(np.abs(recon_e - We).max()),
        'max_abs_diff_Wi_vs_source_formula_uS': float(np.abs(recon_i - Wi).max()),
        'reproduced_from_source_formula': bool(float(np.abs(recon_e - We).max()) == 0.0
                                               and float(np.abs(recon_i - Wi).max()) == 0.0),
        'frozen_pre_post_pairs_identical_to_strict_edge_list': pairs_identical,
        'We_nnz': int(We.nnz), 'Wi_nnz': int(Wi.nnz), 'total_nnz': int(We.nnz + Wi.nnz),
        'We_total_uS': wsum_we, 'Wi_total_uS': wsum_wi, 'module_total_uS': wsum_we + wsum_wi,
        'neurons_with_incoming': int(live.sum()),
        'per_neuron_incoming_total_uS_distinct_values': sorted({float(x) for x in row_tot[live]}),
        'per_edge_weight_uS_min': float(min(We.data.min(), Wi.data.min())),
        'per_edge_weight_uS_max': float(max(We.data.max(), Wi.data.max())),
        'source_synapses_total': float(syn.sum()),
        'source_synapses_We': float(syn[exc_edge].sum()),
        'source_synapses_Wi': float(syn[~exc_edge].sum()),
        'formula': 'weight[e] = 0.02 uS * synapse_counts[e] / sum_both_receptors(synapse_counts into post[e])',
        'weight_uS_global_min': float(min(We.data.min(), Wi.data.min())),
        'weight_uS_global_max': float(max(We.data.max(), Wi.data.max())),
        'weight_uS_global_median': float(np.median(np.concatenate([We.data, Wi.data]))),
        'weight_uS_global_q05': float(np.quantile(np.concatenate([We.data, Wi.data]), .05)),
        'weight_uS_global_q95': float(np.quantile(np.concatenate([We.data, Wi.data]), .95)),
        'motor_neurons_with_any_outgoing_edge': int((np.bincount(pre, minlength=N)[mn_local] > 0).sum()),
        'motor_neurons_incoming_edges_min': int(np.bincount(post, minlength=N)[mn_local].min()),
        'motor_neurons_incoming_edges_max': int(np.bincount(post, minlength=N)[mn_local].max()),
        'motor_neurons_total_incoming_uS': float(.02 * mn_local.size),
    }
    log('weight contract: reproduced=%s total=%.12f uS (We %.12f + Wi %.12f), nnz %d'
        % (weight_contract['reproduced_from_source_formula'], weight_contract['module_total_uS'],
           wsum_we, wsum_wi, weight_contract['total_nnz']))

    # ---------- reach sets and buckets
    mask_direct, direct_targets, mask_2hop, counts2 = leg_masks(pre, post, leg_code, is_mn)
    mask_union = mask_direct | mask_2hop
    bid, n_legs_direct = bucket_ids(mask_direct)
    bid_union, n_legs_union = bucket_ids(mask_union)
    bid[is_mn] = BIDX['pool_mn']
    bid_union[is_mn] = BIDX['pool_mn']
    sizes = np.bincount(bid, minlength=NB)
    sizes_u = np.bincount(bid_union, minlength=NB)
    if int(sizes.sum()) != N or int(sizes_u.sum()) != N:
        raise SystemExit('ABORT: bucket sizes do not sum to 20000')
    pool_sizes = {BUCKETS[i]: int(sizes[i]) for i in range(NB)}
    pool_sizes_u = {BUCKETS[i]: int(sizes_u[i]) for i in range(NB)}
    log('pools built: ' + json.dumps(pool_sizes))

    # ---------- declared port sets (identity only, no biological inference)
    groups = np.load(NI / 'GROUP_CACHE_INDICES.npz', allow_pickle=False)
    review = json.loads((NI / 'SENSORY_CHANNEL_REVIEW.json').read_text())
    excluded_cache = np.array([int(r['cache_index_0based']) for r in review['ambiguous_records']], dtype=np.int64)
    local_of = lambda key: np.flatnonzero(np.isin(e['cache_indices'], groups[key]))
    role_sets = {
        'allowed_sensory_local': np.setdiff1d(local_of('leg_sensory_nerve_union'),
                                              np.flatnonzero(np.isin(e['cache_indices'], excluded_cache))),
        'descending_local': local_of('descending_boundary'),
        'ascending_local': local_of('ascending_boundary'),
        'leg_motor_nerve_union_local': local_of('leg_motor_nerve_union'),
        'vnc_intrinsic_local': local_of('vnc_intrinsic'),
    }
    for leg in LEGS:
        role_sets['leg_motor_nerve_' + leg] = local_of('leg_motor_nerve_' + leg)
    if not np.array_equal(np.sort(role_sets['allowed_sensory_local']), np.sort(circuit.allowed_sensory_local)):
        raise SystemExit('ABORT: reconstructed sensory port set differs from the frozen circuit')
    if not np.array_equal(np.sort(role_sets['descending_local']), np.sort(circuit.descending_local)):
        raise SystemExit('ABORT: reconstructed descending port set differs from the frozen circuit')

    def combos(mask):
        out = {}
        for m in np.unique(mask):
            legs = [LEGS[i] for i in range(len(LEGS)) if (int(m) >> i) & 1]
            out['+'.join(legs) if legs else 'none'] = int((mask == m).sum())
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    pools = {
        'primary_partition': pool_sizes,
        'primary_partition_definition': 'p in M -> pool_mn; else |D(p)|==0 -> pool_nonmotor; |D(p)|==1 -> pool_<leg>; |D(p)|>=2 -> pool_shared',
        'supplementary_partition_union_of_direct_and_2hop': pool_sizes_u,
        'mn_count': int(mn_local.size),
        'mn_leg_counts': {LEGS[i]: int((leg_of_mn == i).sum()) for i in range(len(LEGS))},
        'direct_leg_reach_combo_counts': combos(mask_direct),
        'union_leg_reach_combo_counts': combos(mask_union),
        'neurons_reaching_any_mn_direct': int((mask_direct > 0).sum()),
        'neurons_reaching_any_mn_by_2hop_only_or_more': int((mask_2hop > 0).sum()),
        'neurons_reaching_no_mn_within_two_hops': int((mask_union == 0).sum()),
        'nonmotor_cells_that_do_reach_a_mn_within_2hops': int(((bid == BIDX['pool_nonmotor']) & (n_legs_union > 0)).sum()),
        'neurons_with_no_outgoing_edge': int((np.bincount(pre, minlength=N) == 0).sum()),
        'neurons_with_no_incoming_edge': int((incoming_raw == 0).sum()),
        'role_overlap_counts_per_bucket': {b: {k: int(np.isin(np.flatnonzero(bid == BIDX[b]), v).sum())
                                              for k, v in role_sets.items()} for b in BUCKETS},
        'role_set_sizes': {k: int(v.size) for k, v in role_sets.items()},
        'all_ports_outside_mn': {
            'allowed_sensory_local': int(np.isin(role_sets['allowed_sensory_local'], mn_local).sum()),
            'descending_local': int(np.isin(role_sets['descending_local'], mn_local).sum()),
        },
        'coverage_checks': {
            'sum_of_bucket_sizes': int(sizes.sum()),
            'expected': N,
            'every_neuron_exactly_one_bucket': bool(int(sizes.sum()) == N),
            'note': 'bid is a single int per neuron, so exclusivity is structural; the sum check is the only numeric invariant to test',
        },
    }
    np.savez_compressed(
        HERE / 'POOLS_AND_REACH.npz',
        bucket_id=bid.astype(np.int8), bucket_id_union=bid_union.astype(np.int8),
        leg_mask_direct=mask_direct, leg_mask_2hop=mask_2hop, leg_mask_union=mask_union,
        direct_mn_target_counts=direct_targets, two_hop_path_counts=counts2,
        leg_names=np.array(LEGS), bucket_names=np.array(BUCKETS),
        mn_local=mn_local, leg_of_mn=leg_of_mn,
    )
    (HERE / 'POOLS.json').write_text(json.dumps(pools, indent=2, ensure_ascii=False) + '\n')
    log('POOLS.json written (primary %s)'
        % json.dumps({k: pool_sizes[k] for k in ['pool_lf', 'pool_lh', 'pool_lm', 'pool_rf', 'pool_rh', 'pool_rm',
                                                 'pool_shared', 'pool_nonmotor', 'pool_mn']}))

    # ---------- blocks
    T = weight_contract['module_total_uS']
    exc_tab = block_tables(We, bid, bid)
    inh_tab = block_tables(Wi, bid, bid)
    syn_all = syn_tables(pre, post, syn, bid, bid)
    syn_e = syn_tables(pre[exc_edge], post[exc_edge], syn[exc_edge], bid, bid)
    syn_i = syn_tables(pre[~exc_edge], post[~exc_edge], syn[~exc_edge], bid, bid)
    if int(exc_tab['edges'].sum() + inh_tab['edges'].sum()) != 435126:
        raise SystemExit('ABORT: block edge total != 435126')
    if int(syn_all[0].sum()) != 435126:
        raise SystemExit('ABORT: synapse table edge total != 435126')
    if abs(float(exc_tab['gsum'].sum() + inh_tab['gsum'].sum()) - T) > 1e-9:
        raise SystemExit('ABORT: block conductance total differs from the module total')
    log('blocks quantified: 9x9 for _We and _Wi, edge total 435126, conductance total %.12f uS' % T)

    Ew = exc_tab['edges'].reshape(NB, NB)
    Gw = exc_tab['gsum'].reshape(NB, NB)
    Ei = inh_tab['edges'].reshape(NB, NB)
    Gi = inh_tab['gsum'].reshape(NB, NB)
    E = Ew + Ei
    G = Gw + Gi

    def blocks_axis(tab, syn_tab, receptor):
        out = {}
        for bi, bsrc in enumerate(BUCKETS):
            row = {}
            for bj, bdst in enumerate(BUCKETS):
                u = bi * NB + bj
                n = int(tab['edges'][u])
                g = float(tab['gsum'][u])
                ent = {'edges': n, 'synapses': int(syn_tab[1][u]), 'conductance_uS': g,
                       'block_share_of_module_total': (g / T), 'strength_label': strength_label(g, T)}
                if n:
                    ent.update({'mean_uS': g / n, 'median_uS': float(tab['median'][u]),
                                'min_uS': float(tab['min'][u]), 'max_uS': float(tab['max'][u]),
                                'q05_uS': float(tab['q05'][u]), 'q25_uS': float(tab['q25'][u]),
                                'q75_uS': float(tab['q75'][u]), 'q95_uS': float(tab['q95'][u])})
                row[bdst] = ent
            out[bsrc] = row
        return {'receptor': receptor, 'blocks': out}

    def agg(src_names, dst_names):
        si = np.array([BIDX[b] for b in src_names], dtype=np.int64)
        di = np.array([BIDX[b] for b in dst_names], dtype=np.int64)
        sel = np.ix_(si, di)
        we = {'edges': int(Ew[sel].sum()), 'conductance_uS': float(Gw[sel].sum())}
        wi = {'edges': int(Ei[sel].sum()), 'conductance_uS': float(Gi[sel].sum())}
        return {'_We': we, '_Wi': wi,
                'both': {'edges': we['edges'] + wi['edges'],
                         'conductance_uS': we['conductance_uS'] + wi['conductance_uS']}}

    left = ['pool_lf', 'pool_lh', 'pool_lm']
    right = ['pool_rf', 'pool_rh', 'pool_rm']
    front = ['pool_lf', 'pool_rf']
    hind = ['pool_lh', 'pool_rh']
    sixpools = ['pool_' + l for l in LEGS]
    aggregate = {
        'within_pool_diagonal': {b: agg([b], [b]) for b in sixpools},
        'left_to_right': agg(left, right), 'right_to_left': agg(right, left),
        'left_internal': agg(left, left), 'right_internal': agg(right, right),
        'front_to_hind': agg(front, hind), 'hind_to_front': agg(hind, front),
        'front_internal': agg(front, front), 'hind_internal': agg(hind, hind),
        'leg_pools_to_mn_bucket': agg(sixpools, ['pool_mn']),
        'shared_to_mn_bucket': agg(['pool_shared'], ['pool_mn']),
        'nonmotor_to_mn_bucket': agg(['pool_nonmotor'], ['pool_mn']),
        'shared_to_leg_pools': agg(['pool_shared'], sixpools),
        'leg_pools_to_shared': agg(sixpools, ['pool_shared']),
        'nonmotor_to_leg_pools': agg(['pool_nonmotor'], sixpools),
        'leg_pools_to_nonmotor': agg(sixpools, ['pool_nonmotor']),
        'mn_to_mn': agg(['pool_mn'], ['pool_mn']),
        'shared_internal': agg(['pool_shared'], ['pool_shared']),
        'nonmotor_internal': agg(['pool_nonmotor'], ['pool_nonmotor']),
    }

    leg_i = {l: BIDX['pool_' + l] for l in LEGS}
    legpair = {}
    for lab, Ex, Gx in (('_We', Ew, Gw), ('_Wi', Ei, Gi), ('both', E, G)):
        legpair[lab] = {
            'edges_leg_src_x_leg_dst': {a: {b: int(Ex[leg_i[a], leg_i[b]]) for b in LEGS} for a in LEGS},
            'conductance_uS_leg_src_x_leg_dst': {a: {b: float(Gx[leg_i[a], leg_i[b]]) for b in LEGS} for a in LEGS},
        }

    def pair_block(a, b, lab):
        Ex, Gx = {'_We': (Ew, Gw), '_Wi': (Ei, Gi), 'both': (E, G)}[lab]
        return {'edges_a_to_b': int(Ex[leg_i[a], leg_i[b]]),
                'edges_b_to_a': int(Ex[leg_i[b], leg_i[a]]),
                'edges_both_directions': int(Ex[leg_i[a], leg_i[b]] + Ex[leg_i[b], leg_i[a]]),
                'conductance_a_to_b_uS': float(Gx[leg_i[a], leg_i[b]]),
                'conductance_b_to_a_uS': float(Gx[leg_i[b], leg_i[a]]),
                'conductance_both_directions_uS': float(Gx[leg_i[a], leg_i[b]] + Gx[leg_i[b], leg_i[a]])}

    unordered = {}
    for group, pairs in (('ipsilateral', IPSILATERAL_PAIRS),
                         ('contralateral_same_row', CONTRA_SAME_ROW),
                         ('contralateral_cross_row', CONTRA_CROSS_ROW)):
        unordered[group] = {'%s-%s' % (a, b): {lab: pair_block(a, b, lab) for lab in ('_We', '_Wi', 'both')}
                            for a, b in pairs}
    unordered['tripod_partners_listed_separately'] = {
        '%s-%s' % (a, b): {lab: pair_block(a, b, lab) for lab in ('_We', '_Wi', 'both')} for a, b in TRIPOD}
    unordered['group_totals'] = {}
    for group, pairs in (('ipsilateral', IPSILATERAL_PAIRS),
                         ('contralateral_same_row', CONTRA_SAME_ROW),
                         ('contralateral_cross_row', CONTRA_CROSS_ROW),
                         ('tripod_partners', TRIPOD)):
        unordered['group_totals'][group] = {
            lab: {'edges_both_directions': int(sum(pair_block(a, b, lab)['edges_both_directions'] for a, b in pairs)),
                  'conductance_both_directions_uS': float(sum(pair_block(a, b, lab)['conductance_both_directions_uS']
                                                             for a, b in pairs))}
            for lab in ('_We', '_Wi', 'both')}

    mn_by_leg = {LEGS[i]: mn_local[leg_of_mn == i] for i in range(len(LEGS))}
    drive = {}
    for lab, X in (('_We', We), ('_Wi', Wi)):
        rows = X[mn_local, :]
        pos = {int(g): k for k, g in enumerate(mn_local)}
        row_of_leg = {leg: np.array([pos[int(g)] for g in mn_by_leg[leg]], dtype=np.int64) for leg in LEGS}
        mat = {}
        for bi, bsrc in enumerate(BUCKETS):
            cols = np.flatnonzero(bid == bi)
            sub = rows[:, cols]
            mat[bsrc] = {leg: {'edges': int(sub[row_of_leg[leg], :].nnz),
                               'conductance_uS': float(sub[row_of_leg[leg], :].sum())} for leg in LEGS}
        drive[lab] = mat
    drive['both'] = {bsrc: {leg: {'edges': drive['_We'][bsrc][leg]['edges'] + drive['_Wi'][bsrc][leg]['edges'],
                                  'conductance_uS': drive['_We'][bsrc][leg]['conductance_uS'] + drive['_Wi'][bsrc][leg]['conductance_uS']}
                            for leg in LEGS} for bsrc in BUCKETS}
    syn_drive = {}
    for bi, bsrc in enumerate(BUCKETS):
        sel_src = bid[pre] == bi
        row = {}
        for leg in LEGS:
            sel = sel_src & np.isin(post, mn_by_leg[leg])
            row[leg] = {'edges': int(sel.sum()), 'synapses': int(syn[sel].sum())}
        syn_drive[bsrc] = row

    incoming_decomp = {}
    for lab, Gx in (('_We', Gw), ('_Wi', Gi), ('both', G)):
        incoming_decomp[lab] = {bdst: {
            'total_incoming_conductance_uS': float(Gx[:, bj].sum()),
            'by_source_bucket_uS': {BUCKETS[bi]: float(Gx[bi, bj]) for bi in range(NB)},
            'by_source_bucket_fraction': {BUCKETS[bi]: (float(Gx[bi, bj]) / float(Gx[:, bj].sum())
                                                        if Gx[:, bj].sum() else None) for bi in range(NB)},
        } for bj, bdst in enumerate(BUCKETS)}

    blocks = {
        'bucket_order': BUCKETS,
        'leg_order': LEGS,
        'units': 'uS (real frozen conductance, .02 uS per-neuron incoming convention)',
        'total_module_conductance_uS': T,
        'exc': blocks_axis(exc_tab, syn_e, '_We / acetylcholine'),
        'inh': blocks_axis(inh_tab, syn_i, '_Wi / gaba+glutamate'),
        'aggregates': aggregate,
        'leg_pair_matrices_6x6': legpair,
        'unordered_leg_pairs_with_grouping': unordered,
        'premotor_drive_to_mn_legs': drive,
        'premotor_drive_source_synapses': syn_drive,
        'incoming_conductance_decomposition_per_target_bucket': incoming_decomp,
    }
    (HERE / 'BLOCKS.json').write_text(json.dumps(blocks, indent=2, ensure_ascii=False) + '\n')
    log('BLOCKS.json written')

    # ---------- symmetry
    mir_idx = np.array([BIDX[MIRROR[b]] for b in BUCKETS], dtype=np.int64)

    def mirror_mats(Gx, idx):
        pre_reg = np.zeros_like(Gx)   # PREREGISTERED rule: G'[s][d] = G[mir(d)][mir(s)]  (mirror AND reverse)
        corrected = np.zeros_like(Gx)  # CORRECTED rule:     G'[s][d] = G[mir(s)][mir(d)]  (mirror only)
        nb_ = Gx.shape[0]
        for bi in range(nb_):
            for bj in range(nb_):
                pre_reg[bi, bj] = Gx[idx[bj], idx[bi]]
                corrected[bi, bj] = Gx[idx[bi], idx[bj]]
        return pre_reg, corrected

    def asym_block(Gx, Gp, names):
        num = float(np.abs(Gx - Gp).sum())
        den = float((Gx + Gp).sum())
        idx = num / den if den else None
        nb_ = Gx.shape[0]
        return {'asym_index_L1': idx, 'abs_diff_sum_uS': num, 'sum_uS': den,
                'asymmetry_detected': (abs(idx) >= ASYM_TOL) if idx is not None else None,
                'per_block_directed_uS': {names[bi]: {names[bj]: float(Gx[bi, bj]) for bj in range(nb_)} for bi in range(nb_)},
                'per_block_mirror_target_uS': {names[bi]: {names[bj]: float(Gp[bi, bj]) for bj in range(nb_)} for bi in range(nb_)},
                'per_block_abs_diff_uS': {names[bi]: {names[bj]: float(abs(Gx[bi, bj] - Gp[bi, bj])) for bj in range(nb_)} for bi in range(nb_)}}

    symm, symm_c = {}, {}
    for lab, Gx in (('_We', Gw), ('_Wi', Gi), ('both', G)):
        pr, co = mirror_mats(Gx, mir_idx)
        symm[lab] = asym_block(Gx, pr, BUCKETS)
        symm_c[lab] = asym_block(Gx, co, BUCKETS)

    def cellsym(sizes_map):
        out = {}
        for b in ['pool_lf', 'pool_lh', 'pool_lm', 'pool_shared', 'pool_nonmotor', 'pool_mn']:
            mb = MIRROR[b]
            a, c = sizes_map[b], sizes_map[mb]
            val = (a - c) / (a + c) if (a + c) else None
            out['%s_vs_%s' % (b, mb)] = {'count_a': a, 'count_b': c, 'abs_diff': abs(a - c), 'asym_count': val,
                                         'asymmetry_detected': (abs(val) >= ASYM_TOL) if val is not None else None}
        return out

    leg_cell_sym = {}
    for a, b in (('lf', 'rf'), ('lh', 'rh'), ('lm', 'rm')):
        ca, cb = pools['mn_leg_counts'][a], pools['mn_leg_counts'][b]
        leg_cell_sym['%s_vs_%s' % (a, b)] = {'mn_count_a': ca, 'mn_count_b': cb, 'abs_diff': abs(ca - cb),
                                             'asym_count': (ca - cb) / (ca + cb),
                                             'asymmetry_detected': abs((ca - cb) / (ca + cb)) >= ASYM_TOL}

    pair_sym = {}
    for lab in ('_We', '_Wi', 'both'):
        pair_sym[lab] = {}
        for group, pairs in (('ipsilateral', IPSILATERAL_PAIRS),
                             ('contralateral_same_row', CONTRA_SAME_ROW),
                             ('contralateral_cross_row', CONTRA_CROSS_ROW)):
            for a, b in pairs:
                ma = MIRROR['pool_' + a][len('pool_'):]
                mb_ = MIRROR['pool_' + b][len('pool_'):]
                this = pair_block(a, b, lab)
                x, ex = this['conductance_both_directions_uS'], this['edges_both_directions']
                g_ab, g_ba = this['conductance_a_to_b_uS'], this['conductance_b_to_a_uS']
                self_mirror = (frozenset((ma, mb_)) == frozenset((a, b)))
                entry = {
                    'group': group, 'conductance_both_directions_uS': x,
                    'conductance_a_to_b_uS': g_ab, 'conductance_b_to_a_uS': g_ba,
                    'direction_asym_uS': g_ab - g_ba,
                    'direction_rel_asym': ((g_ab - g_ba) / (g_ab + g_ba)) if (g_ab + g_ba) else None,
                    'edges_both_directions': ex,
                    'mirror_pair': '%s-%s' % (ma, mb_), 'self_mirror_pair': bool(self_mirror)}
                if not self_mirror:
                    other = pair_block(ma, mb_, lab)
                    y, ey = other['conductance_both_directions_uS'], other['edges_both_directions']
                    entry.update({'mirror_conductance_both_directions_uS': y,
                                  'abs_diff_uS': abs(x - y),
                                  'rel_asym': ((x - y) / (x + y)) if (x + y) else None,
                                  'mirror_edges_both_directions': ey,
                                  'edges_abs_diff': abs(ex - ey)})
                else:
                    entry.update({'mirror_conductance_both_directions_uS': None,
                                  'abs_diff_uS': None, 'rel_asym': None,
                                  'mirror_edges_both_directions': None, 'edges_abs_diff': None,
                                  'note': ('this leg pair is its own mirror image, so the unordered-pair mirror test is '
                                           'vacuous; the bilateral asymmetry of this coupling is exactly direction_asym_uS')})
                pair_sym[lab]['%s-%s' % (a, b)] = entry

    n_inflow = np.bincount(post, minlength=N)
    decomp_check = {}
    for bj, bdst in enumerate(BUCKETS):
        n = int((n_inflow[bid == bj] > 0).sum())
        colsum = float(G[:, bj].sum())
        decomp_check[bdst] = {'neurons_with_incoming': n, 'expected_uS': .02 * n,
                              'recomputed_uS': colsum, 'abs_diff_uS': abs(colsum - .02 * n),
                              'passes_1e-12': bool(abs(colsum - .02 * n) <= 1e-12)}

    # restricted index over the 6 leg pools only: the huge self-mirrored nonmotor/shared/mn
    # blocks otherwise dominate the denominator and dilute the asymmetry of the leg blocks
    lp_idx = [BIDX['pool_' + l] for l in LEGS]
    legpool_only = {}
    for lab, Gx in (('_We', Gw), ('_Wi', Gi), ('both', G)):
        sub = Gx[np.ix_(lp_idx, lp_idx)]
        local_mir = np.array([LEGS.index(MIRROR['pool_' + l][len('pool_'):]) for l in LEGS], dtype=np.int64)
        pm, cm = mirror_mats(sub, local_mir)
        names = ['pool_' + l for l in LEGS]
        legpool_only[lab] = {'preregistered_rule': asym_block(sub, pm, names),
                            'corrected_rule': asym_block(sub, cm, names),
                            'bucket_order': names}

    symmetry = {
        'mirror_map': MIRROR,
        'mirror_rule_preregistered': "G'[b_src][b_dst] = G[mir(b_dst)][mir(b_src)]  (mirror AND reverse direction)",
        'mirror_rule_corrected': "G'[b_src][b_dst] = G[mir(b_src)][mir(b_dst)]  (mirror only; an edge A->B mirrors to mir(A)->mir(B))",
        'deviation_note': ('the preregistered rule composes mirroring with a direction reversal; for a block whose '
                           'destination is already the mirror of its source (LF->RF, LM->RM, LH->RH and their reverses) '
                           'it compares the block with itself and can never report asymmetry. See DEVIATIONS.md. '
                           'Both indices are reported; the corrected rule is the primary bilateral-symmetry number.'),
        'asym_tol': ASYM_TOL,
        'asym_index_definition': "sum|G-G'| / sum(G+G') over the whole 9x9 block matrix",
        'conductance_matrix_mirror_preregistered_rule': symm,
        'conductance_matrix_mirror_corrected_rule': symm_c,
        'cell_counts_primary': cellsym(pool_sizes),
        'cell_counts_union_partition': cellsym(pool_sizes_u),
        'mn_leg_counts': leg_cell_sym,
        'leg_pool_only_6x6_mirror': legpool_only,
        'leg_pair_conductance_mirror': pair_sym,
        'incoming_decomposition_integer_check': decomp_check,
    }
    (HERE / 'SYMMETRY.json').write_text(json.dumps(symmetry, indent=2, ensure_ascii=False) + '\n')
    log('symmetry preregistered-rule asym_index We=%.12f Wi=%.12f both=%.12f'
        % (symm['_We']['asym_index_L1'], symm['_Wi']['asym_index_L1'], symm['both']['asym_index_L1']))
    log('symmetry corrected-rule    asym_index We=%.12f Wi=%.12f both=%.12f'
        % (symm_c['_We']['asym_index_L1'], symm_c['_Wi']['asym_index_L1'], symm_c['both']['asym_index_L1']))
    log('symmetry corrected-rule leg-pools-only asym_index We=%.12f Wi=%.12f both=%.12f'
        % (legpool_only['_We']['corrected_rule']['asym_index_L1'], legpool_only['_Wi']['corrected_rule']['asym_index_L1'],
           legpool_only['both']['corrected_rule']['asym_index_L1']))

    # ---------- coupling for the five gait-relevant pairs
    rad = {name: np.bincount(post[bid[pre] == BIDX[name]], minlength=N).astype(np.int64)
           for name in ['pool_' + l for l in LEGS] + ['pool_shared', 'pool_nonmotor', 'pool_mn']}
    fog = {leg: np.bincount(pre[np.isin(post, mn_by_leg[leg])], minlength=N).astype(np.int64) for leg in LEGS}
    notmn = ~is_mn

    def paths2(src_bucket, leg_dst, restrict=None):
        sel = notmn.copy()
        if restrict is not None:
            sel &= (bid == BIDX[restrict])
        prod = rad[src_bucket][sel] * fog[leg_dst][sel]
        return {'paths': int(prod.sum()), 'distinct_intermediates': int((prod > 0).sum())}

    coupling = {
        'levels_documented': ['L0_direct_premotor_interconnect', 'L1_shared_bridge', 'L2_two_hop', 'L3_verdict'],
        'path_lengths_analyzed': [1, 2],
        'path_lengths_not_analyzed': '>=3',
        'definition_of_paths2': 'sum over non-MN x of (#edges into x from the source bucket) * (#edges from x into MN cells of the destination leg)',
        'pairs': {},
    }
    for a, b in GAIT_PAIRS:
        ia, ib = LEG_BIT[a], LEG_BIT[b]
        l0 = {'edges_a_to_b': {lab: pair_block(a, b, lab)['edges_a_to_b'] for lab in ('_We', '_Wi', 'both')},
              'edges_b_to_a': {lab: pair_block(a, b, lab)['edges_b_to_a'] for lab in ('_We', '_Wi', 'both')},
              'conductance_a_to_b_uS': {lab: pair_block(a, b, lab)['conductance_a_to_b_uS'] for lab in ('_We', '_Wi', 'both')},
              'conductance_b_to_a_uS': {lab: pair_block(a, b, lab)['conductance_b_to_a_uS'] for lab in ('_We', '_Wi', 'both')},
              'definitional_note': ('a pool_%s cell has every direct MN target in %s by construction, so a direct '
                                    'pool_%s -> MN_%s edge is impossible by definition, not missing data' % (a, a, a, b))}
        l1_cells = ((mask_direct & ia) != 0) & ((mask_direct & ib) != 0)
        l1_idx = np.flatnonzero(l1_cells)
        l1 = {'cells_targeting_both_legs': int(l1_idx.size)}
        if l1_idx.size:
            in_edge = np.isin(post, l1_idx)
            out_to_mn = np.isin(pre, l1_idx) & is_mn[post]
            l1.update({'their_incoming_edges': int(in_edge.sum()),
                       'their_outgoing_edges_to_mn': int(out_to_mn.sum()),
                       'their_outgoing_conductance_to_mn_uS': float(We[mn_local, :][:, l1_idx].sum()
                                                                    + Wi[mn_local, :][:, l1_idx].sum())})
        else:
            l1.update({'their_incoming_edges': 0, 'their_outgoing_edges_to_mn': 0,
                       'their_outgoing_conductance_to_mn_uS': 0.0})
        l2 = {'a_to_b': paths2('pool_' + a, b), 'b_to_a': paths2('pool_' + b, a),
              'a_to_b_via_shared_intermediate': paths2('pool_' + a, b, 'pool_shared'),
              'b_to_a_via_shared_intermediate': paths2('pool_' + b, a, 'pool_shared')}
        if l0['edges_a_to_b']['both'] + l0['edges_b_to_a']['both'] > 0:
            verdict = 'L0_direct_interconnect_present'
        elif l1['cells_targeting_both_legs'] > 0:
            verdict = 'L1_shared_bridge_present_no_direct_interconnect'
        elif l2['a_to_b']['paths'] + l2['b_to_a']['paths'] > 0:
            verdict = 'L2_two_hop_only_no_direct_no_bridge'
        else:
            verdict = 'ABSENT_no_coupling_within_two_hops_MISSING_DATA'
        coupling['pairs']['%s-%s' % (a, b)] = {
            'L0': l0, 'L1': l1, 'L2': l2, 'L3_verdict': verdict,
            'category': ('contralateral_same_row' if (a, b) in CONTRA_SAME_ROW
                         else 'tripod_partner' if (a, b) in TRIPOD else 'other')}

    allpairs = {}
    for a, b in ALL_PAIRS:
        ia, ib = LEG_BIT[a], LEG_BIT[b]
        cells = ((mask_direct & ia) != 0) & ((mask_direct & ib) != 0)
        allpairs['%s-%s' % (a, b)] = {
            'cells_targeting_both_legs': int(cells.sum()),
            'direct_edges_a_to_b': int(E[leg_i[a], leg_i[b]]),
            'direct_edges_b_to_a': int(E[leg_i[b], leg_i[a]]),
            'paths2_a_to_b': paths2('pool_' + a, b)['paths'],
            'paths2_b_to_a': paths2('pool_' + b, a)['paths'],
        }
    coupling['all_15_pairs_compact'] = allpairs
    coupling['dominant_mode_reference'] = {
        'source': '/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/intrinsic/REPORT.md',
        'source_sha256': 'e356f1a8001cb5f4ed376514414bd6447514f2a9dbf7249907b25fb6fd1ff24d',
        'pc1_variance_fraction': 0.495,
        'pc1_loadings': {'lf': 0.58, 'rf': 0.70, 'lh': -0.31, 'rh': -0.28, 'lm': 0.02, 'rm': 0.05},
        'pearson_lf_rh': 0.036, 'pearson_rf_lh': 0.082,
        'note': 'quoted from the frozen upstream report; not recomputed here (no rate data touched this round)',
    }
    (HERE / 'COUPLING.json').write_text(json.dumps(coupling, indent=2, ensure_ascii=False) + '\n')
    log('coupling classified: ' + json.dumps({k: v['L3_verdict'] for k, v in coupling['pairs'].items()}))

    # ---------- parameterisation proposal (design only)
    named = [tuple(p) for p in CONTRA_SAME_ROW] + [('lf', 'lh'), ('rf', 'rh')] + [tuple(p) for p in TRIPOD]
    classes = {
        'C_contra_same_row': {'scalar': 's_contra_same_row', 'pairs': [tuple(p) for p in CONTRA_SAME_ROW]},
        'C_front_hind': {'scalar': 's_front_hind', 'pairs': [('lf', 'lh'), ('rf', 'rh')]},
        'C_tripod_diagonal': {'scalar': 's_tripod_diagonal', 'pairs': [tuple(p) for p in TRIPOD]},
    }
    cls = {}
    for name, spec in classes.items():
        ent = {'scalar': spec['scalar'], 'leg_pairs': [list(p) for p in spec['pairs']],
               'edges_per_receptor': {}, 'conductance_uS_per_receptor': {}}
        for lab, Ex, Gx in (('_We', Ew, Gw), ('_Wi', Ei, Gi)):
            ent['edges_per_receptor'][lab] = int(sum(Ex[leg_i[a], leg_i[b]] + Ex[leg_i[b], leg_i[a]] for a, b in spec['pairs']))
            ent['conductance_uS_per_receptor'][lab] = float(sum(Gx[leg_i[a], leg_i[b]] + Gx[leg_i[b], leg_i[a]] for a, b in spec['pairs']))
        ent['edges_total'] = int(sum(ent['edges_per_receptor'].values()))
        ent['frozen_conductance_uS_total'] = float(sum(ent['conductance_uS_per_receptor'].values()))
        cls[name] = ent
    fixed_edges = int(E.sum()) - sum(v['edges_total'] for v in cls.values())
    fixed_g = float(G.sum()) - sum(v['frozen_conductance_uS_total'] for v in cls.values())
    cls['C_fixed_everything_else'] = {'scalar': 1.0, 'leg_pairs': 'all remaining edges',
                                      'edges_per_receptor': {'note': 'not split by receptor here'},
                                      'edges_total': fixed_edges, 'frozen_conductance_uS_total': fixed_g}
    if sum(v['edges_total'] for v in cls.values()) != 435126:
        raise SystemExit('ABORT: class partition does not sum to 435126 edges')
    # per-target-neuron share of incoming conductance held by each scalable class.
    # w_from_raw is bit-identical to the frozen matrices (checked above), so it is used here
    # as the per-edge uS value in raw edge order.
    w_edge = w_from_raw
    cls_of_edge = np.zeros(pre.size, dtype=np.int64)  # 0 = fixed, 1..3 = the three scalable classes
    for ci, pairs in enumerate([spec['pairs'] for spec in classes.values()], start=1):
        legset = {(LEGS.index(a), LEGS.index(b)) for a, b in pairs}
        legset |= {(a, b) for (a, b) in [(b, a) for (a, b) in legset]}
        hit = np.zeros(pre.size, dtype=bool)
        for la, lb in legset:
            hit |= (bid[pre] == BIDX['pool_' + LEGS[la]]) & (bid[post] == BIDX['pool_' + LEGS[lb]])
        cls_of_edge[hit] = ci
    if int((cls_of_edge > 0).sum()) != sum(cls[n]['edges_total'] for n in
                                          ['C_contra_same_row', 'C_front_hind', 'C_tripod_diagonal']):
        raise SystemExit('ABORT: per-edge class assignment disagrees with the block class totals')
    row_tot_all = np.bincount(post, weights=w_edge, minlength=N)
    share_stats = {}
    for ci, name in enumerate(['C_contra_same_row', 'C_front_hind', 'C_tripod_diagonal'], start=1):
        cg = np.bincount(post[cls_of_edge == ci], weights=w_edge[cls_of_edge == ci], minlength=N)
        live_rows = cg > 0
        sh = np.zeros(N)
        sh[live_rows] = cg[live_rows] / row_tot_all[live_rows]
        share_stats[name] = {'target_neurons_affected': int(live_rows.sum()),
                             'target_neurons_with_share_gt_0p5': int((sh > 0.5).sum()),
                             'max_share': float(sh.max()) if sh.size else 0.0,
                             'mean_share_over_affected': float(sh[live_rows].mean()) if live_rows.any() else 0.0}
    L, U = 0.25, 4.0
    ceiling = {}
    for name in ['C_contra_same_row', 'C_front_hind', 'C_tripod_diagonal']:
        g0 = cls[name]['frozen_conductance_uS_total']
        ceiling[name] = {
            'frozen_uS': g0,
            'at_scalar_lower_uS': g0 * L, 'at_scalar_upper_uS': g0 * U,
            'delta_at_upper_uS': g0 * (U - 1.0),
            'delta_at_upper_share_of_module_total': g0 * (U - 1.0) / T,
            'ratio_to_contralateral_same_row_class_at_upper':
                (g0 * U) / cls['C_contra_same_row']['frozen_conductance_uS_total'],
        }
    prop = {
        'proposal_id': 'leg-pools-scaling-v1',
        'status': 'DESIGN ONLY - nothing simulated, not validated by any number in this round',
        'scope': 'multiplicative rescaling of EXISTING edge conductances inside the strict subgraph',
        'rule': "w'[e] = s[class(e)] * w[e] with s > 0; no entry created or removed",
        'free_scalars': ['s_contra_same_row', 's_front_hind', 's_tripod_diagonal'],
        'free_scalar_count': 3,
        'bounds': {'lower': 0.25, 'upper': 4.0, 'inclusive': True,
                   'suggested_grid': [0.25, 0.5, 1.0, 2.0, 4.0], 'grid_combinations': 125},
        'class_partition': cls,
        'class_partition_sums_to_435126_edges': True,
        'application': ('row scaling of the frozen csr matrices: for every stored entry (post, pre) multiply '
                        'the uS value in place by the scalar of class(pre, post); row index = postsynaptic neuron'),
        'conductance_convention': {
            'frozen': 'every neuron with incoming edges has exactly 0.02 uS total incoming conductance',
            'V1_row_renormalized': {
                'rule': 'after scaling, divide each target neuron row by (scaled row sum / 0.02 uS) so the row total is exactly 0.02 uS',
                'resulting_per_neuron_incoming_uS': 0.02,
                'rows_untouched': 'neurons with zero incoming edges have no entries and are unaffected'},
            'V2_bounded_unrenormalized': {
                'rule': 'no renormalisation; each target neuron total must stay in [0.5, 2.0] * 0.02 uS',
                'allowed_per_neuron_incoming_uS': [0.01, 0.04],
                'projection': 'if a row total exceeds a bound, multiply that whole row by one per-row factor to land on the bound',
                'units': 'uS'},
            'must_choose_one_next_round': True,
        },
        'scalar_ceiling_if_applied_exactly': ceiling,
        'per_neuron_class_share': share_stats,
        'per_neuron_bound_consequence': {
            'worst_case_row_total_at_all_scalars_upper_uS': U * 0.02,
            'V1_keeps_exact_uS': 0.02,
            'V2_would_project_rows_above_uS': 2.0 * 0.02,
            'note': 'V2 projection is active for any row whose scalable-class share exceeds 1/3 at s=4; V1 is the recommended variant because it keeps the frozen .02 uS convention exactly and only changes relative class weights',
        },
        'feasibility_flags_from_this_audit': {
            'class_has_edges': {k: v['edges_total'] > 0 for k, v in cls.items()},
            'class_edge_totals': {k: v['edges_total'] for k, v in cls.items()},
            'note': ('the scalars only change the strength of edges that already exist; a class with zero direct '
                     'edges cannot be up-weighted by this proposal, and the two-hop / shared-pool routes reported in '
                     'COUPLING.json are NOT reachable by any scalar of this class partition'),
        },
        'not_allowed': ['new edges', 'deleted edges', 'transmitter reclassification', 'sign flips',
                        'external oscillator or phase clock', 'acceptance-criteria changes',
                        'reading body or leg state into the premotor block'],
        'next_round_requirements': ('new timestamped directory, own preregistration, open-loop neural screen before '
                                    'any physics, one scalar set preregistered rather than searched post hoc'),
    }
    (HERE / 'PROPOSED_SCALING_PARAMS.json').write_text(json.dumps(prop, indent=2, ensure_ascii=False) + '\n')
    log('PROPOSED_SCALING_PARAMS.json written (3 free scalars)')

    result = {
        'round': 'banc-descending-learning/20261107T203049Z/leg-pools',
        'task': 'offline strict-subgraph leg-pool structural audit',
        'steps': {'neural_steps': 0, 'physical_steps': 0, 'learning_updates': 0, 'new_edges': 0,
                  'renders': 0, 'forward_calls': 0, 'analyses': 1},
        'preregistration': {'PREREGISTRATION.md': sha256(HERE / 'PREREGISTRATION.md'),
                            'PREREGISTRATION.json': sha256(HERE / 'PREREGISTRATION.json')},
        'analysis_code': {'leg_pools.py': sha256(HERE / 'leg_pools.py')},
        'source_hashes': source_hashes,
        'weight_contract': weight_contract,
        'pool_sizes': pool_sizes,
        'pool_sizes_union_partition': pool_sizes_u,
        'mn_leg_counts': pools['mn_leg_counts'],
        'blocks_edges_both_receptors': {BUCKETS[i]: {BUCKETS[j]: int(E[i, j]) for j in range(NB)} for i in range(NB)},
        'blocks_conductance_uS_both_receptors': {BUCKETS[i]: {BUCKETS[j]: float(G[i, j]) for j in range(NB)} for i in range(NB)},
        'blocks_edges_We': {BUCKETS[i]: {BUCKETS[j]: int(Ew[i, j]) for j in range(NB)} for i in range(NB)},
        'blocks_edges_Wi': {BUCKETS[i]: {BUCKETS[j]: int(Ei[i, j]) for j in range(NB)} for i in range(NB)},
        'blocks_conductance_uS_We': {BUCKETS[i]: {BUCKETS[j]: float(Gw[i, j]) for j in range(NB)} for i in range(NB)},
        'blocks_conductance_uS_Wi': {BUCKETS[i]: {BUCKETS[j]: float(Gi[i, j]) for j in range(NB)} for i in range(NB)},
        'aggregates_summary': {k: (v['both'] if 'both' in v else {kk: vv['both'] for kk, vv in v.items()})
                               for k, v in aggregate.items()},
        'symmetry_headline': {
            'asym_index_L1_We': symm['_We']['asym_index_L1'],
            'asym_index_L1_Wi': symm['_Wi']['asym_index_L1'],
            'asym_index_L1_both': symm['both']['asym_index_L1'],
            'asym_index_L1_We_corrected_rule': symm_c['_We']['asym_index_L1'],
            'asym_index_L1_Wi_corrected_rule': symm_c['_Wi']['asym_index_L1'],
            'asym_index_L1_both_corrected_rule': symm_c['both']['asym_index_L1'],
            'abs_diff_sum_uS_both_corrected_rule': symm_c['both']['abs_diff_sum_uS'],
            'sum_uS_both_corrected_rule': symm_c['both']['sum_uS'],
            'asym_index_L1_legpools_only_corrected_We': legpool_only['_We']['corrected_rule']['asym_index_L1'],
            'asym_index_L1_legpools_only_corrected_Wi': legpool_only['_Wi']['corrected_rule']['asym_index_L1'],
            'asym_index_L1_legpools_only_corrected_both': legpool_only['both']['corrected_rule']['asym_index_L1'],
            'cell_count_asymmetry_primary': cellsym(pool_sizes),
            'mn_leg_counts_asymmetry': leg_cell_sym,
        },
        'coupling_headline': {k: {'L3_verdict': v['L3_verdict'],
                                  'L0_edges_both': v['L0']['edges_a_to_b']['both'] + v['L0']['edges_b_to_a']['both'],
                                  'L1_cells': v['L1']['cells_targeting_both_legs'],
                                  'L2_paths_a_to_b': v['L2']['a_to_b']['paths'],
                                  'L2_paths_b_to_a': v['L2']['b_to_a']['paths']}
                              for k, v in coupling['pairs'].items()},
        'proposal': {'path': 'PROPOSED_SCALING_PARAMS.json', 'free_scalar_count': prop['free_scalar_count'],
                     'bounds': prop['bounds']},
        'environment': {'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                        'python': sys.executable, 'dash_B': True,
                        'numpy': np.__version__, 'scipy': scipy.__version__},
        'files_written': ['POOLS.json', 'POOLS_AND_REACH.npz', 'BLOCKS.json', 'SYMMETRY.json',
                          'COUPLING.json', 'PROPOSED_SCALING_PARAMS.json', 'RESULT.json'],
    }
    (HERE / 'RESULT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    log('RESULT.json written')
    return result


if __name__ == '__main__':
    main()
