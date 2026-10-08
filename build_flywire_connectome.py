"""build_flywire_connectome.py -- turn the FlyWire CSV into a signed weight matrix.

WHAT THE DATA IS
----------------
`connections_princeton` (FAFB v783) has columns
    pre_root_id, post_root_id, neuropil, syn_count, nt_type
and 5,342,446 rows.  That is MORE than the 3,732,460 "connections" quoted on the
Codex download page, because A PAIR OF NEURONS CAN APPEAR IN SEVERAL NEUROPILS.
So the rows are (pair x neuropil) and must be aggregated to unique pairs before
building a matrix; forgetting this inflates the synapse totals and creates
duplicate CSR entries that scipy silently SUMS (giving wrong weights).

`nt_type` is the neurotransmitter of the PRESYNAPTIC neuron.  Drosophila's main
excitatory transmitter is acetylcholine and GABA/glutamate are inhibitory, so
the sign of every edge comes from the data rather than from a modelling choice.
The remaining types in this snapshot are modulators (dopamine, serotonin,
octopamine); treating those as excitatory is an ASSUMPTION and their count is
reported so it cannot be forgotten.

Outputs
-------
data/flywire/fafb_connectome.npz   cached arrays (pre_idx, post_idx, syn_count,
                                   nt_code, root_ids)
outputs/metrics_flywire_connectome.json
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "flywire")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

EDGES = os.path.join(DATA, "fafb_connections_princeton.csv.gz")
TYPES = os.path.join(DATA, "fafb_consolidated_cell_types.csv.gz")
CACHE = os.path.join(DATA, "fafb_connectome.npz")

# Sign policy, stated explicitly.  ACh is the main fast EXCITATORY transmitter
# in Drosophila; GABA and glutamate are INHIBITORY.  DA / SER / OCT are
# neuromodulators: their fast effect is not a simple sign, so 'excitatory' is an
# assumption, and we record how many edges it affects.
NT_SIGN = {"ACH": +1, "GABA": -1, "GLUT": -1,
           "DA": +1, "SER": +1, "OCT": +1}
NT_ASSUMED = {"DA", "SER", "OCT"}


def build(force=False):
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=False)
        print(f"  loaded cache: {CACHE}")
        return z

    t0 = time.time()
    pre_l, post_l, syn_l, nt_l = [], [], [], []
    with gzip.open(EDGES, "rt", errors="replace") as fh:
        r = csv.reader(fh)
        head = next(r)
        expect = ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"]
        if head[:5] != expect:
            raise SystemExit(f"unexpected header: {head}")
        for i, row in enumerate(r):
            pre_l.append(int(row[0]))
            post_l.append(int(row[1]))
            syn_l.append(int(row[3]))
            nt_l.append(row[4])
    n_rows = len(pre_l)
    print(f"  parsed {n_rows:,} rows in {time.time()-t0:.0f}s")

    pre = np.asarray(pre_l, dtype=np.int64)
    post = np.asarray(post_l, dtype=np.int64)
    syn = np.asarray(syn_l, dtype=np.int32)
    nt = np.asarray(nt_l, dtype=object)
    del pre_l, post_l, syn_l, nt_l

    # ---- is nt_type really constant per presynaptic neuron?  verify, don't assume
    uniq_pre, inv = np.unique(pre, return_inverse=True)
    nt_code_all, nt_labels = _encode(nt)
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    code_s = nt_code_all[order].astype(np.int32)
    starts = np.flatnonzero(np.r_[True, inv_s[1:] != inv_s[:-1]])
    cmax = np.maximum.reduceat(code_s, starts)
    cmin = np.minimum.reduceat(code_s, starts)
    multi = int((cmax != cmin).sum())
    print(f"  突触前神经元数: {uniq_pre.size:,}; "
          f"nt_type 在同一突触前神经元内不一致的个数: {multi}")

    # ---- aggregate (pre, post) over neuropil: rows are (pair x neuropil), and
    # ---- forgetting this would both inflate synapse totals and create
    # ---- duplicate CSR entries that scipy silently sums.
    uniq_post, inv_post = np.unique(post, return_inverse=True)
    root_ids = np.union1d(uniq_pre, uniq_post)
    n = root_ids.size
    a_idx_all = np.searchsorted(root_ids, pre)
    b_idx_all = np.searchsorted(root_ids, post)
    key = a_idx_all * n + b_idx_all
    uniq_key, inv_key = np.unique(key, return_inverse=True)
    syn_total = np.bincount(inv_key, weights=syn.astype(np.float64))
    a_idx = (uniq_key // n).astype(np.int64)
    b_idx = (uniq_key % n).astype(np.int64)
    n_pairs = uniq_key.size
    n_rows = pre.size
    print(f"  唯一神经元数: {n:,}  唯一(前,后)对: {n_pairs:,}  "
          f"（原始行 {n_rows:,}，平均每对 {n_rows/n_pairs:.2f} 个神经毡）")

    # ---- nt per unique presynaptic neuron (first row of each group)
    nt_of_neuron = np.zeros(n, dtype=np.int8)
    nt_of_neuron[np.searchsorted(root_ids, uniq_pre)] = code_s[starts].astype(np.int8)
    nt_of_pair = nt_of_neuron[a_idx]

    # ---- cell types
    types = {}
    n_types = 0
    if os.path.exists(TYPES):
        with gzip.open(TYPES, "rt", errors="replace") as fh:
            r = csv.reader(fh)
            head = next(r)
            for row in r:
                try:
                    types[int(row[0])] = row[1]
                except Exception:
                    pass
                n_types += 1

    np.savez_compressed(
        CACHE, pre=a_idx, post=b_idx, syn=syn_total.astype(np.float32),
        nt_pair=nt_of_pair, nt_neuron=nt_of_neuron, root_ids=root_ids,
        nt_labels=np.array(nt_labels, dtype=object),
        n_rows=np.array([n_rows]), n_pairs=np.array([n_pairs]))
    print(f"  cached -> {CACHE} ({os.path.getsize(CACHE)/1e6:.1f} MB)")

    sign = np.array([NT_SIGN.get(l, 1) for l in nt_labels], dtype=np.int8)
    counts = {l: int((nt_of_pair == i).sum()) for i, l in enumerate(nt_labels)}
    assumed_edges = int(sum(counts.get(l, 0) for l in NT_ASSUMED))
    assumed_syn = float(sum(syn_total[nt_of_pair == i].sum()
                            for i, l in enumerate(nt_labels)
                            if l in NT_ASSUMED))
    res = {
        "dataset": "FlyWire FAFB v783",
        "raw_rows": int(n_rows), "unique_pairs": int(n_pairs),
        "neurons_in_connections": int(n),
        "neurons_with_cell_types": int(len(types)),
        "rows_per_pair_mean": float(n_rows / n_pairs),
        "nt_types": nt_labels,
        "edges_by_presynaptic_nt": counts,
        "edges_with_assumed_excitatory_sign": assumed_edges,
        "synapses_with_assumed_excitatory_sign": assumed_syn,
        "total_synapses": float(syn_total.sum()),
        "nt_inconsistent_presynaptic_neurons": int(multi),
        "sign_policy": {k: v for k, v in NT_SIGN.items()},
        "assumed_modulators_treated_as_excitatory": sorted(NT_ASSUMED),
        "self_loops": int((a_idx == b_idx).sum()),
        "note": "nt_type is the presynaptic neuron's transmitter; the sign of "
                "each edge therefore comes from data except for the modulators "
                "listed in assumed_modulators_treated_as_excitatory.",
    }
    with open(os.path.join(OUT, "metrics_flywire_connectome.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in
                      ("raw_rows", "unique_pairs", "neurons_in_connections",
                       "edges_by_presynaptic_nt", "total_synapses",
                       "self_loops", "assumed_modulators_treated_as_excitatory")},
                     indent=1, ensure_ascii=False))
    return np.load(CACHE, allow_pickle=False)


def _encode(nt):
    labels = sorted(set(nt.tolist()))
    lut = {l: i for i, l in enumerate(labels)}
    code = np.array([lut[x] for x in nt.tolist()], dtype=np.int8)
    return code, labels


if __name__ == "__main__":
    build(force="--force" in sys.argv)
