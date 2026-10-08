"""build_banc_cns.py -- the COMPLETE Drosophila CNS: brain + ventral nerve cord.

WHY BANC RATHER THAN FAFB + MANC
--------------------------------
FAFB is the brain only (139,255 neurons).  MANC is the nerve cord only
(23,665).  They are different specimens, so joining them requires solving the
FlyWire "VNC Matching Challenge" -- a nontrivial cell-matching problem.
BANC is a single dataset covering brain AND nerve cord in one consistent
proofreading frame: 158,262 neurons.  So for a whole-CNS model, BANC is the
right single source, and no cross-specimen matching is needed.

BANC also ships a richer `neurons` table than FAFB: it carries Flow, Super
Class, Class, Sub Class, Nerve, Body Part, Function and cell types together,
so the sensorimotor boundary can be read from one file.

Same trap as FAFB applies and is handled: `connections_princeton` rows are
(pair x neuropil), so rows must be aggregated to unique pairs before building
the CSR, otherwise duplicate entries get silently summed by scipy.
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

EDGES = os.path.join(DATA, "banc_connections_princeton.csv.gz")
NEURONS = os.path.join(DATA, "banc_neurons.csv.gz")
CACHE = os.path.join(DATA, "banc_connectome.npz")

NT_ORDER = ["ACH", "DA", "GABA", "GLUT", "OCT", "SER"]
NT_SIGN = {"ACH": +1, "DA": +1, "GABA": -1, "GLUT": -1, "OCT": +1, "SER": +1}
NT_ASSUMED = {"DA", "SER", "OCT"}


def build(force=False):
    if os.path.exists(CACHE) and not force:
        print(f"  loaded cache {CACHE}")
        return np.load(CACHE, allow_pickle=False)

    t0 = time.time()
    pre_l, post_l, syn_l, nt_l = [], [], [], []
    with gzip.open(EDGES, "rt", errors="replace") as fh:
        r = csv.reader(fh)
        head = next(r)
        for row in r:
            pre_l.append(int(row[0])); post_l.append(int(row[1]))
            syn_l.append(int(row[3])); nt_l.append(row[4])
    pre = np.asarray(pre_l, np.int64); post = np.asarray(post_l, np.int64)
    syn = np.asarray(syn_l, np.int32)
    print(f"  parsed {pre.size:,} rows in {time.time()-t0:.0f}s")
    del pre_l, post_l, syn_l

    lut = {l: i for i, l in enumerate(NT_ORDER)}
    nt_code = np.array([lut.get(x, -1) for x in nt_l], dtype=np.int8)
    del nt_l

    root_ids = np.union1d(np.unique(pre), np.unique(post))
    n = root_ids.size
    a_all = np.searchsorted(root_ids, pre); b_all = np.searchsorted(root_ids, post)
    key = a_all * n + b_all
    uniq_key, inv = np.unique(key, return_inverse=True)
    syn_total = np.bincount(inv, weights=syn.astype(np.float64))
    a_idx = (uniq_key // n).astype(np.int64); b_idx = (uniq_key % n).astype(np.int64)
    n_pairs = uniq_key.size
    print(f"  神经元 {n:,}  唯一(前,后)对 {n_pairs:,}  "
          f"（原始行 {pre.size:,}，每对 {pre.size/n_pairs:.2f} 个神经毡）")

    # nt per presynaptic neuron (first row of each group)
    order = np.argsort(a_all, kind="stable")
    a_s = a_all[order]; c_s = nt_code[order]
    starts = np.flatnonzero(np.r_[True, a_s[1:] != a_s[:-1]])
    nt_of_neuron = np.full(n, -1, dtype=np.int8)
    nt_of_neuron[np.unique(a_all)] = c_s[starts]
    nt_of_pair = nt_of_neuron[a_idx]

    # ---- neurons table -> flow / super class / class per root id ----
    ann = {}
    fields = ("flow", "super_class", "class", "nerve", "body_part", "function",
              "primary_type", "nt_verified")
    keymap = {"Flow": "flow", "Super Class": "super_class", "Class": "class",
              "Nerve": "nerve", "Body Part": "body_part", "Function": "function",
              "Primary Cell Type": "primary_type",
              "Verified NT type": "nt_verified"}
    n_rows = 0
    if os.path.exists(NEURONS):
        with gzip.open(NEURONS, "rt", errors="replace") as fh:
            for row in csv.DictReader(fh):
                n_rows += 1
                rid = int(row["Root ID"])
                ann[rid] = {v: (row.get(k, "") or "") for k, v in keymap.items()}
    idx_of = {int(r): i for i, r in enumerate(root_ids)}
    arr = {f: np.empty(n, dtype=object) for f in fields}
    for f in fields:
        arr[f][:] = ""
    matched = 0
    for rid, d in ann.items():
        i = idx_of.get(rid)
        if i is None:
            continue
        matched += 1
        for f in fields:
            arr[f][i] = d[f]
    for f in fields:
        arr[f] = arr[f].astype(str)

    np.savez_compressed(
        CACHE, pre=a_idx, post=b_idx, syn=syn_total.astype(np.float32),
        nt_pair=nt_of_pair, nt_neuron=nt_of_neuron, root_ids=root_ids,
        n_rows=np.array([pre.size]), n_pairs=np.array([n_pairs]),
        **{f"ann_{f}": arr[f] for f in fields})

    from collections import Counter
    flow_c = Counter(arr["flow"].tolist())
    sc_c = Counter(arr["super_class"].tolist())
    sen_c = Counter(arr["class"][arr["super_class"] == "sensory"].tolist())
    res = {
        "dataset": "BANC (Female Adult Fly Brain and Nerve Cord)",
        "raw_rows": int(pre.size), "unique_pairs": int(n_pairs), "neurons": int(n),
        "neurons_table_rows": n_rows, "neurons_matched_to_connectome": matched,
        "total_synapses": float(syn_total.sum()),
        "flow": dict(flow_c.most_common()),
        "super_class": dict(sc_c.most_common(20)),
        "sensory_class": dict(sen_c.most_common(20)),
        "nt_edges": {NT_ORDER[i]: int((nt_of_pair == i).sum()) for i in range(len(NT_ORDER))},
        "note": "BANC covers brain AND nerve cord in one frame, so no "
                "cross-specimen VNC matching is needed.",
    }
    with open(os.path.join(OUT, "metrics_banc_cns.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in
                      ("raw_rows", "unique_pairs", "neurons", "total_synapses",
                       "flow", "sensory_class")}, indent=1, ensure_ascii=False)[:1600])
    return np.load(CACHE, allow_pickle=False)


if __name__ == "__main__":
    build(force="--force" in sys.argv)
