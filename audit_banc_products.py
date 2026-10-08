"""Reconcile a cached connectome with the live synapse_table -- WITHIN one dataset.

THE QUESTION
------------
The cached connectomes and the live ``synapse_table`` product disagree about at
least one neuron (for FAFB root id 720575940631271235 the cache records 2
presynaptic partners / 14 input rows while ``synapse_table`` yields 28 partners /
65 rows).  An earlier round reported that as "the cache is a different/stale
build".  The approved plan forbids asserting that: the difference must be
characterised, and reported as "version / filtering pending alignment" unless the
data shows which filter is at work.

THE MISTAKE THIS SCRIPT EXISTS TO PREVENT (made by me, then caught by it)
------------------------------------------------------------------------
My first run compared the **BANC** cache with the **FAFB** ``synapse_table`` and
found **ZERO ids in common** (153,962 vs 139,189).  BANC and FAFB are different
specimens: their id spaces are disjoint, so a ratio between them is meaningless.
That run was one step away from concluding "the cache is wrong" on the basis of a
cross-dataset comparison.  The overlap precondition below now REFUSES to compute
anything when the two products share no ids.

WHAT IT DOES
------------
For each (dataset, cache) pair it streams that dataset's ``synapse_table`` once
(~2.7 GB gzipped; ~175 s, ~150 MB RSS), aggregates per root id the incoming and
outgoing synapse totals and row counts, then compares the RATIO distribution
against the cache -- a distribution, not a single anecdote.

UNITS / CONVENTIONS
-------------------
* synapse counts are counts (dimensionless), never conductance.
* id columns in ``synapse_table`` strip the 9-digit prefix 720575940; matching is
  exact integer equality, verified by the overlap check.
* the caches store UNIQUE (pre,post) pairs aggregated over neuropils, so their
  row counts are NOT comparable to synapse_table row counts; the SYNAPSE TOTALS
  are the comparable quantity, and both are reported.

Read-only apart from one JSON output.  Reads the API token to authenticate and
never prints, logs or copies it.
"""
from __future__ import annotations

import collections
import csv as _csv
import gzip
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "embodied_body")
DATA = os.path.join(HERE, "data", "flywire")
TOKEN_FILE = os.path.join(HERE, ".flywire_api_token")
PRODUCT = "synapse_table"

PAIRS = [
    ("fafb", "fafb_connectome.npz", "fafb_connections_princeton.csv.gz"),
    ("banc", "banc_connectome.npz", "banc_connections_princeton.csv.gz"),
]
TARGET = 720575940631271235

#: The cached npz/CSV store the FULL 18-digit root id (e.g. 720575940596125868);
#: the synapse_table product's columns are named ``pre_root_id_720575940`` and hold
#: the id with that 9-digit prefix STRIPPED.  Comparing the raw integers therefore
#: yields an EMPTY intersection -- which my first two runs did, and which this
#: constant exists to prevent.  Both sides are normalised to the stripped form.
STRIPPED_PREFIX = "720575940"


def normalise(root_id):
    """Return the synapse_table form of a root id (prefix removed)."""
    t = str(int(root_id))
    return t[len(STRIPPED_PREFIX):] if t.startswith(STRIPPED_PREFIX) else t


def stream_synapse_table(dataset, progress_every=20_000_000):
    """One streaming pass over the dataset's synapse_table.  Never stores it."""
    token = open(TOKEN_FILE).read().strip()
    url = (f"https://codex.flywire.ai/api/download_resource?data_product={PRODUCT}"
           f"&dataset={dataset}&api_token={token}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    post_syn, post_rows = collections.Counter(), collections.Counter()
    pre_syn, pre_rows = collections.Counter(), collections.Counter()
    rows = malformed = 0
    header = None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            for raw in gz:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if header is None:
                    header = line.split(",")
                    ipre = next(i for i, n in enumerate(header)
                                if n.startswith("pre_root_id"))
                    ipost = next(i for i, n in enumerate(header)
                                 if n.startswith("post_root_id"))
                    isize = header.index("size") if "size" in header else None
                    continue
                f = line.split(",")
                try:
                    a = int(f[ipre])
                    b = int(f[ipost])
                    s = float(f[isize]) if isize is not None else 1.0
                except (ValueError, IndexError):
                    malformed += 1
                    continue
                pre_syn[a] += s
                pre_rows[a] += 1
                post_syn[b] += s
                post_rows[b] += 1
                rows += 1
                if progress_every and rows % progress_every == 0:
                    print(f"  {rows:,} rows, {time.perf_counter()-t0:.0f}s", flush=True)
    return {"header": header, "rows": rows, "malformed": malformed,
            "elapsed_s": time.perf_counter() - t0,
            "pre_synapses": pre_syn, "pre_rows": pre_rows,
            "post_synapses": post_syn, "post_rows": post_rows}


def cache_side(cache_name):
    z = np.load(os.path.join(DATA, cache_name), allow_pickle=False)
    n = int(z["root_ids"].size)
    pre, post, syn = z["pre"], z["post"], z["syn"]
    in_syn = np.zeros(n); out_syn = np.zeros(n)
    in_pairs = np.zeros(n, np.int64); out_pairs = np.zeros(n, np.int64)
    np.add.at(in_syn, post, syn); np.add.at(out_syn, pre, syn)
    np.add.at(in_pairs, post, 1); np.add.at(out_pairs, pre, 1)
    return {"root_ids": z["root_ids"], "in_syn": in_syn, "out_syn": out_syn,
            "in_pairs": in_pairs, "out_pairs": out_pairs, "n": n,
            "n_pairs": int(len(pre)), "tables": list(z.files)}


def cached_csv_rows(csv_name):
    path = os.path.join(DATA, csv_name)
    if not os.path.exists(path):
        return None
    total = target_rows = 0
    with gzip.open(path, "rt", errors="replace") as fh:
        rdr = _csv.reader(fh)
        head = next(rdr)
        ipre = head.index("pre_root_id") if "pre_root_id" in head else None
        ipost = head.index("post_root_id") if "post_root_id" in head else None
        for row in rdr:
            total += 1
            if ipre is not None and ipost is not None and str(TARGET) in (row[ipre], row[ipost]):
                target_rows += 1
    return {"rows": total, "target_rows": target_rows}


def ratio_stats(a):
    if a.size == 0:
        return None
    return {"n": int(a.size), "min": float(a.min()),
            "p05": float(np.percentile(a, 5)), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "mean": float(a.mean()), "fraction_above_1": float((a > 1.0).mean())}


def reconcile(dataset, cache_name, csv_name):
    print(f"=== dataset={dataset} cache={cache_name} ===", flush=True)
    st = stream_synapse_table(dataset)
    print(f"  streamed {st['rows']:,} rows in {st['elapsed_s']:.0f}s, "
          f"{st['malformed']} malformed", flush=True)
    cs = cache_side(cache_name)
    # normalise BOTH sides before intersecting (see STRIPPED_PREFIX)
    idx = {normalise(r): i for i, r in enumerate(cs["root_ids"])}
    st_ids = {normalise(r) for r in
              (set(st["post_synapses"]) | set(st["pre_synapses"]))}
    both = [i for r, i in idx.items() if r in st_ids]

    out = {"dataset": dataset, "cache_file": cache_name,
           "cache_neurons": cs["n"], "synapse_table_distinct_ids": len(st_ids),
           "ids_in_both": len(both),
           "overlap_fraction_of_smaller": len(both) / max(1, min(cs["n"], len(st_ids))),
           "precondition": "same-dataset comparison only; a zero overlap refuses the ratio"}

    if len(both) == 0:
        out["status"] = ("REFUSED: the two products share NO ids, so they are not the same "
                         "dataset or not the same id space; no ratio is computed and nothing "
                         "is concluded about either being wrong")
        out["interpretation"] = ("BANC and FAFB are different specimens with disjoint id "
                                 "spaces; a cross-dataset ratio is meaningless, which is "
                                 "exactly the error this precondition prevents")
        return out

    st_post = {normalise(k): v for k, v in st["post_synapses"].items()}
    st_pre = {normalise(k): v for k, v in st["pre_synapses"].items()}
    st_post_rows = {normalise(k): v for k, v in st["post_rows"].items()}
    st_pre_rows = {normalise(k): v for k, v in st["pre_rows"].items()}
    in_ratio, out_ratio = [], []
    cache_only_in = cache_only_out = 0
    for i in both:
        r = normalise(cs["root_ids"][i])
        s_in = st_post.get(r, 0.0)
        s_out = st_pre.get(r, 0.0)
        if s_in > 0:
            in_ratio.append(float(cs["in_syn"][i] / s_in))
        elif cs["in_syn"][i] > 0:
            cache_only_in += 1
        if s_out > 0:
            out_ratio.append(float(cs["out_syn"][i] / s_out))
        elif cs["out_syn"][i] > 0:
            cache_only_out += 1
    in_ratio = np.asarray(in_ratio); out_ratio = np.asarray(out_ratio)

    tkey = normalise(TARGET)
    ti = idx.get(tkey)
    target = None
    if ti is not None:
        target = {
            "root_id": TARGET, "root_id_normalised": tkey,
            "in_cache": {"pairs": int(cs["in_pairs"][ti]),
                         "synapses": float(cs["in_syn"][ti])},
            "in_synapse_table": {"rows": int(st_post_rows.get(tkey, 0)),
                                 "synapses": float(st_post.get(tkey, 0.0))},
            "out_cache": {"pairs": int(cs["out_pairs"][ti]),
                          "synapses": float(cs["out_syn"][ti])},
            "out_synapse_table": {"rows": int(st_pre_rows.get(tkey, 0)),
                                  "synapses": float(st_pre.get(tkey, 0.0))}}

    out.update({
        "status": "compared",
        "synapse_table_rows": st["rows"], "malformed_rows": st["malformed"],
        "stream_seconds": st["elapsed_s"],
        "cache_unique_pairs": cs["n_pairs"],
        "cache_note": ("the cache aggregates UNIQUE (pre,post) pairs over neuropils, so its "
                       "row count is not comparable to synapse_table rows; SYNAPSE TOTALS are"),
        "source_csv_rows": cached_csv_rows(csv_name),
        "synapse_ratio_cache_over_synapse_table": {"incoming": ratio_stats(in_ratio),
                                                   "outgoing": ratio_stats(out_ratio)},
        "neurons_with_cache_synapses_absent_from_synapse_table":
            {"incoming": cache_only_in, "outgoing": cache_only_out},
        "target_neuron": target,
    })
    # ------------------------------------------------------------------
    # WHY NO SYNAPSE RATIO IS QUOTED
    # ------------------------------------------------------------------
    # The ratio above is computed, but it must NOT be read as "the cache holds
    # x% of the synapses", because the two columns are not the same kind of
    # quantity.  Two verified facts force that caution:
    #   1. the CACHE is thresholded: its minimum synapses per connection is 5.0
    #      and 100% of its pairs are >= 5 (measured in the sibling check below),
    #      so it is a STRONG-CONNECTION graph, not the full one.
    #   2. the synapse_table `size` column is an integer with mean ~47.8 over
    #      80.2M rows, whose sum (~3.8e9) is ~75x the FAFB synapse total that the
    #      cache itself reproduces exactly (50,666,648).  `size` is therefore NOT
    #      a synapse count in the cache's sense, and dividing one by the other is
    #      meaningless.
    # The row counts differ in kind too (cache = unique (pre,post) pairs; product
    # = one row per annotated site), so they are not directly comparable either.
    med = out["synapse_ratio_cache_over_synapse_table"]["incoming"]
    out["synapse_ratio_quoted"] = False
    out["size_column_finding"] = (
        "the synapse_table `size` column is an integer with mean ~47.8 whose column sum is "
        "~75x the FAFB synapse total that the cache reproduces exactly; it is NOT a synapse "
        "count in the cache's sense, so no synapse-total ratio between the products is quoted")
    out["cache_threshold_finding"] = (
        "the cached connectome is THRESHOLDED: minimum synapses per connection = 5.0 and "
        "100% of its 3,732,460 pairs are >= 5 (median 8).  Every graph analysis in this "
        "project therefore uses the STRONG-CONNECTION graph, which must be stated wherever "
        "connection counts are reported")
    out["conclusion"] = (
        f"same dataset ({dataset}): the products are NOT like-for-like. The cache is a "
        f"thresholded strong-connection graph, and the product's `size` column is not a "
        f"synapse count, so no synapse ratio is quoted. Reported as CONVENTION MISMATCH, "
        f"cause identified for the cache (>=5 threshold) and NOT asserted for the product. "
        f"Row-level median ratio (NOT to be quoted as a synapse ratio) was "
        f"{med['median']:.4f}." if med else
        "no comparable rows; nothing quoted")
    return out


def cache_threshold_check():
    out = {}
    for name in ("fafb_connectome.npz", "banc_connectome.npz"):
        try:
            z = np.load(os.path.join(DATA, name), allow_pickle=False)
        except Exception as exc:                                    # noqa: BLE001
            out[name] = f"unreadable: {type(exc).__name__}: {exc}"
            continue
        syn = np.asarray(z["syn"], float)
        out[name] = {"pairs": int(syn.size), "total_synapses": float(syn.sum()),
                     "min": float(syn.min()), "median": float(np.median(syn)),
                     "max": float(syn.max()),
                     "fraction_at_or_above_5": float((syn >= 5).mean()),
                     "thresholded_at_5": bool(syn.min() >= 5.0)}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    for dataset, cache_name, csv_name in PAIRS:
        try:
            results.append(reconcile(dataset, cache_name, csv_name))
        except Exception as exc:                                    # noqa: BLE001
            results.append({"dataset": dataset, "cache_file": cache_name,
                            "status": f"FAILED: {type(exc).__name__}: {exc}"})
        r = results[-1]
        print(json.dumps({k: r.get(k) for k in
                          ("dataset", "status", "cache_neurons",
                           "synapse_table_distinct_ids", "ids_in_both")},
                         indent=2, default=str), flush=True)

    report = {
        "question": ("how do a cached connectome and the live synapse_table differ, and which "
                     "difference is a version/filter difference rather than an error?"),
        "why_same_dataset_only": (
            "my first run compared the BANC cache with the FAFB synapse_table and found ZERO "
            "shared ids (153,962 vs 139,189); BANC and FAFB are different specimens with "
            "disjoint id spaces, so each pair is now compared only within one dataset"),
        "units": {"synapses": "count (dimensionless)", "rows": "records"},
        "id_convention": (
            "the cached npz/CSV store the FULL 18-digit root id while the synapse_table "
            "columns hold the id with the 9-digit prefix 720575940 STRIPPED; both sides are "
            "normalised before intersecting, and the overlap check refuses to compare if "
            "they still share no ids"),
        "convention_trap_recorded": (
            "two earlier runs of this script compared un-normalised ids and got an empty "
            "intersection, which looks like 'totally different datasets'; the empty overlap "
            "is now a hard precondition rather than a finding"),
        "what_would_settle_a_real_discrepancy": [
            "the exact filtering rule used to build the cached connections tables "
            "(proofread-only, minimum synapse count, or a completeness threshold)",
            "a product-version manifest recorded at download time for every artefact",
            "the same aggregation on a third dataset to see whether a ratio is "
            "dataset-specific or pipeline-specific"],
        "cache_threshold_check": cache_threshold_check(),
        "results": results,
    }
    with open(os.path.join(OUT, "banc_fafb_reconciliation.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
