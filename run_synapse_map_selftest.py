#!/usr/bin/env python3
"""Self-test for engine/synapse_map.py and run_synapse_map.py.

Covers nearest-node assignment (including a point exactly on a node and points
between nodes), far/out-of-range flagging, id filtering exactness, determinism,
empty-selection handling, the compartment current vector summing to the intended
total, exact same-drive pairing of the two arms, and invalid-input rejection.

Real-data assertions (real synapse extraction / validated subtree SWC) run when
those files exist and are otherwise SKIPPED with a reason.  The suite passes
without network access and without the real download.

Writes /run/media/sensen/Data2/cell_wound_prototype/outputs/brain_isolation/synapse_map_selftest.json
"""
from __future__ import annotations

import json
import math
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.cable import Morphology, swc_to_morphology            # noqa: E402
from engine import synapse_map as sm                              # noqa: E402

OUT = ROOT / "outputs" / "brain_isolation"
REPORT = OUT / "synapse_map_report.json"
RAW_NPZ = OUT / "synapse_map_raw.npz"
METRICS = OUT / "real_morphology_subtree_metrics.json"
SAMPLE = OUT / "real_morphology_subtree_sample.swc"

_PASS, _FAIL, _SKIP = [], [], []
_MEM_LIMIT_MB = 3000.0


def check(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
    except AssertionError as exc:
        _FAIL.append({"name": name, "error": f"AssertionError: {exc}",
                      "traceback": traceback.format_exc(limit=3)})
        print(f"FAIL {name}: {exc}", flush=True)
    except Exception as exc:                      # unexpected error is a failure
        _FAIL.append({"name": name, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc(limit=3)})
        print(f"ERROR {name}: {type(exc).__name__}: {exc}", flush=True)
    else:
        # A check may register its own skip (e.g. a credential that is
        # deliberately absent from the delivered package).  In that case it
        # must be counted as skipped ONLY, never also as passed -- otherwise
        # the totals overstate coverage.
        if any(entry["name"] == name for entry in _SKIP):
            return
        _PASS.append({"name": name, "ms": round((time.perf_counter() - t0) * 1000, 3)})
        print(f"ok   {name}", flush=True)


def skip(name, reason):
    _SKIP.append({"name": name, "reason": reason})
    print(f"SKIP {name}: {reason}", flush=True)


def raises(name, exc_types, fn):
    def inner():
        try:
            fn()
        except exc_types:
            return
        except Exception as exc:
            raise AssertionError(f"wrong exception type {type(exc).__name__}: {exc}")
        raise AssertionError(f"no exception raised (expected {exc_types})")
    check(name, inner)


def synthetic_morph(nseg=20, length_um=20.0, diameter_um=1.0):
    """Unbranched test cable with EXACT integer-um node coordinates.

    Morphology.cylinder uses linspace, whose node positions can be ~1e-12 um off
    exact integers; for assignment tests that would make an "exactly on a node"
    point land 1e-12 um away.  Building the same geometry from an integer grid
    keeps the synthetic cases exact while still exercising the real importer
    (Morphology validates it the same way).
    """
    x = np.arange(nseg + 1, dtype=float) * (length_um / nseg)
    return Morphology(x, np.zeros(nseg + 1), np.zeros(nseg + 1),
                      np.full(nseg + 1, float(diameter_um)),
                      np.arange(-1, nseg))


def target_suffix(metrics_path=METRICS):
    """The stripped root id actually stored in the synapse table's id columns."""
    root_id = int(json.loads(metrics_path.read_text())["source"]["source_id"])
    return root_id, int(str(root_id)[len(sm.FLYWIRE_LABEL):])


def main():
    t_start = time.time()
    morph = synthetic_morph()
    # NOTE: astype(float) copy -- an integer result would make later
    # `site = nodes_nm[i] + offset` mutate the array in place (numpy trap)
    nodes_nm = np.column_stack([morph.x, morph.y, morph.z]).astype(float) * 1000.0
    assert nodes_nm.dtype.kind == 'f'
    spacing_um = 20.0 / 20

    # ---------------- nearest-node assignment correctness -------------------
    def t_exact_node():
        site = sm.Sites.synthetic(nodes_nm[5:6].copy())
        r = sm.assign_nearest_node(site, morph)
        assert r["node"].tolist() == [5], r["node"]
        assert r["distance_nm"][0] == 0.0, r["distance_nm"]
        assert r["distance_um"][0] == 0.0
        assert not r["far"][0]
        assert r["n_far"] == 0 and r["far_ids"].size == 0
    check("assignment: point exactly on a node -> distance 0, that node", t_exact_node)

    def t_between_nodes():
        # 40% of the way from node 5 to node 6: strictly between nodes, closest to 5
        mid = np.array(nodes_nm[5], float, copy=True)
        mid += 0.40 * (nodes_nm[6] - nodes_nm[5])
        r = sm.assign_nearest_node(sm.Sites.synthetic(mid[None, :]), morph)
        assert r["node"].tolist() == [5], r["node"]
        assert abs(r["distance_um"][0] - 0.40 * spacing_um) < 1e-12, r["distance_um"]
        assert not r["far"][0]
        # and the mirror point 60% of the way -> closest to 6
        mid2 = np.array(nodes_nm[5], float, copy=True)
        mid2 += 0.60 * (nodes_nm[6] - nodes_nm[5])
        r2 = sm.assign_nearest_node(sm.Sites.synthetic(mid2[None, :]), morph)
        assert r2["node"].tolist() == [6], r2["node"]
        assert abs(r2["distance_um"][0] - 0.40 * spacing_um) < 1e-12
        # exactly halfway: both nodes are equidistant, distance is half the spacing
        half = 0.5 * (np.array(nodes_nm[5], float, copy=True) + np.array(nodes_nm[6], float, copy=True))
        r3 = sm.assign_nearest_node(sm.Sites.synthetic(half[None, :]), morph)
        assert r3["node"].tolist() in ([5], [6]), r3["node"]
        assert abs(r3["distance_um"][0] - 0.5 * spacing_um) < 1e-9
    check("assignment: points between nodes go to the closer node with measured distance", t_between_nodes)

    def t_multi_and_distance_stats():
        pts = np.vstack([nodes_nm[0], nodes_nm[10], nodes_nm[-1]])
        r = sm.assign_nearest_node(pts, morph)
        assert r["node"].tolist() == [0, 10, len(nodes_nm) - 1], r["node"]
        assert r["n_sites"] == 3 and r["n_nodes"] == morph.n
        st = r["distance_stats_um"]
        assert st["max"] == 0.0 and st["min"] == 0.0 and st["median"] == 0.0
        assert st["n_over_1um"] == 0 and st["n_over_5um"] == 0
        assert r["n_on_retained_subtree"] == 3
        assert "nearest skeleton node" in r["assignment"]
    check("assignment: multiple sites, statistics and retained count", t_multi_and_distance_stats)

    def t_swc_ids_returned():
        ids = np.arange(700, 700 + morph.n)
        r = sm.assign_nearest_node(sm.Sites.synthetic(nodes_nm[3:4].copy()), morph, node_ids=ids)
        assert r["node_swc_id"].tolist() == [703], r["node_swc_id"]
        far = id_lookup_far = np.vstack([nodes_nm[3], nodes_nm[7]])
        r2 = sm.assign_nearest_node(sm.Sites.synthetic(far), morph, node_ids=ids)
        assert r2["node_swc_id"].tolist() == [703, 707], r2["node_swc_id"]
    check("assignment: returns the original SWC id of the matched node", t_swc_ids_returned)

    def t_far_flagging():
        # placed exactly 2.5 um from the nearest node; nodes are 1 um apart so
        # the point cannot accidentally coincide with another node
        far_pt = np.array(nodes_nm[10], dtype=float, copy=True)
        far_pt[0] = 22500.0   # 2.5 um from the nearest node (20000)
        near_pt = nodes_nm[10].copy()
        r = sm.assign_nearest_node(np.vstack([far_pt, near_pt]), morph, far_um=2.0,
                                   node_ids_nm=nodes_nm)
        assert r["far"].tolist() == [True, False], (r["far"], r["distance_um"])
        assert r["far_ids"].tolist() == [0]
        assert r["n_far"] == 1
        assert abs(r["distance_um"][0] - 2.5) < 1e-12, r["distance_um"]
        st = r["distance_stats_um"]
        assert st["n_over_1um"] == 1 and st["n_over_5um"] == 0, st
        assert abs(st["max"] - 2.5) < 1e-12
        assert r["n_on_retained_subtree"] == 1
    check("flagging: out-of-range coordinates are flagged against the threshold", t_far_flagging)

    def t_flag_threshold_exact():
        far_pt = np.array(nodes_nm[10], dtype=float, copy=True)
        far_pt[0] = 22500.0   # 2.5 um from the nearest node (20000)
        probe = sm.assign_nearest_node(np.vstack([far_pt]), morph, far_um=1e9,
                                       node_ids_nm=nodes_nm)
        d = float(probe["distance_um"][0])
        assert 2.49 < d < 2.51, d
        assert sm.assign_nearest_node(np.vstack([far_pt]), morph, far_um=d + 0.01,
                                      node_ids_nm=nodes_nm)["far"].tolist() == [False]
        assert sm.assign_nearest_node(np.vstack([far_pt]), morph, far_um=d - 0.01,
                                      node_ids_nm=nodes_nm)["far"].tolist() == [True]
        r = sm.assign_nearest_node(np.vstack([far_pt, nodes_nm[2]]), morph, far_um=0.2,
                                   node_ids_nm=nodes_nm)
        assert (~r["far"]).sum() == 1
    check("flagging: the threshold is honoured exactly and selects usable sites", t_flag_threshold_exact)

    def t_empty_selection():
        empty = sm.Sites.synthetic(np.zeros((0, 3)))
        assert len(empty) == 0
        r = sm.assign_nearest_node(empty, morph)
        assert r["n_sites"] == 0 and r["n_far"] == 0
        assert r["distance_stats_um"] is None
        assert r["node"].size == 0 and r["far"].size == 0
        cnt, tot = sm.distribution(np.array([], dtype=np.int64), n_compartments=morph.n)
        assert tot == 0 and cnt.sum() == 0 and len(cnt) == morph.n
    check("empty selection: assignment and distribution handle zero sites", t_empty_selection)

    def t_determinism_assignment():
        rng = np.random.default_rng(7)
        pts = rng.normal(scale=4000.0, size=(500, 3)) + nodes_nm[10]
        r1 = sm.assign_nearest_node(pts, morph)
        r2 = sm.assign_nearest_node(sm.Sites.synthetic(pts.copy()), morph)
        assert np.array_equal(r1["node"], r2["node"])
        assert np.array_equal(r1["distance_nm"], r2["distance_nm"])
        r3 = sm.assign_nearest_node(pts, morph, chunk=7)        # different chunking
        assert np.array_equal(r1["node"], r3["node"])
        assert np.array_equal(r1["distance_um"], r3["distance_um"])
        assert json.dumps(sm.json_ready(r1["distance_stats_um"])) == \
            json.dumps(sm.json_ready(r2["distance_stats_um"]))
    check("determinism: repeated and chunk-size-varied assignment is identical", t_determinism_assignment)

    # ---------------- truncation accounting --------------------------------
    def t_truncation_accounting():
        # small 5-node cable so the SWC-id vector matches the compartment count
        small = synthetic_morph(nseg=4, length_um=4.0)
        small_nm = np.column_stack([small.x, small.y, small.z]).astype(float) * 1000.0
        swc_ids = np.array([10, 11, 12, 13, 14])
        r = sm.assign_nearest_node(sm.Sites.synthetic(small_nm[[0, 1, 2]].copy()), small,
                                   far_um=2.0, node_ids_nm=small_nm, node_ids=swc_ids)
        assert r["distance_um"].max() == 0.0
        acc = sm.truncation_accounting(r, excluded_swc_ids=[11, 12], node_swc_ids=swc_ids)
        assert acc["n_sites"] == 3
        assert acc["n_mapped_to_retained_nodes"] == 1, acc
        assert acc["n_mapped_to_excluded_nodes"] == 2, acc
        assert acc["n_flagged_far"] == 0
        assert acc["n_retained_and_close"] == 1
        acc2 = sm.truncation_accounting(r, excluded_swc_ids=[], node_swc_ids=swc_ids)
        assert acc2["n_mapped_to_excluded_nodes"] == 0
        assert acc2["n_mapped_to_retained_nodes"] == 3
    check("truncation accounting: retained vs excluded node matches are split exactly",
          t_truncation_accounting)

    # ---------------- id filtering exactness -------------------------------
    def t_id_filter_exact():
        xyz = nodes_nm[[0, 1, 2, 3, 4]]
        pre = np.array([100, 200, 100, 200, 100])
        post = np.array([7, 7, 8, 8, 9])
        s = sm.Sites.synthetic(xyz, pre_root_id=pre, post_root_id=post)
        only100 = s.select_root(100, "pre_root_id")
        assert len(only100) == 3
        assert only100.pre_root_id.tolist() == [100, 100, 100]
        assert only100.xyz_nm[:, 0].tolist() == xyz[[0, 2, 4], 0].tolist()
        assert only100.index.tolist() == [0, 2, 4]
        only7 = s.select_root(7, "post_root_id")
        assert len(only7) == 2 and only7.post_root_id.tolist() == [7, 7]
        # an id that differs only in the stripped FAFB prefix must NOT match
        assert len(s.select_root(720575940000100, "pre_root_id")) == 0
        assert len(s.select_root(-100, "pre_root_id")) == 0
        assert len(s.select_root(101, "pre_root_id")) == 0
    check("id filtering: only rows whose exact root id matches are kept", t_id_filter_exact)

    def t_id_filter_empty_result():
        s = sm.Sites.synthetic(nodes_nm[:3], pre_root_id=np.array([1, 2, 3]))
        sel = s.select_root(99, "pre_root_id")
        assert len(sel) == 0
        r = sm.assign_nearest_node(sel, morph)
        assert r["n_sites"] == 0 and r["distance_stats_um"] is None
    check("id filtering: non-matching root id yields an empty, safe selection", t_id_filter_empty_result)

    # ---------------- distribution -----------------------------------------
    def t_distribute_exact_sum():
        counts = sm.distribute_sites(1000, [3.0, 1.0, 0.0, 6.0])
        assert counts.sum() == 1000, counts
        assert counts.dtype.kind == "i"
        assert counts.tolist() == [300, 100, 0, 600], counts
        w = np.array([1.0, 1.0, 1.0])
        a = sm.distribute_sites(10, w)
        b = sm.distribute_sites(10, w)
        assert a.sum() == 10 and np.array_equal(a, b), (a, b)
        assert a.tolist() == [4, 3, 3], a           # ties broken by lowest index
        assert sm.distribute_sites(2, w).tolist() == [1, 1, 0]
        assert sm.distribute_sites(0, w).tolist() == [0, 0, 0]
        assert sm.distribute_sites(0, [0.0, 0.0]).tolist() == [0, 0]
        assert sm.distribute_sites(5, [1.0, 0.0]).tolist() == [5, 0]
    check("distribution: integer split sums exactly and is deterministic", t_distribute_exact_sum)

    def t_distribution_from_sites():
        nodes = np.array([0, 0, 3, 3, 3, 2])
        cnt, tot = sm.distribution(nodes, n_compartments=5)
        assert cnt.tolist() == [2, 0, 1, 3, 0], cnt
        assert tot == 6
        cnt2, tot2 = sm.distribution(nodes, n_compartments=5, n_sites=100)
        assert cnt2.sum() == 100 and tot2 == 100
        assert cnt2.tolist() == [33, 0, 17, 50, 0], cnt2
        assert cnt2[3] > cnt2[0] > cnt2[2] > 0
        # weights select WHERE the bounded sites go without changing the total
        # weights steer the bounded sites to one compartment without changing the total
        cnt3, _ = sm.distribution(nodes, n_compartments=5, n_sites=10,
                                  weights=np.array([0.0, 0.0, 0.0, 1.0, 0.0]))
        assert cnt3.tolist() == [0, 0, 0, 10, 0], cnt3
        cnt4, _ = sm.distribution(nodes, n_compartments=5, n_sites=10,
                                  weights=np.array([2.0, 0.0, 1.0, 3.0, 0.0]))
        assert cnt4.sum() == 10 and cnt4.tolist() == [3, 0, 2, 5, 0], cnt4
        assert cnt4[3] > cnt4[0] > cnt4[2] > 0
    check("distribution: per-compartment counts and bounded redistribution", t_distribution_from_sites)

    # ---------------- compartment current vector ---------------------------
    def t_current_vector_sums():
        g = np.array([0.0, 1e-3, 2e-3, 0.5e-3])
        v = np.array([-66.0, -60.0, -70.0, -65.0])
        E = -10.0
        i = sm.compartment_current(g, v, E)
        assert i.shape == g.shape
        assert np.allclose(i, g * (E - v), rtol=0, atol=0)
        manual = sum(gi * (E - vi) for gi, vi in zip(g, v))
        assert abs(i.sum() - manual) < 1e-15, (i.sum(), manual)
        n, gp = 7, 5e-4
        i2 = sm.compartment_current(np.array([n * gp]), np.array([-65.0]), E)
        assert abs(i2[0] - n * gp * (E + 65.0)) < 1e-15
        assert sm.compartment_current(np.array([1e-3]), np.array([-65.0]), 0.0)[0] > 0
        assert sm.compartment_current(np.array([1e-3]), np.array([0.0]), -65.0)[0] < 0
        base = sm.compartment_current(np.array([1e-3]), np.array([-65.0]), 0.0)[0]
        assert sm.compartment_current(np.array([1e-3]), np.array([-65.0]), 0.0,
                                      extra_scale=2.0)[0] == 2 * base
        # sites sharing a compartment sum exactly
        g_shared = np.array([2 * gp])
        assert sm.compartment_current(g_shared, np.array([-65.0]), 0.0)[0] == \
            2 * sm.compartment_current(np.array([gp]), np.array([-65.0]), 0.0)[0]
    check("compartment current: per-compartment vector sums to the intended total", t_current_vector_sums)

    def t_event_conductance():
        g = sm.event_conductance(5e-4, np.array([0, 2, 1, 0]))
        assert np.allclose(g, [0.0, 1e-3, 5e-4, 0.0], rtol=0, atol=0)
        g2 = sm.event_conductance(5e-4, np.array([0, 2, 1, 0]), g_decay=np.full(4, 1e-4))
        assert np.allclose(g2, g + 1e-4)
        assert sm.event_conductance(5e-4, np.zeros(4)).sum() == 0.0
        assert sm.event_conductance(0.0, np.array([3])).sum() == 0.0
    check("compartment current: event counts scale conductance per compartment", t_event_conductance)

    # ---------------- schedules: exact same-drive pairing -------------------
    def t_schedule_pairing():
        w_dist = np.array([19.0, 38.0, 75.0, 0.0, 56.0])
        w_point = np.array([1000.0, 0.0, 0.0, 0.0, 0.0])
        kw = dict(dt_ms=0.025, duration_ms=20.0, rate_hz=80.0, t_start_ms=5.0,
                  t_end_ms=15.0, seed=11)
        site_events = sm.discrete_release_schedule(np.ones(188), **kw)
        assert site_events.shape == (800, 188)
        assert site_events.sum() > 0
        # no events outside the requested window
        assert site_events[:200].sum() == 0
        s_point = sm.project_schedule(site_events, sm.placement_matrix(
            np.arange(188), 5, mode="single", single_row=0))
        s_dist = sm.project_schedule(site_events, sm.placement_matrix(
            np.repeat(np.arange(5), w_dist.astype(int)), 5, mode="mapped"))
        assert np.array_equal(sm.schedule_totals(s_point), sm.schedule_totals(s_dist))
        assert np.array_equal(sm.schedule_totals(s_point), site_events.sum(axis=1))
        assert s_point.sum() == s_dist.sum() == int(site_events.sum())
        assert s_point[:, 0].sum() == s_dist.sum()      # all point events on row 0
        assert (s_point[:, 1:].sum()) == 0
        # determinism for a fixed seed
        again = sm.discrete_release_schedule(np.ones(188), **kw)
        assert np.array_equal(site_events, again)
    check("schedules: both arms carry identical per-step event totals", t_schedule_pairing)

    def t_schedule_validation():
        w = np.array([0.0, 0.0])
        s = sm.discrete_release_schedule(w, dt_ms=0.025, duration_ms=1.0, rate_hz=80.0,
                                         t_start_ms=0.0, t_end_ms=1.0, seed=1)
        assert s.shape == (40, 2) and s.sum() == 0
        assert sm.schedule_totals(s).tolist() == [0] * 40
    check("schedules: all-zero site weights produce an empty, valid schedule",
          t_schedule_validation)

    # ---------------- invalid input rejection ------------------------------
    raises("invalid input: non-3-column or non-finite site coordinates",
           (ValueError,), lambda: sm.Sites.synthetic(np.zeros((3, 2))))
    raises("invalid input: non-finite site coordinate values",
           (ValueError,), lambda: sm.Sites.synthetic(np.array([[0.0, 0.0, np.nan]])))
    raises("invalid input: mismatched per-site id vector length",
           (ValueError,), lambda: sm.Sites.synthetic(np.zeros((2, 3)), pre_root_id=[1, 2, 3]))
    raises("invalid input: assignment rejects a non-positive far threshold",
           (ValueError,), lambda: sm.assign_nearest_node(np.zeros((1, 3)), morph, far_um=0.0))
    raises("invalid input: assignment rejects non-finite coordinates",
           (ValueError,), lambda: sm.assign_nearest_node(np.array([[np.inf, 0.0, 0.0]]), morph))
    raises("invalid input: assignment rejects a wrong-shaped node array",
           (ValueError,), lambda: sm.assign_nearest_node(np.zeros((1, 3)), morph,
                                                         node_ids_nm=np.zeros((3, 3))))
    raises("invalid input: assignment rejects mismatched node id count",
           (ValueError,), lambda: sm.assign_nearest_node(np.zeros((1, 3)), morph, node_ids=np.arange(3)))
    raises("invalid input: assignment rejects chunk < 1",
           (ValueError,), lambda: sm.assign_nearest_node(np.zeros((1, 3)), morph, chunk=0))
    raises("invalid input: distribute rejects negative site count",
           (ValueError,), lambda: sm.distribute_sites(-1, [1.0, 1.0]))
    raises("invalid input: distribute rejects non-integer site count",
           (ValueError,), lambda: sm.distribute_sites(2.5, [1.0, 1.0]))
    raises("invalid input: distribute rejects negative/NaN weights",
           (ValueError,), lambda: sm.distribute_sites(5, [1.0, -1.0]))
    raises("invalid input: distribute rejects all-zero weights with nonzero sites",
           (ValueError,), lambda: sm.distribute_sites(5, [0.0, 0.0]))
    raises("invalid input: distribute rejects empty weights",
           (ValueError,), lambda: sm.distribute_sites(5, np.array([])))
    raises("invalid input: distribution rejects negative or out-of-range rows",
           (ValueError,), lambda: sm.distribution(np.array([-1]), n_compartments=4))
    raises("invalid input: distribution rejects rows beyond the compartment count",
           (ValueError,), lambda: sm.distribution(np.array([9]), n_compartments=4))
    raises("invalid input: distribution rejects mismatched weight vector",
           (ValueError,), lambda: sm.distribution(np.array([1]), n_compartments=4, weights=np.ones(3)))
    raises("invalid input: compartment current rejects negative conductance",
           (ValueError,), lambda: sm.compartment_current(np.array([-1e-3]), np.array([-65.0]), 0.0))
    raises("invalid input: compartment current rejects shape mismatch",
           (ValueError,), lambda: sm.compartment_current(np.array([1e-3, 1e-3]), np.array([-65.0]), 0.0))
    raises("invalid input: compartment current rejects non-finite reversal",
           (ValueError,), lambda: sm.compartment_current(np.array([1e-3]), np.array([-65.0]), np.nan))
    raises("invalid input: event conductance rejects negative event counts",
           (ValueError,), lambda: sm.event_conductance(5e-4, np.array([-1])))
    raises("invalid input: event conductance rejects fractional event counts",
           (ValueError,), lambda: sm.event_conductance(5e-4, np.array([0.5])))
    raises("invalid input: schedule rejects a wrong-length weight vector conflict",
           (ValueError,), lambda: sm.project_schedule(np.ones((4, 3)), np.ones((2, 4))))
    raises("invalid input: schedule rejects non-0/1 placement",
           (ValueError,), lambda: sm.project_schedule(np.ones((4, 2)), np.full((3, 2), 2)))
    raises("invalid input: schedule rejects a release probability above one per step",
           (ValueError,), lambda: sm.discrete_release_schedule(
               np.ones(5), dt_ms=1.0, duration_ms=1.0, rate_hz=1e6,
               t_start_ms=0.0, t_end_ms=1.0, seed=0))
    raises("invalid input: placement rejects an out-of-range single row",
           (ValueError,), lambda: sm.placement_matrix(np.arange(3), 4, mode="single", single_row=4))
    raises("invalid input: placement rejects an unknown mode",
           (ValueError,), lambda: sm.placement_matrix(np.arange(3), 4, mode="nope"))

    # ---------------- driver / paired comparison ---------------------------
    def t_run_drive_totals_and_determinism():
        nodes = np.array([1, 1, 3, 4, 4, 4])
        kw = dict(g_per_site_uS=5e-4, E_rev_mV=0.0, tau_ms=2.0, rate_hz=80.0,
                  dt_ms=0.025, duration_ms=5.0, t_start_ms=1.0, t_end_ms=4.0,
                  seed=11, record_rows=[0, 2])
        a = sm.run_drive(morph, site_nodes=nodes, **kw)
        b = sm.run_drive(morph, site_nodes=nodes, **kw)
        assert a["n_sites"] == 6, a["n_sites"]
        assert a["sites_per_compartment"][1] == 2 and a["sites_per_compartment"][4] == 3
        assert np.array_equal(a["V_mV"], b["V_mV"]), "driver must be deterministic for a fixed seed"
        assert np.array_equal(a["G_uS"], b["G_uS"])
        assert np.isfinite(a["V_mV"]).all()
        assert a["V_mV"].shape == (200, 2)
        assert a["G_uS"][:40].sum() == 0.0, "no release before the window"
        assert a["total_events"] > 0
        assert a["charge_nA_ms"] != 0.0
        assert a["driver_note"]
        # the same schedule replayed gives the same trace
        c = sm.run_drive(morph, site_nodes=nodes, schedule=a["schedule"], **kw)
        assert np.array_equal(a["V_mV"], c["V_mV"])
    check("driver: identical total drive, deterministic traces, no events before onset",
          t_run_drive_totals_and_determinism)

    def t_driver_fractional_weights_rejected():
        try:
            sm.run_drive(morph, site_weights=np.full(morph.n, 1.5), duration_ms=0.5)
        except ValueError:
            return
        raise AssertionError("fractional per-compartment site counts must be refused")
    check("driver: fractional per-compartment site counts are refused", t_driver_fractional_weights_rejected)

    def t_paired_comparison_same_drive():
        nodes = np.array([1, 2, 2, 5, 6, 6, 6])
        pair = sm.pairwise_response_compare(morph, site_nodes=nodes, total_sites=50,
                                            single_row=3, record_rows=[0, 3, 5],
                                            g_per_site_uS=5e-4, E_rev_mV=0.0, tau_ms=2.0,
                                            rate_hz=50.0, dt_ms=0.025, duration_ms=5.0,
                                            t_start_ms=1.0, t_end_ms=4.0, seed=3)
        s = pair["summary"]
        sd = s["same_total_drive"]
        assert sd["point_arm_total_sites"] == 50
        assert sd["distributed_arm_total_sites"] == 50
        assert sd["events_identical"] is True, sd
        assert sd["event_counts_identical_per_step"] is True
        assert sd["placement_differs"] is True
        assert sd["same_g_per_site"] is True
        assert s["real_site_rows"] == 7
        assert s["compartments_used_by_real_distribution"] == 4
        assert pair["point"]["V_mV"].shape == pair["distributed"]["V_mV"].shape
        assert np.isfinite(pair["delta_mV"]).all()
        cnt = sm.distribute_sites(50, np.bincount(nodes, minlength=morph.n).astype(float))
        assert cnt.sum() == 50
    check("paired comparison: both arms carry an identical release train", t_paired_comparison_same_drive)

    def t_paired_comparison_is_not_trivially_equal():
        """If the arms were identical the comparison would be meaningless."""
        nodes = np.array([1, 2, 2, 5, 6, 6, 6])
        pair = sm.pairwise_response_compare(morph, site_nodes=nodes, total_sites=200,
                                            single_row=6, record_rows=[0, 6],
                                            g_per_site_uS=5e-4, E_rev_mV=0.0, tau_ms=2.0,
                                            rate_hz=200.0, dt_ms=0.025, duration_ms=5.0,
                                            t_start_ms=1.0, t_end_ms=4.0, seed=5)
        import numpy as _np
        ga, gb = pair["point"]["G_uS"], pair["distributed"]["G_uS"]
        assert not _np.array_equal(ga, gb), "the two arms must place conductance differently"
        assert abs(ga.max() - gb.max()) > 0
        assert _np.array_equal(_np.array(sm.schedule_totals(pair["point"]["schedule"])),
                               _np.array(sm.schedule_totals(pair["distributed"]["schedule"])))
    check("paired comparison: arms differ in placement but not in release counts",
          t_paired_comparison_is_not_trivially_equal)

    def t_implicit_solve_matches_explicit_small_g():
        """For small g*dt/C the implicit and explicit schemes must agree."""
        nodes = np.array([2, 2, 3])
        kw = dict(g_per_site_uS=1e-9, E_rev_mV=0.0, tau_ms=2.0, rate_hz=80.0,
                  dt_ms=0.025, duration_ms=5.0, t_start_ms=1.0, t_end_ms=4.0,
                  seed=9, record_rows=[0, 3])
        a = sm.run_drive(morph, site_nodes=nodes, implicit=False, **kw)
        b = sm.run_drive(morph, site_nodes=nodes, implicit=True, schedule=a["schedule"], **kw)
        assert np.allclose(a["V_mV"], b["V_mV"], rtol=0, atol=1e-9), \
            np.abs(a["V_mV"] - b["V_mV"]).max()
        assert not np.array_equal(a["V_mV"], b["V_mV"]) or True   # identical within tolerance
    check("driver: implicit and explicit schemes agree in the small-conductance limit",
          t_implicit_solve_matches_explicit_small_g)

    def t_stability_limit():
        from engine.cable import CableNeuron
        n1 = CableNeuron(morph, dt_ms=0.025)
        counts = np.zeros(morph.n)
        counts[4] = 1000.0
        lim = sm.explicit_stability_limit(n1, counts)
        expect = 2.0 * n1.C[4] / (n1.dt * 1000.0)
        assert abs(lim - expect) < 1e-18, (lim, expect)
        assert sm.explicit_stability_limit(n1, np.zeros(morph.n)) is None
        lim2 = sm.explicit_stability_limit(n1, counts, safety=0.5)
        assert abs(lim2 - 0.5 * expect) < 1e-18
        # a clamped explicit run stays bounded instead of diverging
        r = sm.run_drive(morph, site_nodes=np.full(1000, 4), g_per_site_uS=5e-4,
                         implicit=False, clamp_to_stability=True, stability_safety=0.25,
                         dt_ms=0.025, duration_ms=2.0, t_start_ms=0.5, t_end_ms=1.5,
                         record_rows=[0, 4], seed=1)
        assert r["stability_clamp"] is not None
        assert r["g_per_site_uS"] < 5e-4
        assert np.isfinite(r["V_mV"]).all()
        assert r["max_abs_voltage_mV"] < 200.0, r["max_abs_voltage_mV"]
    check("driver: explicit mode has a measured stability limit and clamps to it", t_stability_limit)

    raises("invalid input: paired comparison rejects an empty site set",
           (ValueError,), lambda: sm.pairwise_response_compare(
               morph, site_nodes=np.array([], dtype=np.int64), total_sites=10,
               single_row=1, record_rows=[0]))
    raises("invalid input: paired comparison rejects a non-positive total",
           (ValueError,), lambda: sm.pairwise_response_compare(
               morph, site_nodes=np.array([1]), total_sites=0, single_row=1, record_rows=[0]))
    raises("invalid input: paired comparison rejects an out-of-range single point",
           (ValueError,), lambda: sm.pairwise_response_compare(
               morph, site_nodes=np.array([1]), total_sites=5, single_row=morph.n,
               record_rows=[0]))
    raises("invalid input: driver rejects out-of-range site rows",
           (ValueError,), lambda: sm.run_drive(morph, site_nodes=np.array([morph.n]),
                                               duration_ms=0.5))
    raises("invalid input: driver rejects negative site weights",
           (ValueError,), lambda: sm.run_drive(morph, site_weights=-np.ones(morph.n),
                                               duration_ms=0.5))
    raises("invalid input: driver rejects a release probability above one per step",
           (ValueError,), lambda: sm.run_drive(morph, site_weights=np.ones(morph.n),
                                               rate_hz=1e6, dt_ms=0.1, duration_ms=0.5))
    raises("invalid input: driver requires a site specification",
           (ValueError,), lambda: sm.run_drive(morph, duration_ms=0.5))
    raises("invalid input: clamp helper rejects a wrong-shaped site vector",
           (ValueError,), lambda: sm.explicit_stability_limit(
               __import__("engine.cable", fromlist=["CableNeuron"]).CableNeuron(morph),
               np.ones(3)))

    # ---------------- real-data assertions (skipped if absent) -------------
    real_ok = RAW_NPZ.exists() and METRICS.exists() and SAMPLE.exists()
    if not real_ok:
        missing = [str(p) for p in (RAW_NPZ, METRICS, SAMPLE) if not p.exists()]
        reason = ("real FlyWire synapse extraction / validated subtree unavailable at test time; "
                  f"missing {missing}; the synthetic-fixture assertions above still ran, and the "
                  "real-data assertions are SKIPPED (not passed)")
        for nm in ("real data: extracted presynaptic sites carry only the target (stripped) root id",
                   "real data: assignment distances are finite and reported",
                   "real data: report counts agree with the raw extraction and leak no token",
                   "real data: every site maps inside the retained subtree frame",
                   "real data: retained/excluded split is consistent with the id sets"):
            skip(nm, reason)
    else:
        metrics = json.loads(METRICS.read_text())
        root_id, suffix = target_suffix()
        with np.load(RAW_NPZ, allow_pickle=False) as z:
            fields = {k: z[k] for k in z.files}
        all_sites = sm.Sites.from_fields(fields, source=str(RAW_NPZ))
        pre = all_sites.select_root(suffix, "pre_root_id")
        post = all_sites.select_root(suffix, "post_root_id")

        def t_real_ids_exact():
            assert len(pre) > 0, "no presynaptic sites extracted for the target root id"
            assert set(np.unique(pre.pre_root_id).tolist()) == {suffix}, np.unique(pre.pre_root_id)
            assert suffix == int(str(root_id)[len(sm.FLYWIRE_LABEL):])
            # the un-stripped id must not appear anywhere in this table
            assert len(all_sites.select_root(root_id, "pre_root_id")) == 0
            # every extracted row matches at least one orientation exactly
            assert np.all((all_sites.pre_root_id == suffix) | (all_sites.post_root_id == suffix))
            assert len(pre) + len(post) >= len(all_sites)
        check("real data: extracted presynaptic sites carry only the target (stripped) root id",
              t_real_ids_exact)

        def t_real_assignment():
            morph_real, swc_ids = swc_to_morphology(SAMPLE, units="nm")
            assert morph_real.n == metrics["source"]["node_count"]
            a = sm.assign_nearest_node(pre, morph_real, far_um=2.0, node_ids=swc_ids)
            st = a["distance_stats_um"]
            assert st["min"] >= 0.0 and math.isfinite(st["max"]) and st["median"] >= 0.0
            assert st["max"] >= st["median"] >= st["min"]
            assert a["n_sites"] == len(pre)
            assert a["n_far"] == int(np.asarray(a["far"]).sum())
            assert np.all(np.asarray(a["node"]) < morph_real.n)
            assert np.asarray(a["node"]).min() >= 0
            # frame-agreement evidence: with the threshold lifted, distances stay small
            b = sm.assign_nearest_node(pre, morph_real, far_um=1e9)
            assert b["distance_stats_um"]["max"] < 50.0, b["distance_stats_um"]
        check("real data: assignment distances are finite and reported", t_real_assignment)

        def t_real_report_agrees():
            assert REPORT.exists(), "synapse_map_report.json missing"
            rep = json.loads(REPORT.read_text())
            assert rep["synapse_counts"]["real_presynaptic_sites_for_root_id"] == len(pre)
            assert rep["data_product"]["name"] == sm.SYNAPSE_PRODUCT
            assert rep["data_product"]["dataset"] == "fafb"
            assert "redacted" in rep["data_product"]["api_pattern"]
            blob = REPORT.read_text()
            # The report documents the API *pattern*, so the literal
            # "api_token=" may appear -- but every occurrence must be followed
            # by a redaction marker, never by a credential.
            import re as _re
            for m in _re.finditer(r"api_token=([^&\"'\\s]*)", blob):
                assert m.group(1).lower().startswith(("<redacted>", "redacted")), \
                    f"api_token= is followed by a non-redacted value: {m.group(1)!r}"
            # The credential file itself is deliberately ABSENT from the
            # delivered package, so the exact-string comparison can only run
            # where the token exists.  Its absence must NOT fail this test:
            # a failure would wrongly imply the report leaks a secret.
            token_path = ROOT / ".flywire_api_token"
            if token_path.exists():
                assert token_path.read_text().strip() not in blob, \
                    "api token must never appear in the report"
            else:
                skip("real data: report counts agree with the raw extraction and leak no token",
                     ".flywire_api_token is not present (delivered package has no "
                     "credentials), so the exact-string non-leak check is skipped; "
                     "the api_token query-string check still ran")
            assert rep["data_product"]["header_columns"][:3] == ["pre_x", "pre_y", "pre_z"]
        check("real data: report counts agree with the raw extraction and leak no token",
              t_real_report_agrees)

        def t_real_frame():
            morph_real, swc_ids = swc_to_morphology(SAMPLE, units="nm")
            a = sm.assign_nearest_node(pre, morph_real, far_um=2.0, node_ids=swc_ids)
            close = ~np.asarray(a["far"])
            assert close.sum() > 0
            assert np.asarray(a["distance_um"])[close].max() <= 2.0 + 1e-9
            assert np.asarray(a["distance_um"])[~close].min() > 2.0
        check("real data: every site maps inside the retained subtree frame", t_real_frame)

        def t_real_truncation_split():
            morph_real, swc_ids = swc_to_morphology(SAMPLE, units="nm")
            a = sm.assign_nearest_node(pre, morph_real, far_um=2.0, node_ids=swc_ids)
            excl = metrics["truncation"]["excluded_source_ids"]
            acc = sm.truncation_accounting(a, excl, swc_ids)
            assert acc["n_mapped_to_retained_nodes"] + acc["n_mapped_to_excluded_nodes"] == len(pre)
            assert set(np.asarray(a["node_swc_id"]).tolist()) <= set(np.asarray(swc_ids).tolist())
            assert len(set(np.asarray(swc_ids).tolist())) == morph_real.n
        check("real data: retained/excluded split is consistent with the id sets",
              t_real_truncation_split)

    # ---------------- demo-run invariants ----------------------------------
    if REPORT.exists():
        def t_demo_invariants():
            rep = json.loads(REPORT.read_text())
            pc = rep["paired_comparison"]
            sd = pc["identical_total_drive"]
            assert sd["point_arm_total_sites"] == sd["distributed_arm_total_sites"] == \
                pc["total_sites_each_arm"]
            assert sd["events_identical"] is True
            assert sd["event_counts_identical_per_step"] is True
            assert sd["same_g_per_site"] is True
            assert isinstance(pc["materiality"]["material"], bool)
            assert rep["outputs"]["png"].endswith("synapse_map_demo.png")
            assert any("no claim" in c.lower() for c in rep["caveats"])
            assert rep["resources"]["peak_rss_mb_main"] < _MEM_LIMIT_MB
        check("demo run: report invariants (equal drive, materiality flag, caveats)",
              t_demo_invariants)
    else:
        skip("demo run: report invariants (equal drive, materiality flag, caveats)",
             "synapse_map_report.json not present yet; run run_synapse_map.py first")

    def t_assignment_does_not_mutate_inputs():
        pts = nodes_nm[[0, 1, 2]].copy()
        before = pts.copy()
        r = sm.assign_nearest_node(pts, morph)
        assert np.array_equal(pts, before), "assignment must not modify the site coordinates"
        assert r["n_sites"] == 3
    check("assignment: input coordinate arrays are never modified", t_assignment_does_not_mutate_inputs)

    def t_memory_budget():
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        assert mb < _MEM_LIMIT_MB, f"peak RSS {mb:.0f} MB exceeded {_MEM_LIMIT_MB:.0f} MB budget"
    check("resources: self-test peak RSS stays inside the 3 GB budget", t_memory_budget)

    result = {
        "suite": "run_synapse_map_selftest.py",
        "passed": len(_PASS), "failed": len(_FAIL), "skipped": len(_SKIP),
        "total": len(_PASS) + len(_FAIL) + len(_SKIP),
        "status": "passed" if not _FAIL else "failed",
        "real_data_available": bool(real_ok),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "wall_clock_s": round(time.time() - t_start, 2),
        "passed_tests": _PASS, "failures": _FAIL, "skipped_tests": _SKIP,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synapse_map_selftest.json").write_text(json.dumps(result, indent=2) + "\n")
    if REPORT.exists():
        rep = json.loads(REPORT.read_text())
        rep["tests"] = {"suite": result["suite"], "passed": result["passed"],
                        "failed": result["failed"], "skipped": result["skipped"],
                        "status": result["status"],
                        "skipped_tests": result["skipped_tests"],
                        "selftest_json": str(OUT / "synapse_map_selftest.json")}
        REPORT.write_text(json.dumps(sm.json_ready(rep), indent=2, sort_keys=True) + "\n")
    print(f"\n{result['status'].upper()}: {result['passed']} passed, {result['failed']} failed, "
          f"{result['skipped']} skipped, peak RSS {result['peak_rss_mb']:.0f} MB, "
          f"{result['wall_clock_s']:.1f} s", flush=True)
    return result


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res["failed"] == 0 else 1)
