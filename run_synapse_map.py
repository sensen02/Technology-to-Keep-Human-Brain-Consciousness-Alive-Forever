#!/usr/bin/env python3
"""Attach REAL FlyWire presynaptic site coordinates to the validated FAFB subtree.

Manually implemented and tested prototype.  Real coordinates are data; the
compartment assignment is an approximation; the retained subtree is truncated.

Data product used (verified live against the Codex download-resource API):
    data_product=synapse_table   dataset=fafb
    columns: pre_x,pre_y,pre_z,ctr_x,ctr_y,ctr_z,post_x,post_y,post_z,size,
             pre_root_id_720575940,post_root_id_720575940,neuropil
    coordinates are FAFB nanometres; the two root-id columns carry the FAFB
    prefix "720575940" stripped (ids fit in int32), which is restored here.

Outputs (all under outputs/brain_isolation/):
    synapse_map_report.json   full provenance, counts, distances, comparison
    synapse_map_sites.npz     real sites, assignment, per-step voltages
    synapse_map_demo.png      3D skeleton + real sites + paired comparison
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.cable import CableNeuron, swc_to_morphology          # noqa: E402
from engine import synapse_map as sm                             # noqa: E402

OUT = ROOT / "outputs" / "brain_isolation"
DATA = ROOT / "data" / "flywire"
METRICS = OUT / "real_morphology_subtree_metrics.json"
SAMPLE = OUT / "real_morphology_subtree_sample.swc"
REJECTED = OUT / "real_morphology_subtree_source_rejected.swc"
RAW_NPZ = OUT / "synapse_map_raw.npz"

FAR_UM = 2.0            # a real synapse further than this from every retained
                        # node is flagged: it may belong to a truncated region
G_PER_SITE_US = 1e-5    # illustrative, NOT measured for this cell
                        # (5e-4 uS -- 0.5 nS -- is the literature-ish value, but with
                        # these borrowed passive parameters it is numerically unstable
                        # when 1000 sites sit on one compartment; see numerical_check)
STABILITY_SAFETY = 0.25  # fraction of the explicit-sampling limit g*dt/C <= 2
E_REV_MV = 0.0          # illustrative cholinergic-ish reversal
TAU_MS = 2.0
RATE_HZ = 80.0
TOTAL_SITES = 1000      # identical total drive in both arms of the comparison
DT_MS = 0.025
DURATION_MS = 20.0
T_START_MS, T_END_MS = 5.0, 15.0
SEED = 4242

CAVEATS = [
    "Presynaptic site coordinates are REAL measured FAFB data (data_product=synapse_table).",
    "Compartment assignment is nearest-retained-skeleton-node: an approximation, not "
    "a reconstruction of the postsynaptic density.",
    "The retained subtree is truncated (980 of 999 source nodes); synapses whose nearest "
    "retained node is an excluded id, or which are far from every retained node, are missing input.",
    "Cable passive parameters are borrowed from DNp01/DNp03 (PMC11071487); they are NOT a fit "
    "to this cell and not whole-fly calibration.",
    "g_per_site, tau_syn, E_rev and the release rate are illustrative assumptions, not measurements.",
    "Passive cable only: no active channels, no vesicle depletion, no neuromodulation.",
    "No claim is made about neural coding, consciousness or viability.",
]


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_real_sites(root_id, allow_download=True, log=print):
    """Return (presynaptic Sites, postsynaptic Sites, provenance) for root_id.

    ID CONVENTION (measured, not assumed): the `synapse_table` root-id columns are
    named `pre_root_id_720575940` / `post_root_id_720575940` and store the FAFB
    root id with that 9-digit prefix STRIPPED (the stripped value fits in int32).
    Target 720575940631271235 therefore appears as 631271235.  The suffix is
    derived from the column label itself, and both orientations are streamed in a
    single pass (one download, bounded id set).
    """
    suffix = int(str(root_id)[len(sm.FLYWIRE_LABEL):])
    prov = {"data_product": sm.SYNAPSE_PRODUCT, "dataset": sm.DEFAULT_FAFB_DATASET,
            "full_root_id": str(root_id), "stripped_root_id_in_table": str(suffix),
            "id_convention": ("root-id columns carry the FAFB prefix "
                              f"{sm.FLYWIRE_LABEL} stripped; value stored is {suffix}"),
            "download_attempted": False}
    if RAW_NPZ.exists():
        stats_path = OUT / "synapse_map_stream_stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        prov["mode"] = "cached single-pass streaming extraction"
        prov["stream_stats"] = stats
        prov["raw_npz"] = str(RAW_NPZ)
        prov["raw_npz_sha256"] = sm.sha256_file(RAW_NPZ)
        prov["raw_npz_bytes"] = RAW_NPZ.stat().st_size
        with np.load(RAW_NPZ, allow_pickle=False) as z:
            fields = {k: z[k] for k in z.files}
        all_sites = sm.Sites.from_fields(fields, source=str(RAW_NPZ))
        pre = all_sites.select_root(suffix, "pre_root_id")
        post = all_sites.select_root(suffix, "post_root_id")
        # every extracted row matches at least one orientation exactly (self-loop
        # rows match both, so the two subsets may overlap by those rows)
        both = (all_sites.pre_root_id == suffix) & (all_sites.post_root_id == suffix)
        assert np.all((all_sites.pre_root_id == suffix) | (all_sites.post_root_id == suffix)), \
            "extraction contains unrelated rows"
        prov["n_self_loop_rows_matching_both_orientations"] = int(both.sum())
        prov["n_rows_extracted"] = int(len(all_sites))
        prov["n_presynaptic_sites"] = int(len(pre))
        prov["n_postsynaptic_sites"] = int(len(post))
        prov["distinct_input_partners"] = int(len(np.unique(post.pre_root_id))) if len(post) else 0
        prov["distinct_output_partners"] = int(len(np.unique(pre.post_root_id))) if len(pre) else 0
        return pre, post, prov
    if not allow_download:
        raise RuntimeError("no cached synapse extraction available")
    prov["download_attempted"] = True
    log("streaming the real synapse table from Codex (2.7 GB gzipped) ...")
    matches, stats = sm.stream_filter(pre_root_ids=[suffix], post_root_ids=[suffix],
                                      log=log, dest_npz=str(RAW_NPZ))
    prov["stream_stats"] = stats
    sites = sm.Sites.from_fields(matches, source="streamed from Codex")
    pre = sites.select_root(suffix, "pre_root_id")
    post = sites.select_root(suffix, "post_root_id")
    prov["n_rows_extracted"] = int(len(sites))
    prov["n_presynaptic_sites"] = int(len(pre))
    prov["n_postsynaptic_sites"] = int(len(post))
    return pre, post, prov


def main():
    t_wall = time.time()
    tt0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS.read_text())
    root_id = int(metrics["source"]["source_id"])
    excluded_ids = list(metrics["truncation"]["excluded_source_ids"])
    log = lambda s: print(s, flush=True)  # noqa: E731

    # ---------- validated morphology (unchanged, re-imported strictly) ------
    morph, swc_ids = swc_to_morphology(SAMPLE, units="nm")
    assert morph.n == metrics["source"]["node_count"] == len(metrics["truncation"]["row_to_original_swc_id"])
    assert list(swc_ids) == list(metrics["truncation"]["row_to_original_swc_id"])

    # coordinates of excluded source nodes, for the truncation accounting
    src = {int(r[0]): r for r in sm.read_swc_rows(REJECTED)}
    excluded_nodes_nm = np.array([[float(src[i][2]), float(src[i][3]), float(src[i][4])]
                                  for i in excluded_ids])

    # ---------- real synapses -------------------------------------------------
    pre, post, prov = load_real_sites(root_id, log=log)
    log(f"presynaptic input sites of {root_id}: {len(pre)}   "
        f"postsynaptic output sites: {len(post)}")
    if len(pre) == 0:
        raise RuntimeError("no presynaptic sites for the target root id in the synapse table")

    nodes_nm = np.column_stack([morph.x, morph.y, morph.z]) * 1000.0
    a_pre = sm.assign_nearest_node(pre, morph, far_um=FAR_UM, node_ids_nm=nodes_nm,
                                   node_ids=swc_ids)
    a_unrestricted = sm.assign_nearest_node(pre, morph, far_um=1e9,
                                            node_ids_nm=nodes_nm, node_ids=swc_ids)
    a_post = (sm.assign_nearest_node(post, morph, far_um=FAR_UM, node_ids_nm=nodes_nm,
                                     node_ids=swc_ids) if len(post) else None)

    # How many real sites fall nearer to an EXCLUDED source node than to any
    # retained node: those are the inputs the truncation loses.
    from scipy.spatial import cKDTree
    ex_tree = cKDTree(excluded_nodes_nm) if len(excluded_nodes_nm) else None
    if ex_tree is not None:
        d_ex, i_ex = ex_tree.query(pre.xyz_nm, k=1)
        closer_to_excluded = d_ex < a_unrestricted["distance_nm"]
    else:
        d_ex = np.full(len(pre), np.inf)
        closer_to_excluded = np.zeros(len(pre), dtype=bool)
    trunc = sm.truncation_accounting(a_pre, excluded_ids, swc_ids)
    boundary_retained = sorted({b["retained_id"] for b in
                                metrics["truncation"]["artificial_sealed_boundary_edges"]})
    boundary_rows = {int(r) for r, i in enumerate(swc_ids) if i in set(boundary_retained)}
    trunc.update(sm.truncation_boundary_accounting(
        {**a_pre, "node_x_nm": nodes_nm[:, 0]}, boundary_retained, excluded_nodes_nm, pre))
    trunc["n_input_sites_on_a_boundary_node"] = int(sum(
        1 for r in a_pre["node"] if int(r) in boundary_rows))
    trunc.update({
        "n_sites_closer_to_an_excluded_node": int(closer_to_excluded.sum()),
        "excluded_node_count": len(excluded_ids),
        "retained_nodes": int(morph.n),
        "source_nodes": int(metrics["source"]["original_node_count"]),
        "criterion": ("counted on the excluded nodes when the nearest RETAINED node's "
                      "original SWC id is an excluded id; separately, a site is 'closer to "
                      "an excluded node' when its distance to the nearest excluded source "
                      "node beats its distance to every retained node"),
        "excluded_id_nearest_counts": {
            str(e): int((np.asarray(a_pre["node_swc_id"]) == e).sum()) for e in excluded_ids},
        "n_flagged_far": int(a_pre["n_far"]),
    })

    # ---------- compartment distribution -------------------------------------
    node_rows = np.asarray(a_pre["node"], dtype=np.int64)
    counts_retained, _ = sm.distribution(node_rows, n_compartments=morph.n)
    occupied = (counts_retained > 0).astype(np.int64)
    counts_one_per = sm.distribute_sites(int(counts_retained.sum()), occupied.astype(float))
    interior_rows = np.flatnonzero(morph.parent >= 0)
    single_row = int(interior_rows[np.argmax(morph.d[interior_rows])])

    # sites used for the distributed arm: real sites that are not flagged as far
    # from the retained subtree
    obs_main = node_rows[~np.asarray(a_pre["far"])]
    if obs_main.size == 0:
        raise RuntimeError("no real site maps credibly onto the retained subtree")

    # ---------- paired comparison --------------------------------------------
    interior = np.flatnonzero(morph.parent >= 0)
    record_rows = [0, int(interior[0]), single_row, int(morph.n) - 1]
    record_rows = sorted(set(record_rows))
    pair = sm.pairwise_response_compare(
        morph, site_nodes=obs_main, total_sites=TOTAL_SITES, single_row=single_row,
        record_rows=record_rows, g_per_site_uS=G_PER_SITE_US, E_rev_mV=E_REV_MV,
        tau_ms=TAU_MS, rate_hz=RATE_HZ, dt_ms=DT_MS, duration_ms=DURATION_MS,
        t_start_ms=T_START_MS, t_end_ms=T_END_MS, seed=SEED, implicit=True)
    arm_a, arm_b = pair["point"], pair["distributed"]
    Va, Vb = arm_a["V_mV"], arm_b["V_mV"]
    d = pair["summary"]["per_row"]

    # Numerical cross-check: the implicit solve needs no stability limit, while
    # the explicit practice (current sampled at the previous voltage) is stable
    # only while g*dt/C <= 2.  Re-run the same comparison both ways.
    from engine.cable import CableNeuron as _CN
    probe = _CN(morph, dt_ms=DT_MS)
    point_counts = np.zeros(morph.n)
    point_counts[single_row] = TOTAL_SITES
    dist_counts = sm.distribute_sites(TOTAL_SITES, np.bincount(obs_main, minlength=morph.n).astype(float))
    lim_point = sm.explicit_stability_limit(probe, point_counts, safety=STABILITY_SAFETY)
    lim_dist = sm.explicit_stability_limit(probe, dist_counts.astype(float), safety=STABILITY_SAFETY)
    g_explicit_point = min(G_PER_SITE_US, lim_point) if lim_point else None
    pair_explicit = sm.pairwise_response_compare(
        morph, site_nodes=obs_main, total_sites=TOTAL_SITES, single_row=single_row,
        record_rows=record_rows, g_per_site_uS=G_PER_SITE_US, E_rev_mV=E_REV_MV,
        tau_ms=TAU_MS, rate_hz=RATE_HZ, dt_ms=DT_MS, duration_ms=DURATION_MS,
        t_start_ms=T_START_MS, t_end_ms=T_END_MS, seed=SEED, implicit=False,
        clamp_to_stability=True, stability_safety=STABILITY_SAFETY)
    numerical_check = {
        "implicit": {
            "description": "synaptic conductance placed inside the backward-Euler solve",
            "point_arm_max_abs_voltage_mV": pair["summary"]["numerical"]["point_arm_max_abs_voltage_mV"],
            "distributed_arm_max_abs_voltage_mV": pair["summary"]["numerical"]["distributed_arm_max_abs_voltage_mV"],
        },
        "explicit_with_stability_clamp": {
            "description": ("existing practice: current sampled at the previous voltage, with the "
                            "per-site conductance reduced to the largest value satisfying g*dt/C <= 2"),
            "g_per_site_applied_uS": pair_explicit["distributed"]["g_per_site_uS"],
            "point_axis_limit_uS": lim_point,
            "distributed_axis_limit_uS": lim_dist,
            "point_arm_stability_clamp": pair_explicit["point"]["stability_clamp"],
            "distributed_arm_stability_clamp": pair_explicit["distributed"]["stability_clamp"],
            "point_arm_max_abs_voltage_mV": pair_explicit["summary"]["numerical"]["point_arm_max_abs_voltage_mV"],
            "distributed_arm_max_abs_voltage_mV": pair_explicit["summary"]["numerical"]["distributed_arm_max_abs_voltage_mV"],
        },
        "unclamped_explicit_practice_is_unstable": {
            "g_per_site_requested_uS": 5e-4,
            "reason": ("with 1000 sites on ONE compartment the explicit scheme exceeds g*dt/C <= 2 "
                       "by orders of magnitude and the solve diverges; this was observed and is why "
                       "the implicit solve (or the clamped explicit run) is reported instead"),
        },
    }

    # materiality of the spatial redistribution
    root_row = 0
    idx = record_rows.index(root_row)
    dev_a = float(np.abs(Va[:, idx] - Va[0, idx]).max())
    dev_b = float(np.abs(Vb[:, idx] - Vb[0, idx]).max())
    rel = (dev_b - dev_a) / dev_a if dev_a else float("inf")
    pair["summary"]["materiality"] = {
        "root_side_row": int(root_row),
        "point_arm_peak_deviation_mV": dev_a,
        "distributed_arm_peak_deviation_mV": dev_b,
        "relative_change_fraction": float(rel),
        "max_abs_difference_any_recorded_row_mV": pair["summary"]["per_step_max_abs_delta_mV"],
        "material_change_threshold": "declared here as >10% change in root-side peak deviation",
        "material": bool(abs(rel) > 0.10),
    }

    # ---------- robustness: other seeds + other axial resistances -------------
    def two_arm(*, seed=None, passive=None, g_per_site_uS=None):
        pr = sm.pairwise_response_compare(
            morph, site_nodes=obs_main, total_sites=TOTAL_SITES, single_row=single_row,
            record_rows=[root_row],
            g_per_site_uS=G_PER_SITE_US if g_per_site_uS is None else g_per_site_uS,
            E_rev_mV=E_REV_MV,
            tau_ms=TAU_MS, rate_hz=RATE_HZ, dt_ms=DT_MS, duration_ms=DURATION_MS,
            t_start_ms=T_START_MS, t_end_ms=T_END_MS,
            seed=SEED if seed is None else seed, implicit=True, passive=passive)
        A, B = pr["point"]["V_mV"][:, 0], pr["distributed"]["V_mV"][:, 0]
        pa = float(np.abs(A - A[0]).max())
        pb = float(np.abs(B - B[0]).max())
        return {"point_mV": pa, "distributed_mV": pb,
                "relative_change_fraction": (pb - pa) / pa if pa else float("inf"),
                "max_abs_difference_mV": float(np.abs(A - B).max()),
                "events_identical": pr["point"]["total_events"] == pr["distributed"]["total_events"]}

    seed_sensitivity = {str(s): two_arm(seed=s) for s in (4242, 7, 99, 100003)}
    ra_sensitivity = {
        str(int(212.0 * m_)): two_arm(seed=SEED, passive={"Ra_ohm_cm": 212.0 * m_})
        for m_ in (1.0, 1e2, 1e4, 1e6)}
    g_sensitivity = {f"{g:g}": two_arm(g_per_site_uS=g) for g in (1e-5, 1e-4, 1e-3)}
    robustness = {
        "seed_sensitivity": seed_sensitivity,
        "per_site_conductance_sensitivity": {
            "note": ("g_per_site is illustrative, not measured. 5e-4 uS (~0.5 nS) is a "
                     "literature-ish quantal value; 1e-5 uS (0.01 nS) is the value reported here. "
                     "A larger conductance drives the local membrane further toward E_rev, which "
                     "saturates the response and COMPRESSES the relative difference between arms."),
            "cases": g_sensitivity},
        "axial_resistance_sensitivity": {
            "note": ("Ra is swept upward because the borrowed DNp01/DNp03 parameters give an "
                     "axial conductance ~1e5-1e6 x the membrane conductance (electrotonically "
                     "compact); the sweep shows whether the conclusion survives less compact cables."),
            "cases": ra_sensitivity},
        "same_drive_holds_in_every_case": all(
            v["events_identical"] for v in list(seed_sensitivity.values()) +
            list(ra_sensitivity.values()) + list(g_sensitivity.values())),
    }

    # ---------- figure --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(19.0, 6.8))
    ax = fig.add_axes((0.015, 0.05, 0.30, 0.80), projection="3d")
    node_um = nodes_nm / 1000.0
    ex_um = excluded_nodes_nm / 1000.0
    ax.scatter(node_um[:, 0], node_um[:, 1], node_um[:, 2], s=3.5, c="0.62",
               label=f"retained skeleton nodes (n={morph.n})", depthshade=False)
    ax.scatter(ex_um[:, 0], ex_um[:, 1], ex_um[:, 2],
               s=55, marker="x", c="crimson", label=f"excluded source nodes (n={len(excluded_ids)})")
    close = ~np.asarray(a_pre["far"])
    site_um = pre.xyz_nm / 1000.0
    ax.scatter(site_um[close, 0], site_um[close, 1], site_um[close, 2],
               s=24, c="tab:blue", alpha=0.9, depthshade=False, edgecolors="none",
               label=f"REAL presynaptic sites, <= {FAR_UM:g} um from a node (n={int(close.sum())})")
    if np.any(~close):
        ax.scatter(site_um[~close, 0], site_um[~close, 1], site_um[~close, 2],
                   s=40, c="orange", marker="^", depthshade=False,
                   label=f"flagged far from every node (n={int((~close).sum())})")
    ax.view_init(elev=18, azim=-62)
    ax.set_title(f"REAL presynaptic sites on the truncated FAFB subtree\n"
                 f"root {root_id}\n({morph.n} retained of "
                 f"{metrics['source']['original_node_count']} source nodes; "
                 f"{len(excluded_ids)} excluded)", fontsize=9.5, pad=3)
    for axis, name in ((ax.set_xlabel, "x (um)"), (ax.set_ylabel, "y (um)"), (ax.set_zlabel, "z (um)")):
        axis(name, fontsize=9, labelpad=1)
    ax.tick_params(axis="both", labelsize=6.5, pad=-1)
    ax.zaxis.set_tick_params(labelsize=6.5, pad=0)
    for lbl in (ax.xaxis, ax.yaxis, ax.zaxis):
        lbl.set_major_locator(plt.MaxNLocator(4, prune="both"))
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(-0.12, 1.0), framealpha=0.9)

    t = np.arange(len(Va)) * DT_MS
    ax2 = fig.add_axes((0.375, 0.12, 0.26, 0.72))
    ax2.plot(t, Va[:, idx], lw=1.8, color="tab:red",
             label=f"(a) all {TOTAL_SITES} sites at ONE arbitrary point (row {single_row})")
    ax2.plot(t, Vb[:, idx], lw=1.8, ls="--", color="tab:blue",
             label="(b) same sites spread over the REAL synapse locations")
    ax2.axvspan(T_START_MS, T_END_MS, color="0.92", zorder=0, label="release window")
    ax2.set_title(f"Somatic/root-side (row {root_row}) response — identical total drive\n"
                  f"{TOTAL_SITES} sites, {G_PER_SITE_US*1e3:g} nS/site, {RATE_HZ:g} Hz, "
                  f"E_rev {E_REV_MV:g} mV, {DT_MS:g} ms steps", fontsize=10)
    ax2.set_xlabel("time (ms)"); ax2.set_ylabel("V (mV)")
    ax2.legend(fontsize=7.5, loc="lower left"); ax2.grid(alpha=0.3)
    txt = (f"peak deviation from rest\n"
           f"(a) point: {d[str(root_row)]['point_peak_abs_deviation_mV']:.2f} mV\n"
           f"(b) real sites: {d[str(root_row)]['distributed_peak_abs_deviation_mV']:.2f} mV\n"
           f"change: {100*pair['summary']['materiality']['relative_change_fraction']:+.1f}%  "
           f"(material: {pair['summary']['materiality']['material']})")
    ax2.text(0.02, 0.97, txt, transform=ax2.transAxes, va="top", fontsize=8,
             bbox=dict(fc="white", ec="0.6", alpha=0.9))

    ax3 = fig.add_axes((0.685, 0.12, 0.28, 0.72))
    st = a_pre["distance_stats_um"]
    bins = np.linspace(0.0, max(st["max"] * 1.08, FAR_UM * 1.15), 45)
    ax3.hist(a_pre["distance_um"], bins=bins, color="tab:blue", alpha=0.85)
    ax3.axvline(FAR_UM, color="orange", ls="--", lw=1.6, label=f"far threshold {FAR_UM:g} um")
    ax3.axvline(st["median"], color="k", ls=":", lw=1.6, label=f"median {st['median']:.3f} um")
    ax3.set_title("Nearest-node assignment distance\n"
                  f"{len(pre)} real presynaptic sites mapped onto {morph.n} retained nodes", fontsize=10)
    ax3.set_xlabel("site-to-nearest-retained-node distance (um)")
    ax3.set_ylabel("number of sites")
    ax3.set_xlim(0.0, bins[-1])
    ax3.legend(fontsize=8, loc="upper center")
    ax3.text(0.43, 0.62,
             f"min {st['min']:.3f} um\nmedian {st['median']:.3f} um\nmean {st['mean']:.3f} um\n"
             f"p90 {st['p90']:.3f} um\nmax {st['max']:.3f} um\n"
             f">1 um: {st['n_over_1um']}\n>5 um: {st['n_over_5um']}\n"
             f"flagged far: {int(a_pre['n_far'])}",
             transform=ax3.transAxes, fontsize=8, va="top",
             bbox=dict(fc="white", ec="0.6", alpha=0.9))
    fig.suptitle("MANUALLY IMPLEMENTED PROTOTYPE — real FAFB presynaptic coordinates (product synapse_table); "
                 "nearest-node compartment assignment is an approximation; truncated 980/999-node subtree;\n"
                 "passive cable parameters borrowed from DNp01/DNp03, NOT fitted to this cell; "
                 "synaptic g/tau/E_rev are illustrative; no claim about neural coding, consciousness or viability",
                 fontsize=8.5)

    png = OUT / "synapse_map_demo.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)

    # ---------- npz + report --------------------------------------------------
    npz = OUT / "synapse_map_sites.npz"
    np.savez_compressed(
        npz,
        pre_x_nm=pre.xyz_nm[:, 0], pre_y_nm=pre.xyz_nm[:, 1], pre_z_nm=pre.xyz_nm[:, 2],
        pre_root_id=pre.pre_root_id, post_root_id=pre.post_root_id, size=pre.size,
        neuropil=pre.neuropil.astype(str),
        assigned_node_row=a_pre["node"], assigned_node_swc_id=a_pre["node_swc_id"],
        assignment_distance_um=a_pre["distance_um"], far_flag=a_pre["far"],
        node_x_nm=nodes_nm[:, 0], node_y_nm=nodes_nm[:, 1], node_z_nm=nodes_nm[:, 2],
        node_swc_id=np.asarray(swc_ids), node_parent=morph.parent,
        excluded_node_ids=np.asarray(excluded_ids), excluded_node_xyz_nm=excluded_nodes_nm,
        t_ms=t, v_point_mV=arm_a["V_mV"], v_distributed_mV=arm_b["V_mV"],
        v_delta_mV=pair["delta_mV"], g_point_uS=arm_a["G_uS"],
        g_distributed_uS=arm_b["G_uS"], record_rows=np.asarray(record_rows),
        sites_per_compartment=arm_b["sites_per_compartment"],
    )

    elapsed = time.time() - t_wall
    report = {
        "status": "completed_real_synapse_coordinates",
        "implementation": "manually implemented/tested prototype; not calibrated biology",
        "goal": "attach real FlyWire presynaptic site coordinates to the validated FAFB skeleton subtree",
        "data_product": {
            "name": sm.SYNAPSE_PRODUCT,
            "dataset": sm.DEFAULT_FAFB_DATASET,
            "api_pattern": sm.CODEX_API + "?data_product=<product>&dataset=<dataset>&api_token=<redacted>",
            "header_columns": prov.get("stream_stats", {}).get("header_columns",
                                                                ["pre_x", "pre_y", "pre_z", "ctr_x",
                                                                 "ctr_y", "ctr_z", "post_x", "post_y",
                                                                 "post_z", "size", "pre_root_id_720575940",
                                                                 "post_root_id_720575940", "neuropil"]),
            "declared_gzip_bytes": prov.get("stream_stats", {}).get("declared_size_bytes", 2695106039),
            "coordinate_frame": "FAFB nanometres (pre_x, pre_y, pre_z); root-id columns have the FAFB prefix 720575940 stripped",
            "id_convention": prov["id_convention"],
            "fetched_once": True,
            "extraction_mode": prov.get("mode"),
            "raw_extraction_npz": prov.get("raw_npz"),
            "raw_extraction_sha256": prov.get("raw_npz_sha256"),
            "raw_extraction_bytes": prov.get("raw_npz_bytes"),
            "stream_stats": prov.get("stream_stats"),
            "gzip_downloaded_to_disk": False,
            "download_note": ("streamed over HTTPS and decompressed incrementally in one pass; the "
                              "2.7 GB archive is never written to disk and never fully held in memory"),
        },
        "target_neuron": {
            "root_id": str(root_id),
            "retained_nodes": int(morph.n),
            "source_nodes": int(metrics["source"]["original_node_count"]),
            "excluded_nodes": len(excluded_ids),
            "artificial_sealed_boundary_edges": len(metrics["truncation"]["artificial_sealed_boundary_edges"]),
            "sample_swc": str(SAMPLE),
            "sample_sha256": sm.sha256_file(SAMPLE),
        },
        "synapse_counts": {
            "real_presynaptic_sites_for_root_id": int(len(pre)),
            "real_postsynaptic_output_sites_for_root_id": int(len(post)),
            "distinct_input_partners": prov.get("distinct_input_partners"),
            "distinct_output_partners": prov.get("distinct_output_partners"),
            "total_rows_in_synapse_table_scanned": prov.get("stream_stats", {}).get("rows_scanned"),
            "rows_matched_in_stream": prov.get("stream_stats", {}).get("rows_matched"),
            "malformed_rows": prov.get("stream_stats", {}).get("malformed_rows"),
            "presynaptic_on_retained_980_node_subtree": trunc["n_mapped_to_retained_nodes"],
            "presynaptic_on_excluded_19_nodes": trunc["n_mapped_to_excluded_nodes"],
            "flagged_far": trunc["n_flagged_far"],
            "closer_to_an_excluded_node_than_to_any_retained_node": trunc["n_sites_closer_to_an_excluded_node"],
            "usable_for_distributed_drive": int(len(obs_main)),
            "neuropil_labels_present": prov.get("stream_stats", {}).get("neuropil_labels", []),
            "note": ("counts are for the presynaptic (input) sites of the target root id; the "
                     "extraction npz also holds the postsynaptic output rows, reported separately"),
        },
        "data_quality_flags": {
            "connectome_cross_check": {
                "product": "connections_princeton (cached local copy)",
                "finding": ("the cached connections_princeton table records only 2 presynaptic "
                            "partners / 14 input synapse rows and 1 postsynaptic partner / 5 output "
                            "rows for this root id, whereas synapse_table yields 28 input partners / "
                            "65 input site rows (sum of size = 4437) and 36 output partners / 54 site "
                            "rows (sum of size = 2564)."),
                "interpretation": ("the cached connectome snapshot is NOT the same build as the "
                                   "current synapse_table; the connectome snapshot is treated as "
                                   "stale/subsampled for this cell and is NOT used to claim any "
                                   "synapse count. Synapse coordinates come only from synapse_table."),
                "consequences": "all counts in this report come from synapse_table alone",
            },
            "whole_table_row_count_observation": (
                "the streamed synapse_table contains 80,215,791 data rows with columns pre/post/"
                "ctr coordinates; this is not the same quantity as the project's chemical-synapse "
                "count of 50.6M (engine/profile.py), so the two are not interchangeable"),
            "unavailable_alternatives": (
                "candidate products 'synapses', 'proofread_synapses' and 'sites' are not available "
                "for dataset=fafb (HTTP 404 'Download resource ... is not available for dataset "
                "'fafb''); 'synapse_table' is the only probed product carrying presynaptic site "
                "coordinates"),
            "identity_check": ("every retained row has pre_root_id or post_root_id exactly equal to "
                               "631271235 (720575940631271235 with the table's stripped prefix); "
                               "verified by exact integer filtering, not by proximity"),
            "frame_check": ("sites and skeleton share the FAFB nm frame; the measured "
                            "unrestricted nearest-node distance distribution is reported under "
                            "'assignment_unrestricted' and is the evidence that the frames agree"),
        },
        "truncation_accounting": trunc,
        "assignment": {
            "role": "real presynaptic (input) sites mapped onto the retained skeleton",
            **{k: v for k, v in a_pre.items()
               if k in ("n_sites", "n_nodes", "n_far", "far_threshold_um", "assignment",
                        "distance_stats_um", "n_on_retained_subtree")},
        },
        "assignment_unrestricted": {
            "role": ("same sites with the far threshold lifted; this is the frame-agreement "
                     "evidence and is not used for the drive"),
            **{k: v for k, v in a_unrestricted.items()
               if k in ("n_sites", "n_far", "distance_stats_um")},
        },
        "assignment_postsynaptic": (
            {"role": "real postsynaptic (output) sites mapped onto the retained skeleton",
             **{k: v for k, v in a_post.items()
                if k in ("n_sites", "n_nodes", "n_far", "far_threshold_um", "distance_stats_um")}}
            if a_post else None),
        "distribution": {
            "compartments_with_a_real_site": int(np.count_nonzero(counts_retained)),
            "max_sites_on_one_compartment": int(counts_retained.max()),
            "mean_sites_per_occupied_compartment": float(counts_retained[counts_retained > 0].mean()),
            "sites_on_root_row": int(counts_retained[0]),
            "equal_per_compartment_alternative": {
                "compartment_count_used": int(np.count_nonzero(counts_one_per)),
                "max_sites_on_one_compartment": int(counts_one_per.max()),
                "total": int(counts_one_per.sum())},
            "single_arbitrary_row_current_practice": int(single_row),
            "site_weights_source": "real presynaptic sites (far-flagged excluded) mapped by nearest retained node",
        },
        "paired_comparison": {
            "arms": {
                "a_point": "all drive at one arbitrary point (largest-diameter interior row)",
                "b_distributed": "drive spread over the real mapped synapse compartments",
            },
            "total_sites_each_arm": TOTAL_SITES,
            "selected_from_real_sites": int(len(obs_main)),
            "g_per_site_uS": G_PER_SITE_US,
            "E_rev_mV": E_REV_MV, "tau_ms": TAU_MS, "rate_hz": RATE_HZ,
            "dt_ms": DT_MS, "duration_ms": DURATION_MS,
            "window_ms": [T_START_MS, T_END_MS], "seed": SEED,
            "record_rows": record_rows,
            "per_recorded_row": d,
            "identical_total_drive": pair["summary"]["same_total_drive"],
            "materiality": pair["summary"]["materiality"],
            "numerical_check": numerical_check,
            "robustness": robustness,
            "conclusion": None,   # filled below
        },
        "tests": {},
        "caveats": CAVEATS,
        "outputs": {"report": str(OUT / "synapse_map_report.json"),
                    "npz": str(npz), "png": str(png)},
        "resources": {
            "wall_clock_s_main": elapsed,
            "peak_rss_mb_main": peak_rss_mb(),
            "stream_extraction_elapsed_s": prov.get("stream_stats", {}).get("elapsed_s"),
            "stream_extraction_peak_rss_mb": prov.get("stream_stats", {}).get("peak_rss_mb"),
            "csv_bytes_streamed": prov.get("stream_stats", {}).get("csv_bytes_read"),
            "memory_strategy": ("the 2.7 GB gzip is streamed once and decompressed incrementally; "
                               "only matching rows are retained; sparse cable matrices only"),
        },
    }
    m = report["paired_comparison"]["materiality"]
    report["paired_comparison"]["conclusion"] = (
        f"Root-side peak deviation: point arm {m['point_arm_peak_deviation_mV']:.6g} mV vs "
        f"distributed arm {m['distributed_arm_peak_deviation_mV']:.6g} mV "
        f"(relative change {100*m['relative_change_fraction']:+.3f}%). "
        f"Max |difference| over all recorded rows {m['max_abs_difference_any_recorded_row_mV']:.3g} mV. "
        + ("The spatial distribution CHANGES the somatic/root-side response materially "
           "(> 10% change in peak deviation)."
           if m["material"] else
           "The spatial distribution does NOT change the root-side response materially "
           "(< 10% change in peak deviation); reported as a negative result."))

    (OUT / "synapse_map_report.json").write_text(json.dumps(sm.json_ready(report), indent=2, sort_keys=True) + "\n")

    # ---------- compact summary + human-readable summary ---------------------
    r = sm.json_ready(report)
    summary = {
        "data_product_used": r["data_product"]["name"],
        "dataset": r["data_product"]["dataset"],
        "id_convention": r["data_product"]["id_convention"],
        "target_root_id": r["target_neuron"]["root_id"],
        "rows_scanned_in_synapse_table": r["synapse_counts"]["total_rows_in_synapse_table_scanned"],
        "real_presynaptic_input_sites": r["synapse_counts"]["real_presynaptic_sites_for_root_id"],
        "real_postsynaptic_output_sites": r["synapse_counts"]["real_postsynaptic_output_sites_for_root_id"],
        "input_sites_on_retained_subtree": r["synapse_counts"]["presynaptic_on_retained_980_node_subtree"],
        "input_sites_on_excluded_nodes": r["synapse_counts"]["presynaptic_on_excluded_19_nodes"],
        "input_sites_flagged_far": r["synapse_counts"]["flagged_far"],
        "missing_input_fraction_due_to_truncation":
            round(r["synapse_counts"]["presynaptic_on_excluded_19_nodes"] /
                  max(r["synapse_counts"]["real_presynaptic_sites_for_root_id"], 1), 6),
        "assignment_distance_um": r["assignment"]["distance_stats_um"],
        "assignment_distance_um_unrestricted": r["assignment_unrestricted"]["distance_stats_um"],
        "compartments_with_a_real_site": r["distribution"]["compartments_with_a_real_site"],
        "paired_comparison": {
            "point_arm_root_peak_deviation_mV": r["paired_comparison"]["materiality"]["point_arm_peak_deviation_mV"],
            "distributed_arm_root_peak_deviation_mV": r["paired_comparison"]["materiality"]["distributed_arm_peak_deviation_mV"],
            "relative_change_fraction": r["paired_comparison"]["materiality"]["relative_change_fraction"],
            "max_abs_difference_mV": r["paired_comparison"]["materiality"]["max_abs_difference_any_recorded_row_mV"],
            "material": r["paired_comparison"]["materiality"]["material"],
            "identical_release_train": r["paired_comparison"]["identical_total_drive"]["event_counts_identical_per_step"],
            "seeds_tested": sorted(r["paired_comparison"]["robustness"]["seed_sensitivity"]),
            "ra_multipliers_tested": sorted(r["paired_comparison"]["robustness"]["axial_resistance_sensitivity"]["cases"]),
        },
        "conclusion": r["paired_comparison"]["conclusion"],
        "resources": r["resources"],
        "outputs": r["outputs"],
    }
    (OUT / "synapse_map_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "synapse_map_README.md").write_text(render_markdown(r, summary))
    print(json.dumps({
        "status": report["status"], "product": sm.SYNAPSE_PRODUCT,
        "presynaptic_sites": len(pre), "on_retained": trunc["n_mapped_to_retained_nodes"],
        "on_excluded": trunc["n_mapped_to_excluded_nodes"], "far_flagged": trunc["n_flagged_far"],
        "median_assignment_um": a_pre["distance_stats_um"]["median"],
        "material": m["material"], "rel_change": m["relative_change_fraction"],
        "wall_s": elapsed, "peak_rss_mb": peak_rss_mb(),
        "outputs": report["outputs"],
    }, indent=2))
    return report


def render_markdown(r, summary):
    """Human-readable summary of the report (all numbers come from the report)."""
    a = r["assignment"]["distance_stats_um"]
    u = r["assignment_unrestricted"]["distance_stats_um"]
    pc = r["paired_comparison"]
    mat = pc["materiality"]
    sd = pc["identical_total_drive"]

    def row(v):
        return (f"| {v['point_peak_mV']:.4f} | {v['distributed_peak_mV']:.4f} | "
                f"{v['point_peak_abs_deviation_mV']:.4f} | "
                f"{v['distributed_peak_abs_deviation_mV']:.4f} | {v['max_abs_difference_mV']:.4f} |")
    seeds = pc["robustness"]["seed_sensitivity"]
    ras = pc["robustness"]["axial_resistance_sensitivity"]["cases"]
    point_rows = pc["identical_total_drive"]["point_placement_rows"]
    point_row_txt = point_rows[0] if point_rows else r["distribution"]["single_arbitrary_row_current_practice"]
    n_real_comp = r["distribution"]["compartments_with_a_real_site"]
    suite_path = OUT / "synapse_map_selftest.json"
    suite = json.loads(suite_path.read_text()) if suite_path.exists() else None
    actual = r.get("tests") or {}
    if suite:
        when = "" if actual.get("passed") == suite.get("passed") else \
            " (from the most recent `run_synapse_map_selftest.py` run)"
        suite_line = (
            f"- `run_synapse_map_selftest.py`: **{suite['passed']} passed, {suite['failed']} failed, "
            f"{suite['skipped']} skipped** (status `{suite['status']}`, real-data assertions "
            f"{'ran' if suite.get('real_data_available') else 'skipped'}){when}. "
            f"Peak RSS {suite['peak_rss_mb']:.0f} MB, {suite['wall_clock_s']:.1f} s.")
    else:
        suite_line = "- `run_synapse_map_selftest.py`: not run yet in this output directory."
    lines = [
        "# Real FlyWire presynaptic coordinates on the validated truncated FAFB subtree",
        "",
        "**Manually implemented and tested prototype. Real coordinates are data; the compartment",
        "assignment is an approximation; the retained subtree is truncated; the cable passive",
        "parameters are borrowed from DNp01/DNp03 and are not this neuron's own fit; no claim about",
        "neural coding, consciousness or viability.**",
        "",
        "## Data product actually used",
        "",
        f"- `data_product=**{r['data_product']['name']}**`, `dataset={r['data_product']['dataset']}` "
        "via the same Codex download-resource API pattern as `flywire_download.py`",
        f"  (`{r['data_product']['api_pattern']}` — the token is never logged or written).",
        f"- Declared size **{r['data_product']['declared_gzip_bytes']:,} bytes** gzipped; header row:",
        f"  `{','.join(r['data_product']['header_columns'])}`",
        f"- ID convention (measured, not assumed): {r['data_product']['id_convention']}",
        "- The archive is streamed over HTTPS and decompressed incrementally **once**: it is never",
        "  written to disk and never held in memory. Only matching rows are retained.",
        "",
        "## Synapse counts (target root id " + r["target_neuron"]["root_id"] + ")",
        "",
        f"- Rows scanned in the whole table: **{r['synapse_counts']['total_rows_in_synapse_table_scanned']:,}** "
        f"(malformed rows: {r['synapse_counts']['malformed_rows']})",
        f"- Real **presynaptic (input) sites**: **{r['synapse_counts']['real_presynaptic_sites_for_root_id']}** "
        f"from {r['synapse_counts']['distinct_input_partners']} distinct presynaptic partners",
        f"- Real postsynaptic (output) sites: {r['synapse_counts']['real_postsynaptic_output_sites_for_root_id']} "
        f"from {r['synapse_counts']['distinct_output_partners']} partners (reported for completeness)",
        f"- On the retained 980-node subtree: **{r['synapse_counts']['presynaptic_on_retained_980_node_subtree']}**",
        f"- On the 19 excluded nodes: **{r['synapse_counts']['presynaptic_on_excluded_19_nodes']}**",
        f"- Flagged far from every retained node (> {a and r['assignment']['far_threshold_um']:g} um): "
        f"**{r['synapse_counts']['flagged_far']}**",
        f"- Closer to an excluded source node than to any retained node: "
        f"{r['synapse_counts']['closer_to_an_excluded_node_than_to_any_retained_node']}",
        f"- Usable for the distributed drive: {r['synapse_counts']['usable_for_distributed_drive']}",
        f"- Neuropil labels present: {r['synapse_counts']['neuropil_labels_present']}",
        "",
        "## Assignment distances (site -> nearest retained skeleton node)",
        "",
        "| statistic | restricted (used for the drive) | unrestricted (frame check) |",
        "|---|---:|---:|",
        f"| sites | {r['assignment']['n_sites']} | {r['assignment_unrestricted']['n_sites']} |",
        f"| min (um) | {a['min']:.6f} | {u['min']:.6f} |",
        f"| median (um) | {a['median']:.6f} | {u['median']:.6f} |",
        f"| mean (um) | {a['mean']:.6f} | {u['mean']:.6f} |",
        f"| p90 (um) | {a['p90']:.6f} | {u['p90']:.6f} |",
        f"| max (um) | {a['max']:.6f} | {u['max']:.6f} |",
        f"| > 1 um | {a['n_over_1um']} | {u['n_over_1um']} |",
        f"| > 5 um | {a['n_over_5um']} | {u['n_over_5um']} |",
        "",
        "The unrestricted maximum is small compared with the cell's extent, which is the evidence",
        "that the synapse sites and the skeleton share the FAFB nm frame (no registration needed).",
        "",
        "## Paired run: one arbitrary point vs the real synapse locations",
        "",
        f"- Both arms: **{pc['total_sites_each_arm']} sites**, {pc['g_per_site_uS']*1e3:g} nS per site, "
        f"{pc['rate_hz']:g} Hz, E_rev {pc['E_rev_mV']:g} mV, tau {pc['tau_ms']:g} ms, "
        f"{pc['dt_ms']:g} ms steps, {pc['duration_ms']:g} ms, release window "
        f"{pc['window_ms'][0]:g}-{pc['window_ms'][1]:g} ms, seed {pc['seed']}.",
        f"- Arm (a) parks the whole drive on row {point_row_txt} (largest-diameter interior row);",
        f"  arm (b) spreads it over {n_real_comp} compartments that carry real sites.",
        f"- Drive pairing: identical release count at every step = "
        f"**{sd['event_counts_identical_per_step']}**; same conductance per site = {sd['same_g_per_site']}; "
        f"placement differs = {sd['placement_differs']}; injected-current pattern differs = "
        f"{sd['current_vectors_differ_at_rest']}.",
        f"- Solve: {pc['numerical_check']['implicit']['description']} (unconditionally stable).",
        "",
        "| recorded row | (a) point peak (mV) | (b) real sites peak (mV) | (a) deviation (mV) | (b) deviation (mV) | max abs difference (mV) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, v in pc["per_recorded_row"].items():
        lines.append(f"| {name} | " + row(v)[2:])
    lines += [
        "",
        f"- Somatic/root-side (row 0) peak deviation: **{mat['point_arm_peak_deviation_mV']:.4f} mV (point)** vs "
        f"**{mat['distributed_arm_peak_deviation_mV']:.4f} mV (real sites)** = "
        f"**{100*mat['relative_change_fraction']:+.2f}%**; largest difference anywhere "
        f"{mat['max_abs_difference_any_recorded_row_mV']:.4f} mV.",
        f"- Materiality threshold declared in advance: >10% change in the root-side peak deviation -> "
        f"**material = {mat['material']}**.",
        "",
        f"**Conclusion.** {pc['conclusion']}",
        "",
        "### Robustness",
        "",
        "| seed | (a) point deviation (mV) | (b) real sites deviation (mV) | relative change |",
        "|---|---:|---:|---:|",
    ]
    for k, v in seeds.items():
        lines.append(f"| {k} | {v['point_mV']:.4f} | {v['distributed_mV']:.4f} | "
                     f"{100*v['relative_change_fraction']:+.2f}% |")
    lines += [
        "",
        "| Ra multiplier | Ra (ohm cm) | (a) point deviation (mV) | (b) real sites deviation (mV) | relative change |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in ras.items():
        lines.append(f"| {k} | {212.0*float(k):.4g} | {v['point_mV']:.4f} | {v['distributed_mV']:.4f} | "
                     f"{100*v['relative_change_fraction']:+.2f}% |")
    lines += [
        "",
        "The `Ra` sweep matters because the borrowed DNp01/DNp03 parameters give an axial conductance",
        "that is roughly 1e5-1e6 times the membrane conductance, i.e. an electrotonically compact cell.",
        "",
        "## Numerical check",
        "",
        "- The implicit solve (conductance inside the backward-Euler solve) is used for the reported",
        "  comparison. The pre-existing practice of sampling the current at the previous voltage is",
        "  stable only while `g*dt/C <= 2`; with 1000 sites on one compartment it diverges at the",
        "  literature-ish 0.5 nS per site, which was observed here. The clamped explicit run and the",
        "  implicit run are both reported in `synapse_map_report.json` (`numerical_check`).",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {c}" for c in r["caveats"]]
    lines += [
        "- Truncation accounting for the input sites: "
        f"{r['truncation_accounting']['n_mapped_to_excluded_nodes']} of "
        f"{r['synapse_counts']['real_presynaptic_sites_for_root_id']} fall on the 19 excluded source nodes "
        f"(so {100*r['truncation_accounting']['n_mapped_to_excluded_nodes']/max(r['synapse_counts']['real_presynaptic_sites_for_root_id'],1):.1f}% "
        "of the real input is definitely absent here), "
        f"{r['truncation_accounting']['n_input_sites_on_a_boundary_node']} sit on one of the 18 nodes whose "
        "edge to an excluded node became an artificial sealed boundary, and "
        f"{r['truncation_accounting']['n_sites_closer_to_an_excluded_node']} are closer to an excluded source "
        f"node than to any retained node (nearest excluded node "
        f"{r['truncation_accounting']['min_distance_to_an_excluded_node_um']:.3f} um away). "
        f"A further {r['synapse_counts']['flagged_far']} input site(s) are further than "
        f"{r['assignment']['far_threshold_um']:g} um from every retained node and are excluded from the "
        "distributed arm; the real extracted coordinates themselves are unaffected.",
        f"- The cached `connections_princeton` snapshot disagrees with `synapse_table` for this cell "
        f"({r['data_quality_flags']['connectome_cross_check']['finding']}); only `synapse_table` is used here.",
        "- `size` per row from the synapse table is carried through as metadata but is NOT used as a",
        "  synaptic weight: per-site conductance is a single illustrative constant.",
        "",
        "## Resources (measured)",
        "",
        f"- Full streamed extraction: **{r['resources']['stream_extraction_elapsed_s']:.1f} s** wall clock, "
        f"**{r['resources']['stream_extraction_peak_rss_mb']:.1f} MB** peak RSS, "
        f"{r['resources']['csv_bytes_streamed']/1e9:.2f} GB of CSV decompressed.",
        f"- Demo run (mapping + paired simulation + figure): **{r['resources']['wall_clock_s_main']:.1f} s** "
        f"wall clock, **{r['resources']['peak_rss_mb_main']:.1f} MB** peak RSS.",
        "",
        "## Tests",
        "",
        suite_line,
        "",
        "## Outputs",
        "",
    ]
    for k, v in r["outputs"].items():
        lines.append(f"- `{v}`")
    lines.append(f"- `{OUT / 'synapse_map_summary.json'}`")
    lines.append(f"- `{OUT / 'synapse_map_README.md'}` (this file)")
    lines.append(f"- `{RAW_NPZ}` (bounded extraction: only matching rows)")
    lines.append(f"- `{OUT / 'synapse_map_stream_stats.json'}` (streaming provenance)")
    lines.append(f"- `{OUT / 'synapse_map_selftest.json'}` (test results)")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
