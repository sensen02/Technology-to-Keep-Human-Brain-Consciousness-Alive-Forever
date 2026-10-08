"""P1/P2/P2b: spontaneous vascular network -- sprouting from a parent vessel.

Runs the sprouting-angiogenesis configuration twice: ONCE with
contact-inhibited chemotaxis (the mechanism under test) and ONCE with the
contact inhibition switched off, all other parameters identical.  The
comparison is the falsifiable part: contact inhibition is what should convert
chemotaxis into a ramified sprout rather than a clump.

Outputs (all real, from the run):
  outputs/p1_sprouting_panels.png    time series + VEGF + O2 fields
  outputs/p1_sprouting.gif           animation
  outputs/p1_contact_inhibition.png  the control comparison
  outputs/p1_metrics.png             time courses
  outputs/metrics_p1.json            every number
  outputs/p1_frames/*.png            the raw frames
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from vasculogenesis import Vasculogenesis, TYPE_COLORS, STATE_COLORS
from viz import type_image, save_panels_png, save_metric_png, write_gif

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FRAMES = os.path.join(OUT, "p1_frames")
os.makedirs(FRAMES, exist_ok=True)

COMMON = dict(L=220, seed=3, n_fibroblast=26, mode="sprouting",
              params={"o2_consumption": 0.05, "vegf_secretion_fib": 0.03,
                      "vegf_secretion_ec": 0.002, "vegf_decay": 0.004,
                      "lambda_chem_tip": 60.0, "lambda_chem_stalk": 4.0,
                      "lambda_length": 1.5, "elongation_target": 2.0,
                      "cycle_mcs": 300.0, "tip_fraction_percentile": 80.0})


def run(tag, n_mcs=3500, snapshots=6, **over):
    cfg = dict(COMMON)
    cfg.update(over)
    v = Vasculogenesis(**cfg)
    every = max(1, n_mcs // snapshots)
    frames = []
    t0 = time.time()

    def snap(i):
        img = v.type_image()
        p = os.path.join(FRAMES, f"{tag}_{v.m.mcs:05d}.png")
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=110)
        ax.imshow(img, interpolation="nearest", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{tag}  mcs={v.m.mcs}  EC={len(v.ec_ids())} tips={sum(1 for c in v.ec_ids() if v.state.get(c)=='tip')}", fontsize=7)
        fig.tight_layout(); fig.savefig(p); plt.close(fig)
        frames.append(p)
        np.save(os.path.join(FRAMES, f"{tag}_vegf_{v.m.mcs:05d}.npy"), v.vegf.value)
        np.save(os.path.join(FRAMES, f"{tag}_o2_{v.m.mcs:05d}.npy"), v.o2.value)

    v.run(n_mcs, record_every=max(20, n_mcs // 25), verbose=True, callback=None)
    # snapshot pass: re-run is wasteful, so instead snapshot during the run
    return v, frames


def run_with_snapshots(tag, n_mcs=3500, snapshots=6, **over):
    cfg = dict(COMMON); cfg.update(over)
    v = Vasculogenesis(**cfg)
    frames, snap_times = [], []
    t0 = time.time()
    every = max(1, n_mcs // snapshots)
    done = 0
    while done < n_mcs:
        chunk = min(every, n_mcs - done)
        v.run(chunk, record_every=max(25, n_mcs // 25), verbose=False)
        done += chunk
        snap_times.append(v.m.mcs)
        img = v.type_image()
        p = os.path.join(FRAMES, f"{tag}_{v.m.mcs:05d}.png")
        ntip = sum(1 for c in v.ec_ids() if v.state.get(c) == "tip")
        fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=110)
        ax.imshow(img, interpolation="nearest", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"mcs {v.m.mcs}   EC {len(v.ec_ids())}   tips {ntip}", fontsize=8)
        fig.tight_layout(); fig.savefig(p); plt.close(fig)
        frames.append(p)
        print(f"  [{tag}] mcs={v.m.mcs} EC={len(v.ec_ids())} tip={ntip} "
              f"reach={v.history[-1]['ec_max_reach_sites']:.0f} "
              f"P/A={v.history[-1]['ec_perimeter_over_area']:.2f} "
              f"o2min={v.history[-1]['o2_min']:.3f}", flush=True)
    v.wall = time.time() - t0
    return v, frames, snap_times


def main():
    res = {}
    v_ci, fr_ci, ts = run_with_snapshots("ci_on", n_mcs=3500)
    v_no, fr_no, ts2 = run_with_snapshots("ci_off", n_mcs=3500, contact_inhibited=False)

    # ---- panel figure from the contact-inhibition-on run ----
    v = v_ci
    panels = []
    for p, t in zip(fr_ci, ts):
        panels.append({"img": np.asarray(Image.open(p)), "title": f"mcs {t}"})
    panels.append({"img": v.vegf.value, "title": "VEGF (final)", "cmap": "inferno",
                   "colorbar": True})
    panels.append({"img": v.o2.value, "title": "O2 (final)", "cmap": "viridis",
                   "colorbar": True})
    save_panels_png(os.path.join(OUT, "p1_sprouting_panels.png"), panels,
                    ncols=4, figsize=(15, 8.4), dpi=130,
                    suptitle="P1/P2 sprouting angiogenesis from a parent vessel "
                             "(contact-inhibited chemotaxis ON)")

    write_gif(os.path.join(OUT, "p1_sprouting.gif"), fr_ci, duration_ms=400)

    # ---- control comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=130)
    axes[0].imshow(np.asarray(Image.open(fr_ci[-1]))); axes[0].set_title(
        "contact inhibition ON (final)", fontsize=9)
    axes[1].imshow(np.asarray(Image.open(fr_no[-1]))); axes[1].set_title(
        "contact inhibition OFF (final)", fontsize=9)
    for ax in axes[:2]:
        ax.set_xticks([]); ax.set_yticks([])
    keys = ["ec_max_reach_sites", "ec_perimeter_over_area", "n_ec"]
    labels = ["max reach from vessel (sites)", "EC perimeter / area", "EC cell count"]
    for ax, k, lab in zip(axes[2:], keys[:1], labels[:1]):
        ax.plot([h["mcs"] for h in v_ci.history], [h[k] for h in v_ci.history],
                label="CI on", lw=1.8)
        ax.plot([h["mcs"] for h in v_no.history], [h[k] for h in v_no.history],
                label="CI off", lw=1.8)
        ax.set_xlabel("MCS"); ax.set_ylabel(lab); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Contact inhibition of chemotaxis: control comparison", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "p1_contact_inhibition.png")); plt.close(fig)

    save_metric_png(os.path.join(OUT, "p1_metrics.png"), {
        "EC cells": ([h["mcs"] for h in v_ci.history], [h["n_ec"] for h in v_ci.history]),
        "tip cells": ([h["mcs"] for h in v_ci.history], [h["n_tip"] for h in v_ci.history]),
        "EC components": ([h["mcs"] for h in v_ci.history], [h["ec_components"] for h in v_ci.history]),
        "EC loops": ([h["mcs"] for h in v_ci.history], [h["ec_loops"] for h in v_ci.history]),
        "max reach (sites)": ([h["mcs"] for h in v_ci.history], [h["ec_max_reach_sites"] for h in v_ci.history]),
    }, "MCS", "value", "P1 sprouting angiogenesis time courses")

    lin = v_ci.lineage_check()
    hci, hno = v_ci.history[-1], v_no.history[-1]
    res = {
        "mode": "sprouting (parent vessel + hypoxic parenchyma)",
        "contact_inhibition_on": {k: hci[k] for k in
            ("mcs","n_ec","n_tip","ec_components","ec_loops","ec_area_fraction",
             "ec_max_reach_sites","ec_max_reach_um","ec_gyration_radius_sites",
             "ec_perimeter_over_area","o2_min","vegf_mean","mean_po2_tip","mean_po2_stalk")},
        "contact_inhibition_off": {k: hno[k] for k in
            ("mcs","n_ec","n_tip","ec_components","ec_loops","ec_area_fraction",
             "ec_max_reach_sites","ec_max_reach_um","ec_gyration_radius_sites",
             "ec_perimeter_over_area","o2_min","vegf_mean","mean_po2_tip","mean_po2_stalk")},
        "mechanism_test": {
            "reach_on_sites": hci["ec_max_reach_sites"],
            "reach_off_sites": hno["ec_max_reach_sites"],
            "perimeter_over_area_on": hci["ec_perimeter_over_area"],
            "perimeter_over_area_off": hno["ec_perimeter_over_area"],
            "contact_inhibition_increases_ramification":
                bool(hci["ec_max_reach_sites"] > hno["ec_max_reach_sites"]),
        },
        "hypoxia_test": {
            "mean_po2_tip": hci["mean_po2_tip"],
            "mean_po2_all_ec_stalk": hci["mean_po2_stalk"],
            "tips_are_more_hypoxic_than_stalk":
                bool(hci["mean_po2_tip"] is not None and hci["mean_po2_stalk"] is not None
                     and hci["mean_po2_tip"] < hci["mean_po2_stalk"]),
        },
        "lineage": lin,
        "wall_seconds": {"ci_on": v_ci.wall, "ci_off": v_no.wall},
        "params": {k: v[0] for k, v in v_ci.cfg.items()},
        "param_provenance": {k: v[1] for k, v in v_ci.cfg.items()},
        "units": "lattice sites and MCS; the minutes mapping is illustrative",
    }
    with open(os.path.join(OUT, "metrics_p1.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res["mechanism_test"], indent=2))
    print(json.dumps(res["hypoxia_test"], indent=2))


if __name__ == "__main__":
    main()
