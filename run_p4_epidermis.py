"""P4 runner: self-organising stratified epidermis.

Experiments (all actually run):
  A. control            : hypoxia coupling ON, reference cell-cycle time
  B. fast proliferation : cell-cycle time halved  -> tests thickness scaling
  C. slow proliferation : cell-cycle time doubled
  D. no hypoxia coupling: the O2 gradient is switched off as a driver

Prediction under test: at steady state the thickness is set by
(proliferation rate) x (transit time through the layers), so shortening the
cell-cycle time must thicken the tissue roughly in proportion.

Outputs:
  outputs/p4_epidermis_panels.png
  outputs/p4_epidermis_curves.png
  outputs/p4_epidermis_gradient_sweep.png
  outputs/p4_epidermis_metrics.json
  outputs/p4_frames/*.png
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch

from epidermis import Epidermis, TYPE_COLORS, TYPE_NAMES
from viz import type_image, save_panels_png, write_gif

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FRAMES = os.path.join(OUT, "p4_frames")
os.makedirs(FRAMES, exist_ok=True)

SNAPSHOTS = 6


def run_case(tag, n_mcs=9000, hypoxia=True, params=None, seed=0, snapshots=True):
    e = Epidermis(Lx=180, Ly=240, seed=seed, hypoxia_coupling=hypoxia, params=params)
    frames, times = [], []
    every = max(1, n_mcs // SNAPSHOTS)
    done = 0
    t0 = time.time()
    while done < n_mcs:
        chunk = min(every, n_mcs - done)
        e.run(chunk, record_every=max(50, n_mcs // 40), verbose=False)
        done += chunk
        if snapshots:
            img = type_image(e.m, TYPE_COLORS)
            p = os.path.join(FRAMES, f"{tag}_{e.m.mcs:05d}.png")
            fig, ax = plt.subplots(figsize=(3.8, 5.0), dpi=105)
            ax.imshow(img, interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            h = e.history[-1]
            ax.set_title(f"mcs {e.m.mcs}  thick {h['thickness_um']:.0f} um  "
                         f"{h['n_cells']} cells", fontsize=8)
            fig.tight_layout(); fig.savefig(p); plt.close(fig)
            frames.append(p); times.append(e.m.mcs)
        h = e.history[-1]
        print(f"  [{tag}] mcs={e.m.mcs} thick={h['thickness_um']:.1f}um "
              f"cells={h['n_cells']} layers={ {k: round(v) for k, v in h['layer_thickness_um'].items()} } "
              f"div={h['n_divisions_total']} shed={h['n_shed_total']} "
              f"o2surf={h['o2_at_surface']:.2f}", flush=True)
    e.wall = time.time() - t0
    return e, frames, times


def main():
    cases = {}
    e_ctrl, fr_ctrl, ts_ctrl = run_case("ctrl", n_mcs=9000)
    e_fast, _, _ = run_case("fast", n_mcs=9000, params={"cycle_mcs": 200.0},
                            snapshots=False)
    e_slow, _, _ = run_case("slow", n_mcs=9000, params={"cycle_mcs": 800.0},
                            snapshots=False)
    e_nohyp, _, _ = run_case("nohyp", n_mcs=9000, hypoxia=False, snapshots=False)

    # ---------- panels ----------
    panels = [{"img": np.asarray(Image.open(p)), "title": f"mcs {t}"}
              for p, t in zip(fr_ctrl, ts_ctrl)]
    panels.append({"img": e_ctrl.o2.value.reshape(e_ctrl.Ly, e_ctrl.Lx),
                   "title": "O2 (final)", "cmap": "viridis", "colorbar": True})
    save_panels_png(os.path.join(OUT, "p4_epidermis_panels.png"), panels[:7],
                    ncols=4, figsize=(14, 8.2), dpi=130,
                    suptitle="P4 self-organising stratified epidermis "
                             "(basal=dark blue, spinous=light blue, "
                             "granular=yellow, cornified=red)")
    write_gif(os.path.join(OUT, "p4_epidermis.gif"), fr_ctrl, duration_ms=450)

    # ---------- curves ----------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=130)
    xs = [h["mcs"] for h in e_ctrl.history]
    axes[0].plot(xs, [h["thickness_um"] for h in e_ctrl.history], lw=1.9, label="total")
    for name in ("basal", "spinous", "granular", "cornified"):
        axes[0].plot(xs, [h["layer_thickness_um"][name] for h in e_ctrl.history],
                     lw=1.3, label=name)
    axes[1].plot(xs, [h["n_divisions_total"] for h in e_ctrl.history], lw=1.9,
                 label="divisions (cumulative)")
    axes[1].plot(xs, [h["n_shed_total"] for h in e_ctrl.history], lw=1.9,
                 label="shed (cumulative)")
    axes[1].plot(xs, [h["n_cells"] for h in e_ctrl.history], lw=1.9, label="cells")
    prof = e_ctrl.o2.value.mean(axis=1)
    axes[2].plot(prof, np.arange(e_ctrl.Ly) * e_ctrl.h, lw=1.9)
    axes[2].set_xlabel("O2 (dimensionless)"); axes[2].set_ylabel("height above basement (um)")
    axes[0].set_xlabel("MCS"); axes[0].set_ylabel("layer vertical extent (um)")
    axes[1].set_xlabel("MCS"); axes[1].set_ylabel("count")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title("stratification", fontsize=9)
    axes[1].set_title("turnover", fontsize=9)
    axes[2].set_title("O2 falls with height", fontsize=9)
    fig.suptitle("P4 stratified epidermis: layer formation, turnover and the O2 gradient",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "p4_epidermis_curves.png")); plt.close(fig)

    # ---------- proliferation -> thickness scaling ----------
    labels, thicks, cells = [], [], []
    for e, lab in ((e_slow, "cycle x2 (slow)"), (e_ctrl, "cycle x1 (control)"),
                   (e_fast, "cycle x0.5 (fast)")):
        labels.append(lab)
        thicks.append(e.history[-1]["thickness_um"])
        cells.append(e.history[-1]["n_cells"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=130)
    axes[0].bar(labels, thicks, color=["tab:green", "tab:blue", "tab:orange"])
    axes[0].set_ylabel("steady thickness (um)")
    axes[0].set_title("thickness vs proliferation rate", fontsize=9)
    axes[1].bar(labels, cells, color=["tab:green", "tab:blue", "tab:orange"])
    axes[1].set_ylabel("cell number")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("P4 test: doubling the cell-cycle time should thin the tissue",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "p4_epidermis_gradient_sweep.png"))
    plt.close(fig)

    res = {
        "control": {k: e_ctrl.history[-1][k] for k in
                    ("mcs", "n_cells", "thickness_um", "layer_thickness_um",
                     "o2_at_base", "o2_at_surface", "o2_min",
                     "height_stage_correlation", "n_divisions_total",
                     "n_shed_total", "counts")},
        "fast_proliferation_cycle_200": {
            "thickness_um": e_fast.history[-1]["thickness_um"],
            "n_cells": e_fast.history[-1]["n_cells"],
            "n_divisions_total": e_fast.history[-1]["n_divisions_total"]},
        "slow_proliferation_cycle_800": {
            "thickness_um": e_slow.history[-1]["thickness_um"],
            "n_cells": e_slow.history[-1]["n_cells"],
            "n_divisions_total": e_slow.history[-1]["n_divisions_total"]},
        "no_hypoxia_coupling": {
            "thickness_um": e_nohyp.history[-1]["thickness_um"],
            "n_cells": e_nohyp.history[-1]["n_cells"],
            "layer_thickness_um": e_nohyp.history[-1]["layer_thickness_um"],
            "height_stage_correlation": e_nohyp.history[-1]["height_stage_correlation"]},
        "thickness_scaling_test": {
            "cycle_200_thickness": e_fast.history[-1]["thickness_um"],
            "cycle_400_thickness": e_ctrl.history[-1]["thickness_um"],
            "cycle_800_thickness": e_slow.history[-1]["thickness_um"],
            "monotone_increasing_with_proliferation_rate":
                bool(e_fast.history[-1]["thickness_um"]
                     > e_ctrl.history[-1]["thickness_um"]
                     > e_slow.history[-1]["thickness_um"]),
        },
        "lineage_control": e_ctrl.lineage_check(),
        "history_control": e_ctrl.history,
        "wall_seconds": {"ctrl": e_ctrl.wall, "fast": e_fast.wall,
                         "slow": e_slow.wall, "nohyp": e_nohyp.wall},
        "params": {k: v[0] for k, v in e_ctrl.cfg.items()},
        "param_provenance": {k: v[1] for k, v in e_ctrl.cfg.items()},
        "caveats": [
            "There are NO measured skin parameters in this project's verified "
            "anchor file (MEASURED_ANCHORS.md covers the Drosophila notum). "
            "Every epidermis parameter is illustrative.",
            "Absolute thicknesses depend on the illustrative 1 um lattice "
            "spacing and on illustrative transit times; only the qualitative "
            "architecture, the turnover closure and the direction of the "
            "proliferation->thickness scaling are claims.",
        ],
    }
    with open(os.path.join(OUT, "p4_epidermis_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res["thickness_scaling_test"], indent=2))
    print("layer correlation:", res["control"]["height_stage_correlation"])


if __name__ == "__main__":
    main()
