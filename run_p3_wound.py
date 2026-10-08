"""P3 runner: epithelial wound closure, with and without proliferation.

Produces (all from the actual runs):
  outputs/p3_wound_panels.png      time series of the closing sheet
  outputs/p3_wound_curves.png      open area vs time, +/- proliferation
  outputs/p3_wound_metrics.json    every number + the two-phase fit
  outputs/p3_frames/*.png          raw frames
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

from wound_healing import Epithelium, TYPE_COLORS
from viz import type_image, save_panels_png, write_gif

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FRAMES = os.path.join(OUT, "p3_frames")
os.makedirs(FRAMES, exist_ok=True)

L = 130
N_MCS = 5000
SNAPSHOTS = 6


# Snapshot schedule deliberately biased EARLY: this wound closes within
# ~600 MCS, so evenly spaced snapshots would all show an already-closed sheet.
SNAP_AT = (0, 40, 80, 150, 250, 400, 700, 1200, 2500, 5000)


def run_case(tag, proliferative, radius_um=20.0, seed=1, n_mcs=N_MCS, **params):
    e = Epithelium(L=L, seed=seed, proliferative=proliferative, params=params or None)
    e.clamp_outer_ring(2)
    e.m.step(150, field_every=0)
    info = e.wound(radius_um=radius_um)
    frames, times = [], []
    done = 0
    t0 = time.time()
    schedule = [t for t in SNAP_AT if t <= n_mcs] + [n_mcs]
    for target in schedule:
        if target <= done:
            continue
        e.run(target - done, record_every=max(25, n_mcs // 60), verbose=False)
        done = target
        img = type_image(e.m, TYPE_COLORS)
        p = os.path.join(FRAMES, f"{tag}_{e.m.mcs:05d}.png")
        fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=110)
        ax.imshow(img, interpolation="nearest", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"mcs {e.m.mcs}   closed {100*e.history[-1]['closed_fraction']:.0f}%   "
                     f"cells {e.history[-1]['n_cells']}", fontsize=8)
        fig.tight_layout(); fig.savefig(p); plt.close(fig)
        frames.append(p); times.append(e.m.mcs)
        print(f"  [{tag}] mcs={e.m.mcs} closed={e.history[-1]['closed_fraction']:.3f} "
              f"cells={e.history[-1]['n_cells']} div={e.history[-1]['n_divisions_total']} "
              f"open={e.history[-1]['open_area_um2']:.1f}um2", flush=True)
    e.wall = time.time() - t0
    return e, frames, times, info


def main():
    res = {}
    e_on, fr_on, ts_on, info = run_case("prolif_on", True)
    e_off, fr_off, ts_off, _ = run_case("prolif_off", False)

    fit_on = e_on.two_phase_fit()
    fit_off = e_off.two_phase_fit()

    # ---------- panels: the closing wound ----------
    panels = [{"img": np.asarray(Image.open(p)), "title": f"mcs {t}"}
              for p, t in zip(fr_on, ts_on)][:6]
    save_panels_png(os.path.join(OUT, "p3_wound_panels.png"), panels, ncols=3,
                    figsize=(13, 8.6), dpi=130,
                    suptitle="P3 epithelial wound closure (40 um wound, "
                             "proliferation ON)")
    write_gif(os.path.join(OUT, "p3_wound.gif"), fr_on, duration_ms=450)

    # ---------- curves ----------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=130)
    for e, lab, col in ((e_on, "proliferation ON", "tab:blue"),
                        (e_off, "proliferation OFF", "tab:red")):
        xs = [h["mcs"] for h in e.history]
        axes[0].plot(xs, [h["open_area_um2"] for h in e.history], label=lab,
                     color=col, lw=1.9)
        axes[1].plot(xs, [h["closed_fraction"] * 100 for h in e.history],
                     label=lab, color=col, lw=1.9)
        axes[2].plot(xs, [h["n_cells"] for h in e.history], label=lab, color=col, lw=1.9)
    for ax, lab in zip(axes, ["open wound area (um^2)", "wound closed (%)",
                              "number of cells"]):
        ax.set_xlabel("MCS"); ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title("closure kinetics", fontsize=9)
    axes[2].set_title("area supply: cell division", fontsize=9)
    fig.suptitle("P3 wound closure: the role of proliferation "
                 f"(initial wound {info['area_um2']:.0f} um^2, "
                 f"radius {info['radius_um']:.0f} um)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "p3_wound_curves.png")); plt.close(fig)

    res = {
        "geometry": {"cell_area_um2_measured": 55.6,
                     "lattice_spacing_um": e_on.lattice_spacing_um,
                     "wound_radius_um": info["radius_um"],
                     "wound_area_um2": info["area_um2"],
                     "wound_diameter_um_measured_anchor": 40.0},
        "proliferation_on": {"closed_fraction_final": e_on.history[-1]["closed_fraction"],
                             "closed_98pct_at_mcs": e_on.closure_mcs_98,
                             "n_cells_final": e_on.history[-1]["n_cells"],
                             "divisions_total": e_on.history[-1]["n_divisions_total"],
                             "two_phase_fit": fit_on,
                             "wall_seconds": e_on.wall},
        "proliferation_off": {"closed_fraction_final": e_off.history[-1]["closed_fraction"],
                              "closed_98pct_at_mcs": e_off.closure_mcs_98,
                              "n_cells_final": e_off.history[-1]["n_cells"],
                              "divisions_total": e_off.history[-1]["n_divisions_total"],
                              "two_phase_fit": fit_off,
                              "wall_seconds": e_off.wall},
        "history_proliferation_on": e_on.history,
        "history_proliferation_off": e_off.history,
        "params": {k: v[0] for k, v in e_on.cfg.items()},
        "param_provenance": {k: v[1] for k, v in e_on.cfg.items()},
        "measured_anchors_used": {
            "cell_area_um2": "222.3/4 um^2, notum 4-cell group, 12-13.5 hAPF (Curran 2017)",
            "wound_diameter_um": "~40 um, notum (PMC3718973, PMC13597085)",
            "closure_time_measured_min": "140 min to >360 min for notum wounds",
        },
        "caveats": [
            "The MCS -> minutes mapping is NOT calibrated here, so no closure "
            "time in minutes is claimed.",
            "The mechanical coefficients (lambda_fill, lambda_edge, lambda_V, J) "
            "are illustrative; no absolute force is measured in the notum.",
            "The measured wound anchor is from the notum at a different stage "
            "than the calibration quantities.",
        ],
    }
    with open(os.path.join(OUT, "p3_wound_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: res[k] for k in ("proliferation_on", "proliferation_off")},
                     indent=2)[:1600])


if __name__ == "__main__":
    main()
