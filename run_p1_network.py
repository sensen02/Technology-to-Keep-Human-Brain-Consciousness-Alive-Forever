"""run_p1_network.py -- P1 headline: spontaneous vascular network formation.

This is the de novo vasculogenesis experiment of Merks et al. 2008 (their
Fig. 2): dispersed endothelial cells, each secreting a rapidly-decaying
chemoattractant, migrating up its gradient, with chemotaxis suppressed at
cell-cell interfaces (contact inhibition).

The published control is the same simulation with the contact-inhibition
sensitivity ratio raised to 1 (no contact inhibition): the paper reports that
the cells then form "disconnected vascular islands rather than a vascular
network".

We report, for each configuration and several seeds:
  * number of EC connected components (fewer = more connected plexus)
  * number of enclosed cell-free lacunae (the network signature)
  * compactness C = A_cluster / A_hull (the paper's own metric)

and we render the actual lattices.  The direction of the control is reported
as measured, whichever way it comes out.

Parameters come from `merks.py`, where each is derived from the paper's stated
numbers (2 um pixels, 1 MCS = 30 s, 200 um^2 cells, D = 1e-13 m^2/s,
alpha = epsilon = 1e-3 /s -> diffusion length exactly 10 um = 5 sites).
`lambda_chem` and the temperature are NOT stated in the paper and are
illustrative; they were fixed by the calibration in `run_p1b_merks.py`.
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

from merks import MerksVasculogenesis, TYPE_COLORS, UM_PER_SITE, SECONDS_PER_MCS
from viz import type_image, save_panels_png, write_gif

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FRAMES = os.path.join(OUT, "p1_network_frames")
os.makedirs(FRAMES, exist_ok=True)

L = 230
N_EC = 400
MCS = 6000
LAM = 400.0
SEEDS = (2, 11, 23)


def run_one(chi, seed, snapshots=False, tag=""):
    v = MerksVasculogenesis(L=L, seed=seed, mode="de_novo", n_cells=N_EC,
                            chem_cc_ratio=chi, params={"lambda_chem": LAM})
    frames, times = [], []
    every = MCS // 4
    done = 0
    t0 = time.time()
    while done < MCS:
        chunk = min(every, MCS - done)
        v.run(chunk, record_every=max(200, MCS // 12), verbose=False)
        done += chunk
        if snapshots:
            img = type_image(v.m, TYPE_COLORS)
            p = os.path.join(FRAMES, f"{tag}_{v.m.mcs:05d}.png")
            fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=110)
            ax.imshow(img, interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            h = v.history[-1]
            ax.set_title(f"mcs {v.m.mcs}  ({h['hours']:.0f} h)\n"
                         f"lacunae {h['n_lacunae']}  comp {h['n_components']}", fontsize=8)
            fig.tight_layout(); fig.savefig(p); plt.close(fig)
            frames.append(p); times.append(v.m.mcs)
    h = v.history[-1]
    h["wall_s"] = round(time.time() - t0, 1)
    h["seeded_cells"] = v.n_made
    h["chem_cc_ratio"] = chi
    h["seed"] = seed
    print(f"  chi={chi:.2f} seed={seed}: EC={h['n_ec']} comp={h['n_components']} "
          f"lacunae={h['n_lacunae']} C={h['compactness']:.3f} ({h['wall_s']}s)", flush=True)
    return v, h, frames, times


def main():
    rows = []
    v_ci, h_ci, fr, ts = run_one(0.0, SEEDS[0], snapshots=True, tag="chi0")
    rows.append(h_ci)
    for seed in SEEDS[1:]:
        _, h, _, _ = run_one(0.0, seed)
        rows.append(h)
    for seed in SEEDS:
        _, h, _, _ = run_one(1.0, seed)
        rows.append(h)

    def agg(chi, key):
        vals = [r[key] for r in rows if r["chem_cc_ratio"] == chi and r[key] is not None]
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    summary = {}
    for chi in (0.0, 1.0):
        summary[f"chi_{chi:.1f}"] = {
            k: {"mean": agg(chi, k)[0], "std": agg(chi, k)[1], "n": agg(chi, k)[2]}
            for k in ("n_components", "n_lacunae", "compactness", "ec_area_um2")
        }

    # ---------------- figures ----------------
    panels = [{"img": np.asarray(Image.open(p)), "title": f"mcs {t} ({t*SECONDS_PER_MCS/3600:.0f} h)"}
              for p, t in zip(fr, ts)]
    save_panels_png(os.path.join(OUT, "p1_network_timeseries.png"), panels, ncols=2,
                    figsize=(9.6, 9.6), dpi=130,
                    suptitle="P1 spontaneous vascular network (de novo vasculogenesis, "
                             "Merks et al. 2008 mechanism): cords of endothelial cells "
                             "enclosing cell-free lacunae")
    write_gif(os.path.join(OUT, "p1_network.gif"), fr, duration_ms=500)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=130)
    for k, lab in (("n_components", "EC connected components"),
                   ("n_lacunae", "enclosed cell-free lacunae"),
                   ("compactness", "compactness C")):
        m0, s0, _ = agg(0.0, k)
        m1, s1, _ = agg(1.0, k)
        axes[list(("n_components", "n_lacunae", "compactness")).index(k)].bar(
            ["contact\ninhibition ON\n(chi=0)", "no contact\ninhibition\n(chi=1)"],
            [m0, m1], yerr=[s0, s1], color=["tab:blue", "tab:red"], capsize=4)
        axes[list(("n_components", "n_lacunae", "compactness")).index(k)].set_ylabel(lab)
        axes[list(("n_components", "n_lacunae", "compactness")).index(k)].grid(alpha=0.3, axis="y")
    fig.suptitle(f"P1 de novo vasculogenesis, mean over {len(SEEDS)} seeds "
                 f"({MCS} MCS = {MCS*SECONDS_PER_MCS/3600:.0f} h, {L}x{L} sites "
                 f"= {L*UM_PER_SITE:.0f} um square)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "p1_network_control.png")); plt.close(fig)

    res = {
        "experiment": "de novo vasculogenesis, Merks et al. 2008 mechanism",
        "source": "Merks RMH, Perryn ED, Shirinifard A, Glazier JA (2008) "
                  "PLoS Comput Biol 4(9):e1000163, doi:10.1371/journal.pcbi.1000163",
        "setup": {"lattice": L, "um_per_site": UM_PER_SITE,
                  "seconds_per_mcs": SECONDS_PER_MCS, "mcs": MCS,
                  "simulated_hours": MCS * SECONDS_PER_MCS / 3600,
                  "n_ec_requested": N_EC, "lambda_chem": LAM,
                  "seeds": list(SEEDS)},
        "per_run": rows,
        "summary": summary,
        "control_direction": {
            "contact_inhibition_gives_fewer_components":
                bool(summary["chi_0.0"]["n_components"]["mean"]
                     < summary["chi_1.0"]["n_components"]["mean"]),
            "paper_expectation": "without contact inhibition the cells should form "
                                 "disconnected vascular islands instead of a network",
        },
        "caveats": [
            "lambda_chem (chemotactic sensitivity) and the temperature are NOT "
            "stated in the paper and are illustrative; only their combination was "
            "calibrated (see run_p1b_merks.py), so the model is in the published "
            "regime but is not an exact numerical reproduction.",
            "The paper's de novo figure uses 1,000 cells over 700 um and 10,000 "
            "MCS; we use 400 cells over 460 um and 6,000 MCS for runtime.",
            "Compactness C is defined for a single cluster; for a domain-spanning "
            "network it is not the appropriate discriminator and is reported here "
            "for completeness only.",
        ],
    }
    with open(os.path.join(OUT, "p1_network_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res["summary"], indent=1))
    print("control:", res["control_direction"])


if __name__ == "__main__":
    main()
