"""run_p1b_merks.py -- replication of the published contact-inhibition result.

Follows Merks et al. 2008 (PLoS Comput Biol 4(9):e1000163).  See `merks.py`
for the parameter derivation from the paper's own numbers.

Two experiments, both taken from the paper:

  A. CLUSTER (the paper's Figs. 4 and 5).  A rounded cluster of 128 ECs.
     The paper reports a *phase transition* in compactness
     C = A_cluster / A_hull at a chemotactic sensitivity ratio
     chi(c,c)/chi(c,M) ~= 0.5: below it the cluster sprouts (low C), above it
     it stays rounded and compact (C near 1).  We scan the ratio and look for
     that transition.  This is a quantitative, falsifiable prediction.

  B. DE NOVO (the paper's Fig. 2).  Dispersed ECs self-organise.  The network
     signature is the number of cell-free LACUNAE fully enclosed by EC.

Before the scan we calibrate the chemotactic sensitivity lambda_chem, because
the paper does not state its absolute value: a cluster must at least stay
together (chemotactic aggregation must beat dispersal).  Without that step the
model is not even in the paper's regime.

Outputs:
  outputs/p1b_merks_lambda.png      calibration
  outputs/p1b_merks_clusters.png    cluster morphology vs chi ratio
  outputs/p1b_merks_transition.png  compactness vs chi ratio (the phase test)
  outputs/p1b_merks_denovo.png      de novo network
  outputs/p1b_merks_metrics.json
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

from merks import MerksVasculogenesis, UM_PER_SITE, SECONDS_PER_MCS, TYPE_COLORS, \
    DIFFUSION_LENGTH_SITES, CELL_SITES
from viz import type_image, save_panels_png

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
TRIALS = os.path.join(OUT, "merks_trials")
os.makedirs(TRIALS, exist_ok=True)

L = 200
N_CELLS = 128          # the paper's cluster size
MCS = 6000             # the paper uses 10,000; reduced for runtime (noted)
CHI_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)


def run_cluster(lam, chi, mcs=MCS, tag=None, seed=1, n_cells=N_CELLS, L=L):
    v = MerksVasculogenesis(L=L, seed=seed, mode="cluster", n_cells=n_cells,
                            chem_cc_ratio=chi,
                            params={"lambda_chem": lam})
    v.run(mcs, record_every=max(100, mcs // 12), verbose=False)
    h = v.history[-1]
    h["lambda_chem"] = lam
    h["seeded_cells"] = v.n_made
    return v, h


def main():
    res = {"setup": {
        "source": "Merks et al. 2008, PLoS Comput Biol 4(9):e1000163",
        "um_per_site": UM_PER_SITE, "seconds_per_mcs": SECONDS_PER_MCS,
        "cell_sites": CELL_SITES,
        "diffusion_length_sites": DIFFUSION_LENGTH_SITES,
        "diffusion_length_um": DIFFUSION_LENGTH_SITES * UM_PER_SITE,
        "lattice": L, "n_cells": N_CELLS, "mcs": MCS,
        "mcs_note": "the paper reports compactness at 10,000 MCS; we use 6000",
    }}

    # ---------------- calibration of lambda_chem ----------------
    cal = []
    for lam in (20.0, 100.0, 400.0, 1600.0):
        v, h = run_cluster(lam, 0.0, mcs=1500)
        cal.append(h)
        print(f"  [cal] lambda={lam:7.1f} EC={h['n_ec']} comp={h['n_components']:4d} "
              f"C={h['compactness']} lacunae={h['n_lacunae']}", flush=True)

    # choose the smallest lambda at which the cluster stays largely connected
    best = None
    for h in cal:
        if h["n_components"] <= max(3, 0.1 * N_CELLS):
            best = h["lambda_chem"]
            break
    if best is None:
        # fall back to the most-connected case, and say so
        best = min(cal, key=lambda h: h["n_components"])["lambda_chem"]
        fell_back = True
    else:
        fell_back = False
    res["calibration"] = {"rows": cal, "chosen_lambda_chem": best,
                          "fell_back_to_most_connected": fell_back}
    print("chosen lambda_chem =", best, "fell_back =", fell_back, flush=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=130)
    ax.plot([h["lambda_chem"] for h in cal], [h["n_components"] for h in cal],
            "o-", label="EC connected components")
    ax.axhline(max(3, 0.1 * N_CELLS), ls="--", c="r", lw=1,
               label="aggregation threshold")
    ax.set_xscale("log")
    ax.set_xlabel("lambda_chem (chemotactic sensitivity, illustrative)")
    ax.set_ylabel("connected components")
    ax.set_title(f"Calibration: {N_CELLS}-cell cluster must stay aggregated", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "p1b_merks_lambda.png")); plt.close(fig)

    # ---------------- the chi-ratio scan (the published prediction) ----------
    scan = []
    imgs = {}
    for chi in CHI_RATIOS:
        t0 = time.time()
        v, h = run_cluster(best, chi)
        h["wall_s"] = round(time.time() - t0, 1)
        scan.append(h)
        img = type_image(v.m, TYPE_COLORS)
        fp = os.path.join(TRIALS, f"chi_{chi:.2f}.png")
        fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=110)
        ax.imshow(img, interpolation="nearest", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"chi(c,c)/chi(c,M) = {chi:.2f}\nC = "
                     f"{h['compactness']:.3f}  lacunae {h['n_lacunae']}  "
                     f"comp {h['n_components']}", fontsize=8)
        fig.tight_layout(); fig.savefig(fp); plt.close(fig)
        imgs[chi] = fp
        print(f"  [scan] chi={chi:.2f} C={h['compactness']} comp={h['n_components']} "
              f"lacunae={h['n_lacunae']} ({h['wall_s']}s)", flush=True)

    panels = [{"img": np.asarray(Image.open(imgs[c])), "title": ""} for c in CHI_RATIOS]
    save_panels_png(os.path.join(OUT, "p1b_merks_clusters.png"), panels, ncols=5,
                    figsize=(15.5, 3.6), dpi=130,
                    suptitle="Merks 2008 cluster experiment: morphology vs the "
                             "relative chemotactic sensitivity at cell-cell interfaces "
                             "(chi=0 is full contact inhibition)")

    chi_x = [h["chem_cc_ratio"] for h in scan]
    C_y = [h["compactness"] for h in scan]
    lac_y = [h["n_lacunae"] for h in scan]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=130)
    axes[0].plot(chi_x, C_y, "o-", lw=2, color="tab:blue")
    axes[0].axvline(0.5, ls="--", c="r", lw=1.2,
                    label="paper's reported transition ~0.5")
    axes[0].set_xlabel("chi(c,c) / chi(c,M)")
    axes[0].set_ylabel("compactness  C = A_cluster / A_hull")
    axes[0].set_title("compactness vs contact inhibition", fontsize=9)
    axes[1].plot(chi_x, lac_y, "s-", lw=2, color="tab:green")
    axes[1].set_xlabel("chi(c,c) / chi(c,M)")
    axes[1].set_ylabel("number of enclosed lacunae")
    axes[1].set_title("network signature (lacunae)", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Merks 2008 phase-transition test (paper: transition at ratio ~0.5; "
                 "sprouting = LOW compactness)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "p1b_merks_transition.png"))
    plt.close(fig)

    # direction test: is compactness lower at lower ratio (more contact inhibition)?
    lo = np.mean([c for r, c in zip(chi_x, C_y) if r <= 0.25 and c is not None])
    hi = np.mean([c for r, c in zip(chi_x, C_y) if r >= 0.75 and c is not None])
    res["transition_test"] = {
        "scan": scan,
        "mean_compactness_low_ratio": float(lo),
        "mean_compactness_high_ratio": float(hi),
        "contact_inhibition_lowers_compactness": bool(lo < hi),
        "paper_prediction": "contact inhibition (low ratio) should give sprouting, "
                            "i.e. LOWER compactness",
    }
    print("transition test:", res["transition_test"]["contact_inhibition_lowers_compactness"],
          "low", round(lo, 3), "high", round(hi, 3), flush=True)

    # ---------------- de novo ----------------
    dn = []
    for chi in (0.0, 1.0):
        v = MerksVasculogenesis(L=230, seed=2, mode="de_novo", n_cells=400,
                                chem_cc_ratio=chi, params={"lambda_chem": best})
        v.run(4000, record_every=500, verbose=False)
        h = v.history[-1]
        h["seeded_cells"] = v.n_made
        dn.append(h)
        img = type_image(v.m, TYPE_COLORS)
        fp = os.path.join(TRIALS, f"denovo_chi_{chi:.2f}.png")
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=110)
        ax.imshow(img, interpolation="nearest", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"de novo  chi={chi:.2f}  seeded {v.n_made}\n"
                     f"lacunae {h['n_lacunae']}  comp {h['n_components']}", fontsize=8)
        fig.tight_layout(); fig.savefig(fp); plt.close(fig)
        print(f"  [denovo] chi={chi:.2f} seeded={v.n_made} EC={h['n_ec']} "
              f"comp={h['n_components']} lacunae={h['n_lacunae']} C={h['compactness']}",
              flush=True)

    panels = []
    for chi in (0.0, 1.0):
        fp = os.path.join(TRIALS, f"denovo_chi_{chi:.2f}.png")
        panels.append({"img": np.asarray(Image.open(fp)), "title": ""})
    save_panels_png(os.path.join(OUT, "p1b_merks_denovo.png"), panels, ncols=2,
                    figsize=(9.5, 5.0), dpi=130,
                    suptitle="Merks 2008 de novo vasculogenesis: dispersed ECs, "
                             "chi=0 (left, contact inhibition) vs chi=1 (right)")
    res["de_novo"] = dn

    res["params"] = {}
    res["note"] = ("lambda_chem (chemotactic sensitivity) and the temperature are "
                   "illustrative: the paper does not state their absolute values. "
                   "Everything else in `merks.py` is derived from the paper's "
                   "stated numbers.")
    with open(os.path.join(OUT, "p1b_merks_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote metrics")


if __name__ == "__main__":
    main()
