"""run_p1d_combine.py -- analyse the fine chi scan and render its figures.

Reads outputs/p1d_shard_*.json (written by run_p1d_fine.py) and
outputs/p1c_shard_*.json (the coarse phase plane), and produces:

  outputs/p1d_merks_transition_fine.png   C vs chi with error bars (the phase test)
  outputs/p1d_merks_phaseplane.png        the (lambda_chem, lambda_volume) plane
  outputs/p1d_merks_metrics.json          all numbers

The reasoning in the text is written by the caller, not here; this script only
computes and plots what the runs actually produced.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def load_rows(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, pattern))):
        with open(f) as fh:
            rows += json.load(fh)
    return rows


def main():
    fine = load_rows("p1d_shard_*.json")
    coarse = load_rows("p1c_shard_*.json")
    if not fine:
        print("no fine-scan shards found yet")
        return

    chis = sorted({r["chi"] for r in fine})
    stats = []
    for c in chis:
        vals = np.array([r["compactness"] for r in fine
                         if r["chi"] == c and r["compactness"] is not None], dtype=float)
        comps = np.array([r["n_components"] for r in fine if r["chi"] == c], dtype=float)
        lacs = np.array([r["n_lacunae"] for r in fine if r["chi"] == c], dtype=float)
        stats.append({
            "chi": c, "n": int(vals.size),
            "compactness_mean": float(vals.mean()), "compactness_std": float(vals.std()),
            "components_mean": float(comps.mean()), "components_std": float(comps.std()),
            "lacunae_mean": float(lacs.mean()), "lacunae_std": float(lacs.std()),
        })

    # --- does compactness rise monotonically with chi (i.e. contact inhibition
    # --- gives the LOWER compactness = the paper's direction)?
    C = np.array([s["compactness_mean"] for s in stats])
    Cs = np.array([s["compactness_std"] for s in stats])
    x = np.array([s["chi"] for s in stats])
    lo = C[x <= 0.25].mean()
    hi = C[x >= 0.75].mean()
    # a permutation test on the low-vs-high split, using the per-run values
    low_runs = [r["compactness"] for r in fine if r["chi"] <= 0.25 and r["compactness"]]
    high_runs = [r["compactness"] for r in fine if r["chi"] >= 0.75 and r["compactness"]]
    allr = np.array(low_runs + high_runs, dtype=float)
    obs = np.mean(low_runs) - np.mean(high_runs)
    rng = np.random.default_rng(0)
    n_low = len(low_runs)
    perm = np.array([np.mean(p[:n_low]) - np.mean(p[n_low:])
                     for p in (rng.permutation(allr) for _ in range(20000))])
    pval = float(np.mean(np.abs(perm) >= abs(obs)))
    # also: is the trend across all chi monotone increasing?
    mono = bool(np.all(np.diff(C) >= -max(Cs) * 0.5))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=130)
    axes[0].errorbar(x, C, yerr=Cs, fmt="o-", lw=2, capsize=4, color="tab:blue")
    axes[0].axvline(0.5, ls="--", c="r", lw=1.2, label="paper's transition ~0.5")
    axes[0].set_xlabel("chi(c,c) / chi(c,M)   (0 = full contact inhibition)")
    axes[0].set_ylabel("compactness  C = A_cluster / A_hull")
    axes[0].set_title(f"lambda_chem={fine[0]['lambda_chem']:.0f}, "
                      f"lambda_volume={fine[0]['lambda_volume']:.0f}, "
                      f"{fine[0]['mcs']} MCS", fontsize=9)
    axes[0].legend(fontsize=8)

    axes[1].errorbar(x, [s["components_mean"] for s in stats],
                     yerr=[s["components_std"] for s in stats],
                     fmt="s-", lw=2, capsize=4, color="tab:green")
    axes[1].set_xlabel("chi(c,c) / chi(c,M)")
    axes[1].set_ylabel("EC connected components")
    axes[1].set_title("aggregation vs contact inhibition", fontsize=9)

    lv = sorted({r["lambda_volume"] for r in coarse})
    lc = sorted({r["lambda_chem"] for r in coarse})
    M = np.full((len(lv), len(lc)), np.nan)
    for i, a in enumerate(lv):
        for j, b in enumerate(lc):
            r0 = [r for r in coarse if r["lambda_volume"] == a and r["lambda_chem"] == b
                  and r["chem_cc_ratio"] == 0.0]
            r1 = [r for r in coarse if r["lambda_volume"] == a and r["lambda_chem"] == b
                  and r["chem_cc_ratio"] == 1.0]
            if r0 and r1:
                M[i, j] = r1[0]["compactness"] - r0[0]["compactness"]
    im = axes[2].imshow(M, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    axes[2].set_xticks(range(len(lc)), [f"{v:.0f}" for v in lc])
    axes[2].set_yticks(range(len(lv)), [f"{v:.0f}" for v in lv])
    axes[2].set_xlabel("lambda_chem"); axes[2].set_ylabel("lambda_volume")
    axes[2].set_title("C(chi=1) - C(chi=0)\nBLUE/positive = contact inhibition\ngave the less compact cluster (paper)", fontsize=9)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes[:2]:
        ax.grid(alpha=0.3)
    fig.suptitle("Merks 2008 cluster experiment: searching for the published "
                 "direction of the contact-inhibition control", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p1d_merks_transition_fine.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=130)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax.set_xticks(range(len(lc)), [f"{v:.0f}" for v in lc])
    ax.set_yticks(range(len(lv)), [f"{v:.0f}" for v in lv])
    ax.set_xlabel("lambda_chem (illustrative)")
    ax.set_ylabel("lambda_volume (illustrative)")
    ax.set_title("Phase plane: C(chi=1) - C(chi=0)\n"
                 "BLUE/positive = contact inhibition (chi=0) gives the LESS compact\n"
                 "cluster = the direction reported by Merks et al. 2008", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    for i, a in enumerate(lv):
        for j, b in enumerate(lc):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="k")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p1d_merks_phaseplane.png"))
    plt.close(fig)

    res = {
        "question": "does contact inhibition lower cluster compactness, as "
                    "reported by Merks et al. 2008 (their Fig. 5)?",
        "fine_scan": {"chi_values": chis, "seeds": sorted({r['seed'] for r in fine}),
                      "mcs": fine[0]["mcs"], "lambda_chem": fine[0]["lambda_chem"],
                      "lambda_volume": fine[0]["lambda_volume"], "per_chi": stats},
        "direction_test": {
            "mean_C_low_chi_le_0.25": float(lo),
            "mean_C_high_chi_ge_0.75": float(hi),
            "difference_low_minus_high": float(obs),
            "permutation_p_value": pval,
            "significance_note": "a permutation test over the individual runs",
            "monotone_increasing_with_chi": mono,
            "paper_direction_confirmed": bool(obs < 0 and pval < 0.05),
        },
        "coarse_phase_plane": {
            "lambda_volume_values": lv, "lambda_chem_values": lc,
            "C_chi1_minus_C_chi0": M.tolist(),
            # the paper's direction is C(chi=0) < C(chi=1), i.e. M > 0
            "n_configs_matching_paper_direction": int(np.sum(np.nan_to_num(M) > 0)),
            "n_configs": int(np.sum(~np.isnan(M))),
            "note": "M > 0 (blue) means contact inhibition gave the less compact "
                    "cluster, i.e. the published direction.  This coarse 3,000 MCS "
                    "scan is at 1 seed and is exploratory only: it is used to pick "
                    "the working point, and the direction there is then tested "
                    "properly in 'fine_scan' with 3 seeds and 6,000 MCS.",
        },
        "caveats": [
            "lambda_chem and lambda_volume are illustrative: the paper does not "
            "state their absolute values, so this is a search over an unknown "
            "working point, not a reproduction of a stated parameter set.",
            "The paper measures compactness at 10,000 MCS; these scans use "
            "3,000 (coarse) and 6,000 (fine) MCS.",
            "A positive result here would be a consistency check with the "
            "published direction; it would NOT be a biological validation.",
        ],
    }
    with open(os.path.join(OUT, "p1d_merks_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res["direction_test"], indent=2))
    print("phase plane:",
          res["coarse_phase_plane"]["n_configs_matching_paper_direction"], "/",
          res["coarse_phase_plane"]["n_configs"], "configs match the paper")
    for s in stats:
        print(f"  chi={s['chi']:.3f}  C={s['compactness_mean']:.4f} "
              f"+- {s['compactness_std']:.4f}  comp={s['components_mean']:.1f}")


if __name__ == "__main__":
    main()
