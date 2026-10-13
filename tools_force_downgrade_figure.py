#!/usr/bin/env python3
"""The contact-force channel is a NON-PHYSICAL LOAD, and that is now ENFORCED.  One figure.

The plan allows three states for this channel and forbids a fourth: fix the residual to about 1x,
or explicitly downgrade it and forbid physiological-magnitude conclusions from citing it.  The
forbidden state is the one this project was in -- large numbers, unidentified cause, still used as a
load -- and it was forbidden only in prose.  The figure shows the measurement that forces the choice,
the 1000x mislabel that was living in the names, and the two mechanisms that now make the downgrade
fail loudly instead of relying on a note.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_force_downgrade_figure.py --out <path.png>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Mentions `force_channel`: this figure plots the NON_PHYSICAL_LOAD channel's residual and the
# 1000x `_mN` mislabel that lived in scene.py's names, so it names the channel it is about.
# engine/embodied/force_channel.py is the declaration; no number here may be read as a load.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(HERE / "outputs" / "force_downgrade_audit.json"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "force_downgrade_figure.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(Path(args.audit).read_text())
    eps = [e for e in d["output_scan"]["episodes"] if e["mean_over_weight"] is not None]
    names = [e["episode"].replace("tactile_", "") for e in eps]
    mean = np.array([e["mean_over_weight"] for e in eps])

    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.46, wspace=0.30)

    # --- the residual, per episode
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(len(eps))
    ax.bar(x, mean, color="#c0392b", alpha=0.85)
    ax.axhline(1.0, color="darkgreen", lw=2.0, ls="--")
    ax.annotate("1x = what a body standing or walking\nat constant height should show",
                (0.15, 1.32), color="darkgreen", fontsize=8.5, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="darkgreen", alpha=0.92),
                zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("summed normal force / body weight")
    ax.set_title("the measured residual that forces the choice: 4.29-12.16x body weight on average "
                 "(peak 25.26-27.79x)", fontsize=10.2)
    for xi, m in zip(x, mean):
        ax.text(xi, m + 0.2, f"{m:.1f}", ha="center", fontsize=7.2)

    # --- the mislabel that was living in the names
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    ev = json.loads((HERE / "outputs" / "invariants.json").read_text())
    mis = None
    for r in ev.get("results", []):
        if r.get("check") == "force_unit_labels_are_honest":
            mis = (r.get("value") or {}).get("evidence")
    if mis is None:
        from tools_invariants import MISLABEL_EVIDENCE as mis                                        # noqa: N811
    ax2.text(0, 1,
             "THE 1000x MISLABEL LIVING IN A NAME\n\n"
             "scene_selftest.json reported a blank\n"
             "stance's mean normal force as\n\n"
             f"   {mis['stance_blank_mean_normal_force']:.3f}  _mN   (label)\n"
             f"   {mis['cpg_natural_mean_normal_force']:.3f}  _mN   natural CPG\n\n"
             "read AS mN:\n"
             f"   {mis['as_mN']['stance_over_weight']:.0f}x and "
             f"{mis['as_mN']['cpg_over_weight']:.0f}x body weight\n"
             "   -> impossible; no stance is that.\n\n"
             "read AS uN (the units they were in):\n"
             f"   {mis['as_uN']['stance_over_weight']:.3f}x and "
             f"{mis['as_uN']['cpg_over_weight']:.2f}x body weight\n"
             "   -> what a stance IS.\n\n"
             "The model's force unit is fixed by the\n"
             "project's own ledger: scene.py records\n"
             "model.weight_total = 10.0485, unit \"uN\".\n"
             "A name is what a consumer converts with.",
             va="top", fontsize=8.0, family="monospace")

    # --- the two enforcement mechanisms
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    src = d["source_scan"]
    ax3.text(0, 1,
             "ENFORCEMENT 1: SOURCE ACKNOWLEDGEMENT\n\n"
             f"sources touching the channel : {src['n_consumers']}\n"
             f"not acknowledging           : {len(src['unacknowledged'])}\n"
             f"exempt (with a reason)       : "
             f"{sum(1 for c in src['consumers'] if c['exempt'])}\n\n"
             "INJECTION: a synthetic consumer that\n"
             "reads `normal_force` without the\n"
             "acknowledgement.\n\n"
             "   result: CAUGHT (0 -> 1 unacknowledged)\n\n"
             "LIMIT, stated: this is a TEXT SCAN.  It\n"
             "proves a file ACKNOWLEDGES the downgrade,\n"
             "not that it obeys it.  A token check\n"
             "mistaken for a semantic one is worse than\n"
             "no check, so the limit is written here.",
             va="top", fontsize=7.9)

    # --- the output half
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    out = d["output_scan"]
    ax4.text(0, 1,
             "ENFORCEMENT 2: OUTPUT CONTRACT\n\n"
             f"tactile episode summaries      : {out['n']}\n"
             f"carrying the four fields       : {out['n'] - out['n_noncompliant']}\n"
             f"stating the downgrade          : "
             f"{sum(1 for e in out['episodes'] if e['states_downgrade'])}\n\n"
             "Each carries ITS OWN measured residual,\n"
             "not boilerplate.\n\n"
             "WHY THIS HALF MATTERED: the contract\n"
             "audit globbed only outputs/*.json, so\n"
             "every episode summary sat outside it.\n"
             "Those summaries were reporting absolute\n"
             "micro-newtons and megapascals from a\n"
             "downgraded channel.  The glob now covers\n"
             "the whole tree (backlog 105 -> 356, all\n"
             "counted rather than absent).",
             va="top", fontsize=7.9)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ax5.text(0, 1,
             "WHAT THIS DOES NOT ESTABLISH\n\n"
             "• THE CAUSE of the residual.  The\n"
             "  actuators are excluded as the dominant\n"
             "  cause by MEASUREMENT (gain x1000 and\n"
             "  force range x130 each moved it only\n"
             "  18 % and 16 %), which leaves position-\n"
             "  servo internal stress as a SUSPICION.\n\n"
             "• that the downgrade is OBEYED.  The\n"
             "  source half is a text scan; only the\n"
             "  output half reads an artifact.\n\n"
             "• any load, pressure or tactile-intensity\n"
             "  claim.  The mechanoreceptive drive is\n"
             "  computed FROM this channel, so its\n"
             "  magnitude is not physical either --\n"
             "  only its structure and its dependence\n"
             "  on body state are.\n\n"
             "• nothing here is real-fly data.",
             va="top", fontsize=7.8)

    fig.suptitle("The contact-force channel is a NON_PHYSICAL_LOAD -- measured, and now enforced "
                 "rather than declared", fontsize=11.5, y=0.975)
    fig.text(0.5, 0.012,
             "HAND-BUILT PROTOTYPE, simulator only.  Route chosen: OPTION (b) of "
             "PLATFORM_SELFCONSISTENT_PLAN section 3/P2, explicit downgrade -- because option (a), "
             "softening the contacts to find the source, would invalidate every recorded episode.",
             ha="center", fontsize=8, style="italic", color="dimgray")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=125, bbox_inches="tight")
    print("wrote", out_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
