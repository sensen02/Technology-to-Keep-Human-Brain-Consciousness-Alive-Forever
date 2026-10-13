#!/usr/bin/env python3
"""S5: the declared capture radius, measured -- and NOT confirmed.  One figure.

The declared 50 um is not defended by this measurement; it is DISSOLVED by it.  The figure has to
make three things impossible to misread:

  * below the recruitment threshold the effect is EXACTLY zero, and that zero is injected rather
    than asserted -- transplanting a larger radius's recruitment pattern onto a radius that recruits
    nothing turns it into 0.13702 rad;
  * the response is NON-MONOTONE (50 um sits 35 % BELOW 30 um; 120 um sits 79 % below 80 um);
  * the charge is CONSTANT at every recruiting radius, so what the curve varies is only where the
    current goes -- the spatial pattern, which the module already declares as not established.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_capture_radius_figure.py --out <path.png>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=str(HERE / "outputs" / "capture_radius_dose_response.json"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "capture_radius_figure.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(Path(args.check).read_text())
    rows = d["rows"]
    r = np.array([x["capture_radius_um"] for x in rows])
    n = np.array([x["n_recruited"] for x in rows])
    eff = np.array([x["max_abs_delta_rad"] for x in rows])
    inj = d.get("g2_injection") or {}
    g3, g4 = d["g3_monotonic"], d["g4_effective_threshold"]

    fig = plt.figure(figsize=(14, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)

    # --- the response, on a log radius axis, with the declared value marked
    ax = fig.add_subplot(gs[0, :2])
    pos = eff > 0
    ax.plot(r[pos], eff[pos], "o-", color="#2b7bba", ms=7, lw=1.6, label="effect (loop response)")
    ax.plot(r[~pos], eff[~pos], "s", color="#999999", ms=8,
            label="EXACTLY zero (nothing recruited)")
    for x, y, ni, leg in zip(r[pos], eff[pos], n[pos],
                             [x["worst_leg"] for x in rows if x["max_abs_delta_rad"] > 0]):
        ax.annotate(f"{ni} neurons\n{leg}", (x, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=7.2, color="dimgray")
    ax.axvline(50.0, color="crimson", ls="--", lw=1.4)
    ax.annotate("DECLARED 50 um\nsits in a 35 % trough", (50.0, 0.13), color="crimson",
                fontsize=8.5, ha="left", xytext=(8, 0), textcoords="offset points")
    for v in g3["violations"]:
        ax.annotate("", xy=(v["to_radius_um"], v["to_rad"]), xytext=(v["from_radius_um"], v["from_rad"]),
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.6, alpha=0.7,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.text(np.sqrt(v["from_radius_um"] * v["to_radius_um"]),
                (v["from_rad"] + v["to_rad"]) / 2 - 0.09,
                f"-{100*v['drop_frac']:.0f} %", color="crimson", fontsize=8, ha="center")
    if inj:
        ax.plot([inj["radius_um"]], [inj["max_abs_delta_rad_after_transplant"]], "*", ms=17,
                color="darkgreen",
                label=f"INJECTION: 200 um pattern forced onto {inj['radius_um']:.0f} um")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 15, 20, 30, 50, 80, 120, 200])
    ax.set_xticklabels(["5", "10", "15", "20", "30", "50", "80", "120", "200"])
    ax.set_xlabel("declared capture radius (um)")
    ax.set_ylabel("max |change in joint excursion| vs no stimulus (rad)")
    ax.set_title("the loop's response to the DECLARED capture radius -- non-monotone, so 50 um is "
                 "not confirmed", fontsize=10.0, pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.03, 0.78)
    ax.grid(alpha=0.25)

    # --- recruited count (the geometric rule) alongside
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(r, n, "o-", color="#7a5195", ms=6, lw=1.4)
    ax2.axvline(50.0, color="crimson", ls="--", lw=1.2)
    ax2.set_xscale("log")
    ax2.set_xticks([5, 10, 20, 50, 120, 200])
    ax2.set_xticklabels(["5", "10", "20", "50", "120", "200"], fontsize=8)
    ax2.set_xlabel("radius (um)")
    ax2.set_ylabel("neurons recruited (geometric rule)")
    ax2.set_title("what the rule touches\n(declared, from soma voxel distances)", fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.annotate(f"0 below {g4['largest_radius_with_exactly_zero_effect_um']:.0f} um",
                 (8, 40), fontsize=8, color="dimgray")

    # --- charge is constant, so the curve varies the SPATIAL pattern only
    ax3 = fig.add_subplot(gs[1, 0])
    ch = [x["charge_nA_ms"] for x in rows]
    ax3.bar(range(len(rows)), ch, color=["#cccccc" if c == 0 else "#2b7bba" for c in ch])
    ax3.set_xticks(range(len(rows)))
    ax3.set_xticklabels([f"{x:.0f}" for x in r], rotation=60, fontsize=7.5)
    ax3.set_xlabel("radius (um)")
    ax3.set_ylabel("charge delivered (nA*ms)")
    ax3.set_title("the charge is CONSTANT at every recruiting radius\n"
                  "(weights normalised to sum to 1)", fontsize=8.8)

    # --- the guards
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    lines = [f"overall  {d['overall']}",
             "",
             f"G1 determinism           {d['deterministic']}",
             f"G2 zero recruit -> zero   {d['g2_no_recruitment_no_effect']['all_exactly_zero']}",
             f"   at radii (um)          "
             f"{d['g2_no_recruitment_no_effect']['radius_with_zero_recruitment']}",
             f"   injection (transplant) "
             f"{inj.get('max_abs_delta_rad_after_transplant', float('nan')):.5f} rad",
             f"   -> guard not vacuous   {inj.get('guard_is_not_vacuous')}",
             "",
             f"G3 monotone              {g3['is_monotone']}",
             f"   violations             {len(g3['violations'])}",
             "",
             f"G4 effect appears        {g4['bracket_um'][0]:.0f} .. {g4['bracket_um'][1]:.0f} um",
             f"   DECLARED               {g4['declared_radius_um']:.0f} um",
             "",
             f"charge constant          {d['charge_is_constant_across_radii']}",
             f"   value (nA*ms)          {d['charge_values_nA_ms']}"]
    ax4.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.2)

    # --- what it does not establish
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ax5.text(0, 1, "WHAT THIS DOES NOT ESTABLISH\n\n"
                   "• that 50 um is the right radius.\n"
                   "  It is not confirmed here -- it is\n"
                   "  DISSOLVED: the curve is not\n"
                   "  monotone, so no radius can be\n"
                   "  preferred on behavioural grounds.\n\n"
                   "• anything about a real fly.  This is\n"
                   "  how THIS loop responds to a\n"
                   "  DECLARED geometric rule.\n\n"
                   "• the spatial pattern.  The charge is\n"
                   "  identical at every recruiting radius\n"
                   "  (5000 nA*ms); only WHERE it goes\n"
                   "  changes.  The access map has soma\n"
                   "  voxel positions and nothing else --\n"
                   "  no morphology, no neurite, no soma\n"
                   "  radius.\n\n"
                   "• no charge-safety claim.  The fly\n"
                   "  still does not stand or walk.",
            va="top", fontsize=7.8)

    fig.suptitle("S5: measuring the DECLARED 50 um capture radius through the full loop",
                 fontsize=11.5, y=0.975)
    fig.text(0.5, 0.012,
             "HAND-BUILT PROTOTYPE, simulator only.  G2 is INJECTED, not asserted: transplanting a "
             "larger radius's recruitment pattern onto a radius that recruits nothing turns the "
             "effect from exactly 0.00000 into 0.13702 rad.",
             ha="center", fontsize=8, style="italic", color="dimgray")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
