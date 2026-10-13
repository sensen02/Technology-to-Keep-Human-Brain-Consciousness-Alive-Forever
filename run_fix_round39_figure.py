"""ROUND 39 (FIX): the figure for the fix log.  Four panels, every number from a JSON
artifact produced by this round - nothing is redrawn by hand."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine.canonical_physics as cp  # noqa: E402

OUT = HERE / "outputs/fix_round39"
SWEEP = OUT / "fix_round39_contact_one_at_a_time.json"
IMPACT = OUT / "fix_round39_unit_defect_impact.json"
CHECKS = OUT / "fix_round39_unit_and_stance_checks.json"

C_OK, C_BAD, C_WARN = "#1b7837", "#c0392b", "#e08214"


def main() -> None:
    sweep = json.loads(SWEEP.read_text())
    impact = json.loads(IMPACT.read_text())
    checks = json.loads(CHECKS.read_text())

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle(
        "Round 39 fix: gated re-measurement of the standing body\n"
        "(canonical asset unchanged, SHA256 65d3904d...73b6)",
        fontsize=13, fontweight="bold")

    # ---------------- panel 1: the unit defect, as a height comparison ----------
    ax = axes[0, 0]
    arms = impact["measured"]["arms"]
    names = ["gated\n(root solved)", "old penetration\nconvention",
             "stored value\nconsumed as mm"]
    keys = ["gated_solved_root", "gated_old_penetration_convention",
            "reported_mislabelled_root"]
    heights = [arms[k]["seated_thorax_mm"] for k in keys]
    ncons = [arms[k]["seated_gate"]["n_contact"] for k in keys]
    colors = [C_OK, C_OK, C_BAD]
    bars = ax.bar(range(3), heights, color=colors, width=0.6)
    ax.axhline(checks["measured"]["C2_stance_gate"]["thorax_minimum_mm"], ls="--", c="k",
               lw=1.0, label="geometric minimum (0.8x leg chain)")
    for b, h, n in zip(bars, heights, ncons):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.4, f"{h:.3f} mm\n{n} contacts",
                ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("seated thorax height [mm]")
    ax.set_title("(1) The unit defect, measured:\n"
                 "the stored `root_z_mm` field holds cm; consuming it as mm buries the fly",
                 fontsize=10)
    ax.set_ylim(0, max(heights) * 1.35)
    ax.legend(fontsize=8, loc="upper right")

    # ---------------- panel 2: one factor at a time, escape time ---------------
    ax = axes[0, 1]
    table = [r for r in sweep["measured"]["table"] if r.get("escape_time_s") is not None]
    table.sort(key=lambda r: r["escape_time_s"])
    labels = [r["case"].replace("only_", "").replace("bundle_", "BUNDLE ") for r in table]
    vals = [r["escape_time_s"] * 1000 for r in table]
    base = next((r["escape_time_s"] * 1000 for r in table
                 if r["case"] == "bundle_mujoco_default"), None)
    cols = [C_OK if (base is not None and v > base * 1.15) else
            (C_BAD if (base is not None and v < base * 0.85) else "#7f8c8d") for v in vals]
    ax.barh(range(len(vals)), vals, color=cols, height=0.65)
    if base:
        ax.axvline(base, ls="--", c="k", lw=1.0, label=f"MuJoCo default = {base:.1f} ms")
    for i, v in enumerate(vals):
        ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("time until the thorax leaves 90% of the gated seated height [ms]")
    ax.set_title("(2) One field at a time from the MuJoCo default:\n"
                 "cone=elliptic+noslip3 helps most; mu=2.0 makes it WORSE", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.legend(fontsize=8, loc="lower right")

    # ---------------- panel 3: height trajectories ----------------------------
    ax = axes[1, 0]
    hs = impact["measured"]["height_table_mm"]
    t = [r["t_s"] for r in hs]
    for key, lbl, c in [("gated_thorax_mm", "gated (root solved)", C_OK),
                        ("mislabelled_thorax_mm", "stored value consumed as mm", C_BAD)]:
        ax.plot(t, [r[key] for r in hs], color=c, lw=1.8, label=lbl)
    seat = arms["gated_solved_root"]["seated_thorax_mm"]
    ax.axhline(0.9 * seat, ls=":", c="k", lw=1.0, label="escape threshold (0.90x seated)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("thorax height [mm]")
    ax.set_title("(3) Same pose, same constant command, same contact parameters:\n"
                 "only the INITIAL HEIGHT differs", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ---------------- panel 4: the gate, on the record ------------------------
    ax = axes[1, 1]
    ax.axis("off")
    c1 = checks["measured"]["C1_unit_label_scan"]
    c2 = checks["measured"]["C2_stance_gate"]
    pos = c2["positive_control"]
    neg = c2["negative_control_D2_consumed"]
    txt = (
        "TWO DEFECTS, TWO MECHANICAL CHECKS\n"
        "─────────────────────────────────────────────\n"
        f"D1  `_mm`-named fields holding model units (cm):\n"
        f"        {c1['mm_named_but_model_unit']} of {c1['mm_named_fields_total']} fields, "
        f"across {c1['files_affected']} files\n\n"
        f"D2  a stored height consumed as mm:\n"
        f"        qpos[2] = stored_root_z_mm / 10.0\n\n"
        "C2 GATE - positive control (seated stance)\n"
        f"        thorax {pos['seated']['thorax_centre_mm']:.3f} mm, "
        f"{pos['seated']['n_contact']} contacts,\n"
        f"        tarsus-only = {pos['seated']['tarsus_only']}  -> "
        f"{'PASS' if pos['passed'] else 'FAIL'}\n\n"
        "C2 GATE - negative control (D2 consumed)\n"
        f"        thorax {neg['seated']['thorax_centre_mm']:.3f} mm, "
        f"{neg['seated']['n_contact']} contacts  -> "
        f"{'REJECTED' if neg['rejected'] else 'ACCEPTED'}\n\n"
        "─────────────── WHAT IS STILL TRUE ───────────────\n"
        f"Escape from the gated stance: "
        f"{arms['gated_solved_root']['escape_time_s']*1000:.1f} ms\n"
        f"Final thorax height: "
        f"{arms['gated_solved_root']['final_thorax_mm']:.2f} mm\n"
        "No contact setting makes it stand in 1.0 s.\n"
        "The constant LP balance is a POINT, not an attractor."
    )
    ax.text(0.01, 0.99, txt, va="top", ha="left", fontsize=9.2, family="monospace")
    ax.set_title("(4) The checks, and what the fix does NOT change", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = OUT / "fix_round39_summary.png"
    fig.savefig(out, dpi=125)
    print("wrote", out)

    # machine-readable caption so the figure cannot drift from the artifacts
    cp.write_result(
        OUT / "fix_round39_figure.json",
        question="which figure summarises the round-39 fix?",
        status="PASS",
        measured={
            "figure": str(out),
            "panels": {
                "1": "the unit defect: seated heights in mm for gated / old-convention / "
                     "stored-consumed-as-mm, with contact counts",
                "2": "one-factor-at-a-time escape times from the MuJoCo-default contact set",
                "3": "thorax height vs time for the gated and the mislabelled initial "
                     "condition, same pose and command",
                "4": "the two mechanical checks and what the fix does not change",
            },
            "sources": [str(SWEEP), str(IMPACT), str(CHECKS)],
        },
        provenance={"every number read from": [str(SWEEP), str(IMPACT), str(CHECKS)]},
        what_this_does_not_support=[
            "The figure is a summary of diagnostic measurements of a BODY.  It is not "
            "autonomous behaviour and not a statement about real flies.",
        ],
    )


if __name__ == "__main__":
    main()
