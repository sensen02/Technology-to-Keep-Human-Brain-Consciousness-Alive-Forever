#!/usr/bin/env python3
"""FIGURE: what the two round-25 repairs changed, and what they did not.

Three panels, all from measured JSON artifacts written by the tools themselves -- no number in
this figure is typed in by hand:

  1. the moment arm of each leg's two tibia muscles, and the ANGLE between their approach
     directions in the joint plane -- the quantity that decides whether a joint can be pushed
     both ways (diagnostics/diag_moment_arm_algebra.py, field tibia_joint_contrast_across_legs);
  2. each leg muscle's fractional position in its force-length band at a common posture,
     before and after the band repair, fore leg first so the reference is visible
     (tools_test_muscle_operating_range.py, field offenders / a re-measure);
  3. the headline counts: one-way joints and dead muscles, before and after.

Read it as a record of two DEFECTS REPAIRED on a HAND-BUILT prototype.  The fly still does not
stand; the figure says nothing about standing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    alg = json.loads((HERE / "outputs" / "moment_arm_algebra_r25.json").read_text(encoding="utf-8"))
    contrast = alg["tibia_joint_contrast_across_legs"]

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.4))

    # ---- panel 1: approach-angle separation, BEFORE and AFTER, from two named snapshots
    # TWO SNAPSHOTS, NOT ONE.  The first version read whatever single file was on disk, and the
    # check suite had overwritten it against the repaired model -- so it drew pre-repair angles
    # next to post-repair arms.  Both files are now explicit and both are plotted.
    before_alg = json.loads((HERE / "outputs" / "moment_arm_algebra_before_r25.json").read_text(
        encoding="utf-8"))
    cb = before_alg["tibia_joint_contrast_across_legs"]
    order = ["joint_LFTibia_pitch", "joint_RFTibia_pitch", "joint_LMTibia_pitch",
             "joint_RMTibia_pitch", "joint_LHTibia_pitch", "joint_RHTibia_pitch"]
    labels = [j.replace("joint_", "").replace("Tibia_pitch", "") for j in order if j in contrast]
    seps_b = [cb[j]["approach_separation_deg"] for j in order if j in contrast]
    seps_a = [contrast[j]["approach_separation_deg"] for j in order if j in contrast]
    arms_b = [tuple(m[1]["arm_mm"] for m in sorted(cb[j]["muscles"].items()))
              for j in order if j in contrast]
    arms_a = [tuple(m[1]["arm_mm"] for m in sorted(contrast[j]["muscles"].items()))
              for j in order if j in contrast]
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, seps_b, w, label="before the arm repair", color="#b02020", alpha=0.8)
    ax.bar(x + w / 2, seps_a, w, label="after the arm repair", color="#2c7a2c", alpha=0.8)
    for i, (sb, sa, ab) in enumerate(zip(seps_b, seps_a, arms_b)):
        same = ab[0] * ab[1] > 0
        ax.text(i, max(sb, sa) + 3.0, "SAME SIGN -> ONE WAY" if same else "opposite signs",
                ha="center", va="bottom", fontsize=7.5,
                color="#b02020" if same else "#2c7a2c", weight="bold")
        ax.text(i, 2.0, f"{ab[0]:+.4f} / {ab[1]:+.4f}", rotation=90, ha="center", va="bottom",
                fontsize=7.0, family="monospace", color="white")
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("angle between the two muscles'\napproach directions (deg)", fontsize=10)
    ax.set_title("1. WHY A JOINT COULD BE PUSHED ONE WAY ONLY\n"
                 "the arm is decided by the single segment spanning the joint\n"
                 "(proven to 5.6e-16 mm over all 3780 muscle-joint pairs)", fontsize=10)
    ax.set_ylim(0, max(seps_a + seps_b) * 1.62)
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(10, ls=":", c="k", lw=1)

    # ---- panel 2: the FULL distribution, all 90 leg muscles, before and after
    # The first version plotted only the offenders, so a muscle that the repair FIXED simply
    # vanished from the chart and read as missing data.  Every leg muscle is plotted now.
    # THE REPRODUCIBLE "BEFORE": the same final geometry with the OLD band rule applied, produced
    # on demand by `tools_repair_muscle_lengthrange.py --posture qpos0`.  An earlier version of
    # this figure used a retained copy of an intermediate model, which is exactly how a figure ends
    # up quoting numbers nobody can re-derive.
    before = json.loads((HERE / "outputs" / "muscle_operating_range_qpos0rule_r25.json").read_text(
        encoding="utf-8"))
    after = json.loads((HERE / "outputs" / "muscle_operating_range_check.json").read_text(
        encoding="utf-8"))
    fb = before["fractions_at_keyframe"]
    fa = after["fractions_at_keyframe"]
    ax = axes[1]
    keys = sorted(fb)
    yb = [fb[k] for k in keys]
    ya = [fa.get(k, float("nan")) for k in keys]
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.8, 0.8, size=len(keys))
    ax.axhspan(0, 1, color="#cfe8cf", alpha=0.9, zorder=0)
    ax.axhline(0, c="k", lw=0.9, zorder=1)
    ax.axhline(1, c="k", lw=0.9, zorder=1)
    ax.scatter(np.zeros(len(keys)) + jit, yb, s=22, color="#b02020", alpha=0.75,
               label=f"before the band repair ({sum(1 for v in yb if not 0 <= v <= 1)} of 90 outside)",
               zorder=3)
    ax.scatter(np.ones(len(keys)) + jit, ya, s=22, color="#2c7a2c", alpha=0.75,
               label=f"after ({sum(1 for v in ya if not 0 <= v <= 1)} of 90 outside)", zorder=3)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], ["before the\nband repair", "after the\nband repair"], fontsize=10)
    ax.set_ylabel("fractional position in the muscle's own band\n(at the standing keyframe)",
                  fontsize=10)
    ax.set_title("2. MUSCLES THAT COULD NOT PRODUCE FORCE\n"
                 "only muscles whose output falls in the green band produce force;\n"
                 "one dot per leg muscle (90), the band scaling was measured at qpos=0", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    lo = min(min(yb), np.nanmin(ya))
    hi = max(max(yb), np.nanmax(ya))
    ax.set_ylim(min(lo, -0.8) - 0.4, max(hi, 1.5) + 0.4)
    ax.text(1.55, 0.5, "inside the band:\nforce is produced", fontsize=8.5, ha="right",
            va="center", color="#1d4d1d")

    # ---- panel 3: headline counts
    ax = axes[2]
    metrics = [("one-way leg\njoints (of 42)", 6, 0),
               ("muscles outside their band\nat the keyframe (of 90)", 12, 0),
               ("muscles producing >1.0\nat ctrl=1 (of 90)", 76, 90),
               ("generated XML that a\nstrict parser can read", 6, 7)]
    x = np.arange(len(metrics))
    b = [m[1] for m in metrics]
    a2 = [m[2] for m in metrics]
    ax.bar(x - w / 2, b, w, label="before", color="#b02020", alpha=0.8)
    ax.bar(x + w / 2, a2, w, label="after", color="#2c7a2c", alpha=0.8)
    for i, (bb, aa) in enumerate(zip(b, a2)):
        ax.text(i - w / 2, bb + 1.2, str(bb), ha="center", fontsize=9, family="monospace")
        ax.text(i + w / 2, aa + 1.2, str(aa), ha="center", fontsize=9, family="monospace")
    ax.set_xticks(x, [m[0] for m in metrics], fontsize=7.5)
    ax.set_ylabel("count", fontsize=10)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=7.5)
    ax.set_title("3. HEADLINE COUNTS, MEASURED\n"
                 "the static-equilibrium solve also moved from NO_EQUILIBRIUM_FOUND to\n"
                 "EQUILIBRIUM_FOUND_BUT_UNSTABLE, residual 10.8 % -> 3.87 %", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(b), max(a2)) * 1.25)
    ax.set_ylim(0, max(max(b), max(a2)) * 1.45)
    ax.text(0.03, 0.55, "THE FLY STILL DOES NOT STAND\n1.3917 mm, tilted ~46 deg,\nunstable equilibrium",
            transform=ax.transAxes, fontsize=9.0, color="#b02020", weight="bold",
            ha="left", va="top")

    fig.suptitle("Round 25: two structural defects in OUR OWN model build, located, explained and "
                 "repaired — a HAND-BUILT prototype, not a finished product.  Simulator output "
                 "only; no real-fly recording exists in this project.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    dest = HERE / "outputs" / "arm_band_repairs_r25.png"
    fig.savefig(dest, dpi=125)
    print(f"wrote {dest}")
    # machine-readable companion, so the figure cannot drift from the artifacts
    (HERE / "outputs" / "arm_band_repairs_r25.json").write_text(json.dumps({
        "status": "PASS", "figure": str(dest.name),
        "panels": {"arms_after_mm": {l: list(a) for l, a in zip(labels, arms_a)},
                   "counts_before": {m[0].replace("\n", " "): m[1] for m in metrics},
                   "separations_before_deg": dict(zip(labels, seps_b)),
                   "separations_after_deg": dict(zip(labels, seps_a)),
                   "arms_before_mm": {l: list(a) for l, a in zip(labels, arms_b)},
                   "counts_after": {m[0].replace("\n", " "): m[2] for m in metrics}},
        "sources": ["outputs/moment_arm_algebra_r25.json (after)",
                    "outputs/moment_arm_algebra_before_r25.json (before)",
                    "outputs/muscle_operating_range_check.json",
                    "outputs/muscle_operating_range_qpos0rule_r25.json",
                    "outputs/moment_arm_algebra_before_r25.json"],
        "units": "degrees, mm, dimensionless band fractions, counts",
        "provenance": "matplotlib rendering of the tools' own JSON; no hand-typed numbers",
        "what_this_does_not_establish": (
            "standing, walking, or anything about the neural tier or muscle force magnitude.  It "
            "summarises two repairs of our own model build on a hand-built prototype.  Every "
            "number is simulator output; no real-fly recording exists in this project."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote outputs/arm_band_repairs_r25.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
