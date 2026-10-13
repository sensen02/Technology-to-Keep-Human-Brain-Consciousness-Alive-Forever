#!/usr/bin/env python3
"""Figure: ONE joint angle at ONE time, measured from pixels vs the simulator's own value.

Reads the per-frame CSV written by limb_tracking.py.  Nothing is recomputed here and no model or
recording is touched: this plots the measurement that file already produced.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_limb_angle_series_figure.py arena_bare_seed3
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARENA = HERE / "outputs" / "arena"
EP = ARENA / (sys.argv[1] if len(sys.argv) > 1 else "arena_bare_seed3")

rows = list(csv.DictReader((EP / "limb_joint_angles.csv").open()))
for r in rows:
    for k in ("t_s", "elev_meas_deg", "elev_true_deg", "azim_meas_deg", "azim_true_deg",
              "angle_err_deg"):
        r[k] = float(r[k])
err_all = np.asarray([r["angle_err_deg"] for r in rows])

# the two bones to show in the time panels, chosen by the data rather than by hand: the tibia
# with the most samples, and the femur with the worst median (the honest bad case).
def pick(bone_kind, worst: bool, min_n: int = 1):
    """The bone to show: most samples, or the worst median among bones with >= min_n samples.

    min_n matters for the bad-case panel: the very worst MEDIAN belongs to a bone with only a
    handful of frames, and a 3-point panel is not evidence of anything.
    """
    cands = {}
    for r in rows:
        if r["bone"].endswith(bone_kind):
            cands.setdefault(r["bone"], []).append(r["angle_err_deg"])
    cands = {b: v for b, v in cands.items() if len(v) >= min_n}
    if not cands:
        return None
    key = (lambda b: -np.median(cands[b])) if worst else (lambda b: -len(cands[b]))
    return sorted(cands, key=key)[0]


bone_good = pick("tibia", worst=False)
bone_bad = pick("femur", worst=True, min_n=10)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

for ax, bone, title in ((axes[0], bone_good, "best-sampled TIBIA"),
                        (axes[1], bone_bad, "worst-median FEMUR")):
    sel = sorted([r for r in rows if r["bone"] == bone], key=lambda r: r["t_s"])
    t = np.asarray([r["t_s"] for r in sel])
    em = np.asarray([r["elev_meas_deg"] for r in sel])
    et = np.asarray([r["elev_true_deg"] for r in sel])
    e = np.asarray([r["angle_err_deg"] for r in sel])
    ax.plot(t * 1000, et, "-", color="#444", lw=2.0, label="simulator (truth)")
    ax.plot(t * 1000, em, "o", color="#c0392b", ms=5.5, mfc="none", mew=1.4,
            label="measured from 6 camera images")
    for ti, emi, eti in zip(t * 1000, em, et):
        ax.plot([ti, ti], [emi, eti], "-", color="#c0392b", lw=0.8, alpha=0.55)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("bone elevation above horizontal (deg)")
    ax.set_title(f"{bone}  ({title})\nn={len(sel)} frames, median error "
                 f"{np.median(e):.2f} deg, p95 {np.percentile(e, 95):.2f} deg", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

ax = axes[2]
for bone in sorted({r["bone"] for r in rows}):
    sel = [r for r in rows if r["bone"] == bone]
    ax.plot([r["t_s"] * 1000 for r in sel], [r["angle_err_deg"] for r in sel], "o", ms=3.2,
            alpha=0.6, label=bone)
med = float(np.median(err_all))
ax.axhline(med, color="k", lw=1.4, ls="--", label=f"median {med:.2f} deg")
ax.axhline(5.0, color="#888", lw=1.0, ls=":", label="5 deg")
ax.set_yscale("log")
ax.set_xlabel("time (ms)")
ax.set_ylabel("joint-angle error vs truth (deg, log)")
ax.set_title(f"every measured joint angle: {len(rows)} angles over "
             f"{len({r['frame'] for r in rows})} of 46 frames\n"
             f"median {med:.2f} deg  p95 {np.percentile(err_all, 95):.2f} deg  "
             f"max {err_all.max():.2f} deg", fontsize=10)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7, ncol=2, loc="upper left")

fig.suptitle("A fly leg joint angle at a given time, reconstructed from 6 camera images "
             f"({EP.name}) -- hand-built prototype, scored against the simulator, not a real fly",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
dest = EP / "limb_angle_series_figure.png"
fig.savefig(dest, dpi=130)
print(f"wrote {dest}")
print(f"  panels: {bone_good} and {bone_bad}; {len(rows)} angles, median {med:.3f} deg")
