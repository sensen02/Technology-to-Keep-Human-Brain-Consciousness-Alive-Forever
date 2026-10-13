#!/usr/bin/env python3
"""Are the triangulated points REAL fiducials, and where does the residual live?

Reads the reconstruction JSON produced by run_arena_reconstruct.py and plots, per frame, the
distance from each triangulated marker to its DECLARED position.  Two things this makes visible:

  * whether the triangulated points are the fiducials or grass specks -- specks cannot land on the
    declared x,y, so a small residual is positive evidence of a real detection;
  * that the world-frame RMS is NOT spread evenly: it is carried by two markers.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_fiducial_residual_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARENA = HERE / "outputs" / "arena"

CASES = [("arena_bare_seed0", "reconstruction_pngblobs.json", "seed0 (recorded table BROKEN: 1.4 % hit)"),
         ("arena_bare_seed3", "reconstruction_pngblobs.json", "seed3 (recorded table fine: 93.8 % hit)"),
         ("arena_bare_seed3", "reconstruction_recorded.json", "seed3, same episode via RECORDED table")]

fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.2))

for col, (ep, fn, title) in enumerate(CASES):
    d = json.loads((ARENA / ep / fn).read_text())
    f0 = d["static_frame0"]
    rec = np.asarray(f0["reconstructed_mm"], float)
    dec = np.asarray(f0["declared_mm"], float)
    idx = list(f0["marker_index"])
    err = np.linalg.norm(rec - dec, axis=1)
    # the world RMS over all ok frames, from the per-frame summary
    rms = np.array([r["fit_rms_mm"] for r in d["static_per_frame"] if r["ok"]], float)

    ax = axes[0][col]
    cols = ["#2e7d32" if e < 0.5 else "#c0392b" for e in err]
    ax.bar([f"mk{i}" for i in idx], err * 1000.0, color=cols)
    ax.set_yscale("log")
    ax.set_ylabel("|reconstructed - declared| (um)")
    ax.set_title(f"{title}\n{len(err)} markers triangulated; median "
                 f"{np.median(err)*1000:.0f} um, max {err.max()*1000:.0f} um", fontsize=9.5)
    ax.axhline(500.0, color="#888", lw=1.0, ls="--")
    ax.grid(alpha=0.3, axis="y", which="both")
    for xi, e in enumerate(err):
        ax.text(xi, e * 1000.0 * 1.25, f"{e*1000:.0f}", ha="center", fontsize=8)
    ax.set_ylim(1, max(err.max() * 1000.0 * 6, 10))

    ax = axes[1][col]
    dz = rec[:, 2] - dec[:, 2]
    ax.bar([f"mk{i}" for i in idx], dz * 1000.0, color="#1565c0")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_ylabel("z error (um): reconstructed - declared")
    ax.set_xlabel("declared marker id")
    ax.set_title(f"world-frame RMS over ok frames: median {np.median(rms):.4f} mm "
                 f"(n={len(rms)})", fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")
    lo, hi = dz.min() * 1000.0, dz.max() * 1000.0
    ax.set_ylim(min(lo * 1.4, -50), max(hi * 1.4, 100))

fig.suptitle("Are the triangulated static points real fiducials, and where does the 3-D residual live?\n"
             "green = within 0.5 mm of the declared position; the world RMS is carried by two markers, "
             "not spread evenly  (hand-built prototype, scored against the simulator's own geometry)",
             fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.90))
dest = HERE / "outputs" / "arena" / "fiducial_residual_figure.png"
fig.savefig(dest, dpi=125)
print(f"wrote {dest}")

for ep, fn, title in CASES:
    d = json.loads((ARENA / ep / fn).read_text())
    f0 = d["static_frame0"]
    rec = np.asarray(f0["reconstructed_mm"], float)
    dec = np.asarray(f0["declared_mm"], float)
    err = np.linalg.norm(rec - dec, axis=1)
    ok = [r for r in d["static_per_frame"] if r["ok"]]
    rms = np.array([r["fit_rms_mm"] for r in ok], float)
    print(f"  {title}: n_tri={len(err)} median|d|={np.median(err)*1000:.1f} um "
          f"max|d|={err.max()*1000:.1f} um  worldRMS med {np.median(rms):.4f} mm  frames {len(ok)}")
