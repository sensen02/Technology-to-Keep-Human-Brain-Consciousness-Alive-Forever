#!/usr/bin/env python3
"""Figure for the tracked limb measurement: what is usable, and how wrong is it.

Panel 1: per-marker visibility (fraction of frames in which >=3 cameras see the marker), which
is the thing that decides whether a joint angle can be formed at all.
Panel 2: the front-left TIBIA joint angle through time, measured from the gated track against
the same angle computed from the simulator's own marker positions.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_arena_reconstruct import load_cameras_for_episode, load_projection_offsets, angle_between

ARENA = HERE / "outputs" / "arena"
ep = ARENA / (sys.argv[1] if len(sys.argv) > 1 else "arena_bare_seed3")
geom = json.loads((ARENA / "arena_geometry.json").read_text())
load_projection_offsets(ep / "camera_offsets.json")
cams, _ = load_cameras_for_episode(geom, ep, ARENA / "camera_poses.json")
z = np.load(ep / "episode.npz")
segs = list(geom["marker_body_segments"])
FT = np.asarray(z["truth/marker_fly_world_mm"], dtype=float)
F = FT.shape[0]
tt = np.asarray(z["truth/frame_time_s"], dtype=float)

vis = np.zeros((len(segs), F), dtype=int)
for f in range(F):
    for cam in cams:
        a, b = int(z[f"px/{cam.name}/offsets"][f]), int(z[f"px/{cam.name}/offsets"][f + 1])
        R = np.asarray(z[f"px/{cam.name}/rows"][a:b], dtype=float)
        C = np.asarray(z[f"px/{cam.name}/cols"][a:b], dtype=float)
        if R.size == 0:
            continue
        pr = cam.project(FT[f])
        d = np.hypot(R[None, :] - pr[:, 0:1], C[None, :] - pr[:, 1:2])
        vis[d.min(axis=1) <= 1.5, f] += 1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
frac = (vis >= 3).mean(axis=1)
order = np.argsort(-frac)
ax[0].barh([segs[i] for i in order], frac[order] * 100,
           color=["#27ae60" if frac[i] >= 0.5 else ("#e67e22" if frac[i] >= 0.1 else "#c0392b")
                  for i in order])
ax[0].axvline(50, color="k", ls="--", lw=1)
ax[0].set_xlabel("% of frames with >= 3 cameras seeing the marker")
ax[0].set_title("which markers are actually measurable\n"
                "(a joint angle needs BOTH of its endpoints seen by >= 3 cameras)")
ax[0].grid(True, axis="x", alpha=0.3)

# panel 2: the front-left tibia joint angle, measured vs truth, only where the tracker accepted both ends
tr = json.loads((ep / "limb_tracking.json").read_text())
rows = [r for r in tr.get("joint_angles", []) if r.get("median_deg") is not None]
rows.sort(key=lambda r: r["median_deg"])
labels = [f"{r['bone']}  (n={r['n']})" for r in rows]
med = [r["median_deg"] for r in rows]
p95 = [r["p95_deg"] for r in rows]
y = np.arange(len(rows))
ax[1].barh(y, med, color="#27ae60", label="median error")
ax[1].barh(y, [p - m for p, m in zip(p95, med)], left=med, color="#c0392b", alpha=0.55,
           label="up to p95")
ax[1].set_yticks(y); ax[1].set_yticklabels(labels, fontsize=9)
ax[1].axvline(5, color="k", ls="--", lw=1)
ax[1].set_xlabel("joint-angle error vs simulator truth, degrees")
ax[1].set_title("ANGLE ERROR PER BONE from the gated track\n"
                "green = median, red = the tail up to p95 (the identity failures live there)")
ax[1].legend(fontsize=9)
ax[1].grid(True, axis="x", alpha=0.3)
fig.tight_layout()
out = ep / f"limb_tracking_figure.png"
fig.savefig(out, dpi=110)
print("wrote", out)
