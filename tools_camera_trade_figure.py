#!/usr/bin/env python3
"""Redraw the camera trade-study figure from camera_trade.json (no re-measurement).

Kept separate from tools_camera_trade.py so the figure can be restyled without paying for the
22-frame PNG re-detection and the 12 split-half calibrations, which is the slow part.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tools_limb_angle_precision import FEMUR_MM, TIBIA_MM  # noqa: E402

ARENA = HERE / "outputs" / "arena"
d = json.loads((ARENA / "camera_trade.json").read_text())
a, table = d["part_a_current_rig"], d["part_b_cameras_at_r22"]
ref = d["monte_carlo_reference"]["femur_deg"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
radii = np.array([3.0, 4, 5, 6, 8, 10, 15, 22])
for rows, style in ((800, "-"), (1024, "--"), (2048, "-."), (4096, ":")):
    ax[0].plot(radii, ref * (640.0 / rows) * (radii / 22.0), style, lw=2,
               label=f"{rows}-row sensor")
ax[0].scatter([22] * len(table), [r["femur_angle_deg_r22"] for r in table], c="k",
              zorder=5, s=45, label="real cameras, r=22 mm (see table / json)")
floor = a["held_out_error_mm"]["median"]
ax[0].axhline(math.degrees(math.sqrt(2.0) * floor / FEMUR_MM), color="r", ls="--", lw=2,
              label=f"MEASURED floor, recalibrated: {floor:.4f} mm -> "
                    f"{math.degrees(math.sqrt(2.0) * floor / FEMUR_MM):.1f} deg femur")
ax[0].axhline(math.degrees(math.sqrt(2.0) * 0.0653 / FEMUR_MM), color="#e67e22", ls=":", lw=2,
              label="MEASURED floor, stored offsets: 0.0653 mm -> "
                    f"{math.degrees(math.sqrt(2.0) * 0.0653 / FEMUR_MM):.1f} deg femur")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("camera ring radius, mm")
ax[0].set_ylabel("femur joint-angle error, degrees (noise-only)")
ax[0].set_title("price of optics: ring tightness and sensor size\n"
                "ring r=22 -> 8 mm is FREE and beats every camera upgrade in the table")
ax[0].legend(fontsize=8, loc="lower left"); ax[0].grid(True, which="both", alpha=0.3)

labels = ["no gate\n(mis-associated\nblobs kept)", "outlier gate\n(stored offsets)",
          "outlier gate +\nsplit-half\nrecalibration", "noise-only Monte\nCarlo prediction\n(0.5 px)"]
vals = [0.8696, 0.0653, floor, 0.0417]
bars = ax[1].bar(labels, vals, color=["#c0392b", "#e67e22", "#27ae60", "#2980b9"])
for bar, v in zip(bars, vals):
    ax[1].annotate(f"{v:.4f} mm\n{math.degrees(math.sqrt(2.0) * v / FEMUR_MM):.1f} deg femur",
                   (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
ax[1].set_ylabel("3-D position error, mm (rigid-fit RMS)")
ax[1].set_title("what the same six cameras actually deliver\n"
                "degree labels use sqrt(2)*sigma/bone_length with bone = 0.705 mm")
ax[1].grid(True, axis="y", alpha=0.3)
fig.tight_layout()
p = ARENA / "camera_trade.png"
fig.savefig(p, dpi=110)
print("wrote", p)
