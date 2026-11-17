"""Plot saved model righting traces against clearly labeled literature references."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
KIN = json.loads((HERE / "MODEL_KINEMATICS.json").read_text())
CASES = ("side_fallen", "near_supine")
LABELS = {"side_fallen": "model: side 110.27°", "near_supine": "model: near-supine 161.19°"}
COLORS = {"side_fallen": "#2f6f9f", "near_supine": "#c2533d"}
LEG_ORDER = KIN["method"]["leg_order"]


fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

ax = axes[0, 0]
for case in CASES:
    with np.load(RUN / f"FIT_{case}" / "TRACE.npz", allow_pickle=False) as trace:
        tilt = np.rad2deg(np.arccos(np.clip(trace["upright"], -1, 1)))
    every = 80
    time = np.arange(len(tilt))[::every] * 12.5e-6
    ax.plot(time, tilt[::every], label=LABELS[case], color=COLORS[case], lw=2)
ax.axhline(20, color="black", ls="--", lw=1, label="model upright threshold")
ax.axvline(0.65, color="#6b6b6b", ls=":", lw=1.7, label="Akitake WT clip: 0.65 s total")
ax.axvline(1.19, color="#9b7b25", ls="-.", lw=1.7, label="Akitake WT group mean: 1.19 s")
ax.set(xlabel="time from saved initial state (s)", ylabel="thorax tilt (deg)",
       title="A. Model body reorientation (literature lines are context, not gates)")
ax.set_xlim(0, 2.6)
ax.set_ylim(0, 180)
ax.legend(fontsize=8)
ax.grid(alpha=.2)

ax = axes[0, 1]
x = np.arange(len(LEG_ORDER))
width = .36
for offset, case in zip((-.18, .18), CASES, strict=True):
    values = [KIN["cases"][case]["per_leg"][leg]["pre_upright_vertical_impulse_fraction"]
              for leg in LEG_ORDER]
    ax.bar(x + offset, values, width, label=LABELS[case], color=COLORS[case])
ax.set_xticks(x, LEG_ORDER)
ax.set(ylabel="fraction of foot vertical impulse", title="B. Foot load distribution before first ≤20°")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=.2)

ax = axes[1, 0]
categories = ("foot pads", "wings", "other non-foot")
for offset, case in zip((-.18, .18), CASES, strict=True):
    entry = KIN["cases"][case]
    foot = entry["pre_upright_foot_vertical_impulse_fraction_of_all_contacts"]
    wing = entry["pre_upright_wing_vertical_impulse_fraction_of_all_contacts"]
    values = (foot, wing, 1 - foot - wing)
    ax.bar(np.arange(3) + offset, values, width, label=LABELS[case], color=COLORS[case])
ax.set_xticks(np.arange(3), categories)
ax.set(ylabel="fraction of all vertical contact impulse",
       title="C. Contact support before first ≤20°")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=.2)

ax = axes[1, 1]
axes_names = ("body-x roll", "body-y pitch", "body-z yaw")
keys = ("path_fraction_body_x_roll", "path_fraction_body_y_pitch", "path_fraction_body_z_yaw")
for offset, case in zip((-.18, .18), CASES, strict=True):
    values = [KIN["cases"][case][key] for key in keys]
    ax.bar(np.arange(3) + offset, values, width, label=LABELS[case], color=COLORS[case])
ax.set_xticks(np.arange(3), axes_names)
ax.set(ylabel="fraction of absolute orientation path",
       title="D. Rotation route before first ≤20°")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=.2)

fig.suptitle("Drosophila righting comparison: saved model traces vs adult WT timing context", fontsize=14)
fig.savefig(HERE / "MODEL_VS_BIOLOGY.png", dpi=180)
