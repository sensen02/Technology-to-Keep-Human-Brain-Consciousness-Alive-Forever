#!/usr/bin/env python3
"""TACTILE figures: force, area, penetration and their relations, as a picture.

A long recording has to be LOOKED at, not only tabulated.  These four panels make the two
things that matter visible at a glance:

  1. the recorded FORCE and AREA traces per leg, so the gait rhythm and the area pulses line up;
  2. AREA against PENETRATION with the geometric law A = pi(2Rh - h^2) drawn over it -- the
     agreement is the validation of the area route;
  3. AREA against FORCE, which is the route that FAILED, shown so the comparison is on the
     record rather than only in a number;
  4. the penetration distribution with the cap-crossing depth marked, which is where the
     remaining saturation comes from.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python run_tactile_figures.py --rig tether --seed 1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "tactile" / "figures"
LEG = ("lf", "lm", "lh", "rf", "rm", "rh")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="tether")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = HERE / "outputs" / "tactile" / f"tactile_{args.rig}_seed{args.seed}"
    z = np.load(d / "tactile.npz")
    t = z["truth/time_s"]
    fn = z["truth/normal_force_N"]
    ft = z["truth/tangential_force_N"]
    h = z["truth/penetration_m"]
    A = z["tactile/area_um2"]
    cap = float(np.asarray(z["tactile/cap_um2"]).ravel()[0])
    R = float(json.loads((d / "summary.json").read_text())["tactile_config"]["tip_radius_m"])
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.6), layout="constrained")

    a = ax[0, 0]
    for j, nm in enumerate(LEG):
        a.plot(t, fn[:, j] * 1e6, lw=0.7, label=nm)
    a.set_xlabel("time (s)")
    a.set_ylabel("normal force (uN)")
    a.set_title("contact FORCE per leg")
    a.legend(fontsize=7, ncol=3)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    for j, nm in enumerate(LEG):
        a.plot(t, A[:, j], lw=0.7, label=nm)
    a.axhline(cap, color="r", ls="--", lw=1, label=f"cap {cap:.0f} um^2")
    a.set_xlabel("time (s)")
    a.set_ylabel("contact area (um^2)")
    a.set_title("contact AREA per leg (geometric route)")
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    m = h > 0
    a.plot(h[m] * 1e6, A[m], ".", ms=2, alpha=0.35, label="measured")
    hh = np.linspace(1e-9, h[m].max() if m.any() else 1e-5, 200)
    a.plot(hh * 1e6, np.minimum(math.pi * (2 * R * hh - hh ** 2), cap) * 1e12, "r-", lw=1.4,
           label="pi(2Rh - h^2) from the tip radius")
    a.axhline(cap, color="k", ls=":", lw=1)
    a.set_xlabel("penetration (um)")
    a.set_ylabel("area (um^2)")
    a.set_title("AREA vs PENETRATION -- the validation of the area route")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(fn[m] * 1e6, A[m], ".", ms=2, alpha=0.35, color="tab:orange")
    a.set_xlabel("normal force (uN)")
    a.set_ylabel("area (um^2)")
    a.set_title("AREA vs FORCE -- the route that FAILED (force is not a physical load here)")
    a.grid(alpha=0.3)
    a.text(0.02, 0.95, f"GRF total {fn.sum(axis=1).mean() * 1e6:.0f} uN\n"
                       f"body weight 10.05 uN\n"
                       f"ratio {fn.sum(axis=1).mean() / 1.005e-5:.0f}x",
           transform=a.transAxes, va="top", fontsize=8,
           bbox=dict(fc="white", alpha=0.8, ec="0.6"))

    fig.suptitle(f"TACTILE RECORDING -- {args.rig}, seed {args.seed} "
                 f"(force from mj_contactForce; area from the measured penetration)",
                 fontsize=12)
    dest = OUT / f"tactile_{args.rig}_seed{args.seed}.png"
    fig.savefig(dest, dpi=115)
    plt.close(fig)
    print("wrote", dest)

    # the numbers behind the picture
    hc = h[m]
    print(f"  contacting samples {m.sum()} of {h.size}")
    print(f"  penetration: median {np.median(hc) * 1e6:.2f} um  p95 {np.percentile(hc, 95) * 1e6:.2f} um"
          f"  max {hc.max() * 1e6:.2f} um")
    # UNIT BUG FIXED HERE: ``cap`` is in um^2 and ``R`` in metres, so the ratio is in metres
    # and the FIRST version printed it while calling it um -- it reported 7.85e12 um.  The
    # correct depth is 7.87 um, which is also where the flat part of the lower-left panel
    # begins, so the figure was right and the printed number was wrong.
    h_cap_m = (cap * 1e-12) / (2 * math.pi * R)
    print(f"  the DOME area reaches the cap at h = {h_cap_m * 1e6:.2f} um; "
          f"fraction of contacts above it: {(hc > h_cap_m).mean():.3f}")
    print(f"  penetration median {np.median(hc) * 1e6:.2f} um vs that cap depth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
