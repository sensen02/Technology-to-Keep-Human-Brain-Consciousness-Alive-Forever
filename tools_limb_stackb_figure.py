#!/usr/bin/env python3
"""The camera ring now measures the fly's OWN limbs -- one figure, with the limits on the figure.

WHAT THIS SHOWS.  Stack B's episodes carry 20 markers on the fly itself, so `limb_tracking.py` can
run on the body that the connectome drives, and a joint angle comes out.  The figure shows the
tracked markers against the simulator's own positions and the one bone that clears ten frames.

WHAT IT REFUSES TO HIDE.  119 of 400 frame-marker slots are accepted and ONE bone clears ten frames,
because a six-camera ring around a self-occluding body leaves most markers visible from fewer than
the three cameras triangulation needs.  Frame 0's identity seed is the simulator standing in for a
human annotation.  Both are printed on the figure rather than left to a reader to discover.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_limb_stackb_figure.py --out <path.png>
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
    ap.add_argument("--episode", default=str(HERE / "outputs" / "arena" / "stackb_all_channels"))
    ap.add_argument("--check", default=str(HERE / "outputs" / "limb_stackb_check.json"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "limb_stackb_figure.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ep = Path(args.episode)
    chk = json.loads(Path(args.check).read_text())
    lt = json.loads((ep / "limb_tracking.json").read_text())
    z = np.load(ep / "episode.npz")
    truth = np.asarray(z["truth/marker_fly_world_mm"], dtype=float)
    segs = list(json.loads((ep / "markers.json").read_text())["marker_body_segments"])

    fig = plt.figure(figsize=(14.0, 10.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.05, 1.05], hspace=0.85, wspace=0.30)

    # --- the fly, seen by two cameras, with its own markers
    for k, (cf, fr) in enumerate((("cam1", 0), ("cam4", 19))):
        ax = fig.add_subplot(gs[0, k])
        p = ep / "frames" / cf / f"f{fr:05d}.png"
        ax.imshow(np.asarray(Image.open(p)))
        ax.set_title(f"{cf}, frame {fr}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    n_acc = lt["markers_accepted"]
    n_slot = lt["marker_slots"]
    lines = [f"acceptance: {chk['overall']}  ({chk['n_checks']} checks)",
             "",
             f"markers on the fly          20",
             f"frame-marker slots          {n_slot}",
             f"accepted                    {n_acc}  ({100*n_acc/n_slot:.0f} %)",
             f"position error (median)     {lt['marker_position_error_mm']['median']:.4f} mm",
             f"joint angles measured       {(lt.get('joint_angle_series') or {}).get('n_angles')}",
             f"angle error (median)        "
             f"{(lt.get('joint_angle_series') or {}).get('error_deg_median'):.3f} deg",
             f"bones in >= 10 frames       "
             f"{sum(1 for v in lt.get('bones_with_both_ends', {}).values() if v >= 10)} of 12",
             "",
             "projection offsets applied  "
             f"{lt['projection_offsets']['applied']}  (measured: on = 0/20 markers)"]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.2)

    # --- tracks: which markers were ever accepted, per marker
    ax = fig.add_subplot(gs[1, :])
    fin = lt.get("tracks_finite_fraction", {})
    vals = [fin.get(s, 0.0) * 100 for s in segs]
    ax.bar(range(len(segs)), vals, color=["#2b7bba" if v > 0 else "#cccccc" for v in vals])
    ax.set_xticks(range(len(segs)))
    ax.set_xticklabels(segs, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("frames tracked (%)")
    ax.set_title("which of the fly's own markers were ever tracked -- the blanks are the "
                 "self-occlusion limit, not a tuning failure", fontsize=9.5, pad=10)

    # --- the joint angle time series for the bones that were measured
    ax = fig.add_subplot(gs[2, :2])
    csv = ep / "limb_joint_angles.csv"
    rows = [r.split(",") for r in csv.read_text().splitlines() if r and not r.startswith("t_s")]
    hdr = csv.read_text().splitlines()[0].split(",")
    idx = {h: i for i, h in enumerate(hdr)}
    by_bone: dict[str, list] = {}
    for r in rows:
        by_bone.setdefault(r[idx["bone"]], []).append((float(r[idx["t_s"]]),
                                                       float(r[idx["elev_meas_deg"]]),
                                                       float(r[idx["elev_true_deg"]])))
    for b, v in sorted(by_bone.items()):
        if len(v) < 3:          # a bone with two points is drawn as a stray dot, not a series
            continue
        t = [x[0] * 1e3 for x in v]
        ax.plot(t, [x[1] for x in v], "o-", ms=4, lw=1.0, label=f"{b} (measured)")
        ax.plot(t, [x[2] for x in v], "x--", ms=4, lw=0.8, alpha=0.7, label=f"{b} (truth)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("elevation angle (deg)")
    ax.set_title("the joint angle that comes out, against the simulator's own value",
                 fontsize=9.5, pad=10)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    e = [float(r[idx["angle_err_deg"]]) for r in rows]
    ax.text(0, 1, "WHAT THIS DOES NOT SHOW\n\n"
                  "• only ONE bone clears 10 frames.\n"
                  "  A 6-camera ring around a\n"
                  "  self-occluding body leaves most\n"
                  "  markers in 1-2 cameras per frame;\n"
                  "  triangulation needs 3.\n\n"
                  "• frame 0's marker identity is the\n"
                  "  SIMULATOR standing in for a human\n"
                  "  annotation -- every marker is the\n"
                  "  same sphere, so the pixels alone\n"
                  "  cannot say which is which.\n\n"
                  "• the 'truth' is the simulator too.\n"
                  "  There is no real-fly recording\n"
                  "  anywhere in this project.\n\n"
                  "• all six legs bind the FEMUR, so a\n"
                  "  bone here is marker-to-marker,\n"
                  "  not joint-centre-to-joint-centre.",
            va="top", fontsize=7.6)

    fig.suptitle("Stack B: the connectome-driven body now has its OWN markers, "
                 "and a joint angle can be measured", fontsize=11.5, y=0.975)
    fig.text(0.5, 0.015,
             "HAND-BUILT PROTOTYPE, simulator only.  Marker position error median "
             f"{lt['marker_position_error_mm']['median']:.4f} mm; "
             f"{n_acc}/{n_slot} frame-marker slots accepted.",
             ha="center", fontsize=8, style="italic", color="dimgray")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
