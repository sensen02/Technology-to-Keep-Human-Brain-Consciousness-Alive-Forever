#!/usr/bin/env python3
"""ALL FOUR SEAMS IN ONE CHAIN -- the figure that shows the loop is closed, with its caveats on it.

    sensory drive (body -> brain)      leg joint deviation -> receptor -> tier current
    motor command (brain -> body)      motor-neuron firing IS the muscle activation
    recorder      (brain -> electrode) the electrode reads the post-lesion spikes
    stimulus      (electrode -> brain) injected current changes spikes -> muscles -> motion

Until the stimulus was attached to this loop, the fourth seam had only ever been exercised against a
tier advanced IN ISOLATION, so the chain electrode -> brain -> muscle -> body had a gap where the
muscles should be.  This figure is the closed version: one run, one body, and a stimulus that shows
up in the muscles, in the cameras, and in the electrode.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_four_seams_figure.py --out <path.png>
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
    ap.add_argument("--base", default=str(HERE / "outputs" / "arena" / "stackb_all_channels"))
    ap.add_argument("--stim", default=str(HERE / "outputs" / "arena" / "stackb_stimulated"))
    ap.add_argument("--check", default=str(HERE / "outputs" / "stimulus_stackb_check.json"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "four_seams_figure.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    base, stim = Path(args.base), Path(args.stim)
    chk = json.loads(Path(args.check).read_text())
    sm = json.loads((stim / "stimulus.json").read_text())
    z0, z1 = np.load(base / "episode.npz"), np.load(stim / "episode.npz")
    e0, e1 = np.load(base / "electrode.npz"), np.load(stim / "electrode.npz")

    fig = plt.figure(figsize=(14, 10.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.45, 1.05, 1.0], hspace=0.70, wspace=0.30)

    # --- the closed loop, drawn
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_xlim(-0.5, 10.4)
    ax.set_ylim(0, 4.3)
    nodes = [(0.7, "BODY\njoint angles"), (2.7, "SENSORY DRIVE\nbody -> brain"),
             (4.9, "NEURAL TIER\n12000 neurons"), (7.1, "MOTOR COMMAND\nbrain -> body"),
             (9.2, "MUSCLES -> MOTION")]
    for x, t in nodes:
        ax.add_patch(plt.Rectangle((x - 0.78, 1.35), 1.56, 0.95, fill=False, lw=1.4,
                                   edgecolor="#2b7bba"))
        ax.text(x, 1.82, t, ha="center", va="center", fontsize=8.2)
    for i in range(len(nodes) - 1):
        ax.annotate("", xy=(nodes[i + 1][0] - 0.8, 1.82), xytext=(nodes[i][0] + 0.8, 1.82),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#2b7bba"))
    ax.annotate("", xy=(4.9, 2.55), xytext=(9.2, 2.55),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="crimson",
                                connectionstyle="arc3,rad=0.30"))
    ax.text(7.05, 3.85, "RECORDER   brain -> electrode", ha="center", fontsize=8.6,
            color="crimson")
    ax.annotate("", xy=(4.9, 1.10), xytext=(7.05, 1.10),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="darkgreen",
                                connectionstyle="arc3,rad=0.30"))
    ax.text(5.95, 0.22, "STIMULUS   electrode -> brain", ha="center", fontsize=8.6,
            color="darkgreen")
    ax.text(5.95, 0.62, "(attached to THIS loop for the first time)", ha="center", fontsize=7.6,
            color="darkgreen")
    ax.set_title("one chain, one run: the stimulus acts on the muscles rather than on an isolated "
                 "tier", fontsize=10.0, pad=6)

    # --- before / after, last frame
    for k, (p, lab) in enumerate(((base, "NO STIMULUS"), (stim, "STIMULATED"))):
        ax = fig.add_subplot(gs[1, k])
        f = sorted((p / "frames" / "cam2").glob("*.png"))[-1]
        ax.imshow(np.asarray(Image.open(f)))
        ax.set_title(f"{lab}  (cam2, last frame)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    # --- WHERE the change is, because a whole-frame mean invites "that is noise"
    loc = chk["frame_change_localisation"]
    da = np.asarray(Image.open(sorted((base / "frames" / "cam2").glob("*.png"))[-1]).convert("L"),
                    dtype=float)
    db = np.asarray(Image.open(sorted((stim / "frames" / "cam2").glob("*.png"))[-1]).convert("L"),
                    dtype=float)
    dd = np.abs(da - db)
    b = loc["bbox_rows_cols"]
    r0, r1 = max(0, b[0] - 40), min(dd.shape[0], b[1] + 40)
    c0, c1 = max(0, b[2] - 40), min(dd.shape[1], b[3] + 40)
    axd = fig.add_subplot(gs[1, 2])
    axd.imshow(dd[r0:r1, c0:c1], cmap="inferno", vmin=0, vmax=60)
    axd.set_title("|no stim - stim|\ncropped to the fly", fontsize=8)
    axd.set_xticks([])
    axd.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    lines = [f"acceptance  {chk['overall']}  ({chk['n_checks']} checks, {chk['n_failed']} failed)",
             "",
             f"mode                 {sm['mode']}",
             f"amplitude            {sm['amplitude_nA']:.0f} nA",
             f"capture radius       {sm['capture_radius_um']:.0f} um  (DECLARED)",
             f"neurons recruited    {sm['n_recruited_neurons']} of 12000",
             f"intervals w/ current {sm['n_intervals_with_current']} of {sm['n_intervals']}",
             f"charge delivered     {sm['total_charge_nA_ms']:.1f} nA*ms",
             "",
             f"joint excursion      {chk['amplitude_1000_nA_effect_rad']:.5f} rad",
             f"frames (whole-frame mean) {chk['frame_mean_abs_diff']:.5f}",
             f"  but only {100*loc['fraction_of_frame']:.3f} % of pixels",
             f"  changed >20 grey, inside ONE box",
             f"  where the mean change is {loc['mean_abs_diff_inside_box']:.1f}",
             f"electrode (max |diff|) {chk['electrode_max_abs_diff_mV']:.5f} mV"]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.0)

    # --- electrode traces overlaid
    ax = fig.add_subplot(gs[2, :3])
    t = np.asarray(e0["electrode_time_s"], dtype=float) * 1e3
    ax.plot(t, e0["trace_mV"][0], lw=0.9, color="steelblue", label="no stimulus")
    ax.plot(t, e1["trace_mV"][0], lw=0.9, color="crimson", alpha=0.85, label="stimulated")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("electrode (mV)")
    ax.set_title("the electrode sees the stimulus -- through the network it changed, not by "
                 "pickup", fontsize=9.5)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    ax.text(0, 1, "WHAT THIS DOES NOT SHOW\n\n"
                  "• WHICH neurons get how much\n"
                  "  current is an ASSUMED rule: the\n"
                  "  access map has soma voxel\n"
                  "  positions and nothing else, and\n"
                  "  the capture radius is DECLARED.\n"
                  "  50 um recruits 53 of 7737\n"
                  "  recordable neurons (0.7 %);\n"
                  "  10 um recruits none.  Amplitude\n"
                  "  effects are meaningful; the\n"
                  "  SPATIAL pattern is not.\n\n"
                  "• at this dt the recovery phase is\n"
                  "  never sampled, so the two modes\n"
                  "  differ by SIGN alone, not by\n"
                  "  charge balance.\n\n"
                  "• NO charge-safety claim.\n\n"
                  "• the fly still does not stand or\n"
                  "  walk under stimulation either.",
            va="top", fontsize=7.6)

    fig.suptitle("All four seams in ONE loop: the stimulus reaches the muscles, the cameras and the "
                 "electrode", fontsize=11.5, y=0.975)
    fig.text(0.5, 0.012,
             "HAND-BUILT PROTOTYPE, simulator only, no real fly data anywhere.  Guards: the substep "
             "restructuring is bit-identical (0 of 120000 entries differ), and at 1e-6 nA the "
             "excursion changes by exactly 0.0 rad.",
             ha="center", fontsize=8, style="italic", color="dimgray")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
