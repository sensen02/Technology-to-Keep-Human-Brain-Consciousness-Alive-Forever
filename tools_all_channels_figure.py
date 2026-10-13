#!/usr/bin/env python3
"""One figure a human can read in one look: ONE run, TWO channels, ONE clock.

The point of the figure is the claim the platform now makes -- the camera frames and the electrode
trace come from a single call to `run_six_leg_neural_loop.run`, so they share a time base because one
loop call feeds both sinks, NOT because two logs were aligned afterwards.  Three things it must show
honestly:

  * the 20 frames sit ON the electrode's interval clock, at the indices stored in the artifact;
  * the electrode samples FINER than the frames (0.5 ms neural substeps vs 5 ms command intervals),
    which is why it needs its own axis -- so the frames are drawn as markers on a continuous trace;
  * the amplitude is ILLUSTRATIVE, and the visual channel is absent from this body, both stated on
    the figure rather than left for the reader to discover.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_all_channels_figure.py --out <path.png>
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
    ap.add_argument("--check", default=str(HERE / "outputs" / "all_channels_check.json"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "all_channels_figure.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ep = Path(args.episode)
    chk = json.loads(Path(args.check).read_text())
    z = np.load(ep / "episode.npz")
    e = np.load(ep / "electrode.npz")

    tr = e["trace_mV"]
    t_el = np.asarray(e["electrode_time_s"], dtype=float) * 1e3     # ms
    t_in = np.asarray(e["interval_time_s"], dtype=float) * 1e3
    idx = np.asarray(z["truth/frame_interval_index"], dtype=np.int64)
    t_fr = np.asarray(z["truth/time_s"], dtype=float) * 1e3
    stride = int(np.asarray(z["truth/stride"]).ravel()[0])

    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 1.0, 1.0], hspace=0.55, wspace=0.25)

    for k, cf in enumerate(["cam0", "cam2", "cam4", "cam5"]):
        ax = fig.add_subplot(gs[0, k])
        p = ep / "frames" / cf / f"f{len(t_fr)-1:05d}.png"
        ax.imshow(np.asarray(Image.open(p)))
        ax.set_title(f"{cf}   frame {len(t_fr)-1}   t = {t_fr[-1]:.0f} ms", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[1, :])
    ax.plot(t_el, tr[0], lw=0.8, color="crimson", label="electrode ch0 (0.5 ms substeps)")
    ax.plot(t_el, tr[1], lw=0.8, color="steelblue", alpha=0.85, label="electrode ch1")
    on_clock = np.interp(t_fr, t_el, tr[0])
    ax.plot(t_fr, on_clock, "k.", ms=7, zorder=5,
            label=f"{t_fr.size} rendered frames (stride {stride}) ON the same clock")
    ax.set_xlabel("time (ms)  --  ONE interval counter served both sinks")
    ax.set_ylabel("electrode (mV)")
    ax.set_title(f"ONE run() call, ONE body, ONE tier: {tr.shape[0]} electrode channels + "
                 f"{t_fr.size} frames x 6 cameras", fontsize=10)
    ax.legend(fontsize=7.5, ncol=3, loc="upper right")

    ax = fig.add_subplot(gs[2, :2])
    ax.plot(t_el, tr[0], lw=0.8, color="crimson")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("mV")
    ax.set_title(f"peak {np.abs(tr).max():.5f} mV -- the spike->current kernel is ILLUSTRATIVE,\n"
                 f"so the AMPLITUDE is never established; "
                 f"{int(e['n_neurons_with_soma'][0])}/{int(e['n_tier_neurons'][0])} neurons have a soma",
                 fontsize=8.5)

    ax = fig.add_subplot(gs[2, 2:])
    ax.axis("off")
    lines = []
    for c in chk["results"]:
        head, _, tail = c["check"].partition(":")
        lines.append(f"{'PASS' if c['ok'] else 'FAIL'}  {head.strip():<8} {tail.strip()[:46]}")
    ax.text(0.0, 1.0, f"acceptance: {chk['overall']}  ({chk['n_checks']} checks, "
                      f"{chk['n_failed']} failed)\n\n" + "\n".join(lines),
            va="top", family="monospace", fontsize=6.8)

    fig.suptitle("Camera ring + electrode recorded TOGETHER on the muscle-driven body "
                 "-- vision is structurally absent, and that is measured", fontsize=11.5)
    fig.text(0.5, 0.013,
             "HAND-BUILT PROTOTYPE, simulator only, no real fly data anywhere.  Two channels, not "
             "three: binding the visual front end to this body raises AttributeError "
             "('MjData' has no 'get_raw_vision').",
             ha="center", fontsize=8, style="italic", color="dimgray")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
