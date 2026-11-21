"""Plot actual saved sensor and closed-loop behavior traces."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BEHAVIOR = HERE.parent / "20261118T002600Z-ascending-behavior-comparison"

COLORS = {"native": "#27364c", "boundary": "#237a55", "old": "#b9392e", "repaired": "#245ad3"}


def draw_panel(axes, seed):
    old = BEHAVIOR / "FORMAL" / f"seed{seed}"
    new = HERE / "FORMAL" / f"seed{seed}"
    paths = (
        ("native", "Original brain + engineering receptors", old / "native_full"),
        ("boundary", "Controlled reference", old / "reference_boundary"),
        ("old", "Old translation", old / "translated"),
        ("repaired", "Timing repair, DN feedback retained", new / "repaired"),
    )
    for key, label, folder in paths:
        with np.load(folder / "TRACE.npz") as body:
            pos = body["position"] - body["position"][0]
            yaw = np.rad2deg(np.unwrap(body["yaw"]) - body["yaw"][0])
        color = COLORS[key]
        axes[0].plot(pos[:, 0], pos[:, 1], color=color, label=label, lw=1.6)
        axes[0].scatter(pos[-1, 0], pos[-1, 1], color=color, s=11)
        axes[1].plot(np.arange(len(yaw))/10000, yaw, color=color, lw=1.6, label=label)
    axes[0].set(title=f"Seed {seed} | path", xlabel="Forward axis (mm)", ylabel="Lateral axis (mm)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].set(title=f"Seed {seed} | turn angle", xlabel="Time (s)", ylabel="Yaw change (°)")
    for ax in axes:
        ax.grid(alpha=.18)


def main():
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(2, 2, figsize=(10.3, 7.8), height_ratios=[1, 1])
        for axes, seed in zip(ax, (10901, 11311), strict=True):
            draw_panel(axes, seed)
        fig.suptitle("Front-touch timing repair | frozen brain and body", fontweight="bold")
        handles, labels = ax[0, 1].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="lower center", frameon=False,
                   bbox_to_anchor=(.5, .007))
        fig.tight_layout(rect=(0, .075, 1, .96))
        fig.savefig(HERE / "CLOSED_LOOP.png", dpi=155)
        plt.close(fig)

        seed = 10901
        old = BEHAVIOR / "FORMAL" / f"seed{seed}" / "NEURAL.npz"
        new = HERE / "FORMAL" / f"seed{seed}" / "NEURAL.npz"
        with np.load(old) as hist, np.load(new) as live:
            groups = (
                ("Old fixed-neck arrival", hist["an_arrived"][:, 2], "old"),
                ("Repair fixed-neck arrival", live["an_arrived"][:, 0], "repaired"),
                ("Reference brain input", hist["brain_an_events"][:, 1], "boundary"),
                ("Old translation brain input", hist["brain_an_events"][:, 2], "old"),
                ("Repair brain input", live["brain_an_events"][:, 0], "repaired"),
            )
            fig, axes = plt.subplots(2, 1, figsize=(10.3, 4.2), sharex=True)
            for c, ax in enumerate(axes):
                for y, (label, spikes, key) in enumerate(groups):
                    hits = np.flatnonzero(spikes[:850, c]) / 10
                    ax.vlines(hits, y - .31, y + .31, color=COLORS[key], lw=1.45)
                ax.set_yticks(range(len(groups)), [g[0] for g in groups])
                ax.set_ylim(len(groups)-.45, -.55)
                ax.set_title(("Left" if c == 0 else "Right") + " front-leg input", loc="left")
                ax.grid(axis="x", alpha=.2)
            axes[-1].set(xlim=(0, 85), xlabel="Time (ms)")
            fig.suptitle("Same early contact | arrival and brain input timing", fontweight="bold")
            fig.tight_layout()
            fig.savefig(HERE / "ARRIVAL_TIMING.png", dpi=155)


if __name__ == "__main__":
    main()
