#!/usr/bin/env python3
"""FIGURES for the recording pipeline -- the part you LOOK at.

The rule this project runs on is that a long run's result is verified by looking at
it, not only by a table of numbers.  These figures are that look, and they are also
what goes to the desktop delivery directory.

Run (PROJECT environment):
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python run_recording_figures.py --seeds 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import gridspec          # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "electrode_payload"
FIG = OUT / "figures"
LEG_NAMES = ("lf", "lm", "lh", "rf", "rm", "rh")


def _img(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype != np.uint8:
        a = (255 * np.clip(a, 0, 1)).astype("uint8")
    return a[:, :, :3] if a.ndim == 3 and a.shape[2] == 4 else a


def fig_camera_strip(ep_dir: Path, tag: str, n_pick: int = 6) -> Path | None:
    """A strip of real camera frames -- the raw material of the whole exercise."""
    from PIL import Image
    for cam, sub in (("worldcam", "frames"), ("detailcam", "frames_detailcam")):
        fs = sorted((ep_dir / sub).glob("*.png"))
        if not fs:
            continue
        pick = np.linspace(0, len(fs) - 1, min(n_pick, len(fs))).astype(int)
        fig = plt.figure(figsize=(3.0 * len(pick), 2.6), layout="constrained")
        gs = gridspec.GridSpec(1, len(pick), figure=fig)
        for k, i in enumerate(pick):
            ax = fig.add_subplot(gs[0, k])
            ax.imshow(np.asarray(Image.open(fs[i])))
            ax.set_title(f"{cam} frame {i}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{ep_dir.name} -- {cam} as recorded (no ground truth drawn)",
                     fontsize=11)
        dest = FIG / f"fig_camera_strip_{cam}_{tag}.png"
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        print(f"  {dest}")
    return None


def fig_trajectory(ep_dir: Path, tag: str) -> Path | None:
    """Overhead trajectory: photos-only reconstruction against the simulator."""
    vp, np_ = ep_dir / "vision.json", ep_dir / "episode.npz"
    if not vp.exists() or not np_.exists():
        return None
    v = json.loads(vp.read_text())
    d = np.load(np_)
    t = d["truth/time_s"].astype(float)
    th = d["truth/thorax_mm"].astype(float)
    ft = np.asarray(v["frame_time_s"], dtype=float)
    vx = np.asarray(v["derived"]["x_mm"], dtype=float)
    vy = np.asarray(v["derived"]["y_mm"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    ax = axes[0]
    ax.plot(th[:, 0], th[:, 1], "-", color="0.55", lw=2.6, label="simulator (ground truth)")
    ax.plot(vx, vy, "-", color="tab:red", lw=1.2, label="from photos only")
    ax.plot(vx[0], vy[0], "o", color="tab:green", ms=6)
    ax.plot(vx[-1], vy[-1], "s", color="tab:blue", ms=6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("path recovered from images")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    gx = np.interp(ft, t, th[:, 0])
    gy = np.interp(ft, t, th[:, 1])
    ax.plot(ft, vx - gx, label="dx")
    ax.plot(ft, vy - gy, label="dy")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error (mm)")
    ax.set_title(f"position error, median {np.median(np.hypot(vx-gx, vy-gy)):.3f} mm "
                 f"= {np.median(np.hypot(vx-gx, vy-gy))/v['camera']['mm_per_px']:.2f} px")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(ft, v["observed"]["area_px"], label="blob area (px)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("area (px)", color="tab:blue")
    ax2 = ax.twinx()
    duty = np.mean(d["truth/contact_present"].astype(float), axis=1)
    ax2.plot(t, duty, color="tab:orange", lw=1.0, label="true legs in contact / 6")
    ax2.set_ylabel("contact fraction", color="tab:orange")
    ax.set_title(f"observed signal, alt {v['gait']['alternation_frequency_hz']:.2f} Hz "
                 f"from {v['gait']['signal_used']}")
    ax.grid(alpha=0.3)
    fig.suptitle(f"{ep_dir.name} -- reconstruction from camera photos vs the simulator",
                 fontsize=12)
    dest = FIG / f"fig_trajectory_{tag}.png"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    print(f"  {dest}")
    return dest


def fig_touch(cmp_doc: dict, tag: str) -> Path | None:
    """Inferred touch against true contact, with the chance level shown."""
    eps = cmp_doc.get("episodes", {})
    if not eps:
        return None
    first = next(iter(eps.values()))
    if not first.get("touch"):
        return None
    cams = sorted({c for e in eps.values() for c in (e.get("touch") or {})})
    main_cam = "worldcam" if "worldcam" in cams else cams[0]
    n = len(eps)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), layout="constrained")
    ax = axes[0, 0]
    accs = [e["touch"][main_cam]["pooled_accuracy"] for e in eps.values()]
    chs = [e["touch"][main_cam]["pooled_chance_accuracy"] for e in eps.values()]
    names = list(eps.keys())
    xs = np.arange(n)
    ax.bar(xs - 0.2, accs, 0.4, label="reconstructed from video")
    ax.bar(xs + 0.2, chs, 0.4, color="0.6", label="always-in-contact baseline")
    ax.set_xticks(xs)
    ax.set_xticklabels([nm.replace("episode_", "") for nm in names], rotation=90,
                       fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("frame-level contact accuracy")
    ax.set_title("touch inferred from motion vs truth (pooled over 6 legs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[0, 1]
    per = np.array([[p["accuracy"] for p in e["touch"][main_cam]["per_leg"]]
                    for e in eps.values()])
    ax.boxplot([per[:, i] for i in range(per.shape[1])], tick_labels=list(LEG_NAMES))
    ax.axhline(0.5, color="r", ls="--", lw=1, label="0.5 reference")
    ax.set_ylabel("per-leg accuracy")
    ax.set_title("per-leg contact accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1, 0]
    for i, nm in enumerate(names):
        sc = eps[nm]["touch"][main_cam]["phase_sensitivity"]["scan"]
        ax.plot([s["offset_ms"] for s in sc], [s["pooled_accuracy"] for s in sc],
                marker=".", label=nm.replace("episode_", ""))
    ax.set_xlabel("global phase offset applied to the reconstruction (ms)")
    ax.set_ylabel("accuracy")
    ax.set_title("PHASE SENSITIVITY: how much the answer depends on the one unknown")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.axis("off")
    lines = []
    for nm, e in eps.items():
        t = e["touch"][main_cam]
        ps = t["phase_sensitivity"]
        lines.append(
            f"{nm.replace('episode_', ''):<26s} acc {t['pooled_accuracy']:.3f} "
            f"(baseline {t['pooled_chance_accuracy']:.3f}) "
            f"lag {t['best_lag_ms']:+.0f} ms | best offset "
            f"{ps['best_offset_ms']:+.0f} ms -> {ps['best_pooled_accuracy']:.3f}")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
            fontsize=7.5)
    ax.set_title("per-episode summary", fontsize=10)
    fig.suptitle(f"TOUCH FROM MOTION -- {tag}", fontsize=13)
    dest = FIG / f"fig_touch_{tag}.png"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    print(f"  {dest}")
    return dest


def fig_payload(cmp_doc: dict, tag: str) -> Path | None:
    """The electrode's own weight: paired per-seed effect."""
    pe = cmp_doc.get("payload_effect", {})
    keys = [k for k, v in pe.items() if v.get("per_seed")]
    if not keys:
        return None
    metrics = ["d_mean_speed_mm_s", "d_path_mm", "d_thorax_z_mm", "d_stride_hz",
               "d_legs_in_contact"]
    labels = ["mean speed\n(mm/s)", "path\n(mm)", "thorax z\n(mm)", "stride\n(Hz)",
              "legs in contact"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    ax = axes[0]
    x = np.arange(len(metrics))
    w = 0.8 / len(keys)
    for j, load in enumerate(keys):
        m = [pe[load]["summary"][k]["mean"] for k in metrics]
        lo = [pe[load]["summary"][k]["ci95_low"] for k in metrics]
        hi = [pe[load]["summary"][k]["ci95_high"] for k in metrics]
        yerr = [np.array(m) - np.array(lo), np.array(hi) - np.array(m)]
        ax.bar(x + j * w - 0.4 + w / 2, m, w, yerr=yerr, capsize=3,
               label=f"{load} (paired, n=4)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("change vs same-seed control")
    ax.set_title("electrode payload effect (paired; bars are t-like 95 % intervals)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    for j, load in enumerate(keys):
        for i, ps in enumerate(pe[load]["per_seed"]):
            ax.plot([0, 1], [0, ps["d_mean_speed_mm_s"]], "-o", ms=3,
                    color=f"C{j}", alpha=0.8,
                    label=load if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["control", "with payload"])
    ax.set_ylabel("mean speed (mm/s)")
    ax.set_title("per-seed pairing (each line is one seed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"DOES THE ELECTRODE'S OWN WEIGHT CHANGE THE WALK? -- {tag}", fontsize=12)
    dest = FIG / f"fig_payload_{tag}.png"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    print(f"  {dest}")
    return dest


def fig_electrode(ep_dir: Path, tag: str) -> Path | None:
    """What the electrode records, against its own noise floor."""
    p = ep_dir / "episode.npz"
    if not p.exists():
        return None
    d = np.load(p)
    t = d["elec/time_s"].astype(float)
    rec = d["elec/recorded_uV"].astype(float)
    art = d["elec/artefact_uV"].astype(float)
    noise = d["elec/noise_uV"].astype(float)
    fs = float(d["elec/fs_Hz"])
    ct = d["truth/time_s"].astype(float)
    cont = d["truth/contact_present"].astype(float)
    # The first 30 ms are dropped from the DISPLAY: the declared 1 Hz high-pass starts
    # from zero state, so its step response puts a ~7 mV transient in the first samples
    # that would flatten everything else.  The samples are kept in the data -- this is a
    # display window, and saying so is the difference between a window and a filter.
    t0 = float(t[0]) + 0.03
    m = t >= t0
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.0), layout="constrained", sharex=False)
    ax = axes[0]
    ax.plot(t[m], rec[m], lw=0.5, color="tab:blue",
            label="recorded (modelled artefact + chain noise)")
    ax.plot(t[m], art[m], lw=0.9, color="tab:orange", alpha=0.8,
            label="modelled artefact only")
    ax.set_xlim(t0, float(t[-1]))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("uV")
    ax.set_title(f"{ep_dir.name}: electrode trace, rms {rec.std():.1f} uV "
                 f"(artefact {art.std():.1f}, noise {noise.std():.1f})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax = axes[1]
    f = np.fft.rfftfreq(len(rec), 1 / fs)
    P = np.abs(np.fft.rfft(rec - rec.mean()))
    Pn = np.abs(np.fft.rfft(noise - noise.mean()))
    ax.loglog(f[1:], P[1:], lw=0.8, label="recorded")
    ax.loglog(f[1:], Pn[1:], lw=0.8, color="0.6", label="chain noise only")
    ax.set_xlabel("Hz")
    ax.set_ylabel("|FFT| (uV)")
    ax.set_title("spectrum: the recorded trace has structure the noise does not")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axes[2]
    ax.plot(ct, cont.mean(axis=1), color="tab:green", lw=1.2,
            label="true contact fraction over 6 legs")
    ax.set_ylabel("contact fraction")
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("THE ELECTRODE SIDE (modelled trace -- not a measured recording)",
                 fontsize=12)
    dest = FIG / f"fig_electrode_{tag}.png"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    print(f"  {dest}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tag = args.tag or ("seed" + "".join(str(s) for s in seeds))
    for seed in seeds:
        ep = OUT / f"episode_control_seed{seed}"
        if ep.exists():
            fig_camera_strip(ep, tag)
            fig_trajectory(ep, tag)
            fig_electrode(ep, tag)
    cmp_path = OUT / "comparison.json"
    if cmp_path.exists():
        doc = json.loads(cmp_path.read_text())
        fig_touch(doc, tag)
        fig_payload(doc, tag)
    print(f"figures in {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
