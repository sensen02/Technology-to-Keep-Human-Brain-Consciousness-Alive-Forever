"""Round-1 closed sensorimotor loop: what does the neural tier actually contribute?

WHAT IS RUN
-----------
Two PAIRED arms, same seed, same spawn state, only the command source differs:

  cpg_baseline     FlyGym's demo tripod CPG, untouched.  ENGINEERING BASELINE.  This
                   arm is bit-identical to a plain CPG walk (asserted in the selftest).
  neural_modulated the SAME CPG still produces the gait; the BANC-derived neural
                   tier's descending firing modulates ONLY (i) the CPG intrinsic
                   frequency and (ii) a left/right intrinsic-frequency asymmetry.

Sensation in the neural arm: leg contact FORCE and leg JOINT ANGLE -> MechanoReceptor
-> external current into the tier neurons annotated as mechanosensory for that leg's
nerve.  No vision is implemented and none is claimed.

OUTPUTS (all under outputs/embodied_body/)
-----------------------------------------
  loop_comparison.png            the figure (opened and looked at, not just written)
  loop_report.json               every number, every limitation, wall clock, peak RAM
  loop_episode_<arm>_seed<k>.npz one episode per arm per seed, truth/observed/commands
                                 in SEPARATE namespaces
  loop_neural_walk_seed<k>.mp4   the neural-modulated walk
  loop_frame_montage.png         sampled video frames, so the ground can be checked

Run (BODY environment):
    ./venv_body/bin/python run_embodied_loop.py
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs", "embodied_body")

from engine.embodied import select_gl_backend                      # noqa: E402
from engine.embodied.adapters import (HONESTY, UNITS, build_neural_tier,  # noqa: E402
                                      ReceptorConfig, TierConfig)
from engine.embodied.loop import (LoopConfig, MultirateScheduler,  # noqa: E402
                                  heading_yaw_rad)

#: PRE-REGISTERED run parameters.  Declared here, before any measurement, and not
#: changed afterwards.
DEMO_SEEDS = (0, 1, 2)
DEMO_SECONDS = 8.0
RENDER_SEED = 0
FINGERPRINT_NOTE = (
    "the CPG is FlyGym's demo tripod CPG in BOTH arms; the neural tier modulates the "
    "gait, it does not generate it")


def peak_ram_mb():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def run_arm(arm, seed, built, gl, seconds, render=False):
    cfg = LoopConfig(arm=arm, seed=seed, duration_s=seconds, gl_backend=gl)
    sch = MultirateScheduler(cfg, built_tier=built)
    if render:
        sch.make_renderer(camera_res=(240, 320), output_fps=25)
    t0 = time.perf_counter()
    ep = sch.run()
    wall = time.perf_counter() - t0
    frames = sch.renderer_frames() if render else []
    ownership = sch.check_ownership()
    calibration = sch.calibration
    sch.close()
    return ep, wall, frames, ownership, calibration


def smooth(y, n):
    """Centred moving average with edge padding, for DISPLAY only (never for metrics).

    The thorax speed and heading wobble at the gait frequency (7-12 Hz), so the raw
    traces are drawn thin and the smoothed trace thick; every labelled number in the
    report is computed from the raw arrays.
    """
    y = np.asarray(y, float)
    if n <= 1 or y.size < n:
        return y
    pad = n // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    k = np.ones(n) / n
    return np.convolve(yp, k, mode="valid")[:y.size]


def make_figure(runs, path, tier_coverage, seeds=DEMO_SEEDS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 13.2), layout="constrained")
    (ax_tr, ax_sp), (ax_yaw, ax_rate), (ax_contact, ax_cmd) = axes

    colors = {"cpg_baseline": "#1f4e9c", "neural_modulated": "#c0392b"}
    labels = {"cpg_baseline": "cpg_baseline (FlyGym tripod CPG, ENGINEERING BASELINE)",
              "neural_modulated": "neural_modulated (same CPG + BANC descending command)"}

    # --- (a) top-down trajectory -------------------------------------------
    for arm in ("cpg_baseline", "neural_modulated"):
        for k, seed in enumerate(seeds):
            ep = runs[(arm, seed)][0]
            p = ep.truth["thorax_mm"]
            ax_tr.plot(p[:, 0], p[:, 1], color=colors[arm], lw=2.0 if seed == 0 else 1.0,
                       alpha=1.0 if seed == 0 else 0.4,
                       label=(labels[arm] if k == 0 else None))
            ax_tr.plot(p[0, 0], p[0, 1], "o", color="black", ms=4)
    ax_tr.set_xlabel("x (mm, model units)")
    ax_tr.set_ylabel("y (mm, model units)")
    ax_tr.set_title("(a) thorax path on the ground plane, %g s\n"
                    "black dot = common start; pale lines = other seeds"
                    % DEMO_SECONDS, fontsize=9)
    ax_tr.legend(fontsize=7, loc="best")
    ax_tr.grid(alpha=0.3)
    ax_tr.set_aspect("equal", adjustable="datalim")

    # --- (b) speed over time ------------------------------------------------
    for arm in ("cpg_baseline", "neural_modulated"):
        curves = []
        for seed in seeds:
            ep = runs[(arm, seed)][0]
            p = ep.truth["thorax_mm"][:, :2]
            t = ep.truth["time_s"]
            v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(t)
            curves.append(np.concatenate([[v[0]], v]))
        C = np.asarray(curves)
        tc = runs[(arm, seeds[0])][0].truth["time_s"]
        m_sm = smooth(C.mean(axis=0), 21)
        ax_sp.plot(tc, C.mean(axis=0), color=colors[arm], lw=0.6, alpha=0.35)
        ax_sp.plot(tc, m_sm, color=colors[arm], lw=2.2, label=labels[arm])
        sd_sm = smooth(C.std(axis=0), 21)
        ax_sp.fill_between(tc, m_sm - sd_sm, m_sm + sd_sm, color=colors[arm], alpha=0.18)
    ax_sp.set_xlabel("time (s)")
    ax_sp.set_ylabel("thorax speed (mm/s)")
    ax_sp.set_title("(b) speed over time, mean over %d paired seeds\n"
                    "thick = 105 ms moving average (display only), thin = raw, band = sd "
                    "across seeds" % len(seeds), fontsize=8.5)
    ax_sp.legend(fontsize=7, loc="best")
    ax_sp.grid(alpha=0.3)

    # --- (c) heading --------------------------------------------------------
    for arm in ("cpg_baseline", "neural_modulated"):
        for k, seed in enumerate(seeds):
            ep = runs[(arm, seed)][0]
            yawd = np.degrees(np.unwrap(ep.yaw_series_rad))
            ax_yaw.plot(ep.truth["time_s"], yawd, color=colors[arm], lw=0.5, alpha=0.35)
            ax_yaw.plot(ep.truth["time_s"], smooth(yawd, 31), color=colors[arm],
                        lw=2.2 if seed == 0 else 1.0,
                        alpha=1.0 if seed == 0 else 0.4,
                        label=(labels[arm] if k == 0 else None))
    ax_yaw.set_xlabel("time (s)")
    ax_yaw.set_ylabel("thorax heading (deg, unwrapped)")
    ax_yaw.set_title("(c) heading: the descending command steers through the left/right "
                     "frequency asymmetry\nthin = raw (gait-cycle yaw wobble), thick = "
                     "155 ms moving average (display only)", fontsize=8.5)
    ax_yaw.legend(fontsize=7, loc="best")
    ax_yaw.grid(alpha=0.3)

    # --- (d) descending rate ------------------------------------------------
    for k, seed in enumerate(seeds):
        ep = runs[("neural_modulated", seed)][0]
        ax_rate.plot(ep.commands["time_s"], ep.commands["rate_total_hz"],
                     lw=2.0 if seed == 0 else 1.0, alpha=1.0 if seed == 0 else 0.5,
                     label=("descending readout rate, seed %d%s"
                            % (seed, " (pale = other seeds)" if k == 0 and len(seeds) > 1
                               else "")))
    r0 = runs[("neural_modulated", seeds[0])][4]["reference_rates_hz"]["total_hz"]
    ax_rate.axhline(r0, color="black", ls="--", lw=1.2,
                    label="reference from the pre-registered standing calibration "
                          "(%.1f Hz)" % r0)
    ax_rate.set_xlabel("time (s)")
    ax_rate.set_ylabel("descending readout rate (Hz)")
    n_read = tier_coverage["descending_readout"][
        "in_tier_with_at_least_one_modelled_incoming_edge"]
    ax_rate.set_title("(d) the tier's own signal: %d descending neurons that carry at "
                      "least one modelled in-tier edge" % n_read, fontsize=9)
    ax_rate.legend(fontsize=6.5, loc="best")
    ax_rate.grid(alpha=0.3)

    # --- (e) contact pattern ------------------------------------------------
    ax_contact.axis("off")
    ax_contact.set_title("(e) leg contact pattern, seed %d (black = in contact)"
                         % seeds[0], fontsize=9)
    for j, arm in enumerate(("cpg_baseline", "neural_modulated")):
        ep = runs[(arm, seeds[0])][0]
        c = ep.truth["contact_present"].astype(float).T
        axc = ax_contact.inset_axes([0.16, 0.56 - 0.48 * j, 0.80, 0.38])
        axc.imshow(c, aspect="auto", interpolation="nearest", cmap="Greys",
                   extent=[float(ep.truth["time_s"][0]), float(ep.truth["time_s"][-1]),
                           c.shape[0] - 0.5, -0.5])
        axc.set_yticks(range(c.shape[0]))
        axc.set_yticklabels(["lf", "lm", "lh", "rf", "rm", "rh"], fontsize=6)
        axc.set_ylabel("leg", fontsize=7)
        axc.set_title(arm, fontsize=7.5)
        if j == 0:
            axc.set_xticklabels([])
    axc.set_xlabel("time (s)", fontsize=8)

    # --- (f) commands -------------------------------------------------------
    for k, seed in enumerate(seeds):
        ep = runs[("neural_modulated", seed)][0]
        ax_cmd.plot(ep.commands["time_s"], ep.commands["speed_scale"],
                    color="#1f4e9c", lw=2.0 if seed == 0 else 1.0,
                    alpha=1.0 if seed == 0 else 0.5,
                    label=("speed scale (left axis), seed %d%s"
                           % (seed, " (pale = other seeds)" if k == 0 and len(seeds) > 1
                              else "")))
    ax_cmd.axhline(1.0, color="black", ls="--", lw=1.0,
                   label="neutral value = baseline CPG frequency")
    ax_cmd.set_xlabel("time (s)")
    ax_cmd.set_ylabel("CPG intrinsic-frequency scale (dimensionless)")
    ax_cmd.set_title("(f) the two commanded knobs, decoded from the PREVIOUS completed "
                     "interval", fontsize=9)
    ax_cmd2 = ax_cmd.twinx()
    for k, seed in enumerate(seeds):
        ep = runs[("neural_modulated", seed)][0]
        ax_cmd2.plot(ep.commands["time_s"], ep.commands["turn"], color="#c0392b",
                     lw=2.0 if seed == 0 else 1.0, alpha=1.0 if seed == 0 else 0.5,
                     ls="--", label=("turn command (right axis), seed %d%s"
                                     % (seed, " (pale = other seeds)"
                                        if k == 0 and len(seeds) > 1 else "")))
    ax_cmd2.set_ylabel("turn command (dimensionless; > 0 = left legs faster)")
    h1, l1 = ax_cmd.get_legend_handles_labels()
    h2, l2 = ax_cmd2.get_legend_handles_labels()
    ax_cmd.legend(h1 + h2, l1 + l2, fontsize=6.0, loc="lower center", framealpha=0.95,
                  ncol=2)
    ax_cmd.grid(alpha=0.3)

    fig.suptitle(
        "FIRST CLOSED SENSORIMOTOR LOOP -- physical fly body (FlyGym/NeuroMechFly v2 on "
        "MuJoCo) + bounded BANC conductance tier, ONE process, two paired arms\n"
        "Model units are MILLIMETRES (gravity -9810 mm/s^2). FlyGym's tripod CPG is an "
        "ENGINEERING BASELINE, NOT a brain model: in BOTH arms it generates the gait, and "
        "the neural tier only modulates (i) intrinsic frequency and (ii) a left/right "
        "asymmetry.\n"
        "The map from descending firing rate to those two knobs is ASSUMED, not measured. "
        "Receptor kinetics are ILLUSTRATIVE. The neural tier is a bounded SELECTION of a "
        "dataset ANNOTATION with UNSIGNED edges (nt_pair = -1 on every row), so synapse "
        "counts are bandwidth, not excitation.\n"
        "Sensation is leg contact force + leg joint angle ONLY: NO VISION is implemented. "
        "Walking performance is NOT evidence about experience, perception or "
        "consciousness.", fontsize=8.4)
    fig.savefig(path, dpi=145)
    plt.close(fig)
    return path


def make_montage(frames, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not frames:
        return None
    pick = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6), layout="constrained")
    for ax, i in zip(axes.ravel(), pick):
        f = frames[i]
        img = f if f.dtype == np.uint8 else (255 * np.clip(f, 0, 1)).astype("uint8")
        ax.imshow(img)
        ax.set_title("frame %d of %d" % (i, len(frames)), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def ground_visibility(frames):
    """Is there ground under the fly?  A crude but real test on the rendered pixels.

    The FlyGym flat-ground world renders a checkered plane; a frame where the lower
    part of the image is a single flat colour is a frame with the ground missing (that
    is exactly the milestone-1 failure: a 20 mm ground plane vanished after 1.4 s).
    Returns the fraction of sampled frames whose bottom band has >1 grey level.
    """
    if not frames:
        return {"n_frames": 0, "fraction_with_textured_bottom_band": None}
    ok = 0
    stats = []
    for f in frames:
        img = np.asarray(f)
        if img.dtype != np.uint8:
            img = (255 * np.clip(img, 0, 1)).astype("uint8")
        band = img[int(img.shape[0] * 0.72):, :, :]
        n_levels = int(np.unique(band.reshape(-1, band.shape[-1]), axis=0).shape[0])
        stats.append(n_levels)
        ok += int(n_levels > 2)
    return {"n_frames": len(frames),
            "fraction_with_textured_bottom_band": ok / len(frames),
            "n_distinct_bottom_colours_min": int(min(stats)),
            "n_distinct_bottom_colours_max": int(max(stats)),
            "note": ("the bottom 28% of each frame must contain more than a flat colour, "
                     "which is how a vanished ground plane shows up")}


class LoadedEpisode:
    """A saved episode read back from its npz, for re-rendering figures without re-running.

    The arrays are the SAVED ones: re-rendering cannot silently change a number.
    """

    def __init__(self, path):
        self.path = path
        with np.load(path, allow_pickle=False) as d:
            self.keys = list(d.keys())
            self.truth = {k.split("/", 1)[1]: np.asarray(d[k]) for k in self.keys
                          if k.startswith("truth/")}
            self.observed = {k.split("/", 1)[1]: np.asarray(d[k]) for k in self.keys
                             if k.startswith("observed/")}
            self.commands = {k.split("/", 1)[1]: np.asarray(d[k]) for k in self.keys
                             if k.startswith("commands/")}
            self.meta = json.loads(str(np.asarray(d["meta/json"]).item()))
        self.metrics = {}
        ti = int(self.meta.get("thorax_index", 33))
        self.yaw_series_rad = np.asarray(
            [heading_yaw_rad(r[ti]) for r in self.truth["body_rotations_wxyz"]])
        t = self.truth["time_s"]
        p = self.truth["thorax_mm"]
        self.metrics["displacement_mm"] = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
        self.metrics["mean_speed_mm_s"] = (self.metrics["displacement_mm"] / (t[-1] - t[0])
                                           if t[-1] > t[0] else float("nan"))
        self.metrics["heading_change_deg"] = float(
            np.degrees(self.yaw_series_rad[-1] - self.yaw_series_rad[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEMO_SEEDS))
    ap.add_argument("--seconds", type=float, default=DEMO_SECONDS)
    ap.add_argument("--gl", default=None)
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--figure-only", action="store_true",
                    help="re-render the figure from the saved episode npz files; no "
                         "physics is run and no number changes")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    os.makedirs(OUT, exist_ok=True)
    t_start = time.perf_counter()
    if args.figure_only:
        cov = json.load(open(os.path.join(OUT, "loop_report.json")))["tier"]
        runs = {}
        for seed in seeds:
            for arm in ("cpg_baseline", "neural_modulated"):
                p = os.path.join(OUT, "loop_episode_%s_seed%d.npz" % (arm, seed))
                ep = LoadedEpisode(p)
                runs[(arm, seed)] = (ep, 0.0, [], None,
                                     ep.meta.get("calibration"))
        fig = make_figure(runs, os.path.join(OUT, "loop_comparison.png"), cov, seeds=seeds)
        print("re-rendered from saved episodes:", fig)
        return 0
    if args.gl is not None:
        gl, probe = args.gl, [{"backend": args.gl, "ok": True, "forced": True}]
    else:
        gl, probe = select_gl_backend()
    print("GL backend: %r" % (gl,), flush=True)

    print("building the bounded BANC tier (deterministic selection)...", flush=True)
    built = build_neural_tier(TierConfig())
    coverage = built["coverage"]
    print("  tier neurons %d, exc edges %d, inh edges %d, readout %d"
          % (coverage["tier"]["neurons"], coverage["edges"]["exc_nnz"],
             coverage["edges"]["inh_nnz"],
             coverage["descending_readout"]
             ["in_tier_with_at_least_one_modelled_incoming_edge"]), flush=True)

    runs = {}
    per_seed = []
    for seed in seeds:
        for arm in ("cpg_baseline", "neural_modulated"):
            render = (arm == "neural_modulated" and seed == RENDER_SEED
                      and not args.skip_video)
            ep, wall, frames, ownership, calib = run_arm(arm, seed, built, gl,
                                                         args.seconds, render=render)
            runs[(arm, seed)] = (ep, wall, frames, ownership, calib)
            npz = ep.save_npz(os.path.join(OUT, "loop_episode_%s_seed%d.npz"
                                           % (arm, seed)))
            print("  seed %d %-17s disp %7.2f mm  speed %6.2f mm/s  yaw %7.2f deg  "
                  "wall %5.1fs  npz %s"
                  % (seed, arm, ep.metrics["displacement_mm"],
                     ep.metrics["mean_speed_mm_s"], ep.metrics["heading_change_deg"],
                     wall, os.path.basename(npz)), flush=True)
            per_seed.append({"seed": seed, "arm": arm, "wall_seconds": wall,
                             "npz": npz, "metrics": ep.metrics,
                             "ownership": ownership,
                             "calibration": calib})

    # ---- paired contribution -------------------------------------------------
    paired = []
    for seed in seeds:
        b = runs[("cpg_baseline", seed)][0].metrics
        n = runs[("neural_modulated", seed)][0].metrics
        paired.append({
            "seed": seed,
            "baseline": {k: b[k] for k in
                         ("displacement_mm", "mean_speed_mm_s", "heading_change_deg",
                          "contact_legs_mean", "passed_engineering_gates")},
            "neural_modulated": {k: n[k] for k in
                                 ("displacement_mm", "mean_speed_mm_s",
                                  "heading_change_deg", "contact_legs_mean",
                                  "passed_engineering_gates")},
            "delta_speed_mm_s": n["mean_speed_mm_s"] - b["mean_speed_mm_s"],
            "delta_displacement_mm": n["displacement_mm"] - b["displacement_mm"],
            "delta_heading_deg": n["heading_change_deg"] - b["heading_change_deg"],
            "speed_ratio": (n["mean_speed_mm_s"] / b["mean_speed_mm_s"]
                            if b["mean_speed_mm_s"] else None),
            "command_speed_scale_mean": n.get("command_speed_scale_mean"),
            "command_turn_mean": n.get("command_turn_mean"),
            "descending_rate_total_mean_hz": n.get("descending_rate_total_mean_hz"),
            "reference_rate_hz": runs[("neural_modulated", seed)][4]
            ["reference_rates_hz"]["total_hz"],
        })

    def stat(key):
        v = [p[key] for p in paired if p[key] is not None]
        return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=0)),
                "min": float(np.min(v)), "max": float(np.max(v)),
                "per_seed": [float(x) for x in v]}

    contribution = {
        "definition": ("paired per-seed difference (neural_modulated minus cpg_baseline) "
                       "with the same seed, the same spawn state and the same CPG factory "
                       "call; the ONLY difference is where the two CPG knobs come from"),
        "delta_speed_mm_s": stat("delta_speed_mm_s"),
        "delta_displacement_mm": stat("delta_displacement_mm"),
        "delta_heading_deg": stat("delta_heading_deg"),
        "speed_ratio": stat("speed_ratio"),
        "command_speed_scale_mean": stat("command_speed_scale_mean"),
        "command_turn_mean": stat("command_turn_mean"),
        "descending_rate_total_mean_hz": stat("descending_rate_total_mean_hz"),
        "reference_rate_hz": stat("reference_rate_hz"),
    }

    # ---- figure -------------------------------------------------------------
    fig_path = os.path.join(OUT, "loop_comparison.png")
    make_figure(runs, fig_path, coverage, seeds=seeds)
    print("figure:", fig_path, flush=True)

    # ---- video --------------------------------------------------------------
    video = None
    montage = None
    ground = {"n_frames": 0}
    frames = runs[("neural_modulated", RENDER_SEED)][2] if (("neural_modulated",
                                                             RENDER_SEED) in runs) else []
    if frames:
        ground = ground_visibility(frames)
        montage = make_montage(
            frames, os.path.join(OUT, "loop_frame_montage.png"),
            "NEURAL-MODULATED WALK, sampled rendered frames (seed %d). The ground must be "
            "visible in every frame; a vanished ground plane was a real milestone-1 defect."
            % RENDER_SEED)
        try:
            import mediapy
            imgs = [f if np.asarray(f).dtype == np.uint8
                    else (255 * np.clip(f, 0, 1)).astype("uint8") for f in frames]
            video = os.path.join(OUT, "loop_neural_walk_seed%d.mp4" % RENDER_SEED)
            mediapy.write_video(video, imgs, fps=25)
        except Exception as exc:                                    # noqa: BLE001
            video = "unavailable: %s: %s" % (type(exc).__name__, exc)

    # ---- report -------------------------------------------------------------
    tests_path = os.path.join(OUT, "loop_selftest.json")
    tests = None
    if os.path.isfile(tests_path):
        with open(tests_path) as fh:
            tests = json.load(fh)

    report = {
        "round": "embodied loop round 1: the first closed sensorimotor loop",
        "status": ("one process holds the physical body (FlyGym/MuJoCo) and the bounded "
                   "BANC conductance tier; two paired arms measured"),
        "question": ("what does the connectome-derived neural tier contribute when it is "
                     "wired into a physical body's sensorimotor loop, compared with the "
                     "pure controller baseline?"),
        "answer_summary": {
            "measured": ("see contribution/ below; the neural arm's command is a real "
                         "function of the body's own contact state (the tier is SILENT "
                         "with no sensory drive: measured 0.0 Hz)"),
            "not_measured": ("nothing here shows that the connectome generates the gait, "
                             "that the rate->knob map is correct, or that the fly "
                             "perceives anything"),
        },
        "honesty": HONESTY,
        "fingerprint_note": FINGERPRINT_NOTE,
        "units": dict(UNITS),
        "arms": {"cpg_baseline": ("FlyGym demo tripod CPG, unchanged; ENGINEERING "
                                  "BASELINE, not a brain model"),
                 "neural_modulated": ("the same CPG generates the gait; only the "
                                      "intrinsic frequency and a left/right asymmetry "
                                      "come from the tier's descending firing")},
        "rationale": ("a descending command setting speed and turn while a local pattern "
                      "generator makes the rhythm is the deliberately conservative first "
                      "step. The connectome does NOT generate the gait and no result may "
                      "be described that way."),
        "clocks": {"body_physics_s": 1e-4, "neural_s": 5e-4, "command_decode_s": 5e-3,
                   "causal_order": ("read body state -> transduce -> advance neural tier "
                                    "(spikes at interval END) -> decode command from the "
                                    "spikes of the PREVIOUS completed interval -> hold "
                                    "while the body substeps")},
        "config": {"seeds": list(seeds), "seconds": args.seconds,
                   "gl_backend": gl, "gl_probe": probe,
                   "demand": "seeds and duration were declared before the run"},
        "tier": coverage,
        "models": {"body": "FlyGym 2.1.0 / NeuroMechFly v2 on MuJoCo 3.9.0 "
                           "(Apache-2.0, EPFL Ramdya lab)",
                   "neural": "engine.neural_cond conductance point tier, backward Euler, "
                             "end-of-interval spikes, explicit delay ring",
                   "data": "cached BANC connectome snapshot (SELECTION, UNSIGNED edges)"},
        "per_seed_arm": per_seed,
        "per_seed_paired": paired,
        "contribution": contribution,
        "figures": {"figure": fig_path, "montage": montage, "video": video},
        "ground_visibility": ground,
        "wall_seconds_total": time.perf_counter() - t_start,
        "peak_ram_mb": peak_ram_mb(),
        "selftest": ({"n_checks": tests.get("n_checks"), "n_passed": tests.get("n_passed"),
                      "passed": tests.get("passed")} if tests else
                     "not run yet: run run_embodied_loop_selftest.py"),
        "limitations_left_in_place": [
            "the CPG still generates the gait in the neural arm; the tier has no path to "
            "the muscles other than two assumed knobs",
            "the rate->(frequency, turn) map, its two gains and its rate scale are ASSUMED",
            "receptor kinetics and the force/angle reference scales are ILLUSTRATIVE; the "
            "contact-force channel's ABSOLUTE calibration is unresolved",
            "the tier is a bounded SELECTION (see tier/coverage) of an ANNOTATION, with "
            "UNSIGNED edges; 177 of the 196 in-tier descending neurons receive no modelled "
            "in-tier edge at all under the sign rule and are excluded from the readout",
            "the left/right split of the descending readout is a PROXY (the dataset's "
            "descending annotation carries no side), and the left group has only 7 cells, "
            "so its rate quantum at a 5 ms interval is 28.6 Hz per spike",
            "no vision, no olfaction, no head sensation: only leg contact force and leg "
            "joint angle are transduced",
            "one body, one scene, flat ground, and a single fixed spawn state",
        ],
    }
    rep_path = os.path.join(OUT, "loop_report.json")
    with open(rep_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("\nreport:", rep_path)
    print(json.dumps({
        "n_seeds": len(seeds),
        "delta_speed_mm_s": contribution["delta_speed_mm_s"],
        "speed_ratio": contribution["speed_ratio"],
        "delta_heading_deg": contribution["delta_heading_deg"],
        "wall_seconds_total": report["wall_seconds_total"],
        "peak_ram_mb": report["peak_ram_mb"],
        "ground": ground,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
