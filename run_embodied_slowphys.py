"""Demonstration of the SLOW PHYSIOLOGY state and the behaviour/rest recorder.

Run in the BODY environment:
    /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
        /run/media/sensen/Data2/cell_wound_prototype/run_embodied_slowphys.py

WHAT IS RUN, AND ON WHICH CLOCK
-------------------------------
A physical adult fly (FlyGym 2.1.0 / NeuroMechFly v2 on MuJoCo 3.9.0, model units
MILLIMETRES) walks on flat ground.  FlyGym's demo tripod CPG generates the gait
(ENGINEERING BASELINE, not the connectome).  On top of that:

  * a deterministic PHASE SCRIPT writes the loop's own two CPG knobs (speed_scale,
    turn) to produce still / straight / turning / zig-zag / still epochs, so that the
    behaviour classifier has something to classify and the slow physiology sees a
    range of measured activity.  The script is an INPUT PLAN, not a behaviour model;
    every write is logged with its interval.
  * EVERY command interval, MEASURED body quantities are reduced to a dimensionless
    locomotor load (thorax speed in mm/s from the thorax's own displacement, and the
    time-mean of sum_j |qdot_j| in rad/s), which drives the oxygen demand in nmol/s.
  * the SLOW PHYSIOLOGY module advances on the SAME 5 ms clock: tracheal air-side O2
    <- air bath through a spiracle conductance -> tissue through a tracheole
    conductance -> consumed; plus a SEPARATE haemolymph compartment whose own ledger
    tracks trehalose.  Nothing in this run is blood-borne: the fly's gas exchange is
    TRACHEAL.
  * every interval is labelled locomotion / turning / exploration-like / quiet rest /
    not-supported by thresholds that are DECLARED IN THE MODULE BEFORE THE RUN.

OUTPUTS (all under outputs/embodied_body/, prefixed slowphys_)
--------------------------------------------------------------
  slowphys_report.json    every number, every ledger, every threshold, wall clock,
                          peak RAM, the behaviour-epoch table, and the honesty block
  slowphys_traces.npz     the slow-physiology ledger rows, the measured activity, the
                          per-interval behaviour measures and labels, the truth series
  slowphys_demo.png       substance ledgers over time, locomotor load vs consumption,
                          the behaviour/rest timeline, and the time-jump flag
  slowphys_episode.npz    the underlying scheduler episode (truth/commands/events/meta)

The figure is written and then OPENED AND LOOKED AT (the harness reads it back with
an image tool); a figure nobody looked at is not evidence.
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

from engine.embodied import select_gl_backend                                # noqa: E402
from engine.embodied.loop import Episode, LoopConfig, MultirateScheduler      # noqa: E402
from engine.embodied.slow_physiology import (                                 # noqa: E402
    BEHAVIOUR_LABELS, BehaviourConfig, ENERGY_PROXY_DISCLAIMER,
    ModulatorConfig, SLOWPHYS_HONESTY, SlowPhysiologyConfig, SlowPhysiologyDriver,
    TimeJumpRecord, slowphys_report,
)

#: PRE-REGISTERED run parameters.  Declared here, before any measurement, and not
#: changed afterwards.  24 s at a 5 ms command interval = 4800 intervals, which is
#: long enough for several contiguous behaviour epochs of every kind.
DEMO_SECONDS = 24.0
DEMO_SEED = 0
PUBLISHED = {
    "n_intervals": int(round(DEMO_SECONDS / 5e-3)),
    "behaviour_window_s": 0.5,
    "min_epoch_s": 0.5,
}


def peak_ram_mb():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def build_schedule(dt_command_s=5e-3):
    """The pre-registered phase plan, in command intervals.

    still -> straight locomotion -> turning -> zig-zag exploration -> still.

    The plan is stated as a table of intervals so that every write is reproducible and
    the report can show it verbatim.  Nothing here reads the body: it is an input.
    """
    def span(t_from, t_to, phase, speed, turn):
        return {"phase": phase, "interval_from": int(round(t_from / dt_command_s)),
                "interval_to": int(round(t_to / dt_command_s)) - 1,
                "speed_scale": float(speed), "turn": float(turn)}

    plan = [
        span(0.0, 4.0, "still_1", 0.0, 0.0),
        span(4.0, 9.0, "straight_locomotion", 1.6, 0.0),
        span(9.0, 13.0, "turning", 1.2, -0.9),
        span(13.0, 19.0, "zigzag_exploration", 1.2, 0.0),
        span(19.0, 24.0, "still_2", 0.0, 0.0),
    ]
    return plan


def zigzag_schedule(plan, dt_command_s=5e-3, period_s=1.0):
    """Give the exploration phase an alternating turn, so heading VARIANCE is large.

    The classifier's exploration criterion is heading SD >= threshold with real
    displacement; a straight phase cannot satisfy it and a monotone turn is classified
    as turning.  The zig-zag is therefore planned explicitly, and whether it actually
    produces the intended label is reported rather than assumed.
    """
    out = []
    for row in plan:
        if row["phase"] != "zigzag_exploration":
            out.append(row)
            continue
        lo, hi = row["interval_from"], row["interval_to"]
        period = max(1, int(round(period_s / dt_command_s)))
        for k in range(lo, hi + 1):
            sign = 1.0 if ((k - lo) // period) % 2 == 0 else -1.0
            out.append({"phase": "zigzag_exploration", "interval_from": k,
                        "interval_to": k, "speed_scale": row["speed_scale"],
                        "turn": 0.9 * sign})
    return out


def make_figure(report, traces, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    th = report["behaviour"]["thresholds"]
    t_all = np.asarray(traces["ledger/time_s"], float)
    # A DECLARED TIME JUMP appends ledger rows that are QUASI-STEADY, not real-clock
    # intervals: they carry no measured activity, so panels (c) and (d) plot only the
    # real-clock rows.  Splitting them here is what keeps the activity series and the
    # ledger trace from disagreeing about how long the run was -- the first version of
    # this figure died with "x and y must have same first dimension (5400 vs 4800)".
    n_real = int(np.asarray(traces["activity/load"]).size)
    t = t_all[:n_real] if n_real else t_all
    t_jump = t_all[n_real:] if n_real else np.asarray([], float)
    fig = plt.figure(figsize=(15.5, 13.0), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15])
    ax_o2 = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_ld = fig.add_subplot(gs[1, 0])
    ax_dm = fig.add_subplot(gs[1, 1])
    ax_be = fig.add_subplot(gs[2, :])

    # ---- (a) substance ledgers -------------------------------------------
    for series, kw in (("ledger/o2_tracheal_nmol",
                        dict(lw=2.0, color="#1f4e9c",
                             label="tracheal air side (O2), nmol")),
                       ("ledger/o2_tissue_nmol",
                        dict(lw=2.0, color="#c0392b", label="tissue (O2), nmol")),
                       ("ledger/o2_total_nmol",
                        dict(lw=1.0, color="#555555", ls="--",
                             label="total O2 in the fly, nmol"))):
        y = np.asarray(traces[series], float)
        ax_o2.plot(t, y[:t.size], **kw)
    if t_jump.size:
        # THE DECLARED TIME JUMP IS 600 s, i.e. 25x the real-clock run: plotting it on the
        # same axis (the first version did) squashed every real-clock feature into the left
        # 4% of the panel and made the honest part of the figure unreadable.  The main panel
        # therefore shows the REAL CLOCK ONLY and the jump is stated as numbers, with a
        # small all-time inset so the reader can see the whole record at a glance.
        qa = report["time"].get("quasi_steady_advance", {})
        sb = qa.get("state_before", {})
        sa = qa.get("state_after", {})
        ax_o2.text(0.02, 0.34,
                   "declared TIME JUMP %.0f s (quasi-steady, body NOT simulated)\n"
                   "tracheal %+.3f nmol, tissue %+.3f nmol\nledger residual %.1e nmol\n"
                   "main panels: REAL CLOCK (%.0f s) only"
                   % (qa.get("duration_s", float("nan")),
                      sa.get("tracheal_o2_nmol", np.nan)
                      - sb.get("tracheal_o2_nmol", np.nan),
                      sa.get("tissue_o2_nmol", np.nan)
                      - sb.get("tissue_o2_nmol", np.nan),
                      qa.get("ledger_residual_nmol", float("nan")), t[-1]),
                   transform=ax_o2.transAxes, fontsize=7.5, va="bottom",
                   bbox=dict(fc="#fdecea", ec="#c0392b", lw=0.8))
        # the inset uses a SYMMETRIC LOG axis: the tissue compartment moves in the 7-9 nmol
        # band while the air side moves by ~1 nmol on a 90 nmol level, so on a linear axis
        # either the tissue is invisible or the air side is a flat line (the first attempt
        # showed both as flat lines, which is why it is plotted this way).
        inset = ax_o2.inset_axes([0.30, 0.05, 0.67, 0.26])
        ins = [("ledger/o2_tracheal_nmol", "#1f4e9c", "air side"),
               ("ledger/o2_tissue_nmol", "#c0392b", "tissue")]
        for key, col, name in ins:
            y = np.asarray(traces[key], float)
            inset.plot(t, y[:t.size], lw=1.2, color=col, label=name)
            if y.size > t.size:
                inset.plot(np.concatenate([[t[-1]], t_jump]), y[t.size - 1:], lw=1.2,
                           color=col, ls=(0, (3, 2)))
        inset.set_yscale("symlog", linthresh=1.0)
        inset.set_ylim(-0.2, 200.0)
        inset.set_title("whole record incl. the %.0f s jump (symlog); dashed = the jump"
                        % qa.get("duration_s", 0.0), fontsize=7)
        inset.tick_params(labelsize=6)
        inset.set_ylabel("O2 (nmol)", fontsize=6)
        inset.legend(fontsize=6, loc="upper right", ncol=2)
    ax_o2b = ax_o2.twinx()
    yf = np.asarray(traces["ledger/tissue_o2_fraction"], float)
    ax_o2b.plot(t, yf[:t.size], lw=1.2, color="#117a65", ls=":",
                label="tissue O2 as a fraction of the air side")
    ax_o2b.set_ylabel("tissue O2 fraction of the air side (dimensionless)", color="#117a65")
    ax_o2b.set_ylim(0.0, 1.05)
    ax_o2b.tick_params(axis="y", labelcolor="#117a65")
    h1, l1 = ax_o2.get_legend_handles_labels()
    h2, l2 = ax_o2b.get_legend_handles_labels()
    ax_o2.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right")
    ax_o2.set_xlabel("time (s)")
    ax_o2.set_ylabel("O2 amount (nmol)")
    ax_o2.set_title("(a) TRACHEAL O2 supply -> tissue (real clock, %.0f s).  No\n"
                    "blood-borne O2 is modelled; ledger residual %.2e nmol"
                    % (t[-1], report["ledgers"]["O2"]["residual"]), fontsize=10)

    # ---- (b) haemolymph's OWN substance ledger ----------------------------
    ytr = np.asarray(traces["ledger/trehalose_total_nmol"], float)
    ax_tr.plot(t, ytr[:t.size], lw=2.2, color="#7d3c98",
               label="haemolymph trehalose, nmol (real clock)")
    ax_tr.set_xlim(t[0], t[-1])
    ax_tr.set_ylim(min(58.0, float(ytr.min()) - 1.0), float(ytr.max()) + 3.0)
    ax_tr.set_xlabel("time (s)")
    ax_tr.set_ylabel("amount (nmol)")
    ax_tr.set_title("(b) HAEMOLYMPH: a separate compartment with its own ledger\n"
                    "(trehalose only; it carries NO O2; residual %.1e nmol)"
                    % report["ledgers"]["trehalose"]["residual"], fontsize=10)
    ax_tr.text(0.02, 0.06,
               "the fly's gas exchange is tracheal.\nno respiratory pigment is modelled "
               "anywhere.",
               transform=ax_tr.transAxes, fontsize=8.5, va="bottom",
               bbox=dict(fc="#fdf3e3", ec="#b9770e", lw=0.8))
    ax_tr.legend(fontsize=8, loc="upper right")

    # ---- (c) locomotor load vs consumption -------------------------------
    ax_ld.plot(t, np.asarray(traces["activity/load"])[:t.size], lw=1.4,
               color="#1f4e9c", label="measured load (dimensionless)")
    ax_ld.set_xlabel("time (s)")
    ax_ld.set_ylabel("measured load", color="#1f4e9c")
    ax_ld.tick_params(axis="y", labelcolor="#1f4e9c")
    ax_ldb = ax_ld.twinx()
    ax_ldb.plot(t, np.asarray(traces["activity/demand_nmol_s"])[:t.size], lw=1.6,
                color="#c0392b", label="O2 consumption (nmol/s)")
    ax_ldb.set_ylabel("O2 consumption (nmol/s)", color="#c0392b")
    ax_ldb.tick_params(axis="y", labelcolor="#c0392b")
    r = report["demand_vs_activity_correlation"]["pearson_r"]
    h1, l1 = ax_ld.get_legend_handles_labels()
    h2, l2 = ax_ldb.get_legend_handles_labels()
    ax_ld.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower center")
    ax_ld.set_title("(c) consumption is DRIVEN BY a measured quantity\n"
                    "Pearson r(speed, demand) = %.4f" % r, fontsize=10)

    # ---- (d) the two measured inputs -------------------------------------
    ax_dm.plot(t, np.asarray(traces["activity/speed_xy_mm_s"])[:t.size], lw=1.2,
               color="#1e8449", label="thorax speed (mm/s)")
    ax_dm.plot(t, np.asarray(traces["activity/joint_speed_rad_s"])[:t.size], lw=1.2,
               color="#8e44ad", label="mean sum_j |qdot_j| (rad/s)")
    ax_dm.axhline(th["rest_speed_max_mm_s"], color="#1e8449", ls=":", lw=1.0)
    ax_dm.axhline(th["locomotion_speed_min_mm_s"], color="#1e8449", ls="-.", lw=1.0)
    ax_dm.set_xlabel("time (s)")
    ax_dm.set_ylabel("measured body quantity (mm/s or rad/s)")
    ax_dm.legend(fontsize=8, loc="upper left")
    ax_dmb = ax_dm.twinx()
    ax_dmb.plot(t, np.asarray(traces["behaviour/long_displacement_mm"])[:t.size], lw=1.4,
                color="#b9770e",
                label="net displacement over the %.1f s long window (mm)"
                % th["long_window_s"])
    ax_dmb.axhline(th["turn_loop_displacement_max_mm"], color="#b9770e", ls=":", lw=1.0)
    ax_dmb.axhline(th["exploration_long_displacement_min_mm"], color="#b9770e", ls="--",
                   lw=1.0)
    ax_dmb.set_ylabel("long-window net displacement (mm)", color="#b9770e")
    ax_dmb.tick_params(axis="y", labelcolor="#b9770e")
    h1, l1 = ax_dm.get_legend_handles_labels()
    h2, l2 = ax_dmb.get_legend_handles_labels()
    ax_dm.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower left")
    ax_dm.set_title("(d) the MEASURED inputs.  Green: thorax speed (dotted/dash-dot = "
                    "rest and locomotion thresholds).\nPurple: sum_j |qdot_j|.  Orange: "
                    "long-window net displacement (decides TURNING vs EXPLORATION)",
                    fontsize=10)

    # ---- (e) behaviour / rest timeline -----------------------------------
    traces_t = np.asarray(traces["truth/time_s"], float)
    labels = traces["behaviour/label_index"].astype(int)
    yy = np.zeros_like(traces_t)
    for i, lab in enumerate(BEHAVIOUR_LABELS):
        yy[labels == i] = i
    for i, lab in enumerate(BEHAVIOUR_LABELS):
        ax_be.step(traces_t, np.where(labels == i, i, np.nan), where="post", lw=3.0,
                   alpha=0.85)
    ax_be.set_yticks(range(len(BEHAVIOUR_LABELS)))
    ax_be.set_yticklabels(BEHAVIOUR_LABELS)
    ax_be.set_xlabel("time (s)")
    ax_be.set_title("(e) BEHAVIOUR / REST timeline: labels from PRE-REGISTERED "
                    "thresholds on MEASURED body quantities, on two time scales "
                    "(%.2f s window + %.1f s long window)\n"
                    "QUIET REST IS NOT SLEEP: no sleep criterion is implemented and no "
                    "arousal test stimulus exists in this effort"
                    % (th["window_s"], th["long_window_s"]), fontsize=10)
    counts = report["behaviour"]["label_counts"]
    dt_int = float(np.median(np.diff(traces_t)))
    ax_be.text(0.005, 0.97, "  ".join("%s=%d intervals (%.1f s)"
                                      % (k, v, v * dt_int)
                                      for k, v in counts.items()),
               transform=ax_be.transAxes, fontsize=8.5, va="top",
               bbox=dict(fc="white", ec="#888888", lw=0.6))
    # time-jump flag
    n_jumps = report["time"]["n_time_jumps"]
    jump_txt = ("TIME-JUMP FLAG: %d quasi-steady advance(s) declared"
                % n_jumps) if n_jumps else \
        ("TIME-JUMP FLAG: none used -- the slow module advanced on the SAME real clock "
         "as the body")
    ax_be.text(0.005, 0.06, jump_txt, transform=ax_be.transAxes, fontsize=9,
               va="bottom",
               bbox=dict(fc="#eafaf1" if not n_jumps else "#fdecea",
                         ec="#1e8449" if not n_jumps else "#c0392b", lw=1.0))

    fig.suptitle(
        "slow physiology + behaviour/rest recorder -- EVERY kinetic parameter is "
        "ILLUSTRATIVE; the only tracheal geometry on record in this project is LOCUST, "
        "not Drosophila\n"
        "and is NOT used for any default; this project has no measured adult-fly "
        "metabolic rate; quiet rest is NOT sleep; no consciousness/viability/immortality "
        "claim is made",
        fontsize=9.5)
    fig.text(0.5, 0.005,
             "energy proxy = %s: %s" % (report["energy_proxy"]["name"],
                                        ENERGY_PROXY_DISCLAIMER[:150] + " ..."),
             ha="center", fontsize=7.5, color="#7d3c98")
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=DEMO_SECONDS)
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--gl", default=None)
    ap.add_argument("--time-jump-seconds", type=float, default=0.0,
                    help="if positive, ALSO demonstrate a declared TIME JUMP by "
                         "advancing the slow module alone for this many seconds after "
                         "the real-clock run (quasi-steady, body NOT simulated)")
    ap.add_argument("--figure-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    t_start = time.perf_counter()
    if args.figure_only:
        with open(os.path.join(OUT, "slowphys_report.json")) as fh:
            rep = json.load(fh)
        traces = dict(np.load(os.path.join(OUT, "slowphys_traces.npz")))
        fig = make_figure(rep, traces, os.path.join(OUT, "slowphys_demo.png"))
        print("re-rendered figure from saved artefacts: %s" % fig)
        return 0

    gl = args.gl
    probe = [{"backend": args.gl, "ok": True, "forced": True}] if gl else None
    if gl is None:
        gl, probe = select_gl_backend()
    print("GL backend: %r" % (gl,), flush=True)

    bcfg = BehaviourConfig()
    scfg = SlowPhysiologyConfig()
    mcfg = ModulatorConfig()          # DISABLED by default: unknown targets OFF
    print("behaviour thresholds (PRE-REGISTERED, declared before the run): %s"
          % json.dumps(bcfg.__dict__, sort_keys=True), flush=True)

    cfg = LoopConfig(arm="cpg_baseline", seed=args.seed, duration_s=args.seconds,
                     gl_backend=gl)
    sch = MultirateScheduler(cfg)
    ep = Episode(cfg.arm, cfg.seed, cfg)
    ep.meta["arm_description"] = (
        "FlyGym tripod CPG, unchanged: ENGINEERING BASELINE, not the connectome. The "
        "slow-physiology driver writes only the loop's two existing CPG knobs, from a "
        "deterministic PRE-REGISTERED phase script.")
    driver = SlowPhysiologyDriver(sch, cfg=scfg, modulator_cfg=mcfg)
    ep.meta["phase_script"] = driver.schedule_note
    from engine.embodied.slow_physiology import slowphys_fingerprint, save_traces
    ep.meta["behaviour_thresholds"] = dict(bcfg.__dict__)
    ep.meta["behaviour_thresholds_note"] = (
        "declared in engine/embodied/slow_physiology.py BEFORE the run and never tuned; "
        "they are ILLUSTRATIVE engineering thresholds for this model, not fly norms")
    ep.meta["slowphys_fingerprint"] = slowphys_fingerprint(scfg)
    plan = zigzag_schedule(build_schedule(cfg.dt_command_s), cfg.dt_command_s)
    ep.add_event("episode_start", 0.0, duration_s=cfg.duration_s,
                 n_intervals=cfg.n_intervals,
                 plan_rows=len(plan))
    for row in [plan[i] for i in (0, len(plan) // 4, len(plan) // 2,
                                  3 * len(plan) // 4, len(plan) - 1)]:
        ep.add_event("phase_script_row", row["interval_from"] * cfg.dt_command_s,
                     phase=row["phase"], speed_scale=row["speed_scale"],
                     turn=row["turn"])
    print("running %d intervals (%.1f s) on the real clock..." % (cfg.n_intervals,
                                                                  args.seconds),
          flush=True)
    sch, ep, driver = driver.run(ep, schedule=plan)

    # ------------------------------------------------------------------ report
    wall = time.perf_counter() - t_start
    report = slowphys_report(driver, ep, bcfg,
                             ledger_tolerance_nmol=1e-9,
                             wall_seconds=wall, peak_ram_mb=peak_ram_mb(),
                             gl_backend=gl,
                             extra={"probe": probe})
    # a declared TIME JUMP, if asked for: the slow module ALONE, quasi-steady.
    # The body and the nervous system are NOT simulated during it, so it is labelled and
    # reported as a quasi-steady advance and never counted as continuous simulated time.
    if args.time_jump_seconds > 0:
        tj = args.time_jump_seconds
        S = driver.phys
        before = (S.tracheal_o2_nmol, S.tissue_o2_nmol, S.time_s)
        held = driver.phys.activity_log[-1]
        n_rows_before = len(S.rows)
        # ONE long step with the jump declared: the module sub-steps internally at
        # ``max_substep_s`` (1 s by default), so the stride never spans more than about one
        # time constant of the fast compartment.  Issuing many sub-bound steps instead
        # would work too but is reported as "reached by accumulating sub-bound steps",
        # which is a weaker statement about what happened.
        S.step(tj, demand_nmol_s=held.demand_nmol_s, declared_time_jump=True,
               record_activity=False)
        # NO activity record is appended for the jump: the activity series holds ONLY
        # real-clock intervals, so it stays exactly as long as the classification and the
        # truth series.  The jump lives in the ledger trace and in the report's explicit
        # time_jumps list, which is where a quasi-steady advance belongs.
        report["time"]["time_jumps"] = [t.as_dict() for t in S.time_jumps]
        report["time"]["n_time_jumps"] = len(S.time_jumps)
        report["time"]["time_jump_flag"] = bool(S.time_jumps)
        print("TIME JUMP records: %d of duration(s) %s"
              % (len(S.time_jumps), [t.duration_s for t in S.time_jumps][:3]), flush=True)
        report["time"]["quasi_steady_advance"] = {
            "duration_s": tj,
            "label": "TIME_JUMP_QUASI_STEADY",
            "body_simulated_during_jump": False,
            "nervous_system_simulated_during_jump": False,
            "rows_added_to_the_ledger_trace": int(len(S.rows) - n_rows_before),
            "substepped_internally": bool(S._jump_was_substepped),
            "max_substep_s": float(S.cfg.max_substep_s),
            "ledger_residual_nmol": float(S.o2_ledger.residual()),
            "ledger_closed_within_tolerance": bool(
                S.check_ledgers()["O2"]["closed_within_tolerance"]),
            "load_held_at": float(held.load),
            "state_before": {"tracheal_o2_nmol": before[0], "tissue_o2_nmol": before[1]},
            "state_after": {"tracheal_o2_nmol": S.tracheal_o2_nmol,
                            "tissue_o2_nmol": S.tissue_o2_nmol},
            "note": ("the slow physiology module alone was advanced for this interval "
                     "with the measured load and the air-bath concentration held at "
                     "their last values; the body and the nervous system were NOT "
                     "simulated, so this is a quasi-steady approximation and NOT "
                     "continuous simulated time"),
        }
        ep.add_event("time_jump_declared", S.time_s - tj, duration_s=tj,
                     label="TIME_JUMP_QUASI_STEADY")
        print("declared a %.1f s TIME JUMP (quasi-steady) for the slow module alone"
              % tj)

    # the behaviour label series must be the same one the figure draws
    led = report["ledgers"]
    report["figure"] = os.path.join(OUT, "slowphys_demo.png")
    report["honesty_restated_on_figure"] = True
    report["what_the_ledger_says"] = {
        "o2_rule": led["O2"]["rule"],
        "o2_amount_now_nmol": led["O2"]["amount_now"],
        "o2_initial_nmol": led["O2"]["initial"],
        "o2_inflow_nmol": led["O2"]["inflow"],
        "o2_outflow_nmol": led["O2"]["outflow"],
        "o2_consumption_nmol": led["O2"]["consumption"],
        "o2_residual_nmol": led["O2"]["residual"],
        "o2_residual_relative": led["O2"]["residual_relative"],
        "o2_closed_within_1e-9_nmol": led["O2"]["closed_within_tolerance"],
        "o2_compartment_residuals": led["O2"]["compartment_residuals"],
        "trehalose_amount_now_nmol": led["trehalose"]["amount_now"],
        "trehalose_residual_nmol": led["trehalose"]["residual"],
        "trehalose_closed_within_1e-9_nmol": led["trehalose"]["closed_within_tolerance"],
        "no_silent_clamping": ("every transport is an exact solution or an exact "
                               "integral; the only max()/clip() calls are the gross "
                               "inflow/outflow split of a signed flux and the "
                               "nonnegativity guard, which RAISES instead of fixing"),
        "trehalose_note": ("trehalose source and clearance are switched OFF in the "
                           "default config, so its ledger is flat; that is reported, "
                           "not dressed up as a measured turnover"),
        "energy_proxy_in_any_ledger": led["energy_proxy_in_any_ledger"],
    }
    report["episode"] = {"path": os.path.join(OUT, "slowphys_episode.npz"),
                         "n_events": len(ep.events),
                         "metrics": _jsonable(ep.metrics)}
    report["selftest"] = {"json": os.path.join(OUT, "slowphys_selftest.json"),
                          "note": "filled in by run_embodied_slowphys_selftest.py"}

    ep_path = ep.save_npz(os.path.join(OUT, "slowphys_episode.npz"))
    with open(os.path.join(OUT, "slowphys_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    # traces for the figure and for the npz: ONE writer (the module's save_traces)
    save_traces(os.path.join(OUT, "slowphys_traces.npz"), driver, ep, report)
    traces = dict(np.load(os.path.join(OUT, "slowphys_traces.npz")))

    make_figure(report, traces, os.path.join(OUT, "slowphys_demo.png"))

    # ------------------------------------------------------------------ console
    print("\nLEDGERS (rule: %s)" % led["O2"]["rule"])
    for name in ("O2", "trehalose"):
        L = led[name]
        print("  %-10s initial %12.6f %-5s inflow %10.6f outflow %10.6f "
              "consumption %10.6f -> now %11.6f  residual %9.2e (closed=%s)"
              % (name, L["initial"], L["unit"], L["inflow"], L["outflow"],
                 L["consumption"], L["amount_now"], L["residual"],
                 L["closed_within_tolerance"]))
    sp = report["supply_consumption"]
    print("  MEASURED supply/consumption: mean net bath supply %.4f nmol/s vs mean "
          "consumption %.4f nmol/s"
          % (sp["mean_net_supply_nmol_s"], sp["mean_consumption_nmol_s"]))
    print("  tracheal O2 %.3f -> %.3f nmol; tissue O2 %.3f -> %.3f nmol "
          "(tissue fraction of air side %.4f)"
          % (sp["tracheal_o2_start_nmol"], sp["tracheal_o2_end_nmol"],
             sp["tissue_o2_start_nmol"], sp["tissue_o2_end_nmol"],
             sp["tissue_o2_fraction_end"]))
    ml = report["measured_load"]
    print("  measured load %.4f..%.4f -> demand %.4f..%.4f nmol/s "
          "(Pearson r(speed,demand)=%.4f)"
          % (ml["load_range"][0], ml["load_range"][1],
             ml["demand_nmol_s_range"][0], ml["demand_nmol_s_range"][1],
             report["demand_vs_activity_correlation"]["pearson_r"]))
    print("\nBEHAVIOUR EPOCHS (pre-registered thresholds, window %.2f s)"
          % bcfg.window_s)
    for e in report["behaviour"]["epochs"]:
        print("  %-14s %6.2f - %6.2f s  (%5.2f s, %3d intervals)  peak speed %6.3f mm/s"
              % (e["label"], e["start_time_s"], e["end_time_s"], e["duration_s"],
                 e["n_intervals"], e["speed_mm_s_max"]))
    cov = report["behaviour"]["coverage"]
    print("  exploration statistics (measured, from the path): %d distinct %g mm cells, "
          "convex-hull area %.1f mm^2, path %.1f mm, net %.1f mm"
          % (cov["cells_visited"], cov["cell_size_mm"], cov["convex_hull_area_mm2"],
             cov["path_length_mm"], cov["net_displacement_mm"]))
    print("\nenergy proxy %s: in any ledger = %s"
          % (report["energy_proxy"]["name"], report["energy_proxy"]["in_any_ledger"]))
    print("modulator layer: enabled=%s known_target=%s target_effect=%s"
          % (report["modulator_layer"]["enabled"],
             report["modulator_layer"]["known_target"],
             report["modulator_layer"]["target_effect_enabled"]))
    print("sleep: implemented=%s (%s)" % (report["sleep"]["implemented"],
                                          report["sleep"]["arousal_test_stimulus"]))
    print("time jumps: %d" % report["time"]["n_time_jumps"])
    print("wall %.1f s, peak RAM %.0f MB" % (wall, peak_ram_mb()))
    print("wrote:")
    for nm in ("slowphys_report.json", "slowphys_traces.npz", "slowphys_demo.png",
               "slowphys_episode.npz"):
        print("  %s" % os.path.join(OUT, nm))
    return 0


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


if __name__ == "__main__":
    sys.exit(main())
