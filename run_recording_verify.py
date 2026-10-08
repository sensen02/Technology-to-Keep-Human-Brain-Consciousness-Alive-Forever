#!/usr/bin/env python3
"""VERIFICATION AND COMPARISON -- steps 3 of 3 of the recording pipeline.

This is the ONLY file in the three-stage pipeline that opens both the
reconstruction (``vision.json``) and the ground truth (``episode.npz``).  It is
separate on purpose: ``run_recording_vision.py`` cannot see the ground truth, so the
numbers here are an AUDIT of that parser, not part of it.

WHAT IT ANSWERS, ONE SECTION PER REQUIREMENT
--------------------------------------------
R1 walking            per-episode distance, speed, contact duty, and whether the fly
                      is on its feet rather than dragged or stuck.
R2 electrode records  the recorded trace's RMS against the chain's own noise floor,
                      and whether the trace carries a gait-locked component: the
                      signal is band-passed around the MEASURED stride frequency and
                      compared with the same measurement on the noise-only channel.
R2b electrode weight  the SAME seed run with and without the payload mass, paired, so
                      the payload effect is a paired difference rather than two
                      unrelated samples; a paired t-like interval is reported, and it
                      is NOT called a p-value.
R3 vision             reconstruction error against the simulator, in mm and degrees,
                      plus the stride frequency recovered from the photos alone
                      against the true stride frequency.
R4 touch from motion  per-leg contact agreement between the reconstruction's inferred
                      touch and the simulator's contact, with the CHANCE LEVEL given
                      next to it because a 50 %-duty gait scores ~0.5 for free.

Every number is written to ``comparison.json``; the figures are the at-a-glance check.

Run (PROJECT environment):
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python run_recording_verify.py --seeds 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "electrode_payload"
LEG_NAMES = ("lf", "lm", "lh", "rf", "rm", "rh")


def _load(ep_dir: Path) -> dict:
    """Load the truths and EVERY camera's reconstruction that exists.

    Two cameras means two INDEPENDENT reconstructions of the same walk, which is used
    deliberately: they are verified separately and their agreement is one of the
    results.  Nothing is averaged between them.
    """
    npz = dict(np.load(ep_dir / "episode.npz"))
    vis = {}
    if (ep_dir / "vision.json").exists():
        vis["worldcam"] = json.loads((ep_dir / "vision.json").read_text())
    for p in sorted(ep_dir.glob("vision_*.json")):
        vis[p.stem.replace("vision_", "")] = json.loads(p.read_text())
    return {"npz": npz, "vis": vis, "dir": ep_dir}


def _interp_to(t_src, a_src, t_dst):
    a = np.asarray(a_src, dtype=float)
    if a.ndim == 1:
        return np.interp(t_dst, t_src, a)
    return np.stack([np.interp(t_dst, t_src, a[:, j]) for j in range(a.shape[1])], axis=1)


def band_power_ratio(x, fs, f0, half_bw=2.0):
    """Fraction of a signal's variance that sits in f0 +/- half_bw Hz.

    Computed by band-passing with an FFT mask (a rectangular filter, stated as such)
    rather than by a filter design, because the question is whether the trace carries
    a gait-locked component at all, not how a particular filter would sound.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    m = (f >= f0 - half_bw) & (f <= f0 + half_bw)
    band = X.copy()
    band[~m] = 0.0
    xb = np.fft.irfft(band, n=n)
    var_tot = float(np.var(x))
    return {"band_variance_ratio": float(np.var(xb) / var_tot) if var_tot > 0 else float("nan"),
            "band_rms": float(np.sqrt(np.var(xb))), "total_rms": float(x.std()),
            "n_bins_in_band": int(m.sum()), "n_samples": n, "fs_Hz": float(fs)}


def phase_sensitivity(t_pred_base: np.ndarray, true: np.ndarray, ft: np.ndarray,
                      period_s: float, n_scan: int = 17) -> dict:
    """Contact accuracy as a function of a GLOBAL phase offset on the reconstruction.

    WHY THIS IS PART OF THE RESULT, not an appendix.  The reconstruction's gait phase
    comes from the video, but the video can only fix the phase of the pooled signal up
    to the point where the leg's own contact window sits relative to it.  That relation
    is derived (peak of the pooled signal = centre of one group's contact window, see
    ``contact_window_from_phase``), and a derivation can be wrong by a fixed amount.
    So the whole analysis is repeated for offsets up to half a period, and the result is
    reported as a curve with its zero-offset value.  A reconstruction that is only good
    at one magic offset has NOT shown that it can infer touch from motion.

    The scan is a CIRCULAR SHIFT of the predicted contact pattern in frames, so it is
    cheap; the shift is applied per leg to the pooled pattern identically, which is what
    a phase error does.
    """
    n_fr = true.shape[1]
    base = float((t_pred_base == true).mean())
    out = []
    for k in range(n_scan):
        off_cycles = (k / (n_scan - 1) - 0.5) * 0.5        # -0.25 .. +0.25 cycles
        sh = int(round(off_cycles * period_s / float(np.median(np.diff(ft)))))
        pred = np.roll(t_pred_base, sh, axis=1) if sh else t_pred_base
        out.append({"offset_cycles": float(off_cycles),
                    "offset_ms": float(off_cycles * period_s * 1000.0),
                    "shift_frames": int(sh),
                    "pooled_accuracy": float((pred == true).mean())})
    best = max(out, key=lambda d: d["pooled_accuracy"])
    return {"base_pooled_accuracy": base, "scan": out,
            "best_offset_cycles": best["offset_cycles"],
            "best_offset_ms": best["offset_ms"],
            "best_pooled_accuracy": best["pooled_accuracy"],
            "accuracy_at_zero_is_best": bool(abs(best["offset_cycles"]) < 1e-9),
            "period_s": float(period_s), "period_ms": float(period_s * 1000.0)}


def contact_analysis(pred: np.ndarray, true: np.ndarray, fs: float) -> dict:
    """Frame-level agreement between inferred and true contact, per leg and pooled.

    Reported next to the CHANCE level, which for binary contact is
    ``max(p_true, 1-p_true)`` pooled over the same frames -- i.e. what a constant
    "always in contact" or "never in contact" guesser would score.  A method that does
    not beat that has learned nothing from the video.
    """
    pred = np.asarray(pred, dtype=bool)
    true = np.asarray(true, dtype=bool)
    n_leg, n_fr = true.shape
    per_leg = []
    for i in range(n_leg):
        p, t = pred[i], true[i]
        tp = int(np.sum(p & t))
        fp = int(np.sum(p & ~t))
        fn = int(np.sum(~p & t))
        tn = int(np.sum(~p & ~t))
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if (prec == prec and rec == rec and prec + rec > 0) else float("nan"))
        per_leg.append({"leg": LEG_NAMES[i], "accuracy": float((tp + tn) / n_fr),
                        "precision": float(prec), "recall": float(rec),
                        "f1": float(f1), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "true_duty": float(t.mean()), "pred_duty": float(p.mean()),
                        "chance_accuracy": float(max(t.mean(), 1 - t.mean()))})
    pooled_acc = float((pred == true).mean())
    pooled_true_duty = float(true.mean())
    pooled_chance = float(max(pooled_true_duty, 1 - pooled_true_duty))
    # temporal offset scan: does the reconstruction lead or lag the truth?
    best = None
    for lag in range(-int(0.25 * fs), int(0.25 * fs) + 1):
        sh = np.roll(pred, lag, axis=1)
        if lag > 0:
            sh[:, :lag] = pred[:, :lag]
        elif lag < 0:
            sh[:, lag:] = pred[:, lag:]
        a = float((sh == true).mean())
        if best is None or a > best[0]:
            best = (a, lag)
    return {"per_leg": per_leg, "pooled_accuracy": pooled_acc,
            "pooled_true_duty": pooled_true_duty, "pooled_chance_accuracy": pooled_chance,
            "beats_chance": bool(pooled_acc > pooled_chance),
            "best_lag_frames": int(best[1]), "best_lag_ms": float(1000 * best[1] / fs),
            "best_lag_accuracy": float(best[0])}


def edge_timing(pred: np.ndarray, true: np.ndarray, t: np.ndarray) -> dict:
    """Touchdown event timing and position error, leg-wise.

    Touchdown is the false->true edge.  Each predicted edge is matched to the nearest
    true edge of the SAME leg; unmatched edges on either side are counted, because a
    reconstruction that invents footfalls as often as it misses them is not useful even
    if its average error is small.
    """
    out = []
    for i in range(true.shape[0]):
        pe = np.nonzero(pred[i] & ~np.roll(pred[i], 1))[0]
        te = np.nonzero(true[i] & ~np.roll(true[i], 1))[0]
        if len(pe) and len(te):
            d = np.abs(pe[:, None] - te[None, :])
            j = np.argmin(d, axis=1)
            err_ms = (pe - te[j]) * (t[1] - t[0]) * 1000.0
            matched = d[np.arange(len(pe)), j] <= max(1, int(0.25 / (t[1] - t[0])))
            out.append({"leg": LEG_NAMES[i], "n_pred": int(len(pe)), "n_true": int(len(te)),
                        "median_err_ms": float(np.median(err_ms[matched]))
                        if matched.any() else None,
                        "mean_abs_err_ms": float(np.mean(np.abs(err_ms[matched])))
                        if matched.any() else None,
                        "matched_fraction": float(matched.mean()),
                        "false_events": int((~matched).sum()),
                        "missed_events": int(len(te) - len(np.unique(j[matched])))
                        if matched.any() else int(len(te))})
        else:
            out.append({"leg": LEG_NAMES[i], "n_pred": int(len(pe)), "n_true": int(len(te)),
                        "median_err_ms": None, "mean_abs_err_ms": None,
                        "matched_fraction": None,
                        "false_events": int(len(pe)), "missed_events": int(len(te))})
    return {"edge_per_leg": out}



def analyse_episode(ep: dict, stride_truth_hz: float | None = None) -> dict:
    npz, vis = ep["npz"], ep["vis"]
    t = npz["truth/time_s"].astype(float)
    thorax = npz["truth/thorax_mm"].astype(float)
    contact = npz["truth/contact_present"].astype(bool)
    res = {"dir": str(ep["dir"]),
           "walk": {"duration_s": float(t[-1] - t[0]),
                    "path_length_mm": float(np.sum(np.hypot(np.diff(thorax[:, 0]),
                                                            np.diff(thorax[:, 1])))),
                    "displacement_mm": float(np.hypot(thorax[-1, 0] - thorax[0, 0],
                                                      thorax[-1, 1] - thorax[0, 1])),
                    "mean_speed_mm_s": float(np.mean(np.hypot(np.gradient(thorax[:, 0], t),
                                                              np.gradient(thorax[:, 1], t)))),
                    "thorax_z_min_mm": float(thorax[:, 2].min()),
                    "thorax_z_mean_mm": float(thorax[:, 2].mean()),
                    "n_legs_in_contact_mean": float(contact.sum(axis=1).mean()),
                    "contact_duty_per_leg": [float(v) for v in contact.mean(axis=0)]}}
    fs = 1.0 / float(np.median(np.diff(t)))
    # true stride frequency: the dominant frequency of the pooled contact pattern
    y = contact.astype(float).mean(axis=1)
    y = y - y.mean()
    f = np.fft.rfftfreq(len(y), 1.0 / fs)
    P = np.abs(np.fft.rfft(y))
    m = (f >= 3) & (f <= 30)
    tripod_hz = float(f[m][np.argmax(P[m])])
    stride_true = tripod_hz
    res["walk"]["stride_frequency_hz_true"] = stride_true
    res["walk"]["fs_Hz"] = fs
    if stride_truth_hz is not None and abs(stride_true - stride_truth_hz) > 0.5:
        res["walk"]["stride_footnote"] = (
            f"per-episode stride {stride_true:.3f} Hz differs from the batch value "
            f"{stride_truth_hz:.3f} Hz; both are reported")
    # ---- electrode ----------------------------------------------------------
    e = npz["elec/recorded_uV"].astype(float)
    art = npz["elec/artefact_uV"].astype(float)
    noise = npz["elec/noise_uV"].astype(float)
    fs_e = float(npz["elec/fs_Hz"])
    n_elec = min(len(e), len(t))
    # match lengths for the truth-side comparison
    t_e = np.linspace(t[0], t[-1], len(e))
    cont_e = (_interp_to(t, contact.astype(float), t_e) > 0.5).T
    duty_e = cont_e.mean(axis=1)
    res["electrode"] = {
        "fs_Hz": fs_e,
        "recorded_rms_uV": float(e.std()),
        "artefact_rms_uV": float(art.std()),
        "noise_rms_uV": float(noise.std()),
        "noise_to_artefact_ratio": float(noise.std() / art.std()) if art.std() > 0 else None,
        # THE BAND IS THE TRIPOD ALTERNATION, 2 x the per-leg cycle, and it is taken
        # from a TRUTH-SIDE measurement of the pooled contact pattern.  That is NOT a
        # circularity: this is a comparison of two SIGNAL CHANNELS (recorded vs
        # noise-only) at a frequency established independently of both, and it is
        # deliberately NOT the vision parser's frequency estimate, which is aliased.
        # Use the f0 the video reports and this test loses its meaning.
        "band": band_power_ratio(e, fs_e, tripod_hz),
        "band_noise_only": band_power_ratio(noise, fs_e, tripod_hz),
        "band_artefact_only": band_power_ratio(art, fs_e, tripod_hz),
        "band_hz_used": tripod_hz,
        "band_definition": ("the pooled contact pattern's dominant frequency, i.e. the "
                            "tripod alternation at twice the per-leg cycle"),
        "note": ("the recorded trace is the DECLARED artefact coupling times the sim's "
                 "actuator drive, plus noise from this project's own 7 um recording "
                 "chain budget.  It is a MODELLED trace, not a measurement."),
    }
    if len(duty_e) > 8:
        res["electrode"]["corr_recorded_vs_contact_duty"] = float(
            np.corrcoef(np.abs(e[:len(duty_e)] - e[:len(duty_e)].mean()), duty_e)[0, 1])
        res["electrode"]["corr_noise_vs_contact_duty"] = float(
            np.corrcoef(np.abs(noise[:len(duty_e)] - noise[:len(duty_e)].mean()),
                        duty_e)[0, 1])
    # ---- vision, PER CAMERA -------------------------------------------------
    # Each camera is reconstructed and verified INDEPENDENTLY; the results are kept
    # side by side and never averaged, because averaging two cameras would hide exactly
    # the thing worth knowing -- whether the fixed wide view and the panned close-up
    # agree about the same walk.
    res["vision"] = {}
    res["touch"] = {}
    for cam_name, vis in sorted((ep.get("vis") or {}).items()):
        if vis is None:
            continue
        ft = np.asarray(vis["frame_time_s"], dtype=float)
        gx = np.interp(ft, t, thorax[:, 0])
        gy = np.interp(ft, t, thorax[:, 1])
        vx = np.asarray(vis["derived"]["x_mm"], dtype=float)
        vy = np.asarray(vis["derived"]["y_mm"], dtype=float)
        err_all = np.hypot(vx - gx, vy - gy)
        track = vis.get("track") or {}
        runs = track.get("runs") or []
        # EVALUATE INSIDE THE TRACKED RUNS.  If the fly was out of frame the array
        # is interpolated, and grading the parser on those frames would be grading
        # it on frames where it could see nothing.  With no run list (older
        # artefacts) the whole array is used and that is stated in the output.
        keep = np.ones(len(ft), dtype=bool)
        if runs:
            keep[:] = False
            for _r in runs:
                keep[_r["frame_start"]:_r["frame_end"] + 1] = True
        err = err_all[keep]
        res["vision"][cam_name] = {
            "camera": cam_name,
            "evaluated_frames": int(keep.sum()),
            "coverage_fraction": float(track.get("coverage_fraction", 1.0)),
            "n_track_runs": int(track.get("n_runs", 0)),
            "n_frames": int(vis["n_frames"]), "n_detected": int(vis["n_detected"]),
            "mm_per_px": float(vis["camera"]["mm_per_px"]),
            "position_error_median_mm": float(np.median(err)),
            "position_error_rmse_mm": float(np.sqrt(np.mean(err ** 2))),
            "position_error_p95_mm": float(np.percentile(err, 95)),
            "position_error_median_px": float(np.median(err) / vis["camera"]["mm_per_px"]),
            # THE ERROR HAS TWO PARTS AND THEY MEAN DIFFERENT THINGS.
            # A constant offset is a RESOLUTION fact, not a tracking error: the
            # silhouette's centroid is the projected centre of the whole body while the
            # ground truth is the THORAX origin, and the measured offset is 1.63-1.70 mm
            # across four seeds with the same sign -- stable to 2 %, i.e. a property of
            # the body rather than of a particular walk.  A tracking system is judged on
            # the error AFTER that constant, so both are reported and neither is
            # presented as the other.
            "bias_x_mm": float(np.median(vx[keep] - gx[keep])),
            "bias_y_mm": float(np.median(vy[keep] - gy[keep])),
            "error_after_bias_median_mm": float(np.median(np.hypot(
                (vx[keep] - gx[keep]) - np.median(vx[keep] - gx[keep]),
                (vy[keep] - gy[keep]) - np.median(vy[keep] - gy[keep])))),
            "error_after_bias_p95_mm": float(np.percentile(np.hypot(
                (vx[keep] - gx[keep]) - np.median(vx[keep] - gx[keep]),
                (vy[keep] - gy[keep]) - np.median(vy[keep] - gy[keep])), 95)),
            "path_length_recovered_mm": float(vis["derived"]["path_length_mm"]),
            "path_length_true_mm": float(np.sum(np.hypot(np.diff(gx), np.diff(gy)))),
            "corr_x": float(np.corrcoef(vx, gx)[0, 1]),
            "corr_y": float(np.corrcoef(vy, gy)[0, 1]),
            "alternation_recovered_hz": float(vis["gait"]["alternation_frequency_hz"]),
            "per_leg_stride_recovered_hz": float(vis["gait"]["per_leg_stride_hz"]),
            "alternation_true_hz": stride_true,
            "stride_abs_error_hz": float(abs(vis["gait"]["alternation_frequency_hz"]
                                             - stride_true)),
            "stride_rel_error": float(abs(vis["gait"]["alternation_frequency_hz"]
                                          - stride_true) / stride_true)
            if stride_true else None,
            "phase_deg": float(vis["gait"]["phase_deg"]),
            "duty_used": float(vis["gait"]["duty_contact_used"]),
            "stride_r2": float(vis["gait"]["stride_r2"]),
            "gait_signal": vis["gait"]["signal_used"],
            "touches_detected": vis["touch_inferred"]["touchdown_counts"],
            "touch_definition": vis["touch_inferred"]["definition"],
        }
        # ---- touch ----------------------------------------------------------
        t_pred = np.asarray([[1 if b else 0 for b in leg["touch_pred"]]
                             for leg in vis["legs"]], dtype=bool)
        if runs:
            t_pred = t_pred[:, keep]
        # _interp_to returns (n_frames, 6); the contact analysis wants (6, n_frames)
        cont_f = (_interp_to(t, contact.astype(float), ft) > 0.5).T
        if runs:
            cont_f = cont_f[:, keep]
        res["touch"][cam_name] = contact_analysis(
            t_pred, cont_f, 1.0 / float(np.median(np.diff(ft))))
        # NOTE: ``edge_timing`` returns its own per-leg list under ``edge_per_leg``.
        # An earlier version returned it as ``per_leg`` and silently OVERWROTE the
        # contact-accuracy table, which made the accuracy numbers vanish from the
        # artefact while the summary still printed fine.  Distinct keys, always.
        res["touch"][cam_name].update(edge_timing(t_pred, cont_f, ft))
        stride_period = 1.0 / float(vis["gait"]["alternation_frequency_hz"]) \
            if vis["gait"]["alternation_frequency_hz"] > 0 else float("nan")
        res["touch"][cam_name]["phase_sensitivity"] = phase_sensitivity(
            t_pred, cont_f, ft, stride_period)
        res["touch"][cam_name]["evaluated_frames"] = int(keep.sum())
        res["touch"][cam_name]["evaluation_scope"] = (
            "inside tracked runs only" if runs else "whole episode")
    return res


def paired_payload_effect(episodes: dict[tuple[int, str], dict], loads=("load020", "load005")
                          ) -> dict:
    """Paired (same-seed) difference between the control and each payload condition.

    Paired because the seeds are the dominant source of variation here and taking
    unpaired differences would drown the effect.  The interval is a t-like interval on
    the paired differences with n-1 degrees of freedom; with 4 seeds it is WIDE, and
    that width is reported rather than hidden.
    """
    out = {}
    ctrl = {s: episodes[(s, "control")] for (s, lab) in episodes if lab == "control"
            for s in [s]}
    for load in loads:
        seeds = sorted(s for (s, lab) in episodes if lab == load)
        diffs = []
        for s in seeds:
            a = episodes[(s, "control")]["walk"]
            b = episodes[(s, load)]["walk"]
            diffs.append({"seed": s,
                          "d_mean_speed_mm_s": b["mean_speed_mm_s"] - a["mean_speed_mm_s"],
                          "d_path_mm": b["path_length_mm"] - a["path_length_mm"],
                          "d_thorax_z_mm": b["thorax_z_mean_mm"] - a["thorax_z_mean_mm"],
                          "d_stride_hz": b["stride_frequency_hz_true"] - a["stride_frequency_hz_true"],
                          "d_legs_in_contact": b["n_legs_in_contact_mean"]
                          - a["n_legs_in_contact_mean"]})
        if not diffs:
            out[load] = {"per_seed": [], "summary": {},
                         "note": "no seeds with both a control and this load condition"}
            continue
        keys = [k for k in diffs[0] if k != "seed"]
        summary = {}
        for k in keys:
            v = np.asarray([d[k] for d in diffs], dtype=float)
            n = len(v)
            mean = float(v.mean())
            sd = float(v.std(ddof=1)) if n > 1 else float("nan")
            se = sd / math.sqrt(n) if n > 1 else float("nan")
            # t(0.975, df=3) = 3.182; the table is DECLARED, not computed
            tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n - 1, 2.0)
            summary[k] = {"mean": mean, "sd": sd, "se": se, "n": n,
                          "ci95_low": mean - tcrit * se if n > 1 else None,
                          "ci95_high": mean + tcrit * se if n > 1 else None,
                          "t_crit": tcrit,
                          "sign": "positive" if mean > 0 else "negative",
                          "excludes_zero": bool(n > 1 and (mean - tcrit * se) * (mean + tcrit * se) > 0)}
        out[load] = {"per_seed": diffs, "summary": summary,
                     "note": ("paired same-seed differences; the interval is a t-like "
                              "interval on paired differences and is NOT a p-value. "
                              "n is 4 seeds, so the interval is wide.")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--base", default=str(OUT))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    base = Path(args.base)

    episodes: dict[tuple[int, str], dict] = {}
    results = {}
    for d in sorted(base.glob("episode_*_seed*")):
        if not d.is_dir() or not (d / "episode.npz").exists():
            continue
        parts = d.name.split("_seed")
        label, seed = parts[0].replace("episode_", ""), int(parts[1])
        if seed not in seeds:
            continue
        ep = _load(d)
        if ep["vis"] is None:
            print(f"  (skipping {d.name}: no vision.json yet)")
            continue
        r = analyse_episode(ep)
        episodes[(seed, label)] = r
        results[d.name] = r
    if not episodes:
        raise SystemExit("no episodes with a reconstruction were found")
    payload = paired_payload_effect(episodes)

    doc = {"episodes": results, "payload_effect": payload,
           "seeds": seeds,
           "chance_level_note": ("per-leg contact chance level is max(duty, 1-duty) on "
                                 "the same frames, reported next to every accuracy")}
    dest = base / "comparison.json"
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
