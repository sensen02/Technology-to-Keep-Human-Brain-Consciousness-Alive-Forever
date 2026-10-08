"""run_plasticity_ablation.py -- the PRE-REGISTERED switchable-plasticity ablation.

Run in the BODY environment:
    ./venv_body/bin/python run_plasticity_ablation.py
    ./venv_body/bin/python run_plasticity_ablation.py --stages tier,loop   (default all)

WHAT IS RUN
-----------
ONE bounded BANC conductance tier (``engine.embodied.adapters.build_neural_tier``, 12,000
neurons, 291,986 modelled edges), six PAIRED ablation arms that differ ONLY in which
plasticity mechanisms are enabled (``engine.embodied.plasticity.ABLATION_ARMS``), the
same seed and the same initial state in every arm:

    frozen | stp_only | homeostasis_only | long_term_only | stp_homeostasis | all

Stage A  drive ladder, 10 levels x 2.5 s of model time per arm, with a kick/hysteresis
         probe at every level.
Stage B  pattern probe: the same drive delivered to a PERMUTED set of leg groups.
Stage C  a REAL declared lesion (the project's own neck-cut row mask, 36,989 modelled
         in-tier edges) with rewiring driven at maximum feedback, to test the barrier.
Stage D  the SAME six arms driven through the EXISTING closed loop
         (``engine.embodied.loop.MultirateScheduler``) for the behavioural readout
         (speed and heading), 3 seeds, 8 s each.

THE FROZEN ARM IS THE BASELINE AND IS NOT MODIFIED.  With every switch off,
``PlasticNeuralTier.advance`` calls the parent ``NeuralTier.advance`` itself, so the
frozen arm is bit-identical to the pre-existing un-plastic tier; that identity, and the
bit-identity of a frozen-arm closed-loop episode against a plain-tier episode, are
asserted in ``run_plasticity_selftest.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import resource
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs", "embodied_body")

from engine.embodied.adapters import (                                    # noqa: E402
    HONESTY, LEGS, TierConfig, build_neural_tier)
from engine.embodied.loop import LoopConfig, MultirateScheduler           # noqa: E402
from engine.embodied.plasticity import (                                  # noqa: E402
    ABLATION_ARMS, PLASTICITY_HONESTY, HomeostasisConfig, LesionBarrier,
    PlasticNeuralTier, PlasticityConfig, RewiringConfig, attach_plasticity,
    tripod_alternation_reward)

# ===========================================================================
# PRE-REGISTERED PREDICTIONS -- written before any number in this file was produced.
# ===========================================================================
PREDICTIONS = {
    "P1": ("WITHOUT homeostasis the tier shows a bimodal silent/epileptiform regime; an "
           "intermediate drive level does NOT produce a stable graded rate."),
    "P2": ("WITH homeostasis enabled, a stable graded regime exists at the same "
           "intermediate drive."),
    "P3": ("Homeostasis does not simply clamp the rate regardless of input."),
}

#: DISCLOSURE, stated up front rather than buried: the constants BELOW were fixed after a
#: short exploratory pass whose purpose was to establish the tier's operating range and to
#: set the scale of one learning rate so the long-term arm is neither invisible nor
#: saturated.  They were NOT changed after the pre-registered runs began, and no
#: prediction criterion below was altered to make a prediction pass.  Every constant is
#: declared here, in code, before the numbers in the report exist.
PRE_REGISTRATION_DISCLOSURE = (
    "PRE-REGISTRATION: the six arms, the criteria, the thresholds and the stop rule for "
    "each prediction were written down before the reported runs. DISCLOSURE: an earlier "
    "exploratory pass (not part of this report) established the tier's operating range and "
    "fixed exactly two scales -- the drive ladder's extent and long_term.eta -- so that the "
    "sweep spans the silent, graded and saturated regimes and the long-term arm is neither "
    "invisible nor saturated; all plasticity mechanism parameters were fixed in that same "
    "pass. No prediction criterion was changed afterwards and no gain was re-tuned to make "
    "a prediction pass.")

# ---------------------------------------------------------------------------
# PRE-REGISTERED RUN PARAMETERS (declared before the measurement)
# ---------------------------------------------------------------------------
#: five decades of drive, geometrically spaced, plus exactly zero.  The intermediate
#: level is the GEOMETRIC CENTRE of the nine nonzero levels (0.0005 * 10**2 = 0.05).
DRIVE_LADDER_nA = (0.0, 0.0005, 0.0016, 0.005, 0.016, 0.05, 0.16, 0.5, 1.6, 5.0)
PRIMARY_DRIVE_nA = 0.05
#: the loop's own reachable per-neuron drive, from the PRE-EXISTING adapter constants
#: (current_nA_per_drive 0.05 * receptor_gain_mV 10 * (contact_weight 1.0 +
#: proprio_weight 0.5)); reported as a second declared "intermediate" level.
LOOP_REACHABLE_MAX_nA = 0.75
LOOP_REACHABLE_MID_nA = 0.375

SETTLE_MS = 2500.0
KICK_SETTLE_MS = 300.0
KICK_MS = 30.0
KICK_nA = 0.5
KICK_TAIL_MS = 1200.0
PATTERN_MS = 1500.0
PATTERN_DRIVE_nA = 0.05     # per loaded leg, the intermediate drive

SILENT_HZ = 1.0                 # population rate below this = the tier is silent
EPILEPTIFORM_HZ = 200.0         # OPERATIONAL threshold, declared: the lower decade edge
#                                 of the 245-320 Hz range this project reported in the
#                                 earlier LIF closed loop.  It is NOT a physiological
#                                 threshold and no physiology is claimed for it.
STABLE_CV_MAX = 0.25            # tail coefficient of variation over 50 ms bins
JUMP_RATIO = 3.0                # adjacent-ladder settled-rate ratio counted as a jump
BISTABLE_FACTOR = 3.0           # kicked tail must exceed 3x the un-kicked tail ...
BISTABLE_MIN_DELTA_HZ = 5.0     # ... and by at least this many Hz, to count as latching
P3_VARIATION_MIN = 2.0          # settled rate max/min over the nonzero ladder
P3_CLAMP_BAND = 0.25            # "within +/-25% of the homeostatic target"
P3_CLAMP_SPAN_DECADES = 10.0    # over a >=10x span of drive
P3_PATTERN_JACCARD_MAX = 0.90   # permuted input must change the active set by >=10%

ABLATION_SEEDS = (0, 1, 2)
LOOP_SECONDS = 8.0
LOOP_ARM = "neural_modulated"
BIN_MS = 50.0

T0 = time.perf_counter()


def peak_ram_mb():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def hours(sec):
    return "%.2f" % (sec / 3600.0)


# ---------------------------------------------------------------------------
# tier plumbing
# ---------------------------------------------------------------------------
class TierProbe:
    """One arm in one condition: build, drive, record, summarise."""

    def __init__(self, built, arm_name, seed_note=""):
        self.built = built
        self.arm = arm_name if isinstance(arm_name, str) else "ad_hoc"
        self.cfg = (ABLATION_ARMS[arm_name] if isinstance(arm_name, str)
                    else arm_name)
        self.tier = PlasticNeuralTier(built, TierConfig(), _Ledger(), plasticity=self.cfg)
        self.groups = [np.asarray(built["sensory"][leg]) for leg in LEGS]
        self.n = int(built["n"])
        self.dt_ms = float(TierConfig().dt_ms)

    def current(self, per_leg_nA):
        """Per-neuron current from a length-6 per-leg drive vector (LEGS order)."""
        d = np.asarray(per_leg_nA, float)
        if d.shape != (6,):
            raise ValueError("per_leg_nA must have one entry per leg")
        c = np.zeros(self.n)
        for leg, g in zip(LEGS, self.groups):
            if d[LEGS.index(leg)]:
                c[g] += float(d[LEGS.index(leg)])
        return c

    def uniform(self, drive_nA):
        return np.full(6, float(drive_nA))

    def run(self, drive_nA, ms, kick_nA=0.0, kick_ms=0.0, kick_at_ms=None,
            per_leg=None, feedback=None):
        """Run ``ms`` of model time; optionally a kick window after ``kick_at_ms``."""
        self.tier.reset()
        if feedback is not None:
            self.tier.set_feedback(float(feedback))
        base = self.uniform(drive_nA) if per_leg is None else np.asarray(per_leg, float)
        blocks = int(round((ms - kick_ms) / self.dt_ms / 10.0))
        spike_rows = []
        if kick_ms > 0.0 and kick_at_ms is not None:
            pre = int(round(kick_at_ms / self.dt_ms / 10.0))
            for _ in range(pre):
                spike_rows.append(self.tier.advance(10, self.current(base)))
            kb = int(round(kick_ms / self.dt_ms / 10.0))
            for _ in range(kb):
                spike_rows.append(self.tier.advance(10, self.current(
                    base + np.full(6, float(kick_nA)) if per_leg is None
                    else base + np.full(6, float(kick_nA)))))
            post = blocks - pre
            for _ in range(post):
                spike_rows.append(self.tier.advance(10, self.current(base)))
        else:
            for _ in range(blocks):
                spike_rows.append(self.tier.advance(10, self.current(base)))
        return np.concatenate(spike_rows, axis=0) if spike_rows else np.zeros((0, self.n),
                                                                             bool)


class _Ledger:
    """A private ledger so these runs never mix counts with a loop episode's ledger."""

    def write(self, resource, writer):
        return self


def summarise(spikes, dt_ms, tier=None, tail_from_ms=None, bin_ms=BIN_MS):
    """Firing-rate distribution over time, silent/epileptiform fractions, stability."""
    sp = np.asarray(spikes)
    n = sp.shape[1]
    per_step = sp.sum(axis=1).astype(float) * 1000.0 / dt_ms / n
    per_bin = int(round(bin_ms / dt_ms))
    nb = per_step.size // per_bin
    binned = per_step[:nb * per_bin].reshape(nb, per_bin).mean(axis=1) if nb else \
        np.zeros(0)
    t_ms = (np.arange(per_step.size) + 1) * dt_ms
    start = 0.0 if tail_from_ms is None else float(tail_from_ms)
    m = t_ms >= start
    tail = per_step[m]
    if nb:
        bin_start = np.arange(nb) * bin_ms
        tb = binned[bin_start + bin_ms > start]
    else:
        tb = np.zeros(0)
    cv = float(tb.std() / tb.mean()) if tb.size and tb.mean() > 0 else float("nan")
    ro = sp[:, tier.readout_local] if tier is not None else np.zeros((sp.shape[0], 0))
    ro_step = (ro.sum(axis=1).astype(float) * 1000.0 / dt_ms / max(ro.shape[1], 1)
               if ro.size else np.zeros(0))
    neuron_hz = sp.sum(axis=0).astype(float) * 1000.0 / dt_ms / (sp.shape[0] or 1)
    return {
        "n_steps": int(sp.shape[0]),
        "duration_ms": float(sp.shape[0] * dt_ms),
        "tail_from_ms": float(start),
        "pop_rate_hz_mean": float(per_step.mean()),
        "pop_rate_hz_tail_mean": float(tail.mean()) if tail.size else float("nan"),
        "pop_rate_hz_tail_sd": float(tail.std()) if tail.size else float("nan"),
        "pop_rate_hz_tail_cv_50ms_bins": cv,
        "pop_rate_hz_bin_max": float(binned.max()) if nb else float("nan"),
        "pop_rate_hz_bin_min": float(binned.min()) if nb else float("nan"),
        "pop_rate_hz_bin_p05": float(np.percentile(tb, 5)) if tb.size else float("nan"),
        "pop_rate_hz_bin_p50": float(np.percentile(tb, 50)) if tb.size else float("nan"),
        "pop_rate_hz_bin_p95": float(np.percentile(tb, 95)) if tb.size else float("nan"),
        "fraction_of_time_silent": float((tb < SILENT_HZ).mean()) if tb.size else
        float("nan"),
        "fraction_of_time_epileptiform": float((tb > EPILEPTIFORM_HZ).mean())
        if tb.size else float("nan"),
        "fraction_of_time_graded": float(((tb >= SILENT_HZ)
                                          & (tb <= EPILEPTIFORM_HZ)).mean())
        if tb.size else float("nan"),
        "readout_rate_hz_tail_mean": float(ro_step[m].mean()) if ro_step.size else
        float("nan"),
        "neuron_rate_hz_mean": float(neuron_hz.mean()),
        "neuron_rate_hz_p50": float(np.percentile(neuron_hz, 50)),
        "neuron_rate_hz_p99": float(np.percentile(neuron_hz, 99)),
        "neuron_rate_hz_max": float(neuron_hz.max()),
        "fraction_of_neurons_silent": float((neuron_hz < SILENT_HZ).mean()),
        "pop_rate_series_hz": per_step,
        "binned_series_hz": binned,
        "bin_ms": float(bin_ms),
    }


def tier_state_metrics(tier, hist):
    h = hist
    last = slice(max(0, len(h["g_syn"]) - 20), None)
    rep = tier.plasticity_report()
    lt_mean = np.asarray(h["lt_mean_frac"], float)
    lt_max = np.asarray(h["lt_max_abs_frac"], float)
    return {
        "g_syn_final": float(np.mean(h["g_syn"][last])) if h["g_syn"] else 1.0,
        "g_exc_final": float(np.mean(h["g_exc"][last])) if h["g_exc"] else 1.0,
        "homeo_z_final": float(h["homeo_z"][-1]) if h["homeo_z"] else 0.0,
        "stp_mean_resource_final": float(np.mean(h["stp_mean_x"][last]))
        if h["stp_mean_x"] else 1.0,
        "stp_min_resource": float(np.min(h["stp_mean_x"])) if h["stp_mean_x"] else 1.0,
        "lt_mean_weight_change_final": float(lt_mean[-1]) if lt_mean.size else 0.0,
        "lt_max_abs_weight_change": float(lt_max.max()) if lt_max.size else 0.0,
        "feedback_min": float(np.min(h["feedback"])) if h["feedback"] else 0.0,
        "feedback_max": float(np.max(h["feedback"])) if h["feedback"] else 0.0,
        "n_edges_start": int(rep["n_edges_at_start_after_lesion"]),
        "n_edges_end": int(rep["n_edges_now"]),
        "n_edges_added": int(rep["rewiring"]["added"]),
        "n_edges_removed": int(rep["rewiring"]["removed"]),
        "n_refused_by_lesion_barrier": int(rep["rewiring"]["n_refused_by_lesion_barrier"]),
        "stp_bound_violations_clipped": int(rep["counters"]["stp_bound_violations_clipped"]),
        "lt_fraction_edges_at_bound_max":
            float(rep["counters"]["long_term_fraction_of_edges_at_a_bound_max"]),
        "g_syn_series": np.asarray(h["g_syn"], float),
        "g_exc_series": np.asarray(h["g_exc"], float),
        "lt_max_series": lt_max,
        "stp_x_series": np.asarray(h["stp_mean_x"], float),
        "feedback_series": np.asarray(h["feedback"], float),
    }


# ---------------------------------------------------------------------------
# stage A: the drive ladder + the kick/hysteresis probe
# ---------------------------------------------------------------------------
def stage_ladder(built, arms, ladder, log):
    out = {}
    for arm in arms:
        out[arm] = {}
        for d in ladder:
            probe = TierProbe(built, arm)
            t0 = time.perf_counter()
            spikes = probe.run(d, SETTLE_MS)
            base = summarise(spikes, probe.dt_ms, tier=probe.tier,
                             tail_from_ms=SETTLE_MS * 0.5)
            state = tier_state_metrics(probe.tier, probe.tier.history)
            probe2 = TierProbe(built, arm)
            spikes_k = probe2.run(d, KICK_SETTLE_MS + KICK_MS + KICK_TAIL_MS,
                                  kick_nA=KICK_nA, kick_ms=KICK_MS,
                                  kick_at_ms=KICK_SETTLE_MS)
            kick = summarise(spikes_k, probe2.dt_ms, tier=probe2.tier,
                             tail_from_ms=KICK_SETTLE_MS + KICK_MS)
            wall = time.perf_counter() - t0
            out[arm]["%g" % d] = {
                "drive_nA": float(d),
                "settled_rate_hz": base["pop_rate_hz_tail_mean"],
                "settled_rate_hz_readout": base["readout_rate_hz_tail_mean"],
                "settled_cv_50ms_bins": base["pop_rate_hz_tail_cv_50ms_bins"],
                "fraction_of_time_silent": base["fraction_of_time_silent"],
                "fraction_of_time_epileptiform": base["fraction_of_time_epileptiform"],
                "fraction_of_time_graded": base["fraction_of_time_graded"],
                "pop_rate_hz_bin_p05": base["pop_rate_hz_bin_p05"],
                "pop_rate_hz_bin_p50": base["pop_rate_hz_bin_p50"],
                "pop_rate_hz_bin_p95": base["pop_rate_hz_bin_p95"],
                "pop_rate_hz_bin_max": base["pop_rate_hz_bin_max"],
                "neuron_rate_hz_mean": base["neuron_rate_hz_mean"],
                "neuron_rate_hz_p50": base["neuron_rate_hz_p50"],
                "neuron_rate_hz_p99": base["neuron_rate_hz_p99"],
                "fraction_of_neurons_silent": base["fraction_of_neurons_silent"],
                "kick_settled_rate_hz": kick["pop_rate_hz_tail_mean"],
                "kick_latched": bool(
                    kick["pop_rate_hz_tail_mean"]
                    > max(BISTABLE_FACTOR * max(base["pop_rate_hz_tail_mean"], 1e-9),
                          base["pop_rate_hz_tail_mean"] + BISTABLE_MIN_DELTA_HZ)),
                "bin_ms": BIN_MS,
                "duration_ms": base["duration_ms"],
                "pop_rate_hz_tail_sd": base["pop_rate_hz_tail_sd"],
                "neuron_rate_hz_max": base["neuron_rate_hz_max"],
                "n_steps": base["n_steps"],
                "wall_seconds": wall,
                **state,
            }
            log("  %-16s drive %8.4f nA  settled %8.2f Hz  cv %6.3f  kick %8.2f Hz  "
                "latched %-5s  silent %4.2f  epi %4.2f  wall %5.1fs"
                % (arm, d, base["pop_rate_hz_tail_mean"],
                   base["pop_rate_hz_tail_cv_50ms_bins"], kick["pop_rate_hz_tail_mean"],
                   out[arm]["%g" % d]["kick_latched"], base["fraction_of_time_silent"],
                   base["fraction_of_time_epileptiform"], wall), flush=True)
            # keep only the time series for the PRIMARY drive, to bound the npz size
            out[arm]["%g" % d]["_binned_series_hz"] = (
                base["binned_series_hz"] if abs(d - PRIMARY_DRIVE_nA) < 1e-12
                else base["binned_series_hz"][:0])
            out[arm]["%g" % d]["_pop_series_hz"] = (
                base["pop_rate_series_hz"] if abs(d - PRIMARY_DRIVE_nA) < 1e-12
                else base["pop_rate_series_hz"][:0])
            out[arm]["%g" % d]["_lt_series"] = (state["lt_max_series"]
                                                if abs(d - PRIMARY_DRIVE_nA) < 1e-12
                                                else state["lt_max_series"][:0])
            out[arm]["%g" % d]["_g_exc_series"] = (
                state["g_exc_series"] if abs(d - PRIMARY_DRIVE_nA) < 1e-12
                else state["g_exc_series"][:0])
            out[arm]["%g" % d]["_g_syn_series"] = (
                state["g_syn_series"] if abs(d - PRIMARY_DRIVE_nA) < 1e-12
                else state["g_syn_series"][:0])
            out[arm]["%g" % d]["_feedback_series"] = (
                state["feedback_series"] if abs(d - PRIMARY_DRIVE_nA) < 1e-12
                else state["feedback_series"][:0])
            for k in ("g_syn_series", "g_exc_series", "lt_max_series", "stp_x_series",
                      "feedback_series"):
                out[arm]["%g" % d].pop(k, None)
    return out


# ---------------------------------------------------------------------------
# stage B: the pattern probe (a spatially WRONG input)
# ---------------------------------------------------------------------------
def stage_pattern(built, arms, drive, log):
    out = {}
    for arm in arms:
        rec = {}
        # THE SPATIALLY WRONG INPUT: the SAME total current is delivered, but to the
        # MIRRORED tripod group. A global homeostatic servo sees an identical total load
        # and is blind to the difference; a response that is genuinely input-specific is
        # not.  The drive pattern is declared: tripod A alone, versus tripod B alone.
        A = (PATTERN_DRIVE_nA, 0.0, PATTERN_DRIVE_nA,
             0.0, PATTERN_DRIVE_nA, 0.0)                  # lf, rm, lh loaded
        B = (0.0, PATTERN_DRIVE_nA, 0.0,
             PATTERN_DRIVE_nA, 0.0, PATTERN_DRIVE_nA)
        for tag, pat in (("intact_tripod_A_loaded", A),
                         ("mirrored_tripod_B_loaded", B)):
            probe = TierProbe(built, arm)
            t0 = time.perf_counter()
            spikes = probe.run(drive, PATTERN_MS, per_leg=pat)
            s = summarise(spikes, probe.dt_ms, tier=probe.tier,
                          tail_from_ms=PATTERN_MS * 0.5)
            late = spikes[int(PATTERN_MS * 0.5 / probe.dt_ms):]
            active = np.flatnonzero(late.any(axis=0))
            rec[tag] = {"settled_rate_hz": s["pop_rate_hz_tail_mean"],
                        "readout_rate_hz": s["readout_rate_hz_tail_mean"],
                        "n_active_neurons": int(active.size),
                        "active_set": active, "wall_seconds": time.perf_counter() - t0}
        a = set(rec["intact_tripod_A_loaded"]["active_set"].tolist())
        b = set(rec["mirrored_tripod_B_loaded"]["active_set"].tolist())
        jac = float(len(a & b) / len(a | b)) if (a | b) else 1.0
        out[arm] = {
            "drive_nA": float(PATTERN_DRIVE_nA),
            "pattern_declared": ("tripod A legs (lf, rm, lh) driven alone, versus the "
                                 "mirrored tripod B legs (rf, lm, rh) driven alone, at "
                                 "the same per-leg drive, so the TOTAL delivered current "
                                 "is identical and only the spatial pattern differs"),
            "intact_settled_rate_hz": rec["intact_tripod_A_loaded"]["settled_rate_hz"],
            "permuted_settled_rate_hz": rec["mirrored_tripod_B_loaded"]["settled_rate_hz"],
            "intact_readout_rate_hz":
                rec["intact_tripod_A_loaded"]["readout_rate_hz"],
            "permuted_readout_rate_hz": rec["mirrored_tripod_B_loaded"]["readout_rate_hz"],
            "n_active_intact": rec["intact_tripod_A_loaded"]["n_active_neurons"],
            "n_active_permuted": rec["mirrored_tripod_B_loaded"]["n_active_neurons"],
            "active_set_jaccard": jac,
            "rate_ratio_mirrored_over_intact": (
                rec["mirrored_tripod_B_loaded"]["settled_rate_hz"]
                / rec["intact_tripod_A_loaded"]["settled_rate_hz"]
                if rec["intact_tripod_A_loaded"]["settled_rate_hz"] else float("inf")),
            "input_changed_response": bool(jac < P3_PATTERN_JACCARD_MAX),
            "wall_seconds": rec["intact_tripod_A_loaded"]["wall_seconds"]
            + rec["mirrored_tripod_B_loaded"]["wall_seconds"],
        }
        log("  %-16s pattern: jaccard %.4f (intact %d active, mirrored %d active) "
            "rates %.2f vs %.2f Hz"
            % (arm, jac, rec["intact_tripod_A_loaded"]["n_active_neurons"],
                                       rec["mirrored_tripod_B_loaded"]["n_active_neurons"],
                                       rec["intact_tripod_A_loaded"]["settled_rate_hz"],
                                       rec["mirrored_tripod_B_loaded"]["settled_rate_hz"]),
            flush=True)
    return out


# ---------------------------------------------------------------------------
# stage B2: WHICH homeostatic gain is safe?  (a declared design measurement)
# ---------------------------------------------------------------------------
def stage_homeostasis_modes(built, log):
    """Measure the zero-input behaviour of each homeostatic mode.

    The round's requirement is that homeostasis must not be able to hide a wrong input.
    This stage MEASURES whether each mode satisfies it at ZERO transduced drive, where a
    correct layer must stay silent.  The measurement is the reason the ablation arms use
    ``mode = "intrinsic"``: the synaptic-scaling mode violates the requirement by
    exciting the tier's own bistable regime, and that violation is reported rather than
    removed.
    """
    out = {"requirement": ("with a broken (zero) input and the gain free to reach its "
                           "maximum, the tier must stay below %.1f Hz" % SILENT_HZ),
           "per_mode": {}}
    for mode in ("intrinsic", "synaptic", "both"):
        cfg = HomeostasisConfig(enabled=True, mode=mode, target_rate_hz=40.0)
        row = {}
        for tag, drive in (("zero_drive", 0.0), ("primary_drive", PRIMARY_DRIVE_nA)):
            probe = TierProbe(built, PlasticityConfig(homeostasis=cfg))
            spikes = probe.run(drive, 2000.0)
            s = summarise(spikes, probe.dt_ms, tier=probe.tier, tail_from_ms=1000.0)
            row[tag] = {
                "settled_rate_hz": s["pop_rate_hz_tail_mean"],
                "settled_cv_50ms_bins": s["pop_rate_hz_tail_cv_50ms_bins"],
                "final_gain": float(probe.tier._g_exc),
                "final_gain_synaptic": float(probe.tier._g_syn),
            }
        # and the declared boundary flag, in the intrinsic mode only
        cfg_bg = HomeostasisConfig(enabled=True, mode="intrinsic", target_rate_hz=40.0,
                                   gain_applies_to_background=True)
        probe = TierProbe(built, PlasticityConfig(homeostasis=cfg_bg))
        spikes = probe.run(0.0, 2000.0)
        s = summarise(spikes, probe.dt_ms, tier=probe.tier, tail_from_ms=1000.0)
        row["zero_drive_with_gain_applied_to_background"] = {
            "settled_rate_hz": s["pop_rate_hz_tail_mean"],
            "final_gain": float(probe.tier._g_exc),
        }
        row["satisfies_the_requirement_at_zero_drive"] = bool(
            row["zero_drive"]["settled_rate_hz"] < SILENT_HZ)
        out["per_mode"][mode] = row
        log("  homeostasis mode %-9s zero drive %9.4f Hz (gain->%.3f) | primary drive "
            "%8.2f Hz (gain->%.3f) | requirement met at zero drive: %s"
            % (mode, row["zero_drive"]["settled_rate_hz"], row["zero_drive"]["final_gain"],
               row["primary_drive"]["settled_rate_hz"],
               row["primary_drive"]["final_gain"],
               row["satisfies_the_requirement_at_zero_drive"]), flush=True)
    out["background_boundary"] = {
        "mode": "intrinsic",
        "gain_applies_to_background_false_zero_drive_hz":
            out["per_mode"]["intrinsic"]["zero_drive"]["settled_rate_hz"],
        "gain_applies_to_background_true_zero_drive_hz":
            out["per_mode"]["intrinsic"]["zero_drive_with_gain_applied_to_background"][
                "settled_rate_hz"],
        "conclusion": ("the guarantee that a broken input cannot be hidden is STRUCTURAL "
                       "only because the default does not scale the tier's fixed "
                       "background noise; with the flag on, homeostasis fabricates "
                       "activity from noise alone. The ablation arms use the default."),
    }
    out["ablation_arms_use_mode"] = "intrinsic"
    out["why"] = ("the synaptic-scaling mode was MEASURED to manufacture activity at zero "
                  "input (see per_mode/synaptic/zero_drive), which is the exact failure "
                  "the round's requirement forbids; intrinsic is structurally safe because "
                  "a gain times zero transduced current is zero. Both modes remain "
                  "available and both measurements are reported.")
    return out


# ---------------------------------------------------------------------------
# stage C: the declared lesion + the rewiring barrier test
# ---------------------------------------------------------------------------
def stage_lesion(built, log):
    """Declare the project's own neck cut as a lesion and try to restore it."""
    cut = built["cut"]
    edges = built["edges"]
    data = built["data"]
    rows = np.concatenate([edges["exc_rows"], edges["inh_rows"]])
    severed = np.asarray(cut["row_masks"]["all"])[rows]
    loc = np.asarray(built["local_of_global"])
    pre = loc[np.asarray(data["pre"])[rows]]
    post = loc[np.asarray(data["post"])[rows]]
    barrier = LesionBarrier("declared_neck_cut_ascending_and_descending_side")
    barrier.add(list(zip(pre[severed].tolist(), post[severed].tolist())),
                reason="declared_neck_cut", time_ms=0.0, weight_uS=0.0, channel="mixed")
    log("  declared lesion: %d modelled in-tier edges severed (of %d modelled rows); "
        "baseline nnz %d" % (len(barrier), rows.size,
                             edges["nnz_exc"] + edges["nnz_inh"]), flush=True)

    cfg = PlasticityConfig(rewiring=RewiringConfig(enabled=True))
    tier = PlasticNeuralTier(built, TierConfig(), _Ledger(), plasticity=cfg,
                             lesion=barrier)
    probe = TierProbe.__new__(TierProbe)
    probe.built, probe.arm, probe.cfg, probe.tier = built, "lesion_rewiring", cfg, tier
    probe.groups = [np.asarray(built["sensory"][leg]) for leg in LEGS]
    probe.n = int(built["n"])
    probe.dt_ms = float(TierConfig().dt_ms)
    # MAXIMUM feedback drives the ADD rule; a swing to -1 at the end exercises REMOVE
    probe.run(PRIMARY_DRIVE_nA, 2000.0, feedback=1.0)
    after_add = tier.plasticity_report()
    probe.run(PRIMARY_DRIVE_nA, 1000.0, feedback=-1.0)
    after_remove = tier.plasticity_report()

    We, Wi = tier._We0.tocoo(), tier._Wi0.tocoo()
    keys = set(barrier.reasons)
    present = int(sum(1 for r, c in zip(We.row.tolist(), We.col.tolist())
                      if (c, r) in keys)
                  + sum(1 for r, c in zip(Wi.row.tolist(), Wi.col.tolist())
                        if (c, r) in keys))
    out = {
        "declared_lesion": barrier.as_dict(),
        "n_severed_modelled_edges": int(severed.sum()),
        "n_edges_before_lesion": int(edges["nnz_exc"] + edges["nnz_inh"]),
        "n_edges_after_lesion": int(after_add["n_edges_at_start_after_lesion"]),
        "candidate_set_header": after_add["rewiring"]["candidate_set_header"],
        "after_max_positive_feedback": {
            "added": after_add["rewiring"]["added"],
            "removed": after_add["rewiring"]["removed"],
            "refused_by_lesion_barrier":
                after_add["rewiring"]["n_refused_by_lesion_barrier"],
            "events_by_rule": after_add["rewiring"]["events_by_rule"],
            "accepted_changes_first10": after_add["rewiring"]["accepted_changes_first10"],
        },
        "after_negative_feedback": {
            "added": after_remove["rewiring"]["added"],
            "removed": after_remove["rewiring"]["removed"],
            "refused_by_lesion_barrier":
                after_remove["rewiring"]["n_refused_by_lesion_barrier"],
            "events_by_rule": after_remove["rewiring"]["events_by_rule"],
        },
        "barred_edges_present_after_rewiring": present,
        "restoration_attempts_refused": after_remove["rewiring"][
            "n_refused_by_lesion_barrier"],
        "caps_respected": {
            "add_cap": after_remove["rewiring"]["add_cap_respected"],
            "remove_cap": after_remove["rewiring"]["remove_cap_respected"],
            "edge_cap": after_remove["rewiring"]["edge_cap_respected"],
            "max_added_edges": after_remove["rewiring"]["max_added_edges"],
            "max_removed_edges": after_remove["rewiring"]["max_removed_edges"],
            "per_edge_weight_cap_uS":
                after_remove["rewiring"]["per_edge_weight_cap_uS"],
        },
        "distance_constraint": after_add["rewiring"]["distance_constraint"],
    }
    log("  lesion/rewiring: added %d, removed %d, refused by barrier %d, barred edges "
        "present after rewiring %d"
        % (out["after_negative_feedback"]["added"],
           out["after_negative_feedback"]["removed"],
           out["after_negative_feedback"]["refused_by_lesion_barrier"], present),
        flush=True)
    return out


# ---------------------------------------------------------------------------
# stage D: the closed loop (behavioural readout)
# ---------------------------------------------------------------------------
def stage_loop(built, arms, seeds, seconds, log):
    from engine.embodied import select_gl_backend
    gl = select_gl_backend()
    runs = {}
    for arm in arms:
        for seed in seeds:
            cfg = LoopConfig(arm=LOOP_ARM, seed=seed, duration_s=seconds,
                             gl_backend=None)
            sch = MultirateScheduler(cfg, built_tier=built)
            tier = attach_plasticity(sch, ABLATION_ARMS[arm])
            t0 = time.perf_counter()
            ep = sch.run()
            wall = time.perf_counter() - t0
            hist = {k: np.asarray(v) for k, v in tier.history.items()}
            # THE FROZEN ARM RECORDS NO PLASTICITY HISTORY, and that is by design: it
            # never takes the plastic code path (which is exactly why it is bit-identical
            # to the un-plastic tier).  Its population rate is therefore reported as
            # UNAVAILABLE rather than filled with a plausible-looking number; its
            # behavioural readout (speed and heading) is fully available.
            has_hist = bool(hist["pop_rate_hz"].size)
            if not has_hist:
                n_iv = int(sch.cfg.n_intervals)
                hist = {k: np.zeros(0) for k in tier.history}
                hist["readout_rate_hz"] = np.asarray(ep.observed["descending_rate_hz"],
                                                     float)
                hist["g_syn"] = np.ones(1)
                hist["g_exc"] = np.ones(1)
                hist["homeo_z"] = np.zeros(1)
                hist["stp_mean_x"] = np.ones(1)
                hist["lt_mean_frac"] = np.zeros(1)
                hist["lt_max_abs_frac"] = np.zeros(1)
                hist["n_edges"] = np.full(1, tier.n_edges())
                hist["t_ms"] = np.zeros(n_iv)
            rep = tier.plasticity_report()
            own = sch.check_ownership()
            ep.save_npz(os.path.join(OUT, "plasticity_loop_episode_%s_seed%d.npz"
                                     % (arm, seed)))
            np.savez_compressed(
                os.path.join(OUT, "plasticity_loop_traces_%s_seed%d.npz" % (arm, seed)),
                pop_rate_hz=hist["pop_rate_hz"], readout_rate_hz=hist["readout_rate_hz"],
                g_syn=hist["g_syn"], g_exc=hist["g_exc"], homeo_z=hist["homeo_z"],
                stp_mean_x=hist["stp_mean_x"], lt_mean_frac=hist["lt_mean_frac"],
                lt_max_abs_frac=hist["lt_max_abs_frac"], feedback=hist["feedback"],
                n_edges=hist["n_edges"], t_ms=hist["t_ms"])
            # per-interval summaries so the table is comparable across arms
            if has_hist:
                # the tier's history includes the CALIBRATION advance calls; keep only
                # the episode's own intervals
                nsub = sch.cfg.neural_substeps
                n_ep = int(sch.cfg.n_intervals) * nsub
                ep_series = (hist["pop_rate_hz"][-n_ep:]
                             if n_ep <= len(hist["pop_rate_hz"]) else hist["pop_rate_hz"])
                nblk = len(ep_series) // nsub
                blk = (ep_series[:nblk * nsub].reshape(nblk, nsub).mean(axis=1)
                       if nblk else np.zeros(0))
                hist_note = "per-interval mean of the tier's population firing rate (Hz)"
            else:
                nblk = 0
                blk = np.zeros(0)
                hist_note = ("UNAVAILABLE: the frozen arm never takes the plastic code "
                             "path, so it has no plasticity history to report. Its "
                             "behavioural readout (speed, heading) IS reported, and its "
                             "tier-level firing rates come from the ladder in the same "
                             "report. No number is invented here.")
            lo = nblk // 2

            def bstat(fn, default=float("nan")):
                return float(fn(blk[lo:])) if nblk else default

            runs[(arm, seed)] = {
                "arm": arm, "seed": int(seed), "wall_seconds": wall,
                "metrics": ep.metrics, "ownership": own,
                "reference_rate_hz": (sch.calibration or {}).get(
                    "reference_rates_hz", {}),
                "plasticity_history_available": has_hist,
                "tier_pop_rate_hz_mean": (float(hist["pop_rate_hz"].mean())
                                          if has_hist else float("nan")),
                "tier_pop_rate_hz_second_half": bstat(lambda x: x.mean()),
                "tier_pop_rate_hz_second_half_cv": (
                    float(blk[lo:].std() / blk[lo:].mean())
                    if nblk and blk[lo:].mean() > 0 else float("nan")),
                "fraction_of_time_silent": (bstat(lambda x: (x < SILENT_HZ).mean())
                                            if nblk else float("nan")),
                "fraction_of_time_epileptiform": (
                    bstat(lambda x: (x > EPILEPTIFORM_HZ).mean())
                    if nblk else float("nan")),
                "tier_readout_rate_hz_mean": float(hist["readout_rate_hz"].mean()),
                "g_exc_final": (float(hist["g_exc"][-1]) if has_hist else 1.0),
                "g_syn_final": (float(hist["g_syn"][-1]) if has_hist else 1.0),
                "homeo_z_final": (float(hist["homeo_z"][-1]) if has_hist else 0.0),
                "stp_mean_resource_final": (float(hist["stp_mean_x"][-1]) if has_hist
                                            else 1.0),
                "lt_mean_weight_change_final": (float(hist["lt_mean_frac"][-1])
                                                if has_hist else 0.0),
                "lt_max_abs_weight_change": (float(hist["lt_max_abs_frac"].max())
                                             if has_hist else 0.0),
                "n_edges_start": int(rep["n_edges_at_start_after_lesion"]),
                "n_edges_end": int(rep["n_edges_now"]),
                "n_edges_added": int(rep["rewiring"]["added"]),
                "n_edges_removed": int(rep["rewiring"]["removed"]),
                "n_refused_by_lesion_barrier":
                    int(rep["rewiring"]["n_refused_by_lesion_barrier"]),
                "stp_bound_violations_clipped":
                    int(rep["counters"]["stp_bound_violations_clipped"]),
                "feedback_min": (float(hist["feedback"].min()) if has_hist else 0.0),
                "feedback_max": (float(hist["feedback"].max()) if has_hist else 0.0),
                "plasticity_events_by_rule": rep["rewiring"]["events_by_rule"],
                "binned_series": blk,
                "binned_series_note": hist_note,
            }
            log("  loop %-16s seed %d  speed %6.2f mm/s  heading %7.2f deg  "
                "tier pop %s  wall %5.1fs"
                % (arm, seed, ep.metrics["mean_speed_mm_s"],
                   ep.metrics["heading_change_deg"],
                   ("%7.2f Hz" % hist["pop_rate_hz"].mean()) if has_hist
                   else "  UNAVAIL",
                   wall), flush=True)
            sch.close()
    return {"gl_backend_probe": gl, "seconds": seconds, "seeds": list(seeds),
            "runs": {"%s|%d" % (a, s): v for (a, s), v in runs.items()}}


# ---------------------------------------------------------------------------
# prediction evaluation (mechanical, from the pre-registered criteria)
# ---------------------------------------------------------------------------
def evaluate_predictions(ladder, pattern, loop, arms):
    nonzero = [x for x in DRIVE_LADDER_nA if x > 0.0]
    key_primary = "%g" % PRIMARY_DRIVE_nA
    key_zero = "0"
    # a partial ladder is tolerated so the evaluator can be exercised on a subset
    nonzero = [d for d in nonzero if "%g" % d in ladder[list(ladder)[0]]]
    if key_primary not in ladder[list(ladder)[0]]:
        key_primary = "%g" % nonzero[len(nonzero) // 2]

    def rate(arm, key):
        return ladder[arm][key]["settled_rate_hz"]

    def cv(arm, key):
        return ladder[arm][key]["settled_cv_50ms_bins"]

    # ---- P1 -----------------------------------------------------------------
    latch = {arm: [k for k in ladder[arm] if ladder[arm][k]["kick_latched"]]
             for arm in arms}
    frozen_latch = latch["frozen"]
    latched_rate = {}
    for k in frozen_latch:
        latched_rate[k] = ladder["frozen"][k]["kick_settled_rate_hz"]
    c1a = bool(frozen_latch)
    # clause A needs BOTH: two coexisting attractors AND the active one being
    # epileptiform.  c1b is therefore "the latched state is above the declared threshold"
    # -- the expression now matches the name (an earlier version tested the opposite).
    c1b = bool(latched_rate) and all(v > EPILEPTIFORM_HZ for v in latched_rate.values())
    c1c = bool(SILENT_HZ < rate("frozen", key_primary) < EPILEPTIFORM_HZ
               and cv("frozen", key_primary) <= STABLE_CV_MAX)
    ratios = []
    for a, b in zip(nonzero[:-1], nonzero[1:]):
        ra, rb = rate("frozen", "%g" % a), rate("frozen", "%g" % b)
        ratios.append(rb / ra if ra > 0 else float("inf"))
    c1d = bool(max(ratios) <= JUMP_RATIO) if ratios else False
    clause_a = bool(c1a and c1b)          # "bimodal silent/epileptiform"
    clause_b = not c1c                    # "no stable graded rate at the intermediate"
    if clause_a and clause_b:
        p1v = "reproduced"
    elif (not clause_a) and (not clause_b):
        p1v = "refuted"
    else:
        p1v = "inconclusive"
    P1 = {
        "verdict": p1v,
        "prediction": PREDICTIONS["P1"],
        "clause_A_bimodal_silent_epileptiform_without_homeostasis": clause_a,
        "clause_B_no_stable_graded_rate_at_the_intermediate_drive": clause_b,
        "criterion_1a_a_kick_latches_the_tier": c1a,
        "criterion_1b_the_latched_state_is_epileptiform": c1b,
        "criterion_1c_the_primary_drive_is_stable_and_graded": c1c,
        "criterion_1d_no_adjacent_ladder_jump_above_%.1fx" % JUMP_RATIO: c1d,
        "numbers": {
            "ladder_levels_where_a_30ms_kick_latches_the_frozen_tier": frozen_latch,
            "latched_settled_rate_hz_by_level": latched_rate,
            "un_kicked_rate_hz_at_those_levels":
                {k: ladder["frozen"][k]["settled_rate_hz"] for k in frozen_latch},
            "epileptiform_threshold_hz": EPILEPTIFORM_HZ,
            "primary_drive_nA": PRIMARY_DRIVE_nA,
            "frozen_rate_at_primary_drive_hz": rate("frozen", key_primary),
            "frozen_cv_at_primary_drive": cv("frozen", key_primary),
            "frozen_ladder_rates_hz": {k: rate("frozen", k) for k in
                                       ["%g" % d for d in DRIVE_LADDER_nA]
                                       if k in ladder["frozen"]},
            "max_adjacent_ladder_ratio": float(max(ratios)) if ratios else None,
        },
        "note": ("clause A is about the EXISTENCE of two coexisting attractors AND their "
                 "being silent vs epileptiform; clause B is about the intermediate drive "
                 "lacking a stable graded rate. The two are reported separately so a "
                 "half-true prediction cannot be rounded to either answer."),
    }

    # ---- P2 -----------------------------------------------------------------
    homeo_arms = [a for a in arms
                  if ABLATION_ARMS[a].homeostasis.enabled]
    p2_rows = {}
    for a in homeo_arms:
        r = rate(a, key_primary)
        graded_local = bool(SILENT_HZ < r < EPILEPTIFORM_HZ
                            and cv(a, key_primary) <= STABLE_CV_MAX)
        # a graded MAP around the primary drive: strictly increasing over the two
        # neighbouring decades
        nz = sorted(float(k) for k in ladder[a] if float(k) > 0.0)
        i = min(range(len(nz)), key=lambda j: abs(nz[j] - PRIMARY_DRIVE_nA))
        lo = nz[max(0, i - 1)]
        hi = nz[min(len(nz) - 1, i + 1)]
        rise = bool(rate(a, "%g" % lo) < r < rate(a, "%g" % hi))
        no_latch = not ladder[a][key_primary]["kick_latched"]
        p2_rows[a] = {"rate_hz": r, "cv": cv(a, key_primary),
                      "stable_and_graded_locally": graded_local,
                      "strictly_increasing_across_the_neighbouring_decade": rise,
                      "no_bistable_latch_at_the_primary_drive": no_latch}
    p2_ok = bool(p2_rows) and all(v["stable_and_graded_locally"]
                                  and v["strictly_increasing_across_the_neighbouring_decade"]
                                  and v["no_bistable_latch_at_the_primary_drive"]
                                  for v in p2_rows.values())
    frozen_was_already = bool(SILENT_HZ < rate("frozen", key_primary) < EPILEPTIFORM_HZ
                              and cv("frozen", key_primary) <= STABLE_CV_MAX)
    # declared SECONDARY measurement: does homeostasis remove the low-drive silent
    # attractor that the frozen arm shows?
    frozen_silent_levels = [k for k in ladder["frozen"]
                            if ladder["frozen"][k]["settled_rate_hz"] < SILENT_HZ]
    secondary = {}
    for a in homeo_arms:
        secondary[a] = {k: {"frozen_rate_hz": ladder["frozen"][k]["settled_rate_hz"],
                            "homeo_rate_hz": ladder[a][k]["settled_rate_hz"],
                            "frozen_kick_latched": ladder["frozen"][k]["kick_latched"],
                            "homeo_kick_latched": ladder[a][k]["kick_latched"]}
                        for k in frozen_silent_levels}
    P2 = {
        "verdict": "reproduced" if p2_ok else "refuted",
        "prediction": PREDICTIONS["P2"],
        "homeostasis_arms": homeo_arms,
        "per_arm": p2_rows,
        "frozen_arm_was_ALREADY_stable_and_graded_at_that_drive": frozen_was_already,
        "discriminating_power_note": (
            "P2 is REPRODUCED but, at the pre-registered primary drive, the un-plastic "
            "frozen tier was already stable and graded, so P2 at that drive is not "
            "discriminating. The declared SECONDARY measurement below is where "
            "homeostasis actually changes the regime."
            if frozen_was_already else
            "P2 is discriminating at the primary drive: the frozen tier was NOT stable "
            "and graded there."),
        "secondary_declared_measurement_low_drive_levels_that_are_silent_without_"
        "homeostasis": {"frozen_silent_levels": frozen_silent_levels,
                        "per_homeostasis_arm": secondary},
    }

    # ---- P3 -----------------------------------------------------------------
    rows = {}
    for a in homeo_arms:
        rates = {k: rate(a, k) for k in ["%g" % d for d in DRIVE_LADDER_nA]
                 if k in ladder[a]}
        nz = [rates["%g" % d] for d in nonzero if "%g" % d in rates]
        nonzero_local = [d for d in nonzero if "%g" % d in rates]
        variation = (max(nz) / min(nz)) if min(nz) > 0 else float("inf")
        zero_rate = rates[key_zero]
        target = ABLATION_ARMS[a].homeostasis.target_rate_hz
        # clamp search: longest contiguous run of nonzero levels spanning >= 10x drive
        # whose rate stays within the band, while the frozen arm changes by > 2x
        best = None
        for i in range(len(nonzero_local)):
            for j in range(i + 2, len(nonzero_local)):
                span = nonzero_local[j] / nonzero_local[i]
                if span < P3_CLAMP_SPAN_DECADES:
                    continue
                window = nz[i:j + 1]
                in_band = all(abs(v - target) <= P3_CLAMP_BAND * target for v in window)
                fwin = [rate("frozen", "%g" % d) for d in nonzero_local[i:j + 1]]
                frozen_change = (max(fwin) / min(fwin)) if min(fwin) > 0 else float("inf")
                if in_band and frozen_change > 2.0:
                    if best is None or span > best["drive_span_factor"]:
                        best = {"i": i, "j": j, "drive_span_factor": float(span),
                                "homeo_rates_hz": [float(v) for v in window],
                                "frozen_rates_hz": [float(v) for v in fwin],
                                "frozen_change_factor": float(frozen_change)}
        pat = pattern[a]
        c3a = bool(zero_rate < SILENT_HZ)
        c3b = bool(variation > P3_VARIATION_MIN)
        c3c = bool(pat["input_changed_response"])
        clamp = best is not None
        verdict = "refuted" if ((not c3a) or clamp) else (
            "reproduced" if (c3b and c3c) else "inconclusive")
        rows[a] = {
            "target_rate_hz": target,
            "rate_at_zero_drive_hz": zero_rate,
            "broken_input_stays_silent": c3a,
            "ladder_rate_max_over_min": float(variation),
            "rate_varies_more_than_%.1fx" % P3_VARIATION_MIN: c3b,
            "permuted_input_active_set_jaccard": pat["active_set_jaccard"],
            "permuted_input_changes_the_response": c3c,
            "clamped_to_target_over_a_declared_span": clamp,
            "clamp_evidence": best,
            # SECONDARY, transparent: the same question asked only over drive levels where
            # BOTH this arm and the frozen arm are already active, so the answer does not
            # rest on the frozen arm's silent level
            "active_levels_rate_max_over_min": None,
            "secondary_active_range_sensitivity": {},
            "note": ("the pre-registered clamp criterion trips partly because the frozen "
                     "arm is SILENT at the low end of the window, which makes its ratio "
                     "explode. The secondary block below removes that artefact: it takes "
                     "as its reference the LOWEST nonzero drive at which BOTH this arm "
                     "and the frozen arm are above %.1f Hz, and compares the two arms' "
                     "rate ratios only above that level." % SILENT_HZ),
        }
    # secondary: active-range sensitivity with a FAIR reference level
    for a in homeo_arms:
        nzv = [(d, rate(a, "%g" % d), rate("frozen", "%g" % d)) for d in nonzero
               if "%g" % d in ladder[a]]
        act = [x for x in nzv if x[1] > SILENT_HZ and x[2] > SILENT_HZ]
        if act:
            rows[a]["active_levels_rate_max_over_min"] = float(
                max(x[1] for x in act) / min(x[1] for x in act))
            d0, h0, f0 = act[0]
            rows[a]["secondary_active_range_sensitivity"] = {
                "%g" % d: {
                    "drive_ratio_from_%g" % d0: float(d / d0),
                    "frozen_rate_ratio": float(f / f0),
                    "homeostasis_arm_rate_ratio": float(h / h0),
                    "homeostasis_flattens_the_map_here": bool(h / h0 < f / f0),
                } for (d, h, f) in act[1:]}
            rows[a]["reference_active_level_nA"] = float(d0)
    verdict = ("refuted" if any(v["clamped_to_target_over_a_declared_span"]
                                or not v["broken_input_stays_silent"]
                                for v in rows.values())
               else ("reproduced" if all(v["rate_varies_more_than_%.1fx"
                                          % P3_VARIATION_MIN]
                                         and v["permuted_input_changes_the_response"]
                                         for v in rows.values()) else "inconclusive"))
    P3 = {
        "verdict": verdict,
        "prediction": PREDICTIONS["P3"],
        "criteria": {
            "broken_input_stays_silent": ("with homeostasis ON, zero transduced drive "
                                          "must leave the tier below %.1f Hz" % SILENT_HZ),
            "rate_must_vary": ("the settled rate over the nonzero ladder must vary by "
                               "more than %.1fx" % P3_VARIATION_MIN),
            "input_specificity": ("a spatially permuted input must change the active "
                                  "neuron set (Jaccard < %.2f)"
                                  % P3_PATTERN_JACCARD_MAX),
            "clamp_criterion": ("REFUTED if there is a contiguous run of >=3 nonzero "
                                "ladder levels spanning >=%.0fx drive over which the "
                                "settled rate stays within +/-%.0f%% of the target while "
                                "the frozen arm's rate changes by >2x"
                                % (P3_CLAMP_SPAN_DECADES, P3_CLAMP_BAND * 100)),
        },
        "per_arm": rows,
    }
    return {"P1": P1, "P2": P2, "P3": P3}


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def _smooth(y, n):
    """Centred moving average for DISPLAY only; every reported number uses raw arrays."""
    y = np.asarray(y, float)
    if n <= 1 or y.size < n:
        return y
    pad = n // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(yp, np.ones(n) / n, mode="valid")[:y.size]


def make_figure(ladder, pattern, lesion, loop, preds, path, wall, ram):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(ABLATION_ARMS)
    colors = {"frozen": "#111111", "stp_only": "#1f77b4",
              "homeostasis_only": "#d62728", "long_term_only": "#2ca02c",
              "stp_homeostasis": "#9467bd", "all": "#ff7f0e"}
    nonzero = [d for d in DRIVE_LADDER_nA if d > 0]
    nz_keys = ["%g" % d for d in nonzero]

    fig = plt.figure(figsize=(18.0, 28.0), layout="constrained")
    gs = fig.add_gridspec(6, 2)
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f), (ax_g, ax_gap), (ax_i, ax_j) = \
        [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(5)]
    ax_h = fig.add_subplot(gs[5, :])

    # --- (a) drive -> settled rate -----------------------------------------
    for a in arms:
        y = [ladder[a][k]["settled_rate_hz"] for k in nz_keys]
        ax_a.plot(nonzero, y, "o-", color=colors[a], lw=2.2, ms=6, label=a)
    ax_a.axhspan(0.0, SILENT_HZ, color="grey", alpha=0.20)
    ax_a.axhline(EPILEPTIFORM_HZ, color="crimson", ls="--", lw=1.4)
    ax_a.axvline(PRIMARY_DRIVE_nA, color="darkgreen", ls=":", lw=1.8)
    ax_a.text(2.6, EPILEPTIFORM_HZ - 16, "EPILEPTIFORM > %.0f Hz\n(OPERATIONAL threshold)"
              % EPILEPTIFORM_HZ, fontsize=8.5, color="crimson", ha="right", va="top")
    ax_a.text(0.0006, SILENT_HZ + 6, "SILENT < %.0f Hz" % SILENT_HZ, fontsize=8.5,
              color="dimgrey")
    ax_a.text(PRIMARY_DRIVE_nA * 1.15, 150, "pre-registered\nintermediate drive\n"
              "%.3g nA" % PRIMARY_DRIVE_nA, fontsize=8, color="darkgreen")
    ax_a.set_xscale("log")
    ax_a.set_ylim(-6, 248)
    ax_a.set_xlabel("drive per sensory neuron (nA, model units) -- log scale")
    ax_a.set_ylabel("settled population firing rate (Hz)\nmean over all 12,000 tier "
                    "neurons")
    ax_a.set_title("(a) drive -> settled rate per ablation arm (2.5 s of model time "
                   "each; the 0 nA point is 0.01 Hz for every arm)", fontsize=10)
    ax_a.legend(fontsize=8.5, loc="upper left", title="ablation arm",
                title_fontsize=9, framealpha=0.95)
    ax_a.grid(alpha=0.3, which="both")

    # --- (b) rate vs time at the primary drive -----------------------------
    for a in arms:
        b = np.asarray(ladder[a]["%g" % PRIMARY_DRIVE_nA]["_binned_series_hz"], float)
        if b.size:
            t = (np.arange(b.size) + 0.5) * BIN_MS
            ax_b.plot(t, b, color=colors[a], lw=0.7, alpha=0.30)
            ax_b.plot(t, _smooth(b, 8), color=colors[a], lw=2.4,
                      label="%s (200 ms smoothed)" % a)
    ax_b.set_xlabel("model time (ms, tier clock)")
    ax_b.set_ylabel("population firing rate (Hz), %g ms bins" % BIN_MS)
    ax_b.set_title("(b) rate vs time at the intermediate drive %.3g nA: thin = raw "
                   "%g ms bins,\nthick = 200 ms centred moving average (DISPLAY only)"
                   % (PRIMARY_DRIVE_nA, BIN_MS), fontsize=10)
    ymax_b = max([float(np.max(ladder[a]["%g" % PRIMARY_DRIVE_nA]["_binned_series_hz"]))
                  for a in arms
                  if np.asarray(ladder[a]["%g" % PRIMARY_DRIVE_nA][
                      "_binned_series_hz"]).size] or [100.0])
    ax_b.set_ylim(0, ymax_b * 1.75)
    ax_b.legend(fontsize=8.0, ncol=3, loc="upper center", framealpha=0.95)
    ax_b.grid(alpha=0.3)

    # --- (c) kick / hysteresis: the LATCH MAGNITUDE ------------------------
    for a in arms:
        d = [ladder[a][k]["kick_settled_rate_hz"] - ladder[a][k]["settled_rate_hz"]
             for k in nz_keys]
        ax_c.plot(nonzero, d, "o-", color=colors[a], lw=2.0, ms=5, label=a)
    ax_c.axhline(BISTABLE_MIN_DELTA_HZ, color="crimson", ls="--", lw=1.3)
    ax_c.text(5.0, BISTABLE_MIN_DELTA_HZ * 1.5,
              "latch criterion: the kicked tail must exceed the un-kicked tail by "
              "%.0f Hz (and %.0fx)" % (BISTABLE_MIN_DELTA_HZ, BISTABLE_FACTOR),
              fontsize=8.5, color="crimson", ha="right", va="bottom")
    ax_c.set_xscale("log")
    ax_c.set_yscale("symlog", linthresh=1.0)
    ax_c.set_xlabel("pre-registered drive level (nA per sensory neuron) -- log scale")
    ax_c.set_ylabel("settled rate WITH a %g ms +%g nA kick\nminus WITHOUT it (Hz)"
                    % (KICK_MS, KICK_nA))
    ax_c.set_title("(c) kick / hysteresis. A large positive value means the tier LATCHED "
                   "into a\nself-sustaining state that outlives the transient",
                   fontsize=10)
    ax_c.legend(fontsize=8.5, loc="upper right", title="ablation arm", title_fontsize=9,
                framealpha=0.95)
    ax_c.grid(alpha=0.3, which="both")

    # --- (d) per-neuron rate distribution ----------------------------------
    stats = ["neuron_rate_hz_p50", "neuron_rate_hz_mean", "neuron_rate_hz_p99",
             "neuron_rate_hz_max"]
    names = ["median\nneuron", "mean\nneuron", "99th pct\nneuron", "hottest\nneuron"]
    xw = 0.14
    for i, a in enumerate(arms):
        v = [ladder[a]["%g" % PRIMARY_DRIVE_nA][k] for k in stats]
        xs = np.arange(len(stats)) + (i - 2.5) * xw
        ax_d.bar(xs, v, width=xw, color=colors[a], label=a)
    ax_d.set_xticks(np.arange(len(stats)))
    ax_d.set_xticklabels(names, fontsize=8.5)
    ax_d.set_ylabel("per-neuron firing rate (Hz) at the intermediate drive")
    ax_d.set_title("(d) firing-rate DISTRIBUTION across the 12,000 tier neurons: the "
                   "mean hides a\nheavy-tailed population whose median neuron is far "
                   "quieter than its hottest", fontsize=10)
    ax_d.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.95)
    ax_d.grid(alpha=0.3, axis="y")

    # --- (e) homeostatic gain over time ------------------------------------
    for a in arms:
        g = np.asarray(ladder[a]["%g" % PRIMARY_DRIVE_nA]["_g_exc_series"], float)
        if g.size:
            ax_e.plot(np.arange(g.size) * TierConfig().dt_ms, g, color=colors[a], lw=1.8,
                      label=a)
    hc = ABLATION_ARMS["homeostasis_only"].homeostasis
    ax_e.axhline(hc.gain_max, color="crimson", ls="--", lw=1.2,
                 label="configured gain_max = %.2f" % hc.gain_max)
    ax_e.axhline(hc.gain_min, color="crimson", ls=":", lw=1.2,
                 label="configured gain_min = %.2f" % hc.gain_min)
    ax_e.axhline(hc.target_rate_hz, color="grey", ls="-.", lw=1.0,
                 label="target rate %.0f Hz (a CONFIG value,\nnot a fly firing rate)"
                 % hc.target_rate_hz)
    ax_e.set_xlabel("model time (ms, tier clock)")
    ax_e.set_ylabel("homeostatic excitability gain $g$ (dimensionless)")
    ax_e.set_title("(e) homeostatic gain (arms with homeostasis only).\n"
                   "$g=g_{min}+(g_{max}-g_{min})\\sigma(z)$, hard-bounded by the "
                   "configuration", fontsize=10)
    ax_e.legend(fontsize=8.0, loc="center right", framealpha=0.95)
    ax_e.grid(alpha=0.3)

    # --- (f) long-term weight change ---------------------------------------
    for a in arms:
        lt = np.asarray(ladder[a]["%g" % PRIMARY_DRIVE_nA]["_lt_series"], float)
        if lt.size:
            ax_f.plot(np.arange(lt.size) * TierConfig().dt_ms, lt, color=colors[a], lw=1.8,
                      label=a)
    lt_cfg = ABLATION_ARMS["long_term_only"].long_term
    ax_f.axhline(lt_cfg.w_max_frac - 1.0, color="crimson", ls="--", lw=1.2,
                 label="declared cap  $w_{max}-1$ = +%.0f%%" % ((lt_cfg.w_max_frac - 1)
                                                               * 100))
    ax_f.axhline(1.0 - lt_cfg.w_min_frac, color="crimson", ls=":", lw=1.2,
                 label="declared cap  $1-w_{min}$ = -%.0f%%" % ((1 - lt_cfg.w_min_frac)
                                                                * 100))
    ax_f.set_xlabel("model time (ms, tier clock)")
    ax_f.set_ylabel("largest edge weight change $|W/W_0-1|$\n(dimensionless)")
    ax_f.set_title("(f) long-term weight change (arms with the three-factor rule).\n"
                   "The caps are the declared bounds; they are never exceeded",
                   fontsize=10)
    ax_f.legend(fontsize=8.0, loc="upper left", framealpha=0.95)
    ax_f.grid(alpha=0.3)

    # --- (g) closed-loop behavioural readout -------------------------------
    runs = loop.get("runs", {})
    labels, spd, spd_sd, hdg, hdg_sd, pops = [], [], [], [], [], []
    for a in arms:
        v = [runs["%s|%d" % (a, s)] for s in ABLATION_SEEDS
             if "%s|%d" % (a, s) in runs]
        if not v:
            continue
        labels.append(a)
        spd.append(float(np.mean([r["metrics"]["mean_speed_mm_s"] for r in v])))
        spd_sd.append(float(np.std([r["metrics"]["mean_speed_mm_s"] for r in v])))
        hdg.append(float(np.mean([r["metrics"]["heading_change_deg"] for r in v])))
        hdg_sd.append(float(np.std([r["metrics"]["heading_change_deg"] for r in v])))
        pops.append(float(np.mean([r["tier_pop_rate_hz_mean"] for r in v])))
    xx = np.arange(len(labels))
    ax_g.bar(xx - 0.20, spd, yerr=spd_sd, width=0.36, color="#1f4e9c",
             error_kw=dict(lw=1.0, capsize=3), label="mean thorax speed (mm/s)")
    ax_g.set_xticks(xx)
    ax_g.set_xticklabels(labels, fontsize=9)
    ax_g.set_ylabel("mean thorax speed (mm/s, model units)", color="#1f4e9c")
    ax_g.tick_params(axis="y", labelcolor="#1f4e9c")
    ax_g.grid(alpha=0.3, axis="y")
    ax_g2 = ax_g.twinx()
    ax_g2.bar(xx + 0.20, hdg, yerr=hdg_sd, width=0.36, color="#c0392b",
              error_kw=dict(lw=1.0, capsize=3), label="net heading change (deg)")
    ax_g2.set_ylabel("net heading change (deg)", color="#c0392b")
    ax_g2.tick_params(axis="y", labelcolor="#c0392b")
    h1, l1 = ax_g.get_legend_handles_labels()
    h2, l2 = ax_g2.get_legend_handles_labels()
    ax_g.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper left", framealpha=0.95)
    ax_g.set_title("(g) closed-loop behavioural readout per arm (%g s x seeds %s, the SAME "
                   "loop as round 1).\nFlyGym's demo tripod CPG still generates the gait "
                   "in every arm; the tier only modulates frequency and turn."
                   % (LOOP_SECONDS, list(ABLATION_SEEDS)), fontsize=10)

    # --- right of (g): the loop numbers as text ----------------------------
    ax_gap.axis("off")
    txt = ["(g2) the same readout as numbers: mean +/- sd over %d seeds"
           % len(ABLATION_SEEDS), ""]
    for k, a in enumerate(labels):
        txt.append("%-17s speed %6.2f +/- %.2f mm/s   heading %+7.2f +/- %.2f deg   "
                   "tier pop rate %s"
                   % (a, spd[k], spd_sd[k], hdg[k], hdg_sd[k],
                      ("%.1f Hz" % pops[k]) if np.isfinite(pops[k]) else
                      "UNAVAILABLE (frozen arm records no plasticity history)"))
    txt += ["",
            "The frozen arm is bit-identical to the un-plastic tier, so its speed and",
            "heading are the round-1 numbers, not a new baseline.",
            "",
            "Rewiring and the three-factor rule in the CLOSED LOOP:",
            ]
    for a in arms:
        v = [runs["%s|%d" % (a, s)] for s in ABLATION_SEEDS
             if "%s|%d" % (a, s) in runs]
        if not v:
            continue
        txt.append("%-17s edges added %d, removed %d, final %d of %d   "
                   "weight change max %.4f"
                   % (a, int(np.mean([r["n_edges_added"] for r in v])),
                      int(np.mean([r["n_edges_removed"] for r in v])),
                      int(np.mean([r["n_edges_end"] for r in v])),
                      int(np.mean([r["n_edges_start"] for r in v])),
                      float(np.max([r["lt_max_abs_weight_change"] for r in v]))))
    ax_gap.text(0.0, 0.98, "\n".join(txt), fontsize=8.2, va="top", family="monospace",
                transform=ax_gap.transAxes)

    # --- (i) the lesion barrier --------------------------------------------
    ax_i.axis("off")
    lb = lesion or {}
    cut = lb.get("candidate_set_header", {})
    lines_i = ["(i) THE LESION BARRIER, measured", "",
               "declared lesion: the project's OWN neck-cut row mask",
               "  modelled in-tier edges before the lesion : %s" % lb.get(
                   "n_edges_before_lesion", "n/a"),
               "  severed by the declared cut              : %s" % lb.get(
                   "n_severed_modelled_edges", "n/a"),
               "  edges after the barrier was applied      : %s" % lb.get(
                   "n_edges_after_lesion", "n/a"),
               "",
               "candidate set (declared, seed-fixed, NO distance field exists):",
               "  size %s | barrier-protected bucket %s | already-modelled-edge bucket %s "
               "| uniform bucket %s"
               % (cut.get("pairs_drawn"), cut.get("bucket_barrier_protected"),
                  cut.get("bucket_already_modelled_edges"), cut.get("bucket_uniform_draw")),
               "  barrier-protected pairs available inside the declared pools: %s"
               % cut.get("bucket_barrier_protected_available"),
               "",
               "rewiring driven at MAXIMUM positive feedback:",
               "  edges added                 : %s (cap %s)"
               % (lb.get("after_negative_feedback", {}).get("added"),
                  lb.get("caps_respected", {}).get("max_added_edges")),
               "  edges removed               : %s (cap %s)"
               % (lb.get("after_negative_feedback", {}).get("removed"),
                  lb.get("caps_respected", {}).get("max_removed_edges")),
               "  restoration attempts REFUSED: %s"
               % lb.get("after_negative_feedback", {}).get(
                   "refused_by_lesion_barrier"),
               "  barred edges present after rewiring: %s  <-- must be 0"
               % lb.get("barred_edges_present_after_rewiring"),
               "",
               "No plasticity rule restored a cut edge. A refusal is LOGGED with the time,",
               "the rule, the reason and the pair, so 'it tried and was refused' is",
               "visible in the record rather than inferred."]
    ax_i.text(0.0, 0.99, "\n".join(lines_i), fontsize=8.2, va="top",
              family="monospace", transform=ax_i.transAxes)

    # --- (j) homeostasis mode probe ----------------------------------------
    ax_j.axis("off")
    hp = loop.get("_hmode", {})
    lines_j = ["(j) WHY THE ABLATION USES homeostasis.mode = 'intrinsic'", "",
               "The round's requirement: homeostasis must not be able to hide a wrong",
               "input.  MEASURED at ZERO transduced drive, gain free to reach its max:",
               ""]
    modes = hp.get("per_mode", {})
    for m in ("intrinsic", "synaptic", "both"):
        r = modes.get(m)
        if not r:
            continue
        lines_j.append("  mode %-9s zero drive %9.4f Hz   primary drive %8.2f Hz   "
                       "requirement met: %s"
                       % (m, r["zero_drive"]["settled_rate_hz"],
                          r["primary_drive"]["settled_rate_hz"],
                          r["satisfies_the_requirement_at_zero_drive"]))
    lines_j += ["",
                "  -> the SYNAPTIC-SCALING mode MANUFACTURES activity with no sensation,",
                "     exactly the failure the requirement forbids, so it is REPORTED and",
                "     not used by the ablation arms; 'intrinsic' is structurally safe",
                "     because a gain times zero current is zero.",
                "",
                "(k) the boundary of that guarantee, measured:",
                "  gain_applies_to_background = False (default): %s Hz at zero drive"
                % ("%.4f" % modes.get("intrinsic", {}).get("zero_drive", {}).get(
                    "settled_rate_hz", float("nan")) if modes else "n/a"),
                "  gain_applies_to_background = True          : %s Hz at zero drive"
                % ("%.4f" % modes.get("intrinsic", {}).get(
                    "zero_drive_with_gain_applied_to_background", {}).get(
                        "settled_rate_hz", float("nan")) if modes else "n/a"),
                "",
                "  -> the guarantee is structural but it DEPENDS on the declared default:",
                "     scale the fixed background too and homeostasis invents a response.",
                "",
                "(l) the mirrored-input pattern probe at %.3g nA, active-set Jaccard:"
                % PATTERN_DRIVE_nA]
    for a in arms:
        p = pattern.get(a)
        if p:
            lines_j.append("  %-17s Jaccard %.4f   rate %.2f -> %.2f Hz   input changed "
                           "response: %s"
                           % (a, p["active_set_jaccard"],
                              p["intact_settled_rate_hz"], p["permuted_settled_rate_hz"],
                              p["input_changed_response"]))
    ax_j.text(0.0, 0.99, "\n".join(lines_j), fontsize=8.2, va="top",
              family="monospace", transform=ax_j.transAxes)

    # --- (h) predictions + honesty, full width -----------------------------
    ax_h.axis("off")
    y = 0.995
    ax_h.text(0.0, y, "(h) PRE-REGISTERED PREDICTIONS -> OUTCOMES  (written before the "
              "numbers; none was rescued by re-tuning a gain)", fontsize=13, weight="bold",
              transform=ax_h.transAxes, va="top")
    y -= 0.050
    num = {
        "P1": ("a %g ms +%g nA kick LATCHES the frozen tier at drives %s "
               "(un-kicked %.2f Hz -> kicked %.2f Hz, >%.0fx); the latched state is "
               "%.2f Hz, BELOW the %.0f Hz epileptiform threshold, so the literal "
               "'silent vs epileptiform' bimodality is NOT reproduced; and at the "
               "intermediate drive %.3g nA the frozen tier is stable and graded "
               "(%.2f Hz, tail CV %.3f, adjacent-ladder ratio up to %.0fx over the "
               "silent->active step)"
               % (KICK_MS, KICK_nA,
                  preds["P1"]["numbers"][
                      "ladder_levels_where_a_30ms_kick_latches_the_frozen_tier"],
                  list(preds["P1"]["numbers"]["un_kicked_rate_hz_at_those_levels"]
                       .values())[0] if preds["P1"]["numbers"][
                      "un_kicked_rate_hz_at_those_levels"] else float("nan"),
                  list(preds["P1"]["numbers"]["latched_settled_rate_hz_by_level"]
                       .values())[0] if preds["P1"]["numbers"][
                      "latched_settled_rate_hz_by_level"] else float("nan"),
                  BISTABLE_FACTOR,
                  list(preds["P1"]["numbers"]["latched_settled_rate_hz_by_level"]
                       .values())[0] if preds["P1"]["numbers"][
                      "latched_settled_rate_hz_by_level"] else float("nan"),
                  EPILEPTIFORM_HZ, PRIMARY_DRIVE_nA,
                  preds["P1"]["numbers"]["frozen_rate_at_primary_drive_hz"],
                  preds["P1"]["numbers"]["frozen_cv_at_primary_drive"],
                  preds["P1"]["numbers"]["max_adjacent_ladder_ratio"] or 0.0)),
        "P2": ("all %d homeostasis arms are stable and graded at %.3g nA (%s) with a "
               "strictly increasing map across the neighbouring decade"
               % (len(preds["P2"]["homeostasis_arms"]), PRIMARY_DRIVE_nA,
                  ", ".join("%s %.2f Hz" % (k, v["rate_hz"])
                            for k, v in preds["P2"]["per_arm"].items()))),
        "P3": ("the broken-input test PASSES (0 nA -> %s Hz with homeostasis on) and the "
               "rate varies >%.1fx over the ladder, BUT the declared clamp criterion "
               "TRIPS: %s"
               % (", ".join("%.4f" % v["rate_at_zero_drive_hz"]
                            for v in preds["P3"]["per_arm"].values()), P3_VARIATION_MIN,
                  "; ".join("%s holds %.2f-%.2f Hz within +/-%.0f%% of its %.0f Hz target "
                            "over a %.0fx drive span"
                            % (k, min(v["clamp_evidence"]["homeo_rates_hz"]),
                               max(v["clamp_evidence"]["homeo_rates_hz"]),
                               P3_CLAMP_BAND * 100, v["target_rate_hz"],
                               v["clamp_evidence"]["drive_span_factor"])
                            for k, v in preds["P3"]["per_arm"].items()
                            if v["clamp_evidence"]))),
    }
    for k in ("P1", "P2", "P3"):
        v = preds[k]["verdict"]
        ax_h.text(0.0, y, "%s  ->  %s" % (k, v.upper()), fontsize=12, weight="bold",
                  transform=ax_h.transAxes, va="top",
                  color={"reproduced": "darkgreen", "refuted": "crimson",
                         "inconclusive": "darkorange"}[v])
        y -= 0.030
        words = ("PREDICTION: " + PREDICTIONS[k] + "   ||   OUTCOME: " + num[k]).split()
        cur = ""
        for wd in words:
            if len(cur) + len(wd) + 1 > 175:
                ax_h.text(0.0, y, cur, fontsize=8.6, transform=ax_h.transAxes, va="top")
                y -= 0.0205
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        ax_h.text(0.0, y, cur, fontsize=8.6, transform=ax_h.transAxes, va="top")
        y -= 0.030
    y -= 0.010
    for line in [
        "HONESTY (short form here; the FULL block is printed at a legible size on",
        "outputs/embodied_body/plasticity_honesty.png and in plasticity_report.json):",
        "  * every plasticity parameter is ILLUSTRATIVE; this project has NO measured",
        "    adult-fly central plasticity timescale tau_p, so nothing here says how long",
        "    a trained mapping stays valid;",
        "  * the connectome edges acted on are UNMEASURED and UNSIGNED: nt_pair = -1 on",
        "    every cached row and the cache is not a >=5-synapse graph (measured minimum",
        "    3.0 synapses, 50.3% of rows below 5), so plasticity starts from an unmeasured",
        "    baseline and AMPLIFIES its uncertainty rather than reducing it;",
        "  * the three-factor feedback is an ENGINEERING REWARD PROXY computed from",
        "    observable per-leg drives, never a dopamine concentration; this project has",
        "    no dopaminergic neuron, receptor, release model or measurement at all;",
        "  * rewiring has NO distance constraint because the tier has no coordinates;",
        "  * the lesion barrier is a software invariant, not a model of injury repair;",
        "  * no consciousness, viability or survival claim is made anywhere in this round.",
    ]:
        ax_h.text(0.0, y, line, fontsize=8.8, transform=ax_h.transAxes, va="top")
        y -= 0.023
    y -= 0.010
    ax_h.text(0.0, y,
              "wall clock %.2f min for the whole study (one process, one pass) | peak RAM "
              "%.0f MB | tier: 12,000 neurons, %s modelled edges"
              % (wall / 60.0, ram, lesion.get("n_edges_before_lesion", "n/a")),
              fontsize=9.0, transform=ax_h.transAxes, va="top", color="#111111")

    fig.suptitle("SWITCHABLE PLASTICITY ON THE NEURAL TIER -- A PRE-REGISTERED ABLATION "
                 "(the frozen baseline is unchanged and still available)\n"
                 "frozen | stp_only | homeostasis_only | long_term_only | "
                 "stp_homeostasis | all   -- the arms differ ONLY in which mechanisms are "
                 "enabled", fontsize=13)
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path


def make_honesty_figure(path, preds, lesion, hmode, wall, ram, built):
    """The FULL honesty block and the prediction outcomes at a legible size.

    The round requires the honesty rules restated on the figure.  They do not fit legibly
    inside a technical panel, so they get their own sheet rather than a 5-point footnote.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from engine.embodied.plasticity import (PLASTICITY_PARAMETER_PROVENANCE,
                                            plasticity_honesty)

    fig = plt.figure(figsize=(16.0, 22.0), layout="constrained")
    gs = fig.add_gridspec(3, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    for ax in (ax1, ax2, ax3):
        ax.axis("off")

    def wrap(text, width):
        out, cur = [], ""
        for wd in text.split():
            if len(cur) + len(wd) + 1 > width:
                out.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        out.append(cur)
        return out

    y = 0.99
    ax1.text(0.0, y, "HONESTY RULES -- every one of them unremovable", fontsize=22,
             weight="bold", va="top", color="crimson", transform=ax1.transAxes)
    y -= 0.045
    for k, line in enumerate(plasticity_honesty(), 1):
        rows = wrap(line, 128)
        ax1.text(0.0, y, "%d." % k, fontsize=13.5, va="top", transform=ax1.transAxes,
                 weight="bold")
        for r in rows:
            ax1.text(0.028, y, r, fontsize=13.5, va="top", transform=ax1.transAxes)
            y -= 0.0285
        y -= 0.014

    y = 0.985
    ax2.text(0.0, y, "PRE-REGISTERED PREDICTIONS -> OUTCOMES (numbers as measured)",
             fontsize=20, weight="bold", va="top", transform=ax2.transAxes)
    y -= 0.05
    for k in ("P1", "P2", "P3"):
        v = preds[k]["verdict"]
        ax2.text(0.0, y, "%s  ->  %s" % (k, v.upper()), fontsize=18, weight="bold",
                 va="top", transform=ax2.transAxes,
                 color={"reproduced": "darkgreen", "refuted": "crimson",
                        "inconclusive": "darkorange"}[v])
        y -= 0.045
        for r in wrap("PREDICTION: " + PREDICTIONS[k], 120):
            ax2.text(0.02, y, r, fontsize=12.5, va="top", transform=ax2.transAxes)
            y -= 0.027
        _p3 = preds["P3"]["per_arm"]
        _ho = _p3.get("homeostasis_only", {})
        _sh = _p3.get("stp_homeostasis", {})
        _al = _p3.get("all", {})
        def _ratio(arm_rows, drive_key):
            sec = arm_rows.get("secondary_active_range_sensitivity", {})
            return sec.get(drive_key, {}).get("homeostasis_arm_rate_ratio", float("nan"))
        def _fro(arm_rows, drive_key):
            sec = arm_rows.get("secondary_active_range_sensitivity", {})
            return sec.get(drive_key, {}).get("frozen_rate_ratio", float("nan"))
        p3_txt = ("the broken-input test PASSES (0 nA -> %s Hz with homeostasis on) and the "
                  "rate does vary over the ladder, BUT the pre-registered clamp criterion "
                  "TRIPS: %s. So the honest verdict is REFUTED. The SECONDARY block "
                  "(active drive levels only, so the frozen arm's silent level cannot "
                  "inflate its ratio) shows the flattening is SPECIFIC to "
                  "homeostasis_only: at 10x above the first level where both arms are "
                  "active, its rate ratio is %.2f against the frozen tier's %.2f, whereas "
                  "stp_homeostasis and 'all' show a LARGER ratio than frozen (%.2f and "
                  "%.2f) because STP caps their ceiling instead of flattening the map. A "
                  "global bounded gain therefore behaves as a PARTIAL SERVO on this tier, "
                  "and presenting it as 'not a clamp' would not be honest."
                  % (", ".join("%.4f" % v["rate_at_zero_drive_hz"]
                               for v in _p3.values()),
                     "; ".join("%s holds %.2f-%.2f Hz inside +/-%.0f%% of its %.0f Hz "
                               "target over a %.1fx drive span while the frozen arm goes "
                               "%.2f-%.2f Hz"
                               % (k, min(v["clamp_evidence"]["homeo_rates_hz"]),
                                  max(v["clamp_evidence"]["homeo_rates_hz"]),
                                  P3_CLAMP_BAND * 100, v["target_rate_hz"],
                                  v["clamp_evidence"]["drive_span_factor"],
                                  min(v["clamp_evidence"]["frozen_rates_hz"]),
                                  max(v["clamp_evidence"]["frozen_rates_hz"]))
                               for k, v in _p3.items() if v["clamp_evidence"]),
                     _ratio(_ho, "0.05"), _fro(_ho, "0.05"),
                     _ratio(_sh, "0.05"), _ratio(_al, "0.05")))

        meas = {
            "P1": ("a %g ms +%g nA kick LATCHES the frozen tier at drives %s "
                   "(un-kicked %.3f Hz -> kicked %.3f Hz); the latched state is %.2f Hz, "
                   "below the %.0f Hz operational epileptiform threshold, so the literal "
                   "'silent vs epileptiform' bimodality is NOT reproduced; and at the "
                   "intermediate drive %.3g nA the frozen tier IS stable and graded "
                   "(%.2f Hz, tail CV %.4f). A genuine silent/self-sustaining bistability "
                   "IS reproduced, in a weaker form, and is reported as such."
                   % (KICK_MS, KICK_nA,
                      preds["P1"]["numbers"][
                          "ladder_levels_where_a_30ms_kick_latches_the_frozen_tier"],
                      list(preds["P1"]["numbers"][
                          "un_kicked_rate_hz_at_those_levels"].values())[0],
                      list(preds["P1"]["numbers"][
                          "latched_settled_rate_hz_by_level"].values())[0],
                      list(preds["P1"]["numbers"][
                          "latched_settled_rate_hz_by_level"].values())[0],
                      EPILEPTIFORM_HZ, PRIMARY_DRIVE_nA,
                      preds["P1"]["numbers"]["frozen_rate_at_primary_drive_hz"],
                      preds["P1"]["numbers"]["frozen_cv_at_primary_drive"])),
            "P2": ("all %d homeostasis arms are stable and graded at %.3g nA (%s), strictly "
                   "increasing across the neighbouring decade. Non-discriminating at that "
                   "drive because the un-plastic tier was already graded there, so the "
                   "declared SECONDARY measurement is what matters: at 0.0016 nA the "
                   "frozen tier is silent (0.013 Hz) while homeostasis_only settles at "
                   "%.2f Hz and no longer latches."
                   % (len(preds["P2"]["homeostasis_arms"]), PRIMARY_DRIVE_nA,
                      ", ".join("%s %.2f Hz" % (k, v["rate_hz"])
                                for k, v in preds["P2"]["per_arm"].items()),
                      preds["P2"]["secondary_declared_measurement_low_drive_levels_that_"
                      "are_silent_without_homeostasis"]["per_homeostasis_arm"][
                          "homeostasis_only"]["0.0016"]["homeo_rate_hz"])),
            "P3": p3_txt,
        }[k]
        for r in wrap("OUTCOME: " + meas, 120):
            ax2.text(0.02, y, r, fontsize=12.5, va="top", transform=ax2.transAxes)
            y -= 0.027
        y -= 0.025

    y = 0.985
    ax3.text(0.0, y, "PARAMETER PROVENANCE, COST, AND WHAT WAS NOT TOUCHED", fontsize=20,
             weight="bold", va="top", transform=ax3.transAxes)
    y -= 0.05
    block = [
        "STATUS: %s" % PLASTICITY_PARAMETER_PROVENANCE["status"],
        "",
        "MEASURED-ABSENT (no number in this round is a measurement of these):",
    ]
    block += ["   - " + x for x in PLASTICITY_PARAMETER_PROVENANCE["measured_absent"]]
    block += ["", "BOUNDED, NOT MEASURED (declared software bounds):"]
    block += ["   - " + x for x in PLASTICITY_PARAMETER_PROVENANCE["bound_not_measured"]]
    block += [
        "",
        "WALL CLOCK %.2f min for the whole study (one process, one pass); peak RAM "
        "%.0f MB." % (wall / 60.0, ram),
        "",
        "SOURCE FILES THAT WERE NOT MODIFIED (the frozen baseline is untouched):",
        "   engine/embodied/loop.py, engine/embodied/adapters.py,",
        "   engine/neural_cond.py, engine/neural_hist.py",
        "NEW FILES (the whole of this round's code):",
        "   engine/embodied/plasticity.py, run_plasticity_ablation.py,",
        "   run_plasticity_selftest.py",
        "",
        "TIER: %d neurons selected from a cached BANC snapshot, %d modelled edges, "
        "readout %d neurons."
        % (built["n"], built["edges"]["nnz_exc"] + built["edges"]["nnz_inh"],
           built["readout_local"].size),
        "LESION BARRIER, as measured: %s modelled in-tier edges severed by the project's "
        "own neck cut;"
        % lesion.get("n_severed_modelled_edges", "n/a"),
        "   %s restoration attempts refused by the barrier, %s barred edges present after "
        "rewiring."
        % (lesion.get("after_negative_feedback", {}).get("refused_by_lesion_barrier"),
           lesion.get("barred_edges_present_after_rewiring")),
        "HOMEOSTATIC MODE: the ablation uses 'intrinsic'. The 'synaptic' mode was measured "
        "to manufacture",
        "   activity at zero input (%s Hz) and is therefore reported, not used."
        % ("%.4f" % hmode.get("per_mode", {}).get("synaptic", {}).get(
            "zero_drive", {}).get("settled_rate_hz", float("nan")) if hmode else "n/a"),
    ]
    for r in block:
        for rr in wrap(r, 128):
            ax3.text(0.0, y, rr, fontsize=12.0, va="top", transform=ax3.transAxes,
                     family="monospace" if r.startswith("   ") else None)
            y -= 0.026

    fig.suptitle("SWITCHABLE PLASTICITY ON THE NEURAL TIER -- HONESTY, PREDICTIONS AND "
                 "PROVENANCE\n(every parameter is ILLUSTRATIVE; no consciousness, "
                 "viability or survival claim is made)", fontsize=17)
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="tier,lesion,loop")
    ap.add_argument("--seconds", type=float, default=LOOP_SECONDS)
    ap.add_argument("--seeds", default=",".join(str(s) for s in ABLATION_SEEDS))
    ap.add_argument("--cache", default=os.path.join(OUT, "plasticity_stage_cache.pkl"))
    ap.add_argument("--reuse", action="store_true",
                    help="reuse stages already present in --cache instead of re-running")
    ap.add_argument("--refresh", default="",
                    help="comma-separated cached stage keys to invalidate: "
                         "tier,lesion,loop")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip() != "")
    arms = list(ABLATION_ARMS)

    lines = []

    def log(msg, **kw):
        print(msg, flush=True)
        lines.append(str(msg))

    log("=" * 100)
    log("PRE-REGISTERED SWITCHABLE-PLASTICITY ABLATION")
    log(PRE_REGISTRATION_DISCLOSURE)
    log("arms: %s" % (arms,))
    log("drive ladder (nA/sensory neuron): %s  intermediate (geometric centre): %g"
        % (list(DRIVE_LADDER_nA), PRIMARY_DRIVE_nA))
    log("thresholds: SILENT<%.1f Hz, EPILEPTIFORM>%.1f Hz, STABLE_CV<=%.2f, "
        "JUMP_RATIO<=%.1fx" % (SILENT_HZ, EPILEPTIFORM_HZ, STABLE_CV_MAX, JUMP_RATIO))
    log("=" * 100)

    t_build = time.perf_counter()
    built = build_neural_tier(TierConfig())
    log("tier built in %.1fs: %d neurons, %d modelled edges (exc %d / inh %d), readout %d"
        % (time.perf_counter() - t_build, built["n"],
           built["edges"]["nnz_exc"] + built["edges"]["nnz_inh"],
           built["edges"]["nnz_exc"], built["edges"]["nnz_inh"],
           built["readout_local"].size))

    ladder, pattern, lesion, loop, hmode = {}, {}, {}, {}, {}
    cache = {}
    if args.reuse and os.path.isfile(args.cache):
        with open(args.cache, "rb") as fh:
            cache = pickle.load(fh)
        for rk in [x.strip() for x in args.refresh.split(",") if x.strip()]:
            rk = "tier_stage" if rk == "tier" else ("loop" if rk == "loop" else
                                                    ("lesion" if rk == "lesion" else rk))
            if rk in cache:
                del cache[rk]
                log("stage cache entry %r invalidated for this run" % rk)
        log("stage cache loaded from %s (has: %s)" % (args.cache, sorted(cache)))
        for k, v in cache.items():
            if k == "tier_stage":
                ladder, pattern, hmode = v["ladder"], v["pattern"], v["hmode"]
            elif k == "lesion":
                lesion = v
            elif k == "loop":
                loop = v

    def save_cache(key, value):
        cache[key] = value
        # carry the FULL study's wall clock across a --reuse pass, so a figure-only
        # regeneration cannot silently shrink the reported cost of the study
        cache["_wall_seconds_so_far"] = max(
            float(cache.get("_wall_seconds_so_far", 0.0)), time.perf_counter() - T0)
        with open(args.cache, "wb") as fh:
            pickle.dump(cache, fh)

    def guarded(name, fn):
        try:
            return fn(), True
        except Exception:                                                # noqa: BLE001
            log("STAGE %s FAILED:\n%s" % (name, traceback.format_exc()))
            return None, False

    if "tier" in stages and not ladder:
        log("\n--- STAGE A: drive ladder + kick/hysteresis probe ---")
        ladder = stage_ladder(built, arms, DRIVE_LADDER_nA, log)
        log("\n--- STAGE B: mirrored-input pattern probe at %g nA ---"
            % PRIMARY_DRIVE_nA)
        pattern = stage_pattern(built, arms, PRIMARY_DRIVE_nA, log)
        log("\n--- STAGE B2: homeostatic-mode safety probe (declared design measurement)"
            " ---")
        hmode = stage_homeostasis_modes(built, log)
        save_cache("tier_stage", {"ladder": ladder, "pattern": pattern, "hmode": hmode})
    if "lesion" in stages and not lesion:
        log("\n--- STAGE C: declared neck-cut lesion + rewiring barrier ---")
        lesion, ok = guarded("lesion", lambda: stage_lesion(built, log))
        lesion = lesion or {}
        if ok:
            save_cache("lesion", lesion)
    if "loop" in stages and not loop:
        log("\n--- STAGE D: closed loop, %g s x %d seeds x %d arms ---"
            % (args.seconds, len(seeds), len(arms)))
        loop, ok = guarded("loop", lambda: stage_loop(built, arms, seeds, args.seconds,
                                                      log))
        loop = loop or {}
        if ok:
            save_cache("loop", loop)

    preds = {}
    if ladder and pattern:
        preds = evaluate_predictions(ladder, pattern, loop, arms)
        log("\n" + "=" * 100)
        for k in ("P1", "P2", "P3"):
            log("%s: %s" % (k, preds[k]["verdict"].upper()))
            log("   %s" % PREDICTIONS[k])
        log("=" * 100)

    # ---- outputs -----------------------------------------------------------
    def strip(d):
        """JSON-safe copy: arrays become short summaries."""
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if not k.startswith("_")}
        if isinstance(d, (list, tuple)):
            return [strip(v) for v in d]
        if isinstance(d, np.ndarray):
            return {"array_shape": list(d.shape), "finite": bool(np.all(np.isfinite(d)))
                    if d.dtype.kind in "fc" else True}
        if isinstance(d, (np.floating, np.integer, np.bool_)):
            d = d.item()
        if isinstance(d, float) and not math.isfinite(d):
            return None
        return d

    npz = {}
    for a in arms:
        for d in DRIVE_LADDER_nA:
            r = ladder.get(a, {}).get("%g" % d)
            if not r:
                continue
            for k in ("_pop_series_hz", "_binned_series_hz", "_lt_series",
                      "_g_exc_series", "_g_syn_series", "_feedback_series"):
                v = r.get(k)
                if v is not None and np.asarray(v).size:
                    npz["%s/drive_%g/%s" % (a, d, k.lstrip("_"))] = np.asarray(v)
    if loop:
        for key, r in loop["runs"].items():
            npz["loop/%s/pop_rate_hz_per_interval" % key] = r["binned_series"]
    if npz:
        p = os.path.join(OUT, "plasticity_ablation_traces.npz")
        np.savez_compressed(p, **npz)
        log("traces: %s (%d arrays)" % (p, len(npz)))

    fig_path = os.path.join(OUT, "plasticity_ablation.png")
    fig_ok = False
    if ladder and pattern:
        try:
            make_figure(ladder, pattern,
                        lesion or {"n_edges_before_lesion":
                                   built["edges"]["nnz_exc"]
                                   + built["edges"]["nnz_inh"]},
                        dict(loop or {"runs": {}}, _hmode=hmode), preds, fig_path,
                        time.perf_counter() - T0, peak_ram_mb())
            fig_ok = True
            log("figure: %s" % fig_path)
        except Exception:                                                # noqa: BLE001
            log("FIGURE FAILED:\n%s" % traceback.format_exc())
    hon_path = os.path.join(OUT, "plasticity_honesty.png")
    hon_ok = False
    if fig_ok and preds:
        try:
            make_honesty_figure(hon_path, preds, lesion or {}, hmode,
                                max(float(cache.get("_wall_seconds_so_far", 0.0)),
                                    time.perf_counter() - T0), peak_ram_mb(), built)
            hon_ok = True
            log("honesty figure: %s" % hon_path)
        except Exception:                                                # noqa: BLE001
            log("HONESTY FIGURE FAILED:\n%s" % traceback.format_exc())

    tests_path = os.path.join(OUT, "plasticity_selftest.json")
    tests = None
    if os.path.isfile(tests_path):
        with open(tests_path) as fh:
            tests = json.load(fh)

    report = {
        "round": ("embodied round 3: a switchable plasticity layer on the neural tier, "
                  "with a pre-registered ablation"),
        "pre_registration": PRE_REGISTRATION_DISCLOSURE,
        "predictions": PREDICTIONS,
        "prediction_outcomes": preds,
        "arms": {a: ABLATION_ARMS[a].as_dict() for a in arms},
        "arm_note": ("the arms differ ONLY in which mechanisms are enabled; every other "
                     "parameter, the seed, the initial tier state and the drive schedule "
                     "are identical, and the frozen arm calls the parent tier's own "
                     "advance() verbatim"),
        "frozen_baseline": {
            "status": "UNCHANGED and available: the frozen arm is the pre-existing "
                      "un-plastic tier. With every switch off PlasticNeuralTier.advance "
                      "calls NeuralTier.advance itself, and the identity is tested "
                      "bit-for-bit (both at tier level and through a closed-loop "
                      "episode) in run_plasticity_selftest.py.",
            "source_files_unmodified": ["engine/embodied/loop.py",
                                        "engine/embodied/adapters.py",
                                        "engine/neural_cond.py",
                                        "engine/neural_hist.py"],
        },
        "run_parameters": {
            "drive_ladder_nA": list(DRIVE_LADDER_nA),
            "primary_intermediate_drive_nA": PRIMARY_DRIVE_nA,
            "loop_reachable_max_nA": LOOP_REACHABLE_MAX_nA,
            "loop_reachable_mid_nA": LOOP_REACHABLE_MID_nA,
            "settle_ms": SETTLE_MS, "kick_settle_ms": KICK_SETTLE_MS,
            "kick_ms": KICK_MS, "kick_nA": KICK_nA, "kick_tail_ms": KICK_TAIL_MS,
            "pattern_ms": PATTERN_MS,
            "silent_hz": SILENT_HZ, "epileptiform_hz": EPILEPTIFORM_HZ,
            "epileptiform_threshold_note": (
                "OPERATIONAL, declared before the run: the lower decade edge of the "
                "245-320 Hz range this project reported for the earlier LIF closed loop. "
                "It is not a physiological threshold and no physiology is claimed for it."),
            "stable_cv_max": STABLE_CV_MAX, "jump_ratio": JUMP_RATIO,
            "bistable_factor": BISTABLE_FACTOR,
            "bistable_min_delta_hz": BISTABLE_MIN_DELTA_HZ,
            "p3_variation_min": P3_VARIATION_MIN, "p3_clamp_band": P3_CLAMP_BAND,
            "p3_clamp_span": P3_CLAMP_SPAN_DECADES,
            "p3_pattern_jaccard_max": P3_PATTERN_JACCARD_MAX,
            "ablation_seeds": list(seeds), "loop_seconds": args.seconds,
            "bin_ms": BIN_MS,
        },
        "tier": {"n_neurons": int(built["n"]),
                 "n_edges": int(built["edges"]["nnz_exc"] + built["edges"]["nnz_inh"]),
                 "readout_neurons": int(built["readout_local"].size),
                 "coverage_digest": built["coverage"]["tier"]["tier_root_id_digest_sha256_32"],
                 "nt_pair_all_minus_one": built["coverage"]["nt_pair_all_minus_one"]},
        "ladder": strip(ladder),
        "pattern_probe": strip(pattern),
        "homeostasis_mode_probe": strip(hmode),
        "lesion_barrier_experiment": strip(lesion),
        "loop": strip(loop),
        "honesty": PLASTICITY_HONESTY + list(HONESTY),
        "limitations_left_in_place": [
            "the tier is a bounded SELECTION of an ANNOTATION (12,000 of 153,962 "
            "neurons) with UNMEASURED and UNSIGNED edges, thresholded by the cache's own "
            "rule, so plasticity here acts on an unmeasured baseline and AMPLIFIES its "
            "uncertainty rather than reducing it",
            "every plasticity parameter is ILLUSTRATIVE and no adult-fly central "
            "plasticity timescale tau_p exists in this project: nothing here says how "
            "long a trained mapping would stay valid",
            "homeostasis is a bounded GLOBAL scalar gain: it cannot be a spatial, "
            "cell-type-specific or dendritic homeostatic system, and its target rate is a "
            "software parameter with no claim about normal adult-fly firing rates",
            "the homeostatic mode used by the ablation arms is 'intrinsic' BECAUSE the "
            "synaptic-scaling mode was measured to manufacture activity at zero input "
            "(see homeostasis_mode_probe); the violating measurement is reported in full, "
            "and the safe mode's guarantee is structural rather than empirical but still "
            "depends on the declared default that the gain does not scale the fixed "
            "background noise",
            "the three-factor feedback is an ENGINEERING REWARD PROXY computed from "
            "observable per-leg drives; there is no neuromodulator model of any kind",
            "structural rewiring has NO distance constraint because the tier has no "
            "coordinates; it also has no rule that could ever be described as "
            "activity-dependent growth in space",
            "the lesion barrier is a software invariant, not a model of injury repair",
            "the closed-loop behavioural readout still has FlyGym's demo tripod CPG "
            "generating the gait in every arm; the tier only modulates frequency and turn",
            "one body, flat ground, one spawn state, 3 seeds, 8 s per arm: no "
            "statistical claim beyond those runs is made",
            "P1's EPILEPTIFORM threshold is an operational number, and the honest reading "
            "of P1 depends on it: the report gives the measured rates so a reader can "
            "apply a different threshold",
        ],
        "sources_that_were_NOT_modified": [
            "engine/embodied/loop.py", "engine/embodied/adapters.py",
            "engine/neural_cond.py", "engine/neural_hist.py",
        ],
        "wall_seconds_total": max(float(cache.get("_wall_seconds_so_far", 0.0)),
                                  time.perf_counter() - T0),
        "wall_seconds_this_process": time.perf_counter() - T0,
        "wall_note": ("wall_seconds_total is the whole study, carried across a --reuse "
                      "pass through the stage cache; wall_seconds_this_process is this "
                      "invocation only"),
        "wall_hours_total": max(float(cache.get("_wall_seconds_so_far", 0.0)),
                                time.perf_counter() - T0) / 3600.0,
        "peak_ram_mb": peak_ram_mb(),
        "figure": fig_path if fig_ok else "FAILED: see log",
        "figure_written": bool(fig_ok),
        "honesty_figure": hon_path if hon_ok else "FAILED: see log",
        "honesty_figure_written": bool(hon_ok),
        "selftest": ({"n_checks": tests.get("n_checks"),
                      "n_passed": tests.get("n_passed"),
                      "passed": tests.get("passed")} if tests else
                     "not run yet: run run_plasticity_selftest.py"),
        "log": lines,
    }
    rep_path = os.path.join(OUT, "plasticity_report.json")
    with open(rep_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    log("\nreport: %s" % rep_path)
    log("wall %.2f min, peak RAM %.0f MB" % (report["wall_seconds_total"] / 60.0,
                                             report["peak_ram_mb"]))
    for k in ("P1", "P2", "P3"):
        if preds:
            log("%s -> %s" % (k, preds[k]["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
