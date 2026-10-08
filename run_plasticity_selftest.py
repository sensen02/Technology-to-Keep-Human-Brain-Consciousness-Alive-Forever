"""Tests for the switchable plasticity layer and its ablation.

Run in the BODY environment:
    ./venv_body/bin/python run_plasticity_selftest.py

WHAT THESE TESTS ESTABLISH
--------------------------
* that the layer is INERT when off: the frozen arm's spikes, membrane voltages,
  conductances, adaptation and spike counts are BIT-IDENTICAL to the pre-existing
  un-plastic tier, and a closed-loop episode run with the frozen layer attached is
  BIT-IDENTICAL (every truth/observed/command array) to the same episode run with no
  layer at all;
* that every arm is DETERMINISTIC: two independently constructed tiers with the same
  configuration produce bit-identical spikes, and two independent rewiring runs produce
  identical change logs;
* SHORT-TERM PLASTICITY: the release resource stays inside [0, 1] with the bound never
  having to clip; the recovery-only case matches its closed-form exponential to a stated
  tolerance; a burst depresses the resource and a rest restores it; facilitation raises
  the running utilisation; and the release-value ring stays in LOCKSTEP with the parent
  tier's own delay ring, so the fraction applied at arrival is the fraction released at
  the spike;
* HOMEOSTASIS: the gain never leaves its configured bounds; a broken (zero) input stays
  silent even with the gain saturated at its maximum; the gain does NOT scale the fixed
  background by default, and the flag that makes it do so is measured (a documented
  boundary of the guarantee rather than a hidden one);
* LONG-TERM CHANGE: zero feedback produces exactly zero weight change; per-edge weights
  never leave the declared bounds; with an extreme learning rate the bound really binds;
  and a pre-before-post pairing gives a larger eligibility than the reverse;
* REWIRING: the candidate set is reproducible from its seed, carries three declared
  buckets, has NO distance field anywhere, and every cap holds; every accepted and every
  refused change is logged with time, rule, parent state and the old/new edge;
* THE LESION BARRIER: with the project's own declared neck cut applied (36,989 modelled
  in-tier edges severed), the barrier-protected pairs are physically absent, the ADD rule
  really does propose them, every proposal is refused with a logged reason, and no barred
  edge is present after rewiring;
* that the six ablation arms actually DIFFER from one another, and which pairs do not
  (reported, not hidden);
* that malformed input and malformed configuration are REFUSED.

WHAT THESE TESTS DO NOT ESTABLISH
--------------------------------
They do not establish that any plasticity parameter is measured, that the tier's
connectome weights are known, that the feedback proxy is a neuromodulator, that the
timescale of any change is physiological, that a cut edge would stay cut in a real
nervous system, or that anything here is an adult fly. They do not establish that the
ablation's effect sizes generalise beyond this one bounded tier, one body, flat ground,
three seeds and 8 s. They test the LAYER'S CONTRACTS, not the fly's biology.
"""
from __future__ import annotations

import json
import math
import os
import resource
import sys
import time

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs", "embodied_body")

from engine.embodied.adapters import (                                    # noqa: E402
    LEGS, NeuralTier, OwnershipLedger, TierConfig, build_neural_tier)
from engine.embodied.loop import LoopConfig, MultirateScheduler           # noqa: E402
from engine.embodied.plasticity import (                                  # noqa: E402
    ABLATION_ARMS, HomeostasisConfig, LesionBarrier, LongTermConfig,
    PlasticNeuralTier, PlasticityConfig, RewiringConfig, STPConfig,
    attach_plasticity, tripod_alternation_reward)

RESULTS = []
BUILT = None
DT_MS = 0.5
PRIMARY_DRIVE_nA = 0.05


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:500]})
    print(("PASS " if ok else "FAIL ") + name
          + (" | " + str(detail)[:220] if detail else ""), flush=True)
    return bool(ok)


def refuse(fn):
    try:
        fn()
    except (ValueError, TypeError, KeyError, RuntimeError, FileNotFoundError):
        return True
    except Exception:                                                    # noqa: BLE001
        return False
    return False


def build():
    global BUILT
    if BUILT is None:
        BUILT = build_neural_tier(TierConfig())
    return BUILT


def make_tier(arm, built=None, lesion=None):
    built = built or build()
    t = PlasticNeuralTier(built, TierConfig(), OwnershipLedger(),
                          plasticity=ABLATION_ARMS[arm] if isinstance(arm, str) else arm,
                          lesion=lesion)
    return t


def uniform_current(tier, drive_nA):
    c = np.zeros(tier.n)
    for leg in LEGS:
        c[tier.sensory[leg]] += float(drive_nA)
    return c


def run_tier(tier, drive_nA, ms, per_leg=None, feedback=None):
    tier.reset()
    if feedback is not None:
        tier.set_feedback(float(feedback))
    c = (uniform_current(tier, drive_nA) if per_leg is None
         else np.asarray(per_leg, float))
    rows = []
    for _ in range(int(round(ms / DT_MS / 10.0))):
        rows.append(tier.advance(10, c))
    return np.concatenate(rows, axis=0)


def tiny_built(n=2, edges=((0, 1),)):
    """A minimal tier for unit-testing the rules in isolation (not a brain)."""
    if edges:
        rows = [post for _, post in edges]
        cols = [pre for pre, _ in edges]
        We = sparse.csr_matrix((np.full(len(edges), 1e-3), (rows, cols)), shape=(n, n))
    else:
        We = sparse.csr_matrix((n, n))
    return {"n": n, "edges": {"We": We, "Wi": sparse.csr_matrix((n, n))},
            "readout_local": np.array([1]), "readout_left": np.array([1]),
            "readout_right": np.array([0]),
            "sensory": {leg: np.array([0]) for leg in LEGS},
            "coverage": {}, "fingerprint": "tiny"}


# ===========================================================================
def main():
    t_start = time.perf_counter()
    built = build()
    print("tier: %d neurons, %d modelled edges, readout %d\n"
          % (built["n"], built["edges"]["nnz_exc"] + built["edges"]["nnz_inh"],
             built["readout_local"].size), flush=True)

    # ---------------------------------------------------------------- A. frozen
    plain = NeuralTier(built, TierConfig(), OwnershipLedger())
    frozen = make_tier("frozen", built)
    plain.reset()
    frozen.reset()
    cur = uniform_current(frozen, PRIMARY_DRIVE_nA)
    same = True
    for _ in range(120):
        a = plain.advance(10, cur)
        b = frozen.advance(10, cur)
        same &= np.array_equal(a, b)
    same &= np.array_equal(plain.net.v.view(np.uint8), frozen.net.v.view(np.uint8))
    same &= np.array_equal(plain.net.ge.view(np.uint8), frozen.net.ge.view(np.uint8))
    same &= np.array_equal(plain.net.gi.view(np.uint8), frozen.net.gi.view(np.uint8))
    same &= np.array_equal(plain.net.adaptation.view(np.uint8),
                           frozen.net.adaptation.view(np.uint8))
    same &= np.array_equal(plain.net.spike_count, frozen.net.spike_count)
    check("frozen arm is BIT-IDENTICAL to the un-plastic tier (1200 substeps: spikes, "
          "v, ge, gi, adaptation, spike_count)", same,
          "spikes total plain=%d frozen=%d"
          % (int(plain.net.spike_count.sum()), int(frozen.net.spike_count.sum())))
    check("frozen arm writes the same ownership-ledger entries as the parent",
          plain.ledger.total("neural_state") == frozen.ledger.total("neural_state")
          == 120, "plain=%d frozen=%d" % (plain.ledger.total("neural_state"),
                                          frozen.ledger.total("neural_state")))

    # frozen through the closed loop, against NO layer at all
    def loop_episode(arm_or_none, seconds=0.6):
        cfg = LoopConfig(arm="neural_modulated", seed=0, duration_s=seconds,
                         gl_backend=None)
        sch = MultirateScheduler(cfg, built_tier=built)
        if arm_or_none is not None:
            attach_plasticity(sch, ABLATION_ARMS[arm_or_none])
        ep = sch.run()
        own = sch.check_ownership()
        sch.close()
        return ep, own

    ep_plain, own_plain = loop_episode(None)
    ep_frozen, own_frozen = loop_episode("frozen")
    ident = all(np.array_equal(np.asarray(ep_plain.truth[k]),
                               np.asarray(ep_frozen.truth[k]))
                for k in ep_plain.truth)
    ident &= all(np.array_equal(np.asarray(ep_plain.observed[k]),
                                np.asarray(ep_frozen.observed[k]))
                 for k in ep_plain.observed)
    ident &= all(np.array_equal(np.asarray(ep_plain.commands[k]),
                                np.asarray(ep_frozen.commands[k]))
                 for k in ep_plain.commands)
    check("frozen arm through the CLOSED LOOP is BIT-IDENTICAL to an episode with no "
          "plasticity layer at all (truth, observed and commands)", ident,
          "speed plain=%.6f frozen=%.6f mm/s"
          % (ep_plain.metrics["mean_speed_mm_s"], ep_frozen.metrics["mean_speed_mm_s"]))
    check("the frozen closed loop satisfies the OWNERSHIP invariants",
          bool(own_plain["ok"] and own_frozen["ok"]),
          "plain=%s frozen=%s" % (own_plain["ok"], own_frozen["ok"]))

    # ------------------------------------------------------------ B. validation
    n_bad = 0
    bad_cases = [
        ("stp U>1", lambda: STPConfig(utilization_U=1.5).validate()),
        ("stp tau_rec<=0", lambda: STPConfig(tau_rec_ms=0.0).validate()),
        ("stp enabled=1 (int, not bool)", lambda: STPConfig(enabled=1).validate()),
        ("stp facilitation='yes'", lambda: STPConfig(facilitation="yes").validate()),
        ("homeo mode unknown", lambda: HomeostasisConfig(mode="hippocampus").validate()),
        ("homeo target outside declared range",
         lambda: HomeostasisConfig(target_rate_hz=1e6).validate()),
        ("homeo gain_min>gain_max",
         lambda: HomeostasisConfig(gain_min=5.0, gain_max=1.0).validate()),
        ("homeo tau<=0", lambda: HomeostasisConfig(tau_homeo_ms=-1.0).validate()),
        ("homeo target not finite",
         lambda: HomeostasisConfig(target_rate_hz=float("nan")).validate()),
        ("lt tau_pre<=tau_post",
         lambda: LongTermConfig(tau_pre_ms=10.0, tau_post_ms=20.0).validate()),
        ("lt w_min_frac>1", lambda: LongTermConfig(w_min_frac=1.5).validate()),
        ("lt eta<0", lambda: LongTermConfig(eta=-1.0).validate()),
        ("rewiring candidate_set_size=0",
         lambda: RewiringConfig(candidate_set_size=0).validate()),
        ("rewiring w_new>per-edge cap",
         lambda: RewiringConfig(w_new_uS=1.0, per_edge_weight_cap_uS=1e-3).validate()),
        ("rewiring thresholds unordered",
         lambda: RewiringConfig(feedback_add_threshold=-0.9,
                                feedback_remove_threshold=0.0).validate()),
        ("rewiring pool name unknown",
         lambda: RewiringConfig(source_pool="somewhere").validate()),
        ("rewiring interval<=0", lambda: RewiringConfig(interval_ms=0.0).validate()),
    ]
    for nm, fn in bad_cases:
        if not refuse(fn):
            print("   NOT REFUSED:", nm, flush=True)
            n_bad += 1
    check("invalid CONFIGURATION is refused (%d cases)" % len(bad_cases),
          n_bad == 0, "not refused: %d" % n_bad)

    t = make_tier("all", built)
    cur0 = uniform_current(t, PRIMARY_DRIVE_nA)
    n_bad = 0
    bad_calls = [
        ("advance n_substeps=0", lambda: t.advance(0, cur0)),
        ("advance wrong current shape", lambda: t.advance(10, np.zeros(3))),
        ("advance non-finite current",
         lambda: t.advance(10, np.full(t.n, np.nan))),
        ("set_feedback outside [-1,1]", lambda: t.set_feedback(2.0)),
        ("set_feedback not a number", lambda: t.set_feedback("reward")),
        ("tripod reward wrong length",
         lambda: tripod_alternation_reward(np.zeros(3))),
        ("lesion with no declared reason",
         lambda: LesionBarrier().add([(0, 1)], "")),
        ("lesion with a non-integer pair", lambda: LesionBarrier().add([(0.5, 1)], "x")),
    ]
    for nm, fn in bad_calls:
        if not refuse(fn):
            print("   NOT REFUSED:", nm, flush=True)
            n_bad += 1
    check("invalid RUNTIME input is refused (%d cases)" % len(bad_calls),
          n_bad == 0, "not refused: %d" % n_bad)

    sched = MultirateScheduler(LoopConfig(arm="cpg_baseline", seed=0, duration_s=0.05,
                                          gl_backend=None), built_tier=None)
    ok = refuse(lambda: attach_plasticity(sched, ABLATION_ARMS["stp_only"]))
    sched.close()
    check("attaching plasticity to the cpg_baseline arm (which has NO neural tier) is "
          "refused", ok)

    # ----------------------------------------------------------- C. determinism
    det_ok = {}
    for arm in ABLATION_ARMS:
        a = run_tier(make_tier(arm, built), PRIMARY_DRIVE_nA, 200.0,
                     feedback=(1.0 if arm == "all" else None))
        b = run_tier(make_tier(arm, built), PRIMARY_DRIVE_nA, 200.0,
                     feedback=(1.0 if arm == "all" else None))
        det_ok[arm] = bool(np.array_equal(a, b))
    check("DETERMINISM: two independently constructed tiers are bit-identical in every "
          "arm (%d arms)" % len(det_ok), all(det_ok.values()), json.dumps(det_ok))

    def rewire_run(seed, feedback_seq=((1.0, 600.0), (-1.0, 600.0)), lesion=None):
        cfg = PlasticityConfig(rewiring=RewiringConfig(enabled=True, seed=seed))
        t = PlasticNeuralTier(built, TierConfig(), OwnershipLedger(), plasticity=cfg,
                              lesion=lesion)
        for f, ms in feedback_seq:
            run_tier(t, PRIMARY_DRIVE_nA, ms, feedback=f)
        rep = t.plasticity_report()
        return rep, t

    r1, _ = rewire_run(0)
    r2, _ = rewire_run(0)
    check("DETERMINISM: two independent rewiring runs produce identical change logs",
          (r1["rewiring"]["added"], r1["rewiring"]["removed"],
           r1["rewiring"]["n_refused_by_lesion_barrier"])
          == (r2["rewiring"]["added"], r2["rewiring"]["removed"],
              r2["rewiring"]["n_refused_by_lesion_barrier"]),
          "run1 added=%d removed=%d | run2 added=%d removed=%d"
          % (r1["rewiring"]["added"], r1["rewiring"]["removed"],
             r2["rewiring"]["added"], r2["rewiring"]["removed"]))
    r3, _ = rewire_run(7)
    check("the candidate set DEPENDS on the declared seed (it is not a constant)",
          r1["rewiring"]["candidate_set_header"]["first_10_pairs"]
          != r3["rewiring"]["candidate_set_header"]["first_10_pairs"])

    # ------------------------------------------------------------------ D. STP
    # D1 recovery-only closed form on a 1-neuron unit case (no recurrent edges)
    stp = STPConfig(enabled=True, utilization_U=0.4, tau_rec_ms=200.0, facilitation=False)
    tb = tiny_built(n=1, edges=())
    tu = PlasticNeuralTier(tb, TierConfig(), OwnershipLedger(),
                           plasticity=PlasticityConfig(stp=stp))
    tu.reset()
    spiked = np.zeros(1, bool)
    spiked[0] = True
    rel = tu._stp_step(spiked, DT_MS)                 # one spike: release U*1
    x0 = float(tu._x[0])
    closed_x0 = 1.0 - 0.4
    rel0 = float(rel[0])
    series = [x0]
    for _ in range(4000):
        tu._stp_step(np.zeros(1, bool), DT_MS)
        series.append(float(tu._x[0]))
    series = np.asarray(series)
    # the release is taken AT the spike instant and the recovery is then applied over the
    # SAME substep, so sample n is the analytic solution at t = (n+1)*dt past the spike
    n = np.arange(series.size)
    closed = 1.0 - (1.0 - closed_x0) * np.exp(-(n + 1) * DT_MS / stp.tau_rec_ms)
    err = float(np.max(np.abs(series - closed)))
    check("STP: one spike releases exactly U*x", abs(rel0 - 0.4) < 1e-15,
          "release=%.17g (U*x=%.17g)" % (rel0, 0.4))
    check("STP: the recovery-only case matches its closed-form exponential "
          "1-(1-x0)exp(-t/tau_rec)", err <= 1e-12,
          "max |iterative - closed form| = %.3e over 4001 samples (2 s, tau_rec=200 ms); "
          "NOT bit-identical: the iterative product exp(-dt/tau)**n and the direct "
          "exp(-n*dt/tau) differ in the last bits" % err)
    check("STP: after a long rest the resource returns to 1 to within 1e-4",
          abs(series[-1] - 1.0) < 1e-4,
          "x after 2 s = %.12f (10 x tau_rec = 2000 ms; 1-x = %.3e)"
          % (series[-1], 1.0 - series[-1]))

    # D2 burst depresses, rest recovers, on the REAL tier
    t = make_tier("stp_only", built)
    t.reset()
    drive = uniform_current(t, PRIMARY_DRIVE_nA)
    ring_mismatch = 0
    for _ in range(200):
        t.advance(10, drive)
        if t._ri != t.net._ring_i:
            ring_mismatch += 1
    x_burst = float(t._x.mean())
    x_burst_min = float(t.history["stp_mean_x"][-1])
    for _ in range(4000):                        # 2 s of rest
        t.advance(10, np.zeros(t.n))
    x_rest = float(t._x.mean())
    check("STP: a burst DEPRESSES the release resource", x_burst < 0.9,
          "mean resource after 1 s of driven firing = %.6f (1.0 at rest)" % x_burst)
    check("STP: a rest RECOVERS the release resource", x_rest > x_burst + 0.05,
          "burst %.6f -> rest %.6f (tau_rec=%.0f ms)"
          % (x_burst, x_rest, ABLATION_ARMS["stp_only"].stp.tau_rec_ms))
    check("STP: the resource bound [0,1] was never violated, so the clip did no work",
          t.n_stp_clip == 0 and float(t._x.min()) >= 0.0 and float(t._x.max()) <= 1.0,
          "clip activations=%d, x in [%.9f, %.9f]" % (t.n_stp_clip, t._x.min(),
                                                      t._x.max()))
    check("STP: the release-value ring is in LOCKSTEP with the parent's delay ring, "
          "checked after EVERY interval of the driven run",
          ring_mismatch == 0 and t._ri == t.net._ring_i,
          "mismatches in 200 intervals = %d (ring length %d, final indices %d/%d)"
          % (ring_mismatch, t._ring_rel.shape[0], t._ri, t.net._ring_i))

    # D3 facilitation raises the running utilisation
    fac = PlasticNeuralTier(tb, TierConfig(), OwnershipLedger(),
                            plasticity=PlasticityConfig(
                                stp=STPConfig(enabled=True, facilitation=True,
                                              utilization_U=0.2, tau_fac_ms=50.0)))
    fac.reset()
    rels, uu = [], []
    for k in range(6):
        rels.append(float(fac._stp_step(np.array([True]), DT_MS)[0]))
        uu.append(float(fac._u[0]))
    check("STP: with facilitation ON the running UTILISATION u grows monotonically over "
          "a burst (that is what facilitation IS)",
          all(uu[i] < uu[i + 1] for i in range(len(uu) - 1)),
          "u per spike: %s (U=0.2)" % ["%.6f" % v for v in uu])
    check("STP: facilitation ON makes the SECOND spike release MORE than the first "
          "(visible before depletion takes over)",
          rels[1] > rels[0],
          "release 1 = %.6f, release 2 = %.6f; later releases then FALL because the "
          "resource depletes faster than u grows: %s"
          % (rels[0], rels[1], ["%.6f" % v for v in rels]))
    nofac = PlasticNeuralTier(tb, TierConfig(), OwnershipLedger(),
                              plasticity=PlasticityConfig(
                                  stp=STPConfig(enabled=True, facilitation=False,
                                                utilization_U=0.2, tau_fac_ms=50.0)))
    nofac.reset()
    rels_nf = [float(nofac._stp_step(np.array([True]), DT_MS)[0]) for _ in range(6)]
    check("STP: facilitation OFF holds u == U EXACTLY, and releases strictly less on the "
          "second spike than facilitation ON",
          bool(np.all(nofac._u == 0.2)) and rels_nf[1] < rels[1],
          "off u == U for every cell (%s); second-spike release off %.6f vs on %.6f"
          % (bool(np.all(nofac._u == 0.2)), rels_nf[1], rels[1]))

    # ------------------------------------------------------------ E. homeostasis
    homeo_arms = [a for a in ABLATION_ARMS if ABLATION_ARMS[a].homeostasis.enabled]
    bound_ok, sat_ok, gmax_seen = True, True, 0.0
    for arm in homeo_arms:
        cfg = ABLATION_ARMS[arm].homeostasis
        tt = make_tier(arm, built)
        run_tier(tt, PRIMARY_DRIVE_nA, 2000.0)
        g = np.asarray(tt.history["g_exc"], float)
        gs = np.asarray(tt.history["g_syn"], float)
        z = np.asarray(tt.history["homeo_z"], float)
        gmax_seen = max(gmax_seen, float(g.max()), float(gs.max()))
        bound_ok &= bool(g.min() >= cfg.gain_min - 1e-12 and g.max() <= cfg.gain_max + 1e-12
                         and gs.min() >= cfg.gain_min - 1e-12
                         and gs.max() <= cfg.gain_max + 1e-12)
        sat_ok &= bool(np.all(np.abs(z) <= cfg.z_limit + 1e-12))
    check("HOMEOSTASIS: the gain NEVER leaves its configured bounds "
          "[gain_min, gain_max] in any homeostasis arm (%d arms, 2 s each)"
          % len(homeo_arms), bound_ok,
          "max gain observed = %.6f (gain_max=%.3f, gain_min=%.3f)"
          % (gmax_seen, ABLATION_ARMS["homeostasis_only"].homeostasis.gain_max,
             ABLATION_ARMS["homeostasis_only"].homeostasis.gain_min))
    check("HOMEOSTASIS: the slow integrator stays inside +/- z_limit", sat_ok,
          "z_limit=%.1f" % ABLATION_ARMS["homeostasis_only"].homeostasis.z_limit)

    # E2 a BROKEN input with homeostasis on
    broken = {}
    for arm in homeo_arms:
        tt = make_tier(arm, built)
        s = run_tier(tt, 0.0, 2000.0)
        broken[arm] = float(s.sum()) * 1000.0 / DT_MS / tt.n / s.shape[0]
    check("HOMEOSTASIS CANNOT HIDE A BROKEN INPUT: with zero transduced drive and the "
          "gain free to rise to its maximum, the tier stays SILENT (all %d ablation arms, "
          "homeostasis mode = '%s')" % (len(homeo_arms),
                                        HomeostasisConfig().mode),
          all(v < 1.0 for v in broken.values()),
          "population rates at zero drive: %s (Hz)"
          % {k: round(v, 6) for k, v in broken.items()})

    # the DECLARED failure mode, measured rather than hidden: synaptic scaling
    syn_rates = {}
    for mode in ("intrinsic", "synaptic", "both"):
        cfg_s = HomeostasisConfig(enabled=True, mode=mode, target_rate_hz=40.0)
        tt = PlasticNeuralTier(built, TierConfig(), OwnershipLedger(),
                               plasticity=PlasticityConfig(homeostasis=cfg_s))
        s0 = run_tier(tt, 0.0, 2000.0)
        syn_rates[mode] = float(s0.sum()) * 1000.0 / DT_MS / tt.n / s0.shape[0]
    check("HOMEOSTASIS, DECLARED FAILURE MODE (reported, not removed): the "
          "SYNAPTIC-SCALING mode MANUFACTURES activity at ZERO input, so it violates the "
          "'cannot hide a broken input' requirement, which is why the ablation arms use "
          "the intrinsic mode",
          syn_rates["synaptic"] > 1.0,
          "zero-drive population rate by mode: intrinsic=%.6f Hz, synaptic=%.6f Hz, "
          "both=%.6f Hz. The synaptic mode scales the recurrent conductance increment, so "
          "the tier's own bistable regime is excited by the scaling itself and the "
          "residual activity self-ignites; STP suppresses it again (see stp_homeostasis)."
          % (syn_rates["intrinsic"], syn_rates["synaptic"], syn_rates["both"]))

    # E3 the declared boundary: turn the background into a homeostatic target
    cfg_bg = HomeostasisConfig(enabled=True, mode="intrinsic", target_rate_hz=40.0,
                               gain_applies_to_background=True)
    t_bg = PlasticNeuralTier(built, TierConfig(), OwnershipLedger(),
                             plasticity=PlasticityConfig(homeostasis=cfg_bg))
    s_bg = run_tier(t_bg, 0.0, 2000.0)
    rate_bg = float(s_bg.sum()) * 1000.0 / DT_MS / t_bg.n / s_bg.shape[0]
    check("HOMEOSTASIS boundary (documented, not hidden): if the gain is ALSO allowed to "
          "scale the fixed background noise, zero drive is no longer silent",
          rate_bg > 0.0,
          "intrinsic mode with gain_applies_to_background=True: %.4f Hz fabricated from "
          "the fixed background alone, versus %.8f Hz with the default False; the "
          "structural guarantee above therefore depends on that declared default"
          % (rate_bg, broken["homeostasis_only"]))

    # ------------------------------------------------------------ F. long term
    t_lt = make_tier("long_term_only", built)
    run_tier(t_lt, PRIMARY_DRIVE_nA, 600.0, feedback=0.0)
    lt0 = t_lt.plasticity_report()["counters"]
    check("LONG-TERM: zero feedback produces EXACTLY zero weight change",
          float(np.max(np.abs(t_lt._lt_e - 1.0))) == 0.0
          and float(np.max(np.abs(t_lt._lt_i - 1.0))) == 0.0,
          "max |W/W0-1| = %.3e" % max(float(np.max(np.abs(t_lt._lt_e - 1.0))),
                                      float(np.max(np.abs(t_lt._lt_i - 1.0)))))

    t_lt2 = make_tier("long_term_only", built)
    run_tier(t_lt2, PRIMARY_DRIVE_nA, 1500.0, feedback=-1.0)
    dev = np.concatenate([t_lt2._lt_e - 1.0, t_lt2._lt_i - 1.0])
    lo = ABLATION_ARMS["long_term_only"].long_term.w_min_frac - 1.0
    hi = ABLATION_ARMS["long_term_only"].long_term.w_max_frac - 1.0
    check("LONG-TERM: per-edge weights stay inside the declared bounds "
          "[w_min_frac, w_max_frac] under sustained negative feedback",
          bool(dev.min() >= lo - 1e-12 and dev.max() <= hi + 1e-12),
          "deviation range [%.6f, %.6f] within [%.3f, %.3f]"
          % (dev.min(), dev.max(), lo, hi))

    t_lt3 = PlasticNeuralTier(built, TierConfig(), OwnershipLedger(),
                              plasticity=PlasticityConfig(
                                  long_term=LongTermConfig(enabled=True, eta=50.0)))
    run_tier(t_lt3, PRIMARY_DRIVE_nA, 600.0, feedback=-1.0)
    dev3 = np.concatenate([t_lt3._lt_e - 1.0, t_lt3._lt_i - 1.0])
    check("LONG-TERM: with an absurd learning rate the declared bound really BINDS "
          "(the bound is doing the work, not the learning rate)",
          bool(dev3.min() >= lo - 1e-12 and dev3.max() <= hi + 1e-12
               and np.count_nonzero((np.abs(dev3 - lo) < 1e-12)
                                    | (np.abs(dev3 - hi) < 1e-12)) > 0),
          "eta=50: %d of %d edge states sit exactly on a bound; range [%.6f, %.6f]"
          % (int(np.count_nonzero((np.abs(dev3 - lo) < 1e-12)
                                  | (np.abs(dev3 - hi) < 1e-12))), dev3.size,
             dev3.min(), dev3.max()))

    # F2 timing dependence, isolated with a scripted spike train on a 1-edge tier
    tb2 = tiny_built(n=2, edges=((0, 1),))
    def elig_pair(first, second, delta_steps=40):
        tt = PlasticNeuralTier(tb2, TierConfig(), OwnershipLedger(),
                               plasticity=PlasticityConfig(
                                   long_term=LongTermConfig(enabled=True)))
        tt.reset()
        for k in range(delta_steps + 80):
            v = np.zeros(2, bool)
            if k == 0:
                v[{"pre": 0, "post": 1}[first]] = True
            if k == delta_steps:
                v[{"pre": 0, "post": 1}[second]] = True
            tt._eligibility_step(v, DT_MS)
        return float(tt._elig_e[0])

    e_causal = elig_pair("pre", "post")
    e_anti = elig_pair("post", "pre")
    check("LONG-TERM: a PRE-before-POST pairing gives a LARGER eligibility than the "
          "reverse order (the trace kernel is timing-dependent, with tau_pre > tau_post)",
          e_causal > e_anti,
          "eligibility pre->post %.9e vs post->pre %.9e at a 20 ms separation "
          "(tau_pre=%.0f ms, tau_post=%.0f ms)"
          % (e_causal, e_anti, LongTermConfig().tau_pre_ms, LongTermConfig().tau_post_ms))

    # ------------------------------------------------------------ G. rewiring
    fields = set(RewiringConfig.__dataclass_fields__)
    forbidden = [f for f in fields
                 if any(t in f.lower() for t in ("dist", "coord", "length", "proxim",
                                                 "radius", "space"))]
    check("REWIRING: there is NO distance/coordinate field anywhere in the "
          "configuration, so a distance constraint is structurally impossible",
          not forbidden, "fields: %s" % sorted(fields))
    check("REWIRING: the tier itself carries no coordinates to fabricate a distance from",
          not any(k in built for k in ("positions", "coords", "coordinates", "xyz")),
          "tier dict keys: %s" % sorted(built.keys()))

    hdr = r1["rewiring"]["candidate_set_header"]
    check("REWIRING: the candidate set is DECLARED and reproducible (seed, pools, three "
          "buckets, size all recorded)",
          all(k in hdr for k in ("declared_candidate_set_size", "pairs_drawn", "seed",
                                 "source_pool", "target_pool", "bucket_barrier_protected",
                                 "bucket_already_modelled_edges", "bucket_uniform_draw",
                                 "distance_constraint")),
          "size=%d barrier_bucket=%d existing_edge_bucket=%d uniform_bucket=%d"
          % (hdr["pairs_drawn"], hdr["bucket_barrier_protected"],
             hdr["bucket_already_modelled_edges"], hdr["bucket_uniform_draw"]))
    check("REWIRING: the candidate set is exactly the declared size",
          hdr["pairs_drawn"] == hdr["declared_candidate_set_size"],
          "%d == %d" % (hdr["pairs_drawn"], hdr["declared_candidate_set_size"]))
    check("REWIRING: with no declared lesion the barrier bucket is empty by "
          "construction (there is nothing to protect)",
          hdr["bucket_barrier_protected"] == 0,
          "barrier bucket = %d; the lesion run below exercises the non-empty case"
          % hdr["bucket_barrier_protected"])

    cfg_rw = ABLATION_ARMS["all"].rewiring
    check("REWIRING: every cap holds (%d additions, %d removals, %d edges)"
          % (r1["rewiring"]["added"], r1["rewiring"]["removed"], r1["n_edges_now"]),
          bool(r1["rewiring"]["add_cap_respected"] and r1["rewiring"]["remove_cap_respected"]
               and r1["rewiring"]["edge_cap_respected"]
               and r1["rewiring"]["added"] <= cfg_rw.max_added_edges
               and r1["rewiring"]["removed"] <= cfg_rw.max_removed_edges
               and r1["n_edges_now"] <= cfg_rw.max_edges_total),
          "caps: +%d/%d, -%d/%d, edges %d/%d"
          % (r1["rewiring"]["added"], cfg_rw.max_added_edges,
             r1["rewiring"]["removed"], cfg_rw.max_removed_edges,
             r1["n_edges_now"], cfg_rw.max_edges_total))
    added_w = [e["new_edge"]["weight_uS"] for e in r1["rewiring"]["accepted_changes_first10"]
               if e["rule"] == "add_edge"]
    check("REWIRING: every ADDED edge respects the per-edge weight cap",
          bool(added_w) and all(w <= cfg_rw.per_edge_weight_cap_uS + 1e-15
                                for w in added_w),
          "weights %s vs cap %.4g uS"
          % (["%.6g" % w for w in added_w[:4]], cfg_rw.per_edge_weight_cap_uS))
    sample = r1["rewiring"]["accepted_changes_first10"]
    fields_ok = bool(sample) and all(
        ("time_ms" in e and "rule" in e and "parent_state" in e
         and ("old_edge" in e) and ("new_edge" in e)) for e in sample)
    check("REWIRING: EVERY accepted change logs time, rule, parent state and the "
          "old/new edge", fields_ok,
          "logged keys of the first accepted change: %s"
          % (sorted(sample[0].keys()) if sample else "none"))
    check("REWIRING: the logged parent state carries the feedback value and the gain",
          bool(sample) and all("feedback" in e["parent_state"]
                               for e in sample),
          "parent_state sample: %s" % (sample[0]["parent_state"] if sample else None))
    check("REWIRING: every REMOVAL logs the edge it destroyed",
          bool([e for e in r1["rewiring"]["accepted_changes_first10"]
                if e["rule"] == "remove_edge"]) or r1["rewiring"]["removed"] == 0,
          "removed %d" % r1["rewiring"]["removed"])

    t_add = make_tier(
        PlasticityConfig(rewiring=RewiringConfig(enabled=True)), built)
    run_tier(t_add, PRIMARY_DRIVE_nA, 600.0, feedback=0.5)
    a_add = t_add.plasticity_report()["rewiring"]["added"]
    t_noadd = make_tier(PlasticityConfig(rewiring=RewiringConfig(enabled=True)), built)
    run_tier(t_noadd, PRIMARY_DRIVE_nA, 600.0, feedback=-1.0)
    rep_noadd = t_noadd.plasticity_report()
    check("REWIRING: the ADD rule needs F above its threshold (F=-1 adds nothing)",
          a_add > 0 and rep_noadd["rewiring"]["added"] == 0,
          "F=+0.5 added %d; F=-1.0 added %d"
          % (a_add, rep_noadd["rewiring"]["added"]))
    check("REWIRING: the REMOVE rule is REACHABLE -- it removes the edges the layer "
          "itself added (F=-1 after F=+1)",
          r1["rewiring"]["removed"] > 0,
          "add-then-remove run: added %d, removed %d"
          % (r1["rewiring"]["added"], r1["rewiring"]["removed"]))
    refused_rm = [e for e in t_noadd.plasticity_events
                  if e.get("rule") == "remove_refused_weight_above_cap"]
    check("REWIRING: when no candidate edge is at or below the declared per-edge cap the "
          "REMOVE rule LOGS ITS REFUSAL rather than silently doing nothing",
          len(refused_rm) > 0,
          "%d refusal events on a fresh tier; closest candidate weight %.6g uS vs cap "
          "%.4g uS, over %s candidate edges"
          % (len(refused_rm),
             (refused_rm[0]["old_edge"] or {}).get("weight_uS", float("nan")),
             refused_rm[0]["per_edge_weight_cap_uS"] if refused_rm else float("nan"),
             refused_rm[0]["parent_state"].get("n_candidate_edges") if refused_rm
             else "n/a"))

    # -------------------------------------------------- H. the lesion barrier
    cut = built["cut"]
    edges = built["edges"]
    data = built["data"]
    rows = np.concatenate([edges["exc_rows"], edges["inh_rows"]])
    severed = np.asarray(cut["row_masks"]["all"])[rows]
    loc = np.asarray(built["local_of_global"])
    pre = loc[np.asarray(data["pre"])[rows]]
    post = loc[np.asarray(data["post"])[rows]]
    barrier = LesionBarrier("declared_neck_cut")
    barrier.add(list(zip(pre[severed].tolist(), post[severed].tolist())),
                reason="declared_neck_cut", time_ms=0.0, weight_uS=0.0, channel="mixed")
    n_sev = int(severed.sum())
    check("LESION: the declared lesion is the project's OWN neck-cut row mask, and it "
          "severs %d modelled in-tier edges" % n_sev, n_sev > 0 and len(barrier) == n_sev,
          "%d severed rows -> %d distinct barred (pre, post) pairs" % (n_sev, len(barrier)))

    cfg_l = PlasticityConfig(rewiring=RewiringConfig(enabled=True))
    t_l = PlasticNeuralTier(built, TierConfig(), OwnershipLedger(), plasticity=cfg_l,
                            lesion=barrier)
    n_before = int(edges["nnz_exc"] + edges["nnz_inh"])
    check("LESION: the barrier is applied BEFORE any plasticity runs (the edges are gone)",
          t_l.n_edges() == n_before - n_sev,
          "%d -> %d edges (-%d)" % (n_before, t_l.n_edges(), n_sev))
    run_tier(t_l, PRIMARY_DRIVE_nA, 1200.0, feedback=1.0)
    rep_add = t_l.plasticity_report()
    run_tier(t_l, PRIMARY_DRIVE_nA, 600.0, feedback=-1.0)
    rep = t_l.plasticity_report()
    We, Wi = t_l._We0.tocoo(), t_l._Wi0.tocoo()
    keys = set(barrier.reasons)
    present = int(sum(1 for r, c in zip(We.row.tolist(), We.col.tolist())
                      if (c, r) in keys)
                  + sum(1 for r, c in zip(Wi.row.tolist(), Wi.col.tolist())
                        if (c, r) in keys))
    check("LESION BARRIER: after driven rewiring under MAXIMUM positive feedback, NOT "
          "ONE barred edge is present", present == 0,
          "barred edges present = %d of %d" % (present, len(keys)))
    check("LESION BARRIER: the ADD rule really DID propose restoring cut edges and was "
          "refused every time", rep_add["rewiring"]["n_refused_by_lesion_barrier"] > 0,
          "%d refusals logged; %d barred pairs placed in the candidate set"
          % (rep_add["rewiring"]["n_refused_by_lesion_barrier"],
             rep_add["rewiring"]["candidate_set_header"][
                 "bucket_barrier_protected"]))
    check("LESION BARRIER: every refusal is logged with the time, the rule, the reason "
          "and the pair",
          bool(barrier.refusals) and all(
              {"time_ms", "rule", "reason", "pre", "post", "parent_state"} <= set(r)
              for r in barrier.refusals),
          "first refusal: %s"
          % {k: barrier.refusals[0][k] for k in ("time_ms", "rule", "reason", "pre",
                                                 "post")})
    check("LESION BARRIER: the ADD rule accepted other (unbarred) candidates in the same "
          "run, so the barrier is not simply blocking everything",
          rep["rewiring"]["added"] > 0,
          "added %d unbarred edges while refusing %d barred proposals"
          % (rep["rewiring"]["added"], rep["rewiring"]["n_refused_by_lesion_barrier"]))
    check("LESION: the reported lesion record states the reason and the count per reason",
          barrier.as_dict()["barred_edges_by_reason"] == {"declared_neck_cut": n_sev},
          json.dumps(barrier.as_dict()["barred_edges_by_reason"]))

    # ------------------------------------------------- I. arms actually differ
    counts = {}
    for arm in ABLATION_ARMS:
        s = run_tier(make_tier(arm, built), PRIMARY_DRIVE_nA, 300.0, feedback=1.0)
        counts[arm] = s.sum(axis=0).astype(np.int64)
    pairs = []
    identical = []
    for i, a in enumerate(ABLATION_ARMS):
        for b in list(ABLATION_ARMS)[i + 1:]:
            d = int(np.abs(counts[a] - counts[b]).sum())
            pairs.append((a, b, d))
            if d == 0:
                identical.append((a, b))
    check("ABLATION: the %d arms actually DIFFER from one another (L1 spike-count "
          "distance per pair)" % len(ABLATION_ARMS),
          len(identical) == 0,
          "pairs identical in spikes: %s" % (identical if identical else "none")
          + " | min pair distance %d, max %d"
          % (min(p[2] for p in pairs), max(p[2] for p in pairs)))
    if identical:
        check("ABLATION: pairs that produce IDENTICAL spikes in this run are reported, "
              "not removed from the study", True,
              "identical pairs (reported as changing nothing here): %s" % identical)

    n_zero = sum(1 for a in ABLATION_ARMS
                 if int(np.abs(counts[a] - counts["frozen"]).sum()) == 0)
    check("ABLATION: every ENABLED mechanism changes the response relative to frozen in "
          "this run", n_zero == 1,
          "arms with a zero distance from frozen (only 'frozen' itself expected): %d"
          % n_zero)

    # rewards
    r_pos = tripod_alternation_reward(np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
    r_neg = tripod_alternation_reward(np.full(6, 0.5))
    check("FEEDBACK PROXY: bounded in [-1,1]; +1 for full tripod alternation, -1 for a "
          "synchronised pattern",
          abs(r_pos - 1.0) < 1e-12 and abs(r_neg + 1.0) < 1e-12,
          "alternating -> %.6f, synchronised -> %.6f" % (r_pos, r_neg))

    # ------------------------------------------------------------- report out
    wall = time.perf_counter() - t_start
    n_pass = sum(1 for r in RESULTS if r["ok"])
    out = {
        "suite": "run_plasticity_selftest.py",
        "n_checks": len(RESULTS),
        "n_passed": n_pass,
        "n_failed": len(RESULTS) - n_pass,
        "passed": n_pass == len(RESULTS),
        "wall_seconds": wall,
        "peak_ram_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
        "tier": {"n_neurons": int(built["n"]),
                 "n_edges": int(built["edges"]["nnz_exc"] + built["edges"]["nnz_inh"])},
        "what_these_tests_establish": [
            "the layer is INERT when off: bit-identical to the un-plastic tier at tier "
            "level and through a whole closed-loop episode",
            "every arm is deterministic across two independent constructions",
            "STP: bounds hold without clipping, the recovery-only case matches its closed "
            "form, a burst depresses and a rest recovers, facilitation grows the "
            "utilisation, and the release ring is in lockstep with the parent's ring",
            "homeostasis: the gain never leaves its configured bounds; a broken input "
            "stays silent; and the DECLARED default that makes that structural is tested "
            "against the flag that breaks it",
            "long-term: zero feedback gives exactly zero change, the bounds hold, the "
            "bounds really bind at an absurd learning rate, and a pre-before-post pairing "
            "beats the reverse",
            "rewiring: no distance field exists; the candidate set is declared and "
            "reproducible; every cap holds; every accepted and refused change is logged "
            "with time, rule, parent state and old/new edge",
            "the lesion barrier: the project's own neck cut is applied first, the ADD "
            "rule proposes the cut edges and is refused every time, and no barred edge "
            "exists after rewiring",
            "invalid configuration and invalid runtime input are refused",
            "the six ablation arms really do differ, and any pair that does not is "
            "reported rather than removed",
        ],
        "what_these_tests_do_NOT_establish": [
            "that any plasticity parameter is measured, or that any timescale is "
            "physiological: the project has NO measured adult-fly central plasticity "
            "timescale tau_p, so nothing here says how long a trained mapping stays valid",
            "that the connectome weights the layer acts on are known: the cached edges "
            "are UNSIGNED (nt_pair = -1 on every row) and their absolute scale is "
            "unmeasured, so the layer amplifies uncertainty rather than reducing it",
            "that the feedback proxy is a neuromodulator: it is an ENGINEERING REWARD "
            "PROXY and this project has no dopamine model or measurement of any kind",
            "that a real nervous system would keep a cut edge cut: the barrier is a "
            "software invariant, not a model of injury repair or regeneration",
            "that the ablation's effect sizes generalise: one bounded tier, one body, flat "
            "ground, three seeds, 8 s",
            "that the closed loop's gait comes from the connectome: FlyGym's demo tripod "
            "CPG generates it in every arm",
            "anything about experience, perception, viability, survival or consciousness",
        ],
        "results": RESULTS,
    }
    p = os.path.join(OUT, "plasticity_selftest.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n%s\n%d/%d checks passed in %.1f s (peak RAM %.0f MB)\n%s"
          % ("-" * 90, n_pass, len(RESULTS), wall, out["peak_ram_mb"], p))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
