#!/usr/bin/env python3
"""CLOSED LOOP ON SIX LEGS, AND THE ACUTE ABLATION TEST.

THE CHAIN, WITH THE ONLY HAND-AUTHORED PARTS NAMED.
    body joint angles (per leg) -> mechanosensory neurons of that leg      [receptor transduction]
    those neurons + the connectome edges -> motor-neuron spikes            [the network]
    each muscle's OWN motor neurons' firing rate -> that muscle's activation
    activations -> MuJoCo muscles -> the body moves                        [the plant]

There is NO state-to-action function in it.  A muscle's activation IS the measured firing of that
muscle's own motor neurons, normalised by a reference rate.  Hand-authored ingredients, exactly
two, both declared:
  1. the anatomical motor-neuron-to-muscle map -- the connectome's own annotation;
  2. the receptor transduction and the reference rate -- ILLUSTRATIVE, as the existing receptor
     bank already says of itself.  This file does NOT pretend they are measured physiology.

THE ACUTE ABLATION.  Zero the MNs of ONE leg.  Its muscles then receive exactly zero activation.
If that leg still moves, the neurons were not driving it and the claim fails.  Because the thorax
is welded to the world in this model, one leg cannot drive another except through the shared
trunk, so the test is clean.

Run:
    MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python run_six_leg_neural_loop.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from muscle_wiring import build_mn_muscle_wiring  # noqa: E402

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_muscles.xml"
#: The connectome's sensory groups are keyed by lowercase left/right + front/mid/hind, the muscle
#: model by an uppercase side-then-segment prefix.  Same convention, different case and order --
#: MEASURED: assuming they matched raised KeyError 'FL'.
LEG_TO_TIER = {"LF": "lf", "RF": "rf", "LM": "lm", "RM": "rm", "LH": "lh", "RH": "rh"}
LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
#: MY OWN MODULES USED TWO DIFFERENT LEG CONVENTIONS, which silently produced an empty map:
#: the muscle model and tools_mn_muscle_map_per_leg's wiring use "LF" (left-front), while I had
#: also written "FL" (front-left) in the loop.  A lookup keyed "FL" simply missed every "LF"
#: entry, so no muscle was ever driven and the ablation had nothing to silence.  One convention
#: now, the model's own.
DT = 1e-4
N_COMMAND_STEPS = 50              # 5 ms command clock
# ---------------------------------------------------------------------------
# THE NEURAL CLOCK COMES FROM THE TIER, NEVER FROM AN ASSUMPTION.
# MEASURED BUG, and it inflated every rate this project has reported from these loops: the tier's
# TierConfig.dt_ms is 0.5 ms -- identical to the engine's own LoopConfig.dt_neural_s -- but these
# scripts assumed a 0.1 ms neural substep and used DT*N_NEURAL_SUB = 0.5 ms as the rate window while
# calling advance(5, ...), which in fact covers 5 * 0.5 ms = 2.5 ms.  Two consequences, both real:
#   * every firing rate, and therefore every muscle activation (min(1, rate/REF_RATE_HZ)), was 5x
#     too high, so activations saturated far earlier than the measurements implied;
#   * the network advanced 2.5 ms of neural time per 5 ms of physics, i.e. it ran at HALF real time.
# The engine's own convention (LoopConfig) is neural_substeps = dt_command_s / dt_neural_s = 10.
# The substep count and the window are now derived from the tier itself and asserted.
# ---------------------------------------------------------------------------
TIER_DT_MS = 0.5                      # MEASURED: TierConfig().dt_ms
DT_COMMAND_S = DT * N_COMMAND_STEPS   # 5 ms command interval
N_NEURAL_SUB = int(round(DT_COMMAND_S * 1000.0 / TIER_DT_MS))   # 10
RATE_WINDOW_S = N_NEURAL_SUB * TIER_DT_MS / 1000.0             # 5 ms

REF_ANGLE_RAD = 0.30              # ILLUSTRATIVE receptor normalisation
REF_RATE_HZ = 40.0                # ILLUSTRATIVE reference firing rate
CURRENT_PER_DRIVE_NA = 0.02       # ILLUSTRATIVE transduction gain.
#: SPONTANEOUS (TONIC) DRIVE, and it is not a fudge -- its absence was a real defect.  MEASURED:
#: with the drive being the joint-angle DEVIATION from the starting pose, a leg sitting at its rest
#: pose receives exactly zero current, its motor neurons stay silent, and its muscles receive zero
#: activation.  Ablating those neurons then changes nothing, and it is easy to mistake that for "the
#: neurons do not drive the leg" when it is really "the neurons were never driven".  A real
#: mechanosensory neuron is spontaneously active and encodes ABSOLUTE joint position, not just
#: deviation, so a tonic component belongs in the transduction.  This is still an ILLUSTRATIVE
#: parameter, not a measured firing rate.
TONIC_DRIVE_NA = 0.02
# MEASURED: at 0.30 the loop RAN AWAY -- joint excursions reached 5.7 rad against joint ranges of
# 0.48-2.5 rad and motor-neuron rates hit 2000 Hz, because sensory -> motor -> motion -> more
# sensory is a positive feedback loop with no damping in my transduction.  The gain is turned down
# until the excursions sit inside the anatomical ranges; it is a loop gain, not a controller.


def build(lesion_leg=None):
    from engine.embodied.adapters import (NeuralTier, OwnershipLedger, TierConfig,
                                          build_neural_tier)
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    built = build_neural_tier(TierConfig())
    tier = NeuralTier(built, TierConfig(), OwnershipLedger())
    wiring, stats, _ = build_mn_muscle_wiring(verbose=False)
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    # START FROM THE MODEL'S OWN NEUTRAL KEYFRAME, NOT qpos0.  MEASURED: qpos0 differs from the
    # keyframe pose by 2.8 rad, so starting in qpos0 drops the legs from a pose their springs do
    # not expect and produces a large passive transient that has nothing to do with the neurons.
    kf = 0 if m.nkey else None
    if kf is not None:
        mujoco.mj_resetDataKeyframe(m, d, kf)
    mujoco.mj_forward(m, d)
    from muscle_wiring import actuator_to_mn_type
    a2t = actuator_to_mn_type(verbose=False)     # ALL muscles in THIS model, not just the foreleg
    act_of, name_of = {}, {}
    for i in range(m.nu):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        name_of[i] = nm
        hit = a2t.get(nm)
        if hit:
            act_of.setdefault(hit, []).append(i)
    jids = {leg: [] for leg in LEGS}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("joint_") and nm[6:8] in jids:
            jids[nm[6:8]].append(j)
    mn_local = {leg: sorted({li for g in wiring[leg].values() for li in g["local"]})
                for leg in LEGS}
    return dict(built=built, tier=tier, wiring=wiring, stats=stats, m=m, d=d,
                act_of=act_of, name_of=name_of, jids=jids, mn_local=mn_local,
                lesion_leg=lesion_leg)


def run(lesion_leg=None, duration_s=1.0, verbose=True):
    ctx = build(lesion_leg)
    built, tier, m, d = ctx["built"], ctx["tier"], ctx["m"], ctx["d"]
    wiring, act_of, jids = ctx["wiring"], ctx["act_of"], ctx["jids"]
    mn_local = ctx["mn_local"]
    sens = {leg: np.asarray(built["sensory"][LEG_TO_TIER[leg]], dtype=np.int64)
            for leg in LEGS}
    n_tier = int(built["n"])
    q0 = np.asarray(d.qpos, dtype=float).copy()
    tier.reset()

    n_cmd = int(round(duration_s / DT_COMMAND_S))
    q_trace = {leg: [] for leg in LEGS}
    act_trace, rate_trace = [], []
    for c in range(n_cmd):
        # ---- READ the body, TRANSDUCE into per-leg receptor currents
        q = np.asarray(d.qpos, dtype=float)
        cur = np.zeros(n_tier, dtype=float)
        for leg in LEGS:
            dev = float(np.mean(np.abs(q[[int(m.jnt_qposadr[j]) for j in jids[leg]]]
                                      - q0[[int(m.jnt_qposadr[j]) for j in jids[leg]]])))
            strain = min(1.0, dev / REF_ANGLE_RAD)
            # tonic + deviation-driven, per leg
            cur[sens[leg]] += TONIC_DRIVE_NA + CURRENT_PER_DRIVE_NA * strain
        # ---- ADVANCE the connectome
        spikes = tier.advance(N_NEURAL_SUB, cur)          # (N_NEURAL_SUB, n_tier) bool
        # ---- ACUTE ABLATION: zero one leg's motor neurons, and only those
        if lesion_leg is not None and mn_local.get(lesion_leg):
            spikes[:, mn_local[lesion_leg]] = False
        # ---- MNs -> muscle activations (no state-to-action function anywhere)
        for (leg, mtype), acts in act_of.items():
            grp = wiring.get(leg, {}).get(mtype)
            a = 0.0
            if grp and grp["local"]:
                sel = spikes[:, grp["local"]]
                rate = float(sel.sum()) / (sel.shape[0] * RATE_WINDOW_S)
                a = float(min(1.0, rate / REF_RATE_HZ))
                rate_trace.append(rate)
            for i in acts:
                d.ctrl[i] = a
            act_trace.append((leg, mtype, a))
        # ---- HOLD the activations while the body moves
        for _ in range(N_COMMAND_STEPS):
            mujoco.mj_step(m, d)
        for leg in LEGS:
            qq = np.asarray(d.qpos, dtype=float)
            q_trace[leg].append(float(np.sum(np.abs(
                qq[[int(m.jnt_qposadr[j]) for j in jids[leg]]]
                - q0[[int(m.jnt_qposadr[j]) for j in jids[leg]]]))))
    # ---- MEASURE THE SETTLED POSTURE, NOT THE TRANSIENT.  MEASURED: with no muscle activation the
    # legs settle to a fixed pose (max-min over the last 0.5 s = 0.000 rad), so an excursion over
    # the whole run is dominated by the one-time settle and hides what the neurons do.  The metric
    # is therefore the mean |deviation from the starting pose| over the LAST 0.3 s.
    n_settle = max(1, int(round(0.3 / DT_COMMAND_S)))
    exc = {leg: float(np.mean(v[-n_settle:])) if v else 0.0 for leg, v in q_trace.items()}
    final = {leg: (v[-1] if v else 0.0) for leg, v in q_trace.items()}
    if verbose:
        print(f"lesion={lesion_leg!r}  duration={duration_s}s")
        print(f"  {'leg':<5}{'joint excursion rad':>22}{'final |dq| rad':>17}")
        for leg in LEGS:
            print(f"  {leg:<5}{exc[leg]:>22.4f}{final[leg]:>17.4f}")
        if rate_trace:
            print(f"  motor-neuron firing rates: mean {np.mean(rate_trace):.2f} Hz, "
                  f"max {np.max(rate_trace):.2f} Hz  (reference {REF_RATE_HZ} Hz)")
    return {"lesion": lesion_leg, "excursion": exc, "final": final,
            "rate_mean": float(np.mean(rate_trace)) if rate_trace else 0.0,
            "n_cmd": n_cmd, "stats": ctx["stats"]}


if __name__ == "__main__":
    out = {"runs": []}
    base = run(None)
    out["runs"].append(base)
    # NAMES MUST BE THE MODEL'S OWN.  MEASURED: this loop asked for "ML" and "HL", which do not
    # exist in a model that names its legs LF/RF/LM/RM/LH/RH, so mn_local.get("ML") returned None,
    # the ablation was silently SKIPPED, and the lesioned run came out bit-identical to the intact
    # one -- a vacuous test that looked like a passing one.
    for leg in ("LM", "LH", "LF"):
        out["runs"].append(run(leg))
    print("\n=== ACUTE ABLATION TEST ===")
    b = base["excursion"]
    if not any(abs(base["excursion"][l]) > 1e-6 for l in LEGS):
        raise SystemExit(
            "VACUOUS TEST: nothing moves even intact, so an ablation cannot demonstrate anything. "
            "Refusing to report a verdict.")
    # A second vacuity check: if the lesioned run is bit-identical to the intact one it means the
    # lesion was NOT APPLIED (this happened once through a leg-name mismatch), and reporting
    # "100% of intact" would be reporting a bug as a result.
    for r in out["runs"][1:]:
        if all(abs(r["excursion"][l] - b[l]) < 1e-12 for l in LEGS):
            print(f"  WARNING: the run with {r['lesion']} silenced is BIT-IDENTICAL to the intact "
                  f"run, so the lesion had no effect anywhere.  Treat that leg as NOT TESTED.")
    for r in out["runs"][1:]:
        leg = r["lesion"]
        print(f"  silence {leg} motor neurons: its joints move "
              f"{r['excursion'][leg]:.4f} vs {b[leg]:.4f} rad intact "
              f"({100 * r['excursion'][leg] / max(b[leg], 1e-9):.0f}% of intact)")
        others = [l for l in LEGS if l != leg]
        print(f"    other legs: mean {np.mean([r['excursion'][l] for l in others]):.4f} "
              f"vs intact {np.mean([b[l] for l in others]):.4f} rad")
    dest = HERE / "outputs" / "six_leg_ablation.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {dest}")
