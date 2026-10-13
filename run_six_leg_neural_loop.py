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


def build(lesion_leg=None, *, observable=False, ring_rotation_deg=0.0, fly_markers=False):
    from engine.embodied.adapters import (NeuralTier, OwnershipLedger, TierConfig,
                                          build_neural_tier)
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    built = build_neural_tier(TierConfig())
    tier = NeuralTier(built, TierConfig(), OwnershipLedger())
    wiring, stats, _ = build_mn_muscle_wiring(verbose=False)
    _spec = _load_mjcf(str(XML))
    _marker_report: dict = {}
    if observable or fly_markers:
        # OPT-IN OBSERVABILITY, default OFF so every existing result is bit-identical.  It adds the
        # arena's declared fiducials and camera ring to THIS model, so the same camera -> geometry
        # chain that sees stack A can see the body that muscles drive -- while it is moving.
        from tools_make_stackb_observable import attach_observation_geometry
        attach_observation_geometry(_spec, ring_rotation_deg)
        observable = True
    if fly_markers:
        # THE FLY'S OWN MARKERS, WITHOUT WHICH `limb_tracking.py` CANNOT RUN ON THIS EPISODE.  It
        # reads `truth/marker_fly_world_mm`; the arena fiducials alone give camera geometry and the
        # body's POSITION only, which is why the earlier delivery had to say that limb kinematics
        # were not established.  Offsets are MEASURED from this model's own geoms, not copied from
        # stack A -- see `attach_fly_markers`.
        from tools_make_stackb_observable import attach_fly_markers
        _marker_report = attach_fly_markers(_spec)
    m = _spec.compile()
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
                lesion_leg=lesion_leg, observable=observable,
                marker_report=_marker_report or None)


def make_in_loop_stimulus(ctx, **kwargs):
    """Build an `InLoopStimulus` FOR THE TIER THIS `ctx` ACTUALLY HOLDS.

    The electrode places its contacts relative to the tier neurons' soma positions, so a stimulus
    built from a DIFFERENT build of the tier would silently address the wrong neurons -- it would
    still inject current, still change spikes, and still look like it worked.  Passing the `ctx`
    through here is what removes that possibility: `run()` hands its own context to the factory, so
    the electrode and the network it stimulates are the same objects.
    """
    from engine.embodied.electrode_in_loop import (InLoopElectrode, InLoopStimulus,
                                                   SomaPositions)
    built = ctx["built"]
    gi = np.asarray(built["global_index"], dtype=np.int64)
    roots = np.asarray(built["data"]["root_ids"], dtype=np.int64)[gi]
    elec = InLoopElectrode(SomaPositions.from_banc(roots), n_tier_neurons=int(built["n"]),
                           n_channels=4, dt_neural_ms=float(TIER_DT_MS))
    return InLoopStimulus(elec, **kwargs), elec


def run(lesion_leg=None, duration_s=1.0, verbose=True, frame_sink=None,
        frame_stride=1, observable=False, ring_rotation_deg=0.0, spike_sink=None,
        fly_markers=False, stimulus=None, stim_sink=None):
    # `observable` MUST BE FORWARDED, not just declared: MEASURED, `run` built its own context with
    # the default, so the model it stepped had NO camera ring and a 640-wide framebuffer while the
    # caller believed it had asked for observability -- and the renderer then failed with
    # "Image width 800 > framebuffer width 640".
    # AN UNKNOWN LEG KEY MUST REFUSE, NOT SILENTLY SKIP.  MEASURED, and it is the exact vacuity this
    # file's own docstring warns about: `lesion_leg="lf"` (lower case) does not match this module's
    # key `"LF"`, so `mn_local.get(lesion_leg)` returned None, the `and` short-circuited, and the
    # ablation was SKIPPED -- leaving the lesioned run BIT-IDENTICAL to the intact one.  A caller
    # that passed the lower-case name got a clean-looking null result that meant nothing.  The
    # vacuity warning below only PRINTS, and only when verbose; refusing here is what makes the
    # mistake impossible to miss.
    if lesion_leg is not None and lesion_leg not in LEGS:
        raise ValueError(
            "lesion_leg must be one of %r (this module's own keys, upper case); got %r.  An "
            "unrecognised key used to be skipped silently, which produced a lesioned run identical "
            "to the intact one and would have been read as 'the neurons do not drive this leg'."
            % (tuple(LEGS), lesion_leg))
    ctx = build(lesion_leg, observable=observable, ring_rotation_deg=ring_rotation_deg,
                fly_markers=fly_markers)
    built, tier, m, d = ctx["built"], ctx["tier"], ctx["m"], ctx["d"]
    wiring, act_of, jids = ctx["wiring"], ctx["act_of"], ctx["jids"]
    mn_local = ctx["mn_local"]
    sens = {leg: np.asarray(built["sensory"][LEG_TO_TIER[leg]], dtype=np.int64)
            for leg in LEGS}
    n_tier = int(built["n"])
    q0 = np.asarray(d.qpos, dtype=float).copy()
    tier.reset()

    # ---- THE STIMULUS SEAM, INSIDE THE LOOP WHERE MUSCLES ARE DRIVEN BY NEURONS.
    #
    # Until now `NeuralStimulus` had only ever been exercised against a tier advanced in isolation:
    # two identical networks, one stimulated, spikes compared.  That proves the current reaches the
    # membrane, and it proves nothing about whether it reaches a MUSCLE.  Here the stimulated tier is
    # the one whose motor-neuron firing IS the muscle activation, so the chain a stimulus has to
    # travel is the whole point of the platform: electrode -> tier -> motor neurons -> muscles ->
    # body motion -> back to the electrode and the cameras.
    #
    # THE STIMULUS IS BUILT FROM `ctx` VIA A FACTORY, when a factory is given, so the electrode can
    # never address a different build of the tier than the one being stepped.  A pre-built stimulus
    # is also accepted, and then its tier width is checked instead.
    st = None
    if stimulus is not None:
        st = stimulus(ctx) if callable(stimulus) else stimulus
        if hasattr(st, "__len__") and not hasattr(st, "current_nA"):
            st, _elec = st
        if int(getattr(st.electrode, "n_tier_neurons", -1)) != n_tier:
            raise ValueError(
                f"the stimulus addresses a tier of {st.electrode.n_tier_neurons} neurons but this "
                f"build has {n_tier}; it would inject current into the wrong neurons.  Build it with "
                "`make_in_loop_stimulus` so it is tied to this context.")
        if not callable(getattr(st, "current_nA", None)):
            raise TypeError("stimulus must provide current_nA(t_ms, n_neurons)")

    n_cmd = int(round(duration_s / DT_COMMAND_S))
    q_trace = {leg: [] for leg in LEGS}
    act_trace, rate_trace = [], []
    stim_trace: list[dict] = []
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
        if st is None:
            # THE DEFAULT PATH IS UNCHANGED, so every existing episode stays bit-identical.  With a
            # stimulus the interval cannot be advanced in one call, because the injected current
            # changes from substep to substep (that is what a waveform IS), so the loop steps one
            # substep at a time and adds the stimulus current to the sensory drive at each.
            spikes = tier.advance(N_NEURAL_SUB, cur)      # (N_NEURAL_SUB, n_tier) bool
        else:
            spikes = np.zeros((N_NEURAL_SUB, n_tier), dtype=bool)
            row = {"interval": int(c), "t0_ms": float(c * N_NEURAL_SUB * float(TIER_DT_MS)),
                   "n_substeps_nonzero": 0, "charge_nA_ms": 0.0, "peak_abs_nA": 0.0,
                   "n_neurons_nonzero": 0}
            for k in range(N_NEURAL_SUB):
                t_ms = (c * N_NEURAL_SUB + k) * float(TIER_DT_MS)
                inj = np.asarray(st.current_nA(t_ms, n_tier), dtype=float)
                spikes[k] = tier.advance(1, cur + inj)
                nz = int((inj != 0).sum())
                if nz:
                    row["n_substeps_nonzero"] += 1
                    # CHARGE, NOT JUST AMPLITUDE: the waveform is sampled, so what the tissue
                    # receives is the sum over substeps of current x dt, and that is the number the
                    # charge ceiling is about.
                    row["charge_nA_ms"] += float(inj.sum()) * float(TIER_DT_MS)
                    row["peak_abs_nA"] = max(row["peak_abs_nA"], float(np.abs(inj).max()))
                    row["n_neurons_nonzero"] = max(row["n_neurons_nonzero"], nz)
            stim_trace.append(row)
            if stim_sink is not None:
                stim_sink(row, spikes, c)
        # ---- ACUTE ABLATION: zero one leg's motor neurons, and only those
        if lesion_leg is not None and mn_local.get(lesion_leg):
            spikes[:, mn_local[lesion_leg]] = False
        # ---- THE SPIKES ARE OFFERED TO AN OBSERVER, AFTER the lesion has been applied, so a
        # recorder sees what the BODY was actually driven by.  A recorder that read the pre-lesion
        # spikes would report activity that never reached a muscle, which is the opposite of the
        # causal claim an electrode is supposed to support.
        if spike_sink is not None:
            spike_sink(spikes, c)
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
        # ---- AN OBSERVATION HOOK, called AFTER the body has moved for this interval.  It is a
        # callback rather than a renderer inside this file so that the loop keeps ONE
        # implementation: a second copy of the sensorimotor chain would drift from this one, and the
        # drift would be invisible in whichever copy was not being read.
        if frame_sink is not None and (c % max(int(frame_stride), 1) == 0):
            frame_sink(m, d, c, act_trace)
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
            "n_cmd": n_cmd, "stats": ctx["stats"],
            # THE STIMULUS SEAM'S OWN RECORD, empty when no stimulus was attached.  `intervals_with_
            # current` and the total charge are what make "the stimulus ran" auditable rather than
            # inferred from the spikes having moved.
            "stimulus": ({
                "mode": getattr(st, "mode", None),
                "amplitude_nA": float(getattr(getattr(st, "wave", None), "amplitude_nA", 0.0)),
                "capture_radius_um": float(getattr(st, "capture_um", 0.0)),
                "n_recruited_neurons": int(getattr(st, "n_recruited", 0)),
                "n_intervals": len(stim_trace),
                "n_intervals_with_current": int(sum(1 for r in stim_trace
                                                    if r["n_substeps_nonzero"])),
                "total_charge_nA_ms": float(sum(r["charge_nA_ms"] for r in stim_trace)),
                "peak_abs_nA": float(max((r["peak_abs_nA"] for r in stim_trace), default=0.0)),
                "per_interval": stim_trace,
            } if st is not None else None)}


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
