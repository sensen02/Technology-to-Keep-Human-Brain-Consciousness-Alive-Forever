#!/usr/bin/env python3
"""CLOSE THE LOOP ON LOAD: the insect resistance reflex, wired from the connectome's own edges.

WHY LOAD AND NOT ANGLE.  MEASURED: with the loop driven by JOINT ANGLE alone -- the only sensory
channel it has had so far -- the intact fly is WORSE than the same fly with its motor neurons
silenced (0.801 body weights on the body against 0.629).  And no passive configuration stands, and no
steady hand-set muscle pattern stands, so the loop has to close on the quantity that actually carries
the load.

THE PATHWAY'S SIGN IS NOT CHOSEN HERE, IT IS MEASURED.  diagnostics/diag_reflex_matrix.py shows that
adding current to one leg's mechanosensory neurons raises THAT leg's tibia extensor motor neurons by
+290 to +395 Hz, for all six legs -- the resistance reflex.  Those sensory cells are annotated SNta
(tarsal) and SNch (chordotonal) in BANC, i.e. the right kind of cell to carry a load signal.  So the
load channel below feeds the connectome's own wiring and lets its own synapses decide the sign.

WHAT IS HAND-AUTHORED, AND IT IS STILL EXACTLY TWO THINGS:
    1. the anatomical motor-neuron-to-muscle map, which is the connectome's own annotation;
    2. the receptor transduction -- here the conversion of tarsal contact force to an equivalent
       current -- plus the reference rate.  ILLUSTRATIVE, as the existing receptor bank already says
       of itself.  No activation is ever written by hand, and there is no state-to-action function.

Run:
    MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python run_standing_load_reflex.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from muscle_wiring import build_mn_muscle_wiring, actuator_to_mn_type  # noqa: E402

XML = Path(os.environ.get("STANDING_XML",
                          HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"))
LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
LEG_TO_TIER = {"LF": "lf", "RF": "rf", "LM": "lm", "RM": "rm", "LH": "lh", "RH": "rh"}
DT = 1e-4
N_COMMAND_STEPS = 50             # 5 ms command clock
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

DURATION_S = float(os.environ.get("REFLEX_DURATION", "1.5"))
REF_ANGLE_RAD = 0.30             # ILLUSTRATIVE receptor normalisation (unchanged)
REF_RATE_HZ = float(os.environ.get("REF_RATE_HZ", "40.0"))     # ILLUSTRATIVE reference firing rate
TONIC_DRIVE_NA = 0.02            # ILLUSTRATIVE spontaneous rate
ANGLE_GAIN_NA = float(os.environ.get("ANGLE_GAIN_NA", "0.02"))
#: ILLUSTRATIVE load transduction: a leg carrying one sixth of the fly's weight is taken as the
#: reference, so REF_LOAD_N is one body weight in newtons per leg.  The model's force unit is 1 uN
#: (mass in grams, length in mm, gravity in mm/s^2), so a body weight in force units is m*g.
LOAD_GAIN_NA = float(os.environ.get("LOAD_GAIN_NA", "3.0"))
LOAD_REF_FRACTION = float(os.environ.get("LOAD_REF_FRACTION", "0.1667"))


def build(lesion_all=False, lesion_legs=()):
    from engine.embodied.adapters import (NeuralTier, OwnershipLedger, TierConfig,
                                          build_neural_tier)
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    built = build_neural_tier(TierConfig())
    tier = NeuralTier(built, TierConfig(), OwnershipLedger())
    wiring, stats, _ = build_mn_muscle_wiring(verbose=False)
    m = _load_mjcf(str(XML)).compile()
    # PLANT CALIBRATION, APPLIED TO THE MODEL RATHER THAN SOLVED FOR.  MEASURED: the passive joint
    # stiffness of 0.4 uN*mm/rad was fitted by the source paper for a TETHERED model in which only one
    # leg moves, and as a load-bearing skeleton it leaves the legs carrying only 58% of the weight at
    # the stance (passive vertical acceleration -4149 mm/s^2).  That is a MECHANICAL PARAMETER of the
    # model, not a controller, and this does not compute or impose any activation pattern: the fly's
    # own loop still has to hold it up.  The constraint that keeps the claim honest is that with ZERO
    # activation it must still NOT stand.
    _stiff = float(os.environ.get("PLANT_STIFFNESS", "0.4"))
    _damp = float(os.environ.get("PLANT_DAMPING", "0.02"))
    for _j in range(m.njnt):
        _nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, _j) or ""
        if _nm.startswith("joint_"):
            m.jnt_stiffness[_j] = _stiff
            m.dof_damping[int(m.jnt_dofadr[_j])] = _damp
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    a2t = actuator_to_mn_type(verbose=False)
    act_of = {}
    for i in range(m.nu):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
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
    if lesion_all:
        mn_local["__ALL__"] = sorted({li for leg in LEGS for li in mn_local[leg]})
    # tarsus geoms per leg, to read that leg's tarsal load off MuJoCo's own contacts
    tarsus = {leg: [g for g in range(m.ngeom)
                    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                    .startswith(leg) and "Tarsus" in
                    (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
              for leg in LEGS}
    return dict(built=built, tier=tier, m=m, d=d, act_of=act_of, jids=jids, mn_local=mn_local,
                wiring=wiring, stats=stats, tarsus=tarsus, lesion_all=lesion_all,
                lesion_legs=tuple(lesion_legs))


def tarsal_load(m, d, tarsus, fid):
    """total normal force the floor applies to each leg's tarsus, in force units (1 unit = 1 uN)."""
    out = {leg: 0.0 for leg in LEGS}
    for c in range(d.ncon):
        cc = d.contact[c]
        if cc.geom1 != fid and cc.geom2 != fid:
            continue
        other = cc.geom2 if cc.geom1 == fid else cc.geom1
        for leg in LEGS:
            if other in tarsus[leg]:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, c, f)
                out[leg] += abs(f[0])
    return out


def measure(m, d, mw, tarsus, fid):
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    up = d.xmat[th].reshape(3, 3)[:, 2]
    leg_bw = body_bw = 0.0
    legs, others = set(), set()
    for c in range(d.ncon):
        cc = d.contact[c]
        if cc.geom1 != fid and cc.geom2 != fid:
            continue
        other = cc.geom2 if cc.geom1 == fid else cc.geom1
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other])) or ""
        f = np.zeros(6); mujoco.mj_contactForce(m, d, c, f)
        if b[:2] in LEGS:
            leg_bw += abs(f[0]); legs.add(b[:2])
        else:
            body_bw += abs(f[0]); others.add(b)
    return {"thorax_z": float(d.xpos[th][2]), "up_z": float(up[2]),
            "leg_bw": leg_bw / mw, "body_bw": body_bw / mw,
            "legs_down": sorted(legs), "nonleg": sorted(others),
            "speed": float(np.linalg.norm(d.qvel[:3]))}


def run(mode="load", lesion_all=False, lesion_legs=(), duration_s=None, verbose=True,
        drive_scale=1.0):
    """mode: 'angle' (the previous channel), 'load', or 'both'.

    ``drive_scale`` multiplies every ILLUSTRATIVE receptor gain at once.  It exists because of a
    MEASURED defect: with the gains as first written, the tier's motor neurons fire at 480-504 Hz and
    the mean muscle activation sits at 0.49-0.52, while the SAME network with zero external current
    fires at 0.2 Hz (median 0, max 6).  So the transduction was driving the network about a thousand
    times past its own resting behaviour into saturation, which produces a near-uniform
    half-activation -- and uniform co-contraction was measured separately to leave the body load at
    0.42-0.53 body weights, i.e. the fly pressed onto the floor.  A motor system that is saturated
    carries no pattern.  Scaling the drive is a receptor parameter, not a hand-written action.
    """
    if duration_s is None:
        duration_s = DURATION_S
    ctx = build(lesion_all, lesion_legs)
    built, tier, m, d, wiring = ctx["built"], ctx["tier"], ctx["m"], ctx["d"], ctx["wiring"]
    act_of, jids, mn_local, tarsus = ctx["act_of"], ctx["jids"], ctx["mn_local"], ctx["tarsus"]
    sens = {leg: np.asarray(built["sensory"][LEG_TO_TIER[leg]], dtype=np.int64) for leg in LEGS}
    n_tier = int(built["n"])
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    ref_load = LOAD_REF_FRACTION * mw
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    q0 = np.asarray(d.qpos, dtype=float).copy()
    up0 = d.xmat[th].reshape(3, 3)[:, 2].copy()
    z0 = float(d.xpos[th][2])
    assert int(tier.cfg.dt_ms * 1000) == int(TIER_DT_MS * 1000), (
        f"the tier's dt_ms is {tier.cfg.dt_ms}, not the assumed {TIER_DT_MS}; the neural clock and "
        "the rate window would both be wrong")
    assert abs(N_NEURAL_SUB * TIER_DT_MS / 1000.0 - DT_COMMAND_S) < 1e-12, (
        "the neural substeps must cover exactly one command interval")
    tier.reset()
    n_cmd = int(round(duration_s / DT_COMMAND_S))
    rate_trace, act_trace, trace, load_trace = [], [], [], []
    for c in range(n_cmd):
        q = np.asarray(d.qpos, dtype=float)
        loads = tarsal_load(m, d, tarsus, fid)
        cur = np.zeros(n_tier, dtype=float)
        for leg in LEGS:
            if mode in ("angle", "both"):
                adr = [int(m.jnt_qposadr[j]) for j in jids[leg]]
                dev = float(np.mean(np.abs(q[adr] - q0[adr])))
                cur[sens[leg]] += drive_scale * (TONIC_DRIVE_NA
                                                 + ANGLE_GAIN_NA * min(1.0, dev / REF_ANGLE_RAD))
            if mode in ("load", "both"):
                cur[sens[leg]] += drive_scale * LOAD_GAIN_NA * min(1.0, loads[leg] / ref_load)
        spikes = tier.advance(N_NEURAL_SUB, cur)
        if lesion_all:
            spikes[:, mn_local["__ALL__"]] = False
        for leg in lesion_legs:
            spikes[:, mn_local[leg]] = False
        for (leg, mtype), acts in act_of.items():
            g = wiring.get(leg, {}).get(mtype)
            a = 0.0
            if g and g["local"]:
                sel = spikes[:, g["local"]]
                rate = float(sel.sum()) / (sel.shape[0] * RATE_WINDOW_S)
                a = float(min(1.0, rate / REF_RATE_HZ))
                rate_trace.append(rate)
            for i in acts:
                d.ctrl[i] = a
            act_trace.append(a)
        for _ in range(N_COMMAND_STEPS):
            mujoco.mj_step(m, d)
        st = measure(m, d, mw, tarsus, fid)
        st["t"] = c * DT * N_COMMAND_STEPS
        st["loads"] = loads
        st["act_mean"] = float(np.mean(act_trace[-len(act_of):])) if act_of else 0.0
        trace.append(st)
        load_trace.append(loads)
    fin = trace[-1]
    if verbose:
        print(f"  mode={mode} lesion_all={lesion_all} legs={lesion_legs}")
        print(f"    thorax z {z0:.4f} -> {fin['thorax_z']:.4f} mm; up_z {up0[2]:+.3f} -> "
              f"{fin['up_z']:+.3f}")
        print(f"    tarsi {fin['leg_bw']:.4f} bw, BODY {fin['body_bw']:.4f} bw; "
              f"legs down {len(fin['legs_down'])}/6"
              + (f"; non-leg on floor: {','.join(fin['nonleg'])}" if fin["nonleg"] else ""))
        print(f"    MN firing mean {np.mean(rate_trace):.1f} Hz; mean muscle activation "
              f"{np.mean(act_trace):.4f}")
        print(f"    tarsal load per leg (bw): "
              + " ".join(f"{l}={fin['loads'][l]/mw:.3f}" for l in LEGS))
    return {"mode": mode, "lesion_all": lesion_all, "lesion_legs": list(lesion_legs),
            "start_thorax_z": z0, "final": fin, "trace": trace,
            "rate_mean": float(np.mean(rate_trace)), "rate_max": float(np.max(rate_trace)),
            "act_mean": float(np.mean(act_trace)),
            "stood": bool(fin["body_bw"] < 1e-3 and len(fin["legs_down"]) == 6)}


if __name__ == "__main__":
    out = {"xml": str(XML), "runs": [], "plant": {
        "stiffness": float(os.environ.get("PLANT_STIFFNESS", "0.4")),
        "damping": float(os.environ.get("PLANT_DAMPING", "0.02"))}, "params": {
        "load_gain_nA": LOAD_GAIN_NA, "angle_gain_nA": ANGLE_GAIN_NA,
        "load_ref_fraction_of_bw": LOAD_REF_FRACTION, "ref_rate_hz": REF_RATE_HZ,
        "duration_s": DURATION_S}}
    if os.environ.get("PLANT_SWEEP"):
        # THE JUDGE IS THE FLY'S OWN LOOP, NOT A SOLVER.  No activation pattern is computed or imposed
        # anywhere: the connectome drives the muscles from the tarsi's load and the joints' angles, and
        # the question is simply whether the fly ends up standing.  Each setting is also checked for the
        # constraint that ZERO activation must still fail, so the posture cannot be the springs'.
        for stiff in (float(v) for v in os.environ["PLANT_SWEEP"].split(",")):
            globals()["_PLANT"] = stiff
            os.environ["PLANT_STIFFNESS"] = str(stiff)
            print(f"\n=== plant stiffness {stiff} (the loop is the judge) ===")
            passive = run("none")
            print(f"    [zero activation at this stiffness: body {passive['final']['body_bw']:.4f} bw, "
                  f"thorax {passive['final']['thorax_z']:.4f} mm -> "
                  + ("STANDS: rejected, the posture would be the springs"
                     if passive["stood"] else "collapses, so a stand here would be neural") + "]")
            r = run("both")
            r["tag"] = f"stiffness {stiff}"; r["plant_stiffness"] = stiff
            r["passive"] = passive["final"]; r["passive_stood"] = passive["stood"]
            out["runs"].append(r)
    elif os.environ.get("REF_SWEEP"):
        # SWEEP THE REFERENCE RATE.  REF_RATE_HZ is declared ILLUSTRATIVE, and the corrected clock
        # exposed that its value was doing something indefensible: with REF_RATE_HZ=40 a motor neuron
        # firing at 43 Hz -- which is what the loop actually produces now -- already gives its muscle
        # FULL activation, so the muscle layer had almost no resolution.  Drosophila leg motor neurons
        # fire up to a few hundred Hz and muscle activation is graded across that range, so the
        # reference belongs near the top of it.  This measures the consequence instead of asserting it.
        for ref in (40.0, 100.0, 200.0, 400.0):
            print(f"\n=== REF_RATE_HZ {ref} (load channel) ===")
            globals()["REF_RATE_HZ"] = ref
            r = run("load")
            r["tag"] = f"REF_RATE_HZ={ref}"; r["ref_rate_hz"] = ref
            out["runs"].append(r)
    elif os.environ.get("REFLEX_SWEEP"):
        # FIND THE REGIME WHERE THE NETWORK IS NOT SATURATED, then ask whether it stands there.
        for sc in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3):
            print(f"\n=== drive_scale {sc} (both channels) ===")
            r = run("both", drive_scale=sc)
            r["tag"] = f"drive_scale={sc}"; r["drive_scale"] = sc
            out["runs"].append(r)
    else:
        runs = [("angle only (the previous channel)", "angle", False, ()),
                ("LOAD (resistance reflex)", "load", False, ()),
                ("load + angle", "both", False, ()),
                ("LOAD, all leg motor neurons silenced", "load", True, ()),
                ("LOAD, one leg's motor neurons silenced (LF)", "load", False, ("LF",))]
        for tag, mode, la, ll in runs:
            print(f"\n=== {tag} ===")
            r = run(mode, lesion_all=la, lesion_legs=ll)
            r["tag"] = tag
            out["runs"].append(r)
    print("\n=== VERDICT ===")
    for r in out["runs"]:
        print(f"  {r['tag']:<44} {'STANDS' if r['stood'] else 'does not stand'}  "
              f"(body load {r['final']['body_bw']:.4f} bw, thorax {r['final']['thorax_z']:.4f} mm)")
    dest = HERE / "outputs" / ("plant_sweep.json" if os.environ.get("PLANT_SWEEP")
                              else "ref_rate_sweep.json" if os.environ.get("REF_SWEEP")
                              else "standing_load_reflex_sweep.json"
                              if os.environ.get("REFLEX_SWEEP")
                              else "standing_load_reflex.json")
    dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
