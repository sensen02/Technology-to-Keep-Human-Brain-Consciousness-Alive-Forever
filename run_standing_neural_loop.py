#!/usr/bin/env python3
"""DOES MOTOR-NEURON-DRIVEN MUSCLE ACTIVATION MAKE THE FLY STAND?

MEASURED MOTIVATION, and it is why this experiment exists: on the standing model (free joint, the
tarsi solved onto the floor, the COM 0.62 mm inside the support polygon) NO passive configuration
stands.  Sweeping the leg-joint spring stiffness from 0.4 to 128 and the damping from 0.02 to 5, and
starting the fly anywhere from seated to 0.2 mm above the floor, leaves zero stable cases: soft
springs let it sag until 62% of its weight rests on the thorax, abdomen and proboscis, and stiff
springs make it tip over (64 degrees, one leg down).  Gravity is not at fault -- the floor reaction
measures 1.000 body weight exactly.  The passive skeleton has no stable standing equilibrium.

So standing has to come from the muscles, and the muscles have to be driven by their own motor
neurons.  This runs the closed loop on the STANDING model:

    body joint angles -> that leg's mechanosensory neurons        [receptor transduction]
    the connectome's own edges -> motor-neuron spikes             [the network]
    each muscle's OWN motor neurons' firing rate -> its activation
    activations -> MuJoCo muscles -> the body moves

and then applies the ACUTE ABLATION: silence the motor neurons and see whether the behaviour
disappears.  Silence is applied by zeroing the motor neurons' spikes, so the affected muscles then
receive exactly zero activation -- the muscles are never set to a value by hand.

Run:
    MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python run_standing_neural_loop.py
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
N_COMMAND_STEPS = 50
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

REF_ANGLE_RAD = 0.30          # ILLUSTRATIVE receptor normalisation (unchanged from the verified loop)
REF_RATE_HZ = float(os.environ.get("REF_RATE_HZ", "40.0"))   # ILLUSTRATIVE reference firing rate
CURRENT_PER_DRIVE_NA = 0.02   # ILLUSTRATIVE transduction gain
TONIC_DRIVE_NA = 0.02         # ILLUSTRATIVE spontaneous rate of a leg mechanosensory neuron
DURATION_S = float(os.environ.get("STAND_DURATION", "1.5"))


def build(lesion_all=False, lesion_legs=()):
    from engine.embodied.adapters import (NeuralTier, OwnershipLedger, TierConfig,
                                          build_neural_tier)
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    built = build_neural_tier(TierConfig())
    tier = NeuralTier(built, TierConfig(), OwnershipLedger())
    wiring, stats, _ = build_mn_muscle_wiring(verbose=False)
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    a2t = actuator_to_mn_type(verbose=False)
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
    if lesion_all:
        mn_local["__ALL__"] = sorted({li for leg in LEGS for li in mn_local[leg]})
    return dict(built=built, tier=tier, m=m, d=d, act_of=act_of, name_of=name_of, jids=jids,
                mn_local=mn_local, lesion_all=lesion_all, lesion_legs=tuple(lesion_legs),
                stats=stats, wiring=wiring)


def measure(m, d, mw):
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    up = d.xmat[th].reshape(3, 3)[:, 2]
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    leg = body = 0.0
    legs, others = set(), set()
    for c in range(d.ncon):
        cc = d.contact[c]
        if cc.geom1 != fid and cc.geom2 != fid:
            continue
        other = cc.geom2 if cc.geom1 == fid else cc.geom1
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other])) or ""
        f = np.zeros(6); mujoco.mj_contactForce(m, d, c, f)
        if b[:2] in LEGS:
            leg += abs(f[0]); legs.add(b[:2])
        else:
            body += abs(f[0]); others.add(b)
    return {"thorax_z": float(d.xpos[th][2]), "up_z": float(up[2]),
            "legs_down": sorted(legs), "nonleg_bodies_down": sorted(others),
            "leg_bw": leg / mw, "body_bw": body / mw,
            "speed": float(np.linalg.norm(d.qvel[:3]))}


def run(lesion_all=False, lesion_legs=(), duration_s=None, verbose=True):
    if duration_s is None:
        duration_s = DURATION_S
    ctx = build(lesion_all, lesion_legs)
    built, tier, m, d = ctx["built"], ctx["tier"], ctx["m"], ctx["d"]
    act_of, jids, mn_local = ctx["act_of"], ctx["jids"], ctx["mn_local"]
    sens = {leg: np.asarray(built["sensory"][LEG_TO_TIER[leg]], dtype=np.int64) for leg in LEGS}
    n_tier = int(built["n"])
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    q0 = np.asarray(d.qpos, dtype=float).copy()
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    up0 = d.xmat[th].reshape(3, 3)[:, 2].copy()
    z0 = float(d.xpos[th][2])
    assert int(tier.cfg.dt_ms * 1000) == int(TIER_DT_MS * 1000), (
        f"the tier's dt_ms is {tier.cfg.dt_ms}, not the assumed {TIER_DT_MS}")
    tier.reset()
    n_cmd = int(round(duration_s / DT_COMMAND_S))
    rate_trace, act_trace, state_trace = [], [], []
    for c in range(n_cmd):
        q = np.asarray(d.qpos, dtype=float)
        cur = np.zeros(n_tier, dtype=float)
        for leg in LEGS:
            adr = [int(m.jnt_qposadr[j]) for j in jids[leg]]
            dev = float(np.mean(np.abs(q[adr] - q0[adr])))
            cur[sens[leg]] += TONIC_DRIVE_NA + CURRENT_PER_DRIVE_NA * min(1.0, dev / REF_ANGLE_RAD)
        spikes = tier.advance(N_NEURAL_SUB, cur)
        # ---- ACUTE ABLATION: zero the motor neurons' spikes.  The muscles are then driven by
        # nothing at all; no activation is ever written by hand.
        if lesion_all:
            spikes[:, mn_local["__ALL__"]] = False
        for leg in lesion_legs:
            spikes[:, mn_local[leg]] = False
        # (leg, muscle type) -> tier-local motor-neuron indices, straight from the anatomical map
        for (leg, mtype), acts in act_of.items():
            g = ctx["wiring"].get(leg, {}).get(mtype)
            a = 0.0
            if g and g["local"]:
                sel = spikes[:, g["local"]]
                rate = float(sel.sum()) / (sel.shape[0] * DT * N_NEURAL_SUB)
                a = float(min(1.0, rate / REF_RATE_HZ))
                rate_trace.append(rate)
            for i in acts:
                d.ctrl[i] = a
            act_trace.append(a)
        for _ in range(N_COMMAND_STEPS):
            mujoco.mj_step(m, d)
        st = measure(m, d, mw)
        st["t"] = c * DT * N_COMMAND_STEPS
        st["up"] = float(np.asarray(d.xmat[th]).reshape(3, 3)[:, 2][2])
        state_trace.append(st)
    final = state_trace[-1]
    if verbose:
        print(f"  lesion_all={lesion_all} legs={lesion_legs}  duration={duration_s}s")
        print(f"    start thorax z {z0:.4f} -> final {final['thorax_z']:.4f} mm   "
              f"up_z {up0[2]:+.3f} -> {final['up']:+.3f}")
        print(f"    floor load: tarsi {final['leg_bw']:.4f} bw, BODY {final['body_bw']:.4f} bw")
        print(f"    legs down: {','.join(final['legs_down']) or 'NONE'} ({len(final['legs_down'])}/6)")
        print(f"    non-leg bodies on the floor: {','.join(final['nonleg_bodies_down']) or 'NONE'}")
        print(f"    motor-neuron firing: mean {np.mean(rate_trace):.1f} Hz, "
              f"max {np.max(rate_trace):.1f} Hz; mean muscle activation {np.mean(act_trace):.4f}")
    return {"lesion_all": lesion_all, "lesion_legs": list(lesion_legs),
            "start_thorax_z": z0, "final": final, "trace": state_trace,
            "rate_mean": float(np.mean(rate_trace)), "rate_max": float(np.max(rate_trace)),
            "act_mean": float(np.mean(act_trace)),
            "stood": bool(final["body_bw"] < 1e-3 and len(final["legs_down"]) == 6)}


if __name__ == "__main__":
    out = {"xml": str(XML), "runs": []}
    print("=== INTACT ===")
    intact = run()
    out["runs"].append(intact)
    print("\n=== ACUTE ABLATION: silence ALL leg motor neurons ===")
    silenced = run(lesion_all=True)
    out["runs"].append(silenced)
    print("\n=== ACUTE ABLATION: silence ONE leg (LF) ===")
    one = run(lesion_legs=("LF",))
    out["runs"].append(one)
    print("\n=== VERDICT ===")
    print(f"  intact:   {'STANDS' if intact['stood'] else 'does NOT stand'} "
          f"(body load {intact['final']['body_bw']:.4f} bw, {len(intact['final']['legs_down'])}/6 legs)")
    print(f"  silenced: {'STANDS' if silenced['stood'] else 'does NOT stand'} "
          f"(body load {silenced['final']['body_bw']:.4f} bw, {len(silenced['final']['legs_down'])}/6 legs)")
    d_body = silenced["final"]["body_bw"] - intact["final"]["body_bw"]
    print(f"  silencing the motor neurons changes the body load by {d_body:+.4f} bw "
          f"({'the neurons are doing work' if abs(d_body) > 1e-4 else 'NO EFFECT -- vacuous test'})")
    dest = HERE / "outputs" / "standing_neural_loop.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {dest}")
