"""NEURAL INTERFACE REBUILD, step 28: body in the loop - can the MEASURED command budget keep the
stance up longer than open loop?

WHAT THIS IS FOR
----------------
Steps 4-27 established the interface's measured properties and its limits:

    sensory front end rebuilt and accepted (7 channels, signed, no broadcast)   parts 5-6
    motor pools resolved 74/90 from the right column, zeroing costs 1.01x       steps 11-12, 15
    rate coding cannot work at a 5 ms clock                                      steps 4/16/17/19
    latency coding needs a ~25 ms budget for 3 decodable levels                  step 22
    the motor population's 50% step response takes 88-110 ms                    step 24
    the sign is not decodable from the pattern (held-out accuracy 0.00)          step 27

The objective asks for a quantitative check of CLOSED-LOOP controllability.  The remaining
question that can still be answered with what exists is a BUDGET question, and it does not need the
polarity data: if a corrective command with the MEASURED latency is applied to the stance, does the
body stay up longer than with no correction?  The polarity is handled by testing BOTH signs, so the
result does not depend on knowing which is correct.

METHOD (mirrors the round-39 gated protocol, which is where the 25 ms baseline comes from)
-----------------------------------------------------------------------------------------
  * seat the round-37 pose with the round-39 gate (tarsus-only, no body inside the floor);
  * constant activation = the round-37 LP balance, adhesion pinned to zero;
  * a PD-style corrective command is added on top, with
        - a DELAY tau (0, 5, 10, 25, 50 ms) applied to the deviation, i.e. the command at time t
          reflects the state at t - tau, which is what a delayed loop actually does;
        - a gain, and BOTH signs of the gain;
        - the error signal being the SIGNED tibia joint deviation of each leg (the quantity the
          rebuilt front end produces), mapped through the corrected wiring table of step 11 to
          each muscle's OWN leg;
  * measure the escape time (thorax leaving 90% of the seated height) against the open-loop
    baseline.

PRE-REGISTERED
--------------
  H18  At least one (delay, gain, sign) combination extends the escape time beyond the open-loop
       baseline by more than 20%.
  FALSIFIER: no combination does, i.e. no command with the measured latency and a realistic gain
       can improve this stance - which is the decisive statement about closed-loop controllability
       given the interface as measured.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
import run_fix_round39 as R39        # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
POOLS = OUT / "neural_rebuild_11_mn_pools.json"
HORIZON_S = 0.30
ESCAPE_FRACTION = 0.90
DELAYS_MS = (0.0, 5.0, 10.0, 25.0, 50.0)
GAINS = (0.2, 0.5, 1.0)
SIGNS = (+1.0, -1.0)


def main() -> None:
    import mujoco

    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    pools = json.loads(POOLS.read_text())["measured"]
    resolved = pools["resolved"]
    # actuator name -> its own leg (from the resolver's own parse)
    act_leg = {}
    for r in resolved:
        act_leg[r["actuator"]] = r["leg"]

    model = cp.load_model()
    data = mujoco.MjData(model)
    pose = json.loads(R39.ROUND37_POSE.read_text())
    R39.apply_round37_pose(model, data, pose)
    base = data.qpos.copy()
    z = cp.seat_root(model, data, base)
    data.qpos[:] = base
    data.qpos[2] = z
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    gate = cp.stance_facts(model, data)          # raises unless this is a stance
    seated = np.asarray(data.qpos, float).copy()

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
             for i in range(model.nu)]
    acts = json.loads(R39.ROUND37_STANCE.read_text())["best"]["activation"]
    a_base = np.array([float(acts.get(nm, 0.0)) for nm in names])
    a_base[model.nu - 2:] = 0.0

    # which actuators are flexor-like per leg: from the actuator NAME, declared
    leg_rows = {}
    for i, nm in enumerate(names):
        leg = act_leg.get(nm)
        if leg is None:
            continue
        leg_rows.setdefault(leg.lower(), []).append(i)
    LEG_TO_BODY = {"lf": "front_leg", "lm": "middle_leg", "lh": "hind_leg",
                   "rf": "front_leg", "rm": "middle_leg", "rh": "hind_leg"}
    # the tibia joint index inside each leg's 7-DOF block (measured in part 6)
    hinge = [j for j in range(model.njnt)
             if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    hnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
    tib_angle_row = {}
    for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
        rows = [k for k, nm in enumerate(hnames) if nm and nm.lower().startswith(f"joint_{leg}")]
        tib_angle_row[leg] = rows[-1]
    # the qpos address of each leg's tibia joint
    tib_qadr = {}
    for leg in tib_angle_row:
        idx = hinge[tib_angle_row[leg]]
        tib_qadr[leg] = int(model.jnt_qposadr[idx])
    ref_tib = {leg: float(seated[tib_qadr[leg]]) for leg in tib_qadr}

    def flexor_like(leg):
        """Actuators whose NAME says flex, on this leg; the polarity choice is DECLARED."""
        out = []
        for i in leg_rows.get(leg, []):
            if "flex" in names[i].lower():
                out.append(i)
        return out

    flexor = {leg: flexor_like(leg) for leg in leg_rows}

    def integrate(delay_ms, gain, sign):
        d = mujoco.MjData(model)
        d.qpos[:] = seated
        d.qvel[:] = 0
        n_steps = int(round(HORIZON_S / model.opt.timestep))
        buf_len = max(1, int(round(delay_ms / 1000.0 / model.opt.timestep)) + 1)
        hist = [seated.copy() for _ in range(buf_len)]
        # MEASURED BUG (fixed): `h0` was read before any mj_forward, so xpos was still zero and the
        # escape threshold became 0 - the escape time came out None for every condition, which
        # looked like "nothing escapes" instead of "the check never fired".
        mujoco.mj_forward(model, d)
        h0 = float(d.xpos[model.body("Thorax").id][2])
        escape = None
        for k in range(n_steps):
            hist.append(np.asarray(d.qpos, float).copy())
            past = hist[-buf_len] if len(hist) >= buf_len else hist[0]
            ctrl = a_base.copy()
            for leg in flexor:
                if not flexor[leg]:
                    continue
                dev = float(past[tib_qadr[leg]] - ref_tib[leg])
                delta = sign * gain * np.clip(dev / 0.30, -1.0, 1.0)
                for i in flexor[leg]:
                    if dal := True:
                        ctrl[i] = float(np.clip(ctrl[i] + delta, 0.0, 1.0))
            d.ctrl[:] = ctrl
            d.act[:] = ctrl
            mujoco.mj_step(model, d)
            h = float(d.xpos[model.body("Thorax").id][2])
            if escape is None and h < ESCAPE_FRACTION * h0:
                escape = (k + 1) * model.opt.timestep
        return escape, float(d.xpos[model.body("Thorax").id][2])

    t0 = time.time()
    open_loop = integrate(0.0, 0.0, +1.0)
    rows = []
    for delay in DELAYS_MS:
        for gain in GAINS:
            for sign in SIGNS:
                esc, final = integrate(delay, gain, sign)
                rows.append({"delay_ms": delay, "gain": gain, "sign": sign,
                             "escape_time_s": esc, "final_height_mm": cp.model_to_mm(final)})
        print(f"  delay {delay:5.1f} ms done ({time.time()-t0:.0f}s)", flush=True)

    base_esc = open_loop[0]
    improved = [r for r in rows if r["escape_time_s"] is not None and base_esc is not None
                and r["escape_time_s"] > base_esc * 1.2]
    h18 = bool(improved)
    best = max(rows, key=lambda r: (r["escape_time_s"] or -1.0))
    cp.write_result(
        OUT / "neural_rebuild_28_body_in_loop.json",
        question=("With the interface's MEASURED latency, can any corrective command keep the "
                  "stance up longer than no correction at all?"),
        status="PASS",
        measured={
            "seated": gate.as_dict(),
            "gate_passed": True,
            "horizon_s": HORIZON_S, "escape_fraction": ESCAPE_FRACTION,
            "open_loop": {"escape_time_s": open_loop[0],
                          "final_height_mm": cp.model_to_mm(open_loop[1])},
            "delays_ms": list(DELAYS_MS), "gains": list(GAINS), "signs": list(SIGNS),
            "rows": rows,
            "n_combinations": len(rows),
            "n_improving_beyond_20pct": len(improved),
            "best": best,
            "predictions": {"H18": {"supported": h18,
                                    "falsifier": "no combination extends the escape time by >20%"}},
            "wall_s": time.time() - t0,
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_28_body_in_loop.py"),
            "protocol": "round-39 gated seat + round-37 LP balance as the constant command + "
                        "adhesion pinned to 0",
            "command_path": "the SIGNED tibia deviation of each leg, delayed by tau, mapped onto "
                            "that leg's flexor-named actuators through the corrected wiring table "
                            "of step 11",
            "polarity": "BOTH signs are tested, so the result does not depend on knowing which is "
                        "corrective",
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "This is NOT the connectome in the loop: the corrective command is a declared PD law on "
            "the signed tibia deviation, with the MEASURED latency imposed.  It answers 'can this "
            "body be stabilised at all with a command of this latency', not 'does the tier drive "
            "the muscles'.",
            "The gain is a declared number; no muscle force calibration is claimed.",
            "The maps are name-based (flexor-named actuators, tibia joint), so the sign of the "
            "resulting torque is checked BY TESTING BOTH SIGNS rather than by assumption.",
            "0.30 s is one stance, not a gait and not behaviour.",
        ],
    )
    print(f"  open loop escape {base_esc} s, final {cp.model_to_mm(open_loop[1]):.2f} mm")
    for r in rows:
        if r["escape_time_s"] and base_esc and r["escape_time_s"] > base_esc:
            print(f"    improving: delay {r['delay_ms']} gain {r['gain']} sign {r['sign']:+.0f} "
                  f"-> {r['escape_time_s']*1000:.1f} ms")
    print(f"  H18 (any combination improves >20%): {h18}")
    print("wrote", OUT / "neural_rebuild_28_body_in_loop.json")


if __name__ == "__main__":
    main()
