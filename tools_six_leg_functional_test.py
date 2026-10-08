#!/usr/bin/env python3
"""Does a MIRRORED muscle actually MOVE its leg?  Compiling is not evidence; motion is.

WHAT IS BEING TESTED, AND WHY IT IS THE WHOLE POINT.  The mirrored model compiles and reports 90
muscle actuators, but a tendon attached to a body with no joint produces no motion at all -- it
merely pulls against rigid geometry.  So for each of the four repaired legs this test activates one
muscle and measures three things:
  * whether the actuator produces force at all,
  * whether the joint it should drive changes angle,
  * whether the tendon length changes (a tendon whose length is fixed is not driving anything).
A muscle that activates but moves nothing is reported as INERT.

Run:
    MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_six_leg_functional_test.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_muscles.xml"
#: one muscle per repaired leg, and the joint its name says it should move
CASES = [
    ("LFTibia_flex_93434", "joint_LFTibia_pitch"),
    ("RFTibia_flex_93434", "joint_RFTibia_pitch"),
    ("LMTibia_flex_93434", "joint_LMTibia_pitch"),
    ("RMTibia_flex_93434", "joint_RMTibia_pitch"),
    ("LHTibia_flex_93434", "joint_LHTibia_pitch"),
    ("RHTibia_flex_93434", "joint_RHTibia_pitch"),
    ("LMC_pleural_remotor_and_abductor", "joint_LMCoxa_pitch"),
    ("LFC_pleural_remotor_and_abductor", "joint_LFCoxa_pitch"),
    ("RFC_pleural_remotor_and_abductor", "joint_RFCoxa_pitch"),
]


def main() -> int:
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    spec = _load_mjcf(str(XML))
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"compiled: njnt={m.njnt} nq={m.nq} nu={m.nu} ntendon={m.ntendon}")
    n_musc = sum(1 for i in range(m.nu)
                 if m.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE)
    print(f"muscle-typed actuators: {n_musc}")

    def jid(name):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return i
    def aid(name):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return i

    print(f"\n{'muscle':<38}{'joint':<26}{'force':>10}{'d qpos':>10}{'d tendon mm':>13}  verdict")
    rows = []
    for mus, jnt in CASES:
        a = aid(mus); j = jid(jnt)
        if a < 0 or j < 0:
            print(f"{mus:<38}{jnt:<26}{'MISSING':>10}{'-':>10}{'-':>13}  cannot test")
            rows.append({"muscle": mus, "joint": jnt, "verdict": "missing"})
            continue
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        q0 = float(d.qpos[m.jnt_qposadr[j]])
        t0 = float(d.actuator_length[a])
        peak_f = 0.0
        for _ in range(2000):                     # 0.2 s at 1e-4
            d.ctrl[a] = 1.0                        # full activation
            mujoco.mj_step(m, d)
            peak_f = max(peak_f, abs(float(d.actuator_force[a])))
        q1 = float(d.qpos[m.jnt_qposadr[j]])
        t1 = float(d.actuator_length[a])
        moved = abs(q1 - q0)
        verdict = ("MOVED" if moved > 1e-4 else
                   "INERT (activated but no motion)" if peak_f > 1e-6 else
                   "NO FORCE")
        rows.append({"muscle": mus, "joint": jnt, "peak_force": peak_f,
                     "dq_rad": moved, "d_tendon_mm": abs(t1 - t0), "verdict": verdict})
        print(f"{mus:<38}{jnt:<26}{peak_f:>10.4f}{moved:>10.5f}{abs(t1-t0):>13.4f}  {verdict}")

    print("\nverdicts:")
    for leg in ("LF", "RF", "LM", "RM", "LH", "RH"):
        sub = [r for r in rows if r["muscle"].startswith(leg)]
        ok = [r for r in sub if r.get("verdict") == "MOVED"]
        print(f"  {leg}: {len(ok)}/{len(sub)} tested muscles actually move their joint")

    import json
    dest = HERE / "outputs" / "six_leg_functional_test.json"
    dest.write_text(json.dumps({"xml": str(XML.relative_to(HERE)), "njnt": m.njnt, "nu": m.nu,
                                "rows": rows}, indent=2, sort_keys=True))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
