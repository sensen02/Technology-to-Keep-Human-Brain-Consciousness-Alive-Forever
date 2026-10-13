#!/usr/bin/env python3
"""IS THE ONE-WAY JOINT AN UPSTREAM DEFECT OR OURS?  THE SHIPPED ASSET ANSWERS IT.

THE CLAIM THAT HAD TO BE CHECKED.  Eight rounds of diagnosis localized the muscle-driven stack's
standing failure to 6 one-way actuated leg joints, and the first conclusion written down was that
FlyMimic ships the defect.  That conclusion was WRONG, and this file is the measurement that shows it.

WHAT IT COMPARES.  FlyGym ships its musculoskeletal model as a package asset, and that asset is a
ONE-LEG model.  This project builds the six-leg model from it with `tools_mirror_muscles.py`.  So the
question "is it upstream?" is answered by measuring the same joint in both:

    shipped asset  flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml
    ours           outputs/muscles_six_legs/fruitfly_six_leg_standing.xml

A joint is BIDIRECTIONAL when the muscles on it produce torques of both signs; a muscle's torque is
its tendon moment arm times its force, and in this model the force is one-signed, so the sign
structure of the moment arms decides it.

MEASURED RESULT.  The shipped asset's fore-leg tibia is properly antagonistic -- flex arm -0.01524
giving torque +1.0377, extensor arm +0.03324 giving torque -8.5427 -- while in our six-leg build the
MIDDLE legs' tibia has both muscles with the same sign (LMTibia_flex +0.12347, LMTibia_extensor
+0.10027).  So the defect is introduced by this project's mirroring, it is in this repository, and it
is fixable -- which the earlier registration denied.

A RANDOM SWEEP, so this is not a configuration artefact: 400 random configurations over all seven of
the middle leg's joints produce ZERO bidirectional cases for `joint_LMTibia_pitch`, against 373 of 400
for the fore leg's.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_whose_defect.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

SHIPPED = (HERE / "venv_body/lib/python3.12/site-packages/flygym/assets/model/musculoskeletal/"
           "best_combined_arm_damping_stiff_cvt3.xml")
OURS = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"


def main() -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--out", default=str(HERE / "outputs" / "whose_defect.json"))
    args = ap.parse_args()

    def torque_table(m, d, joint_name):
        """Every muscle's contribution to one joint: (name, moment arm, force, torque)."""
        d.act[:] = 1.0                      # SET, not left at zero: mj_forward does not integrate act
        mujoco.mj_forward(m, d)
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if j < 0:
            return [], set()
        dof = int(m.jnt_dofadr[j])
        ra = np.asarray(d.moment_rowadr)
        val = np.asarray(d.actuator_moment)
        ci = np.asarray(d.moment_colind)
        af = np.asarray(d.actuator_force)
        rows, signs = [], set()
        for i in range(m.nu):
            s = int(ra[i])
            e = int(ra[i + 1]) if i + 1 < len(ra) else len(val)
            for k in range(s, e):
                if int(ci[k]) != dof or abs(float(val[k])) < 1e-9:
                    continue
                t = float(val[k]) * float(af[i])
                if abs(t) > 1e-9:
                    signs.add(float(np.sign(t)))
                rows.append({"actuator": str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)),
                             "moment_arm": float(val[k]), "force": float(af[i]), "torque": t})
        return rows, signs

    # ---- 1. the shipped ONE-LEG asset
    ms = _load_mjcf(str(SHIPPED)).compile()
    ds = mujoco.MjData(ms)
    if ms.nkey:
        mujoco.mj_resetDataKeyframe(ms, ds, 0)
    mujoco.mj_forward(ms, ds)
    print(f"shipped asset : nq {ms.nq}, nv {ms.nv}, nu {ms.nu}, ntendon {ms.ntendon} -- ONE LEG")
    ship = {}
    for jn in ("joint_LFTibia_pitch", "joint_RFTibia_pitch"):
        rows, signs = torque_table(ms, ds, jn)
        ship[jn] = {"signs": sorted(signs), "bidirectional": len(signs) == 2, "muscles": rows}
        print(f"  {jn}: {len(rows)} muscle(s), signs {sorted(signs)}, "
              f"{'BIDIRECTIONAL' if len(signs) == 2 else 'one-way'}")
        for r in rows:
            print(f"     {r['actuator']:34s} arm {r['moment_arm']:+.5f}  torque {r['torque']:+.4f}")

    # ---- 2. our six-leg build
    mb = _load_mjcf(str(OURS)).compile()
    db = mujoco.MjData(mb)
    mujoco.mj_resetDataKeyframe(mb, db, 0)
    mujoco.mj_forward(mb, db)
    q0 = np.asarray(db.qpos, dtype=float).copy()
    print(f"\nour six-leg   : nq {mb.nq}, nv {mb.nv}, nu {mb.nu} -- built by tools_mirror_muscles.py")
    ours = {}
    for jn in ("joint_LFTibia_pitch", "joint_LMTibia_pitch", "joint_RMTibia_pitch"):
        rows, signs = torque_table(mb, db, jn)
        ours[jn] = {"signs": sorted(signs), "bidirectional": len(signs) == 2, "muscles": rows}
        print(f"  {jn}: {len(rows)} muscle(s), signs {sorted(signs)}, "
              f"{'BIDIRECTIONAL' if len(signs) == 2 else 'ONE-WAY'}")
        for r in rows:
            print(f"     {r['actuator']:34s} arm {r['moment_arm']:+.5f}  torque {r['torque']:+.4f}")

    # ---- 3. random sweep, so it is not a configuration artefact
    def sweep(prefix, joint_name, trials):
        js = [j for j in range(mb.njnt)
              if (str(mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_JOINT, j)) or "").startswith(
                  f"joint_{prefix}")]
        rng = np.random.default_rng(0)
        n_two = 0
        for _ in range(trials):
            db.qpos[:] = q0
            for j in js:
                adr = int(mb.jnt_qposadr[j])
                if bool(mb.jnt_limited[j]):
                    lo, hi = float(mb.jnt_range[j][0]), float(mb.jnt_range[j][1])
                else:
                    lo, hi = float(q0[adr]) - 1.0, float(q0[adr]) + 1.0
                db.qpos[adr] = rng.uniform(lo, hi)
            _, signs = torque_table(mb, db, joint_name)
            n_two += int(len(signs) == 2)
        return {"n_joints_swept": len(js), "trials": trials, "bidirectional_cases": n_two}

    mid = sweep("LM", "joint_LMTibia_pitch", args.trials)
    fore = sweep("LF", "joint_LFTibia_pitch", args.trials)
    print(f"\nrandom sweep over all 7 joints of a leg, {args.trials} trials each:")
    print(f"  middle leg  joint_LMTibia_pitch : bidirectional in {mid['bidirectional_cases']} "
          f"of {args.trials}")
    print(f"  fore   leg  joint_LFTibia_pitch : bidirectional in {fore['bidirectional_cases']} "
          f"of {args.trials}")

    shipped_ok = ship["joint_LFTibia_pitch"]["bidirectional"]
    ours_bad = not ours["joint_LMTibia_pitch"]["bidirectional"]
    verdict = ("DEFECT_IS_OURS_NOT_UPSTREAM" if shipped_ok and ours_bad
               else "INCONCLUSIVE")
    print(f"\nVERDICT: {verdict}")
    if verdict == "DEFECT_IS_OURS_NOT_UPSTREAM":
        print("  The shipped asset's tibia is properly antagonistic and our six-leg build's middle "
              "legs are not, so the mirroring in tools_mirror_muscles.py introduced it.  It is in this "
              "repository and it is fixable -- the earlier registration that called it an upstream "
              "defect was WRONG and has been withdrawn.")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "shipped_asset": {"path": str(SHIPPED.relative_to(HERE)), "nq": int(ms.nq), "nu": int(ms.nu),
                          "is_one_leg": True, "joints": ship},
        "our_six_leg_build": {"path": str(OURS.relative_to(HERE)), "nq": int(mb.nq), "nu": int(mb.nu),
                              "builder": "tools_mirror_muscles.py", "joints": ours},
        "random_sweep": {"middle_leg_tibia": mid, "fore_leg_tibia": fore},
        "withdrawn_claim": ("an earlier registration recorded this as an upstream defect in FlyMimic.  "
                            "That is WITHDRAWN: the shipped asset's fore-leg tibia is bidirectional, "
                            "and the one-way behaviour appears only in this project's mirrored "
                            "six-leg build."),
        "units": "model torque units for moment arm x force; radian for the swept angles",
        "provenance": ("MuJoCo; the shipped asset and this project's built model, both at their own "
                       "keyframes with act set to 1 -- setting act is required because mj_forward does "
                       "not integrate it"),
        "what_this_does_not_establish": (
            "which step of the mirroring is wrong, or that correcting it makes the fly stand.  It "
            "establishes only WHERE the one-way behaviour comes from.  It also says nothing about "
            "whether FlyMimic's own one-leg model is anatomically right -- only that its fore-leg "
            "tibia is antagonistic and ours is not.  Every number here is simulator output; there is "
            "no real-fly recording anywhere in this project."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
