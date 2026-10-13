#!/usr/bin/env python3
"""BUILD-TIME ACCEPTANCE: no leg joint may be actuated in one direction only.

WHY THIS EXISTS.
    Round 24 measured that 6 of 42 leg joints in our six-leg stack-B model could only be pushed
    one way, and that the defect was ours.  Three blind fix attempts then failed (round 25), and
    the round-9 record's own conclusion was that the next attempt must be JUDGED IMMEDIATELY by a
    build-time criterion rather than by a narrative.  This is that criterion.

    Round 25 then proved the mechanism (diagnostics/diag_moment_arm_algebra.py: a muscle's arm
    about a joint is decided by the single path segment that joint moves) and repaired it
    (tools_repair_moment_arms.py: those two points are re-placed in the joint's own frame to the
    homologous fore-leg muscle's coordinates).  This check is what makes the repair falsifiable:
    run it against the pre-repair model and it must FAIL.

    The check is therefore usable as an injection test:
        python tools_test_bidirectionality.py --xml <pre-repair model> --out /tmp/x.json
    must exit non-zero and name the six joints.

WHAT IT CHECKS, all of which can FAIL.
    A. exactly 42 leg joints exist (non-vacuity);
    B. every leg joint is actuated by at least one muscle with a nonzero arm -- a joint with no
       muscles is NOT "bidirectional", it is unactuated, and that must not hide inside a pass;
    C. every leg joint has BOTH signs available -- one-way is a failure, and the offenders are
       named with the arms of the muscles on them;
    D. on the joints repaired by tools_repair_moment_arms.py, the arms equal the homologous
       fore-leg muscle's arms (the invariant the repair constructs, tolerance 1e-6 mm);
    E. the comparison is not vacuous: at least 200 nonzero (muscle, joint) moments were seen.

UNITS AND SCOPE.  Arms in mm (MuJoCo model lengths are mm here), counts for joints.  Muscle force
is non-negative, so the sign of the torque a muscle can produce is the sign of its arm.  Nothing
here says anything about muscle FORCE, about the neural tier, or about standing: bidirectional
arms are necessary and nowhere near sufficient.  Every number is simulator output; no real-fly
recording exists anywhere in this project.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
REPAIRED = ["joint_LHTibia_pitch", "joint_LMTibia_pitch", "joint_LMTrochanter_pitch",
            "joint_RHTibia_pitch", "joint_RMTibia_pitch", "joint_RMTrochanter_pitch"]
REFERENCE_LEG = {"LM": "LF", "LH": "LF", "RM": "RF", "RH": "RF"}
DEFAULT_XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
DEFAULT_OUT = HERE / "outputs" / "bidirectionality_check.json"


def main(argv: list[str] | None = None) -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    from tools_repair_moment_arms import census, leg_joints

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xml", default=str(DEFAULT_XML))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--repaired-joints", nargs="*", default=REPAIRED)
    a = ap.parse_args(argv)

    xml = Path(a.xml)
    if not xml.exists():
        print(f"FAIL: {xml} does not exist")
        return 2
    m = _load_mjcf(str(xml)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    d.act[:] = 1.0                      # SET: mj_forward does not integrate act
    mujoco.mj_forward(m, d)
    cen = census(m, d)

    n_joints = len(leg_joints(m))
    unactuated = sorted(k for k, v in cen.items() if not v["arms"])
    one_way = sorted(k for k, v in cen.items() if v["one_way"])
    two_way = sorted(k for k, v in cen.items() if len(v["signs"]) == 2)
    n_nz = sum(len(v["arms"]) for v in cen.values())

    # D: the repair invariant, on the repaired joints only
    worst, worst_at, n_cmp = 0.0, None, 0
    for jnm in a.repaired_joints:
        if jnm not in cen:
            continue
        leg, role = jnm[6:8], jnm[8:]
        ref = REFERENCE_LEG.get(leg)
        if ref is None:
            continue
        ref_jnm = f"joint_{ref}{role}"
        if ref_jnm not in cen:
            continue
        for anm, arm in cen[jnm]["arms"].items():
            ref_anm = ref + anm[2:]
            if ref_anm not in cen[ref_jnm]["arms"]:
                continue
            n_cmp += 1
            e = abs(arm - cen[ref_jnm]["arms"][ref_anm])
            if e > worst:
                worst, worst_at = e, f"{anm} x {jnm} vs {ref_anm} x {ref_jnm}"

    check_A = n_joints == 42
    check_B = len(unactuated) == 0
    check_C = len(one_way) == 0
    check_D = n_cmp > 0 and worst <= 1e-6
    check_E = n_nz >= 200

    print(f"A) leg joints found: {n_joints} (need 42) -- {'PASS' if check_A else 'FAIL'}")
    print(f"B) unactuated leg joints (no muscle with a nonzero arm): {len(unactuated)} "
          f"{unactuated} -- {'PASS' if check_B else 'FAIL'}")
    print(f"C) one-way leg joints: {len(one_way)} (need 0); bidirectional {len(two_way)}/42 "
          f"-- {'PASS' if check_C else 'FAIL'}")
    for jnm in one_way:
        print(f"     {jnm}: " + ", ".join(f"{k} {v:+.5f}" for k, v in
                                         sorted(cen[jnm]["arms"].items())))
    print(f"D) repaired-joint arm match: {n_cmp} comparisons, worst {worst:.3e} mm at {worst_at} "
          f"-- {'PASS' if check_D else 'FAIL'}")
    print(f"E) nonzero (muscle, joint) moments seen: {n_nz} (need >=200) "
          f"-- {'PASS' if check_E else 'FAIL'}")

    ok = check_A and check_B and check_C and check_D and check_E
    verdict = ("EVERY_LEG_JOINT_IS_ACTUATED_BOTH_WAYS" if ok else
               "ONE_WAY_LEG_JOINTS_PRESENT" if not check_C else
               "BIDIRECTIONALITY_CHECK_INVALID")
    print(f"\nVERDICT: {verdict}")

    out = {
        "overall": "PASS" if ok else "FAIL",
        "status": "PASS" if ok else "FAIL",
        "verdict": verdict,
        "xml": str(xml),
        "leg_joints": n_joints,
        "unactuated_joints": unactuated,
        "one_way_joints": {k: {kk: vv for kk, vv in cen[k]["arms"].items()} for k in one_way},
        "bidirectional_count": len(two_way),
        "nonzero_muscle_joint_moments": n_nz,
        "repair_invariant": {"repaired_joints": list(a.repaired_joints), "comparisons": n_cmp,
                             "worst_abs_diff_mm": worst, "worst_at": worst_at,
                             "tolerance_mm": 1e-6, "pass": bool(check_D)},
        "checks": {"A_42_joints": bool(check_A), "B_all_actuated": bool(check_B),
                   "C_none_one_way": bool(check_C), "D_repair_invariant": bool(check_D),
                   "E_non_vacuous": bool(check_E)},
        "units": "moment arms mm, counts for joints",
        "provenance": ("MuJoCo 3.9.0; the model compiled at its own keyframe with act set to 1; "
                       "arms read from MuJoCo's own actuator moment and divided by gear; muscle "
                       "force is non-negative so the arm sign is the torque sign"),
        "what_this_does_not_establish": (
            "standing, walking, or anything about muscle force.  Moment arms are pose-dependent "
            "and this is measured at the model's keyframe; tools_repair_moment_arms.py reports how "
            "wide a neighbourhood the repair holds over, and it is not all of configuration "
            "space.  Nothing here concerns the neural tier.  Every number is simulator output; "
            "no real-fly recording exists in this project."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
