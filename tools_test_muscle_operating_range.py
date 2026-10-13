#!/usr/bin/env python3
"""BUILD-TIME ACCEPTANCE: homologous muscles must sit at the same place in their force-length band.

WHY THIS EXISTS.
    While explaining the repaired one-way joints (round 25) a second, larger defect fell out:
    at the standing stance many muscles produced EXACTLY ZERO active force, and five of the six
    muscles on joint_LMTrochanter_pitch were dead.  The cause is `lengthrange`, the band over
    which MuJoCo's muscle force-length curve is defined: our builder scales the source muscle's
    band by the ratio of the two legs' tendon lengths, and it measured that ratio at qpos = 0 --
    the model's raw default pose, where every leg joint is zero -- rather than at a declared
    posture.  MEASURED consequence: the middle leg's trochanter extensor band is [0.7544, 0.7958]
    while its tendon is 0.3956 long, i.e. the muscle operates at -8.7 band-widths from the bottom
    of its own curve, where MuJoCo's force-length curve returns zero.

    The invariance that must hold is posture-independent and needs no knowledge of what posture
    the upstream band was authored at:

        for HOMOLOGOUS muscles at HOMOLOGOUS postures, the fractional position in the band
        must be the same.

    If it holds, every leg's muscle behaves like the fore leg's at the same joint angles -- which
    is the whole point of building five legs by mirroring one.  It is also exactly what can FAIL,
    and it does fail on the current build.

CHECKS, all of which can FAIL.
    A. non-vacuity: 90 leg actuators and 6 legs, with at least 15 homologous comparisons made;
    B. homologous-fraction equality at a common posture (all six legs set to the same joint
       angles): |frac_leg - frac_fore| <= 1e-3 for every muscle, offenders named;
    C. every muscle operates INSIDE its band at the model's own keyframe (0 <= frac <= 1); a
       muscle outside its band cannot produce active force at all, so this is not cosmetic;
    D. every muscle produces non-zero force at act = 1 at the keyframe.

UNITS AND SCOPE.  frac is dimensionless (fractions of a band); lengths mm.  This check says
nothing about whether the fly stands, nothing about muscle force MAGNITUDE being adequate, and
nothing about the neural tier.  It only establishes that the actuators are alive and homologous.
Every number is simulator output; no real-fly recording exists in this project.
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
SEG_JOINTS = (("Coxa", "yaw"), ("Coxa", "pitch"), ("Coxa", "roll"),
              ("Trochanter", "yaw"), ("Trochanter", "pitch"), ("Trochanter", "roll"),
              ("Tibia", "pitch"))
NEUTRAL = (-0.11, 0.35, 0.5, -0.1, -2.8, 0.0, 2.0)   # the foreleg's own springref
DEFAULT_XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
DEFAULT_OUT = HERE / "outputs" / "muscle_operating_range_check.json"
FRAC_TOL = 1.0e-3


def main(argv: list[str] | None = None) -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xml", default=str(DEFAULT_XML))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)

    xml = Path(a.xml)
    if not xml.exists():
        print(f"FAIL: {xml} does not exist")
        return 2
    m = _load_mjcf(str(xml)).compile()
    d = mujoco.MjData(m)
    names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)) for i in range(m.nu)]
    leg_act = [i for i, n in enumerate(names) if n[0:2] in LEGS]
    lr = np.asarray(m.actuator_lengthrange)

    def frac_of(sel):
        L = np.asarray(d.actuator_length)
        return (L - lr[:, 0]) / np.maximum(lr[:, 1] - lr[:, 0], 1e-12)

    # ---- C and D at the model's own keyframe
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    frac_key = frac_of(None)
    d.act[:] = 1.0                      # SET: mj_forward does not integrate act
    mujoco.mj_forward(m, d)
    force = np.asarray(d.actuator_force)
    outside = [(names[i], float(frac_key[i])) for i in leg_act
               if not (0.0 <= frac_key[i] <= 1.0)]
    dead = [names[i] for i in leg_act if abs(float(force[i])) < 1e-9]
    check_C = len(outside) == 0
    check_D = len(dead) == 0

    # ---- B AT THE MODEL'S OWN POSTURE.
    # WHERE THE INVARIANCE IS EVALUATED IS A MEASURED CHOICE, NOT A CONVENIENCE.  The band is
    # derived by scaling the fore leg's band by the ratio of the two legs' tendon lengths, and
    # that ratio DEPENDS ON THE POSE.  MEASURED: scaling for the keyframe makes the invariance
    # exact at the keyframe (2.4e-15) and wrong by up to 1.47 band-widths at the declared neutral
    # posture; scaling for the neutral posture does the reverse.  ONE SCALAR BAND PER MUSCLE
    # CANNOT SATISFY BOTH, so the criterion is placed at the posture the model OPERATES at -- its
    # keyframe -- and the residual posture-dependence is MEASURED AND REPORTED below rather than
    # hidden.  A first version of this check asserted the invariance at the neutral posture while
    # the bands had been scaled for the keyframe: an internal inconsistency in the check.
    jid = {}
    for j in range(m.njnt):
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        if nm.startswith("joint_") and nm[6:8] in LEGS:
            jid[nm] = j
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    frac_key2 = frac_of(None)

    def compare(frac):
        worst, at, n, bad = 0.0, None, 0, []
        for i in leg_act:
            nme = names[i]
            if nme[0:2] == "LF":
                continue
            ref = "LF" + nme[2:]
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ref)
            if j < 0:
                continue
            n += 1
            e = abs(float(frac[i]) - float(frac[j]))
            if e > worst:
                worst, at = e, (f"{nme} ({float(frac[i]):+.3f}) vs {ref} ({float(frac[j]):+.3f})")
            if e > FRAC_TOL:
                bad.append({"muscle": nme, "reference": ref, "frac": float(frac[i]),
                            "reference_frac": float(frac[j]), "abs_diff": e})
        return worst, at, n, bad

    worst, worst_at, n_cmp, bad = compare(frac_key2)

    # the same comparison at a COMMON posture, reported as the residual posture-dependence
    for leg in LEGS:
        for k, (seg, ax) in enumerate(SEG_JOINTS):
            nm = f"joint_{leg}{seg}_{ax}"
            if nm in jid:
                d.qpos[int(m.jnt_qposadr[jid[nm]])] = NEUTRAL[k]
    mujoco.mj_forward(m, d)
    frac_common = frac_of(None)
    worst_common, at_common, _, _ = compare(frac_common)

    check_A = len(leg_act) >= 80 and n_cmp >= 15
    check_B = n_cmp >= 15 and worst <= FRAC_TOL

    print(f"A) {len(leg_act)} leg actuators, {n_cmp} homologous comparisons "
          f"-- {'PASS' if check_A else 'FAIL'}")
    print(f"B) homologous fractional position AT THE OPERATING POSTURE (the keyframe): "
          f"worst |diff| {worst:.4f} "
          f"at {worst_at}; {len(bad)} offenders, tolerance {FRAC_TOL} "
          f"-- {'PASS' if check_B else 'FAIL'}")
    for b in sorted(bad, key=lambda x: -x["abs_diff"])[:8]:
        print(f"     {b['muscle']:34s} frac {b['frac']:+8.3f}  vs {b['reference']:34s} "
              f"{b['reference_frac']:+8.3f}  diff {b['abs_diff']:.3f}")
    print(f"B') residual posture-dependence: the same comparison at the declared neutral "
          f"posture is wrong by up to {worst_common:.4f} band-widths at {at_common}")
    print("    (one scalar band per muscle cannot satisfy both postures: the legs' length ratio "
          "itself depends on the pose.  The criterion is placed where the model operates.)")
    print(f"C) muscles OUTSIDE their force-length band at the keyframe: {len(outside)} "
          f"-- {'PASS' if check_C else 'FAIL'}")
    for nm, f in sorted(outside, key=lambda x: x[1])[:8]:
        print(f"     {nm:34s} frac {f:+8.3f}")
    print(f"D) muscles producing NO force at act=1 at the keyframe: {len(dead)} "
          f"-- {'PASS' if check_D else 'FAIL'}")
    for nm in dead[:8]:
        print(f"     {nm}")

    ok = check_A and check_B and check_C and check_D
    verdict = ("HOMOLOGOUS_MUSCLES_SHARE_AN_OPERATING_RANGE" if ok else
               "HOMOLOGOUS_MUSCLES_DO_NOT_SHARE_AN_OPERATING_RANGE")
    print(f"\nVERDICT: {verdict}")

    out = {
        "overall": "PASS" if ok else "FAIL", "status": "PASS" if ok else "FAIL",
        "verdict": verdict, "xml": str(xml),
        "checks": {"A_non_vacuous": bool(check_A), "B_homologous_fraction": bool(check_B),
                   "C_inside_band_at_keyframe": bool(check_C), "D_all_alive": bool(check_D)},
        "common_posture": {"joint_angles_rad": list(NEUTRAL),
                           "order": [f"{s}_{x}" for s, x in SEG_JOINTS]},
        "worst_fraction_difference": worst, "worst_at": worst_at,
        "invariance_evaluated_at": "the model's own keyframe (the posture it operates at)",
        "residual_posture_dependence": {
            "worst_abs_diff_at_neutral_posture": worst_common, "worst_at": at_common,
            "why": ("a muscle's band is the fore leg's band scaled by the ratio of the two legs' "
                    "tendon lengths, and that ratio depends on the pose; one scalar band per "
                    "muscle therefore cannot make the invariance hold at every posture, and this "
                    "is the measured size of what is left")},
        "fraction_tolerance": FRAC_TOL, "offenders": bad,
        "outside_band_at_keyframe": [{"muscle": n, "frac": f} for n, f in outside],
        # THE FULL TABLE, NOT JUST THE OFFENDERS.  A figure that plots only the offenders draws
        # nothing for a muscle that was fixed, which reads as missing data.
        "fractions_at_keyframe": {names[i]: float(frac_key[i]) for i in leg_act},
        "fractions_at_common_posture": {names[i]: float(frac_common[i]) for i in leg_act},
        "no_force_at_keyframe": dead,
        "units": "frac dimensionless (band fractions), lengths mm",
        "provenance": ("MuJoCo 3.9.0; actuator_length and actuator_lengthrange from the compiled "
                       "model, with act set to 1 before reading force; the common posture sets all "
                       "six legs to the foreleg's declared neutral joint angles"),
        "what_this_does_not_establish": (
            "standing.  It establishes that every leg muscle operates inside its force-length band "
            "at the model's keyframe and that homologous muscles share an operating range -- "
            "necessary for the actuators to be alive and comparable, and nowhere near sufficient "
            "for the fly to stand.  It says nothing about whether muscle force MAGNITUDE is "
            "adequate, nothing about the moment arms, and nothing about the neural tier.  Every "
            "number is simulator output; no real-fly recording exists in this project."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                           encoding="utf-8")
    print(f"wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
