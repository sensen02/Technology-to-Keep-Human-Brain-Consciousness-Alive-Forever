#!/usr/bin/env python3
"""RE-ESTABLISH THE MUSCLES' OPERATING RANGE ON THE FINAL MODEL, AFTER ANY GEOMETRY SURGERY.

WHY A SECOND PASS IS NEEDED.
    A muscle's operating point depends on two things that the two repairs each touch:
    its tendon LENGTH (changed by `tools_repair_moment_arms.py`, which re-places 40 attachment
    sites) and its declared BAND `lengthrange`.  MEASURED interaction: after the moment-arm
    repair, the middle leg's tibia flexor moved from +0.238 band-widths to -11.047, i.e. the arm
    fix -- exact on its own terms -- pushed the muscle out of its own force-length curve.  So the
    band has to be re-derived on the FINAL geometry, not on an intermediate one.

THE RULE, AND WHY IT IS THE RIGHT ONE.
    For each muscle, take the fore leg's homologue's band and scale it by the ratio of the two
    tendon lengths measured AT A DECLARED COMMON POSTURE (all six legs set to the fore leg's own
    declared neutral joint angles).  Then, by algebra,

        frac_target = (L_t - lo_f . r) / (w_f . r)   with r = L_t / L_f
                    = (L_f - lo_f) / w_f  =  frac_fore

    so homologous muscles sit at the SAME fractional position in their bands at homologous
    postures -- the invariance that makes the five mirrored legs behave like the one modelled leg.
    The fore leg's own band is the base, never the target's current value, so the rule is
    IDEMPOTENT and order-independent.

    The earlier version of this rule measured the ratio at qpos = 0 -- a fresh MjData's default,
    where every leg joint is zero, a collapsed pose no leg is ever in.  MEASURED consequence: the
    middle leg's trochanter extensor band sat at [0.7544, 0.7958] around a 0.3956-long tendon,
    8.7 band-widths outside its own curve, and 14 muscles produced EXACTLY ZERO force at the
    standing pose -- 5 of the 6 muscles on joint_LMTrochanter_pitch among them.  Pinned by
    `tools_test_muscle_operating_range.py`.

WHAT THIS DOES NOT ESTABLISH.
    That the fly stands, or that muscle force magnitude is adequate, or anything about the moment
    arms or the neural tier.  It puts every leg muscle at the same place in its force-length curve
    as the modelled fore leg at the same posture; the fore leg is itself a mirrored, hand-built
    approximation of a one-leg upstream asset, and none of this is measured anatomy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
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


def pose_neutral(m, d, joint_angles=NEUTRAL) -> int:
    """Set all six legs to the same declared joint angles.  Returns how many joints were posed."""
    import mujoco
    n = 0
    for j in range(m.njnt):
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        if not nm.startswith("joint_") or nm[6:8] not in LEGS:
            continue
        for k, (seg, ax) in enumerate(SEG_JOINTS):
            if nm == f"joint_{nm[6:8]}{seg}_{ax}":
                d.qpos[int(m.jnt_qposadr[j])] = joint_angles[k]
                n += 1
    return n


def plan(m, d, posture: str = "keyframe") -> tuple[dict[str, tuple[float, float]], dict]:
    """Return {actuator name: (lo, hi)} derived from the fore-leg band scaled by length ratio.

    POSTURE MATTERS, AND IT IS NOW EXPLICIT.  The ratio is a function of the pose, so it has to be
    measured at the pose the model actually OPERATES at -- its own keyframe.  MEASURED, and this is
    why the first version of this repair was not enough: the standing builder's keyframe is a
    per-leg IK solution, not the declared neutral posture it is described as -- Trochanter_pitch
    sits at -1.34..-1.64 against a declared -2.8 and Tibia_pitch at +0.71..+1.61 against +2.0 --
    so the middle leg's tibia flexor runs at 0.3956 mm at the keyframe against 1.017 mm at the
    neutral posture.  A band scaled for the neutral posture therefore does not contain the
    operating length, and 14 muscles stayed outside it.

    Both postures are supported so the choice is measured rather than assumed; the default is the
    keyframe, because a muscle's force-length band has to contain the length it operates at.
    """
    import mujoco
    lr = np.asarray(m.actuator_lengthrange)
    out: dict[str, tuple[float, float]] = {}
    report = {"scaled": [], "skipped": [], "posture": posture}
    if posture == "neutral":
        n_posed = pose_neutral(m, d)
        if n_posed != 42:
            raise RuntimeError(f"the common posture must pose 42 leg joints, posed {n_posed}")
    elif posture == "qpos0":
        # THE OLD, WRONG RULE, KEPT SO THE DEFECT IS REPRODUCIBLE.  A fresh MjData sits at qpos=0
        # with every leg joint at zero -- a collapsed pose no leg is ever in -- and measuring the
        # length ratio there is what made 5 of the 6 muscles on joint_LMTrochanter_pitch dead.
        # This mode exists so the guard can be run against the defect instead of against a memory
        # of it: `--posture qpos0` must make tools_test_muscle_operating_range.py FAIL.
        n_posed = 0
    else:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        n_posed = 42
    mujoco.mj_forward(m, d)
    if posture == "keyframe":
        # also record where the neutral posture would put the muscles, so the mismatch between the
        # model's keyframe and its declared springref can never be hidden by this repair.
        _Lk = np.asarray(d.actuator_length).copy()
        _lr = np.asarray(m.actuator_lengthrange).copy()
        _n = pose_neutral(m, d)
        mujoco.mj_forward(m, d)
        _Lc = np.asarray(d.actuator_length).copy()
        report["neutral_posture_fracs"] = {
            str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)):
                float((_Lc[i] - _lr[i, 0]) / max(_lr[i, 1] - _lr[i, 0], 1e-12))
            for i in range(m.nu)}
        report["keyframe_fracs_before"] = {
            str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)):
                float((_Lk[i] - _lr[i, 0]) / max(_lr[i, 1] - _lr[i, 0], 1e-12))
            for i in range(m.nu)}
        report["n_neutral_joints_posed"] = _n
        n_posed = _n
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
    L = np.asarray(d.actuator_length)
    idx = {str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)): i for i in range(m.nu)}
    for nm, i in idx.items():
        if nm[0:2] not in LEGS or nm[0:2] == "LF":
            continue
        ref = "LF" + nm[2:]
        if ref not in idx:
            report["skipped"].append({"muscle": nm, "why": f"no fore-leg homologue {ref}"})
            continue
        j = idx[ref]
        if L[j] <= 0:
            report["skipped"].append({"muscle": nm, "why": "zero-length fore-leg tendon"})
            continue
        r = float(L[i]) / float(L[j])
        lo, hi = float(lr[j, 0]) * r, float(lr[j, 1]) * r
        out[nm] = (lo, hi)
        report["scaled"].append({"muscle": nm, "from": ref, "length_ratio": r,
                                 "band_before": [float(lr[i, 0]), float(lr[i, 1])],
                                 "band_after": [lo, hi],
                                 "frac_before": float((L[i] - lr[i, 0])
                                                      / max(lr[i, 1] - lr[i, 0], 1e-12)),
                                 "frac_after": float((L[i] - lo) / max(hi - lo, 1e-12)),
                                 "frac_fore": float((L[j] - lr[j, 0])
                                                    / max(lr[j, 1] - lr[j, 0], 1e-12))})
    report["n_posed_joints"] = n_posed  # 42 in both postures
    report["n_scaled"] = len(out)
    return out, report


def apply_to_tree(root: ET.Element, bands: dict[str, tuple[float, float]]) -> int:
    n = 0
    for a in root.iter("general"):
        nm = a.get("name")
        if nm in bands and a.get("lengthrange"):
            lo, hi = bands[nm]
            a.set("lengthrange", f"{lo:.6g} {hi:.6g}")
            n += 1
    return n


def verify_plan(report: dict) -> tuple[bool, float]:
    """Does the plan satisfy the invariance it claims, on its own arithmetic?"""
    worst = 0.0
    for r in report["scaled"]:
        worst = max(worst, abs(r["frac_after"] - r["frac_fore"]))
    return (report["n_scaled"] > 0 and worst <= 1e-6), worst


def main(argv: list[str] | None = None) -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--posture", choices=("keyframe", "neutral", "qpos0"),
                    default="keyframe",
                    help="where to measure the length ratio; qpos0 reproduces the OLD\n                          buggy rule and exists so the guard can be injection-tested")
    a = ap.parse_args(argv)
    src, dst = Path(a.src), Path(a.dst)
    m = _load_mjcf(str(src)).compile()
    d = mujoco.MjData(m)
    bands, report = plan(m, d, a.posture)
    ok, worst = verify_plan(report)
    tree = ET.parse(src)
    n = apply_to_tree(tree.getroot(), bands)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    txt = dst.read_text(encoding="utf-8")
    prov = ("\n  MUSCLE OPERATING RANGE RE-DERIVED ON THE FINAL GEOMETRY\n"
            f"  (tools_repair_muscle_lengthrange.py): {n} bands set from the fore-leg homologue's\n"
            "  band scaled by the tendon-length ratio measured at the posture the model OPERATES at\n"
            "  (its own keyframe), so homologous muscles sit at the same fractional position.\n"
            
            "  Derived, not measured anatomy; it does not make the fly stand.\n")
    # XML COMMENTS MAY NOT CONTAIN "--": a conformant parser rejects the whole document.
    # MEASURED BUG, found by the well-formedness guard: a provenance comment carrying
    # "arm -- sign included -- the reference's" made a generated model unparseable by xml.etree,
    # while `_load_mjcf`'s lenient reader accepted it -- so only the tools using a conformant
    # parser failed, with "not well-formed (invalid token)".
    from tools_xml_comment import xml_comment
    if ">\n" in txt:
        txt = txt.replace(">\n", ">" + xml_comment(prov) + "\n", 1)
    dst.write_text(txt, encoding="utf-8")
    print(f"re-derived {n} muscle bands; claimed invariance worst |frac - frac_fore| = {worst:.3e} "
          f"({'OK' if ok else 'VIOLATED'})")
    if a.report:
        Path(a.report).write_text(json.dumps(
            {"status": "PASS" if (ok and n > 0) else "FAIL", "n_bands_written": n,
             "invariance_worst_abs_diff": worst, "plan": report,
             "units": "frac dimensionless band fractions, lengths mm",
             "provenance": ("MuJoCo 3.9.0; the model compiled and posed to the fore leg's declared "
                            "neutral joint angles before measuring tendon lengths"),
             "what_this_does_not_establish": (
                 "that the fly stands, that muscle force magnitude is adequate, or anything about "
                 "the moment arms or the neural tier.  Derived parameters on a hand-built mirrored "
                 "model; no real-fly recording exists in this project.")},
            indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"wrote {a.report}")
    return 0 if (ok and n > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
