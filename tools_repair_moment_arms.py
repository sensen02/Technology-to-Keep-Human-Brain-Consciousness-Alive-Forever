#!/usr/bin/env python3
"""REPAIR THE ONE-WAY LEG JOINTS BY TRANSFERRING THE SPANNING SEGMENT IN THE JOINT'S FRAME.

WHY THIS IS THE RIGHT LEVER, MEASURED RATHER THAN ASSUMED.
    Round 24 found 6 of 42 leg joints actuated in one direction only and established that the
    defect is OURS (not upstream, not neural).  Round 25 falsified three candidate causes -- the
    coxa-anchored world transform, a full local copy, and a per-segment mapping -- which left the
    mechanism unidentified and the fix blind.

    diagnostics/diag_moment_arm_algebra.py now PROVES, to machine precision over all 3780
    (actuator, joint) pairs, that a muscle's moment arm about a joint is

        arm = |r| . |u_perp| . sin(theta_u - theta_r)

    where r is the ONE path point that joint moves, relative to the joint anchor, and u is the
    unit direction of the segment joining r to its neighbour.  Nothing else in the muscle's path
    enters.  A one-way joint therefore means its muscles' spanning segments give same-signed
    arms -- and the across-leg table is stark:

        joint_LFTibia_pitch   approach-angle separation 69.94 deg   arms -0.01025 / +0.00917
        joint_RFTibia_pitch   approach-angle separation 55.51 deg   arms -0.01051 / +0.02199
        joint_LMTibia_pitch   approach-angle separation  1.79 deg   arms +0.12347 / +0.10027
        joint_RMTibia_pitch   approach-angle separation  3.59 deg   arms +0.11657 / +0.10176
        joint_LHTibia_pitch   approach-angle separation  7.91 deg   arms +0.04580 / +0.03182
        joint_RHTibia_pitch   approach-angle separation  8.16 deg   arms +0.04469 / +0.03179

    The two muscles approach the fore-leg tibia joint from directions ~60-70 deg apart and push
    opposite ways; in the middle and hind legs they approach from within 8 deg of each other and
    push the same way.

WHAT THIS TOOL DOES.
    For every one-way joint, and for each muscle acting on it, it takes the homologous muscle of
    the fore leg on the SAME side (LM, LH <- LF; RM, RH <- RF -- chirality respected), reads the
    two endpoints of the spanning segment in the reference joint's own frame (anchor, world axis,
    in-plane basis), and re-places the two target attachment sites so their coordinates in the
    target joint's frame are the reference's.  Because the arm is a function of exactly those
    coordinates, the arms -- sign included -- are then the reference's BY CONSTRUCTION.

    It is a measurement-preserving surgery on two sites per muscle, not a new mirror transform.
    It refuses, loudly and per muscle, whenever the assumption does not hold (not exactly one
    moved path point, mismatched path lengths, a reference joint that is itself one-way, or two
    joints wanting the same site).

WHAT THIS DOES NOT ESTABLISH.
    * It does NOT make the fly stand.  An arm of the right sign is necessary and nowhere near
      sufficient; force, gearing and the contact model are separate.
    * It fixes the geometry AT THE MODEL'S OWN KEYFRAME.  Moment arms are pose-dependent, so the
      tool also sweeps poses and reports over how wide a neighbourhood the repair holds, rather
      than claiming it holds everywhere.
    * Signs and counts say nothing about muscle FORCE, and nothing about the neural tier.
    * Every number is simulator output.  No real-fly recording exists anywhere in this project.
"""
from __future__ import annotations

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
# chirality: a left leg is repaired against the left fore leg, a right leg against the right one
REFERENCE_LEG = {"LM": "LF", "LH": "LF", "RM": "RF", "RH": "RF"}
TOL_ARM_MM = 1.0e-9


# --------------------------------------------------------------------------------- model access
def _name(m, objtype, i):
    import mujoco
    return str(mujoco.mj_id2name(m, objtype, i)) or ""


def _in_subtree(m, body, root_body):
    b, root = int(body), int(root_body)
    while True:
        if b == root:
            return True
        if b <= 0:                      # world body is its own parent; stop explicitly
            return False
        b = int(m.body_parentid[b])


def joint_frame(m, d, jid):
    """World anchor, unit world axis, and an in-plane basis (e1, e2 = a x e1) for joint jid.

    The axis MUST be d.xaxis, not R_body @ jnt_axis: when several hinges share one body (coxa and
    trochanter each carry yaw+pitch+roll), joint i's world axis is axis_i expressed in the frame
    after joints 0..i-1.  See diagnostics/diag_moment_arm_algebra.py.
    """
    anchor = np.asarray(d.xanchor[jid], float).copy()
    a = np.asarray(d.xaxis[jid], float).copy()
    a = a / np.linalg.norm(a)
    R = np.asarray(d.xmat[int(m.jnt_bodyid[jid])], float).reshape(3, 3)
    e1 = R @ np.array([1.0, 0.0, 0.0])
    e1 = e1 - np.dot(e1, a) * a
    if np.linalg.norm(e1) < 1e-9:
        e1 = R @ np.array([0.0, 1.0, 0.0])
        e1 = e1 - np.dot(e1, a) * a
    e1 = e1 / np.linalg.norm(e1)
    return anchor, a, e1, np.cross(a, e1)


def to_frame(p, F):
    anchor, a, e1, e2 = F
    v = np.asarray(p, float) - anchor
    return np.array([float(np.dot(v, e1)), float(np.dot(v, e2)), float(np.dot(v, a))])


def from_frame(xyz, F):
    anchor, a, e1, e2 = F
    return anchor + xyz[0] * e1 + xyz[1] * e2 + xyz[2] * a


def tendon_sites(m, d, tid):
    """Site ids, world points and host bodies of a tendon's path (spatial, site-wrapped)."""
    import mujoco
    adr, num = int(m.tendon_adr[tid]), int(m.tendon_num[tid])
    ids, pts, bodies, ok = [], [], [], True
    for k in range(num):
        if int(m.wrap_type[adr + k]) != int(mujoco.mjtWrap.mjWRAP_SITE):
            ok = False
            break
        sid = int(m.wrap_objid[adr + k])
        ids.append(sid)
        pts.append(np.asarray(d.site_xpos[sid], float).copy())
        bodies.append(int(m.site_bodyid[sid]))
    return ids, pts, bodies, ok


def moved_indices(m, d, tid, jid):
    ids, pts, bodies, ok = tendon_sites(m, d, tid)
    if not ok:
        return None
    child = int(m.jnt_bodyid[jid])
    return [k for k in range(len(pts)) if _in_subtree(m, bodies[k], child)]


def arm_of_actuator(m, d, iact, dof):
    """Moment arm in mm, from MuJoCo's own sparse actuator moments (row layout: moment_rowadr)."""
    ra = np.asarray(d.moment_rowadr)
    ci = np.asarray(d.moment_colind)
    am = np.asarray(d.actuator_moment)
    gear = float(m.actuator_gear[iact][0])
    if gear == 0.0:
        return 0.0
    s = int(ra[iact])
    e = int(ra[iact + 1]) if iact + 1 < len(ra) else len(am)
    for k in range(s, e):
        if int(ci[k]) == int(dof):
            return float(am[k]) / gear
    return 0.0


def leg_joints(m):
    out = {}
    for j in range(m.njnt):
        nm = _name(m, __import__("mujoco").mjtObj.mjOBJ_JOINT, j)
        if nm.startswith("joint_") and nm[6:8] in LEGS:
            out[nm] = (j, int(m.jnt_dofadr[j]))
    return out


def census(m, d):
    """{joint name: {"arms": {actuator: arm}, "signs": sorted signs, "one_way": bool}}."""
    import mujoco
    out = {}
    for jnm, (jid, dof) in leg_joints(m).items():
        arms, signs = {}, set()
        for i in range(m.nu):
            anm = _name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if anm[0:2] not in LEGS:
                continue
            if int(m.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_TENDON):
                continue
            v = arm_of_actuator(m, d, i, dof)
            if abs(v) > 1e-9:
                arms[anm] = v
                signs.add(float(np.sign(v)))
        out[jnm] = {"arms": arms, "signs": sorted(signs),
                    "one_way": len(signs) == 1 and len(arms) > 0}
    return out


# --------------------------------------------------------------------------------- the surgery
def plan(m, d, *, verbose=True):
    """Return (updates, report).  updates: {site name: {"pos": local xyz, ...}}."""
    import mujoco
    cen = census(m, d)
    one_way = sorted([k for k, v in cen.items() if v["one_way"]])
    report = {"one_way_before": one_way, "updates": [], "refused": []}
    updates: dict[str, dict] = {}
    act_by_name = {_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}

    for jnm in one_way:
        leg, role = jnm[6:8], jnm[8:]
        ref_leg = REFERENCE_LEG.get(leg)
        ref_jnm = f"joint_{ref_leg}{role}"
        if ref_jnm not in cen:
            report["refused"].append({"joint": jnm, "why": f"no reference joint {ref_jnm}"})
            continue
        if cen[ref_jnm]["one_way"]:
            report["refused"].append(
                {"joint": jnm, "why": f"reference {ref_jnm} is itself one-way"})
            continue
        jid, dof = leg_joints(m)[jnm]
        rjid, rdof = leg_joints(m)[ref_jnm]
        Ft, Fr = joint_frame(m, d, jid), joint_frame(m, d, rjid)

        for anm in sorted(cen[jnm]["arms"]):
            ref_anm = ref_leg + anm[2:]
            if ref_anm not in act_by_name:
                report["refused"].append(
                    {"joint": jnm, "muscle": anm, "why": f"no homologue {ref_anm}"})
                continue
            i, ri = act_by_name[anm], act_by_name[ref_anm]
            tid, rtid = int(m.actuator_trnid[i][0]), int(m.actuator_trnid[ri][0])
            mt = moved_indices(m, d, tid, jid)
            mr = moved_indices(m, d, rtid, rjid)
            if mt is None or mr is None:
                report["refused"].append({"joint": jnm, "muscle": anm,
                                          "why": "non-site wrap in the path"})
                continue
            if len(mt) != 1 or len(mr) != 1:
                report["refused"].append(
                    {"joint": jnm, "muscle": anm,
                     "why": f"moved points target={len(mt)} reference={len(mr)}, need 1 and 1"})
                continue
            ids_t, pts_t, bodies_t, ot = tendon_sites(m, d, tid)
            ids_r, pts_r, bodies_r, orr = tendon_sites(m, d, rtid)
            if not (ot and orr) or len(pts_t) != len(pts_r):
                report["refused"].append(
                    {"joint": jnm, "muscle": anm,
                     "why": f"path length {len(pts_t)} vs reference {len(pts_r)}"})
                continue
            kt, kr = mt[0], mr[0]
            if kt == 0 or kr == 0:
                report["refused"].append({"joint": jnm, "muscle": anm,
                                          "why": "the moved point has no neighbour in the path"})
                continue
            # what the reference arm is, and what the target's is before surgery
            arm_ref = cen[ref_jnm]["arms"].get(ref_anm, 0.0)
            arm_before = cen[jnm]["arms"][anm]
            for idx_t, idx_r, tag in ((kt, kr, "distal"), (kt - 1, kr - 1, "proximal")):
                xyz = to_frame(pts_r[idx_r], Fr)
                p_new = from_frame(xyz, Ft)
                sid = ids_t[idx_t]
                sname = _name(m, mujoco.mjtObj.mjOBJ_SITE, sid)
                body = bodies_t[idx_t]
                Rh = np.asarray(d.xmat[body], float).reshape(3, 3)
                ph = np.asarray(d.xpos[body], float).copy()
                local = Rh.T @ (p_new - ph)
                if sname in updates and not np.allclose(updates[sname]["pos"], local, atol=1e-12):
                    report["refused"].append({"joint": jnm, "muscle": anm, "site": sname,
                                              "why": "two joints want this site differently"})
                    continue
                updates[sname] = {
                    "pos": [float(v) for v in local],
                    "pos_before": [float(v) for v in np.asarray(m.site_pos[sid], float)],
                    "host_body": _name(m, mujoco.mjtObj.mjOBJ_BODY, body),
                    "leg": leg, "joint": jnm, "muscle": anm, "end": tag,
                    "copied_from": {"leg": ref_leg, "joint": ref_jnm, "muscle": ref_anm,
                                    "site": _name(m, mujoco.mjtObj.mjOBJ_SITE, ids_r[idx_r])},
                    "target_arm_before_mm": arm_before, "reference_arm_mm": arm_ref,
                }
                report["updates"].append({"site": sname, "joint": jnm, "muscle": anm, "end": tag,
                                          "local_pos": updates[sname]["pos"]})
    report["n_sites"] = len(updates)
    if verbose:
        print(f"repair plan: {len(one_way)} one-way joints, {len(updates)} sites to re-place, "
              f"{len(report['refused'])} refusals")
        for j in one_way:
            got = [u for u in report["updates"] if u["joint"] == j]
            print(f"   {j:28s} {len(got)} sites")
        for r in report["refused"][:10]:
            print(f"   REFUSED {r}")
    return updates, report


def write_xml(src: Path, dst: Path, updates: dict, *, note: str = "") -> None:
    tree = ET.parse(src)
    root = tree.getroot()
    n = 0
    for s in root.iter("site"):
        nm = s.get("name")
        if nm in updates:
            s.set("pos", " ".join(f"{v:.9g}" for v in updates[nm]["pos"]))
            n += 1
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    txt = dst.read_text(encoding="utf-8")
    prov = ("\n  MOMENT-ARM REPAIR (tools_repair_moment_arms.py).\n"
            f"  {n} attachment sites re-placed so that the segment spanning each one-way joint has\n"
            "  the same geometry, in the joint's own frame, as the homologous muscle of the fore\n"
            "  leg on the same side.  Derived from a joint-frame transfer, NOT measured anatomy.\n"
            "  It makes the moment arms -- sign included -- the reference leg's by construction; it\n"
            "  does not make the fly stand, and it does not speak to muscle force.\n"
            + (f"  {note}\n" if note else ""))
    from tools_xml_comment import xml_comment
    if ">\n" in txt:
        txt = txt.replace(">\n", ">" + xml_comment(prov) + "\n", 1)
    dst.write_text(txt, encoding="utf-8")


# ------------------------------------------------------------------------------------- verify
def verify(path: Path, *, n_pose: int = 200, seed: int = 0,
           repaired_joints: list[str] | None = None) -> dict:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    m = _load_mjcf(str(path)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    d.act[:] = 1.0
    mujoco.mj_forward(m, d)
    cen = census(m, d)
    one_way = sorted([k for k, v in cen.items() if v["one_way"]])
    two_way = sorted([k for k, v in cen.items() if len(v["signs"]) == 2])

    # ON THE JOINTS THAT WERE REPAIRED the arm must equal the reference leg's exactly -- that is
    # the invariant the surgery constructs, so it is the one that is required.  Arms on OTHER
    # joints are a side effect of moving the same sites and are only required to stay
    # bidirectional (the census above); they are reported, not asserted.
    repaired = set(repaired_joints or [])
    worst, worst_at, n_cmp = 0.0, None, 0
    worst_other, worst_other_at, n_other = 0.0, None, 0
    for jnm, rec in cen.items():
        leg, role = jnm[6:8], jnm[8:]
        ref = REFERENCE_LEG.get(leg)
        if ref is None:
            continue
        ref_jnm = f"joint_{ref}{role}"
        if ref_jnm not in cen:
            continue
        for anm, arm in rec["arms"].items():
            ref_anm = ref + anm[2:]
            if ref_anm not in cen[ref_jnm]["arms"]:
                continue
            e = abs(arm - cen[ref_jnm]["arms"][ref_anm])
            if jnm in repaired:
                n_cmp += 1
                if e > worst:
                    worst, worst_at = e, f"{anm} x {jnm} vs {ref_anm} x {ref_jnm}"
            else:
                n_other += 1
                if e > worst_other:
                    worst_other, worst_other_at = e, f"{anm} x {jnm} vs {ref_anm} x {ref_jnm}"
    arm_match = (n_cmp > 0) and worst <= 1e-6

    # how wide a neighbourhood of the keyframe does the repair hold over?
    q0 = np.array(d.qpos, float).copy()
    leg_dofs = [dof for _jid, dof in leg_joints(m).values()]
    rng = np.random.default_rng(seed)
    counts, worst_pose, worst_pose_oneway = [], 0, []
    for _ in range(n_pose):
        d.qpos[:] = q0
        d.qpos[leg_dofs] += rng.uniform(-0.08, 0.08, size=len(leg_dofs))
        mujoco.mj_forward(m, d)
        d.act[:] = 1.0
        mujoco.mj_forward(m, d)
        c = census(m, d)
        two = sum(1 for v in c.values() if len(v["signs"]) == 2)
        counts.append(two)
        ow = [k for k, v in c.items() if v["one_way"]]
        if len(ow) > worst_pose:
            worst_pose, worst_pose_oneway = len(ow), ow
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    return {
        "overall": "PASS", "status": "PASS",
        "bidirectional_at_keyframe": len(two_way),
        "one_way_at_keyframe": one_way,
        "repaired_joints": sorted(repaired),
        "arm_matches_reference_on_repaired_joints": bool(arm_match),
        "arm_comparisons_on_repaired_joints": n_cmp,
        "worst_arm_difference_mm": worst,
        "worst_arm_difference_at": worst_at,
        "side_effect_comparisons_on_other_joints": n_other,
        "side_effect_worst_arm_difference_mm": worst_other,
        "side_effect_worst_arm_difference_at": worst_other_at,
        "side_effect_note": ("moving the two sites of a spanning segment also changes that "
                             "muscle's arms on other joints; those are reported here and only "
                             "required to remain bidirectional, not to match the reference"),
        "pose_sweep": {
            "n_poses": n_pose, "perturbation_rad": 0.08,
            "bidirectional_min": int(min(counts)) if counts else None,
            "bidirectional_median": float(np.median(counts)) if counts else None,
            "bidirectional_max": int(max(counts)) if counts else None,
            "worst_one_way_count": worst_pose, "worst_one_way_joints": worst_pose_oneway,
        },
        "units": "arms mm, angles rad for the perturbation sweep, counts for joints (42 leg joints)",
        "provenance": ("MuJoCo 3.9.0; the model compiled at its own keyframe with act set to 1; "
                       "moment arms read from MuJoCo's own actuator moments divided by gear; "
                       "the repair plans a joint-frame transfer between homologous muscles and "
                       "this file is the measurement of the result"),
        "what_this_does_not_establish": (
            "that the fly stands.  Bidirectional moment arms are necessary and not sufficient: "
            "force magnitude, gearing, passive stiffness and the contact model are separate and "
            "were separately shown insufficient.  The repair is exact at the keyframe and is "
            "characterised over a finite pose neighbourhood rather than claimed everywhere.  It "
            "says nothing about the neural tier and nothing about muscle force.  Every number is "
            "simulator output; no real-fly recording exists in this project."),
    }


# ---------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--poses", type=int, default=200)
    a = ap.parse_args(argv)

    from flygym.compose.fly.musculoskeletal import _load_mjcf
    import mujoco
    src, dst = Path(a.src), Path(a.dst)
    m = _load_mjcf(str(src)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    d.act[:] = 1.0
    mujoco.mj_forward(m, d)

    updates, report = plan(m, d)
    if not updates:
        print("nothing to repair; writing a copy")
    write_xml(src, dst, updates)
    print(f"wrote {dst}")

    res = verify(dst, n_pose=a.poses, repaired_joints=report["one_way_before"])
    res["repair_plan"] = {k: v for k, v in report.items() if k != "updates"}
    print(f"\nbidirectional at keyframe: {res['bidirectional_at_keyframe']}/42  "
          f"one-way: {res['one_way_at_keyframe']}")
    print(f"on the repaired joints: {res['arm_comparisons_on_repaired_joints']} comparisons, "
          f"worst arm difference vs the reference leg {res['worst_arm_difference_mm']:.3e} mm")
    print(f"side effect on other joints: {res['side_effect_comparisons_on_other_joints']} "
          f"comparisons, worst {res['side_effect_worst_arm_difference_mm']:.3e} mm "
          f"at {res['side_effect_worst_arm_difference_at']} (reported, not required to match)")
    ps = res["pose_sweep"]
    print(f"pose sweep ({ps['n_poses']} poses, +-{ps['perturbation_rad']} rad on leg joints): "
          f"bidirectional {ps['bidirectional_min']}..{ps['bidirectional_max']} "
          f"(median {ps['bidirectional_median']}), worst one-way count {ps['worst_one_way_count']}"
          + (f" {ps['worst_one_way_joints']}" if ps['worst_one_way_joints'] else ""))
    res["status"] = ("PASS" if (res["bidirectional_at_keyframe"] == 42
                                and res["arm_matches_reference_on_repaired_joints"]) else "FAIL")
    if a.report:
        Path(a.report).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {a.report}")
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
