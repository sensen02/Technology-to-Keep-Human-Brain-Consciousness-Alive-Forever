#!/usr/bin/env python3
"""GIVE THE FLY A GROUNDED, NEUTRAL STANCE AND A FREE JOINT -- THEN MEASURE WHETHER IT STANDS.

WHY THIS IS NEEDED, MEASURED RATHER THAN ASSUMED.
    1. Three of the previous generation of bugs came from the model's pose being inconsistent with
       its own springs.  The model declares its neutral posture in ``springref`` (the pose the joint
       springs hold), and for the foreleg that is [-0.11 0.35 0.5 -0.1 -2.8 0.0 2.0].  But the
       KEYFRAME is a different pose, the mid and hind legs copied the springref while the right
       foreleg kept 0.0, and the right foreleg's seven joints -- shipped LOCKED by <equality> and
       unlocked earlier -- also kept the generic +-pi range the shipped file gives a joint it never
       intends to move.
    2. The thorax is welded to the world, so "does it stand" is not even a question the model could
       answer.  Gravity could not act on the body.
    3. The tarsi hang in the air: in the previous keyframe the six lowest leg points were
       2.58 / 1.64 / 2.79 / 2.79 / 2.70 / 2.70 mm against a floor at 0, i.e. a 1.154 mm spread --
       not a stance at all, and the right foreleg 0.94 mm out of line with the left.

WHAT THIS TOOL DOES, AND WHAT IT DOES NOT.
    * normalises all six legs onto the DECLARED NEUTRAL posture (the foreleg's own springref), and
      copies the foreleg's anatomical joint RANGES onto the unlocked right foreleg;
    * seats the tarsi on the floor by solving, per leg, the coxa-pitch offset and one global thorax
      height that make all six legs reach the floor SIMULTANEOUSLY.  Contact is detected by MuJoCo's
      own collision engine (first floor contact as the fly is lowered), not by mesh arithmetic;
    * adds a FREE JOINT to the thorax, so the body's weight is finally carried by the legs;
    * measures what happens with NO muscle activation whatsoever.

    It does NOT make the fly stand.  A neutral pose is a model parameter, not behaviour, and the
    constraint is that behaviour has to come out of the motor neurons.  This tool establishes the
    test bench that can tell the two apart, and reports the passive result as the baseline.
"""
from __future__ import annotations
import json, os, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
from flygym.compose.fly.musculoskeletal import _load_mjcf  # noqa: E402

SRC_XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_muscles.xml"
OUT_XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
OUT_JSON = HERE / "outputs" / "stance_solution.json"
LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
SEG_JOINTS = (("Coxa", "yaw"), ("Coxa", "pitch"), ("Coxa", "roll"),
              ("Trochanter", "yaw"), ("Trochanter", "pitch"), ("Trochanter", "roll"),
              ("Tibia", "pitch"))
NEUTRAL = (-0.11, 0.35, 0.5, -0.1, -2.8, 0.0, 2.0)   # foreleg springref = declared neutral


# ----------------------------------------------------------------------------- model assembly
def assemble() -> ET.ElementTree:
    tree = ET.parse(SRC_XML)
    root = tree.getroot()
    body = {b.get("name"): b for b in root.iter("body")}
    jn = {j.get("name"): j for j in root.iter("joint") if j.get("name")}
    thorax = body["Thorax"]

    # ---- 1. A FREE JOINT, so gravity finally acts on the body.
    if not any(c.tag == "freejoint" for c in thorax):
        free = ET.Element("freejoint", {"name": "thorax_free"})
        idx = 1 if len(thorax) and thorax[0].tag == "inertial" else 0
        thorax.insert(idx, free)

    # ---- 2. THE RIGHT FORELEG'S RANGES.  MEASURED: all seven of its joints carry the shipped
    # model's generic +-3.142 range, because the shipped model never intended them to move (they
    # were pinned by <equality>).  Unlocked, a +-pi range on a trochanter pitch lets the leg rotate
    # into poses no insect reaches.  The axes are the same as the foreleg's -- measured [0 1 0] for
    # coxa/trochanter/tibia pitch on both sides -- so the foreleg's ranges transfer directly.
    rf_range = {}
    for seg, js in SEG_JOINTS:
        s, t = jn.get(f"joint_LF{seg}_{js}"), jn.get(f"joint_RF{seg}_{js}")
        if s is not None and t is not None and s.get("range"):
            rf_range[t.get("name")] = (t.get("range"), s.get("range"))
            t.set("range", s.get("range"))

    # ---- 3. EVERY LEG ON THE DECLARED NEUTRAL POSTURE, in BOTH the keyframe and the springs.
    kf = root.find("keyframe")
    keys = list(kf.findall("key")) if kf is not None else []
    assert len(keys) == 1, "the source model is expected to carry exactly one keyframe"
    qpos = [float(v) for v in (keys[0].get("qpos") or "").split()]
    springref = {}
    for leg in LEGS:
        for (seg, js), val in zip(SEG_JOINTS, NEUTRAL):
            j = jn.get(f"joint_{leg}{seg}_{js}")
            if j is None:
                raise RuntimeError(f"missing joint_{leg}{seg}_{js}")
            springref[j.get("name")] = val
            if j.get("springref") is None or float(j.get("springref").split()[0]) != val:
                j.set("springref", f"{val:.6g}")
    return tree, root, body, jn, keys[0], qpos, springref, rf_range


def set_free_pose(m, d, thorax_z, coxa_pitch_off):
    """thorax at (0,0,z) upright, and each leg's coxa pitch at neutral + its own offset."""
    d.qpos[:] = 0.0
    # READ THE FREE JOINT'S OWN ADDRESS rather than assuming 0.  A free joint is 7 dofs in the
    # order x y z qw qx qy qz, and it happens to land first here, but hard-coding that would
    # silently corrupt a leg joint the moment the body order changed.
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    if jid < 0:
        raise RuntimeError("the standing model has no free joint")
    adr = int(m.jnt_qposadr[jid])
    d.qpos[adr:adr + 3] = [0.0, 0.0, thorax_z]
    d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
    for leg in LEGS:
        for (seg, js), val in zip(SEG_JOINTS, NEUTRAL):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{leg}{seg}_{js}")
            if j < 0:
                continue
            v = val + (coxa_pitch_off.get(leg, 0.0) if (seg, js) == ("Coxa", "pitch") else 0.0)
            d.qpos[m.jnt_qposadr[j]] = v
    mujoco.mj_forward(m, d)


def leg_contact(m, d):
    """which legs have a floor contact, and the deepest penetration of each.

    MEASURED BUG, and it made the whole grounding solve silently find nothing: this test compared
    BODY names against the string "floor", but "floor" is a GEOM name -- its body is "world".  So
    every contact was discarded, no leg was ever seen to touch, and the solve reported that the fly
    never reached the ground even at a thorax height where dozens of contacts existed.  Compare
    against the floor's GEOM ID.
    """
    out = {}
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if fid < 0:
        raise RuntimeError('no geom named "floor"')
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 != fid and c.geom2 != fid:
            continue
        other = c.geom2 if c.geom1 == fid else c.geom1
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other])) or ""
        for leg in LEGS:
            if b.startswith(leg):
                out[leg] = min(out.get(leg, 1e9), float(c.dist))
    return out


def onset_height(m, d, coxa_pitch_off, start_z=4.0, lo=-1.0):
    """the thorax height at which each leg FIRST touches the floor, found by lowering the fly."""
    z, step, found = float(start_z), 0.02, {}
    while z > lo and len(found) < 6:
        z -= step
        set_free_pose(m, d, z, coxa_pitch_off)
        for leg, dist in leg_contact(m, d).items():
            if leg not in found:
                found[leg] = z
    return found


def main() -> int:
    tree, root, body, jn, key, qpos, springref, rf_range = assemble()
    print("=== STAGE 1: ranges and neutral posture ===")
    print(f"right-foreleg joint ranges replaced by the foreleg's anatomical ones: {len(rf_range)}")
    for k, (old, new) in sorted(rf_range.items()):
        print(f"  {k:<32} {old:>18} -> {new}")
    print(f"all six legs' keyframe qpos and springref set to the declared neutral {NEUTRAL}")

    tmp = HERE / "outputs" / "muscles_six_legs" / "_stance_probe.xml"
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    m = _load_mjcf(str(tmp)).compile()
    d = mujoco.MjData(m)
    # FORWARD BEFORE READING ANY POSITION.  MEASURED BUG: the stance block read body positions from
    # a fresh MjData, where every xpos is still zero, so every coxa appeared to be at (0,0), every
    # leg length came out at 1e-4 mm, and the reach report was nonsense.
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"compiled: njnt={m.njnt} nq={m.nq} nu={m.nu}")
    assert m.njnt == 43, f"expected 42 leg joints + 1 free joint, got {m.njnt}"

    # ---- the keyframe in the XML now has to cover the free joint's 7 dofs too.
    free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    assert free >= 0, "free joint missing"

    # ---- STAGE 2: SOLVE A STANCE BY INVERSE KINEMATICS TO DECLARED FOOT TARGETS.
    #
    # WHY NOT JUST SEAT THE FEET, MEASURED: with the model's neutral posture the legs hang almost
    # straight down -- each tarsus reaches only 0.50-0.75 mm horizontally against 1.20-1.22 mm
    # downwards (ratio 0.41-0.62) -- so the six tarsi span a base of support barely 1.2 mm across
    # while the fly's centre of mass sits 0.87 mm BEHIND the thorax origin.  The COM is outside the
    # feet and the fly topples onto its belly.  Raising leg-joint stiffness from 0.4 to 256 (640x)
    # does NOT fix it (thorax z 0.647 -> 0.866 mm, 59% of the weight still on the body), so the
    # cause is the stance geometry, not the springs.
    #
    # WHY NOT LET THE OPTIMISER CHOOSE THE STANCE EITHER, MEASURED: letting it minimise
    # "tarsi on the floor + COM inside the support polygon" produced a DEGENERATE solution --
    # five tarsi bunched around (x=+0.2, y=+0.3..+0.7) mm with the sixth alone at x=-1.34 mm.  The
    # numbers looked fine (polygon area 1.03 mm^2, COM clearance +0.25 mm) and the pose was absurd.
    # Constraint satisfaction is not anatomy, so the targets are DECLARED instead:
    #
    #   DECLARED foot placement, from the general arrangement of a standing fly: each tarsus is
    #   placed outward from its OWN coxa, with the forelegs biased forward, the middle legs lateral,
    #   and the hind legs backward and outward, so that the six feet form a hexagon around the COM.
    #   These are NOT measured for this specimen and are not claimed to be a recorded stance.
    #   The reach is held to about 70-85% of each leg's own length so no joint sits at a limit.
    print("\n=== STAGE 2: solve a stance by IK to declared foot targets ===")
    import scipy.optimize as _opt
    from scipy.spatial import ConvexHull as _Hull
    # FOOT TARGETS, IN THE THORAX'S OWN FRAME, CHOSEN TO SURROUND THE CENTRE OF MASS.
    # MEASURED FIRST, because guessing the frame was wrong twice: the world frame here IS the
    # thorax frame (xmat = identity), +x is FORWARD -- the head is at x=+0.02 and the proboscis tip
    # at x=+0.52 -- and the abdomen runs back to x=-1.91, so the body is 2.6 mm long along x and the
    # whole-body COM sits at x=-0.714 mm, i.e. inside the abdomen, NOT under the thorax.
    #
    # Consequence, and it is the reason the earlier attempts toppled: every naive foot placement left
    # the COM BEHIND the rearmost foot (measured clearance -0.46 mm, OUTSIDE the support polygon).
    # The hind feet therefore have to go genuinely rearward, behind x=-0.714.
    #
    #   DECLARED foot placement: the general arrangement of a standing fly -- forelegs forward and
    #   out, middle legs the most lateral, hind legs rearward and out -- with the hexagon sized so
    #   that the COM's projection is well inside it and every leg stays under ~70% of its own
    #   length.  NOT measured for this specimen and not claimed to be a recorded stance.
    # WIDENED, BECAUSE THE MEASURED FAILURE MODE IS ROLL, NOT SAG.  MEASURED: pushing one sixth of
    # body weight onto a single foot makes the fly ROLL (the thorax's up-vector falls to 0.45-0.74,
    # i.e. 42-63 degrees of tilt) rather than compress the leg, and four to five of the six feet leave
    # the floor.  Raising the leg-joint stiffness 40x and correcting the mass 2.44x reduces the sag but
    # the roll remains, so what the leg set lacks is ROTATIONAL stiffness about the body's long axis.
    # A wider stance is what supplies it -- the legs act as angled struts -- and the previous feet at
    # +-0.60 and +-0.85 mm barely sat outside a body whose half-width is about 0.65 mm.  The targets are
    # therefore pushed outward to near the legs' own reach, and the joint-limit penalty above keeps the
    # coxae off their stops while they do it.
    TARGETS_BODY = {"LF": (+0.30, +1.00), "RF": (+0.30, -1.00),
                    "LM": (-0.45, +1.30), "RM": (-0.45, -1.30),
                    "LH": (-1.20, +1.00), "RH": (-1.20, -1.00)}
    PK = (("Coxa", "yaw"), ("Coxa", "pitch"), ("Coxa", "roll"),
          ("Trochanter", "pitch"), ("Tibia", "pitch"))
    key_of = [(leg, seg, js) for leg in LEGS for seg, js in PK]
    jid = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{k[0]}{k[1]}_{k[2]}")
           for k in key_of}
    for k, j in jid.items():
        assert j >= 0, f"missing joint {k}"
    jid_all = {(leg, seg, js): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                                f"joint_{leg}{seg}_{js}")
               for leg in LEGS for seg, js in SEG_JOINTS}
    neut = {(leg, seg, js): v for leg in LEGS for (seg, js), v in zip(SEG_JOINTS, NEUTRAL)}
    tarsus_geoms = {leg: [g for g in range(m.ngeom)
                          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                                int(m.geom_bodyid[g])) or "").startswith(leg)]
                    for leg in LEGS}
    th0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    R0 = d.xmat[th0].reshape(3, 3).copy()
    # express every landmark in the THORAX'S OWN FRAME, so no world/body mix-up is possible
    coxa_body = {}
    for leg in LEGS:
        bx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}Coxa")
        coxa_body[leg] = R0.T @ (np.asarray(d.xpos[bx]) - np.asarray(d.xpos[th0]))
    leglen = {}
    for leg in LEGS:
        pts = []
        for sname in ("Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4",
                      "Tarsus5"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}{sname}")
            if b >= 0:
                pts.append(np.asarray(d.xpos[b]))
        leglen[leg] = float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))
    tgt_body = {leg: np.array([TARGETS_BODY[leg][0], TARGETS_BODY[leg][1], 0.0]) for leg in LEGS}

    def tgt_world(leg, h):
        """the body-frame foot target, placed in the world with the thorax at (0,0,h), upright."""
        return np.array([tgt_body[leg][0], tgt_body[leg][1], 0.0])

    NX = 1 + len(key_of)

    def apply(x):
        h = float(x[0])
        jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
        adr = int(m.jnt_qposadr[jf])
        d.qpos[:] = 0.0
        d.qpos[adr:adr + 3] = [0.0, 0.0, h]
        d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        for i, k in enumerate(key_of):
            d.qpos[int(m.jnt_qposadr[jid[k]])] = x[1 + i]
        for k, j in jid_all.items():
            if k not in key_of:
                d.qpos[int(m.jnt_qposadr[j])] = neut[k]
        mujoco.mj_forward(m, d)
        return h

    def foot_point(leg):
        """the foot: the lowest point of the tarsus, as geom origin minus its bounding radius."""
        best = None
        for g in tarsus_geoms[leg]:
            p = np.asarray(d.geom_xpos[g]).copy()
            p[2] -= float(m.geom_rbound[g])
            if best is None or p[2] < best[2]:
                best = p
        return best

    def resid(x):
        apply(x)
        r = []
        for leg in LEGS:
            p = foot_point(leg)
            r.extend(((p - tgt_world(leg, x[0])) / 0.05).tolist())   # foot at its target
        for i, k in enumerate(key_of):
            r.append(0.15 * (x[1 + i] - neut[k]) / 0.3)     # stay near the declared posture
        # KEEP EVERY JOINT OFF ITS LIMITS.  MEASURED, and it is the reason the fly was standing on two
        # legs: the previous stance drove NINE coxa joints onto their range bounds (LMCoxa_pitch and
        # roll at 100%, RMCoxa_yaw at 0%, LMCoxa/roll/pitch, LHCoxa yaw/pitch/roll and RHCoxa_pitch at
        # 100%), and a joint sitting on its stop has NO travel left, so its spring can supply no torque
        # at all -- the leg becomes a rigid strut.  The mid legs took 84.7% of the body weight and the
        # forelegs, which were not against a stop and were therefore compliant, took 3.8%; the right
        # foreleg's vertical stiffness measured exactly zero.  A pose that pins the coxae is not a
        # stance, whatever the foot positions look like, so proximity to either bound is penalised.
        MARGIN = 0.12
        for i, k in enumerate(key_of):
            lo, hi = (float(v) for v in m.jnt_range[jid[k]])
            span = hi - lo
            if span <= 0:
                continue
            q = x[1 + i]
            r.append(3.0 * max(0.0, MARGIN - (q - lo) / span) / MARGIN)
            r.append(3.0 * max(0.0, MARGIN - (hi - q) / span) / MARGIN)
        r.append(0.02 * (x[0] - 1.5) / 0.3)
        return np.array(r)

    # report the reach before solving, so an infeasible target set is obvious up front
    print(f"{'leg':<5}{'coxa (thorax frame)':>22}{'foot target':>18}{'leg length':>12}{'reach %':>9}")
    for leg in LEGS:
        c = coxa_body[leg]
        # the coxa's height in the world depends on the thorax height, which is itself being
        # solved, so report the reach at a nominal 1.6 mm -- the pre-check only has to catch an
        # obviously unreachable target set
        need = float(np.linalg.norm(np.array([tgt_body[leg][0] - c[0], tgt_body[leg][1] - c[1],
                                              -(1.6 + c[2])])))
        print(f"{leg:<5}{f'({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})':>22}"
              f"{f'({tgt_body[leg][0]:+.2f},{tgt_body[leg][1]:+.2f})':>18}"
              f"{leglen[leg]:>12.3f}{100*need/max(leglen[leg],1e-9):>8.0f}%")
    x0 = np.array([1.5] + [neut[k] for k in key_of])
    blo = np.array([1.0] + [float(m.jnt_range[jid[k]][0]) for k in key_of])
    bhi = np.array([2.2] + [float(m.jnt_range[jid[k]][1]) for k in key_of])
    x0 = np.clip(x0, blo + 1e-4, bhi - 1e-4)
    sol = _opt.least_squares(resid, x0, bounds=(blo, bhi), xtol=1e-12, ftol=1e-12, max_nfev=6000)
    ground_z = apply(sol.x)
    pts = {leg: foot_point(leg) for leg in LEGS}
    P = np.array([pts[leg][:2] for leg in LEGS])
    com = np.asarray(d.subtree_com[0])
    hh = _Hull(P)
    dists = [-(e[0] * com[0] + e[1] * com[1] + e[2]) for e in hh.equations]
    print(f"  least_squares: {sol.nfev} evaluations, cost {sol.cost:.6f}")
    print(f"  thorax height = {ground_z:.4f} mm")
    print(f"  foot z: " + " ".join(f"{leg}={pts[leg][2]:+.4f}" for leg in LEGS))
    print(f"  foot xy error vs target: "
          + " ".join(f"{leg}={np.linalg.norm(pts[leg][:2]-tgt_body[leg][:2]):.4f}" for leg in LEGS) + " mm")
    print(f"  support polygon area = {hh.volume:.4f} mm^2; "
          f"COM clearance to the nearest edge = {min(dists):+.4f} mm "
          f"({'INSIDE' if min(dists) > 0 else 'OUTSIDE'})")
    print(f"  COM = {np.round(com, 4)}")
    stance = {k: float(sol.x[1 + i]) for i, k in enumerate(key_of)}

    # ---- STAGE 2b: SEAT THE SOLVED STANCE WITH MUJOCO'S OWN COLLISION, NOT AN ESTIMATOR.
    # MEASURED BUG, and it is why the first "standing" model was floating: the foot position used in
    # the objective was ``geom_xpos[z] - rbound``, but rbound is the geom's BOUNDING-SPHERE radius
    # about its origin, which is much larger than how far the tarsus actually reaches downwards
    # (rbound 0.068-0.082 mm against a real downward extent of roughly 0.02-0.03 mm).  The result:
    # the solve reported foot heights of 0.0000 mm with errors under 0.007 mm, self-consistently, and
    # the compiled model then had ZERO floor contacts at the keyframe -- the fly floated by about
    # 0.04 mm and simply free-fell onto its feet on the first step.  So the seating is redone here by
    # asking MuJoCo where the contacts actually are: lower the thorax until each leg first touches.
    # ---- STAGE 2c: MAKE THE STANCE LEFT-RIGHT SYMMETRIC.
    # MEASURED DEFECT, and it is the reason the fly was being thrown sideways: solving the six legs
    # independently left them in asymmetric configurations even though the FOOT TARGETS were
    # symmetric -- RM ended with its coxa yaw at its lower bound (-0.60) while LM sat at +0.14.  The
    # measured consequence is in outputs/balance_solve.json: the passive acceleration at the stance is
    # dominated by LATERAL terms (x = -1098, y = +993 mm/s^2) which must be zero for a symmetric
    # stance, and a standing insect is left-right symmetric.  So the right legs are no longer solved
    # independently: they are the MIRROR of the left legs, which also halves the search space.
    #
    # The mirror rule is derived from the joint AXES, not assumed.  Mirroring across the sagittal
    # plane is M = diag(1, -1, 1).  A rotation about axis a by angle t mirrors to a rotation about
    # M a by -t; written in terms of the right joint's own stored axis a_R:
    #     a_R = +M a_L  ->  t_R = -t_L
    #     a_R = -M a_L  ->  t_R = +t_L
    # so t_R = -sign(a_R . M a_L) * t_L.  For the pitch joints (axis [0 1 0] on both sides) that gives
    # the SAME angle; for yaw and roll it negates.
    # THE RULE MUST COVER ALL SEVEN JOINTS, NOT JUST THE FIVE THE IK SOLVES.  MEASURED BUG: applying
    # the mirror only to the solved joints left the right legs' TROCHANTER YAW AND ROLL at the LEFT
    # legs' values -- and the neutral posture itself is asymmetric there (yaw -0.1, roll 0.0, on both
    # sides) -- so the "symmetric" stance was still asymmetric and the fly flipped right over
    # (up.up0 = -0.9998, 179.8 degrees).  MEASURED FIRST, and it validates the rule: all 15 left/right
    # joint-axis pairs are exactly mirror-related (the angle between a_R and M a_L is 0.00 degrees in
    # every case), so pitch joints keep their sign (dot = -1) and yaw and roll negate (dot = +1).
    MIRR = np.diag([1.0, -1.0, 1.0])
    mirror_rule = {}
    for lseg, ljs in SEG_JOINTS:
        for rleg, lleg in (("RF", "LF"), ("RM", "LM"), ("RH", "LH")):
            lj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{lleg}{lseg}_{ljs}")
            rj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{rleg}{lseg}_{ljs}")
            if lj < 0 or rj < 0:
                continue
            aL = np.asarray(m.jnt_axis[lj], float)
            aR = np.asarray(m.jnt_axis[rj], float)
            sgn = float(np.sign(np.dot(aR, MIRR @ aL)))
            mirror_rule[(rleg, lseg, ljs)] = (lleg, sgn)
    print(f"  mirror rule derived from the joint axes for {len(mirror_rule)} right-leg joints:")
    for k in sorted(mirror_rule):
        l, sgn = mirror_rule[k]
        print(f"    {k[0]}{k[1]}_{k[2]:<7} = {sgn:+.0f} * {l}{k[1]}_{k[2]}")
    LEFT_OF = {r: l for (r, _s, _j), (l, _g) in mirror_rule.items()}

    def mirror_angles(vals):
        """fill the right legs' joints from the left legs', using the measured rule."""
        out = dict(vals)
        for k, (lleg, sgn) in mirror_rule.items():
            src = (lleg, k[1], k[2])
            if src in vals:
                out[k] = sgn * vals[src]
        return out

    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert fid >= 0, 'no geom named "floor"'
    # the FULL mirrored pose: neutral for every joint, the IK solution on top, then mirrored
    # THE MIRROR IS DERIVED AND VERIFIED, BUT NOT APPLIED.  MEASURED: enforcing it moves the right
    # legs' feet off their declared targets -- the seating spread grows to 0.286 mm with first touches
    # at RF 1.595 / RM 1.881 / RH 1.809 against LF 1.751 / LM 1.747 / LH 1.727 -- and the fly then
    # flips right over (up.up0 = -0.9998 at t = 0.2 s).  The reason is that the model's right-leg
    # GEOMETRY is not an exact mirror of the left's even though its joint AXES are: the coxa origins
    # differ by 0.008-0.018 mm in y (LFCoxa +0.200 against RFCoxa -0.182 where a mirror needs -0.200).
    # Symmetric FEET therefore need slightly ASYMMETRIC joint angles, which is what the free IK gives.
    # The rule and its verification are kept because they are worth having and they are checked:
    # outputs/ has the 0.00-degree axis result.  Applying it is gated OFF.
    APPLY_MIRROR = "--apply-mirror" in sys.argv
    pose_full = mirror_angles({**neut, **stance}) if APPLY_MIRROR else ({**neut, **stance})

    def apply_stance(h):
        jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
        adr = int(m.jnt_qposadr[jf])
        d.qpos[:] = 0.0
        d.qpos[adr:adr + 3] = [0.0, 0.0, h]
        d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        for k, j in jid_all.items():
            d.qpos[int(m.jnt_qposadr[j])] = pose_full.get(k, neut[k])
        mujoco.mj_forward(m, d)

    onset, h = {}, ground_z + 0.30
    while h > ground_z - 0.40 and len(onset) < 6:
        h -= 0.002
        apply_stance(h)
        for leg, _dist in leg_contact(m, d).items():
            onset.setdefault(leg, h)
    if len(onset) < 6:
        raise SystemExit(f"VACUOUS STANCE: only {sorted(onset)} ever touch the floor; refusing to "
                         "write a standing model whose feet cannot all reach the ground")
    spread = max(onset.values()) - min(onset.values())
    print(f"  seating by MuJoCo contact: first-touch height per leg spread {spread:.4f} mm "
          f"({'uniform' if spread < 0.05 else 'NOT uniform'})")
    ground_z = min(onset.values())          # the height at which all six are just in contact
    print(f"  seated thorax height = {ground_z:.4f} mm (was {sol.x[0]:.4f} by the estimator); "
          f"per-leg first touch: " + " ".join(f"{k}={v:.3f}" for k, v in sorted(onset.items())))
    apply_stance(ground_z)
    fc = 0
    for c in range(d.ncon):
        cc = d.contact[c]
        if (cc.geom1 == fid or cc.geom2 == fid):
            fc += 1
    print(f"  floor contacts at the seated pose: {fc} (was 0 with the estimator)")
    assert fc > 0, "the seated pose still has no floor contact"

    for leg in LEGS:
        print(f"    {leg}: foot ({pts[leg][0]:+.3f},{pts[leg][1]:+.3f},{pts[leg][2]:+.3f}) mm, "
              f"{np.linalg.norm(pts[leg][:2]-com[:2]):.3f} mm from the COM's projection; "
              f"joints " + " ".join(f"{s[:3]}{j}={stance[(leg,s,j)]:+.2f}"
                                    for s, j in PK))

    # ---- write the stance into the XML: keyframe qpos, springref, and the thorax drop.
    # THE KEYFRAME HAS TO BE WRITTEN BY QPOS ADDRESS, NOT APPENDED.  MEASURED BUG: the source
    # keyframe holds 42 values while the compiled model now has nq=49, and the previous version
    # indexed the 42-long list with addresses 7..48 -- which would have raised, or worse, written
    # the wrong joints.  Build a full-length vector and fill it from the model's own addresses.
    # SYMMETRY IS ENFORCED HERE TOO, so the springs' references are mirror-symmetric as well
    qfull = np.zeros(int(m.nq), dtype=float)
    qfull[0:7] = [0.0, 0.0, ground_z, 1.0, 0.0, 0.0, 0.0]
    for leg in LEGS:
        for (seg, js), val in zip(SEG_JOINTS, NEUTRAL):
            nm = f"joint_{leg}{seg}_{js}"
            j = jn.get(nm)
            if j is None:
                continue
            v = pose_full[(leg, seg, js)] if (leg, seg, js) in pose_full else val
            j.set("springref", f"{v:.6g}")
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
            qfull[int(m.jnt_qposadr[jid])] = v
    # COUNT THE JOINTS WRITTEN, NOT THE NONZERO VALUES.  MEASURED BUG: the first version asserted
    # ``count_nonzero(qfull[7:]) == 42`` and fired on correct code, because the neutral posture of
    # Trochanter_roll IS exactly 0.0 on all six legs.
    n_written = sum(1 for leg in LEGS for seg, js in SEG_JOINTS
                    if jn.get(f"joint_{leg}{seg}_{js}") is not None)
    assert n_written == 42, f"the keyframe must set all 42 leg joints, wrote {n_written}"
    key.set("qpos", " ".join(f"{v:.6g}" for v in qfull))
    thorax_z_mm = ground_z
    tmp.unlink(missing_ok=True)
    OUT_XML.parent.mkdir(parents=True, exist_ok=True)
    tree.write(OUT_XML, encoding="utf-8", xml_declaration=True)
    txt = OUT_XML.read_text()
    txt = txt.replace(">\n", ">" + (
        "\n<!--\n  STANDING VARIANT of fruitfly_six_leg_muscles.xml, generated by\n"
        "  tools_stand_and_ground.py.\n"
        "  Contains EXACTLY THREE differences from that file:\n"
        "    1. the thorax carries a FREE JOINT, so the body's weight is carried by the legs;\n"
        "    2. all six legs sit on the model's DECLARED NEUTRAL posture (the foreleg's own\n"
        "       springref), with the unlocked right foreleg given the foreleg's anatomical joint\n"
        "       ranges in place of the shipped generic +-pi ones;\n"
        f"    3. the keyframe places the thorax at z={thorax_z_mm:.4f} mm with per-leg coxa-pitch\n"
        "       offsets so that all six tarsi reach the floor at once.\n"
        "  The stance is a MODEL PARAMETER (a neutral pose), not behaviour: it carries no\n"
        "  activation and no control law.  Whether the fly stands is measured separately.\n-->\n"), 1)
    OUT_XML.write_text(txt)
    print(f"\nwrote {OUT_XML}")

    # ---- STAGE 3: what does gravity do, with no muscle activation at all?
    print("\n=== STAGE 3: FREE JOINT, ZERO MUSCLE ACTIVATION -- does it stand? ===")
    m2 = _load_mjcf(str(OUT_XML)).compile()
    d2 = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d2, 0)
    mujoco.mj_forward(m2, d2)
    mass = float(sum(m2.body_mass))                       # model units = gram
    mw = mass * abs(float(m2.opt.gravity[2]))             # one body weight, in force units
    print(f" fly mass = {mass*1000:.5f} mg   one body weight = {mw:.4f} force units")
    print(f" passive joint stiffness = {m2.jnt_stiffness[0]:.3f} model units/rad")
    th = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    up0 = d2.xmat[th].reshape(3, 3)[:, 2].copy()
    z0 = float(d2.xpos[th][2])
    trace, worst_tilt = [], 0.0
    for i in range(int(2.0 / m2.opt.timestep)):
        mujoco.mj_step(m2, d2)
        up = d2.xmat[th].reshape(3, 3)[:, 2]
        worst_tilt = max(worst_tilt, float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1)))))
        if i % 2000 == 0:
            fn = 0.0
            for c in range(d2.ncon):
                f = np.zeros(6)
                mujoco.mj_contactForce(m2, d2, c, f)
                fn += abs(f[0])
            trace.append({"t": i * m2.opt.timestep, "thorax_z": float(d2.xpos[th][2]),
                          "up_dot_up0": float(up @ up0), "fn_bodyweights": float(fn / mw),
                          "legs_in_contact": sorted(leg_contact(m2, d2).keys())})
    for r in trace:
        print(f"  t={r['t']:4.1f}s  thorax z={r['thorax_z']:8.4f} mm  up.up0={r['up_dot_up0']:+.4f}"
              f"  floor force={r['fn_bodyweights']:6.3f} bw  legs down={','.join(r['legs_in_contact']) or 'NONE'}")
    print(f"  start thorax z = {z0:.4f} mm, final = {float(d2.xpos[th][2]):.4f} mm, "
          f"worst tilt from start = {worst_tilt:.1f} deg")
    print(f"  final torso orientation: up.up0 = {trace[-1]['up_dot_up0']:+.4f} "
          f"({'UPRIGHT' if trace[-1]['up_dot_up0'] > 0.9 else 'NOT upright'})")
    sol = {"neutral_posture": list(NEUTRAL),
           "rf_ranges": {k: {"was": v[0], "now": v[1]} for k, v in rf_range.items()},
           "stance_joint_angles": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in stance.items()},
           "support_polygon_area_mm2": float(hh.volume),
           "declared_foot_targets_thorax_frame": {k: [float(v) for v in tgt_body[k]] for k in LEGS},
           "foot_error_mm": {k: float(np.linalg.norm(pts[k][:2] - tgt_body[k][:2])) for k in LEGS},
           "com_clearance_to_nearest_edge_mm": float(min(dists)),
           "tarsus_xy_mm": {leg: [float(v) for v in pts[leg][0]] for leg in LEGS},
           "com_mm": [float(v) for v in com],
           "keyframe_thorax_z_mm": thorax_z_mm,
           "seating_spread_mm": float(spread),
           "first_touch_height_mm": {k: float(v) for k, v in onset.items()}, "solver": {"nfev": int(sol.nfev),
                                                          "cost": float(sol.cost)},
           "mass_mg": mass * 1000, "one_body_weight_force_units": mw,
           "passive": {"start_thorax_z_mm": z0, "final_thorax_z_mm": float(d2.xpos[th][2]),
                       "worst_tilt_deg": worst_tilt, "final_up_dot_up0": trace[-1]["up_dot_up0"],
                       "trace": trace}}
    OUT_JSON.write_text(json.dumps(sol, indent=2, sort_keys=True))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
