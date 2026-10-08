#!/usr/bin/env python3
"""MIRROR the 15 foreleg muscle-tendon units onto the other five legs.

WHY MIRRORING IS THE DEFENSIBLE MOVE, AND WHERE ITS LIMITS ARE.
Ozdi et al. (arXiv 2509.06426, ICLR 2026) state the warrant: "the muscle structure across legs is
nearly identical, with a few exceptions including tergal depressor of the trochanter (TDT) muscles
in the middle legs".  Their model -- the one FlyGym ships -- has "15 muscle-tendon units (MTUs)
per foreleg" and no other project models the remaining legs.  Meanwhile the CONNECTOME already
carries per-leg muscle-specific motor neurons for all six legs (243 of them; see
tools_mn_muscle_map_per_leg.py), so the muscle actuators are the only missing piece.

TWO STRUCTURAL FACTS THAT MAKE THIS NOT A ONE-TO-ONE COPY, both MEASURED from the model:

  1. ONLY THE FORELEGS HAVE A TROCHANTER SEGMENT.  Front legs have nine bodies
     (Coxa, Trochanter, Femur, Tibia, Tarsus1-5); mid and hind legs have eight, with no
     Trochanter at all.  A site declared on LFTrochanter therefore has no counterpart on a mid or
     hind leg and is mapped onto that leg's FEMUR.  That changes the muscle's geometry and the
     change is recorded per muscle rather than hidden.
  2. THE SEGMENTS ARE NOT THE SAME LENGTH ACROSS LEGS.  Measured in the rest pose, the femur runs
     0.705 mm (front) / 0.784 (mid) / 0.836 (hind) and the tibia 0.518 / 0.667 / 0.684.  A purely
     rigid body-to-body transform would keep the muscle at the FRONT leg's absolute offset, so
     each site is additionally scaled by the axial length ratio of the segment it sits on.  The
     scale is applied ISOTROPICALLY within the segment frame, which is a SIMPLIFICATION: a muscle
     whose fibres run across the segment is not scaled the way a lengthwise one should be.

WHAT THE OUTPUT IS: a new MJCF next to the shipped one, with 15 x 6 = 90 muscle-tendon units
(subject to the mapping above), provenance written into the file, and nothing overwritten.

Run:
    OPENBLAS_NUM_THREADS=1 venv/bin/python tools_mirror_muscles.py
"""
from __future__ import annotations
import json, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = (HERE / "venv_body/lib/python3.12/site-packages/flygym/assets/model/"
       "musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml")
OUT_DIR = HERE / "outputs" / "muscles_six_legs"
SEGMENTS = ["Coxa", "Trochanter", "Femur", "Tibia"] + [f"Tarsus{i}" for i in range(1, 6)]
LEG_ORDER = ["LF", "RF", "LM", "RM", "LH", "RH"]     # LF is the modelled one
LEG_PREFIX = {"LF": "LF", "RF": "RF", "LM": "LM", "RM": "RM", "LH": "LH", "RH": "RH"}


def quat2mat(q):
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def parse_tree(root):
    """{body name: (world position, world rotation)} plus {site name: (body, local pos)}.

    Composed from the MJCF tree WITHOUT COMPILING, because compiling needs the mesh files and the
    mesh download is slow -- and none of this needs geometry, only the rest pose of the bones.
    """
    body_of = {}          # name -> element
    parent_of = {}
    world = {}
    sites = {}

    def rec(el, parent_name, p, R):
        name = el.get("name")
        pos = np.asarray([float(v) for v in (el.get("pos") or "0 0 0").split()], dtype=float)
        quat = (el.get("quat") or "1 0 0 0").split()
        pw = p + R @ pos
        Rw = R @ quat2mat(quat)
        if name:
            world[name] = (pw, Rw)
            parent_of[name] = parent_name
            body_of[name] = el
            for s in el.findall("site"):
                sn = s.get("name")
                if sn:
                    sp = np.asarray([float(v) for v in (s.get("pos") or "0 0 0").split()],
                                    dtype=float)
                    sites[sn] = (name, sp)
        for ch in el.findall("body"):
            rec(ch, name, pw, Rw)

    wb = root.find("worldbody")
    for ch in wb.findall("body"):
        rec(ch, None, np.zeros(3), np.eye(3))
    return world, sites, body_of, parent_of


def segment_length(world, leg, seg):
    """Distance from this segment's origin to its child's, in the rest pose (mm)."""
    order = [s for s in SEGMENTS if f"{leg}{s}" in world]
    i = order.index(seg) if seg in order else -1
    if i < 0 or i + 1 >= len(order):
        return None
    a = world[f"{leg}{order[i]}"][0]
    b = world[f"{leg}{order[i + 1]}"][0]
    return float(np.linalg.norm(b - a))


def body_map(target_leg):
    """source body (LF) -> target body, with the mapping kind recorded."""
    m = {}
    for seg in SEGMENTS:
        src = f"LF{seg}"
        tgt = f"{LEG_PREFIX[target_leg]}{seg}"
        if seg == "Trochanter" and tgt not in TARGET_BODIES:
            m[src] = (f"{LEG_PREFIX[target_leg]}Femur", "trochanter->femur (no trochanter on this leg)")
        else:
            m[src] = (tgt, "same segment")
    return m


def main() -> int:
    global TARGET_BODIES
    tree = ET.parse(SRC)
    root = tree.getroot()
    world, sites, body_of, parent_of = parse_tree(root)
    TARGET_BODIES = set(world)

    # ---- report the per-leg inventory and the segment lengths
    print(f"{'leg':<6}{'bodies':>8}{'femur mm':>10}{'tibia mm':>10}")
    for leg in LEG_ORDER:
        n = sum(1 for s in SEGMENTS if f"{leg}{s}" in world)
        fem = segment_length(world, leg, "Femur")
        tib = segment_length(world, leg, "Tibia")
        print(f"{leg:<6}{n:>8}{fem if fem else float('nan'):>10.3f}"
              f"{tib if tib else float('nan'):>10.3f}")

    # ---- collect the source muscles
    tendon_el = root.find("tendon")
    act_el = root.find("actuator")
    muscles = []
    for sp in tendon_el.findall("spatial"):
        refs = [c.get("site") for c in sp.findall("site")]
        muscles.append({"tendon": sp, "name": sp.get("name"), "site_refs": refs})
    acts = {a.get("tendon"): a for a in act_el.findall("general") if a.get("tendon")}
    print(f"\nsource muscles: {len(muscles)}, actuators: {len(acts)}")

    # ---- JOINTS FOR THE LEGS THAT HAVE NONE, AND THE LOCKING PROBLEM
    # MEASURED, and it changes the previous conclusion: the shipped model is articulated ONLY on
    # the LEFT FORELEG.  The right foreleg HAS seven joints but all seven are held at zero by
    # ``<equality>`` constraints, and the mid and hind legs have no joints at all.  So the shipped
    # model has exactly ONE leg that can move, and mirroring muscles onto the other five produces
    # tendons attached to bodies that cannot move.  Both gaps have to be closed together.
    FORELAG_JOINTS = [                       # (source joint name, target body suffix, joint suffix)
        ("joint_LFCoxa_yaw", "Coxa", "yaw"), ("joint_LFCoxa_pitch", "Coxa", "pitch"),
        ("joint_LFCoxa_roll", "Coxa", "roll"),
        ("joint_LFTrochanter_yaw", None, "yaw"), ("joint_LFTrochanter_pitch", None, "pitch"),
        ("joint_LFTrochanter_roll", None, "roll"),
        ("joint_LFTibia_pitch", "Tibia", "pitch"),
    ]
    src_joints = {}
    for j in root.iter("joint"):
        if j.get("name"):
            src_joints[j.get("name")] = j
    # the class-level joint defaults (armature/stiffness/damping) live in <default>; MuJoCo
    # inherits them, so a new joint needs only its own attributes.
    added_joints = []
    for target in LEG_ORDER:
        if target == "LF":
            continue
        if any(f"joint_{target}" in str(j.get("name")) for j in root.iter("joint")):
            continue                          # RF already has joints (they are locked, see below)
        # which body does the trochanter's job on this leg?
        troch_body = f"{LEG_PREFIX[target]}Trochanter"
        if troch_body not in world:
            troch_body = f"{LEG_PREFIX[target]}Femur"
        src_coxa = world["LFCoxa"][0], world["LFCoxa"][1]
        tgt_coxa = world[f"{LEG_PREFIX[target]}Coxa"][0], world[f"{LEG_PREFIX[target]}Coxa"][1]
        R_T = tgt_coxa[1] @ src_coxa[1].T
        for jname, seg_suffix, jsuffix in FORELAG_JOINTS:
            src_j = src_joints.get(jname)
            if src_j is None:
                continue
            src_body = "LFTrochanter" if seg_suffix is None else f"LF{seg_suffix}"
            tgt_body = troch_body if seg_suffix is None else f"{LEG_PREFIX[target]}{seg_suffix}"
            R_s = world[src_body][1]
            R_t = world[tgt_body][1]
            axis = np.asarray([float(v) for v in src_j.get("axis").split()], dtype=float)
            # the axis keeps the same ANATOMICAL meaning, so it goes through the same
            # coxa-anchored transform the muscle sites used and is then expressed in the target
            # body's own frame
            axis_t = R_t.T @ R_T @ R_s @ axis
            new_attrs = {"name": f"joint_{LEG_PREFIX[target]}{seg_suffix or 'Trochanter'}_{jsuffix}",
                         "pos": src_j.get("pos", "0 0 0"),
                         "axis": " ".join(f"{v:.6g}" for v in axis_t)}
            for k in ("range", "springref", "limited", "stiffness", "damping", "armature"):
                if src_j.get(k) is not None:
                    new_attrs[k] = src_j.get(k)
            ET.SubElement(body_of[tgt_body], "joint", new_attrs)
            added_joints.append({"joint": new_attrs["name"], "body": tgt_body,
                                 "from": jname, "axis": new_attrs["axis"],
                                 "range": new_attrs.get("range")})

    # ---- THE RIGHT FORELEG IS LOCKED.  Seven <equality> constraints hold its joints at zero, so
    # it is rigid in the shipped model.  Mirroring muscles onto it is useless while that holds.
    eq = root.find("equality")
    locked = [e.get("joint1") for e in eq.findall("joint")] if eq is not None else []
    unlock = "--unlock-locked-legs" in sys.argv
    if unlock and eq is not None:
        for e in list(eq.findall("joint")):
            eq.remove(e)

    # ---- RESCALE THE MUSCLE LENGTH RANGES, BECAUSE THE GEOMETRY SCALED AND THE PARAMETERS DID
    # NOT.  MEASURED after fixing the unpacking bug: the mirrored tendons now have sensible lengths
    # (RF 0.498 mm against LF 0.498 mm -- an essentially exact mirror) but the mid and hind ones are
    # 0.80 and 0.89 mm where the source operates at 0.47-0.49 mm, i.e. 1.6-1.8x longer, exactly as
    # the segment-length ratios predict.  The muscle's declared lengthrange was still the
    # foreleg's, so the Hill model operated far outside its working range and produced forces
    # 240-580x too large.  Each mirrored muscle's lengthrange is therefore scaled by the ratio of
    # its OWN rest length to its source's -- measured from the compiled model, not assumed.
    rescale = "--rescale-lengthrange" in sys.argv
    added_lengthrange_rescale = 0

    # ---- TURN OFF COLLISION BETWEEN THE FLY'S OWN PARTS, KEEP IT AGAINST THE FLOOR.
    # MEASURED, and this is the bug that made the acoustic-looking 'silenced but still moving'
    # result: at rest this model has SEVEN self-contacts (Thorax vs each Coxa, plus one femur-femur)
    # all with NEGATIVE dist, i.e. the fly's own geoms interpenetrate by 0.026-0.057 mm, and the
    # solver pushes them apart with 403 force units -- 40x the fly's weight.  Every geom in the
    # model is (contype, conaffinity) = (1, 1), so every part can collide with every other part.
    # The legs are therefore driven mainly by the fly's own internal stress, and the muscles (and so
    # the neurons) are a small term next to it -- which is why zeroing ALL muscle activations still
    # left the legs swinging by 1.8-5.4 rad.
    #
    # The standard MuJoCo idiom fixes it by bitmask: a pair collides iff (contype1 & conaffinity2)
    # or (contype2 & conaffinity1).  Floor stays (1, 1); the fly's geoms become (2, 1):
    #     fly vs fly  : (2 & 1) = 0  or  (2 & 1) = 0   -> NO collision
    #     fly vs floor: (2 & 1) = 0  or  (1 & 1) = 1   -> collision
    no_self_contact = "--keep-self-contact" not in sys.argv
    n_filtered = 0
    if no_self_contact:
        for i, b in enumerate(root.iter("body")):
            if (b.get("name") or "") == "ground":
                continue
            for g in b.findall("geom"):
                g.set("contype", "2")
                g.set("conaffinity", "1")
                n_filtered += 1

    decisions = []
    added_sites = added_tendons = added_acts = 0
    for target in LEG_ORDER:
        if target == "LF":
            continue
        bmap = body_map(target)
        for mus in muscles:
            src_act = acts.get(mus["name"])
            if src_act is None:
                raise RuntimeError(f"no actuator for tendon {mus['name']!r}")
            new_name = mus["name"].replace("LF", LEG_PREFIX[target], 1)
            new_sp = ET.SubElement(tendon_el, "spatial", {"name": new_name})
            added_tendons += 1
            for ref in mus["site_refs"]:
                body, local = sites[ref]
                # ---- ONE FORMULATION FOR EVERY SITE: a world transform ANCHORED AT THE COXA.
                # MEASURED, and it is why a body-to-body rotation was not enough: 7 of the 15
                # muscles are THORACIC, and their sites hang on the shared Thorax body.  Mapping
                # those body-to-body is the IDENTITY -- they would not move at all and every
                # thoracic muscle would be inserted at the wrong leg's socket.  Anchoring the
                # transform on the leg's coxa moves shared-body sites correctly, and it handles
                # leg bodies in the same expression.
                src_coxa = world["LFCoxa"][0], world["LFCoxa"][1]
                tgt_coxa = world[f"{LEG_PREFIX[target]}Coxa"][0], world[f"{LEG_PREFIX[target]}Coxa"][1]
                R_T = tgt_coxa[1] @ src_coxa[1].T
                t_T = tgt_coxa[0] - R_T @ src_coxa[0]
                if body in bmap:
                    tgt_body, kind = bmap[body]
                    seg = body[len("LF"):]
                    ls = segment_length(world, "LF", seg)
                    lt = segment_length(world, target,
                                        seg if f"{target}{seg}" in world else "Femur")
                    k = float(lt / ls) if (ls and lt) else 1.0
                else:
                    # a site on a SHARED body (Thorax): no scaling, the body is the same body
                    tgt_body, kind, k = body, "shared body (thorax), coxa-anchored", 1.0
                # UNPACK IN THE ORDER parse_tree RETURNS, WHICH IS (POSITION, ROTATION).
                # MEASURED BUG: this line read ``Rs, ps = world[body]``, i.e. the POSITION was
                # bound to the rotation and the ROTATION to the position.  NumPy did not raise --
                # matrix + scalar and a dot product silently broadcast -- and the result was
                # mirrored attachment points 20-30x too far from their segment, tendons 45x too
                # long, and muscle forces 240-580x too large.  A rigidity assertion below now
                # makes that impossible to miss.
                ps, Rs = world[body]
                pt, Rt = world[tgt_body]
                p_world_src = ps + Rs @ (k * local)
                p_world_tgt = t_T + R_T @ p_world_src
                new_pos = Rt.T @ (p_world_tgt - pt)
                new_site_name = ref.replace("LF", LEG_PREFIX[target], 1)
                # ---- THE ASSERTION THAT ACTUALLY CATCHES A SWAPPED UNPACK.
                # A magnitude test is the WRONG check: a site on the shared Thorax genuinely
                # moves in world space (that is the point -- it has to reach the target leg's
                # socket), so "distance from the body origin is preserved" is false and fires on
                # correct code.  What a swapped unpack breaks is the TYPE: the position vector
                # gets used as a rotation and NumPy broadcasts it silently.  So the check is that
                # whatever is used as a rotation IS a rotation, and that the points are points.
                for _nm, _R in (("Rs", Rs), ("Rt", Rt)):
                    _R = np.asarray(_R, dtype=float)
                    if _R.shape != (3, 3) or abs(np.linalg.det(_R) - 1.0) > 1e-6 \
                            or np.abs(_R @ _R.T - np.eye(3)).max() > 1e-9:
                        _det = np.linalg.det(_R) if _R.shape == (3, 3) else float("nan")
                        raise RuntimeError(
                            f"{_nm} for site {ref} is not a proper rotation (shape {_R.shape}, "
                            f"det {_det}); the body pose was almost certainly unpacked in the "
                            "wrong order")
                for _nm, _v in (("ps", ps), ("pt", pt)):
                    if np.asarray(_v).shape != (3,):
                        raise RuntimeError(f"{_nm} for site {ref} is not a 3-vector: "
                                           f"shape {np.asarray(_v).shape}")
                ET.SubElement(body_of[tgt_body], "site",
                              {"name": new_site_name,
                               "pos": " ".join(f"{v:.6g}" for v in new_pos)})
                added_sites += 1
                ET.SubElement(new_sp, "site", {"site": new_site_name})
                decisions.append({"muscle": new_name, "site": new_site_name,
                                  "mapping": kind, "source_body": body,
                                  "target_body": tgt_body, "scale": round(k, 4),
                                  "moved_mm": round(float(np.linalg.norm(new_pos - local)), 4)})
            new_act = ET.SubElement(act_el, "general", dict(src_act.attrib))
            new_act.set("name", new_name.replace("_tendon", ""))
            new_act.set("tendon", new_name)
            added_acts += 1

    n_sites = sum(1 for _ in root.find("worldbody").iter("site"))
    print(f"added: {added_tendons} tendons, {added_sites} sites, {added_acts} actuators, "
          f"{len(added_joints)} joints")
    # ---- POSE THE ADDED JOINTS IN THE KEYFRAME, OR THE SPRINGS DRAG THE LEGS WITH NO MUSCLES.
    # MEASURED, and this is the real answer to "why does it still move when silenced": every joint
    # I added copied the FORELAG's springref (-2.8 for a trochanter pitch, +2.0 for a tibia pitch,
    # +0.5 for a coxa roll) while the model's keyframe leaves those joints at 0.0.  A joint spring
    # pulls toward its springref, so 28 springs each dragged their leg up to 2.8 rad with ALL muscle
    # activations at zero.  Gravity was ruled out: with gravity off the excursion was 4.02 rad
    # against 4.08 with it on.  It is the springs, and the model simply was not standing in the pose
    # its springs expect.
    #
    # The fix is to pose the mirror: put the source joint's springref into the keyframe for every
    # added joint, so the neutral pose has the mid and hind legs in the same posture as the foreleg
    # and every spring starts at equilibrium.  It also keeps the copied ranges consistent, since the
    # source range contains the source's own springref.
    pose_added = "--pose-added-joints" in sys.argv
    if pose_added and added_joints:
        import os as _os
        _os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco as _mj
        from flygym.compose.fly.musculoskeletal import _load_mjcf as _load
        tmp2 = OUT_DIR / "_pose_probe.xml"
        tree.write(tmp2, encoding="utf-8", xml_declaration=True)
        _m2 = _load(str(tmp2)).compile()
        kf = root.find("keyframe")
        keys = list(kf.findall("key")) if kf is not None else []
        if keys:
            qpos = [float(v) for v in (keys[0].get("qpos") or "").split()]
            for aj in added_joints:
                jid = _mj.mj_name2id(_m2, _mj.mjtObj.mjOBJ_JOINT, aj["joint"])
                if jid < 0:
                    continue
                adr = int(_m2.jnt_qposadr[jid])
                src_ref = src_joints.get(aj["from"])
                ref = float((src_ref.get("springref") or "0").split()[0]) if src_ref is not None \
                    else 0.0
                while len(qpos) <= adr:
                    qpos.append(0.0)
                qpos[adr] = ref
                aj["keyframe_qpos_set_to"] = ref
            keys[0].set("qpos", " ".join(f"{v:.6g}" for v in qpos))
        tmp2.unlink(missing_ok=True)

    if rescale:
        # measure the rest length of every muscle in the freshly generated model, compare against
        # its LF source, and scale the declared range
        import os as _os
        _os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco as _mj
        from flygym.compose.fly.musculoskeletal import _load_mjcf as _load
        tmp = OUT_DIR / "_rescale_probe.xml"
        tree.write(tmp, encoding="utf-8", xml_declaration=True)
        _m = _load(str(tmp)).compile()
        _d = _mj.MjData(_m)
        _mj.mj_forward(_m, _d)
        for a in act_el.findall("general"):
            if a.get("class") != "muscle" or not a.get("lengthrange"):
                continue
            nm = a.get("name")
            src_name = "LF" + nm[2:]
            i = _mj.mj_name2id(_m, _mj.mjtObj.mjOBJ_ACTUATOR, nm)
            j = _mj.mj_name2id(_m, _mj.mjtObj.mjOBJ_ACTUATOR, src_name)
            if i < 0 or j < 0:
                continue
            L, L0 = float(_d.actuator_length[i]), float(_d.actuator_length[j])
            if L0 <= 0:
                continue
            ratio = L / L0
            lo, hi = (float(x) for x in a.get("lengthrange").split())
            a.set("lengthrange", f"{lo * ratio:.6g} {hi * ratio:.6g}")
            added_lengthrange_rescale += 1
        tmp.unlink(missing_ok=True)

    print(f"lengthrange rescaled for {added_lengthrange_rescale} muscles (flag: {rescale})")
    print(f"added joints posed in the keyframe: {sum(1 for a in added_joints if 'keyframe_qpos_set_to' in a)}"
          f" (flag: {pose_added})")
    print(f"fly geoms switched to (contype=2, conaffinity=1) so the fly's parts stop colliding with "
          f"each other but still hit the floor: {n_filtered} geoms")
    print(f"joints that were LOCKED by equality constraints in the source: {len(locked)} "
          f"({sorted(set(str(x) for x in locked))[:3]}...)")
    print(f"unlock flag given: {unlock}")
    print(f"non-one-to-one site mappings: {len(decisions)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "fruitfly_six_leg_muscles.xml"
    # PROVENANCE GOES IN AN XML COMMENT, NOT A ROOT ATTRIBUTE.  MEASURED: MuJoCo rejects a custom
    # attribute on <mujoco> with "Schema violation: unrecognized attribute: 'provenance'", which
    # would have made the generated model unloadable.  A comment is legal XML and survives parsing.
    tree.write(dest, encoding="utf-8", xml_declaration=True)
    txt = dest.read_text()
    prov = ("\n<!--\n  DERIVED, NOT MEASURED ANATOMY FOR FIVE OF THESE SIX LEGS.\n"
            "  Source: the 15 foreleg muscle-tendon units of Ozdi et al., 'Musculoskeletal\n"
            "  simulation of limb movement biomechanics in Drosophila melanogaster'\n"
            "  (arXiv 2509.06426, ICLR 2026), mirrored onto the other five legs.\n"
            "  Warrant: that paper's own statement that the muscle structure across legs is\n"
            "  nearly identical, with the middle-leg tergal depressor of the trochanter excepted.\n"
            "  Method: each attachment site was carried through a world transform ANCHORED AT THE\n"
            "  COXA (shared thorax sites do not move under a body-to-body map) and scaled by the\n"
            "  axial segment-length ratio.  Mid and hind legs have no Trochanter body, so\n"
            "  trochanter sites are mapped to the femur; 24 sites are affected.\n"
            "  Generated by tools_mirror_muscles.py; see outputs/muscles_six_legs/.\n-->\n")
    txt = txt.replace(">\n", ">" + prov, 1) if ">\n" in txt else txt
    dest.write_text(txt)
    (OUT_DIR / "mirror_decisions.json").write_text(json.dumps({
        "source_xml": str(SRC.relative_to(HERE)),
        "legs": LEG_ORDER,
        "added": {"tendons": added_tendons, "sites": added_sites, "actuators": added_acts},
        "non_one_to_one_mappings": decisions,
        "joints_added": added_joints,
        "locked_joints_in_source": [str(x) for x in locked],
        "unlocked": bool(unlock),
        "self_contact_disabled": bool(no_self_contact),
        "geoms_contact_filtered": n_filtered,
        "known_approximations": [
            "mid and hind legs have no Trochanter body, so trochanter sites are mapped to the femur",
            "site positions scaled isotropically by the segment length ratio (lengthwise fibres "
            "would want an axial-only scale)",
            "no muscles in the tibia and no tergal depressor of the trochanter, inherited from the "
            "source model",
        ],
    }, indent=2, sort_keys=True))
    print(f"wrote {dest}")
    print(f"wrote {OUT_DIR / 'mirror_decisions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
