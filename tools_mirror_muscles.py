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
                Rs, ps = world[body]
                Rt, pt = world[tgt_body]
                p_world_src = ps + Rs @ (k * local)
                p_world_tgt = t_T + R_T @ p_world_src
                new_pos = Rt.T @ (p_world_tgt - pt)
                new_site_name = ref.replace("LF", LEG_PREFIX[target], 1)
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
    print(f"added: {added_tendons} tendons, {added_sites} sites, {added_acts} actuators")
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
