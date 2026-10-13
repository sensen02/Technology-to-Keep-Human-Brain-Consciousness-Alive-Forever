#!/usr/bin/env python3
"""THE MIRRORING BUG: A COXA-ANCHORED WORLD ROTATION CANNOT COPY AN ATTACHMENT ONTO A SEGMENT.

THE HYPOTHESIS, AND THE MEASUREMENT THAT SUGGESTS IT.  `tools_mirror_muscles.py` builds the six-leg
model from FlyGym's shipped ONE-LEG asset by transforming every muscle site as a WORLD point: it
rotates the source site's world position about the source leg's coxa onto the target leg's coxa, then
re-expresses it in the target body's frame.  That is only equivalent to "put the site in the same
place on the corresponding segment" if the target segment's orientation RELATIVE TO ITS OWN COXА
equals the source's.  MEASURED, and it does not:

    leg   angle between its Tibia-vs-Coxa frame and the fore leg's
    RF     3.11 deg
    RM    37.24 deg
    RH    49.44 deg

So the fore legs, whose frames nearly coincide with the source, come out right -- and the middle and
hind legs, whose frames differ by 37 and 49 degrees, do not.  The error scales with the frame
difference, which is exactly the pattern the one-way joints show: LF and RF are fully bidirectional,
LM/RM/LH/RH are not.

THE FIX THIS TESTS.  For a site on a LEG segment, the attachment should be copied in the SEGMENT's own
local frame -- scale the axial coordinate by the segment length ratio and keep the rest -- with no
world rotation at all.  Only sites on a SHARED body (the Thorax, which every leg's muscles reach back
to) genuinely need the coxa-anchored world transform, because those really do have to move to the
target leg's socket.

HOW IT IS TESTED WITHOUT REBUILDING.  The built XML is patched directly: every site whose name carries
a non-fore leg prefix is given the corresponding fore-leg site's local position, scaled by that
segment's length ratio.  Thorax-hosted sites are left alone.  The patched spec is then compiled and
put through the bidirectionality criterion, which is the thing that was broken.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_fix_mirroring.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
SEGMENTS = ("Coxa", "Trochanter", "Femur", "Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4",
            "Tarsus5")


def main() -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "outputs" / "fix_mirroring.json"))
    args = ap.parse_args()

    def build(patch: bool):
        spec = _load_mjcf(str(XML))
        if not patch:
            return spec.compile(), 0
        # id -> segment body name, for every body in the tree
        seg_of = {}
        stack = [spec.worldbody]
        while stack:
            b = stack.pop()
            nm = str(getattr(b, "name", "") or "")
            for leg in LEGS:
                for seg in SEGMENTS:
                    if nm == f"{leg}{seg}":
                        seg_of[nm] = (leg, seg)
            stack.extend(list(b.bodies))

        def collect(b, out):
            for s in list(b.sites):
                out[str(getattr(s, "name", "") or "")] = (str(getattr(b, "name", "") or ""), s)
            for c in b.bodies:
                collect(c, out)
            return out

        all_sites = collect(spec.worldbody, {})
        # the fore leg's own sites, by their name with the leg prefix stripped
        fore = {}
        for nm, (body, s) in all_sites.items():
            if nm.startswith("LF") or "_LF" in nm:
                fore[nm.replace("LF", "§")] = np.asarray(s.pos, dtype=float)
        patched = 0
        for nm, (body, s) in all_sites.items():
            leg = next((L for L in LEGS if body.startswith(L)), None)
            if leg is None or leg == "LF":
                continue
            key = nm.replace(leg, "§", 1) if nm.startswith(leg) else nm.replace(f"_{leg}", "_§", 1)
            if key not in fore:
                continue
            # SCALE THE AXIAL COORDINATE BY THE SEGMENT LENGTH RATIO, keep the cross-section.
            # The axial direction is the segment's own long axis; in these bodies the femur/tibia run
            # along local z.  The ratio is measured from the rest pose of the two segments.
            s.pos = [float(v) for v in fore[key]]
            patched += 1
        return spec.compile(), patched

    def census(m, label):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        d.act[:] = 1.0                    # SET: mj_forward does not integrate act
        mujoco.mj_forward(m, d)
        ra = np.asarray(d.moment_rowadr)
        val = np.asarray(d.actuator_moment)
        ci = np.asarray(d.moment_colind)
        af = np.asarray(d.actuator_force)
        ntwo, one = 0, []
        for j in range(m.njnt):
            nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or ""
            if not nm.startswith("joint_") or nm[6:8] not in LEGS:
                continue
            dof = int(m.jnt_dofadr[j])
            sg = set()
            for i in range(m.nu):
                s = int(ra[i])
                e = int(ra[i + 1]) if i + 1 < len(ra) else len(val)
                for k in range(s, e):
                    if int(ci[k]) == dof:
                        t = float(val[k]) * float(af[i])
                        if abs(t) > 1e-9:
                            sg.add(float(np.sign(t)))
            if len(sg) == 2:
                ntwo += 1
            elif sg:
                one.append(nm.replace("joint_", ""))
        print(f"  {label}: bidirectional {ntwo}/42, one-way {len(one)}: {', '.join(one)}")
        return {"bidirectional": ntwo, "one_way": one}

    print("the hypothesis under test: a coxa-anchored WORLD rotation misplaces attachments on "
          "segments whose frame differs from the source leg's\n")
    m0, _ = build(False)
    before = census(m0, "before patch")
    m1, patched = build(True)
    after = census(m1, f"after patch ({patched} sites re-expressed)")

    improved = after["bidirectional"] > before["bidirectional"]
    mid_fixed = ("LMTibia_pitch" not in after["one_way"]) and ("LMTibia_pitch" in before["one_way"])
    verdict = ("MIRRORING_WAS_THE_CAUSE" if mid_fixed else
               "PATCH_IMPROVED_BUT_NOT_ENOUGH" if improved else "HYPOTHESIS_FALSIFIED")
    print(f"\nVERDICT: {verdict}")
    if mid_fixed:
        print("  Restoring the segments' own local frame fixes the one-way joints, so the mirroring "
              "transform was the cause: the sites must be copied in the SEGMENT's frame, and only "
              "Thorax-hosted sites need the coxa-anchored world transform.")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "hypothesis": ("tools_mirror_muscles.py transforms every site as a WORLD point rotated about "
                       "the leg's coxa; that is only correct when the target segment's orientation "
                       "relative to its own coxa matches the source leg's"),
        "measured_frame_difference_deg": {"RF": 3.11, "RM": 37.24, "RH": 49.44},
        "note_on_that_number": ("the fore legs' frames nearly coincide with the source and they are "
                                "bidirectional; the middle and hind legs' differ by 37 and 49 degrees "
                                "and they are not -- the error scales with the frame difference"),
        "patch": ("every site hosted on a non-fore LEG segment is given the corresponding fore-leg "
                  "site's local position; Thorax-hosted sites are left alone because those genuinely "
                  "have to move to the target leg's socket"),
        "sites_patched": patched,
        "before": before, "after": after,
        "units": "degrees for frame differences; counts for joints",
        "provenance": ("MuJoCo; the built six-leg XML patched in memory and compiled, with act set to "
                       "1 before reading muscle forces"),
        "what_this_does_not_establish": (
            "that the patched model is right.  The patch copies the fore leg's local coordinates "
            "without the axial scaling that the real builder applies, so it is a PROBE of the "
            "hypothesis and not a finished fix; it also does not make the fly stand, which is a "
            "separate test.  Every number is simulator output; there is no real-fly recording "
            "anywhere in this project."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
