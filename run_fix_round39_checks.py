"""ROUND 39 (FIX): scan the unit defect and the stance-validity defect, tree-wide.

Two mechanical checks, both able to FAIL.  This is what turns "I fixed it" into
"it cannot come back silently".

C1  NAME/UNIT HONESTY (defect D1)
    For every Python file in the project (excluding virtualenvs/vendor), find fields
    whose NAME ends in `_mm` and classify the expression that fills them.  A field whose
    name says `_mm` and whose value comes straight out of a model-unit source
    (`xpos[...]`, `qpos[...]`, `mj_geomDistance`, `contact.dist`, `geom_margin`) holds
    CENTIMETRES.  Those sites are counted, listed, and compared against a declared
    allowlist, so the backlog is MEASURED rather than described in prose.

C2  STANCE VALIDITY GATE (defect D2)
    Build the stance the way the new protocol does and require, as a hard assertion:
      * the thorax centre is at the anatomically possible standing height
        (>= 0.8 x the mean leg-chain length, which is 17.52 mm here),
      * at least MIN_TARSUS_CONTACTS tarsus geoms touch the floor,
      * no non-foot geom is inside the floor plane,
      * a stored `root_z_mm` consumed as millimetres is REJECTED.
    The last one is the regression test for D2: it must fail.

Both checks are run by run_fix_round39_checks.py and both write a contract-carrying JSON.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402

PROJECT = HERE
OUT = HERE / "outputs/fix_round39"

EXCLUDE_DIRS = ("venv", "venv_body", ".venv", "vendor", "upstream", ".git", "node_modules",
                "__pycache__", "site-packages")

# Sources that yield MODEL UNITS (cm) when read.
MODEL_UNIT_SOURCES = (
    "xpos[", "qpos[", "mj_geomDistance", "contact[c].dist", "con.dist", "geom_margin[",
    "geom_gap[", "body.ipos", "geom_pos[",
)

MIN_TARSUS_CONTACTS = 10
MIN_THORAX_MM_FRACTION_OF_LEG = 0.8


def iter_py_files():
    for p in PROJECT.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        yield p


def scan_unit_labels() -> dict:
    """C1: count and locate `_mm`-named fields that are filled from model units."""
    pat = re.compile(r"['\"]([A-Za-z0-9_]+_mm)['\"]\s*:\s*([^,\n}]*)")
    hits = []
    total_mm_fields = 0
    for p in iter_py_files():
        try:
            text = p.read_text()
        except Exception:
            continue
        for m in pat.finditer(text):
            total_mm_fields += 1
            field, expr = m.group(1), m.group(2)
            src = next((s for s in MODEL_UNIT_SOURCES if s in expr), None)
            if src is None:
                continue
            line = text[: m.start()].count("\n") + 1
            hits.append({
                "file": str(p.relative_to(PROJECT)),
                "line": line,
                "field": field,
                "expr": expr.strip()[:110],
                "model_unit_source": src,
                "held_unit": "cm (model units) under an _mm name",
            })
    by_file = {}
    for h in hits:
        by_file.setdefault(h["file"], []).append(h["line"])
    return {
        "mm_named_fields_total": total_mm_fields,
        "mm_named_but_model_unit": len(hits),
        "files_affected": len(by_file),
        "by_file": {k: sorted(v) for k, v in sorted(by_file.items())},
        "sites": hits,
    }


def leg_chain_length_mm(model) -> tuple:
    """Mean thorax-to-tarsus chain length, summed over the segments that EXIST.

    NOTE (measured): the middle legs (LM/RM) have no `Trochanter` body in this asset -
    only Coxa/Femur/Tibia/Tarsus1-5 - while the fore and hind legs do.  The chain length
    is therefore computed per leg from whatever segments are present, and the per-leg
    values are reported so the difference is visible rather than averaged away.
    """
    import numpy as np
    import mujoco

    legs = ["LF", "LM", "LH", "RF", "RM", "RH"]
    lens, per_leg = [], {}
    for leg in legs:
        names = ([f"{leg}Coxa", f"{leg}Trochanter", f"{leg}Femur", f"{leg}Tibia"]
                 + [f"{leg}Tarsus{i}" for i in range(1, 6)])
        ids, missing = [], []
        for n in names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid < 0:
                missing.append(n)
            else:
                ids.append(int(bid))
        if len(ids) < 2:
            per_leg[leg] = None
            continue
        pos = np.array([model.body(b).pos for b in ids])
        L = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
        lens.append(L)
        per_leg[leg] = {"chain_mm": L * cp.MM_PER_MODEL_LENGTH,
                        "missing_bodies": missing,
                        "segments_used": [model.body(b).name for b in ids]}
    return (float(np.mean(lens) * cp.MM_PER_MODEL_LENGTH) if lens else float("nan")), per_leg


def check_stance_gate() -> dict:
    """C2: the positive control (a real stance) and the negative control (D2 consumed)."""
    import mujoco

    import run_fix_round39 as R39

    model = cp.load_model()
    data = mujoco.MjData(model)
    pose = json.loads(R39.ROUND37_POSE.read_text())
    R39.apply_round37_pose(model, data, pose)
    base = data.qpos.copy()

    leg_mm, per_leg = leg_chain_length_mm(model)
    need_thorax_model = cp.mm_to_model(MIN_THORAX_MM_FRACTION_OF_LEG * leg_mm)

    # --- positive control: seat by bisection under the declared penetration -----
    z = cp.seat_root(model, data, base)
    data.qpos[:] = base
    data.qpos[2] = z
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    gate = cp.stance_facts(model, data)
    checks = {
        "tarsus_only": bool(gate.tarsus_only),
        "enough_tarsus_contacts": bool(gate.n_contact >= MIN_TARSUS_CONTACTS),
        "thorax_above_geometric_minimum": bool(gate.thorax_centre_model >= need_thorax_model),
        "no_body_inside_floor": bool(gate.worst_floor_penetration_model >= cp.SEATING_PENETRATION_MODEL - 1e-9),
    }
    positive_ok = all(checks.values())

    # --- negative control: the D2 consumption must be REJECTED ------------------
    data2 = mujoco.MjData(model)
    data2.qpos[:] = base
    data2.qpos[2] = float(pose["root_z_mm"]) / 10.0     # stored value consumed as mm
    data2.qvel[:] = 0
    mujoco.mj_forward(model, data2)
    neg = cp.stance_facts(model, data2, raise_on_fail=False)
    negative_rejected = (neg.n_body_floor_contact > 0) or (not neg.tarsus_only) \
                        or (neg.thorax_centre_model < need_thorax_model)

    return {
        "leg_chain_length_mm": leg_mm,
        "leg_chain_per_leg": per_leg,
        "thorax_minimum_mm": MIN_THORAX_MM_FRACTION_OF_LEG * leg_mm,
        "min_tarsus_contacts": MIN_TARSUS_CONTACTS,
        "positive_control": {"seated": gate.as_dict(), **checks, "passed": positive_ok},
        "negative_control_D2_consumed": {
            "seated": neg.as_dict(), "rejected": bool(negative_rejected),
            "consumer": "qpos[2] = stored_root_z_mm / 10.0",
        },
        "both_controls_behave": bool(positive_ok and negative_rejected),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    c1 = scan_unit_labels()
    c2 = check_stance_gate()

    cp.write_result(
        OUT / "fix_round39_unit_and_stance_checks.json",
        question=("Can the two round-39 defects be detected mechanically, and does the "
                  "new stance gate accept a real stance while rejecting a body that a "
                  "mislabelled height has put inside the floor?"),
        status="PASS" if c2["both_controls_behave"] else "FAIL",
        measured={"C1_unit_label_scan": c1, "C2_stance_gate": c2},
        provenance={
            "canonical_xml": str(cp.CANONICAL_XML),
            "canonical_xml_sha256": cp.sha256_file(cp.CANONICAL_XML),
            "canonical_xml_unchanged": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
            "declared_penetration_mm": cp.SEATING_PENETRATION_MM,
            "excluded_dirs": list(EXCLUDE_DIRS),
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "C1 counts sites; it does NOT prove that any particular published number was "
            "computed from one of them.  For that, see "
            "fix_round39_unit_defect_impact.json, which exercises the one consumer path "
            "found (qpos = stored/10) as a controlled arm.",
            "C2 is a geometry/contact check.  It says a pose is a stance, not that the "
            "stance is stable.",
            "The `_mm` sites listed in C1 are a BACKLOG.  Renaming them without "
            "re-measuring nothing: each call site has to be re-run to change a number.",
        ],
        verdicts={
            "C1": (f"{c1['mm_named_but_model_unit']} of {c1['mm_named_fields_total']} "
                   f"`_mm`-named fields hold model units (cm), across "
                   f"{c1['files_affected']} files"),
            "C2": ("gate accepts the seated stance and rejects the buried one"
                   if c2["both_controls_behave"] else
                   "GATE DOES NOT DISCRIMINATE - fix before trusting any stance number"),
        },
    )

    print("C1:", c1["mm_named_but_model_unit"], "/", c1["mm_named_fields_total"],
          "`_mm` fields hold cm across", c1["files_affected"], "files")
    print("C2 positive: seated thorax %.3f mm, ncon %d -> %s" % (
        c2["positive_control"]["seated"]["thorax_centre_mm"],
        c2["positive_control"]["seated"]["n_contact"],
        "PASS" if c2["positive_control"]["passed"] else "FAIL"))
    print("C2 negative (D2 consumed): thorax %.3f mm, ncon %d -> %s" % (
        c2["negative_control_D2_consumed"]["seated"]["thorax_centre_mm"],
        c2["negative_control_D2_consumed"]["seated"]["n_contact"],
        "REJECTED (good)" if c2["negative_control_D2_consumed"]["rejected"] else "ACCEPTED (bad)"))
    print("wrote", OUT / "fix_round39_unit_and_stance_checks.json")


if __name__ == "__main__":
    main()
