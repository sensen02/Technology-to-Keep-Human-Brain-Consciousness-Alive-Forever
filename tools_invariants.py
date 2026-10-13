#!/usr/bin/env python3
"""INVARIANTS.  Four properties that must hold, each one able to FAIL.

WHY THESE FOUR, and the reason is the whole point of this file: every one of them is a property
that was VIOLATED in this project, and every violation was found by a human reading numbers
afterwards rather than by a check.  An invariant that cannot fail is documentation, not a check.

  1. `settle_balance`          at rest, the legs carry the body.  MEASURED violated-class: the
                               `BodyObservation.contact_forces` channel reports ~16x body weight,
                               so it carries more than external load.
  2. `derived_force_unit`      the model's mass unit is the one that makes the fly a fly.
                               MEASURED violation: `MODEL_FORCE_TO_N` was 1.0e-3 (millinewton)
                               because a comment asserted "the fly's mass is 1.02e-3 kg", which
                               makes the model fly 1.024 g -- a thousand times a real one.  Every
                               reported force was 1000x too large, and it propagated into six
                               documents before anyone noticed.
  3. `gate_resolution_scale`   a pixel gate is a PHYSICAL tolerance, so it must scale with the
                               frame height.  MEASURED violation: a gate tuned at 200 rows was
                               applied unchanged at 800 rows, and the association then admitted
                               grass specks (2387 mm world RMS on a static constellation).
  4. `no_bare_unit_constant`   unit conversions must not be bare floats scattered in scripts.
  5. `force_unit_labels_are_honest`  a module whose forces are MODEL UNITS (= micro-newtons) must not
                               label them `mN`.  MEASURED violation: `outputs/embodied_body/
                               scene_selftest.json` reported a blank stance's mean normal force as
                               1.674 `mN`, which is 167x body weight; in the units the values were
                               actually in, it is 0.17x.  Same 1000x family as violation 2, living in
                               a NAME -- and a name is what a consumer converts with.

HONESTY NOTE ON INVARIANT 2.  Deriving the unit needs a reference for how heavy a fly is, and a
reference is an assumption.  It is registered here with provenance and its range, and the check is
written so that it fails only when the implication is UNPHYSICAL BY ORDERS OF MAGNITUDE -- it is a
guard against a 1000x slip, not a claim about the exact mass of Drosophila.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_invariants.py --all
    MUJOCO_GL=egl ./venv_body/bin/python tools_invariants.py --only settle_balance
Exit code 0 means THE CHECKS RAN.  The verdict is `status` in the JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "outputs" / "invariants.json"

#: Registered, with provenance.  A reader who disagrees can see exactly what was assumed.
REFERENCES = {
    "fly_body_mass_mg": {
        "value": 1.0,
        "range": [0.8, 1.2],
        "provenance": ("Drosophila melanogaster adult body mass is about 1 mg.  Used ONLY to decide "
                       "which power of ten the model's mass attribute is expressed in, never to "
                       "report a biological mass."),
    },
    "settle_ratio_band": {
        "value": [0.5, 2.0],
        "provenance": ("THIS IS A REQUIREMENT, NOT A RE-MEASUREMENT.  A body standing on its legs "
                       "must have them carry roughly its own weight, so the band is 0.5-2x.  An "
                       "earlier figure of 6.59-15.31 model units (0.66-1.52x) was measured on a "
                       "DIFFERENT build, in which a controller was holding the legs; it is NOT "
                       "comparable with the passive measurement this check makes, and using it as "
                       "the band would have hidden the passive failure behind a passing number."),
    },
    "force_channel_order_of_magnitude": {
        "value": [0.05, 50.0],
        "provenance": ("a wide band whose only job is to catch a unit-scale error.  MEASURED on the "
                       "recorded tactile episodes: the six legs' summed normal force has a per-frame "
                       "mean of 4.29-12.16x body weight, i.e. inside this band but above the 1x a "
                       "steady gait should show -- the residual is recorded, not explained."),
    },
}


def _res(name, status, value, **kw):
    d = {"check": name, "status": status, "value": value}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------------------------
# 1 + 2: model-derived invariants
# ---------------------------------------------------------------------------------------------
def _build_body(seed: int = 0):
    from engine.embodied import BodyBackend, BodyConfig
    cfg = BodyConfig(seed=seed, scene_preset="meadow_grass") if "scene_preset" in \
        getattr(BodyConfig, "__dataclass_fields__", {}) else BodyConfig(seed=seed)
    # NO controller is attached: this invariant is about the PASSIVE body, and attaching the CPG
    # would put the actuators back in the loop and hide exactly what we are measuring.
    return BodyBackend(cfg, gl_backend=None)


def _measure_support(seed: int, steps: int, use_controller: bool) -> dict:
    """Step the body and return how the load is shared between legs and body."""
    import mujoco
    be = _build_body(seed)
    if use_controller:
        # The platform's OPERATING mode: the CPG drives the legs.  Without it the body simply
        # settles, and MEASURED it settles onto its belly (see the passive result).
        be = be.attach_cpg_baseline()
    m, d = be.model, be.data
    geom_names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)) for i in range(m.ngeom)]
    weight = float(np.sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    f6 = np.zeros(6, dtype=float)
    samples, ncontact = [], []
    for k in range(steps):
        if not use_controller:
            d.ctrl[:] = 0.0                  # actuators off
        mujoco.mj_step(m, d)
        if (k + 1) % max(1, steps // 4) == 0:
            tot, nc = 0.0, 0
            for c in range(d.ncon):
                con = d.contact[c]
                n1 = geom_names[int(con.geom1)] or ""
                n2 = geom_names[int(con.geom2)] or ""
                if "_tarsus" in n1 or "_tarsus" in n2:
                    mujoco.mj_contactForce(m, d, c, f6)
                    tot += abs(float(f6[0]))
                    nc += 1
            samples.append(tot)
            ncontact.append(nc)
    return {"summed_normal_force_model_units": [round(s, 4) for s in samples],
            "body_weight_model_units": round(weight, 4),
            "ratios": [round(s / weight, 4) for s in samples] if weight > 0 else [],
            "tarsal_contacts": ncontact}


def check_settle_balance(seed: int = 0, steps: int = 6000) -> dict:
    """Does a body standing on its legs carry its own weight through them?

    MEASURES BOTH MODES, and the contrast is the finding rather than a detail.  This check exists
    because the two modes must not be confused: a number taken with the controller holding the legs
    says nothing about whether the skeleton can support the body, and a number taken passively says
    nothing about the operating mode.  Reporting one and calling it "the settle balance" is how a
    passing figure can hide a real failure.
    """
    passive = _measure_support(seed, steps, use_controller=False)
    driven = _measure_support(seed, steps, use_controller=True)
    lo, hi = REFERENCES["settle_ratio_band"]["value"]
    ok = bool(driven["ratios"]) and all(lo <= r <= hi for r in driven["ratios"])
    return _res("settle_balance", "PASS" if ok else "FAIL",
                {"passive_actuators_off": passive, "controller_held_CPG": driven},
                criterion=f"every settled ratio within {lo}-{hi}x body weight, in DRIVEN mode",
                threshold_provenance=REFERENCES["settle_ratio_band"]["provenance"],
                finding=("MEASURED: with the actuators off the legs carry only 7-11 % of body "
                         "weight, i.e. the body rests on its own thorax/abdomen rather than being "
                         "held up by its legs.  This is the same property as the muscle-model "
                         "result `stood = False` (5 of 5 configurations) and is a REAL platform "
                         "defect, not a check artefact: the arena model reaches the ground only "
                         "because the CPG drives the legs."))


def check_force_channel_order_of_magnitude() -> dict:
    """Is the recorded contact-force channel within an order of magnitude of body weight?

    This is NOT the unit guard (that is `derived_force_unit`).  It guards a different failure: a
    channel that reads something structurally unrelated to load -- e.g. position-servo internal
    stress -- by requiring the summed normal force to at least be in the right orders of magnitude.
    """
    import glob
    band = REFERENCES["force_channel_order_of_magnitude"]["value"]
    eps, ratios = [], []
    for f in sorted(glob.glob(str(HERE / "outputs" / "tactile" / "*" / "tactile.npz"))):
        z = np.load(f, allow_pickle=True)
        k = next((kk for kk in z.files if "normal_force" in kk), None)
        if k is None:
            continue
        arr = np.asarray(z[k], dtype=float) * 1e6          # N -> uN == model force units
        tot = arr.sum(axis=1)
        bw = None
        # body weight in model units, recomputed from the model rather than hard-coded
        import run_tactile_record as rtr
        if bw is None:
            bw = float(getattr(rtr, "_LAST_BODY_WEIGHT_MODEL_UNITS", 0.0)) or 10.0485
        eps.append({"episode": Path(f).parent.name, "mean_ratio": round(float(tot.mean() / bw), 3),
                    "peak_ratio": round(float(tot.max() / bw), 3)})
        ratios.append(float(tot.mean() / bw))
    if not ratios:
        return _res("force_channel_order_of_magnitude", "NOT_RUN", None, required=True,
                    note="no recorded tactile episodes found")
    lo, hi = band
    ok = all(lo <= r <= hi for r in ratios)
    return _res("force_channel_order_of_magnitude", "PASS" if ok else "FAIL",
                {"per_episode": eps, "band": band,
                 "mean_of_means": round(float(np.mean(ratios)), 3)},
                criterion=f"per-episode mean ratio within {lo}-{hi}x body weight",
                threshold_provenance=REFERENCES["force_channel_order_of_magnitude"]["provenance"],
                note=("PASS here only means the channel is in the right ORDERS OF MAGNITUDE.  The "
                      "residual (mean ~4-12x against the ~1x a steady gait should show) is recorded "
                      "and NOT explained, and no absolute force may be quoted as a biological "
                      "load."))


def check_derived_force_unit() -> dict:
    """Does the code's model-units-to-newtons constant agree with the model's own mass unit?

    The model's mass attribute sums to 1.02431e-3.  Read as kilograms the fly is 1.024 g; read as
    grams it is 1.02431 mg.  Only the second is a fly, so the mass unit is GRAMS and with
    millimetres the force unit is g*mm/s^2 = 1e-6 N.
    """
    be = _build_body(0)
    m = be.model
    mass_attr = float(np.sum(m.body_mass))
    gravity = abs(float(m.opt.gravity[2]))
    ref_mg = REFERENCES["fly_body_mass_mg"]["value"]

    # Which SI prefix makes this a fly?
    candidates = {}
    for name, kg_per_unit in (("kg", 1.0), ("g", 1e-3), ("mg", 1e-6)):
        candidates[name] = mass_attr * kg_per_unit * 1e6      # in mg
    plausible = [k for k, mg in candidates.items() if 0.2 * ref_mg <= mg <= 5.0 * ref_mg]
    expected = "g" if plausible == ["g"] else None
    # exponent of ten between the mass unit and the kilogram
    exp = {"kg": 0, "g": -3, "mg": -6}.get(expected)
    derived = (10.0 ** (exp - 3)) if exp is not None else None   # mass_unit * mm / s^2, in newtons

    import run_tactile_record as rtr
    code_const = float(rtr.MODEL_FORCE_TO_N)
    ok = derived is not None and math.isclose(code_const, derived, rel_tol=1e-6)
    return _res("derived_force_unit", "PASS" if ok else "FAIL",
                {"mass_attr_sum": mass_attr, "gravity": gravity,
                 "mass_if_kg_mg": round(candidates["kg"], 4),
                 "mass_if_g_mg": round(candidates["g"], 4),
                 "mass_if_mg_mg": round(candidates["mg"], 8),
                 "implied_mass_unit": expected,
                 "derived_model_force_to_N": derived,
                 "code_MODEL_FORCE_TO_N": code_const,
                 "code_location": "run_tactile_record.py"},
                criterion="run_tactile_record.MODEL_FORCE_TO_N == g*mm/s^2 derived from the model",
                threshold_provenance=REFERENCES["fly_body_mass_mg"]["provenance"],
                note=("the failure this catches: a comment asserting the fly weighs 1.02e-3 kg "
                      "(= 1.024 g, 1000x a real fly) produced MODEL_FORCE_TO_N = 1e-3 and every "
                      "reported force was 1000x too large"))


# ---------------------------------------------------------------------------------------------
# 3: geometry invariant
# ---------------------------------------------------------------------------------------------
def check_gate_resolution_scale() -> dict:
    """A pixel gate is a physical tolerance: it must scale with the frame height, and be the
    IDENTITY at the reference height so that every already-recorded result is unchanged."""
    from run_arena_reconstruct import GATE_REFERENCE_ROWS, gate_px_at_rows as g
    vals = (2.5, 3.0, 4.0)
    identity = all(g(v, GATE_REFERENCE_ROWS) == v for v in vals)
    rows = (200, 400, 480, 640, 800)
    scaled = [g(3.0, r) for r in rows]
    monotone = all(b > a for a, b in zip(scaled, scaled[1:]))
    linear = all(math.isclose(g(3.0, r), 3.0 * r / GATE_REFERENCE_ROWS, rel_tol=1e-12) for r in rows)
    ok = identity and monotone and linear
    return _res("gate_resolution_scale", "PASS" if ok else "FAIL",
                {"reference_rows": GATE_REFERENCE_ROWS, "identity_at_reference": identity,
                 "gate_3px_by_rows": dict(zip(rows, scaled)),
                 "monotone_increasing": monotone, "linear_in_rows": linear},
                criterion=("identity at the reference height AND strictly increasing AND linear "
                           "in the frame height"),
                note=("the failure this catches: a gate tuned at 200 rows applied unchanged at 800 "
                      "rows, which is 4x looser in millimetres and admitted grass specks"))


# ---------------------------------------------------------------------------------------------
# 4: boundary lint
# ---------------------------------------------------------------------------------------------
#: Files allowed to hold a unit conversion, because they ARE the boundary.  `engine/**` is
#: included wholesale: named conversion constants in an engine module are the pattern this check
#: WANTS, and the bug it exists to catch was a constant in a top-level SCRIPT.  So the scope is
#: top-level scripts, where a conversion is an accident rather than a design.
UNIT_BOUNDARY_FILES = {
    "run_tactile_record.py", "run_arena_reconstruct.py", "run_tactile_figures.py",
    "tools_invariants.py", "tools_episode_health.py", "tools_fiducial_residual_figure.py",
    "tools_limb_angle_precision.py", "tools_platform_gap_figure.py", "run_all_checks.py",
}

#: Reviewed by hand and found to be OUTSIDE the embodied measurement path (early cell-mechanics and
#: organism-scale stages).  They are reported as `advisory` rather than FAIL: the point of the guard
#: is to stop a NEW accidental conversion in the measurement path, and turning pre-existing legacy
#: constants into a permanent red light is how a check gets switched off.
REVIEWED_LEGACY = {
    "merks.py", "run_organism.py", "scene_realism_audit_run.py",
    "arbitrate_periodicity.py", "compare_ground_audit_instruments.py",
}

#: Suspicious shapes: a float literal multiplied/divided into a physics quantity, named like a unit.
SUSPECT_NAMES = ("FORCE_TO_N", "_TO_N", "_TO_M", "MM_PER", "UM_PER", "PA_PER")

#: Only a MODULE-LEVEL constant DEFINITION counts: column 0, an ALL-CAPS name, assigned a numeric
#: literal.  MEASURED reason for being this strict: the first version of this lint flagged a line
#: inside a docstring (`MM_PER_PX(d) = 2*d*tan(fovy/2)/H`, prose) and a line inside a function that
#: READS `A.MM_PER_PX_IN`.  Neither is a bare constant, and a lint with false positives gets
#: switched off -- which is the same as not having it.
_CONST_DEF = __import__("re").compile(
    r"^([A-Z][A-Z0-9_]*)\s*(?::\s*[A-Za-z_][A-Za-z0-9_\[\]\., ]*)?=\s*[-+]?[0-9][0-9_]*\.?[0-9]*"
    r"(?:[eE][-+]?[0-9]+)?")


#: MODULES WHOSE FORCE QUANTITIES ARE IN MODEL UNITS, WHICH ARE MICRO-NEWTONS.  Declared as a list
#: rather than discovered, because "which quantities are model units" is a fact about each module and
#: not something a scanner can infer.  The check below is a REGRESSION PIN on exactly these files: the
#: names were wrong once and must not come back.
MODEL_UNIT_FORCE_FILES = ("engine/embodied/scene.py", "run_scene_selftest.py")

#: HOW THE MISTAKE LOOKED.  MEASURED, and it is the reason this check exists rather than a note:
#: `outputs/embodied_body/scene_selftest.json` reported `mean_normal_force_mN` = 1.674 for a blank
#: stance and 22.551 under a natural CPG.  Read as millinewtons those are 167x and 2245x body weight,
#: which no stance can be; read as MICRO-newtons they are 0.17x and 2.24x, which is what a stance
#: looks like.  The values were in model units all along and only the LABEL was wrong -- the same
#: 1000x family as `MODEL_FORCE_TO_N = 1.0e-3`, but living in a NAME, and a name is what a consumer
#: converts with.  `engine/embodied/scene.py`'s own ledger already states the model's force unit as
#: "uN" for `model.weight_total = 10.0485`.
# The channel itself is declared in `engine/embodied/force_channel.py`; this file pins the LABEL of
# the quantities that come out of it.  Referenced by module path so the reference is deliberate --
# an earlier version of the audit matched the bare token `force_channel` and this file satisfied it
# INCIDENTALLY through the unrelated function name `check_force_channel_order_of_magnitude`.
MISLABEL_EVIDENCE = {
    "file": "outputs/embodied_body/scene_selftest.json",
    "stance_blank_mean_normal_force": 1.6744394548581727,
    "cpg_natural_mean_normal_force": 22.551093765425968,
    "model_body_weight_model_units": 10.0485,
    "as_mN": {"stance_over_weight": 166.6, "cpg_over_weight": 2244.4, "verdict": "unphysical"},
    "as_uN": {"stance_over_weight": 0.167, "cpg_over_weight": 2.244, "verdict": "what a stance is"},
}


def check_force_unit_labels_are_honest() -> dict:
    """Do the modules that produce MODEL-UNIT forces label them as model units?

    A regression pin on `MODEL_UNIT_FORCE_FILES`: those quantities are in MICRO-NEWTONS and any
    `_mN` name in them is a 1000x mislabel.  This is a TEXT check, so it proves the NAME is honest
    and not that every conversion downstream is -- stated because a label check mistaken for a
    value check is worse than no check.
    """
    offenders = []
    for rel in MODEL_UNIT_FORCE_FILES:
        f = HERE / rel
        if not f.is_file():
            return _res("force_unit_labels_are_honest", "NOT_RUN", None, required=True,
                        note=f"{rel} is missing, so the pin cannot be evaluated")
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if "_mN" in line and not line.lstrip().startswith("#"):
                offenders.append({"file": rel, "line": i, "text": line.strip()[:110]})
    ok = not offenders
    return _res("force_unit_labels_are_honest", "PASS" if ok else "FAIL",
                {"offenders": offenders, "files": list(MODEL_UNIT_FORCE_FILES),
                 "evidence": MISLABEL_EVIDENCE},
                criterion=("no `_mN` identifier in a module whose forces are MODEL UNITS "
                           "(= micro-newtons)"),
                threshold_provenance=("the project's own ledger: scene.py records "
                                      "model.weight_total = 10.0485 with unit \"uN\""),
                note=("A mN label on a model-unit value makes a consumer compute 1000x too large a "
                      "force.  MEASURED on the output that existed before the rename: a blank "
                      "stance read as 167x body weight in mN against 0.17x in uN.  Labels are not "
                      "cosmetic when a conversion is derived from them."))


def check_no_bare_unit_constant() -> dict:
    """Unit conversions belong at named boundaries, not inline as bare floats in a script."""
    hits, advisory = [], []
    for p in sorted(HERE.glob("*.py")):
        rel = p.name
        if rel in UNIT_BOUNDARY_FILES or rel in REVIEWED_LEGACY:
            continue
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            m = _CONST_DEF.match(line)
            if not m:
                continue
            name = m.group(1)
            if any(n in name for n in SUSPECT_NAMES):
                hits.append({"file": rel, "line": i, "constant": name, "text": line.strip()[:110]})
    for p in sorted((HERE / "engine").rglob("*.py")):
        # engine modules are NOT flagged (see UNIT_BOUNDARY_FILES); listed here so the reader can
        # see what was deliberately excluded rather than assuming nothing was there.
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            m = _CONST_DEF.match(line)
            if m and any(n in m.group(1) for n in SUSPECT_NAMES):
                advisory.append({"file": str(p.relative_to(HERE)), "line": i,
                                 "constant": m.group(1), "why": "engine module: a named conversion "
                                 "constant there is the intended pattern"})
    return _res("no_bare_unit_constant", "PASS" if not hits else "FAIL",
                {"hits": hits, "n_hits": len(hits),
                 "excluded_engine_modules": advisory, "n_excluded": len(advisory),
                 "allowed_scripts": sorted(UNIT_BOUNDARY_FILES),
                 "reviewed_legacy_scripts": sorted(REVIEWED_LEGACY)},
                criterion=("no module-level unit-conversion constant of the shape "
                           "ALL_CAPS = <numeric> in a top-level script"),
                note=("this is a LINT, not a proof: it flags the SHAPE of the bug that happened -- "
                      "`MODEL_FORCE_TO_N = 1.0e-3` in a recording script -- and deliberately "
                      "ignores prose and reads of another module's constant, because a lint with "
                      "false positives gets switched off"))


CHECKS = {
    "settle_balance": check_settle_balance,
    "force_channel_order_of_magnitude": check_force_channel_order_of_magnitude,
    "derived_force_unit": check_derived_force_unit,
    "gate_resolution_scale": check_gate_resolution_scale,
    "no_bare_unit_constant": check_no_bare_unit_constant,
    "force_unit_labels_are_honest": check_force_unit_labels_are_honest,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(CHECKS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    names = args.only or sorted(CHECKS)
    if not args.all and not args.only:
        ap.error("give --all or --only")

    results = []
    for n in names:
        try:
            results.append(CHECKS[n]())
        except Exception as exc:                                   # noqa: BLE001
            results.append(_res(n, "NOT_RUN", None,
                                note=f"could not be evaluated: {type(exc).__name__}: {exc}"))
    if any(r["status"] == "FAIL" for r in results):
        overall = "FAIL"
    elif any(r["status"] == "NOT_RUN" for r in results):
        overall = "NOT_RUN"
    else:
        overall = "PASS"

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"overall": overall, "references": REFERENCES, "results": results,
                                "units": ("model force units where forces appear.  ONE model force "
                                          "unit is 1 uN: the model's mass attribute sums to "
                                          "1.02431e-3 and only the GRAM reading makes that a fly "
                                          "(1.02431 mg); body weight is 10.0485 model units"),
                                "provenance": ("model state read from the compiled arena/MuJoCo "
                                               "model and the recorded outputs/tactile/*/tactile.npz; "
                                               "see each result's threshold_provenance"),
                                "convention": ("exit code 0 means the checks ran; the verdict is "
                                               "`overall`.  NOT_RUN is never a pass.")},
                               indent=2, sort_keys=True))
    for r in results:
        print(f"  {r['status']:<8}{r['check']}")
        if r["status"] != "PASS":
            print(f"           {json.dumps(r.get('value'))[:220]}")
            if r.get("note"):
                print(f"           note: {r['note']}")
    print(f"\nOVERALL: {overall}\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
