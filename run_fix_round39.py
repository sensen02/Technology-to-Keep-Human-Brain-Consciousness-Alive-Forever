"""ROUND 39 (FIX): the corrections, re-measured under a gated protocol.

WHAT THIS REPLACES
------------------
Rounds 36-38 produced three numbers that are quoted everywhere:
    "escape in 25-34 ms", "margin ceiling 0.370 mm", "diagnostic reflex 0/738".
Those runs were made on a body whose contact parameters were never named, and whose
altitude field was labelled `_mm` while holding centimetres (see engine/canonical_physics.py
for the two defects D1/D2 and the exact measurements).

THE PROTOCOL, FIXED BEFORE ANY NUMBER IS LOOKED AT
--------------------------------------------------
  0. Load the canonical asset.  Assert its SHA256 is unchanged.
  1. Seat the root by bisection so the LOWEST TARSUS point sits 0.05 mm into the floor.
     The 0.05 mm is converted from mm in exactly one place.
  2. GATE the seated pose (engine.canonical_physics.stance_facts):
        - only tarsus geoms may touch the floor
        - no non-foot geom may be inside the floor
     A pose that fails is a hard error; no number is produced from it.
  3. Forward-integrate under the CONSTANT activation of the round-37 zero-adhesion
     exact balance, adhesion actuators pinned to 0, for STABILITY_SECONDS.
  4. Report: escape time (first drop below 90% of the seated thorax height), final and
     minimum thorax height in MM, contact count and tarsus-only at the end.

The escape threshold is deliberately expressed as a FRACTION of the gated seated
height.  The old absolute "escape below 1.8 mm" is exactly the kind of number that made
a 10x unit slip invisible.

This is a DIAGNOSTIC measurement of a body, not an autonomous result, and not a claim
about fly behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402

STABILITY_SECONDS = 1.0
ESCAPE_FRACTION = 0.90          # thorax below 90% of the gated seated height == escape
ROUND37_STANCE = HERE / "outputs/stance_margin_round37/stance_margin.json"
ROUND37_POSE = HERE / "outputs/stance_margin_round37/stance_for_reflex.json"
OUT = HERE / "outputs/fix_round39"


def set_constant_activation(model, data, act_map: dict) -> int:
    """Round-37 LP activation as a constant command; adhesion pinned to zero."""
    import mujoco

    n_set = 0
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
        v = float(act_map.get(nm, 0.0))
        data.ctrl[i] = v
        if v:
            n_set += 1
    # the last two actuators are the adhesion pads; round 36/37/38 all forced them to 0
    data.ctrl[model.nu - 2:] = 0.0
    return n_set


def apply_round37_pose(model, data, pose: dict) -> None:
    """Write the round-37 joint offsets onto the XML keyframe.  NOTE: the JSON's
    `root_z_mm` field holds the KEYFRAME ROOT HEIGHT IN MODEL UNITS (cm) - this is
    defect D1.  It is intentionally NOT used to set the height here; seat_root()
    solves the height instead, which is what makes this protocol unit-safe."""
    import mujoco

    key = model.key_qpos[0].copy()
    q = key.copy()
    for nm, off in pose["joint_offsets"].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if jid < 0:
            raise SystemExit(f"joint {nm!r} from the round-37 pose does not exist")
        adr = int(model.jnt_qposadr[jid])
        q[adr] = key[adr] + float(off)
    data.qpos[:] = q
    data.qvel[:] = 0


def _contact_geom(model) -> int:
    """A geom that actually touches the floor (a tarsus), so the recorded contact
    parameters are the ones in force at the foot-ground contact."""
    return int(cp.tarsus_geom_ids(model)[0])


def run_case(label: str, cs: cp.ContactSet, *, timestep=None, integrator=None,
             iterations=None, tolerance=None, seconds=STABILITY_SECONDS) -> dict:
    import mujoco

    src = cp.CANONICAL_XML.read_text()
    variant = cp.build_variant_xml(src, cs, timestep=timestep, integrator=integrator,
                                   iterations=iterations, tolerance=tolerance)
    model = cp.load_model(variant)
    data = mujoco.MjData(model)

    pose = json.loads(ROUND37_POSE.read_text())
    apply_round37_pose(model, data, pose)
    seated = data.qpos.copy()
    z = cp.seat_root(model, data, seated)            # declared penetration, model units
    data.qpos[:] = seated
    data.qpos[2] = z
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    gate = cp.stance_facts(model, data)               # raises if the pose is not a stance
    h0 = gate.thorax_centre_model

    act = json.loads(ROUND37_STANCE.read_text())["best"]["activation"]
    n_active = set_constant_activation(model, data, act)

    n = int(round(seconds / model.opt.timestep))
    hs = np.empty(n)
    escape = None
    t0 = time.time()
    for k in range(n):
        mujoco.mj_step(model, data)
        h = float(data.xpos[model.body("Thorax").id][2])
        hs[k] = h
        if escape is None and h < ESCAPE_FRACTION * h0:
            escape = (k + 1) * model.opt.timestep
    end_gate = cp.stance_facts(model, data, raise_on_fail=False)

    return {
        "label": label,
        "contact_set": cs.name,
        "contact_provenance": cs.provenance,
        "solver_overrides": {"timestep": timestep, "integrator": integrator,
                             "iterations": iterations, "tolerance": tolerance},
        "seated": gate.as_dict(),
        "seated_thorax_mm": float(cp.model_to_mm(h0)),
        "escape_fraction_of_seated": ESCAPE_FRACTION,
        "escape_time_s": escape,
        "final_thorax_mm": float(cp.model_to_mm(hs[-1])),
        "min_thorax_mm": float(cp.model_to_mm(hs.min())),
        "final_sample_gate": end_gate.as_dict(),
        "n_actuators_commanded_nonzero": n_active,
        "wall_s": time.time() - t0,
        "trace_t_s": (np.arange(n) * model.opt.timestep).round(6).tolist()[::max(1, n // 200)],
        "trace_thorax_mm": cp.model_to_mm(hs).round(4).tolist()[::max(1, n // 200)],
        "physics": {
            "timestep_s": float(model.opt.timestep),
            "integrator": int(model.opt.integrator),
            "cone": int(model.opt.cone),
            "noslip_iterations": int(model.opt.noslip_iterations),
            # measured on a geom that ACTUALLY touches the floor
            "contact_params_geom": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, _contact_geom(model)),
            "solref": [float(v) for v in model.geom_solref[_contact_geom(model)]],
            "solimp": [float(v) for v in model.geom_solimp[_contact_geom(model)]],
            "margin_model": float(model.geom_margin[_contact_geom(model)]),
            "friction": [float(v) for v in model.geom_friction[_contact_geom(model)]],
            "total_mass_kg": float(model.body_mass.sum() * cp.MODEL_MASS_UNIT_KG),
            "body_weight_model": float(model.body_mass.sum() * cp.GRAVITY_MODEL),
        },
    }


def one_at_a_time_cases() -> list:
    """Change ONE numeric field at a time from the MuJoCo-default baseline, so the
    effect of each field is attributable.  Round 38 changed solref+solimp+margin
    together and could not say which one did what."""
    out = []
    base = cp.CONTACT_SETS["mujoco_default"]
    for sr, tag in [((2e-2, 1.0), "solref_0.02_default"),
                    ((2e-3, 1.0), "solref_0.002"),
                    ((2e-4, 1.0), "solref_0.0002_upstream")]:
        import dataclasses
        out.append((f"only_{tag}",
                    dataclasses.replace(base, name=f"only_{tag}",
                                        provenance=f"MuJoCo defaults with solref forced to {sr}",
                                        solref=sr)))
    # friction alone
    for mu in (0.5, 1.0, 2.0):
        import dataclasses
        out.append((f"only_mu_{mu}",
                    dataclasses.replace(base, name=f"only_mu_{mu}",
                                        provenance=f"MuJoCo defaults with sliding friction {mu}",
                                        friction=(mu, 0.005, 0.0001))))
    # geometric margin alone
    import dataclasses
    out.append(("only_margin_1e-3",
                dataclasses.replace(base, name="only_margin_1e-3",
                                    provenance="MuJoCo defaults plus a 1e-3 (0.01 mm) "
                                               "geometric contact margin, as the upstream asset",
                                    margin=1e-3)))
    # cone + noslip alone (the upstream asset's solver-side settings)
    out.append(("only_cone_elliptic_noslip3",
                dataclasses.replace(base, name="only_cone_elliptic_noslip3",
                                    provenance="MuJoCo defaults plus the upstream asset's "
                                               "cone=elliptic and noslip_iterations=3",
                                    cone="elliptic", noslip_iterations=3)))
    # solver effort alone
    out.append(("only_solver_effort",
                dataclasses.replace(base, name="only_solver_effort",
                                    provenance="MuJoCo defaults plus iterations=200, "
                                               "tolerance=1e-12, integrator=implicitfast",
                                    iterations=200, tolerance=1e-12,
                                    integrator="implicitfast")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=STABILITY_SECONDS)
    ap.add_argument("--quick", action="store_true",
                    help="only the two named bundles, skip the one-at-a-time sweep")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    if digest != cp.CANONICAL_XML_SHA256:
        raise SystemExit(f"canonical asset changed: {digest}")

    cases = [(f"bundle_{k}", v) for k, v in cp.CONTACT_SETS.items()]
    if not args.quick:
        cases += one_at_a_time_cases()

    results = []
    for label, cs in cases:
        try:
            r = run_case(label, cs, seconds=args.seconds)
            print(f"[OK]   {label:34s} escape={r['escape_time_s']}  "
                  f"final={r['final_thorax_mm']:.3f} mm  seated={r['seated_thorax_mm']:.3f} mm "
                  f"({r['wall_s']:.1f}s)", flush=True)
        except cp.StanceGateError as e:
            r = {"label": label, "contact_set": cs.name, "gate_error": str(e)}
            print(f"[GATE] {label:34s} REJECTED: {e}", flush=True)
        except Exception as e:  # keep the sweep going, but record it loudly
            r = {"label": label, "contact_set": cs.name,
                 "error": f"{type(e).__name__}: {e}"}
            print(f"[ERR]  {label:34s} {type(e).__name__}: {e}", flush=True)
        results.append(r)

    cp.write_result(
        OUT / "fix_round39_contact_one_at_a_time.json",
        question=("With the round-37 zero-adhesion exact-balance pose seated by the NEW "
                  "gated protocol (only tarsus contact, no body through the floor), how "
                  "long does the body hold, and which single contact/solver field is "
                  "responsible?"),
        status="PASS" if results else "NOT_RUN",
        measured={
            "protocol": {
                "seating": ("bisection on root z until the lowest tarsus point sits at "
                            f"{cp.SEATING_PENETRATION_MM} mm from the floor plane "
                            "(the project's historical convention, declared once in "
                            "engine/canonical_physics.py; the old constant was LABELLED "
                            f"{cp.SEATING_PENETRATION_LABEL_MM} mm while being "
                            "applied as model units, i.e. 0.5 mm)"),
                "gate": "only tarsus geoms may touch the floor; no non-foot geom inside it",
                "command": "round-37 LP zero-adhesion balance activation held constant",
                "adhesion": "pinned to 0 (last two actuators)",
                "escape": f"thorax centre below {ESCAPE_FRACTION:.2f} x gated seated height",
                "seconds": args.seconds,
            },
            "cases": results,
            "table": [
                {"case": r["label"],
                 "contact_set": r.get("contact_set"),
                 "escape_time_s": r.get("escape_time_s"),
                 "final_thorax_mm": r.get("final_thorax_mm"),
                 "seated_thorax_mm": r.get("seated_thorax_mm"),
                 "n_contact_seated": (r.get("seated") or {}).get("n_contact"),
                 "tarsus_only_seated": (r.get("seated") or {}).get("tarsus_only"),
                 "n_contact_final": (r.get("final_sample_gate") or {}).get("n_contact"),
                 "error": r.get("gate_error") or r.get("error")}
                for r in results
            ],
        },
        provenance={
            "canonical_xml": str(cp.CANONICAL_XML),
            "canonical_xml_sha256": digest,
            "canonical_xml_unchanged": True,
            "round37_stance": str(ROUND37_STANCE),
            "round37_pose": str(ROUND37_POSE),
            "upstream_reference": str(cp.UPSTREAM_FLYBODY_XML),
            "code": [str(HERE / "engine/canonical_physics.py"),
                     str(HERE / "run_fix_round39.py")],
            "interpreter": sys.executable,
            "mujoco": __import__("mujoco").__version__,
            "numpy": np.__version__,
        },
        what_this_does_not_support=[
            "A longer escape time is NOT standing: it only says when the pose left the "
            "neighbourhood of ITS OWN seated height.",
            "No result here is autonomous behaviour; the command is a constant LP solution "
            "obtained with adhesion forced to zero.",
            "Contact parameters are model settings, not measurements of a fly's pulvilli.",
            "The command is constant, so this is a measurement of the BODY's open-loop "
            "stability, not of any controller.",
        ],
        defect_fixes={
            "D1_unit_label": ("all heights reported here come from StanceFacts, which "
                              "converts model units (cm) to mm in exactly one place; the "
                              "round-37 `root_z_mm` field is NOT consumed"),
            "D2_buried_initial_condition": ("the gate refuses any pose with a non-foot "
                                            "geom inside the floor; seat_root solves the "
                                            "height instead of accepting a stored one"),
        },
    )
    print("wrote", OUT / "fix_round39_contact_one_at_a_time.json")


if __name__ == "__main__":
    main()
