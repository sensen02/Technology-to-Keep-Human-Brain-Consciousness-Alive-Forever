"""ROUND 39 (FIX), part B: how much did the unit-label defect (D1) actually change?

WHY
---
D1 is established: 73 fields are named `_mm` while holding model units (cm), 39 of them
read straight out of `xpos`/`qpos`.  D2 (a consumer that turns the mislabelled value back
into the model with `/10`) exists.  What is NOT yet established is whether D2 ever
reached a published number, because the round 36-38 scripts may all have seated the root
by bisection and therefore never consumed the mislabelled height.

This script answers that with a one-factor experiment instead of an argument:

    factor: the initial root height used for the forward integration
      arm "gated"   : root solved so the lowest tarsus point is 0.05 mm into the floor
                      -> the pose is a stance: tarsus-only contact, body above the floor
      arm "reported": the same pose with the stored, MISLABELLED height consumed as if it
                      were millimetres (`qpos[2] = stored / 10`)
                      -> the body sits inside the floor plane

    everything else identical: same canonical asset, same constant round-37 activation,
    same contact parameters, same 0.30 s horizon, same escape rule (fraction of the
    gated seated height).

Output: the escape time difference attributable to D1 alone.  If the two arms agree, then
rounds 36-38's published collapse times were NOT contaminated and only their UNITS are
wrong.  If they disagree, the published times are void and must be re-measured.

Diagnostic measurement of a body.  Not an autonomous result.  Not a fly measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
import run_fix_round39 as R39        # noqa: E402

OUT = HERE / "outputs/fix_round39"
SECONDS = 0.30
ESCAPE_FRACTION = 0.90


def trace(model, data, *, seconds, h_ref) -> dict:
    import mujoco

    n = int(round(seconds / model.opt.timestep))
    hs = np.empty(n)
    escape = None
    for k in range(n):
        mujoco.mj_step(model, data)
        h = float(data.xpos[model.body("Thorax").id][2])
        hs[k] = h
        if escape is None and h < ESCAPE_FRACTION * h_ref:
            escape = (k + 1) * model.opt.timestep
    return {
        "escape_time_s": escape,
        "final_thorax_mm": float(cp.model_to_mm(hs[-1])),
        "min_thorax_mm": float(cp.model_to_mm(hs.min())),
        "_hs_model": hs,
    }


def main() -> None:
    import mujoco

    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    pose = json.loads(R39.ROUND37_POSE.read_text())
    stored = pose["root_z_mm"]                      # mislabelled: model units under a _mm name
    act = json.loads(R39.ROUND37_STANCE.read_text())["best"]["activation"]
    cs = cp.CONTACT_SETS["mujoco_default"]

    model = cp.load_model()
    data = mujoco.MjData(model)
    R39.apply_round37_pose(model, data, pose)
    base = data.qpos.copy()

    arms = {}

    # ---- arm A: gated, root solved by bisection -------------------------------
    # DEFAULT_PENETRATION_MM, converted to model units in exactly one place.
    dA = mujoco.MjData(model)
    dA.qpos[:] = base
    z_solved = cp.seat_root(model, dA, base, target_penetration_model=cp.SEATING_PENETRATION_MODEL)
    dA.qpos[:] = base
    dA.qpos[2] = z_solved
    dA.qvel[:] = 0
    mujoco.mj_forward(model, dA)
    gate = cp.stance_facts(model, dA)                # raises if not a stance
    h_ref = gate.thorax_centre_model
    n_act = R39.set_constant_activation(model, dA, act)
    a = trace(model, dA, seconds=SECONDS, h_ref=h_ref)
    arms["gated_solved_root"] = {
        "consumes_stored_height": False,
        "root_z_model": float(z_solved),
        "root_z_mm": float(cp.model_to_mm(z_solved)),
        "seated_thorax_mm": float(cp.model_to_mm(h_ref)),
        "seated_gate": gate.as_dict(),
        "n_actuators_commanded_nonzero": n_act,
        **{k: v for k, v in a.items() if not k.startswith("_")},
        "final_gate": cp.stance_facts(model, dA, raise_on_fail=False).as_dict(),
    }

    # ---- arm B: stored mislabelled height consumed as millimetres -------------
    dB = mujoco.MjData(model)
    dB.qpos[:] = base
    dB.qpos[2] = float(stored) / 10.0                # the D2 consumption: cm value / 10
    dB.qvel[:] = 0
    mujoco.mj_forward(model, dB)
    gateB = cp.stance_facts(model, dB, raise_on_fail=False)
    n_actB = R39.set_constant_activation(model, dB, act)
    b = trace(model, dB, seconds=SECONDS, h_ref=h_ref)
    arms["reported_mislabelled_root"] = {
        "consumes_stored_height": True,
        "root_z_model": float(stored) / 10.0,
        "root_z_mm": float(cp.model_to_mm(float(stored) / 10.0)),
        "seated_thorax_mm": float(cp.model_to_mm(gateB.thorax_centre_model)),
        "seated_gate": gateB.as_dict(),
        "n_actuators_commanded_nonzero": n_actB,
        **{k: v for k, v in b.items() if not k.startswith("_")},
        "final_gate": cp.stance_facts(model, dB, raise_on_fail=False).as_dict(),
    }

    # ---- arm C: the SAME bisection with the old constant's LITERAL value applied
    # verbatim (unconverted).  This is reproducibility, not a defect: the historical
    # convention IS 0.5 mm of penetration, so arm C should match arm A and to that
    # extent confirm that rounds 14-38 ran on a physically reproducible stance.  The
    # defect was that the constant was NAMED `PENETRATION_MM = -0.05`.
    dC = mujoco.MjData(model)
    dC.qpos[:] = base
    z_old_convention = cp.seat_root(model, dC, base, target_penetration_model=-0.05)
    dC.qpos[:] = base
    dC.qpos[2] = z_old_convention
    dC.qvel[:] = 0
    mujoco.mj_forward(model, dC)
    gateC = cp.stance_facts(model, dC, raise_on_fail=False)
    n_actC = R39.set_constant_activation(model, dC, act)
    c = trace(model, dC, seconds=SECONDS, h_ref=h_ref)
    arms["gated_old_penetration_convention"] = {
        "consumes_stored_height": False,
        "penetration_target_as_used_model_units": -0.05,
        "penetration_target_actually_mm": float(cp.model_to_mm(-0.05)),
        "root_z_model": float(z_old_convention),
        "root_z_mm": float(cp.model_to_mm(z_old_convention)),
        "seated_thorax_mm": float(cp.model_to_mm(gateC.thorax_centre_model)),
        "seated_gate": gateC.as_dict(),
        "n_actuators_commanded_nonzero": n_actC,
        **{k: v for k, v in c.items() if not k.startswith("_")},
        "final_gate": cp.stance_facts(model, dC, raise_on_fail=False).as_dict(),
    }

    ea = arms["gated_solved_root"]["escape_time_s"]
    eb = arms["reported_mislabelled_root"]["escape_time_s"]
    verdict = ("D1 CHANGED THE RESULT: the two initial conditions do not agree"
               if (ea is None) != (eb is None) or (ea is not None and eb is not None
                                                   and abs(ea - eb) > 0.02)
               else "D1 DID NOT CHANGE THE RESULT: the published collapse times stand, "
                    "only their units are wrong")

    # height-vs-time table for the log, in mm, every 10 ms
    step = max(1, int(0.01 / model.opt.timestep))
    table = []
    hsA = cp.model_to_mm(a["_hs_model"])
    hsB = cp.model_to_mm(b["_hs_model"])
    for k in range(0, len(hsA), step):
        table.append({"t_s": round((k + 1) * model.opt.timestep, 4),
                      "gated_thorax_mm": round(float(hsA[k]), 4),
                      "mislabelled_thorax_mm": round(float(hsB[k]), 4)})

    cp.write_result(
        OUT / "fix_round39_unit_defect_impact.json",
        question=("Did the `_mm`-named-but-centimetre unit defect actually change the "
                  "collapse times published in rounds 36-38, or only their labels?"),
        status="PASS",
        measured={
            "arms": arms,
            "escape_time_difference_s": (None if (ea is None or eb is None)
                                         else float(eb - ea)),
            "verdict": verdict,
            "height_table_mm": table,
        },
        provenance={
            "canonical_xml": str(cp.CANONICAL_XML),
            "canonical_xml_sha256": digest,
            "canonical_xml_unchanged": True,
            "round37_pose": str(R39.ROUND37_POSE),
            "round37_stance": str(R39.ROUND37_STANCE),
            "stored_field_consumed": "outputs/stance_margin_round37/stance_for_reflex.json"
                                     " :: root_z_mm (holds model units, cm)",
            "contact_set": cs.name,
            "seconds": SECONDS,
            "escape_rule": f"thorax below {ESCAPE_FRACTION:.2f} x gated seated height",
            "interpreter": sys.executable,
            "mujoco": mujoco.__version__,
        },
        what_this_does_not_support=[
            "This does NOT validate the round 36-38 absolute numbers in millimetres - "
            "every one of those labels is 10x off regardless of which arm matches.",
            "A matching escape time does not make the old protocol gated: the old runs "
            "never checked for a body inside the floor.",
            "No controller is involved: the command is one constant activation vector.",
        ],
        defects_exercised={
            "D1": "73 `_mm` fields hold model units (cm); 39 read straight out of xpos/qpos",
            "D2": "the mislabelled value is consumed as mm by `qpos[2] = stored/10`",
        },
    )
    print(verdict)
    print("gated      : escape", arms["gated_solved_root"]["escape_time_s"],
          "s  seated", round(arms["gated_solved_root"]["seated_thorax_mm"], 3), "mm",
          " seated ncon", arms["gated_solved_root"]["seated_gate"]["n_contact"],
          " final", arms["gated_solved_root"]["final_thorax_mm"])
    print("mislabelled: escape", arms["reported_mislabelled_root"]["escape_time_s"],
          "s  seated", round(arms["reported_mislabelled_root"]["seated_thorax_mm"], 3), "mm",
          " seated ncon", arms["reported_mislabelled_root"]["seated_gate"]["n_contact"],
          " final", arms["reported_mislabelled_root"]["final_thorax_mm"])
    oc = arms["gated_old_penetration_convention"]
    print("old-penetration-convention: escape", oc["escape_time_s"],
          "s  seated", round(oc["seated_thorax_mm"], 3), "mm"
          " (target written as 0.05 mm, applied as",
          round(oc["penetration_target_actually_mm"], 3), "mm)")
    print("wrote", OUT / "fix_round39_unit_defect_impact.json")


if __name__ == "__main__":
    main()
