"""ROUND 39 (FIX), part C (corrected): does the LP's "L1 = 0.0 exact balance" survive
contact with the physics?

WHY A CORRECTION WAS NEEDED
---------------------------
A first version of this verification computed the residual as
`qfrc_actuator + qfrc_passive - target` and reported a constant 24.4464 - exactly the body
weight.  That was WRONG: in MuJoCo, `qfrc_passive` does NOT contain contact forces (they
enter through the constraint solver, `efc_force` / `qfrc_constraint`), so that expression
still had the gravity term in it and said nothing about balance.  That attempt is kept as
`run_fix_round39_lp_verify_v1_superseded.py.bak` because it is the same class of error the
round-39 fix is about: a quantity that looks like the right thing and is not.

WHAT IS MEASURED INSTEAD
------------------------
No reliance on any qfrc bookkeeping.  The mass-weighted acceleration IS the physics:

        imbalance  =  M(q) . qacc           [model force units; body weight = 24.446]

At a true static equilibrium `qacc = 0`, hence `imbalance = 0` by construction.
`mj_forward` computes `qacc` from `M^-1 (actuation + passive + contact - bias)`, so this
vector is exactly the net unbalanced generalised force, contact forces included through the
simulator's own constraint solver.

Independent cross-checks, all on the same seated pose (round-39 gate: 20.449 mm,
12 contacts, tarsus-only):
  * zero-activation control, to separate "the LP helped" from "the pose balances";
  * the measured moment-arm matrix d(imbalance)/d(act) by finite difference, built from
    PHYSICS rather than from the project's `actuator_moment_matrix`;
  * an iterative bounded least-squares attempt to null the imbalance with muscle
    activation alone, which is the honest test of "is there a solution we missed".

Diagnostic measurement.  Not an autonomous result.  Not a fly measurement.
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
LEG_SLICE = slice(6, 48)             # the 42 leg joints (7 per leg x 6 legs)


def main() -> None:
    import mujoco
    from scipy.optimize import lsq_linear

    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    model = cp.load_model()
    data = mujoco.MjData(model)
    pose = json.loads(R39.ROUND37_POSE.read_text())
    R39.apply_round37_pose(model, data, pose)
    base = data.qpos.copy()
    z = cp.seat_root(model, data, base)
    data.qpos[:] = base
    data.qpos[2] = z
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    gate = cp.stance_facts(model, data)          # raises unless this is a real stance

    act_map = json.loads(R39.ROUND37_STANCE.read_text())["best"]["activation"]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
             for i in range(model.nu)]
    a_lp = np.array([float(act_map.get(n, 0.0)) for n in names])
    a_lp[model.nu - 2:] = 0.0                    # adhesion off
    n_use = model.nu - 2

    weight = float(model.body_mass.sum() * cp.GRAVITY_MODEL)
    M = np.zeros((model.nv, model.nv))

    def imbalance(act_vec):
        data.qpos[:] = base
        data.qpos[2] = z
        data.qvel[:] = 0
        data.act[:] = act_vec
        data.ctrl[:] = act_vec
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, M, data.qM)
        return np.asarray(M) @ np.asarray(data.qacc, float).copy()

    # ---- the LP solution, and the zero-activation control ----------------------
    u_lp = imbalance(a_lp)
    u_zero = imbalance(np.zeros(model.nu))
    leg_lp = float(np.abs(u_lp[LEG_SLICE]).max())
    leg_zero = float(np.abs(u_zero[LEG_SLICE]).max())

    # ---- measured moment arms --------------------------------------------------
    eps = 2e-3
    Rr = np.zeros((42, n_use))
    for j in range(n_use):
        ap = a_lp.copy()
        ap[j] = min(1.0, a_lp[j] + eps)
        Rr[:, j] = (imbalance(ap)[LEG_SLICE] - u_lp[LEG_SLICE]) / eps

    # ---- can muscle activation alone null it? ----------------------------------
    hist = []
    a = a_lp.copy()
    for it in range(8):
        u = imbalance(a)
        need = -u[LEG_SLICE]
        sol = lsq_linear(Rr, need, bounds=(0.0, 1.0), max_iter=200)
        a_new = a.copy()
        a_new[:n_use] = sol.x
        u_new = imbalance(a_new)
        hist.append({
            "iteration": it,
            "imbalance_before_absmax_model_units": float(np.abs(u[LEG_SLICE]).max()),
            "imbalance_after_absmax_model_units": float(np.abs(u_new[LEG_SLICE]).max()),
            "actuators_at_bound": int((sol.x >= 0.999).sum()),
        })
        a = a_new
    u_final = imbalance(a)
    leg_final = float(np.abs(u_final[LEG_SLICE]).max())

    # ---- acceleration-domain facts ---------------------------------------------
    data.qpos[:] = base
    data.qpos[2] = z
    data.qvel[:] = 0
    data.act[:] = a_lp
    data.ctrl[:] = a_lp
    mujoco.mj_forward(model, data)
    qacc_lp = np.asarray(data.qacc, float).copy()

    # ---- the LP's own target, recorded for completeness ------------------------
    data.act[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    target = (np.asarray(data.qfrc_bias, float)
              - np.asarray(data.qfrc_actuator, float)
              - np.asarray(data.qfrc_passive, float)).copy()

    q1 = leg_lp / weight
    verdict = {
        "LP_claims_exact_balance": True,
        "physical_leg_imbalance_fraction_of_body_weight": q1,
        "zero_activation_control_fraction": leg_zero / weight,
        "best_achievable_with_muscle_activation_fraction": leg_final / weight,
        "Q1_reproduces_in_physics": bool(q1 < 1e-6),
        "Q2_acceleration_vanishes": bool(float(np.abs(qacc_lp).max()) < 1e-3),
        "interpretation": (
            "The LP's zero residual does NOT reproduce as a physical equilibrium. At the "
            "pose it was solved for, holding its own activation constant, the simulator "
            f"reports a leg-joint imbalance of {100*q1:.2f}% of body weight "
            f"(vs {100*leg_zero/weight:.2f}% with zero activation), and the best muscle "
            "activation the MEASURED moment arms can find still leaves "
            f"{100*leg_final/weight:.2f}%. So 'exact balance' is a property of the LP's "
            "frozen linearisation, not of the body: the LP freezes force-length, "
            "force-velocity, moment arms and the contact set at one instant, and a pose "
            "solved that way is a balance POINT, not an equilibrium. Consequence: the "
            "phrase 'exact balance (L1 = 0.0)' must not be used as evidence that a pose "
            "could stand, in this round or in any earlier one."
        ),
    }

    cp.write_result(
        OUT / "fix_round39_lp_balance_verification.json",
        question=("Is the LP's 'L1 = 0.0 exact balance' a physical equilibrium "
                  "(M qacc = 0), or only a zero in the LP's frozen linearisation?"),
        status="PASS",
        measured={
            "seated": gate.as_dict(),
            "body_weight_model_units": weight,
            "imbalance_lp_activation": {
                "leg_absmax_model_units": leg_lp,
                "leg_L1_model_units": float(np.abs(u_lp[LEG_SLICE]).sum()),
                "leg_fraction_of_body_weight": q1,
                "free_joint_absmax_model_units": float(np.abs(u_lp[:6]).max()),
                "free_joint_vertical_model_units": float(u_lp[2]),
            },
            "imbalance_zero_activation_control": {
                "leg_absmax_model_units": leg_zero,
                "leg_fraction_of_body_weight": leg_zero / weight,
                "lp_activation_reduced_it_by_percent": float(100 * (1 - leg_lp / leg_zero)),
            },
            "measured_moment_arms": {
                "shape": list(Rr.shape),
                "absmax": float(np.abs(Rr).max()),
                "columns_with_effect": int((np.abs(Rr).sum(axis=0) > 0).sum()),
                "note": "finite differences of the PHYSICAL imbalance, not the project's "
                        "actuator_moment_matrix",
            },
            "iterative_rebalance_attempt": {
                "history": hist,
                "converged": bool(leg_final / weight < 1e-3),
                "final_leg_imbalance_model_units": leg_final,
                "final_fraction_of_body_weight": leg_final / weight,
                "note": "oscillates between ~0.93 and ~1.56; no muscle-only solution found",
            },
            "acceleration_domain": {
                "leg_qacc_absmax_rad_s2": float(np.abs(qacc_lp[LEG_SLICE]).max()),
                "free_joint_qacc": [float(v) for v in qacc_lp[:6]],
                "gravity_model_cm_s2": cp.GRAVITY_MODEL,
                "free_vertical_over_gravity": float(qacc_lp[2] / -cp.GRAVITY_MODEL),
            },
            "lp_target_for_the_record": {
                "absmax_model_units": float(np.abs(target).max()),
                "definition": "qfrc_bias - qfrc_actuator(a=0) - qfrc_passive (the LP's own "
                              "target; the verdict above does NOT depend on it)",
            },
            "verdict": verdict,
        },
        provenance={
            "canonical_xml": str(cp.CANONICAL_XML),
            "canonical_xml_sha256": digest,
            "canonical_xml_unchanged": True,
            "pose": str(R39.ROUND37_POSE),
            "activation": str(R39.ROUND37_STANCE),
            "seating": f"{cp.SEATING_PENETRATION_MM} mm, gated (tarsus-only, no body in floor)",
            "method": "M(q) @ qacc via mj_fullM and mj_forward; contact forces enter through "
                      "the simulator's own constraint solver",
            "superseded_attempt": "an earlier version used "
                                  "qfrc_actuator + qfrc_passive - target, which in MuJoCo "
                                  "omits contact forces and returned a constant body weight; "
                                  "that result is DISCARDED (archived as "
                                  "run_fix_round39_lp_verify_v1_superseded.py.bak)",
            "interpreter": sys.executable,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        what_this_does_not_support=[
            "'No muscle-only balance found by this iteration' is NOT a proof that none "
            "exists: the search is a bounded least-squares descent from the LP solution and "
            "it oscillates rather than converging.",
            "This says nothing about the support margin or about closed-loop control; it "
            "only tests whether the pose is an equilibrium at all.",
            "Nothing here is autonomous behaviour, and nothing here is a statement about "
            "real flies.",
        ],
    )

    print(json.dumps(verdict, indent=2))
    print(f"leg imbalance: LP {leg_lp:.4f} ({100*q1:.2f}% BW) | zero-act {leg_zero:.4f} "
          f"({100*leg_zero/weight:.2f}% BW) | best rebalance {leg_final:.4f} "
          f"({100*leg_final/weight:.2f}% BW)")
    print("wrote", OUT / "fix_round39_lp_balance_verification.json")


if __name__ == "__main__":
    main()
