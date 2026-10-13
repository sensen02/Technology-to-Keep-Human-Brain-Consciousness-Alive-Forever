"""SUPERSEDED HISTORICAL EXPERIMENT: scientific verdicts below are WITHDRAWN.

See /run/media/sensen/Data2/cell_wound_prototype/outputs/workspace_recheck/VALIDITY_NOTICE.json.
qacc mixes angular/linear acceleration; Mqacc mixes torque/force. Neither full norm
can be called a fraction of gravity. Keyframe initialization and excluded steady
activation are also incorrect. Local failure is NOT an infeasibility proof.
Use /run/media/sensen/Data2/cell_wound_prototype/run_dimensionless_support_search.py instead.
Historical text and computation retained below only for reproducibility.

Can the 74 wired muscles statically support this pose at all?

WHY THIS IS THE RIGHT QUESTION NOW
----------------------------------
Three candidates for "the fly cannot stand" have now been measured and ELIMINATED:
  * the neural interface  - it emits commands and changes the body (measured), but the body
    falls at 24 ms with or without correct-sign feedback;
  * the XML control floor - its steady-state contribution is act=1e-4 (0.01% of range) and
    0.42% of the generalised actuator force; zeroing it changes the 0.05 s outcome by NOTHING
    (bit-identical forces, identical collapse time);
  * the unit labels       - corrected to millimetres, proven by a controlled A/B to move no
    physics at all.
And a new one has appeared: at this pose the collapse is essentially FREE FALL - the measured
0.0149 s sits within 1.9% of the 0.014628 s a body needs to fall the measured distance under
gravity alone.  That says the legs are doing almost nothing, which is a MECHANICAL statement.

THE EXPERIMENT
--------------
Static support is a well-posed property: a pose is supportable by a given actuator set if some
activation inside the allowed range drives the generalised acceleration to zero.  Minimising
``||qacc||`` over the actuator set is therefore a direct test, and this is where earlier audits
were weaker: they minimised the LEG-JOINT generalised torque only, which ignores the 6 root
DOFs and so cannot see the body falling or toppling.

Measured here, at the SAME gated seated pose and with steady-state activation (act = ctrl, the
fixed point of the verified muscle activation rule):
  * ||qacc|| over ALL 48 DOFs, decomposed into root translation (falling), root rotation
    (toppling) and the 42 leg joints;
  * for three actuator sets: all 90, the 74 with an own-leg pool, and the 16 unresolved;
  * the same quantity for the passive-only case (ctrl floor, no neural command) as the baseline.

PRE-REGISTERED
--------------
  H20  The 74-actuator set can drive the full-body acceleration residual below 1% of gravity
       (|qacc| < 98.01 mm/s^2, gravity being 9801 mm/s^2).
  FALSIFIER: the best reachable residual stays above 1% of gravity, in which case the pose is
       NOT statically supportable by the wired muscles and no amount of neural tuning fixes it.

Honest limits are declared in the artifact: this is a LOCAL optimum from finite-difference
descent at one pose, it is a static test (no gait), and passive force / joint stiffness are
part of the model rather than of the actuator command.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

import engine.canonical_physics as cp  # noqa: E402
import engine.memory_guard as mg  # noqa: E402

OUT = HERE / "outputs/workspace_recheck"
POOLS = HERE / "outputs/neural_rebuild/neural_rebuild_11_mn_pools.json"
GRAVITY_MM_S2 = 9801.0


def main(guard_report=None) -> None:
    import mujoco
    import run_fix_round39 as r39
    from scipy.optimize import least_squares

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-nfev", type=int, default=4000)
    parser.add_argument("--pose", choices=("round37", "keyframe"), default="round37")
    parser.add_argument("--objective", choices=("qacc", "mqacc"), default="qacc",
                        help="qacc = raw generalised acceleration (mixes mm and rad); "
                             "mqacc = M(q)*qacc, i.e. the generalised FORCE residual, which has "
                             "one unit and does not mix translation with rotation")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    model = cp.load_model()
    data = mujoco.MjData(model)
    if args.pose == "round37":
        r39.apply_round37_pose(model, data, json.loads(r39.ROUND37_POSE.read_text()))
    base = data.qpos.copy()
    base[2] = cp.seat_root(model, data, base)
    data.qpos[:] = base
    data.qvel[:] = 0
    data.act[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    gate = cp.stance_facts(model, data).as_dict()
    seated = data.qpos.copy()

    pools = json.loads(POOLS.read_text())["measured"]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    covered = np.array([names.index(r["actuator"]) for r in pools["resolved"]], dtype=int)
    uncovered = np.array([names.index(u["actuator"]) for u in pools["unresolved"]], dtype=int)
    allm = np.arange(model.nu)
    muscles = np.array([i for i in range(model.nu)
                        if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE], dtype=int)
    assert set(covered.tolist()) | set(uncovered.tolist()) == set(muscles.tolist())
    assert len(covered) == 74 and len(uncovered) == 16
    actadr = {int(i): int(model.actuator_actadr[i]) for i in muscles}
    lower = float(model.actuator_ctrlrange[0][0])
    root_slice = slice(0, 6)

    M = np.zeros((model.nv, model.nv))

    def acceleration(a):
        """Generalised acceleration with steady-state activation (act = ctrl) at the fixed pose."""
        data.qpos[:] = seated
        data.qvel[:] = 0
        data.ctrl[:] = 0.0
        for i in muscles:
            data.act[actadr[int(i)]] = 0.0
        data.ctrl[:] = a
        for i in muscles:
            data.act[actadr[int(i)]] = float(a[i])
        mujoco.mj_forward(model, data)
        return np.asarray(data.qacc, float).copy()

    def residual_vector(a):
        """The vector actually minimised, chosen so the verdict does not depend on unit mixing."""
        qacc = acceleration(a)
        if args.objective == "mqacc":
            mujoco.mj_fullM(model, M, data.qM)
            return np.asarray(M, float) @ qacc
        return qacc

    def decompose(qacc):
        return {"all_l2": float(np.linalg.norm(qacc)),
                "root_translation_l2": float(np.linalg.norm(qacc[root_slice][:3])),
                "root_rotation_l2": float(np.linalg.norm(qacc[root_slice][3:])),
                "leg_joints_l2": float(np.linalg.norm(qacc[6:])),
                "root_vertical": float(qacc[2]),
                "fraction_of_gravity_all": float(np.linalg.norm(qacc) / GRAVITY_MM_S2),
                "fraction_of_gravity_root_vertical": float(abs(qacc[2]) / GRAVITY_MM_S2)}

    def solve(label, free, seeds=(0.0,)):
        free = np.asarray(free, dtype=int)
        def resid(x):
            a = np.zeros(model.nu)
            a[free] = x
            return residual_vector(a)       # all 48 DOFs, in the declared objective
        best = None
        runs = []
        for s in seeds:
            x0 = np.full(free.size, max(s, lower))
            t0 = time.time()
            sol = least_squares(resid, x0, bounds=(lower, 1.0), xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=args.max_nfev)
            score = float(np.linalg.norm(sol.fun))
            runs.append({"seed": s, "final_residual_l2": score, "nfev": int(sol.nfev),
                         "status": int(sol.status), "cost": float(sol.cost),
                         "optimality": float(sol.optimality),
                         "actuators_at_upper_bound": int((sol.x >= 1 - 1e-9).sum()),
                         "actuators_at_lower_bound": int((sol.x <= lower + 1e-12).sum()),
                         "wall_s": time.time() - t0})
            if best is None or score < best[0]:
                best = (score, sol.x.copy())
        a = np.zeros(model.nu)
        a[free] = best[1]
        return {"label": label, "n_free": int(free.size),
                "free_actuator_names": [names[i] for i in free],
                "best_activation_by_name": {names[i]: float(a[i]) for i in free},
                "per_seed": runs,
                "residual": decompose(acceleration(a))}

    t0 = time.time()
    baseline = decompose(acceleration(np.zeros(model.nu)))
    print("passive baseline ||qacc|| = %.4f (%.4f%% of gravity)"
          % (baseline["all_l2"], 100*baseline["fraction_of_gravity_all"]), flush=True)
    conditions = [solve("ALL90", muscles, seeds=(0.0, 0.5, 1.0)),
                  solve("COVERED74", covered, seeds=(0.0, 0.5, 1.0)),
                  solve("UNCOVERED16", uncovered, seeds=(0.0, 0.5, 1.0))]
    for c in conditions:
        print("%-12s ||qacc||=%.4f  (%.4f%% of g)  root_t=%.4f root_r=%.4f legs=%.4f"
              % (c["label"], c["residual"]["all_l2"],
                 100*c["residual"]["fraction_of_gravity_all"],
                 c["residual"]["root_translation_l2"], c["residual"]["root_rotation_l2"],
                 c["residual"]["leg_joints_l2"]), flush=True)

    h20 = bool(conditions[1]["residual"]["fraction_of_gravity_all"] < 0.01)
    guard = guard_report("completed") if guard_report else None
    cp.write_result(
        OUT / "static_support_workspace.json",
        question=("At the gated seated pose, can the 74 actuators that have an own-leg "
                  "motor-neuron pool drive the FULL generalised acceleration to zero, i.e. is "
                  "the pose statically supportable by the wired muscles?"),
        status="PASS",
        measured={
            "status_meaning": "PASS = the optimisations ran and the artifact is complete; the "
                              "SCIENTIFIC verdict is in predictions.H20, and a negative there "
                              "is a result, not a failure of this script",
            "pose": args.pose, "gate": gate, "seated_root_z_model_units": float(seated[2]),
            "seated_root_z_mm": float(cp.model_to_mm(seated[2])),
            "gravity_mm_s2": GRAVITY_MM_S2,
            "objective": (f"minimise ||{args.objective}||_2 over all 48 DOFs with act = ctrl (the "
                         "verified fixed point of the muscle activation rule) and qvel = 0; "
                         "the verdict is always reported as the PHYSICAL qacc so that the two "
                         "objectives are comparable"),
            "objective_note": "qacc mixes mm (root translation) and rad (joints); mqacc is the "
                              "generalised force residual with one unit and is the more "
                              "principled objective. Both are run.",
            "why_full_qacc": "earlier audits minimised the 42 leg-joint generalised torques, "
                             "which cannot see the body falling or toppling; M is positive "
                             "definite so driving ||qacc|| to zero IS static equilibrium",
            "baseline_ctrl_at_floor_no_command": baseline,
            "conditions": conditions,
            "predictions": {"H20": {
                "statement": "the 74-actuator set reaches a full-body residual below 1% of gravity",
                "supported": h20,
                "falsifier": "best residual above 1% of gravity means the pose is NOT statically "
                             "supportable by the wired actuators",
                "covered74_fraction_of_gravity":
                    conditions[1]["residual"]["fraction_of_gravity_all"]}},
            "memory_guard": guard,
            "wall_s": time.time() - t0,
        },
        provenance={
            "code": str(Path(__file__).resolve()),
            "interpreter": sys.executable,
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "pool_result_sha256": cp.sha256_file(POOLS),
            "pose_source": ("round-37 joint offsets + cp.seat_root (gated)" if args.pose ==
                            "round37" else "the asset's own keyframe 0"),
            "initialisation": "qvel = 0; activation set EQUAL to the commanded value at every "
                              "evaluation, so the muscle steady state is assumed rather than "
                              "integrated; the transient is therefore not modelled here",
            "optimiser": "scipy.optimize.least_squares, bounds [ctrlrange_lower, 1], "
                         f"max_nfev={args.max_nfev}, three seeds",
        },
        units={"length": "mm", "acceleration": "mm/s^2", "gravity": GRAVITY_MM_S2,
               "force": "model force units (SI conversion INFERRED, not established)",
               "fraction_of_gravity": "dimensionless"},
        illustrative_vs_measured="qacc and the gate are simulator measurements; the objective, "
                                 "bounds and seeds are declared choices",
        what_this_does_not_support=[
            "A reachable residual below the threshold would show the pose is statically "
            "supportable by these actuators; it would NOT show a controller exists that finds "
            "or holds it, and it is not a walking result.",
            "A residual above the threshold is a statement about THIS pose, THIS actuator set "
            "and THIS static objective - it does not prove no other pose is supportable.",
            "The result is a LOCAL optimum from finite-difference descent; no global claim.",
            "Steady-state activation is assumed (act = ctrl). The 0.1-0.4 ms activation time "
            "constants make that reasonable, but the transient is not integrated here.",
            "Passive muscle force (~12% of active force in L2, and >= 50% of its own maximum "
            "for 13 of 90 muscles) and joint stiffness/damping are part of the model and are "
            "NOT part of the actuator command; no attempt is made to attribute the residual "
            "between them.",
            "No SI mass calibration, so forces are not comparable to a measured fly.",
        ],
    )
    print("wrote", OUT / "static_support_workspace.json")


if __name__ == "__main__":
    budget = mg.BUDGETS["neural_muscle_closure"]
    with mg.guarded_run("static_support_workspace", **budget) as _rep:
        main(guard_report=_rep)
