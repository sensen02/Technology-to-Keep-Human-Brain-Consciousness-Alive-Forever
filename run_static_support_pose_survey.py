"""SUPERSEDED HISTORICAL EXPERIMENT: H21 and systemic-failure inference WITHDRAWN.

See /run/media/sensen/Data2/cell_wound_prototype/outputs/workspace_recheck/VALIDITY_NOTICE.json.
Generalized acceleration/force norms mix dimensions; gravity normalization below
is invalid. Finite pose sampling and local failures cannot prove infeasibility.
Use /run/media/sensen/Data2/cell_wound_prototype/run_dimensionless_support_search.py instead.
Original computation retained below only for historical reproducibility.

Is the static-support failure specific to one pose, or systemic across the pose family?

WHY
---
A single pose failing to be statically supportable would be a search problem.  The finding that
matters is whether MANY poses in the same family also fail, because the historical stance search
optimised `LEG_SLICE = slice(6, 48)` - the 42 leg-joint generalised forces - and so could not
see the 6 root DOFs where falling and toppling appear.  If arbitrary poses in this family are
all far from equilibrium, the search objective itself was the structural defect.

METHOD
------
Sample poses (round-37 seat, the asset keyframe, and deterministic perturbations of the leg
joint offsets), and for each one run the SAME inner optimisation as
`run_static_support_workspace.py`: minimise the M-weighted generalised residual over the covered
74 actuators from several seeds.  Report the distribution of the best reachable residual,
expressed as a fraction of gravity through the physical ||qacc||.

A perturbation that makes a pose WORSE is not interesting; what matters is whether any sampled
pose gets materially closer to equilibrium.  The perturbations are deterministic (fixed seed),
so the run is reproducible.

Read-only on the canonical asset; guarded; one heavy process.
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
GRAVITY = 9801.0


def main(guard_report=None) -> None:
    import mujoco
    import run_fix_round39 as r39
    from scipy.optimize import least_squares

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perturbations", type=int, default=8)
    parser.add_argument("--sigma-rad", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--max-nfev", type=int, default=1500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert cp.sha256_file(cp.CANONICAL_XML) == cp.CANONICAL_XML_SHA256

    model = cp.load_model()
    data = mujoco.MjData(model)
    pose_doc = json.loads(r39.ROUND37_POSE.read_text())
    joint_names = list(pose_doc["joint_offsets"])
    jids = [model.joint(n).id for n in joint_names]
    qadr = np.array([int(model.jnt_qposadr[j]) for j in jids], dtype=int)

    pools = json.loads(POOLS.read_text())["measured"]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    covered = np.array([names.index(r["actuator"]) for r in pools["resolved"]], dtype=int)
    muscles = np.array([i for i in range(model.nu)
                        if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE], dtype=int)
    actadr = {int(i): int(model.actuator_actadr[i]) for i in muscles}
    lower = float(model.actuator_ctrlrange[0][0])
    M = np.zeros((model.nv, model.nv))

    def build(offsets):
        """Seat the pose with these joint offsets; return (qpos, gate) or None if the gate fails.

        MEASURED BUG (fixed): this first wrote ``data.qpos[qadr] = offsets``, i.e. treated the
        JSON values as ABSOLUTE joint angles.  ``apply_round37_pose`` applies them RELATIVE to the
        XML keyframe (``q[adr] = key[adr] + offset``), so the overwrite discarded the keyframe
        base: the "round-37 pose" then came out with 3 floor contacts instead of 12 and a
        residual of 134.9% of gravity instead of 45.9%.  The offsets are now handed to the same
        function that defines their meaning, so the survey samples the pose it claims to sample.
        """
        doc = {"joint_offsets": {n: float(v) for n, v in zip(joint_names, offsets)}}
        r39.apply_round37_pose(model, data, doc)
        base = data.qpos.copy()
        try:
            base[2] = cp.seat_root(model, data, base)
        except cp.StanceGateError:
            return None
        data.qpos[:] = base
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        try:
            gate = cp.stance_facts(model, data)
        except cp.StanceGateError:
            return None
        return np.asarray(base, float).copy(), gate.as_dict()

    def best_residual(seated, free, seeds=(0.0, 0.5, 1.0)):
        def resid(x):
            a = np.zeros(model.nu)
            a[free] = x
            data.qpos[:] = seated
            data.qvel[:] = 0
            data.ctrl[:] = 0.0
            for i in muscles:
                data.act[actadr[int(i)]] = 0.0
            data.ctrl[:] = a
            for i in muscles:
                data.act[actadr[int(i)]] = float(a[i])
            mujoco.mj_forward(model, data)
            qacc = np.asarray(data.qacc, float).copy()
            mujoco.mj_fullM(model, M, data.qM)
            return np.asarray(M, float) @ qacc
        best = None
        for s in seeds:
            sol = least_squares(resid, np.full(free.size, max(s, lower)), bounds=(lower, 1.0),
                                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=args.max_nfev)
            score = float(np.linalg.norm(sol.fun))
            if best is None or score < best[0]:
                best = (score, sol.x.copy())
        # report the PHYSICAL acceleration for the winning activation, not the weighted residual
        a = np.zeros(model.nu)
        a[free] = best[1]
        data.qpos[:] = seated
        data.qvel[:] = 0
        data.ctrl[:] = a
        for i in muscles:
            data.act[actadr[int(i)]] = float(a[i])
        mujoco.mj_forward(model, data)
        qacc = np.asarray(data.qacc, float).copy()
        return {"weighted_residual_l2": best[0],
                "physical_qacc_l2": float(np.linalg.norm(qacc)),
                "fraction_of_gravity": float(np.linalg.norm(qacc)/GRAVITY),
                "root_translation_l2": float(np.linalg.norm(qacc[:3])),
                "root_rotation_l2": float(np.linalg.norm(qacc[3:6])),
                "leg_joints_l2": float(np.linalg.norm(qacc[6:])),
                "root_vertical_over_gravity": float(qacc[2]/GRAVITY),
                "actuators_at_upper_bound": int((best[1] >= 1-1e-9).sum())}

    base_offsets = np.array([float(pose_doc["joint_offsets"][n]) for n in joint_names], float)
    rng = np.random.default_rng(args.seed)
    samples = [("round37_as_is", base_offsets.copy())]
    for k in range(args.n_perturbations):
        samples.append((f"perturbation_{k:02d}", base_offsets + rng.normal(0, args.sigma_rad,
                                                                          base_offsets.size)))

    rows = []
    t0 = time.time()
    for label, offsets in samples:
        built = build(offsets)
        if built is None:
            rows.append({"label": label, "gate_passed": False})
            print(f"  {label:18s} GATE FAILED (not a stance)", flush=True)
            continue
        seated, gate = built
        row = {"label": label, "gate_passed": True, "n_contact": gate["n_contact"],
               "thorax_height_model_units": float(seated[2]),
               "covered74": best_residual(seated, covered),
               "all90": best_residual(seated, muscles)}
        rows.append(row)
        print(f"  {label:18s} contacts={gate['n_contact']:3d} "
              f"covered74={100*row['covered74']['fraction_of_gravity']:6.2f}% of g  "
              f"all90={100*row['all90']['fraction_of_gravity']:6.2f}% of g", flush=True)

    ok = [r for r in rows if r["gate_passed"]]
    fracs = [r["covered74"]["fraction_of_gravity"] for r in ok]
    best_row = min(ok, key=lambda r: r["covered74"]["fraction_of_gravity"]) if ok else None
    h21 = bool(best_row and best_row["covered74"]["fraction_of_gravity"] < 0.01)
    guard = guard_report("completed") if guard_report else None
    cp.write_result(
        OUT / "static_support_pose_survey.json",
        question=("Across poses sampled from the stance family, does ANY pose become statically "
                  "supportable by the 74 wired actuators, or is the failure systemic?"),
        status="PASS",
        measured={
            "status_meaning": "PASS = the survey ran; the SCIENTIFIC verdict is predictions.H21",
            "protocol": "round-37 joint offsets as the base, seated and gated per pose; "
                        "deterministic Gaussian perturbations of the 42 leg joint offsets",
            "n_poses_requested": len(samples), "n_poses_gated": len(ok),
            "sigma_rad": args.sigma_rad, "seed": args.seed, "max_nfev": args.max_nfev,
            "objective": "M-weighted generalised residual minimised over the actuator set; the "
                         "verdict always reports the PHYSICAL ||qacc|| / gravity",
            "gravity_mm_s2": GRAVITY,
            "rows": rows,
            "covered74_fraction_of_gravity": {
                "min": float(np.min(fracs)) if fracs else None,
                "median": float(np.median(fracs)) if fracs else None,
                "max": float(np.max(fracs)) if fracs else None,
                "values": [float(v) for v in fracs]},
            "best_pose": ({"label": best_row["label"],
                           "fraction_of_gravity": best_row["covered74"]["fraction_of_gravity"]}
                          if best_row else None),
            "predictions": {"H21": {
                "statement": "some pose in the sampled family becomes statically supportable "
                             "(residual below 1% of gravity)",
                "supported": h21,
                "falsifier": "every sampled pose stays far above 1% of gravity, which makes the "
                             "failure systemic to this pose family and points at the historical "
                             "search objective (leg-joint-only) rather than at any single pose"}},
            "memory_guard": guard,
            "wall_s": time.time() - t0,
        },
        provenance={
            "code": str(Path(__file__).resolve()),
            "interpreter": sys.executable,
            "canonical_xml_sha256": cp.sha256_file(cp.CANONICAL_XML),
            "canonical_xml_untouched": True,
            "historical_objective_being_tested": "run_fix_round39_lp_physics.py:50 defines "
                "LEG_SLICE = slice(6, 48) and lines 105-125 minimise the LEG-joint generalised "
                "force only, so the 6 root DOFs are outside the objective while falling and "
                "toppling live exactly there",
            "sampling_is_not_a_search": "this is a deterministic SAMPLE of the family, not an "
                                        "exhaustive stance search; it bounds how pose-specific "
                                        "the failure is, it does not prove no pose is supportable",
        },
        units={"length": "mm", "acceleration": "mm/s^2", "gravity": GRAVITY,
               "force": "model force units (SI conversion INFERRED, not established)"},
        illustrative_vs_measured="poses, gates and residuals are measured; the perturbation "
                                 "distribution, objective and seeds are declared",
        what_this_does_not_support=[
            "A positive H21 would be one supportable pose, not a controller, not a gait.",
            "A negative H21 does NOT prove no supportable pose exists - the family was sampled "
            "around one historical solution, not searched.",
            "Static equilibrium only; no dynamics, no gait, no perturbation rejection.",
            "Passive muscle force and joint stiffness remain inside the model and outside the "
            "actuator command; no attribution between them is attempted.",
        ],
    )
    print(f"best covered74 = {100*min(fracs) if fracs else float('nan'):.2f}% of gravity "
          f"(H21 supported={h21}); wrote {OUT/'static_support_pose_survey.json'}")


if __name__ == "__main__":
    with mg.guarded_run("static_support_pose_survey",
                        **mg.BUDGETS["neural_muscle_closure"]) as _rep:
        main(guard_report=_rep)
