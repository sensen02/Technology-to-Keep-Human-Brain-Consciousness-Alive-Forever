#!/usr/bin/env python3
"""R28 fixed-keyframe, bounded-muscle FULL-state equilibrium diagnostic.

This is a manually implemented simulation diagnostic, not a standing controller.
Contact magnitudes are simulator diagnostics only: engine/embodied/force_channel.py
forbids physiological force claims (NON_PHYSICAL_LOAD). Its legacy stack-A load
ratios are not measurements of this stack-B model. Body-normal/BW here is only a
model-internal contact-envelope test, not validation against real fly loading.
Thresholds are declared engineering tolerances, NOT validated biological criteria.
All nv accelerations must pass their absolute tolerances. Relative improvement and
optimizer success are never substitutes. Failure to find a candidate is not proof
of infeasibility. A finite unperturbed release is not a mathematical stability test.

Requires explicit --xml and --out (JSON); --selftest needs neither. The default
500 objective-call budget includes numerical Jacobian calls; use a small budget
for development. Does not modify any model, builder, canonical file, or runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "pylibs"))
os.environ.setdefault("MUJOCO_GL", "egl")


def acceleration_assessment(qacc, thresholds):
    a, t = np.asarray(qacc), np.asarray(thresholds)
    r = a / t
    return {"passes_absolute_tolerances": bool(np.all(np.isfinite(r)) and np.all(np.abs(r) <= 1)),
            "qacc_signed": a.tolist(), "normalized_residual_signed": r.tolist(),
            "max_abs_normalized_residual": float(np.max(np.abs(r))),
            "l2_normalized_residual": float(np.linalg.norm(r)),
            "failing_dof_indices": np.flatnonzero(~np.isfinite(r) | (np.abs(r) > 1)).tolist()}


def dof_metadata(m, mj, linear_tol, angular_tol):
    rows = []
    qcovered, vcovered = [], []
    types = {int(mj.mjtJoint.mjJNT_FREE): ("free", 7, ["linear"] * 3 + ["angular"] * 3),
             int(mj.mjtJoint.mjJNT_BALL): ("ball", 4, ["angular"] * 3),
             int(mj.mjtJoint.mjJNT_SLIDE): ("slide", 1, ["linear"]),
             int(mj.mjtJoint.mjJNT_HINGE): ("hinge", 1, ["angular"])}
    for j in range(m.njnt):
        kind, nq, units = types[int(m.jnt_type[j])]
        qa, va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        qcovered.extend(range(qa, qa + nq))
        for local, unit in enumerate(units):
            vcovered.append(va + local)
            rows.append({"dof": va + local, "joint_id": j,
                         "joint": mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j),
                         "joint_type": kind, "qpos_address": qa, "component": local,
                         "quantity": unit, "acceleration_unit": "model_length/s^2" if unit == "linear" else "rad/s^2",
                         "absolute_acceleration_tolerance": linear_tol if unit == "linear" else angular_tol})
    if sorted(qcovered) != list(range(m.nq)) or sorted(vcovered) != list(range(m.nv)):
        raise ValueError("joint address mapping does not cover nq/nv exactly once")
    return sorted(rows, key=lambda r: r["dof"])


def actuator_mapping(m, mj):
    """Support standard one-state muscle dynamics only; reject rather than guess."""
    rows, occupied, lo, hi = [], [], [], []
    for i in range(m.nu):
        adr, num = int(m.actuator_actadr[i]), int(m.actuator_actnum[i])
        if (int(m.actuator_dyntype[i]) != int(mj.mjtDyn.mjDYN_MUSCLE)
                or int(m.actuator_gaintype[i]) != int(mj.mjtGain.mjGAIN_MUSCLE)
                or num != 1 or not 0 <= adr < m.na):
            raise ValueError(f"actuator {i}: unsupported mapping/dynamics; expected one-state standard muscle")
        lower, upper = 0.0, 1.0
        if m.actuator_ctrllimited[i]:
            lower = max(lower, float(m.actuator_ctrlrange[i, 0]))
            upper = min(upper, float(m.actuator_ctrlrange[i, 1]))
        if m.actuator_actlimited[i]:
            lower = max(lower, float(m.actuator_actrange[i, 0]))
            upper = min(upper, float(m.actuator_actrange[i, 1]))
        if lower > upper:
            raise ValueError(f"actuator {i}: empty consistent control/activation bounds")
        occupied.append(adr)
        lo.append(lower)
        hi.append(upper)
        rows.append({"actuator": i, "name": mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i),
                     "activation_address": adr, "activation_count": num,
                     "dyn_type": int(m.actuator_dyntype[i]), "gain_type": int(m.actuator_gaintype[i]),
                     "transmission_type": int(m.actuator_trntype[i]),
                     "transmission_ids": m.actuator_trnid[i].tolist(), "bounds": [lower, upper]})
    if not rows or sorted(occupied) != list(range(m.na)):
        raise ValueError("activation mapping must cover all na states exactly once; nu=na is not assumed")
    return rows, np.array(lo), np.array(hi)


class EvaluationBudget(Exception):
    pass


def release(m, mj, fresh, x, rows, args):
    d = fresh(x)
    th = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, args.body)
    if th < 0:
        raise ValueError(f"missing release reference body {args.body!r}")
    up0 = d.xmat[th].reshape(3, 3)[:, 2].copy()
    linear = np.array([r["quantity"] == "linear" for r in rows])
    bw = float(np.sum(m.body_mass) * np.linalg.norm(m.opt.gravity))
    if bw <= 0:
        raise ValueError("release body-weight normalization requires nonzero mass and gravity")
    ground = set(np.flatnonzero(m.geom_type == int(mj.mjtGeom.mjGEOM_PLANE)))
    if not ground:
        raise ValueError("release support classifier requires plane ground")
    legs = ("LF", "LM", "LH", "RF", "RM", "RH")
    geom_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    trace, failures = [], []
    nsteps = int(math.ceil(args.release_seconds / m.opt.timestep))
    initial_warning = np.array(d.warning.number).copy()
    for step in range(nsteps + 1):
        if step:
            mj.mj_step(m, d)
            # mj_step's derived quantities need not describe its newly integrated qpos.
            mj.mj_forward(m, d)
        if (not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel))
                or not np.all(np.isfinite(d.qacc)) or np.any(d.warning.number > initial_warning)
                or abs(float(d.time) - step * m.opt.timestep) > 1e-8):
            failures.append({"step": step, "reason": "nonfinite state, MuJoCo warning, or unexpected time reset"})
            break
        body_normal, leg_normal, contacts = 0.0, 0.0, set()
        force = np.zeros(6)
        for c in range(d.ncon):
            a, b = int(d.contact[c].geom1), int(d.contact[c].geom2)
            if (a in ground) == (b in ground):
                continue
            other = b if a in ground else a
            mj.mj_contactForce(m, d, c, force)
            normal = abs(float(force[0]))
            if geom_names[other][:2] in legs:
                leg_normal += normal
                if normal > 0:
                    contacts.add(geom_names[other][:2])
            else:
                body_normal += normal
        up = d.xmat[th].reshape(3, 3)[:, 2]
        row = {"time_s": float(d.time), "height_model_length": float(d.xpos[th, 2]),
               "tilt_from_initial_deg": float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1)))),
               "body_ground_normal_bw": body_normal / bw, "leg_ground_normal_bw": leg_normal / bw,
               "legs_with_positive_normal_force": sorted(contacts),
               "max_abs_linear_qvel": float(np.max(np.abs(d.qvel[linear]), initial=0)),
               "max_abs_angular_qvel": float(np.max(np.abs(d.qvel[~linear]), initial=0))}
        checks = {"height": row["height_model_length"] >= args.min_height,
                  "body_support": row["body_ground_normal_bw"] <= args.max_body_bw,
                  "tilt": row["tilt_from_initial_deg"] <= args.max_tilt_deg,
                  "linear_velocity": row["max_abs_linear_qvel"] <= args.max_linear_velocity,
                  "angular_velocity": row["max_abs_angular_qvel"] <= args.max_angular_velocity}
        row["violations"] = [k for k, passed in checks.items() if not passed]
        trace.append(row)
    violated = [r for r in trace if r["violations"]]
    return {"status": "WITHIN_DECLARED_ENVELOPE" if not failures and not violated else "LEFT_DECLARED_ENVELOPE",
            "requested_duration_s": args.release_seconds, "evaluated_duration_s": trace[-1]["time_s"] if trace else 0,
            "sampling": "initial state and every integration timestep; includes final ceil-duration step",
            "support_classifier": "plane contacts, LF/LM/LH/RF/RM/RH geom prefixes are leg; other geoms are body; summed absolute normal force, not net vertical force",
            "numerical_failures": failures, "first_envelope_violation": violated[0] if violated else None,
            "extrema": {"min_height": min(r["height_model_length"] for r in trace),
                        **{f"peak_{k}": max(r[k] for r in trace) for k in
                           ("tilt_from_initial_deg", "body_ground_normal_bw", "max_abs_linear_qvel", "max_abs_angular_qvel")}} if trace else {},
            "trace": trace,
            "interpretation": "Finite unperturbed constant-control trajectory only; neither mathematical stability nor instability established."}


def run(args):
    import mujoco as mj
    from scipy.optimize import least_squares
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml = Path(args.xml).resolve(strict=True)
    m = _load_mjcf(str(xml)).compile()
    packed = np.empty(mj.mj_sizeModel(m), dtype=np.uint8)
    mj.mj_saveModel(m, None, packed)
    base = {"schema": "full_state_equilibrium_r28_v1", "xml": str(xml),
            "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
            "compiled_model_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
            "mujoco_version": mj.__version__,
            "dimensions": {k: int(getattr(m, k)) for k in ("nq", "nv", "nu", "na", "neq", "njnt")},
            "declarations": vars(args).copy(),
            "threshold_label": "USER_CONFIGURABLE_ENGINEERING_TOLERANCES_NOT_BIOLOGICALLY_VALIDATED",
            "unit_declaration": "MuJoCo is unitless; linear thresholds use XML model length. Project fly models use mm; angular joint rates use radians.",
            "limitations": "Simulator diagnostic, not neural output, walking, global feasibility, or stability proof. Fixed keyframe pose only."}
    # Conservative: refuse ALL equality constraints, including inactive ones, not just named RF locks.
    eqs = [{"id": i, "name": mj.mj_id2name(m, mj.mjtObj.mjOBJ_EQUALITY, i),
            "type": int(m.eq_type[i]), "active_at_reset": bool(m.eq_active0[i])} for i in range(m.neq)]
    base["lock_check"] = {"policy": "require zero equality constraints, including inactive definitions", "equalities": eqs}
    if eqs:
        return {**base, "status": "INVALID_MODEL", "error": "Equality definitions remain; unlocked model required"}
    if m.opt.disableflags & (int(mj.mjtDisableBit.mjDSBL_ACTUATION) | int(mj.mjtDisableBit.mjDSBL_CONTACT)):
        raise ValueError("actuation or contacts disabled")
    if not 0 <= args.keyframe < m.nkey:
        raise ValueError("requested keyframe does not exist")
    rows = dof_metadata(m, mj, args.linear_accel_tol, args.angular_accel_tol)
    mapping, lower, upper = actuator_mapping(m, mj)
    adr = np.array([r["activation_address"] for r in mapping])
    tol = np.array([r["absolute_acceleration_tolerance"] for r in rows])
    q = m.key_qpos[args.keyframe].copy()

    def fresh(x):
        d = mj.MjData(m)  # NEW data for every objective: no solver warmstart/history reuse.
        mj.mj_resetDataKeyframe(m, d, args.keyframe)
        d.qpos[:] = q
        d.qvel[:] = 0
        d.qacc_warmstart[:] = 0
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        d.act[:] = 0
        d.act[adr] = x
        d.ctrl[:] = x  # bounds intersect ctrlrange; avoids act=0 with clipped ctrl=1e-4.
        d.time = 0
        mj.mj_forward(m, d)
        return d

    x0 = lower + args.initial_activation * (upper - lower)
    initial = fresh(x0)
    best_x, best_cost = x0.copy(), float("inf")
    calls, improvements = 0, []
    free = np.flatnonzero(upper > lower)

    def objective(y):
        nonlocal best_x, best_cost, calls
        if calls >= args.max_evaluations:
            raise EvaluationBudget()
        calls += 1
        x = lower.copy()
        x[free] = y
        r = fresh(x).qacc.copy() / tol
        if not np.all(np.isfinite(r)):
            raise ValueError("nonfinite objective acceleration")
        cost = float(r @ r)
        if cost < best_cost:
            best_x, best_cost = x.copy(), cost
            improvements.append({"evaluation": calls, "sum_squared_normalized_residual": cost})
        return r

    optimizer = {"method": "scipy.optimize.least_squares, bounds, numerical Jacobian",
                 "budget_semantics": "hard cap on ALL objective calls including finite differences",
                 "success": False, "status": None}
    try:
        if free.size:
            result = least_squares(objective, x0[free], bounds=(lower[free], upper[free]),
                                   max_nfev=args.max_evaluations, diff_step=1e-3)
            optimizer.update(success=bool(result.success), status=int(result.status), message=str(result.message),
                             nfev=int(result.nfev), njev=int(result.njev or 0), optimality=float(result.optimality))
        else:
            objective(np.array([]))
            optimizer.update(status="NO_FREE_VARIABLES", message="All activation bounds fixed; evaluated single candidate")
    except EvaluationBudget:
        optimizer.update(status="EVALUATION_BUDGET_EXHAUSTED", message="Best evaluated candidate retained; infeasibility NOT established")
    except Exception as exc:
        optimizer.update(status="OPTIMIZER_EXCEPTION", message=f"{type(exc).__name__}: {exc}; infeasibility NOT established")
    optimizer.update(objective_calls=calls, improvement_history=improvements)
    final = fresh(best_x)
    repeated = fresh(best_x)
    assessment = acceleration_assessment(final.qacc.copy(), tol)
    dynamics_pass = bool(np.all(np.isfinite(final.act_dot)) and np.all(np.abs(final.act_dot) <= args.act_dot_tol))
    base.update(status="DIAGNOSTIC_COMPLETED", dofs=rows, actuator_mapping=mapping,
                target_qpos=q.tolist(), target_qvel=np.zeros(m.nv).tolist(),
                initial_activation=x0.tolist(), activation_by_actuator=best_x.tolist(),
                final_act=final.act.tolist(), final_ctrl=final.ctrl.tolist(),
                objective="minimize sum over ALL nv of (signed qacc / declared absolute acceleration tolerance)^2",
                data_reset_policy="fresh MjData + reset keyframe + explicit zero qvel/applied force/warmstart + mapped act and matching ctrl at every objective call",
                baseline=acceleration_assessment(initial.qacc.copy(), tol), optimizer=optimizer,
                instantaneous_equilibrium=assessment,
                activation_dynamics={"passes_absolute_tolerance": dynamics_pass, "act_dot_signed_per_s": final.act_dot.tolist(),
                                     "absolute_tolerance_per_s": args.act_dot_tol},
                repeat_evaluation_max_abs_qacc_difference=float(np.max(np.abs(final.qacc - repeated.qacc))),
                generalized_force_terms={k: getattr(final, k).tolist() for k in
                                         ("qfrc_actuator", "qfrc_passive", "qfrc_bias", "qfrc_constraint")},
                equilibrium_verdict="FULL_STATE_WITHIN_DECLARED_TOLERANCES" if assessment["passes_absolute_tolerances"] and dynamics_pass
                else "CANDIDATE_FAILS_DECLARED_TOLERANCES_NOT_AN_INFEASIBILITY_PROOF")
    base["release"] = release(m, mj, fresh, best_x, rows, args)
    return base


def selftest():
    # Old root-only criterion passes; full-state correctly fails on a leg hinge.
    a = np.zeros(48)
    a[6] = 2
    assert np.all(a[:6] == 0)
    assert not acceleration_assessment(a, np.ones(48))["passes_absolute_tolerances"]
    # 99% relative decrease remains 100 times above the absolute threshold.
    before, after = np.full(48, 10000.), np.full(48, 100.)
    assert np.linalg.norm(after) / np.linalg.norm(before) < 0.1
    assert not acceleration_assessment(after, np.ones(48))["passes_absolute_tolerances"]
    assert acceleration_assessment(np.array([-1., 1.]), np.ones(2))["passes_absolute_tolerances"]
    assert not acceleration_assessment(np.array([np.nan]), np.ones(1))["passes_absolute_tolerances"]
    import mujoco as mj
    m = mj.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/><body pos="0 0 1"><joint type="ball"/><geom size=".1"/><body pos="0 0 1"><joint type="slide"/><geom size=".1"/><body pos="0 0 1"><joint type="hinge"/><geom size=".1"/></body></body></body></body></worldbody></mujoco>')
    rows = dof_metadata(m, mj, 2, 3)
    assert m.nq == 13 and m.nv == 11 and len(rows) == m.nv
    assert [r["quantity"] for r in rows].count("linear") == 4
    print("PASS: root-zero/leg-nonzero rejected; relative-only improvement rejected; signed boundary/nonfinite checks; free/ball/slide/hinge nq/nv mapping")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", help="explicit input XML (never modified)")
    ap.add_argument("--out", help="explicit output JSON (must differ from XML)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--keyframe", type=int, default=0)
    ap.add_argument("--max-evaluations", type=int, default=500)
    ap.add_argument("--initial-activation", type=float, default=0.1, help="fraction of each allowed activation interval")
    ap.add_argument("--linear-accel-tol", type=float, default=1., help="absolute model_length/s^2, engineering tolerance")
    ap.add_argument("--angular-accel-tol", type=float, default=1., help="absolute rad/s^2, engineering tolerance")
    ap.add_argument("--act-dot-tol", type=float, default=1e-8, help="absolute activation/s")
    ap.add_argument("--release-seconds", type=float, default=0.5)
    ap.add_argument("--body", default="Thorax")
    ap.add_argument("--min-height", type=float, default=1.2, help="model length; standing-envelope choice, not equilibrium tolerance")
    ap.add_argument("--max-body-bw", type=float, default=1e-3)
    ap.add_argument("--max-tilt-deg", type=float, default=5.)
    ap.add_argument("--max-linear-velocity", type=float, default=1., help="model_length/s over entire release")
    ap.add_argument("--max-angular-velocity", type=float, default=1., help="rad/s over entire release")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.xml or not args.out:
        ap.error("--xml and --out are required except with --selftest")
    if Path(args.xml).resolve() == Path(args.out).resolve():
        ap.error("--out must not overwrite --xml")
    for name in ("linear_accel_tol", "angular_accel_tol", "act_dot_tol", "release_seconds",
                 "max_body_bw", "max_tilt_deg", "max_linear_velocity", "max_angular_velocity"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.max_evaluations < 1 or not 0 <= args.initial_activation <= 1 or not math.isfinite(args.min_height):
        ap.error("invalid budget, initial activation, or minimum height")
    try:
        doc = run(args)
    except Exception as exc:
        doc = {"schema": "full_state_equilibrium_r28_v1", "status": "DIAGNOSTIC_ERROR",
               "error": f"{type(exc).__name__}: {exc}", "declarations": vars(args)}
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("status", "equilibrium_verdict", "error") if k in doc}))
    print(f"wrote {out}")
    return 0 if doc["status"] == "DIAGNOSTIC_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
