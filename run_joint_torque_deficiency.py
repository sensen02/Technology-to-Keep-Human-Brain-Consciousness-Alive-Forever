"""Joint torque deficiency audit: required static hold torque vs actual actuator torque.

Read-only on saved states of the 25 us normal branch episode. No integration, no
parameter change, no controller. For every saved sample the full generalized-force
decomposition is checked against M*qacc before any per-joint number is reported.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, signal, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TRACE = HERE / 'outputs/model_branches/clock_refinement/clock_refinement_traces.npz'
XML = HERE / 'outputs/model_branches/clock_refinement/fruitfly_zero_floor_6.25us.xml'
OUT = HERE / 'outputs/model_branches/torque_deficiency'
SNAPSHOTS_MS = (0, 10, 24, 50, 95, 200)
START = time.time()


def main(report):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    with np.load(TRACE) as z:
        qpos = z['6.25us_normal_body_qpos']
        qvel = z['6.25us_normal_body_qvel']
        act = z['6.25us_normal_body_act']
        t = z['6.25us_normal_body_time_s']
    n = qpos.shape[0]
    res = np.zeros(model.nv)
    rows = {k: np.zeros((n, model.nv)) for k in ('bias', 'passive', 'actuator', 'constraint', 'mqacc')}
    residual = np.zeros(n)
    for i in range(n):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        data.act[:] = act[i]
        data.ctrl[:] = 0
        mujoco.mj_forward(model, data)
        rows['bias'][i] = data.qfrc_bias
        rows['passive'][i] = data.qfrc_passive
        rows['actuator'][i] = data.qfrc_actuator
        rows['constraint'][i] = data.qfrc_constraint
        mujoco.mj_mulM(model, data, res, data.qacc)
        rows['mqacc'][i] = res
        # MuJoCo convention: M*qacc + qfrc_bias = qfrc_actuator + qfrc_passive + qfrc_constraint (+applied).
        # Residual of that identity is checked before any per-joint number is reported.
        residual[i] = float(np.abs(rows['mqacc'][i] + rows['bias'][i] - rows['passive'][i]
                                   - rows['actuator'][i] - rows['constraint'][i]).max())
    weight = float(model.body_mass.sum()) * abs(float(model.opt.gravity[2]))
    hinge = [d for d in range(model.nv)
             if model.jnt_type[model.dof_jntid[d]] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.dof_jntid[d])) for d in hinge]
    # Static-hold requirement: actuator torque that would make qacc zero, i.e. bias - passive - constraint.
    required = (rows['bias'] - rows['passive'] - rows['constraint'])[:, hinge]
    actual = rows['actuator'][:, hinge]
    deficit = required - actual
    snap = {}
    for ms in SNAPSHOTS_MS:
        i = min(int(round(ms * 1e-3 / (t[1] - t[0]))), n - 1)
        snap[str(ms)] = dict(
            sample=i, time_s=float(t[i]),
            max_required_abs=float(np.abs(required[i]).max()),
            max_actual_abs=float(np.abs(actual[i]).max()),
            max_deficit_abs=float(np.abs(deficit[i]).max()),
            required_over_weight_max=float(np.abs(required[i]).max() / weight),
            coverage=float(np.abs(actual[i]).sum() / np.abs(required[i]).sum()) if np.abs(required[i]).sum() else None,
            worst_joints=[{'joint': names[j], 'required': float(required[i][j]), 'actual': float(actual[i][j]),
                           'deficit': float(deficit[i][j])}
                          for j in np.argsort(-np.abs(deficit[i]))[:5]],
        )
    leg_of = {}
    for j, name in enumerate(names):
        for leg in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh'):
            if name.lower().startswith('joint_' + leg):
                leg_of.setdefault(leg, []).append(j)
    per_leg = {}
    for leg, idx in leg_of.items():
        per_leg[leg] = dict(max_deficit_abs=float(np.abs(deficit[:, idx]).max()),
                            max_required_abs=float(np.abs(required[:, idx]).max()),
                            coverage_at_0ms=float(np.abs(actual[0, idx]).sum() / np.abs(required[0, idx]).sum()) if np.abs(required[0, idx]).sum() else None,
                            coverage_at_200ms=float(np.abs(actual[-1, idx]).sum() / np.abs(required[-1, idx]).sum()) if np.abs(required[-1, idx]).sum() else None)
    result = dict(
        status='MEASURED_JOINT_TORQUE_DEFICIENCY',
        label='manual read-only torque decomposition on saved states, not a controller and not a standing proof',
        question='For each leg joint, how much generalized torque is needed to hold the body, and how much does the actuator supply?',
        method=dict(requirement='static-hold condition actuator = -(bias + passive + constraint); qacc is not forced to zero, so this is a per-instant comparison against a hold posture',
                    balance_check='max |bias + passive + actuator + constraint - M*qacc| reported per sample; numbers are only reported if this residual is at floating-point level',
                    integrator='none; mj_forward on saved qpos/qvel/act, ctrl irrelevant to muscle force given act',
                    trace=str(TRACE), xml=str(XML), samples=int(n), sample_dt_s=float(t[1] - t[0])),
        weight_model_units=weight,
        balance_residual=dict(max_abs=float(residual.max()), median_abs=float(np.median(residual)),
                              tolerance=1e-6 * weight, within_tolerance=bool(residual.max() < 1e-6 * weight)),
        hinge_joint_names=names,
        per_leg=per_leg,
        snapshots_ms=snap,
        whole_episode=dict(max_required_abs=float(np.abs(required).max()), max_actual_abs=float(np.abs(actual).max()),
                           max_deficit_abs=float(np.abs(deficit).max()),
                           max_deficit_over_weight=float(np.abs(deficit).max() / weight),
                           coverage_min=float(min((np.abs(actual[i]).sum() / np.abs(required[i]).sum()) if np.abs(required[i]).sum() else 1.0 for i in range(n))),
                           time_of_max_deficit_s=float(t[int(np.argmax(np.abs(deficit).max(axis=1)))]),
                           joints_exceeding_weight_at_any_time=int(np.count_nonzero(np.abs(deficit).max(axis=0) > weight))),
        limitations=['Saved-state re-evaluation of one 25 us normal episode; no new dynamics.',
                     'A deficit is a torque shortfall relative to holding the posture, not proof that the posture is achievable.',
                     'Constraint and passive terms are taken as recorded; no term is reassigned.',
                     'Absolute units follow the unresolved mass-unit reading; all ratios are unit-invariant.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'torque_deficiency_results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(OUT / 'torque_deficiency_traces.npz', time_s=t, required=required, actual=actual,
                        deficit=deficit, joint_names=np.asarray(names), weight=weight)
    print(json.dumps({k: result[k] for k in ('balance_residual', 'per_leg', 'whole_episode')}, indent=2))
    return result


if __name__ == '__main__':
    import run_neural_muscle_closure as closure
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60s torque-audit deadline')))
    signal.alarm(60)
    with closure.mg.guarded_run('torque_deficiency', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
    signal.alarm(0)
