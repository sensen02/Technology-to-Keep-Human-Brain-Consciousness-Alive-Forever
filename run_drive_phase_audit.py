"""Drive phase audit: does the neural drive oppose motion (negative feedback) or follow it?

Read-only. Two sharper tests than raw correlation:
  1. detrended alignment: correlation of required and actual torque after removing each series'
     low-frequency trend, so a common sag cannot masquerade as correct control;
  2. phase with motion: correlation of applied torque with the joint's own velocity over the
     episode. Negative means the drive opposes the motion (damping-like); positive means the
     drive acts along the motion.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, re, sys, time, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
ALIGN = HERE / 'outputs/model_branches/torque_alignment'
TRACE = HERE / 'outputs/model_branches/clock_refinement/clock_refinement_traces.npz'
XML = HERE / 'outputs/model_branches/clock_refinement/fruitfly_zero_floor_6.25us.xml'
OUT = HERE / 'outputs/model_branches/drive_phase'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def moving_average(x, w):
    pad = w // 2
    padded = np.pad(x, pad, mode='edge')
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode='valid')[:x.size]


def main(report):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    with np.load(ALIGN / 'torque_alignment.npz') as z:
        t, required, actual = z['time_s'], z['required'], z['actual']
        names = [str(x) for x in z['joint_names']]
    with np.load(TRACE) as z:
        qvel = z['6.25us_normal_body_qvel']
    dof_of = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name in names:
                dof_of[name] = int(model.jnt_dofadr[j])
    missing = [n for n in names if n not in dof_of]
    assert not missing, missing
    per_joint = {}
    for j, name in enumerate(names):
        r, a = required[:, j], actual[:, j]
        v = qvel[:, dof_of[name]]
        wr = moving_average(r, 101)
        wa = moving_average(a, 101)
        rc, ac = r - wr, a - wa
        detrended = float(np.corrcoef(rc, ac)[0, 1]) if np.std(rc) > 0 and np.std(ac) > 0 else None
        phase = float(np.corrcoef(a, v)[0, 1]) if np.std(a) > 0 and np.std(v) > 0 else None
        per_joint[name] = dict(
            raw_correlation=float(np.corrcoef(r, a)[0, 1]) if np.std(r) > 0 and np.std(a) > 0 else None,
            detrended_correlation=detrended,
            torque_velocity_correlation=phase,
            joint_velocity_rms=float(np.sqrt(np.mean(v ** 2))))
    families = defaultdict(list)
    for name in names:
        m = re.match(r'joint_[LR][FMH]([A-Za-z]+)_(yaw|pitch|roll)', name)
        if m:
            families[m.group(1).lower()].append(name)

    def agg(members, key):
        vals = [per_joint[m][key] for m in members if per_joint[m][key] is not None]
        return float(np.mean(vals)) if vals else None

    family_summary = {fam: dict(joints=len(members),
                                mean_detrended_correlation=agg(members, 'detrended_correlation'),
                                mean_torque_velocity_correlation=agg(members, 'torque_velocity_correlation'))
                      for fam, members in families.items()}
    damped = [n for n in names if (per_joint[n]['torque_velocity_correlation'] or 0) < -0.1]
    driven = [n for n in names if (per_joint[n]['torque_velocity_correlation'] or 0) > 0.1]
    dt_corrs = [per_joint[n]['detrended_correlation'] for n in names if per_joint[n]['detrended_correlation'] is not None]
    result = dict(
        status='MEASURED_DRIVE_PHASE',
        label='manual read-only phase audit on saved states, not a controller and not a tuning result',
        question='Does the neural drive oppose the motion (negative feedback) or follow it, once the common trend is removed?',
        method=dict(detrending='moving average over 101 samples (10.1 ms) subtracted from each series',
                    phase='Pearson correlation between applied actuator torque and the joint velocity of the same sample',
                    trace=str(TRACE), trace_sha256=sha(TRACE),
                    alignment_source=str(ALIGN / 'torque_alignment.npz')),
        per_joint=per_joint, per_family=family_summary,
        summary=dict(joints=int(len(names)),
                     mean_detrended_correlation=float(np.mean(dt_corrs)) if dt_corrs else None,
                     median_detrended_correlation=float(np.median(dt_corrs)) if dt_corrs else None,
                     joints_damping_like=len(damped), damping_like_examples=damped[:8],
                     joints_driving_along_motion=len(driven), driving_examples=driven[:8]),
        interpretation=['A high raw correlation that collapses after detrending means the drive follows the global '
                        'sag rather than correcting it.',
                        'A positive torque-velocity correlation means the applied torque acts along the joint motion '
                        'at that instant, i.e. it does not behave as damping for that joint.'],
        limitations=['One episode and one pose; correlations over a transient are not proof of a mechanism.',
                     'Velocity correlation is instantaneous; a delayed corrective response would appear weaker.',
                     'No parameter, gain or mapping was changed.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'drive_phase.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))
    print(json.dumps(family_summary, indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('drive_phase', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
