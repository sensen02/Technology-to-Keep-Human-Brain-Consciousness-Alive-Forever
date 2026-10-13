"""Postural sign audit: is the applied torque restoring or anti-restoring?

Read-only on saved states. For every hinge joint the applied actuator torque is compared with
the joint's deviation from the episode's starting pose. A restoring (positional) drive pushes
against the deviation; an anti-restoring drive pushes along it. Signs are reported per joint
with the raw correlation so no single summary hides a mixed picture.
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
OUT = HERE / 'outputs/model_branches/postural_sign'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    with np.load(ALIGN / 'torque_alignment.npz') as z:
        t = z['time_s']
        actual = z['actual']
        required = z['required']
        names = [str(x) for x in z['joint_names']]
    with np.load(TRACE) as z:
        qpos = z['6.25us_normal_body_qpos']
    qadr = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name in names:
                qadr[name] = int(model.jnt_qposadr[j])
    assert set(qadr) == set(names)
    per_joint = {}
    for j, name in enumerate(names):
        dev = qpos[:, qadr[name]] - qpos[0, qadr[name]]
        a = actual[:, j]
        r = required[:, j]
        nz = np.abs(dev) > 1e-9
        corr = float(np.corrcoef(a, dev)[0, 1]) if np.std(a) > 0 and np.std(dev) > 0 else None
        same_sign = int(np.count_nonzero(np.sign(a[nz]) == np.sign(dev[nz])))
        per_joint[name] = dict(
            deviation_final_rad=float(dev[-1]), deviation_max_abs_rad=float(np.abs(dev).max()),
            torque_deviation_correlation=corr,
            fraction_torque_along_deviation=float(same_sign / int(nz.sum())) if nz.any() else None,
            required_deviation_correlation=float(np.corrcoef(r, dev)[0, 1]) if np.std(r) > 0 and np.std(dev) > 0 else None)
    families = defaultdict(list)
    for name in names:
        m = re.match(r'joint_[LR][FMH]([A-Za-z]+)_(yaw|pitch|roll)', name)
        if m:
            families[m.group(1).lower()].append(name)
    family_summary = {}
    for fam, members in families.items():
        corr = [per_joint[m]['torque_deviation_correlation'] for m in members
                if per_joint[m]['torque_deviation_correlation'] is not None]
        frac = [per_joint[m]['fraction_torque_along_deviation'] for m in members
                if per_joint[m]['fraction_torque_along_deviation'] is not None]
        family_summary[fam] = dict(joints=len(members),
                                   mean_correlation=float(np.mean(corr)) if corr else None,
                                   mean_fraction_along_deviation=float(np.mean(frac)) if frac else None)
    anti = sorted([n for n in names if (per_joint[n]['torque_deviation_correlation'] or 0) > 0.3],
                  key=lambda n: -per_joint[n]['torque_deviation_correlation'])
    restoring = sorted([n for n in names if (per_joint[n]['torque_deviation_correlation'] or 0) < -0.3])
    result = dict(
        status='MEASURED_POSTURAL_SIGN',
        label='manual read-only sign audit on saved states, not a controller and not a tuning result',
        question='Does the applied torque push against the joint deviation from the starting pose, or along it?',
        method=dict(deviation='qpos(t) minus qpos(0) for each hinge joint',
                    statistic='Pearson correlation between applied actuator torque and that deviation, plus the '
                              'fraction of samples where torque and deviation share a sign',
                    source=str(ALIGN / 'torque_alignment.npz'), source_sha256=sha(ALIGN / 'torque_alignment.npz'),
                    trace=str(TRACE), trace_sha256=sha(TRACE), samples=int(len(t))),
        per_joint=per_joint, per_family=family_summary,
        summary=dict(joints=int(len(names)),
                     anti_restoring_above_0p3=len(anti), anti_restoring_examples=anti[:8],
                     restoring_below_minus_0p3=len(restoring), restoring_examples=restoring[:8]),
        interpretation=['Positive correlation means the drive acts in the direction the joint has already moved, '
                        'which is an anti-restoring (positive-feedback-like) signature for that joint.',
                        'Torque and deviation are also linked through gravity and contacts, so this is a signature, '
                        'not a proven mechanism.'],
        limitations=['One episode, one pose; joint-axis sign conventions differ per joint and are not normalised away.',
                     'Correlations over a transient cannot separate the controller from the load.',
                     'No parameter, mapping or gain was changed.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'postural_sign.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))
    print(json.dumps(family_summary, indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('postural_sign', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
