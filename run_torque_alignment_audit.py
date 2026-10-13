"""Torque alignment audit: is the applied joint torque merely weak, or systematically misdirected?

Read-only on the saved torque-deficiency decomposition. For every hinge joint it compares the
applied actuator torque with the static-hold requirement over the episode: correlation, mean
projection (cosine) and the fraction of samples where the drive opposes the requirement.
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
SRC = HERE / 'outputs/model_branches/torque_deficiency'
OUT = HERE / 'outputs/model_branches/torque_alignment'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    with np.load(SRC / 'torque_deficiency_traces.npz') as z:
        t = z['time_s']
        required = z['required']
        actual = z['actual']
        names = [str(x) for x in z['joint_names']]
        weight = float(z['weight'])
    doc = json.loads((SRC / 'torque_deficiency_results.json').read_text())
    assert doc['balance_residual']['within_tolerance'], 'upstream decomposition did not pass its own check'
    n_j, n_t = required.shape[1], required.shape[0]
    per_joint = {}
    for j, name in enumerate(names):
        r, a = required[:, j], actual[:, j]
        rn, an = np.linalg.norm(r), np.linalg.norm(a)
        corr = float(np.corrcoef(r, a)[0, 1]) if rn > 0 and an > 0 else None
        proj = float(np.dot(a, r) / (an * rn)) if rn > 0 and an > 0 else None
        active = np.abs(r) > 1e-12
        opposed = int(np.count_nonzero(np.sign(a[active]) != np.sign(r[active])))
        per_joint[name] = dict(
            required_rms=float(np.sqrt(np.mean(r ** 2))), actual_rms=float(np.sqrt(np.mean(a ** 2))),
            ratio_rms=float(np.sqrt(np.mean(a ** 2)) / np.sqrt(np.mean(r ** 2))) if rn > 0 else None,
            correlation=corr, mean_projection_cosine=proj,
            opposed_fraction=float(opposed / int(active.sum())) if active.any() else None,
            active_samples=int(active.sum()))
    families = defaultdict(list)
    for name in names:
        m = re.match(r'joint_[LR][FMH]([A-Za-z]+)_(yaw|pitch|roll)', name)
        if m:
            families[m.group(1).lower()].append(name)
    family_summary = {}
    for fam, members in families.items():
        corrs = [per_joint[m]['correlation'] for m in members if per_joint[m]['correlation'] is not None]
        cosines = [per_joint[m]['mean_projection_cosine'] for m in members if per_joint[m]['mean_projection_cosine'] is not None]
        ratios = [per_joint[m]['ratio_rms'] for m in members if per_joint[m]['ratio_rms'] is not None]
        opposed = [per_joint[m]['opposed_fraction'] for m in members if per_joint[m]['opposed_fraction'] is not None]
        family_summary[fam] = dict(joints=len(members),
                                   mean_correlation=float(np.mean(corrs)) if corrs else None,
                                   mean_cosine=float(np.mean(cosines)) if cosines else None,
                                   mean_rms_ratio=float(np.mean(ratios)) if ratios else None,
                                   mean_opposed_fraction=float(np.mean(opposed)) if opposed else None)
    anti = sorted([n for n in names if (per_joint[n]['correlation'] or 0) < 0])
    result = dict(
        status='MEASURED_TORQUE_ALIGNMENT',
        label='manual read-only alignment audit on saved states, not a controller and not a tuning result',
        question='Is the applied joint torque weak, or systematically pushing against the hold requirement?',
        method=dict(required='static-hold requirement from the upstream torque-deficiency audit (balance identity verified there)',
                    alignment='per-joint Pearson correlation between required(t) and actual(t), mean projection cosine, '
                              'and the fraction of samples whose sign opposes the requirement',
                    source=str(SRC / 'torque_deficiency_traces.npz'), source_sha256=sha(SRC / 'torque_deficiency_traces.npz'),
                    upstream_balance_residual_max_abs=doc['balance_residual']['max_abs'],
                    samples=int(n_t), joints=n_j, duration_s=float(t[-1])),
        per_joint=per_joint,
        per_family=family_summary,
        summary=dict(joints_anti_correlated=len(anti), anti_correlated_joints=anti,
                     joints_positive_correlation=n_j - len(anti),
                     median_correlation=float(np.median([per_joint[n]['correlation'] for n in names
                                                         if per_joint[n]['correlation'] is not None]))),
        interpretation=['A positive ratio with a negative correlation means the joint received torque of roughly the '
                        'right magnitude aimed the wrong way at the wrong time, which no gain change can fix.',
                        'A positive correlation with a small ratio is a strength or timing shortfall instead.'],
        limitations=['One episode, one pose, one timestep; required is a per-instant hold target, not a reachability proof.',
                     'Correlation over 2001 samples of a transient does not establish a single cause.',
                     'Units are the unresolved model units; the projection terms are unit-free.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'torque_alignment.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(OUT / 'torque_alignment.npz', time_s=t, required=required, actual=actual,
                        joint_names=np.asarray(names))
    print(json.dumps(result['summary'], indent=2))
    print(json.dumps(family_summary, indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('torque_alignment', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
