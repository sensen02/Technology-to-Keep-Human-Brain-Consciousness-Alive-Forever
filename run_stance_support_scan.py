"""Static stance supportability scan.

Read-only. The previous audits showed the deployed initial stance bears only 63.8% of body
weight while one foot is airborne. This scan asks a purely static question: within a small,
declared parameter family, does ANY stance exist that carries the full body weight on six
loaded feet without body contact?

Parameters (both declared, neither fitted to a behaviour score):
  dz  : root height offset from the deployed seated pose
  dtib: one common tibia-pitch offset applied to all six legs
All other joint angles stay at the deployed pose. No dynamics, no controller, no gain.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
XML = HERE / 'outputs/model_branches/fine_clock/fruitfly_zero_floor_25us.xml'
OUT = HERE / 'outputs/model_branches/stance_support_scan'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    _, pose, _, joints, geom_leg = closure.prepare_body()
    weight = float(model.body_mass.sum()) * abs(float(model.opt.gravity[2]))
    floor = closure.cp.floor_geom_id(model)
    d = mujoco.MjData(model)
    buf = np.zeros(6)

    def evaluate(qpos):
        d.qpos[:] = qpos
        d.qvel[:] = 0
        d.act[:] = 0
        d.ctrl[:] = 0
        mujoco.mj_forward(model, d)
        foot_v = 0.0
        other_v = 0.0
        other_contacts = 0
        loaded = []
        per_leg = {leg: 0.0 for leg in closure.LEGS}
        penetration = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if floor not in (g1, g2):
                continue
            mujoco.mj_contactForce(model, d, i, buf)
            frame = np.asarray(c.frame).reshape(3, 3)
            vertical = float((frame.T @ buf[:3])[2])
            penetration = min(penetration, float(c.dist))
            leg = geom_leg.get(g1) if g1 in geom_leg else geom_leg.get(g2)
            if leg:
                foot_v += vertical
                per_leg[leg] += vertical
            else:
                other_v += vertical
                other_contacts += 1
        for leg in closure.LEGS:
            if per_leg[leg] > 0.01 * weight:
                loaded.append(leg)
        return dict(support_ratio=foot_v / weight, foot_vertical=foot_v, other_vertical=other_v,
                    body_contacts=other_contacts, loaded_feet=len(loaded), loaded_legs=loaded,
                    per_leg={k: v / weight for k, v in per_leg.items()}, worst_penetration_mm=penetration)

    dq = np.arange(-0.2, 0.5001, 0.005)
    dtib = np.arange(-0.06, 0.0601, 0.005)
    tib_q = [joints[leg]['tib_q'] for leg in closure.LEGS]
    rows = []
    for dz in dq:
        for dtheta in dtib:
            q = pose.copy()
            q[2] = pose[2] + float(dz)
            for i in tib_q:
                q[i] = pose[i] + float(dtheta)
            m = evaluate(q)
            rows.append(dict(dz_mm=float(dz), dtibia_rad=float(dtheta), **m))
    ok = [r for r in rows if r['body_contacts'] == 0 and r['loaded_feet'] == 6]
    best = min(ok, key=lambda r: abs(r['support_ratio'] - 1.0)) if ok else None
    balanced = min(ok, key=lambda r: max(r['per_leg'].values()) - min(r['per_leg'].values())) if ok else None
    result = dict(
        status='MEASURED_STANCE_SUPPORTABILITY_SCAN',
        label='manual read-only static scan, not a controller and not a standing result',
        question='Does a stance exist in this declared 2-parameter family that carries full body weight on six loaded feet?',
        deployed_pose=evaluate(pose),
        parameters=dict(dz_offsets_mm=[float(dq.min()), float(dq.max()), float(dq[1] - dq[0]), len(dq)],
                        tibia_offsets_rad=[float(dtib.min()), float(dtib.max()), float(dtib[1] - dtib[0]), len(dtib)],
                        total_cases=len(rows), declaration='root height and one common tibia-pitch offset only; '
                                                           'all other joints at the deployed pose'),
        best_support_match=best, best_load_balance=balanced,
        feasible_cases=len(ok),
        best_ceiling=dict(max_support_ratio=float(max(r['support_ratio'] for r in rows)),
                          max_loaded_feet=int(max(r['loaded_feet'] for r in rows))),
        weight_model_units=weight,
        provenance=dict(xml=str(XML), xml_sha256=sha(XML)),
        limitations=['Two parameters only; other joints and per-leg asymmetry are held at the deployed pose.',
                     'Static support is necessary but not sufficient for standing.',
                     'No dynamics were run here; a feasible stance would still need a closed-loop episode.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'stance_support_scan.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(OUT / 'stance_support_scan.npz',
                        dz_mm=np.asarray([r['dz_mm'] for r in rows]),
                        dtibia_rad=np.asarray([r['dtibia_rad'] for r in rows]),
                        support_ratio=np.asarray([r['support_ratio'] for r in rows]),
                        body_contacts=np.asarray([r['body_contacts'] for r in rows]),
                        loaded_feet=np.asarray([r['loaded_feet'] for r in rows]),
                        worst_penetration_mm=np.asarray([r['worst_penetration_mm'] for r in rows]))
    print(json.dumps(dict(deployed=result['deployed_pose'], feasible=len(ok),
                          best=best, ceiling=result['best_ceiling']), indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('stance_support_scan', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
