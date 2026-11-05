"""Give the joint-torque plant the authority the demonstrated warmup actually used.

The muscle-matched plant cannot hold this body: with per-joint limits set to the better of the two
muscle-reachable magnitudes, the torque the assisted trajectory received is still out of range at
111-126 of 1050 sampled DOF-instants, and the warmup itself applies up to 4.45x the muscle limit at
the joints it loads most (joint_RMCoxa_yaw: 12.50 applied vs 2.81 available).

So there are two honest options and they must not be confused:

  muscle-matched  the plant inherits the muscles' strength ceiling. It fails, and that failure IS the
                  finding: in this body model the muscles are under-powered for the loads the only
                  working controller imposes. Kept as MODEL_JOINT_MUSCLE_MATCHED.xml.

  assist-matched  the plant inherits the strength ceiling of the demonstrated warmup instead. It can
                  reproduce a motion this body has actually been shown to perform, but its authority
                  is up to 4.45x what the muscles were given. Any walking result from it is therefore
                  an IDEALIZED-ACTUATOR result and says nothing about fly muscle capability.

Limit rule used here: limit_j = the largest |total hinge torque the warmup applied| at joint j over
its 1.3 s assisted window, where the total is the recorded assist torque plus the muscle torque the
same step carried. Nothing above what the demonstrated controller used is granted.

Usage: set_joint_limits_from_assist.py <frozen RUN dir> <plant dir>
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R)]
EVERY = 1           # every 0.5 ms body sample: subsampling misses instantaneous torque peaks


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2,
                                  default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)) + '\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    src_run, plant_dir = Path(sys.argv[1]), Path(sys.argv[2])
    muscle_matched = plant_dir / 'MODEL_JOINT_MUSCLE_MATCHED.xml'
    before = {str(p): sha(p) for p in (src_run / 'MODEL.xml', muscle_matched)}

    body = dict(np.load(src_run / 'BODY.npz'))
    model = _load_mjcf(src_run / 'MODEL.xml').compile()
    data = mujoco.MjData(model)
    hinges = np.array([j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE])
    va = np.array([model.jnt_dofadr[j] for j in hinges])
    names = [model.joint(int(j)).name for j in hinges]
    t0 = float(json.loads((src_run / 'NEURAL_RECORDS.json').read_text())[0]['absolute_start_s'])
    win = np.flatnonzero((body['time'] <= t0) & (body['time'] > t0 - 1.3))[::EVERY]

    def tau(act):
        data.act[:] = act
        mujoco.mj_forward(model, data)
        return data.qfrc_actuator.copy()

    seen = np.zeros(42)
    for k in win:
        data.qpos[:] = body['qpos'][k]
        data.qvel[:] = body['qvel'][k]
        data.ctrl[:] = 0.0
        total = (tau(body['activation'][k].copy()) + body['assist'][k])[va]
        seen = np.maximum(seen, np.abs(total))
    if not np.all(np.isfinite(seen)) or np.any(seen <= 0):
        raise SystemExit('degenerate assist envelope')

    src_text = (src_run / 'MODEL.xml').read_text()
    block = re.search(r'<actuator>.*?</actuator>', src_text, re.S)
    lines = ['<actuator>']
    for j, nm in enumerate(names):
        lines.append('   <motor name="jt_%s" joint="%s" gear="1" ctrlrange="%s %s" />'
                     % (nm, nm, format(-seen[j], '.17g'), format(seen[j], '.17g')))
    lines.append('</actuator>')
    (plant_dir / 'MODEL_JOINT_TORQUE.xml').write_text(
        src_text[:block.start()] + '\n'.join(lines) + src_text[block.end():])

    m2 = _load_mjcf(plant_dir / 'MODEL_JOINT_TORQUE.xml').compile()
    d2 = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d2, 0)
    mujoco.mj_forward(m2, d2)
    ctrl = np.array([m2.actuator_ctrlrange[a] for a in range(m2.nu)])
    checks = dict(nu=int(m2.nu), nq=int(m2.nq), nv=int(m2.nv), nmesh=int(m2.nmesh),
                  nu_ok=m2.nu == 42, body_ok=m2.nbody == model.nbody, nv_ok=m2.nv == model.nv,
                  limits_equal=bool(np.allclose(ctrl, np.stack([-seen, seen], 1), rtol=0, atol=0)),
                  mass_ok=bool(abs(m2.body_mass.sum() - model.body_mass.sum()) == 0.0),
                  timestep_ok=bool(m2.opt.timestep == model.opt.timestep),
                  trn=sorted({mujoco.mjtTrn(int(t)).name for t in m2.actuator_trntype}))
    if not all(v for k, v in checks.items() if k.endswith('_ok') or k.endswith('_equal')):
        raise SystemExit('assist-matched model failed verification: %r' % checks)

    for _ in range(4000):
        d2.ctrl[:] = 0.0
        mujoco.mj_step(m2, d2)
    smoke = dict(steps=4000, finite=bool(np.isfinite(d2.qpos).all()),
                 thorax_com_z=round(float(d2.subtree_com[m2.body('Thorax').id, 2]), 4))

    plant = json.loads((plant_dir / 'JOINT_PLANT.json').read_text())
    muscle_lim = np.array([j['limit_torque'] for j in plant['joints']])
    dump(plant_dir / 'JOINT_PLANT_LIMITS.json', dict(
        two_rules=dict(
            muscle_matched=dict(model='MODEL_JOINT_MUSCLE_MATCHED.xml',
                                rule='per joint, the larger of the two muscle-reachable magnitudes at the keyframe posture',
                                verdict='cannot hold the body; the warmup exceeds it at 31 of 42 joints, worst ratio 4.45',
                                use='the control that establishes the muscles are under-powered'),
            assist_matched=dict(model='MODEL_JOINT_TORQUE.xml',
                                rule='per joint, the largest |assist + muscle torque| the warmup applied over its 1.3 s window',
                                verdict='can reproduce a motion this body has been shown to perform',
                                use='the working plant for the neural readout'))
        if True else None,
        assist_matched_limits={names[j]: float(seen[j]) for j in range(42)},
        muscle_matched_limits={names[j]: float(muscle_lim[j]) for j in range(42)},
        authority_ratio={names[j]: float(seen[j] / muscle_lim[j]) for j in range(42)},
        authority_ratio_max=float((seen / muscle_lim).max()),
        authority_ratio_median=float(np.median(seen / muscle_lim)),
        joints_where_assist_exceeds_muscles=int((seen > muscle_lim + 1e-9).sum()),
        states_sampled=int(win.size), subsample_stride=EVERY,
        verification=checks, smoke=smoke,
        claim_boundary='assist-matched actuation is an IDEALIZED-ACTUATOR result. It does not show fly '
                       'muscles can do this, and the ratio above is the size of the gap.',
        units='model native; absolute forces and SI NOT_ESTABLISHED', importance='核心必需'))
    after = {str(p): sha(p) for p in (src_run / 'MODEL.xml', muscle_matched)}
    if before != after:
        raise SystemExit('frozen inputs changed')
    print(json.dumps(dict(ratio_max=float((seen / muscle_lim).max()),
                          ratio_median=float(np.median(seen / muscle_lim)),
                          joints_over=int((seen > muscle_lim + 1e-9).sum()),
                          smoke=smoke), ensure_ascii=False))


if __name__ == '__main__':
    main()
