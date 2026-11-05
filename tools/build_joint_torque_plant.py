"""Derive a joint-torque plant from the frozen muscle model, with authority matched to the muscles.

Why this exists: the same-state audit showed the assisted reference trajectory is not reachable by
the 90 muscles -- a bounded joint fit left 77-90% of the demanded torque unmet, and 9-22 of 42 hinge
DOFs per sample fell outside the interval reachable by ANY nonnegative activation vector. So the
muscle plant, not the controller, was the binding constraint. This builds the alternative plant.

Precedent, not invention:
  * flybody (Vaxenburg et al.) states that because faithful anatomical muscle actuation is not yet
    measured, idealized position or torque actuation is "a matter of convenience and can be readily
    reconfigured".
  * Wochner et al. ("Learning with Muscles") compared a muscle-actuated case against an idealized
    torque actuator with the MAX TORQUE MATCHED to the highest torques of the muscle case.
The matching rule is taken from that second paper: each joint's actuator bound is the larger of the
two muscle-reachable magnitudes at the reference posture. Nothing is amplified beyond what the
muscles could already do in their better direction; what is removed is the antagonistic/unilateral
structure and the activation dynamics, and both removals are declared.

What this plant does NOT keep, stated up front:
  * passive muscle force disappears with the muscle actuators (the derived plant has none);
  * activation dynamics disappear (act and its two time constants);
  * moment-arm geometry disappears from the command path: the actuator is an ideal torque source.

env: none. Usage: build_joint_torque_plant.py <frozen RUN dir> <output dir>
"""
import os
os.environ.setdefault('PYTHONHASHSEED', '0')
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
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2,
                                  default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)) + '\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    src_run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    model_path = src_run / 'MODEL.xml'
    inputs = [model_path, src_run / 'MAPPING.json']
    before = {str(p): sha(p) for p in inputs}
    dump(out / 'INPUT_HASHES_BEFORE.json', before)

    src_text = model_path.read_text()
    model = _load_mjcf(model_path).compile()
    data = mujoco.MjData(model)
    if (model.nq, model.nv, model.nu, model.nmesh) != (49, 48, 90, 71):
        raise SystemExit('unexpected source model: %r' % ((model.nq, model.nv, model.nu, model.nmesh),))
    hinges = np.array([j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE])
    va = np.array([model.jnt_dofadr[j] for j in hinges])
    names = [model.joint(int(j)).name for j in hinges]

    # moment arms / force capability at the KEYFRAME posture (the neutral reference, not the walk)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    def tau(act):
        data.act[:] = act
        mujoco.mj_forward(model, data)
        return data.qfrc_actuator.copy()

    f0 = tau(np.zeros(model.nu))
    A = np.empty((model.nv, model.nu))
    for i in range(model.nu):
        e = np.zeros(model.nu)
        e[i] = 1.0
        A[:, i] = tau(e) - f0
    Ah = A[va]
    lo = np.minimum(Ah, 0.0).sum(1)
    hi = np.maximum(Ah, 0.0).sum(1)
    limit = np.maximum(np.abs(lo), np.abs(hi))
    if not np.all(np.isfinite(limit)) or np.any(limit <= 0):
        raise SystemExit('degenerate joint torque limits')

    # which muscles drive which joint: from the MODEL's own moment arms, never invented
    pool_rows = json.loads((src_run / 'MAPPING.json').read_text())['rows']
    assert len(pool_rows) == model.nu, (len(pool_rows), model.nu)
    muscle_to_joint = []
    for i, row in enumerate(pool_rows):
        j = int(np.argmax(np.abs(Ah[:, i])))
        muscle_to_joint.append(dict(actuator=int(i), actuator_name=row['actuator'], leg=row.get('leg'),
                                    joint=names[j], joint_index=j, sign=int(np.sign(Ah[j, i])),
                                    moment=float(Ah[j, i]), pool_size=len(row['local_indices']),
                                    local_indices=list(row['local_indices'])))
    by_joint = {}
    for m in muscle_to_joint:
        by_joint.setdefault(m['joint'], []).append(m)

    block = re.search(r'<actuator>.*?</actuator>', src_text, re.S)
    if block is None:
        raise SystemExit('no <actuator> block in source model')
    lines = ['<actuator>']
    for j, nm in enumerate(names):
        lines.append('   <motor name="jt_%s" joint="%s" gear="1" ctrlrange="%s %s" />'
                     % (nm, nm, format(-limit[j], '.17g'), format(limit[j], '.17g')))
    lines.append('</actuator>')
    new_text = src_text[:block.start()] + '\n'.join(lines) + src_text[block.end():]
    (out / 'MODEL_JOINT_TORQUE.xml').write_text(new_text)

    # verify the derived model: SAME body, different command path
    m2 = _load_mjcf(out / 'MODEL_JOINT_TORQUE.xml').compile()
    d2 = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d2, 0)
    mujoco.mj_forward(m2, d2)
    checks = dict(
        nq=m2.nq, nv=m2.nv, nu=m2.nu, nbody=m2.nbody, ngeom=m2.ngeom, nmesh=m2.nmesh,
        njnt=m2.njnt, nq_ok=m2.nq == model.nq, nv_ok=m2.nv == model.nv,
        body_count_ok=m2.nbody == model.nbody, geom_count_ok=m2.ngeom == model.ngeom,
        mesh_count_ok=m2.nmesh == model.nmesh,
        total_mass_equal=bool(abs(m2.body_mass.sum() - model.body_mass.sum()) == 0.0),
        timestep_equal=bool(m2.opt.timestep == model.opt.timestep),
        gravity_equal=bool(np.array_equal(m2.opt.gravity, model.opt.gravity)),
        keyframe_qpos_equal=bool(np.array_equal(m2.key_qpos[0], model.key_qpos[0])),
        actuator_types=sorted({mujoco.mjtTrn(int(t)).name for t in m2.actuator_trntype}),
        each_hinge_has_one_actuator=bool(
            len({int(m2.actuator_trnid[a, 0]) for a in range(m2.nu)}) == 42),
        actuator_names_match_joints=bool(
            all(m2.actuator(a).name == 'jt_' + m2.joint(int(m2.actuator_trnid[a, 0])).name
                for a in range(m2.nu) for _ in [0])),
        ctrlrange_matches_limit=bool(np.allclose(
            np.array([m2.actuator_ctrlrange[a] for a in range(m2.nu)]),
            np.stack([-limit, limit], 1), rtol=0, atol=0)))
    if not all(v for k, v in checks.items() if k.endswith('_ok') or k.endswith('_equal') or k.startswith('each') or k.startswith('actuator_names') or k.startswith('ctrlrange')):
        raise SystemExit('derived model failed verification: %r' % checks)

    # smoke: hold the keyframe with zero command; the plant must stay finite and must not self-drive
    for _ in range(2000):
        d2.ctrl[:] = 0.0
        mujoco.mj_step(m2, d2)
    smoke = dict(steps=2000, finite=bool(np.isfinite(d2.qpos).all() and np.isfinite(d2.qvel).all()),
                 com_z_after=float(d2.subtree_com[1, 2]), qpos_moved=float(np.abs(d2.qpos - m2.key_qpos[0]).max()))

    dump(out / 'JOINT_PLANT.json', dict(
        source_model=str(model_path), source_model_sha256=before[str(model_path)],
        authority_rule='limit_j = max(|sum of negative muscle moments|, |sum of positive muscle moments|) '
                       'at the keyframe posture: the larger of the two muscle-reachable magnitudes, made bilateral',
        precedent=dict(
            flybody='because faithful anatomical muscle actuation is not measured, idealized position or torque '
                    'actuation is a matter of convenience and can be readily reconfigured (Vaxenburg et al.)',
            learning_with_muscles='idealized torque actuator with maximum torque matched to the highest torques '
                                  'of the muscle-actuated case (Wochner et al.)'),
        not_kept=['passive muscle force', 'activation dynamics (act and its two time constants)',
                  'moment-arm geometry in the command path'],
        joints=[dict(index=j, joint=names[j], limit_torque=float(limit[j]),
                     muscle_interval_lo=float(lo[j]), muscle_interval_hi=float(hi[j]),
                     n_muscles=len(by_joint.get(names[j], [])),
                     muscles=[dict(name=m['actuator_name'], sign=m['sign'], moment=m['moment'])
                              for m in by_joint.get(names[j], [])])
                for j in range(42)],
        verification=checks, smoke=smoke, units='model native; absolute forces and SI NOT_ESTABLISHED',
        importance='核心必需'))
    dump(out / 'POOL_TO_JOINT.json', dict(
        note='derived from the model\'s own actuator moment arms at the keyframe; sign is the sign of the '
             'dominant moment, dominant joint is the largest |moment|. Nothing here is invented anatomy.',
        muscle_to_joint=muscle_to_joint))
    after = {str(p): sha(p) for p in inputs}
    dump(out / 'INPUT_HASHES_AFTER.json', after)
    if before != after:
        raise SystemExit('frozen inputs changed')
    print(json.dumps(dict(limits_min=float(limit.min()), limits_max=float(limit.max()),
                          smoke=smoke, outputs=sorted(p.name for p in out.iterdir())), ensure_ascii=False))


if __name__ == '__main__':
    main()
