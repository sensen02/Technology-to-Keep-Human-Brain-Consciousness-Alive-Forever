"""Is the derived joint-torque plant equivalent to the muscle plant under the demonstrated controller?

Two replays against the archived muscle run, both starting from the same keyframe and using the same
recorded gait spline, settle target and gains:

  (a) TORQUE-ONLY   the warmup's PD law is applied directly as joint torque. This is what a joint
                    torque controller would actually command, so it answers: does the body still walk
                    when its only actuation is ideal joint torque?
Only this one replay is run. The archived total torque (assist plus the muscle torque of the same step)
is NOT transcribed, because the muscle torque would have to be recomputed forward for every archived
sample and zero-order-held, which introduces its own error. The comparison below is therefore against a
muscle run that also received its own muscle torque; a difference is expected and is not by itself a
defect of the derived plant.

Nothing here is a walking result: the motion comes from a transcribed algorithm, not from neurons.

env: none. Usage: verify_joint_plant_replay.py <frozen RUN dir> <plant dir> <out dir>
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R)]
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2,
                                  default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)) + '\n')


def main():
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    run, plant_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=False)

    body = dict(np.load(run / 'BODY.npz'))
    gait_npz = np.load(run / 'GAIT.npz')
    gains = json.loads((run / 'GAINS.json').read_text())
    settle = np.asarray(json.loads((run / 'SETTLE_TARGET.json').read_text())['target'], float)
    kp, kd, clip = float(gains['kp']), np.asarray(gains['kd'], float), float(gains['clip'])
    gait = CubicSpline(gait_npz['phase'], gait_npz['qpos_hinges'], axis=0, bc_type='periodic')

    model = _load_mjcf(plant_dir / 'MODEL_JOINT_TORQUE.xml').compile()
    data = mujoco.MjData(model)
    hinges = np.array([j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE])
    qa = np.array([model.jnt_qposadr[j] for j in hinges])
    va = np.array([model.jnt_dofadr[j] for j in hinges])
    thorax = model.body('Thorax').id
    q0 = model.key_qpos[0].copy()
    limit = np.array([model.actuator_ctrlrange[a, 1] for a in range(model.nu)])
    dt = float(model.opt.timestep)
    duration = 1.3
    n = int(round(duration / dt))
    stride = int(round(0.0005 / dt))

    def trajectory():
        d = data
        mujoco.mj_resetData(model, d)
        d.qpos[:] = q0
        mujoco.mj_forward(model, d)
        rows = []
        for step in range(n):
            t = float(d.time)
            ramp = np.clip((t - .15) / .10, 0, 1)
            phi = max(0, t - .15) * 2 * np.pi * 5
            des = (1 - ramp) * q0[qa] + ramp * gait(phi % (2 * np.pi))
            vel = (ramp * gait(phi % (2 * np.pi), 1) * 2 * np.pi * 5
                   + (gait(phi % (2 * np.pi)) - q0[qa]) * (10. if .15 < t < .25 else 0.))
            torque = kp * (des - d.qpos[qa]) + kd * (vel - d.qvel[va]) + d.qfrc_bias[va] - d.qfrc_passive[va]
            torque = np.clip(torque, -clip, clip)
            d.qfrc_applied[:] = 0.0
            d.xfrc_applied[:] = 0.0
            d.ctrl[:] = np.clip(torque, -limit, limit)
            mujoco.mj_step(model, d)
            if (step + 1) % stride == 0:
                rows.append(dict(time=float(d.time), qpos=d.qpos.copy(), com=float(d.subtree_com[thorax, 2])))
        return rows

    t_start = time.monotonic()
    torque_only = trajectory()
    wall = time.monotonic() - t_start

    # compare against the archived muscle run on the prewarm span
    n_cmp = min(len(torque_only), body['time'].size)
    q_ref = body['qpos'][:n_cmp]
    q_new = np.array([r['qpos'] for r in torque_only[:n_cmp]])
    com_ref = body['com'][:n_cmp, 2]
    com_new = np.array([r['com'] for r in torque_only[:n_cmp]])
    err = np.abs(q_new - q_ref).max(axis=1)
    res = dict(
        question='does the muscle-free joint-torque plant reproduce the warmup when driven by the same PD law?',
        mode='torque_only (the PD law applied as ideal joint torque; no muscle contribution, no assist channel)',
        steps=int(n), dt_s=dt, wall_s=wall, samples_compared=int(n_cmp),
        actuator_limit_min=float(limit.min()), actuator_limit_max=float(limit.max()),
        qpos_max_abs_error=dict(first_50ms=float(err[:100].max()), first_200ms=float(err[:400].max()),
                                end_of_warmup=float(err[-1]), overall_max=float(err.max())),
        com_z=dict(archived_start=float(com_ref[0]), new_start=float(com_new[0]),
                   archived_end=float(com_ref[-1]), new_end=float(com_new[-1]),
                   archived_min=float(com_ref.min()), new_min=float(com_new.min())),
        foot_contact_note='not compared: the derived run records no contact here, only qpos and thorax COM',
        caveat='the muscle run also received its own muscle torque on top of the PD assist, which this replay '
               'does not supply; a difference is therefore expected and is not by itself a plant defect',
        units='model native; absolute forces and SI NOT_ESTABLISHED', importance='核心必需')
    dump(out / 'PLANT_REPLAY.json', res)
    print(json.dumps({k: res[k] for k in ['qpos_max_abs_error', 'com_z', 'wall_s']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
