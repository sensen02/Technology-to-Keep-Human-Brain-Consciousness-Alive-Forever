"""Stance force amplification + trunk velocity feedback. Declared algorithm warm-up, NOT neural walking.

Two additions over gait_v3, both in the declared algorithm layer:

1. Trunk velocity feedback: the stance target additionally asks the foot to retreat at the measured
   trunk speed, so the leg keeps pushing against a moving reference instead of idling once the foot
   has reached its body-frame target.
2. Force amplification: the joint-rate request is scaled before the rate limit, so a stance leg that
   is not yet sliding is allowed to build joint error and therefore PD torque. This is what turns a
   kinematic follow into actual push; it stays inside the unchanged rate and torque limits.

Nothing writes qpos/qvel after reset, no root translation, no external force, no pose locking.
"""
from pathlib import Path
import json, sys, itertools
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gait_teacher import Core, T, LEGS, NL, NJ, MAX_RATE, W, rel_and_world, summarise, GATES
from gait_world import phase_of


def run(period=60, stride=0.03, lift=0.08, settle=300, ticks=1600, tau=0.02,
        k_vel=0.0, gain=1.0, record=True):
    core = Core(); core.reset()
    rest = [np.asarray(core.measure()['foot'][l], float) for l in range(NL)]
    R0 = core.d.xmat[core.th].reshape(3, 3).copy()
    for k in range(settle):
        a = np.zeros(NJ)
        for leg in range(NL):
            rel, world = rel_and_world(core, leg)
            ids = T.leg_joint_action_index(core, leg); ds = core.a_dof[ids]
            cur = np.asarray(core.measure()['foot'][leg], float)
            rv = core.d.qvel.copy(); rv[core.a_dof] = 0.
            A = rel[:2, ds]
            dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(2),
                                       (rest[leg] - cur)[:2] / tau - rel[:2] @ rv)
            a[ids] = dq / MAX_RATE / (1. + np.abs(dq) / MAX_RATE)
        core.tick(np.clip(a, -1., 1.))
    rec = []
    for k in range(ticks):
        a = np.zeros(NJ)
        v_trunk = R0.T @ core.d.qvel[:3]          # trunk velocity in the initial heading frame
        for leg in range(NL):
            rel, world = rel_and_world(core, leg)
            ids = T.leg_joint_action_index(core, leg); ds = core.a_dof[ids]
            ph = phase_of(leg, k, period)
            cur = np.asarray(core.measure()['foot'][leg], float)
            rv = core.d.qvel.copy(); rv[core.a_dof] = 0.
            if ph < 0.5:
                u = ph / 0.5
                dx = stride * (0.5 - u)
                # retreat at the measured trunk speed as well, so the leg keeps doing work
                des = np.array([(rest[leg][0] + dx - cur[0]) / tau - k_vel * v_trunk[0],
                                (rest[leg][1] - cur[1]) / tau - k_vel * v_trunk[1]])
                A = rel[:2, ds]
                dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(2), des - rel[:2] @ rv)
            else:
                u = (ph - 0.5) / 0.5
                des = np.array([(rest[leg][0] + stride * (-0.5 + u) - cur[0]) / tau,
                                (rest[leg][1] - cur[1]) / tau,
                                (rest[leg][2] + lift * np.sin(np.pi * u) - cur[2]) / tau])
                A = rel[:, ds]
                dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(3), des - rel @ rv)
            a[ids] = np.clip(gain * dq / MAX_RATE, -1., 1.)
        m = core.tick(a)
        if record:
            rec.append((float(core.d.time), np.array(core.d.subtree_com[core.th], float),
                        m['loads'].copy(), float(m['nonfoot']), np.array(m['speed'], float)))
    return summarise(rec, (settle + ticks) * 80), core


def main():
    rows = []
    for tau, k_vel, gain in itertools.product([0.02, 0.01], [0.0, 1.0, 2.0], [1.0, 2.5]):
        s, _ = run(tau=tau, k_vel=k_vel, gain=gain)
        rows.append(dict(tau=tau, k_vel=k_vel, gain=gain,
                         **{k: v for k, v in s.items() if k != 'per_leg_slip_ratio'}))
        print('tau %.2f kvel %.1f gain %.1f | fwd %+.4f lat %.4f supmin %.3f nf %5.2f com %.3f slip %6.2f %s' % (
            tau, k_vel, gain, s['forward_signed_native_per_s'], s['lateral_speed_native_per_s'],
            s['foot_support_min'], s['nonfoot_force_max_W'], s['com_height_min'], s['slip_ratio_max'],
            ''.join(k[0].upper() for k, v in s['gates'].items() if v)), flush=True)
        (HERE / 'GAIT_PUSH.json').write_text(json.dumps(
            dict(label='stance push amplification and trunk velocity feedback; algorithm warm-up',
                 gates=GATES, rows=rows), indent=2) + '\n')
    print('passing all gates:', [r for r in rows if r['all_gates']])


if __name__ == '__main__':
    main()
