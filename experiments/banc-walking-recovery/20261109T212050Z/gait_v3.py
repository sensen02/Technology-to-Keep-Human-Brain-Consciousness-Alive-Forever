"""Tripod gait with phase-dependent task dimension. Declared algorithm-assisted warm-up.

Structural fix: every earlier attempt constrained all three axes of all six pads at once, which pins
the trunk completely and makes the stride direction irrelevant (measured: stride +0.03 and -0.03 gave
identical backward drift). Constraints now depend on the phase:

  stance : fore-aft and lateral only. The vertical direction is left to the ground contact, so the
           trunk is free to settle and the leg can push along the ground instead of being pinned.
  swing  : all three axes, so the foot follows its lift profile and lands ahead.

Resolved-rate control uses the thorax-relative Jacobian verified in frame_audit.py. Nothing writes
qpos/qvel after reset, no root translation, no external force, no pose locking.
"""
from pathlib import Path
import json, sys, itertools
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gait_teacher import Core, T, LEGS, NL, NJ, MAX_RATE, W, rel_and_world, summarise, GATES
from gait_world import phase_of, TRIPOD_A


def run(period=120, stride=0.03, lift=0.08, settle=300, ticks=2000, tau=0.05,
        z_push=0.0, com_target=None, record=True, debug=False):
    core = Core(); core.reset()
    rest = [np.asarray(core.measure()['foot'][l], float) for l in range(NL)]
    for k in range(settle):
        a = np.zeros(NJ)
        for leg in range(NL):
            rel, world = rel_and_world(core, leg)
            ids = T.leg_joint_action_index(core, leg); ds = core.a_dof[ids]
            cur = np.asarray(core.measure()['foot'][leg], float)
            A = rel[:2, ds]                      # horizontal hold only; vertical rests on the ground
            rv = core.d.qvel.copy(); rv[core.a_dof] = 0.
            dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(2), (rest[leg] - cur)[:2] / tau - rel[:2] @ rv)
            a[ids] = dq / MAX_RATE / (1. + np.abs(dq) / MAX_RATE)
        core.tick(np.clip(a, -1., 1.))
    com0 = float(core.d.subtree_com[core.th][2])
    tgt_z = com_target if com_target else com0 - 0.02
    rec = []
    for k in range(ticks):
        a = np.zeros(NJ)
        com = float(core.d.subtree_com[core.th][2])
        push = 0.0
        if z_push:
            push = float(np.clip(-z_push * (tgt_z - com), -0.5, 0.0))
        for leg in range(NL):
            rel, world = rel_and_world(core, leg)
            ids = T.leg_joint_action_index(core, leg); ds = core.a_dof[ids]
            ph = phase_of(leg, k, period)
            cur = np.asarray(core.measure()['foot'][leg], float)
            rv = core.d.qvel.copy(); rv[core.a_dof] = 0.
            if ph < 0.5:                                     # stance: horizontal only
                u = ph / 0.5
                dx = stride * (0.5 - u)
                des = np.array([(rest[leg][0] + dx - cur[0]) / tau,
                                (rest[leg][1] - cur[1]) / tau])
                A = rel[:2, ds]
                dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(2), des - rel[:2] @ rv)
                if push:
                    # a downward stance command adds force along the ground without pinning the trunk
                    desz = push
                    az = world[2, ds:ds + 1].reshape(1, -1)
                    dqz = az.T @ np.linalg.solve(az @ az.T + 1e-2, np.array([desz - world[2] @ rv]))
                    dq = dq + dqz
            else:                                            # swing: all three axes
                u = (ph - 0.5) / 0.5
                dx = stride * (-0.5 + u)
                dz = lift * np.sin(np.pi * u)
                des = np.array([(rest[leg][0] + dx - cur[0]) / tau,
                                (rest[leg][1] - cur[1]) / tau,
                                (rest[leg][2] + dz - cur[2]) / tau])
                A = rel[:, ds]
                dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(3), des - rel @ rv)
            a[ids] = dq / MAX_RATE / (1. + np.abs(dq) / MAX_RATE)
        m = core.tick(np.clip(a, -1., 1.))
        if record:
            rec.append((float(core.d.time), np.array(core.d.subtree_com[core.th], float),
                        m['loads'].copy(), float(m['nonfoot']), np.array(m['speed'], float)))
        if debug and k % 200 == 0:
            print('k %4d com %s sup %.3f nf %5.2f' % (
                k, np.round(core.d.subtree_com[core.th], 3), float(m['loads'].sum() / W),
                float(m['nonfoot'])), flush=True)
    return summarise(rec, (settle + ticks) * 80), core


def main():
    rows = []
    for stride, period in itertools.product([0.03, -0.03, 0.06], [60, 120]):
        s, _ = run(period=period, stride=stride, lift=0.08, ticks=1600)
        rows.append(dict(period=period, stride=stride,
                         **{k: v for k, v in s.items() if k != 'per_leg_slip_ratio'}))
        print('p%3d s%+.3f | fwdsigned %+.4f lat %.4f sup %.3f supmin %.3f nf %5.2f com %.3f slip %5.2f %s' % (
            period, stride, s['forward_signed_native_per_s'], s['lateral_speed_native_per_s'],
            s['foot_support_mean'], s['foot_support_min'], s['nonfoot_force_max_W'],
            s['com_height_min'], s['slip_ratio_max'],
            ''.join(k[0].upper() for k, v in s['gates'].items() if v)), flush=True)
        (HERE / 'GAIT_PHASE_DIM.json').write_text(json.dumps(
            dict(label='phase-dependent task dimension tripod gait; algorithm-assisted warm-up',
                 gates=GATES, rows=rows), indent=2) + '\n')
    print('passing all gates:', [r for r in rows if r['all_gates']])


if __name__ == '__main__':
    main()
