"""Bounded world-anchor tripod gait. Declared algorithm-assisted warm-up, NOT neural walking.

The first world-anchor version accumulated unbounded tracking error: when the trunk drifted, the
fixed anchors became unreachable, the joint rates saturated, the front legs were pulled off the
ground and the trunk was dragged backwards (DIAG_WORLD.json). The anchors are therefore bounded: if a
stance pad slides further than `max_err` from its anchor, the anchor is re-grounded at the measured
pad position. That turns a runaway into a friction-limited gait and keeps the tracking error bounded
by construction. Nothing writes qpos/qvel after reset, no root translation, no external force.
"""
from pathlib import Path
import json, sys, itertools
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gait_teacher import Core, T, LEGS, NL, NJ, MAX_RATE, W, rel_and_world, summarise, GATES
from gait_world import phase_of, TRIPOD_A


def run(period=120, stride=0.02, lift=0.08, settle=300, ticks=2000, tau=0.05,
        max_err=0.010, kz=4.0, padz_gain=0.6, record=True, debug=False):
    core = Core(); core.reset()
    rest = [np.asarray(core.measure()['foot'][l], float) for l in range(NL)]
    for k in range(settle):
        a = np.zeros(NJ)
        for leg in range(NL):
            rel, world = rel_and_world(core, leg)
            ids = T.leg_joint_action_index(core, leg); ds = core.a_dof[ids]
            cur = np.asarray(core.measure()['foot'][leg], float)
            des = (rest[leg] - cur) / tau
            rv = core.d.qvel.copy(); rv[core.a_dof] = 0.
            dq = rel[:, ds].T @ np.linalg.solve(rel[:, ds] @ rel[:, ds].T + 1e-2 * np.eye(3),
                                                des - rel @ rv)
            a[ids] = dq / MAX_RATE / (1. + np.abs(dq) / MAX_RATE)
        core.tick(np.clip(a, -1., 1.))
    pad = lambda leg: np.mean([core.d.geom_xpos[g] for g in core.pads[LEGS[leg]]], axis=0)
    anchors = [pad(l).copy() for l in range(NL)]
    reground = [0] * NL
    rec = []
    for k in range(ticks):
        a = np.zeros(NJ)
        for leg in range(NL):
            rel, world = rel_and_world(core, leg)
            ids = T.leg_joint_action_index(core, leg); ds = core.a_dof[ids]
            ph = phase_of(leg, k, period)
            pos = pad(leg)
            if ph < 0.5:
                if float(np.linalg.norm((anchors[leg] - pos)[:2])) > max_err:
                    anchors[leg] = pos.copy(); reground[leg] += 1
                tgt = anchors[leg]; extra = 0.
            else:
                u = (ph - 0.5) / 0.5
                tgt = anchors[leg] + np.array([stride * u, 0., 0.])
                extra = lift * np.sin(np.pi * u)
                if u >= 0.999:
                    anchors[leg] = tgt.copy()
            des = (tgt - pos) / tau
            des[2] += kz * extra - padz_gain * float(pos[2]) * (1. if ph < 0.5 else 0.)
            rv = core.d.qvel.copy(); rv[core.a_dof] = 0.
            A = world[:, ds]
            dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(3), des - world @ rv)
            a[ids] = dq / MAX_RATE / (1. + np.abs(dq) / MAX_RATE)
        m = core.tick(np.clip(a, -1., 1.))
        if record:
            rec.append((float(core.d.time), np.array(core.d.subtree_com[core.th], float),
                        m['loads'].copy(), float(m['nonfoot']), np.array(m['speed'], float)))
        if debug and k % 200 == 0:
            print('k %4d com %s sup %.3f nf %5.2f reground %s' % (
                k, np.round(core.d.subtree_com[core.th], 3), float(m['loads'].sum() / W),
                float(m['nonfoot']), reground), flush=True)
    out = summarise(rec, (settle + ticks) * 80)
    out['reground_events'] = reground
    return out, core


def main():
    rows = []
    for period, stride, me in itertools.product([120, 200], [0.015, 0.03], [0.006, 0.012, 0.025]):
        s, _ = run(period=period, stride=stride, max_err=me, ticks=2000)
        rows.append(dict(period=period, stride=stride, max_err=me,
                         **{k: v for k, v in s.items() if k != 'per_leg_slip_ratio'}))
        print('p%3d s%.3f me%.3f | fwdsigned %+.4f lat %.4f sup %.3f supmin %.3f nf %5.2f com %.3f slip %5.2f %s' % (
            period, stride, me, s['forward_signed_native_per_s'], s['lateral_speed_native_per_s'],
            s['foot_support_mean'], s['foot_support_min'], s['nonfoot_force_max_W'],
            s['com_height_min'], s['slip_ratio_max'],
            ''.join(k[0].upper() for k, v in s['gates'].items() if v)), flush=True)
        (HERE / 'GAIT_BOUNDED_ANCHOR.json').write_text(json.dumps(
            dict(label='bounded world-anchor tripod gait; algorithm-assisted warm-up',
                 gates=GATES, rows=rows), indent=2) + '\n')
    print('passing all gates:', [r for r in rows if r['all_gates']])


if __name__ == '__main__':
    main()
