"""World-anchored tripod gait. Declared algorithm-assisted warm-up, NOT neural walking.

The body-frame ramp used earlier fought the physics: the trunk was locked to the feet, so the legs
skidded and the apparent "forward speed" was really lateral drift. The propulsion mechanism used here
is the one a real walker uses:

  stance : the foot holds a FIXED WORLD anchor, so the leg cannot drag it across the ground
  swing  : the foot lifts and is placed `stride` ahead of that anchor in the world

When a foot touches down ahead of the body the leg is stretched forward, and the body has to travel
forward over it; that is what produces displacement, with the anchors instead of a scripted ramp
setting the pace. A settle window runs first so the measured window is not dominated by the initial
pose relaxing. Nothing writes qpos/qvel after reset, nothing translates the root, no external force.
"""
from pathlib import Path
import json, os, sys, itertools
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gait_teacher as GT
from gait_teacher import Core, rc, T, GATES, LEGS, NL, NJ, PDT, MAX_RATE, W, rel_and_world, summarise

TRIPOD_A = ('LF', 'RM', 'LH')
TRIPOD_B = ('RF', 'LM', 'RH')


def phase_of(leg, k, period):
    offs = 0 if LEGS[leg] in TRIPOD_A else period // 2
    return ((k + offs) % period) / period


def run(period=120, stride=0.02, lift=0.08, settle=300, ticks=2400,
        tau=0.05, kz=4.0, heading=np.array([1.0, 0.0]), record=True, debug=False,
        lateral_gain=0.0):
    core = Core(); core.reset()
    rest_body = [np.asarray(core.measure()['foot'][l], float) for l in range(NL)]
    # settle: hold the measured rest configuration so the window is about walking, not about the
    # initial pose relaxing. This settling segment is reported separately and excluded from the gates.
    for k in range(settle):
        a = np.zeros(NJ)
        for leg in range(NL):
            rel, world = rel_and_world(core, leg)
            ids = T.leg_joint_action_index(core, leg); ds = core.a_dof[ids]
            cur = np.asarray(core.measure()['foot'][leg], float)
            desired = (rest_body[leg] - cur) / tau    # hold the measured reset foot position
            A = rel[:, ds]; full = rel
            rv = core.d.qvel.copy(); rv[core.a_dof] = 0.
            dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(3), desired - full @ rv)
            a[ids] = dq / MAX_RATE / (1. + np.abs(dq) / MAX_RATE)
        core.tick(np.clip(a, -1., 1.))
    anchors = [np.mean([core.d.geom_xpos[g] for g in core.pads[L]], axis=0) for L in LEGS]
    com0 = np.array(core.d.subtree_com[core.th], float)
    rec = []
    lateral_shift = 0.0
    for k in range(ticks):
        a = np.zeros(NJ)
        # Declared stance-anchor correction: if the trunk drifts sideways, move the foothold pattern
        # back under it instead of letting the whole gait slide sideways.
        if lateral_gain:
            drift_y = float(np.asarray(core.d.subtree_com[core.th], float)[1] - com0[1])
            lateral_shift = -lateral_gain * drift_y
        for leg in range(NL):
            rel, world = rel_and_world(core, leg)
            ids = T.leg_joint_action_index(core, leg); ds = core.a_dof[ids]
            ph = phase_of(leg, k, period)
            if ph < 0.5:                                    # stance: hold the world anchor
                tgt = anchors[leg] + np.array([0., lateral_shift, 0.]); extra = 0.
            else:                                           # swing: place ahead, with lift
                u = (ph - 0.5) / 0.5
                tgt = anchors[leg] + np.r_[heading * stride * u, 0.] + np.array([0., lateral_shift, 0.])
                extra = lift * np.sin(np.pi * u)
            # desired world velocity of the pad centre
            pos = np.mean([core.d.geom_xpos[g] for g in core.pads[LEGS[leg]]], axis=0)
            desired = (tgt - pos) / tau
            desired[2] += kz * extra
            A = world[:, ds]; full = world
            rv = core.d.qvel.copy(); rv[core.a_dof] = 0.
            dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(3), desired - full @ rv)
            a[ids] = dq / MAX_RATE / (1. + np.abs(dq) / MAX_RATE)
            # land the anchor on the new foothold at the end of swing
            if ph >= 0.5 and u >= 0.999:
                anchors[leg] = tgt.copy()
        m = core.tick(np.clip(a, -1., 1.))
        if record:
            rec.append((float(core.d.time), np.array(core.d.subtree_com[core.th], float),
                        m['loads'].copy(), float(m['nonfoot']), np.array(m['speed'], float)))
        if debug and k % 50 == 0:
            print('k %4d com %s sup %.3f' % (k, np.round(core.d.subtree_com[core.th], 3),
                                             float(m['loads'].sum() / W)), flush=True)
    return summarise(rec, (settle + ticks) * 80), core


def main():
    rows = []
    for period, stride, lift in itertools.product([120, 200], [0.02, 0.04], [0.08]):
        s, _ = run(period=period, stride=stride, lift=lift, ticks=2000)
        rows.append(dict(period=period, stride=stride, lift=lift,
                         **{k: v for k, v in s.items() if k != 'per_leg_slip_ratio'}))
        print('p%3d s%.3f l%.2f | fwdsigned %+.4f lat %.4f sup %.3f supmin %.3f nf %5.2f com %.3f slip %5.2f %s' % (
            period, stride, lift, s['forward_signed_native_per_s'], s['lateral_speed_native_per_s'],
            s['foot_support_mean'], s['foot_support_min'], s['nonfoot_force_max_W'],
            s['com_height_min'], s['slip_ratio_max'],
            ''.join(k[0].upper() for k, v in s['gates'].items() if v)), flush=True)
        (HERE / 'GAIT_WORLD_ANCHOR.json').write_text(
            json.dumps(dict(label='world-anchored tripod gait after settle; algorithm-assisted warm-up',
                            gates=GATES, rows=rows), indent=2) + '\n')
    print('passing all gates:', [r for r in rows if r['all_gates']])


if __name__ == '__main__':
    main()
