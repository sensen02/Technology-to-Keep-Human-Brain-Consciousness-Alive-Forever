"""Which single joint actually moves the right hind pad forward while it is airborne?

At the raised pose the horizontal Jacobian says Coxa_roll has +0.691 dx/dq with 2.57 rad of room,
yet every swing command so far produced exactly zero airborne advance. This drives one joint at a
time at full authority from the same raised pose and measures what the pad really does, so the
Jacobian claim is checked against the simulator instead of trusted.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

LEG = 'RH'
CLEAR = 0.02


def raise_to(core, leg, clear=CLEAR, max_ticks=220):
    for k in range(max_ticks):
        if float(core.measure()['padz'][leg]) > clear:
            return k
        core.tick(-T.descend_direction(core, leg))
    return None


def drive_one(joint, ticks=120, amp=1.0):
    core = rc.Core(); core.reset()
    leg = rc.LEG_ORDER.index(LEG)
    k_raise = raise_to(core, leg)
    idx = T.leg_joint_action_index(core, leg)
    name_of = {i: core.plant.joint(int(core.actuator.joint_ids[i])).name.replace('joint_' + LEG, '')
               for i in idx}
    x0 = float(core.measure()['foot'][leg][0])
    z0 = float(core.measure()['padz'][leg])
    air = 0; best = 0.; worst = 1e9; zmax = z0
    for k in range(ticks):
        a = np.zeros(rc.NJ)
        a[joint] = amp
        m = core.tick(a)
        padz = float(m['padz'][leg]); dx = float(m['foot'][leg][0]) - x0
        zmax = max(zmax, padz)
        if padz > T.THRESH['place_padz']:
            air += 1; best = max(best, dx); worst = min(worst, dx)
    return dict(joint=name_of[joint], ticks_to_clear=k_raise, airborne_ticks=air,
                airborne_dx_max=best, airborne_dx_min=(worst if worst < 1e9 else None),
                final_dx=float(core.measure()['foot'][leg][0]) - x0,
                final_padz=float(core.measure()['padz'][leg]), max_padz=zmax)


def main():
    probe = rc.Core(); probe.reset()
    leg = rc.LEG_ORDER.index(LEG)
    raise_to(probe, leg)
    Jx = T.foot_horizontal_jacobian(probe, leg)
    rows = []
    for i in T.leg_joint_action_index(probe, leg):
        jx = float(Jx[int(probe.actuator.dof_addresses[i])])
        for amp in ((1.0, -1.0) if abs(jx) > .3 else (1.0,)):
            r = drive_one(i, amp=amp)
            r['jacobian_dx'] = jx
            r['amp'] = amp
            rows.append(r)
            print('%-16s amp %+.1f  Jx %+7.3f  air %3d  airborne dx [%s , %+.4f]  final dx %+.4f padz %+.4f' % (
                r['joint'], amp, jx, r['airborne_ticks'],
                ('%+.4f' % r['airborne_dx_min']) if r['airborne_dx_min'] is not None else '   n/a',
                r['airborne_dx_max'], r['final_dx'], r['final_padz']), flush=True)
    out = dict(label='single joint authority over the right hind pad from a raised pose; measurement only',
               leg=LEG, clearance=CLEAR, rows=rows,
               best_airborne_forward=max((r['airborne_dx_max'] for r in rows), default=0.),
               joints_reaching_the_advance_threshold=sorted(
                   {r['joint'] for r in rows if r['airborne_dx_max'] >= T.THRESH['advance']}))
    (HERE / 'RH_SWING_JOINT_TEST.json').write_text(json.dumps(out, indent=2) + '\n')
    print('best airborne forward from a single joint:', round(out['best_airborne_forward'], 4),
          'threshold', T.THRESH['advance'])
    print('joints that reach it:', out['joints_reaching_the_advance_threshold'])


if __name__ == '__main__':
    main()
