"""A three phase demonstration: raise, swing forward, put down.

The two phase teacher could only raise and lower, so the vertical Jacobian moved a rear leg's
foot straight up and the advance criterion stayed at exactly 0.0000 (LH and RH). A step needs a
middle phase that moves the foot forward while it is in the air, and a little of the raise
command kept in to hold height while it swings.
"""
from pathlib import Path
import json, sys
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def foot_horizontal_jacobian(core, leg):
    rows = []
    for g in core.pads[rc.LEG_ORDER[leg]]:
        point = np.asarray(core.d.geom_xpos[g])
        jacp = np.zeros((3, core.plant.nv))
        mj.mj_jac(core.plant, core.d, jacp, None, point, int(core.plant.geom_bodyid[g]))
        rows.append(jacp[0].copy())
    return np.mean(rows, axis=0)


def forward_direction(core, leg, scale=1.0):
    """Joint rates that move this leg's foot forward (+x), computed at the current pose."""
    J = foot_horizontal_jacobian(core, leg)
    a = np.zeros(rc.NJ)
    for i in T.leg_joint_action_index(core, leg):
        j = float(J[int(core.actuator.dof_addresses[i])])
        if abs(j) > 1e-9:
            a[i] = np.sign(j) * scale
    return a


def run(leg, lift_ticks=90, swing_ticks=90, hold=0.5, ticks=T.T_EP, trace=False):
    core = rc.Core()
    core.reset()
    rest = float(core.measure()['padz'][leg])
    start_x = float(core.measure()['foot'][leg][0])
    maxz = rest; minload = 1e9; sup_min = 1e9; com_min = 1e9; nf = 0.; best_adv = 0.
    last = None; traj = []
    for k in range(ticks):
        raise_a = -T.descend_direction(core, leg)
        if k < lift_ticks:
            a = raise_a
        elif k < lift_ticks + swing_ticks:
            a = np.clip(forward_direction(core, leg) + hold * raise_a, -1., 1.)
        else:
            a = T.descend_direction(core, leg)
        m = core.tick(a)
        padz = float(m['padz'][leg])
        maxz = max(maxz, padz); minload = min(minload, float(m['loads'][leg]))
        sup_min = min(sup_min, float(m['loads'].sum() / rc.WEIGHT))
        com_min = min(com_min, float(m['com'])); nf = max(nf, float(m['nonfoot']))
        best_adv = max(best_adv, float(m['foot'][leg][0]) - start_x)
        last = m
        if trace and (k % 30 == 0 or k == ticks - 1):
            traj.append([k, round(padz, 4), round(float(m['loads'][leg] / rc.WEIGHT), 3),
                         round(float(m['foot'][leg][0]) - start_x, 4)])
    final_load = float(last['loads'][leg] / rc.WEIGHT); final_padz = float(last['padz'][leg])
    crit = dict(unload=bool(minload / rc.WEIGHT <= T.THRESH['unload']),
                lift=bool(maxz >= T.THRESH['lift']),
                advance=bool(best_adv >= T.THRESH['advance']),
                place=bool(final_load >= T.THRESH['place_load']
                           and final_padz <= T.THRESH['place_padz']),
                safe=bool(sup_min >= T.THRESH['support'] and nf <= T.THRESH['nonfoot']
                          and com_min >= T.THRESH['com']))
    return dict(max_padz=maxz, final_padz=final_padz, final_load=final_load,
                min_load=minload / rc.WEIGHT, best_advance=best_adv, support_min=sup_min,
                com_min=com_min, nonfoot_max=nf, criteria=crit, passed=all(crit.values()),
                traj=traj)


def main():
    grid = [(60, 90, .5), (90, 90, .5), (90, 120, .5), (120, 90, .5), (90, 90, .8),
            (90, 120, .8), (120, 120, .8), (150, 120, .8)]
    out = dict(label='three phase teacher self-check (raise, swing forward, place); no learning',
               grid=[{'lift_ticks': a, 'swing_ticks': b, 'hold': c} for a, b, c in grid],
               per_config={})
    best = None
    for lt, st, hd in grid:
        res = {}
        for leg in range(rc.NL):
            r = run(leg, lt, st, hd, trace=True)
            res[rc.LEG_ORDER[leg]] = r
        n = sum(r['passed'] for r in res.values())
        key = 'lift=%d,swing=%d,hold=%.1f' % (lt, st, hd)
        out['per_config'][key] = dict(n_passed=n,
                                      passed_legs=[L for L, r in res.items() if r['passed']],
                                      per_leg=res)
        print('%s -> %d/6 %s' % (key, n, {L: ''.join(k[0].upper() for k, v in r['criteria'].items() if v)
                                          for L, r in res.items()}), flush=True)
        if best is None or n > best[1]:
            best = (key, n)
    out['best'] = best[0]
    out['best_n'] = best[1]
    (HERE / 'DEMO_STEP_TEACHER.json').write_text(json.dumps(out, indent=2) + '\n')
    print('best:', best)


if __name__ == '__main__':
    main()
