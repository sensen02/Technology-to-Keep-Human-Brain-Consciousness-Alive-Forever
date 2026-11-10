"""Both phases from the vertical Jacobian at the current pose: no hand-made directions.

The hand-made candidates in LIFT_DEMO.json were the weak link for the rear legs: the fallback
direction for RM raises the pad by exactly 0.0000 (it drives only four joints), and LH's raises
it by 0.0124, below the 0.015 acceptance bar. Computed at the current pose the same idea gives
RM +0.2089 and LH +0.0563. This teacher is therefore fully derived from measured state.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

LIFT_TICKS = 90


def teacher_action(core, leg, k):
    return (-T.descend_direction(core, leg)) if k < LIFT_TICKS else T.descend_direction(core, leg)


def run(leg, ticks=T.T_EP, lift_ticks=LIFT_TICKS, trace=False):
    core = rc.Core()
    core.reset()
    rest = float(core.measure()['padz'][leg])
    start_x = float(core.measure()['foot'][leg][0])
    maxz = rest; minload = 1e9; sup_min = 1e9; com_min = 1e9; nf = 0.; best_adv = 0.
    last = None; traj = []
    for k in range(ticks):
        a = (-T.descend_direction(core, leg)) if k < lift_ticks else T.descend_direction(core, leg)
        m = core.tick(a)
        padz = float(m['padz'][leg])
        maxz = max(maxz, padz); minload = min(minload, float(m['loads'][leg]))
        sup_min = min(sup_min, float(m['loads'].sum() / rc.WEIGHT))
        com_min = min(com_min, float(m['com'])); nf = max(nf, float(m['nonfoot']))
        best_adv = max(best_adv, float(m['foot'][leg][0]) - start_x)
        last = m
        if trace and (k % 30 == 0 or k == ticks - 1):
            traj.append([k, round(padz, 4), round(float(m['loads'][leg] / rc.WEIGHT), 3)])
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
    out = dict(label='teacher self-check, both phases from the vertical Jacobian; no learning',
               lift_ticks=LIFT_TICKS, per_leg={})
    for leg in range(rc.NL):
        r = run(leg, trace=True)
        out['per_leg'][rc.LEG_ORDER[leg]] = r
        print('%-2s rise %.3f -> final padz %+.4f load %.3f sup %.3f com %.3f nf %.2f  %s  %s' % (
            rc.LEG_ORDER[leg], r['max_padz'], r['final_padz'], r['final_load'],
            r['support_min'], r['com_min'], r['nonfoot_max'],
            ''.join(k[0].upper() for k, v in r['criteria'].items() if v),
            r['traj']), flush=True)
    out['n_passed'] = sum(r['passed'] for r in out['per_leg'].values())
    out['passed_legs'] = [L for L, r in out['per_leg'].items() if r['passed']]
    (HERE / 'DEMO_JAC_TEACHER.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'n_passed': out['n_passed'], 'passed_legs': out['passed_legs']}))


if __name__ == '__main__':
    main()
