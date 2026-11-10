"""A demonstration teacher with no clock: both phase transitions are measured state.

The sweep found that the timed demonstration (90-tick lift, then press) satisfies all five
criteria on four of six legs, but a timer cannot be part of the deployed policy. This replaces
the timer with state: lift until the pad has cleared a declared height AND the foot has advanced,
then press down along the vertical Jacobian recomputed at the current pose, then hold once the
foot is loaded and back on the ground. A teacher may use measured state; a clock would not be
allowed in the policy.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T
import demo_press_probe as P

HEIGHT_THR, ADV_THR = 0.02, 0.03


def run(leg, height_thr=HEIGHT_THR, adv_thr=ADV_THR, ticks=T.T_EP, trace=False):
    core = rc.Core(); demo = T.demo_directions()[leg]
    core.reset()
    rest = float(core.measure()['padz'][leg])
    start_x = float(core.measure()['foot'][leg][0])
    maxz = rest; minload = 1e9; sup_min = 1e9; com_min = 1e9; nonfoot = 0.; best_adv = 0.
    pressing = False; holding = False; last = None; traj = []
    for k in range(ticks):
        m = core.measure()
        padz = float(m['padz'][leg]); load = float(m['loads'][leg])
        adv = float(m['foot'][leg][0]) - start_x
        if holding:
            a = np.zeros(rc.NJ)
        elif pressing:
            a = P.descend_direction(core, leg)
            if load >= T.SWING_END and padz <= T.THRESH['place_padz']:
                holding = True
        else:
            a = demo['unit']
            if padz >= height_thr and adv >= adv_thr:
                pressing = True
        mm = core.tick(a)
        padz2 = float(mm['padz'][leg])
        maxz = max(maxz, padz2); minload = min(minload, float(mm['loads'][leg]))
        sup_min = min(sup_min, float(mm['loads'].sum() / rc.WEIGHT))
        com_min = min(com_min, float(mm['com']))
        nonfoot = max(nonfoot, float(mm['nonfoot']))
        best_adv = max(best_adv, float(mm['foot'][leg][0]) - start_x)
        last = mm
        if trace and (k % 25 == 0 or k == ticks - 1):
            traj.append([k, round(padz2, 4), round(float(mm['loads'][leg] / rc.WEIGHT), 3)])
    final_load = float(last['loads'][leg] / rc.WEIGHT); final_padz = float(last['padz'][leg])
    crit = dict(unload=bool(minload / rc.WEIGHT <= T.THRESH['unload']),
                lift=bool(maxz >= T.THRESH['lift']),
                advance=bool(best_adv >= T.THRESH['advance']),
                place=bool(final_load >= T.THRESH['place_load'] and final_padz <= T.THRESH['place_padz']),
                safe=bool(sup_min >= T.THRESH['support'] and nonfoot <= T.THRESH['nonfoot']
                          and com_min >= T.THRESH['com']))
    return dict(max_padz=maxz, final_padz=final_padz, final_load=final_load,
                min_load=minload / rc.WEIGHT, best_advance=best_adv, support_min=sup_min,
                com_min=com_min, nonfoot_max=nonfoot, criteria=crit, passed=all(crit.values()),
                traj=traj)


def main():
    out = dict(label='clock-free state teacher, teacher self-check only', height_thr=HEIGHT_THR,
               adv_thr=ADV_THR, per_leg={})
    for leg in range(rc.NL):
        r = run(leg, trace=True)
        out['per_leg'][rc.LEG_ORDER[leg]] = r
        print(rc.LEG_ORDER[leg], '%.3f -> final %+.4f load %.3f  %s' % (
            r['max_padz'], r['final_padz'], r['final_load'],
            ''.join(k[0].upper() for k, v in r['criteria'].items() if v)), flush=True)
    out['n_passed'] = sum(r['passed'] for r in out['per_leg'].values())
    out['passed_legs'] = [L for L, r in out['per_leg'].items() if r['passed']]
    (HERE / 'DEMO_STATE_TEACHER.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'n_passed': out['n_passed'], 'passed_legs': out['passed_legs']}))


if __name__ == '__main__':
    main()
