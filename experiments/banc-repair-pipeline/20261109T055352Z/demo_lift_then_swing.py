"""Teacher with separated phases: raise to a measured clearance, then PURE forward swing.

Why previous teachers failed on the hind legs: they mixed a raise component into the swing command,
and the sum was clipped, so the forward part was thrown away exactly where it was needed. Measured
here: from a lifted pose, a pure forward command moves the right hind pad +0.0716 forward over 102
airborne ticks, while any mixed command gave exactly 0.0000.

Phases are triggered by measured pad height, not by a clock: raise until the pad is clear, swing
forward (pure forward, no raise component) until the pad comes down or the swing budget ends, then
press down. Everything is scored with the acceptance definition of advance.
"""
from pathlib import Path
import json, os, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def run(leg, clear, swing_budget, press_ticks, ticks):
    core = rc.Core(); core.reset()
    rest = float(core.measure()['padz'][leg])
    x0 = float(core.measure()['foot'][leg][0])
    state = 'raise'
    maxz = rest; minload = 1e9; sup = 1e9; com = 1e9; nf = 0.
    air_adv = 0.; air = 0; swang = 0; last = None
    for k in range(ticks):
        padz = float(core.measure()['padz'][leg])
        if state == 'raise':
            a = -T.descend_direction(core, leg)
            if padz >= clear:
                state = 'swing'
        elif state == 'swing':
            a = T.forward_direction(core, leg)          # pure forward, no raise component
            swang += 1
            if padz <= T.THRESH['place_padz'] or swang >= swing_budget:
                state = 'press'
        else:
            a = T.descend_direction(core, leg)
        m = core.tick(a)
        padz = float(m['padz'][leg]); dx = float(m['foot'][leg][0]) - x0
        maxz = max(maxz, padz); minload = min(minload, float(m['loads'][leg]))
        sup = min(sup, float(m['loads'].sum() / rc.WEIGHT))
        com = min(com, float(m['com'])); nf = max(nf, float(m['nonfoot']))
        if padz > T.THRESH['place_padz']:
            air += 1; air_adv = max(air_adv, dx)
        last = m
    fl = float(last['loads'][leg] / rc.WEIGHT); fp = float(last['padz'][leg])
    crit = dict(unload=bool(minload / rc.WEIGHT <= T.THRESH['unload']),
                lift=bool(maxz - rest >= T.THRESH['lift']),
                advance=bool(air_adv >= T.THRESH['advance']),
                place=bool(fl >= T.THRESH['place_load'] and fp <= T.THRESH['place_padz']),
                safe=bool(sup >= T.THRESH['support'] and nf <= T.THRESH['nonfoot']
                          and com >= T.THRESH['com']))
    return dict(rise=maxz - rest, air_advance=air_adv, airborne_ticks=air,
                final_load=fl, final_padz=fp, support_min=sup, com_min=com,
                nonfoot_max=nf, criteria=crit, passed=all(crit.values()))


def main():
    ticks = int(os.environ.get('T_EP', '300'))
    grid = [(c, b) for c in (.06, .12, .20) for b in (90, 150, 220)]
    out = dict(label='lift then pure forward swing; phases triggered by measured pad height',
               advance_metric='acceptance definition: only while the pad is above the placement height',
               per_config={})
    best = None
    for clear, budget in grid:
        res = {rc.LEG_ORDER[l]: run(l, clear, budget, 60, ticks) for l in range(rc.NL)}
        n = sum(r['passed'] for r in res.values())
        key = 'clear=%.2f,swing_budget=%d' % (clear, budget)
        out['per_config'][key] = dict(n_passed=n,
                                      passed_legs=[L for L, r in res.items() if r['passed']],
                                      per_leg=res)
        print('%-26s -> %d/6  %s' % (key, n, {L: '%s air=%.3f place=%.2f' % (
            ''.join(k[0].upper() for k, v in r['criteria'].items() if v),
            r['air_advance'], r['final_load']) for L, r in res.items()}), flush=True)
        if best is None or n > best[1]:
            best = (key, n)
    out['best'] = best[0]; out['best_n'] = best[1]
    (HERE / 'DEMO_LIFT_THEN_SWING.json').write_text(json.dumps(out, indent=2) + '\n')
    print('best:', out['best'], out['best_n'])


if __name__ == '__main__':
    main()
