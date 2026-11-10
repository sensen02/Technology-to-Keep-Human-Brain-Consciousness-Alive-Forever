"""A teacher self-check that uses the SAME advance definition as acceptance.

Correction to DEMO_STEP_TEACHER.json: that check measured forward displacement over the whole
episode, while the acceptance criterion measures it only while the pad is off the ground. Under
the acceptance definition the previous teacher scores 4/6, not 6/6: the left and right hind legs
move forward while planted and slightly backwards while airborne, so they record advance 0.0000.

This searches a four phase teacher (raise, hold clear, swing forward, lower) and scores every
candidate with the acceptance definition itself, per leg.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def run(leg, raise_ticks, hold_ticks, swing_ticks, hold_gain, ticks=None):
    ticks = ticks or T.T_EP
    core = rc.Core(); core.reset()
    rest = float(core.measure()['padz'][leg])
    x0 = float(core.measure()['foot'][leg][0])
    maxz = rest; minload = 1e9; sup = 1e9; com = 1e9; nf = 0.
    air_adv = 0.; whole_adv = 0.; air_ticks = 0; last = None
    for k in range(ticks):
        raise_a = -T.descend_direction(core, leg)
        if k < raise_ticks:
            a = raise_a
        elif k < raise_ticks + hold_ticks:
            a = raise_a                      # keep raising: the pad must be clearly airborne
        elif k < raise_ticks + hold_ticks + swing_ticks:
            a = np.clip(T.forward_direction(core, leg) + hold_gain * raise_a, -1., 1.)
        else:
            a = T.descend_direction(core, leg)
        m = core.tick(a)
        padz = float(m['padz'][leg]); dx = float(m['foot'][leg][0]) - x0
        maxz = max(maxz, padz); minload = min(minload, float(m['loads'][leg]))
        sup = min(sup, float(m['loads'].sum() / rc.WEIGHT))
        com = min(com, float(m['com'])); nf = max(nf, float(m['nonfoot']))
        whole_adv = max(whole_adv, dx)
        if padz > T.THRESH['place_padz']:
            air_ticks += 1
            air_adv = max(air_adv, dx)
        last = m
    final_load = float(last['loads'][leg] / rc.WEIGHT); final_padz = float(last['padz'][leg])
    crit = dict(unload=bool(minload / rc.WEIGHT <= T.THRESH['unload']),
                lift=bool(maxz - rest >= T.THRESH['lift']),
                advance=bool(air_adv >= T.THRESH['advance']),
                place=bool(final_load >= T.THRESH['place_load'] and final_padz <= T.THRESH['place_padz']),
                safe=bool(sup >= T.THRESH['support'] and nf <= T.THRESH['nonfoot']
                          and com >= T.THRESH['com']))
    return dict(rise=maxz - rest, air_advance=air_adv, whole_advance=whole_adv,
                air_ticks=air_ticks, final_load=final_load, final_padz=final_padz,
                support_min=sup, com_min=com, nonfoot_max=nf,
                criteria=crit, passed=all(crit.values()))


def main():
    grid = []
    for raise_ticks in (90, 150):
        for hold_ticks in (0, 60):
            for swing_ticks in (120, 180):
                for hold_gain in (0.5, 1.5):
                    grid.append((raise_ticks, hold_ticks, swing_ticks, hold_gain))
    out = dict(label='four phase teacher scored with the acceptance advance definition',
               note='DEMO_STEP_TEACHER.json measured whole-episode advance and therefore overstated the teacher',
               grid=[dict(raise_ticks=a, hold_ticks=b, swing_ticks=c, hold_gain=d) for a, b, c, d in grid],
               per_config={})
    best = None
    for cfg in grid:
        res = {rc.LEG_ORDER[l]: run(l, *cfg) for l in range(rc.NL)}
        n = sum(r['passed'] for r in res.values())
        key = 'raise=%d,hold=%d,swing=%d,gain=%.1f' % cfg
        out['per_config'][key] = dict(n_passed=n,
                                      passed_legs=[L for L, r in res.items() if r['passed']],
                                      per_leg=res)
        print('%-34s -> %d/6  %s' % (key, n,
              {L: ''.join(k[0].upper() for k, v in r['criteria'].items() if v) for L, r in res.items()}),
              flush=True)
        if best is None or n > best[1]:
            best = (key, n, cfg)
    out['best'] = best[0]; out['best_n'] = best[1]; out['best_config'] = dict(
        raise_ticks=best[2][0], hold_ticks=best[2][1], swing_ticks=best[2][2], hold_gain=best[2][3])
    (HERE / 'DEMO_AIR_TEACHER.json').write_text(json.dumps(out, indent=2) + '\n')
    print('best:', out['best'], out['best_n'])


if __name__ == '__main__':
    main()
