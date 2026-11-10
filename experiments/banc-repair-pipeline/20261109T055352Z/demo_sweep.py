"""A teacher must pass the task itself before anything is taught with it.

The two-phase demonstration was used for a training run without this check; DEMO_CYCLE.json then
showed it lifts to pad height 1.33 and oscillates between 1.17 and 1.33, never coming down --
its mirrored 'press' direction only undoes about 0.15 of the rise. This sweep searches the two
declared demo parameters (command magnitude and the excursion at which the press starts) for
configurations where the demonstration itself satisfies the task's five criteria on a leg.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def run_demo(core, leg, demo, scale, switch, ticks=T.T_EP):
    core.reset()
    rest = float(core.measure()['padz'][leg])
    maxz = rest; minload = 1e9; sup_min = 1e9; com_min = 1e9; nonfoot = 0.
    start_x = float(core.measure()['foot'][leg][0]); best_adv = 0.
    last = None
    for k in range(ticks):
        m = core.measure()
        load = float(m['loads'][leg])
        exc = T.excursion(core, leg, demo['unit'])
        unit = demo['unit'] if (load > T.SWING_END or exc < switch) else demo['press']
        mm = core.tick(np.clip(unit * scale, -1., 1.))
        padz = float(mm['padz'][leg])
        maxz = max(maxz, padz)
        minload = min(minload, float(mm['loads'][leg]))
        sup_min = min(sup_min, float(mm['loads'].sum() / rc.WEIGHT))
        com_min = min(com_min, float(mm['com']))
        nonfoot = max(nonfoot, float(mm['nonfoot']))
        best_adv = max(best_adv, float(mm['foot'][leg][0]) - start_x)
        last = mm
    final_load = float(last['loads'][leg] / rc.WEIGHT)
    final_padz = float(last['padz'][leg])
    crit = dict(unload=bool(minload / rc.WEIGHT <= T.THRESH['unload']),
                lift=bool(maxz >= T.THRESH['lift']),
                advance=bool(best_adv >= T.THRESH['advance']),
                place=bool(final_load >= T.THRESH['place_load'] and final_padz <= T.THRESH['place_padz']),
                safe=bool(sup_min >= T.THRESH['support'] and nonfoot <= T.THRESH['nonfoot']
                          and com_min >= T.THRESH['com']))
    return dict(max_padz=maxz, final_padz=final_padz, final_load=final_load,
                min_load=minload / rc.WEIGHT, best_advance=best_adv, support_min=sup_min,
                com_min=com_min, nonfoot_max=nonfoot, criteria=crit, passed=all(crit.values()))


def main():
    core = rc.Core(); demos = T.demo_directions()
    grid = [(s, sw) for s in (1.0, .5, .25, .15, .1) for sw in (.5, .35, .2)]
    out = dict(label='teacher self-check sweep; no learning', grid=[{'scale': s, 'switch': sw}
                                                                   for s, sw in grid],
               per_config={})
    best = None
    for s, sw in grid:
        rows = {}
        for leg in range(rc.NL):
            rows[rc.LEG_ORDER[leg]] = run_demo(core, leg, demos[leg], s, sw)
        n_pass = sum(r['passed'] for r in rows.values())
        out['per_config']['scale=%.2f,switch=%.2f' % (s, sw)] = dict(n_passed=n_pass, per_leg=rows)
        if best is None or n_pass > best[2]:
            best = (s, sw, n_pass)
        print('scale %.2f switch %.2f -> %d/6 pass  %s' % (
            s, sw, n_pass, {L: ''.join(k[0].upper() for k, v in r['criteria'].items() if v)
                            for L, r in rows.items()}), flush=True)
    out['best'] = dict(scale=best[0], switch=best[1], n_passed=best[2])
    out['conclusion'] = ('the demonstration that was used for the last run (scale 1.0, switch 0.5) '
                         'cannot place a foot on any leg; a teacher must pass the task before it is '
                         'used to teach')
    (HERE / 'DEMO_SWEEP.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['best'], indent=2))


if __name__ == '__main__':
    main()
