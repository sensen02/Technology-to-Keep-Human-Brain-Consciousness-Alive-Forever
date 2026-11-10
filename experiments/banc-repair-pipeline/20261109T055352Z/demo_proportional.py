"""Teacher with a proportional raise/forward sum instead of a clipped one.

clip(forward + g*raise, -1, 1) throws the forward component away exactly on the hind legs, which
need a large g to keep their pad up. Summing proportionally, (forward + g*raise)/(1+g), keeps both
directions at full relative weight and never saturates, so the swing can be commanded while the pad
is still being held up. Scored with the implemented acceptance rule.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def run(leg, g, lift, swing, press):
    core = rc.Core(); core.reset()
    st = dict(swing=False, start_x=0.); best = 0.; lift_max = -9.
    sup = 1e9; com = 1e9; nf = 0.; minl = 1e9
    rest = float(core.measure()['padz'][leg])
    for k in range(T.T_EP):
        ra = -T.descend_direction(core, leg)
        if k < lift:
            a = ra
        elif k < lift + swing:
            a = (T.forward_direction(core, leg) + g * ra) / (1.0 + g)
        else:
            a = T.descend_direction(core, leg)
        m = core.tick(a)
        loads = m['loads']; pz = float(m['padz'][leg]); fx = float(m['foot'][leg][0])
        lift_max = max(lift_max, pz)
        minl = min(minl, float(loads[leg]))
        if loads[leg] <= T.SWING_START and not st['swing']:
            st['swing'] = True; st['start_x'] = fx; best = 0.
        if st['swing']:
            best = max(best, fx - st['start_x'])
            if loads[leg] >= T.SWING_END and pz <= T.THRESH['place_padz']:
                st['swing'] = False
        sup = min(sup, float(loads.sum() / rc.WEIGHT)); com = min(com, float(m['com']))
        nf = max(nf, float(m['nonfoot']))
    fl = float(loads[leg] / rc.WEIGHT)
    crit = dict(unload=bool(minl / rc.WEIGHT <= T.THRESH['unload']),
                lift=bool(lift_max - rest >= T.THRESH['lift']),
                advance=bool(best >= T.THRESH['advance']),
                place=bool(fl >= T.THRESH['place_load'] and pz <= T.THRESH['place_padz']),
                safe=bool(sup >= T.THRESH['support'] and nf <= T.THRESH['nonfoot']
                          and com >= T.THRESH['com']))
    return dict(advance=best, rise=lift_max - rest, final_load=fl, final_padz=pz,
                support_min=sup, com_min=com, nonfoot_max=nf,
                criteria=crit, passed=all(crit.values()))


def main():
    out = dict(label='proportional raise/forward sum, scored with the implemented advance rule',
               per_config={})
    best = None
    for g in (0.0, 0.5, 1.0, 2.0, 4.0):
        for lift, swing in ((60, 150), (90, 120)):
            res = {rc.LEG_ORDER[l]: run(l, g, lift, swing, 300 - lift - swing) for l in range(rc.NL)}
            n = sum(r['passed'] for r in res.values())
            key = 'g=%.1f,lift=%d,swing=%d' % (g, lift, swing)
            out['per_config'][key] = dict(n_passed=n,
                                          passed_legs=[L for L, r in res.items() if r['passed']],
                                          per_leg=res)
            print('%-22s -> %d/6  %s' % (key, n, {L: '%s %.3f/%.2f' % (
                ''.join(k[0].upper() for k, v in r['criteria'].items() if v), r['advance'],
                r['final_load']) for L, r in res.items()}), flush=True)
            if best is None or n > best[1]:
                best = (key, n)
    out['best'] = best[0]; out['best_n'] = best[1]
    (HERE / 'DEMO_PROPORTIONAL.json').write_text(json.dumps(out, indent=2) + '\n')
    print('best:', out['best'], out['best_n'])


if __name__ == '__main__':
    main()
