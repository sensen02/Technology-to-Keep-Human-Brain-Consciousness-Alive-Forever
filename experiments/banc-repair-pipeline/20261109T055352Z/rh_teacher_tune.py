"""Tune the proportional teacher around the point where the right hind leg first advanced and placed.

At g=4, lift=90, swing=120 the right hind leg reached advance 0.065 and final load 0.18 under the
implemented rule; only the safety block failed. This maps the neighbourhood so the safety failure can
be attributed: too little support time, a body height dip, or a non-foot contact.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def run(leg, g, lift, swing):
    core = rc.Core(); core.reset()
    st = dict(swing=False, start_x=0.); best = 0.; lift_max = -9.
    sup = 1e9; com = 1e9; nf = 0.; minl = 1e9
    sup_at = com_at = nf_at = -1
    rest = float(core.measure()['padz'][leg])
    for k in range(T.T_EP):
        ra = -T.descend_direction(core, leg)
        if k < lift:
            a = ra
        elif k < lift + swing:
            a = (T.forward_direction(core, leg) + g * ra) / (1.0 + g)
        elif k < lift + swing + 60:
            a = T.descend_direction(core, leg)
        else:
            a = np.zeros(rc.NJ)
        m = core.tick(a)
        loads = m['loads']; pz = float(m['padz'][leg]); fx = float(m['foot'][leg][0])
        lift_max = max(lift_max, pz); minl = min(minl, float(loads[leg]))
        if loads[leg] <= T.SWING_START and not st['swing']:
            st['swing'] = True; st['start_x'] = fx; best = 0.
        if st['swing']:
            best = max(best, fx - st['start_x'])
            if loads[leg] >= T.SWING_END and pz <= T.THRESH['place_padz']:
                st['swing'] = False
        s = float(loads.sum() / rc.WEIGHT); c = float(m['com']); f = float(m['nonfoot'])
        if s < sup: sup, sup_at = s, k
        if c < com: com, com_at = c, k
        if f > nf: nf, nf_at = f, k
    fl = float(loads[leg] / rc.WEIGHT)
    crit = dict(unload=bool(minl / rc.WEIGHT <= T.THRESH['unload']),
                lift=bool(lift_max - rest >= T.THRESH['lift']),
                advance=bool(best >= T.THRESH['advance']),
                place=bool(fl >= T.THRESH['place_load'] and pz <= T.THRESH['place_padz']),
                safe=bool(sup >= T.THRESH['support'] and nf <= T.THRESH['nonfoot']
                          and com >= T.THRESH['com']))
    return dict(advance=best, rise=lift_max - rest, final_load=fl, support_min=sup,
                support_at=sup_at, com_min=com, com_at=com_at, nonfoot_max=nf, nonfoot_at=nf_at,
                criteria=crit, passed=all(crit.values()))


def main():
    out = dict(label='tuning the proportional teacher for the right hind leg',
               threshold=dict(support=T.THRESH['support'], com=T.THRESH['com'],
                              nonfoot=T.THRESH['nonfoot'], advance=T.THRESH['advance']),
               rows=[])
    leg = rc.LEG_ORDER.index('RH')
    for g in (3.0, 4.0, 6.0, 8.0):
        for lift, swing in ((90, 120), (110, 100), (70, 130), (130, 80)):
            r = run(leg, g, lift, swing)
            r.update(g=g, lift=lift, swing=swing)
            out['rows'].append(r)
            print('g=%4.1f lift %3d swing %3d | adv %.4f rise %.4f load %.3f | sup %.3f@%d com %.3f@%d nf %.2f@%d  %s' % (
                g, lift, swing, r['advance'], r['rise'], r['final_load'],
                r['support_min'], r['support_at'], r['com_min'], r['com_at'],
                r['nonfoot_max'], r['nonfoot_at'],
                ''.join(k[0].upper() for k, v in r['criteria'].items() if v)), flush=True)
    out['passing'] = [dict(g=r['g'], lift=r['lift'], swing=r['swing']) for r in out['rows'] if r['passed']]
    (HERE / 'RH_TEACHER_TUNE.json').write_text(json.dumps(out, indent=2) + '\n')
    print('passing configurations:', len(out['passing']), '/', len(out['rows']))


if __name__ == '__main__':
    main()
