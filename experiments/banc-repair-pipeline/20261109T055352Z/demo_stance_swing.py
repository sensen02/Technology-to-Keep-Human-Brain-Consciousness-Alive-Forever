"""Demonstration that first moves the body into the solved five leg stance, then swings the leg.

Every earlier demonstration raised the leg inside the nominal six leg pose, and the right hind leg
could only move forward by letting the body collapse (non-foot 37 to 42 W). The stance solved in
repair_round1j puts the other five pads coplanar with COM at 1.4507, so this reaches that stance by
joint interpolation first, then swings, then presses. Scored with the implemented acceptance rule.
"""
from pathlib import Path
import json, os, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

MAX_RATE = rc.MAX_RATE


def run(leg, pose, gain, swing, lead_max, press_ticks, ticks=None):
    ticks = ticks or T.T_EP
    q5 = np.asarray(pose['q_rad'], float)
    core = rc.Core(); core.reset()
    rest = float(core.measure()['padz'][leg])
    st = dict(swing=False, start_x=0.); best = 0.; lift_max = -9.
    sup = 1e9; com = 1e9; nf = 0.; minl = 1e9
    lead = 0; swang = 0; state = 'lead'
    for k in range(ticks):
        q = np.asarray([float(core.d.qpos[int(a)]) for a in core.a_qpos])
        if state == 'lead':
            a = np.clip((q5 - q) * gain / rc.PDT, -MAX_RATE, MAX_RATE) / MAX_RATE
            lead += 1
            if np.abs(q5 - q).max() < 0.05 or lead >= lead_max:
                state = 'swing'
        elif state == 'swing':
            a = T.forward_direction(core, leg)
            swang += 1
            if swang >= swing:
                state = 'press'
        else:
            a = T.descend_direction(core, leg)
        m = core.tick(a)
        loads = m['loads']; pz = float(m['padz'][leg]); fx = float(m['foot'][leg][0])
        lift_max = max(lift_max, pz); minl = min(minl, float(loads[leg]))
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
    return dict(advance=best, rise=lift_max - rest, final_load=fl, lead_ticks=lead,
                swing_ticks=swang, support_min=sup, com_min=com, nonfoot_max=nf,
                criteria=crit, passed=all(crit.values()))


def main():
    leg = rc.LEG_ORDER.index('RH')
    pose = json.loads((HERE / 'STAND_FIVE_LEG_RH_145.json').read_text())
    out = dict(label='lead in to the solved five leg stance, then swing the right hind leg',
               amendment='PREREG_AMENDMENT_13.json', rows=[])
    best = None
    for gain in (20., 60.):
        for swing in (60, 120):
            for lead_max in (90, 150):
                r = run(leg, pose, gain, swing, lead_max, 60)
                r.update(gain=gain, swing=swing, lead_max=lead_max)
                out['rows'].append(r)
                print('gain %4.0f swing %3d lead_max %3d | adv %.4f rise %.4f lead %3d load %.3f | '
                      'sup %.3f com %.3f nf %.2f  %s' % (
                          gain, swing, lead_max, r['advance'], r['rise'], r['lead_ticks'],
                          r['final_load'], r['support_min'], r['com_min'], r['nonfoot_max'],
                          ''.join(k[0].upper() for k, v in r['criteria'].items() if v)), flush=True)
                if best is None or (r['advance'], r['support_min']) > (best['advance'], best['support_min']):
                    best = r
    out['best'] = dict(gain=best['gain'], swing=best['swing'], lead_max=best['lead_max'],
                       advance=best['advance'], support_min=best['support_min'],
                       passed=best['passed'])
    out['any_passed'] = any(r['passed'] for r in out['rows'])
    out['advance_and_safe'] = any(r['criteria']['advance'] and r['criteria']['safe'] for r in out['rows'])
    (HERE / 'DEMO_STANCE_SWING.json').write_text(json.dumps(out, indent=2) + '\n')
    print('any configuration passing:', out['any_passed'],
          '| advance together with safety:', out['advance_and_safe'])


if __name__ == '__main__':
    main()
