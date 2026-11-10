"""Score a teacher with the SAME criterion code the acceptance uses, per leg.

Correction this file exists for: I described the acceptance advance as "forward displacement while
the pad is off the ground". Reading the implemented evaluator, advance is the maximum forward pad
displacement inside the SWING WINDOW DEFINED BY FOOT LOAD (window opens when the leg's load drops to
5% of body weight, closes when load is back to 10% and the pad is at the placement height). There is
no pad height filter on the advance itself. My teacher self-checks used a height filtered variant,
which is stricter than the acceptance, so "teacher 4/6" was measured under a stricter rule than the
one the network is judged by. This scores both teachers under the implemented rule and under the
stricter height filtered rule, side by side.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def score(leg, policy, ticks=None):
    ticks = ticks or T.T_EP
    core = rc.Core(); core.reset()
    st = dict(swing=False, start_x=0., max_lift=0., best_adv=0., strict=0.)
    sup = 1e9; com = 1e9; nf = 0.; minload = 1e9
    last = None
    for k in range(ticks):
        a = policy(core, leg, k)
        m = core.tick(a)
        loads = m['loads']; pz = float(m['padz'][leg]); fx = float(m['foot'][leg][0])
        minload = min(minload, float(loads[leg]))
        if loads[leg] <= T.SWING_START and not st['swing']:
            st['swing'] = True; st['start_x'] = fx; st['max_lift'] = 0.; st['best_adv'] = 0.; st['strict'] = 0.
        if st['swing']:
            st['max_lift'] = max(st['max_lift'], pz)
            st['best_adv'] = max(st['best_adv'], fx - st['start_x'])
            if pz > T.THRESH['place_padz']:
                st['strict'] = max(st['strict'], fx - st['start_x'])
            if loads[leg] >= T.SWING_END and pz <= T.THRESH['place_padz']:
                st['swing'] = False
        sup = min(sup, float(loads.sum() / rc.WEIGHT)); com = min(com, float(m['com']))
        nf = max(nf, float(m['nonfoot'])); last = m
    fl = float(last['loads'][leg] / rc.WEIGHT); fp = float(last['padz'][leg])
    base = dict(unload=bool(minload / rc.WEIGHT <= T.THRESH['unload']),
                lift=bool(st['max_lift'] >= T.THRESH['lift']),
                place=bool(fl >= T.THRESH['place_load'] and fp <= T.THRESH['place_padz']),
                safe=bool(sup >= T.THRESH['support'] and nf <= T.THRESH['nonfoot']
                          and com >= T.THRESH['com']))
    implemented = dict(base, advance=bool(st['best_adv'] >= T.THRESH['advance']))
    strict = dict(base, advance=bool(st['strict'] >= T.THRESH['advance']))
    return dict(advance_implemented=st['best_adv'], advance_height_filtered=st['strict'],
                max_lift=st['max_lift'], final_load=fl, final_padz=fp,
                support_min=sup, com_min=com, nonfoot_max=nf,
                passed_implemented=all(implemented.values()),
                passed_height_filtered=all(strict.values()),
                criteria_implemented=implemented)


def old_teacher(core, leg, k):
    return T.teacher_action(core, leg, k)


def lift_then_swing(clear, swing_budget):
    def pol(core, leg, k):
        pz = float(core.measure()['padz'][leg])
        state = pol.state
        if state.get('s') is None:
            pol.state = dict(s='raise', swang=0)
            state = pol.state
        if state['s'] == 'raise':
            if pz >= clear:
                state['s'] = 'swing'
            return -T.descend_direction(core, leg)
        if state['s'] == 'swing':
            state['swang'] += 1
            if pz <= T.THRESH['place_padz'] or state['swang'] >= swing_budget:
                state['s'] = 'press'
            return T.forward_direction(core, leg)
        return T.descend_direction(core, leg)
    return pol


def main():
    out = dict(label='teacher scores under the implemented advance rule and under the stricter rule',
               implemented_rule='advance = max forward pad displacement inside the load defined swing window',
               note='the height filtered column is the stricter variant my earlier self-checks used',
               teachers={})
    for name, gate in (('three_phase_used_in_training', None),
                       ('lift_then_swing_clear006', (0.06, 90)),
                       ('lift_then_swing_clear012', (0.12, 150))):
        res = {}
        for leg in range(rc.NL):
            pol = lift_then_swing(*gate) if gate else old_teacher
            pol.state = {}
            res[rc.LEG_ORDER[leg]] = score(leg, pol)
        n_impl = sum(r['passed_implemented'] for r in res.values())
        n_strict = sum(r['passed_height_filtered'] for r in res.values())
        out['teachers'][name] = dict(n_passed_implemented=n_impl, n_passed_strict=n_strict,
                                     passed_implemented=[L for L, r in res.items() if r['passed_implemented']],
                                     passed_strict=[L for L, r in res.items() if r['passed_height_filtered']],
                                     per_leg=res)
        print('%-26s implemented %d/6 %-24s strict %d/6 %s' % (
            name, n_impl, str([L for L, r in res.items() if r['passed_implemented']]),
            n_strict, [L for L, r in res.items() if r['passed_height_filtered']]), flush=True)
        for L, r in res.items():
            print('    %-2s adv(impl) %.4f adv(strict) %.4f lift %.4f load %.3f %s' % (
                L, r['advance_implemented'], r['advance_height_filtered'], r['max_lift'],
                r['final_load'], ''.join(k[0].upper() for k, v in r['criteria_implemented'].items() if v)),
                flush=True)
    (HERE / 'TEACHER_EVAL.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
