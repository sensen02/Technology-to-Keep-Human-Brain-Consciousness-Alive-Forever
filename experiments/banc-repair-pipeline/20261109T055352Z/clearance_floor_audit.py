"""Advance under three clearance floors, and the Core reuse question, in one recorded check.

Two things needed pinning down. First, the pre-registered advance window opens above the placement
height of 4 mm, so a pad hovering 1 mm above that height counts as airborne; reporting advance at
2 cm and 5 cm as well shows which legs swing with real clearance and which only shuffle near the
ground. Second, a hand written re-computation disagreed with evaluate(); running both a reused and
a fresh Core decides whether the recorded numbers or my re-computation were wrong.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

FLOORS = (float(T.THRESH['place_padz']), 0.02, 0.05)


def load_net():
    Z = np.load(HERE / 'MODEL_singleleg_demo_assembled.npz')
    net = T.Net(np.random.default_rng(11))
    for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2', 'wc0', 'bc0', 'wc', 'bc', 'logstd'):
        net.p[k] = Z['actor_' + k].astype(np.float64)
    return net


def per_leg(net, leg, core=None):
    core = core or rc.Core()
    core.reset()
    prompt = T.onehot(leg)
    st = dict(swing=False, start_x=0.)
    adv = {f: 0. for f in FLOORS}
    zmax = -9.; ticks = 0
    for k in range(T.T_EP):
        m = core.tick(np.tanh(net.forward(core.obs(prompt)[None])[2][0]))
        loads = m['loads']; pz = float(m['padz'][leg]); fx = float(m['foot'][leg][0])
        if loads[leg] <= T.SWING_START and not st['swing']:
            st['swing'] = True; st['start_x'] = fx
            adv = {f: 0. for f in FLOORS}; zmax = -9.; ticks = 0
        if st['swing']:
            dx = fx - st['start_x']
            zmax = max(zmax, pz); ticks += 1
            for f in FLOORS:
                if pz > f:
                    adv[f] = max(adv[f], dx)
            if loads[leg] >= T.SWING_END and pz <= T.THRESH['place_padz']:
                st['swing'] = False
    return dict(advance_by_floor={'%.4f' % f: float(adv[f]) for f in FLOORS},
                window_max_padz=float(zmax), window_ticks=ticks,
                real_swing=bool(adv[0.02] >= T.THRESH['advance']))


def main():
    net = load_net()
    reused = rc.Core()
    shared = {rc.LEG_ORDER[l]: per_leg(net, l, reused) for l in range(rc.NL)}
    fresh = {rc.LEG_ORDER[l]: per_leg(net, l) for l in range(rc.NL)}
    same = all(abs(shared[L]['advance_by_floor']['0.0040'] - fresh[L]['advance_by_floor']['0.0040']) < 1e-12
               for L in rc.LEG_ORDER)
    ev = T.evaluate(rc.Core(), net)
    agrees = all(abs(ev[L]['best_advance'] - shared[L]['advance_by_floor']['0.0040']) < 1e-9
                 for L in rc.LEG_ORDER)
    out = dict(label='advance by clearance floor; Core reuse and evaluator agreement',
               floors=list(FLOORS), reused_core=shared, fresh_core_per_leg=fresh,
               reused_equals_fresh=same,
               evaluator_best_advance={L: ev[L]['best_advance'] for L in rc.LEG_ORDER},
               evaluator_agrees_with_floor_0004=agrees,
               passed_legs=ev['_summary']['passed_legs'],
               low_clearance_passes=[L for L in rc.LEG_ORDER
                                     if ev[L]['passed'] and not shared[L]['real_swing']])
    (HERE / 'CLEARANCE_FLOOR_AUDIT.json').write_text(json.dumps(out, indent=2) + '\n')
    print('%-3s %10s %10s %10s %9s %6s' % ('leg', 'adv@4mm', 'adv@2cm', 'adv@5cm', 'max_padz', 'real'))
    for L in rc.LEG_ORDER:
        r = shared[L]
        print('%-3s %10.4f %10.4f %10.4f %9.4f %6s' % (
            L, r['advance_by_floor']['0.0040'], r['advance_by_floor']['0.0200'],
            r['advance_by_floor']['0.0500'], r['window_max_padz'],
            'yes' if r['real_swing'] else 'no'))
    print('reused core == fresh core:', same)
    print('evaluate() agrees with the 4 mm floor:', agrees)
    print('passed legs:', out['passed_legs'])
    print('passing but only a low clearance shuffle:', out['low_clearance_passes'])


if __name__ == '__main__':
    main()
