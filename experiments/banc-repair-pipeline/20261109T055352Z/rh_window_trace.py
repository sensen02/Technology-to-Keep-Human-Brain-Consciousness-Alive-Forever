"""Trace the evaluation's own swing window for the right hind leg.

The acceptance reads "forward displacement of the pad while it is in the air". The implementation
opens that window on a load test and closes it on a load plus height test. For a rear leg the
window may open at tick 0 (rear legs carry almost no load in the stand pose) and close before the
forward swing happens, which would record advance 0.0000 even though the pad does move forward.
This records the window state next to the pad position, tick by tick.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def load_net(path):
    Z = np.load(path)
    net = T.Net(np.random.default_rng(11))
    for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2'):
        net.p[k] = Z['actor_' + k].astype(np.float64)
    net.p['logstd'] = Z['logstd'].astype(np.float64)
    return net


def trace(net, leg):
    core = rc.Core(); core.reset()
    prompt = T.onehot(leg)
    st = dict(swing=False, start_x=0., max_lift=0., best_adv=0., prev_x=0.)
    rows = []
    for k in range(T.T_EP):
        obs = core.obs(prompt)
        m = core.tick(np.tanh(net.forward(obs[None])[2][0]))
        loads = m['loads']; padz = float(m['padz'][leg]); fx = float(m['foot'][leg][0])
        if loads[leg] <= T.SWING_START and not st['swing']:
            st['swing'] = True; st['start_x'] = fx; st['max_lift'] = 0.; st['best_adv'] = 0.
        if st['swing']:
            st['max_lift'] = max(st['max_lift'], padz)
            st['best_adv'] = max(st['best_adv'], fx - st['start_x'])
            if loads[leg] >= T.SWING_END and padz <= T.THRESH['place_padz']:
                st['swing'] = False
        rows.append(dict(k=k, load=round(float(loads[leg] / rc.WEIGHT), 3),
                         padz=round(padz, 4), fx=round(fx, 4), swing=st['swing'],
                         adv=round(st['best_adv'], 4),
                         dx=round(fx - st['start_x'], 4) if st['swing'] else None))
    return rows


def main():
    net = load_net(HERE / 'MODEL_singleleg_demo_soloRH.npz')
    leg = rc.LEG_ORDER.index('RH')
    rows = trace(net, leg)
    air = [r for r in rows if r['padz'] > T.THRESH['place_padz']]
    out = dict(label='evaluation swing-window trace for RH; measurement only',
               swing_open_ticks=[r['k'] for r in rows if r['swing']],
               best_advance=rows[-1]['adv'],
               airborne_ticks=[r['k'] for r in air],
               airborne_forward_range=[min((r['fx'] for r in air), default=None),
                                       max((r['fx'] for r in air), default=None)],
               trace=[r for r in rows if r['k'] % 10 == 0 or r['k'] > T.T_EP - 20])
    (HERE / 'RH_WINDOW_TRACE.json').write_text(json.dumps(out, indent=2) + '\n')
    print('swing window open ticks:', out['swing_open_ticks'][:6], '...',
          'count', len(out['swing_open_ticks']))
    print('best_advance recorded:', out['best_advance'])
    print('airborne ticks:', len(air), 'forward range while airborne:', out['airborne_forward_range'])
    for r in out['trace']:
        print('  k%3d load %.3f padz %+.4f fx %+.4f swing %-5s adv %.4f dx %s' % (
            r['k'], r['load'], r['padz'], r['fx'], r['swing'], r['adv'], r['dx']))


if __name__ == '__main__':
    main()
