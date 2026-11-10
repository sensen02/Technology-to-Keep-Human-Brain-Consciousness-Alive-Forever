"""Assemble one model whose six sub-networks each come from that leg's own run.

After PREREG_AMENDMENT_11 and 12 nothing is shared between legs except the exploration scale,
so this assembly is equivalent to training one network leg by leg. It is a declared engineering
step: the resulting artefact is a single replacement-ANN policy file, selected by the leg prompt,
with no clock and no teacher at evaluation.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

KEYS = ('w0', 'b0', 'w1', 'b1', 'w2', 'b2', 'wc0', 'bc0', 'wc', 'bc')


def main():
    net = T.Net(np.random.default_rng(0))
    src = {}
    for leg in range(rc.NL):
        name = rc.LEG_ORDER[leg]
        f = HERE / ('MODEL_singleleg_demo_solo%s.npz' % name)
        if not f.exists():
            raise SystemExit('missing checkpoint for %s: %s' % (name, f))
        Z = np.load(f)
        for k in KEYS:
            net.p[k][leg] = Z['actor_' + k].astype(np.float64)[leg]
        net.p['logstd'] = Z['logstd'].astype(np.float64)
        src[name] = str(f.name)
    core = rc.Core()
    res = T.evaluate(core, net)
    out = dict(label='assembled from six per-leg runs; single replacement-ANN policy, no clock, no teacher',
               sources=src,
               note='the exploration scale (logstd) is the only shared parameter and is taken from the last source',
               summary=res['_summary'],
               per_leg={L: res[L] for L in rc.LEG_ORDER})
    net.save_to(HERE / 'MODEL_singleleg_demo_assembled.npz',
                dict(scheme=np.array('assembled_per_leg_subnets'),
                     n_passed=np.array(res['_summary']['n_passed']),
                     passed_legs=np.array(','.join(res['_summary']['passed_legs']))))
    (HERE / 'ASSEMBLE_SOLO.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(res['_summary']))
    for L in rc.LEG_ORDER:
        v = res[L]
        print('%-2s lift %+.4f adv %.4f load %.3f padz %+.4f sup %.3f com %.3f nf %.2f  %s' % (
            L, v['max_lift'], v['best_advance'], v['final_load_over_body_weight'], v['final_padz'],
            v['support_min'], v['com_min'], v['nonfoot_max'],
            ''.join(k[0].upper() for k, x in v['criteria'].items() if x)))


if __name__ == '__main__':
    main()
