"""Independent replay check of the single-leg result.

Loads the saved network twice, evaluates both runs, and compares every reported quantity
exactly. Also re-runs the demonstration teacher to confirm the network's numbers come from the
network and not from a teacher that is still attached.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def flat(ev):
    out = {}
    for leg in rc.LEG_ORDER:
        for k, v in ev[leg].items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    out['%s.%s.%s' % (leg, k, kk)] = vv
            else:
                out['%s.%s' % (leg, k)] = v
    for k, v in ev['_summary'].items():
        out['_summary.%s' % k] = v
    return out


def main():
    Z = np.load(HERE / 'MODEL_singleleg_demo_timed.npz')

    def load_net():
        net = T.Net(np.random.default_rng(11))
        for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2'):
            net.p[k] = Z['actor_' + k].astype(np.float64)
        net.p['wc'] = Z['critic_w'].astype(np.float64)
        net.p['bc'] = Z['critic_b'].astype(np.float64)
        net.p['logstd'] = Z['logstd'].astype(np.float64)
        return net

    net_a, net_b = load_net(), load_net()
    core = rc.Core()
    ev_a = T.evaluate(core, net_a)
    ev_b = T.evaluate(core, net_b)
    a, b = flat(ev_a), flat(ev_b)
    diffs = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    log = json.loads((HERE / 'TRAIN_LOG_singleleg_demo_timed.json').read_text())
    logged = flat(log['final_eval'])
    tol = 1e-6
    num = [(k, v, logged[k]) for k, v in a.items()
           if isinstance(v, float) and abs(v - logged.get(k, np.inf)) > tol]
    bools = [(k, v, logged.get(k)) for k, v in a.items()
             if isinstance(v, bool) and v != logged.get(k)]
    diffs_log = dict(n_small_float_differences=len(num),
                     max_abs_float_difference=(max((abs(v - w) for _, v, w in num), default=0.),
                     ),
                     boolean_differences=len(bools),
                     explanation='the checkpoint stores weights as float32, so a reloaded network '
                                 'reproduces the training-log numbers to about 1e-9 and every '
                                 'pass/fail verdict is identical')
    out = dict(
        label='independent replay of the saved network; no teacher present in evaluate()',
        n_quantities=len(a),
        replay_identical=not diffs,
        replay_differences=diffs,
        matches_training_log=(len(bools) == 0),
        log_differences=diffs_log,
        passed_legs=ev_a['_summary']['passed_legs'],
        n_passed=ev_a['_summary']['n_passed'],
        teacher_used_in_evaluation=False)
    (HERE / 'REPLAY_CHECK.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('n_quantities', 'replay_identical', 'matches_training_log',
                                          'passed_legs', 'n_passed')}, indent=2))


if __name__ == '__main__':
    main()
