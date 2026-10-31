"""Engine-free regressions for S178 (mandate item 7). No engine import."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from legjoint_release_core import BlockHillClimb, row_intervals, sign_flip_estimate, BudgetStop


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    tests = []

    def check(name, cond):
        assert cond, name
        tests.append(dict(test=name, passed=True))

    # 1) row interval / violation arithmetic
    A = np.array([[1., -2., 0.], [0., 0., 1.]])
    lo, hi, viol, reach = row_intervals(A, np.array([0.5, 2.0]))
    check('interval lower/upper', np.allclose(lo, [-2., 0.]) and np.allclose(hi, [1., 1.]))
    check('violation and reachability flags', viol[0] == 0. and viol[1] == 1.0 and reach[0] and not reach[1])

    # 2) sign-flip extrapolation from scan samples only
    ang = np.linspace(-.6, .6, 9)
    col = 2.0 + 5.0*ang
    est = sign_flip_estimate(col, ang, base_index=4)
    check('slope recovered', abs(est['slope'] - 5.0) < 1e-9)
    check('angle-need arithmetic', abs(est['angle_needed'] - 2.0/5.0) < 1e-9 and est['flippable'])

    # 3) hill-climb: synthetic evaluate where only block 0 matters
    lower = np.full(14, -1.); upper = np.full(14, 1.)
    calls = []

    def evaluate(q, env):
        calls.append(1)
        m = float((q[0]-0.5)**2 + (q[7]-0.3)**2)
        return dict(root_ok=True, root_norm=1e-12, minimax=m, geom_ok=True, unreachable=[],
                    contact_signature='s0' if q[0] < 0.8 else 's1', residual6=[0.]*6, rows=[], row_violation=[])

    hc = BlockHillClimb(evaluate, lower, upper, blocks=[np.arange(0, 7), np.arange(7, 14)], cap=400,
                        output=out/'ledger_hc.json', name='hc')
    res = hc.run(np.zeros(14))
    check('hill-climb accepts moves', hc.ledger['accepted_moves'] > 0)
    check('hill-climb monotone', all(res['history'][i]['minimax'] >= res['history'][i+1]['minimax'] - 1e-15
                                     for i in range(len(res['history'])-1)))
    check('hill-climb decreases objective', res['final']['minimax'] < 8.0)

    # 4) planted global best must not change accept decisions
    hc2 = BlockHillClimb(evaluate, lower, upper, blocks=[np.arange(0, 7)], cap=200, name='planted')
    hc2.best = dict(minimax=-1e9, summary={})
    pre = hc2.best['minimax']
    hc2.run(np.zeros(14))
    check('planted best untouched by accept logic', hc2.best['minimax'] == pre == -1e9)

    # 5) zero accept must be reported, never treated as evidence
    def flat(q, env):
        return dict(root_ok=True, root_norm=1e-12, minimax=1.0, geom_ok=True, unreachable=[], contact_signature='s',
                    residual6=[0.]*6, rows=[], row_violation=[])
    hc3 = BlockHillClimb(flat, lower, upper, blocks=[np.arange(0, 7)], cap=300, name='flat')
    r3 = hc3.run(np.zeros(14))
    check('zero accept signalled', r3['stop'] == 'ZERO_ACCEPT_ALL_SWEEPS')

    # 6) root-not-balanced trials are rejected even if minimax improves
    def rootfail(q, env):
        return dict(root_ok=False, root_norm=1e-2, minimax=0.0, geom_ok=True, unreachable=[], contact_signature='s',
                    residual6=[1.]*6, rows=[], row_violation=[])
    hc4 = BlockHillClimb(rootfail, lower, upper, blocks=[np.arange(0, 7)], cap=200, name='rootfail')
    r4 = hc4.run(np.zeros(14))
    check('root failure blocks acceptance', hc4.ledger['accepted_moves'] == 0 and hc4.ledger['rejected_root'] > 0)

    # 7) geometry violations are rejected even if minimax improves
    def geomfail(q, env):
        return dict(root_ok=True, root_norm=1e-12, minimax=0.0, geom_ok=False, unreachable=[], contact_signature='s',
                    residual6=[0.]*6, rows=[], row_violation=[])
    hc5 = BlockHillClimb(geomfail, lower, upper, blocks=[np.arange(0, 7)], cap=200, name='geomfail')
    hc5.run(np.zeros(14))
    check('geometry failure blocks acceptance', hc5.ledger['accepted_moves'] == 0 and hc5.ledger['rejected_geometry'] > 0)

    # 8) budget: block refused before starting when remaining is insufficient
    hc6 = BlockHillClimb(evaluate, lower, upper, blocks=[np.arange(0, 7)], cap=8, reserved=6, name='budget')
    r6 = hc6.run(np.zeros(14))
    check('block refused under reserved budget', r6['stop'] == 'BUDGET_BEFORE_BLOCK')

    # 9) no scalar merge: score keys stay separate
    keys = set(res['final'].keys())
    check('separate reporting fields exist', {'root_norm', 'minimax', 'geom_ok'} <= keys and 'score' not in keys)

    check('no engine imported', 'mujoco' not in sys.modules)
    (out/'TEST_RESULTS.json').write_text(json.dumps(dict(passed=True, engineering_only=True, model_load=0, forward=0, tests=tests), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    run(ap.parse_args().output); print('S178 core regressions passed')
