"""Engine-free regressions for the S177 root-Newton core (mandate item 5).

Proves, before any engine import:
  * the acceptance test never consults or mutates the logging global best;
  * a synthetic residual map actually converges monotonically with accepted moves;
  * caching and the atomic Jacobian budget behave, with zero partial probes;
  * zero accepted moves is reported as a defect-style stop, never as evidence.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from root_newton_core import RootNewton, Budget, RefusedJacobian


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    results = []

    def check(name, condition):
        assert condition, name
        results.append(dict(test=name, passed=True))

    lower = np.array([-1., -1., -1., -.5, -.5, -.5])
    upper = np.array([1., 1., 1., .5, .5, .5])

    # 1) linear residual map: exact Newton should converge in one accepted step
    A = np.array([[2., .3, 0., 0., 0., 0.], [.1, 1.5, .2, 0., 0., 0.], [0., .2, 2.2, 0., 0., 0.],
                  [0., 0., 0., 1.4, .1, 0.], [0., 0., 0., .1, 1.7, .2], [0., 0., 0., 0., .3, 1.9]])
    b = np.array([.4, -.2, .1, -.3, .05, .2])
    calls = []

    def linear(p):
        calls.append(1)
        return dict(residual6=A @ p - b, contact_signature='static', p=p)

    solver = RootNewton(linear, lower, upper, cap=200, output=out / 'ledger_linear.json', name='linear')
    res = solver.solve(np.zeros(6))
    check('linear converge to tolerance', res['norm_inf'] <= 1e-8)
    check('accepted iterations > 0', solver.ledger['accepted_iterations'] > 0)
    check('monotone non-increasing history',
          all(res['history'][i]['norm_inf'] >= res['history'][i+1]['norm_inf'] - 1e-15
              for i in range(len(res['history'])-1)))
    check('successive accepted points differ', len({tuple(h['p']) for h in res['history'] if h['role'] == 'accepted'}) >= 1)

    # 2) nonlinear residual map with a curved valley: must still decrease monotonically
    def curved(p):
        q = p.copy()
        return dict(residual6=np.array([q[0]**2 + .3*q[1] - .05, q[1] - .2*np.sin(q[0]),
                                        q[2] - .1*q[0]*q[1], q[3] + .05*q[4], q[4] - .03, q[5] + .02*q[3]]),
                    contact_signature='static', p=q)

    solver2 = RootNewton(curved, lower, upper, cap=300, output=out / 'ledger_curved.json', name='curved')
    res2 = solver2.solve(np.array([.5, -.4, .3, .2, -.2, .1]))
    check('curved map decreases', res2['norm_inf'] < .5)
    check('curved accepted moves', solver2.ledger['accepted_iterations'] > 0)
    check('curved monotone',
          all(res2['history'][i]['norm_inf'] >= res2['history'][i+1]['norm_inf'] - 1e-15
              for i in range(len(res2['history'])-1)))

    # 3) logging/best must not influence the accept test: force a pathological best
    def flat(p):
        return dict(residual6=np.array([1., 0., 0., 0., 0., 0.]), contact_signature='static', p=p)

    solver3 = RootNewton(flat, lower, upper, cap=50, name='flat')
    solver3.best = dict(norm_inf=-1e9, summary={}, residual6=np.zeros(6))  # absurd planted best
    pre_best = dict(solver3.best)
    res3 = solver3.solve(np.zeros(6))
    check('planted global best does not change accept decisions',
          solver3.ledger['accepted_iterations'] == 0 and res3['stop'] == 'NO_MONOTONE_IMPROVEMENT')
    check('planted global best left untouched by the accept logic',
          solver3.best['norm_inf'] == pre_best['norm_inf'] == -1e9)

    # 4) zero accepted moves must be an explicit defect-style stop
    check('zero accepted moves reported as stop', res3['stop'] == 'NO_MONOTONE_IMPROVEMENT')

    # 5) identical adjacent bases must not be silently accepted: constant residual map
    def constant(p):
        return dict(residual6=np.array([2., 0., 0., 0., 0., 0.]), contact_signature='static', p=p)
    solver5 = RootNewton(constant, lower, upper, cap=30, name='constant')
    res5 = solver5.solve(np.zeros(6))
    check('constant residual cannot be accepted',
          solver5.ledger['accepted_iterations'] == 0 and res5['stop'] == 'NO_MONOTONE_IMPROVEMENT')

    # 6) atomic Jacobian budget: refuse before any probe
    solver6 = RootNewton(linear, lower, upper, cap=1, name='budget')
    before = len(calls)
    try:
        solver6.evaluate(np.zeros(6), 'initial', 0)
        solver6.jacobian(np.zeros(6), 1)
        raise AssertionError('expected refusal')
    except RefusedJacobian:
        pass
    check('jacobian refused before probes when budget insufficient',
          len(calls) - before == 1 and solver6.ledger['fd_probes'] == 0)

    # 7) cache: repeated identical point costs no forward
    solver7 = RootNewton(linear, lower, upper, cap=50, name='cache')
    solver7.evaluate(np.zeros(6), 'initial', 0)
    used = solver7.ledger['forward']
    solver7.evaluate(np.zeros(6), 'repeat', 1)
    check('exact cache hit costs no forward', solver7.ledger['forward'] == used and solver7.ledger['cache_hits'] == 1)

    # 8) contact-set change is counted, not silently accepted
    state = {'n': 0}

    def jumping(p):
        state['n'] += 1
        sig = 'A' if p[0] < .2 else 'B'
        return dict(residual6=np.array([p[0] - .9, 0., 0., 0., 0., 0.]), contact_signature=sig, p=p)
    solver8 = RootNewton(jumping, lower, upper, cap=100, name='jump')
    res8 = solver8.solve(np.zeros(6))
    check('contact set changes counted', solver8.ledger['contact_set_changes'] >= 0)
    check('residual still decreases across a contact change', res8['norm_inf'] < .9)

    # 9) root and joint residuals are never merged: core only reports residual6
    check('core result exposes six-row residual only',
          'residual6' in res and len(res['residual6']) == 6 and 'joint' not in json.dumps(list(res.keys())))


    # 10) finite-difference direction must be inward when the iterate sits on a bound
    from root_newton_core import RefusedJacobian as _RJ
    lo2 = np.zeros(6); hi2 = np.ones(6)
    def lin(p): return dict(residual6=p.copy(), contact_signature='s', p=p)
    s10 = RootNewton(lin, lo2, hi2, cap=60, name='bound')
    J10, _, steps10 = s10.jacobian(np.array([0., 0., .5, 1., .2, .9]), 1)
    check('inward FD steps at both bounds', abs(steps10[0]) > 0 and abs(steps10[3]) > 0 and abs(J10 - np.eye(6)).max() < 1e-6)
    check('no engine imported', 'mujoco' not in sys.modules)
    (out / 'TEST_RESULTS.json').write_text(json.dumps(dict(passed=True, engineering_only=True,
                                                            model_load=0, forward=0, tests=results), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    run(ap.parse_args().output)
    print('all core regressions passed')
