"""S166 saved-only independent verifier; never imports engine/producer/neural code.

Only saved binary64 arrays are certified, in native units. A feasible rigid
contact relaxation is NOT dynamic standing, neural attainability or SI validation.
Invoke only after the producer has finished: --output /absolute/new/run/root.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import ast
import csv
import hashlib
import json
import numpy as np
from scipy.optimize import linprog

S165 = Path('/home/sensen/Desktop/engine修复验收/s165-static-support-corrected/20261024T231400000000Z')
CSV = Path('/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_90_MUSCLE_MAPPING.csv')
HELPERS = Path(__file__).with_name('newverify_static_support_corrected.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def close(a, b, label, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), label
    error = float(np.max(np.abs(a-b), initial=0.))
    assert error <= tol, (label, error, tol)
    return error


def helpers():
    # Reuse only four audited pure numeric functions, not module imports/main.
    # The input file and its identity are included in this verifier's manifest.
    tree = ast.parse(HELPERS.read_text())
    names = {'exact_dot', 'separation', 'scalar_rate', 'muscle'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for f in nodes for n in ast.walk(f))
    namespace = {'np': np, 'Fraction': Fraction, 'linprog': linprog}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HELPERS), 'exec'), namespace)
    return namespace


def mapping():
    with CSV.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 90
    rows.sort(key=lambda r: int(r['actuator_index']))
    assert [int(r['actuator_index']) for r in rows] == list(range(90))
    members = [tuple(sorted(map(int, r['local_indices'].split(';')))) if r['local_indices'] else () for r in rows]
    assert all(len(x) == len(set(x)) for x in members)
    unique = sorted(set(x for x in members if x))
    assert len(unique) == 66
    index = {x: j for j, x in enumerate(unique)}
    pool = np.array([index[x] if x else -1 for x in members], dtype=int)
    assert np.array_equal(pool, [int(r['pool_id']) for r in rows])
    assert sum(pool >= 0) == 74 and sum(pool < 0) == 16
    T = np.zeros((90, 66))
    for j, p in enumerate(pool):
        assert rows[j]['mapping_status'] == ('MAPPED' if p >= 0 else 'UNMAPPED_NO_TIER_CELLS')
        if p >= 0:
            T[j, p] = .5
    return T, pool, unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    out = parser.parse_args().output.resolve()
    assert out.is_dir()
    products = [out/'INDEPENDENT_CHECK.json', out/'CERTIFICATES.npz']
    assert not any(p.exists() for p in products), 'Refuse to overwrite verification evidence'
    prereg = json.loads((S165/'PREREG.json').read_text())
    source = Path(prereg['source'])/'init80ms'
    inputs = [Path(__file__).resolve(), HELPERS, CSV, S165/'PREREG.json',
              S165/'AFFINITY_PROOF.json', source/'NATIVE_MODEL_ARRAYS.npz',
              source/'INITIALIZATION.npz', source/'TRAJECTORY.npz',
              out/'MAPPING.npz', out/'SUMMARY.json']
    for i in (0, 15, 83):
        inputs.extend([S165/f'STATIC_AFFINE_{i:03d}.npz', S165/f'ZERO_V_{i:03d}.npz', out/f'STATE_{i:03d}.npz'])
    before = {str(p): sha(p) for p in inputs}
    h = helpers()
    T, pool, unique = mapping()
    m = load(out/'MAPPING.npz')
    close(m['T'], T, 'independent CSV T', 0.)
    assert np.array_equal(m['actuator_pool'], pool)
    summary = json.loads((out/'SUMMARY.json').read_text())
    native = load(source/'NATIVE_MODEL_ARRAYS.npz')
    initial = load(source/'INITIALIZATION.npz')
    trajectory = load(source/'TRAJECTORY.npz')
    assert np.array_equal(trajectory['actuator_pool'], pool), 'S154 actual mapping mismatch'
    proof = json.loads((S165/'AFFINITY_PROOF.json').read_text())
    for k in ('compiled_gain_muscle', 'compiled_bias_muscle', 'compiled_dyn_muscle', 'no_force_limits'):
        assert proof[k] is True, k
    for k in ('gainprm', 'biasprm'):
        close(proof[k], native['actuator_'+k], 'saved native '+k, 0.)
    dyn = np.asarray(proof['dynprm'])
    expected_dyn = native['actuator_dynprm'].copy()
    expected_dyn[:, 0] = .001
    close(dyn, expected_dyn, 'tau-only patch', 0.)
    assert np.all(dyn[:, 1] == .0004)
    assert np.array_equal(proof['actadr'], np.arange(90))
    assert np.array_equal(native['actuator_actadr'], np.arange(90))
    assert np.all(native['actuator_actnum'] == 1)
    assert np.all(native['actuator_gaintype'] == 2) and np.all(native['actuator_biastype'] == 2) and np.all(native['actuator_dyntype'] == 4)
    ranges = np.asarray(proof['ctrlrange'])
    assert ranges.shape == (90, 2) and np.all(ranges[:, 0] == 0) and np.all(ranges[:, 1] == 1)
    assert np.all(proof['ctrllimited'])
    for k in ('qpos', 'qvel', 'act', 'ctrl'):
        close(initial['body_'+k], trajectory[k][0], 'S154 initial '+k, 0.)
    certificates = {'T': T, 'actuator_pool': pool}
    results = []
    expected_bounds = np.column_stack((np.zeros(138), np.r_[np.ones(66), np.full(72, np.inf)]))
    for i in (0, 15, 83):
        s = load(S165/f'ZERO_V_{i:03d}.npz')
        saved = load(S165/f'STATIC_AFFINE_{i:03d}.npz')
        z = load(out/f'STATE_{i:03d}.npz')
        checks = {}
        for k in ('qpos', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied'):
            checks['S154_'+k] = close(s[k], trajectory[k][i], 'S154 '+k, 0.)
        assert not s['qvel'].any() and not s['actuator_velocity'].any()
        assert not s['qfrc_applied'].any() and not s['xfrc_applied'].any()
        close(s['qacc_warmstart_input'], trajectory['native_qacc_warmstart'][i], 'warmstart', 0.)
        gain = []; bias = []
        for j in range(90):
            _, g, b = h['muscle'](s['actuator_length'][j], 0., 0., native['actuator_gainprm'][j], native['actuator_biasprm'][j], native['actuator_lengthrange'][j])
            gain.append(g); bias.append(b)
        gain, bias = np.array(gain), np.array(bias)
        moment, J = s['actuator_moment'], s['efc_J_dense']
        assert moment.shape == (90, 48) and J.shape == (72, 48)
        assert np.all(s['efc_type'] == 6) and np.all(s['contact_dim'] == 3)
        assert len(s['contact_dim']) == 18 and np.all(s['contact_row_count'] == 4)
        assert np.all(s['contact_friction'][:, :2] > 0)
        assert np.array_equal(np.sort(np.concatenate([np.arange(int(a), int(a)+4) for a in s['contact_efc_address']])), np.arange(72))
        B = moment.T*gain[None, :]
        base = moment.T@bias+s['qfrc_passive']-s['qfrc_bias']
        for k, val in dict(B=B, base=base, gain=gain, bias=bias, J=J, moment=moment).items():
            checks[k] = close(saved[k], val, 'independent affine '+k)
        close(gain*s['act']+bias, s['actuator_force'], 'actual force')
        close(moment.T@(gain*s['act']+bias), s['qfrc_actuator'], 'actual torque')
        A = np.column_stack((B@T, J.T)); b = -base
        assert A.shape == (48, 138)
        checks['Aeq'] = close(z['Aeq'], A, 'Aeq', 0.)
        checks['beq'] = close(z['beq'], b, 'beq', 0.)
        assert z['bounds'].shape == (138, 2) and np.array_equal(z['bounds'], expected_bounds)
        for k, shape in [('x', (138,)), ('residual', (48,)), ('activation', (90,)), ('pools', (66,)), ('lambda', (72,))]:
            assert z[k].shape == shape, (k, z[k].shape)
        entry = dict(sample_index=i, checks=checks)
        if np.isfinite(z['x']).all():
            x = z['x']
            assert np.all(x >= 0.) and np.all(x[:66] <= 1.), 'strict saved bounds'
            checks['pools'] = close(z['pools'], x[:66], 'pool solution', 0.)
            checks['lambda'] = close(z['lambda'], x[66:], 'lambda solution', 0.)
            activation = T@x[:66]
            checks['activation'] = close(z['activation'], activation, 'mapped activation', 0.)
            assert np.all(activation >= 0) and np.all(activation <= .5) and not activation[pool < 0].any()
            residual = A@x-b
            checks['residual_saved'] = close(z['residual'], residual, 'saved residual', 0.)
            checks['residual48_maxabs'] = close(residual, np.zeros(48), 'all48 primal feasibility', 1e-8)
            # Actual control clamps, then scalar activation rate: exact zero.
            ctrl = np.where(np.asarray(proof['ctrllimited']), np.clip(activation, ranges[:, 0], ranges[:, 1]), activation)
            assert np.array_equal(ctrl, activation)
            rates = np.array([h['scalar_rate'](float(activation[j]), float(ctrl[j]), dyn[j]) for j in range(90)])
            assert np.all(rates == 0.), 'ctrl=act must give exact zero activation derivative'
            certificates[f'state_{i:03d}_primal'] = x
            entry.update(result='feasible', activation_rate_exact_zero=True,
                         interpretation='Validated saved rigid-contact primal, tolerance 1e-8 native; not dynamic/neural attainability')
        else:
            assert all(np.isnan(z[k]).all() for k in ('x', 'residual', 'activation', 'pools', 'lambda')), 'infeasible schema must be all NaN'
            y, cert = h['separation'](A, b, expected_bounds[:, 0], expected_bounds[:, 1])
            if y is not None:
                certificates[f'state_{i:03d}_separator'] = y
            entry.update(result='certified' if cert['validated'] else 'numeric', certificate=cert)
        results.append(entry)
    after = {str(p): sha(p) for p in inputs}
    assert before == after, 'inputs changed during independent verification'
    report = dict(status='SAVED_ONLY_INDEPENDENT_CHECK_COMPLETE', results=results,
                  producer_summary=summary, producer_status_used_as_proof=False,
                  scope='S166/P1/S2 fixed saved posture mapped rigid-contact relaxation only',
                  new_engine_model_forward_step_neural_calls=0, units='native; SI NOT_ESTABLISHED',
                  input_sha256=before, input_sha256_after=after,
                  pool_tuples=[list(t) for t in unique],
                  verification='Primal residual and bounds or validated exact-rational separation margin only; no solver status establishes infeasibility')
    np.savez_compressed(products[1], **certificates)
    report['certificates_sha256'] = sha(products[1])
    products[0].write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(out), 'results': [{k: r[k] for k in ('sample_index', 'result')} for r in results]}, indent=2))


if __name__ == '__main__':
    main()
