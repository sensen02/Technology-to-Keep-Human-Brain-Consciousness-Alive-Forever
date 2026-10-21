"""Offline independent NumPy matrix oracle for S2/P2 saved endpoints.

No runner/tool/MuJoCo import, model load, forward, or trajectory execution.
A PASS means archive/schema/arithmetic consistency, NOT solver convergence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


ENGINEERING_RTOL = 1e-11
ENGINEERING_ATOL = 1e-10  # Independent float64 arithmetic ONLY, never solver acceptance.
ENDPOINTS = 400


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def offsets(a, name, total, counts=None):
    x = np.asarray(a[name])
    require(x.dtype.kind in 'iu' and x.shape == (ENDPOINTS + 1,), name + ': integer shape')
    require(x[0] == 0 and x[-1] == total and np.all(x[1:] >= x[:-1]), name + ': range/order')
    if counts is not None:
        require(np.array_equal(np.diff(x), counts), name + ': counts mismatch')
    return x


def summary(x):
    """Empty selections are explicitly undefined, not a zero residual PASS."""
    x = np.asarray(x, dtype=np.float64)
    return dict(count=int(x.size), max_abs=float(np.max(np.abs(x))) if x.size else None,
                rms=float(np.sqrt(np.mean(x * x))) if x.size else None)


def histogram(x):
    values, counts = np.unique(x, return_counts=True)
    return {str(int(v)): int(n) for v, n in zip(values, counts)}


def independent_endpoint(a, i):
    """Decode endpoint-local J into a bounded dense matrix; compute using @."""
    elo, ehi = map(int, a['efc_offsets'][i:i+2])
    jlo, jhi = map(int, a['J_offsets'][i:i+2])
    nv = a['qacc'].shape[1]
    nefc = ehi - elo
    require(0 <= nefc <= 4096 and 0 < nv <= 256 and jhi-jlo <= 1048576,
            f'endpoint {i}: matrix bounds')
    values = np.asarray(a['J_values'][jlo:jhi], dtype=np.float64)
    if bool(a['sparse'][i]):
        nnz = a['J_rownnz'][elo:ehi]
        adr = a['J_rowadr'][elo:ehi]
        cols = a['J_colind'][jlo:jhi]
        require(nnz.dtype.kind in 'iu' and adr.dtype.kind in 'iu' and cols.dtype.kind in 'iu',
                f'endpoint {i}: sparse integer metadata')
        require(len(nnz) == nefc and len(adr) == nefc and len(cols) == len(values),
                f'endpoint {i}: sparse lengths')
        require(np.sum(nnz, dtype=np.int64) == int(a['nJ'][i]), f'endpoint {i}: sparse nnz total')
        J = np.zeros((nefc, nv), dtype=np.float64)
        used = np.zeros(len(values), dtype=np.bool_)
        for row in range(nefc):
            start, count = int(adr[row]), int(nnz[row])
            require(start >= 0 and count >= 0 and start + count <= len(values),
                    f'endpoint {i} row {row}: sparse local bounds')
            c = cols[start:start+count]
            require(np.all((c >= 0) & (c < nv)) and np.all(c[1:] > c[:-1]),
                    f'endpoint {i} row {row}: columns unsorted/duplicate/out of range')
            require(not np.any(used[start:start+count]), f'endpoint {i}: overlapping sparse rows')
            used[start:start+count] = True
            J[row, c] = values[start:start+count]
        # Storage padding is allowed; nJ counts referenced nonzeros, not capacity.
    else:
        require(len(values) == nefc * nv, f'endpoint {i}: dense J size')
        # Dense row metadata exists in bindings but has NO sparse semantics.
        J = values.reshape(nefc, nv)
    if 'nJ' in a and not bool(a['sparse'][i]):
        require(int(a['nJ'][i]) == len(values), f'endpoint {i}: dense nJ mismatch')
    f = a['efc_force'][elo:ehi]
    D = a['efc_D'][elo:ehi]
    aref = a['efc_aref'][elo:ehi]
    state = a['efc_state'][elo:ehi]
    types = a['efc_type'][elo:ehi]
    r = J @ a['qacc'][i] - aref
    b_e = J @ a['qacc_smooth'][i] - aref - a['efc_b'][elo:ehi]
    quadratic = state == 1
    e = f[quadratic] + D[quadratic] * r[quadratic]
    denom = np.maximum(np.abs(f[quadratic]), np.abs(D[quadratic] * r[quadratic]))
    relative = np.divide(np.abs(e), denom, out=np.full(e.shape, np.nan), where=denom != 0)
    map_e = a['qfrc_constraint'][i] - J.T @ f
    return dict(r=r, quadratic_e=e, relative_e=relative, map_e=map_e, b_e=b_e,
                quadratic=quadratic, state=state, types=types)


def validate_archive(a, record):
    for key in a.files:
        x = a[key]  # allow_pickle=False rejects even unused object members.
        require(x.dtype.kind in 'biuf', key + ': nonnumeric dtype')
        require(np.all(np.isfinite(x)), key + ': nonfinite')
    require(a['qacc'].ndim == 2 and a['qacc'].shape[0] == ENDPOINTS, 'qacc: endpoint shape')
    nv = a['qacc'].shape[1]
    require(0 < nv <= 256, 'nv: hard bound')
    for key in ('qacc_smooth', 'qfrc_constraint', 'qfrc_smooth'):
        require(a[key].shape == (ENDPOINTS, nv), key + ': DOF shape')
    for key in ('nefc', 'nJ', 'ncon', 'sparse'):
        require(a[key].shape == (ENDPOINTS,), key + ': endpoint shape')
    require(a['sparse'].dtype.kind == 'b', 'sparse: must be runtime bool')
    require(a['nefc'].dtype.kind in 'iu' and np.all((a['nefc'] >= 0) & (a['nefc'] <= 4096)), 'nefc bound')
    require(a['ncon'].dtype.kind in 'iu' and np.all((a['ncon'] >= 0) & (a['ncon'] <= 512)), 'ncon bound')
    total = len(a['efc_force'])
    offsets(a, 'efc_offsets', total, a['nefc'])
    offsets(a, 'J_offsets', len(a['J_values']))
    require(np.all(a['nJ'] >= 0), 'nJ: negative')
    for key in ('efc_aref', 'efc_D', 'efc_R', 'efc_force', 'efc_state', 'efc_type', 'efc_id', 'efc_b'):
        require(a[key].shape == (total,), key + ': ragged row shape')
    for key in ('efc_state', 'efc_type', 'efc_id'):
        require(a[key].dtype.kind in 'iu', key + ': integer dtype')
    require(a['J_values'].ndim == 1 and a['J_colind'].shape == a['J_values'].shape, 'J storage shape')
    require(a['J_rownnz'].shape == (total,) and a['J_rowadr'].shape == (total,), 'J row metadata shape')
    offsets(a, 'contact_offsets', len(a['contact_dim']), a['ncon'])
    for key in a.files:
        if key.startswith('contact_') and key != 'contact_offsets':
            require(a[key].shape[0] == len(a['contact_dim']), key + ': contact ragged shape')
    require(np.allclose(a['time_s'], np.arange(1, ENDPOINTS+1)*25e-6, rtol=0, atol=1e-14)
            and a['time_s'].shape == (ENDPOINTS,), 'endpoint times')
    dt = float(record['dt_s'])
    require(dt in (25e-6, 12.5e-6), 'unsupported body clock')
    steps = round(.01 / dt)
    require(a['step_ctrl'].ndim == 2 and len(a['step_ctrl']) == steps, 'step control shape')
    expected = np.zeros_like(a['step_ctrl'])
    expected[:round(.001/dt), 0] = .1
    require(np.array_equal(a['step_ctrl'], expected), 'fixed pulse mismatch')
    require(a['step_time_s'].shape == (steps,) and np.allclose(a['step_time_s'], np.arange(steps)*dt,
            rtol=0, atol=1e-14), 'step times')
    for key in ('initial_qvel', 'initial_act', 'initial_ctrl'):
        require(np.count_nonzero(a[key]) == 0, key + ': not zero')
    require(a['initial_qvel'].shape == (nv,), 'initial qvel shape')
    warning_keys = [k for k in a.files if 'warning' in k and 'count' in k]
    require(bool(warning_keys), 'missing warning counts')
    for key in warning_keys:
        require(np.count_nonzero(a[key]) == 0, key + ': warning occurred')
    require(sum(a[k].nbytes for k in a.files) <= 256*1024*1024, 'archive numeric byte budget')


def metric_stats(values, relative=None):
    x = np.asarray(values, dtype=np.float64)
    s = summary(x)
    result = dict(count=s['count'], absmax=s['max_abs'], rms=s['rms'])
    if relative is not None:
        relative = np.asarray(relative)
        defined = np.isfinite(relative)
        result.update(relative_undefined=int(np.sum(~defined)),
                      relative_absmax=float(np.max(relative[defined])) if np.any(defined) else None)
    return result


def independent_case(a, record):
    endpoints = [independent_endpoint(a, i) for i in range(ENDPOINTS)]
    cat = lambda key: np.concatenate([z[key] for z in endpoints])
    mapping = np.stack([z['map_e'] for z in endpoints])
    per_type = {}
    for t in np.unique(cat('types')):
        es, rs = [], []
        for z in endpoints:
            select = z['types'][z['quadratic']] == t
            es.extend(z['quadratic_e'][select])
            rs.extend(z['relative_e'][select])
        per_type[str(int(t))] = metric_stats(es, rs)
    return dict(quadratic=metric_stats(cat('quadratic_e'), cat('relative_e')),
                nonquadratic_count=int(np.sum(~cat('quadratic'))),
                state_histogram=histogram(cat('state')), per_type_quadratic=per_type,
                r_reference_deviation=metric_stats(cat('r')), b_consistency=metric_stats(cat('b_e')),
                mapping_per_dof=[dict(dof=j, unit_type=record['dof_types'][j],
                                      **metric_stats(mapping[:, j])) for j in range(mapping.shape[1])])


def report_case(report, record):
    require(report['policy'] == 'REPORT_ONLY', 'report must not claim solver PASS')
    matches = [c for c in report['cases'] if c['npz'] == record['npz']]
    require(len(matches) == 1, 'missing/duplicate report case')
    return matches[0]


def compare_tree(actual, expected, label, differences):
    """Compare JSON metrics recursively, with explicit null handling."""
    if isinstance(actual, dict):
        require(isinstance(expected, dict), label + ': expected object')
        for key, value in actual.items():
            require(key in expected, label + ': missing metric ' + key)
            compare_tree(value, expected[key], label + '.' + key, differences)
    elif isinstance(actual, list):
        require(isinstance(expected, list) and len(actual) == len(expected), label + ': series length')
        for i, (x, y) in enumerate(zip(actual, expected)):
            compare_tree(x, y, f'{label}[{i}]', differences)
    elif isinstance(actual, str):
        require(actual == expected, label + ': label mismatch')
    elif actual is None:
        require(expected is None, label + ': undefined must remain null')
    elif isinstance(actual, (int, bool)):
        require(actual == expected, label + ': count mismatch')
    else:
        require(isinstance(expected, (int, float)) and np.isfinite(expected), label + ': finite scalar required')
        delta = abs(actual - expected)
        differences.append(delta)
        require(np.isclose(actual, expected, rtol=ENGINEERING_RTOL, atol=ENGINEERING_ATOL),
                label + ': independent arithmetic mismatch')


def verify(out):
    """Read archives only, emit independent-check.json even on validation failure."""
    out = Path(out).resolve()
    result = dict(passed=False, scope='S2/P2 offline archive and independent float64 arithmetic only',
                  solver_residual_policy='REPORT_ONLY; no threshold on real r/e/map/b values',
                  fullKKT='unknown: AR/admissibility/cone/complementarity not constructed or verified',
                  stationarity='unknown: no archived mass matrix stationarity computation',
                  massbalance='unknown: no mass matrix/smooth solve/force-balance verification',
                  physical_cause='unknown', convergence='not established',
                  engineering_comparison=dict(rtol=ENGINEERING_RTOL, atol=ENGINEERING_ATOL), cases=[])
    archive_hashes = {}
    try:
        manifest = json.loads((out/'manifest.json').read_text())
        report = json.loads((out/'report.json').read_text())
        records = manifest['records']
        require(len(records) == 2, 'exactly two cases required')
        require({float(r['dt_s']) for r in records} == {25e-6, 12.5e-6}, 'two body clocks required')
        before, after = manifest['source_hashes_before'], manifest['source_hashes_after']
        require(bool(before) and before == after, 'manifest source hashes before/after mismatch')
        for path, digest in before.items():
            require(sha256(path) == digest, 'current source hash mismatch: ' + path)
        paths = [out/r['npz'] for r in records]
        require(len(set(paths)) == 2, 'duplicate archives')
        archive_hashes = {str(p): sha256(p) for p in paths}
        for rec, p in zip(records, paths):
            require(archive_hashes[str(p)] == rec['sha256'], 'manifest archive hash mismatch: ' + str(p))
        common_initial = None
        differences = []
        for record, path in zip(records, paths):
            with np.load(path, allow_pickle=False) as a:
                validate_archive(a, record)
                initial = {k: a[k].copy() for k in ('initial_qpos', 'initial_qvel', 'initial_act', 'initial_ctrl')}
                if common_initial is None:
                    common_initial = initial
                else:
                    require(all(np.array_equal(initial[k], common_initial[k]) for k in initial), 'initial states differ')
                require(int(record['nv']) == a['qacc'].shape[1], 'record nv mismatch')
                require(len(record['dof_types']) == int(record['nv']), 'DOF unit labels missing')
                options = record['options']
                require(int(options['solver']) in (1, 2) and int(options['noslip_iterations']) == 0,
                        'unsupported constitutive branch: requires primal and noslip0')
                case = independent_case(a, record)
                compare_tree(case, report_case(report, record), str(path), differences)
                result['cases'].append(dict(npz=str(path), endpoints=ENDPOINTS, metrics=case))
        for path, digest in before.items():
            require(sha256(path) == digest, 'source changed during verification: ' + path)
        require(all(sha256(p) == h for p, h in archive_hashes.items()), 'archives changed during verification')
        result.update(passed=True, source_hash_count=len(before), archives_unchanged=True,
                      scalar_comparisons=len(differences), comparison_max_abs=max(differences, default=0.),
                      archive_hashes=archive_hashes)
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['archives_unchanged'] = all(Path(p).exists() and sha256(p) == h for p, h in archive_hashes.items())
    (out/'independent-check.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path, help='Existing two-case archive directory; no dynamics are executed')
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.out)['passed'] else 1)
