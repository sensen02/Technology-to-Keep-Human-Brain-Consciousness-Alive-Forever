"""Saved-state center diagnostic. Stdlib-only without explicit parent execution flag."""
from pathlib import Path
import argparse, ast, builtins, dis, hashlib, json, os, resource, signal, struct, sys, time, types, zipfile

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unchanged(ledger):
    for path, digest in ledger.items():
        require(sha(path) == digest, 'Source drift: ' + path)


def fresh(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def headers(path, keys):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for key in keys:
            with archive.open(key + '.npy') as stream:
                require(stream.read(6) == b'\x93NUMPY', 'NPY magic')
                version = stream.read(2)
                require(version in (b'\x01\x00', b'\x02\x00', b'\x03\x00'), 'NPY version')
                size = 2 if version[0] == 1 else 4
                length = struct.unpack('<H' if size == 2 else '<I', stream.read(size))[0]
                require(length <= 65536, 'Bounded NPY header')
                header = ast.literal_eval(stream.read(length).decode('utf8' if version[0] == 3 else 'latin1'))
                require(header['descr'] in ('<f8', '<i8'), 'Simple saved numeric dtype')
                result[key] = header
    return result


def metadata():
    prereg = read(HERE / 'PREREG.json')
    base = Path(prereg['old_root'])
    ledger = {}
    cases = {}
    def bind(path, expected=None):
        path = str(path)
        digest = sha(path)
        require(expected is None or digest == expected, 'Archived hash mismatch: ' + path)
        require(path not in ledger or ledger[path] == digest, 'Conflicting hashes')
        ledger[path] = digest
    for name, binding in prereg['cases'].items():
        folder = base / binding['relative_folder']
        bind(folder / 'RESULT.json', binding['result_sha256'])
        result = read(folder / 'RESULT.json')
        audit = read(folder / 'VERIFY.json')
        identity = read(folder / 'IDENTITY.json')
        length = read(folder / 'BODY_LENGTH.json')
        require(result['condition'] == name and result['control'] == 'ALGORITHM_TEACHER_ONLY_NOT_ANN', 'Actual teacher case')
        require(audit['integrity_all_pass'] and all(audit['checks'].values()) and audit['result_sha256'] == binding['result_sha256'], 'Fresh saved audit')
        canonical = {k: v for k, v in identity.items() if k != 'identity_sha256'}
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        require(digest == identity['identity_sha256'] == result['identity_sha256'], 'Canonical compiled identity')
        require(result['body_length'] == length and length['length_native'] > 0 and length['absolute_SI'] == 'NOT_ESTABLISHED', 'Actual body length metadata')
        require(result['full_duration'] and result['actual_ms'] == 15400, 'Complete saved trajectory')
        for path, expected in result['source_hashes'].items():
            bind(path, expected)
        for path, info in identity['files'].items():
            bind(path, info['sha256'])
        for key in ('trace', 'geometry', 'teacher_samples', 'diagnostics'):
            bind(result[key]['path'], result[key]['sha256'])
        for filename in ('VERIFY.json', 'IDENTITY.json', 'BODY_LENGTH.json', 'runtime_geometry.npz', 'HANDOVER.json'):
            bind(folder / filename)
        th = headers(folder / 'trace_1ms.npz', ('time', 'qpos', 'com', 'control_source'))
        gh = headers(folder / 'geometry_1ms.npz', ('time', 'thorax_xyz', 'thorax_rotation', 'thorax_yaw', 'control_source'))
        require(th['time']['shape'] == (15400,) and th['qpos']['shape'] == (15400, 49) and th['com']['shape'] == (15400, 3), 'Actual trace headers')
        require(gh['time']['shape'] == (14400,) and gh['thorax_xyz']['shape'] == (14400, 3) and gh['thorax_rotation']['shape'] == (14400, 3, 3), 'Actual geometry headers')
        cases[name] = dict(folder=str(folder), result=result, body_length=length,
                           trace_headers=th, geometry_headers=gh,
                           compiled_identity_sha256=digest,
                           actual_XML_references=[p for p in identity['files'] if p.endswith('.xml')])
    for name in ('PREREG.json', 'analyze.py'):
        bind(HERE / name)
    unchanged(ledger)
    return ledger, cases


def globals_used(code):
    names = {x.argval for x in dis.get_instructions(code) if x.opname in ('LOAD_GLOBAL', 'LOAD_NAME')}
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(globals_used(constant))
    return names


def codecheck():
    tree = ast.parse((HERE / 'analyze.py').read_text())
    compile(tree, str(HERE / 'analyze.py'), 'exec')
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        require(checks[name], 'CODECHECK ' + name)
    check('stdlib_only_module_imports', not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(a.name.split('.')[0] in ('numpy', 'torch', 'mujoco', 'physics') for a in n.names) for n in tree.body))
    records = {}
    for name, function in list(globals().items()):
        if isinstance(function, types.FunctionType) and function.__module__ == __name__:
            names = globals_used(function.__code__)
            missing = sorted(n for n in names if n not in function.__globals__ and not hasattr(builtins, n))
            records[name] = dict(global_names=sorted(names), unresolved=missing)
            check(name + '_full_nested_globals', not missing)
    ledger, cases = metadata()
    check('all_three_actual_cases', set(cases) == {'left_pivot', 'right_pivot', 'stop'})
    check('all_source_hashes_and_header_metadata_match', bool(ledger))
    check('no_analysis_output_or_payload_execution', not (HERE / 'CENTER_ANALYSIS.json').exists())
    check('no_numeric_model_imports', not any(k in sys.modules for k in ('numpy', 'torch', 'mujoco', 'physics', 'student_wave')))
    fresh(HERE / 'CODECHECK.json', dict(all_passed=True, checks=checks, source_hashes=ledger,
          recursive_global_analysis=records,
          metadata={k: {n: v for n, v in c.items() if n != 'result'} for k, c in cases.items()},
          array_payload_reads=0, NumPy_imports=0, Core=0, physics_steps=0, NN_forward_calls=0,
          outcome='UNKNOWN_PENDING_PARENT_SAVED_ARRAY_EXECUTION'))
    print(json.dumps(dict(status='READY', checks=len(checks), source_hashes=len(ledger),
                          arrays=0, Core=0, NN=0, physics=0)))


def analyze_case(np, case):
    folder = Path(case['folder'])
    result = case['result']
    L = case['body_length']['length_native']
    with np.load(folder / 'trace_1ms.npz', allow_pickle=False) as archive:
        tt = archive['time']; root = archive['qpos'][:, :3].copy()
        com = archive['com']; source = archive['control_source']
    with np.load(folder / 'geometry_1ms.npz', allow_pickle=False) as archive:
        gt = archive['time']; thorax = archive['thorax_xyz']; rotation = archive['thorax_rotation']
        stored_yaw = archive['thorax_yaw']; gsource = archive['control_source']
    require(np.array_equal(gt, tt[1000:]), 'Full exact trace/geometry time alignment')
    require(all(np.isfinite(x).all() for x in (tt, root, com, thorax, rotation, stored_yaw)), 'Finite required arrays')
    boundary = result['candidate_start']['time']
    ti = int(np.argmin(np.abs(tt - boundary))); gi = int(np.argmin(np.abs(gt - boundary)))
    require(ti == 5799 and gi == 4799 and tt[ti] == gt[gi] == boundary, 'Exact saved 5.8s boundary')
    require(np.max(np.abs(np.diff(tt) - .001)) < 1e-8 and abs(tt[-1] - 15.4) < 1e-8, 'Saved actual clocks')
    require(np.array_equal(source[ti:], gsource[gi:]) and np.all(source[ti+1:] == 6), 'Candidate source labels')
    times = tt[ti:]; root = root[ti:]; com = com[ti:]; thorax = thorax[gi:]; rotation = rotation[gi:]; stored_yaw = stored_yaw[gi:]
    require(len(times) == 9601, 'Boundary plus full 9600 candidate samples')
    require(np.array_equal(com[0], np.asarray(result['candidate_start']['com'])), 'Actual COM boundary metadata')
    orth = float(np.max(np.abs(np.einsum('tji,tjk->tik', rotation, rotation) - np.eye(3))))
    require(orth < 1e-12 and np.max(np.abs(np.linalg.det(rotation) - 1.)) < 1e-12, 'Proper actual rotation matrices')
    yaw = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
    require(np.max(np.abs(np.arctan2(np.sin(yaw-stored_yaw), np.cos(yaw-stored_yaw)))) < 1e-10, 'Actual yaw matches saved')
    require(abs(yaw[0] - result['candidate_start']['yaw']) < 1e-10, 'Boundary yaw metadata')
    yaw_delta = float(yaw[-1] - yaw[0])
    require(abs(yaw_delta - result['candidate_metrics']['yaw_change_rad']) < 1e-10, 'Original yaw preserved')
    def stats(points):
        delta = points - points[0]
        norms = np.linalg.norm(delta, axis=1)
        path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        return dict(initial_native=points[0].tolist(), final_native=points[-1].tolist(),
                    final_displacement_native=delta[-1].tolist(),
                    final_displacement_bodylength=(delta[-1]/L).tolist(),
                    max_drift_native=float(norms.max()), max_drift_bodylength=float(norms.max()/L),
                    final_drift_native=float(norms[-1]), final_drift_bodylength=float(norms[-1]/L),
                    path_native=path, path_bodylength=path/L)
    def vector_summary(vectors):
        xy = np.linalg.norm(vectors[:, :2], axis=1)
        xyz = np.linalg.norm(vectors, axis=1)
        return dict(final_XYZ_native=vectors[-1].tolist(), final_XY_native=vectors[-1, :2].tolist(),
                    final_XY_bodylength=(vectors[-1, :2]/L).tolist(),
                    final_XY_norm_native=float(xy[-1]), final_XY_norm_bodylength=float(xy[-1]/L),
                    final_XYZ_norm_native=float(xyz[-1]), final_XYZ_norm_bodylength=float(xyz[-1]/L),
                    max_XY_norm_native=float(xy.max()), max_XY_norm_bodylength=float(xy.max()/L),
                    max_XYZ_norm_native=float(xyz.max()), max_XYZ_norm_bodylength=float(xyz.max()/L),
                    RMS_XY_native=float(np.sqrt(np.mean(xy**2))), RMS_XY_bodylength=float(np.sqrt(np.mean(xy**2))/L),
                    RMS_XYZ_native=float(np.sqrt(np.mean(xyz**2))), RMS_XYZ_bodylength=float(np.sqrt(np.mean(xyz**2))/L))
    rt = np.einsum('tji,tj->ti', rotation, com - thorax)
    r0 = rotation[0].T @ (com[0] - thorax[0])
    translation = thorax - thorax[0]
    rotation_term = np.einsum('tij,j->ti', rotation - rotation[0], r0)
    internal = np.einsum('tij,tj->ti', rotation, rt - r0)
    actual_delta = com - com[0]
    residual = translation + rotation_term + internal - actual_delta
    max_abs = float(np.max(np.abs(residual)))
    require(max_abs <= 1e-12, 'Exact vector decomposition residual <=1e-12')
    rotation_only_com = thorax[0] + np.einsum('tij,j->ti', rotation, r0)
    error = com - rotation_only_com
    require(float(np.max(np.abs(error - translation - internal))) <= 1e-12, 'Rotation-only error equals translation plus internal')
    metrics = dict(raw_free_root_XY=stats(root[:, :2]), raw_free_root_XYZ=stats(root),
                   Thorax_XY=stats(thorax[:, :2]), Thorax_XYZ=stats(thorax),
                   COM_XY=stats(com[:, :2]), COM_XYZ=stats(com))
    archived = result['candidate_metrics']
    require(abs(metrics['COM_XY']['max_drift_bodylength'] - archived['max_COM_xy_drift_bodylength']) < 1e-10, 'Original COM drift reproduced')
    require(abs(metrics['COM_XY']['path_bodylength'] - archived['COM_xy_path_bodylength']) < 1e-10, 'Original COM path reproduced')
    return dict(condition=result['condition'], control=result['control'],
                source_folder=str(folder), compiled_identity_sha256=case['compiled_identity_sha256'],
                actual_XML_references=case['actual_XML_references'], body_length_native=L, SI='NOT_ESTABLISHED',
                window=dict(first_time_s=float(times[0]), last_time_s=float(times[-1]), inclusive_rows=len(times),
                            candidate_rows=len(times)-1, actual_duration_s=float(times[-1]-times[0]),
                            trace_boundary_index=ti, geometry_boundary_index=gi),
                reference_point_statistics=metrics,
                yaw=dict(initial_rad=float(yaw[0]), final_rad=float(yaw[-1]), change_rad=yaw_delta,
                         change_deg=float(np.degrees(yaw_delta)), archived_change_deg=archived['yaw_change_deg']),
                decomposition=dict(initial_COM_offset_in_Thorax_frame_native=r0.tolist(),
                    internal_offset_change_native=vector_summary(rt-r0),
                    translation=vector_summary(translation), rotation=vector_summary(rotation_term),
                    internal=vector_summary(internal), actual_COM_delta=vector_summary(actual_delta),
                    final_vector_sum_XY_native=(translation[-1,:2]+rotation_term[-1,:2]+internal[-1,:2]).tolist(),
                    max_abs_component_residual_native=max_abs,
                    residual=vector_summary(residual), rotation_only_COM=stats(rotation_only_com),
                    actual_COM_minus_rotation_only_COM=vector_summary(error),
                    rotation_matrix_orthogonality_max_abs=orth),
                original_candidate_metrics_unchanged=archived,
                original_five_unchanged=result['original_five'],
                interpretation='Reference-frame vector identity only. No addition of component norms, rigid-whole-fly assumption, new gate, ANN success or biological claim.')


def execute():
    started = time.monotonic()
    def timeout(signum, frame):
        raise TimeoutError('Saved center analysis wall limit30s')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(30)
    require(not (HERE / 'CENTER_ANALYSIS.json').exists(), 'Exclusive new output required')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'LP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', key + '=1 required')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    proof = read(HERE / 'CODECHECK.json')
    require(proof['all_passed'] and all(proof['checks'].values()), 'ONLYCODE qualification')
    before = dict(proof['source_hashes'])
    before[str(HERE / 'CODECHECK.json')] = sha(HERE / 'CODECHECK.json')
    unchanged(before)
    # Recheck JSON qualifications and source/header bindings before loading any payload.
    current, cases = metadata()
    require(current == proof['source_hashes'], 'Exact registered source ledger')
    import numpy as np
    rows = {name: analyze_case(np, case) for name, case in cases.items()}
    after = {path: sha(path) for path in before}
    require(after == before, 'All sources unchanged after saved analysis')
    wall = time.monotonic() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    require(wall <= 30. and peak <= 256., 'Wall30s/RSS256MiB budget')
    require(not any(k in sys.modules for k in ('torch', 'mujoco', 'physics', 'student_wave', 'flygym')), 'No model or scientific runtime imports')
    fresh(HERE / 'CENTER_ANALYSIS.json', dict(kind='ACTUAL_SAVED_PIVOT_REFERENCE_CENTER_DIAGNOSTIC_V1',
          all_integrity_checks_passed=True, cases=rows, source_hashes_before=before,
          source_hashes_after=after, source_unchanged=True,
          wall_s=wall, peak_RSS_MiB=peak, physics_steps=0, NN_forward_calls=0,
          Core_constructions=0, render_calls=0, training_updates=0,
          verdict_scope='Saved teacher geometry interpretation only; historical gates unchanged; outcome interpreted by parent after actual execution.'))
    signal.alarm(0)
    print(str(HERE / 'CENTER_ANALYSIS.json'))


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check-code-only', action='store_true')
    modes.add_argument('--execute-parent-saved-center-analysis', action='store_true')
    args = parser.parse_args()
    if args.check_code_only:
        codecheck()
    else:
        execute()


if __name__ == '__main__':
    main()
