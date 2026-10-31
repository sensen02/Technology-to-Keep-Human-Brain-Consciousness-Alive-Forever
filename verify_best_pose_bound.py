"""S174/P1-S2 saved-only independent fixed S173-best pose box bound.
Never imports a producer, optimizer or physical engine; never searches for y.
Run only after producer closure: --output /absolute/S174/output.
An infeasible phase-I point is NOT an equilibrium witness. Scope is one pose.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import ast
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np

BASE = Path('/home/sensen/Desktop/engine修复验收')
SOURCE = BASE/'s173-accounted-near-initial-search/20261024T235544196037334Z/RUN'
S154 = BASE/'s154-activation-rate/20261024T215751775489Z/init80ms'
S165 = BASE/'s165-static-support-corrected/20261024T231400000000Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z/MAPPING.npz'
CSV = BASE/'s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_90_MUSCLE_MAPPING.csv'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def read_json(p):
    return json.loads(Path(p).read_text())


def pure_helpers():
    """Pinned numeric definitions only, not top-level code or module imports."""
    ns = dict(np=np, Fraction=Fraction, CSV=CSV, S166=S166, csv=csv)
    specifications = [
        ('verify_fixed_pose_soft.py', '663595930d80a51859b83c0fe3c5a805b5f1a04103f05e33810ce2c827641286',
         {'load', 'close', 'bitwise', 'muscle', 'mapping', 'certificate'}),
        ('verify_near_initial_support.py', '0d78f0d1e4bbf5b829761e921ed8c4267e7af6aeb6ecfdfa664025194a2a8444',
         {'contact_check'}),
    ]
    for filename, digest, names in specifications:
        p = HERE/filename
        assert sha(p) == digest, ('pure-helper identity changed', str(p))
        tree = ast.parse(p.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == names
        assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for f in nodes for n in ast.walk(f))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), ns)
    return ns


def validate_source(h):
    close, bitwise = h['close'], h['bitwise']
    summary = read_json(SOURCE/'SUMMARY.json')
    report = read_json(SOURCE/'S173_SAVED_INDEPENDENT_CHECK.json')
    records = read_json(SOURCE/'EVALUATIONS.json')
    frozen = load(SOURCE/'FROZEN_SEARCH.npz')
    assert report['verified'] and report['input_hashes_unchanged'] and report['output_hashes_unchanged']
    assert report['best_evaluation'] == summary['best_diagnostic']['evaluation'] == 303
    assert len(records) == len(report['evaluations']) == 890
    assert report['classification'] == 'NUMERICAL_BOUNDED_SEARCH_FAILED_NOT_IMPOSSIBILITY'
    assert summary['accepted'] is None and summary['status'] == 'SEARCH_FAILED'
    assert all(v == 0 for v in report['actual_verifier_calls'].values())
    cap = float(frozen['original_foot_depth'])
    for i, (r, verified) in enumerate(zip(records, report['evaluations']), 1):
        assert r['evaluation'] == verified['evaluation'] == i
        geom = bool(r['foot_depth'] <= cap+1e-12 and r['nonfoot_depth'] <= 1e-12
                    and r['contact_legs'] >= 3 and r['friction_margin'] >= -1e-10 and r['control_ok'])
        assert r['geometry_ok'] == geom
        verified_contact = verified['contact']
        verified_depth = max((c['penetration'] for c in verified_contact['contacts'] if c['foot']), default=0.)
        close(r['foot_depth'], verified_depth, 'rank geometry independent report depth', 1e-12)
        close(r['nonfoot_depth'], verified_contact['max_nonfoot_penetration'], 'rank geometry independent report nonfoot', 1e-12)
        close(r['max_abs_native_residual'], verified['max_abs_residual'], 'record/report residual', 1e-9)
        close(r['objective'], verified['objective'], 'record/report objective', 1e-8)
        assert not r['accepted'] and not verified['accepted']
    best = min(records, key=lambda r: (not r['geometry_ok'], r['max_abs_native_residual'], r['objective']))
    assert best['evaluation'] == 303
    assert sha(SOURCE/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz') == sha(SOURCE/'EVAL_303.npz')
    s = load(SOURCE/'EVAL_303.npz')
    probe = load(SOURCE/'FORWARD_606_probe_forward.npz')
    actual = load(SOURCE/'FORWARD_607_optimized_state_forward.npz')
    native = load(S154/'NATIVE_MODEL_ARRAYS.npz')
    meta = read_json(S154/'COM_CONTACT_METADATA.json')
    T, missing = h['mapping']()
    bitwise(s['T'], T, 'best actual full CSV mapping')
    bitwise(frozen['T'], T, 'frozen actual mapping')
    initial = load(S154/'INITIALIZATION.npz')
    q0 = initial['body_qpos']
    bitwise(frozen['qpos0'], q0, 'original S154 pose')
    prereg = read_json(SOURCE/'PREREG.json')
    joints = list(ET.parse(prereg['model']['xml']).getroot().find('worldbody').iter('joint'))
    assert len(joints) == 43 and joints[0].get('type') == 'free'
    ranges = np.array([list(map(float, j.get('range').split())) for j in joints[1:]])
    radius = np.minimum(.15, .1*np.diff(ranges, axis=1).ravel())
    lower = np.r_[.95*q0[2], np.maximum(q0[7:]-radius, ranges[:, 0])]
    upper = np.r_[1.15*q0[2], np.minimum(q0[7:]+radius, ranges[:, 1])]
    indices = np.r_[2, np.arange(7, 49)]
    assert np.array_equal(indices, frozen['indices'])
    bitwise(frozen['joint_ranges'], ranges, 'native joint ranges')
    bitwise(frozen['lower'], lower, 'frozen lower'); bitwise(frozen['upper'], upper, 'frozen upper')
    assert np.all(s['qpos'][indices] >= lower) and np.all(s['qpos'][indices] <= upper)
    fixed = np.setdiff1d(np.arange(49), indices)
    bitwise(s['qpos'][fixed], q0[fixed], 'unchanged root xy/orientation')
    assert np.all(s['qpos'][7:] >= ranges[:, 0]) and np.all(s['qpos'][7:] <= ranges[:, 1])
    original = load(S165/'ZERO_V_000.npz')
    originalcap = float(np.maximum(-original['contact_dist'], 0).max(initial=0))
    assert cap == originalcap == prereg['accept']['foot_depth_cap']
    assert int(s['probe_forward_id']) == 606 and int(s['actual_forward_id']) == 607
    for state in (probe, actual):
        bitwise(state['qpos'], s['qpos'], 'fixed saved pose')
        for k in ('qvel', 'actuator_velocity', 'qacc_warmstart_input', 'qfrc_applied', 'xfrc_applied'):
            assert not np.any(state[k]), k
    assert not np.any(probe['act']) and not np.any(probe['ctrl'])
    assert np.all(native['actuator_gaintype'] == 2) and np.all(native['actuator_biastype'] == 2)
    assert np.all(native['actuator_dyntype'] == 4)
    assert np.array_equal(native['actuator_actadr'], np.arange(90)) and np.all(native['actuator_actnum'] == 1)
    dyn = native['actuator_dynprm']
    assert np.all(dyn[:, 0] > 0) and np.all(dyn[:, 1] > 0)
    assert np.all(dyn[:, 0] == .001) and np.all(dyn[:, 1] == .0004)
    gain, bias = np.array([h['muscle'](probe['actuator_length'][j], native['actuator_gainprm'][j],
                                      native['actuator_biasprm'][j], native['actuator_lengthrange'][j])
                           for j in range(90)]).T
    M = probe['actuator_moment']; assert M.shape == (90, 48)
    B = M.T*gain[None, :]
    base = M.T@bias+probe['qfrc_passive']-probe['qfrc_bias']
    close(B, s['B'], 'independent scalar B'); close(base, s['base'], 'independent scalar base')
    close(gain, s['gain'], 'scalar gain'); close(bias, s['bias'], 'scalar bias')
    close(probe['actuator_force'], bias, 'zero-act native bias')
    close(probe['qfrc_actuator'], M.T@bias, 'zero-act native torque')
    D, aref = probe['efc_D'], probe['efc_aref']
    assert np.isfinite(D).all() and np.all(D > 0) and np.isfinite(aref).all()
    f0 = np.where(probe['efc_type'] == 0, D*aref, np.maximum(D*aref, 0))
    bitwise(f0, s['f0'], 'native fixed f0 law')
    bitwise(probe['efc_J_dense'], s['J'], 'native all-row J')
    contact = h['contact_check'](probe, f0, meta)
    footdepth = max((c['penetration'] for c in contact['contacts'] if c['foot']), default=0.)
    bearing = set()
    for c in contact['contacts']:
        ci = c['contact']; addr = int(probe['contact_efc_address'][ci]); count = int(probe['contact_row_count'][ci])
        if c['foot'] and c['signed_dist'] <= 0 and addr >= 0 and f0[addr:addr+count].sum() > 0:
            bearing.add(c['leg'])
    margin = min((c['pyramid_margin'] for c in contact['contacts']
                  if int(probe['contact_efc_address'][c['contact']]) >= 0), default=0.)
    assert footdepth <= cap+1e-12 and contact['max_nonfoot_penetration'] <= 1e-12
    assert len(bearing) >= 3 and margin >= -1e-10
    close(footdepth, best['foot_depth'], 'best foot depth'); close(margin, best['friction_margin'], 'best friction')
    assert len(bearing) == best['contact_legs']
    activation = T@s['pools']
    bitwise(actual['act'], activation, 'best saved mapped activation')
    bitwise(actual['ctrl'], activation, 'best ctrl=act')
    assert not actual['act_dot'].any() and np.all(activation >= 0) and np.all(activation <= .5)
    residual = s['B']@activation+s['base']+s['J'].T@f0
    close(residual, s['residual48'], 'all48 best original residual', 1e-9)
    close(max(abs(residual)), best['max_abs_native_residual'], 'best ranked residual', 1e-9)
    b = -s['base']-s['J'].T@f0
    return s, T, missing, dyn, dict(best_evaluation=303, records_ranked=len(records),
        best_sha256=sha(SOURCE/'EVAL_303.npz'), independent_report_sha256=sha(SOURCE/'S173_SAVED_INDEPENDENT_CHECK.json'),
        foot_depth=footdepth, original_depth_cap=cap, bearing_legs=sorted(bearing), friction_margin=margin,
        nonfoot_depth=contact['max_nonfoot_penetration'], contact=contact), b


def validate_case(h, out, name, transform, s, b, dyn, solved):
    z = load(out/f'CASE_{name}.npz')
    h['bitwise'](z['T'], transform, name+' T')
    A = s['B']@transform
    assert A.shape == (48, transform.shape[1]) and b.shape == (48,)
    h['bitwise'](z['A'], A, name+' A all48')
    h['bitwise'](z['b'], b, name+' b all48')
    has_primal = 'x' in z
    assert has_primal or not solved
    info = dict(name=name, variables=transform.shape[1], solved=solved)
    if has_primal:
        x = z['x']
        t = float(z['t']) if solved else float(np.max(abs(A@x-b)))
        assert x.shape == (transform.shape[1],) and np.isfinite(x).all()
        assert np.isfinite(t) and t >= 0 and np.all(x >= -1e-10) and np.all(x <= 1+1e-10)
        activation = transform@x; residual = A@x-b
        h['close'](z['activation'], activation, name+' activation')
        h['bitwise'](z['ctrl'], z['activation'], name+' constant ctrl=act')
        h['close'](z['act_dot'], np.zeros(90), name+' analytic saved actdot', 0.)
        assert np.all(activation >= -1e-10) and np.all(activation <= .5+1e-10)
        # Exact zero numerator for ctrl=act; positive muscle tau cannot alter it.
        tau = dyn[:, 1]/(.5+1.5*activation)
        assert np.all(tau > 0) and np.all((activation-activation)/tau == 0)
        h['close'](z['residual'], residual, name+' residual48', 1e-10)
        maxres = float(max(abs(residual)))
        assert maxres <= t+1e-8
        feasible = maxres <= 1e-8 and np.all(x >= -1e-10) and np.all(x <= 1+1e-10)
        info.update(t=t, residual_maxabs=maxres, minimax_point_is_equilibrium_witness=bool(feasible),
                    activation_stationarity='analytic ctrl=act, positive native tau, actdot exactly zero')
        if solved:
            dual = z['dual_marginals']
            assert dual.shape == (96,) and np.isfinite(dual).all() and np.all(dual <= 1e-10)
            h['bitwise'](z['y'], dual[:48]-dual[48:], name+' upper minus lower dual')
            assert sum(abs(z['y'])) <= 1+1e-8
        else:
            assert feasible, 'unsolved embedded primal must be feasible'
    else:
        feasible = False
        assert not any(k in z for k in ('activation', 'ctrl', 'act_dot', 'residual', 't')), 'unsolved child is no witness'
    if feasible:
        info.update(classification='FEASIBLE_WITNESS', separator=None)
    else:
        sep = h['certificate'](A, b, z['y'])
        if 'separator_gap' in z:
            h['close'](z['separator_gap'], sep['gap'], name+' saved separator', 1e-10)
        info.update(classification='INFEASIBLE_CERTIFICATE', separator=sep,
                    minimax_point_is_equilibrium_witness=False)
    return z, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '--outputroot', dest='output', type=Path, required=True)
    args = parser.parse_args(); assert args.output.is_absolute()
    out = args.output.resolve(); assert out.is_dir()
    destination = out/'S174_SAVED_INDEPENDENT_CHECK.json'
    assert not destination.exists(), 'Refuse to overwrite independent evidence'
    inputs = read_json(out/'INPUT_HASHES.json')
    assert inputs == read_json(out/'INPUT_HASHES_AFTER.json')
    assert all(sha(p) == digest for p, digest in inputs.items())
    # Hash the full source linkage and all consumed output evidence before/after.
    paths = [Path(__file__).resolve(), HERE/'verify_fixed_pose_soft.py', HERE/'verify_near_initial_support.py',
             SOURCE/'SUMMARY.json', SOURCE/'S173_SAVED_INDEPENDENT_CHECK.json', SOURCE/'EVALUATIONS.json',
             SOURCE/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz', SOURCE/'EVAL_303.npz', SOURCE/'FROZEN_SEARCH.npz',
             SOURCE/'PREREG.json', SOURCE/'FORWARD_606_probe_forward.npz', SOURCE/'FORWARD_607_optimized_state_forward.npz',
             S154/'INITIALIZATION.npz', S154/'NATIVE_MODEL_ARRAYS.npz', S154/'COM_CONTACT_METADATA.json',
             S165/'ZERO_V_000.npz', S166, CSV, out/'SUMMARY.json', out/'SOLVE_TRACE.json']
    paths.extend(out/f'CASE_{name}.npz' for name in ('wide', 'real', 'fill'))
    source_xml = Path(read_json(SOURCE/'PREREG.json')['model']['xml'])
    paths.append(source_xml)
    before = {str(p): sha(p) for p in paths}
    h = pure_helpers()
    s, T, missing, dyn, source_info, b = validate_source(h)
    transforms = dict(wide=.5*np.eye(90), real=T, fill=np.column_stack((T, .5*np.eye(90)[:, missing])))
    summary = read_json(out/'SUMMARY.json')
    declared = {r['name']: r for r in summary['cases']}
    assert set(declared) == set(transforms)
    cases, results = {}, {}
    for name, transform in transforms.items():
        assert type(declared[name]['solved']) is bool
        cases[name], results[name] = validate_case(h, out, name, transform, s, b, dyn, declared[name]['solved'])
    assert results['wide']['solved']
    if results['wide']['classification'] == 'INFEASIBLE_CERTIFICATE':
        assert not results['real']['solved'] and not results['fill']['solved']
        for name in ('real', 'fill'):
            h['bitwise'](cases[name]['y'], cases['wide']['y'], name+' reused validated widest y')
            assert results[name]['classification'] == 'INFEASIBLE_CERTIFICATE'
        expected = ['wide']
    else:
        assert results['real']['solved']
        if results['real']['classification'] == 'INFEASIBLE_CERTIFICATE':
            assert results['fill']['solved']; expected = ['wide', 'real', 'fill']
        else:
            expected = ['wide', 'real']
            assert not results['fill']['solved']
            assert results['fill']['classification'] == 'FEASIBLE_WITNESS'
            h['bitwise'](cases['fill']['x'], np.r_[cases['real']['x'], np.zeros(16)], 'embedded real feasible fill witness')
    summary = read_json(out/'SUMMARY.json')
    trace = read_json(out/'SOLVE_TRACE.json')
    assert isinstance(trace, list)
    assert [r['name'] for r in trace] == expected
    assert all(r['success'] and r['status'] == 0 for r in trace)
    calls = summary['calls']
    assert calls['phaseI_LPs'] == len(expected) <= 3 and calls['certificate_LPs'] == 0
    assert all(calls[k] == 0 for k in ('model_load', 'forward', 'step', 'neural'))
    declared = {r['name']: r for r in summary['cases']}
    assert set(declared) == set(results)
    for name, actual in results.items():
        assert declared[name]['status'] == actual['classification']
        assert declared[name]['solved'] == actual['solved']
    after = {str(p): sha(p) for p in paths}
    assert before == after and all(sha(p) == digest for p, digest in inputs.items())
    result = dict(status='VERIFIED_SAVED_ONLY', verified=True, source=source_info,
        cases=list(results.values()), solve_order=expected, actual_phaseI_LPs=len(expected),
        matrix_identity='bitwise saved full48 A=B@T, b=-base-J.T@nativef0, independently scalar-reconstructed source',
        certificates='same supplied widest y where applicable, exact Fraction binary64 dot/support confirmation per box',
        no_feasible_witness=all(r['classification'] == 'INFEASIBLE_CERTIFICATE' for r in results.values()),
        calls=dict(engine_import=0, model_load=0, forward=0, step=0, neural=0, optimization=0),
        input_sha256=before, input_sha256_after=after,
        scope='S174/P1-S2 fixed S173 EVAL303 native SOFT saved box only; no global-pose impossibility, neural/FIR attainability, dynamic standing/walking or SI claim; SI NOT_ESTABLISHED')
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True, report=str(destination), cases=result['cases'], actual_phaseI_LPs=len(expected)), ensure_ascii=False))


if __name__ == '__main__':
    main()
