"""S138/P1: one OFFLINE formal-benefit safety QP; import is inert.

New planning problem only: pre-first-failure load >= .8, post-failure delta
>= -.001. This does NOT relax any of the original 17 future physical S119
admission tests. No simulator, neural runtime, retry, scan or checkpoint update.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收/s138-formal-benefit-safety-proposal').resolve()
S136_BASE = Path('/home/sensen/Desktop/engine修复验收/s136-safety-trajectory-jacobian').resolve()
S136_SOURCE = S136_BASE / '20261024T183631Z'
S137_SOURCE = Path('/home/sensen/Desktop/engine修复验收/s137-trajectory-safety-proposal/20261024T191248Z').resolve()
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms/TRAJECTORY.npz')
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
INDICES = np.array([(p, b) for p in (3, 15, 38) for b in range(5)])
VALID = np.array([True, True, True, False, False, True, True, True, True, True, True, True, False, False, True])
SCALE = .001 / np.sqrt(15)
SLACK_SCALE = 1e-4
MIN_IMPROVEMENT = 1e-8
TOL = 1e-9
LOAD_FLOOR = .8
FIRST_LOAD_FAILURE = 77
POST_LOAD_BUDGET = .001


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def save(path, value):
    def serial(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=serial) + '\n')
    tmp.replace(path)


def canonical(hashes):
    result = {}
    for name, digest in hashes.items():
        key = str(Path(name).resolve())
        assert key not in result or result[key] == digest, 'Conflicting path aliases'
        result[key] = digest
    return result


def check_hashes(hashes):
    assert isinstance(hashes, dict) and hashes
    for name, digest in hashes.items():
        assert Path(name).is_absolute() and sha(name) == digest, 'Changed input: ' + name
    canonical(hashes)


def bind_s137(hashes):
    """Bind every rejection artifact, including parent evidence absent in old manifest."""
    old_in = js(S137_SOURCE / 'INPUT_HASHES.json')
    old_out = js(S137_SOURCE / 'OUTPUT_HASHES.json')
    check_hashes(old_in); check_hashes(old_out)
    report = js(S137_SOURCE / 'PROPOSAL_REPORT.json')
    final = js(S137_SOURCE / 'PARENT_SAFETY_PROPOSAL_INDEPENDENT_CHECK.json')
    assert report['proposal_accepted'] is False
    assert final['passed'] is True and final['evidence_passed'] is True
    assert final['proposal_admitted'] is False and final['candidate_admitted'] is False
    assert final['verifier_sha256'] == sha(final['verifier'])
    assert Path(final['verifier']).resolve() == ROOT / 'verify_trajectory_safety_proposal.py'
    assert js(S137_SOURCE / 'COMPLETE.json')['status'] == 'COMPLETE'
    assert not (S137_SOURCE / 'FAILED.json').exists()
    assert not (S137_SOURCE / 'CANDIDATE_WEIGHTS.npz').exists()
    for name in ('PARENT_CONCLUSION.zh-CN.md', 'REJECTED_TRAJECTORY_SAFETY_PROPOSAL.png',
                 'QP.npz', 'SOLUTION.npz', 'QP_CERTIFICATE.json', 'ALL_WEIGHT_DELTAS.csv'):
        assert (S137_SOURCE / name).is_file()
    # Never write to S137; all files, not merely its original output manifest, are inputs.
    bound = canonical(old_in)
    bound.update(canonical(old_out))
    for path in sorted(S137_SOURCE.rglob('*')):
        if path.is_file(): bound[str(path.resolve())] = sha(path)
    for name, digest in bound.items():
        assert name not in hashes or hashes[name] == digest
        hashes[name] = digest
    return hashes


def inputs(source, verification):
    """Fail closed before output creation or optimizer import/call."""
    assert source.is_absolute() and source == S136_SOURCE
    assert verification.is_absolute() and verification.parent == source
    r = js(verification)
    assert r['passed'] is True and r['evidence_passed'] is True
    assert r['safety_output_scope_only'] is True and r['general_gradient_verified'] is False
    assert Path(r['verifier']).resolve() == ROOT / 'verify_safety_trajectory_jacobian.py'
    assert r['verifier_sha256'] == sha(r['verifier'])
    check_hashes(r['helper_source_hashes'])
    original_inputs = js(source / 'INPUT_HASHES.json')
    outputs = js(source / 'OUTPUT_HASHES.json')
    check_hashes(original_inputs); check_hashes(outputs)
    assert canonical(js(source / 'PREREG.json')['source_hashes']) == canonical(original_inputs)
    assert js(source / 'COMPLETE.json')['status'] == 'COMPLETE'
    assert not (source / 'FAILED.json').exists()
    for name in ('JACOBIAN.npz', 'DERIVATIVE_REPORT.json', 'COMPLETE.json', 'PREREG.json'):
        assert canonical(outputs)[str((source / name).resolve())] == sha(source / name)
    j = archive(source / 'JACOBIAN.npz')
    assert exact(j['weight_indices'], INDICES)
    assert exact(j['valid'], VALID) and int(j['valid'].sum()) == 11
    assert r['valid_columns'] == j['valid'].tolist() and r['valid_count'] == 11
    assert js(source / 'DERIVATIVE_REPORT.json')['valid_columns'] == j['valid'].tolist()
    reference = archive(REFERENCE); w = reference['weights']
    assert w.shape == (66, 5) and exact(j['reference_weights'], w)
    assert j['time'].shape == (121,) and j['joint_bounds'].shape == (42, 2)
    assert exact(j['time'], reference['time']) and np.isfinite(j['time']).all()
    assert np.all(np.diff(j['time']) > 0.)
    assert np.allclose(j['time'] - j['time'][0], np.arange(121)*.000025, rtol=0., atol=1e-14)
    assert float(j['h']) == .000125 and float(j['half']) == .0000625
    shapes = dict(signed_joint_position=(121, 42), load_ratio=(121,),
                  root_angle_rad=(121,), com_vertical_velocity=(121,), height=(121,), qvel=(121, 48))
    for key, shape in shapes.items():
        assert j['reference_' + key].shape == shape
        assert np.isfinite(j['reference_' + key]).all()
        for suffix in ('_D_h', '_D_half'):
            assert j[key + suffix].shape == (*shape, 15)
            assert np.isfinite(j[key + suffix]).all()
        assert exact(j['masked_' + key], np.where(j['valid'], j[key + '_D_half'], 0.))
    formal = js(FORMAL)
    load_plan(j, formal)
    hashes = canonical(original_inputs)
    for name, digest in canonical(outputs).items():
        assert name not in hashes or hashes[name] == digest
        hashes[name] = digest
    hashes.update(canonical(r['helper_source_hashes']))
    hashes = bind_s137(hashes)
    for path in (verification, Path(r['verifier']), source / 'OUTPUT_HASHES.json', REFERENCE, FORMAL,
                 Path(__file__).resolve(), ROOT / 'verify_margin_contact_candidate.py',
                 ROOT / 'propose_trajectory_safety_step.py', ROOT / 'verify_trajectory_safety_proposal.py'):
        key = str(path.resolve()); digest = sha(path)
        assert key not in hashes or hashes[key] == digest
        hashes[key] = digest
    return j, w, formal, hashes


def load_plan(j, formal):
    """Fixed planning exchange only; no tunable CLI budget and no physical admission."""
    load = j['reference_load_ratio']
    failed = np.flatnonzero(load < LOAD_FLOOR)
    assert failed.size and int(failed[0]) == FIRST_LOAD_FAILURE
    assert np.all(load[:FIRST_LOAD_FAILURE] >= LOAD_FLOOR)
    rel_ms = (j['time'] - j['time'][0])*1000.
    assert abs(rel_ms[FIRST_LOAD_FAILURE] - formal['benefit_exact_bounds']['first_failures_ms']['load']) < 1e-10
    early = rel_ms <= 1.7 + 1e-10
    assert np.all(load[early] >= LOAD_FLOOR)
    assert formal['benefit_exact_bounds']['load_deficit_integral0_1p7_ratio_ms'] == 0.
    return dict(reference_first_load_failure_sample=FIRST_LOAD_FAILURE,
                reference_first_load_failure_ms=float(rel_ms[FIRST_LOAD_FAILURE]),
                pre_failure_samples=[0, 76], pre_failure_predicted_load_min=LOAD_FLOOR,
                early_window_ms=[0., 1.7], early_deficit_target=0.,
                post_failure_samples=[77, 120], post_delta_load_min=-POST_LOAD_BUDGET,
                all121_load_rows=121, explicit_bounded_planning_exchange=True,
                replaces='S137 all121 delta_load >= -1e-9 planning-only guard',
                original_17_physical_admission_tests='UNCHANGED_NOT_EVALUATED',
                initial_load_support_longest_contacts_first_safety_failures='UNKNOWN_UNTIL_PHYSICAL_TEST')


def problem(j, w, formal):
    """Raw unscaled rows in z=(x[15],y); radians slack t=1e-4*y only on joints."""
    load_plan(j, formal)
    safety = formal['safety_exact_bounds']
    limit = safety['extrema']['limit_excess']['max']
    q = j['reference_signed_joint_position']; bounds_q = j['joint_bounds']
    excess = np.maximum(np.maximum(bounds_q[:, 0] - q, q - bounds_q[:, 1]), 0.)
    initial_peak = float(excess.max()); margin = initial_peak - limit
    assert margin > 0., 'This preregistered task requires positive baseline excess over formal'
    A, lo, hi, labels = [], [], [], []
    def add(matrix, lower, upper, label):
        matrix = np.asarray(matrix).reshape(-1, 16)
        n = len(matrix)
        A.append(matrix); lo.append(np.broadcast_to(lower, (n,)).copy())
        hi.append(np.broadcast_to(upper, (n,)).copy())
        labels.extend([label + ':' + str(i) for i in range(n)])
    def tangent(key):
        d = j['masked_' + key] * SCALE
        return np.c_[d.reshape(-1, 15), np.zeros(d.size // 15)]
    dq = tangent('signed_joint_position')
    lower = np.broadcast_to(bounds_q[:, 0] - limit, q.shape) - q
    upper = np.broadcast_to(bounds_q[:, 1] + limit, q.shape) - q
    lower_rows = dq.copy(); lower_rows[:, 15] = SLACK_SCALE
    upper_rows = dq.copy(); upper_rows[:, 15] = -SLACK_SCALE
    add(lower_rows, lower.ravel(), np.inf, 'all121x42_joint_lower_plus_t')
    add(upper_rows, -np.inf, upper.ravel(), 'all121x42_joint_upper_minus_t')
    load_rows = tangent('load_ratio')
    add(load_rows[:FIRST_LOAD_FAILURE], LOAD_FLOOR-j['reference_load_ratio'][:FIRST_LOAD_FAILURE],
        np.inf, 'samples0to76_formal_benefit_load_floor')
    add(load_rows[FIRST_LOAD_FAILURE:], -POST_LOAD_BUDGET, np.inf, 'samples77to120_bounded_load_exchange')
    add(tangent('height'), safety['extrema']['height_ratio']['min'] - j['reference_height'], np.inf, 'all121_formal_height')
    add(tangent('root_angle_rad'), -np.inf, safety['extrema']['angle_rad']['max'] - j['reference_root_angle_rad'], 'all121_formal_angle')
    add(tangent('com_vertical_velocity'), -safety['max_downward_COM_velocity'] - j['reference_com_vertical_velocity'], np.inf, 'all121_formal_COM_vz')
    root_v = tangent('qvel').reshape(121, 48, 16)[:, 2, :]
    add(root_v, -safety['max_downward_root_velocity'] - j['reference_qvel'][:, 2], np.inf, 'all121_formal_root_vz')
    bounds = np.tile([-1., 1.], (16, 1))
    bounds[~np.r_[j['valid'], True]] = 0.
    for k, (p, b) in enumerate(INDICES):
        bounds[k, 0] = max(bounds[k, 0], -w[p, b] / SCALE)
        bounds[k, 1] = min(bounds[k, 1], (4. - w[p, b]) / SCALE)
        if w[p, b] == 0.: bounds[k] = 0.
    bounds[15] = [0., margin / SLACK_SCALE]
    H = .1 * np.eye(16); g = np.r_[np.zeros(15), 1.]
    initial = np.r_[np.zeros(15), margin / SLACK_SCALE]
    return dict(A=np.vstack(A), lo=np.concatenate(lo), hi=np.concatenate(hi), bounds=bounds,
                H=H, g=g, initial=initial, row_labels=np.asarray(labels),
                initial_peak=initial_peak, initial_margin=margin, formal_limit=limit)


def residual(p, z):
    return max(0., float(np.max(p['lo'] - p['A'] @ z)), float(np.max(p['A'] @ z - p['hi'])),
               float(np.max(p['bounds'][:, 0] - z)), float(np.max(z - p['bounds'][:, 1])))


def predicted_load_metrics(j, predicted):
    relative_ms = (j['time']-j['time'][0])*1000.
    early = relative_ms <= 1.7 + 1e-10
    failures = np.flatnonzero(predicted < LOAD_FLOOR)
    first = int(failures[0]) if failures.size else None
    # All 121 samples partition exactly into 77 protected and 44 budgeted rows.
    delta = predicted-j['reference_load_ratio']
    deficit = np.maximum(LOAD_FLOOR-predicted[early], 0.)
    integral = float(np.sum(.5*(deficit[:-1]+deficit[1:])*np.diff(relative_ms[early])))
    return dict(predicted_load_ratio=predicted, delta_load_ratio=delta,
                first_load_failure_sample=first,
                first_load_failure_ms=None if first is None else float(relative_ms[first]),
                load_failure_count=int(failures.size), protected_sample_count=77, post_sample_count=44,
                pre_failure_min=float(predicted[:77].min()),
                post_delta_min=float(delta[77:].min()), all121_delta_min=float(delta.min()),
                early_deficit_integral0_1p7_ratio_ms=integral,
                first_failure_not_earlier77=bool(first is None or first >= 77),
                protected_load_floor=bool(np.all(predicted[:77] >= LOAD_FLOOR-TOL)),
                early_deficit_zero=bool(integral == 0.),
                post_bounded_exchange=bool(np.all(delta[77:] >= -POST_LOAD_BUDGET-TOL)),
                physical_all17='UNKNOWN_NOT_EVALUATED', physical_first_failure_counts='UNKNOWN_NOT_EVALUATED',
                physical_support_and_longest='UNKNOWN_NOT_EVALUATED')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--verification', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    source = args.source.resolve()
    verification = (args.verification or source / 'SAFETY_TRAJECTORY_JACOBIAN_INDEPENDENT_CHECK.json').resolve()
    j, w, formal, hashes = inputs(source, verification)
    from scipy.optimize import minimize, LinearConstraint, Bounds
    from verify_margin_contact_candidate import qp_certificate
    for name, module in tuple(sys.modules.items()):
        if name.startswith('verify_') and getattr(module, '__file__', None):
            path = str(Path(module.__file__).resolve()); hashes[path] = sha(path)
    assert not any(k in sys.modules for k in ('mujoco', 'torch', 'engine.neural_cond'))
    check_hashes(hashes)
    p = problem(j, w, formal)
    plan = load_plan(j, formal)
    assert residual(p, p['initial']) <= TOL, 'No fabricated feasible baseline/slack'
    now = datetime.now(timezone.utc)
    out = (args.output or BASE / now.strftime('%Y%m%dT%H%M%SZ')).resolve()
    assert out.is_absolute() and out.parent == BASE and not out.exists()
    stamp = datetime.strptime(out.name, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    assert abs((now - stamp).total_seconds()) < 120
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'INPUT_HASHES.json', hashes)
    save(out / 'PREREG.json', dict(task='S138/P1', created_utc=now.isoformat(), source=str(source),
        verification=str(verification), source_hashes=hashes, valid_columns=j['valid'], weight_indices=INDICES,
        decision='15 x plus dimensionless y=t/1e-4', objective='y+.05*sum(x*x)+.05*y*y',
        trust_L2=.001, trust_component=SCALE, slack_scale=SLACK_SCALE, min_slack_improvement=MIN_IMPROVEMENT,
        formal_constraints=formal, baseline_peak=p['initial_peak'], baseline_margin=p['initial_margin'],
        early_postload=plan, predecessor=str(S137_SOURCE), original_physical_tests=17,
        solver='one SLSQP strictly convex QP; no retry/rescale/scan', max_qp_calls=1,
        constraints='full raw A/lo/hi/bounds in QP.npz, all121x42 signed joint bounds; t only for joint radians',
        limitation='LIMITED_LOCAL_PROPOSAL_NOT_TRUTH: component cap ~2.066x measured h; simultaneous L2 .001 exceeds measured single-coordinate .000125; two-scale column tests do not prove finite-step joint validity',
        missing_linear_gates='all original17 require future physical S119 test; initial load, contact/support/longest and safety first-failure counts UNKNOWN, not inferred from predicted load',
        frozen='all non15 coordinates, invalid columns, zero mask and pool36',
        physical_admission=False, promotion=False, checkpoint_replacement=False, units='NOT_ESTABLISHED'))
    np.savez_compressed(out / 'QP.npz', **p, weight_indices=INDICES, valid=j['valid'],
                        reference_weights=w, scale=np.asarray(SCALE), slack_scale=np.asarray(SLACK_SCALE))
    ledger = dict(qp_calls=0, model_load=0, forward=0, step=0, neural_ticks=0)
    save(out / 'CALL_LEDGER.json', ledger)
    try:
        row_scale = np.max(np.abs(p['A']), axis=1)
        row_scale = np.where(row_scale > 0., 1. / np.maximum(row_scale, 1e-300), 1.)
        ledger['qp_calls'] = 1; save(out / 'CALL_LEDGER.json', ledger)
        result = minimize(lambda z: .5*z@p['H']@z + p['g']@z, p['initial'],
            jac=lambda z: p['H']@z+p['g'], method='SLSQP',
            bounds=Bounds(*p['bounds'].T),
            constraints=[LinearConstraint(p['A']*row_scale[:, None], p['lo']*row_scale, p['hi']*row_scale)],
            options=dict(maxiter=1000, ftol=1e-12, disp=False))
        z = np.asarray(result.x); delta = np.zeros_like(w)
        delta[INDICES[:, 0], INDICES[:, 1]] = SCALE*z[:15]
        candidate = w + delta
        actual = z.copy(); actual[:15] = (candidate-w)[INDICES[:, 0], INDICES[:, 1]]/SCALE
        cert = qp_certificate(p['H'], p['g'], p['A'], p['lo'], p['hi'], actual, p['bounds'])
        qpred = j['reference_signed_joint_position'] + np.einsum('tjk,k->tj', j['masked_signed_joint_position'], SCALE*actual[:15])
        loadpred = j['reference_load_ratio'] + j['masked_load_ratio'] @ (SCALE*actual[:15])
        metrics = predicted_load_metrics(j, loadpred)
        qb = j['joint_bounds']
        predicted_peak = float(np.maximum(np.maximum(qb[:, 0]-qpred, qpred-qb[:, 1]), 0.).max())
        slack = SLACK_SCALE*actual[15]
        allowed = np.zeros_like(w, dtype=bool)
        allowed[INDICES[:, 0], INDICES[:, 1]] = j['valid']
        gates = dict(solver_success=bool(result.success), finite=bool(np.isfinite(actual).all()),
            raw_primal=bool(residual(p, actual) <= TOL), certificate=bool(cert['certificate_passed']),
            slack_reduced=bool(p['initial_margin']-slack >= MIN_IMPROVEMENT),
            predicted_peak_reduced=bool(p['initial_peak']-predicted_peak >= MIN_IMPROVEMENT),
            peak_within_formal_plus_explicit_slack=bool(predicted_peak <= p['formal_limit']+slack+TOL),
            weights_in_bounds=bool(np.all((candidate>=0)&(candidate<=4))),
            zero_mask=exact(candidate==0, w==0), frozen=exact(candidate[~allowed],w[~allowed]),
            pool36=exact(candidate[36],w[36]), invalid_delta_zero=bool(np.all(actual[:15][~j['valid']]==0)),
            L2=bool(np.linalg.norm(candidate-w)<=.001+1e-15),
            component=bool(np.max(np.abs(candidate-w))<=SCALE+1e-15),
            no_physical_runtime=not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')),
            source_rehash=False,
            first_load_failure_not_earlier77=metrics['first_failure_not_earlier77'],
            pre_failure_load_floor=metrics['protected_load_floor'],
            early_deficit_zero=metrics['early_deficit_zero'],
            post_bounded_load_exchange=metrics['post_bounded_exchange'])
        check_hashes(hashes); gates['source_rehash'] = True
        accepted = all(gates.values())
        np.savez_compressed(out/'SOLUTION.npz', z=z, actual_z=actual, delta=candidate-w,
            proposed_weights=candidate, gradient=p['H']@actual+p['g'], row_scale=row_scale,
            raw_value=p['A']@actual, predicted_joint_position=qpred,
            predicted_load_ratio=loadpred, delta_load_ratio=metrics['delta_load_ratio'])
        with (out/'ALL_WEIGHT_DELTAS.csv').open('w', newline='') as f:
            writer=csv.writer(f); writer.writerow(['pool','bin','reference','proposed','delta','allowed'])
            for pool, b in np.ndindex(w.shape):
                writer.writerow([pool,b,repr(float(w[pool,b])),repr(float(candidate[pool,b])),repr(float(candidate[pool,b]-w[pool,b])),bool(allowed[pool,b])])
        save(out/'QP_CERTIFICATE.json',cert)
        save(out/'PROPOSAL_REPORT.json',dict(proposal_accepted=accepted,gates=gates,
            solver_success=bool(result.success),solver_status=int(result.status),solver_message=str(result.message),iterations=int(result.nit),
            objective=float(.5*actual@p['H']@actual+p['g']@actual),gradient=p['H']@actual+p['g'],
            initial_peak=p['initial_peak'],formal_limit=p['formal_limit'],predicted_peak=predicted_peak,
            initial_slack=p['initial_margin'],predicted_slack=slack,slack_improvement=p['initial_margin']-slack,
            primal_violation=residual(p,actual),certificate=cert,early_postload=plan,predicted_load_benefit=metrics,
            interpretation='LIMITED_LOCAL_PROPOSAL_NOT_TRUTH; even microscale improvement is not formal safety admission; rejected arrays remain diagnostic',
            physical_admission=False,promotion=False,checkpoint_replacement=False,standing_or_walking_claim=False,
            original_17_physical_tests='UNCHANGED_UNKNOWN_NOT_EVALUATED',
            future_required='independent saved-byte S138 verification then one future full physical trial under unchanged original S119 rules',units='NOT_ESTABLISHED'))
        if accepted: np.savez_compressed(out/'CANDIDATE_WEIGHTS.npz',weights=candidate)
        save(out/'COMPLETE.json',dict(status='COMPLETE',proposal_accepted=accepted,ledger=ledger,
             independent_verification='PENDING',physical_admission=False,promotion=False))
    except BaseException:
        save(out/'FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger,promotion=False)); raise
    finally:
        save(out/'OUTPUT_HASHES.json',{str(path):sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.name!='OUTPUT_HASHES.json'})
    print(str(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
