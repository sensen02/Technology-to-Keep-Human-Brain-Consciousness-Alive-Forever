"""S137/P1: one OFFLINE trajectory-safety QP; import is inert.

Parent executes only after S136 independent verification. This is a limited local
linear proposal, never a physical safety/behavior admission or checkpoint update.
No simulator, producer import, neural runtime, subprocess, retry or line search.
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
BASE = Path('/home/sensen/Desktop/engine修复验收/s137-trajectory-safety-proposal')
S136_BASE = Path('/home/sensen/Desktop/engine修复验收/s136-safety-trajectory-jacobian')
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms/TRAJECTORY.npz')
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
INDICES = np.array([(p, b) for p in (3, 15, 38) for b in range(5)])
SCALE = .001 / np.sqrt(15)
SLACK_SCALE = 1e-4
MIN_IMPROVEMENT = 1e-8
TOL = 1e-9


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


def check_hashes(hashes):
    assert isinstance(hashes, dict) and hashes
    for name, digest in hashes.items():
        assert Path(name).is_absolute() and sha(name) == digest, 'Changed input: ' + name


def inputs(source, verification):
    """Fail closed before output creation or optimizer import/call."""
    assert source.is_absolute() and source.parent == S136_BASE.resolve()
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
    assert js(source / 'PREREG.json')['source_hashes'] == original_inputs
    assert js(source / 'COMPLETE.json')['status'] == 'COMPLETE'
    assert not (source / 'FAILED.json').exists()
    for name in ('JACOBIAN.npz', 'DERIVATIVE_REPORT.json', 'COMPLETE.json', 'PREREG.json'):
        assert {str(Path(p).resolve()): h for p, h in outputs.items()}[str(source / name)] == sha(source / name)
    j = archive(source / 'JACOBIAN.npz')
    assert exact(j['weight_indices'], INDICES)
    assert j['valid'].shape == (15,) and j['valid'].dtype == np.dtype(bool)
    assert r['valid_columns'] == j['valid'].tolist() and r['valid_count'] == int(j['valid'].sum())
    assert js(source / 'DERIVATIVE_REPORT.json')['valid_columns'] == j['valid'].tolist()
    w = archive(REFERENCE)['weights']
    assert w.shape == (66, 5) and exact(j['reference_weights'], w)
    assert j['time'].shape == (121,) and j['joint_bounds'].shape == (42, 2)
    assert float(j['h']) == .000125 and float(j['half']) == .0000625
    shapes = dict(signed_joint_position=(121, 42), load_ratio=(121,),
                  root_angle_rad=(121,), com_vertical_velocity=(121,), height=(121,), qvel=(121, 48))
    for key, shape in shapes.items():
        assert j['reference_' + key].shape == shape
        assert j[key + '_D_half'].shape == (*shape, 15)
        assert np.isfinite(j['reference_' + key]).all() and np.isfinite(j[key + '_D_half']).all()
        assert exact(j['masked_' + key], np.where(j['valid'], j[key + '_D_half'], 0.))
    hashes = dict(original_inputs)
    for name, digest in outputs.items():
        assert name not in hashes or hashes[name] == digest
        hashes[name] = digest
    hashes.update(r['helper_source_hashes'])
    for path in (verification, Path(r['verifier']), source / 'OUTPUT_HASHES.json', REFERENCE, FORMAL,
                 Path(__file__).resolve(), ROOT / 'verify_margin_contact_candidate.py'):
        hashes[str(path)] = sha(path)
    return j, w, js(FORMAL), hashes


def problem(j, w, formal):
    """Raw unscaled rows in z=(x[15],y); radians slack t=1e-4*y only on joints."""
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
    # q + D*delta >= lower-formal-t and <= upper+formal+t.
    lower_rows = dq.copy(); lower_rows[:, 15] = SLACK_SCALE
    upper_rows = dq.copy(); upper_rows[:, 15] = -SLACK_SCALE
    add(lower_rows, lower.ravel(), np.inf, 'all121x42_joint_lower_plus_t')
    add(upper_rows, -np.inf, upper.ravel(), 'all121x42_joint_upper_minus_t')
    add(tangent('load_ratio'), -1e-9, np.inf, 'all121_delta_load_nonworsen')
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--verification', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    source = args.source.resolve()
    verification = (args.verification or source / 'SAFETY_TRAJECTORY_JACOBIAN_INDEPENDENT_CHECK.json').resolve()
    j, w, formal, hashes = inputs(source, verification)
    # Reuse the independent diagnostic certificate, never a producer solver.
    from scipy.optimize import minimize, LinearConstraint, Bounds
    from verify_margin_contact_candidate import qp_certificate
    for name, module in tuple(sys.modules.items()):
        if name.startswith('verify_') and getattr(module, '__file__', None):
            path = str(Path(module.__file__).resolve()); hashes[path] = sha(path)
    assert not any(k in sys.modules for k in ('mujoco', 'torch', 'engine.neural_cond'))
    check_hashes(hashes)
    p = problem(j, w, formal)
    assert residual(p, p['initial']) <= TOL, 'No fabricated feasible baseline/slack'
    now = datetime.now(timezone.utc)
    out = args.output or BASE / now.strftime('%Y%m%dT%H%M%SZ')
    assert out.is_absolute() and out.parent == BASE and not out.exists()
    stamp = datetime.strptime(out.name, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    assert abs((now - stamp).total_seconds()) < 120
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'INPUT_HASHES.json', hashes)
    save(out / 'PREREG.json', dict(task='S137/P1', created_utc=now.isoformat(), source=str(source),
        verification=str(verification), source_hashes=hashes, valid_columns=j['valid'], weight_indices=INDICES,
        decision='15 x plus dimensionless y=t/1e-4', objective='y+.05*sum(x*x)+.05*y*y',
        trust_L2=.001, trust_component=SCALE, slack_scale=SLACK_SCALE, min_slack_improvement=MIN_IMPROVEMENT,
        formal_constraints= formal, baseline_peak=p['initial_peak'], baseline_margin=p['initial_margin'],
        solver='one SLSQP strictly convex QP; no retry/rescale/scan', max_qp_calls=1,
        constraints='full raw A/lo/hi/bounds in QP.npz, all121x42 signed joint bounds; t only for joint radians',
        limitation='LIMITED_LOCAL_PROPOSAL_NOT_TRUTH: component cap ~2.066x measured h; simultaneous L2 .001 exceeds measured single-coordinate .000125; two-scale column tests do not prove finite-step joint validity',
        missing_linear_gates='nonfoot contact, leg contact and original first-failure/support metrics require future full physical S119 test',
        frozen='all non15 coordinates, invalid columns, zero mask and pool36',
        physical_admission=False, promotion=False, checkpoint_replacement=False, units='NOT_ESTABLISHED'))
    np.savez_compressed(out / 'QP.npz', **p, weight_indices=INDICES, valid=j['valid'],
                        reference_weights=w, scale=np.asarray(SCALE), slack_scale=np.asarray(SLACK_SCALE))
    ledger = dict(qp_calls=0, model_load=0, forward=0, step=0, neural_ticks=0)
    save(out / 'CALL_LEDGER.json', ledger)
    try:
        # Normalize rows only for conditioning. All primal/certificate tests use raw rows.
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
        # Roundoff in actual saved weights must not evade the raw-row postgate.
        actual = z.copy(); actual[:15] = (candidate-w)[INDICES[:, 0], INDICES[:, 1]]/SCALE
        cert = qp_certificate(p['H'], p['g'], p['A'], p['lo'], p['hi'], actual, p['bounds'])
        qpred = j['reference_signed_joint_position'] + np.einsum('tjk,k->tj', j['masked_signed_joint_position'], SCALE*actual[:15])
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
            source_rehash=False)
        check_hashes(hashes); gates['source_rehash'] = True
        accepted = all(gates.values())
        np.savez_compressed(out/'SOLUTION.npz', z=z, actual_z=actual, delta=candidate-w,
            proposed_weights=candidate, gradient=p['H']@actual+p['g'], row_scale=row_scale,
            raw_value=p['A']@actual, predicted_joint_position=qpred)
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
            primal_violation=residual(p,actual),certificate=cert,
            interpretation='LIMITED_LOCAL_PROPOSAL_NOT_TRUTH; rejected proposal arrays remain diagnostic, never candidate admission',
            physical_admission=False,promotion=False,checkpoint_replacement=False,standing_or_walking_claim=False,
            future_required='independent saved-byte S137 verification then one future full physical trial under unchanged original S119 rules',units='NOT_ESTABLISHED'))
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
