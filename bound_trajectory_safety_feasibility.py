"""S140/P1: ONE offline HiGHS LP bound on EXACT S138 local planning polytope.
Import is inert. Parent executes once after SOURCE READY; no actual-model imports.
This optimization-bound diagnostic is NEVER a candidate or physical admission.
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
BASE = Path('/home/sensen/Desktop/engine修复验收/s140-trajectory-safety-feasibility').resolve()
S138 = Path('/home/sensen/Desktop/engine修复验收/s138-formal-benefit-safety-proposal/20261024T192136Z').resolve()
S139 = Path('/home/sensen/Desktop/engine修复验收/s139-formal-benefit-candidate-closed-loop/20261024T192550Z').resolve()
S136 = Path('/home/sensen/Desktop/engine修复验收/s136-safety-trajectory-jacobian/20261024T183631Z').resolve()
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json').resolve()
SCALE = .001 / np.sqrt(15)
SLACK_SCALE = 1e-4
PRIMAL_TOL = 1e-9
DUAL_TOL = 1e-8
ZERO_TOL = 1e-9


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    def serial(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=serial)+'\n')


def archive(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def bind_inputs():
    """Canonicalize once, hash each unique file once, never duplicate huge reports.

    Only explicit hash maps are consumed (no recursive giant report traversal).
    All files under S138/S139, including parent finals/conclusions, are bound.
    """
    expected = {}
    resolved = {}
    def key(name):
        name = str(name)
        if name not in resolved: resolved[name] = str(Path(name).resolve())
        return resolved[name]
    def add_map(mapping):
        assert isinstance(mapping, dict)
        if 'before' in mapping:
            assert mapping['before'] == mapping['after']
            mapping = mapping['before']
        for name, digest in mapping.items():
            assert Path(name).is_absolute() and isinstance(digest, str) and len(digest) == 64
            k = key(name)
            assert k not in expected or expected[k] == digest, 'Conflicting canonical hash: '+k
            expected[k] = digest
    r138 = js(S138/'PARENT_FORMAL_BENEFIT_PROPOSAL_INDEPENDENT_CHECK.json')
    r139 = js(S139/'init80ms/FORMAL_BENEFIT_CANDIDATE_INDEPENDENT_CHECK.json')
    r136 = js(S136/'SAFETY_TRAJECTORY_JACOBIAN_INDEPENDENT_CHECK.json')
    assert all(r138[k] is True for k in ('passed','evidence_passed','proposal_admitted','candidate_admitted'))
    assert r139['passed'] is True and r139['S119_short_window_pass'] is False
    assert r139['promotion'] is False and r139['autonomy'] is False
    assert r136['passed'] is True and r136['valid_count'] == 11
    for report in (r136, r138, r139):
        for field in ('input_hashes', 'artifact_hashes', 'helper_source_hashes'):
            if field in report:
                if field == 'artifact_hashes':
                    assert set(report[field]) == {'S139', 'S121', 'S115', 'S113'}
                    for arm_map in report[field].values(): add_map(arm_map)
                else: add_map(report[field])
        if 'verifier' in report:
            add_map({report['verifier']: report['verifier_sha256']})
    for base in (S136, S138, S139):
        assert js(base/'COMPLETE.json')['status'] == 'COMPLETE'
        assert not (base/'FAILED.json').exists()
        for name in ('INPUT_HASHES.json','OUTPUT_HASHES.json'):
            if (base/name).is_file(): add_map(js(base/name))
    assert js(S138/'PROPOSAL_REPORT.json')['proposal_accepted'] is True
    paths = set(expected)
    for base in (S138, S139):
        assert (base/'PARENT_CONCLUSION.zh-CN.md').is_file()
        paths.update(key(p) for p in base.rglob('*') if p.is_file())
    paths.update(key(p) for p in (Path(__file__), ROOT/'propose_formal_benefit_safety_step.py',
                 FORMAL, S136/'JACOBIAN.npz', S136/'OUTPUT_HASHES.json',
                 S136/'SAFETY_TRAJECTORY_JACOBIAN_INDEPENDENT_CHECK.json'))
    actual = {}
    for path in sorted(paths):
        digest = sha(path)
        assert path not in expected or digest == expected[path], 'Changed frozen input: '+path
        actual[path] = digest
    return actual


def canonical_inequalities(p):
    """G z <= h: finite lower/upper raw rows then all 32 box faces.

    Preserve all raw A rows separately. Drop ONLY feasible exactly-zero G rows,
    never drop redundant nonzero rows. Positive maxabs normalization is fixed.
    """
    G, h, labels, origin, side = [], [], [], [], []
    for i, (row, lo, hi, label) in enumerate(zip(p['A'], p['lo'], p['hi'], p['row_labels'])):
        if np.isfinite(lo):
            G.append(-row); h.append(-lo); labels.append(str(label)+':lower'); origin.append(i); side.append(-1)
        if np.isfinite(hi):
            G.append(row); h.append(hi); labels.append(str(label)+':upper'); origin.append(i); side.append(1)
    for i, (lo, hi) in enumerate(p['bounds']):
        row = np.zeros(16); row[i] = 1.
        G.extend([-row, row]); h.extend([-lo, hi])
        labels.extend([f'box:{i}:lower',f'box:{i}:upper'])
        origin.extend([-1,-1]); side.extend([-1,1])
    G, h = np.asarray(G), np.asarray(h)
    divisor = np.max(np.abs(G), axis=1)
    keep = divisor > 0.
    assert np.all(h[~keep] >= 0.), 'Infeasible zero row; no solver/retry permitted'
    d = divisor[keep]
    return dict(G=G, h=h, G_row_labels=np.asarray(labels), G_origin=np.asarray(origin),
                G_side=np.asarray(side), kept_indices=np.flatnonzero(keep),
                discarded_zero_indices=np.flatnonzero(~keep), row_divisor=d,
                G_scaled=G[keep]/d[:,None], h_scaled=h[keep]/d)


def certificate(p, canon, c, z, marginal):
    scaled_lambda = -marginal
    raw_lambda = np.zeros(len(canon['h']))
    raw_lambda[canon['kept_indices']] = scaled_lambda / canon['row_divisor']
    G, h = canon['G'], canon['h']
    stationarity = c + G.T @ raw_lambda
    slack = h-G@z
    primal = max(0., float(np.max(-slack)))
    nonnegative = max(0., float(np.max(-raw_lambda)), float(np.max(-scaled_lambda)))
    dual_obj = float(-h@raw_lambda)
    objective = float(c@z)
    gap = objective-dual_obj
    comp = float(np.max(np.abs(raw_lambda*slack)))
    raw_primal = max(0., float(np.max(p['lo']-p['A']@z)), float(np.max(p['A']@z-p['hi'])),
                     float(np.max(p['bounds'][:,0]-z)), float(np.max(z-p['bounds'][:,1])))
    # Residual-aware finite-box lower bound; not an exact/rational arithmetic proof.
    nonnegative_lambda = np.maximum(raw_lambda, 0.)
    corrected_stationarity = c + G.T@nonnegative_lambda
    support = float(np.sum(np.where(corrected_stationarity >= 0.,
                 corrected_stationarity*p['bounds'][:,0], corrected_stationarity*p['bounds'][:,1])))
    unadjusted_lower_bound = float(-h@nonnegative_lambda + support)
    roundoff_allowance = float(1e-12 * (1. + np.sum(np.abs(h*nonnegative_lambda))
                                + np.sum(np.abs(corrected_stationarity)*np.max(np.abs(p['bounds']),axis=1))))
    lower_bound = unadjusted_lower_bound - roundoff_allowance
    checks = dict(finite=bool(np.isfinite(z).all() and np.isfinite(raw_lambda).all()),
                  raw_primal=raw_primal <= PRIMAL_TOL, canonical_primal=primal <= PRIMAL_TOL,
                  nonnegative=nonnegative <= DUAL_TOL,
                  stationarity=float(np.max(np.abs(stationarity))) <= DUAL_TOL,
                  complementarity=comp <= DUAL_TOL, objective_gap=abs(gap) <= DUAL_TOL)
    cert = dict(certificate_passed=all(checks.values()), checks=checks,
                raw_primal_violation=raw_primal, canonical_primal_violation=primal,
                dual_nonnegative_violation=nonnegative, stationarity_inf=float(np.max(np.abs(stationarity))),
                complementarity_inf=comp, primal_objective=objective, dual_objective=dual_obj,
                primal_dual_gap=gap, residual_aware_box_lower_bound=lower_bound,
                unadjusted_box_lower_bound=unadjusted_lower_bound, roundoff_allowance=roundoff_allowance,
                primal_tolerance=PRIMAL_TOL, dual_tolerance=DUAL_TOL,
                method='HiGHS normalized inequality marginals -> lambda_scaled=-marginal; lambda_raw=lambda_scaled/maxabs(Grow); c+G.T lambda=0; dual=-h.lambda',
                proof_scope='NUMERICAL_PRIMAL_DUAL_LP_CERTIFICATE_NOT_EXACT_ARITHMETIC; global only within saved LOCAL linear trust polytope')
    return cert, raw_lambda, scaled_lambda, stationarity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    hashes = bind_inputs()
    # Only this inert producer is imported; inputs()/main()/runtime are NOT called.
    import propose_formal_benefit_safety_step as producer
    assert producer.SCALE == SCALE and producer.SLACK_SCALE == SLACK_SCALE
    j = archive(S136/'JACOBIAN.npz'); w = j['reference_weights']; formal = js(FORMAL)
    assert exact(j['valid'], producer.VALID) and int(j['valid'].sum()) == 11
    p = producer.problem(j, w, formal)
    old = archive(S138/'QP.npz')
    for field in ('A','lo','hi','bounds','initial','row_labels'):
        assert exact(p[field], old[field]), 'S138 exact polytope mismatch: '+field
    assert p['A'].shape[1] == 16 and np.all(p['bounds'][:15][~j['valid']] == 0.)
    canon = canonical_inequalities(p)
    c = np.r_[np.zeros(15), 1.]
    assert producer.residual(p,p['initial']) <= PRIMAL_TOL
    assert not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond'))
    now = datetime.now(timezone.utc)
    out = (args.output or BASE/now.strftime('%Y%m%dT%H%M%SZ')).resolve()
    assert out.parent == BASE and not out.exists()
    stamp = datetime.strptime(out.name,'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    assert abs((now-stamp).total_seconds()) < 120
    out.mkdir(parents=True,exist_ok=False)
    save(out/'INPUT_HASHES.json',hashes)
    save(out/'PREREG.json',dict(task='S140/P1',created_utc=now.isoformat(),
         source_S138=str(S138),source_S139=str(S139),source_S136=str(S136),
         input_manifest=str(out/'INPUT_HASHES.json'),input_manifest_sha256=sha(out/'INPUT_HASHES.json'),
         objective='minimize y=t/1e-4; c=[0]*15+[1]; NO weight/slack quadratic regularization',
         solver='single scipy.optimize.linprog(method=highs); all variable bounds free; ALL original bounds explicit in G',
         solver_options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9),
         max_lp_calls=1,no_retries=True,no_rescale_after_result=True,no_bound_or_mask_broadening=True,
         normalization='G_scaled=G/maxabs(row); exactly-zero feasible rows dropped only; nonzero redundant rows retained',
         constraints='EXACT S138 A/lo/hi/bounds: all121x42 formal joint +/- t; all121 body; load0..76>=.8; load77..120 delta>=-.001',
         valid=j['valid'],weight_indices=producer.INDICES,trust_component=SCALE,trust_L2=.001,
         slack_scale=SLACK_SCALE,zero_tolerance=ZERO_TOL,primal_tolerance=PRIMAL_TOL,dual_tolerance=DUAL_TOL,
         interpretation='OPTIMIZATION_BOUND_DIAGNOSTIC_ONLY; positive robust LP lower bound excludes zero slack ONLY within same LOCAL linear trust plan; not full-model impossibility',
         physical_admission=False,candidate_promotion=False,autonomy=False,actual_trial=False,
         original_17_physical_tests='UNCHANGED_NOT_EVALUATED',units='NOT_ESTABLISHED'))
    np.savez_compressed(out/'QPROBLEM.npz',A=p['A'],lo=p['lo'],hi=p['hi'],bounds=p['bounds'],
         initial=p['initial'],row_labels=p['row_labels'],c=c,**canon,reference_weights=w,
         weight_indices=producer.INDICES,valid=j['valid'],scale=SCALE,slack_scale=SLACK_SCALE)
    ledger=dict(lp_calls=0,model_load=0,forward=0,step=0,neural_ticks=0)
    save(out/'CALL_LEDGER.json',ledger)
    try:
        from scipy.optimize import linprog
        ledger['lp_calls']=1; save(out/'CALL_LEDGER.json',ledger)
        result=linprog(c,A_ub=canon['G_scaled'],b_ub=canon['h_scaled'],bounds=[(None,None)]*16,
                       method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
        save(out/'SOLVER_STATUS.json',dict(success=bool(result.success),status=int(result.status),message=str(result.message),iterations=int(result.nit)))
        assert result.success, 'Single LP failed; no retry, no alternate solver'
        z=np.asarray(result.x); marginal=np.asarray(result.ineqlin.marginals)
        cert,raw_lambda,scaled_lambda,stationarity=certificate(p,canon,c,z,marginal)
        delta=np.zeros_like(w); delta[producer.INDICES[:,0],producer.INDICES[:,1]]=SCALE*z[:15]
        diagnostic=w+delta
        actual_z=z.copy(); actual_z[:15]=(diagnostic-w)[producer.INDICES[:,0],producer.INDICES[:,1]]/SCALE
        qpred=j['reference_signed_joint_position']+np.einsum('tjk,k->tj',j['masked_signed_joint_position'],SCALE*z[:15])
        loadpred=j['reference_load_ratio']+j['masked_load_ratio']@(SCALE*z[:15])
        metrics=producer.predicted_load_metrics(j,loadpred)
        qb=j['joint_bounds']
        peak=float(np.maximum(np.maximum(qb[:,0]-qpred,qpred-qb[:,1]),0.).max())
        allowed=np.zeros_like(w,dtype=bool); allowed[producer.INDICES[:,0],producer.INDICES[:,1]]=j['valid']
        gates=dict(certificate=cert['certificate_passed'],weights_bounds=bool(np.all((diagnostic>=0)&(diagnostic<=4))),
                   frozen=exact(diagnostic[~allowed],w[~allowed]),zero_mask=exact(diagnostic==0,w==0),
                   positive_existing_weights=bool(np.all(diagnostic[w>0]>0)),pool36=exact(diagnostic[36],w[36]),
                   invalid_four_frozen=bool(np.all(z[:15][~j['valid']]==0.)),
                   trust_component=bool(np.max(np.abs(delta))<=SCALE+1e-15),trust_L2=bool(np.linalg.norm(delta)<=.001+1e-15),
                   saved_weight_raw_primal=producer.residual(p,actual_z)<=PRIMAL_TOL,
                   peak_with_slack=peak<=p['formal_limit']+SLACK_SCALE*z[15]+PRIMAL_TOL,
                   no_actual_model=not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
        classification=('POSITIVE_LOCAL_LINEAR_TRUST_GAP' if cert['certificate_passed'] and cert['residual_aware_box_lower_bound']>ZERO_TOL
                        else 'ZERO_SLACK_LOCAL_LINEAR_REFERENCE' if cert['certificate_passed'] and z[15]<=ZERO_TOL
                        else 'UNRESOLVED_NUMERICAL_BOUND')
        np.savez_compressed(out/'SOLUTION.npz',z=z,actual_weight_z=actual_z,delta=diagnostic-w,
             diagnostic_weights=diagnostic,raw_A_value=p['A']@z,predicted_joint_position=qpred,
             predicted_load_ratio=loadpred,delta_load_ratio=metrics['delta_load_ratio'],
             highs_marginals=marginal,lambda_raw=raw_lambda,lambda_scaled=scaled_lambda,
             stationarity=stationarity,canonical_slack=canon['h']-canon['G']@z,
             highs_inequality_residual=np.asarray(result.ineqlin.residual))
        np.savez_compressed(out/'DIAGNOSTIC_WEIGHTS.npz',weights=diagnostic)
        with (out/'ALL_WEIGHT_DELTAS.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['pool','bin','reference','diagnostic','delta','allowed'])
            for pool,b in np.ndindex(w.shape): writer.writerow([pool,b,repr(float(w[pool,b])),repr(float(diagnostic[pool,b])),repr(float(diagnostic[pool,b]-w[pool,b])),bool(allowed[pool,b])])
        save(out/'LP_CERTIFICATE.json',cert)
        save(out/'BOUND_REPORT.json',dict(classification=classification,evidence_checks_passed=all(gates.values()),gates=gates,
             y_star=float(z[15]),slack_star=SLACK_SCALE*float(z[15]),best_predicted_peak=peak,
             formal_limit=p['formal_limit'],predicted_local_gap=peak-p['formal_limit'],
             initial_peak=p['initial_peak'],initial_slack=p['initial_margin'],
             slack_improvement=p['initial_margin']-SLACK_SCALE*float(z[15]),
             dual_y_lower_bound=cert['residual_aware_box_lower_bound'],
             dual_slack_lower_bound=SLACK_SCALE*cert['residual_aware_box_lower_bound'],
             used_norm=float(np.linalg.norm(diagnostic-w)),used_component=float(np.max(np.abs(diagnostic-w))),
             active_box_faces=int(np.sum(np.abs(canon['h'][-32:]-canon['G'][-32:]@z)<=1e-9)),
             predicted_load=metrics,certificate=cert,
             comparison_S138_predicted_peak=js(S138/'PROPOSAL_REPORT.json')['predicted_peak'],
             interpretation='Only exact saved LOCAL LINEAR trust-plan optimum. Positive bound is NOT full-model impossibility. Zero LP slack is NOT formal pass. Diagnostic weights NEVER candidate/admission. A larger toward-boundary actual step requires a NEW plan and separate verification.',
             physical_admission=False,candidate_promotion=False,autonomy=False,actual_trial=False,
             original_17_physical_tests='UNCHANGED_UNKNOWN_NOT_EVALUATED',units='NOT_ESTABLISHED'))
        assert not (out/'CANDIDATE_WEIGHTS.npz').exists()
        save(out/'COMPLETE.json',dict(status='COMPLETE',evidence_checks_passed=all(gates.values()),classification=classification,
             ledger=ledger,independent_verification='PENDING',physical_admission=False,candidate_promotion=False))
    except BaseException:
        save(out/'FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger,no_retry=True));raise
    finally:
        save(out/'OUTPUT_HASHES.json',{str(path):sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.name!='OUTPUT_HASHES.json'})
    print(str(out))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
