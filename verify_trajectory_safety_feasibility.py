"""S140/P1 independent saved-byte LP certificate verifier; inert imports only.

No optimizer is run here. A positive certified local linear slack bound is NOT
full-model infeasibility, a blocked goal, physical admission or candidate advice.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import sys
import traceback
import numpy as np
from verify_formal_benefit_safety_proposal import rebuild, archive, exact

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收/s140-trajectory-safety-feasibility').resolve()
S136 = Path('/home/sensen/Desktop/engine修复验收/s136-safety-trajectory-jacobian/20261024T183631Z').resolve()
S138 = Path('/home/sensen/Desktop/engine修复验收/s138-formal-benefit-safety-proposal/20261024T192136Z').resolve()
S139 = Path('/home/sensen/Desktop/engine修复验收/s139-formal-benefit-candidate-closed-loop/20261024T192550Z').resolve()
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms/TRAJECTORY.npz').resolve()
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json').resolve()
INDICES = np.asarray([(p,b) for p in (3,15,38) for b in range(5)])
SCALE = .001 / np.sqrt(15)
TARGET = 'PARENT_SAFETY_FEASIBILITY_INDEPENDENT_CHECK_V5.json'
# Frozen before observing producer results; comparisons use raw unscaled rows.
THRESHOLDS = dict(primal=1e-9, dual_nonnegative=1e-10, stationarity=1e-7,
                  complementarity=1e-7, objective_gap=1e-7)


def js(path):
    return json.loads(Path(path).read_text())


class HashAudit:
    """Canonicalize each path and SHA each actual file once per verification.

    Memoization never substitutes manifest text for actual content hashing.
    Old hash-bound helpers are not edited, imported or rerun for validation.
    """
    def __init__(self):
        self.paths = {}
        self.digests = {}
        self.manifests = {}

    def key(self, path):
        text = str(path)
        if text not in self.paths:
            self.paths[text] = str(Path(text).resolve())
        return self.paths[text]

    def sha(self, path):
        key = self.key(path)
        if key not in self.digests:
            h = hashlib.sha256()
            with Path(key).open('rb') as stream:
                for block in iter(lambda: stream.read(1048576), b''):
                    h.update(block)
            self.digests[key] = h.hexdigest()
        return self.digests[key]

    def canonical(self, data):
        result = {}
        for path, digest in data.items():
            assert Path(path).is_absolute(), 'Nonabsolute hash path: '+path
            key = self.key(path)
            assert key not in result or result[key] == digest, 'Conflicting path aliases: '+key
            result[key] = digest
        return result

    def manifest(self, path):
        key = self.key(path)
        if key not in self.manifests:
            data = self.canonical(js(path))
            assert data, 'Empty manifest'
            for name, digest in data.items():
                assert isinstance(digest, str) and len(digest) == 64 and self.sha(name) == digest, 'Hash mismatch: '+name
            self.manifests[key] = data
        return self.manifests[key]


def independent_problem(j, w, formal):
    """Reuse inert S138 verifier row assembly, never producer assembly.

    Remove the old quadratic term: exact same raw constraints, linear min y.
    Order: each original row finite lower then upper, then each box lower/upper.
    """
    p = rebuild(j, w, formal)
    rows=[]; rhs=[]; labels=[]; origins=[]; sides=[]
    for i in range(len(p['A'])):
        for sign, bound, suffix in ((-1.,p['lo'][i],'lower'),(1.,p['hi'][i],'upper')):
            if np.isfinite(bound):
                rows.append(sign*p['A'][i]); rhs.append(sign*bound)
                labels.append(str(p['row_labels'][i])+':'+suffix); origins.append(i); sides.append(int(sign))
    for i in range(16):
        for sign, col, suffix in ((-1.,0,'lower'),(1.,1,'upper')):
            row=np.zeros(16); row[i]=1.
            if sign < 0.: row=-row
            rows.append(row); rhs.append(sign*p['bounds'][i,col])
            labels.append(f'box:{i}:'+suffix); origins.append(-1); sides.append(int(sign))
    p.update(G=np.asarray(rows),h=np.asarray(rhs),G_row_labels=np.asarray(labels),
             G_origin=np.asarray(origins),G_side=np.asarray(sides))
    p['c'] = np.r_[np.zeros(15), 1.]
    maxima = np.max(np.abs(p['G']), axis=1)
    keep=maxima>0.
    assert np.all(p['h'][~keep]>=0.), 'Infeasible discarded zero row'
    p.update(kept_indices=np.flatnonzero(keep),discarded_zero_indices=np.flatnonzero(~keep),
             row_divisor=maxima[keep],G_scaled=p['G'][keep]/maxima[keep,None],h_scaled=p['h'][keep]/maxima[keep])
    assert p['A'].shape == (10769,16) and np.isfinite(p['G']).all() and np.isfinite(p['h']).all()
    del p['H']
    return p


def certificate(p, z, lam):
    """Original unnormalized A+box primal and G<=h dual KKT certificate."""
    assert z.shape == (16,) and lam.shape == p['h'].shape
    assert np.isfinite(z).all() and np.isfinite(lam).all()
    values = p['A'] @ z
    raw_primal = max(0., float(np.max(p['lo']-values)), float(np.max(values-p['hi'])),
                     float(np.max(p['bounds'][:,0]-z)), float(np.max(z-p['bounds'][:,1])))
    residual = p['G'] @ z - p['h']
    stat = p['c'] + p['G'].T @ lam
    primal_objective = float(p['c'] @ z)
    dual_objective = float(-p['h'] @ lam)
    numbers = dict(raw_primal_violation=raw_primal,
        dual_negative_violation=max(0., float(-lam.min())),
        stationarity_maxabs=float(np.max(np.abs(stat))),
        complementarity_maxabs=float(np.max(np.abs(lam*residual))),
        primal_objective=primal_objective, dual_objective=dual_objective,
        objective_gap=abs(primal_objective-dual_objective))
    gates = dict(primal=raw_primal<=THRESHOLDS['primal'],
        dual_nonnegative=numbers['dual_negative_violation']<=THRESHOLDS['dual_nonnegative'],
        stationarity=numbers['stationarity_maxabs']<=THRESHOLDS['stationarity'],
        complementarity=numbers['complementarity_maxabs']<=THRESHOLDS['complementarity'],
        objective_gap=numbers['objective_gap']<=THRESHOLDS['objective_gap'])
    # Finite-box residual correction makes a genuine lower bound despite a
    # nonzero KKT residual. Clip tiny negative multipliers; do NOT call the raw
    # approximate dual objective a rigorous lower bound for all-free variables.
    ld = np.longdouble
    nonnegative = np.maximum(lam,0.).astype(ld)
    G = p['G'].astype(ld); h = p['h'].astype(ld); c = p['c'].astype(ld)
    r = c + G.T @ nonnegative
    box = p['bounds'].astype(ld)
    corrected = -h @ nonnegative + np.minimum(r*box[:,0],r*box[:,1]).sum()
    magnitude = np.abs(h) @ nonnegative + np.sum((np.abs(c)+np.abs(G).T@nonnegative)*np.max(np.abs(box),axis=1))
    allowance = ld(64)*ld(max(G.shape))*np.finfo(ld).eps*(ld(1)+magnitude)
    lower_bound = float(corrected-allowance)
    lower_bound = float(np.nextafter(lower_bound,-np.inf))
    numbers.update(box_corrected_dual_lower_bound=lower_bound,
                   bound_roundoff_allowance=float(allowance),
                   bound_multiplier_clipping=True)
    return dict(numbers=numbers, gates=gates, certificate_passed=all(gates.values()))


def verify_sources(audit, inputs, checks):
    """Check existing independent evidence and actual transitive file hashes once."""
    def check(name, condition):
        assert condition, name
        checks.append(name)
    sources = ((S136, 'SAFETY_TRAJECTORY_JACOBIAN_INDEPENDENT_CHECK.json', 'verify_safety_trajectory_jacobian.py'),
               (S138, 'PARENT_FORMAL_BENEFIT_PROPOSAL_INDEPENDENT_CHECK.json', 'verify_formal_benefit_safety_proposal.py'),
               (S139, 'init80ms/FORMAL_BENEFIT_CANDIDATE_INDEPENDENT_CHECK.json', 'verify_formal_benefit_candidate_probe.py'))
    required = {}
    for directory, proof_name, verifier_name in sources:
        for name in ('INPUT_HASHES.json','OUTPUT_HASHES.json'):
            path = directory/name
            if directory == S139 and name == 'OUTPUT_HASHES.json':
                proof = js(directory/'init80ms/FORMAL_BENEFIT_CANDIDATE_INDEPENDENT_CHECK.json')
                assert set(proof['artifact_hashes']) == {'S139','S121','S115','S113'}
                for arm in proof['artifact_hashes'].values():
                    required.update(audit.canonical(arm))
                continue
            required[audit.key(path)] = audit.sha(path)
            required.update(audit.manifest(path))
        proof_path = directory/proof_name
        proof = js(proof_path)
        check('source evidence '+directory.name, proof['passed'] is True and
              (proof['evidence_passed'] is True if directory != S139 else
               proof['S119_short_window_pass'] is False and proof['promotion'] is False and proof['autonomy'] is False))
        verifier = ROOT/verifier_name
        check('source verifier identity '+directory.name,
              audit.key(proof['verifier']) == audit.key(verifier) and audit.sha(verifier) == proof['verifier_sha256'])
        required[audit.key(proof_path)] = audit.sha(proof_path)
        required[audit.key(verifier)] = audit.sha(verifier)
        for path, digest in proof['helper_source_hashes'].items():
            check('actual helper SHA '+path, audit.sha(path) == digest)
            required[audit.key(path)] = digest
        check('source completion '+directory.name, js(directory/'COMPLETE.json')['status'] == 'COMPLETE'
              and not (directory/'FAILED.json').exists())
    # Parent conclusions/final checks are source evidence too, not just producer artifacts.
    for directory in (S138,S139):
        check('source parent conclusion '+directory.name,(directory/'PARENT_CONCLUSION.zh-CN.md').is_file())
        for path in directory.rglob('*'):
            if path.is_file(): required[audit.key(path)] = audit.sha(path)
    for path in (REFERENCE, FORMAL, S136/'JACOBIAN.npz'):
        required[audit.key(path)] = audit.sha(path)
    check('all source and transitive hashes committed', all(inputs.get(k) == v for k,v in required.items()))
    j = archive(S136/'JACOBIAN.npz'); w = archive(REFERENCE)['weights']; formal = js(FORMAL)
    proof = js(S136/'SAFETY_TRAJECTORY_JACOBIAN_INDEPENDENT_CHECK.json')
    check('exact weight indices', exact(j['weight_indices'], INDICES))
    check('exact source reference weights', exact(j['reference_weights'], w))
    check('11 valid 4 frozen', j['valid'].shape == (15,) and j['valid'].dtype == bool
          and int(j['valid'].sum()) == 11 and j['valid'].tolist() == proof['valid_columns']
          and j['valid'].tolist() == [bool(c['valid']) for c in proof['columns']])
    check('safety-output-only evidence', proof['safety_output_scope_only'] is True
          and proof['general_gradient_verified'] is False)
    check('finite 330 reference weights', w.shape == (66,5) and np.isfinite(w).all() and np.all((w>=0)&(w<=4)))
    check('exact full121 clock', exact(j['time'],archive(REFERENCE)['time']))
    check('exact FD scales', float(j['h']) == .000125 and float(j['half']) == .0000625)
    for key, shape in dict(signed_joint_position=(121,42), load_ratio=(121,), height=(121,),
            root_angle_rad=(121,), com_vertical_velocity=(121,), qvel=(121,48)).items():
        check('finite full source '+key, j['reference_'+key].shape == shape
              and j[key+'_D_half'].shape == (*shape,15)
              and np.isfinite(j['reference_'+key]).all() and np.isfinite(j[key+'_D_half']).all())
        check('exact independent mask '+key, exact(j['masked_'+key],np.where(j['valid'],j[key+'_D_half'],0.)))
    check('joint bounds shape',j['joint_bounds'].shape == (42,2) and np.isfinite(j['joint_bounds']).all())
    relative_ms = (j['time']-j['time'][0])*1000.
    check('25us monotone time', np.allclose(relative_ms,np.arange(121)*.025,rtol=0.,atol=1e-11)
          and np.all(np.diff(relative_ms)>0.))
    check('reference first load failure77', int(np.flatnonzero(j['reference_load_ratio']<.8)[0]) == 77)
    return j,w,formal


def predictions(j, z):
    dz = SCALE*z[:15]
    pred = {}
    for key in ('signed_joint_position','load_ratio','height','root_angle_rad','com_vertical_velocity','qvel'):
        masked = np.where(j['valid'],j[key+'_D_half'],0.)
        pred[key] = j['reference_'+key] + np.einsum('...k,k->...',masked,dz)
    pred['load_ratio'] = j['reference_load_ratio'] + np.where(j['valid'],j['load_ratio_D_half'],0.)@dz
    q = pred['signed_joint_position']; qb = j['joint_bounds']
    pred['joint_excess'] = np.maximum(np.maximum(qb[:,0]-q,q-qb[:,1]),0.)
    pred['delta_load_ratio'] = pred['load_ratio']-j['reference_load_ratio']
    return pred


def main(output):
    out=Path(output).resolve(); target=out/TARGET
    assert out.parent==BASE and out.is_dir() and not target.exists(), 'Exclusive new S140 output root only'
    audit=HashAudit(); checks=[]
    def check(name, condition):
        assert condition, name
        checks.append(name)
    def eq(name,a,b):
        check(name,exact(np.asarray(a),np.asarray(b)))
    result=dict(passed=False,evidence_passed=False,proposal_admitted=False,candidate_admitted=False,
        physical_admission=False,promotion=False,checkpoint_replacement=False,autonomy=False,
        standing_or_walking_claim=False,safety_output_scope_only=True,general_gradient_verified=False,
        verifier=str(Path(__file__).resolve()),verifier_sha256=audit.sha(__file__),thresholds=THRESHOLDS,
        new_calls=dict(lp_calls=0,model_load=0,forward=0,step=0,neural_ticks=0),units='NOT_ESTABLISHED',
        interpretation='Only LOCAL linear trust-plan bound; NOT full-model infeasibility or a blocked goal. Diagnostic weights are NEVER candidates. LP solution uniqueness is NOT asserted.')
    try:
        check('no actual-model modules',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
        inputs=audit.manifest(out/'INPUT_HASHES.json'); outputs=audit.manifest(out/'OUTPUT_HASHES.json')
        expected={'INPUT_HASHES.json','PREREG.json','QPROBLEM.npz','CALL_LEDGER.json','SOLVER_STATUS.json',
                  'SOLUTION.npz','DIAGNOSTIC_WEIGHTS.npz','ALL_WEIGHT_DELTAS.csv','LP_CERTIFICATE.json','BOUND_REPORT.json','COMPLETE.json'}
        check('all outputs bound', all(audit.key(out/name) in outputs for name in expected))
        check('output paths local',all(Path(path).parent==out for path in outputs))
        check('no failed marker or candidate',not (out/'FAILED.json').exists() and not (out/'CANDIDATE_WEIGHTS.npz').exists())
        check('candidate not in output manifest',all(Path(path).name!='CANDIDATE_WEIGHTS.npz' for path in outputs))
        j,w,formal=verify_sources(audit,inputs,checks)
        pre=js(out/'PREREG.json'); report=js(out/'BOUND_REPORT.json'); complete=js(out/'COMPLETE.json')
        check('S140 prereg',pre['task']=='S140/P1' and pre['max_lp_calls']==1 and pre['no_retries'] is True
              and pre['no_rescale_after_result'] is True and pre['no_bound_or_mask_broadening'] is True)
        check('prereg manifest actual SHA',audit.key(pre['input_manifest'])==audit.key(out/'INPUT_HASHES.json')
              and pre['input_manifest_sha256']==audit.sha(out/'INPUT_HASHES.json'))
        for name,path in (('source_S136',S136),('source_S138',S138),('source_S139',S139)):
            check('registered '+name,audit.key(pre[name])==audit.key(path))
        check('registered trust and slack',pre['trust_component']==SCALE and pre['trust_L2']==.001 and pre['slack_scale']==1e-4)
        check('registered objective linear only',pre['objective']=='minimize y=t/1e-4; c=[0]*15+[1]; NO weight/slack quadratic regularization')
        eq('preregistered indices',pre['weight_indices'],INDICES)
        check('preregistered validity',pre['valid']==j['valid'].tolist())
        p=independent_problem(j,w,formal); saved=archive(out/'QPROBLEM.npz'); sol=archive(out/'SOLUTION.npz')
        keys=('A','lo','hi','bounds','initial','row_labels','c','G','h','G_row_labels','G_origin','G_side',
              'kept_indices','discarded_zero_indices','row_divisor','G_scaled','h_scaled')
        for key in keys: eq('independently rebuilt '+key,saved[key],p[key])
        old=archive(S138/'QP.npz')
        for key in ('A','lo','hi','bounds','initial','row_labels'): eq('same S138 exact constraints '+key,p[key],old[key])
        for key,value in dict(reference_weights=w,weight_indices=INDICES,valid=j['valid'],scale=np.asarray(SCALE),slack_scale=np.asarray(1e-4)).items():
            eq('LP metadata '+key,saved[key],value)
        check('no quadratic objective',not any(key in saved for key in ('H','hessian','quadratic')))
        z=sol['z']; lam=sol['lambda_raw']; marginal=sol['highs_marginals']
        check('marginal shape',marginal.shape==p['row_divisor'].shape and np.isfinite(marginal).all())
        eq('scaled dual from HiGHS',sol['lambda_scaled'],-marginal)
        expected_lam=np.zeros(len(p['h'])); expected_lam[p['kept_indices']]=-marginal/p['row_divisor']
        eq('raw dual unnormalization',lam,expected_lam)
        cert=certificate(p,z,lam)
        check('independent original raw LP certificate',cert['certificate_passed'])
        eq('raw original A values',sol['raw_A_value'],p['A']@z)
        eq('canonical original slack',sol['canonical_slack'],p['h']-p['G']@z)
        eq('stationarity vector',sol['stationarity'],p['c']+p['G'].T@lam)
        check('HiGHS inequality residual',np.allclose(sol['highs_inequality_residual'],p['h_scaled']-p['G_scaled']@z,rtol=1e-12,atol=1e-12))
        check('one successful solver result',js(out/'SOLVER_STATUS.json')['success'] is True and js(out/'SOLVER_STATUS.json')['status']==0)
        # Exact all121x42 predictions and all121 load; body rows are recomputed too.
        pred=predictions(j,z)
        eq('all121x42 joint prediction',sol['predicted_joint_position'],pred['signed_joint_position'])
        eq('all121 load prediction',sol['predicted_load_ratio'],pred['load_ratio'])
        eq('all121 saved load delta',sol['delta_load_ratio'],pred['delta_load_ratio'])
        delta=np.zeros_like(w); delta[INDICES[:,0],INDICES[:,1]]=SCALE*z[:15]
        weights=w+delta; saved_delta=weights-w
        eq('diagnostic exact weight addition',sol['diagnostic_weights'],weights)
        eq('diagnostic NPZ',archive(out/'DIAGNOSTIC_WEIGHTS.npz')['weights'],weights)
        eq('saved rounded delta',sol['delta'],saved_delta)
        actual=z.copy(); actual[:15]=saved_delta[INDICES[:,0],INDICES[:,1]]/SCALE
        eq('actual weight coordinates',sol['actual_weight_z'],actual)
        allowed=np.zeros_like(w,dtype=bool); allowed[INDICES[:,0],INDICES[:,1]]=j['valid']
        check('diagnostic weight restrictions',np.all((weights>=0)&(weights<=4)) and exact(weights[~allowed],w[~allowed])
              and exact(weights==0,w==0) and exact(weights[36],w[36]) and np.all(weights[w>0]>0)
              and np.all(z[:15][~j['valid']]==0.) and np.linalg.norm(saved_delta)<=.001+1e-15
              and np.max(np.abs(saved_delta))<=SCALE+1e-15)
        check('rounded diagnostic primal',max(0.,float(np.max(p['lo']-p['A']@actual)),float(np.max(p['A']@actual-p['hi'])),
              float(np.max(p['bounds'][:,0]-actual)),float(np.max(actual-p['bounds'][:,1])))<=THRESHOLDS['primal'])
        relative_ms=(j['time']-j['time'][0])*1000.; early=relative_ms<=1.7+1e-10
        deficit=np.maximum(.8-pred['load_ratio'][early],0.)
        integral=float(np.sum(.5*(deficit[:-1]+deficit[1:])*np.diff(relative_ms[early])))
        check('protected early load and bounded post exchange',pred['load_ratio'][:77].min()>=.8-1e-9
              and pred['delta_load_ratio'][77:].min()>=-.001-1e-9 and integral<=1e-9)
        s=formal['safety_exact_bounds']; peak=float(pred['joint_excess'].max()); slack=1e-4*float(z[-1])
        check('joint formal plus explicit slack',peak<=p['formal_limit']+slack+1e-9)
        check('all121 body formal',pred['height'].min()>=s['extrema']['height_ratio']['min']-1e-9
              and pred['root_angle_rad'].max()<=s['extrema']['angle_rad']['max']+1e-9
              and pred['com_vertical_velocity'].min()>=-s['max_downward_COM_velocity']-1e-9
              and pred['qvel'][:,2].min()>=-s['max_downward_root_velocity']-1e-9)
        for key,value in dict(y_star=float(z[-1]),slack_star=slack,best_predicted_peak=peak,formal_limit=p['formal_limit'],
                predicted_local_gap=peak-p['formal_limit'],initial_peak=p['initial_peak'],initial_slack=p['initial_margin'],
                slack_improvement=p['initial_margin']-slack,used_norm=float(np.linalg.norm(saved_delta)),
                used_component=float(np.max(np.abs(saved_delta)))).items(): check('report '+key,report[key]==value)
        metrics=report['predicted_load']
        eq('reported all121 load',np.asarray(metrics['predicted_load_ratio']),pred['load_ratio'])
        eq('reported all121 delta',np.asarray(metrics['delta_load_ratio']),pred['delta_load_ratio'])
        failures=np.flatnonzero(pred['load_ratio']<.8); first=int(failures[0]) if len(failures) else None
        expected_metrics=dict(predicted_load_ratio=pred['load_ratio'].tolist(),delta_load_ratio=pred['delta_load_ratio'].tolist(),
            first_load_failure_sample=first,first_load_failure_ms=None if first is None else float(relative_ms[first]),
            load_failure_count=int(len(failures)),protected_sample_count=77,post_sample_count=44,
            pre_failure_min=float(pred['load_ratio'][:77].min()),post_delta_min=float(pred['delta_load_ratio'][77:].min()),
            all121_delta_min=float(pred['delta_load_ratio'].min()),early_deficit_integral0_1p7_ratio_ms=integral,
            first_failure_not_earlier77=bool(first is None or first>=77),protected_load_floor=bool(np.all(pred['load_ratio'][:77]>=.8-1e-9)),
            early_deficit_zero=bool(integral==0.),post_bounded_exchange=bool(np.all(pred['delta_load_ratio'][77:]>=-.001-1e-9)),
            physical_all17='UNKNOWN_NOT_EVALUATED',physical_first_failure_counts='UNKNOWN_NOT_EVALUATED',physical_support_and_longest='UNKNOWN_NOT_EVALUATED')
        check('all predicted load metrics exact',metrics==expected_metrics)
        producer_cert=js(out/'LP_CERTIFICATE.json')
        check('saved producer certificate consistent',report['certificate']==producer_cert and producer_cert['certificate_passed'] is True)
        check('producer original primal and dual numeric',producer_cert['raw_primal_violation']==cert['numbers']['raw_primal_violation']
              and producer_cert['primal_objective']==cert['numbers']['primal_objective']
              and producer_cert['dual_objective']==cert['numbers']['dual_objective'])
        with (out/'ALL_WEIGHT_DELTAS.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
        check('full330 weight CSV',len(rows)==330)
        for row,(pool,b) in zip(rows,np.ndindex(w.shape)):
            check('CSV '+str((pool,b)),int(row['pool'])==pool and int(row['bin'])==b and float(row['reference'])==w[pool,b]
                  and float(row['diagnostic'])==weights[pool,b] and float(row['delta'])==saved_delta[pool,b] and row['allowed']==str(bool(allowed[pool,b])))
        ledger=dict(lp_calls=1,model_load=0,forward=0,step=0,neural_ticks=0)
        check('single LP zero physical ledger',js(out/'CALL_LEDGER.json')==ledger==complete['ledger'])
        check('complete saved state',complete['status']=='COMPLETE' and complete['independent_verification']=='PENDING')
        for record in (pre,report,complete):
            check('no physical or candidate promotion',record['physical_admission'] is False and record['candidate_promotion'] is False)
        # Static inspection of the actual bound producer, without importing it.
        producer=ROOT/'bound_trajectory_safety_feasibility.py'
        check('producer source input-bound',inputs.get(audit.key(producer))==audit.sha(producer))
        tree=ast.parse(producer.read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        check('producer no actual-model imports',not any(n.split('.')[0] in {'mujoco','torch','engine','subprocess'} for n in imports))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='linprog']
        check('single static all-free HiGHS call',len(calls)==1 and any(k.arg=='bounds' and ast.literal_eval(k.value.left)==[(None,None)]
              and ast.literal_eval(k.value.right)==16 for k in calls[0].keywords if isinstance(k.value,ast.BinOp))
              and any(k.arg=='method' and ast.literal_eval(k.value)=='highs' for k in calls[0].keywords))
        lower=cert['numbers']['box_corrected_dual_lower_bound']
        positive=bool(lower>0.)
        result.update(passed=True,evidence_passed=True,certificate=cert,
            local_zero_slack_excluded=positive,strict_positive_local_linear_bound=positive,
            local_y_lower_bound=lower,local_slack_lower_bound=1e-4*lower,
            input_count=len(inputs),output_count=len(outputs),actual_SHA_paths_once=len(audit.digests),
            predictions=dict(peak=peak,slack=slack,early_deficit_integral=integral,
                protected_load_min=float(pred['load_ratio'][:77].min()),post_delta_min=float(pred['delta_load_ratio'][77:].min())),
            solver_success_provenance='Saved HiGHS status plus independently reconstructed original raw primal/dual certificate; no solve rerun',
            uniqueness_asserted=False,full_model_infeasible=False,goal_blocked=False)
    except Exception:
        result['error']=traceback.format_exc()
    result['checks']=checks
    result['helper_source_hashes']={str(Path(m.__file__).resolve()):audit.sha(m.__file__) for n,m in tuple(sys.modules.items())
        if n.startswith('verify_') and getattr(m,'__file__',None)}
    with target.open('x') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False); stream.write('\n')
    print(json.dumps({key:result[key] for key in ('passed','evidence_passed','proposal_admitted','candidate_admitted')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args().output))
