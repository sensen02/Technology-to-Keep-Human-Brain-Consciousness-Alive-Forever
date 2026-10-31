"""S138/P1 saved-byte independent verification; inert import, no weight optimization.

Evidence validity is separate from admission. Safety-output Jacobian only: no
claim of general qacc-gradient validity or any physical/standing success.
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

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收/s138-formal-benefit-safety-proposal').resolve()
SOURCE = Path('/home/sensen/Desktop/engine修复验收/s136-safety-trajectory-jacobian/20261024T183631Z').resolve()
PREVIOUS = Path('/home/sensen/Desktop/engine修复验收/s137-trajectory-safety-proposal/20261024T191248Z').resolve()
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms/TRAJECTORY.npz')
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
INDICES = np.asarray([(p,b) for p in (3,15,38) for b in range(5)])
SCALE = .001 / np.sqrt(15)
TOL = 1e-9
TARGET = 'PARENT_FORMAL_BENEFIT_PROPOSAL_INDEPENDENT_CHECK.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()


def js(path): return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}


def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def canonical(data):
    result={}
    for name,digest in data.items():
        key=str(Path(name).resolve())
        assert key not in result or result[key]==digest, 'Conflicting canonical path aliases'
        result[key]=digest
    return result


def manifest(path):
    data=js(path)
    assert isinstance(data,dict) and data
    for name,digest in data.items():
        assert Path(name).is_absolute() and len(digest)==64 and sha(name)==digest, 'Hash mismatch: '+name
    return canonical(data)


def rebuild(j,w,formal):
    """Independently assemble all10769 raw rows, all16 Hessian coordinates."""
    s=formal['safety_exact_bounds']; limit=s['extrema']['limit_excess']['max']
    q=j['reference_signed_joint_position']; qb=j['joint_bounds']
    peak=float(np.maximum(np.maximum(qb[:,0]-q,q-qb[:,1]),0.).max()); margin=peak-limit
    assert margin>0
    pieces=[]; lows=[]; highs=[]; labels=[]
    def add(a,lo,hi,label):
        a=np.asarray(a).reshape(-1,16); n=len(a)
        pieces.append(a); lows.append(np.broadcast_to(lo,(n,)).copy()); highs.append(np.broadcast_to(hi,(n,)).copy())
        labels.extend(label+':'+str(i) for i in range(n))
    def tangent(key):
        d=np.where(j['valid'],j[key+'_D_half'],0.)*SCALE
        return np.column_stack((d.reshape(-1,15),np.zeros(d.size//15)))
    lower=tangent('signed_joint_position'); lower[:,-1]=1e-4
    upper=tangent('signed_joint_position'); upper[:,-1]=-1e-4
    add(lower,(np.broadcast_to(qb[:,0]-limit,q.shape)-q).ravel(),np.inf,'all121x42_joint_lower_plus_t')
    add(upper,-np.inf,(np.broadcast_to(qb[:,1]+limit,q.shape)-q).ravel(),'all121x42_joint_upper_minus_t')
    load_rows=tangent('load_ratio')
    add(load_rows[:77],.8-j['reference_load_ratio'][:77],np.inf,'samples0to76_formal_benefit_load_floor')
    add(load_rows[77:],-.001,np.inf,'samples77to120_bounded_load_exchange')
    add(tangent('height'),s['extrema']['height_ratio']['min']-j['reference_height'],np.inf,'all121_formal_height')
    add(tangent('root_angle_rad'),-np.inf,s['extrema']['angle_rad']['max']-j['reference_root_angle_rad'],'all121_formal_angle')
    add(tangent('com_vertical_velocity'),-s['max_downward_COM_velocity']-j['reference_com_vertical_velocity'],np.inf,'all121_formal_COM_vz')
    add(tangent('qvel').reshape(121,48,16)[:,2],-s['max_downward_root_velocity']-j['reference_qvel'][:,2],np.inf,'all121_formal_root_vz')
    bounds=np.tile([-1.,1.],(16,1))
    for k,(pool,b) in enumerate(INDICES):
        if not j['valid'][k]: bounds[k]=0.
        bounds[k,0]=max(bounds[k,0],-w[pool,b]/SCALE)
        bounds[k,1]=min(bounds[k,1],(4.-w[pool,b])/SCALE)
        if w[pool,b]==0.: bounds[k]=0.
    bounds[-1]=[0.,margin/1e-4]
    return dict(A=np.vstack(pieces),lo=np.concatenate(lows),hi=np.concatenate(highs),bounds=bounds,
                H=.1*np.eye(16),g=np.r_[np.zeros(15),1.],initial=np.r_[np.zeros(15),margin/1e-4],
                row_labels=np.asarray(labels),initial_peak=peak,initial_margin=margin,formal_limit=limit)


def residual(p,z):
    v=p['A']@z
    return max(0.,float(np.max(p['lo']-v)),float(np.max(v-p['hi'])),
               float(np.max(p['bounds'][:,0]-z)),float(np.max(z-p['bounds'][:,1])))


def main(output):
    out=Path(output).resolve(); target=out/TARGET
    assert out.parent==BASE and out.is_dir() and not target.exists(), 'New S138 root only; no overwrite'
    checks=[]
    def check(name,condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def eq(name,a,b): check(name,exact(a,b))
    result=dict(passed=False,evidence_passed=False,proposal_admitted=False,candidate_admitted=False,
        physical_admission=False,promotion=False,checkpoint_replacement=False,standing_or_walking_claim=False,
        safety_output_scope_only=True,general_gradient_verified=False,units='NOT_ESTABLISHED',
        verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),
        new_calls=dict(model_load=0,forward=0,step=0,neural_ticks=0,weight_optimizer=0),
        interpretation='Faithful rejected proposal can pass evidence verification; no physical S119 gates are established.')
    try:
        inputs=manifest(out/'INPUT_HASHES.json'); outputs=manifest(out/'OUTPUT_HASHES.json')
        expected={'PREREG.json','INPUT_HASHES.json','QP.npz','CALL_LEDGER.json','SOLUTION.npz',
                  'ALL_WEIGHT_DELTAS.csv','QP_CERTIFICATE.json','PROPOSAL_REPORT.json','COMPLETE.json'}
        check('mandatory outputs bound',all(str(out/n) in outputs for n in expected))
        check('producer manifest paths local',all(Path(n).parent==out for n in outputs))
        check('no failure output',not (out/'FAILED.json').exists())
        pre=js(out/'PREREG.json'); report=js(out/'PROPOSAL_REPORT.json'); complete=js(out/'COMPLETE.json')
        check('registered S136 source',Path(pre['source']).resolve()==SOURCE)
        check('preregistered hashes exact',canonical(pre['source_hashes'])==inputs)
        source_in=manifest(SOURCE/'INPUT_HASHES.json'); source_out=manifest(SOURCE/'OUTPUT_HASHES.json')
        check('S136 prereg inputs',canonical(js(SOURCE/'PREREG.json')['source_hashes'])==source_in)
        check('S136 complete',js(SOURCE/'COMPLETE.json')['status']=='COMPLETE' and not (SOURCE/'FAILED.json').exists())
        verification=Path(pre['verification']).resolve(); verified=js(verification)
        check('S136 verification local',verification.parent==SOURCE)
        check('S136 safety scope verified',verified['passed'] is True and verified['evidence_passed'] is True
              and verified['safety_output_scope_only'] is True and verified['general_gradient_verified'] is False)
        check('S136 verifier identity',Path(verified['verifier']).resolve()==ROOT/'verify_safety_trajectory_jacobian.py'
              and sha(verified['verifier'])==verified['verifier_sha256'])
        required=dict(source_in); required.update(source_out); required.update(verified['helper_source_hashes'])
        old_inputs=manifest(PREVIOUS/'INPUT_HASHES.json'); old_outputs=manifest(PREVIOUS/'OUTPUT_HASHES.json')
        old_check=PREVIOUS/'PARENT_SAFETY_PROPOSAL_INDEPENDENT_CHECK.json'; old=js(old_check)
        check('S137 evidence passed but rejected',old['passed'] is True and old['evidence_passed'] is True
              and old['proposal_admitted'] is False and old['candidate_admitted'] is False
              and js(PREVIOUS/'PROPOSAL_REPORT.json')['proposal_accepted'] is False
              and not (PREVIOUS/'CANDIDATE_WEIGHTS.npz').exists())
        check('S137 verifier identity',Path(old['verifier']).resolve()==ROOT/'verify_trajectory_safety_proposal.py'
              and sha(old['verifier'])==old['verifier_sha256'])
        required.update(old_inputs); required.update(old_outputs); required.update(old['helper_source_hashes'])
        for path in (SOURCE/'OUTPUT_HASHES.json',verification,Path(verified['verifier']),REFERENCE,FORMAL,
                     PREVIOUS/'OUTPUT_HASHES.json',old_check,Path(old['verifier']),PREVIOUS/'PARENT_CONCLUSION.zh-CN.md',
                     ROOT/'propose_formal_benefit_safety_step.py',ROOT/'verify_margin_contact_candidate.py'):
            required[str(path)]=sha(path)
        for path in PREVIOUS.rglob('*'):
            if path.is_file(): required[str(path.resolve())]=sha(path)
        check('full S136 and S137 proof and formal hashes bound',all(inputs.get(k)==v for k,v in canonical(required).items()))
        j=archive(SOURCE/'JACOBIAN.npz'); w=archive(REFERENCE)['weights']; formal=js(FORMAL)
        eq('indices',j['weight_indices'],INDICES); eq('reference W',j['reference_weights'],w)
        check('W range',w.shape==(66,5) and np.isfinite(w).all() and np.all((w>=0)&(w<=4)))
        check('11 valid 4 frozen',j['valid'].shape==(15,) and j['valid'].dtype==bool
              and j['valid'].tolist()==verified['valid_columns'] and int(j['valid'].sum())==verified['valid_count']==11)
        check('independent column mask',j['valid'].tolist()==[bool(c['valid']) for c in verified['columns']])
        check('source reported mask',js(SOURCE/'DERIVATIVE_REPORT.json')['valid_columns']==j['valid'].tolist())
        check('source shapes',j['time'].shape==(121,) and j['joint_bounds'].shape==(42,2))
        check('FD scales',float(j['h'])==.000125 and float(j['half'])==.0000625)
        for key,shape in dict(signed_joint_position=(121,42),load_ratio=(121,),height=(121,),root_angle_rad=(121,),com_vertical_velocity=(121,),qvel=(121,48)).items():
            check('finite source '+key,j['reference_'+key].shape==shape and j[key+'_D_half'].shape==(*shape,15)
                  and np.isfinite(j['reference_'+key]).all() and np.isfinite(j[key+'_D_half']).all())
            eq('independent mask '+key,j['masked_'+key],np.where(j['valid'],j[key+'_D_half'],0.))
        check('pre constants',pre['task']=='S138/P1' and pre['trust_L2']==.001 and pre['trust_component']==SCALE
              and pre['slack_scale']==1e-4 and pre['min_slack_improvement']==1e-8 and pre['max_qp_calls']==1)
        check('original formal unchanged',pre['formal_constraints']==formal)
        eq('pre validity',pre['valid_columns'],j['valid']); eq('pre indices',pre['weight_indices'],INDICES)
        first=np.flatnonzero(j['reference_load_ratio']<.8)
        check('reference first load failure77',len(first)>0 and int(first[0])==77)
        check('original early deficit zero',formal['benefit_exact_bounds']['load_deficit_integral0_1p7_ratio_ms']==0.)
        eq('time reference exact',j['time'],archive(REFERENCE)['time'])
        relative_ms=(j['time']-j['time'][0])*1000.
        check('time grid',np.isfinite(relative_ms).all() and np.all(np.diff(relative_ms)>0.)
              and np.allclose(relative_ms,np.arange(121)*.025,rtol=0.,atol=1e-11))
        check('original load failure time',abs(relative_ms[77]-formal['benefit_exact_bounds']['first_failures_ms']['load'])<1e-10)
        plan=dict(reference_first_load_failure_sample=77,reference_first_load_failure_ms=float(relative_ms[77]),
            pre_failure_samples=[0,76],pre_failure_predicted_load_min=.8,early_window_ms=[0.,1.7],early_deficit_target=0.,
            post_failure_samples=[77,120],post_delta_load_min=-.001,all121_load_rows=121,explicit_bounded_planning_exchange=True,
            replaces='S137 all121 delta_load >= -1e-9 planning-only guard',original_17_physical_admission_tests='UNCHANGED_NOT_EVALUATED',
            initial_load_support_longest_contacts_first_safety_failures='UNKNOWN_UNTIL_PHYSICAL_TEST')
        check('exact preregistered load plan',pre['early_postload']==plan==report['early_postload'])
        check('unchanged physical17 scope',pre['original_physical_tests']==17 and Path(pre['predecessor']).resolve()==PREVIOUS
              and report['original_17_physical_tests']=='UNCHANGED_UNKNOWN_NOT_EVALUATED')
        p=rebuild(j,w,formal); qp=archive(out/'QP.npz'); sol=archive(out/'SOLUTION.npz')
        check('full10769x16',p['A'].shape==(10769,16))
        for key,value in p.items(): eq('independent QP '+key,qp[key],value)
        for key,value in dict(weight_indices=INDICES,valid=j['valid'],reference_weights=w,scale=np.asarray(SCALE),slack_scale=np.asarray(1e-4)).items(): eq('QP metadata '+key,qp[key],value)
        check('baseline extrema',pre['baseline_peak']==p['initial_peak'] and pre['baseline_margin']==p['initial_margin'])
        check('initial feasible',residual(p,p['initial'])<=TOL)
        z=sol['z']; actual=sol['actual_z']; candidate=sol['proposed_weights']
        check('solution shape',z.shape==actual.shape==(16,) and candidate.shape==w.shape)
        check('solution finite',np.isfinite(z).all() and np.isfinite(actual).all() and np.isfinite(candidate).all())
        delta=np.zeros_like(w); delta[INDICES[:,0],INDICES[:,1]]=SCALE*z[:15]
        eq('exact forward addition',candidate,w+delta); saved_delta=candidate-w
        eq('actual saved delta',sol['delta'],saved_delta)
        expected_actual=z.copy(); expected_actual[:15]=saved_delta[INDICES[:,0],INDICES[:,1]]/SCALE
        eq('actual roundoff decision',actual,expected_actual)
        eq('solution gradient',sol['gradient'],p['H']@actual+p['g'])
        rowmax=np.max(np.abs(p['A']),axis=1); scaling=np.where(rowmax>0.,1./np.maximum(rowmax,1e-300),1.)
        eq('row normalization',sol['row_scale'],scaling); eq('raw values',sol['raw_value'],p['A']@actual)
        pred=j['reference_signed_joint_position']+np.einsum('tjk,k->tj',np.where(j['valid'],j['signed_joint_position_D_half'],0.),SCALE*actual[:15])
        eq('predicted all q',sol['predicted_joint_position'],pred)
        qb=j['joint_bounds']; peak=float(np.maximum(np.maximum(qb[:,0]-pred,pred-qb[:,1]),0.).max()); slack=1e-4*actual[-1]
        load_delta=np.where(j['valid'],j['load_ratio_D_half'],0.)@(SCALE*actual[:15])
        predicted_load=j['reference_load_ratio']+load_delta
        saved_load_delta=predicted_load-j['reference_load_ratio']
        eq('solution predicted load',sol['predicted_load_ratio'],predicted_load)
        eq('solution roundoff load delta',sol['delta_load_ratio'],saved_load_delta)
        early=relative_ms<=1.7+1e-10
        deficit=np.maximum(.8-predicted_load[early],0.)
        integral=float(np.sum(.5*(deficit[:-1]+deficit[1:])*np.diff(relative_ms[early])))
        failures=np.flatnonzero(predicted_load<.8); first_pred=int(failures[0]) if len(failures) else None
        metrics=dict(predicted_load_ratio=predicted_load.tolist(),delta_load_ratio=saved_load_delta.tolist(),
            first_load_failure_sample=first_pred,first_load_failure_ms=None if first_pred is None else float(relative_ms[first_pred]),
            load_failure_count=int(len(failures)),protected_sample_count=77,post_sample_count=44,
            pre_failure_min=float(predicted_load[:77].min()),post_delta_min=float(saved_load_delta[77:].min()),
            all121_delta_min=float(saved_load_delta.min()),early_deficit_integral0_1p7_ratio_ms=integral,
            first_failure_not_earlier77=bool(first_pred is None or first_pred>=77),
            protected_load_floor=bool(np.all(predicted_load[:77]>=.8-TOL)),early_deficit_zero=bool(integral==0.),
            post_bounded_exchange=bool(np.all(saved_load_delta[77:]>=-.001-TOL)),
            physical_all17='UNKNOWN_NOT_EVALUATED',physical_first_failure_counts='UNKNOWN_NOT_EVALUATED',
            physical_support_and_longest='UNKNOWN_NOT_EVALUATED')
        check('all independent predicted load metrics exact',report['predicted_load_benefit']==metrics)
        from verify_margin_contact_candidate import qp_certificate
        cert=qp_certificate(p['H'],p['g'],p['A'],p['lo'],p['hi'],actual,p['bounds'])
        check('independent certificate exact',cert==js(out/'QP_CERTIFICATE.json')==report['certificate'])
        allowed=np.zeros_like(w,dtype=bool); allowed[INDICES[:,0],INDICES[:,1]]=j['valid']
        gates=dict(solver_success=bool(report['solver_success']),finite=bool(np.isfinite(actual).all()),
            raw_primal=bool(residual(p,actual)<=TOL),certificate=bool(cert['certificate_passed']),
            slack_reduced=bool(p['initial_margin']-slack>=1e-8),predicted_peak_reduced=bool(p['initial_peak']-peak>=1e-8),
            peak_within_formal_plus_explicit_slack=bool(peak<=p['formal_limit']+slack+TOL),
            weights_in_bounds=bool(np.all((candidate>=0)&(candidate<=4))),zero_mask=exact(candidate==0,w==0),
            frozen=exact(candidate[~allowed],w[~allowed]),pool36=exact(candidate[36],w[36]),
            invalid_delta_zero=bool(np.all(actual[:15][~j['valid']]==0)),L2=bool(np.linalg.norm(saved_delta)<=.001+1e-15),
            component=bool(np.max(np.abs(saved_delta))<=SCALE+1e-15),
            no_physical_runtime=not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')),source_rehash=True)
        gates.update(first_load_failure_not_earlier77=metrics['first_failure_not_earlier77'],
            pre_failure_load_floor=metrics['protected_load_floor'],early_deficit_zero=metrics['early_deficit_zero'],
            post_bounded_load_exchange=metrics['post_bounded_exchange'])
        manifest(out/'INPUT_HASHES.json')
        check('all20 producer postgates exact',len(gates)==20 and report['gates']==gates)
        numbers=dict(objective=float(.5*actual@p['H']@actual+p['g']@actual),initial_peak=p['initial_peak'],formal_limit=p['formal_limit'],
            predicted_peak=peak,initial_slack=p['initial_margin'],predicted_slack=slack,
            slack_improvement=p['initial_margin']-slack,primal_violation=residual(p,actual))
        for key,value in numbers.items(): check('report number '+key,report[key]==value)
        eq('report gradient',report['gradient'],p['H']@actual+p['g'])
        admitted=all(gates.values())
        check('admission iff all gates',report['proposal_accepted'] is admitted and complete['proposal_accepted'] is admitted)
        check('candidate iff admitted',(out/'CANDIDATE_WEIGHTS.npz').exists()==admitted)
        check('candidate hash iff admitted',(str(out/'CANDIDATE_WEIGHTS.npz') in outputs)==admitted)
        if admitted: eq('exact candidate',archive(out/'CANDIDATE_WEIGHTS.npz')['weights'],candidate)
        with (out/'ALL_WEIGHT_DELTAS.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
        check('330 CSV rows',len(rows)==330)
        for row,(pool,b) in zip(rows,np.ndindex(w.shape)):
            check('CSV '+str((pool,b)),int(row['pool'])==pool and int(row['bin'])==b and float(row['reference'])==w[pool,b]
                  and float(row['proposed'])==candidate[pool,b] and float(row['delta'])==saved_delta[pool,b] and row['allowed']==str(bool(allowed[pool,b])))
        ledger=dict(qp_calls=1,model_load=0,forward=0,step=0,neural_ticks=0)
        check('one QP no physics ledger',js(out/'CALL_LEDGER.json')==ledger==complete['ledger'])
        check('completed scope',complete['status']=='COMPLETE' and complete['independent_verification']=='PENDING')
        for record in (pre,report,complete): check('no physical promotion',record['physical_admission'] is False and record['promotion'] is False)
        check('no checkpoint',pre['checkpoint_replacement'] is False and report['checkpoint_replacement'] is False)
        tree=ast.parse((ROOT/'propose_formal_benefit_safety_step.py').read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        check('producer static no physical imports',not any(n.split('.')[0] in {'mujoco','torch','engine','subprocess'} for n in imports))
        manifest(out/'OUTPUT_HASHES.json'); manifest(out/'INPUT_HASHES.json')
        result.update(passed=True,evidence_passed=True,proposal_admitted=admitted,candidate_admitted=admitted,
            gates=gates,certificate=cert,numbers=numbers,source=str(SOURCE),input_count=len(inputs),output_count=len(outputs),
            solver_success_provenance='Saved status, no optimizer rerun; full raw-matrix certificate independently recomputed',
            runtime_scope_provenance='Saved zero-call ledger and hash-bound static imports, not proof against unrecorded events',
            limitations=pre['limitation'],required_next='Only if admitted: separate original S119 full17 physical evaluation unchanged; not performed here')
    except Exception:
        result['error']=traceback.format_exc()
    result['checks']=checks
    result['helper_source_hashes']={str(Path(m.__file__).resolve()):sha(m.__file__) for n,m in tuple(sys.modules.items()) if n.startswith('verify_') and getattr(m,'__file__',None)}
    with target.open('x') as stream: json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False); stream.write('\n')
    print(json.dumps({k:result[k] for k in ('passed','evidence_passed','proposal_admitted','physical_admission')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args().output))
