"""S132 saved-byte independent verification; NumPy/SciPy only, no simulator.

Evidence integrity, proposal admission and convex-QP certification are distinct.
NNLS recovers diagnostic nonnegative multipliers at the frozen proposal; it never
optimizes or changes weights. Import inert, including when used by the producer.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
from scipy.optimize import nnls
from verify_contact_aware_activation_response import (
    ROOT, BASE, REFERENCE, S124, STATES, Checks, sha, js, archive,
    activation_tangent, criterion, contact_load, independent_gain,
)

SCALE = .002/np.sqrt(15)
TOL = 1e-9
S130 = BASE/'s130-contact-aware-activation-response/20261024T175612Z'
INDICES = np.array([(p,b) for p in (3,15,38) for b in range(5)])


def qp_certificate(H, g, A, lo, hi, x, bounds=None):
    """Certify a *given* x using maxabs-normalized inequalities and NNLS.

    Zero coefficient rows are excluded from the dual system but their original
    primal inequalities remain mandatory. Bounds are explicit dual rows. Both
    sides of equality/tightly active two-sided rows are independently included.
    """
    H, g, A, x = map(lambda a: np.asarray(a,dtype=float), (H,g,A,x))
    lo, hi = np.asarray(lo,dtype=float), np.asarray(hi,dtype=float)
    assert H.shape == (len(x),len(x)) and g.shape == x.shape
    assert A.shape == (len(lo),len(x)) and lo.shape == hi.shape
    assert np.isfinite(H).all() and np.isfinite(g).all() and np.isfinite(A).all() and np.isfinite(x).all()
    assert not np.isnan(lo).any() and not np.isnan(hi).any() and np.all(lo<=hi)
    residual = max(0., float(np.max(lo-A@x,initial=0)), float(np.max(A@x-hi,initial=0)))
    if bounds is not None:
        bounds = np.asarray(bounds,dtype=float)
        assert bounds.shape == (len(x),2) and np.all(bounds[:,0]<=bounds[:,1])
        residual = max(residual,float(np.max(bounds[:,0]-x,initial=0)),float(np.max(x-bounds[:,1],initial=0)))
        A = np.vstack((A,np.eye(len(x))))
        lo,hi = np.r_[lo,bounds[:,0]],np.r_[hi,bounds[:,1]]
    norm = np.max(np.abs(A),axis=1)
    nonzero = norm>0
    B = A[nonzero]/norm[nonzero,None]
    lower,upper = lo[nonzero]/norm[nonzero],hi[nonzero]/norm[nonzero]
    R = np.vstack((B[np.isfinite(upper)],-B[np.isfinite(lower)]))
    rhs = np.r_[upper[np.isfinite(upper)],-lower[np.isfinite(lower)]]
    slack = rhs-R@x
    active = np.flatnonzero(slack<=1e-7)
    grad = H@x+g
    lam = np.zeros(len(R)); failure = None
    try:
        if len(active):
            lam[active],_ = nnls(R[active].T,-grad,maxiter=max(10000,10*len(active)))
    except (RuntimeError,ValueError) as exc:
        failure = str(exc)
    station = float(np.max(np.abs(grad+R.T@lam),initial=0))
    comp = float(np.max(np.abs(lam*slack),initial=0))
    eig = float(np.linalg.eigvalsh((H+H.T)/2).min())
    symmetric = bool(np.max(np.abs(H-H.T),initial=0)<=1e-12)
    proven = bool(failure is None and symmetric and eig>0 and residual<=TOL and station<=1e-5 and comp<=1e-7)
    return dict(strict_convex=bool(symmetric and eig>0),min_eigenvalue=eig,
                primal_violation=residual,stationarity_max_abs=station,
                dual_violation=float(np.max(-lam,initial=0)),complementarity_max_abs=comp,
                kkt_verified=proven,unique_qp=proven,certificate_passed=proven,
                active_tolerance=1e-7,stationarity_tolerance=1e-5,complementarity_tolerance=1e-7,
                normalized_rows=len(R),active_rows=len(active),discarded_zero_rows=int((~nonzero).sum()),
                multiplier_method='scipy.optimize.nnls diagnostic only; no weight optimization',
                nnls_failure=failure)


def checked_manifest(path):
    literal = js(path)
    if 'before' in literal:
        assert literal['before']==literal['after'], 'Changed committed input'
        literal = literal['before']
    assert isinstance(literal,dict) and literal
    canonical = {}
    for name,digest in literal.items():
        p=Path(name)
        assert p.is_absolute() and isinstance(digest,str) and len(digest)==64
        key=str(p.resolve())
        assert key not in canonical or canonical[key]==digest, 'Conflicting canonical aliases'
        assert sha(p)==digest, 'Hash mismatch '+name
        canonical[key]=digest
    return literal,canonical


def branch(raw):
    return np.where(raw<=0,-1,np.where(raw>=1,1,0))


def linear_problem(Q,L,features,raw,ctrl,pool,W):
    """Independently construct S132 native planning inequalities, not actual caps."""
    draw=np.zeros((166,66,15)); dc=np.zeros((166,90,15))
    for j,(p,b) in enumerate(INDICES): draw[:,p,j]=features[:,p,b]
    mapped=pool>=0
    dc[:,mapped]=draw[:,pool[mapped]]*((raw[:,pool[mapped]]>0)&(raw[:,pool[mapped]]<1))[:,:,None]
    mats=[]; lows=[]; highs=[]
    def add(a,l,h):
        a=np.asarray(a).reshape(-1,15); mats.append(a)
        lows.extend(np.broadcast_to(l,(len(a),))); highs.extend(np.broadcast_to(h,(len(a),)))
    add(L*SCALE,0,np.inf)
    add(Q[:,6:8]*SCALE,-.99,.99)
    add(Q*SCALE,-1.98,1.98)
    add(dc*SCALE,-.005,.005)
    add(draw*SCALE,np.where(raw<=0,-np.inf,np.where(raw>=1,1,0)).ravel()-raw.ravel(),
        np.where(raw<=0,0,np.where(raw>=1,np.inf,1)).ravel()-raw.ravel())
    wb=W[INDICES[:,0],INDICES[:,1]]
    bounds=np.stack((np.maximum(-1,-wb/SCALE),np.minimum(1,(4-wb)/SCALE)),axis=1)
    return np.vstack(mats),np.asarray(lows),np.asarray(highs),bounds


def main(out):
    out=Path(out).resolve(); target=out/'PARENT_INDEPENDENT_CHECK_FINAL.json'
    assert not target.exists(), 'Never overwrite prior evidence'
    assert 'mujoco' not in sys.modules
    ilit,inputs=checked_manifest(out/'INPUT_HASHES.json')
    olit,outputs=checked_manifest(out/'OUTPUT_HASHES.json')
    expected={str(p.resolve()) for p in out.iterdir() if p.is_file() and p.name not in
              {'OUTPUT_HASHES.json','PARENT_INDEPENDENT_CHECK.json',target.name}}
    assert expected==set(outputs), 'Incomplete output manifest coverage'
    c=Checks(); pre=js(out/'PREREG.json'); summary=js(out/'SUMMARY.json')
    assert pre['samples']==STATES and pre['allowed_pools']==[3,15,38] and pre['entries']==15
    assert pre['scale']==SCALE and pre['regularization']==.05
    assert pre['predicted_constraints']['per_sample_abs_delta_qacc67_upper']==.99
    assert pre['predicted_constraints']['per_sample_all48_abs_delta_qacc_upper']==1.98
    assert pre['predicted_constraints']['per_sample_delta_load_ratio_lower']==0.
    assert pre['tolerances']['load_lower_numeric']==TOL and pre['tolerances']['safety_caps']==TOL
    required=[REFERENCE/'TRAJECTORY.npz',REFERENCE/'INITIALIZATION.npz',REFERENCE/'CONSTRAINT_ROWS.npz',
              REFERENCE/'COM_CONTACT_METADATA.json',S124/'ACTUATOR_XML_RESOLUTION.json',
              S130/'FIXED_STATE_CHAIN.npz',S130/'RAW_PROBES.npz']
    for p in required: assert inputs.get(str(p.resolve()))==sha(p), 'Missing required input '+str(p)
    z=archive(REFERENCE/'TRAJECTORY.npz'); ini=archive(REFERENCE/'INITIALIZATION.npz')
    historical=archive(REFERENCE/'CONSTRAINT_ROWS.npz'); oldraw=archive(S130/'RAW_PROBES.npz')
    chain=archive(S130/'FIXED_STATE_CHAIN.npz')
    parameters=js(S124/'ACTUATOR_XML_RESOLUTION.json')['all_actuators']
    qp=archive(out/'QP_PROBLEM.npz'); sol=archive(out/'QP_SOLUTION.npz')
    cand=archive(out/'OFFLINE_CANDIDATE.npz'); solver=js(out/'SOLVER.json')
    columns,ctrl,acts,da=activation_tangent(z,ini,parameters)
    c.exact('all 15 free weight indices',columns,INDICES)
    c.equal('historical full121 activation',acts,z['act'],1e-12,0)
    c.equal('reference full166 controls',ctrl,cand['reference_ctrl'],1e-12,0)
    c.equal('reference full121 activation stored',acts,cand['reference_act'],1e-12,0)
    c.equal('reference activation derivative',da,cand['activation_derivative'],1e-12,0)
    W=z['weights']; w=cand['proposed_weights']; delta=cand['delta']; x=sol['x']
    assert W.shape==w.shape==delta.shape==(66,5) and x.shape==(15,)
    c.exact('reference exact original weights',cand['reference_weights'],W)
    expect_delta=np.zeros_like(W); expect_delta[INDICES[:,0],INDICES[:,1]]=SCALE*x
    c.exact('exact registered delta',delta,expect_delta)
    c.exact('exact forward weight addition (not subtraction)',w,W+delta)
    changed=dict(z); changed['weights']=w
    _,newctrl,newacts,_=activation_tangent(changed,ini,parameters)
    features=np.concatenate((ini['features'],z['features']))
    raw=np.sum(features*W,axis=2); rawnew=np.sum(features*w,axis=2)
    for key,value in [('features',features),('reference_raw',raw),('candidate_raw',rawnew),
                      ('candidate_ctrl',newctrl),('candidate_act',newacts)]:
        c.equal('independent '+key,cand[key],value,1e-12,0)
    c.equal('initialization decoded all160',ctrl[:160],ini['decoded'],1e-12,0)
    c.equal('source applied six commands',ctrl[160:],z['applied'],1e-12,0)
    c.exact('zero initial historical ctrl',z['ctrl'][0],np.zeros(90))
    c.exact('zero pre160 activation',ini['body_act'],np.zeros(90))
    for i in range(1,121): c.exact(f'source sample{i} held ctrl',z['ctrl'][i],ctrl[160+(i-1)//20])
    Q=np.array([chain[f'sample{i:03d}_weight_J_qacc'] for i in STATES])
    L=np.array([chain[f'sample{i:03d}_weight_J_load_ratio'] for i in STATES]).reshape(3,15)
    for key,value in [('Q',Q),('L',L),('weight_indices',INDICES)]: c.exact('QP source '+key,qp[key],value)
    g=Q[:,8,:].mean(0)*SCALE
    A,lo,hi,bounds=linear_problem(Q,L,features,raw,ctrl,z['actuator_pool'],W)
    for key,value in [('objective',g),('A',A),('lower',lo),('upper',hi),('bounds',bounds)]:
        c.exact('independent native QP '+key,qp[key],value)
    rowmax=np.max(np.abs(A),axis=1); scaling=np.ones(len(A)); scaling[rowmax>0]=1/rowmax[rowmax>0]
    for key,value in [('row_scale',scaling),('scaled_A',A*scaling[:,None]),('scaled_lower',lo*scaling),('scaled_upper',hi*scaling)]:
        c.exact('registered row scaling '+key,qp[key],value)
    fu=np.flatnonzero(np.isfinite(hi)); fl=np.flatnonzero(np.isfinite(lo))
    G=np.vstack((A[fu],-A[fl],np.eye(15),-np.eye(15)))
    h=np.r_[hi[fu],-lo[fl],bounds[:,1],-bounds[:,0]]
    ds=np.r_[scaling[fu],scaling[fl],np.ones(30)]
    kinds=np.r_[np.full(len(fu),'raw_upper'),np.full(len(fl),'raw_lower'),np.full(15,'box_upper'),np.full(15,'box_lower')]
    for key,value in [('G',G),('h',h),('scaled_G',G*ds[:,None]),('scaled_h',h*ds),('dual_row_scale',ds),('dual_row_kind',kinds),('dual_row_index',np.r_[fu,fl,np.arange(15),np.arange(15)]),('H',.1*np.eye(15)),('planning_qacc67',.99),('planning_qacc48',1.98),('actual_qacc67',1.),('actual_qacc48',2.)]:
        c.exact('canonical QP '+key,qp[key],value)
    for key,value in [('objective',g),('gradient',g+.1*x),('raw_A',A),('lower',lo),('upper',hi),('bounds',bounds),('row_scale',scaling),('H',.1*np.eye(15)),('regularization',.05)]:
        c.exact('frozen solution '+key,sol[key],value)
    c.equal('predicted delta qacc',cand['predicted_delta_qacc'],np.einsum('sdj,j->sd',Q,SCALE*x),1e-12,0)
    c.equal('predicted delta load',cand['predicted_delta_load'],L@(SCALE*x),1e-12,0)
    cert=qp_certificate(.1*np.eye(15),g,A,lo,hi,x,bounds)
    # Certificate failure is reported separately; no optimizer fallback allowed.
    primal=cert['primal_violation']<=TOL
    c.equal('solver native primal residual',solver['linear_max_violation'],cert['primal_violation'],1e-12,0)
    c.equal('solver descent',solver['predicted_mean_descent'],-g@x,1e-12,0)
    gates=dict(weight_range=bool(np.all(w>=0) and np.all(w<=4)),weight_l2=bool(np.linalg.norm(delta)<=.002+TOL),
               weight_box=bool(np.max(np.abs(delta))<=SCALE+TOL),pool36=bool(np.array_equal(w[36],W[36])),
               all_other_entries=bool(np.array_equal(w[~np.isin(np.arange(66),[3,15,38])],W[~np.isin(np.arange(66),[3,15,38])])),
               zero_mask=bool(np.array_equal(w==0,W==0)),ctrl166=bool(np.max(np.abs(newctrl-ctrl))<=.005+TOL),
               clip_branches=bool(np.array_equal(branch(rawnew),branch(raw))))
    c.add('reported offline gates exact',solver['offline_gates']==gates)
    with (out/'ALL_WEIGHT_DELTAS.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==330
    for r,(p,b) in zip(rows,[(p,b) for p in range(66) for b in range(5)]):
        c.add(f'CSV exact weight {p}/{b}',int(r['pool'])==p and int(r['bin'])==b and
              float(r['original'])==W[p,b] and float(r['delta'])==delta[p,b] and float(r['candidate'])==w[p,b]
              and r['allowed']==str(p in [3,15,38]))
    prepass=bool(solver['success'] and primal and -g@x>=1e-8 and all(gates.values()))
    actual_pass=False; baseline_pass=False
    nbase=0; ncandidate=0
    if (out/'RAW_PROBES.npz').exists():
        probes=archive(out/'RAW_PROBES.npz'); baseline_gates=js(out/'BASELINE_GATES.json')
        meta=js(REFERENCE/'COM_CONTACT_METADATA.json')
        weight=float(z['body_mass'][0].sum()*np.linalg.norm(meta['gravity_world']))
        def block(i,kind):
            prefix=f'sample{i:03d}_{kind}_'
            return {k[len(prefix):]:v for k,v in probes.items() if k.startswith(prefix)}
        baseline_results=[]; candidate_results=[]; dqs=[]
        for j,i in enumerate(STATES):
            p=f'sample{i:03d}'; b=block(i,'baseline'); nbase+=bool(b)
            assert b, 'Missing baseline probe'
            bc={}
            for key in ['qacc','qfrc_constraint','qfrc_actuator','qfrc_passive','qfrc_bias','actuator_force','actuator_length','actuator_velocity','actuator_moment']:
                bc[key]=criterion(b[key],z[key][i],1e-8,1e-6)
            for key in ['efc_type','efc_id','contact_geom1','contact_geom2','contact_dim','contact_efc_address']:
                bc[key]=np.array_equal(b[key],historical[p+'_'+key])
            for key in ['contact_pos','contact_frame','efc_J_dense']:
                bc[key]=criterion(b[key],historical[p+'_'+key],1e-10,1e-10)
            bc['efc_force']=criterion(b['efc_force'],historical[p+'_efc_force'],1e-8,1e-6)
            bc['contact_force']=criterion(b['contact_force'],z['contact_force_raw6'][i,:int(b['ncon'])],1e-8,1e-6)
            bc['foot_fz']=criterion(b['foot_fz'],z['foot_force'][i,:,2],1e-8,1e-6)
            bc['load_ratio']=criterion(b['load_ratio'],z['load_ratio'][i],1e-8,1e-6)
            bc['warnings']=np.array_equal(b['warnings'],z['warnings'][i]) and not b['warnings'].any()
            bc['nefc']=int(b['nefc'])==int(historical['nefc'][i]); bc['ncon']=int(b['ncon'])==int(z['contact_ncon'][i])
            for key in ['efc_state','contact_dist','contact_friction']:
                bc['S130_'+key]=bool(np.array_equal(b[key],oldraw[p+'_baseline_'+key])) if key=='efc_state' else criterion(b[key],oldraw[p+'_baseline_'+key],1e-10,1e-10)
            bc['mass']=criterion(b['mass'],oldraw[p+'_mechanics_mass'],1e-10,1e-10)
            bc['J_transpose_force']=criterion(b['efc_J_dense'].T@b['efc_force'],b['qfrc_constraint'],1e-10,1e-10)
            gains=np.array([independent_gain(b['actuator_length'][a],b['actuator_velocity'][a],parameters[a]['resolved']) for a in range(90)])
            bias=np.zeros(90)
            for a,node in enumerate(parameters):
                prm=node['resolved']; bp=list(map(float,prm['biasprm'].split())); low,high=map(float,prm['lengthrange'].split())
                scale=(high-low)/max(1e-15,bp[1]-bp[0]); length=bp[0]+(b['actuator_length'][a]-low)/max(1e-15,scale); midpoint=(1+bp[5])*.5
                value=0. if length<=1 else (.5*((length-1)/max(1e-15,midpoint-1))**2 if length<=midpoint else .5+(length-midpoint)/max(1e-15,midpoint-1))
                bias[a]=-bp[2]*bp[7]*value
            bc['analytic_force']=criterion(gains*b['act']+bias,b['actuator_force'],1e-10,1e-10)
            baseline_results.append(bool(all(bc.values())))
            saved=baseline_gates[j]
            c.add(p+' baseline gate semantics',saved['sample']==i and saved['passed']==baseline_results[-1] and set(saved['checks'])==set(bc))
            for key,value in bc.items(): c.add(p+' baseline reported '+key,bool(value)==saved['checks'][key]['passed'])
            cn=block(i,'candidate'); ncandidate+=bool(cn)
            for label,probe in [('baseline',b)]+([('candidate',cn)] if cn else []):
                for field in ['qpos','qvel','ctrl','qfrc_applied','xfrc_applied','time']:
                    c.exact(p+label+' frozen '+field,probe[field],z[field][i])
                c.equal(p+label+' actual activation',probe['act'],newacts[i] if label=='candidate' else acts[i],1e-12,0)
                c.exact(p+label+' cold warmstart',probe['warmstart_before'],np.zeros(48))
                c.add(p+label+' finite full qacc',probe['qacc'].shape==(48,) and np.isfinite(probe['qacc']).all())
                c.add(p+label+' mass SPD',probe['mass'].shape==(48,48) and np.linalg.eigvalsh(probe['mass']).min()>0)
                c.equal(p+label+' mass symmetric',probe['mass'],probe['mass'].T,1e-12,0)
                c.equal(p+label+' J.T efc',probe['efc_J_dense'].T@probe['efc_force'],probe['qfrc_constraint'],1e-10,1e-10)
                foot,_,_,_=contact_load(probe['contact_force'],probe['contact_frame'],np.stack((probe['contact_geom1'],probe['contact_geom2']),axis=1),meta)
                c.equal(p+label+' oriented foot load',probe['foot_fz'],foot[:,2],1e-10,1e-10)
                c.equal(p+label+' native weight ratio',probe['load_ratio'],foot[:,2].sum()/weight,1e-12,1e-12)
                c.equal(p+label+' analytic force',probe['actuator_force'],gains*probe['act']+bias,1e-10,1e-10)
                c.equal(p+label+' moment force',probe['qfrc_actuator'],probe['actuator_moment'].T@probe['actuator_force'],1e-10,1e-10)
            if not cn: continue
            for field in ['mass','actuator_moment','qfrc_passive','qfrc_bias','actuator_length','actuator_velocity','efc_J_dense','contact_pos','contact_frame','contact_dist','contact_friction']:
                c.equal(p+' candidate frozen mechanics '+field,cn[field],b[field],1e-10,1e-10)
            c.equal(p+' differential dynamics',b['mass']@(cn['qacc']-b['qacc']),cn['qfrc_actuator']-b['qfrc_actuator']+cn['qfrc_constraint']-b['qfrc_constraint'],1e-9,1e-9)
            q=cn['qacc']-z['qacc'][i]; l=float(cn['load_ratio']-z['load_ratio'][i]); dqs.append(q)
            identity=all(np.array_equal(cn[k],b[k]) for k in ['nefc','ncon','efc_type','efc_id','efc_state','contact_geom1','contact_geom2','contact_dim','contact_efc_address'])
            candidate_results.append(dict(sample=i,load_delta=l,max_abs_delta_qacc67=float(np.max(np.abs(q[6:8]))),max_abs_delta_qacc48=float(np.max(np.abs(q))),delta_qacc8=float(q[8]),load_pass=l>=-TOL,qacc67_pass=bool(np.max(np.abs(q[6:8]))<=1+TOL),qacc48_pass=bool(np.max(np.abs(q))<=2+TOL),warnings_zero=not cn['warnings'].any(),contact_identity_unchanged=identity,J_transpose_force=criterion(cn['efc_J_dense'].T@cn['efc_force'],cn['qfrc_constraint'],1e-10,1e-10)))
        baseline_pass=all(baseline_results)
        c.add('no candidate forward after baseline rejection',bool(ncandidate)==baseline_pass)
        if ncandidate:
            assert ncandidate==3
            actual=js(out/'ACTUAL_GATES.json'); mean8=float(np.array(dqs)[:,8].mean())
            assert len(actual['points'])==3
            for r,saved in zip(candidate_results,actual['points']):
                for key,value in r.items():
                    reported=saved[key]['passed'] if isinstance(saved[key],dict) else saved[key]
                    c.add(f'actual sample{r["sample"]} gate {key}',bool(value)==reported if isinstance(value,(bool,np.bool_)) else value==reported)
            c.add('actual mean descent exact',actual['mean_delta_qacc8']==mean8 and actual['mean_descent_pass']==(mean8 < -1e-8))
            actual_pass=bool(all(all(r[k] for k in ['load_pass','qacc67_pass','qacc48_pass','warnings_zero','contact_identity_unchanged','J_transpose_force']) for r in candidate_results) and mean8 < -1e-8)
    # Rejection is evidence, not verifier failure. Promotion additionally requires KKT.
    admission=bool(prepass and baseline_pass and actual_pass and cert['certificate_passed'])
    exists=(out/'CANDIDATE_WEIGHTS.npz').exists()
    c.add('candidate exists iff every registered gate and certificate',exists==admission)
    c.add('summary admission',summary['offline_candidate_saved']==admission)
    if (out/'ACTUAL_GATES.json').exists(): c.add('actual admission report',js(out/'ACTUAL_GATES.json')['offline_candidate_saved']==admission)
    if exists:
        promoted=archive(out/'CANDIDATE_WEIGHTS.npz')
        for key,value in [('weights',w),('reference_weights',W),('delta',delta)]: c.exact('promoted exact '+key,promoted[key],value)
    ledger=js(out/'CALL_LEDGER.json')
    c.add('literal call ledger',ledger==dict(model_load=int(nbase>0),forward=nbase+ncandidate,baseline_forward=nbase,candidate_forward=ncandidate,step=0,neural_ticks=0,solver_invocations=1))
    c.add('summary ledger',summary['ledger']==ledger)
    c.add('no admission claims',summary['physical_admission'] is False and summary['standing_or_walking_claim'] is False and summary['units']=='NOT_ESTABLISHED')
    frozen=js(out/'FROZEN_BEFORE_AFTER.json')
    c.add('literal input count (not canonical deduplicated)',frozen['passed'] is True and frozen['inputs']==len(ilit) and frozen['prereg_sha256']==sha(out/'PREREG.json'))
    c.add('input rehash',checked_manifest(out/'INPUT_HASHES.json')==(ilit,inputs))
    c.add('output rehash',checked_manifest(out/'OUTPUT_HASHES.json')==(olit,outputs))
    c.add('no simulator imported','mujoco' not in sys.modules)
    result=dict(passed=all(r['passed'] for r in c.rows),evidence_passed=all(r['passed'] for r in c.rows),
                proposal_accepted=admission,certificate_passed=cert['certificate_passed'],unique_qp=cert,
                checks=c.rows,model_load=0,forward=0,step=0,neural_ticks=0,physical_admission=False,
                units='NOT_ESTABLISHED',verifier_sha256=sha(__file__),input_files=len(inputs),input_literal_paths=len(ilit),
                output_files=len(outputs),output_literal_paths=len(olit),input_output_hashes_rechecked=True,
                scope='S132 independent saved-byte arithmetic and normalized nonnegative KKT; no independent contact solver or trajectory acceptance')
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
    if not result['passed']: raise SystemExit(1)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output)
