"""S131 independent OFFLINE candidate verifier: saved bytes only, no simulator.

The frozen mechanical-state checks are not a trajectory/closed-loop gradient or
behavioral acceptance. Import only inert numpy S130 verifier helpers.
"""
from pathlib import Path
import argparse, csv, json, sys, numpy as np
from verify_contact_aware_activation_response import (
    BASE, REFERENCE, S124, STATES, Checks, sha, js, archive, manifest,
    activation_tangent, criterion,
)
SCALE=.002/np.sqrt(15); COMPONENT=SCALE

def qp_certificate(H,g,A,lo,hi,x):
    H=(np.asarray(H)+np.asarray(H).T)/2; g=np.asarray(g); A=np.asarray(A); x=np.asarray(x)
    slack_lo=A@x-lo; slack_hi=hi-A@x; active=np.flatnonzero(np.minimum(slack_lo,slack_hi)<=1e-8)
    # Convert upper/lower active rows into Adual lambda convention.
    rows=[]
    for i in active:
        if slack_hi[i] <= slack_lo[i]: rows.append(A[i])
        else: rows.append(-A[i])
    R=np.array(rows) if rows else np.empty((0,len(x))); grad=H@x+g
    lam=np.linalg.lstsq(R.T,-grad,rcond=None)[0] if len(R) else np.empty(0)
    eig=float(np.linalg.eigvalsh(H).min()); station=float(np.max(np.abs(grad+R.T@lam),initial=0))
    primal=float(max(np.max(-slack_lo,initial=0),np.max(-slack_hi,initial=0)))
    dual=float(np.max(-lam,initial=0)); comp=float(np.max(np.abs(lam*np.minimum(slack_lo,slack_hi)[active]),initial=0)) if len(active) else 0.
    # KKT is diagnostic; primal feasibility + strict convexity certify unique minimizer
    return dict(strict_convex=eig>0,min_eigenvalue=eig,primal_violation=primal,
                stationarity_max_abs=station,dual_violation=dual,complementarity_max_abs=comp,
                kkt_verified=bool(station<=1e-7 and dual<=1e-7 and comp<=1e-7),
                unique_qp=bool(eig>0 and primal<=1e-8))

def main(out):
    out=Path(out).resolve(); target=out/'PARENT_INDEPENDENT_CHECK_FINAL.json'; assert not target.exists()
    assert 'mujoco' not in sys.modules
    c=Checks(); prereg=js(out/'PREREG.json'); assert prereg['samples']==[77,100,105]
    assert prereg['allowed_pools']==[3,15,38] and prereg['entries']==15
    assert prereg['scale']==SCALE and prereg['regularization']==.05
    qp=archive(out/'QP_PROBLEM.npz'); cand=archive(out/'OFFLINE_CANDIDATE.npz')
    x=(cand['delta'][np.array([3,3,3,3,3,15,15,15,15,15,38,38,38,38,38]),np.tile(np.arange(5),3)]/SCALE)
    # Above indexing is deliberate exact 15-column ordering.
    g=qp['objective']; A=qp['A']; lo=qp['lower']; hi=qp['upper']
    c.add('QP dimensions',A.shape[1]==15 and g.shape==(15,) and x.shape==(15,))
    c.add('objective exact',np.allclose(g, np.asarray(qp['Q'])[:,8,:].mean(0)*SCALE,atol=1e-14,rtol=0))
    c.add('QP primal residual',max(float(np.max(lo-A@x)),float(np.max(A@x-hi)))<=1e-8)
    cert=qp_certificate(.1*np.eye(15),g,A,lo,hi,x); c.add('strict convex QP unique',cert['unique_qp'])
    c.add('KKT diagnostic',cert['kkt_verified'])
    w=cand['proposed_weights']; ref=cand['reference_weights']; delta=cand['delta']
    c.add('exact 330 weight delta',delta.shape==(66,5) and np.array_equal(w-ref,delta))
    c.add('pool36 unchanged',np.array_equal(w[36],ref[36]))
    c.add('mask unchanged',np.array_equal(w==0,ref==0))
    c.add('trust L2',np.linalg.norm(delta)<=.002+1e-9)
    c.add('component cap',np.max(np.abs(delta))<=SCALE+1e-9)
    c.add('candidate file iff all gates',('CANDIDATE_WEIGHTS.npz' in {p.name for p in out.iterdir()}) == bool(js(out/'ACTUAL_GATES.json')['offline_candidate_saved']))
    actual=js(out/'ACTUAL_GATES.json'); points=actual['points']; c.add('three actual points',len(points)==3 and [p['sample'] for p in points]==[77,100,105])
    for p in points:
        c.add(f"sample{p['sample']} nonlinear gates",all(p[k] for k in ('load_pass','qacc67_pass','qacc48_pass','warnings_zero','contact_identity_unchanged','J_transpose_force')))
    c.add('mean qacc8 descent',actual['mean_delta_qacc8'] < -1e-8)
    c.add('no simulator imported','mujoco' not in sys.modules)
    result=dict(passed=all(r['passed'] for r in c.rows),scope='OFFLINE S131 saved candidate/QP/contact-gate integrity; no simulator execution',checks=c.rows,unique_qp=cert,model_load=0,forward=0,step=0,neural_ticks=0,physical_admission=False,units='NOT_ESTABLISHED',verifier_sha256=sha(__file__))
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    if not result['passed']: raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True,type=Path); main(p.parse_args().output)
