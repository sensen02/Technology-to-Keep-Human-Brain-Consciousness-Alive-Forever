"""Exact-rational structure of the A-gate static relaxation as a function of the force scale s.

Offline only: reads the archived matrices and the independent review receipt; no simulation,
no mj_step, no re-run of any body/neural experiment. Everything reported is an exact fraction
pair where possible.

Certificate convention (Farkas alternative for the declared system):
  primal feasible  <=>  no y with C^T y <= 0, |y| <= 1, b^T y > sum_i max(lo_i t_i, hi_i t_i),
  where t = s*(R^T y) because only the muscle columns are scaled by s.
For a fixed y: gap_s(y) = P(y) - s*U(y) with P = b^T y and U = sum_i max(lo_i t_i, hi_i t_i).
  U > 0  -> certificate holds for every s < P/U
  U < 0  -> certificate holds for every s > P/U   (gap grows with s)
"""
from pathlib import Path
from fractions import Fraction as F
import json,time,hashlib,zipfile
import numpy as np
from scipy.optimize import linprog
GATE=Path('/home/sensen/Desktop/engine修复验收/S2-support-gate-20261014T172624/native_diamond.npz')
CERT=Path('/home/sensen/Desktop/engine修复验收/S2-support-certificate-20261014T175213/certificate.json')
REVIEW=Path('/home/sensen/Desktop/engine修复验收/S2-force-scale-sensitivity-20261015T101510/independent-check.json')
SCAN_LOW=[1.0,1.04,1.045,1.04502,1.045025,1.0450257,1.0450258,1.045026,1.04503,1.0451]
SCAN_HIGH=[300.0,500.0,1000.0,2000.0,4000.0,8000.0,10000.0,1e5]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def enc(x):return [str(x.numerator),str(x.denominator)]
def main():
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-force-scale-exact-partition-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='EXACT_RATIONAL_PARTITION_OF_THE_DECLARED_STATIC_RELAXATION_IN_S',offline_only=True,no_simulation=True)
    hashes={str(p):sha(p) for p in (GATE,CERT,REVIEW,Path(__file__))}
    try:
        with np.load(GATE) as z:A,b,lo,hi=z['A'].copy(),z['target'].copy(),z['lo'].copy(),z['hi'].copy()
        m=len(lo);n=A.shape[0]
        ex=lambda x:F(float(x))
        arch=[x for x in json.loads(CERT.read_text())['results'] if x['floor']=='native'][0]
        # ---- (0) s=1 conclusion reconfirmed exactly, from the archived certificate's own vector
        y0=[F(int(a),int(d)) for a,d in arch['y_rational']]
        P0=sum(ex(b[i])*y0[i] for i in range(n));t0=[sum(ex(A[i,j])*y0[i] for i in range(n)) for j in range(m)]
        U0=sum(max(ex(lo[j])*t0[j],ex(hi[j])*t0[j]) for j in range(m))
        c0=[sum(ex(A[i,j])*y0[i] for i in range(n)) for j in range(m,A.shape[1])]
        gap0=P0-U0
        r['s1_conclusion_reconfirmed']=dict(P=enc(P0),U=enc(U0),gap=enc(gap0),gap_float=float(gap0),gap_strictly_positive=gap0>0,
            contact_dots_all_nonpositive=all(c<=0 for c in c0),contact_max=enc(max(c0)),
            statement='the s=1 infeasibility certificate of the declared static relaxation still holds exactly (P-U>0 with exact rationals); it is NOT withdrawn and is unaffected by the existence of a feasible window elsewhere')
        # ---- (1) verify the independent primal witness at s=21/20 (and its dyadic variant) exactly
        rec=json.loads(REVIEW.read_text())['exact_primal_witness_at_s_1_05']
        witnesses={}
        for label,key in (('s_21_20','exact_rational_witness'),('dyadic_near_1_05','exact_rational_witness_same_vector_against_float64_product_matrix')):
            w=rec[key];scale=F(w['scale']);den=F(w['denominator']);xs=[F(int(v),int(den)) for v in w['x_numerators']]
            assert len(xs)==A.shape[1]
            resid=[sum(ex(A[i,j])*(scale*xs[j] if j<m else xs[j]) for j in range(A.shape[1]))-ex(b[i]) for i in range(n)]
            bounds_ok=all(ex(lo[j])<=xs[j]<=ex(hi[j]) for j in range(m)) and all(x>=0 for x in xs[m:])
            witnesses[label]=dict(scale=enc(scale),denominator=enc(den),equality_residual_exactly_zero=all(v==0 for v in resid),
                bounds_hold_exactly=bounds_ok,feasible_exactly=bool(all(v==0 for v in resid) and bounds_ok),
                min_muscle_slack_float=float(min(min(xs[j]-ex(lo[j]),ex(hi[j])-xs[j]) for j in range(m))),
                min_contact_float=float(min(xs[m:])),x_numerators_count=len(xs))
        r['primal_witness_registered']=dict(source='independent review receipt (independent-check.json)',witnesses=witnesses,
            meaning='this is a DECLARED-MODEL STATIC LP RELAXATION feasible point: A_s x = b exactly, muscle bounds exactly, contacts >= 0 exactly. It is NOT a pose, NOT a supportable configuration, NOT a compliant MuJoCo equilibrium and NOT a biological result; it must never be written as "it can stand" or "increase the force 4.5 percent and it stands".')
        # ---- (2) exact boundaries via LP + exact repair
        def lp(s,scale_muscle=True):
            nv=A.shape[0];c=np.concatenate([-b,np.ones(m)]);rows=[];rhs=[]
            for j in range(m):
                for bd in (lo[j],hi[j]):
                    row=np.zeros(nv+m);row[:nv]=bd*s*A[:,j];row[nv+j]=-1.0;rows.append(row);rhs.append(0.0)
            for j in range(m,A.shape[1]):
                row=np.zeros(nv+m);row[:nv]=A[:,j];rows.append(row);rhs.append(0.0)
            sol=linprog(c,A_ub=np.array(rows),b_ub=np.array(rhs),bounds=[(-1,1)]*nv+[(None,None)]*m,method='highs')
            return (None,None) if not sol.success else (-float(sol.fun),np.array(sol.x[:nv]))
        def repair_and_verify(yv,s):
            y=[ex(v) for v in yv]
            def verify(y):
                P=sum(ex(b[i])*y[i] for i in range(n));t=[sum(ex(A[i,j])*y[i] for i in range(n)) for j in range(m)]
                U=sum(max(ex(lo[j])*t[j],ex(hi[j])*t[j]) for j in range(m))
                cv=[sum(ex(A[i,j])*y[i] for i in range(n)) for j in range(m,A.shape[1])]
                return P,U,cv
            P,U,cv=verify(y)
            if any(v>0 for v in cv) and all(ex(A[2,j])>0 for j in range(m,A.shape[1])):
                corr=max(v/ex(A[2,m+j]) for j,v in enumerate(cv) if v>0)
                y[2]-=corr;P,U,cv=verify(y)
            gap=P-F(s)*U   # exact: s as an exact binary rational, U(y) unscaled
            return dict(P=enc(P),U=enc(U),P_float=float(P),U_float=float(U),contact_max=enc(max(cv)),contact_ok=all(v<=0 for v in cv),
                abs_y_max_exact=enc(max(abs(v) for v in y)),abs_y_max_float=max(abs(float(v)) for v in y),gap_at_s_float=float(gap),gap_at_s_exact=enc(gap),
                valid_at_s=bool(all(v<=0 for v in cv) and max(abs(v) for v in y)<=1 and gap>0),
                threshold=enc(P/U) if U!=0 else None,threshold_float=(float(P/U) if U!=0 else None),U_sign=('positive' if U>0 else 'negative' if U<0 else 'zero'))
        low=[];high=[]
        for s in SCAN_LOW:
            obj,yv=lp(s)
            if obj is not None and obj>0:
                e=repair_and_verify(yv,s);e['s']=s;e['lp_objective']=obj;low.append(e)
        for s in SCAN_HIGH:
            obj,yv=lp(s)
            if obj is not None and obj>0:
                e=repair_and_verify(yv,s);e['s']=s;e['lp_objective']=obj;high.append(e)
        valid_low=[e for e in low if e['valid_at_s'] and e['U_sign']=='positive']
        valid_high=[e for e in high if e['valid_at_s'] and e['U_sign']=='negative' and e['threshold_float']>0]
        r['low_side']=dict(scans=low,best=max(valid_low,key=lambda e:e['threshold_float']) if valid_low else None)
        r['high_side']=dict(scans=high,best=min(valid_high,key=lambda e:e['threshold_float']) if valid_high else None)
        r['partition']=dict(
            infeasible_below=(r['low_side']['best']['threshold'] if r['low_side']['best'] else None),
            feasible_at=witnesses['s_21_20']['scale'],
            infeasible_above=(r['high_side']['best']['threshold'] if r['high_side']['best'] else None),
            status_between_low_and_105='no separating vector found by the float LP on the sampled grid (scans 1.0450258..1.0451); the exact status is open there because no exact primal witness and no exact certificate was constructed at those s',
            status_between_105_and_high='no separating vector found by the float LP on the sampled grid (300..2000); exact status open, and a feasible point exists at 21/20, so the feasible window is nonempty',
            note='boundaries are exact rational thresholds attached to exact certificates; each certificate is valid for EVERY s on its side of the threshold')
        r['status']='PASS_EXACT_PARTITION_REPORTED'
    except Exception as e:
        import traceback;r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc()[-1200:])
    finally:
        r['hashes_before']=hashes;r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['hashes_before']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status'])
    if r['status'].startswith('PASS'):
        print('s=1 gap exact >0:',r['s1_conclusion_reconfirmed']['gap_strictly_positive'])
        print('witness feasible exactly:',{k:v['feasible_exactly'] for k,v in r['primal_witness_registered']['witnesses'].items()})
        print('low best threshold:',r['low_side']['best']['threshold_float'] if r['low_side']['best'] else None)
        print('high best threshold:',r['high_side']['best']['threshold_float'] if r['high_side']['best'] else None)
    else:print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
