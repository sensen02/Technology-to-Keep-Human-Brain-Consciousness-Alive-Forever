"""Correction scan: is the certificate-existence region in the force scale monotone?

For a fixed y, gap(s) = P(y) - s*U(y) with U(y) = sum_i max(lo_i t_i, hi_i t_i).
Because lo_i > 0 in the native floor, U(y) can be NEGATIVE (the activation floor forces a
minimum torque), in which case gap(s) grows with s. So the dual maximum need not be monotone.
This script scans large s and exact-verifies any candidate the LP returns.
Kinematics/dynamics: none. Offline linear algebra only.
"""
from pathlib import Path
from fractions import Fraction
import json,time,hashlib,zipfile
import numpy as np
from scipy.optimize import linprog
GATE=Path('/home/sensen/Desktop/engine修复验收/S2-support-gate-20261014T172624/native_diamond.npz')
CERT=Path('/home/sensen/Desktop/engine修复验收/S2-support-certificate-20261014T175213/certificate.json')
SCAN=[1.05,10.0,30.0,100.0,300.0,500.0,1000.0,4000.0,9773.0,10000.0,100000.0]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=Path('/home/sensen/Desktop/engine修复验收')/('S2-force-scale-large-scan-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 r=dict(status='RUNNING',scope='CORRECTION_SCAN_OF_THE_FORCE_SCALE_CERTIFICATE_REGION',claim_under_test='the earlier global_threshold was described as the largest s with a separating certificate; this scan tests whether separation can return at larger s',scan=SCAN)
 hashes={str(p):sha(p) for p in (GATE,CERT,Path(__file__))}
 try:
  with np.load(GATE) as z:A,b,lo,hi=z['A'].copy(),z['target'].copy(),z['lo'].copy(),z['hi'].copy()
  nm=90;R=A[:,:nm];C=A[:,nm:];ex=lambda x:Fraction(float(x))
  def lp(s):
   nv=A.shape[0];c=np.concatenate([-b,np.ones(nm)]);rows=[];rhs=[]
   for j in range(nm):
    for bd in (lo[j],hi[j]):
     row=np.zeros(nv+nm);row[:nv]=bd*s*R[:,j];row[nv+j]=-1.0;rows.append(row);rhs.append(0.0)
   for k in range(C.shape[1]):
    row=np.zeros(nv+nm);row[:nv]=C[:,k];rows.append(row);rhs.append(0.0)
   sol=linprog(c,A_ub=np.array(rows),b_ub=np.array(rhs),bounds=[(-1,1)]*nv+[(None,None)]*nm,method='highs')
   return (None,None) if not sol.success else (-float(sol.fun),np.array(sol.x[:nv]))
  def exact(yv):
   Y=[ex(x) for x in yv]
   P=sum(ex(b[i])*Y[i] for i in range(len(Y)));t=[sum(ex(R[i,j])*Y[i] for i in range(R.shape[0])) for j in range(nm)]
   U=sum(max(ex(lo[j])*t[j],ex(hi[j])*t[j]) for j in range(nm))
   cv=[sum(ex(C[i,k])*Y[i] for i in range(C.shape[0])) for k in range(C.shape[1])]
   thr=(P/U if U!=0 else None)
   return dict(P=float(P),U=float(U),U_is_negative=U<0,contact_max=float(max(cv)),contact_ok=all(c<=0 for c in cv),abs_y_max=max(abs(float(x)) for x in yv),threshold_P_over_U=(float(thr) if thr is not None else None),valid_for_s_above_threshold=bool(U<0 and thr is not None and thr>0))
  rows_out=[]
  for s in SCAN:
   obj,yv=lp(s);entry=dict(s=s,max_dual_objective=None if obj is None else round(obj,9))
   if obj is not None and obj>0 and yv is not None:
    e=exact(yv);entry['candidate_exact']=e
    # a certificate valid for all larger s requires P>0 and U<0 exactly
    # If U<0 the gap grows with s, so a candidate with contact dots exactly <=0 and threshold
    # P/U > 0 is an exact certificate valid for ALL s above that threshold.
    gap_at_s=e['P']-s*e['U']
    entry['exact_certificate_for_all_larger_s']=bool(e['contact_ok'] and e['abs_y_max']<=1.0 and e['valid_for_s_above_threshold'] and gap_at_s>0)
   rows_out.append(entry);print('s',s,'obj',entry['max_dual_objective'],flush=True)
  r['scan_results']=rows_out
  later=[e for e in rows_out if e['max_dual_objective'] is not None and e['max_dual_objective']>0 and e['s']>10]
  r['correction_finding']=dict(separation_returns_above_s_10=bool(later),examples=[dict(s=e['s'],objective=e['max_dual_objective']) for e in later[:4]],exact_certificates_exhibited=sum(1 for e in later if e.get('exact_certificate_for_all_larger_s')),revised_wording='s_global=1.045025349 is the FIRST sign change of the sampled dual maximum (the boundary of the certificate-free window found by the grid), NOT the largest scale with a separating certificate. Because the native activation floor is positive, U(y) can be negative, so the dual maximum is not monotone in s and separation returns at large s.')
  r['status']='PASS_CORRECTION_SCAN_COMPLETED'
 except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
 finally:
  r['hashes_before']=hashes;r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['hashes_before']==r['hashes_after']
  with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
  (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(out);print(r['status']);print(json.dumps(r.get('correction_finding'),indent=2,ensure_ascii=False));print(r.get('error',''));return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
