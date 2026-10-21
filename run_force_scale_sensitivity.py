"""Declared force-scale sensitivity of the frozen A-gate infeasibility certificate."""
from pathlib import Path
from fractions import Fraction
import json,time,hashlib,zipfile
import numpy as np
from scipy.optimize import linprog
GATE=Path('/home/sensen/Desktop/engine修复验收/S2-support-gate-20261014T172624/native_diamond.npz')
CERT=Path('/home/sensen/Desktop/engine修复验收/S2-support-certificate-20261014T175213/certificate.json')
TOL=1e-6
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=Path('/home/sensen/Desktop/engine修复验收')/('S2-force-scale-sensitivity-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 r=dict(status='RUNNING',scope='DECLARED_FORCE_SCALE_SENSITIVITY_OFFLINE_ONLY',b_scaled=False,
        scale_semantics='s multiplies the 90 muscle torque columns only; the body-weight target b is held fixed, so s is muscle force per unit body weight. Scaling all forces together would be scale invariant and therefore uninformative.',note='no biological calibration source used; no standing claim')
 hashes={str(p):sha(p) for p in (GATE,CERT,Path(__file__))}
 try:
  with np.load(GATE) as z:A,b,lo,hi=z['A'].copy(),z['target'].copy(),z['lo'].copy(),z['hi'].copy()
  nm=90;R=A[:,:nm];C=A[:,nm:]
  res=[x for x in json.loads(CERT.read_text())['results'] if x['floor']=='native'][0]
  y=[Fraction(int(n),int(d)) for n,d in res['y_rational']];assert len(y)==A.shape[0]
  stored=Fraction(int(res['gap'][0]),int(res['gap'][1]));ex=lambda x:Fraction(float(x))
  P=sum(ex(b[i])*y[i] for i in range(len(y)));t=[sum(ex(R[i,j])*y[i] for i in range(R.shape[0])) for j in range(nm)]
  U=sum(max(ex(lo[j])*t[j],ex(hi[j])*t[j]) for j in range(nm))
  contact=[sum(ex(C[i,k])*y[i] for i in range(C.shape[0])) for k in range(C.shape[1])]
  r['certificate_reproduction']=dict(P_float=float(P),U_float=float(U),gap_exact_matches_stored=(P-U==stored),gap_float=float(P-U),stored_gap_float=res['gap_float'],contact_max_float=float(max(contact)),all_contact_nonpositive=all(c<=0 for c in contact))
  assert P-U==stored and all(c<=0 for c in contact)
  sstar=P/U;r['saved_vector_threshold']=dict(s_star_exact_float=float(sstar),interpretation='for every s<s_star the SAVED rational separating vector still witnesses infeasibility exactly; at s>=s_star this vector no longer separates')
  def lp(s):
   # max_y y^T b - s*sum z   s.t. z_i >= lo_i*s*(R^T y)_i, z_i >= hi_i*s*(R^T y)_i, C^T y <= 0, |y|<=1
   nv=A.shape[0];c=np.concatenate([-b,np.ones(nm)])            # minimise -objective
   rows=[];rhs=[]
   for j in range(nm):
    for bd in (lo[j],hi[j]):
     # z_j >= bd*s*(R^T y)_j  as a <= constraint: bd*s*(R^T y)_j - z_j <= 0
     row=np.zeros(nv+nm);row[:nv]=bd*s*R[:,j];row[nv+j]=-1.0;rows.append(row);rhs.append(0.0)
   for k in range(C.shape[1]):
    row=np.zeros(nv+nm);row[:nv]=C[:,k];rows.append(row);rhs.append(0.0)
   # z is free: max(lo_i t_i, hi_i t_i) can be negative when t_i < 0
   bounds=[(-1,1)]*nv+[(None,None)]*nm
   sol=linprog(c,A_ub=np.array(rows),b_ub=np.array(rhs),bounds=bounds,method='highs')
   return (None,None) if not sol.success else (-float(sol.fun),np.array(sol.x[:nv]))
  grid=[.5,.8,1.0,float(sstar),1.05,1.1,1.2,1.5,2.0,3.0,5.0,10.0]
  curve=[]
  for s in grid:
   obj,yv=lp(s);curve.append(dict(s=s,max_dual_objective=None if obj is None else round(obj,9)))
  r['grid']=curve
  hi_s=1.0
  while True:
   obj,_=lp(hi_s*1.5)
   if obj is None or obj<=0:break
   hi_s*=1.5
   if hi_s>1e6:raise RuntimeError('no upper scale found')
  lo_s,high=1.0,hi_s*1.5
  while abs(high-lo_s)>TOL*max(1.0,lo_s):
   mid=.5*(lo_s+high);obj,_=lp(mid)
   if obj is None:raise RuntimeError('LP failure inside bisection')
   lo_s,high=(mid,high) if obj>0 else (lo_s,mid)
  r['global_threshold']=dict(s_global_float=round(.5*(lo_s+high),9),bracket=[round(lo_s,9),round(high,9)],tolerance=TOL,meaning='largest s (within |y|<=1) for which a separating certificate can still be found by this dual; above it no certificate is found, which is NOT a proof of feasibility')
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(8,4));xs=[c['s'] for c in curve if c['max_dual_objective'] is not None];ys=[c['max_dual_objective'] for c in curve if c['max_dual_objective'] is not None]
  ax.semilogx(xs,ys,'o-',label='max dual objective (certificate margin)');ax.axhline(0,color='k',lw=.8);ax.axvline(float(sstar),color='r',linestyle='--',label=f"saved-vector threshold {float(sstar):.4f}");ax.axvline(r['global_threshold']['s_global_float'],color='g',linestyle=':',label=f"global threshold {r['global_threshold']['s_global_float']:.4f}")
  ax.set(xlabel='muscle force scale s (relative to body weight)',ylabel='max dual objective');ax.legend();ax.set_title('Declared force-scale sensitivity of the A-gate infeasibility certificate');fig.tight_layout();fig.savefig(out/'force-scale.png');plt.close(fig)
  r['status']='PASS_DECLARED_SCALE_SENSITIVITY_NO_CALIBRATION_SOURCE'
  def exact_verify(yv):
   """Treat the float64 vector as exact rationals, like the archived certificate does."""
   Y=[ex(x) for x in yv]
   Pv=sum(ex(b[i])*Y[i] for i in range(len(Y)));tv=[sum(ex(R[i,j])*Y[i] for i in range(R.shape[0])) for j in range(nm)]
   Uv=sum(max(ex(lo[j])*tv[j],ex(hi[j])*tv[j]) for j in range(nm))
   cv=[sum(ex(C[i,k])*Y[i] for i in range(C.shape[0])) for k in range(C.shape[1])]
   return dict(P=float(Pv),U=float(Uv),gap_float=float(Pv-Uv),contact_max=float(max(cv)),contact_ok=all(c<=0 for c in cv),abs_y_max=max(abs(float(x)) for x in yv),ratio=float(Pv/Uv) if Uv>0 else None,in_bounds=max(abs(float(x)) for x in yv)<=1.0)
  candidates={}
  for s in (1.0,float(sstar),r['global_threshold']['s_global_float']):
   obj,yv=lp(s)
   if obj is not None and obj>0:candidates[f'lp_at_s_{round(s,9)}']=exact_verify(yv)
  r['exact_verified_lp_candidates']=candidates
  good={k:v for k,v in candidates.items() if v['contact_ok'] and v['in_bounds'] and v['gap_float']>0}
  r['exact_repair_blocked']=dict(explanation='LP-derived vectors separate strongly in float, but their exact rational contact dots are tiny POSITIVE values, so they are not exact certificates. The archived certificate became exact through an explicit repair step (exact_vertical_repair) that this round did not re-apply.',max_positive_contact_exact=(max(v['contact_max'] for v in candidates.values()) if candidates else None),exactly_verified_candidates=len(good))
  r['threshold_bracket']=dict(exactly_proven_lower=float(sstar),float_lp_upper=r['global_threshold']['s_global_float'],statement='infeasibility is EXACTLY proven for every s below exactly_proven_lower using the archived rational vector; above float_lp_upper the float dual found no separating vector, which is NOT a proof of feasibility; between the bounds the exact status is open because the stronger vectors need an exact repair.')
  r['status']='PASS_DECLARED_SCALE_SENSITIVITY_BRACKET_NO_CALIBRATION_SOURCE'
 except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
 finally:
  r['hashes_before']=hashes;r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['hashes_before']==r['hashes_after']
  with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
  (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(out);print(r['status']);print(json.dumps({k:r.get(k) for k in ('certificate_reproduction','saved_vector_threshold','global_threshold')},indent=2));print(r.get('error',''));return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
