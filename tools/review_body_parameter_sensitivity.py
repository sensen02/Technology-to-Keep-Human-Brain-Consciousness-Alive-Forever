"""Parent independent check of the body-parameter sweep: rebuild the constraint systems
from sealed arrays and re-solve a bounded subset at the scanned extremes.
No MuJoCo, no new capture, no model edits.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.optimize import linprog
ROOT=Path('/home/sensen/Desktop/engine修复验收')
P=ROOT/'s2-mapped-support-capacity/20261102T170741Z'
N=ROOT/'s2-native-support-constraints/20261102T172633Z'
S=ROOT/'s2-body-parameter-sensitivity/20261102T190000Z'
O=ROOT/'s2-body-parameter-parent-review/20261102T190000Z'
FLOOR=1e-4
def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def system(m,s,u,k,contacts,internal,use_friction):
 A=m['moment'].T*(m['gain']*s)[None,:];base=(m['qfrc_passive']-m['qfrc_bias'])+m['moment'].T@(m['bias']*s)
 b=-base-A@np.full(90,FLOOR);E=np.column_stack([A,contacts,internal]) if internal.size else np.column_stack([A,contacts])
 nc=contacts.shape[1]//3 if use_friction else contacts.shape[1];nint=int(internal.shape[1]);n=E.shape[1]
 cb=([(-np.inf,np.inf),(-np.inf,np.inf),(0.,np.inf)]*nc) if use_friction else ([(0.,np.inf)]*nc)
 bounds=[(0.,max(u-FLOOR,0.))]*90+cb+[(0.,np.inf)]*nint
 rows=[]
 for c in range(nc if use_friction else 0):
  z=90+3*c;mu=float(m['mu'][c])*k
  for sx,sy in [(1.,0.),(-1.,0.),(0.,1.),(0.,-1.)]:
   r=np.zeros(n);r[z]=sx;r[z+1]=sy;r[z+2]=-mu;rows.append(r)
 U=np.array(rows).reshape(-1,n) if rows else np.zeros((0,n));v=np.zeros(len(U));rs=np.maximum(1,np.maximum(abs(E).max(1),abs(b)))
 return E,b,U,v,bounds,rs
def solve(m,s,u,k,contacts,internal,use_friction):
 E,b,U,v,bounds,rs=system(m,s,u,k,contacts,internal,use_friction)
 r=linprog(np.zeros(E.shape[1]),A_eq=E/rs[:,None],b_eq=b/rs,A_ub=(U if len(U) else None),b_ub=(v if len(U) else None),bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
 if not r.success:return False,int(r.status),None
 x=r.x;res=E@x-b
 ok=bool(np.max(abs(res),initial=0.)<=1e-8 and (np.max(U@x,initial=0.)<=1e-8 if len(U) else True) and all(x[i]>=lo-1e-10 and x[i]<=hi+1e-10 for i,(lo,hi) in enumerate(bounds)))
 return ok,int(r.status),x
def main():
 O.mkdir(parents=True,exist_ok=False)
 result=json.loads((S/'RESULT.json').read_text())
 files=[Path(__file__),S/'RESULT.json',P/'POSE_0.npz',P/'POSE_1.npz',P/'POSE_2.npz',N/'REACTIONS_0.npz',N/'REACTIONS_1.npz',N/'REACTIONS_2.npz'];hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(scope='Rebuild3static systems from sealed arrays and independently re-solve only the default point plus each knob maximum;noMuJoCo/newcapture/modeledit',source_sha256=hashes))
 checks=[];calls=0
 for pose in range(3):
  m=ar(P/f'POSE_{pose}.npz');reaction=ar(N/f'REACTIONS_{pose}.npz');outer=m['J'].transpose(2,0,1).reshape(48,-1)
  # default point must reproduce the known sealed infeasibility
  ok,status,_=solve(m,1.,1.,1.,outer,np.zeros((48,0)),True);calls+=1
  reported=next(r for r in result['poses'] if r['pose']==pose)
  expected_default=reported['default_model']['status_outerbox']
  assert not ok and status==expected_default==2
  row=dict(pose=pose,default_infeasible=True,default_status=status,extremes={})
  for knob,value in [('s',10.),('u',3.),('k',10.)]:
   s=value if knob=='s' else 1.;u=value if knob=='u' else 1.;k=value if knob=='k' else 1.
   ok,status,_=solve(m,s,u,k,outer,np.zeros((48,0)),True);calls+=1
   knobrow=reported['knobs'][knob]
   assert not ok,'independent re-solve found feasibility the primary missed'
   assert knobrow['reachable'] is False and knobrow['verdict']=='NOT_REACHABLE_WITHIN_SCANNED_RANGE' and value==knobrow['maximum_scanned']
   row['extremes'][knob]=dict(multiplier=value,feasible=False,status=status)
  checks.append(row)
 assert result['calls']['linprog']==48
 out=dict(passed=True,source_sha256=hashes,pose_checks=checks,primary_grid=dict(poses=3,knobs='fmax s / activation ceiling u / friction k',total_primary_lp=result['calls']['linprog'],all_not_reachable=True),parent_lp_calls=calls,calls=dict(model_compile=0,forward=0,body_step=0,neural_step=0),limits='Independentconfirmationatdefaultandmaximumscannedmultiplier only; monotonicity of the grid is the primary report. No magnitude knob reaches feasibility for these3savedposes; this does not prove dynamic walking impossible and is not anatomical calibration or permission to edit the model.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',out);print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
