"""Decisive question: is the 1.3 s static obstruction about force MAGNITUDE at all?
Sealed matrices only. Tests (a) very large force/activation/friction multipliers and
(b) the unbounded positive conical hull: if no nonnegative muscle activation with
unbounded magnitude and unbounded contact force can satisfy equilibrium, then no
uniform strength increase or friction increase can ever help, because scaling cannot
change directions.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.optimize import linprog
ROOT=Path('/home/sensen/Desktop/engine修复验收')
P=ROOT/'s2-mapped-support-capacity/20261102T170741Z'
N=ROOT/'s2-native-support-constraints/20261102T172633Z'
O=ROOT/'s2-body-parameter-reachability/20261102T190000Z'
FLOOR=1e-4
def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def assemble(m,s,u,k,contacts,internal,scale_passive,unbounded,use_friction_rows=True):
 gain=m['gain']*s;bias=m['bias']*s;passive=m['qfrc_passive']*(s if scale_passive else 1.)
 A=m['moment'].T*gain[None,:];base=passive-m['qfrc_bias']+m['moment'].T@bias
 if unbounded:
  # No floor offset and no upper activation bound: pure conical hull of muscle columns.
  E=np.column_stack([A,contacts,internal]) if internal.size else np.column_stack([A,contacts]);b=-base
  hi=None
 else:
  b=-base-A@np.full(90,FLOOR)
  E=np.column_stack([A,contacts,internal]) if internal.size else np.column_stack([A,contacts]);hi=u-FLOOR
 nc=contacts.shape[1]//3 if use_friction_rows else contacts.shape[1];nint=int(internal.shape[1]);n=E.shape[1]
 contact_bounds=([(-np.inf,np.inf),(-np.inf,np.inf),(0.,np.inf)]*nc) if use_friction_rows else ([(0.,np.inf)]*nc)
 bounds=[(0.,np.inf if hi is None else max(hi,0.))]*90+contact_bounds+[(0.,np.inf)]*nint
 rows=[]
 for c in range(nc if use_friction_rows else 0):
  z=90+3*c;mu=float(m['mu'][c])*k
  for sx,sy in [(1.,0.),(-1.,0.),(0.,1.),(0.,-1.)]:
   row=np.zeros(n);row[z]=sx;row[z+1]=sy;row[z+2]=-mu;rows.append(row)
 U=np.array(rows).reshape(-1,n) if rows else np.zeros((0,n));v=np.zeros(len(U));rs=np.maximum(1,np.maximum(abs(E).max(1),abs(b)))
 r=linprog(np.zeros(n),A_eq=E/rs[:,None],b_eq=b/rs,A_ub=(U if len(U) else None),b_ub=(v if len(U) else None),bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
 if not r.success:return False,None
 x=r.x;res=E@x-b
 ok=bool(np.max(abs(res),initial=0.)<=1e-8 and (np.max(U@x,initial=0.)<=1e-8 if len(U) else True))
 return ok,x
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'RESULT.json').exists()
 m=ar(P/'POSE_1.npz');rea=ar(N/'REACTIONS_1.npz')
 files=[Path(__file__),P/'POSE_1.npz',N/'REACTIONS_1.npz'];hashes={str(p):sha(p) for p in files}
 outer=m['J'].transpose(2,0,1).reshape(48,-1);foot=rea['native_foot_columns'];internal=rea['internal_columns']
 calls=0;rows=[]
 for knob,grid in [('fmax_gain_bias_only',[1.,10.,100.,1000.,1e4]),('fmax_all_proportional',[1.,10.,100.,1000.,1e4]),('activation_ceiling',[1.,10.,100.,1000.,1e4]),('friction_scale',[1.,10.,100.,1000.,1e4])]:
  first=None
  for value in grid:
   s=value if knob.startswith('fmax') else 1.;u=value if knob=='activation_ceiling' else 1.;k=value if knob=='friction_scale' else 1.
   ok,_=assemble(m,s,u,k,outer,internal,knob=='fmax_all_proportional',False);calls+=1
   if ok:first=value;break
  rows.append(dict(knob=knob,grid=grid,first_feasible=first,reached=first is not None));print(json.dumps(rows[-1]),flush=True)
 cone={};witness={}
 for mode,contacts in [('outerbox',outer),('native',foot)]:
  ok,x=assemble(m,1.,1.,1.,contacts,internal,False,True,use_friction_rows=(mode=='outerbox'));calls+=1;cone[mode]=bool(ok)
  if ok:
   a=x[:90];witness[mode]=dict(max_activation=float(a.max()),sum_activation=float(a.sum()),min_contact_fz=None if contacts.shape[1]==0 else float(min(x[90+3*c+2] for c in range(contacts.shape[1]//3))))
 result=dict(passed=True,source_sha256=hashes,scope='Single saved pose 1.3s; conditional static equilibrium only; no model or XML change',magnitude_scan=rows,unbounded_conical_hull=cone,conical_witness=witness,calls=dict(linprog=calls,model_compile=0,forward=0,body_step=0,neural_step=0),interpretation='The unbounded test allows any nonnegative activation magnitude and any nonnegative contact magnitude inside the friction cone. If it is infeasible, no uniform strength, activation-ceiling or friction increase can make this saved pose statically feasible, because none of them changes force directions; only posture, attachment geometry, contact layout or internal constraints could. Conditional on this pose/contact set/kinematics only; static feasibility is not required for dynamic walking, and none of this is a neural result or permission to edit the model.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
