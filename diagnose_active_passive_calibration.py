"""Static active/passive engineering-scale LP and force-direction certificate."""
from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import linprog
from flygym.compose.fly.musculoskeletal import _load_mjcf
from diagnose_fixed_muscle_authority import O,S,dump,sha

def main():
 start=time.time();pre=O/'ACTIVE_PASSIVE_PREREGISTRATION.json';assert not pre.exists();dump(pre,dict(source_hashes={str(p):sha(p) for p in [Path(__file__),O/'STATIC_DETAIL.json',O/'ACTIVE_PASSIVE_DETAIL.json',S/'MODEL.xml']},scope='Static joint active and passive force calibration, a in[.015625,16],b in[0,1]; z=a*x,0<=z<=a; active floor*a plus passive actuator*b. Exact inner/outer and minimum residual. No runtime optimizer or physiology calibration.',extra='Export native unbounded-positive active LP primal/dual certificate, rank and weak DOF names. 1model0forward for exact native metadata names; no physics integration.'))
 d=json.loads((O/'STATIC_DETAIL.json').read_text());dec=json.loads((O/'ACTIVE_PASSIVE_DETAIL.json').read_text());A=np.array(d['active_force_matrix']);J=np.array(d['contact_force_matrix']);L=np.array(d['joint_limit_force_matrix']).reshape(48,-1);floor=np.array(dec['active_floor_generalized_force']);pa=np.array(dec['passive_actuator_generalized_force']);target=np.array(d['qfrc_bias'])-np.array(d['qfrc_passive']);w=d['bodyweight_native'];contacts=d['contacts'];nc=len(contacts);nl=L.shape[1];rows=[];raw={}
 def solve(inner,exact):
  label=('inner' if inner else 'outer')+('_exact' if exact else '_residual');B=np.column_stack([A,floor,pa,J,L])/w;y=target/w;n=B.shape[1];ub=[];rhs=[]
  if not exact:ub.extend([np.c_[B,-np.ones(48)],np.c_[-B,-np.ones(48)]]);rhs.extend([y,-y])
  for j,c in enumerate(contacts):
   h=np.zeros((4,n+(not exact)));dirs=[(1,1),(1,-1),(-1,1),(-1,-1)] if inner else [(1,0),(-1,0),(0,1),(0,-1)]
   for k,(sx,sy) in enumerate(dirs):h[k,68+3*j:68+3*j+3]=[sx,sy,-c['mu']]
   ub.append(h);rhs.append(np.zeros(4))
  h=np.zeros((66,n+(not exact)));h[np.arange(66),np.arange(66)]=1;h[:,66]=-1;ub.append(h);rhs.append(np.zeros(66));bounds=[(0,None)]*66+[(.015625,16),(0,1)]+[(None,None),(None,None),(0,None)]*nc+[(0,None)]*nl
  if not exact:bounds.append((0,None))
  obj=np.zeros(n+(not exact));obj[66 if exact else -1]=1;sol=linprog(obj,A_ub=np.vstack(ub),b_ub=np.concatenate(rhs),A_eq=B if exact else None,b_eq=y if exact else None,bounds=bounds,method='highs');r=dict(label=label,success=bool(sol.success),status=int(sol.status),objective=float(sol.fun) if sol.success else None,active_scale=float(sol.x[66]) if sol.success else None,passive_scale=float(sol.x[67]) if sol.success else None);rows.append(r)
  if sol.success:raw[label]=dict(solution=sol.x,A_ub=np.vstack(ub),b_ub=np.concatenate(rhs),A_eq=B if exact else None,b_eq=y if exact else None,bounds=bounds,objective=obj,force_residual_over_weight=B@sol.x[:n]-y,inequality_dual=sol.ineqlin.marginals,lower_dual=sol.lower.marginals,upper_dual=sol.upper.marginals)
  print(json.dumps(r),flush=True);dump(O/'ACTIVE_PASSIVE_RESULTS.json',rows);dump(O/'ACTIVE_PASSIVE_SOLUTIONS.json',raw);return r,sol
 selected=None
 for inner in [False,True]:
  r,sol=solve(inner,True);solve(inner,False)
  if inner and r['success']:selected=(r,sol.x)
 if selected:
  r,x=selected;np.savez_compressed(O/'ACTIVE_PASSIVE_CANDIDATE.npz',active_force_scale=x[66],passive_force_scale=x[67],pool_activation=x[:66]/x[66],G=d['G'],qpos=d['qpos'])
 # Native unbounded active directions: independent dual certificate with no pool upper bounds.
 B=np.column_stack([A,J,L])/w;y=(target-np.array(d['native_actuator_floor_force']))/w;n=B.shape[1];ub=[np.c_[B,-np.ones(48)],np.c_[-B,-np.ones(48)]];rhs=[y,-y]
 for j,c in enumerate(contacts):
  h=np.zeros((4,n+1));h[:,66+3*j:66+3*j+3]=[[1,0,-c['mu']],[-1,0,-c['mu']],[0,1,-c['mu']],[0,-1,-c['mu']]];ub.append(h);rhs.append(np.zeros(4))
 bounds=[(0,None)]*66+[(None,None),(None,None),(0,None)]*nc+[(0,None)]*nl+[(0,None)];obj=np.zeros(n+1);obj[-1]=1;H=np.vstack(ub);v=np.concatenate(rhs);sol=linprog(obj,A_ub=H,b_ub=v,bounds=bounds,method='highs');assert sol.success;dual=sol.ineqlin.marginals;dualobj=float(v@dual);np.testing.assert_allclose(dualobj,sol.fun,atol=1e-9,rtol=0)
 m=_load_mjcf(str(S/'MODEL.xml')).compile();names=['root_x','root_y','root_z','root_roll','root_pitch','root_yaw']+[m.joint(int(m.dof_jntid[i])).name for i in range(6,48)];forceweights=dual[:48]-dual[48:96];order=np.argsort(-abs(forceweights));rank=int(np.linalg.matrix_rank(A));dofs=[dict(index=int(i),name=names[i],dual_force_weight=float(forceweights[i]),active_row_norm=float(np.linalg.norm(A[i])),largest_single_positive_slope=float(max(0,A[i].max())),largest_single_negative_slope=float(min(0,A[i].min())),contact_row_norm=float(np.linalg.norm(J[i]))) for i in range(48)]
 dump(O/'NATIVE_UNBOUNDED_DUAL_CERTIFICATE.json',dict(A_ub=H,b_ub=v,bounds=bounds,objective=obj,solution=sol.x,inequality_dual=dual,lower_dual=sol.lower.marginals,upper_dual=sol.upper.marginals,primal_objective=float(sol.fun),dual_objective=dualobj,active_force_matrix_rank=rank,dofs=dofs,largest_dual_dofs=[dofs[i] for i in order[:12]],scope='Separating force weights for native fixed-pose outer-square LP even with unbounded positive 66pool activation; not proof for other postures.'))
 dump(O/'ACTIVE_PASSIVE_EXECUTION.json',dict(model_load=1,forward=0,body_steps=0,neural_steps=0,lp_solves=5,runtime_s=time.time()-start,exit_code=0));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
