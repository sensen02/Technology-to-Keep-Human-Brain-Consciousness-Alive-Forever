"""Saved-matrix LP refinement over continuous uniform Fmax scale; no new physics."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.optimize import linprog
from diagnose_fixed_muscle_authority import O,dump,sha

def main():
 start=time.time();src=[Path(__file__),O/'STATIC_DETAIL.json'];pre=O/'REFINEMENT_PREREGISTRATION.json';assert not pre.exists();dump(pre,dict(source_hashes={str(p):sha(p) for p in src},scope='Saved-matrix LPs only. Uniform Fmax also changes passive muscle forces so feasibility need not be monotonic; earlier prospective bisection would not in general prove minimum, and was not executed.',lower_grid=[.5,.25,.125,.0625,.03125],joint_scale_bounds=[.015625,16],joint_substitution='z=s*x,0<=z_j<=s. Generalized force A z + base*s + J f + L lambda = bias-passive. Global min L-infinity residual, not bisection.',extra='Scale1 active pool activation unbounded positive diagnostic to separate bound vs direction. No runtime control.',new_models=0,new_forward=0,new_body_steps=0,new_neural_steps=0))
 d=json.loads((O/'STATIC_DETAIL.json').read_text());A=np.array(d['active_force_matrix']);J=np.array(d['contact_force_matrix']);L=np.array(d['joint_limit_force_matrix']).reshape(48,-1);base=np.array(d['native_actuator_floor_force']);target=np.array(d['qfrc_bias'])-np.array(d['qfrc_passive']);w=d['bodyweight_native'];contacts=d['contacts'];nc=len(contacts);nl=L.shape[1];rows=[];details={}
 def solve(s,inner,label,joint=False,unbounded=False):
  if joint:B=np.column_stack([A,base,J,L])/w;y=target/w;off=67
  else:B=np.column_stack([s*A,J,L])/w;y=(target-s*base)/w;off=66
  n=B.shape[1];ub=[np.c_[B,-np.ones(48)],np.c_[-B,-np.ones(48)]];rhs=[y,-y]
  for j,c in enumerate(contacts):
   mat=np.zeros((4,n+1));dirs=[(1,1),(1,-1),(-1,1),(-1,-1)] if inner else [(1,0),(-1,0),(0,1),(0,-1)]
   for k,(sx,sy) in enumerate(dirs):mat[k,off+3*j:off+3*j+3]=[sx,sy,-c['mu']]
   ub.append(mat);rhs.append(np.zeros(4))
  if joint:
   h=np.zeros((66,n+1));h[np.arange(66),np.arange(66)]=1;h[:,66]=-1;ub.append(h);rhs.append(np.zeros(66));bounds=[(0,None)]*66+[(.015625,16)]
  else:bounds=[(0,None) if unbounded else (0,1)]*66
  bounds+=[(None,None),(None,None),(0,None)]*nc+[(0,None)]*nl+[(0,None)];obj=np.zeros(n+1);obj[-1]=1.;sol=linprog(obj,A_ub=np.vstack(ub),b_ub=np.concatenate(rhs),bounds=bounds,method='highs');assert sol.success;actuals=float(sol.x[66]) if joint else s;pool=sol.x[:66]/actuals if joint else sol.x[:66];f=sol.x[off:off+3*nc].reshape(nc,3)
  r=dict(label=label,scale=actuals,joint_scale_optimized=joint,inner=inner,unbounded_active=unbounded,optimal_linf_residual_over_weight=float(sol.fun),feasible=bool(sol.fun<1e-7),pool_max=float(pool.max()),maximum_circular_friction_ratio=float(max(np.linalg.norm(ff[:2])/(c['mu']*ff[2]) if ff[2]>1e-10 else 0 for ff,c in zip(f,contacts))));rows.append(r);details[label]=dict(solution=sol.x,pool_activation=pool,scale=actuals,inner=inner,joint=joint,unbounded=unbounded,force_residual_over_weight=B@sol.x[:-1]-y);dump(O/'REFINEMENT_RESULTS.json',rows);dump(O/'REFINEMENT_SOLUTIONS.json',details);print(json.dumps(r),flush=True);return r,pool
 for s in [.5,.25,.125,.0625,.03125]:
  for inner in [False,True]:solve(s,inner,('inner' if inner else 'outer')+'_low_'+str(s))
 for inner in [False,True]:solve(1.,inner,('inner' if inner else 'outer')+'_active_unbounded',unbounded=True)
 best=None
 for inner in [False,True]:
  r,pool=solve(None,inner,('inner' if inner else 'outer')+'_continuous_scale',joint=True)
  if inner:best=(r,pool)
 r,pool=best
 dump(O/'REFINEMENT_SELECTION.json',dict(inner_global_optimized_scale=r['scale'],inner_global_min_residual=r['optimal_linf_residual_over_weight'],feasible=bool(r['feasible']),scale_bounds=[.015625,16],scope='Continuous common Fmax scale optimum over declared interval; static ideal-contact feasibility only, no dynamic/neural success.'))
 if r['feasible']:np.savez_compressed(O/'REFINEMENT_CANDIDATE.npz',force_scale=r['scale'],pool_activation=pool,actuator_control=np.array(d['G'])@pool,G=d['G'],qpos=d['qpos'])
 dump(O/'REFINEMENT_EXECUTION.json',dict(new_model_load=0,new_forward=0,new_body_steps=0,new_neural_steps=0,lp_solves=len(rows),exit_code=0,runtime_s=time.time()-start));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
