"""Static common/active-only gain LPs with actual passive/active decomposition."""
from pathlib import Path
import json,time
import numpy as np,mujoco
from scipy.optimize import linprog
from flygym.compose.fly.musculoskeletal import _load_mjcf
from diagnose_fixed_muscle_authority import O,S,dump,sha

def main():
 start=time.time();pre=O/'DECOMPOSITION_PREREGISTRATION.json';assert not pre.exists();dump(pre,dict(source_hashes={str(p):sha(p) for p in [Path(__file__),O/'STATIC_DETAIL.json',S/'MODEL.xml']},scope='Saved matrix common Fmax scale s>=0 unbounded exact balance LP and minimum residual; then actual gain-only force0 decomposes passive actuator force while biasprm unchanged. Active-only gain a>=0 unbounded exact/minresidual LPs. Engineering NOT_ESTABLISHED, no runtime controller.',calls='1model3forward(nativefloor,gain0floor,gain2candidate proof),0body0neural',substitution='z=s*x or z=a*x,0<=z<=scale; scale unboundedpositive includes0. Exact LP minimizes scale subject all48 balance, alternative epigraph minimizes residual.',selection='Only if inner exact feasible choose nextpower2>=minimum; verify chosen fixed scale before candidate use. Positive outer global residual excludes exact balance only fixedpose declared relaxedcontacts+limits.'))
 sd=json.loads((O/'STATIC_DETAIL.json').read_text());A=np.array(sd['active_force_matrix']);J=np.array(sd['contact_force_matrix']);L=np.array(sd['joint_limit_force_matrix']).reshape(48,-1);base=np.array(sd['native_actuator_floor_force']);target=np.array(sd['qfrc_bias'])-np.array(sd['qfrc_passive']);w=sd['bodyweight_native'];contacts=sd['contacts'];nc=len(contacts);nl=L.shape[1];counts=dict(model_load=0,forward=0,body_steps=0,neural_steps=0);rows=[];details={}
 def solve(varying,constant,label,inner,exact):
  B=np.column_stack([A,varying,J,L])/w;y=(target-constant)/w;n=B.shape[1];mat=[];rhs=[]
  if not exact:mat.extend([np.c_[B,-np.ones(48)],np.c_[-B,-np.ones(48)]]);rhs.extend([y,-y])
  for j,c in enumerate(contacts):
   h=np.zeros((4,n+(not exact)));dirs=[(1,1),(1,-1),(-1,1),(-1,-1)] if inner else [(1,0),(-1,0),(0,1),(0,-1)]
   for k,(sx,sy) in enumerate(dirs):h[k,67+3*j:67+3*j+3]=[sx,sy,-c['mu']]
   mat.append(h);rhs.append(np.zeros(4))
  h=np.zeros((66,n+(not exact)));h[np.arange(66),np.arange(66)]=1;h[:,66]=-1;mat.append(h);rhs.append(np.zeros(66));bounds=[(0,None)]*67+[(None,None),(None,None),(0,None)]*nc+[(0,None)]*nl
  if not exact:bounds.append((0,None))
  obj=np.zeros(n+(not exact));obj[66 if exact else -1]=1;sol=linprog(obj,A_ub=np.vstack(mat),b_ub=np.concatenate(rhs),A_eq=B if exact else None,b_eq=y if exact else None,bounds=bounds,method='highs');r=dict(label=label,inner=inner,exact=exact,success=bool(sol.success),status=int(sol.status),message=sol.message,objective=None if not sol.success else float(sol.fun),scale=None if not sol.success else float(sol.x[66]));rows.append(r)
  if sol.success:details[label]=dict(solution=sol.x,force_residual_over_weight=B@sol.x[:n]-y,variable_base_force=varying,constant_force=constant,inner=inner,exact=exact)
  dump(O/'DECOMPOSITION_RESULTS.json',rows);dump(O/'DECOMPOSITION_SOLUTIONS.json',details);print(json.dumps(r),flush=True);return r,sol
 for inner in [False,True]:
  for exact in [True,False]:solve(base,np.zeros(48),'common_'+('inner' if inner else 'outer')+('_exact' if exact else '_residual'),inner,exact)
 m=_load_mjcf(str(S/'MODEL.xml')).compile();counts['model_load']+=1;d=mujoco.MjData(m);q=np.array(sd['qpos']);G=np.array(sd['G']);oldg=m.actuator_gainprm.copy();oldb=m.actuator_biasprm.copy()
 def probe(x):
  d.qpos[:]=q;d.qvel[:]=0;d.act[:]=.0001+.9999*(G@x);d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);counts['forward']+=1;return d.qfrc_actuator.copy()
 native=probe(np.zeros(66));np.testing.assert_allclose(native,base,atol=1e-12,rtol=0);m.actuator_gainprm[:,2]=0;passive=probe(np.zeros(66));activefloor=base-passive;m.actuator_gainprm[:,2]=oldg[:,2]*2;x=np.full(66,.1);actual=probe(x);pred=passive+2*(activefloor+A@x);np.testing.assert_allclose(actual,pred,atol=1e-10,rtol=1e-12);np.testing.assert_array_equal(m.actuator_biasprm,oldb)
 dump(O/'ACTIVE_PASSIVE_DETAIL.json',dict(passive_actuator_generalized_force=passive,active_floor_generalized_force=activefloor,native_floor_generalized_force=base,validation_gain_scale=2,validation_pool_activation=x,validation_actual_force=actual,validation_predicted_force=pred,validation_max_error=float(np.max(np.abs(actual-pred))),biasprm_unchanged=True,native_gainprm=oldg,native_biasprm=oldb,new_gainprm=m.actuator_gainprm.copy()))
 for inner in [False,True]:
  for exact in [True,False]:solve(activefloor,passive,'active_only_'+('inner' if inner else 'outer')+('_exact' if exact else '_residual'),inner,exact)
 dump(O/'DECOMPOSITION_EXECUTION.json',dict(counts=counts,lp_solves=len(rows),exit_code=0,runtime_s=time.time()-start));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
