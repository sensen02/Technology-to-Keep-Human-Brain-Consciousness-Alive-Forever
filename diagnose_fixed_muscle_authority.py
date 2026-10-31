"""Offline fixed-Fmax force-authority calibration; never a runtime controller."""
from pathlib import Path
import json,time,hashlib,traceback
import numpy as np,mujoco
from scipy.optimize import linprog
from flygym.compose.fly.musculoskeletal import _load_mjcf
S=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
P=Path('/home/sensen/Desktop/engine修复验收/s5-settled-muscle-support/20261028T076000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-fixed-muscle-authority/20261028T081000Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')
def main():
 start=time.time();O.mkdir(parents=True,exist_ok=False);src=[Path(__file__),S/'MODEL.xml',S/'HANDOVER_BEFORE.npz',P/'POOL_MATRIX.npz'];hashes={str(p):sha(p) for p in src}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,scope='Offline actual settled074 pose qvel0, 66shared native muscles16unresolvedfloor. Uniform Fmax scaling affects both active and passive actuator force; not gravity/mass/contact/geometric change. No integration or neural steps, never runtime LP.',force_parameter='All90 gainprm[2] and biasprm[2] positive explicit Fmax; multiply both by scalar s.',grid=[1,2,4,8,16],refinement='Stop ordered grid at first INNER diamond friction LP t<1e-7; 10 bisections if bracketed. Outer square also evaluated. Selected actual-trial scale is first feasible power-of-two grid point, not neural-outcome selection.',model_counts='1preflightcompile0forward previously inspected Fmax semantics, plus1model here with67nativeforward+2scaledvalidationforward',physiology='NOT_ESTABLISHED',feasibility_scope='Static force-cone feasibility with active joint limits, not actual soft-contact equilibrium or neural autonomy.'))
 counts=dict(preflight_model_load=1,model_load=0,forward=0,body_steps=0,neural_steps=0);error=None;rows=[];details={}
 try:
  m=_load_mjcf(str(S/'MODEL.xml')).compile();counts['model_load']+=1;d=mujoco.MjData(m);q=np.load(S/'HANDOVER_BEFORE.npz')['qpos'];G=np.load(P/'POOL_MATRIX.npz')['G'];w=float(m.body_mass.sum()*abs(m.opt.gravity[2]));original_gain=m.actuator_gainprm.copy();original_bias=m.actuator_biasprm.copy();assert np.all(original_gain[:,2]>0) and np.array_equal(original_gain,original_bias)
  def probe(x):
   d.qpos[:]=q;d.qvel[:]=0;d.act[:]=.0001+.9999*(G@x);d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);counts['forward']+=1;return d.qfrc_actuator.copy()
  base=probe(np.zeros(66));bias=d.qfrc_bias.copy();passive=d.qfrc_passive.copy();contacts=[];blocks=[]
  if d.efc_J.size==d.nefc*m.nv:E=d.efc_J.reshape(d.nefc,m.nv).copy()
  else:
   E=np.zeros((d.nefc,m.nv))
   for row in range(d.nefc):
    adr=int(d.efc_J_rowadr[row]);nnz=int(d.efc_J_rownnz[row]);E[row,d.efc_J_colind[adr:adr+nnz]]=d.efc_J[adr:adr+nnz]
  limits=np.flatnonzero(np.asarray(d.efc_type)==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT));L=E[limits].T;nl=len(limits)
  for k in range(d.ncon):
   co=d.contact[k];a,b=int(co.geom1),int(co.geom2);p1=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;p2=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
   if not(p1 or p2):continue
   g=b if p1 else a;name=m.geom(g).name
   if not(any(name.startswith(l) for l in ['LF','LM','LH','RF','RM','RH']) and 'Tars' in name):raise AssertionError('nonfoot contact at anchor')
   jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,co.pos,int(m.geom_bodyid[g]));blocks.append(jp.T);contacts.append(dict(geom=name,position=co.pos.copy(),mu=float(co.friction[0])))
  A=np.column_stack([probe(np.eye(66)[j])-base for j in range(66)]);J=np.column_stack(blocks);nc=len(contacts)
  static=dict(G=G,qpos=q,active_force_matrix=A,contact_force_matrix=J,joint_limit_force_matrix=L,joint_limit_efc_rows=limits,native_actuator_floor_force=base,qfrc_bias=bias,qfrc_passive=passive,bodyweight_native=w,contacts=contacts,native_gainprm=original_gain,native_biasprm=original_bias)
  dump(O/'STATIC_DETAIL.json',static)
  def solve(s,inner,label):
   B=np.column_stack([s*A,J,L])/w;y=(bias-passive-s*base)/w;n=B.shape[1];ub=[np.c_[B,-np.ones(48)],np.c_[-B,-np.ones(48)]];rhs=[y,-y]
   for j,c in enumerate(contacts):
    mat=np.zeros((4,n+1));dirs=[(1,1),(1,-1),(-1,1),(-1,-1)] if inner else [(1,0),(-1,0),(0,1),(0,-1)]
    for row,(sx,sy) in enumerate(dirs):mat[row,66+3*j:66+3*j+3]=[sx,sy,-c['mu']]
    ub.append(mat);rhs.append(np.zeros(4))
   obj=np.zeros(n+1);obj[-1]=1;bounds=[(0,1)]*66+[(None,None),(None,None),(0,None)]*nc+[(0,None)]*nl+[(0,None)];sol=linprog(obj,A_ub=np.vstack(ub),b_ub=np.concatenate(rhs),bounds=bounds,method='highs');assert sol.success,sol.message
   x=sol.x;res=B@x[:-1]-y;f=x[66:66+3*nc].reshape(nc,3);ratios=[np.linalg.norm(ff[:2])/(c['mu']*ff[2]) if ff[2]>1e-10 else 0 for ff,c in zip(f,contacts)];r=dict(label=label,scale=s,inner=inner,optimal_linf_residual_over_weight=float(sol.fun),l2_residual_over_weight=float(np.linalg.norm(res)),maximum_circular_friction_ratio=float(max(ratios)),vertical_ground_force_over_weight=float(f[:,2].sum()/w),feasible=bool(sol.fun<1e-7),global_lp_success=True);rows.append(r);details[label]=dict(solution=x,force_residual_over_weight=res,pool_activation=x[:66],contact_forces=f,joint_limit_multipliers=x[66+3*nc:-1],scale=s,inner=inner);dump(O/'RESULTS.json',rows);dump(O/'SOLUTIONS.json',details);print(json.dumps(r),flush=True);return r,x[:66]
  lower=1.;selected=None;pool=None
  for s in [1.,2.,4.,8.,16.]:
   solve(s,False,'outer_grid_'+str(s));r,x=solve(s,True,'inner_grid_'+str(s))
   if r['feasible']:selected=s;pool=x;break
   lower=s
  bracket=None
  if selected is not None:
   upper=selected
   if selected>1:
    for k in range(10):
     mid=(lower+upper)/2;r,_=solve(mid,True,'inner_bisect_'+str(k))
     if r['feasible']:upper=mid
     else:lower=mid
   bracket=[lower,upper]
  validate_scale=selected if selected is not None else 16.;m.actuator_gainprm[:,2]=original_gain[:,2]*validate_scale;m.actuator_biasprm[:,2]=original_bias[:,2]*validate_scale
  targetpool=pool if pool is not None else details['inner_grid_16.0']['pool_activation'];actualfloor=probe(np.zeros(66));actualcandidate=probe(targetpool);predfloor=validate_scale*base;predcandidate=validate_scale*(base+A@targetpool)
  np.testing.assert_allclose(actualfloor,predfloor,atol=1e-10,rtol=1e-12);np.testing.assert_allclose(actualcandidate,predcandidate,atol=1e-10,rtol=1e-12)
  dump(O/'FORCE_SCALING_VALIDATION.json',dict(scale=validate_scale,native_gainprm=original_gain,native_biasprm=original_bias,new_gainprm=m.actuator_gainprm.copy(),new_biasprm=m.actuator_biasprm.copy(),actual_floor_force=actualfloor,predicted_floor_force=predfloor,actual_candidate_force=actualcandidate,predicted_candidate_force=predcandidate,floor_max_error=float(np.max(np.abs(actualfloor-predfloor))),candidate_max_error=float(np.max(np.abs(actualcandidate-predcandidate))),physiology='NOT_ESTABLISHED',scope='Direct native mj_forward force proof; no integration or runtime optimizer'))
  result=dict(selected_trial_scale=selected,minimum_inner_feasible_scale_bracket=bracket,feasible_by_scale16=selected is not None,contacts=nc,active_joint_limits=nl,scope='Declared static engineering calibration; feasible LP not soft-contact equilibrium or neural behavior, changing Fmax also changes passive forces for all90including16unresolved.');dump(O/'SELECTION.json',result)
  if selected is not None:np.savez_compressed(O/'CANDIDATE.npz',force_scale=selected,pool_activation=pool,actuator_control=G@pool,G=G,qpos=q)
  print('SELECTION',json.dumps(result),flush=True)
 except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());dump(O/'FAILURE.json',error);raise
 finally:
  dump(O/'EXECUTION.json',dict(counts=counts,exit_code=0 if error is None else 1,error=error,runtime_s=time.time()-start,source_unchanged=all(sha(p)==h for p,h in hashes.items())));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
