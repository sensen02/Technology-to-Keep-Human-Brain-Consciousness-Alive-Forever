"""Converged LP force-cone relaxation; not a MuJoCo trajectory or controller."""
from pathlib import Path
import json,time
import numpy as np,mujoco
from scipy.optimize import linprog
from flygym.compose.fly.musculoskeletal import _load_mjcf
from diagnose_settled_native_support import S,O,dump,sha

def main():
 start=time.time();src=[Path(__file__),S/'MODEL.xml',S/'HANDOVER_BEFORE.npz',O/'POOL_MATRIX.npz'];dump(O/'CONE_PREREGISTRATION.json',dict(source_hashes={str(p):sha(p) for p in src},scope='Fixed actual settled074 qpos, free contact forces instead of soft-contact solver. 66tied nativeactivation, unresolvedfloor. LP minimum all48 forcebalance L-infinity residual with nonnegative normals/outer square friction pyramid. No actual body/neural integration, no runtime optimizer.',friction='|fx| <= mu*fz and |fy| <= mu*fz: OUTER square contains true circular cone. Feasibility may violate circle, check reported ratios. Infeasibility strong for fixed contacts. Actual contact friction used; no normal-force upper bound.',physiology='NOT_ESTABLISHED'))
 m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m);q=np.load(S/'HANDOVER_BEFORE.npz')['qpos'];G=np.load(O/'POOL_MATRIX.npz')['G'];w=float(m.body_mass.sum()*abs(m.opt.gravity[2]));counts=dict(model_load=1,forward=0,body_steps=0,neural_steps=0)
 def probe(x):
  d.qpos[:]=q;d.qvel[:]=0;d.act[:]=.0001+.9999*(G@x);d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);counts['forward']+=1;return d.qfrc_actuator.copy()
 base=probe(np.zeros(66));bias=d.qfrc_bias.copy();passive=d.qfrc_passive.copy();contacts=[];blocks=[]
 for k in range(d.ncon):
  co=d.contact[k];a,b=int(co.geom1),int(co.geom2);p1=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;p2=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
  if not(p1 or p2):continue
  g=b if p1 else a;name=m.geom(g).name
  if not(any(name.startswith(l) for l in ['LF','LM','LH','RF','RM','RH']) and 'Tars' in name):continue
  jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,co.pos,int(m.geom_bodyid[g]));blocks.append(jp.T);contacts.append(dict(geom=name,position=co.pos.copy(),mu=float(co.friction[0])))
 A=np.column_stack([probe(np.eye(66)[j])-base for j in range(66)]);J=np.column_stack(blocks);B=np.column_stack([A,J])/w;target=(bias-passive-base)/w;n=B.shape[1];nc=len(contacts)
 ub=[np.c_[B,-np.ones(m.nv)],np.c_[-B,-np.ones(m.nv)]];rhs=[target,-target]
 for j,c in enumerate(contacts):
  mat=np.zeros((4,n+1))
  for row,(sx,sy) in enumerate([(1,0),(-1,0),(0,1),(0,-1)]):mat[row,66+3*j:66+3*j+3]=[sx,sy,-c['mu']]
  ub.append(mat);rhs.append(np.zeros(4))
 objective=np.zeros(n+1);objective[-1]=1.;bounds=[(0,1)]*66+[(None,None),(None,None),(0,None)]*nc+[(0,None)];sol=linprog(objective,A_ub=np.vstack(ub),b_ub=np.concatenate(rhs),bounds=bounds,method='highs');assert sol.success,sol.message
 x=sol.x;res=B@x[:-1]-target;forces=x[66:-1].reshape(nc,3);summary=dict(success=bool(sol.success),status=int(sol.status),message=sol.message,optimal_linf_residual_over_weight=float(x[-1]),l2_residual_over_weight=float(np.linalg.norm(res)),vertical_ground_force_over_weight=float(forces[:,2].sum()/w),total_ground_force_over_weight=(forces.sum(0)/w),max_contact_force_over_weight=float(np.max(np.linalg.norm(forces,axis=1))/w),maximum_circular_friction_ratio=float(max(np.linalg.norm(f[:2])/(c['mu']*f[2]) if f[2]>1e-10 else 0. for f,c in zip(forces,contacts))),contacts=nc,minimum_native_forcecone_balance_found=bool(x[-1]<1e-7),scope='LP relaxation permits arbitrary within-cone contact forces; even exact feasible solution is not a physical MuJoCo equilibrium or neural success. Positive optimal residual excludes exact equilibrium even with the outer friction square at this fixed pose; no global posture impossibility claim.')
 dump(O/'CONE_RESULT.json',summary);dump(O/'CONE_DETAIL.json',dict(activation_matrix=G,active_force_matrix=A,contact_force_matrix=J,target_generalized_force=bias-passive-base,pool_activation=x[:66],contact_forces=forces,contacts=contacts,force_residual_over_weight=res,linprog_solution=x,bodyweight_native=w));dump(O/'CONE_EXECUTION.json',dict(counts=counts,runtime_s=time.time()-start,exit_code=0));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
if __name__=='__main__':main()
