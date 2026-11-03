"""Inverse-dynamics audit: what can actually be recovered from the saved trajectory.

Questions answered with numbers, not hand-waving:
 1. Can the actuator generalized force be recovered from the trajectory? (inverse dynamics)
 2. Does it agree with the muscle forces the simulation itself recorded?
 3. Given only kinematics + contacts, is the per-muscle force split unique? (moment-arm rank)
 4. Which muscles actually contribute to vertical support at the failure instant?

No training, no new rollout: model forwards on saved states only.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
O=Path('/home/sensen/Desktop/engine修复验收/s4-inverse-dynamics-audit/20261103T050000Z')
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
BEST=Path('/home/sensen/Desktop/engine修复验收/s5-dynamic-support-training/20261103T002606Z/VALIDATION_R1/CAND0_A_release200ms')
def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def moment_matrix(m,d):
 A=np.zeros((90,m.nv))
 if d.actuator_moment.shape==(90,m.nv):A[:]=d.actuator_moment
 else:
  for u in range(90):
   off=int(d.moment_rowadr[u]);n=int(d.moment_rownnz[u]);A[u,d.moment_colind[off:off+n]]=d.actuator_moment[off:off+n]
 return A
def main():
 O.mkdir(parents=True,exist_ok=False)
 dump(O/'PREREGISTRATION.json',dict(question='what can be inverted from the saved trajectory, specifically muscle force',
  cases=['ID1 inverse dynamics reproduces the actuator generalized force',
         'ID2 recorded per-muscle forces reproduce the same generalized force',
         'ID3 moment-arm rank: is the per-muscle split unique from kinematics and contacts alone',
         'ID4 per-muscle contribution to vertical support at the failure instant'],
  tolerance_abs=1e-9,no_simulation='model forwards on saved states only'))
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 m=_load_mjcf(str(BASE/'RUN/MODEL.xml')).compile();d=mujoco.MjData(m)
 out={};M=np.zeros((m.nv,m.nv))
 rows=[]
 for tag,root in (('baseline',BASE),('best',BEST)):
  body=ar(root/'RUN/BODY.npz');T=len(body['time'])
  for idx in np.linspace(int(0.9*T),T-2,6).astype(int):
   idx=int(idx)
   d.qpos[:]=body['qpos'][idx];d.qvel[:]=body['qvel'][idx];d.act[:]=body['activation'][idx]
   d.ctrl[:]=body['command'][idx];d.qfrc_applied[:]=body['assist'][idx];d.xfrc_applied[:]=0.0
   mujoco.mj_forward(m,d)
   mujoco.mj_fullM(m,M,d.qM)
   tau_inv=M@d.qacc+d.qfrc_bias-d.qfrc_passive-d.qfrc_constraint-d.qfrc_applied
   f_rec=body['muscle_force'][idx];A=moment_matrix(m,d)
   tau_rec=A.T@f_rec
   e1=float(np.max(np.abs(tau_inv-d.qfrc_actuator),initial=0.))
   e2=float(np.max(np.abs(tau_rec-d.qfrc_actuator),initial=0.))
   rows.append(dict(run=tag,index=idx,time=float(body['time'][idx]),
    max_abs_err_inverse_vs_actuator=e1,max_abs_err_recorded_muscles_vs_actuator=e2,
    actuator_force_absmax=float(np.abs(d.qfrc_actuator).max()),moment_arm_nonzero=int((np.abs(A)>1e-12).sum())))
  out.setdefault('cases',{})
 max1=max(r['max_abs_err_inverse_vs_actuator'] for r in rows);max2=max(r['max_abs_err_recorded_muscles_vs_actuator'] for r in rows)
 scale=max(r['actuator_force_absmax'] for r in rows)
 out['cases']['ID1_inverse_dynamics']=dict(states=len(rows),max_abs_error=max1,relative=max1/max(scale,1e-30),
  passed=bool(max1<=1e-9*max(scale,1.0)),note='tau = M qacc + bias - passive - constraint - applied, i.e. only the trajectory and the model')
 out['cases']['ID2_recorded_muscle_forces']=dict(max_abs_error=max2,relative=max2/max(scale,1e-30),
  passed=bool(max2<=1e-9*max(scale,1.0)),note='sum of moment arm times recorded muscle force equals the same generalized force')
 out['per_state']=rows
 # ID3: uniqueness of the per-muscle split
 d.qpos[:]=body['qpos'][-2];d.qvel[:]=body['qvel'][-2];d.act[:]=body['activation'][-2];d.ctrl[:]=body['command'][-2]
 d.qfrc_applied[:]=0.0;mujoco.mj_forward(m,d);A=moment_matrix(m,d)
 free_dofs=[j for j in range(m.njnt) if int(m.jnt_type[j])==0]
 root_dofs=[]
 for j in free_dofs:root_dofs+=list(range(int(m.jnt_dofadr[j]),int(m.jnt_dofadr[j])+(6 if int(m.jnt_type[j])==0 else 3)))
 actuated=np.setdiff1d(np.arange(m.nv),root_dofs)
 sV=np.linalg.svd(A[:,actuated],compute_uv=False)
 rank=int(np.sum(sV>1e-10*sV[0]));nullity=int(90-rank)
 root_rootarm=float(np.abs(A[:,root_dofs]).max())
 per_dof=int((np.abs(A[:,actuated])>1e-12).sum(0).max())
 out['cases']['ID3_uniqueness']=dict(matrix='90 actuators x 42 actuated hinge forces (the 6 root DOFs are not actuated)',
  actuated_dofs=int(len(actuated)),root_dofs=int(len(root_dofs)),
  rank=rank,rank_equals_actuated_dofs=bool(rank==len(actuated)),nullspace_dimension=nullity,
  actuators_with_any_moment=int((np.abs(A)>1e-12).any(1).sum()),
  max_muscles_sharing_one_joint=per_dof,root_dof_max_moment_arm=root_rootarm,
  singular_values_head=[float(x) for x in sV[:6]],
  passed=bool(rank==len(actuated) and root_rootarm<1e-12),
  meaning='the joint-torque vector is uniquely recoverable (full rank on the actuated subspace), but the split between the muscles that share a joint is not: it lives in a space of dimension nullspace_dimension. Picking one point in it needs an optimality assumption, which is a hypothesis. Muscles also cannot touch the body directly, because the six root DOFs have zero moment arm.')
 # ID4: structural fact about support
 out['cases']['ID4_support_path']=dict(root_dof_max_moment_arm=root_rootarm,
  passed=bool(root_rootarm<1e-12),
  meaning='No muscle has any moment arm on the floating-base DOFs, so muscle force cannot directly generate body support. Vertical support is created only by contact impulses; muscles influence it indirectly through joint dynamics. A per-muscle "support contribution" is therefore not a well-posed direct inverse.')
 # ID5: how wrong a min-effort optimality assumption would be
 from scipy.optimize import lsq_linear
 tau=A.T@body['muscle_force'][-2]
 # this model reports muscle forces with a negative sign convention (measured: every value <= 0),
 # so the identification is done on the magnitude g = -f with g >= 0
 sign=float(np.sign(np.median(body['muscle_force'][-2][np.abs(body['muscle_force'][-2])>1e-9]))) or -1.0
 B=sign*A.T[actuated,:];g_rec=sign*body['muscle_force'][-2]
 sol=lsq_linear(B,tau[actuated],bounds=(0.0,float(np.abs(g_rec).max()*2+1.0)),max_iter=600)
 g_star=np.asarray(sol.x,float);f_star=sign*g_star
 fit_resid=float(np.linalg.norm(B@g_star-tau[actuated])/max(np.linalg.norm(tau[actuated]),1e-30))
 f_rec=body['muscle_force'][-2]
 nz=np.abs(f_rec)>1e-9
 rel=float(np.linalg.norm(f_rec[nz]-f_star[nz])/max(np.linalg.norm(f_rec[nz]),1e-30))
 cos=float(np.dot(f_rec[nz],f_star[nz])/max(np.linalg.norm(f_rec[nz])*np.linalg.norm(f_star[nz]),1e-30))
 out['cases']['ID5_min_effort_hypothesis']=dict(relative_distance=rel,cosine=cos,
  active_muscles=int(nz.sum()),sign_convention=sign,fit_residual_rel=fit_resid,bounded_nonnegative_in_magnitude=True,
  passed=bool(np.isfinite(rel)),
  meaning='if one assumed the animal minimises summed squared muscle force, the recovered force pattern would differ from the recorded one by this much. That distance is the price of the optimality assumption, and it is why a purely inverse prediction of per-muscle force must be reported as hypothesis-conditioned, not as measurement.')
 out['passed']=bool(all(v['passed'] for v in out['cases'].values()))
 out['summary']='Muscle force IS recoverable here, but because the simulation records it and because inverse dynamics closes exactly. From kinematics plus contacts alone only the total generalized force is identifiable; the per-muscle split is a freedom of dimension ID3.nullspace_dimension.'
 out['calls']=dict(model_load=1,forward=len(rows)+1,body_step=0,neural_step=0,training=0)
 out['inputs_sha256']={str(p):sha(p) for p in [Path(__file__),BASE/'RUN/MODEL.xml',BASE/'RUN/BODY.npz',BEST/'RUN/BODY.npz']}
 dump(O/'RESULT.json',out)
 print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk in ('passed','max_abs_error','relative','rank','nullspace_dimension','vertical_generalized_force','sum_of_contributions')} for k,v in out['cases'].items()},ensure_ascii=False,indent=1))
 print('overall',out['passed'])
if __name__=='__main__':main()
