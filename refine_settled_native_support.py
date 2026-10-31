"""Explicit absolute-step Jacobian refinement; offline only, no runtime commands."""
from pathlib import Path
import os,json,hashlib,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,mujoco
from scipy.optimize import least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf
from diagnose_settled_native_support import S,O,dump,sha

def main():
 start=time.time();original=json.loads((O/'DETAIL.json').read_text());G=np.load(O/'POOL_MATRIX.npz')['G'];q0=np.load(S/'HANDOVER_BEFORE.npz')['qpos'].copy();counts=dict(model_load=1,forward=0,body_steps=0,neural_steps=0)
 pre=O/'REFINEMENT_PREREGISTRATION.json';assert not pre.exists();dump(pre,dict(source_hashes={str(p):sha(p) for p in [Path(__file__),O/'DETAIL.json',O/'POOL_MATRIX.npz',S/'MODEL.xml',S/'HANDOVER_BEFORE.npz']},reason='Relative finite difference steps become tiny near active bounds; xtol termination is not stationary convergence. Use absolute activation step1e-4/height1e-5 and max500nfev.',solver=dict(ftol=1e-10,xtol=1e-10,gtol=1e-8,max_nfev=500),scope='Numerical refinement of offline fixed pose/height fit. No native model parameter changes, no body/neural integration.'))
 m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m);w=float(m.body_mass.sum()*abs(m.opt.gravity[2]));M=np.zeros((m.nv,m.nv));allrows=[];details={}
 try:
  for name,height in [('tied66_fixed',False),('tied66_height',True)]:
   old=original[name]['best'];x0=np.array(old['parameters']);n=len(x0);lo=np.r_[np.zeros(66),-.03] if height else np.zeros(66);hi=np.r_[np.ones(66),.03] if height else np.ones(66);steps=np.r_[np.full(66,1e-4),1e-5] if height else np.full(66,1e-4)
   def fun(x,retain=False):
    q=q0.copy();z=float(x[-1]) if height else 0.;q[2]+=z;d.qpos[:]=q;d.qvel[:]=0;d.act[:]=.0001+.9999*(G@x[:66]);d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);counts['forward']+=1;mujoco.mj_fullM(m,M,d.qM);force=M@d.qacc;res=force/w
    if not retain:return np.r_[res,z*.001] if height else res
    return dict(parameters=x.copy(),root_height_variable=height,root_height_offset=z,qpos=q,qacc=d.qacc.copy(),mass_matrix=M.copy(),actuator_activation=d.act.copy(),generalized_force_residual=force,qfrc_actuator=d.qfrc_actuator.copy(),qfrc_constraint=d.qfrc_constraint.copy(),qfrc_passive=d.qfrc_passive.copy(),qfrc_bias=d.qfrc_bias.copy(),force_residual_norm_over_weight=float(np.linalg.norm(res)))
   def jac(x):
    y=fun(x);J=np.zeros((len(y),n))
    for j in range(n):
     xx=x.copy();h=steps[j] if x[j]+steps[j]<=hi[j] else -steps[j];xx[j]+=h;J[:,j]=(fun(xx)-y)/h
    return J
   solved=least_squares(fun,x0,jac=jac,bounds=(lo,hi),max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-8);fit=fun(solved.x,True);row=dict(name=name,status=int(solved.status),success=bool(solved.success),message=str(solved.message),nfev=int(solved.nfev),njev=int(solved.njev),optimality=float(solved.optimality),force_residual_norm_over_weight=fit['force_residual_norm_over_weight'],root_height_offset=fit['root_height_offset'],stationary_gate=bool(solved.optimality<1e-6));allrows.append(row);details[name]=fit;dump(O/'REFINEMENT_RESULTS.json',allrows);dump(O/'REFINEMENT_DETAIL.json',details);print(json.dumps(row),flush=True)
 finally:
  dump(O/'REFINEMENT_EXECUTION.json',dict(counts=counts,runtime_s=time.time()-start,bodyweight_native=w,scope='Offline only; optimizer success/xtol does not imply force balance or first-order stationarity.'));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
