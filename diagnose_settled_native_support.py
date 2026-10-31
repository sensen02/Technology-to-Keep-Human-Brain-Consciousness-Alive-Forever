"""Offline settled-pose native muscle equilibrium; no integration or neural commands."""
from pathlib import Path
import os,json,hashlib,time,traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,mujoco
from scipy.optimize import least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf
S=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-settled-muscle-support/20261028T076000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);start=time.time();sources=[Path(__file__),S/'MODEL.xml',S/'HANDOVER_BEFORE.npz',S/'MAPPING.json'];hashes={str(p):sha(p) for p in sources}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,scope='Offline qvel0 static native muscle/contact force balance at actual settled074 pose. No body integration or neuronal command.',runs=['66 tied activations at fixed settled pose','66 tied activations + root height offset[-.03,.03]','90 independent activations fixed settled pose diagnostic only'],activation_floor=.0001,independent90_scope='Relaxation includes unresolved muscles, NOT admissible neural runtime output',objective='M(q)@qacc/bodyweight, all48 free-root+joint force components; coupled rootheight adds tiny1e-3*height regularizer',solver=dict(max_nfev=100,diff_step=.0001,ftol=1e-8,xtol=1e-8,gtol=1e-8),gate='L2 force residual/bodyweight<.01, footload/bodyweight>.9, no nonfoot contact, warnings0. Local search only.',physiology='NOT_ESTABLISHED'))
 mapping=json.loads((S/'MAPPING.json').read_text());groups={}
 for r in mapping['rows']:
  if r['local_indices']:groups.setdefault(tuple(r['pool_root_ids']),[]).append(r['actuator_index'])
 G=np.zeros((90,66))
 for j,ids in enumerate(groups.values()):G[ids,j]=1
 assert len(groups)==66 and np.sum(G.sum(1)==0)==16
 np.savez_compressed(O/'POOL_MATRIX.npz',G=G)
 m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m);q0=np.load(S/'HANDOVER_BEFORE.npz')['qpos'].copy();w=float(m.body_mass.sum()*abs(m.opt.gravity[2]));counts=dict(model_load=1,forward=0,body_steps=0,neural_steps=0);details={};results=[];error=None
 try:
  seed=np.full(66,.001)
  for name,H,height in [('tied66_fixed',G,False),('tied66_height',G,True),('independent90_fixed',np.eye(90),False)]:
   n=H.shape[1];x0=seed.copy() if n==66 else np.clip(G@seed,1e-9,1-1e-9)
   if height:x0=np.r_[x0,0.]
   M=np.zeros((m.nv,m.nv));calls=0
   def evaluate(x,retain=False):
    nonlocal calls
    q=q0.copy();z=float(x[-1]) if height else 0.;q[2]+=z
    d.qpos[:]=q;d.qvel[:]=0;d.act[:]=.0001+.9999*(H@x[:n]);d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);counts['forward']+=1;calls+=1;mujoco.mj_fullM(m,M,d.qM);force=M@d.qacc;res=force/w
    if not retain:return np.r_[res,z*.001] if height else res.copy()
    foot=0.;nonfoot=0.;footgeom=set();noncount=0
    for k in range(d.ncon):
     co=d.contact[k];g1,g2=int(co.geom1),int(co.geom2);p1=m.geom_type[g1]==mujoco.mjtGeom.mjGEOM_PLANE;p2=m.geom_type[g2]==mujoco.mjtGeom.mjGEOM_PLANE
     if not(p1 or p2):continue
     g=g2 if p1 else g1;f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);fz=((co.frame.reshape(3,3).T@f[:3])*(1 if p1 else -1))[2];gn=m.geom(g).name
     if any(gn.startswith(l) for l in ['LF','LM','LH','RF','RM','RH']) and 'Tars' in gn:foot+=fz;footgeom.add(gn[:2])
     else:nonfoot+=fz;noncount+=1
    return dict(parameters=x.copy(),activation_matrix=H.copy(),root_height_variable=height,root_height_offset=z,qpos=q,qacc=d.qacc.copy(),mass_matrix=M.copy(),actuator_activation=d.act.copy(),generalized_force_residual=force.copy(),qfrc_actuator=d.qfrc_actuator.copy(),qfrc_constraint=d.qfrc_constraint.copy(),qfrc_passive=d.qfrc_passive.copy(),qfrc_bias=d.qfrc_bias.copy(),force_residual_norm_over_weight=float(np.linalg.norm(res)),foot_load_ratio=float(foot/w),nonfoot_load_ratio=float(nonfoot/w),nonfoot_contact_count=noncount,foot_legs_contacting=sorted(footgeom),warning_count=int(d.warning.number.sum()))
   baseline=evaluate(np.r_[np.zeros(n),0.] if height else np.zeros(n),True)
   lo=np.r_[np.zeros(n),-.03] if height else np.zeros(n);hi=np.r_[np.ones(n),.03] if height else np.ones(n)
   solved=least_squares(evaluate,x0,bounds=(lo,hi),max_nfev=100,diff_step=.0001,ftol=1e-8,xtol=1e-8,gtol=1e-8)
   fitted=evaluate(solved.x,True)
   if name=='tied66_fixed':seed=solved.x.copy()
   row=dict(name=name,optimizer_success=bool(solved.success),optimizer_status=int(solved.status),message=str(solved.message),nfev=int(solved.nfev),njev=int(solved.njev),optimality=float(solved.optimality),forward_evaluations=calls,baseline_force_residual=baseline['force_residual_norm_over_weight'],best_force_residual=fitted['force_residual_norm_over_weight'],foot_load_ratio=fitted['foot_load_ratio'],root_height_offset=fitted['root_height_offset'],root_qacc=fitted['qacc'][:6],joint_qacc_norm=float(np.linalg.norm(fitted['qacc'][6:])),static_support_gate_passed=bool(fitted['force_residual_norm_over_weight']<.01 and fitted['foot_load_ratio']>.9 and fitted['nonfoot_contact_count']==0 and fitted['warning_count']==0))
   results.append(row);details[name]=dict(baseline=baseline,best=fitted);dump(O/'RESULTS.json',results);dump(O/'DETAIL.json',details);print(json.dumps(row,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
 except BaseException as exc:
  error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());dump(O/'FAILURE.json',error);raise
 finally:
  dump(O/'EXECUTION.json',dict(counts=counts,exit_code=0 if error is None else 1,error=error,runtime_s=time.time()-start,bodyweight_native=w,source_unchanged=all(sha(Path(p))==h for p,h in hashes.items()),scope='All activations offline only, never a runtime controller or behavior certificate.'))
  (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
