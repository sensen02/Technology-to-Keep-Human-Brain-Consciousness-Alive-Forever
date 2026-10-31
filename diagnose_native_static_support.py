"""Offline bounded native-muscle static feasibility. Never integrate or drive a live fly."""
from pathlib import Path
import os,json,hashlib,time,traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,mujoco
from scipy.optimize import least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf
S=Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T058000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-native-muscle-support/20261028T075000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);start=time.time();sources=[Path(__file__),S/'MODEL.xml',S/'BODY.npz',S/'MAPPING.json'];hashes={str(p):sha(p) for p in sources}
 dump(O/'PREREGISTRATION.json',dict(scope='Offline fixed saved qpos, qvel0, native muscle/contact parameters. Activation optimization only; never body integration, neural command or neural success.',source_hashes=hashes,poses_s=[0.,.915,1.],activation_floor=.0001,pool_count=66,unresolved_control_zero_native_floor=True,objective='All48 generalized force residual M(q)@qacc / bodyweight, at free-root forward/contact solution. Same native length unit used for moment coordinates; no biological normalization claim.',effort='Each pose at most8 least_squares evaluations (Jacobian calls included separately); global at most1700forward. Zero warmstart every evaluation.',support_gate='Force residual L2/bodyweight<.01 AND footload/bodyweight>.9 AND nonfoot force/bodyweight<.001. Local fixed-pose search only, not global impossibility or future stability proof.',physiology='NOT_ESTABLISHED'))
 with np.load(S/'BODY.npz') as z:b={k:z[k] for k in z.files}
 mapping=json.loads((S/'MAPPING.json').read_text());groups={}
 for row in mapping['rows']:
  if row['local_indices']:groups.setdefault(tuple(row['pool_root_ids']),[]).append(row['actuator_index'])
 assert len(groups)==66;G=np.zeros((90,66))
 for j,ids in enumerate(groups.values()):G[ids,j]=1
 assert np.sum(G.sum(1)==0)==16
 m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m);counts=dict(model_load=1,forward=0,body_steps=0,neural_steps=0);weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]));results=[];details=[]
 for target in [0.,.915,1.]:
  i=int(np.argmin(abs(b['time']-target)));q=b['qpos'][i].copy();fullM=np.zeros((m.nv,m.nv));evaluations=0;best=None
  def evaluate(x,retain=False):
   nonlocal evaluations,best
   if counts['forward']>=1700:raise RuntimeError('Declared total forward budget reached')
   d.qpos[:]=q;d.qvel[:]=0;d.act[:]=.0001+.9999*(G@x);d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);counts['forward']+=1;evaluations+=1
   mujoco.mj_fullM(m,fullM,d.qM);force=fullM@d.qacc;residual=force/weight
   if best is None or np.linalg.norm(residual)<best[0]:best=(float(np.linalg.norm(residual)),x.copy())
   if not retain:return residual.copy()
   foot=0.;nonfoot=0.
   for k in range(d.ncon):
    co=d.contact[k];g1,g2=int(co.geom1),int(co.geom2);p1=m.geom_type[g1]==mujoco.mjtGeom.mjGEOM_PLANE;p2=m.geom_type[g2]==mujoco.mjtGeom.mjGEOM_PLANE
    if not(p1 or p2):continue
    g=g2 if p1 else g1;f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);fz=((co.frame.reshape(3,3).T@f[:3])*(1 if p1 else -1))[2];name=m.geom(g).name
    if any(name.startswith(leg) for leg in ['LF','LM','LH','RF','RM','RH']) and 'Tars' in name:foot+=fz
    else:nonfoot+=fz
   return dict(pool_activation=x.copy(),actuator_activation=d.act.copy(),qpos=q.copy(),qacc=d.qacc.copy(),mass_matrix=fullM.copy(),generalized_force_residual=force.copy(),qfrc_actuator=d.qfrc_actuator.copy(),qfrc_constraint=d.qfrc_constraint.copy(),qfrc_passive=d.qfrc_passive.copy(),qfrc_bias=d.qfrc_bias.copy(),force_residual_norm_over_weight=float(np.linalg.norm(residual)),root_force_residual_norm_over_weight=float(np.linalg.norm(residual[:6])),foot_load_ratio=float(foot/weight),nonfoot_load_ratio=float(nonfoot/weight),warning_count=int(d.warning.number.sum()))
  base=evaluate(np.zeros(66),True)
  solved=least_squares(evaluate,np.full(66,.001),bounds=(np.zeros(66),np.ones(66)),max_nfev=8,ftol=1e-7,xtol=1e-7,gtol=1e-7,diff_step=1e-3)
  fitted=evaluate(best[1],True);accepted=bool(fitted['force_residual_norm_over_weight']<.01 and fitted['foot_load_ratio']>.9 and abs(fitted['nonfoot_load_ratio'])<.001)
  row=dict(saved_time_s=float(b['time'][i]),forward_evaluations=evaluations,optimizer_success=bool(solved.success),optimizer_status=int(solved.status),optimizer_message=str(solved.message),optimizer_nfev=int(solved.nfev),baseline_force_residual=base['force_residual_norm_over_weight'],best_force_residual=fitted['force_residual_norm_over_weight'],baseline_foot_load_ratio=base['foot_load_ratio'],best_foot_load_ratio=fitted['foot_load_ratio'],baseline_root_qacc=base['qacc'][:6],best_root_qacc=fitted['qacc'][:6],best_joint_qacc_norm=float(np.linalg.norm(fitted['qacc'][6:])),static_support_gate_passed=accepted)
  results.append(row);details.append(dict(saved_time_s=float(b['time'][i]),baseline=base,best=fitted));dump(O/'RESULTS.json',results);dump(O/'DETAIL.json',details);print(json.dumps(row,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
 dump(O/'EXECUTION.json',dict(counts=counts,exit_code=0,runtime_s=time.time()-start,source_unchanged=all(sha(Path(p))==h for p,h in hashes.items()),bodyweight_native=weight,scope='Local fixed-qpos static feasibility only; fitted activations never applied to a simulation trajectory.'))
 np.savez_compressed(O/'POOL_MATRIX.npz',G=G);(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
