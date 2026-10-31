"""Bounded offline null-space stance search and native-force LP, never runtime control."""
from pathlib import Path
import os,time,json,hashlib,traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,mujoco
from scipy.optimize import least_squares,linprog
from flygym.compose.fly.musculoskeletal import _load_mjcf
P=Path(__file__).parent;S=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z');O=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-compatible-stance/20261028T079000Z');GFILE=Path('/home/sensen/Desktop/engine修复验收/s5-settled-muscle-support/20261028T076000Z/POOL_MATRIX.npz');LEGS=['LF','LM','LH','RF','RM','RH']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);start=time.time();counts=dict(model_load=1,forward=0,body_steps=0,neural_steps=0,preflight_model_load=1);sources=[Path(__file__),S/'MODEL.xml',S/'HANDOVER_BEFORE.npz',GFILE];dump(O/'PREREGISTRATION.json',dict(source_hashes={str(p):sha(p) for p in sources},scope='Offline bounded18 altered stance seeds plus baseline; exact native model/root pose, six Tarsus5 body origins preserved by42hinge IK. All LPs include active native joint limits and detected foot contacts. No integration or neural controller.',selection='Prefer6foot contacts, no body contact, fore/mid/hind skeletal chains stay on own body side, maximum foot-origin error<.005, native bounds; minimum outer-square LP force residual, tie lower pose deviation. If no6-foot candidate, report no admissible transfer.',seeds='6 systematic Coxa_pitch/Trochanter_pitch/Coxa_yaw offsets +/-.4, 4 mirrored roll/yaw offsets +/-.35,8 deterministic RNG10 hip-only offsets std.35 clipped.7; feet retargeted byIK; baseline exact unchanged.',LP='66 tied activations [0,1], unresolved16 floor.0001, contact normals>=0, outer square friction. All48force equilibrium L-infinity objective. Feasible outer LP must also pass inner diamond or actual circular forces check.',physiology='NOT_ESTABLISHED',preflight='one native model compiled only to inspect42jointnames, no forward/step'))
 m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m);q0=np.load(S/'HANDOVER_BEFORE.npz')['qpos'].copy();G=np.load(GFILE)['G'];w=float(m.body_mass.sum()*abs(m.opt.gravity[2]));js=np.array([j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]);qa=m.jnt_qposadr[js];va=m.jnt_dofadr[js];lo=m.jnt_range[js,0];hi=m.jnt_range[js,1];names=[m.joint(j).name for j in js];bids=[m.body(l+'Tarsus5').id for l in LEGS];root=m.body('Thorax').id
 def forward(q,x=None):
  d.qpos[:]=q;d.qvel[:]=0;d.act[:]=.0001 if x is None else .0001+.9999*(G@x);d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);counts['forward']+=1
 forward(q0);target=d.xpos[bids].copy();seeds=[('baseline',q0[qa].copy())]
 for key in ['Coxa_pitch','Trochanter_pitch','Coxa_yaw']:
  mask=np.array([n.endswith(key) for n in names])
  for sign in [-1,1]:seeds.append((key+str(sign),q0[qa]+mask*sign*.4))
 for key in ['Coxa_roll','Trochanter_yaw']:
  mask=np.array([((-1 if '_R' in n else 1) if n.endswith(key) else 0) for n in names])
  for sign in [-1,1]:seeds.append((key+'_mirror'+str(sign),q0[qa]+mask*sign*.35))
 rng=np.random.default_rng(10);hip=np.array(['Tibia' not in n for n in names])
 for k in range(8):seeds.append(('random'+str(k),q0[qa]+hip*np.clip(rng.normal(0,.35,42),-.7,.7)))
 rows=[];best=None;error=None
 def cone(q,inner=False):
  forward(q);base=d.qfrc_actuator.copy();bias=d.qfrc_bias.copy();passive=d.qfrc_passive.copy();contacts=[];blocks=[];nf=0;legs=set()
  E=d.efc_J.reshape(d.nefc,m.nv).copy() if d.efc_J.size==d.nefc*m.nv else np.zeros((d.nefc,m.nv))
  if d.efc_J.size!=d.nefc*m.nv:
   for row in range(d.nefc):
    adr=int(d.efc_J_rowadr[row]);nnz=int(d.efc_J_rownnz[row]);E[row,d.efc_J_colind[adr:adr+nnz]]=d.efc_J[adr:adr+nnz]
  lim=np.flatnonzero(np.asarray(d.efc_type)==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT));L=E[lim].T
  for k in range(d.ncon):
   co=d.contact[k];a,b=int(co.geom1),int(co.geom2);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
   if not(pa or pb):continue
   g=b if pa else a;name=m.geom(g).name
   if not(any(name.startswith(l) for l in LEGS) and 'Tars' in name):nf+=1;continue
   legs.add(name[:2]);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,co.pos,int(m.geom_bodyid[g]));blocks.append(jp.T);contacts.append(dict(geom=name,mu=float(co.friction[0]),pos=co.pos.copy()))
  # Skeletal body-frame side check on all available distal/major leg segment origins.
  side=[]
  for leg in LEGS:
   coords=[]
   for bid in range(m.nbody):
    name=m.body(bid).name
    if name.startswith(leg) and any(s in name for s in ['Coxa','Trochanter','Femur','Tibia','Tarsus']):coords.append(float((d.xmat[root].reshape(3,3).T@(d.xpos[bid]-d.xpos[root]))[1])*(1 if leg[0]=='L' else -1))
   side.append(min(coords) if coords else -1.)
  if not blocks:return None,dict(contacts=0,foot_legs_contacting=[],nonfoot=nf,side_min=side)
  A=np.zeros((48,66))
  for j in range(66):
   x=np.zeros(66);x[j]=1;forward(q,x);A[:,j]=d.qfrc_actuator-base
  J=np.column_stack(blocks);B=np.column_stack([A,J,L])/w;y=(bias-passive-base)/w;n=B.shape[1];nc=len(contacts);ub=[np.c_[B,-np.ones(48)],np.c_[-B,-np.ones(48)]];rhs=[y,-y]
  for j,c in enumerate(contacts):
   mat=np.zeros((4,n+1));dirs=[(1,1),(1,-1),(-1,1),(-1,-1)] if inner else [(1,0),(-1,0),(0,1),(0,-1)]
   for k,(sx,sy) in enumerate(dirs):mat[k,66+j*3:66+j*3+3]=[sx,sy,-c['mu']]
   ub.append(mat);rhs.append(np.zeros(4))
  obj=np.zeros(n+1);obj[-1]=1;bounds=[(0,1)]*66+[(None,None),(None,None),(0,None)]*nc+[(0,None)]*len(lim)+[(0,None)];sol=linprog(obj,A_ub=np.vstack(ub),b_ub=np.concatenate(rhs),bounds=bounds,method='highs');assert sol.success,sol.message;f=sol.x[66:66+nc*3].reshape(nc,3);ratio=max((np.linalg.norm(ff[:2])/(c['mu']*ff[2]) if ff[2]>1e-9 else 0.) for ff,c in zip(f,contacts))
  detail=dict(activation_matrix=G,active_force_matrix=A,contact_force_matrix=J,joint_limit_force_matrix=L,target_generalized_force=bias-passive-base,pool_activation=sol.x[:66],contact_forces=f,contacts=contacts,linprog_solution=sol.x,bodyweight_native=w,force_residual_over_weight=B@sol.x[:-1]-y,friction_inner=inner)
  return detail,dict(lp_t=float(sol.fun),contacts=nc,foot_legs_contacting=sorted(legs),nonfoot=nf,side_min=side,joint_limits=len(lim),circular_friction_ratio=float(ratio),normal_ground_over_weight=float(f[:,2].sum()/w))
 try:
  for k,(name,seed) in enumerate(seeds):
   if k==0:q=q0.copy();ik_success=True
   else:
    seed=np.clip(seed,lo+1e-8,hi-1e-8);cache={}
    def up(x):
     if 'x' not in cache or not np.array_equal(x,cache['x']):q=q0.copy();q[qa]=x;forward(q);cache['x']=x.copy()
    def fun(x):up(x);return np.r_[(10*(d.xpos[bids]-target)).ravel(),.02*(x-seed)]
    def jac(x):
     up(x);arr=[]
     for bid in bids:
      jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,d.xpos[bid],bid);arr.append(jp[:,va]*10)
     return np.vstack(arr+[.02*np.eye(42)])
    fit=least_squares(fun,seed,jac=jac,bounds=(lo+1e-8,hi-1e-8),max_nfev=65,ftol=1e-8,gtol=1e-8,xtol=1e-8);q=q0.copy();q[qa]=fit.x;ik_success=bool(fit.success)
   forward(q);err=float(np.max(np.linalg.norm(d.xpos[bids]-target,axis=1)));detail,row=cone(q);row.update(index=k,name=name,ik_success=ik_success,foot_error=err,hinge_deviation_norm=float(np.linalg.norm(q[qa]-q0[qa])),qpos=q.copy())
   row['eligible']=bool(detail is not None and len(row['foot_legs_contacting'])==6 and row['nonfoot']==0 and min(row['side_min'])>0 and err<.005 and (k==0 or np.all(q[qa]>=lo-1e-7) and np.all(q[qa]<=hi+1e-7)))
   if row['eligible'] and (best is None or (row['lp_t'],row['hinge_deviation_norm'])<(best[0]['lp_t'],best[0]['hinge_deviation_norm'])):
    best=(row.copy(),detail);dump(O/'BEST_DETAIL.json',detail);dump(O/'BEST.json',row);np.savez_compressed(O/'CANDIDATE.npz',qpos=q,qpos_hinges=q[qa],qpos_addresses=qa,dof_addresses=va,pool_activation=detail['pool_activation'],actuator_control=G@detail['pool_activation'],G=G)
   rows.append(row);dump(O/'CANDIDATES.json',rows);print(json.dumps({kk:vv for kk,vv in row.items() if kk!='qpos'}),flush=True)
  assert best is not None,'No candidate retained all six foot contacts and skeletal side separation'
  detail,innerrow=cone(best[0]['qpos'],True);dump(O/'BEST_INNER_DETAIL.json',detail);dump(O/'BEST_INNER.json',innerrow)
 except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());dump(O/'FAILURE.json',error);raise
 finally:
  dump(O/'EXECUTION.json',dict(counts=counts,exit_code=0 if error is None else 1,error=error,runtime_s=time.time()-start,source_unchanged=all(sha(p)==h for p,h in json.loads((O/'PREREGISTRATION.json').read_text())['source_hashes'].items())));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
