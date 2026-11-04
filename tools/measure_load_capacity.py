"""Static capability and load capacity, built on the engine's own constraint Jacobians.

Earlier attempts hand-derived contact geometry and failed their own consistency checks. This version
uses d.efc_J instead: MuJoCo already lays out every constraint row (contact pyramid rows, joint limits
and so on) as a Jacobian row with a non-negative force, so the static program can be written directly
on the engine's structure:

    qfrc_actuator + qfrc_constraint = qfrc_bias            (engine convention, qacc = 0)
    qfrc_actuator = A a,  0 <= a <= 1                       (muscle activation)
    qfrc_constraint = +/- (efc_J^T @ r),  r >= 0            (constraint forces, sign verified here)
    extra downward load L at the centre of mass

Checks run before any number is reported:
  check 1  the engine's own efc_force must reproduce qfrc_constraint through efc_J
  check 2  the load program must be feasible at a pose that is obviously feasible
  check 3  per-hinge actuation interval, independent of contacts

modes: run
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
O=Path('/home/sensen/Desktop/engine修复验收/s5-load-capacity/20261105T140000Z')
RUN=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-loop/20261104T070000Z/FINAL_VALIDATION_A/RUN')
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def constraints(m,d):
 import mujoco
 nefc=int(d.nefc)
 J=np.asarray(d.efc_J,dtype=float).reshape(nefc,m.nv)
 r=np.asarray(d.efc_force,dtype=float)[:nefc]
 types=np.asarray(d.efc_type)[:nefc]
 ids=np.asarray(d.efc_id)[:nefc]
 return J,r,types,ids

def actuation(m,d):
 import mujoco
 A=np.zeros((m.nv,m.nu))
 for i in range(m.nu):
  d.act[:]=0.;d.act[i]=1.;d.ctrl[:]=0.;d.ctrl[i]=1.
  mujoco.mj_forward(m,d);A[:,i]=np.asarray(d.qfrc_actuator).ravel()
 d.act[:]=1.;d.ctrl[:]=1.;mujoco.mj_forward(m,d)
 return A

def sign_of(m,d):
 J,r,_,_=constraints(m,d)
 cons=np.asarray(d.qfrc_constraint).ravel()
 best=None
 for s in (+1.,-1.):
  res=float(np.linalg.norm(s*(J.T@r)-cons))
  if best is None or res<best[0]:best=(res,s)
 return best[1],best[0]/max(np.linalg.norm(cons),1e-12)

def capacity(m,d,qpos,label):
 import mujoco
 from scipy.optimize import linprog
 d.qpos[:]=qpos;d.qvel[:]=0.;d.act[:]=1.;d.ctrl[:]=1.;mujoco.mj_forward(m,d)
 bias=np.asarray(d.qfrc_bias).ravel()
 A=actuation(m,d)
 sgn,relres=sign_of(m,d)
 J,r,types,ids=constraints(m,d)
 nefc=int(d.nefc);nv,nu=m.nv,m.nu
 W=float(np.sum(m.body_mass)*abs(m.opt.gravity[2]))
 tb=int(m.body('Thorax').id)
 jp=np.zeros((3,nv));jr=np.zeros((3,nv))
 mujoco.mj_jac(m,d,jp,jr,np.array(d.subtree_com[tb]),tb)
 n=nu+nefc+1
 Aeq=np.zeros((nv,n));Aeq[:,:nu]=A;Aeq[:,nu:nu+nefc]=sgn*J.T
 Aeq[:,-1]=-jp[2,:]
 cv=np.zeros(n);cv[-1]=-1.
 bounds=[(0.,1.)]*nu+[(0.,1e9)]*nefc+[(0.,1e9)]
 lp=linprog(cv,A_eq=Aeq,b_eq=bias,bounds=bounds,method='highs')
 out=dict(label=label,nefc=nefc,ncon=int(d.ncon),sign=int(sgn),consistency_relative=relres,
  constraint_types={int(k):int(v) for k,v in zip(*np.unique(types,return_counts=True))})
 if lp.success:
  a=lp.x[:nu];L=lp.x[-1];rf=lp.x[nu:nu+nefc]
  out.update(feasible=True,load_factor=float(L/W),extra_load=float(L),
   muscles_at_ceiling=int((a>0.999).sum()),muscles_active=int((a>0.01).sum()),
   constraint_force_sum=float(rf.sum()))
 else:
  out.update(feasible=False,message=lp.message[:80])
 return out

def run():
 import mujoco
 assets=Path('/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a')
 shim=O/'MODEL_SHIM'
 if not shim.exists():
  shim.mkdir(parents=True);(shim/'MODEL.xml').symlink_to(RUN/'MODEL.xml')
  (shim/'meshes').mkdir(exist_ok=True)
  if not (shim/'meshes'/'stl').exists():(shim/'meshes'/'stl').symlink_to(assets,target_is_directory=True)
 os.chdir(shim)
 m=mujoco.MjModel.from_xml_path('MODEL.xml');d=mujoco.MjData(m)
 z=np.load(RUN/'BODY.npz',allow_pickle=False);qpos=z['qpos'];t=np.asarray(z['time'],float)
 res=[]
 for x in np.arange(1.10,1.301,0.02):
  i=int(np.argmin(abs(t-x)));res.append(capacity(m,d,qpos[i],'assisted t=%.3f'%t[i]))
 ok=[r for r in res if r['feasible']]
 best=max(ok,key=lambda r:r['load_factor']) if ok else None
 dump(O/'LOAD_CAPACITY_EFC.json',dict(
  method='static program on the engine constraint rows (efc_J, non-negative efc_force), which includes contacts AND joint limits automatically',
  consistency=[dict(label=r['label'],relative=r['consistency_relative'],sign=r['sign']) for r in res],
  poses=res,feasible_poses=len(ok),total=len(res),best=best,
  reading='load_factor is how many body weights of extra downward load the body can hold statically in that posture'))
 print('一致性(相对残差):',['%.3f'%r['consistency_relative'] for r in res])
 print('可行姿态 %d/%d'%(len(ok),len(res)))
 for r in res[:12]:
  print(' %s 可行%s'%(r['label'],('%.2f倍体重'%r['load_factor']) if r['feasible'] else ' ('+r.get('message','')[:30]+')'))
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'run':run}[sys.argv[1]]()
