"""Proper load capacity: how many body weights can this body hold, as a linear program?

This replaces the earlier full-activation probe. That probe set every actuator to full activation at
one pose and read the engine's contact forces, reporting 110.5 percent of body weight. That was not a
capacity measurement: at that state the model accelerates at about 7.4 g, so the contact force was the
reaction to a co-contraction, not a load the body can hold.

  maximise    L, an extra downward force at the body centre of mass
  subject to  static equilibrium on all DOFs via the engine's own point Jacobians (mj_jac),
              muscle activation in 0..1 with each actuator's force in that pose,
              contact force with a nonnegative vertical component and a box friction cone.

A factor below 1 means the model cannot even stand carrying nothing. modes: run
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

def pose_capacity(m,d,qpos,label):
 import mujoco
 from scipy.optimize import linprog
 nv,nu=m.nv,m.nu
 d.qpos[:]=qpos;d.qvel[:]=0.;d.act[:]=1.;d.ctrl[:]=1.;mujoco.mj_forward(m,d)
 bias=np.asarray(d.qfrc_bias).ravel()
 A=np.zeros((nv,nu));actuated=m.na>=nu and m.na>0
 for i in range(nu):
  if actuated:d.act[:]=0.;d.act[i]=1.;d.ctrl[:]=0.;d.ctrl[i]=1.
  else:d.ctrl[:]=0.;d.ctrl[i]=1.
  mujoco.mj_forward(m,d)
  A[:,i]=np.asarray(d.qfrc_actuator).ravel()
 d.act[:]=1.;d.ctrl[:]=1.;mujoco.mj_forward(m,d)
 planes=m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
 feet=np.zeros(m.ngeom,bool)
 for g in range(m.ngeom):
  b=m.body(m.geom_bodyid[g]).name or ''
  if any(b.startswith(L) and 'Tars' in b for L in ('LF','LM','LH','RF','RM','RH')):feet[g]=True
 pts=[]
 for ci in range(d.ncon):
  c=d.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
  if not (planes[g1] or planes[g2]):continue
  if not (feet[g1] or feet[g2]):continue
  gb=g1 if not planes[g1] else g2
  jp=np.zeros((3,nv));jr=np.zeros((3,nv))
  mujoco.mj_jac(m,d,jp,jr,np.array(c.pos),int(m.geom_bodyid[gb]))
  pts.append((jp,float(max(m.geom_friction[gb,0],0.3))))
 K=len(pts)
 if K==0:return dict(label=label,feasible=False,failed='no foot contact at this pose')
 tb=int(m.body('Thorax').id)
 jp=np.zeros((3,nv));jr=np.zeros((3,nv))
 mujoco.mj_jac(m,d,jp,jr,np.array(d.subtree_com[tb]),tb)
 Aeq=np.zeros((nv,nu+3*K+1));Aeq[:,:nu]=A
 # the engine's convention, verified against qfrc_constraint: the generalised contact force is
 # MINUS J^T f for a world-frame force f at the contact point
 for i,(jc,mu) in enumerate(pts):Aeq[:,nu+3*i:nu+3*i+3]=-jc.T
 Aeq[:,-1]=-jp[2,:]
 c=np.zeros(nu+3*K+1);c[-1]=-1.0
 bounds=[(0.,1.)]*nu
 for i in range(K):bounds+=[(-1e6,1e6),(-1e6,1e6),(0.,1e6)]
 bounds+=[(0.,1e9)]
 Aub=[];bub=[]
 for i,(jc,mu) in enumerate(pts):
  b0=nu+3*i
  for t in (0,1):
   r=np.zeros(nu+3*K+1);r[b0+t]=1;r[b0+2]=-mu;Aub.append(r);bub.append(0.)
   r=np.zeros(nu+3*K+1);r[b0+t]=-1;r[b0+2]=-mu;Aub.append(r);bub.append(0.)
 res=linprog(c,A_ub=np.array(Aub),b_ub=np.array(bub),A_eq=Aeq,b_eq=bias,bounds=bounds,method='highs')
 W=float(np.sum(m.body_mass)*abs(m.opt.gravity[2]))
 if not res.success:return dict(label=label,n_contacts=K,feasible=False,message=res.message)
 x=res.x;a=x[:nu];L=float(x[-1]);fz=float(sum(x[nu+3*i+2] for i in range(K)))
 return dict(label=label,n_contacts=K,feasible=True,load_factor=float(L/W),extra_load=L,body_weight=W,
  total_vertical=fz,vertical_equals_weight_plus_load=bool(abs(fz-(W+L))<1e-6*max(W,1.0)),
  muscles_at_ceiling=int((a>0.999).sum()),muscles_active=int((a>0.01).sum()),total_activation=float(a.sum()))

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
 z=np.load(RUN/'BODY.npz',allow_pickle=False)
 qpos=z['qpos'];t=np.asarray(z['time'],float)
 idx=[int(np.argmin(abs(t-x))) for x in np.arange(1.10,1.302,0.02)]
 out=[pose_capacity(m,d,qpos[i],'assisted t=%.3f'%t[i]) for i in idx]
 ok=[o for o in out if o.get('feasible')]
 best=max(ok,key=lambda o:o['load_factor']) if ok else None
 dump(O/'LOAD_CAPACITY.json',dict(
  question='how many body weights can the modelled body hold statically, best posture over the assisted stride',
  poses=out,feasible_poses=len(ok),total_poses=len(out),best=best,
  retraction='the earlier 110.5 percent figure is withdrawn: it came from a state accelerating at about 7.4 g with every actuator at full activation, which measures co-contraction and not capacity',
  method='mj_jac point Jacobians, static equilibrium on all DOFs, box friction cone, maximise an added load at the body centre of mass'))
 if best:print('可行 %d/%d | 最佳 %.2f 倍体重（%s）激活和 %.2f 饱和 %d'%(len(ok),len(out),best['load_factor'],best['label'],best['total_activation'],best['muscles_at_ceiling']))
 else:print('无可行姿态')
 for o in out:print(' ',o.get('label'),('可行 %.2f'%o['load_factor']) if o.get('feasible') else ('不可行 '+o.get('message','')[:45]))
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'run':run}[sys.argv[1]]()
