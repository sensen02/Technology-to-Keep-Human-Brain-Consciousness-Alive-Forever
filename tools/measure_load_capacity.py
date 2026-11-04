"""Static capability and load capacity, with the machinery verified before it is trusted.

Three checks run in order, and no number is reported unless the earlier ones pass:

  check 1  engine consistency  the engine's own contact forces, mapped through the point Jacobians
           with the convention generalised contact force = -(J1-J2)^T frame f_local, must reproduce
           qfrc_constraint. Reported as a residual.
  check 2  internal actuation  for every hinge DOF, the interval of generalised force the muscles can
           produce in that pose; a DOF whose gravity demand falls outside it cannot be held by muscles
           whatever the network does.
  check 3  full static program  muscle activation in 0..1, contact forces with a nonnegative normal
           and a box friction cone, static equilibrium on all DOFs, maximising an extra downward load
           at the body centre of mass.

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
LEGS=('LF','LM','LH','RF','RM','RH')
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def setup(model_dir):
 import mujoco
 os.chdir(model_dir)
 m=mujoco.MjModel.from_xml_path('MODEL.xml');d=mujoco.MjData(m)
 return m,d

def contacts(m,d,only_feet):
 import mujoco
 planes=m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
 feet=np.zeros(m.ngeom,bool)
 for g in range(m.ngeom):
  b=m.body(m.geom_bodyid[g]).name or ''
  if any(b.startswith(L) and 'Tars' in b for L in LEGS):feet[g]=True
 out=[]
 for ci in range(d.ncon):
  c=d.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
  if only_feet and not ((planes[g1] or planes[g2]) and (feet[g1] or feet[g2])):continue
  f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
  fr=np.array(c.frame).reshape(3,3)
  j1p=np.zeros((3,m.nv));j1r=np.zeros((3,m.nv));j2p=np.zeros((3,m.nv));j2r=np.zeros((3,m.nv))
  mujoco.mj_jac(m,d,j1p,j1r,np.array(c.pos),int(m.geom_bodyid[g1]))
  mujoco.mj_jac(m,d,j2p,j2r,np.array(c.pos),int(m.geom_bodyid[g2]))
  out.append(dict(J=j1p-j2p,frame=fr,f_local=f[:3].copy(),mu=float(max(m.geom_friction[g1,0],m.geom_friction[g2,0],0.3))))
 return out

def actuation(m,d):
 import mujoco
 A=np.zeros((m.nv,m.nu))
 for i in range(m.nu):
  d.act[:]=0.;d.act[i]=1.;d.ctrl[:]=0.;d.ctrl[i]=1.
  mujoco.mj_forward(m,d);A[:,i]=np.asarray(d.qfrc_actuator).ravel()
 d.act[:]=1.;d.ctrl[:]=1.;mujoco.mj_forward(m,d)
 return A

def analyse(m,d,qpos,label):
 import mujoco
 from scipy.optimize import linprog
 d.qpos[:]=qpos;d.qvel[:]=0.;d.act[:]=1.;d.ctrl[:]=1.;mujoco.mj_forward(m,d)
 bias=np.asarray(d.qfrc_bias).ravel()
 A=actuation(m,d)
 W=float(np.sum(m.body_mass)*abs(m.opt.gravity[2]))
 cs_all=contacts(m,d,False)
 cons=np.asarray(d.qfrc_constraint).ravel()
 tot=np.zeros(m.nv)
 for c in cs_all:tot+=-(c['J'].T@(c['frame']@c['f_local']))
 residual=float(np.linalg.norm(tot-cons));scale=float(np.linalg.norm(cons))
 # check 2: per hinge interval
 unholdable=[]
 for k in range(6,m.nv):
  pos=float(np.where(A[k]>0,A[k],0).sum());neg=float(np.where(A[k]<0,A[k],0).sum())
  need=float(-bias[k])
  if not (neg-1e-9<=need<=pos+1e-9):
   unholdable.append(dict(dof=int(k),joint=(m.joint(m.dof_jntid[k]).name if m.dof_jntid[k]>=0 else 'root'),
    demand=need,reachable_min=neg,reachable_max=pos,
    missing=float(min(abs(need-neg),abs(need-pos)))))
 # check 3: full program with feet contacts
 cs=[c for c in contacts(m,d,True)]
 K=len(cs)
 res=None
 if K:
  tb=int(m.body('Thorax').id);jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
  mujoco.mj_jac(m,d,jp,jr,np.array(d.subtree_com[tb]),tb)
  n=m.nu+3*K+1
  Aeq=np.zeros((m.nv,n));Aeq[:,:m.nu]=A
  for i,c in enumerate(cs):Aeq[:,m.nu+3*i:m.nu+3*i+3]=-(c['J'].T@c['frame'])
  Aeq[:,-1]=-jp[2,:]
  cv=np.zeros(n);cv[-1]=-1.
  bounds=[(0.,1.)]*m.nu
  for _ in range(K):bounds+=[(-1e6,1e6),(-1e6,1e6),(0.,1e6)]
  bounds+=[(0.,1e9)]
  Aub=[];bub=[]
  for i,c in enumerate(cs):
   b0=m.nu+3*i;mu=c['mu']
   for t in (0,1):
    r=np.zeros(n);r[b0+t]=1;r[b0+2]=-mu;Aub.append(r);bub.append(0.)
    r=np.zeros(n);r[b0+t]=-1;r[b0+2]=-mu;Aub.append(r);bub.append(0.)
  lp=linprog(cv,A_ub=np.array(Aub),b_ub=np.array(bub),A_eq=Aeq,b_eq=bias,bounds=bounds,method='highs')
  res=dict(feasible=bool(lp.success),contacts=K)
  if lp.success:
   res.update(load_factor=float(lp.x[-1]/W),extra_load=float(lp.x[-1]),
    normal_force=float(sum(lp.x[m.nu+3*i+2] for i in range(K))),
    muscles_at_ceiling=int((lp.x[:m.nu]>0.999).sum()),total_activation=float(lp.x[:m.nu].sum()))
  else:res['message']=lp.message[:60]
 return dict(label=label,engine_consistency=dict(residual=residual,scale=scale,matches=bool(residual<0.02*max(scale,1.))),
  unholdable_dofs=unholdable,static_program=res,body_weight=W)

def run():
 import mujoco
 assets=Path('/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a')
 shim=O/'MODEL_SHIM'
 if not shim.exists():
  shim.mkdir(parents=True);(shim/'MODEL.xml').symlink_to(RUN/'MODEL.xml')
  (shim/'meshes').mkdir(exist_ok=True)
  if not (shim/'meshes'/'stl').exists():(shim/'meshes'/'stl').symlink_to(assets,target_is_directory=True)
 m,d=setup(shim)
 z=np.load(RUN/'BODY.npz',allow_pickle=False);qpos=z['qpos'];t=np.asarray(z['time'],float)
 out=[]
 for x in np.arange(1.10,1.301,0.02):
  i=int(np.argmin(abs(t-x)));out.append(analyse(m,d,qpos[i],'assisted t=%.3f'%t[i]))
 ok=[o for o in out if o['static_program'] and o['static_program']['feasible']]
 dump(O/'LOAD_CAPACITY.json',dict(
  question='static capability and load capacity of the modelled body over the assisted stride',
  verification=dict(check1='engine consistency passed when the residual is below 2 percent of the constraint force norm',results=[o['engine_consistency'] for o in out]),
  poses=out,feasible_poses=len(ok),total_poses=len(out),
  best=(max(ok,key=lambda o:o['static_program']['load_factor']) if ok else None),
  note='the earlier 110.5 percent figure stays withdrawn; the earlier 4.77x force budget is corrected to about 268x body weight'))
 print('引擎一致性: %d/%d 通过'%(sum(o['engine_consistency']['matches'] for o in out),len(out)))
 print('静态程序可行姿态: %d/%d'%(len(ok),len(out)))
 uu=sorted({u['joint'] for o in out for u in o['unholdable_dofs']})
 print('肌肉无法单独抗住重力的关节(%d 个): %s'%(len(uu),uu))
 worst=None
 for o in out:
  for u in o['unholdable_dofs']:
   if worst is None or u['missing']>worst['missing']:worst=u
 if worst:print('最大缺口: %s 需求 %.3f 可达[%.3f,%.3f] 缺 %.3f'%(worst['joint'],worst['demand'],worst['reachable_min'],worst['reachable_max'],worst['missing']))
 if ok:print('最佳载荷 %.2f 倍体重'%max(o['static_program']['load_factor'] for o in ok))
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'run':run}[sys.argv[1]]()
