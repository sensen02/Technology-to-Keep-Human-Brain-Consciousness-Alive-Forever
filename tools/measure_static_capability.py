"""Static capability bound: is the modelled muscle set strong enough to hold the body up?

Three measurements at one fixed pose, all taken from the compiled model itself:

  1  force budget      sum of maximum isometric forces of the 90 actuators, in units of body weight
  2  per-hinge margin  for every hinge DOF, the gravity torque it must cancel versus the maximum
                       torque its muscles can deliver in each direction; a hinge where the demand
                       falls outside that range cannot be held no matter what the network does
  3  current support   how much of the body weight the foot contacts are already carrying in that
                       pose, straight from the engine's contact forces

No controller, no time stepping, no neural involvement. This is a capability measurement used to
decide whether segmenting five extra muscles out of a micro-CT could possibly be the binding factor.
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
O=Path('/home/sensen/Desktop/engine修复验收/s5-static-capability/20261105T060000Z')
RUN=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-loop/20261104T070000Z/FINAL_VALIDATION_A/RUN')
LEGS=('LF','LM','LH','RF','RM','RH')
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def run():
 import mujoco
 # The archived XML refers to meshes/stl/*.stl relative to itself; the mesh assets live in the
 # flygym cache, so a shim directory with the XML and a link to the meshes is built instead of
 # touching the sealed run directory.
 assets=Path('/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a')
 shim=O/'MODEL_SHIM'
 if not shim.exists():
  shim.mkdir(parents=True)
  (shim/'MODEL.xml').symlink_to(RUN/'MODEL.xml')
  (shim/'meshes').mkdir(exist_ok=True)
  if not (shim/'meshes'/'stl').exists():(shim/'meshes'/'stl').symlink_to(assets, target_is_directory=True)
 os.chdir(shim)
 m=mujoco.MjModel.from_xml_path('MODEL.xml')
 d=mujoco.MjData(m)
 with np.load(RUN/'BODY.npz',allow_pickle=False) as z:qpos=z['qpos'][0].copy()
 d.qpos[:]=qpos;d.qvel[:]=0.;d.ctrl[:]=1.;mujoco.mj_forward(m,d)
 if m.na and m.opt.disableflags & 0==0:pass
 # full-activation force of every actuator in this pose (engine's own muscle model)
 F=np.zeros(m.nu)
 for i in range(m.nu):
  Fi=abs(float(d.actuator_force[i]))
  F[i]=Fi
 W=float(np.sum(m.body_mass)*abs(m.opt.gravity[2]))
 # Moment arms are read out of the engine one actuator at a time (the sparse moment array is not
 # exposed in a usable form here): with only actuator i activated, qfrc_actuator is its column.
 mom=np.zeros((m.nv,m.nu))
 use_act=m.na>=m.nu and m.na>0
 for i in range(m.nu):
  if use_act:d.act[:]=0.;d.act[i]=1.;d.ctrl[:]=0.;d.ctrl[i]=1.
  else:d.ctrl[:]=0.;d.ctrl[i]=1.
  mujoco.mj_forward(m,d)
  mom[:,i]=np.asarray(d.qfrc_actuator).ravel()
 d.act[:]=1.;d.ctrl[:]=1.;mujoco.mj_forward(m,d)
 bias=np.asarray(d.qfrc_bias).ravel()       # at zero velocity this is the gravity generalised force
 # per-DOF torque availability
 pos=np.where(mom>0,mom,0.).sum(axis=1)  # largest positive generalised force available
 neg=np.where(mom<0,mom,0.).sum(axis=1)  # largest negative generalised force available
 need=-bias                              # what the muscles must supply if contacts did nothing
 rows=[]
 for k in range(m.nv):
  if np.abs(mom[k]).sum()==0.:continue    # this DOF is not reached by any actuator
  lo,hi=float(neg[k]),float(pos[k])
  # reserve in the direction that opposes gravity, the direction a limb has to be held against
  demand=abs(float(need[k]));one_sided=bool(abs(hi)<1e-12 or abs(lo)<1e-12)
  covers=bool(lo-1e-9<=need[k]<=hi+1e-9)
  rows.append(dict(dof=int(k),dof_name=(m.joint(m.dof_jntid[k]).name if m.dof_jntid[k]>=0 else 'root'),
   gravity_demand=demand,muscle_generalised_force_min=lo,muscle_generalised_force_max=hi,
   one_directional=one_sided,covers_gravity_alone=covers))
 driven={r['dof'] for r in rows}
 undriven=[k for k in range(m.nv) if k not in driven]
 # current contact support from the engine
 feet=np.zeros(m.ngeom,bool)
 for g in range(m.ngeom):
  b=m.body(m.geom_bodyid[g]).name or ''
  if any(b.startswith(L) and 'Tars' in b for L in LEGS):feet[g]=True
 planes=m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
 fz_foot=0.;fz_other=0.;nfoot=0
 for ci in range(d.ncon):
  c=d.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
  if not (planes[g1] or planes[g2]):continue
  f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
  fz=float((c.frame.reshape(3,3).T@f[:3])[2])
  if feet[g1] or feet[g2]:fz_foot+=max(fz,0.);nfoot+=1
  else:fz_other+=max(fz,0.)
 res=dict(pose=dict(source=str(RUN/'BODY.npz'),frame=0,body_weight_model_units=W,total_mass=float(np.sum(m.body_mass))),
  force_budget=dict(total_max_force=float(F.sum()),ratio_to_body_weight=float(F.sum()/W),
   per_actuator_median=float(np.median(F)),per_actuator_max=float(F.max())),
  per_hinge=dict(actuated_dofs=len(rows),
   hinges_covering_gravity_alone=sum(1 for r in rows if r['covers_gravity_alone']),
   hinges_one_directional=sum(1 for r in rows if r['one_directional']),
   hinges_below_gravity_alone=[dict(dof_name=r['dof_name'],demand=r['gravity_demand'],
     min=r['muscle_generalised_force_min'],max=r['muscle_generalised_force_max'],
     one_directional=r['one_directional']) for r in rows if not r['covers_gravity_alone']],
   undriven_dofs=[int(k) for k in undriven],
   caveat='this compares muscle torque against gravity with the contacts contributing nothing, which is the most pessimistic case: a ground reaction at the foot can help hold a joint too, so a hinge listed here is not thereby impossible, it is a hinge with no reserve when the foot cannot help'),
  current_support=dict(foot_contacts=int(nfoot),foot_normal_force=float(fz_foot),other_normal_force=float(fz_other),
   foot_fraction_of_weight=float(fz_foot/W),
   meaning='the engine forces at this pose, i.e. what the ground is already carrying before any neural decision'),
  reading='a hinge whose gravity demand exceeds the torque its muscles can deliver cannot be held by ANY activation pattern, so it bounds what the network can ever do; the force budget and the current support fraction say whether strength or coordination is the limit')
 dump(O/'STATIC_CAPABILITY.json',res)
 print(json.dumps({k:res[k] for k in ('force_budget','per_hinge','current_support')},ensure_ascii=False,indent=1)[:1800])
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'run':run}[sys.argv[1]]()
