"""Static Cartesian retarget prototype only: no physics steps or neural execution."""
from pathlib import Path
import os,json,hashlib,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
from scipy.optimize import least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf
BASE=Path('/home/sensen/Desktop/engine修复验收/s4-integrated-repaired-closure/20261028T032000Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-noncrossing-static-prototype/20261028T041000Z')
LEGS=('LF','LM','LH','RF','RM','RH')
def main():
 OUT.mkdir(parents=True,exist_ok=False);m=_load_mjcf(str(BASE/'INTEGRATED_MODEL.xml')).compile();d=mujoco.MjData(m);counts={'model_load':1,'forward':0,'physics_steps':0,'neural_steps':0}
 q=np.load(BASE/'INITIAL.npz')['qpos'].copy();q[:7]=[0,0,1.5,1,0,0,0]
 hinges=[j for j in range(m.njnt) if int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_HINGE)];adr=np.array([m.jnt_qposadr[j] for j in hinges]);assert len(adr)==42
 lo=m.jnt_range[hinges,0].copy();hi=m.jnt_range[hinges,1].copy();center=np.clip(q[adr],lo+1e-6,hi-1e-6)
 gids=[];local=[]
 for leg in LEGS:
  candidates=[g for g in range(m.ngeom) if m.geom(g).name.startswith(leg) and 'Tarsus5' in m.geom(g).name]
  if len(candidates)!=1:raise ValueError((leg,candidates))
  g=candidates[0];gids.append(g);mid=m.geom_dataid[g];verts=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]];local.append(verts[np.argmin(verts[:,2])].copy())
 local=np.array(local);target=np.array([[.8,.85,0],[0,1.05,0],[-.8,.85,0],[.8,-.85,0],[0,-1.05,0],[-.8,-.85,0]])
 def points(x):
  d.qpos[:]=q;d.qpos[adr]=x;mujoco.mj_forward(m,d);counts['forward']+=1
  return np.array([d.geom_xpos[g]+d.geom_xmat[g].reshape(3,3)@v for g,v in zip(gids,local)])
 vadr=np.array([m.jnt_dofadr[j] for j in hinges])
 def residual(x):return np.r_[((points(x)-target)*10).ravel(),.015*(x-center)]
 def jacobian(x):
  pts=points(x);rows=[]
  for g,pt in zip(gids,pts):
   jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,pt,int(m.geom_bodyid[g]));rows.append(jp[:,vadr]*10)
  return np.vstack(rows+[np.eye(42)*.015])
 result=least_squares(residual,center,jac=jacobian,bounds=(lo+1e-7,hi-1e-7),max_nfev=100,xtol=1e-9,ftol=1e-9,gtol=1e-8)
 points_out=points(result.x);q[adr]=result.x;np.savez_compressed(OUT/'STATIC_STANCE.npz',qpos=q,foot_points=points_out,targets=target,hinge_qpos_addresses=adr)
 error=np.linalg.norm(points_out-target,axis=1);front_separation=float(points_out[0,1]-points_out[3,1]);side_valid=bool(np.all(points_out[:3,1]>.2) and np.all(points_out[3:,1]<-.2));facts=dict(counts=counts,success=bool(result.success),optimizer_status=int(result.status),optimizer_message=result.message,foot_target_error=error.tolist(),foot_points=points_out.tolist(),targets=target.tolist(),front_separation=front_separation,left_right_separation_valid=side_valid,root_height=1.5,max_foot_error=float(error.max()),requires_visual_and_physics_validation=True,scope='Engineering Cartesian footprint target, not measured anatomy or accepted walking. No dynamics or per-step qpos write. Stance source is derived032 repaired model, no model file modified.')
 (OUT/'RESULT.json').write_text(json.dumps(facts,indent=2)+'\n');(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(facts,indent=2))
if __name__=='__main__':main()
