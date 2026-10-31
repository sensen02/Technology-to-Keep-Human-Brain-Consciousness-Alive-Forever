"""One static model,2forwards: distinguish geometric moment signs from active force-length losses."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf
P=Path(__file__).parent;O=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-geometry-provenance/20261028T083000Z');S=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
def dump(n,x):
 def enc(v):return v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)
 (O/n).write_text(json.dumps(x,indent=2,default=enc)+'\n')
def main():
 pre=O/'OPERATING_PREREGISTRATION.json';assert not pre.exists();dump(pre.name,dict(source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),S/'MODEL.xml',S/'HANDOVER_BEFORE.npz']},counts=dict(model_load=1,forward=2,body_steps=0,neural_steps=0),scope='Saved074fixedqvel0, native activation floor vsall90activation1. No model mutation, geometry vsforce-length authority only.'))
 m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m);d.qpos[:]=np.load(S/'HANDOVER_BEFORE.npz')['qpos'];d.qvel[:]=0;d.act[:]=.0001;d.ctrl[:]=0;mujoco.mj_forward(m,d);floor=d.actuator_force.copy();moment=np.zeros((90,48))
 for j in range(90):
  ad=int(d.moment_rowadr[j]);n=int(d.moment_rownnz[j]);moment[j,d.moment_colind[ad:ad+n]]=d.actuator_moment[ad:ad+n]
 lengths=d.actuator_length.copy();d.act[:]=1.;mujoco.mj_forward(m,d);gain=(d.actuator_force-floor)/.9999;A=moment.T*gain;rows=[]
 for name in ['joint_RMCoxa_yaw','joint_RHTrochanter_pitch','joint_LHTrochanter_pitch']:
  v=int(m.jnt_dofadr[m.joint(name).id]);selected=[]
  for j in range(90):
   if abs(moment[j,v])<1e-8:continue
   selected.append(dict(actuator=m.actuator(j).name,moment=float(moment[j,v]),positive_geometric_direction=bool(moment[j,v]<0),force_per_unit_activation=float(gain[j]),active_torque=float(A[v,j]),length=float(lengths[j]),lengthrange=m.actuator_lengthrange[j].copy(),lengthrange_fraction=float((lengths[j]-m.actuator_lengthrange[j,0])/np.diff(m.actuator_lengthrange[j])[0]),gainprm=m.actuator_gainprm[j].copy(),biasprm=m.actuator_biasprm[j].copy()))
  rows.append(dict(joint=name,positive_geometric_count=sum(x['positive_geometric_direction'] for x in selected),positive_effective_count=sum(x['active_torque']>1e-8 for x in selected),actuators=selected))
 dump('OPERATING_AUTHORITY.json',rows);np.savez_compressed(O/'OPERATING_ARRAYS.npz',moment=moment,active_torque=A,force_floor=floor,force_per_unit_activation=gain,lengths=lengths,lengthrange=m.actuator_lengthrange,qpos=d.qpos);dump('OPERATING_EXECUTION.json',dict(counts=dict(model_load=1,forward=2,body_steps=0,neural_steps=0),exit_code=0));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(rows,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),flush=True)
if __name__=='__main__':main()
