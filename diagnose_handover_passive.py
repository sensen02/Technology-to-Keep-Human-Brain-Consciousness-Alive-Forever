"""One saved-boundary FK to decompose spring/damping vs assisted torque; no integration."""
from pathlib import Path
import json,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf
S=Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T058000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-torque-diagnostic/20261028T060000Z')
def main():
 m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m);z=np.load(S/'HANDOVER_BEFORE.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.act[:]=z['act'];d.ctrl[:]=z['ctrl'];mujoco.mj_forward(m,d)
 jids=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE];qa=np.array([m.jnt_qposadr[j] for j in jids]);va=np.array([m.jnt_dofadr[j] for j in jids]);spring=-m.jnt_stiffness[jids]*(d.qpos[qa]-m.qpos_spring[qa]);damper=-m.dof_damping[va]*d.qvel[va];other=d.qfrc_passive[va]-spring-damper;assist=z['qfrc_applied'][va];bias=d.qfrc_bias[va];muscle=d.qfrc_actuator[va];mechanical=assist+bias*0+muscle
 norms={k:float(np.linalg.norm(v)) for k,v in dict(assist=assist,bias=bias,spring=spring,damper=damper,other_passive=other,passive=d.qfrc_passive[va],actual_muscle=muscle,assist_plus_passive=assist+d.qfrc_passive[va],assist_plus_muscle_plus_passive=assist+muscle+d.qfrc_passive[va],assist_minus_bias_plus_passive=assist-bias+d.qfrc_passive[va]).items()}
 rows=[]
 for k,j in enumerate(jids):rows.append(dict(joint=m.joint(j).name,stiffness=float(m.jnt_stiffness[j]),damping=float(m.dof_damping[va[k]]),qvel=float(d.qvel[va[k]]),spring_displacement=float(d.qpos[qa[k]]-m.qpos_spring[qa[k]]),spring_torque=float(spring[k]),damper_torque=float(damper[k]),other_passive_torque=float(other[k]),bias=float(bias[k]),assist=float(assist[k]),muscle=float(muscle[k])))
 candidates=[]
 for stiffscale,dampscale in [(1,1),(.3,.3),(.1,.1),(1,.1),(.1,1),(0,0)]:
  target=mechanical+(1-stiffscale)*spring+(1-dampscale)*damper
  candidates.append(dict(stiffness_scale=stiffscale,damping_scale=dampscale,required_assist_plus_muscle_retargeted_norm=float(np.linalg.norm(target)),target=target.tolist()))
 out=dict(counts=dict(model_load=1,forward=1,body_steps=0,neural_steps=0),norms=norms,rows=rows,passive_decomposition_max_error=float(abs(other).max()),counterfactual_passive_scales=candidates,note='Counterfactual saved-state torque correction if passive parameters scaled at unchanged pose/velocity. No runtime controller or parameter edit applied. Other passive includes tendon/body-fluid effects if present.')
 p=O/'PASSIVE_DECOMPOSITION.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(norms=norms,stiffness_unique=np.unique(m.jnt_stiffness[jids]).tolist(),damping_unique=np.unique(m.dof_damping[va]).tolist(),passive_error=float(abs(other).max()),largest_damper=sorted(rows,key=lambda x:abs(x['damper_torque']),reverse=True)[:5],candidates=candidates),indent=2))
if __name__=='__main__':main()
