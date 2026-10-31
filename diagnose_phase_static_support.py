"""Saved-pose static-gravity counterfactual; no integration/control rollout."""
from pathlib import Path
import os,json,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf
from run_physical_walk_prewarm import contacts
S=Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T058000Z');O=Path('/home/sensen/Desktop/engine修复验收/s5-handover-phase-selection/20261028T064000Z')
def main():
 started=time.time();rank=json.loads((O/'CANDIDATES.json').read_text());z=np.load(S/'BODY.npz');q=z['qpos'];act=z['activation'];m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m);weight=m.body_mass.sum()*abs(m.opt.gravity[2]);rows=[]
 for r in rank:
  i=r['saved_index'];d.qpos[:]=q[i];d.qvel[:]=0;d.act[:]=act[i];d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;mujoco.mj_forward(m,d);f,nf,fc,nc=contacts(m,d)
  rows.append(dict(time_s=r['target_time_s'],eligible=r['eligible'],static_velocity_zero=True,foot_load_ratio=float(f[:,2].sum()/weight),nonfoot_contacts=nc,feet_contacting=int(fc.sum()),root_vertical_qacc=float(d.qacc[2]),root_angular_acceleration_norm=float(np.linalg.norm(d.qacc[3:6])),max_abs_qacc=float(abs(d.qacc).max()),scope='Same saved pose with zero velocity and native floor activation, no controller. Static-gravity FK acceleration, not an equilibrium solution or prospective trajectory.'))
 target=O/'STATIC_GRAVITY_COUNTERFACTUAL.json';assert not target.exists();target.write_text(json.dumps(dict(rows=rows,counts=dict(model_load=1,forward=len(rows),body_steps=0,neural_steps=0),runtime_s=time.time()-started),indent=2)+'\n');(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps([r for r in rows if r['time_s'] in [.82,.915,1.]],indent=2))
if __name__=='__main__':main()
