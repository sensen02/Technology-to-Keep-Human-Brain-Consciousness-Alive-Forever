"""Exactly four fresh closed-loop 50ms episodes. Sole runtime change: body dt."""
from pathlib import Path
import json,time,traceback,os
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
from run_one_second_motor_pair import Observer,INITIAL,sha,dump
import run_neural_muscle_closure as c
from run_motor_body_integration import require_legal_initial_pose
OUT=Path('/home/sensen/Desktop/engine修复验收/s2-body-timestep/20261027T155000Z')
BASE=Path('/home/sensen/Desktop/engine修复验收/s2-one-second-motor-pair/20261027T153000Z')
class FloorObserver(Observer):
 def __call__(self,m,d):
  super().__call__(m,d);has=np.zeros(6,bool);force=np.zeros(6)
  for row in self.contacts[-1]:
   a,b=int(row[1]),int(row[2]);g=b if m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE else a if m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE else None
   if g in self.geomleg:
    j=c.LEGS.index(self.geomleg[g]);has[j]=True;force[j]+=max(row[7],0.)
  self.rows[-1]['foot_contact']=has;self.rows[-1]['foot_normal_force']=force

def main():
 o=OUT;o.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),Path(c.__file__),Path(c.cp.CANONICAL_XML),INITIAL,c.POOLS,Path('/run/media/sensen/Data2/cell_wound_prototype/run_one_second_motor_pair.py')]
 files += [Path('/run/media/sensen/Data2/cell_wound_prototype')/s for s in ('engine/embodied/motor_output.py','engine/embodied/sensory_channels.py','engine/neural_cond.py')]
 dump(o/'PREREGISTRATION.json',dict(goal='S2/P3 numerical step sensitivity; not optimization',source_hashes={str(p):sha(p) for p in files},baseline_manifest_sha256=sha(BASE/'MANIFEST_SHA256.json'),baseline='existing100us first50ms only, no rerun',new_dt_s=[.000025,.0000125],duration_s=.05,neural_dt_s=.0005,command_dt_s=.005,seed=0,cut_s=.015,only_runtime_change='model.opt.timestep; no XML or decoder/network/initial changes',auxiliary=False,comparison='exact integer row subsampling at common100us endpoints; per-field maxabs/RMS coarse-medium vs medium-fine; quaternion geodesic and contact mismatch; paired normal-cut difference. Smaller difference is evidence only, no strict convergence claim.',endpoints='activation peak/count outside[0,1],thorax z, first crossing90pct height,root orientation,qpos/qvel,6 true floor-contact flags and local-normal force (not world support),actual clocks and neural spikes',no_extra_runs=True))
 baseline_manifest=json.loads((BASE/'MANIFEST_SHA256.json').read_text());assert all(sha(BASE/p)==h for p,h in baseline_manifest.items())
 built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));summary={}
 for tag,dt in [('dt25us',.000025),('dt12p5us',.0000125)]:
  (o/tag).mkdir();dump(o/tag/'CONFIG.json',dict(body_dt_s=dt,neural_dt_s=.0005,command_dt_s=.005,cut_s=.015,xml_sha256=sha(c.cp.CANONICAL_XML),initial_sha256=sha(INITIAL)))
  m,q,gate,joints,geomleg=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=dt);require_legal_initial_pose(gate);c.require_six_foot_initial_contacts(m,q,geomleg);assert m.neq==0
  dump(o/tag/'INITIAL_GATE.json',gate)
  names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];manifest=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);groups,_=c.channel_index_map(int(built['n']),built['tier']['root_ids'])
  for arm in ('normal','motor_cut'):
   p=o/tag/arm;p.mkdir();np.random.seed(0);obs=FloorObserver(m,geomleg);ev={};d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);err=None
   try:
    result,tr,cmd,sp=c.episode(m,q,joints,geomleg,built,manifest,groups,arm,.05,stride=1,full_detail_intervals=0,external_touch=False,push=None,evidence=ev,body_observer=obs,body_dt_s=dt)
    np.savez_compressed(p/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(p/'RECORDS.json',result)
   except Exception as e:err=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
   obs.save(p/'BODY.npz');dump(p/'CONTACTS.json',obs.contacts)
   summary[tag+'/'+arm]=dict(error=err,body_steps=len(obs.rows)-1,explicit_step_forwards=len(obs.rows)-1,neural_steps=100 if err is None else None,last_time=obs.rows[-1]['time']);dump(o/'RUN_STATUS.json',summary);print(tag,arm,summary[tag+'/'+arm],flush=True)
 assert all(sha(p)==h for p,h in [(Path(k),v) for k,v in json.loads((o/'PREREGISTRATION.json').read_text())['source_hashes'].items()])
 dump(o/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python '+str(Path(__file__)),exit_code=1 if any(s['error'] for s in summary.values()) else 0,body_step_total=sum(s['body_steps'] for s in summary.values()),neural_step_total=400,model_loads=2,baseline_reruns=0,observation_initial_forward=4,body_prepare_contact_gate_forwards='existing prepare_body + require_six_foot_initial_contacts per model, additional to step forwards'))
 (o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 if any(s['error'] for s in summary.values()):raise SystemExit(1)
if __name__=='__main__':main()
