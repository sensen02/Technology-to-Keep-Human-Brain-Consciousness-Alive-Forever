"""Two250ms dt12.5us fresh closure runs, unchanged controls; no parameter search."""
from pathlib import Path
import os,json,traceback
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_neural_muscle_closure as c
from run_one_second_motor_pair import INITIAL,sha,dump
from run_body_timestep_diagnostic import FloorObserver,OUT as SHORT
from run_motor_body_integration import require_legal_initial_pose
OUT=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z')
class Observer(FloorObserver):
 def __call__(self,m,d):
  super().__call__(m,d);mass=m.body_mass.sum();r=self.rows[-1];com=(d.xipos*m.body_mass[:,None]).sum(0)/mass;vel=np.zeros(3)
  for b in range(1,m.nbody):
   jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBodyCom(m,d,jp,jr,b);vel+=m.body_mass[b]*(jp@d.qvel)/mass
  nonfoot=[];world=np.zeros(3)
  for k in range(d.ncon):
   co=d.contact[k];a,b=int(co.geom1),int(co.geom2);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
   if not(pa or pb):continue
   g=b if pa else a;f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);world+=(co.frame.reshape(3,3).T@f[:3])*(1 if pa else -1)
   if g not in self.geomleg:nonfoot.append(g)
  r.update(com_position=com,com_velocity=vel,nonfoot_floor_count=len(nonfoot),ground_force_world=world)

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 sources=[Path(__file__),Path(c.__file__),INITIAL,Path(c.cp.CANONICAL_XML),Path('/run/media/sensen/Data2/cell_wound_prototype/run_body_timestep_diagnostic.py')]
 dump(OUT/'PREREGISTRATION.json',dict(sources={str(p):sha(p) for p in sources},short_manifest_sha256=sha(SHORT/'MANIFEST_SHA256.json'),dt_body=.0000125,dt_neural=.0005,dt_command=.005,duration=.25,seed=0,cut_s=.015,auxiliary=False,only_change_from_short='duration50ms to250ms and observational COM/nonfoot/world-force fields',rules='First50ms all shared BODY and INTERVAL arrays must match history bitwise before interpreting later response. Report all mismatches. Distinguish height recovery vs body-floor arrest vs foot-only contacts; no new stable-support pass gate.',stability='NOT_JUDGED unless existing formal behavior gate actually evaluated',calls_expected=dict(arms=2,body_steps=40000,neural_steps=1000)))
 m,q,gate,joints,geomleg=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);require_legal_initial_pose(gate);c.require_six_foot_initial_contacts(m,q,geomleg);dump(OUT/'INITIAL_GATE.json',gate)
 built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];manifest=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);groups,_=c.channel_index_map(int(built['n']),built['tier']['root_ids']);status={}
 for arm in ('normal','motor_cut'):
  p=OUT/arm;p.mkdir();np.random.seed(0);obs=Observer(m,geomleg);ev={};d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);err=None
  try:
   records,tr,cmd,sp=c.episode(m,q,joints,geomleg,built,manifest,groups,arm,.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125)
   np.savez_compressed(p/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(p/'RECORDS.json',records)
  except Exception as e:err=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
  obs.save(p/'BODY.npz');dump(p/'CONTACTS.json',obs.contacts)
  match={}
  if err is None:
   with np.load(SHORT/'dt12p5us'/arm/'BODY.npz') as old,np.load(p/'BODY.npz') as new:
    for k in old.files:match['BODY/'+k]=dict(exact=bool(np.array_equal(old[k],new[k][:len(old[k])])),max_abs=float(np.max(abs(old[k].astype(float)-new[k][:len(old[k])].astype(float)))))
   with np.load(SHORT/'dt12p5us'/arm/'INTERVALS.npz') as old,np.load(p/'INTERVALS.npz') as new:
    for k in old.files:match['INTERVAL/'+k]=dict(exact=bool(np.array_equal(old[k],new[k][:len(old[k])])),max_abs=float(np.max(abs(old[k].astype(float)-new[k][:len(old[k])].astype(float)))))
  dump(p/'PREFIX_CHECK.json',match);status[arm]=dict(error=err,body_steps=len(obs.rows)-1,last_time=obs.rows[-1]['time'],prefix_exact=bool(match and all(v['exact'] for v in match.values())));dump(OUT/'STATUS.json',status);print(arm,status[arm],flush=True)
 assert all(sha(Path(p))==h for p,h in json.loads((OUT/'PREREGISTRATION.json').read_text())['sources'].items())
 dump(OUT/'EXECUTION.json',dict(exit_code=0 if all(s['error'] is None and s['prefix_exact'] for s in status.values()) else 1,command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python '+str(Path(__file__)),step_calls=sum(s['body_steps'] for s in status.values()),neural_calls=1000,step_forward_calls=sum(s['body_steps'] for s in status.values()),model_loads=1,observation_initial_forward=2,prepare_contact_gate_forward='existing initialization calls additional'))
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 if not all(s['error'] is None and s['prefix_exact'] for s in status.values()):raise SystemExit(1)
if __name__=='__main__':main()
