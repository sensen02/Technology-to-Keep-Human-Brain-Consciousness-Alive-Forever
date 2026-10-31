"""S2/P3 two fixed 1s closure arms, full substep observation, no auxiliary drive."""
from pathlib import Path
import argparse,json,hashlib,time,traceback,os
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
import run_neural_muscle_closure as c
from run_motor_body_integration import require_legal_initial_pose
INITIAL=Path('/home/sensen/Desktop/engine修复验收/_历史归档/actual-neural-matched-20261018T031000Z/initial_runtime_state.npz')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item(),allow_nan=False)+'\n')
class Observer:
 def __init__(self,m,geomleg):self.rows=[];self.contacts=[];self.geomleg=geomleg;self.counts=dict(step=0,explicit_forward=0)
 def __call__(self,m,d):
  legs=c.LEGS;foot=np.full((6,3),np.nan);has=np.zeros(6,bool);force=np.zeros(6);contact_rows=[]
  for leg in legs:
   ids=[g for g,v in self.geomleg.items() if v==leg];foot[legs.index(leg)]=d.geom_xpos[ids].mean(0)
  for j in range(d.ncon):
   con=d.contact[j];f=np.zeros(6);mujoco.mj_contactForce(m,d,j,f);contact_rows.append([j,con.geom1,con.geom2,con.dist,*con.pos,*f])
   for g in (int(con.geom1),int(con.geom2)):
    if g in self.geomleg:
     k=legs.index(self.geomleg[g]);has[k]=True;force[k]+=abs(f[0])
  self.contacts.append(contact_rows)
  self.rows.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),activation=d.act.copy(),command=d.ctrl.copy(),muscle_force=d.actuator_force.copy(),foot_position=foot,foot_contact=has,foot_normal_force=force,thorax_position=d.xpos[m.body('Thorax').id].copy(),warnings=np.array([w.number for w in d.warning]),external_force=d.xfrc_applied.copy()))
 def save(self,p):
  if self.rows:np.savez_compressed(p,**{k:np.array([r[k] for r in self.rows]) for k in self.rows[0]})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=False)
 sources=[Path(__file__),Path(c.__file__),Path(c.cp.CANONICAL_XML),INITIAL,c.POOLS,Path('/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/motor_output.py'),Path('/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py')]
 prereg=dict(goal='S2/P3 paired mechanical response not walking',source_hashes={str(p):sha(p) for p in sources},duration_per_arm_s=1.,arms=['normal','motor_cut'],cut='closure.episode: applied=0 at interval start t>=0.015s; decoder connected=False from endpoint>=0.015; native ctrl floor unchanged',seed=0,external_touch=False,push=None,auxiliary_prewarm=False,pose_correction=False,replay=False,initialization='exact saved qpos; zero qvel/act/ctrl, not seating or correcting',dt_body=c.DT_BODY,dt_neural=c.DT_NEURAL,dt_command=c.DT_COMMAND,stop='existing nonfinite/warning/time reset exception; preserve partial arrays, do not patch tolerances',interpretation='initial release/gravity/contact must not be mistaken for walking; cut does not zero activation or native actuator floor',expected_per_arm=dict(neural_steps=2000,body_steps=10000))
 dump(o/'PREREGISTRATION.json',prereg)
 np.random.seed(0)
 m,q,gate,joints,geomleg=c.prepare_body(np.load(INITIAL)['qpos']);dump(o/'INITIAL_GATE.json',gate);require_legal_initial_pose(gate);c.require_six_foot_initial_contacts(m,q,geomleg)
 assert m.neq==0,'model contains equality constraints: refuse unreviewed fixture'
 dump(o/'MODEL.json',dict(nq=m.nq,nv=m.nv,nu=m.nu,na=m.na,nmesh=m.nmesh,nmocap=m.nmocap,neq=m.neq,gravity=m.opt.gravity,ctrlrange=m.actuator_ctrlrange,ctrllimited=m.actuator_ctrllimited,legs=c.LEGS,foot_position_definition='mean world position of mapped tarsus geometry centres, not anatomical claw tip'))
 built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];manifest=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);groups,stats=c.channel_index_map(int(built['n']),built['tier']['root_ids']);dump(o/'MAPPING.json',manifest)
 summary={};start=time.time()
 for arm in ('normal','motor_cut'):
  np.random.seed(0);observer=Observer(m,geomleg);evidence={};directory=o/arm;directory.mkdir();initial=mujoco.MjData(m);initial.qpos[:]=q;mujoco.mj_forward(m,initial);observer(m,initial);error=None
  try:
   result,trajectory,commands,spikes=c.episode(m,q,joints,geomleg,built,manifest,groups,arm,1.,stride=1,full_detail_intervals=0,external_touch=False,push=None,evidence=evidence,body_observer=observer)
   np.savez_compressed(directory/'INTERVALS.npz',trajectory=trajectory,commands=commands,spikes=spikes,**{k:v for k,v in evidence.items() if isinstance(v,np.ndarray)})
   dump(directory/'RECORDS.json',result)
  except Exception as ex:error=dict(type=type(ex).__name__,message=str(ex),traceback=traceback.format_exc())
  observer.save(directory/'BODY.npz');dump(directory/'CONTACTS.json',observer.contacts)
  rows=observer.rows;last=rows[-1];summary[arm]=dict(completed=error is None,error=error,observed_body_steps=len(rows)-1,last_time=last['time'],max_activation=float(max(r['activation'].max() for r in rows)),max_command=float(max(r['command'].max() for r in rows)),thorax_change=(last['thorax_position']-rows[0]['thorax_position']),first_warning_time=next((r['time'] for r in rows if r['warnings'].any()),None))
  dump(o/'SUMMARY.json',dict(arms=summary,runtime_s=time.time()-start));print(arm,json.dumps(summary[arm],default=lambda v:v.tolist()),flush=True)
 assert all(sha(Path(p))==h for p,h in prereg['source_hashes'].items())
 dump(o/'SOURCE_UNCHANGED.json',dict(passed=True));(o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 print('OUTPUT',o,flush=True)
 if not all(r['completed'] for r in summary.values()):raise SystemExit(1)
if __name__=='__main__':main()
