"""One artificial29-node threshold -2mV250ms closed-loop intervention."""
from pathlib import Path
import os,json,sys,subprocess,inspect
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_neural_muscle_closure as c
from run_quarter_second_pair import Observer,OUT as BASE
from run_one_second_motor_pair import INITIAL,sha,dump
from engine.neural_cond import ConductanceNetwork
from engine.embodied.runtime_pool_guard import load_runtime_pool
O=Path('/home/sensen/Desktop/engine修复验收/s3-motor-threshold-minus2/20261027T205000Z')
TARGET=Path('/home/sensen/Desktop/engine修复验收/s3-silent-motor-drive/20261027T204100Z/TARGETS.json')
def main():
 O.mkdir(parents=True,exist_ok=False);target=json.loads(TARGET.read_text());nodes=target['unique_local_nodes'];assert len(nodes)==29
 files=[Path(__file__),Path(inspect.getfile(ConductanceNetwork)),Path(c.__file__),TARGET,INITIAL,c.POOLS,Path(c.cp.CANONICAL_XML),Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz'),Path(__file__).parent/'test_motor_threshold_offset.py'];hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,only_parameter_change='29 fixed nodes base threshold -50 to -52mV; others -50; threshold_offset_mV default zero',historical_neural_code_sha256='f4e6d5af5473a187f3da521273e606d726bd5df6272df214457e14e9aa289499',artificial_not_biological=True,duration=.25,body_dt=.0000125,neural_dt=.0005,sensory_command_dt=.005,gains=1,metrics=['target events/first spike','11 pools events commands activation','network and nontarget motor events','saturation early15ms perleg','first nonfoot thorax root COM vz','last50 perleg/NONFOOT contacts and saved normal loads','finite warnings activation bounds'],rules='one intervention no baseline repeat no selection',all_tick_spikes='new run only not historical completion'))
 cmd=[sys.executable,'-m','unittest','test_motor_threshold_offset','-v'];r=subprocess.run(cmd,cwd=Path(__file__).parent,capture_output=True,text=True);(O/'TEST.log').write_text(r.stdout+r.stderr);assert r.returncode==0
 all_events=[]
 class Derived(ConductanceNetwork):
  def __init__(self,*a,**kw):
   n=int(a[0]);offset=np.zeros(n);offset[nodes]=-2.;super().__init__(*a,threshold_offset_mV=offset,**kw);dump(O/'THRESHOLD_CHANGES_29.json',[dict(local_node=i,root_id=None,old_mV=self.p.threshold_mV,new_mV=self.p.threshold_mV+offset[i],offset_mV=offset[i]) for i in nodes])
  def step(self,*a,**kw):
   sp=super().step(*a,**kw);all_events.append(sp.copy());return sp
 m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate);built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));mapping=c.build_motor_manifest(load_runtime_pool(c.POOLS),built['tier']['root_ids'],[m.actuator(i).name for i in range(m.nu)]);dump(O/'MAPPING.json',mapping);groups,_=c.channel_index_map(built['n'],built['tier']['root_ids']);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs=Observer(m,geom);obs(m,d);ev={};np.random.seed(0);old=c.ConductanceNetwork;c.ConductanceNetwork=Derived
 try:rec,tr,cmd,sp=c.episode(m,q,joints,geom,built,mapping,groups,'normal',.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125)
 finally:c.ConductanceNetwork=old
 np.savez_compressed(O/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(O/'RECORDS.json',rec);obs.save(O/'BODY.npz');dump(O/'CONTACTS.json',obs.contacts);np.savez_compressed(O/'ALL_NODE_TICK_EVENTS.npz',spikes=np.array(all_events),step_end_ms=np.arange(1,len(all_events)+1)*.5)
 changes=json.loads((O/'THRESHOLD_CHANGES_29.json').read_text())
 for r in changes:r['root_id']=int(built['tier']['root_ids'][r['local_node']])
 dump(O/'THRESHOLD_CHANGES_29.json',changes);assert all(sha(Path(p))==v for p,v in hashes.items());dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=1,mj_forward=20003,mj_step=20000,neural_step=500,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('one threshold intervention exit0')
if __name__=='__main__':main()
