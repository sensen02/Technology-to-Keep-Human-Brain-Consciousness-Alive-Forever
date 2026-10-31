"""Both5ms and1ms use existing split path. Strict5ms regression before new250ms run."""
from pathlib import Path
import os,json,sys,subprocess
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_neural_muscle_closure as c
from run_quarter_second_pair import Observer,OUT as HISTORY
from run_body_timestep_diagnostic import OUT as SHORT
from run_one_second_motor_pair import INITIAL,sha,dump
from run_motor_body_integration import require_legal_initial_pose
O=Path('/home/sensen/Desktop/engine修复验收/s2-split-write/20261027T163000Z')
def main():
 O.mkdir(parents=True,exist_ok=False);sources=[Path(__file__),Path(c.__file__),Path(c.cp.CANONICAL_XML),INITIAL,c.POOLS]+[Path('/run/media/sensen/Data2/cell_wound_prototype')/p for p in ('engine/embodied/motor_output.py','engine/embodied/motor_cadence_schedule.py','test_split_write_clock.py')]
 dump(O/'PREREGISTRATION.json',dict(sources={str(p):sha(p) for p in sources},old_manifest_sha256=sha(HISTORY/'MANIFEST_SHA256.json'),dt_body=.0000125,dt_neural=.0005,dt_sensory=.005,command_periods=[.005,.001],duration=[.05,.25],seed=0,only_change='completed neural block decode/readout and command hold cadence, not rate/filter normalization or sensory hold',order='sample sensory at5ms boundary; write previous completed decoder result; neural samples of current window; body advance with held command; consume each completed spike once at endpoint, valid next window; no future result applied',compare='5ms split must match old BODY state/INTERVAL currents spikes commands; otherwise stop. New first events/waiting, activation, height, orientation, first nonfoot floor, final50ms contacts; no stable-pass gate',auxiliary=False))
 run=subprocess.run([sys.executable,'-m','unittest','-v','test_split_write_clock'],capture_output=True,text=True);(O/'TEST.log').write_text(run.stdout+run.stderr);assert run.returncode==0
 m,q,gate,joints,geomleg=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);require_legal_initial_pose(gate);c.require_six_foot_initial_contacts(m,q,geomleg);built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];manifest=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);groups,_=c.channel_index_map(int(built['n']),built['tier']['root_ids']);dump(O/'MAPPING.json',manifest)
 for tag,dt,duration in [('default5ms',.005,.05),('write1ms',.001,.25)]:
  p=O/tag;p.mkdir();np.random.seed(0);obs=Observer(m,geomleg);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);ev={}
  result,tr,cmd,sp=c._episode_split_motor(m,q,joints,geomleg,built,manifest,groups,'normal',duration,replay=None,push=None,external_touch=False,disabled_channels=(),evidence=ev,motor_translator=None,body_observer=obs,body_dt_s=.0000125,motor_dt_s=dt)
  obs.save(p/'BODY.npz');dump(p/'CONTACTS.json',obs.contacts)
  arrays={k:v for k,v in ev.items() if isinstance(v,np.ndarray)};arrays.update(commands=cmd,spikes=sp,event_ticks=np.concatenate(ev['motor_event_complete_ticks']),event_local_indices=np.concatenate(ev['motor_event_local_indices']));np.savez_compressed(p/'INTERVALS.npz',**arrays)
  if tag=='default5ms':
   checks={};old=np.load(SHORT/'dt12p5us/normal/BODY.npz');new=np.load(p/'BODY.npz')
   for k in old.files:checks['BODY/'+k]=bool(np.array_equal(old[k],new[k]))
   oldi=np.load(SHORT/'dt12p5us/normal/INTERVALS.npz')
   for a,b in [('commands','commands'),('spikes','spikes'),('sensory_current_nA','sensory_current_nA'),('decoded_next_controls','motor_decoded_next_controls'),('activation','motor_activation_end')]:checks[a]=bool(np.array_equal(oldi[a],arrays[b]))
   dump(O/'DEFAULT_REGRESSION.json',checks)
   if not all(checks.values()):dump(O/'STOP.json',dict(reason='default5ms regression mismatch; new behavior not run'));raise SystemExit(1)
  dump(p/'CLOCK.json',dict(dt_body=.0000125,dt_neural=.0005,dt_sensory=.005,dt_motor=dt,body_steps=len(obs.rows)-1,last_time=obs.rows[-1]['time'],neural_steps=round(duration/.0005)));print(tag,'complete',flush=True)
 assert all(sha(Path(p))==v for p,v in json.loads((O/'PREREGISTRATION.json').read_text())['sources'].items())
 dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,body_steps=24000,body_step_forwards=24000,neural_steps=600,model_loads=1,additional_initialization_forwards='prepare/contact gates and2 recording+2episode initials'))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
