"""One new 250ms command1ms run; historical baseline is read only."""
from pathlib import Path
import os,json,sys,inspect,traceback,time
from unittest.mock import patch
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_neural_muscle_closure as c
from engine.embodied.motor_output import MotorOutput
from run_quarter_second_pair import Observer as QuarterObserver,OUT as HISTORY
from run_one_second_motor_pair import INITIAL,sha,dump
from run_motor_body_integration import require_legal_initial_pose
O=Path('/home/sensen/Desktop/engine修复验收/s4-command1ms/20261027T222000Z')
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')

def recording_decoder(manifest):
 """Production consume source with a copy-only per-tick observer, no math edits."""
 import textwrap
 source=textwrap.dedent(inspect.getsource(MotorOutput.consume))
 needle='        self.rates = decay*self.rates + (1-decay)*instantaneous'
 assert source.count(needle)==1
 source=source.replace(needle,needle+'\n            self.tick_rates.append(self.rates.copy())')
 namespace=dict(vars(sys.modules[MotorOutput.__module__]))
 exec(compile(source,__file__,'exec'),namespace)
 class RecordingMotor(MotorOutput):
  consume=namespace['consume']
 decoder=RecordingMotor(manifest);decoder.tick_rates=[]
 return decoder,source

class Observer(QuarterObserver):
 def __call__(self,m,d):
  super().__call__(m,d);feet=np.zeros((6,3));non=np.zeros(3)
  for k in range(d.ncon):
   co=d.contact[k];a,b=int(co.geom1),int(co.geom2);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
   if not(pa or pb):continue
   g=b if pa else a;f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);world=(co.frame.reshape(3,3).T@f[:3])*(1 if pa else -1)
   if g in self.geomleg:feet[c.LEGS.index(self.geomleg[g])]+=world
   else:non+=world
  np.testing.assert_allclose(feet.sum(0)+non,self.rows[-1]['ground_force_world'],rtol=1e-12,atol=1e-12)
  self.rows[-1].update(ground_force_foot_world=feet,ground_force_nonfoot_world=non)

def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),ROOT/'test_split_write_clock.py',ROOT/'test_command1ms_preflight.py',Path(c.__file__),Path(inspect.getfile(MotorOutput)),ROOT/'engine/embodied/motor_cadence_schedule.py',ROOT/'run_quarter_second_pair.py',ROOT/'run_body_timestep_diagnostic.py',ROOT/'engine/embodied/sensory_channels.py',ROOT/'data/flywire/banc_connectome.npz',INITIAL,c.POOLS,Path(c.cp.CANONICAL_XML)]
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,history=str(HISTORY/'normal'),history_hashes={str(HISTORY/'normal'/p):sha(HISTORY/'normal'/p) for p in ['BODY.npz','INTERVALS.npz','RECORDS.json','CONTACTS.json']},only_change='motor completed-block consume/readout/write hold5ms to1ms; filter5ms and100Hz normalization unchanged',duration_s=.25,dt_body_s=.0000125,dt_neural_s=.0005,dt_sensory_s=.005,dt_command_s=.001,arms=['normal'],seed=0,initial=str(INITIAL),historical_tier_and_pools=True,preparation=0,external_touch=False,push=None,replay=None,expected=dict(models=1,body_steps=20000,neural_steps=500,command_windows=250,sensory_refreshes=50),rules='Tests before dynamics; no baseline dynamics. First events/decode/write/activation, first15ms perleg, total network/motor and saturation, COM/height/vz/nonfoot/last50ms foot and nonfoot reactions/warnings. No stable-support pass or biological calibration claim.',instrumentation='copy-only production filter line tracing, all network emitted spikes, completed-body sensory observations and actual BODY ctrl, perfoot/NONFOOT world reaction'))
 import unittest
 suite=unittest.defaultTestLoader.loadTestsFromNames(['test_split_write_clock','test_command1ms_preflight'])
 with (O/'TEST.log').open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
 if not result.wasSuccessful():raise RuntimeError('preflight tests failed before any dynamics')
 counts=dict(model_load=0,mj_forward=0,mj_step=0,neural_steps=0,decoder_consume=0);real_forward=mujoco.mj_forward;real_step=mujoco.mj_step
 def forward(*a,**kw):counts['mj_forward']+=1;return real_forward(*a,**kw)
 def step(*a,**kw):counts['mj_step']+=1;return real_step(*a,**kw)
 obs=None;error=None;start=time.time();tick_spikes=[];sensory=[];decoder_box=[]
 try:
  with patch.object(mujoco,'mj_forward',forward),patch.object(mujoco,'mj_step',step):
   m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);counts['model_load']+=1;require_legal_initial_pose(gate);c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate)
   built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];manifest=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);groups,_=c.channel_index_map(int(built['n']),built['tier']['root_ids']);dump(O/'MAPPING.json',manifest)
   decoder,source=recording_decoder(manifest);decoder_box.append(decoder);(O/'INSTRUMENTED_CONSUME.py').write_text(source)
   original_consume=decoder.consume
   def consume(*a,**kw):counts['decoder_consume']+=1;return original_consume(*a,**kw)
   decoder.consume=consume
   network_class=c.ConductanceNetwork
   class Network(network_class):
    def step(self,*a,**kw):
     sp=super().step(*a,**kw);tick_spikes.append(sp.copy());counts['neural_steps']+=1;return sp
   observe=c.observe_quantities
   def observed(model,data,*a,**kw):
    quantities=observe(model,data,*a,**kw);sensory.append(dict(physical_time=float(data.time),observation={leg:vars(v) for leg,v in quantities.items()}));return quantities
   p=O/'normal';p.mkdir();obs=Observer(m,geom);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);ev={};np.random.seed(0)
   with patch.object(c,'make_motor_decoder',return_value=decoder),patch.object(c,'ConductanceNetwork',Network),patch.object(c,'observe_quantities',observed):
    rec,tr,cmd,sp=c.episode(m,q,joints,geom,built,manifest,groups,'normal',.25,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125,motor_dt_s=.001)
   np.savez_compressed(p/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)},event_ticks=np.concatenate(ev['motor_event_complete_ticks']),event_local_indices=np.concatenate(ev['motor_event_local_indices']))
   rates=np.asarray(decoder.tick_rates);np.savez_compressed(p/'TICKS.npz',spikes=np.asarray(tick_spikes),rates_hz=rates,decoded_controls=np.clip(rates/decoder.config.rate_for_full_control_hz,0,decoder.config.max_control),tick_complete_s=np.arange(1,501)*.0005,probe_local_indices=sorted({i for row in manifest['rows'] for i in row['local_indices']}|{int(i) for i in groups['lf']['bristle_touch']}))
   dump(p/'RECORDS.json',rec);dump(p/'SENSORY_OBSERVATIONS.json',sensory)
   np.testing.assert_allclose([r['physical_time'] for r in sensory],np.arange(50)*.005,atol=1e-12,rtol=0)
   assert counts['neural_steps']==500 and counts['mj_step']==20000 and counts['decoder_consume']==250 and len(rates)==500
 except Exception as ex:error=dict(type=type(ex).__name__,message=str(ex),traceback=traceback.format_exc())
 finally:
  if obs is not None:obs.save(O/'normal/BODY.npz');dump(O/'normal/CONTACTS.json',obs.contacts)
  dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0 if error is None else 1,counts=counts,error=error,runtime_s=time.time()-start,preflight_physics_calls=0,preflight_neural_simulator_calls=0))
 if error:raise RuntimeError(error)
 assert all(sha(Path(p))==h for p,h in hashes.items())
 dump(O/'SOURCE_UNCHANGED.json',dict(passed=True));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('single command1ms250ms COMPLETE',counts,flush=True)
if __name__=='__main__':main()
