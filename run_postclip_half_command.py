"""One fresh normal closure: scale all post-filter/post-clip commands, not decoder state."""
from pathlib import Path
import os,json,sys,unittest
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_neural_muscle_closure as c
from run_quarter_second_pair import Observer,OUT as BASE
from run_one_second_motor_pair import INITIAL,sha,dump
O=Path('/home/sensen/Desktop/engine修复验收/s2-postclip-half-command/20261027T181000Z')
class ScaledDecoder:
 def __init__(self,decoder,scale=1.):
  if not np.isfinite(scale) or not 0<=scale<=1:raise ValueError('scale must be finite within [0,1]')
  self.decoder=decoder;self.scale=float(scale)
 def consume(self,*args,**kwargs):
  command,trace=self.decoder.consume(*args,**kwargs)
  applied=command*self.scale
  return applied,dict(trace,postclip_command_scale=self.scale,unscaled_postclip_command=command.tolist(),scaled_command=applied.tolist())
class Tests(unittest.TestCase):
 def test_defaults_and_half(self):
  class Fake:
   def __init__(self):self.rates=np.array([3.,4.,5.]);self.elapsed_s=0.
   def consume(self):self.elapsed_s+=.005;return np.array([0.,.25,1.]),{'source_end_s':self.elapsed_s}
  for scale in (1.,.5):
   f=Fake();rates=f.rates.copy();v,t=ScaledDecoder(f,scale).consume();np.testing.assert_array_equal(v,np.array([0.,.25,1.])*scale);np.testing.assert_array_equal(f.rates,rates);self.assertEqual(f.elapsed_s,.005);self.assertEqual(t['source_end_s'],.005)

def main():
 O.mkdir(parents=True,exist_ok=False)
 from io import StringIO
 stream=StringIO();res=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));(O/'TEST.log').write_text(stream.getvalue());assert res.wasSuccessful()
 sources=[Path(__file__),Path(c.__file__),INITIAL,Path(c.cp.CANONICAL_XML),c.POOLS,Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz'),Path('/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/motor_output.py')];hashes={str(p):sha(p) for p in sources}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,baseline_manifest_sha256=sha(BASE/'MANIFEST_SHA256.json'),factor='all90 commands = .5 * original post-filter/post-clip output; unchanged decoder memory and native ctrl minimum',scale=.5,body_dt=.0000125,neural_dt=.0005,sensory_dt=.005,command_dt=.005,duration=.25,seed=0,arms=['normal'],candidate_pool=False,four_node_tier=False,comparisons=['first ctrl/activation response','90 command/activation peaks','decoder clip flags','all-neuron spike changes','thorax/root/com height and vertical velocity','first nonfoot','last50ms true perfoot/nonfoot world ground forces','warnings/nonfinite'],behavior_gate='existing, no new250ms stable-support pass'))
 m,q,gate,joints,geomleg=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);c.require_six_foot_initial_contacts(m,q,geomleg);dump(O/'INITIAL_GATE.json',gate)
 built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];manifest=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);dump(O/'MAPPING.json',manifest);groups,_=c.channel_index_map(int(built['n']),built['tier']['root_ids']);factory=c.make_motor_decoder
 # Exact real-decoder state/timing check using identical synthetic spikes; no neural integration.
 a=factory(manifest);b=factory(manifest);wrapped=ScaledDecoder(b,.5);sp=np.zeros((10,built['n']),np.uint8);sp[0,:]=1
 for k in range(2):
  un,ta=a.consume(sp,spike_dt_s=.0005,start_s=k*.005,end_s=(k+1)*.005);half,tb=wrapped.consume(sp,spike_dt_s=.0005,start_s=k*.005,end_s=(k+1)*.005);np.testing.assert_array_equal(half,un*.5);np.testing.assert_array_equal(a.rates,b.rates);assert a.elapsed_s==b.elapsed_s and a.interval==b.interval
 dump(O/'TESTS.json',dict(default_and_half_zero_nonsaturated_saturated=True,real_decoder_rates_clock_interval_unchanged=True,scale1_identity=True,exit_code=0))
 c.make_motor_decoder=lambda manifest,motor_translator=None:ScaledDecoder(factory(manifest,motor_translator),.5)
 np.random.seed(0);obs=Observer(m,geomleg);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);ev={}
 try:records,tr,cmd,spikes=c.episode(m,q,joints,geomleg,built,manifest,groups,'normal',.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125)
 finally:c.make_motor_decoder=factory
 np.savez_compressed(O/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=spikes,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(O/'RECORDS.json',records);obs.save(O/'BODY.npz');dump(O/'CONTACTS.json',obs.contacts)
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,body_steps=len(obs.rows)-1,step_explicit_forwards=len(obs.rows)-1,neural_steps=500,model_load=1,observation_initial_forward=1,prepare_and_episode_initialization_forwards='unchanged historical path',failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('half250ms exit0')
if __name__=='__main__':main()
