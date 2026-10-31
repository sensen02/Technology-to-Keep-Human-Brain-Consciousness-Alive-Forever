"""One dynamic5ms sensory-current x2 closure; no other input scaled."""
from pathlib import Path
import inspect,os,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_fast_sensory_only as parent
from run_one_second_motor_pair import INITIAL,sha,dump
c=parent.c;BASE=parent.BASE
O=Path('/home/sensen/Desktop/engine修复验收/s2-sensory-current-double/20261027T185000Z')
def scale_sensory(current,multiplier=1.):
 return np.asarray(current)*float(multiplier)
class ScaledEncoder:
 def __init__(self,encoder,multiplier,log):self.encoder=encoder;self.multiplier=multiplier;self.log=log
 def encode(self,quantities):
  raw,channels=self.encoder.encode(quantities);scaled=scale_sensory(raw,self.multiplier);self.log.append(dict(raw_encoder_current=raw.copy(),requested_current=scaled.copy()));return scaled,channels

def tests():
 x=np.array([0.,.2,-.3,.2+.4]);old=x.copy();np.testing.assert_array_equal(scale_sensory(x),old);np.testing.assert_array_equal(scale_sensory(x,2),old*2);np.testing.assert_array_equal(x,old)
 class Fake:
  def __init__(self):self.array=x.copy();self.state=4
  def encode(self,q):return self.array,{'state':self.state}
 f=Fake();enc=ScaledEncoder(f,2,[])
 for _ in range(3):out,_=enc.encode(None);np.testing.assert_array_equal(out,old*2);np.testing.assert_array_equal(f.array,old)
 assert f.state==4
 result=parent.tests();result.update(multiplier1_input_identity=True,multiplier2_zero_positive_negative_overlap_correct=True,no_mutation_or_accumulation=True,exit_code=0);return result

def main():
 O.mkdir(parents=True,exist_ok=False);dump(O/'TESTS.json',tests());files=[Path(__file__),Path(parent.__file__),Path(c.__file__),INITIAL,c.POOLS,Path(c.cp.CANONICAL_XML),Path('/run/media/sensen/Data2/cell_wound_prototype/engine/neural_cond.py'),Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz')];hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,only_factor='sensory encoder completed output times2 before any external input; this changes initial and dynamic drive, not exclusively feedback-dependent component',multiplier=2,sensory_command_dt=.005,neural_dt=.0005,body_dt=.0000125,duration=.25,scale_command=1,preparation=0,external_touch=False,background_current=0,clipping_position='after original encoder processing; ConductanceNetwork.step validates current but no subsequent current amplitude clip',rules=['raw requested effective currents/ranges','full and motor spikes first15ms commands/activation','decoder saturation/act bounds','thorax root COM vz','first nonfoot','last50ms perfoot/nonfoot contacts','warnings finite'],behavior='NOT_JUDGED no short-window pass'))
 m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate);built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];mapping=c.build_motor_manifest(__import__('json').loads(c.POOLS.read_text()),built['tier']['root_ids'],names);dump(O/'MAPPING.json',mapping);groups,_=c.channel_index_map(built['n'],built['tier']['root_ids']);log=[];refresh=[];fn,src=parent.derive(10,refresh);constructor=c.ChannelSensoryEncoder
 fn.__globals__['ChannelSensoryEncoder']=lambda *a,**k:ScaledEncoder(constructor(*a,**k),2,log)
 # Real encoder default1 and2 path, unchanged state, before any physical run.
 initial=mujoco.MjData(m);initial.qpos[:]=q;mujoco.mj_forward(m,initial);quantities=c.observe_quantities(m,initial,q,joints,geom);real=constructor(groups,built['n'],c.ChannelConfig());raw,_=real.encode(quantities);np.testing.assert_array_equal(ScaledEncoder(real,1,[]).encode(quantities)[0],raw);np.testing.assert_array_equal(ScaledEncoder(real,2,[]).encode(quantities)[0],raw*2)
 observer=parent.Observer(m,geom);observer(m,initial);ev={};np.random.seed(0);rec,tr,cmd,sp=fn(m,q,joints,geom,built,mapping,groups,'normal',.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=observer,body_dt_s=.0000125)
 assert len(log)==50 and len(refresh)==50
 for r,e in zip(log,refresh):np.testing.assert_array_equal(r['requested_current'],e['current']);r.update(tick=e['tick'],physical_time=e['physical_time'],effective_input_current=e['current'],observation=e['observation'],subsequent_clip=False)
 np.savez_compressed(O/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(O/'RECORDS.json',rec);observer.save(O/'BODY.npz');dump(O/'CONTACTS.json',observer.contacts);dump(O/'CURRENT_RAW_REQUESTED_EFFECTIVE.json',log);dump(O/'SENSORY_REFRESH.json',refresh);(O/'DERIVED_EPISODE.py').write_text(src)
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=1,mj_forward=20003,mj_step=20000,neural_step=500,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('dynamic sensory x2 normal250ms exit0')
if __name__=='__main__':main()
