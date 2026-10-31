"""Single fixed-initial-observation diagnostic; free physical body, original5ms clocks."""
from pathlib import Path
import copy,inspect,json,os,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_fast_sensory_only as parent
from run_one_second_motor_pair import INITIAL,sha,dump
c=parent.c;BASE=parent.BASE
O=Path('/home/sensen/Desktop/engine修复验收/s2-fixed-initial-sensory/20261027T184000Z')
class ObservationMode:
 def __init__(self,mode='dynamic'):
  if mode not in ('dynamic','fixed_initial'):raise ValueError(mode)
  self.mode=mode;self.initial=None;self.captures=0
 def select(self,actual):
  if self.initial is None:self.initial=copy.deepcopy(actual);self.captures+=1
  return copy.deepcopy(self.initial if self.mode=='fixed_initial' else actual)
def derive(mode,log,refresh_log):
 fn,src=parent.derive(10,refresh_log);selection=ObservationMode(mode)
 def observe(model,d,reference,joints,geom):
  actual=c.observe_quantities(model,d,reference,joints,geom);sent=selection.select(actual)
  log.append(dict(time=float(d.time),actual_observation={k:vars(v) for k,v in actual.items()},encoder_observation={k:vars(v) for k,v in sent.items()}));return sent
 fn.__globals__['observe_quantities']=observe
 return fn,src,selection

def tests():
 # Exercise the derived episode's actual observation selection using its bound callback.
 records=[];refresh=[];dynamic,_,sel=derive('dynamic',records,refresh);fixed,_,fsel=derive('fixed_initial',[],[])
 original=c.observe_quantities
 try:
  from engine.embodied.sensory_channels import ChannelQuantities
  class Data:time=0.
  d=Data();i=[0]
  c.observe_quantities=lambda *a: {'lf':ChannelQuantities(tibia_joint_dev_rad=float(i[0]),tibia_joint_vel_rad_s=0.,multi_joint_dev_rad=0.,contact_force_norm=0.,foot_contact=1.)}
  dy=[];fx=[]
  for j in range(3):
   i[0]=j;d.time=j*.005;dy.append(dynamic.__globals__['observe_quantities'](None,d,None,None,None));fx.append(fixed.__globals__['observe_quantities'](None,d,None,None,None))
  assert [x['lf'].tibia_joint_dev_rad for x in dy]==[0.,1.,2.];assert [x['lf'].tibia_joint_dev_rad for x in fx]==[0.,0.,0.];assert fsel.captures==1
  # Real encoder: same explicit synthetic groups for every channel, two independent identical encoders.
  # Explicit pure encoding reference validates selection+encoding+production scheduler end to end.
  encode=lambda observation:np.array([observation['lf'].tibia_joint_dev_rad+2.])
  def schedule(mode):
   fn,_,_=derive(mode,[],[]);inputs=[];writes=[]
   for k in range(3):
    i[0]=k;d.time=k*.005;sent=fn.__globals__['observe_quantities'](None,d,None,None,None);cur=encode(sent);writes.append(k*10)
    parent.advance_block(k*10,10,cur,lambda t:None,lambda x:(inputs.append(x.copy()) or x.copy()),lambda j:None)
   return np.array(inputs),writes
  di,dw=schedule('dynamic');fi,fw=schedule('fixed_initial');np.testing.assert_array_equal(di,np.repeat([[2.],[3.],[4.]],10,axis=0));np.testing.assert_array_equal(fi,np.full((30,1),2.));assert dw==fw==[0,10,20]
 finally:c.observe_quantities=original
 result=parent.tests();result.update(dynamic_selection_exercised_in_derived_function=True,fixed_captures_once=True,changed_observations_not_used_in_fixed=True,first_observation_equal=True)
 return result

def main():
 O.mkdir(parents=True,exist_ok=False);dump(O/'TESTS.json',tests())
 files=[Path(__file__),Path(parent.__file__),Path(c.__file__),INITIAL,c.POOLS,Path(c.cp.CANONICAL_XML),Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz')];hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,only_difference='encoder receives single captured real release-initial observation instead of changing current body observation; actual observations still logged every5ms',mode='fixed_initial',duration=.25,body_dt=.0000125,neural_dt=.0005,sensory_dt=.005,command_dt=.005,scale=1,preparation=0,rules=['actual vs encoded observations','first15ms perleg emitted motor events command activation','all-neuron output','thorax root COM height vz','first_nonfoot','last50ms perfoot/nonfoot plane contacts','activation bounds finite warnings'],behavior='NOT_JUDGED no new short gate',not_full_state_dependent_feedback=True))
 m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate);built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];mapping=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);dump(O/'MAPPING.json',mapping);groups,_=c.channel_index_map(built['n'],built['tier']['root_ids'])
 # Exact real-encoder behavioral check through derived observation callback with fixed changing quantities.
 d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs0=c.observe_quantities(m,d,q,joints,geom);enc1=c.ChannelSensoryEncoder(groups,built['n'],c.ChannelConfig());enc2=c.ChannelSensoryEncoder(groups,built['n'],c.ChannelConfig());actual_input=enc1.encode(obs0)[0];fixed_input=enc2.encode(ObservationMode('fixed_initial').select(obs0))[0];np.testing.assert_array_equal(actual_input,fixed_input)
 observer=parent.Observer(m,geom);observer(m,d);log=[];refresh=[];fn,src,selection=derive('fixed_initial',log,refresh);ev={};np.random.seed(0)
 rec,tr,cmd,sp=fn(m,q,joints,geom,built,mapping,groups,'normal',.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=observer,body_dt_s=.0000125)
 assert selection.captures==1;assert all(np.array_equal(r['current'],refresh[0]['current']) for r in refresh)
 np.savez_compressed(O/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(O/'RECORDS.json',rec);observer.save(O/'BODY.npz');dump(O/'CONTACTS.json',observer.contacts);dump(O/'OBSERVATION_ACTUAL_VS_ENCODER.json',log);dump(O/'SENSORY_REFRESH.json',refresh);(O/'DERIVED_EPISODE.py').write_text(src)
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=1,mj_forward=20003,mj_step=20000,neural_step=500,sensory_refresh=50,initial_capture_count=selection.captures,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('fixed initial sensory250ms exit0')
if __name__=='__main__':main()
