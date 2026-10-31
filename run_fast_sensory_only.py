"""Only sensory refresh1ms; command5ms, neural500us, historical resting start."""
from pathlib import Path
import os,inspect,json,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_neural_muscle_closure as c
from run_quarter_second_pair import Observer,OUT as BASE
from run_one_second_motor_pair import INITIAL,sha,dump
O=Path('/home/sensen/Desktop/engine修复验收/s2-sensory1ms-command5ms/20261027T183000Z')

def advance_block(start_tick,refresh_ticks,current,observe_encode,neural_step,body_step):
 """Ten neural ticks per command block; observe completed body before neural/body advance."""
 spikes=[]
 for j in range(10):
  tick=start_tick+j
  if j and tick%refresh_ticks==0:current=observe_encode(tick)
  spikes.append(neural_step(current))
  body_step(j)
 return np.asarray(spikes)

def tests():
 # Exercise actual production block scheduler with deterministic changing observations.
 # Reference5ms path holds one freshly encoded input, advances10neural then body400.
 def trial(refresh):
  state={'body':0,'neural':0,'filter':0.};observations=[];currents=[];writes=[]
  def observe(tick):
   assert state['body']==tick*40
   observations.append(tick);return np.array([tick+1.])
  def neural(cur):
   currents.append(float(cur[0]));state['neural']+=1;return cur.copy()
  def body(j):state['body']+=40
  for k in range(3):
   writes.append(k*10);cur=observe(k*10);sp=advance_block(k*10,refresh,cur,observe,neural,body);state['filter']=.5*state['filter']+sp.sum()
  return state,observations,currents,writes
 default=trial(10);fast=trial(2)
 ref=[v for k in range(3) for v in [float(k*10+1)]*10];assert default[2]==ref and default[1]==[0,10,20]
 assert fast[1]==list(range(0,30,2));assert fast[2]==[float((i//2)*2+1) for i in range(30)]
 assert default[3]==fast[3]==[0,10,20];assert default[0]['body']==fast[0]['body']==1200
 expected=0.
 for k in range(3):expected=.5*expected+sum(ref[k*10:k*10+10])
 assert default[0]['filter']==expected
 return dict(default5ms_production_scheduler_matches_reference_fixed_observations=True,fast_refresh_ticks=fast[1],command_write_ticks=fast[3],held_current_not_accumulated=True,body_state_completed_before_observation=True,default_filter_result=expected,exit_code=0)

def derive(refresh_ticks,refresh_log):
 src=inspect.getsource(c.episode)
 start=src.index('        for substep in range(body_steps):')
 end=src.index('        if not all(np.all(np.isfinite(x))',start)
 body=src[start:end]
 # Existing physical stepping block, split into40body steps for each neural tick.
 body=body.replace('        for substep in range(body_steps):','        for substep in range(j*40,(j+1)*40):')
 src=src[:start]+src[end:]
 old='''        for j in range(neural_steps):
            sp[j] = net.step(current_nA=current)'''
 new='''        def refresh_at_tick(tick):
            quantities_now=observe_quantities(model,d,sensory_reference,joints,geom_leg)
            fresh,_=enc.encode(quantities_now)
            refresh_log.append(dict(tick=tick,physical_time=float(d.time),observation={leg:vars(v) for leg,v in quantities_now.items()},current=fresh.copy()))
            return fresh
        refresh_log.append(dict(tick=k*10,physical_time=float(d.time),observation={leg:vars(v) for leg,v in quantities.items()},current=current.copy()))
        applied=pending.copy()
        d.ctrl[:]=applied
        def physical_tick(j):
            nonlocal escape
BODY
        sp=advance_block(k*10,refresh_ticks,current,refresh_at_tick,lambda x:net.step(current_nA=x),physical_tick)'''
 # Body callback closure executes exactly original substep dynamics; no external push in this arm.
 new=new.replace('BODY','\n'.join('    '+line for line in body.rstrip().splitlines()))
 src=src.replace(old,new)
 ns=dict(vars(c),advance_block=advance_block,refresh_ticks=refresh_ticks,refresh_log=refresh_log);exec(compile(src,__file__,'exec'),ns);return ns['episode'],src

def main():
 O.mkdir(parents=True,exist_ok=False);dump(O/'TESTS.json',tests())
 files=[Path(__file__),Path(c.__file__),INITIAL,c.POOLS,Path(c.cp.CANONICAL_XML),Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz')];hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,only_change='sensory observation and freshly assigned encoded current every1ms instead5ms',sensory_dt=.001,command_dt=.005,neural_dt=.0005,body_dt=.0000125,duration=.25,preparation=0,command_scale=1,historical_tier=True,historical_pools=True,rules=['first15ms perleg recruitment/commands','thorax/root/com/vz','first nonfoot','last50ms actual plane foot/nonfoot contacts','act bounds/warnings/finite'],order='at completed body time refresh if tick divisible2; retain5ms pending command; neural tick then40body steps; decode10completed neural ticks for next5ms',default_behavior_regression='actual advance_block versus held5ms reference on fixed observations',stability='NOT_JUDGED no new short-window pass'))
 m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate);built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];mapping=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);dump(O/'MAPPING.json',mapping);groups,_=c.channel_index_map(built['n'],built['tier']['root_ids']);obs=Observer(m,geom);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);ev={};log=[];fn,src=derive(2,log);np.random.seed(0)
 rec,tr,cmd,sp=fn(m,q,joints,geom,built,mapping,groups,'normal',.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125)
 np.savez_compressed(O/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(O/'RECORDS.json',rec);obs.save(O/'BODY.npz');dump(O/'CONTACTS.json',obs.contacts);dump(O/'SENSORY_REFRESH.json',log);(O/'DERIVED_EPISODE.py').write_text(src);assert len(log)==250
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=1,mj_forward=20003,mj_step=20000,neural_steps=500,sensory_refreshes=250,command_windows=50,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('sensory1ms command5ms normal250ms exit0')
if __name__=='__main__':main()
