"""Unique sensory1ms command5ms run; production defaults untouched."""
from pathlib import Path
import os,inspect,sys,json,time,traceback
from unittest.mock import patch
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_neural_muscle_closure as c
from run_fast_sensory_only import advance_block
from run_command1ms_single import Observer,recording_decoder
from run_one_second_motor_pair import INITIAL,sha,dump
from run_motor_body_integration import require_legal_initial_pose
O=Path('/home/sensen/Desktop/engine修复验收/s4-sensory1ms-command5ms/20261027T225000Z');ROOT=Path(__file__).parent

def derive(refresh_ticks,log,currents):
 src=inspect.getsource(c.episode)
 start=src.index('        for substep in range(body_steps):');end=src.index('        if not all(np.all(np.isfinite(x))',start)
 body=src[start:end].replace('        for substep in range(body_steps):','        for substep in range(j*40,(j+1)*40):');src=src[:start]+src[end:]
 old='''        for j in range(neural_steps):
            sp[j] = net.step(current_nA=current)'''
 assert src.count(old)==1
 new='''        def remember(tick,quant,fresh,ch):
            log.append(dict(tick=tick,physical_time=float(d.time),source_body_time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),observation={leg:vars(v) for leg,v in quant.items()},channel_record=ch,current_min=float(fresh.min()),current_max=float(fresh.max())))
            currents.append(fresh.copy())
        def refresh_at_tick(tick):
            quant=observe_quantities(model,d,sensory_reference,joints,geom_leg)
            fresh,ch=enc.encode(quant)
            remember(tick,quant,fresh,ch)
            return fresh
        remember(k*10,quantities,current,channels)
        applied=pending.copy()
        d.ctrl[:]=applied
        def physical_tick(j):
            nonlocal escape
BODY
        sp=advance_block(k*10,refresh_ticks,current,refresh_at_tick,lambda x:net.step(current_nA=x),physical_tick)'''
 new=new.replace('BODY','\n'.join('    '+line for line in body.rstrip().splitlines()))
 src=src.replace(old,new);ns=dict(vars(c),advance_block=advance_block,refresh_ticks=refresh_ticks,log=log,currents=currents);exec(compile(src,__file__,'exec'),ns);return ns['episode'],src

def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),ROOT/'test_sensory1ms_preflight.py',Path(c.__file__),ROOT/'run_fast_sensory_only.py',ROOT/'run_command1ms_single.py',ROOT/'engine/embodied/motor_output.py',ROOT/'engine/embodied/sensory_channels.py',INITIAL,c.POOLS,Path(c.cp.CANONICAL_XML),ROOT/'data/flywire/banc_connectome.npz']
 hashes={str(p):sha(p) for p in files};dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,only_change='sensory refresh5ms to1ms, freshly assigned current; command5ms unchanged',duration_s=.25,body_dt=.0000125,neural_dt=.0005,sensory_dt=.001,command_dt=.005,seed=0,background=0,threshold_offset=0,external_touch=False,push=None,body_initial=str(INITIAL),expected=dict(model=1,body=20000,neural=500,commands=50,sensory=250),metrics=['firstevents/decode/actualwrite/activation','first15ms perleg','sensory current/channel firstdifference versus historical held5ms','clamps separate sensory-no-hard-clamp from native motor floor','network/motor events/saturationtime','nonfoot onset/COMheight/vz','last50 perfoot/nonfootworldforce trueplane','warnings/finite'],interpretation='not pure fixed neural delay; no standing/walking certification'))
 import unittest
 suite=unittest.defaultTestLoader.loadTestsFromName('test_sensory1ms_preflight')
 with (O/'TEST.log').open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
 if not result.wasSuccessful():raise RuntimeError('Tests failed before dynamics')
 counts=dict(model_load=0,mj_forward=0,mj_step=0,neural_steps=0,decoder_consume=0);rf=mujoco.mj_forward;rs=mujoco.mj_step
 def forward(*a,**kw):counts['mj_forward']+=1;return rf(*a,**kw)
 def step(*a,**kw):counts['mj_step']+=1;return rs(*a,**kw)
 obs=None;error=None;ticksp=[];tickcur=[];log=[];currents=[];start=time.time()
 try:
  with patch.object(mujoco,'mj_forward',forward),patch.object(mujoco,'mj_step',step):
   m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);counts['model_load']+=1;require_legal_initial_pose(gate);c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate)
   built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];mapping=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);groups,_=c.channel_index_map(built['n'],built['tier']['root_ids']);dump(O/'MAPPING.json',mapping)
   decoder,source=recording_decoder(mapping);consume0=decoder.consume
   def consume(*a,**kw):counts['decoder_consume']+=1;return consume0(*a,**kw)
   decoder.consume=consume
   nc=c.ConductanceNetwork
   class Net(nc):
    def step(self,*a,**kw):
     tickcur.append(kw['current_nA'].copy());sp=super().step(*a,**kw);ticksp.append(sp.copy());counts['neural_steps']+=1;return sp
   obs=Observer(m,geom);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);ev={};p=O/'normal';p.mkdir();np.random.seed(0)
   with patch.object(c,'make_motor_decoder',return_value=decoder),patch.object(c,'ConductanceNetwork',Net):
    fn,src=derive(2,log,currents);(O/'DERIVED_EPISODE.py').write_text(src);(O/'INSTRUMENTED_CONSUME.py').write_text(source)
    rec,tr,cmd,sp=fn(m,q,joints,geom,built,mapping,groups,'normal',.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125)
   np.savez_compressed(p/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)})
   rates=np.asarray(decoder.tick_rates);np.savez_compressed(p/'TICKS.npz',spikes=ticksp,rates_hz=rates,decoded_controls=np.clip(rates/decoder.config.rate_for_full_control_hz,0,decoder.config.max_control),tick_complete_s=np.arange(1,501)*.0005,current_nA=tickcur)
   np.savez_compressed(p/'SENSORY.npz',current_nA=currents,physical_time=[r['physical_time'] for r in log],qpos=[r['qpos'] for r in log],qvel=[r['qvel'] for r in log]);dump(p/'SENSORY_REFRESH.json',log);dump(p/'RECORDS.json',rec)
   assert counts['neural_steps']==500 and counts['mj_step']==20000 and counts['decoder_consume']==50 and len(log)==250 and len(rates)==500
 except Exception as ex:error=dict(type=type(ex).__name__,message=str(ex),traceback=traceback.format_exc())
 finally:
  if obs is not None:obs.save(O/'normal/BODY.npz');dump(O/'normal/CONTACTS.json',obs.contacts)
  dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0 if error is None else 1,counts=counts,error=error,runtime_s=time.time()-start))
 if error:raise RuntimeError(error)
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'SOURCE_UNCHANGED.json',dict(passed=True));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('UNIQUE sensory1ms command5ms250ms COMPLETE',counts,flush=True)
if __name__=='__main__':main()
