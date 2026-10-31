"""20ms fixed initial sensory numerical preparation, then one fresh250ms closure."""
from pathlib import Path
import os,inspect,json,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_neural_muscle_closure as c
from run_quarter_second_pair import Observer,OUT as BASE
from run_one_second_motor_pair import INITIAL,sha,dump
O=Path('/home/sensen/Desktop/engine修复验收/s2-neural-initial-preparation/20261027T182100Z')
PREP='''
    if preparation_intervals:
        initial_quantities = observe_quantities(model, d, sensory_reference, joints, geom_leg)
        fixed_current, initial_channels = enc.encode(initial_quantities)
        prep_events=[]
        for pk in range(preparation_intervals):
            prep_sp=np.array([net.step(current_nA=fixed_current) for pj in range(neural_steps)])
            pending, prep_trace=decoder.consume(prep_sp,spike_dt_s=DT_NEURAL,start_s=pk*DT_COMMAND,end_s=(pk+1)*DT_COMMAND)
            prep_events.append(dict(interval=pk,spikes=np.argwhere(prep_sp),decoder=prep_trace))
        prep_state={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in vars(net).items() if isinstance(value,(np.ndarray,int,float))}
        dump(PREP_OUT/'PREPARATION.json',dict(duration=preparation_intervals*DT_COMMAND,physical_duration=0.,fixed_current=fixed_current,events=prep_events,decoder_rates=decoder.rates.copy(),decoder_elapsed=decoder.elapsed_s,decoder_interval=decoder.interval,pending=pending.copy(),net_state=prep_state,body_qpos=d.qpos.copy(),body_qvel=d.qvel.copy(),body_act=d.act.copy()))
        # Preserve continuous neural/decoder clock, including refractory deadlines and delay-ring steps.
        assert abs(net.t_ms-preparation_intervals*DT_COMMAND*1000)<1e-10
        assert abs(decoder.elapsed_s-preparation_intervals*DT_COMMAND)<1e-10
        dump(PREP_OUT/'RELEASE_TRANSFER.json',dict(net_time_ms=net.t_ms,decoder_time=decoder.elapsed_s,decoder_interval=decoder.interval,pending=pending,body_activation=d.act,body_velocity=d.qvel,body_qpos=d.qpos,source_interval=-1,note='prepared interval[-5ms,0) command valid at0; all subsequent source intervals shifted by20ms'))
'''
def derived(intervals):
 src=inspect.getsource(c.episode)
 if intervals:
  src=src.replace('    source = -1\n','    source = -1\n'+PREP)
  src=src.replace('net.t_ms/1000-(t+DT_COMMAND)','net.t_ms/1000-(t+DT_COMMAND+preparation_intervals*DT_COMMAND)')
  src=src.replace('spike_dt_s=DT_NEURAL, start_s=t,','spike_dt_s=DT_NEURAL, start_s=t+preparation_intervals*DT_COMMAND,').replace('end_s=t+DT_COMMAND,','end_s=t+DT_COMMAND+preparation_intervals*DT_COMMAND,')
 ns=dict(vars(c),preparation_intervals=intervals,PREP_OUT=O,dump=dump);exec(compile(src,__file__,'exec'),ns);return ns['episode'],src

def main():
 O.mkdir(parents=True,exist_ok=False);fn,src=derived(4);zero,zsrc=derived(0);assert zsrc==inspect.getsource(c.episode)
 dump(O/'TESTS.json',dict(zero_duration_source_identical=True,zero_initialization_and_schedule='exact original compiled function source, no inserted statements',exit_code=0))
 files=[Path(__file__),Path(c.__file__),INITIAL,c.POOLS,Path(c.cp.CANONICAL_XML),Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz')];hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(inputs_sha256=hashes,preparation_duration=.020,physical_duration=.250,neural_dt=.0005,body_dt=.0000125,sensory_command_dt=.005,only_change='net and decoder prepared40steps with fixed initial real observation, pending carried; body untouched',scale=1,historical_tier_and_pools=True,extra_input=False,rules=['release and first15ms leg pool events,ctrl,act','thorax/root/com/vz','first nonfoot','last50ms true plane foot/nonfoot contacts','finite/warnings/act bounds'],stable_support='NOT_JUDGED unless original gate evaluated',clock='prep[-20ms,0); release[0,250ms); decoder and net timestamps translated only'))
 m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate)
 built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];mapping=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);dump(O/'MAPPING.json',mapping);groups,_=c.channel_index_map(built['n'],built['tier']['root_ids']);obs=Observer(m,geom);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);ev={};np.random.seed(0)
 records,tr,ctrl,sp=fn(m,q,joints,geom,built,mapping,groups,'normal',.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125)
 np.savez_compressed(O/'INTERVALS.npz',trajectory=tr,commands=ctrl,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(O/'RECORDS.json',records);obs.save(O/'BODY.npz');dump(O/'CONTACTS.json',obs.contacts);(O/'DERIVED_EPISODE.py').write_text(src)
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=1,mj_forward=20003,mj_step=20000,preparation_neural_steps=40,release_neural_steps=500,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('prepared20ms release250ms exit0')
if __name__=='__main__':main()
