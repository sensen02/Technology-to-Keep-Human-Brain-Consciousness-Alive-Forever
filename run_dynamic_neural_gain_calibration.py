"""Bounded true-neural closed-loop preclip-gain calibration; no surrogate targets."""
from pathlib import Path
import os,json,sys,time,traceback
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
from dataclasses import replace,asdict
from unittest.mock import patch
import run_neural_muscle_closure as c
from engine.embodied.motor_output import MotorOutput,MotorOutputConfig
from engine.embodied.runtime_pool_guard import load_runtime_pool
from run_command1ms_single import Observer
from run_one_second_motor_pair import INITIAL,sha,dump
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s4-dynamic-neural-gain/20261028T011000Z')
H=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z')
GROUPS=['coxa_label','trochanter_flex_label','trochanter_ext_label','tibia_flex_label','tibia_ext_label']
def label(name):
 if 'Tibia_flex' in name:return 3
 if 'Tibia_ext' in name:return 4
 if name[2]=='C':return 0
 if 'flexor' in name:return 1
 if 'extensor' in name:return 2
 raise ValueError(name)
def metrics(b):
 t=b['time'];last=t>t[-1]-.05+1e-9;z=b['com_position'][:,2];h0=z[0];idx=np.flatnonzero(b['nonfoot_floor_count']>0);nf=float((b['nonfoot_floor_count'][last]>0).mean());first=float(t[idx[0]]) if len(idx) else float(t[-1]);shared=float(np.trapezoid(z,t)/(t[-1]*h0)+first/t[-1]+z[last].mean()/h0-nf)
 r=dict(first_nonfoot_s=None if not len(idx) else first,last50_nonfoot_fraction=nf,last50_mean_com_height=float(z[last].mean()),final_com_height=float(z[-1]),final_com_vz=float(b['com_velocity'][-1,2]),mean_com_height=float(np.trapezoid(z,t)/t[-1]),shared_score=shared,minimum_com_height=float(z.min()))
 if 'ground_force_foot_world' in b:
  f=float(b['ground_force_foot_world'][last,:,2].sum(1).mean());n=float(b['ground_force_nonfoot_world'][last,2].mean());fraction=f/(f+n) if f+n>1e-9 else 0.;r.update(last50_foot_fz=f,last50_nonfoot_fz=n,last50_foot_fraction=fraction,score=shared+fraction)
 return r

def main():
 O.mkdir(parents=True,exist_ok=False)
 rng=np.random.default_rng(0);points=rng.uniform(-np.log(4),np.log(4),(6,5));schedule=np.exp(np.stack([points,-points],axis=1).reshape(12,5))
 files=[Path(__file__),Path(c.__file__),P/'engine/embodied/motor_output.py',P/'engine/neural_cond.py',P/'engine/embodied/sensory_channels.py',P/'run_command1ms_single.py',P/'run_quarter_second_pair.py',INITIAL,Path(c.cp.CANONICAL_XML),c.POOLS,P/'data/flywire/banc_connectome.npz',H/'normal/BODY.npz']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,groups=GROUPS,group_rule='actual actuator labels, not validated anatomical functional classes; identical local pool sets must share gain and class or stop',schedule=schedule,seed=0,gain_bounds=[.25,4.],decoder='production MotorOutput replace(per_actuator_gain=tuple(g)); PRECLIP rates*gain then originalclip; nobias',defaults=asdict(MotorOutputConfig()),duration_s=.25,dt_body_s=.0000125,dt_neural_s=.0005,dt_sensory_s=.005,dt_command_s=.005,objective='shared=integral COMheight/(duration*h0)+firstNONFOOTtime/duration+last50meanCOM/h0-last50NONFOOTfraction; score=shared+last50 FOOTFz/(FOOTFz+NONFOOTFz). Baseline onlysharedscore, no imputed partition.',search='12 fixed antithetic loguniform5D candidates, all before dynamics, no outcome-adaptive retuning',validation_gate='candidate last50 NONFOOTfraction<.5 AND last50 meanCOM>historical+.1 AND last50 FOOTforcefraction>.8; select maxscore then exact250msrepeat and500msextend',budget=dict(training=12,validation=2,max_body_steps=300000),prior_work='Prior S233/S237/S253/S254/S256 already exercised gain calibration in older branches; this is not first gain-factor experiment. This run fixes canonical65d model,12.5us,rest tier/no FIR targets and new preregistered dynamic5D design.',no=['externalcontrol','newnodes','activationpreload','modelchanges','initialposechanges','CPG','PD','bias','baseline rerun','staticfit targets'],interpretation='gain calibration engineering hypothesis; no candidate score certifies autonomous standing or physiological calibration'))
 counts=dict(model_load=0,forward=0,step=0,neural_step=0,network_instances=0,rollouts=0);rf=mujoco.mj_forward;rs=mujoco.mj_step;rn=c.ConductanceNetwork.step;init=c.ConductanceNetwork.__init__;error=None;results={};start=time.time()
 def fw(*a,**kw):counts['forward']+=1;return rf(*a,**kw)
 def st(*a,**kw):counts['step']+=1;return rs(*a,**kw)
 def ns(*a,**kw):counts['neural_step']+=1;return rn(*a,**kw)
 def ni(*a,**kw):counts['network_instances']+=1;return init(*a,**kw)
 try:
  with patch.object(mujoco,'mj_forward',fw),patch.object(mujoco,'mj_step',st),patch.object(c.ConductanceNetwork,'step',ns),patch.object(c.ConductanceNetwork,'__init__',ni):
   m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);counts['model_load']+=1;c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate)
   built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];manifest=c.build_motor_manifest(load_runtime_pool(c.POOLS),built['tier']['root_ids'],names);groups,_=c.channel_index_map(int(built['n']),built['tier']['root_ids']);dump(O/'MAPPING.json',manifest)
   classes=np.array([label(n) for n in names]);ties={};unresolved=[]
   for r in manifest['rows']:
    if not r['local_indices']:unresolved.append(r['actuator_index']);continue
    key=tuple(sorted(r['local_indices']));ties.setdefault(key,[]).append(r['actuator_index'])
   assert len(unresolved)==16 and sum(map(len,ties.values()))==74
   for ids in ties.values():assert len(set(classes[ids]))==1,'pool crosses label classes'
   dump(O/'GROUP_ASSIGNMENTS.json',dict(groups=GROUPS,rows=[dict(name=n,label=GROUPS[classes[j]],mapped=j not in unresolved) for j,n in enumerate(names)],shared_variable_count=len(ties),unresolved=unresolved))
   # Production decoder checks with actual network size and nonzero synthetic events; no neural integration.
   a=MotorOutput(manifest);b=MotorOutput(manifest,replace(MotorOutputConfig(),per_actuator_gain=tuple(np.ones(90))));zero=MotorOutput(manifest,replace(MotorOutputConfig(),per_actuator_gain=tuple(schedule[0][classes])))
   sp=np.zeros((10,built['n']),bool);sp[0,:]=1;sp[7,::3]=1
   for k in range(3):
    x,tx=a.consume(sp,spike_dt_s=.0005,start_s=k*.005,end_s=(k+1)*.005);y,ty=b.consume(sp,spike_dt_s=.0005,start_s=k*.005,end_s=(k+1)*.005);assert np.array_equal(x,y) and np.array_equal(a.rates,b.rates) and tx==ty
   zz,_=zero.consume(np.zeros_like(sp),spike_dt_s=.0005,start_s=0.,end_s=.005);assert not zz.any()
   gain=schedule[0][classes];test=MotorOutput(manifest,replace(MotorOutputConfig(),per_actuator_gain=tuple(gain)));u,_=test.consume(sp,spike_dt_s=.0005,start_s=0.,end_s=.005);assert np.array_equal(u,np.clip(test.rates/100*gain,0,1));assert not u[unresolved].any()
   dump(O/'DECODER_CHECKS.json',dict(identity_gain1_exact=True,zero_input_zero_output=True,actual_preclip_formula_exact=True,ties_exact=True,unresolved_zero=True,synthetic_only=True,exit_code=0))
   baseline=metrics(np.load(H/'normal/BODY.npz'));dump(O/'BASELINE_METRICS.json',baseline)
   def run(tag,g5,duration):
    folder=O/tag;folder.mkdir();gains=g5[classes];dump(folder/'CONFIG.json',dict(group_gains=g5,per_actuator_gain=gains,duration_s=duration,source_hashes=hashes,seed=0));np.random.seed(0);observer=Observer(m,geom);d=mujoco_data(m,q);observer(m,d);ev={};err=None;before=counts.copy();counts['rollouts']+=1
    factory=lambda manifest,motor_translator=None:MotorOutput(manifest,replace(MotorOutputConfig(),per_actuator_gain=tuple(gains)))
    try:
     with patch.object(c,'make_motor_decoder',factory):records,tr,cmd,spikes=c.episode(m,q,joints,geom,built,manifest,groups,'normal',duration,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=observer,body_dt_s=.0000125)
     np.savez_compressed(folder/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=spikes,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(folder/'RECORDS.json',records)
    except Exception as e:err=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    finally:observer.save(folder/'BODY.npz');dump(folder/'CONTACTS.json',observer.contacts)
    bdy={k:np.asarray([row[k] for row in observer.rows]) for k in observer.rows[0]}
    met=metrics(bdy);assert np.array_equal(bdy['qpos'][0],q) and not bdy['qvel'][0].any() and not bdy['activation'][0].any();assert not bdy['external_force'].any();assert np.isfinite(bdy['qpos']).all() and not bdy['warnings'].any();assert not bdy['command'][:,unresolved].any()
    for ids in ties.values():assert np.array_equal(bdy['command'][:,ids],np.repeat(bdy['command'][:,ids[:1]],len(ids),axis=1))
    met.update(group_gains=g5,validation_gate=bool(met['last50_nonfoot_fraction']<.5 and met['last50_mean_com_height']>baseline['last50_mean_com_height']+.1 and met['last50_foot_fraction']>.8),error=err,calls={k:counts[k]-before[k] for k in ['forward','step','neural_step','network_instances']});dump(folder/'METRICS.json',met);results[tag]=met;dump(O/'SUMMARY.json',results);print(tag,json.dumps(met,default=lambda x:x.tolist()),flush=True)
    if err:raise RuntimeError(err)
    assert counts['step']<=300000
   for i,g5 in enumerate(schedule):run(f'train_{i:02d}',g5,.25)
   qualified=[k for k,v in results.items() if v['validation_gate']]
   if qualified:
    best=max(qualified,key=lambda k:results[k]['score']);g5=np.asarray(results[best]['group_gains']);run('validation_repeat',g5,.25);run('validation_extend',g5,.5)
   dump(O/'DECISION.json',dict(best_training=max((k for k in results if k.startswith('train_')),key=lambda k:results[k]['score']),qualified_training=qualified,validation_run=bool(qualified),baseline_shared_score=baseline['shared_score'],no_baseline_fullscore=True,standing_certified=False))
 except Exception as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
 dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0 if error is None else 1,counts=counts,error=error,runtime_s=time.time()-start));dump(O/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(k))==v for k,v in hashes.items())));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('OUTPUT',O,flush=True)
 if error:raise RuntimeError(error)
def mujoco_data(m,q):
 d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);return d
if __name__=='__main__':main()
