"""S164 bounded neural-only pair; exact frozen methods, passive tracing, no body imports."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, shutil, sys, types
from datetime import datetime, timezone
import numpy as np
from scipy import sparse
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
A=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
B=Path('/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z')
F=B/'BOUND_INPUT_FREEZE'
ARRAYS=('v','ge','gi','adaptation','refractory_until_ms','spike_count','_ring')
SCALARS=('n','dt','delay_steps','_ring_i','t_ms','steps','voltage_method')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def module(path,name,remove_relative=False,bindings=None):
 m=types.ModuleType(name);sys.modules[name]=m;m.__file__=str(path)
 tree=ast.parse(path.read_text(),filename=str(path))
 if remove_relative:tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or not n.level]
 if bindings:m.__dict__.update(bindings)
 exec(compile(tree,str(path),'exec'),m.__dict__);return m

def prepare(root,create):
 sources={'neural_cond.py':F/'0049_neural_cond.py','motor_output.py':F/'0103_motor_output.py','motor_translation.py':F/'0104_motor_translation.py','MANIFEST.json':F/'0003_MANIFEST.json','INITIALIZATION.npz':A/'init80ms/INITIALIZATION.npz','NEURAL_RELEASE_META.json':A/'init80ms/NEURAL_RELEASE_META.json','CONNECTIVITY.npz':A/'init80ms/CONNECTIVITY.npz','S154_TRAJECTORY.npz':A/'init80ms/TRAJECTORY.npz','S162_TRAJECTORY.npz':B/'init80ms/TRAJECTORY.npz','ENCODER_PARAMETERS.json':B/'init80ms/ENCODER_PARAMETERS.json','OBSERVATION_00.json':B/'init80ms/OBSERVATION_00.json','OBSERVATION_01.json':B/'init80ms/OBSERVATION_01.json','S154_INITIAL_OBSERVATION.json':A/'init80ms/INITIAL_OBSERVATION.json','producer.py':Path(__file__)}
 if create:
  root.mkdir(parents=True,exist_ok=False);(root/'BOUND_INPUT_FREEZE').mkdir()
  for name,p in sources.items():shutil.copyfile(p,root/'BOUND_INPUT_FREEZE'/name)
  js(root/'BOUND_INPUT_FREEZE.json',{name:{'source':str(p),'source_sha256':sha(p),'frozen_sha256':sha(root/'BOUND_INPUT_FREEZE'/name)} for name,p in sources.items()})
 f=root/'BOUND_INPUT_FREEZE';binding=json.loads((root/'BOUND_INPUT_FREEZE.json').read_text())
 for name,r in binding.items():assert sha(f/name)==r['frozen_sha256']==r['source_sha256'] and sha(r['source'])==r['source_sha256'],name
 # Current decoder classes must match archived frozen full-file identity; imports never touch engine package.
 for name in ('motor_output.py','motor_translation.py'):
  assert sha(PROJECT/'engine/embodied'/name)==sha(f/name),name+' current/frozen mismatch'
 z=load(f/'INITIALIZATION.npz');meta=json.loads((f/'NEURAL_RELEASE_META.json').read_text());sc=meta['neural_scalars'];p=meta['neural_configs']['p']
 assert all(k in z for k in ARRAYS+('history','weights','current'))
 assert all(k in sc for k in SCALARS)
 assert sc['n']==17099 and sc['dt']==.5 and sc['steps']==160 and sc['t_ms']==80. and sc['voltage_method']=='backward_euler'
 assert z['history'].shape==(80,66) and z['_ring'].shape==(5,17099)
 assert len(z['spikes'])==160 and z['features'].shape==(160,66,5)
 nn=module(f/'neural_cond.py','s164_neural');mo=module(f/'motor_output.py','s164_motor_output');de=module(f/'motor_translation.py','s164_decoder',True,{'MotorOutput':mo.MotorOutput})
 conn=load(f/'CONNECTIVITY.npz');matrices=[]
 for name in ('We','Wi'):
  matrices.append(sparse.csr_matrix((conn[name+'_data'],conn[name+'_indices'],conn[name+'_indptr']),shape=tuple(conn[name+'_shape'])))
 manifest=json.loads((f/'MANIFEST.json').read_text());nets=[];decs=[]
 for arm in range(2):
  net=nn.ConductanceNetwork(sc['n'],nn.ConductanceParams(**p),dt_ms=sc['dt'],voltage_method=sc['voltage_method']);net.set_connectivity(*matrices)
  for k in ARRAYS:setattr(net,k,z[k].copy())
  for k in SCALARS:setattr(net,k,sc[k])
  for name,original in zip(('We','Wi'),matrices):
   actual=getattr(net,name)
   for k in ('data','indices','indptr'):assert np.array_equal(getattr(actual,k),getattr(original,k))
  dec=de.FiniteMotorTranslator(manifest,z['weights']);dec.history=z['history'].copy();dec.elapsed_s=sc['t_ms']/1000;dec.interval=sc['steps']-1
  assert np.array_equal(dec.weights,z['weights']);assert dec.history.shape==(80,len(dec.pools))
  nets.append(net);decs.append(dec)
 ta=load(f/'S154_TRAJECTORY.npz');tb=load(f/'S162_TRAJECTORY.npz');cfg=json.loads((f/'ENCODER_PARAMETERS.json').read_text());currents=[]
 for tick in (0,1):
  raw=json.loads((f/f'OBSERVATION_{tick:02d}.json').read_text());c=cfg['config'];cur=np.zeros(cfg['n_neurons'])
  for leg,q in raw.items():
   vals={'claw_position':q['tibia_joint_dev_rad']/c['reference_angle_rad'],'hook_direction':q['tibia_joint_vel_rad_s']/c['reference_velocity_rad_s'],'hairplate_angle':q['multi_joint_dev_rad']/c['reference_angle_rad'],'campaniform_strain':q['contact_force_norm']/c['reference_force'],'bristle_touch':q['foot_contact'],'club_vibration':0.,'multidendritic_noxious':0.}
   for ch,value in vals.items():cur[np.asarray(cfg['groups'][leg][ch],dtype=int)]+=c['current_nA_per_unit']*value
  assert np.array_equal(cur,tb['current'][tick]),'independent current not exact'
  currents.append(cur)
 assert np.array_equal(currents[0],z['current']) and np.array_equal(currents[0],ta['current'][0])
 assert json.loads((f/'OBSERVATION_00.json').read_text())==json.loads((f/'S154_INITIAL_OBSERVATION.json').read_text())
 for k in ARRAYS:assert np.array_equal(getattr(nets[0],k),getattr(nets[1],k))
 for k in SCALARS:assert getattr(nets[0],k)==getattr(nets[1],k)
 return z,nets,decs,currents,ta

def snapshot(net,dec):
 return {**{k:getattr(net,k).copy() for k in ARRAYS},**{k:np.asarray(getattr(net,k)) for k in SCALARS},'history':dec.history.copy(),'decoder_elapsed_s':np.asarray(dec.elapsed_s),'decoder_interval':np.asarray(dec.interval),'weights':dec.weights,'actuator_pool':dec.actuator_pool.copy()}

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--root',type=Path);pa.add_argument('--execute',action='store_true');args=pa.parse_args()
 root=args.root or Path('/home/sensen/Desktop/engine修复验收/s164-held-sensory-neural')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 create=not root.exists();z,nets,decs,currents,ta=prepare(root,create)
 schema={'scope':'S164/P1 neural-only held-current open-loop pair, not closed-loop/body/behavior evidence','root':str(root),'passed':True,'new_neural_steps':0,'restore_arrays':list(ARRAYS),'restore_scalars':list(SCALARS),'restored_decoder':'history from INITIALIZATION; elapsed_s=.08 and interval=159 from archived actual PRE160 consume chronology','steps_per_arm':41,'total_authorized_steps':82,'schedule':'shared tick 80->80.5ms using initial S154 current then A 40 held initial ticks; B 40 held S162 tick1 current (body .5ms sample); final 100.5ms','state_schema':'A/B_STATES.npz pre_/post_ for all restore arrays/scalars; input,arrivals,active,candidate_v_BE,v_before_threshold,ge_at_threshold,gi_at_threshold,spikes,reset,history,features,unclipped_pool,clipped_pool,decoded,applied; first axis 41','observer':'unchanged actual frozen step called once/tick; sys.settrace passive observer at original spiked assignment captures locals and candidate rhs/(C/dt+G)','forbidden_new_calls':{'body_imports':0,'model_load':0,'forward':0,'physics_step':0,'PRE_rerun':0,'extra_neural_steps':0}}
 if create:
  js(root/'PREREG.json',schema);js(root/'PREFLIGHT.json',schema)
  for label,net,dec in zip(('A','B'),nets,decs):np.savez_compressed(root/f'{label}_INITIAL.npz',**snapshot(net,dec))
 print(json.dumps(schema),flush=True)
 if not args.execute:return
 assert not (root/'RUN_STATUS.json').exists(),'execution already attempted; never rerun'
 counts={'A':0,'B':0};status={'status':'RUNNING','real_neural_steps':0,'per_arm':counts,'model_load':0,'forward':0,'physics_step':0,'PRE_rerun':0};js(root/'RUN_STATUS.json',status)
 try:
  results={label:{} for label in ('A','B')};code=nets[0].step.__func__.__code__
  source=(root/'BOUND_INPUT_FREEZE/neural_cond.py').read_text().splitlines();threshold_line=next(i+1 for i,s in enumerate(source) if 'spiked = active &' in s)
  for tick in range(41):
   for label,net,dec in zip(('A','B'),nets,decs):
    row={};before=snapshot(net,dec)
    for k in ARRAYS+SCALARS:row['pre_'+k]=before[k]
    drive=currents[1] if label=='B' and tick>=1 else currents[0];row['input']=drive.copy()
    observed={}
    def trace(frame,event,arg):
     if frame.f_code is code and event=='line' and frame.f_lineno==threshold_line:
      d=frame.f_locals;observed.update(arrivals=d['arrivals'].copy(),active=d['active'].copy(),candidate_v_BE=(d['rhs']/(net.p.capacitance_nF/net.dt+d['G'])).copy(),v_before_threshold=net.v.copy(),ge_at_threshold=net.ge.copy(),gi_at_threshold=net.gi.copy())
     return trace
    sys.settrace(trace)
    try:sp=net.step(current_nA=drive).copy()
    finally:sys.settrace(None)
    counts[label]+=1;status['real_neural_steps']=sum(counts.values());js(root/'RUN_STATUS.json',status)
    assert observed
    row.update(observed);row['spikes']=sp;row['reset']=sp.copy()
    start=dec.elapsed_s;u,info=dec.consume(sp[None,:],spike_dt_s=.0005,start_s=start,end_s=net.t_ms/1000)
    row.update(history=dec.history.copy(),features=np.asarray(info['pool_features']),decoded=u.copy(),applied=.5*u)
    row['unclipped_pool']=(row['features']*dec.weights).sum(1);row['clipped_pool']=np.clip(row['unclipped_pool'],0,1)
    after=snapshot(net,dec)
    for k in ARRAYS+SCALARS:row['post_'+k]=after[k]
    row['decoder_elapsed_s']=np.asarray(dec.elapsed_s);row['decoder_interval']=np.asarray(dec.interval)
    for k,v in row.items():results[label].setdefault(k,[]).append(np.asarray(v).copy())
    if tick==0:np.savez_compressed(root/f'{label}_POSTFIRST.npz',**after)
   if tick==0:
    for k in results['A']:assert np.array_equal(results['A'][k][0],results['B'][k][0]),'first tick pair '+k
    for k in ('spikes','features','decoded'):assert np.array_equal(results['A'][k][0],ta[k][0]),'S154 release0 '+k
    js(root/'SHARED_FIRST_CHECK.json',{'passed':True,'full_pair_state_and_observer_equal':True,'S154_release0_spikes_features_decoded_exact':True,'S154_release0_full_state':'not archived; not claimed'})
  for label in ('A','B'):np.savez_compressed(root/f'{label}_STATES.npz',**{k:np.stack(v) for k,v in results[label].items()})
  assert counts=={'A':41,'B':41}
  for name,r in json.loads((root/'BOUND_INPUT_FREEZE.json').read_text()).items():assert sha(root/'BOUND_INPUT_FREEZE'/name)==r['frozen_sha256'] and sha(r['source'])==r['source_sha256']
  assert not any(k=='mujoco' or k.startswith('engine') for k in sys.modules),'forbidden package import'
  a,b=results['A'],results['B'];summary={'scope':'held neural currents, no body feedback after initial sample','real_neural_steps':82,'per_arm':counts,'final_neural_ms':nets[0].t_ms,'relative_end_ms':nets[0].t_ms-80,'spike_different_neurons_per_tick':[int(np.count_nonzero(x!=y)) for x,y in zip(a['spikes'],b['spikes'])],'voltage_maxabs_per_tick':[float(np.max(abs(x-y))) for x,y in zip(a['post_v'],b['post_v'])],'decoded_maxabs_per_tick':[float(np.max(abs(x-y))) for x,y in zip(a['decoded'],b['decoded'])],'input_changed_cells':int(np.count_nonzero(currents[0]!=currents[1]))}
  js(root/'RESULTS.json',summary);status.update(status='COMPLETE');js(root/'RUN_STATUS.json',status);js(root/'COMPLETE.json',summary)
  js(root/'PRODUCER_OUTPUT_HASHES.json',{str(p):sha(p) for p in root.iterdir() if p.is_file() and p.name!='PRODUCER_OUTPUT_HASHES.json'})
  print(json.dumps(summary),flush=True)
 except BaseException as e:
  status.update(status='FAILED',error=repr(e));js(root/'RUN_STATUS.json',status);raise
if __name__=='__main__':main()
