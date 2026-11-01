"""One explicit-contract root888 physical run; reuse historical baseline, never replay it."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import sys, json, hashlib, time, traceback, argparse
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLD = Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
NATIVE = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-native-recording/20261101T191552Z')
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
MODEL_SHA = '3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
DT = .0000125
LEGS = ['LF','LM','LH','RF','RM','RH']
CONFIG = dict(duration=1.3, kp=100., clip=50., dt=DT, neural_duration=.7,
              neural_command_dt_s=.001, neural_prewarm_s=1.1, swing_height=.45, sample_dt_s=.0005)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1048576),b''):h.update(part)
 return h.hexdigest()
def dump(path, value):
 def convert(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  raise TypeError(type(x).__name__)
 Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n')
def prepare():
 OUT.mkdir(parents=True, exist_ok=True)
 p=OUT/'PREREGISTRATION.json'
 spec=dict(task='S1/S3/S4/S5 one new real-time root888 physical feedback trajectory',runner_config=CONFIG,
  baseline=str(OLD),native_baseline=str(NATIVE),expected_model_sha256=MODEL_SHA,
  change='Only admitted root888 sensory recipient identities and MN mapping, effective throughout neural prewarm. Original tier/config/weights*.2/offset/controls and live forward/save cadence preserved.',
  admission='No model/tier/network load until explicit final parent contract supplied. Guard wrapper delegates original(path,version_contract=dict); explicit lookup_path wrapper handles bound default.',
  source='Freeze after parent guard/contractcheck completed; compare historical sources and list approved admission-only differences, never rewrite historical hashes.',
  schedule='0-.2s no neural runtime; .2-1.3s algorithm-assisted online neural prewarm; 1.3-2s neural only. Own body/network/rates/pending retained; no boundary reset.',
  observer='Every physical step: copy full MjData then forward only clone, signed foot/nonfoot forces and raw candidates. At original .5ms shared frames plus initial and release-after only: two-body contact-point Jacobians. No extra live forwards.',
  boundaries='step104000 kind0 is assisted endpoint; same time kind1 is release-after. Initial kind2. Release endpoints kind0 step104001..160000 are 56000 equal bins; common .5ms after-boundary through 2s are 1401 frames, 1400 endpoint bins.',
  endpoints='Contact episodes are consecutive sampled Booleans; first/last sample span and endpoint censoring explicit; flips never counted as walking steps. No >=4-foot or 60ms gate.',
  evaluation=['first nonfoot time relative release','nonfoot release endpoint steps and occupancy','COM trajectory/height and root displacement','signed total foot Fz/W, per-foot forces and friction','contact candidate/nonpositive-gap/positive-Fz events','material-point relative tangential speed on declared .5ms frames','all release assistance/root forces/external/mocap and rewrites absent'],
  failure='Complete fixed 2s even if behavior fails; stop on first nonfinite state or MuJoCo warning, preserve first failure. No scans/duration extension.',
  interpretation='Historical fixed-body/current replay was not this real-time feedback experiment. New sensory input changes neural and subsequent body trajectories. Commands are descriptive, not weight bearing or walking certification. Even favorable short result lacks longer runs, causal controls and same-run video.',
  resources='Expected 1 network/3600 neural/160000 body steps and original 2 model builds. Count actual observer forward/contactForce/Jac separately; no baseline rerun/render.',
  retained_importance={'runner/preregistration/arrays/source/verification':'核心必需','report/manifest/inventory':'重要辅助'},cleanup='No known dispensable outputs at preparation; no frozen files modified.')
 if p.exists():
  assert json.loads(p.read_text())==spec,'prepared preregistration changed'
 else:dump(p,spec)
 print('PREPARED_NO_MODEL_OR_NETWORK_LOAD',str(p),flush=True)
def run(contract_path, pool_path, sensory_path):
 prepare()
 if (OUT/'EXECUTION.json').exists() or (OUT/'RUN').exists():raise RuntimeError('refuse rerun/overwrite existing experiment')
 contract_path=Path(contract_path).resolve();pool_path=Path(pool_path).resolve();sensory_path=Path(sensory_path).resolve()
 contract=json.loads(contract_path.read_text())
 # Delivery hardening: validate both CLI selections before any physical model load.
 from engine.embodied.runtime_pool_guard import load_runtime_pool as preflight_guard
 if sha(sensory_path)!=contract['sensory_lookup_sha256']:
  raise ValueError('CLI sensory path does not match explicit version contract')
 preflight_guard(pool_path,version_contract=contract)
 import mujoco
 from scipy.sparse import save_npz,load_npz
 import run_physical_walk_prewarm as runner
 import run_neural_muscle_closure as closure
 import engine.embodied.runtime_pool_guard as guard
 import engine.embodied.continuous_handover as handover
 from engine.neural_cond import ConductanceNetwork
 old=json.loads((OLD/'PREREGISTRATION.json').read_text())
 paths=set(Path(p) for p in old['source_hashes'])
 paths.update(ROOT/p for p in ['run_neural_muscle_closure.py','check_pool_version_contract.py','engine/embodied/closure_clock.py','engine/embodied/sensory_channels.py','engine/embodied/motor_output.py','engine/neural_cond.py','engine/embodied/continuous_handover.py','engine/neural_tier.py'])
 paths={p for p in paths if p.exists()}
 paths.update(Path(mod.__file__).resolve() for mod in list(sys.modules.values()) if getattr(mod,'__file__',None) and str(Path(mod.__file__).resolve()).startswith(str(ROOT)+'/') and str(mod.__file__).endswith('.py'))
 paths.update([Path(__file__),ROOT/'tools/verify_banc888_physical_handover.py',Path(closure.POOLS),Path(closure.LOOKUP),contract_path,pool_path,sensory_path,OLD/'PREREGISTRATION.json',OLD/'MODEL.xml',OLD/'BODY.npz',OLD/'NEURAL.npz',OLD/'GAIT.npz',NATIVE/'MANIFEST_SHA256.json',NATIVE/'IDENTITIES.npz',NATIVE/'ACTUAL_We.npz',NATIVE/'ACTUAL_Wi.npz'])
 hashes={str(p):sha(p) for p in sorted(paths)}
 changes={p:dict(old_sha256=h,new_sha256=hashes[p]) for p,h in old['source_hashes'].items() if hashes[p]!=h}
 allowed={str(ROOT/'engine/embodied/runtime_pool_guard.py'),str(ROOT/'check_pool_version_contract.py')}
 if not set(changes)<=allowed:raise RuntimeError('unapproved original-source changes: '+str(changes))
 assert sha(OLD/'MODEL.xml')==MODEL_SHA
 dump(OUT/'SOURCE_IDENTITY.json',dict(source_sha256=hashes,historical_changes=changes,approved_reason='Parent controlled external root888 version admission; freeze after parent readiness',contract_path=str(contract_path),pool_path=str(pool_path),sensory_path=str(sensory_path)))
 (OUT/'EXECUTION_SOURCE.py').write_bytes(Path(__file__).read_bytes())
 for p in sorted(paths):
  if p.suffix=='.py' and p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'venv_body'):
   target=OUT/'SOURCES'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
 (OUT/'INPUTS').mkdir()
 for label,p in [('VERSION_CONTRACT.json',contract_path),('VALIDATED_POOL.json',pool_path),('VALIDATED_SENSORY.json',sensory_path)]:
  (OUT/'INPUTS'/label).write_bytes(p.read_bytes())
 (OUT/'NATIVE').mkdir()
 orig_guard=guard.load_runtime_pool;orig_map=closure.channel_index_map;orig_pool=closure.POOLS
 orig_step=mujoco.mj_step;orig_forward=mujoco.mj_forward;orig_jac=mujoco.mj_jac;orig_force=mujoco.mj_contactForce
 orig_build=runner.build;orig_tier=closure.build_neural_tier;orig_contacts=runner.contacts
 orig_netstep=ConductanceNetwork.step;orig_init=ConductanceNetwork.__init__;orig_cont=handover.run_neural_continuous
 counters=dict(body_steps=0,neural_steps=0,network_instances=0,live_forward=0,observer_forward=0,observations=0,observer_jacobian_calls=0,original_jacobian_calls=0,observer_contact_force=0,original_contact_force=0,guard_calls=0,lookup_calls=0,admission_preflight_calls=1)
 chunks=[];samples=[];contact_rows=[];offsets=[0];raster=[];clone=None;meta=None;last_net=None;start=time.time();error=None
 def fail(kind,detail):
  dump(OUT/'FIRST_FAILURE.json',dict(kind=kind,detail=detail,counters=counters.copy()))
  raise RuntimeError(kind+': '+str(detail))
 def admitted(path,*args,**kwargs):
  assert not args and not kwargs and Path(path).resolve()==pool_path
  counters['guard_calls']+=1
  return orig_guard(path,version_contract=contract)
 def channels(n,roots,**kwargs):
  assert not kwargs
  counters['lookup_calls']+=1
  result=orig_map(n,roots,lookup_path=sensory_path)
  dump(OUT/'SENSORY_GROUPS.json',result[0])
  return result
 def build(*args,**kwargs):
  result=orig_build(*args,**kwargs)
  if sha(OUT/'RUN/MODEL.xml')!=MODEL_SHA:fail('MODEL_SHA256',sha(OUT/'RUN/MODEL.xml'))
  with np.load(OLD/'BODY.npz') as b:
   if not np.array_equal(result[1],b['qpos'][0]):fail('INITIAL_QPOS','build differs from original121')
  return result
 def tier(cfg):
  built=orig_tier(cfg);roots=np.asarray(built['tier']['root_ids'])
  with np.load(NATIVE/'IDENTITIES.npz') as ref:assert np.array_equal(roots,ref['root_ids'])
  for name in ['We','Wi']:
   w=built['edges'][name]*.2;delta=w-load_npz(NATIVE/('ACTUAL_'+name+'.npz'))
   assert not delta.nnz or np.max(np.abs(delta.data))==0
   save_npz(OUT/('ACTUAL_'+name+'.npz'),w)
  np.savez_compressed(OUT/'IDENTITIES.npz',root_ids=roots,global_index=built['global_index'])
  dump(OUT/'NEURAL_IDENTITY.json',dict(fingerprint=built['fingerprint'],n=int(built['n']),weight_multiplier=.2,root_order_and_weights_exact_native=True,tier_config=vars(cfg)))
  return built
 def netinit(net,*args,**kwargs):
  nonlocal last_net
  counters['network_instances']+=1;orig_init(net,*args,**kwargs);last_net=net
 def netstep(net,*args,**kwargs):
  spikes=orig_netstep(net,*args,**kwargs);counters['neural_steps']+=1;raster.append(spikes.copy())
  if any(not np.isfinite(v).all() for v in vars(net).values() if isinstance(v,np.ndarray)):fail('NONFINITE_NEURAL',counters['neural_steps'])
  return spikes
 def live_forward(*args,**kwargs):
  counters['live_forward']+=1;return orig_forward(*args,**kwargs)
 def original_jac(*args,**kwargs):
  counters['original_jacobian_calls']+=1;return orig_jac(*args,**kwargs)
 def original_force(*args,**kwargs):
  counters['original_contact_force']+=1;return orig_force(*args,**kwargs)
 def continuous(*args,**kwargs):
  result=orig_cont(*args,**kwargs)
  role='PREWARM' if kwargs.get('runtime_state') is None else 'RELEASE'
  rt=result['runtime_state'];net,encoder,decoder=rt['runtime']
  arrays={k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)}
  arrays.update(decoder_rates=decoder.rates.copy(),pending=rt['pending'].copy(),time_s=net.t_ms/1000,body_time=rt['body_time'],steps=net.steps,ring_pos=net._ring_i)
  np.savez_compressed(OUT/(role+'_FINAL_STATE.npz'),**arrays)
  dump(OUT/(role+'_CLOCK.json'),dict(net_scalars={k:v for k,v in vars(net).items() if isinstance(v,(int,float,np.integer,np.floating))},decoder_elapsed_s=decoder.elapsed_s,decoder_interval=decoder.interval))
  return result
 def flush():
  nonlocal samples,contact_rows,offsets
  if not samples:return
  path=OUT/'NATIVE'/('chunk_%04d.npz'%len(chunks))
  arrays={k:np.asarray([r[k] for r in samples]) for k in samples[0]}
  arrays.update(contact_offsets=np.asarray(offsets,np.int64),contact_rows=np.asarray(contact_rows,float).reshape(-1,30))
  np.savez_compressed(path,**arrays)
  chunks.append(dict(file=str(path),samples=len(samples),contacts=len(contact_rows),sha256=sha(path)))
  dump(OUT/'CHUNKS.json',chunks);samples=[];contact_rows=[];offsets=[0]
 def observe(model,live,kind=0):
  nonlocal clone,meta
  if clone is None:
   clone=mujoco.MjData(model);names=[model.geom(g).name or '' for g in range(model.ngeom)]
   feet=np.array([next((j for j,l in enumerate(LEGS) if name.startswith(l) and 'Tars' in name),-1) for name in names])
   floors=model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE;bids=np.array([model.body(l+'Tarsus5').id for l in LEGS]);thorax=model.body('Thorax').id
   meta=feet,floors,bids,thorax
   schema=json.loads((NATIVE/'CONTACT_SCHEMA.json').read_text());schema.update(jacobian_sampling='Only step%40==0, initial and release-after; other velocity columns NaN (not evaluated)',sample_kind={'0':'physical step endpoint','1':'release after withdrawal','2':'initial'},geom_names=names,geom_bodyid=model.geom_bodyid,floor_geom_indices=np.flatnonzero(floors))
   dump(OUT/'CONTACT_SCHEMA.json',schema)
  mujoco.mj_copyData(clone,model,live);orig_forward(model,clone);counters['observer_forward']+=1
  d=clone;feet,floors,bids,thorax=meta;step=int(round(d.time/DT));jac_frame=step%40==0 or kind!=0
  forces=np.zeros((6,3));flags=np.zeros(6,bool);surface=np.zeros(6,bool);nf=np.zeros(3);nfc=0;speeds=np.zeros(6) if jac_frame else np.full(6,np.nan);mindist=np.full(6,np.inf)
  jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);jp2=np.zeros_like(jp);jr2=np.zeros_like(jp);force=np.zeros(6)
  for j in range(d.ncon):
   c=d.contact[j];g1=int(c.geom1);g2=int(c.geom2);b1=int(model.geom_bodyid[g1]);b2=int(model.geom_bodyid[g2]);frame=c.frame.reshape(3,3);normal=frame[0].copy()
   orig_force(model,d,j,force);counters['observer_contact_force']+=1;fw=frame.T@force[:3]
   if jac_frame:
    orig_jac(model,d,jp,jr,c.pos,b1);orig_jac(model,d,jp2,jr2,c.pos,b2);counters['observer_jacobian_calls']+=2
    rel=(jp2-jp)@d.qvel;vn=float(rel@normal);tangent=rel-vn*normal;speed=float(np.linalg.norm(tangent))
   else:rel=np.full(3,np.nan);vn=np.nan;tangent=rel.copy();speed=np.nan
   leg=-1
   if floors[g1] or floors[g2]:
    other=g2 if floors[g1] else g1;leg=int(feet[other]);signed=fw if floors[g1] else -fw
    if leg>=0:
     flags[leg]=True;forces[leg]+=signed;surface[leg]|=c.dist<=0;mindist[leg]=min(mindist[leg],c.dist)
     if jac_frame:speeds[leg]=max(speeds[leg],speed)
    else:nf+=signed;nfc+=1
   contact_rows.append([g1,g2,b1,b2,leg,c.dist,c.includemargin,int(c.exclude),int(c.efc_address),*c.pos,*normal,*fw,*rel,*tangent,speed,vn,force[0],c.friction[0],c.friction[1],int(c.dim)])
  phase=3 if step>104000 or (step==104000 and kind==1) else 0 if step<=12000 else 1 if step<=20000 else 2
  samples.append(dict(step=step,time=float(d.time),kind=kind,phase=phase,jacobian_sample=jac_frame,foot_force=forces,foot_candidate=flags,foot_surface_candidate=surface,foot_loaded=forces[:,2]>0,nonfoot_force=nf,nonfoot_count=nfc,foot_max_contact_tangent_speed=speeds,foot_min_candidate_dist=mindist,foot_origin=d.xpos[bids].copy(),root=d.qpos[:7].copy(),com=d.subtree_com[thorax].copy(),assist_root_max=float(abs(d.qfrc_applied[:6]).max()),assist_leg_norm=float(np.linalg.norm(d.qfrc_applied[6:])),external_max=float(abs(d.xfrc_applied).max()),mocap_count=int(model.nmocap),warning_count=int(live.warning.number.sum()),ctrl=live.ctrl.copy(),activation=live.act.copy()))
  offsets.append(len(contact_rows));counters['observations']+=1
  if len(samples)>=4000:flush()
 def step(model,data,*args,**kwargs):
  orig_step(model,data,*args,**kwargs);counters['body_steps']+=1;n=counters['body_steps']
  if any(not np.isfinite(v).all() for v in [data.qpos,data.qvel,data.act]):fail('NONFINITE_BODY',n)
  if data.warning.number.any():fail('MUJOCO_WARNING',dict(step=n,numbers=data.warning.number))
  if n>104000 and (data.qfrc_applied.any() or data.xfrc_applied.any()):fail('ASSIST_RELEASE',n)
  observe(model,data)
  if n%8000==0:print('ROOT888_PROGRESS',n,'of160000',round(time.time()-start,2),flush=True)
 def contacts(model,data):
  result=orig_contacts(model,data);n=int(round(data.time/DT))
  if n==0:observe(model,data,2)
  elif n==104000 and not data.qfrc_applied.any():observe(model,data,1)
  return result
 guard.load_runtime_pool=admitted;closure.POOLS=pool_path;closure.channel_index_map=channels
 runner.build=build;closure.build_neural_tier=tier;runner.contacts=contacts
 mujoco.mj_step=step;mujoco.mj_forward=live_forward;mujoco.mj_jac=original_jac;mujoco.mj_contactForce=original_force
 ConductanceNetwork.step=netstep;ConductanceNetwork.__init__=netinit;handover.run_neural_continuous=continuous
 try:
  runner.run(OUT/'RUN',**CONFIG)
  assert (counters['body_steps'],counters['neural_steps'],counters['network_instances'],counters['guard_calls'],counters['lookup_calls'])==(160000,3600,1,1,1)
  assert counters['observations']==160002
 except BaseException:error=traceback.format_exc()
 finally:
  guard.load_runtime_pool=orig_guard;closure.POOLS=orig_pool;closure.channel_index_map=orig_map
  runner.build=orig_build;closure.build_neural_tier=orig_tier;runner.contacts=orig_contacts
  mujoco.mj_step=orig_step;mujoco.mj_forward=orig_forward;mujoco.mj_jac=orig_jac;mujoco.mj_contactForce=orig_force
  ConductanceNetwork.step=orig_netstep;ConductanceNetwork.__init__=orig_init;handover.run_neural_continuous=orig_cont
  flush()
  np.savez_compressed(OUT/'NEURAL_RASTER.npz',spikes=np.asarray(raster,dtype=bool))
  unchanged={p:sha(p)==h for p,h in hashes.items()}
  if not all(unchanged.values()) and error is None:error='source identity changed during execution'
  dump(OUT/'EXECUTION.json',dict(counters=counters,wall_seconds=time.time()-start,source_unchanged=unchanged,exit_code=int(error is not None),error=error))
 if error:raise RuntimeError(error)
 print('ROOT888_COMPLETE',counters,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--contract');p.add_argument('--pool');p.add_argument('--sensory');a=p.parse_args()
 if a.prepare:prepare()
 elif all([a.contract,a.pool,a.sensory]):run(a.contract,a.pool,a.sensory)
 else:p.error('use --prepare or all --contract/--pool/--sensory; no implicit admission')
