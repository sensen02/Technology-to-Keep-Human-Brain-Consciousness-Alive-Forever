"""One identity-frozen 121 replay plus cloned-data native contact observations."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,hashlib,time,traceback
import numpy as np
import mujoco
from scipy.sparse import save_npz,load_npz
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
S=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-native-recording/20261101T191552Z')
REPLAY=Path('/home/sensen/Desktop/engine修复验收/s4-neural-boundary-replay/20261101T185129Z')
DT=.0000125;LEGS=['LF','LM','LH','RF','RM','RH']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def dump(p,v):
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);(O/'NATIVE').mkdir()
 old=json.loads((S/'PREREGISTRATION.json').read_text())
 inputs=[S/n for n in ['BODY.npz','NEURAL.npz','MODEL.xml','GAIT.npz','PREREGISTRATION.json','HANDOVER_BEFORE.npz','HANDOVER_AFTER.npz','NEURAL_BOUNDARY.npz','run_physical_walk_prewarm.py']]+[Path(p) for p in old['source_hashes']]+[Path(__file__),REPLAY/'IDENTITIES.npz',REPLAY/'ACTUAL_We.npz',REPLAY/'ACTUAL_Wi.npz']
 hashes={str(p):sha(p) for p in inputs}
 assert all(hashes[p]==h for p,h in old['source_hashes'].items())
 assert hashes[str(S/'MODEL.xml')]=='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
 config=dict(duration=1.3,kp=100.,clip=50.,dt=DT,neural_duration=.7,neural_command_dt_s=.001,neural_prewarm_s=1.1,swing_height=.45,sample_dt_s=.0005)
 dump(O/'PREREGISTRATION.json',dict(source_sha256=hashes,runner_config=config,only_change='Observation wrapper: original mj_step and all original live forward/save cadence unchanged. Full mj_copyData to separate observer, forward on clone only, Jacobians on clone.',gate='first shared qpos/qvel/activation difference >1e-10 or aggregate neural-count difference aborts. Final all original BODY/NEURAL/boundary arrays <=1e-10, integer exact; first failure saved, no behavior interpretation.',native='Every physical step END plus initial and after-withdrawal boundary observation; boundary duplicate distinguished by kind. Original half-millisecond BODY remains unchanged.',phase='step endpoints up to and including 1.3s are generated under auxiliary controller; separate kind=1 boundary at1.3 is after withdrawal. Phase codes:0 settle,1 ramp,2 steady assisted,3 neural-only.',contact='All current MuJoCo contact candidates, original geom ids, signed dist/includemargin/exclude/efc_address, frame, point, local contact force. Body2 minus body1 material velocity at SAME world contact point via mj_jac*qvel; tangent removes frame-normal component. Not Tarsus origin.',slip='Tangential speed magnitude per contact point, not displacement of moving contact centroid. No identity assumed for contact index across steps. Per-foot peak contact tangent speed integrated only as explicitly labeled proxy; raw all contacts retained.',episodes='Separate exact sampled Boolean candidate, Fz>0 and candidate dist<=0 runs, unmerged; endpoints censored. Grid observations not continuous-time crossing estimates.',resources='One continuous2s integration,160000body steps/3600neural steps; original runner2model compilations, observer MjData on same model. Chunk4000 samples. No render.',claim='Measurement reproduction only; not gait success'))
 import run_physical_walk_prewarm as runner
 import run_neural_muscle_closure as closure
 from engine.neural_cond import ConductanceNetwork
 with np.load(S/'BODY.npz') as z:hist={k:z[k].copy() for k in z.files}
 with np.load(S/'NEURAL.npz') as z:nh={k:z[k].copy() for k in z.files}
 orig_step=mujoco.mj_step;orig_forward=mujoco.mj_forward;orig_contacts=runner.contacts;orig_build=runner.build;orig_tier=closure.build_neural_tier;orig_netstep=ConductanceNetwork.step
 counters=dict(body_steps=0,neural_steps=0,observer_forward=0,observations=0,jacobian_calls=0,shared_state_samples=0)
 chunks=[];samples=[];contacts=[];offset=[0];clone=None;meta=None;gate_fail=None;neural_pair=[];start=time.time();max_shared=0.
 def fail(kind,detail):
  nonlocal gate_fail
  gate_fail=dict(kind=kind,detail=detail,counters=counters.copy());dump(O/'FIRST_MISMATCH.json',gate_fail);raise RuntimeError('REPRODUCTION_GATE_FAILED:'+kind)
 def build(*args,**kwargs):
  result=orig_build(*args,**kwargs)
  if sha(O/'RUN/MODEL.xml')!=hashes[str(S/'MODEL.xml')]:fail('MODEL_HASH',str(O/'RUN/MODEL.xml'))
  return result
 def tier(cfg):
  built=orig_tier(cfg);roots=np.asarray(built['tier']['root_ids']);we=built['edges']['We']*.2;wi=built['edges']['Wi']*.2
  with np.load(REPLAY/'IDENTITIES.npz') as z:
   if not np.array_equal(roots,z['root_ids']):fail('ROOT_ORDER','replay identities differ')
  for name,w in [('We',we),('Wi',wi)]:
   diff=w-load_npz(REPLAY/('ACTUAL_'+name+'.npz'))
   if diff.nnz and np.max(abs(diff.data))>0:fail('WEIGHT_IDENTITY',name)
   save_npz(O/('ACTUAL_'+name+'.npz'),w)
  np.savez_compressed(O/'IDENTITIES.npz',root_ids=roots,global_index=built['global_index'])
  dump(O/'NEURAL_IDENTITY.json',dict(fingerprint=built['fingerprint'],n=int(built['n']),roots_equal_replay=True,weights_equal_replay=True,weight_multiplier=.2))
  return built
 def netstep(net,*args,**kwargs):
  sp=orig_netstep(net,*args,**kwargs);counters['neural_steps']+=1;n=counters['neural_steps']
  if n>2200:
   neural_pair.append(sp.copy())
   if len(neural_pair)==2:
    k=(n-2200)//2-1;cnt=np.sum(neural_pair,axis=0);neural_pair.clear()
    if not np.array_equal(cnt,nh['spike_counts'][k]):fail('NEURAL_COUNTS',dict(window=k,cells=np.flatnonzero(cnt!=nh['spike_counts'][k])[:20]))
  return sp
 def flush():
  nonlocal samples,contacts,offset
  if not samples:return
  num=len(chunks);path=O/'NATIVE'/('chunk_%04d.npz'%num)
  arrays={k:np.array([r[k] for r in samples]) for k in samples[0]}
  arrays['contact_offsets']=np.array(offset,dtype=np.int64)
  width=30
  arrays['contact_rows']=np.asarray(contacts,dtype=float).reshape(-1,width)
  np.savez_compressed(path,**arrays);chunks.append(dict(file=str(path),samples=len(samples),contacts=len(contacts),first_step=int(samples[0]['step']),last_step=int(samples[-1]['step']),sha256=sha(path)))
  dump(O/'CHUNKS.json',chunks);samples=[];contacts=[];offset=[0]
 def observe(model,live,kind=0):
  nonlocal clone,meta
  if clone is None:
   clone=mujoco.MjData(model)
   geom_names=[model.geom(g).name or '' for g in range(model.ngeom)]
   feet=np.array([next((j for j,l in enumerate(LEGS) if name.startswith(l) and 'Tars' in name),-1) for name in geom_names])
   floors=model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
   bids=np.array([model.body(l+'Tarsus5').id for l in LEGS]);thorax=model.body('Thorax').id
   meta=(feet,floors,bids,thorax)
   dump(O/'CONTACT_SCHEMA.json',dict(columns=['geom1','geom2','body1','body2','leg_or_minus1','dist','includemargin','exclude','efc_address','point_x','point_y','point_z','normal_x','normal_y','normal_z','force_on_geom2_x','force_on_geom2_y','force_on_geom2_z','vrel_x','vrel_y','vrel_z','tangent_x','tangent_y','tangent_z','tangent_speed','normal_velocity','normal_force','friction0','friction1','dim'],geom_names=geom_names,geom_bodyid=model.geom_bodyid,leg_order=LEGS,units='model length/s and model force; no biological SI calibration',sample_kind={'0':'physical-step endpoint','1':'after-withdrawal boundary','2':'initial original recorder boundary'}))
  mujoco.mj_copyData(clone,model,live);orig_forward(model,clone);counters['observer_forward']+=1
  d=clone;feet,floors,bids,thorax=meta;forces=np.zeros((6,3));flags=np.zeros(6,bool);surface=np.zeros(6,bool);nf=np.zeros(3);nfc=0;speeds=np.zeros(6);mindist=np.full(6,np.inf)
  jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);jp2=np.zeros_like(jp);jr2=np.zeros_like(jp);force=np.zeros(6)
  for j in range(d.ncon):
   c=d.contact[j];g1=int(c.geom1);g2=int(c.geom2);b1=int(model.geom_bodyid[g1]);b2=int(model.geom_bodyid[g2]);frame=c.frame.reshape(3,3);normal=frame[0].copy()
   mujoco.mj_contactForce(model,d,j,force);fw=frame.T@force[:3]
   mujoco.mj_jac(model,d,jp,jr,c.pos,b1);mujoco.mj_jac(model,d,jp2,jr2,c.pos,b2);counters['jacobian_calls']+=2
   rel=(jp2-jp)@d.qvel;vn=float(rel@normal);tangent=rel-vn*normal;speed=float(np.linalg.norm(tangent))
   leg=-1
   if floors[g1] or floors[g2]:
    other=g2 if floors[g1] else g1;leg=int(feet[other]);signed=fw if floors[g1] else -fw
    if leg>=0:
     flags[leg]=True;forces[leg]+=signed;surface[leg]|=c.dist<=0;speeds[leg]=max(speeds[leg],speed);mindist[leg]=min(mindist[leg],c.dist)
    else:nf+=signed;nfc+=1
   contacts.append([g1,g2,b1,b2,leg,float(c.dist),float(c.includemargin),int(c.exclude),int(c.efc_address),*c.pos,*normal,*fw,*rel,*tangent,speed,vn,float(force[0]),float(c.friction[0]),float(c.friction[1]),int(c.dim)])
  step=int(round(d.time/DT));phase=3 if (step>104000 or (step==104000 and kind==1)) else (0 if step<=12000 else 1 if step<=20000 else 2)
  samples.append(dict(step=step,time=float(d.time),kind=kind,phase=phase,phase_rad=max(0,float(d.time)-.15)*10*np.pi%(2*np.pi),foot_force=forces,foot_candidate=flags,foot_surface_candidate=surface,foot_loaded=forces[:,2]>0,nonfoot_force=nf,nonfoot_count=nfc,foot_max_contact_tangent_speed=speeds,foot_min_candidate_dist=mindist,foot_origin=d.xpos[bids].copy(),root=d.qpos[:7].copy(),com=d.subtree_com[thorax].copy(),assist_root_max=float(abs(d.qfrc_applied[:6]).max()),assist_leg_norm=float(np.linalg.norm(d.qfrc_applied[6:])),external_max=float(abs(d.xfrc_applied).max())))
  offset.append(len(contacts));counters['observations']+=1
  if len(samples)>=4000:flush()
 def step(model,data,*args,**kwargs):
  nonlocal max_shared
  orig_step(model,data,*args,**kwargs);counters['body_steps']+=1;n=counters['body_steps']
  if n%40==0:
   k=n//40
   for field,stored in [('qpos','qpos'),('qvel','qvel'),('act','activation')]:
    error=float(np.max(abs(getattr(data,field)-hist[stored][k])));max_shared=max(max_shared,error)
    if error>1e-10:fail('BODY_SHARED_STATE',dict(step=n,field=field,error=error))
   counters['shared_state_samples']+=1
  observe(model,data)
  if n%8000==0:print('NATIVE_PROGRESS',n,'of160000','seconds',round(time.time()-start,2),flush=True)
 def recorder_contacts(model,data):
  result=orig_contacts(model,data);n=int(round(data.time/DT))
  if n==0:observe(model,data,2)
  elif n==104000 and not np.any(data.qfrc_applied):observe(model,data,1)
  return result
 runner.build=build;runner.contacts=recorder_contacts;closure.build_neural_tier=tier;ConductanceNetwork.step=netstep;mujoco.mj_step=step
 error=None
 try:
  runner.run(O/'RUN',**config)
  execution=json.loads((O/'RUN/EXECUTION.json').read_text())
  if execution['exit_code']!=0:raise RuntimeError('original runner failed: '+str(execution.get('error')))
  if counters['body_steps']!=160000 or counters['neural_steps']!=3600:fail('COUNTERS',counters.copy())
  comparisons={}
  for file in ['BODY.npz','NEURAL.npz','HANDOVER_BEFORE.npz','HANDOVER_AFTER.npz','NEURAL_BOUNDARY.npz','GAIT.npz']:
   with np.load(S/file) as ref,np.load(O/'RUN'/file) as got:
    if set(ref.files)!=set(got.files):fail('FILE_KEYS',file)
    result={}
    for key in ref.files:
     x=ref[key];y=got[key]
     if x.shape!=y.shape:fail('ARRAY_SHAPE',dict(file=file,key=key))
     exact=np.array_equal(x,y);delta=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.size else 0.
     if (x.dtype.kind in 'biu' and not exact) or delta>1e-10:fail('FINAL_ARRAY',dict(file=file,key=key,error=delta))
     result[key]=dict(exact=exact,max_abs_error=delta)
    comparisons[file]=result
  dump(O/'REPRODUCTION_GATE.json',dict(passed=True,comparisons=comparisons,max_online_shared_error=max_shared,model_hash_match=True,interpretation='same trajectory measurement replay, not new behavior success'))
 except Exception:
  error=traceback.format_exc();dump(O/'FAILURE.json',dict(error=error,counters=counters))
 finally:
  mujoco.mj_step=orig_step;runner.build=orig_build;runner.contacts=orig_contacts;closure.build_neural_tier=orig_tier;ConductanceNetwork.step=orig_netstep;flush()
  unchanged=hashes=={str(p):sha(p) for p in inputs}
  dump(O/'EXECUTION.json',dict(counters=counters,original_runner_counts=json.loads((O/'RUN/EXECUTION.json').read_text()).get('counts') if (O/'RUN/EXECUTION.json').exists() else None,wall_seconds=time.time()-start,source_hashes_unchanged=unchanged,exit_code=0 if error is None and unchanged else 1,error=error))
 if error:raise RuntimeError(error)
 print('RECORDING_COMPLETE_GATE_PASSED',counters,flush=True)
if __name__=='__main__':main()
