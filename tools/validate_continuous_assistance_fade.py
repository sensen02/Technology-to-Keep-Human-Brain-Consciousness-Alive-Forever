"""S5 one preregistered continuous-assistance-fade trial. Never edits defaults.
Modes prepare/run/verify/report; no seal before parent review, no delegation.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,math,inspect,difflib,hashlib,shutil,argparse,dataclasses
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R),str(R/'tools')]
O=Path('/home/sensen/Desktop/engine修复验收/s5-continuous-assistance-fade/20261102T074033Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
C=Path('/home/sensen/Desktop/engine修复验收/s4-bending-closedloop-training/20261102T042848Z')
E=Path('/home/sensen/Desktop/engine修复验收/s2-early-muscle-support/20261102T053639Z')
LABEL='CONTINUOUS_FADE_200MS';DT=.0000125;H0=1.2072528224791905
RUNTIME=[];MODEL_IDS=[];NATIVE=[];EVENTS={};BOUNDARIES={};ASSIST=[];LAST={};RAW=[]

def capture_assist_raw(clock,raw,multiplier,scaled):
 if round(clock/DT)>=79999:RAW.append((float(clock),raw.copy(),float(multiplier),scaled.copy()))

def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def js(p):return json.loads(Path(p).read_text())
def dump(p,x):
 Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def rep(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def snapshot_net(net,dec,pending):
 return dict(**{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)},decoder_rates=dec.rates.copy(),pending=np.asarray(pending).copy(),steps=net.steps,ring_pos=net._ring_i,time_s=net.t_ms/1000)

def instrument(origcont,model,data,args,kw,folder,counters,netref):
 """Read-only post-forward native observation plus original actual-assist callback tap."""
 import mujoco
 from engine.embodied.continuous_handover import state_snapshot
 release=kw.get('runtime_state') is not None
 compiled={k:getattr(model,k).copy() for k in dir(model) if isinstance(getattr(model,k),np.ndarray)}
 if not release:
  MODEL_IDS.append(id(model));np.savez_compressed(folder/'ACTUAL_COMPILED_INITIAL.npz',**compiled)
 else:
  assert id(model)==MODEL_IDS[0]
  net,enc,dec=RUNTIME[0]
  np.savez_compressed(folder/'RELEASE_ENTRY_STATE.npz',**snapshot_net(net,dec,kw['runtime_state']['pending']))
 names=[model.geom(i).name or '' for i in range(model.ngeom)]
 feet=np.array([next((j for j,l in enumerate(('LF','LM','LH','RF','RM','RH')) if n.startswith(l) and 'Tars' in n),-1) for n in names])
 planes=model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE;thorax=model.body('Thorax').id
 va=np.arange(6,model.nv);assert len(va)==42
 oldobserver=kw['body_observer'];oldassist=kw.get('assisted_prewarm_callback');oldinterval=kw.get('interval_observer')
 if oldassist:
  def tap(mm,dd):
   clock=float(dd.time);oldassist(mm,dd)
   remain=float(np.clip((1.3-clock)/.2,0,1))
   # Read the force actually applied to this integration step, after original multiplication.
   LAST.update(clock=clock,multiplier=remain,torque=dd.qfrc_applied[va].copy())
   if round(clock/DT)>=79999:
    ASSIST.append((clock,remain,*dd.qfrc_applied[va],float(np.max(abs(dd.qfrc_applied[:6]))),float(np.max(abs(dd.xfrc_applied)))))
  kw['assisted_prewarm_callback']=tap
 def observe(mm,dd):
  oldobserver(mm,dd)
  step=round(dd.time/DT)
  if step<80000:return
  ff=np.zeros(6);nf=0.;agg=0.;nc=0;force=np.zeros(6);positive=[];candidate=[]
  for j in range(dd.ncon):
   co=dd.contact[j];g1=int(co.geom1);g2=int(co.geom2)
   if not(planes[g1] or planes[g2]):continue
   mujoco.mj_contactForce(mm,dd,j,force);counters['native_contact_force']+=1
   gf=g2 if planes[g1] else g1;leg=feet[gf]
   world=(co.frame.reshape(3,3).T@force[:3])*(1 if planes[g1] else -1);fz=float(world[2])
   if leg<0:
    nf+=max(fz,0.);agg+=fz;nc+=1
    identity=dict(contact_index=j,geom_id=gf,geom_name=names[gf],body_id=int(mm.geom_bodyid[gf]),body_name=mm.body(int(mm.geom_bodyid[gf])).name,ground_geom_id=g1 if planes[g1] else g2,signed_world_force=world.tolist(),position=co.pos.copy().tolist(),distance=float(co.dist))
    candidate.append(identity)
    if fz>1e-8*float(mm.body_mass.sum()*abs(mm.opt.gravity[2])):positive.append(identity)
   else:ff[leg]+=fz
  phase=0 if step<=88000 else (1 if not release else 2)
  # Physical poststep force belongs to the pre-step callback clock; initial release has no callback.
  clk=LAST.get('clock',float(dd.time)) if not release else float(dd.time)
  mult=LAST.get('multiplier',0.) if not release else 0.
  norm=float(np.linalg.norm(dd.qfrc_applied[va]));root=float(np.max(abs(dd.qfrc_applied[:6])));ext=float(np.max(abs(dd.xfrc_applied)))
  NATIVE.append((float(dd.time),*ff,nf,agg,float(nc),*dd.subtree_com[thorax],*dd.qpos[:7],float(np.mean(dd.ctrl**2)),norm,clk,mult,root,ext,float(phase),*dd.qfrc_applied[va]))
  key='release' if release else ('fade' if step>88000 else 'prefade')
  for typ,rows in [('candidate',candidate),('positive',positive)]:
   ek=key+'_'+typ
   if rows and ek not in EVENTS:EVENTS[ek]=dict(time_s=float(dd.time),step=step,relative_release_s=float(dd.time)-1.3,identities=rows)
  assert root==0 and ext==0
  if release:assert not dd.qfrc_applied.any()
  if step in (88000,104000,112000):
   role=('FADE_START' if step==88000 else 'FADE_END' if step==104000 else 'FINAL')+('_RELEASE' if release else '_ASSISTED')
   if role not in BOUNDARIES:
    jac=np.zeros((3,mm.nv));mujoco.mj_jacSubtreeCom(mm,dd,jac,thorax);counters['observer_COMJac']=counters.get('observer_COMJac',0)+1
    com_velocity=jac@dd.qvel
    BOUNDARIES[role]=dict(com_velocity=com_velocity,step=step,time=float(dd.time),body_role='poststep observer before completed-window decoder consume',actual_last_assist_norm=norm,multiplier=mult,multiplier_clock=clk,phase=phase,feet_load=ff.tolist(),positive_nonfoot_Fz=nf,algorithm_phase=float((max(0,float(dd.time)-.15)*2*math.pi*5)%(2*math.pi)))
    np.savez_compressed(folder/(role+'_BODY.npz'),**state_snapshot(dd),com=dd.subtree_com[thorax],foot6signedFz=ff,muscle_force=dd.actuator_force)
    net,enc,dec=RUNTIME[0]
    # ctrl is the applied pending window; completed pending captured by interval observer below.
    np.savez_compressed(folder/(role+'_NEURAL_PRECONSUME.npz'),**snapshot_net(net,dec,dd.ctrl))
 def interval(record):
  if oldinterval:oldinterval(record)
  st=round(record['absolute_end_s']/DT)
  if st in (88000,104000,112000):
   role='FADE_START' if st==88000 else 'FADE_END' if st==104000 else 'FINAL'
   net,enc,dec=RUNTIME[0]
   np.savez_compressed(folder/(role+'_NEURAL_COMPLETED.npz'),**snapshot_net(net,dec,record['decoded_next_command']))
 kw['body_observer']=observe;kw['interval_observer']=interval
 ret=origcont(model,data,*args,**kw)
 role='RELEASE' if release else 'PREWARM';net,enc,dec=ret['runtime_state']['runtime']
 np.savez_compressed(folder/(role+'_STATE.npz'),**snapshot_net(net,dec,ret['runtime_state']['pending']))
 if release:
  np.savez_compressed(folder/'BODY_BOUNDARY.npz',**ret['handover_after_withdrawal'])
  np.savez_compressed(folder/'ACTUAL_COMPILED_FINAL.npz',**{k:getattr(model,k).copy() for k in compiled})
  dump(folder/'MODEL_OBJECT_IDENTITY.json',dict(initial_model_id=MODEL_IDS[0],final_model_id=id(model),same_object=True,compiled_array_count=len(compiled),nbody=model.nbody,nmesh=model.nmesh,nmuscle=model.nu))
  from scipy import sparse
  for key in ['We','Wi']:sparse.save_npz(folder/('FINAL_'+key+'.npz'),getattr(net,key))
 return ret

def instrument_runner_source(text):
 before='if assistance_fade_s>0:';scaled='dd.qfrc_applied[va]*=float(np.clip(remain,0.,1.))'
 lines=text.splitlines(True);out=[];counts=[0,0]
 for line in lines:
  indent=line[:len(line)-len(line.lstrip())]
  if line.strip()==before:
   out.append(indent+'raw_clipped_assist=dd.qfrc_applied[va].copy()\n');counts[0]+=1
  out.append(line)
  if line.strip()==scaled:
   out.append(indent+'capture_assist_raw(dd.time,raw_clipped_assist,float(np.clip(remain,0.,1.)),dd.qfrc_applied[va])\n');counts[1]+=1
 assert counts==[1,1]
 result=''.join(out);compile(result,str(O/'RUNNER_PREFLIGHT.py'),'exec');return result

def prepare():
 import train_bending_closedloop as old
 assert not O.exists() or {p.name for p in O.iterdir()}=={'ATTEMPTS'},'refuse existing scientific evidence'
 O.mkdir(parents=True,exist_ok=True)
 text=inspect.getsource(old.trial);original=text
 # No parameter intervention: actual original matrices and threshold object are retained.
 text=rep(text,"row=np.repeat(np.arange(net.n),np.diff(w.indptr));scales=np.ones(net.n);scales[motor]=gain[g[motor]];w=w.copy();w.data*=scales[row];setattr(net,key,w)","assert np.array_equal(gain,np.ones(6))  # original CSR object retained")
 text=rep(text,"source=net.threshold_offset_mV.copy();offset=source.copy();offset[motor]=source[motor]+theta[g[motor]];offset.flags.writeable=False;net.threshold_offset_mV=offset","source=net.threshold_offset_mV.copy();assert not source.any() and not theta.any()")
 text=rep(text,"netref.append(net);return net,enc,dec","dump(folder/'ACTUAL_DECODER_CONFIG.json',dataclasses.asdict(dec.config))\n   RUNTIME.append((net,enc,dec));netref.append(net);return net,enc,dec")
 # inspect preserves the original single-space trial indentation; match it literally.
 text=text.replace("\n   RUNTIME.append", "\n  RUNTIME.append")
 start=text.index(' def continuous(model,data,*a,**kw):');end=text.index(' # Execute original runner',start)
 text=text[:start]+" def continuous(model,data,*a,**kw):\n  return instrument(origcont,model,data,a,kw,folder,counters,netref)\n"+text[end:]
 text=rep(text,"ns['run'](folder/'RUN',**CONFIG,neural_duration=release)","ns['run'](folder/'RUN',**CONFIG,neural_duration=release,assistance_fade_s=.2)")
 text=rep(text,"np.savez_compressed(folder/'BODYLOSS_NATIVE.npz',rows=np.asarray(native,float))","np.savez_compressed(folder/'NATIVE.npz',rows=np.asarray(NATIVE,float))\n  np.savez_compressed(folder/'ACTUAL_ASSIST_CALLBACK.npz',rows=np.asarray(ASSIST,float))\n  dump(folder/'FIRST_NONFOOT_EVENTS.json',EVENTS);dump(folder/'BOUNDARY_CLOCKS.json',BOUNDARIES)")
 text=rep(text,"counters['native_endpoints']=max(0,len(native)-1)","counters['native_endpoints']=len(NATIVE)")
 text=rep(text,"assert len(native)==round(release/DT)+1 and counters['neural']==round((1.1+release)/.0005)","assert len(NATIVE)==32002 and counters['neural']==2400")
 # Insert read-only raw/scaled torque capture in the inspected original runner, retaining its real fade multiplication.
 insertion=" runner_before=text\n text=instrument_runner_source(text)\n (folder/'RUNNER_LITERAL.diff').write_text(''.join(difflib.unified_diff(runner_before.splitlines(True),text.splitlines(True),fromfile=str(R/'run_physical_walk_prewarm.py'),tofile=str(folder/'RUNNER_ADAPTER.py'))))\n"
 import run_physical_walk_prewarm as original_runner
 instrument_runner_source(inspect.getsource(original_runner.run))
 text=rep(text," (folder/'RUNNER_ADAPTER.py').write_text(text);ns=dict(runner.__dict__);exec(compile(text,str(folder/'RUNNER_ADAPTER.py'),'exec'),ns)",insertion+" (folder/'RUNNER_ADAPTER.py').write_text(text);ns=dict(runner.__dict__);ns['capture_assist_raw']=capture_assist_raw;exec(compile(text,str(folder/'RUNNER_ADAPTER.py'),'exec'),ns)")
 text=rep(text,"  dump(folder/'FIRST_NONFOOT_EVENTS.json',EVENTS);dump(folder/'BOUNDARY_CLOCKS.json',BOUNDARIES)","  dump(folder/'FIRST_NONFOOT_EVENTS.json',EVENTS);dump(folder/'BOUNDARY_CLOCKS.json',BOUNDARIES)\n  np.savez_compressed(folder/'ASSIST_RAW_AND_SCALED.npz',clock=[x[0] for x in RAW],raw_clipped=[x[1] for x in RAW],multiplier=[x[2] for x in RAW],scaled=[x[3] for x in RAW])")
 compile(text,str(O/'TRIAL_EXECUTION_SOURCE.py'),'exec')
 (O/'TRIAL_ORIGINAL_SOURCE.py').write_text(original);(O/'TRIAL_EXECUTION_SOURCE.py').write_text(text)
 (O/'TRIAL_LITERAL.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(R/'tools/train_bending_closedloop.py')+':trial',tofile=str(O/'TRIAL_EXECUTION_SOURCE.py'))))
 (O/'EXECUTION_SOURCE.py').write_bytes(Path(__file__).read_bytes())
 paths=[Path(__file__),O/'TRIAL_EXECUTION_SOURCE.py',R/'tools/train_bending_closedloop.py',R/'run_physical_walk_prewarm.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py',B/'RUN/BODY.npz',B/'NEURAL_RASTER.npz',B/'RUN/MODEL.xml',B/'ACTUAL_We.npz',B/'ACTUAL_Wi.npz',B/'IDENTITIES.npz',C/'TRAIN_000_BASELINE/BODYLOSS_NATIVE.npz',E/'COMPILED_MODEL.npz']+list((R/'engine').rglob('*.py'))+[old.A/n for n in ['VERSION_CONTRACT.json','MN_POOLS_SNAPSHOT888.json','SENSORY_LOOKUP_SNAPSHOT888.json']]
 dump(O/'PREREGISTRATION.json',dict(head='8439d4289f3bc70d0f776077f3f8aa5fd347c5ca',stage='S5 single physically motivated continuous load transfer, research only',config=dict(old.CONFIG,assistance_fade_s=.2,neural_duration=.1),phases=[dict(start=0,end=1.1,role='unchanged original assistance; neural starts at .2'),dict(start=1.1,end=1.3,role='linear assistance fade, still ASSISTED'),dict(start=1.3,end=1.4,role='noassist; sole autonomous evaluation window')],rationale='Original abrupt withdrawal begins at COMvz+23.4899 with foot load.4716W; original full algorithm5Hz cycle .2s used for continuous load transfer, not a measured physiological cycle',historical_duplicate_check=dict(tracker_lines='1484..1490',source=str(R/'run_assist_fade_neural.py'),prior='/home/sensen/Desktop/engine修复验收/s5-assist-fade-neural/20261028T087000Z',different='old 1s gait+.15settle+.05hold, only.2neural prewarm,1.2release,.25neural-only; original old pool/model branch, not present root8883470 fullhistory',same_current_protocol_found=False),invariants='Original MODEL3470/initial/root888 WeWi*.2 offset0 tau5ms fullscale100 gain1 seed0 kp100 clip50 swing.45 dt12.5us cmd1ms sample.5ms. Same body/net/encoder/decoder/rates/pending; no reset, no root force, no weight/threshold/decoder changes, no plasticity/training/optimization',support=dict(horizon_s=.1,mean_foot_min_weight=.8,min_COMheight_ratio=.8,nonfoot_epsilon_weight=1e-8,fixed_h0=H0,additional_admission='positive nonfoot at fade end fails admission even if release itself passes',fade_early_nonfoot='report time/body/geom, never mask by stable prone release'),baseline='round13 original 2s read-only; same first100ms native cached TRAIN000, zero baseline integration',prefix_gate='every original BODY field and raster exact through first1.1s aligned by observer phase clock; preserve first mismatch and stop any training, do not relax exact',observation='No new liveforward. Original postforward observer reads native1.0..1.4 every12.5us; .5ms full original BODY/muscleforce, all2400tickraster,1200records; actual original assist callback tap records pre-step clock and42torques. Native initial1.0 endpoint plus release duplicate1.3 endpoint explicitly retained.',planned=dict(body_steps=112000,neural_steps=2400,model_compiles=2,factory=1,training=0,optimization=0,plasticity=0,new_baseline=0),stop='Only nonfinite/warnings abort physical integration; failed support/fade does not truncate fixed1.4s',source_sha256={str(p):sha(p) for p in paths},seal='UNSEALED until parent review; no automatic seal'))
 dump(O/'NATIVE_SCHEMA.json',dict(columns=['time','foot_LF_Fz','foot_LM_Fz','foot_LH_Fz','foot_RF_Fz','foot_RM_Fz','foot_RH_Fz','sum_each_positive_NFz','signed_aggregate_NFz','NFcount','COMx','COMy','COMz','rootx','rooty','rootz','rootqw','rootqx','rootqy','rootqz','ctrl_mean2','actual_assist_norm','multiplier_clock','multiplier','root_qfrc_max','xfrc_max','phase']+['assist_torque_'+str(i) for i in range(42)],phase=dict(prefade=0,fading_still_assisted=1,noassist=2),callback_columns=['prestep_body_time','actual_multiplier']+['actual42torque_'+str(i) for i in range(42)]+['root_qfrc_max','xfrc_max'],solver_force_clock='Poststep original forward; assist multiplier and torque in poststep row were evaluated at previous integration clock, not endpoint clock; release initial row force is after withdrawal. No clock shift fitted.',slip='not measured; no material-slip claim; no additional liveforward or Jacobian'))
 dump(O/'FAILURE_LEDGER.json',dict(failures=[dict(stage='environment preflight before preregistration',command='.venv/bin/python tools/validate_continuous_assistance_fade.py prepare',exit_code=1,reason='numpy absent in .venv; failed import before prepare, zero scientific calls'),dict(stage='environment probe',command='python -c import numpy,mujoco',exit_code=127,reason='python command absent'),dict(stage='environment probe',command='python3 -c import numpy,mujoco',exit_code=1,reason='system python3 has no mujoco')],preflight='expected output absence and absent environment glob ls exit2; recursive Python discovery timed out with no changes; correct existing venv_body supplied by parent; no model/body/neural calls in any probe'))
 print('PREREGISTERED',O,flush=True)

def run():
 import train_bending_closedloop as old
 pr=js(O/'PREREGISTRATION.json')
 for p,h in pr['source_sha256'].items():assert sha(p)==h,p
 scope=dict(old.__dict__);scope.update(O=O,instrument=instrument,RUNTIME=RUNTIME,NATIVE=NATIVE,ASSIST=ASSIST,EVENTS=EVENTS,BOUNDARIES=BOUNDARIES,RAW=RAW,capture_assist_raw=capture_assist_raw,rep=rep,difflib=difflib,instrument_runner_source=instrument_runner_source)
 exec(compile((O/'TRIAL_EXECUTION_SOURCE.py').read_text(),str(O/'TRIAL_EXECUTION_SOURCE.py'),'exec'),scope)
 scope['trial'](LABEL,np.zeros(12),.1)
 dump(O/'SOURCE_HASHES_AFTER_RUN.json',{p:sha(p) for p in pr['source_sha256']})
 print('SINGLE_RUN_COMPLETE',flush=True)

def metrics(rows,W,baseline=False):
 # Endpoint-only integration samples; boundary excluded. Caller has selected same100ms.
 foot=rows[:,1:7].sum(1);nf=rows[:,7];height=rows[:,10] if baseline else rows[:,12]
 root=rows[:,11:18] if baseline else rows[:,13:20]
 mask=nf>1e-8*W;tt=rows[:,0]-1.3
 return dict(endpoints=len(rows),first_positive_nonfoot_s=float(tt[np.flatnonzero(mask)[0]]) if mask.any() else None,nonfoot_occupancy=float(mask.mean()),mean_foot_weight=float(foot.mean()/W),mean_positive_nonfoot_weight=float(nf.mean()/W),min_height_ratio=float(height.min()/H0),min_COMz=float(height.min()),final_COMz=float(height[-1]),last_foot_weight=float(foot[-1]/W),no_nonfoot_gate=not mask.any(),foot_gate=bool(foot.mean()/W>=.8),height_gate=bool(height.min()/H0>=.8),support_gate=bool(not mask.any() and foot.mean()/W>=.8 and height.min()/H0>=.8),final_root=root[-1],walking_certified=False)

def verify():
 # Pure arrays, no mujoco/engine/runner import or body/neural integration.
 from scipy import sparse
 pr=js(O/'PREREGISTRATION.json');f=O/LABEL
 for p,h in pr['source_sha256'].items():assert sha(p)==h,p
 body=ar(f/'RUN/BODY.npz');base=ar(B/'RUN/BODY.npz');native=ar(f/'NATIVE.npz')['rows'];tap=ar(f/'ACTUAL_ASSIST_CALLBACK.npz')['rows']
 ev=np.unpackbits(ar(f/'RASTER.npz')['packed'],axis=1,count=12000).astype(bool);orig=ar(B/'NEURAL_RASTER.npz')['spikes']
 records=js(f/'RUN/PREWARM_NEURAL_RECORDS.json')+js(f/'RUN/NEURAL_RECORDS.json');mapping=js(B/'RUN/MAPPING.json')['rows']
 assert ev.shape==(2400,12000) and len(records)==1200
 checks={};first=[]
 # Identify BODY by absolute phase clocks, not positional role assumptions at exact1.1.
 ix=np.flatnonzero(np.rint(body['time']/DT).astype(int)<=88000);jx=np.flatnonzero(np.rint(base['time']/DT).astype(int)<=88000)
 assert np.array_equal(np.rint(body['time'][ix]/DT),np.rint(base['time'][jx]/DT))
 for k in body:
  a=body[k][ix];b=base[k][jx];ok=np.array_equal(a,b);checks[k]=ok
  if not ok:
   bad=np.argwhere(a!=b)[0];first.append(dict(field=k,index=bad.tolist(),time_s=float(body['time'][ix[bad[0]]]),new=np.asarray(a[tuple(bad)]).item(),baseline=np.asarray(b[tuple(bad)]).item(),maxabs=float(np.max(abs(a.astype(float)-b.astype(float))))))
 prefixr=np.array_equal(ev[:1800],orig[:1800]);checks['raster_1800ticks']=prefixr
 dump(O/'PREFIX_EXACT_GATE.json',dict(passed=all(checks.values()),checks=checks,first_differences=first,raster_first_difference=None if prefixr else np.argwhere(ev[:1800]!=orig[:1800])[0].tolist(),BODY_clock='first1.1s inclusive by rounded bodydt index, no tolerance on values',training_started=False))
 assert all(checks.values()),'exact prefix failed; saved first-difference diagnosis; no training'
 assert native.shape==(32002,69) and tap.shape==(24001,46)
 assert np.all(np.isfinite(native)) and np.all(native[:,24:26]==0) and np.all(tap[:,44:46]==0)
 # Actual physics torque independently compared to original unscaled PD algebra from original runner source.
 # Callback raw PD is separately captured below in RUNNER_ADAPTER; arrays must prove multiplication.
 raw=ar(f/'ASSIST_RAW_AND_SCALED.npz')
 expected=np.clip((1.3-raw['clock'])/.2,0,1)
 assert np.array_equal(raw['multiplier'],expected)
 assert np.array_equal(raw['scaled'],raw['raw_clipped']*expected[:,None])
 assert np.array_equal(tap[:,0],raw['clock']) and np.array_equal(tap[:,1],expected) and np.array_equal(tap[:,2:44],raw['scaled'])
 # All native torques are exactly the most recent actual callback torque (same pre-step clock).
 assisted=native[:,26]<2;indices=np.rint((native[assisted,22]-tap[0,0])/DT).astype(int)
 assert np.array_equal(native[assisted,27:],tap[indices,2:44])
 assert np.array_equal(native[assisted,23],tap[indices,1])
 assert np.all(native[~assisted,21]==0) and np.all(native[~assisted,27:]==0) and np.all(native[~assisted,23]==0)
 assert np.max(abs(np.linalg.norm(native[:,27:],axis=1)-native[:,21]))<1e-12
 assert np.all(body['warning']==0) and np.all(body['external']==0) and np.all(body['assist'][:,:6]==0)
 assert not body['assist'][np.rint(body['time']/DT)>=104000].any()
 p=ar(f/'PARAMETERS.npz');roots=ar(B/'IDENTITIES.npz')['root_ids'];assert np.array_equal(p['root_ids'],roots) and not p['source_offset'].any() and not p['threshold_offset'].any() and not p['z'].any()
 for key in ['We','Wi']:
  b=sparse.load_npz(B/('ACTUAL_'+key+'.npz'))
  for name in ['ACTUAL_','FINAL_']:
   w=sparse.load_npz(f/(name+key+'.npz'))
   for field in ['data','indices','indptr']:assert np.array_equal(getattr(b,field),getattr(w,field))
 c0=ar(f/'ACTUAL_COMPILED_INITIAL.npz');c1=ar(f/'ACTUAL_COMPILED_FINAL.npz');cref=ar(E/'COMPILED_MODEL.npz');assert len(c0)==len(c1)==469
 for k in c0:assert np.array_equal(c0[k],c1[k]) and np.array_equal(c0[k],cref[k]),k
 assert js(f/'MODEL_OBJECT_IDENTITY.json')['same_object']
 assert sha(f/'RUN/MODEL.xml')==sha(B/'RUN/MODEL.xml')
 dc=js(f/'ACTUAL_DECODER_CONFIG.json');assert dc['filter_tau_s']==.005 and dc['rate_for_full_control_hz']==100
 rates=np.zeros(90);pending=np.zeros(90);decay=math.exp(-.0005/.005);maxerr=0.;snap={}
 for j,sp in enumerate(ev):
  inst=np.array([math.fsum(float(sp[i]) for i in row['local_indices'])/len(row['local_indices'])/.0005 if row['local_indices'] else 0. for row in mapping]);rates=decay*rates+(1-decay)*inst
  if j%2:
   rec=records[j//2];assert np.array_equal(pending,rec['applied_command']);err=float(np.max(abs(rates-rec['decoder']['rates_hz'])));maxerr=max(maxerr,err);assert err<1e-12
   pending=np.clip(rates/100.,0,1);assert np.max(abs(pending-rec['decoded_next_command']))<1e-14
   assert rec['applied_source_interval']==j//2-1
  if j+1 in (1800,2200,2400):snap[j+1]=(rates.copy(),pending.copy())
 for role,end in [('FADE_START',1800),('FADE_END',2200),('FINAL',2400)]:
  st=ar(f/(role+'_NEURAL_COMPLETED.npz'));assert np.array_equal(st['spike_count'],ev[:end].sum(0))
  assert np.max(abs(st['decoder_rates']-snap[end][0]))<1e-12 and np.max(abs(st['pending']-snap[end][1]))<1e-14
  assert int(st['steps'])==end and not st['threshold_offset_mV'].any()
 # Actual retained complete network, pending and filter state across withdrawal.
 warm=ar(f/'PREWARM_STATE.npz');entry=ar(f/'RELEASE_ENTRY_STATE.npz');bound=ar(f/'RUN/NEURAL_BOUNDARY.npz')
 for k in warm:assert np.array_equal(warm[k],entry[k]),k
 for k in bound:assert np.array_equal(warm[k],bound[k]),k
 before=ar(f/'RUN/HANDOVER_BEFORE.npz');after=ar(f/'RUN/HANDOVER_AFTER.npz')
 for k in before:
  if k not in ('ctrl','qfrc_applied','xfrc_applied'):assert np.array_equal(before[k],after[k]),k
 assert not after['qfrc_applied'].any() and not after['xfrc_applied'].any()
 assert np.array_equal(after['ctrl'],warm['pending'])
 # Explicit independent ring reconstruction from full saved raster and fixed actual CSR.
 ring=np.zeros_like(warm['_ring']);ri=0
 targets={1800:'FADE_START_NEURAL_COMPLETED',2200:'PREWARM_STATE',2400:'RELEASE_STATE'}
 for j,sp in enumerate(ev):
  ring[ri]=sp;ri=(ri+1)%len(ring)
  if j+1 in targets:
   st=ar(f/(targets[j+1]+'.npz'));assert np.array_equal(st['_ring'],ring) and int(st['ring_pos'])==ri
 final=ar(f/'RELEASE_STATE.npz');assert np.array_equal(final['spike_count'],ev.sum(0)) and int(final['steps'])==2400
 exe=js(f/'EXECUTION.json');assert exe['error'] is None and exe['counters']['body_steps']==112000 and exe['counters']['neural']==2400 and exe['counters']['factory']==1 and exe['counters']['model_compiles']==2
 W=float(js(f/'RUN/GAINS.json')['weight']);nr=native[(native[:,26]==2)&(np.rint(native[:,0]/DT)>104000)];br=ar(C/'TRAIN_000_BASELINE/BODYLOSS_NATIVE.npz')['rows'][1:8001]
 assert len(nr)==len(br)==8000 and np.array_equal(np.rint(nr[:,0]/DT),np.rint(br[:,0]/DT))
 new=metrics(nr,W);baseout=metrics(br,W,True);fade=native[(native[:,26]==1)]
 end=native[(np.rint(native[:,0]/DT)==104000)&(native[:,26]==1)][0];fadeendbad=bool(end[7]>1e-8*W)
 new['root_displacement']=nr[-1,13:16]-after['qpos'][:3];baseout['root_displacement']=br[-1,11:14]-ar(B/'RUN/HANDOVER_AFTER.npz')['qpos'][:3]
 new['admission_passed']=bool(new['support_gate'] and not fadeendbad)
 result=dict(baseline=baseout,continuous_fade=new,fade=dict(first_events=js(f/'FIRST_NONFOOT_EVENTS.json'),positive_occupancy=float(np.mean(fade[:,7]>1e-8*W)),min_COMz=float(fade[:,12].min()),end_COMz=float(end[12]),end_foot_weight=float(end[1:7].sum()/W),end_positive_nonfoot_weight=float(end[7]/W),end_lastassistnorm=float(end[21]),end_multiplier=float(end[23]),end_multiplier_clock=float(end[22]),end_algorithm_phase=float((max(0,end[0]-.15)*2*math.pi*5)%(2*math.pi)),end_nonfoot_admission_failure=fadeendbad),h0=H0,weight=W,status='SUPPORT_PASS_NOT_WALKING' if new['admission_passed'] else 'SUPPORT_ADMISSION_FAILED',limitations=['No motor/sensory cuts or matched replay/passive controls in this one trial','No long duration, different initial states or material slip validation','Biological SI NOT_ESTABLISHED; algorithm5Hz cycle is not a physiological cycle','NF-supported displacement does not count as walking'])
 dump(O/'RESULT.json',result)
 dump(O/'INDEPENDENT_ARRAY_VERIFICATION.json',dict(passed=True,pid=os.getpid(),pure_arrays=True,model_compile=0,body_steps=0,neural_steps=0,training=0,prefix_exact=True,full_decoder_ticks=2400,records=1200,decoder_maxerror=maxerr,actual_multiplier_exact=True,actual_rawPD_times_multiplier_exact=True,all42torque_exact=True,CSR_initial_final_baseline_exact=True,compiled469_initial_final_baseline_exact=True,same_model_object=True,network_withdrawal_continuity_exact=True,baseline_native_reused=8000,phase_masks=True))
 print('INDEPENDENT_ARRAY_PASS',result['status'],flush=True)

def report():
 f=O/LABEL;result=js(O/'RESULT.json');verifyinfo=js(O/'INDEPENDENT_ARRAY_VERIFICATION.json');exe=js(f/'EXECUTION.json')
 dump(O/'RESOURCE_ACCOUNTING.json',dict(main=exe,new_baseline_body=0,new_baseline_neural=0,training=0,optimization=0,plasticity=0,verification=dict(model_compile=0,forward=0,body=0,neural=0),observer_extra_live_forward=0,scope='actual main runner body/neural/compile/factory counts; new native observer contactForce separately counted, original internal Jacobian/contactforce not globally hooked'))
 text='# S5 连续减辅接管：唯一固定200ms协议\n\n研究对照，未证明自主行走。原root888/实际WeWi*.2/offset0/原3470与initial、tau5ms/fullscale100/gain1、kp100clip50swing.45、dt12.5us/cmd1ms/sample.5ms保持。没有plasticity、训练、优化、神经参数改动。\n\n阶段预登记：0..1.1辅助不变，.2起神经在线；1.1..1.3仅原assistance_fade_s=.2线性乘实际42关节PD力矩，仍为辅助；1.3..1.4全部外置控制0为唯一自主评价窗。同body/net/decoder/rates/pending连续，无reset/锁姿/root平移/关节复制。原算法5Hz的一个周期只是参考时钟，非生理周期。旧20261028T087000Z有减辅方法，但停稳/神经预热时长/交接时间/模型池分支不同，不冒充首次方法或重做相同版本。\n\n物理依据为原S2突撤前COMvz+23.4899与足载.4716W，并非静态平衡；本轮只检验原参照一个周期内连续减负，不搜索phase/gain/长度。即使fade内先倒地仍完整运行预定1.4s，fade末非足承载直接准入失败。\n\n结果（所有对照严格同release100ms原生8000端点，固定h0=1.2072528224791905）：\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n实际运行与验证：\n\n```json\n'+json.dumps(dict(execution=exe,verification=verifyinfo),ensure_ascii=False,indent=2)+'\n```\n\n每12.5us1.0..1.4原生记录逐足signedFz/每contact正NF之和/aggregateNF/count/COM/root/ctrlmean²/实际assistnorm/42torque与相位，firstNF身份来自真实contacts。没有新增liveforward；原.5ms完整BODY和muscleforce、2400全网raster与1200因果窗保存。实际回调rawclippedPD、remain与scaledqfrc逐值核验，不能只修改显示记录。初末469compiled来自同实际model object、CSR原值/roots/offset0、完整撤辅network/pending连续及前1.1s原所有BODY/raster exact均独立新进程purearrays核。BODY在边界是poststep、interval completed decoder另存，multiplier用真实prestepclock，不移时迎合。\n\n没有同轮切断/重放/被动/长时与不同初态控制；即使短窗支撑通过也不能称walking。没有物质足滑证明，不以几何脚速度代替。原生单位生理SI NOT_ESTABLISHED。父review前未seal，科学源与历史包只读；无视频或目视通过声明。重要程度及完整路径见FILE_INVENTORY.json。\n'
 (O/'REPORT.zh-CN.md').write_text(text)
 dump(O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='All new scientific arrays, executable adapters, diffs and failure evidence required for review; no confirmed disposable output. PYTHONDONTWRITEBYTECODE prevents new pycache; old packages unchanged.'))
 dump(O/'FILE_INVENTORY.json',{str(p):dict(importance='重要辅助' if p.suffix in ('.md','.log','.diff') or 'LEDGER' in p.name or 'INVENTORY' in p.name or 'ACCOUNTING' in p.name else '核心必需',purpose='continuous fade evidence / source provenance / independent validation; unsealed pending parent review') for p in O.rglob('*') if p.is_file()})
 dump(O/'REVIEW_READY_HASHES.json',{str(p):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='REVIEW_READY_HASHES.json'})
 print('REPORT_READY_UNSEALED',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','verify','report']);args=p.parse_args();globals()[args.mode]()
