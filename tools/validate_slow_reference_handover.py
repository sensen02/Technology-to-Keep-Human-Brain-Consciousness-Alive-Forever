"""S2/S5 one phase-matched 2.5Hz engineering reference, fresh continuous history.
No default edits, no neutral rerun, no sealing before parent review.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,math,inspect,difflib,hashlib,argparse,dataclasses,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'tools')]
D=Path('/home/sensen/Desktop/engine修复验收')
O=D/'s5-slow-reference-handover/20261102T165227Z'
B=D/'s5-banc888-physical-handover/20261101T220018Z'
C=D/'s4-bending-closedloop-training/20261102T042848Z'
E=D/'s2-early-muscle-support/20261102T053639Z'
M=D/'s2-reference-muscle-operating-range/20261102T155412Z'
LABEL='SLOW_2P5HZ_PHASE_MATCHED';DT=.0000125;END=1.25;H0=1.2072528224791905
CONFIG=dict(duration=1.25,kp=100.,clip=50.,dt=DT,neural_command_dt_s=.001,neural_prewarm_s=1.05,swing_height=.45,sample_dt_s=.0005)
RUNTIME=[];MODEL_IDS=[];NATIVE=[];EVENTS={};BOUNDARIES={};ASSIST=[];LAST={};VELOCITY=[]
def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def js(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rep(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def diff(a,b,old,new):return ''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=str(old),tofile=str(new)))
def snapshot_net(net,dec,pending):return dict(**{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)},decoder_rates=dec.rates.copy(),pending=np.asarray(pending).copy(),steps=net.steps,ring_pos=net._ring_i,time_s=net.t_ms/1000)
def runner_source(text):
 assert text.count('*2*np.pi*5')==4
 text=text.replace('*2*np.pi*5','*2*np.pi*2.5')
 text=rep(text,'frequency_hz=5.','frequency_hz=2.5')
 return text

def instrument_source():
 import validate_continuous_assistance_fade as f
 s=inspect.getsource(f.instrument)
 changes=[('if step<80000:return','if step<16000:return'),('remain=float(np.clip((1.3-clock)/.2,0,1))','remain=1.  # abrupt withdrawal, no fade'),('if round(clock/DT)>=79999:','if round(clock/DT)>=16000:'),('phase=0 if step<=88000 else (1 if not release else 2)','phase=0 if not release else 2'),("key='release' if release else ('fade' if step>88000 else 'prefade')","key='release' if release else 'warmup'"),('float(dd.time)-1.3','float(dd.time)-1.25'),('if step in (88000,104000,112000):','if step in (68000,100000,108000):'),("role=('FADE_START' if step==88000 else 'FADE_END' if step==104000 else 'FINAL')","role=('CYCLE_START' if step==68000 else 'WARMUP_END' if step==100000 else 'FINAL')"),('*2*math.pi*5','*2*math.pi*2.5'),('if st in (88000,104000,112000):','if st in (68000,100000,108000):'),("role='FADE_START' if st==88000 else 'FADE_END' if st==104000 else 'FINAL'","role='CYCLE_START' if st==68000 else 'WARMUP_END' if st==100000 else 'FINAL'")]
 for a,b in changes:s=rep(s,a,b)
 s=rep(s,'  NATIVE.append(', '  VELOCITY.append(dd.qvel[va].copy())\n  NATIVE.append(')
 compile(s,str(O/'OBSERVER_EXECUTION_SOURCE.py'),'exec');return s

def prepare():
 import train_bending_closedloop as old
 import run_physical_walk_prewarm as runner
 assert not O.exists() or {p.name for p in O.iterdir()} <= {'HISTORY_DUPLICATE_CHECK.json','PREPARE_001_FAILED_SOURCE.py'},'refuse existing scientific evidence'
 O.mkdir(parents=True,exist_ok=True)
 # Bounded metadata census: every historical preregistration (skip bulky event JSON), no simulation.
 # pathlib enumerates only names; JSON is processed as data rather than shell inspection.
 history=[];matches=[];bad=[]
 history_paths=[]
 for root in [D/'s5-physical-warmup',B,D/'s4-banc888-full-prewarm-replay',D/'s5-continuous-assistance-fade',D/'s4-bending-closedloop-training']:
  history_paths.extend(root.glob('**/PREREGISTRATION.json'))
 for p in history_paths:
  if O in p.parents:continue
  try:
   x=js(p)
   def frequencies(v):
    if isinstance(v,dict):
     for k,w in v.items():
      if ('frequency' in k or k in ('gait_hz','reference_hz')) and isinstance(w,(int,float)):yield (k,w)
      yield from frequencies(w)
    elif isinstance(v,list):
     for w in v:yield from frequencies(w)
   fs=list(frequencies(x))
   if fs:history.append(dict(path=str(p),sha256=sha(p),frequencies=fs))
   if any(v==2.5 for k,v in fs):matches.append(dict(path=str(p),configuration=x))
  except Exception as ex:bad.append(dict(path=str(p),error=str(ex)))
 dump(O/'HISTORY_DUPLICATE_CHECK.json',dict(metadata_records_with_frequency=history,candidates_2p5=matches,parse_failures=bad,same_root888_model3470_fullhistory_found=False,scope='bounded known physicalwarmup/root888 fullhistory/fade/training preregistrations plus repository tracker/reports; not exhaustive desktop; 2.5Hz FFT bins are not reference experiments',prior_search_timeout='all-json grep timed out at30s, zero scientific calls'))
 assert not matches, 'Review historical 2.5Hz metadata before any new run'
 text=inspect.getsource(old.trial);original=text
 text=rep(text,"row=np.repeat(np.arange(net.n),np.diff(w.indptr));scales=np.ones(net.n);scales[motor]=gain[g[motor]];w=w.copy();w.data*=scales[row];setattr(net,key,w)","assert np.array_equal(gain,np.ones(6))  # original CSR object retained")
 text=rep(text,"source=net.threshold_offset_mV.copy();offset=source.copy();offset[motor]=source[motor]+theta[g[motor]];offset.flags.writeable=False;net.threshold_offset_mV=offset","source=net.threshold_offset_mV.copy();assert not source.any() and not theta.any()")
 text=rep(text,"netref.append(net);return net,enc,dec","dump(folder/'ACTUAL_DECODER_CONFIG.json',dataclasses.asdict(dec.config))\n  RUNTIME.append((net,enc,dec));netref.append(net);return net,enc,dec")
 start=text.index(' def continuous(model,data,*a,**kw):');end=text.index(' # Execute original runner',start)
 text=text[:start]+" def continuous(model,data,*a,**kw):\n  return instrument(origcont,model,data,a,kw,folder,counters,netref)\n"+text[end:]
 anchor=" (folder/'RUNNER_ADAPTER.py').write_text(text);ns=dict(runner.__dict__);exec(compile(text,str(folder/'RUNNER_ADAPTER.py'),'exec'),ns)"
 text=rep(text,anchor," original_runner_text=text; text=runner_source(text)\n (folder/'RUNNER_LITERAL.diff').write_text(diff(original_runner_text,text,R/'run_physical_walk_prewarm.py',folder/'RUNNER_ADAPTER.py'))\n"+anchor)
 text=rep(text,"np.savez_compressed(folder/'BODYLOSS_NATIVE.npz',rows=np.asarray(native,float))","np.savez_compressed(folder/'NATIVE.npz',rows=np.asarray(NATIVE,float),joint_velocity=np.asarray(VELOCITY,float))\n  np.savez_compressed(folder/'ACTUAL_ASSIST_CALLBACK.npz',rows=np.asarray(ASSIST,float))\n  dump(folder/'FIRST_NONFOOT_EVENTS.json',EVENTS);dump(folder/'BOUNDARY_CLOCKS.json',BOUNDARIES)")
 text=rep(text,"counters['native_endpoints']=max(0,len(native)-1)","counters['native_endpoints']=len(NATIVE)")
 text=rep(text,"assert len(native)==round(release/DT)+1 and counters['neural']==round((1.1+release)/.0005)","assert len(NATIVE)==92002 and counters['neural']==2300")
 obs=instrument_source();runtext=runner_source(inspect.getsource(runner.run))
 for name,s in [('TRIAL_EXECUTION_SOURCE.py',text),('OBSERVER_EXECUTION_SOURCE.py',obs),('RUNNER_PREFLIGHT.py',runtext)]:
  compile(s,str(O/name),'exec');(O/name).write_text(s)
 (O/'TRIAL_ORIGINAL_SOURCE.py').write_text(original)
 (O/'TRIAL_LITERAL.diff').write_text(diff(original,text,R/'tools/train_bending_closedloop.py',O/'TRIAL_EXECUTION_SOURCE.py'))
 (O/'EXECUTION_SOURCE.py').write_bytes(Path(__file__).read_bytes())
 dump(O/'NATIVE_SCHEMA.json',dict(columns=['time','foot_LF_Fz','foot_LM_Fz','foot_LH_Fz','foot_RF_Fz','foot_RM_Fz','foot_RH_Fz','sum_each_positive_NFz','signed_aggregate_NFz','NFcount','COMx','COMy','COMz','rootx','rooty','rootz','rootqw','rootqx','rootqy','rootqz','ctrl_mean2','actual_assist_norm','multiplier_clock','multiplier','root_qfrc_max','xfrc_max','phase']+['assist_torque_'+str(i) for i in range(42)],phase=dict(algorithm_assisted=0,zeroassist_release=2),joint_velocity='native same observer qvel6:48,92002x42',clock='original post-forward; torque is previous prestep callback; boundary duplicated warm before/release after; no additional liveforward',coverage='native .2..1.35 every12.5us;0...2 original BODY every.5ms only; warmup firstNF below.2 limited to saved BODY grid',callback_columns=['prestep_body_time','actual_multiplier']+['actual42torque_'+str(i) for i in range(42)]+['root_qfrc_max','xfrc_max']))
 paths=[Path(__file__),R/'tools/train_bending_closedloop.py',R/'tools/validate_continuous_assistance_fade.py',R/'run_physical_walk_prewarm.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py',B/'RUN/BODY.npz',B/'NEURAL_RASTER.npz',B/'RUN/MODEL.xml',B/'RUN/GAIT.npz',B/'ACTUAL_We.npz',B/'ACTUAL_Wi.npz',B/'IDENTITIES.npz',C/'TRAIN_000_BASELINE/BODYLOSS_NATIVE.npz',E/'COMPILED_MODEL.npz',M/'AUDIT.npz',M/'COMPILED_ACTUAL_ARRAYS.npz']+list((R/'engine').rglob('*.py'))+[old.A/n for n in ['VERSION_CONTRACT.json','MN_POOLS_SNAPSHOT888.json','SENSORY_LOOKUP_SNAPSHOT888.json']]+[O/n for n in ['TRIAL_EXECUTION_SOURCE.py','OBSERVER_EXECUTION_SOURCE.py','RUNNER_PREFLIGHT.py']]
 dump(O/'PREREGISTRATION.json',dict(head='6a03a02afbcab3672017bf81f3a58354f75ee893',stage='S2/S5 engineering reference speed physical contrast research only',config=CONFIG,release_s=.1,frequency_hz=2.5,baseline_frequency_hz=5.,phase='same initial phi0 and terminal3pi/2; old(1.3-.15)*5=5.75/new(1.25-.15)*2.5=2.75cycles; final cycles old1.1..1.3 new.85..1.25',controls='speed AND mandatory phase-matching warmup50ms difference; neural1.05 vs1.1, not identical full neural history; same source spatial amplitudes/tripod/profile/initial/model3470/root888/WeWi*.2/offset0/seed0/kp100clip50swing.45/tau5ms100Hzgain1;hold.15ramp.15...25 unchanged; no fade',single_candidate=True,new_neutral=False,plasticity=False,training=False,optimization=False,stop='run to1.35 even behavior fail; stop only numerical problem',prefix='only BODY t<=.15 first301samples bitwise exact; after.15 reference ramp differs; no400prefix/raster equality claim',support=dict(horizon_s=.1,deadband_weight=1e-8,mean_foot_min=.8,min_height_ratio=.8,h0=H0,release_clock_end=1.25,boundary_positive_NF_gate_failure=True),mechanism='one original model compile maximum101 saved forwards, candidate.85+4ms*k, baseline401cache index4*k; normalized declared phase, no dynamicpairedclamp, exact savedforce truth; native units biological calibration NOT_ESTABLISHED',budget=dict(body=108000,neural=2300,compile_live=2,factory=1,compile_offline=1,forward_offline=101,extra_liveforward=0),source_sha256={str(p):sha(p) for p in paths}))
 dump(O/'FAILURE_LEDGER.json',dict(failures=[dict(stage='prepare001',exit_code=1,reason='observer insertion matched three spaces but original has two; zero scientific calls',source=str(O/'PREPARE_001_FAILED_SOURCE.py'))],operational_search_timeout='one broad all-JSON grep30s; no scientific call; subsequent bounded metadata census successful'))
 print('PREREGISTERED',O,flush=True)

def run():
 import train_bending_closedloop as old
 pr=js(O/'PREREGISTRATION.json')
 for p,h in pr['source_sha256'].items():assert sha(p)==h,p
 scope=dict(old.__dict__);scope.update(globals());scope['__name__']='slow_execution'
 exec(compile((O/'OBSERVER_EXECUTION_SOURCE.py').read_text(),str(O/'OBSERVER_EXECUTION_SOURCE.py'),'exec'),scope)
 exec(compile((O/'TRIAL_EXECUTION_SOURCE.py').read_text(),str(O/'TRIAL_EXECUTION_SOURCE.py'),'exec'),scope)
 scope['trial'](LABEL,np.zeros(12),.1)
 dump(O/'SOURCE_HASHES_AFTER_RUN.json',{p:sha(p) for p in pr['source_sha256']})
 print('UNIQUE_CANDIDATE_COMPLETE',flush=True)

def mechanism():
 import mujoco as mj
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 from scipy.interpolate import CubicSpline
 from audit_reference_muscle_operating_range import terms
 assert not (O/'MECHANISM.npz').exists(),'refuse scientific overwrite'
 f=O/LABEL;b=ar(f/'RUN/BODY.npz');base=ar(M/'AUDIT.npz');compiled=ar(M/'COMPILED_ACTUAL_ARRAYS.npz')
 calls=dict(model_compile=0,forward=0,muscleGain_actual=0,muscleGain_zero_velocity=0,muscleBias=0,body=0,neural=0,contactForce=0,optimization=0)
 dump(O/'OFFLINE_PREREGISTRATION.json',dict(candidate_BODY_indices=list(range(1700,2501,8)),baseline_AUDIT_indices=list(range(0,401,4)),max_compile=1,max_forward=101,formula_source=str(R/'tools/audit_reference_muscle_operating_range.py'),native_scalar_calls_each=9090,baseline_all_fields_reused_exact=True))
 try:
  m=_load_mjcf(str(B/'RUN/MODEL.xml')).compile();calls['model_compile']+=1
  for k,v in compiled.items():assert np.array_equal(v,getattr(m,k)),k
  ix=np.arange(1700,2501,8);bi=np.arange(0,401,4);assert len(ix)==101
  out={k:v[ix] for k,v in b.items()};out['source_indices']=ix;out['baseline_source_indices']=base['source_indices'][bi];out['baseline_AUDIT_indices']=bi
  collected={k:[] for k in ['length','velocity','force','gain','gain_zero_velocity','bias','moment','qfrc_actuator']}
  for i in ix:
   d=mj.MjData(m)
   for dst,src in [('qpos','qpos'),('qvel','qvel'),('act','activation'),('ctrl','command'),('qfrc_applied','assist'),('xfrc_applied','external')]:getattr(d,dst)[:]=b[src][i]
   d.time=b['time'][i];mj.mj_forward(m,d);calls['forward']+=1
   moment=np.zeros((90,48))
   if d.actuator_moment.shape==moment.shape:moment[:]=d.actuator_moment
   else:
    for u in range(90):
     st=int(d.moment_rowadr[u]);n=int(d.moment_rownnz[u]);moment[u,d.moment_colind[st:st+n]]=d.actuator_moment[st:st+n]
   gain=np.array([mj.mju_muscleGain(d.actuator_length[u],d.actuator_velocity[u],m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_gainprm[u,:9]) for u in range(90)]);calls['muscleGain_actual']+=90
   zero=np.array([mj.mju_muscleGain(d.actuator_length[u],0.,m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_gainprm[u,:9]) for u in range(90)]);calls['muscleGain_zero_velocity']+=90
   bias=np.array([mj.mju_muscleBias(d.actuator_length[u],m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_biasprm[u,:9]) for u in range(90)]);calls['muscleBias']+=90
   for k,v in dict(length=d.actuator_length.copy(),velocity=d.actuator_velocity.copy(),force=d.actuator_force.copy(),gain=gain,gain_zero_velocity=zero,bias=bias,moment=moment,qfrc_actuator=d.qfrc_actuator.copy()).items():collected[k].append(v)
  out.update({k:np.asarray(v) for k,v in collected.items()})
  tt=np.array([[terms(out['length'][t,u],out['velocity'][t,u],m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_gainprm[u]) for u in range(90)] for t in range(101)])
  for j,k in enumerate(['L0','LT','normalized_fiber_length','normalized_fiber_velocity','FL','FV','effective_Fmax','formula_gain','formula_bias']):out[k]=tt[:,:,j]
  gait=ar(B/'RUN/GAIT.npz');phase=np.maximum(0,out['time']-.15)*2*np.pi*2.5%(2*np.pi);out['phase_rad']=phase
  out['reference_joint_target']=CubicSpline(gait['phase'],gait['qpos_hinges'],axis=0,bc_type='periodic')(phase)
  out['reference_joint_error']=out['qpos'][:,gait['qpos_addresses']]-out['reference_joint_target']
  out['force_length_zeroed']=abs(out['gain_zero_velocity'])<=1e-12;out['force_velocity_zeroed']=(abs(out['gain'])<=1e-12)&~out['force_length_zeroed'];out['force_velocity_suppressed']=(abs(out['gain'])<=.001*abs(out['gain_zero_velocity']))&~out['force_length_zeroed']
  for k,v in base.items():
   if v.ndim and len(v)==401:out['baseline_'+k]=v[bi]
  errors=dict(saved_force=float(np.max(abs(out['force']-out['muscle_force']))),gain_formula=float(np.max(abs(out['gain']-out['formula_gain']))),force_formula=float(np.max(abs(out['gain']*out['activation']+out['bias']-out['muscle_force']))),moment=float(np.max(abs(np.einsum('tuv,tu->tv',out['moment'],out['force'])-out['qfrc_actuator']))),phase=float(np.max(abs(np.exp(1j*phase)-np.exp(1j*base['phase_rad'][bi])))))
  assert errors['saved_force']==0 and max(errors.values())<1e-9,errors
  np.savez_compressed(O/'MECHANISM.npz',**out);dump(O/'MECHANISM_EXECUTION.json',dict(calls=calls,errors=errors,exit_code=0,compiled469_reuse=True))
  print('MECHANISM_COMPLETE',errors,flush=True)
 except Exception:
  dump(O/'MECHANISM_FAILURE.json',dict(calls=calls,error=traceback.format_exc()));raise

def metrics(rows,W,baseline=False):
 height=rows[:,10] if baseline else rows[:,12];foot=rows[:,1:7].sum(1);nf=rows[:,7];mask=nf>1e-8*W;tt=rows[:,0]-(1.3 if baseline else END)
 gates=dict(no_nonfoot=not mask.any(),mean_foot=bool(foot.mean()/W>=.8),min_height=bool(height.min()/H0>=.8))
 return dict(endpoints=len(rows),first_positive_nonfoot_s=float(tt[np.flatnonzero(mask)[0]]) if mask.any() else None,nonfoot_occupancy=float(mask.mean()),mean_foot_weight=float(foot.mean()/W),mean_positive_nonfoot_weight=float(nf.mean()/W),min_height_ratio=float(height.min()/H0),min_COMz=float(height.min()),final_COMz=float(height[-1]),gates=gates,support_gate=all(gates.values()),walking_certified=False)

def verify():
 from scipy import sparse
 pr=js(O/'PREREGISTRATION.json');f=O/LABEL
 for p,h in pr['source_sha256'].items():assert sha(p)==h,p
 # Independently reconstruct literal runner change from source, including all physical frequency factors.
 import ast
 source=(R/'run_physical_walk_prewarm.py').read_text();tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run');orig='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
 expected=orig.replace('retain_currents=True','retain_currents=False').replace("sensory_current_nA=np.array([r['sensory_current_nA'] for r in ret['records']])","sensory_current_nA=np.empty((0,int(built['n'])))").replace('*2*np.pi*5','*2*np.pi*2.5').replace('frequency_hz=5.','frequency_hz=2.5')
 assert (f/'RUNNER_ADAPTER.py').read_text()==expected
 body=ar(f/'RUN/BODY.npz');base=ar(B/'RUN/BODY.npz');na=ar(f/'NATIVE.npz');native=na['rows'];tap=ar(f/'ACTUAL_ASSIST_CALLBACK.npz')['rows'];step=np.rint(native[:,0]/DT).astype(int)
 assert native.shape==(92002,69) and na['joint_velocity'].shape==(92002,42) and tap.shape==(84000,46)
 assert np.isfinite(native).all() and np.isfinite(na['joint_velocity']).all()
 for k in body:assert np.array_equal(body[k][:301],base[k][:301]),k
 assert np.max(abs(body['time'][:301]-np.arange(301)*.0005))<1e-12
 assert np.array_equal(step[:84001],np.arange(16000,100001)) and np.array_equal(step[84001:],np.arange(100000,108001))
 release=native[:,26]==2;assert release.sum()==8001
 assert np.all(native[:,24:26]==0) and np.all(native[release,21]==0) and np.all(native[release,23]==0) and np.all(native[release,27:]==0)
 assert np.all(tap[:,1]==1) and not tap[:,44:].any()
 # Warm initial observation at .2 precedes first tap; all subsequent native42 exactly correspond to preceding callback.
 assert np.array_equal(native[1:84001,27:],tap[:,2:44]) and np.array_equal(native[1:84001,22],tap[:,0])
 assert np.max(abs(np.linalg.norm(native[:,27:],axis=1)-native[:,21]))<1e-12
 assert np.all(body['warning']==0) and np.all(body['external']==0) and not body['assist'][:,:6].any()
 assert not body['assist'][np.rint(body['time']/DT)>=100000].any()
 p=ar(f/'PARAMETERS.npz');assert np.array_equal(p['root_ids'],ar(B/'IDENTITIES.npz')['root_ids']) and not p['source_offset'].any() and not p['threshold_offset'].any() and not p['z'].any()
 for key in ['We','Wi']:
  ref=sparse.load_npz(B/('ACTUAL_'+key+'.npz'))
  for prefix in ['ACTUAL_','FINAL_']:
   w=sparse.load_npz(f/(prefix+key+'.npz'))
   for k in ['data','indices','indptr']:assert np.array_equal(getattr(w,k),getattr(ref,k))
 cc=ar(f/'ACTUAL_COMPILED_INITIAL.npz');cf=ar(f/'ACTUAL_COMPILED_FINAL.npz');cr=ar(E/'COMPILED_MODEL.npz');assert len(cc)==469
 for k in cc:assert np.array_equal(cc[k],cf[k]) and np.array_equal(cc[k],cr[k]),k
 assert js(f/'MODEL_OBJECT_IDENTITY.json')['same_object'] and sha(f/'RUN/MODEL.xml')==sha(B/'RUN/MODEL.xml')
 for k,v in ar(f/'RUN/GAIT.npz').items():assert np.array_equal(v,ar(B/'RUN/GAIT.npz')[k]),k
 dc=js(f/'ACTUAL_DECODER_CONFIG.json');assert dc['filter_tau_s']==.005 and dc['rate_for_full_control_hz']==100
 ev=np.unpackbits(ar(f/'RASTER.npz')['packed'],axis=1,count=12000).astype(bool);assert ev.shape==(2300,12000)
 records=js(f/'RUN/PREWARM_NEURAL_RECORDS.json')+js(f/'RUN/NEURAL_RECORDS.json');assert len(records)==1150
 mapping=js(B/'RUN/MAPPING.json')['rows'];rates=np.zeros(90);pending=np.zeros(90);decay=math.exp(-.0005/.005);maxerr=0;snaps={}
 for j,sp in enumerate(ev):
  inst=np.array([sum(int(sp[i]) for i in row['local_indices'])/len(row['local_indices'])/.0005 if row['local_indices'] else 0. for row in mapping]);rates=decay*rates+(1-decay)*inst
  if j%2:
   rec=records[j//2];assert np.array_equal(pending,rec['applied_command']);maxerr=max(maxerr,float(np.max(abs(rates-rec['decoder']['rates_hz']))));pending=np.clip(rates/100,0,1);assert np.max(abs(pending-rec['decoded_next_command']))<1e-14;assert rec['applied_source_interval']==j//2-1
  if j+1 in [1300,2100,2300]:snaps[j+1]=(rates.copy(),pending.copy())
 assert maxerr<1e-12
 for role,n in [('CYCLE_START',1300),('WARMUP_END',2100),('FINAL',2300)]:
  st=ar(f/(role+'_NEURAL_COMPLETED.npz'));assert int(st['steps'])==n and np.array_equal(st['spike_count'],ev[:n].sum(0));assert np.max(abs(st['decoder_rates']-snaps[n][0]))<1e-12 and np.max(abs(st['pending']-snaps[n][1]))<1e-14
 initial=ar(f/'NETWORK_INITIAL.npz');assert np.all(initial['v']==-65)
 for k in ['ge','gi','adaptation','refractory_until_ms','spike_count','_ring','decoder_rates','pending']:assert not initial[k].any(),k
 warm=ar(f/'PREWARM_STATE.npz');entry=ar(f/'RELEASE_ENTRY_STATE.npz');bound=ar(f/'RUN/NEURAL_BOUNDARY.npz')
 for k in warm:assert np.array_equal(warm[k],entry[k]) and np.array_equal(warm[k],bound[k]),k
 ring=np.zeros_like(warm['_ring']);ri=0
 for j,sp in enumerate(ev):
  ring[ri]=sp;ri=(ri+1)%len(ring)
  if j+1 in [1300,2100,2300]:
   name={1300:'CYCLE_START_NEURAL_COMPLETED',2100:'PREWARM_STATE',2300:'RELEASE_STATE'}[j+1];st=ar(f/(name+'.npz'));assert np.array_equal(st['_ring'],ring) and int(st['ring_pos'])==ri
 before=ar(f/'RUN/HANDOVER_BEFORE.npz');after=ar(f/'RUN/HANDOVER_AFTER.npz')
 for k in before:
  if k not in ['ctrl','qfrc_applied','xfrc_applied']:assert np.array_equal(before[k],after[k]),k
 assert not after['qfrc_applied'].any() and not after['xfrc_applied'].any() and np.array_equal(after['ctrl'],warm['pending'])
 # Every BODY command is derived from completed causal window, holding its applied pending value.
 # Non-boundary complete-window BODY rows precede consume; release initial is pending after consume.
 for i,t in enumerate(body['time']):
  st=round(t/DT)
  if st<16000:assert not body['command'][i].any();continue
  if st==100000:want=snaps[2100][1]
  elif st==16000:want=np.zeros(90)
  else:
   wi=max(0,(st-16000-1)//80);want=np.asarray(records[wi]['applied_command'])
  assert np.array_equal(body['command'][i],want),(i,t)
 exe=js(f/'EXECUTION.json');assert exe['error'] is None and exe['counters']['body_steps']==108000 and exe['counters']['neural']==2300 and exe['counters']['model_compiles']==2 and exe['counters']['factory']==1
 W=float(js(f/'RUN/GAINS.json')['weight']);nr=native[release][1:];br=ar(C/'TRAIN_000_BASELINE/BODYLOSS_NATIVE.npz')['rows'][1:8001];assert len(nr)==len(br)==8000
 assert np.array_equal(np.rint((nr[:,0]-1.25)/DT),np.rint((br[:,0]-1.3)/DT))
 new=metrics(nr,W);old=metrics(br,W,True);end=native[release][0];bad=bool(end[7]>1e-8*W);new['boundary_already_NF']=bad;new['admission_passed']=bool(new['support_gate'] and not bad)
 # All shared .5ms native fields directly match saved full-body observations, keeping release role.
 for i,t in enumerate(body['time']):
  st=round(t/DT)
  if st<16000:continue
  candidates=np.flatnonzero((step==st)&(release==(st>=100000)));assert len(candidates)==1
  row=native[candidates[0]]
  assert np.array_equal(row[10:13],body['com'][i]) and np.array_equal(row[13:20],body['qpos'][i,:7]);assert np.max(abs(row[1:7]-body['foot_force'][i,:,2]))<1e-12
 # Mechanism formula arithmetic independent process; native functions used only in saved offline run.
 mech=ar(O/'MECHANISM.npz');sourceaudit=ar(M/'AUDIT.npz');mi=mech['source_indices'];bi=mech['baseline_AUDIT_indices'];assert np.array_equal(mi,np.arange(1700,2501,8)) and np.array_equal(bi,np.arange(0,401,4))
 for k in body:assert np.array_equal(mech[k],body[k][mi]),k
 for k,v in sourceaudit.items():
  if v.ndim and len(v)==401:assert np.array_equal(mech['baseline_'+k],v[bi]),k
 assert np.max(abs(mech['force']-mech['muscle_force']))==0
 def summary(a,prefix=''):
  g=a[prefix+'gain'];z=a[prefix+'gain_zero_velocity'];act=a[prefix+'activation'];vel=a[prefix+'normalized_fiber_velocity'];valid=abs(z)>1e-12
  return dict(samples=101,muscle_samples=9090,FVzero=int(((abs(g)<=1e-12)&valid).sum()),FLzero=int((~valid).sum()),relativecrit=int(((abs(g)<=.001*abs(z))&valid).sum()),nativeV_min=float(vel.min()),nativeV_max=float(vel.max()),activation_mean=float(act.mean()),joint_speed_RMS=float(np.sqrt(np.mean(a[prefix+'qvel'][:,6:]**2))),algorithm_target_tracking_RMS=float(np.sqrt(np.mean(a[prefix+'reference_joint_error']**2))),assist_L1_mean_auxiliary100=float(abs(a[prefix+'assist'][:100,6:]).sum(1).mean()),joint_target='engineering retarget; not animal ground truth')
 cycle=native[(step>=68000)&(step<100000)&~release]
 result=dict(status='SUPPORT_PASS_NOT_WALKING' if new['admission_passed'] else 'SUPPORT_ADMISSION_FAILED',baseline=old,candidate=new,firstNF=js(f/'FIRST_NONFOOT_EVENTS.json'),warmup_boundary=dict(time=END,foot_weight=float(end[1:7].sum()/W),NFpositive_weight=float(end[7]/W),COM=end[10:13],phase_rad=float(((END-.15)*2*math.pi*2.5)%(2*math.pi))),candidate_last_cycle_native=dict(endpoints=len(cycle),mean_foot_weight=float(cycle[:,1:7].sum(1).mean()/W),NFpositive_occupancy=float((cycle[:,7]>1e-8*W).mean()),COM_peak_to_peak=np.ptp(cycle[:,10:13],axis=0),assist_L1_mean=float(abs(cycle[:,27:]).sum(1).mean()),joint_velocity_RMS=float(np.sqrt(np.mean(na['joint_velocity'][(step>=68000)&(step<100000)&~release]**2)))),mechanism_candidate=summary(mech),mechanism_baseline=summary(mech,'baseline_'),h0=H0,weight=W,history=dict(candidate_prewarm_ticks=2100,baseline_prewarm_ticks=2200,candidate_total_ticks=2300,baseline_total_ticks=2400,strict_same_history=False),walking_certified=False,biological_units='NOT_ESTABLISHED')
 dump(O/'RESULT.json',result)
 dump(O/'INDEPENDENT_ARRAY_VERIFICATION.json',dict(passed=True,pid=os.getpid(),pure_arrays=True,compile=0,body=0,neural=0,source_literal_all_frequency_duration_checked=True,first301BODY_exact=True,full_decoder_ticks=2300,decoder_maxerror=maxerr,CSR_initial_final_baseline_exact=True,compiled469_sameobject_initial_final_baseline_exact=True,network_body_withdrawal_continuity_exact=True,baseline_native_reused=8000,mechanism_baseline101_exact=True,mechanism_savedforce_exact=True,zeroassist_release=True))
 print('PURE_ARRAY_PASS',result['status'],flush=True)

def report():
 f=O/LABEL;result=js(O/'RESULT.json');exe=js(f/'EXECUTION.json');off=js(O/'MECHANISM_EXECUTION.json')
 dump(O/'RESOURCE_ACCOUNTING.json',dict(main=exe,offline=off,new_baseline_body=0,new_baseline_neural=0,plasticity=0,training=0,optimization=0,extra_liveforward=0,scope='runner forward counted; new observer contactForce and COMJac separately; original internal Jacobian/contactForce not globally hooked'))
 (O/'REPORT.zh-CN.md').write_text('# S2/S5 工程参考速度有限物理对照\n\n研究结果，不是成品或行走认证。唯一新候选2.5Hz对原5Hz，工程参考速度不是实测动物频率；不改spatial amplitude、tripod、profile或默认模型/肌肉/神经/decoder/guard。STDP仅保留假设，未实施。\n\n为避免末相位混杂，固定同initialphi0与terminal3π/2：新warmup1.25s(神经从.2起1.05s)、突撤100ms至1.35；原1.3s(神经1.1s)突撤100ms复用TRAIN000/原第13native。控制变量包含速度与因相位匹配必需的预热时长50ms差，不称唯有频率变化或严格同神经历史；不按结果挑相位。ramp.15...25不变。原source两处phase和两处velocity均真实修改，有完整literal diff与fresh纯数组source核验。\n\n相位归一101点：原401cache按0,4,...400，新BODY按1700,1708,...2500(4ms)。declaredphase匹配，不是dynamic paired clamp。baseline直接复用savedsourceidentity；candidate离线原模型1compile101forward，nativeactualgain/vel0/FL/FV/act/moment/force保存，savedmuscleforce精确。模型469既有证据复用，不作第三份全量compiled重算。\n\n结果：\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n支撑固定原h0=1.2072528224791905、horizon.1、deadband1e-8W、meanfoot>=.8W、minheight>=.8h0，release clock为1.25。若warmup末已NF直接gatefail，不把趴地称支撑；完整跑到底。firstNF保存实际geom/body/dist/force身份，margin内正dist不称穿透。native.2..1.35每12.5us记录signedfoot6/eachpositiveNF/COM/root/ctrlmean²/实际42assist及jointvelocity；之前0...2只原.5msBODY，不谎称native事件精度。原.5msBODY/raster/filter/applied/next/初边末neuralstate/actualCSR初末/同model469初末均保存。仅原t<=.15的301BODY逐位exact；原400prefix不相同。observer无额外liveforward。\n\n无新neutral、训练、优化、可塑性、phasewarp或真实动作目标评分。没有长时/切断/重复初态试验；短窗支撑通过也不能称walking。绝对生物单位NOT_ESTABLISHED。父review前未seal，无图片/视频或目视通过声明。\n')
 dump(O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='All scientific arrays/executable adapters/failure provenance needed; no confirmed disposable output; -B prevents new pycache'))
 dump(O/'FILE_INVENTORY.json',{str(p):dict(importance='重要辅助' if p.suffix in ['.md','.log','.diff'] or any(x in p.name for x in ['LEDGER','INVENTORY','ACCOUNTING']) else '核心必需',purpose='slowreference scientific evidence/source/verification, pending parent review') for p in O.rglob('*') if p.is_file()})
 dump(O/'REVIEW_READY_HASHES.json',{str(p):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='REVIEW_READY_HASHES.json'})
 print('REPORT_READY_UNSEALED',flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','mechanism','verify','report']);args=parser.parse_args();globals()[args.mode]()
