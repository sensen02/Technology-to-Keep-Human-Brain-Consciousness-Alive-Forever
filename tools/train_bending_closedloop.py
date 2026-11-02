"""S4 full-history physical bending imitation. New artifacts only; no default edits.
run: preregister -> unique fresh neutral exact gate -> 19 training -> one validation.
verify: separate process, saved arrays only, no neural/body integration.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib,time,inspect,traceback,math,argparse,dataclasses,shutil
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s4-bending-closedloop-training/20261102T042848Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
A=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')
T=Path('/home/sensen/Desktop/engine修复验收/s4-invariant-motion-target/20261102T023600Z/INVARIANT_BENDING_TARGETS.npz')
LEGS=('LF','LM','LH','RF','RM','RH');DT=.0000125
CONFIG=dict(duration=1.3,kp=100.,clip=50.,dt=DT,neural_command_dt_s=.001,neural_prewarm_s=1.1,swing_height=.45,sample_dt_s=.0005)
def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def js(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def csrhash(w):
 h=hashlib.sha256()
 for k in ('data','indices','indptr'):h.update(getattr(w,k).tobytes())
 return h.hexdigest()
def angles(points):
 u=points[...,:-2,:]-points[...,1:-1,:];v=points[...,2:,:]-points[...,1:-1,:]
 assert np.isfinite(points).all() and np.all(np.linalg.norm(u,axis=-1)>1e-12) and np.all(np.linalg.norm(v,axis=-1)>1e-12),'degenerate FK prediction'
 return np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),np.sum(u*v,axis=-1))
def groups():
 m=js(B/'RUN/MAPPING.json');roots=arr(B/'IDENTITIES.npz')['root_ids'];owners={}
 for row in m['rows']:
  assert row['leg'] in LEGS
  for i in row['local_indices']:owners.setdefault(int(i),set()).add(row['leg'])
 conflicts={str(i):sorted(v) for i,v in owners.items() if len(v)!=1}
 if conflicts:raise RuntimeError('crosslegshared conflict; no guessing: '+str(conflicts))
 g=np.full(len(roots),-1,int)
 for i,v in owners.items():g[i]=LEGS.index(next(iter(v)))
 assert len(owners)==204
 return m,roots,g

def prepare():
 assert not O.exists(),'Refuse existing evidence directory'
 m,roots,g=groups();O.mkdir(parents=True)
 paths=[Path(__file__),T,B/'RUN/MAPPING.json',B/'RUN/BODY.npz',B/'NEURAL_RASTER.npz',B/'ACTUAL_We.npz',B/'ACTUAL_Wi.npz',B/'IDENTITIES.npz']+[A/n for n in ['VERSION_CONTRACT.json','MN_POOLS_SNAPSHOT888.json','SENSORY_LOOKUP_SNAPSHOT888.json']]
 paths+=list((R/'engine').rglob('*.py'))+[R/'run_physical_walk_prewarm.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py']
 # Hash existing source, no copies of installed environments.
 paths+= [R/'GOAL_ROADMAP.zh-CN.md',R/'GOAL_TRACKER.zh-CN.md']
 targets=arr(T);assert targets['angles_rad'].shape==(37,6,2) and np.array_equal(targets['leg_names'],LEGS)
 np.savez_compressed(O/'PARAMETER_GROUPS.npz',root_ids=roots,leg_index=g,readout_indices=np.flatnonzero(g>=0),leg_names=LEGS)
 dump(O/'PARAMETER_GROUPS.json',dict(source=str(B/'RUN/MAPPING.json'),groups=[dict(leg=l,count=int((g==i).sum()),local_indices=np.flatnonzero(g==i),root_ids=roots[g==i]) for i,l in enumerate(LEGS)],crossleg_conflicts=[],decoder_manifest=m))
 dump(O/'PREREGISTRATION.json',dict(stage='S4 genuine local bending plus physical support; research training',head='7d311d3038cb8671e0b16624be644ddd9854bc5a',source_sha256={str(p):sha(p) for p in paths},config=CONFIG,graph='Original neutral default source graph actual We/Wi scaled*.2 identical round13. NT1/32 is separate failed historical control; not used in trials.',groups='204 actual manifest MN, unique one leg each; LF38 RF38 others32; 12 constants, no pool/decoder changes',parameter_order='z[0:6] log2 postMN incoming We AND Wi gains, z[6:12] threshold shifts /2mV; all [-1,1]',parameter_formula='We/Wi data *= exp(z_gain[leg]*ln2) on postMN CSR rows only; threshold=source_offset+2*ztheta[leg]; source elsewhere fixed; no edges/masks/EI/delay/intrinsics/initial state changes',gate='One fresh1.35s (.2no neural+1.1 online prewarm+.05 release), compare every same body sample to round13 at atol1e-10, raster first2300 and commands exact. Failure stops training, preserve evidence. No retrospective tolerance changes.',history='Every trial starts fresh original body and default resting SNN/zero filters/pending; parameters fixed from neural t0. Original same PD algorithm prewarm only, release SNN->original decoder->native90muscle. Never referencepose/control writes, CPG, PD on release or state swaps.',schedule=dict(gate_release_s=.05,training_release_s=.1,validation_release_s=.3,train_source_indices=list(range(13)),validation_source_indices=list(range(37)),train_source_frames=list(range(29,42)),validation_source_frames=list(range(29,66))),source_time='nominal120Hz, not measured acquisition times; nearest saved.5ms endpoint, <=.25ms; no interpolation/loop/warp/phase',validation='one selected parameters fresh full prewarm; extended same-trial prefix, not independent animal or biological holdout; original22/7/8 split unused',loss='Lmotion masked MSE radians^2 six legs2threepoint interiors at13/37source times; support each12.5us liveforward endpoint after release, exclude release boundary: height mean(((COMz-h0)/h0)^2), foot mean((signedFz/W-1)^2), nonfoot mean(sum_eachcontact positiveNFz/W), rotation .01*mean(relative_root_angle^2), ctrl .001*mean(ctrl^2), reg .001*sum(loggain^2+(theta/2)^2); total sum, h0/q0 neutral gate boundary fixed for all candidates',optimizer=dict(method='bounded SPSA',seed=4213,updates=6,a='.05/(k+1)^.602',c='.05/(k+1)^.101',gradient='(Lplus-Lminus)/(2*c)*delta, nominal c even projected perturbations; center=clip(z-a*g,-1,1)',selection='all successful training trials minimum total, exact tie earlier, no support re-selection'),planned=dict(gate=1,train=19,val=1,complete_trials=21,body_steps=2364000,neural_steps=50400,optimizer_updates=6),observer='Do not alter original liveforward/save cadence; native fields read immediately after original liveforward via original read-only body observer; no raw all-contact rows. Full q/v/act/ctrl/FK .5ms, packed full raster .5ms, MN204 events/filter .5ms and90pool/commands1ms, complete boundary/end/parameter identities.',frozen_h0q0='neutral exact gate release boundary, frozen before training',limitations='NOT_ESTABLISHED biological force/physical length scale; decreased total not walking; raw target tracking noise/unobservable twist/root/contact/Claw remain excluded',importance={'scientific arrays/code/prereg/verify':'核心必需','report/ledgers':'重要辅助'},seal='No seal before parent review; no tracker update or commit'))
 shutil.copyfile(__file__,O/'EXECUTION_SOURCE.py')
 print('PREREGISTERED',O,flush=True)

def trial(label,z,release):
 import mujoco
 import run_physical_walk_prewarm as runner
 import run_neural_muscle_closure as c
 import engine.embodied.continuous_handover as h
 import engine.embodied.runtime_pool_guard as guard
 from engine.neural_cond import ConductanceNetwork
 folder=O/label;folder.mkdir();z=np.asarray(z,float);assert z.shape==(12,) and np.all(abs(z)<=1)
 mfest,roots,g=groups();motor=np.flatnonzero(g>=0);gain=np.exp(z[:6]*math.log(2));theta=2*z[6:]
 origfactory=h._make_runtime;origcont=h.run_neural_continuous;origguard=guard.load_runtime_pool;origmap=c.channel_index_map;origpool=c.POOLS;origstep=ConductanceNetwork.step
 contract=js(A/'VERSION_CONTRACT.json');assert sha(A/'SENSORY_LOOKUP_SNAPSHOT888.json')==contract['sensory_lookup_sha256']
 origguard(A/'MN_POOLS_SNAPSHOT888.json',version_contract=contract)
 native=[];raster=[];mnfilters=[];netref=[];mnrate=np.zeros(204);decay=math.exp(-.0005/.005);counters=dict(factory=0,neural=0,native_contact_force=0,native_endpoints=0,body_steps=0,model_compiles=0)
 def factory(built,*args,**kw):
  net,enc,dec=origfactory(built,*args,**kw);counters['factory']+=1;assert counters['factory']==1 and net.steps==0
  assert np.array_equal(built['tier']['root_ids'],roots)
  assert args[0]==mfest
  for key in ('We','Wi'):
   w=getattr(net,key);ref=sparse.load_npz(B/('ACTUAL_'+key+'.npz'));assert csrhash(w)==csrhash(ref)
   row=np.repeat(np.arange(net.n),np.diff(w.indptr));scales=np.ones(net.n);scales[motor]=gain[g[motor]];w=w.copy();w.data*=scales[row];setattr(net,key,w)
   sparse.save_npz(folder/('ACTUAL_'+key+'.npz'),w)
  source=net.threshold_offset_mV.copy();offset=source.copy();offset[motor]=source[motor]+theta[g[motor]];offset.flags.writeable=False;net.threshold_offset_mV=offset
  np.savez_compressed(folder/'PARAMETERS.npz',z=z,gain=gain,theta_mV=theta,source_offset=source,threshold_offset=net.threshold_offset_mV,root_ids=roots,leg_index=g)
  dump(folder/'NETWORK_CONTRACT.json',dict(params=dataclasses.asdict(net.p),dt_ms=net.dt,delay_steps=net.delay_steps,csr_sha256={k:csrhash(getattr(net,k)) for k in ['We','Wi']},base_steps=0,only_post_MN_rows=True))
  np.savez_compressed(folder/'NETWORK_INITIAL.npz',**{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)},decoder_rates=dec.rates,pending=np.zeros(90))
  netref.append(net);return net,enc,dec
 def step(net,*a,**kw):
  nonlocal mnrate
  assert net is netref[0];sp=origstep(net,*a,**kw);raster.append(np.packbits(sp));mnrate=decay*mnrate+(1-decay)*sp[motor]/.0005;mnfilters.append(mnrate.copy());counters['neural']+=1;return sp
 def admitted(path,*a,**kw):return origguard(path,version_contract=contract)
 def channels(n,rs,**kw):return origmap(n,rs,lookup_path=A/'SENSORY_LOOKUP_SNAPSHOT888.json')
 def continuous(model,data,*a,**kw):
  observer=kw['body_observer'];isrelease=kw.get('runtime_state') is not None
  bids=np.array([[model.body(l+'Coxa').id,model.body(l+('Trochanter' if l[1]=='F' else 'Femur')).id,model.body(l+'Tibia').id,model.body(l+'Tarsus1').id] for l in LEGS]);thorax=model.body('Thorax').id
  names=[model.geom(i).name or '' for i in range(model.ngeom)];feet=np.array([next((j for j,l in enumerate(LEGS) if n.startswith(l) and 'Tars' in n),-1) for n in names]);planes=model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
  if not (folder/'MARKER_MAPPING.json').exists():dump(folder/'MARKER_MAPPING.json',dict(body_ids=bids,body_names=[[model.body(int(i)).name for i in row] for row in bids],definition='physical FK body origins; front composite Trochanter->Tibia; others Femur->Tibia; Claw excluded'))
  fk=[];fkt=[]
  def observe(mm,dd):
   observer(mm,dd)
   st=round(dd.time/DT)
   if st%40==0:fk.append(dd.xpos[bids].copy());fkt.append(float(dd.time))
   if not isrelease:return
   ff=np.zeros(6);nf=0.;nc=0;nfagg=0.;force=np.zeros(6)
   for j in range(dd.ncon):
    co=dd.contact[j];a1=int(co.geom1);a2=int(co.geom2)
    if not (planes[a1] or planes[a2]):continue
    mujoco.mj_contactForce(mm,dd,j,force);counters['native_contact_force']+=1
    fz=float((co.frame.reshape(3,3).T@force[:3])[2])*(1 if planes[a1] else -1);leg=feet[a2 if planes[a1] else a1]
    if leg<0:nf+=max(fz,0);nfagg+=fz;nc+=1
    else:ff[leg]+=fz
   native.append((float(dd.time),*ff,nf,nfagg,float(nc),float(dd.subtree_com[thorax,2]),*dd.qpos[:7],float(np.mean(dd.ctrl**2)),float(np.max(abs(dd.qfrc_applied[:6]))),))
   assert not dd.qfrc_applied.any() and not dd.xfrc_applied.any()
  kw['body_observer']=observe
  ret=origcont(model,data,*a,**kw)
  role='RELEASE' if isrelease else 'PREWARM';rt=ret['runtime_state'];net,enc,dec=rt['runtime']
  np.savez_compressed(folder/(role+'_STATE.npz'),**{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)},decoder_rates=dec.rates,pending=rt['pending'],steps=net.steps,ring_pos=net._ring_i,time_s=net.t_ms/1000)
  np.savez_compressed(folder/(role+'_FK.npz'),time=fkt,points=fk)
  if isrelease:
   np.savez_compressed(folder/'BODY_BOUNDARY.npz',**ret['handover_after_withdrawal'])
  return ret
 # Execute original runner with only output volume changes. Remove full12000 current JSON;
 # retain per-group physical sensory scalars, all events and complete decoder traces.
 text=inspect.getsource(runner.run);text=text.replace('retain_currents=True','retain_currents=False').replace("sensory_current_nA=np.array([r['sensory_current_nA'] for r in ret['records']])","sensory_current_nA=np.empty((0,int(built['n'])))")
 (folder/'RUNNER_ADAPTER.py').write_text(text);ns=dict(runner.__dict__);exec(compile(text,str(folder/'RUNNER_ADAPTER.py'),'exec'),ns)
 start=time.monotonic();error=None
 h._make_runtime=factory;h.run_neural_continuous=continuous;guard.load_runtime_pool=admitted;c.channel_index_map=channels;c.POOLS=A/'MN_POOLS_SNAPSHOT888.json';ConductanceNetwork.step=step
 try:
  ns['run'](folder/'RUN',**CONFIG,neural_duration=release)
 except BaseException:error=traceback.format_exc()
 finally:
  h._make_runtime=origfactory;h.run_neural_continuous=origcont;guard.load_runtime_pool=origguard;c.channel_index_map=origmap;c.POOLS=origpool;ConductanceNetwork.step=origstep
  np.savez_compressed(folder/'RASTER.npz',packed=np.asarray(raster,np.uint8),motor_indices=motor,MN_rates_halfstep_hz=mnfilters)
  np.savez_compressed(folder/'BODYLOSS_NATIVE.npz',rows=np.asarray(native,float))
  if (folder/'RUN/EXECUTION.json').exists():
   ec=js(folder/'RUN/EXECUTION.json');counters.update(body_steps=ec['counts']['body_step'],model_compiles=ec['counts']['model_load'],runner_forward=ec['counts']['forward'])
  counters['native_endpoints']=max(0,len(native)-1)
  dump(folder/'EXECUTION.json',dict(error=error,wall_s=time.monotonic()-start,counters=counters,release_s=release,z=z))
 if error:raise RuntimeError(error)
 assert len(native)==round(release/DT)+1 and counters['neural']==round((1.1+release)/.0005)
 print('TRIAL_COMPLETE',label,'seconds',round(time.monotonic()-start,2),'body',counters['body_steps'],'neural',counters['neural'],flush=True)
 return folder

def gate():
 f=trial('GATE_NEUTRAL_50MS',np.zeros(12),.05);b=arr(f/'RUN/BODY.npz');ref=arr(B/'RUN/BODY.npz');checks={};diff={}
 for k,v in b.items():
  want=ref[k][:len(v)];e=float(np.max(np.abs(v.astype(float)-want.astype(float)))) if v.size else 0.;checks[k]=bool(np.allclose(v,want,atol=1e-10,rtol=0));diff[k]=e
 sp=arr(f/'RASTER.npz')['packed'];orig=arr(B/'NEURAL_RASTER.npz')['spikes'][:2300];checks['raster_exact']=np.array_equal(sp,np.packbits(orig,axis=1))
 records=js(f/'RUN/PREWARM_NEURAL_RECORDS.json')+js(f/'RUN/NEURAL_RECORDS.json');old=js(B/'RUN/PREWARM_NEURAL_RECORDS.json')+js(B/'RUN/NEURAL_RECORDS.json')[:50]
 checks['commands_exact']=all(np.array_equal(r[k],s[k]) for r,s in zip(records,old) for k in ['applied_command','decoded_next_command']) and len(records)==1150
 dump(O/'EXACT_GATE.json',dict(passed=all(checks.values()),checks=checks,body_max_abs=diff,body_samples=len(b['time']),raster_ticks=len(sp),fresh_factory=True,parameter_interface_neutral=True,wall_s=js(f/'EXECUTION.json')['wall_s']))
 if not all(checks.values()):return False
 n=arr(f/'BODYLOSS_NATIVE.npz')['rows'][0];h0=float(n[10]);q0=n[14:18];assert h0>0 and abs(np.linalg.norm(q0)-1)<1e-10
 dump(O/'FROZEN_REFERENCE.json',dict(h0=h0,q0=q0,weight=js(f/'RUN/GAINS.json')['weight'],source='neutral fresh gate release boundary; frozen for all subsequent trials'))
 return True

def score(folder,nframes):
 ref=js(O/'FROZEN_REFERENCE.json');target=arr(T);fk=arr(folder/'RELEASE_FK.npz');times=target['time_relative_s'][:nframes];ix=np.array([np.argmin(abs(fk['time']-(1.3+t))) for t in times]);err=fk['time'][ix]-(1.3+times);assert abs(err).max()<=.00025+1e-10
 p=fk['points'][ix];ang=angles(p);mask=target['angle_valid'][:nframes];diff=ang-target['angles_rad'][:nframes];motion=float(np.mean(diff[mask]**2));rows=arr(folder/'BODYLOSS_NATIVE.npz')['rows'][1:]
 # Native columns: t, foot6, NFpositive, NFsigned, count, COMz, root7, ctrlmean2, rootassist.
 h=rows[:,10];quat=rows[:,14:18];dot=np.clip(abs(quat@np.asarray(ref['q0']))/np.linalg.norm(quat,axis=1),0,1);rot=2*np.arccos(dot)
 z=arr(folder/'PARAMETERS.npz')['z'];terms=dict(motion=motion,height=float(np.mean(((h-ref['h0'])/ref['h0'])**2)),foot=float(np.mean((rows[:,1:7].sum(1)/ref['weight']-1)**2)),nonfoot=float(np.mean(rows[:,7]/ref['weight'])),rotation=float(.01*np.mean(rot**2)),ctrl=float(.001*np.mean(rows[:,18])),reg=float(.001*np.sum((z[:6]*math.log(2))**2+z[6:]**2)))
 total=sum(terms.values());pos=rows[:,7]>0;can=rows[:,9]>0
 metrics=dict(first_candidate_s=float(rows[np.flatnonzero(can)[0],0]-1.3) if can.any() else None,first_positiveNFz_s=float(rows[np.flatnonzero(pos)[0],0]-1.3) if pos.any() else None,nonfoot_positive_occupancy=float(pos.mean()),nonfoot_candidate_occupancy=float(can.mean()),mean_foot_load=float(rows[:,1:7].sum(1).mean()/ref['weight']),mean_positiveNFz=float(rows[:,7].mean()/ref['weight']),COMz_min=float(h.min()),COMz_last=float(h[-1]),root_displacement=rows[-1,11:14]-arr(folder/'BODY_BOUNDARY.npz')['qpos'][:3],per_leg_bending_MSE=np.mean(diff**2,axis=(0,2)),support_loss=sum(terms[k] for k in ['height','foot','nonfoot','rotation','ctrl']),motion_loss=motion,walking_certified=False)
 records=js(folder/'RUN/NEURAL_RECORDS.json');cmd=np.array([r['applied_command'] for r in records]);metrics['release_saturated_command_elements']=int(np.count_nonzero(cmd>=1))
 np.savez_compressed(folder/'TARGET_FK_SAMPLES.npz',points=p,prediction_angles_rad=ang,target_angles_rad=target['angles_rad'][:nframes],mask=mask,source_frame=target['source_frame'][:nframes],source_nominal_time_s=times,physical_absolute_time_s=fk['time'][ix],time_error_s=err,FK_indices=ix)
 out=dict(label=folder.name,z=z,loss=total,terms=terms,metrics=metrics,nframes=nframes,failed=False);dump(folder/'SCORE.json',out);return out

def run():
 prepare();start=time.monotonic()
 if not gate():
  dump(O/'RESULT.json',dict(status='SCIENTIFIC_GATE_FAILURE_TRAINING_NOT_STARTED',train_trials=0,validation_trials=0));report();return 2
 print('EXACT_GATE_PASS',js(O/'EXACT_GATE.json')['wall_s'],flush=True)
 scores=[];history=[];z=np.zeros(12);rng=np.random.default_rng(4213)
 scores.append(score(trial('TRAIN_000_BASELINE',z,.1),13));dump(O/'TRAINING_SCORES.json',scores)
 for k in range(6):
  delta=rng.choice([-1.,1.],12);a=.05/(k+1)**.602;c=.05/(k+1)**.101;plus=np.clip(z+c*delta,-1,1);minus=np.clip(z-c*delta,-1,1)
  ps=score(trial('TRAIN_%03d_PLUS'%(3*k+1),plus,.1),13);scores.append(ps);dump(O/'TRAINING_SCORES.json',scores)
  ms=score(trial('TRAIN_%03d_MINUS'%(3*k+2),minus,.1),13);scores.append(ms);dump(O/'TRAINING_SCORES.json',scores)
  grad=(ps['loss']-ms['loss'])/(2*c)*delta;new=np.clip(z-a*grad,-1,1)
  history.append(dict(k=k,a=a,c=c,delta=delta,z_before=z,z_plus=plus,z_minus=minus,gradient=grad,z_after=new,projection='nominal c; projected perturbations explicitly saved'))
  dump(O/'SPSA_HISTORY.json',history);z=new
  cs=score(trial('TRAIN_%03d_CENTER'%(3*k+3),z,.1),13);scores.append(cs);dump(O/'TRAINING_SCORES.json',scores)
  print('UPDATE_COMPLETE',k+1,'best',min(s['loss'] for s in scores),'center',cs['loss'],flush=True)
 best=min(scores,key=lambda s:s['loss']);dump(O/'SELECTED_CANDIDATE.json',best)
 val=score(trial('VALIDATION_SELECTED_300MS',best['z'],.3),37)
 dump(O/'RESULT.json',dict(status='COMPLETED_NUMERICAL_TRAINING_NO_WALKING_CLAIM',baseline=scores[0],selected=best,validation=val,training_trials=len(scores),validation_trials=1,wall_s=time.monotonic()-start,all_trials=21))
 report();return 0

def report():
 result=js(O/'RESULT.json');gateinfo=js(O/'EXACT_GATE.json');executions={str(p.parent):js(p) for p in O.glob('*/EXECUTION.json')}
 totals={k:sum(e['counters'].get(k,0) for e in executions.values()) for k in ['body_steps','neural','model_compiles','runner_forward','native_contact_force','native_endpoints','factory']}
 dump(O/'RESOURCE_ACCOUNTING.json',dict(complete_trials=sum(e['error'] is None for e in executions.values()),totals=totals,trials=executions))
 text='# S4真实局部弯曲目标：完整预热历史闭环训练\n\n原中性默认图作为控制；低gain NT图最近真实身体失败，独立保留其原封包，不混用失败图继续全部训练。实际manifest的204MN各唯一属于六腿之一，无crosslegshared冲突。同池decoder绑定、原roots/mask/EI、delay/intrinsics、90muscle/42hinge/rootfree与原初态不变。仅12个常量：六腿MN入边We/Wi正gain、六腿readoutMN阈值shift。每trial从最初神经起动固定参数，完整真实身体反馈预热，释放无PD/CPG/目标姿态写入。\n\n中性fresh gate：\n\n```json\n'+json.dumps(gateinfo,ensure_ascii=False,indent=2)+'\n```\n\n结果：\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n训练首13个真实sourceframe29..41；唯一候选验证37帧29..65。仅名义120Hz时间、最近.5ms真实保存帧误差<=.25ms，无插值/循环/逐腿相位/时间warp。验证含训练前缀，非独立动物或生物holdout，旧22/7/8不再当本训练分割。局部弯曲不能监督root/方向/twist/Claw/足地真值，IKqpos不当真值。h0/q0固定使用通过gate的中性释放边界，不以候选低姿态改参考掩盖跌落。运动项与支撑项分列；total下降只可称训练数值改进，非足承载期间位移不算步行，接触翻转不是步。生物力/长度单位仍NOT_ESTABLISHED。\n\n实际资源：\n\n```json\n'+json.dumps(totals,ensure_ascii=False,indent=2)+'\n```\n\n所有trial的原完整prewarm/交接/末态、模型/原初始化与适配执行源、actualCSR矩阵、阈值/roots/分组、原rasterpacked/MN204filter、90pool/因果命令、每12.5us原生力/COMz/rootquat/ctrlmean²、.5ms完整q/v/act/FK、真实source/mask/time匹配均保存。未保存每dt大contactrows；loss直接从liveforward后原生字段读取，没有额外身体forward改调度。独立验证在新进程只重算数组，不再优化/积分；状态以INDEPENDENT_VERIFICATION.json为准。父review前未seal。\n\n重要程度：训练入口、预登记/分组、每trial科学数组/原执行源、目标来源/真实time与mask、选择/SPSA/独立核验为核心必需；本报告/资源与失败/清理台账为重要辅助。实际删除0：目前新增保留文件用于运行复核或失败溯源，没有确认可弃文件；未复制venv源码，未改tracker/旧源/旧模型/默认graph/guard/旧封包，没有提交。\n'
 (O/'REPORT.zh-CN.md').write_text(text)
 dump(O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='No confirmed disposable task artifacts; preserved executable adapter and scientific provenance; no venv copies'))
 dump(O/'FILE_INVENTORY.json',{str(p):dict(importance='重要辅助' if p.name in ['REPORT.zh-CN.md','RESOURCE_ACCOUNTING.json','CLEANUP_LEDGER.json','FILE_INVENTORY.json'] else '核心必需',purpose='Training scientific evidence or execution provenance') for p in O.rglob('*') if p.is_file()})

def verify():
 # Independent process array-only, no imports of mujoco/runner/network.
 pr=js(O/'PREREGISTRATION.json')
 for p,h in pr['source_sha256'].items():assert sha(p)==h,p
 assert js(O/'EXACT_GATE.json')['passed']
 target=arr(T);recomputed=[];maxerr=0.;roots=arr(O/'PARAMETER_GROUPS.npz')['root_ids'];g=arr(O/'PARAMETER_GROUPS.npz')['leg_index'];motor=np.flatnonzero(g>=0)
 folders=sorted(O.glob('TRAIN_*'))+[O/'VALIDATION_SELECTED_300MS']
 for folder in folders:
  saved=js(folder/'SCORE.json');p=arr(folder/'PARAMETERS.npz');z=p['z'];assert np.array_equal(p['root_ids'],roots) and np.array_equal(p['leg_index'],g)
  assert np.all(abs(z)<=1) and np.array_equal(p['gain'],np.exp(z[:6]*math.log(2))) and np.array_equal(p['theta_mV'],2*z[6:])
  expect=p['source_offset'].copy();expect[motor]+=2*z[6:][g[motor]];assert np.array_equal(expect,p['threshold_offset'])
  for key in ['We','Wi']:
   base=sparse.load_npz(B/('ACTUAL_'+key+'.npz'));actual=sparse.load_npz(folder/('ACTUAL_'+key+'.npz'));scale=np.ones(len(roots));scale[motor]=p['gain'][g[motor]];rr=np.repeat(np.arange(len(roots)),np.diff(base.indptr))
   assert np.array_equal(actual.indices,base.indices) and np.array_equal(actual.indptr,base.indptr) and np.array_equal(actual.data,base.data*scale[rr]) and np.all(actual.data>0)
  initial=arr(folder/'NETWORK_INITIAL.npz');assert np.all(initial['v']==-65)
  for k in ['ge','gi','adaptation','refractory_until_ms','spike_count','_ring','decoder_rates','pending']:assert not initial[k].any(),k
  assert np.array_equal(initial['threshold_offset_mV'],expect)
  samples=arr(folder/'TARGET_FK_SAMPLES.npz');n=saved['nframes'];assert np.array_equal(samples['source_frame'],np.arange(29,29+n)) and np.array_equal(samples['mask'],target['angle_valid'][:n])
  assert np.array_equal(samples['source_nominal_time_s'],target['time_relative_s'][:n]) and np.max(abs(samples['time_error_s']))<=.00025+1e-10
  # Independently scalar atan2 for both real target and model.
  for field,points in [('prediction_angles_rad',samples['points']),('target_angles_rad',target['observed_keypoints'][:n])]:
   out=np.empty((n,6,2))
   for fi in range(n):
    for li in range(6):
     for ai in range(2):
      u=points[fi,li,ai]-points[fi,li,ai+1];v=points[fi,li,ai+2]-points[fi,li,ai+1];cross=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);out[fi,li,ai]=math.atan2(math.sqrt(math.fsum(float(x*x) for x in cross)),math.fsum(float(x*y) for x,y in zip(u,v)))
   assert np.max(abs(out-samples[field]))<2e-14
  # Recalculate all loss terms without writing trial outputs.
  ref=js(O/'FROZEN_REFERENCE.json');rows=arr(folder/'BODYLOSS_NATIVE.npz')['rows'][1:];q=rows[:,14:18];q0=np.array(ref['q0']);dot=abs(np.sum(q*q0,axis=1))/np.linalg.norm(q,axis=1);angle=2*np.arccos(np.clip(dot,0,1));d=samples['prediction_angles_rad']-samples['target_angles_rad']
  terms=dict(motion=float(np.mean(d[samples['mask']]**2)),height=float(np.mean(((rows[:,10]-ref['h0'])/ref['h0'])**2)),foot=float(np.mean((rows[:,1:7].sum(1)/ref['weight']-1)**2)),nonfoot=float(np.mean(rows[:,7]/ref['weight'])),rotation=float(.01*np.mean(angle**2)),ctrl=float(.001*np.mean(rows[:,18])),reg=float(.001*np.sum((z[:6]*math.log(2))**2+z[6:]**2)))
  for k,v in terms.items():e=abs(v-saved['terms'][k]);maxerr=max(maxerr,e);assert e<1e-12,(folder,k,e)
  assert abs(sum(terms.values())-saved['loss'])<1e-12 and np.all(rows[:,19]==0)
  # Original decoder on saved events, independent scalar/array filter. No runtime.
  events=np.unpackbits(arr(folder/'RASTER.npz')['packed'],axis=1,count=12000).astype(bool);records=js(folder/'RUN/PREWARM_NEURAL_RECORDS.json')+js(folder/'RUN/NEURAL_RECORDS.json');manifest=js(B/'RUN/MAPPING.json');rates=np.zeros(90);prev=np.zeros(90);decay=math.exp(-.0005/.005);mn=np.zeros(204);mn_saved=arr(folder/'RASTER.npz')['MN_rates_halfstep_hz']
  for j,sp in enumerate(events):
   inst=np.array([float(np.mean(sp[row['local_indices']]))/.0005 if row['local_indices'] else 0 for row in manifest['rows']]);rates=decay*rates+(1-decay)*inst;mn=decay*mn+(1-decay)*sp[motor]/.0005;assert np.array_equal(mn,mn_saved[j])
   if j%2:
    k=j//2;record=records[k];assert np.array_equal(prev,record['applied_command']);assert np.max(abs(rates-np.array(record['decoder']['rates_hz'])))<1e-12
    pending=np.clip(rates/100.,0,1);assert np.max(abs(pending-record['decoded_next_command']))<1e-14;prev=np.array(record['decoded_next_command'])
  for name,end in [('PREWARM',2200),('RELEASE',len(events))]:
   state=arr(folder/(name+'_STATE.npz'));assert np.array_equal(state['spike_count'],events[:end].sum(0));assert np.array_equal(state['threshold_offset_mV'],expect)
  if folder.name.startswith('TRAIN'):recomputed.append(saved)
 # Seed and SPSA replay, exact order and earliest selection.
 scores=js(O/'TRAINING_SCORES.json');assert len(scores)==19;rng=np.random.default_rng(4213);z=np.zeros(12)
 for k,h in enumerate(js(O/'SPSA_HISTORY.json')):
  delta=rng.choice([-1.,1.],12);a=.05/(k+1)**.602;c=.05/(k+1)**.101;assert np.array_equal(delta,h['delta']) and a==h['a'] and c==h['c']
  assert np.array_equal(z,h['z_before']);plus=np.clip(z+c*delta,-1,1);minus=np.clip(z-c*delta,-1,1)
  assert np.array_equal(plus,scores[3*k+1]['z']) and np.array_equal(minus,scores[3*k+2]['z']);grad=(scores[3*k+1]['loss']-scores[3*k+2]['loss'])/(2*c)*delta;z=np.clip(z-a*grad,-1,1);assert np.array_equal(z,scores[3*k+3]['z']) and np.array_equal(grad,h['gradient'])
 best=min(scores,key=lambda s:s['loss']);assert best==js(O/'SELECTED_CANDIDATE.json')
 selected=O/best['label'];val=O/'VALIDATION_SELECTED_300MS';prefix={}
 for filename,fields in [('RASTER.npz',['packed','MN_rates_halfstep_hz']),('BODYLOSS_NATIVE.npz',['rows']),('RUN/BODY.npz',None),('PREWARM_STATE.npz',None),('PARAMETERS.npz',None)]:
  first=arr(selected/filename);second=arr(val/filename)
  for k in fields or list(first):
   v=first[k];w=second[k];ok=np.array_equal(v,w[:len(v)]) if v.ndim and v.shape!=w.shape else np.array_equal(v,w);assert ok,(filename,k);prefix[filename+':'+k]=True
 for p,h in pr['source_sha256'].items():assert sha(p)==h,p
 dump(O/'INDEPENDENT_VERIFICATION.json',dict(passed=True,pid=os.getpid(),method='Fresh process pure saved arrays/scalar threepoint atan2/native terms/CSR groups/SPSA/decoder/validation full prewarm prefix; zero body/neural integration or optimizer',training_losses=19,validation_losses=1,max_loss_term_error=maxerr,selection_exact=True,valprefix_exact=prefix,source_hashes_unchanged=True))
 print('INDEPENDENT_VERIFICATION_PASS',maxerr,flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','verify','report']);mode=parser.parse_args().mode
 sys.exit(globals()[mode]() or 0)
