"""One research-only 2s full body, fixed added-edge 1/32, no optimizer.
run -> fresh-process pure-array verify -> parent review -> seal exactly once.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,hashlib,inspect,argparse,dataclasses,math
import numpy as np
from scipy import sparse
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s5-low-gain-nt-physical/20261102T024247Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
G=Path('/home/sensen/Desktop/engine修复验收/s3-sourced-nt-candidate-graph/20261102T004326Z')
C=Path('/home/sensen/Desktop/engine修复验收/s4-sourced-nt-gain-calibration/20261102T020329Z')
H=Path('/home/sensen/Desktop/engine修复验收/s4-calibrated-nt-full-history/20261102T023600Z')
A=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')
FIT=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-closedloop-parameter-fit/20261101T222509Z')
GAIN=.03125
CONFIG=dict(duration=1.3,kp=100.,clip=50.,dt=.0000125,neural_duration=.7,neural_command_dt_s=.001,neural_prewarm_s=1.1,swing_height=.45,sample_dt_s=.0005)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')
def exact(a,b):return all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ['data','indices','indptr'])
def matrices():
 return {k:(sparse.load_npz(B/('ACTUAL_'+k+'.npz'))+sparse.load_npz(G/('ADDED_'+k+'.npz'))*GAIN).tocsr() for k in ['We','Wi']}
def prepare():
 assert not O.exists(),'Refuse existing output directory'
 assert js(C/'SELECTION_TABLE.json')['selected_gain']==GAIN
 assert js(G/'RESULT.json')['runtime_allowed']==False
 mats=matrices()
 for k in mats:assert exact(mats[k],sparse.load_npz(H/('CANDIDATE_ACTUAL_'+k+'.npz')))
 assert sparse.load_npz(G/'ADDED_We.npz').nnz==17264 and sparse.load_npz(G/'ADDED_Wi.npz').nnz==0
 inputs=[Path(__file__),R/'tools/run_banc888_physical_handover.py',R/'tools/verify_banc888_physical_handover.py',C/'SELECTION_TABLE.json',G/'RESULT.json',G/'PREREGISTRATION.json',G/'INDEPENDENT_VERIFICATION.json',G/'SELECTED_SOURCE_PROVENANCE.json',G/'IDENTITIES_AND_SELECTED.npz',B/'IDENTITIES.npz']
 inputs += [f/(prefix+k+'.npz') for f,prefix in [(B,'ACTUAL_'),(G,'ADDED_'),(G,'CANDIDATE_'),(H,'CANDIDATE_ACTUAL_')] for k in ['We','Wi']]
 inputs += [A/n for n in ['VERSION_CONTRACT.json','MN_POOLS_SNAPSHOT888.json','SENSORY_LOOKUP_SNAPSHOT888.json']]
 inputs += list(FIT.glob('*COMPILED*.npz'))
 for p,h in js(G/'RESULT.json')['source_sha256'].items():assert sha(p)==h,p
 inputs += [Path(p) for p in js(G/'RESULT.json')['source_sha256']]
 # Raw/prediction identities are frozen by the original graph preregistration and sealed manifest.
 for f in [G,C]:
  manifest=js(f/'MANIFEST_SHA256.json');manifest=manifest.get('files',manifest)
  for p,item in manifest.items():assert sha(p)==(item if isinstance(item,str) else item['sha256']),p
  inputs.append(f/'MANIFEST_SHA256.json')
 O.mkdir(parents=True)
 dump(O/'EXPERIMENT_GRAPH_CONTRACT.json',dict(scope='research_only_not_default_or_production',authorization='Direct human continue explicitly authorizes this one complete research body experiment',base=str(B),added=str(G),gain=GAIN,graph_formula='actual neutral round13 baseWe + .03125*ADDED_We; Wi same',added_excitatory_edges=17264,root_count=12000,original_candidate_runtime_allowed=False,production_registration=False,pool_sensory_contract=str(A/'VERSION_CONTRACT.json'),raw_prediction_provenance=js(G/'PREREGISTRATION.json'),candidate_identity=js(G/'RESULT.json'),original_graph_validation=js(G/'INDEPENDENT_VERIFICATION.json'),source_sha256={str(p):sha(p) for p in inputs},matrix_exact_full_history=True,biological_validity='NOT_ESTABLISHED'))
 dump(O/'PREREGISTRATION.json',dict(stage='S1/S3/S4/S5 research candidate load/timing, not imitation training',config=CONFIG,seed=0,baseline=str(B),graph_contract=str(O/'EXPERIMENT_GRAPH_CONTRACT.json'),graph_contract_sha256=sha(O/'EXPERIMENT_GRAPH_CONTRACT.json'),only_variable='17264 added excitatory edges *.03125 from first neural initialization; original baseWe/Wi *.2; neutral ConductanceParams offset0; no learned scalar',factory='original fresh base net steps0 -> fresh candidate; same original encoder/decoder/body/rates/activation/warmstart/pending across boundary; no reset',schedule='0-.2 no neural; .2-1.3 algorithm-assisted real-body neural prewarm;1.3-2 neural only',invariants='3470 original model asset loader/initial state/native frames/native bounds/no sourcepose or IK or bending loss/control; roots12000/newroot888 sensory/MN74mapped67complete7partial16unresolved66distinctpool',primaries=['first nonfoot candidate and positiveNFz both native12.5us and shared.5ms','56000 release endpoint nonfoot occupancy','signed footFz/W mean,last and per-foot','COM height/displacement/root displacement','no assistance,mocap,externalforce,rootlock,qposwrite'],events='Contact flips are not walking steps; no >=4feet gait gate; absent event null with >=.7s right censor',execution='Exactly one new2s fullbody even if behavior fails; stop first nonfinite/warning only; no baseline rerun/no gain scan/no extension/no optimizer',validation='Independent new process pure arrays all3600ticks/1800windows/rates/applied/bodycommand/delay and initial-boundary-final accumulation/actualmatrixdelta/400prefix/zeroassist/allnativeforces/timing/old1400endpoints/source/compiled469',planned=dict(body_steps=160000,neural_steps=3600,total_networks=2,active_networks=1,tier_builds=1,model_compiles=2,training=0,optimizer=0),limitations='No independent sensory/muscle truth; no physiology validation. Even foot support requires parent10/30s/controls/video. Absolute biological units NOT_ESTABLISHED.',cleanup='No confirmed disposable files, do not copy venv source'))
 print('PREREGISTERED',O,flush=True)
def adapted():
 import run_banc888_physical_handover as w
 w.OUT=O;w.CONFIG=CONFIG
 text=inspect.getsource(w.run)
 replacements={' prepare()':' pass # independent research preregistration frozen before body load','(160000,3600,1,1,1)':'(160000,3600,2,1,1)','paths.update([Path(__file__),':"paths.update([R/'tools/run_banc888_physical_handover.py',O/'EXPERIMENT_GRAPH_CONTRACT.json',O/'PREREGISTRATION.json',Path(__file__),"}
 for a,b in replacements.items():assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b)
 w.R=R;w.O=O
 (O/'REUSED_WRAPPER_EXECUTION.py').write_text(text)
 dump(O/'WRAPPER_ADAPTER.json',dict(original=str(R/'tools/run_banc888_physical_handover.py'),original_sha256=sha(R/'tools/run_banc888_physical_handover.py'),replacements=replacements,execution_sha256=sha(O/'REUSED_WRAPPER_EXECUTION.py'),reason='Only output, frozen provenance and two allocated network count change; all original observer/live scheduler unchanged'))
 w.__file__=str(Path(__file__).resolve());exec(compile(text,str(O/'REUSED_WRAPPER_EXECUTION.py'),'exec'),w.__dict__)
 return w
def run():
 prepare()
 from engine.embodied.runtime_pool_guard import load_runtime_pool
 contract=js(A/'VERSION_CONTRACT.json');assert sha(A/'SENSORY_LOOKUP_SNAPSHOT888.json')==contract['sensory_lookup_sha256']
 load_runtime_pool(A/'MN_POOLS_SNAPSHOT888.json',version_contract=contract)
 dump(O/'PREFLIGHT.json',dict(passed=True,before_any_model_load=True,pool_guard_original=True,graph_contract_sha256=sha(O/'EXPERIMENT_GRAPH_CONTRACT.json'),sensory_sha256=sha(A/'SENSORY_LOOKUP_SNAPSHOT888.json')))
 w=adapted()
 import engine.embodied.continuous_handover as h
 from engine.neural_cond import ConductanceNetwork,ConductanceParams
 import run_physical_walk_prewarm as runner
 import mujoco
 factory=h._make_runtime;old_step=ConductanceNetwork.step;old_loader=runner._load_mjcf
 active=base=encoder=decoder=None;factory_calls=active_steps=0;compiled=[]
 def make(built,*args,**kwargs):
  nonlocal active,base,encoder,decoder,factory_calls
  factory_calls+=1;assert factory_calls==1
  base,encoder,decoder=factory(built,*args,**kwargs)
  assert base.steps==0 and base.t_ms==0 and not base.threshold_offset_mV.any()
  assert dataclasses.asdict(base.p)==dataclasses.asdict(ConductanceParams())
  mats=matrices()
  for k in mats:
   assert exact(getattr(base,k),sparse.load_npz(B/('ACTUAL_'+k+'.npz')))
   assert exact(mats[k],sparse.load_npz(H/('CANDIDATE_ACTUAL_'+k+'.npz')))
   sparse.save_npz(O/('BASE_'+k+'.npz'),getattr(base,k))
  active=ConductanceNetwork(base.n,ConductanceParams(),dt_ms=base.dt)
  active.set_connectivity(mats['We'],mats['Wi'])
  for k in mats:sparse.save_npz(O/('CANDIDATE_'+k+'.npz'),getattr(active,k))
  roots=np.asarray(built['tier']['root_ids']);assert np.array_equal(roots,arr(B/'IDENTITIES.npz')['root_ids'])
  np.savez_compressed(O/'NETWORK_INITIAL.npz',**{k:v.copy() for k,v in vars(active).items() if isinstance(v,np.ndarray)},root_ids=roots,decoder_rates=decoder.rates.copy(),pending=np.zeros(90),steps=active.steps,t_ms=active.t_ms)
  dump(O/'NETWORK_IDENTITY.json',dict(params=dataclasses.asdict(active.p),dt_ms=active.dt,delay_steps=active.delay_steps,base_steps_at_injection=base.steps,base_time_at_injection=base.t_ms,base_id=id(base),active_id=id(active),encoder_id=id(encoder),decoder_id=id(decoder),same_encoder_decoder=True,offset0=True))
  return active,encoder,decoder
 def step(net,*args,**kwargs):
  nonlocal active_steps
  assert net is active and base.steps==0 and not net.threshold_offset_mV.any()
  active_steps+=1;return old_step(net,*args,**kwargs)
 class Loader:
  def __init__(self,obj):self.obj=obj
  def compile(self,*args,**kwargs):
   model=self.obj.compile(*args,**kwargs)
   arrays={k:getattr(model,k).copy() for k in dir(model) if isinstance(getattr(model,k,None),np.ndarray)}
   np.savez_compressed(O/('COMPILED_%d.npz'%len(compiled)),**arrays)
   compiled.append(dict(arrays=len(arrays),nq=model.nq,nv=model.nv,nu=model.nu,neq=model.neq,nmocap=model.nmocap,nmesh=model.nmesh,joint0_type=int(model.jnt_type[0]),muscles=int(np.count_nonzero(model.actuator_dyntype==mujoco.mjtDyn.mjDYN_MUSCLE))))
   assert model.nu==90 and model.nmesh==71 and model.nmocap==0 and model.neq==0
   return model
 def loader(*a,**kw):return Loader(old_loader(*a,**kw))
 h._make_runtime=make;ConductanceNetwork.step=step;runner._load_mjcf=loader
 try:w.run(A/'VERSION_CONTRACT.json',A/'MN_POOLS_SNAPSHOT888.json',A/'SENSORY_LOOKUP_SNAPSHOT888.json')
 finally:
  h._make_runtime=factory;ConductanceNetwork.step=old_step;runner._load_mjcf=old_loader
  if active is not None:
   for k in ['We','Wi']:sparse.save_npz(O/('FINAL_'+k+'.npz'),getattr(active,k))
   np.savez_compressed(O/'NETWORK_FINAL_IDENTITY.npz',root_ids=arr(O/'NETWORK_INITIAL.npz')['root_ids'],threshold_offset_mV=active.threshold_offset_mV)
  dump(O/'ACTUAL_RUNTIME_AUDIT.json',dict(factory_calls=factory_calls,active_steps=active_steps,base_steps=None if base is None else base.steps,active_steps_final=None if active is None else active.steps,total_networks=int(base is not None)+int(active is not None),active_networks=int(active is not None),compiled=compiled,model_compiles=len(compiled),tier_builds=int(factory_calls>0),training=0,optimizer=0,guard_calls_total=1+int((O/'EXECUTION.json').exists())+(js(O/'EXECUTION.json')['counters']['guard_calls'] if (O/'EXECUTION.json').exists() else 0)))
def native(folder):
 keys=['step','kind','nonfoot_count','nonfoot_force','foot_force','com','root']
 d={k:[] for k in keys}
 for chunk in js(folder/'CHUNKS.json'):
  z=arr(chunk['file'])
  for k in d:d[k].append(z[k])
 return {k:np.concatenate(v) for k,v in d.items()}
def metrics(z,weight,native=True):
 mask=(z['kind']==0)&(z['step']>104000) if native else np.arange(len(z['time']))>2600
 step=z['step'][mask] if native else np.flatnonzero(mask)*40
 out=dict(endpoint_bins=int(mask.sum()),duration_s=.7)
 for field,label in [('nonfoot_count','candidate'),('nonfoot_force','positive_NFz')]:
  values=z[field][mask]>0 if field=='nonfoot_count' else z[field][mask,2]>0
  ix=np.flatnonzero(values)
  out[label]=dict(count=int(values.sum()),first_s=float((step[ix[0]]-104000)*.0000125) if len(ix) else None,right_censored=not bool(len(ix)),lower_bound_s=.7 if not len(ix) else None)
 f=z['foot_force'][mask,:,2]/weight
 com=z['com'][mask];root=z['root'] if native else z['qpos'][:,:7]
 boundary=np.flatnonzero((z['step']==104000)&(z['kind']==1))[0] if native else 2600
 out.update(foot_Fz_over_W_mean=float(f.sum(1).mean()),foot_Fz_over_W_last=float(f[-1].sum()),per_foot_mean=f.mean(0),per_foot_last=f[-1],positive_nonfoot_Fz_over_W_mean=float(np.maximum(z['nonfoot_force'][mask,2],0).mean()/weight),COM_height_min=float(com[:,2].min()),COM_height_last=float(com[-1,2]),COM_displacement=com[-1]-z['com'][boundary],root_displacement=root[mask][-1,:3]-root[boundary,:3],walking_certified=False)
 return out
def verify():
 # Existing independent contact-force auditor: only baseline location/label adapters.
 p=R/'tools/verify_banc888_physical_handover.py';text=p.read_text()
 replacements={"P=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')":'P=Path('+repr(str(O))+')',"B=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')":'B=Path('+repr(str(B/'RUN'))+')',"N=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-native-recording/20261101T191552Z')":'N=Path('+repr(str(B))+')',"('old121',oldbody)":"('neutral_root888',oldbody)"}
 for a,b in replacements.items():assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b)
 (O/'ARRAY_AUDITOR_EXECUTION.py').write_text(text)
 ns={'__name__':'array_only_auditor'};exec(compile(text,str(O/'ARRAY_AUDITOR_EXECUTION.py'),'exec'),ns);ns['main']()
 graph=js(O/'EXPERIMENT_GRAPH_CONTRACT.json')
 for p,h in graph['source_sha256'].items():assert sha(p)==h,p
 initial=arr(O/'NETWORK_INITIAL.npz');finalid=arr(O/'NETWORK_FINAL_IDENTITY.npz');refid=arr(B/'IDENTITIES.npz')
 assert np.array_equal(initial['root_ids'],refid['root_ids']) and np.array_equal(finalid['root_ids'],refid['root_ids'])
 assert not initial['threshold_offset_mV'].any() and not finalid['threshold_offset_mV'].any()
 for k in ['ge','gi','adaptation','refractory_until_ms','spike_count','_ring','decoder_rates','pending']:assert not initial[k].any(),k
 assert np.all(initial['v']==-65) and initial['steps']==0 and initial['t_ms']==0
 mats=matrices()
 for k in mats:
  assert exact(sparse.load_npz(O/('BASE_'+k+'.npz')),sparse.load_npz(B/('ACTUAL_'+k+'.npz')))
  for prefix in ['CANDIDATE_','FINAL_']:assert exact(sparse.load_npz(O/(prefix+k+'.npz')),mats[k])
  assert exact(mats[k],sparse.load_npz(H/('CANDIDATE_ACTUAL_'+k+'.npz')))
 e=arr(O/'NEURAL_RASTER.npz')['spikes'];ring=initial['_ring'].copy();ge=initial['ge'].copy();gi=initial['gi'].copy();ad=initial['adaptation'].copy();ref=initial['refractory_until_ms'].copy();count=initial['spike_count'].copy();pos=0
 # Rebuild delayed arrivals and all linear accumulators without a neural runtime.
 for t,sp in enumerate(e):
  arrival=ring[pos].copy();ge+=mats['We']@arrival;gi+=mats['Wi']@arrival
  ge*=np.exp(-.5/3);gi*=np.exp(-.5/8);ad*=np.exp(-.5/100);ad[sp]+=.002;ref[sp]=(t+1)*.5+2;count+=sp;ring[pos]=sp;pos=(pos+1)%len(ring)
  if t+1 in [2200,3600]:
   state=arr(O/('PREWARM_FINAL_STATE.npz' if t+1==2200 else 'RELEASE_FINAL_STATE.npz'))
   for k,v in dict(ge=ge,gi=gi,adaptation=ad,refractory_until_ms=ref,spike_count=count,_ring=ring).items():assert np.array_equal(state[k],v),(t,k)
   assert not state['threshold_offset_mV'].any() and state['steps']==t+1 and state['time_s']==(t+1)*.0005
 body=arr(O/'RUN/BODY.npz');old=arr(B/'RUN/BODY.npz')
 for k in body:assert np.array_equal(body[k][:400],old[k][:400]),k
 records=js(O/'RUN/PREWARM_NEURAL_RECORDS.json')+js(O/'RUN/NEURAL_RECORDS.json')
 for j in range(401,4001):
  window=int(math.ceil((j*.0005-.2)/.001-1e-9))-1
  if j==2600:window=1100
  assert np.array_equal(body['command'][j],records[window]['applied_command']),(j,window)
 compiled={}
 for p in FIT.glob('*COMPILED*.npz'):
  reference=arr(p);current=arr(O/'COMPILED_1.npz')
  if len(set(reference)&set(current))>100:
   assert set(reference)==set(current)
   assert all(np.array_equal(reference[k],current[k],equal_nan=True) for k in reference),str(p)
   compiled[str(p)]=len(reference)
 assert compiled and all(n==469 for n in compiled.values())
 c0=arr(O/'COMPILED_0.npz');c1=arr(O/'COMPILED_1.npz');assert set(c0)==set(c1) and all(np.array_equal(c0[k],c1[k],equal_nan=True) for k in c0)
 audit=js(O/'ACTUAL_RUNTIME_AUDIT.json');assert audit['factory_calls']==1 and audit['active_steps']==3600 and audit['base_steps']==0 and audit['active_steps_final']==3600 and audit['model_compiles']==2 and audit['total_networks']==2 and audit['active_networks']==1 and audit['guard_calls_total']==3
 weight=js(O/'RUN/GAINS.json')['weight']
 result={label:dict(native_12_5us=metrics(native(folder),weight),shared_half_ms=metrics(bb,weight,False)) for label,folder,bb in [('candidate',O,body),('neutral_round13',B,old)]}
 # Freeze exact half-ms body provenance and currents; bodies are expected to differ after neural start.
 drive=np.asarray([r['sensory_current_nA'] for r in records]);base_rec=js(B/'RUN/PREWARM_NEURAL_RECORDS.json')+js(B/'RUN/NEURAL_RECORDS.json');base_drive=np.asarray([r['sensory_current_nA'] for r in base_rec])
 def firstdiff(a,b,dt,offset=0):
  ix=np.flatnonzero(np.any(a!=b,axis=tuple(range(1,a.ndim))))
  return None if not len(ix) else float(offset+ix[0]*dt)
 result['first_difference_absolute_body_s']=dict(current_window_start=firstdiff(drive,base_drive,.001,.2),applied_window_start=firstdiff(np.array([r['applied_command'] for r in records]),np.array([r['applied_command'] for r in base_rec]),.001,.2),qpos_sample=firstdiff(body['qpos'],old['qpos'],.0005))
 dump(O/'LOW_GAIN_RESULT.json',result)
 dump(O/'LOW_GAIN_VERIFICATION.json',dict(passed=True,independent_new_process_pure_array=True,body_models=0,body_steps=0,neural_networks=0,neural_steps=0,all_ticks=3600,all_windows=1800,release_native_endpoints=56000,shared_release_endpoint_bins=1400,actual_matrix_delta_exact=True,matrix_exact_fixed_current_replay=True,root_offsets_initial_final_exact=True,delay_accumulators_exact=True,prefix400_all_fields_exact=True,body_command_causality_exact=True,compiled_reference_arrays=compiled,compiled_initial_final_exact=True,neutral_reference=str(B),source_identity_exact=True,no_physiology_claim=True))
 report(result,audit)
 print('LOW_GAIN_PURE_ARRAY_PASS',json.dumps(result,ensure_ascii=False,default=lambda v:v.tolist()),flush=True)
def report(result,audit):
 text='# 1/32来源新增兴奋边：一次完整身体研究验证\n\n仅新增17264条兴奋边乘1/32，原baseWe/Wi*.2与中性ConductanceParams、offset0不变；不是全网gain，也未使用上一轮learnedscalar。原1.3s算法物理预热（前.2s无神经，后1.1s实时感觉神经预热）＋.7s纯神经，共2s，同一候选网络/原encoder/decoder/body/rates/pending连续交接。0训练/0优化，仅此一次完整新身体；原第13轮真实同2s中性记录只读复用。\n\n研究合同scope=research_only_not_default_or_production；旧candidate runtime_allowed=false仍保留，原pool/sensory guard和显式contract预检通过。没有默认注册/旧guard修改，不称已认证生理网络。\n\n结果（事件以对应native12.5us或shared.5ms同口径比较，绝不混减）：\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2,default=lambda v:v.tolist())+'\n```\n\n接触翻转不算迈步，不使用>=4feet判定步态。非足承重期间root位移不当自主推进；本次不宣称自主稳定支撑、行走或生理验证。若未来有真正足支撑，仍需父10/30s、因果controls与同次视频。无独立感觉或肌肉真值，绝对生物单位NOT_ESTABLISHED。未渲染相同失败视频，未作目视通过声明。\n\n独立新进程纯数组复核全部3600ticks/1800窗rates和applied因果、实际矩阵base+added*.03125及初末/root/offset0、delayring/ge/gi/adapt/ref/count、网络/身体交接、初始400帧全部BODY字段bitwise、全部原生接触signed力/flags重建、release56000端点零助力/外力/mocap、shared1400端点和sourcehash。模型XML3470原样，469compiled arrays与原training包、初末compiled逐项exact。身体原liveforward/cache调度与frames不改，不导入sourcepose/IK/弯曲loss控制。\n\n实际计数：\n\n```json\n'+json.dumps(dict(execution=js(O/'EXECUTION.json'),runtime=audit),ensure_ascii=False,indent=2)+'\n```\n\n核心必需：本运行工具/冻结执行源、PREREGISTRATION/研究合同、RUN完整身体/神经/交接、NATIVE逐步力接触、矩阵/初末/root与compiled身份、独立数组验收。重要辅助：本中文报告、来源适配台账、资源与job台账、清单。清理实际删除0：新增产物均用于运行、复核或溯源，未确认可弃文件；未复制venv源码。父review前未封，seal仅一次，封后只读。\n'
 (O/'REPORT.zh-CN.md').write_text(text)
 dump(O/'SOURCE_CHANGE_LEDGER.json',dict(changes={},existing_sources_modified=False,execution_source=str(O/'EXECUTION_SOURCE.py'),delivered_source=str(Path(__file__).resolve()),execution_sha256=sha(O/'EXECUTION_SOURCE.py'),delivery_sha256=sha(__file__),wrapper_adapter=str(O/'WRAPPER_ADAPTER.json')))
def seal():
 assert not (O/'MANIFEST_SHA256.json').exists()
 assert js(O/'PARENT_REVIEW.json')['passed'] and js(O/'LOW_GAIN_VERIFICATION.json')['passed']
 dump(O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='All created evidence retained; no confirmed disposable files; no venv source copies'))
 dump(O/'FILE_INVENTORY.json',{str(p):dict(importance='重要辅助' if p.name in ['REPORT.zh-CN.md','JOB_LEDGER.json','SOURCE_CHANGE_LEDGER.json','WRAPPER_ADAPTER.json','CLEANUP_LEDGER.json'] else '核心必需',purpose='Scientific evidence and source identity or review provenance') for p in O.rglob('*') if p.is_file()})
 dump(O/'MANIFEST_SHA256.json',{str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in O.rglob('*') if p.is_file()})
 for p,item in js(O/'MANIFEST_SHA256.json').items():assert sha(p)==item['sha256']
 for p in O.rglob('*'):
  if p.is_file():p.chmod(0o444)
 for p in sorted([p for p in O.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
 O.chmod(0o555);print('SEALED_ONCE',O,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','verify','seal']);args=parser.parse_args();globals()[args.mode]()
