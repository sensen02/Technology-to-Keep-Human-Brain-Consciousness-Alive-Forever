"""One fixed-parameter full-prewarm physical validation; independent array audit mode.
Reuses the admitted physical observer through an explicit, archived source adapter.
No optimizer, checkpoint parameter switch, baseline rerun or default-table changes.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,hashlib,inspect,argparse,dataclasses,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-fixed-parameter-validation/20261101T230101Z')
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
FIT=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-closedloop-parameter-fit/20261101T222509Z')
ADMIT=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')
SELECT=FIT/'SELECTED_CANDIDATE.json'
CONFIG=dict(duration=1.3,kp=100.,clip=50.,dt=.0000125,neural_duration=2.,neural_command_dt_s=.001,neural_prewarm_s=1.1,swing_height=.45,sample_dt_s=.0005)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def selected():
 s=json.loads(SELECT.read_text())['selected'];assert s['label']=='TRAIN_041_MINUS' and not s['failed']
 x=np.asarray(s['x'],float);assert x.shape==(3,) and np.isfinite(x).all()
 assert np.all((x[:2]>=.5)&(x[:2]<=2)) and -3<=x[2]<=3
 return s

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);s=selected()
 spec=dict(task='S1/S4/S5 full fixed selected SNN parameters from first neural initialization',config=CONFIG,seed=0,selection_source=str(SELECT),selection_sha256=sha(SELECT),selected_label=s['label'],x=s['x'],admission=str(ADMIT),baseline=str(BASE),conditional_fit=str(FIT),
 parameter_injection='Wrap continuous_handover._make_runtime: original resting base net never steps; FrozenConnectomeTuning(actual roots, base We/Wi, intrinsic params, dt, source offset).spawn(selected). Original encoder/decoder retained. Same active net, decoder, rates, pending and body across 1.3s boundary; no parameter switch or reset.',
 engineering_bounds=dict(api_gains=[.25,4],api_threshold_mV=[-5,5],training_gains=[.5,2],training_threshold_mV=[-3,3]),
 schedule='[0,.2) no neural; [.2,1.3] original algorithm PD reference warmup with tuned online SNN; (1.3,3.3] pure neural release. Original live forward cadence unchanged.',
 invariants='MODEL actual3470 hash, initial state, dt12.5us, command1ms, sample.5ms, kp100/clip50/swing.45, rootfree/nmocap0/neq0/90muscles/71meshes, all body parameters/reference/tier/root order/masks/intrinsics unchanged. Only fixed selected gains/threshold differ.',
 preflight='Explicit original guard(version_contract) and sensory CLI sha before any physical model load. No guard bypass/default table/trust-anchor edit.',
 observation='Reuse stepwise clone forward signed native forces/flags/root/COM/ctrl/activation; contact-point dual body Jacobians only .5ms+explicit boundaries. NaN velocities off-grid and inf no-candidate distances are declared sentinels.',
 evaluation='2s release absolute: sustained signed foot load, positive nonfoot support, COM/shape/root displacement and slip, zero assistance and causal neural control. First absent event is right censored >=2s with null event time. Contact flips never walking steps. No >=4feet/60ms rule. Shared .7s comparison only same endpoint/role against admitted baseline; conditional 50/150ms historical trials descriptive only.',
 execution='Exactly one new full 3.3s trajectory even if behavior fails; interrupt only first nonfinite/warning, save first failure. No fitting/reselection/weights-mask change. No new baseline integration. Resource counts actual, expected264000body/6200neural/3100windows/6601BODY,2 total networks1active,1tier,2model compiles.',
 validation='Separate new process array-only: all3100windows/6200ticks/native rows, parameter-mask-root equality,400frame prefix, boundary continuity, raster decoder causality, zero assistance, source hashes and compiled identity. No reintegration.',
 behavior_conclusion='Positive commands or loss improvement do not certify walking. Any coordinated sustained progression must be reported to parent before staged10/30s/controls/video; no goal completion declaration.',
 units='Absolute biological force/length calibration NOT_ESTABLISHED',importance={'script/prereg/trajectory/parameter identities/verification':'核心必需','report/manifest/ledger':'重要辅助'},cleanup='No known dispensable files; no old package writes or venv source copies.')
 p=OUT/'PREREGISTRATION.json'
 if p.exists():assert json.loads(p.read_text())==spec
 else:dump(p,spec)
 print('PREREGISTERED_NO_MODEL_LOAD',str(p),flush=True)

def adapted_wrapper():
 import run_banc888_physical_handover as w
 w.OUT=OUT;w.CONFIG=CONFIG
 # Exact replacements only change duration bookkeeping and add provenance inputs.
 text=inspect.getsource(w.run)
 replacements={" prepare()":" pass # independent preregistration already frozen by caller", "(160000,3600,1,1,1)":"(264000,6200,2,1,1)","==160002":"==264002","'of160000'":"'of264000'", "paths.update([Path(__file__),":"paths.update([ROOT/'tools/run_banc888_physical_handover.py',ROOT/'engine/neural_tuning.py',SELECT,Path(__file__),"}
 for a,b in replacements.items():
  assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b)
 w.SELECT=SELECT
 (OUT/'REUSED_WRAPPER_EXECUTION.py').write_text(text)
 dump(OUT/'WRAPPER_ADAPTER.json',dict(source=str(ROOT/'tools/run_banc888_physical_handover.py'),source_sha256=sha(ROOT/'tools/run_banc888_physical_handover.py'),replacements=replacements,adapted_sha256=sha(OUT/'REUSED_WRAPPER_EXECUTION.py'),reason='Reuse admitted observer without changing old source/package; change duration and network allocation bookkeeping only. Injection is separate factory wrapper.'))
 w.__file__=str(Path(__file__).resolve())
 exec(compile(text,str(OUT/'REUSED_WRAPPER_EXECUTION.py'),'exec'),w.__dict__)
 return w

def run():
 prepare();assert not (OUT/'EXECUTION.json').exists() and not (OUT/'RUN').exists(),'refuse second physical run'
 from engine.embodied.runtime_pool_guard import load_runtime_pool
 c=json.loads((ADMIT/'VERSION_CONTRACT.json').read_text())
 assert sha(ADMIT/'SENSORY_LOOKUP_SNAPSHOT888.json')==c['sensory_lookup_sha256']
 load_runtime_pool(ADMIT/'MN_POOLS_SNAPSHOT888.json',version_contract=c)
 dump(OUT/'PREFLIGHT.json',dict(passed=True,before_any_model_load=True,contract_sha256=sha(ADMIT/'VERSION_CONTRACT.json'),pool_sha256=sha(ADMIT/'MN_POOLS_SNAPSHOT888.json'),sensor_sha256=sha(ADMIT/'SENSORY_LOOKUP_SNAPSHOT888.json')))
 w=adapted_wrapper()
 import engine.embodied.continuous_handover as h
 from engine.neural_tuning import FrozenConnectomeTuning,TuningParameters
 from engine.neural_cond import ConductanceNetwork
 from scipy.sparse import save_npz
 import mujoco
 factory=h._make_runtime;orig_build=w.__dict__.get('unused',None)
 base_build=__import__('run_physical_walk_prewarm').build
 active=None;base=None;enc=None;dec=None;factory_calls=0;active_steps=0;segment=[];compiled=[]
 s=selected();x=np.array(s['x'])
 def make(built,*args,**kwargs):
  nonlocal active,base,enc,dec,factory_calls
  factory_calls+=1;assert factory_calls==1
  base,enc,dec=factory(built,*args,**kwargs);assert base.steps==0 and base.t_ms==0
  tuning=FrozenConnectomeTuning(np.asarray(built['tier']['root_ids']),base.We,base.Wi,params=base.p,dt_ms=base.dt,threshold_offset_mV=base.threshold_offset_mV)
  active=tuning.spawn(TuningParameters(*x))
  for name,gain in [('We',x[0]),('Wi',x[1])]:
   a=getattr(active,name);b=getattr(base,name)
   assert np.array_equal(a.data,b.data*gain) and np.array_equal(a.indices,b.indices) and np.array_equal(a.indptr,b.indptr)
   save_npz(OUT/('BASE_'+name+'.npz'),b);save_npz(OUT/('TUNED_'+name+'.npz'),a)
  np.savez_compressed(OUT/'FIXED_PARAMETERS.npz',root_ids=tuning.root_ids,source_offset=base.threshold_offset_mV,threshold_offset=active.threshold_offset_mV,x=x)
  dump(OUT/'TUNING_CONTRACT.json',dict(api=tuning.contract(),selected_label=s['label'],x=x,params=dataclasses.asdict(base.p),dt_ms=base.dt,base_steps_at_injection=base.steps,base_clock_at_injection=base.t_ms,net_id=id(active),base_id=id(base),encoder_id=id(enc),decoder_id=id(dec)))
  return active,enc,dec
 old_step=ConductanceNetwork.step
 def checked_step(net,*args,**kwargs):
  nonlocal active_steps
  assert net is active and base.steps==0
  assert np.array_equal(net.threshold_offset_mV,base.threshold_offset_mV+x[2])
  active_steps+=1
  return old_step(net,*args,**kwargs)
 # Record all compiled arrays without another model load/forward.
 old_loader=__import__('run_physical_walk_prewarm')._load_mjcf
 class Loader:
  def __init__(self,obj):self.obj=obj
  def compile(self,*args,**kwargs):
   model=self.obj.compile(*args,**kwargs)
   arrays={k:getattr(model,k).copy() for k in dir(model) if isinstance(getattr(model,k,None),np.ndarray)}
   ix=len(compiled);np.savez_compressed(OUT/('COMPILED_%d.npz'%ix),**arrays)
   compiled.append(dict(index=ix,arrays=len(arrays),nq=model.nq,nv=model.nv,nu=model.nu,neq=model.neq,nmocap=model.nmocap,nmesh=model.nmesh,joint0_type=int(model.jnt_type[0]),muscle_actuators=int(np.count_nonzero(model.actuator_dyntype==mujoco.mjtDyn.mjDYN_MUSCLE))))
   assert model.nmocap==0 and model.neq==0 and model.nmesh==71 and model.nu==90
   return model
 def loader(*a,**kw):return Loader(old_loader(*a,**kw))
 h._make_runtime=make;ConductanceNetwork.step=checked_step
 __import__('run_physical_walk_prewarm')._load_mjcf=loader
 try:
  w.run(ADMIT/'VERSION_CONTRACT.json',ADMIT/'MN_POOLS_SNAPSHOT888.json',ADMIT/'SENSORY_LOOKUP_SNAPSHOT888.json')
 finally:
  h._make_runtime=factory;ConductanceNetwork.step=old_step;__import__('run_physical_walk_prewarm')._load_mjcf=old_loader
  if active is not None:
   for name in ['We','Wi']:save_npz(OUT/('FINAL_'+name+'.npz'),getattr(active,name))
   np.savez_compressed(OUT/'FINAL_THRESHOLD.npz',threshold_offset=active.threshold_offset_mV)
  dump(OUT/'FIXED_RUNTIME_AUDIT.json',dict(factory_calls=factory_calls,active_steps=active_steps,base_steps=None if base is None else base.steps,active_steps_final=None if active is None else active.steps,compiled=compiled,model_compiles=len(compiled),active_network_instances=int(active is not None),total_network_instances=int(base is not None)+int(active is not None),tier_builds=int(factory_calls>0)))

def verify():
 # Execute the existing independent array auditor with explicit finite-window adapters.
 # It imports only stdlib and numpy; no model, runtime or SNN construction.
 path=ROOT/'tools/verify_banc888_physical_handover.py';text=path.read_text()
 replacements={"P=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')":"P=Path("+repr(str(OUT))+")", "N=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-native-recording/20261101T191552Z')":"N=Path("+repr(str(BASE))+")", "and actual.get(k)!=v":"and k!='neural_duration_s' and actual.get(k)!=v", "==4001":"==6601", "body['time']-oldbody['time']":"body['time'][:4001]-oldbody['time']", "(3600,12000)":"(6200,12000)","==1800":"==3100", "reshape(700,2,12000)":"reshape(2000,2,12000)","(final,3600)":"(final,6200)","np.arange(1,160001)":"np.arange(1,264001)","release.sum()==56000":"release.sum()==160000", "==160002":"==264002", "release_endpoint_steps=56000":"release_endpoint_steps=160000", "np.savez_compressed(P/'TRAJECTORY_COMPARISON.npz',time=body['time'],new_com=body['com'],old_com=oldbody['com'],new_root=body['qpos'][:,:7],old_root=oldbody['qpos'][:,:7],new_foot_force=body['foot_force'],old_foot_force=oldbody['foot_force'])":"pass # comparison stored separately with common endpoints", "bb['com'][-1]":"bb['com'][4000]", "bb['qpos'][-1,:3]":"bb['qpos'][4000,:3]", "bb['foot_force'][-1,:,2]":"bb['foot_force'][4000,:,2]"}
 for a,b in replacements.items():
  assert a in text,a;text=text.replace(a,b)
 (OUT/'ARRAY_AUDITOR_EXECUTION.py').write_text(text)
 ns={'__name__':'fixed_array_auditor'};exec(compile(text,str(OUT/'ARRAY_AUDITOR_EXECUTION.py'),'exec'),ns);ns['main']()
 from scipy.sparse import load_npz
 params=load(OUT/'FIXED_PARAMETERS.npz');x=np.asarray(selected()['x']);assert np.array_equal(params['x'],x)
 for name,gain in [('We',x[0]),('Wi',x[1])]:
  base=load_npz(OUT/('BASE_'+name+'.npz'));ref=load_npz(BASE/('ACTUAL_'+name+'.npz'))
  for field in ['indices','indptr','data']:assert np.array_equal(getattr(base,field),getattr(ref,field))
  for label in ['TUNED','FINAL']:
   a=load_npz(OUT/(label+'_'+name+'.npz'))
   assert np.array_equal(a.indices,base.indices) and np.array_equal(a.indptr,base.indptr) and np.array_equal(a.data,base.data*gain)
 assert np.array_equal(params['threshold_offset'],params['source_offset']+x[2])
 for path in ['FINAL_THRESHOLD.npz','PREWARM_FINAL_STATE.npz','RELEASE_FINAL_STATE.npz']:
  assert np.array_equal(load(OUT/path)['threshold_offset_mV' if 'STATE' in path else 'threshold_offset'],params['threshold_offset'])
 assert np.array_equal(params['root_ids'],load(BASE/'IDENTITIES.npz')['root_ids'])
 audit=json.loads((OUT/'FIXED_RUNTIME_AUDIT.json').read_text());assert audit['factory_calls']==1 and audit['active_steps']==6200 and audit['base_steps']==0 and audit['active_steps_final']==6200 and audit['model_compiles']==2
 # Exact pre-neural body prefix against admitted root888 baseline, all fields.
 b=load(OUT/'RUN/BODY.npz');old=load(BASE/'RUN/BODY.npz')
 for k in b:assert np.array_equal(b[k][:400],old[k][:400]),k
 pre=load(OUT/'PREWARM_FINAL_STATE.npz');final=load(OUT/'RELEASE_FINAL_STATE.npz')
 assert pre['steps']==2200 and final['steps']==6200
 # Saved BODY commands are the command in the immediately preceding interval.
 rec=json.loads((OUT/'RUN/PREWARM_NEURAL_RECORDS.json').read_text())+json.loads((OUT/'RUN/NEURAL_RECORDS.json').read_text())
 for j in range(401,6601):
  window=int(math.ceil((j*.0005-.2)/.001-1e-9))-1
  if j==2600:window=1100 # release-after role replaces assisted endpoint
  assert np.array_equal(b['command'][j],rec[window]['applied_command']),(j,window)
 # Full compiled model reference saved by the previous conditional checkpoint experiment.
 compiled_candidates=list(FIT.glob('*COMPILED*.npz'))
 compiled_comparison={}
 for p in compiled_candidates:
  z=load(p);cur=load(OUT/'COMPILED_1.npz');common=set(z)&set(cur)
  if len(common)>100:
   assert all(np.array_equal(z[k],cur[k],equal_nan=True) for k in common)
   compiled_comparison[str(p)]=len(common)
 assert compiled_comparison,'missing baseline compiled arrays'
 weight=json.loads((OUT/'RUN/GAINS.json').read_text())['weight'];summary=json.loads((OUT/'SUMMARY.json').read_text())
 # Absolute release and common700ms are explicitly separate. Native positive nonfoot event.
 allz={k:[] for k in ['step','kind','nonfoot_force','foot_force','com','root']}
 for chunk in json.loads((OUT/'CHUNKS.json').read_text()):
  with np.load(chunk['file']) as z:
   for k in allz:allz[k].append(z[k])
 allz={k:np.concatenate(v) for k,v in allz.items()};mask=(allz['kind']==0)&(allz['step']>104000)
 pos=mask&(allz['nonfoot_force'][:,2]>0);ix=np.flatnonzero(pos)
 release=dict(duration_s=2.,native_endpoint_bins=int(mask.sum()),first_positive_nonfoot_event_s=float((allz['step'][ix[0]]-104000)*.0000125) if len(ix) else None,event_right_censored=not bool(len(ix)),event_observation_lower_bound_s=2. if not len(ix) else None,positive_nonfoot_endpoint_count=int(pos.sum()),positive_nonfoot_fz_over_weight_mean=float(np.maximum(allz['nonfoot_force'][mask,2],0).mean()/weight),foot_fz_over_weight_mean=float(allz['foot_force'][mask,:,2].sum(axis=1).mean()/weight),com_height_min=float(allz['com'][mask,2].min()),com_height_last=float(allz['com'][mask,2][-1]),root_displacement=allz['root'][mask][-1,:3]-b['qpos'][2600,:3],body_shape_qpos_hinge_range=np.ptp(b['qpos'][2600:,7:],axis=0),walking_certified=False)
 def shared(bb,end):
  ep=np.arange(2601,2600+round(end/.0005)+1);pos=bb['nonfoot_force'][ep,2]>0
  return dict(duration_s=end,endpoint_bins=len(ep),positive_nonfoot_bins=int(pos.sum()),first_positive_nonfoot_s=float(bb['time'][ep[pos][0]]-1.3) if pos.any() else None,right_censored=not pos.any(),censor_lower_bound_s=end if not pos.any() else None,foot_fz_over_weight_mean=float(bb['foot_force'][ep,:,2].sum(axis=1).mean()/weight),positive_nonfoot_fz_over_weight_mean=float(np.maximum(bb['nonfoot_force'][ep,2],0).mean()/weight),com_last=bb['com'][ep[-1]],root_displacement=bb['qpos'][ep[-1],:3]-bb['qpos'][2600,:3])
 common={str(end):dict(fixed_full_history=shared(b,end),neutral_full_history=shared(old,end)) for end in [.05,.15,.7]}
 dump(OUT/'FIXED_VALIDATION_SUMMARY.json',dict(absolute_release=release,common_endpoint_comparisons=common,previous_conditional_result=json.loads((FIT/'SELECTED_CANDIDATE.json').read_text())['selected'],conditional_comparison_note='Prior TRAIN_041_MINUS50ms and validation150ms inherit neutral checkpoint ge/rates/pending; distinct history intervention, descriptive only. See original trials, not a2s baseline.',native_details=summary['new_native'],units='NOT_ESTABLISHED',interpretation='Contact flips/positive commands/root motion during nonfoot support are not walking; no goal completion claim.'))
 dump(OUT/'FIXED_PARAMETER_VERIFICATION.json',dict(passed=True,model_loads=0,body_steps=0,neural_steps=0,all_ticks=6200,all_windows=3100,body_samples=6601,base_network_never_stepped=True,exact_actual_weight_gain_and_masks=True,root_order_exact=True,threshold_source_plus_shift_exact=True,prefix_all_body_fields400_exact=True,body_command_causality_exact=True,compiled_reference_arrays=compiled_comparison,independent_native_audit=str(OUT/'INDEPENDENT_VERIFICATION.json')))
 print('FIXED_PARAMETER_ARRAY_VERIFICATION_PASS',release,flush=True)

def seal():
 execution=json.loads((OUT/'EXECUTION.json').read_text());verify_result=json.loads((OUT/'FIXED_PARAMETER_VERIFICATION.json').read_text());s=json.loads((OUT/'FIXED_VALIDATION_SUMMARY.json').read_text());r=s['absolute_release']
 text=f'''# 固定已选SNN参数完整预热验证\n\n本轮只验证TRAIN_041_MINUS，从首次神经起动固定使用选中三参数；没有训练、重新挑选、checkpoint参数切换或旧基线重跑。原算法参考预热1.3s后连续纯神经release2s，总3.3s。\n\n结果：未验证自主持续支撑或行走。2s阶段正非足支持端点{r['positive_nonfoot_endpoint_count']}/{r['native_endpoint_bins']}，首次正非足支持相对释放{r['first_positive_nonfoot_event_s']}s，平均足Fz/W={r['foot_fz_over_weight_mean']:.9g}，平均正非足Fz/W={r['positive_nonfoot_fz_over_weight_mean']:.9g}，末COM高度{r['com_height_last']:.9g}（模型单位）。root位移保存在FIXED_VALIDATION_SUMMARY.json；非足承载期位移不能称行走。事件若未见则保留null并右删失>=2s，本次按实际观测记录。\n\n同端点前50/150/700ms与第13轮root888完整预热基线分别在FIXED_VALIDATION_SUMMARY.json；旧700ms结果不当作2s对照。上一条件训练继承中性checkpoint历史，属于不同干预。滑移使用.5ms双body contactpointJac切向速度，完整分足载荷、COM/形变、root及接触记录均保留；接触翻转不是迈步。绝对生物单位NOT_ESTABLISHED。\n\n独立新进程纯数组验收通过：6200tick→3100解码窗、6601BODY、400帧全部BODY字段前缀、1.3s交接8神经数组/身体状态/rates/pending、actual基准矩阵乘gain逐值与mask/root一致、sourceoffset+theta阈值、全部原生signed力/flags及零助力、命令因果与sourcehash。没有重积分。compiled模型数组与历史参考逐项验证，90肌肉/71mesh/rootfree/nmocap0/neq0保留。\n\n实际资源：{json.dumps(execution['counters'],ensure_ascii=False)}；补充编译与网络身份见FIXED_RUNTIME_AUDIT.json。总网络2、活动1，原始基准网络steps=0；同一个tunednet和原decoder贯穿预热和释放。运行退出码{execution['exit_code']}；独立验收passed={verify_result['passed']}。\n\n未生成相同下沉视频或调用read_image；没有目视通过声明，本报告仅数组验证。不宣告长期goal完成；未改tracker、未提交。\n\n核心必需：新脚本、PREREGISTRATION、RUN/BODY与神经记录、NATIVE分块、参数与模型身份、EXECUTION_SOURCE与复用适配源、两份独立验证。重要辅助：本报告、source/adapter台账与MANIFEST。全部文件完整路径列于MANIFEST.json。实际清理0：当前新增文件均用于运行、复核或追溯，没有确认可弃文件；未复制venv源码。\n'''
 (OUT/'REPORT.zh-CN.md').write_text(text)
 dump(OUT/'SOURCE_CHANGE_LEDGER.json',dict(changes={},execution_source=str(OUT/'EXECUTION_SOURCE.py'),execution_sha256=sha(OUT/'EXECUTION_SOURCE.py'),delivered_source=str(Path(__file__).resolve()),delivery_sha256=sha(__file__),same_execution_delivery=sha(__file__)==sha(OUT/'EXECUTION_SOURCE.py'),existing_sources_modified=False,tracker_modified=False))
 entries={str(p):dict(sha256=sha(p),bytes=p.stat().st_size,importance='重要辅助' if p.name in ['REPORT.zh-CN.md','SOURCE_CHANGE_LEDGER.json','WRAPPER_ADAPTER.json'] else '核心必需') for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
 entries[str(Path(__file__).resolve())]=dict(sha256=sha(__file__),bytes=Path(__file__).stat().st_size,importance='核心必需')
 dump(OUT/'MANIFEST.json',dict(files=entries,cleanup_deleted_files=[],cleanup_reason='All new artifacts retained for execution/scientific/source audit; no dispensable duplicates',image_review='No image input; arrays only'))
 assert all(sha(p)==v['sha256'] for p,v in entries.items())
 print('SEALED_MANIFEST',len(entries),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','verify','seal']);a=p.parse_args();globals()[a.mode]()
