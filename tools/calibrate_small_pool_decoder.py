"""S4 fixed K5 actual-present-pool readout calibration; one fresh trial, no training.
Default modules/configs and historical evidence are never modified.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,math,inspect,hashlib,shutil,argparse,time,dataclasses,collections
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s4-small-pool-decoder/20261102T064123Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
C=Path('/home/sensen/Desktop/engine修复验收/s4-bending-closedloop-training/20261102T042848Z')
E=Path('/home/sensen/Desktop/engine修复验收/s2-early-muscle-support/20261102T053639Z')
P=Path('/home/sensen/Desktop/engine修复验收/s4-antagonist-mn-support/20261102T061756Z')
LABEL='FIXED_K5_NEUTRAL';DT=.0000125

def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def js(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csrhash(w):
 h=hashlib.sha256()
 for k in ('data','indices','indptr'):h.update(getattr(w,k).tobytes())
 return h.hexdigest()
def replace(text,before,after):
 assert text.count(before)==1,(before,text.count(before));return text.replace(before,after)

def replay(events,rows,gain):
 rates=np.zeros(90);pending=np.zeros(90);full=[];applied=[];nxt=[];decay=math.exp(-.0005/.005)
 for j,sp in enumerate(events):
  inst=np.array([np.mean(sp[r['local_indices']])/.0005 if r['local_indices'] else 0. for r in rows]);rates=decay*rates+(1-decay)*inst;full.append(rates.copy())
  if j%2:
   applied.append(pending.copy());pending=np.clip((rates/100.)*gain,0,1);nxt.append(pending.copy())
 return dict(rates_halfstep_hz=np.array(full),applied=np.array(applied),next=np.array(nxt))

def prepare():
 import train_bending_closedloop as old
 from engine.embodied.motor_output import MotorOutputConfig
 assert not O.exists() or {p.name for p in O.iterdir()}<={'ATTEMPTS','FAILURE_LEDGER.json'},'refuse existing scientific evidence package'
 m=js(B/'RUN/MAPPING.json');roots=arr(B/'IDENTITIES.npz')['root_ids'];rows=m['rows'];n=np.array([len(r['local_indices']) for r in rows]);gain=np.minimum(1.,n/5.)
 MotorOutputConfig(per_actuator_gain=tuple(gain)).validate()
 groups=[];seen={}
 for r in rows:
  assert np.array_equal(roots[r['local_indices']],r['tier_root_ids'])
  key=tuple(r['local_indices'])
  if not key:continue
  if key not in seen:seen[key]=len(groups);groups.append(dict(local_indices=list(key),root_ids=roots[list(key)],actuator_indices=[],gain=float(gain[r['actuator_index']])))
  g=groups[seen[key]];g['actuator_indices'].append(r['actuator_index']);assert g['gain']==gain[r['actuator_index']]
 assert len(groups)==66 and len(seen)==66 and sum(n==0)==16
 assert not any(set(a)&set(b) for i,a in enumerate(seen) for b in list(seen)[i+1:])
 O.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(O/'FIXED_GAIN_VECTOR.npz',actual_present_n=n,per_actuator_gain=gain,root_ids=roots,actuator_names=[r['actuator'] for r in rows])
 dump(O/'POOL_GAIN_SCHEMA.json',dict(normalization='actual PRESENT MN count, including partial rows; never full source pool count; do not fill missing roots',rows=[dict(actuator=r['actuator'],actuator_index=i,status=r['status'],present_n=int(n[i]),source_pool_n=len(r['pool_root_ids']),missing_tier_roots=r.get('missing_tier_roots',[]),gain=gain[i],local_indices=r['local_indices'],tier_root_ids=r['tier_root_ids']) for i,r in enumerate(rows)],groups=groups,status_counts=dict(collections.Counter(r['status'] for r in rows)),row_population=dict(collections.Counter(map(int,n))),group_population=dict(collections.Counter(len(g['local_indices']) for g in groups)),overlap_pairs=0,shared_groups=[g for g in groups if len(g['actuator_indices'])>1],unresolved_gain_zero_valid=True,weights_duplicated=False))
 text=inspect.getsource(old.trial)
 text=replace(text,"assert args[0]==mfest","assert args[0]==mfest\n  assert dataclasses.asdict(dec.config)==dataclasses.asdict(MOTOR_CONFIG)\n  dump(folder/'ACTUAL_DECODER_CONFIG.json',dataclasses.asdict(dec.config))")
 text=replace(text,"observer=kw['body_observer'];isrelease=kw.get('runtime_state') is not None","observer=kw['body_observer'];isrelease=kw.get('runtime_state') is not None\n  compiled={k:getattr(model,k).copy() for k in dir(model) if isinstance(getattr(model,k),np.ndarray)}\n  if not isrelease:\n   MODEL_IDS.append(id(model));np.savez_compressed(folder/'ACTUAL_COMPILED_INITIAL.npz',**compiled)\n  else:assert id(model)==MODEL_IDS[0]")
 text=replace(text,"np.savez_compressed(folder/'BODY_BOUNDARY.npz',**ret['handover_after_withdrawal'])","np.savez_compressed(folder/'BODY_BOUNDARY.npz',**ret['handover_after_withdrawal'])\n   np.savez_compressed(folder/'ACTUAL_COMPILED_FINAL.npz',**{k:getattr(model,k).copy() for k in compiled})\n   dump(folder/'MODEL_OBJECT_IDENTITY.json',dict(initial_model_id=MODEL_IDS[0],final_model_id=id(model),same_object=True,compiled_array_count=len(compiled)))\n   dump(folder/'MATRIX_FINAL_HASHES.json',{k:csrhash(getattr(net,k)) for k in ['We','Wi']})\n   for key in ['We','Wi']:sparse.save_npz(folder/('FINAL_'+key+'.npz'),getattr(net,key))")
 text=replace(text,"ns['run'](folder/'RUN',**CONFIG,neural_duration=release)","ns['run'](folder/'RUN',**CONFIG,neural_duration=release,neural_motor_config=MOTOR_CONFIG)")
 compile(text,str(O/'TRIAL_EXECUTION_SOURCE.py'),'exec');(O/'TRIAL_EXECUTION_SOURCE.py').write_text(text)
 # Reuse only the explicitly observed original trial source; archive every literal edit.
 dump(O/'ADAPTER_CONTRACT.json',dict(original_source=str(R/'tools/train_bending_closedloop.py'),original_sha256=sha(R/'tools/train_bending_closedloop.py'),modified_function='trial',changes=['explicit runner neural_motor_config=MOTOR_CONFIG','assert/capture actual decoder config','capture 469 compiled initial/final arrays from same model object','capture actual final CSR'],neutral_z=[0.]*12,threshold_offset=0,rawcommand_postscale=False))
 offline=(P/'OFFLINE_SELECTED_STATES.py').read_text()
 (O/'OFFLINE_SELECTED_STATES.py').write_text(offline)
 # Dedicated verifier is generated from the previously independently reviewed pure-array checks.
 verify=(P/'VERIFY_ARRAYS.py').read_text();start=verify.index("static=js(O/'STATIC_POOL_SIGN_AND_OVERLAP.json')");end=verify.index("for key in ['We','Wi']:",start)
 verify=verify[:start]+"mapping=js(B/'RUN/MAPPING.json');roots=ar(B/'IDENTITIES.npz')['root_ids'];expected=np.zeros(12000);scalar_max=0.\nf=O/'FIXED_K5_NEUTRAL';p=ar(f/'PARAMETERS.npz');assert not p['source_offset'].any();exact(p['threshold_offset'],expected);exact(p['root_ids'],roots)\ngain=ar(O/'FIXED_GAIN_VECTOR.npz')['per_actuator_gain'];exact(gain,np.minimum(1.,np.array([len(r['local_indices']) for r in mapping['rows']])/5.))\nassert js(f/'ACTUAL_DECODER_CONFIG.json')['per_actuator_gain']==gain.tolist()\n"+verify[end:]
 verify=replace(verify,"expectedcommand=np.clip(rates/100.,0,1)","expectedcommand=np.clip((rates/100.)*gain,0,1)")
 verify=replace(verify,"threshold_5minus1_2plus1=True","threshold_offset_all_zero=True")
 # Additional independent semantic/source/metric checks execute in the same fresh array-only process.
 verify=replace(verify,"result=dict(passed=True",extra_verifier()+"\nresult=dict(passed=True")
 compile(verify,str(O/'VERIFY_ARRAYS.py'),'exec');(O/'VERIFY_ARRAYS.py').write_text(verify)
 paths=[Path(__file__),R/'tools/train_bending_closedloop.py',O/'TRIAL_EXECUTION_SOURCE.py',O/'VERIFY_ARRAYS.py',O/'OFFLINE_SELECTED_STATES.py',O/'FIXED_GAIN_VECTOR.npz',O/'POOL_GAIN_SCHEMA.json',O/'ADAPTER_CONTRACT.json',B/'RUN/MAPPING.json',B/'RUN/MODEL.xml',B/'RUN/BODY.npz',B/'NEURAL_RASTER.npz',B/'ACTUAL_We.npz',B/'ACTUAL_Wi.npz',B/'IDENTITIES.npz',C/'EXACT_GATE.json',C/'FROZEN_REFERENCE.json',C/'TRAIN_000_BASELINE/BODYLOSS_NATIVE.npz',C/'TRAIN_000_BASELINE/RELEASE_FK.npz',Path(old.T),E/'COMPILED_MODEL.npz']
 paths+=list((R/'engine').rglob('*.py'))+[R/'run_physical_walk_prewarm.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py']+[old.A/name for name in ['VERSION_CONTRACT.json','MN_POOLS_SNAPSHOT888.json','SENSORY_LOOKUP_SNAPSHOT888.json']]
 dump(O/'PREREGISTRATION.json',dict(stage='S4 engineering single-factor small-pool decoder sensitivity calibration, not physiological calibration, optimization or CPG',head='68e5e10',source_sha256={str(p):sha(p) for p in paths},trials=[dict(label=LABEL)],unique_K=5,config=old.CONFIG,motor_config=dataclasses.asdict(MotorOutputConfig(per_actuator_gain=tuple(gain))),formula='rate[t]=exp(-.0005/.005)*rate[t-1]+(1-exp(-.0005/.005))*mean(present spikes)/.0005; cmd=clip((rate/100)*min(1,n_present/5),0,1); no rate state scaling and no scaled-clipped command',causality='consume completed 1ms window, pending applies next window; t0 fresh rates/pending0; same body/net/decoder crosses prewarm; no reset',source_graph='original round13 neutral root888 actual We/Wi*.2 offset0, not seven-node threshold candidate',history='S233/S237/S253/S254/S256 and 20261028 dynamic neural gain already studied other branch gains; not first gain calibration. Current experiment specifically same3470/root888/fullprewarm and actual-present floorK5; no matching K5 found in current Python source search; earlier FIR/65DOF/body branches not baseline',offline_first='before body: same original raster2300 tick old vs K5 rawrate+preclipgain+clip arrays, saturation/earlysequence and analytic onepulse per row including shared/unresolved; this fixed-event replay is not new feedback or prediction of support',baseline='0 new neutral integrations; original round13 neutral, reused exact-gated TRAIN000 first50ms cache',release_s=.05,budget=dict(main_trials=1,body_steps=108000,neural_steps=2300,model_compile=2,runtime=1,training=0,new_neutral=0,offline_compile=1,offline_forward=13),support=dict(horizon_s=.05,deadband_weight=1e-8,meanfoot=.8,minheight=.8,early_tau_s=.02),frozen_reference=js(C/'FROZEN_REFERENCE.json'),side_effects='all other pools n<5 also attenuated; shared duplicates same gain, sixteen unmapped remain rawcommand0; actual muscle floor persists; cannot attribute body response solely to LF',prohibitions='no default module/config/JSON change, no geometry/muscle/limits/force changes, no reference/target/PD/phase input to decoder, no live added forward, no checkpoint reset',status='PRE_REGISTERED_BEFORE_OFFLINE_AND_BODY'))
 shutil.copyfile(__file__,O/'EXECUTION_SOURCE.py');print('PREPARED',flush=True)

def extra_verifier():
 return '''# Full offline replay semantic comparison and shared constraints.
schema=js(O/'POOL_GAIN_SCHEMA.json');assert len(schema['groups'])==66 and schema['overlap_pairs']==0
for group in schema['groups']:
 ids=group['actuator_indices'];assert all(gain[i]==gain[ids[0]] for i in ids)
for name in ['BASELINE','K5']:
 saved=ar(O/('OFFLINE_'+name+'.npz'));ev=ar(B/'NEURAL_RASTER.npz')['spikes'][:2300];rr=np.zeros(90);pp=np.zeros(90);gg=np.ones(90) if name=='BASELINE' else gain
 for j,spike in enumerate(ev):
  ii=np.array([math.fsum(float(spike[i]) for i in row['local_indices'])/len(row['local_indices'])/.0005 if row['local_indices'] else 0 for row in mapping['rows']]);rr=decay*rr+(1-decay)*ii
  assert np.max(abs(rr-saved['rates_halfstep_hz'][j]))<1e-10
  if j%2:
   assert np.max(abs(pp-saved['applied'][j//2]))<1e-14;pp=np.clip((rr/100)*gg,0,1);assert np.max(abs(pp-saved['next'][j//2]))<1e-14
# Distinguishing example rules out erroneous postclip scaling.
assert min(1.,250./100*.4)==1. and .4*min(1.,250./100)==.4
assert len(compiled0)==469 and js(f/'MODEL_OBJECT_IDENTITY.json')['same_object']
assert not body['assist'][2600:].any() and not body['external'][2600:].any()
# Motion and all additional reported native metrics independently recomputed.
target=ar(Path('/home/sensen/Desktop/engine修复验收/s4-invariant-motion-target/20261102T023600Z/INVARIANT_BENDING_TARGETS.npz'))
for record,folder in zip(records_out,[C/'TRAIN_000_BASELINE',f]):
 rows=ar(folder/'BODYLOSS_NATIVE.npz')['rows'][:4001];r=rows[1:];fk=ar(folder/'RELEASE_FK.npz');ix=[int(np.argmin(abs(fk['time']-(1.3+t)))) for t in target['time_relative_s'][:7]];points=fk['points'][ix];angles=np.empty((7,6,2))
 for a in range(7):
  for b in range(6):
   for c in range(2):
    u=points[a,b,c]-points[a,b,c+1];v=points[a,b,c+2]-points[a,b,c+1];angles[a,b,c]=math.atan2(float(np.linalg.norm(np.cross(u,v))),math.fsum(float(x*y) for x,y in zip(u,v)))
 assert abs(float(np.mean((angles-target['angles_rad'][:7])**2))-record['motion_loss'])<1e-12
 for key,value in [('mean_positive_nonfoot_weight',r[:,7].mean()/W),('COMz_last',r[-1,10]),('COMz_min',r[:,10].min()),('ctrl_mean_square',r[:,18].mean())]:assert abs(record[key]-value)<1e-12
 exact(record['per_leg_mean_foot_weight'],r[:,1:7].mean(0)/W);exact(record['root_last'],r[-1,11:18])
'''

def offline():
 pr=js(O/'PREREGISTRATION.json');assert not (O/LABEL).exists()
 for p,h in pr['source_sha256'].items():assert sha(p)==h,p
 rows=js(B/'RUN/MAPPING.json')['rows'];g=arr(O/'FIXED_GAIN_VECTOR.npz')['per_actuator_gain'];sp=arr(B/'NEURAL_RASTER.npz')['spikes'][:2300];old=replay(sp,rows,np.ones(90));new=replay(sp,rows,g)
 records=js(B/'RUN/PREWARM_NEURAL_RECORDS.json')+js(B/'RUN/NEURAL_RECORDS.json')[:50]
 assert len(records)==1150
 for key in ['applied','next']:
  src=np.array([r['applied_command' if key=='applied' else 'decoded_next_command'] for r in records]);assert np.max(abs(old[key]-src))<1e-14
 np.savez_compressed(O/'OFFLINE_BASELINE.npz',**old);np.savez_compressed(O/'OFFLINE_K5.npz',**new)
 d=math.exp(-.0005/.005);analytic=[]
 for i,r in enumerate(rows):
  n=len(r['local_indices']);a=0. if not n else (1-d)/(n*.0005)/100.
  analytic.append(dict(actuator=r['actuator'],present_n=n,gain=g[i],spike_second_half_old=min(1.,a),spike_second_half_K5=min(1.,a*g[i]),spike_first_half_old=min(1.,a*d),spike_first_half_K5=min(1.,a*d*g[i]),unclipped_rate_increment_first_half_hz=a*d*100))
 summary={}
 for name,z in [('BASELINE',old),('K5',new)]:
  summary[name]=dict(prewarm_saturated_elements=int((z['applied'][:1100]>=1).sum()),release_saturated_elements=int((z['applied'][1100:]>=1).sum()),early_release_applied=z['applied'][1100:1120],release_mean=z['applied'][1100:].mean(0))
 dump(O/'OFFLINE_RESPONSE.json',dict(onepulse=analytic,replay=summary,counts='per actuator element, not unique groups',clip_semantic_counterexample=dict(rate_hz=250,gain=.4,correct_preclip=1.,incorrect_postclip=.4),source_sha256={str(B/'NEURAL_RASTER.npz'):sha(B/'NEURAL_RASTER.npz')},no_body_feedback=True,engineering_hypothesis=True,not_support_prediction=True))
 print('OFFLINE_COMPLETE',json.dumps({k:{x:y for x,y in v.items() if x.endswith('elements')} for k,v in summary.items()}),flush=True)

def run():
 import train_bending_closedloop as old
 from engine.embodied.motor_output import MotorOutputConfig
 pr=js(O/'PREREGISTRATION.json');assert (O/'OFFLINE_RESPONSE.json').exists()
 for p,h in pr['source_sha256'].items():assert sha(p)==h,p
 scope=dict(old.__dict__);scope.update(O=O,MODEL_IDS=[],MOTOR_CONFIG=MotorOutputConfig(per_actuator_gain=tuple(arr(O/'FIXED_GAIN_VECTOR.npz')['per_actuator_gain'])))
 exec(compile((O/'TRIAL_EXECUTION_SOURCE.py').read_text(),str(O/'TRIAL_EXECUTION_SOURCE.py'),'exec'),scope)
 scope['trial'](LABEL,np.zeros(12),.05)
 import calibrate_antagonist_mn_support as prior
 prior.O=O;prior.score_all()
 print('MAIN_COMPLETE',flush=True)

def report():
 result=js(O/'RESULT.json');exe=js(O/LABEL/'EXECUTION.json');verification=js(O/'INDEPENDENT_ARRAY_VERIFICATION.json');off=js(O/'OFFLINE_ACCOUNTING.json')
 dump(O/'RESOURCE_ACCOUNTING.json',dict(main=exe,offline=off,independent_array=dict(model_compile=0,body_step=0,neural_step=0,forward=0),new_neutral=0,training=0,optimizer=0,main_trials=1,scope='Runner counts actual, observer contactForce additional; not a comprehensive global hook'))
 text='# S4 小pool脉冲译码灵敏度：固定K5工程校准\n\n唯一新trial使用原中性root888/3470/WeWi*.2/offset0，从fresh神经t0显式传入MotorOutputConfig.per_actuator_gain=min(1,n_present/5)，16unresolved为0。tau5ms、fullscale100Hz、完成窗和pending调度不变。先原始rate滤波，再gain，再clip；不是缩放已clip命令。来源是实际present MN，partial不补缺roots。66源池、8对共享执行器同gain，不复制网络权重；其余所有n<5小池也被衰减，不能把身体变化单归LF。参数是工程假设，生理标定NOT_ESTABLISHED。\n\n历史已有其它分支的S233/S237/S253/S254/S256译码gain校准，本轮不是首次gain实验；本轮独有的对照为同原3470/root888完整预热与actual-present K5 floor。旧FIR/65DOF/bodybranch不能代替当前baseline。前轮7MN阈值±1未叠加。\n\n离线先登记再逐2300tick同源事件重建original/K5全部rawrate与clip命令、饱和及早序列。2MN first-half onepulse .86106665降至.34442666，5MN同样.34442666；完整滤波历史可能饱和，不能用gain×clipped代替。离线只接口核数，不是新身体反馈或承重预测；零事件freshfilter输出0，既有事件记忆按5ms衰减，不引入外置持续drive。\n\n原1.3s参考算法预热+.05s神经释放，同body/net/decoder连续交接、无checkpointreset；释放无PD/phase/target/外力、原肌肉和geometry/limits/dt保持。原neutral只复用TRAIN000前50ms及第13来源，0新baseline积分。\n\n实际评价：\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n支撑使用固定原h0/q0/W、50ms/deadband1e-8/meanfoot.8/minheight.8/tau.02；动作7样本不能抵消失稳。admitted为null时如实不承认成功，未证明自主站立或行走。完整release原生6signedfootFz/逐contact正NFz之和/COMz/rootquat/ctrlmean²/zeroassist保存；.5ms全部q/v/act/force、2300raster及初边末rates/pending/CSR/469compiled同对象初末均独立新进程purearray核验。离线额外1compile13savedforward的实际force/moment与保存force匹配；未改变live调度。\n\nLF两个负池MN各时段事件、原始filteredrate/command/activation/实际force与正池5MN响应，及foot/NF载荷，见POOL_RESPONSE.json和SELECTED_MUSCLE_MOMENT.npz；全体小池副作用见gain schema，不能单因解释。\n\n父review前未seal；未委派，无新图像/视频或目视通过声明。执行失败如有写FAILURE_LEDGER.json，源保持可追溯。重要程度：入口、预登记、gain映射/源身份、全部科学数组/执行源/独立核验为核心必需；报告/资源/清理/manifest为重要辅助。\n'
 (O/'REPORT.zh-CN.md').write_text(text)
 dump(O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='No confirmed disposable files; executable adapters and evidence remain necessary for verification/parent review; old packages unchanged'))
 if not (O/'FAILURE_LEDGER.json').exists():dump(O/'FAILURE_LEDGER.json',dict(failures=[],note='No scientific retry; only one new body trial'))
 dump(O/'FILE_INVENTORY.json',{str(p):dict(importance='重要辅助' if p.name in ['REPORT.zh-CN.md','RESOURCE_ACCOUNTING.json','CLEANUP_LEDGER.json','FILE_INVENTORY.json','MANIFEST_SHA256.json','FAILURE_LEDGER.json'] else '核心必需',purpose='K5 scientific evidence, source provenance or independent audit') for p in O.rglob('*') if p.is_file()})
 print('REPORT_COMPLETE',verification['passed'],flush=True)

def responses():
 f=O/LABEL;sp=np.unpackbits(arr(f/'RASTER.npz')['packed'],axis=1,count=12000);records=js(f/'RUN/PREWARM_NEURAL_RECORDS.json')+js(f/'RUN/NEURAL_RECORDS.json');body=arr(f/'RUN/BODY.npz');rows=js(B/'RUN/MAPPING.json')['rows'];out=[]
 for row in rows:
  if row['actuator'] not in ['LFTibia_extensor_93932','LFTibia_flex_93434']:continue
  i=row['actuator_index'];nodes=row['local_indices'];times=np.arange(1,2301)*.0005-.0;windows=[]
  for name,lo,hi in [('prewarm',0,1.1),('release_0_5ms',1.1,1.105),('release_5_10ms',1.105,1.11),('release_10_18ms',1.11,1.118),('release_18_50ms',1.118,1.15)]:
   ix=(times>lo+1e-10)&(times<=hi+1e-10);windows.append(dict(name=name,per_MN_count=sp[ix][:,nodes].sum(0),spike_end_times_s=[(times[np.flatnonzero(sp[:,node])]-1.1).tolist() for node in nodes],selection='counts use (lo,hi], listed event times cover full history'))
  out.append(dict(actuator=row['actuator'],local_indices=nodes,root_ids=row['tier_root_ids'],windows=windows,record_absolute_start_s=[r['absolute_start_s'] for r in records],filtered_rate_hz=[r['decoder']['rates_hz'][i] for r in records],applied=[r['applied_command'][i] for r in records],decoded_next=[r['decoded_next_command'][i] for r in records],body_time=body['time'],activation=body['activation'][:,i],actual_force=body['muscle_force'][:,i],command=body['command'][:,i]))
 dump(O/'POOL_RESPONSE.json',dict(pools=out,collateral='all n<5 pools attenuated, not LF-only intervention',foot_and_NF_native=str(f/'BODYLOSS_NATIVE.npz'),body_force_is_native_total_active_plus_bias=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','offline','run','responses','report']);a=p.parse_args()
 globals()[a.mode]()
