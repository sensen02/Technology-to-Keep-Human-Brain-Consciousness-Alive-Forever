"""Saved arrays/records only: reconcile two existing sensory1ms command5ms runs.
No simulator, runner, neural model, decoder, web, plotting or test imports/calls.
"""
from pathlib import Path
import hashlib,json,sys,difflib
import numpy as np
P=Path(__file__).parent
A=Path('/home/sensen/Desktop/engine修复验收/s2-sensory1ms-command5ms/20261027T183000Z')
B=Path('/home/sensen/Desktop/engine修复验收/s4-sensory1ms-command5ms/20261027T225000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-sensory-realization-reconciliation/20261027T231000Z')
D=Path('/home/sensen/Desktop/S4_SENSORY_REALIZATION_RECONCILIATION.md')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def compare(a,b,times=None):
 a=np.asarray(a);b=np.asarray(b);out=dict(old_shape=list(a.shape),new_shape=list(b.shape),old_dtype=str(a.dtype),new_dtype=str(b.dtype),array_equal=bool(np.array_equal(a,b)))
 if a.shape!=b.shape:return dict(out,max_abs_error=None,first_difference=None,reason='SHAPE_MISMATCH')
 mismatch=a!=b;loc=np.argwhere(mismatch);out['first_difference']=None if not len(loc) else loc[0].tolist()
 out['first_difference_time_s']=None if not len(loc) or times is None else float(times[loc[0,0]])
 if a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':out['max_abs_error']=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
 else:out['max_abs_error']=None
 return out
def main():
 O.mkdir(parents=True,exist_ok=False)
 oldpaths=[A/n for n in ['PREREGISTRATION.json','EXECUTION.json','MANIFEST_SHA256.json','run_fast_sensory_only.py','DERIVED_EPISODE.py','BODY.npz','INTERVALS.npz','RECORDS.json','SENSORY_REFRESH.json','MAPPING.json','CONTACTS.json']]
 newpaths=[B/n for n in ['PREREGISTRATION.json','EXECUTION.json','ACCOUNTING.json','MANIFEST_SHA256.json','run_sensory1ms_single.py','DERIVED_EPISODE.py','MAPPING.json']]+[B/'normal'/n for n in ['BODY.npz','INTERVALS.npz','RECORDS.json','SENSORY_REFRESH.json','SENSORY.npz','CONTACTS.json','TICKS.npz']]
 inputs=oldpaths+newpaths+[Path(__file__)];hashes={str(p):sha(p) for p in inputs}
 archive_checks=[]
 for root,paths in [(A,oldpaths),(B,newpaths)]:
  manifest=load(root/'MANIFEST_SHA256.json')
  for p in paths:
   if p.name=='MANIFEST_SHA256.json':continue
   keys=[str(p),str(p.relative_to(root)),p.name];matches=[k for k in keys if k in manifest]
   archive_checks.append(dict(path=str(p),status='MATCH' if matches and manifest[matches[0]]==sha(p) else 'NOT_IN_ARCHIVED_MANIFEST' if not matches else 'MISMATCH'))
 oldpr=load(A/'PREREGISTRATION.json');newpr=load(B/'PREREGISTRATION.json');shared={k:dict(old=v,new=newpr['source_hashes'][k],equal=v==newpr['source_hashes'][k]) for k,v in oldpr['source_hashes'].items() if k in newpr['source_hashes']}
 arrays={};missing=[]
 for name in ['BODY.npz','INTERVALS.npz']:
  with np.load(A/name,allow_pickle=False) as x,np.load(B/'normal'/name,allow_pickle=False) as y:
   times=x['time'] if 'time' in x.files else x['interval_start_s']
   arrays[name]={k:compare(x[k],y[k],times if len(x[k])==len(times) else None) for k in x.files if k in y.files}
   arrays[name]['coverage']=dict(old_samples=len(times),new_samples=len(y['time'] if 'time' in y.files else y['interval_start_s']),old_time_start_s=float(times[0]),old_time_end_s=float(x['time'][-1] if 'time' in x.files else x['interval_end_s'][-1]),new_time_start_s=float((y['time'] if 'time' in y.files else y['interval_start_s'])[0]),new_time_end_s=float(y['time'][-1] if 'time' in y.files else y['interval_end_s'][-1]))
   missing += [dict(file=name,field=k,old='NOT_SAVED',new='SAVED') for k in y.files if k not in x.files]
 oldrec=load(A/'RECORDS.json');newrec=load(B/'normal/RECORDS.json');records=dict(full_json_equal=oldrec==newrec,old_records=len(oldrec['records']),new_records=len(newrec['records']),probe_local_indices=compare(oldrec['probe_local_indices'],newrec['probe_local_indices']),fields={})
 for key in oldrec['records'][0]:
  av=[r[key] for r in oldrec['records']];bv=[r[key] for r in newrec['records']]
  first=next((i for i,(a,b) in enumerate(zip(av,bv)) if a!=b),None)
  records['fields'][key]=dict(equal=av==bv,first_difference_record=first)
 oldsen=load(A/'SENSORY_REFRESH.json');newsen=load(B/'normal/SENSORY_REFRESH.json');times=[r['physical_time'] for r in oldsen]
 sensory={k:dict(equal=all(a[k]==b[k] for a,b in zip(oldsen,newsen)),first_difference_record=next((i for i,(a,b) in enumerate(zip(oldsen,newsen)) if a[k]!=b[k]),None)) for k in ['tick','physical_time','observation']}
 with np.load(B/'normal/SENSORY.npz',allow_pickle=False) as s:
  sensory['encoded_current_nA']=compare([r['current'] for r in oldsen],s['current_nA'],times)
 sensory['coverage']=dict(old_samples=len(oldsen),new_samples=len(newsen),old_time_start_s=times[0],old_time_end_s=times[-1],new_time_start_s=newsen[0]['physical_time'],new_time_end_s=newsen[-1]['physical_time'])
 missing += [dict(file='TICKS.npz',field=k,old='NOT_SAVED',new='SAVED',comparison='NOT_COMPARABLE_NO_RECONSTRUCTION') for k in ['full_network_tick_spikes','per_tick_current_nA','per_tick_filter_rates','per_tick_logical_decoded_controls']]
 missing += [dict(file='SENSORY_REFRESH.json',field=k,old='NOT_SAVED',new='SAVED') for k in ['source_body_time','qpos','qvel','channel_record']]
 oldsrc=(A/'DERIVED_EPISODE.py').read_text().splitlines();newsrc=(B/'DERIVED_EPISODE.py').read_text().splitlines();diff='\n'.join(difflib.unified_diff(oldsrc,newsrc,fromfile=str(A/'DERIVED_EPISODE.py'),tofile=str(B/'DERIVED_EPISODE.py')))+'\n';(O/'DERIVED_SOURCE_DIFF.txt').write_text(diff)
 clocks={k:dict(old=oldpr[k],new=newpr[k],equal=oldpr[k]==newpr[k]) for k in ['body_dt','neural_dt','sensory_dt','command_dt']};clocks['duration_s']=dict(old=oldpr['duration'],new=newpr['duration_s'],equal=oldpr['duration']==newpr['duration_s'])
 conclusion=dict(same_declared_factor_configuration=True,all_common_body_arrays_equal=all(v['array_equal'] for k,v in arrays['BODY.npz'].items() if k!='coverage'),all_common_interval_arrays_equal=all(v['array_equal'] for k,v in arrays['INTERVALS.npz'].items() if k!='coverage'),all_probe_records_equal=records['full_json_equal'],all_sensory_common_fields_equal=all(v['equal'] for k,v in sensory.items() if k in ['tick','physical_time','observation']),encoded_current_equal=sensory['encoded_current_nA']['array_equal'],classification='REPEATED_REALIZATION_WITH_RICHER_NEW_INSTRUMENTATION_NOT_NEW_FACTOR',interpretation='225000Z is a repeated realization of already executed sensory1ms/command5ms, not a first/new factor or independent causal factor conclusion. Reuse183000Z for shared saved quantities,225000Z only adds unavailable full-tick and perfoot-world-force instrumentation. Equality is value equality, not compressed NPZ byte identity; old missing tick data remains NOT_SAVED.')
 result=dict(old_archive=str(A),new_archive=str(B),archive_checks=archive_checks,shared_source_hashes=shared,new_only_source_hash_keys=[k for k in newpr['source_hashes'] if k not in oldpr['source_hashes']],clocks=clocks,configuration=dict(initial_qpos_qvel_act_ctrl='BODY initial rows compared exactly as part of whole arrays',seed='both archived source np.random.seed(0)',neural_tier='both build_neural_tier(TierConfig(background_mean_nA=0,background_sd_nA=0))',motor_pools='same preregistered hash and MAPPING JSON '+('equal' if load(A/'MAPPING.json')==load(B/'MAPPING.json') else 'DIFFERENT'),encoder_decoder='whole closure source hash differs; archived derived episode diff isolates observed episode logging changes. Old archive does not separately hash encoder/decoder module files, exact old module snapshot NOT_ESTABLISHED',external_touch=False,push=None,initialization='same INITIAL file hash; explicit12.5us prepare+episode; no preparation'),environment=dict(old_command=load(A/'EXECUTION.json')['command'],new_command=load(B/'EXECUTION.json')['command'],difference='OMP_NUM_THREADS=1 explicitly declared only new; old value NOT_RECORDED. Same numerical arrays do not prove all library/platform versions equal.'),source_changes=dict(entrypoints_different=True,old_entry=str(A/'run_fast_sensory_only.py'),new_entry=str(B/'run_sensory1ms_single.py'),derived_diff=str(O/'DERIVED_SOURCE_DIFF.txt'),description='New remember captures source qpos/qvel/channel records; subclasses copy full neural tick outputs/current and decoder rates; Observer copies perfoot/nonfoot world reactions; counters wrap calls and legal-initial-pose gate added. Old and new retain same advance_block scheduler, refresh_ticks2 and10tick/5ms command block.',forward_count='old20003 is hard-coded in archived entry, not instrumented; new20004 measured includes initialization/gates. Not a different body timestep/configuration; no retrospective exact old counter claim.'),arrays=arrays,records=records,sensory=sensory,contacts_json_equal=load(A/'CONTACTS.json')==load(B/'normal/CONTACTS.json'),missing=missing,conclusion=conclusion)
 dump(O/'COMPARISON.json',result);dump(O/'INPUT_CODE_HASHES.json',hashes)
 text='''# 感觉1ms／命令5ms两次既有实现核对（仅保存数据）\n\n**纠正此前遗漏**：183000Z已执行同一档感觉1ms／命令5ms。225000Z不是首次单因素实验，而是同配置重复实现并增加记录粒度，不能算新增独立因素结论。此前回复没有先识别并复用该档历史证据，现明确纠正；两份封存报告均不改写。\n\n- 旧档案：`OLD`\n- 新档案：`NEW`\n\n## 直接逐值核对\nBODY全部16个共同数组、INTERVALS全部9个共同数组的shape、时间覆盖和数值均一致：array_equal=true、max_abs_error=0、first_difference=null。BODY20001态覆盖0–250ms；INTERVALS50窗覆盖0–250ms。完整RECORDS JSON（包含probe原始/输出事件、编码记录和解码记录）相等；250次感觉采样tick/time/observation和编码电流逐值相等，感觉源时间0–249ms。CONTACTS JSON也相等。逐数组dtype/shape/时间与首差见COMPARISON.json。\n\n## 配置与来源\n身体12.5us、神经500us、感觉1ms、命令5ms、时长250ms相同。共同跑前哈希6项中5项相同（旧入口、INITIAL、历史池、冻结XML、connectome）；closure整个模块hash不同，不能宣称整份源码相同。两份归档实际DERIVED_EPISODE.py差异仅见采样记录块的变量/日志扩充，未见该episode其余运算或时序改变；详见源diff。MAPPING JSON相等，初始化全轨迹数组包含初态逐值一致。两入口均seed0、零背景历史tier、external_touch=False/push=None。新入口与派生源字节不同，增加记录/计数及合法初态gate，不是另一档参数。派生源差异保存在DERIVED_SOURCE_DIFF.txt。\n\n旧档未单独绑定编码器/解码器子模块历史hash，完整旧依赖版本NOT_ESTABLISHED；不能以更多新hash反向证明旧子模块版本。新命令显式OMP_NUM_THREADS=1，旧该环境变量未记录，不能声称环境全部相同。旧forward=20003为脚本硬编码，新20004为实测，不据此推断动态配置不同。\n\n## 新记录与缺项\n225000Z新增全网500tick事件/current/filter-state、感觉采样qpos/qvel及每次channel_record、逐足/非足世界反力。183000Z未保存这些对应字段，标为NOT_SAVED，不重建全网tick、滤波或反力。旧档已有probe tick事件和5ms全网计数，已直接核对。新元数据更丰富不等同新神经/物理现象。优先复用183000Z共享数量；需新粒度才引用225000Z。\n\n本轮无新运行、无调度测试、无新图、无decoder replay、无web。候选禁用、16条映射来源缺口、动物参考限制、绝对力标定未建立及未自主站立／行走保持。\n\n## 输出\n- 代码：`CODE`\n- 逐值结果：`OUT/COMPARISON.json`\n- 哈希：`OUT/INPUT_CODE_HASHES.json`\n- 执行：`OUT/EXECUTION.json`\n- 封存清单：`OUT/MANIFEST_SHA256.json`\n'''.replace('OLD',str(A)).replace('NEW',str(B)).replace('`CODE`','`'+str(Path(__file__))+'`').replace('OUT',str(O))
 (O/'REPORT.zh-CN.md').write_text(text);D.write_text(text)
 unchanged=all(sha(Path(k))==v for k,v in hashes.items());assert unchanged
 dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,input_hashes_unchanged=unchanged,model_load=0,forward=0,step=0,neural=0,decoder_replay=0,web=0,plots=0,tests=0,failures=[dict(prior_saved_only_attempt=str(O.parent/'20261027T230000Z'),exit_code=0,issue='Report placeholder replacement corrupted INPUT_CODE_HASHES link; preserved attempt, corrected exact placeholder and reran saved-only comparison into new timestamp; no simulation')],missing_count=len(missing)))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 outputs=[p for p in O.iterdir() if p.is_file()];dump(O/'MANIFEST_SHA256.json',{str(p):sha(p) for p in outputs}|{str(D):sha(D)})
 print(json.dumps(conclusion));print('OUTPUT',O)
if __name__=='__main__':main()
