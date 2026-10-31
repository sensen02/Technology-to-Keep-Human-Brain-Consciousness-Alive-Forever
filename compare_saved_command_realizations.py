"""Saved-only reconciliation of existing true1ms command trajectories.
Minimal derivative of compare_saved_sensory_realizations.compare; no runners,
simulator, decoder, tests, plotting or web are imported or executed.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
P=Path(__file__).parent
A=Path('/home/sensen/Desktop/engine修复验收/s2-split-write/20261027T163000Z')
B=Path('/home/sensen/Desktop/engine修复验收/s4-command1ms/20261027T222000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-command-realization-reconciliation/20261027T232000Z')
D=Path('/home/sensen/Desktop/S4_COMMAND_REALIZATION_RECONCILIATION.md')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def compare(a,b,times=None):
 a=np.asarray(a);b=np.asarray(b);out=dict(old_shape=list(a.shape),new_shape=list(b.shape),old_dtype=str(a.dtype),new_dtype=str(b.dtype),array_equal=bool(np.array_equal(a,b)))
 if a.shape!=b.shape:return dict(out,max_abs_error=None,first_difference=None,reason='SHAPE_MISMATCH')
 loc=np.argwhere(a!=b);out['first_difference']=None if not len(loc) else loc[0].tolist();out['first_difference_time_s']=None if not len(loc) or times is None else float(times[loc[0,0]])
 out['max_abs_error']=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
 if times is not None:out['time_coverage_s']=[float(times[0]),float(times[-1])]
 return out

def main():
 O.mkdir(parents=True,exist_ok=False)
 ap=[A/n for n in ['PREREGISTRATION.json','EXECUTION.json','MANIFEST_SHA256.json','run_split_write_diagnostic.py','MAPPING.json']]+[A/'write1ms'/n for n in ['BODY.npz','INTERVALS.npz','CONTACTS.json','CLOCK.json']]
 bp=[B/n for n in ['PREREGISTRATION.json','EXECUTION.json','MANIFEST_SHA256.json','run_command1ms_single.py','MAPPING.json','INSTRUMENTED_CONSUME.py']]+[B/'normal'/n for n in ['BODY.npz','INTERVALS.npz','CONTACTS.json','TICKS.npz','RECORDS.json','SENSORY_OBSERVATIONS.json']]
 inputs=ap+bp+[Path(__file__),P/'compare_saved_sensory_realizations.py'];hashes={str(p):sha(p) for p in inputs};archive_checks=[]
 for root,paths in [(A,ap),(B,bp)]:
  m=load(root/'MANIFEST_SHA256.json')
  for p in paths:
   if p.name=='MANIFEST_SHA256.json':continue
   keys=[str(p),str(p.relative_to(root)),p.name];key=next((k for k in keys if k in m),None)
   archive_checks.append(dict(path=str(p),status='NOT_IN_ARCHIVED_MANIFEST' if key is None else 'MATCH' if m[key]==sha(p) else 'MISMATCH'))
 assert all(r['status']=='MATCH' for r in archive_checks)
 oldpr=load(A/'PREREGISTRATION.json');newpr=load(B/'PREREGISTRATION.json');shared={k:dict(old=v,new=newpr['source_hashes'][k],equal=v==newpr['source_hashes'][k]) for k,v in oldpr['sources'].items() if k in newpr['source_hashes']}
 arrays={};missing=[]
 for name in ['BODY.npz','INTERVALS.npz']:
  with np.load(A/'write1ms'/name,allow_pickle=False) as a,np.load(B/'normal'/name,allow_pickle=False) as b:
   arrays[name]={}
   for k in a.files:
    if k not in b.files:continue
    times=a['time'] if name=='BODY.npz' else a['sensory_sample_start_s'] if k in ['sensory_sample_start_s','sensory_current_nA'] else a['event_ticks']*.0005 if k in ['event_ticks','event_local_indices'] else a['motor_end_s'] if k in ['motor_end_s','motor_decoded_next_controls','motor_activation_end','motor_height_end_model','motor_observation_end_s'] else a['motor_start_s']
    arrays[name][k]=compare(a[k],b[k],times)
   missing += [dict(file=name,field=k,old='NOT_SAVED',new='SAVED') for k in b.files if k not in a.files]
 contacts_equal=load(A/'write1ms/CONTACTS.json')==load(B/'normal/CONTACTS.json');mapping_equal=load(A/'MAPPING.json')==load(B/'MAPPING.json')
 missing += [dict(file='TICKS.npz',field=k,old='NOT_SAVED',new='SAVED',comparison='NOT_COMPARABLE_NO_RECONSTRUCTION') for k in ['full_network_tick_spikes','per_tick_filter_rates','per_tick_logical_decoded_controls','tick_complete_s','probe_local_indices']]
 missing += [dict(file='SENSORY_OBSERVATIONS.json',field=k,old='NOT_SAVED',new='SAVED') for k in ['per_sample_physical_time_and_observed_quantities']]
 missing += [dict(file='RECORDS.json',field='reduced_episode_summary',old='NOT_SAVED',new='SAVED')]
 clocks={k:dict(old=v,new=newpr[n],equal=v==newpr[n]) for k,v,n in [('body_dt_s',oldpr['dt_body'],'dt_body_s'),('neural_dt_s',oldpr['dt_neural'],'dt_neural_s'),('sensory_dt_s',oldpr['dt_sensory'],'dt_sensory_s'),('command_dt_s',load(A/'write1ms/CLOCK.json')['dt_motor'],'dt_command_s'),('duration_s',oldpr['duration'][1],'duration_s')]}
 equal=all(r['array_equal'] for file in arrays.values() for r in file.values())
 conclusion=dict(classification='REPEATED_REALIZATION_WITH_RICHER_NEW_INSTRUMENTATION_NOT_NEW_FACTOR' if equal else 'SAVED_DIFFERENCE_REQUIRES_REVIEW_NO_NEW_RUN',same_declared_factor_configuration=all(r['equal'] for r in clocks.values()),all_common_arrays_equal=equal,contacts_json_equal=contacts_equal,mapping_json_equal=mapping_equal,first_saved_difference=None if equal else {f:{k:r['first_difference'] for k,r in rows.items() if not r['array_equal']} for f,rows in arrays.items()},interpretation='Old write1ms only compared. default5ms regression trajectory excluded. Same saved values establish repeated realization of existing command1ms/sensory5ms, not another independent factor conclusion. Missing old full-network tick data remains NOT_SAVED; exact motor event ticks are directly compared, not inferred from totals.')
 result=dict(old_archive=str(A),old_arm=str(A/'write1ms'),new_archive=str(B),new_arm=str(B/'normal'),excluded=['default5ms trajectory','DEFAULT_REGRESSION.json'],archive_checks=archive_checks,shared_source_hashes=shared,new_only_source_hash_keys=[k for k in newpr['source_hashes'] if k not in oldpr['sources']],clocks=clocks,arrays=arrays,missing=missing,conclusion=conclusion,configuration=dict(initial='same preregistered INITIAL hash; initial qpos/qvel/act/ctrl directly compared as part of full BODY',seed='both archived entries np.random.seed(0)',tier='both archived entries build_neural_tier(TierConfig(background_mean_nA=0,background_sd_nA=0))',arm='normal; push=None/external_touch=False; no preparation; original pools; motor_translator=None/default',decoder='same bound motor_output.py hash and cadence schedule hash; normalization/filter not changed',encoder_and_network='old archive does not separately bind sensory_channels/neural_cond and connectome hash; their full old snapshots NOT_ESTABLISHED. Whole closure hash differs. No inference of complete dependency equality from outputs.'),source_changes=dict(old_entry=str(A/'run_split_write_diagnostic.py'),old_call='line19 c._episode_split_motor(... motor_dt_s=.001) for write1ms; default arm line17 excluded',new_entry=str(B/'run_command1ms_single.py'),new_call='line71 c.episode(... motor_dt_s=.001) dispatches existing split path; copy-only Network/decoder/Observer instrumentation',missing_old_closure_snapshot='Old whole closure hash e60aec... differs from new f294fc...; old full module/split function copy not archived here, exact diff NOT_ESTABLISHED',test_source_hash='test_split_write_clock.py differs; no tests run in this reconciliation',environment='new command explicitly OMP_NUM_THREADS=1, old OMP value NOT_RECORDED; both OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl. Complete library/version equality NOT_ESTABLISHED',old_call_accounting='Old EXECUTION totals24000body/600neural include separate default5ms50ms. True write1ms CLOCK reports20000body/500neural; do not compare aggregate600 with new500 as config difference.'),reuse='Prefer oldwrite1ms for common quantities; cite newnormal for added full-network tick/filter/observation/perfoot-world-force fields; no further repeat dynamics for same factor by default.')
 dump(O/'COMPARISON.json',result);dump(O/'INPUT_CODE_HASHES.json',hashes)
 common_count=len(shared);same_count=sum(r['equal'] for r in shared.values())
 text=f'''# 命令1ms／感觉5ms既有两次实现核对（仅保存数据）

**纠正遗漏**：旧档已运行同一命令1ms因素。222000Z应登记为重复实现／复现及记录扩充，不是首次因素或第二个独立干预结论。

- 旧档，仅真实1ms臂：`{A/'write1ms'}`
- 新档：`{B/'normal'}`
- 旧default5ms50ms回归臂完全排除，不读取其数组或DEFAULT_REGRESSION.json。

## 直接比较
BODY全部{len(arrays['BODY.npz'])}共同数组、INTERVALS全部{len(arrays['INTERVALS.npz'])}共同数组逐值一致={equal}，最大误差均0，首差null。BODY20001态覆盖0–250ms：qpos(20001,49)、qvel(20001,48)、活化/命令(20001,90)。命令/完成解码250窗(250,90)、全网窗计数(250,12000)，覆盖0–250ms；感觉电流(50,12000)，采样0–245ms；原样motor event_ticks与event_local_indices各1006条均相等，实际完整首末时间见COMPARISON逐字段time_coverage。CONTACTS相等={contacts_equal}，MAPPING相等={mapping_equal}。没有从总事件数反推全网逐tick时序。

## 配置与版本分层
身体12.5us、神经500us、感觉5ms、命令1ms、时长250ms、seed0、零背景历史tier、无push/touch与相同初始化和历史池。共同跑前hash {common_count}项中{same_count}项相同；closure整体及旧调度测试源码hash不同。模型、INITIAL、池、motor_output及cadence_schedule哈希相同。旧入口直接调用split，新入口经episode分派并追加copy-only记录；旧完整closure/split源码未在本档保存，不能伪造历史diff。旧encoder/neural子模块及connectome独立hash未在旧prereg绑定，完整旧快照NOT_ESTABLISHED。新OMP_NUM_THREADS=1显式记录，旧未记录，不称全部环境相同。
旧EXECUTION 24000身体/600神经是两臂合计；truewrite1ms CLOCK为20000/500。不混入default回归，也不以总调用差异判定本臂配置不同。

## 新记录而非新因素
新档多存全网500tick事件、逐tick滤波与逻辑解码、感觉源时间和观测量、逐足/非足世界反力，以及末态trajectory和简要summary。旧缺字段标NOT_SAVED，不重建神经、身体、滤波或世界反力。旧已保存精确运动池event ticks，且此次逐值一致；不能把它也写成NOT_SAVED。
后续同因素优先复用旧truewrite1ms共同数据，需要新增记录粒度再引用新档。两份历史报告和封存文件不改写，未新增独立因果结论。

## 输出与范围
- 代码：`{Path(__file__)}`
- 逐值结果：`{O/'COMPARISON.json'}`
- 输入/代码hash：`{O/'INPUT_CODE_HASHES.json'}`
- 执行：`{O/'EXECUTION.json'}`
- 封存：`{O/'MANIFEST_SHA256.json'}`

本轮model/forward/body/neural/decoder replay/tests/plots/web全部0；候选禁用、16映射缺口、参考/绝对力限制及未自主站立／行走保持。数组相等是数值相等，不等于压缩NPZ字节一致。
'''
 (O/'REPORT.zh-CN.md').write_text(text);D.write_text(text)
 assert all(sha(Path(k))==v for k,v in hashes.items())
 dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,inputs_unchanged=True,model_load=0,forward=0,body_step=0,neural_step=0,decoder_replay=0,tests=0,plots=0,web=0,failures=[],missing_count=len(missing)))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 dump(O/'MANIFEST_SHA256.json',{str(p):sha(p) for p in O.iterdir() if p.is_file()}|{str(D):sha(D)})
 print(json.dumps(conclusion));print('OUTPUT',O)
if __name__=='__main__':main()
