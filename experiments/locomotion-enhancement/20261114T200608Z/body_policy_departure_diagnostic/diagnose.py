"""Parent savedarrays only, no model/Torch/Core/teacher solver import."""
from pathlib import Path
import argparse,ast,hashlib,json,os,resource,signal,struct,sys,time,zipfile
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
BODY=BASE/'state_aligned_motor_body'
BIND=BASE/'state_aligned_action_calibration/wall_replan_9000/INPUT_BINDINGS.json'
SHA='e9829e4b94f235889b44d495397d16fb072c9d43f8d2d9cffe91569d46956500'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def verify(h):
 for p,d in h.items():
  if sha(p)!=d:raise ValueError('Frozeninputdrift:'+p)
def header(p,k):
 with zipfile.ZipFile(p) as z,z.open(k+'.npy') as f:
  if f.read(6)!=b'\x93NUMPY':raise ValueError('NPYmagic')
  v=tuple(f.read(2));w=2 if v==(1,0) else 4;n=struct.unpack('<H' if w==2 else '<I',f.read(w))[0]
  if n>65536:raise ValueError('Headercap')
  h=ast.literal_eval(f.read(n).decode('latin1'))
 if h['fortran_order'] or 'O' in h['descr']:raise ValueError('Rowmajor_nonobject')
 return list(h['shape'])
def qualify():
 prereg=read(ROOT/'PREREG.json');binding=read(BIND);jobs=read(BODY/'JOBS.json');summary=read(BODY/'SUMMARY.json')
 if len(jobs)!=9 or any(j['exit_code']!=0 for j in jobs):raise ValueError('Actual9body_stages_exit0_required')
 paths=[BIND,BODY/'JOBS.json',BODY/'SUMMARY.json',ROOT/'PREREG.json',ROOT/'diagnose.py',ROOT/'MEMO.zh-CN.md'];entries=[]
 inherited={**binding['source_hashes'],**binding['data_hashes']}
 for condition,index in prereg['comparators'].items():
  e=binding['entries'][index];folder=Path(e['folder']);bf=BODY/'trials'/condition;br=read(bf/'RESULT.json');bv=read(bf/'VERIFY.json');sr=read(folder/'RESULT.json');sv=read(folder/('NN_VERIFY.json' if index==8 else 'VERIFY.json'))
  if e['id']!=('old_walk_control' if condition=='walk' else condition) or br['checkpoint_sha256']!=SHA or br['teacher_calls_after_handover']!=0:raise ValueError('Actualsource_case_identity')
  for result,path,v in ((br,bf/'RESULT.json',bv),(sr,folder/'RESULT.json',sv)):
   if not v['integrity_all_pass'] or not all(v['checks'].values()) or v['result_sha256']!=sha(path):raise ValueError('Actualfresharchiveaudit_required')
  for key in ('trace','neural_decisions'):
   a=br[key]
   if sha(a['path'])!=a['sha256']:raise ValueError('ActualbodyartifactSHA')
   paths.append(Path(a['path']))
  sourcekey='neural_decisions' if index==8 else 'teacher_samples'
  for key in ('trace',sourcekey):
   a=sr[key]
   if sha(a['path'])!=a['sha256'] or inherited.get(a['path'])!=a['sha256']:raise ValueError('Qualifiedtraining_sourceartifactSHA')
   paths.append(Path(a['path']))
  paths.extend([bf/'RESULT.json',bf/'VERIFY.json',bf/'IDENTITY.json',bf/'HANDOVER.json',folder/'RESULT.json',folder/('NN_VERIFY.json' if index==8 else 'VERIFY.json'),Path(e['identity_path'])])
  if header(br['trace']['path'],'target')!=[br['actual_ms'],42] or header(e['path'],'X238')[1:]!=[238]:raise ValueError('Actualheaderlengthshape')
  entries.append(dict(condition=condition,index=index,source=e,body=br,reference=sr))
 hashes={str(p):sha(p) for p in paths};verify(hashes)
 return prereg,binding,entries,hashes

def report(stat):
 lines=['# 新900身体轨迹偏离：保存数组描述诊断','','0NN／0Core／0物理／0教师查询。固定对照stop/L/R原教师、walk只用neutral old_walk_control(index8)。后续不同状态上的动作差只是跨轨迹距离，不是同状态策略误差。','','|条件|实际自主秒|初X逐位匹配|初动作RMS距离|初q/target匹配|','|---|---:|---|---:|---|']
 for r in stat['rows']:
  h=r['handover'];lines.append(f"|{r['condition']}|{r['body_duration_s']:.3f}|{h['X238_exact']}|{h['initial_action_RMS_distance']:.6g}|{h['q_target_exact']}|")
 lines+=['','每条固定时间网格、累计窗口RMS／每通道bias及持续20ms阈值事件见STAT.json。不可用尾窗明确标记，不补零、不外推；阈值是预登记描述尺度，不修改原行为门。COMdrop与非足接触出现顺序不能单独证明根因。','原teacherY作用于教师自己的X/T；首handover只有在X和实际机械状态匹配时才可视同状态初动作差，此后禁止把旧Y重新贴到student新X上当label。state352与load描述保留，但正足载是载荷代理，不是几何contactcount。','下一步有限on-policy teacher-label收集条件与原型范围见MEMO.zh-CN.md。本诊断不训练／改模型或门，不把first20×10是否充分从旧mask权重推断成因果结论。']
 return '\n'.join(lines)+'\n'

def main():
 p=argparse.ArgumentParser();p.add_argument('--selfcheck',action='store_true');p.add_argument('--execute-parent-saved-departure',action='store_true');a=p.parse_args();prereg,binding,entries,hashes=qualify()
 if a.selfcheck:print(json.dumps(dict(STDqualification=True,cases=len(entries),source_hashes=len(hashes),NN=0,Core=0,physical=0)));return
 if not a.execute_parent_saved_departure:p.error('Parent_savedarrays_flag_required')
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(key)!='1':raise ValueError('Threads1_required')
 for n in ('STAT.json','VERIFY.json','REPORT.zh-CN.md'):
  if (ROOT/n).exists():raise FileExistsError(ROOT/n)
 start=time.monotonic()
 def alarm(*unused):raise TimeoutError('Saveddeparture90s_wall_exceeded')
 oldhandler=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,90.)
 try:
  import numpy as np
  os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[];checks={}
  def guard():
   rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
   if rss>1500 or time.monotonic()-start>90:raise RuntimeError('90s_RSS1500_resource_exceeded')
   return rss
  def arrays(path,keys):
   with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
  def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
  def first_sustained(values,threshold,count,dt):
   v=np.asarray(values)>threshold
   if len(v)<count:return None
   candidates=np.flatnonzero(np.convolve(v.astype(np.int64),np.ones(count,dtype=np.int64),'valid')==count)
   return None if not len(candidates) else dict(start_s=float(candidates[0]*dt),confirmed_s=float((candidates[0]+count-1)*dt),threshold=float(threshold),sustain_ms=20)
  qa=binding['normalizer_mapping']['qpos_indices'];half=np.asarray(binding['normalizer_mapping']['halfspan42']);grid=prereg['time_grid_s'];tracekeys=('time','qpos','qvel','target','loads','nonfoot','com','padz','speed','action','control_source')
  for entry in entries:
   condition=entry['condition'];e=entry['source'];br=entry['body'];sr=entry['reference'];source8=entry['index']==8
   bt=arrays(br['trace']['path'],tracekeys);st=arrays(e['trace_path'],tracekeys);bn=arrays(br['neural_decisions']['path'],('time','physical_step_ms','X238','command','actually_applied_action','hidden_before','hidden_after','control_source'));sn=arrays(e['path'],('time','X238','actually_applied_action','command') if source8 else ('time','physical_ms','X238','Y42'))
   bms=bn['physical_step_ms'].astype(int);sms=np.rint(sn['time']*1000).astype(int)
   if not np.array_equal(bms,np.rint(bn['time']*1000).astype(int)) or not np.array_equal(sms,np.arange(len(sms))*5+(0 if source8 else 1000)):raise ValueError('Saveddecisionclock_mismatch')
   if len(bt['time'])!=br['actual_ms'] or not np.allclose(bt['time'],np.arange(1,len(bt['time'])+1)*.001,rtol=0,atol=1e-8):raise ValueError('Body1msclock')
   bi=np.flatnonzero(bms>=5800);si=np.flatnonzero(sms>=5800);N=len(bi);referenceindices=np.searchsorted(sms,bms[bi])
   if N==0 or np.any(referenceindices>=len(sms)) or not np.array_equal(sms[referenceindices],bms[bi]):raise ValueError('Exact_common_decision_times_required')
   commands=bn['command'][bi];expected=np.asarray([1.,0.] if source8 else [0.,0.] if condition=='stop' else [0.,1.] if condition=='left_pivot' else [0.,-1.])
   checks[condition+'_command_and_nohelper']=bool(np.all(commands==expected) and np.all(bn['control_source'][bi]==6))
   if source8:checks[condition+'_neutral_reference_command']=bool(np.all(sn['command'][referenceindices]==expected))
   action=bn['actually_applied_action'][bi];label=(sn['actually_applied_action'] if source8 else sn['Y42'])[referenceindices];adiff=action-label;X=bn['X238'][bi];SX=sn['X238'][referenceindices];pre=bms[bi]-1;qdiff=bt['qpos'][pre][:,qa]-st['qpos'][pre][:,qa];vdiff=bt['qvel'][pre][:,np.asarray(qa)-1]-st['qvel'][pre][:,np.asarray(qa)-1];tdiff=bt['target'][pre]-st['target'][pre]
   start=5799;end=len(bt['time']);bpost={k:v[5800:end] for k,v in bt.items()};weight=float(br['weight_native']);length=float(br['body_length']['length_native']);bodydur=(end-5800)*.001
   h=dict(X238_exact=bool(np.array_equal(X[0],SX[0])),q_target_exact=bool(np.array_equal(bt['qpos'][start],st['qpos'][start]) and np.array_equal(bt['target'][start],st['target'][start])),warm_X_exact=bool(np.array_equal(bn['X238'][200:1160],sn['X238'][200:1160] if source8 else sn['X238'][:960])),warm_qpos_target_exact=bool(np.array_equal(bt['qpos'][999:5800],st['qpos'][999:5800]) and np.array_equal(bt['target'][999:5800],st['target'][999:5800])),initial_action_RMS_distance=rms(adiff[0]),reference_old256_recorded=source8,initial_action_semantics='sameX_initialcomparison_only_if_X_and_mechanics_exact')
   checks[condition+'_matched_handover']=h['X238_exact'] and h['q_target_exact'] and h['warm_X_exact'] and h['warm_qpos_target_exact']
   points=[]
   for seconds in grid:
    decisionms=5800+int(round(seconds*1000));where=np.flatnonzero(bms==decisionms)
    if not len(where):points.append(dict(elapsed_s=seconds,available=False,reason='outside_actual_saved_decisions'));continue
    j=int(np.searchsorted(bms[bi],decisionms));idx=decisionms-1;loads=bt['loads'][idx];groups={name:rms(X[j,lo:hi]-SX[j,lo:hi]) for name,lo,hi in (('jointposition_normalized',0,42),('targeterror_normalized',42,84),('jointvelocity_normalized',84,126),('targetoffset_normalized',126,168),('footload_encoded',168,174),('clearance_and_padrate',183,195))}
    points.append(dict(elapsed_s=seconds,available=True,preaction_trace_index=idx,action_RMS_distance=rms(adiff[j]),action_signed_mean=float(adiff[j].mean()),joint_q_RMS_rad=rms(qdiff[j]),joint_qvel_RMS_rad_s=rms(vdiff[j]),target_RMS_rad=rms(tdiff[j]),COM_native=bt['com'][idx].tolist(),COM_drop_native=float(bt['com'][start,2]-bt['com'][idx,2]),nonfoot_over_weight=float(bt['nonfoot'][idx]/weight),support_over_weight=float(loads.sum()/weight),footloads_native=loads.tolist(),positive_load_legs=int((loads>0).sum()),padz_min_native=float(bt['padz'][idx].min()),hidden352_RMS=rms(bn['hidden_after'][where[0]]),hidden_new96_RMS=rms(bn['hidden_after'][where[0],256:]),observation_group_RMS=groups,action_distance_scope='cross_trajectory_not_same_state_policy_error_afterhandover'))
   windows=[]
   for seconds in prereg['windows_s']:
    if seconds>bodydur+1e-9:windows.append(dict(window_s=seconds,complete=False,available_body_s=bodydur));continue
    mask=(bms[bi]-5800)<int(round(seconds*1000));windows.append(dict(window_s=seconds,complete=True,action_RMS_distance=rms(adiff[mask]),action_signed_bias_per_channel=adiff[mask].mean(0).tolist(),joint_q_RMS_rad=rms(qdiff[mask]),target_RMS_rad=rms(tdiff[mask]),joint_qvel_RMS_rad_s=rms(vdiff[mask])))
   events={}
   for key,values,thresholds in (('action_RMS',np.sqrt(np.mean(adiff**2,axis=1)),prereg['descriptive_thresholds']['action_RMS']),('joint_q_RMS_rad',np.sqrt(np.mean(qdiff**2,axis=1)),prereg['descriptive_thresholds']['joint_q_RMS_rad']),('target_RMS_rad',np.sqrt(np.mean(tdiff**2,axis=1)),prereg['descriptive_thresholds']['target_RMS_rad']),('joint_qvel_RMS_rad_s',np.sqrt(np.mean(vdiff**2,axis=1)),prereg['descriptive_thresholds']['joint_qvel_RMS_rad_s'])):events[key]=[first_sustained(values,t,4,.005) for t in thresholds]
   events['COM_drop_native']=[first_sustained(bt['com'][start,2]-bpost['com'][:,2],t,20,.001) for t in prereg['descriptive_thresholds']['COM_drop_from_ownhandover_native']];events['nonfoot_over_weight']=[first_sustained(bpost['nonfoot']/weight,t,20,.001) for t in prereg['descriptive_thresholds']['nonfoot_over_weight']]
   checks[condition+'_arraysfinite']=all(np.isfinite(v).all() for d in (bt,st,bn,sn) for v in d.values())
   rows.append(dict(condition=condition,source_index=entry['index'],reference_id=e['id'],body_duration_s=bodydur,actual_body_ms=end,actual_decisions=N,handover=h,grid=points,windows=windows,descriptive_sustained_events=events,source_teacher_q_or_target_is_actual_reference_not_a_newlabel=True,contact_count_available=False,saved_fields=list(bt),teacher_source_qualified=True))
   del bt,st,bn,sn,bpost,action,label,adiff,X,SX,qdiff,tdiff,vdiff;guard()
  checks['no_NN_physics_backend_imports']=not any(n in sys.modules for n in ('torch','mujoco','physics','flygym','static_teacher'))
  verify(hashes);rss=guard();stat=dict(kind=prereg['kind'],checks=checks,rows=rows,source_hashes=hashes,wall_s=time.monotonic()-start,peak_RSS_MiB=rss,NN_forwards=0,Core_constructions=0,physical_steps=0,teacher_queries=0,training_updates=0,causal_rootcause_established=False)
  save(ROOT/'STAT.json',stat)
  with (ROOT/'REPORT.zh-CN.md').open('x') as f:f.write(report(stat))
  verify(hashes);guard();save(ROOT/'VERIFY.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,STAT_sha256=sha(ROOT/'STAT.json'),report_sha256=sha(ROOT/'REPORT.zh-CN.md'),source_hashes=hashes,NN_forwards=0,Core_constructions=0,physical_steps=0))
  if not all(checks.values()):raise ValueError('Saveddeparturequalification_failed')
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,oldhandler)
if __name__=='__main__':main()
