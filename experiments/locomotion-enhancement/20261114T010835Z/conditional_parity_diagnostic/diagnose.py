"""One authorized readonly CPU1 diagnostic. Never imports Core/physics/optimizer."""
from pathlib import Path
import os,sys,json,time,hashlib,traceback,resource,signal,inspect,ast
ROOT=Path(__file__).resolve().parent
REC=ROOT.parent/'target_feedback_calibration_env_recovery_v1'
ORIG=ROOT.parent/'target_feedback_calibration'
SHA='b562fcc3e39c2aa56381a87b17f1e80124f203fa1a73e3cdf8888778de119f14'
CHECKPOINT=REC/'TRAIN/checkpoint_000300.pt'
EXPECTED='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'
start=time.monotonic();result={'kind':'READONLY_CONDITIONAL_PARITY_DIAGNOSTIC_V1','status':'STARTED','training_updates':0,'optimizer_operations':0,'Core_constructions':0,'physical_steps':0,'cases':[]};arrays={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save():
 (ROOT/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
def gate(*_):
 if time.monotonic()-start>176:raise RuntimeError('Wall budget reached')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>1800:raise RuntimeError('RSS budget reached')
def alarm(*_):raise RuntimeError('Diagnostic wall limit reached')
signal.signal(signal.SIGALRM,alarm);signal.alarm(178)
try:
 if str(Path(sys.executable).absolute())!=EXPECTED:raise ValueError('Wrong Python')
 sys.path.insert(0,str(REC))
 import learning
 learning.cpu1()
 import torch,numpy as np
 import common,model
 from scan import make_scan
 from data import prepare
 from utils import production_feedback,output_errors
 from exact_components_scan import ExactComponentsScan
 result['environment']={'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE')},'affinity':sorted(os.sched_getaffinity(0))}
 bind=json.loads((REC/'INPUT_BINDINGS.json').read_text());source_before={p:sha(p) for p in bind['source_hashes']}
 result['source_hashes_before']=source_before;result['source_before_matches_binding']=source_before==bind['source_hashes']
 result['zero_checkpoint_existing_proof']={'path':str(REC/'TRAIN/VERIFY_000000.json'),'sha256':sha(REC/'TRAIN/VERIFY_000000.json'),'integrity_all_pass':json.loads((REC/'TRAIN/VERIFY_000000.json').read_text())['integrity_all_pass'],'rescanned':False}
 result['checkpoint_sha256_before']=sha(CHECKPOINT);assert result['checkpoint_sha256_before']==SHA
 rng=learning.rng_snapshot()
 print('STRICT_LOAD_BEGIN',flush=True);save()
 m,payload=common.load_checkpoint(CHECKPOINT,SHA,final=False)
 result['strict_load_rng_unchanged']=learning.rng_equal(rng,learning.rng_snapshot());result['step']=payload['step']
 weights=model.state_digest(m.state_dict());result['weights_digest_before']=weights
 result['optimizer_saved_digest_before']=hashlib.sha256(repr([(k,tuple(v.keys()),float(v['step'])) for k,v in payload['optimizer_state_dict']['state'].items()]).encode()).hexdigest()
 print('STRICT_LOAD_OK',time.monotonic()-start,flush=True);gate()
 # Exact whole schema preparation is retained; numerical output comparisons select only two cases.
 with torch.no_grad():data=prepare(m)
 print('DATA_PREPARED',time.monotonic()-start,flush=True);gate()
 result['schema']={k:list(data[k].shape) for k in ('x','physical','oldafter','old_nontarget','actual_pre_targets','actual_q')}
 result['source_statements']={'base_step':inspect.getsource(m.old.base.step),'combined_step':inspect.getsource(m.old.step)}
 assert 'self.head(torch.tanh(self.hidden(state))) - self.gain() * obs[..., 126:168]' in result['source_statements']['base_step']
 assert 'base_mean+negative*self.right_branch.bound*correction_right+positive*self.left_branch.bound*correction_left' in result['source_statements']['combined_step']
 py=make_scan(m,script=False);jit=make_scan(m,script=True);fixedpy=ExactComponentsScan(m);fixedjit=torch.jit.script(fixedpy)
 names=('mean','residual','h96','pretarget','posttarget','action')
 def args(d,forced=None):return (d['physical'],d['oldafter'],d['commands'],d['old_nontarget'],d['gain'],d['initial_target'],d['q0'],d['halfspan'],d['bounds'],d['max_rate'],forced)
 def fixargs(d,c,forced=None):return (d['physical'],d['oldafter'],d['commands'],*c,d['gain'],d['initial_target'],d['q0'],d['halfspan'],d['bounds'],d['max_rate'],forced)
 def errors(a,b):return {n:float((x-y).abs().max()) for n,x,y in zip(names,a,b)}
 def first(a,b):
  idx=(a!=b).nonzero()
  if not len(idx):return None
  at=tuple(int(v) for v in idx[0]);return {'index':list(at),'a':float(a[at]),'b':float(b[at]),'abs_error':float((a[at]-b[at]).abs())}
 def norm(t,d):return ((t-d['q0'])/d['halfspan']).clamp(-10,10)
 def divergence(a,b,d):
  na=norm(a[3],d);nb=norm(b[3],d);qa=na.float().double();qb=nb.float().double();f=first(qa,qb)
  z={'first_mean_double':first(a[0],b[0]),'first_target_double':first(a[3],b[3]),'first_float32_offset':f}
  if f:
   k,bb,j=f['index'];at=(k,bb,j);lo=min(f['a'],f['b']);hi=max(f['a'],f['b']);mid=(lo+hi)/2
   f.update({'tick_zero_based':k,'time_after_handover_s':k*.005,'channel_zero_based':j,'target_a':float(a[3][at]),'target_b':float(b[3][at]),'normalized_a':float(na[at]),'normalized_b':float(nb[at]),'midpoint_normalized':mid,'target_at_midpoint':float(d['q0'][j]+mid*d['halfspan'][j]),'normalized_minus_midpoint_a':float(na[at])-mid,'normalized_minus_midpoint_b':float(nb[at])-mid,'adjacent_float32':bool(np.nextafter(np.float32(lo),np.float32(np.inf))==np.float32(hi))})
   z['max_before_first_float32_offset']={n:float((x[:k]-y[:k]).abs().max()) if k else 0. for n,x,y in zip(names,a,b)}
   z['at_first_float32_offset']={n:float((x[k]-y[k]).abs().max()) for n,x,y in zip(names,a,b)}
  return z
 with torch.no_grad():
  for i,raw in enumerate(data['raw']):
   if raw['entry']['id'] not in ('stop','old_walk_control'):continue
   ident=raw['entry']['id'];print('CASE_BEGIN',ident,time.monotonic()-start,flush=True)
   d=learning.slice_data(data,i,i+1)
   full,oldfull=production_feedback(m,raw,data,i);full=tuple(v[:,None] for v in full)
   outpy=py(*args(d));outjit=jit(*args(d));forced=jit(*args(d,full[3]));forcedpy=py(*args(d,full[3]))
   # Frozen neural heads are evaluated with EXACT source operations on each actual after-state B1.
   base=[];right=[];left=[]
   for k,after in enumerate(d['oldafter']):
    if k%192==0:gate()
    bh=after[:,:128];rh=after[:,128:192];lh=after[:,192:256]
    turn=d['commands'][k,:,1:2];positive=torch.relu(turn);negative=torch.relu(-turn)
    base.append(m.old.base.head(torch.tanh(m.old.base.hidden(bh))))
    right.append(negative*m.old.right_branch.bound*torch.tanh(m.old.right_branch.head(rh)))
    left.append(positive*m.old.left_branch.bound*torch.tanh(m.old.left_branch.head(lh)))
   components=tuple(torch.stack(v) for v in (base,right,left))
   # Batched heads are an explicit separate probe, never substituted in the exact B1 path.
   after=d['oldafter'][:,0];turn=d['commands'][:,0,1:2];positive=torch.relu(turn);negative=torch.relu(-turn)
   batched=(m.old.base.head(torch.tanh(m.old.base.hidden(after[:,:128])))[:,None],(negative*m.old.right_branch.bound*torch.tanh(m.old.right_branch.head(after[:,128:192])))[:,None],(positive*m.old.left_branch.bound*torch.tanh(m.old.left_branch.head(after[:,192:256])))[:,None])
   actualoffset=d['x'][:,:,126:168]
   directactual=components[0]-d['gain']*actualoffset+components[1]+components[2]
   reconstructed=d['old_nontarget']-d['gain']*actualoffset
   fixed=fixedjit(*fixargs(d,components));fixed_python=fixedpy(*fixargs(d,components));fixed_forced=fixedjit(*fixargs(d,components,full[3]))
   batched_fixed=fixedjit(*fixargs(d,batched))
   row={'id':ident,'rows':len(raw['x']),'old256exact':torch.equal(oldfull,d['oldafter'][:,0]),'original_JIT_vs_full':errors(outjit,full),'original_Python_vs_full':errors(outpy,full),'original_Python_vs_JIT':errors(outpy,outjit),'forced_fullproduction_pretarget_JIT_vs_full':errors(forced,full),'forced_fullproduction_pretarget_Python_vs_full':errors(forcedpy,full),'original_JIT_divergence':divergence(outjit,full,d),'original_Python_divergence':divergence(outpy,full,d),'direct_components_actual_oldmean_exact':torch.equal(directactual,d['oldmean']),'direct_components_actual_oldmean_max_error':float((directactual-d['oldmean']).abs().max()),'cancellation_actual_oldmean_max_error':float((reconstructed-d['oldmean']).abs().max()),'direct_components_JIT_vs_full':errors(fixed,full),'direct_components_Python_vs_full':errors(fixed_python,full),'direct_components_Python_vs_JIT':errors(fixed_python,fixed),'direct_components_forced_vs_full':errors(fixed_forced,full),'direct_components_divergence':divergence(fixed,full,d),'batched_vs_B1_components_max_error':[float((a-b).abs().max()) for a,b in zip(batched,components)],'batched_components_vs_full':errors(batched_fixed,full)}
   row['original_pass']=max(row['original_JIT_vs_full'].values())<=1e-12 and row['old256exact']
   row['direct_components_pass']=max(row['direct_components_JIT_vs_full'].values())<=1e-12 and row['old256exact']
   # Compare full source oldmean at common production targets with exact components.
   fulloffset=norm(full[3],d).float().double()
   fullold=components[0]-d['gain']*fulloffset+components[1]+components[2]
   forcedold=d['old_nontarget']-d['gain']*fulloffset
   row['forced_oldmean_cancellation_max_error']=float((forcedold-fullold).abs().max())
   row['first_oldmean_cancellation_at_common_full_target']=first(forcedold,fullold)
   for n,a,b,c in zip(names,outjit,full,fixed):
    arrays[ident+'_'+n+'_max_abs_error_per_tick']=(a-b).abs().amax(dim=(1,2)).numpy()
    arrays[ident+'_exact_components_'+n+'_max_abs_error_per_tick']=(c-b).abs().amax(dim=(1,2)).numpy()
   f=row['original_JIT_divergence']['first_float32_offset'];k=f['index'][0] if f else 0;lo=max(0,k-2);hi=min(1920,k+3)
   for tag,outputs in (('original',outjit),('full',full),('direct_components',fixed)):
    for n,v in zip(names,outputs):arrays[ident+'_'+tag+'_'+n+'_first_offset_window']=v[lo:hi].numpy().copy()
   row['window_tick_range']=[lo,hi];result['cases'].append(row);save()
   print('CASE_DONE',ident,json.dumps({'original':row['original_JIT_vs_full'],'fixed':row['direct_components_JIT_vs_full'],'first_offset':f}),flush=True);gate()
 result['weights_digest_after']=model.state_digest(m.state_dict());result['weights_unchanged']=weights==result['weights_digest_after'];result['frozen_old_verified']=m.verify_old()
 result['optimizer_saved_digest_after']=hashlib.sha256(repr([(k,tuple(v.keys()),float(v['step'])) for k,v in payload['optimizer_state_dict']['state'].items()]).encode()).hexdigest()
 result['rng_unchanged_entire_diagnostic']=learning.rng_equal(rng,learning.rng_snapshot())
 result['checkpoint_sha256_after']=sha(CHECKPOINT)
 result['source_hashes_after']={p:sha(p) for p in source_before};result['source_hashes_unchanged']=source_before==result['source_hashes_after']
 result['no_physical_modules']=not any(n in sys.modules for n in ('mujoco','physics','flygym','static_teacher','student_wave'))
 result['all_direct_components_pass']=len(result['cases'])==2 and all(r['direct_components_pass'] for r in result['cases'])
 np.savez_compressed(ROOT/'FIRST_DIVERGENCE.npz',**arrays)
 result['arrays']={'path':str(ROOT/'FIRST_DIVERGENCE.npz'),'sha256':sha(ROOT/'FIRST_DIVERGENCE.npz')}
 result['status']='COMPLETED'
except BaseException as exc:
 result['status']='FAILED';result['exception_type']=type(exc).__name__;result['exception']=str(exc);result['traceback']=traceback.format_exc();print(result['traceback'],flush=True)
finally:
 signal.alarm(0)
 result['wall_s']=time.monotonic()-start;result['RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 result['diagnostic_source_hashes']={str(p):sha(p) for p in (ROOT/'PREREG.json',ROOT/'diagnose.py',ROOT/'exact_components_scan.py')}
 save()
 if result['status']=='COMPLETED':
  lines=['# 条件反馈数值诊断（只读，S2/S5）','',f"实际完成两条1920 tick；wall {result['wall_s']:.3f}s，RSS {result['RSS_MiB']:.2f}MiB；训练/优化器/Core/物理均0。",'',f"300 checkpoint SHA256：{SHA}。0步沿用已有PASS，不重扫。",'']
  for r in result['cases']:
   f=r['original_JIT_divergence']['first_float32_offset']
   lines += [f"## {r['id']}",'',f"原JIT相对fullproduction最大误差：{json.dumps(r['original_JIT_vs_full'])}",f"强制相同fullproduction pretarget最大误差：{json.dumps(r['forced_fullproduction_pretarget_JIT_vs_full'])}",f"Python/JIT最大误差：{json.dumps(r['original_Python_vs_JIT'])}",f"首float32 offset分歧：{json.dumps(f,ensure_ascii=False)}",f"分歧前误差：{json.dumps(r['original_JIT_divergence'].get('max_before_first_float32_offset'))}",f"old256 exact={r['old256exact']}；原运算顺序直接三项缓存误差：{json.dumps(r['direct_components_JIT_vs_full'])}；门通过={r['direct_components_pass']}。",f"B1实际oldmean直接还原逐位一致={r['direct_components_actual_oldmean_exact']}；抵消重构误差={r['cancellation_actual_oldmean_max_error']}。",f"批量与sequential B1三项残差={r['batched_vs_B1_components_max_error']}；批量替换后的误差={json.dumps(r['batched_components_vs_full'])}。",'']
  lines += [f"权重前后digest一致={result['weights_unchanged']}；source SHA前后一致={result['source_hashes_unchanged']}；RNG全程一致={result['rng_unchanged_entire_diagnostic']}。",'', '结论仅覆盖这两条固定物理观察的条件数值重放；直接三项是新文件测试，不修改部署，不代表全12条/全batch预检或物理行为通过。','',f"完整记录：{ROOT/'RESULT.json'}；首分歧窗口与逐tick误差：{ROOT/'FIRST_DIVERGENCE.npz'}。"]
 else:lines=['# 条件反馈数值诊断失败（保留完整性门）','',f"失败：{result.get('exception_type')}：{result.get('exception')}",'','未绕过loader、未重试、未改旧scope。已执行数值阶段详见RESULT.json；训练/优化器/Core/物理均0。','',result.get('traceback','')]
 (ROOT/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
 print('FINAL_STATUS',result['status'],'wall',result['wall_s'],'RSS',result['RSS_MiB'],flush=True)
if result['status']!='COMPLETED':sys.exit(1)
