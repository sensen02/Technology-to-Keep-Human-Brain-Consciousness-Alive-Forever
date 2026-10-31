"""Derive frozen candidates and test actual function evidence integration only."""
from pathlib import Path
import argparse,json,hashlib,difflib,importlib.util,sys,subprocess
from unittest.mock import patch
import numpy as np
from scipy import signal
from phase_metrology_evidence import collect_phase_evidence
P=Path(__file__).parent
C=Path('/home/sensen/Desktop/engine修复验收/s345-hilbert-order-candidate/20261027T123302Z')
I=Path('/home/sensen/Desktop/engine修复验收/s345-certification-recovery-regression/20261027T102654Z')
L=Path('/home/sensen/Desktop/engine修复验收/s345-la30-slip-diagnosis/20261027T080727Z/TRACES.npz')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def same(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:same(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 else:assert a==b,(a,b)

def derive(o,name):
 source=(C/name).read_text();new=source
 old='return_trace=False):';assert new.count(old)==1
 new=new.replace(old,'return_trace=False,return_evidence=False):')
 if name=='analyze_s345.py':
  anchor="    if return_trace:\n        result['trace']=dict"
  addition="    if return_evidence:\n        from phase_metrology_evidence import collect_phase_evidence\n        result['phase_evidence']=collect_phase_evidence(trim_s+np.arange(n)/fs,ha,hb,endpoint_samples=edge)\n"
 else:
  anchor="    if return_trace:\n        result['trace'] = dict"
  addition="    if return_evidence:\n        from phase_metrology_evidence import collect_phase_evidence\n        result['phase_evidence']=collect_phase_evidence(trim_s+np.arange(len(delta))/FS,ha,hb,endpoint_samples=edge)\n"
 assert new.count(anchor)==1;new=new.replace(anchor,addition+anchor)
 (o/name).write_text(new)
 (o/(name+'.diff')).write_text(''.join(difflib.unified_diff(source.splitlines(True),new.splitlines(True),fromfile=str(C/name),tofile=str(o/name))))
 return dict(source=str(C/name),old_sha256=sha(C/name),new_path=str(o/name),new_sha256=sha(o/name),reason='optional same-call analytic evidence, legacy statistics untouched')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False)
 names=('analyze_s345.py','audit_s345_independent.py')
 inputs=[C/n for n in names]+[I/(n+'.npz') for n in ('static_165','detuned_002','single_amplitude_collapse')]+[L]
 before={str(p):sha(p) for p in inputs}
 dump(o/'PREREGISTRATION.json',dict(input_hashes=before,scope='same-call optional evidence integration',native_main_fs=40,native_audit_fs=50,deterministic_adapter='fixed preexisting resample_poly 5/4 Kaiser5 constant0; not new observations',neural_ticks=0,physics_steps=0,certification_grid_runs=0,checks=['legacy equality','same-call array identity','strict JSON','exceptions propagate','NOT_ASSESSED not accepted']))
 lineage={n:derive(o,n) for n in names};dump(o/'LINEAGE.json',lineage)
 for name in ('phase_metrology.py','phase_metrology_evidence.py','test_phase_metrology.py',Path(__file__).name):(o/name).write_bytes((P/name).read_bytes())
 # Candidate imports resolve the frozen dependency copies, not mutable workspace files.
 sys.path.insert(0,str(o))
 for mod in ('phase_metrology','phase_metrology_evidence'):sys.modules.pop(mod,None)
 import phase_metrology_evidence as adapter
 from phase_metrology import measure_phase
 run=subprocess.run([sys.executable,'-m','unittest','-v','test_phase_metrology'],cwd=o,capture_output=True,text=True)
 (o/'UNIT_TEST.log').write_text(run.stdout+run.stderr);assert run.returncode==0
 z=np.load(L);rows=[]
 for impl,name in zip(('main','audit'),names):
  old=load(C/name,'frozen_'+impl);new=load(o/name,'integrated_'+impl);fs=40 if impl=='main' else 50
  def call(m,a,b,**kw):
   return m.phase_core(a,b,fs,4.,with_bootstrap=False,**kw) if impl=='main' else m.metrics(a,b,trim_s=4.,bootstrap=False,**kw)
  cases=[]
  for case in ('static_165','detuned_002','single_amplitude_collapse'):
   inp=np.load(I/(case+'.npz'));a=inp['input_A'];b=inp['input_B']
   if impl=='audit':a=signal.resample_poly(a,5,4,window=('kaiser',5),padtype='constant');b=signal.resample_poly(b,5,4,window=('kaiser',5),padtype='constant')
   cases.append((case,a,b))
  bm=25 if impl=='main' else 20
  cases.append(('LA30',z['raw_count_A'].reshape(-1,bm).sum(1),z['raw_count_B'].reshape(-1,bm).sum(1)))
  for case,a,b in cases:
   baseline=call(old,a,b);same(baseline,call(new,a,b))
   evidence_only=call(new,a,b,return_evidence=True);evidence_only.pop('phase_evidence');same(baseline,evidence_only)
   seen=[];analytic=[]
   # Capture actual Hilbert outputs produced within the candidate invocation.
   real_hilbert=signal.hilbert
   def spy_hilbert(*args,**kw):
    value=real_hilbert(*args,**kw);analytic.append(value);return value
   def spy(t,ha,hb,**kw):
    assert len(analytic)==2
    for value,full in zip((ha,hb),analytic):
     k=int(4*fs);assert np.shares_memory(value,full);np.testing.assert_array_equal(value,full[k:-k])
    evidence=adapter.collect_phase_evidence_original(t,ha,hb,**kw)
    direct=measure_phase(t,ha,hb,endpoint_samples=kw['endpoint_samples'])
    same(evidence['measurement'],adapter.json_value(direct));seen.append(True);return evidence
   adapter.collect_phase_evidence_original=adapter.collect_phase_evidence
   with patch.object(signal,'hilbert',side_effect=spy_hilbert),patch.object(adapter,'collect_phase_evidence',side_effect=spy):
    result=call(new,a,b,return_trace=True,return_evidence=True)
   assert seen==[True];ev=result.pop('phase_evidence');trace=result.pop('trace');same(baseline,result)
   traced=call(old,a,b,return_trace=True);same(trace,traced.pop('trace'));same(baseline,traced)
   assert ev['measurement']['locking_status']=='NOT_ASSESSED' and not ev['measurement']['continuous_branch_identified']
   assert 'passed' not in ev and 'certified' not in ev
   text=json.dumps(ev,allow_nan=False);same(ev,json.loads(text))
   dump(o/(impl+'_'+case+'_EVIDENCE.json'),ev)
   rows.append(dict(implementation=impl,case=case,legacy_equal=True,same_call_arrays_verified=True,status=ev['measurement']['status'],branch_intervals=ev['measurement']['branch_change_intervals'],legacy_slips=baseline['slip_count']))
  # Adapter failure must escape the candidate function (no silent complete evidence).
  with patch.object(adapter,'collect_phase_evidence',side_effect=RuntimeError('injected evidence failure')):
   try:call(new,a,b,return_evidence=True)
   except RuntimeError as err:assert str(err)=='injected evidence failure'
   else:raise AssertionError('candidate swallowed evidence exception')
  # Exact sampled analytic zero fixture in actual function path: input is existing LA30;
  # only preprocessing is stubbed. Not a claim that filtered collapse has an exact zero.
  last=json.loads((o/(impl+'_LA30_EVIDENCE.json')).read_text())
  ha=np.array(last['analytic_A']['real'])+1j*np.array(last['analytic_A']['imag']);hb=np.array(last['analytic_B']['real'])+1j*np.array(last['analytic_B']['imag']);ha[len(ha)//2]=0
  def fake_trim(*args,**kw):
   value=fake_trim.values.pop(0)
   return (value,len(value)/len(a)) if impl=='main' else value
  fake_trim.values=[ha,hb]
  with patch.object(new,'trimmed' if impl=='main' else 'band_trim',side_effect=fake_trim):r=call(new,a,b,return_evidence=True)
  ev=r['phase_evidence'];assert ev['measurement']['product_statistics'] is None
  assert ev['measurement']['status']=='UNDEFINED_AT_SAMPLED_ZERO'
  assert isinstance(r['slip_count'],int) and ev['measurement']['legacy_statistics_defined_at_samples'] is False
  dump(o/(impl+'_EXACT_ZERO_CONTRACT_FIXTURE.json'),ev)
 assert {str(p):sha(p) for p in inputs}==before
 dump(o/'RESULTS.json',dict(passed=True,scope='ENGINEERING_INTEGRATION_ONLY',unit_test_exit_code=run.returncode,integration_exit_code=0,cases=rows,exceptions_propagated=True,zero_contract_fixtures=2,source_hashes_unchanged=True,locking_acceptance='NOT_ASSESSED is never PASS',historical='28/36 FAIL, S345 NOT_JUDGED, strict exit1 unchanged'))
 (o/'CONTRACT.zh-CN.md').write_text('# 相位证据集成契约\n\n潜在载波相位：生成器振荡器的相位；只在明确生成公式及定义域内作确定性参考，不等于任意非正弦计数经滤波后的解析相位。\n滤后解析相位：本次候选函数对自身滤波结果作Hilbert产生的ha/hb的角度；证据直接接收这两个数组，不重算。\n离散展开路径：对有限相邻角度选择主支并累积；分别通道和复积两路径可差整圈。离散样本不能证明连续采样充分性、零点之间的绕转或真实神经锁相。\n\nreturn_evidence=False保持历史默认结果。启用后仅附加phase_evidence；旧统计和新measurement各存各的，异常传播。defined_at_samples只说明样本非零；连续支路始终UNIDENTIFIED、locking_status=NOT_ASSESSED，不是认证通过。NaN未定义点JSON为null并保留phase_defined，Infinity拒绝。无新增幅值门、插值或筛除。\n\n已有40Hz确定性输入进审计50Hz仅复用此前固定5/4适配，非原生50Hz输入认证。生成器幅值塌陷不保证滤后解析值恰零；另用明确标注的预处理桩零点fixture验证实际函数链序列化，不冒充自然输入结果。\n\n8次保存输入集成（两实现各3确定性+LA30）默认/证据/旧trace原值一致；同次Hilbert数组共享内存来源核验；2个实际函数异常传播测试、2个零点契约fixture。没有运行神经臂、物理或36格认证。28/36失败、S345 NOT_JUDGED、严格exit1保持。\n')
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file()})
 print(json.dumps(dict(output=str(o),cases=rows,unit_test_exit=0,integration_exit=0)))
if __name__=='__main__':main()
