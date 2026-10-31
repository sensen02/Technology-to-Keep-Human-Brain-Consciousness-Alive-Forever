"""Read-only LA30 interface regression, no resampling or engine calls."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
from phase_metrology import measure_phase
from test_phase_metrology import increment_path
S=Path('/home/sensen/Desktop/engine修复验收/s345-la30-slip-diagnosis/20261027T084034Z')
PREV=Path('/home/sensen/Desktop/engine修复验收/s345-winding-sampling/20261027T144059Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False)
 files=[p for root in (S,PREV) for p in root.iterdir() if p.is_file()];before={str(p):sha(p) for p in files}
 root=Path(__file__).parent;code=[root/n for n in ('phase_metrology.py','test_phase_metrology.py',Path(__file__).name)]
 dump(o/'PREREGISTRATION.json',dict(scope='interface, deterministic tests, four saved native LA30 paths only',code_hashes={str(p):sha(p) for p in code},input_hashes=before,neural_ticks=0,physics_steps=0,resampling=False,amplitude_mask=False,expected_product_counts=[1,0,3,0],expected_separate_counts=[3,1,3,5],continuous_truth='not provided to interface',historical_status='28/36 FAIL; S345 NOT_JUDGED; strict exit1 unchanged'))
 run=subprocess.run([sys.executable,'-m','unittest','-v','test_phase_metrology'],cwd=root,capture_output=True,text=True)
 (o/'UNIT_TEST.log').write_text(run.stdout+run.stderr);assert run.returncode==0
 z=np.load(S/'ORDER_COMPARISON_TRACES.npz');prior=json.loads((PREV/'RESULTS.json').read_text())['rows'];prior={r['pipeline']:r for r in prior if r['factor']==1}
 ledger=json.loads((PREV/'INDEPENDENT_VERIFICATION.json').read_text())['events'];out={};arrays={}
 for key in ('b20_o3','b20_o4','b25_o3','b25_o4'):
  p=key+'_hilbert_then_trim_';a=z[p+'ampA']*np.exp(1j*z[p+'pa']);b=z[p+'ampB']*np.exp(1j*z[p+'pb']);t=z[p+'t'];e=int(.2*float(z[key+'_fs']))
  r=measure_phase(t,a,b,endpoint_samples=e)
  # Exact archived definition, including the main implementation's second unwrap.
  np.testing.assert_allclose(r['legacy_product_rad'],z[p+'u'],atol=1e-10)
  np.testing.assert_allclose(r['legacy_separate_reunwrapped_rad'],z[p+'separate'],atol=1e-10)
  assert abs(r['product_statistics']['advance_deg']-prior[key]['advance_deg'])<1e-8
  assert abs(r['separate_statistics']['advance_deg']-prior[key]['separate_advance_deg'])<1e-8
  assert r['product_statistics']['halfturn_count']==prior[key]['halfturn_count']
  assert r['separate_statistics']['halfturn_count']==prior[key]['separate_halfturn_count']
  expected=[v for v in ledger if v['pipeline']==key and v['factor']==1]
  assert len(r['branch_change_intervals'])==len(expected)
  for actual,ex in zip(r['branch_change_intervals'],expected):
   assert abs(actual['start_s']-ex['time_start_s'])<1e-12 and abs(actual['end_s']-ex['time_end_s'])<1e-12
  oracle=increment_path(a*np.conj(b));np.testing.assert_allclose(r['product_unwrapped_rad']-r['product_unwrapped_rad'][0],oracle,atol=1e-10)
  for name,value in r.items():
   if isinstance(value,np.ndarray):arrays[key+'_'+name]=value
  out[key]={name:value for name,value in r.items() if not isinstance(value,np.ndarray)}
 np.savez_compressed(o/'INTERFACE_TRACES.npz',**arrays)
 assert {str(p):sha(p) for p in files}==before
 dump(o/'REGRESSION.json',dict(passed=True,unit_test_exit_code=run.returncode,regression_exit_code=0,source_hashes_unchanged=True,statistics_unchanged=True,all_prior_branch_intervals_reproduced=True,results=out))
 for p in code:(o/p.name).write_bytes(p.read_bytes())
 lines=['# P3/S1 双路径相位计量接口回归','','确定性单元测试7/7通过，实际exit0；LA30四原生管线保存数组回归实际exit0。无插值、新样本、神经或物理仿真。','','接口保留原复积输出及主实现二次unwrap数值路径；新增分别展开A/B差、相对初始规范对齐后的逐点整圈差、所有差异区间。精确零点分段，跨零统计None；旧数值另存并标为物理未定义。有限样本连续支路一律UNIDENTIFIED，锁相NOT_ASSESSED。传入既有低幅值标志只原样保存，不筛除。','','|管线|复积半圈|分别展开半圈|差异区间|','|---|---:|---:|---|']
 for k,r in out.items():lines.append('|%s|%d|%d|%s|'%(k,r['product_statistics']['halfturn_count'],r['separate_statistics']['halfturn_count'],str(r['branch_change_intervals'])))
 lines+=['','测试公式见 '+str(root/'test_phase_metrology.py')+'；包含远原点、近零非零、精确零、单通道混叠、相对推进混叠及初相规范。独立核算为atan2复数增量累加，不调用unwrap。生成器充分采样知识不传入接口；不可识别例要求不确定而非猜中。','没有改冻结候选/原认证入口。28/36失败、S345 NOT_JUDGED、严格exit1保持。接口完成不等于锁相或物理行走验收。','下一最小动作：在新候选版本接入该证据接口并保持历史输出回归，先明确认证适用相位真值与采样条件，不再追加R归因。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file()})
 print(json.dumps(dict(output=str(o),passed=True,unit_test_exit_code=0,regression_exit_code=0,results=out)))
if __name__=='__main__':main()
